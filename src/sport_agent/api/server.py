"""FastAPI 服务入口：暴露运动前阶段智能体 HTTP 接口。

端点：
- GET  /             : Web 演示页面（运动前流程可视化）
- POST /agent/start  : 启动运动前流程（同步执行，遇 interrupt 返回问询）
- POST /agent/resume : 恢复 interrupt（用户回答问询后继续）
- POST /agent/stream : SSE 流式执行（stream_mode=updates，逐节点输出）
- POST /agent/report/upload : 上传体检报告 PDF 或运动记录图片，返回 report_path 供 start/stream 使用
- GET  /health       : 健康检查
- GET  /config       : 配置信息（LLM/checkpointer 状态）
- GET  /state/{tid}  : 查询某会话当前状态（调试用）
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from sport_agent.config import apply_langsmith_env, get_settings
from sport_agent.graph import get_app
from sport_agent.utils import get_logger

logger = get_logger(__name__)

# 上传件目录与大小上限（20MB，实测样例报告 28 页约 11MB，留足余量）
REPORT_DIR = Path(tempfile.gettempdir()) / "sport_agent_reports"
MAX_REPORT_BYTES = 20 * 1024 * 1024

# 允许的上传格式：pdf 体检报告 + 运动记录截图
ALLOWED_UPLOAD_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}


# ===== 请求 / 响应模型 =====


class AgentStartRequest(BaseModel):
    user_id: str
    session_id: str | None = None
    # None = 由 user_state 表自动判定新老用户（推荐）；显式传值用于测试/演示强制指定
    is_first_time: bool | None = None
    user_message: str | None = None
    location: dict[str, Any] | None = None
    goal: str | None = None
    # 上传件的服务端路径（由 /agent/report/upload 返回），体检报告 pdf 或运动记录图片；
    # 为空则跳过解析节点
    report_path: str | None = None
    # 自定义健康数据：用户提供时覆盖自动采集的 vitals 和运动能力统计
    # {"vitals": {"heart_rate", "spo2", "hrv", "stress", "resting_hr"},
    #  "exercise_capacity": {"avg_duration_7d", "avg_calories", "common_sport_type", "avg_distance_km", "avg_pace"}}
    custom_vitals: dict[str, Any] | None = None


class AgentResumeRequest(BaseModel):
    thread_id: str
    value: str  # 用户对 interrupt 的回答


class AgentResponse(BaseModel):
    thread_id: str
    completed: bool
    interrupted: bool
    interrupt_question: Optional[str] = None
    stage: Optional[str] = None
    state: dict[str, Any] = Field(default_factory=dict)


# ===== 辅助函数 =====


def _config(thread_id: str) -> dict[str, Any]:
    # 多轮对话每轮 ~2 步（plan_review 节点 + 条件路由），50 轮需 100 步，
    # 加上主流程 ~20 步，留余量到 200
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}


def _initial_state(req: AgentStartRequest) -> dict[str, Any]:
    state: dict[str, Any] = {
        "user_id": req.user_id,
        "session_id": req.session_id or str(uuid4()),
    }
    # 不传时留空，交给 health_collect 查 user_state 表判定（新老用户持久化）
    if req.is_first_time is not None:
        state["is_first_time"] = req.is_first_time
    if req.user_message:
        state["user_message"] = req.user_message
        state["messages"] = [HumanMessage(content=req.user_message)]
    if req.location:
        state["location"] = req.location
    if req.goal:
        state["goal"] = req.goal
    if req.report_path:
        state["report_path"] = req.report_path
    if req.custom_vitals:
        state["custom_vitals"] = req.custom_vitals
    return state


def _extract_interrupt(app: Any, config: dict[str, Any]) -> Optional[dict[str, Any]]:
    """从当前图状态中提取 interrupt 信息（如有）。"""
    try:
        snapshot = app.get_state(config)
    except Exception:  # noqa: BLE001
        return None
    if not snapshot:
        return None
    # 注意：不能用 snapshot.next 作为前置判断。同一节点内多次 interrupt 时
    # （如 interview 顺序问 goal/available_time/intensity_pref），resume 后
    # 命中第二个 interrupt 会出现 next=() 但 tasks[0].interrupts 非空的状态，
    # 若先判 next 会把待回答的问询误报为流程结束。
    for task in getattr(snapshot, "tasks", []) or []:
        interrupts = getattr(task, "interrupts", []) or []
        if interrupts:
            intr = interrupts[0]
            value = getattr(intr, "value", None) or {}
            if not isinstance(value, dict):
                return {"question": str(value)}
            # 原样透传 field / attempt / tts_cancel，供客户端渲染快捷回复与停止播报
            return {**value, "question": value.get("question")}
    if snapshot.next:
        # 有 next 但无显式 interrupt 对象，仍视为暂停
        return {"question": None, "next": list(snapshot.next)}
    return None


def _serialize_state(app: Any, config: dict[str, Any]) -> dict[str, Any]:
    """序列化当前 State（过滤不可 JSON 化的 messages 对象）。"""
    try:
        snapshot = app.get_state(config)
        values = dict(snapshot.values) if snapshot else {}
    except Exception:  # noqa: BLE001
        return {}
    safe: dict[str, Any] = {}
    for k, v in values.items():
        if k == "messages":
            # messages 是 BaseMessage 列表，序列化为文本
            safe[k] = [m.content if hasattr(m, "content") else str(m) for m in v]
        else:
            try:
                json.dumps(v)
                safe[k] = v
            except (TypeError, ValueError):
                safe[k] = str(v)
    return safe


def _run(app: Any, config: dict[str, Any], payload: Any) -> AgentResponse:
    """执行图，处理 interrupt 与完成两种情况。"""
    try:
        app.invoke(payload, config=config)
    except Exception as e:  # noqa: BLE001
        logger.error("agent.run.failed", error=str(e))
        raise

    intr = _extract_interrupt(app, config)
    state = _serialize_state(app, config)

    if intr:
        return AgentResponse(
            thread_id=config["configurable"]["thread_id"],
            completed=False,
            interrupted=True,
            interrupt_question=intr.get("question"),
            stage=state.get("stage"),
            state=state,
        )
    return AgentResponse(
        thread_id=config["configurable"]["thread_id"],
        completed=True,
        interrupted=False,
        stage=state.get("stage", "end"),
        state=state,
    )


# ===== FastAPI 应用 =====


def create_app() -> FastAPI:
    apply_langsmith_env()

    # 初始化数据库表（training_history / sync_state），无 Postgres 时静默跳过
    from sport_agent.db import init_db

    init_db()

    app_graph = get_app()

    api = FastAPI(title="AI Sport Coach Agent", version="0.1.0")

    # Web 演示页面
    web_dir = Path(__file__).resolve().parent.parent / "web"

    @api.get("/")
    def index() -> FileResponse:
        return FileResponse(web_dir / "index.html", media_type="text/html")

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/config")
    def config_info() -> dict[str, Any]:
        """返回当前配置状态（供前端展示 LLM/checkpointer 状态）。"""
        s = get_settings()
        return {
            "use_llm": s.use_llm,
            "llm_model": s.llm_model,
            "checkpointer": s.effective_checkpointer,
            "langsmith_tracing": s.langchain_tracing_v2 and bool(s.langsmith_api_key),
            "app_api_base": s.app_api_base or "",
            "postgres": bool(s.postgres_url),
        }

    @api.post("/agent/report/upload")
    async def upload_report(file: UploadFile = File(...)) -> dict[str, Any]:
        """接收体检报告 PDF 或运动记录图片，存到服务端后返回 report_path。

        返回的 report_path 传给 /agent/start 或 /agent/stream 即触发解析节点，
        具体是体检报告还是运动记录由节点内的 LLM 识别，这里只校验容器格式。
        """
        name = file.filename or ""
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_UPLOAD_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"仅支持 {'/'.join(sorted(ALLOWED_UPLOAD_SUFFIXES))}，"
                "体检报告用 pdf，运动记录用截图",
            )

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="上传的文件为空")
        if len(content) > MAX_REPORT_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"文件不能超过 {MAX_REPORT_BYTES // (1024 * 1024)}MB",
            )

        # 用 uuid 作文件名，避免用户文件名里的路径穿越与同名覆盖；后缀保留供节点判型
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        dest = REPORT_DIR / f"{uuid4().hex}{suffix}"
        dest.write_bytes(content)

        logger.info("report.uploaded", filename=name, path=str(dest), bytes=len(content))
        return {
            "report_path": str(dest),
            "filename": name,
            "size": len(content),
            "container": "pdf" if suffix == ".pdf" else "image",
        }

    @api.post("/agent/start", response_model=AgentResponse)
    def start(req: AgentStartRequest) -> AgentResponse:
        thread_id = req.session_id or str(uuid4())
        config = _config(thread_id)
        logger.info("agent.start", user_id=req.user_id, thread_id=thread_id)
        return _run(app_graph, config, _initial_state(req))

    @api.post("/agent/resume", response_model=AgentResponse)
    def resume(req: AgentResumeRequest) -> AgentResponse:
        config = _config(req.thread_id)
        logger.info("agent.resume", thread_id=req.thread_id, value=req.value)
        return _run(app_graph, config, Command(resume=req.value))

    @api.post("/agent/stream")
    def stream(req: AgentStartRequest) -> StreamingResponse:
        """SSE 流式执行：逐节点状态更新 + F008 token 级流式。

        使用同步 app.stream + stream_mode=["updates","custom"]：
        - updates: 每个节点完成后输出状态变更
        - custom: plan_analyze 节点通过 get_stream_writer 发射的 token 流

        注：用同步 stream 而非 astream，以兼容 SqliteSaver/PostgresSaver（sync saver）。
        FastAPI 会把同步 generator 放入线程池执行，不阻塞事件循环。
        """
        thread_id = req.session_id or str(uuid4())
        config = _config(thread_id)
        payload = _initial_state(req)

        def _safe_json(v: Any) -> Any:
            if isinstance(v, list) and v and hasattr(v[0], "content"):
                return [m.content if hasattr(m, "content") else str(m) for m in v]
            try:
                json.dumps(v)
                return v
            except (TypeError, ValueError):
                return str(v)

        def event_generator():
            for mode, chunk in app_graph.stream(
                payload, config=config, stream_mode=["updates", "custom"]
            ):
                if mode == "updates":
                    for node_name, updates in chunk.items():
                        # interrupt 时 updates 可能是 tuple，跳过非 dict 的更新
                        if not isinstance(updates, dict):
                            continue
                        safe = {k: _safe_json(v) for k, v in (updates or {}).items()}
                        yield f"data: {json.dumps({'node': node_name, 'updates': safe}, ensure_ascii=False)}\n\n"
                elif mode == "custom":
                    # plan_analyze token 流：chunk 是 writer 发射的 dict
                    yield f"data: {json.dumps({'stream': chunk}, ensure_ascii=False)}\n\n"

            intr = _extract_interrupt(app_graph, config)
            if intr:
                yield f"data: {json.dumps({'interrupt': intr, 'thread_id': thread_id}, ensure_ascii=False)}\n\n"
            else:
                yield f"data: {json.dumps({'done': True, 'thread_id': thread_id}, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @api.post("/agent/resume/stream")
    def resume_stream(req: AgentResumeRequest) -> StreamingResponse:
        """恢复 interrupt 后的 SSE 流式执行（resume 继续到结束，含 token 流）。"""
        config = _config(req.thread_id)

        def _safe_json(v: Any) -> Any:
            if isinstance(v, list) and v and hasattr(v[0], "content"):
                return [m.content if hasattr(m, "content") else str(m) for m in v]
            try:
                json.dumps(v)
                return v
            except (TypeError, ValueError):
                return str(v)

        def event_generator():
            for mode, chunk in app_graph.stream(
                Command(resume=req.value), config=config, stream_mode=["updates", "custom"]
            ):
                if mode == "updates":
                    for node_name, updates in chunk.items():
                        if not isinstance(updates, dict):
                            continue
                        safe = {k: _safe_json(v) for k, v in (updates or {}).items()}
                        yield f"data: {json.dumps({'node': node_name, 'updates': safe}, ensure_ascii=False)}\n\n"
                elif mode == "custom":
                    yield f"data: {json.dumps({'stream': chunk}, ensure_ascii=False)}\n\n"

            intr = _extract_interrupt(app_graph, config)
            if intr:
                yield f"data: {json.dumps({'interrupt': intr, 'thread_id': req.thread_id}, ensure_ascii=False)}\n\n"
            else:
                yield f"data: {json.dumps({'done': True, 'thread_id': req.thread_id}, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @api.get("/state/{thread_id}")
    def get_state(thread_id: str) -> dict[str, Any]:
        config = _config(thread_id)
        return _serialize_state(app_graph, config)

    return api


# 模块级实例（uvicorn 直接引用）
app = create_app()
