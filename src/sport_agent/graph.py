"""LangGraph 图组装与编译。

流程（运动前阶段）：
    START
      ↓
    memory_load（Mem0 记忆快照，~2ms，供后续各评估节点复用）
      ↓ ← route_after_memory_load 条件路由
    ├─ 上传了体检报告 → report_analyze（解析 PDF + 更新画像）┐
    └─ 未上传 ─────────────────────────────────────────────┤
                                                          ↓
    health_collect (F001)
      ↓
    risk_assess (F002)
      ↓ ← route_after_risk 条件路由
    ├─ high/forbidden → safety_stop → memory_write → END
    └─ low/mid
          ↓
        feasibility (F003) → recovery (F004) → avatar (F005)
          ↓ ← route_after_avatar 条件路由（新老用户分流）
        ├─ 新用户 → interview (F006, interrupt) ┐
        └─ 老用户 ────────────────────────────┤
                                              ↓
        plan_generate (F007) → plan_analyze (F008) → plan_compose (F009) → plan_edit (F010)
          ↓
        device_check (F011) → warmup (F012)
          ↓
        memory_write（Mem0 记忆沉淀，全后台，不阻塞）
          ↓
        END

checkpointer：
- 默认 InMemorySaver（开发可跑通，支持 interrupt resume，重启丢失）。
- 配置 Postgres 后自动切换 PostgresSaver（落盘，支持 7 天基线/长期记忆）。
- interrupt 必须配合 checkpointer + thread_id 才能恢复。
"""

from __future__ import annotations

from typing import Any, Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from sport_agent.nodes import (
    avatar_node,
    device_check_node,
    feasibility_node,
    health_collect_node,
    interview_node,
    memory_load_node,
    memory_write_node,
    plan_analyze_node,
    plan_compose_node,
    plan_edit_node,
    plan_generate_node,
    plan_review_node,
    recovery_node,
    report_analyze_node,
    risk_assess_node,
    route_after_avatar,
    route_after_memory_load,
    route_after_plan_review,
    route_after_risk,
    safety_stop_node,
    warmup_node,
)
from sport_agent.state import SportAgentState
from sport_agent.utils import get_logger

logger = get_logger(__name__)

# 节点名常量（与 State.stage 对应，便于路由与 trace）
N_MEM_LOAD = "memory_load"
N_MEM_WRITE = "memory_write"
N_REPORT = "report_analyze"
N_HEALTH = "health_collect"
N_RISK = "risk_assess"
N_SAFETY = "safety_stop"
N_FEASIBILITY = "feasibility"
N_RECOVERY = "recovery"
N_AVATAR = "avatar"
N_INTERVIEW = "interview"
N_PLAN_GEN = "plan_generate"
N_PLAN_ANALYZE = "plan_analyze"
N_PLAN_REVIEW = "plan_review"
N_PLAN_COMPOSE = "plan_compose"
N_PLAN_EDIT = "plan_edit"
N_DEVICE = "device_check"
N_WARMUP = "warmup"


def _build_graph() -> StateGraph:
    """构造运动前阶段状态图（未编译）。"""
    graph: StateGraph = StateGraph(SportAgentState)

    # 注册节点
    graph.add_node(N_MEM_LOAD, memory_load_node)
    graph.add_node(N_MEM_WRITE, memory_write_node)
    graph.add_node(N_REPORT, report_analyze_node)
    graph.add_node(N_HEALTH, health_collect_node)
    graph.add_node(N_RISK, risk_assess_node)
    graph.add_node(N_SAFETY, safety_stop_node)
    graph.add_node(N_FEASIBILITY, feasibility_node)
    graph.add_node(N_RECOVERY, recovery_node)
    graph.add_node(N_AVATAR, avatar_node)
    graph.add_node(N_INTERVIEW, interview_node)
    graph.add_node(N_PLAN_GEN, plan_generate_node)
    graph.add_node(N_PLAN_ANALYZE, plan_analyze_node)
    graph.add_node(N_PLAN_REVIEW, plan_review_node)
    graph.add_node(N_PLAN_COMPOSE, plan_compose_node)
    graph.add_node(N_PLAN_EDIT, plan_edit_node)
    graph.add_node(N_DEVICE, device_check_node)
    graph.add_node(N_WARMUP, warmup_node)

    # 入口：先读长期记忆快照（~2ms），供后续所有评估节点复用
    graph.add_edge(START, N_MEM_LOAD)
    # 上传了体检报告则先解析并更新画像，再进健康采集；未上传直接跳过（零开销）
    graph.add_conditional_edges(
        N_MEM_LOAD,
        route_after_memory_load,
        {
            "report_analyze": N_REPORT,
            "health_collect": N_HEALTH,
        },
    )
    graph.add_edge(N_REPORT, N_HEALTH)

    # 健康采集 → 风险评估
    graph.add_edge(N_HEALTH, N_RISK)

    # 风险评估后条件路由
    graph.add_conditional_edges(
        N_RISK,
        route_after_risk,
        {
            "safety_stop": N_SAFETY,
            "feasibility": N_FEASIBILITY,
        },
    )

    # 高风险阻断 → 也要沉淀记忆（高风险结论是重要历史档案）→ 终止
    graph.add_edge(N_SAFETY, N_MEM_WRITE)

    # 正常主流程
    graph.add_edge(N_FEASIBILITY, N_RECOVERY)
    graph.add_edge(N_RECOVERY, N_AVATAR)
    # 新老用户分流：新用户走问询收集目标/时长/强度；老用户复用长期偏好直接出方案
    graph.add_conditional_edges(
        N_AVATAR,
        route_after_avatar,
        {
            "interview": N_INTERVIEW,
            "plan_generate": N_PLAN_GEN,
        },
    )
    graph.add_edge(N_INTERVIEW, N_PLAN_GEN)
    graph.add_edge(N_PLAN_GEN, N_PLAN_ANALYZE)
    graph.add_edge(N_PLAN_ANALYZE, N_PLAN_REVIEW)
    # 多轮对话：用户还有疑问就绕回问询节点，最多 MAX_DIALOG_TURNS 轮
    graph.add_conditional_edges(
        N_PLAN_REVIEW,
        route_after_plan_review,
        {
            "plan_review": N_PLAN_REVIEW,
            "plan_compose": N_PLAN_COMPOSE,
        },
    )
    graph.add_edge(N_PLAN_COMPOSE, N_PLAN_EDIT)
    graph.add_edge(N_PLAN_EDIT, N_DEVICE)
    graph.add_edge(N_DEVICE, N_WARMUP)
    # 出口：沉淀本次会话记忆（只提交后台任务，不阻塞）
    graph.add_edge(N_WARMUP, N_MEM_WRITE)
    graph.add_edge(N_MEM_WRITE, END)

    return graph


def _make_postgres_checkpointer(url: str) -> Any:
    """构造 PostgresSaver（生产落盘，支撑 7 天基线校准与长期记忆）。

    优先用连接池（psycopg_pool），未安装则降级单连接。
    """
    from langgraph.checkpoint.postgres import PostgresSaver

    try:
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(url, max_size=20, open=True, kwargs={"autocommit": True})
        cp = PostgresSaver(conn=pool)
    except ImportError:
        import psycopg  # noqa: PLC0415

        conn = psycopg.connect(url, autocommit=True)
        cp = PostgresSaver(conn=conn)
    cp.setup()  # 建表（checkpoints/writes）
    return cp


def _make_sqlite_checkpointer(path: str) -> Any:
    """构造 SqliteSaver（本地落盘，重启不丢，支持 interrupt 跨进程恢复）。"""
    import sqlite3  # noqa: PLC0415

    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect(path, check_same_thread=False)
    cp = SqliteSaver(conn=conn)
    cp.setup()
    return cp


# 进程级 checkpointer 单例（复用连接，避免反复建表/连接）
_checkpointer_singleton: Optional[Any] = None


def _get_checkpointer() -> Any:
    """根据配置选择 checkpointer（auto/postgres/sqlite/memory）。

    优先级（auto 模式）：
    1. 配置了 postgres_url → PostgresSaver（生产，连接池）
    2. 否则 → SqliteSaver（本地落盘 checkpoints.sqlite）
    3. 任一失败 → InMemorySaver（兜底）

    interrupt 必须配合 checkpointer + thread_id 才能恢复；落盘后支持
    7 天基线校准、长期用户记忆与跨会话状态恢复。
    """
    global _checkpointer_singleton
    if _checkpointer_singleton is not None:
        return _checkpointer_singleton

    from sport_agent.config import get_settings

    s = get_settings()
    kind = s.effective_checkpointer

    if kind == "postgres" and s.postgres_url:
        try:
            _checkpointer_singleton = _make_postgres_checkpointer(s.postgres_url)
            logger.info("checkpointer.postgres")
            return _checkpointer_singleton
        except Exception as e:  # noqa: BLE001
            logger.warning("checkpointer.postgres_failed_fallback_sqlite", error=str(e))

    if kind in ("postgres", "sqlite", "auto") and s.sqlite_checkpointer_path:
        try:
            _checkpointer_singleton = _make_sqlite_checkpointer(s.sqlite_checkpointer_path)
            logger.info("checkpointer.sqlite", path=s.sqlite_checkpointer_path)
            return _checkpointer_singleton
        except Exception as e:  # noqa: BLE001
            logger.warning("checkpointer.sqlite_failed_fallback_memory", error=str(e))

    logger.info("checkpointer.memory")
    _checkpointer_singleton = InMemorySaver()
    return _checkpointer_singleton


def build_app(checkpointer: Optional[Any] = None) -> Any:
    """编译并返回可执行的 LangGraph 应用。

    Args:
        checkpointer: 自定义 checkpointer；缺省按配置自动选择。

    Returns:
        编译后的 CompiledGraph，调用 .invoke / .stream / .astream。
    """
    graph = _build_graph()
    cp = checkpointer or _get_checkpointer()
    app = graph.compile(checkpointer=cp)
    logger.info("graph.compiled", checkpointer=type(cp).__name__)
    return app


# 进程级单例（API/CLI 复用）
_app_singleton: Optional[Any] = None


def get_app() -> Any:
    """获取编译后的图单例。"""
    global _app_singleton
    if _app_singleton is None:
        _app_singleton = build_app()
    return _app_singleton
