"""CLI 入口：启动 FastAPI 服务，或直接在命令行跑通一次运动前流程。

用法：
    # 启动 API 服务（默认 0.0.0.0:8000）
    sport-agent serve

    # 命令行跑通一次流程（Mock 数据，验证骨架）
    sport-agent run --user-id 10001

    # 指定端口
    sport-agent serve --port 9000
"""

from __future__ import annotations

import argparse
import json
import sys
from uuid import uuid4

from sport_agent.config import apply_langsmith_env, get_settings
from sport_agent.utils import get_logger

logger = get_logger(__name__)


def cmd_serve(host: str | None, port: int | None) -> None:
    """启动 uvicorn 服务。"""
    import uvicorn

    s = get_settings()
    apply_langsmith_env()
    uvicorn.run(
        "sport_agent.api.server:app",
        host=host or s.agent_service_host,
        port=port or s.agent_service_port,
        reload=False,
    )


def cmd_run(user_id: str, first_time: bool, message: str | None) -> None:
    """命令行跑通一次运动前流程（含 interrupt 恢复演示）。"""
    from langgraph.types import Command

    from sport_agent.graph import get_app

    apply_langsmith_env()
    app = get_app()
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

    initial = {
        "user_id": user_id,
        "session_id": thread_id,
        "is_first_time": first_time,
        "available_time": 40,
    }
    if message:
        initial["user_message"] = message

    logger.info("cli.run.start", user_id=user_id, thread_id=thread_id)
    app.invoke(initial, config=config)

    # 检查是否停在 interview interrupt
    snapshot = app.get_state(config)
    state = dict(snapshot.values) if snapshot else {}

    # 若 interview 未提供 goal 且触发 interrupt，用默认回答恢复
    if not state.get("inquiry_completed") and not state.get("goal"):
        answer = message or "减脂"
        logger.info("cli.run.resume", answer=answer)
        app.invoke(Command(resume=answer), config=config)
        snapshot = app.get_state(config)
        state = dict(snapshot.values) if snapshot else {}

    # 打印关键结果
    _print_summary(state)


def _print_summary(state: dict) -> None:
    print("\n" + "=" * 60)
    print("运动前阶段流程结果")
    print("=" * 60)
    fields = [
        ("stage", "最终阶段"),
        ("risk_level", "风险等级"),
        ("body_energy", "身体电量"),
        ("recovery_status", "恢复状态"),
        ("goal", "运动目标"),
        ("plan", "运动计划"),
        ("device_ready", "设备就绪"),
        ("warmup_guide", "热身指导"),
    ]
    for key, label in fields:
        val = state.get(key)
        if val is None:
            continue
        if isinstance(val, (dict, list)):
            print(f"{label}: {json.dumps(val, ensure_ascii=False, indent=2)}")
        else:
            print(f"{label}: {val}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="sport-agent", description="AI Sport Coach Agent CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="启动 API 服务")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)

    p_run = sub.add_parser("run", help="命令行跑通一次流程")
    p_run.add_argument("--user-id", default="10001")
    p_run.add_argument("--first-time", action="store_true", default=True)
    p_run.add_argument("--message", default=None, help="初始用户消息（如：今天想减脂）")

    args = parser.parse_args()
    if args.cmd == "serve":
        cmd_serve(args.host, args.port)
    elif args.cmd == "run":
        cmd_run(args.user_id, args.first_time, args.message)


if __name__ == "__main__":
    main()
