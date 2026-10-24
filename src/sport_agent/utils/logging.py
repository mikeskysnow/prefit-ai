"""结构化日志：基于 structlog，统一全链路日志格式，便于与 LangSmith trace 对照。"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from sport_agent.config import get_settings

_CONFIGURED = False


def configure_logging() -> None:
    """初始化日志配置（进程级，幂等）。"""
    global _CONFIGURED
    if _CONFIGURED:
        return

    s = get_settings()
    level = getattr(logging, s.log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None, **initial_values: Any) -> structlog.stdlib.BoundLogger:
    """获取一个绑定上下文的 logger。"""
    configure_logging()
    return structlog.get_logger(name).bind(**initial_values)  # type: ignore[return-value]
