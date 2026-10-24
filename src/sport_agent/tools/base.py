"""工具层公共基类：Mock 数据生成与外部 HTTP 客户端。

设计原则（来自《整体架构设计与落地建议》）：
- 风险规则用纯 Python 代码判断，不依赖 LLM（误报率≤3% 的硬指标无法靠 LLM 达成）。
- 工具层只负责「取数」，返回结构化 dict，决策由节点完成。
- 骨架阶段未配置外部 API 时，走稳定的 Mock 数据，保证流程可跑通。
"""

from __future__ import annotations

import httpx
from langchain_core.tools import tool

from sport_agent.config import get_settings


def _is_mock() -> bool:
    return get_settings().use_mock_tools


def http_client() -> httpx.Client:
    """构造带超时的同步 HTTP 客户端（阶段2接入真实业务 API 时使用）。"""
    s = get_settings()
    return httpx.Client(timeout=s.llm_request_timeout)


def async_http_client() -> httpx.AsyncClient:
    s = get_settings()
    return httpx.AsyncClient(timeout=s.llm_request_timeout)


# 统一的 tool 装饰器别名，便于后续追加通用参数（如 return_direct、tags）
sport_tool = tool
