"""LLM 工厂：统一构造 LangChain ChatModel 实例。

设计要点：
- 通过 config 注入 API Key / Base URL / Model，便于切换 OpenAI/DeepSeek/Qwen 等兼容服务。
- 单例缓存，避免节点内重复初始化。
- LangSmith tracing 通过环境变量自动接入（见 config.apply_langsmith_env），此处无需额外代码。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from sport_agent.config import get_settings


@lru_cache(maxsize=1)
def get_llm(
    *,
    model: str | None = None,
    temperature: float | None = None,
    streaming: bool = True,
    timeout: int | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """获取默认 LLM 实例（单例）。

    Args:
        model: 模型名，缺省取配置默认值
        temperature: 温度，缺省取配置默认值
        streaming: 是否启用流式输出（运动方案分析 F008 要求流式呈现，默认开启）
        timeout: 单次请求超时（秒），缺省取配置默认值。长上下文任务（如体检报告
            整篇抽取）需要显式放宽，默认 30s 不够
        **kwargs: 透传给 ChatOpenAI 的其它参数

    Returns:
        BaseChatModel: LangChain 聊天模型实例
    """
    s = get_settings()
    return ChatOpenAI(
        api_key=s.llm_api_key,
        base_url=s.llm_base_url,
        model=model or s.llm_model,
        temperature=temperature if temperature is not None else s.llm_temperature,
        timeout=timeout if timeout is not None else s.llm_request_timeout,
        streaming=streaming,
        **kwargs,
    )
