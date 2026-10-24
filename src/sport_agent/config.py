"""配置管理：通过环境变量加载，支持 .env 文件。

设计原则：
- 所有外部依赖（LLM、数据库、API）通过配置注入，便于 Mock 与切换环境。
- LangSmith tracing 默认开启（文档要求「从第一天起就打开 LangSmith tracing」）。
- 阶段1骨架阶段，数据库/外部API 留空时自动降级到内存/Mock 实现。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置，按环境变量名自动绑定。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ----- LLM -----
    llm_api_key: str = Field(default="sk-mock", description="LLM API Key")
    llm_base_url: str = Field(default="https://api.openai.com/v1", description="LLM API 基址")
    llm_model: str = Field(default="gpt-4o-mini", description="默认模型名")
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    llm_request_timeout: int = Field(default=30, ge=1)

    # ----- Embedding（RAG 向量化，可与 LLM 使用不同服务商）-----
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding 模型名")
    embedding_api_key: Optional[str] = Field(default=None, description="Embedding API Key，为空时复用 llm_api_key")
    embedding_base_url: Optional[str] = Field(default=None, description="Embedding API 基址，为空时复用 llm_base_url")

    # ----- LangSmith / LangChain Tracing -----
    langsmith_api_key: Optional[str] = Field(default=None)
    langsmith_project: str = Field(default="exercise-agent-dev")
    langchain_tracing_v2: bool = Field(default=True)

    # ----- 服务 -----
    agent_service_host: str = Field(default="0.0.0.0")
    agent_service_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    # ----- 数据存储（骨架阶段可空）-----
    postgres_url: Optional[str] = Field(default=None)
    redis_url: Optional[str] = Field(default=None)
    vector_db_url: Optional[str] = Field(default=None)
    # SQLite checkpointer 落盘文件（未配 Postgres 时使用；设为空禁用落盘用内存）
    sqlite_checkpointer_path: str = Field(default="checkpoints.sqlite")
    # checkpointer 类型：auto(默认) / postgres / sqlite / memory
    checkpointer_type: str = Field(default="auto")

    # ----- Mem0 长期记忆（复用 postgres_url + LLM/Embedding 配置）-----
    mem0_enabled: bool = Field(default=True, description="是否启用 Mem0 长期记忆")
    mem0_collection: str = Field(
        default="sport_agent_memories",
        description="Mem0 向量表名（共用库需避开他服务表名）",
    )
    mem0_embedding_dims: int = Field(default=1536, description="Embedding 维度")
    # 记忆保留策略（天），physio 短期衰减、sport_history 永久、pre_sport 中期
    mem0_physio_retention_days: int = Field(default=90)
    mem0_pre_sport_retention_days: int = Field(default=180)
    # 异步抽取写入的线程池大小（add() 实测 ~11s，必须离开关键链路）
    mem0_writer_workers: int = Field(default=2, ge=1, le=8)

    # ----- 外部业务 API（骨架阶段可空，走 Mock）-----
    health_data_api_base: Optional[str] = Field(default=None)
    weather_api_base: Optional[str] = Field(default=None)
    weather_api_key: Optional[str] = Field(default=None)
    # APP 后端运动数据接口基址（下载运动数据接口，不鉴权）
    app_api_base: Optional[str] = Field(default=None)
    # APP 后端接口请求超时（秒），短于 LLM 超时避免数据同步拖慢主流程
    app_api_timeout: int = Field(default=10, ge=1, description="APP 后端 API 请求超时秒数")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def use_mock_tools(self) -> bool:
        """未配置外部业务 API 时，工具层自动走 Mock 数据。"""
        return not self.health_data_api_base

    @computed_field  # type: ignore[prop-decorator]
    @property
    def use_sqlite_checkpointer(self) -> bool:
        """未配置 Postgres 时，使用本地 SQLite checkpointer 落盘。"""
        return not self.postgres_url

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_checkpointer(self) -> str:
        """实际生效的 checkpointer 类型（auto 自动推断）。"""
        if self.checkpointer_type != "auto":
            return self.checkpointer_type
        return "postgres" if self.postgres_url else "sqlite"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def use_llm(self) -> bool:
        """是否启用真实 LLM 调用。

        api_key 为空或占位值(sk-mock)时降级到规则/Mock，保证骨架无 Key 也可跑通。
        配置真实 Key 后，F007/F008 等节点自动切换到 LLM 生成。
        """
        return bool(self.llm_api_key) and self.llm_api_key not in ("", "sk-mock")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_embedding_api_key(self) -> str:
        """Embedding API Key，未单独配置时复用 LLM Key。"""
        return self.embedding_api_key or self.llm_api_key

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_embedding_base_url(self) -> str:
        """Embedding API 基址，未单独配置时复用 LLM 基址。"""
        return self.embedding_base_url or self.llm_base_url

    @computed_field  # type: ignore[prop-decorator]
    @property
    def use_mem0(self) -> bool:
        """Mem0 是否真正可用。

        需要同时满足：开关打开 + 有 Postgres（pgvector 后端）+ 有真实 LLM/Embedding
        （记忆抽取与向量化都依赖）。任一缺失则记忆层整体降级为空，业务不受影响。
        """
        return bool(self.mem0_enabled and self.postgres_url and self.use_llm)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """单例获取配置。"""
    return Settings()  # type: ignore[call-arg]


def apply_langsmith_env() -> None:
    """将 LangSmith 配置写入环境变量，供 LangChain/LangGraph 自动埋点。

    LangSmith 通过读取环境变量自动接入，无需侵入业务代码。
    """
    import os

    s = get_settings()
    if s.langchain_tracing_v2 and s.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = s.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = s.langsmith_project
        # LANGCHAIN_ENDPOINT 一般为默认值，显式设置以便兼容自托管场景
        os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    else:
        # 显式关闭，避免未配置 key 时报错
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
