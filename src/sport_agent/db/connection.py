"""PostgreSQL 连接池管理。

设计要点：
- 进程级单例连接池，复用连接避免反复握手。
- psycopg3 原生支持 autocommit + 连接池。
- 未配置 postgres_url 时返回 None，调用方负责降级。
"""

from __future__ import annotations

from typing import Any, Optional

from sport_agent.config import get_settings
from sport_agent.utils import get_logger

logger = get_logger(__name__)

_pool: Optional[Any] = None


def get_pool() -> Optional[Any]:
    """获取进程级连接池单例。

    Returns:
        psycopg_pool.ConnectionPool 或 None（未配置 Postgres）。
    """
    global _pool
    if _pool is not None:
        return _pool

    s = get_settings()
    if not s.postgres_url:
        return None

    try:
        from psycopg_pool import ConnectionPool

        _pool = ConnectionPool(
            s.postgres_url,
            max_size=10,
            open=True,
            kwargs={"autocommit": True},
        )
        logger.info("db.pool.ready")
        return _pool
    except ImportError:
        # psycopg_pool 未安装，降级为单连接模式
        logger.warning("db.psycopg_pool_missing_fallback_single")
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("db.pool_failed", error=str(e))
        return None


class _DummyConnection:
    """无 DB 时的占位上下文管理器，避免调用方写 None 判断。"""

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("PostgreSQL not configured")


class _DummyPool:
    """无连接池时的占位，get_connection 返回此对象的 connection() 会抛异常。"""

    def connection(self) -> _DummyConnection:
        return _DummyConnection()


def get_connection() -> Any:
    """获取连接上下文管理器（从连接池取或直接 connect）。

    用法：
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")

    无 Postgres 时返回 _DummyConnection，cursor() 会抛异常，
    调用方需 try/except 降级。
    """
    pool = get_pool()
    if pool is not None:
        return pool.connection()

    # 降级：尝试单连接
    s = get_settings()
    if not s.postgres_url:
        return _DummyPool().connection()

    try:
        import psycopg

        return psycopg.connect(s.postgres_url, autocommit=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("db.connect_failed", error=str(e))
        return _DummyPool().connection()


def init_db() -> None:
    """初始化数据库表结构（幂等）。

    在服务启动时调用，创建：
    - training_history / sync_state（运动数据摘要）
    - user_profile / health_daily（健康数据摘要）
    无 Postgres 时静默跳过。
    """
    from sport_agent.db.health_metrics import create_tables as create_health_tables
    from sport_agent.db.training_history import create_tables as create_training_tables
    from sport_agent.db.user_state import create_tables as create_user_state_tables

    try:
        create_training_tables()
    except Exception as e:  # noqa: BLE001
        logger.warning("db.init_training_failed", error=str(e))

    try:
        create_health_tables()
    except Exception as e:  # noqa: BLE001
        logger.warning("db.init_health_failed", error=str(e))

    try:
        create_user_state_tables()
    except Exception as e:  # noqa: BLE001
        logger.warning("db.init_user_state_failed", error=str(e))
