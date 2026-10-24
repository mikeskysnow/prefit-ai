"""数据库层：PostgreSQL 连接管理与表结构初始化。

复用 config.postgres_url，通过 psycopg 连接池管理连接。
未配置 Postgres 时，所有 DB 操作降级跳过，业务层走 Mock 兜底。
"""

from sport_agent.db.connection import get_connection, get_pool, init_db
from sport_agent.db.user_state import get_preferences, mark_visit, save_preferences

__all__ = ["get_connection", "get_pool", "get_preferences", "init_db", "mark_visit", "save_preferences"]
