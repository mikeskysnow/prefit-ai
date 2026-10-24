"""用户状态表：区分新老用户、存储长期偏好。

需求：
- 首次使用 = 新用户 → 走完整问询（目标/时长/强度）
- 第二次及以后 = 老用户 → 跳过问询，直接生成方案
- 长期记忆：保存用户历史偏好（goal/available_time/intensity_pref），
  老用户下次启动时直接复用，无需重复回答。

设计要点：
- user_state 表以 user_id 为主键，记录访问次数与时间戳。
- visit_count=0 表示首次，≥1 表示老用户。
- 原子操作：mark_visit 用 UPSERT + RETURNING 一次往返拿到 visit_count。
- 偏好字段存 JSONB，灵活扩展（goal/available_time/intensity_pref/其他）。
"""

from __future__ import annotations

from typing import Any, Optional

from sport_agent.db.connection import get_connection
from sport_agent.utils import get_logger

logger = get_logger(__name__)

# ===== 建表 SQL =====

_CREATE_USER_STATE = """
CREATE TABLE IF NOT EXISTS user_state (
    user_id          VARCHAR(64) PRIMARY KEY,
    visit_count      INT NOT NULL DEFAULT 0,
    first_visit_at   TIMESTAMPTZ DEFAULT NOW(),
    last_visit_at    TIMESTAMPTZ DEFAULT NOW(),
    
    -- 长期偏好（JSONB 灵活扩展）
    -- 示例: {"goal": "fat_loss", "available_time": 30, "intensity_pref": "mid"}
    preferences      JSONB DEFAULT '{}'::jsonb,
    
    synced_at        TIMESTAMPTZ DEFAULT NOW()
);
"""

_CREATE_INDEX_VISIT = """
CREATE INDEX IF NOT EXISTS idx_user_state_visit
ON user_state (user_id, visit_count);
"""


def create_tables() -> None:
    """幂等建表：user_state + 索引。"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_CREATE_USER_STATE)
            cur.execute(_CREATE_INDEX_VISIT)
    logger.info("db.user_state_tables.ready")


# ===== 原子访问记录 =====

_MARK_VISIT_SQL = """
INSERT INTO user_state (user_id, visit_count, first_visit_at, last_visit_at)
VALUES (%(user_id)s, 1, NOW(), NOW())
ON CONFLICT (user_id) DO UPDATE SET
    visit_count   = user_state.visit_count + 1,
    last_visit_at = NOW(),
    synced_at     = NOW()
RETURNING visit_count;
"""


def mark_visit(user_id: str) -> int:
    """原子记录一次访问，返回本次访问后的总访问次数。
    
    首次调用：插入新行，visit_count=1 → 返回 1（新用户）
    后续调用：递增 visit_count → 返回 ≥2（老用户）
    
    Returns:
        本次访问后的累计次数（1=首次，≥2=老用户）
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_MARK_VISIT_SQL, {"user_id": user_id})
            row = cur.fetchone()
            count = row[0] if row else 1
            logger.info("user_state.mark_visit", user_id=user_id, visit_count=count)
            return count


def is_first_time(user_id: str) -> bool:
    """查询用户是否首次使用（visit_count=0 或无记录 → True）。
    
    注意：此函数只读不写，不会改变访问计数。
    实际流程应调用 mark_visit() 原子地记录 + 判断。
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT visit_count FROM user_state WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if not row:
                return True  # 无记录 → 首次
            return row[0] == 0  # visit_count=0 也算首次（兼容手动插入的初始状态）


# ===== 长期偏好存取 =====

_UPDATE_PREFERENCES_SQL = """
UPDATE user_state
SET preferences = preferences || %(prefs)s::jsonb,
    synced_at   = NOW()
WHERE user_id = %(user_id)s
"""


def save_preferences(user_id: str, prefs: dict[str, Any]) -> None:
    """保存用户长期偏好（goal/available_time/intensity_pref 等）。
    
    JSONB || 操作符会合并字段，新字段覆盖旧字段，保留未更新字段。
    """
    if not prefs:
        return
    from psycopg.types.json import Jsonb

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_UPDATE_PREFERENCES_SQL, {"user_id": user_id, "prefs": Jsonb(prefs)})
            logger.info("user_state.save_preferences", user_id=user_id, keys=list(prefs.keys()))


def get_preferences(user_id: str) -> dict[str, Any]:
    """读取用户长期偏好。
    
    Returns:
        偏好字典（{"goal": "...", "available_time": 30, ...}），无记录返回空 dict。
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT preferences FROM user_state WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if not row or not row[0]:
                return {}
            return dict(row[0])  # JSONB → dict
