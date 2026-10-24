"""training_history 摘要表：建表 + UPSERT + 查询 + sync_state 管理。

对应方案《运动记录数据获取.md》：
- 自建摘要表，将 APP 原始接口的秒级时序数据聚合为 LLM 可消费的结构化摘要。
- UPSERT 幂等写入：以 (user_id, raw_session_id) 唯一约束去重。
- sync_state 记录每用户最后同步时间戳，支持增量同步判断。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sport_agent.db.connection import get_connection
from sport_agent.utils import get_logger

logger = get_logger(__name__)

# ===== 建表 SQL =====

_CREATE_TRAINING_HISTORY = """
CREATE TABLE IF NOT EXISTS training_history (
    id                BIGSERIAL PRIMARY KEY,
    user_id           VARCHAR(64)  NOT NULL,
    sport_type        VARCHAR(32)  NOT NULL,
    start_time        TIMESTAMPTZ  NOT NULL,
    end_time          TIMESTAMPTZ,

    -- 核心摘要字段
    duration_sec      INT          NOT NULL DEFAULT 0,
    distance_m        FLOAT        DEFAULT 0,
    avg_hr            SMALLINT,
    max_hr            SMALLINT,
    avg_pace          FLOAT,
    calories          INT          DEFAULT 0,

    -- 智能体专用衍生字段
    intensity_zone    VARCHAR(16),
    training_load     FLOAT,
    recovery_score    SMALLINT,
    body_battery_delta INT,
    weather_tag       VARCHAR(16),
    location_type     VARCHAR(16),
    fatigue_level     VARCHAR(8),

    -- 元数据
    raw_session_id    VARCHAR(64),
    raw_date          DATE,
    synced_at         TIMESTAMPTZ  DEFAULT NOW(),
    data_version      SMALLINT     DEFAULT 1,

    CONSTRAINT uk_user_session UNIQUE (user_id, raw_session_id)
);
"""

_CREATE_INDEX_RECENT = """
CREATE INDEX IF NOT EXISTS idx_user_recent
ON training_history (user_id, start_time DESC)
INCLUDE (sport_type, duration_sec, avg_hr, intensity_zone, recovery_score, body_battery_delta);
"""

_CREATE_SYNC_STATE = """
CREATE TABLE IF NOT EXISTS sync_state (
    user_id        VARCHAR(64) PRIMARY KEY,
    last_sync_ts   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sync_count     INT         NOT NULL DEFAULT 0,
    last_status    VARCHAR(16) NOT NULL DEFAULT 'ok',
    last_error     TEXT
);
"""


def create_tables() -> None:
    """幂等建表：training_history + sync_state + 索引。"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_CREATE_TRAINING_HISTORY)
            cur.execute(_CREATE_INDEX_RECENT)
            cur.execute(_CREATE_SYNC_STATE)
        logger.info("db.tables.ready")


# ===== UPSERT =====

_UPSERT_SQL = """
INSERT INTO training_history (
    user_id, sport_type, start_time, end_time,
    duration_sec, distance_m, avg_hr, max_hr, avg_pace, calories,
    intensity_zone, training_load, recovery_score, body_battery_delta,
    weather_tag, location_type, fatigue_level,
    raw_session_id, raw_date, synced_at, data_version
) VALUES (
    %(user_id)s, %(sport_type)s, %(start_time)s, %(end_time)s,
    %(duration_sec)s, %(distance_m)s, %(avg_hr)s, %(max_hr)s, %(avg_pace)s, %(calories)s,
    %(intensity_zone)s, %(training_load)s, %(recovery_score)s, %(body_battery_delta)s,
    %(weather_tag)s, %(location_type)s, %(fatigue_level)s,
    %(raw_session_id)s, %(raw_date)s, NOW(), 1
)
ON CONFLICT (user_id, raw_session_id) DO UPDATE SET
    sport_type        = EXCLUDED.sport_type,
    start_time        = EXCLUDED.start_time,
    end_time          = EXCLUDED.end_time,
    duration_sec      = EXCLUDED.duration_sec,
    distance_m        = EXCLUDED.distance_m,
    avg_hr            = EXCLUDED.avg_hr,
    max_hr            = EXCLUDED.max_hr,
    avg_pace          = EXCLUDED.avg_pace,
    calories          = EXCLUDED.calories,
    intensity_zone    = EXCLUDED.intensity_zone,
    training_load     = EXCLUDED.training_load,
    recovery_score    = EXCLUDED.recovery_score,
    body_battery_delta = EXCLUDED.body_battery_delta,
    weather_tag       = EXCLUDED.weather_tag,
    location_type     = EXCLUDED.location_type,
    fatigue_level     = EXCLUDED.fatigue_level,
    raw_date          = EXCLUDED.raw_date,
    synced_at         = NOW(),
    data_version      = training_history.data_version + 1
"""


def upsert_training_records(user_id: str, records: list[dict[str, Any]]) -> int:
    """批量 UPSERT 运动摘要记录（幂等）。

    Args:
        user_id: 用户 ID
        records: 摘要记录列表，每条需包含 raw_session_id, sport_type, start_time 等。

    Returns:
        实际写入/更新的记录数
    """
    if not records:
        return 0

    # 确保 SQL 中引用的所有字段都有值（缺失则填 None）
    _REQUIRED_KEYS = {
        "sport_type", "start_time", "end_time",
        "duration_sec", "distance_m", "avg_hr", "max_hr", "avg_pace", "calories",
        "intensity_zone", "training_load", "recovery_score", "body_battery_delta",
        "weather_tag", "location_type", "fatigue_level",
        "raw_session_id", "raw_date",
    }

    written = 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            for rec in records:
                normalized = {k: rec.get(k) for k in _REQUIRED_KEYS}
                normalized["user_id"] = user_id
                cur.execute(_UPSERT_SQL, normalized)
                written += 1
    logger.info("db.upsert.done", user_id=user_id, count=written)
    return written


# ===== 查询 =====

_QUERY_RECENT = """
SELECT
    user_id, sport_type, start_time, end_time,
    duration_sec, distance_m, avg_hr, max_hr, avg_pace, calories,
    intensity_zone, training_load, recovery_score, body_battery_delta,
    weather_tag, location_type, fatigue_level,
    raw_session_id, raw_date
FROM training_history
WHERE user_id = %(user_id)s
  AND start_time >= %(since)s
ORDER BY start_time DESC
"""


def query_recent_history(user_id: str, days: int = 7) -> list[dict[str, Any]]:
    """查询用户近 N 天运动摘要（用于 F004/F008 分析）。

    Returns:
        摘要列表，字段与 body_battery 算法兼容。
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_QUERY_RECENT, {"user_id": user_id, "since": since})
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

    result = []
    for row in rows:
        item = dict(zip(cols, row, strict=False))
        # 适配 body_battery 算法期望的字段名（date 需为 ISO 字符串）
        raw_date = item.get("raw_date")
        item["date"] = raw_date.isoformat() if raw_date else None
        item["duration"] = item.get("duration_sec", 0) // 60  # 秒→分钟
        item["distance"] = (item.get("distance_m") or 0) / 1000  # 米→公里
        item["load"] = item.get("training_load") or 0
        result.append(item)

    logger.info("db.query.recent", user_id=user_id, days=days, count=len(result))
    return result


# ===== sync_state 管理 =====


def get_last_sync_ts(user_id: str) -> Optional[datetime]:
    """获取用户最后同步时间戳，无记录返回 None。"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT last_sync_ts FROM sync_state WHERE user_id = %s",
                (user_id,),
            )
            row = cur.fetchone()
    return row[0] if row else None


def update_sync_state(
    user_id: str, status: str = "ok", error: Optional[str] = None, count: int = 0
) -> None:
    """更新同步状态（UPSERT）。"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sync_state (user_id, last_sync_ts, sync_count, last_status, last_error)
                VALUES (%s, NOW(), %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    last_sync_ts = NOW(),
                    sync_count   = sync_state.sync_count + %s,
                    last_status  = EXCLUDED.last_status,
                    last_error   = EXCLUDED.last_error
                """,
                (user_id, count, status, error, count),
            )


def is_sync_stale(user_id: str, threshold_minutes: int = 60) -> bool:
    """判断用户数据是否需要增量同步（距上次同步超过阈值）。

    无记录视为需要同步（冷启动）。
    """
    last = get_last_sync_ts(user_id)
    if last is None:
        return True
    return datetime.now(timezone.utc) - last > timedelta(minutes=threshold_minutes)
