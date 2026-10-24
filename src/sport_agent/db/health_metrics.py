"""健康数据表：user_profile + health_daily。

对应接口文档：
- 用户画像：/buyer/data/download/userInfoByUserId → MemberVO
- 心率数据：/buyer/data/download/heartRateByUserId → List[HeartRateDailyVO]
- 血氧数据：/buyer/data/download/bloodOxygenByUserId → List[BloodOxygenDailyVO]
- 压力数据：/buyer/data/download/stressByUserId → List[StressDailyVO]

设计要点：
- user_profile：以 user_id 为主键，UPSERT 幂等写入。
- health_daily：以 (user_id, metric_date) 唯一约束去重，聚合每日心率/血氧/压力摘要。
- "无当日数据取上一次"：查询时 ORDER BY metric_date DESC LIMIT 1 即可获取最近一条。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from sport_agent.db.connection import get_connection
from sport_agent.utils import get_logger

logger = get_logger(__name__)

# ===== 建表 SQL =====

_CREATE_USER_PROFILE = """
CREATE TABLE IF NOT EXISTS user_profile (
    user_id          VARCHAR(64) PRIMARY KEY,
    age              SMALLINT,
    gender           SMALLINT,        -- 1男 0女
    birthday         DATE,
    height           SMALLINT,        -- cm
    weight           FLOAT,           -- kg
    resting_hr       SMALLINT,        -- 静息心率（用户画像中填写）
    bmi              FLOAT,
    nick_name        VARCHAR(128),
    region           VARCHAR(256),
    synced_at        TIMESTAMPTZ DEFAULT NOW(),
    data_version     SMALLINT DEFAULT 1
);
"""

_CREATE_HEALTH_DAILY = """
CREATE TABLE IF NOT EXISTS health_daily (
    id               BIGSERIAL PRIMARY KEY,
    user_id          VARCHAR(64)  NOT NULL,
    metric_date      DATE         NOT NULL,

    -- 心率
    resting_hr       SMALLINT,
    avg_hr           SMALLINT,
    max_hr           SMALLINT,
    min_hr           SMALLINT,
    latest_hr        SMALLINT,

    -- 血氧
    spo2             SMALLINT,       -- 最近一次血氧
    avg_spo2         SMALLINT,
    max_spo2         SMALLINT,
    min_spo2         SMALLINT,

    -- 压力
    stress           SMALLINT,       -- 最近一次压力
    avg_stress       SMALLINT,
    max_stress       SMALLINT,
    min_stress       SMALLINT,

    -- HRV（耳机PPG算法产出，心率接口未直接提供时为NULL）
    hrv              FLOAT,

    -- 元数据
    device_name      VARCHAR(128),
    mac_address      VARCHAR(64),
    synced_at        TIMESTAMPTZ DEFAULT NOW(),
    data_version     SMALLINT DEFAULT 1,

    CONSTRAINT uk_user_date UNIQUE (user_id, metric_date)
);
"""

_CREATE_INDEX_LATEST = """
CREATE INDEX IF NOT EXISTS idx_health_latest
ON health_daily (user_id, metric_date DESC)
INCLUDE (resting_hr, avg_hr, spo2, avg_spo2, stress, avg_stress, hrv);
"""


def create_tables() -> None:
    """幂等建表：user_profile + health_daily + 索引。"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_CREATE_USER_PROFILE)
            cur.execute(_CREATE_HEALTH_DAILY)
            cur.execute(_CREATE_INDEX_LATEST)
    logger.info("db.health_tables.ready")


# ===== user_profile UPSERT =====

_UPSERT_PROFILE_SQL = """
INSERT INTO user_profile (
    user_id, age, gender, birthday, height, weight, resting_hr, bmi,
    nick_name, region, synced_at
) VALUES (
    %(user_id)s, %(age)s, %(gender)s, %(birthday)s, %(height)s, %(weight)s, %(resting_hr)s, %(bmi)s,
    %(nick_name)s, %(region)s, NOW()
)
ON CONFLICT (user_id) DO UPDATE SET
    age         = EXCLUDED.age,
    gender      = EXCLUDED.gender,
    birthday    = EXCLUDED.birthday,
    height      = EXCLUDED.height,
    weight      = EXCLUDED.weight,
    resting_hr  = EXCLUDED.resting_hr,
    bmi         = EXCLUDED.bmi,
    nick_name   = EXCLUDED.nick_name,
    region      = EXCLUDED.region,
    synced_at   = NOW(),
    data_version = user_profile.data_version + 1
"""


def upsert_user_profile(user_id: str, profile: dict[str, Any]) -> None:
    """UPSERT 用户画像（幂等）。"""
    # 计算 BMI
    height_m = (profile.get("height") or 0) / 100
    weight = profile.get("weight") or 0
    bmi = round(weight / (height_m * height_m), 1) if height_m > 0 and weight > 0 else None

    # 解析生日
    birthday = profile.get("birthday")
    if birthday and isinstance(birthday, str):
        try:
            birthday = datetime.fromisoformat(birthday.replace("Z", "+00:00")).date()
        except ValueError:
            try:
                birthday = datetime.strptime(birthday[:10], "%Y-%m-%d").date()
            except ValueError:
                birthday = None

    params = {
        "user_id": user_id,
        "age": profile.get("age"),
        "gender": profile.get("sex"),
        "birthday": birthday,
        "height": profile.get("height"),
        "weight": profile.get("weight"),
        "resting_hr": profile.get("heartRate"),
        "bmi": bmi,
        "nick_name": profile.get("nickName"),
        "region": profile.get("region"),
    }
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_UPSERT_PROFILE_SQL, params)
    logger.info("db.profile.upserted", user_id=user_id)


def query_user_profile(user_id: str) -> Optional[dict[str, Any]]:
    """查询用户画像。"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM user_profile WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [desc[0] for desc in cur.description]
            return dict(zip(cols, row, strict=False))


# ===== health_daily UPSERT =====

_UPSERT_HEALTH_DAILY_SQL = """
INSERT INTO health_daily (
    user_id, metric_date,
    resting_hr, avg_hr, max_hr, min_hr, latest_hr,
    spo2, avg_spo2, max_spo2, min_spo2,
    stress, avg_stress, max_stress, min_stress,
    hrv, device_name, mac_address, synced_at
) VALUES (
    %(user_id)s, %(metric_date)s,
    %(resting_hr)s, %(avg_hr)s, %(max_hr)s, %(min_hr)s, %(latest_hr)s,
    %(spo2)s, %(avg_spo2)s, %(max_spo2)s, %(min_spo2)s,
    %(stress)s, %(avg_stress)s, %(max_stress)s, %(min_stress)s,
    %(hrv)s, %(device_name)s, %(mac_address)s, NOW()
)
ON CONFLICT (user_id, metric_date) DO UPDATE SET
    resting_hr  = COALESCE(EXCLUDED.resting_hr, health_daily.resting_hr),
    avg_hr      = COALESCE(EXCLUDED.avg_hr, health_daily.avg_hr),
    max_hr      = COALESCE(EXCLUDED.max_hr, health_daily.max_hr),
    min_hr      = COALESCE(EXCLUDED.min_hr, health_daily.min_hr),
    latest_hr   = COALESCE(EXCLUDED.latest_hr, health_daily.latest_hr),
    spo2        = COALESCE(EXCLUDED.spo2, health_daily.spo2),
    avg_spo2    = COALESCE(EXCLUDED.avg_spo2, health_daily.avg_spo2),
    max_spo2    = COALESCE(EXCLUDED.max_spo2, health_daily.max_spo2),
    min_spo2    = COALESCE(EXCLUDED.min_spo2, health_daily.min_spo2),
    stress      = COALESCE(EXCLUDED.stress, health_daily.stress),
    avg_stress  = COALESCE(EXCLUDED.avg_stress, health_daily.avg_stress),
    max_stress  = COALESCE(EXCLUDED.max_stress, health_daily.max_stress),
    min_stress  = COALESCE(EXCLUDED.min_stress, health_daily.min_stress),
    hrv         = COALESCE(EXCLUDED.hrv, health_daily.hrv),
    device_name = COALESCE(EXCLUDED.device_name, health_daily.device_name),
    mac_address = COALESCE(EXCLUDED.mac_address, health_daily.mac_address),
    synced_at   = NOW(),
    data_version = health_daily.data_version + 1
"""


def upsert_health_daily(user_id: str, records: list[dict[str, Any]]) -> int:
    """批量 UPSERT 每日健康摘要（幂等）。

    records 中每条需包含 metric_date，其余指标字段可选。
    心率/血氧/压力来自不同接口，合并到同一行（按日期 UPSERT 补充）。
    """
    if not records:
        return 0

    _KEYS = {
        "metric_date", "resting_hr", "avg_hr", "max_hr", "min_hr", "latest_hr",
        "spo2", "avg_spo2", "max_spo2", "min_spo2",
        "stress", "avg_stress", "max_stress", "min_stress",
        "hrv", "device_name", "mac_address",
    }

    written = 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            for rec in records:
                params = {k: rec.get(k) for k in _KEYS}
                params["user_id"] = user_id
                cur.execute(_UPSERT_HEALTH_DAILY_SQL, params)
                written += 1
    logger.info("db.health_daily.upserted", user_id=user_id, count=written)
    return written


# ===== 查询 =====


def query_latest_health(user_id: str, target_date: Optional[date] = None) -> Optional[dict[str, Any]]:
    """查询用户最近一次健康数据。

    对应需求："如果没有当日数据，获取上一次的数据给到智能体作为健康采集数据"。

    Args:
        user_id: 用户 ID
        target_date: 指定日期（默认今天），无当日数据则取最近一条

    Returns:
        健康摘要 dict 或 None
    """
    target = target_date or datetime.now(timezone.utc).date()

    with get_connection() as conn:
        with conn.cursor() as cur:
            # 先查当日，无则取最近一条
            cur.execute(
                """
                SELECT * FROM health_daily
                WHERE user_id = %s AND metric_date <= %s
                ORDER BY metric_date DESC LIMIT 1
                """,
                (user_id, target),
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [desc[0] for desc in cur.description]
            return dict(zip(cols, row, strict=False))


def query_health_history(user_id: str, days: int = 7) -> list[dict[str, Any]]:
    """查询用户近 N 天健康历史（用于 HRV 基线计算）。"""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM health_daily
                WHERE user_id = %s AND metric_date >= %s
                ORDER BY metric_date ASC
                """,
                (user_id, since.date()),
            )
            rows = cur.fetchall()
            if not rows:
                return []
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row, strict=False)) for row in rows]
