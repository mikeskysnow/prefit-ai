"""健康数据增量同步：智能体启动时触发 → 增量拉取 → UPSERT → 降级兜底。

与运动数据同步（training_sync.py）相同的策略：
1. 智能体启动时从 APP 后端增量拉取用户画像 + 心率/血氧/压力数据
2. UPSERT 幂等写入 user_profile + health_daily 表
3. 接口失败时不阻塞主流程，使用已有旧数据 + 标记 stale

关键需求："如果没有当日数据，获取上一次的数据给到智能体作为健康采集数据"
→ 由 query_latest_health 的 ORDER BY metric_date DESC 实现
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sport_agent.config import get_settings
from sport_agent.utils import get_logger

logger = get_logger(__name__)

# 增量拉取回溯天数
_BACKFILL_DAYS = 7


def sync_user_profile(user_id: str) -> dict[str, Any]:
    """同步用户画像。

    API 失败时兜底查 DB 已有画像，标记 db_fallback=True。

    Returns:
        {"synced": bool, "error": str|None, "db_fallback": bool}
    """
    s = get_settings()
    if not s.postgres_url:
        return {"synced": False, "error": "no_db", "db_fallback": False}

    if not s.app_api_base:
        logger.info("health_sync.profile.skip_no_api", user_id=user_id)
        return {"synced": False, "error": "no_api", "db_fallback": False}

    try:
        from sport_agent.db.health_metrics import query_user_profile, upsert_user_profile
        from sport_agent.tools.health_api import fetch_user_profile

        raw_profile = fetch_user_profile(user_id)
        if not raw_profile:
            # API 返回空 → 兜底查 DB
            existing = query_user_profile(user_id)
            if existing:
                logger.info("health_sync.profile.db_fallback", user_id=user_id)
                return {"synced": False, "error": None, "db_fallback": True}
            return {"synced": False, "error": "api_empty", "db_fallback": False}

        upsert_user_profile(user_id, raw_profile)
        logger.info("health_sync.profile.done", user_id=user_id)
        return {"synced": True, "error": None, "db_fallback": False}

    except Exception as e:  # noqa: BLE001
        logger.warning("health_sync.profile.failed", user_id=user_id, error=str(e))
        # 异常时也尝试 DB 兜底
        try:
            from sport_agent.db.health_metrics import query_user_profile

            if query_user_profile(user_id):
                logger.info("health_sync.profile.db_fallback_on_error", user_id=user_id)
                return {"synced": False, "error": str(e), "db_fallback": True}
        except Exception:  # noqa: BLE001
            pass
        return {"synced": False, "error": str(e), "db_fallback": False}


def sync_health_daily(user_id: str) -> dict[str, Any]:
    """增量同步每日健康数据（心率/血氧/压力）。

    流程：
    1. 回溯 7 天拉取三类数据
    2. 按日期合并为 health_daily 记录（同一天的心率/血氧/压力 UPSERT 到同一行）
    3. UPSERT 幂等写入
    4. API 全部失败时兜底：查询本地 DB 已有最新数据，标记 db_fallback=True

    Returns:
        {"synced": int, "error": str|None, "db_fallback": bool}
    """
    s = get_settings()
    if not s.postgres_url:
        return {"synced": 0, "error": "no_db", "db_fallback": False}

    if not s.app_api_base:
        logger.info("health_sync.daily.skip_no_api", user_id=user_id)
        return {"synced": 0, "error": "no_api", "db_fallback": False}

    now = datetime.now(timezone.utc)
    start_date = (now - timedelta(days=_BACKFILL_DAYS)).strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")

    try:
        from sport_agent.db.health_metrics import query_latest_health, upsert_health_daily
        from sport_agent.tools.health_api import (
            fetch_blood_oxygen,
            fetch_heart_rate,
            fetch_stress,
            transform_blood_oxygen,
            transform_heart_rate,
            transform_stress,
        )

        # 拉取三类数据（各自独立超时，互不影响）
        hr_list = fetch_heart_rate(user_id, start_date, end_date)
        spo2_list = fetch_blood_oxygen(user_id, start_date, end_date)
        stress_list = fetch_stress(user_id, start_date, end_date)

        logger.info(
            "health_sync.daily.fetched",
            user_id=user_id,
            hr=len(hr_list),
            spo2=len(spo2_list),
            stress=len(stress_list),
        )

        # API 全部返回空 → 兜底查 DB 已有最新数据
        if not hr_list and not spo2_list and not stress_list:
            latest = query_latest_health(user_id)
            if latest:
                logger.info("health_sync.daily.db_fallback", user_id=user_id, metric_date=str(latest.get("metric_date")))
                return {"synced": 0, "error": None, "db_fallback": True}
            logger.warning("health_sync.daily.no_data_no_fallback", user_id=user_id)
            return {"synced": 0, "error": None, "db_fallback": False}

        # 按日期合并：同一天的心率/血氧/压力合并到一条记录
        merged: dict[str, dict[str, Any]] = {}

        for raw in hr_list:
            item = transform_heart_rate(raw)
            d = item.get("metric_date")
            if d:
                merged.setdefault(d, {"metric_date": d}).update(item)

        for raw in spo2_list:
            item = transform_blood_oxygen(raw)
            d = item.get("metric_date")
            if d:
                merged.setdefault(d, {"metric_date": d}).update(item)

        for raw in stress_list:
            item = transform_stress(raw)
            d = item.get("metric_date")
            if d:
                merged.setdefault(d, {"metric_date": d}).update(item)

        records = list(merged.values())
        written = upsert_health_daily(user_id, records)

        logger.info("health_sync.daily.done", user_id=user_id, synced=written)
        return {"synced": written, "error": None, "db_fallback": False}

    except Exception as e:  # noqa: BLE001
        logger.warning("health_sync.daily.failed", user_id=user_id, error=str(e))
        # 异常时也尝试 DB 兜底
        try:
            from sport_agent.db.health_metrics import query_latest_health

            if query_latest_health(user_id):
                logger.info("health_sync.daily.db_fallback_on_error", user_id=user_id)
                return {"synced": 0, "error": str(e), "db_fallback": True}
        except Exception:  # noqa: BLE001
            pass
        return {"synced": 0, "error": str(e), "db_fallback": False}


def sync_health_data(user_id: str) -> dict[str, Any]:
    """一键同步全部健康数据（画像 + 每日指标）。

    在 health_collect_node 启动时调用。
    """
    profile_result = sync_user_profile(user_id)
    daily_result = sync_health_daily(user_id)

    return {
        "profile_synced": profile_result["synced"],
        "daily_synced": daily_result["synced"],
        "error": profile_result["error"] or daily_result["error"],
        "db_fallback": profile_result.get("db_fallback", False) or daily_result.get("db_fallback", False),
    }
