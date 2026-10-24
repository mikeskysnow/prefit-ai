"""健康域工具：生理数据采集（F001）、用户档案、运动历史。

对应 Agent: Health Agent / Recovery Agent / Safety Agent 的数据来源。

数据来源优先级（与运动数据同步一致）：
1. PostgreSQL 摘要表（由增量同步从 APP 后端拉取并 UPSERT）
2. Mock 数据（未配置 DB 或表为空时降级兜底）
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sport_agent.config import get_settings
from sport_agent.tools.base import _is_mock, http_client, sport_tool
from sport_agent.utils import get_logger

logger = get_logger(__name__)


def _mock_vitals(user_id: str) -> dict[str, Any]:
    """稳定的 Mock 生理数据（健康基线，便于骨架主流程跑通）。

    真实接入时由健康数据 API / 耳机 PPG 服务返回。
    """
    return {
        "user_id": user_id,
        "heart_rate": 72,  # 静息心率
        "spo2": 98,  # 血氧
        "hrv": 55,  # 心率变异性
        "stress": 30,  # 压力值 0-100
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "mock",
    }


def _mock_hrv_history(user_id: str, days: int) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    return [
        {"user_id": user_id, "date": (now - timedelta(days=i)).date().isoformat(), "hrv": 50 + (i % 7)}
        for i in range(days)
    ]


def _mock_profile(user_id: str) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "age": 30,
        "gender": "male",
        "height": 175,
        "weight": 70,
        "bmi": 22.9,
        "goal": "health",
        "level": "beginner",
        "history_disease": [],
        "habit": {"freq_per_week": 3, "pref_time": "evening"},
    }


def _mock_training_history(user_id: str, days: int) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    return [
        {
            "user_id": user_id,
            "date": (now - timedelta(days=i)).date().isoformat(),
            "sport_type": "running",
            "duration": 30,
            "distance": 4.0,
            "load": 200,
        }
        for i in range(0, days, 2)
    ]


@sport_tool
def get_health_data(user_id: str) -> dict[str, Any]:
    """获取用户当前基础健康状态（静息心率、血氧 SpO2、HRV、压力值）。

    用于运动前健康采集（F001），数据来源为智能耳机/健康平台。
    性能要求：抓取时间 ≤5 秒；完整数据返回率 ≥99%。

    数据来源优先级：
    1. PostgreSQL health_daily 表（由增量同步从 APP 后端写入）
       - 如果没有当日数据，获取上一次的数据（ORDER BY metric_date DESC）
    2. Mock 数据（未配置 DB 或表为空时降级兜底）
    """
    s = get_settings()
    if s.postgres_url:
        try:
            from sport_agent.db.health_metrics import query_latest_health

            latest = query_latest_health(user_id)
            if latest:
                return {
                    "user_id": user_id,
                    "heart_rate": latest.get("latest_hr") or latest.get("resting_hr") or 72,
                    "resting_hr": latest.get("resting_hr"),
                    "spo2": latest.get("spo2") or 98,
                    "hrv": latest.get("hrv") or 55,
                    "stress": latest.get("stress") or 30,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metric_date": latest.get("metric_date").isoformat() if latest.get("metric_date") else None,
                    "source": "db",
                }
            logger.info("health_data.empty_fallback_mock", user_id=user_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("health_data.db_failed_fallback_mock", error=str(e))

    if _is_mock():
        return _mock_vitals(user_id)

    # 兼容旧接口（health_data_api_base 配置时）
    with http_client() as c:
        resp = c.get(f"{s.health_data_api_base}/vitals", params={"user_id": user_id})
        resp.raise_for_status()
        return resp.json()


@sport_tool
def get_hrv_history(user_id: str, days: int = 7) -> list[dict[str, Any]]:
    """获取用户近 N 天 HRV 历史，用于身体电量基线计算（F004）。

    校准基线至少需要 7 天运动数据。

    数据来源优先级：
    1. PostgreSQL health_daily 表的 hrv 字段
    2. Mock 数据（未配置 DB 或无 HRV 数据时降级兜底）
    """
    s = get_settings()
    if s.postgres_url:
        try:
            from sport_agent.db.health_metrics import query_health_history

            history = query_health_history(user_id, days=days)
            hrv_history = [
                {
                    "user_id": user_id,
                    "date": h["metric_date"].isoformat() if h.get("metric_date") else None,
                    "hrv": h.get("hrv"),
                }
                for h in history
                if h.get("hrv") is not None
            ]
            if hrv_history:
                return hrv_history
            logger.info("hrv_history.empty_fallback_mock", user_id=user_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("hrv_history.db_failed_fallback_mock", error=str(e))

    return _mock_hrv_history(user_id, days)


@sport_tool
def get_user_profile(user_id: str) -> dict[str, Any]:
    """获取用户画像（年龄/BMI/运动习惯/历史疾病），用于风险评估与计划生成。

    数据来源优先级：
    1. PostgreSQL user_profile 表（由增量同步从 APP 后端写入）
    2. Mock 数据（未配置 DB 或无数据时降级兜底）
    """
    s = get_settings()
    if s.postgres_url:
        try:
            from sport_agent.db.health_metrics import query_user_profile

            profile = query_user_profile(user_id)
            if profile:
                gender_val = profile.get("gender")
                gender_str = "male" if gender_val == 1 else ("female" if gender_val == 0 else "unknown")
                return {
                    "user_id": user_id,
                    "age": profile.get("age") or 30,
                    "gender": gender_str,
                    "height": profile.get("height") or 175,
                    "weight": profile.get("weight") or 70,
                    "bmi": profile.get("bmi") or 22.9,
                    "resting_hr": profile.get("resting_hr"),
                    "goal": "health",
                    "level": "beginner",
                    "history_disease": [],
                    "habit": {"freq_per_week": 3, "pref_time": "evening"},
                    "source": "db",
                }
            logger.info("user_profile.empty_fallback_mock", user_id=user_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("user_profile.db_failed_fallback_mock", error=str(e))

    return _mock_profile(user_id)


@sport_tool
def get_training_history(user_id: str, days: int = 14) -> list[dict[str, Any]]:
    """获取用户近 N 天运动历史，用于恢复评估与方案分析（F004/F008）。

    连续运动 3 天需自动降低训练强度。

    数据来源优先级：
    1. PostgreSQL training_history 摘要表（由增量同步写入）
    2. Mock 数据（未配置 DB 或表为空时降级兜底）
    """
    s = get_settings()
    if s.postgres_url:
        try:
            from sport_agent.db.training_history import query_recent_history

            history = query_recent_history(user_id, days=days)
            if history:
                return history
            logger.info("training_history.empty_fallback_mock", user_id=user_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("training_history.db_failed_fallback_mock", error=str(e))

    return _mock_training_history(user_id, days)
