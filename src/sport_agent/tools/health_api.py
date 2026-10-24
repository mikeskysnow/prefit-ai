"""APP 后端健康数据接口客户端 + ETL 转换。

对应接口文档：
- 用户画像：POST /buyer/data/download/userInfoByUserId
- 心率数据：POST /buyer/data/download/heartRateByUserId
- 血氧数据：POST /buyer/data/download/bloodOxygenByUserId
- 压力数据：POST /buyer/data/download/stressByUserId

职责：
1. 调用 4 个 APP 接口拉取原始健康数据
2. ETL：将 DailyVO 列表转换为 health_daily 摘要
3. 降级：接口不可用时返回空，调用方走兜底逻辑
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sport_agent.config import get_settings
from sport_agent.tools.base import http_client
from sport_agent.utils import get_logger

logger = get_logger(__name__)


def _post(endpoint: str, params: dict[str, Any]) -> Optional[dict[str, Any]]:
    """统一 POST 调用封装，返回 JSON dict 或 None（失败/未配置）。

    使用 app_api_timeout（默认10s）而非 LLM 的30s超时，避免数据同步拖慢主流程。
    """
    s = get_settings()
    if not s.app_api_base:
        return None

    try:
        import httpx

        with httpx.Client(timeout=s.app_api_timeout) as c:
            resp = c.post(
                f"{s.app_api_base}{endpoint}",
                params=params,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        if not data.get("success", False):
            logger.warning("health_api.api_error", endpoint=endpoint, message=data.get("message", ""))
            return None

        return data.get("result")
    except Exception as e:  # noqa: BLE001
        logger.warning("health_api.fetch_failed", endpoint=endpoint, error=str(e))
        return None


# ===== 用户画像 =====


def fetch_user_profile(user_id: str) -> Optional[dict[str, Any]]:
    """拉取用户画像（MemberVO）。"""
    result = _post("/buyer/data/download/userInfoByUserId", {"userId": user_id})
    if result:
        logger.info("health_api.profile.fetched", user_id=user_id, age=result.get("age"))
    return result


# ===== 心率数据 =====


def fetch_heart_rate(user_id: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    """拉取心率数据（List[HeartRateDailyVO]）。"""
    result = _post(
        "/buyer/data/download/heartRateByUserId",
        {"userId": user_id, "startDate": start_date, "endDate": end_date},
    )
    return result or []


def transform_heart_rate(raw: dict[str, Any]) -> dict[str, Any]:
    """ETL：HeartRateDailyVO → health_daily 心率字段。"""
    metric_date = _parse_date(raw.get("date", ""))

    # 估算 HRV（接口未直接提供，从 hrItems 的 RR 间期变化粗估）
    hrv = _estimate_hrv_from_items(raw.get("hrItems", []))

    return {
        "metric_date": metric_date,
        "resting_hr": raw.get("restHeartRate"),
        "avg_hr": raw.get("avgHeartRate"),
        "max_hr": raw.get("maxHeartRate"),
        "min_hr": raw.get("minHeartRate"),
        "latest_hr": raw.get("heartRate"),
        "hrv": hrv,
        "device_name": raw.get("deviceName"),
        "mac_address": raw.get("macAddress"),
    }


# ===== 血氧数据 =====


def fetch_blood_oxygen(user_id: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    """拉取血氧数据（List[BloodOxygenDailyVO]）。"""
    result = _post(
        "/buyer/data/download/bloodOxygenByUserId",
        {"userId": user_id, "startDate": start_date, "endDate": end_date},
    )
    return result or []


def transform_blood_oxygen(raw: dict[str, Any]) -> dict[str, Any]:
    """ETL：BloodOxygenDailyVO → health_daily 血氧字段。"""
    metric_date = _parse_date(raw.get("date", ""))
    return {
        "metric_date": metric_date,
        "spo2": raw.get("bloodOxygen"),
        "avg_spo2": raw.get("avgBloodOxygen"),
        "max_spo2": raw.get("maxBloodOxygen"),
        "min_spo2": raw.get("minBloodOxygen"),
    }


# ===== 压力数据 =====


def fetch_stress(user_id: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    """拉取压力数据（List[StressDailyVO]）。"""
    result = _post(
        "/buyer/data/download/stressByUserId",
        {"userId": user_id, "startDate": start_date, "endDate": end_date},
    )
    return result or []


def transform_stress(raw: dict[str, Any]) -> dict[str, Any]:
    """ETL：StressDailyVO → health_daily 压力字段。"""
    metric_date = _parse_date(raw.get("date", ""))
    return {
        "metric_date": metric_date,
        "stress": raw.get("stress"),
        "avg_stress": raw.get("avgStress"),
        "max_stress": raw.get("maxStress"),
        "min_stress": raw.get("minStress"),
    }


# ===== 辅助函数 =====


def _parse_date(date_str: str):
    """解析日期字符串 'yyyy-MM-dd' → date 对象。"""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _estimate_hrv_from_items(hr_items: list[dict[str, Any]]) -> Optional[float]:
    """从心率明细项粗估 HRV（rMSSD 近似）。

    接口未直接提供 HRV，使用相邻心率采样的变化幅度粗估。
    真实 HRV 需要逐跳 RR 间期数据，此处仅为降级估算。
    """
    if not hr_items or len(hr_items) < 2:
        return None

    try:
        hrs = [item["hr"] for item in hr_items if item.get("hr") and item["hr"] > 0]
    except (KeyError, TypeError):
        return None

    if len(hrs) < 2:
        return None

    # 粗估：相邻心率差的 RMSSD 近似
    diffs = [(hrs[i + 1] - hrs[i]) ** 2 for i in range(len(hrs) - 1)]
    rmssd_approx = (sum(diffs) / len(diffs)) ** 0.5
    # 缩放到 HRV 常见范围（20-80ms）
    estimated = max(15.0, min(90.0, rmssd_approx * 3.5))
    return round(estimated, 1)
