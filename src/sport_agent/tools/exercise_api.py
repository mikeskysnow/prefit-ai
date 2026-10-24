"""APP 后端运动数据接口客户端 + ETL 转换。

对应接口文档《下载运动数据接口文档.md》：
- POST /buyer/data/download/exerciseByUserId
- 入参：userId, startDate, endDate（query params）
- 返回 ExerciseDTO 列表（含秒级心率/GPS等原始数据）

本模块职责：
1. 调用 APP 接口拉取原始运动数据
2. ETL：将庞大的 ExerciseDTO 聚合为 training_history 摘要
3. 降级：接口不可用时返回空列表，调用方走兜底逻辑
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sport_agent.config import get_settings
from sport_agent.tools.base import http_client
from sport_agent.utils import get_logger

logger = get_logger(__name__)

# sportType 映射：APP原始枚举 → 摘要 sport_type
_SPORT_TYPE_MAP: dict[int, str] = {
    0x01: "walking",
    0x02: "running",
    0x03: "walk_run",
    0x04: "easy_run",
    0x05: "progressive_run",
    0x06: "interval_run",
    0x07: "steady_run",
    0x08: "tempo_run",
    0x09: "long_slow_run",
    0x10: "fitness",
    0x11: "stress_relief_walk",
    0x12: "stress_relief_run",
    0x13: "outdoor_walk_run",
    0x14: "indoor",
}


def fetch_exercises(
    user_id: str, start_date: str, end_date: str
) -> list[dict[str, Any]]:
    """调用 APP 后端接口拉取原始运动数据。

    Args:
        user_id: 用户 ID
        start_date: 起始日期 yyyy-MM-dd
        end_date: 终止日期 yyyy-MM-dd

    Returns:
        ExerciseDTO 原始列表；接口不可用或失败时返回空列表（降级）。
    """
    s = get_settings()
    if not s.app_api_base:
        logger.info("exercise_api.skip_no_base_url", user_id=user_id)
        return []

    try:
        import httpx

        with httpx.Client(timeout=s.app_api_timeout) as c:
            resp = c.post(
                f"{s.app_api_base}/buyer/data/download/exerciseByUserId",
                params={
                    "userId": user_id,
                    "startDate": start_date,
                    "endDate": end_date,
                },
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        if not data.get("success", False):
            logger.warning("exercise_api.api_error", message=data.get("message", ""))
            return []

        result = data.get("result", [])
        logger.info("exercise_api.fetched", user_id=user_id, count=len(result))
        return result
    except Exception as e:  # noqa: BLE001
        logger.warning("exercise_api.fetch_failed", user_id=user_id, error=str(e))
        return []


def transform_exercise(raw: dict[str, Any]) -> Optional[dict[str, Any]]:
    """ETL：将单条 ExerciseDTO 转换为 training_history 摘要记录。

    聚合策略：
    - 丢弃秒级时序数据（hrItems/kmSpaceItems/GPS等）
    - 保留核心统计指标（平均心率/配速/距离/卡路里/训练负荷）
    - 衍生计算强度分区、疲劳等级
    """
    sport_id = raw.get("sportId")
    if not sport_id:
        return None

    # 时间戳（秒）→ datetime
    start_ts = raw.get("startSportTime", 0)
    end_ts = raw.get("finishSportTime", 0)
    if not start_ts:
        return None

    try:
        start_time = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    except (ValueError, OSError):
        return None

    end_time = None
    if end_ts:
        try:
            end_time = datetime.fromtimestamp(end_ts, tz=timezone.utc)
        except (ValueError, OSError):
            end_time = None

    # 运动类型映射
    raw_type = raw.get("sportType", 0x02)
    sport_type = _SPORT_TYPE_MAP.get(raw_type, "running")

    # 核心统计
    duration_sec = raw.get("sportDuration", 0) or 0
    distance_m = float(raw.get("totalDistance", 0) or 0)
    avg_hr = raw.get("avgHeatRate") or raw.get("avgHeartRate")
    max_hr = raw.get("maxRealTimeHeatRate")
    avg_pace = raw.get("avgPace")
    calories = raw.get("totalCalories", 0) or raw.get("effectiveCalories", 0) or 0
    training_load = raw.get("trainingLoad")

    # 衍生：强度分区（基于平均心率粗估，真实场景需用户最大心率）
    intensity_zone = _estimate_intensity_zone(avg_hr, raw.get("currentAge", 30))
    fatigue_level = _estimate_fatigue(training_load, duration_sec)
    recovery_score = _estimate_recovery_score(raw.get("recoveryTime", 0))

    # 日期
    raw_date_str = raw.get("date", "")
    raw_date = None
    if raw_date_str:
        try:
            raw_date = datetime.strptime(raw_date_str, "%Y-%m-%d").date()
        except ValueError:
            raw_date = start_time.date()

    return {
        "sport_type": sport_type,
        "start_time": start_time,
        "end_time": end_time,
        "duration_sec": int(duration_sec),
        "distance_m": distance_m,
        "avg_hr": int(avg_hr) if avg_hr else None,
        "max_hr": int(max_hr) if max_hr else None,
        "avg_pace": float(avg_pace) if avg_pace else None,
        "calories": int(calories),
        "intensity_zone": intensity_zone,
        "training_load": float(training_load) if training_load else None,
        "recovery_score": recovery_score,
        "body_battery_delta": None,  # 由 body_battery 算法后续回填
        "weather_tag": None,  # 可后续由天气数据补充
        "location_type": "indoor" if raw_type == 0x14 else "outdoor",
        "fatigue_level": fatigue_level,
        "raw_session_id": str(sport_id),
        "raw_date": raw_date,
    }


def _estimate_intensity_zone(avg_hr: Optional[int], age: int) -> str:
    """基于平均心率粗估强度分区（田中公式估算最大心率）。"""
    if not avg_hr or avg_hr <= 0:
        return "Z2"
    # 田中公式：HRmax = 208 - 0.7 × 年龄（比 220-年龄 更准）
    hr_max = 208 - 0.7 * max(age, 1)
    ratio = avg_hr / hr_max
    if ratio < 0.6:
        return "Z1"
    if ratio < 0.7:
        return "Z2"
    if ratio < 0.8:
        return "Z3"
    if ratio < 0.9:
        return "Z4"
    return "Z5"


def _estimate_fatigue(training_load: Optional[float], duration_sec: int) -> str:
    """粗估疲劳等级。"""
    load = training_load or 0
    duration_min = duration_sec / 60
    # 简化：负荷>300 或 时长>60min → high
    if load > 300 or duration_min > 60:
        return "high"
    if load > 150 or duration_min > 30:
        return "medium"
    return "low"


def _estimate_recovery_score(recovery_time_sec: int) -> Optional[int]:
    """根据恢复时长估算恢复评分（0-100，越长越低）。"""
    if not recovery_time_sec or recovery_time_sec <= 0:
        return None
    recovery_hours = recovery_time_sec / 3600
    # 0h→100, 24h→50, 48h→0 线性递减
    score = max(0, min(100, int(100 - recovery_hours * 2.1)))
    return score


def fetch_and_transform(
    user_id: str, start_date: str, end_date: str
) -> list[dict[str, Any]]:
    """拉取 + ETL 一步完成：APP接口 → 摘要列表。"""
    raw_list = fetch_exercises(user_id, start_date, end_date)
    summaries: list[dict[str, Any]] = []
    for raw in raw_list:
        summary = transform_exercise(raw)
        if summary:
            summaries.append(summary)
    logger.info("exercise_api.transformed", user_id=user_id, raw=len(raw_list), out=len(summaries))
    return summaries
