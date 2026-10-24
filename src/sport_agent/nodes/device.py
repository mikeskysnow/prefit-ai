"""设备域节点：F011 运动设备状态检查。

对应 Agent: Device Agent。
检查项：耳机电量、蓝牙、传感器校准、佩戴状态。
性能要求：检查完成时间 ≤5 秒；电量检测误差 ≤5%。
"""

from __future__ import annotations

from typing import Any

from sport_agent.state import SportAgentState
from sport_agent.tools.device_tools import check_device
from sport_agent.utils import get_logger, measure

logger = get_logger(__name__)

# 需求11：电量低于20%时提醒充电
BATTERY_LOW_PCT = 20

# 耳机平均每分钟耗电百分比（估算：100% → 约300min，约0.33%/min）
BATTERY_DRAIN_RATE = 0.33


def device_check_node(state: SportAgentState) -> dict[str, Any]:
    """F011 设备状态检查。

    需求11 验收标准：
    - 电量 <20% 提醒充电（去重：只报一次，不同时报"电量低"和"电量不足以支撑运动"）。
    - 电量不足以支撑本次计划时长时提醒（需求11备注）。
    - 蓝牙信号弱提示调整手机位置。
    - 传感器异常/佩戴异常提示重新佩戴。

    性能要求：检查完成时间 ≤5 秒。
    """
    with measure("device_check") as t:
        status = check_device.invoke({})

        battery = status.get("battery", 0)
        bluetooth = status.get("bluetooth", "weak")
        sensor = status.get("sensor", "error")
        fit = status.get("fit", "bad")

        # 估算本次计划所需电量
        plan_duration = state.get("plan", {}).get("duration", 30)
        required_pct = plan_duration * BATTERY_DRAIN_RATE

        alerts: list[str] = []

        # 需求11备注：电量不足以支撑本次运动时长 → 优先报（比"电量低"更具体）
        if battery < required_pct:
            alerts.append(
                f"电量 {battery}% 可能不足以完成 {plan_duration} 分钟运动（预计需要约 {required_pct:.0f}%），建议充电"
            )
        elif battery < BATTERY_LOW_PCT:
            # 需求11：电量 <20% 且仍够本次运动 → 提醒充电（不重复上面的告警）
            alerts.append(f"耳机电量仅剩 {battery}%，建议本次运动后充电")

        if bluetooth == "weak":
            alerts.append("蓝牙信号弱，请调整手机位置")
        if sensor != "ok":
            alerts.append("传感器异常，请重新佩戴")
        if fit != "ok":
            alerts.append("佩戴状态异常，请调整")

    ready = not alerts
    device_status = {**status, "alerts": alerts, "required_battery_pct": round(required_pct, 1)}

    logger.info(
        "device_check.done",
        user_id=state.get("user_id"),
        ready=ready,
        battery=battery,
        required_pct=round(required_pct, 1),
        plan_duration=plan_duration,
        ms=round(t.ms, 1),
    )
    return {
        "device_status": device_status,
        "device_ready": ready,
        "stage": "device_check",
        "node_timings": t.result,
    }
