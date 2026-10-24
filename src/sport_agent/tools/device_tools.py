"""设备域工具：耳机/传感器状态检查（F011）。

对应 Agent: Device Agent。
检查项：电量、蓝牙、传感器校准、佩戴状态。
"""

from __future__ import annotations

from typing import Any

from sport_agent.tools.base import sport_tool


@sport_tool
def check_device() -> dict[str, Any]:
    """检查耳机设备状态：电量、蓝牙、传感器、佩戴。

    规则：
    - 电量 <20% 提醒充电；不足以支撑 30min 运动时提醒。
    - 蓝牙信号弱提示调整手机位置。
    - 传感器异常提示重新佩戴。
    """
    # 阶段2：接入耳机硬件驱动接口 / 客户端上报状态
    return {
        "battery": 85,
        "bluetooth": "strong",
        "sensor": "ok",
        "fit": "ok",
    }


@sport_tool
def get_battery() -> int:
    """获取耳机电量百分比。"""
    return check_device()["battery"]  # type: ignore[index]
