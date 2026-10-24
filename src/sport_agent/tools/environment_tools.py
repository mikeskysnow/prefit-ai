"""环境域工具：天气、地理位置（F008 方案综合分析的输入）。

对应 Agent: Coach Agent 的环境感知。
"""

from __future__ import annotations

from typing import Any

from sport_agent.tools.base import _is_mock, http_client, sport_tool


@sport_tool
def get_weather(location: dict[str, Any]) -> dict[str, Any]:
    """获取指定位置的当前天气。

    用于运动方案综合分析：雨天自动推荐室内运动项目。
    """
    if _is_mock():
        return {
            "condition": "clear",
            "temp": 22,
            "humidity": 55,
            "aqi": 30,
            "wind": 3,
            "location": location.get("name", "unknown"),
        }

    import sport_agent.config as _cfg  # noqa: PLC0415

    s = _cfg.get_settings()
    with http_client() as c:
        resp = c.get(
            f"{s.weather_api_base}/weather",
            params={"lat": location.get("lat"), "lng": location.get("lng"), "key": s.weather_api_key},
        )
        resp.raise_for_status()
        return resp.json()


@sport_tool
def get_location() -> dict[str, Any]:
    """获取用户当前位置（用于判断是否常用运动地点，F006）。

    非首次用户在常用地点可跳过问询。
    """
    # 阶段2：由客户端上报地理位置，服务端反查地名 + 历史地点匹配
    return {
        "lat": 31.2304,
        "lng": 121.4737,
        "name": "上海·世纪公园",
        "is_known_place": True,
    }
