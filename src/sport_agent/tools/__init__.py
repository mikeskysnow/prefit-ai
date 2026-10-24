"""LangChain Tools 工具层。

按业务域分组：
- health_tools: 健康数据采集（F001）、用户档案、运动历史
- training_tools: 训练方案知识库、动作库检索（F007/F009）
- environment_tools: 天气、地理位置（F008）
- device_tools: 设备状态检查（F011）
- report_tools: 体检报告 PDF 文本提取 + 运动记录图片编码

骨架阶段：未配置外部 API 时自动走 Mock 数据（见 config.use_mock_tools）。
"""

from sport_agent.tools.device_tools import check_device, get_battery
from sport_agent.tools.environment_tools import get_location, get_weather
from sport_agent.tools.health_tools import (
    get_health_data,
    get_hrv_history,
    get_training_history,
    get_user_profile,
)
from sport_agent.tools.report_tools import (
    IMAGE_SUFFIXES,
    ReportExtractError,
    detect_upload_kind,
    encode_image_data_url,
    extract_pdf_text,
)
from sport_agent.tools.training_tools import search_exercise_library, search_training_plan

__all__ = [
    "extract_pdf_text",
    "ReportExtractError",
    "IMAGE_SUFFIXES",
    "detect_upload_kind",
    "encode_image_data_url",
    "get_health_data",
    "get_hrv_history",
    "get_user_profile",
    "get_training_history",
    "search_training_plan",
    "search_exercise_library",
    "get_weather",
    "get_location",
    "check_device",
    "get_battery",
]
