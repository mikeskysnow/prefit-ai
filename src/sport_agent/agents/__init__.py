"""Agent 角色层：Prompt 模板与结构化输出 Schema。

对应文档《LangGraph Agent技术设计文档》第 11 节 Prompt 设计。
- prompts: 各 Agent 系统提示词与对话模板（Coach/Safety/Conversation）
- schemas: LLM with_structured_output 的 Pydantic 模型，强制结构化输出
"""

from sport_agent.agents.prompts import (
    BODY_BATTERY_INTERPRET_PROMPT,
    COACH_SYSTEM_PROMPT,
    MEDICAL_REPORT_IMAGE_SYSTEM,
    MEDICAL_REPORT_PROMPT,
    PLAN_ANALYZE_PROMPT,
    PLAN_GENERATE_PROMPT,
    RISK_REASONING_PROMPT,
    SPORT_RECORD_SYSTEM,
    UPLOAD_CLASSIFY_SYSTEM,
)
from sport_agent.agents.schemas import (
    MedicalReportAnalysis,
    RiskAssessment,
    SportRecordAnalysis,
    TrainingPlan,
    UploadClassification,
)

__all__ = [
    "COACH_SYSTEM_PROMPT",
    "MEDICAL_REPORT_PROMPT",
    "MEDICAL_REPORT_IMAGE_SYSTEM",
    "SPORT_RECORD_SYSTEM",
    "UPLOAD_CLASSIFY_SYSTEM",
    "MedicalReportAnalysis",
    "SportRecordAnalysis",
    "UploadClassification",
    "PLAN_GENERATE_PROMPT",
    "PLAN_ANALYZE_PROMPT",
    "RISK_REASONING_PROMPT",
    "BODY_BATTERY_INTERPRET_PROMPT",
    "TrainingPlan",
    "RiskAssessment",
]
