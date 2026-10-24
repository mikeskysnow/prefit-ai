"""LangGraph 共享状态定义。

整合自《LangGraph Agent技术设计文档》的 SportAgentState 与
《整体架构设计与落地建议》的 AgentState，覆盖运动前阶段 F001-F012 全部字段。

设计要点：
- 使用 TypedDict + total=False，所有字段可选，便于节点按需更新。
- messages 使用 add_messages reducer，支持打断/追加对话历史。
- 枚举字段用 Literal 约束，降低非法状态。
- stage 字段用于追踪流程当前位置（与 LangSmith trace 对照）。
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages


def merge_timings(left: Optional[dict], right: Optional[dict]) -> dict:
    """node_timings reducer：按节点名合并耗时，避免并发节点互相覆盖。"""
    merged = dict(left or {})
    merged.update(right or {})
    return merged

# 运动前阶段流程位置标识（与节点名一一对应，便于追踪与路由）
StageName = Literal[
    "report_analyze",  # 上传件解析：体检报告或运动记录（仅上传时进入）
    "health_collect",  # F001 健康采集
    "risk_assess",  # F002 风险预警
    "feasibility",  # F003 可行性评估
    "recovery",  # F004 身体电量
    "avatar",  # F005 用户形象
    "interview",  # F006 目标问询
    "plan_generate",  # F007 计划生成
    "plan_analyze",  # F008 方案分析
    "plan_compose",  # F009 动作编排
    "plan_edit",  # F010 计划修改
    "device_check",  # F011 设备检查
    "warmup",  # F012 热身指导
    "safety_stop",  # 高风险阻断
    "end",
]

RiskLevel = Literal["low", "mid", "high", "forbidden"]
RecoveryStatus = Literal["READY", "MODERATE", "TIRED", "INITIALIZING"]
SportGoal = Literal["fat_loss", "cardio", "endurance", "muscle", "health"]
IntensityPref = Literal["low", "mid", "high"]


class SportAgentState(TypedDict, total=False):
    """运动前阶段智能体共享状态。

    所有 Agent/节点读写同一份 State，LangGraph 负责状态流转与持久化。
    """

    # ===== 用户与会话 =====
    user_id: str
    session_id: str
    is_first_time: bool
    user_profile: dict  # 年龄/BMI/身高/体重/运动习惯/历史疾病

    # ===== 长期记忆（Mem0，运动前入口一次性快照读取）=====
    memory: dict  # {"physio": [...], "preference": [...], "history": [...], "pre_sport": [...]}
    memory_ready: bool  # 记忆层是否可用（False = 降级运行，不影响主流程）

    # ===== 流程位置 =====
    stage: StageName

    # ===== F001 健康采集 =====
    vitals: dict  # {"heart_rate", "spo2", "hrv", "stress", ...}
    vitals_fresh: bool  # 数据是否在有效窗口内
    vitals_fetch_ts: str  # ISO8601，数据采集时间
    vitals_age_days: int  # 测量数据距今天数（0=当日）
    vitals_abnormal: list  # F001 异常数据自动标记 [{"metric","value","severity","note"}]
    vitals_advice: str  # 数据过旧时的重新检测建议话术
    training_sync: dict  # 运动历史增量同步结果 {"synced", "stale", "error"}
    health_sync: dict  # 健康数据增量同步结果 {"profile_synced", "daily_synced", "error"}
    habitual_duration: int  # 经常运动时长（分钟）：从近30天运动记录推导，替代问询 available_time
    custom_vitals: dict  # 用户自定义健康数据，覆盖自动采集的 vitals 和运动能力统计

    # ===== F002 风险预警 =====
    risk_level: RiskLevel
    risk_flags: list[str]  # 异常项标记，如 ["resting_hr_high", "spo2_low"]
    risk_reason: list[str]  # 风险原因描述
    risk_suggestion: str  # 运动建议话术
    risk_layer: str  # 判定来源：rule / llm / rule+llm
    risk_rule_level: str  # Layer1 规则引擎的初步结论（LLM 只能维持或下调）
    risk_rag_context: str  # RAG 检索到的医学知识摘要（灰色地带研判依据）

    # ===== F003 可行性评估 =====
    feasibility_dashboard: dict  # 看板聚合：当前数据 + 历史档案 + 风险等级 + 建议

    # ===== F004 身体电量 =====
    body_energy: int  # 0-100
    recovery_status: RecoveryStatus
    baseline_ready: bool  # 是否满 7 天数据（新用户初始化标志）
    body_energy_factors: dict  # 各分量得分：{hrv, cv, sleep, load_recovery}
    body_energy_confidence: str  # high / medium / low（数据量决定置信度）
    body_energy_interpretation: str  # LLM 解读文本

    # ===== 上传件（临时需求：体检报告 PDF → 更新画像；运动记录图片 → 作为计划依据）=====
    report_path: str  # 上传件路径（体检报告 pdf 或运动记录图片）；为空则跳过解析节点
    upload_kind: str  # LLM 识别结果：medical_report / sport_record / unknown
    upload_reason: str  # 识别依据（面向用户说明为什么判成这一类）

    # 体检报告分支
    medical_report: dict  # 分项分析结果（心血管/代谢/骨骼肌肉/呼吸/血液营养/内分泌等）
    medical_report_advice: str  # 面向用户输出的报告分析结论与运动建议
    medical_report_findings: list[str]  # 关键异常发现（供风险研判与计划生成引用）
    medical_report_restrictions: list[str]  # 报告导出的运动禁忌/限制项
    medical_report_ready: bool  # 解析是否成功（False = 未上传或解析失败，走原流程）

    # 运动记录分支
    sport_record: dict  # 图片抽取的单次运动数据（时长/距离/心率/配速/热量等）
    sport_record_summary: str  # 面向用户的一句话总结
    sport_record_implications: list[str]  # 这条记录对今日计划的影响
    sport_record_ready: bool  # 识别是否成功
    sport_record_persisted: bool  # 是否已写入 training_history（无库时为 False）

    # ===== F005 用户形象 =====
    avatar: dict  # {"gender", "bmi", "body_ratio", "style"}

    # ===== F006 目标问询 =====
    goal: SportGoal
    available_time: int  # 分钟
    intensity_pref: IntensityPref
    scene: str  # 非常用地点时问到的运动情景描述（需求6）
    interrupted: bool  # 用户打断标志（打断即清空当前播报）
    inquiry_completed: bool
    tts_cancel: bool  # 打断后要求客户端停止当前语音播报（需求6）
    inquiry_qa: list  # 问询记录 [{"field","question","answer","reasked"}]
    inquiry_reask_count: dict  # 各字段答非所问重问次数

    # ===== 环境数据 =====
    weather: dict  # {"condition", "temp", "humidity", "aqi"}
    location: dict  # {"lat", "lng", "name", "is_known_place"}

    # ===== F007-F010 运动计划 =====
    plan: dict  # {"warmup": [], "main": {}, "cooldown": []}
    plan_analysis_stream: str  # F008 流式分析文本
    plan_review_qa: list  # F008.5 用户问询记录 [{"question", "answer"}]
    dialog_turns: int  # 已进行的对话轮次（问询 + 方案问询累计，上限 MAX_DIALOG_TURNS）
    dialog_limit_reached: bool  # 是否已达轮次上限（达到后不再追问，直接推进流程）
    plan_actions: dict  # F009 动作编排（含动作ID/视频/图文/语音/注意事项）
    plan_adjusted: bool  # F010 是否被用户调整过
    plan_edit_attempts: int  # F010 计划调整轮次（需求10：≤2次）
    plan_edit_feedback: str  # F010 最近一次指令分类结果与反馈
    plan_edit_request: str  # plan_review 中用户提出的调整诉求，交由 plan_edit 执行

    # ===== F011 设备检查 =====
    device_status: dict  # {"battery", "bluetooth", "sensor", "fit"}
    device_ready: bool

    # ===== F012 热身指导 =====
    warmup_guide: dict  # {"steps": [...], "voice_script": "..."}

    # ===== 对话与控制 =====
    messages: Annotated[list, add_messages]  # 对话历史（含打断记录）
    user_message: str  # 最近一次用户输入
    intent: str  # 解析后的意图
    next_action: str  # Agent 控制位（路由用）

    # ===== 性能度量（对照需求各项 SLA，供 LangSmith / 前端校验）=====
    node_timings: Annotated[dict, merge_timings]  # {node: {"ms","sla_ms","within_sla"}}


class UserInput(TypedDict, total=False):
    """外部调用智能体时的初始入参（API 层 → State 初始化）。"""

    user_id: str
    session_id: str
    is_first_time: bool
    user_message: str
    location: dict
    available_time: int
    report_path: str  # 上传件路径：体检报告 pdf 或运动记录图片（png/jpg/jpeg/webp）
