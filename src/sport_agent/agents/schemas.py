"""LLM 结构化输出 Schema：用 Pydantic 强制约束 LLM 返回结构。

对应 LangChain with_structured_output，保证 F007 计划生成结果可被前端直接渲染。
文档要求：方案匹配度≥80%、推荐方案采纳率≥80%，结构化输出是可校验前提。

上传件识别：体检报告 PDF / 运动记录图片 → 结构化抽取 → 更新画像或训练历史。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class TrainingPlan(BaseModel):
    """F007 个性化运动计划（LLM 结构化输出）。"""

    type: Literal["running", "walk", "indoor_bike", "fitness"] = Field(
        description="运动类型：running 跑步 / walk 快走 / indoor_bike 室内骑行 / fitness 健身"
    )
    duration: int = Field(
        ge=5,
        le=120,
        description="单次运动时长（分钟），需不超过用户可用时间",
    )
    zone: str = Field(
        description="目标心率区间，如 Zone1(恢复)/Zone2(有氧)/Zone3(耐力)/Zone4(阈值)",
    )
    intensity: Literal["low", "mid", "high"] = Field(
        description="强度等级：low 低 / mid 中 / high 高",
    )
    reasoning_steps: list[str] = Field(
        default_factory=list,
        description=(
            "推导过程，3-6 条，每条一句话且必须带具体数值或结论来源，"
            "按「安全上限 → 体能与恢复 → 历史负荷 → 目标与偏好 → 环境 → 最终取舍」的顺序展开。"
            "例如「静息心率 52 bpm 且报告示窦性心动过缓，最大强度上限压到 Zone2」。"
            "不得写「根据综合评估」这类没有依据的空话"
        ),
    )
    rationale: str = Field(
        description=(
            "推荐理由：把 reasoning_steps 的推导过程总结成一段连贯说明，150-300 字。"
            "必须引用身体电量/风险等级/体检指标/历史负荷/天气等真实输入数值，"
            "并说明为什么选这个运动类型、这个时长、这个强度，不可编造数据"
        ),
    )

    def to_plan_dict(self) -> dict:
        """转为 State.plan 使用的 dict 结构（与规则方案保持一致）。"""
        return {
            "type": self.type,
            "duration": self.duration,
            "zone": self.zone,
            "intensity": self.intensity,
            "rationale": self.rationale,
            "reasoning_steps": list(self.reasoning_steps),
        }


class MedicalReportAnalysis(BaseModel):
    """体检报告结构化抽取（临时需求：上传 PDF → 分析 → 更新画像）。

    字段按临时需求第 2 条的画像维度划分。所有字段可空：体检套餐项目不固定，
    强制要求反而会诱导 LLM 编造数值 —— 缺项一律留空由下游按「未提供」处理。
    """

    # ----- 基础人口学（用于校准画像）-----
    gender: Optional[Literal["male", "female"]] = Field(
        default=None, description="性别，报告未写则留空"
    )
    age: Optional[int] = Field(default=None, ge=1, le=120, description="年龄（岁）")
    height: Optional[float] = Field(default=None, ge=50, le=250, description="身高（cm）")
    weight: Optional[float] = Field(default=None, ge=20, le=300, description="体重（kg）")

    # ----- 基础体征 -----
    resting_hr: Optional[int] = Field(
        default=None, ge=20, le=200, description="静息心率（次/分），取心电图或体格检查心率"
    )
    systolic_bp: Optional[int] = Field(
        default=None, ge=50, le=300, description="收缩压（高压，mmHg）"
    )
    diastolic_bp: Optional[int] = Field(
        default=None, ge=30, le=200, description="舒张压（低压，mmHg）"
    )

    # ----- 决定运动安全性与强度上限：心电图 -----
    ecg_conclusion: Optional[str] = Field(
        default=None,
        description="心电图结论原文，如「窦性心动过缓」「完全性右束支传导阻滞」；报告未做则留空",
    )
    ecg_abnormal: Optional[bool] = Field(
        default=None,
        description="心电图是否异常。仅在报告明确给出心电图结论时填写，未做心电图留空",
    )

    # ----- 决定有氧能力与能量代谢：血液与代谢指标 -----
    hemoglobin: Optional[float] = Field(
        default=None, ge=30, le=250, description="血红蛋白（g/L），决定携氧能力"
    )
    fasting_glucose: Optional[float] = Field(
        default=None, ge=1, le=40, description="空腹血糖（mmol/L）"
    )
    hba1c: Optional[float] = Field(
        default=None, ge=2, le=20, description="糖化血红蛋白 HbA1c（%）"
    )
    total_cholesterol: Optional[float] = Field(
        default=None, ge=1, le=20, description="总胆固醇（mmol/L）"
    )
    ldl_c: Optional[float] = Field(default=None, ge=0.1, le=15, description="LDL-C（mmol/L）")
    hdl_c: Optional[float] = Field(default=None, ge=0.1, le=10, description="HDL-C（mmol/L）")
    triglycerides: Optional[float] = Field(
        default=None, ge=0.1, le=30, description="甘油三酯（mmol/L）"
    )
    creatinine: Optional[float] = Field(
        default=None, ge=10, le=2000, description="肌酐（umol/L），反映肾功能"
    )
    egfr: Optional[float] = Field(
        default=None, ge=1, le=200, description="eGFR 估算肾小球滤过率（mL/min/1.73m²）"
    )
    electrolytes: Optional[str] = Field(
        default=None,
        description="电解质异常摘要，如「血钾 3.2 mmol/L 偏低」；全部正常或未做则留空",
    )

    # ----- 决定有氧耐力：肺功能与血氧 -----
    spo2: Optional[int] = Field(default=None, ge=50, le=100, description="血氧饱和度（%）")
    fev1_fvc: Optional[float] = Field(
        default=None, ge=1, le=100, description="肺功能 FEV1/FVC 比值（%），报告未做则留空"
    )
    lung_function: Optional[str] = Field(
        default=None, description="肺功能结论原文，如「轻度限制性通气功能障碍」"
    )

    # ----- 影响代谢率与恢复能力：内分泌与营养 -----
    tsh: Optional[float] = Field(
        default=None, ge=0.01, le=100, description="甲状腺功能 TSH（mIU/L）"
    )
    thyroid_conclusion: Optional[str] = Field(
        default=None, description="甲状腺功能结论，如「甲功正常」「亚临床甲减」"
    )
    vitamin_d: Optional[float] = Field(
        default=None, ge=1, le=200, description="25-羟维生素D（ng/mL 或 nmol/L，按报告单位原值填）"
    )
    vitamin_d_unit: Optional[str] = Field(
        default=None, description="维生素D 的单位原文，如 ng/mL 或 nmol/L"
    )
    testosterone: Optional[str] = Field(
        default=None, description="睾酮结果原文（含数值与单位），如「12.4 nmol/L」"
    )

    # ----- 决定运动类型与动作选择：体成分与骨关节 -----
    body_fat_pct: Optional[float] = Field(
        default=None, ge=2, le=70, description="体脂率（%）"
    )
    muscle_mass: Optional[float] = Field(
        default=None, ge=5, le=120, description="肌肉量（kg）"
    )
    bone_density_t: Optional[float] = Field(
        default=None, ge=-8, le=5, description="骨密度 T 值；-1~-2.5 为骨量减少，<-2.5 为骨质疏松"
    )
    bone_density_conclusion: Optional[str] = Field(
        default=None, description="骨密度结论原文，如「骨量减少」"
    )
    joint_condition: Optional[str] = Field(
        default=None,
        description="关节与脊柱状况摘要，如「双膝关节退变、腰椎间盘突出」；决定禁用哪些动作",
    )

    # ----- 分项系统结论（每项一句话，只写报告中确有的异常或明确正常）-----
    cardiovascular: Optional[str] = Field(
        default=None, description="心血管：血压/心率/心电图/心脏超声等结论"
    )
    metabolic: Optional[str] = Field(
        default=None, description="代谢：血糖/血脂/尿酸/肝功能/脂肪肝等结论"
    )
    musculoskeletal: Optional[str] = Field(
        default=None, description="骨骼肌肉：骨密度/关节/脊柱/骨质增生等结论"
    )
    respiratory: Optional[str] = Field(
        default=None, description="呼吸：肺部CT/肺功能/结节等结论"
    )
    blood_nutrition: Optional[str] = Field(
        default=None, description="血液营养：血常规/贫血/白细胞/维生素等结论"
    )
    endocrine: Optional[str] = Field(
        default=None, description="内分泌：甲状腺/激素等结论"
    )
    medication: Optional[str] = Field(
        default=None, description="用药医嘱：报告中提到的长期用药或医师医嘱"
    )

    # ----- 心血管专项（用户要求：上传体检报告时重点关注心血管本身病症表现）-----
    cardiac_conditions: list[str] = Field(
        default_factory=list,
        description=(
            "心血管本身的病症表现，只写报告中确有的，每条带数值或结论原文。"
            "如「窦性心动过缓 52 次/分」「1 级高血压 148/94 mmHg」「完全性右束支传导阻滞」"
            "「左室舒张功能减低」「颈动脉斑块形成」；报告未见心血管异常则空列表"
        ),
    )
    cardiac_risk_level: Optional[Literal["none", "low", "mid", "high"]] = Field(
        default=None,
        description=(
            "心血管风险分级：none 未见异常 / low 轻度（如单纯窦缓、1 级高血压）/ "
            "mid 中度（如 2 级高血压、心肌缺血样改变、心律失常）/ "
            "high 重度（如冠心病、心衰、严重心律失常）。报告无心血管相关信息则留空"
        ),
    )
    cardiac_exercise_limit: Optional[str] = Field(
        default=None,
        description=(
            "心血管状况对运动的具体限制，必须由上面的病症表现推导，"
            "如「强度上限 Zone2，避免憋气与最大用力，运动中心率不超过 130 bpm」；"
            "未见心血管异常则留空"
        ),
    )

    # ----- 运动相关推断 -----
    key_findings: list[str] = Field(
        default_factory=list,
        description="影响运动处方的关键异常发现，每条含指标名与数值，如「心动过缓 52次/分」",
    )
    exercise_restrictions: list[str] = Field(
        default_factory=list,
        description="由报告导出的运动禁忌或限制，如「避免颈部大幅度负重动作」；无则空列表",
    )
    exercise_advice: str = Field(
        description="面向用户的运动建议总结，200 字以内，必须引用报告中的具体指标作为依据"
    )
    exercise_capacity: Optional[Literal["low", "normal", "good"]] = Field(
        default=None, description="综合运动能力评估：low 受限 / normal 一般 / good 良好"
    )


class UploadClassification(BaseModel):
    """上传件内容分类（多模态 LLM 看图/看文后判断这是什么）。

    单独一次轻量调用而不与抽取合并：两类内容的字段集完全不同，
    合成一个大 schema 会让 LLM 在不相关字段上瞎填。
    """

    kind: Literal["medical_report", "sport_record", "unknown"] = Field(
        description=(
            "内容类型：medical_report 体检报告/检验报告；"
            "sport_record 运动记录（手表或 APP 的单次运动数据，含时长/距离/心率/配速等）；"
            "unknown 两者都不是"
        )
    )
    reason: str = Field(description="判断依据，40 字以内，指出画面或文本中的关键线索")


class SportRecordAnalysis(BaseModel):
    """运动记录图片结构化抽取（佳明等手表/APP 的单次运动截图）。

    全字段可空（除运动类型与摘要）：不同手表展示项差异大，
    强制要求会诱导 LLM 编造数值，缺项留空由下游按未知处理。
    """

    sport_type: Literal["running", "walk", "indoor_bike", "fitness", "other"] = Field(
        description="运动类型，无法归类到前四种时填 other"
    )
    sport_type_raw: Optional[str] = Field(
        default=None, description="画面上的原始运动类型文字，如「户外跑步」「力量训练」"
    )

    start_time: Optional[datetime] = Field(
        default=None, description="运动开始时间，画面有日期时间就填，只有日期则时间填 00:00"
    )
    duration_sec: Optional[int] = Field(
        default=None, ge=1, le=86400, description="运动时长（秒），画面通常是 mm:ss 或 h:mm:ss，需换算"
    )
    distance_m: Optional[float] = Field(
        default=None, ge=0, le=500000, description="距离（米），画面通常是公里，需换算"
    )
    avg_hr: Optional[int] = Field(default=None, ge=30, le=250, description="平均心率（次/分）")
    max_hr: Optional[int] = Field(default=None, ge=30, le=250, description="最大心率（次/分）")
    avg_pace_sec_per_km: Optional[int] = Field(
        default=None, ge=60, le=3600, description="平均配速，换算成每公里秒数，如 6'30\" 填 390"
    )
    calories: Optional[int] = Field(default=None, ge=0, le=10000, description="消耗热量（千卡）")

    # ----- 手表衍生指标（佳明等设备特有）-----
    training_effect: Optional[str] = Field(
        default=None, description="训练效果/训练负荷相关读数原文，如「有氧 3.2 无氧 0.8」"
    )
    intensity_zone: Optional[str] = Field(
        default=None, description="主要心率区间，如 Zone2；画面未标注则留空"
    )
    body_battery_delta: Optional[int] = Field(
        default=None, ge=-100, le=100, description="身体电量变化值，画面未标注则留空"
    )

    # ----- 运动画像推断（用户要求：关注习惯运动、强度、过往能力、频率）-----
    habitual_sport: Optional[str] = Field(
        default=None,
        description=(
            "从这条记录推断的习惯运动项目，如「户外跑步」「室内骑行」。"
            "只依据画面上的运动类型填写，不要推测用户还做别的运动"
        ),
    )
    intensity_level: Optional[Literal["low", "mid", "high"]] = Field(
        default=None,
        description=(
            "本次运动的强度档位，依据心率区间、配速、训练效果综合判断："
            "low 恢复/轻松（Zone1-2、配速慢）/ mid 中等（Zone2-3）/ high 高强度（Zone4-5、"
            "接近最大心率或无氧训练效果明显）。画面信息不足以判断则留空"
        ),
    )
    capacity_assessment: Optional[str] = Field(
        default=None,
        description=(
            "过往运动能力分析，一句话，必须引用画面数值作为依据。"
            "如「8.42 公里配速 5:46、平均心率 148 bpm，属有一定基础的中等耐力水平」。"
            "画面数据不足则留空"
        ),
    )
    capacity_level: Optional[Literal["beginner", "intermediate", "advanced"]] = Field(
        default=None,
        description=(
            "运动能力档位：beginner 初学者 / intermediate 有一定基础 / advanced 高水平。"
            "依据距离、配速、心率效率综合判断；无法判断则留空"
        ),
    )
    frequency_hint: Optional[str] = Field(
        default=None,
        description=(
            "运动频率线索，仅在画面上出现周/月统计、周跑量、连续天数、训练日历等信息时填写，"
            "如「本周第 3 次、周累计 21 公里」。单次记录看不出频率就留空，不要推测"
        ),
    )

    summary: str = Field(
        description="这次运动的一句话总结，100 字以内，必须引用画面上的实际数值"
    )
    plan_implications: list[str] = Field(
        default_factory=list,
        description="这条记录对今日计划的影响，如「昨日高强度 45 分钟，今日宜降量」；无明确影响则空列表",
    )


class RiskAssessment(BaseModel):
    """F002 风险推理结构化输出（LLM 灰色地带研判 Layer 2）。

    规则引擎（Layer 1）判定为 mid 时，LLM 结合用户画像和 RAG 知识库
    做综合研判，可维持 mid 或下调为 low（不可上调为 high/forbidden）。
    """

    adjusted_level: Literal["low", "mid"] = Field(
        description="调整后的风险等级：仅可维持 mid 或下调为 low，不可上调"
    )
    reasoning: str = Field(
        description="综合研判推理过程，引用用户画像/生理数据/医学知识库依据"
    )
    suggestion: str = Field(
        description="个性化运动建议（强度/类型/注意事项），如涉及就医需明确建议"
    )
    need_doctor: bool = Field(
        description="是否建议就医检查"
    )
