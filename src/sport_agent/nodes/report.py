"""上传件解析节点：识别上传的是体检报告还是运动记录，分别处理。

对应 docs/临时需求.md「支持用户上传体检报告」与运动记录图片扩展。

两条分支：
- 体检报告（pdf 或图片）→ 结构化分析 → 更新并落库 user_profile
- 运动记录图片（如佳明手表成绩页）→ 抽取单次运动数据 → 写入 training_history，
  同时进 State 供计划生成直接引用

图片一律走多模态 LLM 视觉识别，不引入 OCR：手表界面是图标 + 大字数值的混排，
OCR 只能给出零散字符串，字段归属仍要靠 LLM 还原，多一层只会丢信息。

链路位置（在 health_collect 之前）：
    memory_load →【report_analyze】→ health_collect → risk_assess → ...

放在健康采集之前的原因：报告会改写 user_profile（含静息心率、血压、病史），
后续风险研判与计划生成都依赖画像，必须先落库再让 health_collect 读回来，
否则同一次会话里画像是旧的，报告等于白传。运动记录同理，要先落库才能被
后续按「近 7 天历史」读到。

未上传时该节点被条件路由跳过，零开销。
"""

from __future__ import annotations

from typing import Any

from sport_agent.state import SportAgentState
from sport_agent.utils import get_logger, measure

logger = get_logger(__name__)

# 报告分析的画像字段与 user_profile 列的对应关系
_PROFILE_NUMERIC_FIELDS = ("age", "height", "weight", "resting_hr")

# 报告抽取的临床指标 → 画像键。原样带入 user_profile，由 format_profile 分组文本化后
# 进 prompt。这些字段不是 user_profile 表的列，只活在 State 与本次会话的 prompt 里。
_CLINICAL_FIELDS = (
    # 决定运动安全性与强度上限
    "ecg_conclusion",
    "ecg_abnormal",
    # 决定有氧能力与能量代谢策略
    "hemoglobin",
    "fasting_glucose",
    "hba1c",
    # 影响心血管风险与代谢负荷
    "total_cholesterol",
    "ldl_c",
    "hdl_c",
    "triglycerides",
    "creatinine",
    "egfr",
    "electrolytes",
    # 决定有氧耐力水平
    "spo2",
    "fev1_fvc",
    "lung_function",
    # 影响代谢率与恢复能力
    "tsh",
    "thyroid_conclusion",
    "vitamin_d",
    "vitamin_d_unit",
    "testosterone",
    # 决定运动类型与动作选择
    "body_fat_pct",
    "muscle_mass",
    "bone_density_t",
    "bone_density_conclusion",
    "joint_condition",
    # 心血管专项
    "cardiac_conditions",
    "cardiac_risk_level",
    "cardiac_exercise_limit",
)

# 报告抽取的单次 LLM 超时（秒）。实测 8000+ 字符报告约需 20~40s，
# 全局默认 30s 会概率性超时，这里单独放宽。
REPORT_LLM_TIMEOUT = 120


def route_after_memory_load(state: SportAgentState) -> str:
    """条件路由：上传了文件走解析，否则直接进健康采集。"""
    if state.get("report_path"):
        return "report_analyze"
    return "health_collect"


def _vision_llm():
    """取用于上传件识别的 LLM：温度 0 + 非流式 + 放宽超时。"""
    from sport_agent.utils import get_llm

    return get_llm(temperature=0.0, streaming=False, timeout=REPORT_LLM_TIMEOUT)


def _image_message(data_url: str, instruction: str):
    """构造带图片的 HumanMessage（多模态内容块）。"""
    from langchain_core.messages import HumanMessage

    return HumanMessage(
        content=[
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )


def _classify_upload(kind: str, payload: str) -> Any:
    """让 LLM 判断上传件内容是体检报告还是运动记录。

    Args:
        kind: 容器格式，pdf 或 image
        payload: pdf 时为提取出的文本，image 时为 data URL

    Returns:
        UploadClassification
    """
    from langchain_core.messages import SystemMessage

    from sport_agent.agents import UPLOAD_CLASSIFY_SYSTEM, UploadClassification

    llm = _vision_llm().with_structured_output(UploadClassification)
    system = SystemMessage(content=UPLOAD_CLASSIFY_SYSTEM)

    if kind == "image":
        human = _image_message(payload, "请判断这张图片的内容类型。")
    else:
        from langchain_core.messages import HumanMessage

        # 分类只需要开头就够：报告的标题与项目名都在前部，截短可省 token 与耗时
        human = HumanMessage(content=f"请判断以下文档的内容类型。\n\n{payload[:3000]}")

    return llm.invoke([system, human])


def _system_conclusions(analysis: Any) -> dict[str, str]:
    """收集非空的分项系统结论。"""
    fields = (
        ("cardiovascular", "心血管"),
        ("metabolic", "代谢"),
        ("musculoskeletal", "骨骼肌肉"),
        ("respiratory", "呼吸"),
        ("blood_nutrition", "血液营养"),
        ("endocrine", "内分泌"),
        ("medication", "用药医嘱"),
    )
    return {cn: val for key, cn in fields if (val := getattr(analysis, key, None))}


def _merge_profile(profile: dict[str, Any], analysis: Any) -> dict[str, Any]:
    """把报告抽取结果并入画像。

    报告优先于 APP 侧画像：体检数据是医疗机构实测，比用户自填或设备估算更权威。
    但仅覆盖报告里确实有的字段，None 一律不动原值。
    """
    merged = dict(profile)

    for field in _PROFILE_NUMERIC_FIELDS:
        value = getattr(analysis, field, None)
        if value is not None:
            merged[field] = value

    if analysis.gender:
        merged["gender"] = analysis.gender

    height, weight = merged.get("height"), merged.get("weight")
    if height and weight:
        h_m = float(height) / 100
        merged["bmi"] = round(float(weight) / (h_m * h_m), 1)

    if analysis.systolic_bp and analysis.diastolic_bp:
        merged["blood_pressure"] = f"{analysis.systolic_bp}/{analysis.diastolic_bp}"

    # 关键发现并入既往病史，让 format_profile 与各评估节点自动带上
    findings = list(analysis.key_findings or [])
    if findings:
        existing = list(merged.get("history_disease") or [])
        merged["history_disease"] = existing + [f for f in findings if f not in existing]

    if analysis.exercise_capacity:
        merged["exercise_capacity"] = analysis.exercise_capacity
    if analysis.exercise_restrictions:
        merged["exercise_restrictions"] = list(analysis.exercise_restrictions)

    # 临床指标：报告有就带上，None/空列表不写入（避免把「没做这项」变成「这项为空」）
    for field in _CLINICAL_FIELDS:
        value = getattr(analysis, field, None)
        if value is None or value == []:
            continue
        merged[field] = list(value) if isinstance(value, list) else value

    # 心血管禁忌并入运动禁忌：这是硬约束，必须让 format_profile 与各评估节点都看到
    if analysis.cardiac_exercise_limit:
        limits = list(merged.get("exercise_restrictions") or [])
        if analysis.cardiac_exercise_limit not in limits:
            merged["exercise_restrictions"] = limits + [analysis.cardiac_exercise_limit]

    merged["report_systems"] = _system_conclusions(analysis)
    merged["profile_source"] = "medical_report"
    return merged


def _persist_profile(user_id: str, profile: dict[str, Any]) -> None:
    """把更新后的画像落库，供后续会话直接读到（临时需求第 3 条）。

    落库失败只告警：本次会话的 State 里已有更新后的画像，计划生成不受影响，
    只是下次会话读不到，属可接受降级。
    """
    try:
        from sport_agent.config import get_settings

        if not get_settings().postgres_url:
            return
        from sport_agent.db.health_metrics import upsert_user_profile

        gender = profile.get("gender")
        upsert_user_profile(
            user_id,
            {
                **profile,
                # 表里 gender 是 1男/0女 的 SMALLINT，这里做一次映射
                "gender": 1 if gender == "male" else (0 if gender == "female" else None),
            },
        )
        logger.info("report.profile_persisted", user_id=user_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("report.profile_persist_failed", user_id=user_id, error=str(e))


def _record_to_row(path: str, rec: Any) -> dict[str, Any]:
    """把图片抽取结果转成 training_history 的一行。

    raw_session_id 用文件内容哈希：同一张图重复上传要幂等（UPSERT 而非追加），
    而图片本身没有会话 ID 可用。
    """
    import hashlib
    from datetime import datetime
    from pathlib import Path

    digest = hashlib.sha1(Path(path).read_bytes()).hexdigest()[:24]  # noqa: S324 —— 仅做幂等键，非安全用途
    start = rec.start_time or datetime.now()

    return {
        "raw_session_id": f"img_{digest}",
        "sport_type": rec.sport_type,
        "start_time": start,
        "end_time": None,
        "duration_sec": rec.duration_sec or 0,
        "distance_m": rec.distance_m or 0,
        "avg_hr": rec.avg_hr,
        "max_hr": rec.max_hr,
        "avg_pace": rec.avg_pace_sec_per_km,
        "calories": rec.calories or 0,
        "intensity_zone": rec.intensity_zone,
        "training_load": None,
        "recovery_score": None,
        "body_battery_delta": rec.body_battery_delta,
        "weather_tag": None,
        "location_type": None,
        "fatigue_level": None,
        "raw_date": start.date(),
    }


def _persist_sport_record(user_id: str, path: str, rec: Any) -> bool:
    """把运动记录写入 training_history，供后续按近 7 天历史读到。

    落库失败只告警：State 里已有这条记录并会注入计划生成 prompt，
    本次会话的计划依据不受影响，只是进不了历史负荷统计。
    """
    try:
        from sport_agent.config import get_settings

        if not get_settings().postgres_url:
            return False
        from sport_agent.db.training_history import upsert_training_records

        upsert_training_records(user_id, [_record_to_row(path, rec)])
        logger.info("report.sport_record_persisted", user_id=user_id)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("report.sport_record_persist_failed", user_id=user_id, error=str(e))
        return False


def _analyze_medical_report(kind: str, payload: str) -> Any:
    """抽取体检报告结构化结果（pdf 走文本，图片走视觉）。"""
    from langchain_core.messages import SystemMessage

    from sport_agent.agents import (
        MEDICAL_REPORT_IMAGE_SYSTEM,
        MEDICAL_REPORT_PROMPT,
        MedicalReportAnalysis,
    )

    llm = _vision_llm().with_structured_output(MedicalReportAnalysis)

    if kind == "image":
        return llm.invoke(
            [
                SystemMessage(content=MEDICAL_REPORT_IMAGE_SYSTEM),
                _image_message(payload, "请分析这张体检报告图片，输出结构化结果。"),
            ]
        )

    messages = MEDICAL_REPORT_PROMPT.format_messages(report_text=payload)
    return llm.invoke(messages)


def _analyze_sport_record(data_url: str) -> Any:
    """抽取运动记录图片的结构化结果。"""
    from langchain_core.messages import SystemMessage

    from sport_agent.agents import SPORT_RECORD_SYSTEM, SportRecordAnalysis

    llm = _vision_llm().with_structured_output(SportRecordAnalysis)
    return llm.invoke(
        [
            SystemMessage(content=SPORT_RECORD_SYSTEM),
            _image_message(data_url, "请读取这张运动记录图片，输出结构化结果。"),
        ]
    )


def report_analyze_node(state: SportAgentState) -> dict[str, Any]:
    """识别上传件并按类型处理：体检报告更新画像，运动记录进训练历史。

    失败降级：任一环节出错都置对应的 *_ready=False 并给出面向用户的说明，
    主流程继续按常规健康数据评估，不因上传件问题中断运动。
    """
    user_id = state.get("user_id", "")
    path = state.get("report_path", "")
    logger.info("report_analyze.start", user_id=user_id, path=path)

    with measure("report_analyze") as t:
        try:
            from sport_agent.tools.report_tools import (
                detect_upload_kind,
                encode_image_data_url,
                extract_pdf_text,
            )

            container = detect_upload_kind(path)
            payload = (
                encode_image_data_url(path)
                if container == "image"
                else extract_pdf_text(path)
            )

            classification = _classify_upload(container, payload)
            upload_kind = classification.kind
            logger.info(
                "report_analyze.classified",
                user_id=user_id,
                container=container,
                kind=upload_kind,
                reason=classification.reason,
            )

            if upload_kind == "sport_record":
                if container != "image":
                    raise ValueError("运动记录目前仅支持图片上传")
                record = _analyze_sport_record(payload)
                analysis = None
            elif upload_kind == "medical_report":
                analysis = _analyze_medical_report(container, payload)
                record = None
            else:
                analysis = record = None
        except Exception as e:  # noqa: BLE001 —— 上传件是增强项，任何失败都不能中断运动前流程
            logger.warning("report_analyze.failed", user_id=user_id, error=str(e))
            return {
                "upload_kind": "unknown",
                "medical_report_ready": False,
                "sport_record_ready": False,
                "medical_report_advice": f"上传件解析失败（{e}），本次将仅依据实时体征数据评估。",
                "stage": "report_analyze",
                "node_timings": t.result,
            }

    base = {
        "upload_kind": upload_kind,
        "upload_reason": classification.reason,
        "stage": "report_analyze",
        "node_timings": t.result,
    }

    if upload_kind == "medical_report":
        return {**base, **_report_updates(user_id, state.get("user_profile") or {}, analysis, t)}
    if upload_kind == "sport_record":
        profile = state.get("user_profile") or {}
        return {**base, **_sport_updates(user_id, path, record, profile, t)}

    logger.info("report_analyze.unknown", user_id=user_id, reason=classification.reason)
    return {
        **base,
        "medical_report_ready": False,
        "sport_record_ready": False,
        "medical_report_advice": (
            f"未能识别上传内容（{classification.reason}），"
            "请上传体检报告或运动记录截图；本次将仅依据实时体征数据评估。"
        ),
    }


def _report_updates(
    user_id: str, profile: dict[str, Any], analysis: Any, t: Any
) -> dict[str, Any]:
    """体检报告分支的 State 更新：合并画像 + 落库。"""
    profile = _merge_profile(profile, analysis)
    _persist_profile(user_id, profile)

    findings = list(analysis.key_findings or [])
    restrictions = list(analysis.exercise_restrictions or [])

    logger.info(
        "report_analyze.done",
        user_id=user_id,
        kind="medical_report",
        findings=len(findings),
        restrictions=len(restrictions),
        capacity=analysis.exercise_capacity,
        ms=t.result.get("report_analyze", {}).get("ms"),
    )

    return {
        "medical_report": analysis.model_dump(),
        "medical_report_advice": analysis.exercise_advice,
        "medical_report_findings": findings,
        "medical_report_restrictions": restrictions,
        "medical_report_ready": True,
        "sport_record_ready": False,
        "user_profile": profile,
    }


def _merge_sport_profile(profile: dict[str, Any], rec: Any) -> dict[str, Any]:
    """把运动记录推断出的运动画像并入 user_profile。

    只补 APP 侧画像缺失的项，不覆盖已有值：画像里的 level 可能来自多次历史统计，
    比单张截图的推断更可靠；而习惯运动项目与强度是本次记录的直接观测，可以更新。
    """
    merged = dict(profile)

    if rec.habitual_sport:
        merged["habitual_sport"] = rec.habitual_sport
    if rec.intensity_level:
        merged["last_intensity_level"] = rec.intensity_level
    if rec.capacity_assessment:
        merged["capacity_assessment"] = rec.capacity_assessment
    # level 缺失时才用截图推断补位，已有值不动
    if rec.capacity_level and not merged.get("level"):
        merged["level"] = rec.capacity_level
    if rec.frequency_hint:
        merged["frequency_hint"] = rec.frequency_hint

    return merged


def _sport_updates(
    user_id: str, path: str, record: Any, profile: dict[str, Any], t: Any
) -> dict[str, Any]:
    """运动记录分支的 State 更新：写入训练历史 + 供计划生成引用。"""
    persisted = _persist_sport_record(user_id, path, record)
    implications = list(record.plan_implications or [])
    profile = _merge_sport_profile(profile, record)

    logger.info(
        "report_analyze.done",
        user_id=user_id,
        kind="sport_record",
        sport_type=record.sport_type,
        duration_sec=record.duration_sec,
        distance_m=record.distance_m,
        avg_hr=record.avg_hr,
        persisted=persisted,
        ms=t.result.get("report_analyze", {}).get("ms"),
    )

    return {
        # datetime 需转字符串：State 会被 checkpointer 序列化
        "sport_record": record.model_dump(mode="json"),
        "sport_record_summary": record.summary,
        "sport_record_implications": implications,
        "sport_record_ready": True,
        "sport_record_persisted": persisted,
        "medical_report_ready": False,
        "user_profile": profile,
    }
