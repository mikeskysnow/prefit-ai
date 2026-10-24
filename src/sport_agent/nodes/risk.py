"""风险域节点：F002 风险预警（三层架构）+ 条件路由 + 高风险阻断。

三层架构（对应文档《风险规则方案.md》）：
    Layer 1 ▸ 确定性规则引擎（< 10ms，0% 漏报）
         硬阈值判定：HR>100, SpO2<90%，数据时效性检查
    Layer 2 ▸ LLM 灰色地带推理（仅 mid 级别触发）
         结合用户画像（年龄/病史/用药）综合评估
         可维持 mid 或下调为 low（不可上调）
    Layer 3 ▸ 医学知识库 RAG（pgvector，为 Layer 2 提供循证依据）
         检索运动医学指南、PAR-Q+ 标准、禁忌症文献

响应时间策略（《风险规则方案.md》第五节性能预算 + 第 169 行）：
- 需求2 的「≤1s」约束的是**用户看到预警的时间**，由 Layer1 保证（实测 <10ms）。
- Layer1 出结论后立即通过 stream writer 推送预警，随后才做 Layer2+Layer3 增强。
- 增强结论属于「异步推送」，其耗时记在 risk_gray_zone（不计入 risk_assess SLA）。
- 安全性不受影响：Layer2 只能维持 mid 或下调为 low，永不上调。
"""

from __future__ import annotations

from typing import Any, Literal

from sport_agent.state import SportAgentState
from sport_agent.utils import get_logger, measure

logger = get_logger(__name__)


def risk_assess_node(state: SportAgentState) -> dict[str, Any]:
    """F002 生理风险预警入口：Layer1 先出预警（≤1s），再做灰区增强。

    需求2 要求响应 ≤1 秒。Layer1 纯规则实测 <10ms，先推送给用户；
    Layer2(LLM)+Layer3(RAG) 属增强层，延迟不可控（RAG 检索 ~0.5s + LLM 数秒），
    按方案文档「先返回规则引擎初步结果，LLM 建议异步推送」处理，
    其耗时单独记在 risk_gray_zone，不占用 risk_assess 的 SLA。
    """
    # ===== Layer 1：SLA 关键路径 =====
    with measure("risk_assess") as t:
        updates = _layer1_rules(state)
        _push_alert(updates)
    timings = dict(t.result)

    # ===== Layer 2 + Layer 3：灰区增强（异步推送语义，不占 SLA）=====
    if updates.pop("_needs_gray_zone", False):
        with measure("risk_gray_zone") as gt:
            _apply_gray_zone(state, updates)
        timings.update(gt.result)

    updates["node_timings"] = timings
    logger.info(
        "risk_assess.done",
        user_id=state.get("user_id"),
        level=updates["risk_level"],
        rule_level=updates["risk_rule_level"],
        layer=updates["risk_layer"],
        flags=updates["risk_flags"],
        rag=bool(updates.get("risk_rag_context")),
    )
    return updates


def _push_alert(updates: dict[str, Any]) -> None:
    """Layer1 结论即时推送（需求2：≤1s 内让用户看到预警）。

    图外调用（如单元测试）没有 stream writer，静默跳过。
    """
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:  # noqa: BLE001
        return
    if writer is None:
        return
    writer(
        {
            "node": "risk_assess",
            "type": "risk_alert",
            "risk_level": updates["risk_level"],
            "risk_flags": updates["risk_flags"],
            "suggestion": updates["risk_suggestion"],
        }
    )


def _layer1_rules(state: SportAgentState) -> dict[str, Any]:
    """Layer 1：确定性规则引擎，纯 Python 无外部依赖。

    - SpO2 < 90% → forbidden（强制禁止）
    - HR > 100   → high（禁止剧烈运动）
    - HR > 90 / SpO2 < 95 / HRV < 35 / stress > 70 → mid（灰色地带）

    性能：<10ms；漏报率 0%（红线由代码保证，不依赖 LLM）。
    返回值含内部标记 `_needs_gray_zone`，由调用方消费后移除。
    """
    vitals = state.get("vitals", {})
    # 优先使用 resting_hr（静息心率阈值判定更准确），回退到 heart_rate
    hr = vitals.get("resting_hr") or vitals.get("heart_rate", 0)
    spo2 = vitals.get("spo2", 100)
    hrv = vitals.get("hrv", 50)
    stress = vitals.get("stress", 0)

    # ===== Layer 1: 确定性规则引擎 =====
    flags: list[str] = []
    reason: list[str] = []
    level: Literal["low", "mid", "high", "forbidden"] = "low"

    if spo2 < 90:
        level = "forbidden"
        flags.append("spo2_low")
        reason.append(f"血氧饱和度 {spo2}% 低于 90% 安全阈值")
    elif hr > 100:
        level = "high"
        flags.append("resting_hr_high")
        reason.append(f"静息心率 {hr} 高于 100 次/分")

    # 二级软规则（趋势性 → 灰色地带）
    if level == "low":
        if hr > 90:
            level = "mid"
            flags.append("resting_hr_elevated")
            reason.append(f"静息心率 {hr} 偏高")
        if spo2 < 95 and level != "mid":
            level = "mid"
            flags.append("spo2_borderline")
            reason.append(f"血氧 {spo2}% 接近边界")
        if hrv < 35:
            level = "mid" if level == "low" else level
            flags.append("hrv_low")
            reason.append(f"HRV {hrv} 偏低，恢复不足")
        if stress > 70:
            flags.append("stress_high")
            reason.append(f"压力值 {stress} 偏高")
            if level == "low":
                level = "mid"

    rule_level = level
    risk_layer = "rule"
    suggestion = _risk_suggestion(level)

    return {
        "risk_level": level,
        "risk_rule_level": rule_level,
        "risk_flags": flags,
        "risk_reason": reason,
        "risk_suggestion": suggestion,
        "risk_layer": risk_layer,
        "risk_rag_context": "",
        "_needs_gray_zone": (level == "mid"),
        "stage": "risk_assess",
    }


def _apply_gray_zone(state: dict[str, Any], updates: dict[str, Any]) -> None:
    """Layer 2 + Layer 3：灰区增强（LLM 研判 + RAG 循证依据）。

    原地修改 `updates`，将 Layer1 的初步结论精化为 LLM 增强结论。
    LLM 只能维持 mid 或下调为 low，不可上调（安全约束）。
    """
    vitals = state.get("vitals", {})
    # 优先使用 resting_hr，与 Layer1 规则一致
    hr = vitals.get("resting_hr") or vitals.get("heart_rate", 0)
    spo2 = vitals.get("spo2", 100)
    hrv = vitals.get("hrv", 50)
    stress = vitals.get("stress", 0)
    flags = updates["risk_flags"]
    rule_level = updates["risk_rule_level"]

    # Layer 3: RAG 检索医学知识
    rag_context = _retrieve_rag_context(hr, spo2, hrv, stress, state)
    updates["risk_rag_context"] = rag_context

    # Layer 2: LLM 灰色地带推理
    llm_result = _llm_risk_reasoning(state, rule_level, flags, hr, spo2, hrv, stress, rag_context)
    if llm_result is not None:
        # LLM 只能维持 mid 或下调为 low（安全约束）
        adjusted = llm_result.get("adjusted_level", "mid")
        if adjusted == "low":
            updates["risk_level"] = "low"
            updates["risk_reason"].append(f"LLM 综合研判下调为 low：{llm_result.get('reasoning', '')}")
        updates["risk_suggestion"] = llm_result.get("suggestion", updates["risk_suggestion"])
        updates["risk_layer"] = "rule+llm"
        if llm_result.get("need_doctor"):
            updates["risk_reason"].append("LLM 建议就医检查")


def _retrieve_rag_context(
    hr: int, spo2: float, hrv: float, stress: int, state: dict[str, Any]
) -> str:
    """Layer 3: 从医学知识库检索相关指南（pgvector RAG）。

    若 RAG 不可用返回空字符串，Layer 2 降级到纯 LLM 推理。
    """
    profile = state.get("user_profile", {})
    age = profile.get("age", 30)
    diseases = profile.get("history_disease", [])

    # 构建检索 query。数值按档位归一（年龄按 10 岁、心率/HRV 按 5 为步长），
    # 因为知识库文档本身是按区间描述的，0.1 的精度对检索结果无影响，
    # 归一后同类生理状态能命中 retrieve_medical_knowledge 的缓存。
    age_bucket = int(age) // 10 * 10
    hr_bucket = round(hr / 5) * 5
    hrv_bucket = round(hrv / 5) * 5
    query_parts = [
        f"{age_bucket}岁", f"静息心率{hr_bucket}", f"血氧{int(spo2)}%", f"HRV {hrv_bucket}",
    ]
    if diseases:
        query_parts.append(f"病史：{','.join(sorted(diseases))}")
    if stress > 60:
        query_parts.append("压力偏高")
    query = "运动风险 " + " ".join(query_parts)

    try:
        from sport_agent.tools.rag_tools import retrieve_medical_knowledge

        context = retrieve_medical_knowledge(query)
        if context:
            logger.info("risk.rag.retrieved", query=query[:60], chars=len(context))
        return context
    except Exception as e:  # noqa: BLE001
        logger.warning("risk.rag.failed", error=str(e))
        return ""


def _risk_memories(state: dict[str, Any], flags: list[str]) -> str:
    """为灰区研判组装长期记忆文本。

    优先语义检索（按本次异常项构造 query，命中历史同类异常），
    检索无结果时退回入口快照的生理基线 + 历史档案两层。
    """
    from sport_agent.memory import MEM_HISTORY, MEM_PHYSIO, format_memories, search_memory

    user_id = state.get("user_id", "")
    query = "历史" + ("、".join(flags) if flags else "生理指标异常") + " 与既往风险评估结论"
    hits = search_memory(user_id, query, limit=3)
    if hits:
        return "\n".join(f"- {h}" for h in hits)
    return format_memories(state.get("memory"), types=(MEM_PHYSIO, MEM_HISTORY))


def _llm_risk_reasoning(
    state: dict[str, Any],
    rule_level: str,
    flags: list[str],
    hr: int,
    spo2: float,
    hrv: float,
    stress: int,
    rag_context: str,
) -> dict[str, Any] | None:
    """Layer 2: LLM 灰色地带综合研判（with_structured_output）。

    LLM 只能维持 mid 或下调为 low，不可上调（安全约束）。
    若 LLM 不可用返回 None，调用方降级到规则建议。
    """
    from sport_agent.config import get_settings

    if not get_settings().use_llm:
        return None

    try:
        from sport_agent.agents import RISK_REASONING_PROMPT, RiskAssessment
        from sport_agent.utils import format_profile, get_llm

        profile = format_profile(state.get("user_profile"))
        # 从画像中提取年龄，显式传入 prompt 避免LLM误认为"未提供"
        raw_profile = state.get("user_profile") or {}
        age = raw_profile.get("age") or "未提供"
        # 长期记忆：灰区研判是本次唯一能容忍秒级延迟的位置（已在增强层，SLA 独立），
        # 故这里用语义检索补充历史异常趋势与既往风险结论；快照兜底保证记忆不为空。
        memories = _risk_memories(state, flags)
        llm = get_llm()
        structured_llm = llm.with_structured_output(RiskAssessment)
        chain = RISK_REASONING_PROMPT | structured_llm

        result: RiskAssessment = chain.invoke(
            {
                "rule_level": rule_level,
                "rule_flags": ", ".join(flags) if flags else "无",
                "profile": profile,
                "age": age,
                "hr": hr,
                "spo2": spo2,
                "hrv": hrv,
                "stress": stress,
                "rag_context": rag_context or "（医学知识库不可用，请基于专业知识判断）",
                "memories": memories,
            }
        )
        logger.info("risk.llm.reasoned", adjusted=result.adjusted_level, need_doctor=result.need_doctor)
        return result.model_dump()
    except Exception as e:  # noqa: BLE001
        logger.warning("risk.llm.failed", error=str(e))
        return None


def _risk_suggestion(level: str) -> str:
    """规则版风险建议话术（LLM 不可用时降级）。"""
    if level == "forbidden":
        return "血氧过低，强制禁止高强度运动，建议尽快就医。"
    if level == "high":
        return "静息心率异常，不建议剧烈运动，建议医学检查。"
    if level == "mid":
        return "今日身体状态偏弱，建议降低运动强度，进行低强度恢复活动。"
    return "状态良好，可正常运动。"


def route_after_risk(state: SportAgentState) -> str:
    """条件路由：高风险 → safety_stop；其余 → feasibility。"""
    level = state.get("risk_level", "low")
    if level in ("high", "forbidden"):
        logger.warning("risk_route.safety_stop", user_id=state.get("user_id"), level=level)
        return "safety_stop"
    return "feasibility"


def safety_stop_node(state: SportAgentState) -> dict[str, Any]:
    """高风险阻断终点：终止运动前流程，输出就医建议。"""
    logger.warning(
        "safety_stop",
        user_id=state.get("user_id"),
        risk_level=state.get("risk_level"),
        reason=state.get("risk_reason"),
    )
    return {
        "next_action": "stop",
        "stage": "safety_stop",
    }
