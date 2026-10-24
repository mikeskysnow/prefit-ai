"""计划域节点：F007 计划生成 / F008 方案分析 / F008.5 方案问询(interrupt) / F009 动作编排 / F010 计划调整。

对应 Agent: Coach Agent。
- F007 生成时间 <10s，推荐方案采纳率 ≥80%。
- F008 分析按流式呈现，过程不超过 20s；雨天/连续训练 3 天自动调整。
- F008.5 方案问询：用户可打断提问，LLM 流式回答后继续流程。
- F009 热身/主运动/拉伸与运动类型匹配，图文匹配度 100%。
- F010 支持语音/按键调整，响应 <1s，ASR >95%。
"""

from __future__ import annotations

import re
from typing import Any, Optional

from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from sport_agent.agents import PLAN_ANALYZE_PROMPT, PLAN_GENERATE_PROMPT, TrainingPlan
from sport_agent.config import get_settings
from sport_agent.nodes.recovery import _consecutive_training_days
from sport_agent.state import SportAgentState
from sport_agent.tools.environment_tools import get_weather
from sport_agent.tools.health_tools import get_training_history
from sport_agent.tools.training_tools import search_exercise_library, search_training_plan
from sport_agent.utils import (
    format_medical_report,
    format_profile,
    format_sport_record,
    get_llm,
    get_logger,
    measure,
)

logger = get_logger(__name__)


def _plan_memories(state: SportAgentState) -> str:
    """方案生成用的长期记忆：偏好 + 历史运动档案。

    用入口快照（~2ms）而非语义检索：F007 有 <10s 硬性 SLA 且本身要调 LLM，
    再叠 1.4s 的 embedding 不划算；偏好类记忆总量小，全量给 LLM 即可。
    """
    from sport_agent.memory import MEM_HISTORY, MEM_PREFERENCE, format_memories

    return format_memories(state.get("memory"), types=(MEM_PREFERENCE, MEM_HISTORY))


def plan_generate_node(state: SportAgentState) -> dict[str, Any]:
    """F007 个性化运动计划生成入口：度量耗时后委托 _plan_generate_impl。

    性能要求：方案生成总用时 <10s（不含分析过程，需求7）。
    """
    with measure("plan_generate") as t:
        updates = _plan_generate_impl(state)
    updates["node_timings"] = t.result
    return updates


def _plan_generate_impl(state: SportAgentState) -> dict[str, Any]:
    """F007 个性化运动计划生成：LLM + with_structured_output(TrainingPlan)。

    流程：
    1. 检索知识库候选方案（search_training_plan，RAG 占位）。
    2. 启用 LLM 时，用结构化输出生成最终方案；否则降级到规则候选。
    3. 硬规则覆盖（_apply_hard_rules）：雨天切换室内 + 连续3天转恢复运动。

    需求7：推荐方案采纳率 ≥80%；支持动态调整计划。
    """
    goal = state.get("goal", "health")
    energy = state.get("body_energy", 50)
    risk_level = state.get("risk_level", "low")
    recovery_status = state.get("recovery_status", "MODERATE")
    profile = state.get("user_profile", {})
    # 运动时长：从历史运动记录推导的经常运动时长（不再问询用户）
    habitual_duration = state.get("habitual_duration") or profile.get("habitual_duration") or 30

    # 拉取环境数据（天气影响室内外）
    if state.get("location"):
        location_dict = state["location"]
    else:
        from sport_agent.tools.environment_tools import get_location
        location_dict = get_location.invoke({})
    weather = get_weather.invoke({"location": location_dict})

    # 知识库候选方案
    candidates = search_training_plan.invoke(
        {
            "goal": goal,
            "body_energy": energy,
            "available_time": habitual_duration,
            "risk_level": risk_level,
        }
    )

    # 连续训练天数（影响恢复建议）
    history = get_training_history.invoke({"user_id": state["user_id"], "days": 7})
    consecutive_days = _consecutive_training_days(history)

    plan: dict[str, Any] | None = None
    plan_source = "rule"

    if get_settings().use_llm:
        try:
            llm = get_llm()
            structured_llm = llm.with_structured_output(TrainingPlan)
            chain = PLAN_GENERATE_PROMPT | structured_llm
            plan_obj: TrainingPlan = chain.invoke(
                {
                    "profile": format_profile(profile),
                    "energy": energy,
                    "recovery_status": recovery_status,
                    "risk_level": risk_level,
                    "goal": goal,
                    "habitual_duration": habitual_duration,
                    "weather": weather,
                    "consecutive_days": consecutive_days,
                    "candidates": candidates,
                    # 偏好 + 历史档案：避免推荐用户排斥的运动、重复历史上被改过的安排
                    "memories": _plan_memories(state),
                    # 体检报告：禁忌是硬约束，异常指标决定强度取舍（临时需求第 3 条）
                    "medical_report": format_medical_report(state),
                    # 用户上传的运动记录：最近一次真实负荷，决定今日增减量
                    "sport_record": format_sport_record(state),
                }
            )
            plan = plan_obj.to_plan_dict()
            plan_source = "llm"
        except Exception as e:  # noqa: BLE001
            logger.warning("plan_generate.llm_fallback", error=str(e))
            plan = None

    # LLM 未启用或失败 → 规则候选降级
    if plan is None:
        plan = (
            candidates[0]
            if candidates
            else {"type": "walk", "duration": 20, "zone": "Zone1", "intensity": "low", "rationale": "默认低强度"}
        )
        plan_source = "rule_fallback" if get_settings().use_llm else "rule"

    # ===== 硬规则覆盖（需求8 验收标准，规则优先于 LLM 保证可验收）=====
    plan, rules = _apply_hard_rules(plan, weather, consecutive_days, habitual_duration)
    if rules:
        plan_source = f"{plan_source}+{'+'.join(rules)}"

    logger.info(
        "plan_generate.done",
        user_id=state.get("user_id"),
        source=plan_source,
        rules=rules,
        consecutive_days=consecutive_days,
        plan=plan,
    )
    return {
        "plan": plan,
        "weather": weather,
        "location": location_dict,
        "stage": "plan_generate",
    }


# 需求8 验收标准2：雨天自动推荐室内运动项目
_RAINY_CONDITIONS = ("rain", "storm", "shower", "thunderstorm", "snow", "雨", "雪")

# 室外 → 室内替代映射
_INDOOR_ALTERNATIVE = {
    "running": "treadmill",
    "cycling": "indoor_bike",
    "walk": "indoor_walk",
    "outdoor_run": "treadmill",
}

# 需求8 验收标准3：连续运动天数阈值
CONSECUTIVE_RECOVERY_DAYS = 3


def _is_rainy(weather: dict[str, Any]) -> bool:
    condition = str(weather.get("condition", "")).lower()
    return any(kw in condition for kw in _RAINY_CONDITIONS)


def _apply_hard_rules(
    plan: dict[str, Any],
    weather: dict[str, Any],
    consecutive_days: int,
    habitual_duration: int,
) -> tuple[dict[str, Any], list[str]]:
    """需求8 验收标准的确定性规则，优先于 LLM 输出（保证可验收、可复现）。

    规则2：雨天 → 自动推荐室内运动项目。
    规则3：连续运动 ≥3 天 → 自动推荐低强度恢复运动。

    Returns:
        (调整后的 plan, 命中的规则名列表)
    """
    plan = dict(plan)
    rules: list[str] = []

    # 规则2：雨天自动切换室内
    if _is_rainy(weather):
        sport_type = plan.get("type", "running")
        indoor = _INDOOR_ALTERNATIVE.get(sport_type)
        if indoor:
            plan["type"] = indoor
            plan["indoor"] = True
            plan["rationale"] = f"雨天自动切换为室内运动（原：{sport_type}）"
            rules.append("weather_rule")

    # 规则3：连续运动 3 天后自动推荐低强度恢复运动
    if consecutive_days >= CONSECUTIVE_RECOVERY_DAYS:
        plan["intensity"] = "low"
        plan["zone"] = "Zone1"
        plan["duration"] = max(15, min(plan.get("duration", 30), min(30, habitual_duration)))
        plan["recovery_session"] = True
        plan["rationale"] = f"已连续运动 {consecutive_days} 天，自动调整为低强度恢复运动"
        rules.append("consecutive_recovery_rule")

    return plan, rules


def plan_analyze_node(state: SportAgentState) -> dict[str, Any]:
    """F008 方案综合分析入口：度量耗时后委托 _plan_analyze_impl。

    性能要求：分析过程不超过 20s（需求8），流式呈现。
    """
    with measure("plan_analyze") as t:
        updates = _plan_analyze_impl(state)
    updates["node_timings"] = t.result
    return updates


def _plan_analyze_impl(state: SportAgentState) -> dict[str, Any]:
    """F008 运动方案综合分析：LLM 流式逐 token 呈现分析过程。

    流式实现：
    - 启用 LLM 时，用 LLM.stream() 逐 token 生成，通过 get_stream_writer() 发射
      custom 事件，API 层用 stream_mode=["updates","custom"] 透传到前端（真流式）。
    - 非 LLM 模式发射整段规则文本。

    性能要求：分析过程不超过 20s；分析结果准确匹配度 ≥80%。
    """
    writer = get_stream_writer()
    energy = state.get("body_energy", 50)
    recovery_status = state.get("recovery_status", "MODERATE")
    weather = state.get("weather", {})
    plan = state.get("plan", {})
    risk_level = state.get("risk_level", "low")

    history = get_training_history.invoke({"user_id": state["user_id"], "days": 7})
    consecutive_days = _consecutive_training_days(history)

    ctx = {
        "energy": energy,
        "recovery_status": recovery_status,
        "weather": weather,
        "risk_level": risk_level,
        "consecutive_days": consecutive_days,
        "plan": plan,
    }

    if get_settings().use_llm:
        try:
            llm = get_llm()
            parts: list[str] = []
            for chunk in llm.stream(PLAN_ANALYZE_PROMPT.format_messages(**ctx)):
                token = chunk.content
                if not isinstance(token, str):
                    token = str(token)
                if token:
                    parts.append(token)
                    writer({"node": "plan_analyze", "type": "token", "content": token})
            analysis = "".join(parts)
            logger.info("plan_analyze.llm.done", user_id=state.get("user_id"), chars=len(analysis))
            return {"plan_analysis_stream": analysis, "stage": "plan_analyze"}
        except Exception as e:  # noqa: BLE001
            logger.warning("plan_analyze.llm_fallback", error=str(e))

    # 规则降级：整段发射
    analysis = _rule_analysis(ctx)
    writer({"node": "plan_analyze", "type": "text", "content": analysis})
    logger.info("plan_analyze.rule.done", user_id=state.get("user_id"))
    return {"plan_analysis_stream": analysis, "stage": "plan_analyze"}


def _rule_analysis(ctx: dict[str, Any]) -> str:
    """规则版方案分析文本（LLM 不可用时的降级）。"""
    energy = ctx["energy"]
    recovery_status = ctx["recovery_status"]
    weather = ctx["weather"]
    plan = ctx["plan"]
    consecutive = ctx["consecutive_days"]

    lines = [
        "正在分析：",
        f"✓ 今日恢复状态：身体电量 {energy}（{recovery_status}）",
    ]
    if consecutive >= 3:
        lines.append(f"✓ 历史训练：连续训练 {consecutive} 天，已自动降低强度")
    else:
        lines.append("✓ 历史训练：已结合近 7 天负荷")
    lines.append(f"✓ 天气：{weather.get('condition', '未知')} {weather.get('temp', '')}°C")
    lines.append(f"✓ 推荐强度：{plan.get('zone', 'Zone2')} / {plan.get('intensity', 'mid')}")
    lines.append(f"→ 生成方案：{plan.get('type', 'running')} {plan.get('duration', 30)} 分钟")
    return "\n".join(lines)


# 多轮对话上限。达到后不再追问，直接推进到动作编排，避免无限循环把
# checkpoint 与 recursion_limit 撑爆。
MAX_DIALOG_TURNS = 50

# 用户表示无疑问、可以继续的说法
_CONTINUE_KEYWORDS = ("继续", "没问题", "没有了", "可以了", "就这样", "开始吧", "continue", "ok")


def _is_continue(text: str) -> bool:
    """判断用户是否表示「没有疑问了，继续」。

    只匹配纯粹的确认语：「可以改成骑车吗」里也含「可以」，若做包含匹配会把
    调整诉求误判成结束对话。故先要求整句足够短，再做关键词匹配。
    """
    if not text:
        return True
    lowered = text.lower()
    if len(text) > 12:
        return False
    return any(kw in lowered for kw in _CONTINUE_KEYWORDS)


def _review_question(turn: int) -> str:
    """按轮次生成问询话术：首轮说明用法，后续轮次简短追问。"""
    if turn <= 1:
        return (
            "方案已生成。您对今天的运动方案有什么疑问吗？"
            "可以直接提问，"
            "也可以要求调整，"
            "或回复「继续」直接完成方案编排。"
        )
    return f"还有其他疑问吗？（第 {turn} 轮，可继续提问，或回复「继续」完成编排）"


def _answer_question(
    state: SportAgentState, user_input: str, history: list[dict[str, Any]], writer: Any
) -> str:
    """基于方案上下文与本次会话的问答历史，流式回答用户问题。

    把历史问答一并送进 prompt：多轮对话里用户会说「那改成 20 分钟呢」这类
    依赖上文的话，只给当前这句 LLM 无法理解指代。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from sport_agent.utils import format_profile, get_llm

    llm = get_llm()
    system = SystemMessage(
        content=(
            "你是运动教练。用户针对今日运动方案提出了疑问，请基于方案详情和专业知识回答。"
            "回答简洁专业，100-200字。如果用户的建议合理且安全，可以表示采纳并在后续编排中考虑。"
            "结合此前的对话理解用户的指代，不要重复已经说过的内容。"
            "用户提问的生理指标数据（HRV/心率/血氧/压力等）在下方上下文中给出，"
            "必须引用实际数值回答，不得声称数据未生成或设备初始化中。"
        )
    )

    # 只带最近 6 轮：更早的内容对当前追问几乎无参考价值，却会线性拉长 prompt
    recent = history[-6:]
    context = "\n".join(f"用户：{qa.get('question', '')}\n教练：{qa.get('answer', '')}" for qa in recent)

    # 组装完整上下文：方案 + 生理数据 + 身体电量 + 用户画像
    vitals = state.get("vitals", {})
    body_energy = state.get("body_energy", 50)
    body_energy_factors = state.get("body_energy_factors", {})
    recovery_status = state.get("recovery_status", "")
    body_energy_interp = state.get("body_energy_interpretation", "")
    profile_text = format_profile(state.get("user_profile"))

    # 生理数据摘要
    vital_lines = []
    for key, label, unit in (
        ("heart_rate", "最新心率", "bpm"),
        ("resting_hr", "静息心率", "bpm"),
        ("hrv", "HRV", "ms"),
        ("spo2", "血氧", "%"),
        ("stress", "压力值", ""),
    ):
        v = vitals.get(key)
        if v is not None:
            vital_lines.append(f"{label} {v}{unit}")
    vitals_text = "，".join(vital_lines) if vital_lines else "无"

    # 身体电量详情
    energy_text = f"{body_energy}/100（{recovery_status}）"
    if body_energy_factors:
        factor_parts = [f"{k}={v:.0f}" for k, v in body_energy_factors.items() if isinstance(v, (int, float))]
        if factor_parts:
            energy_text += f"，分量：{', '.join(factor_parts)}"
    if body_energy_interp:
        energy_text += f"\n电量解读：{body_energy_interp}"

    human = HumanMessage(
        content=(
            (f"此前对话：\n{context}\n\n" if context else "")
            + f"用户提问：{user_input}\n\n"
            f"=== 当前运动方案 ===\n{state.get('plan', {})}\n\n"
            f"=== 方案分析 ===\n{state.get('plan_analysis_stream', '')}\n\n"
            f"=== 当前生理数据 ===\n{vitals_text}\n\n"
            f"=== 身体电量 ===\n{energy_text}\n"
            f"风险等级：{state.get('risk_level', 'low')}\n\n"
            f"=== 用户画像 ===\n{profile_text}\n"
        )
    )

    parts: list[str] = []
    for chunk in llm.stream([system, human]):
        token = chunk.content
        if not isinstance(token, str):
            token = str(token)
        if token:
            parts.append(token)
            writer({"node": "plan_review", "type": "token", "content": token})

    answer = "".join(parts)
    logger.info(
        "plan_review.answer_content",
        user_id=state.get("user_id"),
        question=user_input,
        answer_chars=len(answer),
        answer=answer[:500],
    )
    return answer


def plan_review_node(state: SportAgentState) -> dict[str, Any]:
    """F008.5 方案问询（interrupt 打断）：支持多轮对话，最多 MAX_DIALOG_TURNS 轮。

    每次进入本节点处理一轮：
    1. interrupt 暂停，等用户输入
    2. 用户说「继续」或已达轮次上限 → 结束问询，进入 F009 动作编排
    3. 否则 LLM 流式回答，next_action="ask_again" 让条件路由把流程绕回本节点，
       形成多轮对话

    一轮一次节点执行（而不是在节点内 while 循环 interrupt）：LangGraph 的 resume
    会从节点开头重新执行，循环内的历史轮次会被重放，LLM 回答也会重复调用。
    """
    writer = get_stream_writer()
    turn = state.get("dialog_turns", 0) + 1
    history = list(state.get("plan_review_qa") or [])

    user_input = (interrupt({"question": _review_question(turn), "field": "plan_review", "turn": turn}) or "").strip()

    if _is_continue(user_input):
        writer({"node": "plan_review", "type": "text", "content": "好的，继续为您编排动作方案。"})
        logger.info("plan_review.finish", user_id=state.get("user_id"), turns=turn - 1)
        return {"next_action": "continue", "dialog_turns": turn - 1, "stage": "plan_review"}

    answer = ""
    if get_settings().use_llm:
        try:
            answer = _answer_question(state, user_input, history, writer)
        except Exception as e:  # noqa: BLE001
            logger.warning("plan_review.llm_failed", error=str(e))
            writer({
                "node": "plan_review",
                "type": "text",
                "content": "抱歉，暂时无法回答这个问题，您可以换个说法再问，或回复「继续」。",
            })
    else:
        writer({
            "node": "plan_review",
            "type": "text",
            "content": f"已收到您的反馈：「{user_input}」。还有其他问题吗？或回复「继续」。",
        })

    # 达上限：本轮照常回答，但不再追问，直接推进流程
    limit_reached = turn >= MAX_DIALOG_TURNS
    if limit_reached:
        writer({
            "node": "plan_review",
            "type": "text",
            "content": f"本次对话已达 {MAX_DIALOG_TURNS} 轮上限，接下来为您完成方案编排。",
        })

    logger.info(
        "plan_review.answered",
        user_id=state.get("user_id"),
        turn=turn,
        question=user_input[:50],
        chars=len(answer),
        limit_reached=limit_reached,
    )
    return {
        # 覆盖式写回完整历史：plan_review_qa 无 reducer，返回列表即替换，
        # 故这里带上此前轮次，否则多轮只会剩最后一条
        "plan_review_qa": history + [{"question": user_input, "answer": answer, "turn": turn}],
        "plan_edit_request": _edit_request_of(user_input),
        "dialog_turns": turn,
        "dialog_limit_reached": limit_reached,
        "next_action": "continue" if limit_reached else "ask_again",
        "stage": "plan_review",
    }


def route_after_plan_review(state: SportAgentState) -> str:
    """条件路由：用户还有问题就绕回问询节点，否则进入动作编排。"""
    if state.get("next_action") == "ask_again":
        return "plan_review"
    return "plan_compose"


def _compose_actions(plan: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """按运动类型取动作库，并把主运动总时长对齐到 plan.duration。

    动作库里的主运动时长是按 30 分钟写死的默认值。若不对齐，用户把时长改成
    20 分钟后，卡片上会出现「20 分钟方案 + 30 分钟动作清单」的自相矛盾。
    热身/拉伸是固定时长的标准流程，不随总时长缩放。
    """
    sport_type = plan.get("type", "running")
    warmup = search_exercise_library.invoke({"sport_type": sport_type, "phase": "warmup"})
    main = search_exercise_library.invoke({"sport_type": sport_type, "phase": "main"})
    cooldown = search_exercise_library.invoke({"sport_type": sport_type, "phase": "cooldown"})

    target_s = int(plan.get("duration") or 30) * 60
    total_s = sum(int(a.get("duration") or 0) for a in main)
    if total_s > 0 and target_s > 0 and total_s != target_s:
        # 多个主运动动作时按原时长占比分配，保证总和等于目标时长
        main = [
            {**a, "duration": max(60, round(int(a.get("duration") or 0) * target_s / total_s))}
            for a in main
        ]

    return {"warmup": warmup, "main": main, "cooldown": cooldown}


def plan_compose_node(state: SportAgentState) -> dict[str, Any]:
    """F009 方案智能编排：热身/主运动/拉伸动作与运动类型匹配。

    需求9：热身方案、拉伸方案与主运动类型匹配，图文匹配度 100%。
    """
    with measure("plan_compose") as t:
        plan = state.get("plan", {})
        sport_type = plan.get("type", "running")
        actions = _compose_actions(plan)
        warmup, cooldown = actions["warmup"], actions["cooldown"]

    logger.info(
        "plan_compose.done",
        user_id=state.get("user_id"),
        sport_type=sport_type,
        warmup=len(warmup),
        cooldown=len(cooldown),
        ms=round(t.ms, 1),
    )
    return {"plan_actions": actions, "stage": "plan_compose", "node_timings": t.result}


# ===== F010 指令库（需求10：明确/模糊/错误三类指令分别处理）=====

# 查看详情意图
_VIEW_KEYWORDS = ("查看", "详情", "详细", "看看", "念一遍", "什么方案", "detail", "show")

# 强度/时长下调（模糊）
_DECREASE_KEYWORDS = ("少", "降", "减", "轻", "短", "累", "慢", "缓", "低", "小", "easy", "less", "lower")

# 强度/时长上调（模糊）
# 注意：不能放裸的「强」。"强度再低一点"里也含「强」，会把下调误判成上调，
# 故上调只匹配明确表达增量的词组。
_INCREASE_KEYWORDS = ("多", "加", "增", "久", "快", "高一点", "强一点", "强些", "more", "harder")

# 明确的强度档位
_EXPLICIT_INTENSITY = {"低强度": "low", "中强度": "mid", "高强度": "high"}

# 疑问句标记：命中则不按模糊方向指令处理（见 _classify_edit）
_QUESTION_MARKERS = ("？", "?", "吗", "呢", "会不会", "为什么", "为啥", "是不是")

# 运动类型变更（明确指令）
_TYPE_ALIASES: dict[str, str] = {
    "骑车": "cycling",
    "骑行": "cycling",
    "单车": "cycling",
    "自行车": "cycling",
    "跑步": "running",
    "慢跑": "running",
    "散步": "walk",
    "走路": "walk",
    "快走": "walk",
    "游泳": "swimming",
    "跳绳": "jump_rope",
    "室内": "indoor_bike",
    "动感单车": "indoor_bike",
    "跑步机": "treadmill",
}

# 单次运动时长安全上限（分钟），超出视为错误指令不执行
MAX_SAFE_DURATION = 180

# 需求10：不超过 2 次重复对话
MAX_EDIT_ATTEMPTS = 2


def _parse_duration_minutes(text: str) -> Optional[int]:
    """从指令中解析明确时长（分钟）。"""
    m = re.search(r"(\d+)\s*(?:个)?\s*(?:小时|h|hour)", text, re.IGNORECASE)
    if m:
        return int(m.group(1)) * 60
    m = re.search(r"(\d+)\s*(?:分钟|分|min)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _classify_edit(text: str) -> dict[str, Any]:
    """需求10：将用户指令分为 view / explicit / vague / invalid 四类。

    Returns:
        {"intent", "changes", "reason"}；intent=invalid 时 changes 为空（不执行）。
    """
    if any(kw in text for kw in _VIEW_KEYWORDS):
        return {"intent": "view", "changes": {}, "reason": "查看方案详情"}

    changes: dict[str, Any] = {}
    # 疑问句只是询问，不是调整诉求。若不排除，"强度会不会太大？"会因命中
    # _INCREASE_KEYWORDS 的"强"被误判为要求提高强度。明确指令（含具体运动类型/
    # 时长/强度档位）不受影响，仍在下方按 explicit 处理。
    is_question = any(kw in text for kw in _QUESTION_MARKERS)
    explicit = False

    # 明确：指定运动类型
    for alias, sport in _TYPE_ALIASES.items():
        if alias in text:
            changes["type"] = sport
            explicit = True
            break

    # 明确：指定时长
    duration = _parse_duration_minutes(text)
    if duration is not None:
        if duration <= 0 or duration > MAX_SAFE_DURATION:
            return {
                "intent": "invalid",
                "changes": {},
                "reason": f"{duration} 分钟超出单次运动安全范围（5-{MAX_SAFE_DURATION} 分钟），已保留原方案",
            }
        changes["duration"] = max(5, duration)
        explicit = True

    # 明确：指定强度档位
    for phrase, level in _EXPLICIT_INTENSITY.items():
        if phrase in text:
            changes["intensity"] = level
            explicit = True
            break

    if explicit:
        return {"intent": "explicit", "changes": changes, "reason": "已按你的要求调整方案"}

    # 模糊：只表达了方向。疑问句不做方向推断，避免把提问当成调整指令。
    if not is_question:
        if any(kw in text for kw in _DECREASE_KEYWORDS):
            return {"intent": "vague", "changes": {"direction": "down"}, "reason": "已适当降低强度与时长"}
        if any(kw in text for kw in _INCREASE_KEYWORDS):
            return {"intent": "vague", "changes": {"direction": "up"}, "reason": "已适当提高强度与时长"}

    # 错误/无关指令：不执行
    return {"intent": "invalid", "changes": {}, "reason": "没有识别到可执行的方案调整指令"}


def _edit_request_of(text: str) -> str:
    """判断 plan_review 中的用户输入是否是需要 F010 执行的调整诉求。

    plan_review 负责「答疑」，plan_edit 负责「改方案」。纯提问已由 LLM 回答，
    不应再进入 plan_edit 占用调整轮次，故只透传可执行的指令。
    """
    return text if _classify_edit(text)["intent"] != "invalid" else ""


def _apply_direction(plan: dict[str, Any], direction: str) -> dict[str, Any]:
    """模糊指令的保守调整：时长 ±10 分钟，强度升/降一档。"""
    ladder = ["low", "mid", "high"]
    zone_map = {"low": "Zone1", "mid": "Zone2", "high": "Zone3"}
    current = plan.get("intensity", "mid")
    idx = ladder.index(current) if current in ladder else 1

    if direction == "down":
        idx = max(0, idx - 1)
        duration = max(10, plan.get("duration", 30) - 10)
    else:
        idx = min(len(ladder) - 1, idx + 1)
        duration = min(MAX_SAFE_DURATION, plan.get("duration", 30) + 10)

    return {**plan, "intensity": ladder[idx], "zone": zone_map[ladder[idx]], "duration": duration}


def plan_edit_node(state: SportAgentState) -> dict[str, Any]:
    """F010 方案详情及变更：语音/按键指令查看与调整方案。

    需求10 验收标准：
    - 明确指令（「改成骑车」「跑30分钟」「低强度」）→ 识别并正确执行。
    - 模糊指令（「有点累」「轻松点」）→ 识别并保守调整（±10min，强度降一档）。
    - 错误指令（无关内容、超安全范围）→ 不执行并提示用户。
    - 重复对话不超过 2 次：达到上限后保留原方案并给出兜底提示。
    - 安全约束：风险等级 high/forbidden 时拒绝任何上调请求。

    性能要求：响应 <1s → 全流程纯规则，不调用 LLM。
    """
    # plan_review 中提出的调整诉求优先；其次是启动时带入的自由文本
    user_msg = (state.get("plan_edit_request") or state.get("user_message") or "").strip()
    plan = state.get("plan", {})
    attempts = state.get("plan_edit_attempts", 0)

    # 无指令 → 透传，不计入对话轮次
    if not user_msg:
        return {"plan": plan, "plan_adjusted": False, "stage": "plan_edit"}

    with measure("plan_edit") as t:
        # 需求10：不超过 2 次重复对话
        if attempts >= MAX_EDIT_ATTEMPTS:
            feedback = "已经调整过 2 次了，本次先按当前方案开始，运动中可以随时再和我说。"
            logger.info("plan_edit.max_attempts", user_id=state.get("user_id"), attempts=attempts)
            return {
                "plan": plan,
                "plan_adjusted": False,
                "plan_edit_feedback": feedback,
                "intent": "max_attempts",
                "stage": "plan_edit",
                "node_timings": t.result,
            }

        result = _classify_edit(user_msg)
        intent = result["intent"]
        changes = result["changes"]
        feedback = result["reason"]
        adjusted = False
        risk_level = state.get("risk_level", "low")

        if intent == "view":
            actions = state.get("plan_actions", {})
            feedback = (
                f"今日方案：{plan.get('type', 'running')} {plan.get('duration', 30)} 分钟，"
                f"强度 {plan.get('intensity', 'mid')}（{plan.get('zone', 'Zone2')}）。"
                f"热身 {len(actions.get('warmup', []))} 个动作，"
                f"主运动 {len(actions.get('main', []))} 项，拉伸 {len(actions.get('cooldown', []))} 个动作。"
            )

        elif intent == "explicit":
            # 安全约束：高风险状态拒绝上调强度
            if risk_level in ("high", "forbidden") and changes.get("intensity") == "high":
                intent = "invalid"
                feedback = "当前生理指标异常，不能提高运动强度。已保留低强度方案，建议先完成健康检查。"
            else:
                plan = {**plan, **changes, "rationale": f"用户指令调整：{user_msg}"}
                adjusted = True

        elif intent == "vague":
            direction = changes.get("direction", "down")
            if risk_level in ("high", "forbidden") and direction == "up":
                intent = "invalid"
                feedback = "当前生理指标异常，不建议提高强度。已保留原方案。"
            else:
                plan = _apply_direction(plan, direction)
                plan["rationale"] = f"用户模糊指令调整（{direction}）：{user_msg}"
                adjusted = True

        if intent == "invalid":
            remaining = MAX_EDIT_ATTEMPTS - attempts - 1
            if remaining > 0:
                feedback = (
                    f"{feedback}。你可以这样说：「改成骑车」「跑 30 分钟」「换成低强度」，"
                    f"或回复「就这样」直接开始。"
                )

    logger.info(
        "plan_edit.done",
        user_id=state.get("user_id"),
        intent=intent,
        adjusted=adjusted,
        attempts=attempts + 1,
        ms=round(t.ms, 1),
    )
    updates: dict[str, Any] = {
        "plan": plan,
        "plan_adjusted": adjusted,
        "plan_edit_attempts": attempts + 1,
        "plan_edit_feedback": feedback,
        "intent": intent,  # 需求10备注：对指令进行标记并记忆
        "plan_edit_request": "",  # 消费后清空，避免重入时重复应用同一指令
        "stage": "plan_edit",
        "node_timings": t.result,
    }
    if adjusted:
        # plan_compose 在本节点之前执行，动作清单是按调整前的方案编排的。
        # 改了运动类型或时长后必须重新编排，否则会出现「20 分钟方案配 30 分钟
        # 动作」或「改成骑车了却还列着跑步动作」。
        updates["plan_actions"] = _compose_actions(plan)
    return updates
