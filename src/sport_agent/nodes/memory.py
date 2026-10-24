"""记忆节点：运动前快照读取（入口）与记忆写入（出口）。

对应 docs/记忆系统方案.md 的链路：
    START →【Mem0读取运动前快照】→ 健康评估 → 风险预警 → 体能基线
          → 用户问询 → 方案生成 →【Mem0异步写入记忆】→ END

分工：
- memory_load_node   同步读取，走 get_all + metadata 过滤，实测 1~2ms，不调 embedding。
- memory_write_node  只提交后台任务后立即返回，本身耗时 <1ms。
  四层记忆全部走 write_fact_async（infer=False + 业务幂等键），数值不被 LLM 改写。
"""

from __future__ import annotations

from typing import Any

from sport_agent.memory import (
    MEM_HISTORY,
    MEM_PHYSIO,
    MEM_PRE_SPORT,
    MEM_PREFERENCE,
    read_snapshot,
    write_fact_async,
)
from sport_agent.state import SportAgentState
from sport_agent.utils import get_logger, measure

logger = get_logger(__name__)


def memory_load_node(state: SportAgentState) -> dict[str, Any]:
    """读取运动前记忆快照，注入 State 供后续所有评估节点复用。

    只读一次、全链路共享：避免各节点重复查询，也保证各节点看到的记忆一致。
    记忆层不可用时返回空快照并置 memory_ready=False，主流程照常运行。
    """
    user_id = state.get("user_id", "")
    with measure("memory_load") as t:
        snapshot = read_snapshot(user_id)
    total = sum(len(v) for v in snapshot.values())
    logger.info("memory_load.done", user_id=user_id, total=total)
    return {
        "memory": snapshot,
        "memory_ready": total > 0,
        "node_timings": t.result,
    }


def _session_day(state: SportAgentState) -> str:
    """本次会话的归属日期（YYYY-MM-DD），用于构造按天幂等的记忆键。

    优先用实际测量日期，让同一天重复跑覆盖同一条而不是各写一条。
    """
    vitals = state.get("vitals") or {}
    date = vitals.get("metric_date") or (state.get("vitals_fetch_ts") or "")[:10]
    if date:
        return str(date)[:10]
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).date().isoformat()


def _physio_fact(state: SportAgentState) -> str:
    """把本次生理指标与体能结论拼成一条事实文本（不经 LLM，防数值幻觉）。"""
    vitals = state.get("vitals") or {}
    parts: list[str] = []
    date = vitals.get("metric_date") or (state.get("vitals_fetch_ts") or "")[:10]
    if date:
        parts.append(f"{date} 实测")
    for key, label, unit in (
        ("heart_rate", "静息心率", "bpm"),
        ("hrv", "HRV", "ms"),
        ("spo2", "血氧", "%"),
        ("stress", "压力值", ""),
    ):
        v = vitals.get(key)
        if v:
            parts.append(f"{label} {v}{unit}")
    energy = state.get("body_energy")
    if energy is not None:
        parts.append(f"身体电量 {energy}（{state.get('recovery_status', '')}）")
    return "，".join(parts) if len(parts) > 1 else ""


def _assessment_fact(state: SportAgentState) -> str:
    """本次运动前评估结论摘要（风险等级 + 原因），作为历史风险档案。"""
    level = state.get("risk_level")
    if not level:
        return ""
    reasons = state.get("risk_reason") or []
    text = f"运动前风险评估结论：{level}"
    if reasons:
        text += f"，原因：{'、'.join(str(r) for r in reasons[:3])}"
    return text


def _plan_fact(state: SportAgentState) -> str:
    """本次运动计划摘要，供下次生成方案时参考历史采纳情况。"""
    plan = state.get("plan") or {}
    sport = plan.get("type")
    if not sport:
        return ""
    parts = [f"运动方案：{sport}"]
    duration = plan.get("duration") or state.get("habitual_duration")
    if duration:
        parts.append(f"时长 {duration} 分钟")
    intensity = plan.get("intensity") or state.get("intensity_pref")
    if intensity:
        parts.append(f"强度 {intensity}")
    if plan.get("zone"):
        parts.append(f"心率区间 {plan['zone']}")
    if state.get("goal"):
        parts.append(f"目标 {state['goal']}")
    if state.get("plan_adjusted"):
        parts.append("用户调整过该方案")
    return "，".join(parts)


_GOAL_CN = {
    "fat_loss": "减脂",
    "cardio": "有氧",
    "endurance": "耐力提升",
    "muscle": "增肌",
    "health": "健康促进",
}
_INTENSITY_CN = {"low": "低强度", "mid": "中等强度", "high": "高强度"}


def _preference_fact(state: SportAgentState) -> str:
    """把问询收集到的偏好组装成中文偏好句（确定性，不经 LLM 抽取）。

    问询答案已由 interview 的解析器转成结构化字段，这里直接文本化即可。
    不走 mem0 的 LLM 抽取，原因：短答案（如"想减脂"）会被默认抽取 prompt
    判为无价值而丢弃（实测 extracted=0），且确定性组装还省下 11s 的抽取开销。
    """
    parts: list[str] = []
    goal = state.get("goal")
    if goal:
        parts.append(f"运动目标偏好 {_GOAL_CN.get(goal, goal)}")
    t = state.get("habitual_duration")
    if t:
        parts.append(f"经常运动时长约 {t} 分钟")
    intensity = state.get("intensity_pref")
    if intensity:
        parts.append(f"偏好 {_INTENSITY_CN.get(intensity, intensity)}")
    return "用户运动偏好：" + "，".join(parts) if parts else ""


def memory_write_node(state: SportAgentState) -> dict[str, Any]:
    """把本次运动前会话沉淀为长期记忆（全部后台执行）。

    写入四层（全部 infer=False 直写 + 幂等键 upsert）：
    1. physio     本次生理实测 + 身体电量（键含日期，同日重跑覆盖不堆积）
    2. history    风险评估结论 + 计划摘要（键含日期，逐日累积成运动档案）
    3. preference 问询收集到的目标/时长/强度（键固定，始终保留最新一版）
    4. pre_sport  本次场景描述 + 计划调整诉求（键含会话，一次会话一条）

    全部走幂等键：infer=False 不经 mem0 的去重流水线，不加键会每次运行都新增
    重复行（实测两次运行产出两条一模一样的 physio 记忆），既污染快照也拖长 prompt。

    本节点只提交后台任务，自身耗时 <1ms，不影响运动前链路。
    """
    user_id = state.get("user_id", "")
    if not user_id:
        return {"stage": state.get("stage", "end")}

    submitted: list[str] = []
    day = _session_day(state)
    session = state.get("session_id") or day

    physio = _physio_fact(state)
    if physio:
        write_fact_async(user_id, physio, MEM_PHYSIO, {"source": "business"}, key=f"physio:{day}")
        submitted.append(MEM_PHYSIO)

    for name, text in (("assessment", _assessment_fact(state)), ("plan", _plan_fact(state))):
        if text:
            write_fact_async(user_id, text, MEM_HISTORY, {"source": "business"},
                             key=f"{name}:{day}")
            submitted.append(MEM_HISTORY)

    # 偏好：确定性组装后直写 preference 层，下次由问询预填复用。
    # 键固定不带日期 —— 偏好是"当前状态"，新的应覆盖旧的而非并存。
    preference = _preference_fact(state)
    if preference:
        write_fact_async(user_id, preference, MEM_PREFERENCE, {"source": "interview"},
                         key="preference:core")
        submitted.append(MEM_PREFERENCE)

    # 本次一次性的场景描述归入运动前会话档案
    scene = state.get("scene")
    if scene:
        write_fact_async(user_id, f"本次运动场景：{scene}", MEM_PRE_SPORT,
                         {"source": "interview"}, key=f"scene:{session}")
        submitted.append(MEM_PRE_SPORT)

    # 用户主动改计划 = 最强的偏好信号（需求10）：记下原始诉求供下次生成方案参考
    edit_request = state.get("plan_edit_request")
    if edit_request and state.get("plan_adjusted"):
        write_fact_async(user_id, f"用户曾要求调整计划：{edit_request}", MEM_PRE_SPORT,
                         {"source": "plan_edit"}, key=f"plan_edit:{session}")
        submitted.append(MEM_PRE_SPORT)

    logger.info("memory_write.submitted", user_id=user_id, kinds=submitted)
    return {"stage": state.get("stage", "end")}
