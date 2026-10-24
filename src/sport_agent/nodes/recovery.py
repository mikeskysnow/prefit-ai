"""恢复域节点：F004 身体电量（双层架构）、F005 用户形象（P2 占位）。

双层架构（对应文档《身体电量方案.md》）：
    Layer 1 ▸ 电量计算引擎（纯 Python，<1s，100% 可复现）
         HRV4Training 启发的加权模型：HRV(35%) + CV(25%) + Sleep(20%) + LoadRecovery(20%)
         个人基线（30天滑动窗口），不足 7 天用群体基线（置信度 low）
    Layer 2 ▸ LLM 解读（千问）
         解读分数含义 + 个性化建议 + 主观校准引导
         ★ LLM 不做数值计算，仅负责"说明白"

对应 Agent: Recovery Agent / Profile Agent。
"""

from __future__ import annotations

from typing import Any

from sport_agent.algorithms.body_battery import compute_body_battery
from sport_agent.state import SportAgentState
from sport_agent.tools.health_tools import get_hrv_history, get_training_history
from sport_agent.utils import get_logger, measure

logger = get_logger(__name__)

# 校准基线所需最小天数
BASELINE_REQUIRED_DAYS = 7


def recovery_node(state: SportAgentState) -> dict[str, Any]:
    """F004 身体电量入口：度量耗时后委托 _recovery_impl。

    性能要求：基线建立时间 <10s（需求4）。
    """
    with measure("recovery") as t:
        updates = _recovery_impl(state)
    updates["node_timings"] = t.result
    return updates


def _recovery_impl(state: SportAgentState) -> dict[str, Any]:
    """F004 身体电量：Layer 1 算法计算 + Layer 2 LLM 解读。

    需求4：区分首次/非首次使用；校准基线至少需 7 天运动数据。
    """
    user_id = state["user_id"]
    vitals = state.get("vitals", {})

    hrv_history = get_hrv_history.invoke({"user_id": user_id, "days": 30})
    training_history = get_training_history.invoke({"user_id": user_id, "days": 14})

    # 自定义数据中的睡眠评分（0-100），未提供时为 None（算法用群体默认值推断）
    sleep_score = vitals.get("sleep_score")

    # ===== Layer 1: 确定性算法计算 =====
    result = compute_body_battery(
        vitals, hrv_history, training_history, sleep_score=sleep_score,
    )

    energy = result["score"]
    status = result["status"]
    baseline_ready = result["confidence"] != "low"

    logger.info(
        "recovery.algorithm.done",
        user_id=user_id,
        energy=energy,
        status=status,
        level=result["level"],
        confidence=result["confidence"],
        factors=result["factors"],
    )

    # 新用户初始化模式
    if not baseline_ready:
        logger.info("recovery.initializing", user_id=user_id, days=result["baseline"]["days"])
        return {
            "body_energy": energy,
            "recovery_status": "INITIALIZING",
            "baseline_ready": False,
            "body_energy_factors": result["factors"],
            "body_energy_confidence": result["confidence"],
            "body_energy_interpretation": "",
            "stage": "recovery",
        }

    # ===== Layer 2: LLM 解读 =====
    interpretation = _llm_interpret(state, result)

    logger.info("recovery.done", user_id=user_id, energy=energy, status=status, llm=bool(interpretation))
    return {
        "body_energy": energy,
        "recovery_status": status,
        "baseline_ready": True,
        "body_energy_factors": result["factors"],
        "body_energy_confidence": result["confidence"],
        "body_energy_interpretation": interpretation,
        "stage": "recovery",
    }


def _llm_interpret(state: dict[str, Any], result: dict[str, Any]) -> str:
    """Layer 2: LLM 解读身体电量分数与各分量含义。

    若 LLM 不可用返回空字符串，调用方仅展示算法分数。
    """
    from sport_agent.config import get_settings

    if not get_settings().use_llm:
        return ""

    try:
        from sport_agent.agents import BODY_BATTERY_INTERPRET_PROMPT
        from sport_agent.memory import MEM_HISTORY, MEM_PHYSIO, format_memories
        from sport_agent.utils import format_profile, get_llm

        profile = format_profile(state.get("user_profile"))
        llm = get_llm()
        chain = BODY_BATTERY_INTERPRET_PROMPT | llm

        msg = chain.invoke(
            {
                "score": result["score"],
                "level": result["level"],
                "status": result["status"],
                "confidence": result["confidence"],
                "factors": result["factors"],
                "baseline": result["baseline"],
                "profile": profile,
                # 生理基线记忆：让解读能对比"上次 vs 本次"，而不是只描述当前分数
                "memories": format_memories(state.get("memory"), types=(MEM_PHYSIO, MEM_HISTORY)),
            }
        )
        text = msg.content if hasattr(msg, "content") else str(msg)
        logger.info("recovery.llm.interpreted", chars=len(text))
        return text
    except Exception as e:  # noqa: BLE001
        logger.warning("recovery.llm.failed", error=str(e))
        return ""


# ===== 向后兼容：plan.py 引用此函数 =====


def _consecutive_training_days(history: list[dict[str, Any]]) -> int:
    """统计连续运动天数（用于「连续 3 天自动降低强度」规则）。

    委托给 algorithms.body_battery 模块。
    """
    from sport_agent.algorithms.body_battery import (
        _consecutive_training_days as _impl,
    )

    return _impl(history)


def avatar_node(state: SportAgentState) -> dict[str, Any]:
    """F005 用户运动形象模拟。

    需求5：男女各一个固定形象；身材比例随 BMI 变化；后续更新 <30s。
    """
    with measure("avatar") as t:
        profile = state.get("user_profile", {})
        bmi = profile.get("bmi") or 22.0
        gender = profile.get("gender", "male")

        avatar = {
            "gender": gender,
            # 需求5：男女各一个固定形象
            "template_id": "avatar_male_01" if gender == "male" else "avatar_female_01",
            "bmi": bmi,
            "body_ratio": _bmi_to_ratio(bmi),
            "style": state.get("plan", {}).get("type") or "running",
        }

    logger.info("avatar.rendered", user_id=state.get("user_id"), bmi=bmi, template=avatar["template_id"], ms=round(t.ms, 1))
    return {"avatar": avatar, "stage": "avatar", "node_timings": t.result}


def _bmi_to_ratio(bmi: float) -> str:
    if bmi < 18.5:
        return "slim"
    if bmi < 24:
        return "standard"
    if bmi < 28:
        return "muscular"
    return "stocky"
