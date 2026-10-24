"""问询域节点：F006 运动目标智能问询（支持 interrupt 打断）。

对应 Agent: Conversation Agent。

需求要点（docs/产品需求文档.md 需求6）：
- 首次使用：提问运动目标，支持打断；
  对答非所问的回答多次询问，收到符合问题的答案后继续提问，直至收集完成。
- 非首次使用：判断是否在常用运动地点；非常用地需问清目的与运动情景，支持跳过。
- 支持跳过；支持打断，且打断后之前的语音不可继续播报（tts_cancel）。
- 支持模糊指令识别，识别率 >90%；支持按用户长期记忆。
- 运动时长和强度不再询问，改由用户画像中的「经常运动时长」推导（见 health_collect）。

实现：
- 使用 LangGraph interrupt() 暂停图执行，等待用户输入后用 Command(resume=...) 恢复。
- 每次 interrupt 携带 tts_cancel=True，客户端收到即停止当前播报（需求6）。
- 关键词 + 模糊兜底解析；无法归类视为"答非所问"，重问至多 MAX_REASK 次。
"""

from __future__ import annotations

from typing import Any, Optional

from langgraph.types import interrupt

from sport_agent.state import SportAgentState
from sport_agent.utils import get_logger

logger = get_logger(__name__)

# 答非所问时的最大重问次数（超出后取默认值继续，避免卡死流程）
MAX_REASK = 2

# 跳过意图关键词
SKIP_KEYWORDS: tuple[str, ...] = ("跳过", "不用问", "随便", "都行", "无所谓", "skip", "next")

GOAL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "fat_loss": ("减脂", "减肥", "瘦", "掉秤", "刷脂", "fat", "lose"),
    "cardio": ("有氧", "心肺", "cardio"),
    "endurance": ("耐力", "持久", "长距离", "endurance"),
    "muscle": ("增肌", "力量", "肌肉", "muscle", "strength"),
    "health": ("健康", "养生", "保持", "放松", "health"),
}


def _is_skip(text: str) -> bool:
    return bool(text) and any(kw in text.lower() for kw in SKIP_KEYWORDS)


def _parse_goal(text: str) -> Optional[str]:
    """解析运动目标；无法归类返回 None（视为答非所问，触发重问）。"""
    if not text:
        return None
    lowered = text.lower()
    for goal, keywords in GOAL_KEYWORDS.items():
        if any(kw.lower() in lowered for kw in keywords):
            return goal
    # 模糊兜底，如「最近想轻一点」→ fat_loss
    if "轻" in text:
        return "fat_loss"
    if any(kw in text for kw in ("跑", "走", "骑")):
        return "cardio"
    return None


# 首次使用的顺序问询表：(字段, 问题, 解析器, 默认值)
# 运动时长和强度不再询问：时长由画像「经常运动时长」推导，强度由风险等级和恢复状态决定
_QUESTION_FLOW: tuple[tuple[str, str, Any, Any], ...] = (
    (
        "goal",
        "你今天的运动目的是？可以回复：减脂 / 有氧 / 耐力 / 增肌 / 健康（回复「跳过」可跳过）",
        _parse_goal,
        "health",
    ),
)


def _ask(field: str, question: str, attempt: int) -> str:
    """发起一次 interrupt 问询。

    携带 tts_cancel=True：需求6「打断后之前的语音不可继续播报」，
    客户端收到该标志立即停止当前 TTS 播报队列。
    """
    answer = interrupt(
        {
            "question": question,
            "field": field,
            "attempt": attempt,
            "tts_cancel": True,
        }
    )
    return (answer or "").strip() if isinstance(answer, str) else str(answer or "").strip()


def route_after_avatar(state: SportAgentState) -> str:
    """条件路由：新用户 → interview 问询；老用户 → 直接出方案。

    老用户已在 health_collect 阶段从 user_state.preferences 载入历史偏好，
    缺失字段由 plan_generate 的默认值兜底，无需再问。

    例外：老用户在非常用地点时仍进 interview —— 需求6 要求「非常用地需问清
    目的与运动情景」，此时问询节点只问场景一个问题，不会退回完整三连问。
    """
    if state.get("is_first_time", True):
        logger.info("route_after_avatar.interview", user_id=state.get("user_id"), reason="new_user")
        return "interview"

    location = state.get("location", {}) or {}
    if not location.get("is_known_place") and location:
        logger.info("route_after_avatar.interview", user_id=state.get("user_id"), reason="unknown_place")
        return "interview"

    logger.info("route_after_avatar.skip_interview", user_id=state.get("user_id"), reason="returning_user")
    return "plan_generate"


def _persist_preferences(user_id: str, collected: dict[str, Any]) -> None:
    """把问询到的偏好写入 user_state.preferences（长期记忆）。

    只存问询表里的 goal 字段，scene 是本次一次性的场景描述不入长期记忆。
    DB 不可用时静默跳过 —— 长期记忆是增强项，不能因为写库失败阻断问询流程。
    """
    if not user_id:
        return
    prefs = {k: collected[k] for k in ("goal",) if collected.get(k) is not None}
    if not prefs:
        return
    try:
        from sport_agent.db import save_preferences

        save_preferences(user_id, prefs)
    except Exception as e:  # noqa: BLE001
        logger.warning("interview.save_preferences_failed", error=str(e))


def _prefill_from_memory(state: SportAgentState) -> dict[str, Any]:
    """用长期记忆里的偏好预填问询字段，减少重复提问（需求6）。

    mem0 存的是自然语言记忆文本，这里复用已有的关键词解析器把它还原成结构化字段。
    只填 state 里缺的字段，本次显式传入的值优先。预填后对应问题会被跳过。
    """
    from sport_agent.memory import MEM_PREFERENCE

    texts = (state.get("memory") or {}).get(MEM_PREFERENCE) or []
    if not texts:
        return {}
    blob = " ".join(str(t) for t in texts)
    prefilled: dict[str, Any] = {}
    for field, parser in (("goal", _parse_goal),):
        if state.get(field):
            continue
        parsed = parser(blob)
        if parsed is not None:
            prefilled[field] = parsed
    if prefilled:
        logger.info("interview.prefill_from_memory", user_id=state.get("user_id"), fields=prefilled)
    return prefilled


def interview_node(state: SportAgentState) -> dict[str, Any]:
    """F006 运动目的问询。

    - 新用户（首次使用）→ 按 _QUESTION_FLOW 顺序提问，答非所问重问至多 MAX_REASK 次。
    - 老用户 → 由 graph 的 route_after_avatar 直接跳到 plan_generate，不进本节点。
      （仅在老用户处于非常用地点时才会进来问场景）
    """
    # 已完成问询，幂等返回
    if state.get("inquiry_completed"):
        return {}

    location = state.get("location", {}) or {}
    is_first_time = state.get("is_first_time", True)
    is_known_place = bool(location.get("is_known_place"))

    # 非首次 + 常用地点 + 已有目标 → 复用长期记忆，跳过问询
    if not is_first_time and is_known_place and state.get("goal"):
        logger.info("interview.skip", user_id=state.get("user_id"), reason="known_place_with_memory")
        return {
            "inquiry_completed": True,
            "inquiry_qa": [],
            "stage": "interview",
        }

    qa: list[dict[str, Any]] = []
    reask_count: dict[str, int] = {}
    collected: dict[str, Any] = _prefill_from_memory(state)

    # 非首次 + 非常用地点：先问清本次目的与运动情景（需求6）
    if not is_first_time and not is_known_place:
        place = location.get("name", "当前位置")
        scene_answer = _ask(
            "scene",
            f"注意到你在{place}，这不是你常去的运动地点。这次运动的目的和场景是？（回复「跳过」可跳过）",
            1,
        )
        qa.append({"field": "scene", "answer": scene_answer, "skipped": _is_skip(scene_answer)})
        if not _is_skip(scene_answer):
            parsed = _parse_goal(scene_answer)
            if parsed:
                collected["goal"] = parsed
            collected["scene"] = scene_answer
        logger.info("interview.new_place_asked", user_id=state.get("user_id"), place=place)

    # 顺序问询（首次使用；非首次缺字段时同样适用，可采用首次问询内容）
    for field, question, parser, default in _QUESTION_FLOW:
        if collected.get(field) or state.get(field):
            continue

        value = None
        skipped = False
        for attempt in range(1, MAX_REASK + 2):
            prompt = question if attempt == 1 else f"抱歉，我没太理解。{question}"
            answer = _ask(field, prompt, attempt)

            if _is_skip(answer):
                skipped = True
                qa.append({"field": field, "answer": answer, "attempt": attempt, "skipped": True})
                break

            value = parser(answer)
            qa.append(
                {
                    "field": field,
                    "answer": answer,
                    "attempt": attempt,
                    "recognized": value is not None,
                }
            )
            if value is not None:
                break
            # 答非所问 → 记录并重问
            reask_count[field] = reask_count.get(field, 0) + 1
            logger.info("interview.reask", user_id=state.get("user_id"), field=field, attempt=attempt, raw=answer[:40])

        collected[field] = value if value is not None else default
        if skipped or value is None:
            logger.info("interview.default_applied", field=field, value=collected[field], skipped=skipped)

    # 问询结果写入长期记忆，下次作为老用户可直接复用（需求6「支持按用户长期记忆」）
    _persist_preferences(state.get("user_id", ""), collected)

    logger.info(
        "interview.done",
        user_id=state.get("user_id"),
        is_first_time=is_first_time,
        goal=collected.get("goal"),
        reasks=sum(reask_count.values()),
    )

    return {
        **collected,
        "inquiry_qa": qa,
        "inquiry_reask_count": reask_count,
        "inquiry_completed": True,
        "interrupted": False,
        "tts_cancel": False,  # 问询结束，恢复正常播报
        "stage": "interview",
    }
