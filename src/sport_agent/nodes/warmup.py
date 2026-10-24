"""热身域节点：F012 AI 热身指导。

对应 Agent: Coach Agent。
根据运动计划语音指导热身，每个动作提示要领与开始/倒计时。
性能要求：语音播报延迟 ≤200ms；动作匹配 100%。
"""

from __future__ import annotations

from typing import Any

from sport_agent.state import SportAgentState
from sport_agent.utils import get_logger, measure

logger = get_logger(__name__)


def _countdown_marks(duration: int) -> list[int]:
    """生成倒计时提醒节点（秒，剩余时间）。

    需求12 验收标准2：每个动作提示开始与倒计时。
    """
    marks = [m for m in (30, 10, 5, 3) if m < duration]
    return marks or ([duration] if duration <= 3 else [])


def warmup_node(state: SportAgentState) -> dict[str, Any]:
    """F012 热身指导：根据运动类型匹配热身流程并生成语音脚本。

    需求12 验收标准：
    1. 根据选择的运动类型自动匹配对应的热身流程（由 F009 动作库按 sport_type 检索）。
    2. 每个动作语音提示动作要领 + 开始提醒 + 倒计时提醒时长。

    性能要求：语音播报延迟 ≤200ms → 预生成播报脚本，客户端直接播放不再等生成。
    """
    with measure("warmup") as t:
        actions = state.get("plan_actions", {}).get("warmup", [])
        sport_type = state.get("plan", {}).get("type", "running")

        steps = []
        for idx, a in enumerate(actions, start=1):
            duration = a.get("duration", 30)
            name = a.get("name", "热身动作")
            tips = a.get("tips", "")

            # 需求12：要领 + 开始 + 倒计时，三段式播报脚本
            cues = {
                "intro": f"第 {idx} 个动作：{name}，持续 {duration} 秒。要领：{tips}" if tips else f"第 {idx} 个动作：{name}，持续 {duration} 秒。",
                "start": f"{name}，开始。",
                "countdown": _countdown_marks(duration),
                "finish": f"{name}完成。" if idx == len(actions) else f"{name}完成，准备下一个动作。",
            }

            steps.append(
                {
                    "index": idx,
                    "action_id": a.get("action_id"),
                    "name": name,
                    "voice": a.get("voice", f"开始 {name}"),
                    "tips": tips,
                    "duration": duration,
                    "cues": cues,
                    "image": a.get("image", ""),
                    "video": a.get("video", ""),
                }
            )

        total_duration = sum(s["duration"] for s in steps)

        # 预生成完整播报脚本（要领 → 开始 → 倒计时 → 完成）
        script_parts: list[str] = []
        if steps:
            script_parts.append(f"接下来进行 {len(steps)} 个热身动作，总计约 {total_duration // 60} 分钟。")
            for s in steps:
                script_parts.append(s["cues"]["intro"])
                script_parts.append(s["cues"]["start"])
                if s["cues"]["countdown"]:
                    marks = "、".join(f"剩余{m}秒" for m in s["cues"]["countdown"])
                    script_parts.append(f"（倒计时提醒：{marks}）")
                script_parts.append(s["cues"]["finish"])
            script_parts.append("热身完成，身体已经微热，可以开始正式运动了。")
        voice_script = "\n".join(script_parts) if script_parts else "无热身动作"

    logger.info(
        "warmup.done",
        user_id=state.get("user_id"),
        sport_type=sport_type,
        steps=len(steps),
        total_duration=total_duration,
        ms=round(t.ms, 1),
    )
    return {
        "warmup_guide": {
            "sport_type": sport_type,
            "steps": steps,
            "total_duration": total_duration,
            "voice_script": voice_script,
        },
        "stage": "warmup",
        "node_timings": t.result,
    }
