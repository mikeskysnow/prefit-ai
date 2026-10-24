"""训练域工具：运动方案知识库检索（F007）、动作库检索（F009）。

对应 Agent: Coach Agent。
阶段2 接入向量库（pgvector/Milvus）+ LangChain Retriever，骨架阶段返回 Mock。
"""

from __future__ import annotations

from typing import Any

from sport_agent.tools.base import sport_tool


@sport_tool
def search_training_plan(
    goal: str,
    body_energy: int,
    available_time: int,
    risk_level: str = "low",
) -> list[dict[str, Any]]:
    """检索匹配的运动方案候选（来自运动处方知识库）。

    Args:
        goal: 运动目标（fat_loss/cardio/endurance/muscle/health）
        body_energy: 身体电量 0-100
        available_time: 可用时间（分钟）
        risk_level: 风险等级（low/mid/high/forbidden）

    Returns:
        方案候选列表，每项含 type/duration/zone/intensity。
    """
    # 阶段2：用 LangChain Retriever 检索 pgvector 运动处方知识库
    base_duration = min(available_time, 40)
    if risk_level in ("high", "forbidden"):
        return [
            {
                "type": "walk",
                "duration": min(base_duration, 20),
                "zone": "Zone1",
                "intensity": "low",
                "rationale": "风险较高，仅推荐低强度恢复活动",
            }
        ]
    if body_energy < 40:
        return [
            {
                "type": "running",
                "duration": min(base_duration, 25),
                "zone": "Zone2",
                "intensity": "low",
                "rationale": "身体电量偏低，推荐恢复跑",
            }
        ]
    return [
        {
            "type": "running",
            "duration": base_duration,
            "zone": "Zone2" if goal != "endurance" else "Zone3",
            "intensity": "mid",
            "rationale": f"匹配目标 {goal} 的推荐方案",
        }
    ]


# 需求9：热身/拉伸方案必须与主运动类型匹配（图文匹配度 100%）
# 结构：{运动类型: {阶段: [动作...]}}
_EXERCISE_LIBRARY: dict[str, dict[str, list[dict[str, Any]]]] = {
    "running": {
        "warmup": [
            {"action_id": "w_run_dyn_stretch", "name": "动态拉伸", "voice": "开始动态拉伸", "tips": "保持核心收紧，动作缓慢可控", "duration": 180},
            {"action_id": "w_run_high_knees", "name": "高抬腿", "voice": "高抬腿30秒", "tips": "前脚掌着地，大腿抬至水平", "duration": 30},
            {"action_id": "w_run_butt_kicks", "name": "后踢腿", "voice": "后踢腿30秒", "tips": "脚跟贴近臀部，上身保持直立", "duration": 30},
        ],
        "main": [
            {"action_id": "m_run_zone2", "name": "Zone2 持续跑", "voice": "保持 Zone2 心率", "tips": "鼻吸口呼，步频 180 左右", "duration": 1800},
        ],
        "cooldown": [
            {"action_id": "c_run_quad_stretch", "name": "股四头肌拉伸", "voice": "拉伸股四头肌", "tips": "每侧保持 20 秒", "duration": 60},
            {"action_id": "c_run_calf_stretch", "name": "小腿拉伸", "voice": "拉伸小腿", "tips": "后脚跟压地，每侧 20 秒", "duration": 60},
        ],
    },
    "walk": {
        "warmup": [
            {"action_id": "w_walk_ankle", "name": "踝关节绕环", "voice": "踝关节绕环", "tips": "双向各绕 10 圈", "duration": 60},
            {"action_id": "w_walk_arm_swing", "name": "摆臂练习", "voice": "摆臂预热", "tips": "肩部放松，前后摆动", "duration": 60},
        ],
        "main": [
            {"action_id": "m_walk_brisk", "name": "快走", "voice": "保持匀速快走", "tips": "步幅适中，摆臂带动", "duration": 1800},
        ],
        "cooldown": [
            {"action_id": "c_walk_hamstring", "name": "腿后侧拉伸", "voice": "拉伸腿后侧", "tips": "膝盖微屈，每侧 20 秒", "duration": 60},
        ],
    },
    "cycling": {
        "warmup": [
            {"action_id": "w_cyc_leg_swing", "name": "腿部摆动", "voice": "腿部摆动预热", "tips": "扶稳支撑物，前后摆动", "duration": 60},
            {"action_id": "w_cyc_easy_spin", "name": "低阻力慢骑", "voice": "低阻力慢骑热身", "tips": "踏频 80，阻力最低档", "duration": 300},
        ],
        "main": [
            {"action_id": "m_cyc_zone2", "name": "Zone2 骑行", "voice": "保持 Zone2 踏频", "tips": "踏频 85-95，坐姿稳定", "duration": 1800},
        ],
        "cooldown": [
            {"action_id": "c_cyc_quad_stretch", "name": "股四头肌拉伸", "voice": "拉伸股四头肌", "tips": "每侧保持 20 秒", "duration": 60},
            {"action_id": "c_cyc_hip_stretch", "name": "髋部拉伸", "voice": "拉伸髋部", "tips": "骑行后髋部易紧张，每侧 30 秒", "duration": 60},
        ],
    },
}

# 室内运动复用对应室外类型的动作库
_LIBRARY_ALIASES = {
    "treadmill": "running",
    "indoor_bike": "cycling",
    "indoor_walk": "walk",
    "jump_rope": "running",
    "swimming": "walk",
}


@sport_tool
def search_exercise_library(sport_type: str, phase: str) -> list[dict[str, Any]]:
    """检索运动动作库（热身/主运动/拉伸），返回图文+视频+语音资源。

    需求9：热身/拉伸方案与主运动类型匹配，图文匹配度 100%。

    Args:
        sport_type: 运动类型（running/walk/cycling/treadmill/indoor_bike...）
        phase: 阶段（warmup/main/cooldown）

    Returns:
        动作列表，含 action_id/name/video/image/voice/tips/duration。
    """
    # 阶段2：检索 pgvector 动作向量库
    key = _LIBRARY_ALIASES.get(sport_type, sport_type)
    library = _EXERCISE_LIBRARY.get(key) or _EXERCISE_LIBRARY["running"]
    actions = library.get(phase, [])

    # 补齐图文/视频资源字段，保证前端渲染字段完整（图文匹配度 100%）
    return [
        {
            **a,
            "sport_type": sport_type,
            "phase": phase,
            "image": a.get("image", f"/assets/exercise/{a['action_id']}.png"),
            "video": a.get("video", f"/assets/exercise/{a['action_id']}.mp4"),
        }
        for a in actions
    ]
