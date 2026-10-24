"""健康域节点：F001 健康采集、F003 可行性评估看板。

对应 Agent: Health Agent。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sport_agent.state import SportAgentState
from sport_agent.tools.health_tools import get_health_data, get_user_profile
from sport_agent.utils import get_logger, measure

logger = get_logger(__name__)

# 数据时效阈值：测量日期距今超过此天数即视为过旧，需提示重新检测
STALE_AFTER_DAYS = 1

# F001 必采指标（缺失即标记 missing）
_REQUIRED_METRICS: tuple[tuple[str, str], ...] = (
    ("spo2", "血氧"),
    ("heart_rate", "静息心率"),
    ("hrv", "HRV"),
)


def _vitals_age_days(vitals: dict[str, Any]) -> int:
    """计算测量数据距今天数。

    优先用 metric_date（真实测量日期）；缺失时回退 timestamp（抓取时刻）。

    注意：source=db 时 timestamp 是"抓取时刻"而非"测量时刻"，直接用它判断
    时效会让 3 天前的数据也被判为新鲜，导致「数据过久需提示重新检测」永不触发。
    """
    metric_date = vitals.get("metric_date")
    if metric_date:
        try:
            measured = datetime.fromisoformat(str(metric_date)).date()
            return max(0, (datetime.now(timezone.utc).date() - measured).days)
        except ValueError:
            pass
    ts = vitals.get("timestamp")
    if not ts:
        return 999
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return 999
    return max(0, (datetime.now(timezone.utc) - dt).days)


def _is_vitals_fresh(vitals: dict[str, Any]) -> bool:
    """数据有效性检查：测量日期是否在有效窗口内。

    - 有 metric_date（真实测量日）：当日数据为新鲜。
    - 无 metric_date（Mock/实时 PPG）：沿用 1 小时窗口。
    超时则由 Agent 主动提示用户重新检测（需求2备注）。
    """
    if vitals.get("metric_date"):
        return _vitals_age_days(vitals) <= STALE_AFTER_DAYS - 1
    ts = vitals.get("timestamp")
    if not ts:
        return False
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return False
    return datetime.now(timezone.utc) - dt <= timedelta(hours=1)


def _mark_abnormal(vitals: dict[str, Any]) -> list[dict[str, Any]]:
    """F001 验收标准2：异常数据自动标记并传递给下游判断。

    severity：critical（红线，F002 会阻断）/ warning（灰区）/ missing（缺采）。
    """
    abnormal: list[dict[str, Any]] = []

    for key, label in _REQUIRED_METRICS:
        if not vitals.get(key):
            abnormal.append(
                {"metric": key, "label": label, "value": None, "severity": "missing", "note": f"{label}未采集到有效值"}
            )

    spo2 = vitals.get("spo2") or 100
    hr = vitals.get("heart_rate") or 0
    hrv = vitals.get("hrv") or 50
    stress = vitals.get("stress") or 0

    if spo2 < 90:
        abnormal.append({"metric": "spo2", "label": "血氧", "value": spo2, "severity": "critical", "note": "低于 90% 安全阈值"})
    elif spo2 < 95:
        abnormal.append({"metric": "spo2", "label": "血氧", "value": spo2, "severity": "warning", "note": "接近边界值"})

    if hr > 100:
        abnormal.append({"metric": "heart_rate", "label": "静息心率", "value": hr, "severity": "critical", "note": "高于 100 次/分"})
    elif hr > 90:
        abnormal.append({"metric": "heart_rate", "label": "静息心率", "value": hr, "severity": "warning", "note": "偏高"})

    if hrv and hrv < 35:
        abnormal.append({"metric": "hrv", "label": "HRV", "value": hrv, "severity": "warning", "note": "偏低，恢复不足"})

    if stress > 70:
        abnormal.append({"metric": "stress", "label": "压力值", "value": stress, "severity": "warning", "note": "偏高"})

    return abnormal


def _stale_advice(age_days: int, fresh: bool) -> str:
    """需求2备注：最新数据少或过久时，智能体问询或建议用户检测生理数据。"""
    if fresh:
        return ""
    if age_days >= 999:
        return "未获取到您的生理测量数据，建议佩戴耳机测量一次静息心率与血氧后再开始运动。"
    if age_days >= 7:
        return f"最近一次生理数据是 {age_days} 天前的，参考价值有限。建议现在佩戴耳机重新测量一次。"
    return f"当前使用的是 {age_days} 天前的测量数据，建议佩戴耳机重新测量以获得更准确的评估。"


def _compute_habitual_duration(user_id: str) -> int | None:
    """从近 30 天运动记录中提取「经常运动时长」（平均每次运动分钟数）。

    用于替代问询中的 available_time：不再问用户今天有多少时间，
    而是根据其历史运动习惯推导一个合理的默认时长。
    无运动记录时返回 None，由 plan_generate 兜底默认值。
    """
    from sport_agent.tools.health_tools import get_training_history

    history = get_training_history.invoke({"user_id": user_id, "days": 30})
    durations = [h.get("duration", 0) for h in history if h.get("duration")]
    if not durations:
        return None
    avg = sum(durations) / len(durations)
    # 取整到 5 的倍数，避免 23 分钟这种碎片化数字
    return max(10, min(120, round(avg / 5) * 5))


def _enrich_profile_from_memory(profile: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """DB 无画像时，从记忆系统（Mem0）中提取生理基线补充画像。

    Mem0 中的 MEM_PHYSIO 记忆格式如 "2026-08-14 实测 静息心率 68bpm HRV 45ms"，
    可解析出 resting_hr / hrv 等生理指标。年龄/性别等画像字段不在记忆系统中，
    保留 mock 默认值。
    """
    import re

    from sport_agent.memory import MEM_PHYSIO

    snapshot = state.get("memory") or {}
    physio_texts = snapshot.get(MEM_PHYSIO) or []
    if not physio_texts:
        return profile

    blob = " ".join(str(t) for t in physio_texts)
    enriched = {**profile}

    # 从记忆文本中提取生理指标
    m = re.search(r"静息心率\s*(\d+)", blob)
    if m:
        enriched["resting_hr"] = int(m.group(1))
    m = re.search(r"HRV\s*(\d+(?:\.\d+)?)", blob)
    if m:
        enriched["hrv_baseline"] = float(m.group(1))
    m = re.search(r"血氧\s*(\d+(?:\.\d+)?)", blob)
    if m:
        enriched["spo2_baseline"] = float(m.group(1))

    if enriched != profile:
        enriched["source"] = "memory"
        logger.info("health_collect.profile_enriched_from_memory", user_id=state.get("user_id"))
    return enriched


def _resolve_user_stage(user_id: str, sync_result: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """判断新老用户 + 加载长期偏好（跨会话持久化）。

    策略：
    1. 优先查 user_state 表，mark_visit() 原子递增 visit_count：
       - visit_count=1 → 新用户，走完整问询
       - visit_count≥2 → 老用户，加载长期偏好跳过问询
    2. 数据库不可用时降级为旧逻辑：按运动记录条数推导（仅当次会话有效）

    返回：{"is_first_time": bool, "goal": ...}
    """
    try:
        from sport_agent.db import get_preferences, mark_visit

        visit_count = mark_visit(user_id)
        is_new = visit_count == 1

        result: dict[str, Any] = {"is_first_time": is_new}

        # 老用户：加载长期偏好，避免重复问询
        if not is_new:
            prefs = get_preferences(user_id)
            if prefs:
                # 只加载 state 里还没有的字段（用户本次可能已在 API 显式指定）
                for key in ("goal",):
                    if key in prefs and key not in state:
                        result[key] = prefs[key]
                logger.info("user_stage.returning_user", user_id=user_id, visit_count=visit_count, prefs=list(prefs.keys()))
            else:
                logger.info("user_stage.returning_no_prefs", user_id=user_id, visit_count=visit_count)
        else:
            logger.info("user_stage.new_user", user_id=user_id)

        return result
    except Exception as e:  # noqa: BLE001
        # DB 不可用：降级为旧逻辑（按运动记录条数推导，仅当次会话有效）
        logger.warning("user_stage.db_failed_fallback_history", error=str(e))
        history_count = sync_result.get("history_count", 0) or 0
        return {"is_first_time": history_count == 0}


def health_collect_node(state: SportAgentState) -> dict[str, Any]:
    """F001 运动前健康状态采集：抓取静息心率/血氧/HRV/压力。

    智能体启动时触发增量同步（运动数据 + 健康数据）：
    - 运动历史：从 APP 后端增量拉取 → UPSERT training_history
    - 健康数据：从 APP 后端拉取用户画像 + 心率/血氧/压力 → UPSERT user_profile + health_daily
    - 接口失败/无DB 时降级兜底，不阻塞主流程

    "如果没有当日数据，获取上一次的数据给到智能体作为健康采集数据"
    → 由 query_latest_health 的 ORDER BY metric_date DESC 实现

    需求1 验收标准2：异常数据自动标记传递判断 → vitals_abnormal。
    需求2 备注：数据过久 → vitals_advice 建议用户重新检测。
    需求4/6：区分首次/非首次 → 依据真实历史数据条数推导 is_first_time。

    性能要求：抓取时间 ≤5 秒；完整数据返回率 ≥99%。
    """
    user_id = state["user_id"]
    logger.info("health_collect.start", user_id=user_id)

    with measure("health_collect") as t:
        # === 增量同步运动历史（智能体启动时触发） ===
        sync_result = {"synced": 0, "stale": False, "error": None}
        try:
            from sport_agent.tools.training_sync import sync_training_history

            sync_result = sync_training_history(user_id)
            logger.info(
                "health_collect.training_sync_done",
                user_id=user_id,
                synced=sync_result.get("synced", 0),
                stale=sync_result.get("stale", False),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("health_collect.training_sync_failed", error=str(e))
            sync_result = {"synced": 0, "stale": True, "error": str(e)}

        # === 增量同步健康数据（用户画像 + 心率/血氧/压力） ===
        health_sync_result = {"profile_synced": False, "daily_synced": 0, "error": None}
        try:
            from sport_agent.tools.health_sync import sync_health_data

            health_sync_result = sync_health_data(user_id)
            logger.info(
                "health_collect.health_sync_done",
                user_id=user_id,
                profile=health_sync_result.get("profile_synced", False),
                daily=health_sync_result.get("daily_synced", 0),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("health_collect.health_sync_failed", error=str(e))
            health_sync_result = {"profile_synced": False, "daily_synced": 0, "error": str(e)}

        vitals = get_health_data.invoke({"user_id": user_id})

        # 自定义健康数据覆盖：用户提供时以自定义数据为准，标记为当前时间点采集
        custom = state.get("custom_vitals") or {}
        custom_vitals = custom.get("vitals") or {}
        if custom_vitals:
            vitals = {
                **vitals,
                **custom_vitals,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "custom",
            }
            logger.info("health_collect.custom_vitals_override", user_id=user_id, fields=list(custom_vitals.keys()))

        fresh = _is_vitals_fresh(vitals)
        age_days = _vitals_age_days(vitals)
        abnormal = _mark_abnormal(vitals)
        advice = _stale_advice(age_days, fresh)

        # 经常运动时长：自定义运动能力数据优先，否则从近 30 天运动记录推导
        custom_exercise = custom.get("exercise_capacity") or {}
        if custom_exercise.get("avg_duration_7d"):
            habitual_duration = max(10, min(120, int(custom_exercise["avg_duration_7d"])))
        else:
            habitual_duration = _compute_habitual_duration(user_id)

        # 提前加载用户画像写入 state，使后续 risk_assess 节点能读取年龄等画像字段。
        # feasibility_node 会再次加载并合并，不会丢失数据。
        profile = get_user_profile.invoke({"user_id": user_id})
        # DB 无画像时，从记忆系统中提取生理基线补充（API 之前同步过的数据会沉淀在 Mem0）
        if profile.get("source") == "mock":
            profile = _enrich_profile_from_memory(profile, state)

    updates: dict[str, Any] = {
        "vitals": vitals,
        "vitals_fresh": fresh,
        "vitals_age_days": age_days,
        "vitals_abnormal": abnormal,
        "vitals_advice": advice,
        "vitals_fetch_ts": vitals.get("timestamp", ""),
        "training_sync": sync_result,
        "health_sync": health_sync_result,
        "habitual_duration": habitual_duration,
        "user_profile": profile,
        "stage": "health_collect",
        "node_timings": t.result,
    }

    # 自定义运动能力数据合并到画像
    if custom_exercise:
        updates["user_profile"] = {**profile, **custom_exercise}

    # 需求4/6：区分新老用户。调用方未显式指定时以 user_state 表为准（跨会话持久化）。
    if "is_first_time" not in state:
        updates.update(_resolve_user_stage(user_id, sync_result, state))

    logger.info(
        "health_collect.done",
        user_id=user_id,
        fresh=fresh,
        age_days=age_days,
        abnormal=len(abnormal),
        hr=vitals.get("heart_rate"),
        source=vitals.get("source"),
        ms=round(t.ms, 1),
    )
    return updates


def _memory_digest(state: dict[str, Any]) -> dict[str, Any]:
    """把记忆快照压成看板可展示的摘要（各层条数 + 生理/历史两层明细）。"""
    from sport_agent.memory import MEM_HISTORY, MEM_PHYSIO

    mem = state.get("memory") or {}
    return {
        "ready": bool(state.get("memory_ready")),
        "counts": {k: len(v or []) for k, v in mem.items()},
        "physio": (mem.get(MEM_PHYSIO) or [])[:3],
        "history": (mem.get(MEM_HISTORY) or [])[:3],
    }


def feasibility_node(state: SportAgentState) -> dict[str, Any]:
    """F003 运动可行性评估看板：聚合当前数据 + 历史档案，输出风险等级与建议。

    需求3 验收标准：识别显化健康问题 + 输出风险等级（低/中/高）+ 针对异常项给建议。

    性能要求：响应时间 <3 秒；误判率 <1%。
    """
    user_id = state["user_id"]

    with measure("feasibility") as t:
        profile = get_user_profile.invoke({"user_id": user_id})
        # State 里的 user_profile 可能包含上传件解析字段和自定义运动能力数据，
        # 这些都不是 user_profile 表的列，重新读库拿不到。始终合并回来避免丢失。
        state_profile = state.get("user_profile") or {}
        if state_profile:
            profile = {**profile, **state_profile}
        # 经常运动时长：从 health_collect 推导，写入画像供 plan_generate 使用
        if state.get("habitual_duration"):
            profile["habitual_duration"] = state["habitual_duration"]
        vitals = state.get("vitals", {})

        risk_level = state.get("risk_level", "low")
        risk_flags = state.get("risk_flags", [])
        abnormal = state.get("vitals_abnormal", [])

        suggestion = _build_suggestion(risk_level, vitals, profile)

        # 需求3：针对每个异常项给出对应运动建议
        item_advice = [
            {"metric": a["metric"], "label": a["label"], "severity": a["severity"], "advice": _metric_advice(a)}
            for a in abnormal
        ]

        dashboard = {
            "risk_level": risk_level,
            "risk_level_cn": {"low": "低", "mid": "中", "high": "高", "forbidden": "高"}.get(risk_level, "低"),
            "risk_flags": risk_flags,
            "vitals": vitals,
            "vitals_age_days": state.get("vitals_age_days", 0),
            "data_fresh": state.get("vitals_fresh", False),
            "abnormal_items": item_advice,
            "profile": profile,
            "suggestion": suggestion,
            "data_advice": state.get("vitals_advice", ""),
            # 需求3「聚合当前数据 + 历史档案」：把长期记忆里的生理基线与历史档案一并呈现
            "memory": _memory_digest(state),
        }

    logger.info("feasibility.done", user_id=user_id, risk_level=risk_level, abnormal=len(item_advice), ms=round(t.ms, 1))
    return {
        "feasibility_dashboard": dashboard,
        "user_profile": profile,
        "risk_suggestion": suggestion,
        "stage": "feasibility",
        "node_timings": t.result,
    }


def _metric_advice(item: dict[str, Any]) -> str:
    """需求3：针对单个异常项给出运动建议。"""
    metric = item.get("metric")
    severity = item.get("severity")
    if severity == "missing":
        return f"{item.get('label')}缺失，建议佩戴耳机重新测量后再评估"
    advice = {
        "spo2": {
            "critical": "血氧过低，强制禁止高强度运动，请尽快就医",
            "warning": "血氧接近边界，建议仅做低强度活动并留意呼吸感受",
        },
        "heart_rate": {
            "critical": "静息心率过高，不建议剧烈运动，建议医学检查",
            "warning": "静息心率偏高，建议降低强度并延长热身",
        },
        "hrv": {"warning": "HRV 偏低说明恢复不足，建议今日以 Zone1 恢复性活动为主"},
        "stress": {"warning": "压力偏高，建议先做呼吸调节，避免高强度间歇"},
    }
    return advice.get(metric, {}).get(severity, "建议适当降低运动强度")


def _build_suggestion(risk_level: str, vitals: dict[str, Any], profile: dict[str, Any]) -> str:
    """生成运动建议话术（阶段2 由 LLM 润色，骨架阶段用模板）。"""
    if risk_level == "high":
        return "静息心率/血氧异常，建议暂停剧烈运动并就医检查。"
    if risk_level == "forbidden":
        return "血氧过低，强制禁止高强度运动，请尽快就医。"
    if risk_level == "mid":
        return "今日身体状态偏弱，建议降低运动强度，进行 Zone1 恢复性活动。"
    return f"今日状态良好，可按目标 {profile.get('goal', '健康')} 进行训练。"
