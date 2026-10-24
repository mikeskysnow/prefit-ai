"""身体电量算法引擎（HRV4Training 启发的加权模型）。

对应文档《身体电量方案.md》Layer 1：
    BodyEnergy = HRV(35%) + CV(25%) + Sleep(20%) + LoadRecovery(20%)

设计要点：
- 纯 Python 确定性计算，<1s，100% 可复现（LLM 不参与数值计算）。
- 个人基线：30 天滑动窗口均值；不足 7 天用群体基线（置信度 low）。
- 输出 score/level/factors/confidence，供 LLM 解读层（Layer 2）使用。

参考：HRV4Training (Marco Altini, PhD) — rMSSD + RHR + 主观问卷 → 0-100 分数。
"""

from __future__ import annotations

from typing import Any

# ===== 常量 =====

# 加权系数（来自方案文档，可经主观校准微调）
W_HRV = 0.35
W_CV = 0.25
W_SLEEP = 0.20
W_LOAD = 0.20

# 群体基线（成年人群统计中位数，首次用户兜底）
GROUP_HRV_BASELINE = 45.0  # rMSSD ms
GROUP_RHR_BASELINE = 68.0  # 次/分
GROUP_SLEEP_SCORE = 70.0  # 0-100

# 基线校准所需天数
FULL_BASELINE_DAYS = 30
MIN_BASELINE_DAYS = 7

# 等级阈值
LEVEL_HIGH = 75
LEVEL_MEDIUM = 50
LEVEL_LOW = 25


def compute_body_battery(
    vitals: dict[str, Any],
    hrv_history: list[dict[str, Any]],
    training_history: list[dict[str, Any]],
    *,
    sleep_score: float | None = None,
) -> dict[str, Any]:
    """计算身体电量分数（0-100）。

    Args:
        vitals: 当前生理数据 {"heart_rate", "spo2", "hrv", "stress", ...}
        hrv_history: 近 N 天 HRV 历史 [{"date", "hrv"}, ...]
        training_history: 近 N 天训练历史 [{"date", "load", "duration"}, ...]
        sleep_score: 睡眠质量评分 0-100（None 时用群体默认值）

    Returns:
        {
            "score": int,           # 0-100
            "level": str,           # high / medium / low / critical
            "status": str,          # READY / MODERATE / TIRED / INITIALIZING
            "factors": dict,        # 各分量得分 {hrv, cv, sleep, load_recovery}
            "confidence": str,      # high / medium / low
            "baseline": dict,       # 基线信息 {hrv, rhr, days, source}
            "need_doctor": bool,    # 是否建议就医
        }
    """
    current_hrv = float(vitals.get("hrv", 50))
    # 优先使用 resting_hr（更准确的静息心率），回退到 heart_rate
    current_rhr = float(vitals.get("resting_hr") or vitals.get("heart_rate", 70))
    stress = float(vitals.get("stress", 30))

    # ── 1. 个人基线计算 ──
    baseline = _calc_baseline(hrv_history, current_rhr)
    hrv_baseline = baseline["hrv"]
    rhr_baseline = baseline["rhr"]
    data_days = baseline["days"]
    confidence = baseline["confidence"]

    # ── 2. HRV 恢复分量 (35%) ──
    # HRV 越高越好，相对基线比率归一化
    if hrv_baseline > 0:
        hrv_ratio = current_hrv / hrv_baseline
    else:
        hrv_ratio = 1.0
    # 比率 0~1.5 映射到 0~100
    hrv_component = min(max(hrv_ratio, 0), 1.5) / 1.5 * 100

    # ── 3. 心血管压力分量 (25%) ──
    # RHR 越低越好，基线/当前 比率
    if current_rhr > 0:
        rhr_ratio = rhr_baseline / current_rhr
    else:
        rhr_ratio = 1.0
    cv_component = min(max(rhr_ratio, 0), 1.5) / 1.5 * 100
    # 压力高时额外扣分
    cv_component = max(0, cv_component - max(0, stress - 50) * 0.3)

    # ── 4. 睡眠恢复分量 (20%) ──
    sleep_component = sleep_score if sleep_score is not None else GROUP_SLEEP_SCORE
    # 无睡眠数据时，用 HRV 和压力推断（置信度降低）
    if sleep_score is None and confidence != "high":
        # HRV 高 + 压力低 → 推断睡眠好
        inferred_sleep = 50 + (hrv_component - 50) * 0.3 + (50 - stress) * 0.2
        sleep_component = max(0, min(100, inferred_sleep))

    # ── 5. 负荷-恢复平衡分量 (20%) ──
    # 近期训练负荷越高 + 连续训练天数越多 → 恢复空间越小
    activity_load = _calc_training_load(training_history)
    consecutive_days = _consecutive_training_days(training_history)
    # 负荷扣分：每 100 负荷扣 5 分，连续训练每天额外扣 3 分
    load_penalty = activity_load * 0.05 + consecutive_days * 3
    load_recovery = max(0, 100 - load_penalty)

    # ── 6. 加权合成 ──
    raw_score = (
        hrv_component * W_HRV
        + cv_component * W_CV
        + sleep_component * W_SLEEP
        + load_recovery * W_LOAD
    )
    score = int(round(max(0, min(100, raw_score))))

    # ── 7. 分级 ──
    if score >= LEVEL_HIGH:
        level, status = "high", "READY"
    elif score >= LEVEL_MEDIUM:
        level, status = "medium", "MODERATE"
    elif score >= LEVEL_LOW:
        level, status = "low", "TIRED"
    else:
        level, status = "critical", "TIRED"

    # 新用户初始化标记
    # 无历史基线时标记 INITIALIZING，但自定义数据时仍使用计算分数（不强制 50）
    if data_days < MIN_BASELINE_DAYS:
        status = "INITIALIZING"
        if vitals.get("source") != "custom":
            score = 50  # 初始化模式给中性分数（仅无自定义数据时）

    # 就医指征：HRV 极低 + 压力极高
    need_doctor = current_hrv < 20 and stress > 80

    return {
        "score": score,
        "level": level,
        "status": status,
        "factors": {
            "hrv": round(hrv_component, 1),
            "cv": round(cv_component, 1),
            "sleep": round(sleep_component, 1),
            "load_recovery": round(load_recovery, 1),
            "activity_load": round(activity_load, 1),
            "consecutive_days": consecutive_days,
        },
        "confidence": confidence,
        "baseline": baseline,
        "need_doctor": need_doctor,
    }


def _calc_baseline(
    hrv_history: list[dict[str, Any]],
    current_rhr: float,
) -> dict[str, Any]:
    """计算个人基线（30 天滑动窗口）。

    - 30 天以上数据 → 个人基线，confidence=high
    - 7-30 天 → 部分基线，confidence=medium
    - <7 天 → 群体基线，confidence=low
    """
    valid_hrv = [h["hrv"] for h in hrv_history if h.get("hrv") is not None]
    data_days = len(valid_hrv)

    if data_days >= MIN_BASELINE_DAYS:
        hrv_baseline = sum(valid_hrv) / data_days
        # RHR 基线从 HRV 历史间接估计（实际系统应从独立 RHR 历史获取）
        # 用当前 RHR 作为近似，因为 mock 数据中没有独立的 RHR 历史
        rhr_baseline = current_rhr
        confidence = "high" if data_days >= FULL_BASELINE_DAYS else "medium"
        source = "personal"
    else:
        hrv_baseline = GROUP_HRV_BASELINE
        rhr_baseline = GROUP_RHR_BASELINE
        confidence = "low"
        source = "group"

    return {
        "hrv": round(hrv_baseline, 1),
        "rhr": round(rhr_baseline, 1),
        "days": data_days,
        "confidence": confidence,
        "source": source,
    }


def _calc_training_load(history: list[dict[str, Any]]) -> float:
    """计算近 7 天累计训练负荷（TRIMP 简化版）。"""
    if not history:
        return 0.0
    # 取最近 7 天的 load 之和
    from datetime import date as _date, timedelta as _td

    now = _date.today()
    total = 0.0
    for h in history:
        try:
            d = _date.fromisoformat(h["date"])
        except (KeyError, ValueError):
            continue
        if (now - d).days <= 7:
            total += float(h.get("load", 0))
    return total


def _consecutive_training_days(history: list[dict[str, Any]]) -> int:
    """统计连续运动天数（用于「连续 3 天自动降低强度」规则）。"""
    if not history:
        return 0
    dates = sorted({h["date"] for h in history if h.get("date")}, reverse=True)
    if not dates:
        return 0
    from datetime import date as _date

    today = _date.fromisoformat(dates[0])
    count = 1
    for d in dates[1:]:
        prev = _date.fromisoformat(d)
        if (today - prev).days == 1:
            count += 1
            today = prev
        else:
            break
    return count
