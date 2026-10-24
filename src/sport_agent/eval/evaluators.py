"""风险识别评估器。

指标定义（对应 F002 验收标准：误报率≤3%、漏报率≤1%）：
- accuracy: risk_level 完全匹配率
- false_positive_rate: 实际 safe(low/mid) 但预测为 unsafe(high/forbidden)
- false_negative_rate: 实际 unsafe(high/forbidden) 但预测为 safe(low/mid)（最严重）
- forbidden_miss_rate: 实际 forbidden 但未识别为 forbidden（安全致命漏报）
- per_level: 各等级的 precision/recall
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

UNSAFE = {"high", "forbidden"}
SAFE = {"low", "mid"}


def evaluate_risk(predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]) -> dict[str, Any]:
    """评估风险识别结果。

    Args:
        predictions: 每条 {"risk_level": "low|mid|high|forbidden", ...}
        ground_truth: 每条 {"risk_level": "low|mid|high|forbidden"}

    Returns:
        指标字典。
    """
    assert len(predictions) == len(ground_truth), "预测与标注数量不一致"

    n = len(predictions)
    if n == 0:
        return {"n": 0}

    correct = 0
    fp = 0  # 实际 safe，预测 unsafe
    fn = 0  # 实际 unsafe，预测 safe
    forbidden_actual = 0
    forbidden_miss = 0  # 实际 forbidden，预测非 forbidden

    # 各等级统计
    tp_by: dict[str, int] = defaultdict(int)
    fp_by: dict[str, int] = defaultdict(int)
    fn_by: dict[str, int] = defaultdict(int)
    actual_by: dict[str, int] = defaultdict(int)

    for pred, gt in zip(predictions, ground_truth):
        p_level = pred.get("risk_level", "low")
        g_level = gt.get("risk_level", "low")
        actual_by[g_level] += 1

        if p_level == g_level:
            correct += 1
            tp_by[g_level] += 1
        else:
            fp_by[p_level] += 1
            fn_by[g_level] += 1

        # 安全二元：unsafe vs safe
        if g_level in UNSAFE and p_level in SAFE:
            fn += 1
        if g_level in SAFE and p_level in UNSAFE:
            fp += 1

        # forbidden 致命漏报
        if g_level == "forbidden":
            forbidden_actual += 1
            if p_level != "forbidden":
                forbidden_miss += 1

    accuracy = correct / n
    # 误报率分母：实际 safe 数量
    safe_n = sum(actual_by[lv] for lv in SAFE)
    unsafe_n = sum(actual_by[lv] for lv in UNSAFE)
    fpr = fp / safe_n if safe_n else 0.0
    fnr = fn / unsafe_n if unsafe_n else 0.0
    forbidden_miss_rate = forbidden_miss / forbidden_actual if forbidden_actual else 0.0

    # 各等级 precision/recall/f1
    per_level: dict[str, dict[str, float]] = {}
    levels = ["low", "mid", "high", "forbidden"]
    for lv in levels:
        tp = tp_by[lv]
        prec = tp / (tp + fp_by[lv]) if (tp + fp_by[lv]) else 0.0
        rec = tp / (tp + fn_by[lv]) if (tp + fn_by[lv]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_level[lv] = {"precision": prec, "recall": rec, "f1": f1, "support": actual_by[lv]}

    return {
        "n": n,
        "accuracy": accuracy,
        "false_positive_rate": fpr,  # 误报率（目标 ≤3%）
        "false_negative_rate": fnr,  # 漏报率（目标 ≤1%）
        "forbidden_miss_rate": forbidden_miss_rate,  # 致命漏报（目标 0%）
        "distribution": dict(actual_by),
        "per_level": per_level,
        "sla_pass": {
            "fpr_le_3pct": fpr <= 0.03,
            "fnr_le_1pct": fnr <= 0.01,
            "forbidden_miss_zero": forbidden_miss_rate == 0.0,
        },
    }


def format_report(metrics: dict[str, Any]) -> str:
    """格式化评估报告为可读文本。"""
    lines = [
        "=" * 60,
        "F002 风险识别评估报告",
        "=" * 60,
        f"样本数: {metrics['n']}",
        f"分布: {metrics['distribution']}",
        "",
        f"准确率 accuracy:          {metrics['accuracy']:.4f}",
        f"误报率 false_positive:    {metrics['false_positive_rate']:.4f}  (SLA ≤3% → "
        f"{'PASS' if metrics['sla_pass']['fpr_le_3pct'] else 'FAIL'})",
        f"漏报率 false_negative:    {metrics['false_negative_rate']:.4f}  (SLA ≤1% → "
        f"{'PASS' if metrics['sla_pass']['fnr_le_1pct'] else 'FAIL'})",
        f"致命漏报 forbidden_miss:  {metrics['forbidden_miss_rate']:.4f}  (SLA =0% → "
        f"{'PASS' if metrics['sla_pass']['forbidden_miss_zero'] else 'FAIL'})",
        "",
        "各等级 P/R/F1:",
    ]
    for lv in ["low", "mid", "high", "forbidden"]:
        m = metrics["per_level"][lv]
        lines.append(
            f"  {lv:9s} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} (n={m['support']})"
        )
    lines.append("=" * 60)
    return "\n".join(lines)
