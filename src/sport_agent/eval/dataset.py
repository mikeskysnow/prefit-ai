"""风险识别评估数据集生成。

设计：
- 按医学规则标注 ground truth（label_risk），独立于 risk_assess_node 实现，
  作为后续规则/LLM 迭代的回归基线。
- 合成数据按风险等级分层采样，保证异常样本充足（高风险占比高于真实分布，
  便于稳定测量误报率/漏报率）。

标注规则（与 F002 一致）：
- SpO2 < 90% → forbidden（强制禁止高强度）
- 静息 HR > 100 → high（禁止剧烈运动）
- HR > 90 或 SpO2 < 95 或 HRV < 35 → mid
- 其余 → low
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

RiskLevel = Literal["low", "mid", "high", "forbidden"]

# 分层采样目标分布（异常占比高于真实，保证小概率事件评估稳定）
TARGET_DIST: dict[str, float] = {
    "low": 0.55,
    "mid": 0.25,
    "high": 0.12,
    "forbidden": 0.08,
}


def label_risk(vitals: dict[str, Any]) -> RiskLevel:
    """独立实现的医学规则标注（ground truth）。"""
    hr = vitals.get("heart_rate", 70)
    spo2 = vitals.get("spo2", 98)
    hrv = vitals.get("hrv", 50)

    if spo2 < 90:
        return "forbidden"
    if hr > 100:
        return "high"
    if hr > 90 or spo2 < 95 or hrv < 35:
        return "mid"
    return "low"


def _sample_vitals(level: RiskLevel, rng: random.Random) -> dict[str, Any]:
    """按目标风险等级采样一组生理数据（保证落到该等级）。"""
    if level == "forbidden":
        return {
            "heart_rate": rng.randint(70, 130),
            "spo2": rng.randint(80, 89),  # <90
            "hrv": rng.randint(20, 60),
            "stress": rng.randint(20, 90),
        }
    if level == "high":
        return {
            "heart_rate": rng.randint(101, 130),  # >100，但 spo2>=90
            "spo2": rng.randint(90, 99),
            "hrv": rng.randint(20, 60),
            "stress": rng.randint(20, 90),
        }
    if level == "mid":
        # 三选一触发 mid：HR 90-100 / SpO2 90-94 / HRV <35
        trigger = rng.choice(["hr", "spo2", "hrv"])
        if trigger == "hr":
            return {
                "heart_rate": rng.randint(91, 100),
                "spo2": rng.randint(95, 99),
                "hrv": rng.randint(40, 70),
                "stress": rng.randint(10, 70),
            }
        if trigger == "spo2":
            return {
                "heart_rate": rng.randint(60, 88),
                "spo2": rng.randint(90, 94),
                "hrv": rng.randint(40, 70),
                "stress": rng.randint(10, 70),
            }
        return {
            "heart_rate": rng.randint(60, 88),
            "spo2": rng.randint(95, 99),
            "hrv": rng.randint(20, 34),  # <35
            "stress": rng.randint(10, 70),
        }
    # low
    return {
        "heart_rate": rng.randint(55, 88),
        "spo2": rng.randint(96, 100),
        "hrv": rng.randint(45, 80),
        "stress": rng.randint(5, 50),
    }


def generate_risk_dataset(n: int = 10000, seed: int = 42) -> list[dict[str, Any]]:
    """生成 n 组风险识别评估数据集。

    Returns:
        每条 {"vitals": {...}, "expected": {"risk_level": ...}, "id": str}
    """
    rng = random.Random(seed)
    levels = list(TARGET_DIST.keys())
    weights = [TARGET_DIST[lv] for lv in levels]

    items: list[dict[str, Any]] = []
    for i in range(n):
        level = rng.choices(levels, weights=weights, k=1)[0]
        vitals = _sample_vitals(level, rng)
        # 校正：保证采样后标注与目标一致（防止边界抖动）
        expected = label_risk(vitals)
        if expected != level:
            # 极少数边界情况，重新采一次
            vitals = _sample_vitals(level, rng)
        items.append(
            {
                "id": f"risk-{seed:04d}-{i:05d}",
                "vitals": vitals,
                "expected": {"risk_level": label_risk(vitals)},
            }
        )
    return items


def save_dataset(items: list[dict[str, Any]], path: str | Path) -> None:
    """保存为 JSONL。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """加载 JSONL 数据集。"""
    p = Path(path)
    items: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def upload_to_langsmith(items: list[dict[str, Any]], dataset_name: str = "risk-assessment-eval") -> Any:
    """上传数据集到 LangSmith（需配置 LANGSMITH_API_KEY）。

    Returns:
        LangSmith dataset 对象；未配置 key 时返回 None。
    """
    try:
        from langsmith import Client
    except ImportError:
        print("langsmith 未安装，跳过上传")
        return None

    client = Client()  # 自动读取环境变量
    ds = client.create_dataset(
        dataset_name=dataset_name,
        description="运动智能体 F002 风险识别评估数据集（合成+医学规则标注）",
    )
    for it in items:
        client.create_example(
            inputs={"vitals": it["vitals"], "user_id": "eval"},
            outputs={"risk_level": it["expected"]["risk_level"], "risk_flags": [], "risk_reason": []},
            dataset_id=ds.id,
        )
    print(f"已上传 {len(items)} 条到 LangSmith 数据集: {dataset_name}")
    return ds
