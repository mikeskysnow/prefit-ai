"""风险评估运行脚本：生成数据集 → 跑 risk_assess_node → 评估 → 输出报告。

用法：
    python -m sport_agent.eval.run_eval --n 10000
    python -m sport_agent.eval.run_eval --n 10000 --seed 42 --upload
    python -m sport_agent.eval.run_eval --dataset eval_data.jsonl  # 复用已存数据集
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sport_agent.config import apply_langsmith_env
from sport_agent.eval.dataset import (
    generate_risk_dataset,
    load_dataset,
    save_dataset,
    upload_to_langsmith,
)
from sport_agent.eval.evaluators import evaluate_risk, format_report
from sport_agent.utils import get_logger

logger = get_logger(__name__)


def run_risk_assess(items: list[dict]) -> list[dict]:
    """对数据集每条跑 risk_assess_node，返回预测结果。"""
    from sport_agent.nodes.risk import risk_assess_node

    predictions = []
    for it in items:
        # 构造 risk_assess_node 需要的最小 state
        state = {"user_id": "eval", "vitals": it["vitals"]}
        result = risk_assess_node(state)
        predictions.append(
            {
                "risk_level": result["risk_level"],
                "risk_flags": result.get("risk_flags", []),
            }
        )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="F002 风险识别评估")
    parser.add_argument("--n", type=int, default=10000, help="生成样本数（默认 10000）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--dataset", type=str, default=None, help="复用已有 JSONL 数据集路径")
    parser.add_argument("--save", type=str, default=None, help="保存生成数据集到 JSONL")
    parser.add_argument("--upload", action="store_true", help="上传数据集到 LangSmith")
    parser.add_argument("--dataset-name", type=str, default="risk-assessment-eval")
    args = parser.parse_args()

    apply_langsmith_env()

    # 1. 加载/生成数据集
    if args.dataset:
        items = load_dataset(args.dataset)
        logger.info("dataset.loaded", path=args.dataset, n=len(items))
    else:
        items = generate_risk_dataset(n=args.n, seed=args.seed)
        logger.info("dataset.generated", n=len(items), seed=args.seed)
        if args.save:
            save_dataset(items, args.save)
            logger.info("dataset.saved", path=args.save)

    # 2. 上传 LangSmith（可选）
    if args.upload:
        try:
            upload_to_langsmith(items, dataset_name=args.dataset_name)
        except Exception as e:  # noqa: BLE001
            logger.warning("langsmith.upload.failed", error=str(e))

    # 3. 跑 risk_assess_node
    logger.info("eval.running", n=len(items))
    predictions = run_risk_assess(items)
    ground_truth = [{"risk_level": it["expected"]["risk_level"]} for it in items]

    # 4. 评估 + 报告
    metrics = evaluate_risk(predictions, ground_truth)
    report = format_report(metrics)
    print(report)

    # 5. 退出码：SLA 全过为 0
    sla = metrics["sla_pass"]
    sys.exit(0 if all(sla.values()) else 1)


if __name__ == "__main__":
    main()
