"""LangSmith 评估模块：风险识别数据集 + 评估器 + 运行脚本。

对应需求：风险识别误报率≤3%、漏报率≤1%（F002 硬指标）。
能力：
- dataset: 合成风险识别评估数据集（默认 10000 组，按医学规则标注）
- evaluators: 准确率/误报率/漏报率/安全致命漏报率
- run_eval: 本地跑评估 + 可选上传 LangSmith 作为回归基线

用法：
    python -m sport_agent.eval.run_eval --n 10000
    python -m sport_agent.eval.run_eval --n 10000 --upload  # 上传 LangSmith
"""

from sport_agent.eval.dataset import generate_risk_dataset, label_risk, load_dataset, save_dataset
from sport_agent.eval.evaluators import evaluate_risk

__all__ = [
    "generate_risk_dataset",
    "label_risk",
    "save_dataset",
    "load_dataset",
    "evaluate_risk",
]
