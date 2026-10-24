"""节点耗时度量：对照产品需求文档各项 SLA。

产品需求文档规定的性能要求（毫秒）：
    F001 健康采集      ≤5s
    F002 风险预警      ≤1s
    F003 可行性评估    <3s
    F004 身体电量      <10s
    F005 用户形象      <30s（后续更新）
    F007 计划生成      <10s（不含分析）
    F008 方案分析      ≤20s（流式）
    F010 方案变更      <1s
    F011 设备检查      ≤5s
    F012 热身播报      ≤200ms

用法：
    with measure("risk_assess") as t:
        ...
    return {..., "node_timings": t.result}
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

from sport_agent.utils.logging import get_logger

logger = get_logger(__name__)

# 节点 → SLA 上限（毫秒），来源：docs/产品需求文档.md「性能要求」列
NODE_SLA_MS: dict[str, int] = {
    "health_collect": 5000,
    "risk_assess": 1000,
    "risk_gray_zone": 0,  # 灰区增强（Layer2+Layer3），属信息性度量，不设 SLA 上限
    "feasibility": 3000,
    "recovery": 10000,
    "avatar": 30000,
    "plan_generate": 10000,
    "plan_analyze": 20000,
    "plan_compose": 10000,
    "plan_edit": 1000,
    "device_check": 5000,
    "warmup": 200,
}


class _Timing:
    """承载单个节点的耗时结果，供节点回写 State.node_timings。"""

    def __init__(self, node: str) -> None:
        self.node = node
        self.ms: float = 0.0
        self.sla_ms: int = NODE_SLA_MS.get(node, 0)

    @property
    def within_sla(self) -> bool:
        return self.sla_ms <= 0 or self.ms <= self.sla_ms

    @property
    def result(self) -> dict[str, Any]:
        """返回可直接合并进 State.node_timings 的结构。"""
        return {
            self.node: {
                "ms": round(self.ms, 1),
                "sla_ms": self.sla_ms,
                "within_sla": self.within_sla,
            }
        }


@contextmanager
def measure(node: str) -> Iterator[_Timing]:
    """度量节点耗时并在超出 SLA 时告警。"""
    t = _Timing(node)
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.ms = (time.perf_counter() - start) * 1000
        if not t.within_sla:
            logger.warning("node.sla_exceeded", node=node, ms=round(t.ms, 1), sla_ms=t.sla_ms)
