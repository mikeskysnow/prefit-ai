"""LangGraph 节点实现。

每个节点 = 需求清单里的一项功能能力，签名统一为：
    def node(state: SportAgentState) -> dict
返回值是需要合并进 State 的部分字段。

节点与需求映射：
- memory_load     : Mem0 运动前记忆快照读取（入口）
- memory_write    : Mem0 记忆沉淀（出口，全异步）
- report_analyze  : 体检报告解析与画像更新（仅上传时进入）
- health_collect  : F001 健康采集
- risk_assess     : F002 风险预警（纯 Python 规则，非 LLM）
- feasibility     : F003 可行性评估看板
- recovery        : F004 身体电量
- avatar          : F005 用户形象（P2 占位）
- interview       : F006 目标问询（interrupt 打断）
- plan_generate   : F007 计划生成
- plan_analyze    : F008 方案综合分析（流式）
- plan_compose    : F009 动作编排
- plan_edit       : F010 计划查看与调整（占位）
- device_check    : F011 设备检查
- warmup          : F012 热身指导
- safety_stop     : 高风险阻断终点
"""

from sport_agent.nodes.device import device_check_node
from sport_agent.nodes.health import health_collect_node, feasibility_node
from sport_agent.nodes.interview import interview_node, route_after_avatar
from sport_agent.nodes.memory import memory_load_node, memory_write_node
from sport_agent.nodes.plan import (
    plan_analyze_node,
    plan_compose_node,
    plan_edit_node,
    plan_generate_node,
    plan_review_node,
    route_after_plan_review,
)
from sport_agent.nodes.recovery import avatar_node, recovery_node
from sport_agent.nodes.report import report_analyze_node, route_after_memory_load
from sport_agent.nodes.risk import route_after_risk, risk_assess_node, safety_stop_node
from sport_agent.nodes.warmup import warmup_node

__all__ = [
    "memory_load_node",
    "memory_write_node",
    "report_analyze_node",
    "route_after_memory_load",
    "health_collect_node",
    "risk_assess_node",
    "route_after_risk",
    "safety_stop_node",
    "feasibility_node",
    "recovery_node",
    "avatar_node",
    "interview_node",
    "route_after_avatar",
    "plan_generate_node",
    "plan_analyze_node",
    "plan_review_node",
    "route_after_plan_review",
    "plan_compose_node",
    "plan_edit_node",
    "device_check_node",
    "warmup_node",
]
