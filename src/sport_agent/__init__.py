"""AI Sport Coach Agent - 基于 LangGraph 的运动前阶段智能体服务。

模块划分：
- state: LangGraph 共享状态定义
- graph: 图组装与编译
- nodes: 各业务节点（健康采集/风险/恢复/问询/计划/设备/热身）
- tools: LangChain 工具层（健康/训练/环境/设备）
- agents: Agent 角色 Prompt 与 LLM 封装
- api: FastAPI 服务入口
- config: 配置管理
"""

__version__ = "0.1.0"
