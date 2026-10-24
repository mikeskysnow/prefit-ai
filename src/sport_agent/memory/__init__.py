"""Mem0 长期记忆门面。

对业务节点只暴露 4 个函数，内部实现（进程内 SDK / 未来的 HTTP 微服务）可整体替换
而不影响调用方：

- read_snapshot(user_id)    运动前快照读取，一次拉全量分层记忆（实测 ~2ms）
- search_memory(...)        语义检索，带 LRU 缓存（未命中约 1.4s，只用于 LLM 节点）
- write_fact(...)           业务事实直写，不经 LLM 抽取（生理数值防幻觉）
- write_fact_async(...)     上面的后台版（单次 embedding 实测 1~3s，不进同步链路）

降级原则：记忆层是增强项。mem0 不可用、PG 挂掉、LLM 无 Key 时全部静默返回空，
运动前主流程照常跑通。
"""

from sport_agent.memory.client import (
    MEM_HISTORY,
    MEM_PHYSIO,
    MEM_PREFERENCE,
    MEM_PRE_SPORT,
    format_memories,
    read_snapshot,
    search_memory,
    shutdown_writer,
    write_fact,
    write_fact_async,
)

__all__ = [
    "MEM_HISTORY",
    "MEM_PHYSIO",
    "MEM_PREFERENCE",
    "MEM_PRE_SPORT",
    "format_memories",
    "read_snapshot",
    "search_memory",
    "shutdown_writer",
    "write_fact",
    "write_fact_async",
]
