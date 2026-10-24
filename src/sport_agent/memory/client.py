"""Mem0 客户端与四层记忆读写实现。

四层记忆（metadata.type 标签，对应 docs/记忆系统方案.md）：
- physio      生理指标基线：静息心率/HRV/血氧/身体电量基线/BMI
- preference  运动偏好：运动类型/强度/地点/热身/播报偏好
- history     历史运动档案：运动记录摘要/过往目标/运动后反馈/历史风险结论
- pre_sport   运动前会话档案：问询回答/评估输出/计划摘要

关键设计（实测数据支撑）：
1. 全部走 write_fact（infer=False）直写，不经 LLM 抽取：既规避幻觉篡改生理数值，
   也避开默认抽取 prompt 丢弃短答案的问题（「想减脂」实测被判无价值，extracted=0）。
2. 写入统一带业务幂等键：infer=False 绕过 mem0 去重流水线，不带键会逐次堆重复行。
3. 关键链路只用 read_snapshot（get_all + metadata 过滤，实测 1~2ms）。
4. 语义检索单次 1.3~2.3s，瓶颈在远程 embedding，故按归一化 query 做 LRU 缓存。
"""

from __future__ import annotations

import atexit
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Optional
from urllib.parse import urlparse

from sport_agent.utils import get_logger

logger = get_logger(__name__)

MEM_PHYSIO = "physio"
MEM_PREFERENCE = "preference"
MEM_HISTORY = "history"
MEM_PRE_SPORT = "pre_sport"

ALL_TYPES = (MEM_PHYSIO, MEM_PREFERENCE, MEM_HISTORY, MEM_PRE_SPORT)

# 快照单层最多取多少条，防止老用户记忆膨胀拖慢 prompt
SNAPSHOT_LIMIT_PER_TYPE = 20

_memory_singleton: Optional[Any] = None
_init_lock = threading.Lock()
_init_failed = False
_writer: Optional[ThreadPoolExecutor] = None


def _build_config() -> dict[str, Any]:
    """用项目现有配置拼 mem0 config（复用 postgres_url + LLM + embedding）。"""
    from sport_agent.config import get_settings

    s = get_settings()
    # postgres_url 可能带 +psycopg 驱动后缀，urlparse 前先去掉
    u = urlparse(s.postgres_url.replace("postgresql+psycopg://", "postgresql://"))  # type: ignore[union-attr]
    return {
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "user": u.username,
                "password": u.password,
                "host": u.hostname,
                "port": u.port or 5432,
                "dbname": (u.path or "").lstrip("/"),
                "collection_name": s.mem0_collection,
                "embedding_model_dims": s.mem0_embedding_dims,
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": s.llm_model,
                "temperature": 0.1,  # 抽取要稳定，压低随机性
                "openai_base_url": s.llm_base_url,
                "api_key": s.llm_api_key,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": s.embedding_model,
                "openai_base_url": s.effective_embedding_base_url,
                "api_key": s.effective_embedding_api_key,
                "embedding_dims": s.mem0_embedding_dims,
            },
        },
    }


def _get_memory() -> Optional[Any]:
    """进程级 Memory 单例（初始化实测 0.6~0.8s，必须复用）。

    初始化失败只记一次告警并永久降级，避免每次调用都重试拖慢主流程。
    """
    global _memory_singleton, _init_failed
    if _memory_singleton is not None:
        return _memory_singleton
    if _init_failed:
        return None

    from sport_agent.config import get_settings

    if not get_settings().use_mem0:
        _init_failed = True
        logger.info("mem0.disabled")
        return None

    with _init_lock:
        if _memory_singleton is not None:
            return _memory_singleton
        if _init_failed:
            return None
        try:
            from mem0 import Memory

            _memory_singleton = Memory.from_config(_build_config())
            logger.info("mem0.ready", collection=get_settings().mem0_collection)
            return _memory_singleton
        except Exception as e:  # noqa: BLE001
            _init_failed = True
            logger.warning("mem0.init_failed_degraded", error=str(e))
            return None


def _get_writer() -> ThreadPoolExecutor:
    """异步写入线程池（懒创建，进程退出时等待收尾）。"""
    global _writer
    if _writer is None:
        from sport_agent.config import get_settings

        _writer = ThreadPoolExecutor(
            max_workers=get_settings().mem0_writer_workers,
            thread_name_prefix="mem0-writer",
        )
        atexit.register(shutdown_writer)
    return _writer


def shutdown_writer() -> None:
    """等待后台记忆写入完成（进程退出前调用，避免丢记忆）。"""
    global _writer
    if _writer is not None:
        _writer.shutdown(wait=True)
        _writer = None


def _results(raw: Any) -> list[dict]:
    """mem0 返回值归一：新版返回 {"results": [...]}，旧版直接返回 list。"""
    if isinstance(raw, dict):
        return raw.get("results", []) or []
    return raw or []


# ===================== 读取 =====================


def read_snapshot(user_id: str) -> dict[str, list[str]]:
    """读取运动前记忆快照，按四层分组返回记忆文本。

    走 get_all + metadata 过滤，不做 embedding，实测 1~2ms，可安全放在关键链路。

    Returns:
        {"physio": [...], "preference": [...], "history": [...], "pre_sport": [...]}
        任何异常或未启用时返回各层空列表。
    """
    empty: dict[str, list[str]] = {t: [] for t in ALL_TYPES}
    if not user_id:
        return empty

    m = _get_memory()
    if m is None:
        return empty

    snapshot = dict(empty)
    try:
        raw = m.get_all(
            filters={"user_id": user_id},
            limit=SNAPSHOT_LIMIT_PER_TYPE * len(ALL_TYPES),
        )
        for item in _results(raw):
            mem_type = (item.get("metadata") or {}).get("type")
            text = item.get("memory")
            if not text:
                continue
            bucket = snapshot.get(mem_type if mem_type in ALL_TYPES else MEM_PRE_SPORT)
            if bucket is not None and len(bucket) < SNAPSHOT_LIMIT_PER_TYPE:
                bucket.append(text)
        logger.info(
            "mem0.snapshot",
            user_id=user_id,
            counts={k: len(v) for k, v in snapshot.items()},
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("mem0.snapshot_failed", user_id=user_id, error=str(e))
        return empty
    return snapshot


def _search_uncached(user_id: str, query: str, mem_type: str, limit: int) -> tuple[str, ...]:
    """真实语义检索（未缓存路径），返回不可变元组以便 lru_cache 持有。"""
    m = _get_memory()
    if m is None:
        return ()
    filters: dict[str, Any] = {"user_id": user_id}
    if mem_type:
        filters["type"] = mem_type
    try:
        raw = m.search(query, filters=filters, limit=limit)
        return tuple(i["memory"] for i in _results(raw) if i.get("memory"))
    except Exception as e:  # noqa: BLE001
        logger.warning("mem0.search_failed", user_id=user_id, error=str(e))
        return ()


_search_cached = lru_cache(maxsize=256)(_search_uncached)


def search_memory(user_id: str, query: str, mem_type: str = "", limit: int = 3) -> list[str]:
    """语义检索相关记忆（带 LRU 缓存）。

    单次未命中约 1.3~2.3s（瓶颈是远程 embedding），只在已经要调 LLM 的节点用；
    需要低延迟的场景请用 read_snapshot。

    Args:
        mem_type: 限定记忆层（physio/preference/history/pre_sport），空则不限。
    """
    if not user_id or not query:
        return []
    hits = _search_cached(user_id, query.strip(), mem_type, limit)
    if not hits:
        # 空结果不留缓存，避免记忆写入后仍被旧的空值命中
        _search_cached.cache_clear()
    return list(hits)


# ===================== 写入 =====================


def _delete_by_key(m: Any, user_id: str, key: str) -> int:
    """删除同一幂等键的旧记忆，实现 upsert 语义。

    infer=False 绕过了 mem0 的去重/更新流水线，同一条事实每次运行都会新增一行
    （实测两次运行产生两条完全相同的 physio 记忆）。这里按业务幂等键先删后写，
    与项目里运动数据同步的 UPSERT 思路一致。
    """
    removed = 0
    try:
        for item in _results(m.get_all(filters={"user_id": user_id, "key": key}, limit=50)):
            mid = item.get("id")
            if mid:
                m.delete(mid)
                removed += 1
    except Exception as e:  # noqa: BLE001
        logger.warning("mem0.delete_by_key_failed", user_id=user_id, key=key, error=str(e))
    return removed


def write_fact(user_id: str, text: str, mem_type: str, extra: Optional[dict] = None,
               key: str = "") -> bool:
    """业务事实直写（infer=False，不经 LLM 抽取）。

    生理指标、风险结论这类含数字的事实必须走这条路：LLM 抽取会改写数值造成幻觉，
    方案文档明确要求生理数据不走对话抽取。同步执行，实测 1.1~2.7s（一次 embedding）。

    Args:
        key: 业务幂等键。给定时先删同键旧记忆再写（upsert），避免重复堆积。

    Returns:
        是否写入成功（失败仅告警，不抛异常）。
    """
    m = _get_memory()
    if m is None or not user_id or not text:
        return False
    meta: dict[str, Any] = {"type": mem_type, **(extra or {})}
    if key:
        meta["key"] = key
        _delete_by_key(m, user_id, key)
    try:
        m.add(
            [{"role": "user", "content": text}],
            user_id=user_id,
            metadata=meta,
            infer=False,
        )
        logger.info("mem0.write_fact", user_id=user_id, type=mem_type, key=key or None)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("mem0.write_fact_failed", user_id=user_id, error=str(e))
        return False


def write_fact_async(user_id: str, text: str, mem_type: str,
                     extra: Optional[dict] = None, key: str = "") -> None:
    """write_fact 的异步版本（后台线程，立即返回）。

    直写虽不调抽取 LLM，但仍要做一次 embedding，实测 1.1~2.7s，
    因此运动前链路里的事实落库统一走这里。
    """
    if not user_id or not text:
        return
    from sport_agent.config import get_settings

    if not get_settings().use_mem0:
        return
    try:
        _get_writer().submit(write_fact, user_id, text, mem_type, extra, key)
    except Exception as e:  # noqa: BLE001
        logger.warning("mem0.submit_fact_failed", user_id=user_id, error=str(e))


# ===================== prompt 注入 =====================

_TYPE_CN = {
    MEM_PHYSIO: "生理基线",
    MEM_PREFERENCE: "运动偏好",
    MEM_HISTORY: "历史运动档案",
    MEM_PRE_SPORT: "既往运动前会话",
}


def format_memories(snapshot: dict[str, list[str]] | None,
                    types: tuple[str, ...] = ALL_TYPES,
                    limit_per_type: int = 5) -> str:
    """把记忆快照转成中文可读文本供 prompt 使用。

    与 utils.profile.format_profile 保持同一风格：缺失明确写「无」，
    让 LLM 区分「没有这项记忆」和「我没注意到」。
    """
    if not snapshot:
        return "无历史记忆（首次使用或记忆库为空，请勿凭空假设用户历史）"
    lines: list[str] = []
    for t in types:
        items = (snapshot.get(t) or [])[:limit_per_type]
        if items:
            body = "；".join(items)
            lines.append(f"- {_TYPE_CN.get(t, t)}：{body}")
    if not lines:
        return "无历史记忆（首次使用或记忆库为空，请勿凭空假设用户历史）"
    return "\n".join(lines)
