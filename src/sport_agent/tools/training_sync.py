"""运动数据增量同步：智能体启动时触发 → 增量拉取 → UPSERT → 降级兜底。

对应方案《运动记录数据获取.md》核心策略：
1. 智能体启动时检查 last_sync_ts，超过阈值则增量拉取
2. UPSERT 幂等写入 training_history 表
3. 接口失败时不阻塞主流程，使用已有旧数据 + 标记 stale
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sport_agent.config import get_settings
from sport_agent.db.training_history import (
    is_sync_stale,
    query_recent_history,
    update_sync_state,
    upsert_training_records,
)
from sport_agent.tools.exercise_api import fetch_and_transform
from sport_agent.utils import get_logger

logger = get_logger(__name__)

# 增量同步阈值：距上次同步超过此时间则触发拉取
_SYNC_THRESHOLD_MINUTES = 60

# 增量拉取回溯天数（确保不漏数据，覆盖 last_sync 以来的全部记录）
_BACKFILL_DAYS = 7


def sync_training_history(user_id: str, force: bool = False) -> dict[str, Any]:
    """增量同步用户运动数据到 training_history 表。

    流程：
    1. 检查 sync_state.last_sync_ts，判断是否需要同步
    2. 需要时调 APP 接口增量拉取 → ETL → UPSERT
    3. 更新 sync_state
    4. 接口失败/无DB 时降级，返回已有数据 + stale 标记

    Args:
        user_id: 用户 ID
        force: 强制同步（忽略阈值检查）

    Returns:
        {"synced": int, "stale": bool, "error": str|None, "history": list}
    """
    s = get_settings()

    # 无 Postgres → 直接降级返回空
    if not s.postgres_url:
        logger.info("sync.skip_no_db", user_id=user_id)
        return {"synced": 0, "stale": True, "error": "no_db"}

    # 检查是否需要同步
    need_sync = force or is_sync_stale(user_id, _SYNC_THRESHOLD_MINUTES)
    if not need_sync:
        logger.info("sync.fresh_skip", user_id=user_id)
        count = _count_recent(user_id)
        return {"synced": 0, "stale": False, "error": None, "history_count": count}

    # 增量拉取：回溯 _BACKFILL_DAYS 天确保不漏
    now = datetime.now(timezone.utc)
    start_date = (now - timedelta(days=_BACKFILL_DAYS)).strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")

    try:
        summaries = fetch_and_transform(user_id, start_date, end_date)
        written = upsert_training_records(user_id, summaries)
        update_sync_state(user_id, status="ok", count=written)

        count = _count_recent(user_id)
        logger.info("sync.done", user_id=user_id, synced=written, history_count=count)
        return {"synced": written, "stale": False, "error": None, "history_count": count}

    except Exception as e:  # noqa: BLE001
        # 降级兜底：使用已有旧数据，标记 stale
        logger.warning("sync.failed_fallback", user_id=user_id, error=str(e))
        update_sync_state(user_id, status="error", error=str(e), count=0)
        count = _count_recent(user_id)
        return {"synced": 0, "stale": True, "error": str(e), "history_count": count}


def _count_recent(user_id: str) -> int:
    """安全查询近7天记录数（失败返回0）。"""
    try:
        return len(query_recent_history(user_id, days=7))
    except Exception:  # noqa: BLE001
        return 0
