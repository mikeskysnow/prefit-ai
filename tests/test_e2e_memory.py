"""端到端记忆测试：同一用户两次会话，验证记忆沉淀 → 读取 → 注入。

需要真实 mem0（pgvector + LLM），因此默认跳过；显式开启后运行：
    RUN_MEM0_E2E=1 .venv/bin/python -m pytest tests/test_e2e_memory.py -s

RUN1 新用户：完整问询（三次 interrupt）→ 后台写 physio/history/preference
RUN2 老用户：memory_load 读到快照 → 跳过问询 → 记忆注入 plan/recovery
幂等：同日重跑，physio 与 preference 各自仍只有 1 条（业务键 upsert）
"""

from __future__ import annotations

import os
import time

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from sport_agent.graph import build_app
from sport_agent.memory import (
    MEM_HISTORY,
    MEM_PHYSIO,
    MEM_PREFERENCE,
    read_snapshot,
    shutdown_writer,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_MEM0_E2E"),
    reason="需真实 mem0/pgvector 与 LLM，设 RUN_MEM0_E2E=1 才运行",
)

USER_ID = "e2e_mem_9001"


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}


@pytest.fixture()
def app():
    return build_app(checkpointer=InMemorySaver())


@pytest.fixture(autouse=True)
def _cleanup():
    """跑前跑后都清掉探测数据，避免污染真实记忆库与后续断言。"""
    _purge()
    yield
    _purge()


def _purge() -> None:
    from sport_agent.memory.client import _get_memory, _results

    m = _get_memory()
    if m is None:
        return
    for item in _results(m.get_all(filters={"user_id": USER_ID}, limit=200)):
        if item.get("id"):
            m.delete(item["id"])


def _drain_writes() -> None:
    """等后台记忆写入落库（write_fact_async 单次含一次 embedding，1~3s）。"""
    shutdown_writer()
    time.sleep(1)


def _pending_field(app, config) -> str:
    """取当前挂起 interrupt 的问询字段名。

    不用 snapshot.next 判断：resume 之后同一节点再次 interrupt 时 next 为空，
    但 tasks[0].interrupts 里仍有待回答的问题。
    """
    tasks = app.get_state(config).tasks
    if not tasks or not tasks[0].interrupts:
        return ""
    return tasks[0].interrupts[0].value.get("field", "")


def test_memory_accumulates_and_injects(app):
    # ===== RUN1：新用户走完整问询 =====
    config1 = _config(f"{USER_ID}-run1")
    app.invoke(
        {"user_id": USER_ID, "session_id": "session_run1", "is_first_time": True},
        config=config1,
    )
    # 顺序问询：goal → available_time → intensity_pref
    for field, answer in (
        ("goal", "想减脂"),
        ("available_time", "40分钟"),
        ("intensity_pref", "中等强度"),
    ):
        assert _pending_field(app, config1) == field, f"应停在 interview 问 {field}"
        app.invoke(Command(resume=answer), config=config1)
    app.invoke(Command(resume="继续"), config=config1)  # plan_review 放行

    state1 = app.get_state(config1).values
    assert state1["stage"] == "warmup"
    assert state1["goal"] == "fat_loss"
    assert state1["available_time"] == 40

    _drain_writes()
    snap = read_snapshot(USER_ID)
    assert snap[MEM_PHYSIO], "本次生理实测应沉淀"
    assert snap[MEM_HISTORY], "风险结论与计划摘要应沉淀"
    assert snap[MEM_PREFERENCE], "问询收集的偏好应沉淀"
    assert "40" in " ".join(snap[MEM_PREFERENCE]), "偏好里应保留时长原值"

    # ===== RUN2：老用户复用记忆 =====
    config2 = _config(f"{USER_ID}-run2")
    app.invoke(
        {"user_id": USER_ID, "session_id": "session_run2", "is_first_time": False},
        config=config2,
    )
    app.invoke(Command(resume="继续"), config=config2)  # 仅 plan_review 一次打断

    state2 = app.get_state(config2).values
    assert state2["stage"] == "warmup"
    assert state2.get("inquiry_completed") is not True, "老用户应跳过问询"
    # memory_load 把快照带进了 state，供各评估节点注入 prompt
    assert state2["memory"][MEM_PREFERENCE], "老用户应读到偏好记忆"
    assert state2["plan"]["type"] in ("running", "indoor_bike", "walk")

    # ===== 幂等：同日重跑不堆积 =====
    _drain_writes()
    final = read_snapshot(USER_ID)
    assert len(final[MEM_PHYSIO]) == 1, "同日 physio 应按日期键覆盖"
    assert len(final[MEM_PREFERENCE]) == 1, "preference 固定键只保留最新一版"
