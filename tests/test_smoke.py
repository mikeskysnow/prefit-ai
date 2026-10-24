"""冒烟测试：验证运动前阶段骨架流程可端到端跑通。

覆盖：
- 正常主流程（低风险 → 全链路 → warmup）
- 高风险分支（条件路由 → safety_stop）
- interrupt 恢复（interview 问询）
- 新老用户分流（route_after_avatar）
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from sport_agent.graph import build_app


@pytest.fixture(autouse=True)
def _no_mem0(monkeypatch):
    """冒烟测试禁用 mem0，保证流程断言只受本次输入影响。

    记忆库是跨进程持久的：固定 user_id 跑第二遍时，上一遍写入的偏好会被
    memory_load 读回并预填问询字段，interview 直接跳过 → 断言问询打断必然失败。

    直接掐 client 的 _get_memory：get_settings 带 lru_cache，改 MEM0_ENABLED
    环境变量在进程内已经生效不了。
    """
    from sport_agent.memory import client

    monkeypatch.setattr(client, "_get_memory", lambda: None)


@pytest.fixture()
def app():
    """每个测试用独立内存 checkpointer，避免 SQLite 落盘状态污染。"""
    return build_app(checkpointer=InMemorySaver())


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}


def test_normal_flow_runs_to_warmup(app):
    """低风险 Mock 用户应跑完整条主流程到 warmup 阶段。"""
    thread_id = "test-normal"
    config = _config(thread_id)

    # 顺序问询的三个字段（goal/available_time/intensity_pref）均已提供
    # → interview 不会 interrupt，一次跑通
    app.invoke(
        {
            "user_id": "10001",
            "session_id": thread_id,
            "is_first_time": True,
            "goal": "fat_loss",
            "available_time": 40,
            "intensity_pref": "mid",
        },
        config=config,
    )

    # plan_review 会 interrupt 询问是否有疑问 → 回复「继续」放行
    assert app.get_state(config).next, "流程应在 plan_review 节点暂停（interrupt）"
    app.invoke(Command(resume="继续"), config=config)

    state = app.get_state(config).values
    assert state["stage"] == "warmup"
    assert state["risk_level"] == "low"
    assert state["body_energy"] >= 0
    assert state["plan"]["type"] in ("running", "indoor_bike", "walk")
    assert state["device_ready"] is True
    assert state["warmup_guide"]["steps"]  # 有热身动作


def test_high_risk_routes_to_safety_stop(app):
    """血氧<90 → forbidden；静息HR>100 → high；两者均路由到 safety_stop。

    风险规则为纯 Python（非 LLM），独立单元测试更稳定（文档要求误报率<3%/漏报率<1%）。
    """
    from sport_agent.nodes.risk import route_after_risk, risk_assess_node

    # 场景1：血氧过低 → forbidden
    state_low_spo2 = {
        "user_id": "10002",
        "vitals": {"heart_rate": 80, "spo2": 88, "hrv": 40, "stress": 30, "timestamp": ""},
    }
    result = risk_assess_node(state_low_spo2)
    assert result["risk_level"] == "forbidden"
    assert route_after_risk({**state_low_spo2, **result}) == "safety_stop"

    # 场景2：静息心率过高 → high
    state_high_hr = {
        "user_id": "10003",
        "vitals": {"heart_rate": 110, "spo2": 96, "hrv": 35, "stress": 40, "timestamp": ""},
    }
    result2 = risk_assess_node(state_high_hr)
    assert result2["risk_level"] == "high"
    assert route_after_risk({**state_high_hr, **result2}) == "safety_stop"

    # 场景3：健康 → low → feasibility
    state_healthy = {
        "user_id": "10004",
        "vitals": {"heart_rate": 72, "spo2": 98, "hrv": 55, "stress": 30, "timestamp": ""},
    }
    result3 = risk_assess_node(state_healthy)
    assert result3["risk_level"] == "low"
    assert route_after_risk({**state_healthy, **result3}) == "feasibility"


def test_interview_interrupt_and_resume(app):
    """缺字段时 interview 应按顺序 interrupt 提问，逐次 resume 后继续到 warmup。

    覆盖需求6：按顺序提问、答非所问重问、打断时要求停止播报（tts_cancel）。
    """
    thread_id = "test-interrupt"
    config = _config(thread_id)

    app.invoke(
        {
            "user_id": "10003",
            "session_id": thread_id,
            "is_first_time": True,
            "available_time": 30,
        },
        config=config,
    )

    # 应停在 interview 的第一个问题（goal）
    snapshot = app.get_state(config)
    assert snapshot.next, "流程应在 interview 节点暂停（interrupt）"
    payload = snapshot.tasks[0].interrupts[0].value
    assert payload["field"] == "goal"
    assert payload["tts_cancel"] is True, "打断时应要求客户端停止当前语音播报"

    # 答非所问 → 应对同一字段重问
    app.invoke(Command(resume="今天天气不错"), config=config)
    snapshot = app.get_state(config)
    reask = snapshot.tasks[0].interrupts[0].value
    assert reask["field"] == "goal" and reask["attempt"] == 2, "答非所问应重问同一字段"

    # 逐个回答剩余问题（available_time 已在 state 中，会被跳过）
    app.invoke(Command(resume="想减脂"), config=config)
    snapshot = app.get_state(config)
    assert snapshot.tasks[0].interrupts[0].value["field"] == "intensity_pref"

    app.invoke(Command(resume="中等强度"), config=config)

    # plan_review 的方案问询打断 → 回复「继续」放行
    app.invoke(Command(resume="继续"), config=config)

    state = app.get_state(config).values
    assert state.get("goal") == "fat_loss"
    assert state.get("intensity_pref") == "mid"
    assert state.get("tts_cancel") is False, "问询结束应恢复正常播报"
    assert state["stage"] == "warmup"


def test_memory_facts_and_prefill():
    """记忆层：事实文本按原样保留数值；偏好记忆能预填问询字段。

    事实文本不经 LLM 抽取（infer=False），因此这里断言数值原文出现在文本里，
    对应方案文档「生理指标不走对话抽取，规避 LLM 幻觉」的要求。
    """
    from sport_agent.memory import MEM_PHYSIO, MEM_PREFERENCE, format_memories
    from sport_agent.nodes.interview import _prefill_from_memory
    from sport_agent.nodes.memory import _assessment_fact, _physio_fact, _plan_fact

    state = {
        "vitals": {"metric_date": "2026-08-10", "heart_rate": 58, "hrv": 62, "spo2": 98.0},
        "body_energy": 72,
        "recovery_status": "READY",
        "risk_level": "mid",
        "risk_reason": ["静息心率偏高"],
        "plan": {"type": "running", "duration": 30, "intensity": "mid"},
        "goal": "fat_loss",
    }
    physio = _physio_fact(state)
    assert "58" in physio and "62" in physio and "98.0" in physio, "生理数值必须原样保留"
    assert "mid" in _assessment_fact(state)
    assert "running" in _plan_fact(state) and "30" in _plan_fact(state)

    # 记忆快照 → 预填问询字段（老用户不再被重复提问）
    snapshot = {
        MEM_PREFERENCE: ["用户偏好减脂类运动，希望每次 40 分钟，强度中等"],
        MEM_PHYSIO: ["静息心率基线 58 bpm"],
    }
    prefilled = _prefill_from_memory({"memory": snapshot})
    assert prefilled.get("goal") == "fat_loss"
    assert prefilled.get("available_time") == 40
    assert prefilled.get("intensity_pref") == "mid"

    # state 已有的字段优先，不被记忆覆盖
    assert "goal" not in _prefill_from_memory({"memory": snapshot, "goal": "endurance"})

    # 空记忆时给出明确提示，避免 LLM 凭空假设历史
    assert "无历史记忆" in format_memories(None)
    assert "静息心率基线" in format_memories(snapshot)


def test_route_after_avatar_new_vs_returning():
    """新老用户分流：新用户 → interview；老用户 → plan_generate。"""
    from sport_agent.nodes.interview import route_after_avatar

    # 新用户 → interview
    assert route_after_avatar({"is_first_time": True}) == "interview"
    # 缺省（未指定）→ 视为新用户 → interview
    assert route_after_avatar({}) == "interview"

    # 老用户 → 直接出方案
    assert route_after_avatar({"is_first_time": False}) == "plan_generate"
    # 老用户 + 常用地点 → 直接出方案
    assert (
        route_after_avatar({"is_first_time": False, "location": {"is_known_place": True}})
        == "plan_generate"
    )
    # 老用户 + 非常用地点 → 仍需问询（需求6：非常用地需问清场景）
    assert (
        route_after_avatar(
            {"is_first_time": False, "location": {"is_known_place": False, "name": "外滩"}}
        )
        == "interview"
    )
