from __future__ import annotations

import json
from pathlib import Path

import advice_hook
import hermes_plugin
import opslog
from tests.test_hermes_plugin import FakeCtx


def _bind(tmp_path, monkeypatch):
    store = tmp_path / "store"
    store.mkdir()
    monkeypatch.setattr(opslog, "STORE_DIR", str(store))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(store / "operations.jsonl"))
    monkeypatch.setenv("AGENT_LEARNING_OPSLOG_DIR", str(store))
    monkeypatch.setenv("AGENT_LEARNING_LOCAL_STORE_DIR", str(store))
    monkeypatch.delenv(advice_hook.KILL_ENV, raising=False)
    return store


OPEN_GATE = {
    "open": True,
    "reason": "-",
    "n": 40,
    "min_n": 12,
}
ADVICE = {
    "task_id": "choose-scrape-strategy",
    "recommended_action": "use_jina_reader",
    "options": {
        "use_jina_reader": {"n": 20, "wilson_low": 0.4, "median_latency_ms": 120},
        "use_browser": {"n": 20, "wilson_low": 0.2, "median_latency_ms": 800},
    },
}


def test_kill_switch_returns_original_messages(tmp_path, monkeypatch):
    _bind(tmp_path, monkeypatch)
    monkeypatch.setenv(advice_hook.KILL_ENV, "off")
    messages = [{"role": "user", "content": "hello"}]
    out = advice_hook.apply_advice(messages, gate=OPEN_GATE, advice=ADVICE, session_id="s", turn_id="t")
    assert out is messages
    assert out == [{"role": "user", "content": "hello"}]


def test_arm_b_is_byte_identical(tmp_path, monkeypatch):
    _bind(tmp_path, monkeypatch)
    messages = [{"role": "user", "content": "hello"}]
    original = json.dumps(messages)
    found_b = False
    for i in range(40):
        arm = advice_hook.assign_arm("s", str(i), "api")
        if arm != "B":
            continue
        found_b = True
        out = advice_hook.apply_advice(
            messages,
            gate=OPEN_GATE,
            advice=ADVICE,
            session_id="s",
            turn_id=str(i),
            api_request_id="api",
        )
        assert out is messages
        assert json.dumps(out) == original
    assert found_b


def test_arm_l_injects_only_when_gate_open(tmp_path, monkeypatch):
    _bind(tmp_path, monkeypatch)
    messages = [{"role": "user", "content": "hello"}]
    closed = {"open": False, "reason": "n<12", "n": 1, "min_n": 12}
    assert closed["open"] is False
    arm_l_turn = None
    for i in range(40):
        if advice_hook.assign_arm("s", str(i), "api") == "L":
            arm_l_turn = str(i)
            break
    assert arm_l_turn is not None
    unchanged = advice_hook.apply_advice(
        messages, gate=closed, advice=ADVICE, session_id="s", turn_id=arm_l_turn, api_request_id="api"
    )
    assert unchanged is messages
    changed = advice_hook.apply_advice(
        messages, gate=OPEN_GATE, advice=ADVICE, session_id="s", turn_id=arm_l_turn, api_request_id="api"
    )
    assert changed is not messages
    assert changed[0] == messages[0]
    assert advice_hook.ADVICE_MARKER in changed[-1]["content"]
    assert "use_jina_reader" in changed[-1]["content"]


def _legacy_assign_after_gate(messages, *, gate, advice, session_id, turn_id, api_request_id, store: Path):
    """Old deadlock: skip assignment when the gate is closed. Negative control only."""
    if not (isinstance(gate, dict) and gate.get("open")):
        return messages
    arm = advice_hook.assign_arm(session_id, turn_id, api_request_id)
    if arm != "L" or not advice:
        return messages
    return advice_hook.inject_advice(messages, advice)


def test_closed_gate_still_assigns_arm_l_without_injecting(tmp_path, monkeypatch):
    store = _bind(tmp_path, monkeypatch)
    messages = [{"role": "user", "content": "hello"}]
    original = json.dumps(messages)
    closed = {"open": False, "reason": "n<12", "n": 1, "min_n": 12}
    arm_l_turn = None
    for i in range(80):
        if advice_hook.assign_arm("s", str(i), "api") == "L":
            arm_l_turn = str(i)
            break
    assert arm_l_turn is not None
    out = advice_hook.apply_advice(
        messages,
        gate=closed,
        advice=ADVICE,
        session_id="s",
        turn_id=arm_l_turn,
        api_request_id="api",
    )
    assert out is messages
    assert json.dumps(out) == original
    assert advice_hook.ADVICE_MARKER not in json.dumps(out)
    rows = []
    log = Path(store / "advice_calls.jsonl")
    if log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    arms = [row.get("arm") for row in rows]
    assert "L" in arms
    assert all(row.get("injected") is False for row in rows)
    legacy = _legacy_assign_after_gate(
        messages,
        gate=closed,
        advice=ADVICE,
        session_id="s",
        turn_id=arm_l_turn,
        api_request_id="api",
        store=store,
    )
    assert json.dumps(legacy) == original
    assert not (store / "legacy-would-have-logged.jsonl").exists()


def test_fail_open_on_exception(tmp_path, monkeypatch):
    _bind(tmp_path, monkeypatch)
    messages = [{"role": "user", "content": "hello"}]

    def boom(*_args, **_kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(advice_hook, "assign_arm", boom)
    out = advice_hook.apply_advice(messages, gate=OPEN_GATE, advice=ADVICE)
    assert out is messages


def test_pre_llm_call_registered_and_pre_tool_call_still_none():
    ctx = FakeCtx()
    hermes_plugin.register(ctx)
    names = [name for name, _fn in ctx.hooks]
    assert "pre_llm_call" in names
    assert ctx.hooks[0][0] == "pre_tool_call"
    assert hermes_plugin.on_pre_tool_call(tool_name="web_extract") is None
    messages = [{"role": "user", "content": "x"}]
    assert hermes_plugin.on_pre_llm_call(messages) is messages or isinstance(
        hermes_plugin.on_pre_llm_call(messages), list
    )
