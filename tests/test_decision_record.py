from __future__ import annotations

import json
from pathlib import Path

import decision_record
import hermes_plugin
import opslog
import shadow_policy
from tests.test_hermes_plugin import FakeCtx
from tests.test_shadow_policy import _scrape_snapshot, _write_snapshot


def _bind(tmp_path, monkeypatch):
    store = tmp_path / "store"
    store.mkdir()
    monkeypatch.setattr(opslog, "STORE_DIR", str(store))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(store / "operations.jsonl"))
    monkeypatch.setenv("AGENT_LEARNING_OPSLOG_DIR", str(store))
    monkeypatch.setenv("AGENT_LEARNING_LOCAL_STORE_DIR", str(store))
    monkeypatch.delenv("AGENT_LEARNING_DECISION_RECORD", raising=False)
    monkeypatch.delenv("AGENT_LEARNING_SHADOW_KILL", raising=False)
    shadow_policy.reset_cache()
    return store


def test_one_record_per_join_key(tmp_path, monkeypatch):
    _bind(tmp_path, monkeypatch)
    first = decision_record.open_decision(
        join_key_value="s1|c1",
        requested_tool="web_extract",
        context_features={"url_has_query": 1.0},
    )
    second = decision_record.open_decision(
        join_key_value="s1|c1",
        requested_tool="web_extract",
        context_features={"url_has_query": 0.0},
    )
    assert first["decision_id"] == second["decision_id"]
    rows = [
        json.loads(line)
        for line in Path(decision_record.log_path()).read_text().splitlines()
        if line.strip()
    ]
    opens = [row for row in rows if row.get("record") == decision_record.RECORD]
    assert len(opens) == 1


def test_pre_call_fields_drop_html_and_headers():
    cleaned = decision_record.pre_call_features(
        {
            "url_has_query": 1,
            "html": "<p>secret</p>",
            "headers": {"Authorization": "Bearer x"},
            "content": "body",
            "note": "text",
        }
    )
    assert cleaned == {"url_has_query": 1.0}
    assert "html" not in cleaned
    assert "headers" not in cleaned


def test_outcome_without_record_is_orphan_not_added(tmp_path, monkeypatch):
    store = _bind(tmp_path, monkeypatch)
    result = decision_record.record_outcome_event(
        decision_id="missing-id",
        level=3,
        status="ok",
    )
    assert result["ok"] is False
    assert result["orphan"] is True
    assert decision_record.load_decisions() == []
    orphans = [
        json.loads(line)
        for line in (store / "decision_orphans.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(orphans) == 1
    assert orphans[0]["reason"] == "unknown_decision"
    assert not Path(decision_record.log_path()).exists()


def test_record_outcome_validates_and_late_confirm_joins(tmp_path, monkeypatch):
    _bind(tmp_path, monkeypatch)
    opened = decision_record.open_decision(
        join_key_value="s1|c9",
        requested_tool="web_extract",
        advised_tool="use_jina_reader",
        context_features={"url_path_depth": 2.0},
    )
    did = opened["decision_id"]
    first = decision_record.record_outcome(decision_id=did, level="confirmed", status="ok")
    assert first["ok"] is True
    assert first["changed"] is True
    row = decision_record.get_by_id(did)
    assert row["outcomes"]["confirmed"]["status"] == "ok"
    duplicate = decision_record.record_outcome(decision_id=did, level=3, status="error")
    assert duplicate["ok"] is True
    assert duplicate["changed"] is False
    row = decision_record.get_by_id(did)
    assert row["outcomes"]["confirmed"]["status"] == "ok"


def test_record_outcome_rejects_bad_level(tmp_path, monkeypatch):
    _bind(tmp_path, monkeypatch)
    opened = decision_record.open_decision(
        join_key_value="s|c",
        requested_tool="web_extract",
    )
    try:
        decision_record.record_outcome(
            decision_id=opened["decision_id"],
            level="nope",
            status="ok",
        )
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_hooks_write_additive_decision_without_changing_shadow(tmp_path, monkeypatch):
    store = _bind(tmp_path, monkeypatch)
    _write_snapshot(store, _scrape_snapshot())
    assert hermes_plugin.on_pre_tool_call(
        tool_name="web_extract",
        args={"url": "https://example.com"},
        session_id="s1",
        tool_call_id="c9",
    ) is None
    hermes_plugin.on_post_tool_call(
        tool_name="web_extract",
        status="ok",
        duration_ms=41,
        session_id="s1",
        tool_call_id="c9",
    )
    shadow_rows = shadow_policy.load_decisions()
    assert len(shadow_rows) == 1
    decisions = decision_record.load_decisions()
    assert len(decisions) == 1
    row = decisions[0]
    assert row["record"] == "decision.v1"
    assert row["join_key"] == "s1|c9"
    assert row["requested_tool"] == "web_extract"
    assert row["chosen_tool"] == "web_extract"
    assert row["outcomes"]["technical"]["status"] == "ok"
    assert "html" not in row["context_features"]
    ops = [
        json.loads(line)
        for line in (store / "operations.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(ops) == 1
    assert ops[0].get("record") != "decision.v1"


def test_kill_switch_skips_decision_record(tmp_path, monkeypatch):
    store = _bind(tmp_path, monkeypatch)
    monkeypatch.setenv("AGENT_LEARNING_DECISION_RECORD", "off")
    _write_snapshot(store, _scrape_snapshot())
    hermes_plugin.on_pre_tool_call(
        tool_name="web_extract",
        args={"url": "https://example.com"},
        session_id="s1",
        tool_call_id="c1",
    )
    hermes_plugin.on_post_tool_call(
        tool_name="web_extract",
        status="ok",
        session_id="s1",
        tool_call_id="c1",
    )
    assert not Path(decision_record.log_path()).exists()
    assert shadow_policy.load_decisions()


def test_record_outcome_cli_registered():
    ctx = FakeCtx()
    hermes_plugin.register(ctx)
    setup = ctx.cli[0]["setup_fn"]

    class Sub:
        def __init__(self):
            self.names = []

        def add_parser(self, name, **kwargs):
            self.names.append(name)
            return self

        def add_subparsers(self, dest=None):
            return self

        def add_argument(self, *args, **kwargs):
            return self

        def set_defaults(self, **kwargs):
            return self

    class Parser:
        def add_subparsers(self, dest=None):
            self.subs = Sub()
            return self.subs

        def set_defaults(self, **kwargs):
            return self

    parser = Parser()
    setup(parser)
    assert "record-outcome" in parser.subs.names
