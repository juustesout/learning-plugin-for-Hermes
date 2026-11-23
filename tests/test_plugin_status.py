from __future__ import annotations

from pathlib import Path

import hermes_plugin
import opslog
import plugin_status
import shadow_policy
from tests.test_hermes_plugin import FakeCtx


def _bind(tmp_path, monkeypatch):
    store = tmp_path / "store"
    store.mkdir()
    monkeypatch.setattr(opslog, "STORE_DIR", str(store))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(store / "operations.jsonl"))
    monkeypatch.setenv("AGENT_LEARNING_OPSLOG_DIR", str(store))
    monkeypatch.setenv("AGENT_LEARNING_LOCAL_STORE_DIR", str(store))
    shadow_policy.reset_cache()
    return store


def test_unregistered_hooks_are_unknown_not_no(tmp_path, monkeypatch):
    _bind(tmp_path, monkeypatch)
    for name in list(hermes_plugin.HOOKS_REGISTERED):
        hermes_plugin.HOOKS_REGISTERED[name] = None
    text = plugin_status.format_status()
    assert "pre_tool_call: unknown" in text
    assert "post_tool_call: unknown" in text
    assert "pre_llm_call: unknown" in text
    assert "llm_execution_middleware: unknown" in text
    assert "pre_tool_call: no" not in text


def test_no_only_when_hook_is_actually_false(tmp_path, monkeypatch):
    _bind(tmp_path, monkeypatch)
    for name in list(hermes_plugin.HOOKS_REGISTERED):
        hermes_plugin.HOOKS_REGISTERED[name] = None
    hermes_plugin.HOOKS_REGISTERED["pre_tool_call"] = False
    text = plugin_status.format_status()
    assert "pre_tool_call: no" in text
    assert "post_tool_call: unknown" in text


def test_status_is_plain_text_not_json(tmp_path, monkeypatch):
    _bind(tmp_path, monkeypatch)
    text = plugin_status.format_status()
    assert not text.lstrip().startswith("{")
    assert "hooks:" in text
    assert "pre_tool_call:" in text
    assert "post_tool_call:" in text
    assert "pre_llm_call:" in text
    assert "import_source:" in text
    assert "eligible_scrape:" in text
    assert "consolidate:" in text
    assert "gate:" in text


def test_status_shows_hooks_after_register(tmp_path, monkeypatch):
    _bind(tmp_path, monkeypatch)
    hermes_plugin.HOOKS_REGISTERED["pre_tool_call"] = False
    hermes_plugin.HOOKS_REGISTERED["post_tool_call"] = False
    hermes_plugin.HOOKS_REGISTERED["pre_llm_call"] = False
    ctx = FakeCtx()
    hermes_plugin.register(ctx)
    text = plugin_status.format_status()
    assert "pre_tool_call: yes" in text
    assert "post_tool_call: yes" in text
    assert "llm_execution_middleware: yes" in text


def test_status_cli_command_registered():
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
    assert "status" in parser.subs.names
    assert "selftest" in parser.subs.names
    assert "pool" in parser.subs.names
    assert "select" in parser.subs.names
    assert "baseline" in parser.subs.names
    assert "record-post" in parser.subs.names


def test_selftest_does_not_write_live_store(tmp_path, monkeypatch):
    live = _bind(tmp_path, monkeypatch)
    live_log = live / "operations.jsonl"
    live_log.write_text('{"id":"keep-me"}\n', encoding="utf-8")
    before = live_log.read_text(encoding="utf-8")
    result = plugin_status.run_selftest()
    assert result["checks"]["live_store_untouched"] is True
    assert live_log.read_text(encoding="utf-8") == before
    assert Path(result["temp_store"]).resolve() != live.resolve()
    assert result["ok"] is True
    assert result["checks"]["hook_returns_none"] is True
    assert result["checks"]["one_new_operation"] is True
    assert result["checks"]["task_id_filled"] is True
    assert result["checks"]["import_source_filled"] is True
    assert result["checks"]["context_features_pre_call"] is True
    assert result["checks"]["import_failure_health"] is True


def test_selftest_format_is_plain_text(tmp_path, monkeypatch):
    _bind(tmp_path, monkeypatch)
    text = plugin_status.format_selftest()
    assert text.startswith("selftest:")
    assert "live_store_untouched:" in text
    assert "{" not in text.split("\n", 1)[0]
