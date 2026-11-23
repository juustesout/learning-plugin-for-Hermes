from __future__ import annotations

import json
from pathlib import Path

import hermes_plugin
import opslog


class FakeCtx:
    def __init__(self, with_cli: bool = True):
        self.hooks = []
        self.cli = []
        self.middleware = []
        if not with_cli:
            return

        def register_cli_command(*, name, help, setup_fn, handler_fn):
            self.cli.append(
                {
                    "name": name,
                    "help": help,
                    "setup_fn": setup_fn,
                    "handler_fn": handler_fn,
                }
            )

        self.register_cli_command = register_cli_command

    def register_hook(self, name, fn):
        self.hooks.append((name, fn))

    def register_middleware(self, kind, fn):
        self.middleware.append((kind, fn))


def test_register_hook_and_cli():
    ctx = FakeCtx()
    hermes_plugin.register(ctx)
    assert ctx.hooks == [
        ("pre_tool_call", hermes_plugin.on_pre_tool_call),
        ("post_tool_call", hermes_plugin.on_post_tool_call),
        ("post_api_request", hermes_plugin.on_post_api_request),
        ("pre_llm_call", hermes_plugin.on_pre_llm_call),
    ]
    assert len(ctx.cli) == 1
    assert ctx.cli[0]["name"] == "agent-learning"
    assert ctx.middleware == [
        ("llm_execution", hermes_plugin.on_llm_execution_middleware),
    ]


def test_register_without_cli_command():
    ctx = FakeCtx(with_cli=False)
    hermes_plugin.register(ctx)
    assert ctx.hooks[0][0] == "pre_tool_call"
    assert ctx.hooks[1][0] == "post_tool_call"
    assert ctx.hooks[2][0] == "post_api_request"
    assert ctx.hooks[3][0] == "pre_llm_call"
    assert ctx.cli == []
    assert ctx.middleware == [("llm_execution", hermes_plugin.on_llm_execution_middleware)]


def test_generation_ids_flow_from_post_api_request_to_post_tool_call():
    hermes_plugin._GENERATION_IDS_BY_API_REQUEST.clear()
    hermes_plugin.on_post_api_request(
        api_request_id="turn:1:api:1",
        response_headers={"x-openrouter-generation-id": "gen-abc123"},
    )
    payload = hermes_plugin.build_operation(
        tool_name="terminal",
        status="ok",
        api_request_id="turn:1:api:1",
    )
    assert payload["generation_ids"] == ["gen-abc123"]
    assert payload["metadata"]["generation_ids"] == ["gen-abc123"]


def test_llm_middleware_is_kept_for_generation_id_cost_attribution():
    ctx = FakeCtx()
    hermes_plugin.register(ctx)
    assert ctx.middleware == [
        ("llm_execution", hermes_plugin.on_llm_execution_middleware),
    ]
    hermes_plugin._GENERATION_IDS_BY_API_REQUEST.clear()

    def _next_call(_request):
        return {"id": "gen-cost-1", "usage": {"total_tokens": 12}}

    response = hermes_plugin.on_llm_execution_middleware(
        {"messages": []},
        _next_call,
        api_request_id="turn:cost:api:1",
    )
    assert response == {"id": "gen-cost-1", "usage": {"total_tokens": 12}}
    payload = hermes_plugin.build_operation(
        tool_name="terminal",
        status="ok",
        api_request_id="turn:cost:api:1",
    )
    assert payload["generation_ids"] == ["gen-cost-1"]


def test_middleware_kill_switch_is_noop_and_byte_identical():
    hermes_plugin._GENERATION_IDS_BY_API_REQUEST.clear()
    request = {"messages": [{"role": "user", "content": "x"}]}
    expected = {"id": "gen-kill-1", "usage": {"total_tokens": 3}}

    def _next_call(payload):
        assert payload is request
        return dict(expected)

    import os

    old = os.environ.get(hermes_plugin.MIDDLEWARE_KILL_ENV)
    os.environ[hermes_plugin.MIDDLEWARE_KILL_ENV] = "off"
    try:
        response = hermes_plugin.on_llm_execution_middleware(
            request,
            _next_call,
            api_request_id="turn:kill:api:1",
        )
    finally:
        if old is None:
            os.environ.pop(hermes_plugin.MIDDLEWARE_KILL_ENV, None)
        else:
            os.environ[hermes_plugin.MIDDLEWARE_KILL_ENV] = old
    assert response == expected
    payload = hermes_plugin.build_operation(
        tool_name="terminal",
        status="ok",
        api_request_id="turn:kill:api:1",
    )
    assert not payload["generation_ids"]


def test_middleware_response_byte_identical_when_attribution_fails():
    hermes_plugin._GENERATION_IDS_BY_API_REQUEST.clear()
    expected = {"id": "gen-fail-1"}

    def _next_call(_request):
        return expected

    def boom(*_args, **_kwargs):
        raise RuntimeError("extract failed")

    original = hermes_plugin._extract_generation_ids_from_llm_response
    hermes_plugin._extract_generation_ids_from_llm_response = boom
    try:
        response = hermes_plugin.on_llm_execution_middleware(
            {"messages": []},
            _next_call,
            api_request_id="turn:fail:api:1",
        )
    finally:
        hermes_plugin._extract_generation_ids_from_llm_response = original
    assert response is expected


def test_generation_ids_flow_from_llm_middleware_to_post_tool_call():
    hermes_plugin._GENERATION_IDS_BY_API_REQUEST.clear()

    def _next_call(_request):
        return {"id": "gen-mid-1"}

    response = hermes_plugin.on_llm_execution_middleware(
        {"messages": []},
        _next_call,
        api_request_id="turn:2:api:9",
    )
    assert response == {"id": "gen-mid-1"}
    payload = hermes_plugin.build_operation(
        tool_name="terminal",
        status="ok",
        api_request_id="turn:2:api:9",
    )
    assert payload["generation_ids"] == ["gen-mid-1"]
    assert payload["metadata"]["generation_ids"] == ["gen-mid-1"]


def test_hook_writes_top_level_generation_ids(tmp_path, monkeypatch):
    log_path = tmp_path / "operations.jsonl"
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(log_path))
    hermes_plugin._GENERATION_IDS_BY_API_REQUEST.clear()
    hermes_plugin.on_post_api_request(
        api_request_id="turn:3:api:1",
        response_headers={"x-openrouter-generation-id": "gen-top-1"},
    )
    hermes_plugin.on_post_tool_call(
        tool_name="web_search",
        status="ok",
        api_request_id="turn:3:api:1",
    )
    entry = json.loads(log_path.read_text().strip())
    assert entry["generation_ids"] == ["gen-top-1"]
    assert entry["metadata"]["generation_ids"] == ["gen-top-1"]


def test_build_operation_maps_web_extract():
    payload = hermes_plugin.build_operation(
        tool_name="web_extract",
        args={"url": "https://example.com/page"},
        status="ok",
        duration_ms=12.9,
        session_id="s1",
        turn_id="t1",
        tool_call_id="c1",
    )
    assert payload["task_id"] == "choose-scrape-strategy"
    assert payload["action_id"] == "web_extract"
    assert payload["target"] == "https://example.com/page"
    assert payload["success"] is True
    assert payload["duration_ms"] == 12
    assert payload["metadata"]["tool_ok"] is True
    assert payload["metadata"]["task_success"] is False
    assert payload["context_features"]["url_host_overig"] == 1.0
    assert "cloudflare" not in payload["context_features"]
    assert "session_id" not in payload["metadata"]
    assert "result" not in payload
    assert "args" not in payload


def test_build_operation_maps_webclaw_scrape():
    payload = hermes_plugin.build_operation(
        tool_name="mcp_webclaw_webclaw_scrape",
        args={"url": "https://agentcursus.nl/page"},
        status="ok",
    )
    assert payload["task_id"] == "choose-scrape-strategy"
    assert payload["action_id"] == "mcp_webclaw_webclaw_scrape"
    assert payload["target"] == "https://agentcursus.nl/page"
    assert payload["success"] is True


def test_timeout_is_failure():
    payload = hermes_plugin.build_operation(
        tool_name="mcp_webclaw_webclaw_scrape",
        args={"url": "https://example.com/slow"},
        status="timeout",
    )
    assert payload["task_id"] == "choose-scrape-strategy"
    assert payload["success"] is False
    assert payload["metadata"]["hermes_status"] == "timeout"
    assert payload["metadata"]["tool_ok"] is False


def test_build_operation_skips_empty_tool():
    assert hermes_plugin.build_operation(tool_name="  ") is None


def test_blocked_is_failure():
    payload = hermes_plugin.build_operation(
        tool_name="terminal",
        status="blocked",
        error_message="denied",
    )
    assert payload["task_id"] == "hermes-tool"
    assert payload["success"] is False
    assert "denied" in payload["summary"]


def test_extract_target_ignores_command():
    assert hermes_plugin.extract_target({"command": "rm -rf /", "query": "q"}) == "q"
    assert hermes_plugin.extract_target("not-a-dict") == ""


def test_hook_writes_opslog(tmp_path, monkeypatch):
    log_path = tmp_path / "operations.jsonl"
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(log_path))
    hermes_plugin.on_post_tool_call(
        tool_name="web_search",
        args={"query": "bandits"},
        status="ok",
        duration_ms=40,
        result={"hits": 9},
    )
    entry = json.loads(log_path.read_text().strip())
    assert entry["task_id"] == "web-research"
    assert entry["action_id"] == "web_search"
    assert entry["target"] == "bandits"
    assert entry["execution_status"] == "completed"
    assert "hits" not in json.dumps(entry)


def test_hook_fail_open(tmp_path, monkeypatch):
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(tmp_path / "operations.jsonl"))

    def boom(**kwargs):
        raise RuntimeError("store down")

    monkeypatch.setattr(opslog, "log_operation", boom)
    hermes_plugin.on_post_tool_call(tool_name="web_search", status="ok")
    h = opslog.health()
    assert h["ok"] is False
    assert h["failed"] >= 1


def test_opt_out_skips_write(tmp_path, monkeypatch):
    log_path = tmp_path / "operations.jsonl"
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(log_path))
    monkeypatch.setenv("AGENT_LEARNING_OPSLOG_ENABLED", "0")
    hermes_plugin.on_post_tool_call(tool_name="web_search", status="ok")
    assert not log_path.exists()


def test_session_ids_opt_in(monkeypatch):
    monkeypatch.setenv("AGENT_LEARNING_OPSLOG_SESSION_IDS", "1")
    payload = hermes_plugin.build_operation(
        tool_name="web_search",
        status="ok",
        session_id="s1",
        turn_id="t1",
        tool_call_id="c1",
    )
    assert payload["metadata"]["session_id"] == "s1"


def test_tool_task_map_env(monkeypatch):
    monkeypatch.setenv(
        "AGENT_LEARNING_TOOL_TASK_MAP",
        '{"terminal":"db-query","web_search":"custom-research"}',
    )
    assert hermes_plugin.task_id_for_tool("terminal") == "db-query"
    assert hermes_plugin.task_id_for_tool("web_search") == "custom-research"


def test_redacts_query_and_token():
    payload = hermes_plugin.build_operation(
        tool_name="web_extract",
        args={"url": "https://example.com/x?token=abc&api_key=secret"},
        status="error",
        error_message="Authorization: Bearer abcdef.fail",
    )
    assert "token=abc" not in payload["target"]
    assert "secret" not in payload["target"]
    assert "Bearer ***" in payload["summary"] or "***" in payload["summary"]


def test_kernel_roots_honor_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_LEARNING_REPO", str(tmp_path))
    roots = hermes_plugin._kernel_roots()
    assert roots[0] == tmp_path


def test_repo_root_plugin_layout():
    root = Path(__file__).resolve().parents[1]
    assert (root / "plugin.yaml").is_file()
    assert (root / "__init__.py").is_file()
    text = (root / "plugin.yaml").read_text()
    assert "pre_tool_call" in text
    assert "post_tool_call" in text
    assert "name: agent-learning" in text
