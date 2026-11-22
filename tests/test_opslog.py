from __future__ import annotations

import json
from pathlib import Path

import opslog


def test_log_operation_writes_jsonl(tmp_path, monkeypatch):
    log_path = tmp_path / "operations.jsonl"
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(log_path))

    oid = opslog.log_operation(
        task_id="choose-scrape-strategy",
        action_id="use_webclaw_scrape",
        target="example.com",
        success=True,
        duration_ms=3100,
        cost=1.0,
        summary="ok",
        action_logprob=-0.4,
        context_features={"cloudflare": 1.0},
    )
    assert oid
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["id"] == oid
    assert entry["task_id"] == "choose-scrape-strategy"
    assert entry["action_id"] == "use_webclaw_scrape"
    assert entry["action_logprob"] == -0.4
    assert entry["context_features"]["cloudflare"] == 1.0
    assert entry["execution_status"] == "completed"
    assert entry["metadata"]["cost_credits"] == 1.0
    assert entry["request_latency_ms"] == 3100


def test_log_operation_writes_generation_ids_top_level(tmp_path, monkeypatch):
    log_path = tmp_path / "operations.jsonl"
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(log_path))
    opslog.log_operation(
        task_id="llm",
        action_id="mcp_openrouter_send_message",
        generation_ids=["gen-1", "gen-2", "gen-1", ""],
    )
    entry = json.loads(log_path.read_text().strip())
    assert entry["generation_ids"] == ["gen-1", "gen-2"]


def test_log_operation_accepts_operational_context(tmp_path, monkeypatch):
    log_path = tmp_path / "operations.jsonl"
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(log_path))
    opslog.log_operation(
        task_id="hn-curation",
        action_id="vector-filter",
        work_unit="hn-curate-and-publish",
        trigger="scheduled",
        execution_mode="workflow",
        parent_run_id="parent-run",
        skills_involved=["vector-matching", "web-research"],
    )
    metadata = json.loads(log_path.read_text().strip())["metadata"]
    assert metadata["work_unit"] == "hn-curate-and-publish"
    assert metadata["trigger"] == "scheduled"
    assert metadata["execution_mode"] == "workflow"
    assert metadata["parent_run_id"] == "parent-run"
    assert metadata["skills_involved"] == ["vector-matching", "web-research"]


def test_log_failure_status(tmp_path, monkeypatch):
    log_path = tmp_path / "operations.jsonl"
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(log_path))
    opslog.log_operation(task_id="t", action_id="a", success=False)
    entry = json.loads(log_path.read_text().strip())
    assert entry["execution_status"] == "failed"
    assert entry["metadata"]["task_completed"] is False


def test_count_tail_stats(tmp_path, monkeypatch):
    log_path = tmp_path / "operations.jsonl"
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(log_path))
    assert opslog.count_operations() == 0
    assert opslog.tail() == []
    assert opslog.stats()["total"] == 0

    opslog.log_operation(task_id="t1", action_id="a", duration_ms=10, success=True)
    opslog.log_operation(task_id="t1", action_id="b", duration_ms=20, success=False)
    assert opslog.count_operations() == 2
    s = opslog.stats()
    assert s["total"] == 2
    assert s["by_task"]["t1"] == 2
    assert s["by_status"]["completed"] == 1
    assert s["by_status"]["failed"] == 1
    assert s["avg_duration_ms"] == 15.0
    assert len(opslog.tail(1)) == 1


def test_cli_log_with_logprob_and_context(tmp_path, monkeypatch, capsys):
    log_path = tmp_path / "operations.jsonl"
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(log_path))
    rc = opslog.main([
        "log",
        "--task-id", "choose-scrape-strategy",
        "--action-id", "use_jina_reader",
        "--target", "site.nl",
        "--success", "false",
        "--duration-ms", "800",
        "--cost", "0",
        "--summary", "Jina failed",
        "--logprob", "-0.3",
        "--context-features", '{"cloudflare":1.0}',
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("ok ")
    entry = json.loads(Path(log_path).read_text().strip())
    assert entry["action_logprob"] == -0.3
    assert entry["context_features"] == {"cloudflare": 1.0}
    assert entry["execution_status"] == "failed"


def test_cli_invalid_context_features(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(tmp_path / "operations.jsonl"))
    rc = opslog.main([
        "log",
        "--task-id", "t",
        "--action-id", "a",
        "--context-features", "not-json",
    ])
    assert rc == 1


def test_cli_count_stats(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(tmp_path / "operations.jsonl"))
    opslog.log_operation(task_id="t", action_id="a")
    assert opslog.main(["count"]) == 0
    assert capsys.readouterr().out.strip() == "1"
    assert opslog.main(["stats"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 1
    assert "health" in payload


def test_default_store_dir_prefers_opslog_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_LEARNING_OPSLOG_DIR", str(tmp_path / "ops"))
    monkeypatch.setenv("AGENT_LEARNING_LOCAL_STORE_DIR", str(tmp_path / "store"))
    assert opslog.default_store_dir() == str(tmp_path / "ops")


def test_redact_target_strips_query_and_secrets():
    assert "token=" not in opslog.redact_target("https://ex.com/a?token=abc&x=1")
    assert opslog.redact_text("api_key=secret Bearer abcdef") != "api_key=secret Bearer abcdef"
    assert "***" in opslog.redact_text("Authorization: Bearer abcdef")


def test_log_redacts_target_and_summary(tmp_path, monkeypatch):
    log_path = tmp_path / "operations.jsonl"
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(log_path))
    opslog.log_operation(
        task_id="t",
        action_id="a",
        target="https://ex.com/page?api_key=secret",
        summary="token=abcd failure",
    )
    entry = json.loads(log_path.read_text().strip())
    assert "secret" not in entry["target"]
    assert "api_key=" not in entry["target"] or "***" in entry["target"]
    assert "abcd" not in entry["result_summary"]


def test_health_and_prune(tmp_path, monkeypatch):
    log_path = tmp_path / "operations.jsonl"
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(log_path))
    opslog.log_operation(task_id="t", action_id="a")
    h = opslog.health()
    assert h["ok"] is True
    assert h["written"] >= 1
    old = {
        "id": "old-1",
        "timestamp": "2000-01-01T00:00:00+00:00",
        "task_id": "t",
        "action_id": "a",
        "execution_status": "completed",
    }
    new = {
        "id": "new-1",
        "timestamp": "2099-01-01T00:00:00+00:00",
        "task_id": "t",
        "action_id": "a",
        "execution_status": "completed",
    }
    log_path.write_text(json.dumps(old) + "\n" + json.dumps(new) + "\n")
    dropped = opslog.prune_old(1)
    assert dropped == 1
    remaining = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    assert remaining == [new]


def test_cli_health(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(tmp_path / "operations.jsonl"))
    assert opslog.main(["health"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "written" in payload


def test_no_hardcoded_opt_store():
    import inspect
    import consolidate_ops

    assert "/opt/data/profiles/operator/agent-learning/store" not in inspect.getsource(opslog)
    assert "/opt/data/profiles/operator/agent-learning/store" not in inspect.getsource(consolidate_ops)
