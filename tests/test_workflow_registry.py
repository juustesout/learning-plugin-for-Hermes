from __future__ import annotations

import json

import workflow_manifest
import workflow_registry
from workflow_identity import article_id, canonical_url, operation_context


def test_canonical_article_identity_collapses_tracking_and_trailing_slash():
    first = article_id("https://example.com/story/?utm_source=hn", hn_id=1, title="A")
    second = article_id("https://EXAMPLE.com/story?fbclid=abc", hn_id=2, title="B")
    assert canonical_url("https://EXAMPLE.com/story/?utm_source=hn") == "https://example.com/story"
    assert first == second


def test_operation_context_only_contains_explicit_or_environment_values(monkeypatch):
    monkeypatch.delenv("AGENT_LEARNING_WORK_UNIT", raising=False)
    assert operation_context() == {}
    context = operation_context(
        work_unit="hn-curation",
        trigger="scheduled",
        execution_mode="workflow",
        parent_run_id="parent-1",
        skills_involved=["writing", "writing", "research"],
    )
    assert context == {
        "work_unit": "hn-curation",
        "trigger": "scheduled",
        "execution_mode": "workflow",
        "parent_run_id": "parent-1",
        "skills_involved": ["research", "writing"],
    }


def test_bundled_manifests_validate():
    manifests = workflow_manifest.load_manifests()
    assert {row["workflow_id"] for row in manifests} == {"sloppy-hn", "blind-label-ui"}
    for manifest in manifests:
        step_ids = {step["id"] for step in manifest["steps"]}
        assert all(set(step["depends_on"]) <= step_ids for step in manifest["steps"])


def test_registry_status_uses_manifest_and_observed_logs(tmp_path):
    manifests_dir = tmp_path / "workflows"
    manifests_dir.mkdir()
    manifest = {
        "manifest_version": 1,
        "workflow_id": "demo-workflow",
        "version": "1.0.0",
        "name": "Demo",
        "owner": "juust",
        "work_unit": "demo",
        "trigger": "scheduled",
        "execution_mode": "workflow",
        "automation": None,
        "inputs": [],
        "outputs": ["result"],
        "steps": [{"id": "run", "kind": "transform", "depends_on": [], "skills": [], "tools": []}],
    }
    (manifests_dir / "demo.json").write_text(json.dumps(manifest), encoding="utf-8")
    store = tmp_path / "store"
    store.mkdir()
    (store / "operations.jsonl").write_text(
        json.dumps({
            "id": "op-1",
            "execution_status": "completed",
            "action_id": "demo_tool",
            "metadata": {"work_unit": "demo", "cost_credits": 2.5},
        }) + "\n",
        encoding="utf-8",
    )
    payload = workflow_registry.registry(manifests_dir, store)
    item = payload["workflows"][0]
    assert item["last_status"] == "unknown"
    assert item["operations"] == {
        "operations": 1,
        "completed": 1,
        "failed": 0,
        "cost_credits": 2.5,
        "tools": {"demo_tool": 1},
    }
    assert "demo-workflow" in workflow_registry.format_text(payload)
