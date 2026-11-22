"""Declared workflow registry with status derived from existing run and operation logs."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from workflow_manifest import load_manifests

REPO_ROOT = Path(__file__).resolve().parent
PROFILE_ROOT = REPO_ROOT.parent
DEFAULT_MANIFESTS = REPO_ROOT / "workflows"
PLUGIN_STORE = PROFILE_ROOT / "plugins" / "hermes-agent-learning" / "store"
PROJECT_STORE = REPO_ROOT / "store"
SCRIPTS_LOG_DIR = PROFILE_ROOT / "scripts" / "logs"


def parse_ts(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def newest(paths: Iterable[Path]) -> Path | None:
    candidates = [path for path in paths if path.exists()]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def default_operation_store() -> Path:
    configured = os.environ.get("AGENT_LEARNING_OPSLOG_DIR", "").strip()
    if configured:
        return Path(configured)
    active = newest([PLUGIN_STORE / "operations.jsonl", PROJECT_STORE / "operations.jsonl"])
    return active.parent if active else PROJECT_STORE


def latest_sloppy_runs(limit: int = 5) -> list[dict]:
    files = sorted(SCRIPTS_LOG_DIR.glob("sloppy-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    runs: list[dict] = []
    for path in files[:limit]:
        rows = read_jsonl(path)
        if not rows:
            continue
        done = next((row for row in reversed(rows) if row.get("step") == "done"), None)
        start = next((row for row in reversed(rows) if row.get("step") == "start"), rows[0])
        failures = sum(1 for row in rows if row.get("status") in {"fail", "error"})
        last = rows[-1]
        runs.append(
            {
                "run_id": str((done or start).get("run_id") or path.stem),
                "status": "ok" if done and done.get("status") == "ok" and failures == 0 else "failed" if failures else "unknown",
                "started_at": start.get("ts"),
                "ended_at": (done or last).get("ts"),
                "posted": int((done or {}).get("posted", 0)),
                "failed_steps": failures,
                "source_file": str(path),
            }
        )
    return runs


def evidence_dir(store: Path) -> Path:
    live = store / "evidence" / "hn-aidevforum"
    project = PROJECT_STORE / "evidence" / "hn-aidevforum"
    return live if live.exists() else project


def operation_summary(store: Path, manifest: dict) -> dict:
    rows = read_jsonl(store / "operations.jsonl")
    work_unit = manifest["work_unit"]
    workflow_id = manifest["workflow_id"]
    matching = []
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if metadata.get("work_unit") == work_unit or metadata.get("workflow") in {workflow_id, "hn-aidevforum"}:
            matching.append(row)
    completed = sum(1 for row in matching if row.get("execution_status") == "completed")
    failed = sum(1 for row in matching if row.get("execution_status") == "failed")
    costs = [float(row.get("metadata", {}).get("cost_credits")) for row in matching if isinstance(row.get("metadata", {}).get("cost_credits"), (int, float))]
    return {
        "operations": len(matching),
        "completed": completed,
        "failed": failed,
        "cost_credits": round(sum(costs), 4),
        "tools": dict(Counter(row.get("action_id", "unknown") for row in matching).most_common(10)),
    }


def status_for(manifest: dict, op_store: Path) -> dict:
    workflow_id = manifest["workflow_id"]
    row: dict[str, Any] = {
        "workflow_id": workflow_id,
        "version": manifest["version"],
        "name": manifest["name"],
        "owner": manifest["owner"],
        "work_unit": manifest["work_unit"],
        "trigger": manifest["trigger"],
        "execution_mode": manifest["execution_mode"],
        "automation": manifest.get("automation"),
        "operations": operation_summary(op_store, manifest),
        "manifest_steps": len(manifest["steps"]),
    }
    if workflow_id == "sloppy-hn":
        runs = latest_sloppy_runs()
        row["recent_runs"] = runs
        row["last_run_at"] = runs[0]["ended_at"] if runs else None
        row["last_status"] = runs[0]["status"] if runs else "unknown"
        row["alarms"] = ["no_recorded_run"] if not runs else []
    elif workflow_id == "blind-label-ui":
        evidence = evidence_dir(op_store)
        labels = read_jsonl(evidence / "labels.jsonl")
        queue = read_jsonl(evidence / "queue.jsonl")
        candidates = read_jsonl(evidence / "candidates.jsonl")
        latest = max((parse_ts(x.get("labeled_at")) for x in labels), default=None)
        row["sample"] = {
            "labels": len({str(x.get("id")) for x in labels if x.get("id")}),
            "queue": len(queue),
            "candidates": len(candidates),
            "last_label_at": latest.isoformat() if latest else None,
        }
        row["last_run_at"] = latest.isoformat() if latest else None
        row["last_status"] = "ready" if queue else "empty"
        row["alarms"] = [] if labels else ["no_labels"]
    else:
        row["last_run_at"] = None
        row["last_status"] = "unknown"
        row["alarms"] = ["no_adapter"]
    return row


def registry(manifests_dir: Path = DEFAULT_MANIFESTS, store: Path | None = None) -> dict:
    op_store = store or default_operation_store()
    manifests = load_manifests(manifests_dir)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "operation_store": str(op_store),
        "workflows": [status_for(manifest, op_store) for manifest in manifests],
    }


def format_text(payload: dict) -> str:
    lines = [f"Workflow registry · generated {payload['generated_at']}", f"Operation store: {payload['operation_store']}", ""]
    for item in payload["workflows"]:
        lines.extend(
            [
                f"[{item['workflow_id']}@{item['version']}] {item['name']}",
                f"  owner: {item['owner']}  work_unit: {item['work_unit']}",
                f"  trigger: {item['trigger']}  mode: {item['execution_mode']}",
                f"  status: {item['last_status']}  last: {item['last_run_at'] or '-'}",
                f"  declared steps: {item['manifest_steps']}  alarms: {', '.join(item['alarms']) or '-'}",
                f"  contextual operations: {item['operations']['operations']}  completed: {item['operations']['completed']}  failed: {item['operations']['failed']}",
            ]
        )
        if "recent_runs" in item:
            for run in item["recent_runs"][:3]:
                lines.append(f"  run {run['run_id']}: {run['status']} posted={run['posted']} failed_steps={run['failed_steps']}")
        if "sample" in item:
            sample = item["sample"]
            lines.append(f"  labels={sample['labels']} queue={sample['queue']} candidates={sample['candidates']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="workflow-registry")
    sub = parser.add_subparsers(dest="command", required=True)
    validate_parser = sub.add_parser("validate", help="validate all workflow manifests")
    validate_parser.add_argument("--manifests", type=Path, default=DEFAULT_MANIFESTS)
    status_parser = sub.add_parser("status", help="show declared workflows and live status")
    status_parser.add_argument("--all", action="store_true", help="accepted for compatibility")
    status_parser.add_argument("--json", action="store_true")
    status_parser.add_argument("--manifests", type=Path, default=DEFAULT_MANIFESTS)
    status_parser.add_argument("--store", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            manifests = load_manifests(args.manifests)
            print(json.dumps({"ok": True, "manifests": len(manifests)}, indent=2))
            return 0
        payload = registry(args.manifests, args.store)
        print(json.dumps(payload, indent=2, ensure_ascii=False) if args.json else format_text(payload))
        return 0
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
