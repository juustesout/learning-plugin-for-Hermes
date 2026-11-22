"""
opslog — automatische operation logger voor Hermes agent-learning.

Gebruik (CLI):
  python3 opslog.py log --task-id choose-scrape-strategy --action use_webclaw_scrape \\
    --target agentcursus.nl --success true --duration-ms 3100 --cost 1 \\
    --summary "WebClaw scrape, 24500 chars"

Gebruik (Python):
  from opslog import log_operation
  log_operation(task_id="choose-scrape-strategy", action_id="use_webclaw_scrape", ...)
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SECRET_KV_RE = re.compile(
    r"(?i)(api[_-]?key|token|password|secret|passwd|authorization|access[_-]?key)"
    r"\s*[=:]\s*([^\s&\"']+)"
)
_BEARER_RE = re.compile(r"(?i)\b(bearer\s+)[a-z0-9._\-+=/]+")
_KEYISH_RE = re.compile(
    r"\b(sk-[a-zA-Z0-9]{8,}|ghp_[a-zA-Z0-9]{20,}|github_pat_[a-zA-Z0-9_]{20,}"
    r"|glpat-[a-zA-Z0-9_\-]{10,}|xox[baprs]-[a-zA-Z0-9-]{10,})\b"
)


def _env_path(name: str) -> str:
    return os.environ.get(name, "").strip()


def default_store_dir() -> str:
    for key in ("AGENT_LEARNING_OPSLOG_DIR", "AGENT_LEARNING_LOCAL_STORE_DIR"):
        value = _env_path(key)
        if value:
            return value
    return os.path.join(_REPO_ROOT, "store")


def retention_days() -> int:
    raw = os.environ.get("AGENT_LEARNING_OPSLOG_RETENTION_DAYS", "90").strip()
    try:
        return int(raw)
    except ValueError:
        return 30


STORE_DIR = default_store_dir()
OPSLOG_PATH = os.path.join(STORE_DIR, "operations.jsonl")


def health_path() -> str:
    return os.path.join(STORE_DIR, "opslog-health.json")


def _ensure_store():
    os.makedirs(STORE_DIR, exist_ok=True)


def redact_text(value: str) -> str:
    text = str(value or "")
    text = _SECRET_KV_RE.sub(lambda m: f"{m.group(1)}=***", text)
    text = _BEARER_RE.sub(r"\1***", text)
    text = _KEYISH_RE.sub("***", text)
    return text


def redact_target(target: str) -> str:
    text = str(target or "").strip()
    if not text:
        return ""
    if "://" in text or "?" in text or "#" in text:
        try:
            parts = urlsplit(text)
            if parts.scheme or parts.query or parts.fragment:
                text = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
        except ValueError:
            text = text.split("?", 1)[0].split("#", 1)[0]
    return redact_text(text)


def _parse_ts(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _with_lock(path: str, fn):
    lock_path = path + ".lock"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lockf:
        if os.name == "nt":
            import msvcrt

            lockf.seek(0)
            lockf.write("0")
            lockf.flush()
            lockf.seek(0)
            msvcrt.locking(lockf.fileno(), msvcrt.LK_LOCK, 1)
            try:
                return fn()
            finally:
                lockf.seek(0)
                msvcrt.locking(lockf.fileno(), msvcrt.LK_UNLCK, 1)
        import fcntl

        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def record_health(*, ok: bool, error: str = "", extra: Optional[dict] = None) -> None:
    path = health_path()

    def _write():
        current = {
            "ok": True,
            "written": 0,
            "failed": 0,
            "last_error": "",
            "last_error_at": None,
            "last_write_at": None,
            "path": OPSLOG_PATH,
        }
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    current.update(loaded)
            except (OSError, json.JSONDecodeError):
                pass
        now = datetime.now(timezone.utc).isoformat()
        if ok:
            current["ok"] = True
            current["written"] = int(current.get("written") or 0) + 1
            current["last_write_at"] = now
            current["last_error"] = ""
        else:
            current["ok"] = False
            current["failed"] = int(current.get("failed") or 0) + 1
            current["last_error"] = redact_text(error)[:500]
            current["last_error_at"] = now
        current["path"] = OPSLOG_PATH
        if extra:
            current.update(extra)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
        return current

    try:
        return _with_lock(path, _write)
    except OSError:
        return None


def health() -> dict:
    path = health_path()
    if not os.path.exists(path):
        return {
            "ok": True,
            "written": 0,
            "failed": 0,
            "last_error": "",
            "last_error_at": None,
            "last_write_at": None,
            "path": OPSLOG_PATH,
        }
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("path", OPSLOG_PATH)
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {
        "ok": False,
        "written": 0,
        "failed": 0,
        "last_error": "health file unreadable",
        "path": OPSLOG_PATH,
    }


def prune_old(max_age_days: Optional[int] = None) -> int:
    days = retention_days() if max_age_days is None else max_age_days
    if days <= 0 or not os.path.exists(OPSLOG_PATH):
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    def _prune():
        kept = []
        dropped = 0
        with open(OPSLOG_PATH, encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    op = json.loads(raw)
                except json.JSONDecodeError:
                    dropped += 1
                    continue
                ts = _parse_ts(str(op.get("timestamp") or ""))
                if ts is None or ts >= cutoff:
                    kept.append(raw)
                else:
                    dropped += 1
        if dropped:
            tmp = OPSLOG_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for raw in kept:
                    f.write(raw + "\n")
            os.replace(tmp, OPSLOG_PATH)
        return dropped

    return _with_lock(OPSLOG_PATH, _prune)


def log_operation(
    task_id: str,
    action_id: str,
    target: str = "",
    success: bool = True,
    duration_ms: Optional[int] = None,
    cost: Optional[float] = None,
    summary: str = "",
    action_type: str = "",
    action_logprob: Optional[float] = None,
    context_features: Optional[dict] = None,
    metadata: Optional[dict] = None,
    work_unit: str = "",
    trigger: str = "",
    execution_mode: str = "",
    parent_run_id: str = "",
    skills_involved: Optional[list[str]] = None,
):
    """Log één operatie naar operations.jsonl (append-only)."""
    from workflow_identity import operation_context

    _ensure_store()
    operational = operation_context(
        work_unit=work_unit,
        trigger=trigger,
        execution_mode=execution_mode,
        parent_run_id=parent_run_id,
        skills_involved=skills_involved or [],
    )
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_id": "hermes",
        "task_id": task_id,
        "action_id": action_id,
        "action_logprob": action_logprob,
        "target": redact_target(target),
        "execution_status": "completed" if success else "failed",
        "result_summary": redact_text(summary),
        "action_type": action_type,
        "request_latency_ms": duration_ms,
        "metadata": {
            "cost_credits": cost,
            "task_completed": success,
            **operational,
            **(metadata or {}),
        },
        "context_features": context_features or {},
    }

    def _append():
        with open(OPSLOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
            f.flush()
        return entry["id"]

    try:
        oid = _with_lock(OPSLOG_PATH, _append)
        state = record_health(ok=True)
        written = int((state or {}).get("written") or 0)
        if written and written % 100 == 0:
            prune_old()
        return oid
    except OSError as exc:
        record_health(ok=False, error=str(exc))
        raise


def count_operations() -> int:
    """Aantal gelogde operaties."""
    if not os.path.exists(OPSLOG_PATH):
        return 0
    with open(OPSLOG_PATH, encoding="utf-8") as f:
        return sum(1 for _ in f)


def tail(n: int = 10) -> list[dict]:
    """Laatste n operaties."""
    if not os.path.exists(OPSLOG_PATH):
        return []
    with open(OPSLOG_PATH, encoding="utf-8") as f:
        lines = f.readlines()
    return [json.loads(l) for l in lines[-n:]]


def stats() -> dict:
    """Basis statistieken over alle operaties."""
    payload = {
        "total": 0,
        "by_task": {},
        "by_status": {},
        "avg_duration_ms": None,
        "file": OPSLOG_PATH,
        "health": health(),
        "retention_days": retention_days(),
    }
    if not os.path.exists(OPSLOG_PATH):
        return payload
    by_task = {}
    by_status = {}
    durations = []
    total = 0
    with open(OPSLOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                op = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            tid = op.get("task_id", "unknown")
            by_task[tid] = by_task.get(tid, 0) + 1
            status = op.get("execution_status", "unknown")
            by_status[status] = by_status.get(status, 0) + 1
            d = op.get("request_latency_ms")
            if d is not None:
                durations.append(d)
    avg_dur = round(sum(durations) / len(durations), 1) if durations else None
    payload.update({
        "total": total,
        "by_task": by_task,
        "by_status": by_status,
        "avg_duration_ms": avg_dur,
    })
    return payload


def main(argv=None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="opslog", description="Hermes operation logger")
    sub = parser.add_subparsers(dest="cmd")

    p_log = sub.add_parser("log", help="Log een operatie")
    p_log.add_argument("--task-id", required=True)
    p_log.add_argument("--action-id", required=True)
    p_log.add_argument("--target", default="")
    p_log.add_argument("--success", default="true", choices=["true", "false"])
    p_log.add_argument("--duration-ms", type=int, default=None)
    p_log.add_argument("--cost", type=float, default=None)
    p_log.add_argument("--summary", default="")
    p_log.add_argument("--action-type", default="")
    p_log.add_argument("--logprob", type=float, default=None,
                       help="Action log probability (voor off-policy correctie)")
    p_log.add_argument("--context-features", default=None,
                       help="JSON string met context features (bv. '{\"cloudflare\":1.0}')")

    sub.add_parser("count", help="Aantal operaties")
    p_tail = sub.add_parser("tail", help="Laatste N operaties")
    p_tail.add_argument("--n", type=int, default=10)

    sub.add_parser("stats", help="Statistieken")
    sub.add_parser("health", help="Schrijf/fout-status van de opslog")
    p_prune = sub.add_parser("prune", help="Verwijder operaties ouder dan retentie")
    p_prune.add_argument("--days", type=int, default=None)

    args = parser.parse_args(argv)

    if args.cmd == "log":
        cf = None
        if args.context_features:
            try:
                cf = json.loads(args.context_features)
            except json.JSONDecodeError:
                print(f"Invalid context-features JSON: {args.context_features}", file=sys.stderr)
                return 1
        oid = log_operation(
            task_id=args.task_id,
            action_id=args.action_id,
            target=args.target,
            success=args.success == "true",
            duration_ms=args.duration_ms,
            cost=args.cost,
            summary=args.summary,
            action_type=args.action_type,
            action_logprob=args.logprob,
            context_features=cf,
        )
        print(f"ok {oid}")
        return 0
    elif args.cmd == "count":
        print(count_operations())
        return 0
    elif args.cmd == "tail":
        for op in tail(args.n):
            print(json.dumps(op, indent=2)[:500])
            print("---")
        return 0
    elif args.cmd == "stats":
        print(json.dumps(stats(), indent=2))
        return 0
    elif args.cmd == "health":
        print(json.dumps(health(), indent=2))
        return 0
    elif args.cmd == "prune":
        dropped = prune_old(args.days)
        print(dropped)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
