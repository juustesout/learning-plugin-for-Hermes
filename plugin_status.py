"""CLI status and selftest. Observer only. Never writes the live store."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any

import eligibility
import features
import hermes_plugin
import opslog
import replay_advice
import shadow_policy

MIN_N = replay_advice.DEFAULT_MIN_N
SELFTEST_TOOL = "mcp_webclaw_webclaw_scrape"
SELFTEST_URL = "https://example.com/selftest"


def _hooks_state() -> dict[str, Any]:
    return dict(getattr(hermes_plugin, "HOOKS_REGISTERED", {}) or {})


def _hook_label(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def _file_meta(path: str) -> tuple[bool, int, float]:
    if not os.path.isfile(path):
        return (False, 0, 0.0)
    st = os.stat(path)
    return (True, st.st_size, st.st_mtime)


def _last_opslog_ts() -> str:
    if not os.path.isfile(opslog.OPSLOG_PATH):
        return "-"
    last = ""
    with open(opslog.OPSLOG_PATH, encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                last = raw
    if not last:
        return "-"
    try:
        row = json.loads(last)
    except json.JSONDecodeError:
        return "-"
    return str(row.get("timestamp") or "-") or "-"


def _eligible_count() -> int:
    path = opslog.OPSLOG_PATH
    if not os.path.isfile(path):
        return 0
    rows = replay_advice._read_jsonl(path)
    kept, _skip = eligibility.classify_rows(rows)
    return len(kept)


def _consolidate_last() -> str:
    path = os.path.join(opslog.STORE_DIR, "consolidated.txt")
    if not os.path.isfile(path):
        return "never"
    ts = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
    return ts.isoformat()


def _gate_view() -> dict[str, Any]:
    closed = {
        "open": False,
        "n": 0,
        "min_n": MIN_N,
        "n_distinct": 0,
        "reason": "snapshot missing",
    }
    try:
        raw = replay_advice.load_corpus()
    except (OSError, FileNotFoundError, json.JSONDecodeError):
        raw = []
    episodes, _skipped = eligibility.classify_rows(raw)
    try:
        snapshot = shadow_policy.load_snapshot(replay_advice.DEFAULT_TASK_ID)
    except (OSError, ValueError, json.JSONDecodeError):
        snapshot = None
    if not isinstance(snapshot, dict):
        closed["n"] = len(episodes)
        return closed
    try:
        rows = replay_advice.replay(episodes, snapshot)
        status = replay_advice.gate(rows, min_n=MIN_N)
    except features.FeatureNameMismatch:
        closed["n"] = len(episodes)
        closed["reason"] = "feature mismatch"
        return closed
    except Exception:
        closed["n"] = len(episodes)
        closed["reason"] = "replay error"
        return closed
    if status["gate_open"]:
        reason = "-"
    elif status["n"] < MIN_N:
        reason = f"n<{MIN_N}"
    elif status["n_distinct"] < 2:
        reason = "n_distinct<2"
    else:
        reason = "closed"
    return {
        "open": bool(status["gate_open"]),
        "n": int(status["n"]),
        "min_n": MIN_N,
        "n_distinct": int(status["n_distinct"]),
        "reason": reason,
    }


def collect_status() -> dict[str, Any]:
    hooks = _hooks_state()
    module, source, errors = hermes_plugin._load_features()
    health = opslog.health() if opslog is not None else {}
    last_error = str(health.get("last_error") or "") or "-"
    if module is None and errors:
        last_error = "; ".join(errors)
    return {
        "hooks": {
            "pre_tool_call": hooks.get("pre_tool_call"),
            "post_tool_call": hooks.get("post_tool_call"),
            "pre_llm_call": hooks.get("pre_llm_call"),
            "llm_execution_middleware": hooks.get("llm_execution_middleware"),
        },
        "features": {
            "import_source": source or "none",
            "last_error": last_error,
            "loaded": module is not None,
        },
        "opslog": {
            "lines": opslog.count_operations() if opslog is not None else 0,
            "last": _last_opslog_ts() if opslog is not None else "-",
            "eligible_scrape": _eligible_count() if opslog is not None else 0,
        },
        "consolidate": {"last": _consolidate_last() if opslog is not None else "never"},
        "gate": _gate_view(),
    }


def format_status(data: dict[str, Any] | None = None) -> str:
    info = data or collect_status()
    hooks = info["hooks"]
    feats = info["features"]
    log = info["opslog"]
    gate = info["gate"]
    state = "open" if gate.get("open") else "closed"
    pre = _hook_label(hooks.get("pre_tool_call"))
    post = _hook_label(hooks.get("post_tool_call"))
    llm = _hook_label(hooks.get("pre_llm_call"))
    mid = _hook_label(hooks.get("llm_execution_middleware"))
    return (
        f"hooks:\n"
        f"  pre_tool_call: {pre}\n"
        f"  post_tool_call: {post}\n"
        f"  pre_llm_call: {llm}\n"
        f"  llm_execution_middleware: {mid}\n"
        f"features:\n"
        f"  import_source: {feats.get('import_source')}\n"
        f"  last_error: {feats.get('last_error')}\n"
        f"opslog:\n"
        f"  lines: {log.get('lines')}\n"
        f"  last: {log.get('last')}\n"
        f"  eligible_scrape: {log.get('eligible_scrape')}\n"
        f"consolidate:\n"
        f"  last: {info['consolidate'].get('last')}\n"
        f"gate:\n"
        f"  {state}  n={gate.get('n')}/{gate.get('min_n')}  distinct={gate.get('n_distinct')}\n"
        f"  reason: {gate.get('reason')}\n"
    )


def _bind_store(path: str) -> dict[str, Any]:
    prev = {
        "STORE_DIR": opslog.STORE_DIR,
        "OPSLOG_PATH": opslog.OPSLOG_PATH,
        "env_dir": os.environ.get("AGENT_LEARNING_OPSLOG_DIR"),
        "env_store": os.environ.get("AGENT_LEARNING_LOCAL_STORE_DIR"),
    }
    opslog.STORE_DIR = path
    opslog.OPSLOG_PATH = os.path.join(path, "operations.jsonl")
    os.environ["AGENT_LEARNING_OPSLOG_DIR"] = path
    os.environ["AGENT_LEARNING_LOCAL_STORE_DIR"] = path
    features.reset_memory()
    shadow_policy.reset_cache()
    return prev


def _restore_store(prev: dict[str, Any]) -> None:
    opslog.STORE_DIR = prev["STORE_DIR"]
    opslog.OPSLOG_PATH = prev["OPSLOG_PATH"]
    if prev["env_dir"] is None:
        os.environ.pop("AGENT_LEARNING_OPSLOG_DIR", None)
    else:
        os.environ["AGENT_LEARNING_OPSLOG_DIR"] = prev["env_dir"]
    if prev["env_store"] is None:
        os.environ.pop("AGENT_LEARNING_LOCAL_STORE_DIR", None)
    else:
        os.environ["AGENT_LEARNING_LOCAL_STORE_DIR"] = prev["env_store"]
    features.reset_memory()
    shadow_policy.reset_cache()


def _selftest_kwargs() -> dict[str, Any]:
    return {
        "tool_name": SELFTEST_TOOL,
        "args": {"url": SELFTEST_URL},
        "session_id": "selftest-session",
        "turn_id": "selftest-turn",
        "tool_call_id": "selftest-call",
        "status": "ok",
        "duration_ms": 11,
    }


def _run_isolated_call(store: str) -> dict[str, Any]:
    prev = _bind_store(store)
    try:
        before = opslog.count_operations()
        kwargs = _selftest_kwargs()
        pre_ret = hermes_plugin.on_pre_tool_call(**kwargs)
        post_ret = hermes_plugin.on_post_tool_call(**kwargs)
        after = opslog.count_operations()
        ops = opslog.tail(1)
        row = ops[-1] if ops else {}
        ctx = row.get("context_features")
        names = set(ctx) if isinstance(ctx, dict) else set()
        return {
            "pre_none": pre_ret is None,
            "post_none": post_ret is None,
            "new_ops": after - before,
            "task_id": str(row.get("task_id") or ""),
            "import_source": str((row.get("metadata") or {}).get("import_source") or ""),
            "context_features": ctx,
            "feature_names_ok": names == set(features.PRE_CALL_FEATURE_NAMES),
        }
    finally:
        _restore_store(prev)


def _run_import_failure(store: str) -> dict[str, Any]:
    import builtins

    real_import = builtins.__import__

    def _blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "features" or (fromlist and "features" in fromlist):
            raise ImportError("selftest: features blocked")
        return real_import(name, globals, locals, fromlist, level)

    prev = _bind_store(store)
    try:
        builtins.__import__ = _blocked
        payload = hermes_plugin.build_operation(
            tool_name=SELFTEST_TOOL,
            args={"url": SELFTEST_URL},
            status="ok",
        )
        health = opslog.health()
        return {
            "payload": payload,
            "context_null": payload is not None and payload.get("context_features") is None,
            "health_error": "features import failed" in str(health.get("last_error") or ""),
            "import_source": str((payload or {}).get("metadata", {}).get("import_source") or ""),
        }
    finally:
        builtins.__import__ = real_import
        _restore_store(prev)


def run_selftest() -> dict[str, Any]:
    live_path = opslog.OPSLOG_PATH
    live_meta = _file_meta(live_path)
    tmp = tempfile.mkdtemp(prefix="al_selftest_")
    if os.path.realpath(tmp) == os.path.realpath(os.path.dirname(live_path) or "."):
        raise RuntimeError("selftest refuses to use the live store")
    os.makedirs(tmp, exist_ok=True)
    happy = _run_isolated_call(tmp)
    broken_dir = tempfile.mkdtemp(prefix="al_selftest_fail_")
    broken = _run_import_failure(broken_dir)
    after_meta = _file_meta(live_path)
    checks = {
        "hook_returns_none": happy["pre_none"] and happy["post_none"],
        "one_new_operation": happy["new_ops"] == 1,
        "task_id_filled": bool(happy["task_id"]),
        "import_source_filled": happy["import_source"] not in {"", "none"},
        "context_features_pre_call": happy["feature_names_ok"],
        "import_failure_health": broken["health_error"] and broken["context_null"],
        "live_store_untouched": after_meta == live_meta,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "temp_store": tmp,
        "live_path": live_path,
        "task_id": happy["task_id"],
        "import_source": happy["import_source"],
    }


def format_selftest(result: dict[str, Any] | None = None) -> str:
    info = result or run_selftest()
    lines = ["selftest: " + ("ok" if info.get("ok") else "fail")]
    for name, passed in (info.get("checks") or {}).items():
        lines.append(f"  {name}: {'yes' if passed else 'no'}")
    return "\n".join(lines) + "\n"
