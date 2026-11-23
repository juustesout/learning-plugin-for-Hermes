"""Append-only decision.v1 records. Additive: does not replace opslog or shadow logs."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import opslog

RECORD = "decision.v1"
OUTCOME_RECORD = "outcome.v1"
ORPHAN_RECORD = "outcome_orphan.v1"
LOG_NAME = "decisions.jsonl"
ORPHAN_LOG_NAME = "decision_orphans.jsonl"
KILL_ENV = "AGENT_LEARNING_DECISION_RECORD"
LEVEL_NAMES = {1: "technical", 2: "accepted", 3: "confirmed"}
NAME_LEVELS = {name: level for level, name in LEVEL_NAMES.items()}
OK_STATUSES = frozenset({"ok", "success", "completed"})
ERROR_STATUSES = frozenset({"error", "fail", "failed", "timeout", "blocked", "cancelled"})
FORBIDDEN_PRE_KEYS = frozenset(
    {"html", "headers", "content", "body", "result", "raw_html", "response_headers"}
)


def enabled() -> bool:
    raw = os.environ.get(KILL_ENV, "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def log_path() -> str:
    return os.path.join(opslog.STORE_DIR, LOG_NAME)


def orphan_path() -> str:
    return os.path.join(opslog.STORE_DIR, ORPHAN_LOG_NAME)


def join_key(
    *,
    session_id: str = "",
    conversation_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    tool_name: str = "",
) -> str:
    sid = str(session_id or "").strip() or str(conversation_id or "").strip()
    tc = str(tool_call_id or "").strip()
    if tc:
        return f"{sid}|{tc}"
    return f"{sid}|{str(turn_id or '').strip()}|{str(tool_name or '').strip()}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(path: str, row: dict) -> None:
    payload = json.dumps(row, default=str) + "\n"

    def _write():
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()

    opslog._with_lock(path, _write)


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def pre_call_features(features: Any) -> dict[str, float]:
    if not isinstance(features, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in features.items():
        name = str(key or "").strip()
        if not name or name.lower() in FORBIDDEN_PRE_KEYS:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out[name] = float(value)
    return out


def parse_level(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value in LEVEL_NAMES:
        return value
    text = str(value or "").strip().lower()
    if text.isdigit():
        number = int(text)
        if number in LEVEL_NAMES:
            return number
    if text in NAME_LEVELS:
        return NAME_LEVELS[text]
    raise ValueError("level must be 1/2/3 or technical/accepted/confirmed")


def parse_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in OK_STATUSES:
        return "ok"
    if text in ERROR_STATUSES:
        return "error"
    raise ValueError("status must be ok or error")


def open_decision(
    *,
    join_key_value: str,
    requested_tool: str,
    advised_tool: str = "",
    chosen_tool: str = "",
    context_features: Any = None,
    task_id: str = "",
    timestamp: str = "",
    decision_id: str = "",
) -> Optional[dict]:
    if not enabled():
        return None
    key = str(join_key_value or "").strip()
    tool = str(requested_tool or "").strip()
    if not key or not tool:
        return None
    existing = get_by_join_key(key)
    if existing is not None:
        return existing
    row = {
        "record": RECORD,
        "decision_id": str(decision_id or uuid.uuid4()),
        "join_key": key,
        "timestamp": str(timestamp or _now()),
        "requested_tool": tool,
        "advised_tool": str(advised_tool or ""),
        "chosen_tool": str(chosen_tool or ""),
        "task_id": str(task_id or ""),
        "context_features": pre_call_features(context_features),
        "outcomes": {},
    }
    _append(log_path(), row)
    return row


def get_by_id(decision_id: str) -> Optional[dict]:
    wanted = str(decision_id or "").strip()
    if not wanted:
        return None
    return load_index().get(wanted)


def get_by_join_key(key: str) -> Optional[dict]:
    wanted = str(key or "").strip()
    if not wanted:
        return None
    for row in load_index().values():
        if str(row.get("join_key") or "") == wanted:
            return row
    return None


def load_index() -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for row in _read_jsonl(log_path()):
        kind = str(row.get("record") or "")
        if kind == RECORD:
            did = str(row.get("decision_id") or "").strip()
            if not did:
                continue
            current = dict(row)
            outcomes = current.get("outcomes")
            current["outcomes"] = dict(outcomes) if isinstance(outcomes, dict) else {}
            current["context_features"] = pre_call_features(current.get("context_features"))
            merged[did] = current
            continue
        if kind != OUTCOME_RECORD:
            continue
        did = str(row.get("decision_id") or "").strip()
        target = merged.get(did)
        if target is None:
            continue
        _merge_outcome(target, row)
    return merged


def load_decisions() -> list[dict]:
    return list(load_index().values())


def _merge_outcome(target: dict, event: dict) -> bool:
    try:
        level = parse_level(event.get("level"))
        status = parse_status(event.get("status"))
    except ValueError:
        return False
    field = LEVEL_NAMES[level]
    outcomes = target.setdefault("outcomes", {})
    if field in outcomes and outcomes[field] is not None:
        return False
    outcomes[field] = {
        "status": status,
        "at": str(event.get("at") or event.get("timestamp") or _now()),
    }
    if level == 1 and not target.get("chosen_tool"):
        chosen = str(event.get("chosen_tool") or "").strip()
        if chosen:
            target["chosen_tool"] = chosen
    return True


def attach_technical(
    *,
    join_key_value: str,
    status: str,
    chosen_tool: str = "",
    at: str = "",
) -> Optional[dict]:
    if not enabled():
        return None
    key = str(join_key_value or "").strip()
    if not key:
        return None
    try:
        parse_status(status)
    except ValueError:
        return None
    row = get_by_join_key(key)
    if row is None:
        return record_outcome_event(
            decision_id="",
            level=1,
            status=status,
            join_key_value=key,
            chosen_tool=chosen_tool,
            at=at,
        )
    return record_outcome_event(
        decision_id=str(row.get("decision_id") or ""),
        level=1,
        status=status,
        chosen_tool=chosen_tool,
        at=at,
    )


def record_outcome_event(
    *,
    decision_id: str,
    level: Any,
    status: Any,
    join_key_value: str = "",
    chosen_tool: str = "",
    at: str = "",
) -> dict:
    parsed_level = parse_level(level)
    parsed_status = parse_status(status)
    did = str(decision_id or "").strip()
    stamp = str(at or _now())
    if not did:
        orphan = {
            "record": ORPHAN_RECORD,
            "decision_id": did,
            "join_key": str(join_key_value or ""),
            "level": parsed_level,
            "level_name": LEVEL_NAMES[parsed_level],
            "status": parsed_status,
            "at": stamp,
            "reason": "unknown_decision",
        }
        if enabled():
            _append(orphan_path(), orphan)
        return {"ok": False, "orphan": True, "reason": "unknown_decision", "event": orphan}
    target = get_by_id(did)
    if target is None:
        orphan = {
            "record": ORPHAN_RECORD,
            "decision_id": did,
            "join_key": str(join_key_value or ""),
            "level": parsed_level,
            "level_name": LEVEL_NAMES[parsed_level],
            "status": parsed_status,
            "at": stamp,
            "reason": "unknown_decision",
        }
        if enabled():
            _append(orphan_path(), orphan)
        return {
            "ok": False,
            "orphan": True,
            "decision_id": did,
            "reason": "unknown_decision",
            "event": orphan,
        }
    event = {
        "record": OUTCOME_RECORD,
        "decision_id": did,
        "level": parsed_level,
        "level_name": LEVEL_NAMES[parsed_level],
        "status": parsed_status,
        "chosen_tool": str(chosen_tool or ""),
        "at": stamp,
    }
    changed = _merge_outcome(dict(target), event)
    if enabled():
        _append(log_path(), event)
    merged = get_by_id(did)
    return {
        "ok": True,
        "orphan": False,
        "changed": changed,
        "decision_id": did,
        "level": parsed_level,
        "status": parsed_status,
        "decision": merged,
    }


def record_outcome(*, decision_id: str, level: Any, status: Any) -> dict:
    did = str(decision_id or "").strip()
    if not did:
        raise ValueError("decision_id is required")
    parse_level(level)
    parse_status(status)
    return record_outcome_event(decision_id=did, level=level, status=status)
