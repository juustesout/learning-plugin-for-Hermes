"""Prompt advice for pre_llm_call only. Fail-open. Arm B is byte-identical."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

import opslog

KILL_ENV = "AGENT_LEARNING_ADVICE"
ASSIGN_SALT = "advice-llm-2026-11-22"
ADVICE_MARKER = "agent-learning-advice"
LOG_NAME = "advice_calls.jsonl"


def enabled() -> bool:
    raw = os.environ.get(KILL_ENV, "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def log_path() -> str:
    return os.path.join(opslog.STORE_DIR, LOG_NAME)


def assign_arm(*parts: str) -> str:
    material = "|".join(str(p or "") for p in parts) + "|" + ASSIGN_SALT
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return "L" if int(digest[:8], 16) % 2 == 0 else "B"


def _messages_bytes(messages: Any) -> bytes:
    return json.dumps(messages, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")


def format_advice(advice: dict[str, Any]) -> str:
    recommended = str(advice.get("recommended_action") or "").strip()
    task = str(advice.get("task_id") or "").strip()
    options = advice.get("options") if isinstance(advice.get("options"), dict) else {}
    lines = [f"[{ADVICE_MARKER}] task={task or '-'} recommended={recommended or '-'}"]
    for name, stats in list(options.items())[:8]:
        if not isinstance(stats, dict):
            continue
        n = stats.get("n", 0)
        wilson = stats.get("wilson_low")
        median = stats.get("median_latency_ms")
        lines.append(f"  {name}: n={n} wilson_low={wilson} median_ms={median}")
    return "\n".join(lines)


def inject_advice(messages: Any, advice: dict[str, Any]) -> Any:
    note = format_advice(advice)
    if isinstance(messages, list):
        out = list(messages)
        out.append({"role": "system", "content": note})
        return out
    if isinstance(messages, dict):
        out = dict(messages)
        existing = out.get("messages")
        if isinstance(existing, list):
            copied = list(existing)
            copied.append({"role": "system", "content": note})
            out["messages"] = copied
            return out
    return messages


def _append(row: dict) -> None:
    path = log_path()
    payload = json.dumps(row, default=str) + "\n"

    def _write():
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()

    try:
        opslog._with_lock(path, _write)
    except OSError:
        return


def apply_advice(
    messages: Any,
    *,
    gate: Optional[dict[str, Any]] = None,
    advice: Optional[dict[str, Any]] = None,
    session_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
) -> Any:
    """Return messages for the LLM call. Never raises. Arm B is unchanged.

    Arm assignment always runs (p=0.5, salt ASSIGN_SALT) before the gate is
    consulted. Only prompt injection depends on gate.open. A closed gate still
    logs arm L or B; the returned messages stay byte-identical to the input.
    """
    original = messages
    try:
        if not enabled():
            _append(
                {
                    "record": "advice_call",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "arm": "",
                    "injected": False,
                    "reason": "kill_switch",
                }
            )
            return original
        arm = assign_arm(session_id, turn_id, api_request_id)
        status = gate if isinstance(gate, dict) else {"open": False, "reason": "gate_missing"}
        inject = bool(status.get("open")) and arm == "L" and isinstance(advice, dict) and advice
        if not inject:
            _append(
                {
                    "record": "advice_call",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "arm": arm,
                    "injected": False,
                    "reason": "kill_switch"
                    if not enabled()
                    else ("gate_closed" if not status.get("open") else ("arm_b" if arm == "B" else "no_advice")),
                    "gate_open": bool(status.get("open")),
                }
            )
            return original
        changed = inject_advice(original, advice)
        _append(
            {
                "record": "advice_call",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "arm": arm,
                "injected": True,
                "reason": "-",
                "gate_open": True,
                "recommended_action": advice.get("recommended_action") or "",
            }
        )
        return changed
    except Exception:
        return original
