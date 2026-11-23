"""Weekly ITT report for HN slots and decision records. Read-only on live store."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import blind_label
import promotion_gate

try:
    import confound
except Exception:  # pragma: no cover - optional on a P7/P8-only tree
    confound = None

try:
    import decision_record
except Exception:  # pragma: no cover - optional on a P7/P8-only tree
    decision_record = None


def _parse_ts(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _in_window(value: Any, start: datetime, end: datetime) -> bool:
    ts = _parse_ts(value)
    if ts is None:
        return False
    return start <= ts < end


def window(*, now: Optional[datetime] = None, days: int = 7) -> tuple[datetime, datetime]:
    end = now or datetime.now(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    start = end - timedelta(days=days)
    return start, end


def _itt_from_decisions(rows: list[dict]) -> dict[str, Any]:
    n_l = n_b = succ_l = succ_b = 0
    n_contrast = 0
    advised = set()
    for row in rows:
        arm = str(row.get("arm") or "").strip().upper()
        if arm in {"LEARNED"}:
            arm = "L"
        if arm in {"BASELINE"}:
            arm = "B"
        chosen = str(row.get("chosen_id") or row.get("chosen_tool") or "")
        baseline = str(row.get("baseline_pick_id") or "")
        if chosen and baseline and chosen != baseline:
            n_contrast += 1
        advised_tool = str(row.get("advised_tool") or "")
        if advised_tool:
            advised.add(advised_tool)
        success = row.get("quality_success")
        if success is None:
            outcomes = row.get("outcomes") if isinstance(row.get("outcomes"), dict) else {}
            confirmed = outcomes.get("confirmed") if isinstance(outcomes.get("confirmed"), dict) else {}
            if confirmed:
                success = str(confirmed.get("status") or "") == "ok"
            else:
                success = None
        if arm == "L":
            n_l += 1
            if success is True:
                succ_l += 1
        elif arm == "B":
            n_b += 1
            if success is True:
                succ_b += 1
    return {
        "n": n_l + n_b,
        "n_l": n_l,
        "n_b": n_b,
        "successes_l": succ_l,
        "successes_b": succ_b,
        "n_contrast": n_contrast,
        "n_distinct": len(advised),
        "constant_advice": len(advised) == 1 and (n_l + n_b) >= promotion_gate.MIN_N,
        "level": "confirmed",
    }


def build(*, now: Optional[datetime] = None, days: int = 7, view: str = "live") -> dict[str, Any]:
    start, end = window(now=now, days=days)
    decisions = [
        row
        for row in blind_label._view(blind_label.load_decisions(), view)
        if _in_window(row.get("decided_at"), start, end)
    ]
    records = []
    if decision_record is not None:
        records = [
            row
            for row in decision_record.load_decisions()
            if _in_window(row.get("timestamp"), start, end)
        ]
    labels = [
        row
        for row in blind_label._view(blind_label.load_labels(), view)
        if _in_window(row.get("labeled_at"), start, end)
    ]
    itt = _itt_from_decisions(decisions)
    gate = promotion_gate.evaluate(itt)
    overlap = blind_label.selection_overlap(decisions)
    balance = blind_label.arm_balance(decisions)
    confound_status = (
        confound.confound_report(decisions)
        if confound is not None
        else {"ok": True, "alarms": []}
    )
    labeled = sum(1 for row in labels if row.get("relevant") is not None)
    quality_ok = sum(1 for row in labels if row.get("quality_success") is True)
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "days": days,
        "view": view,
        "decisions": len(decisions),
        "decision_records": len(records),
        "labels": labeled,
        "quality_successes": quality_ok,
        "itt": itt,
        "overlap": overlap,
        "balance": balance,
        "confound": confound_status,
        "gate": gate,
        "primary_claim": "itt_all_posted",
        "explanatory": "n_contrast",
    }


def format_report(data: dict[str, Any] | None = None) -> str:
    info = data or build()
    gate = info.get("gate") or {}
    itt = info.get("itt") or {}
    overlap = info.get("overlap") or {}
    state = "open" if gate.get("open") else "closed"
    return (
        "week:\n"
        f"  window: {info.get('start')} .. {info.get('end')}\n"
        f"  decisions: {info.get('decisions')}\n"
        f"  labels: {info.get('labels')}\n"
        f"  quality_successes: {info.get('quality_successes')}\n"
        "itt:\n"
        f"  n={itt.get('n')} L={itt.get('n_l')} B={itt.get('n_b')} "
        f"contrast={itt.get('n_contrast')}\n"
        f"  successes L/B: {itt.get('successes_l')}/{itt.get('successes_b')}\n"
        "gate:\n"
        f"  {state}  reason: {gate.get('reason')}\n"
        f"  min_n={gate.get('min_n')} min_margin={gate.get('min_margin')}\n"
        "overlap:\n"
        f"  n_contrast={overlap.get('n_contrast')} rate={overlap.get('rate')}\n"
        "confound:\n"
        f"  ok={((info.get('confound') or {}).get('ok'))} "
        f"alarms={((info.get('confound') or {}).get('alarms'))}\n"
    )
