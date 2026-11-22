"""Pre-decision option statistics from opslog history (read-only)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import opslog

SUCCESS_STATUSES = {"completed", "success", "ok"}
FAIL_STATUSES = {"failed", "error", "timeout", "blocked", "cancelled"}


def default_operations_path() -> str:
    return os.path.join(opslog.default_store_dir(), "operations.jsonl")


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


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_operations(path: str) -> list[dict]:
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


def _recency_factor(last_used_at: Optional[datetime], now: datetime) -> float:
    if last_used_at is None:
        return 0.6
    days = max((now - last_used_at).total_seconds(), 0.0) / 86400.0
    if days <= 2.0:
        return 1.0
    if days <= 7.0:
        return 0.9
    if days <= 30.0:
        return 0.8
    return 0.7


def _normalize_penalty(
    value: Optional[float],
    *,
    minimum: Optional[float],
    maximum: Optional[float],
) -> float:
    if value is None:
        return 0.5
    if minimum is None or maximum is None or maximum <= minimum:
        return 0.0
    return (value - minimum) / (maximum - minimum)


def get_option_stats(
    *,
    task_id: str,
    options: list[str],
    lookback_days: int = 30,
    min_n: int = 5,
    operations_path: str = "",
    now: Optional[datetime] = None,
) -> dict[str, dict[str, Any]]:
    """Return per-option performance metrics for one task."""
    task = str(task_id or "").strip()
    unique_options = [str(opt).strip() for opt in options if str(opt).strip()]
    if not task or not unique_options:
        return {}

    path = operations_path or default_operations_path()
    rows = _load_operations(path)
    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    cutoff = None
    if lookback_days > 0:
        cutoff = now_utc - timedelta(days=lookback_days)

    by_option: dict[str, dict[str, Any]] = {
        option: {
            "n": 0,
            "successes": 0,
            "latency_total": 0.0,
            "latency_n": 0,
            "cost_total": 0.0,
            "cost_n": 0,
            "last_used_at": None,
        }
        for option in unique_options
    }
    wanted = set(unique_options)

    for row in rows:
        if str(row.get("task_id") or "").strip() != task:
            continue
        action_id = str(row.get("action_id") or "").strip()
        if action_id not in wanted:
            continue
        ts = _parse_ts(row.get("timestamp"))
        if cutoff is not None and ts is not None and ts < cutoff:
            continue

        bucket = by_option[action_id]
        bucket["n"] += 1

        status = str(row.get("execution_status") or "").strip().lower()
        if status in SUCCESS_STATUSES:
            bucket["successes"] += 1
        elif status and status not in FAIL_STATUSES:
            pass

        latency = _safe_float(row.get("request_latency_ms"))
        if latency is not None and latency >= 0:
            bucket["latency_total"] += latency
            bucket["latency_n"] += 1

        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            cost = _safe_float(metadata.get("cost_credits"))
            if cost is not None and cost >= 0:
                bucket["cost_total"] += cost
                bucket["cost_n"] += 1

        last_used = bucket["last_used_at"]
        if ts is not None and (last_used is None or ts > last_used):
            bucket["last_used_at"] = ts

    avg_latencies = [
        (data["latency_total"] / data["latency_n"])
        for data in by_option.values()
        if data["latency_n"] > 0
    ]
    avg_costs = [
        (data["cost_total"] / data["cost_n"])
        for data in by_option.values()
        if data["cost_n"] > 0
    ]
    min_latency = min(avg_latencies) if avg_latencies else None
    max_latency = max(avg_latencies) if avg_latencies else None
    min_cost = min(avg_costs) if avg_costs else None
    max_cost = max(avg_costs) if avg_costs else None

    output: dict[str, dict[str, Any]] = {}
    for option in unique_options:
        data = by_option[option]
        n = int(data["n"])
        successes = int(data["successes"])
        avg_latency = (
            data["latency_total"] / data["latency_n"]
            if data["latency_n"] > 0
            else None
        )
        avg_cost = (
            data["cost_total"] / data["cost_n"]
            if data["cost_n"] > 0
            else None
        )
        last_used = data["last_used_at"]
        base_confidence = min(1.0, n / max(int(min_n), 1))
        confidence = round(base_confidence * _recency_factor(last_used, now_utc), 4)
        success_rate = (successes / n) if n > 0 else None
        success_component = success_rate if success_rate is not None else 0.5
        latency_penalty = _normalize_penalty(
            avg_latency, minimum=min_latency, maximum=max_latency
        )
        cost_penalty = _normalize_penalty(avg_cost, minimum=min_cost, maximum=max_cost)
        expected_utility = round(
            success_component - (0.2 * latency_penalty) - (0.2 * cost_penalty),
            6,
        )
        output[option] = {
            "n": n,
            "success_rate": round(success_rate, 6) if success_rate is not None else None,
            "avg_latency_ms": round(avg_latency, 2) if avg_latency is not None else None,
            "avg_cost": round(avg_cost, 6) if avg_cost is not None else None,
            "last_used_at": last_used.isoformat() if last_used is not None else None,
            "confidence": confidence,
            "has_sufficient_data": n >= max(int(min_n), 1),
            "expected_utility": expected_utility,
        }
    return output


def build_decision_advice(
    *,
    task_id: str,
    options: list[str],
    lookback_days: int = 30,
    min_n: int = 5,
    operations_path: str = "",
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    stats = get_option_stats(
        task_id=task_id,
        options=options,
        lookback_days=lookback_days,
        min_n=min_n,
        operations_path=operations_path,
        now=now,
    )
    recommended = ""
    if stats:
        ranked = sorted(
            stats.items(),
            key=lambda item: (
                float(item[1].get("expected_utility") or 0.0),
                float(item[1].get("confidence") or 0.0),
                int(item[1].get("n") or 0),
                item[0],
            ),
            reverse=True,
        )
        recommended = ranked[0][0]
    snapshot_at = (now or datetime.now(timezone.utc)).isoformat()
    return {
        "task_id": str(task_id or "").strip(),
        "lookback_days": int(lookback_days),
        "min_n": int(min_n),
        "snapshot_at": snapshot_at,
        "operations_path": operations_path or default_operations_path(),
        "recommended_action": recommended,
        "options": stats,
    }


__all__ = ["get_option_stats", "build_decision_advice", "default_operations_path"]
