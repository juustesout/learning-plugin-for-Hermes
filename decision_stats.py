"""Pre-decision option statistics from opslog history (read-only)."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import opslog

SUCCESS_STATUSES = {"completed", "success", "ok"}
FAIL_STATUSES = {"failed", "error", "timeout", "blocked", "cancelled"}
WILSON_Z = 1.96


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


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def wilson_interval(
    successes: int,
    n: int,
    *,
    z: float = WILSON_Z,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if n <= 0:
        return None, None, None
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, center, high


def _quality_flag(row: dict) -> Optional[bool]:
    if "quality_success" in row and row["quality_success"] is not None:
        return bool(row["quality_success"])
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get("quality_success") is not None:
        return bool(metadata["quality_success"])
    return None


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
            "latencies": [],
            "costs": [],
            "quality_n": 0,
            "quality_successes": 0,
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
        if ts is None or ts >= now_utc:
            continue
        if cutoff is not None and ts < cutoff:
            continue

        bucket = by_option[action_id]
        bucket["n"] += 1

        status = str(row.get("execution_status") or "").strip().lower()
        if status in SUCCESS_STATUSES:
            bucket["successes"] += 1

        latency = _safe_float(row.get("request_latency_ms"))
        if latency is not None and latency >= 0:
            bucket["latencies"].append(latency)

        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            cost = _safe_float(metadata.get("cost_credits"))
            if cost is not None and cost >= 0:
                bucket["costs"].append(cost)

        quality = _quality_flag(row)
        if quality is not None:
            bucket["quality_n"] += 1
            if quality:
                bucket["quality_successes"] += 1

        last_used = bucket["last_used_at"]
        if last_used is None or ts > last_used:
            bucket["last_used_at"] = ts

    med_latencies = [
        _median(data["latencies"])
        for data in by_option.values()
        if data["latencies"]
    ]
    med_costs = [
        _median(data["costs"])
        for data in by_option.values()
        if data["costs"]
    ]
    med_latencies = [value for value in med_latencies if value is not None]
    med_costs = [value for value in med_costs if value is not None]
    min_latency = min(med_latencies) if med_latencies else None
    max_latency = max(med_latencies) if med_latencies else None
    min_cost = min(med_costs) if med_costs else None
    max_cost = max(med_costs) if med_costs else None

    output: dict[str, dict[str, Any]] = {}
    for option in unique_options:
        data = by_option[option]
        n = int(data["n"])
        successes = int(data["successes"])
        latencies = data["latencies"]
        costs = data["costs"]
        avg_latency = (sum(latencies) / len(latencies)) if latencies else None
        avg_cost = (sum(costs) / len(costs)) if costs else None
        median_latency = _median(latencies)
        median_cost = _median(costs)
        last_used = data["last_used_at"]
        base_confidence = min(1.0, n / max(int(min_n), 1))
        confidence = round(base_confidence * _recency_factor(last_used, now_utc), 4)
        success_rate = (successes / n) if n > 0 else None
        wilson_low, wilson_center, wilson_high = wilson_interval(successes, n)
        quality_n = int(data["quality_n"])
        quality_rate = (
            data["quality_successes"] / quality_n if quality_n > 0 else None
        )
        success_component = wilson_low if wilson_low is not None else 0.5
        latency_penalty = _normalize_penalty(
            median_latency, minimum=min_latency, maximum=max_latency
        )
        cost_penalty = _normalize_penalty(
            median_cost, minimum=min_cost, maximum=max_cost
        )
        expected_utility = round(
            success_component - (0.2 * latency_penalty) - (0.2 * cost_penalty),
            6,
        )
        output[option] = {
            "n": n,
            "success_rate": round(success_rate, 6) if success_rate is not None else None,
            "wilson_low": round(wilson_low, 6) if wilson_low is not None else None,
            "wilson_center": round(wilson_center, 6) if wilson_center is not None else None,
            "wilson_high": round(wilson_high, 6) if wilson_high is not None else None,
            "avg_latency_ms": round(avg_latency, 2) if avg_latency is not None else None,
            "median_latency_ms": round(median_latency, 2) if median_latency is not None else None,
            "avg_cost": round(avg_cost, 6) if avg_cost is not None else None,
            "median_cost": round(median_cost, 6) if median_cost is not None else None,
            "quality_n": quality_n,
            "quality_rate": round(quality_rate, 6) if quality_rate is not None else None,
            "last_used_at": last_used.isoformat() if last_used is not None else None,
            "confidence": confidence,
            "has_sufficient_data": n >= max(int(min_n), 1),
            "expected_utility": expected_utility,
            "as_of": now_utc.isoformat(),
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
