"""Arm-vs-time confounding checks. Observer/report only; does not change assignment."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

ALPHA = 0.01
MIN_N = 30

HOUR_BUCKETS = 24
WEEKDAYS = 7


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


def _arm(row: dict) -> str:
    raw = str(row.get("arm") or "").strip().upper()
    if raw in {"L", "LEARNED"}:
        return "L"
    if raw in {"B", "BASELINE"}:
        return "B"
    return ""


def _timestamp(row: dict) -> Optional[datetime]:
    for key in ("decided_at", "timestamp", "assigned_at", "queued_at", "posted_at"):
        parsed = _parse_ts(row.get(key))
        if parsed is not None:
            return parsed
    return None


def _chi2_pvalue(table: list[list[int]]) -> Optional[float]:
    rows = len(table)
    cols = len(table[0]) if table else 0
    if rows < 2 or cols < 2:
        return None
    row_tot = [sum(r) for r in table]
    col_tot = [sum(table[r][c] for r in range(rows)) for c in range(cols)]
    n = sum(row_tot)
    if n <= 0:
        return None
    chi = 0.0
    dof = 0
    for r in range(rows):
        for c in range(cols):
            expected = row_tot[r] * col_tot[c] / n
            if expected <= 0:
                continue
            dof += 1
            diff = table[r][c] - expected
            chi += (diff * diff) / expected
    dof = (rows - 1) * (cols - 1)
    if dof <= 0:
        return None
    return _chi2_sf(chi, dof)


def _chi2_sf(x: float, k: int) -> float:
    if x <= 0 or k <= 0:
        return 1.0
    return max(0.0, min(1.0, _gammaincc(k / 2.0, x / 2.0)))


def _gammaincc(s: float, x: float) -> float:
    if x <= 0:
        return 1.0
    if s <= 0:
        return 0.0
    if x < s + 1.0:
        return 1.0 - _gammainc(s, x)
    tiny = 1e-30
    b = x + 1.0 - s
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    try:
        return math.exp(-x + s * math.log(x) - math.lgamma(s)) * h
    except (ValueError, OverflowError):
        return 0.0


def _gammainc(s: float, x: float) -> float:
    term = 1.0 / s
    total = term
    for n in range(1, 200):
        term *= x / (s + n)
        total += term
        if abs(term) < 1e-15 * abs(total):
            break
    try:
        return total * math.exp(-x + s * math.log(x) - math.lgamma(s))
    except (ValueError, OverflowError):
        return 0.0


def _stratum_table(pairs: list[tuple[str, int]], n_bins: int) -> list[list[int]]:
    counts = {
        "L": [0] * n_bins,
        "B": [0] * n_bins,
    }
    used: set[int] = set()
    for arm, bucket in pairs:
        if arm not in counts or bucket < 0 or bucket >= n_bins:
            continue
        counts[arm][bucket] += 1
        used.add(bucket)
    cols = sorted(used)
    if not cols:
        return []
    return [[counts["L"][i] for i in cols], [counts["B"][i] for i in cols]]


def _axis_report(pairs: list[tuple[str, int]], *, n_bins: int, name: str) -> dict:
    n = len(pairs)
    table = _stratum_table(pairs, n_bins)
    p_value = _chi2_pvalue(table) if table else None
    by_bucket: dict[str, dict[str, int]] = {}
    for arm, bucket in pairs:
        key = str(bucket)
        slot = by_bucket.setdefault(key, {"L": 0, "B": 0})
        slot[arm] = slot.get(arm, 0) + 1
    ok = n < MIN_N or p_value is None or p_value >= ALPHA
    return {
        "axis": name,
        "n": n,
        "min_n": MIN_N,
        "alpha": ALPHA,
        "p_value": None if p_value is None else round(p_value, 6),
        "ok": ok,
        "by_bucket": dict(sorted(by_bucket.items(), key=lambda item: int(item[0]))),
    }


def confound_report(decisions: Iterable[dict]) -> dict:
    hour_pairs: list[tuple[str, int]] = []
    weekday_pairs: list[tuple[str, int]] = []
    skipped = 0
    for row in decisions or []:
        if not isinstance(row, dict):
            skipped += 1
            continue
        arm = _arm(row)
        ts = _timestamp(row)
        if not arm or ts is None:
            skipped += 1
            continue
        hour_pairs.append((arm, ts.hour))
        weekday_pairs.append((arm, ts.weekday()))
    hour = _axis_report(hour_pairs, n_bins=HOUR_BUCKETS, name="hour")
    weekday = _axis_report(weekday_pairs, n_bins=WEEKDAYS, name="weekday")
    alarms = []
    if not hour["ok"]:
        alarms.append("hour_confound")
    if not weekday["ok"]:
        alarms.append("weekday_confound")
    return {
        "ok": not alarms,
        "n": hour["n"],
        "min_n": MIN_N,
        "alpha": ALPHA,
        "skipped": skipped,
        "hour": hour,
        "weekday": weekday,
        "alarms": alarms,
        "primary_claim": "itt_all_posted",
        "explanatory": "arm_vs_time",
    }
