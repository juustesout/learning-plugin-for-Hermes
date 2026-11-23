"""Promotion gate: learner advice is off until L beats B on contrast.

Decision rule (fixed): Wilson 95 percent intervals on arm success rates must
not overlap (wilson_l_low > wilson_b_high). Chosen instead of a two-proportion
z-test so the gate reuses decision_stats.wilson_interval and stays conservative
near alpha=0.05 without a new dependency.

A 5 percentage-point gap is Wilson-separable at p approximately 0.5 only once
each arm has n>=1500 (first integer n where 788/1500 vs 712/1500 has
wilson_l_low > wilson_b_high). Contrast decisions are the ones that can differ
by arm, so MIN_N_CONTRAST=1500. MIN_N=12 for the total ITT count is unchanged
and is not a substitute for that contrast floor.

16/20 vs 10/20 (margin 0.30) is not separable: L_low=0.584 overlaps B_high=0.701.
That case must keep the gate closed.
"""

from __future__ import annotations

from typing import Any, Optional

import decision_stats

try:
    _wilson_interval = decision_stats.wilson_interval
except AttributeError:  # pragma: no cover - main decision_stats before P4
    def _wilson_interval(successes: int, n: int, *, z: float = 1.96):
        if n <= 0:
            return None, None, None
        import math

        p = successes / n
        z2 = z * z
        denom = 1.0 + z2 / n
        center = (p + z2 / (2.0 * n)) / denom
        margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
        return max(0.0, center - margin), center, min(1.0, center + margin)

MIN_N = 12
MIN_N_CONTRAST = 1500
MIN_MARGIN = 0.05
LEVEL = "confirmed"


def _rate(successes: int, n: int) -> Optional[float]:
    if n <= 0:
        return None
    return successes / n


def evaluate(report: dict[str, Any]) -> dict[str, Any]:
    n = int(report.get("n") or 0)
    n_contrast = int(report.get("n_contrast") or 0)
    n_l = int(report.get("n_l") or 0)
    n_b = int(report.get("n_b") or 0)
    succ_l = int(report.get("successes_l") or 0)
    succ_b = int(report.get("successes_b") or 0)
    n_distinct = int(report.get("n_distinct") or 0)
    constant = bool(report.get("constant_advice"))
    level = str(report.get("level") or "")
    reasons: list[str] = []
    if level and level != LEVEL:
        reasons.append(f"level!={LEVEL}")
    if n < MIN_N:
        reasons.append(f"n<{MIN_N}")
    if n_contrast < MIN_N_CONTRAST:
        reasons.append(f"n_contrast<{MIN_N_CONTRAST}")
    if n_l < 1 or n_b < 1:
        reasons.append("missing_arm")
    if n_distinct < 2:
        reasons.append("n_distinct<2")
    if constant:
        reasons.append("constant_advice")
    rate_l = _rate(succ_l, n_l)
    rate_b = _rate(succ_b, n_b)
    wilson_l = _wilson_interval(succ_l, n_l)
    wilson_b = _wilson_interval(succ_b, n_b)
    margin = None
    if rate_l is not None and rate_b is not None:
        margin = rate_l - rate_b
        if margin < MIN_MARGIN:
            reasons.append("margin_below_threshold")
    else:
        reasons.append("rate_undefined")
    conservative = None
    if wilson_l[0] is not None and wilson_b[2] is not None:
        conservative = wilson_l[0] - wilson_b[2]
        if conservative <= 0:
            reasons.append("wilson_not_separated")
    else:
        reasons.append("wilson_undefined")
    open_gate = not reasons
    return {
        "open": open_gate,
        "reason": "-" if open_gate else ",".join(reasons),
        "reasons": reasons,
        "n": n,
        "min_n": MIN_N,
        "n_contrast": n_contrast,
        "min_n_contrast": MIN_N_CONTRAST,
        "n_l": n_l,
        "n_b": n_b,
        "successes_l": succ_l,
        "successes_b": succ_b,
        "rate_l": None if rate_l is None else round(rate_l, 6),
        "rate_b": None if rate_b is None else round(rate_b, 6),
        "margin": None if margin is None else round(margin, 6),
        "min_margin": MIN_MARGIN,
        "wilson_l_low": None if wilson_l[0] is None else round(wilson_l[0], 6),
        "wilson_b_high": None if wilson_b[2] is None else round(wilson_b[2], 6),
        "conservative_margin": None if conservative is None else round(conservative, 6),
        "n_distinct": n_distinct,
        "constant_advice": constant,
        "level": level or LEVEL,
        "rollback": False,
    }


def rollback(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Close a previously open gate. Never auto-reopens from this function."""
    now = evaluate(current)
    was_open = bool(previous.get("open"))
    if was_open and not now["open"]:
        now["rollback"] = True
        now["reason"] = "rollback:" + str(now.get("reason") or "closed")
        now["previous_open"] = True
    elif was_open and now["open"]:
        now["rollback"] = False
        now["previous_open"] = True
    else:
        now["rollback"] = False
        now["previous_open"] = was_open
    return now
