from __future__ import annotations

from datetime import datetime, timedelta, timezone

import confound


def _row(arm: str, when: datetime) -> dict:
    return {"arm": arm, "decided_at": when.isoformat()}


def test_min_n_keeps_concentrated_sample_ok():
    start = datetime(2026, 11, 16, 9, tzinfo=timezone.utc)
    rows = [_row("L", start) for _ in range(14)]
    rows += [_row("B", start + timedelta(hours=8)) for _ in range(14)]
    report = confound.confound_report(rows)
    assert report["n"] == 28
    assert report["n"] < confound.MIN_N
    assert report["ok"] is True
    assert report["alarms"] == []
    assert confound.MIN_N == 30
    assert confound.ALPHA == 0.01


def test_hour_split_above_min_n_is_confound():
    morning = datetime(2026, 11, 16, 8, tzinfo=timezone.utc)
    evening = datetime(2026, 11, 16, 20, tzinfo=timezone.utc)
    rows = []
    for day in range(5):
        rows += [_row("L", morning + timedelta(days=day)) for _ in range(4)]
        rows += [_row("B", evening + timedelta(days=day)) for _ in range(4)]
    report = confound.confound_report(rows)
    assert report["n"] == 40
    assert report["ok"] is False
    assert "hour_confound" in report["alarms"]
    assert report["hour"]["p_value"] is not None
    assert report["hour"]["p_value"] < confound.ALPHA


def test_balanced_hours_are_ok():
    start = datetime(2026, 11, 16, 0, tzinfo=timezone.utc)
    rows = []
    for hour in range(24):
        when = start + timedelta(hours=hour)
        rows.append(_row("L", when))
        rows.append(_row("B", when))
    report = confound.confound_report(rows)
    assert report["n"] == 48
    assert report["ok"] is True
    assert report["alarms"] == []
    assert report["hour"]["p_value"] is None or report["hour"]["p_value"] >= confound.ALPHA


def test_weekday_split_above_min_n_is_confound():
    monday = datetime(2026, 11, 16, 12, tzinfo=timezone.utc)
    saturday = datetime(2026, 11, 21, 12, tzinfo=timezone.utc)
    rows = [_row("L", monday + timedelta(weeks=i)) for i in range(20)]
    rows += [_row("B", saturday + timedelta(weeks=i)) for i in range(20)]
    report = confound.confound_report(rows)
    assert report["n"] == 40
    assert "weekday_confound" in report["alarms"]
    assert report["ok"] is False
