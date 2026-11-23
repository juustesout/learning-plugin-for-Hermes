from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import decision_stats


NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


def _write_ops(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return str(path)


def test_wilson_one_success_is_not_certainty():
    low, center, high = decision_stats.wilson_interval(1, 1)
    assert low is not None and center is not None and high is not None
    assert 0.0 < low < 0.9
    assert center < 1.0
    assert high == 1.0
    empty = decision_stats.wilson_interval(0, 0)
    assert empty == (None, None, None)


def test_lookback_excludes_row_at_or_after_t(tmp_path):
    path = _write_ops(
        tmp_path / "operations.jsonl",
        [
            {
                "task_id": "t",
                "action_id": "a",
                "execution_status": "completed",
                "request_latency_ms": 10,
                "timestamp": "2026-11-22T11:59:59+00:00",
            },
            {
                "task_id": "t",
                "action_id": "a",
                "execution_status": "completed",
                "request_latency_ms": 999,
                "timestamp": "2026-11-22T12:00:00+00:00",
            },
            {
                "task_id": "t",
                "action_id": "a",
                "execution_status": "completed",
                "request_latency_ms": 999,
                "timestamp": "2026-11-22T12:00:01+00:00",
            },
        ],
    )
    stats = decision_stats.get_option_stats(
        task_id="t",
        options=["a"],
        lookback_days=30,
        min_n=1,
        operations_path=path,
        now=NOW,
    )
    assert stats["a"]["n"] == 1
    assert stats["a"]["median_latency_ms"] == 10.0
    assert stats["a"]["as_of"] == NOW.isoformat()


def test_median_latency_and_cost_not_mean(tmp_path):
    path = _write_ops(
        tmp_path / "operations.jsonl",
        [
            {
                "task_id": "t",
                "action_id": "a",
                "execution_status": "completed",
                "request_latency_ms": 100,
                "metadata": {"cost_credits": 1.0},
                "timestamp": "2026-11-22T11:00:00+00:00",
            },
            {
                "task_id": "t",
                "action_id": "a",
                "execution_status": "completed",
                "request_latency_ms": 200,
                "metadata": {"cost_credits": 2.0},
                "timestamp": "2026-11-22T11:01:00+00:00",
            },
            {
                "task_id": "t",
                "action_id": "a",
                "execution_status": "failed",
                "request_latency_ms": 10000,
                "metadata": {"cost_credits": 100.0},
                "timestamp": "2026-11-22T11:02:00+00:00",
            },
        ],
    )
    stats = decision_stats.get_option_stats(
        task_id="t",
        options=["a"],
        min_n=1,
        operations_path=path,
        now=NOW,
    )
    assert stats["a"]["n"] == 3
    assert stats["a"]["avg_latency_ms"] == 3433.33
    assert stats["a"]["median_latency_ms"] == 200.0
    assert stats["a"]["avg_cost"] == 34.333333
    assert stats["a"]["median_cost"] == 2.0
    assert stats["a"]["wilson_low"] is not None
    assert stats["a"]["wilson_low"] < stats["a"]["success_rate"]


def test_quality_is_separate_from_technical_success(tmp_path):
    path = _write_ops(
        tmp_path / "operations.jsonl",
        [
            {
                "task_id": "t",
                "action_id": "a",
                "execution_status": "completed",
                "quality_success": False,
                "timestamp": "2026-11-22T11:00:00+00:00",
            },
            {
                "task_id": "t",
                "action_id": "a",
                "execution_status": "completed",
                "metadata": {"quality_success": True},
                "timestamp": "2026-11-22T11:01:00+00:00",
            },
            {
                "task_id": "t",
                "action_id": "a",
                "execution_status": "failed",
                "timestamp": "2026-11-22T11:02:00+00:00",
            },
        ],
    )
    stats = decision_stats.get_option_stats(
        task_id="t",
        options=["a"],
        min_n=1,
        operations_path=path,
        now=NOW,
    )
    assert stats["a"]["n"] == 3
    assert stats["a"]["success_rate"] == round(2 / 3, 6)
    assert stats["a"]["quality_n"] == 2
    assert stats["a"]["quality_rate"] == 0.5


def test_old_mean_ranking_would_prefer_outlier_new_median_does_not(tmp_path):
    path = _write_ops(
        tmp_path / "operations.jsonl",
        [
            {
                "task_id": "t",
                "action_id": "stable",
                "execution_status": "completed",
                "request_latency_ms": 300,
                "metadata": {"cost_credits": 1.0},
                "timestamp": "2026-11-22T11:00:00+00:00",
            },
            {
                "task_id": "t",
                "action_id": "stable",
                "execution_status": "completed",
                "request_latency_ms": 310,
                "metadata": {"cost_credits": 1.0},
                "timestamp": "2026-11-22T11:01:00+00:00",
            },
            {
                "task_id": "t",
                "action_id": "spiky",
                "execution_status": "completed",
                "request_latency_ms": 10,
                "metadata": {"cost_credits": 0.1},
                "timestamp": "2026-11-22T11:02:00+00:00",
            },
            {
                "task_id": "t",
                "action_id": "spiky",
                "execution_status": "completed",
                "request_latency_ms": 10,
                "metadata": {"cost_credits": 0.1},
                "timestamp": "2026-11-22T11:03:00+00:00",
            },
            {
                "task_id": "t",
                "action_id": "spiky",
                "execution_status": "completed",
                "request_latency_ms": 5000,
                "metadata": {"cost_credits": 50.0},
                "timestamp": "2026-11-22T11:04:00+00:00",
            },
        ],
    )
    stats = decision_stats.get_option_stats(
        task_id="t",
        options=["stable", "spiky"],
        min_n=1,
        operations_path=path,
        now=NOW,
    )
    assert stats["spiky"]["avg_latency_ms"] > stats["stable"]["avg_latency_ms"]
    assert stats["spiky"]["median_latency_ms"] < stats["stable"]["median_latency_ms"]
    advice = decision_stats.build_decision_advice(
        task_id="t",
        options=["stable", "spiky"],
        min_n=1,
        operations_path=path,
        now=NOW,
    )
    assert advice["recommended_action"] == "spiky"
    assert stats["spiky"]["expected_utility"] >= stats["stable"]["expected_utility"]
