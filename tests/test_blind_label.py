from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import blind_label


def _isolate(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    monkeypatch.setenv("AGENT_LEARNING_EVIDENCE_DIR", str(evidence))
    monkeypatch.setenv("AGENT_LEARNING_OPSLOG_DIR", str(tmp_path / "store"))
    return evidence


def _item(item_id: str, **extra) -> dict:
    payload = {
        "id": item_id,
        "title": f"Title {item_id}",
        "url": f"https://news.ycombinator.com/item?id={item_id}",
        "summary": "A post about AI development tooling.",
        "arm": "learned",
        "policy_id": "hn-ranker",
        "rank": 1,
        "auto_relevant": True,
        "auto_duplicate": False,
        "choice_prob": 0.5,
    }
    payload.update(extra)
    return payload


def test_should_sample_is_stable():
    first = [blind_label.should_sample(f"id-{i}") for i in range(200)]
    second = [blind_label.should_sample(f"id-{i}") for i in range(200)]
    assert first == second
    assert 20 <= sum(first) <= 60


def test_blind_view_hides_arm_and_scores():
    view = blind_label.blind_view(_item("x1"))
    assert view["title"].startswith("Title")
    assert view["url"].startswith("https://")
    for key in blind_label.HIDDEN_FIELDS:
        assert key not in view


def test_enqueue_samples_and_next_is_blind(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    sampled = None
    for i in range(200):
        result = blind_label.enqueue_item(_item(f"post-{i}"))
        if result["sampled"]:
            sampled = result["id"]
            break
    assert sampled
    card = blind_label.next_card()
    assert card is not None
    assert card["id"] == sampled
    assert "arm" not in json.dumps(card)
    assert "policy" not in json.dumps(card)
    assert "auto_relevant" not in json.dumps(card)


def test_label_and_agreement(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(blind_label, "should_sample", lambda item_id, rate=0.2: True)
    blind_label.enqueue_item(_item("a", auto_relevant=True, auto_duplicate=False))
    blind_label.enqueue_item(_item("b", auto_relevant=True, auto_duplicate=False))
    blind_label.record_label("a", relevant=True, duplicate=False, needs_correction=False)
    blind_label.record_label("b", relevant=False, duplicate=False, needs_correction=False)
    report = blind_label.report()
    assert report["sampled"] == 2
    assert report["labeled"] == 2
    assert report["remaining"] == 0
    assert report["agreement"] == 0.5
    assert report["automatic_measure_ok"] is False
    assert report["human_success_rate"] == 0.5


def test_label_rejects_unsampled_id(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(blind_label, "should_sample", lambda item_id, rate=0.2: False)
    blind_label.enqueue_item(_item("skip-me"))
    with pytest.raises(ValueError, match="not in the blind sample"):
        blind_label.record_label("skip-me", relevant=True, duplicate=False, needs_correction=False)


def test_cli_next_and_label(tmp_path, monkeypatch, capsys):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(blind_label, "should_sample", lambda item_id, rate=0.2: True)
    rc = blind_label.main([
        "enqueue",
        "--json",
        json.dumps(_item("cli-1", arm="baseline")),
    ])
    assert rc == 0
    enq = json.loads(capsys.readouterr().out)
    assert enq["arm"] in {"B", "L"}
    assert blind_label.main(["next"]) == 0
    out = capsys.readouterr().out
    assert "cli-1" in out
    assert "arm" not in out
    assert "learned" not in out
    assert "baseline" not in out
    assert blind_label.main([
        "label",
        "--id", "cli-1",
        "--relevant", "yes",
        "--duplicate", "no",
        "--needs-correction", "no",
    ]) == 0
    capsys.readouterr()
    assert blind_label.main(["report"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["labeled"] == 1
    assert report["automatic_measure_ok"] is True


def test_quality_success():
    assert blind_label.quality_success(relevant=True, duplicate=False) is True
    assert blind_label.quality_success(relevant=True, duplicate=True) is False
    assert blind_label.quality_success(relevant=False, duplicate=False) is False


def test_assign_arm_is_stable_and_balanced():
    first = [blind_label.assign_arm(f"id-{i}")["arm"] for i in range(200)]
    second = [blind_label.assign_arm(f"id-{i}")["arm"] for i in range(200)]
    assert first == second
    learned = first.count("L")
    assert 70 <= learned <= 130
    row = blind_label.assign_arm("id-1")
    assert row["choice_prob"] == 0.5
    assert row["p_learned"] == 0.5


def test_enqueue_splits_assignment_log(tmp_path, monkeypatch):
    evidence = _isolate(tmp_path, monkeypatch)
    result = blind_label.enqueue_item(_item("split-1", http_status=201, arm="learned"))
    items = json.loads((evidence / "items.jsonl").read_text().splitlines()[0])
    assigned = json.loads((evidence / "assignments.jsonl").read_text().splitlines()[0])
    assert "arm" not in items
    assert "choice_prob" not in items
    assert "http_status" not in items
    assert assigned["id"] == "split-1"
    assert assigned["arm"] in {"B", "L"}
    assert assigned["http_status"] == 201
    assert assigned["technical_success"] is True
    assert result["arm"] == assigned["arm"]
    card = blind_label.blind_view(items)
    assert "arm" not in card
    dumped = json.dumps(card)
    assert "learned" not in dumped
    assert '"arm"' not in dumped


def test_enqueue_failed_http(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    result = blind_label.enqueue_item(_item("fail-1", http_status=500, auto_relevant=False))
    assert result["enqueued"] is True
    assigned = blind_label.load_assignments()[0]
    assert assigned["technical_success"] is False
    assert assigned["http_status"] == 500


def test_audit_detects_enqueue_lag(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    blind_label.enqueue_item(_item("only-1", http_status=201))
    blind_label.record_run(attempted=2, posted=2, failed=0, run_id="r1")
    payload = blind_label.audit(expected_posts=8, hours=24)
    assert payload["ok"] is False
    assert "posted_below_expected" in payload["alarms"]
    assert "enqueue_lag" in payload["alarms"]
    assert "posted_without_decision" in payload["alarms"]
    assert blind_label.main(["audit"]) == 2


def test_cli_label_unsampled_is_clear(tmp_path, monkeypatch, capsys):
    _isolate(tmp_path, monkeypatch)
    rc = blind_label.main([
        "label",
        "--id", "missing",
        "--relevant", "yes",
        "--duplicate", "no",
        "--needs-correction", "no",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not in the blind sample" in err


def test_cli_next_empty_explains_status(tmp_path, monkeypatch, capsys):
    _isolate(tmp_path, monkeypatch)
    assert blind_label.main(["next"]) == 0
    out = capsys.readouterr().out
    assert "No unlabeled items in the blind sample" in out
    assert "Last enqueue: never" in out


def test_report_ignores_sim_labels(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(blind_label, "should_sample", lambda item_id, rate=0.2: True)
    monkeypatch.setenv(blind_label.SOURCE_ENV, blind_label.SOURCE_SIM)
    blind_label.enqueue_item(_item("sim-1", auto_relevant=True))
    blind_label.record_label("sim-1", relevant=True, duplicate=False, needs_correction=False, rater="sim")
    live = blind_label.report(view="live")
    sim = blind_label.report(view="sim")
    assert live["labeled"] == 0
    assert live["agreement"] is None
    assert sim["labeled"] == 1
    assert sim["agreement"] == 1.0


def test_arm_balance_and_overlap():
    balanced = [{"arm": "L", "chosen_id": "a", "baseline_pick_id": "b"} for _ in range(20)]
    balanced += [{"arm": "B", "chosen_id": "b", "baseline_pick_id": "b"} for _ in range(20)]
    assert blind_label.arm_balance(balanced)["ok"] is True
    overlap = blind_label.selection_overlap(balanced)
    assert overlap["learned_n"] == 20
    assert overlap["learned_same"] == 0
    assert overlap["n_contrast"] == 20
    assert overlap["learned_n_contrast"] == 20
    assert overlap["rate"] == 0.5
    assert overlap["primary_claim"] == "itt_all_posted"
    assert overlap["explanatory"] == "n_contrast"
    skewed = [{"arm": "L", "chosen_id": "x", "baseline_pick_id": "x"} for _ in range(40)]
    assert blind_label.arm_balance(skewed)["ok"] is False
    assert blind_label.selection_overlap(skewed)["learned_rate"] == 1.0
    assert blind_label.selection_overlap(skewed)["n_contrast"] == 0


def _pool_item(item_id: str, **extra) -> dict:
    payload = {
        "id": item_id,
        "title": f"Title {item_id}",
        "url": f"https://example.com/{item_id}",
        "summary": f"Summary {item_id}",
        "rag_score": 0.8,
        "arm": "L",
        "auto_relevant": True,
    }
    payload.update(extra)
    return payload


def test_enqueue_pool_queues_all_candidates(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    pool = [_pool_item(f"c{i}") for i in range(30)]
    result = blind_label.enqueue_pool(pool, run_id="run-pool")
    assert result["added"] == 30
    assert result["sampled"] == 30
    assert len(blind_label.load_candidates()) == 30
    assert len(blind_label.load_queue()) == 30
    card = blind_label.next_card()
    assert card is not None
    dumped = json.dumps(card)
    assert "arm" not in dumped
    assert "rag_score" not in dumped
    assert "auto_relevant" not in dumped
    stored = {row["id"]: row for row in blind_label.load_candidates()}
    assert stored["c0"]["rag_score"] == 0.8
    again = blind_label.enqueue_pool(pool, run_id="run-pool")
    assert again["added"] == 0
    assert again["skipped"] == 30


def test_posted_item_already_queued_from_pool_is_not_double_sampled(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    pool = [_pool_item("same-1")]
    blind_label.enqueue_pool(pool, run_id="run-same")
    result = blind_label.enqueue_item(_item("same-1", http_status=201))
    assert result["sampled"] is True
    assert len(blind_label.load_queue()) == 1
    assert len(blind_label.load_items()) == 1

def test_jsonl_cache_survives_frozen_mtime_on_append(tmp_path, monkeypatch):
    evidence = _isolate(tmp_path, monkeypatch)
    evidence.mkdir(parents=True, exist_ok=True)
    path = evidence / "items.jsonl"
    blind_label._JSONL_CACHE.clear()
    first = {"id": "keep-a", "title": "one"}
    path.write_text(json.dumps(first) + "\n", encoding="utf-8")
    assert blind_label._read_jsonl(path)[0]["id"] == "keep-a"
    stamp = path.stat().st_mtime_ns
    blind_label._append_jsonl(path, {"id": "keep-b", "title": "two"})
    os.utime(path, ns=(stamp, stamp))
    rows = blind_label._read_jsonl(path)
    assert [row["id"] for row in rows] == ["keep-a", "keep-b"]


def test_jsonl_cache_rejects_same_size_rewrite_with_frozen_mtime(tmp_path, monkeypatch):
    evidence = _isolate(tmp_path, monkeypatch)
    evidence.mkdir(parents=True, exist_ok=True)
    path = evidence / "items.jsonl"
    blind_label._JSONL_CACHE.clear()
    first = {"id": "aa", "title": "one"}
    second = {"id": "bb", "title": "one"}
    payload_a = json.dumps(first) + "\n"
    payload_b = json.dumps(second) + "\n"
    assert len(payload_a) == len(payload_b)
    path.write_text(payload_a, encoding="utf-8")
    assert blind_label._read_jsonl(path)[0]["id"] == "aa"
    stamp = path.stat().st_mtime_ns
    path.write_text(payload_b, encoding="utf-8")
    os.utime(path, ns=(stamp, stamp))
    assert path.stat().st_size == len(payload_a)
    assert path.stat().st_mtime_ns == stamp
    rows = blind_label._read_jsonl(path)
    assert rows[0]["id"] == "bb"


WEEKEND_URLS = (
    "https://example.com/story/?utm_source=hn",
    "https://EXAMPLE.com/story",
    "https://example.com/story?fbclid=abc",
    "https://example.com/story/",
    "https://example.com/story/?token=secret",
)


def _weekend_pool() -> list[dict]:
    return [
        _pool_item(f"hn-{i}", url=url, hn_id=i)
        for i, url in enumerate(WEEKEND_URLS, start=1)
    ]


def test_weekend_fixture_one_url_five_ids_enqueues_once(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    pool = _weekend_pool()
    assert len({row["id"] for row in pool}) == 5
    result = blind_label.enqueue_pool(pool, run_id="weekend")
    assert result["added"] == 1
    assert result["sampled"] == 1
    assert result["skipped"] == 4
    assert len(blind_label.load_candidates()) == 1
    assert len(blind_label.load_queue()) == 1
    pending = blind_label.unlabeled_queue()
    assert len(pending) == 1
    assert blind_label._canonical_url(pending[0]["url"]) == "https://example.com/story"
