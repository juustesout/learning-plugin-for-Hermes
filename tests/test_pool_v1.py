from __future__ import annotations

import json
from pathlib import Path

import pytest

import blind_label
import hn_ranker
import pool_v1


def _isolate(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    monkeypatch.setenv("AGENT_LEARNING_EVIDENCE_DIR", str(evidence))
    monkeypatch.setenv("AGENT_LEARNING_OPSLOG_DIR", str(tmp_path / "store"))
    monkeypatch.delenv(hn_ranker.KILL_SWITCH_ENV, raising=False)
    monkeypatch.delenv("AGENT_LEARNING_EVIDENCE_SOURCE", raising=False)
    evidence.mkdir(parents=True, exist_ok=True)
    return evidence


def _header(**extra) -> dict:
    payload = {
        "schema": "pool.v1",
        "run_id": "run-1",
        "workflow": "hn-aidevforum",
        "source": "hn",
        "fetched_at": "2026-11-19T12:00:00+00:00",
        "producer": "sloppy",
        "producer_version": "test",
    }
    payload.update(extra)
    return payload


def _cand(hn_id: int, rank: int, rag: float = 0.5, **extra) -> dict:
    payload = {
        "id": f"hn-{hn_id}",
        "hn_id": hn_id,
        "title": f"Title {hn_id}",
        "url": f"https://example.com/{hn_id}?token=secret",
        "summary": f"Summary {hn_id}",
        "posted_at": None,
        "source_rank": rank,
        "rag_score": rag,
        "hn_points": 10,
        "hn_comments": 2,
        "age_hours": 3.5,
    }
    payload.update(extra)
    return payload


GOLDEN = [
    _cand(10, 0, 0.10),
    _cand(11, 1, 0.90),
    _cand(12, 2, 0.80),
    _cand(13, 3, 0.20),
]


def _write_pool(path: Path, header=None, candidates=None) -> Path:
    return pool_v1.write_pool_file(path, header or _header(), candidates if candidates is not None else GOLDEN)


def test_import_queues_candidates(tmp_path, monkeypatch):
    evidence = _isolate(tmp_path, monkeypatch)
    src = _write_pool(tmp_path / "inbox" / "pool-run-1.jsonl")
    result = pool_v1.import_pool(src)
    assert result["ok"] is True
    assert result["queued"] == 4
    assert result["skipped_posted"] == 0
    queued = {row["id"] for row in blind_label.load_queue()}
    assert queued == {"hn-10", "hn-11", "hn-12", "hn-13"}
    stored = {row["id"]: row for row in blind_label.load_candidates()}
    assert stored["hn-10"]["url"] == "https://example.com/10"
    assert not src.exists()
    assert (evidence / "inbox" / "done" / "pool-run-1.jsonl").is_file()


def test_invalid_line_rejects_whole_file(tmp_path, monkeypatch):
    evidence = _isolate(tmp_path, monkeypatch)
    src = tmp_path / "pool-bad.jsonl"
    bad = list(GOLDEN)
    bad[1] = {**bad[1], "rag_score": 1.5}
    _write_pool(src, candidates=bad)
    result = pool_v1.import_pool(src)
    assert result["ok"] is False
    assert result["line"] == 3
    assert "rag_score" in result["error"]
    assert blind_label.load_queue() == []
    rejected = evidence / "inbox" / "rejected" / "pool-bad.jsonl"
    assert rejected.is_file()
    reason = Path(str(rejected) + ".reason").read_text(encoding="utf-8")
    assert "line 3" in reason


def test_duplicate_import_is_noop(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    first = _write_pool(tmp_path / "pool-run-1.jsonl")
    pool_v1.import_pool(first)
    before = list(blind_label.load_candidates())
    second = _write_pool(tmp_path / "pool-run-1-again.jsonl")
    result = pool_v1.import_pool(second)
    assert result["ok"] is True
    assert result["idempotent"] is True
    assert result["added"] == 0
    assert blind_label.load_candidates() == before


def test_already_posted_is_skipped_and_not_selected(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    posted = GOLDEN[0]
    blind_label.enqueue_item(
        {
            "id": posted["id"],
            "title": posted["title"],
            "url": "https://example.com/10",
            "summary": posted["summary"],
            "hn_id": posted["hn_id"],
            "http_status": 201,
        }
    )
    src = _write_pool(tmp_path / "pool-run-1.jsonl")
    result = pool_v1.import_pool(src)
    assert result["skipped_posted"] == 1
    assert "hn-10" not in {row["id"] for row in blind_label.load_candidates()}
    selected = pool_v1.select_for_run("run-1")
    chosen = {row["chosen_id"] for row in selected["items"]}
    assert "hn-10" not in chosen
    assert selected["status"] == "ok"


def test_select_writes_before_output_and_is_idempotent(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    pool_v1.import_pool(_write_pool(tmp_path / "pool-run-1.jsonl"))
    first = pool_v1.select_for_run("run-1")
    assert first["status"] == "ok"
    assert len(first["items"]) == 2
    stored = hn_ranker.load_decisions(view="live")
    assert len(stored) == 2
    assert {row["chosen_id"] for row in stored} == {row["chosen_id"] for row in first["items"]}
    for item in first["items"]:
        assert item["fallback"] is False
        assert item["arm"] in {"L", "B"}
        assert "baseline_pick_id" in item
    second = pool_v1.select_for_run("run-1")
    assert [row["chosen_id"] for row in second["items"]] == [row["chosen_id"] for row in first["items"]]
    assert len(hn_ranker.load_decisions(view="live")) == 2


def test_arm_b_matches_golden_baseline(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    pool_v1.import_pool(_write_pool(tmp_path / "pool-run-1.jsonl"))
    decisions = hn_ranker.select_run(
        pool_v1.candidates_for_run("run-1"),
        "run-base",
        force_arms=["B", "B"],
        write=False,
    )
    assert [row["chosen_id"] for row in decisions] == ["hn-10", "hn-11"]
    assert hn_ranker.baseline_select_ids(GOLDEN) == ["hn-10", "hn-11"]


def test_baseline_does_not_touch_store(tmp_path, monkeypatch):
    evidence = _isolate(tmp_path, monkeypatch)
    src = _write_pool(tmp_path / "pool-run-1.jsonl")

    def boom(*args, **kwargs):
        raise AssertionError("baseline must not write the store")

    monkeypatch.setattr(blind_label, "_append_jsonl", boom)
    monkeypatch.setattr(blind_label, "enqueue_pool", boom)
    monkeypatch.setattr(blind_label, "enqueue_item", boom)
    result = pool_v1.baseline_from_pool(src)
    assert result["fallback"] is True
    assert [row["chosen_id"] for row in result["items"]] == ["hn-10", "hn-11"]
    assert result["items"][0]["arm"] == "B"
    assert not list(evidence.glob("*.jsonl"))


def test_record_post_rejects_unselected_id(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    pool_v1.import_pool(_write_pool(tmp_path / "pool-run-1.jsonl"))
    selected = pool_v1.select_for_run("run-1")
    chosen = selected["items"][0]["chosen_id"]
    other = next(item["id"] for item in GOLDEN if item["id"] != chosen and item["id"] != selected["items"][1]["chosen_id"])
    refused = pool_v1.record_post(run_id="run-1", item_id=other, http_status=201)
    assert refused["ok"] is False
    assert "not in selection" in refused["error"]
    ok = pool_v1.record_post(run_id="run-1", item_id=chosen, http_status=201, fallback=False)
    assert ok["ok"] is True
    again = pool_v1.record_post(run_id="run-1", item_id=chosen, http_status=500)
    assert again["idempotent"] is True
    assert again["http_status"] == 201


def test_live_store_refuses_sim_rows(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    src = _write_pool(tmp_path / "pool-sim.jsonl", header=_header(source="sim"))
    result = pool_v1.import_pool(src)
    assert result["ok"] is False
    assert "sim" in result["error"]
    assert blind_label.load_candidates() == []
    item = _cand(99, 0)
    item["source"] = "sim"
    src2 = _write_pool(tmp_path / "pool-sim-item.jsonl", candidates=[item, GOLDEN[1], GOLDEN[2]])
    result = pool_v1.import_pool(src2)
    assert result["ok"] is False
    assert "sim" in result["error"]


def test_broken_store_is_explicit(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    src = _write_pool(tmp_path / "pool-run-1.jsonl")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(blind_label, "enqueue_pool", boom)
    result = pool_v1.import_pool(src)
    assert result["ok"] is False
    assert "store unavailable" in result["error"]
    assert src.is_file()


def test_short_run_returns_no_choice(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    src = _write_pool(tmp_path / "pool-short.jsonl", header=_header(run_id="short"), candidates=GOLDEN[:1])
    imported = pool_v1.import_pool(src)
    assert imported["status"] == "short"
    selected = pool_v1.select_for_run("short")
    assert selected["status"] == "short"
    assert selected["items"] == []


def test_strip_url_matches_canonical_trailing_slash():
    assert pool_v1.strip_url("https://example.com/10?token=secret") == "https://example.com/10"
    assert pool_v1.strip_url("https://EXAMPLE.com/story/?utm_source=hn") == "https://example.com/story"
    assert pool_v1.strip_url("https://example.com/story/") == "https://example.com/story"


def test_weekend_fixture_import_skips_duplicate_canonical_urls(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    urls = (
        "https://example.com/story/?utm_source=hn",
        "https://EXAMPLE.com/story",
        "https://example.com/story?fbclid=abc",
        "https://example.com/story/",
        "https://example.com/story/?token=secret",
    )
    candidates = [
        _cand(20 + i, i, 0.4, url=url)
        for i, url in enumerate(urls)
    ]
    src = _write_pool(
        tmp_path / "pool-weekend.jsonl",
        header=_header(run_id="weekend"),
        candidates=candidates,
    )
    result = pool_v1.import_pool(src)
    assert result["ok"] is True
    assert result["added"] == 1
    assert result["queued"] == 1
    stored = blind_label.load_candidates()
    assert len(stored) == 1
    assert stored[0]["url"] == "https://example.com/story"
    selected = pool_v1.select_for_run("weekend")
    assert selected["status"] == "short"
    assert selected["items"] == []



def test_tmp_file_is_not_read(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    src = tmp_path / "pool-run-1.jsonl.tmp"
    src.write_text("{}\n", encoding="utf-8")
    with pytest.raises(pool_v1.PoolError, match="tmp"):
        pool_v1.parse_pool_file(src)


def test_cli_pool_import(tmp_path, monkeypatch, capsys):
    _isolate(tmp_path, monkeypatch)
    src = _write_pool(tmp_path / "pool-run-1.jsonl")
    rc = pool_v1.main(["pool", "import", str(src)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["queued"] == 4
