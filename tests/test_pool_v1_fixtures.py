from __future__ import annotations

from pathlib import Path

import pytest

import blind_label
import pool_v1
from tests.test_pool_v1 import _isolate

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "pool.v1"


def _copy(tmp_path: Path, name: str) -> Path:
    src = FIXTURES / name
    dest = tmp_path / name
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def test_fixture_valid_imports(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    result = pool_v1.import_pool(_copy(tmp_path, "valid.jsonl"))
    assert result["ok"] is True
    assert result["added"] == 3
    stored = {row["id"]: row for row in blind_label.load_candidates()}
    assert stored["hn-101"]["url"] == "https://example.com/101"


def test_fixture_invalid_line_rejects_whole_file(tmp_path, monkeypatch):
    evidence = _isolate(tmp_path, monkeypatch)
    src = _copy(tmp_path, "invalid-line.jsonl")
    result = pool_v1.import_pool(src)
    assert result["ok"] is False
    assert result["line"] == 3
    assert blind_label.load_candidates() == []
    assert (evidence / "inbox" / "rejected" / "invalid-line.jsonl").is_file()


def test_fixture_sim_is_refused(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    result = pool_v1.import_pool(_copy(tmp_path, "sim-refused.jsonl"))
    assert result["ok"] is False
    assert "sim" in result["error"]
    assert blind_label.load_candidates() == []


def test_fixture_weekend_collapses_five_ids_to_one_url(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    result = pool_v1.import_pool(_copy(tmp_path, "weekend-dupes.jsonl"))
    assert result["ok"] is True
    assert result["added"] == 1
    stored = blind_label.load_candidates()
    assert len(stored) == 1
    assert stored[0]["url"] == "https://example.com/story"
    selected = pool_v1.select_for_run("fixture-weekend")
    assert selected["status"] == "short"
    assert selected["items"] == []


def test_fixture_short_selects_nothing(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    imported = pool_v1.import_pool(_copy(tmp_path, "short.jsonl"))
    assert imported["status"] == "short"
    selected = pool_v1.select_for_run("fixture-short")
    assert selected["status"] == "short"
    assert selected["items"] == []


def test_tmp_suffix_fixture_is_never_read(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    src = tmp_path / "valid.jsonl.tmp"
    src.write_text((FIXTURES / "valid.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(pool_v1.PoolError, match="tmp"):
        pool_v1.parse_pool_file(src)
