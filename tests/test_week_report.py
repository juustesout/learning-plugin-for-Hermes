from __future__ import annotations

import week_report


def test_format_is_plain_text(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_LEARNING_EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv("AGENT_LEARNING_OPSLOG_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("AGENT_LEARNING_LOCAL_STORE_DIR", str(tmp_path / "store"))
    text = week_report.format_report()
    assert text.startswith("week:")
    assert "itt:" in text
    assert "gate:" in text
    assert "n_contrast=" in text
    assert not text.lstrip().startswith("{")
