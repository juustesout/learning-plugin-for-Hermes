from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "docs" / "archive"


def _collect_only_count() -> int:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            "tests",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    for line in reversed((proc.stdout or "").splitlines()):
        text = line.strip()
        if "test collected" in text or "tests collected" in text:
            return int(text.split()[0])
    raise AssertionError(
        "pytest --collect-only did not report a collected count:\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_root_session_docs_are_archived():
    assert not (ROOT / "SESSION-RESUME.md").exists()
    assert not (ROOT / "MONKEY.md").exists()
    assert (ARCHIVE / "SESSION-RESUME.md").is_file()
    assert (ARCHIVE / "MONKEY.md").is_file()


def test_archive_files_are_marked_stale():
    for name in ("SESSION-RESUME.md", "MONKEY.md"):
        text = (ARCHIVE / name).read_text(encoding="utf-8")
        assert text.startswith("> Archived"), name
        assert "STATUS.md" in text
        assert "not current" in text.lower()


def test_status_md_collected_matches_pytest_collect_only():
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    match = re.search(r"^- collected: (\d+)$", status, re.MULTILINE)
    assert match, "STATUS.md must contain '- collected: N'"
    reported = int(match.group(1))
    collected = _collect_only_count()
    assert reported == collected, (
        f"STATUS.md collected={reported} != pytest --collect-only {collected}; "
        "run python3 scripts/status.py"
    )


def test_status_md_hooks_are_not_unchecked_no():
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    for name in (
        "pre_tool_call",
        "post_tool_call",
        "pre_llm_call",
        "llm_execution_middleware",
    ):
        match = re.search(rf"^- {name}: (\w+)$", status, re.MULTILINE)
        assert match, f"STATUS.md missing hook line for {name}"
        assert match.group(1) in {"yes", "unknown"}, (
            f"STATUS.md {name}={match.group(1)!r} is an unchecked 'no'; "
            "generator must register or print unknown"
        )


def test_stale_session_resume_count_is_not_current():
    archived = (ARCHIVE / "SESSION-RESUME.md").read_text(encoding="utf-8")
    assert "210 passed" in archived
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    match = re.search(r"^- collected: (\d+)$", status, re.MULTILINE)
    assert match
    assert int(match.group(1)) != 210
