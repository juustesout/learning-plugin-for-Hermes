"""External automation ingress: pool.v1 import, select, baseline, record-post."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

import env_setup  # noqa: F401
import blind_label
import hn_ranker
from workflow_identity import article_id, canonical_url

SCHEMA = "pool.v1"
WORKFLOW = blind_label.WORKFLOW
MAX_CANDIDATES = 50
MIN_SELECT = 2
IMPORT_RECORD = "pool_import"
HEADER_FIELDS = (
    "schema",
    "run_id",
    "workflow",
    "source",
    "fetched_at",
    "producer",
    "producer_version",
)


class PoolError(ValueError):
    def __init__(self, message: str, line: Optional[int] = None):
        self.line = line
        text = f"line {line}: {message}" if line is not None else message
        super().__init__(text)


def inbox_dir() -> Path:
    path = blind_label.evidence_dir() / "inbox"
    path.mkdir(parents=True, exist_ok=True)
    (path / "done").mkdir(parents=True, exist_ok=True)
    (path / "rejected").mkdir(parents=True, exist_ok=True)
    return path


def imports_path() -> Path:
    return blind_label._path("pool_imports.jsonl")


def load_imports() -> list[dict]:
    return blind_label._read_jsonl(imports_path())


def imported_run_ids() -> set[str]:
    return {
        str(row.get("run_id"))
        for row in load_imports()
        if row.get("run_id") and row.get("record") == IMPORT_RECORD
    }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_str(row: dict, key: str, *, allow_empty: bool = False) -> str:
    if key not in row or row[key] is None:
        raise PoolError(f"{key} is required")
    if not isinstance(row[key], str):
        raise PoolError(f"{key} must be a string")
    text = row[key].strip() if key != "summary" else row[key]
    if key == "summary":
        return str(row[key])
    if not allow_empty and not text:
        raise PoolError(f"{key} is required")
    return text


def strip_url(url: str) -> str:
    raw = str(url or "").strip()
    cleaned = canonical_url(raw)
    parts = urlsplit(cleaned or raw)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise PoolError("url must be http(s)")
    return cleaned or raw


def _parse_iso(value: Any, *, allow_null: bool = False) -> Optional[str]:
    if value is None:
        if allow_null:
            return None
        raise PoolError("timestamp is required")
    if not isinstance(value, str) or not value.strip():
        raise PoolError("timestamp must be ISO-8601")
    ts = blind_label._parse_ts(value)
    if ts is None:
        raise PoolError("timestamp must be ISO-8601")
    return value.strip()


def validate_header(row: dict) -> dict:
    extra = sorted(set(row) - set(HEADER_FIELDS))
    if extra:
        raise PoolError(f"unknown header fields: {extra}")
    schema = _require_str(row, "schema")
    if schema != SCHEMA:
        raise PoolError(f"schema must be {SCHEMA}")
    run_id = _require_str(row, "run_id")
    workflow = _require_str(row, "workflow")
    if workflow != WORKFLOW:
        raise PoolError(f"workflow must be {WORKFLOW}")
    source = _require_str(row, "source")
    if source == blind_label.SOURCE_SIM:
        raise PoolError("sim rows are refused by the live store")
    if source != "hn":
        raise PoolError("source must be hn")
    fetched_at = _parse_iso(row.get("fetched_at"))
    producer = _require_str(row, "producer")
    producer_version = _require_str(row, "producer_version")
    return {
        "schema": SCHEMA,
        "run_id": run_id,
        "workflow": workflow,
        "source": source,
        "fetched_at": fetched_at,
        "producer": producer,
        "producer_version": producer_version,
    }


def validate_candidate(row: dict, *, seen_ids: set[str], seen_hn: set[int]) -> dict:
    if str(row.get("source") or "").strip().lower() == blind_label.SOURCE_SIM:
        raise PoolError("sim rows are refused by the live store")
    hn_id = row.get("hn_id")
    if not _is_int(hn_id) or hn_id <= 0:
        raise PoolError("hn_id must be a positive int")
    item_id = _require_str(row, "id")
    expected = f"hn-{hn_id}"
    if item_id != expected:
        raise PoolError(f"id must be {expected}")
    if item_id in seen_ids:
        raise PoolError("id is not unique in this run")
    if hn_id in seen_hn:
        raise PoolError("hn_id is not unique in this run")
    title = _require_str(row, "title")
    url = strip_url(_require_str(row, "url"))
    if "summary" not in row or row["summary"] is None:
        raise PoolError("summary is required")
    if not isinstance(row["summary"], str):
        raise PoolError("summary must be a string")
    summary = row["summary"]
    if "posted_at" not in row:
        raise PoolError("posted_at is required")
    posted_at = _parse_iso(row.get("posted_at"), allow_null=True)
    source_rank = row.get("source_rank")
    if not _is_int(source_rank) or source_rank < 0:
        raise PoolError("source_rank must be an int >= 0")
    rag = row.get("rag_score")
    if isinstance(rag, bool) or not isinstance(rag, (int, float)):
        raise PoolError("rag_score must be a float 0..1")
    rag_score = float(rag)
    if rag_score < 0.0 or rag_score > 1.0:
        raise PoolError("rag_score must be a float 0..1")
    for key in ("hn_points", "hn_comments"):
        value = row.get(key)
        if not _is_int(value) or value < 0:
            raise PoolError(f"{key} must be an int >= 0")
    age = row.get("age_hours")
    if isinstance(age, bool) or not isinstance(age, (int, float)) or float(age) < 0:
        raise PoolError("age_hours must be a float >= 0")
    return {
        "id": item_id,
        "article_id": article_id(url, hn_id=hn_id, title=title),
        "hn_id": hn_id,
        "title": title,
        "url": url,
        "summary": summary,
        "posted_at": posted_at,
        "source_rank": source_rank,
        "rag_score": rag_score,
        "hn_points": int(row["hn_points"]),
        "hn_comments": int(row["hn_comments"]),
        "age_hours": float(age),
    }


def _read_jsonl_lines(path: Path) -> list[tuple[int, dict]]:
    if path.suffix == ".tmp" or path.name.endswith(".tmp"):
        raise PoolError("incomplete file (.tmp) is never read")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PoolError(f"store unavailable: {exc}") from exc
    rows = []
    for index, line in enumerate(raw.splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PoolError(f"invalid json ({exc.msg})", line=index) from exc
        if not isinstance(payload, dict):
            raise PoolError("expected a JSON object", line=index)
        rows.append((index, payload))
    if not rows:
        raise PoolError("file is empty", line=1)
    return rows


def parse_pool_file(path: Path) -> tuple[dict, list[dict]]:
    rows = _read_jsonl_lines(Path(path))
    line_no, header_raw = rows[0]
    try:
        header = validate_header(header_raw)
    except PoolError as exc:
        raise PoolError(str(exc) if exc.line else exc.args[0], line=line_no) from exc
    candidates = []
    seen_ids: set[str] = set()
    seen_hn: set[int] = set()
    for line_no, raw in rows[1:]:
        if len(candidates) >= MAX_CANDIDATES:
            raise PoolError(f"more than {MAX_CANDIDATES} candidates", line=line_no)
        try:
            item = validate_candidate(raw, seen_ids=seen_ids, seen_hn=seen_hn)
        except PoolError as exc:
            raise PoolError(exc.args[0], line=line_no) from exc
        seen_ids.add(item["id"])
        seen_hn.add(item["hn_id"])
        candidates.append(item)
    return header, candidates


def _canonical_url(value: Any) -> str:
    cleaned = canonical_url(value)
    if cleaned:
        return cleaned
    try:
        return strip_url(str(value or ""))
    except PoolError:
        return str(value or "").strip().lower()


def posted_keys(
    *,
    include_queue: bool = False,
    include_candidates: bool = False,
    include_assignments: bool = False,
) -> tuple[set[str], set[str], set[str]]:
    ids: set[str] = set()
    hn_ids: set[str] = set()
    urls: set[str] = set()

    def _add_row(row: dict) -> None:
        item_id = str(row.get("id") or "").strip()
        if item_id:
            ids.add(item_id)
        hn = row.get("hn_id")
        if hn is not None and str(hn).strip():
            hn_ids.add(str(hn).strip())
        if row.get("url"):
            urls.add(_canonical_url(row.get("url")))

    for row in blind_label.load_items():
        _add_row(row)
    if include_queue:
        for row in blind_label.load_queue():
            _add_row(row)
    if include_candidates:
        for row in blind_label.load_candidates():
            _add_row(row)
    if include_assignments:
        for row in blind_label.load_assignments():
            item_id = str(row.get("id") or "").strip()
            chosen_id = str(row.get("chosen_id") or "").strip()
            if item_id:
                ids.add(item_id)
            if chosen_id:
                ids.add(chosen_id)
    return ids, hn_ids, urls


def _is_posted(item: dict, posted_ids: set[str], posted_hn: set[str], posted_urls: set[str]) -> bool:
    if item["id"] in posted_ids:
        return True
    if str(item["hn_id"]) in posted_hn:
        return True
    if item["url"] in posted_urls:
        return True
    if item.get("posted_at"):
        return True
    return False


def _reason_text(exc: Exception) -> str:
    return str(exc)


def _write_reason(path: Path, text: str) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text.rstrip() + "\n", encoding="utf-8")
    tmp.replace(path)


def _move(path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    if path.resolve() != dest.resolve():
        path.replace(dest)
    return dest


def _archive(src: Path, folder: Path, reason: Optional[str] = None) -> Optional[Path]:
    if not src.exists():
        return None
    dest = _move(src, folder)
    if reason is not None:
        _write_reason(Path(str(dest) + ".reason"), reason)
    return dest


def refuse_sim_live() -> None:
    if blind_label.current_source() == blind_label.SOURCE_SIM:
        live = blind_label.live_evidence_dir()
        current = blind_label.evidence_dir().resolve()
        if current == live:
            raise PoolError("sim rows are refused by the live store")


def import_pool(path: str | Path) -> dict:
    refuse_sim_live()
    src = Path(path)
    if src.suffix == ".tmp" or src.name.endswith(".tmp"):
        return {"ok": False, "error": "incomplete file (.tmp) is never read", "path": str(src)}
    inbox = inbox_dir()
    try:
        header, candidates = parse_pool_file(src)
    except PoolError as exc:
        try:
            _archive(src, inbox / "rejected", _reason_text(exc))
        except OSError:
            pass
        return {
            "ok": False,
            "error": _reason_text(exc),
            "line": exc.line,
            "path": str(src),
        }
    except OSError as exc:
        return {"ok": False, "error": f"store unavailable: {exc}"}
    run_id = header["run_id"]
    if run_id in imported_run_ids():
        try:
            _archive(src, inbox / "done")
        except OSError:
            pass
        return {
            "ok": True,
            "idempotent": True,
            "run_id": run_id,
            "added": 0,
            "queued": 0,
            "skipped_posted": 0,
        }
    posted_ids, posted_hn, posted_urls = posted_keys(
        include_queue=True,
        include_candidates=True,
        include_assignments=True,
    )
    remaining = []
    skipped_posted = 0
    for item in candidates:
        if _is_posted(item, posted_ids, posted_hn, posted_urls):
            skipped_posted += 1
            continue
        remaining.append(item)
    try:
        queued = blind_label.enqueue_pool(remaining, run_id=run_id)
        record = {
            "record": IMPORT_RECORD,
            "run_id": run_id,
            "schema": SCHEMA,
            "n": len(candidates),
            "queued": queued.get("sampled", 0),
            "added": queued.get("added", 0),
            "skipped_posted": skipped_posted,
            "skipped": queued.get("skipped", 0),
            "producer": header["producer"],
            "producer_version": header["producer_version"],
            "fetched_at": header["fetched_at"],
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "workflow": WORKFLOW,
        }
        blind_label._append_jsonl(imports_path(), record)
        _archive(src, inbox / "done")
    except OSError as exc:
        return {"ok": False, "error": f"store unavailable: {exc}", "run_id": run_id}
    return {
        "ok": True,
        "idempotent": False,
        "run_id": run_id,
        "n": len(candidates),
        "added": queued.get("added", 0),
        "queued": queued.get("sampled", 0),
        "skipped_posted": skipped_posted,
        "status": "short" if len(remaining) < MIN_SELECT else "ok",
    }


def candidates_for_run(run_id: str) -> list[dict]:
    rows = [
        row
        for row in blind_label.load_candidates()
        if str(row.get("run_id") or "") == str(run_id)
    ]
    return hn_ranker.normalize_pool(rows)


def decisions_for_run(run_id: str) -> list[dict]:
    rows = [
        row
        for row in hn_ranker.load_decisions(view="live")
        if str(row.get("run_id") or "") == str(run_id)
    ]
    rows.sort(key=lambda row: int(row.get("slot") or 0))
    return rows


def _choice_item(decision: dict, pool: list[dict], *, fallback: bool) -> dict:
    by_id = {row["id"]: row for row in pool}
    chosen = by_id.get(str(decision.get("chosen_id") or decision.get("id") or ""), {})
    return {
        "slot": int(decision.get("slot") or 0),
        "arm": decision.get("arm"),
        "chosen_id": decision.get("chosen_id") or decision.get("id"),
        "id": decision.get("chosen_id") or decision.get("id"),
        "title": chosen.get("title") or "",
        "url": chosen.get("url") or "",
        "summary": chosen.get("summary") or "",
        "baseline_pick_id": decision.get("baseline_pick_id"),
        "fallback": bool(decision.get("fallback", fallback)),
    }


def _select_payload(run_id: str, decisions: list[dict], pool: list[dict], *, fallback: bool, status: str) -> dict:
    return {
        "status": status,
        "run_id": str(run_id),
        "items": [_choice_item(row, pool, fallback=fallback) for row in decisions],
    }


def select_for_run(run_id: str) -> dict:
    refuse_sim_live()
    try:
        existing = decisions_for_run(run_id)
        pool = candidates_for_run(run_id)
        if existing:
            return _select_payload(run_id, existing, pool, fallback=False, status="ok")
        previous_short = any(
            str(row.get("run_id") or "") == str(run_id) and row.get("status") == "short"
            for row in blind_label.load_assignments()
        )
        if previous_short:
            return {"status": "short", "run_id": str(run_id), "n": len(pool), "items": []}
        posted_ids, _, _ = posted_keys()
        remaining = hn_ranker.normalize_pool(pool, posted_ids=posted_ids)
        if len(remaining) < MIN_SELECT:
            record = {
                "record": "select",
                "run_id": str(run_id),
                "status": "short",
                "n": len(remaining),
                "decided_at": datetime.now(timezone.utc).isoformat(),
                "workflow": WORKFLOW,
            }
            blind_label._append_jsonl(blind_label._path("assignments.jsonl"), record)
            return {"status": "short", "run_id": str(run_id), "n": len(remaining), "items": []}
        decisions = hn_ranker.select_run(remaining, run_id, write=True)
        return _select_payload(run_id, decisions, remaining, fallback=False, status="ok")
    except OSError as exc:
        return {"ok": False, "status": "error", "error": f"store unavailable: {exc}", "run_id": str(run_id)}


def baseline_from_pool(path: str | Path) -> dict:
    header, candidates = parse_pool_file(Path(path))
    ordered = hn_ranker.normalize_pool(candidates)
    chosen_ids = hn_ranker.baseline_select_ids(ordered, n=hn_ranker.POSTS_PER_RUN)
    if len(chosen_ids) < MIN_SELECT:
        return {"status": "short", "run_id": header["run_id"], "n": len(ordered), "items": [], "fallback": True}
    by_id = {row["id"]: row for row in ordered}
    items = []
    remaining = list(ordered)
    for slot, item_id in enumerate(chosen_ids, start=1):
        base = hn_ranker.baseline_pick(remaining)
        chosen = by_id[item_id]
        items.append(
            {
                "slot": slot,
                "arm": "B",
                "chosen_id": item_id,
                "id": item_id,
                "title": chosen.get("title") or "",
                "url": chosen.get("url") or "",
                "summary": chosen.get("summary") or "",
                "baseline_pick_id": base["id"] if base else item_id,
                "fallback": True,
            }
        )
        remaining = [row for row in remaining if row["id"] != item_id]
    return {"status": "ok", "run_id": header["run_id"], "items": items, "fallback": True}


def record_post(
    *,
    run_id: str,
    item_id: str,
    http_status: int,
    fallback: bool = False,
) -> dict:
    refuse_sim_live()
    chosen = str(item_id).strip()
    try:
        selected = {str(row.get("chosen_id") or row.get("id")) for row in decisions_for_run(run_id)}
        if chosen not in selected:
            return {
                "ok": False,
                "error": "id not in selection",
                "run_id": str(run_id),
                "id": chosen,
            }
        existing = [
            row
            for row in hn_ranker.load_outcomes(view="live")
            if str(row.get("run_id") or "") == str(run_id)
            and str(row.get("chosen_id") or row.get("id") or "") == chosen
        ]
        if existing:
            return {
                "ok": True,
                "idempotent": True,
                "run_id": str(run_id),
                "id": chosen,
                "http_status": existing[-1].get("http_status"),
            }
        decision = next(
            (row for row in decisions_for_run(run_id) if str(row.get("chosen_id") or row.get("id")) == chosen),
            {},
        )
        pool_item = next((row for row in candidates_for_run(run_id) if str(row.get("id")) == chosen), {"id": chosen})
        item = {
            **pool_item,
            "id": chosen,
            "http_status": int(http_status),
            "run_id": str(run_id),
            "slot": decision.get("slot"),
            "arm": decision.get("arm"),
            "fallback": bool(fallback or decision.get("fallback")),
        }
        result = blind_label.enqueue_item(item)
        return {
            "ok": True,
            "idempotent": False,
            "run_id": str(run_id),
            "id": chosen,
            "http_status": int(http_status),
            "enqueued": result.get("enqueued"),
            "fallback": bool(item["fallback"]),
        }
    except OSError as exc:
        return {"ok": False, "error": f"store unavailable: {exc}", "run_id": str(run_id), "id": chosen}


def write_pool_file(path: str | Path, header: dict, candidates: list[dict]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    lines = [json.dumps(header, default=str)]
    lines.extend(json.dumps(row, default=str) for row in candidates)
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(dest)
    return dest


def _print(payload: dict) -> int:
    print(json.dumps(payload, indent=2, default=str))
    if payload.get("ok") is False or payload.get("status") == "error":
        return 1
    return 0


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="agent-learning")
    sub = parser.add_subparsers(dest="cmd")
    p_pool = sub.add_parser("pool", help="Import a pool.v1 file")
    pool_sub = p_pool.add_subparsers(dest="pool_cmd")
    p_imp = pool_sub.add_parser("import", help="Validate a pool.v1 file and queue candidates")
    p_imp.add_argument("path")
    p_imp_alias = sub.add_parser("import", help="Validate a pool.v1 file and queue candidates")
    p_imp_alias.add_argument("path")
    p_sel = sub.add_parser("select", help="Choose two items for a run")
    p_sel.add_argument("--run-id", required=True)
    p_base = sub.add_parser("baseline", help="Baseline two items without writing the store")
    p_base.add_argument("--pool", required=True)
    p_post = sub.add_parser("record-post", help="Record HTTP result for a selected id")
    p_post.add_argument("--run-id", required=True)
    p_post.add_argument("--id", required=True)
    p_post.add_argument("--http-status", type=int, required=True)
    p_post.add_argument("--fallback", default="false")
    args = parser.parse_args(argv)
    try:
        if args.cmd == "import" or (args.cmd == "pool" and getattr(args, "pool_cmd", None) == "import"):
            return _print(import_pool(args.path))
        if args.cmd == "select":
            return _print(select_for_run(args.run_id))
        if args.cmd == "baseline":
            return _print(baseline_from_pool(args.pool))
        if args.cmd == "record-post":
            return _print(
                record_post(
                    run_id=args.run_id,
                    item_id=args.id,
                    http_status=args.http_status,
                    fallback=blind_label.parse_bool(args.fallback),
                )
            )
    except PoolError as exc:
        return _print({"ok": False, "error": str(exc), "line": exc.line})
    except OSError as exc:
        return _print({"ok": False, "error": f"store unavailable: {exc}"})
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
