"""Blind labels for the HN aidevforum evidence run.

Raters see title/url/summary only. Arm, policy, rank, and auto scores stay
in assignments.jsonl, never in the rater card.
Posted items still get a 20 percent sample. The candidate pool of a run
is queued in full so the other 28 items can be labeled too.
Slot assignment and ranking live in hn_ranker.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import env_setup  # noqa: F401
import opslog

try:
    import confound
except Exception:  # pragma: no cover - optional until P5 lands
    confound = None

WORKFLOW = "hn-aidevforum"
SUCCESS_DEFINITION = "2026-11-17.3"
SAMPLE_RATE = 0.2
DAILY_BLIND_SAMPLE_LIMIT = 50
AGREEMENT_FLOOR = 0.9
ASSIGN_SALT = "hn-aidevforum-2026-11-17.2"
P_LEARNED = 0.5
ARMS = ("B", "L")
ARM_ALIASES = {"B": "B", "L": "L", "baseline": "B", "learned": "L"}
EXPECTED_POSTS_PER_DAY = 8
EXPECTED_RUNS_PER_DAY = 4
DECISION_RECORD = "decision"
OUTCOME_RECORD = "outcome"
POOL_RECORD = "pool"
SOURCE_ENV = "AGENT_LEARNING_EVIDENCE_SOURCE"
SOURCE_SIM = "sim"
SOURCE_LIVE = "live"
_WRITE_SOURCE = SOURCE_LIVE
OVERLAP_ALARM = 0.8
OVERLAP_MIN_N = 30
ARM_BALANCE_MIN_N = 30
ARM_BALANCE_ALPHA = 0.01
BALANCE_MIN_N = ARM_BALANCE_MIN_N
BALANCE_ALPHA = ARM_BALANCE_ALPHA
HIDDEN_FIELDS = frozenset(
    {
        "arm",
        "policy",
        "policy_id",
        "policy_version",
        "rank",
        "choice_prob",
        "auto_relevant",
        "auto_score",
        "rag_score",
        "source_rank",
        "selection_basis",
        "learned_score",
        "baseline",
        "action_logprob",
        "http_status",
        "auto_duplicate",
        "fallback",
        "kill_switch",
        "ranker_score",
        "pool_ids",
        "model_version",
        "slot",
        "chosen_id",
        "baseline_pick_id",
        "features",
        "run_id",
        "hn_points",
        "hn_comments",
        "age_hours",
        "title_len",
        "domain_cat",
    }
)
CANDIDATE_FEATURE_FIELDS = (
    "source_rank",
    "rag_score",
    "hn_points",
    "hn_comments",
    "age_hours",
)
VISIBLE_FIELDS = ("id", "title", "url", "summary", "posted_at", "hn_id", "workflow")
YES = {"1", "true", "yes", "y", "on"}
NO = {"0", "false", "no", "n", "off"}


def evidence_dir() -> Path:
    env = os.environ.get("AGENT_LEARNING_EVIDENCE_DIR", "").strip()
    if env:
        path = Path(env)
    else:
        path = Path(opslog.default_store_dir()) / "evidence" / WORKFLOW
    path.mkdir(parents=True, exist_ok=True)
    return path


def _path(name: str) -> Path:
    return evidence_dir() / name


def live_evidence_dir() -> Path:
    return (Path(opslog.default_store_dir()) / "evidence" / WORKFLOW).resolve()


def set_write_source(source: str) -> None:
    global _WRITE_SOURCE
    if source not in (SOURCE_LIVE, SOURCE_SIM):
        raise ValueError(f"unknown source {source!r}")
    if source == SOURCE_SIM:
        env = os.environ.get("AGENT_LEARNING_EVIDENCE_DIR", "").strip()
        if not env or Path(env).resolve() == live_evidence_dir():
            raise RuntimeError(
                "sim source may not write to the live evidence dir or to an unset dir"
            )
    _WRITE_SOURCE = source
    os.environ[SOURCE_ENV] = source


def current_source() -> str:
    if _WRITE_SOURCE == SOURCE_SIM:
        return SOURCE_SIM
    raw = os.environ.get(SOURCE_ENV, "").strip().lower()
    if raw in {SOURCE_SIM, "simulation"}:
        return SOURCE_SIM
    return SOURCE_LIVE


def is_live(row: Optional[dict]) -> bool:
    if not row:
        return False
    src = str(row.get("source") or SOURCE_LIVE).strip().lower()
    if src in {SOURCE_SIM, "simulation"}:
        return False
    if str(row.get("rater") or "").strip().lower() == SOURCE_SIM:
        return False
    if str(row.get("notes") or "").strip().lower() == SOURCE_SIM:
        return False
    return True


def is_sim(row: Optional[dict]) -> bool:
    if not row:
        return False
    return not is_live(row)


def live_only(rows: Iterable[dict]) -> list[dict]:
    return [row for row in rows if is_live(row)]


def _view(rows, view: str = "live"):
    if view == "live":
        return [row for row in rows if is_live(row)]
    if view == "sim":
        return [row for row in rows if not is_live(row)]
    raise ValueError(f"unknown view {view!r}")


def overlap_stats(decisions: list[dict]) -> dict:
    pairs = [
        (d.get("chosen_id"), d.get("baseline_pick_id"))
        for d in decisions
        if d.get("chosen_id") and d.get("baseline_pick_id")
    ]
    same = sum(1 for chosen, base in pairs if chosen == base)
    n = len(pairs)
    n_contrast = n - same
    return {
        "n": n,
        "same": same,
        "n_contrast": n_contrast,
        "rate": (same / n) if n else None,
        "contrast_rate": (n_contrast / n) if n else None,
        "primary_claim": "itt_all_posted",
        "explanatory": "n_contrast",
    }


def binom_two_sided(k: int, n: int) -> Optional[float]:
    if n == 0:
        return None
    pmf = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    pk = pmf[k]
    return min(1.0, sum(p for p in pmf if p <= pk * (1 + 1e-9)))


def balance_stats(arms: list[str]) -> dict:
    left = sum(1 for a in arms if a == "L")
    right = sum(1 for a in arms if a == "B")
    n = left + right
    p_value = binom_two_sided(left, n)
    return {
        "L": left,
        "B": right,
        "n": n,
        "p_value": p_value,
        "ok": n < BALANCE_MIN_N or p_value is None or p_value >= BALANCE_ALPHA,
    }


def ranker_health_alarms(decisions: list[dict], view: str = "live") -> tuple[list[str], dict, dict]:
    rows = _view(decisions, view)
    overlap = overlap_stats(rows)
    balance = balance_stats([str(d.get("arm") or "") for d in rows])
    alarms = []
    if overlap["n"] >= OVERLAP_MIN_N and overlap["rate"] is not None and overlap["rate"] >= OVERLAP_ALARM:
        alarms.append("high_overlap")
    if balance["n"] >= BALANCE_MIN_N and balance["p_value"] is not None and balance["p_value"] < BALANCE_ALPHA:
        alarms.append("arm_imbalance")
    return alarms, overlap, balance


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in YES:
        return True
    if text in NO:
        return False
    raise ValueError(f"expected yes/no, got {value!r}")


def quality_success(*, relevant: bool, duplicate: bool) -> bool:
    return bool(relevant) and not bool(duplicate)


def _unit(item_id: str, purpose: str) -> float:
    digest = hashlib.sha256(f"{ASSIGN_SALT}:{purpose}:{item_id}".encode("utf-8")).hexdigest()
    return (int(digest, 16) % 10000) / 10000.0


def should_sample(item_id: str, rate: float = SAMPLE_RATE) -> bool:
    return _unit(item_id, "sample") < rate


def _daily_queue_slots(limit: int = DAILY_BLIND_SAMPLE_LIMIT) -> int:
    if current_source() == SOURCE_SIM:
        return 10**9
    today = datetime.now(timezone.utc).date()
    queued_today = 0
    for row in load_queue():
        ts = _parse_ts(row.get("queued_at"))
        if ts is not None:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts.astimezone(timezone.utc).date() == today:
                queued_today += 1
    return max(0, limit - queued_today)


def normalize_arm(value: Any) -> Optional[str]:
    if value is None:
        return None
    return ARM_ALIASES.get(str(value).strip())


def assign_arm(item_id: str, p_learned: float = P_LEARNED) -> dict:
    """Legacy per-item coin. Prefer hn_ranker.assign_slot_arm for posting."""
    draw = _unit(item_id, "arm")
    arm = "L" if draw < p_learned else "B"
    return {
        "id": item_id,
        "arm": arm,
        "p_learned": p_learned,
        "choice_prob": P_LEARNED,
        "draw": draw,
        "salt": ASSIGN_SALT,
    }


def load_decisions() -> list[dict]:
    return [row for row in load_assignments() if row.get("record") == DECISION_RECORD]


def load_outcomes() -> list[dict]:
    return [row for row in load_assignments() if row.get("record") == OUTCOME_RECORD]


def latest_decision_for(item_id: str) -> Optional[dict]:
    found = None
    for row in load_decisions():
        chosen = str(row.get("chosen_id") or row.get("id") or "")
        if chosen == item_id:
            found = row
    return found


def blind_view(item: dict) -> dict:
    view = {key: item.get(key, "") for key in VISIBLE_FIELDS}
    leaked = sorted(HIDDEN_FIELDS & set(view))
    if leaked:
        raise ValueError(f"hidden fields in blind view: {leaked}")
    for key in HIDDEN_FIELDS:
        if key in view:
            raise ValueError(f"hidden field {key} leaked")
    return view


_JSONL_CACHE: dict[str, tuple[int, int, list[dict]]] = {}


def _stat_key(path: Path) -> tuple[int, int]:
    st = path.stat()
    return (int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))), int(st.st_size))


def _read_jsonl(path: Path) -> list[dict]:
    key = str(path)
    if not path.exists():
        _JSONL_CACHE.pop(key, None)
        return []
    stamp = _stat_key(path)
    cached = _JSONL_CACHE.get(key)
    if cached and cached[0] == stamp[0] and cached[1] == stamp[1]:
        return list(cached[2])
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    _JSONL_CACHE[key] = (stamp[0], stamp[1], rows)
    return list(rows)


def _append_jsonl(path: Path, row: dict) -> None:
    _append_jsonl_many(path, [row])


def _append_jsonl_many(path: Path, rows: Iterable[dict]) -> None:
    payloads = [{**row, "source": row.get("source", current_source())} for row in rows]
    if not payloads:
        return

    def _write():
        with path.open("a", encoding="utf-8") as f:
            for payload in payloads:
                f.write(json.dumps(payload, default=str) + "\n")
            f.flush()
        key = str(path)
        cached = _JSONL_CACHE.get(key)
        try:
            stamp = _stat_key(path)
        except OSError:
            _JSONL_CACHE.pop(key, None)
            return
        if cached:
            merged = list(cached[2])
            merged.extend(payloads)
            _JSONL_CACHE[key] = (stamp[0], stamp[1], merged)
        else:
            _JSONL_CACHE.pop(key, None)

    opslog._with_lock(str(path), _write)


def load_items() -> list[dict]:
    return _read_jsonl(_path("items.jsonl"))


def load_candidates() -> list[dict]:
    return _read_jsonl(_path("candidates.jsonl"))


def load_queue() -> list[dict]:
    return _read_jsonl(_path("queue.jsonl"))


def load_labels() -> list[dict]:
    return _read_jsonl(_path("labels.jsonl"))


def load_assignments() -> list[dict]:
    return _read_jsonl(_path("assignments.jsonl"))


def load_health() -> dict:
    path = _path("health.json")
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _ids(rows: Iterable[dict]) -> set[str]:
    return {str(row.get("id")) for row in rows if row.get("id")}


_STRIP_QUERY = frozenset(
    {
        "token",
        "access_token",
        "api_key",
        "key",
        "secret",
        "auth",
        "session",
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "fbclid",
        "gclid",
    }
)


def _canonical_url(url: Any) -> str:
    from workflow_identity import canonical_url

    cleaned = canonical_url(url)
    if cleaned:
        return cleaned
    return str(url or "").strip()


def public_item(item: dict) -> dict:
    record = {key: item.get(key, "") for key in VISIBLE_FIELDS if key in item or key in ("id", "title", "url", "summary")}
    record["id"] = str(item.get("id") or "").strip()
    record["title"] = str(item.get("title") or "").strip()
    record["url"] = str(item.get("url") or "").strip()
    record["summary"] = str(item.get("summary") or "").strip()
    record["workflow"] = WORKFLOW
    if item.get("article_id"):
        record["article_id"] = str(item["article_id"]).strip()
    if item.get("posted_at"):
        record["posted_at"] = item["posted_at"]
    if item.get("hn_id"):
        record["hn_id"] = item["hn_id"]
    return record


def assignment_record(item: dict, assigned: dict) -> dict:
    http_status = item.get("http_status")
    try:
        http_status = int(http_status) if http_status is not None else None
    except (TypeError, ValueError):
        http_status = None
    arm = normalize_arm(assigned.get("arm") or item.get("arm"))
    run_id = assigned.get("run_id") or item.get("run_id")
    slot = assigned.get("slot") if assigned.get("slot") is not None else item.get("slot")
    payload = {
        "record": OUTCOME_RECORD,
        "id": assigned.get("id") or item.get("id"),
        "chosen_id": assigned.get("chosen_id") or assigned.get("id") or item.get("id"),
        "arm": arm,
        "run_id": run_id,
        "slot": slot,
        "p_learned": assigned.get("p_learned", P_LEARNED),
        "choice_prob": assigned.get("choice_prob", P_LEARNED),
        "http_status": http_status,
        "technical_success": technical_success(http_status),
        "auto_relevant": item.get("auto_relevant"),
        "auto_duplicate": item.get("auto_duplicate"),
        "assigned_at": datetime.now(timezone.utc).isoformat(),
        "success_definition": SUCCESS_DEFINITION,
        "workflow": WORKFLOW,
    }
    if assigned.get("fallback") is not None:
        payload["fallback"] = assigned.get("fallback")
    if assigned.get("kill_switch") is not None:
        payload["kill_switch"] = assigned.get("kill_switch")
    return payload


def technical_success(http_status: Any) -> Optional[bool]:
    if http_status is None:
        return None
    try:
        code = int(http_status)
    except (TypeError, ValueError):
        return None
    return 200 <= code < 300


def enqueue_item(item: dict) -> dict:
    item_id = str(item.get("id") or "").strip()
    if not item_id:
        raise ValueError("item.id is required")
    title = str(item.get("title") or "").strip()
    url = str(item.get("url") or "").strip()
    if not title or not url:
        raise ValueError("item.title and item.url are required")
    existing = _ids(load_items())
    if item_id in existing:
        assigned = latest_decision_for(item_id) or next(
            (row for row in load_outcomes() if str(row.get("id") or row.get("chosen_id")) == item_id),
            {},
        )
        return {
            "id": item_id,
            "enqueued": False,
            "sampled": item_id in _ids(load_queue()),
            "arm": normalize_arm(assigned.get("arm")),
            "choice_prob": assigned.get("choice_prob"),
        }
    assigned = latest_decision_for(item_id)
    if assigned is None:
        assigned = assign_arm(item_id)
        assigned["missing_decision"] = True
    record = public_item(item)
    record.setdefault("posted_at", datetime.now(timezone.utc).isoformat())
    _append_jsonl(_path("items.jsonl"), record)
    _append_jsonl(_path("assignments.jsonl"), assignment_record(item, assigned))
    sampled = item_id in _ids(load_queue())
    if not sampled and should_sample(item_id):
        sampled = True
        _append_jsonl(
            _path("queue.jsonl"),
            {
                "id": item_id,
                "queued_at": datetime.now(timezone.utc).isoformat(),
                "success_definition": SUCCESS_DEFINITION,
                "origin": "posted",
            },
        )
    return {
        "id": item_id,
        "enqueued": True,
        "sampled": sampled,
        "arm": normalize_arm(assigned.get("arm")),
        "choice_prob": assigned.get("choice_prob", P_LEARNED),
        "http_status": item.get("http_status"),
        "missing_decision": bool(assigned.get("missing_decision")),
    }


def enqueue_pool(pool: Iterable[dict], *, run_id: str, sample: bool = True) -> dict:
    queued = _ids(load_queue())
    known = _ids(load_candidates()) | _ids(load_items())
    known_urls = set()
    for row in list(load_candidates()) + list(load_items()):
        cu = _canonical_url(row.get("canonical_url") or row.get("url"))
        if cu:
            known_urls.add(cu)
    added = 0
    sampled = 0
    skipped = 0
    cards: list[dict] = []
    queue_rows: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()
    slots = _daily_queue_slots() if sample else 0
    ranked_pool = sorted(pool or [], key=lambda raw: _unit(str(raw.get("id") or ""), "pool-order") if isinstance(raw, dict) else 1.0)
    for raw in ranked_pool:
        if not isinstance(raw, dict):
            continue
        item_id = str(raw.get("id") or "").strip()
        title = str(raw.get("title") or "").strip()
        url = str(raw.get("url") or "").strip()
        if not item_id or not title or not url:
            skipped += 1
            continue
        item_url = _canonical_url(url)
        if item_id in known or (item_url and item_url in known_urls):
            skipped += 1
            continue
        known.add(item_id)
        if item_url:
            known_urls.add(item_url)
        card = {
            **public_item(raw),
            "run_id": str(run_id),
            "record": POOL_RECORD,
            "queued_at": now,
            "canonical_url": item_url,
        }
        for key in CANDIDATE_FEATURE_FIELDS:
            if key in raw and raw[key] is not None:
                card[key] = raw[key]
        cards.append(card)
        added += 1
        if sample and slots > 0 and item_id not in queued:
            queued.add(item_id)
            queue_rows.append(
                {
                    "id": item_id,
                    "queued_at": now,
                    "success_definition": SUCCESS_DEFINITION,
                    "origin": "pool",
                    "run_id": str(run_id),
                }
            )
            sampled += 1
            slots -= 1
    if cards:
        _append_jsonl_many(_path("candidates.jsonl"), cards)
    if queue_rows:
        _append_jsonl_many(_path("queue.jsonl"), queue_rows)
    return {
        "run_id": str(run_id),
        "added": added,
        "sampled": sampled,
        "skipped": skipped,
    }


def unlabeled_queue() -> list[dict]:
    label_rows = load_labels()
    labeled = _ids(label_rows)
    cards = {row["id"]: row for row in load_candidates() if row.get("id")}
    cards.update({row["id"]: row for row in load_items() if row.get("id")})
    labeled_urls = set()
    for row in label_rows:
        direct = _canonical_url(row.get("canonical_url") or row.get("url"))
        if direct:
            labeled_urls.add(direct)
    for lid in labeled:
        lcard = cards.get(lid)
        if lcard:
            cu = _canonical_url(lcard.get("url"))
            if cu:
                labeled_urls.add(cu)
    seen_urls = set()
    pending = []
    queue_rows = sorted(load_queue(), key=lambda row: _unit(str(row.get("id") or ""), "display-order"))
    for row in queue_rows:
        item_id = str(row.get("id") or "")
        if not item_id or item_id in labeled:
            continue
        item = cards.get(item_id)
        if item:
            item_url = _canonical_url(item.get("url"))
            if item_url and (item_url in labeled_urls or item_url in seen_urls):
                continue
            if item_url:
                seen_urls.add(item_url)
            pending.append(item)
    return pending


def queue_status() -> dict:
    items = load_items()
    queue = load_queue()
    labeled = _ids(load_labels())
    pending = unlabeled_queue()
    last_enqueue = None
    for row in items:
        ts = row.get("posted_at")
        if ts and (last_enqueue is None or str(ts) > str(last_enqueue)):
            last_enqueue = ts
    for row in queue:
        ts = row.get("queued_at")
        if ts and (last_enqueue is None or str(ts) > str(last_enqueue)):
            last_enqueue = ts
    for row in load_candidates():
        ts = row.get("queued_at")
        if ts and (last_enqueue is None or str(ts) > str(last_enqueue)):
            last_enqueue = ts
    since = 0
    if last_enqueue:
        cutoff = _parse_ts(last_enqueue)
        if cutoff:
            since = sum(1 for row in items if (_parse_ts(row.get("posted_at")) or cutoff) >= cutoff)
    return {
        "pending": len(pending),
        "sampled": len(queue),
        "candidates": len(load_candidates()),
        "labeled": sum(1 for row in queue if row.get("id") in labeled),
        "posted": len(items),
        "last_enqueue": last_enqueue,
        "items_since_last_enqueue": since,
    }


def next_card() -> Optional[dict]:
    pending = unlabeled_queue()
    if not pending:
        return None
    return blind_view(pending[0])


def record_label(
    item_id: str,
    *,
    relevant: bool,
    duplicate: bool,
    needs_correction: bool,
    notes: str = "",
    rater: str = "juust",
) -> dict:
    queued = _ids(load_queue())
    if item_id not in queued:
        raise ValueError(f"{item_id} is not in the blind sample")
    cards = {str(row.get("id") or ""): row for row in load_candidates() if row.get("id")}
    cards.update({str(row.get("id") or ""): row for row in load_items() if row.get("id")})
    card = cards.get(item_id, {})
    item_url = str(card.get("url") or "").strip()
    payload = {
        "id": item_id,
        "relevant": relevant,
        "duplicate": duplicate,
        "needs_correction": needs_correction,
        "quality_success": quality_success(relevant=relevant, duplicate=duplicate),
        "notes": notes,
        "rater": rater,
        "labeled_at": datetime.now(timezone.utc).isoformat(),
        "success_definition": SUCCESS_DEFINITION,
        "workflow": WORKFLOW,
        "title": str(card.get("title") or "").strip(),
        "url": item_url,
        "canonical_url": _canonical_url(item_url),
    }
    _append_jsonl(_path("labels.jsonl"), payload)
    return payload


def _latest_labels(view: str = "live") -> dict[str, dict]:
    latest = {}
    for row in _view(load_labels(), view):
        item_id = str(row.get("id") or "")
        if item_id:
            latest[item_id] = row
    return latest


def auto_quality(assignment: dict) -> bool:
    if assignment.get("auto_duplicate") is True:
        return False
    relevant = assignment.get("auto_relevant")
    if relevant is None:
        return False
    return bool(relevant)


def _by_id(rows: Iterable[dict]) -> dict[str, dict]:
    out = {}
    for row in rows:
        item_id = str(row.get("chosen_id") or row.get("id") or "")
        if item_id:
            out[item_id] = row
    return out


def report(view: str = "live") -> dict:
    items = _view(load_items(), view)
    queue = _view(load_queue(), view)
    pending = unlabeled_queue()
    decisions = _view(load_decisions(), view)
    outcomes = _view(load_outcomes(), view)
    assignments = _by_id(decisions)
    assignments.update(_by_id(outcomes))
    latest = _latest_labels(view)
    remaining = len(pending)
    labeled_n = max(0, len(queue) - remaining)
    compared = 0
    agree = 0
    human_ok = 0
    by_arm = {arm: {"n": 0, "human_ok": 0, "technical_ok": 0} for arm in ARMS}
    for row in items:
        assigned = assignments.get(str(row.get("id") or ""), {})
        arm = normalize_arm(assigned.get("arm"))
        if arm in by_arm:
            by_arm[arm]["n"] += 1
            if assigned.get("technical_success") is True:
                by_arm[arm]["technical_ok"] += 1
    for row in queue:
        item_id = str(row.get("id") or "")
        label = latest.get(item_id)
        if not label:
            continue
        assigned = assignments.get(item_id, {})
        compared += 1
        if label.get("quality_success"):
            human_ok += 1
            arm = normalize_arm(assigned.get("arm"))
            if arm in by_arm:
                by_arm[arm]["human_ok"] += 1
        if bool(label.get("quality_success")) == auto_quality(assigned):
            agree += 1
    agreement = (agree / compared) if compared else None
    return {
        "workflow": WORKFLOW,
        "success_definition": SUCCESS_DEFINITION,
        "posted": len(items),
        "assignments": len(assignments),
        "sampled": len(queue),
        "candidates": len(_view(load_candidates(), view)),
        "labeled": labeled_n,
        "remaining": remaining,
        "sample_rate": SAMPLE_RATE,
        "p_learned": P_LEARNED,
        "by_arm": by_arm,
        "human_success_rate": (human_ok / compared) if compared else None,
        "agreement": agreement,
        "agreement_floor": AGREEMENT_FLOOR,
        "automatic_measure_ok": None if agreement is None else agreement >= AGREEMENT_FLOOR,
        "view": view,
        "overlap": selection_overlap(decisions),
        "arm_balance": arm_balance(decisions),
    }


def record_run(*, attempted: int, posted: int, failed: int, run_id: str = "") -> dict:
    payload = {
        "run_id": run_id or datetime.now(timezone.utc).isoformat(),
        "attempted": int(attempted),
        "posted": int(posted),
        "failed": int(failed),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "workflow": WORKFLOW,
    }
    _append_jsonl(_path("runs.jsonl"), payload)
    return payload


def load_runs() -> list[dict]:
    return _read_jsonl(_path("runs.jsonl"))


def _parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _binom_sf(k: int, n: int, p: float = 0.5) -> float:
    if n <= 0:
        return 1.0
    k = max(0, min(int(k), int(n)))
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
    return total


def arm_balance(decisions: Iterable[dict], *, p: float = P_LEARNED, alpha: float = ARM_BALANCE_ALPHA) -> dict:
    counts = {arm: 0 for arm in ARMS}
    n = 0
    for row in decisions:
        arm = normalize_arm(row.get("arm"))
        if arm not in counts:
            continue
        counts[arm] += 1
        n += 1
    learned = counts.get("L", 0)
    two_tail = min(1.0, 2.0 * min(_binom_sf(learned, n, p), _binom_sf(n - learned, n, 1.0 - p)))
    return {
        "n": n,
        "L": counts.get("L", 0),
        "B": counts.get("B", 0),
        "p_learned": p,
        "p_value": two_tail,
        "ok": n < ARM_BALANCE_MIN_N or two_tail >= alpha,
    }


def selection_overlap(decisions: Iterable[dict]) -> dict:
    n = 0
    same = 0
    for row in decisions:
        chosen = str(row.get("chosen_id") or "")
        baseline = str(row.get("baseline_pick_id") or "")
        if not chosen or not baseline:
            continue
        n += 1
        if chosen == baseline:
            same += 1
    learned = [row for row in decisions if normalize_arm(row.get("arm")) == "L"]
    learned_n = 0
    learned_same = 0
    for row in learned:
        chosen = str(row.get("chosen_id") or "")
        baseline = str(row.get("baseline_pick_id") or "")
        if not chosen or not baseline:
            continue
        learned_n += 1
        if chosen == baseline:
            learned_same += 1
    n_contrast = n - same
    learned_contrast = learned_n - learned_same
    return {
        "n": n,
        "same": same,
        "n_contrast": n_contrast,
        "rate": (same / n) if n else None,
        "contrast_rate": (n_contrast / n) if n else None,
        "learned_n": learned_n,
        "learned_same": learned_same,
        "learned_n_contrast": learned_contrast,
        "learned_rate": (learned_same / learned_n) if learned_n else None,
        "primary_claim": "itt_all_posted",
        "explanatory": "n_contrast",
    }


def audit(*, expected_posts: int = EXPECTED_POSTS_PER_DAY, hours: int = 24, view: str = "live") -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    items = [row for row in _view(load_items(), view) if (_parse_ts(row.get("posted_at")) or cutoff) >= cutoff]
    outcomes = [
        row
        for row in _view(load_outcomes(), view)
        if (_parse_ts(row.get("assigned_at")) or cutoff) >= cutoff
    ]
    decisions = [
        row
        for row in _view(load_decisions(), view)
        if (_parse_ts(row.get("decided_at")) or cutoff) >= cutoff
    ]
    runs = [row for row in _view(load_runs(), view) if (_parse_ts(row.get("recorded_at")) or cutoff) >= cutoff]
    attempted = sum(int(row.get("attempted") or 0) for row in runs)
    posted_runs = sum(int(row.get("posted") or 0) for row in runs)
    failed_runs = sum(int(row.get("failed") or 0) for row in runs)
    item_ids = {str(row.get("id")) for row in items if row.get("id")}
    decision_ids = {str(row.get("chosen_id") or row.get("id")) for row in decisions}
    outcome_ids = {str(row.get("chosen_id") or row.get("id")) for row in outcomes}
    alarms = []
    if len(items) != len(item_ids):
        alarms.append("duplicate_item_ids")
    if item_ids != outcome_ids:
        alarms.append("items_assignments_mismatch")
    missing_decision = sorted(item_ids - decision_ids)
    if missing_decision:
        alarms.append("posted_without_decision")
    if any(row.get("kill_switch") for row in decisions):
        alarms.append("kill_switch")
    if any(row.get("fallback") for row in decisions):
        alarms.append("ranker_fallback")
    if attempted and posted_runs + failed_runs != attempted:
        alarms.append("run_counts_inconsistent")
    if len(items) < expected_posts - 1:
        alarms.append("posted_below_expected")
    if attempted and len(items) < attempted:
        alarms.append("enqueue_lag")
    ranker_alarms, overlap, balance = ranker_health_alarms(decisions, view)
    alarms.extend(ranker_alarms)
    confound_status = (
        confound.confound_report(decisions)
        if confound is not None
        else {"ok": True, "alarms": []}
    )
    if not confound_status.get("ok"):
        alarms.extend(list(confound_status.get("alarms") or []))
    payload = {
        "ok": not alarms,
        "hours": hours,
        "expected_posts": expected_posts,
        "items": len(items),
        "outcomes": len(outcomes),
        "decisions": len(decisions),
        "assignments": len(load_assignments()),
        "runs": len(runs),
        "attempted": attempted,
        "posted_runs": posted_runs,
        "failed_runs": failed_runs,
        "posted_without_decision": missing_decision,
        "overlap": overlap,
        "balance": balance,
        "confound": confound_status,
        "arm_balance": arm_balance(decisions),
        "view": view,
        "alarms": alarms,
        "success_definition": SUCCESS_DEFINITION,
    }
    _path("health.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _print_card(card: dict) -> None:
    print(f"id: {card.get('id')}")
    print(f"posted_at: {card.get('posted_at')}")
    print(f"title: {card.get('title')}")
    print(f"url: {card.get('url')}")
    print(f"summary: {card.get('summary')}")
    print("")
    print(
        "python3 blind_label.py label --id "
        f"{card.get('id')} --relevant yes --duplicate no --needs-correction no"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="blind_label")
    sub = parser.add_subparsers(dest="cmd")

    p_en = sub.add_parser("enqueue", help="Add a posted or failed HN item")
    p_en.add_argument("--json", required=True, help="Posted item as JSON object")

    p_pool = sub.add_parser("enqueue-pool", help="Queue a run's candidate pool for blind labels")
    p_pool.add_argument("--run-id", required=True)
    p_pool.add_argument("--json", default="")
    p_pool.add_argument("--jsonl", default="")

    p_imp = sub.add_parser("import-jsonl", help="Bulk enqueue posted items")
    p_imp.add_argument("path")

    p_as = sub.add_parser("assign", help="Show coin-flip arm for an item id")
    p_as.add_argument("--id", required=True)

    p_run = sub.add_parser("record-run", help="Record a 6h job attempt count")
    p_run.add_argument("--attempted", type=int, required=True)
    p_run.add_argument("--posted", type=int, required=True)
    p_run.add_argument("--failed", type=int, required=True)
    p_run.add_argument("--run-id", default="")

    sub.add_parser("next", help="Show the next unlabeled blind card")
    sub.add_parser("report", help="Sample progress and rater/auto agreement")
    sub.add_parser("audit", help="Compare posted vs enqueued vs expected")

    p_lab = sub.add_parser("label", help="Record a blind label")
    p_lab.add_argument("--id", required=True)
    p_lab.add_argument("--relevant", required=True)
    p_lab.add_argument("--duplicate", required=True)
    p_lab.add_argument("--needs-correction", required=True)
    p_lab.add_argument("--notes", default="")
    p_lab.add_argument("--rater", default="juust")

    args = parser.parse_args(argv)
    if args.cmd == "enqueue":
        item = json.loads(args.json)
        print(json.dumps(enqueue_item(item), indent=2))
        return 0
    if args.cmd == "enqueue-pool":
        if args.json:
            payload = json.loads(args.json)
            if isinstance(payload, dict):
                pool = list(payload.get("items") or payload.get("pool") or ([payload] if payload.get("id") else []))
            else:
                pool = list(payload)
        elif args.jsonl:
            pool = _read_jsonl(Path(args.jsonl))
        else:
            print("provide --json or --jsonl", file=sys.stderr)
            return 1
        print(json.dumps(enqueue_pool(pool, run_id=args.run_id), indent=2))
        return 0
    if args.cmd == "import-jsonl":
        n = 0
        sampled = 0
        for row in _read_jsonl(Path(args.path)):
            result = enqueue_item(row)
            if result["enqueued"]:
                n += 1
            if result["sampled"]:
                sampled += 1
        print(json.dumps({"imported": n, "sampled": sampled}, indent=2))
        return 0
    if args.cmd == "next":
        card = next_card()
        if card is None:
            status = queue_status()
            last = status["last_enqueue"] or "never"
            print(
                "No unlabeled items in the blind sample. "
                f"Last enqueue: {last}. "
                f"Posted items: {status['posted']}. "
                f"Sampled: {status['sampled']}. "
                f"Labeled: {status['labeled']}."
            )
            return 0
        _print_card(card)
        return 0
    if args.cmd == "label":
        try:
            payload = record_label(
                args.id,
                relevant=parse_bool(args.relevant),
                duplicate=parse_bool(args.duplicate),
                needs_correction=parse_bool(args.needs_correction),
                notes=args.notes,
                rater=args.rater,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(json.dumps(payload, indent=2))
        return 0
    if args.cmd == "report":
        print(json.dumps(report(), indent=2))
        return 0
    if args.cmd == "assign":
        print(json.dumps(assign_arm(args.id), indent=2))
        return 0
    if args.cmd == "record-run":
        print(json.dumps(
            record_run(
                attempted=args.attempted,
                posted=args.posted,
                failed=args.failed,
                run_id=args.run_id,
            ),
            indent=2,
        ))
        return 0
    if args.cmd == "audit":
        payload = audit()
        print(json.dumps(payload, indent=2))
        return 0 if payload["ok"] else 2
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
