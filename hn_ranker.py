"""HN aidevforum item ranker: slot-level coin flip, baseline vs learned.

The ranker only chooses among the RAG-filtered, de-duplicated pool.
v1 is greedy. Randomization is one coin per slot, not per item.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlsplit

import numpy as np

import env_setup  # noqa: F401
import blind_label

WORKFLOW = blind_label.WORKFLOW
SUCCESS_DEFINITION = "2026-11-17.3"
ASSIGN_SALT = "hn-aidevforum-2026-11-17.3"
P_LEARNED = 0.5
POSTS_PER_RUN = 2
MIN_TRAIN_N = 200
KILL_SWITCH_ENV = "AGENT_LEARNING_HN_RANKER_KILL"
PRIOR_VERSION = "prior-v1"
LOGREG_VERSION = "logreg-v1"
DECISION_RECORD = "decision"
OUTCOME_RECORD = "outcome"

LOGGED_NUMERIC = (
    "source_rank",
    "rag_score",
    "hn_points",
    "hn_comments",
    "age_hours",
    "title_len",
)
TRAIN_NUMERIC = (
    "source_rank",
    "rag_score",
    "age_hours",
    "title_len",
)
NUMERIC_FEATURES = LOGGED_NUMERIC
DOMAIN_CATS = (
    "arxiv",
    "blog",
    "docs",
    "github",
    "news",
    "other",
    "social",
    "video",
)
FEATURE_ALLOWLIST = frozenset(NUMERIC_FEATURES + ("domain_cat",))
FORBIDDEN_FIELDS = frozenset(
    {
        "posted_at",
        "http_status",
        "labels",
        "auto_relevant",
        "auto_duplicate",
        "relevant",
        "duplicate",
        "needs_correction",
        "quality_success",
        "arm",
        "ranker_score",
        "chosen_id",
        "label",
        "labeled_at",
        "rater",
    }
)
_DOMAIN_RULES = (
    (("arxiv.org",), "arxiv"),
    (("github.com", "gist.github.com"), "github"),
    (("youtube.com", "youtu.be", "vimeo.com"), "video"),
    (("twitter.com", "x.com", "bsky.app"), "social"),
    (("readthedocs.io", "wikipedia.org"), "docs"),
    (("medium.com", "substack.com"), "blog"),
    (("techcrunch.com", "arstechnica.com", "wired.com", "theverge.com"), "news"),
)


def kill_switch_enabled() -> bool:
    return os.environ.get(KILL_SWITCH_ENV, "").strip().lower() in blind_label.YES


def _unit(text: str) -> float:
    digest = hashlib.sha256(f"{ASSIGN_SALT}:{text}".encode("utf-8")).hexdigest()
    return (int(digest, 16) % 10000) / 10000.0


def assign_slot_arm(run_id: str, slot: int, p_learned: float = P_LEARNED) -> dict:
    draw = _unit(f"slot:{run_id}:{slot}")
    killed = kill_switch_enabled()
    arm = "B" if killed else ("L" if draw < p_learned else "B")
    return {
        "run_id": str(run_id),
        "slot": int(slot),
        "arm": arm,
        "p_learned": p_learned,
        "choice_prob": P_LEARNED,
        "draw": draw,
        "salt": ASSIGN_SALT,
        "kill_switch": killed,
    }


def domain_cat(url: str) -> str:
    host = (urlsplit(str(url or "")).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    for suffixes, cat in _DOMAIN_RULES:
        if any(host == s or host.endswith("." + s) for s in suffixes):
            return cat
    return "other"


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def ensure_allowed(features: dict) -> dict:
    banned = sorted(FORBIDDEN_FIELDS & set(features))
    if banned:
        raise ValueError(f"forbidden feature fields: {banned}")
    extra = sorted(set(features) - FEATURE_ALLOWLIST)
    if extra:
        raise ValueError(f"unknown feature fields: {extra}")
    return features


def extract_features(item: dict, *, source_rank: Optional[int] = None) -> dict:
    title = str(item.get("title") or "")
    rank = item.get("source_rank") if item.get("source_rank") is not None else source_rank
    features = {
        "source_rank": _num(rank, 0.0),
        "rag_score": _num(item.get("rag_score"), 0.0),
        "hn_points": _num(item.get("hn_points"), 0.0),
        "hn_comments": _num(item.get("hn_comments"), 0.0),
        "age_hours": _num(item.get("age_hours"), 0.0),
        "title_len": float(len(title.split())),
        "domain_cat": domain_cat(str(item.get("url") or item.get("domain") or "")),
    }
    return ensure_allowed(features)


def normalize_pool(pool: Iterable[dict], posted_ids: Optional[Iterable[str]] = None) -> list[dict]:
    posted = {str(x) for x in (posted_ids or []) if x}
    out = []
    seen = set()
    seen_urls = set()
    for i, item in enumerate(pool or []):
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        if not item_id or item_id in posted or item_id in seen:
            continue
        item_url = blind_label._canonical_url(item.get("canonical_url") or item.get("url"))
        if item_url and item_url in seen_urls:
            continue
        row = dict(item)
        row["id"] = item_id
        if item_url:
            row["canonical_url"] = item_url
            seen_urls.add(item_url)
        if row.get("source_rank") is None:
            row["source_rank"] = i
        seen.add(item_id)
        out.append(row)
    out.sort(key=lambda row: (_num(row.get("source_rank"), 10**9), row["id"]))
    return out


def baseline_pick(pool: list[dict]) -> Optional[dict]:
    ordered = normalize_pool(pool)
    return ordered[0] if ordered else None


def baseline_select_ids(pool: list[dict], n: int = POSTS_PER_RUN) -> list[str]:
    remaining = normalize_pool(pool)
    chosen = []
    for _ in range(n):
        item = baseline_pick(remaining)
        if item is None:
            break
        chosen.append(item["id"])
        remaining = [row for row in remaining if row["id"] != item["id"]]
    return chosen


def feature_vector(
    features: dict,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> np.ndarray:
    numeric = np.array([_num(features.get(name)) for name in TRAIN_NUMERIC], dtype=float)
    if mean is not None and std is not None:
        safe = np.where(np.asarray(std) == 0, 1.0, np.asarray(std))
        numeric = (numeric - np.asarray(mean)) / safe
    onehot = np.array(
        [1.0 if features.get("domain_cat") == cat else 0.0 for cat in DOMAIN_CATS],
        dtype=float,
    )
    return np.concatenate([numeric, onehot, np.array([1.0])])


def score_item(item: dict, model: Optional[dict] = None) -> float:
    nested = item.get("features") if isinstance(item.get("features"), dict) else {}
    features = extract_features({**item, **nested})
    version = (model or {}).get("model_version") or PRIOR_VERSION
    if version == PRIOR_VERSION or not model or model.get("weights") is None:
        return float(features.get("rag_score") or 0.0)
    weights = np.asarray(model["weights"], dtype=float)
    vec = feature_vector(features, mean=model.get("mean"), std=model.get("std"))
    z = float(np.clip(vec @ weights, -30.0, 30.0))
    return float(1.0 / (1.0 + np.exp(-z)))


def ranker_pick(pool: list[dict], model: Optional[dict] = None) -> tuple[Optional[dict], Optional[float]]:
    remaining = normalize_pool(pool)
    if not remaining:
        return None, None
    scored = []
    for item in remaining:
        scored.append(
            (
                score_item(item, model),
                -_num(item.get("source_rank"), 10**9),
                item["id"],
                item,
            )
        )
    scored.sort(reverse=True)
    best = scored[0]
    return best[3], float(best[0])


def load_model() -> dict:
    path = blind_label.evidence_dir() / "ranker_model.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"model_version": PRIOR_VERSION}
    if not isinstance(payload, dict) or payload.get("weights") is None:
        return {"model_version": PRIOR_VERSION}
    return payload


def save_model(model: dict) -> None:
    path = blind_label.evidence_dir() / "ranker_model.json"
    path.write_text(json.dumps(model, indent=2, default=str) + "\n", encoding="utf-8")


def _row_ts(row: dict) -> Optional[datetime]:
    return blind_label._parse_ts(
        row.get("labeled_at") or row.get("posted_at") or row.get("decided_at")
    )


def select_training_rows(rows: Iterable[dict], cutoff: datetime) -> list[dict]:
    late = []
    selected = []
    for row in rows:
        ts = _row_ts(row)
        if ts is None:
            continue
        if ts > cutoff:
            late.append(row)
        else:
            selected.append(row)
    if late:
        raise ValueError(f"{len(late)} training labels after cutoff")
    return selected


def _features_from_row(row: dict) -> dict:
    feats = row.get("features")
    if isinstance(feats, dict) and feats:
        allowed = {k: feats[k] for k in FEATURE_ALLOWLIST if k in feats}
        if set(allowed) == FEATURE_ALLOWLIST:
            return ensure_allowed(allowed)
    return extract_features(row)


def fit_model(
    rows: list[dict],
    cutoff: datetime,
    *,
    min_n: int = MIN_TRAIN_N,
    l2: float = 1.0,
    steps: int = 400,
    lr: float = 0.4,
) -> dict:
    selected = select_training_rows(rows, cutoff)
    if len(selected) < min_n:
        raise ValueError(f"not enough training rows: {len(selected)} < {min_n}")
    cleaned = [_features_from_row(row) for row in selected]
    labels = [1.0 if row.get("quality_success") else 0.0 for row in selected]
    numeric = np.array(
        [[_num(feats.get(name)) for name in TRAIN_NUMERIC] for feats in cleaned],
        dtype=float,
    )
    mean = numeric.mean(axis=0)
    std = numeric.std(axis=0)
    X = np.array([feature_vector(feats, mean=mean, std=std) for feats in cleaned], dtype=float)
    y = np.array(labels, dtype=float)
    weights = np.zeros(X.shape[1], dtype=float)
    n = max(len(y), 1)
    reg = np.ones_like(weights)
    reg[-1] = 0.0
    for _ in range(steps):
        z = np.clip(X @ weights, -30.0, 30.0)
        pred = 1.0 / (1.0 + np.exp(-z))
        grad = (X.T @ (pred - y)) / n + (l2 / n) * (reg * weights)
        weights -= lr * grad
    return {
        "model_version": LOGREG_VERSION,
        "weights": [float(x) for x in weights],
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "l2": l2,
        "n": int(len(selected)),
        "cutoff": cutoff.isoformat(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(TRAIN_NUMERIC) + list(DOMAIN_CATS) + ["bias"],
        "train_label": "human_blind_label",
        "success_definition": SUCCESS_DEFINITION,
    }


def can_train(*, n_human_labeled: int) -> bool:
    return n_human_labeled >= MIN_TRAIN_N


def _pick_learned(remaining: list[dict], model: Optional[dict]) -> tuple[dict, float, str]:
    ids = {row["id"] for row in remaining}
    chosen, score = ranker_pick(remaining, model)
    if chosen is None or chosen["id"] not in ids:
        raise ValueError("ranker chose an item outside the pool")
    version = (model or {}).get("model_version") or PRIOR_VERSION
    return chosen, float(score), version


def decision_record(
    *,
    run_id: str,
    slot: int,
    arm: str,
    pool: list[dict],
    chosen: dict,
    ranker_score: Optional[float],
    baseline_id: Optional[str],
    model_version: str,
    fallback: bool,
    kill_switch: bool,
) -> dict:
    return {
        "record": DECISION_RECORD,
        "run_id": str(run_id),
        "slot": int(slot),
        "arm": arm,
        "pool_ids": [row["id"] for row in pool],
        "chosen_id": chosen["id"],
        "id": chosen["id"],
        "ranker_score": None if ranker_score is None else float(ranker_score),
        "baseline_pick_id": baseline_id,
        "model_version": model_version,
        "features": extract_features(chosen),
        "fallback": bool(fallback),
        "kill_switch": bool(kill_switch),
        "p_learned": P_LEARNED,
        "choice_prob": P_LEARNED,
        "salt": ASSIGN_SALT,
        "decided_at": datetime.now(timezone.utc).isoformat(),
        "success_definition": SUCCESS_DEFINITION,
        "workflow": WORKFLOW,
    }


def select_run(
    pool: list[dict],
    run_id: str,
    *,
    slots: int = POSTS_PER_RUN,
    model: Optional[dict] = None,
    posted_ids: Optional[Iterable[str]] = None,
    write: bool = True,
    force_arms: Optional[Iterable[str]] = None,
) -> list[dict]:
    remaining = normalize_pool(pool, posted_ids=posted_ids)
    resolved = model if model is not None else load_model()
    forced = [str(a) for a in force_arms] if force_arms is not None else None
    decisions = []
    for slot in range(1, int(slots) + 1):
        if not remaining:
            break
        assigned = assign_slot_arm(run_id, slot)
        if forced is not None and slot - 1 < len(forced) and not assigned["kill_switch"]:
            assigned["arm"] = forced[slot - 1]
        arm = assigned["arm"]
        base = baseline_pick(remaining)
        baseline_id = base["id"] if base else None
        fallback = False
        ranker_score = None
        model_version = resolved.get("model_version") or PRIOR_VERSION
        chosen = base
        if arm == "L":
            try:
                chosen, ranker_score, model_version = _pick_learned(remaining, resolved)
            except Exception:
                chosen = base
                ranker_score = None
                arm = "B"
                fallback = True
                model_version = resolved.get("model_version") or PRIOR_VERSION
        if chosen is None:
            break
        record = decision_record(
            run_id=run_id,
            slot=slot,
            arm=arm,
            pool=remaining,
            chosen=chosen,
            ranker_score=ranker_score if arm == "L" else None,
            baseline_id=baseline_id,
            model_version=model_version,
            fallback=fallback,
            kill_switch=assigned["kill_switch"],
        )
        if write:
            blind_label._append_jsonl(blind_label._path("assignments.jsonl"), record)
        decisions.append(record)
        remaining = [row for row in remaining if row["id"] != chosen["id"]]
    if write:
        blind_label.enqueue_pool(normalize_pool(pool, posted_ids=posted_ids), run_id=run_id)
    return decisions


def load_decisions(view: str = "live") -> list[dict]:
    return [
        row
        for row in blind_label._view(blind_label.load_assignments(), view)
        if row.get("record") == DECISION_RECORD
    ]


def load_outcomes(view: str = "live") -> list[dict]:
    return [
        row
        for row in blind_label._view(blind_label.load_assignments(), view)
        if row.get("record") == OUTCOME_RECORD
    ]


def collect_training_rows() -> list[dict]:
    items = {
        str(row.get("id")): row
        for row in blind_label._view(blind_label.load_items(), "live")
        if row.get("id")
    }
    candidates = {
        str(row.get("id")): row
        for row in blind_label._view(blind_label.load_candidates(), "live")
        if row.get("id")
    }
    decisions = {
        str(row.get("chosen_id") or row.get("id")): row
        for row in load_decisions(view="live")
        if row.get("chosen_id") or row.get("id")
    }
    labels = {}
    for row in blind_label._view(blind_label.load_labels(), "live"):
        item_id = str(row.get("id") or "")
        if item_id:
            labels[item_id] = row
    rows = []
    for item_id, label in labels.items():
        if label.get("quality_success") is None:
            continue
        item = items.get(item_id) or candidates.get(item_id) or {}
        decision = decisions.get(item_id, {})
        if blind_label.is_sim(label) or blind_label.is_sim(item) or blind_label.is_sim(decision):
            continue
        source = {**item, **decision}
        rows.append(
            {
                "id": item_id,
                "quality_success": bool(label.get("quality_success")),
                "features": decision.get("features") or extract_features(source or {"id": item_id}),
                "decided_at": decision.get("decided_at"),
                "posted_at": item.get("posted_at"),
                "labeled_at": label.get("labeled_at"),
            }
        )
    return rows


def train(*, cutoff: datetime, min_n: int = MIN_TRAIN_N) -> dict:
    rows = collect_training_rows()
    if not can_train(n_human_labeled=len(rows)):
        model = {
            "model_version": PRIOR_VERSION,
            "n": len(rows),
            "cutoff": cutoff.isoformat(),
            "reason": "cold_start",
            "train_label": "human_blind_label",
            "success_definition": SUCCESS_DEFINITION,
        }
        save_model(model)
        return model
    model = fit_model(rows, cutoff, min_n=min_n)
    save_model(model)
    return model


def _load_pool_arg(args) -> list[dict]:
    if args.json:
        payload = json.loads(args.json)
        if isinstance(payload, dict):
            return list(payload.get("items") or payload.get("pool") or [])
        return list(payload)
    if args.pool_jsonl:
        return blind_label._read_jsonl(Path(args.pool_jsonl))
    raise ValueError("provide --json or --pool-jsonl")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hn_ranker")
    sub = parser.add_subparsers(dest="cmd")
    p_sel = sub.add_parser("select", help="Choose up to two items from a pool")
    p_sel.add_argument("--run-id", required=True)
    p_sel.add_argument("--json", default="")
    p_sel.add_argument("--pool-jsonl", default="")
    p_sel.add_argument("--slots", type=int, default=POSTS_PER_RUN)
    p_tr = sub.add_parser("train", help="Fit logreg on human blind labels before cutoff")
    p_tr.add_argument("--cutoff", required=True)
    p_tr.add_argument("--min-n", type=int, default=MIN_TRAIN_N)
    p_arm = sub.add_parser("assign", help="Show the slot coin flip without writing")
    p_arm.add_argument("--run-id", required=True)
    p_arm.add_argument("--slot", type=int, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "select":
        pool = _load_pool_arg(args)
        print(json.dumps(select_run(pool, args.run_id, slots=args.slots), indent=2, default=str))
        return 0
    if args.cmd == "train":
        cutoff = blind_label._parse_ts(args.cutoff)
        if cutoff is None:
            raise SystemExit("invalid --cutoff")
        print(json.dumps(train(cutoff=cutoff, min_n=args.min_n), indent=2, default=str))
        return 0
    if args.cmd == "assign":
        print(json.dumps(assign_slot_arm(args.run_id, args.slot), indent=2))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
