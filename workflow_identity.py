"""Stable identities and normalized operational context."""

from __future__ import annotations

import hashlib
import os
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_QUERY_KEYS = frozenset(
    {
        "access_token", "api_key", "auth", "fbclid", "gclid", "key", "secret",
        "session", "token", "utm_campaign", "utm_content", "utm_medium",
        "utm_source", "utm_term",
    }
)


def canonical_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return raw
    query = [
        (key, val)
        for key, val in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in TRACKING_QUERY_KEYS
    ]
    path = (parts.path or "/").rstrip("/") or "/"
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), "")
    )


def article_id(url: Any = "", *, hn_id: Any = None, title: Any = "") -> str:
    """Return a source-independent article identity, preferring canonical URL."""
    canonical = canonical_url(url)
    if canonical:
        basis = "url:" + canonical
    elif hn_id is not None and str(hn_id).strip():
        basis = "hackernews:" + str(hn_id).strip()
    else:
        basis = "title:" + " ".join(str(title or "").strip().lower().split())
    if basis == "title:":
        raise ValueError("article identity needs url, hn_id, or title")
    return "article-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:20]


def operation_context(
    *,
    work_unit: str = "",
    trigger: str = "",
    execution_mode: str = "",
    parent_run_id: str = "",
    skills_involved: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Normalize optional context fields without inventing missing business context."""
    values = {
        "work_unit": work_unit or os.environ.get("AGENT_LEARNING_WORK_UNIT", ""),
        "trigger": trigger or os.environ.get("AGENT_LEARNING_TRIGGER", ""),
        "execution_mode": execution_mode or os.environ.get("AGENT_LEARNING_EXECUTION_MODE", ""),
        "parent_run_id": parent_run_id or os.environ.get("AGENT_LEARNING_PARENT_RUN_ID", ""),
        "skills_involved": sorted({str(x).strip() for x in skills_involved if str(x).strip()}),
    }
    return {key: value for key, value in values.items() if value}
