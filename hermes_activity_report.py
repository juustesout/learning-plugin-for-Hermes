#!/usr/bin/env python3
"""Read-only Hermes SQLite activity reporting CLI."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DB_PATH = r"C:\Users\juust\.hermes\profiles\operator\state.db"


@dataclass
class DbSchema:
    tables: dict[str, set[str]]
    indexes: dict[str, list[dict[str, Any]]]

    def has_table(self, table: str) -> bool:
        return table in self.tables

    def has_columns(self, table: str, *cols: str) -> bool:
        existing = self.tables.get(table, set())
        return all(col in existing for col in cols)


def resolve_db_path(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("HERMES_STATE_DB", "").strip()
    if env:
        return env
    return DEFAULT_DB_PATH


def connect_readonly(db_path: str) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def load_schema(conn: sqlite3.Connection) -> DbSchema:
    tables: dict[str, set[str]] = {}
    indexes: dict[str, list[dict[str, Any]]] = {}
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    for row in rows:
        table = row["name"]
        col_rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        tables[table] = {c["name"] for c in col_rows}
        idx_rows = conn.execute(f"PRAGMA index_list({table})").fetchall()
        idx_info: list[dict[str, Any]] = []
        for idx in idx_rows:
            name = idx["name"]
            cols = [c["name"] for c in conn.execute(f"PRAGMA index_info({name})").fetchall()]
            idx_info.append(
                {
                    "name": name,
                    "unique": bool(idx["unique"]),
                    "origin": idx["origin"],
                    "columns": cols,
                }
            )
        indexes[table] = idx_info
    return DbSchema(tables=tables, indexes=indexes)


def date_expr(column: str, tz: str) -> str:
    if tz == "local":
        return f"date(datetime({column}, 'unixepoch', 'localtime'))"
    return f"date(datetime({column}, 'unixepoch'))"


def where_date_range(date_sql: str, start: str | None, end: str | None) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if start:
        clauses.append(f"{date_sql} >= ?")
        params.append(start)
    if end:
        clauses.append(f"{date_sql} <= ?")
        params.append(end)
    if not clauses:
        return "", params
    return "WHERE " + " AND ".join(clauses), params


def query_rows(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]


def tool_failure_expr() -> str:
    return """
CASE
  WHEN role != 'tool' THEN 0
  WHEN json_valid(content) THEN
    CASE
      WHEN lower(COALESCE(json_extract(content, '$.status'), '')) IN ('error', 'failed', 'fail') THEN 1
      WHEN CAST(COALESCE(json_extract(content, '$.exit_code'), 0) AS INTEGER) != 0 THEN 1
      WHEN trim(lower(COALESCE(CAST(json_extract(content, '$.error') AS TEXT), ''))) NOT IN ('', 'null') THEN 1
      ELSE 0
    END
  WHEN lower(COALESCE(content, '')) LIKE '%"error"%' THEN 1
  ELSE 0
END
""".strip()


def build_queries(schema: DbSchema, tz: str, start: str | None, end: str | None) -> dict[str, tuple[str, list[Any]]]:
    queries: dict[str, tuple[str, list[Any]]] = {}

    sessions_date = date_expr("s.started_at", tz)
    messages_date = date_expr("m.timestamp", tz)
    wf_sessions, p_sessions = where_date_range(sessions_date, start, end)
    wf_messages, p_messages = where_date_range(messages_date, start, end)

    has_sessions = schema.has_table("sessions")
    has_messages = schema.has_table("messages")

    if has_sessions and has_messages:
        daily_sql = f"""
WITH
session_daily AS (
  SELECT
    {sessions_date} AS day,
    COUNT(*) AS sessions,
    SUM(COALESCE(s.input_tokens, 0)) AS input_tokens,
    SUM(COALESCE(s.output_tokens, 0)) AS output_tokens,
    SUM(COALESCE(s.cache_read_tokens, 0)) AS cache_read_tokens,
    SUM(COALESCE(s.cache_write_tokens, 0)) AS cache_write_tokens,
    SUM(COALESCE(s.reasoning_tokens, 0)) AS reasoning_tokens,
    SUM(
      COALESCE(s.input_tokens, 0) + COALESCE(s.output_tokens, 0) +
      COALESCE(s.cache_read_tokens, 0) + COALESCE(s.cache_write_tokens, 0) +
      COALESCE(s.reasoning_tokens, 0)
    ) AS total_tokens
  FROM sessions s
  {wf_sessions}
  GROUP BY 1
),
message_daily AS (
  SELECT
    {messages_date} AS day,
    SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) AS user_messages,
    SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) AS assistant_messages,
    SUM(CASE WHEN m.role = 'tool' THEN 1 ELSE 0 END) AS tool_calls,
    SUM({tool_failure_expr()}) AS tool_errors
  FROM messages m
  {wf_messages}
  GROUP BY 1
),
all_days AS (
  SELECT day FROM session_daily
  UNION
  SELECT day FROM message_daily
)
SELECT
  d.day,
  COALESCE(sd.sessions, 0) AS sessions,
  COALESCE(md.user_messages, 0) AS user_messages,
  COALESCE(md.assistant_messages, 0) AS assistant_messages,
  COALESCE(md.tool_calls, 0) AS tool_calls,
  COALESCE(md.tool_errors, 0) AS errors,
  COALESCE(sd.total_tokens, 0) AS total_tokens,
  COALESCE(sd.input_tokens, 0) AS input_tokens,
  COALESCE(sd.output_tokens, 0) AS output_tokens
FROM all_days d
LEFT JOIN session_daily sd ON sd.day = d.day
LEFT JOIN message_daily md ON md.day = d.day
ORDER BY d.day DESC
"""
        queries["daily"] = (daily_sql, [*p_sessions, *p_messages])

    if has_sessions:
        model_sql = f"""
SELECT
  {sessions_date} AS day,
  COALESCE(s.billing_provider, 'unknown') AS provider,
  COALESCE(s.model, 'unknown') AS model,
  COUNT(*) AS sessions,
  SUM(COALESCE(s.api_call_count, 0)) AS requests,
  SUM(COALESCE(s.input_tokens, 0)) AS input_tokens,
  SUM(COALESCE(s.output_tokens, 0)) AS output_tokens,
  SUM(
    COALESCE(s.input_tokens, 0) + COALESCE(s.output_tokens, 0) +
    COALESCE(s.cache_read_tokens, 0) + COALESCE(s.cache_write_tokens, 0) +
    COALESCE(s.reasoning_tokens, 0)
  ) AS total_tokens,
  SUM(COALESCE(s.estimated_cost_usd, 0)) AS estimated_cost_usd
FROM sessions s
{wf_sessions}
GROUP BY 1, 2, 3
ORDER BY day DESC, requests DESC
"""
        queries["models"] = (model_sql, p_sessions)

    if has_messages:
        tool_sql = f"""
WITH tool_rows AS (
  SELECT
    {messages_date} AS day,
    COALESCE(m.tool_name, 'unknown') AS tool_name,
    CASE
      WHEN m.tool_name LIKE 'mcp_%' THEN
        CASE
          WHEN instr(substr(m.tool_name, 5), '_') > 0 THEN substr(m.tool_name, 5, instr(substr(m.tool_name, 5), '_') - 1)
          ELSE substr(m.tool_name, 5)
        END
      ELSE NULL
    END AS mcp_server,
    {tool_failure_expr()} AS failed
  FROM messages m
  {wf_messages}
)
SELECT
  day,
  tool_name,
  COALESCE(mcp_server, '-') AS mcp_server,
  COUNT(*) AS invocations,
  SUM(CASE WHEN failed = 0 THEN 1 ELSE 0 END) AS success_count,
  SUM(CASE WHEN failed = 1 THEN 1 ELSE 0 END) AS failure_count
FROM tool_rows
WHERE tool_name IS NOT NULL AND tool_name != 'unknown'
GROUP BY 1, 2, 3
ORDER BY day DESC, invocations DESC
"""
        queries["tools"] = (tool_sql, p_messages)

    if has_sessions and has_messages:
        session_sql = f"""
WITH message_rollup AS (
  SELECT
    m.session_id,
    MAX(m.timestamp) AS last_message_at,
    SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) AS user_messages,
    SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) AS assistant_messages,
    SUM(CASE WHEN m.role = 'tool' THEN 1 ELSE 0 END) AS tool_calls,
    SUM({tool_failure_expr()}) AS tool_errors
  FROM messages m
  GROUP BY 1
)
SELECT
  s.id AS session_id,
  s.source,
  s.started_at,
  COALESCE(s.ended_at, mr.last_message_at, s.started_at) AS last_activity_at,
  ROUND(COALESCE(s.ended_at, mr.last_message_at, s.started_at) - s.started_at, 2) AS duration_seconds,
  COALESCE(s.model, 'unknown') AS model,
  COALESCE(s.billing_provider, 'unknown') AS provider,
  COALESCE(mr.user_messages, 0) AS user_messages,
  COALESCE(mr.assistant_messages, 0) AS assistant_messages,
  COALESCE(mr.tool_calls, 0) AS tool_calls,
  COALESCE(mr.tool_errors, 0) AS tool_errors,
  COALESCE(s.api_call_count, 0) AS api_calls,
  COALESCE(s.input_tokens, 0) AS input_tokens,
  COALESCE(s.output_tokens, 0) AS output_tokens,
  (
    COALESCE(s.input_tokens, 0) + COALESCE(s.output_tokens, 0) +
    COALESCE(s.cache_read_tokens, 0) + COALESCE(s.cache_write_tokens, 0) +
    COALESCE(s.reasoning_tokens, 0)
  ) AS total_tokens,
  COALESCE(s.estimated_cost_usd, 0) AS estimated_cost_usd,
  COALESCE(s.end_reason, '') AS end_reason
FROM sessions s
LEFT JOIN message_rollup mr ON mr.session_id = s.id
{wf_sessions}
ORDER BY s.started_at DESC
"""
        queries["sessions"] = (session_sql, p_sessions)

        error_sql = f"""
WITH tool_errors AS (
  SELECT
    {messages_date} AS day,
    'tool_call_failed' AS error_type,
    COALESCE(m.tool_name, 'unknown') AS key_name,
    COUNT(*) AS count
  FROM messages m
  {wf_messages}
    AND m.role = 'tool'
    AND ({tool_failure_expr()}) = 1
  GROUP BY 1, 2, 3
),
session_errors AS (
  SELECT
    {sessions_date} AS day,
    'session_end_reason' AS error_type,
    COALESCE(NULLIF(trim(s.end_reason), ''), 'unknown') AS key_name,
    COUNT(*) AS count
  FROM sessions s
  {wf_sessions}
    AND lower(COALESCE(s.end_reason, '')) NOT IN ('', 'completed', 'done', 'success')
  GROUP BY 1, 2, 3
),
empty_assistant AS (
  SELECT
    {messages_date} AS day,
    'assistant_empty_message' AS error_type,
    'empty_content' AS key_name,
    COUNT(*) AS count
  FROM messages m
  {wf_messages}
    AND m.role = 'assistant'
    AND trim(COALESCE(m.content, '')) = ''
    AND trim(COALESCE(m.tool_calls, '')) = ''
  GROUP BY 1, 2, 3
)
SELECT * FROM tool_errors
UNION ALL
SELECT * FROM session_errors
UNION ALL
SELECT * FROM empty_assistant
ORDER BY day DESC, count DESC
"""
        queries["errors"] = (error_sql, [*p_messages, *p_sessions, *p_messages])

    return queries


def render_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no rows)"
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    body = [" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols) for r in rows]
    return "\n".join([header, sep, *body])


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def recon_summary(conn: sqlite3.Connection, schema: DbSchema, tz: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "tables": sorted(schema.tables.keys()),
        "table_columns": {k: sorted(v) for k, v in schema.tables.items()},
        "indexes": schema.indexes,
        "coverage": {},
    }
    if schema.has_table("sessions"):
        sessions = conn.execute(
            """
SELECT
  COUNT(*) AS sessions,
  MIN(started_at) AS min_started,
  MAX(started_at) AS max_started,
  MIN(ended_at) AS min_ended,
  MAX(ended_at) AS max_ended,
  SUM(COALESCE(api_call_count, 0)) AS api_calls,
  SUM(COALESCE(message_count, 0)) AS message_count,
  SUM(COALESCE(tool_call_count, 0)) AS tool_call_count,
  SUM(COALESCE(input_tokens, 0)) AS input_tokens,
  SUM(COALESCE(output_tokens, 0)) AS output_tokens
FROM sessions
"""
        ).fetchone()
        summary["coverage"]["sessions"] = dict(sessions)
        summary["coverage"]["sources"] = query_rows(
            conn, "SELECT source, COUNT(*) AS sessions FROM sessions GROUP BY source ORDER BY sessions DESC"
        )
        summary["coverage"]["models"] = query_rows(
            conn, "SELECT COALESCE(model, 'unknown') AS model, COUNT(*) AS sessions FROM sessions GROUP BY 1 ORDER BY sessions DESC"
        )
    if schema.has_table("messages"):
        messages = conn.execute(
            """
SELECT
  COUNT(*) AS messages,
  MIN(timestamp) AS min_timestamp,
  MAX(timestamp) AS max_timestamp,
  SUM(CASE WHEN role='tool' THEN 1 ELSE 0 END) AS tool_messages,
  SUM(CASE WHEN role='assistant' THEN 1 ELSE 0 END) AS assistant_messages,
  SUM(CASE WHEN role='user' THEN 1 ELSE 0 END) AS user_messages
FROM messages
"""
        ).fetchone()
        summary["coverage"]["messages"] = dict(messages)
        summary["coverage"]["roles"] = query_rows(
            conn, "SELECT role, COUNT(*) AS count FROM messages GROUP BY role ORDER BY count DESC"
        )
        summary["coverage"]["tool_names"] = query_rows(
            conn,
            """
SELECT COALESCE(tool_name, 'unknown') AS tool_name, COUNT(*) AS invocations
FROM messages
WHERE role='tool'
GROUP BY 1
ORDER BY invocations DESC
LIMIT 30
""",
        )
    summary["timezone"] = tz
    return summary


def coerce_date(value: str | None) -> str | None:
    if value is None:
        return None
    datetime.strptime(value, "%Y-%m-%d")
    return value


def default_day_for_tz(tz: str) -> str:
    now = datetime.now() if tz == "local" else datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d")


def build_daylog_payload(
    conn: sqlite3.Connection,
    schema: DbSchema,
    *,
    day: str,
    tz: str,
    db_path: str,
    limit: int,
) -> dict[str, Any]:
    queries = build_queries(schema, tz, day, day)
    payload: dict[str, Any] = {
        "report_type": "hermes_daily_activity_log",
        "day": day,
        "timezone": tz,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db_path": db_path,
        "sections": {},
        "notes": {
            "read_only": True,
            "token_source": "sessions table aggregate columns",
            "tool_error_detection": "heuristic from tool JSON fields (status/exit_code/error)",
        },
    }
    section_order = ["daily", "models", "tools", "sessions", "errors"]
    for name in section_order:
        if name not in queries:
            payload["sections"][name] = {"available": False, "rows": []}
            continue
        sql, params = queries[name]
        rows = query_rows(conn, sql, params)
        if name in {"sessions", "errors"}:
            rows = rows[: max(limit, 1)]
        payload["sections"][name] = {"available": True, "rows": rows}
    return payload


def write_json(path: str, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Hermes SQLite activity reporter")
    parser.add_argument("--db", help="Path to Hermes state.db (default: HERMES_STATE_DB or profile default)")
    parser.add_argument("--tz", choices=["utc", "local"], default="local", help="Date bucketing timezone")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    parser.add_argument("--csv", help="Optional CSV output path for selected report")
    parser.add_argument("--json-out", help="Optional JSON output path (for --daylog or JSON export)")
    parser.add_argument("--show-sql", action="store_true", help="Print SQL for selected report(s)")
    parser.add_argument("--limit", type=int, default=50, help="Limit rows for sessions/errors reports")
    parser.add_argument(
        "--daylog",
        action="store_true",
        help="Write a single-day JSON snapshot with daily/models/tools/sessions/errors sections",
    )
    parser.add_argument("--day", help="Day for --daylog (YYYY-MM-DD). Defaults to current day in selected timezone")
    parser.add_argument("--daylog-dir", help="Directory for --daylog output JSON (default: ./reports)")
    parser.add_argument(
        "--report",
        choices=["recon", "daily", "models", "tools", "sessions", "errors", "all"],
        default="all",
    )
    args = parser.parse_args()

    db_path = resolve_db_path(args.db)
    start = coerce_date(args.start)
    end = coerce_date(args.end)

    with connect_readonly(db_path) as conn:
        schema = load_schema(conn)
        if args.daylog:
            day = coerce_date(args.day) if args.day else default_day_for_tz(args.tz)
            payload = build_daylog_payload(
                conn,
                schema,
                day=day,
                tz=args.tz,
                db_path=db_path,
                limit=args.limit,
            )
            target = args.json_out
            if not target:
                out_dir = args.daylog_dir or str(Path.cwd() / "reports")
                target = str(Path(out_dir) / f"hermes-daylog-{day}.json")
            write_json(target, payload)
            print(f"daylog written: {target}")
            if args.format == "json":
                print(json.dumps(payload, indent=2, default=str))
            else:
                daily_rows = payload["sections"].get("daily", {}).get("rows", [])
                print("\n## daily")
                print(render_table(daily_rows if isinstance(daily_rows, list) else []))
            return 0
        if args.report == "recon":
            summary = recon_summary(conn, schema, args.tz)
            print(json.dumps(summary, indent=2, default=str))
            return 0

        queries = build_queries(schema, args.tz, start, end)
        selected = (
            ["daily", "models", "tools", "sessions", "errors"]
            if args.report == "all"
            else [args.report]
        )
        for name in selected:
            if name not in queries:
                print(f"\n## {name}\n(unavailable: required tables/columns not present)")
                continue
            sql, params = queries[name]
            rows = query_rows(conn, sql, params)
            if name in {"sessions", "errors"}:
                rows = rows[: max(args.limit, 1)]
            if args.show_sql:
                print(f"\n-- SQL: {name}\n{sql.strip()}\n-- params: {params}")
            print(f"\n## {name}")
            if args.format == "json":
                print(json.dumps(rows, indent=2, default=str))
            else:
                print(render_table(rows))
            if args.csv:
                base = Path(args.csv)
                target = base if len(selected) == 1 else base.with_name(f"{base.stem}_{name}{base.suffix or '.csv'}")
                write_csv(str(target), rows)
                print(f"(csv written: {target})")
            if args.json_out:
                base = Path(args.json_out)
                target = (
                    base
                    if len(selected) == 1
                    else base.with_name(f"{base.stem}_{name}{base.suffix or '.json'}")
                )
                write_json(str(target), {"report": name, "rows": rows})
                print(f"(json written: {target})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
