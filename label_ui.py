"""Loopback UI for one rater to blind-label the HN pool.

Binds 127.0.0.1 only. Serves static HTML and writes via record_label().
"""

from __future__ import annotations

import argparse
import json
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import env_setup  # noqa: F401
import blind_label

RATER = "juust"
NOTES_MAX = 200
LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
HTML_PATH = Path(__file__).with_name("label_ui.html")
API_NEXT = "/api/next"
API_LABEL = "/api/label"


def bind_host(host: str) -> str:
    raw = (host or DEFAULT_HOST).strip() or DEFAULT_HOST
    if raw not in LOOPBACK:
        raise ValueError("label UI binds to loopback only")
    return "127.0.0.1" if raw == "localhost" else raw


def card_payload(item: dict) -> dict:
    view = blind_label.blind_view(item)
    payload = {key: view.get(key, "") for key in blind_label.VISIBLE_FIELDS}
    extra = sorted(set(payload) - set(blind_label.VISIBLE_FIELDS))
    leaked = sorted(set(payload) & set(blind_label.HIDDEN_FIELDS))
    if extra or leaked:
        raise ValueError(f"hidden fields in card: {extra or leaked}")
    return payload


def labeled_windows(now: Optional[datetime] = None) -> tuple[int, int]:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    today_start = datetime(current.year, current.month, current.day, tzinfo=timezone.utc)
    week_start = today_start - timedelta(days=today_start.weekday())
    today = 0
    week = 0
    for row in blind_label._latest_labels().values():
        ts = blind_label._parse_ts(row.get("labeled_at"))
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts >= today_start:
            today += 1
        if ts >= week_start:
            week += 1
    return today, week


def counts_payload() -> dict:
    report = blind_label.report()
    today, week = labeled_windows()
    return {
        "remaining": report["remaining"],
        "labeled": report["labeled"],
        "labeled_today": today,
        "labeled_week": week,
    }


def next_payload() -> dict:
    pending = blind_label.unlabeled_queue()
    item = card_payload(pending[0]) if pending else None
    return {"item": item, "counts": counts_payload()}


def _require_bool(payload: dict, key: str) -> bool:
    if key not in payload or payload[key] is None or payload[key] == "":
        raise ValueError(f"{key} is required")
    return blind_label.parse_bool(payload[key])


def save_label(payload: dict) -> dict:
    item_id = str(payload.get("id") or "").strip()
    if not item_id:
        raise ValueError("id is required")
    relevant = _require_bool(payload, "relevant")
    duplicate = _require_bool(payload, "duplicate")
    needs_correction = _require_bool(payload, "needs_correction")
    notes = str(payload.get("notes") or "")
    if len(notes) > NOTES_MAX:
        raise ValueError(f"notes longer than {NOTES_MAX} characters")
    blind_label.record_label(
        item_id,
        relevant=relevant,
        duplicate=duplicate,
        needs_correction=needs_correction,
        notes=notes,
        rater=RATER,
    )
    return next_payload()


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, default=str).encode("utf-8")


class LabelUI:
    def __init__(self):
        self.server: ThreadingHTTPServer | None = None

    def start(self, host: str = DEFAULT_HOST, port: int = 0) -> str:
        bound = bind_host(host)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_GET(self):
                path = urlparse(self.path).path
                if path in {"/", "/label_ui.html"}:
                    return self._send_html()
                if path == API_NEXT:
                    return self._send_next()
                return self._send_json(404, {"error": "not found"})

            def do_POST(self):
                path = urlparse(self.path).path
                if path != API_LABEL:
                    return self._send_json(404, {"error": "not found"})
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    body = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    return self._send_json(400, {"error": "invalid json"})
                if not isinstance(body, dict):
                    return self._send_json(400, {"error": "invalid json"})
                try:
                    payload = save_label(body)
                except ValueError as exc:
                    return self._send_json(400, {"error": str(exc)})
                except OSError as exc:
                    return self._send_json(500, {"error": f"store unavailable: {exc}"})
                return self._send_json(200, payload)

            def _send_next(self):
                try:
                    payload = next_payload()
                except OSError as exc:
                    return self._send_json(500, {"error": f"store unavailable: {exc}"})
                except ValueError as exc:
                    return self._send_json(500, {"error": str(exc)})
                return self._send_json(200, payload)

            def _send_html(self):
                try:
                    data = HTML_PATH.read_bytes()
                except OSError as exc:
                    return self._send_json(500, {"error": f"ui unavailable: {exc}"})
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'none'; script-src 'self' 'unsafe-inline'; "
                    "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
                    "form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
                )
                self.end_headers()
                self.wfile.write(data)

            def _send_json(self, code: int, payload: dict):
                data = _json_bytes(payload)
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer((bound, port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://{bound}:{self.server.server_address[1]}"

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="label_ui")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    ui = LabelUI()
    url = ui.start(host=args.host, port=args.port)
    print(url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        ui.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
