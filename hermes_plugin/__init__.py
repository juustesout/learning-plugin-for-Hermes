"""Thin Hermes plugin: observer hooks -> opslog/shadow. Kernel stays Hermes-free."""

from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional


def _kernel_roots() -> list[Path]:
    here = Path(__file__).resolve().parent
    roots = [here.parent, here]
    env = os.environ.get("AGENT_LEARNING_REPO", "").strip()
    if env:
        roots.insert(0, Path(env))
    return roots


for _root in _kernel_roots():
    if (_root / "opslog.py").is_file():
        _root_s = str(_root)
        if _root_s not in sys.path:
            sys.path.insert(0, _root_s)
        break

try:
    import opslog
except Exception:  # pragma: no cover - fail-open when kernel is absent
    opslog = None

try:
    import shadow_policy
except Exception:  # pragma: no cover - fail-open when kernel is absent
    shadow_policy = None

PLUGIN_NAME = "agent-learning"
DEFAULT_TASK_ID = "hermes-tool"
HOOKS_REGISTERED: dict[str, bool] = {
    "pre_tool_call": False,
    "post_tool_call": False,
    "post_api_request": False,
    "pre_llm_call": False,
    "llm_execution_middleware": False,
}
MIDDLEWARE_KILL_ENV = "AGENT_LEARNING_MIDDLEWARE"
_SUMMARY_MAX = 500
_TARGET_MAX = 300
_TARGET_KEYS = ("url", "uri", "path", "query", "target", "href")
_GEN_CACHE_MAX = 4096
_TOOL_TASK = {
    "web_search": "web-research",
    "web_extract": "choose-scrape-strategy",
    "browser_navigate": "choose-scrape-strategy",
    "mcp_webclaw_webclaw_scrape": "choose-scrape-strategy",
}
_GENERATION_IDS_BY_API_REQUEST: "OrderedDict[str, list[str]]" = OrderedDict()


def logging_enabled() -> bool:
    raw = os.environ.get("AGENT_LEARNING_OPSLOG_ENABLED", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def middleware_enabled() -> bool:
    raw = os.environ.get(MIDDLEWARE_KILL_ENV, "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def log_session_ids() -> bool:
    raw = os.environ.get("AGENT_LEARNING_OPSLOG_SESSION_IDS", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _load_json_map(raw: str) -> dict[str, str]:
    if not raw.strip():
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {str(key): str(value) for key, value in loaded.items() if str(key).strip()}


def load_tool_task_map() -> dict[str, str]:
    mapping = dict(_TOOL_TASK)
    mapping.update(_load_json_map(os.environ.get("AGENT_LEARNING_TOOL_TASK_MAP", "")))
    path = os.environ.get("AGENT_LEARNING_TOOL_TASK_MAP_FILE", "").strip()
    if path:
        try:
            mapping.update(_load_json_map(Path(path).read_text(encoding="utf-8")))
        except OSError:
            pass
    return mapping


def _truncate(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def extract_target(args: Any) -> str:
    if not isinstance(args, dict):
        return ""
    for key in _TARGET_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            text = _truncate(value.strip(), _TARGET_MAX)
            if opslog is not None:
                return opslog.redact_target(text)
            return text
    return ""


def task_id_for_tool(tool_name: str, mapping: Optional[dict[str, str]] = None) -> str:
    table = mapping if mapping is not None else load_tool_task_map()
    return table.get(tool_name, DEFAULT_TASK_ID)


def summarize_tool_call(
    *,
    tool_name: str,
    status: str,
    error_message: Any = None,
) -> str:
    err = _truncate(error_message, _SUMMARY_MAX) if error_message else ""
    if err:
        text = _truncate(f"{tool_name}:{status} {err}", _SUMMARY_MAX)
    else:
        text = _truncate(f"{tool_name}:{status}", _SUMMARY_MAX)
    if opslog is not None:
        return opslog.redact_text(text)
    return text


def _load_features():
    """Return (module or None, "relative" | "absolute" | None, [import errors]).

    Relative first (works when the plugin loads as a subpackage), absolute as
    fallback. Every failed attempt is recorded so nothing fails silently.
    """
    errors: list[str] = []
    try:
        from .. import features as module

        return module, "relative", errors
    except Exception as exc:
        errors.append(f"relative: {type(exc).__name__}: {exc}")
    try:
        import features as module

        return module, "absolute", errors
    except Exception as exc:
        errors.append(f"absolute: {type(exc).__name__}: {exc}")
    return None, None, errors


def _health_error_text(message: str) -> None:
    try:
        if opslog is not None:
            opslog.record_health(ok=False, error=message)
    except Exception:
        return


def build_operation(
    *,
    tool_name: str = "",
    args: Any = None,
    status: str = "",
    duration_ms: Any = None,
    error_message: Any = None,
    session_id: str = "",
    turn_id: str = "",
    tool_call_id: str = "",
    api_request_id: str = "",
    **_: Any,
) -> Optional[dict]:
    name = str(tool_name or "").strip()
    if not name:
        return None
    outcome = str(status or "ok").strip().lower() or "ok"
    latency = None
    if duration_ms is not None:
        try:
            latency = int(duration_ms)
        except (TypeError, ValueError):
            latency = None
    metadata = {
        "source": "hermes",
        "hermes_status": outcome,
        "tool_ok": outcome == "ok",
        "task_success": False,
    }
    if log_session_ids():
        metadata["session_id"] = session_id or ""
        metadata["turn_id"] = turn_id or ""
        metadata["tool_call_id"] = tool_call_id or ""
        metadata["api_request_id"] = api_request_id or ""
    generation_ids = _consume_generation_ids(str(api_request_id or "").strip())
    if generation_ids:
        metadata["generation_ids"] = generation_ids
    raw_target = ""
    if isinstance(args, dict):
        for key in _TARGET_KEYS:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                raw_target = value.strip()
                break
    target = extract_target(args)
    context_features = None
    feature_register, import_source, import_errors = _load_features()
    metadata["import_source"] = import_source or "none"
    if feature_register is None:
        _health_error_text("features import failed: " + "; ".join(import_errors))
    else:
        try:
            context_features = feature_register.pre_call_features(
                raw_target, action_id=name
            )
        except Exception as exc:
            context_features = None
            _health_error_text(f"features: {type(exc).__name__}: {exc}")
    return {
        "task_id": task_id_for_tool(name),
        "action_id": name,
        "target": target,
        "success": outcome == "ok",
        "duration_ms": latency,
        "summary": summarize_tool_call(
            tool_name=name, status=outcome, error_message=error_message
        ),
        "action_type": name,
        "context_features": context_features,
        "metadata": metadata,
        "generation_ids": generation_ids or None,
    }


def _health_error(exc: BaseException) -> None:
    try:
        if opslog is not None:
            opslog.record_health(ok=False, error=str(exc))
    except Exception:
        return


def _normalize_generation_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) > 200:
        return ""
    return text


def _collect_generation_ids(value: Any, out: set[str]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        gid = _normalize_generation_id(value)
        if gid:
            out.add(gid)
        return
    if isinstance(value, dict):
        # Direct candidates
        for key in (
            "generation_id",
            "openrouter_generation_id",
            "x-openrouter-generation-id",
            "x-openrouter-id",
            "id",
        ):
            if key in value:
                _collect_generation_ids(value.get(key), out)
        # Nested headers variants
        for headers_key in (
            "response_headers",
            "headers",
            "http_headers",
            "raw_headers",
        ):
            headers = value.get(headers_key)
            if isinstance(headers, dict):
                for hk, hv in headers.items():
                    lowered = str(hk).strip().lower()
                    if lowered in {"x-openrouter-generation-id", "x-openrouter-id"}:
                        _collect_generation_ids(hv, out)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_generation_ids(item, out)
        return
    # SDK response objects: inspect known fields only.
    for attr in (
        "generation_id",
        "openrouter_generation_id",
        "id",
        "response_headers",
        "headers",
        "http_headers",
        "raw_headers",
    ):
        if hasattr(value, attr):
            try:
                _collect_generation_ids(getattr(value, attr), out)
            except Exception:
                continue


def _remember_generation_ids(api_request_id: str, generation_ids: list[str]) -> None:
    if not api_request_id or not generation_ids:
        return
    existing = _GENERATION_IDS_BY_API_REQUEST.get(api_request_id, [])
    merged = list(dict.fromkeys([*existing, *generation_ids]))
    _GENERATION_IDS_BY_API_REQUEST[api_request_id] = merged
    _GENERATION_IDS_BY_API_REQUEST.move_to_end(api_request_id)
    while len(_GENERATION_IDS_BY_API_REQUEST) > _GEN_CACHE_MAX:
        _GENERATION_IDS_BY_API_REQUEST.popitem(last=False)


def _consume_generation_ids(api_request_id: str) -> list[str]:
    if not api_request_id:
        return []
    return list(_GENERATION_IDS_BY_API_REQUEST.pop(api_request_id, []))


def _extract_generation_ids_from_post_api_request(kwargs: dict[str, Any]) -> list[str]:
    found: set[str] = set()
    # Known top-level payload slots on post_api_request hooks.
    for key in (
        "generation_id",
        "response_id",
        "response",
        "response_headers",
        "raw_response",
        "api_response",
    ):
        if key in kwargs:
            _collect_generation_ids(kwargs.get(key), found)
    # OpenRouter may include generation headers in a generic headers map.
    headers = kwargs.get("headers")
    if isinstance(headers, dict):
        for hk, hv in headers.items():
            lowered = str(hk).strip().lower()
            if lowered in {"x-openrouter-generation-id", "x-openrouter-id"}:
                _collect_generation_ids(hv, found)
    return sorted(found)


def _extract_generation_ids_from_llm_response(response: Any) -> list[str]:
    found: set[str] = set()
    _collect_generation_ids(response, found)
    return sorted(found)


def _decision_join_key(kwargs: dict[str, Any], tool_name: str = "") -> str:
    try:
        import decision_record
    except Exception:
        return ""
    return decision_record.join_key(
        session_id=str(kwargs.get("session_id") or ""),
        conversation_id=str(kwargs.get("conversation_id") or ""),
        tool_call_id=str(kwargs.get("tool_call_id") or ""),
        turn_id=str(kwargs.get("turn_id") or ""),
        tool_name=tool_name,
    )


def _open_decision_record(kwargs: dict[str, Any], *, tool_name: str, shadow_row: Any = None) -> None:
    try:
        import decision_record
    except Exception:
        return
    features = None
    advised = ""
    task = task_id_for_tool(tool_name)
    if isinstance(shadow_row, dict):
        features = shadow_row.get("context_features")
        advised = str(shadow_row.get("advised_tool") or "")
        task = str(shadow_row.get("task_id") or task)
    if features is None:
        feature_register, _source, _errors = _load_features()
        if feature_register is not None:
            raw_target = ""
            args = kwargs.get("args")
            if isinstance(args, dict):
                for key in _TARGET_KEYS:
                    value = args.get(key)
                    if isinstance(value, str) and value.strip():
                        raw_target = value.strip()
                        break
            try:
                features = feature_register.pre_call_features(raw_target, action_id=tool_name)
            except Exception:
                features = None
    decision_record.open_decision(
        join_key_value=_decision_join_key(kwargs, tool_name),
        requested_tool=tool_name,
        advised_tool=advised,
        context_features=features,
        task_id=task,
    )


def _attach_decision_technical(kwargs: dict[str, Any], *, tool_name: str) -> None:
    try:
        import decision_record
    except Exception:
        return
    decision_record.attach_technical(
        join_key_value=_decision_join_key(kwargs, tool_name),
        status=str(kwargs.get("status") or ""),
        chosen_tool=tool_name,
    )


def on_pre_tool_call(**kwargs) -> None:
    try:
        name = str(kwargs.get("tool_name") or "").strip()
        shadow_row = None
        if shadow_policy is not None:
            shadow_row = shadow_policy.record_pre(
                task_id=task_id_for_tool(name),
                tool_name=name,
                args=kwargs.get("args"),
                session_id=str(kwargs.get("session_id") or ""),
                conversation_id=str(kwargs.get("conversation_id") or ""),
                turn_id=str(kwargs.get("turn_id") or ""),
                tool_call_id=str(kwargs.get("tool_call_id") or ""),
            )
        if name:
            _open_decision_record(kwargs, tool_name=name, shadow_row=shadow_row)
    except Exception as ext:
        _health_error(ext)
    return None


def on_post_api_request(**kwargs) -> None:
    try:
        api_request_id = str(kwargs.get("api_request_id") or "").strip()
        generation_ids = _extract_generation_ids_from_post_api_request(kwargs)
        if api_request_id and generation_ids:
            _remember_generation_ids(api_request_id, generation_ids)
    except Exception as exc:
        _health_error(exc)
    return None


def on_pre_llm_call(*args, **kwargs):
    messages = kwargs.get("messages")
    if messages is None and args:
        messages = args[0]
    original = messages
    try:
        import advice_hook
        import decision_stats

        if not advice_hook.enabled():
            return original
        try:
            import promotion_gate
            import week_report

            report = week_report.build()
            gate = report.get("gate") or promotion_gate.evaluate({"n": 0})
        except Exception:
            gate = {"open": False, "reason": "gate_unavailable"}
        advice = None
        if gate.get("open"):
            try:
                advice = decision_stats.build_decision_advice(
                    task_id="choose-scrape-strategy",
                    options=["use_webclaw_scrape", "use_jina_reader", "use_browser"],
                )
            except Exception:
                advice = None
        return advice_hook.apply_advice(
            original,
            gate=gate,
            advice=advice,
            session_id=str(kwargs.get("session_id") or ""),
            turn_id=str(kwargs.get("turn_id") or ""),
            api_request_id=str(kwargs.get("api_request_id") or ""),
        )
    except Exception as exc:
        _health_error(exc)
        return original


def on_llm_execution_middleware(request, next_call, **kwargs):
    try:
        if not middleware_enabled():
            return next_call(request)
    except Exception:
        return next_call(request)
    try:
        response = next_call(request)
    except Exception:
        raise
    try:
        api_request_id = str(kwargs.get("api_request_id") or "").strip()
        generation_ids = _extract_generation_ids_from_llm_response(response)
        if api_request_id and generation_ids:
            _remember_generation_ids(api_request_id, generation_ids)
    except Exception as exc:
        _health_error(exc)
    return response


def on_post_tool_call(**kwargs) -> None:
    try:
        if logging_enabled() and opslog is not None:
            payload = build_operation(**kwargs)
            if payload is not None:
                opslog.log_operation(**payload)
    except OSError:
        pass
    except Exception as exc:
        _health_error(exc)
    try:
        if shadow_policy is not None:
            shadow_policy.record_post(
                tool_name=str(kwargs.get("tool_name") or ""),
                session_id=str(kwargs.get("session_id") or ""),
                conversation_id=str(kwargs.get("conversation_id") or ""),
                turn_id=str(kwargs.get("turn_id") or ""),
                tool_call_id=str(kwargs.get("tool_call_id") or ""),
                status=str(kwargs.get("status") or ""),
                duration_ms=kwargs.get("duration_ms"),
                result=kwargs.get("result"),
            )
    except Exception as exc:
        _health_error(exc)
    try:
        name = str(kwargs.get("tool_name") or "").strip()
        if name:
            _attach_decision_technical(kwargs, tool_name=name)
    except Exception as exc:
        _health_error(exc)
    return None


def _handle_cli(args) -> None:
    cmd = getattr(args, "al_command", None)
    if cmd == "status":
        import plugin_status

        print(plugin_status.format_status(), end="")
        return
    if cmd == "selftest":
        import plugin_status

        print(plugin_status.format_selftest(), end="")
        return
    if cmd == "record-outcome":
        import decision_record

        result = decision_record.record_outcome(
            decision_id=str(getattr(args, "decision_id", "") or ""),
            level=getattr(args, "level", None),
            status=getattr(args, "status", None),
        )
        print(json.dumps(result, indent=2, default=str))
        return
    if cmd == "report":
        import week_report

        print(week_report.format_report(), end="")
        return
    if opslog is None:
        print("opslog unavailable", file=sys.stderr)
        return
    if cmd == "count":
        print(opslog.count_operations())
        return
    if cmd == "tail":
        n = getattr(args, "n", 10)
        for op in opslog.tail(n):
            print(json.dumps(op, indent=2)[:500])
            print("---")
        return
    if cmd == "stats":
        print(json.dumps(opslog.stats(), indent=2))
        return
    if cmd == "health":
        print(json.dumps(opslog.health(), indent=2))
        return
    try:
        import pool_v1
    except Exception:
        pool_v1 = None
    if pool_v1 is not None:
        if cmd == "pool" and getattr(args, "pool_cmd", None) == "import":
            print(json.dumps(pool_v1.import_pool(args.path), indent=2, default=str))
            return
        if cmd == "select":
            print(json.dumps(pool_v1.select_for_run(args.run_id), indent=2, default=str))
            return
        if cmd == "baseline":
            print(json.dumps(pool_v1.baseline_from_pool(args.pool), indent=2, default=str))
            return
        if cmd == "record-post":
            raw = str(getattr(args, "fallback", "false")).strip().lower()
            fallback = raw in {"1", "true", "yes", "y", "on"}
            print(
                json.dumps(
                    pool_v1.record_post(
                        run_id=args.run_id,
                        item_id=args.id,
                        http_status=args.http_status,
                        fallback=fallback,
                    ),
                    indent=2,
                    default=str,
                )
            )
            return
    print(
        "Usage: hermes agent-learning "
        "<stats|count|tail|health|status|selftest|record-outcome|report|pool|select|baseline|record-post>"
    )


def _setup_argparse(subparser) -> None:
    subs = subparser.add_subparsers(dest="al_command")
    subs.add_parser("stats", help="Show opslog stats")
    subs.add_parser("count", help="Count logged operations")
    subs.add_parser("health", help="Show opslog write/error health")
    subs.add_parser("status", help="Show hooks, features, opslog, gate")
    subs.add_parser("selftest", help="Run hooks against a temporary store")
    p_out = subs.add_parser("record-outcome", help="Record a later outcome on a decision_id")
    p_out.add_argument("--decision-id", required=True)
    p_out.add_argument("--level", required=True)
    p_out.add_argument("--status", required=True)
    subs.add_parser("report", help="Show weekly ITT / gate report")
    p_tail = subs.add_parser("tail", help="Show recent operations")
    p_tail.add_argument("-n", type=int, default=10)
    p_pool = subs.add_parser("pool", help="Import a pool.v1 candidate file")
    pool_sub = p_pool.add_subparsers(dest="pool_cmd")
    p_imp = pool_sub.add_parser("import", help="Validate and queue a pool.v1 file")
    p_imp.add_argument("path")
    p_sel = subs.add_parser("select", help="Choose two items for a run")
    p_sel.add_argument("--run-id", required=True)
    p_base = subs.add_parser("baseline", help="Baseline two items without writing the store")
    p_base.add_argument("--pool", required=True)
    p_post = subs.add_parser("record-post", help="Record HTTP result for a selected id")
    p_post.add_argument("--run-id", required=True)
    p_post.add_argument("--id", required=True)
    p_post.add_argument("--http-status", type=int, required=True)
    p_post.add_argument("--fallback", default="false")
    subparser.set_defaults(func=_handle_cli)


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("post_api_request", on_post_api_request)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    register_middleware = getattr(ctx, "register_middleware", None)
    if register_middleware is not None:
        register_middleware("llm_execution", on_llm_execution_middleware)
    HOOKS_REGISTERED["pre_tool_call"] = True
    HOOKS_REGISTERED["post_tool_call"] = True
    HOOKS_REGISTERED["post_api_request"] = True
    HOOKS_REGISTERED["pre_llm_call"] = True
    HOOKS_REGISTERED["llm_execution_middleware"] = register_middleware is not None
    register_cli = getattr(ctx, "register_cli_command", None)
    if register_cli is not None:
        register_cli(
            name=PLUGIN_NAME,
            help="Inspect agent-learning opslog",
            setup_fn=_setup_argparse,
            handler_fn=_handle_cli,
        )