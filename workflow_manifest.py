"""Versioned workflow manifest loader and validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1
MANIFEST_DIR = Path(__file__).with_name("workflows")
TRIGGERS = {"interactive", "scheduled", "webhook", "subtask", "manual"}
EXECUTION_MODES = {"session", "skill", "workflow", "external", "direct"}
STEP_KINDS = {"source", "transform", "decision", "sink", "human"}


class ManifestError(ValueError):
    pass


def load_manifest(path: str | Path) -> dict:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON in {source}: {exc}") from exc
    validate_manifest(data, source)
    return data


def load_manifests(directory: str | Path = MANIFEST_DIR) -> list[dict]:
    root = Path(directory)
    if not root.exists():
        return []
    manifests = [load_manifest(path) for path in sorted(root.glob("*.json"))]
    ids = [row["workflow_id"] for row in manifests]
    duplicates = sorted({workflow_id for workflow_id in ids if ids.count(workflow_id) > 1})
    if duplicates:
        raise ManifestError(f"duplicate workflow_id: {', '.join(duplicates)}")
    return manifests


def _require_string(row: dict, key: str, where: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{where}.{key} must be a non-empty string")
    return value.strip()


def validate_manifest(data: Any, source: Path | str = "manifest") -> None:
    where = str(source)
    if not isinstance(data, dict):
        raise ManifestError(f"{where} must contain a JSON object")
    if data.get("manifest_version") != MANIFEST_VERSION:
        raise ManifestError(f"{where}.manifest_version must be {MANIFEST_VERSION}")
    workflow_id = _require_string(data, "workflow_id", where)
    _require_string(data, "name", where)
    _require_string(data, "owner", where)
    _require_string(data, "work_unit", where)
    _require_string(data, "version", where)
    trigger = _require_string(data, "trigger", where)
    if trigger not in TRIGGERS:
        raise ManifestError(f"{where}.trigger must be one of {sorted(TRIGGERS)}")
    mode = _require_string(data, "execution_mode", where)
    if mode not in EXECUTION_MODES:
        raise ManifestError(f"{where}.execution_mode must be one of {sorted(EXECUTION_MODES)}")
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ManifestError(f"{where}.steps must be a non-empty list")
    step_ids: set[str] = set()
    for index, step in enumerate(steps):
        step_where = f"{workflow_id}.steps[{index}]"
        if not isinstance(step, dict):
            raise ManifestError(f"{step_where} must be an object")
        step_id = _require_string(step, "id", step_where)
        if step_id in step_ids:
            raise ManifestError(f"{where}: duplicate step id {step_id}")
        step_ids.add(step_id)
        kind = _require_string(step, "kind", step_where)
        if kind not in STEP_KINDS:
            raise ManifestError(f"{step_where}.kind must be one of {sorted(STEP_KINDS)}")
        for field in ("depends_on", "skills", "tools"):
            values = step.get(field, [])
            if not isinstance(values, list) or not all(isinstance(x, str) for x in values):
                raise ManifestError(f"{step_where}.{field} must be a list of strings")
    for step in steps:
        unknown = [x for x in step.get("depends_on", []) if x not in step_ids]
        if unknown:
            raise ManifestError(f"{workflow_id}: unknown dependencies {unknown}")
    outputs = data.get("outputs", [])
    if not isinstance(outputs, list) or not all(isinstance(x, str) for x in outputs):
        raise ManifestError(f"{where}.outputs must be a list of strings")
