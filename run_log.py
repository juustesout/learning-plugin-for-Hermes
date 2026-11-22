"""Automation run logger. Automations write themselves; Hermes only logs agent tools.

    with run_log.run_log("podcast-daily") as run:
        with run.step("research", action_id="llm_suggest") as s:
            s.ok(cost=1.0)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import opslog

SOURCE_AUTOMATION = "automation"
SOURCE_HERMES = "hermes"
RUNS_NAME = "runs_automation.jsonl"


def runs_path() -> str:
    return os.path.join(opslog.STORE_DIR, RUNS_NAME)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _append_run(row: dict) -> None:
    path = runs_path()
    payload = json.dumps(row, default=str) + "\n"

    def _write():
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(payload)
            f.flush()

    opslog._with_lock(path, _write)


def load_runs(workflow: str = "") -> list[dict]:
    path = runs_path()
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if workflow and row.get("workflow") != workflow:
                continue
            rows.append(row)
    return rows


def unfinished_runs(workflow: str = "") -> list[dict]:
    open_runs = {}
    for row in load_runs(workflow):
        rid = str(row.get("run_id") or "")
        if not rid:
            continue
        if row.get("record") == "start":
            open_runs[rid] = row
        elif row.get("record") == "end":
            open_runs.pop(rid, None)
    return list(open_runs.values())


def heartbeat(workflow: str, *, hours: int = 24, expected: int = 1) -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    ended = []
    for row in load_runs(workflow):
        if row.get("record") != "end":
            continue
        ts = _parse_ts(str(row.get("ended_at") or row.get("started_at") or ""))
        if ts is None or ts >= cutoff:
            ended.append(row)
    hanging = unfinished_runs(workflow)
    alarms = []
    if len(ended) < expected:
        alarms.append("missing_runs")
    if hanging:
        alarms.append("unfinished_runs")
    return {
        "ok": not alarms,
        "workflow": workflow,
        "hours": hours,
        "expected": expected,
        "ended": len(ended),
        "unfinished": len(hanging),
        "alarms": alarms,
    }


class Step:
    def __init__(
        self,
        run: "Run",
        name: str,
        *,
        action_id: str,
        task_id: str,
        target: str = "",
        item_id: str = "",
    ):
        self.run = run
        self.name = name
        self.action_id = action_id
        self.task_id = task_id
        self.target = target
        self.item_id = item_id
        self._outcome = None
        self._http_status = None
        self._cost = None
        self._summary = ""
        self._t0 = 0.0
        self.op_id = None

    def ok(self, *, http_status=None, cost=None, cost_credits=None, summary="", item_id=""):
        self._outcome = "ok"
        self._http_status = http_status
        self._cost = cost_credits if cost_credits is not None else cost
        if summary:
            self._summary = summary
        if item_id:
            self.item_id = item_id

    def fail(self, *, http_status=None, cost=None, cost_credits=None, summary="", item_id=""):
        self._outcome = "fail"
        self._http_status = http_status
        self._cost = cost_credits if cost_credits is not None else cost
        if summary:
            self._summary = summary
        if item_id:
            self.item_id = item_id

    def __enter__(self) -> "Step":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        duration_ms = int((time.perf_counter() - self._t0) * 1000)
        if exc_type is not None:
            success = False
            step_status = "exception"
            summary = self._summary or f"{getattr(exc_type, '__name__', 'error')}: {exc}"
            self.run.failed += 1
        elif self._outcome == "ok":
            success = True
            step_status = "ok"
            summary = self._summary
        elif self._outcome == "fail":
            success = False
            step_status = "fail"
            summary = self._summary
            self.run.failed += 1
        else:
            success = False
            step_status = "unknown"
            summary = self._summary or "step ended without ok() or fail()"
            self.run.failed += 1
        metadata = {
            "source": SOURCE_AUTOMATION,
            "run_id": self.run.run_id,
            "workflow": self.run.workflow,
            "step": self.name,
            "step_status": step_status,
            "http_status": self._http_status,
        }
        item_id = self.item_id or self.run.item_id
        if item_id:
            metadata["item_id"] = item_id
        self.op_id = opslog.log_operation(
            task_id=self.task_id,
            action_id=self.action_id,
            target=self.target,
            success=success,
            duration_ms=duration_ms,
            cost=self._cost,
            summary=summary,
            action_type=self.action_id,
            metadata=metadata,
            work_unit=self.run.work_unit,
            trigger=self.run.trigger,
            execution_mode=self.run.execution_mode,
            parent_run_id=self.run.parent_run_id,
            skills_involved=self.run.skills_involved,
        )
        self.run.steps += 1
        return False


class Run:
    def __init__(
        self,
        workflow: str,
        *,
        run_id: str = "",
        item_id: str = "",
        work_unit: str = "",
        trigger: str = "scheduled",
        execution_mode: str = "workflow",
        parent_run_id: str = "",
        skills_involved: Optional[list[str]] = None,
    ):
        self.workflow = workflow
        self.run_id = run_id or str(uuid.uuid4())
        self.item_id = item_id
        self.work_unit = work_unit
        self.trigger = trigger
        self.execution_mode = execution_mode
        self.parent_run_id = parent_run_id
        self.skills_involved = skills_involved or []
        self.steps = 0
        self.failed = 0
        self.status = "running"
        self.started_at = None
        self.ended_at = None

    def step(
        self,
        name: str,
        *,
        action_id: str = "",
        task_id: str = "",
        target: str = "",
        item_id: str = "",
    ) -> Step:
        return Step(
            self,
            name,
            action_id=action_id or name,
            task_id=task_id or self.workflow,
            target=target,
            item_id=item_id or self.item_id,
        )

    def __enter__(self) -> "Run":
        self.started_at = _now()
        _append_run(
            {
                "record": "start",
                "run_id": self.run_id,
                "workflow": self.workflow,
                "item_id": self.item_id,
                "work_unit": self.work_unit,
                "trigger": self.trigger,
                "execution_mode": self.execution_mode,
                "parent_run_id": self.parent_run_id,
                "skills_involved": self.skills_involved,
                "status": "running",
                "source": SOURCE_AUTOMATION,
            }
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.ended_at = _now()
        if exc_type is not None:
            self.status = "crashed"
        elif self.failed:
            self.status = "partial"
        else:
            self.status = "ok"
        _append_run(
            {
                "record": "end",
                "run_id": self.run_id,
                "workflow": self.workflow,
                "item_id": self.item_id,
                "work_unit": self.work_unit,
                "trigger": self.trigger,
                "execution_mode": self.execution_mode,
                "parent_run_id": self.parent_run_id,
                "skills_involved": self.skills_involved,
                "status": self.status,
                "ended_at": self.ended_at,
                "steps": self.steps,
                "failed": self.failed,
                "source": SOURCE_AUTOMATION,
            }
        )
        return False


def run_log(
    workflow: str,
    *,
    run_id: str = "",
    item_id: str = "",
    work_unit: str = "",
    trigger: str = "scheduled",
    execution_mode: str = "workflow",
    parent_run_id: str = "",
    skills_involved: Optional[list[str]] = None,
) -> Run:
    return Run(
        workflow,
        run_id=run_id,
        item_id=item_id,
        work_unit=work_unit,
        trigger=trigger,
        execution_mode=execution_mode,
        parent_run_id=parent_run_id,
        skills_involved=skills_involved,
    )
