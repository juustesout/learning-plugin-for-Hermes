from __future__ import annotations

import json
from pathlib import Path

from workflow_dag import WorkflowDAG, WorkflowNode
from workflow_runner import (
    WorkflowRunner,
    _default_local_reward,
)


def _linear_dag(gamma: float = 0.8) -> WorkflowDAG:
    nodes = [
        WorkflowNode(
            id="a",
            name="A",
            task_id="task-a",
            action_ids=["do_a"],
            actions_short={"do_a": "Do A"},
        ),
        WorkflowNode(
            id="b",
            name="B",
            task_id="task-b",
            action_ids=["do_b"],
            actions_short={"do_b": "Do B"},
        ),
        WorkflowNode(
            id="c",
            name="C",
            task_id="task-c",
            action_ids=["do_c"],
            actions_short={"do_c": "Do C"},
        ),
    ]
    dag = WorkflowDAG(id="t", name="test", nodes=nodes, discount_factor=gamma)
    dag.add("a", "b")
    dag.add("b", "c")
    return dag


def _instant_ok(node, action_id, context):
    return "completed", f"ok {node.id} via {action_id} with plenty of content http://x", {
        "latency_s": 0.01,
        "cost": 0.1,
    }


def _fail_on_b(node, action_id, context):
    if node.id == "b":
        return "failed", "boom", {"latency_s": 0.01}
    return _instant_ok(node, action_id, context)


def test_dag_executes_in_topo_order():
    runner = WorkflowRunner(_linear_dag(), execute_fn=_instant_ok, auto_reward=True)
    run = runner.execute()
    assert [nr.node_id for nr in run.node_results] == ["a", "b", "c"]
    assert all(nr.execution_status == "completed" for nr in run.node_results)


def test_default_decision_picks_first_action():
    runner = WorkflowRunner(_linear_dag(), execute_fn=_instant_ok)
    run = runner.execute()
    assert run.node_results[0].action_id == "do_a"


def test_context_override_action():
    runner = WorkflowRunner(_linear_dag(), execute_fn=_instant_ok)
    run = runner.execute(context={"override_action:a": "alt"})
    assert run.node_results[0].action_id == "alt"


def test_credit_assignment_gamma_distance():
    dag = _linear_dag(gamma=0.5)

    def reward_fn(ctx, results):
        return 1.0, "unit"

    runner = WorkflowRunner(dag, execute_fn=_instant_ok, reward_fn=reward_fn, discount_factor=0.5)
    run = runner.execute()
    credits = {nr.node_id: nr.credit_reward for nr in run.node_results}
    assert credits["c"] == 1.0
    assert credits["b"] == 0.5
    assert credits["a"] == 0.25
    assert run.global_reward == 1.0


def test_failure_still_runs_remaining_nodes():
    runner = WorkflowRunner(_linear_dag(), execute_fn=_fail_on_b)
    run = runner.execute()
    assert [nr.execution_status for nr in run.node_results] == [
        "completed",
        "failed",
        "completed",
    ]
    assert run.global_reward == 2 / 3


def test_episode_logging_to_store(store):
    runner = WorkflowRunner(
        _linear_dag(),
        store=store,
        agent_id="hermes",
        execute_fn=_instant_ok,
    )
    run = runner.execute()
    episodes = store.query_episodes("hermes")
    assert len(episodes) == 3
    assert {ep.task_id for ep in episodes} == {"task-a", "task-b", "task-c"}
    assert all(nr.episode_id for nr in run.node_results)
    rewards = store.query_rewards("hermes")
    assert len(rewards) == 3
    assert all(r.agent_id == "hermes" for r in rewards)


def test_local_reward_stored_as_metric(store):
    runner = WorkflowRunner(
        _linear_dag(),
        store=store,
        agent_id="hermes",
        execute_fn=_instant_ok,
        local_reward_fn=_default_local_reward,
    )
    run = runner.execute()
    assert all(nr.local_reward is not None for nr in run.node_results)
    rewards = store.query_rewards("hermes")
    sources = {r.source.value for r in rewards}
    assert "metric" in sources
    assert "aggregate" in sources


def test_local_reward_detects_error_content():
    node = WorkflowNode(id="x", name="X")
    val, reason = _default_local_reward(
        node, "a", "completed", "timeout cloudflare access denied", {}, {}
    )
    assert val < 0.5
    assert "error_content" in reason


def test_local_reward_failed_status():
    node = WorkflowNode(id="x", name="X")
    val, reason = _default_local_reward(node, "a", "failed", "nope", {}, {})
    assert val == -0.5
    assert reason == "failed"


def test_annotate_execution_keys_in_context():
    runner = WorkflowRunner(_linear_dag(), execute_fn=_instant_ok)
    run = runner.execute()
    assert "_latency:a" in run.context
    assert "_cost:a" in run.context
    assert run.context["_cost:a"] == 0.1


def test_no_store_skips_episode_ids():
    runner = WorkflowRunner(_linear_dag(), execute_fn=_instant_ok)
    run = runner.execute()
    assert all(nr.episode_id is None for nr in run.node_results)


def _write_ops(store_dir: str, rows: list[dict]) -> None:
    path = Path(store_dir) / "operations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_decision_advice_available_before_decision(temp_store_dir):
    _write_ops(
        temp_store_dir,
        [
            {
                "task_id": "task-a",
                "action_id": "fast",
                "execution_status": "completed",
                "request_latency_ms": 100.0,
                "metadata": {"cost_credits": 0.02},
                "timestamp": "2026-11-22T12:00:00+00:00",
            },
            {
                "task_id": "task-a",
                "action_id": "slow",
                "execution_status": "failed",
                "request_latency_ms": 800.0,
                "metadata": {"cost_credits": 0.2},
                "timestamp": "2026-11-22T12:05:00+00:00",
            },
        ],
    )

    node = WorkflowNode(
        id="n1",
        name="N1",
        task_id="task-a",
        action_ids=["slow", "fast"],
    )
    dag = WorkflowDAG(id="decision", nodes=[node])

    def decide(node, context):
        advice = context["decision_advice"][node.id]
        return advice["recommended_action"], None, {"source": "decision_advice"}

    runner = WorkflowRunner(dag, decision_fn=decide, execute_fn=_instant_ok)
    run = runner.execute(context={"decision_stats_min_n": 1})

    advice = run.context["decision_advice"]["n1"]
    assert advice["recommended_action"] == "fast"
    assert set(advice["options"].keys()) == {"slow", "fast"}
    assert run.node_results[0].action_id == "fast"


def test_decision_advice_persisted_in_episode_metadata(store, temp_store_dir):
    _write_ops(
        temp_store_dir,
        [
            {
                "task_id": "task-a",
                "action_id": "do_a",
                "execution_status": "completed",
                "request_latency_ms": 120.0,
                "metadata": {"cost_credits": 0.01},
                "timestamp": "2026-11-22T10:00:00+00:00",
            },
            {
                "task_id": "task-a",
                "action_id": "do_b",
                "execution_status": "failed",
                "request_latency_ms": 900.0,
                "metadata": {"cost_credits": 0.15},
                "timestamp": "2026-11-22T10:05:00+00:00",
            },
        ],
    )

    dag = WorkflowDAG(
        id="t1",
        nodes=[
            WorkflowNode(
                id="a",
                name="A",
                task_id="task-a",
                action_ids=["do_a", "do_b"],
                actions_short={"do_a": "A", "do_b": "B"},
            )
        ],
    )
    runner = WorkflowRunner(dag, store=store, execute_fn=_instant_ok)
    run = runner.execute(context={"decision_stats_min_n": 1})

    assert run.node_results[0].episode_id
    episodes = store.query_episodes("hermes")
    assert len(episodes) == 1
    decision_meta = episodes[0].metadata["decision_meta"]
    assert decision_meta["decision_advice"]["recommended_action"] == "do_a"
    assert decision_meta["decision_advice"]["task_id"] == "task-a"
    assert decision_meta["decision_gap"]["selected_action"] == "do_a"
    assert decision_meta["decision_gap"]["actual"]["success"] is True
    assert decision_meta["decision_gap"]["errors"]["success_error"] == 0.0


def test_decision_gap_tracks_failed_outcome(store, temp_store_dir):
    _write_ops(
        temp_store_dir,
        [
            {
                "task_id": "task-a",
                "action_id": "do_a",
                "execution_status": "completed",
                "request_latency_ms": 100.0,
                "metadata": {"cost_credits": 0.02},
                "timestamp": "2026-11-22T09:00:00+00:00",
            },
            {
                "task_id": "task-a",
                "action_id": "do_b",
                "execution_status": "failed",
                "request_latency_ms": 900.0,
                "metadata": {"cost_credits": 0.3},
                "timestamp": "2026-11-22T09:05:00+00:00",
            },
        ],
    )
    dag = WorkflowDAG(
        id="t2",
        nodes=[
            WorkflowNode(
                id="a",
                name="A",
                task_id="task-a",
                action_ids=["do_a", "do_b"],
            )
        ],
    )

    def choose_forced(_node, _context):
        return "do_a", None, {"source": "forced"}

    def fail_exec(_node, _action_id, _context):
        return "failed", "forced failure", {"latency_s": 0.5, "cost": 0.31}

    runner = WorkflowRunner(dag, store=store, decision_fn=choose_forced, execute_fn=fail_exec)
    run = runner.execute(context={"decision_stats_min_n": 1})

    assert run.node_results[0].action_id == "do_a"
    episodes = store.query_episodes("hermes")
    gap = episodes[0].metadata["decision_meta"]["decision_gap"]
    assert gap["recommended_action"] == "do_a"
    assert gap["actual"]["success"] is False
    assert gap["errors"]["success_error"] < 0
    assert gap["surprise_score"] > 0
