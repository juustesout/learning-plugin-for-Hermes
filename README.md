# Hermes Agent Learning

A self-learning agent system that improves decisions through experience. Built on three levels of learning — from individual edge choices to full graph trajectory optimization.

```
Edge Learning ──→ Path Learning ──→ Graph Evolution
(level 1)         (level 2)         (level 3)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Graph Evolution                          │
│  DagRegistry · MultiDagOrchestrator · GraphOptimizer       │
│  Dashboard (HTML) · RunAnalyzer                            │
├─────────────────────────────────────────────────────────────┤
│                    Path Learning                            │
│  WorkflowDAG · WorkflowRunner · Credit Assignment (γ^d)    │
│  Episode Logging · Reward Propagation                      │
├─────────────────────────────────────────────────────────────┤
│                    Edge Learning                            │
│  SoftmaxPolicy · ContextualSoftmaxPolicy · ThompsonPolicy   │
│  AdaptiveBanditLearner · ContextualAdaptiveBandit          │
├─────────────────────────────────────────────────────────────┤
│                    Storage Layer                            │
│  agent-learning store (local/Cosmos) · Episodes · Rewards  │
│  Policy snapshots · Metrics                                │
└─────────────────────────────────────────────────────────────┘
```

### Level 1 — Edge Learning
Each decision point has a `TaskPolicy` that learns which action works best. Three policy types:

| Policy | Method | Best for |
|--------|--------|----------|
| `SoftmaxPolicy` | REINFORCE with baseline | Established patterns, many episodes |
| `ContextualSoftmaxPolicy` | W·φ gradient per context | Site-type dependent decisions |
| `ThompsonPolicy` | Beta(α,β) sampling | Cold start, few episodes |

### Level 2 — Path Learning (WorkflowRunner)
Multi-step workflows as DAGs. Each node is a decision point with its own policy. After execution, the global reward is backpropagated along the path:

```
credit[node] = global_reward × γ^distance_from_end
```

Example podcast DAG with γ=0.8:
```
topic_select → research → script → audio → video → publish
dist:   5          4          3        2       1         0
γ^d:  0.33       0.41       0.51     0.64    0.80     1.00
```

Pre-decision option stats are available per node via `decision_stats.py`.
`WorkflowRunner` injects a snapshot into `run.context["decision_advice"][node_id]`
before `decision_fn` is called. Override knobs in run context:

- `decision_stats_lookback_days` (default `30`)
- `decision_stats_min_n` (default `5`)
- `decision_stats_operations_path` (optional custom `operations.jsonl` path)

Each episode metadata now includes `decision_options` and `decision_meta`
for traceability of pre-decision evidence, plus `decision_meta.decision_gap`
with expected-vs-actual deltas (`success_error`, `latency_error_ms`,
`cost_error`, `utility_error`, `surprise_score`).

### Level 3 — Graph Evolution
- **DagRegistry** — multi-DAG management with meta-policy selection
- **MultiDagOrchestrator** — sequential, conditional, looped, and meta-DAG execution
- **GraphOptimizer** — gamma optimization, bottleneck detection, reorder suggestions
- **Dashboard** — self-contained HTML dashboard with per-node stats and reward trends

---

## Quick Start

```bash
# Install dependencies
pip install agent-learning numpy

# Train a scrape strategy policy
python3 train_adaptive.py --agent-id hermes --task-id choose-scrape-strategy

# Train with Thompson sampling (cold start)
python3 train_thompson.py --agent-id hermes --task-prefix podcast-
```

### Podcast Workflow

```bash
# Show the DAG
python3 podcast_workflow.py --show

# Dry run (simulated)
python3 podcast_workflow.py --dry-run --topic "AI Agents in 2026"

# With forced failure for reward testing
python3 podcast_workflow.py --dry-run --topic "Test" --force-fail publish

# Show credit assignment table
python3 podcast_workflow.py --simulate-rewards --discount 0.8

# Real run with store logging
python3 podcast_workflow.py --run --topic "AI Agents"
```

### Dashboard

```bash
python3 -c "
from graph_evolution import Dashboard
from agent_learning.storage.cosmos import get_default_store
store = get_default_store()
html = Dashboard(store).html_dashboard()
open('dashboard/index.html', 'w').write(html)
print('Dashboard → dashboard/index.html')
"
```

Security note: dashboard output now HTML-escapes all episode metadata fields
before rendering table cells/headings. This prevents script injection when runs
contain untrusted values (for example custom `action_id` strings).

### Workflow registry

Workflow structure is declared in `workflows/*.json`; runtime logs only provide
observed status and metrics. Validate and inspect the central register with:

```bash
python3 workflow_registry.py validate
python3 workflow_registry.py status --all
python3 workflow_registry.py status --all --json
```

The nightly `opslog-consolidate` job writes the machine-readable snapshot to
`dashboard/registry.json`. Operations can optionally carry `work_unit`,
`trigger`, `execution_mode`, `parent_run_id`, and `skills_involved` without
forcing every operation to belong to a workflow.

---

## Project Structure

```
├── adaptive_learner.py        # AdaptiveBanditLearner (logit decay, failure det.)
├── contextual_learner.py      # ContextualAdaptiveBanditLearner (W·φ gradients)
├── thompson_policy.py         # ThompsonPolicy + ThompsonLearner (Beta sampling)
├── scrape_features.py         # Context feature schema (8 dims)
├── train_adaptive.py          # Training CLI (marginal + contextual)
├── train_thompson.py          # Training CLI (Thompson sampling)
│
├── workflow_dag.py            # DAG dataclasses (Node, Edge, topo sort)
├── workflow_runner.py         # Sequential executor + credit assignment
├── decision_stats.py          # Pre-decision stats from opslog per option
├── workflow_visualize.py      # ASCII + Mermaid + run reports
├── podcast_workflow.py        # Concrete 6-node podcast DAG + CLI
│
├── graph_evolution.py         # DagRegistry, MultiDagOrch, GraphOptimizer, Dashboard
├── hn_ranker.py               # HN slot coin-flip + item ranker (prior-v1 / logreg)
├── hn_sim.py                  # Accelerated local posting job (loopback WP mock)
├── sim/                       # WordPress draft mock + loopback client
├── blind_label.py             # Blind labels, enqueue, audit for HN evidence
├── skill_pack.py              # Export/import shareable workflow.json packs
├── skill_schema/              # Versioned JSON Schema + stdlib validator
│                              #   (workflow, node, episode, reward, case)
├── schema/workflow.schema.json  # compat alias → skill_schema/schemas/skill.schema.json
├── hermes_plugin/             # Hermes observer — standaard sinds 17 Nov 2026
├── shadow_policy.py           # Shadow advisor: snapshot read + decision log
├── features.py                # Track A pre-call / Track B post-fetch register
├── FEATURES.md                # How Track A/B work; what is not built yet
├── replay_advice.py           # Offline Track A advice replay (measurement gate)
├── REPLAY.md                  # Spec: offline advice replay gate
├── MVP.md                     # MVP scope and acceptance criteria
├── STATUS.md                  # Generated status; do not edit by hand
├── scripts/status.py          # Regenerates STATUS.md (pytest collect-only + plugin_status)
├── eligibility.py             # Shared eligible_scrape() for replay/train/consolidate/shadow
│
├── cron_workflow.sh           # Daily cron: train + run + dashboard
├── train_scrape_policy.sh     # Every-6h cron: scrape policy training
│
├── store/                     # Runtime data (gitignored)
│   ├── episodes/
│   ├── policies/
│   ├── rewards/
│   ├── runs/
│   ├── operations.jsonl
│   └── shadow_decisions.jsonl
├── docs/                      # Documentatie voor andere agents/developers
│   ├── hermes-skill.md        # SKILL.md snapshot (Hermes skill store is source of truth)
│   └── archive/               # Stale session notes (SESSION-RESUME.md, MONKEY.md)
└── pyproject.toml
```

---

## Policy Hierarchy

```
Policy
├── SoftmaxPolicy              # Marginal logits → softmax probabilities
│   └── ContextualSoftmaxPolicy  # W·φ feature-weighted logits
└── ThompsonPolicy             # Beta(α,β) posterior per action
```

## Learner Hierarchy

```
Learner
└── ReinforceLearner
    └── AdaptiveBanditLearner (logit decay + failure det. + dynamic LR)
        └── ContextualAdaptiveBanditLearner (W·φ gradients)
```

## Operation Logging (OpsLog)

Tool-operaties worden automatisch gelogd naar `store/operations.jsonl` via de
**Hermes plugin** (zie sectie hieronder). Geen handmatige logging nodig.

De nachtelijke cron (`opslog-consolidate`, 03:00 UTC) consolideert de ruwe logs
naar volwaardige leer-episodes.

| Component | Locatie |
|-----------|---------|
| **Hermes plugin** | `hermes_plugin/__init__.py` — `pre_tool_call` shadow advice + generation capture via `llm_execution` middleware/`post_api_request` + `post_tool_call` → `opslog.log_operation()` (`source=hermes`) |
| **Automation logger** | `run_log.py` — cron/workflow steps write themselves (`source=automation`) |
| **Consolidator** | `consolidate_ops.py` — ruwe log → episode + reward |
| **Live log** | `store/operations.jsonl` (append-only) |
| **Run heartbeat** | `store/runs_automation.jsonl` — start/end per automation run |
| **Cron** | `opslog-consolidate` — elke nacht 03:00 UTC |

Automations (podcast cron, HN 6h job) do not go through the Hermes hook.
They log with `run_log.run_log(workflow)`: one opslog row per step, one
start/end pair in `runs_automation.jsonl`. A step without `ok()`/`fail()`,
or an exception, is a failure and is re-raised. `run_log.heartbeat()`
flags missing or unfinished runs. Podcast/cron wrappers are a follow-up.

### Task-ID conventies

| Task-ID | Action-Type |
|---------|-------------|
| `choose-scrape-strategy` | `web_extract`, `browser_navigate`, `mcp_webclaw_webclaw_scrape` |
| `db-query` | database_query |
| `wp-publish` | wp_rest_api |
| `web-research` | `web_search` |
| `podcast-*` | research, script, audio, publish |

### Cron Schedule

| Job | Schedule | Action |
|-----|----------|--------|
| `train-scrape-policy` | Every 6 hours | Train ContextualAdaptiveBandit on scrape episodes |
| `podcast-workflow-daily` | Daily 13:00 UTC | Train all policies + run podcast workflow + dashboard |
| `opslog-consolidate` | Daily 03:00 UTC | Consolidate operations.jsonl → learning episodes |

Cron scripts zijn nu path-portable: ze gebruiken standaard de repository-map
als basis en accepteren overrides via environment variables:

- `AGENT_LEARNING_LOCAL_STORE_DIR` (default: `<repo>/store`)
- `DASHBOARD_DIR` (default: `<repo>/dashboard`, alleen `cron_workflow.sh`)

## Hermes plugin (standaard sinds 17 Nov 2026)

`hermes_plugin/` is een thin Hermes observer. Hij logt elke Hermes tool-call
automatisch naar `store/operations.jsonl` via `opslog.log_operation()`.
`pre_tool_call` also records what the active policy would advise
(`store/shadow_decisions.jsonl`) and always returns `None`. Shadow mode
does not block, modify, or swap tools. Advice uses Track A pre-call
features from `features.py` (URL + domain memory). HTML/header features
are Track B and only stored after the fetch.

Enabling the plugin is the opt-in. Tool-call success (`status == ok`) is not
user-task success; metadata stores `tool_ok` and `task_success=false`.
When available, OpenRouter generation IDs are correlated per `api_request_id`
and stored as both top-level `generation_ids` and
`metadata.generation_ids` in `operations.jsonl`.

### Installatie

`plugin.yaml` and `__init__.py` live at the repository root, so a Git install
needs no symlinks. `hermes_plugin/` remains the implementation package.

```bash
hermes plugins install juustesout/hermes-agent-learning --enable
hermes plugins doctor agent-learning --ci
```

From a local checkout:

```bash
hermes plugins install . --enable
hermes plugins doctor . --ci
```

Link the whole repository as the plugin (not a subset copy):

```bash
make install-plugin
make check-plugin-sync
```

`make install-plugin` creates a symlink from `HERMES_PLUGIN_DIR`
(default `~/.hermes/plugins/agent-learning`) to this repo. It does not
replace an existing directory. Remove the old copy on the Hermes machine
first if that path already exists.

`make check-plugin-sync` is the drift check. Run it on the Hermes machine
after every install or code change. CI only smoke-tests that install can
create the symlink; it cannot see the live Hermes plugin path.

Keep the repo importable (`PYTHONPATH=<repo>`, `AGENT_LEARNING_REPO=<repo>`,
or `pip install -e .`) so the hook can import `opslog`. If `opslog` is missing,
the hook is a no-op.

### Store and privacy

| Variable | Default | Meaning |
|----------|---------|---------|
| `AGENT_LEARNING_OPSLOG_DIR` | `AGENT_LEARNING_LOCAL_STORE_DIR` or `<repo>/store` | Where `operations.jsonl` is written |
| `AGENT_LEARNING_LOCAL_STORE_DIR` | `<repo>/store` | Shared learning store |
| `AGENT_LEARNING_OPSLOG_ENABLED` | `1` | Set `0` to keep the plugin loaded but silent |
| `AGENT_LEARNING_OPSLOG_SESSION_IDS` | `0` | Set `1` to store Hermes session/turn ids |
| `AGENT_LEARNING_OPSLOG_RETENTION_DAYS` | `30` | Drop older operations; `0` disables prune |
| `AGENT_LEARNING_TOOL_TASK_MAP` | built-in web tools | JSON object `{tool_name: task_id}` |
| `AGENT_LEARNING_TOOL_TASK_MAP_FILE` | unset | JSON file with the same mapping |
| `AGENT_LEARNING_SHADOW_ENABLED` | `1` | Set `0` to skip shadow advice |
| `AGENT_LEARNING_SHADOW_KILL` | unset | Set `1` to disable shadow advice without a code change |

Logged fields: `task_id`, `action_id`, redacted `target` (query strings stripped),
redacted error summary, latency, tool status. No tool results. Session ids are
off unless enabled. Query parameters and common secret patterns are redacted.
There is no encryption; treat the JSONL file as sensitive.

Check write health with `hermes agent-learning health` or `python3 opslog.py health`.

### CLI

```bash
hermes agent-learning stats
hermes agent-learning count
hermes agent-learning health
hermes agent-learning tail -n 20
hermes agent-learning status
hermes agent-learning selftest
```

`status` is one text block: hooks, feature import, opslog counts, last
consolidatie, replay-gate. No JSON, no LLM-call, no tool-call.

`selftest` runs the real `pre_tool_call` / `post_tool_call` hooks with
Hermes kwargs against a temporary store. It never writes the live store.
There is no CLI hook mode and no marker in tool output.

## Blind labels (HN evidence)

Frozen before measurement: `evidence/success_definition.json`,
`evidence/hn_baseline.json`, and `evidence/hn_ranker.json`. Baseline = first
remaining relevant item in source order after hard dedupe. Learned = `hn_ranker.py`
on the same remaining pool. HTTP 2xx is technical success only.

Randomization is one coin per posting slot (`p_learned=0.5`). The 6h job must
select, then post, then enqueue. `select` also queues the whole candidate
pool for blind labels, not only the two posted items:

```bash
python3 hn_ranker.py select --run-id 2026-11-17T12 --json '[{"id":"...","title":"...","url":"...","rag_score":0.8,"source_rank":0}]'
python3 blind_label.py enqueue --json '{"id":"hn-1","title":"...","url":"...","summary":"...","http_status":201}'
python3 blind_label.py enqueue-pool --run-id 2026-11-17T12 --json '[{"id":"...","title":"...","url":"..."}]'
python3 blind_label.py record-run --attempted 2 --posted 2 --failed 0 --run-id 2026-11-17T12
python3 blind_label.py next
python3 blind_label.py label --id hn-1 --relevant yes --duplicate no --needs-correction no
python3 blind_label.py report
python3 blind_label.py audit
```

`items.jsonl` is rater-safe. `assignments.jsonl` holds one decision record per
slot (arm L or B, pool, chosen_id, features) and one outcome record per HTTP
attempt. `audit` exits 2 if posts lag, a post has no decision, or the kill
switch is on (`AGENT_LEARNING_HN_RANKER_KILL=1`).

Posted items keep a 20 percent sample. The candidate pool of each run is
queued in full so the other 28 items can be labeled too. Raters still see
only title, url, and summary. Ranker training uses those human labels,
not the automatic score. `hn_points` and `hn_comments` are logged, not
trained on. The ranker stays on `prior-v1` (`rag_score` descending) until
200 human-labeled pool items.

Local loopback UI for one rater. Binds `127.0.0.1` only and writes through
`record_label()`:

```bash
python3 label_ui.py
```

Open `http://127.0.0.1:8765`. The page shows title, url, and summary.
Keys: 1/2 relevant, 3/4 duplicate, 5/6 correction, Enter save.
If the queue is empty, fill it first. See `RUN_LABELS.md`.

External automations drop `inbox/pool-<run_id>.jsonl` (`pool.v1`). Import,
select, then record the HTTP result. Baseline does not write the store:

```bash
python3 pool_v1.py pool import inbox/pool-2026-11-19T12.jsonl
python3 pool_v1.py select --run-id 2026-11-19T12
python3 pool_v1.py record-post --run-id 2026-11-19T12 --id hn-123 --http-status 201 --fallback false
python3 pool_v1.py baseline --pool inbox/pool-2026-11-19T12.jsonl
```

Same commands exist as `hermes agent-learning pool import|select|baseline|record-post`.
A second import or select for the same `run_id` is a no-op. Sim rows are refused.

### Local mechanical simulation

`hn_sim.py` runs the 6h job against a loopback WordPress mock. Drafts only;
publish is 403. Isolate evidence so production files stay untouched:

```bash
AGENT_LEARNING_EVIDENCE_DIR=/tmp/hn-sim python3 hn_sim.py run --runs 200
pytest -q tests/test_wp_mock.py tests/test_hn_sim.py
```

This checks select → draft POST → enqueue → record-run → audit. It does not
measure whether the content is good. Sim rows are tagged `source=sim`.
`report` and ranker training ignore them. Audit reports L/B overlap,
`n_contrast` (slots where L != B), and a binomial arm-balance check.

The primary lift claim is intention-to-treat on all posted slots. `n_contrast`
and the contrast-subset effect are explanatory: ITT difference is about
`contrast_rate` times the effect where L and B actually differ. Sim contrast
numbers are still not quality evidence.

## CI (GitHub Actions)

Deze repo bevat CI in `.github/workflows/ci.yml`.
Bij pushes naar `main` en op pull requests draait de pipeline op Python 3.11 en 3.12:

- `ruff check . --select E9,F63,F7,F82` (kritieke lint checks voor syntax/name errors)
- `bandit -r .` (security static analysis, exclusief `store/`, `dashboard/` en `tests/`)
- `python -m compileall -q .` (syntax check)
- `pytest tests/`
- `python holdout_eval.py --baseline-only --json`
- `python test_adaptive_vs_standard.py` (vaste seeds)

## Dependency Updates & Vulnerability Audit

- Dependabot configuratie: [.github/dependabot.yml](./.github/dependabot.yml)
  - `pip` updates (wekelijks)
  - GitHub Actions updates (wekelijks)
- Fallback audit workflow: [.github/workflows/security-audit.yml](./.github/workflows/security-audit.yml)
  - draait `pip-audit` op pull requests, wekelijks en handmatig (`workflow_dispatch`)

Als Dependabot in een andere repo niet opent, controleer dan in GitHub:
- **Settings → Security & analysis** (Dependabot alerts + security updates)
- **Actions permissions** en of workflows mogen draaien
- of de default branch echt `main` is

---

## Related Work

This project builds on [Microsoft agent-learning](https://github.com/microsoft/agent-learning) (MIT License, v0.10.x) for the core REINFORCE learner and storage layer. That package is a runtime dependency (`agent-learning>=0.10.0,<0.11`); this repository does not vendor or re-license it. Additional algorithms and architecture are inspired by:

- **Self-Improving Agentic Systems** — LinUCB, Thompson sampling, cost-aware credit
- **AgentForge** — Generic workflow selector pattern
- **OpenJarvis** — Trace storage and skill discovery
- **SAP contextual-bandit-agent-router** — MCP-based bandit service pattern

Hermes is a product of Nous Research. This repository is an independent MIT-licensed observer/learning kernel (`juustesout/hermes-agent-learning`) and is not an official Nous Research distribution.

---

## License

MIT. Copyright (c) 2026 juustesout.