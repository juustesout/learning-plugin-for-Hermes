# REPORT-NACHT

Generated 2026-11-23. Base `main` = `590be16`. No commit on `main`. No merge to `main`. Live store not touched.

## Suite-groen

**Ja, op `261123-integration`.** Command: `python3 -m pytest tests/ -q --tb=line`. Result: **311 passed**. Ruff `E9,F63,F7,F82`: all checks passed.

`STATUS.md` collected=311 matches `pytest --collect-only` on this worktree. Hooks after probe-`register()`: yes.

Hermes/Windows runtime: not in this worktree. Runtime checks remain a checklist, not claimed green.

## Remote SHAs

Pushed to `origin` (`https://github.com/juustesout/hermes-agent-learning`, moved notice to `learning-plugin-for-Hermes.git`). Local == remote:

| branch | SHA |
|---|---|
| `261123-p1-status` | `1c75b80` |
| `261123-p2-decision-record` | `6707f95` |
| `261123-p3-url-dedupe` | `95dca15` |
| `261123-p4-p5-stats-confound` | `4982108` |
| `261123-p6-advice-hook` | `6811d80` |
| `261123-p7-p8-report-gate` | `efdc631` (was `cb032f1`; + Wilson docstring/tests) |
| `261123-p9-middleware-kill` | `feea1ba` |
| `261123-p10-status` | `93ceef4` |
| `261123-p11-fixtures` | `d02eb48` |
| `261123-fix3-jsonl-cache` | `5ee7765` (was `8620972`; + head/tail fingerprint) |
| `261123-integration` | see tip after push |

P11 alleen op `main`-gedrag (5 IDs). Collapse 5 IDs -> 1 URL zit op P3 en op de integratiebranch.

## Fix 1 — armtoewijzing altijd voor de gate

- **Probleem:** closed gate mag geen injectie doen, maar arm L moet wel gelogd. Toewijzing na de gate maakt L onzichtbaar.
- **Fix:** `assign_arm` altijd p=0.5. Alleen prompt-injectie hangt van `gate.open`. Closed gate: arm L in `advice_calls.jsonl`, prompt byte-identiek.
- **Test:** `tests/test_advice_hook.py`.
- **Negatieve controle:** `_legacy_assign_after_gate` logt L niet bij closed gate; nieuwe code wel.
- **Branch:** `261123-p6-advice-hook` (`6811d80`).
- **Kill:** `AGENT_LEARNING_ADVICE=off`. Fail-open. `pre_tool_call` blijft `None`.

## Fix 2 — poortregel Wilson, niet ITT-marge

- **Probleem:** ITT-marge 5pp opent 16/20 vs 10/20 terwijl Wilson 95% overlapt.
- **Regel:** open alleen als unrounded `wilson_l_low > wilson_b_high`, z=1.96:

      denom  = 1 + z^2 / n
      center = (p + z^2 / (2n)) / denom
      margin = (z / denom) * sqrt(p(1-p)/n + z^2 / (4 n^2))
      low    = max(0, center - margin)
      high   = min(1, center + margin)

- **n_contrast is TOTAAL**, niet per arm: aantal beslissingen waar `chosen_id != baseline_pick_id`. Wilson zelf gebruikt `n_l` / `n_b` (ITT per arm). `MIN_N_CONTRAST=1500` is een floor op dat totaal. `MIN_N=12` blijft de totale ITT-count en vervangt die floor niet.
- **5pp bij p≈0.5, n=1500 per arm (788/1500 vs 712/1500):** L_low=0.500030, B_high=0.499970, gap=6.0e-5. Gate opent. Afronden op 4 decimalen geeft 0.5000 vs 0.5000; de code vergelijkt floats, niet de afgeronde print.
- **z-test ter vergelijking:** dezelfde 5pp-split haalt |z|>=1.96 rond n=742/arm (390/742 vs 352/742, z≈1.973). Wilson non-overlap heeft n=1500/arm nodig, ~2.0x die z-test-n. Strenger, expres.
- **16/20 vs 10/20 blijft dicht:** L_low=0.584 overlapt B_high=0.701.
- **Test:** `tests/test_promotion_gate.py` inclusief `test_wilson_5pp_at_1500_per_arm_separates_on_full_floats` en `test_n_contrast_is_total_not_per_arm`.
- **Negatieve controle:** `_legacy_margin_only_open` is True op 16/20 vs 10/20; nieuwe poort blijft dicht.
- **Branch:** `261123-p7-p8-report-gate` (`efdc631`).

## Fix 3 — jsonl-cache, 60s-grens niet omhoog

Omgeving (deze worktree): Python 3.11.2, Linux 6.6.116 x86_64 glibc 2.36, 2 cores Intel Xeon @ 2.50GHz, MemTotal 8157088 kB, load (0.37, 0.18, 0.07).

Drie runs van `hn_sim.run_sim(runs=200, error_rate=0.0, auto_label=True)`:

| run | seed | elapsed_s | wall_s |
|---|---|---|---|
| 1 | 1 | 17.801 | 18.295 |
| 2 | 2 | 17.753 | 18.249 |
| 3 | 3 | 17.739 | 18.230 |

Allemaal < 60. Assert ongewijzigd.

Zonder cache (eerdere ronde, zelfde machineklasse): 69.397 / 68.79 / 69.007, fail `assert 69.52 < 60`.

**mtime-risico:** cache-key is nu `(mtime_ns, size, head64, tail64)`. Tests: frozen-mtime append blijft zichtbaar; same-size rewrite met frozen mtime wordt niet als hit genomen. Bedoeld voor FAT/Windows 2s mtime.

- **Branch:** `261123-fix3-jsonl-cache` (`5ee7765`).

## Fix 4 — STATUS hooks niet unchecked `no`

- Startwaarde `HOOKS_REGISTERED` = `None`. `no` alleen bij expliciet `False`; anders `yes` na `register()` of `unknown`.
- **Branches:** `261123-p1-status` (`1c75b80`), `261123-p10-status` (`93ceef4`).

## Fix 5 — split + integratie

Tien pakketbranches vanaf `main`, plus `261123-integration` met alle tien erop. Twee merge-conflicten in `hermes_plugin/__init__.py` (CLI `record-outcome`+`report`; `HOOKS_REGISTERED` None). Weekend-fixturetest op integratie verwacht collapse (P3+P11 samen).

## Overige pakketten

- **P2** `6707f95`: additive `decision.v1`, orphans, CLI `record-outcome`, kill `AGENT_LEARNING_DECISION_RECORD=off`.
- **P3** `95dca15`: `canonical_url` in pool/queue/ranker. 5 IDs / 1 URL -> 1 candidate.
- **P4+P5** `4982108`: Wilson/median/quality; confound hour/weekday, MIN_N=30, ALPHA=0.01, observer only.
- **P9** `feea1ba`: middleware gehouden; kill `AGENT_LEARNING_MIDDLEWARE=off`; extract-fout byte-identiek.
- **P11** `d02eb48`: on-disk fixtures. Alleen op deze branch: weekend-dupes blijven 5 IDs.

## Drempels niet verlaagd

- replay/status `min_n`: 12
- gate `MIN_N`: 12, `MIN_N_CONTRAST`: 1500, `MIN_MARGIN`: 0.05
- confound `MIN_N`: 30, `ALPHA`: 0.01
- 200-label eis: ongewijzigd
- sim `elapsed_s < 60`: ongewijzigd

## Kill switches / fail-open

| env | default | off |
|---|---|---|
| `AGENT_LEARNING_DECISION_RECORD` | on | skip additive decision log |
| `AGENT_LEARNING_ADVICE` | on | original messages |
| `AGENT_LEARNING_MIDDLEWARE` | on | `next_call(request)` unchanged |

`pre_tool_call` blijft `None`. Geen nieuwe hook-typen buiten `pre_llm_call`.

## Runtime checklist (niet uitgevoerd)

- [ ] plugin load in Hermes
- [ ] `pre_tool_call` still no-op at runtime
- [ ] `AGENT_LEARNING_ADVICE=off` returns original prompt
- [ ] arm B byte-identiek at runtime
- [ ] middleware attributes generation ids
- [ ] `record-outcome` / `report` CLI on installed plugin
- [ ] live evidence store unread/unwritten

## Not done

- Geen commit op `main`, geen merge naar `main`, geen PR.
- Live store / backfill / Windows scheduler: niet aangeraakt.
- N8N / dashboard / graph / Sloppy / Windows-scheduler: niet toegevoegd.
