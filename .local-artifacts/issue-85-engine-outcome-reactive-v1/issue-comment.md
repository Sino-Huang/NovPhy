## Executed #85 — engine channel verified; prevalence gate failed

**Ticket disposition: `readiness_or_precision_insufficient`. This issue remains OPEN holding the typed gate blocker.**

| Question | Disposition |
| --- | --- |
| Q1: engine-truth reactive paired differences | `readiness_or_precision_insufficient` |
| Q2: proxy-versus-engine instrumentation cross-check | `supported` |

### Execution and results

- Phase A: the existing game-side pig death/removal hooks were bound to a new closed-loop measurement channel, not to parser-derived cost. Four frozen rendered controls completed: **2 positive controls recorded pig removal; 2 negative controls recorded survival**. Event/lifecycle evidence agreed with retained rendered frames. `channel-verification.json` and its review gallery were published before the diagnostic pilot. No Unity source modification was necessary.
- Membership/systems/seeds: the #80 frozen 24 development states, four systems, and seeds 20260908/09/10; ordinal prior uses the predeclared nearest-admissible-to-8 fallback, ties lower. No new captures or sealed/final access.
- Pilot: **48/48 cells terminal: 47 valid executions, 1 retained startup timeout** (`hybrid-fixed-h1--seed20260909--issue-77-n1-001-a00`). No replacement or retry of that pilot cell. **Engine-truth first-shot success = 0/47 = 0.0000**, below the frozen 0.10 gate; minimum valid count 24 was met. The full 288-cell matrix was **barred and not executed**. Q1 full-matrix paired outcomes were not scored.
- Phase C: among 47 compared pilot cells, the #82 threshold proxy called success on **42**, versus **0** engine-truth successes: both-success 0, proxy-only 42, engine-only 0, both-failure 5. Raw-cell agreement **5/47 = 0.1064**. State-weighted paired proxy-minus-engine difference **+0.8958**, DESCRIPTIVE 95% bootstrap interval **[+0.6875, +1.0000]**, over four pilot states (systems averaged within seed, then seeds; 10,000 draws, seed 7201). This quantifies the instrumentation divergence; it is not a competence claim.
- Accounting: pilot supervisor wall **2486.9 s**, sum of recorded pilot worker walls **9309.9 s**, active decision work **24.43 s**. Phase-A control walls total **533.21 s**, plus **125.02 s** archived pre-shot harness-failure overhead. Per-cell wall/candidate/transition/MAC counts, per-seed/pooled accounting, and active predictor parameters are published. Active predictor parameters per transition: continuous h1/h5 **1,862,396**; hybrid h1 **1,441,516**; ordinal prior **0** (shared parameter tables counted in full; total hybrid model parameters separately reported as 1,837,690).

### Review, protocol deviations, and corrections

Owner `Engine85` delegated the required read-only review to `Engine85.Issue85Reviewer` after implementation and before gameplay. Four material findings and one media-retention finding were resolved before executed shots. Post-execution reporting corrections received a second read-only review (`Engine85.Issue85FixReviewer`).

The final audit found deviations, retained explicitly rather than erased:
1. Runtime content-hash passes had run during the campaign; post-freeze execution now uses retained identities/payload bindings instead.
2. The original Phase-A invocation timer started before lock acquisition and ended after lock release; its later no-op resume also overwrote the invocation wall. The gameplay itself held `/tmp/novphy-addexp-gpu.lock`; durable accounting was reconstructed from retained control receipts. The code now holds that exact lock over the entire measured phase and separates resume time from execution work.
3. Four earlier verification attempts failed in the harness **before any shot** because a decision key was missing. Their attempt trees/records and recovery chronology remain under `failed-harness/`; the four reported controls each subsequently executed once. No outcome-conditioned action or membership replacement occurred.
4. Pilot compute tables, Phase-C interval reporting, active-versus-total parameter accounting, and gallery/native-channel validation were corrected after execution. The frozen scientific definitions and plan were not rewritten; original publications and source snapshots remain retained.

See `review/reviewer-findings.json`, `review/post-execution-fixes.json`, and `verification-harness-recovery.json`. Exact reviewer completion time was not independently receipted; the review record reports bounded chronology rather than an invented timestamp.

### Deliverables and verification

- Source commit: **90e8d69**, pushed to `origin/env`: https://github.com/Sino-Huang/NovPhy/commit/90e8d69
- Runner: `scripts/run_engine_outcome_reactive_diagnostic.py`; all seven required mutually exclusive modes implemented.
- Local artifacts: `.local-artifacts/issue-85-engine-outcome-reactive-v1/` — frozen/superseded plans, channel verification, pilot gate, ledger, records/receipts, `summary.json`, `comparisons.csv`, `findings.md`, reviews, and preserved original publications.
- Retained rendered media/gallery: `data/issue-85-engine-outcome-reactive/` — channel-verification gallery plus diagnostic `manifest.json`/`index.html` covering all 288 scheduled identities, including failed/not-executed entries.
- Ran with `conda activate novphy` and `source env.sh`: **`python -u -m scripts.run_engine_outcome_reactive_diagnostic --validate` — exit 0**, recomputing published tables and checking retained native channel/gallery evidence.
- Combined focused suite for #85/#86: **54 passed**.
- Source/tests committed; large experiment artifacts/media remain local under the repository's existing ignore policy.

**Typed blocker retained:** engine-truth prevalence below the frozen gate; the reactive comparison is not answerable on this run. All prior ticket verdicts remain unchanged, including #80/#82; #64/#65/#76 remain untouched. No multi-shot, adaptation, or complete-gameplay claim.
