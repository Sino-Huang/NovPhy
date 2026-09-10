## #76 diagnostic/readiness-stop evidence is ready for consolidation

The full #76 existing-calibration diagnostic and its reporting supplement are
complete and exactly validated. This is a **pre-access**
`readiness_or_precision_insufficient` disposition, not a fresh negative experiment.
#75's validated `not_supported_by_this_pilot` result is preserved. #64/#65 remain
blocked/not run; no supported advancement receipt exists.

Scope: 24 calibration states (12 per normal single-force/multiple-forces family),
three paired seeds, 12 actions, 12 fixed systems and cached original/covered
adaptive controls. The 10,368 new fixed diagnostic records completed within the
declared resource budget. The reporting supplement adds no predictions, fitting,
capture or fresh/final access.

The supplement retains all 33 original training/symbolic/adaptation contrasts
and adds 34 omitted comparator contrasts: all 16 systems versus the frozen
ordinal09 prior and uniform expected regret, plus original/covered hybrid
adaptive versus independent continuous h5 selected using #75's full development
publication. No baseline is reselected on the 24-state diagnostic subset.

Key descriptive results, with improvement = reference regret minus tested regret:

- Original hybrid versus prior: -0.069444, paired-state 95% interval [-0.208333, 0.055556].
- Covered hybrid versus prior: -0.069444, interval [-0.180556, 0.013889].
- Each adaptive hybrid versus continuous h5: -0.027778, interval [-0.083333, 0.027778].
- Hybrid-trained continuous h5 has lower offset225 carrier MSE but worse action regret than independent continuous h5; those are separate findings, not a gameplay win.

These are descriptive intervals on reused development data, not new
confirmatory tests. Seed differences are averaged within each state before
resampling. Only 10/24 states have informative regret variation and 2/24 have
a pig-removing/level-clearing candidate. Hybrid adaptive selected no such
candidate in any seed on this subset. Original #15/#72/#74/#75 conclusions are
unchanged; no general impossibility result for hybrid world models follows.

Artifacts and documentation (local, not claimed archived):

- `docs/issue-76-findings.md` — final interpretation, claim limits and reproduction commands.
- `data/issue-76-closeout/summary.json`, `comparisons.csv`, `findings.md` — exact source-bound supplement and all component decisions.
- `.local-artifacts/issue-76-dynamics-diagnostic-v1/` — original plan/results/predictions/targets and explicit CSV validation repair receipt.
- `data/issue-76-review/` — original source-linked gallery, 24 existing WebMs, 72 frame links, curves and summaries.
- `data/issue-76-closeout-{publish,validate}.log` — successful original-plus-supplement validation logs.
- [#76 requirement checklist](https://github.com/Sino-Huang/NovPhy/issues/76#issuecomment-5615651006) — executed versus conditional/unexecuted scope.

The original CSV comparison bug was repaired with a byte-preserving comparison
and an exact original/current source receipt; plans, predictions and publication
were preserved. The original validator and the supplemental validator passed.
All 31 issue-specific tests passed, including nine new pairing/selection/report
regressions and the earlier 22 diagnostic/CSV-repair tests.

For #35, consolidate this evidence with distinct exploratory/development and
pre-access-stop labels; do not replace frozen primary experiments or mark #64/#65
completed. #36 should archive the local source/checkpoint/repair assets with
authority, #37 report the results and limits, and #38 account for conditional
scope. Fresh numeric population/power/protocol, expanded representation/training,
rendered compatibility and novel/fresh gameplay remain unexecuted. The
80-template/40-pair/45-cell inventory is metadata, not broad NovPhy performance.
This handoff does not complete #35–#38 or authorize further collection/repair.
