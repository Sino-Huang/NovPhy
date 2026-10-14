# Issue #108 — Gate B: the #112 v3 unit on the #104 sealed evaluation split

Parent: #108. Shared rules of #85 apply; #111 scoring contract adopted. #104, #109 and #112 records are read-only
inputs; nothing here edits the ICLR 2026 submission.

| | |
| --- | --- |
| Runner | `scripts/run_gate_b.py` |
| Artifacts | `.local-artifacts/issue-108-gate-b-v1/` (`findings.md`, `summary.json`, `compute.json`, `records/`) |
| Freeze | plan frozen 2026-10-14T09:36:20Z, commit abcd9a27, after a passed smoke and before any evaluation score |
| Validation | `python -u -m scripts.run_gate_b --validate` (exit 0); `pytest tests/test_issue_108_gate_b.py` |

Zero training and zero engine seconds. Intervals are member (level)-clustered percentile bootstraps (10 000 draws,
PCG64 7201), DESCRIPTIVE.

## Decision: Gate B **pass**

The unit LC : F-15-macro (hybrid, E99, #112 v3 recipe retrained on the #109 fit split) reaches a pooled grid AUC of
**0.7444 [0.7269, 0.7620]** on 311 mixed grid cells × 3 seeds. The rule needs ≥ 0.65 with a lower bound > 0.55, and at
most 10 % nonfinite verdict slots (observed 0).

| grid (311 levels) | pooled AUC | seed 20260908 | seed 20260909 | seed 20260910 |
| --- | --- | --- | --- | --- |
| **unit LC : F-15-macro** | **0.7444 [0.7269, 0.7620]** | 0.7141 [0.6917, 0.7362] | 0.8011 [0.7826, 0.8199] | 0.7181 [0.6961, 0.7404] |
| C* LC : C-F-15 (matched continuous baseline) | 0.7834 [0.7676, 0.7992] | 0.8262 [0.8091, 0.8431] | 0.7334 [0.7127, 0.7536] | 0.7905 [0.7688, 0.8120] |
| no-model action prior | 0.7203 [0.6965, 0.7442] | seed-free | | |

| reported comparison (grid, paired) | estimate | token |
| --- | --- | --- |
| pre-declared unit − C* | −0.0390 [−0.0442, −0.0338] | not_supported_by_this_experiment |
| unit − action prior | +0.0241 [−0.0008, +0.0487] | readiness_or_precision_insufficient |
| wrong-anchor control: right − wrong (480 fit-split anchors) | +0.1322 [0.1216, 0.1430] | supported |

Reading:
- **The ranking depends on the anchor.** From the wrong anchors the unit falls to 0.6122 [0.5966, 0.6275]. Rank
  stability between right and wrong anchors is 0.59, so the unit does not rank by the action alone.
- **The hybrid unit does not beat its matched baseline.** C*, the parameter-matched pure-continuous model trained with
  the same recipe, ranks better on every inventory except power, where the two tie. The pre-declared contrast is
  negative with an upper bound below 0. Gate B certifies the recipe's ranking, not a hybrid (joint horizon-description)
  advantage.
- **The action prior clears the AUC target on its own.** It ranks each of the 16 shared grid actions by its success
  frequency over the fit split (0.72, lower bound 0.70). The unit is +0.024 above it, with an interval that touches 0.
  This split has no lineage exposure, so the prior is not the inflated #112 held-out 0.90. It still shows that much of
  the grid AUC comes from the action marginal; the anchor-dependent share is what the wrong-anchor control measures.

## Evaluation set

- The #109 evaluation split as captured by the #104 campaign, stream `issue109-evaluation`:
  - 400 levels (200 each of type010103 and type010105);
  - 11 680 branches, all complete: 0 typed failures and 0 incomplete branches. All 400 levels have the 16 grid
    candidates; 120 levels also have angle, offset and power (all 60 candidates).
- Cells: per level × inventory, the complete verdict-bearing candidates in ordinal order with the campaign's engine
  pig-removal verdicts; mixed-verdict cells only. Counts: grid 311, offset 76, angle 78, power 47. These yields were
  published by #104 before the freeze.
- Anchors follow the training rule: E99 batch-1 parse (`wlc.encode_carrier`) of shot frame 0 of the level's first
  complete candidate in suffix order (`run_decision_dynamics_retrain.anchor_branches` / `anchor_carrier`). They were
  parsed after the freeze, under `deterministic_scoring()`.
- Not read: `issue109-policy` and every `issue110-*` stream. Rows of other streams are dropped when `branches.json` is
  loaded. The fit split enters only as the retrain's `fit-verdicts.json` (action prior) and its stored fit anchors
  (wrong-anchor pool).

## Protocol

- **Units:** the handoff `.local-artifacts/issue-112-v3-retrain-v1/gate_b_handoff.json` (token `supported`, 3 usable
  seeds each). Each checkpoint is hash-checked against the handoff before loading.
- **Scoring:** one process that did not train, on one RTX 5090. Per cell and checkpoint,
  `m99.rollout_endpoints(model, None, z0, actions, t96.arm_spec(request))` runs from the anchor to 225 over the cell's
  candidate actions (v1 `endpoint_block` for the unit's request). The cost is tie-free (`EndpointCosts.row` against
  the anchor used). Cell AUC is `m99.view`. Seeds pool as in v3: cells are (level, seed), and the bootstrap resamples
  levels.
- **Smoke (before the freeze, fit split and v3 records only):**
  - the cell construction reproduces all 680 retrain ranking cells (ordinals, verdicts, actions, anchors);
  - the anchor rule reproduces the stored retrain anchors bit for bit;
  - the scoring path equals v1 `endpoint_block` rows for the unit and C* on 3 fit grid cells;
  - the statistics code reproduces the v3 LOLO unit AUC 0.7153 [0.6564, 0.7731] and unit − C* +0.0477
    [−0.0349, 0.1489] exactly;
  - the grid action set is shared by every evaluation level and equals the fit grid;
  - evaluation-stream isolation holds, and the handoff checkpoint hashes match.
- **Guards** (a failure refuses publication): G1 isolation, G2 scoring policy on every record and the anchor store,
  G3 chronology (freeze ≤ anchor parse ≤ first score), G4 checkpoint binding. All pass.
- **Validate:** re-derives every table from the records and byte-compares `compute.json`, `summary.json` and
  `findings.md`. It also re-scores the unit's first seed on all 311 grid cells (max delta 0.0) and re-parses one
  anchor.

## Descriptive inventories (pre-declared; not part of the gate)

| inventory | cells (levels × 3) | unit | C* | action prior (fit overlap) | unit − C* |
| --- | --- | --- | --- | --- | --- |
| offset | 228 | 0.6670 [0.6503, 0.6827] | 0.7748 [0.7524, 0.7962] | 0.5000 (0.00) | −0.1078 [−0.1217, −0.0937] |
| angle | 234 | 0.6791 [0.6663, 0.6912] | 0.7590 [0.7422, 0.7752] | 0.8119 [0.7778, 0.8450] (1.00) | −0.0799 [−0.0929, −0.0673] |
| power | 141 | 0.8058 [0.7729, 0.8396] | 0.8068 [0.7711, 0.8440] | 0.5000 (0.00) | −0.0011 [−0.0096, 0.0072] |

Offset and power actions were never executed in the fit split, so the prior is constant there (AUC 0.5). On angle,
where the fit split executed every action, the prior beats both models.

## Decisions taken under delegation (recorded on #108)

1. **Wrong-anchor pool.** All 480 fit-split anchors (the retrain's `anchors-fit.pt`, the anchors the unit was trained
   on). This is the v2 held-out definition. Wrong rows are computed once per anchor, because the 16 grid actions are
   shared across levels (smoke-checked).
2. **Action prior.** The c2 held-out rule over `fit-verdicts.json`: success frequency of the identical action over
   the fit executions in the same inventory; an action never executed there costs 0. The prior value is repeated per
   seed, which keeps the v1 cell weighting.
3. **Nonfinite rule in the gate.** More than 10 % nonfinite unit grid verdict slots means fail (v1 G4); observed 0.
4. **Extra reported contrast.** Unit − prior (paired). Contrasts carry the repo's δ = 0.02 token rule; the
   wrong-anchor control uses the Q5 rule.
5. **Descriptive inventories.** Offset, angle and power on the 120 full-inventory levels: captured by the campaign and
   pre-declared in the plan as DESCRIPTIVE.
6. **One GPU, one process.** Scoring takes 64 GPU s, so it was not sharded over the 4 GPUs.

## Compute and disclosures

- Smoke 5 s; anchor parse 10 s; scoring 49 s (unit 12 s per seed including the 480 wrong-anchor rollouts, C* 4 s per
  seed). Total 64 GPU s and 64 wall s; validate 62 s; 0 engine seconds.
- RTX 5090 instead of the #85 RTX 3090. Inputs bind the current bytes. The #104 `plan.json` differs from its committed
  bytes only in five machine-local paths (renderer library and player build directories: `/home/sukai` →
  `/srv/home/shua0171`). The frozen plan's disclosure attributes this to the SERVER_SETUP path rewrite; it is a home
  directory rewrite of the same kind (corrected here after the freeze; no input, rule or statistic is affected).
- Known before the freeze: every #112 v1/v2/v3, #117 and retrain result, and the #104 mixed-verdict yields on this
  split. No model output on an evaluation capture existed before the freeze.
