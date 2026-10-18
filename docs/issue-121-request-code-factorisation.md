# Issue #121: request-code factorisation (fused MLP code vs additive vs factored-FiLM)

Parent: #108. The binding shared rules of #85 apply, and #121 adopts the #111 scoring contract. The Gate-B (#108),
#101, #104, #109, #112, #115 and #117 records are read-only inputs. Nothing here edits the ICLR 2026 submission.

| | |
| --- | --- |
| Runner | `scripts/run_request_code_factorisation.py` |
| Module | `world_model/training/request_code.py`: the additive and factored-FiLM predictor classes (subclasses of `CNNHybridPredictor`; `predictor.py` and `cnn_hybrid.py` are hash-pinned by earlier plans and unchanged), the per-arm parameter split, the FiLM table and the two-way coupling decomposition |
| Tests | `tests/test_issue_121_request_code.py` (11 tests): fused equals the original, the additive rank-2 structure, the factored-FiLM composition, matched trunk initialisation, parameter and MAC accounting, the D2 reading beside the frozen token |
| Artifacts | `.local-artifacts/issue-121-request-code-v1/`: `plan.json`, `smoke.json`, `findings.md` (every table), `summary.json`, `compute.json`, `records-manifest.json`, `ledger/`, `logs/`, `training/` reports, `models/policies/*/seed-*/report.json`. Checkpoints, records and the truth store stay local, hashed in `records-manifest.json` |
| Freeze | Plan frozen 2026-10-18T02:09:12Z, commit 750e4757, after a passed smoke and before any training update, evaluation byte or score |
| Validation | `python -u -m scripts.run_request_code_factorisation --validate` (exit 0: the tables are re-derived and byte-compared, and additive F-1-macro and factored J are re-scored on every grid cell); `pytest tests/test_issue_121_request_code.py` (11 passed) |

Engine time is zero seconds. The reading is confirmatory because Gate B passed: one score on the #104 sealed #109
evaluation split. The cells are the Gate-B cells (grid 311, offset 76, angle 78, power 47), × 3 seeds. Grid is primary;
offset, angle, power and the rollout error are DESCRIPTIVE. Intervals are level-clustered percentile bootstraps
(10 000 draws, PCG64 7201) and are DESCRIPTIVE. The margin is δ = 0.02. All seven guards pass.

## Dispositions (pre-declared)

| claim | token | evidence |
| --- | --- | --- |
| "an additive request code matches the fused code within 1.5× per-request rollout error at t = 225" | **`not_supported_by_this_experiment`** | additive / fused at t = 225: F-5-continuous 1.84 [1.79, 1.90] and F-5-micro 2.02 [1.96, 2.08] have lower bounds > 1.5. F-1-micro 1.51 and F-5-macro 1.53 are insufficient. The other five requests are < 1.5 (1.14–1.42) |
| "axis interaction in the request code carries measurable decision value" | **`supported`** under the frozen operational rule; demonstrated claim and qualifications below | fused − additive, grid, nine-fixed-request mean AUC: +0.050 [+0.041, +0.059]. J (reported beside it): +0.013 [+0.004, +0.022], insufficient |

**What the D2 token demonstrates.** The frozen token stays as the historical operational rule, but its demonstrated
claim is narrower than its wording: **the trained fused parameterization outperforms the trained additive
parameterization on fixed-request mean grid AUC.** fused − additive changes the trained parameterization and its
optimization as well as the interaction capacity. Interaction-specific decision value therefore remains unidentified:
this run neither shows it nor rules it out. Three findings qualify the token.

1. **The fused FiLM interaction share is small, not zero** (coupling diagnostic, below).
   - Over the nine requests, the interaction term is 0.26–0.43 % of the variation of the FiLM scale γ per block, and
     0.30–0.39 % of the shift β (seed means).
   - It is 0.69 % of the variation of the code itself.
   - In parameter space the fused FiLM is additive in (Δ, α) to within about 0.5 % (about 99.5 % additive variance).
     This is a diagnostic, not a decision-level test: a small share neither proves nor disproves a decision-relevant
     interaction effect, so it does not show that interaction is irrelevant.
2. **Adding a FiLM-level interaction to the additive code does not close the gap.** The factored-FiLM arm adds a
   multiplicative scale interaction to the additive arm.
   - factored − additive is −0.003 [−0.012, +0.006] (nine-request mean, `not_supported_by_this_experiment`).
   - fused − factored is +0.052 [+0.049, +0.056].

   factored − additive is a different parameterization, not the removal of the fused model's interaction, so it does
   not isolate that interaction's effect either. [INFERENCE] Candidate explanations of the fused − additive gap include
   how the code path is parameterised and optimised (a nonlinear MLP code versus linear per-axis maps, with different
   initial code scales), interaction, and retrain-to-retrain variation (point 3). This run does not separate them.
3. **The seeds disagree in size** (post hoc, DESCRIPTIVE; computed from the retained records after publication, not
   part of any token). Per seed, fused − additive on the grid nine-request mean is:
   - +0.052 (20260908);
   - +0.087 (20260909);
   - +0.012 (20260910).

   The sign is the same in all three seeds, but one is below δ. The level-clustered interval does not contain the
   training-seed variance, and #115 showed that retrain-to-retrain variation exceeds δ on this recipe.

**Neither code beats the no-model prior on grid.**
- Fused nine-request mean − prior is +0.001 [−0.025, +0.026]; fused J − prior is −0.015.
- Additive is −0.049 and −0.028.

The decision value in D2 is therefore relative between the codes. It is not value over the prior.

**Handoff to #105.**
- In parameter space, the implementation's fused code is additive in its FiLM effect to within about 0.5 %. This does
  not establish that encoding-level coupling is irrelevant to decisions, so it neither supports nor undercuts #105's
  schedule-level independence dispositions.
- The particular fused parameterisation still matters for accuracy: replacing it with the additive code costs 1.1–2.0×
  rollout error at t = 225 and −0.05 fixed-request AUC.

## Results

### Rollout error at t = 225 (#100 `engine_position_mse`, every mixed grid candidate)

| request | fused | additive | factored (2 seeds) | additive / fused | factored / fused |
| --- | --- | --- | --- | --- | --- |
| F-1-continuous | 0.0309 | 0.0430 | 0.0352 | 1.34 [1.30, 1.38] | 0.98 [0.95, 1.01] |
| F-1-micro | 0.0303 | 0.0466 | 0.0371 | 1.51 [1.47, 1.55] | 1.07 [1.04, 1.10] |
| F-1-macro | 0.0281 | 0.0416 | 0.0390 | 1.42 [1.37, 1.46] | 1.22 [1.19, 1.26] |
| F-5-continuous | 0.0383 | 0.0693 | 0.0448 | **1.84 [1.79, 1.90]** | 1.05 [1.03, 1.08] |
| F-5-micro | 0.0375 | 0.0741 | 0.0478 | **2.02 [1.96, 2.08]** | 1.16 [1.13, 1.18] |
| F-5-macro | 0.0446 | 0.0666 | 0.0531 | 1.53 [1.48, 1.58] | 1.04 [1.01, 1.07] |
| F-15-continuous | 0.0415 | 0.0573 | 0.0414 | 1.36 [1.33, 1.38] | 0.89 [0.87, 0.91] |
| F-15-micro | 0.0419 | 0.0566 | 0.0425 | 1.33 [1.31, 1.36] | 0.93 [0.91, 0.95] |
| F-15-macro | 0.0474 | 0.0556 | 0.0454 | 1.14 [1.12, 1.17] | 0.88 [0.86, 0.89] |
| J (own policy; DESCRIPTIVE) | 0.0317 | 0.0513 | 0.0404 | 1.56 [1.52, 1.60] | 1.09 [1.07, 1.12] |

Interval columns, and the error intervals in `findings.md`, are level bootstraps. The fused rows reproduce #115's
Gate-B fixed-micro errors exactly (G6).

- **The additive code is worse on every request at every endpoint.** Its ratio to fused grows with t:

  | t | 15 | 30 | 60 | 150 | 225 |
  | --- | --- | --- | --- | --- | --- |
  | additive / fused, range over the nine requests | 1.05–1.13 | 1.03–1.24 | 1.03–1.37 | 1.08–1.71 | 1.14–2.02 |

- **The loss is largest at Δ = 5 for every description** (1.53–2.02). At Δ = 15 it is 1.14–1.36. The ticket predicted
  the largest loss for the long-Δ micro and macro requests. The data do not show that pattern: the losses do not
  concentrate where interaction is plausible.
- **The factored arm stays within 1.5× of fused on every request at every endpoint** (0.84–1.27). At Δ = 15 it is
  below fused from t = 150 (0.84–0.93). Its largest excess is F-1-macro (1.22 at t = 225).

`findings.md` has every endpoint t ∈ {15, 30, 60, 150, 225} per request and arm.

### Ranking AUC (grid, primary; 933 level × seed cells)

| ranker | fused | additive | factored (2 seeds) | fused − additive | fused − factored | factored − additive |
| --- | --- | --- | --- | --- | --- | --- |
| F-1-continuous | 0.677 | 0.648 | 0.599 | +0.029 [+0.018, +0.040] (sup.) | +0.081 (sup.) | −0.055 (no) |
| F-1-micro | 0.650 | 0.598 | 0.663 | +0.052 [+0.040, +0.064] (sup.) | −0.029 (no) | +0.067 (sup.) |
| F-1-macro | 0.666 | 0.680 | 0.627 | −0.014 [−0.026, −0.002] (no) | +0.045 (sup.) | −0.058 (no) |
| F-5-continuous | 0.749 | 0.623 | 0.621 | +0.125 [+0.112, +0.139] (sup.) | +0.151 (sup.) | −0.001 (no) |
| F-5-micro | 0.745 | 0.619 | 0.699 | +0.126 [+0.111, +0.142] (sup.) | +0.073 (sup.) | +0.062 (sup.) |
| F-5-macro | 0.743 | 0.668 | 0.675 | +0.075 [+0.061, +0.089] (sup.) | +0.083 (sup.) | −0.003 (no) |
| F-15-continuous | 0.762 | 0.732 | 0.733 | +0.030 [+0.025, +0.035] (sup.) | +0.052 (sup.) | −0.015 (no) |
| F-15-micro | 0.760 | 0.732 | 0.763 | +0.028 [+0.023, +0.033] (sup.) | +0.031 (sup.) | −0.013 (no) |
| F-15-macro | 0.744 | 0.746 | 0.774 | −0.001 [−0.008, +0.005] (no) | −0.014 (no) | −0.013 (no) |
| **nine-request mean** | 0.722 | 0.672 | 0.684 | **+0.050 [+0.041, +0.059] (sup.)** | +0.052 [+0.049, +0.056] (sup.) | −0.003 [−0.012, +0.006] (no) |
| J (own policy) | 0.706 | 0.693 | 0.650 | +0.013 [+0.004, +0.022] (insuff.) | +0.074 (sup.) | −0.047 (no) |
| no-model prior | 0.720 [0.696, 0.744] | | | | | |

Tokens follow the δ rule: "sup." is `supported`, "no" is `not_supported_by_this_experiment`, "insuff." is
`readiness_or_precision_insufficient`. Contrasts with factored use its two usable seeds. The fused column reproduces
the #101 records exactly (G5): J 0.706, F-15-continuous 0.762.

- **The ranking gap follows the rollout gap.** It is largest at Δ = 5: +0.075 to +0.126 for fused − additive.
- **The J gap is small.** Each arm's own J (#101 recipe) narrows fused − additive to +0.013. The additive J
  policy-split AUCs vary widely by seed (0.667, 0.673, 0.783).
- **Factored J is the weakest** (0.650). Its policy-split AUCs are 0.637 and 0.612.

**DESCRIPTIVE inventories (nine-request mean / J):**

| inventory | fused − additive (mean) | fused − additive (J) | factored − additive (mean) |
| --- | --- | --- | --- |
| offset | −0.010 (no) | +0.026 (insuff.) | −0.147 |
| angle | +0.029 (sup.) | +0.031 (sup.) | −0.107 |
| power | +0.021 (sup.) | +0.043 (sup.) | −0.004 |

The factored arm transfers worst off-grid: offset nine-request mean 0.577 vs fused 0.716 and additive 0.726.

### Controls

- **No-model action prior.**
  - Grid 0.720 [0.696, 0.744]; angle 0.812.
  - Offset and power 0.500: the #109 fit split executes no offset or power candidate (the Gate-B rule).
  - On grid, J − prior is −0.015 (fused), −0.028 (additive) and −0.071 (factored). The nine-request mean − prior is
    +0.001, −0.049 and −0.036. None is `supported`.
- **Wrong anchor (J, grid; 480 fit-split anchors).** Right − wrong is `supported` in every arm:
  - fused +0.034 [+0.030, +0.037] (#114's +0.0336 reproduced);
  - additive +0.034 [+0.030, +0.039];
  - factored +0.026 [+0.020, +0.031].

### Coupling diagnostic (per block; seed means)

| arm | block | γ interaction share | γ var over α (Δ fixed) | γ var over Δ (α fixed) | γ rms | β interaction share | β var over α | β var over Δ | β rms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fused | 0 | 0.0026 | 0.0088 | 0.0071 | 0.825 | 0.0039 | 0.0230 | 0.0526 | 0.232 |
| fused | 1 | 0.0040 | 0.0095 | 0.0264 | 0.439 | 0.0030 | 0.0174 | 0.0129 | 0.189 |
| fused | 2 | 0.0043 | 0.0097 | 0.0333 | 0.344 | 0.0033 | 0.0122 | 0.0084 | 0.187 |
| additive | 0–2 | 0 | 0.009–0.015 | 0.015–0.063 | 0.35–0.91 | 0 | 0.019–0.049 | 0.009–0.052 | 0.20–0.25 |
| factored | 0 | 0.0018 | 0.0259 | 0.0079 | 1.110 | 0 | 0.4184 | 0.0148 | 0.343 |
| factored | 1 | 0.0015 | 0.0918 | 0.0125 | 0.556 | 0 | 0.2376 | 0.0074 | 0.250 |
| factored | 2 | 0.0014 | 0.1479 | 0.0098 | 0.419 | 0 | 0.1250 | 0.0087 | 0.224 |

- **Per seed, the fused interaction share** is 0.19–0.62 % (γ) and 0.23–0.55 % (β) per block, and 0.50–0.79 % for the
  code.
- **The fused modulation is large but mostly request-invariant.** γ rms reaches 0.83 in block 0. Within a block, the
  variation over α at fixed Δ is ≤ 2.3 % of the squared norm (≤ 1 % for γ), and over Δ at fixed α ≤ 5.3 %. So FiLM
  applies largely the same scale and shift to every request, and the small request-dependent part is almost entirely
  additive in parameter space (a diagnostic; see the D2 qualification above).
- **The additive arm is exactly additive** (share 0 in float64), as constructed.
- **The factored arm uses its multiplicative scale only a little** (γ share 0.14–0.18 %). Its α-dependence is larger
  than fused's: β var over α is 0.13–0.42 vs 0.01–0.02.

### MACs and parameters

| arm | parameters (request path) | MACs per call, continuous / micro / macro | MACs per candidate-state: F-1-continuous / F-5-continuous / F-15-continuous / J |
| --- | --- | --- | --- |
| fused | 1 837 690 (367 616) | 1 432 448 / 1 795 456 / 1 464 704 | 322.3M / 64.5M / 21.5M / 184.0M |
| additive | 1 786 234 (316 160) | 1 381 248 / 1 744 256 / 1 413 504 | 310.8M / 62.2M / 20.7M / 189.8M |
| factored | 1 806 842 (336 768) | 1 399 680 / 1 762 688 / 1 431 936 | 314.9M / 63.0M / 21.0M / 158.5M |

- **Per call.** The additive code saves 51 200 linear MACs per predictor call (−3.6 % for continuous); the factored
  code saves 32 768 (−2.3 %).
- **J.** J's MACs differ through its request schedule, not the call cost.
- **Accounting.** The code MLP is counted per row, as #103 / #114 do.

### Training

- **Members.** 6 members (2 arms × 3 seeds), 9000 updates each, 0 skipped updates.
- **One typed terminal failure.** Factored-FiLM seed 20260908 failed with `final_checkpoint_excursion` (#117 rule; the
  end-state local-loss ratio was 49 on F-1-continuous). It was not retried or replaced. It has no policy, its 11
  records are typed failures, and every factored contrast uses seeds 20260909 and 20260910.
- **Final local loss.** Additive 0.129 / 0.153 / 0.123; factored 0.214 (failed) / 0.155 / 0.124.
- **J policy-split AUC.** Additive 0.667 / 0.673 / 0.783; factored — / 0.637 / 0.612; fused (#101) per the #101 records.

## Guards

All seven pass.

| guard | result |
| --- | --- |
| G1 isolation | pass |
| G2 scoring policy | pass, on all 99 records |
| G3 chronology | pass: freeze 02:09:12Z ≤ first training report 02:36:54Z; last training report 03:02:11Z ≤ last policy 03:04:56Z ≤ first score 03:05:41Z; truth built 02:55:21Z |
| G4 checkpoint binding | pass |
| G5 replication | pass. Training path: bit for bit after 12 full-cost updates (smoke). Scoring: 15 360 fused blocks (nine fixed requests + J × 4 inventories × 3 seeds) equal the #101 records in rows, J schedules and linear MACs |
| G6 engine truth | pass. The t = 225 store equals #115's on all 4 976 grid branches; 2 799 fused fixed-micro blocks equal #115's engine errors |
| G7 nonfinite | pass: the fused J grid nonfinite share is 0.0 |

## Compute

| phase | GPU s |
| --- | --- |
| smoke | 151 |
| training (6 members) | 9 355 |
| J policies (5) | 946 |
| scoring (99 records, 4 workers) | 5 696 |
| **total** | **16 148** |

- **Other time.** Engine truth: 2 727 s of CPU wall (24 processes, concurrent with training). Engine seconds: 0.
- **Wall.** Summed over processes, 18 881 s. Elapsed from the freeze to the last score: 1.4 h on 4 GPUs.
- **Hardware.** 4 × RTX 5090 (the #85 protocol named one RTX 3090).
- **Process records.** The stages were launched once, and the worker PIDs are in `logs/pids-train.txt` and
  `logs/pids-score.txt`. The training launcher's own `wait` ran in a pipeline subshell, so it logged
  "exit 127" for each worker. The workers themselves were alive and were awaited by PID. Their completion was confirmed
  from the training and policy reports. Nothing was relaunched.

## Post-freeze amendments

**A1 (validation only).** The frozen `--validate` spot check re-scored the first seed (20260908) of two records:
additive F-1-macro and factored J. Factored seed 20260908 is the typed failure, so the J re-score was silently skipped.
After publication, the spot check was amended to use each record's first seed that is not a typed failure (factored J
→ 20260909).
- **What it changes.** Only what `--validate` re-checks.
- **What it does not change.** No record, table, rule or token changes, and neither do the published bytes.
- **Hashes.** The module hash pinned by the plan is unchanged. The runner hash at the freeze is in `plan.json`; no other
  plan pins this runner. Since the phase-B/C fix (A2), A1 is also listed in `scripts/post_freeze_amendments.py`, the
  one sanctioned registry; it was first disclosed only here.

**A2 (reporting only; phase-B/C review; commit `1dc6ed49`).** The second disposition's rendering now states what the
frozen operational token demonstrates. `compute.json` gains `demonstrated_claim` and `qualification` in
`dispositions.interaction_decision_value`, `summary.json` gains `interaction_decision_value_reading`, and `findings.md`
prints both under the dispositions table and adds to the coupling diagnostic that a small interaction share does not
show interaction to be decision-irrelevant. The frozen claim text, rule and token (`supported`) and every number are
unchanged; no record changed. `--validate` regenerates the new files byte for byte and exits 0. This document's
headline, mechanistic reading and #105 handoff were reworded to match. The runner (A1 and A2) is listed in
`scripts/post_freeze_amendments.py`.

## Claim boundary

- **Scope.** One look at the sealed split; NovPhy E99 carriers; the LC hybrid at the Gate-B budget (9000 updates, which
  #116 found looks undertrained on the fit split).
- **Intervals.** Level-clustered; they do not contain training-seed variance.
- **What "additive" means here.** It is one parameterisation (linear per-axis maps). The factored arm is one
  multiplicative FiLM composition.
- **No retuning.** None of the arms was tuned for its code: the learning rate, initialisation and budget are the Gate-B
  recipe's.

## Design

### How the request reaches the trunk (verified in code)

`CNNHybridPredictor.carrier` (`world_model/training/cnn_hybrid.py`):

1. **Code.** `c = PairConditioner.code(pair)` = MLP[ϕ(Δ); e_α]: ϕ is 16 sinusoids of Δ, e_α a learned 128-d embedding,
   then Linear 144 → 256, SiLU, Linear 256 → 128. The code is broadcast over the batch.
2. **FiLM.** Each of the three `FiLMBlock`s has a zero-initialised `Linear(128, 768)` that gives the scale γ and the
   shift β: `norm(h)(1 + γ) + β`.
3. **Adapter.** α also selects the input adapter: the micro or macro symbolic features of the current carrier are added
   to the input projection. This path depends on α only, never on Δ, and it is the same in every arm.

So the ablation changes steps 1–2 only. Even the additive arm is not additive end to end: α enters through the adapter
and the trunk is nonlinear.

### Arms

| arm | request code / modulation | request-path parameters | total |
| --- | --- | --- | --- |
| (a) fused (control) | `MLP[ϕ(Δ); e_α]`, one `Linear(128, 768)` FiLM generator per block: the Gate-B LC hybrid handoff checkpoints, not retrained | 367 616 | 1 837 690 |
| (b) additive | `c = W_δ ϕ(Δ) + b + W_α e_α` (Linear 16 → 128 with bias, Embedding 3 × 128, Linear 128 → 128 without bias); FiLM blocks unchanged | 316 160 | 1 786 234 |
| (c) factored-FiLM | per block `G_δ: Linear(16, 768)` on ϕ(Δ) and `G_α: Linear(128, 768)` on e_α, both zero-initialised: `norm(h)(1 + γ_δ)(1 + γ_α) + β_δ + β_α`; no code MLP | 336 768 | 1 806 842 |

- **The additive code is rank-2 by construction.** The nine codes are c(Δ, α) = u(Δ) + v(α). Each FiLM generator is
  affine, so γ and β are additive in the two axes too: the Δ-modulation is the same at every α and vice versa.
- **The factored arm nests the additive one.** Its scale gains the multiplicative term γ_δγ_α; its shift stays
  additive. A W_δ / W_α layer in front of the linear generators would be redundant, so the generators read ϕ(Δ) and e_α
  directly.
- **Parameters differ only in the request path.** Trunk, heads and adapters have 1 470 074 parameters in every arm.
- **Matched initialisation.** The arm classes run the `CNNHybridPredictor` constructor first (the same RNG draws) and
  then replace the request path. Under `torch.manual_seed(seed)` the trunk, heads and adapters therefore start equal to
  the fused member of the same seed (smoke-checked and tested).

### Training

- **Recipe.** #112 v3 LC exactly as the Gate-B retrain (`rt.implementation()`; only `s99.new_predictor` returns the
  arm's class): the #77 local loss, long-horizon supervision to 225 and the decision-contrastive ranking loss,
  truncation 45, skip-on-nonfinite, AdamW lr 1e-4, wd 1e-4, clip 1, batch 64, 9000 updates (the matched budget),
  PAIRS[step % 9], fold 'all', seeds 20260908/09/10, on the retrain payload of the #109 fit split (hash-checked).
- **Replication guard (smoke).** The training path of the new arms, configured as fused, equals the retrain's
  `train_group` bit for bit after 12 full-cost updates: maximum parameter difference 0, and every logged local loss is
  equal.
- **Usability.** The retrain rule: complete, at most 45 skipped updates, no #117 end-of-training excursion. An unusable
  member is a typed terminal failure: it is not replaced, and its cells drop.
- **J per arm.** Every new predictor gets its own ICLR request policy J: the #101 pe DP at λ = 1e-4 on the #109 policy
  pool, two DAgger rounds and distillation, under `deterministic_scoring()`. The fused arm uses the #101 J. The smoke
  reproduced #101's round-0 labels exactly.

### Evaluation

- **Cells.** The Gate-B mixed cells (grid 311 levels, offset 76, angle 78, power 47) × 3 seeds, from the Gate-B E99
  anchors, tie-free cost, cell AUC `m99.view`. Grid is primary; offset, angle and power are DESCRIPTIVE.
- **Rankers.** The nine fixed requests F-{1, 5, 15}-{continuous, micro, macro} and the arm's J, through
  `rp.rollout_endpoints` (the #101 path).
- **Rollout error.** The #100 `engine_position_mse` of every mixed grid candidate at t ∈ {15, 30, 60, 150, 225} (the
  #112 / #117 endpoints), against its own branch at min(t, last frame). Fixed requests are the same rollout stopped at
  t; J is read at 225 only, because its remaining-time input is defined for the 225 endpoint. The engine truth
  (`s99.shot_labels` at those positions) was built after the freeze.
- **Controls in every ranking table.** The no-model action prior (Gate-B rule), and the wrong-anchor control: each arm's
  J from the 480 fit-split anchors over the grid candidates.
- **MACs.** `cnn_hybrid.linear_macs` (#103 / #114 accounting) per predictor call per request, and per candidate-state
  (one candidate rollout to t = 225, policy calls included for J). This accounting counts the request code per row,
  although the code depends on the request only and could be cached.
- **Coupling diagnostic (no training).** Per block, the effective γ and β at the nine requests (factored: γ = (1 + γ_δ)
  (1 + γ_α) − 1, β = β_δ + β_α), decomposed two-way over (Δ, α) in float64:
  - the interaction share SS_int / SS_total, which is 0 iff the table is additive;
  - the variance over α at fixed Δ (and over Δ at fixed α), relative to the mean squared norm (the identity FiLM is
    γ = β = 0);
  - the rms entry.

  It is computed on the fused Gate-B checkpoints, and on the additive (exactly 0) and factored arms as references.

### Pre-declared dispositions

**D1: "an additive request code matches the fused code within 1.5× per-request rollout error at t = 225"** (the #117
criterion). Per fixed request, the geometric ratio additive / fused of the t = 225 error is computed as follows: the
candidate mean per (level, seed), the mean over the seeds usable in both arms, the log ratio per level, and a level
bootstrap.
- `supported` iff the upper bound is < 1.5 for all nine requests;
- `not_supported_by_this_experiment` iff any lower bound is > 1.5;
- otherwise `readiness_or_precision_insufficient`.

**D2: "axis interaction in the request code carries measurable decision value"**: fused − additive, grid AUC of the
nine-fixed-request mean (each request 1/9), paired by (level, seed), level-clustered. δ rule:
- `supported` iff the mean is ≥ δ = 0.02 and the lower bound is > 0;
- `not_supported_by_this_experiment` iff the upper bound is < δ;
- otherwise `readiness_or_precision_insufficient`.

The token is operational: the rule compares the two trained codes. A `supported` token shows that the trained fused
parameterization outperforms the trained additive one on this metric; it does not by itself identify decision value of
the interaction (phase-B/C review).

The J contrast is reported beside it with its own token and does not enter D2. fused − factored (code-level
interaction beyond modulation-level interaction) and factored − additive (modulation-level interaction) are
DESCRIPTIVE.

A failed guard turns every token (both dispositions and every table) into `readiness_or_precision_insufficient`.

### Guards

| guard | rule |
| --- | --- |
| G1 isolation | payload on fit levels; scored cells on evaluation levels; disjoint rosters |
| G2 scoring policy | every record carries the #111 `SCORING_POLICY` |
| G3 chronology | frozen_at ≤ training ≤ policies ≤ first score; truth built after the freeze, before the first score |
| G4 checkpoint binding | Gate-B, trained-member and policy hashes equal their reports |
| G5 replication | the smoke training-path replication passed, and the fused rows (and J schedules and linear MACs) equal the #101 records on every inventory, cell and seed |
| G6 engine truth | the t = 225 truth equals the #115 store, and the fused fixed micro t = 225 engine errors equal the #115 Gate-B default records |
| G7 nonfinite | the fused J grid nonfinite share is ≤ 0.10 |
