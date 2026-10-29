# Issue #140: U-native (#127 budget-safeguard control): U trained to A4's training GPU-h

Parent: #127 (external baselines program). Ticket: [#140](https://github.com/Sino-Huang/NovPhy/issues/140). The
binding shared rules of #85 and the #127 shared protocol (`scripts/external_baselines.py`, `eb`) apply. The Gate-B
(#108), #101, #104, #109, #112, #123 and #128 records and stores are read-only inputs. Nothing here edits the ICLR 2026
submission.

| | |
| --- | --- |
| Trigger | #127 budget safeguard (the #123 U-B3 rule), fired for A4 by #128: A4 − U `supported` at 2 434.73 GPU-s per seed, 1.88 × U's B1 training compute (1 294.19 GPU-s per seed). Program decision [#127 issuecomment-6479617332](https://github.com/Sino-Huang/NovPhy/issues/127#issuecomment-6479617332) |
| Recipe decision | [Decision P-adapter (2026-10-29): U-native recipe](https://github.com/Sino-Huang/NovPhy/issues/140#issuecomment-6479886189), posted before any U-native training code ran |
| Runner | `scripts/run_issue_140_u_native.py` (modes: gport / prepare / train / readouts / freeze-readouts / score / publish / validate) |
| Reused code | U's frozen #112 v3 retrain (`scripts/run_decision_dynamics_retrain.py` → the v1 `train_group`); Gate-B's scoring (`scripts/run_gate_b.py`: `score_record`, `request_block`); #123's position curve (`rc.score_reference_curves`) |
| Tests | `tests/test_issue_140_u_native.py` (7 tests, CPU) |
| Scheduler spec | `scripts/gpu_window_issue_140.json` (training GPUs 2 / 3; scoring and validation on GPU 1, the orchestrator's change at enqueue) |
| Artifacts | `.local-artifacts/issue-140-u-native-v1/`: `plan.json`, `decisions.md`, `gport/gport.json`, `runs/U-native--seed*/{report,readouts}.json`, `readouts-manifest.json`, `records/`, `records-manifest.json`, `validation.json`, `summary.json`, `compute.json`, `findings.md`, `ledger/`. The checkpoints (`models/`), progress logs (`updates/`) and training-group reports (`training/`) stay local and are hashed in `readouts-manifest.json` |
| Freeze | pre-freeze implementation `3d46e90a`; `plan.json` frozen 2026-10-28T20:47:23Z, commit `1ca96e21` (sha256 `2cab0ce7…`, 104 pins), before any U-native training or evaluation-split read ([freeze](https://github.com/Sino-Huang/NovPhy/issues/140#issuecomment-6480316174)). No post-freeze amendment |
| Readout freeze | `readouts-manifest.json` frozen 2026-10-29T08:20:25Z (19:20 AEDT; sha256 `0aee843a…`), before any evaluation anchor was read |
| Scoring | once, in a process that did not train: scheduler run `140-score#1`, 2026-10-29 19:25:20–19:26:22 AEDT on GPU 1, exit 0 (3 records, 16 s each) |
| Validation | scheduler run `140-validate#1`, 2026-10-29 19:35:20–19:37:30 AEDT on GPU 1, **exit 0**. It hash-checked the 3 records against `records-manifest.json` (sha256 `4dbc9884…`). It re-scored seed 20260908 on every grid cell through Gate-B's own spot-check rule (`harness.endpoint_replication`; max \|Δ\| 0 on `tie_free`, `pig_presence` and `pig_displacement`). It then re-derived the publication twice, byte-identically. `validation.json` sha256 `aec62745…` |

Engine time is zero seconds. The primary metric is the Gate-B pooled grid AUC over the 311 mixed grid cells × 3 seeds
of the #104 sealed #109 evaluation split, tie-free cost, under `deterministic_scoring()`. Intervals are level-clustered
percentile bootstraps (10 000 draws, PCG64 7201) and are DESCRIPTIVE. The margin is δ = 0.02.

## Outcome

**A4 − U-native is not `supported`: it carries the stop token. By the pre-registered reading, A4's wording stays
budget-qualified.** U-native's disposition is 4: per-contrast tokens only.

- Every global guard passes: G-isolation, G-chronology, G-scoring-policy and G-records. G-port passes bit-exactly.
- U-native **fails G-action on seed 20260908**: endpoint action sensitivity 0.0389 against the frozen 0.05. Seeds
  20260909 and 20260910 pass it (0.0567, 0.0658). Every other arm guard passes on every seed: G-rank, G-nonfinite and
  G-trained (all three seeds usable under U's retrain end-state rule).
- By the frozen rule every token involving U-native is `readiness_or_precision_insufficient`. That covers
  U-native − U / − C\* / − P / − J_dec and both directions of A4 − U-native, even though A4's own side (its guards,
  #128's global guards and G-records, the record hashes) passes.
- **Reading (as pre-registered in the #140 body).** "If A4 − U-native is still `supported`, A4's budget qualification is
  dropped from its wording (an architectural reading is permitted); otherwise the wording stays budget-qualified."
  A4 − U-native is not `supported`, so A4's wording stays budget-qualified: *"A4 at its native budget (1.88 × U's
  training GPU-h) ranks better than U at B1"*. No architectural reading of A4 − U is made.
- Descriptively, U-native ranks the grid at 0.8132 [0.7977, 0.8287]; U is at 0.7444 [0.7269, 0.7620] and A4 at
  0.9945 [0.9922, 0.9965]. A4 − U-native = +0.1813 [+0.1658, +0.1967] and U-native − U = +0.0688 [+0.0579, +0.0795].
  Neither carries a token.
- U-native's realised training compute is 2 140.3 GPU-s per seed (mean): 1.654 × U's and **0.879 × A4's**. The recipe
  scaled the updates (16 929 = 1.881 × 9 000). At U's published per-update rate that projected to 2 434 s per seed. The
  measured rate was lower, at 0.126 s per update against U's reports' 0.144 s.

## Design summary

The design is frozen in `plan.json`. The recipe decision was posted on #140 before the step it governs and is in
`decisions.md`.

**Recipe.** U, the pre-registered Gate-B unit F-15-macro (arm LC, family hybrid, fold all, 1 837 690 trainable
parameters), trained by U's own frozen #112 v3 retrain code:

- **Code.** A fresh copy of `rt.implementation()`, so the retrain's own copy is untouched. It runs the v1 `train_group`
  with the v3 `SkippingEnsemble` (patience 10), the truncated recursion (45 transitions), the logging ensemble and the
  `torch.compile`'d vmapped transition.
- **Data.** The retrain's #109 fit payload (`issue-112-v3-retrain-v1/data/training-data.pt`, sha256 checked against
  the Gate-B handoff before training): 220 286 #99 windows, 69 595 long windows, 680 ranking cells.
- **Objective.** #77 local loss + long-horizon supervision (weight 1.0) + decision-contrastive ranking (weight
  0.1 × ramp, 2 cells per step, temperature 0.1, tie-free cost).
- **Optimiser.** AdamW at a constant lr 1e-4 (the recipe has no LR schedule), weight decay 1e-4, per-member gradient
  clip 1.0, batch 64.
- **Request schedule.** The hybrid request of update k is `FIXED_ARMS[k % 9]`.
- **Seeds.** 20260908 / 09 / 10. No selection.

**Budget scaling** (f = 2435.0 / 1294.19 = 1.88149: #128's published A4 mean over eb's U reference):

| constant | U | U-native | rule |
| --- | --- | --- | --- |
| updates | 9 000 | **16 929** | 9 × round(1000 f): whole 9-request cycles, so every request trains 1 881 times |
| long-horizon curriculum `every` | 500 | 941 | round(500 f); horizon reaches 225 at update 10 351 (61.1 % of training; U 5 500, 61.1 %) |
| ranking ramp `start` | 1 000 | 1 881 | round(1000 f) |
| ranking ramp `length` | 1 000 | 1 881 | round(1000 f); full weight from update 3 761 (U 1 999) |
| skip cap (G2, descriptive) | 45 | 85 | round(45 f) |

Logging (`TRACE_EVERY` 900) and checkpointing (`PROGRESS_EVERY` 1 000) are not training schedules and are unchanged. A
stop signal makes `train_group` write its own progress state after the current update, and the process exits 75. No
pause occurred.

**Scoring.** U's own Gate-B path:

- `gb.score_record`, with the U-native checkpoint substituted for the handoff checkpoint: every mixed cell of every
  inventory from its Gate-B anchor, `m99.rollout_endpoints` with the F-15-macro pair, `EndpointCosts` rows against the
  anchor, tie-free primary. Wrong-anchor rows come from the 480 fit anchors.
- Position error uses #123's U curve rule (`rc.score_reference_curves`).
- Wall per decision is timed on the first 20 grid cells.

## G-port

**PASS, bit-exact** (`gport/gport.json`). Both U's B1 retrain code (`rt.implementation().train_group`, the code that
trained the Gate-B checkpoints) and the U-native implementation ran the first 18 updates (two request cycles) of seed
20260908. They ran on the retrain's fit payload, on the CPU, eagerly; `torch.compile` is disabled in both, a kernel
transformation rather than a code-path change. The two runs were bitwise equal:
- parameters, AdamW state, per-update #77 local loss, pre-clip gradient norm and skip counts (max \|Δ\| 0);
- the two `train_group` functions' bytecode.

The schedule constants first differ at update 500. A CPU test shows that the comparison fails on an injected curriculum
defect.

## Results

### Guards

| seed | G-rank (mean tied-pair share; 480 levels) | G-action (≥ 0.05) | G-nonfinite | G-trained (end state) | skipped / G2 | policy grid AUC (fit/policy only) |
| --- | --- | --- | --- | --- | --- | --- |
| 20260908 | pass (0.00002) | **fail (0.0389)** | pass (0.0) | pass (usable; max excursion ratio 1.12) | 0 / pass | 0.7804 |
| 20260909 | pass (0.0) | pass (0.0567) | pass | pass (usable; max ratio 1.20) | 0 / pass | 0.7595 |
| 20260910 | pass (0.0) | pass (0.0658) | pass | pass (usable; max ratio 1.49) | 0 / pass | 0.8382 |

- **Seed 20260909 gradient spikes.** Its update log shows pre-clip gradient-norm spikes above 1 000 in the last 500
  updates. The first is at update index 16 435 (0-based; F-1-micro, 31 732), and the spikes span the Δ = 1 and Δ = 5 requests. The
  #117 end-state rule reads the local-loss ratios, all below 5.0, so it does not flag an excursion. The spikes are
  reported as the retrain reports them.
- **Guards that do not apply.** G-collapse and G-readout (carrier-space model, no readout).
- **Global guards.** G-isolation and G-chronology (plan 2026-10-28T20:47:23Z ≤ readouts 2026-10-29T08:20:25Z ≤
  evaluation anchors 08:25:33Z ≤ first record 08:25:33Z), G-scoring-policy and G-records all pass.

**G-action context (DESCRIPTIVE; changes nothing).** #129's fit-split calibration
([docs/issue-129-slotformer.md](issue-129-slotformer.md), "G-action calibration context") measured this same
`eb.action_sensitivity` call in carrier space:
- engine truth: 0.0395;
- U's Gate-B checkpoints: 0.0240 / 0.0229 / 0.0220.

U-native's endpoints vary with the launch more than U's on every seed: 0.0389 / 0.0567 / 0.0658. By the integrity rule
the frozen threshold, the seed-20260908 failure and the disposition stand.

### Grid rankers (311 mixed cells × 3 seeds)

| ranker | pooled AUC [CI] | per seed (08 / 09 / 10) | top-1 (chance 0.089) |
| --- | --- | --- | --- |
| U | 0.7444 [0.7269, 0.7620] | 0.7141 / 0.8011 / 0.7181 | 0.195 |
| C\* | 0.7834 [0.7676, 0.7992] | 0.8262 / 0.7334 / 0.7905 | 0.192 |
| P | 0.7203 [0.6965, 0.7442] | — | 0.180 |
| J | 0.7057 [0.6846, 0.7267] | 0.6690 / 0.7012 / 0.7470 | 0.160 |
| J_dec | 0.7460 [0.7277, 0.7637] | 0.7205 / 0.7933 / 0.7242 | 0.180 |
| **U-native** | 0.8132 [0.7977, 0.8287] | 0.7928 / 0.8024 / 0.8444 | 0.315 |
| A4 (#128, published records) | 0.9945 [0.9922, 0.9965] | 0.9946 / 0.9944 / 0.9945 | 0.939 |

No ranker has tied pairs or nonfinite rows except P (tied-pair share 0.002).

### Contrasts (grid, paired per (level, seed))

| contrast | estimate [CI] | token | reverse token |
| --- | --- | --- | --- |
| **A4 − U-native** (the control's purpose) | +0.1813 [+0.1658, +0.1967] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| U-native − U | +0.0688 [+0.0579, +0.0795] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| U-native − C\* | +0.0298 [+0.0197, +0.0395] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| U-native − P | +0.0929 [+0.0711, +0.1145] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| U-native − J_dec | +0.0672 [+0.0553, +0.0790] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |

- **A4 − U-native guards.** The guard requires U-native's global and arm guards and A4's side together. U-native's side
  fails (G-action, seed 20260908). A4's side passes: its arm guards and #128's global guards pass, #128's G-records binds
  its current records, and the three A4 records match #128's `records-manifest.json` (sha256 `29af62ae…`; #128
  `summary.json` sha256 `c9464ba0…`).
- **No token is licensed.** Every token is the stop token because U-native fails G-action. The estimates are reported as
  measured; they license no statement in either direction.
- **U-native − U as a descriptive block.** `extra.u_native` repeats U-native − U without a token, as "what the extra
  budget buys": +0.0688 [+0.0579, +0.0795].

### Wrong-anchor control (right − wrong, 480 fit anchors, grid)

| arm | right | wrong | right − wrong | ranking stability (Spearman) | token |
| --- | --- | --- | --- | --- | --- |
| U-native | 0.8132 [0.7977, 0.8287] | 0.6112 [0.5957, 0.6269] | +0.2020 [+0.1886, +0.2152] | 0.5516 | `readiness_or_precision_insufficient` |

### Descriptive inventories (no tokens)

| inventory (cells) | U | C\* | P | J | J_dec | U-native | A4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| offset (228) | 0.6670 | 0.7748 | 0.5000 | 0.7522 | 0.6904 | 0.8327 | 0.9811 |
| angle (234) | 0.6791 | 0.7590 | 0.8119 | 0.7765 | 0.6988 | 0.8237 | 0.9972 |
| power (141) | 0.8058 | 0.8068 | 0.5000 | 0.8076 | 0.7943 | 0.7740 | 0.8468 |

The other inventories' contrasts all carry stop tokens:

| inventory | A4 − U-native | U-native − U |
| --- | --- | --- |
| offset | +0.1484 [+0.1208, +0.1727] | +0.1658 [+0.1412, +0.1912] |
| angle | +0.1734 [+0.1581, +0.1888] | +0.1446 [+0.1225, +0.1677] |
| power | +0.0728 [+0.0381, +0.1108] | −0.0318 [−0.0599, −0.0060] |

### Engine position error (#100 metric; DESCRIPTIVE)

Mean slot-centre MSE of the predicted carrier against the engine centres on present slots, every complete grid
candidate of every mixed grid cell, from the Gate-B anchors (member-clustered bootstrap; U = the #123 curves):

| t | 15 | 30 | 60 | 150 | 225 |
| --- | --- | --- | --- | --- | --- |
| U-native | 0.0510 | 0.0519 | 0.0523 | 0.0514 | 0.0528 [0.0487, 0.0569] |
| U | 0.0505 | 0.0501 | 0.0484 | 0.0459 | 0.0474 [0.0440, 0.0508] |

## Disposition (δ rule)

- Disposition 1 (U-native ranks launches better than U, scoped): **no**. U-native fails G-action on seed 20260908, so
  U-native − U carries the stop token.
- Disposition 2: **no**. Disposition 3: **no**. Both require every guard to pass.
- **Disposition 4: per-contrast tokens only**, every one `readiness_or_precision_insufficient`.
- U-native is a control and is not in the #127 program set; it neither enters nor vetoes the program claim.
- Budget safeguard: does not fire for U-native (U-native − U is not `supported`).
- **A4 − U-native: not `supported`. A4's #128 wording stays budget-qualified** ("A4 at its native budget (1.88 × U's
  training GPU-h) ranks better than U at B1"). This is reported beside #128's table.

## Deviations and disclosures

1. **Realised budget below A4's.** The pre-registered rule scaled the update count (and the step-indexed schedules),
   not the seconds. The 16 929 updates measured 2 189.4 / 2 203.1 / 2 028.4 GPU-s against the 2 434 s per seed
   projected from U's reports (0.144 s per update). That is 0.899 / 0.905 / 0.833 × A4's 2 434.73 (mean 0.879), and
   1.692 / 1.702 / 1.567 × U's 1 294.19 (mean 1.654). Each seed trained alone on an RTX 5090 (`shares_gpu` false) at
   about 0.126 s per update. Whether U's retrain shared its GPUs is not recorded in its reports [INFERENCE: the
   difference is throughput, not work].
2. **Scoring device.** At enqueue the orchestrator moved `140-score` and `140-validate` from GPU 0 to GPU 1, same device
   class (RTX 5090), depending only on `140-freeze-readouts`, because #137's GPU-0 scoring had about 10 h left.
3. **G-port on the CPU, eager.** GPU kernels (compiled, cuDNN autotune) are not bitwise deterministic from run to run;
   the G-port fixes the code path, not the kernels.
4. **No selection, no tuning.** Only the budget changes against U. Nothing was re-run or re-selected after any score.
5. **No P-crash, no pause, no amendment.** Every scheduler run exited 0 on its first launch.

## Compute

GPU seconds are each process's wall time on its GPU (`eb.mark`, each seed's training counted once from its
`report.json`). Every job ran alone on its GPU.

| item | GPU-s | GPU-h |
| --- | --- | --- |
| training seeds 20260908 / 09 / 10 (`train_group` GPU-s) | 2 189.4 / 2 203.1 / 2 028.4 | 1.784 |
| readouts (3 seeds) | 13.9 / 12.2 / 12.3 | 0.011 |
| scoring (3 records) | 16.5 / 16.0 / 15.9 | 0.013 |
| **recorded** (`compute.json`) | **6 507.8** | **1.808** |
| unrecorded: validate (GPU 1, 131 s run); publish and freeze ran on the CPU | 130.6 | 0.036 |
| **active total** | **≈ 6 638** | **≈ 1.84** (plan projection: 2.03 training + about 0.2) |

| | U-native | U | A4 (#128) |
| --- | --- | --- | --- |
| trainable parameters | 1 837 690 | 1 837 690 | 12 751 957 |
| training GPU-s per seed (08 / 09 / 10) | 2 189.4 / 2 203.1 / 2 028.4 | 1 290.3 / 1 299.9 / 1 292.4 | 2 424.8 / 2 429.2 / 2 450.2 |
| updates per seed | 16 929 | 9 000 | — |
| inference MACs per candidate (to frame 225) | 20.99 M | 20.99 M | 805.39 M |
| wall per decision (16 candidates, GPU, median) | 0.012 s | — | 0.004 s |

Active wall time with a #140 GPU process running was 1.6 h:
- training 18:00:20–19:10:59 AEDT on 2026-10-29 (GPU 2: seeds 20260908 then 20260910; GPU 3: seed 20260909, ended
  18:37:24);
- readouts 19:15:20–19:16:05;
- scoring 19:25:20–19:26:22;
- validation 19:35:20–19:37:30.

Engine seconds: 0. No scope was cut.

## Reproduction

```
conda activate novphy && source env.sh
M="python -u scripts/run_issue_140_u_native.py"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 $M --gport     # gport/gport.json (CPU, bit-exact, 18 updates)
$M --prepare                                             # plan.json (after the freeze: re-verifies the pins)
# scheduler: scripts/gpu_window_issue_140.json (night window only)
CUDA_VISIBLE_DEVICES=2 $M --train --gpu 2                # seeds 20260908, 20260910
CUDA_VISIBLE_DEVICES=3 $M --train --gpu 3                # seed 20260909
CUDA_VISIBLE_DEVICES=3 $M --readouts --gpu 3 && $M --freeze-readouts
CUDA_VISIBLE_DEVICES=1 $M --score                        # once
$M --publish
CUDA_VISIBLE_DEVICES=1 $M --validate                     # exit 0
CUDA_VISIBLE_DEVICES="" pytest tests/test_issue_140_u_native.py   # 7 passed
```
