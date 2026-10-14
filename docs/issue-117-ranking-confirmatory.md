# Issue #117: joint vs specialist per-request ranking on the #104 sealed split (confirmatory)

Parent: #108 (pass branch). The shared rules of #85 apply, and the #111 scoring contract is adopted. The #104, #109,
#112, Gate-B, #101 and #117 records are read-only inputs. Nothing here edits the ICLR 2026 submission.

| | |
| --- | --- |
| Runner | `scripts/run_request_interference_ranking.py` (training code: `run_request_interference.train_unit`, unchanged) |
| Artifacts | `.local-artifacts/issue-117-ranking-confirmatory-v1/`: `findings.md` has every table; also `compute.json`, `summary.json`, `records-manifest.json` |
| Freeze | Plan frozen 2026-10-14T16:12:54Z, commit c8bd1bcb, after a passed smoke and before any training update or score |
| Validation | `python -u -m scripts.run_request_interference_ranking --validate` (exit 0); `pytest tests/test_issue_117_ranking_confirmatory.py` |

Engine seconds: 0. Intervals are level-clustered percentile bootstraps (10 000 draws, PCG64 7201). They are
DESCRIPTIVE. Margin: δ = 0.02.

## Question and disposition

#117 measured per-request ranking EXPLORATORY (LOLO plus held-out 007/008/014/016). Its trigger T2 read "the
request-mean LOLO grid AUC joint − specialist, or that of F-15-macro, has upper bound < 0". After Gate B passed, the
same contrast is confirmatory on the sealed #109 evaluation split.

**Claim:** sharing all weights does not cost per-request ranking on the sealed split. The test is the grid AUC joint −
specialist, request mean and F-15-macro, under LC3 at the per-request budget.

**Rule (frozen):**
- `supported` iff both contrasts have lower bound > −δ (non-inferiority at δ);
- `not_supported_by_this_experiment` iff either has upper bound < −δ;
- otherwise `readiness_or_precision_insufficient`;
- a failed guard turns every token into `readiness_or_precision_insufficient`.

| grid, LC3, per-request budget | J − S | token | T2 condition (upper < 0) |
| --- | --- | --- | --- |
| request mean (9 requests) | +0.0848 [+0.0742, +0.0952] | supported | no |
| F-15-macro (Gate-B unit) | +0.0561 [+0.0435, +0.0682] | supported | no |

**Disposition: `supported`.** All seven guards pass. At the joint's own per-request update count, the joint is
non-inferior to its specialist at δ on the sealed split. The same holds on every one of the nine requests. Each
estimate is at least δ and each lower bound is above 0, which also meets the δ superiority rule (DESCRIPTIVE, not a
frozen token). This agrees with the #117 LOLO reading (+0.006 [−0.050, +0.062]; F-15-macro +0.076), now with
intervals about 5× narrower.

## Design

- **Joint (J).** The Gate-B handoff LC hybrid (the #112 v3 retrain on the #109 fit split), 3 seeds, scored at each
  of its nine requests. Guard G5 checks that its rows equal two published sets exactly (16 800 blocks):
  - the Gate-B unit records: F-15-macro, every inventory and the 480 wrong-anchor blocks;
  - the #101 fixed-request records.
- **Specialists.**
  - **Arms:** S-\* (nine single-request models) and D1/D5/D15 (per-Δ, shared over the descriptions). All are
    `CNNHybridPredictor` with 1 837 690 parameters.
  - **Training:** the #117 arm construction, under LC3 (the Gate-B recipe), on the retrain payload
    `issue-112-v3-retrain-v1/data/training-data.pt` (hash-checked), fold 'all', the three #112 seeds, one member
    per group.
  - **Smoke equivalence:** for the joint arm, the smoke shows this path equals the retrain's `train_group`, the path
    the Gate-B joint was trained on. It is bit for bit: parameter delta 0, and the logged local losses and pre-clip
    norms are identical, both at the real schedule and at full cost.
- **Budgets.**
  - **Per-request (primary):** exactly the joint run's updates of the arm's requests, each at its joint step: 1000
    updates for S, 3000 for D.
  - **Full (DESCRIPTIVE):** 9000 updates. The smoke projected 4.5 h elapsed on 4 GPUs, so the full budget was
    included.
- **Usability.** The retrain rule: complete, at most 45 skipped updates, and no #117 end-of-training excursion
  (`rt.excursion` over the member's own schedule). An unusable member is a typed terminal failure and is not
  replaced. The joint uses the handoff usability.
- **Evaluation.**
  - Cells: the Gate-B mixed cells (grid 311 levels, offset 76, angle 78, power 47) and the Gate-B E99 anchors.
  - Cost: tie-free.
  - Pairing: cells are (level, seed), and joint seed s pairs with specialist seed s. A cell enters a contrast only
    when every piece of it is defined.
  - Inventories: grid is primary; the others are DESCRIPTIVE.
- **Controls.** The no-model action prior (Gate-B rule), and the wrong-anchor control for every scored (unit,
  request): the 16 grid candidates from the 480 fit-split anchors.

## Results (grid)

### Per request, per-request budget

| request | J | S | J − S | D | J − D (token) |
| --- | --- | --- | --- | --- | --- |
| 1-continuous | 0.677 | 0.592 | +0.085 [+0.069, +0.102] | 0.503 | +0.165 (supported) |
| 1-micro | 0.650 | 0.525 | +0.125 [+0.099, +0.152] | 0.577 | +0.064 (supported) |
| 1-macro | 0.666 | 0.565 | +0.101 [+0.076, +0.125] | 0.583 | +0.090 (supported) |
| 5-continuous | 0.749 | 0.681 | +0.067 [+0.056, +0.078] | 0.738 | +0.010 (supported) |
| 5-micro | 0.745 | 0.648 | +0.097 [+0.085, +0.109] | 0.738 | +0.007 (supported) |
| 5-macro | 0.743 | 0.676 | +0.067 [+0.050, +0.083] | 0.752 | −0.009 [−0.024, +0.004] (insufficient) |
| 15-continuous | 0.762 | 0.680 | +0.082 [+0.073, +0.090] | 0.755 | +0.008 (supported) |
| 15-micro | 0.760 | 0.678 | +0.082 [+0.074, +0.091] | 0.751 | +0.009 (supported) |
| 15-macro | 0.744 | 0.688 | +0.056 [+0.044, +0.068] | 0.758 | −0.013 [−0.023, −0.004] (insufficient) |
| **mean** | **0.722** | **0.637** | **+0.085 [+0.074, +0.095]** | **0.684** | **+0.032 [+0.022, +0.043] (supported)** |

J − S is `supported` on every request. J − D (secondary) is `supported` for the request mean and for 7 of 9 requests.
At F-5-macro and F-15-macro the per-Δ model ranks above the joint by less than δ. Neither difference is excluded from
reaching −δ: lower bounds −0.024 and −0.023, so both are `readiness_or_precision_insufficient`. The F-15-macro upper
bound is below 0.

### Item 2: asymmetry, per-request budget

| group | J − S | J − D | D − S |
| --- | --- | --- | --- |
| Δ = 1 | +0.104 [+0.089, +0.119] | +0.106 [+0.091, +0.122] | +0.005 [−0.011, +0.020] |
| Δ = 5 | +0.077 [+0.064, +0.090] | +0.003 [−0.011, +0.016] | +0.074 [+0.062, +0.086] |
| Δ = 15 | +0.074 [+0.064, +0.083] | +0.001 [−0.008, +0.011] | +0.072 [+0.061, +0.084] |
| Δ=1 − Δ=15 | +0.030 [+0.016, +0.045] | +0.108 [+0.092, +0.125] | −0.085 [−0.106, −0.065] |
| continuous | +0.078 [+0.068, +0.088] | +0.049 [+0.034, +0.065] | +0.040 [+0.025, +0.055] |
| micro | +0.102 [+0.091, +0.112] | +0.017 [+0.005, +0.028] | +0.093 [+0.080, +0.106] |
| macro | +0.075 [+0.058, +0.090] | +0.031 [+0.020, +0.042] | +0.068 [+0.057, +0.079] |
| continuous − macro | +0.004 [−0.007, +0.014] | +0.018 [+0.005, +0.032] | −0.028 [−0.041, −0.015] |

- **Δ = 1 does not lose more ranking than Δ = 15 from sharing; it gains more.** J − S is +0.030 larger at Δ = 1.
- The gains come from two different kinds of sharing:
  - Δ = 1 gains from sharing across Δ: J − D +0.106, while D1 − S-1 is ≈ 0.
  - Δ = 5 and 15 gain from sharing across descriptions within Δ: D − S ≈ +0.07, while J − D ≈ 0.
- By description, the J − S differences are within 0.03 of each other. Continuous − macro is +0.004.

### Full budget (DESCRIPTIVE)

| | J − S | J − D |
| --- | --- | --- |
| request mean | +0.015 [−0.001, +0.030] | −0.052 [−0.060, −0.043] |
| Δ = 1 | +0.140 [+0.120, +0.161] | +0.071 [+0.057, +0.085] |
| Δ = 5 | −0.029 [−0.044, −0.014] | −0.122 [−0.136, −0.109] |
| Δ = 15 | −0.105 [−0.122, −0.088] | −0.103 [−0.120, −0.086] |

The full budget gives each specialist 9× its request's updates, against the joint's 1000. This contrast is update
allocation, not interference at matched updates.

- **Δ = 5 and Δ = 15.** The full-budget specialists rank above the joint on five of these six requests, by
  0.054–0.141. The exception is S-5-micro (J +0.088). The per-Δ models rank above the joint on all six, by
  0.098–0.127. Every one of these upper bounds is < −δ.
- **Δ = 1.** The joint keeps a large lead at Δ = 1.
- **Request mean.** J − S passes the δ non-inferiority rule (+0.015, lower bound −0.001) only because the Δ = 1 gains
  offset the Δ = 5/15 losses.

This reading belongs to #116 (request-set scaling) and to any future recipe decision. It is not part of this
disposition, which the plan fixes at the per-request budget.

### Controls

| request mean (grid) | AUC | − prior (δ superiority token) | right − wrong anchor | stability |
| --- | --- | --- | --- | --- |
| J | 0.722 [0.703, 0.741] | +0.002 [−0.025, +0.027] insufficient | +0.082 [+0.075, +0.088] | 0.75 |
| S, per-request | 0.637 [0.618, 0.655] | −0.083 [−0.113, −0.053] not supported | +0.067 [+0.058, +0.076] | 0.57 |
| D, per-request | 0.684 [0.668, 0.700] | −0.036 [−0.063, −0.009] not supported | +0.118 [+0.108, +0.128] | 0.30 |
| S, full | 0.722 [0.713, 0.730] | +0.001 [−0.021, +0.023] insufficient | +0.153 [+0.144, +0.163] | 0.30 |
| D, full | 0.773 [0.760, 0.787] | +0.053 [+0.029, +0.077] supported | +0.181 [+0.172, +0.191] | 0.53 |

No-model action prior: 0.7203 [0.6965, 0.7442], the same value as Gate B.

- **The joint's request mean does not clear the prior by δ.** Per request, J − prior is negative at Δ = 1 (−0.04 to
  −0.07) and +0.02 to +0.04 at Δ = 5/15.
- **The joint's Δ = 1 requests barely use the anchor.** Their right − wrong is +0.001 to +0.002, against +0.10 to
  +0.15 at Δ = 5/15. The per-request specialists are below the prior. The joint's advantage over them is therefore an
  advantage over a weak ranker; it does not show anchor-dependent ranking beyond the action marginal.

## Training

- 72 members (12 arms × 2 budgets × 3 seeds): 0 retired, 0 over the skip cap.
- Two members are typed terminal failures (`final_checkpoint_excursion`). Both are Δ = 1 arms trained without the
  other requests:
  - D1/request seed 20260910: F-1-continuous local loss 9.7× its reference;
  - S-1-macro/full seed 20260908: F-1-macro 13.2×.
- These drop their seed's cells from the contrasts that read them:
  - D/request at Δ = 1: 2 seeds;
  - S/full at F-1-macro: 2 seeds.
- S-1-continuous/full seed 20260909 had a 2.4e11 pre-clip spike, but its end-state ratio of 4.41 is under the
  frozen 5× threshold, so it is usable. The joint's three handoff seeds show no excursion. This matches the #117 post
  hoc pattern: excursions concentrate in Δ = 1 arms trained alone.

## Guards

All seven guards pass:
- G1 isolation;
- G2 scoring policy on all 75 records;
- G3 chronology (freeze ≤ every training ≤ first score);
- G4 checkpoint and payload binding;
- G5 exact replication of the Gate-B unit and #101 rows (16 800 blocks; the joint's F-15-macro grid AUC is 0.7444,
  as in Gate B);
- G6 schedule integrity;
- G7 joint nonfinite share.

`--validate` does the following:
- re-derives `compute.json`, `summary.json`, `findings.md` and `records-manifest.json` (sha256 of all 75 records and
  72 update logs) and byte-compares them;
- re-scores S-15-macro, D15 and S-1-continuous (per-request budget, first seed) on every grid cell, with exact rows.

It took 89 s.

## Compute

- **GPU seconds:** 54 899 in total:
  - training 45 845 (4 × RTX 5090, one process per GPU, frozen LPT assignment);
  - scoring 8 869;
  - the rest: smoke and setup.
- **Elapsed:** about 3.9 h from freeze to publication (training 3.2 h, scoring 0.7 h).
- **Engine seconds:** 0.

Each stage was launched once, and the PIDs are in `logs/train-pids.txt` and `logs/score-pids.txt`. Checkpoints
(506 MB) and records (595 MB) stay local. The plan, smoke, ledger, training reports, logs, tables and the records
manifest are committed.

## Disclosures

- **Hardware.** RTX 5090, not the #85 RTX 3090.
- **Known before the freeze.**
  - Gate B: unit 0.7444, C* 0.7834, prior 0.7203.
  - The joint's nine fixed-request rows on this split, published by #101 and #103. Their per-request AUCs were
    therefore known, while no specialist or per-Δ output on this split existed.
  - Every #117 LOLO and held-out result.
- **First smoke run.** It failed one assertion: an exact-zero check on a stand-in contrast built from 1/9-weighted
  sums. That check was changed to a 1e-12 tolerance, and the rerun passed before the freeze. Neither smoke produced a
  new model output on the evaluation split. Only the joint's F-15-macro was rescored, to reproduce the Gate-B rows.
- **Training PID record.** The first lines of `logs/train-pids.txt` are the launching shells' PIDs. The python
  worker PIDs were appended while the workers ran. Nothing was relaunched.
- **Scope.**
  - LC3 only; the #77 recipe B is not trained on the #109 fit split.
  - NovPhy only. Item 5 (CLEVRER, Physion-Dominoes) is deferred to section (b).
  - Item 4 (mitigation probe) is not run, because item 3 found no conflict and nothing here calls for it.
  - The Gate-B recipe is unchanged.
