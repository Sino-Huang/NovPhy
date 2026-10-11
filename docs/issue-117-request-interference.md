# Issue #117 — interference between requests in the shared predictor (items 1–3, NovPhy)

Parent: #108. Shared rules of #85 apply; #111 scoring contract adopted. #112 records are read-only inputs.

| | |
| --- | --- |
| Runner | `scripts/run_request_interference.py`, `world_model/training/request_interference.py` |
| Artifacts | `.local-artifacts/issue-117-request-interference-v1/` (`findings.md` has every table; `figures/`) |
| Freeze | plan frozen 2026-10-11T03:48:41Z, commit 56a54813, before any training update and any score |
| Validation | `python -u -m scripts.run_request_interference --validate` (exit 0); `pytest tests/test_issue_117_request_interference.py` |

Zero engine seconds. Intervals: member-clustered percentile bootstraps (10 000 draws, PCG64 7201). LOLO results are
EXPLORATORY; held-out results are DESCRIPTIVE.

## Question and decision rule

Before retraining the #112 v3 recipe on the #104 fit split: do the nine requests that share every weight interfere?
If yes, build a v4 with a per-request mitigation; otherwise keep v3 (#112 closing comment).

The frozen v4 trigger reads the v3 recipe (LC3) at the per-request budget, the negative-transfer contrast: the joint
model and a single-request specialist make the same updates of the request, with and without the other eight requests'
updates. It fires on any of:

- **T1** some request's joint / specialist rollout error at t = 225 has lower bound > 1.5;
- **T2** the request-mean LOLO grid AUC joint − specialist, or that of the Gate-B unit F-15-macro, has upper bound < 0;
- **T3** a joint member retires or skips > 45 updates while no LC3 specialist or per-Δ member retires on that request.

## Design

- Encoder E99; the #112 v3 training data, anchors and contexts; 12 fit lineages.
- Recipes:
  - **B**: the #77 recipe;
  - **LC1**: #112 v1 LC, full back-propagation, retire on nonfinite;
  - **LC3**: #112 v3 LC, truncation every 45 transitions, skip-on-nonfinite (the Gate-B recipe).
- Arms, all `CNNHybridPredictor` with 1 837 690 parameters:
  - **J**: joint, nine requests;
  - **D1, D5, D15**: per-Δ, three requests each;
  - **S-\***: nine single-request specialists.
- Budgets:
  - **request**: exactly the joint run's updates of the arm's requests (1000 per request), each at its joint step, so
    the curriculum and ramp match;
  - **full**: 9000 updates.
- 75 units × 13 folds × 3 seeds = 2925 members.
- The smoke shows the J arm is bit-identical (parameter delta 0) to #112 v3 B, v1 LC and v3 LC over 12 updates, with
  probes running.
- Item 3 probes run every 250 steps:
  - every arm: per-request validation loss on the left-out lineage;
  - joint arm: per-request pre-clip gradients on the shared trunk at a fixed training batch (Gram → cosines);
  - every update: loss and pre-clip norm, so every nonfinite update is attributed to its request.

## Results (LOLO, 11 lineages)

### Item 1. Joint / specialist rollout error at t = 225 (geometric mean, request-mean over the nine requests)

| recipe | per-request budget | full budget | largest per-request upper bound (per-request / full) |
| --- | --- | --- | --- |
| B | 0.08 [0.08, 0.10] | 0.62 [0.52, 0.75] | 0.59 / 1.55 |
| LC1 | 0.07 [0.05, 0.10] | n/a (Δ = 1 specialists retired) | 0.89 / — |
| LC3 | 0.40 [0.33, 0.48] | 0.41 [0.33, 0.50] | 0.82 / 1.68 |

- **1.5× disposition** (sharing costs no more than 1.5× per request): **supported** for B, LC1 and LC3 at the
  per-request budget, every request.
- At the full budget the token is `readiness_or_precision_insufficient` for every recipe. Some intervals straddle
  1.5: LC3 F-5-continuous 1.40 [1.12, 1.68] and F-5-macro 1.28 [1.05, 1.55]. LC1 has no Δ = 1 specialist.
- The joint model has lower long-horizon error than every specialist and per-Δ model at the matched per-request
  budget. At the full budget (9× the specialist's updates) its per-request point estimates are ≤ 1.40 and the upper
  bounds ≤ 1.68.

### Ranking (grid AUC, tie-free; joint − specialist, request-mean)

| recipe | per-request budget | full budget | F-15-macro (per-request) |
| --- | --- | --- | --- |
| B | +0.000 [−0.034, +0.038] | +0.012 [−0.021, +0.048] | −0.076 [−0.141, −0.013] |
| LC1 | +0.020 [−0.017, +0.057] | n/a | +0.030 [−0.025, +0.081] |
| LC3 | +0.006 [−0.050, +0.062] | −0.008 [−0.078, +0.058] | **+0.076 [+0.028, +0.124]** |

Sharing does not cost ranking. For the Gate-B unit under LC3, the joint ranks better than the specialist.

### Item 2. Asymmetry (joint / specialist at t = 225, per-request budget)

| recipe | Δ = 1 | Δ = 5 | Δ = 15 | Δ1 / Δ15 ratio of ratios |
| --- | --- | --- | --- | --- |
| B | 0.37 | 0.005 | 0.35 | 1.05 [0.83, 1.33] |
| LC1 | 0.001 | 0.44 | 0.59 | 0.002 |
| LC3 | 0.21 | 0.48 | 0.62 | 0.34 [0.19, 0.62] |

- Δ = 1 does not degrade more than Δ = 15. Under the long-horizon recipes it gains the most from sharing.
- Under B the per-request-budget Δ = 5 specialist is far from converged: 1000 updates, against the joint's 0.26.
- By description, the ratios differ by less than 2× in every recipe (`findings.md`).

### Item 3. Training dynamics

- **Gradient conflict is low.** Mean pairwise trunk cosines over training: B 0.84–0.93, LC3 0.56–0.82, LC1 0.50–0.81.
  The smallest mean cosine of any request pair is 0.39 (LC3). Negative cosines are 0.1 % of probes under B and 2–7 %
  under LC3, rising late in training. Under LC1 and LC3, same-Δ pairs align more than same-description pairs
  (0.81–0.82 vs 0.56–0.63). Under B the order is reversed (0.87 vs 0.93).
- **Retirements by request** (first nonfinite update; every one had a finite loss and a nonfinite gradient norm):

  | unit | retired |
  | --- | --- |
  | LC1 J | 5/39 (F-1-continuous 2, F-1-micro 2, F-1-macro 1) |
  | LC1 D1 per-request | 38/39 |
  | LC1 D1 full | 39/39 |
  | LC1 S-1-\* full | 116/117 |
  | LC1 S-1-\* per-request | 0/117 |
  | LC3, every unit | 0 (one skipped update, D1 full) |
  | B, every unit | 0 |

  Every retirement is on a Δ = 1 update. The D1 per-request model makes exactly the joint's 3000 Δ = 1 updates and
  retires 38/39, against 5/39 for the joint: interleaving Δ = 5 and Δ = 15 updates stabilizes Δ = 1. The #112 v1
  attribution reproduces: 28/28 retirements on Δ = 1.
- **Validation curves** (`figures/validation-*.png`): under LC1 and LC3, the joint's left-out long-horizon loss at
  Δ = 1 settles about 10× below every Δ = 1 specialist and per-Δ model. At Δ = 5 and 15 it tracks them.

### Decision

T1, T2 and T3 do not fire. **No interference problem: v3 stays the Gate-B recipe; no v4.**

## Post hoc: a residual stability risk (not interference)

This was added after every outcome. No token or trigger reads it.

LC3 joint fold-'all' seed 20260908 ended training in a gradient excursion:

- A Δ = 1 update spiked to 1.2e4 at update 8955. At updates 8973–8975, F-1-\* pre-clip norms reached 3e7–9e9; until
  then they were 1–40. The spikes spread to Δ = 5 (up to 6e9) and Δ = 15 (up to 2e4) updates.
- The local loss of every request rose to 0.3–0.4, from 0.003–0.12. The last update, which is the checkpoint, landed
  inside the excursion.
- Its held-out rollout error at 225 is 0.5–1.4, against 0.02–0.06 for seeds 20260909 and 20260910.
- This one member drives the descriptive held-out joint / specialist ratio of 1.85 [1.42, 2.31] for LC3
  per-request. With seeds 20260909 and 20260910 alone, the joint's held-out error is below the specialist's on all
  nine requests: 0.02–0.05 against 0.07–1.41.

This is not interference. End-of-training excursions are rarer in the joint arm than in any arm that trains Δ = 1
without the other requests:

| LC3 arm | members ending in an excursion |
| --- | --- |
| J | 1/39 |
| D1, per-request / full | 4/39, 4/39 |
| S-1-\*, per-request / full | 6/117, 18/117 |
| S-5-\*, full | 13/117 |
| D15 and S-15-\* | 0 |

In the joint arm an excursion that starts on Δ = 1 reaches the other requests through the shared weights. The excursion
itself comes from long recursive back-propagation, which v3 truncates at 45 transitions. It stays finite, so
skip-on-nonfinite never fires.

The #112 v3 Gate-B checkpoints from the RTX 3090 run show no excursion: their local loss at update 9000 is in line with
their trace. A retrain on the #104 fit split carries this risk per seed. Guarding the final checkpoint is a
stability-only change, a separate decision outside the interference question.

## Disclosures

- Hardware: 4× RTX 5090, one process per GPU, against #85's single RTX 3090. The joint arm was retrained here, not
  reused. Its LOLO numbers reproduce #112 v3:
  - LC3 F-15-macro grid AUC 0.725 here vs 0.715;
  - rollout error at 225, 0.040 vs 0.040;
  - B F-1-continuous rollout error at 225, 3841 vs 3622.
- Post-freeze, after every record: the end-of-training excursion table and the per-seed held-out table of joint
  fold-'all' members were added to `compute.json` and `findings.md`. No token, ratio or trigger changed.
- Held-out lineages 007/008/014/016 are exposed (the #112 v1 look). They were scored once, after every LOLO record.
- Scope: NovPhy only. Items 4 (mitigation probe) and 5 (CLEVRER, Physion-Dominoes) are not run here.

Compute: 80 179 GPU s (training 78 373 s; LOLO scoring about 1 340 s over 4 GPUs; held-out about 470 s); engine
seconds 0. Checkpoints (21 GB) and per-update logs (355 MB) stay local. Plan, smoke, ledger, records, training reports,
tables and figures are tracked.
