## #128 result: direct success predictors vs U (published and validated)

**Disposition 1 for A1, A2, A2-native and A4: "X ranks launches better than U on the sealed NovPhy grid (scoped)". A4 is budget-qualified. A3-sym and A3-img: disposition 4 (native-task fidelity not reproduced; tokens only).**

Published and validated, `--validate` exit 0, in two passes:
- primary group (A1, A2, A2-native, A4): 29 Oct ≈ 03:10 AEDT, commit `0597bc46`;
- A3 group: 29 Oct 18:55 AEDT, commit `4c604b32`.

`validation.json` binds `records-manifest.json` sha256 `29af62ae…` (39 records). A2 seed 20260908 was re-scored with max |Δ| 0. Doc: `docs/issue-128-direct-predictors.md`.

### Grid (311 mixed cells × 3 seeds; Gate-B pooled AUC; level-clustered bootstrap, descriptive; δ = 0.02)

| arm | params | guards | grid AUC [CI] | X − U [CI] | token X − U | token U − X | disposition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 DQN-px | 11 309 889 | pass | 0.9946 [0.9917, 0.9971] | +0.2502 [+0.2327, +0.2675] | `supported` | `not_supported_by_this_experiment` | 1 |
| A2 DQN-car (matched to U) | 1 838 781 | pass | 0.9757 [0.9692, 0.9815] | +0.2313 [+0.2131, +0.2493] | `supported` | `not_supported_by_this_experiment` | 1 |
| A2-native | 11 333 771 | pass | 0.9740 [0.9666, 0.9805] | +0.2296 [+0.2114, +0.2474] | `supported` | `not_supported_by_this_experiment` | 1 |
| A4 DynDQN | 12 751 957 | pass | 0.9945 [0.9922, 0.9965] | +0.2501 [+0.2327, +0.2674] | `supported` | `not_supported_by_this_experiment` | 1 (budget-qualified) |
| A3-img | 11 633 920 | fail (G-fid) | 0.9870 [0.9814, 0.9918] | +0.2426 | stop token | stop token | 4 |
| A3-sym | 11 662 208 | fail (G-fid) | 0.9700 [0.9597, 0.9791] | +0.2256 | stop token | stop token | 4 |

Reference rankers (grid AUC): U 0.7444, C* 0.7834, P 0.7203, J_dec 0.7460.

**Secondary contrasts.** X − C\*, X − P and X − J_dec are `supported` for A1, A2, A2-native and A4. For A1 these are +0.2113, +0.2743 and +0.2487.

**Within family.**
- A2 − A1 = −0.0190 [−0.0251, −0.0134]: A2 − A1 is `not_supported_by_this_experiment`; A1 − A2 is the stop token.
- A4 − A1 = −0.0002 [−0.0021, +0.0021]: not `supported` in either direction, so there is no difference of 0.02 or more from the trajectory supervision.

**Guards and fidelity.** Every global guard passes (G-isolation, G-chronology, G-scoring-policy, G-records). G-port is bit-exact for every arm. G-rank, G-nonfinite and G-trained pass on every seed.

| arm | G-fid | result | frozen band | outcome |
| --- | --- | --- | --- | --- |
| A1 | Tier E | AUCCESS 77.63, shipped checkpoints, 10 folds | [75.4, 79.8] | pass |
| A2, A2-native | inherited from A1 | — | — | pass |
| A4 | Tier T | AUCCESS 85.58, retrained, folds 0–2, relaunched once after the 24 Oct host-RAM P-crash | [82.8, 87.2] | pass |
| A3-sym | native task | 0.10 solved | [0.174, 0.826] | fail |
| A3-img | native task | 0.20 solved | [0.280, 0.920] | fail |

### Readings (as pre-registered)

- **A1 ranks launches better than U on the sealed NovPhy grid (scoped)**, at 0.32 × U's training compute. The same holds for A2 and A2-native.
- **Q1:** A2 − U is `supported`. A2 reads the same carrier as U, has matched parameters and is trained on the same fit shots. So, for this task, the paper cannot claim that the world model adds ranking value over direct prediction.
- **A4's budget safeguard fired** (1.88 × U's training GPU-h), so its wording is *"A4 at its native budget (1.88 × U's training GPU-h) ranks better than U at B1"*. The #140 U-native control gave A4 − U-native = +0.1813 [+0.1658, +0.1967] with the stop token (U-native fails G-action on one seed). A4 stays budget-qualified, and no architectural reading is made.
- **#127 program claim: vetoed.** A1, A2 and A4 reach disposition 1.

### Context (descriptive, pre-registered before the freeze, no tokens)

- **Redundancy disclosed before the freeze.** The #109 cohort has two generator templates, and the policy levels have near-duplicate fit levels. A no-model nearest-fit verdict copy scored 0.946 on the policy grid; the policy-split selection AUCs (≈ 0.97–0.999) are inflated for every arm alike.
- **No-model references on the evaluation grid:**
  - NN-px 0.9039 [0.8838, 0.9231] and NN-car 0.7773;
  - A1 − NN-px = +0.0907 [+0.0719, +0.1105];
  - U − NN-px = −0.1595 [−0.1852, −0.1332].
- **Distance strata** (computed by #141 on these records): A1 sits at 0.99 in the near, mid and far distance strata.
- **#110 novelty pairs**, pooled over novelty levels 1–7:

| ranker | novel AUC | paired normal AUC |
| --- | --- | --- |
| A1 | 0.578 | 0.750 |
| A2 | 0.517 | 0.740 |
| A4 | 0.569 | 0.720 |
| U | 0.474 | 0.656 |

On novel levels the direct predictors fall to 0.52–0.58 (U 0.47), far below their ≈ 0.99 on the sealed grid.

### P-follow-up #141 (one line)

F1 (frozen U + A2's FiLM classifier) − U = +0.2321, `supported` (budget-qualified). F1 − F1-D (the same classifier without the rollout) = −0.0044, not `supported` either way. By the pre-registered reading, U's predictions showed no incremental ranking value.

### Selection, amendments, compute

**Selection** (policy split, seed 20260908; the selected run is reused as main seed 20260908):
- A1, A2-native and A4: `tenth` (10 000 updates; A4 selected after its G-fid passed);
- A2: lr 3e-4;
- A3-img: native-paper;
- A3-sym: native-code, re-selected after P-fid repair 1 (dev-mode platform labels; the superseded runs are kept and were never scored).

**Registered post-freeze amendments:**
- A1: drop marker, not exercised;
- A2: repair re-selection;
- A3: engine-seconds reporting.

All are registered in `scripts/post_freeze_amendments.py`, and each was made before the result it could touch.

**Compute:**
- child ledger 25.12 GPU-h, including 8.14 h of superseded A3-sym runs;
- A1 Tier E 0.24 GPU-h;
- A4 Tier T 41.28 GPU-h;
- engine 6.30 worker-h (A3 fidelity).

The A4 overruns were funded by P-cap from the unallocated program budget; nothing was cut.
