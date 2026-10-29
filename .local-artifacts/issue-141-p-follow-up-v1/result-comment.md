## #141 result: F1 (frozen U + A2's FiLM success readout) and F1-D (no-rollout control) — published and validated

The pre-registered design is unchanged. The decisions governing the open details were posted before their steps
([implementation](https://github.com/Sino-Huang/NovPhy/issues/141#issuecomment-6480032792), [strata](https://github.com/Sino-Huang/NovPhy/issues/141#issuecomment-6480033291), [compute/queues](https://github.com/Sino-Huang/NovPhy/issues/141#issuecomment-6480065190)).

**Run and bindings.**
- Freeze: `plan.json` sha256 `cd06c20c…` (`bee50de3`), frozen before main training. The readout freeze (07:30:26Z)
  came before the evaluation read (08:40:33Z). There was no post-freeze amendment.
- Scoring: one pass on 2026-10-29 (`141-score`, 77 s), then the novelty pass (86 s).
- `--validate` exited 0. 28 records were hash-checked and every published byte was re-derived. F1 seed 20260908 was
  re-scored on all inventories with max \|Δ\| = 0.
- Selection on the 94 policy cells:
  - F1: lr 3e-4 (0.9632; 1e-4 0.9590; 1e-3 0.9610).
  - F1-D: lr 1e-3 (0.9646; 3e-4 0.9573; 1e-4 0.9601).

**Guards.** Every guard passes:
- Global: G-isolation, G-chronology, G-scoring-policy, and G-records (including the hash-checked imported #128 / #129
  records).
- Arm guards on every seed: G-rank (tied share 0), G-nonfinite (0) and G-trained.
- G-alignment and G-classifier-only.
- F1 only:
  - G-u-identity.
  - G-rollout-replication. U's endpoints on every mixed cell of every inventory equal the Gate-B U record exactly
    (7 766 / 7 766 rows per seed). Before the freeze, U's G-action matched #129's calibration within 2e-9 and #129's
    U+head policy AUC was reproduced exactly.

G-action is inherited from U and reported, not relaxed: 0.0240 / 0.0229 / 0.0220 against 0.05. Scope (declared): F1 is
a native-success-readout follow-up on #129's U+head route, not a member of #127's disposition set. No program-level
claim is made.

**Sealed grid (311 mixed cells × 3 seeds):**

| Ranker | AUC [CI] |
| --- | --- |
| F1 | 0.9765 [0.9689, 0.9828] |
| F1-D | 0.9809 [0.9748, 0.9863] |
| A1 | 0.9946 |
| A2 | 0.9757 |
| NN-px | 0.9039 |
| U+head | 0.8123 |
| U | 0.7444 |

| Contrast | Estimate [CI] | Token (forward / reverse) |
| --- | --- | --- |
| **F1 − U (primary)** | +0.2321 [+0.2135, +0.2501] | `supported` / `not_supported_by_this_experiment` |
| F1 − F1-D | −0.0044 [−0.0081, −0.0014] | `not_supported_by_this_experiment` / `not_supported_by_this_experiment` |
| F1 − U+head | +0.1642 [+0.1464, +0.1817] | `supported` / `not_supported_by_this_experiment` |
| F1 − A2 | +0.0008 [−0.0019, +0.0034] | `not_supported_by_this_experiment` / `not_supported_by_this_experiment` |
| F1 − A1 (not an acceptance criterion) | −0.0181 [−0.0253, −0.0120] | `not_supported_by_this_experiment` / `readiness_or_precision_insufficient` |
| F1 − NN-px (descriptive) | +0.0726 [+0.0535, +0.0927] | no token |
| F1 − NN-car (descriptive) | +0.1992 [+0.1739, +0.2249] | no token |

**Pre-registered reading, applied exactly:**

| Result | Applies | Reading |
| --- | --- | --- |
| F1 − U `supported`; F1 − F1-D not `supported` | **yes** | the borrowed package helps relative to U's scorer; U's predictions have not shown incremental ranking value |
| F1 − F1-D `supported` | no | — |
| F1 − U+head `supported` | **yes** | this readout package adds value relative to #129's (not an isolated mechanism claim) |
| little change from U+head | no | — |

**Budget safeguard.** It fires for F1 by construction: F1's compute includes U's own training (1.03 × U's training
GPU-h; the classifier itself is 32–33 GPU-s per seed). The F1 − U reading is therefore budget-qualified.

**Memorisation safeguards (descriptive).**
- Frozen distance strata (cut points 0.3246 / 0.4763; near 104, mid 112, far 95 cells):

  | Ranker | near | mid | far |
  | --- | --- | --- | --- |
  | F1 | 0.978 | 0.974 | 0.978 |
  | F1-D | 0.982 | 0.985 | 0.975 |
  | U+head | 0.760 | 0.798 | 0.887 |
  | NN-px | 0.922 | 0.890 | 0.901 |
  | U | 0.704 | 0.743 | 0.791 |

- #110 novelty pairs, pooled novel AUC (paired normal in parentheses), with the novel − normal difference:

  | Ranker | Novel (paired normal) | Novel − normal [CI] |
  | --- | --- | --- |
  | F1 | 0.527 (0.741) | −0.185 [−0.258, −0.112] |
  | F1-D | 0.522 (0.748) | −0.204 [−0.277, −0.132] |
  | U+head | 0.472 (0.727) | −0.233 [−0.300, −0.165] |
  | A1 | 0.578 (0.750) | −0.139 [−0.210, −0.065] |
  | A2 | 0.517 (0.740) | −0.187 [−0.263, −0.112] |
  | U | 0.474 (0.656) | −0.154 [−0.213, −0.096] |
  | NN-px | 0.568 (0.612) | −0.005 [−0.051, +0.040] |

  NN-px pools 57 pairs: 272 of 4 111 candidates on the mirrored level-5 novel sides are unavailable under the
  exact-launch rule.

**Compute.** The child ledger records 381 GPU-s; the scheduler process total is about 0.17 GPU-h, against a 60 GPU-h
reserve.

**Disclosure.** `findings.md` prints "grid AUC n/a" for NN-px / NN-car. This is a display slip in the frozen runner;
the values are in `summary.json` and the doc. Tables and tokens are unaffected.

Doc: `docs/issue-141-p-follow-up.md`.
