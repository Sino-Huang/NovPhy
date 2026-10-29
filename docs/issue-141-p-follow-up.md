# Issue #141: P-follow-up F1 (frozen U + the A-family's discriminative success readout), with the no-rollout control F1-D

Parent: #127 (external baselines program); program decisions [P-follow-up fired](https://github.com/Sino-Huang/NovPhy/issues/127#issuecomment-6479617332),
[ticket opened](https://github.com/Sino-Huang/NovPhy/issues/127#issuecomment-6479872095),
[funded under P-budget item 6](https://github.com/Sino-Huang/NovPhy/issues/127#issuecomment-6480403151). Ticket:
[#141](https://github.com/Sino-Huang/NovPhy/issues/141). The design was pre-registered in the ticket body before any
implementation and is binding. The #127 shared protocol (`scripts/external_baselines.py`, `eb`) applies. The Gate-B,
#101, #128 and #129 records and stores are read-only inputs. Nothing here edits the ICLR 2026 submission.

| | |
| --- | --- |
| Runner | `scripts/run_issue_141_p_follow_up.py`. Modes: cut-points, selection, selection-readouts, smoke, prepare, train, readouts, freeze-readouts, score, score-novelty, publish, validate |
| Reused code (by import) | #128 `world_model/baselines/phyre_dqn/novphy.py` (`A2Net`, `train_indices_sampler`, `make_optimizer`, `train_step`), #128 runner (NN-px / NN-car rules, `pixel_orders`), #129 runner (`UnitRollout`, U+head `cell_rows`, `load_head`) |
| Tests | `tests/test_issue_141_p_follow_up.py` (16 tests, CPU) |
| Scheduler | `scripts/gpu_window_issue_141.json` (selection, selection readouts, smoke); `scripts/gpu_window_issue_141_main.json` (train, readouts, freeze, score, novelty, publish, validate) |
| Artifacts | `.local-artifacts/issue-141-p-follow-up-v1/`: `plan.json`, `smoke.json`, `selection.json`, `decisions.md`, `strata/`, `features/`, `replication/`, `readouts-manifest.json`, `records-manifest.json`, `validation.json`, `summary.json`, `compute.json`, `findings.md`, `ledger/`. Models, features, readouts and records stay local; they are hashed in the manifests |
| Decisions | [implementation](https://github.com/Sino-Huang/NovPhy/issues/141#issuecomment-6480032792), [strata cut points](https://github.com/Sino-Huang/NovPhy/issues/141#issuecomment-6480033291), [compute and queues](https://github.com/Sino-Huang/NovPhy/issues/141#issuecomment-6480065190), all posted before the steps they govern |
| Freeze | Pre-freeze implementation `2504342d`. Cut points written 2026-10-28T20:43:35Z, before any training. `plan.json` frozen 2026-10-29T07:17:34Z (sha256 `cd06c20c…`, 169 pins), commit `bee50de3` ([freeze comment](https://github.com/Sino-Huang/NovPhy/issues/141#issuecomment-6490510965)), after a passed smoke and before any main-seed training or evaluation read. No post-freeze amendment |
| Readout freeze | `readouts-manifest.json` frozen 2026-10-29T07:30:26Z (sha256 `6ef3bd20…`), before any evaluation anchor was read |
| Scoring | Once, in a process that did not train: scheduler run `141-score#1`, 2026-10-29 19:40 AEDT, 77 s (anchors read 08:40:33Z). P-novelty pass `141-score-novelty#1`, 19:45, 86 s |
| Result comment | [issuecomment-6492488967](https://github.com/Sino-Huang/NovPhy/issues/141#issuecomment-6492488967) |
| Validation | `141-validate#1` exited 0 at 19:55 AEDT. 28 records were hash-checked and every published byte was re-derived. F1 seed 20260908 was re-scored on all four inventories with max \|Δ\| = 0. `validation.json` binds `records-manifest.json` sha256 `d0c1e42a…` |

Engine time is zero seconds. The primary metric is the Gate-B pooled grid AUC over the 311 mixed grid cells × 3 seeds of
the sealed #109 evaluation split, under `deterministic_scoring()`. Intervals are level-clustered percentile bootstraps
(10 000 draws, PCG64 7201). The margin is δ = 0.02. Tokens follow the δ-rule under the declared guard scope.

## Outcome

**The pre-registered reading, applied exactly:**

| Pre-registered result | Observed | Applies | Reading |
| --- | --- | --- | --- |
| F1 − U `supported`; F1 − F1-D not `supported` | F1 − U = +0.2321 [+0.2135, +0.2501], `supported`; F1 − F1-D = −0.0044 [−0.0081, −0.0014], `not_supported_by_this_experiment` (both directions) | **yes** | the borrowed package helps relative to U's scorer; U's predictions have not shown incremental ranking value |
| F1 − F1-D `supported` | not `supported` (see above) | no | — |
| F1 − U+head `supported` | +0.1642 [+0.1464, +0.1817], `supported` | **yes** | this readout package adds value relative to #129's (not an isolated mechanism claim) |
| little change from U+head | +0.1642 | no | — |

- **Guards.** Every global guard passes: G-isolation, G-chronology (cut points before training, the novelty pass
  included), G-scoring-policy, and G-records (own records plus the hash-checked imported #128 / #129 records). Every
  declared arm guard of F1 and F1-D passes on every seed:
  - G-rank: mean tied-pair share 0, no fully tied level.
  - G-nonfinite: 0.
  - G-trained.
  - G-alignment and G-classifier-only.
  - F1 only: G-u-identity and G-rollout-replication. U's endpoints on every mixed cell of every inventory equal the
    Gate-B U record exactly: 7 766 of 7 766 rows per seed, max \|Δ\| = 0.
- **Scope (declared).** F1 is a native-success-readout follow-up on #129's U+head route. It is not a member of #127's
  world-model disposition set, and no new program-level claim is made. G-action is inherited from U and reported, not
  relaxed: 0.0240 / 0.0229 / 0.0220 per seed, below 0.05. The `eb` arm block assigns F1 (and F1-D) disposition 1
  ("X ranks launches better than U on the sealed NovPhy grid (scoped)") under this declared guard scope only.
- **Budget safeguard.** It fires for F1 by construction: F1's training compute includes U's own training (1.03 × U).
  The wording is therefore budget-qualified: F1, built on U plus a 33 GPU-s classifier, ranks better than U. It does
  not fire for F1-D (0.025 × U).
- **Secondary contrasts (guarded).**
  - F1 − A2 = +0.0008 [−0.0019, +0.0034]: inside (−δ, +δ), not `supported` either way.
  - F1 − A1 = −0.0181 [−0.0253, −0.0120]. F1 − A1 is `not_supported_by_this_experiment`. A1 − F1 is
    `readiness_or_precision_insufficient`, because its interval straddles δ. This is not an acceptance criterion
    (pre-registered).
- **Memorisation context (descriptive).** NN-px reaches 0.9039; F1 − NN-px = +0.0726 [+0.0535, +0.0927] and
  F1 − NN-car = +0.1992 [+0.1739, +0.2249]. F1 holds roughly level across the frozen distance strata, while U, U+head
  and NN-car vary. On #110 novelty pairs F1 falls from 0.741 (paired normal) to 0.527 (novel), like A1 / A2 and U+head.

## Design summary

**Arms** (one classifier, `A2Net(base_width 70, carrier_dim 472)`, 1 855 301 trainable parameters in both arms):

| Arm | Classifier input | Role |
| --- | --- | --- |
| F1 = U+FiLM | [z₀, ẑ₂₂₅ᵁˢ(z₀, a)] and `launch4(a)`; ẑ from the frozen seed-s Gate-B unit U (#129 `UnitRollout`, F-15-macro, 15 transitions to frame 225) | follow-up (primary) |
| F1-D | [z₀, z₀] and `launch4(a)`, trained independently | control (no rollout) |

- **Recipe.** A2's, unchanged: class-balanced BCE with logits, batches of 64 = 32 + 32 with replacement, Adam, cosine
  over 9 000 updates, scheduler step before optimizer step.
- **Data.** One example per fit shot: 13 919 shots, 838 positive verdicts, 480 levels. No policy, evaluation or #110
  label enters optimisation. U and E99 are frozen.
- **U feature cache.** U's fit / policy endpoints are cached once per U seed (`features/U-seed<s>.pt`, pinned).
  Evaluation and novelty endpoints are rolled inside the scoring process.
- **Pairing.** F1 seed s reads U seed s. For every seed and configuration, F1 and F1-D share the initialisation and the
  batch-index stream (`eb.batch_rng(s, b)`), verified by sha256 in the smoke.
- **Selection** (seed 20260908, 94 policy grid mixed cells, `eb.select_configuration`). Parameters are equal across
  configurations, so ties fall to declaration order. The selected run is reused as seed 20260908.

| Arm | lr 3e-4 | lr 1e-4 | lr 1e-3 | Selected |
| --- | --- | --- | --- | --- |
| F1 | 0.9632 | 0.9590 | 0.9610 | lr 3e-4 |
| F1-D | 0.9573 | 0.9601 | 0.9646 | lr 1e-3 |

**Smoke (pre-freeze, fit / policy only).** Every check passed:
- Data accounting.
- U checkpoint identity against the Gate-B handoff hash, for all 3 seeds.
- Replication against U's records:
  - U's G-action on the cached 480 × 16 fit endpoints equals #129's calibration within 1.8e-9.
  - #129's frozen U+head heads on the cached policy endpoints reproduce #129's policy AUC exactly (0.80170 / 0.81612 /
    0.82089).
- Candidate / action alignment.
- Only classifier parameters update: U's state hash is unchanged, no U parameter receives a gradient, and the feature
  files are unchanged.
- Paired streams.
- CPU tests.

The projection was 1.41 GPU-h against the 60 GPU-h reserve.

## Results (sealed grid, 311 mixed cells × 3 seeds)

| Ranker | Grid AUC [CI] | Top-1 | Role |
| --- | --- | --- | --- |
| F1 | 0.9765 [0.9689, 0.9828] | 0.811 | follow-up |
| F1-D | 0.9809 [0.9748, 0.9863] | 0.837 | control |
| A1 (#128) | 0.9946 [0.9917, 0.9971] | 0.959 | imported |
| A2 (#128) | 0.9757 [0.9692, 0.9815] | 0.798 | imported |
| NN-px (#128, no model) | 0.9039 [0.8838, 0.9231] | 0.807 | descriptive |
| U+head (#129) | 0.8123 [0.7948, 0.8303] | 0.348 | imported control |
| C* | 0.7834 [0.7676, 0.7992] | 0.192 | reference |
| NN-car (#128, no model) | 0.7773 [0.7502, 0.8038] | 0.572 | descriptive |
| J_dec | 0.7460 [0.7277, 0.7637] | 0.180 | reference |
| U | 0.7444 [0.7269, 0.7620] | 0.195 | reference |
| P | 0.7203 [0.6965, 0.7442] | 0.180 | reference |

The chance top-1 rate is 0.089. Per seed, F1 scores 0.9763 / 0.9791 / 0.9741 and F1-D scores 0.9810 / 0.9824 /
0.9794. F1's per-arm contrasts are all `supported`: X − C* +0.1931, X − P +0.2562, X − J_dec +0.2305. F1-D's contrasts
are similar.

**Other inventories (F1 − U, guarded):**

| Inventory | U | F1 | F1-D | F1 − U [CI] |
| --- | --- | --- | --- | --- |
| offset | 0.6670 | 0.9706 | 0.9746 | +0.3037 [+0.2883, +0.3203] |
| angle | 0.6791 | 0.9865 | 0.9904 | +0.3073 [+0.2934, +0.3216] |
| power | 0.8058 | 0.8677 | 0.8650 | +0.0619 [+0.0385, +0.0875] |

All three are `supported`. F1-D − U is `supported` on every inventory.

**Distance strata** (descriptive; memorisation safeguard 2). The cut points 0.3246 / 0.4763 (mean \|Δ\| grey levels to
the nearest fit anchor frame) were frozen on the 94 policy cells before training. Grid mixed cells per stratum: near
104, mid 112, far 95. No stratum is selected.

| Stratum | F1 | F1-D | U+head | A1 | NN-px | NN-car | U |
| --- | --- | --- | --- | --- | --- | --- | --- |
| near | 0.9777 | 0.9816 | 0.7598 | 0.9916 | 0.9216 | 0.7616 | 0.7039 |
| mid | 0.9740 | 0.9851 | 0.7982 | 0.9978 | 0.8903 | 0.8097 | 0.7425 |
| far | 0.9780 | 0.9753 | 0.8865 | 0.9943 | 0.9007 | 0.7562 | 0.7909 |

Intervals are in `summary.json` (`extra.strata`) and `findings.md`.

**#110 novelty pairs** (descriptive; memorisation safeguard 3; mixed levels; 72 paired level pairs, pooled over novelty
levels 1–7). NN's exact-launch rule leaves 272 of 4 111 candidates unavailable, all on the mirrored level 5, so NN pools
57 pairs.

| Ranker | Novel AUC [CI] | Paired normal AUC | Novel − normal [CI] |
| --- | --- | --- | --- |
| F1 | 0.5270 [0.4758, 0.5770] | 0.7412 | −0.1846 [−0.2583, −0.1116] |
| F1-D | 0.5218 [0.4681, 0.5739] | 0.7477 | −0.2043 [−0.2774, −0.1316] |
| U+head (#129, re-scored) | 0.4719 [0.4302, 0.5135] | 0.7265 | −0.2331 [−0.2999, −0.1651] |
| A1 (#128) | 0.5776 [0.5269, 0.6272] | 0.7495 | −0.1392 [−0.2104, −0.0651] |
| A2 (#128) | 0.5172 [0.4661, 0.5681] | 0.7400 | −0.1867 [−0.2629, −0.1122] |
| U | 0.4743 [0.4354, 0.5119] | 0.6562 | −0.1537 [−0.2127, −0.0960] |
| NN-px | 0.5684 [0.5330, 0.6061] | 0.6115 | −0.0053 [−0.0506, +0.0396] |

The per-level table (levels 1–7, C*, P, NN-car, A2-native, A4) is in `findings.md`.

## Compute

| | F1 | F1-D |
| --- | --- | --- |
| Trainable parameters | 1 855 301 (+ frozen U 1 837 690) | 1 855 301 |
| Training GPU-s per seed | 1 325–1 334 (classifier 32–33 + U feature pass 1.3–2.2 + U's own training 1 290–1 300) | 32–33 |
| MACs per candidate | 2.28e7 (U rollout 2.10e7 + classifier 1.84e6) | 1.84e6 |
| Wall per decision (16 candidates) | 6.6 ms | 0.85 ms |

The child ledger records 381 GPU-s. Scheduler process time over the 13 jobs is about 610 s (0.17 GPU-h) against the
60 GPU-h reserve.

## Disclosures

- F1 and F1-D were each selected on the policy split among three pre-declared learning rates; U had no such tuning.
- While reading formats for the implementation (before any F1 component existed), the implementer printed the field
  layout of one row each of the published #128 A2 and #129 U+head records and two entries of #128's NN-px `nearest`
  map. This is disclosed in the implementation decision. Nothing was chosen on them.
- `findings.md`'s "No-model references" lines print "grid AUC n/a" for NN-px / NN-car. This is a display-path slip in
  the frozen runner: it reads the ranker table one level too shallow. The values (0.9039, 0.7773) are in `summary.json`
  (`extra.inventories.grid.rankers.rankers`) and in this document. The tables, tokens and contrasts are unaffected,
  and the frozen runner is left unamended.
