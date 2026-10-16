# Issue #113: macro descriptions that can be observed (items 1 and 3)

Parent: #108. Item 2, the capture window, was delivered in 006b7b78 (`docs`: `.local-artifacts/issue-113-capture-window-v1/`). The #85 shared rules apply. Engine seconds: 0. Prior dispositions and #110's labels are read-only.

| Deliverable | Location |
| --- | --- |
| Label module (#115 item 4, #120) | `world_model/data/observable_macro.py` |
| Temporal head module | `world_model/training/temporal_macro_head.py` |
| Runner (both items, two freezes) | `scripts/run_observable_macro.py` |
| Tests | `tests/test_issue_113_observable_macro.py`, `tests/test_issue_113_temporal_macro_head.py` |
| Item 1 artefacts | `.local-artifacts/issue-113-macro-predicates-v1/` |
| Item 3 artefacts | `.local-artifacts/issue-113-temporal-macro-head-v1/` |
| Item 1 freeze | `98d96bc4`, 2026-10-16T13:31:48Z, before any head training |
| Item 3 freeze | `1c3850ee`, 2026-10-16T13:54:57Z, before any evaluation-split byte was read and before any head training |
| Validation | `python -u -m scripts.run_observable_macro --validate` (exit 0 after the item-3 publication) |

## Owner decisions (binding)

- Item 1 runs first and is frozen before any head training.
- Item 3's held-out score uses item 4's pre-declared readout target. That target is macro F1 per redefined predicate on the #104 held-out split (the #109 evaluation stream), with the #100 audit machinery (readout counts and phases).

## Item 1: definitions (frozen)

Labels are indexed by the landing frame f = t + Δ of a Δ request issued at frame t, with Δ ∈ {1, 5, 15}. One frame is 50 native steps.

- **rest.** At every native step of frame f's interval, every present, simulated, dynamic body (birds included) has |v| ≤ 0.01 u/s and |ω| ≤ 0.01 deg/s. These are the engine's `stable_entered` thresholds.
- **settle_Δ(f).** Let D = max(k, Δ). Settle holds when the launched bird was removed (engine `entity_destroyed`) at or before frame f − D, and no body moved in frames f − D + 1 .. f. It is available iff f ≥ D. The debounce is **k = 1**, so the window is 1 / 5 / 15 frames.
- **structure-change_Δ(f).** S(f) ≠ S(f − Δ), available iff f ≥ Δ. S(f) holds the directed (supporter, supported) pairs with no bird on either side that are present on more than half of the native steps of frame f's interval. Single-step spikes are filtered out. The comparison is between the two endpoints only.
- **Phases.** pre_contact runs until the launched bird's first native-step contact with a non-bird, non-slingshot entity. settled starts at the first later frame after which settle_1 holds to the end. cascade is everything in between.
- **Relation to #110.** The per-Δ rule of #110 (`novelty_labels_v2.window_any`, an OR over (f − Δ, f]) is not adopted. Structure change compares debounced endpoints. Settle is an AND over the window.

### Debounce selection (fit stream and retained captures only)

The rule was frozen in code before it was run. It picks the smallest k in {1, 2, 3, 5, 10, 15, 20, 25, 30, 35} whose relapse rate is ≤ 5 %. A relapse is any motion within 15 frames after a settle frame. The rate is pooled over the #104 fit stream (13 919 shots) and the N1 (178), #93 (391) and #94 (295) captures.

- **First pass, rest only** (`superseded/selection-rest-only.json`). The relapse rate rose with k: 5.1 % at k = 1, 12 % at k = 25 and 89 % at k = 35. Example traces show why. A bird resting against a block is removed by the engine about 30 frames after the scene rests, and the block then tips.
- **Amendment.** Before the freeze, settle was amended to require the launched bird's removal. With that rule, 0 of 53 020 settle frames relapse at any k, so **k = 1**.

### Prevalence

`findings.md` and `prevalence.csv` give the full table, per level type × phase × predicate × Δ.

| source | settle Δ1 / Δ5 / Δ15 | structure-change Δ1 / Δ5 / Δ15 |
| --- | --- | --- |
| fit, type010103 (6 960 shots) | 0.88 / 0.76 / 0.50 % | 0.44 / 1.42 / 2.46 % |
| fit, type010105 (6 959) | 1.84 / 1.61 / 1.16 % | 0.13 / 0.55 / 1.39 % |
| policy, type010103 (1 740) | 0.96 / 0.83 / 0.54 % | 0.42 / 1.40 / 2.42 % |
| policy, type010105 (1 740) | 2.06 / 1.81 / 1.31 % | 0.12 / 0.50 / 1.30 % |
| N1, type010103 / type010105 | 0.18 / 0.10 / 0.05 % and 0.15 / 0.02 / 0.00 % | 0.51 / 1.82 / 3.06 % and 0.09 / 0.40 / 0.99 % |
| #93, type010103 / type010105 | 0.13 / 0.06 / 0.03 % and 0.13 / 0.00 / 0.00 % | 0.63 / 1.95 / 3.26 % and 0.11 / 0.45 / 1.13 % |
| #94, type010103 / type010105 | 0.11 / 0.06 / 0.02 % and 0.04 / 0.00 / 0.00 % | 0.28 / 0.95 / 1.76 % and 0.07 / 0.25 / 0.63 % |

How the positives fall across phases:

- **Settle.** Every settle positive lies in the settled phase. The fit stream has 93 087 settled frames, against 1 settled frame on #100's held-out audit.
- **Structure change.** Positives lie in the cascade phase; the exception is a few Δ-windows that reach back across the settled onset.
- **Capture window.** The retained pre-#113 captures stop at engine rest. They therefore carry almost no settle frames, which is the item-2 finding. The #104 streams, recorded with the rest tail, carry 0.5–2 %.

## Label module API (#115 item 4, #120)

`observable_macro` imports only json, gzip, math and numpy, so process-pool workers can import it.

| call | returns |
| --- | --- |
| `trace_summary(native_root)` | A definition-free summary of a `canonical_native_trace_v1` trace (every native step). Fields: `frame_steps`, `moving`, `moving_nonbird`, `support_index` / `support_sets`, `launch_step`, `launched_bird`, `bird_removed_step`, `destroyed`, `contact_onset_step`, `stability_events`, `terminal`. |
| `write_summary` / `load_summary` / `summary_worker((native_root, target))` | Byte-stable gzip-JSON I/O and the pool entry point. |
| `shot_labels(summary, k=None, deltas=DELTAS)` | `values` and `available` as bool `[F, 2, len(deltas)]`, indexed `[frame, PREDICATES, Δ]`; `phase` as `[F]`; the onsets. `k` defaults to the frozen `DEBOUNCE_FRAMES`. |
| `labels_from_trace(native_root)` | Shorthand for `shot_labels(trace_summary(...))`. |
| `settle_labels`, `change_labels`, `phases`, `select_debounce`, `relapse_counts` | The individual rules. |

Stored summaries (local):

- `.local-artifacts/issue-113-macro-predicates-v1/summaries/<fit|policy|n1|i93|i94>/<key>.json.gz`.
- Their sha256 values are in `summaries-manifest.json` (frozen) and `summaries-manifest-run.json` (policy).
- Shot keys come from `scripts.run_observable_macro.shots(source)`.

## Item 3: temporal head (frozen, run, published)

The head and its interface are defined in `temporal_macro_head.py`:

- **Interface.** `head(history [B, K, 236], Δ) -> logits [B, 2]`. The history is stride-Δ: `history[:, -1]` is the landing carrier, which may be engine-parsed or predicted, and earlier entries sit at f − jΔ, clamped at frame 0.
- **Rollouts.** A Δ-step rollout supplies this history natively (`rollout_history`).
- **Features.** The landing carrier plus the K − 1 stride-Δ differences.
- **Network.** One 128-wide MLP per Δ, the same width as the current `macro_head`.
- **Loss.** Class-balanced BCE.

The frozen plan is `plan.json` (freeze `1c3850ee`):

- **Training data.** Every frame of the 13 919 fit shots (6 947 455 rows, 480 levels), as fresh deterministic E99 parses.
- **Grouped leave-out.** Five level-grouped folds, PCG64(11301).
- **Candidates.** K ∈ {2, 3, 4} × training mix ∈ {0, 0.5}, where mix is the share of each batch whose landing carrier is the paired Gate-B one-step prediction. The single-frame head on the new labels is K = 1 with the same mix choice. The Gate-B `macro_head` is scored as-is at 0.5.
- **Selection.** By the mean of out-of-fold engine and predicted macro F1, with per-source thresholds.
- **Final heads.** Three seeds, paired with the Gate-B seeds.
- **Held-out scoring.** One pass on the #109 evaluation stream (11 680 shots, 5 834 774 rows, 400 levels) with three sources: engine (teacher-forced), predicted (one Δ-step Gate-B prediction from the observed anchor) and rollout (open-loop from the decision frame, descriptive). Records hold #100-style tp/fp/fn/tn per level × predicate × Δ × phase.
- **Contrasts.** Temporal − as-is, temporal − single and single − as-is on the engine and predicted sources, compared with a paired level bootstrap (10 000 draws, PCG64(11302)) against a 0.05 margin: `supported` needs every guard, lower bound > 0 and estimate ≥ 0.05. Rollout contrasts are descriptive.

Artefacts in `.local-artifacts/issue-113-temporal-macro-head-v1/`: `plan.json`, `smoke.json`, `selection.json`, `summary.json`, `findings.md` (every table, including per-phase readout counts), `readout.csv` (every head × source × phase × predicate × Δ, with tp/fp/fn/tn), `compute.json`, `ledger/`, `data/*.meta.json`, `replay/`. The 109 raw readout records and the 46 scored heads (40 grouped-leave-out, 6 final) are kept locally; their sha256 values are committed in `records-manifest.json`.

### Selection (Acceptance 3)

Selected: temporal head **K4-mix50**, single-frame head **K1-mix50**. K = 4 is the largest history in the frozen grid; nothing beyond it was tried. Out-of-fold macro F1 (seed 20260908; thresholds chosen per source):

| candidate | engine | predicted | score (mean) |
| --- | --- | --- | --- |
| K1-mix0 | 0.085 | 0.019 | 0.052 |
| K1-mix50 (single, selected) | 0.076 | 0.080 | 0.078 |
| K2-mix0 | 0.108 | 0.015 | 0.062 |
| K2-mix50 | 0.099 | 0.083 | 0.091 |
| K3-mix0 | 0.158 | 0.014 | 0.086 |
| K3-mix50 | 0.139 | 0.096 | 0.118 |
| K4-mix0 | 0.194 | 0.014 | 0.104 |
| K4-mix50 (temporal, selected) | 0.165 | 0.117 | 0.141 |

Without predicted carriers in training (mix 0), every head fails on predicted landings (macro F1 0.014–0.019). The mix-0.5 heads give up some engine F1 to read predicted ones.

### F1 tables

Grouped leave-out on the fit split (Acceptance 3). The current head is in-sample, since the Gate-B predictor was trained on the fit split. Macro F1 is the mean over predicate × Δ, with a DESCRIPTIVE level-clustered 95 % interval:

| head | source | macro F1 | settle Δ1 / Δ5 / Δ15 | structure-change Δ1 / Δ5 / Δ15 |
| --- | --- | --- | --- | --- |
| temporal (K4-mix50) | engine | 0.165 [0.159, 0.171] | 0.081 / 0.145 / 0.278 | 0.049 / 0.166 / 0.269 |
| temporal (K4-mix50) | predicted | 0.117 [0.111, 0.124] | 0.086 / 0.084 / 0.157 | 0.045 / 0.145 / 0.187 |
| single (K1-mix50) | engine | 0.076 [0.071, 0.082] | 0.059 / 0.058 / 0.066 | 0.033 / 0.081 / 0.160 |
| single (K1-mix50) | predicted | 0.080 [0.076, 0.085] | 0.073 / 0.075 / 0.059 | 0.043 / 0.114 / 0.117 |
| current (Gate-B `macro_head`) | engine | 0.000 [0.000, 0.000] | 0 / 0 / 0 | 0 / 0 / 0 |
| current (Gate-B `macro_head`) | predicted | 0.000 [0.000, 0.000] | 0 / 0 / 0 | 0 / 0 / 0 |

Held-out on the #109 evaluation split, scored once with seed means of 3 (Acceptance 4; the item-4 readout target). The engine source is parsed carriers, predicted is one-step predictions, and rollout is open-loop rollouts (descriptive):

| head | source | macro F1 | settle Δ1 / Δ5 / Δ15 | structure-change Δ1 / Δ5 / Δ15 |
| --- | --- | --- | --- | --- |
| temporal | engine | 0.157 [0.151, 0.164] | 0.086 / 0.145 / 0.226 | 0.050 / 0.170 / 0.266 |
| temporal | predicted | 0.113 [0.107, 0.119] | 0.069 / 0.071 / 0.149 | 0.052 / 0.140 / 0.196 |
| temporal | rollout | 0.015 [0.014, 0.015] | 0.000 / 0.007 / 0.009 | 0.000 / 0.021 / 0.050 |
| single | engine | 0.082 [0.078, 0.087] | 0.070 / 0.065 / 0.073 | 0.034 / 0.084 / 0.167 |
| single | predicted | 0.075 [0.070, 0.080] | 0.065 / 0.062 / 0.047 | 0.048 / 0.111 / 0.116 |
| single | rollout | 0.018 [0.017, 0.019] | 0.010 / 0.015 / 0.009 | 0.000 / 0.016 / 0.057 |
| current | engine / predicted / rollout | 0.000 [0.000, 0.000] | 0 / 0 / 0 | 0 / 0 / 0 |

The current head predicts at most one positive frame per predicate × Δ on fit or held-out (`readout.csv`). This is #100's finding that E99 never predicts a positive, now on the redefined labels.

Held-out contrasts, as macro-F1 differences under the frozen rule:

| contrast | engine | predicted | rollout (descriptive) |
| --- | --- | --- | --- |
| temporal − current | +0.157 [0.151, 0.164] `supported` | +0.113 [0.107, 0.119] `supported` | +0.015 [0.014, 0.015] |
| temporal − single | +0.075 [0.068, 0.082] `supported` | +0.038 [0.034, 0.042] `not_supported_by_this_experiment` | −0.003 [−0.004, −0.002] |
| single − current | +0.082 [0.078, 0.087] `supported` | +0.075 [0.070, 0.080] `supported` | +0.018 [0.017, 0.019] |

Reading under the δ rule (margin 0.05):

- **Temporal vs current head.** On parsed carriers and on one-step predictions, the temporal head clears the margin over the current head. The current head reads nothing on these labels, so this contrast measures the redefined labels and the training as much as the temporal context.
- **Temporal context, holding labels and width fixed.**
  - Engine carriers: +0.075, `supported`.
  - Predicted carriers: numerically higher by 0.038, below the 0.05 margin. No ≥ 0.05 advantage is established.
  - Open-loop rollouts: the single-frame head is numerically higher by 0.003 (descriptive).
- **Rollouts.** Every head is near zero under open-loop rollouts (macro F1 ≤ 0.018). The head cannot recover the predicates from the predictor's long-horizon carriers.
- **Absolute level.** Every held-out macro F1 is far below the #100 macro readout bar of 0.7, which is reported, not tested. The best cell is settle Δ15 in the settled phase on engine carriers, at F1 0.754. Most errors are false positives in pre_contact and cascade frames; `findings.md` gives the readout counts per phase.

### Guards and compute

G1–G7 all pass:

- **G1.** 94.6 % of the 3 904 309 fit rows at positions 0..285 lie within the parse tolerance of the v3 carriers; the threshold is 90 %.
- **G2.** The Gate-B checkpoints match the handoff.
- **G3.** Carrier parsing started at 13:55:25Z and the evaluation scan at 15:31:59Z, both after the 13:54:57Z freeze.
- **G4.** 0 mismatched shots on either split.
- **G5.** 4 of 4 worker replays are identical.
- **G6.** Every readout is finite.
- **G7.** Both caps hold.

Every record was scored under `deterministic_scoring()`. `--validate` regenerates the five publication files byte-for-byte and checks the records manifest. It also re-scores two held-out records (temporal seed 0 engine; current head seed 0 predicted) and one grouped-leave-out record, and all three are identical. It exits 0.

Compute (`compute.json`, ledger):

| phase | GPU s |
| --- | --- |
| carriers | 22 811 |
| grouped leave-out | 686 |
| final heads | 153 |
| scoring | 134 |
| smoke | 138 |
| **total** | **23 921** |

- **Wall time.** 26 776 s of summed process wall, including the 2 760 s evaluation scan, which used 120 945 CPU process-s on 44 processes. The run took about 2.5 h on 4 RTX 5090s, against a projection of 60 139 GPU s; training ran faster than the smoke's 100-step timing implied.
- **Accounting.** GPU s is set equal to process wall per phase. The validation re-score (about 320 s on one GPU) is not ledgered.
- **Engine seconds.** 0.

## Execution disclosures

- **Interrupted first scan.** The first item-1 scan launch lost its parent process to the tool shell after about 1 minute. Its atomically written N1 summaries were kept. The scan was relaunched detached; this was not a typed failure.
- **Orphaned pool workers.** 44 idle spawn workers were found orphaned (PPID 1, 0 % CPU, about 15 h old), most likely left by that lost item-1 launch (44 is the scan pool size). They were still alive during the item-3 smoke, and were killed before the item-3 freeze. They wrote nothing.
- **Rest-only pass discarded.** The rest-only selection pass (above) was superseded before the freeze. Its summaries were rescanned with the removal field added. Both passes are counted in the ledger.
- **Exploration before the rule.** A 400-shot random sample of the fit stream was scanned before the debounce rule was written.
- **Item-3 stages.** Each stage was launched once, and nothing was relaunched or retried. There were no typed failures, and the runner was not edited after the freeze.
  - The first recorded carrier PIDs were the `setsid` wrappers. A wait on them returned at once while the workers kept running. The real worker PIDs were then recorded and waited on.
  - Every stage's completion was confirmed from its ledger entries and record counts: 25 599 carrier files, 82 grouped-leave-out records, 6 final heads and 27 held-out records.
- **Pre-freeze check.** Before the freeze, fit shot lengths were read from the item-1 summaries (at most 644 frames) to size the rollout memory. No evaluation byte was read.
