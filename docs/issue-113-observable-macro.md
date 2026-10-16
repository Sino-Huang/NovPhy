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
| Validation | `python -u -m scripts.run_observable_macro --validate` |

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

## Item 3: temporal head (implemented and smoked; not yet frozen or run)

The head and its interface are defined in `temporal_macro_head.py`:

- **Interface.** `head(history [B, K, 236], Δ) -> logits [B, 2]`. The history is stride-Δ: `history[:, -1]` is the landing carrier, which may be engine-parsed or predicted, and earlier entries sit at f − jΔ, clamped at frame 0.
- **Rollouts.** A Δ-step rollout supplies this history natively (`rollout_history`).
- **Features.** The landing carrier plus the K − 1 stride-Δ differences.
- **Network.** One 128-wide MLP per Δ, the same width as the current `macro_head`.
- **Loss.** Class-balanced BCE.

The plan is coded in `frozen_plan3` but is not yet written to `plan.json`:

- **Training data.** Every frame of the 13 919 fit shots, as fresh deterministic E99 parses.
- **Grouped leave-out.** Five level-grouped folds, PCG64(11301).
- **Candidates.** K ∈ {2, 3, 4} × training mix ∈ {0, 0.5}, where mix is the share of each batch whose landing carrier is the paired Gate-B one-step prediction. The single-frame head on the new labels is K = 1 with the same mix choice. The Gate-B `macro_head` is scored as-is at 0.5.
- **Selection.** By the mean of out-of-fold engine and predicted macro F1, with per-source thresholds.
- **Final heads.** Three seeds, paired with the Gate-B seeds.
- **Held-out scoring.** One pass on the #109 evaluation stream with three sources: engine (teacher-forced), predicted (one Δ-step Gate-B prediction from the observed anchor) and rollout (open-loop from the decision frame, descriptive). Records hold #100-style tp/fp/fn/tn per level × predicate × Δ × phase.
- **Contrasts.** Temporal − as-is, temporal − single and single − as-is, compared with a paired level bootstrap against a 0.05 margin.
- **Guards.** G1–G7.

The pre-freeze smoke (`smoke.json`) passed. It projects 60 139 GPU s, about 4.2 h of wall time on 4 GPUs.

**Not done yet:** the item-3 freeze, run and publication, and therefore the Acceptance 3/4 F1 tables. The remaining sequence is:

```
python -u -m scripts.run_observable_macro --prepare3      # then commit + push the freeze
CUDA_VISIBLE_DEVICES=k python -u -m scripts.run_observable_macro --carriers --workers 4 --worker k   # k = 0..3
python -u -m scripts.run_observable_macro --scan-evaluation
python -u -m scripts.run_observable_macro --assemble --split fit
CUDA_VISIBLE_DEVICES=k ... --cv --workers 4 --worker k ; ... --select3 ; ... --final --workers 4 --worker k
python -u -m scripts.run_observable_macro --assemble --split evaluation
CUDA_VISIBLE_DEVICES=k ... --score --workers 4 --worker k ; ... --publish3 ; ... --validate
```

## Execution disclosures

- **Interrupted first scan.** The first item-1 scan launch lost its parent process to the tool shell after about 1 minute. Its atomically written N1 summaries were kept. The scan was relaunched detached; this was not a typed failure.
- **Rest-only pass discarded.** The rest-only selection pass (above) was superseded before the freeze. Its summaries were rescanned with the removal field added. Both passes are counted in the ledger.
- **Exploration before the rule.** A 400-shot random sample of the fit stream was scanned before the debounce rule was written.

Compute so far: item 1 used CPU only (`compute.json`) and 0 GPU s. The item-3 smoke runs used under 0.1 GPU-h.
