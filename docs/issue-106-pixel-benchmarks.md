# Issue #106: description levels on CLEVRER and Physion-Dominoes, from pixels

Parent: #108. The shared rules of #85 and the #111 scoring contract apply. Owner decisions on #106 apply:
- the encoder is the #99 spatial slot parser (E99), not the JEPA encoder;
- macro predicates follow the #113 observable definitions;
- the download scope is CLEVRER videos + QA and the Physion Dominoes release with its OCP readout and test sets. Other
  Physion scenarios are out of scope.

Prior dispositions and the #79 / #84 / #115 / #117 artefacts are read-only inputs.

| | |
| --- | --- |
| Runner | `scripts/run_pixel_benchmarks.py` (modes `manifest`, `smoke`, `freeze`, `prepare`, `encoder`, `parse`, `train`, `policy`, `score`, `spot`, `publish`, `validate`); modules `world_model/data/pixel_benchmarks.py` (readers, frozen descriptions), `world_model/training/pixel_carrier.py` (parser loss, carrier builder, rollouts, QA resolution, bootstrap); tests `tests/test_pixel_benchmarks.py` |
| Artifacts | `.local-artifacts/issue-106-pixel-benchmarks-v1/` (`plan.json`, `smoke.json`, `data-manifest.json`, `amendments.json`, `summary.json`, `findings.md`, `compute.json`, `ledger.json`, `launches.json`, `spot.json`, training / policy / encoder reports, `records-manifest.json`) |
| Data | `data/clevrer/` and `data/physion-dominoes/`, with `manifest.pre-download.json` (sizes measured before the download) and `manifest.json` (sha256, S3 ETags, extraction counts) |
| Freeze | Plan frozen 2026-10-17T21:51:28Z in commit 2264726a, before any encoder, predictor or policy update and before any score |
| Validation | `python -u -m scripts.run_pixel_benchmarks --validate` (exit 0: summary, findings, compute and records manifest regenerate byte for byte; G5 spot re-parse diff 0.0, re-score diff ≤ 1.2e-7); `pytest tests/test_pixel_benchmarks.py` (7 passed) |

Zero engine seconds. Intervals are DESCRIPTIVE percentile bootstraps (10 000 draws, PCG64 7201). Each draw resamples seeds
(5 predictor members) and items independently; CLEVRER question options are clustered by video. All numbers and per-seed
values are in `findings.md` / `summary.json`.

## Data (downloaded under the owner's scope; sizes measured and recorded before the download)

| benchmark | files | bytes | licence |
| --- | --- | --- | --- |
| CLEVRER | `video_train.zip`, `video_validation.zip`, `questions/{train,validation,test}.json` (test only to confirm withheld answers), annotation zips (already on disk) | 18 943 730 947 | CC0 |
| Physion Dominoes | dynamics (40 538 620 810), readout (20 180 963 523), testing (39 333 918 824), glitchy-stimulus list | 100 053 504 259 | MIT (repository; no separate dataset statement) |

- Every sha256 is recorded in `data/*/manifest.json`, and every S3 multipart ETag matched.
- The 2 GiB Dominoes prefix from 2026-09-21 equals the first 2 GiB of the full archive.
- Not downloaded:
  - other Physion scenarios (out of scope);
  - CLEVRER test videos (answers withheld);
  - `derender_proposals.zip`.
- The Physion testing archive holds every stimulus twice. Only `hdf5s-redyellow` is used, the version that Physion passes to models.

## Design (frozen; full text in `plan.json`)

**Pixel carrier.**
- The E99 `SpatialSlotParser` is unchanged and uses the E99 optimiser recipe (12 epochs, batch 64, AdamW 1e-3).
- Training terms: presence, centres and kinds.
- Inputs: CLEVRER frames are resized to 160 × 240; Physion frames to 256 × 256.
- The parser feeds the deployment-adapter carrier (236 values, motion = centre difference / elapsed). At test time it sees pixels only.

**Slots.**
- CLEVRER: one slot per colour × material key. 41 % of videos repeat a key and are excluded, leaving 6285 train and 3132 validation videos.
- Physion: HDF5 object order (zone, target, probe, middle dominoes, distractors, occluders). The kind is the object's role.

**Parser centre targets.**
- CLEVRER: one RANSAC ground-plane homography (median residual 0.57 px).
- Physion: segmentation centroids.

**Time base.**
- CLEVRER: every frame.
- Physion: every 3rd native frame. The OCP input is native frames 0–36 (the physopt protocol), and rollouts run 90 carrier frames.

**Descriptions** (labels index the landing frame).
- Micro:
  - CLEVRER: collision at the frame.
  - Physion: pairwise contact in the carrier interval.
- Macro: the #113 templates, Δ-indexed (`DeltaMacroPredictor`):
  - CLEVRER enter–exit (debounced in-view set, endpoint comparison);
  - CLEVRER all-at-rest (D = max(1, Δ));
  - Physion red–yellow contact onset;
  - Physion settled (D = max(1, Δ)).

**Predictors.** Two families, 5 seeds each:
- hybrid;
- the #115 label-free binary micro arm.

Recipe: #112 LC as #117 item 5 adapted it, nine requests round-robin, 9000 updates. The curriculum runs 30 → 120 (CLEVRER) or 30 → 90 (Physion), +15 every 900 updates; truncation 45. Each member has 1 838 722 parameters.

**Policies.** J (nine requests) and J-cont (horizon-only, continuous requests) per member, both with the #101 / #114 recipe: prediction-error DP labels, λ = 1e-4, 2 DAgger rounds × 1800 steps over 1024 windows.

**Decisions.**
- CLEVRER predictive and counterfactual questions are answered zero-shot from the member's micro head on request-selected rollouts: gated contact ≥ 0.5, objects resolved from parsed attributes.
- Physion OCP uses the physopt readout protocol (StandardScaler + LogisticRegression, C grid logspace(−8, 8, 17), 5 folds) on simulated features. Two anchors are reported, observed and input, plus a macro-predicate rule.
- Equal compute: every arm of a member uses the same predictor, and J and J-cont share the policy recipe. MACs are reported per decision.

**Dispositions.**
- Primary, per benchmark: J − J-cont under the #114 rule (margin 0.02).
- Secondary: J − F-k, and label-free minus hybrid.
- Guards G1–G6.

## Results

**Guards.** G1–G6 pass.
- G1: CLEVRER scene sets are disjoint, and no Physion full parameter set is shared across splits. Six fixed generator layouts (for example `4midRM1_boxroom`) do recur across splits; they are reported as information.
- G5: the pixels-only spot re-parse and re-score pass.

**Training.** 20 of 20 predictors trained with no skipped update; 40 of 40 policies trained. Compute was 12 010 GPU s (3.3 GPU-h); wall time from freeze to publication was about 1.6 h on 4 GPUs.

**Parser fidelity** (pixels vs annotation targets, held-out split):
- CLEVRER validation: presence F1 0.993, centre error 16.8 px (of 480), kind accuracy 0.984.
- Physion testing: presence F1 0.920, centre error 45.9 px (of 256), kind accuracy 0.885.

Physion localisation is poor even on the training split (45.9 px). The E99 recipe's small smooth-L1 centre term gives little localisation pressure on these scenes.

### Rollout error, hybrid, pixel carrier (position MSE vs the parsed carrier, last endpoint)

| request | CLEVRER t = 120 | Physion t = 90 |
| --- | --- | --- |
| F-1-continuous | 0.163 [0.152, 0.176] | 0.0430 [0.0327, 0.0535] |
| F-1-micro | 0.164 [0.153, 0.176] | 0.0460 [0.0343, 0.0591] |
| F-1-macro | 0.180 [0.163, 0.197] | 0.0660 [0.0482, 0.0894] |
| F-5-continuous | 0.109 [0.104, 0.115] | 0.0227 [0.0184, 0.0272] |
| F-5-micro | 0.109 [0.104, 0.114] | 0.0231 [0.0186, 0.0278] |
| F-5-macro | 0.108 [0.103, 0.112] | 0.0256 [0.0199, 0.0320] |
| F-15-continuous | 0.0996 [0.0958, 0.104] | 0.0143 [0.0108, 0.0182] |
| F-15-micro | 0.0993 [0.0956, 0.103] | 0.0145 [0.0109, 0.0187] |
| F-15-macro | 0.0974 [0.0944, 0.101] | 0.0146 [0.0110, 0.0187] |
| J | 0.157 [0.147, 0.166] | 0.0235 [0.0188, 0.0285] |
| J-cont | 0.162 [0.152, 0.173] | 0.0252 [0.0206, 0.0303] |

- All nine requests, (15, macro) included, now have pixel-carrier rows on both benchmarks; the ICLR table has "—" for that cell. Every endpoint, the label-free family and the annotation-truth errors are in `findings.md` and `summary.json`.
- Coarser Δ gives lower error at the long endpoint. J (mostly Δ = 1 by its DP labels) lies close to F-1. J / J-cont at the last endpoint: log ratio −0.035 [−0.065, −0.006] on CLEVRER and −0.10 [−0.18, −0.03] on Physion (DESCRIPTIVE).

### Downstream accuracy vs MACs (hybrid; frozen readouts)

| arm | CLEVRER option acc. | MACs / question | Physion OCP acc. | MACs / trial |
| --- | --- | --- | --- | --- |
| F-1-continuous | 0.525 [0.520, 0.530] | 6.25e7 | 0.560 [0.487, 0.632] | 1.32e8 |
| F-5-continuous | 0.525 [0.520, 0.530] | 2.09e7 | 0.547 [0.473, 0.619] | 2.64e7 |
| F-15-continuous | 0.525 [0.520, 0.530] | 1.39e7 | 0.529 [0.455, 0.603] | 8.79e6 |
| F-15-macro | 0.525 [0.520, 0.530] | 1.40e7 | 0.529 [0.453, 0.604] | 8.99e6 |
| J | 0.525 [0.520, 0.530] | 5.73e7 | 0.517 [0.445, 0.589] | 3.44e7 |
| J-cont | 0.525 [0.520, 0.530] | 6.27e7 | 0.539 [0.467, 0.612] | 4.72e7 |
| Physion protocol anchors | — | — | observed 0.627 [0.549, 0.704]; input 0.533 [0.460, 0.605] | — |

All eleven arms (the six above plus the five other fixed requests), together with the predictive / counterfactual splits and the label-free arm, are in `findings.md`.

### Dispositions (frozen rule)

| benchmark | J − J-cont | token |
| --- | --- | --- |
| CLEVRER | −0.0001 [−0.0002, 0.0001] | `not_supported_by_this_experiment` |
| Physion-Dominoes | −0.021 [−0.068, 0.025] | `readiness_or_precision_insufficient` |

Secondary results:
- CLEVRER: every J − F-k is `not_supported_by_this_experiment`.
- Physion, J − F-1-{continuous, micro, macro}: `not_supported_by_this_experiment`.
- Physion, J − F-5-* and J − F-15-*: `readiness_or_precision_insufficient`.
- Physion, label-free J − hybrid J: +0.035 [−0.015, 0.087], `readiness_or_precision_insufficient`.
- CLEVRER QA for the label-free arm: `unsupported` (its micro codes carry no collision semantics).

## Reading

- **The CLEVRER token reflects a readout at its floor, not evidence about J.** The frozen 0.5 threshold on the gated micro head never fires. The largest value on any validation video is 0.051, and annotated collisions have a median of 0.013. The max-over-frames existence rule also admits single-frame phantom objects: 362 against 1911 real in 400 videos, which leaves 38 % of choices unresolved. Every arm therefore makes identical decisions.
  - The EXPLORATORY repaired readout (amendment A2) uses a 5-frame existence rule and thresholds fitted on the training split. Every arm still sits at 0.532 option accuracy and 0.50 predictive accuracy.
  - The micro head trained on these pixel carriers does not detect instantaneous collisions.
- **Physion OCP** sits near chance for every rollout arm (0.51–0.57). The observed-protocol anchor reaches 0.627, which shows the parsed carrier holds some OCP signal. The rollouts do not preserve it, and the macro OCP-onset predicate never fires (exploratory fitted threshold: about 0.50).
- The description levels are learnable as heads and the requests run on pixel carriers, but on these two benchmarks no request choice changes the downstream decision. The binding limits are parser localisation (Physion) and the sparse-event micro head (CLEVRER).

## Disclosures

- **A1 (post-freeze, before any predictor update).** The Physion testing frames are stored at 512 × 512, but the frozen parser input is 256 × 256. They are now resized with a box filter, and the segmentation threshold scales with the frame area. Testing labels were re-derived. The first `parse` launch had failed with a typed shape error.
- **A2 (post-score, exploratory only).** See *Reading*. The frozen code paths and tokens are unchanged.
- **Uncommitted first freeze attempt.** Its G1 digest covered initial layouts only. The full parameter sets were adopted, then the smoke was re-run and the plan frozen. Nothing had trained.
- Each stage was launched once, with direct worker PIDs recorded in `launches.json`. Two stages were rerun:
  - `parse`, after A1; the CLEVRER outputs from the first run were kept;
  - `publish`, after A2.
- No typed failure was retried.
- Hardware: 4 × RTX 5090 instead of the #85 RTX 3090.
