# Issue #130: DINO-WM (#127 arms C1-dino and C1-car) on NovPhy (reproduction, guards, and the frozen disposition)

Parent: #127 (external baselines program). Ticket: [#130](https://github.com/Sino-Huang/NovPhy/issues/130). The
binding shared rules of #85 and the #127 shared protocol (`scripts/external_baselines.py`, `eb`) apply. The Gate-B
(#108), #101, #104, #109, #110, #112, #117, #123 and #129 records and stores are read-only inputs. Nothing here edits
the ICLR 2026 submission.

| | |
| --- | --- |
| Paper and code | Zhou, Pan, LeCun, Pinto, *DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning*, ICML 2025, arXiv 2411.04983v2; [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) @ `0a9492f` (MIT); frozen encoder [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) @ `7764ea0` (Apache-2.0), official `dinov2_vits14_pretrain.pth` (sha256 `b938bf1b…`) |
| Question | Q3: does a world model on frozen DINOv2 features rank launches better than U (C1-dino), and does the same predictor on E99 carriers isolate the input features (C1-dino − C1-car)? |
| Runner | `scripts/run_issue_130_dino_wm.py` (selection / selection-readouts / smoke / prepare / train / readouts / freeze-readouts / score / score-novelty / publish / validate) |
| Modules | `world_model/baselines/dino_wm/novphy.py` (NovPhy adapter: the predictor path of `VWorldModel` and `Trainer.init_optimizers` ported line for line, DINOv2 preprocessing, carrier slot tokens, the R1 patch readout); `world_model/baselines/dino_wm/vendor/` (official `models/vit.py` `ViTPredictor` and `models/proprio.py` `ProprioceptiveEmbedding`, `LICENSE`; per-file upstream and vendored sha256 in `VENDOR.json`; one listed edit: the causal mask is a non-persistent buffer instead of `.to('cuda')`) |
| Fidelity / features | `scripts/b127_dino_wm_fidelity.py` (Tier E: the official `plan.py`, released PushT checkpoint; resumable at MPC-iteration boundaries; post-freeze amendments A1, A2), `scripts/b127_dino_wm_port_reference.py` (G-port), `scripts/b127_dino_wm_features.py` (DINOv2 feature cache, C1-dino readiness smoke) |
| Tests | `tests/test_issue_130_dino_wm.py` (19 tests, CPU) |
| Artifacts | `.local-artifacts/issue-130-dino-wm-v1/`: `plan.json`, `smoke.json`, `selection.json`, `decisions.md`, `findings.md`, `summary.json`, `compute.json`, `records-manifest.json`, `validation.json`, `readouts-manifest.json`, `gport/gport.json`, `fidelity/{band,gfid}.json`, `readiness/{readiness.json,r1.pt}`, `features/` (indexes and `MANIFEST.json`), `novelty/`, `envs/b127-dino_wm.txt`, `ledger/`. Checkpoints, the 207 GB feature cache (local NVMe), records and the external downloads (`data/external/dino_wm/`, `MANIFEST.json`) stay local; records and readouts are hashed in the manifests |
| Freeze | pre-freeze implementation `c657f8f6`; `plan.json` frozen 2026-10-25T11:38:19Z (22:38 AEDT), commit `a683498d` (sha256 `07dad64c…`, 116 pins), after a passed smoke and before any main-seed training or evaluation-split read. Post-freeze amendments A1 (`009a87cc`) and A2 (`4048e33b`) of `scripts/b127_dino_wm_fidelity.py`, infrastructure only, registered in `scripts/post_freeze_amendments.py` |
| Readout freeze | `readouts-manifest.json` frozen 2026-10-25T12:15:27Z (23:15 AEDT; sha256 `606c4d96…`), before any evaluation anchor was read; it also pins C1-dino's readiness record |
| Scoring | once, in a process that did not train: scheduler run `130-score#1`, 2026-10-27 04:15:21–04:25:20 AEDT, exit 0 (3 arm-seeds), after the Tier-E G-fid record was decided; P-novelty pass `130-score-novelty#1`, 04:25:20–04:30:21, exit 0 (no arm passed its guards; references only) |
| Validation | `python -u -m scripts.run_issue_130_dino_wm --validate`, scheduler run `130-validate#1`, 2026-10-27 04:35:20–04:40:20 AEDT, **exit 0**: 10 records hash-checked, C1-car seed 20260908 re-scored on every inventory with exact row equality, the publication re-derived twice byte-identically. `validation.json` binds `records-manifest.json` sha256 `a18d25c0…`. Publication commits `b4b0df9a`, `f3be7485` |

Engine time is zero seconds. The primary metric is the Gate-B pooled grid AUC over the 311 mixed grid cells × 3 seeds
of the #104 sealed #109 evaluation split, tie-free cost, under `deterministic_scoring()`. Intervals are level-clustered
percentile bootstraps (10 000 draws, PCG64 7201) and are DESCRIPTIVE. The margin is δ = 0.02.

## Outcome

**Disposition 4 for both arms: per-contrast tokens only. No scoped ranking claim either way.**

- **C1-dino (primary, program set) stopped before any predictor training** with the typed reason
  `readiness_ceiling_below_threshold`. Its R1 readout reaches a policy-split ceiling of 0.5948 against the frozen 0.65
  (G-readout). By the pre-registered rule it is not trained or scored. Every token involving it is
  `readiness_or_precision_insufficient`, and as a dropped program-set member it **vetoes the #127 program-level claim**.
- **C1-car (secondary)** was trained and scored. Every global guard passes (G-isolation, G-chronology including the
  novelty pass, G-scoring-policy, G-records). G-port and G-fid pass: the released PushT checkpoint gives success
  **0.92** in the frozen band [0.82, 0.98].
- C1-car **fails G-action on every seed**: endpoint action sensitivity 0.00044 / 0.00045 / 0.00163 against the
  frozen 0.05. G-rank, G-nonfinite and G-trained pass on every seed. By the frozen rule every C1-car token is
  `readiness_or_precision_insufficient`.
- Descriptively, C1-car ranks the grid at 0.5251 [0.5093, 0.5411] against U's 0.7444 [0.7269, 0.7620]; C1-car − U is
  −0.2193 [−0.2412, −0.1985]. This carries no token.
- **C1-dino − C1-car cannot be computed** (C1-dino was not run), so Q3's within-family question has no answer from
  this child.
- The G-action threshold was frozen on #123's latent states. In carrier space the engine truth itself reaches only
  0.0395, so the frozen threshold is unattainable there by any model (#127 decision; see "G-action calibration
  context"). C1-car's endpoints move with the launch about 2 % as much as the true endpoints do. The frozen guard, its
  failure and the disposition stand.

## Design summary

The design is frozen in `plan.json`; every implementer decision was posted on #130 before the step it governs and is in
`decisions.md` (see "Decision log").

**Model.** The official DINO-WM predictor, used unchanged through the vendored code
([adapter decision](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382576605)):

- `ViTPredictor` (depth 6, 16 heads × 64, MLP 2048, dropout 0.1), a block-causal mask over frames, history 3
  (`num_hist` 3, `num_pred` 1);
- `concat_dim = 1`: the action embedding (official `ProprioceptiveEmbedding`, Conv1d k = 1 → 10-d) is tiled onto every
  token; there is no input projection; no proprio channel (NovPhy has no proprioception); no decoder (the official
  decoder loss sees only `z.detach()`, so predictor and action-encoder gradients are unchanged);
- loss: the official `z_loss`, MSE over every non-action channel; optimizers: the official `Trainer.init_optimizers`,
  AdamW(predictor) and AdamW(action encoder), no scheduler, no clipping, fp32, batch 32;
- launch: DINO-WM's native frame-skip action concatenation. The per-frame action is `eb.launch5` at frame 0 and the
  zero no-op at every other frame, so step 0 = `[launch5, 0 × 20]` and every later step = 0 (25-d). Frame-skip 5;
  45 steps reach frame 225. Ranking rolls out with the official `rollout` from the single anchor frame;
- score: tie-free `EndpointCosts` of the frame-225 prediction (C1-car: the decoded carrier; C1-dino: R1 of the
  predicted patch features).

**Inputs.**

- **C1-dino:** DINOv2 ViT-S/14 `x_norm_patchtokens` of the 320 × 240 frames. DINO-WM's own `x/255` and
  `Normalize(0.5, 0.5)`, then an aspect-preserving bilinear antialiased resize of the full frame to 238 × 322
  (17 × 23 = 391 patches at native scale) instead of DINO-WM's `Resize(224)` + `CenterCrop(224)` → 196, which would
  crop about 12 % of the width on each side and shrink pigs of about 4 px further. Features cached as fp16.
- **C1-car:** each E99 carrier becomes 18 slot tokens: the slot's 13 features plus the carrier's 2-d header tiled onto
  every slot (as `concat_dim = 1` tiles frame-level vectors onto every patch), z-scored with fit-split statistics
  (15-d tokens; predictor width 25). The prediction is decoded back to a carrier (slot features, the mean of the 18
  header copies, de-normalised).

**Arms** (fit split only, seeds 20260908/09/10):

| arm | input | schedule | role | trainable parameters |
| --- | --- | --- | --- | --- |
| C1-dino | 391 DINOv2 patch tokens × 384 | — (stopped at the readiness smoke) | primary, program set | (19 855 570 as built; never trained) |
| **C1-car** | 18 carrier slot tokens × 15 | selected `car-release`: 123 858 steps × 32, predictor / action-encoder lr 5e-4 | secondary | 1 243 648 |

Training windows are 4 frames at stride 5 starting at frame 5k, fully inside [0, min(225, end)]: 591 974 windows from
13 919 fit shots (601 shots end before frame 225); 1 epoch = 18 500 steps. As in the official `TrajSlicerDataset`,
the window order is one permutation reused every epoch and the last batch is partial.

**Selection** ([decision](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6386706483), posted 03:32:55
AEDT on 25 Oct, before the selection jobs launched at 03:38:48; #127 rule: policy-split pooled grid AUC of the primary
score, 94 mixed policy cells, seed 20260908; the selected run is reused as main seed 20260908):

| configuration | predictor lr | action-encoder lr | budget (steps) | policy grid AUC |
| --- | --- | --- | --- | --- |
| car-lr5e-4 (released code and the PushT checkpoint's `hydra.yaml`) | 5e-4 | 5e-4 | 100 epochs (1 850 000) | 0.4811 |
| car-lr5e-5 (the paper's Table 12 predictor lr) | 5e-5 | 5e-4 | 100 epochs (1 850 000) | 0.5189 |
| **car-release** (the samples the released PushT checkpoint carries: its epoch field 2 × its 1 981 721 windows) | 5e-4 | 5e-4 | 3 963 442 samples (123 858) | **0.5409** (selected) |

With no LR scheduler, `car-release` is exactly the first 123 858 steps of `car-lr5e-4` (same seed and window order).
The full 100-epoch run ranks the policy grid worse than its own early state. U had no policy-split tuning of this kind
(#127), disclosed.

## G-fid and G-port

**G-fid, Tier E: PASS.** The released PushT checkpoint (OSF `bmw48` `checkpoints/outputs.zip`, sha256 `425b0d8c…`;
`outputs/pusht/checkpoints/model_latest.pth`, sha256 `e909f0ce…`) evaluated by the official `plan.py` with the
README's PushT command (`plan_pusht.yaml`: 50 cases, seed 99, `goal_source=dset`, `goal_H=5`, MPC-CEM with horizon 5,
300 samples, top-30, 30 CEM steps) gives **success 0.92 (46 / 50)** against the paper's 0.90 (Table 1; Table 8 row MPC)
and the band **[0.82, 0.98]** (±2 binomial SE on 50 cases) frozen before the run
([band](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382394238); `band.json` sha256 `247747e2…`;
`gfid.json` sha256 `020f627e…`).

- Our only protocol setting is `planner.max_iter = 12`. The released config sets `max_iter: null` ("unlimited"), which
  never ends unless all 50 cases succeed; 12 iterations × 5 steps × frame-skip 5 = 300 env steps, the official gym
  registration's `max_episode_steps` for `pusht`. Frozen in `band.json` before the run.
- The cumulative success after MPC iterations 1, 2, …, 12 is 0.90, then 0.92 at every iteration from 2 to 12.
- The official stack ran from the clone in env `b127-dino_wm` (Python 3.10, torch 2.7.1 + cu128, RTX 5090; the
  official env's torch 2.3 / cu121 cannot run on sm_120;
  [P-license](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382422982)). Infrastructure shims, none in
  planning, the model or the metric: empty `mujoco_py` / `d4rl.offline_env` modules (imported only by the PointMaze
  registration), `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (the release pickles whole modules), `WANDB_MODE=disabled`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and from A2 the evaluator's no-decoder branch (below).
- The run was made resumable at MPC-iteration boundaries before the freeze
  ([decision](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6383358026)): the official loop body
  verbatim plus a checkpoint of the full loop state and every RNG state after each iteration. A stopped and resumed CPU
  run (2 cases, 2 iterations) gives a byte-identical `logs.json` and identical planned actions (CPU test).
- The deciding run (`fidelity/runs/tier-e-3`) needed four launches; see "Tier-E runs and infrastructure crashes".

**G-port: PASS.** On fixed CPU float32 inputs, the official PushT configuration (196 patches × 384, proprio 4 → 10,
action 2 × 5 → 10, the released `hydra.yaml`, decoder off) built by the official `hydra.utils.instantiate` path, the
official `VWorldModel` forward with DINOv2 in the loop and the official `Trainer.init_optimizers`, against the adapter:
forward, loss, gradients, AdamW moments and two optimizer steps (DINO-WM has no scheduler, so both steps run at the
positive base LR 5e-4). Loss 3.008074 / 2.600609 on both sides; gradient tensors max relative L2 3.7e-7; parameters
max |Δ| 2.4e-6, 99.996 % within 1e-7 (#123 rule ≥ 99.9 %); optimizer defaults equal; vendor hashes match
(`gport.json` sha256 `7db54ec7…`). The reference ran on torch 2.7.1, the adapter on torch 2.13; the adapter's own
DINOv2 features differ from the official encoder's by at most 6.0e-5 (tolerance 1e-4). A diagnostic arm that takes
the same steps on the adapter's own features misses only the 99.9 % parameter-share criterion on step 2 (99.875 %,
float round-off of those features), recorded in `gport.json`; the gate is decided on the official features.

## C1-dino: readiness smoke and typed stop

Before any predictor training, the #127 readiness smoke measured whether frozen ViT-S/14 patch features can express the
cost at all ([design](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382576605),
[result](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6383216316)):

- **R1** (`novphy.PatchReadout`): 19 learned queries (header + 18 slots) cross-attend over the 391 patch tokens with a
  learned 2-D position embedding, so a pig inside one 14-px patch can be attended to and located (1 150 223
  parameters). Fitted on the fit split only: 273 692 frames (every 15th position of 13 919 shots), E99 carrier targets
  of the same frames, the #123 probe schedule (AdamW 1e-3, batch 128, 25 epochs, 53 450 steps, final z-scored MSE
  0.0276), seed 20260908, under `deterministic_scoring()`.
- **Ceiling:** the policy grid pooled AUC (94 mixed cells, 1 504 candidates) of the tie-free cost of R1(true
  min(225, end) frame features) vs R1(policy anchor features), the #123 `policy_aucs` ceiling.

| quantity | value |
| --- | --- |
| **R1 ceiling** | **0.5948** (threshold 0.65) |
| the same ceiling on the true E99 carriers (the best any carrier readout can reach) | 0.7223 |
| R1 vs E99 on the policy endpoints: pig presence / pig centre / all-column MAE | 0.0018 / 0.0265 / 0.0360 |

0.5948 < 0.65, so C1-dino stopped with `readiness_ceiling_below_threshold` before training. The readout and the
threshold were fixed before the smoke and were not changed after it. R1 reads pig presence almost exactly but places
the pig centre to about 8–9 px at 320 px width, and the cost's displacement term depends on the centre [INFERENCE:
frozen ViT-S/14 patches at native scale localise a ~4-px pig less precisely than the E99 parser]. The readiness record
(`readiness/readiness.json`, sha256 `7a6da66d…`) is pinned in the plan and the readouts manifest. Feature cache:
687 038 fit frames (frame-skip-5 positions to frame 225 plus the R1 probe frames) and 3 600 policy frames (120 anchors,
3 480 true endpoints), 207 GB fp16, hashed in `features/MANIFEST.json`.

## Results

### Guards

| arm | seed | G-rank (mean tied-pair share; fully tied levels) | G-action (≥ 0.05) | G-nonfinite | G-trained | policy grid AUC (fit/policy only) |
| --- | --- | --- | --- | --- | --- | --- |
| C1-car | 20260908 | pass (0.0598; 0.0 %) | **fail (0.00044)** | pass (0.0) | pass | 0.5409 |
| C1-car | 20260909 | pass (0.0295; 0.2 %) | **fail (0.00045)** | pass | pass | 0.5381 |
| C1-car | 20260910 | pass (0.0570; 1.0 %) | **fail (0.00163)** | pass | pass | 0.5127 |
| C1-dino | — | — | — | — | — (typed stop) | G-readout **fail** (ceiling 0.5948) |

G-collapse does not apply to C1-car (its predictor state is the z-scored carrier itself, linearly decoded; no learned
latent; decision in the selection post), and G-readout does not apply (no readout). G-port and G-fid pass. Global
guards: G-isolation, G-chronology (plan 2026-10-25T11:38:19Z ≤ readouts 12:15:27Z ≤ evaluation anchors
2026-10-26T17:15:46Z ≤ first record 17:17:19Z; novelty anchors 17:25:59Z after the last primary record 17:20:23Z),
G-scoring-policy and G-records all pass.

### Grid rankers (311 mixed cells × 3 seeds)

| ranker | pooled AUC [CI] | per seed (08 / 09 / 10) | top-1 (chance 0.089) |
| --- | --- | --- | --- |
| U | 0.7444 [0.7269, 0.7620] | 0.7141 / 0.8011 / 0.7181 | 0.195 |
| C\* | 0.7834 [0.7676, 0.7992] | 0.8262 / 0.7334 / 0.7905 | 0.192 |
| P | 0.7203 [0.6965, 0.7442] | — | 0.180 |
| J | 0.7057 [0.6846, 0.7267] | 0.6690 / 0.7012 / 0.7470 | 0.160 |
| J_dec | 0.7460 [0.7277, 0.7637] | 0.7205 / 0.7933 / 0.7242 | 0.180 |
| **C1-car** | 0.5251 [0.5093, 0.5411] | 0.4997 / 0.5694 / 0.5063 | 0.042 |
| C1-dino | not run (typed stop) | — | — |

C1-car has no tied pairs and no nonfinite rows; P's tied-pair share is 0.002.

### Contrasts (grid, paired per (level, seed))

| contrast | estimate [CI] | token | reverse token |
| --- | --- | --- | --- |
| C1-car − U | −0.2193 [−0.2412, −0.1985] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| C1-car − C\* | −0.2582 [−0.2790, −0.2385] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| C1-car − P | −0.1952 [−0.2256, −0.1648] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| C1-car − J_dec | −0.2208 [−0.2424, −0.1998] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| C1-dino − U / − C\* / − P / − J_dec | n/a (not run) | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| **C1-dino − C1-car** (within family) | n/a: not computable, C1-dino not run | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |

Every token is the stop token: C1-car fails G-action and C1-dino is a typed stop. The C1-car estimates are reported as
measured; they license no statement in either direction.

### Wrong-anchor control (right − wrong, 480 fit anchors, grid)

| arm | right | wrong | right − wrong | ranking stability (Spearman) | token |
| --- | --- | --- | --- | --- | --- |
| C1-car | 0.5251 [0.5093, 0.5411] | 0.5353 [0.5292, 0.5413] | −0.0101 [−0.0259, +0.0053] | 0.1504 | `readiness_or_precision_insufficient` |

The interval of right − wrong contains zero: there is no evidence that C1-car's cost benefits from the correct anchor.

### Descriptive inventories (no tokens)

| inventory (cells) | U | C\* | P | J | J_dec | C1-car | C1-car − U |
| --- | --- | --- | --- | --- | --- | --- | --- |
| offset (228) | 0.6670 | 0.7748 | 0.5000 | 0.7522 | 0.6904 | 0.4798 | −0.1871 [−0.2332, −0.1425] |
| angle (234) | 0.6791 | 0.7590 | 0.8119 | 0.7765 | 0.6988 | 0.4842 | −0.1949 [−0.2267, −0.1630] |
| power (141) | 0.8058 | 0.8068 | 0.5000 | 0.8076 | 0.7943 | 0.5104 | −0.2954 [−0.3581, −0.2380] |

(C1-car − U: stop tokens.)

### Engine position error (#100 metric; DESCRIPTIVE)

Mean slot-centre MSE of the predicted carrier against the engine centres on present slots, every complete grid
candidate of every mixed grid cell, from the Gate-B anchors (member-clustered bootstrap):

| t | 15 | 30 | 60 | 150 | 225 |
| --- | --- | --- | --- | --- | --- |
| C1-car | 0.0529 | 0.0535 | 0.0526 | 0.0537 | 0.0578 [0.0533, 0.0625] |
| U | 0.0505 | 0.0501 | 0.0484 | 0.0459 | 0.0474 [0.0440, 0.0508] |
| C\* | 0.0547 | 0.0545 | 0.0543 | 0.0594 | 0.0670 [0.0617, 0.0723] |

### G-action calibration context (DESCRIPTIVE; changes nothing)

G-action (endpoint action sensitivity ≥ 0.05) was frozen on #123's latent states and is applied, as frozen, to the
236-d predicted endpoint carrier. #127 decided (posted on #127 and
[on #130](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6396584714)) that the frozen guard stands for
every child and that each carrier-space world model reports its sensitivity as a share of the engine truth's, from the
#129 fit-split calibration (`.local-artifacts/issue-129-slotformer-v1/analysis/g-action-calibration.json`, sha256
`8a6c52d8…`, pinned in the #130 plan):

| source | per seed (08 / 09 / 10) | share of engine truth |
| --- | --- | --- |
| engine truth: true carriers at min(225, end) of the 16 executed grid shots of 479 fit levels | — | 0.0395 (100 %) |
| U (Gate-B, #129 calibration) | — | 58 % |
| C\* | — | 85 % |
| B1 (#129) | — | 33 % |
| **C1-car** | 0.00044 / 0.00045 / 0.00163 | **1.1 % / 1.1 % / 4.1 % (mean 2.1 %)** |

The engine truth itself is below 0.05, so the frozen threshold is not attainable in carrier space even by a perfect
model. Independently of the threshold, C1-car's predicted endpoints are nearly launch-invariant (about 2 % of the true
endpoints' variation with the launch; B1 33 %, U 58 %), which is consistent with its chance-level ranking and its
wrong-anchor result. This is context only: the frozen guard, its failure and disposition 4 stand.

### P-novelty (#110 novelty pairs; DESCRIPTIVE)

([decision](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6386831073).) Not scored for C1-car (it fails
G-action; P-novelty scores each arm that passes its guards) or C1-dino (typed stop, never trained)
(`novelty/status.json`). The U / C\* / P novelty references were computed on the 257 mixed levels of the 455 in-scope
#110 levels and are hashed in `records-manifest.json`; `findings.md` lists them as reference rows only (no C1 row, no
tokens, never part of a disposition). G-chronology for the pass holds (novelty anchors read after the last primary
record).

## Disposition (δ rule)

- **C1-dino:** dropped (`readiness_ceiling_below_threshold`), disposition 4, every token
  `readiness_or_precision_insufficient`.
- **C1-car:** disposition 1 (C1-car ranks launches better than U, scoped) **no**; disposition 2 (U ranks launches
  better than C1-car, scoped) **no** — it requires every C1-car guard to pass; disposition 3 **no** — guards fail;
  **disposition 4: per-contrast tokens only**, every one `readiness_or_precision_insufficient`.
- **Within family:** C1-dino − C1-car is not computable; the input-feature question of Q3 is not answered by #130.
- **#127 program claim:** C1-dino is a program-set member that is dropped and fails a guard, so it **vetoes** the
  intersection–union claim. C1-car is secondary and not in the program set.
- **Budget safeguard:** does not fire (C1-car − U is not `supported`); C1-car's mean training compute is 1.53 × U's.

## Tier-E runs and infrastructure crashes (disclosed)

Every Tier-E launch evaluated the same frozen protocol; no stop, crash or relaunch depended on any success value, and no
`final_eval` value existed before the deciding run completed.

| # | scheduler run | directory | what happened | MPC iterations completed | decision |
| --- | --- | --- | --- | --- | --- |
| 0 | `130-gfid-tier-e#1` (2026-10-24 20:15–20:29) | `tier-e-1` | stopped by us before its first MPC iteration: measured ~12.3 h per run did not fit the window; non-resumable then | 0 | [schedule](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382562140) |
| 1 | `130-gfid-tier-e-r#1` (24 Oct 22:00 – 25 Oct 03:31) | `tier-e-2` | CUDA OOM: an orphaned #128 A4 process (18 GB) on GPU 2 | 0 | [P-crash](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6386786676), [addendum](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6386817726) |
| 2 | `#2` (25 Oct 21:55 – 26 Oct 02:00) | `tier-e-3` | CUDA OOM in the plot-only VQ-VAE decode of the evaluator's rollout videos, beside a co-scheduled #131 job (1.5 GB) | 2 (checkpoint) | [P-crash](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6400401642); amendment **A1**: decode one case at a time ([A1](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6400429696)) |
| 3 | `#3` (26 Oct 03:50 – 04:55) | `tier-e-3` (resumed) | CUDA OOM in the same plot path on the exclusive card (CEM peak ~28.4 GiB of 31.4 GiB) | 0 more | [P-crash](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6402614864); amendment **A2**: the evaluator's official no-decoder branch |
| 4 | `#4` (26 Oct 18:00 – 27 Oct 04:15) | `tier-e-3` (resumed from iteration 2) | completed, exit 0: **0.92**, in band | 12 | — |

- **A1** (`009a87cc`): the plot-only decode of the evaluator's videos runs one case at a time instead of four (same
  tensors). It did not remove the root cause.
- **A2** (`4048e33b`): there is no official switch for the videos (`n_plot_samples` slices after the full decode); the
  official evaluator draws them iff `self.wm.decoder is not None` (`planning/evaluator.py`). The evaluator's world
  model is given `decoder = None`, the official branch of a model trained without a decoder. Planning (encode /
  predict), the objective, the env rollouts, the PushT `eval_state` success metric and the RNG streams never use the
  decoder, so the iteration-2 checkpoint resumed unchanged. Moving the decode to the CPU was rejected: it runs after
  every CEM step (`eval_every: 1`) and would have added hours for videos nothing reads.
- Both amendments are registered in `scripts/post_freeze_amendments.py` (A2 `replaces` the frozen and the A1 bytes;
  `tests/test_issue_127_amendments.py`), and `load_plan` accepts them.
- Resume across processes is exact in state (CPU test: byte-identical). On the GPU the restored state is exact, but GPU
  kernels are not bitwise deterministic between any two runs, interrupted or not.
- During the run-3 diagnosis a directory listing of `tier-e-3` showed the official plot file names
  (`plan{iteration}_{case}_success.mp4`) of iterations 0 and 1. No decision used them; the band and protocol had been
  frozen and posted on 24 Oct.

## Deviations and disclosures

1. **Tier-E launches and post-freeze amendments A1 / A2** (above). The `max_iter = 12` cap is ours (the released
   `null` never terminates below 100 % success), frozen before any run.
2. **Environment (P-route).** The official env (Python 3.9, torch 2.3) does not run on sm_120; the fidelity and the
   G-port reference ran the official code unmodified in `b127-dino_wm` (torch 2.7.1 + cu128; package list in
   `envs/`), the adapter in `novphy` (torch 2.13). The infrastructure shims are listed in `gfid.json`.
3. **Learning rates.** The paper's Table 12 lists predictor lr 5e-5; the released config and the PushT checkpoint use
   5e-4. Both were selection candidates; the third candidate carries the released checkpoint's sample budget.
4. **Selected configuration is a prefix.** Without a scheduler, `car-release` equals the first 6.7 % of `car-lr5e-4`'s
   steps; the full 100-epoch run (0.4811) ranked the policy grid below it (0.5409). Selection followed the frozen rule.
5. **Training windows** start only at frame-skip phase 0 (native DINO-WM slices windows at every frame offset); no
   proprio channel; no decoder; `normalize_action` off (`launch5` is already U's normalised launch; dataset z-scoring
   over >99 % no-op frames would shift the no-op away from zero).
6. **C1-dino preprocessing and R1.** Full-frame aspect-preserving resize at native scale (not DINO-WM's crop), and a
   slot-query attention readout instead of #123's vector MLP; both fixed and posted before the readiness smoke.
7. **G-collapse not applied to C1-car** (no learned latent; posted before the selection runs).
8. **Equal tuning allowance.** C1-car chose among three configurations on the policy split; U had no such tuning
   (#127).
9. **Scheduling.** Selection jobs shared GPU 2 with each other (host-gathered batches, 1.27 GB each); the two main
   seeds ran together on GPU 1. The scheduler's cron did not tick between 24 Oct 22:00 and 25 Oct 03:38 (restarted by
   the orchestrator). None of this touched data, models or scores.
10. **Novelty mode and the G-action calibration field** were added before the freeze; the training path did not change
    while selection runs were in flight.

## Compute

GPU seconds are each process's wall time on its GPU (`eb.mark`; each arm-seed's training counted once from its
`report.json`). Jobs that shared a GPU ran slower than the device time the work would need alone.

| item | process GPU-h |
| --- | --- |
| Tier-E fidelity, all launches (run 0 0.24, crashed runs 5.52 + 4.06 + 1.03, deciding run 10.24) | 21.09 |
| selection car-lr5e-4 / car-lr5e-5 / car-release (= C1-car seed 20260908) | 7.07 / 7.07 / 0.71 |
| C1-car seeds 20260909 / 20260910 | 0.47 / 0.47 |
| DINOv2 feature cache / C1-dino readiness smoke | 0.17 / 0.81 |
| readouts (3), scoring (3), score-novelty, selection readouts, smoke | 0.14 |
| **recorded** (`compute.json`) | **38.02** |
| unrecorded: publish 69 s, validate 226 s | 0.08 |
| **active total** | **≈ 38.1 (child cap 88; smoke projection 30.8 before the crashes)** |

About 10.85 GPU-h of the total is infrastructure loss (the stopped run and three crashed launches).

| | C1-car | U | C\* |
| --- | --- | --- | --- |
| trainable parameters | 1 243 648 | 1 837 690 | 1 862 396 |
| training GPU-s per seed (08 / 09 / 10; process) | 2 555 / 1 701 / 1 703 | mean 1 294 | mean 1 024 |
| inference MACs per candidate (to frame 225; 45 predictor calls over a 3-frame window of 54 tokens) | 4.48 G | 20.99 M | 27.26 M |
| wall per decision (16 candidates, GPU, median) | 0.074 s | — | — |

Engine seconds: 0. No scope was cut. The #127 program budget is 400 GPU-h; #130's share was budgeted at 53–88 GPU-h
and used about 38.

## Decision log

All posted on #130 before the step each governs, and appended to `decisions.md`:

- [P-fid: Tier-E band and protocol](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382394238)
- [P-license: vendoring](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382422982)
- [P-other: Tier-E in a full window](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382562140)
- [P-other: adapter design, feature cache, readiness rule](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382576605)
- [P-other: C1-dino readiness stop](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6383216316)
- [P-other: resumable Tier-E](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6383358026)
- [P-other: C1-car runner, schedule, selection, guards, budget](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6386706483)
- [P-crash 1](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6386786676) and [addendum](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6386817726)
- [P-novelty](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6386831073)
- [#127 G-action decision, as it applies here](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6396584714)
- [P-crash 2](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6400401642), [A1](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6400429696)
- [P-crash 3 and A2](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6402614864)

## Reproduction

```
conda activate novphy && source env.sh
M="python -u -m scripts.run_issue_130_dino_wm"
D=/srv/home/shua0171/miniconda3/envs/b127-dino_wm/bin/python
# Phase 0
CUDA_VISIBLE_DEVICES="" $D scripts/b127_dino_wm_port_reference.py --reference .local-artifacts/issue-130-dino-wm-v1/gport
CUDA_VISIBLE_DEVICES="" python scripts/b127_dino_wm_port_reference.py --compare .local-artifacts/issue-130-dino-wm-v1/gport   # exit 0
python scripts/b127_dino_wm_features.py --plan
CUDA_VISIBLE_DEVICES=g python -u scripts/b127_dino_wm_features.py --extract     # GPU, scheduler; resumable
CUDA_VISIBLE_DEVICES=g python -u scripts/b127_dino_wm_features.py --readiness   # C1-dino readiness smoke
CUDA_VISIBLE_DEVICES=g $D -u scripts/b127_dino_wm_fidelity.py --run             # Tier E; resumable; fidelity/gfid.json
# selection, smoke, freeze
CUDA_VISIBLE_DEVICES=g $M --selection --gpu q        # q = 0, 1, 2; night window only (scheduler)
$M --selection-readouts && OMP_NUM_THREADS=4 $M --smoke --device cpu && $M --prepare
# main runs (scripts/gpu_window_issue_130_main.json)
CUDA_VISIBLE_DEVICES=g $M --train --gpu q            # q = 0, 1
CUDA_VISIBLE_DEVICES=g $M --readouts --gpu 0 && $M --freeze-readouts
CUDA_VISIBLE_DEVICES=0 $M --score                    # once; waits for a decided fidelity/gfid.json
CUDA_VISIBLE_DEVICES=0 $M --score-novelty
$M --publish
CUDA_VISIBLE_DEVICES=0 $M --validate                 # exit 0
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 pytest tests/test_issue_130_dino_wm.py   # 19 passed
```
