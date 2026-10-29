# Issue #137: DIAMOND-px (#127 coverage arm T5) on NovPhy (reproduction, guards, and the frozen disposition)

Parent: #127 (external baselines program). Ticket: [#137](https://github.com/Sino-Huang/NovPhy/issues/137). The
binding shared rules of #85 and the #127 shared protocol (`scripts/external_baselines.py`, `eb`) apply. The Gate-B
(#108), #101, #104, #109, #112, #117 and #123 records and stores are read-only inputs. Nothing here edits the ICLR 2026
submission. Result comment: https://github.com/Sino-Huang/NovPhy/issues/137#issuecomment-6505327166. T5 is a **coverage arm** (rule P-coverage, funded by its measured projection,
[program decision](https://github.com/Sino-Huang/NovPhy/issues/127#issuecomment-6436095504)): it is secondary,
reported, and never part of the #127 program claim.

| | |
| --- | --- |
| Paper and code | Alonso, Jelley, Micheli, Kanervisto, Storkey, Pearce, Fleuret, *Diffusion for World Modeling: Visual Details Matter in Atari*, NeurIPS 2024, arXiv 2405.12399; [eloialonso/diamond](https://github.com/eloialonso/diamond) `main` @ `5bcd159` (MIT); released checkpoints on Hugging Face `eloialonso/diamond` @ `5d4abca` |
| Runner | `scripts/run_issue_137_diamond.py` (data / readiness / project / timing / selection / selection-readouts / smoke / prepare / train / readouts / freeze-readouts / score / score-novelty / publish / validate) |
| Modules | `world_model/baselines/diamond/novphy.py` (NovPhy adapter); `world_model/baselines/diamond/vendor/` (34 files of `main` @ 5bcd159, verbatim, no edits; per-file upstream and vendored sha256 in `VENDOR.json`) |
| Fidelity | `scripts/b127_diamond_fidelity.py` (Tier E: released agents, the official evaluation protocol, CPU), `scripts/b127_diamond_port_reference.py` (G-port reference and comparison) |
| Tests | `tests/test_issue_137_diamond.py` (21 tests, CPU); `tests/test_issue_127_amendments.py::test_issue_137_runner_amendment` |
| Artifacts | `.local-artifacts/issue-137-diamond-v1/`: `plan.json`, `smoke.json`, `selection.json`, `projection.json`, `timing.json`, `decisions.md`, `findings.md`, `summary.json`, `compute.json`, `records-manifest.json`, `validation.json`, `readouts-manifest.json`, `readiness/{readiness.json,readout.pt}`, `gport/gport.json`, `fidelity/{band,gfid}.json` (+ `fidelity/run/`), `hf-checkpoints.json`, `envs/b127-diamond.txt`, `ledger/`. Checkpoints, readouts, records, the 64 × 64 frame cache (`~/.cache/novphy-issue-137-diamond-frames/`) and the external downloads (`data/external/diamond/`, `MANIFEST.json`) stay local; records and readouts are hashed in the manifests |
| Freeze | pre-freeze implementation `9a0c96ea` + `9a9abbd6`; `plan.json` frozen 2026-10-28T10:36:53Z (21:36 AEDT), commit `7677a2fd` ([P-freeze](https://github.com/Sino-Huang/NovPhy/issues/137#issuecomment-6466783445)), sha256 `178b1d40…`, 147 pins, after a passed smoke and before any main-seed training or evaluation-split read. One post-freeze amendment, #137 A1 (reporting only, after scoring; see "Deviations and disclosures") |
| Readout freeze | `readouts-manifest.json` frozen 2026-10-28T17:35:26Z (04:35 AEDT, 29 Oct; sha256 `87640b5a…`), before any evaluation anchor was read |
| Scoring | once, in processes that did not train: evaluation anchors read 2026-10-28T17:40:32Z by `137-score#1`, which the 06:50 window stop ended before its first record was written; `137-score#2` (29 Oct 18:00 AEDT) wrote the three records (first at 2026-10-29T09:17:23Z); P-novelty pass `137-score-novelty#1`, 30 Oct 00:55 AEDT (T5 not scored: guard failure; references only) |
| Validation | `python -u scripts/run_issue_137_diamond.py --publish && … --validate`, scheduler run `137-republish-validate#1`, 2026-10-30 03:20:20–05:40 AEDT, **exit 0**: 11 records hash-checked, T5 seed 20260908 re-scored on every inventory with exact row equality, the publication re-derived twice byte-identically. `validation.json` (validated 2026-10-29T18:38:15Z) binds `records-manifest.json` sha256 `a79e8f18…`; `summary.json` sha256 `452d7e4d…`. Run after amendment A1 (see disclosures) |

Engine time is zero seconds. The primary metric is the Gate-B pooled grid AUC over the 311 mixed grid cells × 3 seeds
of the #104 sealed #109 evaluation split, tie-free cost, under `deterministic_scoring()`. Intervals are level-clustered
percentile bootstraps and are DESCRIPTIVE. The margin is δ = 0.02.

## Outcome

**Disposition 4: per-contrast tokens only. No scoped ranking claim either way.**

- Every global guard passes: G-isolation, G-chronology (including the novelty pass), G-scoring-policy and G-records.
  G-port (bit-identical) and G-fid (Tier E, three Atari games inside their bands) pass, and so does the pre-registered
  readiness check (readout ceiling 0.6783 ≥ 0.65).
- T5 **fails G-action on every seed**: endpoint action sensitivity 0.0286 / 0.0287 / 0.0277 against the frozen 0.05.
  Every other arm guard (G-rank, G-readout, G-nonfinite, G-trained) passes on every seed.
- By the frozen rule every token involving T5 is `readiness_or_precision_insufficient`. Neither disposition 1 nor 2
  applies. As a coverage arm T5 neither supports nor vetoes the #127 program claim.
- Descriptively, T5 ranks the grid at 0.7176 [0.6897, 0.7453] against U's 0.7444 [0.7269, 0.7620]
  (T5 − U −0.0268 [−0.0589, +0.0046]); this carries no token.
- In the carrier-space calibration of the frozen G-action threshold (#127 issuecomment-6396584235; descriptive, fit
  split only), the engine truth reaches 0.0395, below 0.05. T5's endpoints vary with the launch at 70–73 % of the true
  endpoints' variation (U 58 %, C\* 85 %). The frozen guard, its failure and the disposition stand.

## Design summary

The design is frozen in `plan.json`; every implementer decision was posted on #137 before the step it governs and is in
`decisions.md`.

**Route and model** ([P-license](https://github.com/Sino-Huang/NovPhy/issues/137#issuecomment-6426430212),
[P-route](https://github.com/Sino-Huang/NovPhy/issues/137#issuecomment-6426430665),
[adapter](https://github.com/Sino-Huang/NovPhy/issues/137#issuecomment-6426541110)). The official `Denoiser` /
`InnerModel` of the Atari 100k code at `config/agent/default.yaml` (= paper Table 2), used unchanged through the vendored
code: EDM preconditioning (σ_data 0.5, offset noise 0.3), U-Net 2D [2, 2, 2, 2] × [64, 64, 64, 64], conditioning dim
256, frame stacking of L = 4 frames, adaptive group norm on the noise level and the last 4 actions; 4 406 019
parameters. The adapter fixes only the data plumbing:

- frames: 64 × 64 RGB (DIAMOND's native resolution) in [−1, 1], area-resized from the #123 pixel stores (exact
  cv2 `INTER_AREA`, the official Atari preprocessing; full frame, 4:3 → 1:1);
- episodes: stride 5, the decision frame 3 extra times (the scene is static before the launch) then positions 0, 5, …,
  225 (49 frames; a shot ending earlier repeats its last frame); the launch `eb.launch5` is the action at the decision
  frame, zeros (no-op) after, embedded through the official action table in the `csgo` branch's real-valued form
  (`launch5 @ act_emb[0].weight`);
- training: the official loss (`Denoiser.forward`, 6-frame windows, 2 predicted frames) and `Trainer.train_component`
  step (AdamW lr 1e-4, wd 1e-2 on Linear / Conv weights, ε 1e-8, warm-up 100, clip 1.0), batch 32, fp32 with TF32 as
  upstream, windows drawn by the official `BatchSampler` rule from `eb.batch_rng(seed, b)`;
- rollout: `WorldModelEnv.step` semantics, 45 steps with the official sampler (Euler, 3 denoising steps, Karras σ 5 →
  2e-3, ρ 7), **8 paths per candidate with common random numbers** (path k's initial noise at each step shared by every
  candidate and level);
- score: tie-free `EndpointCosts` of the path-mean readout carrier of the predicted frame 225 against the readout of the
  decision frame;
- readout: `FrameReadout` (64 patch tokens of 8 × 8 × 3 pixels + patch centre, a mean-colour token, the #135
  `ParticleReadout` architecture, the #130 R1 recipe; 652 559 parameters), fitted **once** on 220 887 true fit frames
  (E99 targets) on CPU in Phase 0 and shared by every configuration and seed.

**Readiness (pre-registered stop)** ([result](https://github.com/Sino-Huang/NovPhy/issues/137#issuecomment-6428213317)).
The readout's policy ceiling on true 64 × 64 frames (94 mixed policy cells, 1 504 candidates) is **0.6783** against
the threshold 0.65 (E99 true-carrier ceiling on the same cells 0.7223); T5 was not stopped.

**Selection** ([decision](https://github.com/Sino-Huang/NovPhy/issues/137#issuecomment-6466769084); #127 rule:
policy-split pooled grid AUC of the primary score, 94 mixed policy cells, seed 20260908, full budget each; the selected
run is reused as main seed 20260908):

| configuration | updates (batch 32) | change from the Atari 100k recipe | policy grid AUC | training GPU-h (shared GPUs) |
| --- | --- | --- | --- | --- |
| **native** | 409 600 | none (paper Table 3: 10 000 + 999 × 400) | **0.6863** (selected) | 8.82 |
| ar4 | 409 600 | `num_autoregressive_steps` 4 (the `csgo` branch value) | 0.6802 | 15.40 |
| native-2x | 819 200 | 2 × the updates (the fit set has 682 k frames vs Atari 100k's 100 k) | 0.6523 | 12.85 |

U had no policy-split tuning of this kind (#127), disclosed.

## G-fid and G-port

**G-fid, Tier E: PASS** ([band](https://github.com/Sino-Huang/NovPhy/issues/137#issuecomment-6426803157), frozen
before the run; `band.json` sha256 `a21b44ec…`; [result](https://github.com/Sino-Huang/NovPhy/issues/137#issuecomment-6428213735)).
The released Atari 100k agents, built and loaded as `play.py --pretrained`, evaluated with the official test env and
collector (ε 0, 100 episodes per game, no life-loss termination; `Trainer.collect_test(final=True)` semantics) on CPU
in env `b127-diamond` (torch 2.13.0+cu130, otherwise the upstream pins); 0 repair attempts:

| game | ours (100 episodes, mean ± SE) | paper 5-seed mean (Table 1) | band (5-seed mean ± 2 SD) | pass |
| --- | --- | --- | --- | --- |
| Pong | 20.89 ± 0.05 | 20.36 (20.4) | [18.963, 21.757] | yes |
| Freeway | 32.22 ± 0.08 | 33.32 (33.3) | [32.108, 34.532] | yes |
| Boxing | 90.74 ± 0.49 | 86.90 (86.9) | [63.676, 110.124] | yes |

**G-port: PASS, bit-identical.** The official code (`Trainer.train_component`, `utils.configure_opt` / `get_lr_sched`,
`Denoiser.forward`, `DiffusionSampler.sample`, `WorldModelEnv.step`, the official `Episode` / `make_segment` /
`collate_segments_to_batch`; objects instantiated from the vendored configs) against the adapter, both on torch
2.13.0+cu130, CPU float32, three cases (`atari-discrete`: the unchanged official model at the full LR; `launch`: real
NovPhy fit windows with left padding and the launch frame, warm-up steps, and a full 45-step rollout; `launch-ar4`):
initial state, batch, forward loss, two optimizer steps (loss, LR, pre-clip gradient norm, post-clip gradients,
parameters), one sampler call and all 45 rollout frames have every delta exactly 0. The CPU tests show that three
injected defects are detected.

## Results

### Guards

| seed | G-rank (mean tied-pair share) | G-action (≥ 0.05) | G-readout (ceiling ≥ 0.65) | G-nonfinite | G-trained | policy grid AUC (fit/policy only) |
| --- | --- | --- | --- | --- | --- | --- |
| 20260908 | pass (0.0016) | **fail (0.0286)** | pass (0.6783) | pass (0.0) | pass | 0.6863 |
| 20260909 | pass (0.0014) | **fail (0.0287)** | pass (0.6783) | pass | pass | 0.7255 |
| 20260910 | pass (0.0022) | **fail (0.0277)** | pass (0.6783) | pass | pass | 0.6839 |

G-collapse does not apply (pixel state read through a readout; G-readout applies). G-port and G-fid pass. Global
guards: G-isolation, G-chronology (plan 2026-10-28T10:36:53Z ≤ readouts 2026-10-28T17:35:26Z ≤ evaluation anchors
17:40:32Z ≤ first record 2026-10-29T09:17:23Z; novelty anchors 2026-10-29T13:55:44Z after the last primary record
13:50:49Z), G-scoring-policy and G-records all pass.

### Grid rankers (311 mixed cells × 3 seeds)

| ranker | pooled AUC [CI] | per seed (08 / 09 / 10) | top-1 (chance 0.089) |
| --- | --- | --- | --- |
| U | 0.7444 [0.7269, 0.7620] | 0.7141 / 0.8011 / 0.7181 | 0.195 |
| C\* | 0.7834 [0.7676, 0.7992] | 0.8262 / 0.7334 / 0.7905 | 0.192 |
| P | 0.7203 [0.6965, 0.7442] | — | 0.180 |
| J | 0.7057 [0.6846, 0.7267] | 0.6690 / 0.7012 / 0.7470 | 0.160 |
| J_dec | 0.7460 [0.7277, 0.7637] | 0.7205 / 0.7933 / 0.7242 | 0.180 |
| **T5** | 0.7176 [0.6897, 0.7453] | 0.6928 / 0.7557 / 0.7044 | 0.325 |

T5 has no tied pairs and no nonfinite or typed rows; P has a tied-pair share of 0.002.

### Contrasts (grid, paired per (level, seed))

| contrast | estimate [CI] | token | reverse token |
| --- | --- | --- | --- |
| T5 − U | −0.0268 [−0.0589, +0.0046] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| T5 − C\* | −0.0657 [−0.0973, −0.0346] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| T5 − P | −0.0027 [−0.0409, +0.0347] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| T5 − J_dec | −0.0283 [−0.0610, +0.0036] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |

Every token is the stop token because T5 fails G-action. The estimates are reported as measured; they license no
statement in either direction. No within-family contrast exists (one arm).

### Wrong-anchor control (right − wrong, 480 fit anchors, grid)

| arm | right | wrong | right − wrong | ranking stability (Spearman) | token |
| --- | --- | --- | --- | --- | --- |
| T5 | 0.7176 [0.6897, 0.7453] | 0.5575 [0.5475, 0.5677] | +0.1601 [+0.1315, +0.1880] | 0.1269 | `readiness_or_precision_insufficient` |

### Descriptive inventories (no tokens)

| inventory (cells) | U | C\* | P | J | J_dec | T5 |
| --- | --- | --- | --- | --- | --- | --- |
| offset (228) | 0.6670 | 0.7748 | 0.5000 | 0.7522 | 0.6904 | 0.6389 |
| angle (234) | 0.6791 | 0.7590 | 0.8119 | 0.7765 | 0.6988 | 0.6512 |
| power (141) | 0.8058 | 0.8068 | 0.5000 | 0.8076 | 0.7943 | 0.6604 |

T5 − U: offset −0.0280 [−0.0890, +0.0303], angle −0.0280 [−0.0915, +0.0335], power −0.1453 [−0.2206, −0.0724]
(stop tokens).

### Engine position error (#100 metric; DESCRIPTIVE)

Mean slot-centre MSE of the readout carrier (path mean) against the engine centres on present slots, every complete grid
candidate of every mixed grid cell, from the Gate-B anchors (member-clustered bootstrap):

| t | 15 | 30 | 60 | 150 | 225 |
| --- | --- | --- | --- | --- | --- |
| T5 | 0.0531 | 0.0530 | 0.0513 | 0.0491 | 0.0511 [0.0466, 0.0556] |
| U | 0.0505 | 0.0501 | 0.0484 | 0.0459 | 0.0474 [0.0440, 0.0508] |
| C\* | 0.0547 | 0.0545 | 0.0543 | 0.0594 | 0.0670 [0.0617, 0.0723] |

### G-action calibration context (DESCRIPTIVE; fit split only; changes nothing)

The same `eb.action_sensitivity` call on the 480 fit anchors × 16 grid launches in carrier space (#127
issuecomment-6396584235; reference values from `.local-artifacts/issue-129-slotformer-v1/analysis/g-action-calibration.json`):

| source | per seed (08 / 09 / 10) | mean | fraction of engine truth |
| --- | --- | --- | --- |
| engine truth | — | **0.0395** | 100 % |
| U | — | 0.0229 | 58 % |
| C\* | — | 0.0334 | 85 % |
| T5 (frozen readouts) | 0.0286 / 0.0287 / 0.0277 | 0.0283 | 72 / 73 / 70 % |

The engine truth itself is below 0.05: the frozen threshold is not attainable in carrier space even by a perfect model.
This is context only; the frozen guard, its failure and disposition 4 stand.

### P-novelty (#110 novelty pairs)

Not scored for T5: P-novelty scores each arm that passes its guards, and T5 fails G-action (`novelty/arms.json`). The
U / C\* / P novelty references were computed and are hashed in `records-manifest.json` (`novelty/references/`); the
descriptive reference table is in `findings.md`.

## Disposition (δ rule)

- Disposition 1 (T5 ranks launches better than U, scoped): **no** — T5 fails G-action; T5 − U carries the stop token.
- Disposition 2 (U ranks launches better than T5, scoped): **no** — requires every T5 guard to pass.
- Disposition 3 (no AUC difference of 0.02 or more): **no** — guards fail.
- **Disposition 4: per-contrast tokens only**, every one `readiness_or_precision_insufficient`.
- #127 program claim: T5 is a coverage arm outside the program set; it neither supports nor vetoes it.
- Budget safeguard: does not fire (T5 − U is not `supported`); T5's mean training compute is 18.87 × U's (process
  seconds, shared GPUs).

## Deviations and disclosures

1. **Post-freeze amendment #137 A1 (reporting only, after scoring)**
   ([decision](https://github.com/Sino-Huang/NovPhy/issues/137#issuecomment-6500578672),
   [addendum](https://github.com/Sino-Huang/NovPhy/issues/137#issuecomment-6500794143); registered in
   `scripts/post_freeze_amendments.py`, commit `7321eb1c`). The first publication reported T5's trainable parameters
   as 0: the frozen `readouts_arm` counted `requires_grad` parameters of the frozen model. A1 changes one line of the
   runner, so `arm_compute` takes the count from each training `report.json` (4 406 019). A1 also replaces one
   test-setup line: since `--score` the real `data/` also holds the read evaluation anchors, and the CPU ladder test
   now links the fit / policy products one by one. No record, readout, guard, token or disposition changed. The first
   `137-publish-validate` run re-scored seed 20260908 with exact row equality and wrote its validation receipt; its
   final `publish()` then exited non-zero, because the process predated the amendment registry. The re-publication
   (`137-republish-validate`) re-derived and re-validated everything.
2. **Window-stop loss.** `137-score#1` read the evaluation anchors and was ended by the 06:50 window stop before its
   first record (2.16 GPU-h, not in the ledger). `137-score#2` re-used the stored anchors, so `read_at` is unchanged
   and G-chronology holds. Records are resumable per record.
3. **Small objects at 64 × 64.** The native resolution leaves a pig a few pixels wide. The readiness ceiling on true
   frames (0.6783) is 0.044 below the E99 true-carrier ceiling (0.7223).
4. **Continuous launch.** `main`'s action table is categorical. The launch enters through the same table in the form
   the official `csgo` branch uses for multi-hot actions; the paper's driving model (continuous actions, App. M) is not
   released.
5. **Stochastic sampler.** 8 paths per candidate with common random numbers; rows are bitwise reproducible for a fixed
   batch (one level per call) and agree across batch compositions up to CPU / GPU convolution blocking amplified by the
   official 8-bit quantisation of each denoised frame.
6. **Equal tuning allowance.** T5 chose among three configurations on the policy split; U had no such tuning (#127).
7. **G-fid protocol.** The official final collect draws its 100 episodes in one call; the fidelity runner collects
   one seeded episode per call so it can resume. The per-episode distribution is the same. Freeway's mean (32.22) lies
   0.11 above its band's lower end.
8. **Native head.** DIAMOND's reward / termination model predicts Atari rewards and was not trained, so there is no
   native success head and no U + head control.

## Compute

GPU seconds are each process's wall time on its GPU (`eb.mark`; each arm-seed's training is counted once, from its
`report.json`). Most jobs ran with `shares_gpu`, so process time on a shared GPU exceeds the device time the work would
need alone. The exclusive timing probe measured 0.058 s per native update and 6.38 s per scoring decision
(`timing.json`).

| item | process GPU-h |
| --- | --- |
| timing probe (exclusive) | 0.05 |
| selection native / ar4 / native-2x (native = T5 seed 20260908) | 8.82 / 15.40 / 12.85 |
| T5 seeds 20260909 / 20260910 | 5.74 / 5.79 |
| selection readouts; readouts (3 arm-seeds) | 0.50; 3.04 |
| scoring (3 records) | 6.84 |
| **recorded** (`compute.json`) | **59.03** |
| unrecorded: score#1 (window stop) 2.16, smoke 0.10, novelty 0.01, publish-validate#1 2.30, republish-validate 2.33 | ≈ 6.90 |
| **active total** | **≈ 65.9 (child cap 136; funded projection 80.3)** |

CPU work, not in the GPU ledger: the readiness readout (75 CPU-min), G-fid (51 CPU-min), the G-port (minutes) and the
64 × 64 frame cache (5 CPU-min).

| | T5 | U | C\* |
| --- | --- | --- | --- |
| trainable parameters | 4 406 019 (+ 652 559 readout) | 1 837 690 | 1 862 396 |
| training GPU-s per seed (08 / 09 / 10; process) | 31 765 / 20 674 / 20 825 | 1 290 / 1 300 / 1 292 | 985 / 1 044 / 1 044 |
| inference MACs per candidate (to frame 225, 8 paths) | 3.29 T | 20.99 M | 27.26 M |
| wall per decision (16 candidates, GPU, median) | 6.33 s | — | — |

## Reproduction

```
conda activate novphy && source env.sh
M="python -u scripts/run_issue_137_diamond.py"
# Phase 0 (CPU)
CUDA_VISIBLE_DEVICES="" $M --data && CUDA_VISIBLE_DEVICES="" $M --readiness && CUDA_VISIBLE_DEVICES="" $M --project
CUDA_VISIBLE_DEVICES="" python scripts/b127_diamond_port_reference.py --inputs
CUDA_VISIBLE_DEVICES="" /srv/home/shua0171/miniconda3/envs/b127-diamond/bin/python scripts/b127_diamond_port_reference.py --reference
CUDA_VISIBLE_DEVICES="" python scripts/b127_diamond_port_reference.py --compare      # gport/gport.json, exit 0
CUDA_VISIBLE_DEVICES="" /srv/home/shua0171/miniconda3/envs/b127-diamond/bin/python scripts/b127_diamond_fidelity.py --run
/srv/home/shua0171/miniconda3/envs/b127-diamond/bin/python scripts/b127_diamond_fidelity.py --finish   # fidelity/gfid.json
# GPU, night window only (scheduler): timing, selection, smoke, freeze
CUDA_VISIBLE_DEVICES=g $M --timing
CUDA_VISIBLE_DEVICES=g $M --selection --gpu q        # q = 0, 1, 2 (native, ar4, native-2x)
$M --selection-readouts && $M --smoke && $M --prepare
# main runs (scripts/gpu_window_issue_137_main.json)
CUDA_VISIBLE_DEVICES=g $M --train --gpu q            # q = 0, 1
CUDA_VISIBLE_DEVICES=g $M --readouts --gpu q && $M --freeze-readouts
CUDA_VISIBLE_DEVICES=0 $M --score                    # once
CUDA_VISIBLE_DEVICES=0 $M --score-novelty
$M --publish && CUDA_VISIBLE_DEVICES=0 $M --validate # exit 0
CUDA_VISIBLE_DEVICES="" pytest tests/test_issue_137_diamond.py    # 21 passed
```
