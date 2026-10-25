
<!-- posted: https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382394238 -->
## Decision P-fid (2026-10-24): Tier-E G-fid band and protocol frozen before the run

**Trigger.** Phase 0, Tier-E G-fid for C1 (DINO-WM). The band must be frozen and posted before the run (#127 P-fid).

**What was verified.**
- The README (`gaoyuezhou/dino_wm` @ `0a9492fa12044b852ae9e001cc74604b79c8bb0c`, MIT) advertises checkpoints on OSF (`bmw48`, `checkpoints/outputs.zip`). They are downloadable: 953 204 628 bytes, sha256 `425b0d8c…9179482` (equal to OSF's own hash). It contains `outputs/{pusht,point_maze,wall_single}/{hydra.yaml, checkpoints/model_latest.pth}`. The PushT `model_latest.pth` has sha256 `e909f0cec958fc0b49a79f2e85730ae6b5f83dee61a224705f91883eb3bb74c5`, epoch field 2.
- The PushT dataset `datasets/pusht_noise.zip` is 2 785 304 515 bytes, sha256 `442f5dee…f92da08` (equal to OSF's hash). It was downloaded in parallel byte ranges and re-hashed after assembly.
- The DINOv2 ViT-S/14 weights are the official `dinov2_vits14_pretrain.pth` from `dl.fbaipublicfiles.com`: 88 283 115 bytes, sha256 `b938bf1b…60cd9`. The hub code is pinned at `facebookresearch/dinov2` @ `7764ea0f`. The official `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` resolves to both through a pre-populated `TORCH_HOME`, so nothing is fetched at run time. Upstream fetched the unpinned default branch.
- Paper (arXiv 2411.04983v2): Table 1 gives PushT SR 0.90 for DINO-WM. Table 8 shows this is the **MPC** row; open-loop CEM gives 0.86. The paper evaluates 50 initial/goal pairs, with goals 25 steps away.

**Band (frozen).** Paper 0.90 ± 2 binomial SE on 50 cases (SE = 0.0424, so ±2 SE = [0.815, 0.985]). On the 1/50 grid this is **[0.82, 0.98] inclusive** (41–49 successes). The statistic is the official `final_eval/success_rate` of `plan.py`. A case counts as solved at the first MPC iteration whose env rollout meets PushT `eval_state` (position error < 20 and angle error < π/9). One run. `fidelity/band.json` sha256 `247747e2f20c383e18d95ba261b99b1d6c929bf22c33ef120351be2797e809ab`.

**Protocol.** The README's PushT command `python plan.py --config-name plan_pusht.yaml model_name=pusht` runs on the released checkpoint with these settings:
- 50 cases, seed 99, `goal_source=dset`, `goal_H=5`, frameskip 5;
- MPC-CEM: horizon 5, 300 samples, top-30, 30 CEM steps, 5 actions taken per iteration;
- objective MSE on the last frame.

One setting is ours: `planner.max_iter=12`.
- *Why a cap is needed.* The released config sets `max_iter: null` ("unlimited if null"), so the loop ends only when all 50 cases succeed. A rate of 0.90 therefore cannot be the end state of an uncapped run (upstream issue #29: "very large / infinite cap").
- *Why 12.* 12 iterations × 5 planning steps × frameskip 5 = 300 env steps, which is the official gym registration's `max_episode_steps=300` for `pusht`.
- *Options considered:*
  - uncapped (no defined end);
  - 2 iterations (the 50-step budget of a third-party re-implementation, not the paper's);
  - 12 (chosen: the code's own episode limit).
- *Wall time.* 12 iterations may take several hours. The full per-iteration success curve is recorded.

**Environment (rule P-route).** The official env (Python 3.9, torch 2.3 / cu121) cannot run on RTX 5090 (sm_120). The new env `b127-dino_wm` has Python 3.10, torch 2.7.1+cu128 and the official pins for everything else: gym 0.23.1, pymunk 6.8.0, hydra-core 1.2.0, einops 0.4.1, decord 0.6.0, numpy 1.26.4 and the rest. The official code runs unmodified, from the clone. Disclosed infrastructure shims, none of which touches planning, the model or the metric:
1. Empty `sys.modules` stand-ins for `mujoco_py` / `d4rl.offline_env`. `import env` registers PointMaze, which imports them at import time; PushT never builds a MuJoCo or D4RL object.
2. `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. torch ≥ 2.6 defaults to `weights_only=True`, and the release pickles whole modules.
3. `WANDB_MODE=disabled`, which is the official `DummyWandbRun` path.
4. The plot-only VQ-VAE decode (`VWorldModel.decode_obs`) runs in chunks of 4 cases. The tensors are identical, and this guards against the OOM reported in upstream #23.

A CPU smoke run (2 and 6 cases, 4 samples, 1 CEM step, 1 MPC iteration) runs end to end.

**Run.** Scheduler job `130-gfid-tier-e`, GPU 2, `scripts/b127_dino_wm_fidelity.py --run`, which writes `fidelity/gfid.json` (schema `issue_127_gfid_v1`). If the run fails the band, P-fid allows at most two logged repair attempts, each fixing a diagnosed port or infrastructure defect, never the band. After that the arm is dropped with `fidelity_not_reproduced`.

**Disclosed discrepancy.** The paper's Table 12 lists predictor LR 5e-5. The released config and the PushT checkpoint's `hydra.yaml` use 5e-4. The NovPhy arms follow the released code and config (5e-4).

<!-- posted: https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382422982 -->
## Decision P-license (2026-10-24): vendor DINO-WM (MIT) and pin DINOv2 (Apache-2.0)

**Trigger.** C1 vendoring, Phase 0.

**Licenses verified.**
- `gaoyuezhou/dino_wm` @ `0a9492fa12044b852ae9e001cc74604b79c8bb0c` ships `LICENSE`: MIT, "Copyright (c) 2025 gaoyuezhou". The file is vendored verbatim; its sha256 is recorded in `VENDOR.json`.
- `facebookresearch/dinov2` @ `7764ea0f912e53c92e82eb78a2a1631e92725fc8` ships `LICENSE`: Apache-2.0. The ViT-S/14 weights `dinov2_vits14_pretrain.pth` are covered by the same repo license. DINOv2 is used as a frozen feature extractor, loaded from the pinned clone and the downloaded weights. It is not vendored and never committed.
- The released DINO-WM PushT checkpoint and dataset (OSF `bmw48`) are used only for the Tier-E fidelity run. They are not redistributed (they stay in `data/external/dino_wm/`, which is gitignored).

**Choice.** Vendor into `world_model/baselines/dino_wm/vendor/` only the files the NovPhy arms import:
- `models/vit.py` (`ViTPredictor`: the causal frame-block ViT predictor);
- `models/proprio.py` (`ProprioceptiveEmbedding`: the action encoder, and the proprio encoder in the official configuration);
- `LICENSE`.

`vit.py` gets one listed edit. The causal mask's `generate_mask_matrix(...).to('cuda')` in `Attention.__init__` becomes a non-persistent registered buffer. The values are identical, the module follows `.to(device)`, and it can be built on CPU for G-port and the tests. Upstream cannot be constructed without CUDA.

**Not vendored, ported instead (each held to G-port).**
- The predictor path of `models/visual_world_model.py` (`VWorldModel.encode` with `concat_dim=1` tiling, `predict`, the `z_loss` MSE over non-action channels, and `rollout` / `replace_actions_from_z`). It imports `torchvision`, which the `novphy` env does not have, and it hard-wires a proprio channel and an image encoder. NovPhy has no proprioception, and C1 reads precomputed DINOv2 features.
- The optimizer setup of `train.py` (`Trainer.init_optimizers`): `AdamW(predictor, lr 5e-4)` and `AdamW(action_encoder ∪ proprio_encoder, lr 5e-4)`, with no scheduler and no clipping. `train.py` pulls in wandb, accelerate and hydra.

Both are ported line for line into `world_model/baselines/dino_wm/novphy.py`. G-port checks forward, loss, gradients and two optimizer steps against the official `VWorldModel` and `Trainer.init_optimizers`, run from the clone in the `b127-dino_wm` env. The check uses the official PushT configuration with the DINOv2 encoder in the loop. The decoder is off, which is an official option: its loss only sees `z.detach()`, so it never reaches the predictor or action-encoder gradients.

<!-- posted: https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382562140 -->
## Decision P-other (2026-10-24): Tier-E runs in a full night window; tonight's GPU 2 goes to the feature cache

**Trigger.** Job `130-gfid-tier-e` (run 1) launched at 20:15 AEDT. Its measured wall time is about 2.05 min per CEM optimisation step on one RTX 5090: the per-step output files are written at 20:17:24, 20:19:27, 20:21:35, 20:23:37, 20:25:39 and 20:27:42. The frozen protocol needs 30 CEM steps per MPC iteration and 12 iterations (`max_iter` frozen in `band.json`). That is about 61 min per iteration and about 12.3 h per run. The window remaining tonight is about 10.3 h, so the run cannot finish. The official MPC loop has no resume point.

**Options.**
1. Let the run continue: it is stopped at 06:50 with about 10 of 12 iterations done and never produces the frozen statistic.
2. Change `max_iter`: rejected, because the band and protocol are frozen.
3. Shard the 50 cases over GPUs: rejected, because it changes the official RNG streams and goal sampling.
4. **Stop run 1 now and run the frozen protocol unchanged in a full window.**

**Choice: option 4.** The full window runs from 18:00 to 06:50, about 12.8 h, against an estimate of about 12.3 h. Run 1 is stopped before its first MPC iteration ends, so no MPC or `final_eval` success value exists. Nothing about the outcome informs this decision: it rests only on the wall time. The stopped run directory `fidelity/runs/tier-e-1` is kept and disclosed as an infrastructure stop. It is not a repair attempt, and the band and protocol are unchanged.

The rerun is scheduler job `130-gfid-tier-e-full`: GPU 2, non-resumable, `est_minutes` 760, so it can only launch at the start of a window. It uses the same command, `scripts/b127_dino_wm_fidelity.py --run`. Tonight's GPU 2 goes to `130-features`, then `130-readiness`.

<!-- posted: https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382576605 -->
## Decision P-other (2026-10-24): C1 adapter design (preprocessing, launch encoding, schedule, C1-car tokens, R1, feature cache, readiness smoke)

**Trigger.** C1 Phase 0 leaves the adapter design open. It must be fixed before the feature cache, the readiness R1 and any predictor training. Priority: validity, then fairness to the baseline (#127 P-other). Code: `world_model/baselines/dino_wm/novphy.py`. G-port passes (`gport/gport.json`; see the Phase-0 report).

**1. DINOv2 preprocessing of the 320 × 240 frames.**
- *Options:*
  - (a) DINO-WM's own pipeline: `Resize(224)` + `CenterCrop(224)`, then the encoder's `Resize(196)`, giving 14 × 14 patches. It crops about 12 % of the frame width on each side, where level structures and pigs can sit, and scales by 0.82.
  - (b) Aspect-preserving at DINO-WM's patch scale: 196 × 266, 14 × 19 patches.
  - (c) **Aspect-preserving full frame at native scale: 238 × 322, 17 × 23 = 391 patches (chosen).**
- *Reason.* Pigs in our frames are about 4 px wide. (a) can crop them away and (a)/(b) shrink them further. (c) keeps every pixel at native scale, which is the most favourable resolution for the baseline without upsampling.
- *Kept from DINO-WM:* `x/255` and `Normalize(0.5, 0.5)` (`datasets/img_transforms.py`), a bilinear antialiased resize (torchvision `Resize` semantics; the aspect stretch is 1.353 vs 1.333), and the `x_norm_patchtokens` feature key. The encoder is frozen and runs in fp32 under `deterministic_scoring()`.
- *Storage:* features are stored as fp16 (`[391, 384]` per frame).
- *Disclosed cost:* 391 tokens per frame against 196 upstream, so the predictor sequence is 3 × 391 = 1173. Batch 32 in fp32 needs micro-batching, which the runner will declare.

**2. Launch encoding: the native action encoding of a 5-d action.**
- DINO-WM concatenates the per-frame actions of each frame-skip step (`TrajSlicerDataset` "concat"). The per-frame action is `eb.launch5` at frame 0 and the zero no-op at every other frame. So step 0 = `[launch5, 0 × 20]` and every later step = 0 (25-d), encoded by the official action encoder (Conv1d k=1 → 10-d, tiled on every patch, `concat_dim=1`).
- `normalize_action` is off. `launch5` is already U's normalised launch vector. Dataset z-scoring over per-frame actions (more than 99 % of frames are no-ops) would map the no-op to a non-zero vector and inflate the launch about 15×. For a linear action encoder, normalisation is only an affine reparametrisation.
- Rejected: a bare 5-d action per step, which drops the native frame-skip concatenation. It has the same function class but is not the native encoding.

**3. Schedule and history (native PushT settings).**
- Frame-skip 5 gives 45 steps to frame 225. `num_hist` 3, `num_pred` 1.
- Training windows are 4 frames at stride 5, starting at frame 5k and lying fully inside `[0, min(225, end)]`. 4.3 % of fit shots end before 225; the shortest ends at 138.
- Native training slices windows at every frame offset. C1 uses only offset 0, the phase of the ranking rollout, so the cache holds stride-5 positions only. This is disclosed.
- Ranking rolls out with the official `rollout` from the single anchor frame (the official 1-frame start; the block-causal mask trains 1-, 2- and 3-frame contexts): 44 no-op steps plus the final prediction reach frame 225.
- There is no proprio channel (NovPhy has no proprioception). There is no decoder: the official decoder loss sees only `z.detach()`, so predictor and action-encoder gradients are unchanged.
- The predictor is native: ViT depth 6, 16 heads × 64, MLP 2048, dropout 0.1. Optimisers are AdamW(predictor, 5e-4) and AdamW(action encoder, 5e-4), as in the released config and checkpoint; the paper's table says 5e-5 for the predictor. Batch 32, no scheduler.
- Trainable parameters: C1-dino 19 855 570; C1-car 1 243 648.

**4. C1-car: how a 236-d carrier becomes predictor tokens.**
- *Options:*
  - one 236-d token per frame, the geometry of the paper's DINO-CLS ablation (PushT 0.44 vs 0.90 for patch tokens);
  - **18 slot tokens per frame (chosen).**
- Each slot token holds the slot's 13 features plus the carrier's 2-d frame header. The header is tiled onto every slot, as `concat_dim=1` tiles frame-level vectors onto every patch, giving 15-d tokens. Carrier columns are z-scored with fit-split statistics.
- Decoding takes the slot features plus the mean of the 18 header copies, de-normalises, and applies `EndpointCosts` to the decoded carrier. No readout is involved.
- *Reason:* the carrier's object slots play the role of DINO patches. As in DINO-WM, there is no input projection, so the predictor width is 15 + 10.

**5. R1 readout (C1-dino primary score path).** DINOv2 patch features → E99 carrier (`PatchReadout`).
- *Architecture:* LayerNorm plus a learned position embedding over the 391 patch tokens. 19 learned queries (header + 18 slots) cross-attend over them (8 heads), followed by a residual MLP (512) and linear heads to 2 and 13 outputs per query. 1 150 223 parameters.
- *Training:* targets z-scored with fit statistics, MSE loss, AdamW at lr 1e-3, weight decay 1e-4, batch 128, 25 epochs (the #123 probe schedule), under `deterministic_scoring()`.
- *Reason:* #123's R1 is a 2-layer MLP on a vector. On 391 × 384 patch tokens it would either flatten (about 77 M parameters) or pool, and pooling loses the location of a 4-px pig. Slot-query attention can attend to the single patch that holds a pig and locate it.
- *Fit frames (fit split only):* every fit shot at #123's R1 probe positions (every 15th to 285, clamped to the end): 273 692 frames from 13 919 shots. Targets are the E99 carriers of the same frames (the #123 fit memmap).
- *Freeze and seeding:* R1 does not depend on the predictor, because DINOv2 is frozen. It is fitted per seed with this recipe (the readiness smoke uses 20260908) and frozen with sha256 before any evaluation anchor is encoded.

**6. Feature cache.**
- Location: `~/.cache/novphy-issue-130-dino-features/` on local NVMe; indexes and `MANIFEST.json` (sha256) are in `features/`.
- *Fit frames:* frame-skip-5 positions 0..min(225, end) plus the R1 probe positions: 687 038 frames (633 731 training, 273 692 probe), 206.3 GB.
- *Policy frames:* the 120 policy anchors (frame 0 of the anchor branch) and every complete policy shot at its true endpoint min(225, end): 3 600 frames, 1.1 GB. The #123 policy store keeps every 15th position, clamped, so min(225, end) is always stored.
- Estimate: about 45 GPU-min for extraction (resumable) and about 45 for readiness [INFERENCE until measured].

**7. Readiness smoke (G-readout, before any predictor training).**
- *Ceiling:* the pooled policy grid AUC (Gate-B mixed cells, `eb.policy_grid_auc`) of the tie-free cost of R1(true endpoint features) vs R1(policy anchor features). This is the #123 `policy_aucs` ceiling.
- *Threshold:* 0.65. Below it, C1-dino stops with the typed reason `readiness_ceiling_below_threshold`, and C1-car still runs.
- *Diagnostics, no gate:* the same ceiling on the true E99 carriers, and R1's pig presence and centre error against E99 on the policy endpoints.
- No evaluation-split data is read.

<!-- posted: https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6383216316 -->
## Decision P-other (2026-10-24): C1-dino stops at the readiness smoke (typed `readiness_ceiling_below_threshold`); C1-car proceeds

**Trigger.** Readiness smoke `130-readiness` (scheduler run 1, GPU 2, 20:50–21:39 AEDT) finished under the rule frozen in issuecomment-6382576605.

**Result** (`readiness/readiness.json`; R1 `readiness/r1.pt`, sha256 `62266b64…44206bc`; feature manifest sha256 `dca36d65…72c2012`):

| quantity | value |
| --- | --- |
| R1 ceiling: pooled policy grid AUC, tie-free cost of R1(true min(225, end) features) vs R1(anchor features), 94 mixed cells / 1 504 candidates | **0.5948** |
| G-readout threshold | 0.65 |
| same ceiling on the true E99 carriers (diagnostic; the best any carrier readout can reach) | 0.7223 |
| R1 vs E99 on the policy endpoints: pig presence MAE / pig centre MAE (normalised screen units) / all-column MAE | 0.0018 / 0.0265 / 0.0360 |
| R1 fit | 273 692 fit frames, 13 919 shots, 53 450 steps, final z-scored MSE 0.0276, seed 20260908 |

**Decision.** 0.5948 < 0.65. By the frozen rule, **C1-dino stops with typed reason `readiness_ceiling_below_threshold`** before any predictor training. C1-car (E99 carriers, no readout) is unaffected and proceeds to its selection and training under the #130 runner.
- C1-dino is in the #127 program set, so by #127 this stop vetoes the program-level claim. It stays in every report as a typed stop.
- The readout and the threshold are not changed after seeing this number. A post-hoc R1 redesign would make the decision depend on the outcome (#127 integrity rule). No port defect is diagnosed: G-port passes, the features match the official encoder path within 6e-5, and the R1 fit converged.
- R1 reads pig presence almost exactly (MAE 0.0018) but places the pig centre less precisely (MAE 0.027, about 8–9 px at 320 px width). The cost's displacement term depends on the centre. [INFERENCE: the frozen ViT-S/14 patch features at native scale localise a ~4-px pig less precisely than the E99 parser.]

**Compute.** Feature cache 618 GPU-s (687 038 + 3 600 frames, 207 GB fp16, `features/MANIFEST.json`). Readiness 2 923 GPU-s. Tier-E run 1, stopped, 858 GPU-s.

<!-- posted: https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6383358026 -->
## Decision P-other (2026-10-24): Tier-E fidelity run made resumable at MPC-iteration boundaries (protocol unchanged)

**Trigger.** `130-gfid-tier-e-full` was queued as non-resumable with `est_minutes` 760, which is only ~10 min inside a 770-min window. A 06:50 SIGTERM would discard the whole run. This is a scheduling risk; no outcome is involved (no MPC success value exists yet).

**Options.**
- (a) An exact checkpoint of the official MPC loop at iteration boundaries **(chosen)**.
- (b) Split the 50 cases into separately seeded chunks. Rejected: it changes the official RNG streams and the goal sampling.
- (c) Shorter `max_iter`. Rejected: frozen in `band.json`.

**Mechanism** (`scripts/b127_dino_wm_fidelity.py`, `install_mpc_checkpointing`):
- `planning.mpc.MPCPlanner.plan` is replaced by the official loop body verbatim (0a9492f `planning/mpc.py`). A line diff against upstream shows only the two added hooks.
- Hook 1, at the end of each MPC iteration (after the official `dump_logs`), atomically saves every piece of loop state: `is_success`, `action_len`, `planned_actions`, `memo_actions`, `cur_obs_0`, the evaluator's init condition, `iter`, the `logs.json` length, and the torch CPU / CUDA, numpy and python RNG states.
- Hook 2, at loop start, restores that state when a checkpoint exists, and truncates `logs.json` to the saved length so no iteration log is duplicated.
- Everything else the loop reads is rebuilt identically by the official `PlanWorkspace`: seed 99 gives the same 50 targets, and every env rollout re-seeds and re-runs from `state_0`.
- SIGTERM / SIGUSR1 are forwarded to the official process. The job exits 75, and the next launch resumes the unfinished run directory from its last completed iteration. The interrupted iteration is recomputed from the restored RNG state, so at most ~61 min is redone.
- The CEM samples come from the CPU RNG (`torch.randn(...).to(device)`), so the restored CPU state reproduces them.

**CPU verification** (official `plan.py`, 2 cases, 4 samples, 2 CEM steps, 2 MPC iterations):
- An uninterrupted run vs. a run stopped after iteration 1 (exit 75) and resumed gives a **byte-identical `logs.json`** and identical planned actions, `action_len` and `is_success`.
- The same holds for a run SIGTERMed in the middle of iteration 2 and resumed (exit 75, then 0).
- The check is in `tests/test_issue_130_dino_wm.py::test_tier_e_resume_equals_uninterrupted`.
- On the GPU the restored state is exact. GPU kernels are not bitwise deterministic between any two runs, interrupted or not.

**Scheduling.**
- The queued non-resumable `130-gfid-tier-e-full` cannot be edited in place by the scheduler. It is retired by `--update-deps` to depend on the stopped run `130-gfid-tier-e` (failed), which blocks it permanently. It never launches.
- New job `130-gfid-tier-e-r`: GPU 2, resumable, `est_minutes` 760, the same command `scripts/b127_dino_wm_fidelity.py --run`.
- GPU 2 is free now, so it starts tonight and resumes in the 2026-10-25 window. The scheduler's per-run seconds give the active GPU time.
- Band, protocol and statistic are unchanged. The infrastructure stop of run 1 (`tier-e-1`) stays disclosed. The next run is `tier-e-2`.

<!-- posted: https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6386706483 -->
## Decision P-other (2026-10-25): C1-car runner, native schedule, selection configurations, guards, budget

**Trigger.** C1-car (secondary arm; the only C1 arm that runs) needs its frozen schedule, its ≤ 3 pre-declared configurations and its guards **before any selection run**. Runner: `scripts/run_issue_130_dino_wm.py` (round-3 child-runner contract; `eb`).

**Native schedule** (released `conf/train.yaml` = the PushT checkpoint's `hydra.yaml`):
- Batch 32, 100 epochs, AdamW for the predictor and AdamW for the action encoder, no scheduler, no clipping, fp32. No decoder, no proprio.
- Windows are 4 frames (`num_hist` 3 + `num_pred` 1) at frame-skip 5, phase 0 (frames 5k). There are 591 974 fit windows from 13 919 shots; 601 shots end before frame 225 and contribute shorter sequences.
- 1 epoch = ⌈591 974 / 32⌉ = 18 500 steps. As in the official `TrajSlicerDataset`, the window order is one permutation (`eb.batch_rng(seed, 0)`) reused every epoch, the loader does not shuffle, and the last batch is partial.
- Seeds: `torch.manual_seed(seed)` for initialisation and dropout.
- **No micro-batching.** The C1-car sequence is 3 × 18 = 54 tokens of width 25 (1 243 648 parameters), so batch 32 fits trivially in fp32.
- Checkpoints: every 20 000 steps and on SIGTERM (resume equality is checked by a CPU test).
- A nonfinite loss is a typed terminal failure.

**Selection configurations** (seed 20260908; the selected run is reused as main seed 20260908; rule `eb.select_configuration`; policy-split pooled grid AUC of the tie-free cost):

| name | predictor LR | action-encoder LR | budget | steps | why |
| --- | --- | --- | --- | --- | --- |
| `car-lr5e-4` | 5e-4 | 5e-4 | 100 epochs | 1 850 000 | the released code and the PushT checkpoint's `hydra.yaml` |
| `car-lr5e-5` | 5e-5 | 5e-4 | 100 epochs | 1 850 000 | the paper's Table 12 predictor LR |
| `car-release` | 5e-4 | 5e-4 | 3 963 442 samples (the released PushT checkpoint's epoch field 2 × its 1 981 721 train windows) | 123 858 | the budget the released checkpoint carries [INFERENCE: epoch 2 = the epochs it trained] |

**Guards for C1-car:**
- G-port (pass on record).
- G-fid (DINO-WM Tier E). Until `fidelity/gfid.json` passes, every C1-car token is the stop token. `gfid.json` is read live at publication, not pinned in the plan, because it lands after the freeze.
- G-rank (480 fit anchors × 16 grid launches, primary cost), G-action (≥ 0.05), G-nonfinite (≤ 0.10), G-trained.
- The wrong-anchor control (right − wrong, 480 fit anchors) and the global gates (G-isolation, G-chronology, G-scoring-policy, G-records).
- **G-collapse does not apply.** C1-car's predictor state is the z-scored E99 carrier itself (slot tokens plus the tiled action channel), and it is decoded linearly back to the carrier. It has no learned latent representation that could collapse, which is the thing G-collapse tests in latent models (#123). A predictor that ignores the launch or the anchor is caught by G-action, G-rank and the wrong-anchor control. This is the same reading as #129 B1, a world model on E99 carriers.
- G-readout does not apply: no readout, because the cost is computed on the decoded carrier.
- There is no native success head. DINO-WM plans by goal-feature matching and has no success head, so there is no U + head control.

**C1-dino** appears in the publication as a dropped program-set arm. It carries the typed reason `readiness_ceiling_below_threshold`, every token is the stop token, and it vetoes the program claim. Its readiness record is pinned in the plan and in the readouts manifest. **C1-dino − C1-car cannot be computed** because C1-dino was not run; it is published as an absent estimate with the stop token. C1-car's contrasts (X − U, X − C\*, X − P, X − J_dec) are secondary.

**Budget projection** (C1 share 88 GPU-h: C1-car plus fidelity). The step rate is [INFERENCE] until the selection runs measure it, at about 5–8 ms per step on an RTX 5090.
- Selection ≈ 2 × 1.85 M + 0.12 M steps ≈ 5.3–8.5 GPU-h.
- Two further seeds ≈ 5.1–8.2 GPU-h.
- Tier-E fidelity ≈ 13 GPU-h, plus the 0.24 h stopped run 1.
- Feature cache and readiness 0.98 GPU-h (measured).
- Readouts and score ≈ 1.5 GPU-h.
- Total ≈ 26–33 GPU-h, within the 88 GPU-h share and the 20 GPU-h per-seed cap. `--smoke` re-projects from the measured selection rates before the freeze (P-cap if over).

**Scheduling.** Selection jobs `130-sel-car-lr5e-4`, `130-sel-car-lr5e-5` and `130-sel-car-release` run on GPU 2, resumable, with `shares_gpu`. They use under 2 GB beside the Tier-E job, which uses about 28 GB of 32 GB. `130-sel-readouts` (CPU-light, GPU 2) depends on all three.

<!-- posted: https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6386786676 -->
## Decision P-crash (2026-10-25): Tier-E run 2 (`130-gfid-tier-e-r#1`) crashed on CUDA OOM caused by a foreign GPU process; relaunched once

**Trigger.** `130-gfid-tier-e-r` run 1 launched at 2026-10-24 22:00:09 AEDT on GPU 2 and ended at 03:31:25 with exit 1. The cause was `torch.OutOfMemoryError` inside the official ViT predictor (`models/vit.py: torch.matmul(attn, v)`), raised at "Process 2253138 has 17.96 GiB memory in use".
- The scheduler logged pid 2253138 as a **foreign GPU process on GPU 2 (18 394 MiB)** from 21:50 onward. It was not launched by the scheduler or by #130, and it has since exited.
- The run never completed a CEM step (0 success lines), so no MPC iteration and no success value exists.
- The OOM is an infrastructure crash, not a typed failure and not a fidelity outcome. The GPU time (19 876 s on the wrapper clock) is booked in the ledger as lost fidelity compute.

**Decision (P-crash: relaunch once, with disclosure).**
- `enqueue --requeue 130-gfid-tier-e-r` on GPU 2, resumable. The same command, protocol and frozen band apply. The run continues in `fidelity/runs/tier-e-2` (no MPC checkpoint exists, so it starts at iteration 0).
- The fidelity script now sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before CUDA initialises. This is an allocator setting only and does not change numerics.
- A second infrastructure crash would not be relaunched without a new posted decision.
- C1-car selection jobs share GPU 2 (`shares_gpu`). They gather batches on the host and need under 1 GB of GPU memory each, beside about 27.6 GiB for the Tier-E run on a 31.4 GiB card.

<!-- posted: https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6386817726 -->
## Decision P-crash (2026-10-25): addendum to issuecomment-6386786676 — cause identified, relaunch placement

- **Cause (from the orchestrator).** The 18 GB process on GPU 2 (pid 2253138) was an orphaned #128 A4 fold process. It was left behind by host-RAM exhaustion at about 21:50, after three A4 folds had been co-scheduled, and was killed at about 03:31. The scheduler's cron did not tick between 22:00 and 03:38. The relaunch decision is unchanged: one relaunch under P-crash, the same protocol and the same frozen band.
- **Requeued** `130-gfid-tier-e-r` (resumable) at 03:41.
- **Memory headroom.** The three C1-car selection jobs on GPU 2 use 1.27 GB each (3.8 GB in total). The Tier-E run needs about 27.6 GiB on a 31.4 GiB card, so sharing the card is not safe. `130-gfid-tier-e-r` therefore depends on the three selection jobs and starts on GPU 2 when they finish (projected about 19:55 on 2026-10-25). Being resumable, it continues into the next window if needed. It may instead be placed on another free GPU at 18:00 by the orchestrator.
- **Timing disclosure.** The C1-car selection decision (issuecomment-6386706483) was posted at about 03:33, before `130-sel-car-*` launched at 03:38:48.

## Decision P-novelty (2026-10-25): the descriptive #110 novelty scoring pass (owner decision D6)

**Rule.** #127 P-novelty: every arm of this child that passes its guards is scored on the #110 novelty evaluation pairs after its primary scoring pass. Descriptive only: no token, never part of a disposition or the program claim. Fixed now, before any arm of the program has an evaluation score.

**Population.** The `issue110-evaluation` levels of the 29 in-scope cells under #107's frozen rule (`run_granularity_under_novelty.in_scope_cells`: #110-supported, has mixed-verdict evaluation levels, not rollout-error-only), on both sides of every pair: the novel side at novelty level k = 1–7 and its paired level-0 normal side. Grid inventory (16 launches; the level-5 slingshot cells use #110's mirrored grid). Engine verdicts from the #110 handoff (`handoff_107.json`). AUC on mixed-verdict levels only.

**Inputs.** #107's anchor rule: the level's first complete grid candidate in suffix order, shot frame 0 (pre-launch). Pixel arms read that frame; carrier arms read its E99 v1 batch-1 parse (`rt.anchor_carrier` with the frozen E99 encoder, under `deterministic_scoring()`, the #102 N2 pattern); A3-sym builds its state from the level's `decision-physics-v1.json`. The E99 v1 carrier does not represent most novelty entities (#110: v1 fully supports 2 of 45 cells); carrier arms therefore see the novel levels through that limited contract, which is part of the descriptive result. No input contract is changed for this pass.

**Scoring.** Each guard-passing arm, all three seeds, its primary score (logit / Q for direct predictors; tie-free cost of the predicted endpoint carrier for world models), once, in a process that did not train, under `deterministic_scoring()`, after the arm's primary scoring pass. References on the same cells and anchors: U and C* (the Gate-B checkpoints rolled out as Gate-B scores them) and P (the Gate-B prior rule over the fit verdicts).

**Numbers (DESCRIPTIVE).** Per ranker: pooled grid AUC per novelty level on the novel side, on the paired normal sides, pooled over novelty levels 1–7, and the paired contrast novel − paired normal (one row per (pair, seed) where both sides are mixed); level-clustered (pair-clustered for the paired contrast) percentile bootstrap, 10 000 draws, PCG64 7201. Records are hashed into this child's records manifest and re-derived by `--validate`.

Shared implementation: `scripts/external_baselines.py` (`novelty_cells`, `read_novelty_anchors`, `novelty_reference_records`, `novelty_table`), pinned by every child plan; each child runner gets a `--score-novelty` mode before its freeze.

<!-- posted: https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6400401642 -->
## Decision P-crash (2026-10-26): second Tier-E infrastructure crash (distinct cause); relaunch on an exclusive GPU 2, resuming from the MPC checkpoint

**Trigger.** `130-gfid-tier-e-r` run 2 ran from 2026-10-25 21:55:20 AEDT on GPU 2 and ended at 02:00 with exit 1.
- The error was a CUDA out-of-memory raised in the plot-only VQ-VAE decode inside the official evaluator (`evaluator.py:132` `decode_obs`, `vqvae` `conv_transpose2d`). It hit while iteration 3's evaluation rollout was being drawn.
- A #131 D1 selection job with `shares_gpu` (pid 3084350, 1.53 GiB) had been placed on GPU 2 beside the Tier-E process, which needs about 29 GiB. The orchestrator has acknowledged this as a scheduling error.
- The cause differs from the first crash (issuecomment-6386786676: an orphaned #128 process holding 18 GB).
- **Disclosure.** The first relaunch under P-crash (run 1 of `130-gfid-tier-e-r`, directory `tier-e-2`) also ended in an infrastructure OOM, before any MPC iteration. This is the second relaunch, and each is logged.
- No `final_eval` success value exists, and none has been read. The decision does not depend on any outcome.

**State saved by the resumable runner** (`fidelity/runs/tier-e-3/mpc_checkpoint.pt`):
- MPC iterations 1 and 2 of 12 are complete. The checkpoint was written at 00:37:18 after iteration 2, together with the `logs.json` length, the planner state and the torch CPU/CUDA, numpy and python RNG states.
- The crash happened inside iteration 3, after the checkpoint, so the checkpoint is clean.
- On resume, iteration 3 is recomputed from the restored RNG state. The CPU test shows this resume is byte-exact. On the GPU the restored state is exact, but GPU kernels are not bitwise deterministic between any two runs.
- `tier-e-2` holds no MPC iteration and is kept, disclosed, but unused.

**Decision.**
- Relaunch once more, resuming `tier-e-3` from its iteration-2 checkpoint on GPU 2, with no co-scheduled job. The orchestrator keeps every other job off GPU 2 until Tier-E completes. `130-gfid-tier-e-r` already has `shares_gpu: false`; the fault was the neighbour's `shares_gpu`.
- Plot-path OOM guard: the plot-only decode now runs one case at a time, down from 4. Its memory grows with the rollout length (12 iterations × 5 steps). It produces identical tensors and does not touch planning, the objective or the success metric.
- Protocol, band (`band.json` sha256 `247747e2…`) and statistic are unchanged.
- Rate measured in run 2: about 81 min per MPC iteration on an exclusive RTX 5090, slower than the 61 min seen under contention before. That leaves about 10 iterations, roughly 13.5 GPU-h.
- Any further infrastructure crash gets a new posted decision before any relaunch.

<!-- posted: https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6402614864 -->
## Decision P-crash (2026-10-26): third Tier-E infrastructure OOM (the plot path again); root fix = the official evaluator's no-decoder branch; resume from the iteration-2 checkpoint on the exclusive GPU 2

**Trigger.** `130-gfid-tier-e-r` run 3 started at 03:50:21 AEDT on GPU 2 with no other job on the card. It resumed from the iteration-2 checkpoint and ended at 04:55:20 with exit 1.
- The error was a CUDA out-of-memory in the **plot-only** decode of the official evaluator's rollout videos: `plan` (MPC loop) → `evaluator.py:132 decode_obs` → `vqvae` `conv_transpose2d`. It happened even with the one-case-per-decode chunking added after run 2. At that point 30.64 GiB was allocated by the process, against a CEM peak of about 28.4 GiB on a 31.36 GiB card.
- This is the same path as run 2's crash (issuecomment-6400401642). Chunking only shrank each decode, not the total, so the earlier fix did not address the root.

**All three crashes (disclosed; none produced or exposed a success value):**

| # | job run | directory | cause | MPC iterations finished |
| --- | --- | --- | --- | --- |
| 1 | `130-gfid-tier-e-r#1` (2026-10-24 22:00 – 2026-10-25 03:31) | `tier-e-2` | orphaned #128 A4 process (18 GB) on GPU 2 | 0 |
| 2 | `#2` (2026-10-25 21:55 – 2026-10-26 02:00) | `tier-e-3` | plot decode OOM beside a co-scheduled #131 job (1.5 GB) | 2 (checkpoint 00:37:18) |
| 3 | `#3` (2026-10-26 03:50 – 04:55) | `tier-e-3` (resumed) | plot decode OOM on the exclusive card | 0 more (iteration 3 not finished) |

The earlier stopped run (`130-gfid-tier-e`, 2026-10-24, `tier-e-1`) was a scheduling stop before its first iteration (issuecomment-6382562140). No `final_eval` value exists, and none has been read.

**Options.**
- An official configuration switch that disables the videos. **Checked: there is none.** `plan_pusht.yaml` has only `n_plot_samples`, and it slices after the full decode. The evaluator draws videos iff `self.wm.decoder is not None` (`planning/evaluator.py:131`), and the decoder comes from the checkpoint.
- Move the decode to the CPU. Rejected: `eval_actions` runs after every CEM step (`eval_every: 1`), so all 50 cases' growing rollouts would be decoded more than 30 times per MPC iteration on the CPU. That adds hours, for videos nothing reads.
- **Chosen: the official no-decoder branch.** The evaluator's world model gets `decoder = None`, the code path of a model trained with `has_decoder: False`.
  - No rollout videos are drawn.
  - CEM/MPC rollouts (encode/predict), the objective, the env rollouts and the PushT `eval_state` success metric never read the decoder.
  - The decode consumes no RNG, so the MPC-iteration checkpoint (including its RNG states) resumes unchanged.
  - Protocol, frozen band (`band.json` sha256 `247747e2…`), checkpoint, cases and metric are unchanged.

**Registration.** This is post-freeze amendment **A2** of `scripts/b127_dino_wm_fidelity.py` in `scripts/post_freeze_amendments.py`.
- `replaces`: the frozen bytes `sha256:ceb3c7bb…` and the A1 bytes `sha256:e5c86145…`.
- `sha256`: `0ba6941b…`.
- `tests/test_issue_127_amendments.py` passes, as do the #101 registry tests. `load_plan` of #130 passes. The CPU resume-equality test (official `plan.py`, 2 cases, 2 MPC iterations) passes under A2.

**Relaunch.** This is the third P-crash relaunch.
- It resumes `tier-e-3` from the iteration-2 checkpoint (saved 2026-10-25T13:37:18Z). Iteration 3 is recomputed from the restored RNG state.
- It runs on GPU 2, exclusive (the orchestrator keeps every other job off the card), from 18:00 on 2026-10-26.
- Any further crash gets a new posted decision.
