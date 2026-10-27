# #138 decisions

## Decision P-license (2026-10-27): vendor facebookresearch/vjepa2 @ 204698b (MIT) verbatim with package-relative imports; released weights used for non-redistributed research evaluation only

**Trigger.** #138 Phase 0 needs the official V-JEPA 2 code (the frozen ViT encoder, the action-conditioned predictor `VisionTransformerPredictorAC`, its teacher-forcing + rollout losses and the optimizer / schedules of `app/vjepa_droid`) for the adapter and G-port, and the released checkpoints for readiness, training features and G-fid.

**Facts.**
- `facebookresearch/vjepa2` `main` @ `204698b45b3712590f06245fbfba32d3be539812` (2026-03-23, "Fix figure (#143)"; V-JEPA 2.1 was added in 45d025f; the V-JEPA 2 / 2-AC code is unchanged since the initial release). Root `LICENSE`: plain MIT, "Copyright (c) Meta Platforms, Inc. and affiliates". README "License": "The majority of V-JEPA 2 is licensed under MIT", except `src/datasets/utils/video/randaugment.py`, `randerase.py` and `src/datasets/utils/worker_init_fn.py`, which are Apache-2.0 (`APACHE-LICENSE`). None of those three files is needed by the port.
- Weights: `https://dl.fbaipublicfiles.com/vjepa2/` (`vjepa2-ac-vitg.pt` 11 760 743 310 B: the AC encoder + predictor; `vitl.pt`; `evals/diving48-vitl-256.pt`). The direct downloads carry no separate licence text. The Hugging Face cards of the same models say: `facebook/vjepa2-vitl-fpc64-256` MIT, `vjepa2-vitl-fpc32-256-diving48` MIT, `vjepa2-vith-fpc64-256` MIT, `vjepa2-vitg-fpc64-256` / `-384` Apache-2.0. There is no Hugging Face card for the AC checkpoint. No non-commercial clause anywhere.

**Options.** (a) Vendor verbatim (allowed by MIT); (b) re-implement under G-port (not needed); (c) drop (`license`): no ground.

**Choice.** (a). Vendor @ 204698b into `world_model/baselines/vjepa2/vendor/` with `VENDOR.json` (schema `issue_127_vendor_v1`, per-file upstream / vendored sha256):
1. Files: `LICENSE`, `README.md`; imported by the adapter: `src/models/ac_predictor.py`, `src/models/vision_transformer.py`, `src/models/utils/{modules,patch_embed,pos_embs}.py`, `src/masks/utils.py`, `src/utils/{tensors,schedulers}.py`; reference copies (read, not imported): `app/vjepa_droid/{train,utils}.py`, `configs/train/vitg16/droid-256px-8f.yaml`; new empty `__init__.py` package markers.
2. Edits: only the allowed one, making the `from src.<...> import` lines of the imported files package-relative (the repository root already has an unrelated `src` package). Each edit is listed in `VENDOR.json`. `modules.py` keeps its `from timm.models.layers import drop_path` line. The `novphy` env has neither timm nor torchvision, so during the vendor load only, the adapter serves `timm.models.layers` with a stub whose `drop_path` raises. `drop_path` runs only when `drop_path_rate > 0`, and no T6 module uses that (the AC config and the hub entries use 0; asserted).
3. Env `b127-vjepa2` (py3.11, **torch 2.13.0+cu130, the novphy build**, torchvision 0.28.0+cu130, timm 1.0.30, einops 0.8.2, decord 0.6.0, the remaining `requirements.txt` packages G-fid needs). It runs the official code from the pinned clone `/mnt/array/sukaih/novphy-baselines-src/vjepa2@204698b/` for the G-port reference and G-fid. Package list: artefact `envs/b127-vjepa2.txt`. The adapter and the runner use `novphy`.
4. Weights go to `data/external/vjepa2/` (gitignored) with `MANIFEST.json` (url, bytes, sha256, downloaded_at). They are used only for NovPhy research evaluation, and are never committed or redistributed.

**Reason.** MIT permits use, modification and redistribution with the notice, which is kept verbatim. The only edit is the import-path change that CONVENTIONS allows. Running the reference under the same torch build keeps G-port exact (as in #134, #136 and #137).

Posted: https://github.com/Sino-Huang/NovPhy/issues/138#issuecomment-6436220188

## Decision P-route (2026-10-27): T6 = the released V-JEPA 2-AC ViT-g/16 encoder (frozen, from `vjepa2-ac-vitg.pt`) + the official AC predictor architecture trained on NovPhy; no smaller encoder

**Trigger.** #138 asks whether T6 uses ViT-g, the encoder whose features the released AC predictor was trained on, or a smaller released V-JEPA 2 encoder. The choice must weigh fidelity against memory and time. The choice fixes the feature cache, the readiness job, the adapter and G-port.

**Facts** (vendored @ 204698b, Decision P-license issuecomment-6436220188; checkpoints in `data/external/vjepa2/MANIFEST.json`):
- The released AC checkpoint `vjepa2-ac-vitg.pt` (11.76 GB; `epoch` 315, `batch_size` 8, `world_size` 32 = global batch 256, i.e. 315 × 300 = 94 500 updates) holds `encoder` (ViT-g/16 `vit_giant_xformers`: 40 blocks, width 1408, 22 heads, RoPE, 256 px, tubelet 2; 1.012 B parameters; tensor-for-tensor equal to its `target_encoder`) and `predictor`. The predictor is `VisionTransformerPredictorAC` (24 blocks, width 1024, 16 heads, RoPE, frame-causal mask with action and state tokens per frame; 305 M parameters; `embed_dim` = 1408). The hub entry `vjepa2_ac_vit_giant` loads both from this file. The training config `configs/train/vitg16/droid-256px-8f.yaml` takes the frozen `target_encoder` of `vitg.pt` (`app/vjepa_droid/train.py`: `forward_target` under `no_grad`) and trains only the predictor, from scratch (`load_predictor: false`).
- Released encoders: ViT-L (300 M), ViT-H (600 M), ViT-g (1 B) at 256 px, and ViT-g at 384 px. Only the ViT-g features have a released AC predictor. Pairing a smaller encoder with the AC predictor is a configuration that the paper never trained or evaluated.
- Measured on CPU (4 threads, fp32; FlopCounterMode): one frame through ViT-g costs 518 GFLOP / 1.92 s, and through ViT-L 155 GFLOP / 0.61 s. Each frame is encoded on its own as a 2-frame tubelet, as `forward_target` does. One native AC predictor update per 8-frame clip (teacher forcing over 7 frames plus the 2-step rollout, forward and backward) costs 4.24 TFLOP, with predictor input width 1408. The predictor dominates training compute. Its cost does not depend on the encoder width apart from the input and output projections.
- The frozen encoder costs one feature pass over the NovPhy frames: about 0.64 M fit frames at stride 5, or 0.28 M for the readiness R1 frame set. That is ≈ 1.4–3.3·10^17 FLOP for ViT-g [INFERENCE: ≈ 0.5–2 GPU-h on one RTX 5090]. The ViT-g model needs ≈ 2 GB of weights in bf16, so memory is not binding. The fp16 cache is 256 × 1408 × 2 B = 0.72 MB per frame (≈ 200 GB for the readiness frames) against 0.52 MB per frame for ViT-L.

**Options.** (a) **ViT-g from the AC checkpoint**: the AC system as released. (b) ViT-L or ViT-H plus the AC predictor architecture: about 3.3× cheaper feature extraction, but a pairing that does not exist upstream. (c) ViT-g at 384 px: no AC predictor was trained on it, and it costs 2.25× more tokens.

**Choice.** (a).
1. **Encoder**: `vit_giant_xformers` with the `encoder` weights of `vjepa2-ac-vitg.pt`, loaded exactly as the hub entry does (`_clean_backbone_key` + `load_state_dict(strict=False)`; the run records any missing or unexpected keys). It is frozen and run in eval mode. Each NovPhy frame is encoded on its own as in `forward_target` (repeated to a 2-frame tubelet; 256 tokens × 1408), then normalised (`F.layer_norm`; `normalize_reps: true`). Features are computed under bf16 autocast, as upstream training does (`meta.dtype: bfloat16`), and cached in fp16.
2. **Preprocessing**: the official `VideoTransform` of `app/vjepa_droid/transforms.py` with the config's `random_resize_scale` [1.777, 1.777] and aspect range [0.75, 1.35]. On a 320 × 240 frame, the area target always exceeds the frame, so `_get_param_spatial_crop` takes its deterministic fallback: the whole image, since 4:3 lies inside the aspect range. Then bilinear `interpolate` to 256 × 256 (`align_corners=False`) and ImageNet mean / std. The adapter re-implements this fallback, and G-port checks it against the official transform.
3. **Predictor**: the official `VisionTransformerPredictorAC` at the config's size (depth 24, width 1024, 16 heads, RoPE, frame-causal, no extrinsics). It is trained from scratch on NovPhy fit-split features, as V-JEPA 2-AC itself was trained from scratch on DROID. The released predictor weights are used only by G-port, which loads them with `strict=True` and runs them through the adapter; they are not used as an initialisation (their action and state spaces are 7-D robot end-effector poses).
4. G-fid and the adapter details (actions, windows, rollout, readout, configurations) follow in Decisions P-fid and P-adapter, each posted before the step it governs.

**Reason.** Only (a) is the V-JEPA 2-AC system as published: the AC predictor architecture is defined over ViT-g features, and the released AC checkpoint fixes both. Its extra cost over ViT-L is one feature pass of ≈ 1 GPU-h [INFERENCE]. That is small next to predictor training, which does not depend on the encoder, so a smaller encoder would weaken fidelity without changing the projection materially.

Posted: https://github.com/Sino-Huang/NovPhy/issues/138#issuecomment-6436321490

## Decision P-adapter (2026-10-27): T6 adapter (frames, launch, clips, losses, optimizer, native budget, rollout, readout) and the three pre-declared configurations

**Trigger.** #138 item 3: the adapter `world_model/baselines/vjepa2/novphy.py` and the selection configurations must be fixed before selection is implemented and before any T6 training. Routes: Decision P-route issuecomment-6436321490 (released ViT-g AC encoder, frozen; official AC predictor trained from scratch); vendored code @ 204698b (issuecomment-6436220188).

**Upstream facts** (`configs/train/vitg16/droid-256px-8f.yaml`, `app/vjepa_droid/{train,utils}.py`, `src/utils/schedulers.py`, released `vjepa2-ac-vitg.pt` metadata):
- Clips of 8 frames at 4 fps. Each frame is encoded on its own by the frozen target encoder (`forward_target`: frame repeated to a 2-frame tubelet, then `F.layer_norm`).
- Loss = teacher forcing plus rollout. TF: the predictor sees frames 0..6 with actions a_0..a_6 and states s_0..s_6 and predicts frames 1..7. Rollout: from [frame 0, the TF prediction of frame 1], `auto_steps` = 2 autoregressive predictions in total. Each prediction is layer-normed, and both terms are the mean `|z − h|` (`loss_exp` 1). Total loss = TF + rollout.
- Optimizer: AdamW (betas 0.9 / 0.999, eps 1e-8). Weight decay applies to non-bias, non-1-D tensors only. The LR schedule is `WSDSchedule`: warm-up 15 × 300 steps from 7.5e-5 to 4.25e-4, constant, then a linear anneal over the last 15 × 300 steps to 0. Weight decay follows `CosineWDSchedule` at 0.04 → 0.04. Training runs under bf16 autocast with a `GradScaler`.
- Native budget: `ipe` 300 × 315 epochs = **94 500 updates** at **global batch 256** (8 per GPU × 32 GPUs; the released checkpoint records `batch_size` 8, `world_size` 32, `epoch` 315).
- **Upstream defect (observed, CPU, env `b127-vjepa2`, official code):** `init_video_model` builds the predictor with `num_frames` = 8 and `tubelet_size` = 2. Its frame-causal mask then covers T = 4 frames (1032 tokens), so the 7-frame TF forward (1806 tokens) raises `RuntimeError: The size of tensor a (1806) must match the size of tensor b (1032)`. The hub entry `vjepa2_ac_vit_giant` builds the predictor with `num_frames` = 64, a mask over 32 frames. The mask content (block-causal, full history) does not depend on `num_frames` beyond its size.

**Choice.**
1. **Frames.** Fit-split shots (`eb.FitShots`, 13 919 complete branches) at stride 5, positions 0, 5, …, 225: 46 frames. Shots that end earlier repeat their last frame, so the scene is static after the end and the actions are no-ops. Each frame is the uint8 320 × 240 #123 pixel-store frame with the official transform (Decision P-route item 2: whole frame, bilinear to 256 × 256, ImageNet normalisation). It is encoded once by the frozen ViT-g under bf16 autocast, as upstream training does. The encoder output (fp32 from the bf16 matmuls, before `layer_norm`) is cached in fp16: [256, 1408] per unique frame, about 0.64 M frames, ≈ 0.46 TB. The cache replaces `forward_target`'s per-step encoder pass, which upstream recomputes every step for the same frozen weights; `layer_norm` is applied at load time.
2. **Launch and state.** The action token carries `eb.launch5` at t0 and the zero no-op at every later step. A clip's action at step k is the launch if and only if its absolute step is 0. The predictor's `action_embed_dim` is 5. NovPhy has no proprioceptive state, so the state token input is the zero vector (the official `state_encoder` is kept; its output is then its bias). There are no extrinsics, as in the config.
3. **Clips and batch.** A clip is 8 consecutive stride-5 frames starting at step s, uniform over the 39 starts of a uniformly drawn fit shot (the official loader draws a random clip of a random trajectory), drawn from `eb.batch_rng(seed, update)`. One update = 256 clips, processed as 32 micro-batches of 8 (the per-GPU batch upstream) with gradients accumulated as `loss / 32`. This is equal to DDP's mean over 32 equal ranks.
4. **Model, loss, optimizer.** The official `VisionTransformerPredictorAC` at the config's size (depth 24, width 1024, 16 heads, RoPE, frame-causal, `uniform_power` true, `use_sdpa` true; 305 M parameters with `embed_dim` 1408), built with `num_frames` = 64 as the hub does (the defect above). The official TF + rollout loss runs with `normalize_reps` and `loss_exp` 1. AdamW param groups follow `init_opt` (predictor groups only: the frozen encoder has no gradients upstream either, and AdamW skips tensors without gradients). `WSDSchedule` and `CosineWDSchedule` are imported unchanged from the vendored `src/utils/schedulers.py` and stepped at the start of each update, as `train_step` does. Training uses bf16 autocast and a `GradScaler`, as upstream. A nonfinite loss is a typed terminal failure. Resumable at every update (SIGTERM → `resume.pt` → exit 75).
5. **Rollout (scoring).** The anchor's decision frame is encoded and normalised. The predictor runs autoregressively for 45 steps to frame 225: launch at step 0, no-ops after, each prediction layer-normed and appended, exactly as `step_predictor` in `notebooks/utils/world_model_wrapper.py` does. The context is a sliding window of the last ≤ 7 frames, the longest context the TF loss trains (frames 0..6 → 7). Beyond 7 frames the oldest frame drops, with its action.
6. **Readout and score** (as C1-dino). R1 = the #130 `PatchReadout` and `fit_readout` recipe (`world_model.baselines.dino_wm.novphy`, imported unchanged; slot-query cross-attention, dim 1408, 256 patches). It is fitted on the normalised true features of the fit frames at `rc.probe_positions` (E99 carrier targets, seed 20260908) in the readiness job. It is frozen there and shared by every configuration and seed, since the encoder is frozen. The primary score is the tie-free `EndpointCosts` cost of R1(predicted frame-225 latent) against R1(anchor latent). The readiness ceiling is the same cost on R1(true endpoint features) on the policy grid cells, threshold 0.65 (typed stop `readiness_ceiling_below_threshold`).
7. **Configurations** (≤ 3, declaration order, all at the native 94 500 updates × 256 clips; seed 20260908; policy-split pooled grid AUC; ties → fewer parameters; the selected run is reused as main seed 20260908):
   - `native`: the released config (LR 4.25e-4 / start 7.5e-5 / final 0, warm-up and anneal 4 500 updates each, WD 0.04, `auto_steps` 2);
   - `ar4`: `native` with `auto_steps` 4. This is the official rollout-loss knob, and NovPhy scoring rolls out 45 steps;
   - `lr-half`: `native` with LR 2.125e-4 and start LR 3.75e-5.
   If a projection forces P-cap, the declared cut order is `lr-half`, then `ar4`.

**Reason.** Every item is the official recipe, with only NovPhy's frames and launch substituted. The departures are (i) the mask size, which repairs an upstream crash by matching the hub's construction, (ii) the feature cache in place of a per-step pass of the same frozen encoder, (iii) zero state and extrinsics, which NovPhy does not have, and (iv) the sliding context window, which keeps the 45-step rollout within the context lengths the predictor was trained on. G-port checks each component against the official code.

Posted: https://github.com/Sino-Huang/NovPhy/issues/138#issuecomment-6436355731

## Decision P-fid (2026-10-27): G-fid = Tier E, the released ViT-L/16 + released Diving-48 probe on the Diving-48 V2 test split (paper 89.0 %); band frozen before any run; GPU job (CPU infeasible)

**Trigger.** #138 G-fid: "the cheapest faithful reproduction of a published V-JEPA 2 / V-JEPA 2-AC number with released weights (Tier E), with a numeric band derived from the paper's reported quantity and frozen here before the run". If no such number is reproducible within the remainder, the arm is not run.

**Facts.**
- **No published V-JEPA 2-AC number can be reproduced with released assets.** The paper's AC results are Franka success rates in two labs (Table 2), planning-time comparisons, and an energy landscape on the authors' own trajectory: Fig. 9 has ground-truth action (Δx, Δy) = (0, −0.1) and minimum ≈ (0, −0.05). App. B.4 sweeps camera positions over 201-step trajectories, which are not released. The only released AC data is `notebooks/franka_example_traj.npz`, a 2-frame example. Its ground-truth action is (0.092, 0.031, 0.084) (computed with the official `poses_to_diff`), so it is not Fig. 9's trajectory, and the notebook stores no outputs or reference number.
- **Published V-JEPA 2 encoder numbers with released probes** (README; Table 4): ViT-L SSv2 73.7 (24 777 validation videos; registration-gated download), **ViT-L Diving-48 89.0**, ViT-L EK100 32.7 R@5, and the ViT-g-384 probes (SSv2 77.3, Diving-48 90.2, EK100 39.7). The ViT-g-384 probes cost ≈ 8× ViT-L per clip (2.25× tokens, 3.3× width). No probe is released for the 256-px ViT-g that T6 uses.
- Diving-48 is public (`svcl.ucsd.edu/projects/resound`, Nextcloud links reachable 2026-10-27): RGB videos (10.3 GB) and splits. V1 test has 2 096 videos; **V2 test** (the 2020 cleaned annotations, "poorly segmented videos removed") has 1 970 videos. The V-JEPA 2 config reads `Diving48_test_paths.csv` and does not say which version. The current standard release is V2 (for example, mmaction2 downloads V2 only).
- Cost: inference config `configs/inference/vitl/diving48.yaml` (32 frames × 4 segments × 3 views, frame step 2, 256 px, multilayer probe on layers 17/19/21/23, bf16, `val_only`). One 32-frame clip is 4 096 tokens, ≈ 4.1 TFLOP through ViT-L. 12 clips × 1 970 videos ≈ 1.0·10^17 FLOP. That is ≈ 105 CPU-h at the measured 4-thread rate (> the 12 CPU-h allowance), and ≈ 0.3–0.7 GPU-h on one RTX 5090 [INFERENCE].

**Options.** (a) An AC number: not reproducible (above). (b) SSv2 ViT-L: about 12× the videos, plus gated data. (c) **Diving-48 ViT-L, V2 test, released probe**: the cheapest. (d) Diving-48 / SSv2 with ViT-g-384: the same architecture class as T6's encoder, at ≈ 8× the cost and a different weight file. (e) Not run: unwarranted, because (c) is reproducible.

**Choice.** (c), Tier E.
1. **Run**: `scripts/b127_vjepa2_fidelity.py`. `--data` (CPU) extracts the videos and writes the official CSVs; every V2 test video must exist and open with decord. `--run` (GPU job `138-gfid`, env `b127-vjepa2`) is the official `python -m evals.main` with the inference config unchanged except its paths: the released `vitl.pt` (`sha256:5346856e…7012`), and the released probe `evals/diving48-vitl-256.pt` (`sha256:6525c2d1…4ddf`) served as `latest.pt`. `--finish` (CPU) writes `fidelity/gfid.json` (schema `issue_127_gfid_v1`).
2. **Number**: the official evaluation's top-1 on the Diving-48 V2 test split (1 970 videos). V1 is not run.
3. **Band** (`fidelity/band.json`, frozen now, `sha256:f45da0b1da8caa21018803c16066ddbb21f198780bc83f2194d3b8121d15e0c3`): the paper's 89.0 ± (2 binomial SE on 1 970 videos + 0.05 for one-decimal rounding) = **[87.54, 90.46] %**. G-fid passes iff the top-1 lies inside the band.
4. **Where**: GPU, night window. The spec is `scripts/gpu_window_issue_138_gfid.json` (est. 60 min, `shares_gpu` false), enqueued by the orchestrator only if T6 is funded. On CPU it would take ≈ 105 CPU-h, beyond the 12 CPU-h allowance.
5. **Failure**: at most two logged repair attempts, each fixing a diagnosed defect of our setup (data, environment), never the band. Then T6 is dropped with `fidelity_not_reproduced`. Selection refuses until `gfid.json` has passed; scoring refuses unless it passed.

**Scope (disclosed).** This G-fid checks the released V-JEPA 2 encoder code path in our environment against a published number: the vendored `VisionTransformer` (RoPE, 3-D patch embedding, SDPA) shared by ViT-L and ViT-g, checkpoint loading, and the official evaluation stack on torch 2.13 + cu130. It does not reproduce a V-JEPA 2-AC result, and none is reproducible. The ViT-g AC weights and the AC predictor are covered by G-port: equality with the official hub loader, `forward_target`, the released predictor's forward, `train_step`, and `step_predictor` on fixed CPU inputs.

**Reason.** This is the cheapest published V-JEPA 2 number that can be reproduced with released weights and public data. The band comes from the paper's own value and the test-set sampling error, and it is fixed before any run.

Posted: https://github.com/Sino-Huang/NovPhy/issues/138#issuecomment-6436495373

## #138 Phase 0 gate results (2026-10-27): G-port passes; G-fid band frozen (GPU spec); readiness is a GPU job (spec); CPU projection 2 109–5 271 GPU-h [INFERENCE]

**G-port** (`gport/gport.json`, schema `issue_127_gport_v1`, **pass**). The reference is the official code from the pinned clone in env `b127-vjepa2`: the hub's `_make_vjepa2_ac_model` with the released `vjepa2-ac-vitg.pt`, `VideoTransform`, `init_opt`, `world_model_wrapper.step_predictor`, and `forward_target` / `forward_predictions` / `loss_fn` / `train_step` executed from the source text of `app/vjepa_droid/train.py`. It is compared with the adapter in env `novphy`, CPU float32, torch 2.13.0+cu130 on both sides. Inputs: 2 real fit shots × 4 stride-5 frames, with the launch at step 0.
- Preprocessing, frozen ViT-g features (load: no missing or unexpected keys) and the released AC predictor's forward (strict load, all 300 tensors) are **bit-identical**.
- Training path, cases `native` (auto_steps 2) and `ar3`: initial state equal; loss, gradients and parameters after two official updates **exact**; LR (7.5078e-5, 7.5156e-5) and WD (0.04) equal.
- The 7-step `step_predictor` rollout is within max |Δ| 2.9e-6 (tolerance 1e-5).

**G-fid** (Decision P-fid issuecomment-6436495373). The band (Diving-48 V2 test, ViT-L released probe) is frozen at **[87.54, 90.46] %** (`fidelity/band.json`, `sha256:f45da0b1…e0c3`). The data are prepared on CPU: all 1 970 V2 test videos present and decodable (`fidelity/data.json`). The run is the GPU job `138-gfid` (`scripts/gpu_window_issue_138_gfid.json`, est. 60 min [INFERENCE]); on CPU it would take ≈ 105 CPU-h. Not enqueued.

**Readiness** (C1-dino's R1 procedure; P-adapter item 6). The ceiling has not been computed. Encoding the 277 292 needed frames costs 148 CPU-h on 4 threads (1.92 s per ViT-g frame, measured), beyond the CPU allowance. Spec: `scripts/gpu_window_issue_138_readiness.json` (`138-features-readiness` 75 min → `138-readiness` 180 min, ≈ 2–4 GPU-h [INFERENCE], ~200 GB NVMe). Its decision is drafted for the orchestrator. Not enqueued.

**Projection** (`projection.json`, [INFERENCE]): CPU FlopCounterMode on the official code paths, mapped to an RTX 5090 at 100 / 40 sustained bf16 TFLOP/s, with a 3 GB/s feature-read bound.
- The native budget is 94 500 updates × 256 clips. One update costs 1.09 PFLOP (`native`, `lr-half`) or 1.93 PFLOP (`ar4`).
- Native per seed: **285–713 GPU-h** (ar4: 507–1 268). Scoring and readouts: 0.74 PFLOP, i.e. 7–19 s, per decision.
- **Child total: 2 109–5 271 GPU-h** against the ≈ 73 GPU-h remainder, the 90 GPU-h child cap and the 24 GPU-h seed cap.
- A ≤ 1 GPU-h timing probe is prepared (`scripts/gpu_window_issue_138_timing.json`, decision drafted). A `cap` not-run decision is also drafted for the orchestrator.

Nothing has been trained or scored, and no GPU time has been used. The evaluation split has not been read.

Posted: https://github.com/Sino-Huang/NovPhy/issues/138#issuecomment-6436833302


## Decision P-coverage (2026-10-27): V-JEPA 2-AC (#138, T6) is not run — typed reason `cap` (projection far above the remainder)

**Trigger.** Program decision #127 issuecomment-6436095504 funds T6 only if its projection fits the remainder (≈ 73 GPU-h after DIAMOND). #138 Phase 0 is complete on CPU:
- licence, route, adapter and fidelity design are posted: issuecomment-6436220188 / -6436321490 / -6436355731 / -6436495373;
- G-port passes (`gport/gport.json`: exact on the official training path, rollout ≤ 2.9e-6);
- the G-fid band is frozen and readiness is a GPU job (both specs prepared, not run).

**Projection** (`projection.json`, [INFERENCE]: CPU FlopCounterMode on the official code paths, mapped to an RTX 5090 at 100 / 40 sustained bf16 TFLOP/s):
- Native budget (the released checkpoint: 315 epochs × 300 iterations = 94 500 updates at global batch 256 = 8 per GPU × 32 GPUs). One update is 256 clips × 4.24 TFLOP = 1.09 PFLOP (ar4: 1.93 PFLOP).
- Native per seed: 285–713 GPU-h (ar4: 507–1 268). Selection (3 configurations) + 2 seeds + features (≈ 1–2.5) + readiness (≈ 0.5–1) + readouts / scoring (≈ 15–37, 0.74 PFLOP per decision) gives **2 109–5 271 GPU-h**.
- The remainder is ≈ 73 GPU-h. Even one native seed at the optimistic bound (≥ 285 GPU-h) is ≈ 3.9× the remainder and ≈ 12× the 24 GPU-h seed cap. The child cap is 90 GPU-h, the upper end of the scout estimate.

**Options.**
- (a) Fund: impossible within the remainder.
- (b) Cut under P-cap. The declared cut order drops `lr-half` and then `ar4`, leaving native × 3 seeds ≥ 855 GPU-h. A shorter schedule would weaken the baseline below its native budget, which #127 forbids for a fair comparison.
- (c) **Not run, typed `cap`**.
- (d) Spend the ≤ 1 GPU-h timing probe first (`draft-decision-timing.md`). The optimistic CPU bound already exceeds the remainder by ≈ 29×, so no plausible measurement changes the outcome. A sustained rate of ~2.9 PFLOP/s would be needed, > 10× the RTX 5090's dense bf16 peak.

**Choice.** (c). Write `not-run.json` `{"typed_reason": "cap", "decision": <this comment>}` and publish the not-run child summary (`python scripts/run_issue_138_vjepa2_ac.py --publish`, CPU). T6 is listed as not run in the program summary. No GPU time is spent on #138; the readiness, G-fid, timing and phase-A specs stay prepared and unqueued.

**Reason.** The projection depends only on the official native budget and the measured FLOPs of the official code paths, not on any result. The decision is the P-coverage rule applied as written.
