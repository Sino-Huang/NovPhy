# #137 decisions

## Decision P-license (2026-10-27): vendor eloialonso/diamond @ `main` 5bcd159 (MIT) verbatim

**Trigger.** #137 Phase 0 needs the official DIAMOND code (EDM denoiser U-Net, its denoising loss and training step, the diffusion sampler, the world-model rollout) for the adapter and G-port, and the official agent / Atari pipeline for G-fid.

**Facts.** `eloialonso/diamond` has three branches: `main` @ `5bcd1599755b4f2fae8e5e079e02f0728e174965` (2024-12-06, the Atari 100k code of the paper), `csgo` @ `851cefb497733d27f1b85c804104638765860fca` (CS:GO world model + upsampler), `ddpm` (Section 5.1 DDPM ablation). The root `LICENSE` is the plain MIT text, "Copyright (c) 2024 Eloi Alonso" (no added clauses). The released weights (Hugging Face `eloialonso/diamond`) carry no separate licence tag; they are used only for the non-redistributed G-fid evaluation and are never committed (`data/external/diamond/`, gitignored).

**Options.** (a) Vendor verbatim (allowed); (b) re-implement from the paper under G-port (not needed); (c) drop (`license`): no ground.

**Choice.** (a). Vendor `main` @ 5bcd159 into `world_model/baselines/diamond/vendor/` with `VENDOR.json` (schema `issue_127_vendor_v1`, per-file upstream / vendored sha256, all equal):
1. Files: `LICENSE`, `README.md`, `requirements.txt`, `config/` (agent/default.yaml, env/atari.yaml, trainer.yaml), `results/data/DIAMOND.json` (the paper's per-game, per-seed Atari 100k returns) and `src/` except `src/game/` and `src/play.py` (pygame UI). 34 files. Not vendored: `scripts/`, `.gitignore`, `src/game/`, `src/play.py`.
2. Edits: none. The official code runs from `vendor/src` on `sys.path` (as upstream `python src/main.py`) in env `b127-diamond` (G-port reference, fidelity runner). The adapter (env `novphy`) loads `src/models/blocks.py` and `src/models/diffusion/{inner_model,denoiser,diffusion_sampler}.py` by file location; `denoiser.py`'s two top-level imports (`from data import Batch`, `from utils import LossAndLogs`) are satisfied during that load only, by the vendored `src/data/batch.py` (+ `segment.py`) and the upstream `LossAndLogs` alias, because upstream `utils.py` imports omegaconf / wandb, which `novphy` lacks.
3. Env `b127-diamond`: py3.11, **torch 2.13.0+cu130 (the novphy build)** instead of upstream `torch==2.4.1` (no kernels for our RTX 5090s; the G-port reference and the adapter then run the same torch), otherwise the upstream pins (gymnasium 0.29.1, ale-py 0.9.0 with bundled ROMs, hydra-core 1.3, numpy 1.26.0, opencv-python 4.10.0.84, h5py 3.11.0, huggingface-hub 0.17.2, pillow 10.3.0, torcheval 0.0.7, tqdm 4.66.4, wandb 0.17.0 disabled); pygame not installed (UI only). The official imports load and `make_atari_env("PongNoFrameskip-v4", ...)` steps on CPU. Package list: artefact `envs/b127-diamond.txt`. The adapter and the runner use `novphy`.

**Reason.** MIT permits use, modification and redistribution with the notice, which is kept verbatim; verbatim vendoring plus a same-torch G-port is the most faithful route (as #134 / #136).

Posted: https://github.com/Sino-Huang/NovPhy/issues/137#issuecomment-6426430212

## Decision P-route (2026-10-27): `main` @ 5bcd159 with the Atari 100k configuration is the reference; released Atari 100k checkpoints (HF @ 5d4abca) serve G-fid

**Trigger.** #137 needs one reference implementation for the port, G-port and G-fid; the ticket fixes the arm to "the official DIAMOND diffusion world model (EDM denoiser conditioned on past frames and actions) ... at DIAMOND's native resolution ... rolled to frame 225 with the paper's sampler".

**Facts (vendored at `main` @ 5bcd159, P-license above).**
- `main` is the code of the paper's Atari 100k results (Table 1; README "hyperparameters used in the paper"). The world model is `models.diffusion.Denoiser` (EDM preconditioning, σ_data 0.5, offset noise 0.3, log-normal training σ (loc −0.4, scale 1.2, clipped to [2e-3, 20])) around `InnerModel` (U-Net 2D, depths [2,2,2,2], channels [64,64,64,64], no attention except the mid blocks, conditioning dim 256, frame stacking of L = 4 previous frames, action + noise-level conditioning by adaptive group norm) = `config/agent/default.yaml` = paper Table 2. Training (`trainer.yaml` + `Trainer.train_component`): AdamW lr 1e-4, wd 1e-2 (decay on Linear / Conv weights only, `utils.configure_opt`), ε 1e-8, linear warm-up 100 steps, gradient-norm clip 1.0, batch 32, windows of L + 1 + `num_autoregressive_steps` = 6 frames (the loss averages 2 predicted frames, the second conditioned on the model's own quantised prediction), fp32 (TF32 matmuls on CUDA, no autocast); 10 000 updates in epoch 1 then 400 per epoch for 1000 epochs = **409 600 updates** (paper Table 3). Sampler (`world_model_env.diffusion_sampler`): **Euler, 3 denoising steps**, Karras σ schedule σ_max 5 → σ_min 2e-3, ρ 7, no churn (paper Sec. 5.2 / App. L: 3 steps, the Table 1 setting). Observations 64 × 64 RGB in [−1, 1] (`AtariPreprocessing`: `cv2.resize(..., INTER_AREA)` of the full frame); rollout = `WorldModelEnv.step` (roll the 4-frame / 4-action buffers, append the sampled frame).
- `main`'s `InnerModel` embeds discrete actions (`nn.Embedding(num_actions, 64)` per conditioning step). The `csgo` branch @ 851cefb embeds multi-hot action vectors through the same table, `act.float() @ act_emb[0].weight` (then the same flatten): the official form for non-categorical actions. The paper's App. M driving model uses continuous actions (steering, acceleration) at 64 × 64; its code is not released. The CS:GO model (56 × 30 + upsampler, 381 M parameters) is not the paper's Atari model.
- Released checkpoints: Hugging Face `eloialonso/diamond` @ `5d4abca9af6ab1b3ab1c6fc228cacb7281130f7f`, one full Atari 100k agent per game (`atari_100k/models/<Game>.pt`, 54.3 MB each: denoiser + reward/termination model + actor-critic) for all 26 games, plus `csgo/`. The card does not say which of the paper's 5 seeds each file is. All 26 sha256 (LFS oids) are in the artefact `hf-checkpoints.json`; the three downloaded for G-fid (`data/external/diamond/hf-5d4abca/`, `MANIFEST.json`, hashes verified): Pong `sha256:addd34bd91f8c8e9c5c7f954c873d4db9fb42440c9028e93bf5e434c5ecb9042`, Boxing `sha256:fcb54ee18a7a19371800c86f18afcf78050bd453df99878601d3be9e0e7a5e07`, Freeway `sha256:fdc18d43fa8ee5c66e62ca3544d34e42d0395b4bb7b5e6792faca9859cdc029e`. The repository's `results/data/DIAMOND.json` holds the paper's per-seed returns (5 seeds × 26 games), the G-fid band source.

**Options.** (a) `csgo` branch model (not the paper's Atari model; 381 M parameters at 56 × 30); (b) **`main`, Atari 100k configuration**; (c) re-implement App. M's driving model (no code).

**Choice.** (b):
1. **Model**: the vendored `Denoiser` / `InnerModel` at `config/agent/default.yaml`, imported unchanged; the launch is a 5-d action vector (`eb.launch5`; zeros = no-op) embedded by the official table as in the `csgo` branch (`launch5 @ act_emb[0].weight`, `num_actions` = 5); frames 64 × 64 (the native resolution); 45 autoregressive steps at frame stride 5 to frame 225 with the official Euler 3-step sampler.
2. **Training**: `Denoiser.forward` (the official loss) with the `Trainer.train_component` step semantics (AdamW groups of `configure_opt`, warm-up `LambdaLR`, clip 1.0), re-implemented without Hydra / wandb in `world_model/baselines/diamond/novphy.py`; window sampling = the official `BatchSampler` rule on a static dataset (`ignore_sample_weights`), drawn from `eb.batch_rng`.
3. **G-port** runs the official code (env `b127-diamond`, same torch) against the adapter on fixed CPU inputs: forward, denoising loss, gradients, two optimizer steps (`Trainer.train_component` itself), one sampler call and a full 45-step rollout (`WorldModelEnv.step` buffer semantics).
4. **G-fid** (Decision P-fid, posted before any fidelity run): Tier E with the released checkpoints and the paper's evaluation protocol (`Trainer.collect_test(final=True)`: 100 test episodes, ε 0, no life-loss termination) on a few games; band from `DIAMOND.json`'s 5-seed spread.
5. Data, configurations, readout, readiness stop: Decision P-adapter, posted before any of them is computed.

**Reason.** `main` is the code and configuration behind the paper's published Atari numbers and its released checkpoints; it is the only route with both a released model to reproduce and the native 64 × 64 frame-stacking world model the ticket names.

Posted: https://github.com/Sino-Huang/NovPhy/issues/137#issuecomment-6426430665

## Decision P-other (2026-10-27): T5 DIAMOND-px adapter — data, readout, readiness stop, three pre-declared configurations, native update count and batch, sampler

(The ticket calls this Decision P-adapter.)

**Trigger.** #137 needs the NovPhy adapter of the official DIAMOND world model (Decision P-route: `main` @ 5bcd159, Atari 100k configuration) and its ≤ 3 selection configurations, fixed before the readiness check, the G-port and any selection run. Nothing has been trained or scored; the evaluation split has not been read.

**Facts.**
- The official denoiser at `config/agent/default.yaml` with `num_actions` = 5 has **4 406 019** parameters. CPU micro-benchmark (4 threads, fp32): one official update at batch 32 (6-frame windows, 2 predicted frames) 3.6 s; one 45-step rollout of 16 candidates (Euler, 3 denoising steps) 26 s. Paper App. I: 88 ms per denoiser update on an RTX 4090.
- `resize_frames` (exact area averaging, float64, round-half-even) is bit-identical to `cv2.resize(..., INTER_AREA)` (opencv 4.10, the official preprocessing) on 42 fit frames (max |Δ| 0).

**Choice** (`world_model/baselines/diamond/novphy.py`; the vendored code is unchanged).
1. **Model**: the official `Denoiser` / `InnerModel` (U-Net [2,2,2,2] × [64,64,64,64], conditioning dim 256, L = 4 stacked frames, EDM σ_data 0.5, offset noise 0.3, training σ log-normal (−0.4, 1.2) clipped to [2e-3, 20]). Action = `eb.launch5` through the official action table in the `csgo` branch's multi-hot form (`launch5 @ act_emb[0].weight`, the table initialised exactly as `nn.Embedding(5, 64)`; zeros = no-op embed to zero).
2. **Frames and episodes** (fit split only): 64 × 64 RGB in [−1, 1] (DIAMOND's native resolution; `u8/255·2−1` as `TorchEnv`), area-resized from the #123 fit pixel store (`eb.FitShots().frame`, 320 × 240, full frame, 4:3 → 1:1 as Atari's 210 × 160 → 64 × 64). Each of the 13 919 fit shots is an episode of 49 frames: the decision frame 3 extra times (the scene is static before the launch) then positions 0, 5, …, 225 (stride 5; a shot ending earlier repeats its last frame); actions zero except the launch at the decision frame, so the first prediction (frame 5) is conditioned on the decision frame ×4 and actions [no-op, no-op, no-op, launch]. Cache: `~/.cache/novphy-issue-137-diamond-frames/fit-frames.u8` (13 919 × 46 × 3 × 64 × 64, 7.9 GB) + the policy anchor / policy true-endpoint / 480 fit-anchor frames at 64 × 64 (the #135 sources), sha256 in `data/frames-manifest.json`. **Small-object risk (disclosed):** the 5 × 3.75 downscale leaves a pig a few pixels wide [INFERENCE]; the readiness check below measures whether the cost survives it.
3. **Training** (the official recipe): `Denoiser.forward` (official loss over L + 1 + `num_autoregressive_steps` frame windows; with AR steps 1 it averages 2 predicted frames, the second conditioned on the model's own quantised prediction); windows drawn by the official `BatchSampler` rule for a static dataset (episodes uniform, padding only before the start) from `eb.batch_rng(seed, b)`; the `Trainer.train_component` step (backward, clip 1.0, AdamW with `configure_opt` groups, lr 1e-4, wd 1e-2, ε 1e-8, linear warm-up 100); **batch 32**; fp32 with TF32 on CUDA as upstream (no autocast). `resume.pt` every 2 000 updates and on SIGTERM → exit 75 (torch RNG streams checkpointed; batch b depends only on (seed, b)); a nonfinite loss ends the run (`nonfinite_loss`, `model.pt` with `complete False`, never retried).
4. **Configurations** (selection: seed 20260908, full budget, policy-split pooled grid AUC of the primary score, ties → fewer parameters (equal here) → declaration order; the selected run is reused as main seed 20260908; then seeds 20260909 / 20260910):
   - **`native`** — the Atari 100k recipe exactly: **409 600 updates** (10 000 + 999 × 400, paper Table 3), batch 32, AR steps 1.
   - **`ar4`** — the same with `num_autoregressive_steps` 4 (the `csgo` branch's training value; 9-frame windows, 5 predicted frames, 4 of them conditioned on the model's own predictions): the official knob for longer autoregressive rollouts (ours: 45 steps vs Atari's 15-step imagination).
   - **`native-2x`** — the Atari recipe for **819 200 updates** (our fit set has 682 k frames vs Atari 100k's 100 k, so the native count is ~19 window draws per frame instead of ~131).
5. **Sampler and score**: the official sampler (Euler, **3 denoising steps**, Karras σ 5 → 2e-3, ρ 7, no churn: paper Table 1 setting); rollout = `WorldModelEnv.step` buffer semantics, 45 steps from the decision frame. **S = 8 paths per candidate with common random numbers**: path k's initial sampler noise at rollout step t is one fixed draw (CPU generator seed 20261027, [8, 45, 3, 64, 64]) shared by every candidate and level (`sample_next` = the official `DiffusionSampler.sample` body with its initial noise supplied; G-port checks it against the official call); one level's 16 × 8 rows per batch. Primary score = tie-free `EndpointCosts` row of the **mean over the 8 paths of readout(predicted frame 225)** against readout(decision frame). Nonfinite → excluded typed row.
6. **Readout** (G-readout applies): `FrameReadout` = the #135 `ParticleReadout` architecture over 64 patch tokens of the 64 × 64 frame (8 × 8 × 3 pixels + patch centre) and a mean-colour token, 19 slot queries → the 236-d E99 carrier, the #130 R1 recipe (dim 256, 8 heads, MLP 512, AdamW 1e-3 / wd 1e-4, batch 128, 25 epochs, MSE on z-scored carriers, seed 20260908); 652 559 parameters. Fitted **once** on true fit frames only (every fit shot at positions 0, 15, …, 225 clamped; E99 targets from the #123 fit memmap), on CPU in Phase 0; it does not depend on the world model, so the same frozen readout serves every configuration and seed (sha256 pinned in `plan.json`).
7. **Readiness stop (pre-registered, as C1-dino's R1)**: ceiling = pooled policy-grid AUC (94 mixed cells) of tie-free `EndpointCosts` of readout(true endpoint frame at min(225, end), 64 × 64) against readout(policy anchor frame); the E99 true-carrier ceiling (#130: 0.7223) reported beside it. **Ceiling < 0.65 → T5 stops** with typed reason `readiness_ceiling_below_threshold` before any GPU work (no timing probe, selection or scoring; `not-run.json` and the not-run child summary, as #135). ≥ 0.65 → Phase 0 continues. One computation, no re-run, no change of the readout after its result.
8. **Guards**: G-rank (480 fit anchors × 16 grid launches, primary cost), G-action (`eb.action_sensitivity` on the path-mean readout carriers at frame 225), G-readout (item 7's ceiling), G-nonfinite, G-trained, the wrong-anchor control, the #127 global gates; G-port and G-fid as extra guards. Position error descriptive (readout carriers at frames 15, 30, 60, 150, 225 = rollout steps 3, 6, 12, 30, 45). No G-collapse (pixel state; G-readout is the representation guard); no native success head, so no U + head control.
9. **Cap**: child cap 136 GPU-h (upper end of the #127 scout estimate), 24 GPU-h per native seed, 48 h wall on 4 GPUs; funding follows the measured projection (P-coverage); `--selection` / `--train` refuse without the orchestrator's `funded.json`, scoring refuses unless G-fid passed.

**Reason.** The model, loss, optimizer, batch, update count and sampler are the paper's Atari 100k recipe; the launch uses the official table in the code's own non-categorical form; the readout and readiness rule repeat the pixel arms' (#130, #135) so the arm stops cheaply if 64 × 64 frames cannot carry the cost. The two alternatives change one official knob each (autoregressive training length; update count for a 6.8× larger dataset), fixed before any result.

Posted: https://github.com/Sino-Huang/NovPhy/issues/137#issuecomment-6426541110

## Decision P-fid (2026-10-27): G-fid = Tier E, released Atari 100k agents on Pong, Freeway, Boxing with the paper's evaluation protocol; bands frozen before any run

**Trigger.** #137 G-fid: "the cheapest faithful reproduction of a published DIAMOND number (released checkpoints → Tier E, e.g. per-game Atari 100k returns of the released agents with the paper's evaluation protocol), with a numeric band derived from the paper's reported quantity and frozen here before the run".

**Facts.**
- Released checkpoints (Decision P-route): one full Atari 100k agent per game on Hugging Face `eloialonso/diamond` @ 5d4abca (denoiser + reward/end model + actor-critic, 54.3 MB). The paper's per-game number (Table 1) is the mean over 5 seeds of each seed's final evaluation; the repository's `results/data/DIAMOND.json` holds the 5 per-seed values of every game. The card does not say which seed each released file is.
- Evaluation protocol in the code (`Trainer.collect_test(final=True)`): the `test` env of `config/env/atari.yaml` (`<Game>NoFrameskip-v4`, frame skip 4 with max-pooling, up to 30 no-op starts, 64 × 64 `INTER_AREA`, **no** termination on life loss, no step cap), the official collector / env loop with ε 0.0 (actions sampled from the actor-critic's categorical), **100 episodes**, mean return.
- CPU cost (4 threads, measured with an untrained actor-critic on Pong, no released weights): 3.7 ms per agent step including the ALE. Pong (one 21-point game), Freeway (fixed 2 min 16 s = 2 048 agent steps) and Boxing (≤ 2 min ≈ 1 780 steps) have the shortest bounded episodes: 3 × 100 episodes ≈ 0.6 M steps ≈ 40–60 CPU-min. A 26-game HNS (1.46) reproduction is ~10× that with unbounded episodes.
- 5-seed spread (`DIAMOND.json`): Pong 19.3, 20.9, 20.9, 20.7, 20.0 (mean 20.36, SD 0.70; Table 1: 20.4); Freeway 33.8, 33.7, 32.4, 33.0, 33.7 (33.32, SD 0.61; Table 1: 33.3); Boxing 99.4, 84.0, 97.5, 71.4, 82.2 (86.90, SD 11.61; Table 1: 86.9).

**Options.** (a) The 26-game mean HNS (most expensive, unbounded episodes); (b) one game (weak: one checkpoint); (c) **three games with short bounded episodes and tight relative spreads** (Pong and Boxing are also in the paper's App. L ablation).

**Choice.** (c), Tier E.
1. **Run**: `scripts/b127_diamond_fidelity.py --run` in env `b127-diamond` on CPU (fp32, 4 threads), the released checkpoints (sha256 Pong `addd34bd…9042`, Freeway `fdc18d43…029e`, Boxing `fcb54ee1…5e07`, full hashes in `band.json`), agent and env built exactly as `play.py --pretrained`, the official collector with ε 0.0, **100 episodes per game**. Resumable per episode: episode i is one `collector.send(NumToCollect(episodes=1))` (`reset_every_collect`, a fresh env loop per collect) after seeding random / numpy / torch with 20260908 + 1000 · (game index) + i, so an interrupted episode replays identically (SIGTERM → exit 75). Disclosure: the official final collect draws its 100 episodes in one call (later episodes follow the vector env's autoreset); here each episode starts from its own seeded reset; the per-episode distribution (no-op starts from the env RNG, stochastic policy) is the same.
2. **Number**: per game, the mean return over the 100 episodes.
3. **Band** (`fidelity/band.json`, frozen now, `sha256:a21b44eca6e9b7962a674c66a1a0fbf0d9a60d3ba49f4ee5f2f458d4f1fc0cca`): the paper's 5-seed mean ± 2 SD per game — **Pong [18.963, 21.757], Freeway [32.108, 34.532], Boxing [63.676, 110.124]**; a game passes iff its mean lies inside its band; **G-fid passes iff all three pass**.
4. **Where**: CPU, daytime, now (≈ 1 CPU-h); `fidelity/gfid.json` (schema `issue_127_gfid_v1`) by `--finish`.
5. **Failure**: at most two logged repair attempts, each fixing a diagnosed port / environment defect, never the band; then T5 is dropped with `fidelity_not_reproduced`. Selection refuses until `gfid.json` passes; scoring refuses unless it passed.

**Reason.** The released agents and the official evaluation code reproduce the paper's own reported quantity at the lowest cost; the band is the paper's own seed spread. Together with the bit-exact G-port of the world model's loss, optimizer steps, sampler and rollout (`gport/gport.json`), this checks both the released DIAMOND pipeline in our environment and our port of its world model.

Posted: https://github.com/Sino-Huang/NovPhy/issues/137#issuecomment-6426803157

band.json sha256 at freeze: sha256:a21b44eca6e9b7962a674c66a1a0fbf0d9a60d3ba49f4ee5f2f458d4f1fc0cca

## Decision P-other (2026-10-27): readiness ceiling 0.6783 ≥ 0.65 — T5 is not stopped; Phase 0 continues

**Trigger.** The pre-registered readiness stop (Decision P-adapter, issuecomment-6426541110, item 7): the shared `FrameReadout`'s policy ceiling on true 64 × 64 frames, computed on CPU before any GPU work; < 0.65 stops T5 with `readiness_ceiling_below_threshold`.

**Result** (`readiness/readiness.json`, one run; fit / policy data only, evaluation not read; `deterministic_scoring()`):
- Readout: `FrameReadout` (652 559 parameters) fitted once on **220 887 true fit frames** (13 919 fit shots at positions 0, 15, …, 225 clamped; E99 targets), the #130 R1 recipe (25 epochs, 43 125 AdamW steps, batch 128, seed 20260908), CPU 4 threads, 75 min; final batch loss 0.051 (z-scored MSE); `readiness/readout.pt` `sha256:7c6d6bbb82071aa3e0d11f524fc643a12548795207aa4f3c2555c7d7fbeb0b70` — the frozen readout of every configuration and seed.
- **Ceiling = 0.6783** (pooled policy-grid AUC, 94 mixed cells, 1 504 candidates; tie-free `EndpointCosts` of readout(true endpoint frame at min(225, end)) against readout(anchor frame)); threshold 0.65 → **pass**. The E99 true-carrier ceiling on the same cells is 0.7223. Readout error against the E99 carriers of the same policy endpoints: carrier MAE 0.0157, pig-centre MAE 0.0211, pig-presence MAE 0.0012.

**Consequence.** No stop: the 64 × 64 frames carry the cost through the readout up to 0.678 (the small-object risk costs 0.044 AUC against the E99 carriers, disclosed). Phase 0 continues (G-port, G-fid, timing probe); the readiness record is also T5's G-readout value.

Posted: https://github.com/Sino-Huang/NovPhy/issues/137#issuecomment-6428213317

## #137 Phase 0 gate results (2026-10-27): G-port bit-exact; G-fid (Tier E) passes; CPU projection [INFERENCE]

**G-port** (`gport/gport.json`, schema `issue_127_gport_v1`, **pass, exact**). Official code (`Trainer.train_component`, `utils.configure_opt` / `get_lr_sched`, `Denoiser.forward`, `DiffusionSampler.sample`, `WorldModelEnv.step`, the official `Episode` / `make_segment` / `collate_segments_to_batch`; objects instantiated from the vendored `config/*.yaml`; env `b127-diamond`) against the adapter (env `novphy`), CPU float32, torch 2.13.0+cu130 on both sides, three cases: `atari-discrete` (the unchanged official model, 6 discrete actions, full LR 1e-4), `launch` (num_actions 5, the `csgo`-branch real-valued action table; real NovPhy fit windows incl. left padding and the launch frame; warm-up steps at LR 0 and 1e-6; plus a full **45-step rollout** of 2 fit anchors under 2 grid launches) and `launch-ar4` (9-frame windows). Initial state, the official batch, forward loss, both optimizer steps (loss, LR, pre-clip gradient norm, post-clip gradients, parameters), one sampler call and all 45 rollout frames: **all deltas 0**. Injected defects are detected (CPU tests): launch one frame early (batch differs), no warm-up (LR differs), one noise draw reused at every rollout step (rollout differs).

**G-fid** (`fidelity/gfid.json`, schema `issue_127_gfid_v1`, **pass**; band frozen before the run, issuecomment-6426803157, `sha256:a21b44ec…0cca`). Released agents (HF @ 5d4abca), official test env and collector, ε 0, 100 episodes per game, CPU (≈ 17 CPU-min per game), 0 repair attempts:

| game | ours (100 episodes, mean ± SE) | paper 5-seed mean (Table 1) | band (mean ± 2 SD) | pass |
| --- | --- | --- | --- | --- |
| Pong | 20.89 ± 0.05 | 20.36 (20.4) | [18.963, 21.757] | yes |
| Freeway | 32.22 ± 0.08 | 33.32 (33.3) | [32.108, 34.532] | yes |
| Boxing | 90.74 ± 0.49 | 86.90 (86.9) | [63.676, 110.124] | yes |

(Freeway lies 0.11 above its band's lower end; the paper's own seeds span 32.4–33.8.)

**Projection** (`projection.json`, [INFERENCE] CPU micro-benchmark; the measured probe replaces it): FLOPs per official update native 1.17 / ar4 2.92 TFLOP (batch 32), per scoring decision 105 TFLOP (16 candidates × 8 paths × 45 steps × 3 denoising steps + readouts). Mapped to an RTX 5090 at 40 / 10 sustained TFLOP/s: **40.3–161.1 GPU-h** total (selection 3.3–13.3 + 8.3–33.3 + 6.7–26.6; two seeds at ar4 16.6–66.5; readouts 1.3–5.0; scoring 3.9–15.5), serial-ladder wall 21.1–84.5 h. Paper anchor (App. I RTX 4090 timings, unscaled): **109.7 GPU-h**, wall 53.9 h (ar4 25.0 GPU-h per seed). The range straddles the program remainder (≈ 122 GPU-h): a ≤ 1 GPU-h timing probe on an exclusive GPU is drafted for the orchestrator (`draft-decision-timing.md`, spec `scripts/gpu_window_issue_137_timing.json`, not enqueued). Nothing has been trained on a GPU; the evaluation split has not been read.

Posted: https://github.com/Sino-Huang/NovPhy/issues/137#issuecomment-6428213735

## Decision P-coverage (2026-10-27): a GPU timing probe (≤ 1 GPU-h, exclusive GPU) to measure DIAMOND's projection before the funding decision

**Trigger.** Program decision #127 issuecomment-6426185139: DIAMOND (#137, the next coverage arm) is funded only if its measured projection fits the remainder (≈ 122 GPU-h after #136 and #134). #137 Phase 0 is complete on CPU: vendoring (issuecomment-6426430212), route (issuecomment-6426430665), adapter (issuecomment-6426541110), G-port bit-exact (`gport/gport.json`), the readiness ceiling 0.6783 ≥ 0.65 (issuecomment-6428213317), G-fid Tier E passed on CPU (Pong 20.89, Freeway 32.22, Boxing 90.74, all inside the frozen bands; issuecomment-6428213735). The CPU projection (`projection.json`, [INFERENCE]: FLOPs of one official update per configuration (native 1.17, ar4 2.92 TFLOP at batch 32) and of one scoring decision (16 candidates × 8 paths × 45 steps × 3 denoising steps + readouts: 105 TFLOP), mapped to an RTX 5090 at 40 / 10 sustained TFLOP/s; and the paper's own RTX 4090 timings (App. I: 88 ms per denoiser update, 4.2 ms per denoising step of 32 rows) unscaled as a pessimistic anchor) is **40.3–161.1 GPU-h** (FLOP model), **109.7 GPU-h** (paper anchor; selection 10.0 + 25.0 + 20.0, two seeds at ar4 50.1, readouts 1.1, scoring 3.3), serial-ladder wall **21.1–84.5 h** (paper anchor 53.9 h, over the 48 h cap only if ar4 is selected; ar4 per seed 25.0 GPU-h vs the 24 GPU-h seed cap). The range straddles the remainder; only a GPU measurement can decide.

**Decision.** Run one probe on an exclusive GPU in the night window: `scripts/run_issue_137_diamond.py --timing` (job `137-timing`, spec `scripts/gpu_window_issue_137_timing.json`, GPU 1 at 18:00, `shares_gpu: false`; the GPU-1 jobs `134-select-g0.978` and `133-select-lr3e-4` wait for it, so no other job is on GPU 1 meanwhile; est. 15 min). It loads the 64 × 64 fit frame cache and measures, keeping no model:
1. 300 official updates of `native` (= `native-2x` per update; batch 32, 6-frame windows, fp32 + TF32 as upstream) and 300 of `ar4` (9-frame windows), on fit-split batches drawn as in training;
2. the full scoring path of 16 fit anchors (after one warm-up level): 45-step rollouts of the 16 grid candidates × 8 paths with the official Euler 3-step sampler (common random numbers, 128 rows per batch) + the shared readout, under `deterministic_scoring()` — the cost per decision.

`timing.json` (+ `ledger/timing.json`) then becomes the primary (measured) projection in `projection.json` (`projection_measured`: three selection runs, two more seeds at the most expensive configuration, selection readouts, readouts and scoring of three seeds incl. novelty; G-fid already done on CPU, 0 GPU-h). **Cap: 1 GPU-h of device time, charged to #137.** Nothing from the probe is reused by a later run. The funding decision (`funded.json`, or `not-run.json` with typed reason `cap`) stays separate and follows the measured projection; if the measured projection exceeds the wall (48 h) or seed (24 GPU-h) cap, P-cap applies (proposed cut order, fixed before any selection result: `ar4` first, the most expensive configuration).

## Decision P-coverage (2026-10-27): T5 funded

Funded under P-coverage (program decision https://github.com/Sino-Huang/NovPhy/issues/127#issuecomment-6436095504): the exclusive-GPU timing probe measures **80.3 GPU-h** (selection 6.6 + 15.9 + 13.2; two more seeds 31.7; readouts 3.1; scoring 9.4; serial-ladder wall 42.6 h), within the child, seed and wall caps and the remainder (≈ 153 GPU-h). `funded.json` written by the orchestrator. Selection runs: native GPU 0, ar4 GPU 1, native-2x GPU 3 (GPU 2 reserved for #128's A4).

## Decision P-other (2026-10-28): selection result — `native` (pre-declared rule); main seeds 20260909 / 20260910 train `native`

**Trigger.** The three pre-declared configurations (Decision P-adapter, issuecomment-6426541110) finished their seed-20260908 runs at the full budget; the rule (`eb.select_configuration`: largest policy-split pooled grid AUC at 4 d.p., ties → fewer parameters → declaration order) picks the configuration whose run is reused as main seed 20260908. Fit / policy data only; the evaluation split has not been read.

**Result** (`selection.json`, `deterministic_scoring()`, 94 policy grid mixed cells, 16 candidates × 8 paths each, the frozen readout):

| configuration | updates | policy grid AUC | training GPU-h (shared GPUs) | typed failure |
| --- | --- | --- | --- | --- |
| **native** | 409 600 | **0.6863** | 8.82 | none |
| ar4 | 409 600 | 0.6802 | 15.40 | none |
| native-2x | 819 200 | 0.6523 | 12.85 | none |

All three have 4 406 019 parameters. **Selected: `native`** (the Atari 100k recipe exactly).

**Smoke** (`smoke.json`, pass): accounting (13 919 fit shots, 480 fit anchors, 94 policy cells, 220 887 readout frames), G-port record current (adapter sha unchanged), G-fid / readiness / funding gates pass, CPU tests 21 passed. Projection of what is left at the measured rates: two seeds 17.6 GPU-h (8.8 each), readouts 3.1, scoring + novelty 9.4 → **child total 67.2 GPU-h** (37.6 already in the ledger incl. timing and selection readouts) against the funded 80.3 GPU-h and the 136 GPU-h cap; serial wall 19.2 h on 4 GPUs (cap 48 h). Next: the orchestrator's `--prepare` freeze, then phase B (`scripts/gpu_window_issue_137_main.json`).

Posted: https://github.com/Sino-Huang/NovPhy/issues/137#issuecomment-6466769084

## Decision P-freeze (2026-10-28): #137 frozen — T5 selected `native`

**Selection** (issuecomment-6466769084; frozen rule: largest policy-split pooled grid AUC at 4 d.p.; ties → fewer parameters → declaration order), seed 20260908, 94 policy cells, deterministic scoring:

| configuration | policy grid AUC | parameters |
| --- | --- | --- |
| **native (selected)** — the Atari-100k recipe unchanged | **0.6863** | 4 406 019 |
| ar4 | 0.6802 | 4 406 019 |
| native-2x | 0.6523 | 4 406 019 |

The selected run is reused as seed 20260908; seeds 20260909 and 20260910 train `native`.

**Smoke** (`smoke.json`, pass): accounting (13 919 fit shots, 480 fit anchors, 94 policy cells, 220 887 readout frames), G-port current, funding / G-port / G-fid / readiness gates pass, 21 CPU tests. Projection of the rest at measured rates: two seeds 17.6 GPU-h, readouts 3.1, scoring + novelty 9.4 → child total 67.2 GPU-h (funded 80.3; cap 136); wall 19.2 h.

**Frozen plan**: `plan.json` sha256 `178b1d4044dfa9f3d027bd01a2c5b9af33ffc8c9aa7899097ed6790e85ac5024`, 147 pins, frozen_at 2026-10-28T10:36:53Z. Training queues: q0 T5 s20260909 (GPU 0), q1 T5 s20260910 (GPU 1); readouts, freeze-readouts, one scoring pass (GPU 0), novelty, publish + validate chained (`scripts/gpu_window_issue_137_main.json`).

## Decision P-other (2026-10-30): reporting amendment A1 — the published trainable-parameter count

**Trigger.** The first `--publish` (job `137-publish-validate`, after the three primary records and the novelty pass) reports T5's `trainable_parameters` as **0** in `summary.json` / `compute.json` / `findings.md`. Cause: `readouts_arm` counted `requires_grad` parameters on the frozen model (`load_trained` sets `requires_grad_(False)`), and `arm_compute` copies that count. The trained model has **4 406 019** parameters (`model.pt` payload, each `report.json`, `selection.json`; the selection rule used the payload value). The program table and the manuscript copy this field, so a published 0 would propagate.

**Options.** (a) Keep the frozen output and disclose the right count; (b) a reporting-only post-freeze amendment.

**Choice (b), orchestrator decision.** Amendment **#137 A1**, registered in `scripts/post_freeze_amendments.py` (replaces the frozen runner bytes `sha256:5547ce58…f816` with `sha256:5aae6fb6…7b37`): one line of `scripts/run_issue_137_diamond.py`, where `arm_compute` takes the parameter count from each arm-seed's training `report.json`. Nothing else changes. The records, readouts, guards (T5 fails G-action on every seed), tokens and the disposition are untouched, as are the budget safeguard and MACs per candidate. `tests/test_issue_127_amendments.py` checks that the registry replaces exactly the frozen bytes, that `load_plan` accepts the amended runner and that the published count is 4 406 019. Then one job `137-republish-validate` (GPU 0, exclusive; spec `scripts/gpu_window_issue_137_republish.json`), after the running `137-publish-validate`, re-derives the publication and re-validates it (records hash-checked, seed 20260908 re-scored with exact row equality, publication re-derived twice byte-identically).

Posted: https://github.com/Sino-Huang/NovPhy/issues/137#issuecomment-6500578672

## Addendum to Decision P-other (2026-10-30) (issuecomment-6500578672): A1 also covers one test-setup line

Running the #137 CPU tests for A1 turned up a second issue: 20 passed, 1 failed. The CPU ladder test linked the whole real `data/` directory into its temporary output. Since `--score`, that directory also holds the read evaluation anchors (`anchors-evaluation.pt`), so the test's `freeze_readouts` refused with "evaluation anchors already read: G-chronology would fail". The guard behaved correctly: the defect is in the test setup, not the runner. A1 therefore also replaces the frozen `tests/test_issue_137_diamond.py` (`sha256:09d781fa…c655` → `sha256:d14ef283…f6d`): the test now links the three fit / policy data products one by one. Registered in `scripts/post_freeze_amendments.py`, and checked by `tests/test_issue_127_amendments.py`. With the amendment: #137 tests 21 passed; amendment-registry tests (incl. the #101 registry checks) 6 passed. No runner behaviour, record, guard, token or disposition changes.

Posted: https://github.com/Sino-Huang/NovPhy/issues/137#issuecomment-6500794143
