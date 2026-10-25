# #135 decisions

## Decision P-license (2026-10-25): vendor taldatech/ddlp @ 5686275 (MIT) — the DDLP model closure, verbatim

**Trigger.** Phase 0 of #135 vendors the official DDLP code (rule P-license).

**Facts checked.** https://github.com/taldatech/ddlp at `5686275800ee58d9832714af5ed8363c4fb606bc` (2026-03-25, "added interactive gui"; the model code is unchanged since the 2023-09-28 "initial code release") carries a root `LICENSE`: MIT, "Copyright (c) 2023 Tal Daniel" (sha256 recorded in `VENDOR.json`). MIT permits use, copying and redistribution with the notice. The released checkpoints and the PHYRE data are hosted by the authors on MEGA.nz and linked from the README without a separate licence; they are used here for non-commercial research evaluation only, stored under the gitignored `data/external/ddlp/` and never committed or redistributed.

**Options.** (a) vendor the official code verbatim; (b) re-implement from the paper (only without a licence). The licence allows (a).

**Decision.** Vendor into `world_model/baselines/ddlp/vendor/` (schema `issue_127_vendor_v1`, per-file upstream / vendored sha256): `models.py` (`ObjectDynamicsDLP`: DLPv2 particle encoder with the tracking posterior, background module, object decoder, `calc_elbo` with every loss term, `sample`), `modules/modules.py` (CNN / patch-prior / particle-attribute / glimpse encoders and decoders), `modules/dynamics_modules.py` (PINT: `DynamicsDLP`, particle projection, causal particle Transformer with temporal and particle relative-position bias, particle decoder), `utils/loss_functions.py` (ChamferKL, KL terms, reconstruction losses) and `utils/util_func.py` (spatial transformer, correlation maps, masks, reparameterisation), plus the upstream `LICENSE`. The only edits are package-relative imports (listed per file in `VENDOR.json`) and new `__init__.py` package markers. The training loop (`train_ddlp.py`), datasets and the evaluation script are not vendored: the G-port reference and the fidelity harness run them from the read-only clone `/mnt/array/sukaih/novphy-baselines-src/ddlp@5686275/`.

Posted: https://github.com/Sino-Huang/NovPhy/issues/135#issuecomment-6396690148

## Decision P-route (2026-10-25): the released DDLP checkpoints are reachable → G-fid stays Tier E (PHYRE-128); official code on torch 2.13 in env `b127-ddlp`

**Trigger.** #135 makes checkpoint reachability the first decisive Phase-0 check: the README's model zoo hosts DDLP checkpoints on MEGA.nz; if none is reachable, the Tier-E G-fid becomes Tier T (retrain at the paper's budget). Separately, taldatech/ddlp pins `torch==2.0.1` / `torchvision==0.15.2` (no RTX 5090 kernels) while our stack is torch 2.13 + CUDA 13 (rule P-route).

**Facts (2026-10-25, this host).** No MEGA client is installed (no megatools / mega-get). MEGA's public-link API answers from this host (`POST https://g.api.mega.co.nz/cs` `[{"a":"g","g":1,"p":<id>}]` → size, encrypted attributes, temporary download URL). A 150-line fetcher (`scripts/b127_ddlp_mega.py`: AES-128-CTR decryption with the link key and MEGA's chunked CBC-MAC checked against the key's meta-MAC; download rejected on mismatch) downloaded, all with **meta-MAC verified**:

| file (README link) | bytes | sha256 |
| --- | --- | --- |
| `ddlp-phyre.zip` (DDLP PHYRE 128×128, `UBcl2LgQ`) | 86 888 423 | `efa5878b6c6317896886c6e77a7a0ef83672390e67ecd194031657d51aceb06a` |
| `ddlp-clevrer.zip` (DDLP CLEVRER 128×128, `oUUjXY4B`) | 34 712 687 | `427388ea49fcb67cb0b1b39a6f6a0a52af905f98351a88fd6d0ea827825da450` |
| `phyre_dataset.zip` (the authors' generated PHYRE data, `lIkDnZZD`) | 453 909 158 | `22d346893213b5b619139ba1958ee035e00e6de4de9ce226b57d06355d0a20ec` |

`ddlp-phyre/phyre_ddlp.pth` loads into the official `ObjectDynamicsDLP` built from its `hparams.json` with all keys matched (23 386 410 parameters); its hparams equal the paper's Table 4 PHYRE column (K = 25 particles, L = 64 prior proposals, 15 training frames, MSE loss, feature dim 4). The PHYRE data holds train / valid / test = 2 574 / 312 / 400 episodes, exactly the paper's appendix G split. Stored in `data/external/ddlp/{checkpoints,data}/` with `MANIFEST.json` (url, bytes, sha256, downloaded_at, meta_mac_ok); gitignored, never committed.

**Options.** (a) Tier E on PHYRE-128 (released checkpoint + released test data, official `eval/eval_gen_metrics.py`); (b) Tier E on CLEVRER-128 (needs the CLEVRER validation videos from MIT and the authors' unspecified per-video trimming); (c) Tier T retrain.

**Decision.** (a). Tier T is not needed. PHYRE is also the closest published setting to NovPhy (2-D physics puzzles) and the configuration the NovPhy adapter starts from. The band is frozen and posted in a separate Decision P-fid before any fidelity run. The evaluation is cheap enough for the CPU (measured: `model.sample` of 10 episodes × 100 frames in 110 s at 4 threads → ≈ 75 min for the 400 test episodes plus LPIPS), so G-fid needs no GPU funding.

**Environment (P-route).** `b127-ddlp` = a clone of `novphy` (Python 3.11, torch 2.13.0+cu130) plus `torchvision 0.28.0+cu130` (installed `--no-deps`, so torch is unchanged), `piqa 1.3.1` (the repo's pin, used by the official metrics), `opencv-python-headless`, `scikit-image`, `imageio`, `accelerate` (imports only) and `pycryptodome` (the fetcher). The official code imports and runs on torch 2.13 with no edit. The G-port reference (the unedited clone) and the adapter both run in this env on the same torch build (the #129 lesson: no cross-version Adam round-off). The NovPhy runner also runs in this env (the vendored code imports torchvision / cv2, which `novphy` lacks); `novphy` itself is not modified. Package list: `.local-artifacts/issue-135-ddlp-v1/envs/b127-ddlp.txt`.

**Upstream defect handled without editing the clone.** `datasets/get_dataset.py` imports `datasets.langtable_ds`, a file that does not exist at any commit of the repo, so `eval/eval_gen_metrics.py` cannot be imported as released. The fidelity harness registers an empty stand-in module under that name before running the unchanged script (the PHYRE path never touches it); disclosed in `fidelity/gfid.json`.

Posted: https://github.com/Sino-Huang/NovPhy/issues/135#issuecomment-6396701779

## Decision P-fid (2026-10-25): frozen G-fid band for T2 — DDLP Table 3 PHYRE-128 LPIPS 0.055 → [0.0505, 0.0595]

**Trigger.** #127 requires the G-fid band to be frozen and posted before any fidelity run. Route: Tier E with the released PHYRE-128 checkpoint (Decision P-route, issuecomment-6396701779).

**Target.** Daniel & Tamar, DDLP (TMLR 2024, arXiv 2306.05957 v2), Table 3, PHYRE, DDLP: PSNR 26.98 ± 5.3, SSIM 0.95 ± 0.04, **LPIPS 0.055 ± 0.04**; §5.2: condition on 10 frames, predict the next 90; appendix G: 400 PHYRE test episodes; footnote 5: PIQA metrics.

**Statistic.** The `lpips` value of the official `eval/eval_gen_metrics.py` (taldatech/ddlp @ 5686275, unchanged, from the read-only clone): mean piqa-1.3.1 LPIPS (VGG) over every predicted frame, 400 episodes × 90 frames. Run: `-d phyre -p <run>/ddlp-phyre --checkpoint ddlp-phyre/phyre_ddlp.pth -c 10 --horizon 100 -b 2 --cpu`, deterministic (the script's default), the released `hparams.json` with only `root` pointed at the authors' released PHYRE data, torch 2.13 CPU float32, one run. Batch 2 instead of the script's default 10 to keep host RAM low (measured 18 GiB peak RSS at batch 10); this model has no batch-coupled layer in eval mode (GroupNorm only), so the batch size does not change the predictions.

**Options for the tolerance.** (a) the script's sampling variability (`--sample`); not applicable: the default evaluation is deterministic, so a re-run reproduces itself; (b) only the rounding of the published value (±0.0005): ignores that a test set of 400 episodes is a sample; (c) **the sampling variability of the test-set mean over episodes**, bounded from the paper's own spread.

**Decision: (c).** The paper's ±0.04 is the SD over all predicted frames (the script's `lpips_std`). With 90 frames per episode, the variance of episode means is at most the variance of frame values (law of total variance), so the SE of the 400-episode mean is ≤ 0.04 / √400 = 0.002. **Band = 0.055 ± (2 × 0.002 + 0.0005) = [0.0505, 0.0595]**; the 0.0005 term is half of the last reported digit. **Rule:** pass iff the official `lpips` lies in the band. PSNR and SSIM are reported against bands built the same way ([26.445, 27.515], [0.941, 0.959]), descriptive only. A fail follows P-fid: at most two logged repair attempts, each fixing a diagnosed harness defect, never the band; then T2 is dropped with `fidelity_not_reproduced`.

**Frozen file.** `.local-artifacts/issue-135-ddlp-v1/fidelity/band.json` (schema `issue_127_gfid_band_v1`), sha256 `06cbd43b680c7996c4719833d7ad70888cfc274fd43a02b100d45794a0254581`. No fidelity run has started. The run is CPU-only (≈ 2–3 h at 4 threads), so it needs no GPU funding; the harness is `scripts/b127_ddlp_fidelity.py` (`--prepare` done: inputs verified against the meta-MAC-checked downloads, 400 test episodes extracted and hashed).

Posted: https://github.com/Sino-Huang/NovPhy/issues/135#issuecomment-6396727991

## Decision P-other (2026-10-25): the T2 DDLP NovPhy adapter (frames, episodes, launch, rollout, readout, gates)

**Trigger.** #135 needs a NovPhy adapter for the vendored official DDLP (P-license issuecomment-6396690148, P-route issuecomment-6396701779) before the G-port fixes what it compares and before the frame cache is built.

**Decision (`world_model/baselines/ddlp/novphy.py`; the vendored code is unchanged).**
1. **Model = the official `ObjectDynamicsDLP`** (DLPv2 particle encoder with the tracking posterior, background module, object decoder, PINT, `calc_elbo`) at the **released PHYRE-128 checkpoint's `hparams.json`** (= paper Tables 4/5: K = 25 particles, L = 64 prior proposals, glimpse 16 px (`anchor_s` 0.125), feature dim 4, PINT 6 × 8 heads × 512, `timestep_horizon` 15, τ = 4 burn-in frames, MSE reconstruction, β_KL = β_dyn = 0.15 (Table 4 prints 0.1; the checkpoint's 0.15 is used), scale / offset prior σ = 1, Adam 2e-4, ε 1e-4, StepLR γ 0.95 per epoch, warm-up and noisy epochs); 23 386 410 parameters + 156 for the launch particle. fp32, torch default flags (the repo's `mixed_precision: 'no'`).
2. **Frames.** The #123 pixel stores (uint8 320 × 240) resized to the PHYRE configuration's **128 × 128** with `cv2.resize(…, interpolation=cv2.INTER_AREA)`: the interpolation the official `phyre_preparation.py` names (its call passes `cv2.INTER_AREA` positionally as `dst`, so it actually ran bilinear; INTER_AREA, the stated intent, keeps small pigs from aliasing at a 2.5× downscale). Full frame (4:3 → 1:1; particles carry x and y separately). No colour inversion (the official PHYRE loader inverts PHYRE's white background; that is dataset-specific).
3. **Episodes and stride.** Frames at **stride 5** (0.1 s at the engine's 50 Hz = the 10 fps of the paper's PHYRE data and inside DDLP's small-displacement tracking assumption; stride 15 would move a bird ~20–40 px per step at 128 px against a 16-px glimpse): positions 0, 5, …, 225, shots ending earlier repeat their last frame (as the official PHYRE loader pads). The scene is static before the launch, so each episode starts with the decision frame **four times** (the τ = 4 burn-in context, constant-prior KL as official), then frames 5, …, 225: **49 frames**; the launch acts at t0 = the fourth frame, so PINT's prediction of frame 5 (episode index 4 ≥ τ) is trained. Training windows are the official `timestep_horizon + 1 = 16` consecutive frames at every offset (the official `PhyreDataset` indexing): 34 per shot, **473 246** windows over the 13 919 fit shots. Training data = the #109 fit split only (`eb.FitShots`). A uint8 128 × 128 cache of the 46 stored frames per shot (~31.5 GB, local NVMe) is built on the CPU by `--data`.
4. **Launch: a launch particle.** One extra particle per frame, appended after the K foreground particles (the background token stays last, as official), whose attributes [xy 2, scale 2, transparency 1, depth 1, features 4] are a learned linear map of `eb.launch5`; the official `ParticleFeatureProjection` projects it like any particle; PINT's particle relative-position bias covers K + 2 tokens. Its predicted next state is discarded: it is re-set to the launch after every rollout step and excluded from every loss (the losses see exactly the official K-particle outputs). Options considered: (a) moving the projectile particle to a launch state for the burn-in frames (the paper's "what-if" edit) needs the launch's pixel trajectory, which only the engine knows; (b) a launch token outside the particle set needs new projection code; the launch particle uses only official modules and keeps the launch visible inside PINT's sliding 15-frame window for the whole rollout (as B1's launch token, #129).
5. **Rollout to frame 225.** Deterministic (posterior means and PINT means: the official `sample` default): the decision frame ×4 is encoded exactly as `ObjectDynamicsDLP.sample` encodes its conditioning frames, then **45 PINT steps** with the official `DynamicsDLP.sample` (sliding context of 15 frames, as official). 45 steps lie inside the paper's PHYRE horizon (condition 10, predict 90), so no chunking is needed.
6. **Readout (G-readout applies).** `ParticleReadout`: the particle set (K tokens [xy, scale, transparency, depth, features] + the background-feature token) → the 236-d E99 carrier: 19 queries (header + 18 slots) cross-attending over the embedded particle tokens (no particle-index embedding: permutation-invariant), the #130 R1 recipe (dim 256, 8 heads, MLP 512, AdamW 1e-3 / wd 1e-4, batch 128, 25 epochs, MSE on z-scored carriers, seed 20260908). Fitted per arm-seed on **fit frames only**: every fit shot at positions 0, 15, …, 225 (clamped, ≤ 16 per shot), particles of the true frame = the encoding the rollout applies to the decision frame (the frame ×4, last-frame posterior), targets = the E99 carriers of the same frames; frozen in `readouts-manifest.json` before any evaluation read.
7. **Score.** Primary = tie-free `EndpointCosts` of readout(particles at frame 225) against readout(particles of the decision frame) (the arm's input is the decision frame only). Position error (descriptive) from the readout carriers at frames 15, 30, 60, 150, 225; wrong-anchor rows from the 480 fit anchor frames.
8. **Guards.** G-rank and G-action on the readout carriers at frame 225 (480 fit anchors × 16 grid launches), with the carrier-space G-action calibration reported beside it (#127 decision issuecomment-6396584235: engine truth 0.0395, U 0.0229, C* 0.0334, B1 0.0130; the frozen 0.05 stands); **G-readout** = the readout's policy ceiling (true endpoint frames) ≥ 0.65 (mean over seeds, `eb.arm_guard_block`); G-nonfinite; G-trained; G-port and G-fid as extra guards. G-collapse: not applied (the state is the explicit particle set read through a readout, so G-readout is the representation guard). No native success head (DDLP has none) → no native-head score and no U + head control; no matched-budget arm.
9. **Runner and env.** `scripts/run_issue_135_ddlp.py` (contract modes + `--data`, `--readiness`, `--timing`, `--score-novelty`), env `b127-ddlp` (P-route above). Every GPU mode refuses without the orchestrator's `funded.json`; scoring also refuses unless G-fid passed.

Posted: https://github.com/Sino-Huang/NovPhy/issues/135#issuecomment-6396857866

## Decision P-other (2026-10-25): T2 training recipe, the three pre-declared configurations, and the P-cap cut order

**Trigger.** #127's selection rule needs ≤ 3 configurations declared before any selection run, and the native budget must be fixed before the projection that P-budget funding will use.

**Native recipe (every configuration unless stated).** The official `train_ddlp.py` step on NovPhy windows (Decision P-other adapter, issuecomment-6396857866): `model(x, x_prior=x, warmup, noisy, forward_dyn=True, …)` + `calc_elbo` with the official dynamics-loss ramp (per-step discount over `max(warmup_epoch, 1) · max(10 000, iterations per epoch)` iterations), warm-up epoch 0, noisy epochs 0–1, Adam 2e-4 (β 0.9 / 0.999, ε 1e-4), StepLR γ 0.95 at every epoch end, fp32. **Batch 12** = the repository's multi-GPU recipe (`accel_conf.yml`: 4 processes × `batch_size` 3, DDP-averaged; the paper trained on 4 RTX 2080s), run as one batch of 12 on one GPU. An epoch is one pass over the training windows without replacement (the official shuffled `DataLoader`, `drop_last`); data-epoch e's window order is a permutation drawn from `eb.batch_rng(seed, 10^9 + e)`; posterior sampling and PINT dropout draw from the torch generator seeded with the run seed (checkpointed). Resumable (SIGTERM → `resume.pt` at the next step → exit 75); a nonfinite loss is a typed terminal failure. The last checkpoint is used (the official best-LPIPS checkpoint selection needs a validation split; ours is the policy split and selects configurations only).

**Configurations (declaration order; seed 20260908, full budget each; the selected run is reused as main seed 20260908).**
1. **`native`**: the released PHYRE-128 checkpoint's hparams, **40 epochs over the NovPhy windows** (Table 4 PHYRE: 40 epochs; the program's epochs-over-our-data convention, #129 / #130): 473 246 windows → 39 437 steps per epoch → **1 577 480 steps**.
2. **`paper-schedule`**: the same model with **the paper's PHYRE step schedule**: 40 epochs of 18 232 steps (= the paper's 2 574 × 85 = 218 790 PHYRE windows / 12), i.e. the authors' sample count (8.75 M windows) and LR / warm-up / ramp timing, drawing windows from the NovPhy window stream → **729 280 steps**. (NovPhy has 2.16× the paper's windows; this asks whether the paper's training length suffices.)
3. **`repo-config`**: the repository's `configs/phyre.json` model (encoder channels 32-64-128-256, prior 16-32-64, L = 30 proposals, scale / offset prior σ 0.3 / 0.2; 26 052 582 parameters with the launch particle), native schedule.

Selection: `eb.select_configuration` on the policy-split pooled grid AUC of the primary score (`--selection-readouts`), after the readiness stop (separate decision) and after G-fid passed. Then seeds 20260909 / 20260910 at the selected configuration.

**P-cap cut order (pre-registered).** If the funded remainder cannot cover the child: first drop `repo-config` from the grid (#127's "three configurations to two"); never shorten the selected configuration or drop a seed; if it still does not fit, T2 is listed as not run (typed `cap`).

**Projection [INFERENCE], before any GPU measurement.** FLOPs per optimizer step at batch 12 (torch `FlopCounterMode`, CPU): native 1.98 TFLOP, repo-config 3.16 TFLOP, static R0 1.70 TFLOP; a 45-step rollout of 16 candidates 11.7 TFLOP. At an assumed sustained 10–40 TFLOP/s on one RTX 5090 (fp32 conv + small-batch sequential tracking; unmeasured): native 22–88 GPU-h per run, paper-schedule 10–41, repo-config 35–139; child with three configurations 87–546 GPU-h (lower end: paper-schedule selected), with two (after the cut) 52–305 GPU-h, plus readiness R0 ≈ 1.4–5.7 and readouts / scoring ≈ 3–6. The spread is too wide to fund on; a `--timing` probe (≤ 0.5 GPU-h; its own decision before enqueue) measures s/step for each configuration, R0's static step, the rollout and the readout encoding, and replaces this projection (`projection.json`).

Posted: https://github.com/Sino-Huang/NovPhy/issues/135#issuecomment-6396890056

## Decision P-other (2026-10-25): pre-registered readiness stop for T2 — the particle readout ceiling before any DDLP training

**Trigger.** As C1-dino (#130, issuecomment-6383216316), a representation that cannot express the cost (small pigs at the model's resolution) should stop the arm before the expensive dynamics training. DDLP's particle encoder is learned on our frames, so unlike DINOv2 there is no pretrained representation to probe; the released PHYRE encoder never saw Angry Birds frames, so a zero-shot probe of it would not be a fair test.

**Options.** (a) No readiness stop (the G-readout guard after training only); (b) the released PHYRE encoder zero-shot (unfair: untrained on our domain); (c) **R0**: the DDLP particle model trained alone in the official code's static mode, then the readout ceiling.

**Decision: (c).**
- **R0** = the `native` configuration trained in `train_ddlp.py`'s static branch (`epoch < start_dyn_epoch`: `forward_dyn=False`, `calc_elbo_static`; G-port case `phyre-released-static`, bit-exact): the DLPv2 particle encoder, background and decoder that DDLP itself trains, without PINT. Seed 20260908, batch 12 windows (16 frames each, the training windows), **30 000 steps** in three official phases of 10 000 steps each (warm-up, noisy, regular; 10 000 = the official minimum dynamics-warm-up length, used as R0's epoch), StepLR after each phase. ≈ 5.8 M frame reconstructions (≈ 9 passes over the 640 k stored fit frames). [INFERENCE] 1.4–5.7 GPU-h.
- **Ceiling** (fit / policy data only, `deterministic_scoring()`): `ParticleReadout` fitted on R0's particles of the fit readout frames (every fit shot at 0, 15, …, 225 clamped; E99 targets) exactly as the main readouts; then the pooled policy-grid AUC of tie-free `EndpointCosts` of readout(particles of the **true** endpoint frame, the #123 policy pixel store at min(225, end)) against readout(particles of the policy anchor frame), over the 94 policy grid mixed cells. The E99 true-carrier ceiling (#130: 0.7223) is reported beside it.
- **Rule:** ceiling **< 0.65** (G-readout's threshold) → T2 **stops** with typed reason `readiness_ceiling_below_threshold` before any DDLP training (no selection, no seeds, no scoring; listed as not run, with the readiness record). ≥ 0.65 → selection may start. One run, no re-run, no change of R0's budget after its result.
- **Order:** R0 is a GPU job and needs the funding marker like every T2 GPU job; `--selection` refuses unless `readiness/readiness.json` passed, G-port passed and G-fid passed. The per-arm G-readout guard after training is unchanged.

Posted: https://github.com/Sino-Huang/NovPhy/issues/135#issuecomment-6396890266

## Decision P-fid (2026-10-25): T2 G-fid Tier E is outside the frozen band — two pre-registered diagnostics before any repair

**Result (attempt 1, frozen band issuecomment-6396727991, `band.json` sha256 `06cbd43b…4581`).** The official `eval/eval_gen_metrics.py -d phyre -c 10 --horizon 100 -b 2 --cpu` on the released `ddlp-phyre/phyre_ddlp.pth` and the released PHYRE test split (400 episodes × 90 predicted frames), torch 2.13 CPU, exit 0, 5 261 s CPU, 0 GPU-s:

| metric | ours | paper (Table 3, DDLP) | band | paper G-SWM row |
| --- | --- | --- | --- | --- |
| **LPIPS (gating)** | **0.0745** (SD 0.053) | 0.055 (SD 0.04) | [0.0505, 0.0595] | 0.078 |
| PSNR (descriptive) | 24.55 (SD 5.13) | 26.98 (SD 5.3) | [26.445, 27.515] | 24.64 |
| SSIM (descriptive) | 0.935 (SD 0.045) | 0.95 (SD 0.04) | [0.941, 0.959] | 0.93 |

**G-fid = FAIL** (`fidelity/gfid.json` sha256 `785fa3ef…ed46`, log sha256 `e1cb633a…b4c9`). The band is not changed.

**Rule.** P-fid allows at most two logged repair attempts, each fixing a *diagnosed* defect of our harness, never the band; otherwise T2 is dropped (`fidelity_not_reproduced`). Our harness differs from the authors' in exactly two respects that could matter: (1) the software stack (P-route: torch 2.13 / torchvision 0.28 / numpy 2.4 vs the repository's pins torch 2.0.1 / torchvision 0.15.2 / numpy 1.24.3; piqa 1.3.1 on both), (2) the batch size (2 vs the script default 10). The data, checkpoint, hparams (except the data root), script and command are the authors'.

**Decision: two diagnostics (CPU, no GPU), defined before they run; nothing else is varied.**
- **D1 (stack).** A new env `b127-ddlp-orig` with the repository's `requirements.txt` pins for every package the evaluation imports (torch 2.0.1 CPU, torchvision 0.15.2, numpy 1.24.3, piqa 1.3.1, einops 0.6.1, matplotlib 3.7.1, imageio 2.28.1, scikit-image 0.19.2; opencv headless where the pinned 3.4 wheel is unavailable); the unchanged official script on the **first 50 test episodes** (the official dataset wrapped in `Subset(range(50))` by the harness; batch 2), versus the same 50 episodes in `b127-ddlp`.
- **D2 (batch).** The same 50 episodes in `b127-ddlp` at batch 10 (the script default) versus batch 2.
- **Defect criterion:** a diagnostic **differs** iff its LPIPS differs from the `b127-ddlp` batch-2 value by more than 1e-4 relative. A differing diagnostic is a diagnosed harness defect → repair attempt 1 = the full frozen protocol (400 episodes) without that difference, judged against the unchanged band (logged as attempt 2 in `gfid.json`). If neither differs, no harness defect is diagnosed: G-fid fails and **T2 is dropped with typed reason `fidelity_not_reproduced`** before any GPU work (listed as not run; the G-port pass, the G-fid record and these diagnostics are published).
- **Not done** (outcome-driven protocol changes): a shorter horizon (the paper's appendix figures show 40 predicted PHYRE frames, but §5.2 states 90 for the metrics, and the band froze 90), `--sample`, another checkpoint, a different test split. **Disclosed context [INFERENCE]:** the released `hparams.json` has β_KL 0.15 (`run_prefix` "_bkl_0.15_std_1.0") where Table 4 prints 0.1 for PHYRE, so the released checkpoint may not be the run behind Table 3; our numbers sit at the paper's G-SWM row. A Tier-T retrain at the paper's budget (several days on 4 GPUs) is outside P-fid's repair rule and the remaining budget.

Posted: https://github.com/Sino-Huang/NovPhy/issues/135#issuecomment-6397579863

## Decision P-fid (2026-10-25): T2 DDLP G-fid repair attempt 1 (diagnosis D1 met the pre-registered defect criterion)

**Trigger.** Tier-E G-fid attempt 1 failed: LPIPS **0.0745** on PHYRE-128 (official `eval_gen_metrics.py`, released checkpoint, 400 test episodes) against the frozen band [0.0505, 0.0595] (paper 0.055). The diagnosis plan posted before the diagnostics ran (issuecomment-6397579863) fixed the rule: a diagnostic whose result differs from ours by more than 1e-4 relative on the first 50 test episodes is a diagnosed harness defect and triggers one repair attempt without that difference; otherwise T2 is dropped.

**Diagnostics** (`fidelity/diagnostics/`; first 50 test episodes, CPU):
| diagnostic | LPIPS | vs ours (torch 2.13, batch 2) 0.048373 |
| --- | --- | --- |
| D1: the repo's pinned stack (torch 2.0.1, env `b127-ddlp-orig`), batch 2 | 0.048361 | 2.6e-4 relative — over 1e-4: **diagnosed** |
| D2: our stack, batch 10 | 0.0483747 | 2.6e-5 relative — under 1e-4: no defect |

**Decision.** Repair attempt 1: rerun the full frozen Tier-E protocol unchanged (same checkpoint, data, script, metric and band) in the repo's pinned stack (`b127-ddlp-orig`, torch 2.0.1), CPU only (≈ 90 min, no GPU). It is logged as attempt 2 in `fidelity/gfid.json` and judged against the unchanged band. If it lands outside the band, T2 is dropped with typed reason `fidelity_not_reproduced` (P-fid): the D1 difference is float round-off (model outputs agree to ~1e-6), about 1000× smaller than the gap to the band, so no second repair attempt is diagnosed. A Tier-T retrain is not a port-defect repair and is not attempted (the released hyperparameters show βKL 0.15 where the paper's Table 4 prints 0.1, which suggests the released checkpoint may not be the Table 3 run [INFERENCE]; disclosed).

## Decision P-fid (2026-10-26): T2 DDLP dropped — `fidelity_not_reproduced`

**Repair attempt 1** (issuecomment-6397847883): the frozen Tier-E protocol rerun unchanged in the repo's pinned stack (`b127-ddlp-orig`: torch 2.0.1+cpu, torchvision 0.15.2, numpy 1.24.3, piqa 1.3.1), CPU, 4 824 s, exit 0. Official `eval/eval_gen_metrics.py`, released PHYRE-128 checkpoint, 400 test episodes × 90 predicted frames:

| metric | attempt 1 (torch 2.13) | attempt 2 (pinned torch 2.0.1) | paper (DDLP, Table 3) | frozen band |
| --- | --- | --- | --- | --- |
| LPIPS (gating) | 0.0745 | **0.0745** (0.074488) | 0.055 | [0.0505, 0.0595] |
| PSNR (descriptive) | 24.55 | 24.55 | 26.98 | [26.445, 27.515] |
| SSIM (descriptive) | 0.935 | 0.935 | 0.95 | [0.941, 0.959] |

**Decision.** Outside the band again. The only diagnosed difference (D1: float round-off between torch versions, 2.6e-4 relative on 50 episodes) is removed and changes nothing at the reported precision; no further port defect is diagnosed, so P-fid allows no second repair. **T2 DDLP is dropped with typed reason `fidelity_not_reproduced`.** It stays in the program summary as a triggered Tier-C arm that was not run (no NovPhy training, no evaluation read), with this fidelity table. Disclosed: our released-checkpoint numbers sit at the paper's G-SWM row (LPIPS 0.078, PSNR 24.64, SSIM 0.93), and the released hyperparameters show βKL 0.15 where the paper's Table 4 prints 0.1, so the released checkpoint may not be the run behind Table 3 [INFERENCE]; a Tier-T retrain was not attempted (not a port-defect repair). No GPU time was spent on T2.
