# #131 D1 THICK — decisions log

Every decision below is also posted on #131 (URL after each).

## Decision P-route (2026-10-24): THICK route (b), our PyTorch port under G-port

**Trigger.** #127 D5 / rule P-route: switch to route (a), the JAX port [CognitiveModeling/thix](https://github.com/CognitiveModeling/thix), only if that repository carries a license file when Phase 0 starts. Phase 0 started 2026-10-24 19:45 AEDT.

**Check (automatic, GitHub API, 2026-10-24 ~19:50 AEDT).**
- `CognitiveModeling/thix` @ `a5054b448703d93afcd9839114f3e00ed5bac74e` (last push 2025-10-03): repository license `null`; root entries `README.md, docs, dreamerv3, embodied, tune_sparsity.md, utils` — **no LICENSE file**.
- `CognitiveModeling/THICK` @ `934980a4a25d2b6baa357fe234b6e841cd3cd101` (last push 2026-04-28): `LICENSE` present, SPDX **MIT**; TensorFlow 2 on the DreamerV2 code base.

**Decision.** Route **(b)**: our PyTorch port of THICK's context kernel (GateL0RD) and RSSM, held to **G-port** against the official TF2 code (vendored at `934980a`, MIT) on fixed CPU inputs. thix is not vendored and not used as a reference implementation (no license file, rule P-license). Nobody waits for an author reply.

**Consequences.** The readiness gate stays as written: G-port must pass, then the G-fid run (THICK's advantage over DreamerV2 on VisualPinPad) with its band read from the paper's figures and frozen here before the run. Where the official TF2 stack can run on the RTX 5090s, G-fid may use the official code; otherwise it uses the G-port-verified port (rule P-route). That choice is posted as its own decision before the fidelity run.



Posted: https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382162858 (orchestrator)

---

## Decision P-license (2026-10-24): vendor THICK @ 934980a (MIT), verbatim, as the G-port / G-fid reference

**Trigger.** Rule P-license / CONVENTIONS "Vendoring": the official code must be vendored at a pinned commit with its upstream LICENSE before it is used as the G-port reference (route (b), Decision P-route above).

**License check.** [CognitiveModeling/THICK](https://github.com/CognitiveModeling/THICK) @ `934980a4a25d2b6baa357fe234b6e841cd3cd101`: `LICENSE` = MIT, "Copyright (c) 2024 cgumbsch" (SPDX `MIT`). MIT permits copying and modification with the notice kept; the upstream `LICENSE` is vendored next to the code.

**What is vendored** (to `world_model/baselines/thick/vendor/`, layout `thick/...` as upstream, plus `LICENSE`; per-file upstream and vendored sha256 in `VENDOR.json`): the import closure of the official training entry points for VisualPinPad and the THICK world model —
`thick/{agent,expl,cem_agent,hl_mcts,train,train_from_replay,train_only_on_replay}.py`, `thick/configs.yaml`, `thick/common/{__init__,config,counter,dists,driver,env_register,envs,flags,logger,nets,other,pinpad_env,replay,tfutils,thick_utils,when}.py`.

**Not vendored:** `thick/multiworld/` (MuJoCo Sawyer suite and assets), `common/{minihack_utils,multiworld_utils,minigrid_utils,plot}.py`, `des_envs/`, `media/` — not needed for the context kernel, RSSM, high level, losses, optimizer or VisualPinPad (`env_register` imports them only inside the MiniHack / MiniGrid / Multiworld branches).

**Edits: none.** The files are byte-identical copies. Upstream uses top-level imports (`import common`, `import agent`); the reference script runs the vendored code exactly as upstream does, with `vendor/thick` on `sys.path`, in the CPU env `b127-thick-tf` (Python 3.8, tensorflow-gpu 2.4.2, tensorflow-probability 0.11.0, numpy 1.19.5 as pinned upstream). The vendored code is never imported by our PyTorch port at run time.

**Recorded upstream defects (not edited now):**
1. `common/nets.py` `EnsembleRSSM.__init__` reads an undefined name `rnn_type` (`self._rnn_type = rnn_type`), so every non-THICK world model (`c_rssm: False`: the DreamerV2 baseline and Plan2Explore) raises `NameError` at construction at this commit. It does not affect THICK (`ContextRSSM`). If the G-fid route ever runs the official DreamerV2 / Plan2Explore, deleting that one line will be a listed `VENDOR.json` edit with this reason, posted before the run.
2. The upstream `requirements.txt` is not jointly installable (`nevergrad==0.13.0` needs numpy>=1.24 vs `numpy==1.19.5`; `GitPython>=3.1.47` needs typing-extensions>=3.10 vs tensorflow-gpu 2.4.2's `~=3.7.4`; `sklearn==0.0.post9` is the deprecated stub). `b127-thick-tf` installs every pin except the MiniHack / Atari / DMC / MuJoCo / optimisation-only packages (`nevergrad, bayesian-optimization, cma, minihack, nle, atari-py, ale-py, dm-control, dm-env, labmaze, mujoco, robodesk, pyuv, sklearn, scikit-learn, seaborn, glfw, PyOpenGL, lxml, joblib, threadpoolctl`; none is imported on the VisualPinPad / THICK path) and pins `GitPython==3.1.31` and `wandb==0.15.12` (imported by `train.py`, unpinned upstream). Exact package list: `envs/b127-thick-tf.txt`.

**Use.** Reference for G-port (CPU, float32, eager mode = upstream `jit: False`), the official CPU smoke, and a possible official-code G-fid route. MIT allows committing the vendored copy.

Posted: https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382307800

---

## Decision P-fid (2026-10-24): frozen G-fid band for THICK on VisualPinPad (before any fidelity run)

**Trigger.** #131 readiness gate 2 / rule P-fid: the G-fid band "is a GAP: read the final-step mean ± std from the paper's figures and freeze it before the run". No fidelity run has started (nothing has been run on a GPU for D1).

**Source.** THICK, ICLR 2024 proceedings PDF (`data/external/thick/paper/thick_iclr2024.pdf`, `sha256:` in `data/external/thick/MANIFEST.json`; the OpenReview PDF endpoint served a bot check; arXiv 2310.05167 is *Hieros*, not THICK, and no arXiv version of THICK exists). **Fig. 8** (p. 8): VisualPinPad mean evaluation returns over env steps, **7 seeds, ± standard error**, THICK Dreamer vs Dreamer (DreamerV2) vs Director, replay buffers pre-filled with **1M Plan2Explore steps** (Suppl. F.4). **Table 3** (Suppl. F.5): env steps to 95 % of the max mean reward (threshold 359 PinPadFour, 274 PinPadFive).

**How the numbers were read.** Fig. 8 is vector graphics. `scripts/b127_thick_fidelity.py --band` reads the PDF paths (PyMuPDF): per method the stroked polyline is the seed mean (one point per 1e4 env steps), the filled polygon of the same colour is mean ± 1 SE; x is calibrated on the tick labels, y on the grid lines; SD = SE·√7. Check: the digitised seed-mean curves cross the Table-3 thresholds at exactly the Table-3 steps (PinPadFour 140k vs 200k, PinPadFive 340k vs 590k). The PDF keeps curve points beyond each plotted x-window (clipped); the band uses the last point inside the window.

| task (final window step) | THICK Dreamer mean ± SD (SE) | band [mean ± 2 SD] | DreamerV2 mean ± SD (SE) | band | steps to threshold THICK / Dreamer |
| --- | --- | --- | --- | --- | --- |
| PinPadThree (390k) | 575.1 ± 9.5 (3.6) | [556.0, 594.1] | 575.8 ± 6.0 (2.3) | [563.8, 587.7] | — (not in Table 3) |
| PinPadFour (400k) | 374.3 ± 7.3 (2.8) | [359.7, 388.8] | 372.8 ± 5.8 (2.2) | [361.3, 384.4] | 140k / 200k (thr. 359) |
| PinPadFive (600k) | 288.6 ± 3.5 (1.3) | [281.6, 295.6] | 276.4 ± 13.9 (5.2) | [248.6, 304.2] | 340k / 590k (thr. 274) |

The paper's advantage of THICK over DreamerV2 on VisualPinPad is **sample efficiency** (Table 3; "matches Dreamer in PinPadThree and is slightly more sample efficient in the more challenging tasks"); final returns are within one SD of each other on Three and Four.

**Frozen pass rule (per G-fid task).** (1) *Returns:* for THICK Dreamer and for DreamerV2 separately, our seed-mean evaluation return at the task's final window step lies in the band above. (2) *Advantage:* THICK Dreamer's seed-mean evaluation-return curve (one evaluation episode per 1e4 env steps, upstream `eval_every` / `eval_eps`) reaches the task's Table-3 threshold strictly earlier than DreamerV2's (DreamerV2 never reaching it counts as later). G-fid passes iff (1) for both methods and (2). At most two logged repair attempts, each fixing a diagnosed port defect, never the band; then D1 is dropped with `fidelity_not_reproduced` (rule P-fid).

**Scope.** All three task bands are frozen now; which task(s), how many seeds per method, the exploration data and the execution route are fixed in the G-fid plan decision (P-route / P-cap), also before any run. Fewer than the paper's 7 seeds will be disclosed.

`fidelity/band.json` sha256 `c48251b3c3e5295451169b58479b6ed45bde7bce813051c37d121b3024c4c2dd` (schema `issue_127_gfid_band_v1`, with the digitised curves).

Posted: https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382475412

---

## Decision P-other (2026-10-24): D1 THICK NovPhy adapter (declared before any training)

**Trigger.** #131 / #127 D1 arm: THICK on E99 carriers with the launch at t0 into the low level, low level rolled at stride 5 to frame 225, tie-free cost on the decoded endpoint carrier. These adapter choices are fixed before any training (nothing has been trained). Code: `world_model/baselines/thick/novphy.py` around the G-port-verified port `world_model/baselines/thick/port.py`.

**1. Model (native THICK).** The vendored `thick_pinpad` configuration (the paper's THICK sizes): C-RSSM with deter / hidden 256, 32 × 32 discrete latents, a 16-d GateL0RD context (gate noise σ 0.1, `ctxt_always_sample`), THICKNet 3 × 1024 with 5 high-level actions and 2 × 1024 readouts; KL balance 0.8, free 0; loss scales `ctxt_sparsity` 1 (upstream also adds `gate_activity`, the same gate mean, at the default scale 1), `thick_act` 0.1, `thick_time` 0.1 (the paper's rule for task-free offline data, Suppl. B, and the code's `thick_pinpad`); two Adam optimizers (lr 1e-4, eps 1e-5, clip 100, wd 1e-6, Keras semantics); float32 (upstream `precision: 32`), TF32 off. **Observation pathway:** THICK's own MLP pathway (`mlp_keys: carrier`, `cnn_keys: $^`) replaces the image CNN: encoder 4 × 400 ELU, fine and context decoders 4 × 400 ELU with `mse` heads on the 236-d carrier. 21 852 337 trainable parameters (low level with heads 7 064 730; high level 14 787 607).

**2. Episodes.** One fit shot = 46 observations: E99 carriers at frames 0, 5, …, 225. Positions after a shot's end repeat its final carrier (601 of 13 919 fit shots end before frame 225; the scene is at rest). Fit split only (`eb.FitShots`, 13 919 complete shots).

**3. Launch.** Upstream transition layout (`common/driver.py`): `action[t]` is the action that led into observation t, index 0 the reset's zero action. The launch (`eb.launch5`, 5-d) is applied at t0, so it sits at **index 1**; every later action is the zero no-op. The action space is continuous, so THICK's high-level action head is its truncated normal on [−1, 1] with the NLL loss (the paper's continuous-action setting); every fit launch lies in [−1, 1] (guarded).

**4. No task signal.** Reward 0 at every step (the reward heads stay, as upstream's architecture), no termination, `pred_discount` False, segmentation on the context gates only (`thick_segment_crit: [gates]`, as VisualPinPad / Multiworld). No actor, critic or planner is trained: NovPhy ranks with the cost on the decoded endpoint.

**5. Training.** Offline on the fit split; batches of 16 whole episodes (native 16 × 50) drawn uniformly with replacement with `eb.batch_rng(seed, b)`; each update is upstream `Agent.train`'s world-model part: the low-level step, then the high-level step on the updated low level. Native budget: 200 000 updates (upstream `steps: 1e6` / `train_every: 5`), capped at 24 GPU-h per seed; the measured throughput goes into the frozen plan, and a projection over the cap triggers P-cap. Secondary (matched): 9 000 updates × batch 64 with the width scaled to 1.84 M ± 5 % parameters (fixed in the plan freeze).

**6. Selection (≤ 3 pre-declared configurations).** `ctxt_sparsity` (β_sparse) ∈ {1, 10, 25}: the paper's per-suite values (VisualPinPad / MiniHack-Corridor 1, MiniHack 10, Multiworld 25) within its tuning guide {1, 5, 10, 25, 50}. Seed 20260908; policy-split pooled grid AUC; ties → fewer parameters (equal here) → declaration order; then 3 seeds.

**7. Readout (scoring).** Observe the anchor carrier at t0 (`obs_step`, `is_first`), then 45 low-level `img_step`s with the launch at the first and no-ops after (frames 5 … 225). Deterministic: latent modes and zero GateL0RD noise (upstream `eval_ctxt_mean: True`; upstream's `ctxt_always_sample: True` would keep the noise on at evaluation — disclosed). **Primary score:** the tie-free cost (`EndpointCosts` via `eb.endpoint_rows`) of the fine decoder's mean carrier at frame 225. **Commit points (adaptive Δ):** the rollout steps whose context gates open (any of the 16 GateL0RD gates > 0), recorded per candidate with their count. **Exploratory only:** the context decoder's endpoint carrier and the high level's jump prediction from the post-launch state (predicted elapsed steps and decoded carrier).

**8. Guards.** As #127: G-rank, G-action (endpoint carriers), G-nonfinite, wrong-anchor control; G-collapse on the decoder input at the endpoint (`get_feat`: 1024 one-hot stochastic + 256 deterministic + 16 context).

**Not changed from upstream:** every layer, initialiser, distribution, loss and optimizer (G-port: `gport/gport.json`, pass on the carrier configuration and on the official VisualPinPad configuration).

Posted: https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382505350

---


## Decision P-route (2026-10-24): G-fid on our PyTorch port (route ii), not the official TF2 stack

**Trigger.** Rule P-route / #131 readiness gate 2: the G-fid run (THICK Dreamer vs DreamerV2 on VisualPinPad, band frozen in the Decision P-fid comment) may use the official code only if the official TF2 stack runs on the RTX 5090s.

**Evidence (scheduler jobs `131-tf-gpu-probe`, `131-gpu-probe-2`, GPU 3, 2026-10-24 20:25–20:55 AEDT; `fidelity/tf242-probe-summary.json`, `fidelity/tf-gpu-probe-tf2x*.json`).**
- Official pins (tensorflow-gpu 2.4.2 + CUDA 11.0 / cuDNN 8.0.5, env `b127-thick-tf`): the 5090 (compute capability 12.0) is registered after ~20 s; the first matmul takes ~2.4 min (CUDA 11.0 has no sm_120 code: PTX JIT) and logs "TensorFloat-32 will be used"; the first cuDNN 8.0.5 convolution had not returned after 5.3 min when the 8-min limit ended the probe. The official THICK step (CNN encoder/decoder) was never reached. Round 1 (5 min) produced no output at all.
- Current TensorFlow (2.21.0 `[and-cuda]`, CUDA 12.5 build, SASS sm_60–sm_89 + PTX compute_90, env `b127-thick-tf2x`): "Cannot dlopen some GPU libraries … Skipping registering GPU devices" — no GPU. It would also need code edits (Keras 3, the removed `mixed_precision.experimental` API), i.e. not the official code.
- Paper Suppl. G: the official code took 15–20 h per DreamerV2 run and 35–45 h per THICK Dreamer run on one GPU; 3 + 3 seeds + exploration data would be ≥ 160 GPU-h even at that speed.

**Decision.** Route **(ii)**: extend the G-port-verified PyTorch port (`gport/gport.json`: pass on the carrier and the official VisualPinPad configurations; worst update relative L2 1.6e-5) to the full THICK Dreamer agent, DreamerV2 and Plan2Explore. Every added module passes G-port against the vendored TF2 code on CPU before any G-fid run: EnsembleRSSM (the TF reference needs one listed vendored edit — deleting the line `self._rnn_type = rnn_type`, an undefined name that makes every non-THICK model crash at construction at 934980a), ActorCritic (actor, critic, slow target, StreamNorm, λ-return), `WorldModel.imagine`, THICK Dreamer's `thick_imagine` / `long_term_return` / coarse critic, Plan2Explore's disagreement ensemble, acting, replay, driver and the PinPad environment with its wrappers (CPU equality tests), and the `train.py` / `train_from_replay.py` loops (resumable: SIGTERM → checkpoint → exit 75).

**G-fid design (frozen with this decision).** VisualPinPadFour (the paper's tuning task; band threshold 359; Table 3: 140k vs 200k). One Plan2Explore run of 1M env steps (`train.py`, configs `visualpinpad plan2explore`) whose replay pre-fills every run (`train_from_replay.py`); the paper trained two exploration models and kept the higher-return one — disclosed. THICK Dreamer (`thick_pinpad`) and DreamerV2 (`visualpinpad`) × seeds 20260908, 20260909, 20260910, each to 400k env steps (the Fig. 8 window); one 2000-step evaluation episode every 1e4 env steps (upstream `eval_every` / `eval_eps`). Precision: upstream `precision: 32` (float32 parameters and activations) with **TF32 matmul/convolution on the GPU**: the official TF 2.4.2 stack itself selects TF32 on this hardware (the probe log on the RTX 5090: "TensorFloat-32 will be used"), so TF32 is the official runtime behaviour on our GPUs; G-port stays float32 on CPU. This supersedes the "TF32 off" line of the adapter decision (issuecomment-6382505350) for every D1 GPU run (G-fid and NovPhy), before any training. Pass rule: the frozen band (returns of both methods at 400k inside their bands; THICK reaches 359 strictly earlier than DreamerV2). 3 seeds per method instead of the paper's 7 — disclosed.

**Projection** (`fidelity/plan.json`, measured port seconds per update on a 5090 plus a behaviour-update proxy; [INFERENCE] until the first run's throughput replaces it): THICK Dreamer 9.2 GPU-h per run, DreamerV2 6.8, Plan2Explore 19.4 → 67.2 GPU-h in strict float32, **51.4 GPU-h with TF32** (the precision above). The first 1 000 updates of each kind are timed and the projection re-checked before the rest runs.


## Decision P-cap (2026-10-24): D1 projection within the child share; fidelity above the #127 sub-estimate

**Trigger.** Rule P-cap: the G-fid projection (67.2 GPU-h) exceeds #127's D1 fidelity estimate (24–48 GPU-h).

**Child projection** (`fidelity/plan.json`): fidelity 67.2 + selection 3 configurations × 14.0 (native 200 000 updates at the measured 0.16–0.22 s per update on the carrier model, +15 % overhead) = 42.0 + training of two more seeds 28.0 + matched secondary ~2 + scoring / validate ~2 = **~141 GPU-h**, inside the child share (~86–146) and ~35 h wall on 4 GPUs (≤ 48 h).

**Decision.** No cut: the child total stays within its share, so P-cap's cut order does not apply. If a measured smoke pushes the child total over 146 GPU-h, P-cap applies in its order (unallocated program budget first; then matched secondary runs; then the selection grid 3 → 2; never the primary arm's native budget or its three seeds). Reducing the G-fid design (fewer seeds, a shorter window) is not an option under P-cap.

**Precision note.** These numbers are strict float32. Under the TF32 precision of the Decision P-route posted with this one, the G-fid projection is 51.4 GPU-h and the NovPhy runs are re-measured by the #131 smoke; the child total can only fall.

## Decision P-other (2026-10-24): D1 runner — guards (G-collapse latent, G-action), matched D1-m, exploratory outputs, readiness and scoring gates

**Trigger.** #131 runner `scripts/run_issue_131_thick.py` (child-runner contract; template #129). These choices govern the D1 selection runs (not started; nothing trained on NovPhy) and everything after. The model, episodes, launch, training step, selection grid and readout are the adapter decision's (Decision P-other "D1 THICK NovPhy adapter" above); nothing there changes.

**1. G-collapse (latent model, #123 definition).** Latent = the low level's decoder input `rssm.get_feat` (one-hot stochastic mode 32 × 32 = 1024 + deterministic 256 + GateL0RD context 16 = **1296-d**) of the **posterior** after observing each **fit** episode deterministically (latent modes, zero GateL0RD noise), at frames 0, 15, …, 225 (the #123 probe stride: 16 frames × 13 919 fit shots). Effective rank ≥ 0.1 · 1296 and χ (within-episode share, episodes = shots) ≥ 0.05 on every seed (`eb.GUARD`). This refines item 8 of the adapter decision ("decoder input at the endpoint") to #123's fit-frame definition: G-collapse asks whether the representation of observed states is collapsed, which predicted endpoints cannot show. The endpoint latent's statistics are not a guard.

**2. G-action.** `eb.action_sensitivity` (#123) on the **decoded frame-225 carriers** (the scored quantity), 480 fit anchors × 16 grid launches, ≥ 0.05 on every seed (as #129; adapter item 8). G-rank (480 × 16 tie-free costs), G-nonfinite, G-trained per #127; **G-port** (world-model and agent records) and **G-fid** are extra arm guards.

**3. Matched secondary D1-m.** U's B1 recipe (9 000 updates × 64 episodes) and every hidden width × **0.166**: RSSM hidden / deter 42, encoder and both decoders 4 × 66, reward heads 66, THICKNet width (body, high-level action prior / posterior) 170, THICK readouts 2 × 170; discrete latents 32 × 32, the 16-d context and all depths native → **1 835 187** trainable parameters (U 1 837 690, −0.14 %; the factor is the one in [0.150, 0.180] at step 0.001 closest to U; `ctxt_rnn_out_size` is unused by THICK's headless GateL0RD). Every other hyperparameter = the selected configuration's.

**4. No native success head, no U + head.** NovPhy gives THICK no reward signal (reward heads trained on zeros, adapter item 4), so there is no native success-head score and the #127 U + head control does not apply to D1. Stated in the plan and summary.

**5. Exploratory outputs (never in a claim; stored per candidate).** (a) commit points: rollout steps whose GateL0RD gates open (THICK's learned Δ), per candidate + summaries; (b) `coarse_tie_free`: tie-free cost of the context decoder's endpoint carrier; (c) `hl_tie_free` / `hl_time`: tie-free cost of the high level's decoded jump prediction from the post-launch state and its predicted elapsed steps. Views D1-coarse and D1-hl appear in the ranker tables and as labelled within-family contrasts (D1-coarse − D1, D1-hl − D1).

**6. Contrasts.** Per arm X − U / C\* / P / J_dec (both directions); within-family **D1 − J_dec** (guarded by D1's and the global guards) and D1 − D1-m; wrong-anchor control for D1 and D1-m; position error at t ∈ {15, 30, 60, 150, 225} (rollout carriers, descriptive).

**7. Readiness and scoring gates.** `--selection` refuses (exit 1, nothing trained) unless `gport/gport.json` and `gport/agent/gport.json` pass and `fidelity/gfid-timing.json` reports `feasible: true`; the selection jobs also `depends_on` ThickD1's `131-gfid-timing` (exits 0 only when the measured G-fid projection is feasible). `--score` refuses (exit 1, evaluation not read) unless `fidelity/gfid.json` passes; a final G-fid failure drops D1 with `fidelity_not_reproduced` (P-fid), which vetoes the program claim. G_fid is also an arm guard, so a failed G-fid could only ever yield stop tokens.

**8. Runtime.** Float32, TF32 off on NovPhy runs (adapter item 1). Batch b = `tn.sample_shots(eb.batch_rng(seed, b))`; all latent / GateL0RD draws from one torch generator seeded with the run seed; resume.pt (model, both TF-Adam states, generator state, step) every 2 000 updates and on SIGTERM → exit 75 (CPU test: bitwise identical to an uninterrupted run); a nonfinite loss or gradient norm is a typed terminal failure (`nonfinite_loss`), never retried. Selection: one configuration per scheduler job (`131-select-c1/c2/c3` = sparsity 1 / 10 / 25), then `131-selro` (policy AUCs, `selection.json`), then `131-smoke` (projection vs the 146 GPU-h child cap, 24 GPU-h per seed and 48 h on 4 GPUs; P-cap if over). Training, readouts and scoring jobs are enqueued only after the plan freeze.
## Decision P-other (2026-10-24): G-fid runner implementation (route ii), declared before the G-fid runs

**Trigger.** Decision P-route (route ii) above: the G-fid runs use our port extended to the full THICK Dreamer agent, DreamerV2 and Plan2Explore. This records how the port and the runner reproduce the official loops, before the exploration run starts (only the timing-only job `131-gfid-timing` has been launched).

**G-port of the added modules** (`gport/agent/gport.json`, pass; CPU, float32; each arm starts from the official state after one warm-up `Agent.train`, then two recorded `Agent.train` calls are compared, with every optimizer: world model, high level, actor, critic, exploration ensemble, exploration actor and critic):

| arm (config) | parameters | worst gradient rel. L2 (per optimizer) | worst loss / metric rel. | worst update rel. L2 | acting (`Agent.policy`) |
| --- | --- | --- | --- | --- | --- |
| THICK Dreamer (`thick_pinpad`) | 45 426 467 | model 2.7e-6, high level 1.4e-5, actor 3.3e-6, critic + coarse critic 7.4e-6 | 4.3e-6 | 1.1e-5 | actions equal (train, eval) |
| DreamerV2 (`visualpinpad`) | 17 210 347 | model 3.2e-6, actor 1.6e-6, critic 1.9e-6 | 1.7e-6 | 9.0e-6 | actions equal |
| Plan2Explore (`visualpinpad plan2explore`) | 34 256 194 | model 3.3e-6, ensemble 1.0e-6, actors 1.6e-6, critics 1.7e-6 | 1.4e-6 | 8.8e-6 | actions equal (train, eval, explore) |

Injected defects (λ 0.9, ψ 0.1, unbiased disagreement std) each fail it (`gport/agent/sensitivity.md`). The world-model G-port (`gport/gport.json`) was re-run after the vendored edit and still passes. **Vendored edit:** `thick/common/nets.py`, the line `self._rnn_type = rnn_type` deleted (listed in `VENDOR.json`); no other vendored file differs from upstream.

**Environment.** VisualPinPad is the THICK repo's own `common/pinpad_env.py` (MIT, vendored), executed verbatim; `OneHotAction`, `TimeLimit`, `Driver`, `Replay`, `RandomAgent` are line-by-line ports (`world_model/baselines/thick/envs.py`, CPU tests).

**Loops** (`world_model/baselines/thick/fidelity_runner.py`):
- **Plan2Explore** follows `train.py`: 10k random prefill steps, the first update plus `pretrain` updates, then cycles of one evaluation episode and 1e4 train steps, with an update every 5 env steps. It explores for the whole run (`expl_until` 1e6) and runs to 1M env steps.
- **THICK Dreamer and DreamerV2** follow `train_from_replay.py`: the replay is pre-filled with the exploration run's episodes, env steps count from 0, there is no prefill, and each run ends after the evaluation at 400k env steps (the band window; upstream's loop would continue to `steps` 1e6).

**Implementation differences from upstream (no change to the learning algorithm):**
- Every RNG is seeded and checkpointed, so runs are reproducible and resumable. Upstream uses unseeded `RandomState` objects and global numpy for the replay length jitter.
- Batches are drawn synchronously; upstream `tf.data` prefetches up to 5 batches.
- Not run, because they are logging only: video and `report` passes, the random evaluation episodes that fill upstream's evaluation replay, and W&B.
- The exploration episodes are memory-mapped and shared across the six runs; upstream decompresses them per process.
- GPU precision is **TF32** (as posted in P-route; the official TF 2.4 stack selects TF32 on this GPU). G-port stays float32 on CPU.

**Resumability and ledger.** SIGTERM finishes the current env step and its update, writes a checkpoint and exits 75. The checkpoint holds the agent, optimizers, counters, RNGs, replay order, drivers with partial episodes, environments and loop phase. A periodic checkpoint is also written every 1e4 env steps. A CPU test checks that a run interrupted after its 5th update and then resumed ends with bitwise-identical weights and evaluation returns to an uninterrupted run. GPU/wall seconds are recorded in `ledger/gfid-<run>.json`, and `report.json` is written on completion.

**Feasibility gate.** `131-gfid-timing` times the first 1 000 updates of each kind and re-projects the G-fid GPU-h (`fidelity/gfid-timing.json`). It exits non-zero, blocking the exploration and G-fid runs and D1 selection, if the projection exceeds 72 GPU-h: the child share of 146 minus the rest of the projected child (selection 42, training 28, matched 2, scoring 2). In that case P-cap applies before anything else runs.

Posted: https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6383312669

---


## Decision P-other (2026-10-24): correction to the D1 runner decision, item 8 — training precision is float32 with TF32 on (per Decision P-route)

**Trigger.** Item 8 of the D1 runner decision (issuecomment-6383193006) said "Float32, TF32 off on NovPhy runs (adapter item 1)". The Decision P-route "G-fid on our PyTorch port (route ii)" (issuecomment-6382756572), posted earlier, already superseded the adapter's "TF32 off" for **every D1 GPU run**: float32 parameters / activations with TF32 matmul / convolution on the GPU, the precision the official TF 2.4.2 stack itself selects on the RTX 5090. No D1 NovPhy run has started (the selection jobs are still blocked on `131-gfid-timing`), so this correction precedes every run it governs.

**Corrected item 8 (precision part).**
- **Training** (selection runs, main seeds, D1-m, the smoke's throughput measurement): float32 with `torch.backends.cuda.matmul.allow_tf32 = True` and `torch.backends.cudnn.allow_tf32 = True`, set by `train_run` for the run and restored after it; the flags actually in force are recorded in every `model.pt` / `report.json` (`precision`) and in `plan.json` (`training_precision`), next to the scoring policy.
- **Readouts, selection readouts, scoring and validation** run under `harness.deterministic_scoring()` (TF32 off, cuDNN deterministic), as every #127 scoring path. G-port stays float32 on CPU.
- CPU tests: the flags are set during every training step and restored afterwards, deterministic_scoring turns them off, and the bitwise resume check (CPU, where the TF32 flags are inert) is unchanged.

Everything else in the runner decision stands. Runner `scripts/run_issue_131_thick.py` sha256 `b7a5940e0e3ecedf08a8bf0c41b15c442bdb8ba44be26c11fc468dab4b771293`.
## Decision P-crash (2026-10-25): G-fid timing run 2 contaminated by host-RAM exhaustion; clean rerun

**Trigger.** Rule P-crash (infrastructure). `131-gfid-timing` run 2 (launched 2026-10-24 21:55 AEDT, GPU 3, `shares_gpu`) times the first 1 000 updates of each G-fid kind on the port.
- **Plan2Explore** was measured cleanly at about 21:57: 0.101 s per update and 0.0217 wall-s per env step (`fidelity/timing-run2-contaminated/plan2explore/timing.json`).
- From about 22:00 the host ran out of RAM: MemAvailable under 1 GB of 125 GB, swap full, load 80–150. The largest resident processes belonged to another child's job on the same GPU.
- THICK Dreamer's 1 000 updates then took 19.4 s each (5.6 h). That is an artefact of the host thrashing, not a property of the port. The DreamerV2 timing never finished.
- I stopped the run at 03:35 AEDT (exit 1). Nothing was trained beyond the timing-only runs, and no G-fid or NovPhy run has started.

**Decision.** The run-2 numbers are not used for the feasibility gate. They are kept for the record in `fidelity/timing-run2-contaminated/`. The orchestrator sets `shares_gpu: false` on every `131-gfid-*` job, and `131-gfid-timing` is requeued once with no other change. A timing run now always starts from an empty directory (`b127_thick_fidelity.py --gfid-timing`). The timing job and its rule (exit 0 iff the re-projection is ≤ 72 GPU-h) are unchanged. The GPU-h of both timing runs are recorded in `ledger/gfid-timing-*.json`. This is the one relaunch P-crash allows for this job.

Posted: https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6386806922

---


## Decision P-novelty (2026-10-25): the descriptive #110 novelty scoring pass (owner decision D6)

**Rule.** #127 P-novelty: every arm of this child that passes its guards is scored on the #110 novelty evaluation pairs after its primary scoring pass. Descriptive only: no token, never part of a disposition or the program claim. Fixed now, before any arm of the program has an evaluation score.

**Population.** The `issue110-evaluation` levels of the 29 in-scope cells under #107's frozen rule (`run_granularity_under_novelty.in_scope_cells`: #110-supported, has mixed-verdict evaluation levels, not rollout-error-only), on both sides of every pair: the novel side at novelty level k = 1–7 and its paired level-0 normal side. Grid inventory (16 launches; the level-5 slingshot cells use #110's mirrored grid). Engine verdicts from the #110 handoff (`handoff_107.json`). AUC on mixed-verdict levels only.

**Inputs.** #107's anchor rule: the level's first complete grid candidate in suffix order, shot frame 0 (pre-launch). Pixel arms read that frame; carrier arms read its E99 v1 batch-1 parse (`rt.anchor_carrier` with the frozen E99 encoder, under `deterministic_scoring()`, the #102 N2 pattern); A3-sym builds its state from the level's `decision-physics-v1.json`. The E99 v1 carrier does not represent most novelty entities (#110: v1 fully supports 2 of 45 cells); carrier arms therefore see the novel levels through that limited contract, which is part of the descriptive result. No input contract is changed for this pass.

**Scoring.** Each guard-passing arm, all three seeds, its primary score (logit / Q for direct predictors; tie-free cost of the predicted endpoint carrier for world models), once, in a process that did not train, under `deterministic_scoring()`, after the arm's primary scoring pass. References on the same cells and anchors: U and C* (the Gate-B checkpoints rolled out as Gate-B scores them) and P (the Gate-B prior rule over the fit verdicts).

**Numbers (DESCRIPTIVE).** Per ranker: pooled grid AUC per novelty level on the novel side, on the paired normal sides, pooled over novelty levels 1–7, and the paired contrast novel − paired normal (one row per (pair, seed) where both sides are mixed); level-clustered (pair-clustered for the paired contrast) percentile bootstrap, 10 000 draws, PCG64 7201. Records are hashed into this child's records manifest and re-derived by `--validate`.

Shared implementation: `scripts/external_baselines.py` (`novelty_cells`, `read_novelty_anchors`, `novelty_reference_records`, `novelty_table`), pinned by every child plan; each child runner gets a `--score-novelty` mode before its freeze.

## Note (2026-10-25, orchestrator): runner hash in the precision correction
The precision-correction entry above records the runner sha256 `b7a5940e…` as it was right after the TF32 change. The runner changed afterwards (the `--score-novelty` mode, P-novelty issuecomment-6386831318), so `b7a5940e…` matches no committed file; the TF32 behaviour it describes is unchanged. The frozen `plan.json` pins the final runner bytes.

## Decision P-other (2026-10-25): the G-fid runs may share their GPU again (supersedes the shares_gpu part of the P-crash decision for every run after the timing run)

**Trigger.** The P-crash decision (issuecomment-6386806922) asked for `shares_gpu: false` on every `131-gfid-*` job so that the timing rerun would be clean. That timing run (run 3) is finished and recorded (`fidelity/gfid-timing.json`; P2E 0.142 s/update, THICK 0.317, DreamerV2 0.091). The remaining G-fid runs measure returns, not speed: sharing a GPU changes only their wall time, never their results (seeded, checkpointed, no time-dependent logic). P2E alone used about 45 % of its GPU, and the program budget counts device time (Decision P-budget on #127), so serialising six under-utilising runs on one GPU wastes budget and calendar time.

**Decision.** `shares_gpu: true` on `131-gfid-p2e`, the three `131-gfid-thick-*` and the three `131-gfid-dreamer-*` jobs (THICK on GPU 3, DreamerV2 on GPU 1). Their per-run ledgers keep the recorded process seconds; the program summary flags them as shared. The frozen band, pass rule, seeds, precision and window are unchanged.
## Decision P-crash (2026-10-26): shared replay-cache race at G-fid launch; fix, then relaunch all six runs once from scratch

**Trigger.** Rule P-crash (infrastructure; not a typed failure). After `131-gfid-p2e` completed (1M env steps, 8.54 GPU-h), the six G-fid runs (THICK Dreamer and DreamerV2 × seeds 20260908/09/10) launched together at 2026-10-25 23:45:20 AEDT. Each run lazily uncompressed the shared Plan2Explore episodes into the `.npy` cache. Several processes rewrote the same cache files at once (`os.replace` over files that other processes had already memory-mapped, on the mergerfs volume). Four runs failed within seconds with `OSError: [Errno 116] Stale file handle` at `np.load(..., mmap_mode="r")`: thick s20260908/09/10 and dreamer s20260908. The two that started (dreamer s20260909/10, about 0.4 h each) recorded no loaded-episode list or digest. Their replay therefore cannot be proven identical to a verified copy, and they may hold maps of replaced files.

**Fix (no change to training semantics).**
- The exploration cache is now built exactly once, by a separate CPU step (`b127_thick_fidelity.py --build-cache`). It is written to a temporary directory, every file is fsynced, the directory is renamed into place, and a `MANIFEST.json` records each episode's npz name, size and length and each array's shape, dtype and sha256.
- Runs only open the cache, read-only. They refuse unless the manifest covers exactly the source episodes with matching npz sizes, every array has its recorded shape and dtype, and the action and reward arrays re-hash to their recorded sha256. Each run writes the manifest's sha256 and its loaded counts to `replay_source.json`.
- The old lazily-built cache directory is no longer read.
- A full verification (`--verify-cache`) compares every array of every Plan2Explore episode with its cached copy and its sha256. Result (`fidelity/cache-verification.json`): **pass**. 500 episodes, 3 000 arrays and 1 000 000 steps match their cached copies and their sha256, with 0 mismatches. Manifest `sha256:ef527042…a43f092`.
- Code: `world_model/baselines/thick/envs.py` (`build_cache`, `open_cache`, `verify_cache`) and `fidelity_runner.py`. New CPU tests cover refusal of a missing, tampered or incomplete cache.

**Decision.** Stop dreamer s20260909/10 (exit 75), delete all six run directories, and relaunch all six runs once, from scratch, on the verified cache. All six then start from identical data. The Plan2Explore run is not repeated. Its episodes are the cache's verified source.
- **Ledger:** the discarded GPU time (about 0.4 h × 2 plus seconds for the four failed runs) stays in the ledger and is disclosed.
- **Placement:** THICK runs on GPU 3 and DreamerV2 runs on GPU 1, `shares_gpu`.
- **P-crash relaunch:** this uses the single relaunch for these six jobs.

Posted: https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6398071674

---

**Addendum to Decision P-crash (2026-10-26), comment 6398071674.** The orchestrator SIGTERMed dreamer s20260909/10; they exited 75 (paused), but the scheduler resumed both from their checkpoints at the next tick (run 2). They were then SIGKILLed and recorded as failed. Their run directories were deleted, and both were requeued to restart from scratch on the verified cache. The GPU time of their discarded runs (0.44 h of run 1, plus the seconds of run 2) is not counted toward the G-fid runs. The four relaunched runs (thick s20260908/09/10, dreamer s20260908) each record the verified manifest sha256 in `replay_source.json`.

Posted: https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6398164689

---

