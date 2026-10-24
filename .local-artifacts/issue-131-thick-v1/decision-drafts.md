# #131 D1 THICK — decision drafts (not posted; for the orchestrator)

## Decision P-route (2026-10-24): G-fid on our PyTorch port (route ii), not the official TF2 stack

**Trigger.** Rule P-route / #131 readiness gate 2: the G-fid run (THICK Dreamer vs DreamerV2 on VisualPinPad, band frozen in the Decision P-fid comment) may use the official code only if the official TF2 stack runs on the RTX 5090s.

**Evidence (scheduler jobs `131-tf-gpu-probe`, `131-gpu-probe-2`, GPU 3, 2026-10-24 20:25–20:55 AEDT; `fidelity/tf242-probe-summary.json`, `fidelity/tf-gpu-probe-tf2x*.json`).**
- Official pins (tensorflow-gpu 2.4.2 + CUDA 11.0 / cuDNN 8.0.5, env `b127-thick-tf`): the 5090 (compute capability 12.0) is registered after ~20 s; the first matmul takes ~2.4 min (CUDA 11.0 has no sm_120 code: PTX JIT) and logs "TensorFloat-32 will be used"; the first cuDNN 8.0.5 convolution had not returned after 5.3 min when the 8-min limit ended the probe. The official THICK step (CNN encoder/decoder) was never reached. Round 1 (5 min) produced no output at all.
- Current TensorFlow (2.21.0 `[and-cuda]`, CUDA 12.5 build, SASS sm_60–sm_89 + PTX compute_90, env `b127-thick-tf2x`): "Cannot dlopen some GPU libraries … Skipping registering GPU devices" — no GPU. It would also need code edits (Keras 3, the removed `mixed_precision.experimental` API), i.e. not the official code.
- Paper Suppl. G: the official code took 15–20 h per DreamerV2 run and 35–45 h per THICK Dreamer run on one GPU; 3 + 3 seeds + exploration data would be ≥ 160 GPU-h even at that speed.

**Decision.** Route **(ii)**: extend the G-port-verified PyTorch port (`gport/gport.json`: pass on the carrier and the official VisualPinPad configurations; worst update relative L2 1.6e-5) to the full THICK Dreamer agent, DreamerV2 and Plan2Explore. Every added module passes G-port against the vendored TF2 code on CPU before any G-fid run: EnsembleRSSM (the TF reference needs one listed vendored edit — deleting the line `self._rnn_type = rnn_type`, an undefined name that makes every non-THICK model crash at construction at 934980a), ActorCritic (actor, critic, slow target, StreamNorm, λ-return), `WorldModel.imagine`, THICK Dreamer's `thick_imagine` / `long_term_return` / coarse critic, Plan2Explore's disagreement ensemble, acting, replay, driver and the PinPad environment with its wrappers (CPU equality tests), and the `train.py` / `train_from_replay.py` loops (resumable: SIGTERM → checkpoint → exit 75).

**G-fid design (frozen with this decision).** VisualPinPadFour (the paper's tuning task; band threshold 359; Table 3: 140k vs 200k). One Plan2Explore run of 1M env steps (`train.py`, configs `visualpinpad plan2explore`) whose replay pre-fills every run (`train_from_replay.py`); the paper trained two exploration models and kept the higher-return one — disclosed. THICK Dreamer (`thick_pinpad`) and DreamerV2 (`visualpinpad`) × seeds 20260908, 20260909, 20260910, each to 400k env steps (the Fig. 8 window); one 2000-step evaluation episode every 1e4 env steps (upstream `eval_every` / `eval_eps`). Float32, TF32 off (as the G-port and the D1 adapter). Pass rule: the frozen band (returns of both methods at 400k inside their bands; THICK reaches 359 strictly earlier than DreamerV2). 3 seeds per method instead of the paper's 7 — disclosed.

**Projection** (`fidelity/plan.json`, measured port seconds per update on a 5090 plus a behaviour-update proxy; [INFERENCE] until the first run's throughput replaces it): THICK Dreamer 9.2 GPU-h per run, DreamerV2 6.8, Plan2Explore 19.4 → **67.2 GPU-h** (51.4 with TF32). The first 1 000 updates of each kind are timed and the projection re-checked before the rest runs.

## Decision P-cap (2026-10-24): D1 projection within the child share; fidelity above the #127 sub-estimate

**Trigger.** Rule P-cap: the G-fid projection (67.2 GPU-h) exceeds #127's D1 fidelity estimate (24–48 GPU-h).

**Child projection** (`fidelity/plan.json`): fidelity 67.2 + selection 3 configurations × 14.0 (native 200 000 updates at the measured 0.16–0.22 s per update on the carrier model, +15 % overhead) = 42.0 + training of two more seeds 28.0 + matched secondary ~2 + scoring / validate ~2 = **~141 GPU-h**, inside the child share (~86–146) and ~35 h wall on 4 GPUs (≤ 48 h).

**Decision.** No cut: the child total stays within its share, so P-cap's cut order does not apply. If a measured smoke pushes the child total over 146 GPU-h, P-cap applies in its order (unallocated program budget first; then matched secondary runs; then the selection grid 3 → 2; never the primary arm's native budget or its three seeds). Reducing the G-fid design (fewer seeds, a shorter window) is not an option under P-cap.
