## #123 implementer decisions (posted before the freeze)

Runner `scripts/run_hjepa_comparison.py`; adapter `world_model/hjepa/novphy.py`; vendored official code `world_model/hjepa/vendor/` (kevinghst/H-JEPA @ 97054ec, MIT, per-file upstream and vendored sha256 in `VENDOR.json`; the only edits make the top-level `loss` / `models.*` imports package-relative). Fidelity runner `scripts/run_hjepa_fidelity.py`, Tier-T launcher `scripts/hjepa_train_ckpt.py`, G-port reference `scripts/hjepa_port_reference.py`. Artefacts `.local-artifacts/issue-123-hjepa-comparison-v1/`; H-JEPA's own data and checkpoints in `HJEPA_HOME=/mnt/array/sukaih/hjepa-home`.

### Stage 0 (environment, G-fid, G-port)
- **D-env.** `hjepa` env: Python 3.11.13, torch 2.7.1+cu128 (sm_120 present), stable-pretraining 0.1.7, **transformers 5.20.1** (not `<5`): the released FourRoom checkpoints pickle `transformers.models.vit.modeling_vit.ViTMLP`, which exists only from transformers 5.9; with 4.57.6 every Tier-E load failed (`AttributeError: ViTMLP`). `huggingface-hub` 1.33 for `datasets`. Frozen stack in `HJEPA_HOME/env-freeze.txt`. The novphy env got `transformers==5.20.1`, `einops`, `zstandard` (no torch change).
- **D-fid-order.** G-fid runs concurrently with the NovPhy work instead of before the freeze: its pass rules were frozen in the ticket before any run, no information flows from FourRoom into any NovPhy choice, and running it serially (~25–30 h) would not fit the 72 h wall cap. It gates every H-JEPA token at publication.
- **D-tierT.** Tier T is the official `main_hjepa.py` through `scripts/hjepa_train_ckpt.py`, which (1) enables the ViT's non-reentrant activation checkpointing (batch 128 OOMs on a 32 GB RTX 5090; gradients unchanged, ViT has no dropout) and (2) calls `gc.collect()` every 10 batches: a per-step reference cycle keeps the step's graph alive until a rare generation-2 collection (measured +1.8 GB/step from step ~555 to an OOM at 578; flat ≤ 8.8 GiB over 1300 steps with it). Overrides: `final_probing_decoding_eval.enabled=false` (post-hoc, not part of G-fid), `num_workers=8`, `prefetch_factor=2` (host memory with 4 concurrent runs), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Three H-JEPA-3 launches (seeds 42/43/44) died of that CUDA OOM before step 900; they are infrastructure failures (no model output), relaunched once from scratch with the fix and disclosed; their logs and run dirs are archived.
- **D-gport.** G-port compares, on CPU in float32 with fixed inputs (batch 4, the AdaLN-zero modulation set to seeded N(0, 0.02) on both sides so every block has a nonzero gradient), the official code in the `hjepa` env against the vendored modules plus this port's loader-side sparse frame gathering, optimizer (AdamW per-level groups) and per-step warm-up/cosine scheduler, for HJ-px, HWM-px, LeWM-px, HJ-car-m and HJ-car-m-fs5. Tolerances (declared in the smoke): loss rel 1e-4, embeddings abs 1e-4, global grad norm rel 1e-4, per-tensor grad rel-L2 1e-3 (tensors whose official gradient is below 1e-6 × the global norm: abs-L2 below that floor), ≥ 99.9 % of parameters within 1e-7 after the step, LRs within 1e-12, identical level-1 frame sets and optimizer defaults. **Result: PASS for all five** (max loss rel 2.9e-6, max grad-tensor rel 3.9e-5).

### Stage 1 (data; fit and policy only)
- **Pixels.** Agent frames resized to 320 × 240 with `s99.image_tensor` (PIL bilinear, the E99 input), every position 0..min(last, 285) of the 13 919 fit shots (3 904 309 frames) and every 15th position of the 3 480 policy shots; zstd-3 RGB shards (66.8 GB), per-shard sha256 in `pixels/<split>/MANIFEST.json`, read from a byte-verified local-NVMe mirror (random reads through the mergerfs array were ~33 ms/frame).
- **Carriers.** Fit: the #112 retrain shot cache (the carriers the Gate-B unit trained on; #113 compared them at 6.6e-5). Policy: E99 parses of the stored positions and the level anchors, under `deterministic_scoring()`.
- **Probe labels.** Engine slot presence and engine-projected centres (`run_issue_70_parser_repair.targets`) at every 15th position of every fit shot, read only from the native chunks holding those steps (checked equal to `s99.shot_labels`).

### Stage 2 (arms)
- **Port.** Training does not use Lightning: one loop replicating spt 0.1.7 manual optimisation (bf16-mixed autocast, global clip 1.0, AdamW per-level LRs 5e-5 / 1e-5, wd 1e-3, linear warm-up over max(1, int(0.01·steps)) then cosine to 0, stepped per update), the objective is the vendored `hjepa_forward` unchanged. The loader draws each batch's crop starts and gathers only the level-1 frames some kept step reads (the official sparse encode), from `PCG64([seed, batch index])`.
- **ViT at 320 × 240.** `image_size=320`, patch 16: a 20 × 20 position table bicubic-interpolated to the 15 × 20 grid by the official `interpolate_pos_encoding` path (a rectangular table breaks that path).
- **Sizes.** HJ-px / HJ-px-B1 / HWM-px / LeWM-px use the paper's Push-T/Cube sizes (ViT-Tiny + MLP projectors; L1 predictor 4/8/1024; upper 6/16/2048; D 192): 35.2 M / 34.9 M / 11.35 M. Carrier arms: MLP(236→384→112) level-1 encoder, latent-MLP(112→224→112) upper encoders, ARPredictor depth 2, 4×28 heads, MLP 448, no projectors: HJ-car-m 1 855 172 (+0.95 % of 1 837 690); LeWM-car-m depth 7 / MLP 480: 1 834 220 (−0.19 %); HWM-car-m MLP 512: 1 840 100 (+0.13 %); HJ-car-m-fs5 depth 3 / MLP 544: 1 835 440 (−0.12 %); HJ-car-native 30.5 M.
- **Budgets.** B1 = 9000 × 64; B3 = ⌊valid clips / 128⌋ × 5: HJ/HWM 128 045 steps (3 277 954 valid H3 clips), LeWM-px 150 880.
- **Launch encoding.** `[drag_x/480, drag_y/480, release/1000, tap/1000, 1]` at frame 0, zeros after, no z-scoring; persist repeats it every frame; fs5 concatenates five frame actions per level-1 step.
- **Mixture sampler.** The first half of every batch starts at frame 0 of a uniformly drawn shot; the rest is uniform over all valid (shot, start) pairs. One crop start per level and batch (as the official model draws them).
- **λ selection.** HJ-px and HJ-car-m at B1, seed 20260908, λ ∈ {0.09, 0.18}; policy-split pooled grid AUC under R1; ties at 4 d.p. → 0.09. Each input type's λ applies to all its arms.
- **Typed failure.** A NaN loss ends the run with `complete: false`; never retried.

### Stage 3 (readouts)
- Probes follow H-JEPA App. D: MLP(D→512, LN, GELU→out), AdamW 1e-3, wd 1e-4, batch 128, 25 epochs, z-scored targets, fitted under `deterministic_scoring()` with the arm seed.
- R1 on the top-level encodings of every 15th fit frame (≈ 278 k frames) → the E99 carrier. R2: predicted top-level endpoints of every fit shot (its level anchor, its executed launch) → carrier at min(225, end). R3: mean top-level encoding of the true frame min(225, end) of the 838 pig-removing fit shots; cost = squared distance. R4: R2 + 0.1 × softplus((c_success − c_failure)/0.1) over 8 random fit mixed cells (grid, angle) per step. P_c: slot presence (BCE) + engine centres (MSE on present slots).
- G-readout ceiling: R1 on the true frame-min(225, last) encodings of the policy-split grid candidates, pooled AUC averaged over the arm's seeds ≥ 0.65. G-collapse and G-action per seed on the fit frames / 480 fit anchors × 16 grid launches (effective rank of centred top-level encodings, Roy–Vetterli; χ frame-weighted over shots; sensitivity = mean ‖z_i − z_j‖/‖z_i‖ over ordered launch pairs).

### Stage 4 (scoring)
- Anchors: carriers = the Gate-B anchor store (bit-identical inputs to U); pixels = frame 0 of the same anchor branches, read after the readout freeze. Top-level rollout: 15 steps of 15 frames (H3, fs5) or 225 steps (LeWM). Cost = `EndpointCosts.row(R1(ẑ_end), R1(z_0))`. U, C\* rows are their published Gate-B records; J / J_dec rows are the #101 published records (`J-pe-1e-04`, `J-dec-1e-01`); P is the Gate-B prior rule. S10 for HJ-px, HJ-car-m, HJ-px-B1, HJ-car-native with the 480 fit anchors.
- Position error: every grid candidate of the 311 mixed cells, t ∈ {15, 30, 60, 150, 225}, engine labels at min(t, last); H-JEPA via P_c on the predicted top-level state, floor = P_c on the true frame encoding; U / C\* via their predicted carriers.
- CLEVRER / Physion (EXPLORATORY): HJ-car-m and LeWM-car-m configs on the #117 item-5 annotation carriers, fold 'all', B1, uniform sampler, null action; R1 fitted in-sample on every clip frame (the #117 in-sample cohort); #79 `field_errors` position_mse at t ∈ {15, 30, 60, 120} from contexts 0–3, next to #117 item-5 J (LC, full, fold all).

### Compute and hardware
- GPU seconds = per-process wall on the GPU it ran on; GPUs are shared (Tier E is latency-bound and runs beside training), which inflates GPU-s. RTX 5090 instead of H200.

### Smoke and λ (pre-freeze; fit and policy splits only)
- **λ.** Selection readouts (R1 under `deterministic_scoring()`, 94 policy-split mixed grid cells), pooled policy grid AUC under R1 at B1, seed 20260908: HJ-px 0.4847 (λ 0.09) vs **0.5087 (λ 0.18)**; HJ-car-m 0.4800 (λ 0.09) vs **0.5053 (λ 0.18)**. No tie at 4 d.p., so **λ = 0.18 for both input types**, applied to every arm of that type. Both selected AUCs are near chance; this is the pre-declared rule, not a quality claim.
- **Smoke** (`smoke.json`, ok = true): 13/13 tests pass; G-port pass; 13 919 fit shots / 480 levels / 3 904 309 pixel frames, matching the carrier store; 13 919 label shots; 3 480 policy shots / 120 levels; only the fit and policy streams read. Throughput on one RTX 5090 beside the fidelity jobs (s/step at the budget batch → GPU-h per seed): HJ-px 0.442 → 15.7, HWM-px 0.410 → 14.6, LeWM-px 0.159 → 6.7, HJ-car-native 0.068 → 2.4, HJ-px-B1 0.242 → 0.60, carrier B1 arms 0.02–0.04 s → ≤ 0.09. **Projection 215 GPU-h, 49.9 h wall, against the 260 GPU-h / 72 h cap: within cap.** This includes 121 h of training, 7 h of readouts, 6 h of scoring, 2 h of benchmarks, 12 h of fidelity already spent and 63 h of fidelity remaining at the measured rates. The four LPT training queues hold about 30.3 h each.
- **Selection-run diagnostics** (fit/policy only, B1, for information; the guards apply to the main runs). Top-level effective rank: HJ-px 44 / 192, HJ-car-m 14 / 112. Within-episode share χ: HJ-px 0.002, HJ-car-m 0.017. Endpoint action sensitivity: about 2–5e-7 for all four runs. The trained top-level action pooler maps every grid launch to the same macro-action, up to 2e-6 against an embedding norm of 13; the level-1 action embedding still separates launches, with spread 0.4. The policy readout ceiling is HJ-px 0.79–0.83 and HJ-car-m 0.52–0.57. These are the App. K/L slow-feature signatures the guards were declared for. Nothing is changed in response: the objective, the guards and the thresholds stay as frozen in the ticket.
- **Scoring-path rehearsal (pre-freeze, policy/fit only).** A scratch run of `score_arm_seed` / `score_reference_curves` / `compute_tables` / `findings_md` used the λ-0.18 selection models on policy-split cells, with stand-ins for every evaluation input: policy anchors, policy carriers, policy frames and engine labels, and `_position_entry` on fit keys. It ran end to end, and the re-score replicated bit for bit. No evaluation feature, label or anchor was used. (Corrected in A2: this line first read "No evaluation file was opened", which is literally inaccurate; see *Evaluation-artefact access before scoring* below.)
- **Infrastructure.** The first `--smoke` process died after about 50 s without output or a traceback, and wrote no `smoke.json`. It was not detached from the launching shell. It was relaunched once, detached (`setsid`), and completed. Its log is kept as `logs/smoke-attempt1-killed.log`.
- **Docstring.** The runner's docstring was completed with the protocol before the freeze. No logic changed.
- **Compute rule.** The ticket's cap is 260 GPU-h and 72 h wall. The projection's 49.9 h wall assumes uncontended GPUs. Training runs beside the remaining Tier T and Tier E jobs, so the measured wall time will be longer. If the measured wall time passes 72 h, this is reported on the ticket. Scope is not cut.

## Post-freeze amendment A1: GPU-window schedule (2026-10-19, scheduling only)

**Owner rule** (10:40 AEDT, binding): GPU jobs run only from 18:00 to 07:00 Australia/Melbourne. The orchestrator stopped every #123 GPU process at 10:38, and none has run since. Readouts and scoring had not started, and no evaluation feature or label had been read. (Corrected in A2: this first read "no evaluation-split file had been read"; see *Evaluation-artefact access before scoring* below.) From the 19 October window on, every remaining #123 GPU stage goes through the night-window scheduler (`scripts/gpu_window.py`, `docs/gpu-window-scheduler.md`):
- cron runs a tick every 5 min;
- the scheduler launches jobs only in [18:00, 06:50);
- it sends SIGTERM at 06:50 and SIGKILL to anything left at 07:00.

**Amended files** (registered in `scripts/post_freeze_amendments.py`; frozen → amended sha256):
- **`scripts/run_hjepa_comparison.py`** (`db7f87ae…` → `b3390059…`).
  - `--train` handles SIGTERM and SIGUSR1. The running arm-seed writes `resume.pt` after its current optimizer step; the queue then exits 75, and the relaunch resumes there. The write is atomic (tmp + `os.replace`), as the frozen 2000-step checkpoints already were.
  - The CPU and CUDA RNG are restored after the loader iterator is created. Creating the iterator draws the worker seed from the CPU generator, which had shifted a resumed run's CPU stream; on the GPUs SIGReg draws from the CUDA generator.
  - `cuda_rng` is saved only when CUDA exists.
  - Objective, optimizer, schedule and batches are unchanged; batch *b* is `PCG64([seed, b])`, so the sampler position is the step.
  - `tests/test_issue_123_training_resume.py` (HJ-car-m, CPU) compares N uninterrupted steps against k steps, SIGTERM, resume and N − k steps. Model, optimizer state, scheduler and CPU RNG are bitwise identical, and so is the final `model.pt`. The test fails with the frozen restore order.
  - On the GPUs the restored state is exact, but the kernels (cuDNN autotune, bf16) are not bitwise reproducible run to run, paused or not.
- **`scripts/hjepa_train_ckpt.py`** (`ae6c27b8…` → `f07eed6a…`).
  - On SIGTERM, the official `ResumeCheckpoint` saves `lightning_resume/last.ckpt` through its own `_save` at the end of the current train batch, and the run exits 75. Lightning's own SIGTERM stop, during validation or the epoch-end hooks, also exits 75.
  - **Mid-epoch resume, the official semantics** (measured on CPU):
    - global step, optimizer and LR continue exactly;
    - the interrupted epoch runs only its remaining batches;
    - the data order is **not** restored: the remaining batches come from a new shuffle sequence.
  - A relaunch from a finished resume file runs, once, the train-end hooks that Lightning skips: the final object dump and the planning eval, which resumes from its atomic per-task chunks.
  - `main_hjepa.py` and every hyperparameter are unchanged.
- **`scripts/run_hjepa_fidelity.py`** (`0e78d456…` → `05bbd271…`).
  - `--queue` forwards SIGTERM to its job, records the job's exit in the ledger, starts no further job and returns 75.
  - It returns 1 if a job exited non-zero, or if a Tier-T job left no planning-eval metrics; before, it always returned 0.
  - `--table` and the G-fid rules are unchanged.

**Corrections and disclosures.**
- The periodic `resume.pt` writes were already atomic in the frozen code (`atomic_torch`: tmp + replace).
- Current checkpoints:
  - training resumes HJ-px/20260908 at step 100 000 and HJ-px/20260909, HJ-px/20260910 and HWM-px/20260908 at step 66 000;
  - Tier T H-JEPA-3 resumes at global step 6566 (seed 42, end of epoch 0), 2000 (seed 43) and 4000 (seed 44) of 45 962.
- The H-JEPA-3 relaunch segments started at 09:14 (seed 42), 10:03 (seed 43) and 09:47 (seed 44). They were killed at 10:38 and have no ledger line: about 2.9 GPU-h, counted below. Steps after each segment's last checkpoint are recomputed.
- Tier E finished 9/9 before the rule. Its chunk files are atomic and `metrics.yaml` is written last, so an interrupted task is never counted.

**Compute.** The caps count active compute time, not calendar time:
- a run's GPU seconds are its time on the GPU;
- the scheduler records each run's seconds;
- daytime pauses do not count.

Recorded so far: about 117 GPU-h.
- 43.9 h of checkpointed training;
- about 0.5 h of training steps recomputed;
- 64.5 h in the fidelity ledger;
- 2.9 h of unrecorded Tier-T segments;
- 4.9 h of benchmarks and selection.

Remaining, projected: about 141 GPU-h.
- training 93 h (queues 18 / 24 / 24 / 27 h);
- Tier T H-JEPA-3 32 h, including the train-end planning eval;
- readouts 8 h;
- scoring 6 h;
- publish and validate about 1.5 h.

The total, about 258 GPU-h, is close to the 260 GPU-h cap. Shared GPUs count each process's full wall time. If the measured total passes the cap, it is reported here and on the ticket. Scope is not cut.

Nights to finish: 141 GPU-h / (4 GPUs × 13 h) ≈ 2.7, so 3 nights. The critical path is training queue 3 (27 h), followed by readouts, freeze, score, publish and validate. The scheduler projects the last job, `--validate`, to end on 22 October at about 05:00.

## Post-freeze amendment A2: pre-scoring fidelity fixes (2026-10-19, before any readout or scoring)

**Trigger.** The pre-scoring fidelity review (`round2/review/123-prescore-fidelity.md`) found four must-fix defects. All four predate A1. No readout, readout freeze, record or evaluation-feature read existed when they were fixed: main training was paused by the GPU-window rule. Nothing was retuned, and no threshold, contrast, rule or token definition changed. The work was done on CPU only, in daytime.

**Amended files** (registered in `scripts/post_freeze_amendments.py`; frozen → amended sha256; the registry test `tests/test_issue_101_decision_aware_policy.py::test_issue_123_prescore_amendments_are_registered` checks each link and `load_plan`'s acceptance):
- **`scripts/run_hjepa_comparison.py`**: the chain is `db7f87ae…` (frozen) → `b3390059…` (A1) → `045530a9…` (A2). Only the docstring, the G-port mode and the guard / publication / validation code changed. Training, resume, RNG, data, readouts and scoring are byte-for-byte the A1 code.
- **`scripts/hjepa_port_reference.py`**: `7e2db8db…` → `c3087869…`.
- **`tests/test_issue_123_hjepa_comparison.py`**: `d4f22c42…` → `749ed45d…`. The 13 frozen tests are unchanged and 5 tests were added.

**The fixes.**
1. **G-records gates publication (P1).**
   - Before A2, `--publish` emitted tokens before the required re-score, and a failed `--validate` left them in place.
   - Now G-records is a global guard. It passes only if `validation.json` exists, binds the sha256 of the current `records-manifest.json`, that manifest lists exactly the current records, and the receipt's re-score replicated.
   - `--publish` therefore emits only `readiness_or_precision_insufficient` until validation has passed.
   - `--validate` keeps the frozen checks: every record hash, every published table re-derived and byte-compared, and one record re-scored under `deterministic_scoring()`. It then writes the receipt and re-derives the publication with G-records passing.
   - Any failure withdraws the publication, writes a failed receipt and re-derives the publication with every token the stop token, and `--validate` exits non-zero.
   - The scheduler queue is unchanged (publish, then validate). `--validate` now derives the tables twice; its 60 min estimate stands, and the dry-run still ends on 22 October at about 05:00.
2. **Failed-arm publication (P1).**
   - An absent paired estimate (no cell with both views) is now the stop token; before, `reverse(None)` crashed.
   - The unguarded `raw_token` / `raw_reverse_token` (contrasts) and `raw_token` (S10) fields were removed. The estimates stay as numbers.
   - The contrast and S10 assembly moved into `grid_contrasts` / `wrong_anchor_table`, with unchanged rules.
3. **G-nonfinite per readout (P2).**
   - Before A2, G-nonfinite counted R1 `excluded` flags only, so >10 % invalid R2 costs could still yield an S9 token.
   - It now counts the actual cost values (None or nonfinite) of each token-bearing readout separately: R1 (PC1, PC2, S1–S8, S10, S11) and R2 (S9).
   - Either share above 10 % fails the arm. By the frozen rule "a failed guard turns every token involving the arm into the stop token", that covers all of the arm's tokens, S9 included. `compute.json` reports `nonfinite_shares: {R1, R2}`, and `findings.md` shows both.
4. **G-port optimizer parity (P2).**
   - The frozen check stepped at the schedule's step-0 LR (0), so parameter-update parity was vacuous, and AdamW moments were not compared.
   - The reference and the port now run the frozen step unchanged, then advance the scheduler to the end of warm-up (every group at its base LR: 5e-5 level 1, 1e-5 upper), zero the gradients, and take a second step on the same inputs and crop starts.
   - Checks use the declared tolerances:
     - loss, gradient norm and per-tensor gradients, as for step 1;
     - LRs at and after the step within 1e-12, and every LR > 0;
     - AdamW step counts equal; `exp_avg` and `sqrt(exp_avg_sq)` under the per-tensor gradient rule, with the floor at 1e-6 × each moment's global norm;
     - ≥ 99.9 % of the updated parameters within 1e-7, and max |Δ| ≤ 2 × LR + 1e-7;
     - every parameter group moves (max |Δ| > 1e-7) on both sides.
   - The publication's G-port passes only if both the frozen `gport/gport.json` and the A2 `gport/a2/gport.json` pass.

**G-port A2 re-run (CPU, float32, all five arms): PASS.**
- The reference ran in the `hjepa` env (torch 2.7.1) and the comparison in `novphy` (torch 2.13), with `CUDA_VISIBLE_DEVICES=""`.
- Files: references in `gport/a2/reference-*.pt`; record `gport/a2/gport.json` (`sha256:c2b52c9d681d556128ad8ca2ceb207451491125b7a2d8b7fc45aafbc1e1ac184`); logs `logs/gport-a2-reference.log` and `logs/gport-a2.log`.
- The frozen `gport/gport.json` (`3be3bfea…`, pass) and its references are kept unchanged; the A2 record cites it by hash.
- The A2 references' step-1 fields (initial state, inputs, gradients, embeddings, losses, post-step state, LRs, crop draws) are bitwise identical to the frozen references.
- A wrong gradient clip (0.25) still passes every step-1 quantity but fails the A2 `exp_avg` check (test `test_positive_lr_step_detects_what_the_lr_zero_step_cannot`).

| arm | LRs at the step | loss rel | grad rel-L2 | exp_avg / √exp_avg_sq rel-L2 | params within 1e-7 | max \|Δ\| | min moved share | update rel-L2 (diagnostic) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HJ-car-m | 5e-05/1e-05/1e-05 | 1.2e-07 | 1.8e-06 | 1.8e-06 / 1.9e-06 | 0.99999 | 3.6e-07 | 0.990 | 5.2e-06 |
| HJ-car-m-fs5 | 5e-05/1e-05 | 2.1e-07 | 1.9e-06 | 2.7e-06 / 1.6e-06 | 1.00000 | 1.2e-07 | 0.994 | 4.5e-06 |
| HJ-px | 5e-05/1e-05/1e-05 | 2.3e-06 | 3.1e-05 | 2.6e-05 / 2.5e-05 | 0.99989 | 1.8e-05 | 0.996 | 6.6e-04 |
| HWM-px | 5e-05/1e-05/1e-05 | 2.1e-06 | 3.8e-05 | 3.3e-05 / 3.2e-05 | 0.99987 | 2.5e-05 | 0.996 | 8.8e-04 |
| LeWM-px | 5e-05 | 2.9e-06 | 3.0e-05 | 2.8e-05 / 2.7e-05 | 0.99966 | 1.6e-05 | 0.999 | 7.2e-04 |

The pixel arms' largest single-element deltas, up to 2.5e-5 against the 1e-4 bound, sit in the tensors whose exact gradient is 0 (16–20 per arm). There Adam divides float noise by ε, so the update is noise. LeWM-px's close share, 0.99966, is the nearest to its bound (0.999).

**Evaluation-artefact access before scoring.** The earlier wording "no evaluation file read / opened" was literally inaccurate. Before any scoring:
- `stream_rows` parsed the mixed-stream campaign `branches.json`, which holds fit, policy and evaluation rows, and kept only the fit / policy rows; it refuses the evaluation stream.
- The #109 cohort plan, which lists every split's level identities, was parsed for split membership.
- `--prepare` and every `load_plan` (each `--train` launch included) sha256-hash the pinned inputs. Among them are evaluation artefacts: `branches.json`, the Gate-B `anchors-evaluation.pt`, the Gate-B U / C\* records, plan and summary, and the #101 J / J_dec records. Hashing reads the bytes but decodes nothing.

No evaluation feature (anchor, frame, carrier) or label (verdict, engine label) entered training, readout fitting, λ selection, the smoke or the scoring-path rehearsal; the pre-scoring review found no such flow either. This is a wording correction, not evidence of tuning leakage. The runner docstring was corrected in the same way.

**Verification (CPU, `CUDA_VISIBLE_DEVICES=""`).**
- Test suites:
  - `tests/test_issue_123_hjepa_comparison.py`: 18 passed;
  - `tests/test_issue_123_training_resume.py` (resume exactness): 3 passed;
  - `tests/test_gpu_window.py`: 17 passed;
  - `tests/test_issue_101_decision_aware_policy.py` (amendment registry): 18 passed;
  - `tests/test_issue_104_heldout_contrasts.py`: 7 passed.
- `load_plan` accepts the amended files.
- `python scripts/gpu_window.py dry-run` is unchanged: at 18:00 it launches the four training queues and the three Tier-T jobs, and every job ends by 22 October at 05:00.

## Post-freeze amendment A3: compute accounting (2026-10-21, after publication `c5883f55`)
The first publication printed 310.3 GPU-h. Its `train()` ledger had counted four arm-seeds twice: HJ-px 20260908 / 09 / 10 and HWM-px 20260908, each exactly 2 × its `report.json`, +285 598 GPU-s (79.3 GPU-h). A relaunched training queue re-visits finished arm-seeds, `train_run` returns their stored report, and `train()` marked it again. A3 is registered in `scripts/post_freeze_amendments.py`; the chain of `scripts/run_hjepa_comparison.py` is `db7f87ae…` (frozen) → `b3390059…` (A1) → `045530a9…` (A2) → `86e8ee9e…` (A3), commit `aa8e702f`.

- **`train()`.** An arm-seed's training seconds are marked only if its `model.pt` did not exist before the call.
- **`ledger_totals` (rule).** Each arm-seed's main training counts once. A `train-<arm>-seed<s>` phase whose `models/<arm>/seed-<s>/report.json` exists counts that report's seconds. That is the elapsed time `train_run` carries across resumes, exactly what one completion marks. A ledger entry that differs keeps its raw seconds as `ledger_gpu_seconds` / `ledger_wall_seconds` and is not counted. Every other phase counts its ledger seconds. The ledger files themselves are left as written.
- **Publication.** `compute.json` adds `training_gpu_seconds`, `training_rule` and `ledger_overcount_gpu_seconds`, and `findings.md` gets one training line under Compute.
- **#125.** #125 aggregates through the same `ledger_totals`. Its plan pins the A2 bytes `045530a9…`, which are in A3's `replaces`.
- **Unchanged.** Training math, resume, RNG, data, readouts, scoring, records, tokens, contrasts, guards and dispositions.

**Re-publication and re-validation.** Both ran through the night-window scheduler on GPU 0, sharing it with `125-train-q0` (`scripts/gpu_window_issue_123_a3.json`):
- `123-publish-a3#1` ran 18:50:05–18:54:14 AEDT and exited 0.
- `123-validate-a3#1` ran 18:55:07–19:04:04 AEDT and **exited 0**:
  - 51 records hash-checked;
  - every table re-derived and byte-compared;
  - HJ-car-m seed 20260908 re-scored on 311 grid cells, max delta 0.0.
- Both exit codes come from the scheduler's `ended` events (`logs/scheduler-123-events.jsonl`).

**Diff against `c5883f55`.**
- **Unchanged.** `summary.json`, `records-manifest.json` and `readouts-manifest.json` are byte-identical. Every `compute.json` key other than `compute` is identical. `findings.md` is identical up to its `## Compute` section.
- **`compute.json` → `compute`:**
  - `issue_123_gpu_seconds`: 788 488 → 502 890;
  - `issue_123_wall_seconds`: 788 692 → 503 094;
  - `total_gpu_hours`: 310.32 → 230.99;
  - the four doubled phase entries now count their report seconds;
  - three keys were added.
- **`validation.json`.** Only `validated_at` changed.

**Corrected compute.**
- Training (33 `report.json`): 133.5 GPU-h.
- #123 recorded: 139.7.
- Fidelity: 91.3.
- **Recorded total: 231.0.**
- Unrecorded:
  - ≈ 3.6 (H-JEPA-3 segments killed at 10:38, recomputed steps, first launches);
  - ≈ 0.4 (scheduler `--publish` / `--validate` runs, which write no ledger line).
- **Active total: ≈ 235.0 GPU-h, against the 260 cap.**

**Tests (CPU, `CUDA_VISIBLE_DEVICES=""`).**
- `tests/test_issue_123_compute_ledger.py`: 4 passed:
  - a relaunch over a finished arm-seed adds no ledger seconds;
  - once-per-arm-seed aggregation on both the #123 and #125 layouts;
  - the #123 training total equals the `report.json` sum.
- `tests/test_issue_123_hjepa_comparison.py`: 18 passed.
- `tests/test_issue_123_training_resume.py` (resume exactness): 3 passed.
- `tests/test_issue_125_hjepa_idm.py`: 18 passed.
- `tests/test_issue_125_amendments.py`: 1 passed.
- `tests/test_issue_101_decision_aware_policy.py` (registry, incl. `test_issue_123_a3_is_accepted_by_both_plans`): 20 passed.
- `tests/test_issue_104_heldout_contrasts.py`: 7 passed.
- `tests/test_gpu_window.py`: 22 passed.
