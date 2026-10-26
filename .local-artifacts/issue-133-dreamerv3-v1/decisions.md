# #133 decisions

## Decision P-coverage (2026-10-25): start DreamerV3's CPU implementation now; GPU funding decided by its smoke projection

**Trigger.** Rule P-coverage funds coverage arms in order (DreamerV3 first) "each only if its smoke projection fits the remainder" after P-budget items (1)–(4). At 06:22 on 25 Oct the program had used 37.5 device GPU-h of 400 (`run_issue_127_program --budget`); the program-set children project their remaining work in their smokes, and no Tier-C trigger can be evaluated before their publications.

**Decision.** Start this child's CPU work now (vendoring, port, G-port, the PushT fidelity harness, the frozen band, the NovPhy runner and its smoke), because a smoke projection is what P-coverage needs and the CPU work does not draw on the GPU budget. No GPU job of this child is enqueued until a later "Decision P-coverage (funding)" compares its smoke projection with the remainder after (1)–(4) are allocated (device time; Decision P-budget on #127). If it does not fit, the arm is listed as not run (typed `cap`).

**Not affected:** the program set, its order of funding, and every frozen child rule. This arm never enters the program claim.

## Decision P-license (2026-10-25): vendor NM512/dreamerv3-torch @ 6ef8646 (MIT) — world-model import closure, verbatim

**Trigger.** Phase 0 of #133 vendors the official DreamerV3 code (rule P-license).

**Facts checked.** https://github.com/NM512/dreamerv3-torch at `6ef8646d807cd10ce0c88e10a7e943211e7fc44c` (2026-03-08, "Merge pull request #81", repository archived) carries a root `LICENSE`: MIT, "Copyright (c) 2023 NM512" (sha256 recorded in `VENDOR.json`). MIT permits use, copying and redistribution with the notice. This is the implementation the DINO-WM appendix (A.9) credits for its DreamerV3 baseline.

**Options.** (a) vendor the official code verbatim; (b) re-implement from the paper (only if no licence). The licence allows (a).

**Decision.** Vendor into `world_model/baselines/dreamerv3/vendor/` (schema `issue_127_vendor_v1`, per-file upstream / vendored sha256): `models.py` (`WorldModel`: encoder, RSSM, heads, `_train` with the KL-balanced loss and the optimizer call), `networks.py` (`RSSM` with discrete 32×32 latents, unimix, KL balancing + free bits; `MultiEncoder` / `MultiDecoder` incl. the MLP path for vector observations with symlog inputs and the symlog-MSE vector output; CNN encoder / decoder for the PushT fidelity model; `MLP`, `GRUCell`), `tools.py` (`symlog` / `symexp`, `OneHotDist`, `DiscDist` (two-hot, reward head only), `SymlogDist`, `Optimizer` (Adam, grad clip 1000), `static_scan`, weight inits), `configs.yaml` (the official defaults, read by the adapter) and `LICENSE`. Not vendored: `dreamer.py` (online agent loop, actor-critic driver), `exploration.py`, `parallel.py`, `envs/`. Allowed edits: imports made package-relative (`import networks` → `from . import networks`, `import tools` → `from . import tools`) and one listed compatibility edit (moving `from torch.utils.tensorboard import SummaryWriter` into `Logger.__init__`, the only user; recorded under P-route). Nothing is redistributed beyond the MIT notice requirements (LICENSE vendored).

Posted: https://github.com/Sino-Huang/NovPhy/issues/133#issuecomment-6388618813

## Decision P-route (2026-10-25): run DreamerV3 (pinned torch 2.4.1) on torch 2.13 in the `novphy` env; G-port reference on the same build

**Trigger.** NM512/dreamerv3-torch @ 6ef8646 pins `torch==2.4.1` (and gym / mujoco / tensorboard for its online agent); our stack is torch 2.13 + CUDA 13 (RTX 5090). Rule P-route applies when an official stack does not run here.

**Facts (CPU, 2026-10-25).** In `novphy` (Python 3.11, torch 2.13.0+cu130) the official world-model files import and train after (i) one compatibility edit: `tools.py` imports `torch.utils.tensorboard.SummaryWriter` at module level for its `Logger` (not used by the world model; tensorboard is not installed) — moved into `Logger.__init__`, listed in `VENDOR.json`; the rest is unchanged. Two run-time facts handled without editing any file: `networks.MLP` defaults to `device='cuda'` for a fixed-std tensor that the encoder / symlog-MSE decoder never read (CUDA-less hosts, i.e. every CPU test and the G-port, cannot build it) → the adapter and the G-port reference set that default to the run device while a model is constructed; `WorldModel.preprocess` divides `obs['image']` by 255 unconditionally (upstream envs always deliver an image) → the adapter's subclass skips it when there is no image, and the G-port reference feeds the official code a dummy `image` that no encoder / decoder key selects. `torch.cuda.amp.GradScaler / autocast(enabled=False)` only emit deprecation warnings (precision 32: inert). Upstream's `torch.compile` (config `compile: True`) is not used: the adapter runs eagerly (same graph).

**Options.** (a) a dedicated `b127-dreamerv3` env with torch 2.4.1 (no sm_120 kernels for the RTX 5090 in 2.4.1 wheels [INFERENCE], and a cross-version G-port fails the #123 |Δ| ≤ 1e-7 share by Adam round-off — the #129 lesson); (b) run the official code on torch 2.13 with the listed edit / shims, and run the G-port reference (the unedited clone) on the same torch 2.13 build.

**Decision.** (b). No new env for the world model. G-port: both sides in `novphy` (torch 2.13, CPU, float32, separate processes) under #123's tolerances and #123's exact parameter rule. The PushT fidelity harness's planning stage runs DINO-WM's unchanged planner in the existing `b127-dino_wm` env (Python 3.10, torch 2.7.1; #130), loading the DreamerV3 checkpoint trained in `novphy`; recorded under the fidelity decision.

Posted: https://github.com/Sino-Huang/NovPhy/issues/133#issuecomment-6388683089

## Decision P-other (2026-10-25): the T1 DreamerV3-car adapter (model, data, training length, readout, gates)

**Trigger.** #133 needs a NovPhy adapter for the vendored official world model (P-license issuecomment-6388618813, P-route above) before the G-port fixes what it compares.

**Decision (`world_model/baselines/dreamerv3/novphy.py`).**
1. **Model = the official `models.WorldModel`** at dreamerv3-torch's `defaults` with the official vector-observation block `dmc_proprio` (`mlp_keys '.*'`, `cnn_keys '$^'`): RSSM deter 512 / hidden 512, 32 × 32 discrete latents, unimix 1 %, learned initial state, LayerNorm + SiLU; encoder MLP 5 × 1024 with symlog inputs on the 236-d E99 carrier; decoder MLP 5 × 1024 with the symlog-MSE output; KL balancing dyn 0.5 / rep 0.1 with free bits 1.0; Adam lr 1e-4, eps 1e-8, grad clip 1000; fp32 (upstream `precision: 32`, torch's default TF32 flags), eager (upstream compiles). Size preset: **dreamerv3-torch `defaults`** (the configuration of the code DINO-WM credits; not mapped to a paper size label). **14 671 084 trainable parameters** (encoder 4 446 208, RSSM 4 205 568, decoder 6 019 308).
2. **No reward / continue heads** (offline, task-free data, as DINO-WM trained its DreamerV3 baseline): the heads are removed and the official optimizer rebuilt over the rest; this equals the official model trained with `reward_head.loss_scale = cont_head.loss_scale = 0` (checked by G-port). The training step is the official `WorldModel._train` unchanged (sampled posteriors, decoder loss + KL loss, `tools.Optimizer`); the only override is `preprocess` (no `image / 255` without an image).
3. **Episodes.** One fit shot = 46 observations, carriers at frames 0, 5, …, 225 (stride 5); shots ending before 225 repeat their final carrier (as #131). The launch `eb.launch5` is the action at index 1 (upstream layout: `action[t]` led into observation t, so the launch is applied at t0 and leads into frame 5); every later action is the zero no-op; `is_terminal` 0; no reward. Training data = the #109 fit split only (`eb.FitShots`, 13 919 shots).
4. **Batches = upstream's**: 16 rows × 64 steps built as `tools.sample_episodes` builds them (a random start in [0, 44] of a first episode, then whole episodes from index 0, `is_first` at each segment start), episodes uniform (equal lengths). Batch b's draws come from `eb.batch_rng(seed, b)`; the torch generator of the posterior samples is seeded and checkpointed (resumable, SIGTERM → `resume.pt` → exit 75).
5. **Training length = DreamerV3's replay ratio over the offline dataset**: dreamer.py performs one update every `batch_size × batch_length / train_ratio` agent steps, i.e. `train_ratio` (512) replayed steps per dataset step: 13 919 × 46 × 512 / (16 × 64) = **320 137 updates** per native run. A nonfinite loss is a typed terminal failure (never retried).
6. **Readout.** Observe the anchor carrier at t0 (`obs_step` from the initial state), then 45 `img_step`s (launch at the first, no-ops after), deterministic (latent modes, `sample=False`); the endpoint carrier is the decoder mode (symexp) at frame 225 → tie-free `EndpointCosts` (primary). No native success head (no reward head) → no native-head score, no U + head control.
7. **Guards.** G-rank / G-action on the decoded frame-225 carriers (480 fit anchors × 16 grid launches); **G-collapse on the RSSM feature** `get_feat` (one-hot stochastic mode 1024 + deter 512 = 1536-d) of the deterministic posterior on fit episodes at frames 0, 15, …, 225; G-nonfinite; wrong-anchor control (480 fit anchors) and position error at t ∈ {15, 30, 60, 150, 225} (descriptive) from the decoded rollout. One arm (T1, primary, coverage; never in the program claim); no matched-budget arm.
8. **Gates of the runner.** `--selection` refuses unless the orchestrator's funding marker (`funded.json`, Decision P-coverage funding) exists, G-port passed and **G-fid (PushT) passed** (validation ladder: cheapest decisive rung first); `--score` refuses unless G-fid passed; a G-fid failure after ≤ 2 logged repairs drops T1 (`fidelity_not_reproduced`).

Posted: https://github.com/Sino-Huang/NovPhy/issues/133#issuecomment-6388683322

## Decision P-fid (2026-10-25): frozen G-fid band for T1 — DINO-WM Table 1 DreamerV3 PushT 0.30 → [0.18, 0.42]

**Trigger.** #127 requires the G-fid band to be frozen and posted before any fidelity run. Target (#133 body): DINO-WM (arXiv 2411.04983v2) Table 1, row DreamerV3, PushT success rate **0.30** — the offline DreamerV3 world model (NM512/dreamerv3-torch, appendix A.9) trained on DINO-WM's PushT dataset without reward or task information (§4.2), then MPC with CEM over the learned world model on 50 evaluation cases.

**Options.** (a) ±2 binomial SE on 50 cases around 0.30 (the ticket's proposal); (b) a wider band to absorb the reconstruction uncertainty (DINO-WM released no baseline harness: training length, resolution and latent choices are ours, posted before the run); (c) an exact-match rule.

**Decision.** (a), as #130 did for DINO-WM's own row: SE = √(0.3 · 0.7 / 50) = 0.0648, ±2 SE = [0.1704, 0.4296]; on the 1/50 success grid **9..21 successes = [0.18, 0.42] inclusive**. Statistic: `final_eval/success_rate` of DINO-WM's unchanged `PlanWorkspace.perform_planning` (sticky per case; success = PushT `eval_state`: position error < 20 and angle error < π/9), one run, seed 99, `plan_pusht.yaml` (goal_source dset, goal_H 5, MPC-CEM horizon 5, 300 samples, top-30, 30 CEM steps, var_scale 1, 5 actions taken per iteration), MPC capped at **12 iterations** (= 300 env steps, the `pusht` gym registration's `max_episode_steps`; identical to #130's frozen cap, since the released config's `max_iter: null` never terminates while a case is unsolved). (b) is rejected: widening a band for our own reconstruction choices would let a harness defect pass; a miss gets at most two logged repairs, each fixing a diagnosed defect (never the band), then T1 is dropped with `fidelity_not_reproduced`.

**Frozen file.** `.local-artifacts/issue-133-dreamerv3-v1/fidelity/band.json` (schema `issue_127_gfid_band_v1`), sha256 `656ea9238476e528441ef91ceac3d2ee14404b32ac9c5d910d771151456f9e66`. No fidelity job has run (none may run before the P-coverage funding decision).

Posted: https://github.com/Sino-Huang/NovPhy/issues/133#issuecomment-6388719519

## Decision P-other (2026-10-25): reconstruction of DINO-WM's DreamerV3 PushT harness (every choice, before any run)

**Trigger.** DINO-WM @ 0a9492f released no baseline code. §4.2 states only: DreamerV3 (NM512/dreamerv3-torch, A.9) trained on DINO-WM's offline datasets without reward or task information, then MPC on the learned world model; §3.2 / A.5.1: CEM over actions, cost ‖ẑ_T − z_g‖² with ẑ_0 = enc(o_0), z_g = enc(o_g). Everything else is reconstructed here; `scripts/b127_dreamerv3_fidelity.py` implements it.

**Choices.**
1. **Data.** DINO-WM's released `pusht_noise` (OSF, sha256 in `data/external/dino_wm/MANIFEST.json`, #130). Training: the train split only (18 685 episodes, 2 336 736 frames), read through DINO-WM's own `PushTDataset` (its normalised 2-d relative actions and its normalised 4-d proprio = agent position + velocity, `with_velocity`). Planning targets: the val split through DINO-WM's unchanged `PlanWorkspace.prepare_targets` (seed 99) — the same 50 start/goal pairs as #130's run (checked against its `plan_targets.pkl`).
2. **Resolution 64 × 64.** dreamerv3-torch's default `size: [64, 64]`; its CNN encoder / decoder need a power-of-two side (stages = log2(side / 4)), so DINO-WM's 224 × 224 frames cannot be decoded at 224 (the decoder would emit 128 × 128). Each 224 frame in [0, 1] is resized by antialiased bilinear interpolation (`F.interpolate(..., antialias=True)`) and rounded to uint8 — the same function in the cache and in the planner's encoder.
3. **Observation keys.** `image` (CNN, the official `dmc_vision` block: depth 32, minres 4) + `proprio` (the official MLP pathway, 5 × 1024, symlog inputs; image + vector keys as the official minecraft config combines them); the decoder reconstructs both. No reward / continue heads (= official loss scales 0; G-port arm `pusht-64`); 23 971 719 parameters.
4. **Frameskip 5, 10-d actions.** One model step = 5 env frames (the released PushT `frameskip: 5`); action[k] = the 5 raw normalised actions between steps k−1 and k concatenated (DINO-WM's `concat` processing; the planner's actions are the same 10-d objects), action[0] = 0 with `is_first`. Every frame is used: each train episode yields 5 phase sequences (frames p, p+5, …; p = 0..4), as DINO-WM's slicer uses every start.
5. **Training = DreamerV3's native recipe offline.** Official `WorldModel._train`, batch 16 × 64 drawn as upstream `tools.sample_episodes` (length-weighted episode, random start, whole episodes appended, `is_first` at segment starts), Adam 1e-4, fp32, seed 0 (upstream default); **233 674 updates** = replay ratio 512 over the dataset's agent steps (2 336 736 frames / frameskip 5 = 467 347) / (16 × 64), the same rule as the NovPhy arm. Resumable (SIGTERM → `resume.pt` → exit 75).
6. **Planning latent.** enc(o) = the posterior from the initial state given that single frame (`obs_step`, `is_first`, latent mode); rollout = `img_step` with latent modes (`sample=False`) under the planned 10-d actions; **z = the categorical probabilities of the stochastic state (32 × 32, with unimix)** — DreamerV3's "categorical representation"; the deterministic GRU state is excluded because a single-frame goal encoding has no history (its deter is the learned initial state, not a predictable target). The `proprio` entry of the objective is zero (the latent already embeds proprio), so DINO-WM's unchanged `objective_fn` (mode last, alpha 1) is the MSE between the predicted probabilities at step H and the goal's.
7. **Planner unchanged.** `planning/` and `PlanWorkspace` / `PlanEvaluator` imported from the clone, `plan_pusht.yaml` values, MPC capped at 12 iterations (P-fid band decision). The evaluator's rollout plots need a decoder of DINO-WM's latents; the wrapper sets `decoder = None`, so the official code skips them (they never enter success).
8. **Stacks.** Cache build and planning in `b127-dino_wm` (#130's env: Python 3.10, torch 2.7.1, decord, gym, the PushT env); training in `novphy` (torch 2.13, the G-port build). Planning resumes from #130's MPC-iteration checkpoint (`install_mpc_checkpointing`, imported). Every stage refuses to run without the orchestrator's `funded.json`.
9. **Statistic** as frozen in `fidelity/band.json` (one run).

Posted: https://github.com/Sino-Huang/NovPhy/issues/133#issuecomment-6388719795

## Decision P-other (2026-10-25): T1 runner — three pre-declared selection configurations, budgets, queues and P-novelty

**Trigger.** #127 allows ≤ 3 pre-declared configurations per arm, posted before any selection run; the child-runner contract (`round3/child-runner-contract.md`) and P-novelty (`round3/score-novelty-mode.md`) fix the rest. No selection run exists; none may run before the P-coverage funding decision (`funded.json`).

**Options considered.** DreamerV3's paper keeps its hyperparameters fixed across domains and names two scaling knobs, model size and replay ratio. Larger presets multiply cost (the default model already projects ~24 GPU-h per native run [INFERENCE]); a learning-rate check is the standard tuning axis and costs one native run.

**Decision (`world_model/baselines/dreamerv3/novphy.py: CONFIGURATIONS`, `scripts/run_issue_133_dreamerv3.py`).**
1. Configurations, in declaration order: **`native`** (dreamerv3-torch defaults: Adam 1e-4, train_ratio 512 → 320 137 updates), **`lr3e-4`** (model_lr 3e-4, 320 137 updates), **`rr128`** (train_ratio 128 → 80 034 updates). Same model (14 671 084 parameters), batch 16 × 64, seed 20260908, full budget each, fp32; G-port covers `native` and `lr3e-4` (`rr128` changes only the update count).
2. Selection: policy-split pooled grid AUC of the primary score (tie-free cost of the decoded frame-225 carrier) under `deterministic_scoring()`, `eb.select_configuration` (4 d.p., ties → fewer parameters → declaration order; a failed run is never selected). The selected run is reused as main seed 20260908; seeds 20260909 / 20260910 at the selected configuration.
3. Gates: `--selection` / `--train` refuse without `funded.json`, a passing G-port and a passing PushT G-fid; `--score` refuses without G-fid pass. Scheduler specs are prepared, not enqueued: `scripts/gpu_window_issue_133.json` (cache, PushT training and planning, 3 selection jobs on GPUs 0/1/2, selection readouts, smoke) and `scripts/gpu_window_issue_133_main.json` (after the freeze: 2 training queues, readouts, freeze-readouts, score, score-novelty, publish + validate).
4. P-novelty (the protocol posted on #128–#131): after the primary scoring pass, T1 (if its guards pass) is scored on the 29 in-scope #110 cells' mixed novelty levels from the E99 v1 anchors through the same primary scoring path (`--score-novelty`); descriptive only, no tokens, never in a disposition.

Posted: https://github.com/Sino-Huang/NovPhy/issues/133#issuecomment-6388865221

## Decision P-coverage (2026-10-25): a GPU timing probe (≤ 1 GPU-h) to measure this child's smoke projection before the funding decision

**Trigger.** P-coverage funds a coverage arm "only if its smoke projection fits the remainder". This child's CPU-scaled projection is 141–149 GPU-h [INFERENCE] (fidelity 20.8–45.9; one native NovPhy run 23.7–27.7), which is too uncertain to decide funding against a remainder of roughly the same size. The program-set projections will be measured by their smokes this week.

**Decision.** Run one timing probe on a GPU in the night window: the first 1 000 updates of the PushT fidelity world model and of the native NovPhy model (and of the `rr128` configuration's model if its update differs), plus one CEM planning case; nothing is kept for any later run. Cap: 1 GPU-h of device time, charged to this child. The probe replaces the CPU-scaled projection with measured seconds per update; the funding decision (P-coverage) follows when the program-set projections are measured. No selection, training or fidelity run of this child starts before that decision.

## Decision P-coverage (2026-10-26): timing probe repeated on an exclusive GPU

The first timing probe (25 Oct) ran on GPU 3 beside Plan2Explore and B1 training (GPU util 99%), so its s/update is contended and cannot serve as the funding projection. The probe is repeated once, unchanged (same `--timing` mode, same update counts), as job `133-timing-clean` on GPU 3 tonight after #131's THICK G-fid runs and D1-m queue finish, with `shares_gpu: false` and no other job on GPU 3 while it runs (#133 first, then #134, sequentially). The contended record is kept as `timing-contended.json` (and `ledger/timing-contended.json`, so its GPU seconds stay counted); the clean probe rewrites `timing.json` and `projection.json`. Funding (`funded.json`) is decided afterwards under P-coverage on #127 from the clean projection against the program remainder; no NovPhy data beyond the probe's fit-split batches are read. Cost ≤ 1 GPU-h (#133) / ≤ 0.5 GPU-h (#134).

## Decision P-coverage (2026-10-26): T1 funded

Program decision https://github.com/Sino-Huang/NovPhy/issues/127#issuecomment-6412711831: clean projection 59.75 GPU-h within the child cap (76) and the remainder (≈ 181.7); funded.json written by the orchestrator.
