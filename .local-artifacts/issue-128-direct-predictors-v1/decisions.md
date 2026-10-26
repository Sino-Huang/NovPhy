
## Decision P-license (2026-10-24): A4 DynDQN — vendor facebookresearch/DynamicsAware @ eb65576 (Apache-2.0)

**Trigger.** A4 needs the official Dynamics-aware DQN code (hand-crafted trajectory-distance auxiliary loss) vendored before the port and G-port.

**Facts checked.**
- Upstream: https://github.com/facebookresearch/DynamicsAware, commit `eb65576d8a9e9b1f74c64518b5dd68af9d0a7eff` (the only commit, "Initial commit."; repo archived). Branches: `main` (+ `gh-pages`). README: "This branch trains models that use the hand crafted loss" — confirmed in code: `agents/neural_agent.py::sample_distance_loss` / `comparative_loss_single_batch` + `agents/rollout_dist.py::measure_distances`, driven by `agents/experiments/hand_crafted_experiments/06_hand_crafted_test_run_table_1.conf` (the paper's Table 1 run).
- License: root `LICENSE` = Apache License 2.0 (sha256 `50e6751797c50dedd75ef1b8a0d9e42f5f8472e9fbce91f34718e9f97b0c780a`), holder "Facebook, Inc. and its affiliates" (header of every vendored `.py`). Apache-2.0 permits use, modification and redistribution with the license and notices kept; no NC clause.
- The bundled PHYRE (`src/python`, phyre 0.2.2, Apache-2.0) is not vendored into NovPhy; it is built into the conda env `b127-dynaware` from the pinned clone (engine sources = upstream phyre; see the P-route decision).

**Options.** (a) vendor verbatim with VENDOR.json (allowed: Apache-2.0); (b) re-implement from the paper (only needed without a license) — rejected.

**Choice.** (a). Vendored to `world_model/baselines/dynamics_aware/vendor/` with upstream `LICENSE` and `VENDOR.json` (schema `issue_127_vendor_v1`, per-file upstream/vendored sha256 and every edit):
`agents/nets.py` (ResNetFilmAction: ResNet-18, FiLM before the last block, embeddor + distance regressor), `agents/neural_agent.py` (training loop, `sample_distance_loss`, `comparative_loss_single_batch`), `agents/rollout_dist.py` (trajectory distance), `agents/psim.py` (parallel PHYRE simulator for the per-batch rollouts), `agents/offline_agents.py` + `agents/train.py` + `agents/extended_evaluation.py` (agent wrapper, fold split, AUCCESS evaluation), `agents/neural_agent_contrastive.py` (imported by `offline_agents.py`; unused by the hand-crafted path), and the Table-1 config `06_hand_crafted_test_run_table_1.conf`. Upstream clone kept read-only at `/mnt/array/sukaih/novphy-baselines-src/DynamicsAware@eb65576/`.

**Reason.** Apache-2.0 allows vendoring; verbatim official code is what G-port and the Tier-T fidelity check need.

## Decision P-license (2026-10-24): A3 PhyQ-RelDQN — vendor phy-q/benchmark @ 514491c (MIT) model and training code

**Trigger.** Before the port, A3 needs the official Phy-Q relational DQN code vendored: `DQNRelationalSymbolic`, `DQNRelationalImage`, their DQN training step, the replay memory, the symbolic-state transform, the action mapping and the hyperparameters.

**Facts checked.**
- **Upstream:** https://github.com/phy-q/benchmark, commit `514491c5e77dbacdd61066d11139434f982e3d9e`. Read-only clone at `/mnt/array/sukaih/novphy-baselines-src/benchmark@514491c/`.
- **License:** root `LICENSE` is the MIT License, "Copyright (c) 2021 Cheng-Xue" (sha256 `ce90a0fac67404af0877eaa1c8370f07326d78784822891503102d4723b0bda7`). MIT permits use, modification and redistribution as long as the notice is kept.
- **Authorship of the vendored files:** all are Phy-Q's own Python, under `sciencebirdsagents/`. No file carries a different header.
- **Excluded content:** the bundled third-party JARs and the vendored RPIN port are not part of A3 and are not vendored. Their licenses are handled in #132.

**Options.**
- (a) Vendor verbatim with `VENDOR.json`. MIT allows this.
- (b) Re-implement from the paper. This is only needed without a license, so it is rejected.

**Choice: (a).**
- **Destination:** `world_model/baselines/phyq/vendor/`, with the upstream `LICENSE` and `VENDOR.json` (schema `issue_127_vendor_v1`, per-file upstream and vendored sha256).
- **Files:** the import closure of `Utils/Parameters.py`, `LearningAgents/DQNDiscreteAgent.py` and `TrainLearningAgent.py`:
  - networks: `LearningAgents/RLNetwork/{DQNBase, DQNSymbolicBase, DQNRelationalSymbolic, DQNRelationalImage}.py`, plus the networks that `Parameters.py` imports;
  - training: `LearningAgents/{Memory, MemoryDataset, DQNDiscreteAgent, LearningAgent, LearningAgentThread}.py` and `TrainLearningAgent.py`;
  - symbolic state: `StateReader/*` (the reader modules) and `Utils/{model, target_class}`;
  - the environment and client modules this closure imports.
- **Only edits:** imports made package-relative, plus the torch 1.8.1 → 2.13 compatibility edits, each listed with its reason in `VENDOR.json`.

**Reason.** MIT allows vendoring, and verbatim official code is what G-port and the native-task fidelity run need.

## Decision P-license (2026-10-24): A1/A2 DQN — vendor facebookresearch/phyre @ d0765dd (Apache-2.0) and torchvision v0.4.0 ResNet (BSD-3-Clause)

**Trigger.** A1 DQN-px and A2 DQN-car need the official PHYRE DQN network and training-step code vendored before the adapter and G-port.

**Facts checked.**
- Upstream: https://github.com/facebookresearch/phyre (archived), commit `d0765dd2f6d6cf41b933a3a57173c63d21fbfa6a` (HEAD, 2022-04-20, "Update installation instructions & build more wheels (#76)"). The sha written in #127/#128 (`d0765dd6f6d6…`) does not exist upstream (`git fetch` → "not our ref"); it is a one-character typo of this HEAD (same short sha `d0765dd`).
- License: root `LICENSE` = Apache License 2.0; every vendored `.py` carries the "Copyright (c) Facebook, Inc. and its affiliates. Licensed under the Apache License, Version 2.0" header. Apache-2.0 permits use, modification and redistribution with the license and notices kept; no NC clause.
- `agents/nets.py` builds the trunk with `torchvision.models.resnet18(pretrained=False)`. The `novphy` env has no torchvision, so the ResNet definition is vendored from torchvision **v0.4.0** (tag commit `d31eafa9c59cd59533dd0b89a36d371e55337a48`, Aug 2019, contemporaneous with PHYRE): `torchvision/models/resnet.py` and `torchvision/models/utils.py` verbatim, license BSD-3-Clause (Copyright (c) Soumith Chintala 2016), `LICENSE` vendored. The structure, parameter names and initialisation (kaiming-normal fan-out for convs, BN weight 1 / bias 0) are those of every torchvision release since; G-port checks the vendored trunk against the torchvision 0.28 trunk the official code builds in `b127-phyre`.
- PHYRE's simulator, task data, folds, evaluator and 100k cache are **not** vendored: they are built into the conda env `b127-phyre` from the pinned clone (G-port reference and the Tier-E fidelity run only).

**Options.** (a) vendor verbatim with VENDOR.json (allowed: Apache-2.0, BSD-3); (b) re-implement from the paper — rejected (licenses allow vendoring; verbatim code is what G-port needs); (c) install torchvision into `novphy` — rejected (shared env; no change to it).

**Choice.** (a). Vendored to `world_model/baselines/phyre_dqn/vendor/` with `VENDOR.json` (schema `issue_127_vendor_v1`; per-file upstream / vendored sha256 and every edit):
- `nets.py` (`ActionNetwork`, `FilmActionNetwork`, `ResNet18FilmAction`, `SimpleNetWithAction`); edits: `import torchvision` → `from . import torchvision`, `import phyre` → `from . import phyre_constants as phyre` (the only use is `phyre.NUM_COLORS` = 7; `phyre_constants.py` is a new one-line module copying `NUM_COLORS` from `src/python/phyre/creator/constants.py`).
- `neural_agent.py` (the official training loop: class-balanced sampler, `ce_loss` use, Adam + `CosineAnnealingLR(T_max=updates)` with `scheduler.step()` before each `optimizer.step()`, checkpointing; `eval_actions`); edits: `import nets` → `from . import nets`, `import phyre` → `from . import phyre_constants as phyre`. Vendored for provenance and so the NovPhy training step (`novphy.py`) can cite it line by line; its cache/simulator functions are not callable in `novphy` and are never called there. The official file runs unmodified from the clone in `b127-phyre` for G-port.
- `torchvision/models/resnet.py`, `torchvision/models/utils.py` (verbatim, no edits) + `torchvision/LICENSE` (BSD-3); new package markers `torchvision/__init__.py` and `torchvision/models/__init__.py` (re-export `resnet18`, `ResNet`, `BasicBlock`).
- `LICENSE` (PHYRE Apache-2.0, verbatim), `__init__.py` (new package marker).
Upstream clones kept read-only at `/mnt/array/sukaih/novphy-baselines-src/phyre@d0765dd/` and `/mnt/array/sukaih/novphy-baselines-src/torchvision@v0.4.0/`.

**Reason.** Both licenses allow vendoring with notices; verbatim official code is what the G-port and the Tier-E fidelity check need.

## Decision P-route (2026-10-24): A4 DynDQN — port the official DynamicsAware stack off Python 3.6 / CUDA 10.2 to a modern env, held to G-port

**Trigger.** The official stack (`env.yml`: Python 3.6, boost 1.67, thrift-cpp 0.11, pybind11 2.2.4; README: `conda install pytorch ... cudatoolkit=10.2`) cannot run on RTX 5090 (sm_120 needs CUDA ≥ 12.8); conda-forge no longer solves thrift-cpp 0.11 / Python 3.6.

**Options.** (a) port the official code to the current stack, minimal edits, G-port against the unmodified official code; (b) re-implement from the paper — rejected (P-route forbids an unofficial re-implementation when the official code can be ported).

**Choice.** (a).
- Env `b127-dynaware` = clone of A1's `b127-phyre` (Python 3.11, torch 2.13.0+cu130, numpy 2.4.6, thrift 0.11.0 C++ from source, upstream PHYRE `d0765dd2` built from source) + tensorboard. The DynamicsAware bundle's PHYRE (`src/`) is engine-identical to upstream `d0765dd2`: the C++/Python diffs are only the self-supervised-only `get_random_images` frame sampling, `std::vector`-vs-`py::bytes` pybind11 bindings (the upstream fix needed by modern pybind11) and boost-vs-std filesystem; the vendored Box2D tree equals upstream's submodule commit `33af62e` except a testbed `imgui.ini`. The hand-crafted path never uses `get_random_images`.
- Vendored code edits (all in `vendor/VENDOR.json`): package-relative imports; **E1** `nets.py`: torchvision → vendored torchvision v0.4.0 `resnet.py` (BSD-3; novphy has no torchvision), `phyre` → `phyre_constants.NUM_COLORS = 7`; **E2** `torch.load(..., weights_only=False)` (torch ≥ 2.6 default rejects the official checkpoint dict); **E3** SIGTERM stop: after the current optimizer step the official checkpoint dict is written atomically and `TrainingPaused` is raised (night-window resumability; resume = the official `ckpt.*` path); **E4** `np.long` → `np.int64` in `psim.py` (removed in NumPy 1.24; same dtype as the `c_long` buffers).
- G-port (CPU float32, `scripts/b127_dynamics_aware_port_reference.py`): the unmodified official clone (b127-dynaware) vs our adapter in the official configuration on two real PHYRE fold-0 tasks × 8 simulated actions — distances, forward incl. aux head, CE / aux / total loss, all 70 gradient tensors, parameters + BN buffers after two Adam + cosine updates and the LR schedule at 7 probe steps: **bitwise identical** (all deltas 0.0; `gport/a4/gport.json` pass).

**Reason.** Faithful official code on our hardware; every change is listed and the numerics of the hand-crafted path are shown unchanged.

## Decision P-fid (2026-10-24): A4 DynDQN G-fid Tier T band frozen before any run — [82.8, 87.2]

**Trigger.** #128 G-fid A4 Tier T: retrain the official hand-crafted recipe on PHYRE-B within-template folds 0–2 against the paper's 85.0; the SD was a GAP to be read from the paper and frozen before the run. No Tier-T run (and no GPU job of A4) has started.

**Source (quoted).** arXiv 2102.10336v2 (PDF sha256 `d0b5c830…8de282`), Table 1, row "Ours (Hand-crafted)": `85.0±1.1 38.6±8.4 74.0±1.5 23.3±8.8` (PHYRE-1B Within = 85.0±1.1). Caption: "Results reported are averaged over 10 test folds." The paper does not name the ± statistic. The official repo's summary for exactly this run — `agents/experiments/hand_crafted_experiments/06_hand_crafted_test_run_table_1.conf` (`seed 0,…,9`, `fmt EVALSETUP->MEAN±STD`) — is `agents/run_sweep_file.py`, which fills `STD` with `np.std` over the per-fold `results.json` scores. So **1.1 = the population SD of per-fold AUCCESS over the 10 test folds**.

**Band.** 85.0 ± 2 × 1.1 = **[82.8, 87.2]**, applied to the mean AUCCESS × 100 over folds 0, 1, 2 (one run per fold; `use_test_split=1`: train on train + dev, evaluate AUCCESS@100 on the test tasks ranking the first 10 000 cached actions = `results.json["target_metric"]`). ±2 SD of single folds is wider than ±2 SE of a three-fold mean (1.27) and also covers the systematic difference between the folds 0–2 mean and the 10-fold mean.

**Protocol.** The official `agents/train.py` entry with the exact `run_sweep_file.py` command for conf 06 at `ball_within_template` (100 000 updates × 512 = 64 tasks × 8, Adam 3e-4 + cosine, aux: 20 bins, cutoff 1.414, outer merging, linear 256-d embeddor, whole rollout; in-training evals every 5000 updates kept as official, logging only). Runner `scripts/b127_dynamics_aware_fidelity.py --fold k` (env b127-dynaware, GPU 3, scheduler jobs `128-gfid-a4-fold{0,1,2}`, resumable); `--table` writes `fidelity/a4/gfid.json`.

**Declared deviations (before the run).** (1) modern stack + vendor edits E1–E4 (Decision P-route; G-port bitwise identical); (2) upstream PHYRE `d0765dd2` engine (identical engine sources to the bundled copy); (3) one RTX 5090: the official launcher defaults to `--n-gpu 4` (DataParallel, BatchNorm per 128-sample shard) [INFERENCE: the GPU count is not stated in the paper]; here BatchNorm sees the full 512 batch; cuDNN is not bitwise deterministic; (4) torch init unseeded and sampler RandomState(42), as official.

**Repairs.** At most two logged repair attempts, each fixing a diagnosed port defect, never this band; otherwise A4 is dropped with `fidelity_not_reproduced` (P-fid).

`fidelity/a4/band.json` sha256 **`aba5a789931d25e9ffd6d0bb27144ee8fa07a31ff978af93fed350c9b7f2beea`**.

## Decision P-fid (2026-10-24): A1 Tier-E G-fid band frozen before the run — PHYRE-B within-template AUCCESS in [75.4, 79.8]

**Trigger.** A1's Tier-E fidelity check (shipped PHYRE DQN checkpoints + caches) must have its band written and recorded before any run.

**Frozen band** (`.local-artifacts/issue-128-direct-predictors-v1/fidelity/band.json`, `sha256:def37bb43436e786387d55966b675f11437fb8f7afa89510ff61c22f3ec7f426`):
- Benchmark: PHYRE-B (ball tier), within-template: eval setup `ball_within_template`, folds 0–9, `use_test_split=1` (train+dev → the fold's test tasks; 500 test tasks per fold).
- Metric: AUCCESS ×100 on each fold's test tasks exactly as the official `agents/train.py` computes it (`results['target_metric'] = Evaluator.compute_all_metrics()['independent_solved_by_aucs'][100]`); the arm's number is the 10-fold mean.
- Paper: Bakhtin et al. 2019 (arXiv 1908.05656) Table 1a, DQN, PHYRE-B within: **77.6 ± 1.1** (SD over 10 folds). **Band [75.4, 79.8]** = ±2 SD. Pass iff the 10-fold mean lies in the band.
- Procedure: the unmodified official pipeline from the pinned clone (facebookresearch/phyre @ `d0765dd2f6d6cf41b933a3a57173c63d21fbfa6a`, simulator built from source in `b127-phyre`): `agents/train.py --use-test-split=1 --eval-setup-name=ball_within_template --fold-id=<k> --agent-type=dqn --dqn-load-from=<shipped dqn_10k ball_within_template seed k> --dqn-rank-size=10000 --dqn-eval-batch-size=128 --dqn-finetune-iterations=0 --dqn-refine-iterations=0 --max-test-attempts-per-task=100` → `DQNAgent.real_eval` ranks the first 10 000 actions of the shipped 100k ball cache and reads their statuses from the cache. One GPU (`real_eval` calls `model.cuda()`), via the night-window scheduler (job `128-gfid-a1-tier-e`, GPU 0). Only environment change: `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (torch ≥ 2.6 defaults `torch.load` to weights_only; the 2020 checkpoints pickle the numpy sampler state).
- Inputs (sha256 in band.json and `data/external/phyre_dqn/MANIFEST.json`): the 10 shipped `ckpt.00100000` + `results.json` (agents/download_dqn_ckps.sh URLs, all still served, HTTP 200) and the 100k ball simulation cache. The checkpoints' `model_kwargs` are `{resnet18, action_space_dim 3, fusion last, action_hidden_size 256, action_layers 1}`.
- Diagnostic only (not the band): the shipped `results.json` record the authors' own AUCCESS@100 for these checkpoints (10-fold mean 77.63); ours is reported next to it per fold to diagnose build / platform drift.

**Options.** (a) the paper's ±2 SD band on the shipped checkpoints (#127/#128 frozen text); (b) a band around the shipped results.json values — rejected (the ticket fixes the paper's number).

**Choice / reason.** (a), as #128 specifies. At most two logged repair attempts, each fixing a diagnosed port defect, never the band; then drop with `fidelity_not_reproduced`.

## Decision P-other (2026-10-24): A4 DynDQN — mapping the official hand-crafted trajectory distance onto NovPhy fit-shot engine trajectories

**Trigger.** A4 = A1 + the paper's hand-crafted auxiliary loss. The official loss (`neural_agent.comparative_loss_single_batch` + `rollout_dist.measure_distances`, Eq. 7–8 and App. A of arXiv 2102.10336) is defined on PHYRE featurized objects (stride 60, ≤ 17 frames, unit-square coordinates, α = 1.414). NovPhy needs an equivalent trajectory source and a rule for things PHYRE never has. This governs the A4 NovPhy training (Phase 1); nothing has been trained.

**Official definition (code, Table-1 config).** For two actions on the same task: T = joint valid prefix (min of the two rollout lengths, `time_slice 0` = whole rollout); O = objects with a dynamic colour (red/green/blue/gray) at frame 0 of rollout_1; d(o,t) = Euclidean distance of centres, clipped at α; label = Σ d / (|O|·T·α) (= 1 − v of the paper), binned `round(x·(K−1))`, K = 20; CE on all 8×8 ordered pairs (incl. self pairs) of each task's 8 samples, mean over the batch; outer-product merge of the linear 256-d projections; loss = BCE + 1.0 × aux.

**Choice (implemented in `world_model/baselines/dynamics_aware/novphy.py`; the pair loss and distance core are G-port-bitwise equal to the official code in PHYRE form; toy-pair tests in `tests/test_issue_128_dynamics_aware.py`).**
1. *Source*: the #123 fit probe labels (`.local-artifacts/issue-123-hjepa-comparison-v1/labels/fit/labels-*.pt`, engine slot presence [P,18] and engine-projected centres [P,18,2], `rc.build_label_shard`), fit split only. The native traces are not needed: the labels already carry every slot at every 15th position.
2. *Time grid*: positions 0, 15, 30, … ≤ shot end (≤ 285; up to 20 frames). The clamped non-multiple end position is dropped so that equal indices are equal times, as on PHYRE's stride grid; pairs use the joint valid prefix (official T = min(t1, t2)).
3. *Objects*: O = dynamic slots (`bird:*`, `block:*`, `pig:*`) present at position 0 of rollout_1 (official: dynamic colours at frame 0 of rollout_1). Platforms, slingshot and terrain are static (PHYRE's black/purple). The bird is the action object, as PHYRE's red ball.
4. *Coordinates and α*: isotropic screen units (x/W, y/W with the 640×480 viewport; the labels are x/W, y/H) and α = the screen diagonal √(1 + 0.75²) = **1.25** — the official rationale for α = √2 is "the maximum possible distance" in the unit-square PHYRE scene.
5. *Destroyed objects* (NovPhy pigs/blocks/birds disappear; PHYRE objects never do): an object present in exactly one of the two rollouts at t contributes α (maximally dissimilar), absent in both contributes 0 (identical); the (o, t) term always counts in the normaliser. Alternatives rejected: masking absent objects (would hide exactly the pig-removal differences the success label depends on) and freezing at the last position (a destroyed pig would look identical to an intact one).
6. *Everything else official*: K = 20 bins, outer merge, linear 512→256 embeddor, weight 1.0, BCE on the engine verdict, 64 tasks × 8 shots per batch (4 failures + 4 successes per level; a level without successes uses failures, as official; a level without failures uses successes — NovPhy has all-success levels, the official sampler would crash), Adam 3e-4 with cosine, scheduler stepped before each update; action MLP 1 × 256 on `eb.launch4` (the official default, as A1), FiLM before the last ResNet-18 block; input = the 320×240 RGB level-anchor frame /255 through the official stem with 3 input channels (A1's convention).

**Reason.** Validity first: the mapping preserves the official metric's definition and normalisation and only adds rules for cases PHYRE cannot produce, each chosen to keep the metric's meaning (rollout similarity); fairness: the native loss, head and optimiser are kept.

## Decision P-cap (2026-10-24): A4 DynDQN Tier-T fidelity projects above the #127 estimate — fund from unallocated program budget, no cut

**Trigger.** #127 estimated A4 Tier-T fidelity at ~6–12 GPU-h [INFERENCE]; child A's fidelity share is ~8–14 GPU-h. The CPU probe (`fidelity/a4/probe-cpu.json`, `scripts/b127_dynamics_aware_fidelity.py --probe-cpu`) measures the official synchronous update: PHYRE rollouts for the 512 actions on 20 CPU workers 0.072 s + aux distance labels 0.055 s per update, and 7.57 TFLOP per update (FlopCounter, forward + backward, batch 512 at 256×256). At an assumed 25–50 TFLOP/s effective on one RTX 5090 that is 0.28–0.43 s per update → **7.7–11.9 GPU-h per fold, 23–36 GPU-h for folds 0–2**, plus 21 in-training evals and the final test eval per fold. A GPU timing probe (`128-gfid-a4-probe`, 600 updates, runs first tonight on GPU 3) replaces the assumed throughput with a measurement (`fidelity/a4/probe-gpu.json`).

**Options (P-cap order).** (1) draw the overrun from unallocated program budget; (2) cut matched-budget secondary runs / selection grid / secondary arms — not applicable to a fidelity check; never cut the native recipe (fewer updates, fewer folds or mixed precision would weaken the check of a primary arm).

**Choice.** (1): run the native recipe on folds 0–2 as frozen in the band decision; the overrun (≈ 11–24 GPU-h above the child's fidelity share) is drawn from unallocated program budget. Each fold is a resumable scheduler job on GPU 3 (`128-gfid-a4-fold{0,1,2}`, `est_minutes` 600 each); at one fold per ~10 h window GPU 3 needs about three nights. The child's 48 h-on-4-GPUs projection is re-checked by the child plan with the measured probe.

**Reason.** P-cap forbids weakening a primary arm's native budget; the program budget (400 GPU-h) has room.

## Decision P-other (2026-10-24): A1/A2 adapter choices (PHYRE DQN recipe on the NovPhy fit shots), fixed before any training

**Trigger.** #128 Phase 0: the PHYRE DQN recipe needs NovPhy-specific input, data and schedule choices before the A1/A2 adapter is frozen and trained. Everything below is implemented in `world_model/baselines/phyre_dqn/novphy.py` (vendored official code in `vendor/`, see the P-license decision) and passed G-port (`gport/gport.json`: A1 and A2 losses, gradients, LRs, parameters, BN buffers and Adam moments identical to the official code over three updates; native and matched LR schedules identical at every update).

**Facts read from the official code (facebookresearch/phyre @ d0765dd2, `agents/`).** `run_experiment.py` `DQN_BASE_ARGS` (the `dqn_10k` agent of the paper): `--dqn-updates=100000 --dqn-train-batch-size=64 --dqn-learning-rate=3e-4 --dqn-cosine-scheduler=1 --dqn-balance-classes=1 --dqn-rank-size=10000`; `offline_agents.py` defaults `--dqn-action-hidden-size=256 --dqn-action-layers=1 --dqn-fusion-place=last --dqn-network-type=resnet18`; `neural_agent.train`: `Adam(model.parameters(), lr)`, `CosineAnnealingLR(T_max=updates)` with `scheduler.step()` **before** every `optimizer.step()`, class-balanced batches (`train_batch_size // 2` positives and the rest negatives, sampled with replacement by `np.random.RandomState(42)`), `ce_loss` = `binary_cross_entropy_with_logits` on the binary outcome, no weight decay, no gradient clipping, no augmentation, the final model is used (no early stopping); `eval_actions`: eval mode, the observation encoded once, one logit per action. The shipped checkpoints' `model_kwargs` and every shipped `results.json` confirm `action_hidden_size=256`.

**Choices.**
1. **Action MLP width = 256 (official), not 512.** #127/#128 write "action MLP 1 × 512 ReLU", following the paper's text (§4.1: "a single hidden layer with 512 units"). The released code, its `dqn_10k` arguments and the shipped checkpoints that produced the paper's 77.6 (our Tier-E target) all use 256. "Native PHYRE DQN … exactly as official" → 256 (one hidden layer, ReLU, output 2 × 256 = FiLM gain and bias on the 256 channels entering `layer4`). Options: (a) 256 = the code that produced the number; (b) 512 = the paper text. Chose (a); disclosed as a paper/code discrepancy. A1 then has **11 309 889** trainable parameters (RGB stem).
2. **A1 input: the 320 × 240 RGB decision frame at native resolution, uint8 / 255, no mean/std normalisation, no resize.** The official input is PHYRE's own 256 × 256 observation grid as a 7-channel one-hot; NovPhy frames are RGB, not palette-indexed, and #127 fixes "320 × 240 RGB". The only network change is the stem convolution: `Conv2d(3, 64, 7, stride 2, padding 3, bias=False)` in place of `Conv2d(7, 64, …)` (same geometry, PyTorch default init as the official stem); the rest of ResNet-18 is resolution-agnostic (adaptive max-pool), giving layer3 15 × 20 → FiLM → layer4 8 × 10 → max-pool → `Linear(512, 1)`. Scaling to [0, 1] keeps the official input range (one-hot ∈ {0, 1}); ImageNet statistics are meaningless without pretraining (the official trunk is `pretrained=False`). Resizing to 256 × 256 was rejected (distorts the aspect ratio for no benefit).
3. **Action vector: `eb.launch4` = [drag_x/480, drag_y/480, release_ms/1000, tap_ms/1000]** (#127), 4-d in place of PHYRE-B's 3-d (x, y, radius) ∈ [0, 1]³; no further normalisation.
4. **Training examples: one per fit shot**, (level anchor, launch4, engine verdict): the 13 919 complete #109 fit shots of `eb.FitShots()` (13 920 scheduled; `issue-109-sc-fit-type010105-049-c15` is the typed failure excluded by `rc.FitData`), 480 levels. The observation of every shot is its level's anchor — frame 0 of the anchor branch (`eb.fit_anchor_frames()`, A1) or the anchor carrier (`eb.fit_anchors()["carriers"]`, A2) — the same input the arm is scored on, and the PHYRE structure (one initial scene per task, many action/outcome pairs). Implemented as the official `TrainData` layout (`task_indices` into 480 observations).
5. **Loss and batches as official:** BCE with logits on the engine verdict (pig removal), class-balanced batches of 64 (32 positive / 32 negative, with replacement), the official sampler with `np.random.RandomState(seed)`.
6. **Optimizer as official:** Adam lr 3e-4 (default betas/eps), cosine annealing to 0 over the run, scheduler stepped before each update; no weight decay, clipping or augmentation (none in the official recipe); the final model is scored.
7. **Updates.** A1 primary (native): **100 000 updates × batch 64** (`DQN_BASE_ARGS`), cap 2 GPU-h per seed (#127). On 13 919 examples this is ≈ 460 passes (PHYRE trained on ~10⁸ cached outcomes per fold); recorded here, not changed — the native schedule is what #127 calls native, and any shorter schedule is one of the ≤ 3 pre-declared configurations of the frozen plan, not an adapter choice. A2 primary (matched): **9 000 × 64** (U's B1 recipe). A2 secondary (native-size): native schedule, 100 000 × 64.
8. **A2 architecture.** The official `preprocess` / `_forward` / `ce_loss` and `FilmActionNetwork` (inherited unchanged) over an MLP trunk on the 236-d carrier viewed as a [B, 236, 1, 1] tensor: stem 1 × 1 conv → BN → ReLU, then the ResNet-18 stage ladder (widths b, 2b, 4b, 8b; two residual blocks per stage; torchvision BasicBlock with both 3 × 3 convolutions replaced by 1 × 1; projection shortcuts; torchvision init), FiLM (1 × 256 ReLU action MLP → gain/bias on 4b channels) before the last stage, max-pool (identity on 1 × 1), `Linear(8b, 1)`. One width knob: **matched b = 70 → 1 838 781 trainable parameters** (U 1 837 690, +0.06 %; every trainable module incl. the action MLP, FiLM output layer and head); **native-size b = 179 → 11 333 771** (A1 11 309 889, +0.21 %).
9. **Seeds:** `torch.manual_seed(seed)` before the model is built and `np.random.RandomState(seed)` for the sampler, seed ∈ {20260908, 20260909, 20260910} (the official run uses the fixed 42 for the sampler and the default torch seed).
10. **Scoring:** eval mode (BatchNorm running statistics), the anchor encoded once, the success logit of each grid candidate (`eval_actions`); score = logit, ranking cost = −logit (#128).
11. **Correction to the P-license decision:** vendored `neural_agent.py` keeps `import phyre` (its module-level `TrainData` alias evaluates `phyre.ActionSimulator`), so its only edit is `import nets` → `from . import nets`; it is vendored for provenance (the official file runs from the clone for G-port) and is not imported in `novphy`. `VENDOR.json` lists exactly these edits.

**Reason.** Each choice keeps the official network, fusion, loss, sampler, optimizer, schedule and scoring verbatim and changes only what the NovPhy input forces (#127: RGB 320 × 240 frame or E99 carrier, launch4, fit shots); where paper and code disagree, the code that produced the paper's number wins.

## Decision P-other (2026-10-24): #128 Phase-1 selection design (≤ 3 configurations per arm), native budgets, child budget projection — fixed before any selection run

**Trigger.** #128 Phase 1 (Q1): the runner `scripts/run_issue_128_direct_predictors.py` trains the pre-declared configurations of every arm at seed 20260908, scores them on the policy split only (pooled grid AUC of the primary score, `eb.select_configuration`: largest AUC at 4 d.p., ties → fewer parameters → declaration order), and reuses the selected configuration's seed-20260908 run as main seed 20260908. This decision fixes the configurations, budgets and projection before the first selection run. It is outcome-blind: no policy-split or evaluation-split number exists for any arm.

**Data-scale gap (why a training-length axis).** The official update counts were tuned for PHYRE's cached outcomes: about 2 000 train+dev tasks × 100 000 actions ≈ 10⁸ (task, action) pairs per fold, so 100 000 × 64 updates are a fraction of one epoch there. Our fit split has **13 919 examples** (480 levels × ≤ 29 shots). At the native schedule that is 100 000 × 64 / 13 919 ≈ **460 epochs** for A1 / A2-native and 100 000 × 512 / 13 919 ≈ **3 700 epochs** for A4. Because the schedule is cosine to 0 over the run, the update count is the recipe's only regularisation-like knob: the official recipe has no weight decay, no augmentation and no early stopping. A reviewer can fairly ask whether the native length over-fits our much smaller set. Training length is therefore the equal-tuning axis for every arm whose budget is "native". Learning rate is the axis where the budget is fixed by definition (A2 matched = U's B1 recipe). The native configuration is always in the set and is declared first.

**Configurations (seed 20260908; each configuration has its own cosine schedule over its own run; everything else official):**

| arm | role | budget | configurations (declaration order) |
| --- | --- | --- | --- |
| A1 DQN-px | primary | native: 100 000 updates × batch 64, Adam, cosine to 0 | `native` 100 000 updates, lr 3e-4 · `tenth` 10 000, lr 3e-4 · `hundredth` 1 000, lr 3e-4 |
| A2 DQN-car | primary | matched: 9 000 × 64 (U's B1 recipe), 1 838 781 parameters | `lr3e-4` (official) · `lr1e-4` · `lr1e-3`, all at 9 000 updates |
| A2-native | secondary | native schedule, A1-size MLP (11 333 771 parameters) | `native` 100 000 · `tenth` 10 000 · `hundredth` 1 000, lr 3e-4 |
| A4 DynDQN | primary | native Table-1 recipe: 100 000 updates × batch 512 (64 levels × 8 shots), aux weight 1 | `native` 100 000 · `tenth` 10 000 · `hundredth` 1 000, lr 3e-4 |
| A3-sym / A3-img | secondary, never program set | native Phy-Q (AdamW 3e-4, batch 32); adapter being built by PhyqA3 | posted in a separate decision before A3's selection |

The factor-10 spacing spans ≈ 460 / 46 / 4.6 epochs (A1, A2-native) and ≈ 3 700 / 370 / 37 epochs (A4). Training length was chosen over a learning-rate grid for the native-budget arms because the over-fitting risk comes from the data-scale gap, not from the optimiser. Three configurations equals U's tuning allowance (≤ 3, #127).

**Selection is on the policy split only; the evaluation split is never read before the readout freeze.** Primary score = the success logit, cost = −logit (`eb.score_rows`, key `neg_score`). The selection readouts also record G-rank (480 fit anchors × 16 grid launches) and MACs per candidate. A typed failure (nonfinite loss) has no AUC and is never selected.

**Other run rules.**
- Batch b of every run is drawn from `eb.batch_rng(seed, b)` = PCG64([seed, b]) with the official sampler rule (A1/A2: 32 positive / 32 negative with replacement; A4: 64 levels × 4 negative + 4 positive). The official code instead draws batches from one sequential `RandomState`. The rule is identical; the stream is per batch so that a run resumes exactly after a SIGTERM (checked on CPU: a resumed run's parameters equal the uninterrupted run's).
- `torch.manual_seed(seed)` before the model is built. Main seeds 20260909 / 20260910 train the selected configuration.
- Precision: PyTorch defaults (fp32; cuDNN may use TF32 for convolutions), `cudnn.benchmark` on. Recorded in every `report.json`.

**Child budget projection** (child cap 66 GPU-h = the upper end of A's share, incl. fidelity). These are FLOP-based estimates; `--smoke` replaces them with the measured selection-run throughput before the plan freeze.
- A1: one native update ≈ 0.53 TFLOP (ResNet-18 at 320×240, ×3 for backward, batch 64), ≈ 15–25 ms on an RTX 5090, so the native configuration is ≈ 0.4–0.7 GPU-h/seed (cap 2). Selection (1.11 × native) plus 2 main seeds ≤ 2.2 GPU-h.
- A2: < 0.1 GPU-h total. A2-native: ≈ 0.1–0.3 GPU-h/seed native.
- A4: DynA4 projects 5–10 GPU-h/seed for the native configuration (no simulator in our version; ≈ 8.9 TFLOP/update incl. backward), above A4's 3 GPU-h/seed cap. Selection costs ≈ 1.11 × native ≈ 5.6–11 GPU-h. The 2 main seeds cost 0.1–1 GPU-h (hundredth/tenth) or 10–20 GPU-h (native).
- A3 (both variants, selection + seeds): ≤ 10 GPU-h, to be fixed in its own decision. Readouts + scoring ≈ 2 GPU-h.
- Fidelity: A1 Tier E spent 0.24 GPU-h (measured). A4 Tier T 23–36 GPU-h (DynA4's P-cap decision, issuecomment-6382423910, already drawn from unallocated program budget).
- Total: ≈ 45–55 GPU-h if A4 selects a shortened configuration, up to ≈ 85 GPU-h if A4 selects `native` and its per-seed cost sits at the top of the range. Over the 66 GPU-h child cap, rule P-cap step 1 applies (draw from the unallocated program budget); A4's native budget and its 3 seeds are never cut. **A P-cap decision with the measured smoke numbers will be posted before A4's selection runs.** Wall: the remaining main runs fit well inside 48 h on 4 GPUs (LPT queues in the plan).
- Order of GPU work: A1 / A2 / A2-native selection and selection readouts tonight on GPU 0. A4 selection after its Tier-T G-fid (#127: "A4 (Tier-T G-fid first)") and after the P-cap decision. A3 after its adapter, G-port and selection decision. Then smoke, plan freeze, main seeds, readouts, freeze, a single scoring pass (A3 scored later, with `--score --arms A3-sym,A3-img` and a publication addendum that re-scores nothing).

**Reason.** The tuning allowance is the same size as U's (≤ 3 configurations) and is chosen on the policy split. It targets the one place a faithful port is most likely to be unfair to the baseline (an update count set for ~10⁸ examples, applied to 14 k), and it keeps the native recipe as a candidate so that the official setting is always tried.

## Decision P-cap (2026-10-24): A4 Tier-T fidelity — measured projection 40.3 GPU-h (folds 0–2), funded from unallocated program budget

**Trigger.** Follow-up of the A4 P-cap decision above (CPU projection 23–36 GPU-h). The GPU timing probe ran before any fold (`128-gfid-a4-probe`, exit 0; official fold-0 recipe shortened to 600 updates in a scratch dir, never a fidelity result): `fidelity/a4/probe-gpu.json` — RTX 5090, **2.22 updates/s** (0.45 s per update incl. the 20-worker PHYRE rollouts and the aux labels), in-training eval 154 s (21 per fold), final test eval 89 s, peak GPU memory 18.2 GB.

**Projection.** 100 000 / 2.22 s + 21 × 154 s + 89 s = **13.4 GPU-h per fold, 40.3 GPU-h for folds 0–2** (vs the #127 estimate of ~6–12 and child A's fidelity share ~8–14). 18.2 GB rules out `shares_gpu`; on GPU 3 alone the three resumable folds need about 3.5 night windows (projected end 2026-10-27/28).

**Choice.** Unchanged native recipe and protocol (as frozen in the band decision, in-training evals kept as official); the ≈ 26–32 GPU-h overrun is drawn from unallocated program budget (P-cap step 1; the native budget of a primary arm is never cut). The orchestrator may move folds 1–2 to another free GPU (same command, `CUDA_VISIBLE_DEVICES` = that GPU) to shorten the calendar time; the result does not depend on the GPU index.

## Decision P-other (2026-10-24): A3 PhyQ-RelDQN — state source: Phy-Q's symbolic state rebuilt from our engine's ground-truth JSON (A3-sym); the 320 × 240 frame through the official image transform (A3-img)

**Trigger.** A3-sym needs Phy-Q's symbolic state of the decision state, and A3-img needs the official image input. This decision governs the A3 data builder and all A3 training. Nothing has been trained.

**Facts.**
- **Phy-Q's symbolic input** is a 12 × 120 × 160 binary occupancy image: `DQNSymbolicBase.transform_full` → `SymbolicStateReader.get_symbolic_image`, the dense path that `TrainLearningAgent.py --if_save_local True` stores. It is built from the game's non-dev GT JSON:
  - each object's first contour gives a bounding box, which fills cells;
  - the object type comes from a linear colour classifier (`Utils/model`) applied to the object's colour map;
  - Slingshot is labelled; Ground is skipped.
- **Our captures keep exactly this at frame 0 of every shot.** `decision-physics-v1.json` (fixed step 30 000 = `m99.DECISION_FIXED_STEP`) is written by `PhysicalSnapshotExporter.Capture`, which parses the engine's own `SymbolicGameState.GetGTJson(useNoise=false)`:
  - one node per GT feature;
  - `screen_polygon` = the feature's first contour (the reader only uses `contours[0]`);
  - `object_type` = the dev label = the sprite name.
- **The colour map is recoverable.** The non-dev colour map is the deterministic `ColorDataLookUp(sprite)` (no noise), i.e. the sprite's entry in the engine's `StreamingAssets/ColorCodes/BaseMap.json`. Platforms use `effects_21` (the sprite of `Platform.prefab`).
- **Our BaseMap.json matches Phy-Q's.** Every fit and policy attempt uses one file (`sha256:8bc9c25b…`). It contains all 256 sprites of Phy-Q's own BaseMap.json (`sciencebirdsgames/Linux.zip`) with identical values, plus 148 NovPhy-only sprites.
- **The official classifier works on our objects.** The 600 fit/policy anchors contain only standard objects (red bird, small pig, wood/ice/stone blocks, platforms, slingshot, ground). On the rebuilt JSON, the official classifier recovers the true material of **5 222 / 5 222** objects, and the reader runs without error on every anchor.

**Options.**
- (a) Rebuild the non-dev GT JSON from the capture and run the vendored reader verbatim, classifier included.
- (b) Use oracle types instead of the classifier. Unnecessary, since (a) is exact on these objects.
- (c) Feed the E99 carrier slots through the object pathway. This is a different representation and not faithful to Phy-Q.

**Choice: (a)**, implemented as `novphy.gt_json` + `novphy.symbolic_state`.
- **Stores:** `a3/data/symbolic-{fit,policy}.pt` hold the 480 fit and 120 policy anchors. Evaluation anchors are read the same way, but only in `--score`, after the freeze.
- **Only runtime change:** the removed `np.float` alias is set around the reader (NumPy ≥ 1.24; VENDOR.json compatibility entry).
- **Disclosed differences from a live Phy-Q client:**
  - feature order and ids differ (the occupancy image does not depend on them);
  - colour-map percents are float32, as in C#.
- **G-port:** the official `transform` on the same JSON is bit-identical to the adapter's.

**A3-img.**
- **Screenshot layout:** the 320 × 240 RGB decision frame (`eb.*_anchor_frames`) is replicated 2× to the engine's 640 × 480 screen and put in BGR order. This is the `[3, 480, 640]` array that `agent_client.read_image_from_stream` returns.
- **Official transform:** `DQNRelationalImage.transform` is then applied, re-expressed on PIL:
  - `transpose(2, 1, 0)` (the scene ends up transposed, as in Phy-Q);
  - `CenterCrop((360, 480))`, which keeps screen x 140–500, so the slingshot falls outside the crop, as it does in Phy-Q;
  - `Resize((120, 160))`, `ToTensor`, `Normalize(0.5, 0.5)`.
- **G-port:** the PIL version is bit-identical to torchvision 0.28.
- **Diagnostic:** on 8 fit anchors, the final input differs from the one computed on the native 640 × 480 PNG of frame 0 by a mean |Δ| of 0.50 grey levels (0–255 scale).

**Reason.** Validity first: this is the input Phy-Q's own pipeline would compute from our engine. Nothing is learned or approximated, apart from the 2× pixel replication of the image.

Posted: https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382972582

## Decision P-other (2026-10-24): A3 PhyQ-RelDQN — action mapping: each launch → the nearest native action of Phy-Q's 180-way angle head

**Trigger.** #127 requires that "Phy-Q's relative-to-slingshot action grid is mapped onto our launch parameters". This decision governs the A3 training targets and scoring.

**Facts.**
- **Phy-Q's action space** (`outputs=180`):
  - `DQNDiscreteAgent.__degToShot` maps action a to angle a + 90 deg and a release offset of (200 cos, 200 sin) px from the slingshot centre;
  - `SBEnvironmentWrapper.step` applies it as (sling.X + dx, sling.Y − dy), i.e. screen y up;
  - red bird: tap 0; `shoot(x, y, t1=0, t2=tap)`.
  - So Phy-Q's action is an angle only, always at a 200 px drag.
- **Our launches** release at (drag_x, drag_y) px from the slingshot with screen y down (`game_y = start_y − dy`), with release time 1000 ms and tap 0.
- **Power is saturated in all fit data.**
  - `ABBird.DragBird` clamps the drag to `_dragRadius` = 1 world unit (all bird prefabs). That is ≈ 18.3 px on the 640 × 480 screen at orthographic size 13.125, and launch speed = min(|d|, R)/R × 10.
  - All 29 fit candidates have |d| ≥ 40.3 px, so the fit split carries no power variation.
  - Empirical check: the same-direction pair c00 (−80, 10) and c13 (−40, 5) agrees on 475 / 480 fit verdicts.
  - Sub-saturation drags (≤ 17 px) exist only in the evaluation power inventory.

**Options.**
- (a) Keep the native 180-way head and map each launch to the nearest native action: a = round(atan2(−drag_y, drag_x) in degrees, mod 360) − 90.
- (b) Extend the head over power (angle × power bins). Its power outputs can never be trained from the fit split, since it contains no sub-saturation launch.
- (c) Use a discrete head over the 29 fit launches. It cannot score the offset/power inventories and departs from Phy-Q's head.

**Choice: (a).**
- **Grid** → actions [97, 127, 144, 153, 94, 111, 125, 135, 92, 104, 115, 124, 92, 101, 109, 117]: **15 distinct Q outputs**.
  - g08 (−120, 5) and g12 (−160, 5) are 0.6° apart and share action 92. They agree on 480 / 480 fit verdicts (neither ever passes).
  - G-rank consequence: one tied pair out of 120 per level (0.83 % against the 50 % limit) and no fully tied level, so G-rank cannot fail by construction (unit-tested with `eb.g_rank`).
- **Angle (13) and offset (11) inventories** map injectively.
- **Power inventory:** same-direction launches with different drags land on the same or neighbouring actions; they differ only through rounding of their small integer drags. Its descriptive AUC therefore reflects the native head's lack of power; this is disclosed.
- **Not mapped:** release time (1000 ms vs Phy-Q's 0) and tap (0; red birds have no ability) are the same for every candidate in every inventory.
- **Out of range:** rightward pulls (not in #109) raise an error.

**Reason.** This keeps Phy-Q's head and loss exactly. The only thing lost is a dimension the fit split cannot teach.

Posted: https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382972743

## Decision P-other (2026-10-24): A3 PhyQ-RelDQN — upstream training defects: the is_done long-index bug is not on the official path; the image agents' uint8 memmap cast is fixed (disclosed)

**Trigger.** An indexing bug in Phy-Q's Huber Q-regression was reported (EngineE1 via PhyreA1). Reading the training code also turned up a second defect, on the path the official script actually runs.

**Defect 1: Huber target in `DQNBase.train_model_memory`.**
- **The bug:** `is_done_batch = torch.Tensor(batch.is_done).long(); next_state_values[is_done_batch] = 0` indexes rows by the integer values 0/1. It zeroes rows 0 and 1, not the terminal rows, so every other terminal row keeps its bootstrap γ·max_a Q_target(s′).
- **Effect, measured with the official code** (toy `DQNBase`, 8 terminal transitions; `gport/a3/gport.json` → `upstream_bug.toy_demo`):
  - `train_model_memory` bootstraps **7 of 8** targets (only row 1 is zeroed); batch loss **0.333** against **0.175** for the true terminal targets;
  - `train_model_loader` on the same data gives the correct targets (0 rows bootstrapped).
- **Which path the published results used:**
  - `TrainLearningAgent.sh` calls `TrainLearningAgent.py` without `--if_save_local`, whose default is `True`.
  - That runs `MemoryDatasetMemmap` + `train_model_loader`, where `is_done` arrives as a bool tensor (pandas bool column → `default_collate`), so the mask is correct.
  - `train_model_memory` runs only with `--if_save_local False`. For image agents it would also crash (`Tensor.transpose(2, 1, 0)` on the already-transformed state) [observed in source, not executed].
  - So the published numbers did not use the bugged line [INFERENCE from the released scripts].
- **Choice:**
  - G-port reproduces the official `train_model_loader` as is (`gport/a3/gport.json`: bitwise identical).
  - NovPhy training uses the same loader-path semantics. Every fit shot is terminal (one bird per level; 'passing' reward = int(did_win) = engine pig removal), so the target is the verdict.
  - This is official behaviour, not a fix.

**Defect 2: image inputs on the official path.**
- **The bug:**
  - `AgentThread.save_local` stores `network.transform(s)`, which for `DQNRelationalImage` is a float tensor in [−1, 1].
  - `MemoryDatasetMemmap` writes it into a `dtype='uint8'` memmap. The unsafe cast maps −1 → 255, 1 → 1 and everything else → 0 (verified under numpy 2.4).
  - The official image agents therefore train on a near-empty saturation map, while acting (`select_action`) on the clean transform.
  - Symbolic agents are unaffected (their values are 0/1).
- **Options:**
  - (a) Replicate the defect: train on the uint8 cast and score on the clean transform, a train/test mismatch by construction.
  - (b) Train and score on the clean official transform, as a disclosed fix.
- **Choice: (b).**
  - A skeptical reviewer would call (a) a deliberately broken baseline: the defect corrupts the model's input, not the method.
  - The published Relational-Image number (local gen. 0.2393) was presumably produced with the defect [INFERENCE]. This is disclosed next to any A3-img result, and the Phase-3 native-task fidelity run (P-engine) is where it would show.

**Reason.**
- **Validity first:** G-port compares against the released code exactly as it is.
- **Fairness in training:** the arm uses the path the released script actually runs, where defect 1 is absent, and does not inherit a data-pipeline corruption (defect 2) that breaks the agent's own train/act consistency.

Posted: https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382972952

## Decision P-other (2026-10-24): A3 PhyQ-RelDQN — native recipe on the fit shots, 3 pre-declared configurations, projected GPU time

**Trigger.** A3's training recipe and its ≤ 3 selection configurations must be fixed before selection and training start.

**Official recipe (from the code).**
- `Utils/Parameters.py`: h 120, w 160, 180 outputs, train_batch 32, lr 3e-4, gamma 0.99, reward 'passing'.
- `TrainLearningAgent.py`:
  - `optimizer = optim.AdamW` with torch defaults (betas 0.9/0.999, eps 1e-8, weight decay 0.01);
  - N = len(training levels) // 5 outer steps (line 219);
  - per outer step, `train_time_per_ep = max(1, previous − step)` epochs (line 224): 32, 31, 29, 26, 22, 17, 11, 4, then 1;
  - the target net is re-copied from the online net at each outer step.
- `train_model_loader`:
  - builds a new AdamW per call, so Adam moments start fresh at every outer step;
  - draws class-balanced batches with `WeightedRandomSampler(num_samples=len(D))` and `drop_last`;
  - uses SmoothL1, clamps gradients to ±1, and keeps a constant LR.
- The paper text differs: it says "trained for 10 epochs with a batch size of 32" per collection step, with N = training tasks per template / 5.

**NovPhy mapping.**
- **Examples:** one per fit shot (13 919 shots on 480 levels). Input = the level anchor (symbolic state for A3-sym, frame for A3-img); action = the mapped launch; reward = the engine verdict; every transition terminal.
- **What changes:** the online collection loop is replaced by the fixed fit set. Everything else is official.
- **Epoch and batches:** one epoch = 13 919 // 32 = 434 batches, drawn i.i.d. with replacement under the class-balanced weights (`runner_batch`, `eb.batch_rng`).
- **Optimizer:** the AdamW state is emptied at the start of each outer step (`OuterSteps`); G-port is bitwise identical across a reset.
- **Other settings:**
  - BatchNorm runs in train mode during training and in eval mode for scoring (the official `select_action` path);
  - ImageNet weights are torchvision 0.9.1's `resnet18-5c106cde.pth`;
  - the target network never enters the loss, since every transition is terminal.

**Configurations** (identical for A3-sym and A3-img; seed 20260908; selection by policy-split grid AUC of Q(s, a)):

| configuration | source | outer steps N | epochs | updates | AdamW resets |
| --- | --- | --- | --- | --- | --- |
| `native-code` | the code | 480 // 5 = 96 | 32, 31, 29, 26, 22, 17, 11, 4, then 1 × 88 = 260 | **112 840** | 95 |
| `native-paper` | the paper text | 96 | 10 each = 960 | **416 640** | 95 |
| `native-template` | the published run length (80 training tasks / 5) | 16 | code schedule = 180 | **78 120** | 15 |

**Why these three.**
- The code and the paper disagree on the number of epochs per step.
- N scales with the size of the training set: 96 here against 16 in the within-template runs behind the paper's numbers.
- The three configurations bracket the native range, and nothing else is tuned. Batch 32, lr 3e-4 and every other setting are official in all three.

**Projected GPU time [INFERENCE].**
- **Cost per update:** FlopCounter gives 153 GFLOP (sym) and 136 GFLOP (img) per update at batch 32.
- **Per seed:** at ~6–10 ms per update on one RTX 5090, 0.19–0.31 GPU-h (`native-code`), 0.69–1.16 GPU-h (`native-paper`) and 0.13–0.22 GPU-h (`native-template`).
- **A3 total:** ≤ ~8 GPU-h (2 variants × 3 selection runs, plus 2 more seeds of each selected configuration).
- **Under P-cap:** A3 is the first secondary arm to be cut.
- **CPU reference:** 0.30–0.39 s per update on 24 threads.

**Reason.** These are Phy-Q's own hyperparameters and loop structure. The only change is that online data collection is replaced by the fit shots every arm trains on.

Posted: https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382973132

## Decision P-other (2026-10-24): A4 Tier-T runtime — torch CPU threads capped at 8 (fold 0 paused and resumed once)

**Trigger.** `128-gfid-a4-fold0` (launched 21:06) ran at 0.3 updates/s instead of the probe's 2.22: the official loop computes the aux distance labels on the CPU in the training process, and torch's default of one intra-op thread per core (48) oversubscribed the shared host (load average ~135; the process sat at ~1050 % CPU). A tiny THICK step showed the same effect (100 s with 48 threads vs 0.09 s with 1).

**Choice.** `scripts/b127_dynamics_aware_fidelity.py` now calls `torch.set_num_threads(8)` (env `B127_A4_CPU_THREADS`) before the official entry point. Fold 0 was stopped once with SIGTERM at 21:35 (official checkpoint `ckpt.00000587` written by vendor edit E3, exit 75) and resumes from it through the official resume path at the next scheduler tick; folds 1–2 start with the cap. Not a recipe change: the thread count only schedules CPU work (the PHYRE rollouts run in the 20 simulator workers either way); the frozen band and protocol are unchanged.

## Decision P-cap (2026-10-25): A4 DynDQN's native budget is measured above its 3 GPU-h/seed cap → draw from unallocated program budget; no cut

**Trigger.** #127 caps A4 at 3 GPU-h per seed. The runner's throughput probe (job `128-throughput-a4-g3`, `throughput/A4.json`, RTX 5090) measured the native A4 update on NovPhy input (Table-1 recipe, batch 512 = 64 levels × 8 shots, 320×240 RGB, hand-crafted aux loss, 200 scratch updates, median of the last 150): **0.2484 s/update, peak 20.0 GiB**.
- Native configuration (100 000 updates): **6.90 GPU-h per seed**, 2.3 × the cap.
- A4 selection (native + tenth + hundredth = 111 000 updates): **7.66 GPU-h**.
- A4 total: at most 7.66 + 2 × 6.90 = **21.5 GPU-h** if `native` is selected; 7.66 + 2 × 0.69 = 9.0 GPU-h if `tenth` is.

The child projection is above the 66 GPU-h child cap. Measured terms:
- A1 selection 0.85 GPU-h and the two remaining A1 seeds ≤ 1.6 GPU-h (0.77 GPU-h per native seed);
- A2 and A2-native < 0.4 GPU-h together;
- A3 selection ≈ 2.3 GPU-h at the measured 6.6 ms/update. The ledger also holds ≈ 5 h of wall time from tonight's host stall, counted as spent because the GPU was held;
- A1 Tier E 0.24 GPU-h.

Unmeasured terms:
- A4 Tier T 23–36 GPU-h (issuecomment-6382423910); its folds must re-run after tonight's exit −9;
- A3 main seeds ≤ 3 GPU-h;
- readouts + scoring ≈ 2 GPU-h.

The worst case is ≈ 75–80 GPU-h, ≈ 10–15 GPU-h above the child cap.

**Options (P-cap order).** (1) Draw from the unallocated program budget (400 GPU-h). (2) Cut, in order: matched-budget secondary runs, then the selection grid from 3 to 2 configurations, then secondary arms. P-cap never cuts a primary arm's native budget or its three seeds.

**Choice.** (1). A4 runs its full declared protocol: all three configurations at seed 20260908 after its Tier-T G-fid passes, then 2 seeds of the selected configuration (up to 6.9 GPU-h each). Child #128 draws **up to 20 GPU-h** from unallocated program budget above its 66 GPU-h share, recorded in the runner (`P_CAP["drawn_gpu_hours"] = 20.0`, citing this decision) and in the frozen plan's projection. The smoke re-projects with every measured term before the freeze; anything above 86 GPU-h stops the freeze for a new P-cap decision. Wall: A4 selection (≈ 7.7 h on one GPU, resumable) plus 2 parallel seeds (≤ 6.9 h) fits inside the 48 h on 4 GPUs.

**Reason.** The native Table-1 recipe at our resolution costs 2.3 × the #127 per-seed estimate, which was set before any measurement. P-cap forbids shortening a primary arm's native budget, and the program budget has room.

## Decision P-crash (2026-10-25): A4 Tier-T folds killed by host-RAM exhaustion (operator-caused); relaunched once, at most two folds at a time

**What happened.** At 21:45 AEDT on 24 Oct the orchestrator moved the queued `128-gfid-a4-fold1` (to GPU 2) and `128-gfid-a4-fold2` (to GPU 0, `shares_gpu`) so that all three A4 Tier-T folds ran at once. Each fold runs 20 PHYRE simulator workers beside its training process (about 30–40 GB of host RAM per fold). With the other night jobs and the E1 engine smoke, host RAM ran out within five minutes (MemAvailable < 1 GB, swap full). The kernel OOM killer ended the three fold processes at about 21:50 (scheduler: exit −9 for fold 0 run 2, fold 1 run 1, fold 2 run 1); their 55 simulator workers survived as orphans and held about 73 GB until the orchestrator killed them at 03:31 on 25 Oct. The cron daemon also stopped (no scheduler tick from 22:00 to 03:38). `128-throughput-a4` (GPU 0) failed with a CUDA OOM beside fold 2's process.

**Classification.** Infrastructure crash, not a typed failure: no fold reached an evaluation, and no fidelity number exists. Cause: the orchestrator's co-scheduling, not the port or the recipe.

**Decision (rule P-crash: relaunch once, with disclosure).**
- Each fold resumes from its last official checkpoint (the runner's resume path); nothing about the recipe, data, seeds or the frozen band changes.
- At most **two** A4 folds run at the same time, and never beside an engine (Unity) run; the remaining fold waits for one of them (scheduler `depends_on`).
- Tonight (25 Oct, until 06:50): fold 0 only. From 18:00 on 25 Oct: folds 1 and 2, fold 0 continuing as GPUs free up.
- A fold that crashes again for an infrastructure reason is not relaunched a second time without a new posted decision.
- The ledger counts only the GPU seconds each run actually recorded; the killed segments (about 0.8 GPU-h in total) are listed as unrecorded in the compute section.

## Decision P-novelty (2026-10-25): the descriptive #110 novelty scoring pass (owner decision D6)

**Rule.** #127 P-novelty: every arm of this child that passes its guards is scored on the #110 novelty evaluation pairs after its primary scoring pass. Descriptive only: no token, never part of a disposition or the program claim. Fixed now, before any arm of the program has an evaluation score.

**Population.** The `issue110-evaluation` levels of the 29 in-scope cells under #107's frozen rule (`run_granularity_under_novelty.in_scope_cells`: #110-supported, has mixed-verdict evaluation levels, not rollout-error-only), on both sides of every pair: the novel side at novelty level k = 1–7 and its paired level-0 normal side. Grid inventory (16 launches; the level-5 slingshot cells use #110's mirrored grid). Engine verdicts from the #110 handoff (`handoff_107.json`). AUC on mixed-verdict levels only.

**Inputs.** #107's anchor rule: the level's first complete grid candidate in suffix order, shot frame 0 (pre-launch). Pixel arms read that frame; carrier arms read its E99 v1 batch-1 parse (`rt.anchor_carrier` with the frozen E99 encoder, under `deterministic_scoring()`, the #102 N2 pattern); A3-sym builds its state from the level's `decision-physics-v1.json`. The E99 v1 carrier does not represent most novelty entities (#110: v1 fully supports 2 of 45 cells); carrier arms therefore see the novel levels through that limited contract, which is part of the descriptive result. No input contract is changed for this pass.

**Scoring.** Each guard-passing arm, all three seeds, its primary score (logit / Q for direct predictors; tie-free cost of the predicted endpoint carrier for world models), once, in a process that did not train, under `deterministic_scoring()`, after the arm's primary scoring pass. References on the same cells and anchors: U and C* (the Gate-B checkpoints rolled out as Gate-B scores them) and P (the Gate-B prior rule over the fit verdicts).

**Numbers (DESCRIPTIVE).** Per ranker: pooled grid AUC per novelty level on the novel side, on the paired normal sides, pooled over novelty levels 1–7, and the paired contrast novel − paired normal (one row per (pair, seed) where both sides are mixed); level-clustered (pair-clustered for the paired contrast) percentile bootstrap, 10 000 draws, PCG64 7201. Records are hashed into this child's records manifest and re-derived by `--validate`.

Shared implementation: `scripts/external_baselines.py` (`novelty_cells`, `read_novelty_anchors`, `novelty_reference_records`, `novelty_table`), pinned by every child plan; each child runner gets a `--score-novelty` mode before its freeze.

## Decision P-other (2026-10-25): capture-noise tolerance for the frame-0 check; fit/policy layout redundancy measured; NN-px / NN-car no-model references added (descriptive)

**Trigger.** The pre-freeze review Review128 found no leakage: inputs are pre-launch anchors, the score path is label-free, and the selection AUCs reproduce on CPU with 0.0 delta. It raised two blocking items and two wording items. They are fixed here, before `--smoke` and the plan freeze. No evaluation-split content has been read. Every number below is from the fit / policy splits (CPU).

**1. Frame-0 check with a capture-noise tolerance.** The previous smoke check required frame 0 of every branch of a level to equal the level's anchor exactly. That is false on real data because the captures are nondeterministic. It is replaced by an exhaustive check on both splits (`anchor_frame_check`, smoke `accounting.anchor_frame_pre_launch`):
- the anchor equals frame 0 of its own anchor branch **exactly**;
- frame 0 of every other branch is within **≤ 32 differing pixels** (positions where any channel differs) **and mean |Δ| ≤ 0.01 grey levels** over the 230 400 uint8 elements;
- the number of levels over tolerance must be 0.

The tolerance and its pixel unit follow Review128's suggestion.

Measured:

| split | levels | branches | anchor exact | branches with any difference | levels | worst differing pixels | worst mean abs diff | over tolerance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fit | 480 | 13 919 | yes | 293 | 43 | 24 | 0.0020 | 0 |
| policy | 120 | 3 480 | yes | 43 | 15 | 18 | 0.0020 | 0 |

Training and scoring use the anchor branch's frame 0 only. The runner docstring is corrected.

**2. Fit/policy layout redundancy, measured and disclosed (`policy_fit_redundancy`, smoke.json; a disclosure, not a gate).** The #109 cohort has two generator templates, and the policy levels have near-duplicate fit levels.

Nearest anchor distances:

| distance | to nearest fit anchor (p50 / p95 / max) | to nearest other policy anchor (p50 / p95) |
| --- | --- | --- |
| mean abs pixel diff, grey levels | 0.41 / 0.87 / 1.11 | 0.60 / 1.89 |
| carrier L2 | 0.10 / 0.26 | 0.14 / 1.55 |

- 100 of 120 policy levels are closer to a fit level than to any other policy level.
- A parameter-free nearest-fit verdict copy scores a **pooled policy grid AUC of 0.946 with pixel distance (NN-px)** and **0.810 with carrier distance (NN-car)**. On the angle inventory these are 0.945 and 0.820.
- Every policy grid / angle candidate launch was executed by the fit split (1504 / 1504 and 949 / 949).

This explains the very high policy-split selection AUCs (A1 ≈ 0.996, A2 ≈ 0.973). The inflation applies to every arm's selection alike. The same effect on the evaluation split is measured by the two references below.

**3. Two no-model reference rankers, pre-registered (definitions frozen in `plan.json` → `no_model_references`):**
- **NN-px.** For each evaluation (Gate-B) anchor, find the nearest of the 480 fit anchors by mean absolute difference over the 320×240 RGB anchor frames (exact integer sums).
- **NN-car.** The same, by L2 distance between E99 anchor carriers (float64).
- **Score.** A candidate's score is the engine verdict (1/0) of the identical launch executed in that nearest fit level. Cost = −score (`eb.score_rows`). Distance ties go to the lowest fit member id. If that fit level did not execute the launch, the next fit level in (distance, member) order that did is used.
- **When scored.** Seed-free, like P. Scored once in the first scoring pass on every mixed cell of every inventory from the Gate-B anchors (`references/NN-px.json`, `references/NN-car.json`, included in the records manifest).
- **Reporting.** DESCRIPTIVE rows beside U / C* / P: pooled AUC, plus estimates of X − NN-px, U − NN-px, X − NN-car and U − NN-car without tokens. They are never part of a disposition, a guard or the program claim.

**4. Wording corrections (P3).**
- The A4 sampler docstring (and the A4 P-other text, line "64 tasks × …" in decisions.md) said NovPhy has "all-success levels". The fit split has **0 all-success and 55 all-fail levels**: no positives, so the official negative fallback is used. The positives fallback is only a guard. The code is unchanged and the docstring is fixed by DynA4. `world_model/baselines/dynamics_aware/novphy.py` is now sha256 `9cc48078…324b`; G-port was re-run and still passes with all deltas 0.
- `policy_fit_level_overlap` now intersects all 120 policy levels with the 480 fit levels. It is 0.

**Options.**
- Exact frame-0 equality: rejected, it fails on capture noise unrelated to the launch.
- Dropping the frame-0 check: rejected, the reviewer needs the pre-launch evidence.
- No reference for the redundancy: rejected, a reader could not tell how much of any arm's evaluation AUC a template lookup could reach.

**Reason.** The check keeps the integrity guarantee (inputs are pre-launch and identical up to capture noise), and the redundancy is disclosed and measured with reference rankers before any evaluation number exists, without changing any arm, configuration, selection rule or budget.

Addendum to Decision P-other 2026-10-25 (https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6387263066): NN-px / NN-car are defined on grid and angle only; the fit split never executed offset or power launches, so those inventory columns are empty by construction.

## Decision P-engine (2026-10-25): A3 native-task fidelity design (target, band, cost) and its funding under the shared engine allowance

**Trigger.** Rule P-engine. A3's dispositions are licensed only after a native-task fidelity run: a Phy-Q within-template training run in the engine that reproduces a published per-template result. That run is funded only if its projection fits the remainder of the 190 CPU worker-hours shared with E1, and E1 runs first. A3 is secondary and never in the program set. This decision is posted before any A3 engine run.

**Published target.**
- **Source:** Phy-Q reports per-template within-template results only as a bar chart (arXiv 2108.13696 v3, Supplementary D, `Fig_within_template_submission_2_avg_pass_rates.PNG`; no SD; one run per template).
- **Digitisation** (`scripts/b127_phyq_digitise_within_template.py` → `a3/within-template-digitised.{json,md}`):
  - Pixel readout from the PNG inside the arXiv source tarball (sha256 recorded). Bar colours come from the legend swatches. The y scale comes from each panel's 0 and 1 ticks, and each bar's top edge is read to sub-pixel accuracy.
  - Calibration against the four heuristic agents, whose per-template pass rates are recomputed from the released playdata. Over 300 bars, mean absolute error is 0.004. For bars with true value ≥ 0.1, the calibrated error is p95 0.0036 and max 0.0071 (leave-one-scenario-out).
  - Figure slots are mapped to template ids by matching the heuristic bars. 68 of 75 slots are pinned by the data alone.
  - Independent check: the calibrated learning-agent bars lie within about 0.001 of multiples of 1/20, consistent with 20 test tasks per template.
- **Chosen template: 3_02_01** (Relative height, figure slot `08_3`):
  - its slot is pinned by the data;
  - single-bird tasks;
  - wide, clean bars;
  - both relational values lie mid-range, so a failed reproduction is detectable;
  - Relational-Symbolic reads **0.50** and Relational-Image **0.60** (calibrated 0.501 / 0.602; readout error ≤ 0.007).

  The other candidates (3_09_03, 2_01_09, 3_04_03, 2_04_05) are ranked in the JSON. The choice depends only on readability, never on any NovPhy result.

**Band** (frozen here; the statistic is the pass rate on the template's 20 test tasks, exactly as the paper reports it):
- **Statistic:** the published value comes from one run on 20 test tasks, and so does ours. If both runs share the same true rate p, their difference has SD √(2·p(1−p)/20).
- **Band:** the published value ± (2 SD + 0.01 readout allowance).
  - Relational-Symbolic: 0.50 ± 0.326 → **[0.174, 0.826]**.
  - Relational-Image: 0.60 ± 0.320 → **[0.280, 0.920]**.
- **Pass:** our 20-task test pass rate lies inside the band.
- **What the band covers:** test-set sampling noise only. Training-run variance is not reported in the paper, so the band is not widened for it. The test is weak (one template), and this is disclosed.

**Run definition (Phy-Q released code):**
- `TrainLearningAgent.py` within-template mode: 80 training tasks and 20 test tasks.
- 16 update steps: step 0 plays 80 tasks × 10 episodes; steps 1–15 play 10 episodes each.
- `simulation_speed` 100, AdamW lr 3e-4, batch 32.
- Then one greedy test pass on the 20 test tasks.
- Engine cost: 970 shots per run (1 bird per task; `a3/harness-check.json`).

**Cost.**
- The E1 smoke measured PigShooter on the same build at the same speed 100: median 10.7 s from shot to game result. Adding the environment's reload (load, zoom, a 2 s sleep, ground-truth polls) gives about **15 s per episode** [INFERENCE].
- That is about **4.0 worker-h per variant** and **8.1 worker-h for both**, plus a short A3 smoke (counted).
- Network training runs on CPU between collection steps, inside the daytime engine window. It is CPU time, not engine time.

**Funding (P-engine).**
- E1 has spent 6.19 worker-h and projects 182.4 more (#132 decision of today), leaving about **1.4 worker-h**. That is below A3's projection.
- **Rule:** after E1 finishes, let R = 190 − E1's realised worker-hours (smoke and probes included).
  - If R ≥ 1.25 × 4.0 worker-h, Relational-Symbolic's fidelity run goes ahead. The 25 % margin covers the [INFERENCE] per-episode cost, and an A3 smoke of the first 50 episodes re-checks the projection.
  - Relational-Image follows if what is left still fits.
  - Any variant not funded stays **tokens-only, typed `engine_budget`**, and is reported as such.
- **Why Relational-Symbolic first:** its published 75-template average is the higher of the two (0.3220 vs 0.2393). The order is fixed now, before any A3 result.

**Reason.** This is the cheapest faithful design: one template, Phy-Q's own protocol and budget, a target read reproducibly from the paper's figure and calibrated against known values, and a band whose statistic matches the published one. Funding follows P-engine's E1-first order without weakening E1.

**Frozen band file.** `.local-artifacts/issue-128-direct-predictors-v1/a3/fidelity-band.json`, sha256 `84b17d0914791dcde3ae54899b0373064bd564c77c7de73f7c53a90eaa99077f`.

## Decision P-other (2026-10-27): the primary scoring group can be frozen and scored with A4 recorded as dropped (post-freeze amendment, decided before A4's G-fid outcome)

**Trigger.** A4's Tier-T G-fid (folds 0–2) finishes overnight 27→28 Oct; its outcome is not known to anyone yet (no fold result has been read). The frozen #128 plan scores the group `primary` "after every seed of A1, A2, A2-native and A4 is trained and read out", and `--freeze-readouts` refuses a partial group. If A4 were dropped under P-fid (`fidelity_not_reproduced` after the permitted repair attempts), A1, A2 and A2-native could then never be scored — a gap in the frozen runner, not a choice anyone made.

**Decision.** Close the gap now, before the outcome, so that no later step depends on it:
- A primary-group arm that is **dropped with a typed reason** under a posted decision (P-fid `fidelity_not_reproduced`, or a typed terminal training failure) is recorded in the group's readouts manifest as dropped (reason + decision URL), with no models, readouts or records.
- The remaining arms of the group freeze and score exactly as frozen; nothing else changes (anchors, scoring, guards, tokens, contrasts, the one-pass rule, G-chronology).
- The dropped arm is published with no records, every token the stop token, disposition 4, and it vetoes the program claim (as C1-dino).
- Implemented as a registered post-freeze amendment of `scripts/run_issue_128_direct_predictors.py` (and its tests), with the A4 G-fid gate job unchanged: on a pass, nothing in the amendment is exercised.

**Reason.** This is a mechanical completion of the frozen dropping rule (P-fid already defines the drop and its reporting); it does not depend on any result, and it keeps A1/A2/A2-native — trained and read out on 25 Oct — reportable whatever A4's fidelity outcome.

**Also:** the post-G-fid chain is enqueued as `scripts/gpu_window_issue_128_a4_post.json` (gate on `fidelity/a4/gfid.json` pass → A4 deferred selection → selection readouts → `--select-deferred` → A4 seeds 20260909/10 → A4 readouts → `--freeze-readouts --arms A1,A2,A2-native,A4` → score → score-novelty → publish → validate). It supersedes `scripts/gpu_window_issue_128_scoring.json` (same scoring jobs without dependencies), which is removed.
