# Issue #131: THICK (#127 arm D1) on NovPhy (port, fidelity, guards, and the frozen disposition)

Parent: #127 (external baselines program). Ticket: [#131](https://github.com/Sino-Huang/NovPhy/issues/131). The
binding shared rules of #85 and the #127 shared protocol (`scripts/external_baselines.py`, `eb`) apply. The Gate-B
(#108), #101, #104, #109, #112, #117 and #123 records and stores are read-only inputs. Nothing here edits the ICLR 2026
submission.

| | |
| --- | --- |
| Paper and code | Gumbsch, Sajid, Martius, Butz, *Learning Hierarchical World Models with Adaptive Temporal Abstractions from Discrete Latent Dynamics*, ICLR 2024 (OpenReview TjCDNssXKU; no arXiv version); [CognitiveModeling/THICK](https://github.com/CognitiveModeling/THICK) @ `934980a` (MIT, TensorFlow 2, DreamerV2 code base) |
| Route | (b): our PyTorch port held to G-port against the official TF2 code on CPU. The JAX port [CognitiveModeling/thix](https://github.com/CognitiveModeling/thix) carries no LICENSE file and was not used ([P-route](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382162858)) |
| Runner | `scripts/run_issue_131_thick.py` (selection / selection-readouts / smoke / prepare / train / readouts / freeze-readouts / score / score-novelty / publish / validate) |
| Modules | `world_model/baselines/thick/port.py` (THICK world model: C-RSSM with the GateL0RD context kernel, THICKNet high level, losses, Keras-semantics Adam); `novphy.py` (NovPhy adapter); `agent.py` (THICK Dreamer, DreamerV2, Plan2Explore); `envs.py` (VisualPinPad, wrappers, driver, replay, exploration cache); `fidelity_runner.py` (resumable `train.py` / `train_from_replay.py` loops); `vendor/` (the official code, 25 files with per-file sha256 in `VENDOR.json`; one listed edit, below) |
| Fidelity | `scripts/b127_thick_port_reference.py` (G-port reference and comparison, world model and agent), `scripts/b127_thick_fidelity.py` (band, GPU probes, timing gate, exploration cache, G-fid runs, G-fid record) |
| Tests | `tests/test_issue_131_thick.py`, `tests/test_issue_131_thick_gfid.py`, `tests/test_issue_131_thick_runner.py` (45 tests, CPU) |
| Artifacts | `.local-artifacts/issue-131-thick-v1/`: `plan.json`, `smoke.json`, `selection.json`, `decisions.md`, `findings.md`, `summary.json`, `compute.json`, `records-manifest.json`, `validation.json`, `readouts-manifest.json`, `gport/gport.json`, `gport/agent/gport.json` (+ `sensitivity.md` in both), `fidelity/{band,plan,gfid-timing,gfid,cache-verification}.json`, `fidelity/runs/*/` (G-fid run logs), `envs/`, `ledger/`. Checkpoints, readouts, records, the G-port reference arrays and the paper PDF (`data/external/thick/`, `MANIFEST.json`) stay local; records and readouts are hashed in the manifests |
| Freeze | pre-freeze implementation `9d905f35`; `plan.json` frozen 2026-10-25T20:09:20Z (07:09 AEDT, 26 Oct), commit `0d1faeac` (sha256 `557d4572…`, 134 pins), after a passed smoke and before any main-seed training or evaluation-split read ([P-freeze](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6402578948)). No post-freeze amendment |
| Readout freeze | `readouts-manifest.json` frozen 2026-10-26T17:55:28Z (04:55 AEDT, 27 Oct; sha256 `b0919640…`), after G-fid passed and before any evaluation anchor was read |
| Scoring | once, in a process that did not train: scheduler run `131-score#1`, 2026-10-27 05:00:20–05:05:20 AEDT, exit 0 (6 arm-seeds; evaluation anchors read 18:00:37Z, first record 18:01:12Z); P-novelty pass `131-score-novelty#1`, 05:05–05:10, exit 0 (no arm passed its guards; references only) |
| Validation | `python -u -m scripts.run_issue_131_thick --validate`, scheduler run `131-validate#1`, 2026-10-27 05:15:20–05:20:20 AEDT, **exit 0**: 14 records hash-checked, D1 seed 20260908 re-scored on every inventory with exact row equality. `validation.json` binds `records-manifest.json` sha256 `170dde49…`. Publication commits `b4b0df9a`, `f3be7485` |

Engine time is zero seconds. The primary metric is the Gate-B pooled grid AUC over the 311 mixed grid cells × 3 seeds
of the #104 sealed #109 evaluation split, tie-free cost, under `deterministic_scoring()`. Intervals are level-clustered
percentile bootstraps (10 000 draws, PCG64 7201) and are DESCRIPTIVE. The margin is δ = 0.02.

## Question

#127 Q4: does a learned, event-triggered temporal-abstraction world model, THICK (a low level that updates a context
sparsely in time through GateL0RD gates, and a high level that predicts the state before the next context change),
rank the frozen 16-launch grid better than U and than J_dec, our explicit request selection? THICK's adaptive Δ is the
set of steps at which its context gates open.

## Outcome

**Disposition 4: per-contrast tokens only. No scoped ranking claim either way.**

- Every global guard passes: G-isolation, G-chronology (including the novelty pass), G-scoring-policy and G-records.
  G-port (world model and full agent) and G-fid pass.
- D1 (primary, program set) fails **G-action on every seed**: endpoint action sensitivity 0.00064 / 0.00461 / 0.00166
  against the frozen 0.05. D1-m (matched secondary) fails G-action on every seed (0.00025 / 0.00055 / 0.00050) and
  **G-collapse** on every seed (effective rank 55.9 / 57.4 / 53.1 of 1 082, below 0.1 · D = 108.2). Every other arm
  guard (G-rank, G-nonfinite, G-trained) passes on every seed.
- By the frozen rule every token involving D1 or D1-m is `readiness_or_precision_insufficient`. Neither disposition 1
  nor disposition 2 applies, and D1 vetoes the #127 program-level claim (guard failed).
- Descriptively, D1 ranks the grid at 0.4989 [0.4810, 0.5165] against U's 0.7444 [0.7269, 0.7620]; D1 − U is
  −0.2456 [−0.2689, −0.2226] and D1 − J_dec −0.2471 [−0.2710, −0.2236]. These carry no token.
- **Tier-C trigger (owned here): not met.** #127 runs Diffusion Forcing or γ-models as Tier-C arms if D1 − U or
  D1 − J_dec is `supported`; both carry the stop token. They are therefore not triggered as Tier C; they run as
  coverage arms under P-coverage (orchestrator).

## Design summary

The design is frozen in `plan.json`; every implementer decision was posted on #131 before the step it governs and is in
`decisions.md` (list at the end).

**Model.** Native THICK, the vendored `thick_pinpad` configuration with THICK's own MLP observation pathway on the
236-d E99 carrier ([adapter decision](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382505350)):

- C-RSSM with deter / hidden 256, 32 × 32 discrete latents, a 16-d GateL0RD context (gate noise σ 0.1); THICKNet
  3 × 1024 with 5 high-level actions; encoder and both decoders 4 × 400 ELU with `mse` heads on the carrier; KL
  balance 0.8; `thick_act` / `thick_time` loss scales 0.1; two Adam optimizers (lr 1e-4, eps 1e-5, clip 100, wd 1e-6,
  Keras semantics);
- episode = one fit shot, carriers at frames 0, 5, …, 225 (46 observations; positions after a shot's end repeat its
  final carrier, 601 of 13 919 shots); the 5-d launch (`eb.launch5`) is the action at index 1 (upstream transition
  layout: applied at t0), zero no-ops after;
- no task signal: reward 0, no termination, segmentation on the context gates only; batches of 16 episodes; one update
  is the low-level step, then the high-level step on the updated low level;
- readout: observe the anchor at t0, then 45 deterministic low-level steps (latent modes, zero GateL0RD noise, the
  launch at the first step); score = tie-free `EndpointCosts` of the fine decoder's frame-225 carrier.

**Arms** (fit split only, seeds 20260908/09/10; [runner decision](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6383193006)):

| arm | model | schedule | role | trainable parameters |
| --- | --- | --- | --- | --- |
| **D1** | native THICK, `ctxt_sparsity` 1 (selected) | 200 000 updates × 16 episodes (upstream `steps` 1e6 / `train_every` 5) | primary, program set | 21 852 337 |
| D1-m | every hidden width × 0.166 (RSSM 42, MLPs 66, THICKNet 170); latents, context, depths native | U's B1 recipe 9 000 × 64 | secondary (matched budget and parameters; U 1 837 690, −0.14 %) | 1 835 187 |
| D1-coarse | D1's context decoder at the endpoint (`[stoch, context]`) | — (same model) | exploratory view | — |
| D1-hl | D1's high-level jump prediction from the post-launch state, decoded by the context decoder | — (same model) | exploratory view | — |

THICK has no native success head on NovPhy (the reward heads are trained on zeros), so there is no native-head score
and the #127 U + head control does not apply to D1.

**Selection** (#127 rule: policy-split pooled grid AUC of the primary score, 94 mixed policy cells, seed 20260908,
full budget each; the selected run is reused as D1 seed 20260908):

| configuration | `loss_scales.ctxt_sparsity` | policy grid AUC |
| --- | --- | --- |
| **sparsity-1** | 1 (the paper's VisualPinPad value) | **0.5312** (selected) |
| sparsity-10 | 10 (MiniHack) | 0.4802 |
| sparsity-25 | 25 (Multiworld) | 0.4598 |

All three configurations sit near chance on the policy split (disclosed; no re-selection). U had no policy-split
tuning of this kind (#127), disclosed.

## G-port and G-fid

**G-port: PASS** (CPU, float32, the official TF 2.4.2 code in env `b127-thick-tf` against the port on the same fixed
inputs, weights copied TF → torch, every random draw recorded on the TF side and replayed in call order; tolerances as
#123: loss / grad-norm relative 1e-4, per-tensor relative L2 1e-3, parameters within 1e-7).

- *World model* (`gport/gport.json`), two configurations: the NovPhy carrier configuration (7 064 730 low-level +
  14 787 607 high-level parameters) and the official VisualPinPad image configuration. 40 / 40 module forwards
  (encoder, GRU, GateL0RD with and without noise, observe / imagine, decoders, heads, THICKNet) pass; two optimizer
  steps at the constant LR 1e-4 (THICK has no schedule, so the step-0 LR is the positive LR): loss terms within
  4.5e-7 relative, worst gradient tensor relative L2 1.6e-5, updates within 1.6e-5, every parameter within 1e-7.
- *Full agent* (`gport/agent/gport.json`), for the G-fid route: from the official state after one warm-up
  `Agent.train`, two recorded `Agent.train` calls with every optimizer compared:

| arm | parameters | worst gradient rel. L2 per optimizer | worst loss / metric rel. | worst update rel. L2 | acting |
| --- | --- | --- | --- | --- | --- |
| THICK Dreamer | 45 426 467 | world model 2.7e-6, high level 1.4e-5, actor 3.3e-6, critic + coarse critic 7.4e-6 | 4.3e-6 | 1.1e-5 | actions equal |
| DreamerV2 | 17 210 347 | world model 3.2e-6, actor 1.6e-6, critic 1.9e-6 | 1.7e-6 | 9.0e-6 | actions equal |
| Plan2Explore | 34 256 194 | world model 3.3e-6, ensemble 1.0e-6, actors 1.6e-6, critics 1.7e-6 | 1.4e-6 | 8.8e-6 | actions equal |

- Sensitivity (not part of the gate): injected defects fail it — TF-style vs torch-style Adam epsilon, GateL0RD gates
  without the straight-through ceil, flipped KL balance (world model); λ 0.9, ψ 0.1, biased vs unbiased disagreement
  std (agent).

**G-fid: PASS** (THICK Dreamer's advantage over DreamerV2 on VisualPinPadFour).

- *Band* ([P-fid](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382475412), frozen before any run;
  `band.json` sha256 `c48251b3…`): read from the vector paths of the paper's Fig. 8 (7 seeds, ± SE; SD = SE · √7),
  checked against Table 3 (the digitised curves cross the 95 % thresholds at exactly the tabulated steps). PinPadFour:
  THICK 374.3 ± 7.3 → [359.7, 388.8], DreamerV2 372.8 ± 5.8 → [361.3, 384.4]; threshold 359 reached at 140k vs 200k.
  Pass rule: both final (400k) seed-mean returns in band, and THICK's seed-mean curve reaches 359 strictly earlier.
- *Route* ([P-route](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382756572)): the official stack
  does not run on the RTX 5090 (TF 2.4.2 + CUDA 11.0 registers the GPU but a first cuDNN convolution did not return in
  5 min; TF 2.21 registers no GPU), so G-fid runs on the port extended to the full agent (G-port above). Design: one
  Plan2Explore run of 1M env steps whose replay pre-fills every run (upstream `train_from_replay.py`); THICK Dreamer and
  DreamerV2 × seeds 20260908/09/10 to 400k env steps; one 2000-step evaluation episode per 1e4 env steps; TF32 on
  (the official stack's own choice on this GPU). Feasibility gate: a timing job measured the first 1 000 updates of
  each kind and projected 37.8 GPU-h (cap 72); [runner decision](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6383312669).
- *Result* (`fidelity/gfid.json`, sha256 `52e1ff84…`):

| method | final (400k) seed-mean return | band | in band | steps to 359 (paper) |
| --- | --- | --- | --- | --- |
| THICK Dreamer | 370.0 | [359.7, 388.8] | yes | 120k (140k) |
| DreamerV2 | 376.7 | [361.3, 384.4] | yes | 210k (200k) |

  THICK reaches the threshold 90k env steps before DreamerV2 (paper: 60k). Both P-crash incidents of the G-fid
  pipeline are in "Deviations and disclosures" (items 1 and 2); neither touches training semantics or the band.

## Results

### Guards

| arm | seed | G-rank (mean tied-pair share) | G-action (≥ 0.05) | G-collapse (eff. rank / D; χ) | G-nonfinite | G-trained | policy grid AUC (fit/policy only) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D1 | 20260908 | pass (0.0410) | **fail (0.00064)** | pass (171.3 / 1 296; 0.184) | pass (0.0) | pass | 0.5312 |
| D1 | 20260909 | pass (0.1240) | **fail (0.00461)** | pass (170.3 / 1 296; 0.325) | pass | pass | 0.4971 |
| D1 | 20260910 | pass (0.0335) | **fail (0.00166)** | pass (167.7 / 1 296; 0.155) | pass | pass | 0.4474 |
| D1-m | 20260908 | pass (0.0057) | **fail (0.00025)** | **fail (55.9 / 1 082; 0.155)** | pass (0.0) | pass | 0.4067 |
| D1-m | 20260909 | pass (0.0041) | **fail (0.00055)** | **fail (57.4 / 1 082; 0.250)** | pass | pass | 0.5538 |
| D1-m | 20260910 | pass (0.0123) | **fail (0.00050)** | **fail (53.1 / 1 082; 0.191)** | pass | pass | 0.4570 |

G-collapse is the #123 definition on the posterior decoder input (`get_feat`: 1 024 stochastic + deter + 16 context)
of the fit episodes at frames 0, 15, …, 225; G-readout does not apply (carrier-space score, no readout). Global guards:
G-isolation, G-chronology (plan 2026-10-25T20:09:20Z ≤ readouts 2026-10-26T17:55:28Z ≤ evaluation anchors 18:00:37Z ≤
first record 18:01:12Z; novelty anchors 18:05:55Z after the last primary record 18:04:02Z), G-scoring-policy and
G-records all pass.

**G-action in carrier space (descriptive; program decision on
[#127](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6396584893)).** On the fit split the engine truth's
own endpoint action sensitivity is 0.0395 (U 0.0229, C\* 0.0334, B1 0.0130; #129). D1's mean 0.00230 is **5.8 %** of
the engine truth's, D1-m's 0.00043 **1.1 %**: THICK's decoded endpoints barely depend on the launch. The frozen
guard, its failure and the disposition stand.

### Grid rankers (311 mixed cells × 3 seeds)

| ranker | pooled AUC [CI] | per seed (08 / 09 / 10) | top-1 (chance 0.089) |
| --- | --- | --- | --- |
| U | 0.7444 [0.7269, 0.7620] | 0.7141 / 0.8011 / 0.7181 | 0.195 |
| C\* | 0.7834 [0.7676, 0.7992] | 0.8262 / 0.7334 / 0.7905 | 0.192 |
| P | 0.7203 [0.6965, 0.7442] | — | 0.180 |
| J | 0.7057 [0.6846, 0.7267] | 0.6690 / 0.7012 / 0.7470 | 0.160 |
| J_dec | 0.7460 [0.7277, 0.7637] | 0.7205 / 0.7933 / 0.7242 | 0.180 |
| **D1** | 0.4989 [0.4810, 0.5165] | 0.5151 / 0.5211 / 0.4603 | 0.044 |
| D1-m | 0.4820 [0.4659, 0.4974] | 0.4011 / 0.5954 / 0.4496 | 0.039 |
| D1-coarse (exploratory) | 0.4900 [0.4748, 0.5053] | 0.5183 / 0.5295 / 0.4222 | 0.038 |
| D1-hl (exploratory) | 0.4971 [0.4809, 0.5136] | 0.4896 / 0.4912 / 0.5105 | 0.053 |

No ranker has tied pairs or nonfinite rows except P (tied-pair share 0.002).

### Contrasts (grid, paired per (level, seed))

| contrast | estimate [CI] | token | reverse token |
| --- | --- | --- | --- |
| D1 − U | −0.2456 [−0.2689, −0.2226] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| D1 − C\* | −0.2845 [−0.3076, −0.2615] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| D1 − P | −0.2214 [−0.2472, −0.1953] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| D1 − J_dec | −0.2471 [−0.2710, −0.2236] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| D1-m − U | −0.2624 [−0.2903, −0.2351] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| D1-m − C\* | −0.3014 [−0.3282, −0.2754] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| D1-m − P | −0.2383 [−0.2663, −0.2110] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| D1-m − J_dec | −0.2640 [−0.2929, −0.2358] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |

Every token is the stop token because D1 and D1-m fail G-action (D1-m also G-collapse). The estimates are reported as
measured; they license no statement in either direction.

**Within family.**

| contrast | estimate [CI] | token | reverse token |
| --- | --- | --- | --- |
| D1 − J_dec (#131 within-family) | −0.2471 [−0.2710, −0.2236] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| D1 − D1-m | +0.0168 [−0.0063, +0.0405] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| D1-coarse − D1 (exploratory) | −0.0088 [−0.0295, +0.0117] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| D1-hl − D1 (exploratory) | −0.0018 [−0.0239, +0.0200] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |

### Wrong-anchor control (right − wrong, 480 fit anchors, grid)

| arm | right | wrong | right − wrong | ranking stability (Spearman) | token |
| --- | --- | --- | --- | --- | --- |
| D1 | 0.4989 [0.4810, 0.5165] | 0.5164 [0.5140, 0.5188] | −0.0176 [−0.0353, −0.0002] | 0.0334 | `readiness_or_precision_insufficient` |
| D1-m | 0.4820 [0.4659, 0.4974] | 0.4614 [0.4486, 0.4741] | +0.0206 [+0.0081, +0.0326] | 0.2762 | `readiness_or_precision_insufficient` |

### Descriptive inventories (no tokens)

| inventory (cells) | U | C\* | P | J | J_dec | D1 | D1-m | D1-coarse | D1-hl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offset (228) | 0.6670 | 0.7748 | 0.5000 | 0.7522 | 0.6904 | 0.4955 | 0.4475 | 0.5549 | 0.5377 |
| angle (234) | 0.6791 | 0.7590 | 0.8119 | 0.7765 | 0.6988 | 0.5264 | 0.4485 | 0.5082 | 0.5557 |
| power (141) | 0.8058 | 0.8068 | 0.5000 | 0.8076 | 0.7943 | 0.5357 | 0.3825 | 0.5243 | 0.5787 |

D1 − U: offset −0.1714 [−0.2132, −0.1281], angle −0.1527 [−0.1857, −0.1204], power −0.2701 [−0.3402, −0.2023]
(stop tokens).

### Engine position error (#100 metric; DESCRIPTIVE)

Mean slot-centre MSE of the predicted carrier against the engine centres on present slots, every complete grid
candidate of every mixed grid cell, from the Gate-B anchors (member-clustered bootstrap):

| t | 15 | 30 | 60 | 150 | 225 |
| --- | --- | --- | --- | --- | --- |
| D1 | 0.0528 | 0.0528 | 0.0512 | 0.0497 | 0.0521 [0.0474, 0.0569] |
| D1-m | 0.0396 | 0.0407 | 0.0394 | 0.0391 | 0.0423 [0.0390, 0.0457] |
| U | 0.0505 | 0.0501 | 0.0484 | 0.0459 | 0.0474 [0.0440, 0.0508] |
| C\* | 0.0547 | 0.0545 | 0.0543 | 0.0594 | 0.0670 [0.0617, 0.0723] |

D1-m's position error is the lowest of the four while its endpoints are nearly launch-invariant (G-action 1.1 % of the
engine truth's); position error does not measure ranking.

### Commit points (THICK's adaptive Δ; EXPLORATORY)

On the 14 928 grid candidates (all seeds), D1's context gates open at a mean of **42.2 of the 45 rollout steps**, every
candidate has at least one commit, and every step from 1 to 45 has commits for at least 13 172 of the candidates
(steps 1–3 for all). D1-m's gates open at every step of every candidate (45.0). On the fit anchors the high level's
predicted time to the next context change averages 0.12 / 0.00 / 0.00 steps (seeds 08 / 09 / 10). With
`ctxt_sparsity` 1 the context kernel therefore did not learn sparse commit points on NovPhy's 5-frame stride: the
adaptive Δ collapses to Δ = 1 step. sparsity-10 and sparsity-25 (sparser by construction) ranked worse on the policy
split and were not selected; no post-hoc configuration is run.

### P-novelty (#110 novelty pairs)

Not scored for D1 (G-action) or D1-m (G-action, G-collapse): P-novelty scores each arm that passes its guards
([P-novelty](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6386831318); `summary.json`
`extra.novelty.not_scored`). The U / C\* / P references on the 29 in-scope cells were computed after the last primary
record and are hashed in `records-manifest.json`; their table is in `findings.md` (descriptive; e.g. pooled novelty
levels 1–7, novel − paired normal: U −0.1537 [−0.2127, −0.0960], C\* −0.1693 [−0.2293, −0.1092], P −0.1012
[−0.1781, −0.0274]).

## Disposition (δ rule)

- Disposition 1 (D1 ranks launches better than U, scoped): **no** — D1 fails G-action; D1 − U carries the stop token.
- Disposition 2 (U ranks launches better than D1, scoped): **no** — requires every D1 guard to pass.
- Disposition 3 (no AUC difference of 0.02 or more): **no** — guards fail.
- **Disposition 4: per-contrast tokens only**, every one `readiness_or_precision_insufficient`.
- #127 program claim: D1 is a program-set member that fails a guard, so it **vetoes** the intersection–union claim.
- Budget safeguard: does not fire (D1 − U is not `supported`); D1's mean training compute is 30.5 × U's.
- Tier-C trigger (D1 − U or D1 − J_dec `supported`): not met; Diffusion Forcing and γ-models are not Tier-C arms of
  this child (they run as P-coverage arms).

## Deviations and disclosures

1. **G-fid timing run relaunched (P-crash).** The first timing run crashed on a device bug in the port's optimizer
   state (fixed: `Agent.to` moves and rebinds the Adam moments). Its relaunch shared GPU 3 with another child's job
   while the host ran out of RAM (MemAvailable < 1 GB, swap full); THICK's timing came out at 19.4 s/update, a
   host-thrash artefact. It was stopped, archived (`fidelity/timing-run2-contaminated/`), and rerun once alone
   ([P-crash](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6386806922)); the clean run gave 37.8 GPU-h.
   The G-fid runs were then allowed to share GPUs again
   ([decision](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6389079169)).
2. **Exploration-cache race at G-fid launch (P-crash).** The six G-fid runs launched together and each lazily
   uncompressed the shared Plan2Explore episodes into one `.npy` cache; concurrent rewrites of memory-mapped files on
   the mergerfs volume made four runs fail within seconds (`OSError: [Errno 116] Stale file handle`); the two that
   started could not be proven to have loaded identical data. Fix (no change to training semantics): the cache is
   built once by a separate CPU step (temporary directory, fsync, rename, `MANIFEST.json` with per-array sha256), and
   runs open it read-only and refuse an incomplete or mismatching cache, logging the manifest sha256 in
   `replay_source.json`. Verification: 500 episodes, 3 000 arrays, 1 000 000 steps equal to the Plan2Explore episodes,
   0 mismatches (`fidelity/cache-verification.json`, manifest `sha256:ef527042…`). All six runs were restarted from
   scratch on the verified cache; the two running DreamerV2 runs were stopped, resumed once by the scheduler from their
   checkpoints, killed and deleted before the restart
   ([P-crash](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6398071674),
   [addendum](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6398164689)). **Correction to the
   addendum:** it says their discarded time is not counted; in fact the run-1 process seconds of dreamer s20260909 and
   s20260910 (about 0.42 h each) are included in their ledger entries (`gfid-dreamer-…-seed20260909/10`), and the
   killed run 2 (300 s each) is unrecorded. The G-fid result is unaffected.
3. **Vendored edit.** `thick/common/nets.py` upstream line 39, `self._rnn_type = rnn_type`, reads an undefined name and
   makes every non-THICK model (DreamerV2, Plan2Explore) crash at construction at `934980a`; it is deleted (listed in
   `VENDOR.json`; P-route). No other vendored file differs from upstream.
4. **Upstream requirements.** The official `requirements.txt` is not jointly installable (nevergrad vs numpy 1.19.5,
   GitPython vs typing-extensions, the deprecated `sklearn` stub). `b127-thick-tf` installs every pin except packages
   used only by MiniHack / Atari / DMC / MuJoCo / tuning, plus `GitPython==3.1.31` and `wandb==0.15.12`
   ([P-license](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382307800); `envs/`). The official
   THICK Dreamer ran on CPU (PinPadFour smoke, exit 0).
5. **G-fid design versus the paper.** 3 seeds per method (paper 7); one Plan2Explore exploration run (the paper trained
   two and kept the higher-return one); upstream behaviour kept otherwise, with these implementation differences:
   seeded and checkpointed RNGs, synchronous batches (upstream prefetches up to 5), no video / report passes, the
   exploration episodes memory-mapped and shared. THICK upstream also carries a `OneHotDist.get_logits` defect
   (returns probabilities); the port reproduces it.
6. **Numeric choices.** Training (selection, main seeds, D1-m, G-fid) is float32 with TF32 matmul / convolution on the
   GPU, the official stack's own choice on this GPU ([P-route](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382756572),
   [correction](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6383381590)). G-port is float32 on CPU.
   Every readout, selection-readout, scoring and validation path runs under `deterministic_scoring()`.
7. **Deterministic readout.** The rollout uses zero GateL0RD noise and latent modes (upstream `eval_ctxt_mean: True`);
   upstream's `ctxt_always_sample: True` would keep the gate noise on at evaluation.
8. **G-collapse definition.** Refined in the runner decision (before any training) from "decoder input at the
   endpoint" to #123's fit-frame posterior latent; the endpoint latent's statistics are reported beside it.
9. **No native success head** (no reward signal on NovPhy), hence no native-head score and no U + head control.
10. **Equal tuning allowance.** D1 chose among three `ctxt_sparsity` values on the policy split; U had no such
    tuning (#127). All three were near chance.
11. **`report.json` wall seconds of the G-fid runs** exclude each run's final process segment (written before the
    segment's seconds are added; dreamer runs show 0.0). The ledger (`ledger/gfid-*.json`) is the record used for
    compute; the G-fid result does not depend on it.
12. **GPU sharing.** Selection, G-fid and main training jobs ran with `shares_gpu`; process seconds on a shared GPU
    exceed the device time the work would need alone.

## Compute

GPU seconds are each process's wall time on its GPU (`eb.mark`, each arm-seed's training counted once from its
`report.json`; `compute.json`).

| item | process GPU-h |
| --- | --- |
| G-fid: Plan2Explore 1M steps | 8.53 |
| G-fid: THICK Dreamer × 3 (400k steps each) | 23.33 |
| G-fid: DreamerV2 × 3 (incl. ≈ 0.84 h of discarded runs, disclosure 2) | 10.86 |
| G-fid timing (incl. ≈ 5.6 h of the contaminated run, disclosure 1) | 5.79 |
| selection sparsity-10 / sparsity-25 | 12.19 / 12.10 |
| D1 seeds 20260908 (= selection sparsity-1) / 20260909 / 20260910 | 12.54 / 9.59 / 10.81 |
| D1-m × 3 | 1.55 |
| readouts (6), scoring (6), smoke, selection readouts | 0.12 |
| **recorded** (`compute.json`) | **107.42** |
| unrecorded: GPU probes 0.23 h; failed launches, killed runs, score-novelty, publish, validate, freeze ≈ 0.32 h | 0.55 |
| **active total** | **≈ 108.0 (child cap 146; smoke projection 106.2)** |

| | D1 | D1-m | U | C\* |
| --- | --- | --- | --- | --- |
| trainable parameters | 21 852 337 | 1 835 187 | 1 837 690 | 1 862 396 |
| training GPU-s per seed (08 / 09 / 10; process) | 45 126 / 34 531 / 38 928 | 2 004 / 1 554 / 2 013 | mean 1 294 | mean 1 024 |
| inference MACs per candidate (to frame 225) | 147.18 M | 15.82 M | 20.99 M | 27.26 M |
| wall per decision (16 candidates, GPU, median) | 0.026 s | 0.026 s | — | — |

#131 GPU processes ran in the windows 2026-10-24 20:25 → 25 04:00 (probes, timing), 25 04:00 → 06:50 and 25 18:00 →
26 19:15 (exploration, selection, G-fid), 26 18:00 → 27 05:20 (main training, readouts, scoring, publication,
validation), within the 48 h-on-4-GPUs rule. Engine seconds: 0. No scope was cut.

## Decisions (all on #131, also in `decisions.md`)

| decision | link |
| --- | --- |
| P-route: route (b), our PyTorch port (thix has no LICENSE) | [6382162858](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382162858) |
| P-license: vendor THICK @ 934980a (MIT) | [6382307800](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382307800) |
| P-fid: frozen G-fid band (VisualPinPad, Fig. 8) | [6382475412](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382475412) |
| P-other: NovPhy adapter | [6382505350](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382505350) |
| P-route: G-fid on the port (route ii) | [6382756572](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382756572) |
| P-cap: projection within the child share | [6382756903](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382756903) |
| P-other: D1 runner (guards, D1-m, exploratory views, gates) | [6383193006](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6383193006) |
| P-other: G-fid runner implementation | [6383312669](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6383312669) |
| P-other: training precision correction (TF32 on) | [6383381590](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6383381590) |
| P-crash: contaminated timing run | [6386806922](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6386806922) |
| P-novelty: descriptive novelty pass | [6386831318](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6386831318) |
| P-other: G-fid runs may share GPUs | [6389079169](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6389079169) |
| Program decision (#127): G-action in carrier space | [6396584893](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6396584893) |
| P-crash: exploration-cache race, relaunch from scratch | [6398071674](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6398071674) |
| P-crash addendum (two DreamerV2 runs) | [6398164689](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6398164689) |
| P-freeze: D1 selected sparsity-1 | [6402578948](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6402578948) |

## Reproduction

```
conda activate novphy && source env.sh
M="python -u -m scripts.run_issue_131_thick"
F="python -u scripts/b127_thick_fidelity.py"
TF=/srv/home/shua0171/miniconda3/envs/b127-thick-tf/bin/python
# Phase 0 (CPU): G-port
CUDA_VISIBLE_DEVICES="" $TF scripts/b127_thick_port_reference.py --reference .local-artifacts/issue-131-thick-v1/gport
CUDA_VISIBLE_DEVICES="" python scripts/b127_thick_port_reference.py --compare .local-artifacts/issue-131-thick-v1/gport          # exit 0
CUDA_VISIBLE_DEVICES="" $TF scripts/b127_thick_port_reference.py --agent-reference .local-artifacts/issue-131-thick-v1/gport/agent
CUDA_VISIBLE_DEVICES="" python scripts/b127_thick_port_reference.py --agent-compare .local-artifacts/issue-131-thick-v1/gport/agent  # exit 0
uv run --no-project --with pymupdf --with numpy python scripts/b127_thick_fidelity.py --band   # fidelity/band.json
# G-fid (scheduler, scripts/gpu_window_issue_131.json)
$F --gfid-timing                                              # exit 0 iff feasible
$F --run plan2explore --seed 20260908 --steps 1000000
P=.local-artifacts/issue-131-thick-v1/fidelity/runs/plan2explore-pinpad_four-seed20260908
CUDA_VISIBLE_DEVICES="" $F --build-cache $P && CUDA_VISIBLE_DEVICES="" $F --verify-cache $P   # exit 0
$F --run thick --seed S --steps 400000 --load-dir $P         # S = 20260908, 20260909, 20260910
$F --run dreamer --seed S --steps 400000 --load-dir $P
$F --gfid                                                     # fidelity/gfid.json
# selection, smoke, freeze, main runs (scripts/gpu_window_issue_131{,_main}.json)
CUDA_VISIBLE_DEVICES=g $M --selection --gpu q && $M --selection-readouts && $M --smoke && $M --prepare
CUDA_VISIBLE_DEVICES=g $M --train --gpu q                     # q = 0..3
CUDA_VISIBLE_DEVICES=g $M --readouts --gpu q && $M --freeze-readouts
CUDA_VISIBLE_DEVICES=0 $M --score                             # once
CUDA_VISIBLE_DEVICES=0 $M --score-novelty && $M --publish
CUDA_VISIBLE_DEVICES=0 $M --validate                          # exit 0
CUDA_VISIBLE_DEVICES="" pytest tests/test_issue_131_thick.py tests/test_issue_131_thick_gfid.py tests/test_issue_131_thick_runner.py   # 45 passed
```
