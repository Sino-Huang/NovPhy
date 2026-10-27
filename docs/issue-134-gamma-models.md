# Issue #134: γ-models on E99 carriers (#127 coverage arm T3) on NovPhy (reproduction, guards, and the frozen disposition)

Parent: #127 (external baselines program). Ticket: [#134](https://github.com/Sino-Huang/NovPhy/issues/134). The
binding shared rules of #85 and the #127 shared protocol (`scripts/external_baselines.py`, `eb`) apply. The Gate-B
(#108), #101, #104, #109, #112, #117 and #123 records and stores are read-only inputs. Nothing here edits the ICLR 2026
submission. T3 is a **coverage arm** (P-coverage; also a Tier-C arm if D1 − U or D1 − J_dec is `supported`). It is
secondary, it is reported, and it never enters the #127 program claim.

| | |
| --- | --- |
| Paper and code | Janner, Mordatch, Levine, *γ-Models: Generative Temporal Difference Learning for Infinite-Horizon Prediction*, NeurIPS 2020, arXiv 2010.14496; [jannerm/gamma-models](https://github.com/jannerm/gamma-models) @ `bb23e06` (MIT). The fidelity run also loads the released SAC policy through [jannerm/rlkit](https://github.com/jannerm/rlkit) @ `5d355ba` (MIT) and gym 0.17.3 |
| Runner | `scripts/run_issue_134_gamma_models.py` (project / timing / selection / selection-readouts / smoke / prepare / train / readouts / freeze-readouts / score / score-novelty / publish / validate) |
| Modules | `world_model/baselines/gamma_models/novphy.py` (NovPhy adapter); `world_model/baselines/gamma_models/vendor/` (the official `gamma` package: conditional neural spline flow, `BootstrapTarget`, `soft_update_from_to`, pendulum visualisation, and the Colab notebook `scripts/gamma-pendulum.ipynb`). `VENDOR.json` lists per-file upstream and vendored sha256. Edits: package-relative imports, the dropped video import, and the policy `torch.load(map_location, weights_only=False)` |
| Fidelity | `scripts/b127_gamma_models_fidelity.py` (Tier E: the official Colab notebook on the released pendulum pool and policy, Fig. 4 digitised); `scripts/b127_gamma_models_port_reference.py` (G-port reference and comparison) |
| Tests | `tests/test_issue_134_gamma_models.py` (29 tests, CPU) |
| Artifacts | `.local-artifacts/issue-134-gamma-models-v1/`: `plan.json`, `smoke.json`, `selection.json`, `decisions.md`, `findings.md`, `summary.json`, `compute.json`, `records-manifest.json`, `validation.json`, `readouts-manifest.json`, `projection.json`, `timing.json` (+ `timing-contended.json`), `gport/gport.json`, `fidelity/{band,gfid,projection}.json` + `fidelity/digitised/`, `envs/b127-gamma_models.txt`, `ledger/`. Checkpoints, readouts, records and the external downloads (`data/external/gamma_models/`, `MANIFEST.json`) stay local. Records and readouts are hashed in the manifests |
| Freeze | pre-freeze implementation `3b270722`; `plan.json` frozen 2026-10-27T13:32:33Z (00:32 AEDT, 28 Oct), commit `6be18dc4` (sha256 `4ef3ac5f…`, 131 pins). The freeze came after a passed smoke and before any main-seed training or evaluation-split read. No post-freeze amendment |
| Readout freeze | `readouts-manifest.json` frozen 2026-10-27T18:05:26Z (05:05 AEDT, 28 Oct; sha256 `8065d414…`), before any evaluation anchor was read |
| Scoring | once, in a process that did not train: scheduler run `134-score#1`, 2026-10-28 05:10–05:14 AEDT, exit 0 (3 arm-seeds). P-novelty pass `134-score-novelty#1`, 05:15–05:16, exit 0 (T3 failed its guards, so only the references were computed) |
| Validation | `134-publish-validate#1` (`--publish && --validate`), 2026-10-28 05:20–05:25 AEDT, **exit 0**: records hash-checked against `records-manifest.json` (sha256 `65970c95…`, 11 records), T3 seed 20260908 re-scored on every inventory with exact row equality, and the publication re-derived twice byte-identically. `validation.json` sha256 `41baa245…` |

Engine time is zero seconds. The primary metric is the Gate-B pooled grid AUC over the 311 mixed grid cells × 3 seeds
of the #104 sealed #109 evaluation split, with the tie-free cost under `deterministic_scoring()`. Intervals are
level-clustered percentile bootstraps and are DESCRIPTIVE. The margin is δ = 0.02.

## Outcome

**Disposition 4: per-contrast tokens only. No scoped ranking claim either way.**

- Every global guard passes: G-isolation, G-chronology (including the novelty pass), G-scoring-policy and G-records.
  G-port and G-fid pass.
- T3 **fails G-action on every seed**. The action sensitivity of the per-candidate expected occupancy carrier is
  0.0056 / 0.0265 / 0.0074 (seeds 08 / 09 / 10) against the frozen 0.05. Every other arm guard (G-rank, G-nonfinite,
  G-trained) passes on every seed.
- By the frozen rule every token involving T3 is `readiness_or_precision_insufficient`. Neither disposition 2 ("U ranks
  launches better than T3 (scoped)") nor disposition 1 applies. T3 is not a program-set member, so it does not touch
  the #127 program claim.
- Descriptively, T3 ranks the grid at 0.5102 [0.4919, 0.5281] against U's 0.7444 [0.7269, 0.7620]. This carries no token.

## Design summary

The design is frozen in `plan.json`. Every implementer decision was posted on #134 before the step it governs and is in
`decisions.md`.

**Model** ([adapter decision](https://github.com/Sino-Huang/NovPhy/issues/134#issuecomment-6391005347)). The official
conditional neural spline flow (`gamma.flows.make_conditional_flow`) is used at the Colab notebook's architecture
(hidden 3 × 256) and the function defaults (3 × [ActNorm, invertible 1×1 conv, RQ-spline coupling], K 16, tail bound
10, logistic prior). It runs over the 236-d E99 carrier, conditioned on `{s: carrier, a: launch5}`, with 10 230 696
trainable parameters. Training is the notebook's loop body unchanged: the bootstrapped target
(1 − γ)·N(s′, σ = 0.1) + γ·target-model(·|s′, a′), sample discount 0.9 (0.5 with γ = 0.5 for the first 2 000 updates),
MSE between model and target log-densities, Adam 1e-4 with weight decay 1e-5, Polyak τ = 0.005, batch 1 024 and
50 000 updates, in fp32.

- **Transitions.** Each fit shot gives the carriers at frames 0, 5, …, 225 (46 states; past a shot's end the final
  carrier repeats, as in #131) and 46 transitions. The launch is the action of the first transition; every later
  action and every next action is the zero no-op. The frame-225 carrier is absorbing. This gives 13 919 × 46 = 640 274
  transitions, drawn uniformly per batch from `eb.batch_rng(seed, step)`.
- **γ mapping.** A sample from (s0, a) is the state after k + 1 transitions with weight (1 − γ)γ^k. The absorbing
  frame-225 state carries the tail γ^44.
- **Score.** The expected tie-free `EndpointCosts` cost over S = 64 samples of p_γ(·|s0, a). All candidates share one
  fixed uniform base draw (common random numbers). A candidate with a nonfinite sample or a sampling assertion is a
  retained typed row; none occurred.
- **Guards.** G-action is computed on the per-candidate sample-mean carrier. G-collapse does not apply (the flow is a
  bijection of the carrier; there is no latent). Engine position error does not apply (the occupancy has no
  timestep). There is no native success head and no U + head control.

**Selection** (decision as above; #127 rule: policy-split pooled grid AUC of the primary score over 94 mixed policy
cells, seed 20260908, full budget each; the selected run is reused as main seed 20260908):

| configuration | γ | horizon 1/(1 − γ), transitions (frames) | policy grid AUC |
| --- | --- | --- | --- |
| **g0.95** | 0.95 | 20 (100) | **0.6375** (selected) |
| g0.978 | 1 − 1/45 | 45 (225) | 0.6212 |
| g0.99 | 0.99 (the notebook's γ) | 100 (500) | 0.6117 |

## G-fid and G-port

**G-fid, Tier E: PASS.** The paper reports no pendulum number. Its Fig. 4 (γ = 0.99 value estimates and value
iteration on a discretised pendulum) was digitised from the arXiv v4 PDF (page 9 rasters, inverse magma colour map,
max residual 1.5/255; orientation checked by Spearman 0.9999 between the digitised Rewards panel and the analytic
reward). The statistic is the Spearman ρ between a value map and the Ground-truth panel on `visualize_values`' 41 × 41
grid. The paper's own value map gives 0.7828. The band [0.633, 1.0] was frozen before the run
([band](https://github.com/Sino-Huang/NovPhy/issues/134#issuecomment-6390881384); `band.json` sha256 `38dfd895…`).

The official Colab notebook ran unchanged (50 000 steps, the released pool and SAC policy) on CPU (4 threads, 1.30 h)
in env `b127-gamma_models`. It gave **ρ = 0.6603** (0.6496 at n = 20; 0.7281 against the paper's own value map).
Deviations, all logged in `gfid.json`: device CPU; fixed seeds; the display-only in-loop plotting skipped; a resumable
checkpoint in place of `save_model`; the torch 2.13 deepcopy prior shim.

**G-port: PASS, bit-identical.** On fixed CPU float32 inputs, the official clone and the adapter ran in separate
processes on the same torch 2.13. Four arms were compared (the pendulum notebook geometry and the three carrier γ
values): the flow log-likelihood, sampling with a fixed generator, the bootstrapped loss, gradients, the target network
and two Adam steps (a burn-in update and a post-burn-in update). Every delta is exactly 0. Three injected defects were
detected:

- the sample discount set equal to γ;
- the launch passed as the next action;
- τ = 0.05.

## Results

### Guards

| arm | seed | G-rank (mean tied-pair share) | G-action (≥ 0.05) | G-nonfinite | G-trained | policy grid AUC (fit/policy only) |
| --- | --- | --- | --- | --- | --- | --- |
| T3 | 20260908 | pass (0.0005) | **fail (0.0056)** | pass (0.0) | pass | 0.6375 |
| T3 | 20260909 | pass (0.0001) | **fail (0.0265)** | pass (0.0) | pass | 0.4776 |
| T3 | 20260910 | pass (0.0) | **fail (0.0074)** | pass (0.0) | pass | 0.4307 |

G-collapse and G-readout do not apply. G-port and G-fid pass. Global guards: G-isolation, G-chronology, G-scoring-policy
and G-records all pass. The G-chronology timestamps are: plan 2026-10-27T13:32:33Z ≤ readouts 18:05:26Z ≤ evaluation
anchors 18:10:38Z ≤ first record 18:12:12Z. The novelty anchors were read at 18:15:57Z, after the last primary record
at 18:14:19Z.

The G-action threshold was frozen on #123's latent states and applies here, as frozen, to a 236-d carrier. #129's
fit-split calibration found that the engine truth itself stays below this threshold in carrier space
(`docs/issue-129-slotformer.md`, "G-action calibration context"). That is context only. The frozen guard, its failure
and the disposition stand.

### Grid rankers (311 mixed cells × 3 seeds)

| ranker | pooled AUC [CI] | per seed (08 / 09 / 10) | top-1 (chance 0.089) |
| --- | --- | --- | --- |
| U | 0.7444 [0.7269, 0.7620] | 0.7141 / 0.8011 / 0.7181 | 0.195 |
| C\* | 0.7834 [0.7676, 0.7992] | 0.8262 / 0.7334 / 0.7905 | 0.192 |
| P | 0.7203 [0.6965, 0.7442] | — | 0.180 |
| J | 0.7057 [0.6846, 0.7267] | 0.6690 / 0.7012 / 0.7470 | 0.160 |
| J_dec | 0.7460 [0.7277, 0.7637] | 0.7205 / 0.7933 / 0.7242 | 0.180 |
| **T3** | 0.5102 [0.4919, 0.5281] | 0.6302 / 0.4652 / 0.4352 | 0.030 |

T3 has no tied pairs and no nonfinite rows (0 typed rows). P's tied-pair share is 0.0022.

### Contrasts (grid, paired per (level, seed))

| contrast | estimate [CI] | token | reverse token |
| --- | --- | --- | --- |
| T3 − U | −0.2342 [−0.2610, −0.2074] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| T3 − C\* | −0.2732 [−0.2978, −0.2486] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| T3 − P | −0.2101 [−0.2293, −0.1908] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| T3 − J_dec | −0.2358 [−0.2641, −0.2077] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |

Every token is the stop token because T3 fails G-action. The estimates are reported as measured. They license no
statement in either direction.

### Wrong-anchor control (right − wrong, 480 fit anchors, grid)

| arm | right | wrong | right − wrong | ranking stability (Spearman) | token |
| --- | --- | --- | --- | --- | --- |
| T3 | 0.5102 [0.4919, 0.5281] | 0.5038 [0.4871, 0.5200] | +0.0064 [+0.0004, +0.0125] | 0.8928 | `readiness_or_precision_insufficient` |

### Descriptive inventories (no tokens)

| inventory (cells) | U | C\* | P | J | J_dec | T3 |
| --- | --- | --- | --- | --- | --- | --- |
| offset (228) | 0.6670 | 0.7748 | 0.5000 | 0.7522 | 0.6904 | 0.3129 |
| angle (234) | 0.6791 | 0.7590 | 0.8119 | 0.7765 | 0.6988 | 0.3712 |
| power (141) | 0.8058 | 0.8068 | 0.5000 | 0.8076 | 0.7943 | 0.3511 |

T3 − U: offset −0.3541 [−0.3971, −0.3113], angle −0.3080 [−0.3457, −0.2702], power −0.4546 [−0.5235, −0.3891]
(stop tokens). On these three inventories T3's pooled AUC is below 0.5: its expected cost orders the candidates
inversely to the engine verdicts.

### P-novelty (#110 novelty pairs)

T3 was not scored. P-novelty scores each arm that passes its guards, and T3 fails G-action (`novelty/arms.json`). The
U / C\* / P novelty references were computed and are hashed in `records-manifest.json` (`novelty/references/`). They
appear in `findings.md` as references only; there is no T3 row.

## Disposition (δ rule)

- Disposition 1 (T3 ranks launches better than U, scoped): **no**. T3 fails G-action, so T3 − U carries the stop token.
- Disposition 2 (U ranks launches better than T3, scoped): **no**. It requires every T3 guard to pass.
- Disposition 3 (no AUC difference of 0.02 or more): **no**. The guards fail.
- **Disposition 4: per-contrast tokens only**, every one `readiness_or_precision_insufficient`.
- #127 program claim: T3 is a coverage arm and not a program-set member. It neither supports nor vetoes the claim.
- Budget safeguard: does not fire (T3 − U is not `supported`). T3's mean training compute, in process seconds, is
  12.26 × U's.

## Deviations and disclosures

1. **Smoke cap basis (P-budget, before the smoke).** The selection runs shared GPUs with other jobs, so their process
   seconds per update (0.431 / 0.240 / 0.066 for g0.95 / g0.978 / g0.99) exceed the clean exclusive rate (0.0607,
   `timing.json`). The smoke's cap check was switched to exclusive-equivalent device time; the process basis is recorded
   beside it in `smoke.json`. This was posted before the smoke and before any selection result was read
   ([decision](https://github.com/Sino-Huang/NovPhy/issues/134#issuecomment-6438810533)).
2. **Two timing probes.** The first probe (25 Oct) ran on a busy GPU (0.456 s/update, kept as `timing-contended.json`
   and counted in the ledger). It was repeated once on an exclusive GPU (0.0607 s/update), and that rate is the funding
   projection. Funding followed the program's P-coverage decision
   ([#127](https://github.com/Sino-Huang/NovPhy/issues/127#issuecomment-6426185139)).
3. **Fidelity target constructed from a figure.** The band is derived from a digitised Fig. 4 (the paper reports no
   pendulum number). The run's ρ = 0.6603 lies inside the band, nearer its lower bound. The band, its margin and its
   rationale were frozen before the run.
4. **Training dynamics.** The official bootstrapped loss fluctuates strongly from step to step in all runs (as in the
   notebook's recipe). No run was stopped, repeated or tuned because of it, and none had a typed failure.
5. **Selected run vs later seeds.** The selected configuration's seed-20260908 run (policy AUC 0.6375) is reused as the
   first main seed (#127 rule). Seeds 20260909 / 20260910 at the same configuration reach 0.4776 / 0.4307 on the policy
   split and 0.4652 / 0.4352 on the sealed grid.
6. **Data scale.** Upstream trains on a 201 000-transition pendulum pool; here the fit split gives 640 274 carrier
   transitions with the same 50 000 × 1 024 update budget.
7. **Equal tuning allowance.** T3 chose its γ among three pre-declared values on the policy split; U had no such tuning
   (#127).
8. **Compatibility.** torch 2.13's `copy.deepcopy` breaks the copied logistic prior, so the target network shares the
   online prior (parameter-free; G-port covers it). The released policy snapshot is loaded with
   `map_location` / `weights_only=False` (one vendored edit).

## Compute

GPU seconds are each process's wall time on its GPU (`eb.mark`; each arm-seed's training counted once from its
`report.json`). Jobs ran with `shares_gpu`, so process time on a shared GPU exceeds the device time the work would need
alone.

| item | process GPU-h |
| --- | --- |
| selection g0.95 (= T3 seed 20260908) / g0.978 / g0.99 | 5.991 / 3.332 / 0.921 |
| T3 seeds 20260909 / 20260910 | 2.856 / 4.372 |
| timing probes (contended / exclusive) | 0.054 / 0.064 |
| selection readouts, readouts (3), scoring (3) | 0.117 |
| **recorded** (`compute.json`) | **17.71** |
| pendulum G-fid (CPU) | 0 (1.30 CPU-h) |

**Device vs process.** All 250 000 training updates at the exclusive rate (0.0607 s/update) amount to about 4.2 GPU-h
[estimate], within the smoke's 5.0 GPU-h exclusive-equivalent projection and the 12 GPU-h child cap. The ledger keeps
the process seconds.

| | T3 | U | C\* |
| --- | --- | --- | --- |
| trainable parameters | 10 230 696 | 1 837 690 | 1 862 396 |
| training GPU-s per seed (08 / 09 / 10; process) | 21 569 / 10 280 / 15 739 | mean 1 294 | mean 1 024 |
| inference MACs per candidate (64 occupancy samples) | 646.4 M | 20.99 M | 27.26 M |
| wall per decision (16 candidates, GPU, median) | 0.041 s | — | — |

Active wall time with a #134 GPU process running was about 11.0 h over two windows (selection 2026-10-27 18:00 →
28 00:10; main 28 00:35 → 05:25 AEDT), against the 48 h-on-4-GPUs rule. Engine seconds: 0. No scope was cut.

## Reproduction

```
conda activate novphy && source env.sh
M="python -u scripts/run_issue_134_gamma_models.py"
# Phase 0 (CPU)
CUDA_VISIBLE_DEVICES="" python scripts/b127_gamma_models_port_reference.py --run      # gport/gport.json, exit 0
python scripts/b127_gamma_models_fidelity.py --digitise && python scripts/b127_gamma_models_fidelity.py --write-band
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 /srv/home/shua0171/miniconda3/envs/b127-gamma_models/bin/python \
  scripts/b127_gamma_models_fidelity.py --run                                          # fidelity/gfid.json (CPU, ~1.3 h)
$M --project
CUDA_VISIBLE_DEVICES=g $M --timing                     # scheduler only (scripts/gpu_window_issue_134_timing.json)
# selection, smoke, freeze (scripts/gpu_window_issue_134.json; needs funded.json)
CUDA_VISIBLE_DEVICES=g $M --selection --gpu q          # q = 0, 1, 2 (g0.95, g0.978, g0.99)
$M --selection-readouts && $M --smoke && $M --prepare
# main runs (scripts/gpu_window_issue_134_main.json)
CUDA_VISIBLE_DEVICES=g $M --train --gpu q              # q = 0, 1
CUDA_VISIBLE_DEVICES=g $M --readouts --gpu q && $M --freeze-readouts
CUDA_VISIBLE_DEVICES=0 $M --score                      # once
CUDA_VISIBLE_DEVICES=0 $M --score-novelty
$M --publish && CUDA_VISIBLE_DEVICES=0 $M --validate   # exit 0
CUDA_VISIBLE_DEVICES="" pytest tests/test_issue_134_gamma_models.py   # 29 passed
```
