# Issue #128: direct success predictors (#127 arms A1, A2, A2-native, A3-sym, A3-img, A4) on NovPhy

Parent: #127 (external baselines program), question Q1. Ticket: [#128](https://github.com/Sino-Huang/NovPhy/issues/128).
The binding shared rules of #85 and the #127 shared protocol (`scripts/external_baselines.py`, `eb`) apply. The Gate-B
(#108), #101, #104, #109, #112 and #123 records and stores are read-only inputs. Nothing here edits the ICLR 2026
submission.

| | |
| --- | --- |
| Papers and code | A1 / A2: Bakhtin et al., *PHYRE: A New Benchmark for Physical Reasoning*, NeurIPS 2019, arXiv 1908.05656; [facebookresearch/phyre](https://github.com/facebookresearch/phyre) @ `d0765dd2f6d6cf41b933a3a57173c63d21fbfa6a` (Apache-2.0), ResNet from [pytorch/vision](https://github.com/pytorch/vision) v0.4.0 @ `d31eafa` (BSD-3-Clause). A4: Ahmed, Bakhtin, van der Maaten, Girdhar, *Physical Reasoning Using Dynamics-Aware Models*, arXiv 2102.10336; [facebookresearch/DynamicsAware](https://github.com/facebookresearch/DynamicsAware) @ `eb65576` (Apache-2.0, `main` = hand-crafted loss). A3: Xue et al., *Phy-Q*, Nature MI 2023, arXiv 2108.13696; [phy-q/benchmark](https://github.com/phy-q/benchmark) @ `514491c` (MIT) |
| Runner | `scripts/run_issue_128_direct_predictors.py` (selection / selection-readouts / throughput / smoke / prepare / select-deferred / train / readouts / freeze-readouts / score / score-novelty / publish / validate) |
| Modules | `world_model/baselines/phyre_dqn/{novphy.py, vendor/}` (A1, A2, A2-native); `world_model/baselines/dynamics_aware/{novphy.py, vendor/}` (A4); `world_model/baselines/phyq/{novphy.py, vendor/}` (A3-sym, A3-img). Each `vendor/VENDOR.json` (schema `issue_127_vendor_v1`) lists every file's upstream and vendored sha256 and its edits (package-relative imports; for PHYRE also the `torchvision` / `phyre.NUM_COLORS` import routes) |
| Fidelity / G-port | `scripts/b127_phyre_dqn_{fidelity,port_reference}.py` (A1 Tier E, A1/A2 G-port); `scripts/b127_dynamics_aware_{fidelity,port_reference}.py` (A4 Tier T, G-port); `scripts/b127_phyq_{fidelity,harness_check}.py` and the A3 G-port reference (A3) |
| Tests | `tests/test_issue_128_{direct_predictors,phyre_dqn,dynamics_aware,phyq}.py` (CPU; 56 passed at the last run), `tests/test_issue_127_amendments.py` |
| Artifacts | `.local-artifacts/issue-128-direct-predictors-v1/`: `plan.json`, `smoke.json`, `selection-deferred.json`, `decisions.md`, `readouts-manifest.json`, `records/`, `references/` (NN-px, NN-car), `novelty/`, `records-manifest.json`, `validation.json`, `summary.json`, `compute.json`, `findings.md`, `gport/` (+ `a3/`, `a4/`), `fidelity/` (+ `a3/`, `a4/`), `superseded-repair1/`, `throughput/`, `envs/`, `ledger/`. Checkpoints, readouts, records and the external downloads (`data/external/{phyre_dqn,dynamics_aware,phyq}/`, each with `MANIFEST.json`) stay local and are hashed in the manifests |
| Freeze | pre-freeze implementation `e5496e9a`; `plan.json` frozen 2026-10-25T09:26:24Z (20:26 AEDT), commit `a3decdc6` (sha256 `0cad4620…`, 185 pins), after a passed smoke and before any main-seed training or evaluation-split read. Three registered post-freeze amendments (A1 `23e71ee3`, A2 / A3 `3444cba1`) and the A3-sym repair 1, all before the evaluation result they could touch (see "Amendments and the A3-sym repair") |
| Readout freezes | group `primary` (A1, A2, A2-native, A4) 2026-10-28T15:50:27Z; group `a3` (A3-sym, A3-img) 2026-10-29T07:00:27Z; each before any evaluation anchor of its pass was read (`readouts-manifest.json`, sha256 `cc984c9f…`) |
| Scoring | once per group, in a process that did not train. `primary`: evaluation anchors read 15:55:39Z, records 15:56:29–15:57:14Z (29 Oct 02:56 AEDT), P-novelty pass (anchors 16:00:56Z, records 16:03Z). `a3`: records 2026-10-29T07:40Z (18:40 AEDT) |
| Validation | `--validate` exit 0 after each pass (primary 29 Oct ≈ 03:10 AEDT; `a3` 18:55 AEDT): 39 records hash-checked (18 arm-seed records, 2 NN references, 12 novelty records, 7 novelty references), every published table re-derived byte for byte, A2 seed 20260908 re-scored on its 311 grid cells with max \|Δ\| 0. `validation.json` binds `records-manifest.json` sha256 `29af62ae…`. Publication commits `0597bc46` (primary group) and `4c604b32` (A3 group; the primary records unchanged) |

The primary metric is the Gate-B pooled grid AUC over the 311 mixed grid cells × 3 seeds of the #104 sealed #109
evaluation split, under `deterministic_scoring()`. The score of a direct predictor is its success logit (Q for A3) and
the ranking cost is its negative (`eb.score_rows`). Intervals are level-clustered percentile bootstraps (10 000 draws,
PCG64 7201) and are DESCRIPTIVE. The margin is δ = 0.02.

## Question

Q1 of #127: does U (the pre-registered Gate-B unit, pooled grid AUC 0.744) rank launches better than direct success
predictors trained on the same 13 919 fit shots, which never roll anything out? The arms span the input (A1 pixels,
A2 the same E99 carrier U reads), trajectory supervision without rollout at ranking time (A4) and a domain-native
learner (A3, secondary). Pre-registered reading (#127): **A2 − U or A4 − U `supported`** → the paper cannot claim, for
this task, that the world model adds ranking value over direct prediction.

## Outcome

**Disposition 1 for A1, A2, A2-native and A4: "X ranks launches better than U on the sealed NovPhy grid (scoped)".
A3-sym and A3-img: disposition 4 (tokens only; native-task fidelity not reproduced).**

- Every global guard passes: G-isolation, G-chronology (both passes, the novelty pass included), G-scoring-policy and
  G-records. Every arm passes G-port (bit-exact against the official code) and G-rank, G-nonfinite (share 0) and
  G-trained on every seed. G-fid passes for A1 (Tier E), A2 / A2-native (inherited from A1) and A4 (Tier T); it fails
  for A3-sym and A3-img.
- **A1 ranks launches better than U on the sealed NovPhy grid (scoped):** A1 − U = +0.2502 [+0.2327, +0.2675],
  `supported`. The same holds for **A2** (the matched-input arm: same carrier, 1 838 781 parameters ≈ U's
  1 837 690): A2 − U = +0.2313 [+0.2131, +0.2493], `supported`; and for A2-native (+0.2296, `supported`).
- **A4** − U = +0.2501 [+0.2327, +0.2674], `supported`. The budget safeguard fires for A4 (its training compute is
  1.88 × U's), so its wording is budget-qualified: *"A4 at its native budget (1.88 × U's training GPU-h) ranks better
  than U at B1"*. The #140 U-native control (U trained to A4's compute) gave A4 − U-native = +0.1813 [+0.1658, +0.1967]
  with the stop token (U-native fails G-action on one seed), so **A4 stays budget-qualified** and no architectural
  reading of A4 − U is made. The safeguard does not fire for A1 (0.32 × U's training compute), A2 (0.04 ×) or
  A2-native (0.05 ×).
- By the pre-registered reading, A2 − U `supported` means the paper cannot claim, for this task, that the world model
  adds ranking value over direct prediction. A1, A2 and A4 reach disposition 1 and therefore **veto the #127
  program-level claim** ("U ranks launches better than every baseline in the set").
- Within family: A2 − A1 = −0.0190 [−0.0251, −0.0134] (A2 − A1 `not_supported_by_this_experiment`; A1 − A2
  `readiness_or_precision_insufficient`); A4 − A1 = −0.0002 [−0.0021, +0.0021], inside (−δ, +δ), not `supported`
  in either direction: the trajectory-distance auxiliary loss shows no AUC difference of 0.02 or more on this grid.
- **Memorisation context (descriptive, pre-registered before training).** The #109 fit and policy/evaluation levels
  share two generator templates. A parameter-free nearest-fit verdict copy, NN-px, reaches 0.9039 [0.8838, 0.9231] on
  the evaluation grid (NN-car 0.7773); A1 − NN-px = +0.0907 [+0.0719, +0.1105] and U − NN-px = −0.1595 (no tokens).
  On the #110 novelty pairs every arm drops sharply (A1: novel 0.578 vs paired normal 0.750).

## Design summary

The design is frozen in `plan.json`; every implementer decision was posted on #128 before the step it governs and is
in `decisions.md` (links at the end).

**Arms** (fit split only: 13 919 shots of 480 levels, one example per shot = (level anchor, launch, engine verdict);
seeds 20260908/09/10):

| arm | input | model (vendored) | role | trainable parameters |
| --- | --- | --- | --- | --- |
| **A1** DQN-px | frame 0 of the level's anchor branch, 320 × 240 RGB / 255 | PHYRE `ResNet18FilmAction`: ResNet-18, action MLP 1 × 256 ReLU on `eb.launch4` → FiLM gain / bias before `layer4`, max-pool, `Linear(512, 1)`; stem conv 3 → 64 instead of the 7-colour one-hot | primary, program set | 11 309 889 |
| **A2** DQN-car | the 236-d E99 anchor carrier | A1's `preprocess` / `_forward` / `ce_loss` and `FilmActionNetwork` over a 1 × 1-conv ResNet-18 stage ladder (base width 70), FiLM before the last stage | primary, program set (matched to U, +0.06 %) | 1 838 781 |
| A2-native | the carrier | the same trunk at base width 179 (A1-size) | secondary | 11 333 771 |
| **A4** DynDQN | as A1 | DynamicsAware `ResNetFilmAction` (Table-1 configuration: ResNet-18, FiLM before the last block, linear 512 → 256 embeddor, outer merge, 20-bin distance classifier) + the hand-crafted trajectory-distance auxiliary loss on the #123 fit-shot engine trajectories (15-frame grid, α = screen diagonal 1.25) | primary, program set | 12 751 957 |
| A3-sym | Phy-Q 12 × 120 × 160 symbolic state rebuilt from our engine's ground-truth JSON of the anchor (dev-mode labels) | Phy-Q `DQNRelationalSymbolic`; each launch → the nearest of the 180 native angle actions | secondary, never program set | 11 662 208 |
| A3-img | the anchor frame through the official image transform | Phy-Q `DQNRelationalImage` (ImageNet ResNet-18) | secondary, never program set | 11 633 920 |

**Training recipes** (official, verbatim through the vendored code, G-ported): A1 / A2 / A2-native class-balanced
batches of 64, BCE on the verdict, Adam 3e-4, `CosineAnnealingLR` stepped before each update, no weight decay,
clipping or augmentation. A4 the Table-1 recipe: 64 levels × 8 shots per batch (512), BCE + aux loss (weight 1), Adam
3e-4, cosine. A3 the released `TrainLearningAgent` loop: AdamW 3e-4 renewed at every outer step, batch 32,
class-balanced sampling, SmoothL1 on the terminal target, gradient clamp ±1. Batch b of every run is drawn from
`PCG64([seed, b])` with the official sampler rule (resumable).

**Selection** (seed 20260908, policy-split pooled grid AUC of the primary score over 94 mixed cells, `eb.select_configuration`;
the selected run is reused as main seed 20260908; [design decision](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382674719)).
The official update counts were set for PHYRE's ~10⁸ cached outcomes; on 13 919 examples the native schedule is
≈ 460 epochs (A1) and ≈ 3 700 (A4), so the native-budget arms tune training length (native, ÷10, ÷100) with the native
configuration always in the set; A2 (matched budget, 9 000 × 64) tunes the learning rate.

| arm | configurations (policy grid AUC) | selected |
| --- | --- | --- |
| A1 | native 0.9960 · **tenth 0.9964** · hundredth 0.9908 | tenth (10 000 × 64) |
| A2 | **lr3e-4 0.9732** · lr1e-4 0.9544 · lr1e-3 0.9704 | lr 3e-4 |
| A2-native | native 0.9657 · **tenth 0.9695** · hundredth 0.9423 | tenth |
| A4 (deferred: after its Tier-T G-fid passed, mechanical `--select-deferred`, 28 Oct) | native 0.9875 · **tenth 0.9985** · hundredth 0.9908 | tenth (10 000 × 512) |
| A3-img | native-code 0.9908 · **native-paper 0.9952** · native-template 0.9925 | native-paper |
| A3-sym, frozen (superseded by repair 1) | native-code 0.9872 · native-paper 0.9684 · native-template 0.9880 | (native-template, never scored) |
| A3-sym, re-selection after repair 1 (28 Oct) | **native-code 0.9941** · native-paper 0.9920 · native-template 0.9897 | native-code |

The policy-split AUCs are inflated for every arm alike by the fit/policy layout redundancy (disclosed before the
freeze; see below). U had no policy-split tuning of this kind (#127), disclosed.

## G-fid and G-port

**A1, Tier E: PASS.** The ten shipped PHYRE `dqn_10k` ball-within-template checkpoints, evaluated by the unmodified
official `agents/train.py` (10 000 cached actions ranked per task) in the source-built env `b127-phyre`: per-fold
AUCCESS 76.82 / 79.71 / 78.21 / 75.86 / 77.03 / 78.42 / 78.01 / 77.34 / 78.06 / 76.87, **mean 77.63** against the band
[75.4, 79.8] (paper 77.6 ± 1.1), frozen before the run
([band](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382404486)). Per fold the result is within
0.016 of the authors' shipped `results.json`. A2 and A2-native inherit A1's pipeline check (disclosed, #127).

**A4, Tier T: PASS.** Retrained with the ported official stack on PHYRE-B within-template folds 0–2 (train + dev →
test): AUCCESS 85.70 / 85.93 / 85.12, **mean 85.58** against the band [82.8, 87.2] (paper hand-crafted row
85.0 ± 1.1, ±2 SD), frozen before any run ([band](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382388098);
[result](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6452377738)). On 24 Oct the three folds were
killed by host-RAM exhaustion after the orchestrator co-scheduled them; they were relaunched once from their last
official checkpoints, at most two at a time ([P-crash](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6386813372)).

**A3, native-task fidelity (P-engine): both variants missed → A3 tokens only.** One released `TrainLearningAgent.py`
within-template run per variant on template 3_02_01 in Phy-Q's own game build: Relational-Symbolic solved 2 / 20 test
tasks (0.10; published 0.50, frozen band [0.174, 0.826]) and Relational-Image 4 / 20 (0.20; published 0.60, band
[0.280, 0.920]). No diagnosed port defect was found; the arm keeps its typed result `fidelity_not_reproduced`
([outcome](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6481885735)). A3 was still trained and scored
once (as frozen); every A3 token is the stop token.

**G-port: PASS, bit-exact for every arm.** On fixed CPU float32 inputs our adapters reproduce the official code's
forward pass, loss, gradients, LRs, parameters, BatchNorm buffers and optimizer state over three updates (A1 in the
official 7-colour configuration through the unmodified `neural_agent.train`; the A2 head path on A2 shapes; the native
and matched LR schedules at every update): every delta is exactly 0 (`gport/gport.json`, `gport/a4/gport.json`,
`gport/a3/gport.json`, the last re-run after repair 1).

## Amendments and the A3-sym repair

All four changes are registered in `scripts/post_freeze_amendments.py` against exactly the frozen bytes (and earlier
amendments); `load_plan` accepts only those bytes.

- **A1 (27 Oct, before A4's G-fid outcome; [decision](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6426445961)).**
  A scoring-group arm dropped with a typed reason under a posted decision (`dropped/<arm>.json`) is recorded as dropped
  and published like #130's C1-dino (stop tokens, disposition 4, veto), so A1 / A2 / A2-native could be scored even if
  A4 were dropped. Without a marker the outputs are byte-identical to the frozen runner (tested). **Not exercised:**
  A4 passed.
- **A3-sym repair 1 (28 Oct, P-fid, before any A3 record or A3 policy look; [decision](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6464616983)).**
  Phy-Q starts its jar with `--dev`, whose ground truth labels platforms `"Platform"`, and `SymbolicStateReader` drops
  them (channel 11 empty). Our store used non-dev labels and filled channel 11. The adapter now writes dev labels; both
  symbolic stores were rebuilt and G-port re-run. The A3-sym selection and main seeds were re-run on the corrected
  stores; the superseded runs are kept in `superseded-repair1/` and never scored (their GPU seconds stay in the ledger).
- **A2 (with repair 1).** A repaired arm re-runs its frozen selection (same configurations, seed, rule);
  `--select-deferred` records the mechanical re-selection (A3-sym: `native-code`, frozen choice `native-template`).
- **A3 (28 Oct, reporting only).** `--publish` sums `engine_seconds` from `ledger/engine-*.json` into `summary.json`
  and `compute.json` (per A3 variant); byte-identical when no engine file exists (tested).

## Results

### Arms (grid, 311 mixed cells × 3 seeds)

| arm | role | params | guards | grid AUC [CI] | per seed (08 / 09 / 10) | top-1 (chance 0.089) | X − U [CI] | token X − U | token U − X | disposition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | primary (program set) | 11 309 889 | pass | 0.9946 [0.9917, 0.9971] | 0.9950 / 0.9938 / 0.9951 | 0.959 | +0.2502 [+0.2327, +0.2675] | `supported` | `not_supported_by_this_experiment` | 1 |
| A2 | primary (program set) | 1 838 781 | pass | 0.9757 [0.9692, 0.9815] | 0.9770 / 0.9767 / 0.9734 | 0.798 | +0.2313 [+0.2131, +0.2493] | `supported` | `not_supported_by_this_experiment` | 1 |
| A2-native | secondary | 11 333 771 | pass | 0.9740 [0.9666, 0.9805] | 0.9739 / 0.9746 / 0.9736 | 0.791 | +0.2296 [+0.2114, +0.2474] | `supported` | `not_supported_by_this_experiment` | 1 |
| A4 | primary (program set) | 12 751 957 | pass | 0.9945 [0.9922, 0.9965] | 0.9946 / 0.9944 / 0.9945 | 0.939 | +0.2501 [+0.2327, +0.2674] | `supported` | `not_supported_by_this_experiment` | 1 (budget-qualified) |
| A3-img | secondary | 11 633 920 | fail (G-fid) | 0.9870 [0.9814, 0.9918] | 0.9901 / 0.9864 / 0.9845 | 0.934 | +0.2426 [+0.2244, +0.2610] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` | 4 |
| A3-sym | secondary | 11 662 208 | fail (G-fid) | 0.9700 [0.9597, 0.9791] | 0.9682 / 0.9723 / 0.9693 | 0.915 | +0.2256 [+0.2057, +0.2455] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` | 4 |

Reference rankers (from their published records): U 0.7444 [0.7269, 0.7620] (top-1 0.195), C\* 0.7834 [0.7676,
0.7992], P 0.7203 [0.6965, 0.7442], J 0.7057 [0.6846, 0.7267], J_dec 0.7460 [0.7277, 0.7637]. No arm has nonfinite
rows; tied-pair shares are 0 except A3-img / A3-sym (0.0002, launches mapped to the same native action) and P (0.002).

### Within family

| contrast | estimate [CI] | token | reverse token |
| --- | --- | --- | --- |
| A2 − A1 (input: carrier vs pixels) | −0.0190 [−0.0251, −0.0134] | `not_supported_by_this_experiment` | `readiness_or_precision_insufficient` |
| A4 − A1 (trajectory supervision) | −0.0002 [−0.0021, +0.0021] | `not_supported_by_this_experiment` | `not_supported_by_this_experiment` |

### Secondary contrasts (grid)

| arm | X − C\* | X − P | X − J_dec |
| --- | --- | --- | --- |
| A1 | +0.2113 [+0.1954, +0.2268] `supported` | +0.2743 [+0.2505, +0.2980] `supported` | +0.2487 [+0.2311, +0.2666] `supported` |
| A2 | +0.1923 [+0.1760, +0.2084] `supported` | +0.2554 [+0.2318, +0.2787] `supported` | +0.2297 [+0.2113, +0.2484] `supported` |
| A2-native | +0.1906 [+0.1745, +0.2066] `supported` | +0.2537 [+0.2297, +0.2772] `supported` | +0.2280 [+0.2096, +0.2463] `supported` |
| A4 | +0.2111 [+0.1954, +0.2267] `supported` | +0.2742 [+0.2504, +0.2977] `supported` | +0.2485 [+0.2309, +0.2665] `supported` |
| A3-img | +0.2036 (stop token) | +0.2667 (stop token) | +0.2410 (stop token) |
| A3-sym | +0.1866 (stop token) | +0.2497 (stop token) | +0.2240 (stop token) |

Every reverse token (reference − X) is `not_supported_by_this_experiment` for A1 / A2 / A2-native / A4.

### Descriptive inventories (no tokens)

| inventory (cells) | U | C\* | P | J_dec | A1 | A2 | A2-native | A4 | A3-img | A3-sym | NN-px | NN-car |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offset (228) | 0.6670 | 0.7748 | 0.5000 | 0.6904 | 0.9794 | 0.9680 | 0.9728 | 0.9811 | 0.5966 | 0.6033 | — | — |
| angle (234) | 0.6791 | 0.7590 | 0.8119 | 0.6988 | 0.9957 | 0.9890 | 0.9886 | 0.9972 | 0.9836 | 0.9694 | 0.9466 | 0.8114 |
| power (141) | 0.8058 | 0.8068 | 0.5000 | 0.7943 | 0.8041 | 0.8853 | 0.8982 | 0.8468 | 0.6117 | 0.3808 | — | — |

NN-px / NN-car are defined on grid and angle only (the fit split never executed offset or power launches). A3's
action mapping keeps only the launch angle (the nearest of 180 native actions), so candidates that differ in other
launch parameters can share one action.

### No-model nearest-fit references and the policy–fit redundancy (DESCRIPTIVE; pre-registered before the freeze)

**Disclosure before the freeze** ([decision](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6387263066)).
The #109 cohort has two generator templates, and every policy anchor has a near-duplicate fit anchor: mean
\|pixel diff\| to the nearest fit anchor p50 0.41 / p95 0.87 / max 1.11 grey levels, against p50 0.60 / p95 1.89 to the
nearest other policy anchor; 100 of 120 policy levels are closer to a fit level than to any other policy level. A
parameter-free nearest-fit verdict copy scored 0.946 (NN-px) and 0.810 (NN-car) on the policy grid. Frame 0 of a
level's branches differs only by capture noise (worst 24 pixels, mean \|Δ\| 0.002; tolerance 32 pixels / 0.01).

**NN-px / NN-car on the evaluation grid** (score = the engine verdict of the identical launch in the nearest fit level;
pixel L1 / carrier L2; seed-free):

| reference | pooled grid AUC [CI] | top-1 | tied-pair share |
| --- | --- | --- | --- |
| NN-px | 0.9039 [0.8838, 0.9231] | 0.807 | 0.176 |
| NN-car | 0.7773 [0.7502, 0.8038] | 0.572 | 0.391 |

| estimate (no tokens) | vs NN-px | vs NN-car |
| --- | --- | --- |
| A1 | +0.0907 [+0.0719, +0.1105] | +0.2174 [+0.1910, +0.2442] |
| A2 | +0.0717 [+0.0530, +0.0916] | +0.1984 [+0.1735, +0.2237] |
| A2-native | +0.0701 [+0.0512, +0.0898] | +0.1967 [+0.1719, +0.2219] |
| A4 | +0.0905 [+0.0718, +0.1103] | +0.2172 [+0.1908, +0.2441] |
| A3-img | +0.0830 [+0.0644, +0.1028] | +0.2097 [+0.1826, +0.2372] |
| A3-sym | +0.0660 [+0.0466, +0.0864] | +0.1927 [+0.1661, +0.2199] |
| U | −0.1595 [−0.1852, −0.1332] | −0.0329 [−0.0638, −0.0012] |

**Distance context** (computed by #141 on these published records, with distance strata cut on the policy cells
before #141's training; descriptive): A1 ranks the grid at 0.9916 / 0.9978 / 0.9943 in the near / mid / far strata of
distance to the nearest fit anchor frame (104 / 112 / 95 cells), NN-px at 0.9216 / 0.8903 / 0.9007, U at
0.7039 / 0.7425 / 0.7909. A1's level does not fall with distance to the fit split within this cohort.

### P-novelty (#110 novelty pairs; DESCRIPTIVE, no tokens, never in a disposition)

455 issue110-evaluation levels of #107's 29 in-scope cells, 257 mixed; 72 paired level pairs pooled over novelty
levels 1–7. A3 was not novelty-scored (fails G-fid).

| ranker | novel AUC [CI] (levels 1–7 pooled) | paired normal AUC | novel − normal [CI] | all normal sides |
| --- | --- | --- | --- | --- |
| A1 | 0.5776 [0.5269, 0.6272] | 0.7495 | −0.1392 [−0.2104, −0.0651] | 0.7739 |
| A2 | 0.5172 [0.4661, 0.5681] | 0.7400 | −0.1867 [−0.2629, −0.1122] | 0.7420 |
| A2-native | 0.5174 [0.4627, 0.5706] | 0.7629 | −0.2230 [−0.3015, −0.1448] | 0.7707 |
| A4 | 0.5690 [0.5127, 0.6237] | 0.7198 | −0.1160 [−0.2000, −0.0278] | 0.7512 |
| U | 0.4743 [0.4354, 0.5119] | 0.6562 | −0.1537 [−0.2127, −0.0960] | 0.6506 |
| C\* | 0.4551 [0.4148, 0.4950] | 0.6656 | −0.1693 [−0.2293, −0.1092] | 0.6654 |
| P | 0.6022 [0.5459, 0.6559] | 0.7501 | −0.1012 [−0.1781, −0.0274] | 0.7645 |

The direct predictors' ≈ 0.99 on the sealed grid falls to 0.52–0.58 on novel levels, the range of U, C\* and P; the
per-level table is in `findings.md`.

## Disposition (δ rule)

- A1, A2, A2-native, A4: **disposition 1** — "X ranks launches better than U on the sealed NovPhy grid (scoped)".
  Scope: the #109 sealed evaluation split, the Gate-B mixed grid cells (311 × 3 seeds), cost = −logit, the selected
  configurations above, U = the Gate-B unit F-15-macro.
- A4's wording is **budget-qualified** (safeguard fired, 1.88 × U's training GPU-h); after #140, A4 − U-native =
  +0.1813 carries the stop token, so the qualification stays and no architectural reading is made.
- A3-sym, A3-img: **disposition 4** (G-fid `fidelity_not_reproduced`; every token the stop token).
- **#127 program claim: vetoed** (A1, A2 and A4 reach disposition 1, not 2).
- Pre-registered Q1 reading: A2 − U `supported` → for this task the paper cannot claim that the world model adds
  ranking value over direct prediction.

## P-follow-up #141

The P-follow-up ([#141](https://github.com/Sino-Huang/NovPhy/issues/141), [result](https://github.com/Sino-Huang/NovPhy/issues/141#issuecomment-6492488967))
gave A2's FiLM classifier U's own predicted endpoints (F1 = frozen U + the A-family readout) beside the same classifier
without any rollout (F1-D). F1 − U = +0.2321 [+0.2135, +0.2501] was `supported` (budget-qualified: F1 includes U's
training), while F1 − F1-D = −0.0044 [−0.0081, −0.0014] was `not_supported_by_this_experiment` in both directions.
By its pre-registered reading the borrowed readout package helps relative to U's scorer, but U's predictions showed no
incremental ranking value over the anchor and launch alone; F1 − U+head = +0.1642 was `supported`.

## Deviations and disclosures

1. **Upstream pin.** The PHYRE sha in #127 / #128 (`d0765dd6f6d6…`) does not exist upstream; the pinned commit is the
   HEAD with the same short sha, `d0765dd2f6d6…`.
2. **Action MLP width 256**, as the official code, `dqn_10k` arguments and shipped checkpoints (which produced the
   paper's 77.6); the PHYRE paper text and the #127 arm text say 512
   ([decision](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382444832)).
3. **PHYRE build.** The pip wheel targets Python ≤ 3.9 and the documented conda recipe no longer solves; the simulator
   was built from source in `b127-phyre` (Python 3.11, thrift 0.11.0 C++ from source, `-include cstdio`), recipe in
   `envs/b127-phyre.md`. 1 000 cached outcomes re-simulated with 0 mismatches. The 2020 checkpoints load with
   `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (environment only).
4. **Inputs.** A1 / A4 read the 320 × 240 RGB anchor frame / 255 (a 3-channel stem conv in place of PHYRE's 7-colour
   256 × 256 one-hot); actions are `eb.launch4`. One training example per fit shot, observation = the level anchor.
5. **Training length as the tuning axis** for the native-budget arms (data-scale gap); A1, A2-native and A4 selected
   `tenth`. The policy-split AUCs used for selection are inflated by the fit/policy redundancy for every arm alike.
6. **Batch stream.** Batch b comes from `PCG64([seed, b])` with the official sampler rule instead of one sequential
   `RandomState(42)`; the rule is identical and a resumed run equals an uninterrupted one.
7. **A2 / A2-native G-fid** is inherited from A1's pipeline check (#127).
8. **A4 budget.** A4's native update costs 0.248 s at batch 512 on our input (6.9 GPU-h per native seed, above the
   3 GPU-h/seed cap); P-cap drew from unallocated program budget, no cut
   ([P-cap](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6386804015)). The selected `tenth` costs
   ≈ 0.67 GPU-h per seed. A4's Tier-T fidelity cost ≈ 41 GPU-h (P-cap, [decision](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382844952)).
9. **A4 trajectory mapping.** The #123 fit probe labels (slot presence and engine-projected centres every 15th
   position) stand in for PHYRE's featurised objects; destroyed objects count as maximally distant
   ([decision](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382423720)).
10. **A3 adaptations.** Symbolic state rebuilt from our engine's ground truth (dev labels after repair 1); launches
    mapped to the nearest of Phy-Q's 180 angle actions (offset and power are not representable); A3-img uses the clean
    image transform, while the released image pipeline stores the normalised image in a uint8 memmap
    ([state](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382972582),
    [actions](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382972743),
    [defects](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382972952)). The A3 fidelity harness ran
    on CPU with torch 2.13, at most 5 game instances at a time, one run per variant on 20 test tasks; symbolic attempt
    1 stopped on the torch ≥ 2.6 `torch.load` default and was relaunched once
    ([P-crash](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6479672825)).
11. **Host stall (24–25 Oct).** The A3-sym selection ran at 0.4–30 s per update for ≈ 4 h while the host was
    oversubscribed; the held GPU time is in the ledger (superseded runs, 7.5 GPU-h).
12. **A3 GPU scoring fix (pre-freeze).** The A3 adapter takes CPU observations and launches; the runner passes them
    on the CPU (found when the first A3 selection readouts failed on GPU; adapter unchanged at the time).
13. **Equal tuning allowance.** Every arm chose among three configurations on the policy split; U had no such tuning.
14. The evaluation split was scored by earlier tickets for our models; it is pristine for these arms.

## Compute

GPU seconds are each process's wall time on its GPU (`eb.mark`; each arm-seed's training counted once from its
`report.json`). Several jobs shared a GPU, so process time can exceed the device time the work needs alone.

| item | process GPU-h |
| --- | --- |
| A1 selection (3) + 2 seeds | 0.85 + 0.27 |
| A2 / A2-native selection + seeds | 0.08 / 0.19 |
| A4 selection (3) + 2 seeds (+ throughput probe) | 7.46 + 1.36 (+ 0.02) |
| A3-img selection + 2 seeds | 2.55 + 2.62 |
| A3-sym after repair 1: selection + 2 seeds | 1.09 + 0.43 |
| A3-sym superseded by repair 1 (selection + seeds) | 8.14 |
| readouts, scoring, novelty | ≈ 0.07 |
| **child ledger** (`compute.json`) | **25.12** |
| A1 Tier E (10 folds, evaluation only) | 0.24 |
| A4 Tier T (folds 0–2, process hours; plus ≈ 0.8 unrecorded in the killed segments) | 41.28 |
| **total** | **≈ 66.6** (child cap 66 + P-cap draws) |

**Engine time** (CPU game-server worker-hours, A3 native-task fidelity; `summary.json` `engine_seconds` 22 686.5):
**6.30 h** (A3-sym 3.23 h, of which 0.94 h in the stopped attempt 1; A3-img 3.07 h), within the shared 190 worker-h
allowance with E1.

| | A1 | A2 | A2-native | A4 | A3-img | A3-sym | U |
| --- | --- | --- | --- | --- | --- | --- | --- |
| training GPU-s per seed (08 / 09 / 10) | 269 / 487 / 486 | 48 / 58 / 63 | 54 / 68 / 75 | 2 425 / 2 429 / 2 450 | 6 452 / 4 482 / 4 962 | 742 / 792 / 737 | mean 1 294 |
| ratio to U (mean) | 0.32 | 0.04 | 0.05 | 1.88 | 4.09 | 0.59 | 1 |
| MACs per candidate | 805 M | 1.43 M | 8.74 M | 805 M | 45.2 M | 53.7 M | 20.99 M |
| wall per decision (16 candidates, GPU, median) | 4.3 ms | 3.5 ms | 7.0 ms | 4.3 ms | 3.7 ms | 1.7 ms | — |

MACs are FlopCounter counts of scoring 16 candidates from one anchor (the trunk once, the FiLM block and head per
candidate), divided by 16.

## Decisions (on #128, in order)

- P-license: [A4](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382269414), [A3](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382277009), [A1/A2](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382309351)
- A4: [P-route](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382387919), [Tier-T band](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382388098), [trajectory mapping](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382423720), [Tier-T P-cap](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382423910), [measured Tier-T P-cap](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382844952), [CPU threads](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6383201139), [native-budget P-cap](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6386804015), [P-crash](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6386813372), [Tier-T pass](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6452377738)
- A1/A2: [Tier-E band](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382404486), [adapter choices](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382444832), [selection design and projection](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382674719)
- A3: [state source](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382972582), [action mapping](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382972743), [upstream defects](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382972952), [recipe and configurations](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382973132), [P-engine design](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6388391126), [funding](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6464530188), [repair 1](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6464616983), [harness](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6464821176), [P-crash](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6479672825), [image funding](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6480740764), [fidelity outcome](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6481885735)
- Program: [P-novelty](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6386830623), [frame-0 tolerance, redundancy, NN references](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6387263066) ([addendum](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6388000077)), [drop-marker amendment](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6426445961) ([registered](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6426667241))

## Reproduction

```
conda activate novphy && source env.sh
M="python -u -m scripts.run_issue_128_direct_predictors"
# Phase 0 (A1/A2; A4 and A3 have their own fidelity / port scripts)
/srv/home/shua0171/miniconda3/envs/b127-phyre/bin/python scripts/b127_phyre_dqn_fidelity.py         # fidelity/gfid.json (GPU)
CUDA_VISIBLE_DEVICES="" /srv/home/shua0171/miniconda3/envs/b127-phyre/bin/python scripts/b127_phyre_dqn_port_reference.py reference
CUDA_VISIBLE_DEVICES="" python scripts/b127_phyre_dqn_port_reference.py compare                       # gport/gport.json, exit 0
# selection, smoke, freeze (night window only, via the scheduler)
$M --selection --arms A1,A2,A2-native && $M --selection-readouts --arms A1,A2,A2-native
$M --selection --arms A3-sym,A3-img && $M --selection-readouts --arms A3-sym,A3-img
$M --throughput --arms A4 && $M --smoke && $M --prepare
# after the freeze
$M --train --gpu q                                      # q = 0..3 (frozen queues)
$M --selection --arms A4 && $M --selection-readouts --arms A4 && $M --select-deferred   # after A4's G-fid
$M --readouts --gpu q
$M --freeze-readouts --arms A1,A2,A2-native,A4 && $M --score && $M --score-novelty && $M --publish && $M --validate
$M --freeze-readouts --arms A3-sym,A3-img && $M --score --arms A3-sym,A3-img && $M --publish && $M --validate
CUDA_VISIBLE_DEVICES="" pytest tests/test_issue_128_direct_predictors.py tests/test_issue_128_phyre_dqn.py \
    tests/test_issue_128_dynamics_aware.py tests/test_issue_128_phyq.py
```
