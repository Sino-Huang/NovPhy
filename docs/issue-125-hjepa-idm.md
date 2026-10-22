# Issue #125: H-JEPA with its inverse-dynamics remedy on NovPhy

Parent: #108. Ticket: [#125](https://github.com/Sino-Huang/NovPhy/issues/125) (dump: `round2/issues/125.md`),
pre-registered 2026-10-19 before any #123 evaluation-split read, as the follow-up to
[#123](https://github.com/Sino-Huang/NovPhy/issues/123) (`docs/issue-123-hjepa-comparison.md`). The #85 rules and the
round-2 conventions apply. Nothing here edits #123 or the ICLR 2026 submission.

Status (2026-10-23): **published and validated.**

| | |
| --- | --- |
| Freeze | `plan.json` frozen 2026-10-21T07:26:34Z (18:26:34 AEDT), commit `c9d0b64c`, after the trigger, the γ₁ selection and a passed smoke, and before any #125 main training or evaluation read. Post-freeze amendments A1 (`d655ee35`) and A2 (`cae21709`), registered in `scripts/post_freeze_amendments.py`; the compute aggregation comes from #123 A3 (`aa8e702f`, disclosed in `558f63e1`) |
| Training | 12 arm-seeds over two night windows (2026-10-21 18:30 → 10-22 06:50, paused with `resume.pt`; 10-22 18:00 → 10-23 00:47 AEDT); no typed failure |
| Readout freeze | `readouts-manifest.json` frozen 2026-10-22T14:00:14Z (01:00 AEDT, 23 Oct), before any evaluation anchor was read (14:08:55Z) |
| Scoring | once: scheduler run `125-score#1`, 2026-10-23 01:05:06–01:10:44 AEDT, exit 0 (12 arm-seeds) |
| Publication | `125-publish#2`, 01:30:08–01:32:10 AEDT, exit 0 (`125-publish#1` died with SIGSEGV, see Disclosures) |
| Validation | `python -u -m scripts.run_hjepa_idm --validate`, scheduler run `125-validate#1`, 01:35:06–01:39:10 AEDT, **exit 0**: 18 records hash-checked (12 arm-seeds + 6 U / C\* curve records), every table re-derived and byte-compared, HJ-car-m-IDM seed 20260908 re-scored on 311 grid cells and 480 wrong anchors, max delta 0.0 (tie-free, pig presence, pig displacement). `validation.json` binds `records-manifest.json` sha256 `108cc617…6f74e25` |

The primary metric is #123's: the Gate-B pooled grid AUC over the 311 mixed grid cells × 3 seeds of the #104 sealed
#109 evaluation split, tie-free cost, R1 readout, under `deterministic_scoring()`. Intervals are level-clustered
percentile bootstraps (10 000 draws, PCG64 7201) and are DESCRIPTIVE. The margin is δ = 0.02. Engine time is zero.

## Outcome

**Disposition 3: per-contrast tokens only (outcome branch: no scoped ranking claim). Disposition 4 holds: the paper's
own IDM + action-SIGReg remedy does not prevent the collapse.** Both primary arms, HJ-px-IDM and HJ-car-m-IDM, still
fail G-collapse and G-action on every seed.

- Every global guard passes: G-fid, G-port, G-port-IDM, G-isolation, G-chronology, G-scoring-policy and G-records.
- Every IDM arm fails at least one arm guard, so by the frozen rule every token, primary or secondary, in both
  directions, is `readiness_or_precision_insufficient`.
- Disposition 2 ("our Gate-B model ranks better than H-JEPA with its remedy (scoped)") needs every guard to pass. It
  does not apply. This document makes no "better than H-JEPA" or state-of-the-art statement.

## Design summary

The design is frozen in the ticket and `plan.json`; the implementer decisions are in `decisions.md` (posted on #125).
It tests H-JEPA with the paper's own anti-collapse remedy: an inverse-dynamics (IDM) term and action-SIGReg (§4.3 Eq. 8,
App. K, the DROID recipe). Only the IDM terms change from #123's frozen plan: data and splits, H3-Δ layout,
anchor-mixture sampler, launch encoding, λ = 0.18, optimizer and schedule, R1 readout, cost, cells, bootstrap and
guards are #123's.

| arm | input | model | budget | role | parameters |
| --- | --- | --- | --- | --- | --- |
| **HJ-px-IDM** | pixels 320 × 240 | H-JEPA H3-Δ + IDM + action-SIGReg, paper sizes | B3 (128 045 × 128) | primary, faithful + remedy | 35 860 884 |
| **HJ-car-m-IDM** | E99 carriers | as above, matched to U (IDM heads included) | B1 (9000 × 64) | primary, controlled + remedy | 1 836 500 |
| LeWM-px-IDM | pixels | flat LeWM + level-1 IDM (DROID LeWM+IDM) | B3 (150 880 × 128) | temporal lever under the remedy | 11 563 900 |
| LeWM-car-m-IDM | carriers | flat + IDM, matched | B1 (9000 × 64) | temporal lever, carriers | 1 835 196 |

Seeds 20260908/09/10, trained on the #109 fit split only. Contrasts (δ = 0.02, #123's estimator and token rule): PI1
HJ-px-IDM − U, PI2 HJ-car-m-IDM − U; SI1 / SI2 the IDM effect against #123's published HJ-px / HJ-car-m records (never
re-scored); SI3 / SI4 HJ-IDM − LeWM-IDM; SI5–SI7 against C\*, P, J and J_dec; SI8 the wrong-anchor control. Guards: every
#123 guard per arm, G-fid inherited from #123, and G-port-IDM. A failed guard turns every token involving the arm into
`readiness_or_precision_insufficient`. Dispositions 1–4 are quoted in the Disposition section. Cap: 120 GPU-h of active
compute.

## Files

| path | role |
| --- | --- |
| `scripts/run_hjepa_idm.py` | runner (modes below) |
| `world_model/hjepa/idm.py` | IDM arms, configs, heads, matched sizes |
| `scripts/hjepa_idm_port_reference.py` | official side of G-port-IDM (hjepa env, CPU) |
| `tests/test_issue_125_hjepa_idm.py`, `tests/test_issue_125_amendments.py` | CPU tests (22 pass) |
| `scripts/gpu_window_issue_125.json` | scheduler jobs (selection, smoke, main jobs); `125-trigger` is in `scripts/gpu_window_issue_123.json` |
| `.local-artifacts/issue-125-hjepa-idm-v1/` (tracked) | `plan.json`, `trigger.json`, `trigger-parts/`, `gport-idm.json`, `selection/` (`report.json`, `readouts.json`, `train-log.jsonl`), `smoke.json`, `findings.md` (every table), `summary.json`, `compute.json`, `records-manifest.json`, `readouts-manifest.json`, `validation.json`, `ledger/`, `logs/` (including copies of the scheduler's #125 run logs, `logs/scheduler-125-<job>.log`, and #125 events, `logs/scheduler-125-events.jsonl`) |
| `.local-artifacts/issue-125-hjepa-idm-v1/` (local only) | checkpoints (`models/`, `selection/*/model.pt`), readout weights, `gport-idm/*.pt`, `data/`, the 18 records (hashed in `records-manifest.json`; readouts in `readouts-manifest.json`) |
| `.local-artifacts/issue-125-hjepa-idm-v1/decisions.md` | implementer decisions (posted on #125) |

Nothing of #123 is edited. The runner imports `scripts/run_hjepa_comparison.py`, `world_model/hjepa/novphy.py` and
the vendored official code, and reads #123's data stores, checkpoints and frozen plan.

## Method

- **IDM.** The official `loss.InverseDynamicsModel` (`MLP(2D → 256 → 256 → D)`, ReLU) inside the official
  `loss.InverseDynamicsLoss`, `mean ‖I(z_t, z_{t+1}) − sg[a_t]‖²`, on every level. The vendored `hjepa_forward`
  adds it with the config key `idm_coeff`. The target is the level's action embedding:
  - level 1: the embedded launch on the launch transition, and the embedded no-op elsewhere;
  - levels ≥ 2: the pooled macro-action.
- **Action-SIGReg.** The official `SIGReg()` on the macro-action embeddings of levels ≥ 2, with weight 0.005.
- **Weights.** γ₂ = γ₃ = 50. γ₁ ∈ {100, 225} is selected per input type. λ = 0.18.
- **Heads.** The heads are attached after #123's model is built, because `create_world_model` reads
  `predictor.action_dim` and `ARPredictor` does not accept that key. They use the official classes and arguments.
  With the same seed, the backbone initialisation equals the #123 arm's.
- **LeWM arms.** IDM at level 1 only, with no action-SIGReg. This is the DROID LeWM+IDM baseline.
- **Matched carriers.** Total parameters, IDM heads included, are matched to the Gate-B unit (1 837 690, ±2 %).
  Only the predictor MLP width changes from #123's matched sizes:
  - HJ-car-m-IDM: 448 → 96, total 1 836 500;
  - LeWM-car-m-IDM: 480 → 384, total 1 835 196.
- **Training.** #123's loader, optimizer, schedule and SIGTERM stop: `resume.pt` is written at the next optimizer
  step and the run exits 75. Resume is bitwise exact on CPU (tested). `model.pt` uses #123's schema, so #123's
  readouts and scoring load the backbone; the IDM heads are stored beside it.

## G-port-IDM (CPU, 2026-10-19T06:16:37Z): pass

The official code with the remedy (hjepa env, torch 2.7.1) and the port (torch 2.13) were compared on fixed CPU
inputs under #123's declared tolerances. The comparison covers the forward pass with every IDM and action-SIGReg
term, the gradients (the IDM heads included), the step-0-LR step and a positive-LR step (AdamW state, updated
parameters). All four IDM arms pass:
- worst relative loss delta 2.9e-6;
- worst per-tensor gradient relative L2 4.4e-5;
- every IDM / action-SIGReg term within 1.2e-7 of the official value.

Details are in `gport-idm.json`.

## Trigger

`--trigger` loads the trained #123 HJ-px and HJ-car-m `model.pt`, seeds 20260908/09/10 (`resume.pt` is never
used). It computes G-collapse and G-action with #123's readout code, guard expressions and thresholds, on fit probe
frames and fit anchors only:
- G-collapse: top-level effective rank ≥ 0.1·D and within-episode variance share ≥ 0.05;
- G-action: endpoint launch sensitivity ≥ 0.05.

The ticket is triggered iff any seed of either arm fails. The result is written to `trigger.json`. No evaluation
file is read. The trigger runs only before #123's first evaluation read. If #123 records, evaluation anchors or the
evaluation position cache exist at its start or its end, it writes `trigger-refused.json` instead and exits 1.

### Trigger record (`trigger.json`): triggered

Evaluated 2026-10-20T18:59:24Z (05:59 AEDT on 21 Oct), scheduler run `125-trigger#1` (exit 0), before #123's first
evaluation read. #123's state at its start and its end: 0 records scored, no evaluation anchors read, no evaluation
position cache. Data: every 15th position 0–285 of every fit shot; the 480 #112 retrain fit anchors × the 16 grid
launches; 0 evaluation files read. All six #123 primary arm-seeds fail both guards:

| arm | seed | erank / D (≥ 0.1·D) | χ (≥ 0.05) | endpoint launch sensitivity (≥ 0.05) | G-collapse | G-action |
| --- | --- | --- | --- | --- | --- | --- |
| HJ-px | 20260908 | 111.7 / 192 | 0.0014 | 1.14e-06 | fail | fail |
| HJ-px | 20260909 | 112.7 / 192 | 0.0015 | 1.14e-06 | fail | fail |
| HJ-px | 20260910 | 113.0 / 192 | 0.0015 | 1.14e-06 | fail | fail |
| HJ-car-m | 20260908 | 14.2 / 112 | 0.0167 | 1.71e-07 | fail | fail |
| HJ-car-m | 20260909 | 13.8 / 112 | 0.0167 | 1.66e-07 | fail | fail |
| HJ-car-m | 20260910 | 14.1 / 112 | 0.0161 | 1.67e-07 | fail | fail |

Both arms pass the effective-rank half of G-collapse and fail its χ half.

## γ₁ selection (policy split; frozen in `plan.json`)

HJ-px-IDM and HJ-car-m-IDM at B1 (9000 × 64), seed 20260908, γ₁ ∈ {100, 225}, #123's readouts; rule: the largest
policy-split pooled grid AUC under R1 (94 mixed policy grid cells), ties at 4 d.p. → 100 (the DROID value); the
selected γ₁ applies to every IDM arm of that input type. Records: `selection/<arm>/gamma1-<γ>/seed-20260908/`
(`report.json`, `readouts.json`, `train-log.jsonl`).

| arm | γ₁ = 100 | γ₁ = 225 | selected |
| --- | --- | --- | --- |
| HJ-px-IDM | 0.4830 | **0.5170** | **225** (pixels) |
| HJ-car-m-IDM | 0.4275 | **0.4406** | **225** (carriers) |

There is no tie. The selected AUCs are near or below chance: this is the pre-declared rule, not a quality claim.

## Schedule

Night window only (18:00–07:00 Australia/Melbourne), through `scripts/gpu_window.py`:
1. `125-trigger` runs after all four `123-train-q*` queues, on GPU 1 beside `123-readouts-g1`. `123-score` depends
   on it, so #123 scores only after the trigger.
2. `125-select-q0/q1/q2` run on GPUs 1, 2 and 3.
3. `125-smoke` runs on GPU 1.

Before the freeze, #123's jobs kept priority and no #125 job used GPU 0. After the freeze (`--prepare`, an agent
commit), the main jobs were enqueued:
4. `125-train-q3` (LeWM-px-IDM × 3, 18.1 GPU-h) runs on GPU 1, `125-train-q1` on GPU 2, `125-train-q2` on GPU 3 and
   `125-train-q0` on GPU 0, once `123-validate` frees it. Each is resumable: SIGTERM writes `resume.pt` and the
   job exits 75.
5. `125-readouts-g0..g3` run after all four training queues, then `125-freeze-readouts`.
6. `125-score` runs after `125-freeze-readouts` and `123-validate`, because SI1 and SI2 read #123's validated
   publication. `125-publish` and `125-validate` follow.

Projection: 62.1 GPU-h against the 120 GPU-h cap, 18.9 h wall, 2 nights.

### Chronology

`plan.json` was frozen after #123's evaluation scoring (18:03–18:10 AEDT on 21 Oct) but before the #125
implementer read any #123 result. Every frozen choice comes from the ticket (pre-registered 2026-10-19), the
policy-split γ₁ selection, or the fit-split trigger and smoke. The trigger was evaluated before #123's first
evaluation read. Details are in `decisions.md`.

### Post-freeze amendment A1 (night 1, before any #125 readout or evaluation read)

- **Ledger.** `--train` now adds an arm-seed's seconds to the compute ledger only if its `model.pt` did not exist
  before the call. The frozen code counted a finished run again whenever its queue relaunched.
- **Plan loading.** `load_plan` no longer hash-checks the shared registry `scripts/post_freeze_amendments.py`, which
  the frozen plan had pinned.

A1 is registered in that registry and covered by tests. Nothing that produces evidence changed (`decisions.md`).

### Compute aggregation (inherited from #123 A3, `aa8e702f`; no #125 runner change)

`compute_tables` sums `rc.ledger_totals` from the pinned #123 runner. #123's amendment A3 (`045530a9…` → `86e8ee9e…`,
registered; the #125 plan's pinned hash is in its `replaces`) makes it count each arm-seed's training once: a
`train-<arm>-seed<s>` phase counts its `models/<arm>/seed-<s>/report.json` seconds (the elapsed time carried across
resumes, exactly what one completion marks). A ledger entry that differs keeps its raw seconds as
`ledger_gpu_seconds` / `ledger_wall_seconds` and is not counted. Other phases count their ledger seconds.

### Post-freeze amendment A2 (pre-scoring review, 2026-10-22; validation only, before any #125 readout or evaluation read)

- **Inherited pins.** `load_plan` hashed #123's `plan.json` but never ran #123's `load_plan`, so the inputs that
  plan pins (scoring harness, vendored H-JEPA code, fit anchors and verdicts, Gate-B and #101 reference records)
  were not checked. After #125's own pins, `load_plan` now runs `rc.load_plan` on #123's output. It accepts #123's
  registered amendments (A1–A3) and nothing else, so every `--train`, `--readouts`, `--freeze-readouts`, `--score`,
  `--publish` and `--validate` refuses if an inherited input drifts.
- **Frozen guard reports.** `--freeze-readouts` records each arm-seed's `readouts.json` hash (`report_sha256`), but
  nothing checked it, and `arm_guards` reads the live reports. A #125 `load_manifest` now wraps #123's (`model.pt` /
  `readouts.pt` hashes) and also verifies every report hash, typed failures included. `--score`, `compute_tables`
  (`--publish` / `--validate`) and the spot check use it. A report-only change therefore stops scoring and
  publication. In `--validate` it withdraws the publication and writes a failed receipt, and the job exits non-zero.
- **Smoke.** `--smoke` refuses once `plan.json` exists. `smoke.json` is a frozen input, and a re-run would add its
  seconds to the smoke ledger phase again. A pre-freeze re-run would add the seconds it really spends.

A2 is registered (`run_hjepa_idm.py` `56e7cc65…` → `d53e44d0…`) and tested in `tests/test_issue_125_amendments.py`.
Training, resume, RNG, data, readout fitting, scoring math and tokens are unchanged (`decisions.md`).

## Results

### Guards

| global guard | pass |
| --- | --- |
| G-fid (inherited from #123) | True |
| G-port (inherited from #123) | True |
| G-port-IDM | True |
| G-isolation | True |
| G-chronology | True |
| G-scoring-policy | True |
| G-records | True |

G-chronology: trigger 2026-10-20T18:59:24Z ≤ plan freeze 10-21T07:26:34Z ≤ readout freeze 10-22T14:00:14Z ≤
evaluation anchors read 14:08:55Z ≤ first record 14:09:00Z. G-records: `validation.json` binds the current records
manifest (spot check above).

Arm guards (#123's `arm_guards` and thresholds; G-readout uses the arm's 3-seed mean policy ceiling, threshold 0.65;
G-nonfinite threshold 0.10):

| arm | role | params | G-collapse | G-action | G-readout (policy ceiling) | G-nonfinite (R1 / R2 share) | trained | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HJ-px-IDM | primary, faithful + remedy | 35,860,884 | False | False | True (0.8562) | True (0.000 / 0.000) | True | False |
| HJ-car-m-IDM | primary, controlled + remedy | 1,836,500 | False | False | False (0.5623) | True (0.000 / 0.000) | True | False |
| LeWM-px-IDM | temporal lever under the remedy (pixels) | 11,563,900 | True | True | False (0.5698) | True (0.000 / 0.000) | True | False |
| LeWM-car-m-IDM | temporal lever under the remedy (carriers) | 1,835,196 | False | False | False (0.6193) | True (0.000 / 0.000) | True | False |
| HJ-px (#123, for SI1) | | | | | | | | False |
| HJ-car-m (#123, for SI2) | | | | | | | | False |

Failing arm guards: HJ-px-IDM: G-collapse, G-action. HJ-car-m-IDM and LeWM-car-m-IDM: G-collapse, G-action, G-readout.
LeWM-px-IDM: G-readout. #123's HJ-px and HJ-car-m fail their #123 arm guards (#123's global guards and G-records pass).

Per-seed diagnostics (top level; G-collapse needs both the rank and the χ threshold):

| arm | seed | erank / D (≥ 0.1·D) | χ within-episode share (≥ 0.05) | endpoint launch sensitivity (≥ 0.05) | G-collapse | G-action | policy R1 AUC | policy ceiling AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HJ-px-IDM | 20260908 | 110.8 / 192 | 0.0014 | 1.67e-06 | fail | fail | 0.4871 | 0.8711 |
| HJ-px-IDM | 20260909 | 109.0 / 192 | 0.0018 | 1.75e-05 | fail | fail | 0.4891 | 0.8613 |
| HJ-px-IDM | 20260910 | 110.9 / 192 | 0.0018 | 2.65e-06 | fail | fail | 0.5246 | 0.8362 |
| HJ-car-m-IDM | 20260908 | 13.9 / 112 | 0.0162 | 1.92e-05 | fail | fail | 0.4406 | 0.5749 |
| HJ-car-m-IDM | 20260909 | 13.6 / 112 | 0.0140 | 5.03e-04 | fail | fail | 0.4652 | 0.5813 |
| HJ-car-m-IDM | 20260910 | 13.2 / 112 | 0.0157 | 4.54e-05 | fail | fail | 0.5448 | 0.5306 |
| LeWM-px-IDM | 20260908 | 162.7 / 192 | 0.0578 | 9.95e-01 | pass | pass | 0.4720 | 0.5885 |
| LeWM-px-IDM | 20260909 | 158.0 / 192 | 0.0640 | 9.50e-01 | pass | pass | 0.5015 | 0.5519 |
| LeWM-px-IDM | 20260910 | 159.4 / 192 | 0.0554 | 1.07e+00 | pass | pass | 0.4563 | 0.5690 |
| LeWM-car-m-IDM | 20260908 | 25.7 / 112 | 0.0440 | 8.75e-03 | fail | fail | 0.4995 | 0.6174 |
| LeWM-car-m-IDM | 20260909 | 25.5 / 112 | 0.0449 | 1.29e-02 | fail | fail | 0.4606 | 0.6511 |
| LeWM-car-m-IDM | 20260910 | 25.4 / 112 | 0.0450 | 9.88e-03 | fail | fail | 0.5221 | 0.5893 |

### Grid rankers (311 mixed cells × 3 seeds, R1)

| ranker | pooled AUC | seed 20260908 | seed 20260909 | seed 20260910 | top-1 | chance top-1 |
| --- | --- | --- | --- | --- | --- | --- |
| U | 0.7444 [0.7269, 0.7620] | 0.7141 [0.6917, 0.7362] | 0.8011 [0.7826, 0.8199] | 0.7181 [0.6961, 0.7404] | 0.195 | 0.089 |
| C\* | 0.7834 [0.7676, 0.7992] | 0.8262 [0.8091, 0.8431] | 0.7334 [0.7127, 0.7536] | 0.7905 [0.7688, 0.8120] | 0.192 | 0.089 |
| P | 0.7203 [0.6965, 0.7442] | seed-free | | | 0.180 | 0.089 |
| J | 0.7057 [0.6846, 0.7267] | 0.6690 [0.6449, 0.6931] | 0.7012 [0.6787, 0.7234] | 0.7470 [0.7261, 0.7678] | 0.160 | 0.089 |
| J_dec | 0.7460 [0.7277, 0.7637] | 0.7205 [0.6989, 0.7417] | 0.7933 [0.7739, 0.8124] | 0.7242 [0.7035, 0.7451] | 0.180 | 0.089 |
| **HJ-px-IDM** | 0.4948 [0.4779, 0.5122] | 0.4828 [0.4531, 0.5123] | 0.5069 [0.4793, 0.5357] | 0.4945 [0.4645, 0.5250] | 0.069 | 0.089 |
| **HJ-car-m-IDM** | 0.4900 [0.4685, 0.5112] | 0.4447 [0.4144, 0.4744] | 0.4788 [0.4504, 0.5071] | 0.5465 [0.5171, 0.5755] | 0.040 | 0.089 |
| LeWM-px-IDM | 0.4971 [0.4801, 0.5144] | 0.5037 [0.4736, 0.5344] | 0.5052 [0.4757, 0.5353] | 0.4824 [0.4523, 0.5124] | 0.062 | 0.089 |
| LeWM-car-m-IDM | 0.4790 [0.4630, 0.4949] | 0.4832 [0.4565, 0.5108] | 0.4743 [0.4452, 0.5035] | 0.4794 [0.4523, 0.5068] | 0.054 | 0.089 |
| HJ-px (#123) | 0.4997 [0.4827, 0.5165] | 0.4874 [0.4575, 0.5169] | 0.4995 [0.4681, 0.5308] | 0.5121 [0.4820, 0.5422] | 0.093 | 0.089 |
| HJ-car-m (#123) | 0.4896 [0.4694, 0.5099] | 0.4982 [0.4660, 0.5298] | 0.4856 [0.4552, 0.5157] | 0.4852 [0.4536, 0.5172] | 0.079 | 0.089 |

Readout variants (grid pooled AUC; R1 primary, R2 secondary, R3 / R4 exploratory; per-seed rows in `findings.md`):

| arm | R1 | R2 | R3 | R4 |
| --- | --- | --- | --- | --- |
| HJ-px-IDM | 0.4948 | 0.4898 | 0.5120 | 0.4909 |
| HJ-car-m-IDM | 0.4900 | 0.4932 | 0.4850 | 0.4940 |
| LeWM-px-IDM | 0.4971 | 0.5035 | 0.5096 | 0.5095 |
| LeWM-car-m-IDM | 0.4790 | 0.4783 | 0.4863 | 0.4811 |

### Contrasts (grid, paired per (level, seed); δ = 0.02)

| contrast | first − second | estimate (DESCRIPTIVE) | token (first − second) | token (second − first) | guards pass |
| --- | --- | --- | --- | --- | --- |
| **PI1** | HJ-px-IDM − U | −0.2496 [−0.2739, −0.2246] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| **PI2** | HJ-car-m-IDM − U | −0.2544 [−0.2863, −0.2243] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI1 | HJ-px-IDM − HJ-px (#123) | −0.0049 [−0.0291, +0.0197] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI2 | HJ-car-m-IDM − HJ-car-m (#123) | +0.0003 [−0.0282, +0.0284] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI3 | HJ-px-IDM − LeWM-px-IDM | −0.0023 [−0.0268, +0.0214] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI4 | HJ-car-m-IDM − LeWM-car-m-IDM | +0.0110 [−0.0171, +0.0385] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI5 pixels | HJ-px-IDM − C\* | −0.2886 [−0.3116, −0.2653] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI5 carriers | HJ-car-m-IDM − C\* | −0.2934 [−0.3233, −0.2649] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI6 pixels | HJ-px-IDM − P | −0.2255 [−0.2539, −0.1966] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI6 carriers | HJ-car-m-IDM − P | −0.2303 [−0.2651, −0.1962] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI7 pixels, J | HJ-px-IDM − J | −0.2110 [−0.2380, −0.1834] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI7 pixels, J_dec | HJ-px-IDM − J_dec | −0.2512 [−0.2758, −0.2260] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI7 carriers, J | HJ-car-m-IDM − J | −0.2157 [−0.2499, −0.1832] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI7 carriers, J_dec | HJ-car-m-IDM − J_dec | −0.2560 [−0.2882, −0.2258] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |

Every token is the stop token because every contrast involves an arm that fails its guards (SI1 / SI2 also involve
#123's failing HJ-px / HJ-car-m). The estimates are reported as numbers only.

### SI8: wrong-anchor control (right − wrong, 480 fit anchors)

| arm | right | wrong | right − wrong | token |
| --- | --- | --- | --- | --- |
| HJ-px-IDM | 0.4948 [0.4779, 0.5122] | 0.5037 [0.5026, 0.5048] | −0.0089 [−0.0257, +0.0086] | readiness_or_precision_insufficient |
| HJ-car-m-IDM | 0.4900 [0.4685, 0.5112] | 0.5019 [0.5002, 0.5036] | −0.0119 [−0.0333, +0.0093] | readiness_or_precision_insufficient |

### Descriptive inventories (no tokens)

Offset, angle and power pooled AUC, R1 (the R2–R4 inventory rows are in `findings.md`):

| ranker | offset | angle | power |
| --- | --- | --- | --- |
| U | 0.6670 [0.6503, 0.6827] | 0.6791 [0.6663, 0.6912] | 0.8058 [0.7729, 0.8396] |
| C\* | 0.7748 [0.7524, 0.7962] | 0.7590 [0.7422, 0.7752] | 0.8068 [0.7711, 0.8440] |
| P | 0.5000 [0.5000, 0.5000] | 0.8119 [0.7778, 0.8450] | 0.5000 [0.5000, 0.5000] |
| J | 0.7522 [0.7039, 0.7978] | 0.7765 [0.7402, 0.8115] | 0.8076 [0.7695, 0.8463] |
| J_dec | 0.6904 [0.6737, 0.7062] | 0.6988 [0.6827, 0.7147] | 0.7943 [0.7555, 0.8331] |
| HJ-px-IDM | 0.4810 [0.4419, 0.5196] | 0.4936 [0.4589, 0.5290] | 0.4986 [0.4538, 0.5440] |
| HJ-car-m-IDM | 0.4807 [0.4504, 0.5098] | 0.4872 [0.4670, 0.5064] | 0.4775 [0.4328, 0.5239] |
| LeWM-px-IDM | 0.4901 [0.4547, 0.5258] | 0.5129 [0.4791, 0.5463] | 0.4807 [0.4327, 0.5285] |
| LeWM-car-m-IDM | 0.5090 [0.4680, 0.5494] | 0.4799 [0.4476, 0.5119] | 0.4713 [0.4203, 0.5256] |
| HJ-px (#123) | 0.5004 [0.4625, 0.5386] | 0.4367 [0.3995, 0.4741] | 0.4947 [0.4541, 0.5370] |
| HJ-car-m (#123) | 0.4792 [0.4338, 0.5235] | 0.5019 [0.4638, 0.5397] | 0.4982 [0.4547, 0.5428] |

### Engine position error (#100 metric; DESCRIPTIVE)

IDM arms: the slot-centre probe P_c on the predicted top-level state; floor = P_c on the true frame encoding. U / C\*:
their predicted carriers. The two read-out paths differ, so the IDM rows are not like for like with U / C\*.

| model | t = 15 | t = 30 | t = 60 | t = 150 | t = 225 | floor t = 225 |
| --- | --- | --- | --- | --- | --- | --- |
| HJ-px-IDM | 0.0138 [0.0132, 0.0144] | 0.0139 [0.0133, 0.0145] | 0.0126 [0.0120, 0.0132] | 0.0124 [0.0118, 0.0130] | 0.0155 [0.0149, 0.0161] | 0.0147 [0.0140, 0.0153] |
| HJ-car-m-IDM | 0.0130 [0.0123, 0.0136] | 0.0152 [0.0144, 0.0159] | 0.0160 [0.0151, 0.0170] | 0.0282 [0.0261, 0.0303] | 0.0390 [0.0359, 0.0422] | 0.0113 [0.0109, 0.0118] |
| LeWM-px-IDM | 0.0115 [0.0110, 0.0120] | 0.0128 [0.0123, 0.0134] | 0.0130 [0.0125, 0.0135] | 0.0198 [0.0190, 0.0205] | 0.0247 [0.0238, 0.0256] | 0.0105 [0.0100, 0.0110] |
| LeWM-car-m-IDM | 0.0135 [0.0130, 0.0140] | 0.0138 [0.0133, 0.0144] | 0.0138 [0.0132, 0.0144] | 0.0148 [0.0142, 0.0154] | 0.0187 [0.0181, 0.0193] | 0.0120 [0.0115, 0.0126] |
| U | 0.0505 [0.0460, 0.0550] | 0.0501 [0.0457, 0.0545] | 0.0484 [0.0443, 0.0526] | 0.0459 [0.0423, 0.0496] | 0.0474 [0.0440, 0.0508] | n/a |
| C\* | 0.0547 [0.0500, 0.0595] | 0.0545 [0.0499, 0.0592] | 0.0543 [0.0497, 0.0589] | 0.0594 [0.0544, 0.0645] | 0.0670 [0.0617, 0.0723] | n/a |

## Mechanism

- **The HJ arms stay collapsed under the remedy.** HJ-px-IDM's χ is 0.0014–0.0018 (#123 HJ-px: 0.0014–0.0015) and
  HJ-car-m-IDM's 0.0140–0.0162 (#123 HJ-car-m: 0.0161–0.0167), against the 0.05 threshold. Endpoint launch sensitivity
  rises from #123's 1.1e-6 (pixels) and 1.7e-7 (carriers) to 1.7e-6–1.7e-5 and 1.9e-5–5.0e-4, still 100× to 30 000×
  below the 0.05 threshold. As in #123, the top-level latent barely moves within an episode: the slow-feature collapse
  of the paper's App. L persists with the remedy.
- **The encoder still holds the outcome; the prediction ignores the launch, as in #123.** HJ-px-IDM's R1 readout
  ceiling on the true frame-225 encodings of the policy grid candidates is 0.8362–0.8711 per seed (arm mean 0.8562;
  #123 HJ-px 0.8812), so G-readout passes. Ranking the predicted endpoints is at chance (grid AUC 0.4948; SI8
  −0.0089).
- **IDM does make the flat pixel latent launch-sensitive, but not decision-relevant.** LeWM-px-IDM's sensitivity is
  0.95–1.07 per seed (#123 LeWM-px: 0.034–0.052) with χ 0.055–0.064, so it passes G-collapse and G-action. Its readout
  ceiling, 0.5698 (per seed 0.5519–0.5885), is below the 0.65 G-readout bar, and it ranks at chance (grid AUC 0.4971).
  In this arm, action sensitivity alone does not yield an outcome representation that ranks launches.
- LeWM-car-m-IDM sits just below the χ threshold (0.044–0.045) with sensitivity 0.009–0.013 and ceiling 0.6193.
- SI1 / SI2 (the IDM effect) are −0.0049 and +0.0003 and SI3 / SI4 (the temporal lever) −0.0023 and +0.0110,
  DESCRIPTIVE, with stop tokens.

## Disposition (δ rule)

Frozen dispositions (ticket):
1. "H-JEPA with its inverse-dynamics remedy ranks better than our Gate-B model (scoped)" iff PI1 or PI2 is `supported`
   in the HJ − U direction, G-fid, G-port-IDM and the arm's guards pass, and SI8 is `supported`.
2. "Our Gate-B model ranks better than H-JEPA with its remedy (scoped)" iff U − HJ-px-IDM and U − HJ-car-m-IDM are both
   `supported` and every guard passes.
3. Otherwise, per-contrast tokens only, under the δ rule.
4. If the IDM arms still fail G-collapse / G-action, the finding is that the paper's own remedy does not prevent
   collapse on single-launch, action-sparse data, reported with the diagnostics.

Result:
- **Disposition 1 does not apply.** PI1 and PI2 are `readiness_or_precision_insufficient` in the HJ − U direction,
  and so is SI8.
- **Disposition 2 does not apply.** U − HJ-px-IDM and U − HJ-car-m-IDM are `readiness_or_precision_insufficient`,
  because both primary arms fail G-collapse and G-action (HJ-car-m-IDM also G-readout). The manuscript may not state,
  on this experiment, that our model beats or is better than H-JEPA (with or without its remedy), LeWM + IDM, and may
  not use "state of the art" against them.
- **Disposition 3 applies: per-contrast tokens only; outcome branch: no scoped ranking claim.** Every token in the
  contrast table is `readiness_or_precision_insufficient`; the estimates are DESCRIPTIVE numbers, not effects.
- **Disposition 4 holds** (`disposition_4_remedy_does_not_prevent_collapse: true`): HJ-px-IDM and HJ-car-m-IDM fail
  G-collapse and G-action on every seed (diagnostics above).
- The permitted statement is: "With the paper's own inverse-dynamics + action-SIGReg remedy (γ₁ = 225 selected on the
  policy split), H-JEPA still collapses on single-launch, action-sparse NovPhy: its predicted endpoint ignores the
  launch and ranks at chance; the frozen rule therefore makes no ranking claim."

## Deviations and disclosures

1. **Chronology.** The ticket was pre-registered 2026-10-19, before any #123 evaluation read (pre-trigger
   implementation `3b20ba7a`). The trigger ran on fit data and wrote `trigger.json` at 05:59 AEDT on 21 Oct, before
   #123's first evaluation read; `123-score` depended on `125-trigger` (`f7809a21`). `plan.json` was frozen at
   2026-10-21T07:26:34Z (18:26:34 AEDT), **after** #123's evaluation scoring (18:03–18:10 AEDT) but **before** the #125
   implementer read any #123 result: no #123 record, summary, compute, findings, validation or publication table was
   opened, and the #123 issue comments were not refreshed, before the freeze commit `c9d0b64c` was pushed. Every
   frozen choice comes from the ticket, the policy-split γ₁ selection, or the fit-split trigger and smoke.
2. **Amendments.** A1 (`d655ee35`, night 1, compute ledger and plan loading only) and A2 (`cae21709`, between nights 1
   and 2, validation only), both before any #125 readout or evaluation read, registered in
   `scripts/post_freeze_amendments.py` (`run_hjepa_idm.py` 5e533591… → 56e7cc65… → d53e44d0…) and tested; the inherited
   #123 A3 aggregation (`aa8e702f`, disclosed `558f63e1`). Objective, training, resume, RNG, data, readout fitting,
   scoring math, tokens and dispositions are unchanged. Full text above and in `decisions.md`.
3. **The `--publish` SIGSEGV and its single relaunch.** `125-publish#1` (launched 01:15:07 AEDT, 23 Oct) died with
   SIGSEGV (exit −11) after 25 s, during table derivation, after `records-manifest.json` was written (the first step of
   `--publish`). It left no traceback. This is an infrastructure crash, not a typed experimental failure; scoring had
   completed and no record was touched. `PYTHONFAULTHANDLER=1` was enabled for scheduled jobs (`ed5acdd3`), the job
   was requeued once (01:25:25), and `125-publish#2` (01:30:08–01:32:10) exited 0. Publishing is deterministic from the
   frozen records, and `125-validate#1` re-derived and byte-compared every table (exit 0). Logs:
   `logs/scheduler-125-publish.log`, `logs/scheduler-125-events.jsonl`.
4. **Night-window schedule and pauses.** Every GPU job ran through `scripts/gpu_window.py` inside 18:00–07:00
   Australia/Melbourne. Two scheduled pauses, both by SIGTERM at 06:50 → `resume.pt` at the next optimizer step → exit
   75 → relaunch at 18:00: `125-select-q0` (21 Oct; HJ-car-m-IDM γ₁ 225 finished on relaunch), and all four training
   queues (22 Oct; HJ-px-IDM 20260908 / 09 / 10 paused at steps 114 381 / 114 991 / 112 997 of 128 045, LeWM-px-IDM
   20260909 at 139 314 of 150 880). Night 2 resumed them; `125-train-q3#2` logged `model.pt exists; skipped` for the
   finished LeWM-px-IDM 20260908, which A1 keeps out of the ledger. There was no other interruption and no typed
   failure. Logs: `logs/scheduler-125-*.log`.
5. **Hardware.** 4 × RTX 5090 (32 GB) instead of the paper's H200. The bf16 / cuDNN training kernels are not bitwise
   reproducible run to run; the restored state on resume is exact, and scoring is deterministic (re-score max delta 0).
6. **Matched-carrier confound (from the freeze decisions).** To fit the IDM heads into U's parameter budget, only the
   predictor MLP width shrinks: HJ-car-m-IDM 448 → 96, LeWM-car-m-IDM 480 → 384. SI2 / SI4 therefore also change the
   carrier predictor width.

## Compute

Caps count active compute time (GPU seconds = each process's wall time on its GPU; daytime pauses do not count). Cap:
120 GPU-h on 4 × RTX 5090. Aggregation: `rc.ledger_totals` (#123 A3). **Check (A1):** each of the 12 training phases
counts once; every `ledger/train-<arm>-seed<s>.json` equals its `models/<arm>/seed-<s>/report.json` seconds, and no
phase carries `ledger_gpu_seconds` (no overcount).

| item | GPU-s | GPU-h |
| --- | --- | --- |
| main training, 12 arm-seeds (= sum of `report.json`) | 219 319 | 60.92 |
| γ₁ selection (4 runs, train + readouts) | 4 865 | 1.35 |
| readouts (12 arm-seeds) | 1 690 | 0.47 |
| trigger | 257 | 0.07 |
| scoring (12 arm-seeds incl. SI8; the evaluation cache is CPU, 208 s wall) | 103 | 0.03 |
| smoke | 45 | 0.01 |
| **recorded** (`compute.json` `total_gpu_hours`) | **226 279** | **62.86** |
| unrecorded: scheduler `--publish` / `--validate` runs, which write no ledger line (25 + 122 + 244 s) | 391 | 0.11 |
| **active compute total** | | **≈ 63.0 (cap 120)** |

Per seed: HJ-px-IDM 13.7–14.0 GPU-h (49 393–50 258 s; smoke projection 13.0), LeWM-px-IDM 6.3–6.5 GPU-h, the carrier
arms 2.4–4.6 min. The smoke projected 62.1 GPU-h. Wall time with any #125 GPU process running was about 21 h over four
windows (10-21 05:55 → 06:50; 10-21 18:00 → 18:11; 10-21 18:30 → 10-22 06:50; 10-22 18:00 → 10-23 01:39). Engine
seconds: 0. No scope was cut.

## Commands

```
conda activate novphy && source env.sh
M="python -u -m scripts.run_hjepa_idm"
CUDA_VISIBLE_DEVICES="" pytest -q tests/test_issue_125_*.py
CUDA_VISIBLE_DEVICES= $M --gport-configs                     # then --gport-reference (hjepa env), --gport
$M --trigger | --selection --gpu k --workers 3 | --smoke     # scheduler, night window
$M --prepare                                                 # freeze (CPU)
$M --train --gpu g | --readouts --gpu g | --freeze-readouts  # scheduler, night window
CUDA_VISIBLE_DEVICES=1 $M --score                            # once
CUDA_VISIBLE_DEVICES=1 $M --publish
CUDA_VISIBLE_DEVICES=1 $M --validate                         # exit 0
```
