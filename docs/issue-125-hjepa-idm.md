# Issue #125: H-JEPA with its inverse-dynamics remedy on NovPhy

Status (2026-10-21): **frozen.** The trigger fired: all six #123 primary arm-seeds fail G-collapse and G-action
on fit data. γ₁ = 225 was selected for both input types on the policy split. `plan.json` was frozen at
2026-10-21T07:26:34Z. Main training runs in the night window. This document is completed at publication.

Pre-registered follow-up to #123 (ticket: `round2/issues/125.md`). It tests H-JEPA with the paper's own anti-collapse
remedy: an inverse-dynamics (IDM) term and action-SIGReg (§4.3 Eq. 8, App. K, the DROID recipe). Only the IDM terms
change from #123's frozen plan.

## Files

| path | role |
| --- | --- |
| `scripts/run_hjepa_idm.py` | runner (modes below) |
| `world_model/hjepa/idm.py` | IDM arms, configs, heads, matched sizes |
| `scripts/hjepa_idm_port_reference.py` | official side of G-port-IDM (hjepa env, CPU) |
| `tests/test_issue_125_hjepa_idm.py` | CPU tests |
| `scripts/gpu_window_issue_125.json` | scheduler jobs (selection, smoke); `125-trigger` is in `scripts/gpu_window_issue_123.json` |
| `.local-artifacts/issue-125-hjepa-idm-v1/` | `gport-idm.json`, `trigger.json`, selection, `smoke.json`, `plan.json`, models, records, publication |
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

## G-port-IDM (CPU, 2026-10-19): pass

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

## Commands

```
CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_issue_125_hjepa_idm.py
CUDA_VISIBLE_DEVICES= python -u -m scripts.run_hjepa_idm --gport-configs   # then --gport-reference, --gport
python -u -m scripts.run_hjepa_idm --trigger | --selection --gpu k --workers 3 | --smoke     # scheduler
python -u -m scripts.run_hjepa_idm --prepare                                                 # freeze (CPU)
python -u -m scripts.run_hjepa_idm --train --gpu g | --readouts --gpu g | --freeze-readouts | --score | --publish
python -u -m scripts.run_hjepa_idm --validate
```
