# Issue #125: H-JEPA with its inverse-dynamics remedy on NovPhy

Status (2026-10-19): **pre-trigger.** Code, CPU tests and G-port-IDM are in; the trigger, the γ₁ selection and the
smoke are queued for the night window after #123's training. Nothing has run on a GPU. This document is completed
at publication.

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
| `scripts/gpu_window_issue_125.json` | scheduler jobs (trigger, selection, smoke) |
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
file is read.

## Schedule

Night window only (18:00–07:00 Australia/Melbourne), through `scripts/gpu_window.py`:
1. `125-trigger` runs after all four `123-train-q*` queues, on GPU 1 beside `123-readouts-g1`.
2. `125-select-q0/q1/q2` run on GPUs 1, 2 and 3.
3. `125-smoke` runs on GPU 1.

#123's jobs keep priority, and no #125 job uses GPU 0. Training, readouts and scoring are enqueued only after the
#125 freeze (`--prepare`, an agent commit).

## Commands

```
CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_issue_125_hjepa_idm.py
CUDA_VISIBLE_DEVICES= python -u -m scripts.run_hjepa_idm --gport-configs   # then --gport-reference, --gport
python -u -m scripts.run_hjepa_idm --trigger | --selection --gpu k --workers 3 | --smoke     # scheduler
python -u -m scripts.run_hjepa_idm --prepare                                                 # freeze (CPU)
python -u -m scripts.run_hjepa_idm --train --gpu g | --readouts --gpu g | --freeze-readouts | --score | --publish
python -u -m scripts.run_hjepa_idm --validate
```
