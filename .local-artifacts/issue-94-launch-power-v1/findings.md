# Issue-94: launch-power x angle inventory inside the 18.49 px drag clamp — findings

- identity `issue-94-launch-power-v1`, plan `issue_94_launch_power_plan_v1` v1 (terminal), frozen 2026-09-24T04:00:30Z before any engine slot
- validation command: `python -u -m scripts.run_launch_power_probe --validate`
- action-bound authority (WP0): drag_x in [-160, -10] (successor_cohort.py ACTION_BOUNDS) and [-160, -40] (gameplay_success.py action_bounds) are enforced only by pipeline samplers and plan/candidate validators (successor_cohort sampler, gameplay_success validator, lineage_scaling ActionRankingState). The executed path used here - ReactiveSelector.action_tensor (scoring), slingshot_readiness._anchored_action_and_command (command), ScienceBirdsBridge.shoot (socket), the interface jar ShootAndTapSchema.shoot, AIBirdsConnection.TapShoot, HUD.Drag and ABBird.DragBird (engine) - contains no drag bound; the engine's only limit is the _dragRadius clamp from above. The runner therefore uses its own frozen action contract (the 20 listed candidates, tap 0 ms, release 1000 ms).
- training support (WP0): the N1 predictor pool (the only release-1000 training actions) spans pull radius 79.51-80.66 px; 16/16 inner-radius candidates are outside it on the ranker input (out of training support); 0 candidates are exact training actions. Printed, not a reason to exclude.
- every interval is DESCRIPTIVE (10000 draws, seed 7201, quantiles 0.025/0.975)

## Dispositions (frozen rules) vs stated predictions

| claim | outcome | stated prediction | rationale |
|---|---|---|---|
| C24 | **supported** | supported | pooled top-1 was at or below inventory-matched chance in point estimate on all three saturated-speed inventories (0/108, 7/81, 8/126), and every inner-radius candidate is out of the rankers' training support, so nothing predicts a gain in selection. |
| C25 | **not_supported_by_this_experiment** | no prediction | no directional prediction: member-clustered AUC moved with the inventory (0.4180 / 0.5512 / 0.6178), so the prior evidence does not fix its sign on a new engine-side coordinate. |

- reading-matrix row 2 (C24 supported, C25 not_supported_by_this_experiment): "Selection fails while ordering tracks an engine-relevant coordinate; the ordering/selection dissociation is shown on an engine-effective second coordinate (earns r6 I49's "We show", scoped to this inventory)."
- readiness: {"ceiling_members": 8, "ready": true, "verdict_coverage": 0.9833333333333333}

## Estimands

- slots: 300 scheduled; {"typed_failure": 5, "verdict": 295}; verdict coverage **0.9833**; typed failures {"execution_failure": 5}
- 1. ceiling: **8/15** measurable members = 0.5333 (interval 0.5333 [0.2667, 0.8000]); ceiling members ['issue-77-n1-004', 'issue-77-n1-005', 'issue-77-n1-006', 'issue-77-n1-007', 'issue-77-n1-008', 'issue-77-n1-009', 'issue-77-n1-013', 'issue-77-n1-014']
- 2. **pooled top-1**: 0/72 = 0.0000 vs chance 0.0521; member-clustered paired difference **-0.0521 [-0.0563, -0.0500]**
  - continuous-fixed-h1: 0/24 = 0.0000 vs chance 0.0521; paired difference -0.0521 [-0.0563, -0.0500]
  - continuous-fixed-h5: 0/24 = 0.0000 vs chance 0.0521; paired difference -0.0521 [-0.0563, -0.0500]
  - hybrid-fixed-h1: 0/24 = 0.0000 vs chance 0.0521; paired difference -0.0521 [-0.0563, -0.0500]
- 3. **within-state AUC**: cell-unit 0.7441 [0.7232, 0.7651] over 72 cells; member-clustered AUC_m **0.7441 [0.6923, 0.8026]** over 8 members
  - continuous-fixed-h1: member-clustered 0.7641 [0.7243, 0.8199] (24 cells)
  - continuous-fixed-h5: member-clustered 0.7531 [0.7105, 0.8033] (24 cells)
  - hybrid-fixed-h1: member-clustered 0.7151 [0.6449, 0.7873] (24 cells)
- 4. top-3: 9/72 = 0.1250 vs chance 0.1563; paired difference -0.0313 [-0.1688, 0.2250]; miss-all probability 0.6518
- 5. engine-side structure (DESCRIPTIVE, median Spearman rho of predicted cost):
  - within fixed angle vs realized launch speed: 10deg -1.0000; 30deg -1.0000; 50deg -1.0000; 70deg -1.0000 (pooled -1.0000)
  - within fixed radius vs realized launch angle: 5px -1.0000; 9px -0.8000; 13px -0.8000; 17px -1.0000; 80px -1.0000 (pooled -1.0000)
  - full inventory vs expected speed -0.9317; vs ordinal -0.4827
  - chosen-ordinal histogram (selector over full inventory): {"14": 67, "19": 68}
  - success-ordinal histogram (members): {"8": 3, "9": 1, "13": 4}
  - saturated-column chosen share: {"admissible_top1": 1.0, "cells": 135, "selector_full_inventory": 1.0, "uniform_reference": 0.2}
- 6. realized launch speed per radius column (engine telemetry); inner radii distinct sub-maximal: **True**
  - 5 px: 59 slots, speed 2.1934–2.6421 (median 2.3351); realized − expected median -0.2251; angle offset -9.16…-1.71 deg
  - 9 px: 60 slots, speed 4.2300–4.8717 (median 4.6387); realized − expected median -0.2727; angle offset -4.79…-0.88 deg
  - 13 px: 58 slots, speed 6.4725–7.0455 (median 6.5871); realized − expected median -0.3199; angle offset -3.40…-0.43 deg
  - 17 px: 59 slots, speed 8.9022–9.2806 (median 8.9202); realized − expected median -0.2891; angle offset -2.56…-0.42 deg
  - 80 px: 59 slots, speed 9.9992–9.9995 (median 9.9992); realized − expected median -0.0008; angle offset -0.54…-0.08 deg

## Execution and compute

- decision frames byte-identical to the member's retained anchor: 295/295
- oracle wall 4.01 h (stop reason None); completion pass wall None; ranker scoring GPU 5.8 s; caps respected True
- per-slot worker wall: {"count": 300, "max": 257.1047220659998, "mean": 165.64756354500997, "median": 173.7819111429999, "min": 35.08935822500007, "p10": 106.79837443600081, "p90": 213.23401949599997}

## Claim boundary

pooled top-1 and within-state engine-truth AUC of the three frozen N1 rankers on one angle x launch-power inventory over 15 N1 source members; does not test other models, adaptation, fresh gameplay, sealed benchmarks (#64/#65 sealed) or state difficulty; never pooled with N1 or #93; does not re-open #87/#89/#90/#92/#93 dispositions. No state-difficulty claim is drawn from any row. `bound_issue87_only` / `first_shot_success_issue87_only` in slot_join.csv keep #92's column names and bind the chosen admissible candidate to this experiment's own oracle table.
