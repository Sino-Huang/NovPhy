# Issue-93: within-state selection AUC on a second candidate parameterization — findings

- identity `issue-93-second-parameterization-v1`, plan `issue_93_second_parameterization_plan_v1` v1 (terminal), frozen 2026-09-23T15:10:02Z before any scored engine slot
- validation command: `python -u -m scripts.run_second_parameterization_probe --validate`
- engine realization: the engine projects any drag beyond _dragRadius (1 world unit = 18.49 px at the fixed camera) onto the drag circle, so every candidate of both arms launches at the saturated speed 10 in its drag direction (234/234 retained #87 launches at 9.9992-9.9997); the grid's engine-side variation is launch angle only
- every interval is DESCRIPTIVE (10000 draws, seed 7201); the interval is never the decision rule

## Dispositions (frozen rule)

- **C22 (Arm B, drag grid): not_supported_by_this_experiment**
- **C23 (Arm A, offset sweep): readiness_or_precision_insufficient**
- reading-matrix row 5: "Report descriptively in the appendix only; body scope sentences unchanged except a pointer."

## Arm B — Cartesian drag grid (C22, primary)

- claim: On a Cartesian drag-grid inventory (non-monotone in the candidate ordinal; realized in the engine as 16 distinct launch angles at saturated launch speed, its pull-radius axis present on the ranker-input side only), the frozen rankers' within-state engine-truth selection AUC is at or below chance.
- slots: 240 scheduled; {"typed_failure": 8, "verdict": 232}; verdict coverage **0.9667**; typed failures {"execution_failure": 8}
- ceiling: **14/15** measurable members = 0.9333 (interval 0.9333 [0.8000, 1.0000]); ceiling members ['issue-77-n1-001', 'issue-77-n1-003', 'issue-77-n1-004', 'issue-77-n1-005', 'issue-77-n1-006', 'issue-77-n1-007', 'issue-77-n1-008', 'issue-77-n1-009', 'issue-77-n1-010', 'issue-77-n1-011', 'issue-77-n1-012', 'issue-77-n1-013', 'issue-77-n1-014', 'issue-77-n1-015']
- **within-state AUC**: cell-unit 0.6178 [0.5784, 0.6554] over 126 cells; member-clustered **0.6178 [0.5126, 0.7139]** over 14 members
  - continuous-fixed-h1: member-clustered 0.5983 [0.4798, 0.7051] (42 cells)
  - continuous-fixed-h5: member-clustered 0.6046 [0.4854, 0.7125] (42 cells)
  - hybrid-fixed-h1: member-clustered 0.6503 [0.5562, 0.7422] (42 cells)
- top-1: 8/126 = 0.0635 vs chance 0.0804; paired difference -0.0169 [-0.0938, 0.1190]
- top-3: 31/126 = 0.2460 vs chance 0.2357; paired difference 0.0103 [-0.1343, 0.1782]
- mirror (argmax-cost) top-1, post-hoc: 0/126 = 0.0000 vs chance 0.0804; paired difference -0.0804 [-0.0938, -0.0670]
- miss-all probability (one uniform draw per ceiling member): 0.3074
- structure (median Spearman rho of predicted cost, DESCRIPTIVE): ordinal -0.7500 (|rho| 0.7500); drag_x 0.5700 (|rho| 0.5700); drag_y -0.8125 (|rho| 0.8125); radius -0.7882 (|rho| 0.7882); angle -0.3588 (|rho| 0.3588)
- chosen-ordinal histogram (selector over full inventory): {"3": 15, "15": 120}
- success-ordinal histogram (members): {"1": 1, "2": 1, "5": 2, "6": 1, "7": 4, "9": 1, "10": 3, "13": 1, "14": 2, "15": 1}
- disposition inputs: {"auc_member_clustered": 0.6177626606198033, "ceiling_members": 14, "verdict_coverage": 0.9666666666666667} -> **not_supported_by_this_experiment**

## Arm A — offset angle sweep (C23)

- claim: On an offset launch-angle sweep at radius 80 px (11 angles disjoint from the 13 N1 samples), the frozen rankers' within-state engine-truth selection AUC is at or below chance.
- slots: 165 scheduled; {"typed_failure": 6, "verdict": 159}; verdict coverage **0.9636**; typed failures {"execution_failure": 6}
- ceiling: **9/15** measurable members = 0.6000 (interval 0.6000 [0.3333, 0.8667]); ceiling members ['issue-77-n1-001', 'issue-77-n1-003', 'issue-77-n1-004', 'issue-77-n1-005', 'issue-77-n1-006', 'issue-77-n1-007', 'issue-77-n1-008', 'issue-77-n1-009', 'issue-77-n1-014']
- **within-state AUC**: cell-unit 0.5512 [0.4963, 0.6080] over 81 cells; member-clustered **0.5512 [0.4235, 0.6685]** over 9 members
  - continuous-fixed-h1: member-clustered 0.6463 [0.5056, 0.7741] (27 cells)
  - continuous-fixed-h5: member-clustered 0.5704 [0.4148, 0.7111] (27 cells)
  - hybrid-fixed-h1: member-clustered 0.4370 [0.3444, 0.5259] (27 cells)
- top-1: 7/81 = 0.0864 vs chance 0.1030; paired difference -0.0166 [-0.0781, 0.0575]
- top-3: 28/81 = 0.3457 vs chance 0.3091; paired difference 0.0366 [-0.1246, 0.1978]
- mirror (argmax-cost) top-1, post-hoc: 0/81 = 0.0000 vs chance 0.1030; paired difference -0.1030 [-0.1273, -0.0909]
- miss-all probability (one uniform draw per ceiling member): 0.3732
- structure (median Spearman rho of predicted cost, DESCRIPTIVE): ordinal -0.8818 (|rho| 0.8818); drag_x -0.8818 (|rho| 0.8818); drag_y -0.8818 (|rho| 0.8818); radius 0.2187 (|rho| 0.2551); angle -0.8818 (|rho| 0.8818)
- chosen-ordinal histogram (selector over full inventory): {"5": 13, "6": 53, "7": 1, "8": 29, "9": 8, "10": 31}
- success-ordinal histogram (members): {"2": 2, "3": 1, "4": 1, "5": 4, "6": 1}
- disposition inputs: {"auc_member_clustered": 0.5512345679012346, "ceiling_members": 9, "verdict_coverage": 0.9636363636363636} -> **readiness_or_precision_insufficient**

## Execution and compute

- decision frames byte-identical to the member's retained #87 anchor: 375/378
- executed launch speed range 9.9992–9.9996 (saturated at 10 by the drag clamp)
- oracle wall 5.85 h (stop reason None); completion pass wall None; ranker scoring GPU 8.6 s
- per-slot worker wall: {"count": 405, "max": 300.7674613678828, "mean": 179.24065412462883, "median": 188.56218966376036, "p10": 110.27877555787563, "p90": 243.3714932501316}

## Training action support (WP0)

- grid: out of the release-1000 training support in radius on the ranker-input side (N1 radius [79.51100552753688, 80.65977932030313]); 0/16 exact training actions; the #71 pool covers the drag plane at release 600 only
- offset: inside the N1 radius range; 0/11 exact training actions

## Claim boundary

within-state engine-truth selection AUC of the three frozen N1 rankers on two new candidate inventories over 15 N1 source members; does not test other models, adaptation, fresh gameplay, sealed benchmarks (#64/#65 sealed) or state difficulty; does not re-open #87/#89/#90/#92 dispositions. Arms are never pooled with each other or with N1. No state-difficulty claim is drawn from any row. `bound_issue87_only` / `first_shot_success_issue87_only` in slot_join.csv keep #92's column names and bind the chosen admissible candidate to this experiment's own oracle table.
