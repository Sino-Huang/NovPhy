# Issue-96 tau_ad-W: within-checkpoint selection-schedule contrast — findings

- identity `issue-96-tau-ad-within-checkpoint-v1`, plan v1 frozen 2026-09-25T10:50:13Z before any scoring run
- validation command: `python -u -m scripts.run_tau_ad_within_checkpoint --validate`; zero engine seconds; endpoint 225; every interval DESCRIPTIVE (member-clustered, 10000 draws, PCG64 7201)
- disclosure: The pre-freeze Oracle review read #92's J step-pair frequencies (horizon_trace, trace-only, outcome-free) and F*'s cross-fit inventories are chosen from published fixed-arm results; no outcome statistic of this contrast was computed before freeze. The #92 G1 reference records are read for chosen_ordinal and selected_predicted_cost only.

## Disposition

- **primary theta_grid = AUC_J - AUC_F\* (F\* = F-5-macro): 0.0700 [-0.0057, 0.1437] over 42 cells / 14 members -> readiness_or_precision_insufficient** (delta 0.02)
- secondary J-M (grid): -0.0215 [-0.0706, 0.0241] -> **readiness_or_precision_insufficient**
- secondary J-OL (grid): -0.0520 [-0.1192, 0.0173] -> **readiness_or_precision_insufficient**
- fixed-pair spread on the grid (member mean of per-cell max_F - min_F): 0.5098; near-inert rule triggered: False
- replication of J - F\* on the secondary inventories (theta >= delta AND lower > 0): 1/3 offset 0.0037 [-0.0796, 0.0926], angle 0.0595 [0.0079, 0.1131], power 0.0273 [-0.0691, 0.1281]
- reading: tau_ad stays unmeasured-in-effect; the five 'unmeasured' manuscript sentences are still corrected to the accurate three-reading statement

## Guards

| guard | pass | observed |
|---|---|---|
| G1 | True | `{"run_records": [{"abs_cost_delta": 0.0, "arm": "J", "chosen": 5, "ok": true, "reference": "decision--hybrid-adaptive-e225--seed20260908--issue-77-n1-001-a00", "reference_chosen": 5}, {"abs_cost_delta": 0.0, "arm": "F-15-continuous", "chosen": 12, "ok": true, "reference": "decision--hybrid-fixed-h15-e225--seed20260908--issue-77-n1-001-a00", "reference_chosen": 12}], "smoke": [{"abs_cost_delta": 0.0, "chosen_ordinal_fresh": 5, "chosen_ordinal_retained": 5, "control": "G1_replication_J"}, {"abs_cost_delta": 0.0, "chosen_ordinal_fresh": 12, "chosen_ordinal_retained": 12, "control": "G1_replication_F-15-continuous"}]}` |
| G2 | True | `{"equal": 15, "members": 15}` |
| G3 | True | `{"nonfinite_share": 0.0, "verdict_resolution": 1.0}` |
| G4 | False | `{"cells": 42, "tied_cells": 17, "tied_share": 0.40476190476190477}` |
| G5 | True | `{"J_modal_pair": "1-macro", "J_nonmodal_step_share": 0.24147201147021924, "rank_divergent_share": 0.35714285714285715}` |
| G6 | False | `{"half_width": 0.07467403628117913}` |
| caps | True | `{"gpu_seconds": 503.75759667830425, "status": "terminal", "wall_seconds": 519.6720574840001}` |

## Contrasts per inventory (never pooled; AUC difference, member-clustered)

| inventory | mixed members / cells | J - F* [F*] | J - F_hindsight [F_h] | J - F(1,macro) | J - F(1,continuous) | J - M | J - OL | J - T | J - D | C-J - C-F* [C-F*] |
|---|---|---|---|---|---|---|---|---|---|---|
| grid | 14 / 42 | 0.0700 [-0.0057, 0.1437] | -0.0420 [-0.1295, 0.0475] | -0.0420 [-0.1295, 0.0475] | -0.0398 [-0.1084, 0.0229] | -0.0215 [-0.0706, 0.0241] | -0.0520 [-0.1192, 0.0173] | -0.0277 [-0.0654, 0.0114] | -0.0387 [-0.1206, 0.0443] | 0.0364 [-0.0424, 0.1152] |
| offset | 9 / 27 | 0.0037 [-0.0796, 0.0926] | -0.0870 [-0.1500, -0.0259] | 0.0037 [-0.0796, 0.0926] | -0.0093 [-0.0815, 0.0630] | -0.0167 [-0.0889, 0.0500] | -0.0019 [-0.0500, 0.0444] | -0.0333 [-0.0889, 0.0185] | -0.0222 [-0.1111, 0.0630] | -0.0778 [-0.1481, 0.0148] |
| angle | 7 / 21 | 0.0595 [0.0079, 0.1131] | -0.0691 [-0.1207, -0.0119] | 0.0595 [0.0079, 0.1131] | 0.0456 [0.0079, 0.0933] | -0.0058 [-0.0278, 0.0119] | 0.0220 [-0.0256, 0.0714] | 0.0121 [-0.0177, 0.0498] | 0.0795 [-0.0377, 0.2242] | -0.0305 [-0.1364, 0.0945] |
| power | 8 / 24 | 0.0273 [-0.0691, 0.1281] | 0.0013 [-0.0656, 0.0610] | 0.0013 [-0.0656, 0.0610] | 0.0196 [-0.0482, 0.0866] | -0.0210 [-0.0789, 0.0338] | -0.0084 [-0.0647, 0.0578] | 0.0056 [-0.0592, 0.0801] | 0.0201 [-0.0395, 0.0774] | -0.0471 [-0.1700, 0.0702] |

| inventory | F* (cross-fit) | F_hindsight | C-F* | AUC J | AUC F* | top-1 J / F* / chance | fixed-pair spread | coverage (reached/successful slots) | J modal pair, non-modal share |
|---|---|---|---|---|---|---|---|---|---|
| grid | F-5-macro | F-1-macro | C-F-5 | 0.4706 [0.4158, 0.5256] | 0.4006 [0.3475, 0.4612] | 0/42 / 0/42 / 0.0804 | 0.5098 [0.4238, 0.5969] | 11/17 | 1-macro, 0.2415 |
| offset | F-1-macro | F-5-macro | C-F-5 | 0.5074 [0.4556, 0.5593] | 0.5037 [0.4444, 0.5593] | 0/27 / 1/27 / 0.1030 | 0.4444 [0.3611, 0.5389] | 3/9 | 1-macro, 0.2246 |
| angle | F-1-macro | F-5-macro | C-F-5 | 0.5655 [0.5198, 0.6250] | 0.5060 [0.4226, 0.5833] | 0/21 / 0/21 / 0.0778 | 0.4969 [0.4315, 0.5595] | 5/7 | 1-macro, 0.2486 |
| power | F-5-macro | F-1-macro | C-F-5 | 0.5287 [0.4693, 0.5906] | 0.5014 [0.4135, 0.6056] | 0/24 / 0/24 / 0.0521 | 0.4909 [0.4594, 0.5263] | 4/8 | 1-macro, 0.2364 |

- top-1 power note: top-1 over ~42 grid cells against chance ~0.06-0.10 cannot resolve differences below ~0.15; descriptive only

## Per-arm member-mean AUC (mixed cells)

| arm | grid | offset | angle | power |
|---|---|---|---|---|
| F-1-continuous | 0.5104 | 0.5167 | 0.5198 | 0.5092 |
| F-1-micro | 0.4932 | 0.5000 | 0.5020 | 0.5198 |
| F-1-macro | 0.5126 | 0.5037 | 0.5060 | 0.5274 |
| F-5-continuous | 0.4145 | 0.5741 | 0.5808 | 0.4582 |
| F-5-micro | 0.4253 | 0.5759 | 0.6093 | 0.4688 |
| F-5-macro | 0.4006 | 0.5944 | 0.6346 | 0.5014 |
| F-15-continuous | 0.4918 | 0.4407 | 0.4502 | 0.4618 |
| F-15-micro | 0.4791 | 0.4870 | 0.4502 | 0.4516 |
| F-15-macro | 0.4912 | 0.5222 | 0.5335 | 0.4994 |
| J | 0.4706 | 0.5074 | 0.5655 | 0.5287 |
| M | 0.4921 | 0.5241 | 0.5712 | 0.5497 |
| T | 0.4984 | 0.5407 | 0.5534 | 0.5232 |
| D | 0.5093 | 0.5296 | 0.4859 | 0.5087 |
| OL | 0.5226 | 0.5093 | 0.5435 | 0.5371 |
| C-F-1 | 0.4678 | 0.5204 | 0.4771 | 0.5088 |
| C-F-5 | 0.4856 | 0.6241 | 0.5765 | 0.6051 |
| C-F-15 | 0.5057 | 0.5519 | 0.4747 | 0.5766 |
| C-J | 0.5220 | 0.5463 | 0.5460 | 0.5580 |

## Schedules (outcome-free)

- seed 20260908: J step counts {'1-continuous': 23534, '1-micro': 698, '1-macro': 83, '5-continuous': 24694, '5-micro': 0, '5-macro': 9398, '15-continuous': 0, '15-micro': 0, '15-macro': 185}; alpha_T continuous, Delta_D 5
- seed 20260909: J step counts {'1-continuous': 103, '1-micro': 0, '1-macro': 159592, '5-continuous': 0, '5-micro': 775, '5-macro': 883, '15-continuous': 7, '15-micro': 133, '15-macro': 1831}; alpha_T macro, Delta_D 1
- seed 20260910: J step counts {'1-continuous': 3097, '1-micro': 23636, '1-macro': 129967, '5-continuous': 1483, '5-micro': 6676, '5-macro': 11, '15-continuous': 0, '15-micro': 0, '15-macro': 0}; alpha_T macro, Delta_D 1

## Compute (reported, not matched)

| arm | records | transition calls | controller calls | linear MACs | GPU s |
|---|---|---|---|---|---|
| F-1-continuous | 177 | 592650 | 0 | 848940307200 | 49.5 |
| F-1-micro | 177 | 592650 | 0 | 1064076998400 | 59.9 |
| F-1-macro | 177 | 592650 | 0 | 868056825600 | 54.3 |
| F-5-continuous | 177 | 118530 | 0 | 169788061440 | 10.0 |
| F-5-micro | 177 | 118530 | 0 | 212815399680 | 11.9 |
| F-5-macro | 177 | 118530 | 0 | 173611365120 | 10.8 |
| F-15-continuous | 177 | 39510 | 0 | 56596020480 | 3.4 |
| F-15-micro | 177 | 39510 | 0 | 70938466560 | 4.1 |
| F-15-macro | 177 | 39510 | 0 | 57870455040 | 3.6 |
| J | 177 | 386786 | 386786 | 587803681280 | 44.1 |
| M | 177 | 386994 | 386994 | 587803075584 | 47.6 |
| T | 177 | 397610 | 397610 | 593203044864 | 44.0 |
| D | 177 | 434610 | 434610 | 658166889984 | 48.2 |
| OL | 177 | 387818 | 0 | 577041049856 | 35.2 |
| C-F-1 | 177 | 592650 | 0 | 1097706330000 | 38.8 |
| C-F-5 | 177 | 118530 | 0 | 219541266000 | 7.8 |
| C-F-15 | 177 | 39510 | 0 | 73180422000 | 2.6 |
| C-J | 177 | 350982 | 350982 | 661353627690 | 28.0 |

- run GPU 503.8 s, wall 519.7 s (caps 10800 / 21600 s); engine seconds 0

## Claim boundary

decision-only rollouts of the frozen #77 N1 checkpoints from the sealed member anchors to endpoint 225 on four retained engine-verdicted inventories; the controller selects the prediction pair from a policy trained by dynamic programming on the same frozen corpus, never from task outcomes; the record bounds decision-time pair-schedule adaptivity on this checkpoint, not policy learning; no engine access, rendering or retraining; prior dispositions are inputs and are not re-opened; inventories are never pooled; every interval is DESCRIPTIVE.
