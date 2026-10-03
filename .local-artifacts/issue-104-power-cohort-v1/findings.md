# #104 joint cohort plan (`issue-104-power-cohort-v1`)

Frozen 2026-10-03T10:06:16Z, before any capture. `python -u -m scripts.prepare_power_cohort --validate`.

## Power (from #96 member-level records; required mixed members, design = max of two estimators)

| inventory | contrast | members | estimate | half-width | sd | delta 0.02 | delta 0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| angle | C-J-C-F* | 7 | -0.0305 | 0.1154 | 0.1687 | 274 | 44 |
| angle | J-D | 7 | +0.0795 | 0.1310 | 0.1891 | 344 | 55 |
| angle | J-F(1,continuous) | 7 | +0.0456 | 0.0427 | 0.0625 | 38 | 6 |
| angle | J-F(1,macro) | 7 | +0.0595 | 0.0526 | 0.0789 | 60 | 10 |
| angle | J-F* | 7 | +0.0595 | 0.0526 | 0.0789 | 60 | 10 |
| angle | J-F_hindsight | 7 | -0.0691 | 0.0544 | 0.0786 | 60 | 10 |
| angle | J-M | 7 | -0.0058 | 0.0198 | 0.0278 | 8 | 2 |
| angle | J-OL | 7 | +0.0220 | 0.0485 | 0.0707 | 49 | 8 |
| angle | J-T | 7 | +0.0121 | 0.0337 | 0.0497 | 24 | 4 |
| grid | C-J-C-F* | 14 | +0.0364 | 0.0788 | 0.1561 | 235 | 38 |
| grid | J-D | 14 | -0.0387 | 0.0824 | 0.1622 | 253 | 41 |
| grid | J-F(1,continuous) | 14 | -0.0398 | 0.0657 | 0.1287 | 159 | 26 |
| grid | J-F(1,macro) | 14 | -0.0420 | 0.0885 | 0.1746 | 293 | 47 |
| grid | J-F* | 14 | +0.0700 | 0.0747 | 0.1469 | 208 | 34 |
| grid | J-F_hindsight | 14 | -0.0420 | 0.0885 | 0.1746 | 293 | 47 |
| grid | J-M | 14 | -0.0215 | 0.0474 | 0.0935 | 84 | 14 |
| grid | J-OL | 14 | -0.0520 | 0.0683 | 0.1353 | 176 | 29 |
| grid | J-T | 14 | -0.0277 | 0.0384 | 0.0766 | 57 | 10 |
| offset | C-J-C-F* | 9 | -0.0778 | 0.0815 | 0.1331 | 171 | 28 |
| offset | J-D | 9 | -0.0222 | 0.0870 | 0.1409 | 191 | 31 |
| offset | J-F(1,continuous) | 9 | -0.0093 | 0.0722 | 0.1196 | 138 | 22 |
| offset | J-F(1,macro) | 9 | +0.0037 | 0.0861 | 0.1421 | 194 | 32 |
| offset | J-F* | 9 | +0.0037 | 0.0861 | 0.1421 | 194 | 32 |
| offset | J-F_hindsight | 9 | -0.0870 | 0.0620 | 0.1033 | 103 | 17 |
| offset | J-M | 9 | -0.0167 | 0.0694 | 0.1140 | 125 | 20 |
| offset | J-OL | 9 | -0.0019 | 0.0472 | 0.0770 | 57 | 10 |
| offset | J-T | 9 | -0.0333 | 0.0537 | 0.0870 | 73 | 12 |
| power | C-J-C-F* | 8 | -0.0471 | 0.1201 | 0.1847 | 328 | 53 |
| power | J-D | 8 | +0.0200 | 0.0584 | 0.0915 | 81 | 13 |
| power | J-F(1,continuous) | 8 | +0.0196 | 0.0674 | 0.1050 | 106 | 17 |
| power | J-F(1,macro) | 8 | +0.0013 | 0.0633 | 0.0973 | 91 | 15 |
| power | J-F* | 8 | +0.0273 | 0.0986 | 0.1526 | 224 | 36 |
| power | J-F_hindsight | 8 | +0.0013 | 0.0633 | 0.0973 | 91 | 15 |
| power | J-M | 8 | -0.0210 | 0.0564 | 0.0882 | 75 | 12 |
| power | J-OL | 8 | -0.0084 | 0.0612 | 0.0959 | 89 | 15 |
| power | J-T | 8 | +0.0056 | 0.0697 | 0.1100 | 117 | 19 |

| inventory | development mixed | held-out mixed | conservative share |
| --- | --- | --- | --- |
| angle | 5/10 | 2/4 | 0.5 |
| grid | 11/11 | 3/4 | 0.75 |
| offset | 6/11 | 3/4 | 0.5455 |
| power | 5/11 | 3/4 | 0.4545 |

## Depth

- rule: grid (primary, 16 candidates): the deepest roster prefix needed for every #96 grid contrast at delta 0.02; union (angle + grid + offset + power, 60 candidates): the prefix needed for every #96 contrast of offset, angle and power at delta 0.05; levels per family = required mixed members / (2 families x conservative mixed share), rounded up to 20, capped at the roster; conservative share = min(development, held-out) member share
- grid: 200 levels per family (needs 293 mixed members; primary J-F* needs 208)
- union inventories: first 60 levels per family
- expected mixed members: {'angle': 60.0, 'grid': 300.0, 'offset': 65.5, 'power': 54.5}
- fit 240 / policy 60 per family; #110 8 per split per side

## Membership (dispatch order)

| stream | role | levels | branches | wall h | TiB |
| --- | --- | --- | --- | --- | --- |
| issue109-evaluation | final_evaluation | 400 | 11680 | 35.7 | 1.062 |
| issue110-evaluation | final_evaluation | 640 | 10240 | 31.3 | 0.931 |
| issue110-fit | training | 640 | 10240 | 31.3 | 0.931 |
| issue109-fit | training | 480 | 13920 | 42.5 | 1.266 |
| issue109-policy | training | 120 | 3480 | 10.6 | 0.317 |
| total | | 2280 | 49560 | 151.4 | 4.51 |

Workers 16; amortized 11.0 s/branch (cold-start bound); worker-hours 1610.7.

## Window (#113)

- censoring: a shot not in a tail at pre_rest_cap_steps is censored there (native_time_window_limit), exactly where the old player censored it
- clear_and_fail: level_clear and level_fail finalize immediately, as before, also inside a tail
- name: run-to-rest plus a fixed post-rest tail; the old 12 s cap is unchanged for shots not at rest
- pre_rest_cap_steps: 30000
- rest_tail_frames: 50
- tail: stable_entered at step s <= pre_rest_cap_steps no longer finalizes; recording continues to s + rest_tail_steps and then finalizes with terminal reason rest_tail_complete
- tail_cancel: stable_exited during a tail cancels it; recording continues and censors at pre_rest_cap_steps if that step has passed, otherwise the next stable_entered starts a new tail

## Capture stack

- player assembly `ac4182509104d937...` (parent 82db3f42 + #113 window)
- renderer {'renderer': 'llvmpipe (LLVM 22.1.6, 256 bits)', 'version': '4.6 (Core Profile) Mesa 26.1.2'}
- pipeline `sha256:d5a151f1e01a649e...`

## Smoke gate

- **S1_capture_reliability**: typed capture failures over all smoke branches <= 0.05 (the #109 gate)
- **S2_renderer**: every complete branch logs the frozen renderer {'renderer': 'llvmpipe (LLVM 22.1.6, 256 bits)', 'version': '4.6 (Core Profile) Mesa 26.1.2'} (enforced per branch)
- **S3_pixel_identity**: every replay branch's decision frame (paused-before.png) equals the retained #109 frame outside the animated Bird/Pig boxes
- **S4_window_rule**: every complete branch declares (30000, 2500); every trace whose first post-launch stable_entered lies at or before the cap ends with rest_tail_complete exactly 2500 steps after its last stable_entered, or with level_clear / level_fail / censoring after a stable_exited; every other trace ends as before (clear, fail, or censored exactly at the cap)
- membership: 168 branches (160 novelty + 8 replay)

## Gate B (bound, not executed)

- unit {'arm': 'LC', 'encoder': 'E99', 'family': 'hybrid', 'request': 'F-15-macro', 'usable_seeds': [20260908, 20260909, 20260910]}
- target: Gate B (#108): the frozen recipe on the #104 sealed held-out split; grid AUC >= 0.65 with lower bound > 0.55 under the tie-free cost
- controls: ['no-model action prior', 'wrong-anchor control of the selected unit']

## Stop rules

- S1_failure_rate: after >= 200 dispatched branches, cumulative typed-failure share > 0.05 stops the campaign
- S2_storage: free bytes on the output filesystem < 500 GiB (limits.minimum_free_bytes)
- S3_artifact_cap: accounted unique bytes > limits.artifact_bytes (1.25 x projection)
- S4_wall_cap_hours: 302.8
- effect: a stop is terminal: every undispatched branch is recorded as campaign_stopped:<rule>; no outcome-conditioned retry, replacement or re-freeze

## Disclosures

- SERVER_SETUP.md byte-rewrote /p/Project/NovPhy in frozen records (owner decision); the #109 sealed-cohort --validate now fails on its #96 input hash. Inputs here are bound to the current bytes; level files are re-checked against their frozen sha256 at every dispatch
- physics is not bit-identical across machines: engineering re-captures on this server agree with the retained #109 smoke on decision frames and on 16/20 event sequences, but differ in float digits from the bird's first collision; within this server the parent and window players agree on 19/20 full native prefixes
- the #113 tail is cut by level_fail when the launched bird dies after rest (single-bird levels); engineering runs kept 1.7-46 tail frames; the smoke reports the realized settle frames
