# #113 item 1: observable macro predicates (`issue-113-macro-predicates-v1`)

Frozen 2026-10-16T13:31:48Z before any head training. Label module `world_model/data/observable_macro.py` (`DEBOUNCE_FRAMES = 1`). Zero engine seconds.

## Definitions (frozen)

- **rest**: every present, simulated, dynamic body (birds included) has |v| <= 0.01 u/s and |omega| <= 0.01 deg/s (the engine stable_entered thresholds); moving[f] = some native step of frame f's interval (s_(f-1), s_f] is not at rest
- **settle**: settle_Delta(f) = the launched bird was removed (engine entity_destroyed) at or before s_(f-D) AND moving[f-D+1..f] all false, with D = max(debounce, Delta); available iff f >= D
- **structure-change**: structure-change_Delta(f) = S(f) != S(f-Delta), S(f) = the directed non-bird (supporter, supported) entity pairs present on more than half of the native steps of frame f's interval (frame 0: its single step); available iff f >= Delta
- **label_index**: labels are indexed by the landing frame f = t + Delta of a Delta request issued at frame t
- **phases**: ['pre_contact', 'cascade', 'settled']
- **relation_to_110**: #110 ORs single-frame labels over (f-Delta, f]; #113 does not adopt it: structure change compares debounced endpoints, settle requires rest over the whole window

Settle window D = max(1, Delta): Delta 1: 1 frames, Delta 15: 15 frames, Delta 5: 5 frames

## Debounce selection (fit stream + retained captures only)

Rule: smallest debounce k in [1, 2, 3, 5, 10, 15, 20, 25, 30, 35] frames whose relapse rate (share of settle_k frames f with f + 15 <= last frame that have a moving frame in (f, f + 15]) is <= 0.05, pooled over the #104 fit stream and the retained N1, #93 and #94 captures.

| k | settle frames (with 15-frame look-ahead) | relapsed | rate |
|---|---|---|---|
| 1 | 53020 | 0 | 0.00% |
| 2 | 50741 | 0 | 0.00% |
| 3 | 48506 | 0 | 0.00% |
| 5 | 44125 | 0 | 0.00% |
| 10 | 33959 | 0 | 0.00% |
| 15 | 24712 | 0 | 0.00% |
| 20 | 16078 | 0 | 0.00% |
| 25 | 8264 | 0 | 0.00% |
| 30 | 2353 | 0 | 0.00% |
| 35 | 24 | 0 | 0.00% |

Chosen: **1 frames**.

## Prevalence (all phases; positives / available frames)

| source | level type | shots | settle Δ1 | settle Δ5 | settle Δ15 | structure-change Δ1 | structure-change Δ5 | structure-change Δ15 |
|---|---|---|---|---|---|---|---|---|
| n1 | novelty_level_0/type010103 | 74 | 68 / 37298 (0.18%) | 36 / 37002 (0.10%) | 17 / 36262 (0.05%) | 190 / 37298 (0.51%) | 672 / 37002 (1.82%) | 1111 / 36262 (3.06%) |
| n1 | novelty_level_0/type010105 | 104 | 81 / 52278 (0.15%) | 12 / 51862 (0.02%) | 0 / 50822 (0.00%) | 47 / 52278 (0.09%) | 206 / 51862 (0.40%) | 504 / 50822 (0.99%) |
| i93 | novelty_level_0/type010103 | 175 | 122 / 90740 (0.13%) | 56 / 90040 (0.06%) | 26 / 88290 (0.03%) | 568 / 90740 (0.63%) | 1753 / 90040 (1.95%) | 2874 / 88290 (3.26%) |
| i93 | novelty_level_0/type010105 | 216 | 138 / 105515 (0.13%) | 5 / 104651 (0.00%) | 0 / 102491 (0.00%) | 112 / 105515 (0.11%) | 468 / 104651 (0.45%) | 1158 / 102491 (1.13%) |
| i94 | novelty_level_0/type010103 | 135 | 75 / 70310 (0.11%) | 44 / 69770 (0.06%) | 17 / 68420 (0.02%) | 199 / 70310 (0.28%) | 663 / 69770 (0.95%) | 1201 / 68420 (1.76%) |
| i94 | novelty_level_0/type010105 | 160 | 33 / 85040 (0.04%) | 0 / 84400 (0.00%) | 0 / 82800 (0.00%) | 57 / 85040 (0.07%) | 213 / 84400 (0.25%) | 519 / 82800 (0.63%) |
| fit | novelty_level_0/type010103 | 6960 | 31241 / 3564466 (0.88%) | 26897 / 3536626 (0.76%) | 17380 / 3467026 (0.50%) | 15555 / 3564466 (0.44%) | 50127 / 3536626 (1.42%) | 85353 / 3467026 (2.46%) |
| fit | novelty_level_0/type010105 | 6959 | 61846 / 3369070 (1.84%) | 53923 / 3341234 (1.61%) | 37902 / 3271644 (1.16%) | 4465 / 3369070 (0.13%) | 18395 / 3341234 (0.55%) | 45561 / 3271644 (1.39%) |
| policy | novelty_level_0/type010103 | 1740 | 8450 / 882522 (0.96%) | 7251 / 875562 (0.83%) | 4648 / 858162 (0.54%) | 3730 / 882522 (0.42%) | 12227 / 875562 (1.40%) | 20761 / 858162 (2.42%) |
| policy | novelty_level_0/type010105 | 1740 | 16985 / 824848 (2.06%) | 14833 / 817888 (1.81%) | 10473 / 800488 (1.31%) | 957 / 824848 (0.12%) | 4070 / 817888 (0.50%) | 10396 / 800488 (1.30%) |

## Prevalence per phase (Delta = 1 / 5 / 15 positives, pooled over level types)

| source | phase | frames | settle | structure-change |
|---|---|---|---|---|
| n1 | pre_contact | 28878 | 0 / 0 / 0 | 0 / 0 / 0 |
| n1 | cascade | 60727 | 0 / 0 / 0 | 237 / 878 / 1615 |
| n1 | settled | 149 | 149 / 48 / 17 | 0 / 0 / 0 |
| i93 | pre_contact | 55640 | 0 / 0 / 0 | 0 / 0 / 0 |
| i93 | cascade | 140746 | 0 / 0 / 0 | 680 / 2221 / 4032 |
| i93 | settled | 260 | 260 / 61 / 26 | 0 / 0 / 0 |
| i94 | pre_contact | 39464 | 0 / 0 / 0 | 0 / 0 / 0 |
| i94 | cascade | 116073 | 0 / 0 / 0 | 256 / 876 / 1720 |
| i94 | settled | 108 | 108 / 44 / 17 | 0 / 0 / 0 |
| fit | pre_contact | 1964598 | 0 / 0 / 0 | 0 / 0 / 0 |
| fit | cascade | 4889770 | 0 / 0 / 0 | 20016 / 68457 / 130691 |
| fit | settled | 93087 | 93087 / 80820 / 55282 | 4 / 65 / 223 |
| policy | pre_contact | 492002 | 0 / 0 / 0 | 0 / 0 / 0 |
| policy | cascade | 1193413 | 0 / 0 / 0 | 4687 / 16285 / 31128 |
| policy | settled | 25435 | 25435 / 22084 / 15121 | 0 / 12 / 29 |

Full table (per level type x phase x predicate x Delta, with available frames and shots with a positive): `prevalence.csv`.

Compute: 337800 CPU process-seconds, 8013 s wall, 0 GPU s, 0 engine s.
