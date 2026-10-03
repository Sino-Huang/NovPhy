# #113 capture-window spec (`issue-113-capture-window-v1`)

Frozen 2026-10-03T09:03:55Z; zero engine seconds; `python -u -m scripts.derive_capture_window_spec --validate`.

## Rule

- **name**: run-to-rest plus a fixed post-rest tail; the old 12 s cap is unchanged for shots not at rest
- **rest_trigger**: the engine stable_entered event, unchanged: every dynamic Rigidbody2D has |v|^2 <= 1e-4 and |omega| <= 0.01 deg/s for 100 consecutive native steps, after the launch
- **pre_rest_cap_steps**: 30000
- **rest_tail_frames**: 50
- **rest_tail_steps**: 2500
- **hard_limit_steps**: 32500
- **tail**: stable_entered at step s <= pre_rest_cap_steps no longer finalizes; recording continues to s + rest_tail_steps and then finalizes with terminal reason rest_tail_complete
- **tail_cancel**: stable_exited during a tail cancels it; recording continues and censors at pre_rest_cap_steps if that step has passed, otherwise the next stable_entered starts a new tail
- **clear_and_fail**: level_clear and level_fail finalize immediately, as before, also inside a tail
- **censoring**: a shot not in a tail at pre_rest_cap_steps is censored there (native_time_window_limit), exactly where the old player censored it
- **old_window_identity**: for every shot the prefix up to the old stop step is the old trace: the change only removes the stable_entered finalization and adds frames after it
- **terminal_reason**: rest_tail_complete
- **stop_kind**: rest_tail_complete maps to stable_without_clear (scene at engine rest, no clear)
- **manifest_fields**: {'maximum_shot_steps': 30000, 'rest_tail_steps': 2500}

## Evidence per level type (retained traces, every native step)

| level type | shots | stop kinds (old window) | at engine rest | settled frames old -> new (upper) | censored: structure at rest at cap | censored moving at end | collapse shots | last change frame p50/p90/max | bird destroyed in window | frames +% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| novelty_level_0/type010102 | 104 | {'censored': 62, 'stable_entered': 42} | 42 (0.40) | 42 -> 2142 | 61/62 | {'bird': 61, 'pig': 1} | 31 (0.30) | [192.0, 321.0, 492.0] | 39 | 4.0 |
| novelty_level_0/type010103 | 104 | {'censored': 63, 'level_clear': 2, 'level_fail': 18, 'stable_entered': 21} | 21 (0.20) | 21 -> 1071 | 63/63 | {'bird': 63} | 37 (0.36) | [157.0, 407.4, 561.0] | 30 | 1.9 |
| novelty_level_0/type010105 | 104 | {'censored': 61, 'level_clear': 3, 'level_fail': 9, 'stable_entered': 31} | 31 (0.30) | 31 -> 1581 | 60/61 | {'bird': 60, 'bird+block': 1} | 23 (0.22) | [173.0, 307.4, 379.0] | 39 | 3.0 |
| novelty_level_1/type010101 | 103 | {'censored': 55, 'level_clear': 1, 'level_fail': 9, 'stable_entered': 38} | 38 (0.37) | 38 -> 1938 | 55/55 | {'bird': 54, 'none': 1} | 33 (0.32) | [181.0, 244.2, 377.0] | 44 | 3.8 |
| novelty_level_1/type010102 | 83 | {'censored': 43, 'stable_entered': 40} | 40 (0.48) | 40 -> 2040 | 43/43 | {'bird': 43} | 23 (0.28) | [220.0, 336.2, 475.0] | 35 | 5.0 |

## Longer fixed cap (rejected alternative)

- censored shots: 284; structure at rest at the cap: 282; only the launched bird moving: 281
- projected launched-bird death by cap (DESCRIPTIVE projection, censored shots per level type):
  - novelty_level_0/type010102: {'12': 3, '16': 31, '20': 37, '24': 40, '30': 41, '40': 41} of 62 (20 without a projection)
  - novelty_level_0/type010103: {'12': 3, '16': 19, '20': 28, '24': 30, '30': 33, '40': 36} of 63 (25 without a projection)
  - novelty_level_0/type010105: {'12': 2, '16': 24, '20': 32, '24': 36, '30': 38, '40': 40} of 61 (18 without a projection)
  - novelty_level_1/type010101: {'12': 2, '16': 18, '20': 24, '24': 30, '30': 30, '40': 31} of 55 (20 without a projection)
  - novelty_level_1/type010102: {'12': 3, '16': 20, '20': 23, '24': 25, '30': 27, '40': 28} of 43 (13 without a projection)
- frame cost of a longer cap for every censored shot: 16 s +23%, 20 s +45%, 24 s +68%, 30 s +102%, 40 s +159%
- the frozen rule adds at most 3.4% frames

## Rationale

- the macro events are absent because a stable shot stops on the first rest step (0 settled frames after it), not because the cap is short
- at the 12 s cap the structure (every non-bird body) is already at engine rest in almost every censored shot; the only moving body is the launched bird rolling on the ground (see per_level_type.censored_moving_at_end)
- a longer cap therefore adds frames of a lone rolling bird; reaching engine rest needs the bird to fall below 0.01 u/s and die (projection table), at a capture and storage cost proportional to the added seconds
- structure changes stop well before the cap (last non-bird support change), so collapse yield is set by whether the shot reaches the structure (candidate inventory), not by the window
- 50 tail frames (1.0 s) hold a settle debounce of up to 35 frames plus the Delta = 15 macro request horizon and a 15-frame temporal-head history inside the window, and cover the bird-death check that follows rest (storm onset on novelty level 8 needs the first bird destroyed)

## Not covered

- no retained capture exists for novelty levels 2-8 (or for level 0/1 families outside type010101/02/03/05); their settle, collapse and storm-onset yields are measured by the #104 rendered smoke under this rule and reported there, not estimated here
- the #113 item-1 predicate definitions (settle debounce, per-Delta structure change); the rule only guarantees the frames those definitions need
