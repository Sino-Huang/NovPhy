# Action geometry and next development hypothesis

## Observed evidence

The existing `trajectory_guided` helper consumes visible polygons. It is not
an RGB-only policy by itself; an agent-only use would require a learned parser
adapter. Do not feed engine geometry to it and call that shared learned
perception.

`ABBird.DragBird` caps drag displacement to a radius before launch. The first
native sample after each `bird_launched` event in source 001's twelve retained
replays gives these world-coordinate velocities (units/second):

| Action | vx | vy |
| --- | ---: | ---: |
| 1 | 1.207 | -9.929 |
| 2 | 5.950 | -8.039 |
| 3 | 8.054 | -5.928 |
| 4 | 8.924 | -4.513 |
| 5 | 5.374 | -0.439 |
| 6 | 9.998 | -0.135 |
| 7 | 9.999 | -0.075 |
| 8 | 9.999 | -0.052 |
| 9 | 1.231 | 9.922 |
| 10 | 6.027 | 7.977 |
| 11 | 8.110 | 5.847 |
| 12 | 8.959 | 4.438 |

The event-step samples themselves show zero velocity; these measurements use
the immediately following native step, not that earlier sample. No new capture
or model inference was performed. The evidence concerns source 001 and must
not be presented as a measured launch audit of all five families.

Four grid actions launch downward, four approximately horizontally, and four
upward. Actions 6–8 have nearly identical capped horizontal speeds, although
their small vertical components differ. Twelve command values therefore do not
provide twelve well-spaced useful launch directions. This is a coverage
limitation, not proof that action geometry caused all prediction failures.

## Prospective candidate, not execution authorization

A shared outcome-independent candidate grid can use twelve upward pull angles
5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75 and 82 degrees, at commanded radius
80 pixels, rounding coordinates to integers. This gives:

`(-80,7), (-78,17), (-76,26), (-72,35), (-67,44), (-61,51),
(-55,59), (-47,65), (-39,70), (-30,74), (-21,77), (-11,79)`.

Keep release 600 ms and tap 0 ms, and verify legality through the shared action
adapter. No pig positions, engine outcomes or model preference select these
angles. Any future model comparison must use the same generator for both arms
and retain the strongest simple-prior comparison. The old grid results and
failed readiness remain unchanged.

Before rendering, a separate exact training-only inventory and source-bound
execution protocol must define membership, resource costs, capture endpoints,
comparability, failure handling and the decision this pilot can inform. Reusing
the five assigned training lineages would remain a small development diagnostic,
not a precision study or independent evidence of gameplay advantage. It cannot
authorize a new optimizer sweep or fresh #76 evaluation. A successful action
design still requires supported dynamics training, terminal-aware evaluation,
useful multi-shot/adaptive gameplay, and all original advancement contrasts.
