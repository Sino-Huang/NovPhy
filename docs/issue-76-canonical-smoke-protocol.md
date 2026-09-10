# #76 C2 prospective canonical-player engineering smoke

This consumes **at most eight new shot captures**, within the operator-approved
8-hour canonical-player engineering allowance. It is not eight three-shot
episodes. The later training/development collection and fitting are not started
by this protocol. No fresh/final access, model comparison or power claim follows.

## Exact membership and intervention schedule

| Episode | Base cluster | Condition / template family | Generator seed | Unity RNG seed | Maximum shots |
| --- | --- | --- | ---: | ---: | ---: |
| canonical-01 | appearance | normal / type010102 | 760710001 | 760720001 | 3 |
| canonical-02 | appearance | appearance-category / type010102 | 760710001 | 760720001 | 3 |
| canonical-03 | rolling | normal / type010103 | 760710002 | 760720002 | 1 |
| canonical-04 | falling | normal / type010204 | 760710003 | 760720003 | 1 |

The normal/appearance pair remains one analysis/exposure cluster. All members
are calibration engineering data, excluded from subsequent fitting, fresh and
sealed evaluation. Single-force and sliding are not newly tested by C2. The
exact template/workbook row, generated XML and scenario identity are copied
into the machine-readable plan before any rendering, using the C1 inventory's
condition-aware materializer. Only red birds are allowed by this port's current
spawn-identity support. Recorded prior exposure includes the earlier global
metadata inventories and C1's membership; no private final outcomes are opened.

The paired episodes use the predeclared relative drags **(-80, +10)**,
**(-60, +45)** and **(-10, +80)** in that order: three upward-right launch
directions, not an outcome-selected action sweep. The one-shot episodes use
the first direction. All use tap time **0ms** and release time **600ms**,
anchored only to the visible/public slingshot actuator. Before C2 freeze, the
actuator transform and C1's recorded launch velocity showed that C1's
(-10, -80) drag launched downward; this sign/direction audit motivates the new
engineering schedule. No C2 outcome was rendered to choose these actions.
Preparation and execution remain at simulation speed
**1**, not the older helper's speed-50 startup. The normal fast-shot interface
is used; the legacy ground-truth-batch shot command's automatic pause/frame
cap is not used. The observer independently records request-71 physics and RGB.

No engine pig/block geometry, health or hit counter selects an action. The
bridge agent ID is separately fixed to 760001. Unlike C1's mislabeled
`environment_seed` configure argument, `NOVPHY_ENVIRONMENT_SEED` really seeds
Unity before scene load and emits a seed receipt in the Unity log. A Unity
fixture checks the RNG state, not merely the presence of that log message.

## Runtime, observation and continuity

The player is recovered from the original 2019.3.4f1 game, rebuilt with the
required 2019.4.41f2 editor, and instrumented in a separate project. Use the
declared `player-build-03` output, observer-source receipt and shader repair.
The earlier players and all failed editor/C1 attempts remain preserved.
All seven resource/behavior/XML/clock/seed fixtures must pass before freeze.
Recovery and compilation do not establish cross-version physical or visual
equivalence. The actual big-pig normal/novel life and collision-detection
differences are retained; this is not an appearance-only treatment claim.

One graphics-enabled process per episode, with private engine/agent/physics
ports and work directories. Intended viewport is 640x480, native RGB8 agent
observations, stride one fixed step. The observer samples immediately after
the original manual physics call, using its explicit step clock. Preserve all
raw aligned PNG/metadata, derived observations, request-71 traces and logs.

The physical level is not reset between shots. Each shot gets a distinct
capture-ID directory, and the episode clock must strictly increase across
segments. Later segments are continuation captures, not independent rollouts
from the authored XML. Existing low-level wire fields named `rollout_id` or
`rollout_identity` carry the shot-segment identity; the new episode manifest,
not the old independent-rollout cohort reader, defines their analysis grouping.
Exact source/observation alignment and one actual launch
per completed segment are required. Inter-shot unobserved intervals remain
explicit clock gaps; no frames are invented. The future shared history carries
across shots, but no parser/history/predictor is fitted or run in C2.

Stop an episode at native won/lost state, its shot allowance or its first
technical failure. Structurally unavailable later shots are recorded as such;
do not reload/restart the level to fill them. No retries, replacements, seed
search or solvability screening. The initial `--smoke-test` runs only episode1;
`--run-smoke` resumes the same freeze and never replays completed/interrupted
episodes.

## Limits and interpretation

At most 30 minutes total active work, 420 seconds per episode, 180 seconds per
shot wait, 3GiB aggregate process RSS, 2GiB runtime/capture/derived attempt data,
one worker and zero GPU fitting. A 0.25-second watchdog stops at fixed-step
offset600 (601 frames including the initial frame). Any polling overshoot is
retained and reported, never truncated or presented as compliance with the cap.
All failures consume their assigned attempt. Global limits stop the stage.

This stage tests the engineering data path and multi-shot continuity, not a
viable learned candidate. Completion requires reporting all assigned episodes,
including failures and unused shots. A pipeline pass does not authorize #64/#65.
The later symmetric refit remains within the separately approved resource
ceilings and requires its exact training/development freeze and operator-run
commands. Fresh power/margins and archival remain later gates.
