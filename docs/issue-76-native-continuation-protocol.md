# #76 prospective native-time continuation of unused C2 assignments

This applies the [native-time correction](issue-76-native-time-capture-amendment.md)
only to the three original, unattempted C2 episodes. The failed normal episode1
is not retried; its unused shots are not reassigned. This is engineering
continuation under the operator-approved budget, not a new comparison or
permission to access fresh/final data.

| Original assignment | Condition | Generator seed | Unity RNG seed | Fixed shots |
| --- | --- | ---: | ---: | --- |
| canonical-02 | appearance-category/type010102 | 760710001 | 760720001 | (-80,+10), (-60,+45), (-10,+80) |
| canonical-03 | normal rolling/type010103 | 760710002 | 760720002 | (-80,+10) |
| canonical-04 | normal falling/type010204 | 760710003 | 760720003 | (-80,+10) |

Exact templates, workbook rows, XML, scenario identities and calibration roles
are copied unchanged from C2's existing freeze. All remain excluded from future
fitting and fresh/sealed membership. Each transport action has tap0ms and
release600ms; the original game's actual drag handler uses its fixed1-second
hold. Do not describe the transport argument as a measured600ms engine hold.
The public slingshot actuator anchor is the only geometry used to execute these
fixed actions. Agent ID760001 is separate from the actual Unity RNG seed.

## Native player and complete physical evidence

Use the separate `novphy-canonical-native-v1/player-build-01` player, not a
modified C2 player. Preserve the 2019.3.4f1 reference assets/gameplay behavior
and the explicit2019.4.41f2 rebuild; do not claim cross-version equivalence.
The0.0004-second physics timestep is unchanged. Ten Unity fixtures must pass,
including the microstep window, bounded chunk drain, cross-chunk event retention
and failure-without-fake-terminal cases. The actual C# fixture must also pass
the Python native reader. Its synthetic fixture is not a research-data member.

Every native-step state/collider/contact inventory and callback event is
retained in ordered gzip chunks of at most250 samples/16MiB decompressed. The
latest sample remains buffered until the next step so late same-step callbacks
can update it. Chunk boundaries do not create terminal events. The small native
manifest records the real terminal or an explicit failed capture. The old
monolithic request71 endpoint is deliberately unavailable for a drained trace.

RGB stride is50 native steps/0.02 seconds, with a genuine forced terminal frame
when off-grid; no interpolation or invented frames. Intended viewport is640x480
native RGB8. The observer's low-velocity stopping persistence is100 native
steps/0.04 seconds, not the old two-tick/0.8ms interpretation. Native gameplay
win/fail events remain authoritative and distinct from stable stopping.

At most30000 native steps/12 seconds per shot; at most601 RGB observations
including the initial/terminal boundary. Each complete segment must have one
actual launch, fully contiguous native evidence and exactly aligned RGB.
Each shot gets a new capture-ID directory. The same episode's physical level
and clock continue without reset; unobserved inter-shot time remains a gap.
Stop at native won/lost state, first failure or the existing shot allowance.

## Unchanged cumulative envelope and disposition

At mostfive new captures across these assignments; no retries or replacements.
One worker/private display and runtime/ports/XDG directory per episode. No
`-nographics` rendering. Per-episode wall420s, per-shot wait180s. Combined C2+
continuation active wall1800s, peak aggregate RSS3GiB and attempt data2GiB.
Charge C2's32.69 seconds and136,883,002 bytes to those totals. Global resource
limits stop the stage; preserve all failure prefixes and pending assignments.
Progress logs show native chunk count, RGB count, elapsed/combined time, RSS
and ETA. Resume never replays completed or interrupted assignments.

No parser, history model, predictor or controller is fitted or evaluated here.
No numerical success-power claim, normal/novel effect or model-superiority
claim is licensed by a pipeline pass. The unmatched failed C2 normal attempt
remains in the combined report. A viable symmetrically fitted candidate,
archived assets and the later fresh statistical/gameplay protocol are still
required before #64/#65.
