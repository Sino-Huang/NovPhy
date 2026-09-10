# #76 native-time capture correction: engineering design, before new rendering

The first C2 episode stopped before any launch. Its original frames and failed
attempt are retained; shots2/3 and episodes2–4 were not executed. The 600-step
watchdog represented only **0.24 seconds** in the recovered original game,
whose `TimeManager.asset` has a **0.0004-second** fixed timestep. The original
input handler holds the drag for one second (at least2500 native steps) before
dispatching release. This is an invalid engineering capture window, not an
input-selection, model or gameplay-performance result.

The native timestep will **not** be changed to the older aligned player's
0.02 seconds. Keep the original C2 project, player, plan, sources and failed
attempt unchanged. Stop the remaining C2 attempts, distinguishing this protocol
stop from resource exhaustion. The earlier `environment_seed`/agent-ID
mislabeling is also not retroactively presented as native RNG seeding.

## Costed correction within the approved envelope

A separate native-streaming player/data contract is required. Simply raising
the tick limit in the old monolithic request-71/64MiB JSON path is inadequate:
the old 148-step sample alone occupies about2.6MiB of formatted JSON. Native
one-second drag already entails2500 samples; a12-second window entails30000.
This estimate concerns storage representation, not measured new-game throughput.

- Preserve **every native-step** state, collider/contact inventory and callback
  event, but stream compressed chunks of at most250 samples instead of retaining
  an entire shot or creating fake terminal events at file boundaries.
- Render every50 native steps: one RGB observation per0.02 seconds, with an
  additional genuine off-grid terminal observation when needed.
- A shot window is at most30000 native steps/12 seconds, still at most601
  scheduled RGB observations plus any off-grid terminal observation. Require
  0.04 seconds/100 native steps of the declared low-velocity condition for the
  observer's stable stop; do not silently shorten persistence to0.8ms.
- Bound each decompressed chunk to16MiB. Stream readers must validate contiguous
  coverage, complete inventories, real event/frame references, terminal status
  and observation alignment without assembling a whole native trace in RAM.
- Model horizons will be50/250/750 native steps (20/100/300ms), with equal-time
  endpoints expressed in actual native steps. The old1/5/15-step physics meaning
  is not silently transferred. A4.5-second prediction endpoint is11250 steps.
- Keep the approved cumulative30-minute engineering-capture wall,3GiB aggregate
  RSS,2GiB attempt-data and at-most-eight-capture ceilings. Charge C2's used
  32.69 seconds and136,883,002 bytes to the combined engineering budget. No new
  GPU fitting is authorized or performed by the correction.

The prospective continuation may use only the original unattempted episodes2–4
(at mostfive shots), with exactly their existing templates, seeds, roles and
action schedule. Do not retry episode1 or reassign its unused shots. Its
normal/appearance counterpart comparison is not intact evidence after this
engineering change; no superiority or paired appearance-effect claim is made.

This is the design/cost record **before implementation**, not an assertion that
the streaming player, validators or continuation are ready. Freeze the concrete
new source/player/format and rerun bounded fixtures before rendering any of
those remaining assignments. New protocol failures remain failures, not new
permission to enlarge limits or repeat cases.

The common history/representation still belongs equally to both arms. The
approved later ceilings remain3 GPU-hours common fitting,6 paired-predictor
GPU-hours and3 controller/readiness GPU-hours. The350-lineage training/development
ceiling and12-hour collection cap are separate later work, not permission to
open fresh data. Exact data/fitting schedules, a viable candidate, durable asset
archival and the fresh statistical protocol remain required before #64/#65.
