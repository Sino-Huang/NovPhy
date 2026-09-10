# #76 C2: approved engineering, runtime migration in progress

The operator approved the [costed proposal](issue-76-canonical-migration-proposal.md)
on 2026-09-10: 8 hours of canonical-player engineering, 16 further hours of
history/refit integration, and the stated collection/compute/storage ceilings.
Scoped prerequisite/new-work commit and push are authorized. The durable
large-asset archive destination is still unspecified. This approval is not
an execution freeze for training, fresh evaluation or #64/#65.

Completed prerequisites were committed and pushed to `env` as
[`01322fe`](https://github.com/Sino-Huang/NovPhy/commit/01322fe).
The in-progress C2 changes are separate. The combined #70–#76 Python regression
run passed all 160 tests; five additional history-dynamics tests subsequently
passed with the new history/preparation tests (16 tests in that focused run).

## Reference recovery

Recovered the bundled original `sciencebirdsgames/Linux` with AssetRipper 2.0.0
into `.local-artifacts/issue-76-canonical-recovery-v1/ExportedProject`.
The recovered project is separate from both the original player and the aligned
Unity project. Its source version is **2019.3.4f1**. Building with the required
2019.4.41f2 editor is an explicit migration requiring runtime verification, not
evidence of identical simulation.

The recovered `ABGameWorld.DecodeLevel` resolves PinkBigPig through
`ABWorldAssets.NOVELTIES`, which is loaded from `BenchmarkNovelties`; it does not
require PinkBigPig to be in the ordinary-pig dictionary. The earlier C1 aligned
implementation omitted this original lookup path.

Recovered/decompiled behavior was checked against the original assembly's IL:

- Both BasicBig and PinkBigPig use `DieOnBirdHitCountPig`, with threshold 3.
- Its startup records the identities of the birds present at startup.
- A collision from a previously uncounted startup bird increments the count.
  At the threshold it calls the pig death path, then the ordinary pig collision
  callback. Repeated contact from the same bird does not increment it.
- The original pig damage is relative speed, not the aligned player's
  mass/defense formula. Life initializes in Awake, not after the aligned
  two-second immortality coroutine. Original death destroys immediately rather
  than waiting for the aligned particle delay.
- Serialized BasicBig life is 100,000,000; PinkBigPig life is 10,000. Both have
  mass 1.5, gravity scale 0.5, angular drag 0.05 and the same 15-point polygon.
  Their collision-detection settings still differ. These are not certified
  appearance-only equivalents.

UnityPy 1.25.3 plus TypeTreeGeneratorAPI 0.0.10 were used only in the isolated
asset-tool directory to inspect serialized custom fields. No model environment
or checkpoint was changed. Assembly-generated MonoBehaviour headers can contain
misdecoded script pointers; script identities must be resolved from the native
header, not copied from those generated header fields.

AssetRipper's default export includes dummy custom shaders; recovery alone
therefore does **not** establish rendered-observation equivalence. Custom
shader usage, resource bindings, compile compatibility, callbacks and actual
graphics remain required checks before an engineering capture freeze.

## Implemented commands

Initialize the `novphy` conda environment and source `env.sh` first.

```bash
python -u -m scripts.run_issue_76_canonical_player --dry-run
python -u -m scripts.run_issue_76_canonical_player --prepare
python -u -m scripts.run_issue_76_canonical_player --test
python -u -m scripts.run_issue_76_canonical_player --build
```

Preparation refuses to overwrite an existing working project. Tests/builds
print elapsed-time progress every 15 seconds, with a 30-minute per-command
ceiling, and retain detailed Unity logs. These commands do not collect data,
fit models or establish candidate readiness. The build is not yet an
instrumented, validated replacement for the old player.

The recovered project compiled with the required Unity editor, and all three
reference-asset fixtures passed. Initial editor startup failures (missing
display, repository-local license lookup), a case-colliding resource import,
an overlapping-editor refusal and the interrupted slow import are retained in
`.local-artifacts/issue-76-canonical-player-v1`. No gameplay captures were made.
The native-filesystem run passed in 543.8 seconds, including first import.

The pristine recovered player built successfully in 99.3 seconds. The separate
observer port compiled, then the expanded seven-fixture suite passed (including
distinct-bird counting without duplicate/non-bird increments, XML identity
propagation, manual clock and actual Unity RNG seeding). The instrumented build
with the repository's existing solid-color shader completed in 93.0 seconds
at `player-build-03`. Earlier builds and the failed fixture/license attempts
remain retained; the current player is still awaiting rendered engineering
validation. The latest focused Python run passed 22 tests, including capture
failure/resume and actuator-speed regressions.

The active working project is `/home/sukaih/.cache/novphy-canonical-player-v2`;
add `--work-dir /home/sukaih/.cache/novphy-canonical-player-v2` to the commands
above. The default working directory is on the repository's mergerfs mount;
the native filesystem avoids its substantial asset-import overhead.
Its preparation changes the editor version explicitly, adds the test-framework
dependency and copies the build helper and reference-asset fixtures. The
pristine recovered project remains untouched.

The redundant lowercase `Assets/resources` contains extra exported copies of
runtime asset-bundle contents. It is retained outside the build as
`recovered-bundle-exports`; unchanged original bundles remain in StreamingAssets.
This removes the case collision with the actual `Assets/Resources` tree without
discarding the source evidence.

## Shared history implementation, not a fitted candidate

The successor uses a 64-value GRU memory over permitted observation/action
events, appended to the 236-value visual carrier. The two dynamics arms
independently predict this same 300-value carrier. The pure arm instantiates no
symbolic heads or mode embedding; selected hybrid modes decode predicates from
the current carrier. Parameter-count matching chooses the continuous width
without outcome access. This is not a claim of equal active compute or latency.

History records distinguish observation-present, executed-action and padding.
Action-only events update memory, missing observations cannot leak through
masked fields, padding holds memory, and elapsed time is measured between valid
events on the agent clock. A new episode resets memory; a new shot does not.
Tests verify streaming/prefix equivalence, no future leakage, action effects,
masking, resets and gradient participation. No engine hit counter is an input.

The shared encoder is to be fitted once per paired seed and frozen identically
for both arms. Its complete standalone fitting/deployment cost is charged to
each arm's method accounting; shared data processing is reported separately,
not erased. The approved ceilings remain 3 GPU-hours for common perception/
history, 6 for both predictors across three paired seeds, and 3 for controllers
and readiness. Exact data membership and update schedules must still be frozen
before collection/fitting. No representation or predictor fitting has occurred.
