# Unity 2019.4 LTS server resume

## Durable state

- Plan: `.omo/plans/world-model-physics-instrumentation.md`
- Progress: **Todos 1-10 complete and independently verified. The final verification wave F1-F4 has run
  against one exact pinned commit.** Unity licensing is resolved; the exact editor builds, and the staged
  player runs live.
- Migration worktree: `/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4`
- Migration branch: `physics-unity-2019.4`
- Migrated project: `/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/tasks/task_template_designer`
- Exact Editor: Unity `2019.4.41f2 (6b23d448b533)`
- Editor executable: `$HOME/.local/share/novphy-unity/2019.4.41f2-6b23d448b533/editor/Editor/Unity`
- Official archive SHA-256: `5426954a630036f59d6d1b40e548bc4d0a47a092475a7530db8013ff5c4aea63`
- Editor executable SHA-256: `32252cb8eca087743e500596e093061a906203703915c2d3c2fb2f8a372bc150`

Never open the canonical `tasks/task_template_designer` with Unity 2019.4 and never build into
`sciencebirdsgames/Linux`.

## Base environment

```bash
export UNITY_2019_4_41F2="$HOME/.local/share/novphy-unity/2019.4.41f2-6b23d448b533/editor/Editor/Unity"
export MIGRATED_UNITY_PROJECT="/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/tasks/task_template_designer"
export UNITY_LTS_LIBS="/tmp/opencode/unity-2019.4-libssl1.1/root/usr/lib/x86_64-linux-gnu:/tmp/opencode/unity-2019.3.4f1-libs/root/usr/lib/x86_64-linux-gnu:/tmp/opencode/unity-2019.3.4f1-libs/root/usr/lib"
```

The GConf and OpenSSL 1.1 compatibility bundles are private and checksum-verified. They must remain
scoped through `LD_LIBRARY_PATH`; do not install them system-wide. **They live under `/tmp`, so a reboot
clears them — check they still exist before any Unity invocation and re-extract if not.**

## Licensing: resolved

Unity Hub registration was completed and exact `2019.4.41f2` became visible, after which activation
succeeded and Todo 2 proceeded. Builds and EditMode runs now work without further licensing action.

Historical detail, retained because it cost significant effort: Ubuntu 24.04 lacks legacy
`libgconf-2.so.4`, the bundled Licensing Client needed `DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1` to get
past an ICU crash and a private Focal `libssl1.1` to get past `No usable version of libssl was found`,
and six bounded probes reached live IPC and entitlement refresh but failed activation with
`Missing or bad username or password` until Hub was registered with a path whose parent directory is
literally the version string. Hub 3.20.0 derives the version from
`basename(dirname(dirname(editorPath)))` and never inspects the binary, so a path ending in
`.../editor/Editor/Unity` is rendered as version `editor` and cached as unsupported.

Official source on the Personal boundary:
https://support.unity.com/hc/en-us/articles/23081758513172-I-can-t-open-projects-created-in-Unity-2019-4-and-below

## Running EditMode tests

Two operational rules, both learned the hard way:

1. **Do not pass `-quit` alongside `-runTests`.** Unity exits batchmode before running any test and
   writes no XML, while still exiting 0. That looks like success and is not.
2. **Run partitioned by test class with `-testFilter`, and read the XML, not the exit code.** Every run
   ends with a SIGABRT or SIGSEGV inside `CefBrowserMessageLoop::DoMessageLoopIteration` during headless
   editor shutdown, so the process exits 134 or 139 *after* writing a complete NUnit receipt. The XML
   `result` attribute is the authority.

```bash
DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1 \
LD_LIBRARY_PATH="$UNITY_LTS_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$UNITY_2019_4_41F2" -batchmode -nographics -projectPath "$MIGRATED_UNITY_PROJECT" \
  -runTests -testPlatform EditMode -testFilter <TestClass> \
  -testResults /abs/path/<TestClass>.xml -logFile /abs/path/<TestClass>.log
```

The eight EditMode classes are `ABBirdLaunchTests`, `ABGameWorldLifecycleTests`, `LegacyGroundTruthTests`,
`PhysicalEntityRegistryTests`, `PhysicalShotRecorderTests`, `PhysicalSnapshotExporterTests`,
`PhysicsCaptureProtocolTests`, and `PhysicsShotRecorderTests`, totalling 44 tests. Run all eight — a
partial set silently hid a permanently failing test through many prior receipts.

**EditMode does not raise `Awake`.** A component added with `AddComponent` in an EditMode test has
un-run lifecycle, so `ABGameObject._rigidBody` is null and `PhysicalSnapshotRuntime.Active` is unbound,
which makes every static record callback a silent no-op. Drive the real `Awake` chain reflectively
rather than assigning private fields.

## Unity logs contain licensing identity

`-logFile` output includes the Licensing Client channel (`LicenseClient-<user>`), a Unity-masked serial,
and per-run `LICENSE SYSTEM` timestamps. Never publish a raw Unity log. `scripts/redact_unity_log.sh`
replaces those lines in place, fails closed if any survive, and renames atomically;
`build_physics_player.sh` uses it for the stage copy.

## Cost of the protected-root receipt

`scripts/smoke_protection.nested_manifest_digest` enumerates the active data root, which holds
**14,432,052 files across 535 GB**. One scan takes minutes and the smoke driver does two per run. Budget
at least 1800 s for any command that invokes it. A smoke timeout is an inconclusive run to repeat with a
larger bound — never grounds for reconstructing a report by hand.

## Building and publishing

```bash
bash scripts/build_physics_player.sh                      # publishes into sciencebirdsgames/physics-v1
NOVPHY_PHYSICS_STAGE=/tmp/some-stage bash scripts/build_physics_player.sh   # isolated build
```

Builds are ~25 s with a warm `Library` and are byte-reproducible. `provenance.json` embeds
`project.git_head` and `project.git_tree`, so **any commit changes the archive SHA even when no shipped
byte of the player changes.** Expect to rebuild, republish, and re-smoke after any re-pin.

Migration provenance and evidence live under:

`/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/.omo/evidence/world-model-physics-instrumentation/`
