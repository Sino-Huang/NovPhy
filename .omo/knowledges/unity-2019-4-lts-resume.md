# Unity 2019.4 LTS server resume

## Durable state

- Plan: `.omo/plans/world-model-physics-instrumentation.md`
- Progress: Todo 1 complete; Todo 2 in progress; all later tasks blocked by Todo 2.
- Migration worktree: `/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4`
- Migration branch: `physics-unity-2019.4`
- Migrated project: `/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/tasks/task_template_designer`
- Exact Editor: Unity `2019.4.40f1 (ffc62b691db5)`
- Editor executable: `$HOME/.local/share/novphy-unity/2019.4.40f1-ffc62b691db5/editor/Editor/Unity`
- Official archive SHA-256: `c592296df9dd888e5239ad7dda276bb718b33075c679a2fec9c080764644435f`
- Editor executable SHA-256: `1fdc5220ec0cc3e7d2832412eb2ed39bb0ad9ea0a712fa4619b6f82045865918`

Never open the canonical `tasks/task_template_designer` with Unity 2019.4 and never build into `sciencebirdsgames/Linux`.

## Base environment

```bash
export UNITY_2019_4_40F1="$HOME/.local/share/novphy-unity/2019.4.40f1-ffc62b691db5/editor/Editor/Unity"
export MIGRATED_UNITY_PROJECT="/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/tasks/task_template_designer"
export UNITY_LTS_LIBS="/tmp/opencode/unity-2019.3.4f1-libs/root/usr/lib/x86_64-linux-gnu:/tmp/opencode/unity-2019.3.4f1-libs/root/usr/lib"

LD_LIBRARY_PATH="$UNITY_LTS_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$UNITY_2019_4_40F1" -batchmode -nographics -quit -logFile -
```

The GConf compatibility bundle is private and checksum-verified. It must remain scoped through `LD_LIBRARY_PATH`; do not install it system-wide.

## Current licensing-client failure

The user reports that Unity Hub/Personal activation was completed, but no successful batch entitlement result exists yet.

Observed sequence:

1. Without compatibility flags, the bundled Licensing Client terminated with `Couldn't find a valid ICU package installed on the system`.
2. `DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1` bypassed the ICU crash.
3. The client then terminated with `No usable version of libssl was found`.
4. IPC timed out, Unity fell back to legacy licensing, and legacy activation printed `Missing or bad username or password`.

Ubuntu 24.04 provides `libssl.so.3`/`libcrypto.so.3`; the old Licensing Client expects OpenSSL 1.1. The next session should download the official Ubuntu Focal package `libssl1.1` version `1.1.1f-1ubuntu2.24` from Canonical/Launchpad, verify it, extract it into the existing private compatibility root (or another private prefix), and prepend the extracted directory to `UNITY_LTS_LIBS`. Do not use `dpkg -i`, add Focal repositories, or modify host OpenSSL.

Then run a bounded probe with both compatibility settings and a real log path:

```bash
DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1 \
LD_LIBRARY_PATH="$UNITY_LTS_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$UNITY_2019_4_40F1" \
  -batchmode -nographics -quit \
  -logFile "/tmp/opencode/unity-2019.4-license-probe.log"
```

Do not accept exit code alone. Inspect the log and require the Licensing Client to stay alive, connect through IPC, and report a recognized Hub entitlement. `Pro License: NO` is normal for Personal; the decisive failure signals are Licensing Client crash/timeout, fallback to legacy activation, or activation error.

## Todo 2 after licensing succeeds

1. Import only `$MIGRATED_UNITY_PROJECT` with exact Unity 2019.4.40f1 and audit migration changes.
2. Run focused Todo 2 EditMode tests twice, then all EditMode tests, producing `task-2-unity.xml` and a dedicated log.
3. Produce legacy `SymbolicGameState.GetGTJson()` parity evidence.
4. Recompute canonical-project and production-player manifests and require exact equality with their baselines.
5. Independently verify the DoneClaim before checking Todo 2 or proceeding to Todo 3.

Migration provenance and existing evidence live under:

`/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/.omo/evidence/world-model-physics-instrumentation/`
