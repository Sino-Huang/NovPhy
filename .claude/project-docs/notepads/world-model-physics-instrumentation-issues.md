# Issues — world-model-physics-instrumentation

Problems and gotchas encountered during work on this plan.

_Auto-scaffolded by /start-work. Append new entries below - never overwrite._

---

## 2026-07-31T22:13:36+10:00 - Todo 1

- Python LSP diagnostics could not run: `basedpyright` is not installed and the user previously declined installation. `py_compile`, `compileall`, the focused tests, and the strict no-excuse checker were used as available static/runtime gates.
- Repository-wide `git diff --check` reports pre-existing trailing whitespace only in unrelated user-modified proposal/presentation files. No task-owned file was implicated.

## 2026-08-01T00:17:04+10:00 - Todo 2

- `UNITY_2019_3` is unset; `unity-editor`, `Unity`, and `unityhub` are absent from `PATH`; `/opt/Unity/Editor/Unity` and `/home/sukai/Unity/Hub/Editor/2019.3.4f1/Editor/Unity` do not exist. Both the pre-production legacy characterization attempt and final required EditMode command failed before Unity could start.
- No `task-2-unity.xml` was produced, so zero-failure XML, the runtime legacy SHA-256 observable, dynamic/static assertions, and focused rerun remain unverified.
- C# LSP diagnostics are unavailable because `csharp-ls` is not installed, and installation is prohibited by the task. No `mcs`, `csc`, or `dotnet` compiler is available as a secondary syntax gate.

## 2026-08-01 - Todo 2 resumed verification

- Rechecked all local execution surfaces for the pinned `2019.3.4f1 (4f139db2fdbd)` editor: the environment variable, `PATH`, standard system/user paths, Unity Hub manifests, package/snap installs, `/mnt/array`, and cached Docker images contain no usable Unity editor.

## 2026-08-01 - Todo 2 official editor provisioning

- Downloaded `https://download.unity3d.com/download_unity/4f139db2fdbd/LinuxEditorInstaller/Unity-2019.3.4f1.tar.xz` with resumable `curl` to `/tmp/opencode/Unity-2019.3.4f1.tar.xz`.
- Official response metadata matched `Content-Length: 1555657836`, `ETag: "cbfaf57ab22561677cfe35fdc1eb45fd"`, and `Accept-Ranges: bytes`. The local archive has exactly `1555657836` bytes, passed `xz --test`, and has SHA-256 `11687f1ada2826c363991c01c6703fe56384657ef3349e1194dee5f941949ca8`.
- Extracted the complete archive to `/tmp/opencode/unity-2019.3.4f1`; `Editor/Unity` is executable and embeds the exact identity `2019.3.4f1 (4f139db2fdbd)`.
- The bounded version and required EditMode invocations both stopped in the dynamic loader before Unity initialization: `libgconf-2.so.4: cannot open shared object file: No such file or directory`. No Unity log, test XML, compilation, or license check was reached. Product code was not edited or committed.

## 2026-08-01 - Todo 2 private Focal GConf runtime

- Downloaded official Ubuntu Focal packages `libgconf-2-4_3.2.6-6ubuntu1_amd64.deb` and `gconf2-common_3.2.6-6ubuntu1_all.deb` under `/tmp/opencode/unity-2019.3.4f1-libs/debs` and extracted them with `dpkg-deb -x` under the private `root` directory. No maintainer scripts or host integration ran.
- Published SHA-256 values matched exactly: `libgconf-2-4` = `6d153642be0fab4d79633ded04157986d2648372e84ff1f94e41eeef4d880565`; `gconf2-common` = `ba4f1afb39d3f91d385bfb4a0359a8fcba6f15df2f6fa2d7e05a29adf6e6f341`.
- Scoped `ldd` resolved `libgconf-2.so.4` from the private extraction and reported no unresolved dependencies. Full output is preserved at `/tmp/opencode/unity-2019.3.4f1-libs/ldd.txt`.
- Bounded version and required EditMode runs reached Unity licensing with child-only private `LD_LIBRARY_PATH` and `XDG_DATA_DIRS`. No valid existing license was available; legacy fallback reported `Failed to activate/update license Missing or bad username or password`. No credentials or license material were supplied, no XML was generated, and compilation/runtime parity were not reached.
- Non-sensitive logs are preserved at `/tmp/opencode/unity-2019.3.4f1-version-private.log` and `/tmp/opencode/unity-2019.3.4f1-task-2-private.log`. Product code was not edited or committed.

## 2026-08-01 - Todo 2 bundled static compilation

- Unity 2019.3.4f1 bundles Mono JIT `5.11.0`, mcs `5.11.0.0`, and Roslyn csc `2.9.1.65535 (9d34608e)`. The project has no `Library/ScriptAssemblies`, `.asmdef`, package manifest, or cached `Assembly-CSharp.dll`, so no stale project assembly was used.
- Production response `/tmp/opencode/unity-2019.3.4f1-todo2-production.rsp` compiles all 62 non-Editor default-assembly scripts, including the five physical snapshot files and modified `ABGameWorld.cs`, against Unity's `unityjit` profile, Unity 2019.3 managed modules, UnityEditor/UI, SimpleJSON source, and the project WebSocket dependency. Roslyn exited `0`; nine warnings are pre-existing legacy warnings outside Todo 2 behavior.
- Test response `/tmp/opencode/unity-2019.3.4f1-todo2-tests.rsp` compiles all three Todo 2 Editor tests separately against the fresh production DLL, Unity test-runner assemblies, and the editor-bundled `com.unity.ext.nunit-1.0.0` custom NUnit framework. Roslyn exited `0` with no test diagnostics.
- Consolidated evidence is `/tmp/opencode/unity-2019.3.4f1-todo2-compile.log` with `DoneClaim=STATIC_COMPILE_PASS_RUNTIME_UNVERIFIED`. This rejects syntax/reference API defects only; licensing still prevents EditMode execution, XML, runtime behavior, and legacy parity verification. No product source was changed or committed by this check.

## 2026-08-01 - Todo 2 license-state retry

- Presence-only checks found all standard Unity license files and the `UNITY_LICENSE`, `UNITY_LICENSE_FILE`, `UNITY_SERIAL`, `UNITY_EMAIL`, `UNITY_USERNAME`, and `UNITY_PASSWORD` environment names absent; no values or license contents were inspected.
- One bounded private-runtime initialization probe exited `1` after legacy fallback with the unchanged non-sensitive blocker `Failed to activate/update license Missing or bad username or password`. Sanitized evidence is `/tmp/opencode/unity-2019.3.4f1-license-retry.safe.log`.
- Because initialization did not obtain a usable license, the exact EditMode command was not run, no XML was generated, and source remained uncommitted.

## 2026-08-01 - Todo 2 current licensing compatibility

- Current official Unity licensing guidance confirms that Unity Personal has no serial key and cannot use manual `.alf`/`.ulf` activation. Named User Licensing supports the 2019 stream only from Unity 2019.4.27 onward, so it cannot activate the pinned pre-LTS Unity 2019.3.4f1 editor.
- Keeping the pinned editor requires a legitimate serial-based entitlement that Unity confirms is compatible with 2019.3.4f1. The supported workflow is to generate a machine-specific `.alf`, upload it through Unity's manual activation portal, and import the returned private `.ulf` with `-manualLicenseFile`.
- The alternative is an explicit project migration to Unity 2019.4.27 or newer and Unity Hub Personal activation. That changes the plan's pinned-engine constraint and requires user authorization plus migration/parity verification; it cannot be assumed by the executor.
- Dependency reread confirms no independent implementation lane remains: Todo 3 requires verified Todo 2, Todos 4-8 are transitively blocked, Todo 10 requires Todo 5, Todo 9 requires Todos 2/4/6/8/10, and F1-F4 require the completed staged build and smoke evidence.

## 2026-08-03 - Todo 2 Unity 2019.4 server resume boundary

- User authorized the Unity 2019.4 LTS migration. A sibling worktree at `/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4` contains the exact Todo 2 overlay and staged `ProjectVersion.txt`; canonical project/player manifests remained unchanged.
- Exact Editor `2019.4.40f1 (ffc62b691db5)` is installed under `$HOME/.local/share/novphy-unity/2019.4.40f1-ffc62b691db5/editor/Editor/Unity`. Private GConf libraries under `/tmp/opencode/unity-2019.3.4f1-libs/root` resolve all direct ELF dependencies.
- The user reports completing Unity Hub/Personal activation, but the batch probe has not validated it. The Licensing Client first crashed on missing ICU. `DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1` removed that crash, after which it failed with `No usable version of libssl was found`, timed out its IPC channel, and fell back to legacy activation. Ubuntu 24.04 exposes OpenSSL 3 only; the bundled client requires an OpenSSL 1.1 compatibility library.
- Next session must privately extract official Ubuntu Focal `libssl1.1_1.1.1f-1ubuntu2.24_amd64.deb`, add the extracted library directory to the scoped `LD_LIBRARY_PATH`, and rerun the Licensing Client with globalization invariant mode. Do not install Focal packages or downgrade OpenSSL system-wide.
- `-logFile -` emits nonfatal `CreateDirectory '' failed` on this Editor. Use a concrete absolute log path for evidence.

## 2026-08-03T11:33:19Z - Todo 2 evidence caveat

- The three retained sanitized probe logs show an assignment record after entitlement update but do not identify the assignment as Personal. Later prose must use `assignment record observed` unless a future sanitized artifact proves the license type.
- The captured Hub boundary records the terms screen with `action_taken=false`; later user-reported activation has no retained GUI proof. This is a user-action boundary, not proof that Personal activation completed.

## 2026-08-03T22:56:16Z - Todo 2 independent blocked-state verification

- Atlas independently verified the replacement Editor evidence after the single authorized `2019.4.41f2` probe: all 24 checksum-index entries passed, evidence JSON parsed, the official archive passed `xz --test`, and the recorded archive/editor SHA-256 values matched the retained files.
- Fresh canonical-project and production-player manifests remained byte-for-byte unchanged, and zero Unity processes from the Todo 2 execution were retained.
- The explicit activation failure occurred before project import, so absence of `task-2-unity.xml` and `task-2-legacy-parity.json` is the expected fail-closed result, not missing completion evidence.
- Boulder is paused with active `todo:2` blocked. A future probe is invalid until a materially changed supported Unity Hub activation state exists and the user explicitly authorizes it.

## 2026-08-04T00:08:30Z - Todo 2 second refreshed-state probe

- The bounded probe used exact Unity `2019.4.41f2`, no project path, globalization invariant mode, and only the private OpenSSL/GConf library paths. It exited `1` after successful IPC and entitlement refresh with the explicit activation error.
- Sanitized evidence is `task-2-license-probe-2019.4.41f2-refresh-2.safe.log`, SHA-256 `d133a2e4376c3ad72d9e168d35b786bb5490dae125a7ceeaba41597363f7df08`; the raw log was removed.
- Fresh canonical-project and production-player manifests remained byte-for-byte unchanged. No generated staged-project state or Unity Editor process remains.

Final verification wave, disclosed non-blocking items

- `tests.test_collect_rollouts.CollectRolloutsTest.test_known_dataset_artifacts_classify_gameplay_and_reported_menu_shots` requires `data/novphy_rollouts_dataset`, absent from both trees. Present at the plan scope base commit, so it predates this plan.
- `git diff --check` reports 42 trailing-whitespace hits over the plan range, all in Unity-generated `.cs.meta` boilerplate, zero in hand-written source.
- Quality debt Q-1: `PhysicsShotRecorder.cs` 678 LOC, `PhysicsCaptureProtocol.cs` 440, `PhysicsShotRecorderTests.cs` 491, `PhysicsCaptureProtocolTests.cs` 492, `scripts/physics_rollout_persistence.py` 434, all over the 250 LOC guidance.
- Quality debt Q-2: 13 reflection call sites in `PhysicsCaptureProtocolTests.cs` reach private members.
- Ledger line 24 is malformed JSON. It belongs to `world-model-data-pipeline.md`, predates this work, and was left untouched.
