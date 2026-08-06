# Todo 2 exact Unity licensing probe debug journal

Started: 2026-08-05 Australia/Melbourne

Goal: Execute exactly one bounded no-project Unity 2019.4.41f2 licensing probe after the user-authorized Hub Personal activation transition, decide from sanitized runtime evidence, and stop before project import.

## Environment snapshot

- Runtime: exact retained native Unity Editor `2019.4.41f2 (6b23d448b533)`.
- Entry: `/home/sukai/.local/share/novphy-unity/2019.4.41f2-6b23d448b533/editor/Editor/Unity -batchmode -nographics -quit -logFile <absolute-private-raw-log>`.
- Project path: intentionally absent.
- Compatibility: `DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1`; private OpenSSL/GConf paths prepended through `LD_LIBRARY_PATH` only.
- References read: debugging `SKILL.md`, `references/runtimes/native-binary.md`, `references/methodology/00-setup.md`, `02-investigate.md`, and `09-cleanup.md`.

## Hypotheses

1. H1: Hub Personal activation lets the exact Editor initialize licensing. Distinguishing evidence: Licensing Client remains live through IPC, expected Personal/assignment state appears, and no activation error occurs. Result classification: `PROBE PASS`.
2. H2: The old Editor reaches IPC but still emits the prior credential activation error. Distinguishing evidence: live IPC/entitlement lines followed by `Missing or bad username or password` or another activation error. Result classification: `BLOCKED`.
3. H3: Compatibility/runtime startup fails before decisive licensing state. Distinguishing evidence: Licensing Client crash, timeout, ICU/OpenSSL failure, legacy fallback, or no live IPC. Result classification: `BLOCKED`.

## Artifacts and cleanup

- Private directory `/tmp/opencode/todo2-license-probe-20260805`; contains raw log, exit receipt, and pre/post manifests. Cleanup: remove the entire directory after sanitization and comparisons.
- Raw log `/tmp/opencode/todo2-license-probe-20260805/unity-license-probe.raw.log`. Cleanup: delete with the private directory; never retain or quote unsanitized contents.
- Sanitized retained log `.omo/evidence/world-model-physics-instrumentation/task-2-license-probe-2026-08-05.safe.log`. Retained evidence; scrub credentials, account IDs, serials, tokens, license material, seat identifiers, and request IDs.
- Probe result JSON `.omo/evidence/world-model-physics-instrumentation/task-2-license-probe-2026-08-05.json`. Retained evidence.
- Main-worktree summary `.omo/evidence/world-model-physics-instrumentation/task-2-license-probe-summary-2026-08-05.json`. Retained evidence.
- Any exact Editor or child licensing process surviving the bounded command must be terminated and absence verified.
- Unexpected probe-created repository-root directories `/mnt/array/sukaih/Project/NovPhy/{Assets,Packages,ProjectSettings,Library,Logs}`. Observed timestamps begin at `2026-08-05 00:59`; Git tracking check required before cleanup. Cleanup: remove exactly these five directories after confirming they are untracked/ignored probe residue.

## Scope invariants

- Exactly one Editor invocation; never retry.
- No `-projectPath`; no project import/open.
- Canonical project and production player pre/post manifests must compare byte-for-byte.
- Migration project generated directories (`Library`, `Temp`, `Obj`, `Logs`, `UserSettings`) must retain their pre-probe existence/content state.
- No Hub storage or GUI changes in this probe.

## Findings

- Single invocation exited `0`.
- Licensing runtime connected to LicensingClient IPC, initiated entitlement-based licensing, refreshed licenses, reported an assigned non-Pro license, and reported the current license valid and activated.
- No credential activation error, activation failure, ICU failure, OpenSSL failure, legacy fallback, or timeout appeared.
- H1 confirmed; H2 and H3 refuted.
- The old Editor created transient repository-root Unity project state despite omission of `-projectPath`; exact residue is journaled for cleanup. The canonical Unity project remains `tasks/task_template_designer` and was not opened.
- Probe raw log/private directory removed; repository-root transient Unity state removed; protected manifests match their retained baselines.
- Newly authorized isolated import used only the migration project with exact `2019.4.41f2`; import log contains no compiler error.
- Focused EditMode XML passed `5/5` twice; all EditMode XML passed `5/5`; legacy exact-byte SHA-256 repeated as `c323079aeb64e0928cd4dd93122d9f639d3c9e4012a16a9f9cadfac01ccd8e08`.
- Unity 2019.4 CEF raised SIGSEGV during Test Runner shutdown only after each complete XML was saved. This was classified by log ordering and not hidden behind exit-code success.
- Isolated generated `Library`, `Temp`, `Logs`, and ignored `Packages` were removed. No `Obj`, `UserSettings`, or `TestResults` remained.
