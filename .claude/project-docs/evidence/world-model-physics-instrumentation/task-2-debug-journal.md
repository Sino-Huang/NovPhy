# Debug Journal - Unity 2019.4 Licensing and EditMode Gate

Started: 2026-08-03T19:57:42+10:00
Goal: Complete Todo 2 by proving supported Hub activation, importing only the isolated stage, and running all required Unity/parity/original-integrity gates.

## Environment Snapshot

- Runtime: Unity Editor 2019.4.40f1 (ffc62b691db5), native Linux editor with bundled Licensing Client.
- Entry: `/home/sukai/.local/share/novphy-unity/2019.4.40f1-ffc62b691db5/editor/Editor/Unity`.
- Project: `/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/tasks/task_template_designer` only.
- Git HEAD: `d72d1af1eb7b9a3cd60b0acb5364689e16735ea5`; branch `physics-unity-2019.4`; existing Todo 2 overlay is dirty and preserved.
- Editor SHA-256: `1fdc5220ec0cc3e7d2832412eb2ed39bb0ad9ea0a712fa4619b6f82045865918`.
- Archive SHA-256: `c592296df9dd888e5239ad7dda276bb718b33075c679a2fec9c080764644435f`.
- C# LSP: unavailable because the configured `csharp` server is not installed; Unity compilation and EditMode XML are authoritative.
- References read: durable resume note, selected plan Todo 2, canonical/worktree notepads, debugging `00-setup.md` and `02-investigate.md`.

## Hypotheses

1. [CONFIRMED RESOLVED] The bundled Licensing Client needed OpenSSL 1.1 in its child library path. The private Focal libraries removed the crash and allowed IPC connection.
2. [BLOCKED AT USER ACTION] The client reaches IPC and exposes an assignment record, but every probe still emits an activation error. The retained sanitized evidence does not prove the assignment type, and the captured Hub terms boundary remains unaccepted.
3. [REFUTED] Another runtime incompatibility or stale process prevents Licensing Client startup; the client launches and stays connected under the private compatibility environment.

## Artifacts To Clean

- RETAINED: `/tmp/opencode/unity-2019.4-libssl1.1/` - intentionally retained as the checksum-verified private compatibility resource for a reproducible rerun; no host install or linker change.
- [x] `/tmp/opencode/unity-2019.4-license-probe.log` - removed after sanitized evidence was persisted.
- [x] `/tmp/opencode/unity-2019.4-license-probe-2.log` - removed after sanitized evidence was persisted.
- [x] `/tmp/opencode/unity-2019.4-license-probe-force-free.log` - removed after sanitized evidence was persisted.
- [x] Unity-generated `Library/`, `Temp/`, `Obj/`, `Logs/`, `UserSettings/`, test-result, and crash artifacts under the staged project - none were created because import never started.
- [x] Any Todo 2 Unity or Licensing Client process started in this session - zero remain.

## Findings

### 2026-08-03T19:57:42+10:00 - Preflight

- Source: exact editor/archive `sha256sum`.
- Value: editor and archive digests equal their recorded baselines.
- Interpretation: the intended exact editor is still present and unmodified.
- Confirms/refutes: build-vs-runtime mismatch is not currently indicated.

- Source: Launchpad `openssl 1.1.1f-1ubuntu2.24` source page and official binary URLs.
- Value: source version is published for Focal security/updates; `libssl1.1_1.1.1f-1ubuntu2.24_amd64.deb` resolves from both `launchpad.net/ubuntu/+archive/primary/+files/` and `security.ubuntu.com/ubuntu/pool/main/o/openssl/`.
- Interpretation: the exact authorized package is available from Canonical-operated infrastructure.
- Confirms/refutes: package provenance is established before download.

### 2026-08-03T20:05:18+10:00 - Private OpenSSL 1.1 compatibility

- Source: Launchpad binary publication, HTTP headers, `sha256sum`, `dpkg-deb --info`, and extracted ELF files.
- Value: `1323248` bytes; package `libssl1.1`, version `1.1.1f-1ubuntu2.24`, architecture `amd64`; SHA-256 `7cf39d70a639017d1dd7c8d36daa2258063608688e449fddf40ffdd46f992a78` equals the published changes-file digest.
- Interpretation: the exact official Focal package is verified and extracted privately with no host installation or linker change.
- Confirms/refutes: the required OpenSSL 1.1 compatibility files are available to test Hypothesis 1.

### 2026-08-03T20:15:45+10:00 - First compatible license probe

- Source: bounded exact-editor probe with child-only OpenSSL 1.1/GConf paths and globalization invariant mode.
- Value: Licensing Client launched, connected over IPC, initiated entitlement licensing, updated licenses, and exposed an assignment record; the retained sanitized evidence does not prove its type, and the editor then emitted `Failed to activate/update license Missing or bad username or password`.
- Interpretation: OpenSSL was the Licensing Client crash root cause, but this process does not satisfy acceptance because the legacy activation error is explicit.
- Confirms/refutes: Hypothesis 1 dependency fix confirmed; Hypothesis 2 remains open pending a clean subsequent process.
- Privacy: the raw identifier-bearing line was redacted immediately; only `task-2-license-probe.safe.log` remains.

### 2026-08-03T20:23:55+10:00 - Identical second-process probe

- Source: a second bounded probe with the exact first-probe command and environment.
- Value: the same IPC/entitlement/assignment-record signals were followed by the same legacy activation error.
- Interpretation: activation did not become valid merely by starting a subsequent process.
- Confirms/refutes: the persisted-activation variant of Hypothesis 2 is refuted.

### 2026-08-03T20:24:34+10:00 - Final probe decision

- Source: official Unity 2019.4 Editor command-line documentation and exact-editor binary strings.
- Value: Unity 2019.4 documents and implements `-force-free` to run the Editor using the free license path.
- Interpretation: `-force-free` does not prove Hub terms acceptance or Personal activation; it was only a distinct supported probe mode and did not remove the activation error.
- Confirms/refutes: this is the third materially distinct and final licensing approach.

### 2026-08-03T20:28:09+10:00 - Official `-force-free` probe

- Source: bounded exact-editor probe with the documented `-force-free` argument and the same private compatibility environment.
- Value: Licensing Client again launched, connected over IPC, initiated entitlement licensing, updated licenses, and exposed an assignment record; the retained sanitized evidence does not prove its type, and the editor again emitted `Failed to activate/update license Missing or bad username or password`.
- Interpretation: selecting Unity's explicit free/Personal mode does not remove the legacy activation error, so Todo 2's license acceptance gate is not satisfied.
- Confirms/refutes: Hypothesis 2 is confirmed at the external licensing boundary; Hypothesis 3 is refuted because the client launches and stays connected rather than crashing or timing out.
- Privacy: the raw identifier-bearing line was redacted immediately; only `task-2-license-probe-force-free.safe.log` remains.

### 2026-08-03T20:37:42+10:00 - Blocked return and integrity verification

- Source: three materially distinct probes, process checks, generated-directory scan, and fresh full-file manifests for both protected originals.
- Value: all probes have the same activation error; zero Unity/Licensing Client processes remain; no staged Unity-generated directories exist; the live canonical-project manifest SHA-256 is `a986acc9dc911009e7e04bea85ee531ba4b9f9d692afa915456d81bcf7c8bb56`; the live production-player manifest SHA-256 is `e92a83fb14638b24ba8a0a05d588829823000f3db349ae9eaf39f188f68e8cd5`; both equal their baselines byte-for-byte.
- Interpretation: no further equivalent executor-side probe is justified. The supported user action is Hub Settings -> Licenses -> Refresh, terms acceptance, and confirmation of visible active Unity Personal. Import, EditMode, parity, and completion-claim gates remain unrun rather than being inferred from exit codes or source inspection.
- Cleanup: all raw probe logs are absent; three sanitized logs remain as evidence. The verified private compatibility prefix is retained for a reproducible rerun and has not changed the host linker or package database.

### 2026-08-03 - Remote-desktop Hub refresh continuation

- External change: the user opened Unity Hub directly in the remote-desktop session and reports refreshing the Personal license. This authorizes exactly one new bounded non-project probe; it is not accepted as activation proof by itself.
- Process boundary: a user-owned desktop Hub process tree and one Licensing Client were present before the probe. Preserve them; clean up only processes created by the probe command.
- Hypothesis A: refreshed Hub state is visible to the exact Editor. Distinguishing evidence: live IPC/assignment signals with no activation, timeout, or client error.
- Hypothesis B: refresh did not produce batch-visible Editor activation. Distinguishing evidence: live IPC/assignment signals followed by an explicit activation error.
- Hypothesis C: the Editor cannot use the live desktop Licensing Client. Distinguishing evidence: IPC timeout, client launch/crash, or fallback distinct from the prior activation error.
- PLANNED RAW ARTIFACT: `/tmp/opencode/unity-2019.4-license-probe-remote-desktop.raw.log`; remove after sanitizing.
- PLANNED RETAINED ARTIFACT: `task-2-license-probe-remote-desktop.safe.log`; redact assignment payloads and any private identifiers before persistence.
- Stop condition: any activation error or Licensing Client failure prohibits project import and any second equivalent probe.

### 2026-08-03T15:08:51Z - One authorized post-refresh probe consumed

- Launch determination: the initial shell wrapper failed before Unity launch because zsh reserves `status` as read-only; no raw log or Unity process existed from that wrapper. The corrected wrapper launched the one authorized Editor probe and produced the retained safe log at local mtime `2026-08-04 01:06:17.066953980 +1000`.
- Command: `timeout --signal=TERM --kill-after=20s 120s env DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1 LD_LIBRARY_PATH=<private OpenSSL/GConf paths plus existing value> <exact Unity 2019.4.40f1> -batchmode -nographics -quit -logFile /tmp/opencode/unity-2019.4-license-probe-remote-desktop.raw.log`.
- Value: exit `1`; exact Editor `2019.4.40f1 (ffc62b691db5)`; batch mode; successful Licensing Client IPC; entitlement licensing initiated; licenses updated; then `Failed to activate/update license Missing or bad username or password`.
- Interpretation: the remote-desktop Hub refresh did not produce batch-visible Editor activation. Hypothesis B is confirmed; project import and every downstream Todo 2 runtime gate remain prohibited.
- Probe count: exactly one post-refresh Unity process launch. No duplicate is authorized or run.
- Privacy/cleanup: the raw log was allowlist-sanitized and removed. Retained `task-2-license-probe-remote-desktop.safe.log` SHA-256 is `75c6b9b25fad940b1ee1145ea0c9275fdc6ef8410f651c88085488f18d600d44`; it contains no assignment payload. Zero Unity Editor processes remain; the user's pre-existing desktop Hub and Licensing Client remain untouched.
- PLANNED TEMP VERIFICATION: `/tmp/opencode/task-2-canonical-project-post-refresh.tsv` and `/tmp/opencode/task-2-production-player-post-refresh.tsv`; generate from live protected roots, compare to existing before manifests, then remove.

### 2026-08-03T15:12:00Z - Post-probe cleanup and protected-root verification

- Fresh live canonical-project manifest matched `task-2-canonical-project-before.tsv`, SHA-256 `a986acc9dc911009e7e04bea85ee531ba4b9f9d692afa915456d81bcf7c8bb56`; fresh live production-player manifest matched its baseline, SHA-256 `e92a83fb14638b24ba8a0a05d588829823000f3db349ae9eaf39f188f68e8cd5`.
- Temporary manifests were removed after comparison. No Unity Editor process remains; the pre-existing desktop Hub and Licensing Client remain preserved. No staged `Library`, `Temp`, `Obj`, `Logs`, `UserSettings`, or test-result directory exists.
- No `task-2-unity.xml` or `task-2-legacy-parity.json` exists because the activation gate failed before project import. Todo 2 is objectively incomplete.

### 2026-08-04 - Authorized 2019.4.41f2 replacement preflight

- Authorized target: exact Unity `2019.4.41f2 (6b23d448b533)` from `https://download.unity3d.com/download_unity/6b23d448b533/LinuxEditorInstaller/Unity-2019.4.41f2.tar.xz`.
- Replacement root: `/home/sukai/.local/share/novphy-unity/2019.4.41f2-6b23d448b533`; it did not exist at preflight. The historical `2019.4.40f1` root and all four sanitized failed-probe logs remain unchanged.
- Capacity: `/home/sukai/.local/share` has approximately `1.3T` available and `/mnt/array` has approximately `14T` available.
- Process boundary: no Unity Editor process is running. The pre-existing desktop Licensing Client PID `1423579` is user-owned and must remain untouched.
- Worktree boundary: branch `physics-unity-2019.4` remains at `d72d1af1eb7b9a3cd60b0acb5364689e16735ea5`; the dirty Todo 2 product/evidence overlay is preserved.
- PLANNED DOWNLOAD ARTIFACT: `/home/sukai/.local/share/novphy-unity/2019.4.41f2-6b23d448b533/Unity-2019.4.41f2.tar.xz.part`; resumable only, promoted after content-length, SHA-256, and `xz --test` checks.
- PLANNED ARCHIVE ARTIFACT: `/home/sukai/.local/share/novphy-unity/2019.4.41f2-6b23d448b533/Unity-2019.4.41f2.tar.xz`; retain as verified official archive.
- PLANNED EXTRACTION ARTIFACT: `/home/sukai/.local/share/novphy-unity/2019.4.41f2-6b23d448b533/editor`; private extraction only, with no host package or linker changes.
- PLANNED IDENTITY LOG: `/home/sukai/.local/state/novphy-unity-migration/physics-unity-2019.4/editor-2019.4.41f2-version.stdout-stderr.log`; retain after bounded exact-identity verification.
- PLANNED RAW PROBE: `/tmp/opencode/unity-2019.4.41f2-license-probe.raw.log`; run exactly once after editor identity verification and remove immediately after sanitization.
- PLANNED SAFE PROBE: `task-2-license-probe-2019.4.41f2.safe.log`; retain only allowlisted non-private licensing signals.
- Stop condition: any archive/identity mismatch, activation failure, licensing-client crash/fallback, or unresolved entitlement stops execution before staged-project import.

### 2026-08-03T22:20:13Z - Replacement archive and Editor identity verified

- HTTP metadata: status `200`; content length `1913850596`; ETag `524b6ddeb47049191905c7ee265b3c09`; Last-Modified `Tue, 14 Oct 2025 14:27:07 GMT`; byte ranges supported.
- Archive: resumable `.part` download reached exactly `1913850596` bytes; SHA-256 `5426954a630036f59d6d1b40e548bc4d0a47a092475a7530db8013ff5c4aea63`; full `xz --test` passed before promotion.
- Extraction: promoted archive was privately extracted under `/home/sukai/.local/share/novphy-unity/2019.4.41f2-6b23d448b533/editor`; historical `2019.4.40f1` files were not changed.
- Executable: `/home/sukai/.local/share/novphy-unity/2019.4.41f2-6b23d448b533/editor/Editor/Unity`; SHA-256 `32252cb8eca087743e500596e093061a906203703915c2d3c2fb2f8a372bc150`.
- Bounded identity run: exit `0`, exact output `2019.4.41f2 (6b23d448b533)`; retained log SHA-256 `f744e09da0620a2d479e472037a33e063f09d2a29063305a6abf258792ab7801`.
- Process check: no Unity Editor remains; pre-existing desktop Licensing Client PID `1423579` remains untouched.
- Gate decision: archive integrity and exact Editor identity passed. Exactly one bounded non-project `2019.4.41f2` licensing probe is now permitted.

### 2026-08-03T22:30:42Z - Single 2019.4.41f2 probe consumed and blocked

- Command: one bounded non-project launch of exact Unity `2019.4.41f2` with `DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1`, the private OpenSSL/GConf `LD_LIBRARY_PATH`, `-batchmode -nographics -quit`, and a concrete raw log path.
- Value: exit `1`; exact Editor identity; successful Licensing Client IPC; entitlement-based licensing initiated; licenses updated; then `Failed to activate/update license Missing or bad username or password`.
- Decision: activation acceptance failed. No project path was supplied, no staged project import occurred, and no second replacement-version probe is authorized or run.
- Privacy: the `1857`-byte raw log was allowlist-sanitized and removed. Retained `task-2-license-probe-2019.4.41f2.safe.log` SHA-256 is `fc9afc03b36b6a9805eb4518e6d60bcf8b81cc945a851fc863ed066435b98d0e`.
- Stage target: `ProjectSettings/ProjectVersion.txt` now declares exact `2019.4.41f2 (6b23d448b533)` for a future authorized first import; changing this text did not launch Unity or create generated state.
- Protected roots: fresh canonical-project and production-player manifests compare byte-for-byte equal to baseline, retaining SHA-256 values `a986acc9dc911009e7e04bea85ee531ba4b9f9d692afa915456d81bcf7c8bb56` and `e92a83fb14638b24ba8a0a05d588829823000f3db349ae9eaf39f188f68e8cd5`.
- Cleanup: no raw probe log, partial download, Unity Editor process, or staged `Library`, `Temp`, `Obj`, `Logs`, `UserSettings`, or `TestResults` directory remains. The pre-existing desktop Licensing Client PID `1423579` remains untouched.
- Blocked gates: first import, focused/all EditMode tests, legacy parity, Todo 2 completion, and the intended commit remain unrun.

### 2026-08-03T23:57:59Z - Second materially changed Hub-state continuation

- External change: the user reports completing the supported Unity Hub Settings -> Licenses -> Refresh action and explicitly authorized resumption. This is a distinct activation state and permits exactly one new bounded non-project probe.
- Boulder boundary: active work `world-model-physics-instrumentation-622663e5` is active and its preserved Todo 2 executor session `opencode:ses_038fe0d30ffespRr40cAhphkBE` is running.
- Preflight: exact archive SHA-256 `5426954a630036f59d6d1b40e548bc4d0a47a092475a7530db8013ff5c4aea63`; exact executable SHA-256 `32252cb8eca087743e500596e093061a906203703915c2d3c2fb2f8a372bc150`; archive `xz --test` passed; staged `ProjectVersion.txt` declares `2019.4.41f2 (6b23d448b533)`.
- Process boundary: no Unity Editor process is running. The pre-existing desktop Licensing Client PID `1914909` is user-owned and must remain untouched.
- Generated-state boundary: staged `Library`, `Temp`, `Obj`, `Logs`, `UserSettings`, and `TestResults` are absent before the probe.
- Hypothesis A: refreshed Hub state is accepted by the exact Editor. Distinguishing evidence: Licensing Client IPC succeeds and the log has no activation, fallback, crash, timeout, or unresolved-entitlement error.
- Hypothesis B: refreshed Hub state remains unavailable to batch Editor activation. Distinguishing evidence: live IPC/entitlement update followed by an explicit activation error.
- Hypothesis C: the exact Editor cannot use the live desktop Licensing Client. Distinguishing evidence: IPC timeout, client crash, or fallback to legacy activation.
- PLANNED RAW ARTIFACT: `/tmp/opencode/unity-2019.4.41f2-license-probe-refresh-2.raw.log`; remove immediately after allowlist sanitization.
- PLANNED RETAINED ARTIFACT: `task-2-license-probe-2019.4.41f2-refresh-2.safe.log`; retain only non-private identity, IPC, entitlement, and error-state signals.
- Stop condition: any activation error, fallback, crash, timeout, or unresolved entitlement prohibits project import and a second equivalent probe.

### 2026-08-04T00:02:57Z - Refreshed-state probe consumed and failed closed

- Command: `timeout --signal=TERM --kill-after=20s 120s env DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1 LD_LIBRARY_PATH=<exact private OpenSSL/GConf paths> <exact Unity 2019.4.41f2> -batchmode -nographics -quit -logFile /tmp/opencode/unity-2019.4.41f2-license-probe-refresh-2.raw.log`.
- Value: exit `1`; exact Editor `2019.4.41f2 (6b23d448b533)`; successful Licensing Client IPC; entitlement-based licensing initiated; licenses updated; then `Failed to activate/update license Missing or bad username or password`.
- Interpretation: Hypothesis B is confirmed. The materially changed Hub refresh still did not produce batch-visible activation; Hypotheses A and C are refuted by the explicit error after successful IPC/update.
- Probe count: exactly one Editor launch in this refreshed state. No second equivalent probe is authorized or run.
- Decision: project import, EditMode tests, runtime legacy parity, Todo 2 completion, and commit remain prohibited.
- Privacy: the `1857`-byte raw log was allowlist-sanitized to `task-2-license-probe-2019.4.41f2-refresh-2.safe.log`; no assignment payload, license contents, or account data were retained.
- SAFE LOG SHA-256: `d133a2e4376c3ad72d9e168d35b786bb5490dae125a7ceeaba41597363f7df08`; raw log removed after hashing the retained artifact.
- PLANNED TEMP VERIFICATION: `/tmp/opencode/task-2-canonical-project-refresh-2.tsv` and `/tmp/opencode/task-2-production-player-refresh-2.tsv`; generate from protected roots, compare byte-for-byte to baseline manifests, then remove.

### 2026-08-04T00:08:30Z - Second-refresh blocked return verified

- Fresh canonical-project manifest matched baseline SHA-256 `a986acc9dc911009e7e04bea85ee531ba4b9f9d692afa915456d81bcf7c8bb56`; fresh production-player manifest matched baseline SHA-256 `e92a83fb14638b24ba8a0a05d588829823000f3db349ae9eaf39f188f68e8cd5`.
- Temporary manifests and the raw probe log were removed. No staged `Library`, `Temp`, `Obj`, `Logs`, `UserSettings`, or `TestResults` directory exists.
- `task-2-unity.xml` and `task-2-legacy-parity.json` remain absent because the activation gate failed before import. This is the required fail-closed result.
- Zero Unity Editor processes remain; the user's pre-existing desktop Licensing Client remains untouched.
