# Problems — world-model-physics-instrumentation

Unresolved blockers and technical debt discovered during work on this plan.

_Auto-scaffolded by /start-work. Append new entries below - never overwrite._

---

## 2026-07-31T22:13:36+10:00 - Todo 1

- No unresolved Todo 1 contract blocker. Unity production, transport, collector, canonical episode acceptance, and data-loader integration remain intentionally deferred to Todos 2-10.

## 2026-08-01T00:17:04+10:00 - Todo 2

- BLOCKED: Unity 2019.3.4f1 is unavailable in this environment. Task 2 code and static review can be completed, but acceptance cannot be claimed and commits must not be created until the required EditMode suite exits zero and its XML is inspected.
- The 2026-08-01 resumed search also found no locally cached Unity container image, so offline container execution cannot unblock the acceptance suite.
- The official editor is now preserved under `/tmp/opencode`, but Ubuntu 24.04 lacks the legacy `libgconf-2.so.4` runtime dependency and has no configured `libgconf-2-4` package candidate. Todo 2 remains blocked before licensing and compilation; system package installation is outside the provisioning boundary.
- RESOLVED: private extraction of the official Ubuntu Focal GConf packages provides complete dynamic-linker closure without changing the host.
- BLOCKED: Unity now reaches licensing, but no valid existing license permits batch execution. Todo 2 compilation, EditMode XML, runtime legacy parity, and commits remain blocked; no activation was attempted.
- Static compiler closure now passes for both production and Todo 2 tests using Unity's bundled Roslyn, but this does not resolve the licensing/runtime acceptance blocker and must not be treated as Unity Editor compilation or test execution.

## 2026-08-03T11:33:19Z - Todo 2 activation boundary

- BLOCKED: private OpenSSL 1.1 compatibility is complete, and three bounded Unity 2019.4 probes reached Licensing Client IPC, entitlement update, and an assignment record before the explicit activation error `Missing or bad username or password`.
- USER ACTION REQUIRED: sign in to Unity Hub, open Settings -> Licenses, choose Refresh, accept the displayed Editor Software Terms, and confirm visible active Unity Personal. Todo 2 remains unchecked; EditMode XML and legacy parity are absent.

## 2026-08-03T15:08:51Z - Todo 2 post-refresh probe

- BLOCKED: the one authorized non-project Unity 2019.4.40f1 probe after the remote-desktop Hub refresh exited `1`. It connected to Licensing Client IPC and updated entitlements, then emitted `Failed to activate/update license Missing or bad username or password`.
- No project path was supplied, no project was imported, no equivalent probe was repeated, and Todo 2 remains unchecked without EditMode XML or legacy parity.

## 2026-08-03 - Todo 2 documented version incompatibility

- ROOT BLOCKER: Unity Support states that Personal activation starts at `2019.4.41f2`. The pinned `2019.4.40f1` Editor cannot be activated with Personal under the current licensing system; earlier versions require a serial-based paid entitlement.
- USER DECISION REQUIRED: authorize migration to exact `2019.4.41f2` or later. Another `2019.4.40f1` Personal probe is not permitted.

## 2026-08-03T22:56:16Z - Todo 2 replacement-version blocker verified

- BLOCKED: exact Unity `2019.4.41f2 (6b23d448b533)` connected to Licensing Client IPC and refreshed entitlements, then failed with `Missing or bad username or password` before project import.
- Independent verification passed all 24 checksum-index entries, JSON parsing, archive `xz --test`, exact archive/editor hashes, unchanged canonical-project and production-player manifests, and zero retained Unity processes from the Todo 2 execution.
- `task-2-unity.xml` and `task-2-legacy-parity.json` remain absent by design because the activation gate prohibited import and runtime verification. Todo 2 and all downstream tasks remain blocked.
- NEXT ACTION: another probe requires a materially changed, supported Unity Hub activation state and explicit authorization; do not repeat an equivalent batch probe in the current state.

## 2026-08-04T00:08:30Z - Todo 2 second supported Hub refresh

- BLOCKED: the exactly one authorized non-project `2019.4.41f2` probe connected to Licensing Client IPC and refreshed entitlements, then failed with `Missing or bad username or password` before project import.
- No second equivalent probe was run. `task-2-unity.xml` and `task-2-legacy-parity.json` remain absent by design, so Todo 2 and all downstream tasks remain blocked.
- NEXT ACTION: any future probe requires another materially changed supported activation state and explicit authorization.

Final verification wave

- No unresolved blocker. F1-F4 all recommend APPROVE with zero blockers at commit `e2d19ae`, tree `f0347223`, archive `429cac1d`.
- Two findings were raised and fixed during the wave rather than carried: P0-1 (Unity licensing identity in the published build log) and P3-1 (an EditMode test that had never been captured passing).
- Remaining items are disclosed non-blocking debt, recorded in issues.md.
- Nothing is promoted beyond the staged player. The plan requires explicit user approval before completion is declared.
