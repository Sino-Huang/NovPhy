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
