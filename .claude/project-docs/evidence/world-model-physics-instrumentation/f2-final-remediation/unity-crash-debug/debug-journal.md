## Hypotheses
- H1 CEF browser-loop incompatibility causes SIGSEGV in headless 2019.4; distinguish with `-nographics`, `-disable-gpu`, and software display variants on non-project probes.
- H2 project/import state triggers the crash; distinguish with the same Editor and libraries on a fresh empty project versus the migration project.
- H3 graphics/display/library environment causes the crash; distinguish with explicit `DISPLAY`/software-rendering and library-path variants while preserving exact Editor identity.

## Artifacts
- This journal records every bounded probe output under this directory.
- No product, test, plan, ledger, Boulder, published-stage, or protected-root edits are permitted.

## Probe artifacts (journaled before creation)
- [ ] /mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/.omo/evidence/world-model-physics-instrumentation/f2-final-remediation/unity-crash-debug/probe-1.log; /mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/.omo/evidence/world-model-physics-instrumentation/f2-final-remediation/unity-crash-debug/probe-2.log; /mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/.omo/evidence/world-model-physics-instrumentation/f2-final-remediation/unity-crash-debug/probe-3.log — remove after audit
- [x] /tmp/unity-crash-debug-empty-510501 — temporary empty project; removed after audit
- [x] Unity locks, cores, and caches under probe temp/runtime paths — none remained after audit

## Findings
### 2026-08-06T03:51:26Z — probe 1
- Source: `probe-1.log`
- Value: `error while loading shared libraries: libgconf-2.so.4: cannot open shared object file: No such file or directory`; numeric exit/signal fields were not appended because the zsh wrapper used reserved variable `status` after Unity returned.
- Interpretation: exact Editor did not start; no CEF or SIGSEGV evidence.
- Refutes/Confirms: Refutes a runtime-confirmed H1 SIGSEGV for this environment; does not distinguish H2/H3.

### 2026-08-06T03:52:12Z — probes 2 and 3
- Source: `probe-2.log`, `probe-3.log`
- Value: both report `error while loading shared libraries: libgconf-2.so.4: cannot open shared object file: No such file or directory`; both `exit_status=127`; both `signal=none`
- Interpretation: GPU flags and a fresh empty project cannot be evaluated because the loader fails before Unity initialization.
- Refutes/Confirms: Refutes a runtime-confirmed H1 SIGSEGV for this environment; H2/H3 remain untested.

## Final audit boundary
- Exactly three Unity launches were attempted.
- Probe 1 output is complete through the loader error, but its wrapper result metadata is unavailable; no fourth launch was made.
- The migration project was not passed to Unity and was not modified.
- No EditMode success is claimed.
