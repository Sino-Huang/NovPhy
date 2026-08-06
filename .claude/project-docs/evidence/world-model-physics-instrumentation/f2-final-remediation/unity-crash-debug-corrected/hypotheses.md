# Corrected Unity 2019.4.41f2 Crash Audit

## Runtime and scope

- Editor: `/home/sukai/.local/share/novphy-unity/2019.4.41f2-6b23d448b533/editor/Editor/Unity`
- Expected revision: `2019.4.41f2 (6b23d448b533)`
- Compatibility environment: `LD_LIBRARY_PATH=/tmp/opencode/unity-2019.4-libssl1.1/root/usr/lib/x86_64-linux-gnu:/tmp/opencode/unity-2019.3.4f1-libs/root/usr/lib/x86_64-linux-gnu:/tmp/opencode/unity-2019.3.4f1-libs/root/usr/lib`
- Probe scope: exact Editor, no migration project for probes 1-2; fresh temporary empty project for probe 3.

## Hypotheses

1. **H1: Unity 2019.4.41f2's embedded CEF shutdown has an external native crash independent of project content.** Distinguishing evidence: probes 1 and 2 reproduce SIGSEGV with the same CEF shutdown frames and no project path; fix would require a different Editor/runtime. 
2. **H2: GPU/backend initialization triggers the crash.** Distinguishing evidence: probe 2 (`-disable-gpu -force-gfx-st`) exits without SIGSEGV or has a materially different loader/CEF result while probe 1 crashes; fix would be a graphics launch configuration.
3. **H3: Project/import state triggers the crash.** Distinguishing evidence: probes 1-2 are clean but probe 3, a fresh empty project outside the migration worktree, reproduces; fix would be project isolation or import repair.
4. **H4: The prior crash was caused by an incomplete compatibility-library environment.** Distinguishing evidence: corrected probes fail at libgconf/loader startup rather than reaching CEF, or differ from prior CEF traces; fix would be restoring the pinned private library roots.

The audit must not claim EditMode success. If all corrected exact-library probes reach and reproduce CEF SIGSEGV, the classification is an external Editor/runtime blocker and no product fix is made.
