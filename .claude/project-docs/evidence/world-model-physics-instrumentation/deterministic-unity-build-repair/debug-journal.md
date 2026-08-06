# Debug Journal - Deterministic Unity Build Repair

Started: 2026-08-06
Goal: Make two exact Unity 2019.4.41f2 staged-player builds byte-identical without touching protected or shared product state.

## Environment snapshot

- Runtime: Unity 2019.4.41f2 (6b23d448b533), pinned Linux editor SHA-256 `32252cb8eca087743e500596e093061a906203703915c2d3c2fb2f8a372bc150`.
- Entry: `scripts/build_physics_player.sh` invokes `NovPhyBuild.BuildPhysicsLinux` with bounded external timeouts supplied by the caller.
- Git baseline: branch `physics-unity-2019.4`; tracked product source clean, unrelated `.omo` files dirty.
- Retained RED: commit `95e7b82919c7cc2cd1aa911c4ebf9ffcfba8f9fe` produced three different archive SHA-256 values and consecutive archive/provenance/receipt `cmp` exits of 1.
- References read: full `.omo/plans/world-model-physics-instrumentation.md`, `deterministic-build-repair-task.md`, final archive BlockedClaim and cleanup receipt, debugging phases 0-3/6-10, programming, git-master, and start-work skills.

## Hypotheses

1. [SUPPORTED BY RETAINED EVIDENCE] Unity compiler/build identity defaults regenerate managed PE checksum/MVID bytes and a player build identifier. Distinguishing evidence: deterministic Roslyn response option plus `BuildOptions.NoUniqueIdentifier` makes two clean exact-method builds compare byte-identically. If true, fix is: supported flags.
2. [REFUTED BY RETAINED EVIDENCE] Archive packaging metadata causes the variance. Distinguishing evidence: packager fixes tar uid/gid/names/mtime/order and gzip timestamp, while extracted managed DLLs and `globalgamemanagers` differ. If true, fix is: package normalization.
3. [REFUTED BY RETAINED EVIDENCE] Stale generated project state or retained payload substitution causes the variance. Distinguishing evidence: three fresh payload roots differ only in Unity-generated identities, with stable source and clean temp-root receipts. If true, fix is: clean imports.

## Artifacts to clean or preserve

- [ ] Preserve this evidence directory and final machine-readable DoneClaim; never stage `.omo`.
- [ ] `tests/test_deterministic_unity_build.py` - focused source contract; commit with the production fix.
- [ ] `tasks/task_template_designer/Assets/csc.rsp` and `.meta` - deterministic Roslyn project inputs; commit with the contract.
- [ ] `build-1-stage` and `build-2-stage` - post-commit evidence stages; remove after hashes, extracted provenance, and receipts are preserved.
- [ ] `protected-*-before.*` / `protected-*-after.*` - read-only protected-root manifests; preserve as evidence.
- [ ] `focused-red.*`, `focused-green-*.*`, build/verifier/comparison receipts - preserve as evidence.
- [ ] Remove owned build stage roots after hashes/comparisons are recorded.
- [ ] Remove owned Unity/player/Xvnc/Xvfb/Java processes, listeners, locks, and temporary unpack/build roots.
- [ ] Preserve canonical `/mnt/array/sukaih/Project/NovPhy/tasks/task_template_designer`.
- [ ] Preserve production `/mnt/array/sukaih/Project/NovPhy/sciencebirdsgames/Linux`.
- [ ] Preserve active `/mnt/array/sukaih/Project/NovPhy/data/novphy_rollouts_dataset_20260708_171531`.

## Findings

- Retained archive comparison localizes differences to four managed DLL PE checksum/MVID regions and `globalgamemanagers` build identity; package ordering and metadata are stable.
- Current `NovPhyBuild.cs` uses only `BuildOptions.StrictMode`; no `Assets/csc.rsp` exists and no focused build-option contract test exists.
- 2026-08-06 branch check: `GIT_MASTER=1 git branch --show-current` -> `physics-unity-2019.4` (exit 0).
- 2026-08-06 source check: `GIT_MASTER=1 git rev-parse HEAD` -> `95e7b82919c7cc2cd1aa911c4ebf9ffcfba8f9fe` (exit 0).
- 2026-08-06 retained behavioral RED: archive SHA-256 values `e8018cb56525b719a854793060638dbc4345cd2a0921c258bf44bd4f9a5627a6`, `61c39c22d4a670e1fde15762ae1441f483240f7c06c7e9edbe13a0a107f19130`, and `20341205a130f5f46378ed5fe69dcb3d093a175b6be6563ab0dbae60fea2c913`; consecutive archive comparisons both recorded `cmp=1` in `final-archive-rebuild/stop-hook-3-direct-verification.json`.
- 2026-08-06 tracked baseline: `GIT_MASTER=1 git status --short --untracked-files=no` listed only six pre-existing `.omo` plan/notepad/ledger modifications; `GIT_MASTER=1 git diff --name-only -- . ':(exclude).omo/**'` produced no output (both exit 0).
- 2026-08-06 pinned editor check: `sha256sum $HOME/.local/share/novphy-unity/2019.4.41f2-6b23d448b533/editor/Editor/Unity` -> `32252cb8eca087743e500596e093061a906203703915c2d3c2fb2f8a372bc150` (exit 0).
- 2026-08-06 process scan: `timeout 10s pgrep -af 'Unity|Xvfb|Xvnc|novphy_physics_build_|verify_physics_player|9001-player|game_playing_interface'` found only this assigned OpenCode process, the scan command, and the environment-owned Unity Licensing Client; no unfamiliar build/player process will be interrupted.
- 2026-08-06 protected baseline commands (all exit 0): canonical and production used `timeout --signal=TERM --kill-after=5s 180s bash -c 'rg --files -0 --hidden | LC_ALL=C sort -z | xargs -0 -r sha256sum'`; active dataset used bounded root-only `stat -c '%d:%i:%h:%s:%Y:%Z'` to avoid racing 6M+ active files.
- 2026-08-06 protected baseline artifacts: canonical 504 entries, SHA-256 `8b64bf84d75fb233ea02efdfdde2d75e65a5cf647a81d53d8189565979cea1a4`; production 29714 entries, SHA-256 `40d21d8e3f31b4f778357b44db61afd0c2ce966c90d912c40bc2131dc7038410`; active stat SHA-256 `692a76ac7fd71f902381dbf99725254432a1433f4d1d65cab9eb615895791261`.
- 2026-08-06 installed API check: `strings .../Editor/Data/Managed/UnityEditor.dll | rg '^(StrictMode|NoUniqueIdentifier)$'` -> both `StrictMode` and `NoUniqueIdentifier` (exit 0).
- 2026-08-06 compatibility sources: Unity 2019.4 documents `BuildOptions.NoUniqueIdentifier`, `BuildOptions.StrictMode`, combinable `BuildOptions`, and Roslyn response files under `Assets/csc.rsp`; Microsoft documents exact `-deterministic` csc spelling. Exact patch compatibility remains gated on a real Unity import/build.

### Focused RED (2026-08-06)

- Command: `timeout --signal=TERM --kill-after=5s 60s python -m unittest tests.test_deterministic_unity_build -v`
- Exit: `1`.
- Artifact: `focused-red.log`, SHA-256 `d3ae199229d526b7c884fa8f9a60b3023a896e684ee9c0ae866072917e1955b9`; exit artifact SHA-256 `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`.
- Verbatim failure observables: missing `Assets/csc.rsp` raises `FileNotFoundError`; parsed build options lack `BuildOptions.NoUniqueIdentifier` while retaining `BuildOptions.StrictMode`.
- Test parser correction before accepted RED: the first draft matched the local variable assignment; it was narrowed to the `BuildPlayerOptions` initializer and rerun. The preserved accepted RED above fails only for the two missing deterministic identity requirements.
- LSP result: Python diagnostics unavailable because the configured `basedpyright` server is not installed and installation was previously declined; runtime unittest and later `py_compile`/static checks are the fallback gates.

### GREEN and exact Unity import (2026-08-06)

- Production fix: `Assets/csc.rsp` contains exactly `-deterministic`; stable `DefaultImporter` metadata is present; `NovPhyBuild.cs` combines `BuildOptions.NoUniqueIdentifier | BuildOptions.StrictMode`.
- Focused commands: two independent bounded invocations of `timeout --signal=TERM --kill-after=5s 60s python -m unittest tests.test_deterministic_unity_build -v`; both exit `0`, run two tests, and produce identical log SHA-256 `78dfb02cb1358c3bde235d30e5f193fc5f705eea3028298f570efde4e0b0fc88`.
- First direct Unity attempt omitted the launcher's compatibility library environment and failed before editor startup: `/Editor/Unity: error while loading shared libraries: libgconf-2.so.4: cannot open shared object file` (nonzero). No source or project output was accepted from this attempt.
- Corrected exact import command: `timeout --signal=TERM --kill-after=30s 600s env DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1 LD_LIBRARY_PATH=<build-script-compat> <pinned-Unity> -batchmode -nographics -projectPath <exact-project> -quit -logFile unity-import-compile.log` -> exit `0`.
- Unity import artifact SHA-256 `a55e7860f156c76c46d68589b002755ea3cfa1d0b46f25d33dd3b18039a2bdda`; observables: `Initialize engine version: 2019.4.41f2 (6b23d448b533)`, `Batchmode quit successfully invoked`, `Exiting batchmode successfully now`, and no compiler error matches.
- Static gates: `GIT_MASTER=1 git diff --check` exit `0`; `python -m py_compile tests/test_deterministic_unity_build.py` exit `0`; strict Python no-excuse audit reports `no violations in 1 file(s)`; changed Python/C# total pure LOC `72`.
- Existing packager suite: bounded `python -m unittest tests.test_package_physics_player -v` exit `0`, 3/3 pass including tracked product dirtiness rejection; log SHA-256 `e85dbe745f818ce99f392647603781056a821bc7cc23739a65843809fdb5f519`.
- Diagnostics: configured Python and C# LSP servers are not installed (previously declined); `.rsp` and `.meta` have no configured LSP. Exact Unity compilation and executable/static checks provide compatibility evidence.

### Atomic commit (2026-08-06)

- Git-master Phase 0: branch `physics-unity-2019.4`; no upstream; merge-base with `main` is `68480cd06c7ce364a3d203be99bb487c87defb73`; dominant recent style is English semantic/conventional commits (17/30).
- Commit plan exception: the explicit task requires exactly one atomic commit; the response file, Unity metadata, build option, and direct contract test are inseparable because any split intermediate violates the deterministic-build contract.
- Staged files only: `NovPhyBuild.cs`, `Assets/csc.rsp`, `Assets/csc.rsp.meta`, and `tests/test_deterministic_unity_build.py`; `.omo`, generated stages, player output, and wrapper were never staged.
- Initial staged `diff --check` found three trailing spaces in blank Unity metadata values; those were removed and the repeated staged check exited `0`.
- Commit command: `GIT_MASTER=1 git commit -m "fix(unity-build): make staged player reproducible"` -> exit `0`.
- Full commit: `55d6ec93cafc77807342cd7573283f1ed20ca691`; exact subject confirmed; four changed files confirmed by `git diff-tree`.
- Post-commit tracked product check: `GIT_MASTER=1 git diff --quiet HEAD -- . ':(exclude).omo/**'` -> exit `0`; tracked status lists only the six pre-existing `.omo` modifications.

### Post-commit reproducibility proof (2026-08-06)

- Fresh-root gate: both `build-1-stage` and `build-2-stage` were absent before either invocation (exit `0`); the builds used separate stage roots and independently created temporary payload roots.
- Build 1 command: `timeout --signal=TERM --kill-after=30s 900s env NOVPHY_PHYSICS_STAGE="$PWD/.../build-1-stage" ./scripts/build_physics_player.sh` -> exit `0`; Unity log initialized exact `2019.4.41f2 (6b23d448b533)` and exited batch mode successfully.
- Build 2 command: same bounded invocation with separate `build-2-stage` -> exit `0`; same exact Unity identity and successful batch exit.
- Archive SHA-256 for both builds: `1c2a1bbcea87175150451ad8981e7f28ca09195be98f7da4cb8af577d431fef4`.
- Static verifier commands: bounded `python scripts/verify_physics_player.py --stage <each-stage> --skip-runtime`; both exit `0`, report archive checksum and payload checksums true, archive SHA equal, and `runtime: null`.
- Explicit byte comparisons: archive `cmp -s` exit `0`; independently extracted `provenance.json` `cmp -s` exit `0`; `archive.sha256` receipt `cmp -s` exit `0`.
- Extracted provenance SHA-256 for both: `7ea3222052889f0a9c6e3c305bfd718b9f18c5cfbaf496ccb120bec564015ead`; archive receipt SHA-256 for both: `350ab7a9fc4d592a0ad053a89e9be62b74a7997b8a9fc6677cf43103e7c881cf`.
- Post-commit focused contract: bounded run exit `0`, 2/2 pass, log SHA-256 `78dfb02cb1358c3bde235d30e5f193fc5f705eea3028298f570efde4e0b0fc88`.
- Malformed expected digest: bounded verifier with `--expect-sha deadbeef --skip-runtime` exits `1`; stderr verbatim `archive SHA-256 mismatch against --expect-sha`.
- Protected after manifests reproduce baseline hashes exactly; canonical, production, and active-root `cmp -s` exits are each `0`.

### Cleanup and final gates (2026-08-06)

- Preserved root-level archive receipts, extracted provenance bytes, verifier JSON, command exits, hashes, and explicit comparison exits before deleting both owned stage directories.
- Removed only `build-1-stage`, `build-2-stage`, and `tests/__pycache__/test_deterministic_unity_build.cpython-311.pyc`; absence check exited `0`.
- Process, monitored listener, `/tmp/novphy_physics_build_*`, and Unity lock scans are empty (empty-file SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).
- No process was killed. The environment-owned Unity Licensing Client was explicitly excluded; no unfamiliar/shared process was interrupted.
- Deliberate mid-build interruption was not performed because it could leave the shared Unity project Library/lock inconsistent; bounded TERM/KILL wrappers supplied safe cancellation and both builds completed within bounds.
- Final source: commit `55d6ec93cafc77807342cd7573283f1ed20ca691`, tree `c779ce2c5ff7182e1f4af3d99620029245756bad`; product diff excluding `.omo` remains exit `0`.
- Final validation commands: `python -m json.tool` for DoneClaim, cleanup receipt, and reproducibility receipt all exit `0`; machine assertions for `done`, exact commit, equal archives, all three reproducibility comparisons, and all three protected comparisons exit `0`.
- Final evidence SHA-256: DoneClaim `1ec3b8a1c4879318d85741303884be381d80f59cc407a97ee7afafaee73b56bf`; cleanup receipt `ece3c9c510ac458e4dd6bfd72e139a7f569991d4955e7f24f483afeb113a387f`; reproducibility receipt `46eb703a15477df43b096c2891ae37dccef7dcc681bb409d6195408db90fe86c`.
- Final product diff, stage/scan absence, protected `cmp`, exact commit hash, and exact commit subject checks all exit `0`.
