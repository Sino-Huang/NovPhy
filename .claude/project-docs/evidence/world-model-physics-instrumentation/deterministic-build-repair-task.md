TASK: Repair the Unity 2019.4 staged-player build so two independent exact-method builds from one committed source SHA produce byte-identical archives, then commit the minimal verified source fix.

You are a high-difficulty implementation worker. You are not alone in this shared worktree. Preserve all unfamiliar changes and never revert other agents' work. Work only in `/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4`.

DELIVERABLE

- A minimal failing-first deterministic-build repair, committed atomically on branch `physics-unity-2019.4`.
- A DoneClaim JSON under `.omo/evidence/world-model-physics-instrumentation/deterministic-build-repair/` with exact commit SHA, changed files, tests, real-surface QA, adversarial probes, risks, and cleanup receipt.
- At least two independently invoked builds from the exact committed SHA with byte-identical archive and provenance bytes (`cmp` exit 0 and equal SHA-256 values).

SCOPE

- Primary source ownership: `tasks/task_template_designer/Assets/Scripts/Editor/NovPhyBuild.cs` and the narrowest directly relevant tests. Touch `scripts/build_physics_player.sh`, `scripts/package_physics_player.py`, or their tests only if runtime evidence proves the C# build option alone is insufficient.
- Evidence ownership: `.omo/evidence/world-model-physics-instrumentation/deterministic-build-repair/`.
- Generated stage outputs may be written only beneath the assigned evidence directory until the final verified archive is published to `sciencebirdsgames/physics-v1`.
- Never modify canonical `/mnt/array/sukaih/Project/NovPhy/tasks/task_template_designer`, production `/mnt/array/sukaih/Project/NovPhy/sciencebirdsgames/Linux`, or active dataset `/mnt/array/sukaih/Project/NovPhy/data/novphy_rollouts_dataset_20260708_171531`.
- Preserve all `.omo` and unrelated dirty state. Do not mark F1-F4 or edit Boulder/ledger/plan.

1. PIN / RED / ROOT CAUSE

- Read and follow the full `omo:debugging` skill, including Python/runtime setup and methodology references for phases entered. Read and follow `omo:git-master` before committing.
- Create the debug journal before new debug artifacts.
- PIN the unchanged build interface and exact Unity identity.
- Use the retained RED evidence directly: commit `95e7b82919c7cc2cd1aa911c4ebf9ffcfba8f9fe` produced archives `e8018cb56525b719a854793060638dbc4345cd2a0921c258bf44bd4f9a5627a6`, `61c39c22d4a670e1fde15762ae1441f483240f7c06c7e9edbe13a0a107f19130`, and `20341205a130f5f46378ed5fe69dcb3d093a175b6be6563ab0dbae60fea2c913`; both consecutive `cmp` checks exit 1. Artifact: `.omo/evidence/world-model-physics-instrumentation/final-archive-rebuild/stop-hook-3-direct-verification.json`.
- Form at least three orthogonal hypotheses: Unity build options, packager metadata, and stale/generated project state. Run distinguishing checks. Confirm root cause by toggling the suspected cause and showing the mismatch toggles.
- Add a failing test or executable source/build assertion before production edits. A static assertion that merely mirrors a new constant is insufficient by itself; the retained real double-build mismatch is the required behavioral RED surface.

2. IMPLEMENTATION

- Make the smallest supported Unity 2019.4-compatible fix. Investigate `BuildOptions.DeterministicBuild`, but do not assume it is sufficient without real builds.
- Do not post-process arbitrary player bytes, replace generated payloads with retained bytes, or weaken provenance/dirty-source checks.
- Preserve `BuildOptions.StrictMode`, exact `2019.4.41f2 (6b23d448b533)`, request 38/62 compatibility, and request 70 behavior.

3. AUTOMATED VERIFY

- Run the narrow relevant Unity/EditMode and Python/static suites before and after the fix.
- Run `git diff --check` and compile/static checks applicable to every changed source/test file.
- Commit only the minimal source/test fix after PIN/RED/GREEN evidence exists. Use repository commit style and record the full SHA.

4. MANUAL QA CHANNEL

- Exact terminal surface after commit, twice from clean generated roots:
  `timeout --signal=TERM --kill-after=30s 900s env NOVPHY_PHYSICS_STAGE="$PWD/.omo/evidence/world-model-physics-instrumentation/deterministic-build-repair/build-1-stage" ./scripts/build_physics_player.sh`
  and the same command with `build-2-stage`.
- PASS only when both exit 0, archive SHA-256 values are identical, archive `cmp -s` exits 0, extracted provenance bytes are identical, and `python scripts/verify_physics_player.py --stage <each-stage> --skip-runtime` exits 0.
- Then run a bounded real staged-player verifier against one final stage and require legacy request 62 plus request 70 true. Also run the exact request 38/62 fixture compatibility test.
- Publish the verified byte-identical archive/receipt/build log to `sciencebirdsgames/physics-v1` only after all gates pass and verify its SHA equals the two evidence builds.

5. ADVERSARIAL QA

- `stale_state`: clean generated roots and compare both builds; ensure no retained payload substitution.
- `dirty_worktree`: packager must still reject tracked product dirtiness before publication while preserving unrelated `.omo` state.
- `hung_or_long_commands`: every Unity/runtime invocation bounded by timeout with TERM/KILL behavior recorded.
- `flaky_tests`: repeat the focused deterministic test/build comparison.
- `misleading_success_output`: decide by exits, hashes, `cmp`, decoded verifier JSON, and filesystem bytes, not Unity success text.
- `cancel_resume` and `repeated_interruptions`: terminate an interrupted evidence-only build or document why no interruption is safely needed; prove no lock/process/temp residue.
- `malformed_input`: wrong expected SHA must fail.
- `prompt_injection`: not applicable because inputs are local structured build artifacts; record that reason.

6. CLEANUP

- Remove debug journal/temp roots and all owned Unity/Xvnc/Java/player processes, listeners, locks, and temporary clones. Preserve required evidence and the final published stage only.
- Capture a JSON cleanup receipt proving port 2004 and all allocated ports are free, owned processes absent, temp roots absent, and protected manifests unchanged.

7. VERIFY / RETURN

- Return `DoneClaim` only if all criteria pass. Otherwise return `BlockedClaim` with exact failing invocation and artifact; do not claim partial work as done.
- Every criterion must name scenario, exact invocation, binary observable, and nonempty artifact path.
- Send `WORKING: deterministic build repair - <phase>` before long build passes.
