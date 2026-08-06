# Debug Journal - Todo 7 symbolic runtime
Started: 2026-08-05T18:00:00+10:00
Goal: Repair the staged-player SymbolicGameState.GetGTJson null failure without changing valid legacy bytes.

## Environment snapshot (Phase 0)
- Runtime: Unity 2019.4.41f2 editor and Linux Mono player.
- Entry: Unity EditMode tests, scripts/build_physics_player.sh, staged Java bridge verifier.
- Ports / sockets: task-owned isolated player/bridge ports only; none launched yet.
- Git HEAD: d72d1af1eb7b9a3cd60b0acb5364689e16735ea5; shared dirty overlay preserved.
- References read: debugging/SKILL.md, methodology/00-setup.md, methodology/02-investigate.md.

## Hypotheses
1. [CONFIRMED] The verifier launched by filename resolves `src.webui.bridge` from the canonical checkout supplied by `PYTHONPATH`, before the worktree root. Distinguishing evidence: reproduce the import from `scripts/` and print `bridge.__file__` plus method presence.
2. [REFUTED] `prepare_for_play` mutates or replaces the bridge class between request 62 and request 70. Distinguishing evidence: search the function and imports for `sys.modules`, reload, class assignment, or attribute mutation.
3. [REFUTED] The worktree bridge itself lacks the method or has a stale bytecode cache. Distinguishing evidence: fresh import from the worktree root, source hash, source path, and method presence.

## Failed hypothesis round counter
- Round 1: decisive; H1 confirmed, H2/H3 refuted.

## Artifacts to revert
- [ ] `.debug-journal.md` - remove after promoting evidence copy.
- [ ] `tasks/task_template_designer/Assets/Tests/Editor/LegacyGroundTruthTests.cs` - promote focused regression.
- [ ] `tasks/task_template_designer/Assets/Scripts/GroundTruth/SymbolicGameState.cs` - promote minimal fix.
- [ ] `.omo/evidence/world-model-physics-instrumentation/task-7-symbolic-runtime-*` - retain completion evidence.
- [ ] `sciencebirdsgames/physics-v1` final rebuilt stage - retain requested output.
- [ ] task-owned temporary player/Java/Xvfb processes and temp roots - remove before completion.

## Findings
- 2026-08-05T18:00:00+10:00 - IL artifact places live failure at the pig GTObject.ToJsonString call; synthetic missing-collider green does not reproduce the staged prefab.
- 2026-08-05T17:45:00+10:00 - Fresh worktree-root import resolves `/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/src/webui/bridge.py`; `hasattr(ScienceBirdsBridge, "get_physics_capture_v1")` is `True`.
- 2026-08-05T17:48:00+10:00 - Filename entrypoint environment has `PYTHONPATH=/home/sukai/Project/NovPhy`. Reproducing the import from `scripts/` resolves `/home/sukai/Project/NovPhy/src/webui/bridge.py`; method presence is `False`.
- 2026-08-05T17:48:00+10:00 - `scripts/manual_agent.py` contains no module/class mutation. It inserts its own root only when absent; the verifier has no root bootstrap.

## Final fix
- `scripts/verify_physics_player.py` inserts its resolved worktree root before importing project modules, preventing an inherited canonical-checkout `PYTHONPATH` from supplying an obsolete bridge.
- Red: `.omo/evidence/world-model-physics-instrumentation/task-7-verifier-import-red.log` resolves the shadow bridge and fails.
- Green: `.omo/evidence/world-model-physics-instrumentation/task-7-verifier-tests-green.log` passes 4/4.
- Manual QA: `.omo/evidence/world-model-physics-instrumentation/task-7-final4-runtime.json` proves request 62, request 70, render frame 9759, archive checksum, and payload checksums.
- Remaining integrity blocker: the protected canonical checkout gained untracked `ABGameWorldLifecycleTests.cs` and `.meta` at 16:48, after the Todo 7 baseline and before this resumed investigation. These unfamiliar files were preserved.
