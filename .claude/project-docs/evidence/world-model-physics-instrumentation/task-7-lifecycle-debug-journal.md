# Debug Journal — Todo 7 migrated scene lifecycle
Started: 2026-08-05T16:47:26+10:00
Goal: Initialize ABGameWorld.slingshotBaseTransform on every loading path and prove request 62/70 compatibility in the staged player.

## Environment snapshot (Phase 0)
- Runtime: Unity 2019.4.41f2 migrated project; staged Linux Mono player; Python verifier.
- Entry: Unity EditMode tests, scripts/build_physics_player.sh, scripts/verify_physics_player.py.
- Ports / sockets: verifier defaults agent=22004, game=29001, physics=2004; no task-owned runtime process present at baseline.
- Git HEAD: d72d1af1eb7b9a3cd60b0acb5364689e16735ea5; shared dirty worktree preserved.
- References read: debugging/SKILL.md, runtimes/native-binary.md, methodology/00-setup.md, methodology/02-investigate.md.

## Hypotheses
1. [CONFIRMED BY SOURCE PATH + PRIOR LIVE SYMPTOM] The pre-populated scene branch skips the sole sling-base assignment after DecodeLevel — distinguishing evidence: invoke ABGameWorld.Start with Blocks.childCount > 0 and inspect slingshotBaseTransform — if true, fix is: shared initializer.
2. [REFUTED BY PATH] DecodeLevel creates no slingshot_base child — distinguishing evidence: instantiated Slingshot prefab hierarchy and post-DecodeLevel lookup — if true, fix is: prefab repair.
3. [REFUTED BY PATH] Request handlers run before Start — distinguishing evidence: PLAYING state and repeated ABSlingshot.Update null access after scene start — if true, fix is: readiness gate.

## Failed hypothesis round counter
- Round 1: H1 distinguished by exclusive Start branches; H2/H3 contradicted by the existing empty-scene assignment and PLAYING lifecycle evidence.

## Artifacts to revert
- [ ] `.debug-journal.md` — session ledger. Remove after evidence is promoted.
- [ ] `tasks/task_template_designer/Assets/Tests/Editor/ABGameWorldLifecycleTests.cs` and `.meta` — regression test; promote as shipped test.
- [ ] `tasks/task_template_designer/Assets/Scripts/GameWorld/ABGameWorld.cs` — minimal lifecycle fix; promote as shipped fix.
- [ ] `.omo/evidence/world-model-physics-instrumentation/task-7-lifecycle-*` — durable evidence; retain.
- [ ] staged `sciencebirdsgames/physics-v1` rebuild — task output outside protected Linux root; retain only final stage.
- [ ] Unity/player/Java/Xvfb processes and verifier temporary extraction — terminate/remove before completion.

## Findings
- 2026-08-05T16:47:26+10:00 — Source `ABGameWorld.cs`: Start assigns slingshotBaseTransform only inside the empty-scene `else`, after DecodeLevel.
- 2026-08-05T16:47:26+10:00 — Prior Todo 7 DoneClaim: player reached PLAYING but request 62 timed out on ABGameWorld null references and request 70 closed without envelope.

## Final fix
- Pending red-to-green regression.
