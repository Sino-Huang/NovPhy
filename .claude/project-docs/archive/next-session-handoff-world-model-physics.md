# World Model Physics Instrumentation Handoff

Resume `.omo/plans/world-model-physics-instrumentation.md` from the isolated migration worktree:

`/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4`

Before acting, read the full plan, `.omo/boulder.json`, and `.omo/knowledges/unity-2019-4-lts-resume.md`. The user completed the Unity Hub GUI registration and confirmed exact Unity `2019.4.41f2 (6b23d448b533)` is visible. Todo 2 is no longer license-blocked. Todos 1-10 are checked; only F1-F4 remain unchecked.

## Exact Current Release State

- Source HEAD: `1c4e70b2d01d5c6b8b18b0f130ef3bbde06439c6`.
- Two isolated exact-Unity builds matched.
- Current archive SHA-256: `ed6f59723e5b1b5291552487b9e7f2dfb634d01859b09d72c12796e90990c472`.
- Published stage: `sciencebirdsgames/physics-v1` in the migration worktree.
- Accepted-shot metadata binds player `d74bf3f869525a6731b992e30e3beb62da14484c16a6e1ad7a0c73c30ff976fa`, protocol `b8ebda1806e8e440eda2e810c7c2892bf4a7251252bb79f9fb827a1c7af81e7a`, and the archive above.
- Fresh runtime produced one synchronized state and seven events, including exactly one `bird_launched` event.
- `python scripts/verify_physics_capture_docs.py docs` passed immediately before the worker was interrupted.

## Safe Stop State

The active OpenCode release worker was interrupted through its PTY and exited `130`. No release worker, Unity player, smoke driver, Xvfb process, or owned listener remains. Unity Hub was intentionally left untouched. `/tmp/novphy-smoke-output` remains as disposable runtime evidence and must be cleaned after evidence is safely retained.

## Evidence That Must Not Yet Be Treated as Final

- The worker stopped after creating `.omo/evidence/world-model-physics-instrumentation/final-wave-exact-final-native/1c4e70b` but before populating it or writing a fresh cleanup receipt.
- The original smoke command timed out at 300 seconds after creating a valid accepted shot but before writing its report. The worker then reconstructed `final-published-runtime/smoke-report.json` from the retained shot. Independently rerun or rigorously validate the real smoke surface; do not accept the reconstructed report by itself as F3 approval.
- The current `final-published-runtime/cleanup-receipt.json` is from the earlier morning run and is stale for this release.
- Publication used `cp` commands even though it was described as atomic. Audit this claim. If necessary, republish using a same-filesystem temporary name plus atomic rename/replace, with the receipt committed last.
- The worker's manual protected-root check looked under the migration worktree and found production-player/dataset paths absent. That is not proof that the real canonical project, `sciencebirdsgames/Linux`, and `data/novphy_rollouts_dataset_20260708_171531` were unchanged. Recompute and compare the authoritative protected manifests against `/mnt/array/sukaih/Project/NovPhy`.

## Remaining Todos

1. Independently validate the current archive, receipt, provenance, accepted shot, request 38/62 compatibility, request 70 state/events, atomic publication, and actual protected-root manifests.
2. Produce fresh exact-SHA evidence under `final-wave-exact-final-native/1c4e70b`, including current cleanup receipts. Replace reconstructed/stale evidence where required.
3. Run the focused Python verification matrix and `scripts/verify_physics_capture_docs.py docs` against the refreshed evidence.
4. Obtain fresh exact-SHA APPROVE verdicts for F1 plan compliance, F2 code quality, F3 real manual QA, and F4 scope fidelity. Prior verdicts at earlier SHAs are stale.
5. Run the mandatory `review-work` five-lane review and the debugging runtime audit against full SHA `1c4e70b2d01d5c6b8b18b0f130ef3bbde06439c6`.
6. Append every exact-SHA verdict and artifact to `.omo/start-work/ledger.jsonl`, then mark F1-F4 only after approval and reconcile the plan and `.omo/boulder.json` to completed.

## Hard Constraints

- Use only the sibling migration worktree with exact Unity `2019.4.41f2`.
- Never open/build the canonical `tasks/task_template_designer` with Unity 2019.4.
- Never modify `sciencebirdsgames/Linux` or the active dataset.
- Keep enriched capture opt-in and preserve legacy request 38/62 behavior.
- Keep private compatibility libraries scoped through `LD_LIBRARY_PATH`.
- Do not expose credentials, license files, tokens, account identifiers, or raw licensing material.
- Preserve unrelated dirty `.omo` and generated evidence; do not broadly clean or reset the worktree.
