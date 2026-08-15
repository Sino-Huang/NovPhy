# Current Gates and Recovery

## Authority Order

When records disagree, use this order:

1. Current machine-readable blocker or acceptance evidence in the active worktree.
2. The active plan that names that evidence.
3. The high-level research roadmap.
4. This compact note.
5. Historical knowledge notes, logs, and temporary handoffs.

Current authorities and ownership:

- **Unity worktree:** `/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4`, branch `physics-unity-2019.4`, recovery baseline commit `fc3e34c5f6115b55751874df21c3a9286bd09a5e`.
- **Current decision:** `/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/.claude/project-docs/evidence/enriched-cohort-oracle-labels/task-7-unity-exporter-boundary.json`.
- **Original collection rejection:** `/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/.claude/project-docs/evidence/enriched-cohort-oracle-labels/task-7-collection-blocker.json`.
- **Historical live-contract blocker:** `/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/.claude/project-docs/evidence/world-model-physics-instrumentation/task-8-live-contract-blocker.json`; this predates the later task-7 decision and does not override it.
- **Detailed active plan:** `/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/.claude/project-docs/plans/enriched-cohort-oracle-labels.md`.
- **High-level roadmap:** `/mnt/array/sukaih/Project/NovPhy/docs/high_level_plans/bg_ns_jepa_research_execution.md`; where it implies readiness, the later task-7 blocker controls.

These owner paths must be promoted into a durable canonical note before any original knowledge file or temporary handoff is archived. Do not treat `/tmp` handoffs, `.omo/`, static-build receipts, or old test summaries as current status without checking their provenance.

## Active Gate

The status remains `still_blocked`. The enriched `physics_capture_v1` cohort is blocked until the final candidate proves both:

1. **Listener-to-candidate provenance:** one unambiguous listener process bound to the exact executable, cwd/payload, assembly digest, archive provenance, and launched process tree.
2. **Collision validity:** at least one genuine collision with non-empty, sorted, unique `contact_ids` and finite non-negative `relative_speed`.

A valid request-70 identity also requires two responses with the same non-empty `capture_id` and strictly increasing `sequence`.

`ready_for_repin_approval` means the exact candidate passed the proof boundary. It does not authorize publication. Re-pin or publication requires a separate explicit approval.

## Prohibited While Blocked

Do not:

- collect the Milestone 0 enriched cohort;
- emit the Todo 8 health report;
- start F1-F4, JEPA, SPSG, or controller work;
- weaken the frozen schema, taxonomy, or Python consumer;
- re-pin, publish, or overwrite `sciencebirdsgames/physics-v1/`;
- claim multi-regime coverage from single-level work;
- touch protected roots, active rollout data, `.omo/`, `.claude/logs/`, or protected untracked review evidence.

The oracle-symbol paper fallback is downstream scope only. It still requires a valid enriched-data basis and does not bypass this gate.

## Recovery Procedure

1. Lock branch, committed source, staged pin, editor, package inputs, and protected-root baseline.
2. Resolve the port-2004 listener from socket ownership first; reject ambiguous or substring-based process identity.
3. Bind the listener PID to `/proc/<pid>/exe`, cwd/payload, `Assembly-CSharp.dll`, archive provenance, and the launched Java process tree.
4. Add focused failing regression tests before changing smoke or provenance tooling.
5. Preserve response evidence even if a later assertion fails.
6. Build the exact committed candidate twice in isolated stages and compare archive, player, assembly, and provenance digests.
7. Run exactly one bounded full live smoke against one verified candidate.
8. Require valid request identity and a genuine valid collision.
9. Verify cleanup and protected-root equality.
10. Run exact-source review and record machine-readable evidence.

There is no fallback success path. Any stale process, occupied port, ambiguous ownership, digest mismatch, malformed identity, missing collision, nondeterminism, cleanup failure, or protected-root drift ends `still_blocked`.

## Provenance Rule

Mutable facts must be linked, not copied as timeless knowledge. Record current branch/commit, digests, test counts, smoke result, cleanup state, editor/license state, and approval state in dated evidence:

> Claim: `<invariant or status>`
> Evidence: `<receipt path>`, commit `<commit>`, SHA-256 `<digest>`

Historical static builds or test passes do not override the current runtime gate.
