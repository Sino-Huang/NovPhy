# Milestone 0a roadmap reconciliation and execution plan — success handoff

## Completed

- Completed the dependency-ready roadmap reconciliation and Milestone 0a planning item in commit `cbe0bd37b4f909645fd4fb0277730cabfba7e9a7` (`docs(research): plan Milestone 0a macro labels`).
- Resolved the independent review's sole authority-wording contradiction and recorded both reviews in commit `0e947219d07a98f98cfb4c9842736e615c798ee1` (`docs(research): resolve Milestone 0a review`).
- Closed the persistent planning record in commit `ba0588666ea144d5a34f3aaa50f7ab06b30f5a5a` (`docs: close Milestone 0a planning record`).
- Added the corrected current roadmap at `docs/high_level_plans/bg_ns_jepa_research_execution.md`.
- Added the implementation-ready specification at `.claude/project-docs/plans/milestone-0a-macro-outcome-labels.md`.
- Reconciled the accepted runtime re-pin (`repin_complete`, archive `de59061350f78f79420d76ec33f1c506aa17c1cfc25d197cdd2f5f770874e838`) and the completed continuous-only Milestone 1e/1f temporal projection without claiming a full oracle-symbol or joint-alpha result.
- Closed the fixture software contract for the five macro predicates, outcome/capture disposition, terminal equilibrium observation, fixed-step clustering, state/RGB/event identity, canonical `physics_macro_labels_v1` bytes, fail-closed validation, fixtures, evidence, and dependencies.
- Identified the accepted-runtime clock rule that supersedes the pre-repin scaffolding: event occurrence is keyed by `fixed_step`; event `render_frame` is serialization provenance and must not drive event-to-state joins.
- Initial validation passed: `git diff HEAD^ HEAD --check` exited 0 and 29/29 document assertions passed.
- Post-review validation passed all prior assertions plus the explicit fixture-only authority checks. Final roadmap sha256 is `8b7397803060a3d732f0b6d235b9486a6adb4c93fbc5780283cdc6cf8412945d`; final specification sha256 is `e1ad5ef0a4656f8aeb5ab2aade1e37114545382a7d0e3b455339c63ce57013a5`.
- `critics/2026-08-13-critic-1.md` recorded the initial blocking wording contradiction. `critics/2026-08-13-critic-2.md` verified the correction and returned `ACCEPT — safe for the fixture-only implementation` with no remaining blocking, important, or optional issue.
- No runtime/player command, smoke rerun, publication, cohort collection, real label write, health report, training, frozen-schema edit, or protected-root write occurred.

## Authority / Limits

- Runtime publication remains a separate owner decision. This planning work does not authorize it.
- Enriched-cohort collection remains unauthorized. Representative semantic acceptance cannot be manufactured from fixtures or the consumed smoke.
- Fixture software implementation may proceed before publication or cohort authorization, but it must stop before any real cohort operation.
- `cascade-active` termination, `collapsed`, pig-tag completeness, and sparse multi-state projection remain bounded hypotheses pending authorized representative enriched shots.
- Milestone 0b remains separate: it owns the oracle gate `phi*`, threshold authority, and unified enriched supervision.
- `micro` and full `macro` pair scoring, SPSG supervision, physical-violation metrics, the full oracle-symbol ceiling, controller distillation, and joint-alpha claims remain blocked.
- The continuous-only temporal verdict remains `not_supported`; it is not evidence for or against the full joint `(Delta, alpha)` hypothesis.
- The preliminary `physics_derived_labels_v1` implementation is preserved as historical scaffolding and must not be silently reinterpreted or written into a real cohort.

## Next Plan Action

The exact next dependency-ready action is a **fixture-only Milestone 0a implementation session**.

It is next on the critical path because the software contract can now be implemented and verified without runtime publication or cohort collection, while representative semantic acceptance and every downstream oracle-symbol claim still require authorized data.

Required inputs:

1. `.claude/project-docs/plans/milestone-0a-macro-outcome-labels.md`;
2. `docs/data_contracts/physics_capture_v1.schema.json`;
3. `scripts/physics_label_derivation.py:319-516` and `world_model/data/supervision.py:95-165` as preliminary surfaces to supersede;
4. archived accepted-smoke sidecars under `.claude/project-docs/evidence/runtime-repin-gate-20260810/session-5-full-smoke/shot_001/`.

Hard boundaries:

- create a new `physics_macro_labels_v1` path; do not mutate frozen or legacy sidecars;
- use fixed-step event clustering/bracketing and exact state/RGB identity;
- keep Milestone 0b oracle-gate fields out of the artifact;
- operate only on committed/synthetic fixtures and temporary outputs;
- do not launch the player, rerun smoke, publish, collect, label real cohort data, run real health/training, or touch protected roots.

Acceptance criteria:

- schema/types/deriver/validator and explicit opt-in reader support exist;
- accepted-smoke-faithful and synthetic fixtures cover all positive, negative, unavailable, delayed, same-step, stale, malformed, and canonical-byte cases specified by the plan;
- repeated derivation is byte-identical and source-digest bound;
- focused fixture and adjacent regression tests pass with machine-readable evidence under `.claude/project-docs/evidence/milestone-0a-macro-outcome-labels/`;
- the implementation handoff leaves representative semantic acceptance, publication, cohort collection, Milestone 0b, and full symbolic Milestone 1 explicitly blocked.

Smallest first inspection:

```bash
python -m unittest -v tests.test_physics_label_derivation
```

Use the existing test only as a baseline inventory. The implementation must add a new focused `tests.test_physics_macro_labels` suite and must not treat the pre-repin passing result as acceptance of the new semantics.
