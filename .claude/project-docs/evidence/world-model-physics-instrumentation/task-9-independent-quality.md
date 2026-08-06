# Todo 9 Independent Quality Review

Verdict: **FAIL**

- `codeQualityStatus`: `BLOCK`
- `recommendation`: `REQUEST_CHANGES`
- Scope: `docs/data_contracts/physics_capture_v1.md`,
  `docs/world_model_data_pipeline.md`,
  `scripts/verify_physics_capture_docs.py`, and
  `tests/test_verify_physics_capture_docs.py`, plus the Task 8 archive receipt
  and smoke report.

## Findings

### HIGH

1. The verifier reports success after deletion of multiple required public
   contract sections, so it does not enforce the stated Todo 9 documentation
   contract. `REQUIRED_CLAUSES` contains only timing, provenance, opt-in/API,
   and command fragments ([verify_physics_capture_docs.py:23](/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/scripts/verify_physics_capture_docs.py:23)); it contains no requirement for the Unity-authority definition, support rule, event taxonomy, units/clocks, or bounded-failure policy. `verify_docs` therefore only searches that incomplete tuple before it returns success
   ([verify_physics_capture_docs.py:171](/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/scripts/verify_physics_capture_docs.py:171)).

   An isolated `TemporaryDirectory` probe copied `docs`, deleted each section,
   ran the real CLI, and cleaned up automatically. Results:

   ```text
   support: exit=0; stdout='physics capture documentation verified'; stderr=''
   events: exit=0; stdout='physics capture documentation verified'; stderr=''
   units: exit=0; stdout='physics capture documentation verified'; stderr=''
   failure: exit=0; stdout='physics capture documentation verified'; stderr=''
   temporary_fixture_cleanup=completed
   ```

   The public documentation supplies these sections at
   [physics_capture_v1.md:5](/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/docs/data_contracts/physics_capture_v1.md:5),
   [physics_capture_v1.md:13](/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/docs/data_contracts/physics_capture_v1.md:13),
   [physics_capture_v1.md:23](/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/docs/data_contracts/physics_capture_v1.md:23),
   [physics_capture_v1.md:27](/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/docs/data_contracts/physics_capture_v1.md:27), and
   [physics_capture_v1.md:42](/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/docs/data_contracts/physics_capture_v1.md:42), but the claimed verifier does not guard them. Extend the verifier and focused tests so all required public-contract claims are checked through stable, meaningful invariants rather than a partial phrase list.

2. The provenance check is not bound to the referenced archive receipt or the
   archive file. It reads only the Task 8 smoke JSON and compares its embedded
   digest with `ARCHIVE_SHA256` ([verify_physics_capture_docs.py:159](/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/scripts/verify_physics_capture_docs.py:159)). Neither
   `sciencebirdsgames/physics-v1/archive.sha256` nor the named staged archive
   is opened or hashed anywhere in the verifier. Consequently, a changed or
   stale archive receipt/archive can coexist with an old accepted smoke report
   and still receive `physics capture documentation verified`. The independent
   review did verify that all three currently agree, but that check is missing
   from the delivered verifier. Read and validate the receipt (and, where
   practical, hash the named archive) before success; add an adversarial test
   for receipt/report disagreement.

### MEDIUM

None.

### LOW

None.

## Evidence Checked

- Archive receipt `sciencebirdsgames/physics-v1/archive.sha256` contains
  `c7f9fa4c98480c1c1c8e580cb00454beda4fed4bf28a4822d31c561997906992`.
  An independent `sha256sum` of the archived player returned the same digest.
- The actual accepted Task 8 report at
  `.omo/evidence/world-model-physics-instrumentation/task-8-smoke.json` has
  `status: "accepted"`, `protected_unchanged: true`, a nonempty
  `accepted_shot`, and the same `provenance.archive_sha256`. The verifier
  reads this repository-local report and checks all four properties
  ([verify_physics_capture_docs.py:159](/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/scripts/verify_physics_capture_docs.py:159)).
- Manual QA command: `python scripts/verify_physics_capture_docs.py docs`
  returned exit `0`, stdout `physics capture documentation verified`, and no
  stderr. This pass is genuine for its implemented, limited checks, not proof
  of the full Todo 9 public contract.
- Focused tests: `python -m unittest tests.test_verify_physics_capture_docs -v`
  returned exit `0` (`3` tests, `0` failures). The timing/provenance removal
  test specifically asserts nonzero exit and the `missing required
  documentation clause` error ([test_verify_physics_capture_docs.py:31](/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/tests/test_verify_physics_capture_docs.py:31)). It does not cover the omitted HIGH finding.
- `git diff --check` returned exit `0`. The worktree is intentionally dirty;
  the only scoped tracked documentation changes are the two requested docs,
  while the verifier and its tests are untracked. No unrelated changes were
  modified by this review.

## Quality Perspective Check

The required `remove-ai-slops` and `programming` skill perspectives were read
and applied before judging maintainability/test relevance.

- `remove-ai-slops`: no needless production extraction, parsing, or
  normalization was found in the compact schema-example validator; it is
  scoped to the `capture_failure` branch and that branch uses only the
  keywords it implements. The test uses context-managed temporary fixtures
  and does not leak resources. The literal timing/provenance mutation test is
  relevant to the verifier's published functional boundary rather than a
  deletion-only test with no behavior.
- `programming`: no untyped escape hatch, broad exception catch, needless
  helper, or avoidable production-boundary validation was identified in the
  verifier. Its typed `DocumentationError` and CLI boundary handling are
  appropriate. The critical defect is contract incompleteness, not a code
  hygiene violation.

## UltraQA

- Malformed input: covered for invalid JSON and wrong scalar type in the
  embedded compact example; the `capture_failure` schema branch has only the
  implemented constraints.
- Stale report state: the verifier reads the live repository Task 8 report and
  compares its digest/status/protected-root marker, but cannot establish report
  freshness beyond those contents. No stale report was observed in this run.
- Dirty worktree: handled read-only; scoped files were inspected without
  reverting unrelated work.
- Misleading success output: `physics capture documentation verified` is
  misleading as a full-contract success claim because of the HIGH finding;
  executor evidence paths were present, so there is no separate
  missing-artifact-path blocker.
- Flaky rerun coverage: the focused tests are deterministic and use
  `TemporaryDirectory`; one direct CLI rerun was executed. A second explicit
  rerun test is absent, but no nondeterministic dependency was observed.
- Prompt injection: not applicable; no untrusted instructions were consumed.
- Cancel/resume: not applicable; the verifier is a bounded local CLI with no
  resume state.
- Hung commands: not applicable; all executed commands completed within the
  review timeout and no persistent process was created.
- Repeated interruptions: not applicable; no command interruption occurred.

## Required Resolution

Add meaningful verifier checks and adversarial tests for every required Todo 9
public-contract item, especially authority, support/event definitions,
units/clocks, and bounded capture failure. Re-run the focused suite and CLI
afterward.
