# Debug Journal - F2 Source Remediation

Started: 2026-08-06T00:00:00+10:00
Goal: Reproduce the nine F2/F1 allegations, fix only confirmed defects, and retain exact red/green/QA evidence.

## Environment snapshot (Phase 0)

- Runtime: CPython 3.11.15 at `/home/sukai/miniconda3/envs/novphy/bin/python3`; Unity 2019.4 C# sources and POSIX shell packaging scripts.
- Entry: focused `python3 -m unittest` tests plus repository packaging/verifier CLIs and bounded Unity EditMode tests.
- Git HEAD: `55d6ec93cafc77807342cd7573283f1ed20ca691` on `physics-unity-2019.4`.
- Existing dirty state: `.omo` plan/notepad/ledger and evidence artifacts, untracked published stage, and untracked `scripts/9001-player-wrapper.sh`; preserve all unrelated state.
- References read: programming `SKILL.md`, Python `README.md`; debugging `SKILL.md`, Python runtime, setup, investigation, fix, QA, cleanup; git-master `SKILL.md`.

## Hypotheses

1. [OPEN] Contract-shape drift lets the docs consumer accept `status=accepted` plus nested provenance while the collection consumer insists on `status=passed` plus top-level hashes. Distinguishing evidence: one marker accepted by the docs verifier and rejected by `resolve_physics_capture_provenance`. If true, fix is: canonical parser.
2. [OPEN] Accepted-shot publication exposes a renamed directory before final metadata is rewritten and validated. Distinguishing evidence: inject a rewrite/validation failure and observe the final shot path exists. If true, fix is: finalize before rename.
3. [OPEN] Request-70 serialization constructs strings/lists before enforcing caps, and synchronous socket receive has no deadline. Distinguishing evidence: oversized contact payload allocates/materializes past caps or a nonresponsive peer blocks beyond the requested bound. If true, fix is: bounded streaming/deadline.
4. [OPEN] Output confinement validates descriptors but later finalizes metadata through path-based operations, allowing ancestor or child replacement after validation. Distinguishing evidence: swap a checked path to a symlink before finalization and observe outside-root modification. If true, fix is: descriptor-relative finalize.
5. [OPEN] Packaging derives input identity only from tracked Git state and writes the final archive directly, while generated worker extraction trusts tar members. Distinguishing evidence: untracked wrapper changes archive content without provenance rejection; interrupted packaging leaves a final-path artifact; hostile archive escapes or expands without bounds. If true, fix is: input inventory/atomic archive/safe extraction.
6. [OPEN] Verifier archive parsing lacks regular-file, member-count, size, and safe-path checks; bridge JSON recursion leaks `RecursionError`. Distinguishing evidence: crafted tar members and nested JSON produce acceptance or untyped exception. If true, fix is: bounded boundary parser.
7. [OPEN] Shot recorder timestamps use absolute `Time.fixedTime`. Distinguishing evidence: source-level/fixture test expects first shot-relative fixed time near zero but observes session time. If true, fix is: subtract origin.
8. [OPEN] Unity JSON escaping handles only named escapes, leaving other U+0000-U+001F raw. Distinguishing evidence: serialize U+0001 and parse failure/raw byte. If true, fix is: unicode escape.
9. [OPEN] The staged archive includes the currently untracked wrapper, so the build can consume content absent from the bound Git tree. Distinguishing evidence: archive member hash equals untracked file while `git ls-files --error-unmatch` fails. If true, fix is: tracked-input gate.

## Failed hypothesis round counter

- Round 1: pending.

## Artifacts to revert

- [ ] Temporary directories created by focused reproductions. Revert: remove only paths named in `cleanup-receipt.json`.
- [ ] Test/source edits in assigned files. Revert: none; these become the requested fix and direct regression coverage.

## Findings

### Ownership conflict stop

- Source: root coordinator message during red-test authoring.
- Value: five active remediation workers own Unity runtime, verifier, persistence/collector, package/provenance, and bridge/docs paths.
- Interpretation: every assigned production path overlaps; continuing would violate shared-worktree ownership.
- State at stop: no production edits and no commits. Four direct test files contain uncommitted red-test additions and are intentionally preserved for the owning workers to inspect or incorporate.

## Final fix

Blocked by overlapping active ownership before production implementation.
