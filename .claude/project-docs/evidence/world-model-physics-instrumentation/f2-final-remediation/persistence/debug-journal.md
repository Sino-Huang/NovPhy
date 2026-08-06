# Debug Journal - Atomic collector publication

Started: 2026-08-06
Goal: Finalize and validate accepted physics-shot metadata in `shot_NNN.tmp`, then publish with one descriptor-relative atomic rename and no accepted-shot write afterward.

## Environment snapshot

- Runtime: CPython 3.13.9 via `python3`
- Plan: `.omo/plans/world-model-physics-instrumentation.md`, read in full
- Owned source: `scripts/physics_rollout_persistence.py`, `scripts/collect_rollouts.py`, `tests/test_collect_rollouts.py`
- References read: debugging Python runtime; setup, investigation, fix, QA, cleanup; programming Python README
- Working tree: dirty before this task; unrelated edits must remain untouched

## Hypotheses

1. [CONFIRMED BY SOURCE] Publication ordering is wrong: `collect_rollouts()` renames `shot_NNN.tmp` to `shot_NNN`, then rewrites path metadata, revalidates, and rewrites `metadata.json`. Distinguishing evidence: a write-open or injected failure after the publication rename is observable under `shot_NNN`. If true, fix is: finalize before publish.
2. [CONFIRMED BY SOURCE] Trusted descriptor lifetime is too short: `persist_physics_rollout()` closes the shot descriptor before metadata installation, so a root-path swap can redirect the final metadata write. Distinguishing evidence: swap the opened temporary-shot pathname during metadata finalization and observe whether an external sentinel or replacement tree receives `metadata.json`. If true, fix is: retain descriptor.
3. [CONFIRMED BY ADVERSARIAL PROBE] Path-based replacement permits symlink/destination substitution during finalization or publication. Distinguishing evidence: install a metadata or final-shot symlink immediately before replacement and observe whether external bytes change or publication follows the substituted path. Fix: descriptor-relative no-follow replacement with destination and inode checks.

## Artifacts to clean

- [x] Probe scratch directories under this evidence directory; removed after QA.
- [x] Python bytecode caches under the evidence directory; none present after QA.
- [ ] Temporary red-test output; retain as evidence, not a cleanup target.
- [x] Absolute scratch root matching `probe-scratch-*` under this evidence directory; removed recursively and recorded absent in `probe-cleanup.json`.

## Findings

- Source inspection: accepted enriched flow writes `metadata.json` after `shot_dir.replace(final_shot_dir)`.
- Source inspection: `_record_fresh_engine_attempt_metadata()` can reopen published enriched metadata after collection returns.
- Source inspection: `persist_physics_rollout()` closes `output_descriptor` before capture and writes metadata by mutable pathname.

### Red phase

Command: `python3 -m unittest -v tests.test_collect_rollouts.PhysicsCapturePersistenceTests.test_metadata_finalization_stays_on_trusted_shot_after_root_symlink_swap tests.test_collect_rollouts.PhysicsCapturePersistenceTests.test_accepted_physics_shot_is_complete_before_single_atomic_publication`

Observed: `FAILED (failures=5, errors=1)`.

- H1 confirmed: validation paths were `[shot_001.tmp, shot_001]`, published metadata bytes differed from pre-publication bytes, and two published `metadata.json` write-opens were observed.
- H2 confirmed: the root swap overwrote the external sentinel and left no metadata in the descriptor-retained directory.
- H3 confirmed and closed: descriptor-relative metadata replacement preserved external sentinels, and destination/temporary-entry swaps failed closed without publication.

### Independent real-filesystem adversarial probes

- CPython 3.13.9 executed six cases with actual directories, symlinks, external sentinels, file descriptors, and delegated `os.replace` calls.
- Observed result: `6/6` passed; the successful shot publication used one `shot_005.tmp` to `shot_005` replacement with equal non-null directory fds, byte-identical metadata, and zero post-rename write-opens beneath the accepted shot.
- Cleanup result: all seven recorded absolute scratch paths had `lexists=false`; no evidence symlink, `__pycache__`, spawned process, or opened port remained.

## Final fix

- `persist_physics_rollout()` retains the trusted shot descriptor through sidecar closure and atomic metadata installation.
- `install_physics_metadata()` creates, flushes, and descriptor-relatively replaces `metadata.json`, then syncs the shot directory.
- `collect_rollouts()` enriches accepted metadata, validates the descriptor-stable temporary shot, and delegates exactly one publication rename to `publish_physics_shot()`.
- `publish_physics_shot()` verifies temporary-entry inode identity and destination absence before one equal-directory-fd `os.replace`, then syncs the root directory.
- Accepted enriched fresh-engine rollouts carry `fresh_engine_attempt` before publication and are not reopened for metadata mutation afterward.

## Final verification

- Failing-first pair: red result was `FAILED (failures=5, errors=1)` for the intended ordering and descriptor-lifetime defects; both tests pass after the fix.
- Persistence class: `30` tests passed in `1.467s` on CPython 3.13.9.
- Real filesystem rerun: `6/6` adversarial cases passed against current source; publication count `1`, equal non-null root directory fds, byte-identical metadata, and zero post-publication write opens.
- Static syntax gate: `py_compile` passed for all three owned Python files.
- Diff integrity: `git diff --check` passed for all three owned Python files.
- No-excuse audit: the remediation's two new silent `FileNotFoundError` handlers were removed; `53` findings remain on unchanged legacy lines in the pre-existing oversized modules.
- LSP diagnostics: unavailable because `basedpyright` is not installed and installation was previously declined.
- Cleanup: the probe scratch root and dedicated verification pycache are absent.
