# Learnings

## 2026-07-28 Resume at Todo 7
- Todos 1-6 are independently verified and checked in the plan.
- Curriculum work must derive candidates only from the immutable catalog/dataset snapshot and explicit schedule/seed.
- Existing unrelated dirty-worktree changes must be preserved.

## 2026-07-29 Todo 7 independent verification
- The inherited five-test suite was green but did not catch Python equality accepting boolean/float step bindings or `repr`-based noncanonical identities.
- The corrected curriculum suite has 15 tests and the full `tests.test_world_model_data` module has 66 passing tests under `/home/sukai/miniconda3/bin/python`.
- A real collection-plan fixture proved novelty/scenario filters select only exact `source_level_key` provenance; no directory-name fallback is used.

## 2026-07-29 Todo 8 temporal ablations
- Canonical digest-based candidate ranking provides deterministic, process-stable local-seed selection without global random state or version-sensitive process hashing.
- Fixed draw count does not imply fixed compute: two draws at one prediction step cost 2 predicted-frame units, while two draws at four prediction steps cost 8 under the same rule.
- Equal compute does not imply sample matching: four short draws and one long draw can both cost 4 while their draw counts differ.
- A real temporary catalog and Todo 7 policy produced stable JSON manifests without opening images or sidecars, and TemporaryDirectory cleanup removed the fixture root.

## 2026-07-29 Todo 9 integration and active-root QA
- A real plan-backed train/dev/test fixture proves stable catalog, sampler, curriculum, collator, sampled-provenance, and ablation-manifest behavior across resume.
- Large read-only inspection must summarize validated frame counts without materializing millions of frame records; normal catalog validation still checks individual frame readability.
- The active root produced 4,975 accepted train episodes and 463 accepted dev episodes within the 300-second bound, while typed partial artifacts remained `missing_artifact` rejections.
- Before/after metadata fingerprints were identical across 7,393,607 files, proving the inspector did not mutate the active root.

## 2026-07-30 Todo 9 independent-verification remediation
- Metadata-summary eligibility is not canonical acceptance when per-frame accessibility is intentionally omitted for bounded operation; distinct result and report types prevent overclaiming.
- No-plan inspection cannot report source-level novelty/scenario composition honestly, even when directory names appear structured; explicit unavailability is the only valid result.
- Keeping public imports stable does not require keeping responsibilities colocated: immutable outcomes, plan parsing, record conversion, typed errors, and provenance invariants can be split without changing catalog order.
