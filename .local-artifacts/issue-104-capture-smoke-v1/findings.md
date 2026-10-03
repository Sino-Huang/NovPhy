# #104 rendered smoke (`issue-104-capture-smoke-v1`)

`python -u -m scripts.run_power_cohort_smoke --validate`; cohort plan `sha256:28d891dee3765b5c...`.

**Campaign launch: `supported`** — every smoke gate passed: the frozen campaign may launch.

| gate | pass | evidence |
| --- | --- | --- |
| S1_capture_reliability | True | {"typed_failures": 0, "branches": 168, "rate": 0.0, "classes": {}} |
| S2_renderer | True | {"mismatched": [], "renderer_mismatch_failures": 0} |
| S3_pixel_identity | True | {"equal": 8, "replay_branches": 8} |
| S4_window_rule | True | {"violations": [], "explanations": {"censored at the cap": 89, "level_clear inside a tail": 6, "rest tail complete": 22, "level_fail inside a tail": 29, "level_fail": 19, "level_clear": 3}} |

Cells: 45 supported, 0 unsupported (#110 per-template criteria).

| level type | branches | complete | terminals | shots with rest frames | rest frames | collapse shots |
| --- | --- | --- | --- | --- | --- | --- |
| novelty_level_0 | 80 | 80 | {'native_time_window_limit': 44, 'level_clear': 2, 'rest_tail_complete': 8, 'level_fail': 26} | 27 | 917 | 23 |
| novelty_level_1 | 10 | 10 | {'native_time_window_limit': 6, 'level_clear': 1, 'rest_tail_complete': 1, 'level_fail': 2} | 4 | 90 | 1 |
| novelty_level_2 | 10 | 10 | {'native_time_window_limit': 9, 'rest_tail_complete': 1} | 1 | 51 | 2 |
| novelty_level_3 | 10 | 10 | {'native_time_window_limit': 9, 'rest_tail_complete': 1} | 1 | 51 | 4 |
| novelty_level_4 | 10 | 10 | {'level_clear': 3, 'rest_tail_complete': 2, 'level_fail': 5} | 2 | 102 | 4 |
| novelty_level_5 | 10 | 10 | {'native_time_window_limit': 6, 'level_fail': 3, 'rest_tail_complete': 1} | 3 | 87 | 3 |
| novelty_level_6 | 10 | 10 | {'level_fail': 5, 'native_time_window_limit': 4, 'rest_tail_complete': 1} | 6 | 139 | 2 |
| novelty_level_7 | 10 | 10 | {'native_time_window_limit': 5, 'rest_tail_complete': 1, 'level_clear': 1, 'level_fail': 3} | 3 | 122 | 5 |
| novelty_level_8 | 10 | 10 | {'native_time_window_limit': 4, 'rest_tail_complete': 6} | 6 | 306 | 3 |

| template | supported | reasons | force sign | storm onset in window |
| --- | --- | --- | --- | --- |
| issue-110-smoke-type010101-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010101-novel-001 | True |  | not_applicable | None |
| issue-110-smoke-type010102-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010102-novel-001 | True |  | not_applicable | None |
| issue-110-smoke-type010103-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010103-novel-001 | True |  | not_applicable | None |
| issue-110-smoke-type010104-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010104-novel-001 | True |  | not_applicable | None |
| issue-110-smoke-type010105-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010105-novel-001 | True |  | not_applicable | None |
| issue-110-smoke-type010201-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010201-novel-001 | True |  | disagree | None |
| issue-110-smoke-type010202-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010202-novel-001 | True |  | disagree | None |
| issue-110-smoke-type010203-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010203-novel-001 | True |  | disagree | None |
| issue-110-smoke-type010204-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010204-novel-001 | True |  | disagree | None |
| issue-110-smoke-type010205-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010205-novel-001 | True |  | not_exercised | None |
| issue-110-smoke-type010301-normal-001 | True |  | not_exercised | None |
| issue-110-smoke-type010301-novel-001 | True |  | not_exercised | None |
| issue-110-smoke-type010302-normal-001 | True |  | not_exercised | None |
| issue-110-smoke-type010302-novel-001 | True |  | agree | None |
| issue-110-smoke-type010303-normal-001 | True |  | not_exercised | None |
| issue-110-smoke-type010303-novel-001 | True |  | not_exercised | None |
| issue-110-smoke-type010304-normal-001 | True |  | not_exercised | None |
| issue-110-smoke-type010304-novel-001 | True |  | not_exercised | None |
| issue-110-smoke-type010305-normal-001 | True |  | not_exercised | None |
| issue-110-smoke-type010305-novel-001 | True |  | agree | None |
| issue-110-smoke-type010401-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010401-novel-001 | True |  | agree | None |
| issue-110-smoke-type010402-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010402-novel-001 | True |  | agree | None |
| issue-110-smoke-type010403-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010403-novel-001 | True |  | agree | None |
| issue-110-smoke-type010404-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010404-novel-001 | True |  | agree | None |
| issue-110-smoke-type010405-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010405-novel-001 | True |  | not_exercised | None |
| issue-110-smoke-type010501-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010501-novel-001 | True |  | not_applicable | None |
| issue-110-smoke-type010502-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010502-novel-001 | True |  | not_applicable | None |
| issue-110-smoke-type010503-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010503-novel-001 | True |  | not_applicable | None |
| issue-110-smoke-type010504-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010504-novel-001 | True |  | not_applicable | None |
| issue-110-smoke-type010505-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010505-novel-001 | True |  | not_applicable | None |
| issue-110-smoke-type010601-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010601-novel-001 | True |  | not_applicable | None |
| issue-110-smoke-type010602-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010602-novel-001 | True |  | not_applicable | None |
| issue-110-smoke-type010603-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010603-novel-001 | True |  | not_applicable | None |
| issue-110-smoke-type010604-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010604-novel-001 | True |  | not_applicable | None |
| issue-110-smoke-type010605-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010605-novel-001 | True |  | not_applicable | None |
| issue-110-smoke-type010701-normal-001 | True |  | not_exercised | None |
| issue-110-smoke-type010701-novel-001 | True |  | not_exercised | None |
| issue-110-smoke-type010702-normal-001 | True |  | not_exercised | None |
| issue-110-smoke-type010702-novel-001 | True |  | not_exercised | None |
| issue-110-smoke-type010703-normal-001 | True |  | agree | None |
| issue-110-smoke-type010703-novel-001 | True |  | agree | None |
| issue-110-smoke-type010704-normal-001 | True |  | not_exercised | None |
| issue-110-smoke-type010704-novel-001 | True |  | not_exercised | None |
| issue-110-smoke-type010705-normal-001 | True |  | not_exercised | None |
| issue-110-smoke-type010705-novel-001 | True |  | not_exercised | None |
| issue-110-smoke-type010801-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010801-novel-001 | True |  | not_exercised | [False, False] |
| issue-110-smoke-type010802-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010802-novel-001 | True |  | not_exercised | [False, True] |
| issue-110-smoke-type010803-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010803-novel-001 | True |  | not_exercised | [False, True] |
| issue-110-smoke-type010804-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010804-novel-001 | True |  | not_exercised | [True, True] |
| issue-110-smoke-type010805-normal-001 | True |  | not_applicable | None |
| issue-110-smoke-type010805-novel-001 | True |  | not_exercised | [True, True] |

Replay vs retained #109 smoke: decision frames equal 8/8; stop kinds equal 6; native prefixes identical 0 (cross-machine float divergence after the first collision is disclosed in the plan).

Throughput (16 workers): 12.67 s/branch amortized (single pass incl. ramp), branch wall mean 191.5 s (p90 237.5), cold-start section 11.93 s, 88.6 MB/branch.
Campaign re-projection: {'branches': 49560, 'wall_hours_at_smoke_rate': 174.4, 'wall_hours_cold_start_bound': 164.8, 'artifact_tib': 3.99, 'frozen_projection': {'wall_hours': 151.4, 'artifact_tib': 4.51}}.

Engine seconds (sum of branch walls): 32176.0.
