# Issues

## 2026-07-05 Start Work Context
- LSP diagnostics attempted for `scripts/collect_rollouts.py` and `tests/test_collect_rollouts.py`; unavailable because `basedpyright-langserver` is not installed.
- `.omo/boulder.json` currently contains two active work entries for the same plan/session; continuing with `active_work_id` selected by the hook because both entries point to the same plan.

## 2026-07-05 Search Agent Issues
- Search result system reminders listed broad pre-existing workspace changes; research agents were read/search only and did not complete any plan checkbox. Do not mark plan checkboxes for search-only background tasks.
- `fresh_engine_attempt` is not currently copied by `write_action_logs()`, which may matter for later retry/quarantine tasks.
