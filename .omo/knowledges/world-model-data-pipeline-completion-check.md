2026-07-31: `world-model-data-pipeline.md` completion check

- The plan checklist marks todos 1-9 and final verification items F1-F4 complete.
- Use `/home/sukai/miniconda3/bin/python` for verification; the default `python` resolves to the `novphy` env and currently lacks `torch`.
- Verified command: `/home/sukai/miniconda3/bin/python -m unittest tests.test_world_model_data tests.test_prepare_rollout_dataset` -> 113 tests passed.
- Verified command: `/home/sukai/miniconda3/bin/python -m compileall -q world_model tests/test_world_model_data.py scripts/prepare_rollout_dataset.py scripts/rollout_artifacts.py scripts/rollout_validation_types.py` -> exit 0.
- Verified command: `/home/sukai/miniconda3/bin/python -m world_model.data.inspect --root data/novphy_rollouts_dataset_20260708_171531 --splits train dev --json` -> exit 0; train reports 4975 accepted episodes / 59700 shots and dev reports 463 accepted episodes / 5556 shots.
- Caveat: Momus rejected the saved plan as an executable plan artifact because Task 2 still cites stale `scripts/prepare_rollout_dataset.py` line ranges after the canonical validator moved to `scripts/rollout_artifacts.py`. This is a documentation/reference issue in the plan, not a failing implementation gate.

2026-07-31 update: Installed `world_model/requirements.txt` into the env activated by `source ~/cd_novphy` using `uv pip install -r world_model/requirements.txt`. That env now imports `torch` and `Pillow`, and `python -m unittest tests.test_world_model_data tests.test_prepare_rollout_dataset` passes there with 113 tests.
