# World-Model Data Pipeline Todo 9 Boundary

Todo 9 completion is validated by `WorldModelDataIntegrationTests`, not by the active rollout root alone. The synthetic fixture is the source of truth for train/dev/test plan-backed source disjointness, no-plan provenance unavailability, duplicate source-key rejection, partial artifact exclusion, and stable resumed ablation manifests.

The active root inspection command is a read-only health check:

```bash
/home/sukai/miniconda3/bin/python -m world_model.data.inspect --root data/novphy_rollouts_dataset_20260708_171531 --splits train dev --json
```

Because the inspector does not accept a collection plan, it must report novelty/scenario composition as unavailable and must not be cited as source-level leakage validation.
