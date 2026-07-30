# Todo 3 Independent Gate Review

```json
{
  "AdversarialVerify": {
    "verdict": "confirmed",
    "confidence": 0.95,
    "evidence": [
      "python -m unittest tests.test_prepare_rollout_dataset: 26 tests, exit 0",
      "three Todo 3 acceptance tests: 3 tests, exit 0",
      "fresh CLI transition: opt-in scripts=[collect_train_dev_test.sh], default scripts=[collect_train_dev.sh], opt-in again scripts=[collect_train_dev_test.sh]",
      "invalid --include-test with zero test target: nonzero exit and expected validation message",
      "fresh TemporaryDirectory roots removed=true"
    ],
    "repro": "python -m unittest tests.test_prepare_rollout_dataset"
  }
}
```

The current planner leaves test unscheduled unless `--include-test` is present, removes the inactive generated command script after a same-directory mode switch, preserves source partitioning, and rejects invalid opt-in targets.
