from copy import deepcopy
import unittest

import torch

from scripts.resume_issue_76_balanced_native_norm import (
    REPAIRED,
    resume_checkpoint,
    source_repair,
)


class NormRecoveryTests(unittest.TestCase):
    def test_only_the_declared_source_repair_is_accepted(self):
        original = {"identity": "fixture", "lr": .0003,
                    "source_text": {name: "old" for name in (*REPAIRED, "unchanged.py")}}
        current = deepcopy(original)
        for name in REPAIRED:
            current["source_text"][name] = "new"
        self.assertEqual(set(source_repair(original, current)), REPAIRED)
        current["lr"] = .00003
        with self.assertRaisesRegex(ValueError, "numeric/data"):
            source_repair(original, current)
        current["lr"] = original["lr"]
        current["source_text"]["unchanged.py"] = "other"
        with self.assertRaisesRegex(ValueError, "unrelated"):
            source_repair(original, current)

    def test_resume_clears_only_failure_and_preserves_optimizer_exposure(self):
        archived = {
            "complete": False, "failure": "RuntimeError: The total norm fixture",
            "plan_identity": "fixture", "updates_requested": 12000,
            "updates_completed": 9114, "updates_applied": 9041, "updates_skipped": 73,
            "backward_scale": 2 ** -16, "model": {"weight": torch.ones(2)},
            "optimizer": {"param_groups": [{"lr": .0003}],
                          "state": {0: {"step": torch.tensor(9041.),
                                        "exp_avg": torch.ones(2)}}},
        }
        current = deepcopy(archived)
        resumed = resume_checkpoint(current, archived)
        self.assertIsNone(resumed["failure"])
        self.assertEqual(resumed["updates_completed"], 9114)
        self.assertEqual(current["failure"], archived["failure"])
        self.assertTrue(torch.equal(resumed["optimizer"]["state"][0]["exp_avg"],
                                    archived["optimizer"]["state"][0]["exp_avg"]))
        current["model"]["weight"][0] += 1
        with self.assertRaisesRegex(ValueError, "state differs"):
            resume_checkpoint(current, archived)


if __name__ == "__main__":
    unittest.main()
