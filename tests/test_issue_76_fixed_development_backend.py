from copy import deepcopy
import unittest

import torch

from scripts import score_issue_76_fixed_development as scoring
from scripts import validate_issue_76_fixed_development as independent
from scripts.validate_issue_76_fixed_development_backend import SourceBackendTransitions
from world_model.training.native_history_model import NativeHistoryDynamics


class SourceBackendVerificationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(76)

    def test_source_backend_transport_preserves_independent_checks(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            model = NativeHistoryDynamics(pure=True, width=16).to(device).eval()
            members = scoring.synthetic_members(2, 76)
            cell = {"recipe": "synthetic", "seed": 1, "pure": True}
            plan = {"identity": "execution", "batch_size": 2,
                    "independent_validation": {"metric_relative_tolerance": .001, "metric_absolute_tolerance": .00001}}
            result = scoring.score_cell(model, cell, members, device, batch_size=2, check_resources=lambda: None)
            result.update(execution_plan_identity="execution", role="synthetic-not-research")
            transport = SourceBackendTransitions(model)
            receipt = independent.validate_cell(plan, cell, result, members, transport, lambda: None)
            self.assertEqual(receipt["curve_points_validated"], 30)
            self.assertEqual(receipt["batches_validated"], 3)
            changed = deepcopy(result)
            policy = next(iter(changed["rows"][0]["policies"]))
            metric = changed["rows"][0]["policies"][policy]["curves"]["11250"]["recursive"]
            metric["memory_mse"] = 2 * metric["memory_mse"] + 1
            with self.assertRaises(ValueError):
                independent.validate_cell(plan, cell, changed, members, transport, lambda: None)
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
