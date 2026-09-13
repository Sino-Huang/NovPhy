from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import score_issue_76_fixed_development as score
from scripts import validate_issue_76_fixed_development as validate
from world_model.training.native_history_data import VISUAL_DIM, VOCABULARY
from world_model.training.native_history_model import NativeHistoryDynamics, STATE_DIM


SPEC = {"metric_relative_tolerance": .001, "metric_absolute_tolerance": .00001}


class FixedDevelopmentValidationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(76)

    def test_independent_field_formulas_and_undefined_centers(self):
        predicted, target = np.zeros(STATE_DIM), np.zeros(STATE_DIM)
        presence = np.zeros(len(VOCABULARY))
        centers = np.full((len(VOCABULARY), 2), np.nan)
        presence[10] = 1
        centers[10] = [.25, .5]
        predicted[2 + 13 * 10] = .75
        predicted[7 + 13 * 10] = .75
        predicted[8 + 13 * 10] = 1.
        predicted[VISUAL_DIM:] = 3
        actual = validate.metrics(predicted, target, presence, centers)
        self.assertEqual(actual["center_mse_to_engine"], .25)
        self.assertEqual(actual["pig_count_absolute_error"], .25)
        self.assertEqual(actual["memory_mse"], 9.)
        self.assertEqual(actual["carrier_absolute_bound_excess"], 1.)
        empty = validate.metrics(target, target, presence * 0, centers)
        self.assertIsNone(empty["center_mse_to_engine"])
        self.assertEqual(validate.metrics(predicted, target, presence, centers, failed=True)["carrier_mse"], 1e9)
        altered = dict(actual, carrier_mse=actual["carrier_mse"] * 2)
        with self.assertRaises(ValueError):
            validate.compare_metrics(altered, actual, SPEC)

    def test_full_independent_curve_check_detects_numerical_and_work_errors(self):
        model = NativeHistoryDynamics(pure=True, width=16).eval()
        members = score.synthetic_members(2, 76)
        cell = {"recipe": "synthetic", "seed": 1, "pure": True}
        plan = {"identity": "execution", "batch_size": 2, "independent_validation": SPEC}
        result = score.score_cell(model, cell, members, "cpu", batch_size=2, check_resources=lambda: None)
        result.update(execution_plan_identity="execution", role="synthetic-not-research")
        receipt = validate.validate_cell(plan, cell, result, members, model, lambda: None)
        self.assertEqual(receipt["curve_points_validated"], 30)
        self.assertEqual(receipt["batches_validated"], 3)
        policy = next(iter(result["rows"][0]["policies"]))
        for kind in ("numeric", "work", "availability", "failure"):
            changed = deepcopy(result)
            record = changed["rows"][0]["policies"][policy]
            if kind == "numeric":
                metric = record["curves"]["11250"]["recursive"]
                metric["memory_mse"] = metric["memory_mse"] * 2 + 1
            elif kind == "work":
                changed["batches"][0]["local_transitions"] += 1
            elif kind == "availability":
                record["curves"]["11250"]["available"] = False
            else:
                record["prediction_failure"] = True
            with self.assertRaises(ValueError):
                validate.validate_cell(plan, cell, changed, members, model, lambda: None)

    def test_independent_preparation_check_rejects_borrowed_target_or_context(self):
        entry = dict(member_identity="synthetic", exposure_role="calibration", usable=True)
        shard = {"segment_ranges": [{"start": 0, "stop": 3, "action": torch.zeros(5)}],
                 "tensors": {"fixed_steps": torch.tensor([100, 850, 1600]),
                             "presence": torch.zeros(3, len(VOCABULARY)),
                             "centers": torch.zeros(3, len(VOCABULARY), 2)}}
        carrier = torch.randn(3, STATE_DIM)
        member = dict(entry=entry, initial=carrier[0], action=shard["segment_ranges"][0]["action"],
            targets={750: dict(carrier=carrier[1], presence=shard["tensors"]["presence"][1], centers=shard["tensors"]["centers"][1]),
                     1500: dict(carrier=carrier[2], presence=shard["tensors"]["presence"][2], centers=shard["tensors"]["centers"][2])},
            contexts={50: {}, 250: {}, 750: {750: carrier[0], 1500: carrier[1]}})
        plan = {"inventory": {"entries": [entry], "data_plan_path": "/synthetic/plan.json", "data_root": "/synthetic"}}
        with patch.object(validate.run.fit.files, "read", return_value={}), \
                patch.object(validate.run.fit, "load_shard", return_value=shard), \
                patch.object(validate.run.fit, "load_carrier", return_value=carrier):
            validate.validate_prepared(plan, "calibration", 1, [member], lambda: None)
            changed = deepcopy(member)
            changed["targets"][11250] = changed["targets"][1500]
            with self.assertRaises(ValueError):
                validate.validate_prepared(plan, "calibration", 1, [changed], lambda: None)
            changed = deepcopy(member)
            changed["contexts"][50][750] = carrier[0]
            with self.assertRaises(ValueError):
                validate.validate_prepared(plan, "calibration", 1, [changed], lambda: None)


if __name__ == "__main__":
    unittest.main()
