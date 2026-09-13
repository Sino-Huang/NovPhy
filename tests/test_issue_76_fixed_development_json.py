import json
import unittest
from unittest.mock import patch

from scripts import run_issue_76_fixed_development as run
from tests.test_issue_76_fixed_development_scoring import selection_fixture


class FixedDevelopmentJSONTests(unittest.TestCase):
    def test_real_result_loader_restores_frozen_pair_order_after_sorted_json(self):
        inventory, results = selection_fixture()
        expected = run.scoring.calibration_choice(inventory, results)
        plan = {"identity": "execution", "inventory": inventory}
        complete = {"execution_plan_identity": "execution", "role": "calibration",
                    "cells": [run.cell_name(cell) for cell in inventory["models"]]}
        saved = json.loads(json.dumps(results, sort_keys=True))
        with patch.object(run.fit.files, "read", side_effect=[complete, *saved]), \
                patch.object(run.fit, "require_finished_budget", return_value={}):
            loaded = run.load_results(plan, "calibration")
        # Values must be byte-serialization equivalent; only map ordering changes.
        self.assertEqual(json.dumps(loaded, sort_keys=True), json.dumps(results, sort_keys=True))
        self.assertEqual(run.scoring.calibration_choice(inventory, loaded), expected)

    def test_loading_must_not_reconstruct_a_missing_policy(self):
        inventory, results = selection_fixture()
        plan = {"identity": "execution", "inventory": inventory}
        complete = {"execution_plan_identity": "execution", "role": "calibration",
                    "cells": [run.cell_name(cell) for cell in inventory["models"]]}
        saved = json.loads(json.dumps(results, sort_keys=True))
        saved[0]["rows"][0]["policies"].pop(next(iter(saved[0]["rows"][0]["policies"])))
        with patch.object(run.fit.files, "read", side_effect=[complete, *saved]), \
                patch.object(run.fit, "require_finished_budget", return_value={}):
            with self.assertRaises(ValueError):
                run.load_results(plan, "calibration")


if __name__ == "__main__":
    unittest.main()
