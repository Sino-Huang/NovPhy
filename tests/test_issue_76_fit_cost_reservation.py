from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import issue_76_reserve_fit_cost as reservation


class EngineeringFitCostTests(unittest.TestCase):
    def test_actual_category_costs_are_debited_symmetrically_without_new_allowance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "data/issue-76-native-refit-synthetic-smoke/attempt-001.json"
            value = {"device": "cuda", "research_fits": 0, "budget_records": {
                "common-1": {"active_seconds": 3}, "predictor-hybrid-1": {"active_seconds": 2},
                "predictor-pure-1": {"active_seconds": 1}, "controller-hybrid-1": {"active_seconds": 6}}}
            reservation.files.write(path, value)
            with patch.object(reservation.files, "ROOT", root):
                report = reservation.reservation({"identity": "fixture", "seeds": [1, 2, 3]})
            self.assertEqual(report["actual_category_seconds"], {"common": 3., "predictor": 3., "controller": 6.})
            self.assertEqual(report["initial_seconds_per_budget"], {"common": 1., "predictor": .5, "controller": 1.})


if __name__ == "__main__":
    unittest.main()
