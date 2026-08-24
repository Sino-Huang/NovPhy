import unittest

from scripts.run_cohort_v2_pair_measurements import issue_7_compute_calibration
from scripts.run_cohort_v2_trajectory_labels import issue_8_cost_spec


class Issue8TrajectoryLabelTests(unittest.TestCase):
    def test_primary_teacher_objective_uses_declared_unit_weights_and_compute_reference(self):
        calibration = issue_7_compute_calibration()
        spec = issue_8_cost_spec(calibration)

        self.assertEqual(spec.physical_violation_weight, 1.0)
        self.assertEqual(spec.compute_weight, 1.0)
        self.assertEqual(
            spec.compute_reference,
            calibration.transition_per_decision,
        )
        self.assertIn("cohort-v2-trajectory-cost-v1", spec.identity)


if __name__ == "__main__":
    unittest.main()
