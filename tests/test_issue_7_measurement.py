import unittest

from scripts.run_cohort_v2_pair_measurements import issue_7_compute_calibration


class Issue7MeasurementTests(unittest.TestCase):
    def test_issue_6_checkpoint_calibration_charges_the_executed_model_paths(self) -> None:
        calibration = issue_7_compute_calibration()

        self.assertEqual(calibration.unit, "multiply_accumulate")
        self.assertEqual(calibration.controller_per_decision, 0.0)
        self.assertEqual(calibration.continuous_adapter_per_decision, 0.0)
        self.assertEqual(calibration.micro_adapter_per_decision, 768.0)
        self.assertEqual(calibration.macro_adapter_per_decision, 1536.0)
        self.assertEqual(calibration.micro_graph_per_contact, 147456.0)
        self.assertEqual(calibration.micro_graph_per_support, 294912.0)
        self.assertEqual(calibration.transition_per_decision, 1398656.0)
        self.assertEqual(calibration.continuous_readout_per_decision, 0.0)
        self.assertEqual(calibration.micro_readout_per_decision, 74496.0)
        self.assertEqual(calibration.macro_readout_per_decision, 75648.0)
        self.assertEqual(calibration.shared_initial_perception_per_rollout, 0.0)


if __name__ == "__main__":
    unittest.main()
