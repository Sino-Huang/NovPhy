from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.issue_59_aligned_observation_collection import dry_run


class Issue59AlignedObservationCollectionTests(unittest.TestCase):
    def test_dry_run_wires_all_roles_without_final_access_or_writes(self) -> None:
        with patch(
            "scripts.issue_59_aligned_observation_collection._player",
            return_value={"source_snapshot_commit": "commit-59"},
        ):
            result = dry_run(implementation_commit="commit-59")

        self.assertEqual(result["planned_rollouts"], 24)
        self.assertEqual(
            result["planned_role_counts"],
            {
                "training": 6,
                "calibration": 6,
                "model_selection": 6,
                "final_evaluation": 6,
            },
        )
        self.assertFalse(result["final_outcomes_accessed"])
        self.assertFalse(result["files_written"])
        self.assertIn("python -u -m", result["actual_command"])


if __name__ == "__main__":
    unittest.main()
