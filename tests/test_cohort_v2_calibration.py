from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from world_model.training import (
    CohortV2CalibrationError,
    CohortV2CalibrationRecord,
    CohortV2StressGapRecord,
    analyze_cohort_v2_calibration,
    validate_cohort_v2_calibration,
    write_cohort_v2_calibration,
)


class CohortV2CalibrationTests(unittest.TestCase):
    def _records(self):
        values = {
            "candidate": ((0.7, 0.8), (0.6, 0.7)),
            "fixed": ((1.0, 1.2), (0.9, 1.0)),
            "two_head": ((1.1, 1.3), (1.0, 1.1)),
        }
        records = []
        for configuration, (calibration, selection) in values.items():
            for role, errors in (
                ("calibration", calibration),
                ("model_selection", selection),
            ):
                for index, error in enumerate(errors):
                    records.append(CohortV2CalibrationRecord(
                        configuration_id=configuration,
                        exposure_role=role,
                        attempt_id=f"{role}:{index}",
                        scenario_lineage_identity=f"lineage:{role}",
                        coverage_stratum=f"stratum:{index}",
                        checkpoint_identity=f"checkpoint:{configuration}",
                        seed=10,
                        state_count=4,
                        mean_endpoint_prediction_error=error,
                        mean_endpoint_violation_rate=0.01 * index,
                        mean_policy_compute_per_simulated_frame=100.0 + index,
                        mean_full_compute_per_simulated_frame=120.0 + index,
                    ))
        stress = tuple(
            CohortV2StressGapRecord(
                stress_id="symbol-removal",
                exposure_role="calibration",
                attempt_id=f"calibration:{index}",
                scenario_lineage_identity="lineage:calibration",
                coverage_stratum=f"stratum:{index}",
                reference_configuration_id="candidate",
                stressed_configuration_id="no-symbol",
                metric="endpoint_error",
                degradation_gap=0.2 + index * 0.1,
            )
            for index in range(2)
        )
        return tuple(records), stress

    def test_selects_comparator_before_calibration_and_records_insufficient_work(self):
        records, stress = self._records()
        result = analyze_cohort_v2_calibration(
            records,
            stress,
            candidate_configuration_id="candidate",
            eligible_comparator_ids=("fixed", "two_head"),
            source_bindings={"release_identity": "release:fixture"},
            missing_integrations=("train integrated checkpoint",),
            downstream_work=("parser stress after issue #15",),
            bootstrap_seed=4,
            bootstrap_replicates=100,
        )

        self.assertEqual(
            result["proposals_for_issue_34"]["strongest_comparator_id"], "fixed"
        )
        self.assertEqual(result["independent_calibration_replicates"], 2)
        self.assertEqual(result["disposition"]["status"], "insufficient_evidence")
        self.assertFalse(result["exposure_audit"]["final_evaluation_artifacts_accessed"])
        self.assertEqual(
            result["endpoint_scope"]["evaluation_mode"],
            "teacher_forced_local_successor_prediction",
        )
        self.assertFalse(
            result["endpoint_scope"]["recursive_fixed_h1_accumulation_assessed"]
        )
        self.assertEqual(result["downstream_work"], ["parser stress after issue #15"])

    def test_artifact_recomputes_and_detects_changed_metrics(self):
        records, stress = self._records()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "calibration"
            manifest = write_cohort_v2_calibration(
                root,
                records,
                stress,
                candidate_configuration_id="candidate",
                eligible_comparator_ids=("fixed", "two_head"),
                source_bindings={"release_identity": "release:fixture"},
                missing_integrations=(),
                implementation_revision="implementation:fixture",
                bootstrap_seed=4,
                bootstrap_replicates=100,
            )

            self.assertEqual(validate_cohort_v2_calibration(root), manifest)
            path = root / "replicate_metrics.jsonl"
            path.write_bytes(path.read_bytes().replace(b'"state_count":4', b'"state_count":5', 1))
            with self.assertRaisesRegex(CohortV2CalibrationError, "identity"):
                validate_cohort_v2_calibration(root)


if __name__ == "__main__":
    unittest.main()
