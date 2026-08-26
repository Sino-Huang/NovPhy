from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.cohort_v2_statistical_protocol import (
    CohortV2ProtocolError,
    build_protocol,
    load_protocol,
    validate_protocol,
    write_protocol,
)
from scripts.run_cohort_v2_statistical_protocol import main


ROOT = Path(__file__).resolve().parents[1]


class CohortV2StatisticalProtocolTests(unittest.TestCase):
    def test_protocol_freezes_calibrated_budgets_comparators_and_decision(self):
        protocol = build_protocol(ROOT, implementation_commit="commit:fixture")
        rows = protocol["experiment_matrix"][
            "confirmatory_oracle_symbol_issue_15"
        ]["comparisons"]

        self.assertEqual(
            [(row["budget"], row["strongest_comparator_id"]) for row in rows],
            [
                (106659.08223346318, "matched_capacity_two_head"),
                (1398656.0, "fixed_pair"),
            ],
        )
        self.assertIn(
            "one-sided lower gain bound",
            protocol["statistical_analysis"]["confirmatory_decision"],
        )
        self.assertEqual(
            protocol["replicate_and_seed_policy"]["fixed_replicate_count"], 6
        )
        self.assertIn(
            "max(10 percent",
            protocol["calibration_basis"]["practical_effect_threshold"],
        )

    def test_protocol_separates_confirmatory_and_parser_stress_decisions(self):
        protocol = build_protocol(ROOT, implementation_commit="commit:fixture")
        matrix = protocol["experiment_matrix"]

        self.assertEqual(
            set(matrix) - {"configuration_freeze_rule"},
            {
                "confirmatory_oracle_symbol_issue_15",
                "learned_feature_symbol_stress_issue_16",
                "frozen_visual_symbol_stress_issue_17",
            },
        )
        self.assertIn(
            "cannot change or rescue",
            protocol["statistical_analysis"]["stress_decision"],
        )
        self.assertEqual(
            protocol["claim_boundary"]["gameplay_protocol_owner"], "issue_57"
        )
        self.assertEqual(
            matrix["learned_feature_symbol_stress_issue_16"]["comparisons"][0][
                "endpoint_bootstrap_seed"
            ],
            20260926,
        )
        self.assertEqual(
            matrix["frozen_visual_symbol_stress_issue_17"]["comparisons"][0][
                "endpoint_bootstrap_seed"
            ],
            20261026,
        )

    def test_exposure_audit_and_later_consumer_are_explicit(self):
        protocol = build_protocol(ROOT, implementation_commit="commit:fixture")

        self.assertTrue(protocol["exposure_audit"]["passed"])
        self.assertFalse(protocol["exposure_audit"]["sealed_final_bundle_opened"])
        self.assertEqual(
            protocol["final_evaluation_access"]["later_consumer_identity"],
            "issue-15-oracle-symbol-confirmatory-v1",
        )
        self.assertEqual(
            protocol["final_evaluation_access"]["current_manifest_authorization_state"],
            "pending",
        )

    def test_identity_detects_protocol_changes(self):
        protocol = build_protocol(ROOT, implementation_commit="commit:fixture")
        changed = json.loads(json.dumps(protocol))
        changed["replicate_and_seed_policy"]["fixed_replicate_count"] = 7

        with self.assertRaisesRegex(CohortV2ProtocolError, "identity is stale"):
            validate_protocol(changed)

    def test_write_is_immutable_and_validate_cli_rebuilds_exactly(self):
        protocol = build_protocol(ROOT, implementation_commit="commit:fixture")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "protocol.json"
            write_protocol(protocol, output)
            self.assertEqual(load_protocol(output), protocol)
            with self.assertRaisesRegex(CohortV2ProtocolError, "already exists"):
                write_protocol(protocol, output)
            self.assertEqual(
                main([
                    "--repository-root",
                    str(ROOT),
                    "--output",
                    str(output),
                    "--validate",
                ]),
                0,
            )

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "protocol.json"
            self.assertEqual(
                main([
                    "--repository-root",
                    str(ROOT),
                    "--output",
                    str(output),
                    "--dry-run",
                ]),
                0,
            )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
