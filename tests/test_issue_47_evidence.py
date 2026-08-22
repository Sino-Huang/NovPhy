from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_issue_47_evidence import (
    build_issue_47_evidence,
    validate_issue_47_evidence,
)


class Issue47EvidenceTests(unittest.TestCase):
    def test_published_issue_47_bundle_is_current(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = validate_issue_47_evidence(
            repository,
            repository / "data/runtime_evidence/issue-47",
        )
        self.assertTrue(
            result["partition_manifest_identity"].startswith(
                "cohort-v2-partition-exposure-manifest-v1:1:"
            )
        )

    def test_real_planned_identities_and_required_mutations_are_audited(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "issue-47"
            result = build_issue_47_evidence(repository, root)

            self.assertEqual(validate_issue_47_evidence(repository, root), result)
            partition = json.loads(
                (root / "partition-exposure-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [entry["exposure_role"] for entry in partition["entries"]],
                ["training", "calibration", "model_selection", "final_evaluation"],
            )
            self.assertEqual(
                {entry["lineage_quota"] for entry in partition["entries"]},
                {1},
            )
            self.assertEqual(
                partition["entries"][-1]["sealed_scenario_manifest_reference"],
                "sealed-final-evaluation-v1:issue-45",
            )

            leakage = json.loads(
                (root / "lineage-template-leakage-audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(leakage["real_manifest_audit"]["passed"])
            self.assertEqual(
                {case["mutation"] for case in leakage["mutation_results"]},
                {
                    "missing_role",
                    "duplicate_lineage",
                    "held_out_level_instance_reuse",
                    "unknown_lineage",
                    "replay_role_leak",
                    "derivation_role_leak",
                    "observation_variant_role_leak",
                    "undeclared_artifact_provenance",
                },
            )
            self.assertTrue(
                all(case["rejected"] for case in leakage["mutation_results"])
            )

            access = json.loads(
                (root / "representative-access-audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(
                all(report["passed"] for report in access["ordinary_workflows"])
            )
            self.assertTrue(access["preauthorization_final_access"]["rejected"])


if __name__ == "__main__":
    unittest.main()
