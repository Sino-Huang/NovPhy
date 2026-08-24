from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_issue_55_evidence import (
    AUDIT_NAME,
    BUNDLE_NAME,
    REVIEW_NAME,
    Issue55EvidenceError,
    build_audit,
    build_independent_review,
    build_issue_55_evidence,
    validate_issue_55_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


class Issue55EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = build_audit()

    def test_primary_audit_passes_every_capability_and_section_16_condition(self) -> None:
        self.assertTrue(self.audit["passed"])
        self.assertEqual(len(self.audit["capability_matrix"]), 13)
        self.assertTrue(all(row["disposition"] == "PASS" for row in self.audit["capability_matrix"]))
        self.assertEqual(
            [row["disposition"] for row in self.audit["section_16_matrix"]],
            ["PASS"] * 7,
        )
        self.assertEqual(self.audit["authorities"]["production"]["accepted_rollouts"], 24)
        self.assertEqual(set(self.audit["authorities"]["label_floors"]), {
            "contact", "supports", "steady-state", "structure-unstable",
            "excess_penetration", "unsupported_stationary_or_floating_body",
        })

    def test_independent_review_rejects_a_promoted_failure(self) -> None:
        mutated = deepcopy(self.audit)
        mutated["section_16_matrix"][4]["disposition"] = "PARTIAL"
        with self.assertRaisesRegex(Issue55EvidenceError, "PASS matrices"):
            build_independent_review(mutated)

    def test_builder_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "issue-55"
            result = build_issue_55_evidence(output, dry_run=True)
            self.assertTrue(result["passed"])
            self.assertFalse(output.exists())

    def test_published_bundle_revalidates_exactly(self) -> None:
        output = ROOT / "data/runtime_evidence/issue-55"
        if not output.exists():
            self.skipTest("immutable issue-55 publication has not been built")
        result = validate_issue_55_evidence(output)
        self.assertTrue(result["passed"])
        self.assertEqual(
            sorted(path.name for path in output.iterdir()),
            sorted((AUDIT_NAME, REVIEW_NAME, BUNDLE_NAME)),
        )
        for name in (AUDIT_NAME, REVIEW_NAME, BUNDLE_NAME):
            self.assertIsInstance(json.loads((output / name).read_text(encoding="utf-8")), dict)


if __name__ == "__main__":
    unittest.main()
