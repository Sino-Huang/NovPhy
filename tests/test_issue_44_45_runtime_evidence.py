from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import unittest

from scripts.cohort_v2_scenarios import validate_deterministic_scenario_receipt
from scripts.physics_capture_v2 import (
    load_physics_capture_v2,
    normalized_initial_engine_state_identity,
)
from scripts.physics_capture_v2_capability_report import (
    load_physics_capture_v2_capability_report,
)


ROOT = Path(__file__).resolve().parents[1]
ISSUE_44 = ROOT / ".claude/project-docs/evidence/issue-44-physics-v2"
ISSUE_45 = ROOT / ".claude/project-docs/evidence/issue-45-cohort-v2-lineage"


class IssueRuntimeEvidenceTests(unittest.TestCase):
    def test_public_capture_bundle_resolves_exact_validated_probe_bytes(self) -> None:
        bundle = json.loads((ISSUE_44 / "capture-bundle-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(set(bundle["captures"]), {
            "no-contact", "collision", "support", "support-change", "stable-terminal",
        })
        for value in bundle["captures"].values():
            path = ROOT / value["path"]
            self.assertEqual(sha256(path.read_bytes()).hexdigest(), value["sha256"])
            self.assertEqual(load_physics_capture_v2(path).capture_id, value["capture_id"])

    def test_capability_report_preserves_the_actual_unavailable_facts(self) -> None:
        report = load_physics_capture_v2_capability_report(ISSUE_44 / "capability-report.json")
        facts = report.record["facts"]
        self.assertEqual(facts["configured_fixed_step_capture_stride"]["status"], "demonstrated")
        self.assertEqual(facts["complete_raw_non_trigger_contacts"]["status"], "demonstrated")
        self.assertEqual(facts["causal_identity_source_bindings"]["status"], "demonstrated")
        self.assertEqual(facts["final_frame_covers_termination"]["status"], "demonstrated")
        self.assertEqual(facts["collider_geometry_and_separation"]["status"], "unavailable")
        self.assertEqual(facts["gravity_body_lifecycle_motion_support_world"]["status"], "unavailable")

    def test_training_unity_reset_receipt_resolves_two_equal_initial_states(self) -> None:
        receipt = json.loads(
            (ISSUE_45 / "receipts/training-unity-reset.json").read_text(encoding="utf-8")
        )
        validate_deterministic_scenario_receipt(receipt)
        bundle = json.loads((ISSUE_44 / "capture-bundle-manifest.json").read_text(encoding="utf-8"))
        first = load_physics_capture_v2(ROOT / bundle["captures"]["no-contact"]["path"])
        second = load_physics_capture_v2(ROOT / bundle["captures"]["collision"]["path"])
        self.assertEqual(receipt["first_capture_sha256"], bundle["captures"]["no-contact"]["sha256"])
        self.assertEqual(receipt["second_capture_sha256"], bundle["captures"]["collision"]["sha256"])
        self.assertEqual(
            normalized_initial_engine_state_identity(first),
            normalized_initial_engine_state_identity(second),
        )
        self.assertEqual(
            normalized_initial_engine_state_identity(first),
            receipt["normalized_initial_engine_state_identity"],
        )


if __name__ == "__main__":
    unittest.main()
