from __future__ import annotations

import copy
import unittest

from scripts.physics_capture_v2_capability_report import (
    PhysicsCaptureV2CapabilityReportError,
    validate_physics_capture_v2_capability_report,
)


SHA = "a" * 64


def report() -> dict:
    cases = ("no-contact", "collision", "support", "support-change", "stable-terminal")
    probes = [
        {"source": "unity_exporter_probe", "case": case, "capture_id": f"capture-{index}", "capture_sha256": SHA, "scenario_lineage_id": f"lineage-{index % 2}", "level_instance_id": f"level-{index % 2}", "scenario_template_id": f"template-{index % 2}", "final_evaluation": False}
        for index, case in enumerate(cases)
    ]
    facts = {
        fact: {"status": "demonstrated", "capture_sha256": SHA, "reason": None}
        for fact in (
            "configured_fixed_step_capture_stride", "complete_raw_non_trigger_contacts",
            "collider_geometry_and_separation", "gravity_body_lifecycle_motion_support_world",
            "causal_identity_source_bindings", "final_frame_covers_termination",
        )
    }
    return {"schema_version": "physics_capture_v2_exporter_capability_report_v1", "report_id": "report-1", "provenance": {"engine_sha256": SHA, "player_sha256": SHA, "protocol_sha256": SHA, "exporter_code_sha256": SHA}, "probes": probes, "facts": facts}


class PhysicsCaptureV2CapabilityReportTests(unittest.TestCase):
    def test_accepts_a_complete_actual_probe_accounting_report(self) -> None:
        parsed = validate_physics_capture_v2_capability_report(report())
        self.assertEqual(parsed.report_id, "report-1")

    def test_rejects_fixture_reports_and_incomplete_probe_coverage(self) -> None:
        invalid = report()
        invalid["probes"][0]["source"] = "fixture"
        with self.assertRaisesRegex(PhysicsCaptureV2CapabilityReportError, "fixture"):
            validate_physics_capture_v2_capability_report(invalid)
        invalid = report()
        invalid["probes"] = invalid["probes"][:-1]
        with self.assertRaisesRegex(PhysicsCaptureV2CapabilityReportError, "probe cases are incomplete"):
            validate_physics_capture_v2_capability_report(invalid)

    def test_preserves_explicit_unavailability(self) -> None:
        unavailable = report()
        unavailable["facts"]["final_frame_covers_termination"] = {"status": "unavailable", "capture_sha256": None, "reason": "probe did not retain terminal frame"}
        self.assertEqual(
            validate_physics_capture_v2_capability_report(unavailable).record["facts"]["final_frame_covers_termination"]["status"],
            "unavailable",
        )
        invalid = copy.deepcopy(unavailable)
        invalid["facts"]["final_frame_covers_termination"]["reason"] = None
        with self.assertRaisesRegex(PhysicsCaptureV2CapabilityReportError, "explicit reason"):
            validate_physics_capture_v2_capability_report(invalid)

    def test_rejects_demonstrated_fact_without_a_validated_probe_digest(self) -> None:
        invalid = report()
        invalid["facts"]["final_frame_covers_termination"]["capture_sha256"] = "b" * 64
        with self.assertRaisesRegex(PhysicsCaptureV2CapabilityReportError, "validated Unity exporter probe"):
            validate_physics_capture_v2_capability_report(invalid)
