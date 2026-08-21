from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts.build_issue_44_runtime_evidence import _case_observation, _identity
from scripts.physics_capture_v2 import load_physics_capture_v2
from scripts.physics_capture_v2_capability_report import (
    load_physics_capture_v2_capability_report,
)
from scripts.verify_physics_player import verify_physics_player_archive


ROOT = Path(__file__).resolve().parents[1]
ISSUE_44 = ROOT / "data/runtime_evidence/issue-44"


class RuntimeEvidenceIdentityTests(unittest.TestCase):
    def test_identity_includes_plan_capture_and_source_commit_keys(self) -> None:
        first = _identity("runtime-evidence-v1", {
            "schema": "runtime_bundle_v1",
            "source_snapshot_commit": "commit-a",
            "collection_plan_identity": "plan-a",
            "captures": {"probe": {"capture_id": "capture-a"}},
        })
        changed_capture = _identity("runtime-evidence-v1", {
            "schema": "runtime_bundle_v1",
            "source_snapshot_commit": "commit-a",
            "collection_plan_identity": "plan-a",
            "captures": {"probe": {"capture_id": "capture-b"}},
        })
        changed_source = _identity("runtime-evidence-v1", {
            "schema": "runtime_bundle_v1",
            "source_snapshot_commit": "commit-b",
            "collection_plan_identity": "plan-a",
            "captures": {"probe": {"capture_id": "capture-a"}},
        })
        self.assertNotEqual(first, changed_capture)
        self.assertNotEqual(first, changed_source)
        self.assertIn("collection_plan_identity=plan-a", first)
        self.assertIn("source_snapshot_commit=commit-a", first)
        self.assertIn("capture=probe=capture-a", first)

    def test_no_contact_means_the_launched_bird_has_no_raw_contact(self) -> None:
        capture = SimpleNamespace(record={
            "events": [{
                "event_type": "bird_launched",
                "participants": ["runtime:bird:0000"],
            }],
            "fixed_step_samples": [{
                "contacts": [{
                    "entity_a_id": "runtime:block:0000",
                    "entity_b_id": "runtime:world:landscape:0000",
                }],
            }],
        })
        self.assertEqual(_case_observation("no-contact", capture), (True, None))

        capture.record["fixed_step_samples"][0]["contacts"].append({
            "entity_a_id": "runtime:bird:0000",
            "entity_b_id": "runtime:world:landscape:0000",
        })

        self.assertEqual(_case_observation("no-contact", capture), (
            False, "the launched bird produced a raw contact",
        ))

    def test_issue_44_builder_has_no_closed_issue_45_output_surface(self) -> None:
        source = (ROOT / "scripts/build_issue_44_runtime_evidence.py").read_text(encoding="utf-8")
        self.assertNotIn("ISSUE_45", source)
        self.assertNotIn("issue-45", source)
        self.assertNotIn("unity-reset", source)


class Issue44RuntimeEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        if not (ISSUE_44 / "capture-bundle-manifest.json").is_file():
            self.skipTest("issue #44 runtime evidence is not materialized")

    def test_public_capture_bundle_resolves_exact_validated_probe_bytes(self) -> None:
        bundle = json.loads((ISSUE_44 / "capture-bundle-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(set(bundle["captures"]), {
            "no-contact", "collision", "support", "support-change", "stable-terminal",
        })
        for value in bundle["captures"].values():
            path = ROOT / value["path"]
            self.assertEqual(set(value), {"capture_id", "path"})
            self.assertEqual(load_physics_capture_v2(path).capture_id, value["capture_id"])

    def test_all_six_capability_facts_are_demonstrated(self) -> None:
        report = load_physics_capture_v2_capability_report(ISSUE_44 / "capability-report.json")
        self.assertEqual(len(report.record["facts"]), 6)
        self.assertEqual(
            {fact["status"] for fact in report.record["facts"].values()},
            {"demonstrated"},
        )

    def test_runtime_bundle_uses_archive_source_commit_provenance(self) -> None:
        bundle = json.loads(
            (ISSUE_44 / "runtime-bundle-manifest.json").read_text(encoding="utf-8")
        )
        verified = verify_physics_player_archive(
            ROOT / "sciencebirdsgames/physics-v2", physics_v2=True,
        )
        self.assertEqual(bundle["source_snapshot_commit"], verified["source_snapshot_commit"])
        self.assertEqual(bundle["source_tree"], verified["source_tree"])
        self.assertEqual(
            {value["status"] for value in bundle["case_observations"].values()},
            {"demonstrated"},
        )


if __name__ == "__main__":
    unittest.main()
