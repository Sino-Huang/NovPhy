from __future__ import annotations

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.webui.physics_v2_review import (
    PhysicsV2ReviewSession,
    coverage_verdict,
)
from tests.test_physics_capture_v2 import capture


def engine_capture() -> dict:
    value = capture()
    value.pop("source_bindings")
    value["schema_version"] = "physics_capture_v2_engine_v1"
    return value


def collision_capture() -> dict:
    value = engine_capture()
    value["events"].insert(
        0,
        {
            "event_id": "collision-1",
            "event_type": "collision",
            "fixed_step": 1,
            "participants": ["bird:1", "world:static:1"],
            "payload": {"relative_speed": 1.0},
        },
    )
    return value


def supported_capture() -> dict:
    value = engine_capture()
    second = value["fixed_step_samples"][1]
    first = value["fixed_step_samples"][0]
    first["contacts"] = copy.deepcopy(second["contacts"])
    first["supports"] = copy.deepcopy(second["supports"])
    first["entities"] = copy.deepcopy(second["entities"])
    value["minimum_contact_separation"]["fixed_step"] = 0
    return value


def write_probe_plan(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    scenarios = []
    for index in range(2):
        manifest_path = root / f"scenario-{index}.json"
        manifest_path.write_text(
            json.dumps({"identity": f"scenario-manifest-v1:{index}"}),
            encoding="utf-8",
        )
        scenarios.append({
            "scenario_id": f"scenario-{index}",
            "scenario_manifest_reference": str(manifest_path),
            "scenario_template_identity": f"scenario-template-v1:{index}",
            "level_instance_identity": f"level-instance-v1:{index}",
            "scenario_lineage_identity": f"scenario-lineage-v1:{index}",
        })
    plan_path = root / "probe-plan.json"
    plan_path.write_text(
        json.dumps({"identity": "physics-v2-probe-plan-v1:review", "scenarios": scenarios}),
        encoding="utf-8",
    )
    return plan_path


class PhysicsV2CoverageVerdictTests(unittest.TestCase):
    def test_verdicts_use_authoritative_contacts_and_support_sets(self) -> None:
        self.assertFalse(coverage_verdict("collision", engine_capture())["demonstrated"])
        self.assertTrue(coverage_verdict("collision", collision_capture())["demonstrated"])
        self.assertFalse(coverage_verdict("persistent support", engine_capture())["demonstrated"])
        self.assertTrue(coverage_verdict("persistent support", supported_capture())["demonstrated"])
        self.assertTrue(coverage_verdict("support change", engine_capture())["demonstrated"])


class PhysicsV2ReviewSessionTests(unittest.TestCase):
    def test_unavailable_diagnostic_cannot_be_frozen_as_confirmatory_evidence(self) -> None:
        with TemporaryDirectory() as temporary:
            session = PhysicsV2ReviewSession(
                Path(temporary),
                probe_plan_path=write_probe_plan(Path(temporary) / "stage"),
            )
            session.stage("collision", {
                "action_type": "drag_hold_release",
                "coordinate_frame": "slingshot_relative",
                "drag_start": [97, 227],
                "drag_release": [-80, 8],
                "tapTime": 0,
                "holdTime": 1000,
                "frame_height": 480,
            })
            session.begin_exploration()
            session.complete_exploration(engine_capture())

            with self.assertRaisesRegex(ValueError, "demonstrated diagnostic pilot"):
                session.freeze_replay()

    def test_diagnostic_capture_freezes_exact_action_before_one_replay(self) -> None:
        with TemporaryDirectory() as temporary:
            session = PhysicsV2ReviewSession(
                Path(temporary),
                probe_plan_path=write_probe_plan(Path(temporary) / "stage"),
            )
            action = {
                "action_type": "drag_hold_release",
                "coordinate_frame": "slingshot_relative",
                "drag_start": [97, 227],
                "drag_release": [-80, 8],
                "tapTime": 0,
                "holdTime": 1000,
                "frame_height": 480,
            }

            staged = session.stage("collision", action)
            self.assertEqual(staged["state"], "staged")
            self.assertEqual(session.begin_exploration()["socket_command"], {
                "x": 17,
                "y": 260,
                "tapTime": 0,
                "releaseTime": 1000,
            })

            explored = session.complete_exploration(collision_capture())
            self.assertEqual(explored["state"], "explored")
            self.assertTrue(explored["verdict"]["demonstrated"])
            self.assertFalse(explored["eligible_for_issue_44"])

            frozen = session.freeze_replay()
            plan_path = Path(frozen["replay_plan_path"])
            frozen_bytes = plan_path.read_bytes()
            plan = json.loads(frozen_bytes)
            self.assertEqual(plan["action"], staged["action"])
            self.assertEqual(plan["max_attempts"], 1)
            self.assertEqual(plan["selection_provenance"], {
                "kind": "diagnostic_pilot",
                "capture_path": "diagnostic/engine-envelope.json",
            })
            self.assertFalse(plan["diagnostic_capture_eligible"])
            self.assertEqual(session.begin_replay()["socket_command"], staged["socket_command"])

            completed = session.complete_replay(collision_capture())
            self.assertEqual(completed["state"], "demonstrated")
            self.assertTrue(completed["eligible_for_issue_44_review"])
            self.assertEqual(plan_path.read_bytes(), frozen_bytes)
            self.assertTrue((Path(temporary) / session.session_id / "accepted" / "physics_capture_v2.json").is_file())

            with self.assertRaisesRegex(ValueError, "one confirmatory replay"):
                session.begin_replay()

    def test_unavailable_replay_is_quarantined(self) -> None:
        with TemporaryDirectory() as temporary:
            session = PhysicsV2ReviewSession(
                Path(temporary),
                probe_plan_path=write_probe_plan(Path(temporary) / "stage"),
            )
            session.stage("collision", {
                "action_type": "drag_hold_release",
                "coordinate_frame": "slingshot_relative",
                "drag_start": [97, 227],
                "drag_release": [-80, 8],
                "tapTime": 0,
                "holdTime": 1000,
                "frame_height": 480,
            })
            session.begin_exploration()
            session.complete_exploration(collision_capture())
            session.freeze_replay()
            session.begin_replay()

            result = session.complete_replay(engine_capture())

            self.assertEqual(result["state"], "unavailable")
            self.assertFalse(result["eligible_for_issue_44_review"])
            self.assertTrue((Path(temporary) / session.session_id / "quarantine" / "physics_capture_v2.json").is_file())


if __name__ == "__main__":
    unittest.main()
