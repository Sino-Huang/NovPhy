from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from scripts import issue_76_event_validation as validation


class EventValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.member = dict(identity="event", base_cluster="base", generation_seed=1, engine_seed=2,
            exposure_role="calibration", scenario={"scenario_manifest": {"scenario_lineage": {"identity": "lineage"}}},
            generated_slots=["bird:0000"], actions=[{"drag_x": -80, "drag_y": 10, "release_time_ms": 600, "tap_time_ms": 0}])
        self.plan = {"identity": "execution", "inventory": {"limits": {"native_steps_per_shot_max": 30000,
                                                                       "rgb_frames_per_shot_max": 601}}}
        self.snapshot_root = self.root / "attempts/event/decision-1"
        self.snapshot_root.mkdir(parents=True)
        image = Image.new("RGB", (640, 480))
        image.putpixel((0, 0), (255, 255, 255))
        image.save(self.snapshot_root / "agent.png")
        frame = {"fixed_step": 1, "fixed_time_seconds": 1., "agent_observation": {"relative_path": "agent.png"},
                 "capture_metadata": {"viewport": {}, "world_to_observation_transform": {}}}
        self.snapshot = {"identity": "snapshot", "exposure_role": "calibration", "scenario_lineage_identity": "derived",
                         "source_bindings": {"source_scenario_lineage_identity": "lineage", "rollout_identity": "event:shot-1"},
                         "observation_configuration": {}, "frame_records": [frame]}
        self.captured = deepcopy(self.snapshot)
        self.captured.update(identity="captured", frame_records=[dict(frame, fixed_step=2, fixed_time_seconds=1.1)])
        summary = {"sample_count": 1, "observed_window_valid": True}
        self.result = {"member_identity": "event", "base_cluster": "base", "exposure_role": "calibration", "engine_seed": 2,
            "complete": True, "failure": None, "attempted_shots": [1],
            "ports": dict(zip(("agent", "game", "physics"), validation.worker_ports(3))),
            "decisions": [{"action": self.member["actions"][0], "trained_model_used": False, "candidates_scored": 0,
                           "observation_manifest": "snapshot"}],
            "segments": [{"native_root": str(self.root / "attempts/event/aligned/capture"), "capture_id": "capture",
                          "summary": summary, "observation_manifest": "captured",
                          "action": {"coordinate_frame": "slingshot_relative", "drag_release": [-80, 10],
                                     "release_time": 600, "tap_time": 0}}],
            "policy": {"decision_time": 1., "executed_action_time": 1.1, "executed_actions": 1,
                       "decisions": 1, "observations": 2}}
        self.receipt = {"execution_within_limits": True, "worker_exitcode": 0, "worker_slot": 3,
                        "execution_plan_identity": "execution", "member_identity": "event"}
        initial = {"entities": [{"scenario_object_id": "bird:0000"}]}
        self.trace = SimpleNamespace(summary=summary, manifest={"engine_seed": 2, "capture_id": "capture",
                                                               "frame_records": [{"fixed_step": 2}]},
            trace=SimpleNamespace(chunks=lambda: iter([{"fixed_step_samples": [initial]}])))

    def read_case(self):
        validation.inventory.files.write(self.root / "results/event.json", self.result)
        validation.inventory.files.write(self.root / "supervision/event.json", self.receipt)
        with patch.object(validation, "validate_observation_trace", side_effect=[self.snapshot, self.captured]), patch.object(
                validation, "NativeSegmentTrace", return_value=self.trace):
            return validation.read_case(self.root, self.plan, self.member)

    def test_all_existing_roles_are_preserved_without_relabeling(self):
        for role in validation.inventory.ROLES:
            for value in (self.member, self.result, self.snapshot, self.captured):
                value["exposure_role"] = role
            case = self.read_case()
            self.assertEqual(case["exposure_role"], role)
            self.assertEqual((case["decision_time"], case["capture_time"]), (1., 1.1))
            self.assertEqual(case["rgb"].shape, (480, 640, 3))

    def test_wrong_observation_role_or_rollout_is_rejected(self):
        for key, value in (("exposure_role", "training"), ("rollout_identity", "other:shot-1")):
            original = deepcopy(self.captured)
            if key == "exposure_role":
                self.captured[key] = value
            else:
                self.captured["source_bindings"][key] = value
            with self.assertRaises(ValueError):
                self.read_case()
            self.captured = original

    def test_wrong_executed_action_and_worker_ports_are_rejected(self):
        original = deepcopy(self.result)
        self.result["segments"][0]["action"]["drag_release"] = [-10, 80]
        with self.assertRaises(ValueError):
            self.read_case()
        self.result = original
        self.result["ports"]["physics"] += 1
        with self.assertRaises(ValueError):
            self.read_case()

    def test_noncausal_decision_and_invalid_supervision_are_rejected(self):
        self.snapshot["frame_records"][0]["fixed_time_seconds"] = 2.
        self.result["policy"]["decision_time"] = 2.
        with self.assertRaises(ValueError):
            self.read_case()
        self.receipt["execution_within_limits"] = False
        with self.assertRaises(ValueError):
            self.read_case()

    def test_native_clock_mismatch_and_constant_decision_image_are_rejected(self):
        self.trace.manifest["frame_records"] = [{"fixed_step": 3}]
        with self.assertRaises(ValueError):
            self.read_case()
        self.trace.manifest["frame_records"] = [{"fixed_step": 2}]
        Image.new("RGB", (640, 480)).save(self.snapshot_root / "agent.png")
        with self.assertRaises(ValueError):
            self.read_case()
