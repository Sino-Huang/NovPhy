from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from scripts import validate_issue_76_action_replay as validation


class ReplayValidationTests(unittest.TestCase):
    def test_saved_predecision_image_and_native_clock_bindings_are_used(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            member = dict(identity="replay", base_cluster="base", generation_seed=1, engine_seed=2,
                exposure_role="training", scenario={"scenario_manifest": {"scenario_lineage": {"identity": "lineage"}}},
                generated_slots=["bird:0000"], actions=[{"drag_x": -80, "drag_y": 10}])
            plan = {"identity": "execution", "inventory": {"limits": {"native_steps_per_shot_max": 30000,
                                                                        "rgb_frames_per_shot_max": 601}}}
            snapshot_root = root / "attempts/replay/decision-1"
            snapshot_root.mkdir(parents=True)
            image = Image.new("RGB", (640, 480))
            image.putpixel((0, 0), (255, 255, 255))
            image.save(snapshot_root / "agent.png")
            frame = {"fixed_step": 1, "fixed_time_seconds": 1., "agent_observation": {"relative_path": "agent.png"},
                     "capture_metadata": {"viewport": {}, "world_to_observation_transform": {}}}
            snapshot = {"identity": "snapshot", "exposure_role": "training", "scenario_lineage_identity": "lineage",
                        "observation_configuration": {}, "frame_records": [frame]}
            captured = dict(snapshot, identity="captured", frame_records=[dict(frame, fixed_step=2, fixed_time_seconds=1.1)])
            summary = {"sample_count": 1}
            result = {"member_identity": "replay", "base_cluster": "base", "exposure_role": "training", "engine_seed": 2,
                "complete": True, "failure": None, "attempted_shots": [1],
                "decisions": [{"action": member["actions"][0], "trained_model_used": False, "candidates_scored": 0,
                               "observation_manifest": "snapshot"}],
                "segments": [{"native_root": "unused", "summary": summary, "observation_manifest": "captured"}],
                "policy": {"decision_time": 1., "executed_action_time": 1.1, "executed_actions": 1, "decisions": 1, "observations": 2}}
            supervision = {"execution_within_limits": True, "worker_exitcode": 0,
                           "execution_plan_identity": "execution", "member_identity": "replay"}
            validation.run.files.write(root / "results/replay.json", result)
            validation.run.files.write(root / "supervision/replay.json", supervision)
            initial = {"entities": [{"scenario_object_id": "bird:0000"}]}
            trace = SimpleNamespace(summary=summary, manifest={"engine_seed": 2, "frame_records": [{"fixed_step": 2}]},
                                    trace=SimpleNamespace(chunks=lambda: iter([{"fixed_step_samples": [initial]}])))
            with patch.object(validation, "validate_observation_trace", side_effect=[snapshot, captured]), \
                    patch.object(validation, "NativeSegmentTrace", return_value=trace):
                case = validation.read_case(root, plan, member)
            self.assertEqual(case["decision_time"], 1.)
            self.assertEqual(case["capture_time"], 1.1)
            self.assertEqual(case["rgb"].shape, (480, 640, 3))
            self.assertEqual(case["decision_image_path"], str(snapshot_root / "agent.png"))
            supervision["execution_within_limits"] = False
            validation.run.files.write(root / "supervision/replay.json", supervision)
            with self.assertRaises(ValueError):
                validation.read_case(root, plan, member)


if __name__ == "__main__":
    unittest.main()
