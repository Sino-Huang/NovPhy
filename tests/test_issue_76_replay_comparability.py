from copy import deepcopy
import unittest

import numpy as np

from scripts.issue_76_replay_comparability import compare


def fixture():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[0] = 255
    return dict(base_cluster="training-base", scenario={"identity": "scenario"}, generation_seed=1, engine_seed=2,
        exposure_role="training", observation_configuration={}, viewport={"width_pixels": 640, "height_pixels": 480},
        decision_time=1., capture_time=1.1, transform={"world_to_camera_matrix": [1., 0.], "method": "orthographic"},
        rgb=image, initial={"world": {"world_id": "unity-physics2d", "gravity_vector": [0, -9.8]},
        "entities": [{"scenario_object_id": "bird:0000", "lifecycle": "active", "body_present": True,
                      "body": {"body_type": "dynamic", "simulated": True, "gravity_scale": 0,
                               "gravity_applicable": False, "position": [0., 1.], "velocity": [0., 0.],
                               "rotation_degrees": 0., "angular_velocity_degrees_per_second": 0.}}]})


class ReplayComparabilityTests(unittest.TestCase):
    def test_equal_evidence_reports_real_time_gap_without_claiming_hidden_identity(self):
        value = fixture()
        result = compare(value, deepcopy(value))
        self.assertTrue(result["comparable"])
        self.assertAlmostEqual(result["decision_to_capture_seconds"][0], .1)
        self.assertFalse(result["hidden_state_identity_proven"])

    def test_source_camera_body_bird_and_image_changes_are_not_ignored(self):
        original = fixture()
        for kind in ("seed", "camera", "bird", "image", "time", "inventory"):
            changed = deepcopy(original)
            if kind == "seed": changed["engine_seed"] = 3
            elif kind == "camera": changed["transform"]["world_to_camera_matrix"][0] += .01
            elif kind == "bird": changed["initial"]["entities"][0]["body"]["position"][0] += .02
            elif kind == "image": changed["rgb"][100:200] = 255
            elif kind == "time": changed["capture_time"] = .9
            else: changed["initial"]["entities"] = []
            self.assertFalse(compare(original, changed)["comparable"], kind)

    def test_small_numeric_motion_and_wrapped_rotation_use_declared_limits(self):
        original, changed = fixture(), fixture()
        changed["initial"]["entities"][0]["body"]["position"][0] = .005
        changed["initial"]["entities"][0]["body"]["rotation_degrees"] = 359.5
        self.assertTrue(compare(original, changed)["comparable"])
        changed["initial"]["entities"][0]["body"]["velocity"][0] = .02
        self.assertFalse(compare(original, changed)["comparable"])


if __name__ == "__main__":
    unittest.main()
