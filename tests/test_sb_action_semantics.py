import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sciencebirdsagents"))

from SBEnvironment.action_utils import normalize_release_action


class PointLike:
    def __init__(self, x, y):
        self.X = x
        self.Y = y


class ReleaseActionSemanticsTests(unittest.TestCase):
    def test_legacy_two_value_sequence_defaults_tap_time(self):
        self.assertEqual(normalize_release_action([-100, 50]), (-100, 50, 0))

    def test_legacy_three_value_sequence_preserves_tap_time(self):
        self.assertEqual(normalize_release_action((-120, 45, 70)), (-120, 45, 70))

    def test_legacy_long_sequence_uses_first_two_values_and_default_tap_time(self):
        self.assertEqual(normalize_release_action([-120, 45, 70, 99]), (-120, 45, 0))

    def test_point_like_action_uses_existing_relative_release_semantics(self):
        self.assertEqual(normalize_release_action(PointLike(-80, 30)), (-80, 30, 0))

    def test_dict_drag_release_defaults_to_slingshot_relative(self):
        action = {"action_type": "drag_release", "drag_release": [-95, 40], "tap_time": 65}

        self.assertEqual(normalize_release_action(action), (-95, 40, 65))

    def test_dict_release_alias_is_supported(self):
        self.assertEqual(normalize_release_action({"release": [-50, 20]}), (-50, 20, 0))

    def test_dict_absolute_release_converts_from_game_coordinates(self):
        action = {"drag_release": [120, 260], "coordinate_frame": "absolute", "tapTime": 72}

        self.assertEqual(normalize_release_action(action, sling_center=PointLike(200, 300)), (-80, 40, 72))

    def test_absolute_release_requires_sling_center(self):
        with self.assertRaisesRegex(ValueError, "sling_center"):
            normalize_release_action({"release": [120, 260], "coordinate_frame": "absolute"})

    def test_missing_release_point_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "release"):
            normalize_release_action({"action_type": "drag_release", "tap_time": 10})

    def test_unknown_coordinate_frame_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "coordinate_frame"):
            normalize_release_action({"release": [1, 2], "coordinate_frame": "screen"})

    def test_environment_wrappers_use_release_action_normalizer(self):
        paths = [
            ROOT / "sciencebirdsagents" / "SBEnvironment" / "SBEnvironmentWrapper.py",
            ROOT / "sciencebirdsagents" / "SBEnvironment" / "SBEnvironmentWrapperOpenAI.py",
            ROOT / "modules" / "benchmark" / "sciencebirdsagents" / "SBEnvironment" / "SBEnvironmentWrapper.py",
            ROOT / "modules" / "benchmark" / "sciencebirdsagents" / "SBEnvironment" / "SBEnvironmentWrapperOpenAI.py",
        ]

        for path in paths:
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                self.assertIn("from SBEnvironment.action_utils import normalize_release_action", source)
                self.assertIn("normalize_release_action(action, sling_center=self.sling_center)", source)

    def test_openai_wrappers_accept_dict_actions_before_action_space_contains(self):
        paths = [
            ROOT / "sciencebirdsagents" / "SBEnvironment" / "SBEnvironmentWrapperOpenAI.py",
            ROOT / "modules" / "benchmark" / "sciencebirdsagents" / "SBEnvironment" / "SBEnvironmentWrapperOpenAI.py",
        ]

        for path in paths:
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                self.assertIn("if not isinstance(action, dict):\n            assert self.action_space.contains(action)", source)
                self.assertIn("if isinstance(action, dict):\n            dx, dy, tap_time", source)

    def test_benchmark_helper_matches_top_level_helper(self):
        top_level = ROOT / "sciencebirdsagents" / "SBEnvironment" / "action_utils.py"
        benchmark = ROOT / "modules" / "benchmark" / "sciencebirdsagents" / "SBEnvironment" / "action_utils.py"

        self.assertEqual(top_level.read_text(encoding="utf-8"), benchmark.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
