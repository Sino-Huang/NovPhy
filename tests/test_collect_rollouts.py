import io
import json
import signal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from scripts.collect_rollouts import (
    PRE_DRAG_OVERLAY_TEXT,
    _action_guide_points,
    _launch_guide_points,
    action_to_shot,
    capture_desktop_rollout,
    collect_fresh_engine_rollouts,
    collect_rollouts,
    build_parser,
    format_action_overlay_text,
    load_actions_from_action_log,
    prepare_rollout_video_frames,
    slingshot_reference_point_from_symbolic_state,
    main,
    RolloutCollectionError,
    select_level_in_display,
    stop_owned_engine,
    validate_rollout_artifact,
    write_action_plan,
)


class FakeBridge:
    def __init__(self):
        self.shots = []
        self.frame_index = 0
        self.symbolic_state = None

    def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
        self.shots.append((x, y, tap_time, fast, release_time))
        return 1

    def configure(self, agent_id, mode):
        self.configured = (agent_id, mode)
        return (0, 0, 1)

    def set_speed(self, speed):
        self.speed = speed
        return 1

    def disconnect(self):
        self.disconnected = True

    def screenshot(self):
        class Screenshot:
            width = 4
            height = 3

            def __init__(self, frame_index):
                value = 50 + frame_index
                self.rgb = bytes(channel for pixel in range(4 * 3) for channel in (value + pixel, 20, 30))

        screenshot = Screenshot(self.frame_index)
        self.frame_index += 1
        return screenshot

    def get_game_state(self):
        from src.webui.bridge import GameState

        return GameState.PLAYING

    def get_current_score(self):
        return 200 + self.frame_index

    def get_current_level(self):
        return 7

    def get_symbolic_state_without_screenshot(self):
        return self.symbolic_state


class CollectRolloutsTest(unittest.TestCase):
    def test_action_to_shot_matches_webui_game_coordinates_then_flips_once_for_bridge(self):
        action = {
            "coordinate_frame": "slingshot_relative",
            "drag_start": [100, 200],
            "drag_release": [30, 50],
            "tapTime": 70,
            "holdTime": 600,
        }

        shot = action_to_shot(action, frame_height=480)

        self.assertEqual(shot["gameX"], 130)
        self.assertEqual(shot["gameY"], 150)
        self.assertEqual(shot["x"], 130)
        self.assertEqual(shot["y"], 329)
        self.assertEqual(shot["releaseTime"], 600)

    def test_action_guide_endpoint_matches_actual_bridge_shot_pixel(self):
        action = {"coordinate_frame": "slingshot_relative", "drag_start": [100, 200], "drag_release": [30, 50]}
        shot = action_to_shot(action, frame_height=480)

        start, end = _action_guide_points(action, shot, image_height=480)

        self.assertEqual(start, (100, 279))
        self.assertEqual(end, (shot["x"], shot["y"]))

    def test_launch_guide_points_opposite_of_pull_vector_for_rightward_shots(self):
        action = {"coordinate_frame": "slingshot_relative", "drag_start": [300, 220], "drag_release": [-50, 40]}
        shot = action_to_shot(action, frame_height=480)

        start, release_end = _action_guide_points(action, shot, image_height=480)
        launch_start, launch_end = _launch_guide_points(action, image_height=480)

        self.assertEqual(start, (300, 259))
        self.assertEqual(release_end, (shot["x"], shot["y"]))
        self.assertEqual(launch_start, start)
        self.assertGreater(launch_end[0], launch_start[0])
        self.assertLess(launch_end[1], launch_start[1])

    def test_action_to_shot_uses_height_minus_one_y_boundaries(self):
        top = action_to_shot({"coordinate_frame": "absolute", "release": [10, 479]}, frame_height=480)
        bottom = action_to_shot({"coordinate_frame": "absolute", "release": [10, 0]}, frame_height=480)

        self.assertEqual(top["y"], 0)
        self.assertEqual(bottom["y"], 479)

    def test_slingshot_reference_point_from_symbolic_state_uses_request_62_geojson_vertices(self):
        symbolic_state = [
            {
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"label": "Slingshot"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[10, 20], [10, 40], [30, 40], [30, 20]]],
                        },
                    }
                ]
            }
        ]

        reference = slingshot_reference_point_from_symbolic_state(symbolic_state, frame_height=100)

        self.assertEqual(reference, {"gameX": 19, "gameY": 72, "canvasX": 19, "canvasY": 27})

    def test_collect_rollouts_writes_manifest_and_per_shot_metadata(self):
        actions = [
            {
                "action_type": "drag_hold_release",
                "coordinate_frame": "slingshot_relative",
                "drag_start": [300, 220],
                "drag_release": [-50, 40],
                "tapTime": 70,
                "holdTime": 120,
            },
            {
                "action_type": "drag_hold_release",
                "coordinate_frame": "slingshot_relative",
                "drag_start": [300, 220],
                "drag_release": [-80, -20],
                "tapTime": 0,
            },
        ]
        now = [5.0]

        def clock():
            return now[0]

        def sleeper(seconds):
            now[0] += seconds

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                FakeBridge(),
                Path(tmp),
                actions,
                target_fps=2,
                duration_seconds=1.0,
                frame_height=480,
                max_frames=2,
                clock=clock,
                sleeper=sleeper,
            )

            self.assertEqual(manifest["attempt_count"], 2)
            self.assertEqual(manifest["accepted_rollout_count"], 0)
            self.assertEqual(manifest["rollout_count"], 0)
            self.assertEqual(manifest["rollouts"][0]["shoot_response"], 1)
            self.assertEqual(manifest["rollouts"][0]["frame_count"], 2)
            self.assertTrue((Path(tmp) / "shot_001" / "metadata.json").is_file())
            saved_manifest = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_manifest["attempt_count"], 2)
            self.assertEqual(saved_manifest["accepted_rollout_count"], 0)
            self.assertEqual(saved_manifest["rollout_count"], 0)

    def test_collect_rollouts_records_protocol_state_evidence(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]
        events = []

        class TrackingBridge(FakeBridge):
            def __init__(self):
                super().__init__()
                self.phase = "pre-shot"

            def get_game_state(self):
                events.append("state")
                from src.webui.bridge import GameState

                return GameState.PLAYING

            def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
                events.append("shoot")
                self.phase = "after-shoot"
                return super().shoot(x, y, tap_time=tap_time, fast=fast, release_time=release_time)

        bridge = TrackingBridge()

        def pre_shot_grabber():
            events.append("baseline")
            bridge.phase = "after-baseline"
            from PIL import Image

            image = Image.new("RGB", (20, 20), (5, 5, 5))
            image.putpixel((0, 0), (6, 5, 5))
            return image

        def capture_rollout(bridge, output_dir, **kwargs):
            from PIL import Image

            events.append("capture")
            bridge.phase = "after-capture"
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 20), (50, 60, 70)).save(frames_dir / "frame_000000.png", format="PNG")
            Image.new("RGB", (20, 20), (10, 20, 30)).save(frames_dir / "frame_000001.png", format="PNG")
            metadata = {
                "frame_count": 2,
                "frames_dir": str(frames_dir),
            }
            (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            return metadata

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                bridge,
                Path(tmp),
                actions,
                target_fps=1,
                duration_seconds=1,
                pre_shot_grabber=pre_shot_grabber,
                capture_rollout=capture_rollout,
                video_runner=lambda command, check, stdout, stderr: Path(command[-1]).write_bytes(b"mp4"),
            )
            metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["rollouts"][0]["shoot_response"], 1)
        self.assertEqual(events[0], "state")
        self.assertLess(events.index("baseline"), events.index("shoot"))
        self.assertLess(events.index("shoot"), events.index("capture"))
        self.assertEqual(metadata["pre_shot_protocol_state"]["game_state"], "PLAYING")
        self.assertEqual(metadata["pre_shot_protocol_state"]["current_level"], 7)
        self.assertEqual(metadata["post_recovery_protocol_state"]["game_state"], "PLAYING")
        self.assertEqual(metadata["post_shoot_protocol_state"]["game_state"], "PLAYING")
        self.assertEqual(metadata["post_capture_protocol_state"]["game_state"], "PLAYING")
        self.assertEqual(metadata["artifact_validation"]["classification"], "gameplay-valid")
        self.assertTrue(metadata["artifact_validation"]["accepted"])
        self.assertIn("recovery_action", metadata)

    def test_invalid_rollout_records_state_and_reason(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]
        events = []

        class TrackingBridge(FakeBridge):
            def __init__(self):
                super().__init__()
                self.phase = "pre-shot"

            def get_game_state(self):
                events.append("state")
                from src.webui.bridge import GameState

                return GameState.PLAYING

            def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
                events.append("shoot")
                self.phase = "after-shoot"
                return super().shoot(x, y, tap_time=tap_time, fast=fast, release_time=release_time)

        bridge = TrackingBridge()

        def baseline_image():
            from PIL import Image

            image = Image.new("RGB", (20, 20), (50, 60, 70))
            image.putpixel((0, 0), (51, 60, 70))
            return image

        def pre_shot_grabber():
            events.append("baseline")
            bridge.phase = "after-baseline"

            return baseline_image()

        def capture_rollout(bridge, output_dir, **kwargs):
            from PIL import Image

            events.append("capture")
            bridge.phase = "after-capture"
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            baseline_image().save(frames_dir / "frame_000000.png", format="PNG")
            metadata = {
                "frame_count": 1,
                "frames_dir": str(frames_dir),
            }
            (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            return metadata

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                bridge,
                Path(tmp),
                actions,
                target_fps=1,
                duration_seconds=1,
                pre_shot_grabber=pre_shot_grabber,
                capture_rollout=capture_rollout,
                video_runner=lambda command, check, stdout, stderr: Path(command[-1]).write_bytes(b"mp4"),
            )
            metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["attempt_count"], 1)
        self.assertEqual(manifest["accepted_rollout_count"], 0)
        self.assertEqual(manifest["rollout_count"], 0)
        self.assertFalse(manifest["rollouts"][0]["accepted"])
        self.assertEqual(events[0], "state")
        self.assertLess(events.index("baseline"), events.index("shoot"))
        self.assertLess(events.index("shoot"), events.index("capture"))
        self.assertFalse(metadata["artifact_validation"]["accepted"])
        self.assertEqual(metadata["artifact_validation"]["classification"], "no-frame-motion")
        self.assertEqual(metadata["artifact_validation"]["invalid_reason"], "no_frame_motion")
        self.assertFalse(metadata["artifact_validation"]["retryable"])
        self.assertEqual(metadata["artifact_validation"]["retry_decision"], "quarantine")
        self.assertIn("pre_shot_protocol_state", metadata)
        self.assertIn("post_shoot_protocol_state", metadata)
        self.assertIn("post_capture_protocol_state", metadata)
        self.assertIn("post_recovery_protocol_state", metadata)

    def test_post_shot_gate_rejects_menu_capture(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]

        def menu_like_capture(bridge, output_dir, **kwargs):
            from PIL import Image

            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 20), (245, 245, 245)).save(frames_dir / "frame_000000.png", format="PNG")
            metadata = {
                "frame_count": 1,
                "frames_dir": str(frames_dir),
            }
            (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            return metadata

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                FakeBridge(),
                Path(tmp),
                actions,
                target_fps=1,
                duration_seconds=1,
                capture_rollout=menu_like_capture,
                video_runner=lambda command, check, stdout, stderr: Path(command[-1]).write_bytes(b"mp4"),
            )
            metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))
            action_log = json.loads((Path(tmp) / "action_log.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["attempt_count"], 1)
        self.assertEqual(manifest["accepted_rollout_count"], 0)
        self.assertEqual(manifest["rollout_count"], 0)
        self.assertFalse(metadata["artifact_validation"]["accepted"])
        self.assertEqual(metadata["artifact_validation"]["classification"], "menu-detected")
        self.assertEqual(metadata["artifact_validation"]["invalid_reason"], "menu_detected")
        self.assertFalse(metadata["artifact_validation"]["retryable"])
        self.assertEqual(metadata["artifact_validation"]["retry_decision"], "quarantine")
        self.assertEqual(action_log["attempt_count"], 1)
        self.assertEqual(action_log["accepted_trial_count"], 0)
        self.assertEqual(action_log["trial_count"], 1)

    def test_post_shot_gate_flags_low_motion_capture(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]

        def low_motion_capture(bridge, output_dir, **kwargs):
            from PIL import Image

            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            first_frame = Image.new("RGB", (20, 20), (50, 60, 70))
            second_frame = Image.new("RGB", (20, 20), (50, 60, 70))
            for x in range(2):
                for y in range(11):
                    second_frame.putpixel((x, y), (51, 60, 70))
            first_frame.save(frames_dir / "frame_000000.png", format="PNG")
            second_frame.save(frames_dir / "frame_000001.png", format="PNG")
            metadata = {
                "frame_count": 2,
                "frames_dir": str(frames_dir),
            }
            (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            return metadata

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                FakeBridge(),
                Path(tmp),
                actions,
                target_fps=1,
                duration_seconds=1,
                capture_rollout=low_motion_capture,
                video_runner=lambda command, check, stdout, stderr: Path(command[-1]).write_bytes(b"mp4"),
            )
            metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))
            action_log = json.loads((Path(tmp) / "action_log.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["attempt_count"], 1)
        self.assertEqual(manifest["accepted_rollout_count"], 0)
        self.assertEqual(manifest["rollout_count"], 0)
        self.assertFalse(metadata["artifact_validation"]["accepted"])
        self.assertEqual(metadata["artifact_validation"]["classification"], "low-motion-suspicious")
        self.assertEqual(metadata["artifact_validation"]["invalid_reason"], "low_motion_suspicious")
        self.assertTrue(metadata["artifact_validation"]["retryable"])
        self.assertEqual(metadata["artifact_validation"]["retry_decision"], "retry")
        self.assertEqual(action_log["attempt_count"], 1)
        self.assertEqual(action_log["accepted_trial_count"], 0)
        self.assertEqual(action_log["trial_count"], 1)

    def test_pre_shot_guard_rejects_menu_surface_even_when_protocol_playing(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]
        events = []

        class MenuSurfaceBridge(FakeBridge):
            def get_game_state(self):
                from src.webui.bridge import GameState

                return GameState.PLAYING

            def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
                events.append("shoot")
                return super().shoot(x, y, tap_time=tap_time, fast=fast, release_time=release_time)

        def menu_pre_shot_grabber():
            events.append("baseline")
            from PIL import Image

            image = Image.new("RGB", (40, 30), (245, 245, 245))
            image.putpixel((3, 3), (230, 40, 40))
            image.putpixel((4, 3), (40, 130, 230))
            return image

        def capture_rollout(bridge, output_dir, **kwargs):
            events.append("capture")
            raise AssertionError("collector must not capture after a menu-like pre-shot surface")

        bridge = MenuSurfaceBridge()
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "recovery_failed"):
                collect_rollouts(
                    bridge,
                    Path(tmp),
                    actions,
                    target_fps=1,
                    duration_seconds=1,
                    pre_shot_grabber=menu_pre_shot_grabber,
                    capture_rollout=capture_rollout,
                )
            metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(events, ["baseline"])
        self.assertEqual(bridge.shots, [])
        self.assertEqual(metadata["pre_shot_guard"]["status"], "recovery_failed")
        self.assertEqual(metadata["pre_shot_guard"]["invalid_reason"], "menu_like_pre_shot")
        self.assertEqual(metadata["pre_shot_guard"]["protocol_state"]["game_state"], "PLAYING")
        self.assertTrue(metadata["pre_shot_guard"]["visual_evidence"]["menu_like"])
        self.assertIn("post_recovery_protocol_state", metadata)
        self.assertEqual(metadata["recovery_action"], None)

    def test_pre_shot_guard_rejects_menu_inside_default_desktop_crop(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]
        events = []

        class DesktopMenuBridge(FakeBridge):
            def get_game_state(self):
                from src.webui.bridge import GameState

                return GameState.PLAYING

            def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
                events.append("shoot")
                return super().shoot(x, y, tap_time=tap_time, fast=fast, release_time=release_time)

        def desktop_pre_shot_grabber():
            events.append("baseline")
            from PIL import Image

            image = Image.new("RGB", (1024, 768), (0, 0, 0))
            for x in range(32, 672):
                for y in range(64, 544):
                    image.putpixel((x, y), (245, 245, 245))
            image.putpixel((35, 67), (230, 40, 40))
            image.putpixel((36, 67), (40, 130, 230))
            return image

        def capture_rollout(bridge, output_dir, **kwargs):
            events.append("capture")
            raise AssertionError("collector must not capture when the cropped game viewport is menu-like")

        bridge = DesktopMenuBridge()
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "recovery_failed"):
                collect_rollouts(
                    bridge,
                    Path(tmp),
                    actions,
                    target_fps=1,
                    duration_seconds=1,
                    pre_shot_grabber=desktop_pre_shot_grabber,
                    capture_rollout=capture_rollout,
                )
            metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(events, ["baseline"])
        self.assertEqual(bridge.shots, [])
        visual_evidence = metadata["pre_shot_guard"]["visual_evidence"]
        self.assertEqual(metadata["pre_shot_guard"]["invalid_reason"], "menu_like_pre_shot")
        self.assertTrue(visual_evidence["menu_like"])
        self.assertEqual(visual_evidence["width"], 640)
        self.assertEqual(visual_evidence["height"], 480)

    def test_pre_shot_guard_recovers_new_trial_before_shooting(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]
        events = []

        class NewTrialBridge(FakeBridge):
            def __init__(self):
                super().__init__()
                self.state_names = ["NEWTRIAL", "PLAYING", "PLAYING", "PLAYING", "PLAYING"]
                self.ready_calls = 0

            def get_game_state(self):
                events.append("state")
                from src.webui.bridge import GameState

                state_name = self.state_names.pop(0) if self.state_names else "PLAYING"
                return getattr(GameState, state_name)

            def ready_for_new_set(self):
                events.append("ready_for_new_set")
                self.ready_calls += 1
                return (1, 0, 0, 0, 0, 0, 0)

            def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
                events.append("shoot")
                return super().shoot(x, y, tap_time=tap_time, fast=fast, release_time=release_time)

        def gameplay_pre_shot_grabber():
            events.append("baseline")
            from PIL import Image

            image = Image.new("RGB", (40, 30), (20, 30, 40))
            image.putpixel((3, 3), (80, 90, 100))
            image.putpixel((12, 8), (150, 90, 40))
            return image

        def capture_rollout(bridge, output_dir, **kwargs):
            from PIL import Image

            events.append("capture")
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (40, 30), (50, 60, 70)).save(frames_dir / "frame_000000.png", format="PNG")
            Image.new("RGB", (40, 30), (90, 100, 110)).save(frames_dir / "frame_000001.png", format="PNG")
            return {"frame_count": 2, "frames_dir": str(frames_dir)}

        bridge = NewTrialBridge()
        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                bridge,
                Path(tmp),
                actions,
                target_fps=1,
                duration_seconds=1,
                pre_shot_grabber=gameplay_pre_shot_grabber,
                capture_rollout=capture_rollout,
                video_runner=lambda command, check, stdout, stderr: Path(command[-1]).write_bytes(b"mp4"),
            )
            metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))

        self.assertLess(events.index("ready_for_new_set"), events.index("shoot"))
        self.assertLess(events.index("shoot"), events.index("capture"))
        self.assertEqual(bridge.ready_calls, 1)
        self.assertEqual(len(bridge.shots), 1)
        self.assertEqual(manifest["rollouts"][0]["shoot_response"], 1)
        self.assertEqual(metadata["pre_shot_protocol_state"]["game_state"], "NEWTRIAL")
        self.assertEqual(metadata["post_recovery_protocol_state"]["game_state"], "PLAYING")
        self.assertEqual(metadata["recovery_action"], "ready_for_new_set")
        self.assertEqual(metadata["pre_shot_guard"]["status"], "accepted_after_recovery")
        self.assertEqual(metadata["pre_shot_guard"]["recovery_attempts"], 1)
        self.assertEqual(metadata["pre_shot_guard"]["invalid_reason"], None)
        self.assertTrue(metadata["artifact_validation"]["accepted"])

    def test_collect_rollouts_can_reset_before_each_action_and_use_custom_capture(self):
        actions = [
            {"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0},
            {"coordinate_frame": "absolute", "release": [240, 250], "tapTime": 45},
        ]
        reset_calls = []
        capture_calls = []

        def reset_rollout(index, action):
            reset_calls.append((index, action["tapTime"]))

        def capture_rollout(bridge, output_dir, **kwargs):
            capture_calls.append((output_dir.name, kwargs["action"]["tapTime"]))
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "metadata.json").write_text(json.dumps({"frame_count": 1}), encoding="utf-8")
            return {"frame_count": 1}

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                FakeBridge(),
                Path(tmp),
                actions,
                target_fps=1,
                duration_seconds=1,
                reset_rollout=reset_rollout,
                capture_rollout=capture_rollout,
            )

        self.assertEqual(reset_calls, [(1, 0), (2, 45)])
        self.assertEqual(capture_calls, [("shot_001", 0), ("shot_002", 45)])
        self.assertEqual(manifest["attempt_count"], 2)
        self.assertEqual(manifest["accepted_rollout_count"], 0)
        self.assertEqual(manifest["rollout_count"], 0)

    def test_collect_rollouts_anchors_same_episode_actions_to_symbolic_slingshot(self):
        bridge = FakeBridge()
        bridge.symbolic_state = [
            {
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"label": "Slingshot"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[80, 230], [80, 270], [120, 270], [120, 230]]],
                        },
                    }
                ]
            }
        ]
        actions = [
            {
                "action_type": "drag_hold_release",
                "coordinate_frame": "slingshot_relative",
                "drag_start": [300, 220],
                "drag_release": [-45, 4],
                "tapTime": 0,
                "holdTime": 600,
            }
        ]

        def capture_rollout(bridge, output_dir, **kwargs):
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True)
            from PIL import Image

            Image.new("RGB", (160, 90), (50, 60, 70)).save(frames_dir / "frame_000000.png", format="PNG")
            return {"frame_count": 1, "frames_dir": str(frames_dir)}

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                bridge,
                Path(tmp),
                actions,
                target_fps=1,
                duration_seconds=1,
                frame_height=480,
                capture_rollout=capture_rollout,
                video_runner=lambda command, check, stdout, stderr: Path(command[-1]).write_bytes(b"mp4"),
            )

        anchored_action = manifest["rollouts"][0]["action"]
        self.assertEqual(anchored_action["drag_start"], [98, 235])
        self.assertEqual(manifest["rollouts"][0]["slingshot_reference"], {"gameX": 98, "gameY": 235, "canvasX": 98, "canvasY": 244})
        self.assertEqual(bridge.shots[0], (53, 248, 0, True, 600))

    def test_capture_desktop_rollout_records_imagegrab_frames(self):
        class Grabber:
            def __init__(self):
                self.calls = 0

            def grab(self):
                from PIL import Image

                self.calls += 1
                image = Image.new("RGB", (4, 3), (10, 20, 30))
                image.putpixel((0, 0), (70, 80, 90))
                if self.calls == 1:
                    image.putpixel((1, 0), (100, 110, 120))
                else:
                    image.putpixel((1, 0), (100, 110, 120))
                    image.putpixel((2, 0), (130, 140, 150))
                return image

        from PIL import Image

        baseline = Image.new("RGB", (4, 3), (10, 20, 30))
        baseline.putpixel((0, 0), (70, 80, 90))
        now = [2.0]

        def clock():
            return now[0]

        def sleeper(seconds):
            now[0] += seconds

        with TemporaryDirectory() as tmp:
            metadata = capture_desktop_rollout(
                FakeBridge(),
                Path(tmp),
                target_fps=2,
                duration_seconds=1,
                max_frames=2,
                action={"tapTime": 0},
                pre_shot_image=baseline,
                pre_shot_sample={"state": "PLAYING", "score": 222},
                grabber=Grabber(),
                clock=clock,
                sleeper=sleeper,
            )

            self.assertEqual(metadata["capture_source"], "desktop-imagegrab")
            self.assertEqual(metadata["frame_count"], 2)
            self.assertEqual(metadata["pre_shot_path"], str(Path(tmp) / "pre_shot.png"))
            self.assertFalse(metadata["frames"][0]["uniform"])
            self.assertIsNone(metadata["frames"][0]["frame_delta"])
            self.assertEqual(metadata["frames"][0]["pre_shot_delta"]["changed_pixel_count"], 1)
            self.assertEqual(metadata["frames"][1]["frame_delta"]["changed_pixel_count"], 1)
            self.assertEqual(metadata["frames"][1]["pre_shot_delta"]["changed_pixel_count"], 2)
            self.assertEqual(metadata["max_pre_shot_delta"], 2)
            self.assertEqual(metadata["max_pre_shot_delta_bbox"], [1, 0, 3, 1])
            self.assertEqual(metadata["pre_shot_sample"], {"state": "PLAYING", "score": 222})
            self.assertTrue((Path(tmp) / "pre_shot.png").is_file())
            self.assertTrue((Path(tmp) / "frames" / "frame_000000.png").is_file())

    def test_known_dataset_artifacts_classify_gameplay_and_reported_menu_shots(self):
        valid_artifact_root = Path(
            "data/novphy_rollouts_dataset/train/novelty_level_0_type010101_00001_0_1_010101_0_1"
        )
        reported_menu_artifact_root = Path(
            "data/novphy_rollouts_dataset/train/novelty_level_0_type010101_00002_0_1_010101_0_1"
        )

        valid = validate_rollout_artifact(valid_artifact_root / "shot_001")
        menu_static = validate_rollout_artifact(reported_menu_artifact_root / "shot_001")

        self.assertTrue(valid["accepted"])
        self.assertEqual(valid["classification"], "gameplay-valid")
        self.assertEqual(valid["max_frame_delta"], 965)
        self.assertEqual(valid["score"], 1770)
        self.assertIn("gameplay-valid", valid["signals"])

        self.assertFalse(menu_static["accepted"])
        self.assertEqual(menu_static["classification"], "menu-detected")
        self.assertEqual(menu_static["invalid_reason"], "menu_detected")
        self.assertEqual(menu_static["max_frame_delta"], 0)
        self.assertEqual(menu_static["max_pre_shot_delta"], 0)
        self.assertIn("menu-detected", menu_static["signals"])
        self.assertIn("no-frame-motion", menu_static["signals"])

    def test_rollout_artifact_validator_rejects_missing_frames(self):
        with TemporaryDirectory() as tmp:
            shot_dir = Path(tmp) / "shot_001"
            shot_dir.mkdir()
            metadata = {
                "frame_count": 2,
                "frames_dir": str(shot_dir / "frames"),
                "frames": [
                    {"path": str(shot_dir / "frames" / "frame_000000.png")},
                    {"path": str(shot_dir / "frames" / "frame_000001.png")},
                ],
                "max_frame_delta": 0,
            }
            (shot_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            result = validate_rollout_artifact(shot_dir)

        self.assertFalse(result["accepted"])
        self.assertEqual(result["classification"], "missing-artifact")
        self.assertEqual(result["invalid_reason"], "missing_artifact")
        self.assertIn("missing frames", result["message"])

    def test_rollout_artifact_validator_rejects_static_menu_with_inflated_metadata_deltas(self):
        from PIL import Image

        with TemporaryDirectory() as tmp:
            shot_dir = Path(tmp) / "shot_001"
            frames_dir = shot_dir / "frames"
            frames_dir.mkdir(parents=True)
            frame_path = frames_dir / "frame_000000.png"
            Image.new("RGB", (40, 30), (245, 245, 245)).save(frame_path, format="PNG")
            metadata = {
                "frame_count": 1,
                "frames_dir": str(frames_dir),
                "frames": [{"path": str(frame_path)}],
                "max_frame_delta": 999,
                "max_pre_shot_delta": 999,
            }
            (shot_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            result = validate_rollout_artifact(shot_dir)

        self.assertFalse(result["accepted"])
        self.assertIn(result["invalid_reason"], {"menu_detected", "no_frame_motion", "low_motion_suspicious"})
        self.assertIn("no-frame-motion", result["signals"])
        self.assertEqual(result["observed_max_frame_delta"], 0)

    def test_capture_desktop_rollout_crops_xvnc_desktop_to_game_viewport(self):
        class Grabber:
            def grab(self):
                from PIL import Image

                image = Image.new("RGB", (1024, 768), (0, 0, 0))
                for x in range(32, 672):
                    for y in range(64, 544):
                        image.putpixel((x, y), (10, 20, 30))
                image.putpixel((32, 64), (70, 80, 90))
                return image

        from PIL import Image

        baseline = Image.new("RGB", (1024, 768), (0, 0, 0))
        for x in range(32, 672):
            for y in range(64, 544):
                baseline.putpixel((x, y), (10, 20, 30))
        baseline.putpixel((32, 64), (70, 80, 90))

        with TemporaryDirectory() as tmp:
            metadata = capture_desktop_rollout(
                FakeBridge(),
                Path(tmp),
                target_fps=1,
                duration_seconds=1,
                max_frames=1,
                pre_shot_image=baseline,
                grabber=Grabber(),
            )

        self.assertEqual(metadata["desktop_crop"], [32, 64, 672, 544])
        self.assertEqual(metadata["frames"][0]["width"], 640)
        self.assertEqual(metadata["frames"][0]["height"], 480)

    def test_capture_desktop_rollout_keeps_frame_crop_when_pre_shot_is_already_cropped(self):
        class Grabber:
            def grab(self):
                from PIL import Image

                image = Image.new("RGB", (1024, 768), (0, 0, 0))
                for x in range(32, 672):
                    for y in range(64, 544):
                        image.putpixel((x, y), (10, 20, 30))
                image.putpixel((32, 64), (70, 80, 90))
                return image

        from PIL import Image

        cropped_baseline = Image.new("RGB", (640, 480), (10, 20, 30))
        cropped_baseline.putpixel((0, 0), (70, 80, 90))

        with TemporaryDirectory() as tmp:
            metadata = capture_desktop_rollout(
                FakeBridge(),
                Path(tmp),
                target_fps=1,
                duration_seconds=1,
                max_frames=1,
                pre_shot_image=cropped_baseline,
                grabber=Grabber(),
            )

        self.assertEqual(metadata["desktop_crop"], [32, 64, 672, 544])
        self.assertEqual(metadata["frames"][0]["width"], 640)
        self.assertEqual(metadata["frames"][0]["height"], 480)

    def test_capture_desktop_rollout_does_not_let_state_polling_throttle_frames(self):
        class Grabber:
            def grab(self):
                from PIL import Image

                image = Image.new("RGB", (4, 3), (10, 20, 30))
                image.putpixel((0, 0), (70, 80, 90))
                return image

        now = [0.0]

        def clock():
            return now[0]

        def sleeper(seconds):
            now[0] += seconds

        class SlowStateBridge(FakeBridge):
            def get_game_state(self):
                now[0] += 0.5
                return super().get_game_state()

            def get_current_score(self):
                now[0] += 0.5
                return super().get_current_score()

        with TemporaryDirectory() as tmp:
            metadata = capture_desktop_rollout(
                SlowStateBridge(),
                Path(tmp),
                target_fps=30,
                duration_seconds=1,
                grabber=Grabber(),
                clock=clock,
                sleeper=sleeper,
            )

        self.assertEqual(metadata["frame_count"], 30)
        self.assertEqual(len(metadata["state_samples"]), 1)

    def test_capture_desktop_rollout_starts_capture_before_shoot_callback(self):
        from PIL import Image

        events = []

        class Grabber:
            def __init__(self):
                self.calls = 0

            def grab(self):
                events.append(f"grab-{self.calls}")
                image = Image.new("RGB", (4, 3), (10 + self.calls, 20, 30))
                image.putpixel((0, 0), (70, 80, 90))
                self.calls += 1
                return image

        now = [1.0]

        def clock():
            return now[0]

        def sleeper(seconds):
            now[0] += seconds

        def shoot():
            events.append("shoot")
            return 1

        with TemporaryDirectory() as tmp:
            metadata = capture_desktop_rollout(
                FakeBridge(),
                Path(tmp),
                target_fps=2,
                duration_seconds=1.0,
                max_frames=2,
                grabber=Grabber(),
                shoot=shoot,
                clock=clock,
                sleeper=sleeper,
            )

        self.assertEqual(events, ["grab-0", "shoot", "grab-1"])
        self.assertEqual(metadata["shoot_response"], 1)
        self.assertEqual(metadata["shoot_frame_index"], 0)

    def test_capture_desktop_rollout_counts_duration_after_blocking_shoot_callback(self):
        from PIL import Image

        class Grabber:
            def __init__(self):
                self.calls = 0

            def grab(self):
                image = Image.new("RGB", (4, 3), (10 + self.calls, 20, 30))
                image.putpixel((0, 0), (70, 80, 90))
                self.calls += 1
                return image

        now = [0.0]

        def clock():
            return now[0]

        def sleeper(seconds):
            now[0] += seconds

        def shoot():
            now[0] += 1.0
            return 1

        with TemporaryDirectory() as tmp:
            metadata = capture_desktop_rollout(
                FakeBridge(),
                Path(tmp),
                target_fps=2,
                duration_seconds=1.0,
                max_duration_seconds=3.0,
                settle_seconds=0.0,
                settle_pixel_threshold=999,
                grabber=Grabber(),
                shoot=shoot,
                clock=clock,
                sleeper=sleeper,
            )

        self.assertGreaterEqual(metadata["frames"][-1]["t"], 2.0)
        self.assertEqual(metadata["shoot_frame_index"], 0)
        self.assertEqual(metadata["capture_stop_reason"], "settled")

    def test_capture_desktop_rollout_waits_for_visual_settle_after_minimum_duration(self):
        from PIL import Image

        class Grabber:
            def __init__(self):
                self.calls = 0

            def grab(self):
                value = 10 + min(self.calls, 3)
                image = Image.new("RGB", (4, 3), (value, 20, 30))
                image.putpixel((0, 0), (70, 80, 90))
                self.calls += 1
                return image

        now = [0.0]

        def clock():
            return now[0]

        def sleeper(seconds):
            now[0] += seconds

        with TemporaryDirectory() as tmp:
            metadata = capture_desktop_rollout(
                FakeBridge(),
                Path(tmp),
                target_fps=4,
                duration_seconds=0.25,
                max_duration_seconds=3.0,
                settle_seconds=0.5,
                settle_pixel_threshold=0,
                grabber=Grabber(),
                shoot=lambda: 1,
                clock=clock,
                sleeper=sleeper,
            )

        self.assertEqual(metadata["capture_stop_reason"], "settled")
        self.assertGreaterEqual(metadata["frame_count"], 6)
        self.assertEqual(metadata["frames"][-1]["frame_delta"]["changed_pixel_count"], 0)

    def test_collect_rollouts_saves_desktop_pre_shot_baseline_before_shoot_and_records_delta(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]
        events = []

        class LoggingBridge(FakeBridge):
            def get_game_state(self):
                events.append("state")
                return super().get_game_state()

            def get_current_score(self):
                events.append("score")
                return super().get_current_score()

            def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
                events.append("shoot")
                return super().shoot(x, y, tap_time=tap_time, fast=fast, release_time=release_time)

        class Grabber:
            def __init__(self):
                self.calls = 0

            def grab(self):
                from PIL import Image

                self.calls += 1
                image = Image.new("RGB", (4, 3), (10, 20, 30))
                image.putpixel((0, 0), (70, 80, 90))
                image.putpixel((1, 0), (100, 110, 120))
                if self.calls > 1:
                    image.putpixel((2, 0), (130, 140, 150))
                return image

        from PIL import Image

        def pre_shot_grabber():
            events.append("baseline")
            baseline = Image.new("RGB", (4, 3), (10, 20, 30))
            baseline.putpixel((0, 0), (70, 80, 90))
            return baseline

        now = [1.0]

        def clock():
            return now[0]

        def sleeper(seconds):
            now[0] += seconds

        def capture_rollout(bridge, output_dir, **kwargs):
            return capture_desktop_rollout(bridge, output_dir, grabber=Grabber(), **kwargs)

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                LoggingBridge(),
                Path(tmp),
                actions,
                target_fps=2,
                duration_seconds=1.0,
                frame_height=480,
                max_frames=2,
                pre_shot_grabber=pre_shot_grabber,
                capture_rollout=capture_rollout,
                clock=clock,
                sleeper=sleeper,
            )
            metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))

            self.assertTrue((Path(tmp) / "shot_001" / "pre_shot.png").is_file())

        self.assertEqual(events[:3], ["state", "score", "baseline"])
        self.assertLess(events.index("baseline"), events.index("shoot"))
        self.assertEqual(manifest["attempt_count"], 1)
        self.assertEqual(manifest["accepted_rollout_count"], 0)
        self.assertEqual(manifest["rollout_count"], 0)
        self.assertEqual(metadata["frames"][0]["pre_shot_delta"]["changed_pixel_count"], 1)
        self.assertEqual(metadata["frames"][1]["pre_shot_delta"]["changed_pixel_count"], 2)
        self.assertEqual(metadata["max_pre_shot_delta"], 2)
        self.assertEqual(metadata["max_pre_shot_delta_bbox"], [1, 0, 3, 1])
        self.assertEqual(metadata["pre_shot_path"], str(Path(tmp) / "shot_001" / "pre_shot.png"))
        self.assertEqual(metadata["pre_shot_sample"], {"state": "PLAYING", "score": 200})
        self.assertEqual(metadata["frames"][1]["frame_delta"]["changed_pixel_count"], 1)

    def test_collect_rollouts_can_defer_shoot_until_desktop_capture_starts(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]
        events = []

        class LoggingBridge(FakeBridge):
            def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
                events.append("shoot")
                return super().shoot(x, y, tap_time=tap_time, fast=fast, release_time=release_time)

        def capture_rollout(bridge, output_dir, **kwargs):
            events.append("capture-start")
            response = kwargs["shoot"]()
            events.append("capture-after-shoot")
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True)
            from PIL import Image

            Image.new("RGB", (160, 90), (50, 60, 70)).save(frames_dir / "frame_000000.png", format="PNG")
            return {"frame_count": 1, "frames_dir": str(frames_dir), "shoot_response": response, "shoot_frame_index": 0}

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                LoggingBridge(),
                Path(tmp),
                actions,
                target_fps=1,
                duration_seconds=1,
                capture_rollout=capture_rollout,
                shoot_before_capture=False,
            )

        self.assertEqual(events, ["capture-start", "shoot", "capture-after-shoot"])
        self.assertEqual(manifest["rollouts"][0]["shoot_response"], 1)

    def test_collect_rollouts_writes_review_mp4_for_each_shot(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]
        runner_calls = []

        def capture_rollout(bridge, output_dir, **kwargs):
            from PIL import Image

            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True)
            for index in range(2):
                image = Image.new("RGB", (20, 20), (10 + index * 80, 20, 30))
                image.putpixel((0, 0), (70, 80, 90))
                image.save(frames_dir / f"frame_{index:06d}.png", format="PNG")
            metadata = {"frame_count": 2, "frames_dir": str(frames_dir)}
            (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            return metadata

        def video_runner(command, check, stdout, stderr):
            runner_calls.append(command)
            Path(command[-1]).write_bytes(b"mp4")

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                FakeBridge(),
                Path(tmp),
                actions,
                target_fps=30,
                duration_seconds=1,
                capture_rollout=capture_rollout,
                video_runner=video_runner,
            )
            metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["rollouts"][0]["video_path"], str(Path(tmp) / "shot_001" / "rollout.mp4"))
        self.assertEqual(metadata["video_path"], str(Path(tmp) / "shot_001" / "rollout.mp4"))
        self.assertIn("-framerate", runner_calls[0])
        self.assertIn("30", runner_calls[0])

    def test_prepare_rollout_video_frames_prefixes_pre_shot_and_overlays_action_text(self):
        from PIL import Image, ImageChops

        action = {
            "action_type": "drag_hold_release",
            "coordinate_frame": "slingshot_relative",
            "drag_start": [300, 220],
            "drag_release": [-50, 40],
            "tapTime": 70,
            "holdTime": 120,
        }

        with TemporaryDirectory() as tmp:
            shot_dir = Path(tmp)
            frames_dir = shot_dir / "frames"
            frames_dir.mkdir()
            pre_shot = Image.new("RGB", (160, 90), (20, 30, 40))
            pre_shot.putpixel((10, 10), (90, 100, 110))
            pre_shot_path = shot_dir / "pre_shot.png"
            pre_shot.save(pre_shot_path, format="PNG")
            for index, color in enumerate(((50, 60, 70), (80, 90, 100))):
                Image.new("RGB", (160, 90), color).save(frames_dir / f"frame_{index:06d}.png", format="PNG")

            video = prepare_rollout_video_frames(
                shot_dir,
                frames_dir,
                action=action,
                shot={"x": 250, "y": 299, "tapTime": 70, "releaseTime": 120},
                fps=4,
                pre_shot_path=pre_shot_path,
                lead_in_seconds=0.5,
            )

            first_frame = Image.open(Path(video["video_frames_dir"]) / "frame_000000.png")
            raw_first_frame = Image.open(frames_dir / "frame_000000.png")
            raw_pre_shot = Image.open(pre_shot_path)

        self.assertEqual(video["pre_action_frame_count"], 2)
        self.assertEqual(video["video_frame_count"], 4)
        self.assertEqual(video["video_input_pattern"], str(Path(tmp) / "video_frames" / "frame_%06d.png"))
        self.assertIn("drag_mode=slingshot_relative", video["video_overlay"]["text"])
        self.assertIn("drag_xy=(300,220)", video["video_overlay"]["text"])
        self.assertIn("pull_release_xy=(-50,40)", video["video_overlay"]["text"])
        self.assertIn("launch_xy=(350,260)", video["video_overlay"]["text"])
        self.assertIn("tapTime=70", video["video_overlay"]["text"])
        self.assertIn("releaseTime=120", video["video_overlay"]["text"])
        self.assertIn("green=launch", video["video_overlay"]["action_guide"])
        self.assertIsNotNone(ImageChops.difference(first_frame.crop((0, 0, 160, 24)), raw_pre_shot.crop((0, 0, 160, 24))).getbbox())
        self.assertIsNone(ImageChops.difference(raw_first_frame, Image.new("RGB", (160, 90), (50, 60, 70))).getbbox())

    def test_prepare_rollout_video_frames_starts_with_neutral_pre_drag_frame(self):
        from PIL import Image, ImageChops

        action = {
            "action_type": "drag_hold_release",
            "coordinate_frame": "slingshot_relative",
            "drag_start": [80, 40],
            "drag_release": [-30, 10],
            "tapTime": 70,
            "holdTime": 120,
        }

        with TemporaryDirectory() as tmp:
            shot_dir = Path(tmp)
            frames_dir = shot_dir / "frames"
            frames_dir.mkdir()
            pre_shot = Image.new("RGB", (160, 90), (20, 30, 40))
            pre_shot_path = shot_dir / "pre_shot.png"
            pre_shot.save(pre_shot_path, format="PNG")
            Image.new("RGB", (160, 90), (50, 60, 70)).save(frames_dir / "frame_000000.png", format="PNG")

            video = prepare_rollout_video_frames(
                shot_dir,
                frames_dir,
                action=action,
                shot={"x": 50, "y": 59, "tapTime": 70, "releaseTime": 120},
                fps=4,
                pre_shot_path=pre_shot_path,
                lead_in_seconds=1.0,
            )

            first_frame = Image.open(Path(video["video_frames_dir"]) / "frame_000000.png")
            aim_frame = Image.open(Path(video["video_frames_dir"]) / "frame_000002.png")
            raw_pre_shot = Image.open(pre_shot_path)

        action_area = (0, 24, 160, 90)
        self.assertEqual(video["pre_drag_frame_count"], 2)
        self.assertEqual(video["aim_hold_frame_count"], 2)
        self.assertEqual(video["video_phase_counts"], {"pre_drag": 2, "aim_hold": 2, "rollout": 1})
        self.assertEqual(PRE_DRAG_OVERLAY_TEXT, "phase=pre_drag pre_shot_baseline")
        self.assertIsNone(ImageChops.difference(first_frame.crop(action_area), raw_pre_shot.crop(action_area)).getbbox())
        self.assertIsNotNone(ImageChops.difference(aim_frame.crop(action_area), raw_pre_shot.crop(action_area)).getbbox())

    def test_format_action_overlay_text_describes_absolute_and_slingshot_actions(self):
        slingshot_text = format_action_overlay_text(
            {
                "action_type": "drag_hold_release",
                "coordinate_frame": "slingshot_relative",
                "drag_start": [118, 315],
                "drag_release": [-42, -15],
                "tapTime": 45,
                "holdTime": 120,
            },
            {"x": 76, "y": 149, "tapTime": 45, "releaseTime": 120},
        )
        absolute_text = format_action_overlay_text(
            {"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0},
            {"x": 250, "y": 219, "tapTime": 0, "releaseTime": 0},
        )

        self.assertIn("drag_mode=slingshot_relative", slingshot_text)
        self.assertIn("release_mode=slingshot_relative", slingshot_text)
        self.assertIn("drag_xy=(118,315)", slingshot_text)
        self.assertIn("pull_release_xy=(-42,-15)", slingshot_text)
        self.assertIn("socket_xy=(76,149)", slingshot_text)
        self.assertIn("drag_mode=absolute", absolute_text)
        self.assertIn("pull_release_xy=(250,260)", absolute_text)

    def test_collect_rollouts_writes_action_logs_and_uses_video_frames_for_mp4(self):
        actions = [
            {
                "action_type": "drag_hold_release",
                "coordinate_frame": "slingshot_relative",
                "drag_start": [300, 220],
                "drag_release": [-50, 40],
                "tapTime": 70,
                "holdTime": 120,
            }
        ]
        runner_calls = []

        def capture_rollout(bridge, output_dir, **kwargs):
            from PIL import Image

            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True)
            for index in range(2):
                Image.new("RGB", (160, 90), (50 + index, 60, 70)).save(frames_dir / f"frame_{index:06d}.png", format="PNG")
            return {"frame_count": 2, "frames_dir": str(frames_dir), "pre_shot_path": str(output_dir / "pre_shot.png")}

        def pre_shot_grabber():
            from PIL import Image

            image = Image.new("RGB", (160, 90), (20, 30, 40))
            image.putpixel((1, 1), (80, 90, 100))
            return image

        def video_runner(command, check, stdout, stderr):
            runner_calls.append(command)
            Path(command[-1]).write_bytes(b"mp4")

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                FakeBridge(),
                Path(tmp),
                actions,
                target_fps=4,
                duration_seconds=1,
                frame_height=480,
                capture_rollout=capture_rollout,
                pre_shot_grabber=pre_shot_grabber,
                video_runner=video_runner,
            )
            action_log = json.loads((Path(tmp) / "action_log.json").read_text(encoding="utf-8"))
            jsonl_lines = (Path(tmp) / "action_log.jsonl").read_text(encoding="utf-8").splitlines()
            metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(action_log["trial_count"], 1)
        self.assertEqual(len(jsonl_lines), 1)
        self.assertEqual(json.loads(jsonl_lines[0])["action"], actions[0])
        self.assertEqual(manifest["action_log_path"], str(Path(tmp) / "action_log.json"))
        self.assertEqual(manifest["action_log_jsonl_path"], str(Path(tmp) / "action_log.jsonl"))
        self.assertIn("video_frames/frame_%06d.png", runner_calls[0][runner_calls[0].index("-i") + 1])
        self.assertGreater(metadata["video_frame_count"], metadata["frame_count"])
        self.assertIn("video_overlay", metadata)

    def test_collect_rollouts_records_varied_trials_within_one_episode(self):
        actions = [
            {"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0},
            {"coordinate_frame": "absolute", "release": [240, 250], "tapTime": 45},
            {"coordinate_frame": "absolute", "release": [230, 240], "tapTime": 70},
        ]

        def capture_rollout(bridge, output_dir, **kwargs):
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True)
            from PIL import Image

            Image.new("RGB", (160, 90), (50, 60, 70)).save(frames_dir / "frame_000000.png", format="PNG")
            return {"frame_count": 1, "frames_dir": str(frames_dir)}

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                FakeBridge(),
                Path(tmp),
                actions,
                target_fps=1,
                duration_seconds=1,
                capture_rollout=capture_rollout,
                video_runner=lambda command, check, stdout, stderr: Path(command[-1]).write_bytes(b"mp4"),
            )
            action_log = json.loads((Path(tmp) / "action_log.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["replay_mode"], "same-episode-varied-trials")
        self.assertEqual(manifest["attempt_count"], 3)
        self.assertEqual(manifest["accepted_rollout_count"], 0)
        self.assertEqual(manifest["rollout_count"], 0)
        self.assertEqual(action_log["attempt_count"], 3)
        self.assertEqual(action_log["accepted_trial_count"], 0)
        self.assertEqual(action_log["trial_count"], 3)
        self.assertEqual([trial["shot_name"] for trial in action_log["trials"]], ["shot_001", "shot_002", "shot_003"])
        self.assertEqual(len({tuple(trial["action"].get("release", [])) for trial in action_log["trials"]}), 3)

    def test_collect_rollouts_records_video_error_without_losing_frames(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]

        def capture_rollout(bridge, output_dir, **kwargs):
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True)
            metadata = {"frame_count": 1, "frames_dir": str(frames_dir)}
            (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            return metadata

        def video_runner(command, check, stdout, stderr):
            raise FileNotFoundError("ffmpeg")

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                FakeBridge(),
                Path(tmp),
                actions,
                target_fps=30,
                duration_seconds=1,
                capture_rollout=capture_rollout,
                video_runner=video_runner,
            )
            metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["rollouts"][0]["frame_count"], 1)
        self.assertIn("ffmpeg", metadata["video_error"])

    def test_parser_defaults_to_high_fps_for_review_collection(self):
        args = build_parser().parse_args(["--output-dir", "data/review"])

        self.assertEqual(args.fps, 30.0)

    def test_select_level_in_display_clicks_play_inputs_level_and_confirms(self):
        calls = []
        sleeps = []

        def runner(command, check):
            calls.append(command)

        select_level_in_display(3, runner=runner, sleeper=sleeps.append)

        self.assertEqual(
            calls,
            [
                ["xdotool", "mousemove", "512", "390", "click", "1"],
                ["xdotool", "mousemove", "495", "343", "click", "1"],
                ["xdotool", "type", "3"],
                ["xdotool", "mousemove", "492", "465", "click", "1"],
            ],
        )
        self.assertTrue(sleeps)

    def test_invalid_attempt_retries_and_quarantines_evidence(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]
        events = []

        class FakeProcess:
            def __init__(self, pid):
                self.pid = pid
                self.terminated = False
                self.waited = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                self.waited = True

        processes = []
        bridges = [FakeBridge(), FakeBridge()]

        def start_engine_func(game_dir, headless):
            process = FakeProcess(5000 + len(processes))
            processes.append(process)
            events.append(("start", process.pid))
            return process

        def connect_func(host, port, timeout, deadline_seconds):
            bridge = bridges.pop(0)
            events.append(("connect", len(processes)))
            return bridge

        def capture_rollout(bridge, output_dir, **kwargs):
            from PIL import Image

            output_dir.mkdir(parents=True, exist_ok=True)
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            first_frame = Image.new("RGB", (20, 20), (50, 60, 70))
            second_frame = Image.new("RGB", (20, 20), (50, 60, 70))
            if len(processes) == 1:
                for x in range(2):
                    for y in range(11):
                        second_frame.putpixel((x, y), (51, 60, 70))
            else:
                second_frame = Image.new("RGB", (20, 20), (180, 20, 40))
            first_frame.save(frames_dir / "frame_000000.png", format="PNG")
            second_frame.save(frames_dir / "frame_000001.png", format="PNG")
            metadata = {"frame_count": 2, "frames_dir": str(frames_dir)}
            (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            return metadata

        with TemporaryDirectory() as tmp:
            manifest = collect_fresh_engine_rollouts(
                Path(tmp),
                actions,
                game_dir=Path("game"),
                host="127.0.0.1",
                port=2004,
                agent_id=28888,
                speed=1,
                connect_timeout=1,
                read_timeout=2,
                prepare_timeout=3,
                frame_height=480,
                fast=True,
                headless=False,
                target_fps=1,
                duration_seconds=1,
                ui_level=None,
                ui_settle_seconds=0,
                fresh_engine_attempts=2,
                start_engine_func=start_engine_func,
                connect_func=connect_func,
                prepare_func=lambda bridge, timeout, poll_delay: bridge.get_game_state(),
                capture_rollout=capture_rollout,
                video_runner=lambda command, check, stdout, stderr: Path(command[-1]).write_bytes(b"mp4"),
            )
            accepted_metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))
            action_log = json.loads((Path(tmp) / "action_log.json").read_text(encoding="utf-8"))
            invalid_attempt = manifest["invalid_attempts"][0]
            quarantined_path = Path(invalid_attempt["quarantined_path"])
            quarantined_path_exists = quarantined_path.is_dir()
            quarantined_metadata = json.loads((quarantined_path / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["attempt_count"], 2)
        self.assertEqual(manifest["accepted_rollout_count"], 1)
        self.assertEqual(manifest["rollout_count"], 1)
        self.assertEqual(len(manifest["invalid_attempts"]), 1)
        self.assertEqual([rollout["accepted"] for rollout in manifest["rollouts"]], [False, True])
        self.assertTrue(quarantined_path_exists)
        self.assertEqual(quarantined_path.name, "shot_001_attempt_01")
        self.assertEqual(invalid_attempt["attempt_status"], "invalid_retryable")
        self.assertEqual(invalid_attempt["recovery_action"], "fresh_engine_retry")
        self.assertEqual(invalid_attempt["retry_attempt"], 1)
        self.assertEqual(quarantined_metadata["attempt_status"], "invalid_retryable")
        self.assertEqual(accepted_metadata["attempt_status"], "accepted")
        self.assertTrue(accepted_metadata["accepted"])
        self.assertEqual(accepted_metadata["retry_attempt"], 2)
        self.assertEqual(accepted_metadata["prior_invalid_attempts"][0]["quarantined_path"], str(quarantined_path))
        self.assertEqual(action_log["attempt_count"], 2)
        self.assertEqual(action_log["accepted_trial_count"], 1)
        self.assertEqual([trial["accepted"] for trial in action_log["trials"]], [False, True])
        self.assertEqual(action_log["trials"][0]["fresh_engine_attempt"], 1)
        self.assertEqual(action_log["trials"][1]["fresh_engine_attempt"], 2)
        self.assertEqual(len(processes), 2)
        self.assertTrue(all(process.terminated and process.waited for process in processes))

    def test_quarantined_invalid_attempt_metadata_is_self_contained_after_retry_overwrite(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]

        class FakeProcess:
            def __init__(self, pid):
                self.pid = pid
                self.terminated = False
                self.waited = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                self.waited = True

        processes = []
        bridges = [FakeBridge(), FakeBridge()]

        def start_engine_func(game_dir, headless):
            process = FakeProcess(5500 + len(processes))
            processes.append(process)
            return process

        def connect_func(host, port, timeout, deadline_seconds):
            return bridges.pop(0)

        def capture_rollout(bridge, output_dir, **kwargs):
            from PIL import Image

            output_dir.mkdir(parents=True, exist_ok=True)
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            pre_shot_path = output_dir / "pre_shot.png"
            Image.new("RGB", (20, 20), (50, 60, 70)).save(pre_shot_path, format="PNG")
            first_frame = Image.new("RGB", (20, 20), (50, 60, 70))
            second_frame = Image.new("RGB", (20, 20), (50, 60, 70))
            if len(processes) == 1:
                for x in range(2):
                    for y in range(11):
                        second_frame.putpixel((x, y), (51, 60, 70))
            else:
                second_frame = Image.new("RGB", (20, 20), (180, 20, 40))
            frame_paths = [frames_dir / "frame_000000.png", frames_dir / "frame_000001.png"]
            first_frame.save(frame_paths[0], format="PNG")
            second_frame.save(frame_paths[1], format="PNG")
            metadata = {
                "frame_count": 2,
                "frames_dir": str(frames_dir),
                "frames": [{"path": str(frame_path)} for frame_path in frame_paths],
                "metadata_path": str(output_dir / "metadata.json"),
                "pre_shot_path": str(pre_shot_path),
            }
            (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            return metadata

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            manifest = collect_fresh_engine_rollouts(
                output_dir,
                actions,
                game_dir=Path("game"),
                host="127.0.0.1",
                port=2004,
                agent_id=28888,
                speed=1,
                connect_timeout=1,
                read_timeout=2,
                prepare_timeout=3,
                frame_height=480,
                fast=True,
                headless=False,
                target_fps=1,
                duration_seconds=1,
                ui_level=None,
                ui_settle_seconds=0,
                fresh_engine_attempts=2,
                start_engine_func=start_engine_func,
                connect_func=connect_func,
                prepare_func=lambda bridge, timeout, poll_delay: bridge.get_game_state(),
                capture_rollout=capture_rollout,
                video_runner=lambda command, check, stdout, stderr: Path(command[-1]).write_bytes(b"mp4"),
            )
            invalid_attempt = manifest["invalid_attempts"][0]
            accepted_metadata = json.loads((output_dir / "shot_001" / "metadata.json").read_text(encoding="utf-8"))
            action_log = json.loads((output_dir / "action_log.json").read_text(encoding="utf-8"))
            quarantined_path = Path(invalid_attempt["quarantined_path"])
            quarantined_metadata = json.loads((quarantined_path / "metadata.json").read_text(encoding="utf-8"))
            quarantined_validation = validate_rollout_artifact(quarantined_path)
            canonical_validation = validate_rollout_artifact(output_dir / "shot_001")

        self.assertTrue(canonical_validation["accepted"])
        self.assertFalse(quarantined_validation["accepted"])
        self.assertEqual(quarantined_validation["invalid_reason"], "low_motion_suspicious")
        self.assertEqual(quarantined_metadata["metadata_path"], str(quarantined_path / "metadata.json"))
        self.assertEqual(quarantined_metadata["pre_shot_path"], str(quarantined_path / "pre_shot.png"))
        self.assertEqual(quarantined_metadata["frames_dir"], str(quarantined_path / "frames"))
        self.assertEqual(quarantined_metadata["video_path"], str(quarantined_path / "rollout.mp4"))
        self.assertEqual(
            [frame["path"] for frame in quarantined_metadata["frames"]],
            [str(quarantined_path / "frames" / "frame_000000.png"), str(quarantined_path / "frames" / "frame_000001.png")],
        )
        self.assertEqual(quarantined_metadata["artifact_validation"]["shot_dir"], str(quarantined_path))
        self.assertEqual(quarantined_metadata["artifact_validation"]["metadata_path"], str(quarantined_path / "metadata.json"))
        self.assertEqual(quarantined_metadata["artifact_validation"]["frames_dir"], str(quarantined_path / "frames"))
        self.assertEqual(invalid_attempt["metadata_path"], str(quarantined_path / "metadata.json"))
        self.assertEqual(invalid_attempt["artifact_validation"]["metadata_path"], str(quarantined_path / "metadata.json"))
        self.assertEqual(action_log["trials"][0]["artifact_validation"]["shot_dir"], str(quarantined_path))
        self.assertEqual(accepted_metadata["prior_invalid_attempts"][0]["metadata_path"], str(quarantined_path / "metadata.json"))
        self.assertEqual(accepted_metadata["prior_invalid_attempts"][0]["artifact_validation"]["frames_dir"], str(quarantined_path / "frames"))

    def test_invalid_attempt_retry_exhaustion_fails_closed(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]

        class FakeProcess:
            def __init__(self, pid):
                self.pid = pid
                self.terminated = False
                self.waited = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                self.waited = True

        processes = []
        bridges = [FakeBridge(), FakeBridge()]

        def start_engine_func(game_dir, headless):
            process = FakeProcess(6000 + len(processes))
            processes.append(process)
            return process

        def connect_func(host, port, timeout, deadline_seconds):
            return bridges.pop(0)

        def capture_rollout(bridge, output_dir, **kwargs):
            from PIL import Image

            output_dir.mkdir(parents=True, exist_ok=True)
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            first_frame = Image.new("RGB", (20, 20), (50, 60, 70))
            second_frame = Image.new("RGB", (20, 20), (50, 60, 70))
            for x in range(2):
                for y in range(11):
                    second_frame.putpixel((x, y), (51, 60, 70))
            first_frame.save(frames_dir / "frame_000000.png", format="PNG")
            second_frame.save(frames_dir / "frame_000001.png", format="PNG")
            metadata = {"frame_count": 2, "frames_dir": str(frames_dir)}
            (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            return metadata

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with self.assertRaisesRegex(
                RolloutCollectionError,
                "fresh-engine retries exhausted.*accepted 0/1.*2 attempts.*low_motion_suspicious",
            ) as raised:
                collect_fresh_engine_rollouts(
                    output_dir,
                    actions,
                    game_dir=Path("game"),
                    host="127.0.0.1",
                    port=2004,
                    agent_id=28888,
                    speed=1,
                    connect_timeout=1,
                    read_timeout=2,
                    prepare_timeout=3,
                    frame_height=480,
                    fast=True,
                    headless=False,
                    target_fps=1,
                    duration_seconds=1,
                    ui_level=None,
                    ui_settle_seconds=0,
                    fresh_engine_attempts=2,
                    start_engine_func=start_engine_func,
                    connect_func=connect_func,
                    prepare_func=lambda bridge, timeout, poll_delay: bridge.get_game_state(),
                    capture_rollout=capture_rollout,
                    video_runner=lambda command, check, stdout, stderr: Path(command[-1]).write_bytes(b"mp4"),
                )
            self.assertIn(str(output_dir / "manifest.json"), str(raised.exception))
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            final_metadata = json.loads((output_dir / "shot_001" / "metadata.json").read_text(encoding="utf-8"))
            action_log = json.loads((output_dir / "action_log.json").read_text(encoding="utf-8"))
            quarantine_paths = [Path(attempt["quarantined_path"]) for attempt in manifest["invalid_attempts"]]
            quarantine_paths_exist = [path.is_dir() for path in quarantine_paths]

        self.assertEqual(manifest["attempt_count"], 2)
        self.assertEqual(manifest["accepted_rollout_count"], 0)
        self.assertEqual(manifest["rollout_count"], 0)
        self.assertEqual(manifest["collection_status"], "retry_exhausted")
        self.assertEqual(manifest["collection_error"]["invalid_reasons"], ["low_motion_suspicious"])
        self.assertEqual(len(manifest["invalid_attempts"]), 2)
        self.assertEqual([attempt["attempt_status"] for attempt in manifest["invalid_attempts"]], ["invalid_retryable", "invalid_exhausted"])
        self.assertEqual([attempt["retry_attempt"] for attempt in manifest["invalid_attempts"]], [1, 2])
        self.assertEqual(final_metadata["attempt_status"], "invalid_exhausted")
        self.assertFalse(final_metadata["accepted"])
        self.assertEqual(final_metadata["invalid_reason"], "low_motion_suspicious")
        self.assertEqual(final_metadata["retry_attempt"], 2)
        self.assertEqual(final_metadata["recovery_action"], "fresh_engine_attempts_exhausted")
        self.assertEqual(final_metadata["quarantined_path"], str(quarantine_paths[1]))
        self.assertEqual(final_metadata["prior_invalid_attempts"][0]["quarantined_path"], str(quarantine_paths[0]))
        self.assertTrue(all(quarantine_paths_exist))
        self.assertEqual([path.name for path in quarantine_paths], ["shot_001_attempt_01", "shot_001_attempt_02"])
        self.assertEqual(action_log["attempt_count"], 2)
        self.assertEqual(action_log["accepted_trial_count"], 0)
        self.assertEqual([trial["accepted"] for trial in action_log["trials"]], [False, False])
        self.assertEqual(len(processes), 2)
        self.assertTrue(all(process.terminated and process.waited for process in processes))

    def test_pre_shot_guard_retry_exhaustion_writes_manifest_and_action_logs(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]

        class FakeProcess:
            def __init__(self, pid):
                self.pid = pid
                self.terminated = False
                self.waited = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                self.waited = True

        class GuardFailBridge(FakeBridge):
            def __init__(self):
                super().__init__()
                self.next_calls = 0
                self.novelty_calls = 0

            def get_novelty_info(self):
                self.novelty_calls += 1
                return -1

            def load_next_available_level(self):
                self.next_calls += 1
                return 1

        processes = []
        bridges = [GuardFailBridge(), GuardFailBridge()]

        def start_engine_func(game_dir, headless):
            process = FakeProcess(6500 + len(processes))
            processes.append(process)
            return process

        def connect_func(host, port, timeout, deadline_seconds):
            return bridges.pop(0)

        def menu_pre_shot_grabber():
            from PIL import Image

            image = Image.new("RGB", (40, 30), (245, 245, 245))
            image.putpixel((3, 3), (230, 40, 40))
            image.putpixel((4, 3), (40, 130, 230))
            return image

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with self.assertRaisesRegex(
                RolloutCollectionError,
                "fresh-engine retries exhausted.*accepted 0/1.*2 attempts.*missing_artifact",
            ) as raised:
                collect_fresh_engine_rollouts(
                    output_dir,
                    actions,
                    game_dir=Path("game"),
                    host="127.0.0.1",
                    port=2004,
                    agent_id=28888,
                    speed=1,
                    connect_timeout=1,
                    read_timeout=2,
                    prepare_timeout=3,
                    frame_height=480,
                    fast=True,
                    headless=False,
                    target_fps=1,
                    duration_seconds=1,
                    ui_level=None,
                    ui_settle_seconds=0,
                    fresh_engine_attempts=2,
                    start_engine_func=start_engine_func,
                    connect_func=connect_func,
                    prepare_func=lambda bridge, timeout, poll_delay: bridge.get_game_state(),
                    pre_shot_grabber=menu_pre_shot_grabber,
                    sleeper=lambda seconds: None,
                )
            self.assertIn(str(output_dir / "manifest.json"), str(raised.exception))
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            action_log = json.loads((output_dir / "action_log.json").read_text(encoding="utf-8"))
            jsonl_trials = [json.loads(line) for line in (output_dir / "action_log.jsonl").read_text(encoding="utf-8").splitlines()]
            quarantine_paths = [Path(attempt["quarantined_path"]) for attempt in manifest["invalid_attempts"]]
            quarantine_paths_exist = [path.is_dir() for path in quarantine_paths]
            quarantine_metadata = [json.loads((path / "metadata.json").read_text(encoding="utf-8")) for path in quarantine_paths]

        self.assertEqual(manifest["attempt_count"], 2)
        self.assertEqual(manifest["accepted_rollout_count"], 0)
        self.assertEqual(manifest["rollout_count"], 0)
        self.assertEqual(manifest["collection_status"], "retry_exhausted")
        self.assertEqual(manifest["collection_error"]["invalid_reasons"], ["missing_artifact"])
        self.assertEqual([attempt["accepted"] for attempt in manifest["invalid_attempts"]], [False, False])
        self.assertEqual([attempt["attempt_status"] for attempt in manifest["invalid_attempts"]], ["invalid_retryable", "invalid_exhausted"])
        self.assertEqual([attempt["invalid_reason"] for attempt in manifest["invalid_attempts"]], ["missing_artifact", "missing_artifact"])
        self.assertEqual([attempt["retry_attempt"] for attempt in manifest["invalid_attempts"]], [1, 2])
        self.assertEqual([path.name for path in quarantine_paths], ["shot_001_attempt_01", "shot_001_attempt_02"])
        self.assertTrue(all(quarantine_paths_exist))
        self.assertEqual([metadata["pre_shot_guard"]["status"] for metadata in quarantine_metadata], ["recovery_failed", "recovery_failed"])
        self.assertEqual(action_log["attempt_count"], 2)
        self.assertEqual(action_log["accepted_trial_count"], 0)
        self.assertEqual([trial["accepted"] for trial in action_log["trials"]], [False, False])
        self.assertEqual([trial["attempt_status"] for trial in jsonl_trials], ["invalid_retryable", "invalid_exhausted"])
        self.assertEqual(len(processes), 2)
        self.assertTrue(all(process.terminated and process.waited for process in processes))

    def test_collect_fresh_engine_rollouts_restarts_engine_per_action(self):
        actions = [
            {"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0},
            {"coordinate_frame": "absolute", "release": [240, 250], "tapTime": 45},
        ]
        bridges = [FakeBridge(), FakeBridge()]
        processes = []
        selected_levels = []

        class FakeProcess:
            def __init__(self, pid):
                self.pid = pid
                self.terminated = False
                self.waited = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                self.waited = True

        def start_engine_func(game_dir, headless):
            process = FakeProcess(1000 + len(processes))
            processes.append(process)
            return process

        def connect_func(host, port, timeout, deadline_seconds):
            return bridges.pop(0)

        def capture_rollout(bridge, output_dir, **kwargs):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "metadata.json").write_text(json.dumps({"frame_count": 1}), encoding="utf-8")
            return {"frame_count": 1}

        with TemporaryDirectory() as tmp:
            manifest = collect_fresh_engine_rollouts(
                Path(tmp),
                actions,
                game_dir=Path("game"),
                host="127.0.0.1",
                port=2004,
                agent_id=28888,
                speed=1,
                connect_timeout=1,
                read_timeout=2,
                prepare_timeout=3,
                frame_height=480,
                fast=True,
                headless=False,
                target_fps=1,
                duration_seconds=1,
                ui_level=1,
                ui_settle_seconds=0,
                start_engine_func=start_engine_func,
                connect_func=connect_func,
                prepare_func=lambda bridge, timeout, poll_delay: bridge.get_game_state(),
                capture_rollout=capture_rollout,
                select_level_func=lambda level: selected_levels.append(level),
            )

            saved_manifest = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["attempt_count"], 2)
        self.assertEqual(manifest["accepted_rollout_count"], 0)
        self.assertEqual(manifest["rollout_count"], 0)
        self.assertEqual(saved_manifest["replay_mode"], "fresh-engine-per-rollout")
        self.assertEqual(selected_levels, [1, 1])
        self.assertEqual(len(processes), 2)
        self.assertTrue(all(process.terminated and process.waited for process in processes))

    def test_collect_fresh_engine_rollouts_waits_after_engine_start_and_connect_before_configure(self):
        events = []

        class FakeProcess:
            pid = 1234

            def poll(self):
                return None

            def terminate(self):
                events.append("terminate")

            def wait(self, timeout=None):
                events.append("wait")

        def start_engine_func(game_dir, headless):
            events.append("start")
            return FakeProcess()

        def sleeper(seconds):
            events.append(("sleep", seconds))

        def connect_func(host, port, timeout, deadline_seconds):
            events.append("connect")
            return FakeBridge()

        def prepare_func(bridge, timeout, poll_delay):
            events.append("prepare")
            return bridge.get_game_state()

        def capture_rollout(bridge, output_dir, **kwargs):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "metadata.json").write_text(json.dumps({"frame_count": 1, "frames_dir": str(output_dir / "frames")}), encoding="utf-8")
            return {"frame_count": 1, "frames_dir": str(output_dir / "frames")}

        with TemporaryDirectory() as tmp:
            collect_fresh_engine_rollouts(
                Path(tmp),
                [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}],
                game_dir=Path("game"),
                host="127.0.0.1",
                port=2004,
                agent_id=28888,
                speed=1,
                connect_timeout=1,
                read_timeout=2,
                prepare_timeout=3,
                frame_height=480,
                fast=True,
                headless=False,
                target_fps=1,
                duration_seconds=1,
                ui_level=None,
                ui_settle_seconds=0,
                engine_settle_seconds=7,
                agent_settle_seconds=11,
                start_engine_func=start_engine_func,
                connect_func=connect_func,
                prepare_func=prepare_func,
                capture_rollout=capture_rollout,
                sleeper=sleeper,
                video_runner=lambda command, check, stdout, stderr: Path(command[-1]).write_bytes(b"mp4"),
            )

        self.assertEqual(events[:5], ["start", ("sleep", 7), "connect", ("sleep", 11), "prepare"])

    def test_collect_fresh_engine_rollouts_retries_prepare_timeout_with_new_engine(self):
        events = []
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]

        class FakeProcess:
            def __init__(self, pid):
                self.pid = pid
                self.terminated = False
                self.waited = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True
                events.append(("terminate", self.pid))

            def wait(self, timeout=None):
                self.waited = True
                events.append(("wait", self.pid))

        processes = []
        bridges = [FakeBridge(), FakeBridge()]

        def start_engine_func(game_dir, headless):
            process = FakeProcess(4000 + len(processes))
            processes.append(process)
            events.append(("start", process.pid))
            return process

        def connect_func(host, port, timeout, deadline_seconds):
            bridge = bridges.pop(0)
            events.append(("connect", len(processes)))
            return bridge

        prepare_calls = []

        def prepare_func(bridge, timeout, poll_delay):
            prepare_calls.append(bridge)
            if len(prepare_calls) == 1:
                raise TimeoutError("Science Birds did not reach PLAYING before timeout")
            return bridge.get_game_state()

        def capture_rollout(bridge, output_dir, **kwargs):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "metadata.json").write_text(json.dumps({"frame_count": 1}), encoding="utf-8")
            return {"frame_count": 1}

        with TemporaryDirectory() as tmp:
            manifest = collect_fresh_engine_rollouts(
                Path(tmp),
                actions,
                game_dir=Path("game"),
                host="127.0.0.1",
                port=2004,
                agent_id=28888,
                speed=1,
                connect_timeout=1,
                read_timeout=2,
                prepare_timeout=3,
                frame_height=480,
                fast=True,
                headless=False,
                target_fps=1,
                duration_seconds=1,
                ui_level=None,
                ui_settle_seconds=0,
                fresh_engine_attempts=2,
                start_engine_func=start_engine_func,
                connect_func=connect_func,
                prepare_func=prepare_func,
                capture_rollout=capture_rollout,
                video_runner=lambda command, check, stdout, stderr: Path(command[-1]).write_bytes(b"mp4"),
            )

        self.assertEqual(manifest["attempt_count"], 1)
        self.assertEqual(manifest["accepted_rollout_count"], 0)
        self.assertEqual(manifest["rollout_count"], 0)
        self.assertEqual(len(processes), 2)
        self.assertTrue(all(process.terminated and process.waited for process in processes))
        self.assertIn(("start", 4000), events)
        self.assertIn(("start", 4001), events)

    def test_collect_fresh_engine_rollouts_anchors_slingshot_relative_actions_from_symbolic_state(self):
        actions = [
            {
                "action_type": "drag_hold_release",
                "coordinate_frame": "slingshot_relative",
                "drag_start": [300, 220],
                "drag_release": [-50, 40],
                "tapTime": 0,
            }
        ]
        bridge = FakeBridge()
        bridge.symbolic_state = [
            {
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"label": "Slingshot"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[100, 150], [100, 190], [140, 190], [140, 150]]],
                        },
                    }
                ]
            }
        ]
        processes = []

        class FakeProcess:
            def __init__(self, pid):
                self.pid = pid
                self.terminated = False
                self.waited = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                self.waited = True

        def start_engine_func(game_dir, headless):
            process = FakeProcess(2000 + len(processes))
            processes.append(process)
            return process

        def connect_func(host, port, timeout, deadline_seconds):
            return bridge

        def capture_rollout(bridge, output_dir, **kwargs):
            output_dir.mkdir(parents=True, exist_ok=True)
            payload = {"frame_count": 1, "action": kwargs["action"]}
            (output_dir / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")
            return {"frame_count": 1}

        with TemporaryDirectory() as tmp:
            manifest = collect_fresh_engine_rollouts(
                Path(tmp),
                actions,
                game_dir=Path("game"),
                host="127.0.0.1",
                port=2004,
                agent_id=28888,
                speed=1,
                connect_timeout=1,
                read_timeout=2,
                prepare_timeout=3,
                frame_height=480,
                fast=True,
                headless=False,
                target_fps=1,
                duration_seconds=1,
                ui_level=1,
                ui_settle_seconds=0,
                start_engine_func=start_engine_func,
                connect_func=connect_func,
                prepare_func=lambda bridge, timeout, poll_delay: bridge.get_game_state(),
                capture_rollout=capture_rollout,
                select_level_func=lambda level: None,
            )

            saved_metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["rollouts"][0]["slingshot_reference"], {"gameX": 118, "gameY": 315, "canvasX": 118, "canvasY": 164})
        self.assertEqual(manifest["rollouts"][0]["action"]["drag_start"], [118, 315])
        self.assertEqual(bridge.shots[0], (68, 204, 0, True, 600))
        self.assertEqual(saved_metadata["slingshot_reference"], {"gameX": 118, "gameY": 315, "canvasX": 118, "canvasY": 164})
        self.assertTrue(all(process.terminated and process.waited for process in processes))

    def test_collect_fresh_engine_rollouts_stops_engine_when_disconnect_raises(self):
        class DisconnectFailBridge(FakeBridge):
            def disconnect(self):
                raise RuntimeError("disconnect failed")

        class FakeProcess:
            pid = 3000

            def __init__(self):
                self.terminated = False
                self.waited = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                self.waited = True

        bridge = DisconnectFailBridge()
        process = FakeProcess()

        def capture_rollout(bridge, output_dir, **kwargs):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "metadata.json").write_text(json.dumps({"frame_count": 1}), encoding="utf-8")
            return {"frame_count": 1}

        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "disconnect failed"):
                collect_fresh_engine_rollouts(
                    Path(tmp),
                    [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}],
                    game_dir=Path("game"),
                    host="127.0.0.1",
                    port=2004,
                    agent_id=28888,
                    speed=1,
                    connect_timeout=1,
                    read_timeout=2,
                    prepare_timeout=3,
                    frame_height=480,
                    fast=True,
                    headless=False,
                    target_fps=1,
                    duration_seconds=1,
                    ui_level=None,
                    ui_settle_seconds=0,
                    start_engine_func=lambda game_dir, headless: process,
                    connect_func=lambda host, port, timeout, deadline_seconds: bridge,
                    prepare_func=lambda bridge, timeout, poll_delay: bridge.get_game_state(),
                    capture_rollout=capture_rollout,
                )

        self.assertTrue(process.terminated)
        self.assertTrue(process.waited)

    def test_write_action_plan_dry_run_writes_actions_without_bridge(self):
        with TemporaryDirectory() as tmp:
            path = write_action_plan(Path(tmp), count=3)

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["action_count"], 3)
            self.assertEqual(len(payload["actions"]), 3)
            self.assertEqual(payload["actions"][0]["action_type"], "drag_hold_release")

    def test_load_actions_from_action_log_returns_exact_logged_actions(self):
        logged_actions = [
            {
                "action_type": "drag_hold_release",
                "coordinate_frame": "slingshot_relative",
                "drag_start": [97, 227],
                "drag_release": [-80, 7],
                "tapTime": 0,
                "holdTime": 1000,
                "slingshot_reference": {"gameX": 97, "gameY": 227, "canvasX": 97, "canvasY": 252},
            },
            {"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 45},
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "action_log.json"
            path.write_text(
                json.dumps(
                    {
                        "episode_dir": "original_episode",
                        "trial_count": 2,
                        "trials": [
                            {"shot_name": "shot_001", "action": logged_actions[0]},
                            {"shot_name": "shot_002", "action": logged_actions[1]},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            actions = load_actions_from_action_log(path)

        self.assertEqual(actions, logged_actions)

    def test_load_actions_from_action_log_rejects_malformed_log(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "action_log.json"
            path.write_text(json.dumps({"trials": [{"shot_name": "shot_001"}]}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "trial 1 is missing action"):
                load_actions_from_action_log(path)

    def test_main_wires_actions_from_log_without_reanchoring(self):
        logged_action = {
            "action_type": "drag_hold_release",
            "coordinate_frame": "slingshot_relative",
            "drag_start": [97, 227],
            "drag_release": [-80, 7],
            "tapTime": 0,
            "holdTime": 1000,
            "slingshot_reference": {"gameX": 97, "gameY": 227, "canvasX": 97, "canvasY": 252},
        }
        with TemporaryDirectory() as tmp:
            action_log = Path(tmp) / "action_log.json"
            action_log.write_text(
                json.dumps({"trial_count": 1, "trials": [{"shot_name": "shot_001", "action": logged_action}]}),
                encoding="utf-8",
            )
            args = ["collect_rollouts.py", "--output-dir", tmp, "--actions-from-log", str(action_log), "--no-prepare"]

            with (
                patch("sys.argv", args),
                patch("scripts.collect_rollouts.connect_or_start_engine", return_value=(FakeBridge(), None)),
                patch("scripts.collect_rollouts.collect_rollouts", return_value={"rollout_count": 1}) as collect,
            ):
                main()

        self.assertEqual(collect.call_args.args[2], [logged_action])
        self.assertFalse(collect.call_args.kwargs["anchor_actions"])

    def test_main_wires_desktop_fresh_engine_baseline_capture(self):
        with TemporaryDirectory() as tmp:
            args = [
                "collect_rollouts.py",
                "--output-dir",
                tmp,
                "--count",
                "1",
                "--fps",
                "1",
                "--duration",
                "0.01",
                "--capture-source",
                "desktop",
                "--fresh-engine-per-rollout",
                "--ui-level",
                "1",
            ]

            with patch("sys.argv", args), patch(
                "scripts.collect_rollouts.collect_fresh_engine_rollouts", return_value={"rollout_count": 1}
            ) as fresh:
                main()

        self.assertEqual(fresh.call_count, 1)
        self.assertIsNotNone(fresh.call_args.kwargs["pre_shot_grabber"])
        self.assertEqual(fresh.call_args.kwargs["capture_rollout"].__name__, "capture_desktop_rollout")
        self.assertEqual(fresh.call_args.kwargs["engine_settle_seconds"], 20.0)
        self.assertEqual(fresh.call_args.kwargs["agent_settle_seconds"], 45.0)

    def test_connect_or_start_engine_auto_starts_engine_after_connection_refusal(self):
        class FakeProcess:
            pid = 4321

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                return None

        args = type(
            "Args",
            (),
            {
                "host": "127.0.0.1",
                "port": 2004,
                "read_timeout": 3,
                "connect_timeout": 4,
                "game_headless": False,
            },
        )()
        bridge = FakeBridge()
        process = FakeProcess()

        with (
            patch("scripts.collect_rollouts.connect_with_retry", side_effect=[RuntimeError("connection refused"), bridge]),
            patch("scripts.collect_rollouts.start_engine", return_value=process) as start,
        ):
            result_bridge, result_process = __import__("scripts.collect_rollouts", fromlist=["connect_or_start_engine"]).connect_or_start_engine(args)

        self.assertIs(result_bridge, bridge)
        self.assertIs(result_process, process)
        self.assertEqual(start.call_count, 1)

    def test_connect_or_start_engine_forwards_custom_engine_ports(self):
        class FakeProcess:
            pid = 4321

            def poll(self):
                return None

        args = type(
            "Args",
            (),
            {
                "host": "127.0.0.1",
                "port": 2014,
                "read_timeout": 3,
                "connect_timeout": 4,
                "game_dir": Path("/tmp/novphy-worker-engine"),
                "game_headless": True,
                "engine_agent_port": 2014,
                "engine_game_port": 9011,
            },
        )()
        bridge = FakeBridge()
        process = FakeProcess()

        with (
            patch("scripts.collect_rollouts.connect_with_retry", side_effect=[RuntimeError("connection refused"), bridge]),
            patch("scripts.collect_rollouts.start_engine", return_value=process) as start,
        ):
            result_bridge, result_process = __import__("scripts.collect_rollouts", fromlist=["connect_or_start_engine"]).connect_or_start_engine(args)

        self.assertIs(result_bridge, bridge)
        self.assertIs(result_process, process)
        start.assert_called_once_with(Path("/tmp/novphy-worker-engine"), True, agent_port=2014, game_port=9011)

    def test_main_stops_explicit_started_engine_when_connection_fails(self):
        class FakeProcess:
            pid = 2468

            def __init__(self):
                self.terminated = False
                self.waited = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                self.waited = True

        process = FakeProcess()
        with TemporaryDirectory() as tmp:
            args = [
                "collect_rollouts.py",
                "--output-dir",
                tmp,
                "--count",
                "1",
                "--start-engine",
            ]

            with (
                patch("sys.argv", args),
                patch("scripts.collect_rollouts.start_engine", return_value=process),
                patch("scripts.collect_rollouts.connect_with_retry", side_effect=RuntimeError("connection refused")),
            ):
                with self.assertRaisesRegex(RuntimeError, "connection refused"):
                    main()

        self.assertTrue(process.terminated)
        self.assertTrue(process.waited)

    def test_stop_owned_engine_kills_process_after_terminate_timeout(self):
        class SlowProcess:
            pid = 4321

            def __init__(self):
                self.terminated = False
                self.killed = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                if not self.killed:
                    raise TimeoutError("still running")

            def kill(self):
                self.killed = True

        process = SlowProcess()

        stop_owned_engine(process)

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)

    def test_stop_owned_engine_terminates_process_group_for_started_engine(self):
        class GroupProcess:
            pid = 5432
            novphy_process_group = True

            def __init__(self):
                self.terminated = False
                self.killed = False
                self.waited = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.killed = True

            def wait(self, timeout=None):
                self.waited = True

        process = GroupProcess()
        with patch("scripts.collect_rollouts.os.getpgid", return_value=9876), patch("scripts.collect_rollouts.os.killpg") as killpg:
            stop_owned_engine(process)

        killpg.assert_called_once_with(9876, signal.SIGTERM)
        self.assertTrue(process.waited)
        self.assertFalse(process.terminated)
        self.assertFalse(process.killed)

    def test_stop_owned_engine_falls_back_to_process_when_group_lookup_fails(self):
        class MissingGroupProcess:
            pid = 6543
            novphy_process_group = True

            def __init__(self):
                self.terminated = False
                self.waited = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                self.waited = True

        process = MissingGroupProcess()
        with patch("scripts.collect_rollouts.os.getpgid", side_effect=ProcessLookupError), patch("scripts.collect_rollouts.os.killpg") as killpg:
            stop_owned_engine(process)

        killpg.assert_not_called()
        self.assertTrue(process.terminated)
        self.assertTrue(process.waited)

    def test_stop_owned_engine_escalates_process_group_after_timeout(self):
        class SlowGroupProcess:
            pid = 7654
            novphy_process_group = True

            def __init__(self):
                self.wait_calls = 0
                self.killed = False

            def poll(self):
                return None

            def kill(self):
                self.killed = True

            def wait(self, timeout=None):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    raise TimeoutError("still running")

        process = SlowGroupProcess()
        with patch("scripts.collect_rollouts.os.getpgid", return_value=8765), patch("scripts.collect_rollouts.os.killpg") as killpg:
            stop_owned_engine(process)

        self.assertEqual(killpg.call_args_list[0].args, (8765, signal.SIGTERM))
        self.assertEqual(killpg.call_args_list[1].args, (8765, signal.SIGKILL))
        self.assertEqual(process.wait_calls, 2)
        self.assertFalse(process.killed)

    def test_main_reports_auto_start_failure_without_traceback(self):
        args = ["collect_rollouts.py", "--output-dir", "data/rollout-plan-debug", "--count", "3"]
        stderr = io.StringIO()

        with (
            patch("sys.argv", args),
            patch("scripts.collect_rollouts.connect_with_retry", side_effect=RuntimeError("connection refused")),
            patch("scripts.collect_rollouts.start_engine", side_effect=FileNotFoundError("missing jar")),
        ):
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("could not start the local engine", stderr.getvalue())
        self.assertIn("--dry-run", stderr.getvalue())

    def test_main_preflights_output_dir_before_starting_engine(self):
        args = ["collect_rollouts.py", "--output-dir", "/data/collect_rollouts_debug", "--count", "1"]
        stderr = io.StringIO()

        with (
            patch("sys.argv", args),
            patch("scripts.collect_rollouts.ensure_output_dir", side_effect=PermissionError("denied")) as ensure,
            patch("scripts.collect_rollouts.start_engine") as start,
            patch("scripts.collect_rollouts.connect_with_retry") as connect,
        ):
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main()

        self.assertEqual(raised.exception.code, 2)
        ensure.assert_called_once()
        start.assert_not_called()
        connect.assert_not_called()
        self.assertIn("Cannot write output directory", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
