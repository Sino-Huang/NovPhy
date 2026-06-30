import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from scripts.collect_rollouts import (
    _action_guide_points,
    _launch_guide_points,
    action_to_shot,
    capture_desktop_rollout,
    collect_fresh_engine_rollouts,
    collect_rollouts,
    build_parser,
    format_action_overlay_text,
    prepare_rollout_video_frames,
    slingshot_reference_point_from_symbolic_state,
    main,
    select_level_in_display,
    stop_owned_engine,
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

            self.assertEqual(manifest["rollout_count"], 2)
            self.assertEqual(manifest["rollouts"][0]["shoot_response"], 1)
            self.assertEqual(manifest["rollouts"][0]["frame_count"], 2)
            self.assertTrue((Path(tmp) / "shot_001" / "metadata.json").is_file())
            saved_manifest = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_manifest["rollout_count"], 2)

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
        self.assertEqual(manifest["rollout_count"], 2)

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

        self.assertEqual(events[:4], ["baseline", "state", "score", "shoot"])
        self.assertEqual(manifest["rollout_count"], 1)
        self.assertEqual(metadata["frames"][0]["pre_shot_delta"]["changed_pixel_count"], 1)
        self.assertEqual(metadata["frames"][1]["pre_shot_delta"]["changed_pixel_count"], 2)
        self.assertEqual(metadata["max_pre_shot_delta"], 2)
        self.assertEqual(metadata["max_pre_shot_delta_bbox"], [1, 0, 3, 1])
        self.assertEqual(metadata["pre_shot_path"], str(Path(tmp) / "shot_001" / "pre_shot.png"))
        self.assertEqual(metadata["pre_shot_sample"], {"state": "PLAYING", "score": 200})
        self.assertEqual(metadata["frames"][1]["frame_delta"]["changed_pixel_count"], 1)

    def test_collect_rollouts_writes_review_mp4_for_each_shot(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]
        runner_calls = []

        def capture_rollout(bridge, output_dir, **kwargs):
            from PIL import Image

            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True)
            for index in range(2):
                image = Image.new("RGB", (4, 3), (10 + index, 20, 30))
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
        self.assertEqual(manifest["rollout_count"], 3)
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

        self.assertEqual(manifest["rollout_count"], 2)
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
