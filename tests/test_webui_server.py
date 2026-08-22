import base64
from io import BytesIO
import json
import subprocess
import tarfile
import threading
import time
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from xml.etree import ElementTree as ET

from src.webui.bridge import GameState, Screenshot
from src.webui.server import AppState, build_arg_parser, create_handler
from tests.test_physics_v2_review import collision_capture, write_probe_plan


ROOT = Path(__file__).resolve().parents[1]


class FakeBridge:
    connected = True

    def __init__(self):
        self.shots = []
        self.loaded = []
        self.loaded_next = []
        self.restarted = 0
        self.configured = 0
        self.speeds = []
        self.ready_calls = 0
        self.novelty_calls = 0
        self.zoomed_out = 0
        self.states = [GameState.PLAYING]
        self.number_of_levels = 20
        self.symbolic_state = [
            {
                "features": [
                    {
                        "geometry": {},
                        "type": "Feature",
                        "properties": {"id": "-166", "label": "Ground", "yindex": 286, "colormap": []},
                    },
                    {
                        "geometry": {
                            "coordinates": [[[210.0, 33.0], [210.0, 100.0], [232.0, 100.0], [232.0, 33.0]]],
                            "type": "Polygon",
                        },
                        "type": "Feature",
                        "properties": {"currentLife": 3.402823e38, "id": "-302", "label": "Slingshot", "colormap": []},
                    },
                ]
            }
        ]

    def configure(self, agent_id, mode):
        self.configured += 1
        return (0, 0, 0)

    def set_speed(self, speed):
        self.speeds.append(speed)
        return 1

    def screenshot(self):
        return Screenshot(width=640, height=480, rgb=bytes([1, 2, 3, 4, 5, 6]) * (640 * 480))

    def get_game_state(self):
        if len(self.states) > 1:
            return self.states.pop(0)
        return self.states[0]

    def get_current_score(self):
        return 123

    def get_current_level(self):
        return 2

    def get_number_of_levels(self):
        return self.number_of_levels

    def get_symbolic_state_without_screenshot(self):
        return self.symbolic_state

    def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
        if release_time:
            self.shots.append((x, y, tap_time, fast, release_time))
        else:
            self.shots.append((x, y, tap_time, fast))
        return 1

    def load_level(self, level):
        self.loaded.append(level)
        return 1

    def load_next_available_level(self):
        self.loaded_next.append(True)
        return 1

    def restart_level(self):
        self.restarted += 1
        return 1

    def fully_zoom_out(self):
        self.zoomed_out += 1
        return 1

    def ready_for_new_set(self):
        self.ready_calls += 1
        return (9000, 60000, 20, 5, 1, 0, 0)

    def get_novelty_info(self):
        self.novelty_calls += 1
        return 0


class FakePhysicsV2Bridge:
    connected = True

    def __init__(self):
        self.requests = 0

    def connect(self):
        self.connected = True

    def get_physics_capture_v2(self):
        self.requests += 1
        return SimpleNamespace(record=collision_capture())


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.app = AppState(root=Path("/tmp/nonexistent-webui-root"))
        self.app.bridge = FakeBridge()
        self.review_tmp = TemporaryDirectory()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(self.app))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.review_tmp.cleanup()

    def request(self, method, path, payload=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        body = None if payload is None else json.dumps(payload)
        connection.request(method, path, body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = json.loads(response.read().decode("utf-8"))
        connection.close()
        return response.status, data

    def request_bytes(self, path):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("GET", path)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, data

    def write_review_archive(self, stage: Path, *, mutate=None, omitted: set[str] | None = None, extras: dict[str, bytes] | None = None) -> Path:
        stage.mkdir(parents=True, exist_ok=True)
        required = {
            "game_playing_interface.jar": b"jar",
            "9001.x86_64": b"wrapper",
            "9001-player.x86_64": b"player",
            "config.xml": b"<evaluation />",
            "9001_Data/placeholder": b"data",
        }
        provenance = {
            "schema_version": "novphy_physics_player_stage_v1",
            "unity": {"version": "2019.4.41f2"},
            "capture": {"schema_version": "physics_capture_v2_engine_v1", "protocol_version": 1},
            "files": {name: len(content) for name, content in required.items()},
        }
        if mutate is not None:
            mutate(provenance)
        members = dict(required)
        members.update(extras or {})
        for name in omitted or set():
            members.pop(name, None)
        members["provenance.json"] = json.dumps(provenance).encode("utf-8")
        archive = stage / "novphy-physics-player-2019.4.41f2.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            for name, content in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                bundle.addfile(info, BytesIO(content))
        review_levels = stage / "review-levels"
        review_levels.mkdir()
        for name in ("training.xml", "support-ready.xml"):
            (review_levels / name).write_text("<Level />", encoding="utf-8")
        review_manifests = stage / "review-manifests"
        review_manifests.mkdir()
        scenarios = []
        for index, name in enumerate(("training", "support-ready")):
            manifest_path = review_manifests / f"{name}.json"
            manifest_path.write_text(
                json.dumps({"identity": f"scenario-manifest-v1:{index}"}),
                encoding="utf-8",
            )
            scenarios.append({
                "scenario_id": f"scenario-{index}",
                "scenario_manifest_reference": f"removed-evidence/manifests/{name}.json",
                "scenario_template_identity": f"scenario-template-v1:{index}",
                "level_instance_identity": f"level-instance-v1:{index}",
                "scenario_lineage_identity": f"scenario-lineage-v1:{index}",
            })
        (stage / "probe-plan.json").write_text(
            json.dumps({"identity": "physics-v2-probe-plan-v1:review", "scenarios": scenarios}),
            encoding="utf-8",
        )
        return archive

    def test_status_reports_connected_fake_bridge(self):
        status, data = self.request("GET", "/api/status")

        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertTrue(data["connected"])

    def test_webui_exposes_guided_review_controls_and_world_space_playback(self):
        status, content = self.request_bytes("/")

        self.assertEqual(status, 200)
        page = content.decode("utf-8")
        self.assertIn('id="physicsReviewPanel"', page)
        self.assertIn('id="reviewGoal"', page)
        self.assertIn('id="worldCanvas"', page)
        self.assertIn('id="reviewTimeline"', page)

    def test_physics_v2_review_stages_explores_freezes_and_replays_exact_action(self):
        self.app.physics_v2_review = True
        self.app.review_output_root = Path(self.review_tmp.name)
        self.app.review_probe_plan_path = write_probe_plan(
            Path(self.review_tmp.name) / "stage"
        )
        self.app.physics_bridge = FakePhysicsV2Bridge()
        resets = []
        self.app.review_reset_callback = lambda goal: resets.append(goal)
        action = {
            "action_type": "drag_hold_release",
            "coordinate_frame": "slingshot_relative",
            "drag_start": [97, 227],
            "drag_release": [-80, 8],
            "tapTime": 0,
            "holdTime": 1000,
            "frame_height": 480,
        }

        status, loaded = self.request(
            "POST", "/api/physics-v2-review/load-goal", {"goal": "collision"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(loaded["level"], 1)
        self.assertEqual(resets, ["collision"])

        status, staged = self.request(
            "POST", "/api/physics-v2-review/stage", {"goal": "collision", "action": action}
        )
        self.assertEqual(status, 200)
        self.assertEqual(staged["session"]["state"], "staged")

        status, explored = self.request("POST", "/api/physics-v2-review/explore")
        self.assertEqual(status, 200)
        self.assertEqual(explored["session"]["state"], "explored")
        self.assertEqual(self.app.bridge.shots[-1], (17, 260, 0, True, 1000))
        self.assertFalse(explored["session"]["eligible_for_issue_44"])

        self.assertEqual(
            self.request("POST", "/api/physics-v2-review/freeze")[1]["session"]["state"],
            "frozen",
        )
        status, replayed = self.request("POST", "/api/physics-v2-review/replay")
        self.assertEqual(status, 200)
        self.assertEqual(replayed["session"]["state"], "demonstrated")
        self.assertTrue(replayed["session"]["eligible_for_issue_44_review"])
        self.assertEqual(self.app.physics_bridge.requests, 2)
        self.assertEqual(resets, ["collision", "collision"])
        status, steps = self.request("GET", "/api/physics-v2-review/steps?start=1&count=1")
        self.assertEqual(status, 200)
        self.assertEqual(steps["total"], 2)
        self.assertEqual([step["fixed_step"] for step in steps["steps"]], [1])

    def test_physics_v2_review_waits_for_stable_slingshot_before_shooting(self):
        class MovingSlingshotBridge(FakeBridge):
            def __init__(self):
                super().__init__()
                self.positions = [250.0, 225.0, 210.0, 210.0]
                self.position_observations = 0
                self.observations_at_shot = None

            def get_symbolic_state_without_screenshot(self):
                index = min(self.position_observations, len(self.positions) - 1)
                min_x = self.positions[index]
                self.position_observations += 1
                return [{
                    "features": [{
                        "geometry": {
                            "coordinates": [[
                                [min_x, 33.0],
                                [min_x, 100.0],
                                [min_x + 22.0, 100.0],
                                [min_x + 22.0, 33.0],
                            ]],
                            "type": "Polygon",
                        },
                        "type": "Feature",
                        "properties": {"label": "Slingshot"},
                    }],
                }]

            def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
                self.observations_at_shot = self.position_observations
                return super().shoot(x, y, tap_time, fast, release_time)

        self.app.physics_v2_review = True
        self.app.review_output_root = Path(self.review_tmp.name)
        self.app.review_probe_plan_path = write_probe_plan(
            Path(self.review_tmp.name) / "stage"
        )
        self.app.physics_bridge = FakePhysicsV2Bridge()
        self.app.bridge = MovingSlingshotBridge()
        self.app.readiness_poll_delay = 0
        self.app.stage_physics_v2_review("collision", {
            "action_type": "drag_hold_release",
            "coordinate_frame": "slingshot_relative",
            "drag_start": [97, 227],
            "drag_release": [-80, 8],
            "tapTime": 0,
            "holdTime": 1000,
            "frame_height": 480,
        })

        self.app.run_physics_v2_review(replay=False)

        self.assertEqual(self.app.bridge.observations_at_shot, 4)

    def test_packaged_physics_v2_review_action_can_be_staged(self):
        self.app.root = ROOT
        self.app.physics_v2_review = True
        self.app.review_output_root = Path(self.review_tmp.name)
        action = {
            "action_type": "drag_hold_release",
            "coordinate_frame": "slingshot_relative",
            "drag_start": [127, 223],
            "drag_release": [-5, 23],
            "tapTime": 0,
            "holdTime": 600,
            "frame_height": 480,
        }

        status, staged = self.request(
            "POST", "/api/physics-v2-review/stage", {"goal": "collision", "action": action}
        )

        self.assertEqual(status, 200, staged)
        self.assertEqual(staged["session"]["state"], "staged")

    def test_default_readiness_timeout_allows_slow_generated_levels(self):
        self.assertGreaterEqual(AppState().readiness_timeout, 60)

    def test_physics_v2_review_is_an_explicit_cli_mode(self):
        defaults = build_arg_parser().parse_args([])
        review = build_arg_parser().parse_args(["--physics-v2-review", "--physics-port", "2015"])

        self.assertFalse(defaults.physics_v2_review)
        self.assertTrue(review.physics_v2_review)
        self.assertEqual(review.physics_port, 2015)

    def test_review_preflight_accepts_the_staged_archive_without_a_pin_file(self):
        with TemporaryDirectory() as temporary:
            stage = Path(temporary)
            self.write_review_archive(stage)
            app = AppState(physics_v2_review=True, review_stage=stage)

            self.assertEqual(app.preflight_errors(), [])

    def test_review_preflight_reports_missing_source_bound_review_inputs(self):
        with TemporaryDirectory() as temporary:
            stage = Path(temporary)
            (stage / "novphy-physics-player-2019.4.41f2.tar.gz").write_bytes(b"archive")
            app = AppState(physics_v2_review=True, review_stage=stage)

            errors = app.preflight_errors()

            self.assertTrue(any("probe-plan.json" in error for error in errors))
            self.assertTrue(any("review-manifests/training.json" in error for error in errors))

    def test_review_runtime_validates_declared_archive_provenance(self):
        with TemporaryDirectory() as temporary:
            stage = Path(temporary) / "stage"
            self.write_review_archive(stage)
            app = AppState(physics_v2_review=True, review_stage=stage)
            runtime = app.prepare_physics_v2_review_runtime()
            self.assertTrue((runtime / "game_playing_interface.jar").is_file())
            level_root = runtime / "9001_Data/StreamingAssets/Levels/novelty_level_0/type2/Levels"
            self.assertTrue((level_root / "training.xml").is_file())
            config = ET.parse(runtime / "config.xml")
            configured_paths = [
                node.attrib["level_path"] for node in config.findall(".//game_levels")
            ]
            self.assertEqual(configured_paths, [
                "9001_Data/StreamingAssets/Levels/novelty_level_0/type2/Levels/training.xml",
                "9001_Data/StreamingAssets/Levels/novelty_level_0/type2/Levels/support-ready.xml",
            ])
            app.review_runtime_temporary.cleanup()

    def test_review_goal_restart_places_the_bound_scenario_first(self):
        with TemporaryDirectory() as temporary:
            stage = Path(temporary) / "stage"
            self.write_review_archive(stage)
            resets = []
            app = AppState(physics_v2_review=True, review_stage=stage)
            app.bridge = FakeBridge()
            app.review_reset_callback = lambda goal: resets.append(goal)

            app.load_physics_v2_review_goal("support change")
            runtime = app.prepare_physics_v2_review_runtime()
            config = ET.parse(runtime / "config.xml")
            configured_paths = [
                node.attrib["level_path"] for node in config.findall(".//game_levels")
            ]

            self.assertEqual(resets, ["support change"])
            self.assertTrue(configured_paths[0].endswith("/support-ready.xml"))
            app.review_runtime_temporary.cleanup()

    def test_review_runtime_rejects_stale_or_incomplete_declared_provenance(self):
        cases = {
            "schema": lambda value: value.update(schema_version="unsupported"),
            "unity": lambda value: value["unity"].update(version="2020.1"),
            "capture": lambda value: value["capture"].update(schema_version="physics_capture_v1"),
            "protocol": lambda value: value["capture"].update(protocol_version=2),
            "inventory": lambda value: value["files"].pop("9001-player.x86_64"),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), TemporaryDirectory() as temporary:
                stage = Path(temporary) / "stage"
                self.write_review_archive(stage, mutate=mutate)
                app = AppState(physics_v2_review=True, review_stage=stage)
                with self.assertRaisesRegex(ValueError, "provenance|unsupported|required files|inventory"):
                    app.prepare_physics_v2_review_runtime()
                self.assertIsNone(app.review_runtime_dir)

        with TemporaryDirectory() as temporary:
            stage = Path(temporary) / "stage"
            self.write_review_archive(stage, omitted={"9001-player.x86_64"})
            app = AppState(physics_v2_review=True, review_stage=stage)
            with self.assertRaisesRegex(ValueError, "inventory does not match|required files"):
                app.prepare_physics_v2_review_runtime()

    def test_review_runtime_rejects_an_unexpected_extra_file(self):
        with TemporaryDirectory() as temporary:
            stage = Path(temporary) / "stage"
            self.write_review_archive(stage, extras={"unexpected.txt": b"extra"})
            app = AppState(physics_v2_review=True, review_stage=stage)
            with self.assertRaisesRegex(ValueError, "inventory does not match"):
                app.prepare_physics_v2_review_runtime()

    def test_review_runtime_rejects_a_declared_size_mismatch(self):
        with TemporaryDirectory() as temporary:
            stage = Path(temporary) / "stage"

            def wrong_size(provenance):
                provenance["files"]["9001-player.x86_64"] += 1

            self.write_review_archive(stage, mutate=wrong_size)
            app = AppState(physics_v2_review=True, review_stage=stage)
            with self.assertRaisesRegex(ValueError, "inventory does not match"):
                app.prepare_physics_v2_review_runtime()

    def test_frame_returns_base64_rgb_and_metadata(self):
        status, data = self.request("GET", "/api/frame")

        self.assertEqual(status, 200)
        self.assertEqual(data["width"], 640)
        self.assertEqual(data["height"], 480)
        self.assertEqual(base64.b64decode(data["rgbBase64"]), bytes([1, 2, 3, 4, 5, 6]) * (640 * 480))
        self.assertEqual(data["state"]["name"], "PLAYING")
        self.assertEqual(data["score"], 123)
        self.assertEqual(data["numberOfLevels"], 20)
        center = data["trajectorySlingCenter"]
        expected_pixels_per_world_unit = 67 / 2.055
        self.assertAlmostEqual(center["canvasX"], 210 + 0.335 * expected_pixels_per_world_unit)
        self.assertAlmostEqual(center["canvasY"], 33 + 0.25 * expected_pixels_per_world_unit)
        self.assertAlmostEqual(center["gameX"], center["canvasX"])
        self.assertAlmostEqual(center["gameY"], 480 - center["canvasY"])
        self.assertAlmostEqual(center["pixelsPerWorldUnit"], expected_pixels_per_world_unit)

    def test_frame_exposes_configured_level_camera_width_for_trajectory(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            game_dir = root / "sciencebirdsgames" / "Linux"
            level = game_dir / "levels" / "level-1.xml"
            level.parent.mkdir(parents=True)
            level.write_text(
                """
                <Level>
                  <Camera x="0" y="2" minWidth="20" maxWidth="30" />
                </Level>
                """,
                encoding="utf-8",
            )
            (game_dir / "config.xml").write_text(
                """
                <evaluation>
                  <trials>
                    <trial>
                      <game_level_set>
                        <game_levels level_path="levels/level-1.xml" />
                      </game_level_set>
                    </trial>
                  </trials>
                </evaluation>
                """,
                encoding="utf-8",
            )
            app = AppState(root=root)
            app.bridge = FakeBridge()
            app.bridge.number_of_levels = 1
            app.bridge.states = [GameState.PLAYING]
            app.bridge.get_current_level = lambda: 1
            server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
                connection.request("GET", "/api/frame", headers={"Content-Type": "application/json"})
                response = connection.getresponse()
                data = json.loads(response.read().decode("utf-8"))
                connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

        self.assertEqual(response.status, 200)
        self.assertEqual(data["trajectoryWorldWidth"], 30.0)

    def test_frame_falls_back_to_configured_level_count_when_socket_reports_zero(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            game_dir = root / "sciencebirdsgames" / "Linux"
            game_dir.mkdir(parents=True)
            (game_dir / "config.xml").write_text(
                """
                <evaluation>
                  <trials>
                    <trial>
                      <game_level_set>
                        <game_levels level_path="level-1.xml" />
                        <game_levels level_path="level-2.xml" />
                      </game_level_set>
                    </trial>
                  </trials>
                </evaluation>
                """,
                encoding="utf-8",
            )
            app = AppState(root=root)
            app.bridge = FakeBridge()
            app.bridge.number_of_levels = 0
            server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
                connection.request("GET", "/api/frame", headers={"Content-Type": "application/json"})
                response = connection.getresponse()
                data = json.loads(response.read().decode("utf-8"))
                connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

        self.assertEqual(response.status, 200)
        self.assertEqual(data["numberOfLevels"], 2)

    def test_shot_validates_and_calls_bridge(self):
        status, data = self.request("POST", "/api/shot", {"x": 10, "y": 20, "tapTime": 30, "fast": True})

        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(self.app.bridge.shots, [(10, 459, 30, True)])

    def test_shot_uses_latest_frame_height_for_bridge_y_conversion(self):
        self.request("GET", "/api/frame")

        status, data = self.request("POST", "/api/shot", {"x": 1, "y": 0})

        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(self.app.bridge.shots, [(1, 479, 0, False)])

    def test_shot_passes_release_time_to_bridge(self):
        status, data = self.request("POST", "/api/shot", {"x": 10, "y": 20, "tapTime": 30, "releaseTime": 120, "fast": True})

        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(self.app.bridge.shots, [(10, 459, 30, True, 120)])

    def test_async_shot_returns_before_bridge_ack(self):
        started = threading.Event()
        release = threading.Event()

        class SlowBridge(FakeBridge):
            def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
                started.set()
                release.wait(timeout=2)
                return super().shoot(x, y, tap_time=tap_time, fast=fast, release_time=release_time)

        self.app.bridge = SlowBridge()

        before = time.monotonic()
        status, data = self.request("POST", "/api/shot", {"x": 10, "y": 20, "fast": True, "async": True})
        elapsed = time.monotonic() - before

        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertTrue(data["scheduled"])
        self.assertLess(elapsed, 1)
        self.assertTrue(started.wait(timeout=1))
        self.assertEqual(self.app.bridge.shots, [])
        release.set()

        deadline = time.monotonic() + 2
        while not self.app.bridge.shots and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.app.bridge.shots, [(10, 459, 0, True)])

    def test_agent_action_translates_relative_drag_release_without_shooting(self):
        payload = {
            "action": {
                "action_type": "drag_release",
                "coordinate_frame": "slingshot_relative",
                "drag_start": [300, 220],
                "drag_release": [-50, 40],
                "tapTime": 70,
            },
            "fast": True,
        }

        status, data = self.request("POST", "/api/agent-action", payload)

        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(data["shot"], {"x": 250, "y": 180, "tapTime": 70, "fast": True})
        self.assertEqual(data["action"]["drag_start"], [300, 220])
        self.assertEqual(data["action"]["drag_release"], [-50, 40])
        self.assertEqual(self.app.bridge.shots, [])

    def test_agent_action_translates_relative_drag_hold_release_without_shooting(self):
        payload = {
            "action": {
                "action_type": "drag_hold_release",
                "coordinate_frame": "slingshot_relative",
                "drag_start": [300, 220],
                "drag_release": [-50, 40],
                "holdTime": 120,
                "tapTime": 70,
            },
            "fast": True,
        }

        status, data = self.request("POST", "/api/agent-action", payload)

        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(data["shot"], {"x": 250, "y": 180, "tapTime": 70, "fast": True, "releaseTime": 120})
        self.assertEqual(data["action"]["action_type"], "drag_hold_release")
        self.assertEqual(data["action"]["drag_start"], [300, 220])
        self.assertEqual(data["action"]["drag_release"], [-50, 40])
        self.assertEqual(data["action"]["holdTime"], 120)
        self.assertEqual(self.app.bridge.shots, [])

    def test_agent_action_translates_absolute_release_without_shooting(self):
        payload = {
            "action": {
                "action_type": "drag_release",
                "coordinate_frame": "absolute",
                "drag_start": [300, 220],
                "release": [250, 180],
                "tap_time": 65,
            }
        }

        status, data = self.request("POST", "/api/agent-action", payload)

        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(data["shot"], {"x": 250, "y": 180, "tapTime": 65, "fast": False})
        self.assertEqual(data["action"]["drag_release"], [-50, 40])
        self.assertEqual(self.app.bridge.shots, [])

    def test_agent_action_requires_drag_start_for_relative_frame(self):
        status, data = self.request("POST", "/api/agent-action", {"action": {"drag_release": [-50, 40]}})

        self.assertEqual(status, 400)
        self.assertFalse(data["ok"])
        self.assertIn("drag_start", data["error"])

    def test_agent_action_rejects_bad_coordinate_frame(self):
        payload = {"action": {"drag_start": [300, 220], "release": [250, 180], "coordinate_frame": "screen"}}

        status, data = self.request("POST", "/api/agent-action", payload)

        self.assertEqual(status, 400)
        self.assertFalse(data["ok"])
        self.assertIn("coordinate_frame", data["error"])

    def test_agent_action_rejects_bad_action_type(self):
        payload = {"action": {"action_type": "teleport", "drag_start": [300, 220], "drag_release": [-50, 40]}}

        status, data = self.request("POST", "/api/agent-action", payload)

        self.assertEqual(status, 400)
        self.assertFalse(data["ok"])
        self.assertIn("action_type", data["error"])

    def test_agent_action_rejects_non_numeric_coordinates_as_bad_request(self):
        payload = {"action": {"drag_start": [300, 220], "drag_release": ["left", 40]}}

        status, data = self.request("POST", "/api/agent-action", payload)

        self.assertEqual(status, 400)
        self.assertFalse(data["ok"])
        self.assertIn("integer", data["error"])

    def test_load_level_rejects_invalid_level(self):
        status, data = self.request("POST", "/api/load-level", {"level": 0})

        self.assertEqual(status, 400)
        self.assertFalse(data["ok"])
        self.assertIn("level", data["error"])

    def test_load_level_uses_next_available_protocol(self):
        status, data = self.request("POST", "/api/load-level", {"level": 1})

        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(data["response"], 1)
        self.assertEqual(self.app.bridge.loaded, [])
        self.assertEqual(self.app.bridge.loaded_next, [True])

    def test_restart_uses_next_available_protocol(self):
        status, data = self.request("POST", "/api/restart", {})

        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(data["response"], 1)
        self.assertEqual(self.app.bridge.restarted, 0)
        self.assertEqual(self.app.bridge.loaded_next, [True])

    def test_connect_accepts_new_training_set_with_ready_for_new_set(self):
        self.app.bridge.states = [GameState.NEWTRAININGSET, GameState.PLAYING]

        status = self.app.connect_bridge(configure=True)

        self.assertTrue(status["connected"])
        self.assertEqual(self.app.bridge.ready_calls, 1)
        self.assertEqual(self.app.bridge.loaded, [])

    def test_connect_loads_current_level_from_menu_states(self):
        self.app.bridge.states = [GameState.MAIN_MENU, GameState.LOADING, GameState.PLAYING]

        status = self.app.connect_bridge(configure=True)

        self.assertTrue(status["connected"])
        self.assertEqual(self.app.bridge.novelty_calls, 1)
        self.assertEqual(self.app.bridge.loaded, [])
        self.assertEqual(self.app.bridge.loaded_next, [True])
        self.assertGreaterEqual(self.app.bridge.zoomed_out, 1)

    def test_connect_waits_through_loading_until_playing(self):
        self.app.bridge.states = [GameState.LOADING, GameState.LOADING, GameState.PLAYING]

        status = self.app.connect_bridge(configure=True)

        self.assertTrue(status["connected"])
        self.assertEqual(self.app.bridge.loaded, [])

    def test_start_game_launches_java_in_process_group(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            game_dir = root / "sciencebirdsgames" / "Linux"
            game_dir.mkdir(parents=True)
            for name in ("game_playing_interface.jar", "9001.x86_64", "config.xml"):
                (game_dir / name).write_text("ok", encoding="utf-8")
            (game_dir / "9001_Data").mkdir()
            app = AppState(root=root)
            app.connect_bridge = lambda configure=True: app.status()

            with patch("src.webui.server.subprocess.Popen") as popen:
                popen.return_value.poll.return_value = None
                app.start_game()

            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertEqual(popen.call_args.kwargs["stdout"], subprocess.DEVNULL)
            self.assertEqual(popen.call_args.kwargs["stderr"], subprocess.DEVNULL)

    def test_review_runtime_launches_with_positive_stride_and_separate_physics_port(self):
        with TemporaryDirectory() as tmp:
            game_dir = Path(tmp)
            for name in ("game_playing_interface.jar", "9001.x86_64", "config.xml"):
                (game_dir / name).write_text("ok", encoding="utf-8")
            (game_dir / "9001_Data").mkdir()
            app = AppState(
                root=Path("/tmp/nonexistent-webui-root"),
                physics_v2_review=True,
                physics_port=2005,
                engine_game_port=29001,
                review_runtime_dir=game_dir,
            )
            app.connect_bridge = lambda configure=True: app.status()

            with patch("src.webui.server.subprocess.Popen") as popen:
                popen.return_value.poll.return_value = None
                app.start_game()

            command = popen.call_args.args[0]
            self.assertIn("--physics-port", command)
            self.assertEqual(command[command.index("--physics-port") + 1], "2005")
            self.assertEqual(popen.call_args.kwargs["env"]["NOVPHY_PHYSICS_CAPTURE_V2_STRIDE"], "1")

    def test_stop_terminates_started_process_group(self):
        process = type("Process", (), {})()
        process.pid = 12345
        process.wait_calls = 0
        process.poll = lambda: None

        def wait(timeout=None):
            process.wait_calls += 1
            return 0

        process.wait = wait
        app = AppState(root=Path("/tmp/nonexistent-webui-root"), game_process=process)

        with patch("src.webui.server.os.getpgid", return_value=12345), patch("src.webui.server.os.killpg") as killpg:
            status = app.stop()

        self.assertFalse(status["gameProcessRunning"])
        self.assertEqual(killpg.call_count, 1)


if __name__ == "__main__":
    unittest.main()
