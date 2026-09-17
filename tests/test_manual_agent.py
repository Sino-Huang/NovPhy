import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from scripts.manual_agent import (
    capture_pixel_rollout,
    deduplicate_similar_actions,
    generate_diverse_drag_release_actions,
    image_is_uniform,
    main,
    prepare_for_play,
    render_symbolic_frame,
    repl,
    save_frame,
    start_engine,
    stop_started_engine,
)
from scripts.slingshot_readiness import slingshot_observation_from_symbolic_state
from scripts.process_lifecycle import REGISTRY_ENV, REGISTRY_FAILURE_ENV, registry_snapshot


def _pid_is_running(pid):
    try:
        os.kill(pid, 0)
        stat = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
        return stat[stat.rindex(")") + 2] != "Z"
    except (OSError, ProcessLookupError):
        return False


def _wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(.02)
    return predicate()


def _start_fake_engine(directory, launcher_exits):
    child_pid_path = Path(directory) / "child.pid"
    child = (
        "import os,signal,time;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"open({str(child_pid_path)!r},'w').write(str(os.getpid()));"
        "time.sleep(300)"
    )
    launcher = (
        "import subprocess,sys;"
        f"child=subprocess.Popen([sys.executable,'-c',{child!r}]);"
        + ("" if launcher_exits else "child.wait()")
    )
    process = subprocess.Popen([sys.executable, "-c", launcher], start_new_session=True)
    setattr(process, "novphy_process_group", True)
    setattr(process, "novphy_process_group_id", process.pid)
    if not _wait_for(child_pid_path.exists):
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)
        raise RuntimeError("fake engine child did not start")
    return process, int(child_pid_path.read_text(encoding="ascii"))


def _force_stop_fake_engine(process, child_pid):
    if _pid_is_running(process.pid) or _pid_is_running(child_pid):
        try:
            os.killpg(getattr(process, "novphy_process_group_id"), signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


class FakeBridge:
    def __init__(self):
        self.shots = []

    def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
        self.shots.append((x, y, tap_time, fast, release_time))
        return 1

    def set_speed(self, _speed):
        return 1

    def fully_zoom_out(self):
        return 1

    def screenshot(self):
        class Screenshot:
            width = 8
            height = 6
            rgb = bytes([205, 205, 205] * (8 * 6))

        return Screenshot()

    def get_symbolic_state_without_screenshot(self):
        return [
            {
                "features": [
                    {
                        "geometry": {},
                        "type": "Feature",
                        "properties": {"id": "-166", "label": "Ground", "yindex": 4, "colormap": []},
                    },
                    {
                        "geometry": {"coordinates": [[[1, 4], [1, 1], [2, 1], [2, 4]]], "type": "Polygon"},
                        "type": "Feature",
                        "properties": {"id": "-302", "label": "Slingshot", "colormap": []},
                    },
                    {
                        "geometry": {"coordinates": [[[5, 3], [6, 2], [5, 1], [4, 2]]], "type": "Polygon"},
                        "type": "Feature",
                        "properties": {"id": "-398", "label": "pig_basic_big_1", "colormap": []},
                    },
                ]
            }
        ]


class ManualAgentTest(unittest.TestCase):
    def test_generate_diverse_actions_spreads_angle_strength_hold_and_tap(self):
        actions = generate_diverse_drag_release_actions(
            drag_start=(300, 220),
            count=8,
            strengths=(40, 90),
            angles_degrees=(-60, -15, 30, 60),
            tap_times=(0, 70),
            hold_times=(600, 900),
        )

        self.assertEqual(len(actions), 8)
        self.assertEqual({action["action_type"] for action in actions}, {"drag_hold_release"})
        self.assertEqual({tuple(action["drag_start"]) for action in actions}, {(300, 220)})
        self.assertGreaterEqual(len({tuple(action["drag_release"]) for action in actions}), 4)
        self.assertEqual({action["tapTime"] for action in actions}, {0, 70})
        self.assertEqual({action.get("holdTime", 0) for action in actions}, {600, 900})
        self.assertTrue(all(action["holdTime"] >= 600 for action in actions))

        unique_actions = deduplicate_similar_actions(actions + actions, strength_bin=10, angle_bin_degrees=10)
        self.assertEqual(unique_actions, actions)

    def test_default_generated_actions_aim_launches_to_the_right(self):
        actions = generate_diverse_drag_release_actions(count=6)

        self.assertTrue(all(action["drag_release"][0] < 0 for action in actions))
        self.assertTrue(all(action["drag_release"][1] > 0 for action in actions))
        self.assertTrue(all(action["holdTime"] >= 1000 for action in actions))
        self.assertTrue(all((action["drag_release"][0] ** 2 + action["drag_release"][1] ** 2) ** 0.5 >= 75 for action in actions))

    def test_bidirectional_generated_actions_mirror_odd_indices_without_changing_other_fields(self):
        default_actions = generate_diverse_drag_release_actions(count=6)
        explicitly_default_actions = generate_diverse_drag_release_actions(count=6, bidirectional_launches=False)
        bidirectional_actions = generate_diverse_drag_release_actions(count=6, bidirectional_launches=True)

        self.assertEqual(default_actions, explicitly_default_actions)
        self.assertEqual(
            [action["drag_release"][0] > 0 for action in bidirectional_actions],
            [False, True, False, True, False, True],
        )
        for index, (default_action, bidirectional_action) in enumerate(zip(default_actions, bidirectional_actions)):
            expected_action = (
                default_action
                if index % 2 == 0
                else {
                    **default_action,
                    "drag_release": [-default_action["drag_release"][0], default_action["drag_release"][1]],
                }
            )
            self.assertEqual(bidirectional_action, expected_action)

    def test_bidirectional_generated_actions_with_odd_count_include_three_standard_and_two_mirrored_launches(self):
        actions = generate_diverse_drag_release_actions(count=5, bidirectional_launches=True)
        release_x_values = [action["drag_release"][0] for action in actions]

        self.assertEqual(len(actions), 5)
        self.assertEqual(sum(release_x < 0 for release_x in release_x_values), 3)
        self.assertEqual(sum(release_x > 0 for release_x in release_x_values), 2)
        self.assertEqual([release_x > 0 for release_x in release_x_values], [False, True, False, True, False])

    def test_capture_pixel_rollout_records_timestamped_frames_at_target_fps(self):
        class RolloutBridge(FakeBridge):
            def __init__(self):
                super().__init__()
                self.frame_index = 0

            def screenshot(self):
                class Screenshot:
                    width = 4
                    height = 3

                    def __init__(self, frame_index):
                        value = 40 + frame_index
                        self.rgb = bytes(channel for pixel in range(4 * 3) for channel in (value + pixel, 10, 20))

                screenshot = Screenshot(self.frame_index)
                self.frame_index += 1
                return screenshot

            def get_game_state(self):
                from src.webui.bridge import GameState

                return GameState.PLAYING

            def get_current_score(self):
                return 100 + self.frame_index

        now = [10.0]

        def clock():
            return now[0]

        def sleeper(seconds):
            now[0] += seconds

        with TemporaryDirectory() as tmp:
            metadata = capture_pixel_rollout(
                RolloutBridge(),
                Path(tmp),
                target_fps=4,
                duration_seconds=1.0,
                max_frames=4,
                clock=clock,
                sleeper=sleeper,
            )

            self.assertEqual(metadata["target_fps"], 4)
            self.assertEqual(metadata["frame_count"], 4)
            self.assertEqual([round(frame["t"], 2) for frame in metadata["frames"]], [0.0, 0.25, 0.5, 0.75])
            self.assertEqual(len(metadata["state_samples"]), 4)
            self.assertTrue((Path(tmp) / "frames" / "frame_000000.png").is_file())
            saved_metadata = json.loads((Path(tmp) / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_metadata["frame_count"], 4)

    def test_prepare_for_play_recovers_from_lost_with_next_level(self):
        class PrepareBridge:
            def __init__(self):
                self.states = ["LOST", "PLAYING"]
                self.novelty_calls = 0
                self.next_calls = 0
                self.zoom_out_calls = 0

            def get_game_state(self):
                from src.webui.bridge import GameState

                state = self.states.pop(0)
                return getattr(GameState, state)

            def get_novelty_info(self):
                self.novelty_calls += 1
                return -1

            def load_next_available_level(self):
                self.next_calls += 1
                return 1

            def fully_zoom_out(self):
                self.zoom_out_calls += 1
                return 1

        bridge = PrepareBridge()
        state = prepare_for_play(bridge, timeout=2, poll_delay=0)

        self.assertEqual(state.name, "PLAYING")
        self.assertEqual(bridge.novelty_calls, 1)
        self.assertEqual(bridge.next_calls, 1)
        self.assertEqual(bridge.zoom_out_calls, 1)

    def test_prepare_for_play_recovers_from_evaluation_terminated_with_next_level(self):
        class PrepareBridge:
            def __init__(self):
                self.states = ["EVALUATION_TERMINATED", "PLAYING"]
                self.novelty_calls = 0
                self.next_calls = 0
                self.zoom_out_calls = 0

            def get_game_state(self):
                from src.webui.bridge import GameState

                state = self.states.pop(0)
                return getattr(GameState, state)

            def get_novelty_info(self):
                self.novelty_calls += 1
                return -1

            def load_next_available_level(self):
                self.next_calls += 1
                return 1

            def fully_zoom_out(self):
                self.zoom_out_calls += 1
                return 1

        bridge = PrepareBridge()
        state = prepare_for_play(bridge, timeout=2, poll_delay=0)

        self.assertEqual(state.name, "PLAYING")
        self.assertEqual(bridge.novelty_calls, 1)
        self.assertEqual(bridge.next_calls, 1)
        self.assertEqual(bridge.zoom_out_calls, 1)

    def test_prepare_for_play_does_not_spam_same_new_trial_transition(self):
        class PrepareBridge:
            def __init__(self):
                self.states = ["NEWTRIAL", "NEWTRIAL", "NEWTRIAL", "PLAYING"]
                self.ready_calls = 0
                self.zoom_out_calls = 0

            def get_game_state(self):
                from src.webui.bridge import GameState

                state = self.states.pop(0)
                return getattr(GameState, state)

            def ready_for_new_set(self):
                self.ready_calls += 1
                return (1, 0, 0, 0, 0, 0, 0)

            def fully_zoom_out(self):
                self.zoom_out_calls += 1
                return 1

        bridge = PrepareBridge()
        state = prepare_for_play(bridge, timeout=2, poll_delay=0)

        self.assertEqual(state.name, "PLAYING")
        self.assertEqual(bridge.ready_calls, 1)
        self.assertEqual(bridge.zoom_out_calls, 1)

    def test_prepare_for_play_reissues_new_trial_after_bounded_wait(self):
        class PrepareBridge:
            def __init__(self):
                self.ready_calls = 0
                self.zoom_out_calls = 0

            def get_game_state(self):
                from src.webui.bridge import GameState

                if self.ready_calls >= 2:
                    return GameState.PLAYING
                return GameState.NEWTRIAL

            def ready_for_new_set(self):
                self.ready_calls += 1
                return (1, 0, 0, 0, 0, 0, 0)

            def fully_zoom_out(self):
                self.zoom_out_calls += 1
                return 1

        now = [0.0]

        def clock():
            return now[0]

        def sleeper(seconds):
            now[0] += seconds

        bridge = PrepareBridge()
        state = prepare_for_play(
            bridge,
            timeout=5,
            poll_delay=0.5,
            transition_retry_seconds=1.0,
            clock=clock,
            sleeper=sleeper,
        )

        self.assertEqual(state.name, "PLAYING")
        self.assertEqual(bridge.ready_calls, 2)
        self.assertEqual(bridge.zoom_out_calls, 1)

    def test_prepare_for_play_uses_next_level_after_persistent_new_trial_retries(self):
        class PrepareBridge:
            def __init__(self):
                self.ready_calls = 0
                self.novelty_calls = 0
                self.next_calls = 0
                self.zoom_out_calls = 0

            def get_game_state(self):
                from src.webui.bridge import GameState

                if self.next_calls:
                    return GameState.PLAYING
                return GameState.NEWTRIAL

            def ready_for_new_set(self):
                self.ready_calls += 1
                return (1, 0, 0, 0, 0, 0, 0)

            def get_novelty_info(self):
                self.novelty_calls += 1
                return -1

            def load_next_available_level(self):
                self.next_calls += 1
                return 1

            def fully_zoom_out(self):
                self.zoom_out_calls += 1
                return 1

        now = [0.0]

        def clock():
            return now[0]

        def sleeper(seconds):
            now[0] += seconds

        bridge = PrepareBridge()
        state = prepare_for_play(
            bridge,
            timeout=10,
            poll_delay=0.5,
            transition_retry_seconds=1.0,
            new_set_ready_attempts=2,
            clock=clock,
            sleeper=sleeper,
        )

        self.assertEqual(state.name, "PLAYING")
        self.assertEqual(bridge.ready_calls, 2)
        self.assertEqual(bridge.novelty_calls, 1)
        self.assertEqual(bridge.next_calls, 1)
        self.assertEqual(bridge.zoom_out_calls, 1)

    def test_image_is_uniform_detects_solid_frame(self):
        bridge = FakeBridge()
        image = render_symbolic_frame(bridge.get_symbolic_state_without_screenshot(), 8, 6)

        self.assertFalse(image_is_uniform(image))

    def test_save_frame_rejects_uniform_screenshot_without_symbolic_fallback(self):
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            with self.assertRaisesRegex(RuntimeError, "uniform Science Birds screenshot"):
                save_frame(bridge, path)

            self.assertFalse(path.exists())

    def test_drag_hold_release_sends_relative_release(self):
        bridge = FakeBridge()
        commands = iter([
            "drag 300 220",
            "hold 120",
            "release 250 180 70 fast",
            "quit",
        ])

        with patch("builtins.input", lambda _: next(commands)), redirect_stdout(io.StringIO()):
            repl(bridge, frame_height=480)

        reference = slingshot_observation_from_symbolic_state(
            bridge.get_symbolic_state_without_screenshot(), 480
        )
        game_x = int(reference["gameX"]) - 50
        game_y = int(reference["gameY"]) - 40
        self.assertEqual(
            bridge.shots,
            [(game_x, 479 - game_y, 70, True, 120)],
        )

    def test_release_requires_drag_start(self):
        bridge = FakeBridge()
        commands = iter(["release 250 180", "quit"])
        output = io.StringIO()

        with patch("builtins.input", lambda _: next(commands)), redirect_stdout(output):
            repl(bridge, frame_height=480)

        self.assertEqual(bridge.shots, [])
        self.assertIn("drag", output.getvalue())

    def test_start_engine_logs_output_to_file_for_runtime_diagnostics(self):
        with TemporaryDirectory() as tmp:
            game_dir = Path(tmp)
            (game_dir / "game_playing_interface.jar").write_text("jar", encoding="utf-8")
            opened = []

            def fake_popen(command, **kwargs):
                opened.append(kwargs["stdout"])
                return type("Process", (), {"pid": 1234})()

            with patch("scripts.manual_agent.subprocess.Popen", side_effect=fake_popen) as popen:
                process = start_engine(game_dir, headless=False)

        self.assertEqual(popen.call_args.args[0], ["java", "-jar", "./game_playing_interface.jar", "--dev"])
        self.assertIsNot(popen.call_args.kwargs["stdout"], subprocess.DEVNULL)
        self.assertEqual(popen.call_args.kwargs["stderr"], subprocess.STDOUT)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertTrue(opened[0].name.startswith("/tmp/novphy_game_engine_"))
        self.assertFalse(opened[0].closed)
        self.assertTrue(process.novphy_process_group)
        self.assertEqual(process.novphy_process_group_id,process.pid)
        opened[0].close()

    def test_start_engine_accepts_isolated_worker_ports(self):
        with TemporaryDirectory() as tmp:
            game_dir = Path(tmp)
            (game_dir / "game_playing_interface.jar").write_text("jar", encoding="utf-8")

            def fake_popen(command, **kwargs):
                return type("Process", (), {"pid": 1234})()

            with patch("scripts.manual_agent.subprocess.Popen", side_effect=fake_popen) as popen:
                start_engine(game_dir, headless=True, agent_port=2014, game_port=9011, physics_port=2015)

        self.assertEqual(
            popen.call_args.args[0],
            [
                "java",
                "-jar",
                "./game_playing_interface.jar",
                "--headless",
                "--agent-port",
                "2014",
                "--game-start-port",
                "9011",
                "--physics-port",
                "2015",
                "--dev",
            ],
        )
        popen.call_args.kwargs["stdout"].close()

    def test_real_start_engine_records_group_before_dummy_jar_exits(self):
        with TemporaryDirectory() as tmp:
            game_dir = Path(tmp)
            (game_dir/"game_playing_interface.jar").write_text("not a jar",encoding="ascii")
            process = start_engine(game_dir,headless=True)
            log_path = Path(process.novphy_log_file.name)
            try:
                self.assertTrue(process.novphy_process_group)
                self.assertEqual(process.novphy_process_group_id,process.pid)
                process.wait(timeout=5)
                stop_started_engine(process)
                self.assertTrue(process.novphy_log_file.closed)
            finally:
                if process.poll() is None: process.kill(); process.wait(timeout=2)
                if not process.novphy_log_file.closed: process.novphy_log_file.close()
                log_path.unlink(missing_ok=True)

    def test_child_registration_failure_prevents_engine_exec(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); game = root/"game"; game.mkdir()
            (game/"game_playing_interface.jar").write_text("fixture",encoding="ascii")
            binary = root/"bin"; binary.mkdir()
            marker = root/"java-executed"
            java = binary/"java"
            java.write_text(f"#!/bin/sh\ntouch {marker}\nsleep 300\n",encoding="ascii")
            java.chmod(0o755)
            registry = root/"groups.txt"; registry.write_text("",encoding="ascii")
            environment = {REGISTRY_ENV:str(registry),REGISTRY_FAILURE_ENV:"1",
                           "PATH":str(binary)+os.pathsep+os.environ["PATH"]}
            with patch.dict(os.environ,environment):
                process = start_engine(game,headless=False)
                process.wait(timeout=5)
                stop_started_engine(process)
            self.assertFalse(marker.exists())
            self.assertEqual(registry_snapshot(registry),({},set()))

    def test_main_stops_started_engine_when_connection_fails(self):
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
        args = ["manual_agent.py", "--start-engine", "--no-prepare"]

        with (
            patch("sys.argv", args),
            patch("scripts.manual_agent.start_engine", return_value=process),
            patch("scripts.manual_agent.connect_with_retry", side_effect=RuntimeError("connection refused")),
        ):
            with self.assertRaisesRegex(RuntimeError, "connection refused"):
                main()

        self.assertTrue(process.terminated)
        self.assertTrue(process.waited)

    def test_stop_started_engine_terminates_process_group_for_started_engine(self):
        class GroupProcess:
            pid = 1357
            novphy_process_group = True

            def __init__(self):
                self.terminated = False
                self.waited = False
                self.novphy_log_file = io.BytesIO()

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                self.waited = True

        process = GroupProcess()
        with patch("scripts.manual_agent.os.getpgid", return_value=2468), patch("scripts.manual_agent.os.killpg") as killpg:
            stop_started_engine(process)

        killpg.assert_called_once_with(2468, signal.SIGTERM)
        self.assertTrue(process.waited)
        self.assertFalse(process.terminated)
        self.assertTrue(process.novphy_log_file.closed)

    def test_stop_started_engine_reaps_group_while_launcher_is_alive(self):
        with TemporaryDirectory() as tmp:
            process, child_pid = _start_fake_engine(tmp, launcher_exits=False)
            try:
                with patch("scripts.manual_agent.ENGINE_STOP_GRACE_SECONDS", .2):
                    stop_started_engine(process)
                self.assertTrue(_wait_for(lambda: not _pid_is_running(child_pid)))
            finally:
                _force_stop_fake_engine(process, child_pid)

    def test_stop_started_engine_reaps_group_after_launcher_exits(self):
        with TemporaryDirectory() as tmp:
            process, child_pid = _start_fake_engine(tmp, launcher_exits=True)
            try:
                process.wait(timeout=2)
                self.assertTrue(_pid_is_running(child_pid))
                with patch("scripts.manual_agent.ENGINE_STOP_GRACE_SECONDS", .2):
                    stop_started_engine(process)
                self.assertTrue(_wait_for(lambda: not _pid_is_running(child_pid)))
            finally:
                _force_stop_fake_engine(process, child_pid)

    def test_main_startup_failure_reaps_child_after_launcher_exits(self):
        with TemporaryDirectory() as tmp:
            process,child_pid = _start_fake_engine(tmp,launcher_exits=True)
            try:
                process.wait(timeout=2)
                with (
                    patch("sys.argv",["manual_agent.py","--start-engine","--no-prepare"]),
                    patch("scripts.manual_agent.start_engine",return_value=process),
                    patch("scripts.manual_agent.connect_with_retry",side_effect=RuntimeError("startup failed")),
                    patch("scripts.manual_agent.ENGINE_STOP_GRACE_SECONDS",.2),
                ):
                    with self.assertRaisesRegex(RuntimeError,"startup failed"): main()
                self.assertTrue(_wait_for(lambda: not _pid_is_running(child_pid)))
            finally:
                _force_stop_fake_engine(process,child_pid)

    def test_main_stops_started_engine_when_disconnect_fails(self):
        class DisconnectFailBridge(FakeBridge):
            def configure(self, agent_id, mode):
                return (0, 0, 1)

            def set_speed(self, speed):
                return 1

            def disconnect(self):
                raise RuntimeError("disconnect failed")

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
        args = ["manual_agent.py", "--start-engine", "--no-prepare"]

        with (
            patch("sys.argv", args),
            patch("scripts.manual_agent.start_engine", return_value=process),
            patch("scripts.manual_agent.connect_with_retry", return_value=DisconnectFailBridge()),
            patch("scripts.manual_agent.repl", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "disconnect failed"):
                main()

        self.assertTrue(process.terminated)
        self.assertTrue(process.waited)


if __name__ == "__main__":
    unittest.main()
