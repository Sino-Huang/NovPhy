from __future__ import annotations

from pathlib import Path
import unittest

from scripts.slingshot_readiness import (
    SlingshotReadinessError,
    prepare_screen_shot,
    slingshot_observation_from_symbolic_state,
)


ROOT = Path(__file__).resolve().parents[1]


def sling(left=100.0, top=200.0, width=20.55):
    return [
        {
            "type": "Slingshot",
            "vertices": [
                [left, top],
                [left + 8, top],
                [left + 8, top + width],
                [left, top + width],
            ],
        }
    ]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Bridge:
    def __init__(self, states):
        self.states = list(states)
        self.last_state = self.states[-1] if self.states else []
        self.calls = []

    def set_speed(self, speed):
        self.calls.append(("speed", speed))
        return 1

    def fully_zoom_out(self):
        self.calls.append(("zoom",))
        return 1

    def get_symbolic_state_without_screenshot(self):
        self.calls.append(("state",))
        if self.states:
            self.last_state = self.states.pop(0)
        return self.last_state

    def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
        self.calls.append(("shoot", x, y, tap_time, fast, release_time))
        return 1


class SlingshotReadinessTests(unittest.TestCase):
    def test_legacy_agent_client_uses_same_prepared_interface(self):
        state = sling()

        class Legacy:
            def __init__(self):
                self.calls = []

            def set_game_simulation_speed(self, speed):
                self.calls.append(("speed", speed))

            def fully_zoom_out(self):
                self.calls.append(("zoom",))

            def get_symbolic_state_without_screenshot(self):
                self.calls.append(("state",))
                return state

            def shoot(self, x, y, release, tap, polar):
                self.calls.append(("shoot", x, y, release, tap, polar))
                return 1

        client = Legacy()
        clock = Clock()
        prepared = prepare_screen_shot(
            client,
            {
                "coordinate_frame": "slingshot_relative",
                "drag_start": [1, 2],
                "drag_release": [-10, 5],
                "releaseTime": 0,
            },
            execution_speed=80,
            clock=clock,
            sleeper=clock.sleep,
        )
        prepared.execute()
        self.assertEqual(client.calls[0], ("speed", 50))
        self.assertEqual(client.calls[5], ("state",))
        self.assertEqual(client.calls[-1][-1], False)
        self.assertEqual(prepared.evidence["execution_speed"], 50)

    def test_exact_order_and_live_anchor(self):
        state = sling()
        bridge = Bridge([state] * 4)
        clock = Clock()
        prepared = prepare_screen_shot(
            bridge,
            {
                "coordinate_frame": "slingshot_relative",
                "drag_start": [1, 2],
                "drag_release": [-10, 5],
                "tapTime": 3,
            },
            execution_speed=1,
            clock=clock,
            sleeper=clock.sleep,
        )

        self.assertEqual(
            bridge.calls,
            [
                ("speed", 50),
                ("zoom",),
                ("state",),
                ("state",),
                ("speed", 1),
                ("state",),
                ("state",),
            ],
        )
        reference = slingshot_observation_from_symbolic_state(state, 480)
        self.assertEqual(prepared.action["drag_start"], [int(reference["gameX"]), int(reference["gameY"])])
        prepared.execute()
        self.assertEqual(bridge.calls[-1][0], "shoot")

    def test_frozen_command_is_never_reanchored(self):
        state = sling()
        expected = slingshot_observation_from_symbolic_state(state, 480)
        frozen = {"x": 17, "y": 281, "tapTime": 4, "releaseTime": 900}
        bridge = Bridge([state] * 4)
        clock = Clock()
        prepared = prepare_screen_shot(
            bridge,
            {
                "coordinate_frame": "slingshot_relative",
                "drag_start": [int(expected["gameX"]), int(expected["gameY"])],
                "drag_release": [-90, 10],
            },
            frozen_socket_command=frozen,
            retained_anchor=expected,
            clock=clock,
            sleeper=clock.sleep,
        )
        self.assertEqual(prepared.socket_command, frozen)
        self.assertEqual(prepared.action["drag_start"], [int(expected["gameX"]), int(expected["gameY"])])

    def test_stable_wrong_projection_fails_closed_and_restores_speed(self):
        state = sling()
        bridge = Bridge([state, state])
        clock = Clock()
        with self.assertRaisesRegex(SlingshotReadinessError, "retained_anchor_mismatch"):
            prepare_screen_shot(
                bridge,
                {"coordinate_frame": "absolute", "release": [2, 3]},
                frozen_socket_command={"x": 2, "y": 476},
                retained_anchor={"canvasX": 160, "canvasY": 200},
                execution_speed=7,
                clock=clock,
                sleeper=clock.sleep,
            )
        self.assertEqual(bridge.calls[-1], ("speed", 7))
        self.assertFalse(any(call[0] == "shoot" for call in bridge.calls))

    def test_frozen_command_without_retained_anchor_fails_closed(self):
        bridge = Bridge([sling()] * 4)
        clock = Clock()
        with self.assertRaisesRegex(SlingshotReadinessError, "retained_anchor_missing"):
            prepare_screen_shot(
                bridge,
                {"coordinate_frame": "absolute", "release": [2, 3]},
                frozen_socket_command={"x": 2, "y": 476},
                clock=clock,
                sleeper=clock.sleep,
            )
        self.assertFalse(any(call[0] == "shoot" for call in bridge.calls))

    def test_missing_slingshot_times_out_without_shooting(self):
        bridge = Bridge([[]])
        clock = Clock()
        with self.assertRaisesRegex(SlingshotReadinessError, "slingshot_missing"):
            prepare_screen_shot(
                bridge,
                {"coordinate_frame": "absolute", "release": [2, 3]},
                timeout=0.11,
                poll_interval=0.05,
                execution_speed=2,
                clock=clock,
                sleeper=clock.sleep,
            )
        self.assertEqual(bridge.calls[-1], ("speed", 2))

    def test_projection_change_after_speed_switch_fails_closed(self):
        first = sling()
        second = sling(left=101)
        bridge = Bridge([first, first, second, second])
        clock = Clock()
        with self.assertRaisesRegex(
            SlingshotReadinessError, "projection_changed_after_execution_speed"
        ):
            prepare_screen_shot(
                bridge,
                {"coordinate_frame": "absolute", "release": [2, 3]},
                clock=clock,
                sleeper=clock.sleep,
            )

    def test_supported_workflows_do_not_call_unchecked_transport(self):
        supported = [
            ROOT / "scripts/collect_rollouts.py",
            ROOT / "scripts/manual_agent.py",
            ROOT / "scripts/capture_issue_48_evidence.py",
            ROOT / "scripts/smoke_physics_capture.py",
            ROOT / "src/webui/server.py",
            ROOT / "sciencebirdsagents/SBEnvironment/SBEnvironmentWrapper.py",
            ROOT / "sciencebirdsagents/SBEnvironment/SBEnvironmentWrapperOpenAI.py",
        ]
        for path in supported:
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                self.assertNotIn(".shoot(", source)
                self.assertNotIn(".fast_shoot(", source)
                self.assertNotIn(".shoot_and_record_ground_truth(", source)


if __name__ == "__main__":
    unittest.main()
