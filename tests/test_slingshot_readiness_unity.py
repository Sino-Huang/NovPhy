from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.manual_agent import connect_with_retry, prepare_for_play
from scripts.slingshot_readiness import (
    prepare_screen_shot,
    slingshot_observation_from_symbolic_state,
)
from scripts.smoke_physics_capture import (
    archive_details,
    free_port,
    launch_environment,
    start_display,
    terminate,
)
from src.webui.bridge import PlayingMode


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "sciencebirdsgames/physics-v2"
PRODUCTION_ANCHOR = {"canvasX": 127, "canvasY": 256}


@unittest.skipUnless(
    os.environ.get("NOVPHY_RUN_UNITY_REGRESSION") == "1",
    "set NOVPHY_RUN_UNITY_REGRESSION=1 to run the packaged Unity camera regression",
)
class RealUnitySlingshotReadinessTests(unittest.TestCase):
    def test_accelerated_zoom_reaches_and_retains_production_projection(self):
        with tempfile.TemporaryDirectory(prefix="novphy-slingshot-readiness-") as temporary:
            root = Path(temporary)
            player = root / "player"
            archive_details(STAGE, player)
            display, display_process = start_display(root / "display.log")
            agent_port = free_port()
            game_port = free_port()
            environment, _ = launch_environment(display, os.environ)
            engine = subprocess.Popen(
                [
                    "java",
                    "-jar",
                    "./game_playing_interface.jar",
                    "--agent-port",
                    str(agent_port),
                    "--game-start-port",
                    str(game_port),
                    "--dev",
                ],
                cwd=player,
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            bridge = None
            try:
                bridge = connect_with_retry(
                    "127.0.0.1", agent_port, timeout=300, deadline_seconds=90
                )
                bridge.configure(28888, PlayingMode.TRAINING)
                bridge.set_speed(1)
                prepare_for_play(bridge, timeout=120, poll_delay=0.5)
                direct = slingshot_observation_from_symbolic_state(
                    bridge.get_symbolic_state_without_screenshot(), 480
                )
                self.assertIsNotNone(direct)
                direct_error = max(
                    abs(direct[key] - PRODUCTION_ANCHOR[key])
                    for key in ("canvasX", "canvasY")
                )
                self.assertGreaterEqual(direct_error, 25)
                self.assertLessEqual(direct_error, 55, direct)

                prepared = prepare_screen_shot(
                    bridge,
                    {
                        "coordinate_frame": "slingshot_relative",
                        "drag_start": [127, 223],
                        "drag_release": [-1, 1],
                    },
                    execution_speed=1,
                    frozen_socket_command={"x": 126, "y": 257},
                    retained_anchor=PRODUCTION_ANCHOR,
                )
                accelerated = prepared.slingshot
                self.assertLessEqual(
                    max(
                        abs(accelerated[key] - PRODUCTION_ANCHOR[key])
                        for key in ("canvasX", "canvasY")
                    ),
                    2,
                )
                self.assertEqual(
                    prepared.evidence["startup"]["stable_slingshot"],
                    prepared.evidence["execution"]["stable_slingshot"],
                )
            finally:
                if bridge is not None:
                    bridge.disconnect()
                terminate(engine)
                terminate(display_process)


if __name__ == "__main__":
    unittest.main()
