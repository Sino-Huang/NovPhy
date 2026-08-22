from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class PhysicsPlayerWrapperTests(unittest.TestCase):
    def test_forwards_scoped_physics_capture_port_on_consecutive_launches(self) -> None:
        repository = Path(__file__).parents[1]
        wrapper_source = (repository / "scripts" / "9001-player-wrapper.sh").read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            wrapper = runtime / "9001.x86_64"
            player = runtime / "9001-player.x86_64"
            wrapper.write_bytes(wrapper_source)
            wrapper.chmod(0o755)
            player.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > player-args.txt\n",
                encoding="utf-8",
            )
            player.chmod(0o755)
            environment = os.environ.copy()
            environment["NOVPHY_PHYSICS_CAPTURE_PORT"] = "32123"

            for _ in range(2):
                subprocess.run(
                    [str(wrapper), "--port", "9000"],
                    cwd=runtime,
                    env=environment,
                    check=True,
                )
                arguments = (runtime / "player-args.txt").read_text(
                    encoding="utf-8"
                ).splitlines()
                self.assertEqual(
                    arguments[-4:],
                    ["--physics-port", "32123", "--port", "9000"],
                )

            self.assertEqual(wrapper.read_bytes(), wrapper_source)
            self.assertTrue(player.is_file())


if __name__ == "__main__":
    unittest.main()
