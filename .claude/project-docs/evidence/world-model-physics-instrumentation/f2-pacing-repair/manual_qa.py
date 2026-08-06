from __future__ import annotations

import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.collect_rollouts import capture_physics_rollout
from src.webui.bridge import PhysicsCaptureV1


FIXTURES = ROOT / "tests/fixtures/physics_capture_v1"


def main() -> None:
    states = [
        json.loads(line)
        for line in (FIXTURES / "physics_state.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for state in states:
        state["shot_id"] = "shot_000"
    states[1]["rgb_frame"].update(
        {"relative_path": "frames/frame_000000.png", "width_pixels": 4, "height_pixels": 3}
    )
    png_stream = io.BytesIO()
    Image.new("RGB", (4, 3), (30, 20, 10)).save(png_stream, format="PNG")
    png = png_stream.getvalue()
    now = 10.0
    request_times: list[float] = []
    sleep_durations: list[float] = []

    class Bridge:
        request_count = 0

        def get_physics_capture_v1(self) -> PhysicsCaptureV1:
            state = json.loads(json.dumps(states[1]))
            request_times.append(now)
            state["sequence"] += self.request_count
            state["render_frame"] += self.request_count
            state["fixed_step"] += self.request_count
            state["render_time"] += self.request_count / 4
            state["fixed_time"] += self.request_count / 4
            state["rgb_frame"]["render_frame"] = state["render_frame"]
            self.request_count += 1
            return PhysicsCaptureV1(png, state, ())

    def clock() -> float:
        return now

    def sleeper(seconds: float) -> None:
        nonlocal now
        sleep_durations.append(seconds)
        now += seconds

    with TemporaryDirectory(prefix="f2-pacing-repair-") as temporary:
        root = Path(temporary)
        shot = root / "shot_000.tmp"
        metadata = capture_physics_rollout(
            Bridge(),
            shot,
            target_fps=4,
            duration_seconds=1,
            max_frames=4,
            state_header=states[0],
            clock=clock,
            sleeper=sleeper,
            player_sha256="a" * 64,
            protocol_sha256="b" * 64,
            archive_sha256="c" * 64,
        )
        observable = {
            "request_times": request_times,
            "sleep_durations": sleep_durations,
            "planned_elapsed_seconds": request_times[-1] - request_times[0],
            "frame_count": metadata["frame_count"],
            "state_sidecar_lines": len(
                (shot / "physics_state.jsonl").read_text(encoding="utf-8").splitlines()
            ),
            "event_count": metadata["physics_event_count"],
            "used_real_sleep": False,
        }
        assert request_times == [10.0, 10.25, 10.5, 10.75]
        assert sleep_durations == [0.25, 0.25, 0.25]
        assert metadata["frame_count"] == 4
        assert observable["state_sidecar_lines"] == 5
        temporary_root = root

    observable["temporary_root_removed"] = not temporary_root.exists()
    assert observable["temporary_root_removed"]
    print(json.dumps(observable, sort_keys=True))


if __name__ == "__main__":
    main()
