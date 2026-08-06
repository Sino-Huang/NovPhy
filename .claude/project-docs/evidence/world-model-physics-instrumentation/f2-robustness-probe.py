from __future__ import annotations

import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.collect_rollouts import collect_rollouts, validate_rollout_artifact
from src.webui.bridge import PhysicsCaptureV1
from tests.test_collect_rollouts import FakeBridge, PhysicsCapturePersistenceTests


def main() -> None:
    helper = PhysicsCapturePersistenceTests()
    states, events = helper._records()
    png = helper._png()

    class Bridge(FakeBridge):
        def get_physics_capture_v1(self) -> PhysicsCaptureV1:
            return PhysicsCaptureV1(png, states[1], tuple(events))

    action = {
        "coordinate_frame": "absolute",
        "drag_start": [100, 200],
        "drag_release": [130, 150],
        "tapTime": 70,
        "holdTime": 600,
    }
    guard = {
        "pre_shot_image": None,
        "pre_shot_sample": None,
        "post_recovery_protocol_state": {},
        "recovery_action": None,
        "pre_shot_guard": {"status": "accepted", "invalid_reason": None},
    }
    temporary_path: Path | None = None
    result: dict[str, object] = {}
    with TemporaryDirectory(prefix="novphy-f2-robustness-") as temporary:
        temporary_path = Path(temporary)
        root = temporary_path / "rollouts"
        kwargs = {
            "target_fps": 1,
            "duration_seconds": 1,
            "max_frames": 1,
            "anchor_actions": False,
            "video_runner": lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0),
            "physics_capture_v1": True,
            "physics_player_sha256": "a" * 64,
            "physics_protocol_sha256": "b" * 64,
            "physics_archive_sha256": "c" * 64,
        }
        with patch("scripts.collect_rollouts._run_pre_shot_guard", return_value=guard):
            first = collect_rollouts(Bridge(), root, [action], **kwargs)
        result["first_collection_accepted"] = first["accepted_rollout_count"] == 1

        state_path = root / "shot_001" / "physics_state.jsonl"
        external = temporary_path / "outside-root-state.jsonl"
        state_path.replace(external)
        state_path.symlink_to(external)
        escaped_validation = validate_rollout_artifact(
            root / "shot_001", capture_contract="physics_capture_v1"
        )
        result["symlink_escape_accepted"] = escaped_validation.get("accepted")
        result["symlink_target_outside_shot"] = not external.is_relative_to(root / "shot_001")

        state_path.unlink()
        external.replace(state_path)
        try:
            with patch("scripts.collect_rollouts._run_pre_shot_guard", return_value=guard):
                collect_rollouts(Bridge(), root, [action], **kwargs)
        except Exception as error:
            result["repeated_resume_exception"] = type(error).__name__
            result["repeated_resume_message"] = str(error)
        else:
            result["repeated_resume_exception"] = None

        result["tmp_attempt_after_resume"] = (root / "shot_001.tmp").exists()
        result["final_attempt_after_resume"] = (root / "shot_001").exists()
        result["temporary_root"] = str(temporary_path)
    result["cleanup_root_absent"] = temporary_path is not None and not temporary_path.exists()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
