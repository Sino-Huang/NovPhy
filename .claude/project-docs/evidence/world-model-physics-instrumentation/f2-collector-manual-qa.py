from __future__ import annotations

import io
import json
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
from unittest.mock import patch

from PIL import Image

from scripts.collect_rollouts import (
    RolloutCollectionError,
    cleanup_incomplete_physics_attempts,
    collect_rollouts,
    main,
)
from scripts.prepare_rollout_dataset import (
    CollectionOptions,
    LevelEntry,
    PhysicsCaptureProvenance,
    PlannedEpisode,
    WorkerSpec,
    _collection_command_lines,
)
from src.webui.bridge import GameState, PhysicsCaptureV1


ROOT = Path(tempfile.mkdtemp(prefix="novphy-f2-collector-"))
FIXTURES = Path("tests/fixtures/physics_capture_v1")
STATE = json.loads(FIXTURES.joinpath("physics_state.jsonl").read_text(encoding="utf-8").splitlines()[1])


class Bridge:
    def __init__(self, malformed: bool = False):
        self.malformed = malformed
        self.recorder_shots = 0
        self.call_order: list[str] = []

    def get_physics_capture_v1(self):
        self.call_order.append("request-70")
        state = {"schema_version": "physics_capture_v1"} if self.malformed else STATE
        return PhysicsCaptureV1(PNG, state, ())

    def shoot_and_record_ground_truth(self, *_args, **_kwargs):
        self.call_order.append("recorder-action")
        self.recorder_shots += 1
        return 1

    def get_game_state(self):
        return GameState.PLAYING

    def get_current_score(self):
        return 0

    def get_current_level(self):
        return 1


png_buffer = io.BytesIO()
Image.new("RGB", (4, 3), (10, 20, 30)).save(png_buffer, format="PNG")
PNG = png_buffer.getvalue()
guard = {
    "pre_shot_image": Image.new("RGB", (4, 3), (1, 2, 3)),
    "pre_shot_sample": {"state": "PLAYING", "score": 0},
    "post_recovery_protocol_state": {},
    "recovery_action": None,
    "pre_shot_guard": {"status": "accepted", "invalid_reason": None},
}


def generated_argv(output_dir: Path) -> list[str]:
    episode = PlannedEpisode(
        "train",
        LevelEntry("novelty_level_1", "type01001", "level.xml"),
        output_dir,
        "scheduled",
    )
    provenance = PhysicsCaptureProvenance(
        ROOT / "player.tar",
        ROOT / "smoke.json",
        "a" * 64,
        "b" * 64,
        "c" * 64,
    )
    command = " ".join(
        line.strip().removesuffix("\\").strip()
        for line in _collection_command_lines(
            episode,
            CollectionOptions(count=1, fps=1, duration=1),
            WorkerSpec(0, ":149", 2004, 9001),
            provenance,
        )[2:]
    )
    return ["collect_rollouts.py", *shlex.split(command)]


def run_generated(output_dir: Path, bridge: Bridge) -> dict:
    def run_actual(path, actions, **kwargs):
        return collect_rollouts(
            bridge,
            path,
            actions,
            target_fps=kwargs["target_fps"],
            duration_seconds=kwargs["duration_seconds"],
            max_frames=1,
            shoot_before_capture=kwargs["shoot_before_capture"],
            anchor_actions=False,
            physics_capture_v1=kwargs["physics_capture_v1"],
            physics_player_sha256=kwargs["physics_player_sha256"],
            physics_protocol_sha256=kwargs["physics_protocol_sha256"],
            physics_archive_sha256=kwargs["physics_archive_sha256"],
        )

    with (
        patch.object(sys, "argv", generated_argv(output_dir)),
        patch("scripts.collect_rollouts.collect_fresh_engine_rollouts", side_effect=run_actual),
        patch("scripts.collect_rollouts._run_pre_shot_guard", return_value=guard),
        patch("PIL.ImageGrab.grab"),
    ):
        main()
    return json.loads(output_dir.joinpath("manifest.json").read_text(encoding="utf-8"))


accepted_root = ROOT / "accepted"
accepted_bridge = Bridge()
accepted_manifest = run_generated(accepted_root, accepted_bridge)
assert accepted_bridge.recorder_shots == 1
assert accepted_bridge.call_order == ["recorder-action", "request-70"]
assert accepted_manifest["rollout_count"] == 1
assert accepted_root.joinpath("shot_001").is_dir()
assert not accepted_root.joinpath("shot_001.tmp").exists()

malformed_root = ROOT / "malformed"
try:
    run_generated(malformed_root, Bridge(malformed=True))
except RolloutCollectionError as error:
    malformed_error = str(error)
else:
    raise AssertionError("malformed capture was accepted")
assert "malformed_capture" in malformed_error
assert not malformed_root.joinpath("shot_001").exists()
assert not malformed_root.joinpath("shot_001.tmp", "metadata.json").exists()

misleading_root = ROOT / "misleading"


def misleading_capture(_bridge, output_dir, **_kwargs):
    output_dir.joinpath("frames").mkdir(parents=True, exist_ok=True)
    metadata = {
        "capture_contract": "physics_capture_v1",
        "schema_version": "physics_capture_v1",
        "protocol_version": 1,
        "frame_count": 1,
        "sidecars_closed": True,
    }
    output_dir.joinpath("metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


with (
    patch("scripts.collect_rollouts.capture_physics_rollout", side_effect=misleading_capture),
    patch("scripts.collect_rollouts._run_pre_shot_guard", return_value=guard),
):
    misleading_manifest = collect_rollouts(
        Bridge(),
        misleading_root,
        [{"coordinate_frame": "absolute", "release": [130, 150]}],
        target_fps=1,
        duration_seconds=1,
        max_frames=1,
        anchor_actions=False,
        physics_capture_v1=True,
        physics_player_sha256="a" * 64,
        physics_protocol_sha256="b" * 64,
        physics_archive_sha256="c" * 64,
    )
assert misleading_manifest["rollout_count"] == 0
assert not misleading_root.joinpath("shot_001").exists()
assert not misleading_root.joinpath("shot_001.tmp").exists()

stale_root = ROOT / "stale"
stale_root.joinpath("shot_099.tmp").mkdir(parents=True)
assert cleanup_incomplete_physics_attempts(stale_root) == ("shot_099.tmp",)
assert not stale_root.joinpath("shot_099.tmp").exists()

result = {
    "status": "passed",
    "generated_enriched_desktop": {
        "accepted_final": str(accepted_root / "shot_001"),
        "temporary_absent": True,
        "recorder_shot_count": accepted_bridge.recorder_shots,
        "call_order": accepted_bridge.call_order,
    },
    "malformed_capture": {"final_absent": True, "success_metadata_absent": True, "error": malformed_error},
    "misleading_success": {"final_absent": True, "temporary_absent": True, "rollout_count": 0},
    "stale_tmp": {"removed": "shot_099.tmp"},
}
print(json.dumps(result, indent=2))
shutil.rmtree(ROOT)
assert not ROOT.exists()
print(json.dumps({"cleanup_status": "passed", "removed_root": str(ROOT)}))
