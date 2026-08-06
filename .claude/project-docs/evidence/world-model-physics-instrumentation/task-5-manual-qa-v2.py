import hashlib
import io
import json
import os
import shutil
from pathlib import Path

from PIL import Image

from scripts.collect_rollouts import capture_physics_rollout, cleanup_incomplete_physics_attempts, recover_physics_capture_attempts, validate_rollout_artifact
from scripts.rollout_artifacts import validate_physics_shot_artifact, validate_rollout_episode
from scripts.rollout_validation_types import EpisodeAccepted, EpisodeValidationContract, PhysicsArtifactError
from src.webui.bridge import PhysicsCaptureV1
from world_model.data.types import PHYSICS_CAPTURE_V1


qa_root = Path(os.environ["QA_ROOT"])
fixture_root = Path("tests/fixtures/physics_capture_v1")
states = [json.loads(line) for line in (fixture_root / "physics_state.jsonl").read_text(encoding="utf-8").splitlines()]
events = [json.loads(line) for line in (fixture_root / "physics_events.jsonl").read_text(encoding="utf-8").splitlines()]
for record in states + events:
    record["shot_id"] = "shot_000"
states[1]["rgb_frame"].update({"relative_path": "frames/frame_000000.png", "width_pixels": 4, "height_pixels": 3})
png_buffer = io.BytesIO()
Image.new("RGB", (4, 3), (12, 34, 56)).save(png_buffer, format="PNG")
png = png_buffer.getvalue()


class Bridge:
    def get_physics_capture_v1(self):
        return PhysicsCaptureV1(png, states[1], tuple(events))


temporary_shot = qa_root / "shot_000.tmp"
capture_physics_rollout(Bridge(), temporary_shot, target_fps=1, duration_seconds=1, max_frames=1, state_header=states[0], player_sha256="a" * 64, protocol_sha256="b" * 64, archive_sha256="c" * 64)
temporary_validation = validate_rollout_artifact(temporary_shot, capture_contract="physics_capture_v1")
final_shot = qa_root / "shot_000"
temporary_shot.replace(final_shot)
summary = validate_physics_shot_artifact(final_shot)
descriptor = {"contract_name": PHYSICS_CAPTURE_V1.contract_name, "contract_version": PHYSICS_CAPTURE_V1.contract_version, "artifact_layout_version": PHYSICS_CAPTURE_V1.artifact_layout_version, "player_sha256": "a" * 64, "protocol_sha256": "b" * 64, "archive_sha256": "c" * 64, "declared_capabilities": list(PHYSICS_CAPTURE_V1.declared_capabilities), "sidecar_paths": [{"relative_path": sidecar.relative_path, "capabilities": list(sidecar.capabilities)} for sidecar in PHYSICS_CAPTURE_V1.sidecar_paths]}
attempt = {"accepted": True, "attempt_status": "accepted", "artifact_validation": {"accepted": True, "classification": "gameplay-valid", "retryable": False, "retry_decision": "accept"}}
(qa_root / "manifest.json").write_text(json.dumps({"capture_source": "capture_physics_rollout", "replay_mode": "fresh-engine-per-rollout", "target_fps": 1, "duration_seconds": 1, "ui_level": 1, "accepted_rollout_count": 1, "rollout_count": 1, "attempt_count": 1, "attempts": [attempt], "capture_contract": descriptor}), encoding="utf-8")
trial = {"shot_name": "shot_000", "accepted": True, "action": {"drag_start": [100, 200], "drag_release": [-30, 50], "holdTime": 600}}
(qa_root / "action_log.json").write_text(json.dumps({"accepted_trials": [trial]}), encoding="utf-8")
(qa_root / "action_log.jsonl").write_text(json.dumps(trial) + "\n", encoding="utf-8")
canonical = validate_rollout_episode(qa_root, EpisodeValidationContract(1, 1, 1), capture_contract="physics_capture_v1")

resume_cycles = []
for _ in range(2):
    interrupted = qa_root / "shot_001.tmp"
    interrupted.mkdir()
    (interrupted / "physics_state.jsonl").write_text('{"truncated":', encoding="utf-8")
    resume_cycles.append(cleanup_incomplete_physics_attempts(qa_root))

corrupt = qa_root / "shot_001"
shutil.copytree(final_shot, corrupt)
(corrupt / "physics_state.jsonl").write_text('{"truncated":', encoding="utf-8")
recovery = recover_physics_capture_attempts(qa_root)
quarantined = qa_root / recovery.quarantined[0]
try:
    validate_physics_shot_artifact(quarantined)
    rejected = False
except PhysicsArtifactError:
    rejected = True

report = {
    "qa_root": str(qa_root),
    "invocation": "qa_root=$(mktemp -d -p /tmp novphy-task5-qa.XXXXXX) followed by actual persistence driver",
    "accepted_final_exists": final_shot.is_dir(),
    "temporary_absent": not temporary_shot.exists(),
    "temporary_validation": temporary_validation["accepted"],
    "shared_validator_counts": {"states": summary.state_count, "events": summary.event_count},
    "canonical_predicate": isinstance(canonical, EpisodeAccepted),
    "sha256": {"png": hashlib.sha256((final_shot / "frames/frame_000000.png").read_bytes()).hexdigest(), "state": summary.state_sha256, "events": summary.event_sha256},
    "resume_cycles": resume_cycles,
    "corrupt_final_absent": not corrupt.exists(),
    "quarantine_relative_path": recovery.quarantined[0],
    "quarantined_rejected": rejected,
}
shutil.rmtree(qa_root)
report["cleanup"] = {"absolute_root": str(qa_root), "removed": not qa_root.exists()}
print(json.dumps(report, indent=2, sort_keys=True))
