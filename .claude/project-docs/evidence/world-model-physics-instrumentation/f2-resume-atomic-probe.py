import hashlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from scripts.collect_rollouts import cleanup_incomplete_physics_attempts, recover_physics_capture_attempts
from scripts.physics_rollout_contract import CaptureProvenance
from scripts.physics_rollout_persistence import persist_physics_rollout
from scripts.rollout_artifacts import validate_physics_shot_artifact
from src.webui.bridge import PhysicsCaptureV1


FIXTURE = ROOT / "tests/fixtures/physics_capture_v1"
STATES = [json.loads(line) for line in (FIXTURE / "physics_state.jsonl").read_text().splitlines()]
EVENTS = [json.loads(line) for line in (FIXTURE / "physics_events.jsonl").read_text().splitlines()]
for record in STATES + EVENTS:
    record["shot_id"] = "shot_001"


class Bridge:
    def __init__(self):
        self.index = 0
        self.pngs = []
        for rgb in ((30, 20, 10), (90, 20, 10)):
            stream = io.BytesIO()
            Image.new("RGB", (640, 480), rgb).save(stream, format="PNG")
            self.pngs.append(stream.getvalue())

    def get_physics_capture_v1(self):
        index = self.index
        self.index += 1
        return PhysicsCaptureV1(self.pngs[index], STATES[index + 1], tuple(EVENTS if index == 0 else ()))


def tree_hash(path: Path):
    entries = []
    for candidate in sorted(path.rglob("*")):
        relative = str(candidate.relative_to(path))
        if candidate.is_symlink():
            entries.append((relative, "symlink", str(candidate.readlink())))
        elif candidate.is_file():
            entries.append((relative, "file", hashlib.sha256(candidate.read_bytes()).hexdigest()))
        elif candidate.is_dir():
            entries.append((relative, "dir", ""))
    return entries


def outcome(call):
    try:
        return {"status": "accepted", "value": str(call())}
    except Exception as error:  # noqa: BLE001 - machine-readable QA result
        return {"status": "rejected", "type": type(error).__name__, "message": str(error)}


report = {"probe": "f2-resume-atomic-audit", "repo": str(ROOT)}
with tempfile.TemporaryDirectory(prefix="novphy-f2-") as temporary:
    root = Path(temporary)
    report["temp_root_during"] = str(root)
    temporary_shot = root / "shot_001.tmp"
    persist_physics_rollout(
        Bridge(), temporary_shot, target_fps=2, duration_seconds=1, max_frames=2,
        state_header=STATES[0], provenance=CaptureProvenance("a" * 64, "b" * 64, "c" * 64),
    )
    temporary_shot.replace(root / "shot_001")
    final = root / "shot_001"
    before = tree_hash(final)
    first_validation = outcome(lambda: validate_physics_shot_artifact(final))
    recovery_one = recover_physics_capture_attempts(root)
    after_one = tree_hash(final)
    recovery_two = recover_physics_capture_attempts(root)
    after_two = tree_hash(final)
    report["accepted_final"] = {
        "validation": first_validation,
        "final_exists": final.is_dir(),
        "recovery_one": {"removed_temporary": recovery_one.removed_temporary, "quarantined": recovery_one.quarantined},
        "recovery_two": {"removed_temporary": recovery_two.removed_temporary, "quarantined": recovery_two.quarantined},
        "hash_stable_after_recovery_one": before == after_one,
        "hash_stable_after_recovery_two": before == after_two,
        "tree_hash_entries": len(before),
    }

    interrupted = root / "shot_002.tmp"
    interrupted.mkdir()
    (interrupted / "physics_state.jsonl").write_text('{"truncated":')
    report["repeated_tmp_cleanup"] = {
        "first_removed": cleanup_incomplete_physics_attempts(root),
        "second_removed": cleanup_incomplete_physics_attempts(root),
        "exists_after": interrupted.exists(),
    }

    sidecars = {}
    for name in ("physics_state.jsonl", "physics_events.jsonl"):
        case = root / name.replace(".jsonl", "_symlink_case")
        shutil.copytree(final, case, symlinks=True)
        external = root.parent / f"{root.name}-{name}"
        shutil.copy2(final / name, external)
        (case / name).unlink()
        (case / name).symlink_to(external)
        sidecars[name] = {
            "external": str(external),
            "is_symlink": (case / name).is_symlink(),
            "validator": outcome(lambda case=case: validate_physics_shot_artifact(case)),
        }
    traversal = root / "metadata_traversal"
    shutil.copytree(final, traversal, symlinks=True)
    metadata = json.loads((traversal / "metadata.json").read_text())
    metadata["physics_state_path"] = "../outside.jsonl"
    (traversal / "metadata.json").write_text(json.dumps(metadata))
    malformed = root / "malformed"
    malformed.mkdir()
    (malformed / "metadata.json").write_text("{bad")
    outside_tmp = root / "outside_tmp"
    outside_tmp.mkdir()
    symlink_tmp = root / "shot_999.tmp"
    symlink_tmp.symlink_to(outside_tmp, target_is_directory=True)
    report["sidecar_path_cases"] = {
        "symlinks": sidecars,
        "metadata_parent_traversal": outcome(lambda: validate_physics_shot_artifact(traversal)),
    }
    report["malformed_case"] = outcome(lambda: validate_physics_shot_artifact(malformed))
    report["symlink_tmp_cleanup"] = {
        "removed": cleanup_incomplete_physics_attempts(root),
        "symlink_still_exists": symlink_tmp.is_symlink(),
        "target_still_exists": outside_tmp.exists(),
    }
    report["root_listing_after"] = sorted(path.name for path in root.iterdir())
    report["accepted_tree_hash"] = before
report["temp_root_absent_after_context"] = not Path(report["temp_root_during"]).exists()
print(json.dumps(report, indent=2, sort_keys=True))
