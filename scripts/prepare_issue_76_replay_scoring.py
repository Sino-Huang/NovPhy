"""Bind existing model pool and common causal anchors; scoring remains disabled."""
import argparse
from pathlib import Path
import time

from scripts import validate_issue_76_action_replay as replay
from scripts.observation_trace import MANIFEST_NAME


def prepare(root=replay.run.ROOT):
    started = time.monotonic()
    plan = replay.run.load_plan(root)
    validation = replay.run.files.read(root / "collection-validation.json")
    headroom = replay.run.files.read(root / "headroom.json")
    pool_path = replay.run.files.ROOT / ".local-artifacts/issue-76-fixed-development-v1/inventory.json"
    pool = replay.run.files.read(pool_path)
    anchors = []
    for group in validation["groups"]:
        members = [member for member in plan["inventory"]["members"]
                   if member["source_member_identity"] == group["source_member_identity"]]
        reference, = [member for member in members if member["original_action_reference"]]
        observation_root = root / "attempts" / reference["identity"] / "decision-1"
        manifest = replay.run.files.read(observation_root / MANIFEST_NAME)
        frame, = manifest["frame_records"]
        result = replay.run.files.read(root / "results" / (reference["identity"] + ".json"))
        image = observation_root / frame["agent_observation"]["relative_path"]
        if (manifest["identity"] != result["decisions"][0]["observation_manifest"]
                or str(image) != validation["decision_image_paths"][reference["identity"]]
                or frame["fixed_time_seconds"] != result["policy"]["decision_time"]
                or not image.is_file()):
            raise ValueError("common anchor is not the validated genuine decision observation")
        candidates = [member for member in members if not member["original_action_reference"]]
        anchors.append({"source_member_identity": group["source_member_identity"],
                        "reference_member_identity": reference["identity"],
                        "paired_ranking_admissible": group["paired_ranking_admissible"],
                        "observation_manifest_identity": manifest["identity"],
                        "decision_image_path": str(image), "timestamp": frame["fixed_time_seconds"],
                        "candidate_members": candidates,
                        "history_initialization": "reset; one genuine decision RGB observation; no executed action events",
                        "candidate_postdecision_observations_used": False})
    for model in pool["models"]:
        for path in model["checkpoint_paths"]:
            if not Path(path).is_file():
                raise ValueError(f"retained model checkpoint missing: {path}")
    elapsed = time.monotonic() - started
    value = {"identity": "issue-76-replay-scoring-input-inventory-v1",
             "source_text": {"scripts/prepare_issue_76_replay_scoring.py": Path(__file__).read_text()},
             "model_pool_source": str(pool_path), "model_pool_identity": pool["identity"],
             "models": pool["models"], "seeds": pool["seeds"], "anchors": anchors,
             "failed_training_qualifications_preserved": pool["failed_training_qualifications_preserved"],
             "model_scoring_authorized": False, "scoring_implementation_freeze_pending": True,
             "parser_history_checkpoint_binding_pending": True,
             "new_optimizer_updates": 0, "fresh_access": False,
             "preparation_wall_seconds": elapsed,
             "remaining_offline_preparation_seconds": headroom["remaining_offline_preparation_seconds"] - elapsed}
    replay.immutable(root / "scoring-input-inventory.json", value)
    replay.immutable(replay.OUTPUT / "scoring-input-inventory.json", value)
    print({"model_instances": len(value["models"]), "common_anchors": len(anchors),
           "model_scoring_authorized": False, "preparation_wall_seconds": elapsed})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    if parser.parse_args().prepare:
        prepare()
    else:
        print("No-write: --prepare binds metadata only; no model scoring or fitting.")
