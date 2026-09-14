"""Prospective upward-angle action-design inventory; no capture or model access."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import math

from scripts import prepare_issue_76_action_replay as previous
from world_model.planning.gameplay import SlingshotAction

ROOT = previous.fit.files.ROOT / ".local-artifacts/issue-76-angular-replay-v1"
OUTPUT = previous.fit.files.ROOT / "data/issue-76-angular-replay"
ANGLES = tuple(range(5, 83, 7))


def angular_actions():
    bounds = previous.action_bounds()
    actions = [SlingshotAction(round(-80 * math.cos(math.radians(angle))),
                              round(80 * math.sin(math.radians(angle))), 0) for angle in ANGLES]
    if len(set((action.drag_x, action.drag_y) for action in actions)) != 12 or not all(bounds.contains(action) for action in actions):
        raise ValueError("angular grid must contain twelve distinct legal actions")
    return [{**asdict(action), "release_time_ms": bounds.release_time_ms} for action in actions]


def angular_members(original):
    actions = angular_actions()
    members = deepcopy(original)
    for member in members:
        member["identity"] = member["identity"].replace("issue-76-action-replay-", "issue-76-angular-replay-")
        if not member["original_action_reference"]:
            member["actions"] = [dict(actions[member["candidate_ordinal"] - 1])]
    return members


def prepare():
    source = previous.fit.files.read(previous.ROOT / "inventory.json")
    measured = previous.fit.files.read(previous.ROOT / "capture-budget.json")
    members = angular_members(source["members"])
    files = ("scripts/prepare_issue_76_angular_replay.py", "tests/test_issue_76_angular_inventory.py",
             "docs/issue-76-angular-replay-protocol.md")
    value = {"identity": ROOT.name, "parent_inventory_identity": source["identity"],
             "parent_inventory_path": str(previous.ROOT / "inventory.json"),
             "source_collection_plan_path": source["source_collection_plan_path"],
             "source_collection_plan_identity": source["source_collection_plan_identity"],
             "player_source": source["player_source"], "members": members,
             "angles_degrees": ANGLES, "commanded_radius_pixels": 80,
             "action_bounds": source["action_bounds"], "ranking": source["ranking"],
             "smoke_member_identities": [member["identity"] for member in members[:2]],
             "limits": {**source["limits"], "offline_scoring_cuda_wall_seconds": 0},
             "prior_collection_measured_cost": measured,
             "pilot_viability": {"minimum_comparable_lineages": 4,
                                 "minimum_comparable_lineages_with_native_clear": 3,
                                 "authorizes_training_or_fresh_access": False},
             "capture_execution_authorized": False, "model_scoring_authorized": False,
             "new_training_updates": 0, "fresh_access": False, "issue_64_authorized": False,
             "source_text": {name: (previous.fit.files.ROOT / name).read_text() for name in files}}
    previous.immutable(ROOT / "inventory.json", value)
    previous.immutable(OUTPUT / "inventory.json", value)
    print("65 angular replay assignments frozen; capture, model scoring and fitting disabled.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    if parser.parse_args().prepare:
        prepare()
    else:
        print("No-write: metadata-only angular action-design pilot.")
