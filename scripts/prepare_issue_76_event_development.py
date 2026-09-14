"""Freeze causal event-development assignments using existing lineage roles only."""
import argparse
from collections import Counter
from copy import deepcopy

from scripts import prepare_issue_76_angular_replay as angular
from scripts.run_issue_76_development import FAMILIES

files = angular.previous.fit.files
ROOT = files.ROOT / ".local-artifacts/issue-76-event-development-v1"
OUTPUT = files.ROOT / "data/issue-76-event-development"
PROTOCOL = "docs/issue-76-event-development-protocol.md"
ROLES = ("training", "calibration", "model_selection")


def select_sources(members):
    selected = []
    for family in FAMILIES:
        for role in ROLES:
            values = sorted((m for m in members if m["generator_family"] == family
                             and m["exposure_role"] == role), key=lambda m: m["identity"])
            expected = 50 if role == "training" else 10
            if len(values) != expected:
                raise ValueError("event development requires the exact existing role/family population")
            ranks = [i * 49 // 19 for i in range(20)] if role == "training" else range(10)
            selected.extend(deepcopy(values[i]) for i in ranks)
    return selected


def assignments(sources):
    actions = angular.angular_actions()
    rows = []
    for source in sources:
        for ordinal, action in enumerate([source["actions"][0], *actions]):
            rows.append({"identity": f"issue-76-event-development-{source['ordinal']:03d}-a{ordinal:02d}",
                         "ordinal": len(rows) + 1, "source_member_identity": source["identity"],
                         "candidate_ordinal": ordinal, "original_action_reference": ordinal == 0,
                         "action": dict(action)})
    return rows


def materialize_assignment(source, assignment):
    if source["identity"] != assignment["source_member_identity"]:
        raise ValueError("assignment source differs")
    member = deepcopy(source)
    member.update({key: assignment[key] for key in
                   ("identity", "ordinal", "source_member_identity", "candidate_ordinal", "original_action_reference")})
    member.update(maximum_shots=1, actions=[dict(assignment["action"])])
    return member


def prepare(root=ROOT, output=OUTPUT):
    source_path = files.ROOT / ".local-artifacts/issue-76-native-development-v1/plan.json"
    source = files.read(source_path)
    sources = select_sources(source["members"])
    rows = assignments(sources)
    first_per_cell = {}
    for member in sources:
        first_per_cell.setdefault((member["generator_family"], member["exposure_role"]), member["identity"])
    smoke = [row["identity"] for row in rows if row["source_member_identity"] in first_per_cell.values()
             and row["candidate_ordinal"] in (0, 1)]
    prior = files.read(angular.ROOT / "capture-budget.json")
    sources_to_freeze = ("scripts/prepare_issue_76_event_development.py",
                         "tests/test_issue_76_event_inventory.py", PROTOCOL,
                         "scripts/prepare_issue_76_angular_replay.py")
    value = {"identity": root.name, "source_collection_plan_identity": source["identity"],
             "source_collection_plan_path": str(source_path), "source_members": sources,
             "assignments": rows, "smoke_member_identities": smoke,
             "selection": "per family: training sorted ranks floor(i*49/19), i=0..19; all calibration/model-selection",
             "player_source": str(source_path.parent / "player"),
             "limits": {"workers": 8, "aggregate_cpu_rss_mib": 32768, "worker_cpu_rss_mib": 4096,
                        "collection_wall_seconds": 172800, "smoke_wall_seconds": 3600,
                        "attempt_seconds": 420, "shot_seconds": 180, "artifact_bytes": 768 * 2**30,
                        "minimum_free_bytes_before_attempt": 256 * 2**30,
                        "new_shots_max": 2600, "smoke_shots_max": 30, "technical_retries": 0,
                        "native_step_seconds": .0004, "native_steps_per_shot_max": 30000,
                        "rgb_stride_native_steps": 50, "rgb_frames_per_shot_max": 601,
                        "offline_validation_wall_seconds": 43200, "offline_media_wall_seconds": 14400},
             "planning_reference": {"prior_65_assignment_budget": prior,
                                    "linear_serial_seconds": prior["active_seconds"] * 40,
                                    "ideal_eight_worker_seconds": prior["active_seconds"] * 5,
                                    "linear_artifact_bytes": prior["artifact_bytes"] * 40,
                                    "parallel_speedup_guaranteed": False},
             "data_readiness": {"minimum_valid_capture_fraction_each_role_family": .9,
                                "minimum_clear_training_lineages": 10,
                                "minimum_clear_calibration_lineages": 5,
                                "minimum_clear_model_selection_lineages": 5,
                                "authorizes_training_or_fresh_access": False},
             "capture_execution_authorized": False, "model_training_authorized": False,
             "new_optimizer_updates": 0, "fresh_access": False, "advancement_authorized": False,
             "source_text": {name: (files.ROOT / name).read_text() for name in sources_to_freeze}}
    angular.previous.immutable(root / "inventory.json", value)
    angular.previous.immutable(output / "inventory.json", value)
    print({"source_lineages": len(sources), "roles": dict(Counter(m["exposure_role"] for m in sources)),
           "assignments": len(rows), "smoke_assignments": len(smoke), "capture_authorized": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    if parser.parse_args().prepare:
        prepare()
    else:
        print("No-write: --prepare freezes existing-role metadata only; no capture or fitting.")
