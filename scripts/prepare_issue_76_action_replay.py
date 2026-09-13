"""Freeze training-lineage action-replay assignments without rendering or scoring."""
import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from scripts import run_issue_76_native_refit as fit
from scripts.run_issue_76_development import FAMILIES
from scripts.run_issue_76_order_probe import immutable
from world_model.data.corrective_ranking_cohort import action_bounds
from world_model.training.action_ranking_probe import broad_action_candidates

ROOT = fit.files.ROOT / ".local-artifacts/issue-76-action-replay-v1"
OUTPUT = fit.files.ROOT / "data/issue-76-action-replay"


def replay_members(source_members):
    """First assigned training member per family; no usability/outcome selection."""
    selected = [next(member for member in source_members if member["exposure_role"] == "training"
                     and member["generator_family"] == family) for family in FAMILIES]
    bounds = action_bounds()
    replays = []
    for source in selected:
        if source["novelty_level"] != 0:
            raise ValueError("this action-replay diagnostic is normal-mechanics training only")
        actions = [(0, source["actions"][0])]
        actions.extend((candidate.ordinal, {**asdict(candidate.action), "release_time_ms": bounds.release_time_ms})
                       for candidate in broad_action_candidates(source["identity"], bounds))
        for candidate_ordinal, action in actions:
            member = deepcopy(source)
            member.update(identity=f"issue-76-action-replay-{source['ordinal']:03d}-a{candidate_ordinal:02d}",
                ordinal=len(replays) + 1, source_member_identity=source["identity"],
                candidate_ordinal=candidate_ordinal, original_action_reference=candidate_ordinal == 0,
                maximum_shots=1, actions=[dict(action)])
            # Source identity for the scenario/lineage is independent of the new
            # rollout identity; XML, scenario, seeds and base cluster stay exact.
            replays.append(member)
    return replays


def make_inventory():
    path = fit.files.ROOT / ".local-artifacts/issue-76-native-development-v1/plan.json"
    source = fit.files.read(path)
    members = replay_members(source["members"])
    player = path.parent / "player"
    if not (player / "9001.x86_64").is_file():
        raise ValueError("existing native development player is missing")
    files = ("scripts/prepare_issue_76_action_replay.py", "tests/test_issue_76_action_replay_inventory.py",
             "docs/issue-76-action-replay-protocol.md", "scripts/issue_76_live_episode.py",
             "scripts/issue_76_censored_episode.py", "scripts/issue_76_native_outcomes.py",
             "world_model/training/action_ranking_probe.py", "world_model/planning/task_objective.py")
    return {"identity": ROOT.name, "source_collection_plan_identity": source["identity"],
        "source_collection_plan_path": str(path), "player_source": str(player),
        "source_member_identities": list(dict.fromkeys(member["source_member_identity"] for member in members)),
        "members": members, "action_bounds": asdict(action_bounds()),
        "capture_order": "family order; original-action reference first, then candidate ordinals 1 through 12",
        "smoke_member_identities": [member["identity"] for member in members[:2]],
        "limits": {"workers": 1, "new_shots_max": 65, "smoke_shots_max": 2,
            "collection_wall_seconds": 14400, "smoke_wall_seconds": 1200,
            "attempt_seconds": 420, "shot_seconds": 180, "cpu_rss_mib": 4096,
            "artifact_bytes": 32 * 2**30, "technical_retries": 0, "engine_speed": 1,
            "native_step_seconds": .0004, "native_steps_per_shot_max": 30000, "rgb_stride_native_steps": 50,
            "rgb_frames_per_shot_max": 601, "offline_preparation_wall_seconds": 1800,
            "offline_scoring_cuda_wall_seconds": 900},
        "source_collection_budget_reference": str(path.parent / "budget.json"),
        "source_collection_recorded_bytes": fit.files.read(path.parent / "budget.json")["artifact_bytes"],
        "ranking": {"candidate_ordinals": list(range(1, 13)), "reference_ordinal": 0,
            "objective": "1000 * native active pig count + native active block count",
            "target": "actually observed physical terminal, not an invented 4.5-second carrier",
            "failure_cost": 1e9, "all_failed_states_informative": False,
            "report": ["absolute count regret", "normalized regret", "top1", "top3", "ties",
                       "action-discriminating lineages", "native level-clear evidence", "censored and failed assignments"],
            "five_training_lineages_not_a_power_or_advancement_test": True},
        "capture_execution_authorized": False, "model_scoring_authorized": False,
        "pending": ["source-bound execution driver and tests", "initial-state comparability checks",
                    "bounded rendered smoke and independent smoke validation", "offline scorer freeze before model scores"],
        "new_training_updates": 0, "fresh_access": False, "issue_64_authorized": False,
        "source_text": {name: (fit.files.ROOT / name).read_text() for name in files}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if not args.prepare:
        print("No-write: five existing training lineages, 12 ranked actions plus one reference each; capture disabled.")
        return
    value = make_inventory()
    immutable(ROOT / "inventory.json", value)
    immutable(OUTPUT / "inventory.json", value)
    print("65 training replay assignments frozen; no rendering, scoring, fitting or fresh access.")


if __name__ == "__main__":
    main()
