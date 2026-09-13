"""Assigned and observed training-action support; no outcome or predictor scores."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

from scripts import run_issue_76_native_refit as fit
from scripts.run_issue_76_order_probe import immutable
from world_model.data.corrective_ranking_cohort import action_bounds
from world_model.planning.gameplay import SlingshotActionBounds
from world_model.training.action_ranking_probe import broad_action_candidates
from world_model.training.native_history_data import action_context

ROOT = fit.files.ROOT / ".local-artifacts/issue-76-action-support-v1"
OUTPUT = fit.files.ROOT / "data/issue-76-action-support"
FIELDS = ("drag_x", "drag_y", "release_time_ms", "tap_time_ms")


def action_key(action):
    return tuple(action[field] for field in FIELDS)


def count_rows(counts):
    return [{"action": dict(zip(FIELDS, key)), "count": count} for key, count in sorted(counts.items())]


def assigned_support(members, bounds):
    assigned, first, by_family_shot = Counter(), Counter(), defaultdict(Counter)
    for member in members:
        for ordinal, action in enumerate(member["actions"], 1):
            key = action_key(action)
            assigned[key] += 1
            by_family_shot[member["generator_family"], ordinal][key] += 1
            if ordinal == 1:
                first[key] += 1
    candidates = broad_action_candidates("issue-76-action-support-metadata", bounds)
    grid = []
    for candidate in candidates:
        action = dict(drag_x=candidate.action.drag_x, drag_y=candidate.action.drag_y,
                      release_time_ms=bounds.release_time_ms, tap_time_ms=candidate.action.tap_time_ms)
        key = action_key(action)
        grid.append({"ordinal": candidate.ordinal, "action": action,
                     "assigned_first_shots": first[key], "assigned_all_shots": assigned[key]})
    return {"assigned_members": len(members), "assigned_first_actions": count_rows(first),
            "assigned_all_actions": count_rows(assigned),
            "family_shot_support": [{"family": family, "shot": shot, "actions": count_rows(counts)}
                                    for (family, shot), counts in sorted(by_family_shot.items())],
            "grid": grid, "grid_actions_seen_at_first_shot": sum(row["assigned_first_shots"] > 0 for row in grid),
            "grid_actions_seen_at_any_shot": sum(row["assigned_all_shots"] > 0 for row in grid)}


def observed_training_support(members, entries, load_shard, check):
    source = {member["identity"]: member for member in members}
    counts, first, groups = Counter(), Counter(), defaultdict(Counter)
    records = []
    for entry in entries:
        if entry["exposure_role"] != "training":
            raise ValueError("observed-action audit may read training shards only")
        check()
        member = source[entry["member_identity"]]
        if member["exposure_role"] != "training":
            raise ValueError("collection and shard exposure roles differ")
        actions = []
        if entry["usable"]:
            shard = load_shard(entry)
            for index, segment in enumerate(shard["segment_ranges"]):
                action = member["actions"][index]
                if not torch.equal(segment["action"], action_context(action)):
                    raise ValueError("observed training action differs from its assigned shot")
                key = action_key(action)
                counts[key] += 1
                groups[member["generator_family"], index + 1][key] += 1
                if index == 0:
                    first[key] += 1
                actions.append(action)
        records.append({"member_identity": entry["member_identity"], "family": member["generator_family"],
                        "usable": entry["usable"], "observed_actions": actions})
    return {"assigned_members": len(entries), "observed_first_actions": count_rows(first),
            "observed_all_actions": count_rows(counts), "member_inventory": records,
            "family_shot_support": [{"family": family, "shot": shot, "actions": count_rows(values)}
                                    for (family, shot), values in sorted(groups.items())]}


def audit():
    collection_path = fit.files.ROOT / ".local-artifacts/issue-76-native-development-v1/plan.json"
    data_root = fit.files.ROOT / ".local-artifacts/issue-76-repaired-representation-v1"
    collection, data_plan = fit.files.read(collection_path), fit.files.read(data_root / "plan.json")
    index = fit.files.read(data_root / "data-index.json")
    if index["plan_identity"] != data_plan["identity"]:
        raise ValueError("training index source differs")
    training = [member for member in collection["members"] if member["exposure_role"] == "training"]
    entries = [entry for entry in index["entries"] if entry["exposure_role"] == "training"]
    if [member["identity"] for member in training] != [entry["member_identity"] for entry in entries]:
        raise ValueError("training assignment inventory differs")
    bounds = action_bounds()
    # Native live loader accepts caller-provided bounds; its unit-test grid is a
    # different design, not a frozen learned-gameplay protocol.
    fixture_bounds = SlingshotActionBounds((-160, -40), (-80, 80), (0, 0), 600)
    with fit.fitting_budget(ROOT, "training-action-metadata", 60, "cpu") as budget:
        observed = observed_training_support(training, entries,
            lambda entry: fit.load_shard(data_root, entry, data_plan, fitting=True), budget.check)
        budget.check()
    roles = {role: assigned_support([m for m in collection["members"] if m["exposure_role"] == role], bounds)
             for role in ("training", "calibration", "model_selection")}
    return {"identity": ROOT.name, "collection_plan_identity": collection["identity"],
        "collection_plan_path": str(collection_path), "data_plan_identity": data_plan["identity"],
        "assigned_by_role": roles, "observed_training": observed,
        "historical_broad_grid_bounds": asdict(bounds),
        "native_live_test_grid_training_support": assigned_support(training, fixture_bounds),
        "native_learned_gameplay_bounds_frozen_by_this_audit": False,
        "counterfactual_same_initial_state_replays_in_source_plan": False,
        "scope": "native 350-lineage source; later shots are different decision states, not first-shot counterfactuals",
        "interpretation": "exact support only; absent exact actions do not by themselves prove failed interpolation or causally explain hybrid failure",
        "source_text": {path: (fit.files.ROOT / path).read_text() for path in (
            "scripts/audit_issue_76_action_support.py", "tests/test_issue_76_action_support.py",
            "scripts/run_issue_76_development.py", "scripts/run_issue_76_canonical_smoke.py",
            "scripts/issue_76_live_models.py", "world_model/training/action_ranking_probe.py")},
        "budget": fit.require_finished_budget(ROOT, "training-action-metadata"),
        "outcome_fields_inspected": False, "prediction_scores_computed": False, "optimizer_updates": 0,
        "new_captures": 0, "fresh_access": False, "advancement_authorized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if not args.audit:
        print("No-write: action metadata only; no outcomes, predictions, fitting or collection.")
        return
    if (ROOT / "report.json").exists():
        print("Retaining completed action-support audit; no shard reads repeated.")
        return
    torch.set_num_threads(1)
    value = audit()
    immutable(ROOT / "report.json", value)
    immutable(OUTPUT / "report.json", value)
    print("Action support audited; no performance or advancement claim.", flush=True)


if __name__ == "__main__":
    main()
