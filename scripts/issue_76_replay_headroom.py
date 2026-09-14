"""Describe native task headroom; never score predictors or authorize advancement."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time

from scripts import validate_issue_76_action_replay as replay
from scripts.issue_76_native_outcomes import terminal_evidence


def endpoint_counts(sample, terminal):
    counts = Counter(entity["scenario_object_id"].split(":")[0]
                     for entity in sample["entities"] if entity["lifecycle"] == "active")
    reason = None if terminal is None else terminal["reason"]
    observed_cost = 1000 * counts["pig"] + counts["block"]
    usable = reason in ("stable_entered", "level_clear")
    return {"active_pigs": counts["pig"], "active_blocks": counts["block"],
            "observed_count_cost": observed_cost, "terminal_reason": reason,
            "settled_count_cost": observed_cost if usable else None,
            "ranking_failure": not usable, "ranking_cost": observed_cost if usable else 1e9,
            "native_level_clear": reason == "level_clear"}


def summarize(rows):
    candidates = [row for row in rows if not row["original_action_reference"]]
    settled = sorted({row["settled_count_cost"] for row in candidates
                      if row["settled_count_cost"] is not None})
    costs = [row["ranking_cost"] for row in candidates]
    best = min(costs)
    return {"candidate_count": len(candidates), "settled_count_costs": settled,
            "count_discriminating": len(settled) > 1,
            "settled_count_range": max(settled) - min(settled) if settled else None,
            "all_failed": all(row["ranking_failure"] for row in candidates),
            "all_ranking_costs_tied": len(set(costs)) == 1,
            "best_cost_tie_count": costs.count(best),
            "best_action_ordinals": [row["candidate_ordinal"] for row in candidates if row["ranking_cost"] == best],
            "native_level_clears": sum(row["native_level_clear"] for row in candidates)}


def publish(root=replay.run.ROOT):
    started = time.monotonic()
    plan = replay.run.load_plan(root)
    validation = replay.run.files.read(root / "collection-validation.json")
    members = plan["inventory"]["members"]
    if (not validation["all_capture_contracts_validated"]
            or validation["member_identities"] != [member["identity"] for member in members]):
        raise ValueError("headroom requires the complete validated assignment inventory")
    rows = []
    for member in members:
        result = replay.run.files.read(root / "results" / (member["identity"] + ".json"))
        segment, = result["segments"]
        terminal = terminal_evidence(segment)
        native = Path(segment["native_root"])
        manifest = replay.run.files.read(native / "native-manifest.json")
        with gzip.open(native / manifest["chunks"][-1]["path"], "rt") as stream:
            sample = json.load(stream)["fixed_step_samples"][-1]
        if sample["fixed_step"] != manifest["last_fixed_step"]:
            raise ValueError("task endpoint differs from the validated native endpoint")
        rows.append({"member_identity": member["identity"],
                     **{key: member[key] for key in ("source_member_identity", "candidate_ordinal", "original_action_reference")},
                     "terminal_evidence": terminal, "fixed_step": sample["fixed_step"],
                     "native_root": str(native), **endpoint_counts(sample, terminal)})
    groups = []
    for group in validation["groups"]:
        selected = [row for row in rows if row["source_member_identity"] == group["source_member_identity"]]
        groups.append({"source_member_identity": group["source_member_identity"],
                       "paired_ranking_admissible": group["paired_ranking_admissible"],
                       "rows": selected, **summarize(selected)})
    elapsed = time.monotonic() - started
    prior = validation["validation_wall_seconds"] + 0.5871588701847941
    report = {"identity": "issue-76-replay-headroom-v1", "groups": groups,
              "source_text": {"scripts/issue_76_replay_headroom.py": Path(__file__).read_text()},
              "native_level_clears": sum(row["native_level_clear"] for row in rows),
              "admissible_count_discriminating_lineages": sum(group["paired_ranking_admissible"] and group["count_discriminating"] for group in groups),
              "endpoint_count_authority": "native active lifecycle; not predicted carrier presence",
              "wall_seconds": elapsed, "prior_offline_preparation_seconds": prior,
              "remaining_offline_preparation_seconds": plan["inventory"]["limits"]["offline_preparation_wall_seconds"] - prior - elapsed,
              "model_scores_measured": False, "fresh_access": False, "advancement_authorized": False}
    replay.immutable(root / "headroom.json", report)
    replay.immutable(replay.OUTPUT / "headroom.json", report)
    print({key: report[key] for key in ("native_level_clears", "admissible_count_discriminating_lineages", "wall_seconds")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    if parser.parse_args().publish:
        publish()
    else:
        print("No-write: --publish describes existing validated training replays only.")
