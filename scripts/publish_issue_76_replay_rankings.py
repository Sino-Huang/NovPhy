"""Complete descriptive fixed-policy rankings; no new inference or advancement."""
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
import time

from scripts import run_issue_76_replay_scoring as scoring
from world_model.planning.task_objective import ranking_diagnostic


def ranking(predicted, targets):
    costs = [row["ranking_cost"] for row in targets]
    result = ranking_diagnostic(predicted, costs)
    chosen = None if result["selected"] is None else targets[result["selected"]]
    settled = [row["settled_count_cost"] for row in targets if row["settled_count_cost"] is not None]
    all_failed = all(row["ranking_failure"] for row in targets)
    if all_failed:
        result.update(top1=False, top3=False)
    return {**result, "all_outcomes_failed": all_failed,
            "selected_action_ordinal": None if chosen is None else chosen["candidate_ordinal"],
            "selected_outcome_failure": chosen is None or chosen["ranking_failure"],
            "absolute_penalty_regret": 1e9 if chosen is None else chosen["ranking_cost"] - min(costs),
            "absolute_settled_count_regret": None if chosen is None or chosen["settled_count_cost"] is None
            else chosen["settled_count_cost"] - min(settled),
            "selected_native_level_clear": False if chosen is None else chosen["native_level_clear"],
            "settled_count_discriminating": len(set(settled)) > 1}


def summary(rows):
    counts = [row["absolute_settled_count_regret"] for row in rows
              if row["absolute_settled_count_regret"] is not None]
    return {"decisions": len(rows), "normalized_penalty_regret": mean(row["regret"] for row in rows),
            "absolute_penalty_regret": mean(row["absolute_penalty_regret"] for row in rows),
            "top1_fraction": mean(row["top1"] for row in rows), "top3_fraction": mean(row["top3"] for row in rows),
            "selected_outcome_failures": sum(row["selected_outcome_failure"] for row in rows),
            "prediction_failures": sum(row["failed"] for row in rows),
            "predicted_all_tied": sum(row["predicted_all_tied"] for row in rows),
            "all_outcomes_failed": sum(row["all_outcomes_failed"] for row in rows),
            "available_settled_count_regrets": len(counts),
            "conditional_mean_settled_count_regret": mean(counts) if counts else None,
            "native_level_clears": sum(row["selected_native_level_clear"] for row in rows)}


def publish(root=scoring.replay.run.ROOT):
    started = time.monotonic()
    files = scoring.replay.run.files
    plan = scoring.load_plan(root)
    completion = files.read(root / "scoring-completion.json")
    if (completion["completed_models"] != len(plan["pool"]["models"])
            or completion["budget"]["running"] or completion["budget"]["stopped"]):
        raise ValueError("rankings require the completed bounded scoring inventory")
    headroom = files.read(root / "headroom.json")
    groups = {group["source_member_identity"]: group for group in headroom["groups"]}
    rows, aggregated = [], defaultdict(list)
    for ordinal, cell in enumerate(plan["pool"]["models"], 1):
        record = files.read(root / "fixed-action-scores" / f"model-{ordinal:03d}.json")
        if record["cell"] != cell or record["plan_identity"] != plan["identity"]:
            raise ValueError("score model identity differs from freeze")
        pairs = scoring.fixed.fit.CONTINUOUS_PAIRS if cell["pure"] else scoring.fixed.fit.PAIRS
        expected = [(anchor["source_member_identity"], scoring.fixed.fit.policy_name(pair))
                    for anchor in plan["inputs"]["anchors"] for pair in pairs]
        if [(row["source_member_identity"], row["policy"]) for row in record["rows"]] != expected:
            raise ValueError("score lineage/policy inventory differs from freeze")
        for row in record["rows"]:
            group = groups[row["source_member_identity"]]
            targets = [target for target in group["rows"] if not target["original_action_reference"]]
            if ([candidate["ordinal"] for candidate in row["candidates"]] != list(range(12))
                    or [target["candidate_ordinal"] for target in targets] != list(range(1, 13))
                    or row["paired_ranking_admissible"] != group["paired_ranking_admissible"]):
                raise ValueError("candidate order or comparability changed")
            predicted = [None if candidate["prediction_failed"] else candidate["predicted_cost"]
                         for candidate in row["candidates"]]
            value = {**{key: cell[key] for key in ("recipe", "pure", "seed")},
                     "source_member_identity": row["source_member_identity"], "policy": row["policy"],
                     "paired_ranking_admissible": group["paired_ranking_admissible"],
                     "predicted_costs": predicted, "realized_costs": [target["ranking_cost"] for target in targets],
                     **ranking(predicted, targets)}
            rows.append(value)
            if value["paired_ranking_admissible"]:
                aggregated[cell["pure"], cell["recipe"], row["policy"]].append(value)
    table = [{"pure": pure, "recipe": recipe, "policy": policy, **summary(values)}
             for (pure, recipe, policy), values in aggregated.items()]
    priors = []
    for ordinal in range(12):
        values = []
        for group in groups.values():
            predicted = [1.] * 12
            predicted[ordinal] = 0.
            targets = [row for row in group["rows"] if not row["original_action_reference"]]
            values.append({"source_member_identity": group["source_member_identity"],
                           "paired_ranking_admissible": group["paired_ranking_admissible"],
                           **ranking(predicted, targets)})
        priors.append({"action_ordinal": ordinal + 1, "rows": values,
                       "admissible_summary": summary([row for row in values if row["paired_ranking_admissible"]])})
    report = {"identity": "issue-76-replay-rankings-v1", "scoring_plan_identity": plan["identity"],
              "source_text": {"scripts/publish_issue_76_replay_rankings.py": Path(__file__).read_text()},
              "rows": rows, "admissible_recipe_policy_summaries": table, "all_fixed_action_priors": priors,
              "independent_admissible_lineages": 4, "repeated_model_seeds": 3,
              "selection_optimism_disclosed": True, "confidence_intervals_established": False,
              "adaptive_policy_evaluated": False, "fresh_access": False, "advancement_authorized": False,
              "scoring_completion": completion, "analysis_wall_seconds": time.monotonic() - started}
    scoring.replay.immutable(root / "rankings.json", report)
    scoring.replay.immutable(scoring.replay.OUTPUT / "rankings.json", report)
    print({"policy_lineage_rows": len(rows), "recipe_policy_summaries": len(table), "fixed_priors": len(priors),
           "analysis_wall_seconds": report["analysis_wall_seconds"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    if parser.parse_args().publish:
        publish()
    else:
        print("No-write: --publish analyzes existing scores only.")
