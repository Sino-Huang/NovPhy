"""Source-frozen fixed-pair action scoring on the retained training replays."""
import argparse
from pathlib import Path
import time

import torch

from scripts import validate_issue_76_action_replay as replay
from scripts import score_issue_76_fixed_development as fixed
from scripts.issue_76_replay_representation import load_representation, repaired
from world_model.planning.native_gameplay import agent_image_tensor
from world_model.planning.replay_action_scoring import score_actions
from world_model.training.cnn_hybrid import linear_macs

SOURCES = (*fixed.fit.SOURCES, "scripts/run_issue_76_replay_scoring.py",
           "scripts/score_issue_76_fixed_development.py", "scripts/issue_76_replay_representation.py",
           "scripts/run_issue_76_repaired_representation.py", "world_model/planning/native_gameplay.py",
           "world_model/planning/replay_action_scoring.py", "world_model/planning/task_objective.py",
           "world_model/training/fixed_development.py", "tests/test_replay_action_scoring.py",
           "tests/test_issue_76_replay_scoring_driver.py", "docs/issue-76-replay-scoring-protocol.md")


def prepare(root=replay.run.ROOT):
    inputs = replay.run.files.read(root / "scoring-input-inventory.json")
    representation = replay.run.files.read(root / "representation-binding.json")
    pool = replay.run.files.read(Path(inputs["model_pool_source"]))
    if pool["identity"] != inputs["model_pool_identity"] or pool["models"] != inputs["models"]:
        raise ValueError("retained model pool changed")
    plan = {"identity": "issue-76-replay-fixed-scoring-execution-v1", "inputs": inputs,
            "representation": representation, "pool": pool,
            "source_text": {path: (replay.run.files.ROOT / path).read_text() for path in SOURCES},
            "endpoint_native_steps_from_decision": 11250, "device": "cuda", "threads": 1,
            "scoring_wall_seconds": 900, "cpu_rss_mib": 12288, "cuda_allocated_mib": 8192,
            "new_scoring_artifact_bytes": 2**28, "technical_retries": 0,
            "model_scoring_authorized": True, "new_optimizer_updates": 0, "fresh_access": False,
            "adaptive_controller_used": False, "advancement_authorized": False}
    replay.immutable(root / "scoring-execution-plan.json", plan)
    replay.immutable(replay.OUTPUT / "scoring-execution-plan.json", plan)
    print("Fixed scoring source frozen; no inference executed.")


def load_plan(root):
    plan = replay.run.files.read(root / "scoring-execution-plan.json")
    if not plan["model_scoring_authorized"] or plan["fresh_access"] or plan["new_optimizer_updates"]:
        raise ValueError("only frozen training-replay inference is authorized")
    for path, source in plan["source_text"].items():
        if (replay.run.files.ROOT / path).read_text() != source:
            raise ValueError(f"scoring source changed: {path}")
    for filename, key in (("scoring-input-inventory.json", "inputs"), ("representation-binding.json", "representation")):
        if replay.run.files.read(root / filename) != plan[key]:
            raise ValueError("frozen input/representation binding changed")
    return plan


def run(root=replay.run.ROOT):
    plan = load_plan(root)
    output = root / "fixed-action-scores"
    if (root / "budgets/replay-fixed-scoring.json").exists():
        raise ValueError("scoring already attempted; retain results and audit before any continuation")
    torch.set_num_threads(plan["threads"])
    torch.cuda.init()
    representation_plan = replay.run.files.read(repaired.ROOT / "plan.json")
    anchors = plan["inputs"]["anchors"]
    totals = {"representation_seconds": 0., "model_loading_seconds": 0., "prediction_seconds": 0.,
              "candidate_transitions": 0, "linear_macs": 0, "completed_models": 0}
    with fixed.fit.fitting_budget(root, "replay-fixed-scoring", plan["scoring_wall_seconds"], "cuda") as budget:
        def check():
            budget.check()
            size = sum(path.stat().st_size for path in output.glob("*.json"))
            if size > plan["new_scoring_artifact_bytes"]:
                budget.value["stopped"] = True
                budget.save()
                raise fixed.fit.FitBudgetExceeded("scoring artifact allowance exceeded")

        carriers = {}
        for seed in plan["inputs"]["seeds"]:
            check()
            started = time.monotonic()
            history, receipt = load_representation(representation_plan, seed, "cuda")
            expected, = [row for row in plan["representation"]["receipts"] if row["seed"] == seed]
            if receipt != expected:
                raise ValueError("representation completion receipt differs from freeze")
            for anchor in anchors:
                history.reset()
                image = agent_image_tensor(Path(anchor["decision_image_path"]).read_bytes())
                carriers[seed, anchor["source_member_identity"]] = history.observe(image, anchor["timestamp"])
            torch.cuda.synchronize()
            totals["representation_seconds"] += time.monotonic() - started
            del history
            check()
        for ordinal, cell in enumerate(plan["pool"]["models"], 1):
            started = time.monotonic()
            model, receipts = fixed.load_cell(plan["pool"], cell, "cuda")
            torch.cuda.synchronize()
            totals["model_loading_seconds"] += time.monotonic() - started
            rows = []
            for anchor in anchors:
                actions = [member["actions"][0] for member in anchor["candidate_members"]]
                for pair in model.pairs:
                    check()
                    started = time.monotonic()
                    score = score_actions(model, carriers[cell["seed"], anchor["source_member_identity"]],
                                          actions, pair, endpoint=plan["endpoint_native_steps_from_decision"])
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - started
                    macs = score["candidate_transitions"] * linear_macs(model, pair)
                    rows.append({"source_member_identity": anchor["source_member_identity"],
                                 "paired_ranking_admissible": anchor["paired_ranking_admissible"],
                                 "policy": fixed.fit.policy_name(pair), "wall_seconds": elapsed,
                                 "linear_macs": macs, **score})
                    totals["prediction_seconds"] += elapsed
                    totals["candidate_transitions"] += score["candidate_transitions"]
                    totals["linear_macs"] += macs
                    check()
            replay.immutable(output / f"model-{ordinal:03d}.json",
                             {"plan_identity": plan["identity"], "cell": cell, "checkpoint_receipts": receipts, "rows": rows})
            totals["completed_models"] = ordinal
            del model
            check()
            print(f"Replay scoring models={ordinal}/54 active={budget.value['active_seconds']:.1f}s", flush=True)
        check()
    report = {"plan_identity": plan["identity"], **totals,
              "budget": replay.run.files.read(root / "budgets/replay-fixed-scoring.json"),
              "full_flops_measured": False, "matched_deployment_compute_established": False,
              "fresh_access": False, "advancement_authorized": False}
    replay.immutable(root / "scoring-completion.json", report)
    replay.immutable(replay.OUTPUT / "scoring-completion.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.run:
        run()
    else:
        print("No-write: --prepare freezes source; --run performs the bounded fixed-action diagnostic.")
