"""Load the identical completed parser/history source for both replay arms."""
import argparse
from pathlib import Path
import time

import torch

from scripts import run_issue_76_repaired_representation as repaired
from scripts import validate_issue_76_action_replay as replay
from world_model.planning.native_gameplay import NativeObservationHistory


def load_representation(plan, seed, device):
    fit = repaired.fit
    path = fit.common_path(repaired.ROOT, seed)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved["plan_identity"] != plan["identity"] or saved["seed"] != seed
            or saved["synthetic"] or saved["auxiliary_deployed"]
            or saved["parser_source_checkpoint"] != plan["parser_checkpoints"][str(seed)]):
        raise ValueError("common representation differs from the repaired seed/source")
    budget = fit.require_finished_budget(repaired.ROOT, f"common-{seed}")
    parser = repaired.load_parser(plan, seed, "cpu")
    if set(saved["parser"]) != set(parser.state_dict()) or any(
            not torch.equal(value, saved["parser"][key]) for key, value in parser.state_dict().items()):
        raise ValueError("common parser weights differ from the completed balanced parser")
    history_path = repaired.ROOT / "checkpoints" / f"history-{seed}.pt"
    history_fit = torch.load(history_path, map_location="cpu", weights_only=False)
    if (history_fit["plan_identity"] != plan["identity"] or not history_fit["complete"]
            or history_fit["failure"] is not None or history_fit["updates_requested"] != 500
            or history_fit["updates_completed"] != 500 or history_fit["updates_applied"] != 496
            or history_fit["updates_skipped"] != 4):
        raise ValueError("common history is not the completed repaired fit")
    history = fit.ObservedHistoryEncoder(carrier_dim=fit.VISUAL_DIM)
    history.load_state_dict(saved["history"], strict=True)
    for key, value in history.state_dict().items():
        if not torch.equal(value, history_fit["model"]["history." + key]) or not torch.isfinite(value).all():
            raise ValueError("common history weights differ from the completed history fit")
    representation = NativeObservationHistory(parser.to(device), history.to(device))
    receipt = {"seed": seed, "common_checkpoint": str(path), "plan_identity": plan["identity"],
               "parser_checkpoint": saved["parser_source_checkpoint"], "history_checkpoint": str(history_path),
               "common_completion_budget": budget, "source_weights_equal": True}
    return representation, receipt


def bind(root=replay.run.ROOT):
    started = time.monotonic()
    torch.set_num_threads(1)
    inputs = replay.run.files.read(root / "scoring-input-inventory.json")
    pool = replay.run.files.read(Path(inputs["model_pool_source"]))
    plan = replay.run.files.read(repaired.ROOT / "plan.json")
    if plan["identity"] != pool["data_plan_identity"]:
        raise ValueError("representation is not the retained model pool's shared data source")
    receipts = []
    for seed in inputs["seeds"]:
        representation, receipt = load_representation(plan, seed, "cpu")
        receipts.append(receipt)
        del representation
    elapsed = time.monotonic() - started
    value = {"identity": "issue-76-replay-representation-binding-v1", "receipts": receipts,
             "source_text": {"scripts/issue_76_replay_representation.py": Path(__file__).read_text()},
             "input_inventory_identity": inputs["identity"], "representation_plan": plan["identity"],
             "inference_performed": False, "model_scoring_authorized": False, "fresh_access": False,
             "wall_seconds": elapsed,
             "remaining_offline_preparation_seconds": inputs["remaining_offline_preparation_seconds"] - elapsed}
    replay.immutable(root / "representation-binding.json", value)
    replay.immutable(replay.OUTPUT / "representation-binding.json", value)
    print({"seed_bindings": len(receipts), "source_weights_equal": True, "wall_seconds": elapsed})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", action="store_true")
    if parser.parse_args().bind:
        bind()
    else:
        print("No-write: --bind checks completed representations without inference.")
