"""Read-only checkpoint replay of the full-duration local-retention failure."""
import argparse
from pathlib import Path

import torch

from scripts import run_issue_76_full_duration_dynamics as run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(760930001, 760930002, 760930003))
    parser.add_argument("--pure-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = run.fit.files.read(run.ROOT / "plan.json")
    failures = []
    for seed in plan["seeds"]:
        if args.seed is not None and seed != args.seed:
            continue
        for pure in ((True,) if args.pure_only else (False, True)):
            policy = "fixed-750-continuous" if pure else "fixed-750-micro"
            means = {}
            for name, root in (("boundary", run.source.ROOT), ("full_duration", run.ROOT)):
                saved = torch.load(run.fit.model_path(root, seed, pure, "predictor"),
                                   map_location="cpu", weights_only=False)
                model = run.fit.new_predictor(plan, pure, "cpu").eval().requires_grad_(False)
                model.load_state_dict(saved["model"])
                errors = []
                for entry in plan["evaluation"]:
                    shard = run.fit.load_shard(Path(plan["data_root"]), entry, plan["data_binding"], fitting=True)
                    carrier = run.fit.load_carrier(Path(plan["data_root"]), seed, entry, plan["data_binding"])
                    segment = shard["segment_ranges"][0]
                    start, stop = segment["start"], segment["stop"]
                    steps = shard["tensors"]["fixed_steps"].tolist()
                    end = next(i for i in range(start, stop) if steps[i] == steps[start] + 750)
                    pair = next(p for p in model.pairs if run.fit.policy_name(p) == policy)
                    predicted = model.carrier(carrier[start:start + 1], segment["action"][None], pair)
                    errors.append(float((predicted - carrier[end:end + 1]).square().mean()))
                means[name] = sum(errors) / len(errors)
            passed = means["full_duration"] <= plan["retention_ratio"] * means["boundary"]
            print(dict(seed=seed, pure=pure, means=means, retained=passed), flush=True)
            if not passed:
                failures.append((seed, pure))
    assert not failures, f"Frozen local-retention requirement failed: {failures}"


if __name__ == "__main__":
    main()
