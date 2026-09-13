"""Replay the two residual pure-model local-accuracy failures without fitting."""
import time
from pathlib import Path
from statistics import mean

import torch

from scripts import run_issue_76_lower_rate_dynamics as run

ROOT = Path(".local-artifacts/issue-76-local-retention-diagnosis-v1")


def main():
    torch.set_num_threads(1)
    started = time.monotonic()
    report = run.fit.files.read(run.OUTPUT / "report.json")
    plan = report["plan"]
    retained = []
    with run.fit.fitting_budget(ROOT, "replay", 180, "cpu") as budget:
        for seed in (760930001, 760930002):
            saved = torch.load(run.fit.model_path(run.ROOT, seed, True, "predictor"),
                               map_location="cpu", weights_only=False)
            model = run.fit.new_predictor(plan, True, "cpu").eval().requires_grad_(False)
            model.load_state_dict(saved["model"])
            pair = next(p for p in model.pairs if p.delta == 750)
            errors = []
            for entry in plan["evaluation"]:
                budget.check()
                shard = run.fit.load_shard(Path(plan["data_root"]), entry, plan["data_binding"], fitting=True)
                carrier = run.fit.load_carrier(Path(plan["data_root"]), seed, entry, plan["data_binding"])
                segment = shard["segment_ranges"][0]
                start, stop = segment["start"], segment["stop"]
                steps = shard["tensors"]["fixed_steps"].tolist()
                target = next(i for i in range(start, stop) if steps[i] - steps[start] == 750)
                prediction = model.carrier(carrier[start:start + 1], segment["action"][None], pair)
                errors.append(float((prediction - carrier[target:target + 1]).square().mean()))
            summary = next(s for s in report["summaries"] if s["seed"] == seed and s["pure"])
            actual = mean(errors)
            assert actual == summary["anchor_metrics"]["lower_rate"]["750"], "minimal replay differs"
            reference = summary["anchor_metrics"]["boundary"]["750"]
            passed = actual <= 1.1 * reference
            retained.append(passed)
            print(dict(seed=seed, local_mse=actual, reference_mse=reference,
                       ratio=actual / reference, initial_accuracy_retained=passed), flush=True)
    print(dict(seconds=time.monotonic() - started, optimization_performed=False, fresh_access=False), flush=True)
    return 0 if all(retained) else 1


if __name__ == "__main__":
    raise SystemExit(main())
