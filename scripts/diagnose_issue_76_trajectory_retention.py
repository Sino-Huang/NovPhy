"""Minimal frozen-checkpoint replay of the remaining local and endpoint failures."""
import time
from pathlib import Path
from statistics import mean

import torch

from scripts import run_issue_76_endpoint_anchor as run
from world_model.model import Abstraction

ROOT = run.fit.files.ROOT / ".local-artifacts/issue-76-trajectory-retention-v1"


def main():
    torch.set_num_threads(1)
    started = time.monotonic()
    report = run.fit.files.read(run.OUTPUT / "report.json")
    plan = report["plan"]
    retained = []
    with run.fit.fitting_budget(ROOT, "replay", 180, "cpu") as budget:
        for seed, pure, elapsed, reference_stage in ((760930001, False, 750, "local_anchor"),
                (760930001, True, 750, "local_anchor"), (760930001, True, 11250, "boundary_focus"),
                (760930003, True, 11250, "local_anchor")):
            saved = torch.load(run.fit.model_path(run.ROOT, seed, pure, "predictor"),
                               map_location="cpu", weights_only=False)
            model = run.fit.new_predictor(plan, pure, "cpu").eval().requires_grad_(False)
            model.load_state_dict(saved["model"])
            pair = next(p for p in model.pairs if p.delta == 750 and p.abstraction == (
                Abstraction.CONTINUOUS if pure else Abstraction.MICRO))
            errors = []
            for entry in plan["evaluation"]:
                budget.check()
                shard = run.fit.load_shard(Path(plan["data_root"]), entry, plan["data_binding"], fitting=True)
                carrier = run.fit.load_carrier(Path(plan["data_root"]), seed, entry, plan["data_binding"])
                segment = shard["segment_ranges"][0]
                start, stop = segment["start"], segment["stop"]
                steps = shard["tensors"]["fixed_steps"].tolist()
                target = next(i for i in range(start, stop) if steps[i] - steps[start] == elapsed)
                prediction = carrier[start:start + 1]
                for _ in range(elapsed // pair.delta):
                    prediction = model.carrier(prediction, segment["action"][None], pair)
                errors.append(float((prediction - carrier[target:target + 1]).square().mean()))
            summary = next(s for s in report["summaries"] if s["seed"] == seed and s["pure"] == pure)
            actual = mean(errors)
            assert actual == summary["anchor_metrics"]["endpoint_anchor"][str(elapsed)], "minimal replay differs"
            reference = summary["anchor_metrics"][reference_stage][str(elapsed)]
            passed = actual <= 1.1 * reference
            retained.append(passed)
            print(dict(seed=seed, pure=pure, elapsed=elapsed, mse=actual, reference_mse=reference,
                       ratio=actual / reference, accuracy_retained=passed), flush=True)
    print(dict(seconds=time.monotonic() - started, optimization_performed=False, fresh_access=False), flush=True)
    return 0 if all(retained) else 1


if __name__ == "__main__":
    raise SystemExit(main())
