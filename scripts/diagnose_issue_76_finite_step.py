"""Measure one-step overshoot on temporary model copies, never saved checkpoints."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts import diagnose_issue_76_gradient_tradeoff as previous

run = previous.run
ROOT = run.fit.files.ROOT / ".local-artifacts/issue-76-finite-step-v1"
OUTPUT = run.fit.files.ROOT / "data/issue-76-finite-step/report.json"


def anchor_loss(model, batch, pair):
    mask = batch["available"][:, 0] & batch["available"][:, 1]
    predicted = model.carrier(batch["z"][:, 0], batch["action"], pair)
    return F.mse_loss(predicted[mask], batch["z"][mask, 1])


@torch.no_grad()
def measures(model, batch, pair, anchor_batch, anchor_pair):
    local, recursive, symbolic, _ = previous.terms(model, batch, pair)
    return dict(local=float(local), recursive=float(recursive),
                total=float(local + recursive + (0 if symbolic is None else symbolic)),
                anchor=float(anchor_loss(model, anchor_batch, anchor_pair)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print("No-write: all 36 completed pair/seed cells, 0.1x and 1x saved AdamW delta; no checkpoint writes; 300s CUDA cap.")
        return
    torch.set_num_threads(1)
    torch.cuda.init()
    plan = run.fit.files.read(run.ROOT / "plan.json")
    rows = []
    with run.fit.fitting_budget(ROOT, "finite-step-probe", 300, "cuda") as budget:
        for seed in plan["seeds"]:
            shards = [run.fit.load_shard(Path(plan["data_root"]), e, plan["data_binding"], fitting=True)
                      for e in plan["evaluation"]]
            carriers = [run.fit.load_carrier(Path(plan["data_root"]), seed, e, plan["data_binding"])
                        for e in plan["evaluation"]]
            anchors = torch.tensor([[i, s["segment_ranges"][0]["start"]] for i, s in enumerate(shards)])
            for pure in (False, True):
                saved = torch.load(run.fit.model_path(run.ROOT, seed, pure, "predictor"),
                                   map_location="cuda", weights_only=False)
                model = run.fit.new_predictor(plan, pure, "cuda").eval()
                model.load_state_dict(saved["model"])
                parameters = tuple(model.parameters())
                anchor_pair = next(p for p in model.pairs if p.delta == 750 and
                                   p.abstraction == (previous.Abstraction.CONTINUOUS if pure else previous.Abstraction.MICRO))
                anchor_batch = run.trajectory_batch(shards, carriers, anchors, 750, symbolic=not pure)
                anchor_batch = {k: v.cuda() for k, v in anchor_batch.items()}
                for pair in model.pairs:
                    budget.check()
                    batch = run.trajectory_batch(shards, carriers, anchors, pair.delta, symbolic=not pure)
                    batch = {k: v.cuda() for k, v in batch.items()}
                    local, recursive, symbolic, _ = previous.terms(model, batch, pair)
                    total = local + recursive + (0 if symbolic is None else symbolic)
                    gradients = torch.autograd.grad(total, parameters, allow_unused=True)
                    direction = previous.adamw_delta(parameters, gradients, saved["optimizer"])
                    anchor_grad = previous.flatten(torch.autograd.grad(anchor_loss(model, anchor_batch, anchor_pair),
                                                                     parameters, allow_unused=True), parameters)
                    row = dict(seed=seed, pure=pure, policy=run.fit.policy_name(pair),
                               base=measures(model, batch, pair, anchor_batch, anchor_pair),
                               total_directional_derivative=float(previous.flatten(gradients, parameters).dot(direction)),
                               anchor_directional_derivative=float(anchor_grad.dot(direction)), perturbations={})
                    for scale in (.1, 1.):
                        cursor = 0
                        with torch.no_grad():
                            for name, p in model.named_parameters():
                                size = p.numel()
                                p.copy_(saved["model"][name] + scale * direction[cursor:cursor + size].reshape_as(p))
                                cursor += size
                        row["perturbations"][str(scale)] = measures(model, batch, pair, anchor_batch, anchor_pair)
                    model.load_state_dict(saved["model"])
                    assert all(torch.equal(v, saved["model"][k]) for k, v in model.state_dict().items())
                    rows.append(row)
                    run.fit.log(f"finite step {len(rows)}/36")
                del model, saved, parameters, gradients, direction, anchor_grad
        budget.check()
    value = dict(source_plan_identity=plan["identity"], evaluation=plan["evaluation"], rows=rows,
                 checkpoint_writes=False, temporary_parameters_restored=True, optimizer_steps_applied=0,
                 fresh_access=False, budget=run.fit.require_finished_budget(ROOT, "finite-step-probe"),
                 source_text=Path(__file__).read_text(),
                 interpretation="Finite parameter perturbations on five training anchors, not fitted candidates or advancement evidence")
    if len(json.dumps(value, indent=2).encode()) > 10 * 2**20:
        raise ValueError("finite-step publication exceeds 10 MiB")
    run.source.previous.parent.previous.immutable(OUTPUT, value)


if __name__ == "__main__":
    main()
