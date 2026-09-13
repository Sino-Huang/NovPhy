"""Frozen-weight local/recursive and cross-horizon gradient diagnosis."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts import run_issue_76_full_duration_dynamics as run
from world_model.model import Abstraction
from world_model.training.native_history_fit import symbolic_loss

ROOT = run.fit.files.ROOT / ".local-artifacts/issue-76-gradient-tradeoff-v1"
OUTPUT = run.fit.files.ROOT / "data/issue-76-gradient-tradeoff/report.json"


def flatten(values, parameters):
    return torch.cat([(torch.zeros_like(p) if v is None else v).flatten()
                      for v, p in zip(values, parameters)])


def cosine(a, b):
    denominator = a.norm() * b.norm()
    return float(a.dot(b) / denominator) if denominator > 0 else None


def adamw_delta(parameters, gradients, state):
    """Compute, but do not apply, the next clipped AdamW parameter change."""
    norm = flatten(gradients, parameters).norm()
    clip = (1 / (norm + 1e-6)).clamp(max=1)
    result, cursor = [], 0
    for group in state["param_groups"]:
        beta1, beta2 = group["betas"]
        for index in group["params"]:
            p, gradient = parameters[cursor], gradients[cursor]
            cursor += 1
            if gradient is None:
                result.append(torch.zeros_like(p))
                continue
            old = state["state"][index]
            step = float(old["step"]) + 1
            g = gradient * clip
            m = old["exp_avg"] * beta1 + g * (1 - beta1)
            v = old["exp_avg_sq"] * beta2 + g.square() * (1 - beta2)
            update = (m / (1 - beta1 ** step)) / ((v / (1 - beta2 ** step)).sqrt() + group["eps"])
            result.append(-group["lr"] * (update + group["weight_decay"] * p.detach()))
    return flatten(result, parameters)


def terms(model, batch, pair):
    z, available, action = batch["z"], batch["available"], batch["action"]
    local, recursive = [], []
    current = z[:, 0]
    counts = []
    for offset in range(1, z.shape[1]):
        mask = available[:, offset - 1] & available[:, offset]
        if not bool(mask.any()):
            break
        current = model.carrier(current, action, pair)
        counts.append(int(mask.sum()))
        if offset <= 4:
            observed = current if offset == 1 else model.carrier(z[:, offset - 1], action, pair)
            local.append(F.mse_loss(observed[mask], z[mask, offset]))
        recursive.append(F.mse_loss(current[mask], z[mask, offset])
                         + .01 * F.relu(current[mask].abs() - 2).square().mean())
    symbolic = []
    if pair.abstraction != Abstraction.CONTINUOUS:
        name, masks = (("relations", "relation_mask") if pair.abstraction == Abstraction.MICRO
                       else ("macros", "macro_mask"))
        for offset in range(5):
            valid = available[:, offset]
            if bool(valid.any()):
                symbolic.append(symbolic_loss(model, z[valid, offset], batch[name][valid, offset],
                                              batch[masks][valid, offset], pair.abstraction))
    return torch.stack(local).mean(), torch.stack(recursive).mean(), (
        torch.stack(symbolic).mean() if symbolic else None), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    plan = run.fit.files.read(run.ROOT / "plan.json")
    if not args.run:
        print("No-write: 72 pair/checkpoint probes; 5 assigned training anchors; no parameter updates; CUDA cap 600s.")
        return
    torch.set_num_threads(1)
    torch.cuda.init()
    rows = []
    with run.fit.fitting_budget(ROOT, "gradient-probe", 600, "cuda") as budget:
        for seed in plan["seeds"]:
            shards = [run.fit.load_shard(Path(plan["data_root"]), e, plan["data_binding"], fitting=True)
                      for e in plan["evaluation"]]
            carriers = [run.fit.load_carrier(Path(plan["data_root"]), seed, e, plan["data_binding"])
                        for e in plan["evaluation"]]
            anchors = torch.tensor([[i, s["segment_ranges"][0]["start"]] for i, s in enumerate(shards)])
            for pure in (False, True):
                for stage, root in (("boundary", run.source.ROOT), ("full_duration", run.ROOT)):
                    budget.check()
                    saved = torch.load(run.fit.model_path(root, seed, pure, "predictor"),
                                       map_location="cuda", weights_only=False)
                    model = run.fit.new_predictor(plan, pure, "cuda").eval()
                    model.load_state_dict(saved["model"])
                    parameters = tuple(model.parameters())
                    anchor_pair = next(p for p in model.pairs if p.delta == 750 and
                                       p.abstraction == (Abstraction.CONTINUOUS if pure else Abstraction.MICRO))
                    anchor_batch = run.trajectory_batch(shards, carriers, anchors, 750, symbolic=not pure)
                    anchor_batch = {k: v.cuda() for k, v in anchor_batch.items()}
                    mask = anchor_batch["available"][:, 0] & anchor_batch["available"][:, 1]
                    prediction = model.carrier(anchor_batch["z"][:, 0], anchor_batch["action"], anchor_pair)
                    anchor_loss = F.mse_loss(prediction[mask], anchor_batch["z"][mask, 1])
                    anchor_gradient = flatten(torch.autograd.grad(anchor_loss, parameters, allow_unused=True), parameters)
                    for pair in model.pairs:
                        budget.check()
                        batch = run.trajectory_batch(shards, carriers, anchors, pair.delta, symbolic=not pure)
                        batch = {k: v.cuda() for k, v in batch.items()}
                        local, recursive, symbolic, counts = terms(model, batch, pair)
                        total = local + recursive + (0 if symbolic is None else symbolic)
                        with torch.no_grad():
                            official = (run.full_continuous_loss(model, batch["z"], batch["available"], batch["action"], pair)
                                        if pure else run.full_hybrid_loss(model, batch, pair))
                            torch.testing.assert_close(total, official)
                        gl = flatten(torch.autograd.grad(local, parameters, retain_graph=True, allow_unused=True), parameters)
                        gr = flatten(torch.autograd.grad(recursive, parameters, retain_graph=True, allow_unused=True), parameters)
                        gradients = torch.autograd.grad(total, parameters, allow_unused=True)
                        gt = flatten(gradients, parameters)
                        delta = adamw_delta(parameters, gradients, saved["optimizer"])
                        rows.append(dict(seed=seed, pure=pure, stage=stage, policy=run.fit.policy_name(pair),
                            anchor_loss=float(anchor_loss.detach()), local_loss=float(local.detach()),
                            recursive_loss=float(recursive.detach()), symbolic_loss=None if symbolic is None else float(symbolic.detach()),
                            adjacent_available_counts=counts, local_gradient_norm=float(gl.norm()), recursive_gradient_norm=float(gr.norm()),
                            local_recursive_cosine=cosine(gl, gr), anchor_total_gradient_cosine=cosine(anchor_gradient, gt),
                            anchor_adamw_delta_cosine=cosine(anchor_gradient, delta)))
                        run.fit.log(f"gradient tradeoff {len(rows)}/72")
                    assert all(torch.equal(v, saved["model"][k]) for k, v in model.state_dict().items())
                    del model, saved, parameters, anchor_gradient, gradients, gl, gr, gt, delta
        budget.check()
    value = dict(source_plan_identity=plan["identity"], evaluation=plan["evaluation"], rows=rows,
                 weights_unchanged=True, optimization_performed=False, fresh_access=False,
                 budget=run.fit.require_finished_budget(ROOT, "gradient-probe"),
                 source_text=Path(__file__).read_text(),
                 interpretation="Training-anchor directional derivatives, not a causal fit intervention or advancement test")
    if len(json.dumps(value, indent=2).encode()) > 10 * 2**20:
        raise ValueError("gradient publication exceeds 10 MiB")
    run.source.previous.parent.previous.immutable(OUTPUT, value)


if __name__ == "__main__":
    main()
