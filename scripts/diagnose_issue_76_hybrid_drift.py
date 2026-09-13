"""Frozen-weight gradient and recursive-drift probes; no optimizer or fresh data."""
from itertools import combinations
from pathlib import Path
import time

import torch

from scripts import run_issue_76_matched_dynamics as run
from world_model.model import Abstraction
from world_model.training.native_history_fit import symbolic_loss

OUTPUT = run.fit.files.ROOT / "data/issue-76-hybrid-drift-diagnostic/report.json"


def gradient_vector(loss, parameters, *, retain_graph=False):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph)
    return torch.cat([g.detach().flatten() for g in gradients])


def alignment(left, right):
    a, b = float(left.norm()), float(right.norm())
    return dict(left_norm=a, right_norm=b,
                cosine=float(torch.dot(left, right)) / (a * b) if a and b else None)


def gradient_probe(model, shard, carrier, seed):
    shared = [p for n, p in model.named_parameters()
              if n.startswith(("input_projection.", "blocks.", "output_norm.", "output_projection."))]
    rows = []
    for horizon in (50, 250, 750):
        batch = run.fit.transition_batch(shard, carrier, horizon, run.fit.generator(seed, 0), symbolic=True)
        shared_gradients, heads, losses = {}, {}, {}
        for pair in (p for p in model.pairs if p.delta == horizon):
            mode = pair.abstraction.value
            dynamic = run.fit.continuous_loss(model, batch["z"], batch["available"], batch["action"], pair)
            if dynamic is None:
                continue
            shared_gradients[mode] = gradient_vector(dynamic, shared, retain_graph=True)
            losses[mode] = float(dynamic.detach())
            if pair.abstraction == Abstraction.CONTINUOUS:
                continue
            head = list((model.micro_head if pair.abstraction == Abstraction.MICRO else model.macro_head).parameters())
            dynamic_gradient = gradient_vector(dynamic, head)
            name, mask_name = ("relations", "relation_mask") if pair.abstraction == Abstraction.MICRO else ("macros", "macro_mask")
            terms = []
            for offset in range(5):
                valid = batch["available"][:, offset]
                if bool(valid.any()):
                    terms.append(symbolic_loss(model, batch["z"][valid, offset], batch[name][valid, offset],
                                               batch[mask_name][valid, offset], pair.abstraction))
            auxiliary = torch.stack(terms).mean()
            heads[mode] = dict(dynamics_vs_symbolic=alignment(dynamic_gradient, gradient_vector(auxiliary, head)),
                               symbolic_loss=float(auxiliary.detach()))
        rows.append(dict(horizon=horizon, dynamic_losses=losses, head_gradients=heads,
                         shared_mode_gradients={f"{a}:{b}": alignment(shared_gradients[a], shared_gradients[b])
                                                for a, b in combinations(shared_gradients, 2)}))
    return rows


@torch.no_grad()
def drift_curves(model, shard, carrier):
    segment = shard["segment_ranges"][0]
    start, stop = segment["start"], segment["stop"]
    steps = shard["tensors"]["fixed_steps"].tolist()
    lookup = {steps[i] - steps[start]: i for i in range(start, stop)}
    action = segment["action"][None]
    result = {}
    for pair in model.pairs:
        current = carrier[start:start + 1]
        selected = {pair.delta * i for i in (1, 2, 4, 8)} | {750, 1500, 3000, 6000, run.fit.ENDPOINT}
        rows = []
        for elapsed in range(pair.delta, run.fit.ENDPOINT + 1, pair.delta):
            current = model.carrier(current, action, pair)
            if elapsed not in selected:
                continue
            if elapsed not in lookup or elapsed - pair.delta not in lookup:
                rows.append(dict(elapsed=elapsed, available=False))
                continue
            target = carrier[lookup[elapsed]][None]
            local = model.carrier(carrier[lookup[elapsed - pair.delta]][None], action, pair)
            rows.append(dict(elapsed=elapsed, recursive_steps=elapsed // pair.delta, available=True,
                local_mse=float((local - target).square().mean()),
                recursive_mse=float((current - target).square().mean()),
                unchanged_mse=float((carrier[start:start + 1] - target).square().mean())))
        result[run.fit.policy_name(pair)] = rows
    return result


def main():
    torch.set_num_threads(1)
    began = time.monotonic()
    plan = run.fit.files.read(run.ROOT / "plan.json")
    rows = []
    for seed in plan["seeds"]:
        for pure in (False, True):
            run.fit.require_finished_budget(run.ROOT, f"predictor-{'pure' if pure else 'hybrid'}-{seed}")
            checkpoint = torch.load(run.fit.model_path(run.ROOT, seed, pure, "predictor"),
                                    map_location="cpu", weights_only=False)
            if not checkpoint["complete"] or checkpoint["plan_identity"] != plan["identity"]:
                raise ValueError("diagnosis requires the frozen complete matched predictor")
            model = run.fit.new_predictor(plan, pure, "cpu").eval()
            model.load_state_dict(checkpoint["model"])
            for entry in plan["evaluation"]:
                shard = run.fit.load_shard(run.parent.ROOT, entry, plan["parent_plan"], fitting=True)
                carrier = run.fit.load_carrier(run.parent.ROOT, seed, entry, plan["parent_plan"])
                rows.append(dict(seed=seed, pure=pure, member_identity=entry["member_identity"],
                    family=entry["family"], curves=drift_curves(model, shard, carrier),
                    gradients=None if pure else gradient_probe(model, shard, carrier, seed)))
                if time.monotonic() - began > 180:
                    raise RuntimeError("diagnostic exceeded 180-second CPU ceiling")
                print(f"hybrid drift diagnostic {len(rows)}/30", flush=True)
    run.parent.previous.immutable(OUTPUT, dict(source_plan_identity=plan["identity"],
        selected_entries=plan["evaluation"], rows=rows, source_text=Path(__file__).read_text(),
        optimization_performed=False, fresh_access=False, active_seconds=time.monotonic() - began,
        interpretation="Training-only frozen-weight diagnosis; gradient alignment is not a causal intervention"))


if __name__ == "__main__":
    main()
