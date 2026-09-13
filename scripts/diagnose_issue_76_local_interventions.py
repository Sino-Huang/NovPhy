"""Paired one-factor local-retention probes; never write predictor checkpoints."""
import argparse
from pathlib import Path

import torch

from scripts import diagnose_issue_76_gradient_tradeoff as gradients
from scripts import diagnose_issue_76_finite_step as finite
from scripts import run_issue_76_lower_rate_dynamics as run

ROOT = run.fit.files.ROOT / ".local-artifacts/issue-76-local-interventions-v1"
OUTPUT = run.fit.files.ROOT / "data/issue-76-local-interventions/report.json"
VARIANTS = ("control", "local_weight_10", "boundary_only", "reset_optimizer_history")


def reset_history(optimizer):
    return {**optimizer, "state": {key: {name: torch.zeros_like(value)
            for name, value in state.items()} for key, state in optimizer["state"].items()}}


@torch.no_grad()
def endpoint(model, batch, pair):
    current = batch["z"][:, 0]
    for _ in range(1, batch["z"].shape[1]):
        current = model.carrier(current, batch["action"], pair)
    mask = batch["available"][:, -1]
    assert bool(mask.all()), "assigned first-shot endpoint missing"
    return float((current - batch["z"][:, -1]).square().mean())


@torch.no_grad()
def measure(model, mixed, pair, anchor_batch, anchor_pair, boundary_batch):
    local, recursive, symbolic, _ = gradients.terms(model, mixed, pair)
    return dict(local=float(local), recursive=float(recursive),
                total=float(local + recursive + (0 if symbolic is None else symbolic)),
                anchor=float(finite.anchor_loss(model, anchor_batch, anchor_pair)),
                anchor_endpoint=endpoint(model, anchor_batch, anchor_pair),
                policy_endpoint=endpoint(model, boundary_batch, pair))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    if not parser.parse_args().run:
        print("No-write: 36 paired cells x 4 one-factor variants; 600s CUDA cap; no checkpoint writes.")
        return
    torch.set_num_threads(1)
    torch.cuda.init()
    report = run.fit.files.read(run.OUTPUT / "report.json")
    plan = report["plan"]
    rows, assignments = [], []
    with run.fit.fitting_budget(ROOT, "interventions", 600, "cuda") as budget:
        for seed in plan["seeds"]:
            shards = [run.fit.load_shard(Path(plan["data_root"]), e, plan["data_binding"], fitting=True)
                      for e in plan["evaluation"]]
            carriers = [run.fit.load_carrier(Path(plan["data_root"]), seed, e, plan["data_binding"])
                        for e in plan["evaluation"]]
            anchors = torch.tensor([[i, s["segment_ranges"][0]["start"]] for i, s in enumerate(shards)])
            generator = torch.Generator().manual_seed(seed)
            uniform = torch.tensor([[i, int(torch.randint(len(c), (), generator=generator))]
                                    for i, c in enumerate(carriers)])
            indices = torch.cat((anchors, uniform))
            assignments.append(dict(seed=seed, boundary_rows=anchors.tolist(), uniform_rows=uniform.tolist()))
            for pure in (False, True):
                saved = torch.load(run.fit.model_path(run.ROOT, seed, pure, "predictor"),
                                   map_location="cuda", weights_only=False)
                assert saved["complete"] and saved["plan_identity"] == plan["identity"]
                model = run.fit.new_predictor(plan, pure, "cuda").eval()
                model.load_state_dict(saved["model"])
                parameters = tuple(model.parameters())
                anchor_pair = next(p for p in model.pairs if p.delta == 750 and p.abstraction == (
                    gradients.Abstraction.CONTINUOUS if pure else gradients.Abstraction.MICRO))
                anchor_batch = run.source.trajectory_batch(shards, carriers, anchors, 750, symbolic=not pure)
                anchor_batch = {k: v.cuda() for k, v in anchor_batch.items()}
                for pair in model.pairs:
                    mixed = run.source.trajectory_batch(shards, carriers, indices, pair.delta, symbolic=not pure)
                    mixed = {k: v.cuda() for k, v in mixed.items()}
                    boundary = {k: v[:len(anchors)] for k, v in mixed.items()}
                    base = measure(model, mixed, pair, anchor_batch, anchor_pair, boundary)
                    row = dict(seed=seed, pure=pure, policy=run.fit.policy_name(pair), base=base, variants={})
                    for variant in VARIANTS:
                        budget.check()
                        batch = boundary if variant == "boundary_only" else mixed
                        local, recursive, symbolic, _ = gradients.terms(model, batch, pair)
                        loss = (10 if variant == "local_weight_10" else 1) * local + recursive
                        loss = loss + (0 if symbolic is None else symbolic)
                        grad = torch.autograd.grad(loss, parameters, allow_unused=True)
                        optimizer = reset_history(saved["optimizer"]) if variant == "reset_optimizer_history" else saved["optimizer"]
                        direction = gradients.adamw_delta(parameters, grad, optimizer)
                        cursor = 0
                        with torch.no_grad():
                            for name, parameter in model.named_parameters():
                                size = parameter.numel()
                                parameter.copy_(saved["model"][name] + direction[cursor:cursor + size].reshape_as(parameter))
                                cursor += size
                        row["variants"][variant] = measure(model, mixed, pair, anchor_batch, anchor_pair, boundary)
                        model.load_state_dict(saved["model"])
                        assert all(torch.equal(value, saved["model"][key]) for key, value in model.state_dict().items())
                    rows.append(row)
                    run.fit.log(f"local interventions {len(rows)}/36")
                del model, saved, parameters, grad, direction, optimizer
        budget.check()
    run.immutable(OUTPUT, dict(source_plan_identity=plan["identity"], evaluation=plan["evaluation"],
        assignments=assignments, rows=rows, variants=list(VARIANTS), checkpoint_writes=False,
        temporary_parameters_restored=True, fresh_access=False, advancement_passed=False,
        budget=run.fit.require_finished_budget(ROOT, "interventions"),
        prior_replay_budget=run.fit.files.read(ROOT.parent / "issue-76-local-retention-diagnosis-v1/budgets/replay.json"),
        source_text={p: (run.fit.files.ROOT / p).read_text() for p in (
            "scripts/diagnose_issue_76_local_interventions.py", "docs/issue-76-local-interventions-protocol.md",
            "scripts/diagnose_issue_76_gradient_tradeoff.py", "scripts/diagnose_issue_76_finite_step.py")},
        interpretation="Temporary single-batch interventions, not trained candidates or advancement evidence"))


if __name__ == "__main__":
    main()
