"""Read-only engineering replay of the retained balanced-native gradient stop."""

import argparse
from pathlib import Path

import torch

from scripts import run_issue_76_native_refit as fit
from scripts.issue_76_full_duration_loss import full_hybrid_loss, trajectory_batch
from scripts.issue_76_matched_batches import mixed_schedule, sampled_schedule
from scripts.issue_76_scaled_fit import backward_clipped


def replay(root, inspect):
    plan = fit.files.read(root / "plan.json")
    parent = fit.load_plan()
    seed, pure = 760930003, False
    saved = torch.load(fit.model_path(root, seed, pure, "predictor"),
                       map_location="cuda", weights_only=False)
    update = saved["updates_completed"]
    if update != 9114 or not all(bool(torch.isfinite(v).all()) for v in saved["model"].values()):
        raise ValueError("probe requires the finite pre-update-9114 failure state")
    training = [e for e in plan["entries"] if e["exposure_role"] == "training"]
    lengths = [e["frames"] if e["usable"] else 0 for e in training]
    schedule = mixed_schedule(sampled_schedule(lengths, seed, pure,
        plan["updates"]["predictor"], plan["batch_size"]), seed, pure)
    rows = schedule[update]
    shards, carriers = [None] * len(training), [None] * len(training)
    for index in sorted(set(rows[:, 0].tolist()) | {0}):
        entry = training[index]
        shards[index] = fit.load_shard(fit.ROOT, entry, parent, fitting=True)
        value = torch.load(fit.carrier_path(root, seed, entry),
                           map_location="cpu", weights_only=False)
        if value["plan_identity"] != plan["identity"]:
            raise ValueError("probe carrier plan differs")
        carriers[index] = value["carrier"]
    pair = fit.pair_for_update(pure, update)
    batch = trajectory_batch(shards, carriers, rows, pair.delta, symbolic=True)
    batch = {k: v.cuda() for k, v in batch.items()}
    for exponent in ((0, 16, 32, 48) if inspect else (16,)):
        model = fit.new_predictor(plan, pure, "cuda")
        model.load_state_dict(saved["model"])
        model.train()
        loss = full_hybrid_loss(model, batch, pair)
        scale = 2.0 ** -exponent
        if inspect:
            (loss * scale).backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            finite = all(bool(torch.isfinite(g).all()) for g in gradients)
            norm32 = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(g) for g in gradients]))
            norm64 = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(g, dtype=torch.float64) for g in gradients]))
            print({"update": update, "scale_exponent": exponent,
                   "loss": float(loss.detach()), "gradients_finite": finite,
                   "norm32": float(norm32), "norm64": float(norm64)}, flush=True)
        else:
            backward_clipped(loss, model.parameters(), backward_scale=scale)
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            norm = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(g, dtype=torch.float64) for g in gradients]))
            if not all(bool(torch.isfinite(g).all()) for g in gradients) or float(norm) > 1.000001:
                raise ValueError("replayed clipped gradient is nonfinite or exceeds unit norm")
            print({"update": update, "loss": float(loss.detach()),
                   "clipped_gradient_norm": float(norm), "replay_passed": True}, flush=True)
        del model, loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
        default=fit.files.ROOT / ".local-artifacts/issue-76-balanced-native-v2")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    replay(args.root, args.inspect)


if __name__ == "__main__":
    main()
