"""One no-update forward/backward per arm/pair to cost full-duration fitting."""
import gc
from pathlib import Path
import time

import torch

from scripts import run_issue_76_boundary_dynamics as source
from scripts.issue_76_full_duration_loss import trajectory_batch, full_continuous_loss, full_hybrid_loss

ROOT = source.fit.files.ROOT / ".local-artifacts/issue-76-full-duration-resource-probe"
OUTPUT = source.fit.files.ROOT / "data/issue-76-full-duration-resource-probe/report.json"


def main():
    torch.set_num_threads(1)
    plan = source.fit.files.read(source.ROOT / "plan.json")
    seed, entry = plan["seeds"][0], plan["evaluation"][0]
    records = []
    with source.fit.fitting_budget(ROOT, "probe", 180, "cuda") as budget:
        shard = source.fit.load_shard(source.previous.parent.ROOT, entry, plan["data_binding"], fitting=True)
        carrier = source.fit.load_carrier(source.previous.parent.ROOT, seed, entry, plan["data_binding"])
        original = source.previous.sampled_schedule([len(carrier)], seed, False, 1)
        rows = source.boundary_schedule(original, [[s["start"] for s in shard["segment_ranges"]]], seed)[0]
        for pure in (False, True):
            checkpoint = torch.load(source.fit.model_path(source.ROOT, seed, pure, "predictor"),
                                    map_location="cpu", weights_only=False)
            if not checkpoint["complete"] or checkpoint["plan_identity"] != plan["identity"]:
                raise ValueError("resource probe requires the complete source-bound checkpoint")
            model = source.fit.new_predictor(plan, pure, "cuda")
            model.load_state_dict(checkpoint["model"])
            model.train()
            for pair in model.pairs:
                budget.check()
                model.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                began = time.monotonic()
                batch = trajectory_batch([shard], [carrier], rows, pair.delta, symbolic=not pure)
                batch = {k: v.cuda() for k, v in batch.items()}
                loss = (full_continuous_loss(model, batch["z"], batch["available"], batch["action"], pair)
                        if pure else full_hybrid_loss(model, batch, pair))
                loss.backward()
                torch.cuda.synchronize()
                elapsed = time.monotonic() - began
                finite = bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all())
                                                          for p in model.parameters() if p.grad is not None)
                record = dict(pure=pure, pair=source.fit.policy_name(pair), batch_size=len(rows),
                    recursive_steps=batch["z"].shape[1] - 1, forward_backward_seconds=elapsed,
                    peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                    finite_loss_and_gradients=finite)
                records.append(record)
                print(record, flush=True)
                if not finite:
                    raise ValueError("full-duration loss or gradients are nonfinite")
                budget.check()
                del loss, batch
            if not all(torch.equal(value.detach().cpu(), checkpoint["model"][key])
                       for key, value in model.state_dict().items()):
                raise ValueError("no-update resource probe changed checkpoint weights")
            del model, checkpoint
            gc.collect()
            torch.cuda.empty_cache()
    source.previous.parent.previous.immutable(OUTPUT, dict(source_plan_identity=plan["identity"],
        seed=seed, member_identity=entry["member_identity"], records=records,
        budget=source.fit.require_finished_budget(ROOT, "probe"), optimization_performed=False,
        fresh_access=False, weights_unchanged=True,
        source_text={"scripts/probe_issue_76_full_duration_resources.py": Path(__file__).read_text(),
                     "scripts/issue_76_full_duration_loss.py": (source.fit.files.ROOT / "scripts/issue_76_full_duration_loss.py").read_text()}))


if __name__ == "__main__":
    main()
