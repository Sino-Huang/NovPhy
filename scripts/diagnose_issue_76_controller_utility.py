"""Read-only model probes on the first assigned training lineage per family."""
from collections import Counter
from pathlib import Path
import time

import torch

from scripts import run_issue_76_native_refit as fit
from scripts.run_issue_76_repaired_representation import previous
from world_model.training.cnn_hybrid import linear_macs
from world_model.training.native_history_fit import (
    ENDPOINT, REFERENCE_TRANSITION_MACS, NativeHistoryController,
    controller_targets, rollout,
)

ROOT = fit.files.ROOT / ".local-artifacts/issue-76-repaired-representation-v1"
OUTPUT = fit.files.ROOT / "data/issue-76-controller-utility-diagnostic/report.json"


@torch.no_grad()
def local_costs(model, carrier, fixed_steps, action):
    keep = [i for i, s in enumerate(fixed_steps)
            if (s - fixed_steps[0]) % 50 == 0 and s - fixed_steps[0] <= ENDPOINT]
    z = carrier[keep]
    steps = [fixed_steps[i] for i in keep]
    lookup = {s: i for i, s in enumerate(steps)}
    raw = z.new_full((len(z), len(model.pairs)), torch.inf)
    for column, pair in enumerate(model.pairs):
        rows = [i for i, s in enumerate(steps) if s + pair.delta in lookup]
        if rows:
            ends = [lookup[steps[i] + pair.delta] for i in rows]
            predicted = model.carrier(z[rows], action[None].expand(len(rows), -1), pair)
            raw[rows, column] = (predicted - z[ends]).square().mean(-1)
    return z, steps, raw


def solve(costs, pairs, steps):
    lookup = {s: i for i, s in enumerate(steps)}
    values = costs.new_zeros(len(steps))
    q = costs.clone()
    for i in range(len(steps) - 2, -1, -1):
        for column, pair in enumerate(pairs):
            end = lookup.get(steps[i] + pair.delta)
            if end is not None:
                q[i, column] += values[end]
        values[i] = q[i].min()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("teacher has no exact finite endpoint path")
    return q[:-1].argmin(-1), values[:-1], q[:-1]


@torch.no_grad()
def probe(model, controller, shard, carrier):
    names = [fit.policy_name(p) for p in model.pairs]
    penalties = carrier.new_tensor([.01 * linear_macs(model, p) / REFERENCE_TRANSITION_MACS
                                    for p in model.pairs])
    scales = carrier.new_tensor([p.delta * .0004 for p in model.pairs])
    counts = {"original": Counter(), "zero_compute_penalty": Counter()}
    predictions = Counter()
    raw_sum, raw_count = torch.zeros(len(names)), torch.zeros(len(names))
    regrets, gaps = [], []
    correct = total = 0
    for segment in shard["segment_ranges"]:
        start, stop = segment["start"], segment["stop"]
        fixed_steps = shard["tensors"]["fixed_steps"][start:stop].tolist()
        original = controller_targets(model, carrier[start:stop], fixed_steps, segment["action"])
        if original is None:
            continue
        z, steps, raw = local_costs(model, carrier[start:stop], fixed_steps, segment["action"])
        finite = torch.isfinite(raw)
        raw_sum += torch.where(finite, raw, 0).sum(0)
        raw_count += finite.sum(0)
        labels, values, q = solve(raw * scales + penalties, model.pairs, steps)
        if not torch.equal(labels, original["labels"]):
            raise ValueError("diagnostic does not reproduce the unchanged teacher")
        zero, _, _ = solve(raw * scales, model.pairs, steps)
        counts["original"].update(names[i] for i in labels.tolist())
        counts["zero_compute_penalty"].update(names[i] for i in zero.tolist())
        predicted = controller(original["z"], original["action"], original["remaining"]).argmax(-1)
        predictions.update(names[i] for i in predicted.tolist())
        regrets.extend((q.gather(1, predicted[:, None])[:, 0] - values).tolist())
        sorted_q = q.sort(-1).values
        gap = sorted_q[:, 1] - sorted_q[:, 0]
        gaps.extend(gap[torch.isfinite(gap)].tolist())
        correct += int((labels == predicted).sum())
        total += len(labels)
    r, g = torch.tensor(regrets), torch.tensor(gaps)
    local = dict(teacher_rows=total, teacher_reproduced_exactly=True,
                 teacher_counts={k: dict(v) for k, v in counts.items()},
                 controller_predictions=dict(predictions), controller_label_accuracy=correct / total,
                 mean_local_mse=dict(zip(names, (raw_sum / raw_count).tolist())),
                 original_per_transition_penalty=dict(zip(names, penalties.tolist())),
                 mean_teacher_regret=float(r.mean()), max_teacher_regret=float(r.max()),
                 regret_above_1e_4=int((r > 1e-4).sum()), regret_above_1e_3=int((r > 1e-3).sum()),
                 median_best_second_gap=float(g.median()), gaps_below_1e_4=int((g < 1e-4).sum()))
    segment = shard["segment_ranges"][0]
    start, stop = segment["start"], segment["stop"]
    steps = shard["tensors"]["fixed_steps"].tolist()
    ends = [i for i in range(start, stop) if steps[i] == steps[start] + ENDPOINT]
    recursive = {"available": len(ends) == 1, "policies": {}}
    if len(ends) == 1:
        target = carrier[ends[0]]
        recursive["unchanged_carrier_mse"] = float((carrier[start] - target).square().mean())
        for name, pair in zip(names, model.pairs):
            predicted, work = rollout(model, None, carrier[start], segment["action"], fixed_pair=pair)
            mse = float((predicted - target).square().mean())
            cost = .01 * sum(w["linear_macs"] for w in work) / REFERENCE_TRANSITION_MACS
            recursive["policies"][name] = dict(carrier_mse=mse, compute_penalty=cost,
                                               endpoint_error_plus_cost=ENDPOINT * .0004 * mse + cost)
        recursive["best_fixed_for_endpoint_error_plus_cost"] = min(
            names, key=lambda name: recursive["policies"][name]["endpoint_error_plus_cost"])
    return dict(local=local, recursive=recursive)


def main():
    torch.set_num_threads(1)
    plan = fit.files.read(ROOT / "plan.json")
    selected = {}
    for entry in fit.data_index(ROOT, plan)["entries"]:
        if entry["exposure_role"] == "training":
            selected.setdefault(entry["family"], entry)
    began, rows = time.monotonic(), []
    for seed in plan["seeds"]:
        for pure in (False, True):
            model = fit.load_predictor(ROOT, plan, seed, pure, "cpu")
            controller = NativeHistoryController(pure).eval()
            controller.load_state_dict(torch.load(fit.model_path(ROOT, seed, pure, "controller"),
                                                  map_location="cpu", weights_only=False)["model"])
            for entry in selected.values():
                row = dict(seed=seed, pure=pure, member_identity=entry["member_identity"],
                           family=entry["family"], usable=entry["usable"])
                if entry["usable"]:
                    row.update(probe(model, controller, fit.load_shard(ROOT, entry, plan, fitting=True),
                                     fit.load_carrier(ROOT, seed, entry, plan)))
                rows.append(row)
                if time.monotonic() - began > 180:
                    raise RuntimeError("180-second diagnostic ceiling exceeded; no fitting or retry")
                print(f"utility diagnostic {len(rows)}/30", flush=True)
    report = dict(schema="issue_76_controller_utility_diagnostic_v1", source_plan_identity=plan["identity"],
                  selected_entries=list(selected.values()), rows=rows, fresh_access=False,
                  optimization_performed=False, active_seconds=time.monotonic() - began,
                  source_text=Path(__file__).read_text(),
                  interpretation="Exploratory existing-training-data diagnosis, not a deployable policy or advancement test")
    previous.immutable(OUTPUT, report)
    print(f"published {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
