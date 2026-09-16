"""Refit paired native dynamics around the frozen balanced perception surface."""

import argparse
from collections import Counter
import gc
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts import run_issue_76_balanced_pig as balanced
from scripts import run_issue_76_native_refit as fit
from scripts.issue_76_full_duration_loss import (
    full_continuous_loss,
    full_hybrid_loss,
    trajectory_batch,
)
from scripts.issue_76_matched_batches import indexed_batch, mixed_schedule, sampled_schedule
from world_model.training.cnn_hybrid import linear_macs
from world_model.training.native_history_fit import ENDPOINT, REFERENCE_TRANSITION_MACS


ROOT = fit.files.ROOT / ".local-artifacts/issue-76-balanced-native-v1"
OUTPUT = fit.files.ROOT / "data/issue-76-balanced-native"
SOURCES = (
    "scripts/run_issue_76_balanced_native.py",
    "scripts/issue_76_full_duration_loss.py",
    "scripts/issue_76_matched_batches.py",
    "tests/test_issue_76_balanced_native.py",
    "docs/issue-76-balanced-native-protocol.md",
)


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if fit.files.read(path) != value:
            raise ValueError(f"different immutable balanced-native artifact: {path}")
    else:
        fit.files.write(path, value)


def training_phase(update):
    if not 0 <= update < 12000:
        raise ValueError("balanced-native update is outside the frozen schedule")
    return "local" if update < 6000 else "full_duration"


def _balanced_sources(parent):
    saved_plan = fit.files.read(balanced.ROOT / "plan.json")
    if saved_plan != balanced.make_plan():
        raise ValueError("balanced parser plan changed")
    report = fit.files.read(balanced.OUTPUT / "report.json")
    if (report["plan"] != saved_plan or not report["balance_hypothesis_supported"]
            or len(report["summaries"]) != 3
            or not all(row["balance_hypothesis_supported"] for row in report["summaries"])):
        raise ValueError("balanced parser did not pass its frozen semantic diagnostic")
    cells = []
    for seed in parent["seeds"]:
        path = balanced.ROOT / "checkpoints" / f"visual-balanced-{seed}.pt"
        budget = fit.require_finished_budget(balanced.ROOT, f"balanced-{seed}")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (saved["plan_identity"] != saved_plan["identity"] or not saved["complete"]
                or saved["failure"] is not None or saved["updates_completed"] != 1500):
            raise ValueError("balanced parser checkpoint is incomplete")
        cells.append({"seed": seed, "path": str(path),
                      "plan_identity": saved["plan_identity"],
                      "updates_completed": saved["updates_completed"],
                      "source_budget": budget})
    return saved_plan, cells


def make_plan():
    parent = fit.load_plan()
    index = fit.data_index(fit.ROOT, parent)
    balanced_plan, parsers = _balanced_sources(parent)
    return {
        "schema": "issue_76_balanced_native_plan_v1",
        "identity": ROOT.name,
        "parent_plan_identity": parent["identity"],
        "parent_data_root": str(fit.ROOT),
        "entries": index["entries"],
        "seeds": list(parent["seeds"]),
        "capacity": parent["capacity"],
        "balanced_parser_plan_identity": balanced_plan["identity"],
        "balanced_parsers": parsers,
        "updates": {"history": 1000, "predictor": 12000, "controller": 4000},
        "predictor_phases": {"local_updates": 6000, "full_duration_updates": 6000},
        "batch_size": 32,
        "learning_rates": {"history": .001, "predictor": .0003, "controller": .001},
        "controller_compute_weight": .0001,
        "limits": {"common_seconds_per_seed": 3600,
                   "predictor_seconds_per_arm_seed": 14400,
                   "controller_seconds_per_arm_seed": 3600,
                   "cpu_rss_mib": 12288, "cuda_allocated_mib": 8192,
                   "new_artifact_bytes": 24 * 2**30},
        "source_text": {name: (fit.files.ROOT / name).read_text() for name in SOURCES},
        "event_outcomes_used": False,
        "fresh_access": False,
        "readiness_claimed": False,
        "advancement_authorized": False,
    }


def load_plan():
    plan = fit.files.read(ROOT / "plan.json")
    if plan != make_plan():
        raise ValueError("balanced-native source, data, or numeric freeze changed")
    return plan


def prepare():
    if ROOT.exists():
        raise ValueError("balanced-native stage is already prepared")
    plan = make_plan()
    immutable(ROOT / "plan.json", plan)
    immutable(OUTPUT / "plan.json", plan)
    fit.log("balanced-native protocol frozen; no new encoding or fitting performed")


def _parent(plan):
    parent = fit.load_plan()
    if parent["identity"] != plan["parent_plan_identity"]:
        raise ValueError("balanced-native parent plan differs")
    return parent


def _parser_cell(plan, seed):
    return next(cell for cell in plan["balanced_parsers"] if cell["seed"] == seed)


def load_parser(plan, seed, device):
    cell = _parser_cell(plan, seed)
    saved = torch.load(cell["path"], map_location=device, weights_only=False)
    if (saved["plan_identity"] != cell["plan_identity"] or not saved["complete"]
            or saved["failure"] is not None
            or saved["updates_completed"] != cell["updates_completed"]):
        raise ValueError("balanced parser source binding differs")
    parser = fit.NativeVisualParser().to(device).eval().requires_grad_(False)
    parser.load_state_dict(saved["model"])
    return parser


def visual_path(seed, entry):
    return fit.visual_path(ROOT, seed, entry)


def carrier_path(seed, entry):
    return fit.carrier_path(ROOT, seed, entry)


def load_visual(plan, seed, entry):
    value = torch.load(visual_path(seed, entry), map_location="cpu", weights_only=False)
    if (value["plan_identity"] != plan["identity"] or value["seed"] != seed
            or value["member_identity"] != entry["member_identity"]):
        raise ValueError("balanced-native visual source differs")
    return value["visual"]


def load_carrier(plan, seed, entry):
    value = torch.load(carrier_path(seed, entry), map_location="cpu", weights_only=False)
    if (value["plan_identity"] != plan["identity"] or value["seed"] != seed
            or value["member_identity"] != entry["member_identity"]):
        raise ValueError("balanced-native carrier source differs")
    return value["carrier"]


def _parent_shard(parent, entry, *, fitting=False):
    return fit.load_shard(fit.ROOT, entry, parent, fitting=fitting)


def train_common(plan, device):
    parent = _parent(plan)
    training = [entry for entry in plan["entries"] if entry["exposure_role"] == "training"]
    for seed in plan["seeds"]:
        final = fit.common_path(ROOT, seed)
        if final.exists():
            value = torch.load(final, map_location="cpu", weights_only=False)
            if (value["plan_identity"] != plan["identity"] or value["seed"] != seed
                    or value["parser_checkpoint"] != _parser_cell(plan, seed)):
                raise ValueError("balanced-native common checkpoint differs")
            fit.require_finished_budget(ROOT, f"common-{seed}")
            fit.log(f"retain completed balanced-native common seed={seed}")
            continue
        torch.manual_seed(seed)
        with fit.fitting_budget(ROOT, f"common-{seed}",
                                plan["limits"]["common_seconds_per_seed"], device) as budget:
            parser = load_parser(plan, seed, device)
            for ordinal, entry in enumerate(plan["entries"], 1):
                budget.check()
                if not entry["usable"]:
                    continue
                target = visual_path(seed, entry)
                if not target.exists():
                    shard = _parent_shard(parent, entry)
                    visual = fit.encode_visual(parser, shard["tensors"])
                    fit.atomic_torch(target, {"plan_identity": plan["identity"], "seed": seed,
                        "member_identity": entry["member_identity"], "visual": visual})
                if ordinal % 25 == 0 or ordinal == len(plan["entries"]):
                    budget.save()
                    fit.log(f"balanced-native seed={seed} visual={ordinal}/{len(plan['entries'])}")
            common = fit.CommonHistoryFit().to(device)

            def history_loss(update):
                entry = training[update % len(training)]
                if not entry["usable"]:
                    return None
                shard = _parent_shard(parent, entry, fitting=True)
                visual = load_visual(plan, seed, entry)
                return common.loss(visual.to(device),
                                   shard["tensors"]["timestamps"].to(device),
                                   shard["segment_ranges"])

            fit.fit_updates(ROOT / "checkpoints" / f"history-{seed}.pt", common,
                            plan_identity=plan["identity"], updates=plan["updates"]["history"],
                            lr=plan["learning_rates"]["history"], make_loss=history_loss,
                            budget=budget, device=device)
            common.eval().requires_grad_(False)
            for ordinal, entry in enumerate(plan["entries"], 1):
                budget.check()
                if not entry["usable"]:
                    continue
                target = carrier_path(seed, entry)
                if not target.exists():
                    shard = _parent_shard(parent, entry)
                    visual = load_visual(plan, seed, entry)
                    with torch.no_grad():
                        carrier = fit.encode_history(common.history, visual.to(device),
                            shard["tensors"]["timestamps"].to(device), shard["segment_ranges"]).cpu()
                    fit.atomic_torch(target, {"plan_identity": plan["identity"], "seed": seed,
                        "member_identity": entry["member_identity"], "carrier": carrier})
                if ordinal % 25 == 0 or ordinal == len(plan["entries"]):
                    budget.save()
                    fit.log(f"balanced-native seed={seed} carrier={ordinal}/{len(plan['entries'])}")
            budget.check()
            fit.atomic_torch(final, {"schema": "issue_76_balanced_native_common_v1",
                "plan_identity": plan["identity"], "seed": seed,
                "parser_checkpoint": _parser_cell(plan, seed),
                "history": common.history.state_dict()})
        del parser, common
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def train_predictors(plan, device):
    parent = _parent(plan)
    training = [entry for entry in plan["entries"] if entry["exposure_role"] == "training"]
    lengths = [entry["frames"] if entry["usable"] else 0 for entry in training]
    for seed in plan["seeds"]:
        fit.require_finished_budget(ROOT, f"common-{seed}")
        shards = [_parent_shard(parent, entry, fitting=True) if entry["usable"] else None
                  for entry in training]
        carriers = [load_carrier(plan, seed, entry) if entry["usable"] else None
                    for entry in training]
        for pure in (False, True):
            if fit.retain_completed_fit(ROOT, plan, seed, pure, "predictor"):
                continue
            name = "pure" if pure else "hybrid"
            original = sampled_schedule(lengths, seed, pure, plan["updates"]["predictor"],
                                        plan["batch_size"])
            schedule = mixed_schedule(original, seed, pure)
            torch.manual_seed(seed)
            with fit.fitting_budget(ROOT, f"predictor-{name}-{seed}",
                                    plan["limits"]["predictor_seconds_per_arm_seed"], device) as budget:
                model = fit.new_predictor(plan, pure, device)

                def predictor_loss(update):
                    if update not in schedule:
                        return None
                    pair = fit.pair_for_update(pure, update)
                    if training_phase(update) == "local":
                        batch = indexed_batch(shards, carriers, schedule[update], pair.delta,
                                              symbolic=not pure)
                        batch = {key: value.to(device) for key, value in batch.items()}
                        return (fit.continuous_loss(model, batch["z"], batch["available"],
                                                    batch["action"], pair) if pure
                                else fit.hybrid_loss(model, batch, pair))
                    batch = trajectory_batch(shards, carriers, schedule[update], pair.delta,
                                             symbolic=not pure)
                    batch = {key: value.to(device) for key, value in batch.items()}
                    return (full_continuous_loss(model, batch["z"], batch["available"],
                                                 batch["action"], pair) if pure
                            else full_hybrid_loss(model, batch, pair))

                result = fit.fit_updates(fit.model_path(ROOT, seed, pure, "predictor"), model,
                    plan_identity=plan["identity"], updates=plan["updates"]["predictor"],
                    lr=plan["learning_rates"]["predictor"], make_loss=predictor_loss,
                    budget=budget, device=device)
                if (result["updates_applied"], result["updates_skipped"]) != (len(schedule), 96):
                    raise ValueError("balanced-native predictor exposure differs")
            del model, result, original, schedule
            gc.collect()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        del shards, carriers
        gc.collect()


@torch.no_grad()
def controller_targets(model, carrier, fixed_steps, action, *, compute_weight):
    keep = [index for index, step in enumerate(fixed_steps)
            if (step - fixed_steps[0]) % 50 == 0 and step - fixed_steps[0] <= ENDPOINT]
    if len(keep) < 2:
        return None
    z = carrier[keep]
    steps = [fixed_steps[index] for index in keep]
    lookup = {step: index for index, step in enumerate(steps)}
    costs = z.new_full((len(z), len(model.pairs)), torch.inf)
    for column, pair in enumerate(model.pairs):
        rows = [index for index, step in enumerate(steps) if step + pair.delta in lookup]
        if not rows:
            continue
        ends = [lookup[steps[index] + pair.delta] for index in rows]
        predicted = model.carrier(z[rows], action[None].expand(len(rows), -1), pair)
        costs[rows, column] = (pair.delta * .0004 * (predicted - z[ends]).square().mean(-1)
                               + compute_weight * linear_macs(model, pair)
                               / REFERENCE_TRANSITION_MACS)
    values = z.new_zeros(len(z))
    labels = torch.zeros(len(z) - 1, dtype=torch.long, device=z.device)
    for index in range(len(z) - 2, -1, -1):
        choices = costs[index].clone()
        for column, pair in enumerate(model.pairs):
            end = lookup.get(steps[index] + pair.delta)
            if end is not None:
                choices[column] += values[end]
        values[index], labels[index] = choices.min(0)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("balanced-native controller teacher lacks an exact path")
    return {"z": z[:-1], "action": action[None].expand(len(z) - 1, -1),
            "labels": labels,
            "remaining": torch.tensor([steps[-1] - step for step in steps[:-1]], device=z.device)}


def train_controllers(plan, device):
    parent = _parent(plan)
    training = [entry for entry in plan["entries"] if entry["exposure_role"] == "training"]
    for seed in plan["seeds"]:
        for pure in (False, True):
            if fit.retain_completed_fit(ROOT, plan, seed, pure, "controller"):
                continue
            name = "pure" if pure else "hybrid"
            torch.manual_seed(seed)
            with fit.fitting_budget(ROOT, f"controller-{name}-{seed}",
                                    plan["limits"]["controller_seconds_per_arm_seed"], device) as budget:
                model = fit.load_predictor(ROOT, plan, seed, pure, device)
                for ordinal, entry in enumerate(training, 1):
                    budget.check()
                    if not entry["usable"]:
                        continue
                    target = fit.teacher_path(ROOT, seed, pure, entry)
                    if not target.exists():
                        shard = _parent_shard(parent, entry, fitting=True)
                        carrier = load_carrier(plan, seed, entry).to(device)
                        parts = []
                        for segment in shard["segment_ranges"]:
                            start, stop = segment["start"], segment["stop"]
                            value = controller_targets(model, carrier[start:stop],
                                shard["tensors"]["fixed_steps"][start:stop].tolist(),
                                segment["action"].to(device),
                                compute_weight=plan["controller_compute_weight"])
                            if value:
                                parts.append({key: tensor.cpu() for key, tensor in value.items()})
                        tensors = ({key: torch.cat([part[key] for part in parts]) for key in parts[0]}
                                   if parts else {})
                        fit.atomic_torch(target, {"plan_identity": plan["identity"],
                            "member_identity": entry["member_identity"], "seed": seed,
                            "pure": pure, "exposure_role": "training", "tensors": tensors})
                    if ordinal % 10 == 0 or ordinal == len(training):
                        budget.save()
                        fit.log(f"balanced-native controller {name} seed={seed} teacher={ordinal}/{len(training)}")
                controller = fit.NativeHistoryController(pure).to(device)

                def controller_loss(update):
                    entry = training[update % len(training)]
                    if not entry["usable"]:
                        return None
                    value = torch.load(fit.teacher_path(ROOT, seed, pure, entry),
                                       map_location="cpu", weights_only=False)
                    if (value["plan_identity"] != plan["identity"]
                            or value["exposure_role"] != "training"
                            or value["seed"] != seed or value["pure"] != pure):
                        raise ValueError("balanced-native controller teacher differs")
                    data = value["tensors"]
                    if not data:
                        return None
                    rows = torch.randint(len(data["z"]), (plan["batch_size"],),
                                         generator=fit.generator(seed, update))
                    batch = {key: tensor[rows].to(device) for key, tensor in data.items()}
                    return F.cross_entropy(controller(batch["z"], batch["action"],
                                                      batch["remaining"]), batch["labels"])

                fit.fit_updates(fit.model_path(ROOT, seed, pure, "controller"), controller,
                    plan_identity=plan["identity"], updates=plan["updates"]["controller"],
                    lr=plan["learning_rates"]["controller"], make_loss=controller_loss,
                    budget=budget, device=device)
            del model, controller
            gc.collect()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()


def diagnose(plan, device):
    report_path = ROOT / "report.json"
    if report_path.exists():
        value = fit.files.read(report_path)
        if value["plan_identity"] != plan["identity"]:
            raise ValueError("balanced-native report belongs to another plan")
        immutable(OUTPUT / "report.json", value)
        return value
    rows = []
    for seed in plan["seeds"]:
        usable = [entry for entry in plan["entries"] if entry["usable"]]
        first = torch.stack([load_carrier(plan, seed, entry)[0] for entry in usable])
        seed_row = {"seed": seed,
                    "first_carrier_mean_feature_std": float(first.std(0).mean()),
                    "first_carrier_max_feature_std": float(first.std(0).max()),
                    "arms": {}}
        for pure in (False, True):
            name = "pure" if pure else "hybrid"
            fit.require_finished_budget(ROOT, f"predictor-{name}-{seed}")
            fit.require_finished_budget(ROOT, f"controller-{name}-{seed}")
            model = fit.load_predictor(ROOT, plan, seed, pure, device)
            saved = torch.load(fit.model_path(ROOT, seed, pure, "controller"),
                               map_location=device, weights_only=False)
            if saved["plan_identity"] != plan["identity"] or not saved["complete"]:
                raise ValueError("balanced-native controller is incomplete")
            controller = fit.NativeHistoryController(pure).to(device).eval().requires_grad_(False)
            controller.load_state_dict(saved["model"])
            counts, failures = Counter(), 0
            with torch.no_grad():
                for entry in [value for value in plan["entries"]
                              if value["exposure_role"] == "training" and value["usable"]][:5]:
                    shard = _parent_shard(_parent(plan), entry, fitting=True)
                    segment = shard["segment_ranges"][0]
                    try:
                        _, trace = fit.rollout(model, controller,
                            load_carrier(plan, seed, entry)[segment["start"]].to(device),
                            segment["action"].to(device))
                        counts.update((row["mode"], row["horizon_native_steps"]) for row in trace)
                    except (RuntimeError, ValueError):
                        failures += 1
            seed_row["arms"][name] = {"trace_choice_counts": {
                f"{mode}-{horizon}": count for (mode, horizon), count in sorted(counts.items())},
                "trace_failures": failures}
        rows.append(seed_row)
    budgets = {path.stem: fit.files.read(path) for path in sorted((ROOT / "budgets").glob("*.json"))}
    value = {"schema": "issue_76_balanced_native_report_v1",
             "plan_identity": plan["identity"], "rows": rows, "budgets": budgets,
             "all_jobs_complete": len(budgets) == 15
                                  and not any(item["stopped"] or item.get("running")
                                              for item in budgets.values()),
             "event_outcomes_used": False, "fresh_access": False,
             "readiness_claimed": False, "advancement_authorized": False}
    immutable(report_path, value)
    immutable(OUTPUT / "report.json", value)
    fit.log(f"balanced-native engineering report complete jobs={len(budgets)}")
    return value


def smoke_test(device):
    """Exercise real parser/data and both losses without saving research weights."""

    plan = make_plan()
    parent = _parent(plan)
    entries = [entry for entry in plan["entries"]
               if entry["exposure_role"] == "training" and entry["usable"]][:3]
    seed = plan["seeds"][0]
    parser = load_parser(plan, seed, device)
    shards = [_parent_shard(parent, entry, fitting=True) for entry in entries]
    visuals = [fit.encode_visual(parser, shard["tensors"]) for shard in shards]
    common = fit.CommonHistoryFit().to(device)
    loss = common.loss(visuals[0].to(device), shards[0]["tensors"]["timestamps"].to(device),
                       shards[0]["segment_ranges"])
    loss.backward()
    common.eval().requires_grad_(False)
    with torch.no_grad():
        carriers = [fit.encode_history(common.history, visual.to(device),
            shard["tensors"]["timestamps"].to(device), shard["segment_ranges"]).cpu()
            for visual, shard in zip(visuals, shards, strict=True)]
    lengths = [len(value) for value in carriers]
    for pure in (False, True):
        rows = sampled_schedule(lengths, seed, pure, 1, batch_size=4)[0]
        pair = fit.pair_for_update(pure, 0)
        model = fit.new_predictor(plan, pure, device)
        local = indexed_batch(shards, carriers, rows, pair.delta, symbolic=not pure)
        local = {key: value.to(device) for key, value in local.items()}
        local_loss = (fit.continuous_loss(model, local["z"], local["available"],
                                          local["action"], pair) if pure
                      else fit.hybrid_loss(model, local, pair))
        local_loss.backward()
        model.zero_grad(set_to_none=True)
        full = trajectory_batch(shards, carriers, rows, pair.delta, symbolic=not pure)
        full = {key: value.to(device) for key, value in full.items()}
        full_loss = (full_continuous_loss(model, full["z"], full["available"],
                                          full["action"], pair) if pure
                     else full_hybrid_loss(model, full, pair))
        full_loss.backward()
        segment = shards[0]["segment_ranges"][0]
        teacher = controller_targets(model.eval(), carriers[0][segment["start"]:segment["stop"]].to(device),
            shards[0]["tensors"]["fixed_steps"][segment["start"]:segment["stop"]].tolist(),
            segment["action"].to(device), compute_weight=plan["controller_compute_weight"])
        if teacher is None or not bool(torch.isfinite(teacher["z"]).all()):
            raise ValueError("balanced-native smoke controller target is unavailable")
    fit.log("balanced-native real smoke passed; no research checkpoint was written")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "smoke-test", "prepare", "fit-common", "fit-predictors",
                 "fit-controllers", "diagnose", "run"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.dry_run:
        plan = make_plan()
        print({"entries": len(plan["entries"]), "seeds": plan["seeds"],
               "updates": plan["updates"], "event_outcomes_used": False}, flush=True)
        return
    if args.smoke_test:
        smoke_test(args.device)
        return
    if args.prepare:
        prepare()
        return
    plan = load_plan()
    if args.fit_common or args.run:
        train_common(plan, args.device)
    if args.fit_predictors or args.run:
        train_predictors(plan, args.device)
    if args.fit_controllers or args.run:
        train_controllers(plan, args.device)
    if args.diagnose or args.run:
        diagnose(plan, args.device)


if __name__ == "__main__":
    main()
