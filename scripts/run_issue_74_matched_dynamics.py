"""Source-frozen, bounded-memory matched dynamics training; no fresh evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from pathlib import Path
import resource
import subprocess
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from scripts import run_issue_71_hybrid_readiness as old
from scripts import run_issue_72_matched_grid as grid
from world_model.model import Abstraction, PredictionPair
from world_model.planning.gameplay import SlingshotAction
from world_model.planning.task_objective import TaskObjective
from world_model.training.cnn_hybrid import CNNHybridPredictor, PAIRS, training_loss, intervention_audit
from world_model.training.matched_dynamics import (
    ContinuousDynamics, CONTINUOUS_PAIRS, MatchedController, capacity_contract,
    continuous_loss, controller_labels, parameter_count, pairs_for, rollout, work, active_capacity,
)

ROOT = old.repair.experiment.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-74-matched-dynamics-v1"
SEEDS = (20260908, 20260909, 20260910)
ARMS = ("continuous", "hybrid")
POLICIES = ("continuous_h1", "continuous_h5", "continuous_h15", "continuous_adaptive",
            "hybrid_continuous_h15", "hybrid_adaptive")
FILES = tuple(dict.fromkeys((*old.SOURCE_FILES, "scripts/run_issue_72_matched_grid.py",
                            "scripts/run_issue_74_matched_dynamics.py", "world_model/training/matched_dynamics.py",
                            "world_model/planning/task_objective.py")))
read, write = old.read, old.write


def log(message):
    print(f"[issue-74] {message}", flush=True)


def memory(device):
    return {"peak_cpu_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device)/2**20
            if str(device).startswith("cuda") else 0.}


def source_args(args):
    return argparse.Namespace(output=args.issue71, source=args.parser, device=args.device)


def grid_args(args):
    return argparse.Namespace(output=grid.OUTPUT, issue71=args.issue71, parser=args.parser, device=args.device)


def make_plan(args):
    source = old.load_plan(source_args(args))
    development = grid.load_plan(grid_args(args))
    if read(args.issue71/"data.json")["lineages"] != 3000:
        raise ValueError("requires completed #71 training shards")
    return {"schema": "issue_74_matched_dynamics_v1", "identity": "issue-74-matched-dynamics-v1",
            "issue71": str(args.issue71.resolve()), "parser": str(args.parser.resolve()),
            "source_plan_identity": source["identity"], "contract": source["contract"],
            "source_release_identity": source["source_release_identity"],
            "development_plan_identity": development["identity"],
            "capacity": capacity_contract(), "seeds": list(SEEDS),
            "training": {"steps": 9000, "batch_size": 64, "learning_rate": .0001,
                         "weight_decay": .0001, "grad_clip": 1., "unroll": 4,
                         "schedule": "#71 three fitting lineages per nine steps; identical random minibatches per paired seed",
                         "horizons": "hybrid PAIRS[step%9]; continuous uses same horizon, all 9000 updates",
                         "recipe": "local MSE + recursive MSE + .01 carrier bound penalty; hybrid adds .1 symbolic losses",
                         "selection": "last update, no best-validation checkpoint",
                         "fit_lineages": 2400, "controller_lineages": 600,
                         "parser_overlap": "shared frozen CNN saw all 3000 training lineages; none of calibration used for fitting"},
            "controller": {"steps_per_round": 1800, "batch_size": 128, "learning_rate": .001,
                           "rounds": 2, "compute_weight": .0001,
                           "coverage": "first window per shot, both arms; #75 later-window intervention NOT applied",
                           "utility": "duration times continuous carrier MSE + own continuous-h15-normalized linear MAC cost + DP continuation",
                           "aggregation": "one round; predicted visited contexts, unchanged training-only continuous targets",
                           "capacity_rule": "hybrid width128; nearest continuous width to its parameter count is131; tolerance0.2%; no dead output logits",
                           "capacity": {a: parameter_count(MatchedController(a == 'continuous')) for a in ARMS}},
            "systems": {"continuous": "independently initialized dynamics; no symbolic heads/adapters/labels",
                        "hybrid_adaptive": "new matched hybrid fit; exclusive learned mode per step",
                        "hybrid_continuous_h15": "SAME hybrid checkpoint, NOT pure-continuous training",
                        "prior": "frozen #72 fixed ordinal09; no new prior selection"},
            "development": {"role": "calibration", "states": 200, "candidates": 12, "endpoint": 225,
                            "policies": list(POLICIES), "batch": 1,
                            "selection": "one continuous policy minimizing mean regret across ALL three seeds/states; ties by listed order; no seed selection",
                            "failure": "prediction failure gives state regret1 and disqualifies policy from selection",
                            "uncertainty": "paired state bootstrap descriptive only; three per-seed means, not seed-robust proof",
                            "prior": "fixed ordinal09 from #72 development"},
            "compute": {"comparison": "common 12 actions and 225 steps; report full fixed/adaptive work-quality frontier, NOT equal FLOPs",
                        "selection_budget": "all h1/h5/h15/adaptive policies eligible; strongest mean regret, no forced h15 handicap",
                        "fresh_budget": "#76 must freeze numerical equal-work/frontier protocol before fresh access",
                        "accounting": "linear MACs, calls, synchronized inference wall; CNN and B3 same-image audit separate; no infilling",
                        "training_compute_matched": False},
            "cost_estimates_not_measurements": {"new_capture_count": 0, "new_disk_gib_upper_planning": 2,
                                               "cpu_ram_gib_planning": 3, "cuda_ram_gib_planning": 2,
                                               "training_and_labels_gpu_hours_planning": 4},
            "claim_boundary": "shared semantically supervised CNN; pure CONTINUOUS DYNAMICS, not symbol-free end-to-end vision; readiness not superiority; no unique joint mechanism claim; #15 unchanged",
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_text": {p: (ROOT/p).read_text() for p in FILES},
            "archived_release": False, "fresh_evaluation_opened": False,
            "final_evaluation_opened": False, "issue_64_authorized": False}


def load_plan(args):
    plan = read(args.output/"plan.json")
    current = make_plan(args)
    # A later commit of the identical source is archival metadata, not a new fit.
    current["source_revision"] = plan["source_revision"]
    if plan != current:
        raise ValueError("frozen #74 source/settings differ; retain old output and explicitly version any change")
    return plan


def cell(args, seed, arm):
    return args.output/f"seed-{seed}"/arm


def new_model(plan, arm):
    return ContinuousDynamics(plan["capacity"]["continuous_width"]) if arm == "continuous" else CNNHybridPredictor()


def binding(plan, seed, arm, component):
    return {"plan_identity": plan["identity"], "contract": plan["contract"], "capacity": plan["capacity"],
            "source_release_identity": plan["source_release_identity"], "seed": seed, "arm": arm,
            "component": component, "training": plan["training"], "controller": plan["controller"]}


def check_binding(value, expected):
    if value["binding"] != expected:
        raise ValueError("checkpoint source/representation/training binding differs")


def shard(args, source, index, pure=False):
    data = old.load_shard(source_args(args), source, index)
    return {k: data[k] for k in ("z", "action", "length")} if pure else data


def loss_for(model, batch, pair):
    if isinstance(model, ContinuousDynamics):
        return continuous_loss(model, batch["z"], batch["action"], batch["length"], pair)
    return training_loss(model, batch, pair, True)


def train_cell(args, plan, source, seed, arm, stop_after=None):
    """Resume optimizer/sampler exactly. stop_after is for bounded smoke/tests only."""
    root = cell(args, seed, arm); target = root/"predictor.pt"
    if target.exists():
        load_predictor(args, plan, seed, arm); log(f"seed={seed} arm={arm} predictor reused"); return
    torch.manual_seed(seed)
    model = new_model(plan, arm).to(args.device)
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    recipe = plan["training"]
    optim = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    expected = binding(plan, seed, arm, "predictor")
    start, prior, gradients, counts = 0, 0., set(), Counter()
    progress = root/"predictor-progress.pt"
    if progress.exists():
        saved = torch.load(progress, map_location="cpu", weights_only=True); check_binding(saved, expected)
        model.load_state_dict(saved["model"], strict=True); optim.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"])
        start, prior = saved["step"], saved["wall_seconds"]
        gradients, counts = set(saved["gradient_parameters"]), Counter(saved["pair_counts"])
    began = time.monotonic(); cached = None
    finish = min(recipe["steps"], stop_after if stop_after is not None else recipe["steps"])
    for step in range(start, finish):
        group = old.fitting_lineage_group(step)
        if group != cached:
            rows = [shard(args, source, i, arm == "continuous") for i in group]
            data = {k: torch.cat([r[k] for r in rows]) for k in rows[0]}; cached = group
        batch = old.sample(data, recipe["batch_size"], generator, args.device)
        selected = PAIRS[step % 9]
        pair = PredictionPair(selected.delta, Abstraction.CONTINUOUS) if arm == "continuous" else selected
        optim.zero_grad(set_to_none=True)
        loss = loss_for(model, batch, pair)
        if not bool(torch.isfinite(loss)):
            raise ValueError(f"nonfinite training loss seed={seed} arm={arm} step={step+1}")
        loss.backward()
        gradients.update(n for n, p in model.named_parameters() if p.grad is not None and bool((p.grad != 0).any()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["grad_clip"]); optim.step()
        counts[str(pair.identity)] += 1
        if (step+1) % 90 == 0 or step+1 == finish:
            elapsed = prior+time.monotonic()-began
            saved = {"binding": expected, "model": model.state_dict(), "optimizer": optim.state_dict(),
                     "generator": generator.get_state(), "step": step+1, "wall_seconds": elapsed,
                     "gradient_parameters": sorted(gradients), "pair_counts": dict(counts),
                     "changed_parameters": [n for n,p in model.named_parameters() if not torch.equal(p.detach().cpu(), initial[n])],
                     "optimizer_examples": (step+1)*recipe["batch_size"], "memory": memory(args.device)}
            old.repair.atomic_torch(progress, saved)
            log(f"train seed={seed} arm={arm} step={step+1}/{recipe['steps']} loss={float(loss.detach()):.6f} elapsed={elapsed:.1f}s eta={elapsed/(step+1)*(recipe['steps']-step-1):.1f}s memory={saved['memory']}")
    if finish == recipe["steps"]:
        old.repair.atomic_torch(target, saved)


def load_predictor(args, plan, seed, arm):
    value = torch.load(cell(args, seed, arm)/"predictor.pt", map_location="cpu", weights_only=True)
    check_binding(value, binding(plan, seed, arm, "predictor"))
    model = new_model(plan, arm).to(args.device); model.load_state_dict(value["model"], strict=True)
    names = set(dict(model.named_parameters()))
    expected_counts = {str(p.identity): plan["training"]["steps"]//len(pairs_for(model)) for p in pairs_for(model)}
    if (value["step"] != plan["training"]["steps"] or value["pair_counts"] != expected_counts
            or set(value["gradient_parameters"]) != names or set(value["changed_parameters"]) != names):
        raise ValueError("incomplete predictor budget/coverage/task gradients/updates")
    return model.eval(), value


def first_windows(args, index):
    raw = torch.load(old.shard_path(source_args(args), index), map_location="cpu", weights_only=True)
    seen, selected = set(), []
    for i, window in enumerate(raw["windows"]):
        if window["shot"] not in seen:
            seen.add(window["shot"]); selected.append(i)
    return selected


def label_lineage(args, source, model, controller, index, aggregated):
    data = shard(args, source, index, pure=True)
    records = []
    for position in first_windows(args, index):
        batch = {k: v[position:position+1].to(args.device) for k,v in data.items()}
        z, a, length = batch["z"], batch["action"], batch["length"]
        contexts = z.clone()
        if aggregated:
            current, t = z[:, 0], 0
            with torch.no_grad():
                while t < int(length[0]):
                    pair = controller.pairs[int(controller(current, a, length-t).argmax(-1))]
                    current = model.carrier(current, a, pair); t += pair.delta
                    contexts[:, t] = current
        labels = controller_labels(model, controller, z, a, length, contexts)
        n = int(length[0])
        records.append({"z": contexts[0, :n].cpu(), "action": a.expand(n, -1).cpu(),
                        "remaining": torch.arange(n, 0, -1), "labels": labels[0, :n].cpu()})
    return {k: torch.cat([r[k] for r in records]) for k in records[0]}


def train_control(args, plan, source, seed, arm, indices=tuple(range(5, 3001, 5))):
    root = cell(args, seed, arm)
    model, _ = load_predictor(args, plan, seed, arm)
    if (root/"controller.pt").exists():
        load_control(args, plan, seed, arm); log(f"seed={seed} arm={arm} controller reused"); return
    torch.manual_seed(seed+7400); control = MatchedController(arm == "continuous").to(args.device)
    initial = {n: p.detach().cpu().clone() for n,p in control.named_parameters()}
    expected = binding(plan, seed, arm, "controller"); recipe = plan["controller"]
    gradients = set()
    for round_index in (0, 1):
        checkpoint = root/f"controller-round-{round_index}.pt"
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True); check_binding(saved, expected)
            control.load_state_dict(saved["model"], strict=True); gradients.update(saved["gradient_parameters"]); continue
        began = time.monotonic()
        for ordinal, index in enumerate(indices, 1):
            path = root/f"labels-{round_index}"/f"lineage-{index:04d}.pt"
            if not path.exists():
                started = time.monotonic()
                tensors = label_lineage(args, source, model, control, index, bool(round_index))
                wall = time.monotonic()-started
                old.repair.atomic_torch(path, {"binding": expected, "record": source["records"][index-1],
                                              "round": round_index, "tensors": tensors, "wall_seconds": wall})
            else:
                cached = torch.load(path, weights_only=True, map_location="cpu")
                check_binding(cached, expected)
                if cached["record"] != source["records"][index-1] or cached["round"] != round_index:
                    raise ValueError("controller labels source/round differs")
            elapsed = time.monotonic()-began
            log(f"labels seed={seed} arm={arm} round={round_index+1}/2 lineage={ordinal}/{len(indices)} elapsed={elapsed:.1f}s eta={elapsed/ordinal*(len(indices)-ordinal):.1f}s")
        optim = torch.optim.AdamW(control.parameters(), lr=recipe["learning_rate"])
        generator = torch.Generator().manual_seed(seed+7400+round_index)
        progress = root/f"controller-progress-{round_index}.pt"
        start, prior = 0, 0.
        if progress.exists():
            saved = torch.load(progress, weights_only=True, map_location="cpu"); check_binding(saved, expected)
            control.load_state_dict(saved["model"], strict=True); optim.load_state_dict(saved["optimizer"])
            generator.set_state(saved["generator"]); start, prior = saved["step"], saved["wall_seconds"]
            gradients.update(saved["gradient_parameters"])
        began = time.monotonic()
        for step in range(start, recipe["steps_per_round"]):
            index = indices[step % len(indices)]
            rows = [torch.load(root/f"labels-{r}"/f"lineage-{index:04d}.pt", weights_only=True, map_location="cpu")["tensors"]
                    for r in range(round_index+1)]
            data = {k: torch.cat([r[k] for r in rows]) for k in rows[0]}
            batch = old.sample(data, recipe["batch_size"], generator, args.device)
            loss = F.cross_entropy(control(batch["z"], batch["action"], batch["remaining"]), batch["labels"])
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite controller loss")
            optim.zero_grad(set_to_none=True); loss.backward()
            gradients.update(n for n,p in control.named_parameters() if p.grad is not None and bool((p.grad != 0).any()))
            optim.step()
            if (step+1) % 60 == 0 or step+1 == recipe["steps_per_round"]:
                elapsed = prior+time.monotonic()-began
                saved = {"binding": expected, "model": control.state_dict(), "optimizer": optim.state_dict(),
                         "generator": generator.get_state(), "round": round_index, "step": step+1,
                         "wall_seconds": elapsed, "gradient_parameters": sorted(gradients),
                         "changed_parameters": [n for n,p in control.named_parameters() if not torch.equal(p.detach().cpu(), initial[n])],
                         "memory": memory(args.device)}
                old.repair.atomic_torch(progress, saved)
                log(f"controller seed={seed} arm={arm} round={round_index+1}/2 step={step+1}/{recipe['steps_per_round']} elapsed={elapsed:.1f}s eta={elapsed/(step+1)*(recipe['steps_per_round']-step-1):.1f}s")
        old.repair.atomic_torch(checkpoint, saved)
    old.repair.atomic_torch(root/"controller.pt", saved)


def load_control(args, plan, seed, arm):
    value = torch.load(cell(args, seed, arm)/"controller.pt", map_location="cpu", weights_only=True)
    check_binding(value, binding(plan, seed, arm, "controller"))
    control = MatchedController(arm == "continuous").to(args.device)
    control.load_state_dict(value["model"], strict=True)
    names = set(dict(control.named_parameters()))
    if (value["round"] != 1 or value["step"] != plan["controller"]["steps_per_round"]
            or set(value["gradient_parameters"]) != names or set(value["changed_parameters"]) != names):
        raise ValueError("controller is incomplete or lacks task-trained parameters")
    return control.eval(), value


def policy_parts(policy):
    arm = "continuous" if policy.startswith("continuous_") else "hybrid"
    fixed = None if policy.endswith("adaptive") else PredictionPair(int(policy.rsplit("h", 1)[1]), Abstraction.CONTINUOUS)
    return arm, fixed


def predict(model, controller, context, action, fixed_pair, objective):
    """Agent runtime boundary: no candidate outcome, future observation or target."""
    device = next(model.parameters()).device
    bounds = old.repair.experiment.old.probe._bounds(); shot = SlingshotAction(**action)
    if not bounds.contains(shot):
        raise ValueError("illegal candidate action")
    a = torch.tensor([[shot.drag_x/480, shot.drag_y/480, bounds.release_time_ms/1000, shot.tap_time_ms/1000, 1.]], device=device)
    grid.synchronize(device); began = time.monotonic()
    z, trace = rollout(model, controller, context[None].to(device), a, fixed_pair=fixed_pair)
    z = z[0].cpu(); cost = objective(z); grid.synchronize(device)
    return {"cost": cost, "endpoint": z.tolist(), "trace": trace, "failure": None,
            "wall_seconds": time.monotonic()-began}


def score(args, plan):
    gp = grid.load_plan(grid_args(args)); adapter = old.repair.load_repaired_adapter(args.parser, args.device)
    objective = TaskObjective.from_vocabulary(plan["contract"]["vocabulary"])
    for seed in SEEDS:
        models = {a: load_predictor(args, plan, seed, a)[0] for a in ARMS}
        controls = {a: load_control(args, plan, seed, a)[0] for a in ARMS}
        began = time.monotonic()
        for i in range(200):
            row = grid.endpoint(grid_args(args), gp, i)
            path = args.output/f"seed-{seed}"/"development"/f"state-{i+1:04d}.json"
            if path.exists():
                check_state(read(path), row, plan, seed, models, controls)
                log(f"score seed={seed} state={i+1}/200 cached"); continue
            log(f"score seed={seed} state={i+1}/200 perception start")
            context, perception = grid.perceive(grid_args(args), gp, i, row, adapter)
            arms = {}
            for policy in POLICIES:
                arm, fixed = policy_parts(policy); results = []
                for j, candidate in enumerate(row["candidates"], 1):
                    started = time.monotonic()
                    try:
                        item = predict(models[arm], controls[arm], context, candidate["action"], fixed, objective)
                    except ValueError as error:
                        item = {"cost": None, "endpoint": None, "trace": [], "failure": str(error),
                                "wall_seconds": time.monotonic()-started}
                    results.append({"candidate_identity": candidate["identity"], "action": candidate["action"], **item})
                    log(f"score seed={seed} state={i+1}/200 policy={policy} candidate={j}/12 failure={item['failure']}")
                arms[policy] = results
            record = {"plan_identity": plan["identity"], "seed": seed, "state": row["state"],
                      "perception": perception, "policies": arms}
            check_state(record, row, plan, seed, models, controls); write(path, record)
            elapsed = time.monotonic()-began
            log(f"score seed={seed} state={i+1}/200 complete elapsed={elapsed:.1f}s eta={elapsed/(i+1)*(199-i):.1f}s")


def check_state(record, row, plan, seed, models, controls):
    if (record["plan_identity"] != plan["identity"] or record["seed"] != seed or record["state"] != row["state"]
            or set(record["policies"]) != set(POLICIES)):
        raise ValueError("scoring source/seed/policy inventory differs")
    objective = TaskObjective.from_vocabulary(plan["contract"]["vocabulary"])
    for policy, items in record["policies"].items():
        arm, fixed = policy_parts(policy)
        for item, candidate in zip(items, row["candidates"], strict=True):
            if item["candidate_identity"] != candidate["identity"] or item["action"] != candidate["action"]:
                raise ValueError("candidate source/action differs")
            if item["failure"]:
                if item["cost"] is not None:
                    raise ValueError("failed prediction has cost")
                continue
            if objective(torch.tensor(item["endpoint"])) != item["cost"]:
                raise ValueError("cost does not match predicted carrier")
            t = 0
            for item_trace in item["trace"]:
                pair = PredictionPair(item_trace["horizon"], Abstraction(item_trace["mode"]))
                expected = {"start_fixed_step": t, "horizon": pair.delta, "mode": str(pair.abstraction),
                            "linear_macs": work(models[arm], pair, controls[arm] if fixed is None else None),
                            "controller_calls": int(fixed is None), "transition_calls": 1,
                            "symbol_decoder_calls": int(pair.abstraction != Abstraction.CONTINUOUS)}
                if pair not in pairs_for(models[arm]) or (fixed is not None and pair != fixed) or item_trace != expected:
                    raise ValueError("execution/work trace differs")
                t += pair.delta
            if t != 225:
                raise ValueError("different physical endpoint")


def publication(args, plan):
    gp = grid.load_plan(grid_args(args)); by_seed = {}; total_fit = 0.; total_labels = 0.; checkpoints = []
    objective = TaskObjective.from_vocabulary(plan["contract"]["vocabulary"])
    for seed in SEEDS:
        models, controls = {}, {}
        for arm in ARMS:
            models[arm], fitted = load_predictor(args, plan, seed, arm)
            controls[arm], control = load_control(args, plan, seed, arm)
            root = cell(args, seed, arm); total_fit += fitted["wall_seconds"]
            for r in (0, 1):
                total_fit += torch.load(root/f"controller-round-{r}.pt", map_location="cpu", weights_only=True)["wall_seconds"]
                for index in range(5, 3001, 5):
                    label = torch.load(root/f"labels-{r}"/f"lineage-{index:04d}.pt", map_location="cpu", weights_only=True)
                    check_binding(label, binding(plan, seed, arm, "controller")); total_labels += label["wall_seconds"]
                log(f"validate seed={seed} arm={arm} controller round={r+1}/2 labels checked")
            probe = torch.zeros(1, 236, device=args.device); action = torch.zeros(1, 5, device=args.device)
            executed = {str(p.identity): active_capacity(models[arm], probe, action, p) for p in pairs_for(models[arm])}
            checkpoints.append({"seed": seed, "arm": arm, "predictor": str(root/"predictor.pt"),
                                "controller": str(root/"controller.pt"), "parameters": parameter_count(models[arm]),
                                "controller_parameters": parameter_count(controls[arm]), "binding": fitted["binding"],
                                "trainable_parameters": sum(p.numel() for p in models[arm].parameters() if p.requires_grad),
                                "executed_operator_parameters_by_pair": executed,
                                "active_capacity_scope": "executed leaf operators; embedding tables counted in full, not just selected rows; active capacity differs and is not claimed equal",
                                "pair_counts": fitted["pair_counts"], "optimizer_examples": fitted["optimizer_examples"],
                                "predictor_memory": fitted["memory"], "controller_memory": control["memory"]})
        metrics = {p: [] for p in POLICIES}; macs = Counter(); usages = {p: Counter() for p in POLICIES}
        errors = {p: [] for p in POLICIES}; counts = {p: [] for p in POLICIES}; walls = {p: [] for p in POLICIES}
        prior = []; audit_seconds = 0.
        for i in range(200):
            row = grid.endpoint(grid_args(args), gp, i)
            record = read(args.output/f"seed-{seed}"/"development"/f"state-{i+1:04d}.json")
            check_state(record, row, plan, seed, models, controls)
            audit_seconds += record["perception"]["anchor_validation_seconds"]
            prior.append(grid.normalized_regrets(row["candidates"])[0][8])
            for policy, items in record["policies"].items():
                metrics[policy].append(grid.ranked([x["cost"] for x in items], row["candidates"]))
                walls[policy].append(record["perception"]["wall_seconds"]+sum(x["wall_seconds"] for x in items))
                for item, target in zip(items, row["candidates"], strict=True):
                    for t in item["trace"]:
                        macs[policy] += t["linear_macs"]; usages[policy][f"{t['horizon']}:{t['mode']}"] += 1
                    if item["endpoint"] is not None and target.get("horizon_carrier") is not None:
                        pred, truth = torch.tensor(item["endpoint"]), torch.tensor(target["horizon_carrier"])
                        errors[policy].append(float((pred-truth).square().mean()))
                        counts[policy].append([abs(a-b) for a,b in zip(objective.counts(pred), objective.counts(truth), strict=True)])
            if (i+1) % 10 == 0:
                log(f"validate seed={seed} state={i+1}/200")
        by_seed[str(seed)] = {"policies": {p: {"mean_regret": float(np.mean([m['regret'] for m in metrics[p]])),
                                              "state_regrets": [m["regret"] for m in metrics[p]],
                                              "top1_fraction": float(np.mean([m['top1'] for m in metrics[p]])),
                                              "top3_fraction": float(np.mean([m['top3'] for m in metrics[p]])),
                                              "prediction_failure_states": sum(m["prediction_failure"] for m in metrics[p]),
                                              "compute_complete": not any(m["prediction_failure"] for m in metrics[p]),
                                              "mac_scope": "successful predictions; failed partial traces unavailable, not zero-cost evidence",
                                              "predicted_all_tied_states": sum(m["all_tied"] for m in metrics[p]),
                                              "linear_macs": macs[p], "pair_usage": dict(usages[p]),
                                              "mean_endpoint_carrier_mse": float(np.mean(errors[p])) if errors[p] else None,
                                              "parsed_pig_block_count_mae": np.mean(counts[p], axis=0).tolist() if counts[p] else None,
                                              "mean_perception_planning_seconds": float(np.mean(walls[p])),
                                              "max_perception_planning_seconds": max(walls[p])} for p in POLICIES},
                                  "prior_mean_regret": float(np.mean(prior)), "offline_anchor_audit_seconds": audit_seconds}
    selected = select_continuous(by_seed)
    usable_hybrid = not any(s["policies"][p]["prediction_failure_states"] for s in by_seed.values()
                            for p in ("hybrid_adaptive", "hybrid_continuous_h15"))
    contrasts = {}
    for comparator in ([selected] if selected else []) + ["hybrid_continuous_h15"]:
        paired = np.mean([np.array(s["policies"][comparator]["state_regrets"])-s["policies"]["hybrid_adaptive"]["state_regrets"]
                          for s in by_seed.values()], axis=0)
        contrasts[comparator] = grid.paired_interval(paired)
    return {"schema": "issue_74_readiness_v1", "plan_identity": plan["identity"], "checkpoints": checkpoints,
            "selected_continuous_policy": selected, "selection_optimism": True, "per_seed": by_seed,
            "contrasts_positive_favors_hybrid": contrasts,
            "readiness": "matched_systems_complete" if selected and usable_hybrid else "readiness_or_precision_insufficient",
            "model_quality_is_not_a_readiness_gate": True, "capacity": plan["capacity"],
            "cost": {"completed_fit_wall_seconds": total_fit, "label_wall_seconds": total_labels,
                     "scope": "all six completed fits and controllers/labels; interrupted work since last atomic checkpoint not recoverable; smoke separate"},
            "claim_boundary": plan["claim_boundary"], "compute": plan["compute"],
            "archive_handoff": {"source_revision": plan["source_revision"], "source_snapshot": str(args.output/"plan.json"),
                                "archived_release": False, "required": "commit/push source with permission; archive output plus referenced #70/#71 data separately"},
            "fresh_evaluation_opened": False, "final_evaluation_opened": False, "issue_64_authorized": False}


def select_continuous(by_seed):
    eligible = [p for p in POLICIES[:4] if not any(s["policies"][p]["prediction_failure_states"] for s in by_seed.values())]
    return min(eligible, key=lambda p: (np.mean([s["policies"][p]["mean_regret"] for s in by_seed.values()]),
                                        POLICIES.index(p))) if eligible else None


def smoke(args):
    plan = make_plan(args); source = old.load_plan(source_args(args))
    plan["training"]["steps"] = 18
    plan["controller"]["steps_per_round"] = 6
    began = time.monotonic(); result = {"production_evidence": False, "cells": {}}
    with tempfile.TemporaryDirectory(prefix="novphy-issue74-smoke-") as temporary:
        test = copy.copy(args); test.output = Path(temporary)
        smoke_models, smoke_controls, scored = {}, {}, {}
        # Real shards/full-sized architectures; reduced temporary training budget only.
        for arm in ARMS:
            train_cell(test, plan, source, SEEDS[0], arm, stop_after=9)
            train_cell(test, plan, source, SEEDS[0], arm, stop_after=18)
            saved = torch.load(cell(test, SEEDS[0], arm)/"predictor-progress.pt", weights_only=True, map_location="cpu")
            model = new_model(plan, arm).to(args.device); model.load_state_dict(saved["model"], strict=True)
            train_control(test, plan, source, SEEDS[0], arm, indices=(5,))
            train_control(test, plan, source, SEEDS[0], arm, indices=(5,))
            control, _ = load_control(test, plan, SEEDS[0], arm)
            smoke_models[arm], smoke_controls[arm] = model, control
            gp = grid.load_plan(grid_args(args)); row = grid.endpoint(grid_args(args), gp, 3)
            adapter = old.repair.load_repaired_adapter(args.parser, args.device)
            context, perception = grid.perceive(grid_args(args), gp, 3, row, adapter)
            objective = TaskObjective.from_vocabulary(plan["contract"]["vocabulary"])
            outcomes = {}
            for policy in POLICIES:
                owner, fixed = policy_parts(policy)
                if owner == arm:
                    outcomes[policy] = []
                    for j, candidate in enumerate(row["candidates"], 1):
                        outcome = predict(model, control, context, candidate["action"], fixed, objective)
                        outcomes[policy].append({"candidate_identity": candidate["identity"], "action": candidate["action"], **outcome})
                        log(f"smoke state=4 policy={policy} candidate={j}/12 complete")
            scored.update(outcomes)
            data = shard(args, source, 1)
            audit = intervention_audit(model, data["z"][:1, 0].to(args.device), data["action"][:1].to(args.device)) if arm == "hybrid" else None
            names = set(dict(model.named_parameters()))
            if set(saved["gradient_parameters"]) != names or set(saved["changed_parameters"]) != names:
                raise ValueError("smoke found untrained parameter tensors")
            if audit is not None and not audit["passed"]:
                raise ValueError("hybrid mode/horizon intervention failed")
            result["cells"][arm] = {"resume_step": saved["step"], "gradient_tensors": len(names),
                                      "policies": outcomes, "perception": perception, "intervention": audit}
        record = {"plan_identity": plan["identity"], "seed": SEEDS[0], "state": row["state"],
                  "perception": perception, "policies": scored}
        check_state(record, row, plan, SEEDS[0], smoke_models, smoke_controls)
    result.update({"wall_seconds": time.monotonic()-began, "memory": memory(args.device), "capacity": plan["capacity"]})
    write(args.output/"smoke.json", result)
    log(f"real smoke complete wall={result['wall_seconds']:.1f}s memory={result['memory']}; temporary weights removed, no production training")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "smoke-test", "prepare", "train", "score-development", "validate"):
        modes.add_argument("--"+mode, action="store_true")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--issue71", type=Path, default=old.ROOT)
    parser.add_argument("--parser", type=Path, default=old.repair.ROOT)
    args = parser.parse_args(); torch.set_num_threads(2)
    try:
        if args.dry_run:
            plan = make_plan(args)
            log(f"dry run: seeds={plan['seeds']} six new fits; reuse 3000 shards, no captures; capacity={plan['capacity']}")
            log(f"estimates (not measured full run)={plan['cost_estimates_not_measurements']}; no files written"); return 0
        if args.smoke_test:
            smoke(args); return 0
        if args.prepare:
            if (args.output/"plan.json").exists(): load_plan(args)
            else: write(args.output/"plan.json", make_plan(args))
            log("plan frozen; no corpus copy, no training or calibration scoring started"); return 0
        plan = load_plan(args)
        if args.train:
            source = old.load_plan(source_args(args))
            for seed in SEEDS:
                for arm in ARMS:
                    log(f"training cell seed={seed} arm={arm} start")
                    train_cell(args, plan, source, seed, arm)
                    train_control(args, plan, source, seed, arm)
            log(f"six matched fits/controllers complete memory={memory(args.device)}")
        elif args.score_development:
            score(args, plan); result = publication(args, plan); write(args.output/"readiness.json", result)
            log(f"development complete readiness={result['readiness']} selected_continuous={result['selected_continuous_policy']}; no #64 authorization")
        else:
            if publication(args, plan) != read(args.output/"readiness.json"):
                raise ValueError("readiness publication differs from source records")
            log("exact saved-evidence validation passed; neural inference/wall timings not rerun; no #64 authorization")
        return 0
    except (ValueError, OSError) as error:
        log(f"error: {error}"); return 1


if __name__ == "__main__":
    raise SystemExit(main())
