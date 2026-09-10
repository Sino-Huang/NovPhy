"""One frozen controller-window coverage pilot; no new dynamics or gameplay."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from scripts import run_issue_74_matched_dynamics as base
from world_model.model import Abstraction
from world_model.training.matched_dynamics import MatchedController, controller_labels

OUTPUT = base.ROOT / ".local-artifacts/issue-75-controller-coverage-v1"
SEEDS = base.SEEDS
ARMS = base.ARMS
INDICES = tuple(range(5, 3001, 5))
NEW = {"continuous": "covered_continuous", "hybrid": "covered_hybrid"}
POLICIES = (*base.POLICIES, *NEW.values())
read, write = base.read, base.write
PROTOCOL = "docs/issue-75-controller-coverage.md"
RUNNER_SOURCE = "scripts/run_issue_75_controller_coverage.py"
PUBLICATION_REPAIR = "publication-json-repair.json"


def log(message):
    print(f"[issue-75] {message}", flush=True)


def source_args(args):
    return argparse.Namespace(output=args.issue74, issue71=base.old.ROOT,
                              parser=base.old.repair.ROOT, device=args.device)


def protocol():
    return {"intervention": "all existing first AND later windows; controller only, both arms",
            "seeds": list(SEEDS), "controller_indices": list(INDICES), "states": 200,
            "steps_per_round": 1800, "batch_size": 128, "learning_rate": .001, "rounds": 2,
            "initialization": "same #74 seed+7400; independent reinitialization, no warm start",
            "training_seconds_max": 3600, "scoring_seconds_max": 3600,
            "cpu_mib_max": 3072, "cuda_mib_max": 2048, "artifact_bytes_max": 6*2**30,
            "minimum_informative": 20, "regret_margin": .02,
            "minimum_improved_seeds": 2, "second_axis_fraction_min": .05,
            "planning_seconds_max": 30., "symbol_intervention_min": 1e-7,
            "mean_recursive_mse_must_not_increase": True,
            "fresh_evaluation_opened": False, "final_evaluation_opened": False, "issue_64_authorized": False}


def make_plan(args):
    previous = base.load_plan(source_args(args))
    ready = read(args.issue74/"readiness.json")
    if ready["plan_identity"] != previous["identity"] or ready["readiness"] != "matched_systems_complete":
        raise ValueError("requires completed #74 matched readiness")
    files = (*base.FILES, "scripts/run_issue_75_controller_coverage.py", PROTOCOL)
    diagnosis = read(base.ROOT/".local-artifacts/issue-73-headroom-v1/result.json")
    return {"schema": "issue_75_coverage_plan_v1", "identity": "issue-75-controller-coverage-v1",
            "issue74": str(args.issue74.resolve()), "source_plan_identity": previous["identity"],
            "source_checkpoints": ready["checkpoints"], "source_readiness": ready,
            "contract": previous["contract"], "protocol": protocol(),
            "diagnosis": "#73 first-window pre-launch coverage; cause of poor task ranking not established",
            "diagnosis_binding": {k: diagnosis[k] for k in ("identity", "plan_identity", "schema")},
            "unchanged": ["all six predictors", "CNN", "carrier", "DP utility", "optimizer exposure",
                          "12 candidate actions", "225-step recursive endpoint", "settled task objective"],
            "claim_boundary": ready["claim_boundary"],
            "source_text": {p: (base.ROOT/p).read_text() for p in files},
            "archived_release": False, "fresh_evaluation_opened": False,
            "final_evaluation_opened": False, "issue_64_authorized": False}


def load_plan(args):
    plan = read(args.output/"plan.json")
    current = make_plan(args)
    receipt_path = args.output/PUBLICATION_REPAIR
    if plan != current and receipt_path.exists():
        receipt = read(receipt_path)
        if (receipt["plan_identity"] != plan["identity"]
                or receipt["original_source"] != plan["source_text"][RUNNER_SOURCE]
                or receipt["repaired_source"] != current["source_text"][RUNNER_SOURCE]):
            raise ValueError("publication repair does not bind this exact frozen/current source")
        current["source_text"][RUNNER_SOURCE] = plan["source_text"][RUNNER_SOURCE]
    if plan != current:
        raise ValueError("frozen pilot source/contract differs; preserve evidence, do not refreeze over it")
    return plan


def repair_publication(args):
    """Explicit technical source amendment; never rewrite the experimental freeze."""
    plan = read(args.output/"plan.json"); current = make_plan(args)
    if (args.output/PUBLICATION_REPAIR).exists():
        load_plan(args); log("existing publication JSON repair verified"); return
    repaired = current["source_text"][RUNNER_SOURCE]
    current["source_text"][RUNNER_SOURCE] = plan["source_text"][RUNNER_SOURCE]
    if plan != current:
        raise ValueError("publication repair cannot change protocol, inputs, checkpoints or other source files")
    write(args.output/PUBLICATION_REPAIR, {
        "identity": "issue-75-publication-json-repair-v1", "plan_identity": plan["identity"],
        "original_source": plan["source_text"][RUNNER_SOURCE], "repaired_source": repaired,
        "scope": "Python float for recursive-MSE comparison; JSON boolean type only, plus exact source-repair support",
        "plan_preserved": True, "training_and_scores_preserved": True,
        "thresholds_and_numerical_comparison_unchanged": True})
    load_plan(args)
    log("publication JSON repair recorded; original plan, training, scores and thresholds preserved")


def root(args, seed, arm):
    return args.output/f"seed-{seed}"/arm


def binding(plan, seed, arm):
    return {"plan_identity": plan["identity"], "contract": plan["contract"], "seed": seed,
            "arm": arm, "protocol": plan["protocol"], "source_plan_identity": plan["source_plan_identity"]}


class BudgetStop(ValueError):
    pass


class Budget:
    """Small persisted active-wall counter, checked at saved work boundaries."""
    def __init__(self, args, plan, stage):
        self.args, self.plan, self.stage = args, plan, stage
        self.path = args.output/f"budget-{stage}.json"
        self.value = read(self.path) if self.path.exists() else {
            "plan_identity": plan["identity"], "stage": stage, "active_seconds": 0., "stopped": False}
        if self.value["plan_identity"] != plan["identity"]:
            raise ValueError("resource counter belongs to a different plan")
        self.started = time.monotonic()
        if self.value["stopped"]:
            raise BudgetStop("frozen resource budget already stopped; publish/validate the stop")

    def tick(self, *, disk=False):
        now = time.monotonic(); self.value["active_seconds"] += now-self.started; self.started = now
        self.value.update(base.memory(self.args.device))
        p = self.plan["protocol"]
        checks = {"active_wall": self.value["active_seconds"] <= p[f"{self.stage}_seconds_max"],
                  "cpu_memory": self.value["peak_cpu_rss_mib"] <= p["cpu_mib_max"],
                  "cuda_memory": self.value["peak_cuda_allocated_mib"] <= p["cuda_mib_max"]}
        if disk:
            self.value["artifact_bytes"] = sum(f.stat().st_size for f in self.args.output.rglob("*") if f.is_file())
            checks["disk"] = self.value["artifact_bytes"] <= p["artifact_bytes_max"]
        self.value["stopped"] = not all(checks.values())
        self.value["checks"] = checks
        write(self.path, self.value)
        if self.value["stopped"]:
            raise BudgetStop(f"resource stop {checks}; preserve partial evidence, publish/validate")


def label_lineage(args, source, model, controller, index, aggregated):
    """Only the window membership differs from #74; same utility and aggregation."""
    sa = source_args(args)
    data = base.shard(sa, source, index, pure=True)
    metadata = torch.load(base.old.shard_path(base.source_args(sa), index), map_location="cpu", weights_only=True)
    records, coverage, seen = [], [], set()
    for position, window in enumerate(metadata["windows"]):
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
        coverage.append({**window, "rows": n, "first_window": window["shot"] not in seen})
        seen.add(window["shot"])
    return {"tensors": {k: torch.cat([r[k] for r in records]) for k in records[0]}, "coverage": coverage}


def validate_label(value, expected, source, index, round_index):
    base.check_binding(value, expected)
    if value["record"] != source["records"][index-1] or value["round"] != round_index:
        raise ValueError("controller label lineage/round differs")
    if sum(w["rows"] for w in value["coverage"]) != len(value["tensors"]["z"]):
        raise ValueError("coverage rows do not match labels")


def train_cell(args, plan, previous, source, seed, arm, budget, indices=INDICES, stop_after=None):
    directory = root(args, seed, arm); final = directory/"controller.pt"
    if final.exists():
        load_control(args, plan, seed, arm); log(f"seed={seed} arm={arm} controller reused"); return
    model, _ = base.load_predictor(source_args(args), previous, seed, arm)
    model.requires_grad_(False)
    torch.manual_seed(seed+7400); control = MatchedController(arm == "continuous").to(args.device)
    initial = {n: p.detach().cpu().clone() for n,p in control.named_parameters()}
    expected = binding(plan, seed, arm); recipe = plan["protocol"]; gradients = set()
    for r in (0, 1):
        checkpoint = directory/f"controller-round-{r}.pt"
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True); base.check_binding(saved, expected)
            control.load_state_dict(saved["model"], strict=True); gradients.update(saved["gradient_parameters"]); continue
        began = time.monotonic()
        for ordinal, index in enumerate(indices, 1):
            path = directory/f"labels-{r}"/f"lineage-{index:04d}.pt"
            if not path.exists():
                started = time.monotonic()
                record = label_lineage(args, source, model, control, index, bool(r))
                base.old.repair.atomic_torch(path, {"binding": expected, "record": source["records"][index-1],
                                                  "round": r, "wall_seconds": time.monotonic()-started, **record})
            else:
                validate_label(torch.load(path, map_location="cpu", weights_only=True), expected, source, index, r)
            budget.tick()
            elapsed = time.monotonic()-began
            log(f"labels seed={seed} arm={arm} round={r+1}/2 lineage={ordinal}/{len(indices)} elapsed={elapsed:.1f}s eta={elapsed/ordinal*(len(indices)-ordinal):.1f}s")
        optimizer = torch.optim.AdamW(control.parameters(), lr=recipe["learning_rate"])
        generator = torch.Generator().manual_seed(seed+7400+r)
        progress = directory/f"controller-progress-{r}.pt"; start, prior = 0, 0.
        if progress.exists():
            saved = torch.load(progress, map_location="cpu", weights_only=True); base.check_binding(saved, expected)
            control.load_state_dict(saved["model"], strict=True); optimizer.load_state_dict(saved["optimizer"])
            generator.set_state(saved["generator"]); start, prior = saved["step"], saved["wall_seconds"]
            gradients.update(saved["gradient_parameters"])
        began = time.monotonic()
        finish = min(recipe["steps_per_round"], stop_after or recipe["steps_per_round"])
        for step in range(start, finish):
            index = indices[step % len(indices)]
            rows = [torch.load(directory/f"labels-{j}"/f"lineage-{index:04d}.pt", map_location="cpu", weights_only=True)["tensors"] for j in range(r+1)]
            data = {k: torch.cat([item[k] for item in rows]) for k in rows[0]}
            batch = base.old.sample(data, recipe["batch_size"], generator, args.device)
            loss = F.cross_entropy(control(batch["z"], batch["action"], batch["remaining"]), batch["labels"])
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite covered-controller training loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradients.update(n for n,p in control.named_parameters() if p.grad is not None and bool((p.grad != 0).any()))
            optimizer.step()
            if (step+1) % 60 == 0 or step+1 == finish:
                elapsed = prior+time.monotonic()-began
                saved = {"binding": expected, "model": control.state_dict(), "optimizer": optimizer.state_dict(),
                         "generator": generator.get_state(), "round": r, "step": step+1,
                         "wall_seconds": elapsed, "gradient_parameters": sorted(gradients),
                         "changed_parameters": [n for n,p in control.named_parameters() if not torch.equal(p.detach().cpu(), initial[n])],
                         "optimizer_examples_this_round": (step+1)*recipe["batch_size"], "memory": base.memory(args.device)}
                base.old.repair.atomic_torch(progress, saved); budget.tick()
                log(f"train seed={seed} arm={arm} round={r+1}/2 step={step+1}/{recipe['steps_per_round']} loss={float(loss.detach()):.4f} elapsed={elapsed:.1f}s eta={elapsed/(step+1)*(recipe['steps_per_round']-step-1):.1f}s memory={saved['memory']}")
        if finish < recipe["steps_per_round"]:
            return  # bounded smoke/test pause only; no deployed checkpoint
        base.old.repair.atomic_torch(checkpoint, saved)
    base.old.repair.atomic_torch(final, saved); budget.tick(disk=True)


def load_control(args, plan, seed, arm):
    value = torch.load(root(args, seed, arm)/"controller.pt", map_location="cpu", weights_only=True)
    base.check_binding(value, binding(plan, seed, arm))
    controller = MatchedController(arm == "continuous").to(args.device)
    controller.load_state_dict(value["model"], strict=True)
    names = set(dict(controller.named_parameters()))
    if (value["round"] != 1 or value["step"] != plan["protocol"]["steps_per_round"]
            or set(value["gradient_parameters"]) != names or set(value["changed_parameters"]) != names):
        raise ValueError("incomplete covered controller or missing task gradients/updates")
    return controller.eval(), value


def source_state(args, index, seed):
    return read(args.issue74/f"seed-{seed}"/"development"/f"state-{index+1:04d}.json")


def check_state(record, original, row, plan, previous, seed, models, controls):
    if (record["plan_identity"] != plan["identity"] or record["seed"] != seed or record["state"] != row["state"]
            or set(record["policies"]) != set(NEW.values())
            or record["controller_bindings"] != {a: binding(plan, seed, a) for a in ARMS}):
        raise ValueError("pilot state/controller source differs")
    merged = {**original, "policies": dict(original["policies"])}
    for arm, policy in NEW.items():
        merged["policies"][f"{arm}_adaptive"] = record["policies"][policy]
    base.check_state(merged, row, previous, seed, models, controls)


def score_state(args, plan, previous, gp, seed, index, models, controls, adapter):
    sa = source_args(args); ga = base.grid_args(sa)
    row = base.grid.endpoint(ga, gp, index)
    context, perception = base.grid.perceive(ga, gp, index, row, adapter)
    objective = base.TaskObjective.from_vocabulary(plan["contract"]["vocabulary"])
    policies = {}
    for arm in ARMS:
        items = []
        for j,candidate in enumerate(row["candidates"], 1):
            began = time.monotonic()
            try:
                item = base.predict(models[arm], controls[arm], context, candidate["action"], None, objective)
            except ValueError as error:
                item = {"cost": None, "endpoint": None, "trace": [], "failure": str(error),
                        "wall_seconds": time.monotonic()-began}
            items.append({"candidate_identity": candidate["identity"], "action": candidate["action"], **item})
            log(f"score seed={seed} state={index+1}/200 arm={arm} candidate={j}/12 failure={item['failure']}")
        policies[NEW[arm]] = items
    record = {"plan_identity": plan["identity"], "seed": seed, "state": row["state"], "perception": perception,
              "policies": policies, "controller_bindings": {a: binding(plan, seed, a) for a in ARMS}}
    check_state(record, source_state(args, index, seed), row, plan, previous, seed, models, controls)
    return record


def score(args, plan, previous):
    budget = Budget(args, plan, "scoring"); sa = source_args(args)
    gp = base.grid.load_plan(base.grid_args(sa)); adapter = base.old.repair.load_repaired_adapter(sa.parser, args.device)
    for seed in SEEDS:
        models = {a: base.load_predictor(sa, previous, seed, a)[0] for a in ARMS}
        controls = {a: load_control(args, plan, seed, a)[0] for a in ARMS}
        began = time.monotonic()
        for index in range(200):
            path = args.output/f"seed-{seed}"/"states"/f"state-{index+1:04d}.json"
            if path.exists():
                row = base.grid.endpoint(base.grid_args(sa), gp, index)
                check_state(read(path), source_state(args, index, seed), row, plan, previous, seed, models, controls)
            else:
                write(path, score_state(args, plan, previous, gp, seed, index, models, controls, adapter))
            budget.tick(disk=(index+1) % 50 == 0)
            elapsed = time.monotonic()-began
            log(f"score seed={seed} state={index+1}/200 complete elapsed={elapsed:.1f}s eta={elapsed/(index+1)*(199-index):.1f}s")


def second_fraction(counts):
    values = sorted(counts.values(), reverse=True)
    return values[1]/sum(values) if len(values) > 1 else 0.


def decision(summaries, informative, symbol_passed, p):
    means = {policy: float(np.mean([s[policy]["mean_regret"] for s in summaries.values()])) for policy in POLICIES}
    eligible = [policy for policy in (*base.POLICIES[:4], "covered_continuous")
                if not any(s[policy]["prediction_failure_states"] for s in summaries.values())]
    selected = min(eligible, key=lambda key: (means[key], (*base.POLICIES[:4], "covered_continuous").index(key))) if eligible else None
    comparisons = ([means[selected]] if selected else []) + [means["hybrid_continuous_h15"], .304995]
    strongest = min(comparisons)
    modes, horizons = Counter(), Counter()
    for s in summaries.values():
        for key, count in s["covered_hybrid"]["pair_usage"].items():
            h, m = key.split(":"); horizons[h] += count; modes[m] += count
    old_mse = float(np.mean([s["hybrid_adaptive"]["mean_endpoint_carrier_mse"] for s in summaries.values()]))
    new_mse = [s["covered_hybrid"]["mean_endpoint_carrier_mse"] for s in summaries.values()]
    checks = {
        "complete_seed_state_inventory": len(summaries) == 3 and all(s["covered_hybrid"]["states"] == 200 for s in summaries.values()),
        "informative_states": informative >= p["minimum_informative"],
        "no_prediction_failures": not any(s[k]["prediction_failure_states"] for s in summaries.values() for k in NEW.values()),
        "improves_original_hybrid": means["hybrid_adaptive"]-means["covered_hybrid"] >= p["regret_margin"]-1e-12,
        "beats_strongest_comparator": selected is not None and strongest-means["covered_hybrid"] >= p["regret_margin"]-1e-12,
        "improves_two_seeds": sum(s["covered_hybrid"]["mean_regret"] < s["hybrid_adaptive"]["mean_regret"] for s in summaries.values()) >= p["minimum_improved_seeds"],
        "recursive_mse_not_worse": all(x is not None for x in new_mse) and float(np.mean(new_mse)) <= old_mse,
        "mode_use": second_fraction(modes) >= p["second_axis_fraction_min"],
        "horizon_use": second_fraction(horizons) >= p["second_axis_fraction_min"],
        "symbol_content_intervention": symbol_passed,
        "planning_wall": all(s[k]["max_perception_planning_seconds"] <= p["planning_seconds_max"] for s in summaries.values() for k in NEW.values()),
    }
    return {"checks": checks, "means": means, "selected_continuous_policy": selected,
            "disposition": "correction_supported_for_prospective_test" if all(checks.values()) else "not_supported_by_this_pilot",
            "fresh_evaluation_opened": False, "final_evaluation_opened": False, "issue_64_authorized": False}


def summarize(items, candidates, perception, objective):
    metric = base.grid.ranked([x["cost"] for x in items], candidates)
    errors, counts, usage, macs = [], [], Counter(), 0
    for item,target in zip(items, candidates, strict=True):
        for t in item["trace"]:
            usage[f"{t['horizon']}:{t['mode']}"] += 1; macs += t["linear_macs"]
        if item["endpoint"] is not None and target.get("horizon_carrier") is not None:
            z, truth = torch.tensor(item["endpoint"]), torch.tensor(target["horizon_carrier"])
            errors.append(float((z-truth).square().mean()))
            counts.append([abs(a-b) for a,b in zip(objective.counts(z), objective.counts(truth), strict=True)])
    return {**metric, "endpoint_mse": float(np.mean(errors)) if errors else None,
            "count_mae": np.mean(counts, axis=0).tolist() if counts else None,
            "pair_usage": dict(usage), "linear_macs": macs,
            "wall_seconds": perception["wall_seconds"]+sum(x["wall_seconds"] for x in items)}


def aggregate(rows):
    usage = Counter()
    for row in rows: usage.update(row["pair_usage"])
    errors = [r["endpoint_mse"] for r in rows if r["endpoint_mse"] is not None]
    counts = [r["count_mae"] for r in rows if r["count_mae"] is not None]
    return {"states": len(rows), "mean_regret": float(np.mean([r["regret"] for r in rows])),
            "top1_fraction": float(np.mean([r["top1"] for r in rows])),
            "top3_fraction": float(np.mean([r["top3"] for r in rows])),
            "state_regrets": [r["regret"] for r in rows],
            "prediction_failure_states": sum(r["prediction_failure"] for r in rows),
            "all_tied_states": sum(r["all_tied"] for r in rows), "pair_usage": dict(usage),
            "mean_endpoint_carrier_mse": float(np.mean(errors)) if errors else None,
            "parsed_count_mae": np.mean(counts, axis=0).tolist() if counts else None,
            "linear_macs": sum(r["linear_macs"] for r in rows),
            "mean_perception_planning_seconds": float(np.mean([r["wall_seconds"] for r in rows])),
            "max_perception_planning_seconds": max(r["wall_seconds"] for r in rows)}


def symbol_audit(args, previous, seed):
    sa = source_args(args); model, _ = base.load_predictor(sa, previous, seed, "hybrid")
    source = base.old.load_plan(base.source_args(sa)); data = base.shard(sa, source, 5, pure=True)
    z, action = data["z"][:1, 0].to(args.device), data["action"][:1].to(args.device)
    result = {}
    with torch.no_grad():
        for pair in (base.PAIRS[7], base.PAIRS[8]):
            actual = model.carrier(z, action, pair)
            neutral = torch.zeros_like(model.symbolic_features(z, pair.abstraction))
            with patch.object(model, "symbolic_features", return_value=neutral):
                ablated = model.carrier(z, action, pair)
            result[str(pair.identity)] = float((actual-ablated).abs().max())
    return result


def inventory(args):
    return {"controllers": sum((root(args,s,a)/"controller.pt").exists() for s in SEEDS for a in ARMS),
            "states": sum((args.output/f"seed-{s}"/"states"/f"state-{i:04d}.json").exists() for s in SEEDS for i in range(1,201))}


def publication(args, plan, previous):
    inv = inventory(args)
    budgets = {stage: read(args.output/f"budget-{stage}.json") for stage in ("training", "scoring")
               if (args.output/f"budget-{stage}.json").exists()}
    common = {"schema": "issue_75_coverage_result_v1", "plan_identity": plan["identity"], "inventory": inv,
              "budgets": budgets, "claim_boundary": plan["claim_boundary"], "archived_release": False,
              "fresh_evaluation_opened": False, "final_evaluation_opened": False, "issue_64_authorized": False}
    if any(b["plan_identity"] != plan["identity"] for b in budgets.values()):
        raise ValueError("budget records belong to another pilot")
    if (inv != {"controllers": 6, "states": 600} or set(budgets) != {"training", "scoring"}
            or any(b["stopped"] for b in budgets.values())):
        return {**common, "disposition": "budget_or_readiness_insufficient",
                "reason": "incomplete inventory or frozen resource stop; no favorable partial-data inference"}
    sa = source_args(args); gp = base.grid.load_plan(base.grid_args(sa)); source = base.old.load_plan(base.source_args(sa))
    objective = base.TaskObjective.from_vocabulary(plan["contract"]["vocabulary"])
    summaries, coverage, checkpoints, intervention = {}, {}, [], {}
    outcomes = {}; headroom = None
    for seed in SEEDS:
        models = {a: base.load_predictor(sa, previous, seed, a)[0] for a in ARMS}
        controls = {a: load_control(args, plan, seed, a)[0] for a in ARMS}
        for arm in ARMS:
            windows = Counter(); label_seconds = 0.
            for r in (0, 1):
                for index in INDICES:
                    value = torch.load(root(args,seed,arm)/f"labels-{r}"/f"lineage-{index:04d}.pt", weights_only=True, map_location="cpu")
                    validate_label(value, binding(plan,seed,arm), source,index,r)
                    label_seconds += value["wall_seconds"]
                    for w in value["coverage"]:
                        windows[f"round{r}_{'first' if w['first_window'] else 'later'}_windows"] += 1
                        windows[f"round{r}_rows"] += w["rows"]
                log(f"validate seed={seed} arm={arm} labels round={r+1}/2 complete")
            coverage[f"{seed}:{arm}"] = dict(windows)
            checkpoints.append({"seed": seed, "arm": arm, "controller": str(root(args,seed,arm)/"controller.pt"),
                                "predictor": str(base.cell(sa,seed,arm)/"predictor.pt"), "binding": binding(plan,seed,arm),
                                "label_seconds": label_seconds})
        if coverage[f"{seed}:continuous"] != coverage[f"{seed}:hybrid"]:
            raise ValueError("continuous/hybrid coverage is not symmetric")
        if not coverage[f"{seed}:hybrid"]["round0_later_windows"]:
            raise ValueError("later-window coverage was not applied")
        intervention[str(seed)] = symbol_audit(args,previous,seed)
        rows = {p: [] for p in POLICIES}; selected_outcomes = {p: Counter() for p in (*POLICIES,"prior_ordinal09")}; available = Counter()
        for index in range(200):
            row = base.grid.endpoint(base.grid_args(sa),gp,index); original = source_state(args,index,seed)
            record = read(args.output/f"seed-{seed}"/"states"/f"state-{index+1:04d}.json")
            check_state(record,original,row,plan,previous,seed,models,controls)
            metadata = [read(Path(gp["source_release"])/"candidate-results"/f"cal-s{index+1:04d}-c{j:02d}.json") for j in range(1,13)]
            for candidate, item in zip(row["candidates"],metadata,strict=True):
                if item["candidate_identity"] != candidate["identity"]:
                    raise ValueError("replay outcome source differs")
            for event in ("pig_removed", "level_clear", "collision:bird:pig", "collision:bird:block", "non_bird_support_change"):
                available[event] += any(event in item.get("interaction_coverage",[]) for item in metadata)
            available["informative"] += base.grid.normalized_regrets(row["candidates"])[1]
            selected_outcomes["prior_ordinal09"].update(metadata[8].get("interaction_coverage",[]))
            selected_outcomes["prior_ordinal09"]["realized_count_cost_sum"] += row["candidates"][8]["realized_count_cost"]
            for policy, items in {**original["policies"], **record["policies"]}.items():
                metric = summarize(items,row["candidates"], record["perception"] if policy in NEW.values() else original["perception"],objective)
                rows[policy].append(metric)
                selected = metric["selected"]
                if selected is not None and row["candidates"][selected]["accepted"]:
                    item = metadata[selected]
                    selected_outcomes[policy].update(item.get("interaction_coverage",[]))
                    selected_outcomes[policy]["realized_count_cost_sum"] += row["candidates"][selected]["realized_count_cost"]
                    selected_outcomes[policy]["accepted_selected"] += 1
                else:
                    selected_outcomes[policy]["failed_selection"] += 1
            if (index+1) % 10 == 0: log(f"validate seed={seed} state={index+1}/200")
        summaries[str(seed)] = {p: aggregate(rows[p]) for p in POLICIES}
        # Original controls are recomputed from saved action scores, never silently replaced.
        for p in base.POLICIES:
            if summaries[str(seed)][p]["state_regrets"] != plan["source_readiness"]["per_seed"][str(seed)]["policies"][p]["state_regrets"]:
                raise ValueError("original #74 control regrets changed")
        outcomes[str(seed)] = {p: dict(v) for p,v in selected_outcomes.items()}; headroom = dict(available)
    symbol_passed = all(v > plan["protocol"]["symbol_intervention_min"] for pairs in intervention.values() for v in pairs.values())
    result = decision(summaries,headroom["informative"],symbol_passed,plan["protocol"])
    contrasts = {}
    for new,reference in (("covered_hybrid","hybrid_adaptive"), ("covered_continuous","continuous_adaptive"),
                          ("covered_hybrid",result["selected_continuous_policy"]), ("covered_hybrid","hybrid_continuous_h15")):
        if reference is not None:
            values = np.mean([np.array(s[reference]["state_regrets"])-s[new]["state_regrets"] for s in summaries.values()],axis=0)
            contrasts[f"{new}_vs_{reference}"] = base.grid.paired_interval(values)
    return {**common, **result, "per_seed": summaries, "coverage": coverage, "checkpoints": checkpoints,
            "headroom_states": headroom, "selected_replay_outcomes": outcomes, "symbol_content_interventions": intervention,
            "paired_state_contrasts_positive_is_improvement": contrasts,
            "limitations": ["development reused, not fresh confirmation", "same200 states across seeds", "finite errors may be large",
                            "225-step prediction versus settled replay task cost", "60-step DP utility unchanged",
                            "common candidate budget, NOT equal FLOPs", "partial failed-prediction work unavailable",
                            "no new or multi-shot gameplay", "lost work since killed checkpoint not exactly recoverable"]}


def smoke(args):
    plan = make_plan(args); previous = base.load_plan(source_args(args))
    source = base.old.load_plan(base.source_args(source_args(args)))
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="novphy-issue75-smoke-") as temp:
        small = copy.copy(args); small.output = Path(temp)
        plan = copy.deepcopy(plan); plan["protocol"]["steps_per_round"] = 6
        models, controls, coverages = {}, {}, {}
        budget = Budget(small,plan,"training")
        for arm in ARMS:
            train_cell(small,plan,previous,source,SEEDS[0],arm,budget,indices=(5,10),stop_after=3)
            train_cell(small,plan,previous,source,SEEDS[0],arm,budget,indices=(5,10))
            models[arm] = base.load_predictor(source_args(args),previous,SEEDS[0],arm)[0]
            controls[arm] = load_control(small,plan,SEEDS[0],arm)[0]
            coverages[arm] = []
            for index in (5,10):
                value = torch.load(root(small,SEEDS[0],arm)/"labels-0"/f"lineage-{index:04d}.pt",weights_only=True,map_location="cpu")
                coverages[arm].extend(value["coverage"])
        if coverages["continuous"] != coverages["hybrid"] or not any(not x["first_window"] for x in coverages["hybrid"]):
            raise ValueError("smoke did not exercise symmetric later-window coverage")
        sa = source_args(args); gp = base.grid.load_plan(base.grid_args(sa))
        adapter = base.old.repair.load_repaired_adapter(sa.parser,args.device)
        scored = score_state(small,plan,previous,gp,SEEDS[0],3,models,controls,adapter)
        row = base.grid.endpoint(base.grid_args(sa),gp,3)
        objective = base.TaskObjective.from_vocabulary(plan["contract"]["vocabulary"])
        metrics = {p: summarize(items,row["candidates"],scored["perception"],objective) for p,items in scored["policies"].items()}
        # Exercise legitimate publication of an incomplete pilot; never infer a pass.
        stopped = publication(small,plan,previous)
        assert stopped["disposition"] == "budget_or_readiness_insufficient"
        write(small.output/"result.json",stopped)
        assert publication(small,plan,previous) == read(small.output/"result.json")
        result = {"production_evidence": False, "coverage": coverages, "state4_metrics": metrics,
                  "scored": scored, "wall_seconds": time.monotonic()-started, "memory": base.memory(args.device)}
        interventions = {str(seed): symbol_audit(args,previous,seed) for seed in SEEDS}
        if not all(x > plan["protocol"]["symbol_intervention_min"] for values in interventions.values() for x in values.values()):
            raise ValueError("frozen hybrid symbolic-content intervention did not change the transition")
        result["symbol_content_interventions"] = interventions
        result["wall_seconds"] = time.monotonic()-started
        result["memory"] = base.memory(args.device)
    write(args.output/"smoke.json",result)
    log(f"real smoke passed wall={result['wall_seconds']:.1f}s memory={result['memory']}; temporary weights removed, no full pilot")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run","smoke-test","prepare","train","score-development","publish","validate","repair-publication"):
        modes.add_argument("--"+mode,action="store_true")
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--issue74",type=Path,default=base.OUTPUT)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args = parser.parse_args(); torch.set_num_threads(2)
    try:
        if args.dry_run:
            p = make_plan(args)
            log(f"no-write dry-run passed: six controller fits, no predictors/captures; training cap={p['protocol']['training_seconds_max']}s scoring cap={p['protocol']['scoring_seconds_max']}s"); return 0
        if args.smoke_test: smoke(args); return 0
        if args.repair_publication: repair_publication(args); return 0
        if args.prepare:
            if (args.output/"plan.json").exists(): load_plan(args)
            else: write(args.output/"plan.json",make_plan(args))
            log("pilot plan frozen; no training/scoring started"); return 0
        plan = load_plan(args); previous = base.load_plan(source_args(args))
        if args.train:
            budget = Budget(args,plan,"training"); source = base.old.load_plan(base.source_args(source_args(args)))
            for seed in SEEDS:
                for arm in ARMS:
                    log(f"controller cell seed={seed} arm={arm} start")
                    train_cell(args,plan,previous,source,seed,arm,budget)
            log("six covered controllers complete; predictors unchanged")
        elif args.score_development:
            score(args,plan,previous); log("development scoring complete; run --publish then --validate")
        elif args.publish:
            result = publication(args,plan,previous); write(args.output/"result.json",result)
            log(f"published disposition={result['disposition']} checks={result.get('checks')}; no #64 authorization")
        else:
            if publication(args,plan,previous) != read(args.output/"result.json"):
                raise ValueError("publication differs from source-bound saved evidence")
            log("exact saved-evidence validation passed; no full neural rerun or #64 authorization")
        return 0
    except (ValueError,OSError) as error:
        log(f"error: {error}"); return 1


if __name__ == "__main__":
    raise SystemExit(main())
