"""Prospective, non-final corrected h15 advancement experiment (#69)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import multiprocessing
from pathlib import Path
import resource
import re
import sys
import time

import numpy as np
import torch

from scripts import run_issue_66_ranking_probe as probe
from scripts import run_issue_67_short_unroll as unroll
from world_model.model import Abstraction, PredictionPair
from world_model.training.action_ranking_probe import (
    BROAD_ACTION_DESIGN_ID, DISAGREEMENT_PENALTY_GRID, pessimistic_ensemble_cost,
)
from world_model.training.lineage_scaling import (
    CarrierKind, LineageScalingError, TrainingCell,
    load_action_ranking_bundle, load_carrier_lineage_bundle,
)
from world_model.training.short_unroll import (
    evaluate_recursive_carrier, load_short_unroll_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".local-artifacts/issue-69-corrected-experiment-v1"
RELEASE = ROOT / ".local-artifacts/issue-68-corrective-cohort-v2/release"
SUMMARY = ROOT / "data/runtime_evidence/issue-69/corrected-experiment-v1.json"
SEEDS = unroll.TRAINING_SEEDS
CORRECTIONS = tuple(item[0] for item in unroll.CONFIGURATIONS[1:])


def log(message):
    print(f"[issue-69] {message}", flush=True)


def read(path):
    return unroll._read_json(Path(path))


def write(path, value):
    """Resume only exactly matching immutable JSON; never replace evidence."""
    value = json.loads(json.dumps(value, allow_nan=False))
    if path.exists():
        if read(path) != value:
            raise LineageScalingError(f"existing issue-69 artifact differs: {path}")
    else:
        unroll._write_json(path, value)


def contract():
    return {
        "schema": "issue_69_contract_v1",
        "seeds": list(SEEDS),
        "matched_optimizer_examples": 8_000_000,
        "original_reference_optimizer_examples": 4_000_000,
        "original_reference_is_unmatched_external_reference": True,
        "candidate_design": BROAD_ACTION_DESIGN_ID,
        "states_per_role": 200, "candidates_per_state": 12,
        "horizon": 15, "ranking_recursive_steps": 15,
        "carrier_bound": 2.0, "prediction_clamping": False,
        "ranking_cost": "endpoint carrier MSE to frozen no-active-pig/block target",
        "ensemble_ranking": "mean member endpoint costs + lambda * population standard deviation",
        "ensemble_recursive": "mean carrier at each step, fed back to every member",
        "penalty_grid": list(DISAGREEMENT_PENALTY_GRID),
        "ranking_estimand": "per-state (selected cost - minimum) / (maximum - minimum); all ties=0; prediction failure=1; retained replay failure cost=1e9",
        "selection": "calibration minimum mean normalized regret, then h15 AUC, then system name; select exactly one before model selection",
        "recursive_relative_margin": 0.20,
        "ranking_absolute_margin": 0.05,
        "absolute_mean_bound_excess_max": 0.01,
        "absolute_step_mse_max": 1.0,
        "late_to_early_step_mse_ratio_max": 4.0,
        "physical_squared_bound_regression_margin": 0.0,
        "uncertainty_calibration_quantile": 0.95,
        "uncertainty_exceedance_fraction_max": 0.10,
        "uncertainty_regret_regression_margin": 0.0,
        "execution_and_nonfinite_failures_max": 0,
        "planning_seconds_per_state_max": 30.0,
        "planning_model_evaluations_per_state_max": 540,
        "compute_budget_scope": "prospective #69 ceiling for eligibility to #64 non-final pilot; #64 final matrix budget remains to be frozen",
        "uncertainty": "paired independent-state percentile bootstrap; one-sided Bonferroni across 7 frozen contrasts; seeds averaged within state, never treated as independent states",
        "alpha": 0.05, "contrast_count": 7,
        "bootstrap_replicates": 20_000, "bootstrap_seed": 20260906,
        "failed_run_treatment": "retain all states, prediction failure regret=1 and advancement veto; infrastructure interruption resumes same cell; no replacements",
        "physical_diagnostic_scope": "carrier range proxy, not an engine physical-plausibility claim",
        "h1_diagnostic_only": True,
        "final_evaluation_opened": False,
    }


def cell_inventory(plan):
    cells = []
    for seed in SEEDS:
        cells.append({"name": f"original-{seed}", "kind": "original", "seed": seed,
                      "checkpoint": str(probe._checkpoint_path(Path(plan["issue63"]), TrainingCell("full", CarrierKind.DEPLOYMENT, seed)))})
    for raw in plan["training_plan"]["configurations"]:
        spec = unroll._spec_from_payload(raw)
        cells.append({"name": f"{spec.name}-{spec.seed}", "kind": "single", "seed": spec.seed,
                      "spec": raw, "checkpoint": str(unroll._checkpoint_path(Path(plan["issue67"]), spec))})
    for name in CORRECTIONS:
        cells.append({"name": f"{name}-ensemble", "kind": "ensemble", "configuration": name,
                      "members": [item for item in cells if item.get("spec", {}).get("name") == name]})
    return cells


def prepare(args):
    training = read(args.issue67 / "plan.json")
    expected = unroll._specs(
        optimizer_example_budget=8_000_000, batch_size=512, learning_rate=1e-4,
        weight_decay=1e-4, grad_clip=1.0, carrier_bound=2.0,
        lineage_manifest_reference=training["training_bundle"],
    )
    if training["configurations"] != json.loads(json.dumps([asdict(s) for s in expected])):
        raise LineageScalingError("#67 training matrix differs from matched #69 design")
    # Collection manifests contain descriptive outcome counts. Do not inspect them
    # for candidate/threshold selection; the accepted release is bound by its plan.
    source_plan = read(args.release / "production-plan.json")
    if (source_plan["identity"] != "issue-68-production-plan-v2:states-per-role-200"
            or source_plan["action_design"]["identity"] != BROAD_ACTION_DESIGN_ID):
        raise LineageScalingError("#69 requires the exact #68 v2 production plan")
    plan = {"schema": "issue_69_plan_v1", "contract": contract(),
            "release": str(args.release), "issue63": str(args.issue63),
            "issue67": str(args.issue67), "training_plan": training,
            "source_plan_identity": source_plan["identity"], "final_evaluation_opened": False}
    plan["cells"] = cell_inventory(plan)
    for cell in plan["cells"]:
        if cell["kind"] != "ensemble" and not Path(cell["checkpoint"]).is_file():
            raise LineageScalingError(f"missing checkpoint: {cell['checkpoint']}")
    write(args.output / "plan.json", plan)
    log(f"prepared cells={len(plan['cells'])}; no model-selection bundles opened")


def load_plan(args):
    plan = read(args.output / "plan.json")
    if plan["contract"] != contract() or plan["cells"] != cell_inventory(plan):
        raise LineageScalingError("frozen #69 protocol differs; cannot resume")
    return plan


class MeanCarrier(torch.nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)

    def carrier(self, context, action, pair):
        return torch.stack([model.carrier(context, action, pair) for model in self.members]).mean(0)


def load_model(cell, device):
    if cell["kind"] == "ensemble":
        loaded = [load_model(member, device) for member in cell["members"]]
        return MeanCarrier([item[0] for item in loaded]), [item[1] for item in loaded]
    if cell["kind"] == "original":
        model, _, _ = probe._load_checkpoint_without_rehashing(
            Path(cell["checkpoint"]), TrainingCell("full", CarrierKind.DEPLOYMENT, cell["seed"]),
            expected_carrier_identity=unroll.TemporalVisualCarrierAdapter.identity,
            expected_config=unroll._predictor_config(), device=device)
        # Historical sidecars contain >1 GB recursive identities. Their numeric
        # wall-time field is at the end; don't parse/retain the expanded identity.
        sidecar = Path(cell["checkpoint"]).with_suffix(".training.json")
        with sidecar.open("rb") as stream:
            stream.seek(max(0, sidecar.stat().st_size - 4096))
            tail = stream.read().decode("utf-8")
        match = re.search(r'"wall_seconds":\s*([0-9.eE+-]+)', tail)
        if match is None:
            raise LineageScalingError("original training compute receipt missing")
        return model, {"checkpoint": cell["checkpoint"], "optimizer_examples": 4_000_000,
                       "wall_seconds": float(match.group(1)), "sidecar": str(sidecar),
                       "compute_reference": "immutable issue-63 checkpoint sidecar"}
    spec = unroll._spec_from_payload(cell["spec"])
    model, training = load_short_unroll_checkpoint(Path(cell["checkpoint"]), spec, device=device)
    if training.lineage_count != 3000:
        raise LineageScalingError("corrected checkpoint must use all 3000 training lineages")
    sidecar = read(Path(cell["checkpoint"]).with_suffix(".training.json"))
    if sidecar != unroll._training_payload(training):
        raise LineageScalingError("checkpoint training sidecar differs")
    return model, sidecar


def role_data(plan, role):
    if role not in ("calibration", "model_selection"):
        raise LineageScalingError("final role forbidden")
    root = Path(plan["release"])
    log(f"loading role={role} carrier/ranking bundles")
    lineages = load_carrier_lineage_bundle(root / "carrier-bundles" / f"{role}-deployment.pt")
    states = load_action_ranking_bundle(root / "ranking-bundles" / f"{role}-deployment.pt")
    if len(lineages) != 200 or len(states) != 200:
        raise LineageScalingError("fresh role requires all 200 states")
    by_lineage = {item.scenario_lineage_identity: item for item in lineages}
    source_plan = read(root / "production-plan.json")
    expected_states = [s["identity"] for s in source_plan["states"] if s["exposure_role"] == role]
    if [s.identity for s in states] != expected_states:
        raise LineageScalingError("role differs from the frozen #68 state inventory")
    ordered = []
    for state in states:
        lineage = by_lineage[state.scenario_lineage_identity]
        design = probe.broad_action_candidates(state.identity, state.action_bounds)
        if (state.exposure_role != role or lineage.exposure_role != role or not lineage.complete
                or state.carrier_identity != unroll.TemporalVisualCarrierAdapter.identity
                or tuple(c.interface_action for c in state.candidates) != tuple(c.action for c in design)):
            raise LineageScalingError("role isolation or frozen candidate inventory differs")
        ordered.append(lineage)
    if len(by_lineage) != 200 or len({s.scenario_lineage_identity for s in states}) != 200:
        raise LineageScalingError("independent-state inventory contains duplicates")
    return tuple(ordered), states


def score_state(model, lineage, state, *, ranking=True):
    device = next(model.parameters()).device
    result = {"state": state.identity, "lineage": state.scenario_lineage_identity,
              "candidate_ids": [c.identity for c in state.candidates],
              "realized_costs": [c.realized_cost for c in state.candidates]}
    for horizon in (1, 15):
        result[f"h{horizon}"] = asdict(evaluate_recursive_carrier(
            model, (lineage,), horizon=horizon, carrier_bound=2.0))
    local = {1: [], 15: []}
    squared_bounds = []
    failures = []
    model.eval()
    with torch.inference_mode():
        for t in lineage.transitions:
            try:
                pred = model.carrier(t.context.to(device)[None], t.action.to(device)[None],
                                     PredictionPair(t.horizon, Abstraction.CONTINUOUS))[0]
                error = float((pred - t.target.to(device)).square().mean())
                if not math.isfinite(error):
                    raise ValueError("nonfinite local prediction")
                local[t.horizon].append(error)
                squared_bounds.append(float(torch.relu(pred.abs() - 2.0).square().mean()))
            except (RuntimeError, ValueError) as error:
                failures.append(f"local:{type(error).__name__}:{error}")
    result["local_mse"] = {str(h): mean(v) if v else None for h, v in local.items()}
    diagnostics = {}
    for transition in lineage.transitions:
        for name, value in transition.physical_diagnostics.items():
            if type(value) in (bool, int, float) and math.isfinite(value):
                diagnostics.setdefault(name, []).append(float(value))
    result["target_physical_diagnostics"] = {name: mean(v) for name, v in diagnostics.items()}
    result["physical_squared_bound_excess"] = mean(squared_bounds) if squared_bounds else None
    result["local_failures"] = failures
    result["candidate_costs"] = []
    result["candidate_failures"] = []
    started = time.monotonic()
    if ranking:
        with torch.inference_mode():
            for ordinal, candidate in enumerate(state.candidates, 1):
                log(f"state={state.identity} candidate={ordinal}/{len(state.candidates)}")
                try:
                    pred = state.context.to(device)
                    for _ in range(15):
                        pred = model.carrier(pred[None], candidate.action.to(device)[None],
                                             PredictionPair(15, Abstraction.CONTINUOUS))[0]
                    cost = probe._ranking_cost(state, candidate, pred.cpu())
                    if not math.isfinite(cost):
                        raise ValueError("nonfinite ranking cost")
                    result["candidate_costs"].append(cost)
                except (RuntimeError, ValueError) as error:
                    result["candidate_costs"].append(None)
                    result["candidate_failures"].append(f"candidate={ordinal}:{type(error).__name__}:{error}")
    result["ranking_seconds"] = time.monotonic() - started
    result["ranking_model_evaluations"] = 180 if ranking else 0
    member_count = len(model.members) if isinstance(model, MeanCarrier) else 1
    result["total_model_evaluations"] = member_count * (
        result["h1"]["model_evaluations"] + result["h15"]["model_evaluations"]
        + len(lineage.transitions)) + result["ranking_model_evaluations"]
    return result


def worker(plan, cell, role, device, limit, connection):
    try:
        torch.set_num_threads(1)
        lineages, states = role_data(plan, role)
        if limit:
            lineages, states = lineages[:limit], states[:limit]
        log(f"loading model={cell['name']} device={device}")
        model, training = load_model(cell, device)
        rows = []
        started = time.monotonic()
        for i, (lineage, state) in enumerate(zip(lineages, states, strict=True), 1):
            log(f"role={role} model={cell['name']} state={i}/{len(states)} start")
            rows.append(score_state(model, lineage, state, ranking=cell["kind"] != "ensemble"))
            elapsed = time.monotonic() - started
            log(f"role={role} model={cell['name']} state={i}/{len(states)} complete elapsed={elapsed:.1f}s eta={elapsed/i*(len(states)-i):.1f}s")
        connection.send(("ok", {"schema": "issue_69_cell_score_v1", "cell": cell,
            "contract": plan["contract"], "source_plan_identity": plan["source_plan_identity"],
            "role": role, "rows": rows, "training": training, "wall_seconds": time.monotonic()-started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            "device": device, "final_evaluation_opened": False}))
    except Exception as error:
        connection.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def isolated_score(plan, cell, role, device, limit=None):
    ctx = multiprocessing.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    process = ctx.Process(target=worker, args=(plan, cell, role, device, limit, send))
    process.start()
    send.close()
    try:
        while not receive.poll(10):
            log(f"role={role} model={cell['name']} worker active")
            if not process.is_alive():
                raise LineageScalingError(f"score worker exited code={process.exitcode}")
        status, payload = receive.recv()
    finally:
        receive.close()
        process.join()
    if status != "ok" or process.exitcode != 0:
        raise LineageScalingError(f"model={cell['name']} failed: {payload}")
    log(f"model={cell['name']} complete peak_rss={payload['peak_rss_mib']:.1f}MiB; worker memory reclaimed")
    return payload


def score(args, role):
    plan = load_plan(args)
    if role == "model_selection":
        frozen = read(args.output / "frozen-decision.json")
        if frozen != freeze_payload(plan, load_scores(args, "calibration")):
            raise LineageScalingError("decision does not match calibration freeze")
        write(args.output / "model-selection-access.json", {
            "schema": "issue_69_access_v1", "frozen_decision": frozen,
            "role": role, "final_evaluation_opened": False})
    for i, cell in enumerate(plan["cells"], 1):
        path = args.output / role / f"{cell['name']}.json"
        log(f"role={role} cell={i}/{len(plan['cells'])} model={cell['name']}")
        if path.exists():
            validate_cell(read(path), cell, role)
            log("validated existing cell; resume")
        else:
            if role == "calibration" and (args.output / "model-selection-access.json").exists():
                raise LineageScalingError("cannot add calibration scores after model-selection access")
            write(path, isolated_score(plan, cell, role, args.device))
    log(f"score role={role} complete cells={len(plan['cells'])}")


def validate_cell(score, cell, role):
    if (score.get("schema") != "issue_69_cell_score_v1"
            or score.get("contract") != contract()
            or score.get("source_plan_identity") != "issue-68-production-plan-v2:states-per-role-200"
            or score["cell"] != cell or score["role"] != role or len(score["rows"]) != 200
            or score["final_evaluation_opened"] is not False
            or len({r["state"] for r in score["rows"]}) != 200
            or any(len(r["candidate_ids"]) != 12 for r in score["rows"])):
        raise LineageScalingError("score cell inventory differs")


def load_scores(args, role):
    plan = load_plan(args)
    result = {}
    inventory = None
    source = read(Path(plan["release"]) / "production-plan.json")
    expected_states = [s["identity"] for s in source["states"] if s["exposure_role"] == role]
    for i, cell in enumerate(plan["cells"], 1):
        log(f"validate role={role} cell={i}/{len(plan['cells'])} model={cell['name']}")
        value = read(args.output / role / f"{cell['name']}.json")
        validate_cell(value, cell, role)
        if [r["state"] for r in value["rows"]] != expected_states:
            raise LineageScalingError("saved score state inventory differs from frozen source")
        keys = [(r["state"], r["lineage"], r["candidate_ids"], r["realized_costs"]) for r in value["rows"]]
        if inventory is not None and keys != inventory:
            raise LineageScalingError("models did not score identical states and candidates")
        inventory = keys
        result[cell["name"]] = value
    return result


def mean(values):
    return math.fsum(values) / len(values)


def regret(costs, realized):
    if any(c is None or not math.isfinite(c) for c in costs):
        return 1.0, None
    selected = min(range(len(costs)), key=lambda i: costs[i])
    span = max(realized) - min(realized)
    return (0.0 if span == 0 else (realized[selected]-min(realized))/span), selected


def systems(scores, penalty):
    """All systems retain the same states, including tied and failed states."""
    result = {}
    for name, score in scores.items():
        if score["cell"]["kind"] == "ensemble":
            continue
        result[name] = {"rows": score["rows"], "members": [name], "penalty": 0.0,
                        "regrets": [regret(r["candidate_costs"], r["realized_costs"])[0] for r in score["rows"]],
                        "disagreement": [0.0] * len(score["rows"])}
    for configuration in CORRECTIONS:
        members = [f"{configuration}-{seed}" for seed in SEEDS]
        base_rows = scores[f"{configuration}-ensemble"]["rows"]
        for suffix, coefficient in (("mean", 0.0), ("pessimistic", penalty)):
            rows, regrets, deviations = [], [], []
            for i, base in enumerate(base_rows):
                source = [scores[m]["rows"][i] for m in members]
                costs, stds = [], []
                for j in range(len(base["candidate_ids"])):
                    values = tuple(r["candidate_costs"][j] for r in source)
                    if any(v is None for v in values):
                        costs.append(None)
                        stds.append(0.0)
                    else:
                        _, std, cost = pessimistic_ensemble_cost(values, coefficient)
                        costs.append(cost)
                        stds.append(std)
                r, selected = regret(costs, base["realized_costs"])
                regrets.append(r)
                deviations.append(0.0 if selected is None else stds[selected])
                rows.append({**base, "candidate_costs": costs, "candidate_disagreement": stds,
                             "candidate_failures": [f for r in source for f in r["candidate_failures"]],
                             "ranking_seconds": sum(r["ranking_seconds"] for r in source),
                             "ranking_model_evaluations": 540})
            result[f"{configuration}-{suffix}"] = {"rows": rows, "members": members,
                "penalty": coefficient, "regrets": regrets, "disagreement": deviations}
    return result


def metric(system, name):
    values = []
    for row in system["rows"]:
        h15 = row["h15"]
        if name == "auc":
            value = h15["error_auc"]
        elif name == "physical":
            value = row["physical_squared_bound_excess"]
        else:
            value = row[name]
        values.append(value)
    return values


def eligible_name(name):
    return any(name.startswith(c + "-") for c in CORRECTIONS)


def freeze_payload(plan, scores):
    choices = []
    for penalty in contract()["penalty_grid"]:
        for name, system in systems(scores, penalty).items():
            if eligible_name(name):
                auc = metric(system, "auc")
                choices.append((mean(system["regrets"]),
                    mean(auc) if all(v is not None for v in auc) else 1e30,
                    name, penalty if name.endswith("pessimistic") else 0.0))
    _, _, selected, penalty = min(choices)
    available = systems(scores, penalty)
    selected_system = available[selected]
    best_member = min(selected_system["members"], key=lambda m: (mean(available[m]["regrets"]), m))
    threshold = float(np.quantile(selected_system["disagreement"], 0.95))
    return {"schema": "issue_69_frozen_decision_v1", "contract": plan["contract"],
        "selected_system": selected, "penalty": penalty, "best_single_member": best_member,
        "uncertainty_threshold": threshold,
        "checkpoint_paths": [scores[m]["cell"]["checkpoint"] for m in selected_system["members"]],
        "calibration_selection_table": [{"regret": r, "auc": a, "system": s, "penalty": p}
                                        for r, a, s, p in sorted(set(choices))],
        "final_evaluation_opened": False}


def lower_bound(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        return None
    rng = np.random.default_rng(contract()["bootstrap_seed"])
    samples = values[rng.integers(0, len(values), size=(contract()["bootstrap_replicates"], len(values)))].mean(axis=1)
    return float(np.quantile(samples, contract()["alpha"] / contract()["contrast_count"]))


def report(scores, frozen):
    available = systems(scores, frozen["penalty"])
    selected = available[frozen["selected_system"]]
    n = len(selected["rows"])
    contrasts = {}
    for prefix in ("original", "teacher-forced-h15"):
        references = [available[f"{prefix}-{seed}"] for seed in SEEDS]
        aucs = [metric(s, "auc") for s in references]
        current = metric(selected, "auc")
        auc_difference = []
        for i in range(n):
            if current[i] is None or any(a[i] is None for a in aucs):
                auc_difference.append(float("nan"))
            else:
                reference = mean([a[i] for a in aucs])
                auc_difference.append((reference-current[i])/max(reference, 1e-12))
        contrasts[f"{prefix}:recursive_relative_improvement"] = {
            "lower_bound": lower_bound(auc_difference), "margin": 0.20}
        contrasts[f"{prefix}:regret_improvement"] = {"lower_bound": lower_bound([
            mean([s["regrets"][i] for s in references])-selected["regrets"][i] for i in range(n)]), "margin": 0.05}
    matched = [available[f"teacher-forced-h15-{seed}"] for seed in SEEDS]
    physical = metric(selected, "physical")
    contrasts["physical_nonregression"] = {"lower_bound": lower_bound([
        mean([metric(s, "physical")[i] or 0.0 for s in matched])-(physical[i] if physical[i] is not None else float("nan"))
        for i in range(n)]), "margin": 0.0}
    member = available[frozen["best_single_member"]]
    contrasts["uncertainty_regret_nonregression"] = {"lower_bound": lower_bound([
        a-b for a, b in zip(member["regrets"], selected["regrets"], strict=True)]), "margin": 0.0}
    # The outcome-independent no-model comparator rotates through all 12 actions.
    prior = []
    for i, row in enumerate(selected["rows"]):
        costs = [1.0] * len(row["realized_costs"])
        costs[i % len(costs)] = 0.0
        prior.append(regret(costs, row["realized_costs"])[0])
    contrasts["action_prior_regret_improvement"] = {"lower_bound": lower_bound([
        a-b for a, b in zip(prior, selected["regrets"], strict=True)]), "margin": 0.05}
    rows = selected["rows"]
    failures = sum(len(r["local_failures"])+len(r["candidate_failures"])
                   +len(r["h15"]["execution_failures"])+r["h15"]["nonfinite_failures"] for r in rows)
    curves = [r["h15"]["step_curve"] for r in rows]
    max_step = max((v["mean_mse"] for curve in curves for v in curve), default=1e30)
    late_ratios = []
    for curve in curves:
        split = max(1, len(curve)//2)
        early = [v["mean_mse"] for v in curve[:split]]
        late = [v["mean_mse"] for v in curve[split:]] or early
        late_ratios.append(mean(late)/max(mean(early), 1e-8) if early else 1e30)
    bound = [r["h15"]["mean_absolute_carrier_bound_excess"] for r in rows]
    gates = {name: item["lower_bound"] is not None and (
                 item["lower_bound"] > item["margin"] if item["margin"] > 0 else item["lower_bound"] >= 0)
             for name, item in contrasts.items()}
    gates.update({"no_failures": failures == 0,
        "absolute_bound": all(v is not None and v <= 0.01 for v in bound),
        "absolute_step_mse": max_step <= 1.0,
        "no_late_explosion": max(late_ratios) <= 4.0,
        "uncertainty_exceedance": mean([float(v > frozen["uncertainty_threshold"]) for v in selected["disagreement"]]) <= 0.1,
        "compute_seconds": max(r["ranking_seconds"] for r in rows) <= 30.0,
        "compute_evaluations": max(r["ranking_model_evaluations"] for r in rows) <= 540})
    supported = all(gates.values())
    table = []
    for name, system in available.items():
        auc = metric(system, "auc")
        table.append({"system": name, "normalized_regret": mean(system["regrets"]),
            "h15_auc": mean(auc) if all(v is not None for v in auc) else None,
            "mean_ranking_seconds": mean(metric(system, "ranking_seconds")),
            "selected_candidate_indices": [regret(r["candidate_costs"], r["realized_costs"])[1] for r in system["rows"]],
            "mean_selected_disagreement": mean(system["disagreement"])})
    seed_aggregates = []
    for prefix in ("original", "teacher-forced-h15", *CORRECTIONS):
        group = [r for r in table if r["system"] in {f"{prefix}-{s}" for s in SEEDS}]
        seed_aggregates.append({"configuration": prefix,
            "mean_h15_auc": mean([r["h15_auc"] for r in group]) if all(r["h15_auc"] is not None for r in group) else None,
            "mean_normalized_regret": mean([r["normalized_regret"] for r in group])})
    return {"schema": "issue_69_result_v1", "decision": "supported" if supported else "not_supported_by_this_experiment",
        "frozen_decision": frozen, "contrasts": contrasts, "gates": gates,
        "selected_configuration": frozen if supported else None,
        "issue_64_authorized": supported, "final_evaluation_opened": False,
        "table": table, "seed_aggregates": seed_aggregates,
        "no_model_prior_mean_regret": mean(prior),
        "state_count": n, "candidate_count": sum(len(r["candidate_ids"]) for r in rows),
        "all_tied_states": sum(len(set(r["realized_costs"])) == 1 for r in rows),
        "retained_replay_failures": sum(c >= 1e9 for r in rows for c in r["realized_costs"]),
        "selected_failures": failures, "maximum_step_mse": max_step,
        "maximum_late_ratio": max(late_ratios),
        "interpretation": "model/ranking advancement evidence only; no gameplay success claim"}


def publish(args, validate=False):
    plan = load_plan(args)
    calibration = load_scores(args, "calibration")
    frozen = freeze_payload(plan, calibration)
    if read(args.output / "frozen-decision.json") != frozen:
        raise LineageScalingError("frozen decision changed")
    access = read(args.output / "model-selection-access.json")
    if access["frozen_decision"] != frozen or access["final_evaluation_opened"] is not False:
        raise LineageScalingError("model-selection access was not bound to frozen decision")
    selection = load_scores(args, "model_selection")
    cal_ids = {r["lineage"] for r in next(iter(calibration.values()))["rows"]}
    ms_ids = {r["lineage"] for r in next(iter(selection.values()))["rows"]}
    if cal_ids & ms_ids:
        raise LineageScalingError("calibration/model-selection lineage leakage")
    result = report(selection, frozen)
    lines = ["# Issue 69 corrected experiment", "", f"Decision: `{result['decision']}`", "",
             "| System | h15 AUC | Normalized regret | Ranking seconds/state |", "|---|---:|---:|---:|"]
    lines += [f"| {r['system']} | {r['h15_auc']} | {r['normalized_regret']:.6f} | {r['mean_ranking_seconds']:.3f} |" for r in result["table"]]
    target = args.summary.with_suffix(".md")
    content = "\n".join(lines) + "\n"
    if validate:
        if read(args.output / "result.json") != result or read(args.summary) != result:
            raise LineageScalingError("published #69 result differs from exact recomputation")
        if target.read_text() != content:
            raise LineageScalingError("published table differs from exact recomputation")
        log(f"exact validation passed decision={result['decision']} states=200 candidates=2400 final_evaluation=unopened")
    else:
        write(args.output / "result.json", result)
        write(args.summary, result)
        if target.exists() and target.read_text() != content:
            raise LineageScalingError("published table differs")
        if not target.exists():
            target.write_text(content)
        log(f"published decision={result['decision']} summary={args.summary}")


def dry_run():
    torch.set_num_threads(1)
    lineage = unroll._fixture_lineages("calibration", count=1)[0]
    state = replace(probe._dry_states()[0], context=torch.zeros(4), cost_target=torch.zeros(4))
    scores = {}
    class Toy(torch.nn.Module):
        def __init__(self, seed):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(seed/10000.0))
        def carrier(self, context, action, pair):
            return context * 0.9 + action[:, :1] * 0.01 + self.bias
    for prefix in ("original", "teacher-forced-h15", *CORRECTIONS):
        for ordinal, seed in enumerate(SEEDS):
            name = f"{prefix}-{seed}"
            log(f"dry-run model={name}")
            scores[name] = {"cell": {"kind": "single", "checkpoint": f"no-write/{name}"},
                            "rows": [score_state(Toy(ordinal), lineage, state)]}
    for name in CORRECTIONS:
        scores[f"{name}-ensemble"] = {"cell": {"kind": "ensemble"},
            "rows": [score_state(MeanCarrier([Toy(i) for i in range(3)]), lineage, state, ranking=False)]}
    frozen = freeze_payload({"contract": contract()}, scores)
    result = report(scores, frozen)
    log(f"dry-run complete cells={len(scores)} systems={len(result['table'])} freeze/decision/ensemble paths exercised files_written=false final_evaluation=unopened")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "score-calibration", "freeze-decision", "score-model-selection", "publish", "validate", "dry-run", "smoke-test"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--release", type=Path, default=RELEASE)
    parser.add_argument("--issue63", type=Path, default=unroll.DEFAULT_ISSUE_63_OUTPUT)
    parser.add_argument("--issue67", type=Path, default=unroll.DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    for key in ("output", "release", "issue63", "issue67", "summary"):
        setattr(args, key, getattr(args, key).resolve())
    try:
        if args.dry_run:
            dry_run()
        elif args.prepare:
            prepare(args)
        elif args.score_calibration:
            score(args, "calibration")
        elif args.freeze_decision:
            value = freeze_payload(load_plan(args), load_scores(args, "calibration"))
            write(args.output / "frozen-decision.json", value)
            log(f"decision frozen candidate={value['selected_system']} lambda={value['penalty']}; model selection remains unopened")
        elif args.score_model_selection:
            score(args, "model_selection")
        elif args.smoke_test:
            plan = load_plan(args)
            for cell in plan["cells"]:
                isolated_score(plan, cell, "calibration", args.device, limit=1)
            log("real-checkpoint calibration smoke test passed; no score files written")
        else:
            publish(args, validate=args.validate)
    except (LineageScalingError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
