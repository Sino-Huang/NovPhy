"""Independent CPU reconstruction of fixed-development curves and field metrics."""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np
import torch

from scripts import run_issue_76_fixed_development as run
from world_model.training.native_history_data import VISUAL_DIM, VOCABULARY

TIMES = (750, 1500, 3000, 6000, 11250)
SOURCES = ("scripts/validate_issue_76_fixed_development.py", "tests/test_issue_76_fixed_development_validation.py")


def metrics(prediction, target, presence, centers, failed=False):
    """Independent NumPy formulas; do not call the production metric helper."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    actual_presence = np.asarray(presence, dtype=np.float64)
    actual_centers = np.asarray(centers, dtype=np.float64)
    if failed or not np.isfinite(prediction).all():
        return {"failure": "nonfinite_prediction_or_metric", "carrier_mse": 1e9}
    squared = np.square(prediction - target)
    slots = prediction[2:VISUAL_DIM].reshape(len(VOCABULARY), 13)
    present = actual_presence.astype(bool)
    values = {"carrier_mse": float(squared.mean()), "visual_mse": float(squared[:VISUAL_DIM].mean()),
        "memory_mse": float(squared[VISUAL_DIM:].mean()),
        "presence_mse_to_engine": float(np.square(slots[:, 0] - actual_presence).mean()),
        "center_mse_to_engine": float(np.square(slots[present, 5:7] - actual_centers[present]).mean()) if present.any() else None,
        "carrier_absolute_bound_excess": float(np.maximum(np.abs(prediction) - 2., 0.).max())}
    for kind in ("pig", "block"):
        indices = [i for i, name in enumerate(VOCABULARY) if name.split(":")[0] == kind]
        values[kind + "_count_absolute_error"] = float(abs(np.clip(slots[indices, 0], 0, 1).sum() - actual_presence[indices].sum()))
    if any(not math.isfinite(value) or value > np.finfo(np.float32).max for value in values.values() if value is not None):
        return {"failure": "nonfinite_prediction_or_metric", "carrier_mse": 1e9}
    return {**values, "failure": None}


def compare_metrics(actual, expected, spec):
    if set(actual) != set(expected):
        raise ValueError("field metric or failure inventory differs")
    for key, value in expected.items():
        if value is None or isinstance(value, str):
            if actual[key] != value:
                raise ValueError(f"metric availability/failure differs: {key}")
        elif not math.isclose(actual[key], value, rel_tol=spec["metric_relative_tolerance"], abs_tol=spec["metric_absolute_tolerance"]):
            raise ValueError(f"independent metric differs: {key}: stored={actual[key]} CPU={value}")


def compare_tensor(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def validate_prepared(plan, role, seed, members, check):
    """Rebuild exact first-shot availability directly, without prepare_member."""
    inventory = plan["inventory"]
    source_plan = run.fit.files.read(Path(inventory["data_plan_path"]))
    entries = [entry for entry in inventory["entries"] if entry["exposure_role"] == role]
    if [member["entry"] for member in members] != entries:
        raise ValueError("prepared member order differs")
    for entry, member in zip(entries, members, strict=True):
        check()
        if not entry["usable"]:
            if member["initial"] is not None or member["targets"] or member["contexts"]:
                raise ValueError("unusable assignment was imputed")
            continue
        shard = run.fit.load_shard(Path(inventory["data_root"]), entry, source_plan)
        carrier = run.fit.load_carrier(Path(inventory["data_root"]), seed, entry, source_plan)
        shot = shard["segment_ranges"][0]
        start, stop = shot["start"], shot["stop"]
        steps = shard["tensors"]["fixed_steps"].tolist()
        native = steps[start:stop]
        if not native or any(a >= b for a, b in zip(native, native[1:])):
            raise ValueError("source first-shot observation ordering differs")
        lookup = {step - native[0]: start + i for i, step in enumerate(native)}
        compare_tensor(member["initial"], carrier[start])
        compare_tensor(member["action"], shot["action"])
        if set(member["targets"]) != {time for time in TIMES if time in lookup}:
            raise ValueError("exact target availability differs")
        for elapsed, target in member["targets"].items():
            index = lookup[elapsed]
            compare_tensor(target["carrier"], carrier[index])
            for key in ("presence", "centers"):
                compare_tensor(target[key], shard["tensors"][key][index])
        if set(member["contexts"]) != {50, 250, 750}:
            raise ValueError("local horizon inventory differs")
        for horizon in (50, 250, 750):
            expected = {time for time in TIMES if time in lookup and time - horizon in lookup}
            if set(member["contexts"][horizon]) != expected:
                raise ValueError("observed local-context availability differs")
            for elapsed in expected:
                compare_tensor(member["contexts"][horizon][elapsed], carrier[lookup[elapsed - horizon]])


def target_metrics(predicted, target, failed=False):
    return metrics(predicted.numpy(), target["carrier"].numpy(), target["presence"].numpy(), target["centers"].numpy(), failed)


@torch.no_grad()
def validate_cell(plan, cell, result, members, model, check):
    if result["cell"] != cell or result["execution_plan_identity"] != plan["identity"]:
        raise ValueError("scored cell or execution source differs")
    if len(result["rows"]) != len(members):
        raise ValueError("assigned denominator differs")
    policies = [run.fit.policy_name(pair) for pair in model.pairs]
    spec = plan["independent_validation"]
    active = []
    for i, (row, member) in enumerate(zip(result["rows"], members, strict=True)):
        entry = member["entry"]
        for key in ("member_identity", "base_cluster", "family", "exposure_role"):
            if row[key] != entry[key]:
                raise ValueError("result member identity/role differs")
        if result["role"] != entry["exposure_role"] or any(row[key] != cell[key] for key in ("recipe", "seed", "pure")):
            raise ValueError("result cell/role projection differs")
        if row["available"] != (11250 in member["targets"]):
            raise ValueError("endpoint availability differs")
        if not row["available"] and row["unavailable_reason"] != member["unavailable_reason"]:
            raise ValueError("missing endpoint reason differs")
        if list(row["policies"]) != (policies if member["initial"] is not None else []):
            raise ValueError("fixed-policy inventory differs")
        if member["initial"] is not None:
            active.append(i)
            reference = row["unchanged_carrier_reference"]
            if reference["kind"] != "prediction_reference_not_gameplay_prior" or set(reference["curves"]) != set(map(str, member["targets"])):
                raise ValueError("unchanged reference inventory differs")
            for elapsed, target in member["targets"].items():
                compare_metrics(reference["curves"][str(elapsed)], target_metrics(member["initial"], target), spec)
    batch_index, curve_points = 0, 0
    for offset in range(0, len(active), plan["batch_size"]):
        indices = active[offset:offset + plan["batch_size"]]
        group = [members[i] for i in indices]
        initial = torch.stack([member["initial"] for member in group])
        actions = torch.stack([member["action"] for member in group])
        for pair in model.pairs:
            check()
            policy = run.fit.policy_name(pair)
            current = initial.clone()
            failed = ~torch.isfinite(current).all(-1) | ~torch.isfinite(actions).all(-1)
            policy_failed = failed.clone()
            local_calls = 0
            for i in indices:
                if set(result["rows"][i]["policies"][policy]["curves"]) != set(map(str, TIMES)):
                    raise ValueError("curve time inventory differs")
            for elapsed in range(pair.delta, 11251, pair.delta):
                current = model.carrier(current, actions, pair)
                failed |= ~torch.isfinite(current).all(-1)
                policy_failed |= failed
                if elapsed not in TIMES:
                    continue
                check()
                local_indices = [j for j, member in enumerate(group) if elapsed in member["contexts"][pair.delta]]
                local = {}
                if local_indices:
                    contexts = torch.stack([group[j]["contexts"][pair.delta][elapsed] for j in local_indices])
                    predictions = model.carrier(contexts, actions[local_indices], pair)
                    local = dict(zip(local_indices, predictions, strict=True))
                    local_calls += len(local_indices)
                for j, i in enumerate(indices):
                    member = group[j]
                    point = result["rows"][i]["policies"][policy]["curves"][str(elapsed)]
                    available = elapsed in member["targets"]
                    if point["available"] != available:
                        raise ValueError("curve target availability differs")
                    if available:
                        expected = target_metrics(current[j], member["targets"][elapsed], bool(failed[j]))
                        compare_metrics(point["recursive"], expected, spec)
                        policy_failed[j] |= expected["failure"] is not None
                        curve_points += 1
                    elif point["recursive"] is not None:
                        raise ValueError("missing curve target received a score")
                    if j in local:
                        expected = target_metrics(local[j], member["targets"][elapsed])
                        compare_metrics(point["observed_context_local"], expected, spec)
                        policy_failed[j] |= expected["failure"] is not None
                    elif point["observed_context_local"] is not None:
                        raise ValueError("missing observed context received a local prediction")
            for j, i in enumerate(indices):
                if result["rows"][i]["policies"][policy]["prediction_failure"] != bool(policy_failed[j]):
                    raise ValueError("cumulative per-policy prediction failure differs")
            batch = result["batches"][batch_index]
            expected = {"policy": policy, "member_identities": [member["entry"]["member_identity"] for member in group],
                        "batch_size": len(group), "recursive_transitions": len(group) * 11250 // pair.delta,
                        "local_transitions": local_calls, "controller_calls": 0, "batch_one_deployment_latency_measured": False}
            expected["linear_macs"] = (expected["recursive_transitions"] + local_calls) * run.scoring.linear_macs(model, pair)
            if any(batch[key] != value for key, value in expected.items()) or not math.isfinite(batch["wall_seconds"]) or batch["wall_seconds"] < 0:
                raise ValueError("batch work or timing record differs")
            batch_index += 1
    if batch_index != len(result["batches"]):
        raise ValueError("extra batch records")
    return {"cell": cell, "assigned_members": len(members), "curve_points_validated": curve_points,
            "batches_validated": batch_index, "all_fields_and_local_metrics_checked": True}


def validation_plan(plan):
    return {"identity": "issue-76-fixed-development-validation-v1", "execution_plan_identity": plan["identity"],
            "specification": plan["independent_validation"],
            "source_text": {path: (run.fit.files.ROOT / path).read_text() for path in SOURCES},
            "fresh_access": False, "optimizer_updates": 0}


def validate_role(plan, role):
    frozen = run.fit.files.read(run.ROOT / "validation-plan.json")
    if frozen != validation_plan(plan):
        raise ValueError("independent validation source differs from its freeze")
    run.check_role_access(plan, role)
    if run.phase_path(role, "validation.json").exists():
        print(f"Retaining completed {role} validation.", flush=True)
        return
    results = run.load_results(plan, role)
    summaries = []
    with run.fit.fitting_budget(run.ROOT, "independent-validation", plan["limits"]["independent_validation_wall_seconds_total"], "cpu") as budget:
        check = lambda: run.check_storage(plan, budget)
        prepared = {}
        for seed in plan["inventory"]["seeds"]:
            prepared[seed] = run.load_prepared(plan, role, seed)
            validate_prepared(plan, role, seed, prepared[seed], check)
        for index, (cell, result) in enumerate(zip(plan["inventory"]["models"], results, strict=True), 1):
            check()
            model, receipts = run.scoring.load_cell(plan["inventory"], cell, "cpu")
            if receipts != result["source_receipts"]:
                raise ValueError("checkpoint source/completion receipts differ")
            summaries.append(validate_cell(plan, cell, result, prepared[cell["seed"]], model, check))
            del model
            budget.save()
            print(f"Independent {role} validation {index}/54 active={budget.value['active_seconds']:.2f}s", flush=True)
        check()
    report = {"execution_plan_identity": plan["identity"], "validation_plan_identity": frozen["identity"],
              "role": role, "validated": True, "cells_validated": len(summaries),
              "all_assigned_members_checked": True, "summaries": summaries,
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "budget": run.fit.require_finished_budget(run.ROOT, "independent-validation")}
    run.immutable(run.phase_path(role, "validation.json"), report)
    run.immutable(run.OUTPUT / f"{role}-validation.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--validate-calibration", action="store_true")
    modes.add_argument("--validate-model-selection", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.prepare:
        value = validation_plan(run.load_plan())
        run.immutable(run.ROOT / "validation-plan.json", value)
        run.immutable(run.OUTPUT / "validation-plan.json", value)
        print("Independent validation source frozen; no development records read.", flush=True)
    elif args.validate_calibration or args.validate_model_selection:
        validate_role(run.load_plan(), "calibration" if args.validate_calibration else "model_selection")
    else:
        print("No-write: independent CPU reconstruction only; no fresh access.")


if __name__ == "__main__":
    main()
