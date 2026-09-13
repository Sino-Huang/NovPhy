"""Fixed-only development scoring; execution authorization lives in its driver.

No controller, optimizer, collection or fresh-data access is exposed here.
"""
from collections import defaultdict
import argparse
from pathlib import Path
from statistics import mean
import time

import torch

from scripts import run_issue_76_native_refit as fit
from world_model.training.cnn_hybrid import linear_macs
from world_model.training.fixed_development import (
    CURVE_TIMES, field_metrics, first_shot_indices, observed_context_prediction, recursive_curves,
)


def load_cell(inventory, cell, device):
    """Validate completed source checkpoints, then form only the frozen midpoint."""
    if cell not in inventory["models"]:
        raise ValueError("model is outside the frozen inventory")
    identities = (cell["source_plan_identities"] if cell["recipe"] == "fixed-midpoint"
                  else [cell["source_plan_identity"]])
    if cell["parameter_weights"] != ([.5, .5] if cell["recipe"] == "fixed-midpoint" else [1.]):
        raise ValueError("only the frozen single source or equal parameter midpoint is permitted")
    model = fit.new_predictor(inventory, cell["pure"], "cpu")
    combined, sources = None, []
    for name, identity, weight in zip(cell["checkpoint_paths"], identities, cell["parameter_weights"], strict=True):
        path = Path(name)
        root = path.parent.parent
        if path != fit.model_path(root, cell["seed"], cell["pure"], "predictor"):
            raise ValueError("checkpoint filename does not bind the declared arm and seed")
        plan = fit.files.read(root / "plan.json")
        if plan["identity"] != identity or plan["capacity"] != inventory["capacity"] or cell["seed"] not in plan["seeds"]:
            raise ValueError("checkpoint source plan, capacity or seed differs")
        binding = (plan["identity"] if cell["recipe"] == "repaired-representation" else
                   (plan["parent_plan"]["identity"] if cell["recipe"] == "matched-dynamics"
                    else plan["data_binding"]["identity"]))
        if binding != inventory["data_plan_identity"]:
            raise ValueError("checkpoint does not use the common repaired carrier source")
        budget = fit.require_finished_budget(root, path.stem)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        requested = plan["updates"]["predictor"]
        applied = plan.get("expected_applied", 5952)
        skipped = plan.get("expected_skipped", 48)
        if (saved["plan_identity"] != identity or not saved["complete"] or saved["failure"] is not None
                or saved["updates_requested"] != requested or saved["updates_completed"] != requested
                or saved["updates_applied"] != applied or saved["updates_skipped"] != skipped
                or applied + skipped != requested):
            raise ValueError("checkpoint is not a completed source-bound fit")
        model.load_state_dict(saved["model"], strict=True)
        if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("checkpoint contains nonfinite model state")
        if combined is None:
            combined = {key: value * weight for key, value in saved["model"].items()}
        else:
            for key, value in saved["model"].items():
                combined[key].add_(value, alpha=weight)
        sources.append({"checkpoint_path": name, "plan_identity": identity,
                        "updates_requested": requested, "updates_applied": applied,
                        "updates_skipped": skipped, "completed_budget": budget})
        del saved
    model.load_state_dict(combined, strict=True)
    return model.to(device).eval().requires_grad_(False), sources


def prepare_member(root, plan, seed, entry):
    """Compact CPU tensors only, retaining every unavailable assignment."""
    result = {"entry": entry, "initial": None, "targets": {}, "contexts": {}}
    if not entry["usable"]:
        result["unavailable_reason"] = "no_usable_observed_segment"
        return result
    shard = fit.load_shard(root, entry, plan)
    indices = first_shot_indices(shard)
    carrier = fit.load_carrier(root, seed, entry, plan)
    result.update(initial=carrier[indices["initial"]].clone(), action=shard["segment_ranges"][0]["action"].clone())
    for elapsed, index in indices["targets"].items():
        if index is not None:
            result["targets"][elapsed] = {"carrier": carrier[index].clone(),
                "presence": shard["tensors"]["presence"][index].clone(),
                "centers": shard["tensors"]["centers"][index].clone()}
    for horizon, times in indices["local_contexts"].items():
        result["contexts"][horizon] = {elapsed: carrier[index].clone() for elapsed, index in times.items()
                                        if index is not None and elapsed in result["targets"]}
    if CURVE_TIMES[-1] not in result["targets"]:
        result["unavailable_reason"] = "first_shot_has_no_exact_4_5_second_observed_endpoint"
    return result


def _metrics(predicted, failed, members, elapsed, device):
    targets = [member["targets"][elapsed] for member in members]
    return field_metrics(predicted,
        torch.stack([target["carrier"] for target in targets]).to(device),
        torch.stack([target["presence"] for target in targets]).to(device),
        torch.stack([target["centers"] for target in targets]).to(device), failed=failed)


@torch.no_grad()
def score_cell(model, cell, members, device, *, batch_size, check_resources):
    """Complete fixed-policy curves with batch work, never deployment latency."""
    rows, batches = [], []
    for member in members:
        entry = member["entry"]
        row = {key: entry[key] for key in ("member_identity", "base_cluster", "family", "exposure_role")}
        row.update(recipe=cell["recipe"], seed=cell["seed"], pure=cell["pure"],
                   available=CURVE_TIMES[-1] in member["targets"], policies={})
        if not row["available"]:
            row["unavailable_reason"] = member["unavailable_reason"]
        rows.append(row)
    active = [i for i, member in enumerate(members) if member["initial"] is not None]
    for offset in range(0, len(active), batch_size):
        indices = active[offset:offset + batch_size]
        group = [members[i] for i in indices]
        initial = torch.stack([member["initial"] for member in group]).to(device)
        action = torch.stack([member["action"] for member in group]).to(device)
        for pair in model.pairs:
            check_resources()
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            started = time.monotonic()
            curves = recursive_curves(model, initial, action, pair)
            policy = fit.policy_name(pair)
            records = [{"prediction_failure": False, "curves": {}} for _ in group]
            local_calls = 0
            for elapsed, curve in curves.items():
                for j, record in enumerate(records):
                    record["prediction_failure"] |= bool(curve["failed"][j])
                    record["curves"][str(elapsed)] = {"available": elapsed in group[j]["targets"],
                        "recursive": None, "observed_context_local": None}
                available = [j for j, member in enumerate(group) if elapsed in member["targets"]]
                if available:
                    metrics = _metrics(curve["predicted"][available], curve["failed"][available],
                                       [group[j] for j in available], elapsed, device)
                    for j, metric in zip(available, metrics, strict=True):
                        records[j]["curves"][str(elapsed)]["recursive"] = metric
                        records[j]["prediction_failure"] |= metric["failure"] is not None
                local = [j for j, member in enumerate(group) if elapsed in member["contexts"][pair.delta]]
                if local:
                    context = torch.stack([group[j]["contexts"][pair.delta][elapsed] for j in local]).to(device)
                    predicted = observed_context_prediction(model, context, action[local], pair)
                    local_calls += len(local)
                    metrics = _metrics(predicted["predicted"], predicted["failed"], [group[j] for j in local], elapsed, device)
                    for j, metric in zip(local, metrics, strict=True):
                        records[j]["curves"][str(elapsed)]["observed_context_local"] = metric
                        records[j]["prediction_failure"] |= metric["failure"] is not None
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            recursive_calls = len(group) * CURVE_TIMES[-1] // pair.delta
            batches.append({"policy": policy, "member_identities": [member["entry"]["member_identity"] for member in group],
                "batch_size": len(group), "wall_seconds": time.monotonic() - started,
                "recursive_transitions": recursive_calls, "local_transitions": local_calls,
                "linear_macs": (recursive_calls + local_calls) * linear_macs(model, pair),
                "controller_calls": 0, "batch_one_deployment_latency_measured": False})
            for i, record in zip(indices, records, strict=True):
                rows[i]["policies"][policy] = record
            check_resources()
        for j, i in enumerate(indices):
            rows[i]["unchanged_carrier_reference"] = {"kind": "prediction_reference_not_gameplay_prior", "curves": {}}
            for elapsed in members[i]["targets"]:
                metric = _metrics(initial[j:j+1], None, [members[i]], elapsed, device)[0]
                rows[i]["unchanged_carrier_reference"]["curves"][str(elapsed)] = metric
    return {"cell": cell, "rows": rows, "batches": batches}


def calibration_choice(inventory, results):
    """One recipe/pair per arm; equal family/seed weight and frozen-order ties."""
    entries = [entry for entry in inventory["entries"] if entry["exposure_role"] == "calibration"]
    if len(results) != len(inventory["models"]):
        raise ValueError("calibration requires the entire frozen model matrix")
    expected = [{key: entry[key] for key in ("member_identity", "base_cluster", "family", "exposure_role")}
                for entry in entries]
    availability = None
    by_cell = {}
    for cell, result in zip(inventory["models"], results, strict=True):
        if cell != result["cell"]:
            raise ValueError("calibration cells or their ordering differ from the freeze")
        actual = [{key: row[key] for key in expected[0]} for row in result["rows"]]
        if actual != expected:
            raise ValueError("calibration member inventory, role or ordering differs")
        available = [row["available"] for row in result["rows"]]
        if availability is not None and available != availability:
            raise ValueError("model-specific endpoint availability is forbidden")
        availability = available
        pairs = fit.CONTINUOUS_PAIRS if cell["pure"] else fit.PAIRS
        for row in result["rows"]:
            if row["policies"] and list(row["policies"]) != [fit.policy_name(pair) for pair in pairs]:
                raise ValueError("calibration fixed-policy matrix is incomplete")
            if row["available"] and not row["policies"]:
                raise ValueError("available calibration member has no predictions")
        by_cell[cell["recipe"], cell["seed"], cell["pure"]] = result["rows"]
    families = list(dict.fromkeys(entry["family"] for entry in entries))
    table, selected = [], {}
    for pure in (False, True):
        for recipe in inventory["recipes"]:
            for pair in (fit.CONTINUOUS_PAIRS if pure else fit.PAIRS):
                policy = fit.policy_name(pair)
                groups, failure = defaultdict(list), False
                for seed in inventory["seeds"]:
                    for row in by_cell[recipe, seed, pure]:
                        if row["policies"]:
                            failure |= row["policies"][policy]["prediction_failure"]
                        if row["available"]:
                            metric = row["policies"][policy]["curves"][str(CURVE_TIMES[-1])]["recursive"]
                            failure |= metric["failure"] is not None
                            groups[seed, row["family"]].append(metric["carrier_mse"])
                complete = all(groups[seed, family] for seed in inventory["seeds"] for family in families)
                score = mean(mean(groups[seed, family]) for seed in inventory["seeds"] for family in families) if complete else None
                record = dict(pure=pure, recipe=recipe, policy=policy, eligible=complete and not failure,
                              endpoint_carrier_mse=score, prediction_failure=failure)
                table.append(record)
                arm = "pure" if pure else "hybrid"
                if record["eligible"] and (arm not in selected or score < selected[arm]["endpoint_carrier_mse"]):
                    selected[arm] = record
    if set(selected) != {"hybrid", "pure"}:
        raise ValueError("no complete failure-free calibration selection for both arms")
    return {"selected": selected, "table": table, "model_selection_accessed": False,
            "selection_metric": "equal-family-equal-seed calibration endpoint carrier MSE",
            "gameplay_policy_selection": False, "advancement_authorized": False}


def synthetic_members(count, seed):
    """Synthetic-only shape/resource smoke; these identities have no research role."""
    from world_model.training.native_history_data import VOCABULARY
    from world_model.training.native_history_model import STATE_DIM
    generator = torch.Generator().manual_seed(seed)
    members = []
    for i in range(count):
        initial = torch.randn(STATE_DIM, generator=generator) * .1
        targets = {elapsed: {"carrier": torch.randn(STATE_DIM, generator=generator) * .1,
                            "presence": torch.zeros(len(VOCABULARY)),
                            "centers": torch.zeros(len(VOCABULARY), 2)} for elapsed in CURVE_TIMES}
        members.append({"entry": {"member_identity": f"synthetic-{i}", "base_cluster": f"synthetic-{i}",
                                   "family": "synthetic", "exposure_role": "synthetic-not-research"},
                        "initial": initial, "action": torch.zeros(5), "targets": targets,
                        "contexts": {h: {t: initial.clone() for t in CURVE_TIMES} for h in (50, 250, 750)}})
    return members


def smoke(device):
    """Frozen, bounded synthetic smoke. Does not authorize development scoring."""
    from scripts import prepare_issue_76_fixed_development as metadata
    root, output = metadata.ROOT, metadata.OUTPUT
    inventory = fit.files.read(root / "inventory.json")
    sources = (*fit.SOURCES, "scripts/score_issue_76_fixed_development.py",
               "scripts/prepare_issue_76_fixed_development.py", "world_model/training/fixed_development.py",
               "tests/test_fixed_development.py", "tests/test_issue_76_fixed_development_scoring.py",
               "docs/issue-76-fixed-development-protocol.md")
    plan = dict(identity="issue-76-fixed-development-synthetic-smoke-v1", inventory_identity=inventory["identity"],
        cells=inventory["models"][:2], synthetic_seed=760940001, synthetic_members=50, batch_size=50,
        device=device, threads=1, checkpoint_validation_wall_limit_seconds=300,
        scoring_wall_limit_seconds=60, rss_limit_mib=12288, cuda_allocated_limit_mib=8192,
        new_artifact_limit_bytes=2**30, development_records_read=0, fresh_access=False,
        score_execution_authorized=False, source_text={path: (fit.files.ROOT / path).read_text() for path in sources})
    metadata.source.immutable(root / "smoke-plan.json", plan)
    metadata.source.immutable(output / "smoke-plan.json", plan)
    if (root / "smoke-report.json").exists():
        print("Retaining completed source-bound synthetic smoke; development scoring remains unauthorized.")
        return
    torch.set_num_threads(plan["threads"])
    validated = []
    with fit.fitting_budget(root, "smoke-checkpoints", 300, "cpu") as budget:
        for index, cell in enumerate(inventory["models"], 1):
            budget.check()
            model, receipts = load_cell(inventory, cell, "cpu")
            validated.append(dict(cell=cell, sources=receipts, model_state_finite=True))
            del model
            budget.check()
            if index % 6 == 0:
                print(f"Checkpoint validation {index}/{len(inventory['models'])}; no development records read", flush=True)
    metadata.source.immutable(root / "smoke-checkpoints.json", validated)
    members = synthetic_members(plan["synthetic_members"], plan["synthetic_seed"])
    summaries = []
    if device == "cuda":
        torch.cuda.init()
    with fit.fitting_budget(root, "synthetic-smoke", 60, device) as budget:
        def check():
            budget.check()
            if sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) > plan["new_artifact_limit_bytes"]:
                raise fit.FitBudgetExceeded("fixed development derived artifact allowance exceeded")
        for cell in plan["cells"]:
            model, _ = load_cell(inventory, cell, device)
            result = score_cell(model, cell, members, device, batch_size=plan["batch_size"], check_resources=check)
            if len(result["rows"]) != len(members) or not all(row["available"] for row in result["rows"]):
                raise ValueError("synthetic smoke lost declared members or endpoints")
            if any(policy["prediction_failure"] for row in result["rows"] for policy in row["policies"].values()):
                raise ValueError("synthetic smoke contains prediction failures")
            summaries.append(dict(cell=cell, rows=len(result["rows"]), batches=result["batches"],
                                  policies_per_member=len(model.pairs), all_predictions_finite=True))
            print(f"Synthetic smoke {'pure' if cell['pure'] else 'hybrid'} complete; {len(model.pairs)} policies, batch 50", flush=True)
            del model, result
            check()
    report = dict(plan_identity=plan["identity"], summaries=summaries,
        checkpoint_instances_validated=len(validated), distinct_checkpoint_files=48,
        checkpoint_budget=fit.require_finished_budget(root, "smoke-checkpoints"),
        scoring_budget=fit.require_finished_budget(root, "synthetic-smoke"),
        timing_semantics="measured wall-time upper bound, not CPU time, CUDA kernel time or deployment latency",
        synthetic_only=True, development_records_read=0, fresh_access=False, optimization_performed=False,
        advancement_authorized=False, score_execution_authorized=False,
        failed_training_qualifications_preserved=True)
    metadata.source.immutable(root / "smoke-report.json", report)
    metadata.source.immutable(output / "smoke-report.json", report)
    print("Synthetic resource/shape smoke complete. No development scoring or advancement authorization.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.smoke:
        smoke(args.device)
    else:
        print("No-write: fixed-only scoring primitives. Only synthetic --smoke is exposed; development scoring is not authorized.")


if __name__ == "__main__":
    main()
