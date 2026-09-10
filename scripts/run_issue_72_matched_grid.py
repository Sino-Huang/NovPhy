"""Prospective #72 calibration screening; never authorizes fresh/final access."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import math
from pathlib import Path
import time

import numpy as np
import torch

from scripts import run_issue_71_hybrid_readiness as hybrid
from world_model.data.deployment_temporal import AgentObservation, TemporalObservationContext
from world_model.model import Abstraction, PredictionPair
from world_model.planning.gameplay import SlingshotAction
from world_model.planning.task_objective import TaskObjective
from world_model.training.cnn_hybrid import PAIRS, adaptive_rollout, linear_macs


ROOT = hybrid.repair.experiment.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-72-matched-grid-v1"
SCHEMA = "issue_72_development_v1"
ARMS = ("teacher_forced", "corrected_fixed", "adaptive")
RUNNER_SOURCE = "scripts/run_issue_72_matched_grid.py"
ANCHOR_REPAIR = "anchor-validation-repair.json"
read, write = hybrid.read, hybrid.write


def log(message):
    print(f"[issue-72] {message}", flush=True)


def contract():
    return {"schema": SCHEMA, "phase": "development", "role": "calibration",
            "state_count": 200, "actions_per_state": 12, "prediction_fixed_steps": 225,
            "objective": "1000 * expected remaining pigs + expected remaining blocks",
            "realized_target": "settled replay engine count cost; NOT an engine label at step 225",
            "recursive_target": "CNN carrier at offset 225, earlier stable terminal absorbing",
            "ranking": "accepted-cost range normalization; failed selections or all-failed states regret=1",
            "tie_rule": "cost then original candidate ordinal",
            "prior_selection": "minimum calibration mean regret among 12 fixed ordinals and uniform random expectation",
            "screen": {"minimum_informative_states": 20, "minimum_regret_improvement": .02,
                       "prediction_failures_max": 0, "planning_seconds_per_state_max": 30.,
                       "minimum_second_mode_fraction": .05, "minimum_second_horizon_fraction": .05},
            "uncertainty": {"bootstrap_draws": 10000, "seed": 7201,
                            "interpretation": "paired calibration bootstrap, descriptive only; includes baseline selection optimism"},
            "no_gameplay_in_this_phase": True, "fresh_evaluation_opened": False,
            "final_evaluation_opened": False, "issue_64_authorized": False}


def source_args(args):
    return argparse.Namespace(output=args.issue71, source=args.parser, device=args.device)


def candidate_actions(state):
    bounds = hybrid.repair.experiment.old.probe._bounds()
    return [asdict(c.action) for c in hybrid.repair.experiment.old.probe.broad_action_candidates(state, bounds)]


def make_plan(args):
    prior = hybrid.load_plan(source_args(args))
    ready = read(args.issue71 / "readiness.json")
    if ready["readiness_passed"] is not True or ready["contract"] != prior["contract"]:
        raise ValueError("validated CNN hybrid readiness is required")
    endpoints_plan = read(args.parser / "experiment/plan.json")
    states = endpoints_plan["states"]
    if (len(states) != 200 or len({s["identity"] for s in states}) != 200
            or any(s["exposure_role"] != "calibration" for s in states)):
        raise ValueError("requires the existing 200-state calibration inventory only")
    code = hybrid.source_receipt()
    code["source_text"]["scripts/run_issue_72_matched_grid.py"] = Path(__file__).read_text()
    return {"schema": SCHEMA, "identity": "issue-72-development-plan-v1", "contract": contract(),
            "issue71": str(args.issue71.resolve()), "parser": str(args.parser.resolve()),
            "hybrid_contract": prior["contract"], "readiness_identity": ready["identity"],
            "checkpoints": {name: value["checkpoint_identity"] for name, value in ready["predictors"].items()},
            "controller_identity": ready["controller_identity"], "states": states,
            "source_plan_identity": endpoints_plan["source_plan_identity"], "source_release": endpoints_plan["release"],
            "source_readiness_warning": ready["training_role_recursive_diagnostics"],
            "code": code, "full_issue_complete": False}


def load_plan(args):
    plan = read(args.output / "plan.json")
    if (plan["contract"] != contract() or plan["issue71"] != str(args.issue71.resolve())
            or plan["parser"] != str(args.parser.resolve())):
        raise ValueError("frozen development source/contract differs")
    for path, text in plan["code"]["source_text"].items():
        if (ROOT / path).read_text() != text:
            receipt_path = args.output / ANCHOR_REPAIR
            if path == RUNNER_SOURCE and receipt_path.exists():
                receipt = read(receipt_path)
                if (receipt["plan_identity"] == plan["identity"] and receipt["original_source"] == text
                        and receipt["repaired_source"] == (ROOT / path).read_text()):
                    continue
            raise ValueError(f"source changed since freeze: {path}; retain this run and use a new output")
    return plan


def endpoint(args, plan, index):
    state = plan["states"][index]
    row = read(args.parser / "experiment/endpoints" / f"state-{index+1:04d}.json")
    hybrid.repair.experiment.validate_endpoint(row, state)
    if row["parser_identity"] != plan["hybrid_contract"]["parser_identity"] or len(row["context"]) != 236:
        raise ValueError("endpoint parser/carrier differs")
    if [c["action"] for c in row["candidates"]] != candidate_actions(row["state"]):
        raise ValueError("candidate actions differ from the common frozen 12-action grid")
    if row["objective"] != asdict(TaskObjective.from_vocabulary(plan["hybrid_contract"]["vocabulary"])):
        # JSON converts tuples to lists.
        expected = TaskObjective.from_vocabulary(plan["hybrid_contract"]["vocabulary"])
        if row["objective"] != {"pig_slots": list(expected.pig_slots), "block_slots": list(expected.block_slots)}:
            raise ValueError("outcome-independent task objective differs")
    return row


def prepare(args):
    if (args.output / "plan.json").exists():
        load_plan(args); log("frozen development plan reused"); return
    log("checking #71 publication against checkpoints and exact source snapshot")
    if hybrid.readiness(source_args(args)) != read(args.issue71 / "readiness.json"):
        raise ValueError("#71 publication failed exact validation")
    write(args.output / "plan.json", make_plan(args))
    log("development frozen: 200 calibration states, three matched model arms, 12 actions; no fresh access")


def load_models(args, plan):
    old = hybrid.load_plan(source_args(args))
    teacher, t = hybrid.load_model(source_args(args), old, "teacher_forced")
    corrected, c = hybrid.load_model(source_args(args), old, "corrected")
    controller, control = hybrid.load_controller(source_args(args), old, c["identity"])
    if ({"teacher_forced": t["identity"], "corrected": c["identity"]} != plan["checkpoints"]
            or control["identity"] != plan["controller_identity"]):
        raise ValueError("deployed checkpoints differ from plan")
    return {"teacher_forced": teacher, "corrected_fixed": corrected, "adaptive": corrected}, controller


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def parse_anchor(adapter, observation, cached, device):
    synchronize(device); began = time.monotonic()
    context = TemporalObservationContext(None, observation)
    result = adapter.build(context).tensor
    synchronize(device)
    deployment_seconds = time.monotonic()-began
    began = time.monotonic()
    # Historical endpoint caches used B=3/5. A B=1 CUDA convolution can
    # differ materially, including crossing discrete carrier thresholds.
    # Validate with copies of ONLY the current image, never future frames.
    parsed = adapter.parse_batch((observation,)*3)[0]
    reproduced = adapter.build_from_parsed(context, parsed, None).tensor
    synchronize(device)
    if not torch.allclose(reproduced, cached, atol=1e-5, rtol=1e-5):
        raise ValueError("current CNN perception does not reproduce the frozen anchor")
    return result, {"cached_anchor_max_abs_difference": float((result-cached).abs().max()),
                    "anchor_validation_max_abs_difference": float((reproduced-cached).abs().max()),
                    "anchor_validation_batch_size": 3,
                    "anchor_validation_seconds": time.monotonic()-began,
                    "wall_seconds": deployment_seconds}


def perceive(args, plan, index, row, adapter):
    """One current agent observation; no realized outcome enters perception."""
    state = plan["states"][index]
    anchor = row["candidates"][state["carrier_anchor_candidate_ordinal"]-1]
    root = Path(plan["source_release"]) / anchor["source_trajectory"]
    trajectory = read(root / "trajectory.json")
    shot = trajectory["shots"][-1]
    obsroot = root / shot["path"] / "observation-trace"
    manifest = read(obsroot / "observation_trace_manifest.json")
    if manifest["exposure_role"] != "calibration" or manifest["identity"] != shot["observation_manifest_identity"]:
        raise ValueError("anchor observation provenance differs")
    f = manifest["frame_records"][0]; ref = f["agent_observation"]
    synchronize(args.device); began = time.monotonic()
    observation = AgentObservation(ref["identity"], f["fixed_step"], f["fixed_time_seconds"],
                                   (obsroot / ref["relative_path"]).read_bytes(), "agent")
    png_read_seconds = time.monotonic()-began
    result, audit = parse_anchor(adapter, observation, torch.tensor(row["context"]), args.device)
    audit["wall_seconds"] += png_read_seconds
    return result, {"observation_identity": observation.identity, "parser_calls": 1, **audit,
                    "scope": "agent PNG read, CNN, carrier construction; model load and manifest I/O excluded"}


def repair_anchor_validation(args):
    """Disclose this exact validation-only repair without rewriting the freeze."""
    old = read(args.output / "plan.json")
    current = make_plan(args)
    if {k:v for k,v in old.items() if k != "code"} != {k:v for k,v in current.items() if k != "code"}:
        raise ValueError("anchor repair cannot change experiment settings or source membership")
    for path, source in old["code"]["source_text"].items():
        if path != RUNNER_SOURCE and (ROOT/path).read_text() != source:
            raise ValueError(f"anchor repair cannot amend another component: {path}")
    target = args.output / ANCHOR_REPAIR
    if target.exists():
        load_plan(args); log("existing anchor-validation repair verified"); return
    models, _ = load_models(args, old)
    retained = []
    for index in range(200):
        path = args.output / "states" / f"state-{index+1:04d}.json"
        if path.exists():
            record = read(path)
            check_state(record, endpoint(args, old, index), old, models)
            retained.append({"path": str(path), "state": record["state"]})
    write(target, {"identity": "issue-72-anchor-validation-repair-v1", "plan_identity": old["identity"],
                   "original_source": old["code"]["source_text"][RUNNER_SOURCE],
                   "repaired_source": (ROOT/RUNNER_SOURCE).read_text(), "retained_states": retained,
                   "change": "same-image batched cache audit only; B=1 scoring, models, candidates and gates unchanged",
                   "original_plan_preserved": True, "final_evaluation_opened": False})
    log(f"anchor-validation repair recorded; preserved completed_states={len(retained)} and original plan")


def score_action(model, controller, context, action, *, adaptive, objective):
    """Deployment boundary: no candidate outcomes or target carriers accepted."""
    device = next(model.parameters()).device
    bounds = hybrid.repair.experiment.old.probe._bounds()
    action = SlingshotAction(**action)
    if not bounds.contains(action):
        raise ValueError("illegal grid action")
    a = torch.tensor([[action.drag_x/480., action.drag_y/480., bounds.release_time_ms/1000.,
                       action.tap_time_ms/1000., 1.]], device=device)
    synchronize(device); began = time.monotonic()
    predicted, trace = adaptive_rollout(model, controller, context[None].to(device), a,
                                       fixed_steps=225, fixed_pair=None if adaptive else PAIRS[6])
    predicted_cpu = predicted[0].cpu()
    cost = objective(predicted_cpu); synchronize(device)
    return {"cost": cost, "endpoint": predicted_cpu.tolist(), "trace": trace, "trace_complete": True,
            "wall_seconds": time.monotonic()-began, "failure": None}


def score_state(args, plan, index, models, controller, adapter):
    row = endpoint(args, plan, index)
    log(f"state={index+1}/200 perception start")
    context, perception = perceive(args, plan, index, row, adapter)
    objective = TaskObjective.from_vocabulary(plan["hybrid_contract"]["vocabulary"])
    arms = {}
    for arm in ARMS:
        results = []
        for j, candidate in enumerate(row["candidates"], 1):
            log(f"state={index+1}/200 arm={arm} candidate={j}/12 start")
            began = time.monotonic()
            try:
                result = score_action(models[arm], controller, context, candidate["action"],
                                      adaptive=arm == "adaptive", objective=objective)
            except (ValueError, RuntimeError) as error:
                # Infrastructure OOM must not be recorded as ordinary prediction failure.
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    raise
                result = {"cost": None, "endpoint": None, "trace": [], "trace_complete": False,
                          "wall_seconds": time.monotonic()-began,
                          "failure": f"{type(error).__name__}: {error}"}
            result["candidate_identity"] = candidate["identity"]
            result["action"] = candidate["action"]
            results.append(result)
            log(f"state={index+1}/200 arm={arm} candidate={j}/12 complete steps={len(result['trace'])} failure={result['failure']}")
        arms[arm] = results
    return {"schema": SCHEMA, "plan_identity": plan["identity"], "state": row["state"],
            "checkpoints": plan["checkpoints"], "controller_identity": plan["controller_identity"],
            "perception": perception, "arms": arms}


def normalized_regrets(candidates):
    costs = [c["realized_count_cost"] for c in candidates if c["accepted"]]
    if not costs:
        return [1.] * len(candidates), False
    low, high = min(costs), max(costs)
    return [1. if not c["accepted"] else (0. if high == low else (c["realized_count_cost"]-low)/(high-low))
            for c in candidates], high > low


def ranked(costs, candidates):
    regrets, informative = normalized_regrets(candidates)
    if any(c is None or not math.isfinite(c) for c in costs):
        return {"selected": None, "regret": 1., "top1": False, "top3": False, "prediction_failure": True,
                "informative": informative, "all_tied": False}
    order = sorted(range(len(costs)), key=lambda i: (costs[i], i))
    accepted = any(c["accepted"] for c in candidates)
    return {"selected": order[0], "regret": regrets[order[0]], "top1": accepted and regrets[order[0]] == 0,
            "top3": accepted and any(regrets[i] == 0 for i in order[:3]), "prediction_failure": False,
            "informative": informative, "all_tied": max(costs)-min(costs) <= 1e-8}


def check_state(record, row, plan, models):
    if (record["schema"] != SCHEMA or record["plan_identity"] != plan["identity"] or record["state"] != row["state"]
            or record["checkpoints"] != plan["checkpoints"] or record["controller_identity"] != plan["controller_identity"]
            or set(record["arms"]) != set(ARMS)):
        raise ValueError("scored state/checkpoint inventory differs")
    objective = TaskObjective.from_vocabulary(plan["hybrid_contract"]["vocabulary"])
    for arm in ARMS:
        if len(record["arms"][arm]) != 12:
            raise ValueError("missing candidate; do not publish partial scoring")
        for item, candidate in zip(record["arms"][arm], row["candidates"], strict=True):
            if item["candidate_identity"] != candidate["identity"] or item["action"] != candidate["action"]:
                raise ValueError("scored candidate order/action differs")
            if item["failure"]:
                if item["cost"] is not None:
                    raise ValueError("failed prediction has a cost")
                continue
            if objective(torch.tensor(item["endpoint"])) != item["cost"]:
                raise ValueError("stored cost differs from outcome-independent predicted endpoint")
            elapsed = 0
            for t in item["trace"]:
                pair = PredictionPair(t["requested_horizon"], Abstraction(t["mode"]))
                if (pair not in PAIRS or t["start_fixed_step"] != elapsed or t["effective_horizon"] != pair.delta
                        or t["linear_macs"] != linear_macs(models[arm], pair, arm == "adaptive")
                        or t["controller_calls"] != int(arm == "adaptive")
                        or t["transition_calls"] != 1 or t["symbol_decoder_calls"] != int(pair.abstraction != Abstraction.CONTINUOUS)
                        or (arm != "adaptive" and pair != PAIRS[6])):
                    raise ValueError("selected pair/endpoint/compute trace differs")
                elapsed += pair.delta
            if elapsed != 225:
                raise ValueError("different physical prediction endpoint")


def score(args):
    plan = load_plan(args); models, controller = load_models(args, plan)
    adapter = hybrid.repair.load_repaired_adapter(args.parser, args.device)
    began = time.monotonic()
    for index in range(200):
        path = args.output / "states" / f"state-{index+1:04d}.json"
        if path.exists():
            check_state(read(path), endpoint(args, plan, index), plan, models)
            log(f"state={index+1}/200 validated cached")
        else:
            record = score_state(args, plan, index, models, controller, adapter)
            check_state(record, endpoint(args, plan, index), plan, models)
            write(path, record)
        elapsed = time.monotonic()-began
        log(f"state={index+1}/200 complete elapsed={elapsed:.1f}s eta={elapsed/(index+1)*(199-index):.1f}s")


def paired_interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(7201)
    draws = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return {"mean": float(values.mean()), "descriptive_95_percent_interval": np.quantile(draws, [.025, .975]).tolist()}


def decision(metrics, usage, informative, worst_wall, failures):
    """Prospective screening only. Even a pass cannot authorize advancement."""
    screen = contract()["screen"]
    mode, horizon = Counter(), Counter()
    for pair, count in usage.items():
        h, m = pair.split(":"); mode[m] += count; horizon[h] += count
    def second_fraction(counts):
        values = sorted(counts.values(), reverse=True)
        return 0. if len(values) < 2 else values[1]/sum(values)
    strongest = min(("teacher_forced", "corrected_fixed", "prior"), key=lambda s: (metrics[s], s))
    checks = {"enough_informative_states": informative >= screen["minimum_informative_states"],
              "no_prediction_failures": failures == 0,
              "adaptive_beats_corrected_fixed": metrics["corrected_fixed"]-metrics["adaptive"] >= screen["minimum_regret_improvement"],
              "adaptive_beats_strongest_declared_baseline": metrics[strongest]-metrics["adaptive"] >= screen["minimum_regret_improvement"],
              "mode_use": second_fraction(mode) >= screen["minimum_second_mode_fraction"],
              "horizon_use": second_fraction(horizon) >= screen["minimum_second_horizon_fraction"],
              "planning_wall_limit": worst_wall <= screen["planning_seconds_per_state_max"]}
    passed = all(checks.values())
    return {"development_passed": passed, "checks": checks, "strongest_declared_baseline": strongest,
            "disposition": None if passed else "readiness_or_precision_insufficient",
            "status": "fresh_protocol_required" if passed else "stopped_at_development_screen",
            "fresh_evaluation_opened": False, "final_evaluation_opened": False, "issue_64_authorized": False,
            "fresh_gameplay_claim": False}


def report(args):
    plan = load_plan(args); models, _ = load_models(args, plan)
    results = {arm: [] for arm in ARMS}; prior_regrets = []; usage = Counter(); wall = {arm: [] for arm in ARMS}
    errors = {arm: [] for arm in ARMS}; count_errors = {arm: [] for arm in ARMS}
    macs = {arm: 0 for arm in ARMS}; details = []
    audit_seconds, audited_states = 0., 0
    objective = TaskObjective.from_vocabulary(plan["hybrid_contract"]["vocabulary"])
    for index in range(200):
        row = endpoint(args, plan, index)
        record = read(args.output / "states" / f"state-{index+1:04d}.json")
        check_state(record, row, plan, models)
        audit_seconds += record["perception"].get("anchor_validation_seconds", 0.)
        audited_states += int("anchor_validation_seconds" in record["perception"])
        regrets, _ = normalized_regrets(row["candidates"]); prior_regrets.append(regrets)
        state = {"state": row["state"], "arms": {}}
        for arm in ARMS:
            scored = record["arms"][arm]
            metric = ranked([x["cost"] for x in scored], row["candidates"])
            results[arm].append(metric); state["arms"][arm] = metric
            wall[arm].append(record["perception"]["wall_seconds"] + sum(x["wall_seconds"] for x in scored))
            for item, candidate in zip(scored, row["candidates"], strict=True):
                macs[arm] += sum(t["linear_macs"] for t in item["trace"])
                if arm == "adaptive":
                    usage.update(f"{t['requested_horizon']}:{t['mode']}" for t in item["trace"])
                if item["endpoint"] is not None and candidate.get("horizon_carrier") is not None:
                    errors[arm].append(float((torch.tensor(item["endpoint"])-torch.tensor(candidate["horizon_carrier"])).square().mean()))
                    predicted_counts = objective.counts(torch.tensor(item["endpoint"]))
                    parsed_counts = objective.counts(torch.tensor(candidate["horizon_carrier"]))
                    count_errors[arm].append([abs(a-b) for a,b in zip(predicted_counts, parsed_counts, strict=True)])
        details.append(state)
        if (index+1) % 10 == 0:
            log(f"publication validation state={index+1}/200")
    priors = np.array(prior_regrets)
    choices = {f"fixed_ordinal_{i+1:02d}": priors[:, i] for i in range(12)}
    choices["uniform_random_expected"] = priors.mean(1)
    selected_prior = min(choices, key=lambda k: (float(choices[k].mean()), k))
    values = {arm: [r["regret"] for r in results[arm]] for arm in ARMS}
    values["prior"] = choices[selected_prior].tolist()
    means = {k: float(np.mean(v)) for k,v in values.items()}
    summary = {arm: {"mean_regret": means[arm], "top1_fraction": float(np.mean([r['top1'] for r in results[arm]])),
                     "top3_fraction": float(np.mean([r['top3'] for r in results[arm]])),
                     "prediction_failure_states": sum(r["prediction_failure"] for r in results[arm]),
                     "all_tied_states": sum(r["all_tied"] for r in results[arm]),
                     "compute_complete": not any(r["prediction_failure"] for r in results[arm]),
                     "mean_horizon_carrier_mse": float(np.mean(errors[arm])) if errors[arm] else None,
                     "horizon_parsed_pig_block_count_mae": np.mean(count_errors[arm], axis=0).tolist() if count_errors[arm] else None,
                     "linear_macs": macs[arm], "mean_perception_and_planning_seconds": float(np.mean(wall[arm])),
                     "max_perception_and_planning_seconds": max(wall[arm])} for arm in ARMS}
    gate = decision(means, usage, sum(r["informative"] for r in results["adaptive"]),
                    max(v for walls in wall.values() for v in walls),
                    sum(r["prediction_failure"] for rows in results.values() for r in rows))
    return {"schema": SCHEMA, "identity": "issue-72-development-result-v1", "plan_identity": plan["identity"],
            "contract": contract(), "decision": gate, "systems": summary,
            "prior": {"selected": selected_prior, "mean_regret": means["prior"],
                      "selection_table": {k:float(v.mean()) for k,v in choices.items()}},
            "contrasts_positive_is_better": {
                "corrected_vs_teacher_forced": paired_interval(np.array(values['teacher_forced'])-values['corrected_fixed']),
                "adaptive_vs_corrected_fixed": paired_interval(np.array(values['corrected_fixed'])-values['adaptive']),
                "adaptive_vs_teacher_forced": paired_interval(np.array(values['teacher_forced'])-values['adaptive']),
                "adaptive_vs_prior": paired_interval(np.array(values['prior'])-values['adaptive'])},
            "pair_usage": dict(usage), "state_count": 200, "candidate_count_per_arm": 2400, "details": details,
            "checkpoint_identities": plan["checkpoints"], "controller_identity": plan["controller_identity"],
            "scope": "reused calibration replay ranking, not fresh gameplay; intervals descriptive, selected prior optimistic",
            "gameplay_success": None, "shots_to_success": None, "predicted_physical_violations": "unavailable",
            "compute_scope": "linear MACs include model/selected decoder/adapter/controller; wall includes common PNG/CNN/carrier preprocessing; not a matched FLOP claim",
            "offline_anchor_validation": {"wall_seconds": audit_seconds, "states_with_separate_audit": audited_states,
                                          "included_in_deployment_time": False},
            "source_repair_receipt": str(args.output / ANCHOR_REPAIR) if (args.output / ANCHOR_REPAIR).exists() else None,
            "code_revision": plan["code"]["revision"], "source_snapshot": str(args.output / "plan.json"), "archived_release": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "smoke-test", "prepare", "score-development", "run-development", "publish", "validate", "repair-anchor-validation"):
        modes.add_argument("--"+name, action="store_true")
    parser.add_argument("--issue71", type=Path, default=hybrid.ROOT)
    parser.add_argument("--parser", type=Path, default=hybrid.repair.ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv); torch.set_num_threads(4)
    try:
        if args.repair_anchor_validation:
            repair_anchor_validation(args)
        elif args.dry_run:
            plan = make_plan(args)
            log(f"no-write dry run passed states={len(plan['states'])} model_arms=3 candidates_per_arm=2400")
            print(contract(), flush=True)
        elif args.smoke_test:
            plan = make_plan(args); models, controller = load_models(args, plan)
            adapter = hybrid.repair.load_repaired_adapter(args.parser, args.device)
            value = score_state(args, plan, 0, models, controller, adapter)
            check_state(value, endpoint(args, plan, 0), plan, models)
            write(args.output / "smoke.json", {"production_evidence": False, "state": value})
            log("bounded real-agent-RGB / trained-checkpoint smoke passed; no Unity gameplay or gate decision")
        elif args.prepare:
            prepare(args)
        elif args.score_development:
            score(args)
        elif args.run_development:
            prepare(args); score(args); result = report(args)
            write(args.output / "result.json", result); log(f"development complete {result['decision']}")
        elif args.publish:
            value = report(args); write(args.output / "result.json", value); log(f"published {value['decision']}")
        else:
            if report(args) != read(args.output / "result.json"):
                raise ValueError("publication differs from complete source-bound score records")
            log("exact development publication validation passed; no fresh/final authorization")
    except (ValueError, OSError, KeyError) as error:
        log(f"error: {error}"); return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
