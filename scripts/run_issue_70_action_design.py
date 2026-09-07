"""Audit an outcome-independent task objective, action rankings, and CEM pilot."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

from scripts import run_issue_69_corrected_experiment as old
from scripts.run_issue_63_matched_experiment import DEFAULT_VISUAL_PARSER
from world_model.data.deployment_temporal import AgentObservation, TemporalObservationContext
from world_model.planning.gameplay import (
    CEMConfig, CEMPlanner, PlanningObservation, SlingshotAction,
    VisualPlanningObservationAdapter,
)
from world_model.planning.task_objective import (
    TaskObjective, TaskCandidateEvaluator, ranking_diagnostic,
    summarize_rankings, objective_gate,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".local-artifacts/issue-70-action-design-v1"
AUDIT = ROOT / "data/issue-70-pilot-audit"
SUMMARY = ROOT / "data/runtime_evidence/issue-70/action-design-v1.json"


def log(message):
    print(f"[issue-70] {message}", flush=True)


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write(path, value):
    old.write(Path(path), value)


def load_adapter(device):
    """Reuse the already validated parser without reopening its training corpus."""
    from world_model.training.cohort_v2_visual_parser import (
        CohortV2VisualParserConfig, CohortV2VisualPredicateParser, MANIFEST_SCHEMA,
    )
    root=Path(DEFAULT_VISUAL_PARSER)
    manifest=read(root/"manifest.json")
    payload=torch.load(root/manifest["checkpoint_path"],map_location="cpu",weights_only=True)
    if manifest["schema"]!=MANIFEST_SCHEMA or payload["model_state_identity"]!=manifest["model_state_identity"]:
        raise ValueError("frozen parser checkpoint metadata differs")
    config=CohortV2VisualParserConfig(**payload["config"])
    model=CohortV2VisualPredicateParser(config,tuple(payload["object_vocabulary"]))
    model.load_state_dict(payload["model_state"],strict=True)
    return VisualPlanningObservationAdapter(model.to(device).eval(),
        parser_checkpoint_identity=manifest["identity"],temperatures=payload["temperatures"],
        thresholds=payload["thresholds"],object_kind_temperature=payload["object_kind_temperature"],
        latent_dim=197,max_entities=15)


def contract():
    return {"identity": "issue-70-task-objective-experiment-v1", "role": "calibration",
        "exploratory_only": True, "final_evaluation_opened": False,
        "objective": "1000 * expected remaining pigs + expected remaining blocks",
        "presence_transform": "probability clip [0,1] only in scoring; recursive carrier unclamped",
        "outcome_derived_targets": False, "recursive_h15_steps": 15,
        "horizon_target": "observation at fixed-step offset 225; earlier stable terminal is absorbing",
        "selection": "calibration mean count-regret, then system name, then smaller disagreement coefficient",
        "penalty_grid": [0.0, 0.25, 0.5, 1.0, 2.0],
        "objective_gate": {"count_discriminating_states_min": 20, "mean_regret_max": 0.10,
                           "top3_min": 0.80, "new_failures_max": 0},
        "ranking_pilot_gate": "corrected mean count-regret strictly below frozen fixed-action prior; no prediction failures",
        "pilot_levels": 12, "pilot_seed_base": 700_500_000,
        "pilot_max_shots": 3, "pilot_retries": 1,
        "cem": {"population_size": 4, "elite_count": 2, "iterations": 3,
                "sequence_length": 1, "minimum_std": 1.0},
        "max_candidate_evaluations_per_shot": 12, "max_planning_seconds_per_shot": 30,
        "failure_treatment": "retain every state/candidate/level; realized capture failure cost=1e9; prediction failure regret=1; pilot failure success=0; no replacements",
        "pilot_inference": "paired level bootstrap 10000 draws seed 7001; descriptive 95% intervals only",
        "issue_64_authorized": False}


def prepare(args):
    legacy = read(args.issue69 / "plan.json")
    source = read(Path(legacy["release"]) / "production-plan.json")
    states = [s for s in source["states"] if s["exposure_role"] == "calibration"]
    if len(states) != 200 or source["identity"] != "issue-68-production-plan-v2:states-per-role-200":
        raise ValueError("requires the completed #68 v2 calibration inventory")
    plan = {"schema": "issue_70_plan_v1", "contract": contract(),
            "release": legacy["release"], "source_plan_identity": source["identity"],
            "states": states, "models": [c for c in legacy["cells"] if c["kind"] != "ensemble"],
            "old_result_preserved": str(args.issue69 / "result.json")}
    write(args.output / "plan.json", plan)
    log("plan frozen calibration_states=200 models=12; no model-selection/final access")


def plan_for(args):
    plan = read(args.output / "plan.json")
    if plan["contract"] != contract():
        raise ValueError("issue-70 frozen contract changed")
    return plan


def endpoint_path(args, state):
    return args.output / "endpoints" / f"state-{state['role_ordinal']+1:04d}.json"


def observations_at_endpoints(root):
    trajectory = read(root / "trajectory.json")
    shot = trajectory["shots"][-1]
    observation_root = root / shot["path"] / "observation-trace"
    manifest = read(observation_root / "observation_trace_manifest.json")
    if manifest["exposure_role"] != "calibration" or manifest["identity"] != shot["observation_manifest_identity"]:
        raise ValueError("endpoint observation role/source binding differs")
    frames = manifest["frame_records"]
    first = frames[0]["fixed_step"]
    target = min(first+225, frames[-1]["fixed_step"])
    index = next(i for i, frame in enumerate(frames) if frame["fixed_step"] == target)
    if index == 0:
        raise ValueError("candidate has no temporal prediction target")
    chosen = sorted(set((0, index-1, index, len(frames)-2, len(frames)-1)))
    observations = []
    for i in chosen:
        item = frames[i]
        ref = item["agent_observation"]
        observations.append(AgentObservation(ref["identity"], item["fixed_step"],
            item["fixed_time_seconds"], (observation_root/ref["relative_path"]).read_bytes(), "agent"))
    return chosen, observations, index, len(frames), target-first


def prepare_endpoint_state(plan, state, adapter):
    root = Path(plan["release"])
    candidates = []
    anchor = None
    objective = TaskObjective.from_vocabulary(adapter.model.object_vocabulary)
    for candidate in state["candidates"]:
        ordinal = candidate["ordinal"]
        record = read(root / "candidate-results" / f"cal-s{state['role_ordinal']+1:04d}-c{ordinal:02d}.json")
        if (record["candidate_identity"] != candidate["identity"]
                or record["state_identity"] != state["identity"]
                or record["exposure_role"] != "calibration"
                or record["plan_identity"] != plan["source_plan_identity"]):
            raise ValueError("candidate source differs")
        log(f"endpoint state={state['role_ordinal']+1}/200 candidate={ordinal}/12 status={record['status']}")
        row = {"identity": candidate["identity"],
               "action": {k: candidate[k] for k in ("drag_x", "drag_y", "tap_time_ms")},
               "accepted": record["status"] == "accepted",
               "realized_count_cost": record["goal_count_cost"] if record["status"] == "accepted" else 1e9,
               "realized_progress_cost": record["realized_cost"],
               "terminal_carrier": None, "horizon_carrier": None, "parsed_task_cost": None}
        if row["accepted"]:
            chosen, obs, index, frame_count, delta = observations_at_endpoints(root/record["trajectory_relative_path"])
            parsed = adapter.parse_batch(tuple(obs))
            by_index = dict(zip(chosen, zip(obs, parsed), strict=True))
            def carrier(current, prior):
                observation, parse = by_index[current]
                previous, previous_parse = (None, None) if prior is None else by_index[prior]
                return adapter.build_from_parsed(TemporalObservationContext(previous, observation), parse, previous_parse).tensor
            initial = carrier(0, None)
            terminal = carrier(frame_count-1, frame_count-2)
            horizon = carrier(index, index-1)
            row.update({"terminal_carrier": terminal.tolist(), "horizon_carrier": horizon.tolist(),
                        "observed_horizon_fixed_steps": delta, "parsed_task_cost": objective(terminal),
                        "parsed_counts": objective.counts(terminal),
                        "actual_counts": [record["endpoint_outcome"]["active_pigs"], record["endpoint_outcome"]["active_blocks"]],
                        "source_trajectory": record["trajectory_relative_path"]})
            if ordinal == state["carrier_anchor_candidate_ordinal"]:
                anchor = initial.tolist()
        candidates.append(row)
    if anchor is None:
        raise ValueError("prospectively designated anchor unavailable")
    return {"schema": "issue_70_endpoint_state_v1", "contract": contract(),
            "state": state["identity"], "generation_seed": state["generation_seed"],
            "context": anchor, "objective": asdict(objective),
            "parser_identity": adapter.parser_checkpoint_identity,
            "candidates": candidates, "role": "calibration"}


def prepare_endpoints(args, *, smoke=False):
    plan = plan_for(args)
    log("loading frozen deployment parser; endpoint-only RGB reads, no full trace integrity scan")
    adapter = load_adapter(args.device)
    states = plan["states"][:1] if smoke else plan["states"]
    started = time.monotonic()
    for i, state in enumerate(states, 1):
        path = endpoint_path(args, state)
        if path.exists() and not smoke:
            validate_endpoint(read(path), state)
            log(f"endpoint state={i}/{len(states)} validated existing")
            continue
        value = prepare_endpoint_state(plan, state, adapter)
        if not smoke:
            write(path, value)
        elapsed = time.monotonic()-started
        log(f"endpoint state={i}/{len(states)} complete elapsed={elapsed:.1f}s eta={elapsed/i*(len(states)-i):.1f}s")
    if smoke:
        return value


def validate_endpoint(row, state):
    if (row["contract"] != contract() or row["state"] != state["identity"] or row["role"] != "calibration"
            or [c["identity"] for c in row["candidates"]] != [c["identity"] for c in state["candidates"]]
            or [c["action"] for c in row["candidates"]] != [{k:c[k] for k in ("drag_x","drag_y","tap_time_ms")} for c in state["candidates"]]):
        raise ValueError("endpoint cache does not match frozen inventory")


def endpoint_rows(args):
    plan = plan_for(args)
    rows = []
    for state in plan["states"]:
        row = read(endpoint_path(args, state))
        validate_endpoint(row, state)
        rows.append(row)
    if len({json.dumps(r["objective"], sort_keys=True) for r in rows}) != 1:
        raise ValueError("parser vocabulary changed across states")
    return rows


def audit_payload(rows):
    count_rows, progress_rows, errors = [], [], []
    for row in rows:
        objective = TaskObjective(**{k: tuple(v) for k,v in row["objective"].items()})
        predicted = [objective(torch.tensor(c["terminal_carrier"])) if c["accepted"] else 1e9 for c in row["candidates"]]
        count_rows.append(ranking_diagnostic(predicted, [c["realized_count_cost"] for c in row["candidates"]]))
        progress_rows.append(ranking_diagnostic(predicted, [c["realized_progress_cost"] for c in row["candidates"]]))
        errors.extend([abs(a-b) for c in row["candidates"] if c["accepted"]
                       for a,b in zip(objective.counts(torch.tensor(c["terminal_carrier"])), c["actual_counts"], strict=True)])
    counts = summarize_rankings(count_rows)
    return {"schema": "issue_70_objective_audit_v1", "contract": contract(), "count_ranking": counts,
            "progress_ranking_secondary": summarize_rankings(progress_rows),
            "mean_absolute_entity_count_error": float(np.mean(errors)),
            "states": [{"state": row["state"], "count": c, "progress": p} for row,c,p in zip(rows,count_rows,progress_rows,strict=True)],
            "passed": objective_gate(counts), "exploratory_only": True}


def diagnose_parser(args):
    rows=endpoint_rows(args)
    audit=audit_payload(rows)
    groups={}
    for index,name in enumerate(("pigs","blocks")):
        actual=[];predicted=[]
        for row in rows:
            objective=TaskObjective(**{k:tuple(v) for k,v in row["objective"].items()})
            for candidate in row["candidates"]:
                if candidate["accepted"]:
                    actual.append(candidate["actual_counts"][index])
                    predicted.append(objective.counts(torch.tensor(candidate["terminal_carrier"]))[index])
        actual=np.asarray(actual);predicted=np.asarray(predicted)
        groups[name]={"predicted_min":float(predicted.min()),"predicted_max":float(predicted.max()),
            "mean_absolute_error":float(np.abs(actual-predicted).mean()),
            "by_actual_count":[{"actual":float(value),"candidates":int((actual==value).sum()),
                "predicted_mean":float(predicted[actual==value].mean())} for value in sorted(set(actual))]}
        log(f"parser diagnostic {name}: {groups[name]}")
    report={"schema":"issue_70_parser_diagnosis_v1","objective_passed":audit["passed"],
        "count_ranking":audit["count_ranking"],"counts":groups,
        "legacy_architecture_limitation":"presence logit(image,slot) = shared image term + fixed slot bias",
        "repair_architecture":"visual-parser-slot-conditioned-v2",
        "repair_trained_on_real_data":False,"old_checkpoints_changed":False,
        "required_next_step":"prospective parser retraining and validation, then regenerated carriers and matched world-model retraining",
        "pilot_authorized":False,"final_evaluation_opened":False}
    target=args.output/"parser-diagnosis.json"
    write(target,report)
    log(f"diagnosis saved={target}; current pilot remains blocked")


def observation(row):
    return PlanningObservation(row["state"], torch.tensor(row["context"]),
        tuple(row["objective"]["pig_slots"]), (0,0), frame_height=480)


def score_rows(cell, rows, device):
    model, training = old.load_model(cell, device)
    return evaluate_rows(model,training,cell,rows)


def evaluate_rows(model,training,cell,rows):
    scored = []
    started = time.monotonic()
    for i,row in enumerate(rows,1):
        objective = TaskObjective(**{k: tuple(v) for k,v in row["objective"].items()})
        evaluator = TaskCandidateEvaluator((model,), objective, old.probe._bounds(), progress=log)
        costs, diagnostics = [], []
        for candidate in row["candidates"]:
            try:
                result = evaluator.evaluate(observation(row), (SlingshotAction(**candidate["action"]),))
                predicted = evaluator.last_member_endpoints[0]
                measured = None if candidate["horizon_carrier"] is None else torch.tensor(candidate["horizon_carrier"])
                costs.append(result.total_cost)
                diagnostics.append({**evaluator.records[-1],
                    "horizon_mse": None if measured is None else float((predicted-measured).square().mean()),
                    "predicted_counts": objective.counts(predicted),
                    "horizon_count_error": None if measured is None else [a-b for a,b in zip(objective.counts(predicted),objective.counts(measured),strict=True)]})
            except (ValueError, RuntimeError) as error:
                costs.append(None)
                diagnostics.append({**evaluator.records[-1],"failure": f"{type(error).__name__}: {error}"})
        scored.append({"state": row["state"], "candidate_ids": [c["identity"] for c in row["candidates"]],
                       "costs": costs, "diagnostics": diagnostics})
        elapsed=time.monotonic()-started
        log(f"model={cell['name']} state={i}/{len(rows)} complete elapsed={elapsed:.1f}s eta={elapsed/i*(len(rows)-i):.1f}s")
    return {"schema": "issue_70_model_scores_v1", "cell": cell, "contract": contract(),
            "parser_identity":rows[0]["parser_identity"],"objective":rows[0]["objective"],
            "rows": scored, "training": training, "wall_seconds": time.monotonic()-started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}


def score_worker(cell, paths, device, connection):
    try:
        torch.set_num_threads(1)
        connection.send(("ok", score_rows(cell, [read(path) for path in paths], device)))
    except Exception as error:
        connection.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def isolated_score(cell, paths, device):
    ctx = multiprocessing.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    process = ctx.Process(target=score_worker, args=(cell,paths,device,send))
    process.start(); send.close()
    try:
        while not receive.poll(10):
            log(f"model={cell['name']} worker active")
            if not process.is_alive():
                raise RuntimeError(f"worker exited code={process.exitcode}")
        try:
            status,payload=receive.recv()
        except EOFError as error:
            raise RuntimeError(f"model worker ended without a result: {cell['name']}") from error
    finally:
        receive.close(); process.join()
    if status != "ok" or process.exitcode != 0:
        raise RuntimeError(str(payload))
    log(f"model={cell['name']} complete rss={payload['peak_rss_mib']:.1f}MiB; worker reclaimed")
    return payload


def validate_scores(value, cell, endpoints):
    if (value["cell"] != cell or value["contract"] != contract()
            or value["parser_identity"]!=endpoints[0]["parser_identity"] or value["objective"]!=endpoints[0]["objective"]
            or [r["state"] for r in value["rows"]] != [r["state"] for r in endpoints]
            or any(r["candidate_ids"] != [c["identity"] for c in e["candidates"]]
                   or len(r["costs"]) != 12 for r,e in zip(value["rows"],endpoints,strict=True))):
        raise ValueError("scoring inventory differs")


def score_models(args):
    plan=plan_for(args); rows=endpoint_rows(args)
    for i,cell in enumerate(plan["models"],1):
        log(f"scoring cell={i}/{len(plan['models'])} model={cell['name']}")
        path=args.output/"scores"/f"{cell['name']}.json"
        if path.exists():
            validate_scores(read(path),cell,rows)
            log("validated existing model")
        else:
            write(path,isolated_score(cell,[endpoint_path(args,s) for s in plan["states"]],args.device))
    systems=ranking_systems(load_scores(args,rows),rows)
    write(args.output/"ranking-diagnostics.json",{"contract":contract(),"systems":systems})
    for system in systems:
        summary=system["summary"]["all_states"]
        log(f"ranking system={system['name']} penalty={system['penalty']} regret={summary['mean_regret']:.4f} top1={summary['top1_fraction']:.3f} top3={summary['top3_fraction']:.3f}")


def ranking_systems(scores, endpoints):
    systems=[]
    for name, score in scores.items():
        systems.append({"name":name,"members":[name],"penalty":0.0,"costs":[r["costs"] for r in score["rows"]]})
    for prefix in ("original","teacher-forced-h15",*old.CORRECTIONS):
        members=[f"{prefix}-{s}" for s in old.SEEDS]
        for penalty in contract()["penalty_grid"]:
            costs=[]
            for i,row in enumerate(endpoints):
                candidate_costs=[]
                for j in range(12):
                    values=[scores[m]["rows"][i]["costs"][j] for m in members]
                    candidate_costs.append(None if any(v is None for v in values) else float(np.mean(values)+penalty*np.std(values)))
                costs.append(candidate_costs)
            systems.append({"name":f"{prefix}-ensemble", "members":members,"penalty":penalty,"costs":costs})
    for system in systems:
        diagnostics=[ranking_diagnostic(p,[c["realized_count_cost"] for c in row["candidates"]]) for p,row in zip(system["costs"],endpoints,strict=True)]
        system["summary"]=summarize_rankings(diagnostics)
        system["states"]=diagnostics
        system["progress_secondary"]=summarize_rankings([ranking_diagnostic(p,[c["realized_progress_cost"] for c in row["candidates"]]) for p,row in zip(system["costs"],endpoints,strict=True)])
    return systems


def freeze_payload(plan, rows, scores):
    audit=audit_payload(rows)
    systems=ranking_systems(scores,rows)
    def key(system):
        return system["summary"]["all_states"]["mean_regret"], system["name"], system["penalty"]
    corrected=min((s for s in systems if any(s["name"].startswith(c) for c in old.CORRECTIONS)),key=key)
    original=min((s for s in systems if s["name"].startswith("original")),key=key)
    priors=[]
    for j in range(12):
        predicted=[1.0]*12;predicted[j]=0
        regrets=[ranking_diagnostic(predicted,[c["realized_count_cost"] for c in row["candidates"]])["regret"] for row in rows]
        priors.append(float(np.mean(regrets)))
    prior=min(range(12),key=lambda j:(priors[j],j))
    def selected(system):
        return {"name":system["name"],"penalty":system["penalty"],
                "members":[next(c for c in plan["models"] if c["name"]==name) for name in system["members"]]}
    return {"schema":"issue_70_pilot_freeze_v1","contract":contract(),
            "objective":rows[0]["objective"],"parser_identity":rows[0]["parser_identity"],
            "objective_passed":audit["passed"],"corrected":selected(corrected),"original":selected(original),
            "prior_action":rows[0]["candidates"][prior]["action"],"prior_ordinal":prior+1,
            "prior_regret":priors[prior],"corrected_regret":key(corrected)[0],
            "pilot_allowed":audit["passed"] and corrected["summary"]["prediction_failures"]==0 and key(corrected)[0]<priors[prior],
            "comparison_table":[{k:s[k] for k in ("name","penalty","summary","progress_secondary")} for s in systems],
            "exploratory_only":True,"issue_64_authorized":False}


def load_scores(args, rows):
    scores={}
    for i,cell in enumerate(plan_for(args)["models"],1):
        log(f"validate model={i}/12 {cell['name']}")
        value=read(args.output/"scores"/f"{cell['name']}.json")
        validate_scores(value,cell,rows);scores[cell["name"]]=value
    return scores


def freeze_pilot(args):
    from scripts.issue_70_live_pilot import pilot_plan
    rows=endpoint_rows(args);plan=plan_for(args)
    frozen=freeze_payload(plan,rows,load_scores(args,rows))
    write(args.output/"pilot-freeze.json",frozen)
    if frozen["pilot_allowed"]:
        write(args.output/"pilot-plan.json",pilot_plan(frozen))
    log(f"pilot frozen allowed={frozen['pilot_allowed']} objective_passed={frozen['objective_passed']} corrected_regret={frozen['corrected_regret']:.4f} prior_regret={frozen['prior_regret']:.4f}")


def publish(args, validate=False):
    from scripts.issue_70_live_pilot import pilot_report
    rows=endpoint_rows(args);plan=plan_for(args);scores=load_scores(args,rows)
    frozen=freeze_payload(plan,rows,scores)
    if read(args.output/"pilot-freeze.json")!=frozen:
        raise ValueError("pilot freeze differs from calibration")
    ranking=ranking_systems(scores,rows)
    model_diagnostics={}
    for name,score in scores.items():
        diagnostics=[d for r in score["rows"] for d in r["diagnostics"]]
        valid=[d for d in diagnostics if "failure" not in d]
        aligned=[d for d in valid if d["horizon_mse"] is not None]
        model_diagnostics[name]={"failed_predictions":len(diagnostics)-len(valid),
            "mean_aligned_horizon_mse":float(np.mean([d["horizon_mse"] for d in aligned])) if aligned else None,
            "mean_absolute_pig_count_error":float(np.mean([abs(d["horizon_count_error"][0]) for d in aligned])) if aligned else None,
            "mean_absolute_block_count_error":float(np.mean([abs(d["horizon_count_error"][1]) for d in aligned])) if aligned else None,
            "maximum_carrier_bound_excess":max((d["maximum_carrier_bound_excess"] for d in valid),default=None),
            "model_evaluations":sum(d["model_evaluations"] for d in diagnostics),"score_wall_seconds":score["wall_seconds"]}
    result={"schema":"issue_70_result_v1","objective_audit":audit_payload(rows),
            "ranking_systems":ranking,"model_diagnostics":model_diagnostics,
            "freeze":frozen,"pilot":pilot_report(args) if frozen["pilot_allowed"] else None,
            "status":"exploratory_pilot_complete" if frozen["pilot_allowed"] else "not_ready_for_pilot",
            "issue_64_authorized":False,"final_evaluation_opened":False}
    if validate:
        if read(args.summary)!=result or read(args.output/"ranking-diagnostics.json")!={"contract":contract(),"systems":ranking}:
            raise ValueError("published result differs")
        log(f"exact validation passed status={result['status']}")
    else:
        write(args.summary,result)
        log(f"published {args.summary} status={result['status']}; no #64 authorization")


def dry_run():
    from scripts.issue_70_live_pilot import dry_run as pilot_dry_run
    torch.set_num_threads(1)
    objective=TaskObjective((0,),(1,))
    carrier=torch.zeros(197);carrier[2]=1;carrier[15]=1
    assert objective(carrier)==1001
    carrier[2]=0
    assert objective(carrier)==1
    assert ranking_diagnostic([3,2,1],[2,1,0])["top1"]
    assert ranking_diagnostic([None,2,1],[0,1,2])["regret"]==1
    class Toy(torch.nn.Module):
        def __init__(self,offset):
            super().__init__();self.offset=torch.nn.Parameter(torch.tensor(offset))
        def carrier(self,current,action,pair):
            output=current.clone();output[:,2]=torch.sigmoid(action[:,1]*4+self.offset)
            return output
    rows=[]
    for i in range(2):
        candidates=[]
        for j,candidate in enumerate(old.probe.broad_action_candidates(f"dry-{i}",old.probe._bounds())):
            endpoint=torch.zeros(197);endpoint[2]=float(j!=i)
            candidates.append({"identity":candidate.identity,"action":asdict(candidate.action),
                "accepted":True,"terminal_carrier":endpoint.tolist(),"horizon_carrier":endpoint.tolist(),
                "actual_counts":[float(j!=i),0],"realized_count_cost":float(j!=i)*1000,
                "realized_progress_cost":float(j!=i)*1000})
        rows.append({"state":f"dry-{i}","objective":asdict(objective),"parser_identity":"synthetic",
                     "context":torch.zeros(197).tolist(),"candidates":candidates})
    scores={};cells=[]
    for prefix in ("original","teacher-forced-h15",*old.CORRECTIONS):
        for i,seed in enumerate(old.SEEDS):
            cell={"name":f"{prefix}-{seed}","kind":"synthetic"};cells.append(cell)
            scores[cell["name"]]=evaluate_rows(Toy(i*0.01),{"synthetic":True},cell,rows)
    audit=audit_payload(rows)
    frozen=freeze_payload({"models":cells},rows,scores)
    assert not audit["passed"] and not frozen["pilot_allowed"] # miniature is below the 20-state minimum
    assert len(frozen["comparison_table"])==32
    pilot_dry_run()
    log("dry-run passed 12 models/32 ranking configurations/objective/freeze/CEM/grid/prior/closed-loop; files_written=false final_access=false")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    for mode in ("prepare","prepare-endpoints","audit-objective","diagnose-parser","score-models","freeze-pilot","run-pilot","publish","validate","dry-run","smoke-test"):
        modes.add_argument("--"+mode,action="store_true")
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--issue69",type=Path,default=old.OUTPUT)
    parser.add_argument("--summary",type=Path,default=SUMMARY)
    parser.add_argument("--audit",type=Path,default=AUDIT)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--start-display",action="store_true")
    args=parser.parse_args(argv)
    torch.set_num_threads(1)
    for field in ("output","issue69","summary","audit"):
        setattr(args,field,getattr(args,field).resolve())
    try:
        if args.dry_run:dry_run()
        elif args.prepare:prepare(args)
        elif args.prepare_endpoints:prepare_endpoints(args)
        elif args.smoke_test:
            row=prepare_endpoints(args,smoke=True)
            cell=next(c for c in plan_for(args)["models"] if c["name"]=="self-conditioned-h15-u2-20260901")
            score_rows(cell,[row],args.device)
            log("real endpoint/checkpoint smoke test complete; no artifact writes")
        elif args.audit_objective:
            report=audit_payload(endpoint_rows(args));write(args.output/"objective-audit.json",report)
            log(f"objective audit passed={report['passed']} {report['count_ranking']}")
        elif args.diagnose_parser:diagnose_parser(args)
        elif args.score_models:score_models(args)
        elif args.freeze_pilot:freeze_pilot(args)
        elif args.run_pilot:
            from scripts.issue_70_live_pilot import run_pilot
            run_pilot(args)
        else:publish(args,validate=args.validate)
    except (OSError,ValueError,RuntimeError) as error:
        print(f"error: {error}",file=sys.stderr,flush=True);return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
