"""Issue76 Phase A: fixed-pair curves, novelty inventory, and gated feasibility."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import html
import io
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from urllib.parse import quote

import numpy as np
import torch

from scripts import run_issue_74_matched_dynamics as base
from scripts import run_issue_75_controller_coverage as covered
from scripts.issue_76_novelty_inventory import build_inventory
from world_model.data.deployment_temporal import AgentObservation, TemporalObservationContext
from world_model.model import Abstraction, PredictionPair


ROOT = base.ROOT
OUTPUT = ROOT/".local-artifacts/issue-76-dynamics-diagnostic-v1"
REVIEW = ROOT/"data/issue-76-review"
PROTOCOL = "docs/issue-76-diagnostic-protocol.md"
RUNNER_SOURCE = "scripts/run_issue_76_dynamics_diagnostic.py"
VALIDATION_REPAIR = "validation-csv-repair.json"
TIMES = (15,30,60,150,225)
HORIZONS = (1,5,15)
SEEDS = base.SEEDS
SYSTEMS = {f"continuous_h{h}":("continuous",PredictionPair(h,Abstraction.CONTINUOUS)) for h in HORIZONS}
SYSTEMS.update({f"hybrid_{str(mode)}_h{h}":("hybrid",PredictionPair(h,mode)) for h in HORIZONS for mode in Abstraction})
ADAPTIVE = ("continuous_adaptive", "hybrid_adaptive", "covered_continuous", "covered_hybrid")
FIELDS = {"presence":([0],1), "kind":([2,11],4), "position":([5,6],7), "motion":([8,9],10)}
read, write = base.read, base.write


def log(message):
    print(f"[issue-76] {message}",flush=True)


def source_args(args):
    return argparse.Namespace(output=args.issue74,issue71=base.old.ROOT,parser=base.old.repair.ROOT,device=args.device)


def covered_args(args):
    return argparse.Namespace(output=args.issue75,issue74=args.issue74,device=args.device)


def definitions():
    return {"states_per_family":12,"seeds":list(SEEDS),"times":list(TIMES),"horizons":list(HORIZONS),
            "candidate_count":12,"endpoint":225,"batch_size":1,
            "systems":{name:{"arm":arm,"horizon":p.delta,"mode":str(p.abstraction)} for name,(arm,p) in SYSTEMS.items()},
            "membership":"round(j*(n-1)/11), j=0..11, within each family sorted by full identity",
            "local":"observed context at t-h -> t, evaluation only; shared initial B1 when t-h=0",
            "recursive":"one shared initial carrier; predictions feed next transition; no truth resets",
            "target_parser_batch":"original endpoint convention: sorted unique (0,p-1,p,last-1,last); cache only same frame/batch size",
            "target225":"retain archived #70 carrier after same-batch reparse allclose atol/rtol1e-5",
            "absorption":"only recorded stable_entered; retain requested/observed times and unavailable targets",
            "field_masks":"target availability only; presence[0]/mask1, kind[2,11]/mask4, position[5,6]/mask7, motion[8,9]/mask10",
            "aggregate_error":"unmasked mean squared error of all236 carrier values, as #74",
            "time_fields":"history_available and prior_elapsed_seconds; squared error separately",
            "task":"unchanged predicted count cost at225 versus actual settled replay cost; not a local MSE",
            "bootstrap":{"draws":10000,"seed":7201,"unit":"state; seed differences averaged before resampling; descriptive development only"},
            "diagnostic_seconds_max":3600,"cpu_mib_max":3072,"cuda_mib_max":2048,"artifact_bytes_max":2*2**30,
            "new_fits":0,"new_captures":0,"fresh_evaluation_opened":False,"final_evaluation_opened":False,"issue_64_authorized":False}


def select_states(states):
    selected = []
    for family in ("type010101","type010102"):
        members = sorted((s for s in states if s["generator_family"] == family),key=lambda s:s["identity"])
        if len(members) < 12 or any(s["exposure_role"] != "calibration" for s in members):
            raise ValueError("requires sufficient calibration-only family inventory")
        selected.extend(members[round(j*(len(members)-1)/11)] for j in range(12))
    if len({s["identity"] for s in selected}) != 24: raise ValueError("duplicate diagnostic membership")
    return sorted(selected,key=lambda s:s["identity"])


def needed_offsets():
    return sorted({0,*TIMES,*(t-h for t in TIMES for h in HORIZONS)})


def candidate_source(release, state, ordinal):
    record = read(release/"candidate-results"/f"cal-s{state['role_ordinal']+1:04d}-c{ordinal:02d}.json")
    declared = state["candidates"][ordinal-1]
    if (record["candidate_identity"] != declared["identity"] or record["state_identity"] != state["identity"]
            or record["exposure_role"] != "calibration"):
        raise ValueError("diagnostic candidate role/identity differs")
    result = {"identity":record["candidate_identity"],"status":record["status"],
              "ordinal":ordinal,"source_plan_identity":record["plan_identity"],"audit_manifest":record.get("audit_manifest")}
    if record["status"] != "accepted": return result
    relative = record["trajectory_relative_path"]; directory = release/relative
    trajectory = read(directory/"trajectory.json"); shot = trajectory["shots"][-1]
    obsroot = directory/shot["path"]/"observation-trace"
    manifest = read(obsroot/"observation_trace_manifest.json")
    if manifest["exposure_role"] != "calibration" or manifest["identity"] != shot["observation_manifest_identity"]:
        raise ValueError("target observation role/source differs")
    frames = manifest["frame_records"]; first = frames[0]["fixed_step"]
    if any(f["fixed_step"] != first+i for i,f in enumerate(frames)):
        raise ValueError("diagnostic requires the retained stride1 observations")
    last = len(frames)-1
    positions = {0,last-1,last}
    for offset in needed_offsets():
        p = min(offset,last); positions.add(p)
        if p: positions.add(p-1)
    observations = [{"index":i,"fixed_step":frames[i]["fixed_step"],"fixed_time_seconds":frames[i]["fixed_time_seconds"],
                     **frames[i]["agent_observation"]} for i in sorted(positions)]
    return {**result,"trajectory":relative,"trajectory_identity":trajectory["trajectory_identity"],"shot_path":shot["path"],
            "capture_id":shot["capture_id"],"manifest_identity":manifest["identity"],"terminal_reason":shot["terminal_reason"],
            "first_fixed_step":first,"last_offset":last,"frame_count":len(frames),"observations":observations}


def make_plan(args):
    sa = source_args(args); previous = base.load_plan(sa)
    cp = covered.load_plan(covered_args(args)); cr = read(args.issue75/"result.json")
    if cr["plan_identity"] != cp["identity"] or cr["inventory"] != {"controllers":6,"states":600}:
        raise ValueError("requires completed #75 publication, not a partial pilot")
    gp = base.grid.load_plan(base.grid_args(sa)); selected = select_states(gp["states"])
    release = Path(gp["source_release"])
    samples = [{"state":s,"sources":[candidate_source(release,s,j) for j in range(1,13)]} for s in selected]
    inventory = build_inventory(ROOT,previous["contract"]["vocabulary"])
    files = tuple(dict.fromkeys((*base.FILES,"scripts/run_issue_75_controller_coverage.py",
        "scripts/run_issue_76_dynamics_diagnostic.py","scripts/issue_76_novelty_inventory.py",PROTOCOL,
        "tasks/task_generator/canonical_materialization.py","tasks/task_generator/utils/data_classes.py")))
    return {"schema":"issue_76_phase_a_plan_v1","identity":"issue-76-phase-a-v1","definitions":definitions(),
            "issue74":str(args.issue74.resolve()),"issue75":str(args.issue75.resolve()),
            "source74_plan_identity":previous["identity"],"source75_plan_identity":cp["identity"],
            "source75_disposition":cr["disposition"],"source75_checks":cr["checks"],
            "source75_cost":cr["budgets"],
            "source75_controller_bindings":{str(seed):{a:covered.binding(cp,seed,a) for a in base.ARMS} for seed in SEEDS},
            "source75_repair":read(args.issue75/covered.PUBLICATION_REPAIR) if (args.issue75/covered.PUBLICATION_REPAIR).exists() else None,
            "source74_readiness":read(args.issue74/"readiness.json"),"contract":previous["contract"],
            "source_release":str(release),"source_release_plan_identity":gp["source_plan_identity"],
            "samples":samples,"inventory_source":inventory,"source_text":{p:(ROOT/p).read_text() for p in files},
            "source_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "archived_release":False,"fresh_evaluation_opened":False,"final_evaluation_opened":False,"issue_64_authorized":False}


def load_plan(args):
    plan = read(args.output/"plan.json"); current = make_plan(args)
    current["source_revision"] = plan["source_revision"]  # committing identical source is not a new experiment
    receipt_path = args.output/VALIDATION_REPAIR
    if plan != current and receipt_path.exists():
        receipt = read(receipt_path)
        if (receipt["plan_identity"] != plan["identity"]
                or receipt["original_source"] != plan["source_text"][RUNNER_SOURCE]
                or receipt["repaired_source"] != current["source_text"][RUNNER_SOURCE]):
            raise ValueError("validation repair does not bind this exact frozen/current source")
        current["source_text"][RUNNER_SOURCE] = plan["source_text"][RUNNER_SOURCE]
    if plan != current: raise ValueError("frozen #76 source/membership/definitions changed; preserve and explicitly version the run")
    return plan


def repair_validation(args):
    """Record a validation-only source amendment without rewriting the experiment."""
    if (args.output/VALIDATION_REPAIR).exists():
        load_plan(args); log("existing CSV validation repair verified"); return
    plan = read(args.output/"plan.json"); current = make_plan(args)
    repaired = current["source_text"][RUNNER_SOURCE]
    current["source_text"][RUNNER_SOURCE] = plan["source_text"][RUNNER_SOURCE]
    current["source_revision"] = plan["source_revision"]
    if plan != current:
        raise ValueError("validation repair cannot change protocol, inputs, checkpoints or other source files")
    write(args.output/VALIDATION_REPAIR, {
        "identity":"issue-76-validation-csv-repair-v1","plan_identity":plan["identity"],
        "original_source":plan["source_text"][RUNNER_SOURCE],"repaired_source":repaired,
        "scope":"CSV validation compares exact UTF-8 bytes without text newline conversion; plus exact source-repair support",
        "plan_preserved":True,"diagnostic_records_and_publication_preserved":True,
        "numerical_comparisons_unchanged":True})
    load_plan(args)
    log("CSV validation repair recorded; original plan, diagnostics and publication preserved")


def target_path(args, state):
    return args.output/"targets"/f"state-{state['role_ordinal']+1:04d}.json"


def curve_path(args, seed, state, system, ordinal):
    return args.output/"fixed"/f"seed-{seed}"/f"state-{state['role_ordinal']+1:04d}"/system/f"candidate-{ordinal:02d}.json"


def target_offsets(last, terminal_reason):
    return {str(t):{"requested_offset":t,"observed_offset":min(t,last),
                    "status":"observed" if t <= last else ("absorbed_stable_terminal" if terminal_reason == "stable_entered" else "unavailable_nonstable_truncation")}
            for t in needed_offsets()}


def target_candidate(args, source, context, cached, adapter):
    if source["status"] != "accepted":
        return {"source":source,"status":"unavailable_failed_replay","carriers":{},"timing":None}
    root = Path(args.release)/source["trajectory"]/source["shot_path"]
    obsroot = root/"observation-trace"; refs = {f["index"]:f for f in source["observations"]}
    observations, parsed = {}, {}
    parse_calls, parse_images, png_reads = 0,0,0
    began = time.monotonic()
    def observation(i):
        nonlocal png_reads
        if i not in observations:
            ref = refs[i]
            observations[i] = AgentObservation(ref["identity"],ref["fixed_step"],ref["fixed_time_seconds"],
                                               (obsroot/ref["relative_path"]).read_bytes(),"agent")
            png_reads += 1
        return observations[i]
    def at(p):
        nonlocal parse_calls,parse_images
        if p == 0: return context.clone()
        last = source["last_offset"]
        chosen = sorted({0,p-1,p,last-1,last}); size = len(chosen)
        if any((size,i) not in parsed for i in chosen):
            values = adapter.parse_batch(tuple(observation(i) for i in chosen))
            parsed.update({(size,i):v for i,v in zip(chosen,values,strict=True)})
            parse_calls += 1; parse_images += size
        return adapter.build_from_parsed(TemporalObservationContext(observation(p-1),observation(p)),
                                         parsed[(size,p)],parsed[(size,p-1)]).tensor
    positions = target_offsets(source["last_offset"],source["terminal_reason"])
    tensors = {}
    validation_difference = None
    for key, spec in positions.items():
        if spec["status"].startswith("unavailable"):
            tensors[key] = None; continue
        value = at(spec["observed_offset"])
        if int(key) == 225:
            archived = torch.tensor(cached["horizon_carrier"])
            validation_difference = float((value-archived).abs().max())
            if not torch.allclose(value,archived,atol=1e-5,rtol=1e-5):
                raise ValueError("same-batch reconstructed225 target does not match archived carrier")
            value = archived
        tensors[key] = value.tolist()
    capture = read(root/"physics_capture_v2.json")
    if capture["capture_id"] != source["capture_id"]: raise ValueError("capture timing source differs")
    relevant = ("bird_launched","collision","stable_entered","level_clear")
    events = {k:[e["fixed_step"]-source["first_fixed_step"] for e in capture["events"] if e["event_type"] == k] for k in relevant}
    base.grid.synchronize(args.device)
    return {"source":source,"status":"available","offsets":positions,"carriers":tensors,
            "timing":{"events":events,"terminal_reason":source["terminal_reason"],"last_offset":source["last_offset"]},
            "target225_cache_max_abs_difference":validation_difference,
            "target_parser":{"calls":parse_calls,"image_examples":parse_images,"unique_png_reads":png_reads,
                             "wall_seconds":time.monotonic()-began,"scope":"evaluation target construction only, not deployment latency"}}


def prepare_targets(args, plan, budget):
    sa = source_args(args); ga = base.grid_args(sa); gp = base.grid.load_plan(ga)
    adapter = base.old.repair.load_repaired_adapter(sa.parser,args.device)
    began = time.monotonic(); count = len(plan["samples"])
    for n,sample in enumerate(plan["samples"],1):
        state = sample["state"]; path = target_path(args,state)
        if path.exists():
            check_targets(read(path),sample,plan); log(f"targets state={n}/{count} cached"); continue
        row = base.grid.endpoint(ga,gp,state["role_ordinal"])
        context, perception = base.grid.perceive(ga,gp,state["role_ordinal"],row,adapter)
        targets = []
        target_args = argparse.Namespace(release=plan["source_release"],device=args.device)
        for source in sample["sources"]:
            candidate = row["candidates"][source["ordinal"]-1]
            log(f"targets state={n}/{count} candidate={source['ordinal']}/12 start")
            targets.append(target_candidate(target_args,source,context,candidate,adapter))
            log(f"targets state={n}/{count} candidate={source['ordinal']}/12 complete")
        result = {"plan_identity":plan["identity"],"state":state["identity"],"context":context.tolist(),
                  "perception":perception,"candidates":targets}
        check_targets(result,sample,plan); write(path,result); budget.tick(disk=True)
        elapsed = time.monotonic()-began
        log(f"targets state={n}/{count} complete elapsed={elapsed:.1f}s eta={elapsed/n*(count-n):.1f}s")


def check_targets(value, sample, plan):
    if value["plan_identity"] != plan["identity"] or value["state"] != sample["state"]["identity"] or len(value["context"]) != 236:
        raise ValueError("target state/parser binding differs")
    for target,source in zip(value["candidates"],sample["sources"],strict=True):
        if target["source"] != source: raise ValueError("target observation manifest binding differs")
        if source["status"] == "accepted":
            if target["offsets"] != target_offsets(source["last_offset"],source["terminal_reason"]):
                raise ValueError("target timestamp/absorption contract differs")
            for key,spec in target["offsets"].items():
                vector = target["carriers"][key]
                unavailable = spec["status"].startswith("unavailable")
                if unavailable != (vector is None) or (vector is not None and (len(vector) != 236 or not all(math.isfinite(x) for x in vector))):
                    raise ValueError("target availability/representation differs")


def action_tensor(action, device):
    bounds = base.old.repair.experiment.old.probe._bounds(); shot = base.SlingshotAction(**action)
    if not bounds.contains(shot): raise ValueError("action outside frozen grid bounds")
    return torch.tensor([[shot.drag_x/480.,shot.drag_y/480.,bounds.release_time_ms/1000.,shot.tap_time_ms/1000.,1.]],device=device)


@torch.no_grad()
def fixed_curve(model, context, action, pair):
    """Deployment boundary: current carrier/action/pair only, never observed targets."""
    device = next(model.parameters()).device
    if context.shape != (236,) or pair not in base.pairs_for(model):
        raise ValueError("incompatible fixed carrier/horizon/mode")
    z = context[None].to(device); a = action_tensor(action,device)
    output, elapsed, segments, transition_wall = {},0,[],0.
    mac = base.work(model,pair)
    for target in TIMES:
        base.grid.synchronize(device); began = time.monotonic()
        while elapsed < target:
            z = model.carrier(z,a,pair); elapsed += pair.delta
            if not bool(torch.isfinite(z).all()):
                return {"outputs":output,"failure":"nonfinite_recursive_carrier","completed_steps":elapsed,
                        "segments":segments,"linear_macs":mac*(elapsed//pair.delta),"transition_wall_seconds":transition_wall+time.monotonic()-began}
        base.grid.synchronize(device); duration = time.monotonic()-began; transition_wall += duration
        output[str(target)] = z[0].cpu().tolist()
        segments.append({"endpoint":target,"transition_calls":(target-(segments[-1]["endpoint"] if segments else 0))//pair.delta,
                         "wall_seconds":duration})
    return {"outputs":output,"failure":None,"completed_steps":elapsed,"segments":segments,
            "linear_macs":mac*(elapsed//pair.delta),"transition_wall_seconds":transition_wall}


@torch.no_grad()
def local_predictions(model, target, context, action, pair):
    """Observed-context diagnostics only; never used to choose an action."""
    device = next(model.parameters()).device; a = action_tensor(action,device)
    outputs, calls, unavailable = {},0,{}
    base.grid.synchronize(device); began = time.monotonic()
    for t in TIMES:
        value = context.tolist() if t == pair.delta else target["carriers"].get(str(t-pair.delta))
        if value is None or target["carriers"].get(str(t)) is None:
            outputs[str(t)] = None; unavailable[str(t)] = "unavailable_observed_context_or_target"; continue
        predicted = model.carrier(torch.tensor([value],device=device),a,pair); calls += 1
        outputs[str(t)] = predicted[0].cpu().tolist() if bool(torch.isfinite(predicted).all()) else None
        if outputs[str(t)] is None: unavailable[str(t)] = "nonfinite_local_prediction"
    base.grid.synchronize(device)
    return {"outputs":outputs,"unavailable":unavailable,"calls":calls,"linear_macs":calls*base.work(model,pair),
            "wall_seconds":time.monotonic()-began,"scope":"evaluation local-error probe, excluded from deployment work"}


def fixed_record(args, plan, previous, sample, target_state, row, seed, system, ordinal, model, original):
    _,pair = SYSTEMS[system]; candidate = row["candidates"][ordinal-1]
    context = torch.tensor(target_state["context"])
    prediction = fixed_curve(model,context,candidate["action"],pair)
    target = next(t for t in target_state["candidates"] if t["source"]["ordinal"] == ordinal)
    local = local_predictions(model,target,context,candidate["action"],pair)
    objective = base.TaskObjective.from_vocabulary(plan["contract"]["vocabulary"])
    cost = None if prediction["failure"] else objective(torch.tensor(prediction["outputs"]["225"]))
    endpoint_audit = None
    old_policy = system if system.startswith("continuous_") else ("hybrid_continuous_h15" if system == "hybrid_continuous_h15" else None)
    if old_policy is not None:
        old = original["policies"][old_policy][ordinal-1]
        if prediction["failure"] is None and old["failure"] is None:
            vector = torch.tensor(prediction["outputs"]["225"]); before = torch.tensor(old["endpoint"])
            endpoint_audit = float((vector-before).abs().max())
            if not torch.allclose(vector,before,atol=1e-5,rtol=1e-5):
                raise ValueError("fixed225 prediction does not reproduce unchanged #74 control")
    return {"plan_identity":plan["identity"],"seed":seed,"system":system,"state":sample["state"]["identity"],
            "candidate_identity":candidate["identity"],"ordinal":ordinal,"action":candidate["action"],
            "prediction":prediction,"local":local,"cost":cost,"unchanged_control225_max_abs_difference":endpoint_audit,
            "work_scope":"instrumented fixed-rollout wall with prefix synchronization; target parsing/local probes separate; MACs not full FLOPs"}


def check_curve(record, plan, sample, row, seed, system, ordinal, model):
    candidate = row["candidates"][ordinal-1]; _,pair = SYSTEMS[system]
    if any(record[k] != v for k,v in {"plan_identity":plan["identity"],"seed":seed,"system":system,
            "state":sample["state"]["identity"],"candidate_identity":candidate["identity"],"ordinal":ordinal,"action":candidate["action"]}.items()):
        raise ValueError("fixed diagnostic checkpoint/state/action binding differs")
    pred = record["prediction"]; elapsed = pred["completed_steps"]
    if pred["linear_macs"] != base.work(model,pair)*(elapsed//pair.delta) or elapsed % pair.delta:
        raise ValueError("fixed execution work differs")
    if not pred["failure"]:
        if elapsed != 225 or set(pred["outputs"]) != {str(t) for t in TIMES}:
            raise ValueError("missing common physical endpoint")
        for values in pred["outputs"].values():
            if len(values) != 236 or not all(math.isfinite(v) for v in values): raise ValueError("invalid predicted carrier")
        prior = 0
        for segment,t in zip(pred["segments"],TIMES,strict=True):
            if segment["endpoint"] != t or segment["transition_calls"] != (t-prior)//pair.delta:
                raise ValueError("fixed prefix execution timeline differs")
            prior = t
        objective = base.TaskObjective.from_vocabulary(plan["contract"]["vocabulary"])
        if objective(torch.tensor(pred["outputs"]["225"])) != record["cost"]:
            raise ValueError("predicted task cost differs from225 carrier")
    elif record["cost"] is not None: raise ValueError("failed rollout has a task cost")
    if record["local"]["linear_macs"] != record["local"]["calls"]*base.work(model,pair):
        raise ValueError("local diagnostic work differs")
    if set(record["local"]["outputs"]) != {str(t) for t in TIMES}:
        raise ValueError("local endpoints differ")


def stored_curve(args, plan, previous, sample, targets, row, seed, system, ordinal, model, original):
    path = curve_path(args,seed,sample["state"],system,ordinal)
    if path.exists():
        record = read(path)
    else:
        record = fixed_record(args,plan,previous,sample,targets,row,seed,system,ordinal,model,original)
        check_curve(record,plan,sample,row,seed,system,ordinal,model)
        write(path,record)
    check_curve(record,plan,sample,row,seed,system,ordinal,model)
    return record


def run_diagnostic(args, plan):
    # Reuse the small persisted counter; all phase-specific limits live in this plan.
    existing = artifact_inventory(args,plan)
    if (existing["target_states"] == existing["expected_target_states"]
            and existing["fixed_candidate_records"] == existing["expected_fixed_candidate_records"]):
        log("complete diagnostic inventory already exists; no new inference; use --publish/--validate for evidence checks")
        return
    budget_plan = {"identity":plan["identity"],"protocol":plan["definitions"]}
    budget = covered.Budget(args,budget_plan,"diagnostic")
    access = args.output/"diagnostic-access.json"
    if not access.exists(): write(access,{"plan_identity":plan["identity"],"role":"already_opened_calibration",
                                          "first_access_utc":datetime.now(timezone.utc).isoformat(),"fresh_evaluation_opened":False})
    prepare_targets(args,plan,budget)
    sa = source_args(args); previous = base.load_plan(sa); gp = base.grid.load_plan(base.grid_args(sa))
    began = time.monotonic(); completed = 0; total = len(plan["samples"])*len(SEEDS)*len(SYSTEMS)*12
    for seed in SEEDS:
        models = {a:base.load_predictor(sa,previous,seed,a)[0].requires_grad_(False) for a in base.ARMS}
        for n,sample in enumerate(plan["samples"],1):
            state = sample["state"]; targets = read(target_path(args,state)); check_targets(targets,sample,plan)
            row = base.grid.endpoint(base.grid_args(sa),gp,state["role_ordinal"])
            original = read(args.issue74/f"seed-{seed}"/"development"/f"state-{state['role_ordinal']+1:04d}.json")
            for system,(arm,pair) in SYSTEMS.items():
                for ordinal in range(1,13):
                    record = stored_curve(args,plan,previous,sample,targets,row,seed,system,ordinal,models[arm],original)
                    budget.tick()
                    completed += 1; elapsed = time.monotonic()-began
                    log(f"diagnostic={completed}/{total} seed={seed} state={n}/24 system={system} candidate={ordinal}/12 failure={record['prediction']['failure']} elapsed={elapsed:.1f}s eta={elapsed/completed*(total-completed):.1f}s")
            budget.tick(disk=True)
    log("fixed-pair diagnostics complete; no fresh experiment or advancement authorization")


def field_errors(prediction, target, objective):
    if prediction is None or target is None:
        return None
    p,t = torch.tensor(prediction),torch.tensor(target)
    if p.shape != (236,) or t.shape != (236,): raise ValueError("field diagnostic requires236-value carriers")
    slots_p,slots_t = p[2:].reshape(18,13),t[2:].reshape(18,13)
    result = {"carrier_mse":float((p-t).square().mean()),
              "history_available_mse":float((p[0]-t[0]).square()),
              "prior_elapsed_seconds_mse":float((p[1]-t[1]).square()),
              "availability_bits_mse":float((slots_p[:,[1,4,7,10,12]]-slots_t[:,[1,4,7,10,12]]).square().mean())}
    for field,(columns,mask_column) in FIELDS.items():
        mask = slots_t[:,mask_column] > .5
        squared = (slots_p[:,columns]-slots_t[:,columns]).square()
        result[field+"_available_values"] = int(mask.sum())*len(columns)
        result[field+"_mse"] = float(squared[mask].mean()) if bool(mask.any()) else None
    pc,tc = objective.counts(p),objective.counts(t)
    result["parsed_pig_count_absolute_error"] = abs(pc[0]-tc[0])
    result["parsed_block_count_absolute_error"] = abs(pc[1]-tc[1])
    if any(isinstance(v,float) and not math.isfinite(v) for v in result.values()):
        raise ValueError("nonfinite error statistic; do not serialize as null or favorable missing data")
    return result


def average_errors(values, *, unit="candidates"):
    present = [v for v in values if v is not None and "carrier_mse" in v]
    if not present: return {"available_"+unit:0}
    result = {"available_"+unit:len(present)}
    for key in present[0]:
        items = [v[key] for v in present if v.get(key) is not None]
        output_key = "mean_"+key if key == "available_candidates" else key
        result[output_key] = float(np.mean(items)) if items else None
    return result


def outcome_flags(record):
    if record["status"] != "accepted": return {k:False for k in ("pig_removed","level_clear","pig_contact","block_contact","support_change")}
    events = record["interaction_coverage"]; endpoint = record["endpoint_outcome"]
    return {"pig_removed":"pig_removed" in events,"level_clear":"level_clear" in events,
            "pig_contact":endpoint["pig_contact"],"block_contact":endpoint["block_contact"],"support_change":endpoint["support_change"]}


def task_metrics(costs, row, raw):
    metrics = base.grid.ranked(costs,row["candidates"])
    selected = metrics["selected"]
    accepted = [i for i,c in enumerate(row["candidates"]) if c["accepted"]]
    best = min((row["candidates"][i]["realized_count_cost"] for i in accepted),default=1e9)
    selected_cost = 1e9 if selected is None else row["candidates"][selected]["realized_count_cost"]
    flags = {k:False for k in ("pig_removed","level_clear","pig_contact","block_contact","support_change")} if selected is None else outcome_flags(raw[selected])
    return {**metrics,"selected_ordinal":None if selected is None else selected+1,
            "realized_count_cost":selected_cost,"excess_count_cost_over_best":selected_cost-best,"selected_outcomes":flags}


def check_cached_adaptive(items, row, model, controller, objective):
    costs = {p:base.work(model,p,controller) for p in base.pairs_for(model)}
    for item,candidate in zip(items,row["candidates"],strict=True):
        if item["candidate_identity"] != candidate["identity"] or item["action"] != candidate["action"]:
            raise ValueError("cached adaptive candidate differs")
        if item["failure"]:
            if item["cost"] is not None: raise ValueError("failed cached adaptive prediction has a score")
            continue
        if objective(torch.tensor(item["endpoint"])) != item["cost"]:
            raise ValueError("cached adaptive score differs from its predicted endpoint")
        elapsed = 0
        for t in item["trace"]:
            pair = PredictionPair(t["horizon"],Abstraction(t["mode"]))
            if (pair not in costs or t["start_fixed_step"] != elapsed or t["linear_macs"] != costs[pair]
                    or t["transition_calls"] != 1 or t["controller_calls"] != 1
                    or t["symbol_decoder_calls"] != int(pair.abstraction != Abstraction.CONTINUOUS)):
                raise ValueError("cached adaptive execution/work trace differs")
            elapsed += pair.delta
        if elapsed != 225: raise ValueError("cached adaptive endpoint differs")


def aggregate_states(rows):
    return {"states":len(rows),"mean_regret":float(np.mean([r["task"]["regret"] for r in rows])),
            "top1_fraction":float(np.mean([r["task"]["top1"] for r in rows])),
            "top3_fraction":float(np.mean([r["task"]["top3"] for r in rows])),
            "prediction_failure_states":sum(r["task"]["prediction_failure"] for r in rows),
            "local_prediction_failures":sum(r.get("local_prediction_failures",0) for r in rows),
            "predicted_all_tied_states":sum(r["task"]["all_tied"] for r in rows),
            "mean_realized_count_cost":float(np.mean([r["task"]["realized_count_cost"] for r in rows])),
            "selected_outcome_counts":{k:sum(r["task"]["selected_outcomes"][k] for r in rows) for k in rows[0]["task"]["selected_outcomes"]},
            "transition_linear_macs":sum(r["work"]["linear_macs"] for r in rows),
            "executed_calls":{k:sum(r["work"].get(k,0) for r in rows) for k in
                              ("controller_calls","transition_calls","symbol_decoder_calls","symbol_adapter_calls","local_transition_calls")},
            "local_probe_linear_macs":sum(r["work"].get("local_linear_macs",0) for r in rows),
            "local_probe_wall_seconds":sum(r["work"].get("local_wall_seconds",0.) for r in rows),
            "mean_initial_perception_seconds":float(np.mean([r["work"].get("perception_seconds",0.) for r in rows])),
            "mean_instrumented_model_seconds_per_state":float(np.mean([r["work"]["wall_seconds"] for r in rows])),
            "curves":{str(t):average_errors([r["recursive"][str(t)] for r in rows if str(t) in r["recursive"]],unit="states") for t in TIMES},
            "local_curves":{str(t):average_errors([r["local"][str(t)] for r in rows if str(t) in r["local"]],unit="states") for t in TIMES}}


def artifact_inventory(args, plan):
    expected = len(plan["samples"])*len(SEEDS)*len(SYSTEMS)*12
    targets = sum(target_path(args,s["state"]).exists() for s in plan["samples"])
    curves = sum(curve_path(args,seed,s["state"],name,j).exists() for seed in SEEDS for s in plan["samples"] for name in SYSTEMS for j in range(1,13))
    return {"target_states":targets,"expected_target_states":len(plan["samples"]),"fixed_candidate_records":curves,"expected_fixed_candidate_records":expected}


def requirement_ledger(complete, budget_stopped):
    measured = "completed" if complete else ("stopped_at_declared_budget" if budget_stopped else "pending_operator_diagnostic")
    gate = "not_executed_preaccess_readiness_stop"
    return [
        {"id":"inputs","requirement":"#73/#74/#75 source-bound predecessors and preserved negative evidence","status":"completed"},
        {"id":"claims","requirement":"training/symbolic-execution/adaptation/task claims separated; no end-to-end symbol-free or joint-mechanism claim","status":"completed"},
        {"id":"a01_membership","requirement":"24 metadata-selected calibration states, 3 paired seeds, 12 actions; exact source/target references","status":"completed"},
        {"id":"a01_curves","requirement":"pure3 and hybrid9 fixed pairs, recursive/local15/30/60/150/225 diagnostics and field masks","status":measured},
        {"id":"a01_targets","requirement":"B=1 initial perception, same-image cache audit, predecessor motion, stable absorption and actual timing","status":measured},
        {"id":"a01_outcomes","requirement":"settled action regret/top-k, headroom, absolute progress, all seeds, failures and ties","status":measured},
        {"id":"a01_adaptation","requirement":"original/covered adaptive controls at225 only, no interpolated adaptive curves","status":measured},
        {"id":"a01_uncertainty","requirement":"paired-state descriptive intervals; candidates/seed repeats not independent","status":measured},
        {"id":"a01_work","requirement":"actual fixed/local/parser/controller/decoder work and wall; no infilling or equal-FLOP claim","status":measured},
        {"id":"a02_matrix","requirement":"all45 scenario/novelty cells,80 templates,40 paired workbook/source inventories","status":"completed"},
        {"id":"a02_interfaces","requirement":"authored slots, appearances, actions, novel force/event observability and current collector scope","status":"completed"},
        {"id":"a03_recommendation","requirement":"metadata-based staged normal/novel recommendations; zero-shot versus adaptation and paired roles","status":"completed_costed_proposal_only"},
        {"id":"a03_new_engineering","requirement":"versioned broader collector/carrier/perception changes, if needed","status":"requires_explicit_approval","reason":"outside current two-family interface; no silent refit or dropped objects"},
        {"id":"a03_rendered_smoke","requirement":"new rendered compatibility smoke after exact approved membership/resource freeze","status":gate,"reason":"only a metadata proposal exists; no new collection/role/seed authorization"},
        {"id":"fresh_protocol","requirement":"fresh population, counts, seeds, hypotheses/margins, multiplicity, endpoints, compute and failure budgets","status":gate,"reason":"#75 negative candidate screen; no eligible archived fresh protocol to freeze"},
        {"id":"fresh_precision","requirement":"prospective paired success power/precision with rare events and clustered repeats","status":gate,"reason":"fresh protocol not opened; no fictitious numeric power claim"},
        {"id":"fresh_roles","requirement":"global lineage/normal-novel variant disjointness, generation inventory and access ordering","status":gate,"reason":"no fresh instances generated; public template metadata is not proof of fresh lineages"},
        {"id":"fresh_execution","requirement":"all declared systems on all paired fresh units, no replacements or favorable selection","status":gate,"reason":"pre-access stop, not an executed fresh negative experiment"},
        {"id":"fresh_metrics","requirement":"actual gameplay success, shots, failures, ranking, common-endpoint error, physical validity where available","status":gate,"reason":"no fresh gameplay outcomes; carrier MSE not called physical violation"},
        {"id":"fresh_graphics","requirement":"agent frames/WebM incl failures, graphics-enabled isolated engine processes, progress/resume","status":gate,"reason":"no engine/capture path changed; review page links existing source evidence only"},
        {"id":"archive","requirement":"source/checkpoint archive before fresh access","status":"requires_commit_push_and_archive_authority","reason":"local exact source snapshot is not an archived release"},
        {"id":"validation","requirement":"source-bound inventories/metrics/repairs/decision validation and JSON round trip","status":measured},
        {"id":"advancement","requirement":"all primary/mode/prior/compute/uncertainty gates before #64/#65","status":"not_authorized","reason":"diagnostics cannot override the #75 negative result"},
        {"id":"stop_handoff","requirement":"explicit pre-access disposition; preserve #15/#72/#74/#75 and support honest consolidation","status":"completed"},
    ]


def summarize_contrasts(per_state):
    pairs = []
    for h in HORIZONS:
        pairs.append((f"hybrid_continuous_h{h}",f"continuous_h{h}","training_effect"))
        for mode in ("micro","macro"):
            pairs.append((f"hybrid_{mode}_h{h}",f"hybrid_continuous_h{h}","symbolic_execution"))
    for adaptive in ("hybrid_adaptive","covered_hybrid"):
        for fixed in (k for k in SYSTEMS if k.startswith("hybrid_")):
            pairs.append((adaptive,fixed,"adaptation"))
    for adaptive in ("continuous_adaptive","covered_continuous"):
        for h in HORIZONS: pairs.append((adaptive,f"continuous_h{h}","adaptation"))
    result = []
    for tested,reference,kind in pairs:
        ids = sorted(per_state[str(SEEDS[0])][tested])
        differences = [float(np.mean([per_state[str(seed)][reference][s]["task"]["regret"]-
                                     per_state[str(seed)][tested][s]["task"]["regret"] for seed in SEEDS])) for s in ids]
        curves = {}
        for t in TIMES:
            paired = []
            for state in ids:
                across_seeds = []
                for seed in SEEDS:
                    a = per_state[str(seed)][tested][state]["recursive"].get(str(t),{}).get("carrier_mse")
                    b = per_state[str(seed)][reference][state]["recursive"].get(str(t),{}).get("carrier_mse")
                    if a is not None and b is not None: across_seeds.append(b-a)
                if len(across_seeds) == len(SEEDS): paired.append(float(np.mean(across_seeds)))
            curves[str(t)] = {"paired_states":len(paired),**(base.grid.paired_interval(paired) if paired else {})}
        result.append({"tested":tested,"reference":reference,"kind":kind,
                       "positive_is_improvement":True,"regret":base.grid.paired_interval(differences),"carrier_mse":curves,
                       "scope":"descriptive reused-calibration, not multiplicity-adjusted confirmation or a new eligibility test"})
    return result


def publication(args, plan):
    inv = artifact_inventory(args,plan)
    inv_path = args.output/"novelty-inventory.json"
    inventory = read(inv_path) if inv_path.exists() else plan["inventory_source"]
    if inventory != plan["inventory_source"] or build_inventory(ROOT,plan["contract"]["vocabulary"]) != inventory:
        raise ValueError("novelty/template/constraint inventory differs from frozen source")
    budget_path = args.output/"budget-diagnostic.json"
    budget = read(budget_path) if budget_path.exists() else None
    if budget is not None and budget["plan_identity"] != plan["identity"]: raise ValueError("diagnostic budget binding differs")
    complete = inv["target_states"] == inv["expected_target_states"] and inv["fixed_candidate_records"] == inv["expected_fixed_candidate_records"]
    if complete and budget is None: raise ValueError("completed diagnostics are missing their resource-accounting receipt")
    stopped = bool(budget and budget["stopped"])
    common = {"schema":"issue_76_phase_a_report_v1","identity":"issue-76-phase-a-report-v1","plan_identity":plan["identity"],
              "inventory":inv,"diagnostics_complete":complete,"diagnostic_budget_stopped":stopped,"budget":budget,
              "novelty_inventory_summary":inventory["summary"],"wider_recommendation":inventory["wider_recommendation"],
              "requirement_ledger":requirement_ledger(complete,stopped),
              "disposition":"readiness_or_precision_insufficient","disposition_scope":"pre-access feasibility stop, NOT an executed fresh negative experiment",
              "preaccess_reasons":["#75 did not establish an eligible corrective candidate","broader template/interface/perception compatibility unvalidated","source/checkpoint archive and fresh numerical protocol not authorized/frozen"],
              "fresh_protocol_frozen":False,"fresh_scheduled_lineages":0,"fresh_executed_lineages":0,
              "new_fitting_updates":0,"new_captures":0,"archived_release":False,
              "fresh_evaluation_opened":False,"final_evaluation_opened":False,"issue_64_authorized":False,
              "ticket_completion_recommendation":"review_preaccess_stop" if complete or stopped else "await_operator_diagnostic",
              "limitations":["shared semantically supervised CNN, not end-to-end symbol-free comparison",
                             "only normal single-force/multiple-forces model evidence; template inventory is not wider performance evidence",
                             "prediction225 may precede settlement; local errors are separate","linear MACs not full FLOPs or matched total work",
                             "cached adaptive wall times from older runs not paired latency evidence","no unique joint/nonseparable mechanism claim"]}
    if not complete: return common
    sa = source_args(args); previous = base.load_plan(sa); gp = base.grid.load_plan(base.grid_args(sa))
    covered_plan = covered.load_plan(covered_args(args))
    objective = base.TaskObjective.from_vocabulary(plan["contract"]["vocabulary"])
    all_states, summary, headroom, timings, target_cost = {},{},[],[],0.
    for sample in plan["samples"]:
        target_state = read(target_path(args,sample["state"])); check_targets(target_state,sample,plan)
        timings.append({"state":sample["state"]["identity"],"candidates":[t["timing"] for t in target_state["candidates"]]})
        target_cost += sum(t.get("target_parser",{}).get("wall_seconds",0.) for t in target_state["candidates"])
    for seed in SEEDS:
        models = {a:base.load_predictor(sa,previous,seed,a)[0] for a in base.ARMS}
        controllers = {a+"_adaptive":base.load_control(sa,previous,seed,a)[0] for a in base.ARMS}
        controllers.update({"covered_"+a:covered.load_control(covered_args(args),covered_plan,seed,a)[0] for a in base.ARMS})
        seed_rows = {name:{} for name in (*SYSTEMS,*ADAPTIVE)}
        for n,sample in enumerate(plan["samples"],1):
            state = sample["state"]; targets = read(target_path(args,state))
            row = base.grid.endpoint(base.grid_args(sa),gp,state["role_ordinal"])
            raw = [read(Path(plan["source_release"])/"candidate-results"/f"cal-s{state['role_ordinal']+1:04d}-c{j:02d}.json") for j in range(1,13)]
            for source,candidate,record in zip(sample["sources"],row["candidates"],raw,strict=True):
                if source["identity"] != candidate["identity"] or record["candidate_identity"] != source["identity"]:
                    raise ValueError("outcome source differs")
            if seed == SEEDS[0]:
                flags = [outcome_flags(r) for r in raw]
                regrets,informative = base.grid.normalized_regrets(row["candidates"])
                headroom.append({"state":state["identity"],"family":state["generator_family"],"informative":informative,
                                 "prior_ordinal09_regret":regrets[8],"uniform_expected_regret":float(np.mean(regrets)),
                                 "opportunities":{k:any(f[k] for f in flags) for k in flags[0]}})
            for system,(arm,pair) in SYSTEMS.items():
                records = [read(curve_path(args,seed,state,system,j)) for j in range(1,13)]
                for j,record in enumerate(records,1): check_curve(record,plan,sample,row,seed,system,j,models[arm])
                recursive,local = {},{}
                for t in TIMES:
                    recursive[str(t)] = average_errors([field_errors(r["prediction"]["outputs"].get(str(t)),target["carriers"].get(str(t)),objective)
                                                        for r,target in zip(records,targets["candidates"],strict=True)])
                    local[str(t)] = average_errors([field_errors(r["local"]["outputs"].get(str(t)),target["carriers"].get(str(t)),objective)
                                                    for r,target in zip(records,targets["candidates"],strict=True)])
                seed_rows[system][state["identity"]] = {"family":state["generator_family"],"task":task_metrics([r["cost"] for r in records],row,raw),
                    "local_prediction_failures":sum(v == "nonfinite_local_prediction" for r in records for v in r["local"]["unavailable"].values()),
                    "recursive":recursive,"local":local,"work":{"linear_macs":sum(r["prediction"]["linear_macs"] for r in records),
                    "wall_seconds":sum(r["prediction"]["transition_wall_seconds"] for r in records),
                    "local_linear_macs":sum(r["local"]["linear_macs"] for r in records),
                    "local_transition_calls":sum(r["local"]["calls"] for r in records),
                    "local_wall_seconds":sum(r["local"]["wall_seconds"] for r in records),
                    "controller_calls":0,"transition_calls":sum(r["prediction"]["completed_steps"]//pair.delta for r in records),
                    "symbol_decoder_calls":sum(r["prediction"]["completed_steps"]//pair.delta for r in records) if pair.abstraction != Abstraction.CONTINUOUS else 0,
                    "symbol_adapter_calls":sum(r["prediction"]["completed_steps"]//pair.delta for r in records) if pair.abstraction != Abstraction.CONTINUOUS else 0,
                    "perception_seconds":targets["perception"]["wall_seconds"]}}
            original = read(args.issue74/f"seed-{seed}"/"development"/f"state-{state['role_ordinal']+1:04d}.json")
            new = read(args.issue75/f"seed-{seed}"/"states"/f"state-{state['role_ordinal']+1:04d}.json")
            if (original["state"] != state["identity"] or new["state"] != state["identity"] or original["seed"] != seed or new["seed"] != seed
                    or original["plan_identity"] != plan["source74_plan_identity"] or new["plan_identity"] != plan["source75_plan_identity"]
                    or new["controller_bindings"] != plan["source75_controller_bindings"][str(seed)]):
                raise ValueError("cached adaptation state/seed differs")
            for name in ADAPTIVE:
                cached = new if name.startswith("covered_") else original
                items = cached["policies"][name]
                arm = "continuous" if "continuous" in name else "hybrid"
                check_cached_adaptive(items,row,models[arm],controllers[name],objective)
                seed_rows[name][state["identity"]] = {"family":state["generator_family"],"task":task_metrics([r["cost"] for r in items],row,raw),
                    "recursive":{"225":average_errors([field_errors(item["endpoint"],target["carriers"].get("225"),objective) for item,target in zip(items,targets["candidates"],strict=True)])},
                    "local":{},"intermediate_curves":"unavailable: no interpolation of adaptive skipped endpoints",
                    "work":{"linear_macs":sum(t["linear_macs"] for r in items for t in r["trace"]),
                            "wall_seconds":sum(r["wall_seconds"] for r in items),
                            "controller_calls":sum(t["controller_calls"] for r in items for t in r["trace"]),
                            "transition_calls":sum(t["transition_calls"] for r in items for t in r["trace"]),
                            "symbol_decoder_calls":sum(t["symbol_decoder_calls"] for r in items for t in r["trace"]),
                            "symbol_adapter_calls":sum(t["symbol_decoder_calls"] for r in items for t in r["trace"]),
                            "perception_seconds":cached["perception"]["wall_seconds"],"source":"cached #74/#75 executed traces, not new inference"}}
            log(f"publication seed={seed} state={n}/24 validated")
        all_states[str(seed)] = seed_rows
        summary[str(seed)] = {name:aggregate_states(list(rows.values())) for name,rows in seed_rows.items()}
    contrasts = summarize_contrasts(all_states)
    return {**common,"per_seed":summary,"per_state":all_states,"contrasts":contrasts,"headroom":headroom,"timing":timings,
            "target_parsing_seconds":target_cost,"source_training_cost":{"issue74":plan["source74_readiness"]["cost"],"issue75":plan["source75_cost"],
                "scope":"both original and failed covered-controller attempts retained; earlier common CNN/shard preparation referenced by frozen predecessor plans, not double-counted here"},
            "independent_state_count":len(plan["samples"]),"source_family_counts":dict(Counter(s["state"]["generator_family"] for s in plan["samples"])),
            "mean_prior_regret":float(np.mean([r["prior_ordinal09_regret"] for r in headroom])),
            "mean_uniform_expected_regret":float(np.mean([r["uniform_expected_regret"] for r in headroom]))}


def compact_report(result):
    return {k:v for k,v in result.items() if k not in ("per_state","timing")}


def curve_csv(result):
    stream = io.StringIO(); writer = csv.writer(stream)
    writer.writerow(("seed","system","elapsed_fixed_steps","kind","metric","value","available_states"))
    for seed,systems in result.get("per_seed",{}).items():
        for name,system in systems.items():
            for kind,key in (("recursive","curves"),("local","local_curves")):
                for t,metrics in system[key].items():
                    for metric,value in metrics.items():
                        if metric != "available_states": writer.writerow((seed,name,t,kind,metric,value,metrics["available_states"]))
    return stream.getvalue()


def svg_chart(title, series):
    """Code-native plot of recorded fixed-pair points, not synthetic observations."""
    palette = ("#1d4ed8","#b91c1c","#047857","#a16207","#7e22ce","#0e7490")
    available = [v for points in series.values() for _,v in points if v is not None and math.isfinite(v)]
    if not available: return "<p>No complete curve values available.</p>"
    logs = [math.log10(max(v,1e-12)) for v in available]; low,high = math.floor(min(logs)),math.ceil(max(logs))
    if low == high: high = low+1
    def x(t): return 65+640*(t-15)/210
    def y(v): return 265-210*(math.log10(max(v,1e-12))-low)/(high-low)
    lines = [f'<svg viewBox="0 0 770 {310+22*len(series)}" role="img" aria-label="{html.escape(title)}">',
             f'<text x="20" y="22" font-size="15">{html.escape(title)}</text>',
             '<path d="M65 45V265H715" fill="none" stroke="#444"/>']
    for t in TIMES: lines.append(f'<text x="{x(t)-10:.2f}" y="285" font-size="12">{t}</text>')
    for exponent in (low,high): lines.append(f'<text x="5" y="{y(10.**exponent):.2f}" font-size="12">1e{exponent}</text>')
    for i,(name,points) in enumerate(series.items()):
        valid = [(t,v) for t,v in points if v is not None and math.isfinite(v)]; color = palette[i%len(palette)]
        coords = " ".join(f"{x(t):.2f},{y(v):.2f}" for t,v in valid)
        lines.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for t,v in valid: lines.append(f'<circle cx="{x(t):.2f}" cy="{y(v):.2f}" r="3" fill="{color}"/>')
        lines.append(f'<text x="65" y="{310+22*i}" fill="{color}" font-size="12">{html.escape(name)}</text>')
    lines.append('</svg>')
    return "".join(lines)


def mean_series(result, name, kind="curves", metric="carrier_mse"):
    points = []
    for t in TIMES:
        values = [s[name][kind][str(t)].get(metric) for s in result.get("per_seed",{}).values()]
        points.append((t,float(np.mean(values)) if values and all(v is not None for v in values) else None))
    return points


def review_manifest(args, plan):
    entries = []
    video_root = ROOT/"data/issue-68-ranking-audit-v2/production"
    for sample in plan["samples"]:
        source = sample["sources"][0]; state = sample["state"]
        video = None
        if source.get("audit_manifest"):
            audit_path = video_root/source["audit_manifest"]
            audit = read(audit_path)
            candidate = video_root/audit["video_path"]
            video = str(candidate) if candidate.is_file() else None
        frames = []
        if source["status"] == "accepted":
            obsroot = Path(plan["source_release"])/source["trajectory"]/source["shot_path"]/"observation-trace"
            refs = {r["index"]:r for r in source["observations"]}
            for offset in (0,150,225):
                p = min(offset,source["last_offset"]); ref = refs[p]
                frames.append({"requested_offset":offset,"observed_offset":p,"identity":ref["identity"],
                               "path":str(obsroot/ref["relative_path"]),"exists":(obsroot/ref["relative_path"]).is_file()})
        entries.append({"state":state["identity"],"family":state["generator_family"],"ordinal":source["ordinal"],
                        "status":source["status"],"source_manifest":source.get("manifest_identity"),"video":video,"frames":frames})
    return {"schema":"issue_76_existing_source_review_v1","plan_identity":plan["identity"],"entries":entries,
            "new_captures":0,"selection":"ordinal1 in each predeclared state, not outcome selected",
            "playback":"existing videos retain source capture timing; no new or edited video claimed"}


def review_html(args, result, inventory, manifest):
    parts = ['<!doctype html><html><head><meta charset="utf-8"><title>Issue76 Phase A review</title>',
             '<style>body{font:15px system-ui;max-width:1150px;margin:24px auto;color:#18202a}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:6px;text-align:left}svg{width:100%;max-width:770px;border:1px solid #ddd}img{width:30%;margin:4px}code{overflow-wrap:anywhere}.note{background:#fff4d8;padding:14px}</style></head><body>',
             '<h1>Issue76: dynamics diagnostics and novelty compatibility</h1>',
             f'<p class="note">{html.escape(result["disposition"])}: pre-access stop, not a fresh experiment. '
             f'Diagnostics complete: {result["diagnostics_complete"]}. No new capture, training or #64 authorization.</p>',
             '<p><a href="curves.csv">Exact per-seed curve values (CSV)</a> · <a href="summary.json">Machine-readable summary</a> · <a href="manifest.json">Source review manifest</a></p>']
    if result.get("per_seed"):
        parts.append('<h2>Error versus elapsed fixed steps</h2><p>Mean carrier MSE; logarithmic y-axis, zero plotted at1e-12. Points are recorded fixed-pair endpoints. Adaptive intermediate curves are unavailable, not interpolated. Local and recursive errors are separate. See CSV for masks and available counts.</p>')
        selected = [f"{arm}_continuous_h{h}" if arm == "hybrid" else f"continuous_h{h}" for arm in ("continuous","hybrid") for h in HORIZONS]
        parts.append(svg_chart("Pure versus hybrid-trained continuous paths (recursive)",{name:mean_series(result,name) for name in selected}))
        for h in HORIZONS:
            names = [f"hybrid_{m}_h{h}" for m in ("continuous","micro","macro")]
            parts.append(svg_chart(f"Hybrid fixed symbolic execution at h{h}",{name:mean_series(result,name) for name in names}))
            local = {f"{name} {kind}":mean_series(result,name,key) for name in (f"continuous_h{h}",f"hybrid_continuous_h{h}")
                     for kind,key in (("recursive","curves"),("local","local_curves"))}
            parts.append(svg_chart(f"Local versus accumulated error at h{h}",local))
        parts.append('<h2>Action ranking and work (development only)</h2><table><tr><th>System</th><th>Mean regret</th><th>Top1</th><th>Linear MACs (all selected states/seeds)</th><th>Failure states</th></tr>')
        for name in (*SYSTEMS,*ADAPTIVE):
            values = [s[name] for s in result["per_seed"].values()]
            parts.append(f'<tr><td>{html.escape(name)}</td><td>{np.mean([v["mean_regret"] for v in values]):.6f}</td><td>{np.mean([v["top1_fraction"] for v in values]):.3f}</td><td>{sum(v["transition_linear_macs"] for v in values):,}</td><td>{sum(v["prediction_failure_states"] for v in values)}</td></tr>')
        parts.append('</table><p>Lower regret is better; top1 is not gameplay success. MACs exclude nonlinear work and are not full FLOPs. Cached adaptive timings are not a paired latency comparison.</p>')
    parts.append('<h2>All45 scenario/novelty cells</h2><p>Metadata-only static slot/action compatibility is not demonstrated recognition, reachability or performance.</p><table><tr><th>Novelty</th><th>Scenario</th><th>Templates</th><th>Static slot/action fit</th><th>Current source templates</th></tr>')
    for cell in inventory["cells"]:
        parts.append(f'<tr><td>{cell["novelty_level"]}: {html.escape(cell["novelty_category"])}</td><td>{html.escape(cell["scenario"])}</td><td>{len(cell["templates"])}</td><td>{cell["static_slot_action_fit_count"]}</td><td>{cell["current_source_count"]}</td></tr>')
    parts.append('</table><h2>Requirement-by-requirement disposition</h2><table><tr><th>Requirement</th><th>Status</th><th>Reason</th></tr>')
    for item in result["requirement_ledger"]:
        parts.append(f'<tr><td>{html.escape(item["requirement"])}</td><td>{html.escape(item["status"])}</td><td>{html.escape(item.get("reason",""))}</td></tr>')
    parts.append('</table><h2>Existing source frames/videos</h2><p>Ordinal1 of each frozen state, selected before diagnostic scores. Requested225 may use an earlier stable-terminal source frame; both offsets are shown. No new frames or videos were generated.</p>')
    for entry in manifest["entries"]:
        parts.append(f'<details><summary>{html.escape(entry["state"])} — {html.escape(entry["family"])}</summary>')
        if entry["video"]:
            link = quote(os.path.relpath(entry["video"],args.review))
            parts.append(f'<p><a href="{link}">Existing source WebM</a></p>')
        for frame in entry["frames"]:
            if frame["exists"]:
                link = quote(os.path.relpath(frame["path"],args.review))
                parts.append(f'<a href="{link}"><img loading="lazy" src="{link}" alt="requested {frame["requested_offset"]}, observed {frame["observed_offset"]}"></a>')
                parts.append(f'<span>requested {frame["requested_offset"]}, observed {frame["observed_offset"]}</span>')
        parts.append('</details>')
    parts.append('</body></html>')
    return "\n".join(parts)


def write_review(args, plan, result):
    args.review.mkdir(parents=True,exist_ok=True)
    manifest = review_manifest(args,plan)
    write(args.review/"manifest.json",manifest)
    write(args.review/"summary.json",compact_report(result))
    (args.review/"curves.csv").write_text(curve_csv(result))
    (args.review/"index.html").write_text(review_html(args,result,plan["inventory_source"],manifest))
    log(f"review gallery ready: {args.review/'index.html'}")


def smoke(args):
    plan = make_plan(args); previous = base.load_plan(source_args(args))
    began = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="novphy-issue76-smoke-") as directory:
        small = copy.copy(args); small.output = Path(directory); small.review = Path(directory)/"review"
        plan = copy.deepcopy(plan); plan["identity"] += ":smoke"
        chosen = [next(s for s in plan["samples"] if s["state"]["generator_family"] == family) for family in ("type010101","type010102")]
        for sample in chosen: sample["sources"] = [s for s in sample["sources"] if s["ordinal"] in (1,12)]
        plan["samples"] = chosen
        budget = covered.Budget(small,{"identity":plan["identity"],"protocol":plan["definitions"]},"diagnostic")
        prepare_targets(small,plan,budget)
        prepare_targets(small,plan,budget)  # target-cache resume, no repeat image parse
        sa = source_args(args); gp = base.grid.load_plan(base.grid_args(sa)); seed = SEEDS[0]
        models = {a:base.load_predictor(sa,previous,seed,a)[0].requires_grad_(False) for a in base.ARMS}
        cp = covered.load_plan(covered_args(args))
        controls = {a+"_adaptive":base.load_control(sa,previous,seed,a)[0] for a in base.ARMS}
        controls.update({"covered_"+a:covered.load_control(covered_args(args),cp,seed,a)[0] for a in base.ARMS})
        records = []
        objective = base.TaskObjective.from_vocabulary(plan["contract"]["vocabulary"])
        for sample in chosen:
            state = sample["state"]; targets = read(target_path(small,state))
            row = base.grid.endpoint(base.grid_args(sa),gp,state["role_ordinal"])
            original = read(args.issue74/f"seed-{seed}"/"development"/f"state-{state['role_ordinal']+1:04d}.json")
            new = read(args.issue75/f"seed-{seed}"/"states"/f"state-{state['role_ordinal']+1:04d}.json")
            for name in ADAPTIVE:
                arm = "continuous" if "continuous" in name else "hybrid"
                cached = new if name.startswith("covered_") else original
                check_cached_adaptive(cached["policies"][name],row,models[arm],controls[name],objective)
                log(f"smoke cached adaptive state={state['role_ordinal']+1} system={name} all12 candidates validated")
            for system,(arm,pair) in SYSTEMS.items():
                for ordinal in (1,12):
                    record = fixed_record(small,plan,previous,sample,targets,row,seed,system,ordinal,models[arm],original)
                    check_curve(record,plan,sample,row,seed,system,ordinal,models[arm])
                    path = curve_path(small,seed,state,system,ordinal); write(path,record)
                    check_curve(read(path),plan,sample,row,seed,system,ordinal,models[arm])
                    target = next(t for t in targets["candidates"] if t["source"]["ordinal"] == ordinal)
                    errors = {str(t):field_errors(record["prediction"]["outputs"].get(str(t)),target["carriers"].get(str(t)),objective) for t in TIMES}
                    records.append({"state":state["identity"],"system":system,"ordinal":ordinal,"failure":record["prediction"]["failure"],"errors":errors,
                                    "unchanged_control225_max_abs_difference":record["unchanged_control225_max_abs_difference"]})
                    log(f"smoke state={state['role_ordinal']+1} system={system} candidate={ordinal} complete")
        report = publication(small,plan)
        if report["diagnostics_complete"]: raise ValueError("smoke must not masquerade as complete diagnostics")
        write(small.output/"result.json",report)
        if read(small.output/"result.json") != report: raise ValueError("report JSON round trip differs")
        write_review(small,plan,report)
        result = {"schema":"issue_76_real_smoke_v1","production_evidence":False,"records":records,
                  "targets":[read(target_path(small,s["state"])) for s in chosen],
                  "inventory_summary":plan["inventory_source"]["summary"],"wall_seconds":time.monotonic()-began,"memory":base.memory(args.device)}
    write(args.output/"smoke.json",result)
    log(f"real smoke complete cells={len(records)} wall={result['wall_seconds']:.1f}s memory={result['memory']}; no full diagnostic or fresh access")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run","smoke-test","prepare","inventory","run-diagnostic","publish","validate","repair-validation"):
        modes.add_argument("--"+mode,action="store_true")
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--review",type=Path,default=REVIEW)
    parser.add_argument("--issue74",type=Path,default=base.OUTPUT)
    parser.add_argument("--issue75",type=Path,default=covered.OUTPUT)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args = parser.parse_args(); torch.set_num_threads(2)
    try:
        if args.dry_run:
            p = make_plan(args)
            log(f"no-write dry-run passed: states={len(p['samples'])} fixed_systems={len(SYSTEMS)} seeds=3 fixed_candidate_records=10368; inventory={p['inventory_source']['summary']}")
            log("fresh access blocked; diagnostic budget3600s/2GiB, no new fits/captures; no files written"); return 0
        if args.smoke_test: smoke(args); return 0
        if args.repair_validation: repair_validation(args); return 0
        if args.prepare:
            if (args.output/"plan.json").exists(): load_plan(args)
            else: write(args.output/"plan.json",make_plan(args))
            log("Phase A protocol frozen; no images scored and no fresh experiment authorized"); return 0
        plan = load_plan(args)
        if args.inventory:
            inventory = build_inventory(ROOT,plan["contract"]["vocabulary"])
            if inventory != plan["inventory_source"]: raise ValueError("inventory changed after freeze")
            write(args.output/"novelty-inventory.json",inventory)
            log(f"metadata inventory complete {inventory['summary']}; no wider runtime/performance claim")
        elif args.run_diagnostic:
            run_diagnostic(args,plan)
        elif args.publish:
            result = publication(args,plan); write(args.output/"result.json",result)
            write(args.output/"summary.json",compact_report(result)); write_review(args,plan,result)
            log(f"published disposition={result['disposition']} diagnostics_complete={result['diagnostics_complete']} recommendation={result['ticket_completion_recommendation']}")
        else:
            result = publication(args,plan)
            if result != read(args.output/"result.json") or compact_report(result) != read(args.output/"summary.json"):
                raise ValueError("published diagnostic report differs from bound source evidence")
            manifest = review_manifest(args,plan)
            if (manifest != read(args.review/"manifest.json") or compact_report(result) != read(args.review/"summary.json")
                    or curve_csv(result).encode("utf-8") != (args.review/"curves.csv").read_bytes()
                    or review_html(args,result,plan["inventory_source"],manifest) != (args.review/"index.html").read_text()):
                raise ValueError("review/CSV/source linkage differs from publication")
            log(f"exact saved-evidence validation passed; diagnostics_complete={result['diagnostics_complete']}; fresh evaluation not run, #64 unauthorized")
        return 0
    except (ValueError,OSError) as error:
        log(f"error: {error}"); return 1


if __name__ == "__main__":
    raise SystemExit(main())
