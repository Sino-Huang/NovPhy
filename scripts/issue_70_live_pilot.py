"""Bounded non-final closed-loop pilot using the existing captured-shot runner."""

from dataclasses import asdict
from html import escape
from io import BytesIO
import json
import math
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from scripts import run_issue_62_successor_cohort as capture
from scripts import run_issue_69_corrected_experiment as old
from scripts import run_issue_70_action_design as experiment
from scripts.smoke_physics_capture import archive_details, start_display, terminate
from world_model.planning.gameplay import (
    CEMConfig,CEMPlanner,SlingshotAction,SlingshotActionBounds,
    PlanningObservation,TerminalStatus,
)
from world_model.planning.task_objective import TaskCandidateEvaluator,TaskObjective


SYSTEMS=("original_cem","corrected_cem","corrected_grid","fixed_prior")
RENDER_REPAIR_PLAN = "pilot-render-repair-plan.json"
RENDER_REPAIR_DIRECTORY = "pilot-rendered-v1"


def prepare_render_repair(args):
    """Disclose one graphics-enabled repeat of the wholly uncaptured pilot."""
    plan=experiment.read(args.output/"pilot-plan.json")
    frozen=experiment.read(args.output/"pilot-freeze.json")
    if plan!=pilot_plan(frozen) or not frozen["pilot_allowed"]:
        raise ValueError("render repair requires the unchanged authorized pilot plan")
    sources=experiment.read(args.output/"pilot-source-inventory.json")
    original=[]
    for level in plan["levels"]:
        level={**level,"authored_birds":sources["authored_birds"][level["ordinal"]]}
        for system in plan["systems"]:
            slot=slot_for(level,system)
            path=Path("pilot-results")/f"{slot['slot_identity']}.json"
            result=experiment.read(args.output/path)
            if (result["slot"]!=slot or result["shots"]!=0 or result["video"] is not None
                    or result["failure"]!="LegacyGroundTruthProtocolError: request-38 record_count: is truncated"):
                raise ValueError("render repair is restricted to the original wholly uncaptured request-38 failure run")
            original.append({"path":str(path),"result":result})
    repair={"schema":"issue_70_render_capture_repair_v1","pilot_plan":plan,
            "reason":"Unity NullGfxDevice from headless=True/-nographics crashed Camera.Render before capture",
            "graphics_enabled":True,"output_directory":RENDER_REPAIR_DIRECTORY,
            "additional_attempts_per_trial":1,"repeat_all_scheduled_trials":True,
            "membership_changed":False,"outcome_conditioned_replacement":False,
            "protocol_deviation_disclosed":True,"original_results":original,
            "final_evaluation_opened":False}
    experiment.write(args.output/RENDER_REPAIR_PLAN,repair)
    experiment.log(f"render repair frozen original_failures={len(original)}; old evidence retained; one new attempt per frozen trial")
    return repair


def pilot_plan(frozen):
    return {"schema":"issue_70_live_plan_v1","freeze":frozen,
        "levels":[{"ordinal":i,"generation_seed":700_500_000+i,
                   "generator_family":("type010101","type010102")[i%2]} for i in range(12)],
        "systems":(["teacher_forced_cem",*SYSTEMS[1:]] if "parser_repair_root" in frozen else list(SYSTEMS)),
        "role":"calibration","final_evaluation_opened":False}


def slot_for(level,system):
    name=f"issue-70-pilot-l{level['ordinal']+1:02d}-{system}"
    return {**level,"role_ordinal":level["ordinal"],"phase":"pilot",
        "exposure_role":"calibration","slot_identity":name,"behavior_policy":system,
        "planned_actions":[{"identity":f"{name}-shot-{j+1}","action_stratum":system,
            "selection_mode":"online_task_planner","drag_x":-10,"drag_y":-80,
            "tap_time_ms":0} for j in range(min(3,level.get("authored_birds",3)))]}


def select_action(system,observation,evaluator,prior,seed):
    bounds=old.probe._bounds()
    # RedBird has no tap ability. Fixed tap is also frozen for CEM.
    red_bounds=SlingshotActionBounds(bounds.drag_x,bounds.drag_y,(0,0),bounds.release_time_ms)
    started=time.monotonic()
    if system=="fixed_prior":
        return SlingshotAction(**prior),{"candidate_count":0,"model_evaluations":0,"predicted_cost":None,"wall_seconds":0.0}
    before=len(evaluator.records)
    if system.endswith("cem"):
        config=CEMConfig(seed=seed,**experiment.contract()["cem"])
        result=CEMPlanner(config,red_bounds,evaluator,progress=experiment.log).plan(observation)
        action=result.actions[0];cost=result.selected_evaluation.total_cost
        iterations=[asdict(i) for i in result.iterations]
        for iteration in iterations:
            iteration["candidate_costs"]=[c if math.isfinite(c) else None for c in iteration["candidate_costs"]]
    else:
        choices=[]
        for candidate in old.probe.broad_action_candidates(observation.identity,bounds):
            evaluation=evaluator.evaluate(observation,(candidate.action,))
            choices.append(evaluation)
        result=min(choices,key=lambda r:r.total_cost)
        action=result.actions[0];cost=result.total_cost;iterations=[]
    records=evaluator.records[before:]
    # Retain the best sampled action across all iterations, including an early
    # sample that later refitting might not reproduce.
    best=min((r for r in records if r["cost"] is not None),key=lambda r:r["cost"])
    action=SlingshotAction(**best["action"]);cost=best["cost"]
    elapsed=time.monotonic()-started
    if elapsed>30:
        raise RuntimeError(f"planner exceeded frozen 30-second budget: {elapsed:.1f}s")
    return action,{"candidate_count":12,"model_evaluations":sum(r["model_evaluations"] for r in records),
        "predicted_cost":cost,"wall_seconds":elapsed,"candidates":records,"iterations":iterations}


def audit_video(args,root,label):
    frames=[];ranges=[]
    for path in sorted(root.glob("shots/shot-*/observation-trace/observation_trace_manifest.json")):
        manifest=experiment.read(path);start=len(frames)
        for frame in manifest["frame_records"]:
            source=path.parent/frame["agent_observation"]["relative_path"]
            if source.is_file():frames.append(source)
        ranges.append({"shot":path.parent.parent.name,"frame_start":start,"frame_end":len(frames)})
    if not frames:return None
    output=args.audit/f"{label}.webm"
    if not output.exists():capture._encode_agent_frames_webm(frames,output)
    return {"path":output.name,"frames":len(frames),"shots":ranges,
            "fps":capture.PILOT_AUDIT_FPS}


def run_trial(args,frozen,level,system):
    sources=experiment.read(args.output/"pilot-source-inventory.json")
    level={**level,"authored_birds":sources["authored_birds"][level["ordinal"]]}
    slot=slot_for(level,system);label=slot["slot_identity"]
    output=Path(getattr(args,"pilot_output",args.output))
    root=output/"pilot"/label
    result_path=output/"pilot-results"/f"{label}.json"
    if result_path.exists():return experiment.read(result_path)
    started=time.monotonic();failure=None;record=None
    try:
        if (root/"trajectory.json").exists():
            record=experiment.read(root/"trajectory.json")
        elif root.exists():
            raise RuntimeError("interrupted fixed single attempt retained as failure")
        else:
            root.mkdir(parents=True)
            adapter=experiment.load_adapter(args.device, frozen.get("parser_repair_root"))
            objective=TaskObjective.from_vocabulary(adapter.model.object_vocabulary)
            if (asdict(objective)!={k:tuple(v) for k,v in frozen["objective"].items()}
                    or adapter.parser_checkpoint_identity!=frozen["parser_identity"]):
                raise ValueError("pilot parser differs from calibration")
            configuration=frozen["original"] if system in ("original_cem","teacher_forced_cem") else frozen["corrected"]
            models=[] if system=="fixed_prior" else [old.load_model(c,args.device)[0] for c in configuration["members"]]
            evaluator=None if not models else TaskCandidateEvaluator(models,objective,old.probe._bounds(),penalty=configuration["penalty"],progress=experiment.log)
            game=output/"pilot-game"
            def choose(reference,bridge,shot_index):
                shot_started=time.monotonic()
                screenshot=bridge.screenshot()
                image=Image.frombytes("RGB",(screenshot.width,screenshot.height),screenshot.rgb)
                buffer=BytesIO();image.save(buffer,format="PNG")
                anchor=(int(reference["gameX"]),int(reference["gameY"]))
                observed=adapter.from_agent_rgb(identity=f"{label}-decision-{shot_index+1}",
                    png=buffer.getvalue(),slingshot_anchor=anchor,terminal_status=TerminalStatus.ONGOING)
                action,evidence=select_action(system,observed,evaluator,frozen["prior_action"],
                                              700_100+10*level["ordinal"]+shot_index)
                evidence.update({"shot":shot_index+1,"observation_identity":observed.identity,
                    "observation_to_action_seconds":time.monotonic()-shot_started,
                    "action":asdict(action),"system":system})
                experiment.write(root/f"decision-{shot_index+1}.json",evidence)
                experiment.log(f"level={level['ordinal']+1}/12 system={system} shot={shot_index+1}/3 selected={asdict(action)} predicted_cost={evidence['predicted_cost']}")
                interface=action.to_interface_action(anchor,old.probe._bounds())
                interface["selection_evidence"]=evidence
                return interface
            record=capture._collect_lineage_attempt(slot,root,game,
                # RGB/aligned capture calls Camera.Render; -nographics uses
                # Unity's NullGfxDevice and crashes before the initial frame.
                release_identity="issue-70-exploratory-pilot-v1",speed=10,headless=False,
                action_selector=choose)
    except Exception as error:
        failure=f"{type(error).__name__}: {error}"
        experiment.log(f"level={level['ordinal']+1} system={system} failed={failure}")
    video=audit_video(args,root,label)
    result={"schema":"issue_70_trial_v1","slot":slot,"failure":failure,
        "success":record is not None and record["terminal_reason"]=="success" and failure is None,
        "terminal_reason":None if record is None else record["terminal_reason"],
        "shots":0 if record is None else record["executed_action_count"],
        "trajectory_identity":None if record is None else record["trajectory_identity"],
        "scenario_lineage_identity":None if record is None else record["scenario_lineage_identity"],
        "level_instance_identity":None if record is None else record["level_instance_identity"],
        "video":video,"wall_seconds":time.monotonic()-started,"final_evaluation_opened":False}
    experiment.write(result_path,result)
    return result


def trial_worker(args,frozen,level,system,connection):
    try:
        torch.set_num_threads(1)
        connection.send(("ok",run_trial(args,frozen,level,system)))
    except Exception as error:
        connection.send(("error",f"{type(error).__name__}: {error}"))
    finally:connection.close()


def check_disjointness(args,plan):
    """Materialize all prospective layouts before gameplay; never screen outcomes."""
    source=experiment.plan_for(args)
    # #68 v2 was itself exactly checked against #62/#57. Also explicitly check
    # current generated identities against each previous release inventory.
    inventories=[]
    for path in (Path(source["release"])/"production-plan.json",
                 capture.DEFAULT_RELEASE/"production-plan.json"):
        inventories.append(experiment.read(path))
    seed_set={level["generation_seed"] for level in plan["levels"]}
    def seeds(value):
        if isinstance(value,dict):
            return ([value["generation_seed"]] if "generation_seed" in value else [])+sum((seeds(v) for v in value.values()),[])
        if isinstance(value,list):return sum((seeds(v) for v in value),[])
        return []
    if any(seed_set&set(seeds(inventory)) for inventory in inventories):
        raise ValueError("pilot generation seeds overlap prior cohorts")
    previous=[]
    for state in inventories[0]["states"]:
        role="cal" if state["exposure_role"]=="calibration" else "ms"
        previous.append(experiment.read(Path(source["release"])/"candidate-results"/
            f"{role}-s{state['role_ordinal']+1:04d}-c{state['carrier_anchor_candidate_ordinal']:02d}.json"))
    previous.append(experiment.read(capture.DEFAULT_RELEASE/"manifest.json"))
    previous.append(experiment.read(capture.ROOT/"data/runtime_evidence/issue-57/cohort-v2-gameplay-success-protocol-v2.json"))
    def source_ids(value):
        if isinstance(value,dict):
            for v in value.values():yield from source_ids(v)
        elif isinstance(value,list):
            for v in value:yield from source_ids(v)
        elif isinstance(value,str) and value.startswith(("level-instance-v1:","scenario-lineage-v1:")):
            yield value
    previous_ids=set(source_ids(previous))
    identities=[];authored_birds=[]
    for level in plan["levels"]:
        root=args.output/"pilot-authorities"/f"level-{level['ordinal']+1:02d}"
        if (root/"scenario-manifest.json").exists():
            manifest=experiment.read(root/"scenario-manifest.json")
        else:
            root.mkdir(parents=True,exist_ok=True)
            capture._materialize_slot(slot_for(level,"authority"),root)
            manifest=experiment.read(root/"scenario-manifest.json")
        identities.append(manifest)
        authored_birds.append(capture._materialized_bird_count(root/"scenario.xml"))
        if set(source_ids(manifest))&previous_ids:
            raise ValueError("pilot source identity overlaps prior cohorts/final levels")
        experiment.log(f"pilot source validation level={level['ordinal']+1}/12 seed={level['generation_seed']}")
    experiment.write(args.output/"pilot-source-inventory.json",{
        "levels":identities,"authored_birds":authored_birds,"seeds":sorted(seed_set),"previous_source_identity_count":len(previous_ids),
        "source_overlap_count":0,"final_evaluation_opened":False})


def run_pilot(args):
    frozen=experiment.read(args.output/"pilot-freeze.json")
    rows=experiment.endpoint_rows(args)
    expected=experiment.freeze_payload(experiment.plan_for(args),rows,experiment.load_scores(args,rows))
    if frozen!=expected or not frozen["pilot_allowed"]:
        raise ValueError("calibration prerequisite gates do not authorize pilot")
    plan=experiment.read(args.output/"pilot-plan.json")
    if plan!=pilot_plan(frozen):raise ValueError("pilot plan differs")
    output=Path(getattr(args,"pilot_output",args.output))
    if output==args.output/RENDER_REPAIR_DIRECTORY:prepare_render_repair(args)
    check_disjointness(args,plan)
    capture._verify_webm_encoder()
    output.mkdir(parents=True,exist_ok=True)
    game=output/"pilot-game"
    if not game.exists():
        archive_details(capture.STAGE_ROOT,game)
    if not (game/"provenance.json").is_file():
        raise ValueError("shared pilot game unpacking is incomplete; inspect pilot-game")
    display_process=None;previous_display=os.environ.get("DISPLAY")
    try:
        if args.start_display:
            display,display_process=start_display(output/"pilot-display.log")
            os.environ["DISPLAY"]=display
        ctx=multiprocessing.get_context("spawn")
        completed=[]
        for level in plan["levels"]:
            for system in plan["systems"]:
                label=slot_for(level,system)["slot_identity"]
                cached=(output/"pilot-results"/f"{label}.json").exists()
                experiment.log(f"pilot level={level['ordinal']+1}/12 system={system} start")
                receive,send=ctx.Pipe(duplex=False)
                process=ctx.Process(target=trial_worker,args=(args,frozen,level,system,send))
                process.start();send.close()
                try:
                    while not receive.poll(10):
                        experiment.log(f"pilot level={level['ordinal']+1} system={system} worker active")
                        if not process.is_alive():raise RuntimeError(f"pilot worker exited {process.exitcode}")
                    try:
                        status,result=receive.recv()
                    except EOFError as error:
                        raise RuntimeError(f"pilot worker ended without a result: level={level['ordinal']+1} system={system}") from error
                finally:receive.close();process.join()
                if status!="ok" or process.exitcode!=0:raise RuntimeError(str(result))
                experiment.log(f"pilot level={level['ordinal']+1}/12 system={system} complete success={result['success']} shots={result['shots']} failure={result['failure']}")
                completed.append(result)
                if not cached and result["failure"] is not None and result["shots"]==0:
                    write_gallery(args,{"trials":completed,"incomplete":True})
                    raise RuntimeError("pilot paused after an uncaptured trial; inspect its engine log before continuing; failed attempt retained")
        write_gallery(args,pilot_report(args))
    finally:
        if display_process is not None:terminate(display_process)
        if previous_display is None:os.environ.pop("DISPLAY",None)
        else:os.environ["DISPLAY"]=previous_display


def pilot_report(args):
    frozen=experiment.read(args.output/"pilot-freeze.json")
    plan=experiment.read(args.output/"pilot-plan.json")
    if plan!=pilot_plan(frozen):raise ValueError("pilot publication plan differs")
    sources=experiment.read(args.output/"pilot-source-inventory.json")
    if sources["seeds"]!=sorted(level["generation_seed"] for level in plan["levels"]) or sources["source_overlap_count"]!=0:
        raise ValueError("pilot source inventory differs")
    output=Path(getattr(args,"pilot_output",args.output))
    repair=prepare_render_repair(args) if output==args.output/RENDER_REPAIR_DIRECTORY else None
    trials=[]
    for level in plan["levels"]:
        level={**level,"authored_birds":sources["authored_birds"][level["ordinal"]]}
        for system in plan["systems"]:
            slot=slot_for(level,system)
            result=experiment.read(output/"pilot-results"/f"{slot['slot_identity']}.json")
            if result["slot"]!=slot or result["final_evaluation_opened"] is not False:
                raise ValueError("pilot result binding differs")
            if result["failure"] is None:
                root=output/"pilot"/slot["slot_identity"]
                trajectory=experiment.read(root/"trajectory.json")
                if (trajectory["trajectory_identity"]!=result["trajectory_identity"]
                        or (trajectory["terminal_reason"]=="success")!=result["success"]
                        or trajectory["executed_action_count"]!=result["shots"]):
                    raise ValueError("pilot result differs from captured trajectory")
                expected=sources["levels"][level["ordinal"]]["scenario_manifest"]
                if (result["scenario_lineage_identity"]!=expected["scenario_lineage"]["identity"]
                        or result["level_instance_identity"]!=expected["level_instance"]["identity"]):
                    raise ValueError("paired gameplay did not execute its frozen level instance")
                for shot in trajectory["shots"]:
                    decision=experiment.read(root/f"decision-{shot['shot_index']+1}.json")
                    action=SlingshotAction.from_interface_action(shot["action"]["interface_action"])
                    if asdict(action)!=decision["action"] or decision["system"]!=system:
                        raise ValueError("executed action differs from logged planner decision")
                if result["video"] is None or not (args.audit/result["video"]["path"]).is_file():
                    raise ValueError("accepted pilot video missing")
            trials.append(result)
    if not any(r["failure"] is None for r in trials):
        raise ValueError("pilot has no completed captures; cannot publish an all-infrastructure-failure run as a gameplay comparison")
    counts={s:{"successes":sum(r["success"] for r in trials if r["slot"]["behavior_policy"]==s),
               "failures":sum(r["failure"] is not None for r in trials if r["slot"]["behavior_policy"]==s)} for s in plan["systems"]}
    differences={}
    for comparator in (plan["systems"][0],"corrected_grid","fixed_prior"):
        x=np.array([float(next(r for r in trials if r["slot"]["ordinal"]==i and r["slot"]["behavior_policy"]=="corrected_cem")["success"])
                    -float(next(r for r in trials if r["slot"]["ordinal"]==i and r["slot"]["behavior_policy"]==comparator)["success"]) for i in range(12)])
        rng=np.random.default_rng(7001)
        samples=x[rng.integers(0,12,size=(10000,12))].mean(1)
        differences[comparator]={"mean_paired_success_difference":float(x.mean()),
                                 "descriptive_95_percent_interval":np.quantile(samples,[0.025,0.975]).tolist()}
    report={"schema":"issue_70_pilot_report_v1","trials":trials,"counts":counts,
            "paired_comparisons":differences,"exploratory_only":True,"issue_64_authorized":False,
            "final_evaluation_opened":False}
    if repair is not None:report["capture_repair"]=repair
    return report


def write_gallery(args,report):
    lines=["<!doctype html><meta charset='utf-8'><title>Issue 70 pilot audit</title>",
           "<h1>Issue 70 exploratory closed-loop pilot</h1>"]
    if report.get("incomplete"):lines.append("<p>Partial pilot: paused after a capture failure. This is not a complete gameplay comparison.</p>")
    if report.get("diagnostic_only"):lines.append("<p>Live rendering smoke test only; these trials are not included in the pilot comparison.</p>")
    if report.get("capture_repair"):lines.append("<p>Graphics-enabled corrective run. Original uncaptured failures are retained and disclosed in the report.</p>")
    for trial in report["trials"]:
        lines.append(f"<h2>{trial['slot']['slot_identity']}</h2><p>Success: {trial['success']}; shots: {trial['shots']}</p>")
        if trial.get("failure"):lines.append(f"<p>Capture failure: {escape(trial['failure'])}</p>")
        if trial["video"]:lines.append(f"<video controls preload='none' width='800' src='{trial['video']['path']}'></video>")
    args.audit.mkdir(parents=True,exist_ok=True)
    (args.audit/"index.html").write_text("\n".join(lines))
    experiment.log(f"pilot gallery {args.audit/'index.html'}")


def smoke_pilot(args):
    """Two actual live shots, separate from production/recovery evidence."""
    from copy import copy
    args=copy(args)
    args.pilot_output=args.output/"rendering-smoke-v1"
    args.audit=experiment.ROOT/"data/issue-70-pilot-rendering-smoke-v1"
    frozen=experiment.read(args.output/"pilot-freeze.json")
    if not frozen["pilot_allowed"]:raise ValueError("live smoke requires an authorized pilot")
    plan=experiment.read(args.output/"pilot-plan.json")
    if plan!=pilot_plan(frozen):raise ValueError("live smoke pilot plan differs")
    check_disjointness(args,plan)
    capture._verify_webm_encoder()
    args.pilot_output.mkdir(parents=True,exist_ok=True)
    game=args.pilot_output/"pilot-game"
    if not game.exists():archive_details(capture.STAGE_ROOT,game)
    if not (game/"provenance.json").is_file():raise ValueError("live smoke game unpacking is incomplete")
    process=None;previous=os.environ.get("DISPLAY")
    try:
        if args.start_display:
            display,process=start_display(args.pilot_output/"display.log");os.environ["DISPLAY"]=display
        results=[]
        for system in ("fixed_prior","corrected_cem"):
            experiment.log(f"live rendering smoke system={system}; diagnostic only")
            result=run_trial(args,frozen,plan["levels"][0],system)
            if result["failure"] is not None or result["shots"]==0 or result["video"] is None:
                raise RuntimeError(f"live rendering smoke failed: {result['failure']}")
            results.append(result)
        write_gallery(args,{"trials":results,"diagnostic_only":True})
        experiment.log("live rendering smoke passed; production/recovery trials unchanged")
    finally:
        if process is not None:terminate(process)
        if previous is None:os.environ.pop("DISPLAY",None)
        else:os.environ["DISPLAY"]=previous


def dry_run():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()))
        def carrier(self,current,action,pair):
            output=current.clone();output[:,2]=torch.sigmoid(action[:,1]*4)+self.anchor*0
            return output
    objective=TaskObjective((0,),(1,))
    for system in SYSTEMS:
        carrier=torch.zeros(197);carrier[2]=1;carrier[15]=1
        evaluator=TaskCandidateEvaluator((Toy(),),objective,old.probe._bounds(),progress=experiment.log)
        prior={"drag_x":-10,"drag_y":-80,"tap_time_ms":0}
        for shot in range(3):
            observed=PlanningObservation(f"dry-{system}-{shot}",carrier,(0,),(100,200))
            action,evidence=select_action(system,observed,evaluator,prior,7001+shot)
            assert old.probe._bounds().contains(action)
            assert evidence["candidate_count"]==(0 if system=="fixed_prior" else 12)
            # The next real observation is deliberately not the imagined endpoint.
            carrier=carrier.clone();carrier[15]=1-shot/3
            experiment.log(f"dry-run system={system} shot={shot+1}/3 reobserved candidate_count={evidence['candidate_count']}")
