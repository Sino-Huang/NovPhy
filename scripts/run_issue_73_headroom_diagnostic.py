"""Read-only headroom, timing, controller and video audit for issue #73."""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
import math
import os
from pathlib import Path
import resource
import time
from urllib.parse import quote

import numpy as np
import torch

from scripts import run_issue_72_matched_grid as matched
from world_model.model import Abstraction, PredictionPair
from world_model.planning.task_objective import TaskObjective
from world_model.training.cnn_hybrid import adaptive_rollout, trajectory_labels


ROOT = matched.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-73-headroom-v1"
GALLERY = ROOT / "data/issue-73-review"
VIDEOS = ROOT / "data/issue-68-ranking-audit-v2/production"
SCHEMA = "issue_73_readonly_diagnostic_v1"
OUTCOMES = ("pig_removed", "level_clear", "pig_contact", "block_contact", "support_change",
            "pig_displacement", "block_displacement")
read, write = matched.read, matched.write


def log(text):
    print(f"[issue-73] {text}", flush=True)


def source_args(args):
    return argparse.Namespace(output=args.issue72, issue71=args.issue71, parser=args.parser, device=args.device)


def definitions():
    return {"calibration_states": 200, "candidates_per_state": 12,
            "best_observed": "minimum accepted settled engine count cost; tie original ordinal; not a deployable/multi-shot oracle",
            "outcomes": list(OUTCOMES), "useful": "any recorded outcome above or discordant accepted count cost",
            "block_destruction": "direct entity_destroyed events for authored block IDs, bounded raw audit only",
            "health_damage": "unavailable unless authoritatively present; never inferred from score/appearance",
            "timing_state_indices": [round(i*199/7) for i in range(8)],
            "training_lineage_indices": list(range(5, 3001, 250)),
            "prediction_fixed_steps": 225, "controller_training_window_steps": 60,
            "gallery_rule": "first two cases in each of missed pig removal, missed count, contact only, no recorded progress; paired best/selected; cap 16 clips; add first-state high arc",
            "followup": "one costed hypothesis; no implementation, training, collection or optimizer sweep",
            "roles": ["training", "calibration"], "exploratory_only": True,
            "fresh_evaluation_opened": False, "final_evaluation_opened": False, "issue_64_authorized": False}


def plan_payload(args):
    old = matched.load_plan(source_args(args))
    result = read(args.issue72 / "result.json")
    if result["plan_identity"] != old["identity"] or result["state_count"] != 200:
        raise ValueError("requires complete source-bound #72 publication")
    return {"schema": SCHEMA, "identity": "issue-73-headroom-plan-v1", "definitions": definitions(),
            "issue72": str(args.issue72.resolve()), "issue71": str(args.issue71.resolve()),
            "parser": str(args.parser.resolve()), "video_root": str(args.videos.resolve()),
            "issue72_plan_identity": old["identity"], "issue72_result_identity": result["identity"],
            "issue72_repair_receipt": result["source_repair_receipt"], "checkpoints": old["checkpoints"],
            "controller_identity": old["controller_identity"], "source_release": old["source_release"],
            "source_plan_identity": old["source_plan_identity"],
            "state_ids": [s["identity"] for s in old["states"]], "prior": result["prior"]["selected"],
            "source_code": Path(__file__).read_text(), "code_revision": old["code"]["revision"],
            "archived_release": False}


def load_plan(args):
    plan = read(args.output / "plan.json")
    if plan != plan_payload(args):
        raise ValueError("diagnostic source/definitions/code differ from freeze; retain evidence")
    return plan


def prepare(args):
    if (args.output / "plan.json").exists():
        load_plan(args); log("existing diagnostic freeze validated"); return
    write(args.output / "plan.json", plan_payload(args))
    log("diagnostic frozen: 200 states, 96 calibration traces, 12 training lineages; no new data/training")


def candidate_record(plan, index, ordinal, endpoint):
    path = Path(plan["source_release"]) / "candidate-results" / f"cal-s{index+1:04d}-c{ordinal:02d}.json"
    value = read(path)
    if (value["state_identity"] != plan["state_ids"][index] or value["candidate_identity"] != endpoint["identity"]
            or value["exposure_role"] != "calibration" or value["plan_identity"] != plan["source_plan_identity"]
            or (value["status"] == "accepted") != endpoint["accepted"]):
        raise ValueError("candidate role/source differs")
    return value


def flags(candidate):
    if candidate["status"] != "accepted":
        return {k: False for k in OUTCOMES}
    coverage = candidate["interaction_coverage"]
    outcome = candidate["endpoint_outcome"]
    return {"pig_removed": "pig_removed" in coverage, "level_clear": "level_clear" in coverage,
            "pig_contact": outcome["pig_contact"], "block_contact": outcome["block_contact"],
            "support_change": outcome["support_change"],
            "pig_displacement": outcome["pig_displacement_world"] > 0,
            "block_displacement": outcome["block_displacement_world"] > 0}


def headroom_state(row, scored, records, prior):
    accepted = [i for i,c in enumerate(row["candidates"]) if c["accepted"]]
    best = min(accepted, key=lambda i:(row["candidates"][i]["realized_count_cost"],i)) if accepted else None
    best_cost = 1e9 if best is None else row["candidates"][best]["realized_count_cost"]
    tied = [i for i in accepted if row["candidates"][i]["realized_count_cost"] == best_cost]
    regress, informative = matched.normalized_regrets(row["candidates"])
    observed = [flags(c) for c in records]
    possible = {k:any(f[k] for f in observed) for k in OUTCOMES}
    policy = {}
    for name in (*matched.ARMS, "prior", "uniform_random", "best_observed"):
        ranking = None
        if name in matched.ARMS:
            ranking = matched.ranked([c["cost"] for c in scored["arms"][name]], row["candidates"])
            selected = ranking["selected"]
        elif name == "prior":
            selected = int(prior.removeprefix("fixed_ordinal_"))-1 if prior.startswith("fixed_ordinal_") else None
        else:
            selected = best if name == "best_observed" else None
        uniform = name == "uniform_random" or (name == "prior" and prior == "uniform_random_expected")
        weights = [(i, 1/12) for i in range(12)] if uniform else ([] if selected is None else [(selected, 1.)])
        outcome_rates = {k: sum(w*observed[i][k] for i,w in weights) for k in OUTCOMES}
        cost = sum(w*row["candidates"][i]["realized_count_cost"] for i,w in weights) if weights else 1e9
        policy[name] = {"selected_ordinal": None if selected is None else selected+1,
                        "penalized_count_cost": cost, "excess_count_cost_over_grid_best": cost-best_cost,
                        "regret": sum(w*regress[i] for i,w in weights) if weights else 1.,
                        "best_set_probability": sum(w*(i in tied) for i,w in weights),
                        "top3": ranking["top3"] if ranking else None,
                        "outcomes": outcome_rates,
                        "missed_opportunities": {k:float(possible[k])*(1-outcome_rates[k]) for k in OUTCOMES},
                        "missed_lower_count_probability": sum(w*(i not in tied) for i,w in weights) if weights else float(bool(accepted))}
    return {"state": row["state"], "best_ordinal": None if best is None else best+1, "best_count_cost": best_cost,
            "best_set_size": len(tied), "informative": informative, "failed_candidates": 12-len(accepted),
            "outcome_opportunities": possible, "no_recorded_progress": bool(accepted) and not informative and not any(possible.values()),
            "policies": policy, "records": [{"ordinal":i+1,"status":c["status"],"flags":observed[i],
                                             "outcome":c["endpoint_outcome"],"audit_manifest":c["audit_manifest"],
                                             "trajectory":c.get("trajectory_relative_path")} for i,c in enumerate(records)]}


def headroom(args, plan):
    source = matched.load_plan(source_args(args)); models, _ = matched.load_models(source_args(args), source)
    began = time.monotonic()
    for i in range(200):
        target = args.output / "headroom" / f"state-{i+1:04d}.json"
        row = matched.endpoint(source_args(args), source, i)
        scored = read(args.issue72 / "states" / f"state-{i+1:04d}.json")
        matched.check_state(scored, row, source, models)
        records = [candidate_record(plan,i,j+1,c) for j,c in enumerate(row["candidates"])]
        value = headroom_state(row, scored, records, plan["prior"])
        if target.exists() and read(target) != value:
            raise ValueError("existing headroom diagnostic differs")
        if not target.exists():write(target, value)
        if (i+1)%10 == 0:
            elapsed=time.monotonic()-began
            log(f"headroom state={i+1}/200 elapsed={elapsed:.1f}s eta={elapsed/(i+1)*(199-i):.1f}s")


def active_counts(sample):
    return {kind: sum(e["lifecycle"] == "active" and e["body_present"] and
                      e["scenario_object_id"].startswith(kind+":") for e in sample["entities"])
            for kind in ("pig", "block")}


def timing_summary(capture, manifest, action):
    samples = capture["fixed_step_samples"]; frames = manifest["frame_records"]
    if [s["fixed_step"] for s in samples] != [f["fixed_step"] for f in frames]:
        raise ValueError("physics/observation timing is misaligned")
    first, last = samples[0]["fixed_step"], samples[-1]["fixed_step"]
    times = {f["fixed_step"]: f["fixed_time_seconds"]-frames[0]["fixed_time_seconds"] for f in frames}
    events = [{"type":e["event_type"],"offset":e["fixed_step"]-first,
               "seconds":times.get(e["fixed_step"]),"participants":e["participants"],"payload":e["payload"]}
              for e in capture["events"]]
    launches = [e for e in events if e["type"] == "bird_launched"]
    collisions = [e for e in events if e["type"] == "collision"]
    target = min(first+225, last)
    at = next(s for s in samples if s["fixed_step"] == target)
    counts = [active_counts(s) for s in (samples[0], at, samples[-1])]
    dt = [frames[i+1]["fixed_time_seconds"]-frames[i]["fixed_time_seconds"] for i in range(len(frames)-1)]
    types = {e["entity_id"]:e["scenario_object_id"].split(":")[0] for s in samples for e in s["entities"]}
    destroyed = {participant for e in events if e["type"] == "entity_destroyed"
                 for participant in e["participants"] if types.get(participant) == "block"}
    damaged = [e for e in events if "damage" in e["type"] or "health" in e["type"]]
    return {"capture_id":capture["capture_id"],"first_step":first,"last_step":last,
            "pre_intervention_offset":capture["pre_intervention_fixed_step"]-first,
            "frames":len(frames),"duration_seconds":times[last],"events":events,
            "launch_offset": launches[0]["offset"] if launches else None,
            "launch_seconds":launches[0]["seconds"] if launches else None,
            "first_collision_offset":collisions[0]["offset"] if collisions else None,
            "hold_milliseconds":action["engine_relative_action"]["hold_milliseconds"],
            "interface_action":action["interface_action"],
            "release_clock_note":"hold milliseconds and simulated fixed-time are distinct; wall-clock speed mapping not inferred",
            "initial_counts":counts[0],"step225_counts":counts[1],"settled_counts":counts[2],
            "step225_differs_from_settled":counts[1]!=counts[2],
            "terminal_after_prediction_steps":max(0,last-first-225),
            "first_60_entirely_prelaunch":bool(launches) and launches[0]["offset"]>60,
            "prelaunch_frame_count":sum(s['fixed_step']-first < launches[0]['offset'] for s in samples) if launches else None,
            "block_destroyed_event_count":len(destroyed),
            "health_damage_evidence":damaged if damaged else "unavailable: no authoritative health/damage fields/events in this audit",
            "consecutive_fixed_steps":all(samples[i+1]['fixed_step']==samples[i]['fixed_step']+1 for i in range(len(samples)-1)),
            "frame_dt_range_seconds":[min(dt),max(dt)] if dt else None,
            "normal_50fps_compatible":bool(dt) and all(abs(x-.02)<.0001 for x in dt),
            "bird_colliders": [c for c in samples[0]['colliders'] if types.get(c['entity_id'])=='bird'],
            "predicted_trajectory_overlay":"unavailable in this replay audit; inspect real bird volume and contacts"}


def trace_at(root, shot, role):
    directory = root / shot["path"]
    capture = read(directory / "physics_capture_v2.json")
    manifest = read(directory / "observation-trace/observation_trace_manifest.json")
    if capture['capture_id'] != shot['capture_id'] or manifest['identity'] != shot['observation_manifest_identity'] or manifest['exposure_role'] != role:
        raise ValueError("raw trace role/source differs")
    return timing_summary(capture, manifest, shot['action'])


def calibration_trace(plan, index, ordinal):
    root = Path(plan['source_release'])
    record = read(root / 'candidate-results' / f'cal-s{index+1:04d}-c{ordinal:02d}.json')
    if record['exposure_role']!='calibration' or record['state_identity']!=plan['state_ids'][index]:
        raise ValueError('timing source differs')
    if record['status']!='accepted':return {'status':record['status'],'state':record['state_identity'],'ordinal':ordinal}
    directory=root/record['trajectory_relative_path']; raw=read(directory/'trajectory.json')
    if raw['trajectory_identity']!=record['trajectory_identity'] or raw['exposure_role']!='calibration':
        raise ValueError('timing trajectory differs')
    result=trace_at(directory,raw['shots'][-1],'calibration')
    if result['settled_counts']!={'pig':record['endpoint_outcome']['active_pigs'],'block':record['endpoint_outcome']['active_blocks']}:
        raise ValueError('settled raw counts differ from accepted outcomes')
    return {'status':'accepted','state':record['state_identity'],'ordinal':ordinal,**result}


def timing(args, plan):
    for i in plan['definitions']['timing_state_indices']:
        for ordinal in range(1,13):
            target=args.output/'timing'/f'cal-{i+1:04d}-c{ordinal:02d}.json'
            if not target.exists():write(target,calibration_trace(plan,i,ordinal))
            log(f'timing state={i+1}/200 candidate={ordinal}/12 complete')


@torch.no_grad()
def observed_and_recursive(model, z, action, length, horizon):
    end=length-length%horizon
    pair=PredictionPair(horizon,Abstraction.CONTINUOUS)
    positions=list(range(0,end,horizon))
    predicted=model.carrier(z[positions],action.expand(len(positions),-1),pair)
    local=float((predicted-z[[p+horizon for p in positions]]).square().mean())
    current=z[:1]
    for _ in positions:current=model.carrier(current,action,pair)
    return {'physical_endpoint':end,'local_mse':local,'recursive_endpoint_mse':float((current-z[end]).square().mean())}


def training_audit(args, plan):
    old=matched.hybrid.load_plan(matched.source_args(source_args(args)))
    model,_=matched.hybrid.load_model(matched.source_args(source_args(args)),old,'corrected')
    controller,_=matched.hybrid.load_controller(matched.source_args(source_args(args)),old,plan['checkpoints']['corrected'])
    source=matched.hybrid.repair.load_plan(args.parser)
    for index in plan['definitions']['training_lineage_indices']:
        target=args.output/'training'/f'lineage-{index:04d}.json'
        if target.exists():log(f'training audit lineage={index} cached');continue
        record=source['training_records'][index-1]
        root=Path(source['training_release']);directory=root/record['path']
        raw=matched.hybrid.repair.trajectory(root,record,source['training_release_identity'])
        shard=torch.load(matched.hybrid.shard_path(matched.source_args(source_args(args)),index),weights_only=True,map_location='cpu')
        if shard['record']!=record or shard['contract']!=old['contract']:raise ValueError('training audit shard differs')
        data=shard['tensors']; windows=[]; traces={}
        for shot in raw['shots']:
            traces[shot['shot_index']]=trace_at(directory,shot,'training')
        for j,window in enumerate(shard['windows']):
            z=data['z'][j].to(args.device); a=data['action'][j:j+1].to(args.device);length=int(data['length'][j])
            metrics={str(h):observed_and_recursive(model,z,a,length,h) for h in (1,5,15)}
            batch={k:v[j:j+1].to(args.device) for k,v in data.items()}
            labels=trajectory_labels(model,batch)[0,:length]
            remaining=torch.arange(length,0,-1,device=z.device)
            with torch.no_grad():chosen=controller(z[:length],a.expand(length,-1),remaining).argmax(-1)
            launch=traces[window['shot']]['launch_offset']
            first_window=not any(w['shot']==window['shot'] for w in shard['windows'][:j])
            mask=data['macros_mask'][j,:length].bool();truth=data['macros'][j,:length].bool()
            windows.append({**window,'length':length,'used_for_controller_labels':first_window,
                            'entirely_prelaunch':launch is not None and window['start']+length<launch,
                            'launch_offset_within_window':None if launch is None else launch-window['start'],
                            'observed_vs_recursive':metrics,'dp_label_counts':dict(Counter(str(matched.PAIRS[int(x)].identity) for x in labels.cpu())),
                            'controller_label_agreement':float((labels==chosen).float().mean()),
                            'available_macro_labels':mask.sum(0).tolist(),'positive_macro_labels':(mask&truth).sum(0).tolist()})
        write(target,{'lineage_index':index,'scenario_lineage_identity':record['scenario_lineage_identity'],
                      'role':'training','shots':list(traces.values()),'windows':windows,
                      'teacher_utility':'duration * carrier MSE + compute penalty + DP continuation; not settled task cost'})
        log(f'training audit lineage={index} shots={len(traces)} windows={len(windows)} complete')


def controller_audit(args, plan):
    source=matched.load_plan(source_args(args));models,controller=matched.load_models(source_args(args),source)
    adapter=matched.hybrid.repair.load_repaired_adapter(args.parser,args.device)
    model=models['corrected_fixed'];objective=TaskObjective.from_vocabulary(source['hybrid_contract']['vocabulary'])
    began=time.monotonic()
    for number,i in enumerate(plan['definitions']['timing_state_indices'],1):
        path=args.output/'controller'/f'state-{i+1:04d}.json'
        if path.exists():log(f'controller state={i+1} cached');continue
        row=matched.endpoint(source_args(args),source,i)
        context,_=matched.perceive(source_args(args),source,i,row,adapter)
        scored={}
        for policy in ('h1','h5','h15','adaptive'):
            evaluations=[]
            for j,candidate in enumerate(row['candidates'],1):
                action=candidate['action'];bounds=matched.hybrid.repair.experiment.old.probe._bounds()
                a=torch.tensor([[action['drag_x']/480.,action['drag_y']/480.,bounds.release_time_ms/1000.,action['tap_time_ms']/1000.,1.]],device=args.device)
                try:
                    prediction,trace=adaptive_rollout(model,controller,context[None].to(args.device),a,225,
                        fixed_pair=None if policy=='adaptive' else PredictionPair(int(policy[1:]),Abstraction.CONTINUOUS))
                    z=prediction[0].cpu(); observed=torch.tensor(candidate['horizon_carrier']) if candidate['horizon_carrier'] is not None else None
                    evaluations.append({'cost':objective(z),'endpoint_mse':None if observed is None else float((z-observed).square().mean()),
                                        'trace':trace,'failure':None})
                except ValueError as e:evaluations.append({'cost':None,'endpoint_mse':None,'trace':[],'failure':str(e)})
                log(f'controller state={i+1}/200 policy={policy} candidate={j}/12 complete')
            ranked=matched.ranked([v['cost'] for v in evaluations],row['candidates'])
            scored[policy]={'ranking':ranked,'candidates':evaluations}
        write(path,{'state':row['state'],'checkpoint_identity':plan['checkpoints']['corrected'],
                    'scope':'same hybrid-trained checkpoint, not pure-continuous baseline','policies':scored})
        elapsed=time.monotonic()-began
        log(f'controller sampled_state={number}/8 elapsed={elapsed:.1f}s eta={elapsed/number*(8-number):.1f}s')


def aggregate_headroom(rows):
    names=rows[0]['policies']; result={}
    for name in names:
        values=[r['policies'][name] for r in rows]
        result[name]={'mean_count_cost':float(np.mean([x['penalized_count_cost'] for x in values])),
                      'mean_regret':float(np.mean([x['regret'] for x in values])),
                      'best_set_fraction':float(np.mean([x['best_set_probability'] for x in values])),
                      'top3_fraction':float(np.mean([x['top3'] for x in values])) if values[0]['top3'] is not None else None,
                      'missed_lower_count_states':sum(x['missed_lower_count_probability'] for x in values),
                      'outcome_states':{k:sum(x['outcomes'][k] for x in values) for k in OUTCOMES},
                      'missed_opportunity_states':{k:sum(x['missed_opportunities'][k] for x in values) for k in OUTCOMES}}
    return {'states':len(rows),'failed_candidates':sum(r['failed_candidates'] for r in rows),
            'informative_states':sum(r['informative'] for r in rows),
            'best_set_size_counts':dict(sorted(Counter(str(r['best_set_size']) for r in rows).items())),
            'no_recorded_progress_states':sum(r['no_recorded_progress'] for r in rows),
            'opportunity_states':{k:sum(r['outcome_opportunities'][k] for r in rows) for k in OUTCOMES},
            'policies':result}


def review_cases(rows):
    categories={
        'missed pig removal':lambda r:r['policies']['adaptive']['missed_opportunities']['pig_removed']>0,
        'missed lower count':lambda r:r['policies']['adaptive']['missed_lower_count_probability']>0,
        'contact without removal':lambda r:r['outcome_opportunities']['pig_contact'] and not r['outcome_opportunities']['pig_removed'],
        'no recorded progress':lambda r:r['no_recorded_progress']}
    result=[(0,9,'predeclared high-arc review')];seen={(0,9)}
    for name,rule in categories.items():
        for i in [i for i,r in enumerate(rows) if rule(r)][:2]:
            observed=rows[i]['best_ordinal']
            if name=='contact without removal':
                observed=next(c['ordinal'] for c in rows[i]['records'] if c['flags']['pig_contact'])
            for ordinal in (observed, rows[i]['policies']['adaptive']['selected_ordinal']):
                if ordinal is not None and (i,ordinal) not in seen:
                    result.append((i,ordinal,name));seen.add((i,ordinal))
    return result[:16]


def gallery(args, plan, rows):
    entries=[]
    for i,ordinal,reason in review_cases(rows):
        log(f'gallery state={i+1} candidate={ordinal} reason={reason}')
        c=rows[i]['records'][ordinal-1]
        timing_path=args.output/'timing'/f'cal-{i+1:04d}-c{ordinal:02d}.json'
        audit=read(timing_path) if timing_path.exists() else calibration_trace(plan,i,ordinal)
        video=None;status='unavailable';source_audit=None
        if c['audit_manifest']:
            source_audit=read(args.videos/c['audit_manifest'])
            if source_audit['state_identity']!=rows[i]['state'] or source_audit['candidate_ordinal']!=ordinal:
                raise ValueError('video source differs')
            relative=source_audit['video_path']
            if relative and (args.videos/relative).is_file():video=str((args.videos/relative).resolve());status='available'
        verified = bool(source_audit and audit.get('normal_50fps_compatible')
                        and source_audit['frame_count']==audit.get('frames') and source_audit['playback_fps']==50)
        entries.append({'state_index':i+1,'state':rows[i]['state'],'candidate_ordinal':ordinal,'reason':reason,
                        'selection':'outcome-stratified diagnostic example; not changed membership',
                        'video':video,'video_status':status,'normal_playback_timing_verified':verified,
                        'source_audit':source_audit,'timing':audit})
    args.gallery.mkdir(parents=True,exist_ok=True)
    write(args.gallery/'manifest.json',{'schema':SCHEMA,'plan_identity':plan['identity'],'entries':entries})
    cards=[]
    for e in entries:
        caption=html.escape(f"State {e['state_index']:04d}, candidate {e['candidate_ordinal']:02d}: {e['reason']}")
        player='<p>Video unavailable; no successful capture is implied.</p>'
        if e['video']:
            link=quote(os.path.relpath(e['video'],args.gallery),safe='/')
            player=f'<video controls preload="metadata" width="840" src="{link}"></video>'
        if not e['normal_playback_timing_verified']:
            player += '<p>Timing or frame completeness not verified; consult the source details.</p>'
        events=e['timing'].get('events',[])
        times='; '.join(f"{x['type']} at {x['seconds']:.2f}s" for x in events if x['seconds'] is not None and x['type'] in ('bird_launched','collision','level_clear','stable_entered'))
        cards.append(f'<section><h2>{caption}</h2>{player}<p>{html.escape(times)}</p><details><summary>Source/timing details</summary><pre>{html.escape(json.dumps(e,indent=2))}</pre></details></section>')
    page='<!doctype html><meta charset="utf-8"><title>Issue 73 replay review</title><h1>Issue 73 diagnostic review</h1><p>Existing agent videos, playback rate 1.0. Times are relative to capture start, not release. See source manifest/frame spacing for sampling and gaps. No new rollout was collected.</p><p>Review actual launch direction, bird-volume/obstacle collision, pig response, high arcs, and settling. A support change is not proof of collapse. Outcome-selected examples are diagnostic only.</p>'+''.join(cards)
    (args.gallery/'index.html').write_text(page)
    log(f'gallery complete clips={len(entries)} path={args.gallery / "index.html"}')
    return entries


def publish(args, plan):
    rows=[read(args.output/'headroom'/f'state-{i+1:04d}.json') for i in range(200)]
    if [r['state'] for r in rows]!=plan['state_ids']:raise ValueError('headroom inventory differs')
    traces=[read(args.output/'timing'/f'cal-{i+1:04d}-c{j:02d}.json') for i in definitions()['timing_state_indices'] for j in range(1,13)]
    accepted=[t for t in traces if t['status']=='accepted']
    training=[read(args.output/'training'/f'lineage-{i:04d}.json') for i in definitions()['training_lineage_indices']]
    control=[read(args.output/'controller'/f'state-{i+1:04d}.json') for i in definitions()['timing_state_indices']]
    first=[w for t in training for w in t['windows'] if w['used_for_controller_labels']]
    later=[w for t in training for w in t['windows'] if not w['used_for_controller_labels']]
    totals=aggregate_headroom(rows)
    timing_summary={'audited_candidates':len(traces),'accepted_candidates':len(accepted),
                    'launch_offset_counts':dict(Counter(str(t['launch_offset']) for t in accepted)),
                    'first_60_prelaunch_candidates':sum(t['first_60_entirely_prelaunch'] for t in accepted),
                    'step225_settled_count_mismatch_candidates':sum(t['step225_differs_from_settled'] for t in accepted),
                    'block_destroyed_events':sum(t['block_destroyed_event_count'] for t in accepted),
                    'health_damage':'see per-trace availability; no score/appearance-based health labels'}
    training_summary={'lineages':len(training),'controller_windows':len(first),'other_windows':len(later),
                      'controller_windows_entirely_prelaunch':sum(w['entirely_prelaunch'] for w in first),
                      'other_windows_entirely_prelaunch':sum(w['entirely_prelaunch'] for w in later),
                      'macro_available_first':np.sum([w['available_macro_labels'] for w in first],axis=0).tolist(),
                      'macro_positive_first':np.sum([w['positive_macro_labels'] for w in first],axis=0).tolist(),
                      'macro_available_later':np.sum([w['available_macro_labels'] for w in later],axis=0).tolist(),
                      'macro_positive_later':np.sum([w['positive_macro_labels'] for w in later],axis=0).tolist(),
                      'controller_dp_agreement':{name:float(np.mean([w['controller_label_agreement'] for w in windows]))
                                                 for name,windows in (('first',first),('later',later))},
                      'observed_vs_recursive':{name:{h:{metric:float(np.mean([w['observed_vs_recursive'][h][metric] for w in windows]))
                                                                   for metric in ('local_mse','recursive_endpoint_mse')}
                                                         for h in ('1','5','15')}
                                               for name,windows in (('first',first),('later',later))}}
    controller_summary={p:{'mean_regret':float(np.mean([r['policies'][p]['ranking']['regret'] for r in control])),
                            'mean_endpoint_mse':float(np.mean([c['endpoint_mse'] for r in control for c in r['policies'][p]['candidates'] if c['endpoint_mse'] is not None]))}
                        for p in ('h1','h5','h15','adaptive')}
    waiting=bool(first) and all(w['entirely_prelaunch'] for w in first)
    recommendation={
        'branch':'source_action_time_and_controller_coverage_audit' if waiting else 'mixed_ranking_and_coverage; no single repair established',
        'evidence_not_causality':'timing/coverage can identify a mismatch; a controlled follow-up is needed to establish benefit',
        'one_proposed_intervention':'Under #74/#75, test one controller-only coverage change: use the existing first AND later windows (including launch/post-launch) for DP labels and the same one aggregation round, instead of first-window-only labels. Keep the predictor, task objective and runtime input unchanged; apply equivalent coverage to the independent continuous baseline controller.',
        'do_not':'Do not initialize counterfactual action scoring from a candidate-specific post-drag/post-launch frame. Do not add an optimizer sweep.',
        'pure_continuous_baseline':'still missing; #74 must train it independently and apply any shared data/time correction symmetrically',
        'followup_budget':{'new_rollouts':0,'new_images':0,'engineering':'one bounded controller-data/utility coverage pilot; cost/refreeze before implementation',
                           'disk':'reuse existing #71 shards; new labels/checkpoints only, estimate <1 GiB',
                           'cpu_ram':'one to three shards; allow ~2 GiB based on prior smoke',
                           'gpu_ram':'small predictor/controller, allow 1 GiB; measure smoke before full run',
                           'runtime':'budget up to one GPU-hour for the single controller pilot; #74 matched baseline training cost separate',
                           'stop_rule':'freeze dataset, objectives, evaluation unit and numerical improvement/compute criteria BEFORE pilot; preserve negative result; no automatic second repair'} }
    return {'schema':SCHEMA,'identity':'issue-73-headroom-diagnostic-v1','plan_identity':plan['identity'],
            'headroom':totals,'timing':timing_summary,'training':training_summary,'controller':controller_summary,
            'recommendation':recommendation,'definitions':definitions(),'full_gameplay_claim':False,
            'source_plan_identity':plan['source_plan_identity'],'checkpoints':plan['checkpoints'],
            'missing_labels_note':'global block destruction/health are not inferred from remaining counts; direct destruction inspected in bounded raw sample',
            'code_revision':plan['code_revision'],'archived_release':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','run','validate','dry-run','smoke-test'):modes.add_argument('--'+name,action='store_true')
    parser.add_argument('--issue72',type=Path,default=matched.OUTPUT)
    parser.add_argument('--issue71',type=Path,default=matched.hybrid.ROOT)
    parser.add_argument('--parser',type=Path,default=matched.hybrid.repair.ROOT)
    parser.add_argument('--videos',type=Path,default=VIDEOS)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--gallery',type=Path,default=GALLERY)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args(argv);torch.set_num_threads(4)
    try:
        if args.dry_run:
            p=plan_payload(args);log(f'no-write dry run passed: {p["definitions"]}');return 0
        if args.smoke_test:
            p=plan_payload(args);source=matched.load_plan(source_args(args));row=matched.endpoint(source_args(args),source,0)
            scored=read(args.issue72/'states/state-0001.json')
            records=[candidate_record(p,0,j+1,c) for j,c in enumerate(row['candidates'])]
            h=headroom_state(row,scored,records,p['prior']);t=calibration_trace(p,0,1)
            assert h['state']==p['state_ids'][0] and t['status']=='accepted'
            log('real metadata/raw-trace smoke passed; not production evidence');return 0
        if args.prepare:prepare(args);return 0
        if args.validate:
            p=load_plan(args)
            headroom(args,p)
            for i in definitions()['timing_state_indices']:
                for ordinal in range(1,13):
                    path=args.output/'timing'/f'cal-{i+1:04d}-c{ordinal:02d}.json'
                    if read(path)!=calibration_trace(p,i,ordinal):raise ValueError('raw timing evidence differs')
                log(f'validate raw timing state={i+1}/200')
            if publish(args,p)!=read(args.output/'result.json'):raise ValueError('diagnostic publication differs')
            g=read(args.gallery/'manifest.json')
            if g['plan_identity']!=p['identity'] or not (args.gallery/'index.html').exists():raise ValueError('gallery missing')
            for e in g['entries']:
                if e['video'] and not Path(e['video']).exists():raise ValueError('gallery video unavailable')
            log('exact diagnostic aggregate/gallery validation passed; no fresh/final authorization');return 0
        prepare(args);p=load_plan(args);began=time.monotonic()
        for i,(name,action) in enumerate((('headroom',headroom),('timing',timing),('training coverage',training_audit),('controller comparisons',controller_audit)),1):
            log(f'stage={i}/5 {name}');action(args,p)
        log('stage=5/5 publication/gallery')
        rows=[read(args.output/'headroom'/f'state-{i+1:04d}.json') for i in range(200)]
        gallery(args,p,rows);result=publish(args,p);write(args.output/'result.json',result)
        log(f'diagnostic complete elapsed={time.monotonic()-began:.1f}s rss_mib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.1f} result={args.output / "result.json"}')
        print(json.dumps({k:result[k] for k in ('headroom','timing','training','controller','recommendation')},indent=2),flush=True)
    except (ValueError,OSError,KeyError) as e:
        log(f'error: {e}');return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
