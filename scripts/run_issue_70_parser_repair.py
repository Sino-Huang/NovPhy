"""Train the v2 parser, rebuild h15 carriers, and rerun #70 in separate artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
from io import BytesIO
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from scripts import run_issue_62_successor_cohort as collection
from scripts import run_issue_67_short_unroll as unroll
from scripts import run_issue_70_action_design as experiment
from world_model.data.deployment_temporal import AgentObservation, TemporalObservationContext
from world_model.planning.gameplay import VisualPlanningObservationAdapter, SlingshotAction
from world_model.training.cohort_v2_visual_parser import (
    CohortV2VisualParserConfig, SlotConditionedVisualPredicateParser, ENTITY_KINDS,
)
from world_model.training.lineage_scaling import (
    CarrierKind, CarrierLineage, ContinuousTransitionExample,
    load_carrier_lineage_bundle, save_carrier_lineage_bundle,
)
from world_model.training.short_unroll import (
    train_short_unroll_predictor, save_short_unroll_checkpoint, load_short_unroll_checkpoint,
)

ROOT = experiment.ROOT / ".local-artifacts/issue-70-parser-repair-v2"
PARSER_ID = "issue-70-slot-conditioned-parser-v2:seed-7002001"
CARRIER_ID = VisualPlanningObservationAdapter.identity + ":slot-conditioned-parser-v2"


def log(message):
    print(f"[parser-repair] {message}", flush=True)


class RepairedAdapter(VisualPlanningObservationAdapter):
    identity = CARRIER_ID
    carrier_adapter_identity = CARRIER_ID


def read(path):
    return experiment.read(path)


def write(path, value):
    experiment.write(path, value)


def atomic_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_bundle(path, lineages):
    temporary=path.with_suffix(".tmp.pt")
    # Only discard our interrupted, unpublished temporary bundle.
    temporary.unlink(missing_ok=True)
    save_carrier_lineage_bundle(temporary,lineages)
    os.replace(temporary,path)


def prepare(args):
    manifest = read(collection.DEFAULT_RELEASE / "manifest.json")
    records = [r for r in manifest["trajectories"] if r["exposure_role"] == "training"]
    if len(records) != 3000 or manifest["passed"] is not True:
        raise ValueError("repair requires the accepted 3000-lineage #62 training release")
    legacy = experiment.load_adapter("cpu")
    vocabulary = list(legacy.model.object_vocabulary)
    config = CohortV2VisualParserConfig(seed=7002001, image_height=64, image_width=96,
        hidden_dim=128, epochs=20, batch_size=128, learning_rate=1e-3, device="cuda")
    plan = {"schema": "issue_70_parser_repair_plan_v2", "identity": PARSER_ID,
        "config": asdict(config), "vocabulary": vocabulary,
        "training_release": str(collection.DEFAULT_RELEASE), "training_release_identity": manifest["identity"],
        "training_records": [{k:r[k] for k in ("path","trajectory_identity","scenario_lineage_identity","exposure_role")} for r in records],
        "calibration_release": str(experiment.old.RELEASE),
        "frames_per_shot": 8, "sampling": "uniform index spacing including first and last; every training lineage and shot",
        "loss": {"presence":4.0,"count":1.0,"center":1.0,"kind":1.0,"relations":1.0,"macros":1.0},
        "temperatures": {k:1.0 for k in ("object_presence","contact","supports","steady-state","structure-unstable")},
        "thresholds": {k:0.5 for k in ("object_presence","contact","supports","steady-state","structure-unstable")},
        "carrier_identity": CARRIER_ID, "world_optimizer_examples":8_000_000,
        "calibration_fits_parameters":False,"model_selection_opened":False,"final_evaluation_opened":False}
    write(args.root / "plan.json", plan)
    log("plan frozen training_lineages=3000 parser_epochs=20 snapshots_per_shot=8; old artifacts untouched")


def load_plan(root):
    plan = read(root / "plan.json")
    if (plan["identity"] != PARSER_ID or plan["carrier_identity"] != CARRIER_ID
            or plan["calibration_fits_parameters"] or len(plan["training_records"]) != 3000
            or any(r["exposure_role"] != "training" for r in plan["training_records"])):
        raise ValueError("invalid repaired training plan")
    return plan


def load_shot(root, record, shot):
    """Read already-validated source JSON, never hash or rescan the image corpus."""
    shot_root = root / record["path"] / shot["path"]
    capture = read(shot_root / "physics_capture_v2.json")
    manifest = read(shot_root / "observation-trace/observation_trace_manifest.json")
    if (manifest["exposure_role"] != "training" or capture["capture_id"] != shot["capture_id"]
            or manifest["identity"] != shot["observation_manifest_identity"]):
        raise ValueError("training shot source or role binding differs")
    samples = capture["fixed_step_samples"]
    frames = manifest["frame_records"]
    if [s["fixed_step"] for s in samples] != [f["fixed_step"] for f in frames]:
        raise ValueError("training images and engine targets are misaligned")
    labels = {}
    for item in shot["derivations"]:
        if item["kind"] in ("micro", "macro"):
            raw = read(shot_root / item["path"])
            if raw["identity"] != item["identity"] or [r["fixed_step"] for r in raw["labels"]] != [s["fixed_step"] for s in samples]:
                raise ValueError("training predicate targets are misaligned")
            labels[item["kind"]] = raw["labels"]
    return shot_root, samples, frames, labels


def trajectory(root, record, release_identity):
    value = read(root / record["path"] / "trajectory.json")
    if (value["trajectory_identity"] != record["trajectory_identity"]
            or value["scenario_lineage_identity"] != record["scenario_lineage_identity"]
            or value["exposure_role"] != "training" or not value["complete"]
            or value["release_identity"] != release_identity):
        raise ValueError("training lineage source binding differs")
    return value


def targets(sample, metadata, micro, macro, vocabulary):
    slots = {name:i for i,name in enumerate(vocabulary)}
    n = len(slots)
    presence = torch.zeros(n); centers = torch.zeros(n,2)
    active_ids = {}
    for entity in sample["entities"]:
        if entity["scenario_object_id"] not in slots:
            raise ValueError(f"unseen authored slot: {entity['scenario_object_id']}")
        if entity["lifecycle"] != "active" or not entity["body_present"]:
            continue
        i = slots[entity["scenario_object_id"]]
        presence[i] = 1
        active_ids[entity["entity_id"]] = i
        transform = metadata["world_to_observation_transform"]
        x,y = entity["body"]["position"]
        clip = np.asarray(transform["camera_to_clip_matrix"]).reshape(4,4) @ np.asarray(transform["world_to_camera_matrix"]).reshape(4,4) @ np.array([x,y,0,1])
        if abs(clip[3]) < 1e-9:
            raise ValueError("singular engine projection")
        ndc = clip[:3]/clip[3]
        pixel = np.asarray(transform["ndc_to_observation_matrix"]).reshape(3,3) @ np.array([ndc[0],ndc[1],1])
        viewport = metadata["viewport"]
        centers[i] = torch.tensor(np.clip(pixel[:2]/[viewport["width_pixels"],viewport["height_pixels"]],0,1))
    relations = torch.zeros(n,n,2); relation_mask = torch.zeros(n,n,2,dtype=torch.bool)
    for p,name in enumerate(("contact","supports")):
        label = micro["predicates"][name]
        if label["availability"] != "available":continue
        for a in active_ids.values():
            for b in active_ids.values():
                if a != b:relation_mask[a,b,p] = True
        for a,b in label["relations"]:
            if a in active_ids and b in active_ids:
                relations[active_ids[a],active_ids[b],p] = 1
                if name == "contact":relations[active_ids[b],active_ids[a],p] = 1
    macros = torch.zeros(2); macro_mask = torch.zeros(2,dtype=torch.bool)
    for p,name in enumerate(("steady-state","structure-unstable")):
        label = macro["predicates"][name]
        if label["availability"] == "available":
            macros[p] = float(label["value"]); macro_mask[p] = True
    return {"presence":presence,"centers":centers,"relations":relations,
            "relation_mask":relation_mask,"macros":macros,"macro_mask":macro_mask}


def image_tensor(path, config):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((config.image_width,config.image_height),Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(image,dtype=np.uint8).copy()).permute(2,0,1)


def prepare_shard(plan, record):
    root = Path(plan["training_release"])
    raw = trajectory(root, record, plan["training_release_identity"])
    config = CohortV2VisualParserConfig(**plan["config"])
    rows = []; references = []
    for shot in raw["shots"]:
        shot_root,samples,frames,labels = load_shot(root,record,shot)
        indices = np.linspace(0,len(samples)-1,min(plan["frames_per_shot"],len(samples)),dtype=int)
        for i in indices:
            row = targets(samples[i],frames[i]["capture_metadata"],labels["micro"][i],labels["macro"][i],plan["vocabulary"])
            row["images"] = image_tensor(shot_root/"observation-trace"/frames[i]["agent_observation"]["relative_path"],config)
            rows.append(row)
            references.append({"shot":shot["shot_index"],"fixed_step":samples[i]["fixed_step"],
                               "observation_identity":frames[i]["agent_observation"]["identity"]})
    return {"schema":"issue_70_parser_training_shard_v2","plan_identity":plan["identity"],
        "record":record,"references":references,"tensors":{k:torch.stack([r[k] for r in rows]) for k in rows[0]}}


def prepare_data(args):
    plan = load_plan(args.root); inventory = []; started = time.monotonic()
    manifest_path=args.root/"parser-data.json"
    if manifest_path.exists():
        existing=read(manifest_path)
        if (existing["plan_identity"]!=plan["identity"] or existing["role"]!="training"
                or [s["lineage"] for s in existing["shards"]]!=[r["scenario_lineage_identity"] for r in plan["training_records"]]
                or any(not Path(s["path"]).is_file() for s in existing["shards"])):
            raise ValueError("completed parser data inventory differs")
        log("parser data complete: validated existing 3000-shard inventory");return
    for i,record in enumerate(plan["training_records"],1):
        log(f"parser data lineage={i}/3000 start")
        path = args.root / "parser-data" / f"lineage-{i:04d}.pt"
        value = torch.load(path,weights_only=True) if path.exists() else prepare_shard(plan,record)
        if value["record"] != record or value["plan_identity"] != plan["identity"]:
            raise ValueError("parser shard source differs")
        if not path.exists():atomic_torch(path,value)
        inventory.append({"path":str(path),"frames":len(value["references"]),"lineage":record["scenario_lineage_identity"]})
        elapsed = time.monotonic()-started
        log(f"parser data lineage={i}/3000 frames={inventory[-1]['frames']} elapsed={elapsed:.1f}s eta={elapsed/i*(3000-i):.1f}s")
    write(args.root/"parser-data.json",{"plan_identity":plan["identity"],"role":"training","shards":inventory})


def batches(paths, batch_size, generator):
    pending = []; count = 0
    for index in torch.randperm(len(paths),generator=generator).tolist():
        tensors = torch.load(paths[index],weights_only=True)["tensors"]
        pending.append(tensors); count += len(tensors["images"])
        if count >= batch_size:
            merged = {k:torch.cat([t[k] for t in pending]) for k in tensors}
            order = torch.randperm(count,generator=generator)
            for start in range(0,count,batch_size):
                yield {k:v[order[start:start+batch_size]] for k,v in merged.items()}
            pending=[];count=0
    if pending:
        yield {k:torch.cat([t[k] for t in pending]) for k in pending[0]}


def parser_loss(model, batch, vocabulary):
    output = model(batch["images"])
    present = batch["presence"].bool()
    loss = 4 * F.binary_cross_entropy_with_logits(output["presence_logits"],batch["presence"])
    probability = output["presence_logits"].sigmoid()
    for kind in ("pig","block"):
        indices = [i for i,v in enumerate(vocabulary) if v.startswith(kind+":")]
        loss = loss + F.mse_loss(probability[:,indices].sum(1),batch["presence"][:,indices].sum(1))
    if present.any():
        loss = loss + F.smooth_l1_loss(output["centers"][present],batch["centers"][present])
        kinds = torch.tensor([ENTITY_KINDS.index(v.split(":",1)[0]) for v in vocabulary],device=present.device)
        loss = loss + F.cross_entropy(output["kind_logits"][present],kinds[None].expand(len(present),-1)[present])
    for key,mask,target in (("relation_logits","relation_mask","relations"),("macro_logits","macro_mask","macros")):
        if batch[mask].any():loss = loss + F.binary_cross_entropy_with_logits(output[key][batch[mask]],batch[target][batch[mask]])
    return loss


def train_parser(args):
    plan = load_plan(args.root); config = CohortV2VisualParserConfig(**plan["config"])
    data = read(args.root/"parser-data.json")
    if data["role"] != "training" or len(data["shards"]) != 3000 or [s["lineage"] for s in data["shards"]] != [r["scenario_lineage_identity"] for r in plan["training_records"]]:
        raise ValueError("parser training inventory incomplete or role leakage")
    if (args.root/"parser.pt").exists():
        load_repaired_adapter(args.root,args.device); log("validated existing final parser"); return
    torch.manual_seed(config.seed)
    model = SlotConditionedVisualPredicateParser(config,tuple(plan["vocabulary"])).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    progress_path = args.root/"parser-progress.pt"; first_epoch = 0; reports = []
    if progress_path.exists():
        previous = torch.load(progress_path,map_location="cpu",weights_only=True)
        validate_parser_payload(previous,plan)
        model.load_state_dict(previous["model_state"]); optimizer.load_state_dict(previous["optimizer_state"])
        first_epoch = previous["epoch"]; reports = previous["reports"]
        payload = previous
    for epoch in range(first_epoch,config.epochs):
        model.train(); started=time.monotonic(); total=0.0; seen=0; steps=0
        generator = torch.Generator().manual_seed(config.seed+epoch)
        for batch in batches([s["path"] for s in data["shards"]],config.batch_size,generator):
            batch={k:v.to(args.device) for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True); loss=parser_loss(model,batch,plan["vocabulary"])
            if not torch.isfinite(loss):raise ValueError("nonfinite repaired parser loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            n=len(batch["images"]);seen+=n;steps+=1;total+=float(loss.detach())*n
            if steps%10==0:log(f"parser epoch={epoch+1}/{config.epochs} frames={seen}/{sum(s['frames'] for s in data['shards'])} loss={total/seen:.5f} seconds={time.monotonic()-started:.1f}")
        reports.append({"epoch":epoch+1,"frames":seen,"steps":steps,"mean_loss":total/seen,"wall_seconds":time.monotonic()-started})
        payload={"schema":"issue_70_repaired_parser_checkpoint_v2","identity":PARSER_ID,
            "architecture":model.architecture_identity,"config":plan["config"],"vocabulary":plan["vocabulary"],
            "training_release_identity":plan["training_release_identity"],"epoch":epoch+1,"reports":reports,"device":args.device,
            "model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},"optimizer_state":optimizer.state_dict()}
        atomic_torch(progress_path,payload)
        log(f"parser epoch={epoch+1}/{config.epochs} complete loss={total/seen:.5f}; resume saved")
    payload.pop("optimizer_state")
    atomic_torch(args.root/"parser.pt",payload)
    log("repaired parser trained; calibration audit is required before carrier/model rebuild")


def validate_parser_payload(value,plan):
    if (value["schema"] != "issue_70_repaired_parser_checkpoint_v2" or value["identity"] != PARSER_ID
            or value["architecture"] != SlotConditionedVisualPredicateParser.architecture_identity
            or value["config"] != plan["config"] or value["vocabulary"] != plan["vocabulary"]
            or value["training_release_identity"] != plan["training_release_identity"]
            or not 1 <= value["epoch"] <= plan["config"]["epochs"]
            or len(value["reports"]) != value["epoch"]):
        raise ValueError("repaired parser checkpoint binding differs")


def load_repaired_adapter(root,device):
    plan=load_plan(root);payload=torch.load(root/"parser.pt",map_location="cpu",weights_only=True)
    validate_parser_payload(payload,plan)
    if payload["epoch"]!=plan["config"]["epochs"]:raise ValueError("parser training incomplete")
    model=SlotConditionedVisualPredicateParser(CohortV2VisualParserConfig(**plan["config"]),tuple(plan["vocabulary"]))
    model.load_state_dict(payload["model_state"],strict=True)
    return RepairedAdapter(model.to(device).eval(),parser_checkpoint_identity=PARSER_ID,
        temperatures=plan["temperatures"],thresholds=plan["thresholds"],object_kind_temperature=1.0,
        latent_dim=197,max_entities=15)


def specs_for(root,plan):
    return tuple(replace(s,carrier_identity=CARRIER_ID) for s in unroll._specs(
        optimizer_example_budget=plan["world_optimizer_examples"],batch_size=512,learning_rate=1e-4,
        weight_decay=1e-4,grad_clip=1.0,carrier_bound=2.0,
        lineage_manifest_reference=str(root/"training-carriers.pt")))


def rerun_args(args):
    return argparse.Namespace(output=args.root/"experiment",device=args.device,
        audit=experiment.ROOT/"data/issue-70-pilot-audit-v2",
        summary=experiment.ROOT/"data/runtime_evidence/issue-70/action-design-v2.json",
        start_display=getattr(args,"start_display",False))


def prepare_rerun(args):
    plan=load_plan(args.root); load_repaired_adapter(args.root,args.device)
    source=read(Path(plan["calibration_release"])/"production-plan.json")
    result={"schema":"issue_70_plan_v1","contract":experiment.contract(),
        "release":plan["calibration_release"],"source_plan_identity":source["identity"],
        "states":[s for s in source["states"] if s["exposure_role"]=="calibration"],
        "models":[{"name":f"{s.name}-{s.seed}","kind":"single","seed":s.seed,"spec":asdict(s),
                   "checkpoint":str(unroll._checkpoint_path(args.root/"world-models",s))} for s in specs_for(args.root,plan)],
        "parser_repair_root":str(args.root),"old_result_preserved":str(experiment.OUTPUT)}
    write(args.root/"experiment/plan.json",result)
    log("v2 rerun prepared: repaired parser, 9 matched cells, separate output and videos")


def require_audit(args):
    report=experiment.audit_payload(experiment.endpoint_rows(rerun_args(args)))
    if not report["passed"]:
        raise ValueError("repaired parser still fails objective audit; inspect v2 objective log before retraining world models")


def build_lineage(plan,record,adapter,index):
    root=Path(plan["training_release"]);raw=trajectory(root,record,plan["training_release_identity"])
    transitions=[];ends=[];offset=0
    for shot in raw["shots"]:
        observation_root=root/record["path"]/shot["path"]/"observation-trace"
        manifest=read(observation_root/"observation_trace_manifest.json")
        if manifest["exposure_role"]!="training" or manifest["identity"]!=shot["observation_manifest_identity"]:
            raise ValueError("carrier observations crossed role or source")
        frames=manifest["frame_records"];intervals=len(frames)-1
        if intervals<1 or any(f["fixed_step"]!=frames[0]["fixed_step"]+i for i,f in enumerate(frames)):
            raise ValueError("h15 rebuild requires complete stride-one observation sequence")
        positions=sorted(set([*range(0,intervals,15),intervals]))
        needed=sorted(set(positions+[p-1 for p in positions if p>0]))
        by_position={}
        for start in range(0,len(needed),32):
            indices=needed[start:start+32];observations=[]
            for i in indices:
                f=frames[i];ref=f["agent_observation"]
                observations.append(AgentObservation(ref["identity"],f["fixed_step"],f["fixed_time_seconds"],
                    (observation_root/ref["relative_path"]).read_bytes(),"agent"))
            parsed=adapter.parse_batch(tuple(observations))
            by_position.update(zip(indices,zip(observations,parsed),strict=True))
        carriers={}
        for p in positions:
            observation,parsed=by_position[p];prior,prior_parsed=(None,None) if p==0 else by_position[p-1]
            carriers[p]=adapter.build_from_parsed(TemporalObservationContext(prior,observation),parsed,prior_parsed).tensor
        action=SlingshotAction.from_interface_action(shot["action"]["interface_action"])
        action_tensor=experiment.old.probe._action_tensor(action,experiment.old.probe._bounds(),480)
        for p in range(0,intervals,15):
            target=min(p+15,intervals)
            transitions.append(ContinuousTransitionExample(
                identity=f"repair-l{index}-s{shot['shot_index']}-p{p}-h15",context=carriers[p],action=action_tensor,
                target=carriers[target],physical_diagnostics={},decision_index=offset+p,horizon=15,target_decision_index=offset+target))
        offset+=intervals;ends.append(offset)
    return CarrierLineage(record["trajectory_identity"],record["scenario_lineage_identity"],"training",
        plan["training_release_identity"],CarrierKind.DEPLOYMENT,CARRIER_ID,tuple(transitions),True,offset,tuple(ends))


def rebuild_carriers(args):
    require_audit(args);plan=load_plan(args.root)
    target=args.root/"training-carriers.pt"
    if target.exists():
        existing=load_carrier_lineage_bundle(target)
        if ([l.scenario_lineage_identity for l in existing]!=[r["scenario_lineage_identity"] for r in plan["training_records"]]
                or any(l.carrier_identity!=CARRIER_ID or l.exposure_role!="training" for l in existing)):
            raise ValueError("completed repaired carrier inventory differs")
        log("validated existing complete repaired carrier bundle");return
    adapter=load_repaired_adapter(args.root,args.device)
    paths=[];started=time.monotonic()
    for i,record in enumerate(plan["training_records"],1):
        log(f"carrier lineage={i}/3000 start")
        path=args.root/"carrier-shards"/f"lineage-{i:04d}.pt";paths.append(path)
        if path.exists():
            lineage=load_carrier_lineage_bundle(path)[0]
            if lineage.scenario_lineage_identity!=record["scenario_lineage_identity"] or lineage.carrier_identity!=CARRIER_ID:
                raise ValueError("existing carrier shard source differs")
        else:
            lineage=build_lineage(plan,record,adapter,i)
            atomic_bundle(path,(lineage,))
        elapsed=time.monotonic()-started
        log(f"carrier lineage={i}/3000 h15_windows={len(lineage.transitions)} elapsed={elapsed:.1f}s eta={elapsed/i*(3000-i):.1f}s")
    target=args.root/"training-carriers.pt"
    if not target.exists():
        log("assembling compact h15-only training bundle")
        lineages=tuple(load_carrier_lineage_bundle(path)[0] for path in paths)
        atomic_bundle(target,lineages)
    log("training carriers complete: 3000 full lineages, every shot, h15 grid; no h1 duplication")


def train_world_models(args):
    require_audit(args);plan=load_plan(args.root);specs=specs_for(args.root,plan)
    output=args.root/"world-models";pending=[]
    for i,spec in enumerate(specs,1):
        if unroll._validate_existing_cell(output,spec):log(f"world model={i}/9 validated existing")
        else:pending.append((i,spec))
    if not pending:return
    log("loading compact repaired h15 training bundle")
    lineages=load_carrier_lineage_bundle(args.root/"training-carriers.pt")
    if (len(lineages)!=3000 or [l.scenario_lineage_identity for l in lineages]!=[r["scenario_lineage_identity"] for r in plan["training_records"]]
            or any(l.exposure_role!="training" or l.carrier_identity!=CARRIER_ID for l in lineages)):
        raise ValueError("repaired world-model training bundle differs")
    for i,spec in pending:
        log(f"world model={i}/9 name={spec.name} seed={spec.seed} start")
        model,report=train_short_unroll_predictor(spec,lineages,device=args.device,progress=log)
        path=unroll._checkpoint_path(output,spec);temporary=path.with_suffix(".tmp.pt")
        temporary.unlink(missing_ok=True)
        save_short_unroll_checkpoint(temporary,model,report);os.replace(temporary,path)
        write(unroll._training_report_path(output,spec),unroll._training_payload(report))
        del model;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    log("repaired matched world models complete: 9/9")


def validate_repair(args, *, world_models=True):
    plan=load_plan(args.root);load_repaired_adapter(args.root,"cpu")
    payload=torch.load(args.root/"parser.pt",map_location="cpu",weights_only=True)
    inventory=read(args.root/"parser-data.json")
    expected_frames=sum(s["frames"] for s in inventory["shards"])
    if (len(inventory["shards"])!=3000 or inventory["role"]!="training"
            or [s["lineage"] for s in inventory["shards"]]!=[r["scenario_lineage_identity"] for r in plan["training_records"]]
            or len(payload["reports"])!=plan["config"]["epochs"]
            or any(r["frames"]!=expected_frames for r in payload["reports"])):
        raise ValueError("parser training exposure accounting differs")
    log(f"validate parser epochs={len(payload['reports'])} frames_per_epoch={expected_frames}")
    if not world_models:return
    lineages=load_carrier_lineage_bundle(args.root/"training-carriers.pt")
    if (len(lineages)!=3000 or [l.scenario_lineage_identity for l in lineages]!=[r["scenario_lineage_identity"] for r in plan["training_records"]]
            or any(l.exposure_role!="training" or l.carrier_identity!=CARRIER_ID for l in lineages)):
        raise ValueError("repaired training lineage inventory differs")
    from world_model.training.short_unroll import build_short_unroll_windows
    build_short_unroll_windows(lineages,unroll_steps=4)
    for i,spec in enumerate(specs_for(args.root,plan),1):
        model,report=load_short_unroll_checkpoint(unroll._checkpoint_path(args.root/"world-models",spec),spec,device="cpu")
        if read(unroll._training_report_path(args.root/"world-models",spec))!=unroll._training_payload(report):
            raise ValueError("repaired world-model report differs")
        del model
        log(f"validate repaired checkpoint={i}/9 name={spec.name} seed={spec.seed}")


def smoke_test(args):
    plan=load_plan(args.root)
    log("smoke: extracting all sampled targets from one real training lineage")
    shard=prepare_shard(plan,plan["training_records"][0])
    config=CohortV2VisualParserConfig(**plan["config"])
    torch.manual_seed(config.seed)
    model=SlotConditionedVisualPredicateParser(config,tuple(plan["vocabulary"])).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate)
    batch={k:v.to(args.device) for k,v in shard["tensors"].items()}
    for step in range(2):
        optimizer.zero_grad();loss=parser_loss(model,batch,plan["vocabulary"])
        if not torch.isfinite(loss):raise ValueError("smoke parser loss nonfinite")
        loss.backward();optimizer.step();log(f"smoke parser step={step+1}/2 loss={float(loss.detach()):.5f}")
    adapter=RepairedAdapter(model.eval(),parser_checkpoint_identity=PARSER_ID,
        temperatures=plan["temperatures"],thresholds=plan["thresholds"],object_kind_temperature=1.0,latent_dim=197,max_entities=15)
    lineage=build_lineage(plan,plan["training_records"][0],adapter,1)
    spec=replace(specs_for(args.root,plan)[-1],optimizer_example_budget=4,batch_size=2)
    _,report=train_short_unroll_predictor(spec,(lineage,),device=args.device,progress=log)
    log(f"real smoke passed parser_frames={len(batch['images'])} h15_windows={len(lineage.transitions)} world_examples={report.optimizer_examples}; no weights/data written")


def dry_run():
    config=CohortV2VisualParserConfig(seed=7002001,image_height=8,image_width=8,hidden_dim=8,epochs=1,device="cpu")
    vocabulary=("pig:0000","block:0000")
    torch.manual_seed(config.seed)
    model=SlotConditionedVisualPredicateParser(config,vocabulary)
    batch={"images":torch.randint(0,256,(4,3,8,8),dtype=torch.uint8),
        "presence":torch.tensor([[1.,0.],[0.,1.],[1.,1.],[0.,0.]]),"centers":torch.zeros(4,2,2),
        "relations":torch.zeros(4,2,2,2),"relation_mask":torch.ones(4,2,2,2,dtype=torch.bool),
        "macros":torch.zeros(4,2),"macro_mask":torch.ones(4,2,dtype=torch.bool)}
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.01)
    for step in range(3):
        optimizer.zero_grad();loss=parser_loss(model,batch,vocabulary);loss.backward();optimizer.step()
        log(f"dry parser step={step+1}/3 loss={float(loss.detach()):.5f}")
    lineages=tuple(replace(l,carrier_identity=CARRIER_ID) for l in unroll._fixture_lineages("training",count=2))
    from world_model.model import PredictorConfig
    specs=unroll._specs(optimizer_example_budget=4,batch_size=2,learning_rate=1e-3,weight_decay=0.0,
        grad_clip=1.0,carrier_bound=2.0,lineage_manifest_reference="synthetic-only",
        predictor_config=PredictorConfig(latent_dim=4,action_dim=5,hidden_dim=8,depth=1))
    for i,spec in enumerate(specs,1):
        train_short_unroll_predictor(replace(spec,carrier_identity=CARRIER_ID),lineages,device="cpu",progress=log)
        log(f"dry world cell={i}/9 complete")
    experiment.dry_run()
    log("dry-run passed parser learning, 9 repaired h15 cells, ranking/CEM; files_written=false")


def audit_parser(args):
    result=experiment.audit_payload(experiment.endpoint_rows(rerun_args(args)))
    write(args.root/"experiment/objective-audit.json",result)
    log(f"repaired objective passed={result['passed']} {result['count_ranking']}")
    return result["passed"]


def run_repair(args):
    stages=(("prepare",prepare),("parser data",prepare_data),("train parser",train_parser),
            ("prepare rerun",prepare_rerun),
            ("calibration endpoints",lambda a:experiment.prepare_endpoints(rerun_args(a))))
    for i,(name,run) in enumerate(stages,1):
        log(f"repair stage={i}/10 {name}");run(args)
    log("repair stage=6/10 parser objective audit")
    if not audit_parser(args):
        log("repair stopped: parser objective gate failed; publish/validate the negative report; gameplay remains blocked")
        return
    for i,(name,run) in enumerate((("rebuild carriers",rebuild_carriers),("train matched world models",train_world_models),
            ("score actions",lambda a:experiment.score_models(rerun_args(a))),
            ("freeze pilot",lambda a:experiment.freeze_pilot(rerun_args(a)))),7):
        log(f"repair stage={i}/10 {name}");run(args)
    frozen=read(args.root/"experiment/pilot-freeze.json")
    log(f"repair complete pilot_allowed={frozen['pilot_allowed']}; run-pilot is a separate command")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    for name in ("run-repair","prepare","prepare-parser-data","train-parser","prepare-rerun","prepare-endpoints","audit-objective",
                 "rebuild-carriers","train-world-models","score-models","freeze-pilot","run-pilot","publish","validate","dry-run","smoke-test"):
        mode.add_argument("--"+name,action="store_true")
    parser.add_argument("--root",type=Path,default=ROOT);parser.add_argument("--device",default="cuda")
    parser.add_argument("--start-display",action="store_true")
    args=parser.parse_args(argv);args.root=args.root.resolve();torch.set_num_threads(1)
    try:
        if args.run_repair:run_repair(args)
        elif args.prepare:prepare(args)
        elif args.prepare_parser_data:prepare_data(args)
        elif args.train_parser:train_parser(args)
        elif args.prepare_rerun:prepare_rerun(args)
        elif args.rebuild_carriers:rebuild_carriers(args)
        elif args.train_world_models:train_world_models(args)
        elif args.dry_run:dry_run()
        elif args.smoke_test:smoke_test(args)
        elif args.prepare_endpoints:experiment.prepare_endpoints(rerun_args(args))
        elif args.audit_objective:audit_parser(args)
        elif args.score_models:require_audit(args);experiment.score_models(rerun_args(args))
        elif args.freeze_pilot:experiment.freeze_pilot(rerun_args(args))
        elif args.run_pilot:
            from scripts.issue_70_live_pilot import run_pilot
            run_pilot(rerun_args(args))
        else:
            audit=experiment.audit_payload(experiment.endpoint_rows(rerun_args(args)))
            validate_repair(args,world_models=audit["passed"])
            if audit["passed"]:
                experiment.publish(rerun_args(args),validate=args.validate)
            else:
                result={"schema":"issue_70_parser_repair_result_v2","status":"parser_not_ready",
                    "objective_audit":audit,"world_models_retrained":False,"issue_64_authorized":False,
                    "final_evaluation_opened":False}
                if args.validate:
                    if read(rerun_args(args).summary)!=result:raise ValueError("parser failure publication differs")
                    log("exact negative parser report validation passed")
                else:write(rerun_args(args).summary,result);log("published parser_not_ready; old models not reused")
    except (OSError,ValueError,RuntimeError) as error:
        log(f"error: {error}");return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
