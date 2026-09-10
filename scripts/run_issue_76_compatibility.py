"""Run the prospectively frozen #76 broader-condition compatibility smoke."""
from __future__ import annotations

import argparse
import html
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image
import torch

from scripts import issue_76_expansion as expansion
from scripts import run_issue_62_successor_cohort as capture
from scripts import run_issue_70_parser_repair as parser_repair
from scripts.smoke_physics_capture import _process_tree, _stat_fields
from world_model.data.deployment_temporal import AgentObservation, TemporalObservationContext


def log(message):
    print(f"[issue-76-compatibility] {message}",flush=True)


def result_path(output,member):
    return output/"results"/f"{member['identity']}.json"


def process_rss(pid):
    total = 0
    for item in _process_tree(pid):
        try: total += int(_stat_fields(item)[21])*os.sysconf("SC_PAGE_SIZE")
        except FileNotFoundError: pass
    return total/2**20


def terminate_worker(process):
    if not process.is_alive(): return
    children = [pid for pid in _process_tree(process.pid) if pid != process.pid]
    for child in reversed(children):
        try: os.kill(child,signal.SIGTERM)
        except ProcessLookupError: pass
    process.terminate(); process.join(5)
    if process.is_alive(): process.kill(); process.join(5)
    for child in children:
        try: os.kill(child,signal.SIGKILL)
        except ProcessLookupError: pass


def encode_video(frames,fps,path):
    if not frames: return None
    with tempfile.TemporaryDirectory(prefix="issue76-video-") as directory:
        for i,frame in enumerate(frames):
            (Path(directory)/f"frame-{i:06d}.png").symlink_to(frame.resolve())
        subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-framerate",str(fps),
                        "-i",str(Path(directory)/"frame-%06d.png"),"-an","-c:v","libvpx","-deadline","realtime",
                        "-cpu-used","5","-crf","10","-b:v","4M","-pix_fmt","yuv420p",str(path)],check=True)
    return {"path":str(path),"frames":len(frames),"fps":fps,"timing":"recorded fixed-step cadence; no frame subsampling"}


def parser_measurements(shot_root,member,device,parser_root=parser_repair.ROOT):
    physics = expansion.read(shot_root/"physics_capture_v2.json")
    obsroot = shot_root/"observation-trace"
    manifest = expansion.read(obsroot/"observation_trace_manifest.json")
    frames = manifest["frame_records"]; samples = physics["fixed_step_samples"]
    if manifest["exposure_role"] != "calibration" or [f["fixed_step"] for f in frames] != [s["fixed_step"] for s in samples]:
        raise ValueError("compatibility observation/physics synchronization or role differs")
    adapter = parser_repair.load_repaired_adapter(parser_root,device)
    vocabulary = adapter.model.object_vocabulary
    missing = sorted({e["scenario_object_id"] for sample in samples for e in sample["entities"]}-set(vocabulary))
    first = frames[0]["fixed_step"]; last = frames[-1]["fixed_step"]-first
    frame_index = {f["fixed_step"]-first:i for i,f in enumerate(frames)}
    observations,parsed = {},{}
    def parse(i):
        if i not in parsed:
            frame = frames[i]; ref = frame["agent_observation"]
            observations[i] = AgentObservation(ref["identity"],frame["fixed_step"],frame["fixed_time_seconds"],
                                              (obsroot/ref["relative_path"]).read_bytes(),"agent")
            parsed[i] = adapter.parse_batch((observations[i],))[0]
        return parsed[i]
    results = []
    empty_micro = {"predicates":{k:{"availability":"unavailable"} for k in ("contact","supports")}}
    empty_macro = {"predicates":{k:{"availability":"unavailable"} for k in ("steady-state","structure-unstable")}}
    for requested in (0,150,225,"terminal"):
        offset = last if requested == "terminal" else min(requested,last)
        i = frame_index[offset]; frame = frames[i]
        with Image.open(obsroot/frame["agent_observation"]["relative_path"]) as image:
            pixels = np.asarray(image.convert("RGB"))
        entry = {"requested_offset":requested,"observed_offset":offset,"earlier_terminal":requested != "terminal" and offset != requested,
                 "frame_identity":frame["identity"],"agent_png":str(obsroot/frame["agent_observation"]["relative_path"]),
                 "nonconstant_rgb":bool(np.any(np.ptp(pixels.reshape(-1,3),axis=0))),"missing_slots":missing}
        if missing:
            entry["parser_status"] = "unsupported_runtime_slots_no_truncation"
        else:
            current = parse(i); previous = parse(i-1) if i else None
            carrier = adapter.build_from_parsed(TemporalObservationContext(observations.get(i-1) if i else None,observations[i]),current,previous).tensor
            truth = parser_repair.targets(samples[i],frame["capture_metadata"],empty_micro,empty_macro,vocabulary)
            task_indices = [j for j,name in enumerate(vocabulary) if name.startswith(("pig:","block:"))]
            active = [j for j in task_indices if truth["presence"][j] > .5]
            metrics = {}
            for kind in ("pig","block"):
                indices = [j for j,name in enumerate(vocabulary) if name.startswith(kind+":")]
                true_count = float(truth["presence"][indices].sum()); predicted = float(current["presence"][indices].sum())
                metrics.update({kind+"_true_count":true_count,kind+"_predicted_count":predicted,
                                kind+"_count_absolute_error":abs(predicted-true_count)})
            metrics.update({"task_presence_mse":float((current["presence"][task_indices]-truth["presence"][task_indices]).square().mean()),
                "task_presence_threshold_errors":int(((current["presence"][task_indices] >= .5) != (truth["presence"][task_indices] >= .5)).sum()),
                "task_center_absolute_error":float((current["centers"][active]-truth["centers"][active]).abs().mean()) if active else None})
            entry.update({"parser_status":"measured","metrics":metrics,"carrier_finite":bool(torch.isfinite(carrier).all()),
                          "carrier_values":carrier.tolist(),"parser_batch_size":1})
        results.append(entry)
    return {"frames":results,"runtime_missing_slots":missing,"last_offset":last,
            "terminal_reason":physics["terminal_evidence"]["reason"],"physics_capture_id":physics["capture_id"],
            "observation_manifest_identity":manifest["identity"],"parser_identity":adapter.parser_checkpoint_identity,
            "evaluation_only_engine_targets":True,"shared_parser_evaluated_once":True}


def compatibility_checks(measured,limits):
    initial = measured["frames"][0]; metrics = initial.get("metrics",{})
    checks = {"runtime_slots_supported":not measured["runtime_missing_slots"],
              "nonconstant_rgb":all(f["nonconstant_rgb"] for f in measured["frames"]),
              "finite_carriers":all(f.get("carrier_finite",False) and len(f["carrier_values"]) == 236 for f in measured["frames"]),
              "within_fixed_step_limit":measured["last_offset"] <= limits["fixed_step_offset_max"]}
    for key,maximum in limits["parser_criteria"].items():
        value = metrics.get(key.removesuffix("_max"))
        checks[key.removesuffix("_max")] = value is not None and value <= maximum
    return checks


def capture_worker(output,member,limits,device,player):
    torch.set_num_threads(1)
    root = output/"attempts"/member["identity"]; root.mkdir(parents=True)
    game = output/"runtimes"/member["identity"]
    engine = display_process = bridge = None
    started = time.monotonic()
    result = {"schema":"issue_76_compatibility_attempt_v1","member_identity":member["identity"],
              "base_cluster":member["base_cluster"],"novelty_level":member["novelty_level"],
              "generation_seed":member["generation_seed"],"environment_seed":member["environment_seed"],
              "exposure_role":"calibration","failure":None,"measurements":None,"video":None,"checks":{},
              "fresh_evaluation_opened":False,"final_evaluation_opened":False}
    try:
        capture.archive_details(capture.STAGE_ROOT,game)
        if expansion.read(game/"provenance.json") != player["provenance"]:
            raise ValueError("runtime player provenance differs from frozen archive")
        relative = expansion.install_level(game,member)
        result["runtime_xml"] = str(game/relative)
        _,scenario = expansion.materialize(member,member["template"],output/"authorities"/member["identity"])
        if scenario.to_dict() != member["scenario"]: raise ValueError("runtime materialization differs from frozen authority")
        display,display_process = capture.start_display(root/"display.log"); os.environ["DISPLAY"] = display
        aligned = root/"aligned-current"
        ports = []
        while len(ports) < 3:
            port = capture.free_port()
            if port not in ports: ports.append(port)
        agent_port,game_port,physics_port = ports
        result["ports"] = {"agent":agent_port,"game":game_port,"physics":physics_port}
        os.environ["NOVPHY_PHYSICS_CAPTURE_PORT"] = str(physics_port)
        os.environ["NOVPHY_PHYSICS_CAPTURE_V2_STRIDE"] = "1"
        os.environ["NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT"] = str(aligned)
        engine = capture.start_engine(game,False,agent_port=agent_port,game_port=game_port,physics_port=physics_port)
        result["engine_log"] = str(engine.novphy_log_file.name)
        result["graphics_enabled"] = True
        expansion.write(root/"runtime.json",{**result,"engine_pid":engine.pid,"display":display,
                                           "runtime_config":(game/"config.xml").read_text()})
        bridge = capture.connect_with_retry("127.0.0.1",agent_port,timeout=180,deadline_seconds=60)
        bridge.configure(member["environment_seed"],capture.PlayingMode.TRAINING)
        bridge.set_speed(limits["engine_speed"])
        capture.prepare_for_play(bridge,timeout=60,poll_delay=.5)
        if bridge.get_current_level() != 1: raise ValueError("condition-aware runtime did not load its only level")
        planned = {**limits["action"],"selection_mode":"fixed_compatibility_action"}
        prepared = capture.prepare_screen_shot(bridge,lambda observation:capture._resolve_planned_interface_action(planned,observation,bridge),
            frame_height=480,execution_speed=limits["engine_speed"],fast=True,record_ground_truth=True,ground_truth_frequency=1)
        shot_root = root/"shot"
        rollout = member["identity"]+":shot-1"
        capture.capture_physics_v2_rollout(capture.ScienceBirdsBridge("127.0.0.1",physics_port,timeout=180),shot_root,
            shoot=lambda:capture._execute_shot_with_progress(prepared.execute,aligned,lineage_index=member["ordinal"],shot_index=1),
            source_bindings=capture._source_bindings(scenario,rollout_identity=rollout,action_identity=rollout+":action"),
            scenario_manifest_identity=scenario.identity,deadline_seconds=180,aligned_observation_capture_root=aligned,
            observation_configuration=capture.OBSERVATION_CONFIGURATION,
            observation_source_bindings=capture._observation_bindings(scenario,rollout_identity=rollout),
            observation_exposure_role="calibration")
        result["game_state_after"] = bridge.get_game_state().name
        bridge.disconnect(); bridge = None
        capture.stop_started_engine(engine); engine = None
        measured = parser_measurements(shot_root,member,device,output/"parser")
        result["measurements"] = measured
        result["checks"] = compatibility_checks(measured,limits)
        result["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated()/2**20 if device == "cuda" else 0.
        if result["peak_cuda_allocated_mib"] > limits["cuda_allocated_mib"]:
            raise ValueError("parser CUDA allocation exceeded frozen budget")
        manifest = expansion.read(shot_root/"observation-trace/observation_trace_manifest.json")
        frames = manifest["frame_records"]
        delta = frames[1]["fixed_time_seconds"]-frames[0]["fixed_time_seconds"]
        result["video"] = encode_video([shot_root/"observation-trace"/f["agent_observation"]["relative_path"] for f in frames],
                                      1/delta,root/"agent.webm")
    except Exception as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        log(f"attempt {member['ordinal']}/8 failure retained: {result['failure']}")
    finally:
        if bridge is not None: bridge.disconnect()
        capture.stop_started_engine(engine)
        if display_process is not None: capture.terminate(display_process)
    result["wall_seconds"] = time.monotonic()-started
    result["compatibility_passed"] = result["failure"] is None and bool(result["checks"]) and all(result["checks"].values())
    expansion.write(result_path(output,member),result)


def run_smoke(args,plan,pair_only=False):
    budget_path = args.output/"budget.json"
    budget = expansion.read(budget_path) if budget_path.exists() else {
        "plan_identity":plan["identity"],"active_seconds":0.,"peak_cpu_rss_mib":0.,"artifact_bytes":0,"stopped":False}
    if budget["stopped"]: raise ValueError("compatibility budget already stopped; publish the retained partial evidence")
    cap = plan["limits"]; beginning = time.monotonic(); previous_active = budget["active_seconds"]
    members = plan["members"][:2] if pair_only else plan["members"]
    for member in members:
        path = result_path(args.output,member)
        if path.exists():
            log(f"resume attempt={member['ordinal']}/8 retained; no replacement or repeat")
            continue
        attempt_root = args.output/"attempts"/member["identity"]
        if attempt_root.exists():
            expansion.write(path,{"member_identity":member["identity"],"failure":"interrupted_single_attempt_retained_no_retry",
                                 "compatibility_passed":False,"checks":{},"measurements":None,"video":None})
            continue
        if budget["active_seconds"] >= cap["active_seconds"]:
            budget["stopped"] = True; break
        log(f"start attempt={member['ordinal']}/8 cluster={member['base_cluster']} condition={member['novelty_level']}/{member['generator_family']} seed={member['generation_seed']}")
        process = multiprocessing.get_context("spawn").Process(target=capture_worker,args=(args.output,member,cap,args.device,plan["player"]))
        process.start(); started = last_log = time.monotonic(); stop_reason = None
        try:
            while process.is_alive():
                process.join(.25); now = time.monotonic()
                rss = process_rss(os.getpid()); budget["peak_cpu_rss_mib"] = max(budget["peak_cpu_rss_mib"],rss)
                budget["active_seconds"] = previous_active+now-beginning
                frame_count = sum(1 for _ in (attempt_root/"aligned-current").glob("frame_*.json"))
                if now-started > cap["attempt_seconds"]: stop_reason = "attempt_wall_limit"
                if frame_count > cap["fixed_step_offset_max"]+1: stop_reason = "fixed_step_capture_limit"
                if rss > cap["cpu_rss_mib"]: stop_reason = "aggregate_cpu_memory_limit"
                if budget["active_seconds"] > cap["active_seconds"]: stop_reason = "cumulative_wall_limit"
                if now-last_log >= 5:
                    budget["artifact_bytes"] = sum(p.stat().st_size for p in (args.output/"attempts").rglob("*") if p.is_file())
                    if budget["artifact_bytes"] > cap["artifact_bytes"]: stop_reason = "captured_data_limit"
                    completed = sum(result_path(args.output,m).exists() for m in plan["members"])
                    eta = None if completed == 0 else budget["active_seconds"]/completed*(8-completed)
                    log(f"attempt={member['ordinal']}/8 frames={frame_count} attempt_elapsed={now-started:.1f}s total_active={budget['active_seconds']:.1f}s eta={eta} CPU={rss:.1f}MiB")
                    capture._replace_json(budget,budget_path); last_log = now
                if stop_reason:
                    terminate_worker(process); break
        finally:
            if process.is_alive(): terminate_worker(process)
        if not path.exists():
            expansion.write(path,{"member_identity":member["identity"],"failure":stop_reason or f"worker_exit_{process.exitcode}",
                                 "compatibility_passed":False,"checks":{},"measurements":None,"video":None})
        budget["active_seconds"] = previous_active+time.monotonic()-beginning
        budget["artifact_bytes"] = sum(p.stat().st_size for p in (args.output/"attempts").rglob("*") if p.is_file())
        budget["stopped"] = stop_reason in ("cumulative_wall_limit","aggregate_cpu_memory_limit","captured_data_limit")
        capture._replace_json(budget,budget_path)
        result = expansion.read(path)
        log(f"completed attempt={member['ordinal']}/8 compatibility_passed={result['compatibility_passed']} failure={result['failure']} active={budget['active_seconds']:.1f}s")
        if budget["stopped"]: break


def publication(args,plan):
    entries = []
    for member in plan["members"]:
        path = result_path(args.output,member)
        result = expansion.read(path) if path.exists() else None
        if result is not None and result["member_identity"] != member["identity"]: raise ValueError("attempt membership differs")
        if result and result["measurements"]:
            actual = parser_measurements(args.output/"attempts"/member["identity"]/"shot",member,args.device,args.output/"parser")
            if actual != result["measurements"] or compatibility_checks(actual,plan["limits"]) != result["checks"]:
                raise ValueError("saved parser compatibility evidence differs")
        entries.append({"member":{k:v for k,v in member.items() if k not in ("xml","scenario","template")},"result":result})
    complete = all(e["result"] is not None for e in entries)
    report = {"schema":"issue_76_compatibility_report_v1","plan_identity":plan["identity"],"complete":complete,
              "attempts_recorded":sum(e["result"] is not None for e in entries),"scheduled_attempts":8,
              "independent_base_clusters":6,"compatibility_passed":complete and all(e["result"]["compatibility_passed"] for e in entries),
              "entries":entries,"budget":expansion.read(args.output/"budget.json") if (args.output/"budget.json").exists() else None,
              "scope":"development compatibility, not power or model superiority; failures and paired conditions retained",
              "fresh_evaluation_opened":False,"final_evaluation_opened":False,"issue_64_authorized":False}
    report["readiness"] = expansion.readiness(plan,report)
    return report


def gallery(args,report):
    review_root = args.review/f"attempts-{report['attempts_recorded']:02d}"
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Issue76 compatibility</title></head><body>",
             "<h1>#76 broader-condition compatibility smoke</h1>",
             f"<p>Complete: {report['complete']}; compatibility passed: {report['compatibility_passed']}. Development only; fresh/#64/#65 unauthorized.</p>"]
    for entry in report["entries"]:
        member,result = entry["member"],entry["result"]
        parts.append(f"<h2>{html.escape(member['identity'])}: {member['novelty_level']}/{member['generator_family']}</h2>")
        if result is None: parts.append("<p>Not run.</p>"); continue
        parts.append(f"<p>Failure: {html.escape(str(result['failure']))}; checks: {html.escape(str(result['checks']))}</p>")
        if result.get("video"):
            link = os.path.relpath(result["video"]["path"],review_root)
            parts.append(f'<video controls width="640" src="{html.escape(link)}"></video>')
        if result.get("measurements"):
            for frame in result["measurements"]["frames"]:
                link = os.path.relpath(frame["agent_png"],review_root)
                parts.append(f'<p>Requested {frame["requested_offset"]}, observed {frame["observed_offset"]}</p><img width="420" src="{html.escape(link)}">')
        else:
            parts.append(f"<p>Partial files, if any, retained under attempts/{member['identity']}; unavailable media is not replaced.</p>")
    return "\n".join(parts)+"</body></html>\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run","prepare","smoke-test","run-smoke","publish","validate","readiness"):
        modes.add_argument("--"+mode,action="store_true")
    parser.add_argument("--output",type=Path,default=expansion.OUTPUT)
    parser.add_argument("--review",type=Path,default=expansion.REVIEW)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args = parser.parse_args(); torch.set_num_threads(1)
    try:
        if args.dry_run:
            plan = expansion.make_plan(args.output)
            log(f"no-write metadata/materialization dry-run passed: {len(plan['members'])} attempts, six base clusters, five normal scenarios plus appearance pair; missing-slot counts={[len(m['missing_slots']) for m in plan['members']]}")
            return 0
        if args.prepare:
            plan = expansion.make_plan(args.output)
            expansion.write(args.output/"plan.json",plan)
            from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_bytes
            capture._player()  # existing hash-free archive/file-inventory validation, no rendering
            for name in ("plan.json","parser.pt"):
                write_immutable_cohort_v2_bytes((Path(plan["parser_source"])/name).read_bytes(),args.output/"parser"/name)
            for member in plan["members"]:
                root = args.output/"authorities"/member["identity"]
                write_immutable_cohort_v2_bytes(member["xml"].encode(),root/"scenario.xml")
                expansion.write(root/"scenario-manifest.json",member["scenario"])
            log("exact development protocol, condition-aware authorities and exposure bindings frozen before rendering")
            return 0
        plan = expansion.load_plan(args.output)
        if args.smoke_test or args.run_smoke:
            run_smoke(args,plan,args.smoke_test); return 0
        report = publication(args,plan)
        if args.readiness:
            log(str(report["readiness"])); return 0
        page = gallery(args,report)
        if args.publish:
            # Publication may progress from the first pair to all eight attempts;
            # preserve each complete inventory size as a distinct report version.
            root = args.review/f"attempts-{report['attempts_recorded']:02d}"
            expansion.write(root/"report.json",report)
            (root/"index.html").write_text(page)
            log(f"published {root/'index.html'} complete={report['complete']} compatibility_passed={report['compatibility_passed']}")
        else:
            root = args.review/f"attempts-{report['attempts_recorded']:02d}"
            if (expansion.read(root/"report.json") != report or (root/"index.html").read_text() != page):
                raise ValueError("compatibility report/gallery differs from frozen evidence")
            log(f"exact compatibility validation passed; complete={report['complete']}; fresh evaluation unauthorized")
        return 0
    except (ValueError,OSError,RuntimeError) as error:
        log(f"error: {error}"); return 1


if __name__ == "__main__":
    raise SystemExit(main())
