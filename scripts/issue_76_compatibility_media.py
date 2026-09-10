"""Publish retained successful and interrupted #76 compatibility media, without recapture."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from scripts import issue_76_expansion as expansion
from scripts.run_issue_76_compatibility import encode_video, log, result_path


SOURCE = "scripts/issue_76_compatibility_media.py"


def fixed_cadence(frames):
    if len(frames) < 2: return None
    steps = frames[-1]["fixed_step"]-frames[0]["fixed_step"]
    if steps <= 0: raise ValueError("review fixed-step order differs")
    cadence = round((frames[-1]["fixed_time_seconds"]-frames[0]["fixed_time_seconds"])/steps,6)
    # Unity exports float32 absolute times; adjacent differences have microsecond
    # cancellation noise. Fixed steps remain exact and no gaps are interpolated.
    for a,b in zip(frames,frames[1:]):
        if b["fixed_step"]-a["fixed_step"] != 1 or cadence <= 0 or abs(b["fixed_time_seconds"]-a["fixed_time_seconds"]-cadence) > 1e-5:
            raise ValueError("review frames do not have a uniform recorded fixed-time cadence")
    return cadence


def media_inventory(output,review):
    plan = expansion.load_plan(output)
    completed = sum(result_path(output,m).exists() for m in plan["members"])
    root = review/f"attempts-{completed:02d}"
    entries = []
    pig_root = expansion.ROOT/"tasks/task_template_designer/Assets/Resources/Prefabs/GameWorld/Characters/Pigs"
    registered_pigs = sorted(p.stem for p in pig_root.glob("*.prefab"))
    for member in plan["members"]:
        path = result_path(output,member)
        result = expansion.read(path) if path.exists() else None
        entry = {"member_identity":member["identity"],"novelty_level":member["novelty_level"],
                 "status":"not_run" if result is None else ("failed" if result["failure"] else "captured"),
                 "failure":None if result is None else result["failure"],"frames":[],"video":None,"incomplete_capture":False}
        requested_pigs = sorted({p.attrib["type"] for p in ET.fromstring(member["xml"]).findall(".//Pig")})
        entry["resource_diagnosis"] = {"requested_builtin_pigs":requested_pigs,"registered_builtin_pigs":registered_pigs,
            "unresolved_builtin_pigs":sorted(set(requested_pigs)-set(registered_pigs)),
            "scope":"post-capture source/asset diagnosis, not a model-performance judgment or replacement rule"}
        attempt = output/"attempts"/member["identity"]
        finalized = attempt/"shot/observation-trace/observation_trace_manifest.json"
        if finalized.exists():
            manifest = expansion.read(finalized)
            if manifest["exposure_role"] != "calibration": raise ValueError("review source is not compatibility development")
            entry["frames"] = [{"path":str(finalized.parent/f["agent_observation"]["relative_path"]),
                                "fixed_step":f["fixed_step"],"fixed_time_seconds":f["fixed_time_seconds"]} for f in manifest["frame_records"]]
            entry["video"] = result["video"] if result else None
        else:
            partial = sorted((attempt/"aligned-current").glob("frame_*.json"))
            for metadata_path in partial:
                metadata = expansion.read(metadata_path)
                png = metadata_path.with_suffix(".png")
                if png.exists():
                    entry["frames"].append({"path":str(png),"fixed_step":metadata["fixed_step"],
                                            "fixed_time_seconds":metadata["fixed_time_seconds"],"metadata":str(metadata_path)})
            if partial: entry["incomplete_capture"] = True
        frames = entry["frames"]
        if frames:
            if not all(Path(f["path"]).is_file() for f in frames): raise ValueError("retained review frame missing")
            cadence = fixed_cadence(frames)
            entry["first_fixed_step"] = frames[0]["fixed_step"]
            entry["last_observed_offset"] = frames[-1]["fixed_step"]-frames[0]["fixed_step"]
            entry["watchdog_offset_overshoot"] = max(0,entry["last_observed_offset"]-plan["limits"]["fixed_step_offset_max"])
            if entry["incomplete_capture"] and cadence:
                entry["video"] = {"path":str(root/f"{member['identity']}-partial.webm"),"frames":len(frames),"fps":1/cadence,
                                  "timing":"recorded fixed-step cadence; incomplete capture, not settled outcome"}
        runtime = attempt/"runtime.json"
        entry["runtime_receipt"] = str(runtime) if runtime.exists() else None
        entry["unity_logs"] = [str(p) for p in sorted((output/"runtimes"/member["identity"]).glob("sciencebirds_*.log"))]
        entries.append(entry)
    return root,{"schema":"issue_76_retained_compatibility_media_v1","plan_identity":plan["identity"],"entries":entries,
                 "new_captures":0,"failure_outcomes_reclassified_as_success":False,
                 "source_text":{SOURCE:(expansion.ROOT/SOURCE).read_text()}}


def page(root,manifest):
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Issue76 retained failure media</title></head><body>",
             "<h1>#76 compatibility: all retained attempts</h1><p>Incomplete captures are not settled outcomes or model-performance results. No recapture, replacement, or fabricated frames.</p>"]
    for entry in manifest["entries"]:
        parts.append(f"<h2>{html.escape(entry['member_identity'])}: {entry['status']}</h2><p>{html.escape(str(entry['failure']))}</p>")
        parts.append(f"<p>Resource diagnosis: {html.escape(str(entry['resource_diagnosis']))}</p>")
        frames = entry["frames"]
        if frames:
            parts.append(f"<p>Retained frames: {len(frames)}; last observed offset {entry['last_observed_offset']}; watchdog overshoot {entry['watchdog_offset_overshoot']}; incomplete={entry['incomplete_capture']}</p>")
            if entry["video"]:
                parts.append(f'<video width="640" controls src="{html.escape(os.path.relpath(entry["video"]["path"],root))}"></video>')
            for index in sorted({0,len(frames)//2,len(frames)-1}):
                frame = frames[index]
                parts.append(f'<p>Actual fixed step {frame["fixed_step"]}</p><img width="420" src="{html.escape(os.path.relpath(frame["path"],root))}">')
        for path in entry["unity_logs"]:
            parts.append(f'<p><a href="{html.escape(os.path.relpath(path,root))}">Retained Unity log</a></p>')
    return "\n".join(parts)+"</body></html>\n"


def check_video(video):
    value = json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-count_frames",
        "-show_entries","stream=nb_read_frames,r_frame_rate,time_base","-of","json",video["path"]],text=True))["streams"][0]
    numerator,denominator = map(float,value["r_frame_rate"].split("/"))
    tick_numerator,tick_denominator = map(float,value["time_base"].split("/"))
    duration_error = abs(video["frames"]/(numerator/denominator)-video["frames"]/video["fps"])
    # WebM stores a rational frame rate and millisecond timestamps. Require
    # exact frame count and agreement within one declared container time tick.
    if int(value["nb_read_frames"]) != video["frames"] or duration_error > tick_numerator/tick_denominator:
        raise ValueError("review video frame count/timing differs from retained source")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--publish",action="store_true"); mode.add_argument("--validate",action="store_true")
    parser.add_argument("--output",type=Path,default=expansion.OUTPUT)
    parser.add_argument("--review",type=Path,default=expansion.REVIEW)
    args = parser.parse_args()
    try:
        root,manifest = media_inventory(args.output,args.review)
        rendered = page(root,manifest)
        if args.publish:
            root.mkdir(parents=True,exist_ok=True)
            for entry in manifest["entries"]:
                if entry["incomplete_capture"] and entry["video"] and not Path(entry["video"]["path"]).exists():
                    log(f"encoding retained incomplete media: {entry['member_identity']} frames={len(entry['frames'])}")
                    encode_video([Path(f["path"]) for f in entry["frames"]],entry["video"]["fps"],Path(entry["video"]["path"]))
                if entry["video"]: check_video(entry["video"])
            expansion.write(root/"media.json",manifest)
            (root/"media.html").write_text(rendered)
            log(f"retained success/failure gallery: {root/'media.html'}")
        else:
            if expansion.read(root/"media.json") != manifest or (root/"media.html").read_text() != rendered:
                raise ValueError("retained media source/gallery differs")
            if any(e["video"] and not Path(e["video"]["path"]).is_file() for e in manifest["entries"]):
                raise ValueError("retained review video missing")
            for entry in manifest["entries"]:
                if entry["video"]: check_video(entry["video"])
            log("retained success/failure media validation passed; no recapture")
        return 0
    except (ValueError,OSError) as error:
        log(f"error: {error}"); return 1


if __name__ == "__main__": raise SystemExit(main())
