"""Publish retained agent RGB for every angular assignment, including its failure."""
import argparse
import html
from pathlib import Path
import shutil
import time

from scripts import run_issue_76_angular_replay as angular
from scripts.run_issue_76_compatibility import encode_video


def frame_inventory(root):
    manifest = angular.supervisor.files.read(root / "observation_trace_manifest.json")
    return [{"source_path": str(root / frame["agent_observation"]["relative_path"]),
             "fixed_step": frame["fixed_step"], "fixed_time_seconds": frame["fixed_time_seconds"]}
            for frame in manifest["frame_records"]]


def page(entries):
    parts = ["<!doctype html><meta charset='utf-8'><title>Angular replay evidence</title>",
             "<style>body{font:16px sans-serif;max-width:1000px;margin:30px auto}img,video{max-width:100%}section{border-top:1px solid #aaa;padding:20px 0}</style>",
             "<h1>Angular replay: all 65 assignments</h1>",
             "<p>Training-only pilot: viability failed. No model comparison or advancement. "
             "Failed and censored footage is retained, not counted as a win. "
             "Videos contain every retained agent frame at nominal 50 Hz. A shortened final "
             "capture interval is padded to 20 ms in video playback; exact source timestamps "
             "are in <a href='manifest.json'>the manifest</a>. "
             "The separate decision image precedes the video and is not prepended to it.</p>"]
    for entry in entries:
        parts.extend([f"<section><h2>{html.escape(entry['member_identity'])}</h2>",
                      f"<p>Outcome: {html.escape(entry['outcome'])}; capture validated: {entry['capture_validated']}; "
                      f"paired lineage: {entry['paired_lineage']}. {entry['frame_count']} captured frames, "
                      f"{entry['observed_seconds']:.6f} physical seconds. "
                      f"Decision-to-capture gap: {entry['decision_to_capture_seconds']:.6f} s.</p>",
                      f"<p>Collection failure: {html.escape(str(entry['collection_failure']))}</p>",
                      f"<p>Pre-decision RGB</p><img loading='lazy' width='320' src='{entry['decision_png']}'>",
                      f"<p>First / last retained capture RGB</p><img loading='lazy' width='320' src='{entry['first_png']}'> "
                      f"<img loading='lazy' width='320' src='{entry['last_png']}'>",
                      f"<p><video controls preload='none' width='640' src='{entry['video_path']}'></video></p></section>"])
    return "\n".join(parts) + "\n"


def publish(root=angular.metadata.ROOT, output=angular.metadata.OUTPUT / "media"):
    started = time.monotonic()
    plan = angular.load_plan(root)
    files = angular.supervisor.files
    audit = files.read(root / "collection-validation.json")
    if not audit["all_assigned_records_audited"]:
        raise ValueError("media requires the completed full-assignment audit")
    allowance = audit["remaining_offline_preparation_seconds"] - 1.691
    if output.exists():
        raise ValueError("do not overwrite or silently re-encode an existing media publication")
    output.mkdir(parents=True)
    lookup = {row["member_identity"]: (group, row) for group in audit["groups"] for row in group["rows"]}
    entries = []
    for member in plan["inventory"]["members"]:
        if time.monotonic() - started >= allowance:
            raise ValueError("media exhausted the remaining offline allowance")
        identity = member["identity"]
        group, row = lookup[identity]
        attempt = root / "attempts" / identity
        decision, = frame_inventory(attempt / "decision-1")
        frames = frame_inventory(attempt / "shot-1/observation-trace")
        if not frames or not all(Path(frame["source_path"]).is_file() for frame in [decision, *frames]):
            raise ValueError("retained media inventory contains a missing image")
        deltas = [b["fixed_step"] - a["fixed_step"] for a, b in zip(frames, frames[1:])]
        if any(delta != 50 for delta in deltas[:-1]) or (deltas and not 0 < deltas[-1] <= 50):
            raise ValueError("nominal-cadence video requires regular interior capture intervals")
        names = {}
        for role, frame in (("decision", decision), ("first", frames[0]), ("last", frames[-1])):
            name = identity + "-" + role + ".png"
            shutil.copyfile(frame["source_path"], output / name)
            names[role + "_png"] = name
        video_path = identity + ".webm"
        encode_video([Path(frame["source_path"]) for frame in frames], 50, output / video_path)
        entries.append({"member_identity": identity, "capture_validated": row["capture_contract_validated"],
                        "collection_failure": row["recorded_failure"],
                        "outcome": row.get("invalid_outcome_reason") or row["terminal_reason"],
                        "paired_lineage": group["paired_ranking_admissible"], **names,
                        "decision_frame": decision, "frames": frames, "frame_count": len(frames),
                        "observed_seconds": (frames[-1]["fixed_step"] - frames[0]["fixed_step"]) * .0004,
                        "decision_to_capture_seconds": frames[0]["fixed_time_seconds"] - decision["fixed_time_seconds"],
                        "video_path": video_path, "video_nominal_fps": 50,
                        "video_final_interval_padding_seconds": (50 - deltas[-1]) * .0004 if deltas else 0,
                        "failed_footage_not_rehabilitated": not row["capture_contract_validated"]})
        print(f"Published retained media {len(entries)}/65: {identity}", flush=True)
    elapsed = time.monotonic() - started
    if elapsed > allowance:
        raise ValueError("media exceeded the remaining offline allowance")
    value = {"identity": "issue-76-angular-retained-media-v1", "execution_plan_identity": plan["identity"],
             "audit_identity": audit["identity"], "entries": entries,
             "wall_seconds": elapsed, "prior_targeted_trace_probe_seconds": 1.691,
             "remaining_offline_seconds": allowance - elapsed,
             "bytes": sum(path.stat().st_size for path in output.iterdir() if path.is_file()),
             "source_text": {name: (files.ROOT / name).read_text() for name in
                             ("scripts/publish_issue_76_angular_media.py", "scripts/run_issue_76_compatibility.py")},
             "new_captures": 0, "synthetic_frames": 0, "new_optimizer_updates": 0, "fresh_access": False}
    angular.validation.immutable(output / "manifest.json", value)
    (output / "index.html").write_text(page(entries))
    print({"assignments": len(entries), "wall_seconds": elapsed, "media_bytes": value["bytes"]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    if parser.parse_args().publish:
        publish()
    else:
        print("No-write: --publish encodes all retained agent frames, not new captures.")
