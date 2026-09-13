"""Publish original smoke frames and full-rate review clips; no model scoring."""
import argparse
import html
from pathlib import Path
import shutil
import time

from scripts import run_issue_76_action_replay as run
from scripts.run_issue_76_compatibility import encode_video
from scripts.run_issue_76_order_probe import immutable


def publish():
    plan = run.load_plan()
    output = run.metadata.OUTPUT / "smoke-gallery"
    target = output / "manifest.json"
    if target.exists():
        print("Retaining completed smoke gallery.")
        return
    validated = run.files.read(run.ROOT / "smoke-validation.json")
    if not validated["validated"] or validated["execution_plan_identity"] != plan["identity"]:
        raise ValueError("the exact replay smoke must be validated before continuation publication")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    entries, cards = [], []
    for member in run.selected_members(plan, run.ROOT, True):
        name = member["identity"]
        result = run.files.read(run.ROOT / "results" / (name + ".json"))
        attempt = run.ROOT / "attempts" / name
        snapshot_root = attempt / "decision-1"
        shot_root = attempt / "shot-1/observation-trace"
        snapshot = run.files.read(snapshot_root / run.live.MANIFEST_NAME)
        captured = run.files.read(shot_root / run.live.MANIFEST_NAME)
        png, video = output / (name + ".png"), output / (name + ".webm")
        if not png.exists():
            shutil.copy2(snapshot_root / snapshot["frame_records"][0]["agent_observation"]["relative_path"], png)
        frames = [shot_root / frame["agent_observation"]["relative_path"] for frame in captured["frame_records"]]
        if not video.exists():
            encode_video(frames, 50, video)
        entry = {"member_identity": name, "source_member_identity": member["source_member_identity"],
            "candidate_ordinal": member["candidate_ordinal"], "action": member["actions"][0],
            "decision_png": png.name, "video": video.name, "frame_count": len(frames), "display_fps": 50,
            "physical_first_time": captured["frame_records"][0]["fixed_time_seconds"],
            "physical_last_time": captured["frame_records"][-1]["fixed_time_seconds"],
            "censored": result["segments"][0]["summary"]["censored"], "native_outcome": result["outcome"],
            "source_result_path": str(run.ROOT / "results" / (name + ".json"))}
        entries.append(entry)
        cards.append(f'<section><h2>{html.escape(name)}</h2><p>Assigned action: {html.escape(str(entry["action"]))}; '
                     f'censored: {entry["censored"]}. No learned policy selected this action.</p>'
                     f'<img width="640" src="{png.name}" alt="Original pre-decision agent observation">'
                     f'<video controls width="640" src="{video.name}"></video></section>')
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Training action-replay smoke</title>'
        '<h1>Two assigned training replays</h1><p>Original pre-decision frames and every recorded shot frame at nominal '
        '50 fps. The final displayed frame lasts 20 ms; display duration is not an additional physical observation. '
        'The decision image precedes the video prelaunch frame. No gameplay or advancement advantage is claimed.</p>' + ''.join(cards))
    immutable(target, {"execution_plan_identity": plan["identity"], "entries": entries,
        "publication_wall_seconds": time.monotonic() - started, "source_text": Path(__file__).read_text(),
        "fresh_access": False, "trained_policy_tested": False, "advancement_authorized": False})
    print("Smoke review gallery published.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    if parser.parse_args().publish:
        publish()
    else:
        print("No-write: original smoke frames and timing-disclosed review video only.")


if __name__ == "__main__":
    main()
