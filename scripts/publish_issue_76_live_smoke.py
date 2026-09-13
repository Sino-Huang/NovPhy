"""Reviewable engineering evidence; no trained-model or advancement inference."""
import argparse
import html
from pathlib import Path
import shutil

from scripts import run_issue_76_live_smoke as smoke
from scripts.issue_76_native_outcomes import terminal_evidence, gameplay_outcome
from scripts.run_issue_76_compatibility import encode_video

files = smoke.files
OUTPUT = files.ROOT / "data/issue-76-live-engineering"


def report():
    original = files.read(smoke.OUTPUT / "plan.json")
    correction = files.read(smoke.OUTPUT / "observation-json-correction.json")
    plan = correction["plan"]
    if plan != smoke.make_plan():
        raise ValueError("live engineering correction source or membership changed")
    entries = []
    for member in plan["members"]:
        result = files.read(smoke.OUTPUT / "results" / (member["identity"] + ".json"))
        observations, previous_time, shots = 0, None, []
        if result.get("segments"):
            derived = gameplay_outcome(result, [terminal_evidence(s) for s in result["segments"]])
            if derived != result["outcome"] or derived["success"] != result["gameplay_success"]:
                raise ValueError("live outcome differs from native terminal evidence")
        for index, segment in enumerate(result.get("segments", []), 1):
            attempt = smoke.OUTPUT / "attempts" / member["identity"]
            decision_root = attempt / f"decision-{index}"
            decision = files.read(decision_root / smoke.live.MANIFEST_NAME)
            observed_root = attempt / f"shot-{index}/observation-trace"
            observed = files.read(observed_root / smoke.live.MANIFEST_NAME)
            if result["decisions"][index - 1]["action"] != smoke.ACTION:
                raise ValueError("engineering action differs from prospective policy")
            if (result["decisions"][index - 1]["observation_manifest"] != decision["identity"]
                    or segment["observation_manifest"] != observed["identity"]):
                raise ValueError("decision/segment observation identities differ")
            frames = decision["frame_records"] + observed["frame_records"]
            for frame in frames:
                now = frame["fixed_time_seconds"]
                if previous_time is not None and now <= previous_time:
                    raise ValueError("live observation clock is not strictly increasing across shots")
                previous_time = now
            observations += len(frames)
            names = [str(observed_root / f["agent_observation"]["relative_path"])
                     for f in observed["frame_records"]]
            shots.append(dict(ordinal=index, decision_png=str(decision_root /
                decision["frame_records"][0]["agent_observation"]["relative_path"]),
                frame_paths=names, first_time=observed["frame_records"][0]["fixed_time_seconds"],
                last_time=observed["frame_records"][-1]["fixed_time_seconds"],
                native_terminal=terminal_evidence(segment), censored=segment["summary"]["censored"]))
        if result["complete"]:
            if (result["policy"]["observations"] != observations
                    or result["policy"]["executed_action_events"] != len(shots)):
                raise ValueError("streaming history event counts differ from actual observations/actions")
        entries.append(dict(member_identity=member["identity"], generator_family=member["generator_family"],
            generation_seed=member["generation_seed"], engine_seed=member["engine_seed"],
            result=result, shots=shots))
    if entries[0]["result"] != correction["original_failed_result"]:
        raise ValueError("first failed assignment changed after technical correction")
    return dict(schema="issue_76_live_engineering_report_v1", plan_identity=plan["identity"],
        original_source_text=original["source_text"], corrected_source_text=plan["source_text"],
        publisher_source=Path(__file__).read_text(), entries=entries,
        assigned=len(entries), complete=sum(e["result"]["complete"] for e in entries),
        gameplay_successes=sum(e["result"]["gameplay_success"] for e in entries),
        budget=files.read(smoke.OUTPUT / "budget.json"), fitting_excluded=True,
        trained_model_used=False, fresh_access=False, issue_64_authorized=False,
        limitation="Engineering only: fixed actions and untrained history; no model/compute/power comparison.")


def publish(value):
    target = OUTPUT / "report.json"
    if target.exists():
        if files.read(target) != value:
            raise ValueError("do not replace different published engineering evidence")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sections = []
    for entry in value["entries"]:
        name = entry["member_identity"]
        result = entry["result"]
        sections.append(f"<h2>{html.escape(name)}</h2><p>Complete: {result['complete']}; "
                        f"success: {result['gameplay_success']}; failure: {html.escape(str(result['failure']))}</p>")
        if not entry["shots"]:
            sections.append("<p>Failed before a shot; no persisted agent image available.</p>")
        for shot in entry["shots"]:
            stem = name + f"-shot-{shot['ordinal']}"
            png, video = stem + ".png", stem + ".webm"
            shutil.copy2(shot["decision_png"], OUTPUT / png)
            encode_video([Path(p) for p in shot["frame_paths"]], 50, OUTPUT / video)
            sections.append(f'<p>Shot {shot["ordinal"]}; native terminal: {html.escape(str(shot["native_terminal"]))}; '
                f'censored: {shot["censored"]}</p><img width="480" src="{png}" alt="Pre-decision agent RGB">'
                f'<video controls width="480" src="{video}"></video>')
    document = '<!doctype html><meta charset="utf-8"><title>Issue 76 live engineering</title>'
    document += '<h1>Native live-interface engineering</h1><p>Fixed actions, untrained history; excluded from fitting '
    document += 'and fresh evaluation. Videos show every saved shot frame at nominal 50fps. The forced final '
    document += 'off-grid frame may shorten the final native interval; display timing is not wall-runtime timing. '
    document += 'The separate pre-decision snapshot precedes the video prelaunch frame.</p>' + "\n".join(sections)
    # Publishing generated artifacts, not editing maintained source files.
    (OUTPUT / "index.html").write_text(document)
    files.write(target, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    value = report()
    if args.publish:
        publish(value)
    elif files.read(OUTPUT / "report.json") != value:
        raise ValueError("published report differs from observed evidence")
    print(f"live engineering evidence: {value['complete']}/{value['assigned']} complete, "
          f"{value['gameplay_successes']} wins; no advancement inference", flush=True)


if __name__ == "__main__":
    main()
