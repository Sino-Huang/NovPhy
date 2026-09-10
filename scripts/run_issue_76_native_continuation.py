"""Execute only C2's unused assignments under the prospectively frozen native contract."""
import argparse
import multiprocessing
import os
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

from scripts import issue_76_expansion as files
from scripts import issue_76_native_episode as episode
from scripts import issue_76_native_player as player
from scripts import run_issue_76_canonical_smoke as c2
from scripts.canonical_native_trace import NativeTrace
from scripts.run_issue_76_compatibility import process_rss, terminate_worker

ROOT = files.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-76-native-continuation-v1"
PROTOCOL = "docs/issue-76-native-continuation-protocol.md"
SOURCES = ("scripts/run_issue_76_native_continuation.py", "scripts/issue_76_native_episode.py",
           "scripts/issue_76_native_player.py", "scripts/canonical_native_trace.py",
           "tasks/issue_76_canonical/CanonicalNativeTrace.cs", "tasks/issue_76_canonical/CanonicalPigNativeTraceTests.cs",
           "docs/issue-76-native-time-capture-amendment.md", PROTOCOL)


def make_plan():
    prior = c2.load_plan()
    stopped = files.read(c2.OUTPUT / "native-time-window-stop.json")
    first = files.read(c2.OUTPUT / "results/issue-76-canonical-01.json")
    if first["attempted_shots"] != [1] or first["failure"] != "fixed_step_watchdog":
        raise ValueError("continuation requires the retained first C2 failed attempt")
    if any(c2.result_path(c2.OUTPUT, m).exists() for m in prior["members"][1:]):
        raise ValueError("continuation would repeat a C2 assignment")
    tests = ET.parse(player.WORK / "tests-02.xml").getroot()
    if tests.attrib.get("passed") != "10" or tests.attrib.get("failed") != "0":
        raise ValueError("native continuation requires all ten exact native fixtures")
    build = player.WORK / "player-build-01"
    if not (build / "9001.x86_64").is_file() or "Exiting batchmode successfully now!" not in (player.WORK / "build-01.log").read_text():
        raise ValueError("declared native player build is incomplete")
    return {"schema": "issue_76_native_continuation_plan_v1", "identity": "issue-76-native-continuation-v1",
            "parent_plan_identity": prior["identity"], "parent_stop": stopped,
            "members": prior["members"][1:], "prior_failed_result": first,
            "limits": {"workers": 1, "active_seconds": 1800, "episode_seconds": 420,
                       "shot_seconds": 180, "cpu_rss_mib": 3072, "artifact_bytes": 2 * 2**30,
                       "native_steps_max": 30000, "rgb_frames_max": 601, "chunk_samples_max": 250,
                       "chunk_uncompressed_bytes_max": 16 * 2**20, "technical_retries": 0,
                       "new_shot_captures_max": 5, "fixed_delta_seconds": .0004,
                       "observation_stride": 50, "stable_native_steps": 100},
            "build_source": str(build), "transport_source": prior["transport_source"],
            "parent_source_text": prior["source_text"],
            "source_text": {p: (ROOT / p).read_text() for p in SOURCES},
            "runtime_source_text": {str(p.relative_to(player.WORK / "project")): p.read_text()
                                    for p in sorted((player.WORK / "project/Assets/Scripts").rglob("*.cs"))},
            "native_port": files.read(player.WORK / "native-port.json"),
            "model_fitting": False, "future_training_excluded": True, "fresh_access": False,
            "candidate_eligible": False, "issue_64_authorized": False, "issue_65_authorized": False}


def prepare():
    plan = make_plan()
    if OUTPUT.exists():
        raise ValueError("native continuation already exists; do not overwrite it")
    OUTPUT.mkdir(parents=True)
    runtime = OUTPUT / "player"
    shutil.copytree(plan["build_source"], runtime,
                    ignore=lambda directory, names: ["Levels"] if Path(directory).name == "StreamingAssets" else [])
    (runtime / "9001.x86_64").rename(runtime / "9001-player.x86_64")
    shutil.copy2(ROOT / "scripts/9001-player-wrapper.sh", runtime / "9001.x86_64")
    shutil.copy2(plan["transport_source"], runtime / "game_playing_interface.jar")
    shutil.copy2(Path(plan["transport_source"]).parent / "serverbackup", runtime / "serverbackup")
    files.write(OUTPUT / "plan.json", plan)
    used = plan["parent_stop"]["budget_before_stop"]
    files.write(OUTPUT / "budget.json", {"active_seconds": used["active_seconds"],
        "artifact_bytes": used["artifact_bytes"], "parent_artifact_bytes": used["artifact_bytes"],
        "peak_cpu_rss_mib": used["peak_cpu_rss_mib"], "stopped": False})
    episode.old.log("native continuation frozen: original episodes2–4 only, max5 new captures; C2 costs charged")


def load_plan():
    saved = files.read(OUTPUT / "plan.json")
    if saved != make_plan():
        raise ValueError("native continuation source/player/membership freeze changed")
    return saved


def run(plan, first_only):
    budget_path = OUTPUT / "budget.json"
    budget = files.read(budget_path)
    if budget["stopped"]:
        raise ValueError("combined engineering budget stopped; do not retry")
    initial, started = budget["active_seconds"], time.monotonic()
    cap = plan["limits"]
    members = plan["members"][:1] if first_only else plan["members"]
    for position, member in enumerate(members, 1):
        result = c2.result_path(OUTPUT, member)
        root = OUTPUT / "attempts" / member["identity"]
        if result.exists():
            episode.old.log(f"native resume retains {member['identity']}; no replay")
            continue
        if root.exists():
            files.write(result, c2.interrupted_result(root, member, "interrupted_native_attempt_no_retry"))
            continue
        if initial + time.monotonic() - started >= cap["active_seconds"]:
            budget["stopped"] = True
            break
        process = multiprocessing.get_context("spawn").Process(target=episode.capture_episode, args=(OUTPUT, member, cap))
        process.start()
        beginning = last_log = time.monotonic()
        stop = None
        try:
            while process.is_alive():
                process.join(.25)
                now = time.monotonic()
                rss = process_rss(os.getpid())
                budget["peak_cpu_rss_mib"] = max(budget["peak_cpu_rss_mib"], rss)
                budget["active_seconds"] = initial + now - started
                folders = list((root / "aligned").glob("capture-v2:*"))
                counts = [sum(1 for _ in folder.glob("frame_*.json")) for folder in folders]
                if max(counts, default=0) > cap["rgb_frames_max"] or len(folders) > len(member["actions"]):
                    stop = "native_frame_or_capture_count_limit"
                if now - beginning > cap["episode_seconds"]:
                    stop = "native_episode_wall_limit"
                if rss > cap["cpu_rss_mib"]:
                    stop = "aggregate_memory_limit"
                if budget["active_seconds"] > cap["active_seconds"]:
                    stop = "global_wall_limit"
                if now - last_log >= 5:
                    budget["artifact_bytes"] = budget["parent_artifact_bytes"] + c2.artifact_bytes(OUTPUT)
                    if budget["artifact_bytes"] > cap["artifact_bytes"]:
                        stop = "data_limit"
                    completed = sum(c2.result_path(OUTPUT, m).exists() for m in plan["members"])
                    new_active = budget["active_seconds"] - plan["parent_stop"]["budget_before_stop"]["active_seconds"]
                    eta = new_active / completed * (3 - completed) if completed else None
                    chunks = sum(1 for _ in (root / "aligned").glob("capture-v2:*/native/chunk-*.json.gz"))
                    episode.old.log(f"native episode={member['ordinal']}/4 remaining-assignment={position}/3 RGB={counts} chunks={chunks} wall={now-beginning:.1f}s combined={budget['active_seconds']:.1f}s ETA={eta} RSS={rss:.1f}MiB")
                    c2.capture_budget(budget_path, budget)
                    last_log = now
                if stop:
                    terminate_worker(process)
                    break
        finally:
            if process.is_alive():
                terminate_worker(process)
        if not result.exists():
            files.write(result, c2.interrupted_result(root, member, stop or f"native_worker_exit_{process.exitcode}"))
        budget["active_seconds"] = initial + time.monotonic() - started
        budget["artifact_bytes"] = budget["parent_artifact_bytes"] + c2.artifact_bytes(OUTPUT)
        budget["stopped"] = stop in ("aggregate_memory_limit", "global_wall_limit", "data_limit")
        c2.capture_budget(budget_path, budget)
        episode.old.log(f"native completed {member['identity']} failure={files.read(result).get('failure')}")
        if budget["stopped"]:
            break


def publication(plan):
    from scripts.observation_trace import validate_observation_trace
    entries = []
    for member in plan["members"]:
        path = c2.result_path(OUTPUT, member)
        result = files.read(path) if path.exists() else None
        if result:
            for index, segment in enumerate(result["segments"], 1):
                if NativeTrace(segment["native_root"]).validate() != segment["summary"]:
                    raise ValueError("native segment summary differs from full streamed evidence")
                root = OUTPUT / "attempts" / member["identity"] / f"shot-{index}"
                observed = validate_observation_trace(root / "observation-trace")
                if observed["identity"] != segment["observation_manifest"] or len(observed["frame_records"]) != segment["frame_count"]:
                    raise ValueError("native observation evidence differs")
        entries.append({"member": member, "result": result})
    return {"schema": "issue_76_native_continuation_report_v1", "plan_identity": plan["identity"],
            "prior_failed_C2_attempt": plan["prior_failed_result"], "entries": entries,
            "all_remaining_assignments_attempted": all(e["result"] is not None for e in entries),
            "native_pipeline_passed": all(e["result"] and e["result"]["complete"] for e in entries),
            "budget": files.read(OUTPUT / "budget.json"), "candidate_eligible": False,
            "fresh_access_allowed": False, "issue_64_authorized": False, "issue_65_authorized": False,
            "interpretation": "engineering evidence only; original failed C2 attempt retained; no paired novelty-effect or model-superiority claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "prepare", "smoke-test", "run-smoke", "publish", "validate"):
        modes.add_argument("--" + mode, action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        plan = make_plan()
        episode.old.log(f"native no-write dry-run passed: {len(plan['members'])} unused assignments, max5 shots, 12s native windows")
    elif args.prepare:
        prepare()
    else:
        plan = load_plan()
        if args.smoke_test or args.run_smoke:
            run(plan, args.smoke_test)
        else:
            value = publication(plan)
            count = sum(e["result"] is not None for e in value["entries"])
            target = ROOT / f"data/issue-76-native-continuation/attempts-{count}/report.json"
            if args.validate:
                if files.read(target) != value:
                    raise ValueError("native published report differs")
                episode.old.log("native report, microsteps and observations validated without recapture")
            else:
                files.write(target, value)
                episode.old.log(f"native report published {target}")


if __name__ == "__main__":
    main()
