"""Freeze and execute at most eight C2 engineering shot captures, never fresh data."""
import argparse
import html
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

from scripts import issue_76_expansion as expansion
from scripts import issue_76_episode_capture as episode
from scripts.run_issue_76_compatibility import process_rss, terminate_worker

ROOT = expansion.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-76-canonical-smoke-v1"
WORK = Path("/home/sukaih/.cache/novphy-canonical-player-v2")
BUILD = WORK / "player-build-03"
PROTOCOL = "docs/issue-76-canonical-smoke-protocol.md"
SOURCES = ("scripts/run_issue_76_canonical_smoke.py", "scripts/issue_76_episode_capture.py",
           "scripts/issue_76_canonical_instrumentation.py", "scripts/issue_76_expansion.py",
           "scripts/run_issue_76_canonical_player.py", "scripts/run_issue_76_compatibility.py",
           "scripts/collect_rollouts.py", "scripts/observation_trace.py", "scripts/slingshot_readiness.py",
           "scripts/manual_agent.py", "src/webui/bridge.py", "scripts/9001-player-wrapper.sh", PROTOCOL)
SOURCES += ("scripts/run_issue_62_successor_cohort.py", "scripts/smoke_physics_capture.py",
            "scripts/cohort_v2_scenarios.py", "scripts/scenario_manifest.py", "scripts/physics_capture_v2.py",
            "scripts/physics_capture_v2_persistence.py", "tasks/task_generator/canonical_materialization.py",
            "tasks/task_generator/utils/generate_variations.py")
ACTIONS = tuple({"drag_x": x, "drag_y": y, "tap_time_ms": 0, "release_time_ms": 600}
                for x, y in ((-80, 10), (-60, 45), (-10, 80)))
CELLS = (("appearance", 760710001, 760720001, 0, "type010102", 3),
         ("appearance", 760710001, 760720001, 1, "type010102", 3),
         ("rolling", 760710002, 760720002, 0, "type010103", 1),
         ("falling", 760710003, 760720003, 0, "type010204", 1))


def limits():
    return {"max_shot_captures": 8, "episodes": 4, "base_clusters": 3, "workers": 1,
            "active_seconds": 1800, "episode_seconds": 420, "shot_seconds": 180,
            "fixed_step_offset_max": 600, "cpu_rss_mib": 3072, "artifact_bytes": 2 * 2**30,
            "technical_retries": 0, "engine_speed": 1, "viewport": [640, 480], "capture_stride": 1,
            "models_run": [], "model_training_updates": 0, "new_gpu_fitting_seconds": 0}


def make_plan(output=OUTPUT):
    old = expansion.read(expansion.OUTPUT / "plan.json")
    templates = {(m["novelty_level"], m["generator_family"]): m["template"] for m in old["members"]}
    exposure = expansion.exposure_sources() + [{"path": str((expansion.OUTPUT / "plan.json").relative_to(ROOT)),
                                               **expansion.exposure_projection(old)}]
    reserved = {s for p in exposure for s in p["generation_or_reserved_seeds"]}
    if reserved & {c[1] for c in CELLS}:
        raise ValueError("C2 generator seed collides with recorded prior exposure")
    if not (BUILD / "9001.x86_64").is_file():
        raise ValueError("the declared instrumented player build is missing")
    if "Exiting batchmode successfully now!" not in (WORK / "build-03.log").read_text():
        raise ValueError("the declared build did not finish successfully")
    tests = ET.parse(WORK / "tests-06.xml").getroot()
    if tests.attrib.get("passed") != "7" or tests.attrib.get("failed") != "0":
        raise ValueError("all seven canonical resource/behavior/clock/seed fixtures must pass")
    members = []
    for ordinal, (cluster, generation, engine, novelty, family, shots) in enumerate(CELLS, 1):
        template = templates[novelty, family]
        member = {"ordinal": ordinal, "identity": f"issue-76-canonical-{ordinal:02}",
                  "base_cluster": cluster, "generation_seed": generation, "engine_seed": engine,
                  "novelty_level": novelty, "generator_family": family, "template": template,
                  "exposure_role": "calibration", "actions": [dict(a) for a in ACTIONS[:shots]]}
        generated, scenario = expansion.materialize(member, template, output / "authorities" / member["identity"])
        tree = ET.fromstring(generated.xml_content)
        birds = tree.findall("./Birds/Bird")
        if any(b.attrib["type"] != "BirdRed" for b in birds):
            raise ValueError("this observer smoke does not authorize unported ability-spawn identities")
        member.update(xml=generated.xml_content.decode(), scenario=scenario.to_dict(),
                      generated_slots=sorted(n.attrib["scenarioObjectId"] for n in tree.iter() if "scenarioObjectId" in n.attrib))
        members.append(member)
    return {"schema": "issue_76_canonical_smoke_plan_v1", "identity": "issue-76-canonical-smoke-v1",
            "authorization": "operator approved the C2 resource envelope; development engineering only",
            "limits": limits(), "members": members, "prior_exposure": exposure,
            "build_source": str(BUILD), "player_source_version": "2019.3.4f1", "player_build_version": "2019.4.41f2",
            "observer_port": expansion.read(WORK / "instrumentation.json"),
            "shader_restoration": expansion.read(WORK / "shader-restoration.json"),
            "runtime_source_text": {str(p.relative_to(WORK / "project")): p.read_text()
                                    for p in sorted((WORK / "project/Assets/Scripts").rglob("*.cs"))},
            "source_text": {p: (ROOT / p).read_text() for p in SOURCES},
            "transport_source": str(expansion.OUTPUT / "runtimes/issue-76-compatibility-01/game_playing_interface.jar"),
            "static_level_catalog": "excluded from runtime copy; only the assigned frozen XML is installed",
            "seed_correction": "C1 environment_seed was a bridge agent ID; C2 separately seeds Unity before scene load",
            "future_training_excluded": True, "fresh_evaluation_opened": False,
            "final_evaluation_opened": False, "power_established": False, "candidate_eligible": False}


def prepare(output=OUTPUT):
    plan = make_plan(output)
    if output.exists():
        raise ValueError("C2 output already exists; preserve its version instead of overwriting it")
    output.mkdir(parents=True)
    player = output / "player"
    shutil.copytree(BUILD, player, ignore=lambda directory, names: ["Levels"] if Path(directory).name == "StreamingAssets" else [])
    (player / "9001.x86_64").rename(player / "9001-player.x86_64")
    shutil.copy2(ROOT / "scripts/9001-player-wrapper.sh", player / "9001.x86_64")
    shutil.copy2(plan["transport_source"], player / "game_playing_interface.jar")
    source_runtime = Path(plan["transport_source"]).parent
    shutil.copy2(source_runtime / "serverbackup", player / "serverbackup")
    for member in plan["members"]:
        target = output / "authorities" / member["identity"]
        target.mkdir(parents=True)
        (target / "scenario.xml").write_text(member["xml"])
        expansion.write(target / "scenario.json", member["scenario"])
    expansion.write(output / "plan.json", plan)
    episode.log(f"freeze prepared episodes=4 maximum_shot_captures=8: {output}")


def load_plan(output=OUTPUT):
    saved = expansion.read(output / "plan.json")
    if saved != make_plan(output):
        raise ValueError("C2 source/membership freeze changed; preserve all current evidence")
    return saved


def result_path(output, member):
    return output / "results" / (member["identity"] + ".json")


def artifact_bytes(output):
    return sum(p.stat().st_size for p in (output / "attempts").rglob("*") if p.is_file())


def interrupted_result(root, member, failure):
    progress = expansion.read(root / "progress.json") if (root / "progress.json").exists() else {}
    segments = [expansion.read(p) for p in sorted(root.glob("shot-*.json"))]
    attempted = progress.get("attempted_shots", [])
    return {"member_identity": member["identity"], "complete": False, "failure": failure,
            "segments": segments, "attempted_shots": attempted,
            "unattempted_shots": [i for i in range(1, len(member["actions"]) + 1) if i not in attempted]}


def run(output, plan, first_only=False):
    cap = plan["limits"]
    path = output / "budget.json"
    budget = expansion.read(path) if path.exists() else {"active_seconds": 0., "peak_cpu_rss_mib": 0., "artifact_bytes": 0, "stopped": False}
    if budget["stopped"]:
        raise ValueError("C2 global budget has stopped; publish retained evidence")
    initial, started = budget["active_seconds"], time.monotonic()
    for member in plan["members"][:1] if first_only else plan["members"]:
        result = result_path(output, member)
        root = output / "attempts" / member["identity"]
        if result.exists():
            episode.log(f"resume retains {member['identity']}; no replay")
            continue
        if root.exists():
            expansion.write(result, interrupted_result(root, member, "interrupted_single_attempt_retained_no_retry"))
            continue
        if initial + time.monotonic() - started >= cap["active_seconds"]:
            budget["stopped"] = True
            break
        process = multiprocessing.get_context("spawn").Process(target=episode.capture_episode, args=(output, member, cap))
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
                if max(counts, default=0) > cap["fixed_step_offset_max"] + 1:
                    stop = "fixed_step_watchdog"
                if len(folders) > len(member["actions"]):
                    stop = "unexpected_extra_shot_capture"
                if now - beginning > cap["episode_seconds"]:
                    stop = "episode_wall_limit"
                if budget["active_seconds"] > cap["active_seconds"]:
                    stop = "global_wall_limit"
                if rss > cap["cpu_rss_mib"]:
                    stop = "aggregate_memory_limit"
                if now - last_log >= 5:
                    budget["artifact_bytes"] = artifact_bytes(output)
                    if budget["artifact_bytes"] > cap["artifact_bytes"]:
                        stop = "data_limit"
                    completed = sum(result_path(output, m).exists() for m in plan["members"])
                    eta = budget["active_seconds"] / completed * (4 - completed) if completed else None
                    episode.log(f"episode={member['ordinal']}/4 shots={len(folders)}/{len(member['actions'])} frames={counts} elapsed={now-beginning:.1f}s total={budget['active_seconds']:.1f}s ETA={eta} RSS={rss:.1f}MiB")
                    capture_budget(path, budget)
                    last_log = now
                if stop:
                    terminate_worker(process)
                    break
        finally:
            if process.is_alive():
                terminate_worker(process)
        if not result.exists():
            expansion.write(result, interrupted_result(root, member, stop or f"worker_exit_{process.exitcode}"))
        budget["active_seconds"] = initial + time.monotonic() - started
        budget["artifact_bytes"] = artifact_bytes(output)
        budget["stopped"] = stop in ("global_wall_limit", "aggregate_memory_limit", "data_limit")
        capture_budget(path, budget)
        episode.log(f"completed {member['identity']} failure={expansion.read(result).get('failure')}")
        if budget["stopped"]:
            break


def capture_budget(path, value):
    from scripts.run_issue_62_successor_cohort import _replace_json
    _replace_json(value, path)


def report(output, plan):
    entries = []
    for member in plan["members"]:
        path = result_path(output, member)
        entries.append({"member": member, "result": expansion.read(path) if path.exists() else None})
    return {"schema": "issue_76_canonical_smoke_report_v1", "plan_identity": plan["identity"],
            "entries": entries, "all_episodes_attempted": all(e["result"] is not None for e in entries),
            "capture_pipeline_passed": all(e["result"] and e["result"]["complete"] for e in entries),
            "budget": expansion.read(output / "budget.json"), "candidate_eligible": False,
            "fresh_access_allowed": False, "issue_64_authorized": False, "issue_65_authorized": False,
            "interpretation": "engineering data-pipeline evidence only; no prediction, superiority, success-power or appearance-only equivalence claim"}


def validate_captures(output, plan):
    from scripts.observation_trace import validate_observation_trace
    from scripts.physics_capture_v2 import load_physics_capture_v2
    for member in plan["members"]:
        path = result_path(output, member)
        if not path.exists():
            continue
        result = expansion.read(path)
        if result["member_identity"] != member["identity"]:
            raise ValueError("episode result membership differs from freeze")
        for index, segment in enumerate(result["segments"], 1):
            root = output / "attempts" / member["identity"] / f"shot-{index}"
            if expansion.read(root.parent / f"shot-{index}.json") != segment:
                raise ValueError("episode and standalone segment records differ")
            load_physics_capture_v2(root / "physics_capture_v2.json")
            observed = validate_observation_trace(root / "observation-trace")
            physics = expansion.read(root / "physics_capture_v2.json")
            frames = observed["frame_records"]
            if len(frames) != segment["frames"] or frames[0]["fixed_step"] != segment["first_fixed_step"] or frames[-1]["fixed_step"] != segment["last_fixed_step"]:
                raise ValueError("reported segment frame coverage differs")
            if sum(e["event_type"] == "bird_launched" for e in physics["events"]) != 1:
                raise ValueError("saved segment lacks its single actual launch")
        episode.log(f"validated retained episode {member['ordinal']}/4 segments={len(result['segments'])} failure={result.get('failure')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "prepare", "smoke-test", "run-smoke", "publish", "validate"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.dry_run:
        plan = make_plan(args.output)
        episode.log(f"no-write dry run passed: episodes={len(plan['members'])} max_shot_captures={sum(len(m['actions']) for m in plan['members'])}; no models/fresh data")
    elif args.prepare:
        prepare(args.output)
    else:
        plan = load_plan(args.output)
        if args.smoke_test or args.run_smoke:
            run(args.output, plan, args.smoke_test)
        else:
            validate_captures(args.output, plan)
            value = report(args.output, plan)
            destination = ROOT / "data/issue-76-canonical-smoke" / ("attempts-" + str(sum(e["result"] is not None for e in value["entries"])))
            if args.validate:
                if expansion.read(destination / "report.json") != value:
                    raise ValueError("C2 published report differs from retained evidence")
                episode.log("C2 report and retained captures validated; no recapture")
                return
            destination.mkdir(parents=True, exist_ok=True)
            expansion.write(destination / "report.json", value)
            rows = "".join(f"<tr><td>{html.escape(e['member']['identity'])}</td><td>{html.escape(str(e['result'].get('failure') if e['result'] else 'not attempted'))}</td></tr>" for e in value["entries"])
            (destination / "index.html").write_text("<!doctype html><meta charset='utf-8'><h1>C2 canonical engineering smoke</h1><p>No fitted candidate or fresh-access authorization.</p><table>" + rows + "</table>")
            episode.log(f"published {destination}")


if __name__ == "__main__":
    main()
