"""Source-bound censored engineering smoke and operator-run development collection."""
import argparse
import multiprocessing
import os
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

from scripts import issue_76_expansion as files
from scripts import issue_76_censored_episode as episode
from scripts import run_issue_76_canonical_smoke as c2
from scripts import run_issue_76_native_continuation as parent
from scripts.native_segment_trace import NativeSegmentTrace
from scripts.observation_trace import validate_observation_trace
from scripts.run_issue_76_compatibility import process_rss, terminate_worker

ROOT = files.ROOT
PROTOCOL = "docs/issue-76-censored-development-protocol.md"
SMOKE = ROOT / ".local-artifacts/issue-76-censored-smoke-v1"
DEVELOPMENT = ROOT / ".local-artifacts/issue-76-native-development-v1"
FAMILIES = ("type010101", "type010102", "type010103", "type010204", "type010105")
SOURCES = ("scripts/run_issue_76_development.py", "scripts/issue_76_censored_episode.py",
           "scripts/native_segment_trace.py", PROTOCOL)


def assignments(stage):
    if stage == "smoke":
        return [(f, 760810001 + i, 760820001 + i, "calibration", 1)
                for i, f in enumerate((FAMILIES[0], FAMILIES[1], FAMILIES[4]))]
    return [(f, 760900001 + family * 70 + i, 760910001 + family * 70 + i,
             "training" if i < 50 else "calibration" if i < 60 else "model_selection", 3)
            for family, f in enumerate(FAMILIES) for i in range(70)]


def exposure(stage):
    excluded = {SMOKE if stage == "smoke" else DEVELOPMENT}
    values = files.exposure_sources()
    for path in sorted((ROOT / ".local-artifacts").glob("issue-76-*/plan.json")):
        if path.parent not in excluded:
            values.append({"path": str(path.relative_to(ROOT)), **files.exposure_projection(files.read(path))})
    return values


def make_plan(stage, prior_exposure=None):
    previous = parent.load_plan()
    source = files.read(files.OUTPUT / "plan.json")
    templates = {m["generator_family"]: m["template"] for m in source["members"] if m["novelty_level"] == 0}
    prior = exposure(stage) if prior_exposure is None else prior_exposure
    if prior_exposure is not None:
        for entry in prior:
            current = {"path": entry["path"], **files.exposure_projection(files.read(ROOT / entry["path"]))}
            if current != entry:
                raise ValueError("a frozen prior-exposure source changed")
    prior_seeds = {s for entry in prior for s in entry["generation_or_reserved_seeds"]}
    prior_ids = {s for entry in prior for s in entry["scenario_identities"]}
    output = SMOKE if stage == "smoke" else DEVELOPMENT
    members = []
    for ordinal, (family, seed, engine, role, maximum) in enumerate(assignments(stage), 1):
        if seed in prior_seeds:
            raise ValueError("new generation seed overlaps a prior exposure")
        identity = f"issue-76-{stage}-{ordinal:03}"
        member = {"ordinal": ordinal, "identity": identity, "base_cluster": identity,
                  "generation_seed": seed, "engine_seed": engine, "novelty_level": 0,
                  "generator_family": family, "exposure_role": role, "template": templates[family]}
        generated, scenario = files.materialize(member, member["template"], output / "authority" / identity)
        if prior_ids & set(files.exposure_projection(scenario.to_dict())["scenario_identities"]):
            raise ValueError("new lineage overlaps a prior exposure")
        tree = ET.fromstring(generated.xml_content)
        birds = tree.findall("./Birds/Bird")
        if not birds or any(b.attrib["type"] != "BirdRed" for b in birds):
            raise ValueError("the declared collection supports authored red birds only")
        member.update(xml=generated.xml_content.decode(), scenario=scenario.to_dict(),
                      actions=[dict(a) for a in c2.ACTIONS[:min(maximum, len(birds))]],
                      generated_slots=sorted(n.attrib["scenarioObjectId"] for n in tree.iter() if "scenarioObjectId" in n.attrib))
        members.append(member)
    limits = {**previous["limits"], "new_shot_captures_max": sum(len(m["actions"]) for m in members)}
    if stage == "development":
        limits.update(active_seconds=43200, cpu_rss_mib=12288, artifact_bytes=256 * 2**30)
    return {"schema": "issue_76_censored_collection_v1", "identity": output.name,
            "stage": stage, "members": members, "limits": limits, "prior_exposure": prior,
            "parent_plan_identity": previous["identity"],
            "parent_source_text": previous["source_text"],
            "source_text": {p: (ROOT / p).read_text() for p in SOURCES},
            "player_source": str(parent.OUTPUT / "player"),
            "parent_budget": files.read(parent.OUTPUT / "budget.json") if stage == "smoke" else None,
            "fitting_excluded": stage == "smoke", "fresh_access": False,
            "gameplay_timeout_penalty": 1, "technical_retries": 0}


def load_plan(stage):
    root = SMOKE if stage == "smoke" else DEVELOPMENT
    saved = files.read(root / "plan.json")
    if saved != make_plan(stage, saved["prior_exposure"]):
        raise ValueError("collection source, membership or exposure freeze changed")
    return saved


def prepare(stage):
    root = SMOKE if stage == "smoke" else DEVELOPMENT
    if root.exists():
        raise ValueError("collection already prepared; do not overwrite or replay it")
    if stage == "development" and not report(SMOKE, load_plan("smoke"))["collection_pipeline_passed"]:
        raise ValueError("development collection requires the frozen censored smoke to pass")
    plan = make_plan(stage)
    root.mkdir(parents=True)
    shutil.copytree(plan["player_source"], root / "player")
    used = plan["parent_budget"] or {"active_seconds": 0, "artifact_bytes": 0, "peak_cpu_rss_mib": 0}
    files.write(root / "plan.json", plan)
    files.write(root / "budget.json", {"active_seconds": used["active_seconds"],
        "prior_artifact_bytes": used["artifact_bytes"], "artifact_bytes": used["artifact_bytes"],
        "peak_cpu_rss_mib": used["peak_cpu_rss_mib"], "stopped": False})
    episode.old.log(f"{stage} frozen: episodes={len(plan['members'])}, max_shots={plan['limits']['new_shot_captures_max']}")


def run(root, plan, first_only=False):
    budget_path = root / "budget.json"
    budget = files.read(budget_path)
    if budget["stopped"]:
        raise ValueError("collection resource budget stopped; retain all unattempted assignments")
    initial, started = budget["active_seconds"], time.monotonic()
    cap = plan["limits"]
    for member in plan["members"][:1] if first_only else plan["members"]:
        target = c2.result_path(root, member)
        attempt = root / "attempts" / member["identity"]
        if target.exists():
            episode.old.log(f"resume retains {member['identity']}; no replay")
            continue
        if attempt.exists():
            files.write(target, c2.interrupted_result(attempt, member, "interrupted_attempt_no_retry"))
            continue
        process = multiprocessing.get_context("spawn").Process(target=episode.capture_episode, args=(root, member, cap))
        if initial + time.monotonic() - started >= cap["active_seconds"]:
            budget["stopped"] = True
            c2.capture_budget(budget_path, budget)
            break
        process.start()
        beginning = last_log = time.monotonic()
        stop = None
        try:
            while process.is_alive():
                process.join(.25)
                now = time.monotonic()
                rss = process_rss(os.getpid())
                budget["active_seconds"] = initial + now - started
                budget["peak_cpu_rss_mib"] = max(budget["peak_cpu_rss_mib"], rss)
                folders = list((attempt / "aligned").glob("capture-v2:*"))
                counts = [sum(1 for _ in f.glob("frame_*.json")) for f in folders]
                if len(folders) > len(member["actions"]) or max(counts, default=0) > cap["rgb_frames_max"]:
                    stop = "native_frame_or_capture_count_limit"
                if now - beginning > cap["episode_seconds"]:
                    stop = "episode_wall_limit"
                if rss > cap["cpu_rss_mib"]:
                    stop = "aggregate_memory_limit"
                if budget["active_seconds"] >= cap["active_seconds"]:
                    stop = "global_wall_limit"
                if now - last_log >= 5:
                    budget["artifact_bytes"] = budget["prior_artifact_bytes"] + c2.artifact_bytes(root)
                    if budget["artifact_bytes"] > cap["artifact_bytes"]:
                        stop = "data_limit"
                    completed = sum(c2.result_path(root, m).exists() for m in plan["members"])
                    eta = (now - started) / completed * (len(plan["members"]) - completed) if completed else None
                    chunks = sum(1 for _ in (attempt / "aligned").glob("*/native/chunk-*.json.gz"))
                    episode.old.log(f"{plan['stage']} episode={member['ordinal']}/{len(plan['members'])} RGB={counts} chunks={chunks} active={budget['active_seconds']:.1f}s RSS={rss:.1f}MiB bytes={budget['artifact_bytes']} ETA={eta}")
                    c2.capture_budget(budget_path, budget)
                    last_log = now
                if stop:
                    terminate_worker(process)
                    break
        finally:
            if process.is_alive():
                terminate_worker(process)
        if not target.exists():
            files.write(target, c2.interrupted_result(attempt, member, stop or f"worker_exit_{process.exitcode}"))
        budget["active_seconds"] = initial + time.monotonic() - started
        budget["artifact_bytes"] = budget["prior_artifact_bytes"] + c2.artifact_bytes(root)
        budget["stopped"] = stop in ("aggregate_memory_limit", "global_wall_limit", "data_limit")
        c2.capture_budget(budget_path, budget)
        value = files.read(target)
        episode.old.log(f"recorded {member['identity']} collection_complete={value['complete']} gameplay_success={value.get('gameplay_success', False)} failure={value['failure']}")
        if budget["stopped"]:
            break


def report(root, plan):
    entries = []
    for member in plan["members"]:
        path = c2.result_path(root, member)
        result = files.read(path) if path.exists() else None
        if result:
            for index, segment in enumerate(result["segments"], 1):
                trace = NativeSegmentTrace(segment["native_root"])
                if trace.summary != segment["summary"]:
                    raise ValueError("native segment summary differs from its streamed evidence")
                observed = validate_observation_trace(root / "attempts" / member["identity"] / f"shot-{index}/observation-trace")
                if observed["identity"] != segment["observation_manifest"] or len(observed["frame_records"]) != segment["frame_count"]:
                    raise ValueError("observed segment source binding differs")
        entries.append({"member": member, "result": result})
    successes = sum(bool(e["result"] and e["result"].get("gameplay_success")) for e in entries)
    return {"schema": "issue_76_censored_collection_report_v1", "plan_identity": plan["identity"],
            "entries": entries, "collection_pipeline_passed": all(e["result"] and e["result"]["complete"] for e in entries),
            "assigned_episode_count": len(entries), "gameplay_successes": successes,
            "gameplay_failure_or_unattempted_count": len(entries) - successes,
            "budget": files.read(root / "budget.json"), "candidate_eligible": False, "fresh_access_allowed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "development"), required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "prepare", "smoke-test", "collect", "publish", "validate"):
        modes.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    root = SMOKE if args.stage == "smoke" else DEVELOPMENT
    if args.dry_run:
        plan = make_plan(args.stage)
        episode.old.log(f"no-write {args.stage} dry-run passed: {len(plan['members'])} members, {plan['limits']['new_shot_captures_max']} maximum shots; source/exposure checked, no images opened")
    elif args.prepare:
        prepare(args.stage)
    else:
        plan = load_plan(args.stage)
        if args.collect or args.smoke_test:
            run(root, plan, first_only=args.smoke_test)
        else:
            value = report(root, plan)
            count = sum(e["result"] is not None for e in value["entries"])
            target = ROOT / f"data/issue-76-{args.stage}-censored/attempts-{count}/report.json"
            if args.publish:
                if target.exists() and files.read(target) != value:
                    raise ValueError("do not overwrite a different published report")
                files.write(target, value)
            elif files.read(target) != value:
                raise ValueError("published collection report differs")
            episode.old.log(f"report {'published' if args.publish else 'validated'}: {target}")


if __name__ == "__main__":
    main()
