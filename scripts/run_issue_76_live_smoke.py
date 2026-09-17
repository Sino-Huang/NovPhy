"""Prospective, bounded engineering smoke for the native live policy interface."""
import argparse
import multiprocessing
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import torch

from scripts import issue_76_live_episode as live
from scripts import run_issue_76_development as development
from scripts.run_issue_76_compatibility import process_rss, start_isolated_worker, terminate_worker
from world_model.planning.native_gameplay import NativeObservationHistory
from world_model.training.native_history_fit import NativeVisualParser
from world_model.training.native_history_data import VISUAL_DIM
from world_model.training.observed_history import ObservedHistoryEncoder

files = live.files
OUTPUT = files.ROOT / ".local-artifacts/issue-76-live-engineering-v1"
PROTOCOL = "docs/issue-76-live-engineering-protocol.md"
SOURCES = ("scripts/run_issue_76_live_smoke.py", "scripts/issue_76_live_episode.py",
           "world_model/planning/native_gameplay.py", "scripts/issue_76_native_outcomes.py", PROTOCOL)
ACTION = dict(drag_x=-80, drag_y=0, release_time_ms=600, tap_time_ms=0)


def make_plan():
    prior = development.exposure("development")
    prior.append({"path": str((development.DEVELOPMENT / "plan.json").relative_to(files.ROOT)),
                  **files.exposure_projection(files.read(development.DEVELOPMENT / "plan.json"))})
    prior = [e for e in prior if e["path"] != str((OUTPUT / "plan.json").relative_to(files.ROOT))]
    seeds = {s for e in prior for s in e["generation_or_reserved_seeds"]}
    identities = {s for e in prior for s in e["scenario_identities"]}
    parent = files.read(development.DEVELOPMENT / "plan.json")
    templates = {m["generator_family"]: m["template"] for m in parent["members"]}
    members = []
    for i, family in enumerate(("type010101", "type010102", "type010105"), 1):
        member = dict(identity=f"issue-76-live-engineering-{i:03}", ordinal=i,
            base_cluster=f"issue-76-live-engineering-{i:03}", generation_seed=760950000 + i,
            engine_seed=760960000 + i, novelty_level=0, generator_family=family,
            exposure_role="calibration", template=templates[family])
        if member["generation_seed"] in seeds:
            raise ValueError("engineering generation seed overlaps a prior assignment")
        generated, scenario = files.materialize(member, member["template"], OUTPUT / "authority" / member["identity"])
        if identities & set(files.exposure_projection(scenario.to_dict())["scenario_identities"]):
            raise ValueError("engineering scenario lineage overlaps a prior assignment")
        xml = ET.fromstring(generated.xml_content)
        birds = xml.findall("./Birds/Bird")
        if not birds or any(b.attrib["type"] != "BirdRed" for b in birds):
            raise ValueError("engineering protocol supports authored red birds only")
        member.update(xml=generated.xml_content.decode(), scenario=scenario.to_dict(),
                      maximum_shots=min(3, len(birds)))
        members.append(member)
    return dict(schema="issue_76_live_engineering_plan_v1", identity=OUTPUT.name,
        members=members, source_text={p: (files.ROOT / p).read_text() for p in SOURCES},
        parent_plan_identity=parent["identity"], parent_source_text=parent["source_text"],
        player_source=str(development.DEVELOPMENT / "player"), prior_exposure=prior,
        limits=dict(active_seconds=1800, attempt_seconds=420, shot_seconds=180,
                    cpu_rss_mib=4096, artifact_bytes=4 * 2**30, workers=1, technical_retries=0,
                    maximum_new_shots=sum(m["maximum_shots"] for m in members), readiness_action=ACTION),
        policy=dict(action=ACTION, mode="fixed_engineering_action_untrained_history_smoke",
                    initialization_seed=760940001, torch_version=str(torch.__version__), device="cpu"),
        fitting_excluded=True, model_comparison_performed=False, fresh_access=False,
        issue_64_authorized=False)


def worker(member, limits):
    torch.set_num_threads(1)
    torch.manual_seed(760940001)
    history = NativeObservationHistory(NativeVisualParser(), ObservedHistoryEncoder(carrier_dim=VISUAL_DIM))
    policy = live.HistoryPolicy(history, lambda carrier: {
        "action": dict(ACTION), "selection": "prospectively_fixed_engineering_action",
        "trained_model_used": False, "candidates_scored": 0})
    live.play_episode(OUTPUT, member, limits, policy)


def run(plan, first_only=False):
    cap = plan["limits"]
    budget_path = OUTPUT / "budget.json"
    budget = files.read(budget_path)
    if budget["stopped"]:
        raise ValueError("engineering resource stop is terminal for this assignment")
    initial, started = budget["active_seconds"], time.monotonic()
    for member in plan["members"][:1] if first_only else plan["members"]:
        target = OUTPUT / "results" / (member["identity"] + ".json")
        attempt = OUTPUT / "attempts" / member["identity"]
        if target.exists():
            continue
        stop = "interrupted_attempt_no_retry" if attempt.exists() else None
        process = None
        if stop is None:
            process = start_isolated_worker(multiprocessing.get_context("spawn"),worker,(member, cap))
            beginning = last_log = time.monotonic()
            try:
                while process.is_alive():
                    process.join(1)
                    now = time.monotonic()
                    budget["active_seconds"] = initial + now - started
                    budget["peak_cpu_rss_mib"] = max(budget["peak_cpu_rss_mib"], process_rss(process.pid))
                    if now - beginning > cap["attempt_seconds"]:
                        stop = "episode_wall_limit"
                    if budget["peak_cpu_rss_mib"] > cap["cpu_rss_mib"]:
                        stop = "memory_limit"
                    if budget["active_seconds"] > cap["active_seconds"]:
                        stop = "global_wall_limit"
                    if now - last_log >= 5:
                        budget["artifact_bytes"] = development.c2.artifact_bytes(OUTPUT)
                        if budget["artifact_bytes"] > cap["artifact_bytes"]:
                            stop = "storage_limit"
                        live.capture.old.log(f"live smoke {member['ordinal']}/3 active={budget['active_seconds']:.1f}s RSS={budget['peak_cpu_rss_mib']:.1f}MiB bytes={budget['artifact_bytes']}")
                        live._replace_json(budget, budget_path)
                        last_log = now
                    if stop: break
            finally:
                terminate_worker(process)
        if not target.exists():
            files.write(target, dict(member_identity=member["identity"], complete=False,
                failure=stop or f"worker_exit_{process.exitcode}", gameplay_success=False,
                gameplay_failure_penalty=1, fresh_access=False, raw_attempt_root=str(attempt)))
        budget["active_seconds"] = initial + time.monotonic() - started
        budget["artifact_bytes"] = development.c2.artifact_bytes(OUTPUT)
        budget["stopped"] = stop in ("global_wall_limit", "memory_limit", "storage_limit")
        live._replace_json(budget, budget_path)
        value = files.read(target)
        live.capture.old.log(f"live smoke recorded {member['identity']} complete={value['complete']} success={value['gameplay_success']} failure={value['failure']}")
        if budget["stopped"]:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-only", action="store_true")
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "prepare", "record-correction", "run", "report"):
        modes.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    if args.dry_run or args.prepare:
        plan = make_plan()
        if args.prepare:
            OUTPUT.mkdir(parents=True)
            shutil.copytree(plan["player_source"], OUTPUT / "player")
            files.write(OUTPUT / "plan.json", plan)
            files.write(OUTPUT / "budget.json", dict(active_seconds=0., peak_cpu_rss_mib=0.,
                                                     artifact_bytes=0, stopped=False))
        print(f"live engineering {'prepared' if args.prepare else 'no-write dry-run'}: 3 excluded lineages, "
              f"at most {plan['limits']['maximum_new_shots']} shots; no model comparison or fresh access", flush=True)
    else:
        plan = files.read(OUTPUT / "plan.json")
        correction = OUTPUT / "observation-json-correction.json"
        if args.record_correction:
            current = make_plan()
            if ({k: v for k, v in plan.items() if k != "source_text"}
                    != {k: v for k, v in current.items() if k != "source_text"}):
                raise ValueError("technical correction cannot change assignments, policy or limits")
            first = files.read(OUTPUT / "results" / (plan["members"][0]["identity"] + ".json"))
            if (first["failure"] != "ObservationTraceError: camera position is incomplete"
                    or first["attempted_shots"] or first["segments"]):
                raise ValueError("requires the exact retained pre-shot serialization failure")
            if any((OUTPUT / "attempts" / m["identity"]).exists() for m in plan["members"][1:]):
                raise ValueError("correction must precede remaining engineering access")
            if correction.exists():
                raise ValueError("technical correction is already frozen")
            files.write(correction, dict(reason="recursively convert typed bridge metadata to JSON lists",
                original_failed_result=first, original_budget=files.read(OUTPUT / "budget.json"),
                original_plan_preserved=True, failed_assignment_replayed=False, plan=current))
            print("technical correction frozen; original failed attempt/costs retained", flush=True)
            return
        if correction.exists():
            receipt = files.read(correction)
            first = files.read(OUTPUT / "results" / (plan["members"][0]["identity"] + ".json"))
            if first != receipt["original_failed_result"]:
                raise ValueError("original engineering failure changed")
            plan = receipt["plan"]
        if plan != make_plan():
            raise ValueError("frozen engineering source/membership changed; retain original attempts")
        if args.run:
            run(plan, args.first_only)
        else:
            results = [files.read(OUTPUT / "results" / (m["identity"] + ".json")) for m in plan["members"]]
            print({"completed": sum(r["complete"] for r in results), "assigned": len(results),
                   "gameplay_successes": sum(r["gameplay_success"] for r in results),
                   "failures": [r["failure"] for r in results], "fresh_access": False}, flush=True)


if __name__ == "__main__":
    main()
