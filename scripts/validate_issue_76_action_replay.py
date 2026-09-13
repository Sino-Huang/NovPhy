"""Freeze replay execution and validate its bounded, outcome-independent smoke."""
import argparse
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from scripts import run_issue_76_action_replay as run
from scripts import issue_76_replay_comparability as comparison
from scripts.native_segment_trace import NativeSegmentTrace
from scripts.observation_trace import validate_observation_trace
from scripts.run_issue_76_order_probe import immutable

OUTPUT = run.metadata.OUTPUT
SOURCES = ("scripts/validate_issue_76_action_replay.py", "scripts/run_issue_76_action_replay.py",
           "scripts/issue_76_fixed_replay_policy.py", "scripts/issue_76_replay_comparability.py",
           "tests/test_issue_76_replay_comparability.py", "tests/test_issue_76_action_replay_supervisor.py",
           "tests/test_issue_76_fixed_replay_policy.py", "tests/test_issue_76_action_replay_validation.py",
           "scripts/native_segment_trace.py", "scripts/canonical_native_trace.py", "scripts/observation_trace.py",
           "scripts/issue_76_episode_capture.py", "scripts/slingshot_readiness.py", "src/webui/bridge.py",
           "scripts/run_issue_76_compatibility.py", "docs/issue-76-replay-execution.md")


def prepare(root=run.ROOT):
    inventory = run.files.read(root / "inventory.json")
    sources = dict(inventory["source_text"])
    for path, text in sources.items():
        if (run.files.ROOT / path).read_text() != text:
            raise ValueError("frozen replay inventory source changed")
    sources.update({path: (run.files.ROOT / path).read_text() for path in SOURCES})
    plan = {"identity": "issue-76-action-replay-execution-v1", "inventory": inventory,
        "capture_execution_authorized": True, "comparability_source_frozen": True,
        "comparability_limits": comparison.LIMITS, "source_text": sources,
        "readiness_action": inventory["members"][0]["actions"][0],
        "model_scoring_authorized": False, "new_optimizer_updates": 0, "fresh_access": False}
    immutable(root / "execution-plan.json", plan)
    immutable(OUTPUT / "execution-plan.json", plan)
    if not (root / "player").exists():
        started = time.monotonic()
        shutil.copytree(inventory["player_source"], root / "player")
        immutable(root / "player-preparation.json", {"source": inventory["player_source"],
                  "wall_seconds": time.monotonic() - started, "new_player_build": False})
    run.load_plan(root)
    print("Execution source frozen and existing player copied; no capture started.", flush=True)


def read_case(root, plan, member):
    result = run.files.read(root / "results" / (member["identity"] + ".json"))
    supervisor = run.files.read(root / "supervision" / (member["identity"] + ".json"))
    if (not supervisor["execution_within_limits"] or supervisor["worker_exitcode"] != 0
            or supervisor["execution_plan_identity"] != plan["identity"]
            or supervisor["member_identity"] != member["identity"]
            or not result["complete"] or result["failure"] is not None):
        raise ValueError("replay did not complete within its supervised capture contract")
    for key in ("member_identity", "base_cluster", "exposure_role", "engine_seed"):
        if result[key] != member["identity" if key == "member_identity" else key]:
            raise ValueError("captured replay source differs")
    if len(result["decisions"]) != 1 or len(result["segments"]) != 1 or result["attempted_shots"] != [1]:
        raise ValueError("replay is not the exact assigned one-shot capture")
    decision, segment = result["decisions"][0], result["segments"][0]
    if decision["action"] != member["actions"][0] or decision["trained_model_used"] or decision["candidates_scored"]:
        raise ValueError("executed decision differs from the fixed assignment")
    attempt = root / "attempts" / member["identity"]
    snapshot_root, captured_root = attempt / "decision-1", attempt / "shot-1/observation-trace"
    snapshot, captured = validate_observation_trace(snapshot_root), validate_observation_trace(captured_root)
    lineage = member["scenario"]["scenario_manifest"]["scenario_lineage"]["identity"]
    if (snapshot["identity"] != decision["observation_manifest"] or captured["identity"] != segment["observation_manifest"]
            or len(snapshot["frame_records"]) != 1
            or any(value["exposure_role"] != "training" or value["scenario_lineage_identity"] != lineage for value in (snapshot, captured))):
        raise ValueError("decision/capture observation bindings differ")
    trace = NativeSegmentTrace(segment["native_root"])
    if (trace.summary != segment["summary"] or int(trace.manifest["engine_seed"]) != member["engine_seed"]
            or trace.summary["sample_count"] > plan["inventory"]["limits"]["native_steps_per_shot_max"] + 1
            or len(captured["frame_records"]) > plan["inventory"]["limits"]["rgb_frames_per_shot_max"]
            or [f["fixed_step"] for f in captured["frame_records"]] != [f["fixed_step"] for f in trace.manifest["frame_records"]]):
        raise ValueError("native clock, bounds or trace/observation alignment differs")
    initial = next(trace.trace.chunks())["fixed_step_samples"][0]
    if not set(member["generated_slots"]).issubset({entity["scenario_object_id"] for entity in initial["entities"]}):
        raise ValueError("initial native capture omitted an authored entity")
    frame = snapshot["frame_records"][0]
    first = captured["frame_records"][0]
    evidence = result["policy"]
    if (evidence["decision_time"] != frame["fixed_time_seconds"] or evidence["executed_action_time"] != first["fixed_time_seconds"]
            or evidence["executed_actions"] != 1 or evidence["decisions"] != 1
            or evidence["observations"] != 1 + len(captured["frame_records"])):
        raise ValueError("causal observation/action callback inventory differs")
    image_path = snapshot_root / frame["agent_observation"]["relative_path"]
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    return {**{key: member[key] for key in ("base_cluster", "scenario", "generation_seed", "engine_seed", "exposure_role")},
            "observation_configuration": snapshot["observation_configuration"],
            "viewport": frame["capture_metadata"]["viewport"],
            "transform": frame["capture_metadata"]["world_to_observation_transform"],
            "decision_time": frame["fixed_time_seconds"], "capture_time": first["fixed_time_seconds"],
            "rgb": rgb, "initial": initial, "decision_image_path": str(image_path)}


def validate_smoke(root=run.ROOT):
    plan = run.load_plan(root)
    budget = run.files.read(root / "capture-budget.json")
    if budget["running"] or budget["stopped"]:
        raise ValueError("smoke capture budget is not cleanly completed")
    members = run.selected_members(plan, root, True)
    started = time.monotonic()
    cases = [read_case(root, plan, member) for member in members]
    compared = comparison.compare(cases[0], cases[1], plan["comparability_limits"])
    report = {"execution_plan_identity": plan["identity"], "member_identities": [m["identity"] for m in members],
        "validated": compared["comparable"], "comparability": compared, "capture_budget": budget,
        "validation_wall_seconds": time.monotonic() - started,
        "decision_image_paths": [case["decision_image_path"] for case in cases],
        "gameplay_success_used_as_continuation_criterion": False, "model_scoring_authorized": False,
        "fresh_access": False, "advancement_authorized": False}
    immutable(root / "smoke-validation.json", report)
    immutable(OUTPUT / "smoke-validation.json", report)
    print({"smoke_validated": report["validated"], "comparability_failures": compared["failures"]}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--validate-smoke", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.validate_smoke:
        validate_smoke()
    else:
        print("No-write: freeze execution or verify exactly the two assigned smoke replays.")


if __name__ == "__main__":
    main()
