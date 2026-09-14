"""Validate causal event captures without changing their frozen exposure roles.

The earlier training-only replay validator is source-frozen. Reuse its low-level
trace validators here, with explicit role, rollout, executed-action and port
bindings for the new development contract; do not patch its global role check.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import prepare_issue_76_event_development as inventory
from scripts.issue_76_event_resources import worker_ports
from scripts.native_segment_trace import NativeSegmentTrace
from scripts.observation_trace import validate_observation_trace


def observation_source_matches(value, member):
    return (member["exposure_role"] in inventory.ROLES
            and value["exposure_role"] == member["exposure_role"]
            and value["source_bindings"]["source_scenario_lineage_identity"]
                == member["scenario"]["scenario_manifest"]["scenario_lineage"]["identity"]
            and value["source_bindings"]["rollout_identity"] == member["identity"] + ":shot-1")


def read_case(root, plan, member):
    root = Path(root)
    result = inventory.files.read(root / "results" / (member["identity"] + ".json"))
    receipt = inventory.files.read(root / "supervision" / (member["identity"] + ".json"))
    if (not receipt["execution_within_limits"] or receipt["worker_exitcode"] != 0
            or receipt["execution_plan_identity"] != plan["identity"]
            or receipt["member_identity"] != member["identity"]
            or not result["complete"] or result["failure"] is not None):
        raise ValueError("event assignment did not complete within its supervised capture contract")
    for key in ("member_identity", "base_cluster", "exposure_role", "engine_seed"):
        if result[key] != member["identity" if key == "member_identity" else key]:
            raise ValueError("event capture source or exposure role differs")
    ports = dict(zip(("agent", "game", "physics"), worker_ports(receipt["worker_slot"])))
    if result["ports"] != ports:
        raise ValueError("capture ports differ from the isolated worker slot")
    if len(result["decisions"]) != 1 or len(result["segments"]) != 1 or result["attempted_shots"] != [1]:
        raise ValueError("event assignment is not the exact one-shot capture")
    decision, segment = result["decisions"][0], result["segments"][0]
    action = member["actions"][0]
    if decision["action"] != action or decision["trained_model_used"] or decision["candidates_scored"]:
        raise ValueError("event decision differs from its fixed assignment")
    executed = segment["action"]
    if (executed["coordinate_frame"] != "slingshot_relative"
            or executed["drag_release"] != [action["drag_x"], action["drag_y"]]
            or executed["release_time"] != action["release_time_ms"]
            or executed["tap_time"] != action["tap_time_ms"]):
        raise ValueError("executed actuator action differs from its fixed assignment")
    attempt = root / "attempts" / member["identity"]
    native_root = attempt / "aligned" / segment["capture_id"]
    if Path(segment["native_root"]).resolve() != native_root.resolve():
        raise ValueError("native capture belongs to a different attempt")
    snapshot_root, captured_root = attempt / "decision-1", attempt / "shot-1/observation-trace"
    snapshot, captured = validate_observation_trace(snapshot_root), validate_observation_trace(captured_root)
    if (snapshot["identity"] != decision["observation_manifest"]
            or captured["identity"] != segment["observation_manifest"]
            or len(snapshot["frame_records"]) != 1
            or snapshot["scenario_lineage_identity"] != captured["scenario_lineage_identity"]
            or snapshot["observation_configuration"] != captured["observation_configuration"]
            or any(not observation_source_matches(value, member) for value in (snapshot, captured))):
        raise ValueError("decision/capture observation source, role or rollout binding differs")
    trace = NativeSegmentTrace(native_root)
    limits = plan["inventory"]["limits"]
    if (trace.summary != segment["summary"] or not trace.summary["observed_window_valid"]
            or trace.manifest["capture_id"] != segment["capture_id"]
            or int(trace.manifest["engine_seed"]) != member["engine_seed"]
            or trace.summary["sample_count"] > limits["native_steps_per_shot_max"] + 1
            or len(captured["frame_records"]) > limits["rgb_frames_per_shot_max"]
            or [f["fixed_step"] for f in captured["frame_records"]]
                != [f["fixed_step"] for f in trace.manifest["frame_records"]]):
        raise ValueError("native clock, bounds or trace/observation alignment differs")
    initial = next(trace.trace.chunks())["fixed_step_samples"][0]
    if not set(member["generated_slots"]).issubset({e["scenario_object_id"] for e in initial["entities"]}):
        raise ValueError("initial capture omitted an authored entity")
    frame, first = snapshot["frame_records"][0], captured["frame_records"][0]
    times = [frame["fixed_time_seconds"], first["fixed_time_seconds"]]
    evidence = result["policy"]
    if (not np.isfinite(times).all() or times[0] > times[1]
            or evidence["decision_time"] != times[0] or evidence["executed_action_time"] != times[1]
            or evidence["executed_actions"] != 1 or evidence["decisions"] != 1
            or evidence["observations"] != 1 + len(captured["frame_records"])):
        raise ValueError("causal observation/action callback inventory differs")
    image_path = snapshot_root / frame["agent_observation"]["relative_path"]
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    if rgb.shape != (480, 640, 3) or np.ptp(rgb) == 0:
        raise ValueError("decision RGB has wrong dimensions or is constant")
    return {**{key: member[key] for key in ("base_cluster", "scenario", "generation_seed", "engine_seed", "exposure_role")},
            "observation_configuration": snapshot["observation_configuration"],
            "viewport": frame["capture_metadata"]["viewport"],
            "transform": frame["capture_metadata"]["world_to_observation_transform"],
            "decision_time": times[0], "capture_time": times[1],
            "rgb": rgb, "initial": initial, "decision_image_path": str(image_path)}
