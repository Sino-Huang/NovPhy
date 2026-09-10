"""Bounded multi-shot capture on the versioned canonical observer player.

Episode continuity is explicit. The old single-shot captures and collector are
unchanged; raw aligned frames are retained, not removed after derivation.
"""
import json
from io import BytesIO
import os
from pathlib import Path
import shutil
import time
from PIL import Image

from scripts import issue_76_expansion as expansion
from scripts import run_issue_62_successor_cohort as capture
from scripts.slingshot_readiness import (
    PreparedScreenShot, _anchored_action_and_command, _stable_observation,
    adapt_slingshot_client,
)
from scripts.collect_rollouts import persist_physics_capture_v2, validate_physics_capture_v2_artifact
from scripts.observation_trace import load_aligned_observation_captures, persist_observation_trace
from src.webui.bridge import PhysicsCaptureV2Failure


def log(message):
    print(f"[canonical-episode] {message}", flush=True)


def prepare_action(bridge, action, *, clock=time.monotonic, sleeper=time.sleep):
    """Use only the public actuator anchor, at speed 1 throughout preparation.

    The older helper accelerates startup to speed 50. That is not legal for
    this manual-physics capture profile. No pig/block geometry selects actions.
    """
    adapted = adapt_slingshot_client(bridge)
    adapted.set_simulation_speed(1)
    adapted.fully_zoom_out()
    observation, count = _stable_observation(adapted, frame_height=480,
        deadline=clock() + 30, poll_interval=.25, clock=clock, sleeper=sleeper)
    requested = {"drag_release": [action["drag_x"], action["drag_y"]],
                 "coordinate_frame": "slingshot_relative", "tap_time": action["tap_time_ms"],
                 "release_time": action["release_time_ms"]}
    anchored, command = _anchored_action_and_command(requested, observation, 480)
    return PreparedScreenShot(adapted, anchored, command, observation,
                              {"speed": 1, "stable_observations": count,
                               "action_selection": "prospectively fixed; slingshot anchor only"},
                              fast=True, record_ground_truth=False)


def persist_segment(physics, aligned, destination, scenario, identity):
    bindings = capture._source_bindings(scenario, rollout_identity=identity,
                                        action_identity=identity + ":action")
    metadata = persist_physics_capture_v2(destination, physics, source_bindings=bindings,
                                         scenario_manifest_identity=scenario.identity)
    validate_physics_capture_v2_artifact(destination, metadata)
    captures = load_aligned_observation_captures(aligned, physics)
    for frame in captures:
        with Image.open(BytesIO(frame["canonical_png"])) as image:
            if image.size != (640, 480) or not any(high > low for low, high in image.convert("RGB").getextrema()):
                raise ValueError("canonical engineering observation has wrong viewport or constant RGB")
    observed = persist_observation_trace(destination / "observation-trace", captures,
        observation_configuration=capture.OBSERVATION_CONFIGURATION,
        source_bindings=capture._observation_bindings(scenario, rollout_identity=identity),
        exposure_role="calibration")
    record = expansion.read(destination / "physics_capture_v2.json")
    events = record["events"]
    samples = record["fixed_step_samples"]
    launches = [e for e in events if e["event_type"] == "bird_launched"]
    if len(launches) != 1:
        raise ValueError("a shot segment must contain exactly one actual bird launch")
    if [f["fixed_step"] for f in observed["frame_records"]] != [s["fixed_step"] for s in samples]:
        raise ValueError("shot segment observations and physics are not exactly aligned")
    return {"identity": identity, "capture_id": physics["capture_id"],
            "frames": len(samples), "first_fixed_step": samples[0]["fixed_step"],
            "last_fixed_step": samples[-1]["fixed_step"],
            "terminal_reason": record["terminal_evidence"]["reason"],
            "launch": launches[0], "pig_removed": sum(e["event_type"] == "pig_removed" for e in events),
            "level_clear": any(e["event_type"] == "level_clear" for e in events),
            "level_fail": any(e["event_type"] == "level_fail" for e in events),
            "initial_entity_ids": [e["scenario_object_id"] for e in samples[0]["entities"]],
            "final_entity_ids": [e["scenario_object_id"] for e in samples[-1]["entities"] if e["lifecycle"] == "active"],
            "observation_manifest": observed["identity"], "raw_aligned_root": str(aligned)}


def capture_segment(bridge, physics_bridge, aligned_root, destination, scenario,
                    identity, action, *, timeout=180, clock=time.monotonic, sleeper=time.sleep):
    prepared = prepare_action(bridge, action, clock=clock, sleeper=sleeper)
    previous = set(aligned_root.iterdir()) if aligned_root.exists() else set()
    started = last_log = clock()
    response = prepared.execute()
    if response != 1:
        raise ValueError(f"shot transport did not accept the intervention: {response}")
    created = set(aligned_root.iterdir()) - previous
    if len(created) != 1:
        raise ValueError("accepted shot did not create exactly one new aligned capture directory")
    aligned = created.pop()
    while True:
        try:
            physics = physics_bridge.get_physics_capture_v2().record
            break
        except PhysicsCaptureV2Failure as error:
            if error.code != 3 or clock() - started >= timeout:
                raise
        now = clock()
        if now - last_log >= 5:
            log(f"{identity} recording frames={sum(1 for _ in aligned.glob('frame_*.json'))} elapsed={now-started:.1f}s")
            last_log = now
        sleeper(.25)
    if physics["capture_id"] != aligned.name:
        raise ValueError("request 71 returned a stale or different shot capture")
    result = persist_segment(physics, aligned, destination, scenario, identity)
    if result["last_fixed_step"] - result["first_fixed_step"] > 600:
        raise ValueError("finalized shot exceeded the frozen fixed-step cap; all frames retained")
    result.update({"action": prepared.action, "socket_command": prepared.socket_command,
                   "actuator_readiness": prepared.evidence, "wall_seconds": clock() - started})
    return result


def capture_episode(output, member, limits):
    output = Path(output)
    root = output / "attempts" / member["identity"]
    root.mkdir(parents=True)
    game = root / "runtime"
    result = {"schema": "issue_76_canonical_episode_v1", "member_identity": member["identity"],
              "base_cluster": member["base_cluster"], "exposure_role": "calibration",
              "segments": [], "failure": None, "unattempted_shots": [], "attempted_shots": [],
              "engine_seed": member["engine_seed"], "agent_id": 760001,
              "fresh_evaluation_opened": False, "model_fitting": False}
    started = time.monotonic()
    engine = bridge = display_process = None
    try:
        shutil.copytree(output / "player", game)
        expansion.install_level(game, member)
        _, scenario = expansion.materialize(member, member["template"], output / "authorities" / member["identity"])
        if scenario.to_dict() != member["scenario"]:
            raise ValueError("materialized scenario differs from the freeze")
        display, display_process = capture.start_display(root / "display.log")
        os.environ["DISPLAY"] = display
        ports = set()
        while len(ports) < 3:
            ports.add(capture.free_port())
        agent_port, game_port, physics_port = sorted(ports)
        aligned = root / "aligned"
        os.environ.update(NOVPHY_PHYSICS_CAPTURE_PORT=str(physics_port),
                          NOVPHY_PHYSICS_CAPTURE_V2_STRIDE="1",
                          NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT=str(aligned),
                          NOVPHY_ENVIRONMENT_SEED=str(member["engine_seed"]),
                          XDG_DATA_HOME=str(root / "xdg"))
        result["ports"] = {"agent": agent_port, "game": game_port, "physics": physics_port}
        expansion.write(root / "runtime.json", result)
        engine = capture.start_engine(game, False, agent_port=agent_port, game_port=game_port, physics_port=physics_port)
        result["engine_log"] = str(engine.novphy_log_file.name)
        bridge = capture.connect_with_retry("127.0.0.1", agent_port, timeout=180, deadline_seconds=60)
        bridge.configure(result["agent_id"], capture.PlayingMode.TRAINING)
        bridge.set_speed(1)
        capture.prepare_for_play(bridge, timeout=60, poll_delay=.5)
        if bridge.get_current_level() != 1:
            raise ValueError("episode did not load its sole frozen level")
        for index, action in enumerate(member["actions"], 1):
            state = bridge.get_game_state().name
            if state in ("WON", "LOST"):
                result["unattempted_shots"] = list(range(index, len(member["actions"]) + 1))
                result["stopping_reason"] = state.lower()
                break
            identity = member["identity"] + f":shot-{index}"
            result["attempted_shots"].append(index)
            from scripts.run_issue_62_successor_cohort import _replace_json
            _replace_json(result, root / "progress.json")
            log(f"start {identity}/{len(member['actions'])}")
            shot = capture_segment(bridge, capture.ScienceBirdsBridge("127.0.0.1", physics_port, timeout=180),
                                   aligned, root / f"shot-{index}", scenario, identity, action,
                                   timeout=limits["shot_seconds"])
            if result["segments"] and shot["first_fixed_step"] <= result["segments"][-1]["last_fixed_step"]:
                raise ValueError("episode clock reset or overlapped between shots")
            if index == 1 and not set(member["generated_slots"]).issubset(shot["initial_entity_ids"]):
                raise ValueError("initial capture omitted authored scenario objects")
            result["segments"].append(shot)
            expansion.write(root / f"shot-{index}.json", shot)
            log(f"complete {identity} frames={shot['frames']} terminal={shot['terminal_reason']}")
        result["game_state_after"] = bridge.get_game_state().name
        result.setdefault("stopping_reason", "frozen_shot_budget")
    except Exception as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        result["unattempted_shots"] = [i for i in range(1, len(member["actions"]) + 1) if i not in result["attempted_shots"]]
        log(f"failure retained {member['identity']}: {result['failure']}")
    finally:
        if bridge is not None:
            bridge.disconnect()
        capture.stop_started_engine(engine)
        if display_process is not None:
            capture.terminate(display_process)
    result["wall_seconds"] = time.monotonic() - started
    result["complete"] = result["failure"] is None
    expansion.write(output / "results" / (member["identity"] + ".json"), result)
