"""Prospective terminal-or-censored collection; preserves the original native collector."""
from io import BytesIO
import os
from pathlib import Path
import shutil
import time

from PIL import Image

from scripts import issue_76_expansion as files
from scripts import issue_76_episode_capture as old
from scripts.native_segment_trace import NativeSegmentTrace
from scripts.observation_trace import load_aligned_observation_captures, persist_observation_trace
from scripts.run_issue_62_successor_cohort import _replace_json


def capture_segment(bridge, aligned, destination, member, scenario, identity, action, timeout):
    prepared = old.prepare_action(bridge, action)
    previous = set(aligned.iterdir()) if aligned.exists() else set()
    started = last_log = time.monotonic()
    response = prepared.execute()
    if response != 1:
        raise ValueError(f"native shot was not accepted: {response}")
    added = set(aligned.iterdir()) - previous
    if len(added) != 1:
        raise ValueError("native shot did not create one new capture directory")
    raw = added.pop()
    while not (raw / "native-manifest.json").exists():
        now = time.monotonic()
        if now - started > timeout:
            raise TimeoutError("native shot manifest deadline; partial chunks and RGB retained")
        if now - last_log >= 5:
            old.log(f"{identity} native_chunks={sum(1 for _ in (raw/'native').glob('chunk-*.json.gz'))} RGB={sum(1 for _ in raw.glob('frame_*.json'))} wall={now-started:.1f}s")
            last_log = now
        time.sleep(.25)
    trace = NativeSegmentTrace(raw)
    if trace.manifest["capture_id"] != raw.name or int(trace.manifest["engine_seed"]) != member["engine_seed"]:
        raise ValueError("native manifest capture/seed binding differs")
    summary = trace.summary
    if summary["event_counts"].get("bird_launched", 0) != 1:
        raise ValueError("native segment does not contain exactly one actual launch")
    captures = load_aligned_observation_captures(raw, trace.manifest)
    for frame in captures:
        with Image.open(BytesIO(frame["canonical_png"])) as image:
            if image.size != (640, 480) or not any(b > a for a, b in image.convert("RGB").getextrema()):
                raise ValueError("native RGB has wrong viewport or is constant")
    destination.mkdir(parents=True)
    observed = persist_observation_trace(destination / "observation-trace", captures,
        observation_configuration=old.capture.OBSERVATION_CONFIGURATION,
        source_bindings=old.capture._observation_bindings(scenario, rollout_identity=identity),
        exposure_role=member["exposure_role"])
    initial = next(trace.observed_samples())
    result = {"identity": identity, "capture_id": raw.name, "native_root": str(raw),
              "summary": summary, "initial_entity_ids": [e["scenario_object_id"] for e in initial["entities"]],
              "observation_manifest": observed["identity"], "frame_count": len(observed["frame_records"]),
              "action": prepared.action, "socket_command": prepared.socket_command,
              "actuator_readiness": prepared.evidence, "wall_seconds": time.monotonic() - started,
              "source_bindings": old.capture._source_bindings(scenario, rollout_identity=identity,
                                                              action_identity=identity + ":action")}
    files.write(destination / "segment.json", result)
    return result


def capture_episode(output, member, limits):
    output = Path(output)
    root = output / "attempts" / member["identity"]
    root.mkdir(parents=True)
    result = {"schema": "issue_76_censored_episode_v1", "member_identity": member["identity"],
              "base_cluster": member["base_cluster"], "exposure_role": member["exposure_role"],
              "segments": [], "attempted_shots": [], "unattempted_shots": [], "failure": None,
              "engine_seed": member["engine_seed"], "fresh_access": False}
    started = time.monotonic()
    bridge = engine = display_process = None
    try:
        game = root / "runtime"
        shutil.copytree(output / "player", game)
        files.install_level(game, member)
        _, scenario = files.materialize(member, member["template"], root / "authority")
        if scenario.to_dict() != member["scenario"]:
            raise ValueError("native member differs from frozen authority")
        display, display_process = old.capture.start_display(root / "display.log")
        ports = set()
        while len(ports) < 3:
            ports.add(old.capture.free_port())
        agent_port, game_port, physics_port = sorted(ports)
        aligned = root / "aligned"
        os.environ.update(DISPLAY=display, XDG_DATA_HOME=str(root / "xdg"),
                          NOVPHY_PHYSICS_CAPTURE_PORT=str(physics_port), NOVPHY_PHYSICS_CAPTURE_V2_STRIDE="50",
                          NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT=str(aligned),
                          NOVPHY_ENVIRONMENT_SEED=str(member["engine_seed"]))
        result["ports"] = {"agent": agent_port, "game": game_port, "physics": physics_port}
        engine = old.capture.start_engine(game, False, agent_port=agent_port, game_port=game_port, physics_port=physics_port)
        result["engine_log"] = str(engine.novphy_log_file.name)
        files.write(root / "runtime.json", result)
        bridge = old.capture.connect_with_retry("127.0.0.1", agent_port, timeout=180, deadline_seconds=60)
        bridge.configure(760001, old.capture.PlayingMode.TRAINING)
        bridge.set_speed(1)
        old.capture.prepare_for_play(bridge, timeout=60, poll_delay=.5)
        if bridge.get_current_level() != 1:
            raise ValueError("native episode did not load its assigned single level")
        for index, action in enumerate(member["actions"], 1):
            state = bridge.get_game_state().name
            if state in ("WON", "LOST"):
                result["stopping_reason"] = state.lower()
                break
            result["attempted_shots"].append(index)
            _replace_json(result, root / "progress.json")
            identity = member["identity"] + f":shot-{index}"
            old.log(f"start native {identity}/{len(member['actions'])}")
            segment = capture_segment(bridge, aligned, root / f"shot-{index}", member, scenario,
                                      identity, action, limits["shot_seconds"])
            if index == 1 and not set(member["generated_slots"]).issubset(segment["initial_entity_ids"]):
                raise ValueError("native initial frame omitted authored objects")
            if result["segments"] and segment["summary"]["first_fixed_step"] <= result["segments"][-1]["summary"]["last_fixed_step"]:
                raise ValueError("native episode clock reset or overlapped between shots")
            result["segments"].append(segment)
            files.write(root / f"shot-{index}.json", segment)
            old.log(f"observed native {identity} steps={segment['summary']['sample_count']} RGB={segment['frame_count']} seconds={segment['summary']['physical_seconds']:.4f} censored={segment['summary']['censored']}")
            if segment["summary"]["censored"]:
                result["stopping_reason"] = "native_time_window_limit"
                break
        result["game_state_after"] = bridge.get_game_state().name
        result.setdefault("stopping_reason", "frozen_shot_budget")
    except Exception as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        old.log(f"native failure retained {member['identity']}: {result['failure']}")
    finally:
        if bridge is not None:
            bridge.disconnect()
        old.capture.stop_started_engine(engine)
        if display_process is not None:
            old.capture.terminate(display_process)
    result["unattempted_shots"] = [i for i in range(1, len(member["actions"]) + 1) if i not in result["attempted_shots"]]
    result["complete"] = result["failure"] is None  # Collection integrity, not gameplay success.
    result["gameplay_success"] = (result["complete"] and result.get("game_state_after") == "WON"
                                  and result.get("stopping_reason") != "native_time_window_limit")
    result["gameplay_failure_penalty"] = 0 if result["gameplay_success"] else 1
    result["wall_seconds"] = time.monotonic() - started
    files.write(output / "results" / (member["identity"] + ".json"), result)

