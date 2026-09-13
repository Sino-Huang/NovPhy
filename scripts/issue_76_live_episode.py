"""Native gameplay orchestration; policy callbacks receive only agent RGB/time/actions."""
import os
from pathlib import Path
import select
import shutil
import subprocess
import time

from scripts import issue_76_censored_episode as capture
from scripts import issue_76_expansion as files
from scripts.issue_76_native_outcomes import terminal_evidence, gameplay_outcome
from scripts.observation_trace import MANIFEST_NAME, _configuration, _plain_json, _transform, persist_observation_trace
from scripts.run_issue_62_successor_cohort import _replace_json
from src.webui.bridge import ScienceBirdsBridge
from world_model.planning.native_gameplay import agent_image_tensor


def start_display(log_path):
    """Let Xvnc reserve its own display; do not race random display probes."""
    reader, writer = os.pipe()
    process = None
    try:
        with Path(log_path).open("ab") as log:
            process = subprocess.Popen(
                ["Xvnc", "-displayfd", str(writer), "-geometry", "1024x768", "-depth", "24",
                 "-SecurityTypes", "None", "-rfbport", "0", "-localhost"],
                pass_fds=(writer,), stdout=log, stderr=subprocess.STDOUT)
        os.close(writer)
        writer = None
        if not select.select([reader], [], [], 15)[0]:
            raise TimeoutError("Xvnc did not allocate a display within 15 seconds")
        number = os.read(reader, 32).decode().strip()
        if not number.isdecimal() or process.poll() is not None:
            raise RuntimeError("Xvnc failed to reserve a display; see display.log")
        return ":" + number, process
    except Exception:
        if process is not None:
            capture.old.capture.terminate(process)
        raise
    finally:
        os.close(reader)
        if writer is not None:
            os.close(writer)


def observe_snapshot(endpoint, destination, scenario, identity, role, policy):
    """Persist provenance outside the policy; expose only transformed RGB and time."""
    frame = endpoint.get_observation_capture()
    metadata = _plain_json(frame.metadata)
    manifest = persist_observation_trace(destination,
        [{**metadata, "canonical_png": frame.canonical_png}],
        observation_configuration=capture.old.capture.OBSERVATION_CONFIGURATION,
        source_bindings=capture.old.capture._observation_bindings(scenario, rollout_identity=identity),
        exposure_role=role)
    configuration = _configuration(capture.old.capture.OBSERVATION_CONFIGURATION)
    policy.observe(_transform(frame.canonical_png, configuration), metadata["fixed_time_seconds"])
    return manifest["identity"]


def consume_executed_segment(observation_root, action, policy):
    """Replay accepted action history before the NEXT decision, never a candidate.

    The first capture frame is the recorded pre-intervention image. Insert the
    accepted action immediately after it, as in fitting. The endpoint snapshot
    used to choose this shot is an additional, earlier real observation.
    """
    manifest = files.read(Path(observation_root) / MANIFEST_NAME)
    for index, frame in enumerate(manifest["frame_records"]):
        png = (Path(observation_root) / frame["agent_observation"]["relative_path"]).read_bytes()
        timestamp = frame["fixed_time_seconds"]
        policy.observe(png, timestamp)
        if index == 0:
            policy.executed(action, timestamp)


class HistoryPolicy:
    """Common RGB/history integration with an injected fixed or learned selector."""
    def __init__(self, history, selector):
        self.history, self.selector = history, selector
        self.perception_seconds = 0.

    def reset(self):
        self.history.reset()
        self.perception_seconds = 0.

    def observe(self, png, timestamp):
        started = time.monotonic()
        self.history.observe(agent_image_tensor(png), timestamp)
        self.synchronize()
        self.perception_seconds += time.monotonic() - started

    def executed(self, action, timestamp):
        started = time.monotonic()
        self.history.record_executed_action(action, timestamp)
        self.synchronize()
        self.perception_seconds += time.monotonic() - started

    def synchronize(self):
        if self.history.device.type == "cuda":
            import torch
            torch.cuda.synchronize(self.history.device)

    def choose(self):
        return self.selector(self.history.current.clone())

    def evidence(self):
        return {"observations": self.history.observations,
                "executed_action_events": self.history.executed_actions,
                "perception_and_history_seconds": self.perception_seconds,
                "full_flops_measured": False, "matched_compute_established": False}


def play_episode(output, member, limits, policy):
    output = Path(output)
    root = output / "attempts" / member["identity"]
    root.mkdir(parents=True)  # Existing attempts must never be replayed implicitly.
    result = {"schema": "issue_76_live_episode_v1", "member_identity": member["identity"],
              "base_cluster": member["base_cluster"], "exposure_role": member["exposure_role"],
              "engine_seed": member["engine_seed"], "segments": [], "decisions": [],
              "attempted_shots": [], "failure": None, "fresh_access": False}
    started = time.monotonic()
    bridge = endpoint = engine = display_process = None
    terminals = []
    policy.reset()
    environment = {k: os.environ.get(k) for k in (
        "DISPLAY", "XDG_DATA_HOME", "NOVPHY_PHYSICS_CAPTURE_PORT",
        "NOVPHY_PHYSICS_CAPTURE_V2_STRIDE", "NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT",
        "NOVPHY_ENVIRONMENT_SEED")}
    try:
        game = root / "runtime"
        shutil.copytree(output / "player", game)
        files.install_level(game, member)
        _, scenario = files.materialize(member, member["template"], root / "authority")
        if scenario.to_dict() != member["scenario"]:
            raise ValueError("live member differs from frozen scenario authority")
        display, display_process = start_display(root / "display.log")
        ports = set()
        while len(ports) < 3:
            ports.add(capture.old.capture.free_port())
        agent_port, game_port, physics_port = sorted(ports)
        aligned = root / "aligned"
        os.environ.update(DISPLAY=display, XDG_DATA_HOME=str(root / "xdg"),
            NOVPHY_PHYSICS_CAPTURE_PORT=str(physics_port), NOVPHY_PHYSICS_CAPTURE_V2_STRIDE="50",
            NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT=str(aligned),
            NOVPHY_ENVIRONMENT_SEED=str(member["engine_seed"]))
        result["ports"] = {"agent": agent_port, "game": game_port, "physics": physics_port}
        engine = capture.old.capture.start_engine(game, False, agent_port=agent_port,
                                                  game_port=game_port, physics_port=physics_port)
        result["engine_log"] = str(engine.novphy_log_file.name)
        files.write(root / "runtime.json", result)
        bridge = capture.old.capture.connect_with_retry("127.0.0.1", agent_port, timeout=180, deadline_seconds=60)
        endpoint = ScienceBirdsBridge("127.0.0.1", physics_port, timeout=30)
        bridge.configure(760001, capture.old.capture.PlayingMode.TRAINING)
        bridge.set_speed(1)
        capture.old.capture.prepare_for_play(bridge, timeout=60, poll_delay=.5)
        if bridge.get_current_level() != 1:
            raise ValueError("live episode did not load its assigned level")
        for index in range(1, member["maximum_shots"] + 1):
            if time.monotonic() - started >= limits["attempt_seconds"]:
                raise TimeoutError("declared episode wall limit")
            state = bridge.get_game_state().name
            if state != "PLAYING":
                result["stopping_reason"] = "interface_" + state.lower()
                break
            decision_started = time.monotonic()
            # Readiness uses only a public actuator anchor; no shot is sent here.
            capture.old.prepare_action(bridge, limits["readiness_action"])
            identity = member["identity"] + f":shot-{index}"
            snapshot = observe_snapshot(endpoint, root / f"decision-{index}", scenario,
                                        identity, member["exposure_role"], policy)
            decision = policy.choose()
            action = decision["action"]
            decision.update(observation_manifest=snapshot,
                            wall_seconds_with_readiness=time.monotonic() - decision_started)
            result["decisions"].append(decision)
            result["attempted_shots"].append(index)
            _replace_json(result, root / "progress.json")
            segment = capture.capture_segment(bridge, aligned, root / f"shot-{index}", member,
                                             scenario, identity, action, limits["shot_seconds"])
            result["segments"].append(segment)
            terminal = terminal_evidence(segment)
            terminals.append(terminal)
            consume_executed_segment(root / f"shot-{index}/observation-trace", action, policy)
            _replace_json(result, root / "progress.json")
            capture.old.log(f"live {identity} terminal={terminal} censored={segment['summary']['censored']}")
            if segment["summary"]["censored"]:
                result["stopping_reason"] = "native_time_window_limit"
                break
            if terminal and terminal["reason"] in ("level_clear", "level_fail"):
                result["stopping_reason"] = terminal["reason"]
                break
        result.setdefault("stopping_reason", "frozen_shot_budget")
    except Exception as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        capture.old.log(f"live failure retained {member['identity']}: {result['failure']}")
    finally:
        for connection in (bridge, endpoint):
            if connection is not None:
                connection.disconnect()
        capture.old.capture.stop_started_engine(engine)
        if display_process is not None:
            capture.old.capture.terminate(display_process)
        for key, value in environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    result["complete"] = result["failure"] is None
    # On a failed terminal-validation path, keep a failure, never synthesize a win.
    terminals.extend([None] * (len(result["segments"]) - len(terminals)))
    result["outcome"] = gameplay_outcome(result, terminals)
    result["gameplay_success"] = result["outcome"]["success"]
    result["gameplay_failure_penalty"] = int(not result["gameplay_success"])
    result["unattempted_shots"] = [i for i in range(1, member["maximum_shots"] + 1)
                                   if i not in result["attempted_shots"]]
    result["policy"] = policy.evidence()
    result["wall_seconds"] = time.monotonic() - started
    files.write(output / "results" / (member["identity"] + ".json"), result)
    return result
