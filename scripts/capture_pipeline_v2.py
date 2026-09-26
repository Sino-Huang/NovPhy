"""Fixed rendered branch-capture pipeline (issue #109, work item 2).

Root causes addressed, each evidenced from retained artifacts
(``docs/issue-109-data-foundation.md``):

* ``TimeoutError: timed out`` (#87, 51/51 cases): a port-reservation race.
  Ports were probed free and released, then bound by Unity 10-20 s later, so
  cells dispatched within that window received identical ports; the loser's
  physics listener failed with ``Address already in use`` and its bridge
  blocked for the 180 s socket timeout. Here every worker slot owns a
  disjoint port block, and a bind conflict in the engine log fails fast.
* ``corrected single differs from the latest corrected history frame`` and
  ``state/RGB advanced during readiness or hold``: the physics state is
  identical, but the bird/pig blink animation (``ABCharacter.Blink``) is
  rendered at a different animator phase by the fixed-step barrier render and
  by the later frame-time endpoint render. Here RGB invariance is required
  everywhere except inside the projected screen boxes of Bird/Pig nodes, and
  the corrected-single view is the barrier-sealed frame itself.
* ``native shot manifest deadline`` (180 s wall): the engine was still
  progressing (446-601 of 601 RGB frames) when the fixed deadline fired, so
  long (right-censored) branches were cut preferentially. Here the shot wait
  fails only on a progress stall, under a generous wall cap.
* ``did not reach PLAYING before timeout``: cold-start contention. Engine cold
  starts are serialized through a file-lock gate and bounds are extended.

Every attempt records per-stage wall seconds for throughput profiling.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
from io import BytesIO
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import socket
import time

import numpy as np
from PIL import Image

PIPELINE_IDENTITY = "capture-pipeline-v2"
RESULT_SCHEMA = "capture_pipeline_v2_branch_v1"
PAIRED_SCHEMA = "capture_pipeline_v2_paired_inputs_v1"
PORT_BASE = 29100            # outside the Linux ephemeral range (32768-60999)
PORT_SLOT_STRIDE = 30        # ten 3-port sub-blocks per worker slot
PORT_SUBBLOCKS = 10
CHARACTER_BOX_PAD_PIXELS = 3
CHARACTER_CLASS_PREFIXES = ("Bird", "Pig")
DEFAULT_LIMITS = {
    "connect_deadline_seconds": 120,
    "agent_socket_timeout_seconds": 180,
    "menu_ready_seconds": 180,
    "history_ready_seconds": 300,
    "inference_hold_seconds": 2,
    "shot_stall_seconds": 120,
    "shot_wall_cap_seconds": 900,
    "cold_start_gate": True,
}
FAILURE_CLASSES = (
    ("port_bind_conflict", ("PortBindConflict",)),
    ("port_block_unavailable", ("no free port sub-block",)),
    ("engine_connect_timeout", ("Could not connect to Science Birds",)),
    ("menu_readiness_timeout", ("did not reach PLAYING",)),
    ("barrier_readiness_timeout", ("native decision barrier readiness deadline",)),
    ("render_invariance_violation", ("differs outside animated character sprites",)),
    ("decision_state_advanced", ("decision physics state advanced",)),
    ("shot_progress_stall", ("native shot progress stalled",)),
    ("shot_wall_cap", ("native shot wall cap",)),
    ("socket_timeout", ("TimeoutError: timed out",)),
    ("attempt_wall_limit", ("attempt_wall_limit",)),
    ("worker_memory_limit", ("worker_memory_limit", "aggregate_memory_limit")),
    ("supervisor_interrupted", ("supervisor_interrupted",)),
)
TERMINAL_KINDS = {"level_clear": "native_clear", "level_fail": "native_fail",
                  "stable_entered": "stable_without_clear"}


class PortBindConflict(RuntimeError):
    """The engine could not bind one of its assigned ports."""


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def failure_class(failure):
    """Typed failure taxonomy over the recorded failure string."""
    if failure is None:
        return None
    text = str(failure)
    for name, needles in FAILURE_CLASSES:
        if any(needle in text for needle in needles):
            return name
    return "other"


# ------------------------------------------------------------------ ports

def slot_ports(slot, sequence):
    """Deterministic disjoint (agent, game, physics) ports for one attempt."""
    base = PORT_BASE + slot * PORT_SLOT_STRIDE + 3 * (sequence % PORT_SUBBLOCKS)
    return base, base + 1, base + 2


def ports_free(ports):
    for port in ports:
        for host in ("127.0.0.1", "0.0.0.0"):
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                probe.bind((host, port))
            except OSError:
                return False
            finally:
                probe.close()
    return True


def choose_ports(slot, sequence):
    """First free sub-block of this slot, starting at the sequence position.

    Blocks are slot-private, so the choice never races another worker; a busy
    sub-block (e.g. a lingering socket from the slot's previous attempt) is
    skipped deterministically, never retried against the same ports.
    """
    for offset in range(PORT_SUBBLOCKS):
        ports = slot_ports(slot, sequence + offset)
        if ports_free(ports):
            return ports, offset
    raise RuntimeError(f"no free port sub-block in slot {slot}")


def engine_bind_conflict(runtime):
    for log in Path(runtime).glob("sciencebirds_*.log"):
        try:
            if "Address already in use" in log.read_text(errors="ignore"):
                return log.name
        except OSError:
            continue
    return None


# ------------------------------------------------------ render invariance

def character_boxes(legacy_state, pad=CHARACTER_BOX_PAD_PIXELS):
    """Screen boxes (x0, y0, x1, y1 inclusive) of animated Bird/Pig nodes."""
    if legacy_state["coordinates"]["screen_origin"] != "top_left":
        raise ValueError("physics capture screen origin differs from the RGB origin")
    boxes = []
    for node in legacy_state["nodes"]:
        if not str(node["object_class"]).startswith(CHARACTER_CLASS_PREFIXES):
            continue
        xs = [point["x"] for point in node["screen_polygon"]]
        ys = [point["y"] for point in node["screen_polygon"]]
        if xs and ys:
            boxes.append([min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad])
    return boxes


def _rgb(png):
    with Image.open(BytesIO(png)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.int16)


def render_invariance(reference_png, candidate_png, boxes):
    """Pixel-exact equality outside animated character boxes.

    Returns evidence; raises when any pixel outside the boxes differs.
    """
    if reference_png == candidate_png:
        return {"byte_equal": True, "differing_pixels": 0, "outside_pixels": 0}
    reference, candidate = _rgb(reference_png), _rgb(candidate_png)
    if reference.shape != candidate.shape:
        raise ValueError("decision RGB differs outside animated character sprites (shape)")
    differing = np.any(reference != candidate, axis=-1)
    allowed = np.zeros_like(differing)
    height, width = differing.shape
    for x0, y0, x1, y1 in boxes:
        allowed[max(0, y0):min(height, y1 + 1), max(0, x0):min(width, x1 + 1)] = True
    outside = int((differing & ~allowed).sum())
    evidence = {"byte_equal": False, "differing_pixels": int(differing.sum()),
                "outside_pixels": outside}
    if outside:
        raise ValueError(f"decision RGB differs outside animated character sprites "
                         f"({outside} pixels)")
    return evidence


# ---------------------------------------------------------- cold starts

@contextmanager
def cold_start_gate(path, enabled=True):
    """Serialize engine cold starts across workers (file lock)."""
    if not enabled:
        yield 0.0
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield time.monotonic() - began
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class StageClock:
    def __init__(self):
        self.started = self.mark = time.monotonic()
        self.stages = {}

    def lap(self, name):
        now = time.monotonic()
        self.stages[name] = self.stages.get(name, 0.0) + now - self.mark
        self.mark = now


# ---------------------------------------------------------------- shot

def wait_for_manifest(raw, stall_seconds, cap_seconds, identity, log):
    """Wait for the native manifest; fail only on a progress stall or wall cap."""
    started = last_progress = last_log = time.monotonic()
    progress = None
    while not (raw / "native-manifest.json").exists():
        now = time.monotonic()
        chunks = sum(1 for _ in (raw / "native").glob("chunk-*.json.gz"))
        frames = sum(1 for _ in raw.glob("frame_*.json"))
        if (chunks, frames) != progress:
            progress, last_progress = (chunks, frames), now
        if now - last_progress > stall_seconds:
            raise TimeoutError(f"native shot progress stalled for {stall_seconds}s at "
                               f"chunks={chunks} RGB={frames}; partial capture retained")
        if now - started > cap_seconds:
            raise TimeoutError(f"native shot wall cap {cap_seconds}s at chunks={chunks} "
                               f"RGB={frames}; partial capture retained")
        if now - last_log >= 30:
            log(f"{identity} native_chunks={chunks} RGB={frames} wall={now - started:.0f}s")
            last_log = now
        time.sleep(.25)
    return time.monotonic() - started


def capture_segment(bridge, aligned, destination, member, scenario, identity, action, limits, log):
    """The #76 censored shot collector with a progress-based manifest wait."""
    from scripts import issue_76_censored_episode as censored
    from scripts.native_segment_trace import NativeSegmentTrace
    from scripts.observation_trace import load_aligned_observation_captures, persist_observation_trace
    old = censored.old
    prepared = old.prepare_action(bridge, action)
    previous = set(aligned.iterdir()) if aligned.exists() else set()
    started = time.monotonic()
    response = prepared.execute()
    if response != 1:
        raise ValueError(f"native shot was not accepted: {response}")
    added = set(aligned.iterdir()) - previous
    if len(added) != 1:
        raise ValueError("native shot did not create one new capture directory")
    raw = added.pop()
    manifest_wait = wait_for_manifest(raw, limits["shot_stall_seconds"],
                                      limits["shot_wall_cap_seconds"], identity, log)
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
              "manifest_wait_seconds": manifest_wait,
              "source_bindings": old.capture._source_bindings(scenario, rollout_identity=identity,
                                                              action_identity=identity + ":action")}
    write_json(destination / "segment.json", result)
    return result, trace


# ------------------------------------------------------------ paired views

def seal_paired_inputs(destination, member, decision_fixed_step, legacy, history_rows, endpoint_evidence):
    """Seal three views of one native decision state before the shot.

    corrected-single is the barrier-sealed decision frame (identical bytes to
    the last corrected-history frame); the frame-time endpoint renders are
    retained as invariance evidence only.
    """
    from scripts.issue_76_bounded_transfer_capture import _frame, _json_digest, _fixed_time
    from scripts.observation_trace import _plain_json
    legacy_state = _plain_json(legacy.state)
    rows = [_plain_json(row) for row in history_rows]
    if [row["fixed_step"] for row in rows] != [decision_fixed_step - 100, decision_fixed_step - 50,
                                                decision_fixed_step]:
        raise ValueError("corrected history must use native steps -100/-50/0")
    if any(abs(_fixed_time(b) - _fixed_time(a) - .02) > 1e-5 for a, b in zip(rows, rows[1:])):
        raise ValueError("corrected history must use actual twenty-millisecond spacing")
    if (legacy_state["fixed_step"] != decision_fixed_step
            or _fixed_time(legacy_state) != _fixed_time(rows[-1])):
        raise ValueError("decision physics state advanced: legacy capture is not the sealed decision state")
    decision = rows[-1]
    frames = {
        "legacy-single": [_frame("legacy-single/frame_000001.png", legacy.png, legacy_state,
                                 decision_fixed_step, "request_70_screen_capture_request_72_v1_equivalent")],
        "corrected-single": [_frame("corrected-single/frame_000001.png", decision["canonical_png"],
                                    decision, decision_fixed_step, "sealed_aligned_render_canonical_rgb")],
        "corrected-history": [_frame(f"corrected-history/frame_{ordinal:06}.png", row["canonical_png"],
                                     row, decision_fixed_step, "sealed_aligned_render_canonical_rgb")
                              for ordinal, row in enumerate(rows, 1)],
    }
    scenario_binding = {"member_identity": member["identity"], "base_cluster": member["base_cluster"],
                        "engine_seed": member["engine_seed"],
                        "scenario_sha256": _json_digest(_plain_json(member["scenario"]))}
    state_binding = {"legacy_capture_id": legacy_state["capture_id"],
                     "history_capture_id": decision["capture_id"],
                     "decision_fixed_step": decision_fixed_step,
                     "decision_fixed_time_seconds": _fixed_time(decision),
                     "physics_barrier_paused": True, "action_executed_before_capture": False}
    manifest = {
        "schema": PAIRED_SCHEMA,
        "identity": "paired-input-v2:" + _json_digest({"scenario": scenario_binding, "state": state_binding}),
        "scenario_binding": scenario_binding, "decision_state_binding": state_binding,
        "same_native_decision_state": True, "post_action_frames_used": False,
        "corrected_single_source": "barrier-sealed decision frame (bytes equal to corrected-history[-1])",
        "endpoint_render_invariance": endpoint_evidence,
        "views": {name: {"frames": value} for name, value in frames.items()},
    }
    destination = Path(destination)
    destination.mkdir(parents=True)
    pngs = {"legacy-single/frame_000001.png": legacy.png,
            "corrected-single/frame_000001.png": decision["canonical_png"]}
    pngs.update({f"corrected-history/frame_{ordinal:06}.png": row["canonical_png"]
                 for ordinal, row in enumerate(rows, 1)})
    for relative, png in pngs.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
    write_json(destination / "paired-inputs.json", manifest)
    return manifest


# -------------------------------------------------------------- collector

def capture_branch(output, record, limits, slot, sequence):
    """Capture one frozen branch; always persists a typed result."""
    from scripts import issue_76_display_start as display_start
    from scripts import issue_76_live_episode as live
    from scripts import issue_76_shared_history_capture as shared
    from scripts.issue_76_fixed_replay_policy import FixedReplayPolicy
    from scripts.issue_76_shared_player_storage import clone_player
    from scripts.observation_trace import _plain_json
    from scripts.process_lifecycle import cleanup_actions, persist_after_cleanup, record_cleanup_failures
    from src.webui.bridge import ScienceBirdsBridge

    files, old = live.files, live.capture.old
    limits = {**DEFAULT_LIMITS, **limits}
    output = Path(output).resolve()  # the engine resolves capture roots from its own cwd
    identity = record["identity"]
    root = output / "attempts" / identity
    root.mkdir(parents=True)
    log = lambda message: print(f"[{PIPELINE_IDENTITY}] {message}", flush=True)
    result = {"schema": RESULT_SCHEMA, "pipeline": PIPELINE_IDENTITY, "member_identity": identity,
              "source_member_identity": record.get("source_member_identity"),
              "exposure_role": record["exposure_role"], "base_cluster": record["base_cluster"],
              "failure": None, "failure_class": None, "segments": [], "slot": slot,
              "sequence": sequence, "fresh_access": False, "started_at_utc": utc_now()}
    clock = StageClock()
    bridge = endpoint = engine = display_process = None
    variables = ("DISPLAY", "XDG_DATA_HOME", "NOVPHY_PHYSICS_CAPTURE_PORT",
                 "NOVPHY_PHYSICS_CAPTURE_V2_STRIDE", "NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT",
                 "NOVPHY_ENVIRONMENT_SEED", shared.TARGET_VARIABLE)
    environment = {key: os.environ.get(key) for key in variables}
    step = limits["decision_fixed_step"]
    policy = FixedReplayPolicy(record["actions"][0])
    try:
        game = root / "runtime"
        clone_player(output / "player", game)
        files.install_level(game, record)
        _, scenario = files.materialize(record, record["template"], root / "authority")
        if scenario.to_dict() != record["scenario"]:
            raise ValueError("branch member differs from its frozen source authority")
        clock.lap("runtime_prepare")
        display, display_process = display_start.start_display(root / "display.log")
        clock.lap("display_start")
        (agent_port, game_port, physics_port), skipped = choose_ports(slot, sequence)
        result["ports"] = {"agent": agent_port, "game": game_port, "physics": physics_port,
                           "skipped_subblocks": skipped}
        aligned = root / "aligned"
        os.environ.update(DISPLAY=display, XDG_DATA_HOME=str(root / "xdg"),
                          NOVPHY_PHYSICS_CAPTURE_PORT=str(physics_port),
                          NOVPHY_PHYSICS_CAPTURE_V2_STRIDE="50",
                          NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT=str(aligned),
                          NOVPHY_ENVIRONMENT_SEED=str(record["engine_seed"]),
                          NOVPHY_NATIVE_DECISION_STEP=str(step))
        with cold_start_gate(output / "cold-start.lock", limits["cold_start_gate"]) as waited:
            result["cold_start_gate_wait_seconds"] = waited
            clock.lap("cold_start_gate_wait")
            engine = old.capture.start_engine(game, False, agent_port=agent_port,
                                              game_port=game_port, physics_port=physics_port)
            write_json(root / "runtime.json", result)
            deadline = time.monotonic() + limits["connect_deadline_seconds"]
            last_error = None
            while True:
                conflict = engine_bind_conflict(game)
                if conflict:
                    raise PortBindConflict(f"engine log {conflict}: Address already in use")
                candidate = ScienceBirdsBridge("127.0.0.1", agent_port,
                                               timeout=limits["agent_socket_timeout_seconds"])
                try:
                    candidate.connect()
                    bridge = candidate
                    break
                except OSError as error:
                    last_error = error
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Could not connect to Science Birds at 127.0.0.1:{agent_port}: "
                                       f"{last_error}")
                time.sleep(.5)
            clock.lap("engine_boot_connect")
            endpoint = ScienceBirdsBridge("127.0.0.1", physics_port, timeout=30)
            bridge.configure(760001, old.capture.PlayingMode.TRAINING)
            bridge.set_speed(1)
            old.capture.prepare_for_play(bridge, timeout=limits["menu_ready_seconds"], poll_delay=.5)
            conflict = engine_bind_conflict(game)
            if conflict:
                raise PortBindConflict(f"engine log {conflict}: Address already in use")
            clock.lap("menu_to_playing")
        if bridge.get_current_level() != 1:
            raise ValueError("episode did not load its single assigned level")
        deadline = time.monotonic() + limits["history_ready_seconds"]
        while not list(aligned.glob("decision-history-*/ready.json")):
            if time.monotonic() >= deadline:
                raise TimeoutError("native decision barrier readiness deadline; no retry")
            time.sleep(.1)
        clock.lap("replay_to_decision")
        _, rows = shared.read_history(aligned, step)
        manifest = shared.expose_history(rows, root / "decision-1", scenario, identity + ":shot-1",
                                         record["exposure_role"], policy)
        result.update(observation_manifest=manifest["identity"], predecision_frames=len(rows),
                      predecision_fixed_steps=[row["fixed_step"] for row in rows])
        old.prepare_action(bridge, record["actions"][0])
        legacy = endpoint.get_physics_capture_v1()
        before = endpoint.get_observation_capture()
        time.sleep(limits["inference_hold_seconds"])
        after = endpoint.get_observation_capture()
        for name, frame in (("before", before), ("after", after)):
            write_json(root / f"paused-{name}.json", _plain_json(frame.metadata))
            (root / f"paused-{name}.png").write_bytes(frame.canonical_png)
        legacy_state = _plain_json(legacy.state)
        write_json(root / "decision-physics-v1.json", legacy_state)
        for frame in (before, after):
            if (frame.metadata["fixed_step"] != step
                    or frame.metadata["fixed_time_seconds"] != rows[-1]["fixed_time_seconds"]):
                raise ValueError("decision physics state advanced during readiness or inference hold")
        boxes = character_boxes(legacy_state)
        invariance = {"character_boxes": boxes,
                      "before": render_invariance(rows[-1]["canonical_png"], before.canonical_png, boxes),
                      "after": render_invariance(rows[-1]["canonical_png"], after.canonical_png, boxes)}
        paired = seal_paired_inputs(root / "paired-inputs", record, step, legacy, rows, invariance)
        result.update(paired_input_manifest=paired["identity"], paired_input_views=list(paired["views"]),
                      endpoint_render_invariance=invariance)
        clock.lap("decision_seal_and_hold")
        decision = policy.choose()
        segment, trace = capture_segment(bridge, aligned, root / "shot-1", record, scenario,
                                         identity + ":shot-1", decision["action"], limits, log)
        result["segments"].append(segment)
        clock.lap("shot_capture")
        initial_metadata = files.read(Path(segment["native_root"]) / "frame_000001.json")
        initial_png = (Path(segment["native_root"]) / "frame_000001.png").read_bytes()
        if (segment["summary"]["first_fixed_step"] != step
                or initial_metadata["fixed_time_seconds"] != rows[-1]["fixed_time_seconds"]):
            raise ValueError("decision physics state advanced before the chosen shot")
        result["shot_initial_render_invariance"] = render_invariance(
            rows[-1]["canonical_png"], initial_png, boxes)
        initial = next(trace.observed_samples())
        if not set(record["generated_slots"]).issubset(segment["initial_entity_ids"]):
            raise ValueError("initial shot omitted an authored object")
        write_json(root / "initial-physics.json", initial)
        launch = [event for chunk in trace.trace.chunks() for event in chunk["events"]
                  if event["event_type"] == "bird_launched"]
        if len(launch) != 1:
            raise ValueError("shot did not contain exactly one native launch")
        clock.lap("shot_validate")
        result.update(native_launch_offset=launch[0]["fixed_step"] - step,
                      requested_release_time_ms=record["actions"][0]["release_time_ms"])
        shared.consume_shared_segment(root / "shot-1/observation-trace", decision["action"],
                                      rows[-1]["fixed_time_seconds"], policy)
        result["policy"] = policy.evidence()
        result["stop_kind"] = stop_kind(segment)
        clock.lap("policy_replay")
    except Exception as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        clock.lap("failed_stage")
    finally:
        actions = [("connection.disconnect", connection.disconnect)
                   for connection in (bridge, endpoint) if connection is not None]
        actions.append(("stop_started_engine", lambda: old.capture.stop_started_engine(engine)))
        if display_process is not None:
            actions.append(("display.terminate", lambda: old.capture.terminate(display_process)))

        def restore_environment():
            for key, value in environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        actions.append(("environment.restore", restore_environment))
        cleanup_failures = cleanup_actions(actions)
        record_cleanup_failures(result, cleanup_failures)
        clock.lap("cleanup")
    result["failure_class"] = failure_class(result["failure"])
    result.update(complete=result["failure"] is None, wall_seconds=time.monotonic() - clock.started,
                  stage_seconds=clock.stages)
    persist_after_cleanup(cleanup_failures,
                          lambda: write_json(output / "results" / f"{identity}.json", result))
    log(f"{identity} complete={result['complete']} failure={result['failure']} "
        f"wall={result['wall_seconds']:.0f}s")
    return result


def stop_kind(segment):
    """Native terminal kind of one shot window (four-kind #77 label contract)."""
    from scripts.issue_76_native_outcomes import terminal_evidence
    if segment["summary"].get("censored") is True:
        return "right_censored"
    terminal = terminal_evidence(segment)
    return TERMINAL_KINDS.get((terminal or {}).get("reason"), "unresolved")


def branch_outcome(segment):
    """Engine-truth outcome over a retained shot: the #85 channel + terminal kind.

    Computed off the capture critical path (publish time); reads only the
    retained native root.
    """
    from scripts.run_engine_outcome_reactive_diagnostic import channel_scan
    channel = channel_scan(segment["native_root"])
    kind = stop_kind(segment)
    return {"stop_kind": kind, "pig_removed": bool(channel["pig_removed"]),
            "native_clear": kind == "native_clear",
            "channel": {key: channel[key] for key in ("pig_removed", "pig_lifecycle_destroyed",
                                                      "bird_launches", "consistency")}}


# ------------------------------------------------------------- supervisor

def _worker_entry(output, record, limits, slot, sequence):
    capture_branch(output, record, limits, slot, sequence)


def _attempt_bytes(root, seen):
    total = 0
    for path in Path(root).rglob("*"):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        if not path.is_file() or path.is_symlink():
            continue
        key = (metadata.st_dev, metadata.st_ino)
        if key not in seen:
            seen.add(key)
            total += metadata.st_size
    return total


def branch_entry(output, identity):
    output = Path(output)
    result_path = output / "results" / f"{identity}.json"
    receipt_path = output / "receipts" / f"{identity}.json"
    if not (result_path.is_file() and receipt_path.is_file()):
        return {"status": "unattempted"}
    result, receipt = read_json(result_path), read_json(receipt_path)
    failure = result.get("failure") or receipt.get("stop")
    complete = (result.get("complete") is True and failure is None
                and receipt.get("worker_exitcode") == 0)
    if not complete and failure is None:
        failure = f"worker_exitcode={receipt.get('worker_exitcode')}"
    return {"status": "complete" if complete else "failed", "failure": failure,
            "failure_class": failure_class(failure), "wall_seconds": receipt.get("wall_seconds"),
            "peak_cpu_rss_mib": receipt.get("peak_cpu_rss_mib"),
            "stop_kind": result.get("stop_kind") if complete else None}


def run_campaign(output, plan_identity, records, limits, label, is_admitted=lambda identity: True):
    """Dispatch every scheduled branch once to isolated worker slots.

    ``records`` is the ordered list of dispatch records. Retained results are
    never repeated; an attempt directory without a result is sealed as a typed
    ``supervisor_interrupted`` failure (no silent retry).
    """
    from scripts.process_lifecycle import start_isolated_worker
    from scripts.run_issue_76_compatibility import process_rss, terminate_worker
    output = Path(output).resolve()
    limits = {**DEFAULT_LIMITS, **limits}
    for sub in ("results", "receipts", "markers", "attempts"):
        (output / sub).mkdir(parents=True, exist_ok=True)
    pending = []
    for record in records:
        identity = record["identity"]
        entry = branch_entry(output, identity)
        if entry["status"] != "unattempted":
            continue
        if (output / "attempts" / identity).exists() or (output / "markers" / f"{identity}.json").exists():
            write_json(output / "results" / f"{identity}.json",
                       {"schema": RESULT_SCHEMA, "member_identity": identity, "complete": False,
                        "failure": "supervisor_interrupted: retained attempt without result",
                        "failure_class": "supervisor_interrupted", "segments": []})
            write_json(output / "receipts" / f"{identity}.json",
                       {"member_identity": identity, "plan_identity": plan_identity,
                        "stop": "supervisor_interrupted", "worker_exitcode": None,
                        "wall_seconds": 0.0, "peak_cpu_rss_mib": 0.0, "technical_retries": 0})
            continue
        pending.append(record)
    context = multiprocessing.get_context("spawn")
    slots = list(range(limits["workers"]))
    sequences = {slot: 0 for slot in slots}
    live = []
    seen_inodes = set()
    _attempt_bytes(output / "player", seen_inodes)
    artifact_bytes = 0
    started = time.monotonic()
    done_walls = []
    total = len(pending)
    finished_count = 0
    interrupted = False
    previous = signal.getsignal(signal.SIGTERM)

    def on_sigterm(signum, frame):
        raise KeyboardInterrupt("SIGTERM")
    signal.signal(signal.SIGTERM, on_sigterm)
    try:
        while pending or live:
            while pending and len(live) < len(slots):
                if shutil.disk_usage(output).free < limits["minimum_free_bytes"]:
                    raise RuntimeError("minimum free storage reached")
                if artifact_bytes > limits["artifact_bytes"]:
                    raise RuntimeError("artifact byte cap reached")
                record = pending.pop(0)
                used = {worker["slot"] for worker in live}
                slot = next(slot for slot in slots if slot not in used)
                sequence = sequences[slot]
                sequences[slot] += 1
                write_json(output / "markers" / f"{record['identity']}.json",
                           {"plan_identity": plan_identity, "member_identity": record["identity"],
                            "slot": slot, "sequence": sequence, "dispatched_at_utc": utc_now(),
                            "live_at_dispatch": len(live) + 1})
                process = start_isolated_worker(context, _worker_entry,
                                                (str(output), record, limits, slot, sequence))
                live.append({"process": process, "record": record, "slot": slot,
                             "started": time.monotonic(), "peak": 0.0, "stop": None})
            time.sleep(1)
            now = time.monotonic()
            for worker in live:
                if worker["process"].is_alive():
                    rss = process_rss(worker["process"].pid)
                    worker["peak"] = max(worker["peak"], rss)
                    if now - worker["started"] >= limits["attempt_seconds"]:
                        worker["stop"] = "attempt_wall_limit"
                    elif worker["peak"] > limits["worker_cpu_rss_mib"]:
                        worker["stop"] = "worker_memory_limit"
            for worker in [w for w in live if w["stop"] or not w["process"].is_alive()]:
                terminate_worker(worker["process"])
                live.remove(worker)
                identity = worker["record"]["identity"]
                wall = time.monotonic() - worker["started"]
                result_path = output / "results" / f"{identity}.json"
                if not result_path.is_file():
                    failure = worker["stop"] or f"worker_exitcode={worker['process'].exitcode}: result_missing"
                    write_json(result_path, {"schema": RESULT_SCHEMA, "member_identity": identity,
                                             "complete": False, "failure": failure,
                                             "failure_class": failure_class(failure), "segments": []})
                artifact_bytes += _attempt_bytes(output / "attempts" / identity, seen_inodes)
                write_json(output / "receipts" / f"{identity}.json",
                           {"member_identity": identity, "plan_identity": plan_identity,
                            "stop": worker["stop"], "worker_exitcode": worker["process"].exitcode,
                            "wall_seconds": wall, "peak_cpu_rss_mib": worker["peak"],
                            "technical_retries": 0, "slot": worker["slot"]})
                entry = branch_entry(output, identity)
                finished_count += 1
                done_walls.append(wall)
                elapsed = time.monotonic() - started
                rate = elapsed / finished_count
                eta = rate * (total - finished_count)
                print(f"[{label}] {finished_count}/{total} {identity} {entry['status']} "
                      f"{entry.get('failure_class') or ''} wall={wall:.0f}s "
                      f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m", flush=True)
    except (KeyboardInterrupt, Exception) as error:
        interrupted = True
        print(f"[{label}] supervisor interrupted: {type(error).__name__}: {error}", flush=True)
        for worker in live:
            terminate_worker(worker["process"])
            identity = worker["record"]["identity"]
            result_path = output / "results" / f"{identity}.json"
            if not result_path.is_file():
                write_json(result_path, {"schema": RESULT_SCHEMA, "member_identity": identity,
                                         "complete": False, "failure": "supervisor_interrupted",
                                         "failure_class": "supervisor_interrupted", "segments": []})
            write_json(output / "receipts" / f"{identity}.json",
                       {"member_identity": identity, "plan_identity": plan_identity,
                        "stop": "supervisor_interrupted", "worker_exitcode": worker["process"].exitcode,
                        "wall_seconds": time.monotonic() - worker["started"],
                        "peak_cpu_rss_mib": worker["peak"], "technical_retries": 0,
                        "slot": worker["slot"]})
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        write_json(output / "run-log" / f"run-{int(time.time())}.json",
                   {"plan_identity": plan_identity, "interrupted": interrupted,
                    "wall_seconds": time.monotonic() - started, "attempts": finished_count,
                    "workers": len(slots), "artifact_bytes_added": artifact_bytes,
                    "finished_at_utc": utc_now()})
    return time.monotonic() - started
