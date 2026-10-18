"""Issue #120: alternative endpoint-cost verdicts from retained native traces (frozen at the #120 freeze).

Light module: process-pool workers import only this file plus json/gzip/math (the #102/#110/#113 lesson:
worker entry points stay light). The runner is ``scripts/run_cost_alignment.py``.

``branch_quantities(native_root)`` reads one ``canonical_native_trace_v1`` trace (a #104 campaign attempt's
``aligned/<capture>/``) into a small, definition-free record:

* the engine events of every chunk (``bird_launched``, ``collision``, ``entity_destroyed``, ...). Each chunk
  JSON ends with its ``events`` array, so only that tail is parsed; the full parse of the chunks that are
  needed anyway is cross-checked in the #120 smoke;
* the frame-0 sample (the decision frame, native step ``first_fixed_step``): the dynamic blocks and pig
  and their positions;
* each block's and the pig's last present position: at the step before its ``entity_destroyed`` event,
  or at the trace's last step if it is never destroyed;
* the present pigs and blocks at the trace's last native step (the engine endpoint);
* the first ``collision`` event between the launched bird and a block or a pig.

``verdicts(quantities, pig_removed)`` turns a record into the four frozen verdicts. ``True`` is the
decision goal of each cost (the outcome a planner wants; lower model cost = predicted goal):

* (a) current: the pig is removed (the #104 campaign verdict, #85 engine channel), passed in;
* (b) structure: at least ``BLOCKS_DESTROYED_MIN`` block entity is destroyed during the shot (the
  surviving block count falls below the initial count);
* (c) motion magnitude: the summed displacement of the dynamic blocks and the pig, frame 0 to their last
  present step, is at least ``DISPLACEMENT_MIN`` world units;
* (d) early contact: the launched bird has an engine ``collision`` event with a block or the pig at or
  before frame ``EARLY_FRAMES`` (native step ``first_fixed_step + 50 * EARLY_FRAMES``).
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

SCHEMA_QUANTITIES = "issue_120_branch_quantities_v1"
FRAME_STEPS = 50                # native steps per observation frame (#104 window, observation_stride)
COSTS = ("a", "b", "c", "d")
COST_NAMES = {"a": "current (pig removal)", "b": "structure (block destroyed)",
              "c": "motion magnitude (structure displacement)", "d": "early contact (bird hits structure)"}
BLOCKS_DESTROYED_MIN = 1
DISPLACEMENT_MIN = 0.5          # world units, summed over the dynamic blocks and the pig
EARLY_FRAMES = 150              # frames after the decision frame (frame 0); a multiple of every Delta
STRUCTURE_KINDS = ("block", "pig")
PIG_CHANNEL_EVENTS = ("pig_removed", "entity_death", "entity_destroyed")  # #85 engine.pig_channel

DEFINITIONS = {
    "a": ("engine pig removal of the branch (#104 campaign branches.json pig_removed, #85 engine channel); model "
          "cost = the #111 tie-free primary at frame 225: pig presence - 0.1 x pig-slot centre displacement"),
    "b": (f"at least {BLOCKS_DESTROYED_MIN} block entity destroyed (engine entity_destroyed) during the shot, i.e. "
          "fewer blocks survive than the level starts with; model cost at frame 225 = sum over the block slots of "
          "presence - 0.1 x sum over the block slots of centre displacement (the tie-free form on the blocks)"),
    "c": (f"summed displacement of the dynamic blocks and the pig from frame 0 to their last present native step >= "
          f"{DISPLACEMENT_MIN} world units; model cost at frame 225 = minus the summed block- and pig-slot centre "
          "displacement"),
    "d": (f"engine collision event between the launched bird and a block or the pig at or before frame "
          f"{EARLY_FRAMES} (native step first_fixed_step + {FRAME_STEPS} x {EARLY_FRAMES}); model cost = minus the "
          f"summed block- and pig-slot centre displacement at the rollout's last visited frame <= {EARLY_FRAMES}"),
}


def kind(entity_id):
    """Entity kind of a runtime id (``runtime:block:0001`` -> ``block``; #113 ``_kind``)."""
    parts = entity_id.split(":")
    return parts[1] if parts[0] == "runtime" and len(parts) > 1 else parts[0]


def chunk_events(raw):
    """The ``events`` array of one decompressed chunk (the last key of the chunk object)."""
    start = raw.rfind(b'"events":')
    end = raw.rfind(b"}")
    if start < 0 or end < start:
        raise ValueError("chunk has no trailing events array")
    return json.loads(raw[start + len(b'"events":'):end])


def _present(entity):
    return bool(entity.get("body_present")) and bool(entity.get("body"))


def _structure(entities):
    """Dynamic blocks and pigs present in one sample: id -> position."""
    return {e["entity_id"]: tuple(e["body"]["position"]) for e in entities
            if kind(e["entity_id"]) in STRUCTURE_KINDS and _present(e) and e["body"].get("body_type") == "dynamic"}


def branch_quantities(native_root):
    """Definition-free quantities of one canonical native trace (see the module docstring)."""
    root = Path(native_root)
    with open(root / "native-manifest.json") as stream:
        manifest = json.load(stream)
    if not manifest.get("complete_every_native_step", False):
        raise ValueError(f"{root}: trace does not hold every native step")
    chunks = manifest["chunks"]
    raws = []
    events = []
    for descriptor in chunks:
        with gzip.open(root / descriptor["path"], "rb") as stream:
            raw = stream.read()
        raws.append(raw)
        events.extend(chunk_events(raw))
    parsed = {}

    def chunk(index):
        if index not in parsed:
            parsed[index] = json.loads(raws[index])
        return parsed[index]

    def chunk_of(step):
        return next(k for k, c in enumerate(chunks) if c["first_fixed_step"] <= step <= c["last_fixed_step"])

    first_step, last_step = manifest["first_fixed_step"], manifest["last_fixed_step"]
    first = chunk(0)["fixed_step_samples"][0]
    if first["fixed_step"] != first_step:
        raise ValueError(f"{root}: the first sample is not the first fixed step")
    initial = _structure(first["entities"])
    launch = next((e for e in events if e["event_type"] == "bird_launched"), None)
    bird = None if launch is None else next((p for p in launch["participants"] if kind(p) == "bird"), None)
    destroyed = {}
    for e in events:
        if e["event_type"] == "entity_destroyed":
            for p in e["participants"]:
                destroyed.setdefault(p, e["fixed_step"])
    last = {}
    for entity_id in sorted(initial):
        target = destroyed[entity_id] - 1 if entity_id in destroyed else last_step
        index = chunk_of(max(target, first_step))
        position = None
        while position is None and index >= 0:
            for sample in reversed(chunk(index)["fixed_step_samples"]):
                if sample["fixed_step"] > target:
                    continue
                found = next((e for e in sample["entities"] if e["entity_id"] == entity_id), None)
                if found is not None and _present(found):
                    position = tuple(found["body"]["position"])
                    break
            index -= 1
        if position is None:
            raise ValueError(f"{root}: {entity_id} has no present sample before step {target}")
        last[entity_id] = position
    final = chunk(len(chunks) - 1)["fixed_step_samples"][-1]
    if final["fixed_step"] != last_step:
        raise ValueError(f"{root}: the last sample is not the last fixed step")
    present_end = {k: sum(1 for e in final["entities"] if kind(e["entity_id"]) == k and _present(e))
                   for k in STRUCTURE_KINDS}
    contact = None
    if bird is not None:
        contact = next((e["fixed_step"] for e in events if e["event_type"] == "collision" and bird in e["participants"]
                        and any(kind(p) in STRUCTURE_KINDS for p in e["participants"] if p != bird)), None)
    pig_events = sorted({e["event_type"] for e in events if e["event_type"] in PIG_CHANNEL_EVENTS
                         and any(kind(p) == "pig" for p in e["participants"])})
    displacement = {k: math.dist(last[k], initial[k]) for k in sorted(initial)}
    return {
        "schema": SCHEMA_QUANTITIES, "capture_id": manifest.get("capture_id"), "status": manifest["status"],
        "first_step": first_step, "last_step": last_step, "frames": len(manifest["frame_records"]),
        "launch_step": None if launch is None else launch["fixed_step"], "launched_bird": bird,
        "initial": {k: sum(1 for e in initial if kind(e) == k) for k in STRUCTURE_KINDS},
        "destroyed": {k: sum(1 for e in destroyed if kind(e) == k and e in initial) for k in STRUCTURE_KINDS},
        "present_end": present_end,
        "structure_displacement": float(sum(displacement.values())),
        "pig_displacement": float(sum(v for k, v in displacement.items() if kind(k) == "pig")),
        "first_structure_contact_step": contact,
        "first_structure_contact_frame": None if contact is None else (contact - first_step) / FRAME_STEPS,
        "pig_channel_events": pig_events,
        "collision_events": sum(1 for e in events if e["event_type"] == "collision"),
    }


def verdicts(quantities, pig_removed):
    """The four frozen verdicts (True = the cost's decision goal) of one branch."""
    contact = quantities["first_structure_contact_step"]
    return {
        "a": bool(pig_removed),
        "b": quantities["destroyed"]["block"] >= BLOCKS_DESTROYED_MIN,
        "c": quantities["structure_displacement"] >= DISPLACEMENT_MIN,
        "d": contact is not None and contact <= quantities["first_step"] + FRAME_STEPS * EARLY_FRAMES,
    }


def pig_removed_from_events(quantities):
    """The #85 engine pig channel re-read from the events (pig_removed / entity_death / entity_destroyed)."""
    return bool(quantities["pig_channel_events"])


def native_root(attempt):
    """The single ``aligned/<capture>/`` native trace root of a #104 campaign attempt directory."""
    roots = sorted(Path(attempt, "aligned").glob("*/native-manifest.json"))
    if len(roots) != 1:
        raise ValueError(f"{attempt}: expected one native trace, found {len(roots)}")
    return roots[0].parent


def settle_at_end(summary):
    """The #113 settle flag (settle_1, frozen debounce) at the trace's last frame."""
    from world_model.data import observable_macro as om
    return bool(om.shot_labels(summary)["values"][-1, 0, 0])


def record_worker(job):
    """Process-pool entry point: (key, attempt, #113 summary path) -> (key, record, seconds). The record holds the
    trace quantities, the settle flag at the endpoint and the summary's hash and frame count."""
    import hashlib
    import time
    from world_model.data import observable_macro as om
    key, attempt, summary_path = job
    began = time.monotonic()
    quantities = branch_quantities(native_root(attempt))
    raw = Path(summary_path).read_bytes()
    summary = om.load_summary(summary_path)
    record = {"quantities": quantities, "settle_end": settle_at_end(summary),
              "summary_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
              "summary_frames": len(summary["frame_steps"])}
    return key, record, time.monotonic() - began
