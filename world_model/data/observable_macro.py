"""Issue #113 item 1: observable macro predicates, declared per request horizon Delta (frozen).

Light module: process-pool workers import only this file plus json/gzip/math/numpy (the #102/#110
lesson: worker entry points stay light). #115 item 4, #120 and the #113 temporal head import it.

Two layers:

1. ``trace_summary(native_root)`` -- one pass over every native step of a canonical native trace
   (schema ``canonical_native_trace_v1``) into a small, definition-free physics summary per observed
   frame (JSON-serialisable): whether any dynamic body moved inside the frame interval, the
   debounced non-bird support set, the launch step and the launched bird's first contact.
2. ``shot_labels(summary)`` -- the frozen predicates of every frame f and every Delta in ``DELTAS``
   from a summary; ``labels_from_trace(native_root)`` does both.

Frames. Frame f is the observation frame at native step s_f (``frame_records[f].fixed_step``;
s_f = s_0 + 50 f except a terminal frame). Its interval is (s_{f-1}, s_f]; frame 0's interval is
{s_0}. Labels are indexed by the frame a request lands on: the label of a Delta request issued
at frame t is read at f = t + Delta.

Predicates (frozen at the #113 item-1 freeze; ``DEFINITIONS`` holds the text):

* rest(step): every present, simulated, dynamic body (birds included) has |v| <= REST_SPEED and
  |omega| <= REST_ANGULAR_DEG (the engine's stable_entered thresholds).
  moving[f] = some native step in frame f's interval is not at rest.
* settle_Delta(f): the shot's dynamics are over and the scene has been at rest over the whole
  request window and at least the debounce: with D = max(DEBOUNCE_FRAMES, Delta), the launched
  bird was removed (engine entity_destroyed; the engine removes a bird that has come to rest, or
  that leaves the world) at or before s_{f-D}, AND moving[f - D + 1 .. f] are all false.
  Available iff f >= D. (Rest before the removal is not settled: a resting bird that props a
  block is removed after a delay and the block then tips, the relapse found by the selection pass.)
* structure-change_Delta(f): the debounced support set at f differs from the one at f - Delta.
  The debounced support set S(f) is the set of directed (supporter, supported) entity pairs with
  no bird on either side that hold on more than half of the native steps of f's interval (frame 0:
  its one step), so a single-step spike neither creates nor removes a pair. Endpoint comparison:
  a change that reverts inside the window is not a change. Available iff f >= Delta.

Relation to the #110 per-Delta rule (``novelty_labels_v2.window_any``): #110 ORs single-frame
labels over (f - Delta, f]. #113 does not adopt it. Structure change compares debounced endpoints
(net change across the window, no single-step spikes); settle requires rest over the whole window
(an AND, not an OR). #110's published labels are unchanged and read-only.

Phases (#100-style, from the native trace): pre_contact = frames with s_f before the launched
bird's first contact with any entity that is neither a bird nor the slingshot (every frame if it
never touches one); settled = from the first frame >= that onset after which settle at the
smallest Delta holds to the shot's end; cascade = the frames between. (#100 read contact from the
18-slot relations at frame steps; here every native step of the launched bird counts.)
"""
from __future__ import annotations

from collections import Counter
import gzip
import json
import math
from pathlib import Path

import numpy as np

SCHEMA_SUMMARY = "issue_113_trace_summary_v2"
SCHEMA_LABELS = "issue_113_observable_macro_labels_v1"
IDENTITY = "issue-113-macro-predicates-v1"
DELTAS = (1, 5, 15)
PREDICATES = ("settle", "structure-change")
PHASES = ("pre_contact", "cascade", "settled")
REST_SPEED = 0.01          # u/s, the engine stable_entered threshold (|v|^2 <= 1e-4)
REST_ANGULAR_DEG = 0.01    # deg/s, the engine stable_entered threshold
DEBOUNCE_CANDIDATES = (1, 2, 3, 5, 10, 15, 20, 25, 30, 35)
RELAPSE_HORIZON = max(DELTAS)
RELAPSE_MAX = 0.05
DEBOUNCE_FRAMES = 1        # frozen by the item-1 selection rule (``select_debounce``; selection.json): no relapse at any k

DEFINITIONS = {
    "rest": (f"every present, simulated, dynamic body (birds included) has |v| <= {REST_SPEED} u/s and |omega| <= "
             f"{REST_ANGULAR_DEG} deg/s (the engine stable_entered thresholds); moving[f] = some native step of "
             "frame f's interval (s_(f-1), s_f] is not at rest"),
    "settle": ("settle_Delta(f) = the launched bird was removed (engine entity_destroyed) at or before s_(f-D) AND "
               "moving[f-D+1..f] all false, with D = max(debounce, Delta); available iff f >= D"),
    "structure-change": ("structure-change_Delta(f) = S(f) != S(f-Delta), S(f) = the directed non-bird "
                         "(supporter, supported) entity pairs present on more than half of the native steps of "
                         "frame f's interval (frame 0: its single step); available iff f >= Delta"),
    "label_index": "labels are indexed by the landing frame f = t + Delta of a Delta request issued at frame t",
    "phases": ("pre_contact = frames before the launched bird's first native-step contact with a non-bird, "
               "non-slingshot entity (all frames if none); settled = from the first frame >= onset after which "
               "settle at the smallest Delta holds to the end; cascade = between"),
    "relation_to_110": ("#110 ORs single-frame labels over (f-Delta, f]; #113 does not adopt it: structure change "
                        "compares debounced endpoints, settle requires rest over the whole window"),
    "debounce_rule": (f"smallest debounce k in {list(DEBOUNCE_CANDIDATES)} frames whose relapse rate (share of "
                      f"settle_k frames f with f + {RELAPSE_HORIZON} <= last frame that have a moving frame in "
                      f"(f, f + {RELAPSE_HORIZON}]) is <= {RELAPSE_MAX}, pooled over the #104 fit stream and the "
                      "retained N1, #93 and #94 captures"),
}


# ------------------------------------------------------------------ summary

def _bird(entity_id):
    return ":bird:" in entity_id


def _kind(entity_id):
    parts = entity_id.split(":")
    return parts[1] if parts[0] == "runtime" and len(parts) > 1 else parts[0]


def _moving(entities):
    """(any dynamic body moving, a non-bird dynamic body moving) at one native sample."""
    any_moving = nonbird = False
    for entity in entities:
        body = entity.get("body")
        if not entity.get("body_present") or not body or body.get("body_type") != "dynamic" or not body.get("simulated"):
            continue
        if math.hypot(*body["velocity"]) > REST_SPEED or abs(body["angular_velocity_degrees_per_second"]) > REST_ANGULAR_DEG:
            any_moving = True
            if not _bird(entity.get("entity_id", "")):
                return True, True
    return any_moving, nonbird


def _pairs(supports):
    return {(s["supporter_entity_id"], s["supported_entity_id"]) for s in supports
            if not _bird(s["supporter_entity_id"]) and not _bird(s["supported_entity_id"])}


def trace_summary(native_root):
    """Definition-free physics summary of one canonical native trace (every native step read)."""
    root = Path(native_root)
    with open(root / "native-manifest.json") as stream:
        manifest = json.load(stream)
    if not manifest.get("complete_every_native_step", False):
        raise ValueError(f"{root}: trace does not hold every native step")
    frame_steps = [record["fixed_step"] for record in manifest["frame_records"]]
    frame_of = {}
    for f, step in enumerate(frame_steps):
        for s in range(frame_steps[f - 1] + 1 if f else step, step + 1):
            frame_of[s] = f
    count = len(frame_steps)
    moving = [False] * count
    moving_nonbird = [False] * count
    steps_in = [0] * count
    pair_counts = [Counter() for _ in range(count)]
    events = []
    launch = launched_bird = onset = None
    pending = []
    previous = None
    for descriptor in manifest["chunks"]:
        with gzip.open(root / descriptor["path"], "rb") as stream:
            chunk = json.loads(stream.read())
        for event in chunk["events"]:
            events.append((event["fixed_step"], event["event_type"], list(event["participants"])))
        launches = [(e["fixed_step"], e["participants"]) for e in chunk["events"] if e["event_type"] == "bird_launched"]
        pending.extend(launches)
        for sample in chunk["fixed_step_samples"]:
            step = sample["fixed_step"]
            if previous is not None and step != previous + 1:
                raise ValueError(f"{root}: native steps are not contiguous at {step}")
            previous = step
            if launch is None and pending and pending[0][0] <= step:
                launch = pending[0][0]
                launched_bird = next((p for p in pending[0][1] if _bird(p)), None)
            if onset is None and launched_bird is not None:
                for contact in sample["contacts"]:
                    a, b = contact["entity_a_id"], contact["entity_b_id"]
                    other = b if a == launched_bird else a if b == launched_bird else None
                    if other is not None and not _bird(other) and _kind(other) != "slingshot":
                        onset = step
                        break
            f = frame_of.get(step)
            if f is None:
                continue
            steps_in[f] += 1
            if not moving_nonbird[f]:
                any_moving, nonbird = _moving(sample["entities"])
                moving[f] = moving[f] or any_moving
                moving_nonbird[f] = nonbird
            pair_counts[f].update(_pairs(sample["supports"]))
    if any(n == 0 for n in steps_in):
        raise ValueError(f"{root}: a frame interval has no native sample")
    table, index, support = {}, [], []
    for f in range(count):
        pairs = tuple(sorted(f"{a}>{b}" for (a, b), n in pair_counts[f].items() if 2 * n > steps_in[f]))
        if pairs not in table:
            table[pairs] = len(support)
            support.append(list(pairs))
        index.append(table[pairs])
    terminal = (manifest.get("terminal_evidence") or {}).get("reason") or manifest.get("failure")
    destroyed = [[s, p] for s, t, p in events if t == "entity_destroyed"]
    removed = next((s for s, p in destroyed if launched_bird in p and s >= launch), None) if launch is not None else None
    return {"schema": SCHEMA_SUMMARY, "capture_id": manifest.get("capture_id"), "status": manifest["status"],
            "terminal": terminal, "first_step": manifest["first_fixed_step"], "frame_steps": frame_steps,
            "moving": [int(m) for m in moving], "moving_nonbird": [int(m) for m in moving_nonbird],
            "support_index": index, "support_sets": support,
            "launch_step": launch, "launched_bird": launched_bird, "contact_onset_step": onset,
            "bird_removed_step": removed, "destroyed": destroyed,
            "stability_events": [[s, t] for s, t, _ in events if t in ("stable_entered", "stable_exited")]}


def write_summary(path, summary):
    """Atomic gzip-JSON write (mtime 0, so equal summaries are equal bytes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
        stream.write(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode())
    temporary.replace(path)


def load_summary(path):
    with gzip.open(path, "rb") as stream:
        summary = json.loads(stream.read())
    if summary.get("schema") != SCHEMA_SUMMARY:
        raise ValueError(f"{path}: not an issue-113 trace summary")
    return summary


def summary_worker(job):
    """Process-pool entry point: (native_root, target) -> writes the summary; returns (target, frames, seconds)."""
    import time
    native_root, target = job
    began = time.monotonic()
    summary = trace_summary(native_root)
    write_summary(target, summary)
    return str(target), len(summary["frame_steps"]), time.monotonic() - began


# ------------------------------------------------------------------- labels

def rest_windows(moving, frame_steps, gate_step, span):
    """bool [F]: frames f >= span whose last ``span`` intervals are at rest and whose window starts at or after
    ``gate_step`` (the launched bird's removal); all false without a gate."""
    moving = np.asarray(moving, dtype=bool)
    count = len(moving)
    out = np.zeros(count, dtype=bool)
    if gate_step is None or count <= span:
        return out
    run = 0
    for f in range(count):
        run = 0 if moving[f] else run + 1
        if f >= span and run >= span and frame_steps[f - span] >= gate_step:
            out[f] = True
    return out


def settle_labels(summary, k, deltas=DELTAS):
    """(values, available) bool [F, len(deltas)] of settle_Delta with debounce ``k``."""
    count = len(summary["moving"])
    values = np.zeros((count, len(deltas)), dtype=bool)
    available = np.zeros((count, len(deltas)), dtype=bool)
    for j, delta in enumerate(deltas):
        span = max(k, delta)
        values[:, j] = rest_windows(summary["moving"], summary["frame_steps"], summary["bird_removed_step"], span)
        available[span:, j] = True
    return values, available


def change_labels(summary, deltas=DELTAS):
    """(values, available) bool [F, len(deltas)] of structure-change_Delta."""
    index = np.asarray(summary["support_index"])
    count = len(index)
    values = np.zeros((count, len(deltas)), dtype=bool)
    available = np.zeros((count, len(deltas)), dtype=bool)
    for j, delta in enumerate(deltas):
        if count > delta:
            values[delta:, j] = index[delta:] != index[:-delta]
            available[delta:, j] = True
    return values, available


def phases(summary, settled):
    """int [F] phase index (PHASES) from the contact onset and a settle column; also (onset, settled) frames."""
    steps = summary["frame_steps"]
    count = len(steps)
    phase = np.zeros(count, dtype=np.int64)
    onset_step = summary["contact_onset_step"]
    if onset_step is None:
        return phase, None, None
    onset = next((f for f, s in enumerate(steps) if s >= onset_step), None)
    if onset is None:
        return phase, None, None
    first = None
    for f in range(count - 1, onset - 1, -1):
        if not settled[f]:
            break
        first = f
    phase[onset:] = 1
    if first is not None:
        phase[first:] = 2
    return phase, onset, first


def debounce():
    if DEBOUNCE_FRAMES is None:
        raise ValueError("the #113 settle debounce is not frozen yet")
    return DEBOUNCE_FRAMES


def shot_labels(summary, k=None, deltas=DELTAS):
    """Every frozen predicate of one shot: values / available [F, len(PREDICATES), len(deltas)], phase [F]."""
    if summary.get("schema") != SCHEMA_SUMMARY:
        raise ValueError("not an issue-113 trace summary")
    k = debounce() if k is None else k
    settle, settle_available = settle_labels(summary, k, deltas)
    change, change_available = change_labels(summary, deltas)
    phase, onset, settled = phases(summary, settle[:, 0])
    return {"schema": SCHEMA_LABELS, "debounce_frames": k, "deltas": list(deltas), "predicates": list(PREDICATES),
            "values": np.stack((settle, change), 1), "available": np.stack((settle_available, change_available), 1),
            "phase": phase, "contact_onset_frame": onset, "settled_onset_frame": settled,
            "frames": len(summary["frame_steps"])}


def labels_from_trace(native_root, k=None, deltas=DELTAS):
    return shot_labels(trace_summary(native_root), k, deltas)


# --------------------------------------------------------- debounce selection

def relapse_counts(summary, k, horizon=RELAPSE_HORIZON):
    """(settle_k frames with f + horizon <= last, of which some frame in (f, f + horizon] moves)."""
    moving = np.asarray(summary["moving"], dtype=bool)
    settle = rest_windows(moving, summary["frame_steps"], summary["bird_removed_step"], k)
    last = len(moving) - 1
    frames = [f for f in np.nonzero(settle)[0].tolist() if f + horizon <= last]
    relapsed = sum(bool(moving[f + 1:f + horizon + 1].any()) for f in frames)
    return len(frames), relapsed


def select_debounce(summaries, candidates=DEBOUNCE_CANDIDATES, maximum=RELAPSE_MAX):
    """The frozen selection rule: the smallest k whose pooled relapse rate is <= ``maximum``."""
    table = []
    for k in candidates:
        frames = relapsed = 0
        for summary in summaries:
            n, r = relapse_counts(summary, k)
            frames, relapsed = frames + n, relapsed + r
        table.append({"k": k, "frames": frames, "relapsed": relapsed,
                      "rate": None if frames == 0 else relapsed / frames})
    chosen = next((row["k"] for row in table if row["rate"] is not None and row["rate"] <= maximum), None)
    return chosen, table
