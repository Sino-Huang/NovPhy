"""Issue #110 contract-v2 shot labels from the #104 campaign captures (outcome-free).

Light module: the spawned label workers import only this file, ``novelty_contract`` and
numpy/torch/PIL (the #102 lesson: worker entry points stay light).

Per shot position p (shot frame p at native step 30000 + 50 p, p = 0..min(last, 285)):

* presence / centers / kinds of the 28 v2 slots. Presence is the #70/#99 rule (lifecycle active
  and body present) except the novelty slot, which is present while its source is active and
  has a renderer (Fan, air turbulence, Magnet, Pink*; never InverseGravity/Storm). Centers are
  the #70 engine projection into the agent view; a bodiless source (air turbulence) is projected
  at its authored position. Kinds are ``KIND_VOCABULARY_V2`` indices: the slot prefix, and for
  the novelty slot ``NOVELTY_VISUAL_KIND`` of its source type.
* micro relations [28, 28, 3]: contact (symmetric) and supports (directed), the cohort-v2
  micro derivation; force-on (source slot -> body slot) from the repaired labeler below.
* macros [3]: steady-state and structure-unstable (the #99 ``shot_labels`` engine-event form)
  and external-force-active (any force-on edge at the frame).
* scene targets: gravity_y / 9.8 and storm-active (first destroyed bird <= step, Storm levels).
* engine features [28, 4] (world x, y, vx, vy) for the identifiability ceiling probe.

Labeler repair (#110 smoke, the fan sign/zone disagreement): the Fan prefab carries the
``Fan`` MonoBehaviour (serialized fanForce 50, windWidth 4, windHeight 1), not ``ABFan`` (whose
script is attached but disabled). ``Fan`` pushes ``transform.right * 50`` on every non-platform
rigidbody that has entered its trigger box (size windWidth * 2 x windHeight = 8 x 1, centred on
the fan). The trace records that trigger box and every body collider, so overlap is computed
from the recorded geometry: a body's collider bounding box, taken in the trigger frame,
intersects the box (``FAN_LAW_REPAIRED``).
"""
from __future__ import annotations

from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import torch

from world_model.data import novelty_contract as contract

SCHEMA_LABELS = "issue_110_v2_shot_labels_v1"
VOCABULARY = contract.VOCABULARY_V2
SLOTS = len(VOCABULARY)
SLOT_INDEX = {name: i for i, name in enumerate(VOCABULARY)}
NOVELTY_SLOT = SLOT_INDEX["novelty:0000"]
PIG_SLOT = SLOT_INDEX["pig:0000"]
KINDS = contract.KIND_VOCABULARY_V2
RELATIONS = ("contact", "supports", "force-on")
MACROS = ("steady-state", "structure-unstable", "external-force-active")
DECISION_FIXED_STEP = 30000
NATIVE_STRIDE = 50
MAX_POSITION = 285
IMAGE_STRIDE = 5
IMAGE_SIZE = (320, 240)  # PIL (width, height), the E99 input
FIXED_DELTA = 0.0004
BIRDS = tuple(f"runtime:bird:{i:04}" for i in range(4))
RENDERED_SOURCES = frozenset(t for t, kind in contract.NOVELTY_VISUAL_KIND.items() if kind is not None)
PACKED = (SLOTS * SLOTS * len(RELATIONS) + 7) // 8

FAN_LAW_REPAIRED = {
    "types": ("Fan",),
    "script": "Fan.cs (attached, enabled); ABFan.cs is attached to the prefab but disabled (m_Enabled: 0)",
    "serialized": {"fanForce": 50.0, "windWidth": 4.0, "windHeight": 1.0},
    "rule": ("force = transform.right * 50 on every non-platform rigidbody inside the Fan's trigger box "
             "(BoxCollider2D size (windWidth * 2, windHeight) = (8, 1), centred on the fan); no speed gate, no "
             "launch condition"),
    "label": ("force-on(novelty:0000, body) at a step iff the body is dynamic, not a platform, and the "
              "bounding box of its non-trigger colliders in the trigger frame intersects the recorded trigger box"),
    "replaces": "contract.FORCE_LAWS['bird_zone_push'] (ABFan: -right * 25 on the launched bird, 2 x 1.5 left zone)",
}
FAN_FORCE = FAN_LAW_REPAIRED["serialized"]["fanForce"]


# ------------------------------------------------------------------ level

def level_source(xml_text):
    """(contract source, slot materials, slingshot side) of a materialized level xml."""
    tree = ET.fromstring(xml_text)
    materials = {node.attrib["scenarioObjectId"]: node.attrib.get("material") or None
                 for node in tree.iter() if "scenarioObjectId" in node.attrib}
    side = contract.slingshot_side(float(tree.find("Slingshot").attrib["x"]))
    return contract.source_from_scenario(tree), materials, side


def novelty_kind(source):
    if source is None:
        return KINDS.index("other")
    kind = contract.NOVELTY_VISUAL_KIND[source.type]
    return KINDS.index(kind if kind is not None else "other")


def slot_kinds(source):
    """Engine kind index per v2 slot (the novelty slot takes its source's visual kind)."""
    out = []
    for name in VOCABULARY:
        prefix = contract.slot_kind(name)
        out.append(novelty_kind(source) if prefix == "novelty" else KINDS.index(prefix))
    return torch.tensor(out, dtype=torch.long)


def expected_kinds():
    """The carrier's expected kind per slot (v1 rule: the slot prefix; otherwise 'other')."""
    return torch.tensor([KINDS.index(p) if p in KINDS[:-1] else len(KINDS) - 1
                         for p in (contract.slot_kind(n) for n in VOCABULARY)], dtype=torch.long)


# ------------------------------------------------------------- trace reader

SAMPLE_START = re.compile(r'\{"fixed_step":(\d+),"complete_raw_non_trigger_contacts"')
DECODER = json.JSONDecoder()


def manifest(native_root):
    with open(Path(native_root) / "native-manifest.json") as stream:
        return json.load(stream)


def chunk_text(native_root, descriptor):
    with gzip.open(Path(native_root) / descriptor["path"], "rb") as stream:
        return stream.read().decode()


def selected_samples(native_root, steps, *, full=False):
    """Native samples at the requested steps plus every event, parsing only those samples
    (``full``: every sample). Returns ({step: sample}, events, manifest)."""
    m = manifest(native_root)
    wanted, samples, events = set(steps), {}, []
    for descriptor in m["chunks"]:
        text = chunk_text(native_root, descriptor)
        if full:
            value = json.loads(text)
            events.extend(value["events"])
            for sample in value["fixed_step_samples"]:
                samples[sample["fixed_step"]] = sample
            continue
        if not (descriptor["first_fixed_step"] <= max(wanted) and descriptor["last_fixed_step"] >= min(wanted)):
            index = text.rfind(',"events":[')
        else:
            for match in SAMPLE_START.finditer(text):
                if int(match.group(1)) in wanted:
                    sample, _ = DECODER.raw_decode(text, match.start())
                    samples[sample["fixed_step"]] = sample
            index = text.rfind(',"events":[')
        if index < 0:
            raise ValueError(f"chunk {descriptor['path']} has no events array")
        chunk_events, _ = DECODER.raw_decode(text, index + len(',"events":'))
        events.extend(chunk_events)
    missing = wanted - set(samples)
    if missing:
        raise ValueError(f"native trace lacks samples at {sorted(missing)[:5]}")
    return samples, events, m


# ---------------------------------------------------------------- geometry

def _collider_bounds(shape, angle):
    """Axis-aligned bounds of a collider shape in a frame rotated by ``angle`` degrees."""
    cos, sin = math.cos(math.radians(-angle)), math.sin(math.radians(-angle))

    def rotate(point):
        return point[0] * cos - point[1] * sin, point[0] * sin + point[1] * cos
    if shape["kind"] == "circle":
        x, y = rotate(shape["center"])
        r = shape["radius"]
        return x - r, x + r, y - r, y + r
    if shape["kind"] == "box":
        cx, cy = shape["center"]
        hx, hy = shape["size"][0] / 2, shape["size"][1] / 2
        a = math.radians(shape.get("angle_degrees", 0.0))
        corners = [(cx + dx * math.cos(a) - dy * math.sin(a), cy + dx * math.sin(a) + dy * math.cos(a))
                   for dx in (-hx, hx) for dy in (-hy, hy)]
        points = [rotate(p) for p in corners]
    else:
        points = [rotate(p) for path in shape["paths"] for p in path]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs), max(xs), min(ys), max(ys)


def fan_trigger(sample):
    """The Fan's recorded trigger box (centre, size, angle) at a sample, or None when inactive."""
    for collider in sample["colliders"]:
        if (collider["entity_id"] == "runtime:novelty:0000" and collider["is_trigger"] and collider["enabled"]
                and collider["shape"]["kind"] == "box"):
            return collider["shape"]
    return None


def fan_edges(sample):
    """Repaired fan law: {body slot: force vector} for the bodies in the Fan's trigger box."""
    box = fan_trigger(sample)
    if box is None:
        return {}
    angle = box.get("angle_degrees", 0.0)
    cos, sin = math.cos(math.radians(-angle)), math.sin(math.radians(-angle))
    cx, cy = box["center"][0] * cos - box["center"][1] * sin, box["center"][0] * sin + box["center"][1] * cos
    hx, hy = box["size"][0] / 2, box["size"][1] / 2
    right = (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
    colliders = defaultdict(list)
    for collider in sample["colliders"]:
        if collider["enabled"] and not collider["is_trigger"]:
            colliders[collider["entity_id"]].append(collider["shape"])
    out = {}
    for entity in sample["entities"]:
        slot = entity["scenario_object_id"]
        body = entity.get("body")
        if (entity["lifecycle"] != "active" or not entity["body_present"] or body is None
                or body["body_type"] != "dynamic" or contract.slot_kind(slot) in ("platform", "novelty")):
            continue
        for shape in colliders.get(entity["entity_id"], ()):
            x0, x1, y0, y1 = _collider_bounds(shape, angle)
            if x1 >= cx - hx and x0 <= cx + hx and y1 >= cy - hy and y0 <= cy + hy:
                out[slot] = (right[0] * FAN_FORCE, right[1] * FAN_FORCE)
                break
    return out


def force_edges(source, sample, materials, launched, storm_active):
    """{body slot: force} of the v2 force-on labeler at one native sample (repaired fan law)."""
    if source is None or contract.LAW_OF_TYPE.get(source.type) in (None, "global_gravity"):
        return {}
    if source.type == "Fan":
        return fan_edges(sample)
    out = {}
    for body in contract.bodies_from_sample(sample, materials, launched, source.slot):
        force = contract.force_on(source, body, storm_active=storm_active)
        if force:
            out[body.slot] = force
    return out


def legacy_fan_edges(source, sample, materials, launched):
    """The frozen pre-repair labeler (ABFan law, contract.force_on) -- kept for the repair record."""
    out = {}
    for body in contract.bodies_from_sample(sample, materials, launched, source.slot):
        force = contract.force_on(source, body)
        if force:
            out[body.slot] = force
    return out


def project(points, metadata):
    """#70 engine projection: world (x, y) -> agent-view normalized pixel, clipped to [0, 1]."""
    transform = metadata["world_to_observation_transform"]
    clip_m = np.asarray(transform["camera_to_clip_matrix"], dtype=np.float64).reshape(4, 4)
    cam_m = np.asarray(transform["world_to_camera_matrix"], dtype=np.float64).reshape(4, 4)
    ndc_m = np.asarray(transform["ndc_to_observation_matrix"], dtype=np.float64).reshape(3, 3)
    viewport = metadata["viewport"]
    out = []
    for x, y in points:
        clip = clip_m @ cam_m @ np.array([x, y, 0.0, 1.0])
        if abs(clip[3]) < 1e-9:
            raise ValueError("singular engine projection")
        ndc = clip[:3] / clip[3]
        pixel = ndc_m @ np.array([ndc[0], ndc[1], 1.0])
        out.append(np.clip(pixel[:2] / [viewport["width_pixels"], viewport["height_pixels"]], 0, 1))
    return out


# ---------------------------------------------------------------- bit packing

def pack(relations):
    """bool [..., 28, 28, 3] -> uint8 [..., PACKED] (little bit order)."""
    flat = relations.reshape(*relations.shape[:-3], -1).to(torch.uint8)
    pad = PACKED * 8 - flat.shape[-1]
    if pad:
        flat = torch.cat((flat, torch.zeros(*flat.shape[:-1], pad, dtype=torch.uint8)), -1)
    flat = flat.reshape(*flat.shape[:-1], PACKED, 8)
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)
    return (flat * weights).sum(-1, dtype=torch.uint8)


def unpack(packed):
    """uint8 [..., PACKED] -> bool [..., 28, 28, 3] (any device)."""
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bits = (packed[..., None] >> shifts) & 1
    bits = bits.reshape(*packed.shape[:-1], PACKED * 8)[..., :SLOTS * SLOTS * len(RELATIONS)]
    return bits.reshape(*packed.shape[:-1], SLOTS, SLOTS, len(RELATIONS)).bool()


def relation_mask(presence, available):
    """The #70 relation mask: both slots present, distinct, and the predicate available.
    presence bool [..., 28], available bool [..., 3] -> bool [..., 28, 28, 3]."""
    pair = presence[..., :, None] & presence[..., None, :]
    pair = pair & ~torch.eye(SLOTS, dtype=torch.bool, device=presence.device)
    return pair[..., None] & available[..., None, None, :]


def window_any(values, delta):
    """The #110 per-Delta rule over a shot's positions: out[t] = any(values[max(0, t - delta + 1) .. t]),
    i.e. the label of a prediction landing at t for a request at Delta (frames (t - Delta, t])."""
    out = values.clone()
    for shift in range(1, delta):
        out[shift:] |= values[:-shift]
    return out


# ------------------------------------------------------------------- labels

def shot_frames(attempt):
    shot_root = Path(attempt) / "shot-1"
    observation_root = shot_root / "observation-trace"
    with open(observation_root / "observation_trace_manifest.json") as stream:
        frames = json.load(stream)["frame_records"]
    if (len(frames) < 2 or frames[0]["fixed_step"] != DECISION_FIXED_STEP
            or any(f["fixed_step"] != DECISION_FIXED_STEP + NATIVE_STRIDE * i for i, f in enumerate(frames[:-1]))):
        raise ValueError(f"observation cadence differs for {attempt}")
    with open(shot_root / "segment.json") as stream:
        segment = json.load(stream)
    return observation_root, frames, segment


def action_of(segment):
    """The v1 action encoding (contract.ACTION_ENCODING; s99.action_of)."""
    action = segment["action"]
    x, y = action["drag_release"]
    return torch.tensor((x / 480., y / 480., action.get("release_time", 1000) / 1000.,
                         action.get("tap_time", 0) / 1000., 1.), dtype=torch.float32)


def image_tensor(path):
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)


def shot_labels(attempt, xml_text, *, images=False):
    """Every v2 label of one complete capture (see the module docstring)."""
    observation_root, frames, segment = shot_frames(attempt)
    end = min(len(frames) - 1, MAX_POSITION)
    steps = [frames[p]["fixed_step"] for p in range(end + 1)]
    native = segment["native_root"]
    if not Path(native).is_dir():  # attempts were moved with the campaign; re-root the native trace
        native = str(Path(attempt) / "aligned" / Path(native).name)
    previous = [s - 1 for s in steps if s - 1 >= DECISION_FIXED_STEP]
    samples, events, m = selected_samples(native, steps + previous)
    if [f["fixed_step"] for f in m["frame_records"]] != [f["fixed_step"] for f in frames]:
        raise ValueError("native physics and observations do not share exact endpoints")
    source, materials, side = level_source(xml_text)
    kinds_of_slot = slot_kinds(source)
    stability = [(e["fixed_step"], e["event_type"] == "stable_entered") for e in events
                 if e["event_type"] in ("stable_entered", "stable_exited")]  # trace order: last event wins
    launches = sorted((e["fixed_step"], p.replace("runtime:", "")) for e in events
                      if e["event_type"] == "bird_launched" for p in e["participants"])
    storm = None
    if source is not None and source.type == "Storm":
        storm = contract.storm_onset_step(events, BIRDS)
    source_center = None if source is None else source.position
    T = end + 1
    presence = torch.zeros(T, SLOTS, dtype=torch.bool)
    centers = torch.zeros(T, SLOTS, 2)
    kinds = torch.full((T, SLOTS), -1, dtype=torch.int8)
    relations = torch.zeros(T, SLOTS, SLOTS, len(RELATIONS), dtype=torch.bool)
    available = torch.zeros(T, len(RELATIONS), dtype=torch.bool)
    macros = torch.zeros(T, len(MACROS), dtype=torch.bool)
    macro_mask = torch.zeros(T, len(MACROS), dtype=torch.bool)
    scene = torch.zeros(T, 2)
    engine = torch.zeros(T, SLOTS, 4)
    forced = torch.zeros(T, SLOTS, dtype=torch.bool)
    outside = set()
    for p, step in enumerate(steps):
        sample = samples[step]
        if sample["complete_raw_non_trigger_contacts"] is not True:
            raise ValueError(f"contact enumeration is incomplete at {step}")
        steady = None
        for s, value in stability:
            if s <= step:
                steady = value
        launched = {slot for s, slot in launches if s <= step}
        active, points, slots = {}, [], []
        for entity in sample["entities"]:
            slot = entity["scenario_object_id"]
            if slot not in SLOT_INDEX:
                if entity["lifecycle"] == "active" and entity["body_present"]:
                    outside.add(slot)
                continue
            i = SLOT_INDEX[slot]
            if slot == "novelty:0000":
                if source is None or source.type not in RENDERED_SOURCES or entity["lifecycle"] != "active":
                    continue
                body = entity.get("body") if entity["body_present"] else None
                position = body["position"] if body else source_center
                velocity = body["velocity"] if body else (0.0, 0.0)
            else:
                if entity["lifecycle"] != "active" or not entity["body_present"]:
                    continue
                position, velocity = entity["body"]["position"], entity["body"]["velocity"]
            presence[p, i] = True
            kinds[p, i] = int(kinds_of_slot[i])
            active[entity["entity_id"]] = i
            points.append(position)
            slots.append(i)
            engine[p, i] = torch.tensor((*position, *velocity), dtype=torch.float32)
        for i, center in zip(slots, project(points, frames[p]["capture_metadata"])):
            centers[p, i] = torch.from_numpy(center).float()
        available[p, 0] = available[p, 1] = available[p, 2] = True
        for contact in sample["contacts"]:
            a, b = active.get(contact["entity_a_id"]), active.get(contact["entity_b_id"])
            if a is not None and b is not None and a != b:
                relations[p, a, b, 0] = relations[p, b, a, 0] = True
        for support in sample["supports"]:
            a, b = active.get(support["supporter_entity_id"]), active.get(support["supported_entity_id"])
            if a is not None and b is not None and a != b:
                relations[p, a, b, 1] = True
        storm_active = storm is not None and storm <= step
        edges = force_edges(source, sample, materials, launched, storm_active)
        for slot in edges:
            forced[p, SLOT_INDEX[slot]] = True
            relations[p, NOVELTY_SLOT, SLOT_INDEX[slot], 2] = True
        before = samples.get(step - 1)
        supports = {(s["supporter_entity_id"], s["supported_entity_id"]) for s in sample["supports"]}
        prior = None if before is None else {(s["supporter_entity_id"], s["supported_entity_id"])
                                             for s in before["supports"]}
        if steady is not None:
            macros[p, 0], macro_mask[p, 0] = bool(steady), True
            if prior is not None:
                macros[p, 1], macro_mask[p, 1] = (steady is False and supports != prior), True
        macros[p, 2], macro_mask[p, 2] = bool(edges), True
        scene[p, 0] = sample["world"]["gravity_vector"][1] / contract.STANDARD_GRAVITY
        scene[p, 1] = float(storm_active)
    out = {"schema": SCHEMA_LABELS, "end": end, "frame_count": len(frames), "steps": torch.tensor(steps),
           "times": torch.tensor([frames[p]["fixed_time_seconds"] for p in range(T)], dtype=torch.float64),
           "action": action_of(segment), "side": side, "source_type": None if source is None else source.type,
           "storm_onset_step": storm, "outside_v2": sorted(outside),
           "presence": presence, "centers": centers, "kinds": kinds, "relations": pack(relations),
           "relation_available": available, "macros": macros, "macro_mask": macro_mask, "scene": scene,
           "engine": engine, "forced": forced}
    if images:
        positions = list(range(0, end + 1, IMAGE_STRIDE))
        out["image_positions"] = torch.tensor(positions)
        out["images"] = torch.stack([image_tensor(observation_root / frames[p]["agent_observation"]["relative_path"])
                                     for p in positions])
    return out


def label_worker(job):
    """Process-pool entry point: (attempt, xml path, target, images) -> writes target, returns its summary."""
    attempt, xml_path, target, images = job
    torch.set_num_threads(1)
    target = Path(target)
    if target.is_file():
        return str(target), None
    try:
        labels = shot_labels(attempt, Path(xml_path).read_text(), images=images)
    except Exception as error:  # a typed label failure is recorded by the caller, never retried
        return str(target), f"{type(error).__name__}: {error}"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(labels, temporary)
    temporary.replace(target)
    return str(target), None


# ------------------------------------------------------------ fan repair check

def residual_check(attempt, xml_text):
    """Residual acceleration of contact-free dynamic bodies against the legacy (ABFan) and the
    repaired (Fan) labels over every native step of one fan capture. Returns per-labeler lists of
    projections on the labelled force direction, plus confusion counts against 'anomalous'
    (|residual| > 10 m/s^2) steps."""
    _, frames, segment = shot_frames(attempt)
    native = segment["native_root"]
    if not Path(native).is_dir():
        native = str(Path(attempt) / "aligned" / Path(native).name)
    m = manifest(native)
    source, materials, _ = level_source(xml_text)
    launches = defaultdict(set)
    out = {name: {"projections": [], "tp": 0, "fp": 0, "fn": 0, "tn": 0} for name in ("legacy", "repaired")}
    previous = None
    for descriptor in m["chunks"]:
        value = json.loads(chunk_text(native, descriptor))
        for e in value["events"]:
            if e["event_type"] == "bird_launched":
                launches[e["fixed_step"]] |= {p.replace("runtime:", "") for p in e["participants"]}
        for sample in value["fixed_step_samples"]:
            step = sample["fixed_step"]
            launched = set().union(*[v for s, v in launches.items() if s <= step]) if launches else set()
            bodies = {e["scenario_object_id"]: e for e in sample["entities"]
                      if e["lifecycle"] == "active" and e["body_present"] and e.get("body")
                      and e["body"]["body_type"] == "dynamic"}
            gravity = sample["world"]["gravity_vector"]
            if previous is not None and previous[0] == step - 1:
                _, old_bodies, old_sample, old_launched = previous
                labels = {"legacy": legacy_fan_edges(source, old_sample, materials, old_launched),
                          "repaired": fan_edges(old_sample)}
                for slot, entity in bodies.items():
                    before = old_bodies.get(slot)
                    if before is None or entity["contact_ids"] or before["contact_ids"]:
                        continue
                    scale = before["body"]["gravity_scale"]
                    rx = (entity["body"]["velocity"][0] - before["body"]["velocity"][0]) / FIXED_DELTA \
                        - gravity[0] * scale
                    ry = (entity["body"]["velocity"][1] - before["body"]["velocity"][1]) / FIXED_DELTA \
                        - gravity[1] * scale
                    if math.hypot(rx, ry) > 5000:  # launch impulse
                        continue
                    anomalous = math.hypot(rx, ry) > 10
                    for name, edges in labels.items():
                        force = edges.get(slot)
                        block = out[name]
                        if force:
                            norm = math.hypot(*force)
                            block["projections"].append((rx * force[0] + ry * force[1]) / norm)
                        key = ("tp" if anomalous else "fp") if force else ("fn" if anomalous else "tn")
                        block[key] += 1
            previous = (step, bodies, sample, launched)
    return out
