"""Observed RGB/endpoint data and separately masked native oracle supervision.

Only the training driver decides which frozen role may be fitted. Engine
metadata and labels never enter the visual/history encoder's inference inputs.
"""
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from scripts.native_segment_trace import NativeSegmentTrace
from scripts.observation_trace import validate_observation_trace
from world_model.data.deployment_temporal import OBJECT_KIND_VOCABULARY

VOCABULARY = tuple([f"bird:{i:04}" for i in range(3)] + [f"block:{i:04}" for i in range(7)]
                   + ["pig:0000"] + [f"platform:{i:04}" for i in range(8)]
                   + ["slingshot:0000", "world:landscape:0000", "world:landscape:0001"])
VISUAL_DIM = 2 + 13 * len(VOCABULARY)
KIND_INDICES = tuple(OBJECT_KIND_VOCABULARY.index(name.split(":", 1)[0]) for name in VOCABULARY)


def action_context(action):
    """Known shot context, not a repeated executed-action event."""
    if "drag_release" in action:
        x, y = action["drag_release"]
        hold, tap = action.get("releaseTime", action.get("release_time", 600)), action.get("tapTime", action.get("tap_time", 0))
    else:
        x, y = action["drag_x"], action["drag_y"]
        hold, tap = action["release_time_ms"], action["tap_time_ms"]
    return torch.tensor([x / 480., y / 480., hold / 1000., tap / 1000., 1.], dtype=torch.float32)


def project_center(position, metadata):
    transform = metadata["world_to_observation_transform"]
    clip = (np.asarray(transform["camera_to_clip_matrix"]).reshape(4, 4)
            @ np.asarray(transform["world_to_camera_matrix"]).reshape(4, 4)
            @ np.array([*position, 0., 1.]))
    if abs(clip[3]) < 1e-9:
        raise ValueError("native label has a singular projection")
    ndc = clip[:3] / clip[3]
    pixel = np.asarray(transform["ndc_to_observation_matrix"]).reshape(3, 3) @ np.array([*ndc[:2], 1.])
    viewport = metadata["viewport"]
    return torch.tensor(np.clip(pixel[:2] / [viewport["width_pixels"], viewport["height_pixels"]], 0, 1), dtype=torch.float32)


def physical_center(entity, colliders):
    if entity["body"] is not None:
        return entity["body"]["position"]
    points = []
    for collider in colliders:
        if not collider["enabled"] or collider["is_trigger"]:
            continue
        shape = collider["shape"]
        if shape["kind"] in ("circle", "box", "capsule"):
            points.append(shape["center"])
        elif shape["kind"] == "polygon":
            points.extend(p for path in shape["paths"] for p in path)
        elif shape["kind"] == "edge":
            points.extend(shape["points"])
    return np.mean(points, axis=0).tolist() if points else None


def native_labels(trace, metadata_by_step):
    """Native event-established stability, never the legacy two-tick derivation.

    Before the first actual stability event, both macro values are unavailable.
    Structure change uses the immediately preceding native support set, even
    when it lies in another chunk or between two rendered observations.
    """
    stability_events = {}
    for chunk in trace.trace.chunks():
        for event in chunk["events"]:
            if event["event_type"] in ("stable_entered", "stable_exited"):
                step = event["fixed_step"]
                if step in stability_events:
                    raise ValueError("multiple native stability events at one fixed step")
                stability_events[step] = event["event_type"] == "stable_entered"
    slot_ids = {"runtime:" + name: i for i, name in enumerate(VOCABULARY)}
    steady = None
    previous_supports = None
    last_log = time.monotonic()
    for chunk in trace.trace.chunks():
        for sample in chunk["fixed_step_samples"]:
            step = sample["fixed_step"]
            if step in stability_events:
                steady = stability_events[step]
            supports = {(s["supporter_entity_id"], s["supported_entity_id"]) for s in sample["supports"]}
            if step in metadata_by_step:
                colliders = defaultdict(list)
                for collider in sample["colliders"]:
                    colliders[collider["entity_id"]].append(collider)
                presence = torch.zeros(len(VOCABULARY))
                centers = torch.zeros(len(VOCABULARY), 2)
                for entity in sample["entities"]:
                    identity = entity["entity_id"]
                    if identity not in slot_ids or entity["lifecycle"] != "active":
                        continue
                    center = physical_center(entity, colliders[identity])
                    if center is not None:
                        slot = slot_ids[identity]
                        presence[slot] = 1
                        centers[slot] = project_center(center, metadata_by_step[step])
                active = presence.bool()
                mask = active[:, None] & active[None, :] & ~torch.eye(len(VOCABULARY), dtype=torch.bool)
                relations = torch.zeros(len(VOCABULARY), len(VOCABULARY), 2)
                for contact in sample["contacts"]:
                    a, b = contact["entity_a_id"], contact["entity_b_id"]
                    if a in slot_ids and b in slot_ids:
                        relations[slot_ids[a], slot_ids[b], 0] = 1
                        relations[slot_ids[b], slot_ids[a], 0] = 1
                for a, b in supports:
                    if a in slot_ids and b in slot_ids:
                        relations[slot_ids[a], slot_ids[b], 1] = 1
                macro_mask = torch.tensor([steady is not None, steady is not None and previous_supports is not None])
                macros = torch.tensor([bool(steady), steady is False and previous_supports is not None and supports != previous_supports], dtype=torch.float32)
                yield {"fixed_step": step, "presence": presence, "centers": centers,
                       "relations": relations, "relation_mask": mask[..., None].expand(-1, -1, 2).clone(),
                       "macros": macros, "macro_mask": macro_mask}
            previous_supports = supports
        if time.monotonic() - last_log >= 5:
            print(f"[native-data] labels step={sample['fixed_step']}/{trace.summary['last_fixed_step']}", flush=True)
            last_log = time.monotonic()


def prepare_episode(collection_root, member, result):
    """Derive a source-bound shard; collection role is retained, never promoted."""
    rows, references, segment_ranges = [], [], []
    if result["member_identity"] != member["identity"]:
        raise ValueError("prepared data member differs from collection result")
    for ordinal, segment in enumerate(result["segments"], 1):
        trace = NativeSegmentTrace(segment["native_root"])
        if trace.summary != segment["summary"]:
            raise ValueError("prepared native segment source differs")
        root = Path(collection_root) / "attempts" / member["identity"] / f"shot-{ordinal}/observation-trace"
        manifest = validate_observation_trace(root)
        if manifest["identity"] != segment["observation_manifest"] or manifest["exposure_role"] != member["exposure_role"]:
            raise ValueError("prepared observation source or exposure role differs")
        frames = manifest["frame_records"]
        if [f["fixed_step"] for f in frames] != [f["fixed_step"] for f in trace.manifest["frame_records"]]:
            raise ValueError("native physics and observations do not share exact endpoints")
        labels = native_labels(trace, {f["fixed_step"]: f["capture_metadata"] for f in frames})
        start = len(rows)
        for frame, label in zip(frames, labels, strict=True):
            if frame["fixed_step"] != label.pop("fixed_step"):
                raise ValueError("native labels are misaligned")
            with Image.open(root / frame["agent_observation"]["relative_path"]) as opened:
                rgb = np.asarray(opened.convert("RGB").resize((96, 64), Image.Resampling.BILINEAR), dtype=np.uint8).copy()
            label.update(images=torch.from_numpy(rgb).permute(2, 0, 1),
                         timestamps=torch.tensor(frame["fixed_time_seconds"], dtype=torch.float64),
                         fixed_steps=torch.tensor(frame["fixed_step"], dtype=torch.int64))
            rows.append(label)
            references.append({"capture_id": trace.manifest["capture_id"], "fixed_step": frame["fixed_step"],
                               "observation_identity": frame["agent_observation"]["identity"]})
        segment_ranges.append({"start": start, "stop": len(rows), "censored": trace.censored,
                               "action": action_context(member["actions"][ordinal - 1]),
                               "raw_capture_failure": trace.manifest["failure"]})
    return {"schema": "issue_76_native_history_shard_v1", "member_identity": member["identity"],
            "base_cluster": member["base_cluster"], "exposure_role": member["exposure_role"],
            "source_result": result, "references": references, "segment_ranges": segment_ranges,
            "tensors": {key: torch.stack([r[key] for r in rows]) for key in rows[0]} if rows else {},
            "macro_semantics": "native-event-established-steady-state; immediately-prior-native-step support changes",
            "relation_scope": "complete induced relation labels on 22 represented active physical slots; not a full-world graph",
            "synthetic": False}


def visual_carriers(output, timestamps):
    """Existing temporal feature definitions applied to the 22-slot native schema."""
    presence = output["presence_logits"].sigmoid()
    centers = output["centers"]
    kinds = output["kind_logits"].softmax(-1)
    kind_confidence, kind_index = kinds.max(-1)
    expected = torch.tensor(KIND_INDICES, device=presence.device)
    expected_probability = kinds.gather(-1, expected[None, :, None].expand(len(presence), -1, -1)).squeeze(-1)
    available = presence >= .5
    elapsed = torch.zeros_like(timestamps)
    elapsed[1:] = timestamps[1:] - timestamps[:-1]
    if bool((elapsed[1:] <= 0).any()):
        raise ValueError("visual episode timestamps must strictly increase")
    motion_available = torch.zeros_like(available)
    motion_available[1:] = available[1:] & available[:-1]
    motion = torch.zeros_like(centers)
    motion[1:] = (centers[1:] - centers[:-1]) / elapsed[1:, None, None].to(centers)
    motion = torch.where(motion_available[..., None], motion, 0.)
    ones = torch.ones_like(presence)
    features = torch.stack((presence, ones, kind_index.to(presence) / 6, kind_confidence, ones,
        torch.where(available, centers[..., 0], 0.), torch.where(available, centers[..., 1], 0.),
        available.to(presence), motion[..., 0], motion[..., 1], motion_available.to(presence),
        expected_probability, ones), -1).flatten(1)
    prior = torch.ones_like(elapsed)
    prior[0] = 0
    return torch.cat((prior[:, None].to(features), elapsed[:, None].to(features), features), -1)


def history_events(visual, timestamps, segments):
    """One action-only event follows each pre-intervention observation exactly once."""
    starts = {s["start"]: s["action"].to(visual) for s in segments}
    carriers, actions, times, observed, acted, observation_indices = [], [], [], [], [], []
    for i, carrier in enumerate(visual):
        observation_indices.append(len(carriers))
        carriers.append(carrier)
        actions.append(visual.new_zeros(5))
        times.append(timestamps[i])
        observed.append(True)
        acted.append(False)
        if i in starts:
            carriers.append(torch.zeros_like(carrier))
            actions.append(starts[i])
            times.append(timestamps[i])
            observed.append(False)
            acted.append(True)
    return {"carriers": torch.stack(carriers)[None], "actions": torch.stack(actions)[None],
            "timestamps": torch.stack(times)[None],
            "observation_mask": torch.tensor([observed], dtype=torch.bool, device=visual.device),
            "action_mask": torch.tensor([acted], dtype=torch.bool, device=visual.device)}, observation_indices
