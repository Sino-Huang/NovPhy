"""Frozen CLEVRER representation and horizon-boundary estimands for issue #79.

Phase-0 mapping (declared before any training; real numbers are published in the
frozen plan by ``scripts.run_clevrer_boundary_replication --prepare``):

- The carrier reuses the unchanged issue-71 layout: 2 global values plus 18
  slots of 13 columns. Slot columns follow the deployment-carrier semantics:
  ``0`` presence, ``1`` presence mask, ``2`` kind code, ``3`` kind confidence,
  ``4`` kind mask, ``5`` position x, ``6`` position y, ``7`` position mask,
  ``8`` velocity x, ``9`` velocity y, ``10`` velocity mask, ``11`` expected-kind
  probability, ``12`` slot-valid mask. The same builder serves both arms.
- CLEVRER ground plane is x-y (z is the near-constant vertical axis); carrier
  positions carry x/SPEED-independent POS_SCALE and y/POS_SCALE, velocities
  carry vx/VEL_SCALE and vy/VEL_SCALE. Vertical location/velocity, orientation
  matrices, and angular velocities are parsed into the per-scene shards for the
  parser-contract record but have no carrier column referent (declared).
- Presence is engine existence (every declared object occupies its slot for all
  128 frames); the ``inside_camera_view`` flag is the availability mask in
  columns 7 and 10 and in every masked estimand. Unavailable fields are zeroed
  exactly as the deployment carrier does.
- Micro predicates: channel 0 ``contact`` from collision events (symmetric),
  channel 1 ``velocity-bin`` (symmetric same-tier indicator). The macro arm is
  ``unsupported`` on CLEVRER (no supports/structure referent) and its estimand
  is dropped: macro pairs are never trained and never evaluated.
- Actions: CLEVRER cascades have no mid-cascade agent action; a null-action
  tensor (zeros, 5-wide) is held constant for both arms.
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass

import numpy as np
import torch

from world_model.training.cnn_hybrid import DIM, SLOTS

FRAMES_PER_CLIP = 128
CLIP_SECONDS = 5.0
PRIOR_ELAPSED_SECONDS = CLIP_SECONDS / FRAMES_PER_CLIP
KIND_VOCABULARY = ("cube", "cylinder", "sphere")
POS_SCALE = 5.0
MEMBER_SCENES = 12
WINDOWS_PER_SCENE = 40
WINDOW_STARTS = tuple(range(WINDOWS_PER_SCENE))
WINDOW_LENGTH = 61
EVAL_CONTEXTS = (0, 1, 2, 3)
ENDPOINTS = (15, 30, 60, 120)
HORIZONS = (1, 5, 15)
NULL_ACTION = (0., 0., 0., 0., 0.)
CONTACT_CHANNEL, VELOCITY_BIN_CHANNEL = 0, 1
SLOT_COLUMNS = (
    ("presence", 0), ("presence_mask", 1), ("kind_code", 2), ("kind_confidence", 3),
    ("kind_mask", 4), ("position_x", 5), ("position_y", 6), ("position_mask", 7),
    ("velocity_x", 8), ("velocity_y", 9), ("velocity_mask", 10),
    ("expected_kind_probability", 11), ("slot_valid_mask", 12),
)
POSITION_COLUMNS, VELOCITY_COLUMNS = (5, 6), (8, 9)
KIND_COLUMNS = (2, 11)
ENTITY_FEATURES = 13
IDENTITY = "clevrer-annotation-feature-carrier-v1"
SLOT_CONTRACT = ("18 fixed slots; declared CLEVRER objects occupy slots 0..n-1 in"
                 " object_id order for every frame; remaining slots stay zero;"
                 " identical builder for both arms (symmetric Phase-0 declaration)")


class ClevrerBoundaryError(ValueError):
    """Typed blocker for representation or membership failures."""


# ---------------------------------------------------------------- annotations


@dataclass(frozen=True)
class SceneAnnotations:
    """One parsed CLEVRER annotation JSON, tensor form."""

    scene_index: int
    video_filename: str
    objects: tuple[dict, ...]
    location: torch.Tensor          # (frames, n, 3)
    orientation: torch.Tensor       # (frames, n, 3) euler angles
    velocity: torch.Tensor          # (frames, n, 3)
    angular_velocity: torch.Tensor  # (frames, n, 3)
    inside_camera_view: torch.Tensor  # (frames, n) bool
    collision_frames: torch.Tensor    # (events,) long
    collision_pairs: torch.Tensor     # (events, 2) long


def parse_scene(payload: dict) -> SceneAnnotations:
    """Tensorize one annotation dict; typed failure on contract violations."""
    for key in ("scene_index", "video_filename", "object_property", "motion_trajectory", "collision"):
        if key not in payload:
            raise ClevrerBoundaryError(f"scene {payload.get('scene_index')}: missing field {key}")
    objects = tuple(payload["object_property"])
    count = len(objects)
    if count == 0:
        raise ClevrerBoundaryError(f"scene {payload['scene_index']}: no declared objects")
    for obj in objects:
        if obj.get("shape") not in KIND_VOCABULARY:
            raise ClevrerBoundaryError(
                f"scene {payload['scene_index']}: shape {obj.get('shape')!r} outside frozen vocabulary")
    frames = payload["motion_trajectory"]
    if len(frames) != FRAMES_PER_CLIP or [f["frame_id"] for f in frames] != list(range(FRAMES_PER_CLIP)):
        raise ClevrerBoundaryError(
            f"scene {payload['scene_index']}: motion trajectory is not {FRAMES_PER_CLIP} contiguous frames")
    location = torch.zeros((FRAMES_PER_CLIP, count, 3))
    orientation = torch.zeros((FRAMES_PER_CLIP, count, 3))
    velocity = torch.zeros((FRAMES_PER_CLIP, count, 3))
    angular = torch.zeros((FRAMES_PER_CLIP, count, 3))
    visible = torch.zeros((FRAMES_PER_CLIP, count), dtype=torch.bool)
    for frame in frames:
        time = frame["frame_id"]
        seen = set()
        for entry in frame["objects"]:
            index = entry["object_id"]
            if not 0 <= index < count or index in seen:
                raise ClevrerBoundaryError(
                    f"scene {payload['scene_index']}: object {index} missing or duplicated at frame {time}")
            seen.add(index)
            location[time, index] = torch.tensor(entry["location"])
            orientation[time, index] = torch.tensor(entry["orientation"])
            velocity[time, index] = torch.tensor(entry["velocity"])
            angular[time, index] = torch.tensor(entry["angular_velocity"])
            visible[time, index] = bool(entry["inside_camera_view"])
        if len(seen) != count:
            raise ClevrerBoundaryError(
                f"scene {payload['scene_index']}: frame {time} declares {len(seen)} of {count} objects")
    events = payload["collision"]
    pairs, times = [], []
    for event in events:
        first, second = event["object_ids"]
        if not (0 <= first < count and 0 <= second < count) or first == second:
            raise ClevrerBoundaryError(
                f"scene {payload['scene_index']}: collision pair {event['object_ids']} violates slot contract")
        if not 0 <= event["frame_id"] < FRAMES_PER_CLIP:
            raise ClevrerBoundaryError(
                f"scene {payload['scene_index']}: collision frame {event['frame_id']} outside clip")
        pairs.append((first, second))
        times.append(event["frame_id"])
    return SceneAnnotations(int(payload["scene_index"]), str(payload["video_filename"]), objects,
                            location, orientation, velocity, angular, visible,
                            torch.tensor(times, dtype=torch.long), torch.tensor(pairs, dtype=torch.long))


def collision_complete(scene: SceneAnnotations) -> bool:
    """Frozen membership predicate: contiguous full trajectories and >= 1 collision."""
    return bool(len(scene.collision_frames) >= 1)


def iter_validation_annotations(zip_path):
    """Bounded-memory ascending scan of the validation annotation zip."""
    with zipfile.ZipFile(zip_path) as archive:
        for name in sorted(entry for entry in archive.namelist() if entry.endswith(".json")):
            yield name, json.loads(archive.read(name))


def select_membership(zip_path, *, count=MEMBER_SCENES, scan_limit=15000):
    """First ``count`` collision-complete validation scenes, ascending index.

    Typed failures for skipped scenes are returned, never silently dropped.
    """
    members, skipped, scanned = [], [], 0
    for name, payload in iter_validation_annotations(zip_path):
        if scanned >= scan_limit or len(members) == count:
            break
        scanned += 1
        index = int(payload.get("scene_index", -1))
        try:
            scene = parse_scene(payload)
        except ClevrerBoundaryError as error:
            skipped.append({"scene_index": index, "reason": str(error)})
            continue
        if not collision_complete(scene):
            skipped.append({"scene_index": index, "reason": "no collision event (not collision-complete)"})
            continue
        members.append(scene)
    if len(members) != count:
        raise ClevrerBoundaryError(
            f"membership incomplete: {len(members)} collision-complete scenes within the scanned prefix")
    return members, {"scanned": scanned, "skipped": skipped}


# ---------------------------------------------------------------- carrier


def speed_tiers(speeds: torch.Tensor, edges) -> torch.Tensor:
    """Frozen tier index per speed; ``edges`` are ascending tier boundaries."""
    return sum(speeds >= edge for edge in edges)


def frozen_velocity_scale(scenes: tuple[SceneAnnotations, ...]) -> float:
    """Smallest multiple of 0.5 at or above the membership maximum speed."""
    peak = max(float(scene.velocity.norm(dim=-1).max()) for scene in scenes)
    return float(np.ceil(peak * 2.0) / 2.0)


def frozen_tier_edges(scenes: tuple[SceneAnnotations, ...]) -> tuple[float, float]:
    """Tertile boundaries of per-object speeds over membership frames 0..99."""
    speeds = torch.cat([scene.velocity[:100].norm(dim=-1).reshape(-1) for scene in scenes])
    lower, upper = np.quantile(speeds.numpy(), [1 / 3, 2 / 3])
    return (round(float(lower), 4), round(float(upper), 4))


def clip_carriers(scene: SceneAnnotations, velocity_scale: float) -> torch.Tensor:
    """(FRAMES_PER_CLIP, DIM) carrier clip from one scene; both arms share it."""
    count = len(scene.objects)
    if count > SLOTS:
        raise ClevrerBoundaryError(f"scene {scene.scene_index}: {count} objects exceed the {SLOTS}-slot contract")
    clip = torch.zeros((FRAMES_PER_CLIP, DIM))
    clip[:, 0] = 1.0
    clip[:, 1] = PRIOR_ELAPSED_SECONDS
    visible = scene.inside_camera_view
    kinds = torch.tensor([KIND_VOCABULARY.index(obj["shape"]) for obj in scene.objects])
    for slot in range(count):
        base = 2 + slot * ENTITY_FEATURES
        clip[:, base + 0] = 1.0
        clip[:, base + 1] = 1.0
        clip[:, base + 2] = kinds[slot].item() / (len(KIND_VOCABULARY) - 1)
        clip[:, base + 3] = 1.0
        clip[:, base + 4] = 1.0
        clip[:, base + 5] = torch.where(visible[:, slot], scene.location[:, slot, 0] / POS_SCALE, 0.)
        clip[:, base + 6] = torch.where(visible[:, slot], scene.location[:, slot, 1] / POS_SCALE, 0.)
        clip[:, base + 7] = visible[:, slot].float()
        clip[:, base + 8] = torch.where(visible[:, slot], scene.velocity[:, slot, 0] / velocity_scale, 0.)
        clip[:, base + 9] = torch.where(visible[:, slot], scene.velocity[:, slot, 1] / velocity_scale, 0.)
        clip[:, base + 10] = visible[:, slot].float()
        clip[:, base + 11] = 1.0
        clip[:, base + 12] = 1.0
    return clip


def clip_relations(scene: SceneAnnotations, tier_edges) -> tuple[torch.Tensor, torch.Tensor]:
    """(FRAMES_PER_CLIP, SLOTS, SLOTS, 2) micro labels and label mask."""
    count = len(scene.objects)
    relations = torch.zeros((FRAMES_PER_CLIP, SLOTS, SLOTS, 2))
    mask = torch.zeros((FRAMES_PER_CLIP, SLOTS, SLOTS, 2), dtype=torch.bool)
    declared = torch.zeros(SLOTS, dtype=torch.bool)
    declared[:count] = True
    pair_mask = declared[None, :] & declared[:, None] & ~torch.eye(SLOTS, dtype=torch.bool)
    mask[:] = pair_mask[None, :, :, None]
    tiers = torch.full((FRAMES_PER_CLIP, SLOTS), -1, dtype=torch.long)
    tiers[:, :count] = speed_tiers(scene.velocity[:, :count].norm(dim=-1), tier_edges)
    same = (tiers[:, :, None] == tiers[:, None, :]) & pair_mask[None]
    relations[:, :, :, VELOCITY_BIN_CHANNEL] = same.float()
    for time, (first, second) in zip(scene.collision_frames.tolist(), scene.collision_pairs.tolist()):
        relations[time, first, second, CONTACT_CHANNEL] = 1.0
        relations[time, second, first, CONTACT_CHANNEL] = 1.0
    return relations, mask


# ---------------------------------------------------------------- windows


def training_windows(clip: torch.Tensor, relations: torch.Tensor, relations_mask: torch.Tensor):
    """40 windows of WINDOW_LENGTH frames at the frozen starts, window order frozen."""
    z, labels, masks = [], [], []
    for start in WINDOW_STARTS:
        stop = start + WINDOW_LENGTH
        if stop > FRAMES_PER_CLIP:
            raise ClevrerBoundaryError(f"window start {start} exceeds the {FRAMES_PER_CLIP}-frame clip")
        z.append(clip[start:stop])
        labels.append(relations[start:stop])
        masks.append(relations_mask[start:stop])
    window = {"z": torch.stack(z), "action": torch.zeros((len(WINDOW_STARTS), 5)),
              "length": torch.full((len(WINDOW_STARTS),), WINDOW_LENGTH - 1, dtype=torch.long),
              "relations": torch.stack(labels), "relations_mask": torch.stack(masks)}
    return window


def check_eval_grid():
    """Frozen divisibility and clip-bound rule for every (context, endpoint, horizon)."""
    for context in EVAL_CONTEXTS:
        for endpoint in ENDPOINTS:
            if context + endpoint > FRAMES_PER_CLIP - 1:
                raise ClevrerBoundaryError(f"context {context} + endpoint {endpoint} leaves the clip")
            for horizon in HORIZONS:
                if endpoint % horizon:
                    raise ClevrerBoundaryError(f"endpoint {endpoint} not divisible by horizon {horizon}")


# ---------------------------------------------------------------- estimands


def field_errors(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    """Recursive-endpoint field errors: position/velocity under the camera mask."""
    if prediction.shape != (DIM,) or target.shape != (DIM,):
        raise ClevrerBoundaryError("estimand requires one full carrier per unit")
    slots_p = prediction[2:].reshape(SLOTS, ENTITY_FEATURES)
    slots_t = target[2:].reshape(SLOTS, ENTITY_FEATURES)
    result = {"carrier_mse": float((prediction - target).square().mean()),
              "presence_mse": float((slots_p[:, 0] - slots_t[:, 0]).square().mean())}
    for name, columns, mask_column in (("position", POSITION_COLUMNS, 7), ("velocity", VELOCITY_COLUMNS, 10),
                                       ("kind", KIND_COLUMNS, 12)):
        mask = slots_t[:, mask_column] > .5
        squared = (slots_p[:, columns] - slots_t[:, columns]).square()
        result[name + "_available_values"] = int(mask.sum()) * len(columns)
        result[name + "_mse"] = float(squared[mask].mean()) if bool(mask.any()) else None
    if any(isinstance(value, float) and not np.isfinite(value) for value in result.values()):
        raise ClevrerBoundaryError("nonfinite error statistic; do not serialize as favorable missing data")
    return result


def interval_active(collision_frames: torch.Tensor, context: int, endpoint: int) -> bool:
    """Contact-active regime flag: any collision event inside (context, context+endpoint]."""
    return bool(((collision_frames > context) & (collision_frames <= context + endpoint)).any())
