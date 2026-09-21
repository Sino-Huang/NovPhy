"""Frozen Physion-Dominoes representation and horizon-boundary estimands for issue #84.

Third-family replication vehicle: the horizon-decay boundary measured on NovPhy
(#77) and CLEVRER (#79) gains its out-of-family tie-breaker here. Phase-0
mapping (verified against the released data BEFORE any training; real numbers
are published in the frozen plan by
``scripts.run_third_family_boundary_replication --prepare``):

- Source family: Physion (NeurIPS 2021 D&B) scenario family ``Dominoes``,
  dynamics-training HDF5 release, pinned by URL/ETag/byte size/Last-Modified.
  The archive is read as a bounded prefix (frozen byte cap); its stored member
  order is the release's published index.
- Verified per-frame fields (dt = 0.01 s, verified from free-fall
  dv_y = -0.098 m/s per frame = g * dt): ``objects/positions`` are engine
  transform ORIGINS (base pivots, world frame, y-up, meters),
  ``objects/rotations`` are world quaternions (x, y, z, w),
  ``objects/velocities`` are center-of-mass velocities (m/s),
  ``objects/angular_velocities`` the angular counterparts. Origins and
  center-of-mass velocities are distinct engine readouts and are carried
  verbatim under their published semantics.
- The carrier reuses the unchanged issue-71 layout: 2 global values plus 18
  slots of 13 columns. Ground-plane axes are x and z (columns 5, 6 and 8, 9
  carry x, z); the vertical axis y, the orientation quaternions, and the
  angular velocities are parsed into the per-scene shards for the
  parser-contract record and have no carrier column referent (declared).
- Availability: the release is dense (every declared object appears in every
  frame), so the frozen rule is the DECLARED ALL-FRAMES RULE: presence is
  engine existence over the whole trial and every availability mask is 1.
  Membership additionally enforces the density as a completeness predicate.
- Micro predicates: channel 0 ``contact`` from per-frame object-object
  collision events (any recorded enter/stay/exit state labels that frame,
  symmetric); channel 1 ``velocity-bin`` (symmetric same-tier indicator at
  frozen tertile edges). The macro arm is ``unsupported`` (the family has no
  support-structure referent in its annotations) and its estimand is DROPPED:
  macro pairs are never trained and never evaluated.
- Actions: dominoes cascades have no mid-cascade interface action (the probe
  push is part of the recorded physics); a null-action tensor (zeros, 5-wide)
  is held constant for both arms.
"""
from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass

import h5py
import numpy as np
import torch

from world_model.training.cnn_hybrid import DIM, SLOTS
from world_model.training.clevrer_boundary import (
    CONTACT_CHANNEL,
    ENTITY_FEATURES,
    KIND_COLUMNS,
    POSITION_COLUMNS,
    VELOCITY_COLUMNS,
    VELOCITY_BIN_CHANNEL,
    field_errors,
    interval_active,
    speed_tiers,
)

PHYSION_DT_SECONDS = 0.01
GROUND_AXIS_X, GROUND_AXIS_Z, VERTICAL_AXIS_Y = 0, 2, 1
FRAMES_MINIMUM = 124
MEMBER_SCENES = 12
WINDOWS_PER_SCENE = 40
WINDOW_LENGTH = 61
EVAL_CONTEXTS = (0, 1, 2, 3)
ENDPOINTS = (15, 30, 60, 120)
HORIZONS = (1, 5, 15)
NULL_ACTION = (0., 0., 0., 0., 0.)
REGIME_WINDOW = 30
IDENTITY = "physion-dominoes-annotation-feature-carrier-v1"
SLOT_CONTRACT = ("18 fixed slots; the declared dominoes-trial objects occupy slots 0..n-1 in"
                 " static object_ids order for every frame; remaining slots stay zero;"
                 " identical builder for both arms (symmetric Phase-0 declaration)")
COMPLETE_RULE = ("frame-complete: contiguous frames 0000..n-1 with n >= 124, the frozen"
                 " static object_ids present densely in every frame, finite positions and"
                 " velocities, and at least one object-object collision event")


class ThirdFamilyError(ValueError):
    """Typed blocker for representation or membership failures."""


# ---------------------------------------------------------------- annotations


@dataclass(frozen=True)
class TrialAnnotations:
    """One parsed Physion dominoes trial HDF5, tensor form."""

    scene_index: int               # member ordinal within the archive scan
    stimulus_name: str             # hdf5 member basename, the released identity
    model_names: tuple[str, ...]   # static model name per declared object
    object_ids: torch.Tensor       # (N,) long
    positions: torch.Tensor        # (n, N, 3) transform origins (world, y-up, m)
    rotations: torch.Tensor        # (n, N, 4) world quaternions x y z w
    velocity: torch.Tensor         # (n, N, 3) center-of-mass velocity (m/s)
    angular_velocity: torch.Tensor # (n, N, 3)
    collision_frames: torch.Tensor  # (E,) long frames with object-object events
    collision_pairs: torch.Tensor   # (E, 2) long colliding object slots


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def parse_trial(name: str, payload, scene_index: int) -> TrialAnnotations:
    """Tensorize one trial HDF5 (bytes or file object); typed failure on violations."""
    stimulus = name.rsplit("/", 1)[-1].removesuffix(".hdf5")
    try:
        return _parse_trial(stimulus, payload, scene_index)
    except OSError as error:
        raise ThirdFamilyError(f"{stimulus}: unreadable trial HDF5 ({error})") from error


def _parse_trial(stimulus: str, payload, scene_index: int) -> TrialAnnotations:
    with h5py.File(io.BytesIO(payload) if isinstance(payload, bytes) else payload, "r") as handle:
        for key in ("static", "frames"):
            if key not in handle:
                raise ThirdFamilyError(f"{stimulus}: missing group {key}")
        static = handle["static"]
        for key in ("object_ids", "model_names"):
            if key not in static:
                raise ThirdFamilyError(f"{stimulus}: static is missing {key}")
        object_ids = torch.tensor(np.asarray(static["object_ids"][:]), dtype=torch.long)
        count = int(object_ids.numel())
        if count == 0 or len(set(object_ids.tolist())) != count:
            raise ThirdFamilyError(f"{stimulus}: object_ids empty or not unique")
        model_names = tuple(_decode(v) for v in np.asarray(static["model_names"][:]).tolist())
        if len(model_names) != count:
            raise ThirdFamilyError(f"{stimulus}: model_names declares {len(model_names)} of {count} objects")
        frames = sorted(handle["frames"].keys())
        n = len(frames)
        if n < FRAMES_MINIMUM or frames != [f"{index:04d}" for index in range(n)]:
            raise ThirdFamilyError(
                f"{stimulus}: frames are not contiguous 0000..{n - 1} with n >= {FRAMES_MINIMUM}")
        positions = torch.zeros((n, count, 3))
        rotations = torch.zeros((n, count, 4))
        velocity = torch.zeros((n, count, 3))
        angular = torch.zeros((n, count, 3))
        for time, key in enumerate(frames):
            frame = handle["frames"][key]
            objects = frame["objects"] if "objects" in frame else None
            if objects is None or "positions" not in objects or "velocities" not in objects \
                    or "rotations" not in objects or "angular_velocities" not in objects:
                raise ThirdFamilyError(f"{stimulus}: frame {time} lacks the objects state group")
            shapes = (tuple(objects["positions"].shape), tuple(objects["rotations"].shape),
                      tuple(objects["velocities"].shape), tuple(objects["angular_velocities"].shape))
            if shapes != ((count, 3), (count, 4), (count, 3), (count, 3)):
                raise ThirdFamilyError(
                    f"{stimulus}: frame {time} state shapes {shapes} violate the dense (N,3)/(N,4) contract")
            positions[time] = torch.tensor(np.asarray(objects["positions"][:]), dtype=torch.float32)
            rotations[time] = torch.tensor(np.asarray(objects["rotations"][:]), dtype=torch.float32)
            velocity[time] = torch.tensor(np.asarray(objects["velocities"][:]), dtype=torch.float32)
            angular[time] = torch.tensor(np.asarray(objects["angular_velocities"][:]), dtype=torch.float32)
        pairs, times = [], []
        lookup = {int(v): slot for slot, v in enumerate(object_ids.tolist())}
        for time, key in enumerate(frames):
            collisions = handle["frames"][key]["collisions"] if "collisions" in handle["frames"][key] else None
            if collisions is None or "object_ids" not in collisions:
                raise ThirdFamilyError(f"{stimulus}: frame {time} lacks the collisions group")
            event_pairs = np.asarray(collisions["object_ids"][:]).reshape(-1, 2)
            for first, second in event_pairs.tolist():
                if first not in lookup or second not in lookup or first == second:
                    raise ThirdFamilyError(
                        f"{stimulus}: collision pair ({first}, {second}) at frame {time} violates the id contract")
                pairs.append((lookup[first], lookup[second]))
                times.append(time)
    if not bool(torch.isfinite(positions).all()) or not bool(torch.isfinite(velocity).all()):
        raise ThirdFamilyError(f"{stimulus}: nonfinite object state in the released trajectory")
    return TrialAnnotations(scene_index, stimulus, model_names, object_ids,
                            positions, rotations, velocity, angular,
                            torch.tensor(times, dtype=torch.long),
                            torch.tensor(pairs, dtype=torch.long))


def trial_complete(trial: TrialAnnotations) -> bool:
    """Frozen membership predicate: frame-complete per COMPLETE_RULE."""
    return bool(trial.positions.shape[0] >= FRAMES_MINIMUM
                and len(trial.collision_frames) >= 1
                and torch.isfinite(trial.positions).all()
                and torch.isfinite(trial.velocity).all())


def iter_archive_members(prefix_path):
    """Bounded-memory ascending scan of a downloaded archive prefix.

    Yields ``(member_name, payload_bytes)`` for complete hdf5 members in the
    archive's stored order; the truncated tail member of a bounded prefix is
    skipped, never silently accepted.
    """
    with tarfile.open(prefix_path, "r:gz") as archive:
        while True:
            try:
                member = archive.next()
            except (tarfile.TarError, EOFError, OSError):
                return
            if member is None:
                return
            if not member.isfile() or not member.name.endswith(".hdf5"):
                continue
            try:
                yield member.name, archive.extractfile(member).read()
            except (tarfile.TarError, EOFError, OSError):
                return


def select_membership(prefix_path, *, count=MEMBER_SCENES, member_limit=256):
    """First ``count`` frame-complete trials in the archive's stored member order.

    The stored member order is the release's published index. Typed failures
    for skipped members are returned, never silently dropped.
    """
    members, skipped, scanned = [], [], 0
    for name, payload in iter_archive_members(prefix_path):
        if scanned >= member_limit or len(members) == count:
            break
        scanned += 1
        try:
            trial = parse_trial(name, payload, scene_index=scanned - 1)
        except ThirdFamilyError as error:
            skipped.append({"member": name, "reason": str(error)})
            continue
        if not trial_complete(trial):
            skipped.append({"member": name, "reason": "not frame-complete (COMPLETE_RULE)"})
            continue
        members.append(trial)
    if len(members) != count:
        raise ThirdFamilyError(
            f"membership incomplete: {len(members)} frame-complete trials within the scanned prefix")
    return members, {"members_scanned": scanned, "skipped": skipped}


# ---------------------------------------------------------------- frozen scale rules


def frozen_velocity_scale(trials: tuple[TrialAnnotations, ...]) -> float:
    """Smallest multiple of 0.5 at or above the membership maximum CoM speed."""
    peak = max(float(trial.velocity.norm(dim=-1).max()) for trial in trials)
    return float(np.ceil(peak * 2.0) / 2.0)


def frozen_position_scale(trials: tuple[TrialAnnotations, ...]) -> float:
    """Smallest multiple of 1.0 at or above the membership maximum ground-plane |axis|."""
    peak = max(float(trial.positions[..., [GROUND_AXIS_X, GROUND_AXIS_Z]].abs().max())
               for trial in trials)
    return float(np.ceil(peak))


def frozen_tier_edges(trials: tuple[TrialAnnotations, ...]) -> tuple[float, float]:
    """Tertile boundaries of per-object CoM speeds over membership frames 0..99."""
    speeds = torch.cat([trial.velocity[:100].norm(dim=-1).reshape(-1) for trial in trials])
    lower, upper = np.quantile(speeds.numpy(), [1 / 3, 2 / 3])
    return (round(float(lower), 4), round(float(upper), 4))


def frozen_kind_vocabulary(trials: tuple[TrialAnnotations, ...]) -> tuple[str, ...]:
    """Sorted distinct released model names across the membership."""
    return tuple(sorted({name for trial in trials for name in trial.model_names}))


# ---------------------------------------------------------------- carrier


def clip_carriers(trial: TrialAnnotations, position_scale: float, velocity_scale: float,
                  kind_vocabulary: tuple[str, ...]) -> torch.Tensor:
    """(n, DIM) carrier clip from one trial; both arms share it."""
    count = len(trial.model_names)
    if count > SLOTS:
        raise ThirdFamilyError(f"trial {trial.stimulus_name}: {count} objects exceed the {SLOTS}-slot contract")
    if not kind_vocabulary or any(name not in kind_vocabulary for name in trial.model_names):
        raise ThirdFamilyError(f"trial {trial.stimulus_name}: model name outside the frozen kind vocabulary")
    n = trial.positions.shape[0]
    clip = torch.zeros((n, DIM))
    clip[:, 0] = 1.0
    clip[:, 1] = PHYSION_DT_SECONDS
    for slot in range(count):
        base = 2 + slot * ENTITY_FEATURES
        kind_code = 0.0 if len(kind_vocabulary) == 1 \
            else kind_vocabulary.index(trial.model_names[slot]) / (len(kind_vocabulary) - 1)
        clip[:, base + 0] = 1.0
        clip[:, base + 1] = 1.0
        clip[:, base + 2] = kind_code
        clip[:, base + 3] = 1.0
        clip[:, base + 4] = 1.0
        # declared all-frames rule: every declared object is present and
        # available in every frame; no availability masking applies
        clip[:, base + 5] = trial.positions[:, slot, GROUND_AXIS_X] / position_scale
        clip[:, base + 6] = trial.positions[:, slot, GROUND_AXIS_Z] / position_scale
        clip[:, base + 7] = 1.0
        clip[:, base + 8] = trial.velocity[:, slot, GROUND_AXIS_X] / velocity_scale
        clip[:, base + 9] = trial.velocity[:, slot, GROUND_AXIS_Z] / velocity_scale
        clip[:, base + 10] = 1.0
        clip[:, base + 11] = 1.0
        clip[:, base + 12] = 1.0
    return clip


def clip_relations(trial: TrialAnnotations, tier_edges) -> tuple[torch.Tensor, torch.Tensor]:
    """(n, SLOTS, SLOTS, 2) micro labels and label mask."""
    count = len(trial.model_names)
    relations = torch.zeros((trial.positions.shape[0], SLOTS, SLOTS, 2))
    mask = torch.zeros((trial.positions.shape[0], SLOTS, SLOTS, 2), dtype=torch.bool)
    declared = torch.zeros(SLOTS, dtype=torch.bool)
    declared[:count] = True
    pair_mask = declared[None, :] & declared[:, None] & ~torch.eye(SLOTS, dtype=torch.bool)
    mask[:] = pair_mask[None, :, :, None]
    tiers = torch.full((trial.positions.shape[0], SLOTS), -1, dtype=torch.long)
    tiers[:, :count] = speed_tiers(trial.velocity[:, :count].norm(dim=-1), tier_edges)
    same = (tiers[:, :, None] == tiers[:, None, :]) & pair_mask[None]
    relations[:, :, :, VELOCITY_BIN_CHANNEL] = same.float()
    for time, (first, second) in zip(trial.collision_frames.tolist(), trial.collision_pairs.tolist()):
        relations[time, first, second, CONTACT_CHANNEL] = 1.0
        relations[time, second, first, CONTACT_CHANNEL] = 1.0
    return relations, mask


# ---------------------------------------------------------------- windows


def window_starts(frames: int) -> tuple[int, ...]:
    """Frozen start rule: 40 evenly spread starts spanning [0, frames - WINDOW_LENGTH]."""
    last = frames - WINDOW_LENGTH
    if last < 0:
        raise ThirdFamilyError(f"trial with {frames} frames cannot host a {WINDOW_LENGTH}-frame window")
    return tuple(i * last // (WINDOWS_PER_SCENE - 1) for i in range(WINDOWS_PER_SCENE))


def training_windows(clip: torch.Tensor, relations: torch.Tensor, relations_mask: torch.Tensor):
    """40 windows of WINDOW_LENGTH frames at the frozen starts, window order frozen."""
    starts = window_starts(clip.shape[0])
    z, labels, masks = [], [], []
    for start in starts:
        stop = start + WINDOW_LENGTH
        z.append(clip[start:stop])
        labels.append(relations[start:stop])
        masks.append(relations_mask[start:stop])
    return {"z": torch.stack(z), "action": torch.zeros((len(starts), 5)),
            "length": torch.full((len(starts),), WINDOW_LENGTH - 1, dtype=torch.long),
            "relations": torch.stack(labels), "relations_mask": torch.stack(masks)}


def check_eval_grid(frames: int):
    """Frozen divisibility and trial-bound rule for every (context, endpoint, horizon)."""
    for context in EVAL_CONTEXTS:
        for endpoint in ENDPOINTS:
            if context + endpoint > frames - 1:
                raise ThirdFamilyError(f"frames={frames}: context {context} + endpoint {endpoint} leaves the trial")
            for horizon in HORIZONS:
                if endpoint % horizon:
                    raise ThirdFamilyError(f"endpoint {endpoint} not divisible by horizon {horizon}")


# ---------------------------------------------------------------- estimands


def interval_event_flags(collision_frames: torch.Tensor, context: int, window: int = REGIME_WINDOW) -> bool:
    """Contact-active regime flag: any collision event inside (context, context+window]."""
    return interval_active(collision_frames, context, window)
