"""Lazy action-conditioned temporal-window dataset over an EpisodeCatalog snapshot.

TemporalWindowDataset indexes eligible (episode, accepted-shot, start-frame)
candidates at construction time without opening any PNG files.  Image decoding
happens only inside __getitem__.

Design constraints (from Todo 4):
- Index is materialized once at construction from the catalog snapshot; no
  later filesystem scan can alter the index.
- A TemporalWindowRequest(prediction_steps, stride_frames) requires every shot
  to have at least (1 + prediction_steps * stride_frames) frames for a window
  to be eligible.
- If no eligible window exists across the entire catalog, NoEligibleTemporalWindowError
  is raised at construction time.
- __getitem__ raises FrameReadError (a typed IOError subclass) if a cataloged
  image file is missing or unreadable; it never rescans, bridges shots, or
  fabricates targets.
- Physics / supervision sidecars are never opened, parsed, tensorized, or used.
- The default image transform converts an RGB PIL image to a float32 [0,1] CHW
  tensor without resize or normalization.  An explicit callable transform
  replaces this default for every image in the sample.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, NotRequired, Optional, Sequence

import torch
from PIL import Image, UnidentifiedImageError
from typing import TypedDict

from world_model.data.catalog import EpisodeCatalog
from world_model.data.types import (
    CaptureContractDescriptor,
    ContractValueError,
    EpisodeRecord,
    FrameRecord,
    ShotRecord,
    TemporalWindowRequest,
)
from world_model.data.supervision import (
    AuthoritativePhysicsSidecars,
    PhysicsFrameSupervision,
    PhysicsSupervisionRequest,
    read_physics_shot,
)


# ---------------------------------------------------------------------------
# Public typed errors
# ---------------------------------------------------------------------------


class NoEligibleTemporalWindowError(ValueError):
    """Raised at construction when no eligible temporal window exists in the catalog.

    This is a typed ValueError subclass so callers can distinguish an empty
    catalog/window configuration from other construction errors.
    """

    def __init__(self, split: str, prediction_steps: int, stride_frames: int, horizon_frames: int) -> None:
        self.split = split
        self.prediction_steps = prediction_steps
        self.stride_frames = stride_frames
        self.horizon_frames = horizon_frames
        super().__init__(
            f"No eligible temporal window found in catalog (split={split!r})"
            f" for TemporalWindowRequest(prediction_steps={prediction_steps},"
            f" stride_frames={stride_frames})."
            f" Every shot must have at least {1 + horizon_frames} frames."
        )


# Backwards-compatible alias.
NoEligibleWindowError = NoEligibleTemporalWindowError


class FrameReadError(IOError):
    """Raised when a cataloged frame file cannot be opened during __getitem__.

    This is a typed IOError subclass so callers can distinguish read failures
    from other exceptions without catching the broad OSError hierarchy.
    """

    def __init__(self, frame_path: str | Path, original: Exception) -> None:
        self.frame_path = str(frame_path)
        self.original = original
        super().__init__(f"Cannot read cataloged frame {self.frame_path!r}: {original}")


# ---------------------------------------------------------------------------
# Typed sample mapping
# ---------------------------------------------------------------------------


class WindowSample(TypedDict):
    """Typed dictionary returned by TemporalWindowDataset.__getitem__.

    Keys
    ----
    context_image : torch.Tensor  [C, H, W] float32 in [0, 1]
    target_images : list[torch.Tensor]  prediction_steps × [C, H, W]
    action        : torch.Tensor  [5] float32
    frame_indices : list[int]  len = 1 + prediction_steps  (context + targets)
    horizon_frames  : int
    prediction_steps: int
    stride_frames   : int
    provenance      : dict  (immutable provenance metadata)
    """

    context_image: torch.Tensor
    target_images: list
    action: torch.Tensor
    frame_indices: list
    horizon_frames: int
    prediction_steps: int
    stride_frames: int
    provenance: dict
    supervision: NotRequired[tuple[PhysicsFrameSupervision, ...]]


# ---------------------------------------------------------------------------
# Internal index entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _WindowEntry:
    """One eligible (episode, shot, start_frame) triple stored in the flat index."""

    episode: EpisodeRecord
    shot: ShotRecord
    start_frame: int


# ---------------------------------------------------------------------------
# Default image-to-tensor conversion
# ---------------------------------------------------------------------------


def _pil_to_float_chw(image: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to a float32 CHW tensor in [0, 1]."""
    # Convert to RGB to ensure 3 channels even for palettized sources.
    rgb = image.convert("RGB")
    # to_tensor equivalent: HWC uint8 → CHW float32 / 255.
    import numpy as np  # noqa: PLC0415 – deferred to keep top-level imports minimal
    arr = np.array(rgb, dtype=np.float32) / 255.0
    # HWC → CHW
    return torch.from_numpy(arr.transpose(2, 0, 1))


# ---------------------------------------------------------------------------
# TemporalWindowDataset
# ---------------------------------------------------------------------------


class TemporalWindowDataset(torch.utils.data.Dataset):
    """PyTorch Dataset yielding action-conditioned temporal windows from an EpisodeCatalog.

    Parameters
    ----------
    catalog:
        An immutable EpisodeCatalog snapshot.  The index is built once from
        catalog.episodes; later filesystem changes do not affect it.
    request:
        A TemporalWindowRequest specifying prediction_steps and stride_frames.
        horizon_frames = prediction_steps * stride_frames.
    transform:
        Optional callable applied to every decoded CHW float32 tensor
        (context_image and each element of target_images).  Defaults to
        lossless [0,1] CHW float32 conversion.

    Raises
    ------
    NoEligibleTemporalWindowError:
        If no eligible (episode, shot, start_frame) triple exists in the
        catalog for the given request.  Subclasses ValueError.
    """

    def __init__(
        self,
        catalog: EpisodeCatalog,
        request: TemporalWindowRequest,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        supervision: PhysicsSupervisionRequest | None = None,
        authoritative_sidecars: Mapping[Path, AuthoritativePhysicsSidecars] | None = None,
    ) -> None:
        super().__init__()

        self._catalog = catalog
        self._request = request
        self._transform, self._supervision_request = transform, supervision

        # Derive the catalog root from a private attribute.
        # EpisodeCatalog stores its root in _root; we access it via the internal
        # object.__getattribute__ bypass that the class itself uses.
        self._root: Path = object.__getattribute__(catalog, "_root")
        self._supervision: dict[tuple[str, str], tuple[PhysicsFrameSupervision, ...]] = {}
        bindings: dict[Path, AuthoritativePhysicsSidecars] = {}
        if authoritative_sidecars is not None:
            if not isinstance(authoritative_sidecars, Mapping):
                raise ContractValueError("authoritative_sidecars", "must be a path mapping")
            for path, binding in authoritative_sidecars.items():
                if not isinstance(path, Path) or type(binding) is not AuthoritativePhysicsSidecars:
                    raise ContractValueError(
                        "authoritative_sidecars",
                        "must map Paths to AuthoritativePhysicsSidecars",
                    )
                resolved = path.resolve()
                if resolved in bindings:
                    raise ContractValueError("authoritative_sidecars", "paths must be unique")
                bindings[resolved] = binding
        if bindings and supervision is None:
            raise ContractValueError(
                "authoritative_sidecars", "requires a supervision request"
            )
        if supervision is not None:
            self._load_supervision(catalog, supervision, bindings)

        # Build the flat deterministic index.
        self._index: tuple[_WindowEntry, ...] = self._build_index(catalog, request)

        if not self._index:
            raise NoEligibleTemporalWindowError(
                split=catalog.split,
                prediction_steps=request.prediction_steps,
                stride_frames=request.stride_frames,
                horizon_frames=request.horizon_frames,
            )

    def _load_supervision(
        self,
        catalog: EpisodeCatalog,
        request: PhysicsSupervisionRequest,
        authoritative_sidecars: Mapping[Path, AuthoritativePhysicsSidecars],
    ) -> None:
        if catalog.capture_contract.contract_name != "physics_capture_v1":
            raise ContractValueError("supervision", "physics_capture_v1 is required")
        declared = set(catalog.capture_contract.declared_capabilities)
        missing = tuple(
            capability
            for capability in request.required_capabilities
            if capability not in declared
        )
        if missing:
            raise ContractValueError("supervision capabilities", f"missing {missing!r}")
        consumed: set[Path] = set()
        for episode in catalog.episodes:
            for shot in episode.shots:
                shot_prefix = Path(shot.relative_path)
                frame_paths = tuple(
                    str(Path(frame.relative_path).relative_to(shot_prefix))
                    for frame in shot.frames
                )
                shot_path = (self._root / shot.relative_path).resolve()
                sidecars = authoritative_sidecars.get(shot_path)
                if sidecars is not None:
                    consumed.add(shot_path)
                self._supervision[(episode.name, shot.name)] = read_physics_shot(
                    shot_path,
                    shot.name,
                    frame_paths,
                    request,
                    authoritative_sidecars=sidecars,
                )
        if consumed != set(authoritative_sidecars):
            raise ContractValueError(
                "authoritative_sidecars", "contains a path outside the catalog"
            )

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_index(
        catalog: EpisodeCatalog,
        request: TemporalWindowRequest,
    ) -> tuple[_WindowEntry, ...]:
        """Build a deterministic flat index of eligible window entries.

        Eligibility: a start_frame s in shot is eligible when
            s + horizon_frames < len(shot.frames)
        i.e. context is at s and the last target is at s + horizon_frames,
        both within the shot's frame sequence.
        """
        horizon = request.horizon_frames
        entries: list[_WindowEntry] = []

        for episode in catalog.episodes:
            for shot in episode.shots:
                frame_count = len(shot.frames)
                for start in range(frame_count):
                    if start + horizon < frame_count:
                        entries.append(_WindowEntry(episode, shot, start))

        return tuple(entries)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> WindowSample:
        """Return a sample dictionary for the idx-th eligible window.

        Keys
        ----
        context_image : torch.Tensor  [C, H, W] float32 in [0, 1]
        target_images : list[torch.Tensor]  prediction_steps × [C, H, W]
        action        : torch.Tensor  [5] float32
        frame_indices : list[int]  len = 1 + prediction_steps  (context + targets)
        horizon_frames  : int
        prediction_steps: int
        stride_frames   : int
        provenance      : dict  (immutable provenance metadata)

        Raises
        ------
        FrameReadError:
            If a cataloged frame file is missing or unreadable.
        """
        entry = self._index[idx]
        episode = entry.episode
        shot = entry.shot
        start = entry.start_frame
        request = self._request

        # Collect the frame indices: context at start, targets at stride offsets.
        context_idx = start
        target_idxs = [
            start + step * request.stride_frames
            for step in range(1, request.prediction_steps + 1)
        ]
        all_frame_indices = [context_idx] + target_idxs

        # Resolve absolute paths.
        # FrameRecord.relative_path was rebased by the catalog to be relative to
        # the catalog root (catalog.py _build_episode_record_from_validated prepends
        # episode_relative to each frame's path).
        # So: abs_path = catalog_root / frame_record.relative_path  (no episode prefix needed)
        all_frame_records = [shot.frames[i] for i in all_frame_indices]

        # Decode images (lazy – nothing was opened before this point).
        decoded: list[torch.Tensor] = []
        for frame_record in all_frame_records:
            abs_path = self._root / frame_record.relative_path
            decoded.append(self._decode_frame(abs_path))

        context_image = decoded[0]
        target_images = decoded[1:]

        # Action tensor: strict [5] float32.
        action_values = shot.action.values
        action_tensor = torch.tensor(
            [float(v) for v in action_values], dtype=torch.float32
        )

        # Provenance (immutable metadata, no sidecar files opened).
        descriptor: CaptureContractDescriptor = episode.capture_contract
        provenance = {
            "split": episode.split,
            "source_level_key": episode.source_level_key,
            "episode": episode.name,
            "shot": shot.name,
            "shot_frame_count": len(shot.frames),
            "capture_contract": descriptor,
            "declared_capabilities": descriptor.declared_capabilities,
            "sidecar_paths": descriptor.sidecar_paths,
        }

        sample = WindowSample(
            context_image=context_image,
            target_images=target_images,
            action=action_tensor,
            frame_indices=all_frame_indices,
            horizon_frames=request.horizon_frames,
            prediction_steps=request.prediction_steps,
            stride_frames=request.stride_frames,
            provenance=provenance,
        )
        if self._supervision_request is not None:
            available = self._supervision[(episode.name, shot.name)]
            sample["supervision"] = tuple(available[index] for index in all_frame_indices)
        return sample

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_frame(self, abs_path: Path) -> torch.Tensor:
        """Decode a single PNG frame to a float32 CHW tensor.

        Raises FrameReadError on any I/O or decode failure.
        """
        try:
            with Image.open(abs_path) as img:
                tensor = _pil_to_float_chw(img)
        except (OSError, UnidentifiedImageError) as exc:
            raise FrameReadError(abs_path, exc) from exc

        if self._transform is not None:
            tensor = self._transform(tensor)
        return tensor
