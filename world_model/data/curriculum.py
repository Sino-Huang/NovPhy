"""Immutable, temporal-only curriculum policy over a fixed episode catalog."""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from typing import TypedDict

from world_model.data.catalog import EpisodeCatalog
from world_model.data.types import (
    ContractValueError,
    EpisodeRecord,
    ShotRecord,
    TemporalWindowRequest,
)

def _require_nonempty_string(value: str, field: str) -> None:
    if type(value) is not str or not value.strip():
        raise ContractValueError(field, "must be a nonempty string")


def _require_integer(value: int, field: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ContractValueError(field, f"must be a {qualifier} integer")


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    name: str
    start_step: int; end_step: int
    temporal_choices: tuple[TemporalWindowRequest, ...]
    novelty_levels: tuple[str, ...] | None = None; scenario_types: tuple[str, ...] | None = None
    start_frame_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.name, "stage name")
        _require_integer(self.start_step, "start_step", 0)
        if type(self.end_step) is not int or self.end_step <= self.start_step:
            raise ContractValueError("end_step", "must be an integer greater than start_step")
        if type(self.temporal_choices) is not tuple or not self.temporal_choices:
            raise ContractValueError("temporal_choices", "must be a nonempty immutable tuple")
        if any(type(choice) is not TemporalWindowRequest for choice in self.temporal_choices):
            raise ContractValueError("temporal_choices", "must contain temporal window requests")
        if len(set(self.temporal_choices)) != len(self.temporal_choices):
            raise ContractValueError("temporal_choices", "must be unique")
        self._validate_filter("novelty_levels", self.novelty_levels)
        self._validate_filter("scenario_types", self.scenario_types)
        self._validate_start_frame_range(self.start_frame_range)

    @staticmethod
    def _validate_filter(name: str, values: tuple[str, ...] | None) -> None:
        if values is None:
            return
        if type(values) is not tuple or not values:
            raise ContractValueError(name, "must be a nonempty immutable tuple when specified")
        if any(type(value) is not str or not value.strip() for value in values):
            raise ContractValueError(name, "must contain nonempty strings")
        if len(set(values)) != len(values):
            raise ContractValueError(name, "must contain unique values")

    @staticmethod
    def _validate_start_frame_range(value: tuple[float, float] | None) -> None:
        if value is None:
            return
        if type(value) is not tuple or len(value) != 2:
            raise ContractValueError("start_frame_range", "must be an immutable pair")
        if any(type(bound) not in (int, float) for bound in value):
            raise ContractValueError("start_frame_range", "bounds must be numeric and not boolean")
        start, end = value
        if not 0.0 <= start < end <= 1.0:
            raise ContractValueError("start_frame_range", "must be a normalized half-open range within [0, 1]")


@dataclass(frozen=True, slots=True)
class CurriculumSchedule:
    version: str
    total_steps: int; stages: tuple[CurriculumStage, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.version, "schedule version")
        _require_integer(self.total_steps, "total_steps", 1)
        if type(self.stages) is not tuple or not self.stages:
            raise ContractValueError("stages", "must be a nonempty immutable tuple")
        if any(type(stage) is not CurriculumStage for stage in self.stages):
            raise ContractValueError("stages", "must contain curriculum stages")
        expected_start = 0
        names: set[str] = set()
        for stage in self.stages:
            if stage.name in names:
                raise ContractValueError("stage names", "must be unique")
            if stage.start_step != expected_start:
                raise ContractValueError("stages", "must be ordered, nonoverlapping, and fully covered")
            names.add(stage.name)
            expected_start = stage.end_step
        if expected_start != self.total_steps:
            raise ContractValueError("stages", "must cover exactly [0, total_steps)")

    def active_stage(self, global_step: int, total_steps: int) -> CurriculumStage:
        _require_integer(global_step, "global_step", 0)
        _require_integer(total_steps, "total_steps", 1)
        if total_steps != self.total_steps:
            raise ContractValueError("total_steps", "must match the schedule binding")
        if global_step >= total_steps:
            raise ContractValueError("global_step", "must be less than total_steps")
        for stage in self.stages:
            if stage.start_step <= global_step < stage.end_step:
                return stage
        raise ContractValueError("global_step", "is not covered by the schedule")

    @property
    def identity(self) -> str:
        return f"curriculum-schedule-v1:{self.version}:{self.total_steps}"


@dataclass(frozen=True, slots=True)
class CurriculumCandidate:
    candidate_id: str
    episode: EpisodeRecord; shot: ShotRecord
    start_frame: int; normalized_start_frame: float
    request: TemporalWindowRequest


@dataclass(frozen=True, slots=True)
class CurriculumCandidateView:
    active_stage: CurriculumStage
    candidates: tuple[CurriculumCandidate, ...]


CurriculumStatePayload = TypedDict("CurriculumStatePayload", {
    "global_step": int, "total_steps": int, "schedule_version": str,
    "schedule_identity": str, "catalog_identity": str, "sampler_seed": int,
    "active_stage_name": str,
})


@dataclass(frozen=True, slots=True)
class CurriculumState:
    global_step: int; total_steps: int
    schedule_version: str; schedule_identity: str; catalog_identity: str
    sampler_seed: int
    active_stage_name: str

    def __post_init__(self) -> None:
        _require_integer(self.global_step, "global_step", 0)
        _require_integer(self.total_steps, "total_steps", 1)
        _require_nonempty_string(self.schedule_version, "schedule_version")
        _require_nonempty_string(self.schedule_identity, "schedule_identity")
        _require_nonempty_string(self.catalog_identity, "catalog_identity")
        if type(self.sampler_seed) is not int:
            raise ContractValueError("sampler_seed", "must be an integer")
        _require_nonempty_string(self.active_stage_name, "active_stage_name")

    def to_dict(self) -> CurriculumStatePayload:
        return CurriculumStatePayload(global_step=self.global_step, total_steps=self.total_steps,
                                      schedule_version=self.schedule_version,
                                      schedule_identity=self.schedule_identity,
                                      catalog_identity=self.catalog_identity, sampler_seed=self.sampler_seed,
                                      active_stage_name=self.active_stage_name)


@dataclass(frozen=True, slots=True)
class CurriculumBindingMismatchError(ValueError):
    binding: str

    def __str__(self) -> str:
        return f"curriculum checkpoint binding mismatch: {self.binding}"


class CurriculumPolicy:
    def __init__(
        self,
        catalog: EpisodeCatalog,
        schedule: CurriculumSchedule,
        *,
        sampler_seed: int,
    ) -> None:
        if type(sampler_seed) is not int:
            raise ContractValueError("sampler_seed", "must be an integer")
        self._catalog = catalog
        self._schedule = schedule
        self._sampler_seed = sampler_seed
        self._catalog_identity = catalog_identity(catalog)

    @property
    def catalog_identity(self) -> str:
        return self._catalog_identity

    def candidate_view(self, global_step: int, total_steps: int) -> CurriculumCandidateView:
        stage = self._schedule.active_stage(global_step, total_steps)
        candidates = [
            candidate
            for episode in self._catalog.episodes
            if _episode_matches(episode, stage)
            for shot in episode.shots
            for request in stage.temporal_choices
            for candidate in _shot_candidates(episode, shot, request, stage)
        ]
        candidates.sort(key=lambda candidate: _candidate_order_key(self._sampler_seed, candidate))
        return CurriculumCandidateView(stage, tuple(candidates))

    def state(self, global_step: int, total_steps: int) -> CurriculumState:
        active_stage = self._schedule.active_stage(global_step, total_steps)
        return CurriculumState(global_step, total_steps, self._schedule.version,
                               self._schedule.identity, self._catalog_identity,
                               self._sampler_seed, active_stage.name)

    def validate_resume(self, state: CurriculumState) -> None:
        active_stage = self._schedule.active_stage(state.global_step, state.total_steps)
        bindings = (
            ("schedule_version", state.schedule_version, self._schedule.version),
            ("sampler_seed", state.sampler_seed, self._sampler_seed),
            ("active_stage_name", state.active_stage_name, active_stage.name),
        )
        for binding, recorded, active in bindings:
            if recorded != active:
                raise CurriculumBindingMismatchError(binding)


def _episode_matches(episode: EpisodeRecord, stage: CurriculumStage) -> bool:
    if stage.novelty_levels is None and stage.scenario_types is None:
        return True
    if episode.source_level_key is None:
        return False
    source_parts = episode.source_level_key.split("/")
    novelty_level = next((part for part in source_parts if part.startswith("novelty_level_")), None)
    scenario_type = next((part for part in source_parts if part.startswith("type")), None)
    if stage.novelty_levels is not None and novelty_level not in stage.novelty_levels:
        return False
    return stage.scenario_types is None or scenario_type in stage.scenario_types


def _shot_candidates(
    episode: EpisodeRecord,
    shot: ShotRecord,
    request: TemporalWindowRequest,
    stage: CurriculumStage,
) -> tuple[CurriculumCandidate, ...]:
    frame_count = len(shot.frames)
    candidates: list[CurriculumCandidate] = []
    for start_frame in range(frame_count - request.horizon_frames):
        normalized_start = _normalized_start_frame(start_frame, frame_count)
        if stage.start_frame_range is not None:
            range_start, range_end = stage.start_frame_range
            if not range_start <= normalized_start < range_end:
                continue
        candidate_id = ":".join((str(episode.split), episode.relative_path, shot.relative_path,
                                 str(start_frame), str(request.prediction_steps), str(request.stride_frames)))
        candidates.append(CurriculumCandidate(candidate_id, episode, shot, start_frame,
                                              normalized_start, request))
    return tuple(candidates)


def _normalized_start_frame(start_frame: int, frame_count: int) -> float:
    return 0.0 if frame_count == 1 else start_frame / (frame_count - 1)


def _candidate_order_key(seed: int, candidate: CurriculumCandidate) -> tuple[int, str]:
    # deterministic non-cryptographic derivation, not an integrity check
    order = zlib.crc32(f"{seed}:{candidate.candidate_id}".encode("utf-8"))
    return order, candidate.candidate_id


def catalog_identity(catalog: EpisodeCatalog) -> str:
    """Return the catalog's plain declared provenance identity."""
    contract = catalog.capture_contract
    declared_fields = (
        catalog.cohort_identity,
        catalog.collection_plan_identity,
        catalog.split,
        contract.contract_name,
        contract.contract_version,
        contract.artifact_layout_version,
    )
    encoded = json.dumps(
        declared_fields, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )
    return f"episode-catalog-v1:{encoded}"
