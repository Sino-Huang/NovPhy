from __future__ import annotations

import json
import zlib
from collections import Counter
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Final, Literal, TypeAlias, TypedDict, cast

from world_model.data.curriculum import (
    CurriculumCandidate,
    CurriculumPolicy,
    CurriculumState,
)
from world_model.data.types import ContractValueError, TemporalWindowRequest

PresetName: TypeAlias = Literal[
    "fixed_short", "fixed_long", "temporal_uniform", "temporal_curriculum"
]
JsonValue: TypeAlias = str | int | float | bool | None | tuple["JsonValue", ...]
Distribution: TypeAlias = tuple[tuple[int, int], ...]

_PRESET_DEFINITIONS: Final = (
    ("fixed_short", ((1, 1),)),
    ("fixed_long", ((4, 2),)),
    ("temporal_uniform", ((1, 1), (2, 1), (4, 2))),
    ("temporal_curriculum", ((1, 1), (2, 1), (4, 2))),
)


def _require_string(value: str, field: str) -> None:
    if type(value) is not str or not value.strip():
        raise ContractValueError(field, "must be a nonempty string")


def _require_integer(value: int, field: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ContractValueError(field, f"must be an integer greater than or equal to {minimum}")


def _declared_pairs(name: str) -> tuple[tuple[int, int], ...]:
    for declared_name, pairs in _PRESET_DEFINITIONS:
        if name == declared_name:
            return pairs
    raise ContractValueError("preset name", f"unsupported temporal preset {name!r}")


def _identity(namespace: str, *fields: JsonValue) -> str:
    declared = json.dumps(fields, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    return f"{namespace}:{declared}"


def _derivation(namespace: str, *fields: object) -> int:
    payload = f"{namespace}:{json.dumps(fields, ensure_ascii=True, separators=(',', ':'))}"
    # deterministic non-cryptographic derivation, not an integrity check
    return zlib.crc32(payload.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class TemporalAblationPreset:
    name: PresetName
    temporal_choices: tuple[TemporalWindowRequest, ...]

    def __post_init__(self) -> None:
        _require_string(self.name, "preset name")
        if type(self.temporal_choices) is not tuple or not self.temporal_choices:
            raise ContractValueError("temporal_choices", "must be a nonempty immutable tuple")
        if any(type(choice) is not TemporalWindowRequest for choice in self.temporal_choices):
            raise ContractValueError("temporal_choices", "must contain temporal window requests")
        if len(set(self.temporal_choices)) != len(self.temporal_choices):
            raise ContractValueError("temporal_choices", "must contain unique values")
        observed = tuple((choice.prediction_steps, choice.stride_frames)
                         for choice in self.temporal_choices)
        if observed != _declared_pairs(self.name):
            raise ContractValueError("temporal_choices", "must equal the named preset declaration")

    @property
    def identity(self) -> str:
        pairs = tuple((choice.prediction_steps, choice.stride_frames)
                      for choice in self.temporal_choices)
        return _identity("temporal-ablation-preset-v1", self.name, pairs)


@dataclass(frozen=True, slots=True)
class WindowCostRule:
    name: str
    base_window_cost: int
    predicted_frame_cost: int

    def __post_init__(self) -> None:
        _require_string(self.name, "cost rule name")
        _require_integer(self.base_window_cost, "base_window_cost", 0)
        _require_integer(self.predicted_frame_cost, "predicted_frame_cost", 1)

    @property
    def identity(self) -> str:
        return _identity("window-cost-rule-v1", self.name, self.base_window_cost,
                         self.predicted_frame_cost)


@dataclass(frozen=True, slots=True)
class AblationRunConfig:
    preset: TemporalAblationPreset
    sampling_seed: int
    draw_count: int
    cost_rule: WindowCostRule

    def __post_init__(self) -> None:
        if type(self.preset) is not TemporalAblationPreset:
            raise ContractValueError("preset", "must be a temporal ablation preset")
        if type(self.sampling_seed) is not int:
            raise ContractValueError("sampling_seed", "must be an integer")
        _require_integer(self.draw_count, "draw_count", 1)
        if type(self.cost_rule) is not WindowCostRule:
            raise ContractValueError("cost_rule", "must be a window cost rule")


class TemporalAblationManifestPayload(TypedDict):
    preset_name: str
    preset_identity: str
    catalog_identity: str
    schedule_version: str
    schedule_identity: str
    active_stage_name: str
    sampling_seed: int
    sampled_provenance: list[list[str | None]]
    draw_count: int
    cost_rule_identity: str
    prediction_steps_distribution: Mapping[str, int]
    stride_frames_distribution: Mapping[str, int]
    horizon_frames_distribution: Mapping[str, int]
    total_prediction_steps: int
    effective_prediction_steps: float
    total_base_window_cost: int
    total_predicted_frame_cost: int
    temporal_controller_cost: int
    computed_budget_total: int


@dataclass(frozen=True, slots=True)
class TemporalAblationManifest:
    preset_name: PresetName; preset_identity: str
    catalog_identity: str; schedule_version: str; schedule_identity: str
    active_stage_name: str; sampling_seed: int
    sampled_provenance: tuple[tuple[str, str | None], ...]
    draw_count: int; cost_rule_identity: str
    prediction_steps_distribution: Distribution; stride_frames_distribution: Distribution
    horizon_frames_distribution: Distribution; total_prediction_steps: int
    effective_prediction_steps: float; total_base_window_cost: int
    total_predicted_frame_cost: int; temporal_controller_cost: int
    computed_budget_total: int

    def to_dict(self) -> TemporalAblationManifestPayload:
        return TemporalAblationManifestPayload(
            preset_name=self.preset_name, preset_identity=self.preset_identity,
            catalog_identity=self.catalog_identity, schedule_version=self.schedule_version,
            schedule_identity=self.schedule_identity, active_stage_name=self.active_stage_name,
            sampling_seed=self.sampling_seed,
            sampled_provenance=[list(item) for item in self.sampled_provenance],
            draw_count=self.draw_count, cost_rule_identity=self.cost_rule_identity,
            prediction_steps_distribution=_distribution_payload(self.prediction_steps_distribution),
            stride_frames_distribution=_distribution_payload(self.stride_frames_distribution),
            horizon_frames_distribution=_distribution_payload(self.horizon_frames_distribution),
            total_prediction_steps=self.total_prediction_steps,
            effective_prediction_steps=self.effective_prediction_steps,
            total_base_window_cost=self.total_base_window_cost,
            total_predicted_frame_cost=self.total_predicted_frame_cost,
            temporal_controller_cost=self.temporal_controller_cost,
            computed_budget_total=self.computed_budget_total,
        )


TemporalAblationComparisonPayload = TypedDict("TemporalAblationComparisonPayload", {
    "sample_match": str, "compute_match": str, "left_computed_budget_total": int,
    "right_computed_budget_total": int,
})


@dataclass(frozen=True, slots=True)
class TemporalAblationComparison:
    sample_match: Literal["sample_matched", "sample_unmatched"]
    compute_match: Literal["compute_matched", "compute_unmatched"]
    left_computed_budget_total: int
    right_computed_budget_total: int

    def to_dict(self) -> TemporalAblationComparisonPayload:
        return TemporalAblationComparisonPayload(
            sample_match=self.sample_match, compute_match=self.compute_match,
            left_computed_budget_total=self.left_computed_budget_total,
            right_computed_budget_total=self.right_computed_budget_total,
        )


@dataclass(frozen=True, slots=True)
class ComputeBudgetMismatchError(ValueError):
    left_total: int
    right_total: int

    def __str__(self) -> str:
        return f"required compute match differs: {self.left_total} != {self.right_total}"


def get_temporal_ablation_preset(name: str) -> TemporalAblationPreset:
    pairs = _declared_pairs(name)
    return TemporalAblationPreset(
        cast(PresetName, name),
        tuple(TemporalWindowRequest(*pair) for pair in pairs),
    )


def build_temporal_ablation_manifest(
    policy: CurriculumPolicy,
    state: CurriculumState,
    config: AblationRunConfig,
) -> TemporalAblationManifest:
    if type(policy) is not CurriculumPolicy or type(state) is not CurriculumState:
        raise ContractValueError("curriculum binding", "must contain a policy and state")
    policy.validate_resume(state)
    view = policy.candidate_view(state.global_step, state.total_steps)
    allowed = frozenset(config.preset.temporal_choices)
    candidates = tuple(candidate for candidate in view.candidates if candidate.request in allowed)
    if not candidates:
        raise ContractValueError("ablation candidates", "preset has no eligible catalog candidates")
    selected = _select_candidates(candidates, config)
    prediction_steps = tuple(candidate.request.prediction_steps for candidate in selected)
    strides = tuple(candidate.request.stride_frames for candidate in selected)
    horizons = tuple(candidate.request.horizon_frames for candidate in selected)
    total_steps = sum(prediction_steps)
    predicted_cost = total_steps * config.cost_rule.predicted_frame_cost
    base_cost = config.draw_count * config.cost_rule.base_window_cost
    provenance = tuple((candidate.candidate_id, candidate.episode.source_level_key)
                       for candidate in selected)
    return TemporalAblationManifest(
        config.preset.name, config.preset.identity, state.catalog_identity,
        state.schedule_version, state.schedule_identity, state.active_stage_name,
        config.sampling_seed, provenance,
        config.draw_count, config.cost_rule.identity, _distribution(prediction_steps),
        _distribution(strides), _distribution(horizons), total_steps,
        total_steps / config.draw_count, base_cost, predicted_cost, 0,
        base_cost + predicted_cost,
    )


def compare_temporal_ablation_manifests(
    left: TemporalAblationManifest,
    right: TemporalAblationManifest,
    *,
    require_compute_match: bool = False,
) -> TemporalAblationComparison:
    if type(require_compute_match) is not bool:
        raise ContractValueError("require_compute_match", "must be boolean")
    sample_matched = (left.draw_count == right.draw_count
                      and left.cost_rule_identity == right.cost_rule_identity)
    compute_matched = left.computed_budget_total == right.computed_budget_total
    if require_compute_match and not compute_matched:
        raise ComputeBudgetMismatchError(left.computed_budget_total,
                                         right.computed_budget_total)
    return TemporalAblationComparison(
        "sample_matched" if sample_matched else "sample_unmatched",
        "compute_matched" if compute_matched else "compute_unmatched",
        left.computed_budget_total, right.computed_budget_total,
    )


def _select_candidates(
    candidates: tuple[CurriculumCandidate, ...],
    config: AblationRunConfig,
) -> tuple[CurriculumCandidate, ...]:
    if config.preset.name == "temporal_uniform":
        groups = tuple(tuple(candidate for candidate in candidates if candidate.request == choice)
                       for choice in config.preset.temporal_choices)
        groups = tuple(group for group in groups if group)
        return tuple(group[_stable_index(config.sampling_seed, draw, len(group), "candidate")]
                     for draw in range(config.draw_count)
                     for group in (groups[_stable_index(config.sampling_seed, draw,
                                                        len(groups), "choice")],))
    ranked = tuple(sorted(candidates, key=lambda candidate:
                          (_derivation("ablation-rank-v1", config.sampling_seed,
                                       candidate.candidate_id), candidate.candidate_id)))
    if config.draw_count <= len(ranked):
        return ranked[:config.draw_count]
    return tuple(ranked[_stable_index(config.sampling_seed, draw, len(ranked), "candidate")]
                 for draw in range(config.draw_count))


def _stable_index(seed: int, draw: int, size: int, namespace: str) -> int:
    return _derivation("ablation-draw-v1", namespace, seed, draw) % size


def _distribution(values: tuple[int, ...]) -> Distribution:
    return tuple(sorted(Counter(values).items()))


def _distribution_payload(distribution: Distribution) -> Mapping[str, int]:
    return {str(value): count for value, count in distribution}
