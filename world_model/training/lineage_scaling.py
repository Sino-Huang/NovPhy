"""Lineage-scaled continuous-predictor training and matched evaluation.

The contracts in this module keep the two experimental factors independent:
complete training-lineage coverage and the carrier used to represent each state.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
import json
import math
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any

import torch
from torch.nn import functional as F

from world_model.data.deployment_temporal import (
    TrajectoryLineageBinding,
    TrajectoryLineageManifest,
)
from world_model.model import (
    Abstraction,
    DualOutputPredictor,
    PredictionPair,
    PredictorConfig,
    identity,
)
from world_model.planning import (
    CEMConfig,
    CEMPlanner,
    ControlConfig,
    ControlMode,
    ContinuousCheckpointWorldModel,
    GameplayCost,
    GameplayCostConfig,
    SlingshotAction,
    SlingshotActionBounds,
    WorldModelCandidateEvaluator,
)
from world_model.planning.gameplay import PlanningObservation
from world_model.training.cohort_v2_micro import cohort_v2_model_state_identity


class LineageScalingError(ValueError):
    """The lineage-scaling protocol, data, or checkpoint is inconsistent."""


class CarrierKind(StrEnum):
    SOURCE = "source"
    DEPLOYMENT = "deployment"


@dataclass(frozen=True, slots=True)
class FrozenLineageScale:
    """One prospectively frozen subset of complete training lineages."""

    name: str
    lineage_manifest_identity: str
    source_release_identity: str
    lineage_identities: tuple[str, ...]
    bindings: tuple[TrajectoryLineageBinding, ...]

    @classmethod
    def from_manifest(
        cls,
        name: str,
        manifest: TrajectoryLineageManifest,
    ) -> "FrozenLineageScale":
        if not isinstance(manifest, TrajectoryLineageManifest):
            raise LineageScalingError("lineage scale requires a frozen lineage manifest")
        roles = {binding.exposure_role for binding in manifest.bindings}
        if roles != {"training"}:
            raise LineageScalingError("training scale manifest crossed its exposure role")
        return cls(
            name=name,
            lineage_manifest_identity=manifest.identity,
            source_release_identity=manifest.source_release_identity,
            lineage_identities=tuple(
                binding.scenario_lineage_identity for binding in manifest.bindings
            ),
            bindings=manifest.bindings,
        )

    def __post_init__(self) -> None:
        if not self.name or not self.lineage_manifest_identity or not self.source_release_identity:
            raise LineageScalingError("lineage scale identity is incomplete")
        if (
            type(self.lineage_identities) is not tuple
            or not self.lineage_identities
            or any(not isinstance(value, str) or not value for value in self.lineage_identities)
            or len(set(self.lineage_identities)) != len(self.lineage_identities)
        ):
            raise LineageScalingError("lineage scale membership is malformed")
        if (
            type(self.bindings) is not tuple
            or tuple(item.scenario_lineage_identity for item in self.bindings)
            != self.lineage_identities
            or TrajectoryLineageManifest.create(
                self.source_release_identity, self.bindings
            ).identity
            != self.lineage_manifest_identity
        ):
            raise LineageScalingError("lineage scale manifest binding is stale")


@dataclass(frozen=True, slots=True)
class TrainingCell:
    scale_name: str
    carrier: CarrierKind
    seed: int

    def __post_init__(self) -> None:
        if not self.scale_name or type(self.seed) is not int or self.seed < 0:
            raise LineageScalingError("training cell identity is invalid")
        try:
            object.__setattr__(self, "carrier", CarrierKind(self.carrier))
        except ValueError as error:
            raise LineageScalingError("training cell carrier is unsupported") from error

    @property
    def identity(self) -> str:
        return identity((
            "lineage-scaling-training-cell-v1",
            self.scale_name,
            self.carrier.value,
            self.seed,
        ))


@dataclass(frozen=True, slots=True)
class LineageScalingProtocol:
    """The frozen factors shared by every matched primary training cell."""

    training_scales: tuple[FrozenLineageScale, ...]
    evaluation_manifests: tuple[TrajectoryLineageManifest, ...]
    training_seeds: tuple[int, ...]
    optimizer_example_budget: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    predictor_config: PredictorConfig
    source_carrier_identity: str
    deployment_carrier_identity: str
    configuration_basis: str

    def __post_init__(self) -> None:
        if (
            type(self.training_scales) is not tuple
            or len(self.training_scales) < 2
            or any(type(item) is not FrozenLineageScale for item in self.training_scales)
        ):
            raise LineageScalingError("protocol requires at least two frozen training scales")
        if self.training_scales[0].name != "six" or len(
            self.training_scales[0].lineage_identities
        ) != 6:
            raise LineageScalingError("the first training scale must be the exact six-lineage subset")
        if len({item.name for item in self.training_scales}) != len(self.training_scales):
            raise LineageScalingError("training scale names must be unique")
        releases = {item.source_release_identity for item in self.training_scales}
        if len(releases) != 1:
            raise LineageScalingError("training scales target different source releases")
        for smaller, larger in zip(
            self.training_scales, self.training_scales[1:], strict=False
        ):
            smaller_members = set(smaller.lineage_identities)
            larger_members = set(larger.lineage_identities)
            if not smaller_members < larger_members:
                raise LineageScalingError(
                    "training lineage scales must be strictly nested complete subsets"
                )

        if (
            type(self.evaluation_manifests) is not tuple
            or len(self.evaluation_manifests) != 2
            or any(
                not isinstance(item, TrajectoryLineageManifest)
                for item in self.evaluation_manifests
            )
        ):
            raise LineageScalingError(
                "protocol requires calibration and model-selection lineage manifests"
            )
        evaluation_roles = tuple(
            {binding.exposure_role for binding in manifest.bindings}
            for manifest in self.evaluation_manifests
        )
        if evaluation_roles != ({"calibration"}, {"model_selection"}):
            raise LineageScalingError(
                "evaluation manifests must be ordered calibration then model_selection"
            )
        if any(
            manifest.source_release_identity not in releases
            for manifest in self.evaluation_manifests
        ):
            raise LineageScalingError("evaluation manifests target another source release")
        training_lineages = set(self.training_scales[-1].lineage_identities)
        evaluation_lineages: set[str] = set()
        for manifest in self.evaluation_manifests:
            for binding in manifest.bindings:
                if binding.scenario_lineage_identity in training_lineages:
                    raise LineageScalingError("scenario lineage leaked across exposure roles")
                if binding.scenario_lineage_identity in evaluation_lineages:
                    raise LineageScalingError("scenario lineage leaked across evaluation roles")
                evaluation_lineages.add(binding.scenario_lineage_identity)

        if (
            type(self.training_seeds) is not tuple
            or len(self.training_seeds) < 3
            or len(set(self.training_seeds)) != len(self.training_seeds)
            or any(type(seed) is not int or seed < 0 for seed in self.training_seeds)
        ):
            raise LineageScalingError(
                "protocol requires at least three unique frozen training seeds"
            )
        if (
            type(self.optimizer_example_budget) is not int
            or self.optimizer_example_budget <= 0
            or type(self.batch_size) is not int
            or self.batch_size <= 0
            or type(self.predictor_config) is not PredictorConfig
            or self.predictor_config.action_dim != 5
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or self.grad_clip < 0.0
        ):
            raise LineageScalingError("training architecture or optimizer budget is invalid")
        if (
            not self.source_carrier_identity
            or not self.deployment_carrier_identity
            or self.source_carrier_identity == self.deployment_carrier_identity
        ):
            raise LineageScalingError("source and deployment carrier identities must differ")
        if self.configuration_basis != "prospectively_frozen":
            raise LineageScalingError(
                "training configuration must be prospectively frozen, not outcome conditioned"
            )

    @property
    def identity(self) -> str:
        return identity((
            "lineage-scaling-protocol-v1",
            tuple(
                (
                    item.name,
                    item.lineage_manifest_identity,
                    item.lineage_identities,
                )
                for item in self.training_scales
            ),
            tuple(item.identity for item in self.evaluation_manifests),
            self.training_seeds,
            self.optimizer_example_budget,
            self.batch_size,
            self.learning_rate,
            self.weight_decay,
            self.grad_clip,
            self.predictor_config.identity,
            self.source_carrier_identity,
            self.deployment_carrier_identity,
            self.configuration_basis,
        ))

    @property
    def cells(self) -> tuple[TrainingCell, ...]:
        return tuple(
            TrainingCell(scale.name, carrier, seed)
            for scale in self.training_scales
            for carrier in (CarrierKind.SOURCE, CarrierKind.DEPLOYMENT)
            for seed in self.training_seeds
        )

    @property
    def primary_cells(self) -> tuple[TrainingCell, ...]:
        names = (self.training_scales[0].name, self.training_scales[-1].name)
        return tuple(cell for cell in self.cells if cell.scale_name in names)

    def scale(self, name: str) -> FrozenLineageScale:
        try:
            return next(item for item in self.training_scales if item.name == name)
        except StopIteration as error:
            raise LineageScalingError(f"unknown training scale {name!r}") from error

    def carrier_identity(self, carrier: CarrierKind) -> str:
        return (
            self.source_carrier_identity
            if CarrierKind(carrier) is CarrierKind.SOURCE
            else self.deployment_carrier_identity
        )


def _binding_payload(binding: TrajectoryLineageBinding) -> dict[str, Any]:
    return {
        "trajectory_identity": binding.trajectory_identity,
        "scenario_lineage_identity": binding.scenario_lineage_identity,
        "exposure_role": binding.exposure_role,
        "transition_identities": list(binding.transition_identities),
        "initial_observation_identity": binding.initial_observation_identity,
        "terminal_observation_identity": binding.terminal_observation_identity,
    }


def _manifest_payload(manifest: TrajectoryLineageManifest) -> dict[str, Any]:
    return {
        "schema": manifest.schema,
        "identity": manifest.identity,
        "source_release_identity": manifest.source_release_identity,
        "bindings": [_binding_payload(item) for item in manifest.bindings],
    }


def _manifest_from_payload(value: Any) -> TrajectoryLineageManifest:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "identity",
        "source_release_identity",
        "bindings",
    }:
        raise LineageScalingError("lineage manifest payload fields differ")
    raw_bindings = value["bindings"]
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise LineageScalingError("lineage manifest bindings are missing")
    bindings = []
    binding_fields = {
        "trajectory_identity",
        "scenario_lineage_identity",
        "exposure_role",
        "transition_identities",
        "initial_observation_identity",
        "terminal_observation_identity",
    }
    for raw in raw_bindings:
        if not isinstance(raw, Mapping) or set(raw) != binding_fields:
            raise LineageScalingError("lineage manifest binding fields differ")
        transitions = raw["transition_identities"]
        if not isinstance(transitions, list):
            raise LineageScalingError("lineage transition identities are malformed")
        bindings.append(TrajectoryLineageBinding(
            trajectory_identity=raw["trajectory_identity"],
            scenario_lineage_identity=raw["scenario_lineage_identity"],
            exposure_role=raw["exposure_role"],
            transition_identities=tuple(transitions),
            initial_observation_identity=raw["initial_observation_identity"],
            terminal_observation_identity=raw["terminal_observation_identity"],
        ))
    manifest = TrajectoryLineageManifest.create(
        value["source_release_identity"], tuple(bindings)
    )
    if (
        value["schema"] != manifest.schema
        or value["identity"] != manifest.identity
    ):
        raise LineageScalingError("lineage manifest identity is stale")
    return manifest


def _protocol_payload(protocol: LineageScalingProtocol) -> dict[str, Any]:
    return {
        "schema": "lineage_scaling_protocol_v1",
        "identity": protocol.identity,
        "training_scales": [
            {
                "name": scale.name,
                "manifest": _manifest_payload(TrajectoryLineageManifest(
                    scale.lineage_manifest_identity,
                    scale.source_release_identity,
                    scale.bindings,
                )),
            }
            for scale in protocol.training_scales
        ],
        "evaluation_manifests": [
            _manifest_payload(item) for item in protocol.evaluation_manifests
        ],
        "training_seeds": list(protocol.training_seeds),
        "optimizer": {
            "example_budget": protocol.optimizer_example_budget,
            "batch_size": protocol.batch_size,
            "learning_rate": protocol.learning_rate,
            "weight_decay": protocol.weight_decay,
            "grad_clip": protocol.grad_clip,
        },
        "predictor_config": asdict(protocol.predictor_config),
        "carrier_identities": {
            "source": protocol.source_carrier_identity,
            "deployment": protocol.deployment_carrier_identity,
        },
        "configuration_basis": protocol.configuration_basis,
    }


def save_lineage_scaling_protocol(
    path: Path,
    protocol: LineageScalingProtocol,
) -> None:
    target = Path(path)
    if target.exists():
        raise LineageScalingError(f"protocol already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_protocol_payload(protocol), allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def load_lineage_scaling_protocol(path: Path) -> LineageScalingProtocol:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema",
            "identity",
            "training_scales",
            "evaluation_manifests",
            "training_seeds",
            "optimizer",
            "predictor_config",
            "carrier_identities",
            "configuration_basis",
        }:
            raise LineageScalingError("protocol payload fields differ")
        if payload["schema"] != "lineage_scaling_protocol_v1":
            raise LineageScalingError("protocol schema is unsupported")
        raw_scales = payload["training_scales"]
        if not isinstance(raw_scales, list):
            raise LineageScalingError("protocol training scales are malformed")
        scales = []
        for raw in raw_scales:
            if not isinstance(raw, Mapping) or set(raw) != {"name", "manifest"}:
                raise LineageScalingError("protocol training scale fields differ")
            scales.append(FrozenLineageScale.from_manifest(
                raw["name"], _manifest_from_payload(raw["manifest"])
            ))
        raw_evaluation = payload["evaluation_manifests"]
        if not isinstance(raw_evaluation, list):
            raise LineageScalingError("protocol evaluation manifests are malformed")
        optimizer = payload["optimizer"]
        carriers = payload["carrier_identities"]
        if not isinstance(optimizer, Mapping) or set(optimizer) != {
            "example_budget", "batch_size", "learning_rate", "weight_decay", "grad_clip"
        }:
            raise LineageScalingError("protocol optimizer fields differ")
        if not isinstance(carriers, Mapping) or set(carriers) != {"source", "deployment"}:
            raise LineageScalingError("protocol carrier fields differ")
        protocol = LineageScalingProtocol(
            training_scales=tuple(scales),
            evaluation_manifests=tuple(
                _manifest_from_payload(item) for item in raw_evaluation
            ),
            training_seeds=tuple(payload["training_seeds"]),
            optimizer_example_budget=optimizer["example_budget"],
            batch_size=optimizer["batch_size"],
            learning_rate=optimizer["learning_rate"],
            weight_decay=optimizer["weight_decay"],
            grad_clip=optimizer["grad_clip"],
            predictor_config=PredictorConfig(**payload["predictor_config"]),
            source_carrier_identity=carriers["source"],
            deployment_carrier_identity=carriers["deployment"],
            configuration_basis=payload["configuration_basis"],
        )
        if payload["identity"] != protocol.identity:
            raise LineageScalingError("protocol identity is stale")
        return protocol
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, LineageScalingError):
            raise
        raise LineageScalingError(f"lineage-scaling protocol is invalid: {error}") from error


def _require_vector(value: torch.Tensor, size: int, field: str) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or value.shape[0] != size
        or not bool(torch.isfinite(value).all())
    ):
        raise LineageScalingError(f"{field} must be a finite vector of width {size}")


@dataclass(frozen=True, slots=True)
class ContinuousTransitionExample:
    identity: str
    context: torch.Tensor
    action: torch.Tensor
    target: torch.Tensor
    physical_diagnostics: Mapping[str, float | bool | None]
    decision_index: int = 0
    horizon: int = 1

    def __post_init__(self) -> None:
        if not self.identity:
            raise LineageScalingError("continuous transition identity is missing")
        if (
            type(self.decision_index) is not int
            or self.decision_index < 0
            or type(self.horizon) is not int
            or self.horizon <= 0
        ):
            raise LineageScalingError("continuous transition position or horizon is invalid")
        if not isinstance(self.context, torch.Tensor) or self.context.ndim != 1:
            raise LineageScalingError("continuous transition context is malformed")
        _require_vector(self.target, self.context.shape[0], "continuous target")
        _require_vector(self.action, 5, "continuous action")
        if not isinstance(self.physical_diagnostics, Mapping):
            raise LineageScalingError("physical diagnostics are malformed")
        object.__setattr__(
            self,
            "physical_diagnostics",
            MappingProxyType(dict(self.physical_diagnostics)),
        )


@dataclass(frozen=True, slots=True)
class CarrierLineage:
    """One complete trajectory encoded under exactly one carrier factor."""

    trajectory_identity: str
    scenario_lineage_identity: str
    exposure_role: str
    source_release_identity: str
    carrier: CarrierKind
    carrier_identity: str
    transitions: tuple[ContinuousTransitionExample, ...]
    complete: bool

    def __post_init__(self) -> None:
        if (
            not self.trajectory_identity
            or not self.scenario_lineage_identity
            or not self.source_release_identity
            or not self.carrier_identity
        ):
            raise LineageScalingError("carrier lineage identity is incomplete")
        try:
            object.__setattr__(self, "carrier", CarrierKind(self.carrier))
        except ValueError as error:
            raise LineageScalingError("carrier lineage kind is unsupported") from error
        if self.exposure_role not in ("training", "calibration", "model_selection"):
            raise LineageScalingError("carrier lineage exposure role is not public")
        if self.complete is not True:
            raise LineageScalingError("carrier training requires complete trajectories")
        if (
            type(self.transitions) is not tuple
            or not self.transitions
            or any(type(item) is not ContinuousTransitionExample for item in self.transitions)
            or len({item.identity for item in self.transitions}) != len(self.transitions)
            or len({(item.decision_index, item.horizon) for item in self.transitions})
            != len(self.transitions)
        ):
            raise LineageScalingError("carrier lineage transitions are malformed")


def save_carrier_lineage_bundle(
    path: Path,
    lineages: tuple[CarrierLineage, ...],
) -> None:
    target = Path(path)
    if target.exists():
        raise LineageScalingError(f"carrier lineage bundle already exists: {target}")
    if type(lineages) is not tuple or not lineages:
        raise LineageScalingError("carrier lineage bundle is empty")
    if len({item.scenario_lineage_identity for item in lineages}) != len(lineages):
        raise LineageScalingError("carrier lineage bundle repeats a scenario lineage")
    header = {
        (
            item.source_release_identity,
            item.exposure_role,
            item.carrier,
            item.carrier_identity,
        )
        for item in lineages
    }
    if len(header) != 1:
        raise LineageScalingError("carrier lineage bundle crossed a role, release, or carrier")
    release, role, carrier, carrier_identity = next(iter(header))
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "carrier_lineage_bundle_v1",
            "source_release_identity": release,
            "exposure_role": role,
            "carrier": carrier.value,
            "carrier_identity": carrier_identity,
            "lineages": [
                {
                    "trajectory_identity": lineage.trajectory_identity,
                    "scenario_lineage_identity": lineage.scenario_lineage_identity,
                    "complete": lineage.complete,
                    "transitions": [
                        {
                            "identity": transition.identity,
                            "context": transition.context.detach().cpu(),
                            "action": transition.action.detach().cpu(),
                            "target": transition.target.detach().cpu(),
                            "decision_index": transition.decision_index,
                            "horizon": transition.horizon,
                            "physical_diagnostics": dict(
                                transition.physical_diagnostics
                            ),
                        }
                        for transition in lineage.transitions
                    ],
                }
                for lineage in lineages
            ],
        },
        target,
    )


def load_carrier_lineage_bundle(path: Path) -> tuple[CarrierLineage, ...]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema",
            "source_release_identity",
            "exposure_role",
            "carrier",
            "carrier_identity",
            "lineages",
        }:
            raise LineageScalingError("carrier lineage bundle fields differ")
        if payload["schema"] != "carrier_lineage_bundle_v1":
            raise LineageScalingError("carrier lineage bundle schema is unsupported")
        raw_lineages = payload["lineages"]
        if not isinstance(raw_lineages, list) or not raw_lineages:
            raise LineageScalingError("carrier lineage bundle has no lineages")
        lineages = []
        for raw_lineage in raw_lineages:
            if not isinstance(raw_lineage, Mapping) or set(raw_lineage) != {
                "trajectory_identity",
                "scenario_lineage_identity",
                "complete",
                "transitions",
            }:
                raise LineageScalingError("carrier lineage fields differ")
            raw_transitions = raw_lineage["transitions"]
            if not isinstance(raw_transitions, list):
                raise LineageScalingError("carrier lineage transitions are malformed")
            transitions = []
            for raw_transition in raw_transitions:
                if not isinstance(raw_transition, Mapping) or set(raw_transition) != {
                    "identity",
                    "context",
                    "action",
                    "target",
                    "decision_index",
                    "horizon",
                    "physical_diagnostics",
                }:
                    raise LineageScalingError("continuous transition fields differ")
                transitions.append(ContinuousTransitionExample(
                    identity=raw_transition["identity"],
                    context=raw_transition["context"],
                    action=raw_transition["action"],
                    target=raw_transition["target"],
                    physical_diagnostics=raw_transition["physical_diagnostics"],
                    decision_index=raw_transition["decision_index"],
                    horizon=raw_transition["horizon"],
                ))
            lineages.append(CarrierLineage(
                trajectory_identity=raw_lineage["trajectory_identity"],
                scenario_lineage_identity=raw_lineage["scenario_lineage_identity"],
                exposure_role=payload["exposure_role"],
                source_release_identity=payload["source_release_identity"],
                carrier=CarrierKind(payload["carrier"]),
                carrier_identity=payload["carrier_identity"],
                transitions=tuple(transitions),
                complete=raw_lineage["complete"],
            ))
        return tuple(lineages)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        if isinstance(error, LineageScalingError):
            raise
        raise LineageScalingError(f"carrier lineage bundle is invalid: {error}") from error


def validate_carrier_alignment(
    scale: FrozenLineageScale,
    source_lineages: tuple[CarrierLineage, ...],
    deployment_lineages: tuple[CarrierLineage, ...],
    *,
    source_carrier_identity: str,
    deployment_carrier_identity: str,
) -> dict[str, int | str]:
    """Prove that changing the carrier does not change training membership or actions."""

    expected_lineages = set(scale.lineage_identities)
    if (
        not source_lineages
        or not deployment_lineages
        or {item.scenario_lineage_identity for item in source_lineages}
        != expected_lineages
        or {item.scenario_lineage_identity for item in deployment_lineages}
        != expected_lineages
    ):
        raise LineageScalingError(
            "carrier alignment differs from the frozen full-lineage membership"
        )
    source_by_lineage = {
        item.scenario_lineage_identity: item for item in source_lineages
    }
    deployment_by_lineage = {
        item.scenario_lineage_identity: item for item in deployment_lineages
    }
    bindings_by_lineage = {
        item.scenario_lineage_identity: item for item in scale.bindings
    }
    transition_count = 0
    for lineage_identity in scale.lineage_identities:
        source = source_by_lineage[lineage_identity]
        deployment = deployment_by_lineage[lineage_identity]
        binding = bindings_by_lineage[lineage_identity]
        if (
            source.trajectory_identity != deployment.trajectory_identity
            or source.trajectory_identity != binding.trajectory_identity
            or source.exposure_role != "training"
            or deployment.exposure_role != "training"
            or source.source_release_identity != scale.source_release_identity
            or deployment.source_release_identity != scale.source_release_identity
            or source.carrier is not CarrierKind.SOURCE
            or deployment.carrier is not CarrierKind.DEPLOYMENT
            or source.carrier_identity != source_carrier_identity
            or deployment.carrier_identity != deployment_carrier_identity
            or tuple(item.identity for item in source.transitions)
            != tuple(item.identity for item in deployment.transitions)
            or tuple(item.identity for item in source.transitions)
            != binding.transition_identities
            or len(source.transitions) != len(deployment.transitions)
        ):
            raise LineageScalingError(
                "carrier alignment changed a trajectory, transition, role, or carrier binding"
            )
        for source_transition, deployment_transition in zip(
            source.transitions, deployment.transitions, strict=True
        ):
            if (
                source_transition.action.shape != deployment_transition.action.shape
                or not torch.equal(
                    source_transition.action, deployment_transition.action
                )
                or source_transition.decision_index
                != deployment_transition.decision_index
                or source_transition.horizon != deployment_transition.horizon
            ):
                raise LineageScalingError(
                    "carrier alignment changed an executed legal action"
                )
        transition_count += len(source.transitions)
    return {
        "lineage_manifest_identity": scale.lineage_manifest_identity,
        "source_carrier_identity": source_carrier_identity,
        "deployment_carrier_identity": deployment_carrier_identity,
        "lineage_count": len(expected_lineages),
        "transition_count": transition_count,
    }


def validate_matched_carrier_lineages(
    protocol: LineageScalingProtocol,
    source_lineages: tuple[CarrierLineage, ...],
    deployment_lineages: tuple[CarrierLineage, ...],
) -> dict[str, int | str]:
    result = validate_carrier_alignment(
        protocol.training_scales[-1],
        source_lineages,
        deployment_lineages,
        source_carrier_identity=protocol.source_carrier_identity,
        deployment_carrier_identity=protocol.deployment_carrier_identity,
    )
    return {"protocol_identity": protocol.identity, **result}


def validate_evaluation_lineages(
    protocol: LineageScalingProtocol,
    lineages: tuple[CarrierLineage, ...],
    *,
    carrier: CarrierKind,
) -> TrajectoryLineageManifest:
    """Bind one scoring bundle to its exact public exposure-role manifest."""

    if not lineages or len({item.exposure_role for item in lineages}) != 1:
        raise LineageScalingError("evaluation lineages crossed or omitted their role")
    role = lineages[0].exposure_role
    try:
        manifest = next(
            item
            for item in protocol.evaluation_manifests
            if {binding.exposure_role for binding in item.bindings} == {role}
        )
    except StopIteration as error:
        raise LineageScalingError("evaluation role has no frozen lineage manifest") from error
    bindings = {
        item.scenario_lineage_identity: item for item in manifest.bindings
    }
    if {item.scenario_lineage_identity for item in lineages} != set(bindings):
        raise LineageScalingError(
            "evaluation lineages differ from the frozen role manifest"
        )
    expected_carrier_identity = protocol.carrier_identity(carrier)
    for lineage in lineages:
        binding = bindings[lineage.scenario_lineage_identity]
        if (
            lineage.trajectory_identity != binding.trajectory_identity
            or tuple(item.identity for item in lineage.transitions)
            != binding.transition_identities
            or lineage.source_release_identity != manifest.source_release_identity
            or lineage.carrier is not carrier
            or lineage.carrier_identity != expected_carrier_identity
        ):
            raise LineageScalingError(
                "evaluation trajectory, transition, release, or carrier mismatch"
            )
    return manifest


@dataclass(frozen=True, slots=True)
class TrainingReport:
    protocol_identity: str
    cell: TrainingCell
    lineage_manifest_identity: str
    source_release_identity: str
    carrier_identity: str
    predictor_config: PredictorConfig
    optimizer_examples: int
    optimizer_steps: int
    lineage_count: int
    transition_count: int
    horizon_example_counts: tuple[tuple[int, int], ...]
    epochs: float
    final_loss: float
    wall_seconds: float
    parameter_count: int

    @property
    def identity(self) -> str:
        return identity((
            "lineage-scaled-training-report-v1",
            self.protocol_identity,
            self.cell.identity,
            self.lineage_manifest_identity,
            self.source_release_identity,
            self.carrier_identity,
            self.predictor_config.identity,
            self.optimizer_examples,
            self.optimizer_steps,
            self.lineage_count,
            self.transition_count,
            self.horizon_example_counts,
            self.epochs,
            self.final_loss,
            self.parameter_count,
        ))


def _validate_training_inputs(
    protocol: LineageScalingProtocol,
    cell: TrainingCell,
    lineages: tuple[CarrierLineage, ...],
) -> tuple[FrozenLineageScale, tuple[ContinuousTransitionExample, ...]]:
    if type(protocol) is not LineageScalingProtocol or cell not in protocol.cells:
        raise LineageScalingError("training cell is not part of the frozen protocol")
    if type(lineages) is not tuple or not lineages:
        raise LineageScalingError("training lineages are empty")
    scale = protocol.scale(cell.scale_name)
    if (
        len({item.scenario_lineage_identity for item in lineages}) != len(lineages)
        or {item.scenario_lineage_identity for item in lineages}
        != set(scale.lineage_identities)
    ):
        raise LineageScalingError(
            "training inputs differ from the frozen complete-lineage membership"
        )
    expected_carrier = protocol.carrier_identity(cell.carrier)
    if any(
        item.exposure_role != "training"
        or item.source_release_identity != scale.source_release_identity
        or item.carrier is not cell.carrier
        or item.carrier_identity != expected_carrier
        for item in lineages
    ):
        raise LineageScalingError("training input role, release, or carrier mismatch")
    bindings = {
        item.scenario_lineage_identity: item for item in scale.bindings
    }
    for lineage in lineages:
        binding = bindings[lineage.scenario_lineage_identity]
        if (
            lineage.trajectory_identity != binding.trajectory_identity
            or tuple(item.identity for item in lineage.transitions)
            != binding.transition_identities
        ):
            raise LineageScalingError(
                "training inputs differ from the frozen trajectory manifest"
            )
    examples = tuple(
        transition
        for lineage_identity in scale.lineage_identities
        for lineage in lineages
        if lineage.scenario_lineage_identity == lineage_identity
        for transition in lineage.transitions
    )
    for item in examples:
        _require_vector(item.context, protocol.predictor_config.latent_dim, "context")
        _require_vector(item.target, protocol.predictor_config.latent_dim, "target")
    return scale, examples


def train_continuous_predictor(
    protocol: LineageScalingProtocol,
    cell: TrainingCell,
    lineages: tuple[CarrierLineage, ...],
    *,
    device: str,
    progress: Callable[[str], None] | None = None,
) -> tuple[DualOutputPredictor, TrainingReport]:
    """Train one matched cell with an exact optimizer-example exposure budget."""

    scale, examples = _validate_training_inputs(protocol, cell, lineages)
    torch.manual_seed(cell.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cell.seed)
    target_device = torch.device(device)
    model = DualOutputPredictor(protocol.predictor_config).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=protocol.learning_rate,
        weight_decay=protocol.weight_decay,
    )
    generator = torch.Generator().manual_seed(cell.seed)
    order = torch.randperm(len(examples), generator=generator).tolist()
    cursor = 0
    examples_seen = 0
    steps = 0
    loss_value = math.nan
    started = time.monotonic()
    while examples_seen < protocol.optimizer_example_budget:
        if cursor == len(order):
            order = torch.randperm(len(examples), generator=generator).tolist()
            cursor = 0
        take = min(
            protocol.batch_size,
            protocol.optimizer_example_budget - examples_seen,
            len(order) - cursor,
        )
        batch = tuple(examples[index] for index in order[cursor:cursor + take])
        cursor += take
        contexts = torch.stack(tuple(item.context for item in batch)).to(target_device)
        actions = torch.stack(tuple(item.action for item in batch)).to(target_device)
        targets = torch.stack(tuple(item.target for item in batch)).to(target_device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=target_device)
        for horizon in sorted({item.horizon for item in batch}):
            indices = tuple(
                index for index, item in enumerate(batch) if item.horizon == horizon
            )
            selected = torch.tensor(indices, device=target_device)
            predicted = model.carrier(
                contexts[selected],
                actions[selected],
                PredictionPair(horizon, Abstraction.CONTINUOUS),
            )
            loss = loss + (
                F.mse_loss(predicted, targets[selected])
                * len(indices)
                / len(batch)
            )
        if not bool(torch.isfinite(loss)):
            raise LineageScalingError("training produced a nonfinite loss")
        loss.backward()
        if protocol.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), protocol.grad_clip)
        optimizer.step()
        examples_seen += take
        steps += 1
        loss_value = float(loss.detach().cpu())
        if progress is not None:
            progress(
                f"[train {cell.scale_name}/{cell.carrier.value}/seed-{cell.seed}] "
                f"examples {examples_seen}/{protocol.optimizer_example_budget} "
                f"loss={loss_value:.8f}"
            )
    report = TrainingReport(
        protocol_identity=protocol.identity,
        cell=cell,
        lineage_manifest_identity=scale.lineage_manifest_identity,
        source_release_identity=scale.source_release_identity,
        carrier_identity=protocol.carrier_identity(cell.carrier),
        predictor_config=protocol.predictor_config,
        optimizer_examples=examples_seen,
        optimizer_steps=steps,
        lineage_count=len(lineages),
        transition_count=len(examples),
        horizon_example_counts=tuple(
            (horizon, sum(item.horizon == horizon for item in examples))
            for horizon in sorted({item.horizon for item in examples})
        ),
        epochs=examples_seen / len(examples),
        final_loss=loss_value,
        wall_seconds=time.monotonic() - started,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )
    return model, report


@dataclass(frozen=True, slots=True)
class LineageScaledCheckpoint:
    identity: str
    model_state_identity: str
    training_report_identity: str
    protocol_identity: str
    cell: TrainingCell
    lineage_manifest_identity: str
    source_release_identity: str
    carrier_identity: str
    predictor_config: PredictorConfig
    optimizer_examples: int
    horizon_example_counts: tuple[tuple[int, int], ...]
    epochs: float


def _checkpoint_identity(
    report: TrainingReport,
    model_state_identity: str,
) -> str:
    return identity((
        "lineage-scaled-continuous-checkpoint-v1",
        report.identity,
        model_state_identity,
    ))


def save_lineage_scaled_checkpoint(
    path: Path,
    model: DualOutputPredictor,
    report: TrainingReport,
) -> LineageScaledCheckpoint:
    target = Path(path)
    if target.exists():
        raise LineageScalingError(f"checkpoint already exists: {target}")
    if not isinstance(model, DualOutputPredictor) or model.config != report.predictor_config:
        raise LineageScalingError("checkpoint model differs from its training report")
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    state_identity = cohort_v2_model_state_identity(state)
    metadata = LineageScaledCheckpoint(
        identity=_checkpoint_identity(report, state_identity),
        model_state_identity=state_identity,
        training_report_identity=report.identity,
        protocol_identity=report.protocol_identity,
        cell=report.cell,
        lineage_manifest_identity=report.lineage_manifest_identity,
        source_release_identity=report.source_release_identity,
        carrier_identity=report.carrier_identity,
        predictor_config=report.predictor_config,
        optimizer_examples=report.optimizer_examples,
        horizon_example_counts=report.horizon_example_counts,
        epochs=report.epochs,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "lineage_scaled_continuous_checkpoint_v1",
            "metadata": {
                "identity": metadata.identity,
                "model_state_identity": metadata.model_state_identity,
                "training_report_identity": metadata.training_report_identity,
                "protocol_identity": metadata.protocol_identity,
                "cell": {
                    "scale_name": metadata.cell.scale_name,
                    "carrier": metadata.cell.carrier.value,
                    "seed": metadata.cell.seed,
                },
                "lineage_manifest_identity": metadata.lineage_manifest_identity,
                "source_release_identity": metadata.source_release_identity,
                "carrier_identity": metadata.carrier_identity,
                "predictor_config": asdict(metadata.predictor_config),
                "optimizer_examples": metadata.optimizer_examples,
                "horizon_example_counts": metadata.horizon_example_counts,
                "epochs": metadata.epochs,
            },
            "model_state": state,
        },
        target,
    )
    return metadata


def load_lineage_scaled_checkpoint(
    path: Path,
    protocol: LineageScalingProtocol,
    *,
    expected_cell: TrainingCell,
    device: str,
) -> tuple[DualOutputPredictor, LineageScaledCheckpoint]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload["schema"] != "lineage_scaled_continuous_checkpoint_v1":
            raise KeyError("schema")
        raw = payload["metadata"]
        cell = TrainingCell(
            raw["cell"]["scale_name"],
            CarrierKind(raw["cell"]["carrier"]),
            raw["cell"]["seed"],
        )
        config = PredictorConfig(**raw["predictor_config"])
        state = payload["model_state"]
        state_identity = cohort_v2_model_state_identity(state)
        metadata = LineageScaledCheckpoint(
            identity=raw["identity"],
            model_state_identity=raw["model_state_identity"],
            training_report_identity=raw["training_report_identity"],
            protocol_identity=raw["protocol_identity"],
            cell=cell,
            lineage_manifest_identity=raw["lineage_manifest_identity"],
            source_release_identity=raw["source_release_identity"],
            carrier_identity=raw["carrier_identity"],
            predictor_config=config,
            optimizer_examples=raw["optimizer_examples"],
            horizon_example_counts=tuple(
                tuple(item) for item in raw["horizon_example_counts"]
            ),
            epochs=raw["epochs"],
        )
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        raise LineageScalingError(f"lineage-scaled checkpoint is invalid: {error}") from error
    scale = protocol.scale(expected_cell.scale_name)
    expected_identity = identity((
        "lineage-scaled-continuous-checkpoint-v1",
        metadata.training_report_identity,
        state_identity,
    ))
    if (
        expected_cell not in protocol.cells
        or metadata.cell != expected_cell
        or metadata.protocol_identity != protocol.identity
        or metadata.predictor_config != protocol.predictor_config
        or metadata.lineage_manifest_identity != scale.lineage_manifest_identity
        or metadata.source_release_identity != scale.source_release_identity
        or metadata.carrier_identity != protocol.carrier_identity(expected_cell.carrier)
        or metadata.optimizer_examples != protocol.optimizer_example_budget
        or metadata.model_state_identity != state_identity
        or metadata.identity != expected_identity
    ):
        raise LineageScalingError("checkpoint differs from its frozen training cell")
    model = DualOutputPredictor(config)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise LineageScalingError(f"checkpoint weights are invalid: {error}") from error
    return model.to(torch.device(device)), metadata


def validate_lineage_scaled_checkpoint_matrix(
    protocol: LineageScalingProtocol,
    checkpoints: Mapping[TrainingCell, Path],
    *,
    device: str,
) -> tuple[LineageScaledCheckpoint, ...]:
    """Reload every declared cell and reject missing, extra, or cross-bound checkpoints."""

    expected = set(protocol.cells)
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != expected:
        raise LineageScalingError(
            "exact validation requires the complete checkpoint matrix"
        )
    metadata = []
    for cell in protocol.cells:
        _model, checkpoint = load_lineage_scaled_checkpoint(
            checkpoints[cell],
            protocol,
            expected_cell=cell,
            device=device,
        )
        metadata.append(checkpoint)
    return tuple(metadata)


@dataclass(frozen=True, slots=True)
class RecursivePredictionResult:
    horizon: int
    mean_mse: float | None
    error_auc: float | None
    evaluated_transitions: int


@dataclass(frozen=True, slots=True)
class PredictionEvaluation:
    local_mse: float | None
    local_by_horizon: Mapping[int, float]
    recursive: tuple[RecursivePredictionResult, ...]
    nonfinite_failures: int
    execution_failures: tuple[str, ...]
    target_physical_diagnostics: Mapping[str, float]
    predicted_physical_diagnostics: Mapping[str, float]
    model_evaluations: int
    wall_seconds: float


def evaluate_continuous_prediction(
    model: DualOutputPredictor,
    lineages: tuple[CarrierLineage, ...],
    *,
    horizons: tuple[int, ...] = (1, 15),
    physical_diagnostic: Callable[[torch.Tensor], Mapping[str, float | bool]] | None = None,
    progress: Callable[[str], None] | None = None,
) -> PredictionEvaluation:
    """Report local and recursive errors without converting failures into errors."""

    if not isinstance(model, DualOutputPredictor) or not lineages:
        raise LineageScalingError("prediction evaluation requires a model and lineages")
    if (
        type(horizons) is not tuple
        or not horizons
        or any(type(value) is not int or value <= 0 for value in horizons)
    ):
        raise LineageScalingError("prediction horizons must be positive integers")
    device = next(model.parameters()).device
    failures: list[str] = []
    nonfinite = 0
    model_evaluations = 0
    local_errors: list[float] = []
    local_errors_by_horizon: dict[int, list[float]] = {}
    target_diagnostic_values: dict[str, list[float]] = {}
    predicted_diagnostic_values: dict[str, list[float]] = {}
    for lineage in lineages:
        for transition in lineage.transitions:
            for name, value in transition.physical_diagnostics.items():
                if value is not None and type(value) in (bool, int, float):
                    number = float(value)
                    if math.isfinite(number):
                        target_diagnostic_values.setdefault(name, []).append(number)
    started = time.monotonic()
    model.eval()
    with torch.no_grad():
        for lineage in lineages:
            for transition in lineage.transitions:
                try:
                    model_evaluations += 1
                    predicted = model.carrier(
                        transition.context.to(device).unsqueeze(0),
                        transition.action.to(device).unsqueeze(0),
                        PredictionPair(
                            transition.horizon, Abstraction.CONTINUOUS
                        ),
                    )[0]
                    if not bool(torch.isfinite(predicted).all()):
                        nonfinite += 1
                        continue
                    value = float(F.mse_loss(
                        predicted, transition.target.to(device)
                    ).cpu())
                    local_errors.append(value)
                    local_errors_by_horizon.setdefault(
                        transition.horizon, []
                    ).append(value)
                    if physical_diagnostic is not None:
                        for name, diagnostic in physical_diagnostic(
                            predicted.detach().cpu()
                        ).items():
                            predicted_diagnostic_values.setdefault(name, []).append(
                                float(diagnostic)
                            )
                except Exception as error:
                    failures.append(
                        f"local:{transition.identity}:{type(error).__name__}: {error}"
                    )

        recursive_results = []
        for horizon in horizons:
            errors: list[float] = []
            lineage_aucs: list[float] = []
            for lineage_index, lineage in enumerate(lineages, start=1):
                windows = {
                    item.decision_index: item
                    for item in lineage.transitions
                    if item.horizon == horizon
                }
                lineage_errors: list[tuple[int, float]] = []
                position = min(windows) if windows else 0
                current = (
                    windows[position].context.to(device) if windows else None
                )
                while position in windows:
                    transition = windows[position]
                    try:
                        model_evaluations += 1
                        assert current is not None
                        predicted = model.carrier(
                            current.unsqueeze(0),
                            transition.action.to(device).unsqueeze(0),
                            PredictionPair(horizon, Abstraction.CONTINUOUS),
                        )[0]
                        if not bool(torch.isfinite(predicted).all()):
                            nonfinite += 1
                            break
                        value = float(F.mse_loss(
                            predicted, transition.target.to(device)
                        ).cpu())
                        errors.append(value)
                        position += horizon
                        lineage_errors.append((position, value))
                        current = predicted
                        if physical_diagnostic is not None:
                            for name, diagnostic in physical_diagnostic(
                                predicted.detach().cpu()
                            ).items():
                                predicted_diagnostic_values.setdefault(name, []).append(
                                    float(diagnostic)
                                )
                    except Exception as error:
                        failures.append(
                            f"recursive-h{horizon}:{transition.identity}:"
                            f"{type(error).__name__}: {error}"
                        )
                        break
                if lineage_errors:
                    previous_position = lineage_errors[0][0] - horizon
                    previous_error = 0.0
                    area = 0.0
                    for target_position, error in lineage_errors:
                        area += (
                            (previous_error + error)
                            * (target_position - previous_position)
                            / 2.0
                        )
                        previous_position = target_position
                        previous_error = error
                    lineage_aucs.append(area)
                if progress is not None:
                    progress(
                        f"[score recursive-h{horizon}] lineage "
                        f"{lineage_index}/{len(lineages)}"
                    )
            recursive_results.append(RecursivePredictionResult(
                horizon=horizon,
                mean_mse=None if not errors else sum(errors) / len(errors),
                error_auc=None if not lineage_aucs else sum(lineage_aucs) / len(lineage_aucs),
                evaluated_transitions=len(errors),
            ))
    return PredictionEvaluation(
        local_mse=None if not local_errors else sum(local_errors) / len(local_errors),
        local_by_horizon=MappingProxyType({
            horizon: sum(values) / len(values)
            for horizon, values in local_errors_by_horizon.items()
        }),
        recursive=tuple(recursive_results),
        nonfinite_failures=nonfinite,
        execution_failures=tuple(failures),
        target_physical_diagnostics=MappingProxyType({
            name: sum(values) / len(values)
            for name, values in target_diagnostic_values.items()
        }),
        predicted_physical_diagnostics=MappingProxyType({
            name: sum(values) / len(values)
            for name, values in predicted_diagnostic_values.items()
        }),
        model_evaluations=model_evaluations,
        wall_seconds=time.monotonic() - started,
    )


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    identity: str
    action: torch.Tensor
    realized_cost: float

    def __post_init__(self) -> None:
        if not self.identity:
            raise LineageScalingError("action candidate identity is missing")
        _require_vector(self.action, 5, "action candidate")
        if not math.isfinite(float(self.realized_cost)):
            raise LineageScalingError("realized action cost must be finite")


@dataclass(frozen=True, slots=True)
class ActionRankingState:
    identity: str
    scenario_lineage_identity: str
    trajectory_identity: str
    decision_transition_identity: str
    exposure_role: str
    carrier: CarrierKind
    carrier_identity: str
    context: torch.Tensor
    candidates: tuple[ActionCandidate, ...]
    cost_target: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (
            not self.identity
            or not self.scenario_lineage_identity
            or not self.trajectory_identity
            or not self.decision_transition_identity
            or not self.carrier_identity
        ):
            raise LineageScalingError("action-ranking state identity is incomplete")
        if self.exposure_role not in ("calibration", "model_selection"):
            raise LineageScalingError("action-ranking state must use an isolated evaluation role")
        object.__setattr__(self, "carrier", CarrierKind(self.carrier))
        if not isinstance(self.context, torch.Tensor) or self.context.ndim != 1:
            raise LineageScalingError("action-ranking context is malformed")
        if (
            type(self.candidates) is not tuple
            or len(self.candidates) < 2
            or len({item.identity for item in self.candidates}) != len(self.candidates)
        ):
            raise LineageScalingError("declared legal candidate set is malformed")
        if self.cost_target is not None:
            _require_vector(
                self.cost_target,
                self.context.shape[0],
                "action-ranking cost target",
            )

    @property
    def candidate_set_identity(self) -> str:
        return identity((
            "realized-action-candidate-set-v1",
            self.identity,
            tuple(
                (item.identity, item.action.tolist(), float(item.realized_cost))
                for item in self.candidates
            ),
        ))


def save_action_ranking_bundle(
    path: Path,
    states: tuple[ActionRankingState, ...],
) -> None:
    target = Path(path)
    if target.exists():
        raise LineageScalingError(f"action-ranking bundle already exists: {target}")
    if (
        type(states) is not tuple
        or not states
        or len({item.identity for item in states}) != len(states)
        or any(item.cost_target is None for item in states)
    ):
        raise LineageScalingError(
            "action-ranking bundle requires unique states and frozen cost targets"
        )
    header = {
        (item.exposure_role, item.carrier, item.carrier_identity) for item in states
    }
    if len(header) != 1:
        raise LineageScalingError("action-ranking bundle crossed a role or carrier")
    role, carrier, carrier_identity = next(iter(header))
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "action_ranking_state_bundle_v1",
            "exposure_role": role,
            "carrier": carrier.value,
            "carrier_identity": carrier_identity,
            "states": [
                {
                    "identity": state.identity,
                    "scenario_lineage_identity": state.scenario_lineage_identity,
                    "trajectory_identity": state.trajectory_identity,
                    "decision_transition_identity": state.decision_transition_identity,
                    "context": state.context.detach().cpu(),
                    "cost_target": state.cost_target.detach().cpu(),
                    "candidate_set_identity": state.candidate_set_identity,
                    "candidates": [
                        {
                            "identity": candidate.identity,
                            "action": candidate.action.detach().cpu(),
                            "realized_cost": candidate.realized_cost,
                        }
                        for candidate in state.candidates
                    ],
                }
                for state in states
            ],
        },
        target,
    )


def load_action_ranking_bundle(path: Path) -> tuple[ActionRankingState, ...]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema",
            "exposure_role",
            "carrier",
            "carrier_identity",
            "states",
        }:
            raise LineageScalingError("action-ranking bundle fields differ")
        if payload["schema"] != "action_ranking_state_bundle_v1":
            raise LineageScalingError("action-ranking bundle schema is unsupported")
        raw_states = payload["states"]
        if not isinstance(raw_states, list) or not raw_states:
            raise LineageScalingError("action-ranking bundle is empty")
        states = []
        for raw_state in raw_states:
            if not isinstance(raw_state, Mapping) or set(raw_state) != {
                "identity",
                "scenario_lineage_identity",
                "trajectory_identity",
                "decision_transition_identity",
                "context",
                "cost_target",
                "candidate_set_identity",
                "candidates",
            }:
                raise LineageScalingError("action-ranking state fields differ")
            raw_candidates = raw_state["candidates"]
            if not isinstance(raw_candidates, list):
                raise LineageScalingError("action-ranking candidates are malformed")
            candidates = []
            for raw_candidate in raw_candidates:
                if not isinstance(raw_candidate, Mapping) or set(raw_candidate) != {
                    "identity", "action", "realized_cost"
                }:
                    raise LineageScalingError("action candidate fields differ")
                candidates.append(ActionCandidate(
                    raw_candidate["identity"],
                    raw_candidate["action"],
                    raw_candidate["realized_cost"],
                ))
            state = ActionRankingState(
                identity=raw_state["identity"],
                scenario_lineage_identity=raw_state["scenario_lineage_identity"],
                trajectory_identity=raw_state["trajectory_identity"],
                decision_transition_identity=raw_state[
                    "decision_transition_identity"
                ],
                exposure_role=payload["exposure_role"],
                carrier=CarrierKind(payload["carrier"]),
                carrier_identity=payload["carrier_identity"],
                context=raw_state["context"],
                candidates=tuple(candidates),
                cost_target=raw_state["cost_target"],
            )
            if state.candidate_set_identity != raw_state["candidate_set_identity"]:
                raise LineageScalingError("action candidate set identity is stale")
            states.append(state)
        return tuple(states)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        if isinstance(error, LineageScalingError):
            raise
        raise LineageScalingError(f"action-ranking bundle is invalid: {error}") from error


def validate_action_ranking_states(
    protocol: LineageScalingProtocol,
    states: tuple[ActionRankingState, ...],
    *,
    carrier: CarrierKind,
) -> TrajectoryLineageManifest:
    if not states or len({item.exposure_role for item in states}) != 1:
        raise LineageScalingError("action-ranking states crossed or omitted their role")
    role = states[0].exposure_role
    try:
        manifest = next(
            item
            for item in protocol.evaluation_manifests
            if {binding.exposure_role for binding in item.bindings} == {role}
        )
    except StopIteration as error:
        raise LineageScalingError("ranking role has no frozen lineage manifest") from error
    bindings = {
        item.scenario_lineage_identity: item for item in manifest.bindings
    }
    expected_carrier_identity = protocol.carrier_identity(carrier)
    for state in states:
        binding = bindings.get(state.scenario_lineage_identity)
        if (
            binding is None
            or state.trajectory_identity != binding.trajectory_identity
            or state.decision_transition_identity not in binding.transition_identities
            or state.carrier is not carrier
            or state.carrier_identity != expected_carrier_identity
        ):
            raise LineageScalingError(
                "ranking state, candidate source, role, or carrier mismatch"
            )
    return manifest


def validate_matched_action_ranking_states(
    protocol: LineageScalingProtocol,
    source_states: tuple[ActionRankingState, ...],
    deployment_states: tuple[ActionRankingState, ...],
) -> dict[str, int | str]:
    source_manifest = validate_action_ranking_states(
        protocol, source_states, carrier=CarrierKind.SOURCE
    )
    deployment_manifest = validate_action_ranking_states(
        protocol, deployment_states, carrier=CarrierKind.DEPLOYMENT
    )
    if source_manifest.identity != deployment_manifest.identity:
        raise LineageScalingError("ranking carriers use different role manifests")
    source_by_identity = {item.identity: item for item in source_states}
    deployment_by_identity = {item.identity: item for item in deployment_states}
    if set(source_by_identity) != set(deployment_by_identity):
        raise LineageScalingError("ranking carriers use different decision states")
    for state_identity, source in source_by_identity.items():
        deployment = deployment_by_identity[state_identity]
        if (
            source.scenario_lineage_identity
            != deployment.scenario_lineage_identity
            or source.trajectory_identity != deployment.trajectory_identity
            or source.decision_transition_identity
            != deployment.decision_transition_identity
            or source.candidate_set_identity != deployment.candidate_set_identity
        ):
            raise LineageScalingError(
                "ranking carriers changed a state or declared legal candidate set"
            )
    return {
        "evaluation_manifest_identity": source_manifest.identity,
        "state_count": len(source_states),
        "candidate_count": sum(len(item.candidates) for item in source_states),
    }


@dataclass(frozen=True, slots=True)
class ActionRankingStateResult:
    state_identity: str
    candidate_set_identity: str
    selected_candidate_identity: str
    best_realized_candidate_identity: str
    top_action_regret: float


@dataclass(frozen=True, slots=True)
class ActionRankingEvaluation:
    state_count: int
    mean_top_action_regret: float | None
    states: tuple[ActionRankingStateResult, ...]
    execution_failures: tuple[str, ...]
    model_evaluations: int
    wall_seconds: float


def evaluate_action_ranking(
    model: DualOutputPredictor,
    states: tuple[ActionRankingState, ...],
    *,
    horizon: int,
    predicted_cost: Callable[
        [ActionRankingState, ActionCandidate, torch.Tensor], float
    ],
    progress: Callable[[str], None] | None = None,
) -> ActionRankingEvaluation:
    """Measure realized regret on each state's one frozen legal candidate set."""

    if not states or type(horizon) is not int or horizon <= 0:
        raise LineageScalingError("action-ranking evaluation inputs are invalid")
    if len({(item.carrier, item.carrier_identity) for item in states}) != 1:
        raise LineageScalingError("action-ranking states contain a carrier mismatch")
    device = next(model.parameters()).device
    pair = PredictionPair(horizon, Abstraction.CONTINUOUS)
    failures = []
    results = []
    model_evaluations = 0
    started = time.monotonic()
    model.eval()
    with torch.no_grad():
        for state in states:
            scores = []
            try:
                for candidate in state.candidates:
                    model_evaluations += 1
                    predicted = model.carrier(
                        state.context.to(device).unsqueeze(0),
                        candidate.action.to(device).unsqueeze(0),
                        pair,
                    )[0]
                    if not bool(torch.isfinite(predicted).all()):
                        raise RuntimeError("candidate prediction is nonfinite")
                    score = float(predicted_cost(state, candidate, predicted.detach().cpu()))
                    if not math.isfinite(score):
                        raise RuntimeError("candidate predicted cost is nonfinite")
                    scores.append(score)
                selected_index = min(range(len(scores)), key=scores.__getitem__)
                best_index = min(
                    range(len(state.candidates)),
                    key=lambda index: state.candidates[index].realized_cost,
                )
                selected = state.candidates[selected_index]
                best = state.candidates[best_index]
                results.append(ActionRankingStateResult(
                    state_identity=state.identity,
                    candidate_set_identity=state.candidate_set_identity,
                    selected_candidate_identity=selected.identity,
                    best_realized_candidate_identity=best.identity,
                    top_action_regret=float(selected.realized_cost - best.realized_cost),
                ))
                if progress is not None:
                    progress(
                        f"[rank] state={state.identity} "
                        f"candidates={len(state.candidates)}"
                    )
            except Exception as error:
                failures.append(f"{state.identity}:{type(error).__name__}: {error}")
    return ActionRankingEvaluation(
        state_count=len(results),
        mean_top_action_regret=(
            None
            if not results
            else sum(item.top_action_regret for item in results) / len(results)
        ),
        states=tuple(results),
        execution_failures=tuple(failures),
        model_evaluations=model_evaluations,
        wall_seconds=time.monotonic() - started,
    )


@dataclass(frozen=True, slots=True)
class MatchedGameplayProtocol:
    action_candidate_set_identity: str
    cost_terms_identity: str
    action_bounds: SlingshotActionBounds
    cost_config: GameplayCostConfig
    population_size: int
    elite_count: int
    cem_iterations: int
    sequence_length: int
    max_shots: int
    max_planner_compute: float
    fixed_steps_per_shot: int
    transition_compute: float
    controller_compute: float
    seed: int = 20260831

    def __post_init__(self) -> None:
        if not self.action_candidate_set_identity or not self.cost_terms_identity:
            raise LineageScalingError("gameplay candidate or cost identity is missing")
        if (
            type(self.action_bounds) is not SlingshotActionBounds
            or type(self.cost_config) is not GameplayCostConfig
            or any(
                type(value) is not int or value <= 0
                for value in (
                    self.population_size,
                    self.elite_count,
                    self.cem_iterations,
                    self.sequence_length,
                    self.max_shots,
                    self.fixed_steps_per_shot,
                )
            )
            or self.elite_count > self.population_size
            or not math.isfinite(float(self.max_planner_compute))
            or self.max_planner_compute <= 0
            or not math.isfinite(float(self.transition_compute))
            or self.transition_compute < 0.0
            or not math.isfinite(float(self.controller_compute))
            or self.controller_compute < 0.0
            or type(self.seed) is not int
            or self.seed < 0
        ):
            raise LineageScalingError("matched gameplay limits are invalid")

    @property
    def identity(self) -> str:
        return identity((
            "lineage-scaled-matched-gameplay-protocol-v1",
            self.action_candidate_set_identity,
            self.cost_terms_identity,
            (
                self.action_bounds.drag_x,
                self.action_bounds.drag_y,
                self.action_bounds.tap_time_ms,
                self.action_bounds.release_time_ms,
            ),
            tuple(asdict(self.cost_config).items()),
            self.population_size,
            self.elite_count,
            self.cem_iterations,
            self.sequence_length,
            self.max_shots,
            self.max_planner_compute,
            self.fixed_steps_per_shot,
            self.transition_compute,
            self.controller_compute,
            self.seed,
        ))


@dataclass(frozen=True, slots=True)
class GameplayCheckpointBindings:
    legacy_predictor: Path
    retrained_predictor: Path
    adaptive_controller: Path

    def __post_init__(self) -> None:
        for value in (
            self.legacy_predictor,
            self.retrained_predictor,
            self.adaptive_controller,
        ):
            if not isinstance(value, Path) or not value.is_absolute():
                raise LineageScalingError(
                    "gameplay planning requires explicit absolute checkpoint paths"
                )
        if self.legacy_predictor == self.retrained_predictor:
            raise LineageScalingError(
                "legacy and retrained gameplay checkpoints must be explicit and distinct"
            )


@dataclass(frozen=True, slots=True)
class GameplaySystemSpec:
    identity: str
    checkpoint_role: str
    mode: str
    predictor_checkpoint: Path
    controller_checkpoint: Path | None
    fixed_horizon: int | None
    protocol_identity: str


def matched_gameplay_systems(
    protocol: MatchedGameplayProtocol,
    checkpoints: GameplayCheckpointBindings,
) -> tuple[GameplaySystemSpec, ...]:
    """Expose the same h1, h15, and adaptive MPC systems for both checkpoints."""

    systems = []
    for checkpoint_role, predictor in (
        ("legacy", checkpoints.legacy_predictor),
        ("retrained", checkpoints.retrained_predictor),
    ):
        for mode, fixed_horizon in (
            ("continuous-h1", 1),
            ("continuous-h15", 15),
            ("adaptive", None),
        ):
            system_identity = identity((
                "lineage-scaled-gameplay-system-v1",
                checkpoint_role,
                mode,
                str(predictor),
                (
                    str(checkpoints.adaptive_controller)
                    if mode == "adaptive"
                    else None
                ),
                protocol.identity,
            ))
            systems.append(GameplaySystemSpec(
                identity=system_identity,
                checkpoint_role=checkpoint_role,
                mode=mode,
                predictor_checkpoint=predictor,
                controller_checkpoint=(
                    checkpoints.adaptive_controller if mode == "adaptive" else None
                ),
                fixed_horizon=fixed_horizon,
                protocol_identity=protocol.identity,
            ))
    return tuple(systems)


@dataclass(frozen=True, slots=True)
class MatchedGameplayPlanner:
    system: GameplaySystemSpec
    world_model: ContinuousCheckpointWorldModel
    planner: CEMPlanner
    control: ControlConfig


def build_matched_gameplay_planners(
    protocol: MatchedGameplayProtocol,
    systems: tuple[GameplaySystemSpec, ...],
    *,
    predictor_loader: Callable[[Path], DualOutputPredictor],
    adaptive_selector_loader: Callable[
        [Path], Callable[[PlanningObservation, SlingshotAction], int]
    ],
    progress: Callable[[str], None] | None = None,
) -> tuple[MatchedGameplayPlanner, ...]:
    """Build six real CEM/MPC planners from one matched configuration."""

    expected = {
        (checkpoint_role, mode)
        for checkpoint_role in ("legacy", "retrained")
        for mode in ("continuous-h1", "continuous-h15", "adaptive")
    }
    if (
        type(systems) is not tuple
        or {(item.checkpoint_role, item.mode) for item in systems} != expected
        or len(systems) != len(expected)
        or any(item.protocol_identity != protocol.identity for item in systems)
    ):
        raise LineageScalingError("gameplay systems do not form the matched six-system matrix")
    predictors: dict[Path, DualOutputPredictor] = {}
    selectors: dict[Path, Callable[[PlanningObservation, SlingshotAction], int]] = {}
    built = []
    for system in systems:
        predictor = predictors.get(system.predictor_checkpoint)
        if predictor is None:
            predictor = predictor_loader(system.predictor_checkpoint)
            if not isinstance(predictor, DualOutputPredictor):
                raise LineageScalingError("gameplay checkpoint did not load a continuous predictor")
            predictor.eval()
            predictors[system.predictor_checkpoint] = predictor
        selector = None
        if system.mode == "adaptive":
            if system.controller_checkpoint is None:
                raise LineageScalingError("adaptive gameplay system has no controller checkpoint")
            selector = selectors.get(system.controller_checkpoint)
            if selector is None:
                selector = adaptive_selector_loader(system.controller_checkpoint)
                if not callable(selector):
                    raise LineageScalingError("adaptive controller checkpoint did not load a selector")
                selectors[system.controller_checkpoint] = selector
        elif system.controller_checkpoint is not None or system.fixed_horizon not in (1, 15):
            raise LineageScalingError("fixed gameplay system has a checkpoint or horizon mismatch")
        world_model = ContinuousCheckpointWorldModel(
            predictor=predictor,
            fixed_steps_per_shot=protocol.fixed_steps_per_shot,
            release_time_ms=protocol.action_bounds.release_time_ms,
            transition_compute=protocol.transition_compute,
            fixed_horizon=system.fixed_horizon,
            horizon_selector=selector,
            controller_compute=protocol.controller_compute,
        )
        evaluator = WorldModelCandidateEvaluator(
            world_model,
            protocol.action_bounds,
            GameplayCost(protocol.cost_config),
        )
        prefix = f"{system.checkpoint_role}/{system.mode}"
        planner = CEMPlanner(
            CEMConfig(
                population_size=protocol.population_size,
                elite_count=protocol.elite_count,
                iterations=protocol.cem_iterations,
                sequence_length=protocol.sequence_length,
                seed=protocol.seed,
            ),
            protocol.action_bounds,
            evaluator,
            progress=(
                None
                if progress is None
                else lambda value, prefix=prefix: progress(f"[{prefix}] {value}")
            ),
        )
        built.append(MatchedGameplayPlanner(
            system=system,
            world_model=world_model,
            planner=planner,
            control=ControlConfig(
                ControlMode.MPC,
                protocol.max_shots,
                protocol.max_planner_compute,
            ),
        ))
    return tuple(built)


__all__ = [
    "ActionCandidate",
    "ActionRankingEvaluation",
    "ActionRankingState",
    "ActionRankingStateResult",
    "CarrierKind",
    "CarrierLineage",
    "ContinuousTransitionExample",
    "FrozenLineageScale",
    "GameplayCheckpointBindings",
    "GameplaySystemSpec",
    "LineageScaledCheckpoint",
    "LineageScalingError",
    "LineageScalingProtocol",
    "MatchedGameplayProtocol",
    "MatchedGameplayPlanner",
    "PredictionEvaluation",
    "RecursivePredictionResult",
    "TrainingCell",
    "TrainingReport",
    "evaluate_action_ranking",
    "evaluate_continuous_prediction",
    "build_matched_gameplay_planners",
    "load_action_ranking_bundle",
    "load_carrier_lineage_bundle",
    "load_lineage_scaling_protocol",
    "load_lineage_scaled_checkpoint",
    "matched_gameplay_systems",
    "save_action_ranking_bundle",
    "save_carrier_lineage_bundle",
    "save_lineage_scaling_protocol",
    "save_lineage_scaled_checkpoint",
    "train_continuous_predictor",
    "validate_carrier_alignment",
    "validate_action_ranking_states",
    "validate_evaluation_lineages",
    "validate_lineage_scaled_checkpoint_matrix",
    "validate_matched_action_ranking_states",
    "validate_matched_carrier_lineages",
]
