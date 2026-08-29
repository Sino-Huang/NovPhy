"""Deployment-valid temporal carriers and multi-shot decision contracts."""
from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from types import MappingProxyType
from typing import Any, Final, Mapping

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch

from world_model.model import BooleanTransitionValue, RelationTransitionValue, identity


OBJECT_KIND_VOCABULARY: Final = (
    "bird",
    "pig",
    "block",
    "platform",
    "slingshot",
    "world",
    "other",
)
ENTITY_FEATURES: Final = (
    "presence_probability",
    "presence_available",
    "kind_index_normalized",
    "kind_confidence",
    "kind_available",
    "center_x_normalized",
    "center_y_normalized",
    "center_available",
    "motion_x_normalized_per_second",
    "motion_y_normalized_per_second",
    "motion_available",
    "expected_kind_probability",
    "slot_declared",
)
INTERFACE_ACTION_FIELDS: Final = frozenset((
    "action_type",
    "coordinate_frame",
    "drag_start",
    "drag_release",
    "frame_height",
    "releaseTime",
    "tapTime",
))
ENGINE_ACTION_FIELDS: Final = frozenset((
    "schema",
    "drag_delta_canvas_pixels",
    "hold_milliseconds",
    "tap_time_milliseconds",
))


class DeploymentTemporalError(ValueError):
    """A deployment observation or temporal carrier violates its public contract."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _observation_payload(value: "AgentObservation") -> dict[str, Any]:
    return {
        "identity": value.identity,
        "observation_role": value.observation_role,
        "fixed_step": value.fixed_step,
        "fixed_time_seconds": (
            None
            if value.fixed_time_seconds is None
            else float(value.fixed_time_seconds)
        ),
    }


@dataclass(frozen=True, slots=True)
class AgentObservation:
    identity: str
    fixed_step: int | None
    fixed_time_seconds: float | None
    png: bytes
    observation_role: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity:
            raise DeploymentTemporalError("agent observation identity is missing")
        if (self.fixed_step is None) != (self.fixed_time_seconds is None):
            raise DeploymentTemporalError(
                "agent observation time identities must be both available or unavailable"
            )
        if self.fixed_step is not None and (
            type(self.fixed_step) is not int
            or self.fixed_step < 0
            or type(self.fixed_time_seconds) not in (int, float)
            or not math.isfinite(float(self.fixed_time_seconds))
            or self.fixed_time_seconds < 0
        ):
            raise DeploymentTemporalError("agent observation time identities are invalid")
        if type(self.png) is not bytes:
            raise DeploymentTemporalError("agent observation pixels must be bytes")
        if self.observation_role != "agent":
            raise DeploymentTemporalError("only agent observations are deployment inputs")


@dataclass(frozen=True, slots=True)
class TemporalObservationContext:
    prior: AgentObservation | None
    current: AgentObservation

    def __post_init__(self) -> None:
        if type(self.current) is not AgentObservation:
            raise DeploymentTemporalError("temporal context requires a current agent observation")
        if self.prior is not None and (
            type(self.prior) is not AgentObservation
            or self.prior.fixed_step is None
            or self.current.fixed_step is None
            or self.prior.fixed_time_seconds is None
            or self.current.fixed_time_seconds is None
            or self.prior.fixed_step >= self.current.fixed_step
            or self.prior.fixed_time_seconds >= self.current.fixed_time_seconds
        ):
            raise DeploymentTemporalError(
                "prior observation must strictly precede the current observation"
            )


@dataclass(frozen=True, slots=True)
class ExecutedAction:
    identity: str
    interface_action: Mapping[str, Any]
    engine_relative_action: Mapping[str, Any]
    legal: bool

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity:
            raise DeploymentTemporalError("executed action identity is missing")
        if (
            not isinstance(self.interface_action, Mapping)
            or not isinstance(self.engine_relative_action, Mapping)
        ):
            raise DeploymentTemporalError("executed action bindings are malformed")
        if self.legal is not True:
            raise DeploymentTemporalError("decision transitions require an executed legal action")
        if (
            not set(self.interface_action).issubset(INTERFACE_ACTION_FIELDS)
            or not set(self.engine_relative_action).issubset(ENGINE_ACTION_FIELDS)
        ):
            raise DeploymentTemporalError(
                "executed action contains unsupported expert or target inputs"
            )
        interface_drag = self.interface_action.get("drag_release")
        engine_drag = self.engine_relative_action.get("drag_delta_canvas_pixels")
        if (
            not isinstance(interface_drag, (tuple, list))
            or not isinstance(engine_drag, (tuple, list))
            or tuple(interface_drag) != tuple(engine_drag)
            or self.interface_action.get("releaseTime")
            != self.engine_relative_action.get("hold_milliseconds")
            or self.interface_action.get("tapTime")
            != self.engine_relative_action.get("tap_time_milliseconds")
        ):
            raise DeploymentTemporalError(
                "executed action binding differs between interface and engine-relative forms"
            )
        object.__setattr__(self, "interface_action", _freeze(self.interface_action))
        object.__setattr__(self, "engine_relative_action", _freeze(self.engine_relative_action))


@dataclass(frozen=True, slots=True)
class DecisionTargets:
    next_observation: AgentObservation
    source_frame_record_identity: str
    source_state_identity: str
    source_targets: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.next_observation) is not AgentObservation:
            raise DeploymentTemporalError("decision targets require the next agent observation")
        if not self.source_frame_record_identity or not self.source_state_identity:
            raise DeploymentTemporalError("decision target source identities are missing")
        if not isinstance(self.source_targets, Mapping):
            raise DeploymentTemporalError("source-derived decision targets are malformed")
        object.__setattr__(self, "source_targets", _freeze(self.source_targets))


@dataclass(frozen=True, slots=True)
class DecisionInference:
    observations: TemporalObservationContext
    action: ExecutedAction


@dataclass(frozen=True, slots=True)
class DeploymentFrameRecordSymbols:
    frame_record_identity: str
    contact: RelationTransitionValue
    supports: RelationTransitionValue
    steady_state: BooleanTransitionValue
    structure_unstable: BooleanTransitionValue


@dataclass(frozen=True, slots=True)
class DecisionTransition:
    identity: str
    scenario_lineage_identity: str
    exposure_role: str
    decision_index: int
    prior_observation: AgentObservation | None
    current_observation: AgentObservation
    action: ExecutedAction
    targets: DecisionTargets
    terminal_status: str
    source_bindings: Mapping[str, Any]
    schema: str = "deployment_decision_transition_v1"

    def __post_init__(self) -> None:
        if self.schema != "deployment_decision_transition_v1":
            raise DeploymentTemporalError("decision transition schema is unsupported")
        if not self.identity or not self.scenario_lineage_identity:
            raise DeploymentTemporalError("decision transition identity is missing")
        if self.exposure_role not in (
            "training",
            "calibration",
            "model_selection",
            "final_evaluation",
        ):
            raise DeploymentTemporalError("decision transition exposure role is invalid")
        if type(self.decision_index) is not int or self.decision_index < 0:
            raise DeploymentTemporalError("decision transition index is invalid")
        observations = TemporalObservationContext(
            self.prior_observation, self.current_observation
        )
        if type(self.action) is not ExecutedAction or type(self.targets) is not DecisionTargets:
            raise DeploymentTemporalError("decision transition action or targets are malformed")
        next_observation = self.targets.next_observation
        if (
            self.current_observation.fixed_step is None
            or self.current_observation.fixed_time_seconds is None
            or next_observation.fixed_step is None
            or next_observation.fixed_time_seconds is None
            or next_observation.fixed_step <= self.current_observation.fixed_step
            or next_observation.fixed_time_seconds
            <= self.current_observation.fixed_time_seconds
        ):
            raise DeploymentTemporalError(
                "next observation must strictly follow the current observation"
            )
        if not isinstance(self.terminal_status, str) or not self.terminal_status:
            raise DeploymentTemporalError("decision terminal status is missing")
        if not isinstance(self.source_bindings, Mapping):
            raise DeploymentTemporalError("decision source bindings are malformed")
        object.__setattr__(self, "source_bindings", _freeze(self.source_bindings))

    @property
    def inference(self) -> DecisionInference:
        return DecisionInference(
            TemporalObservationContext(self.prior_observation, self.current_observation),
            self.action,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "identity": self.identity,
            "scenario_lineage_identity": self.scenario_lineage_identity,
            "exposure_role": self.exposure_role,
            "decision_index": self.decision_index,
            "prior_observation": (
                None
                if self.prior_observation is None
                else _observation_payload(self.prior_observation)
            ),
            "current_observation": _observation_payload(self.current_observation),
            "action": {
                "identity": self.action.identity,
                "legal": self.action.legal,
                "interface_action": _thaw(self.action.interface_action),
                "engine_relative_action": _thaw(self.action.engine_relative_action),
            },
            "next_observation": _observation_payload(
                self.targets.next_observation
            ),
            "terminal_status": self.terminal_status,
            "source_bindings": _thaw(self.source_bindings),
            "source_targets": {
                "source_frame_record_identity": (
                    self.targets.source_frame_record_identity
                ),
                "source_state_identity": self.targets.source_state_identity,
                "values": _thaw(self.targets.source_targets),
            },
        }


@dataclass(frozen=True, slots=True)
class DeploymentTrajectory:
    identity: str
    scenario_lineage_identity: str
    exposure_role: str
    transitions: tuple[DecisionTransition, ...]
    complete: bool
    schema: str = "deployment_decision_trajectory_v1"

    def __post_init__(self) -> None:
        if self.schema != "deployment_decision_trajectory_v1":
            raise DeploymentTemporalError("deployment trajectory schema is unsupported")
        if not self.identity or not self.scenario_lineage_identity:
            raise DeploymentTemporalError("deployment trajectory identity is missing")
        if self.complete is not True:
            raise DeploymentTemporalError("readers require complete trajectories")
        if (
            type(self.transitions) is not tuple
            or not self.transitions
            or any(type(item) is not DecisionTransition for item in self.transitions)
        ):
            raise DeploymentTemporalError("deployment trajectory transitions are malformed")
        for index, transition in enumerate(self.transitions):
            if (
                transition.scenario_lineage_identity != self.scenario_lineage_identity
                or transition.exposure_role != self.exposure_role
                or transition.decision_index != index
            ):
                raise DeploymentTemporalError(
                    "deployment trajectory crossed its lineage or exposure role"
                )
            if index:
                previous = self.transitions[index - 1]
                if (
                    transition.current_observation.identity
                    != previous.targets.next_observation.identity
                    or transition.prior_observation is None
                    or transition.prior_observation.identity
                    != previous.current_observation.identity
                ):
                    raise DeploymentTemporalError(
                        "deployment trajectory decision observations are not contiguous"
                    )
        if self.transitions[-1].terminal_status == "ongoing":
            raise DeploymentTemporalError("complete trajectory lacks terminal status")


@dataclass(frozen=True, slots=True)
class TrajectoryLineageBinding:
    trajectory_identity: str
    scenario_lineage_identity: str
    exposure_role: str
    transition_identities: tuple[str, ...]
    initial_observation_identity: str
    terminal_observation_identity: str

    def __post_init__(self) -> None:
        if (
            not self.trajectory_identity
            or not self.scenario_lineage_identity
            or not self.transition_identities
            or not self.initial_observation_identity
            or not self.terminal_observation_identity
        ):
            raise DeploymentTemporalError("trajectory lineage binding is incomplete")


@dataclass(frozen=True, slots=True)
class TrajectoryLineageManifest:
    identity: str
    source_release_identity: str
    bindings: tuple[TrajectoryLineageBinding, ...]
    schema: str = "deployment_trajectory_lineage_manifest_v1"

    @classmethod
    def create(
        cls,
        source_release_identity: str,
        bindings: tuple[TrajectoryLineageBinding, ...],
    ) -> "TrajectoryLineageManifest":
        manifest_identity = cls._identity(source_release_identity, bindings)
        return cls(manifest_identity, source_release_identity, bindings)

    @staticmethod
    def _identity(
        source_release_identity: str,
        bindings: tuple[TrajectoryLineageBinding, ...],
    ) -> str:
        return identity((
            "deployment-trajectory-lineage-manifest-v1",
            source_release_identity,
            tuple(
                (
                    item.trajectory_identity,
                    item.scenario_lineage_identity,
                    item.exposure_role,
                    item.transition_identities,
                    item.initial_observation_identity,
                    item.terminal_observation_identity,
                )
                for item in bindings
            ),
        ))

    def __post_init__(self) -> None:
        if (
            self.schema != "deployment_trajectory_lineage_manifest_v1"
            or not self.source_release_identity
            or not self.bindings
            or self.identity
            != self._identity(self.source_release_identity, self.bindings)
        ):
            raise DeploymentTemporalError(
                "trajectory lineage manifest identity or schema differs"
            )


class DeploymentTrajectoryReader:
    """Expose complete trajectories for exactly one permitted exposure role."""

    def __init__(
        self,
        trajectories: tuple[DeploymentTrajectory, ...],
        *,
        exposure_role: str,
        lineage_manifest: TrajectoryLineageManifest,
    ) -> None:
        if (
            type(trajectories) is not tuple
            or not trajectories
            or any(type(item) is not DeploymentTrajectory for item in trajectories)
        ):
            raise DeploymentTemporalError("reader inputs must be complete trajectories")
        if any(item.exposure_role != exposure_role for item in trajectories):
            raise DeploymentTemporalError("reader crossed its declared exposure role")
        if type(lineage_manifest) is not TrajectoryLineageManifest:
            raise DeploymentTemporalError("reader lineage manifest is missing")
        bindings = {
            item.trajectory_identity: item for item in lineage_manifest.bindings
        }
        if (
            len(bindings) != len(lineage_manifest.bindings)
            or set(bindings) != {item.identity for item in trajectories}
        ):
            raise DeploymentTemporalError(
                "reader trajectory inventory differs from lineage manifest"
            )
        for trajectory in trajectories:
            binding = bindings.get(trajectory.identity)
            if (
                binding is None
                or binding.scenario_lineage_identity
                != trajectory.scenario_lineage_identity
                or binding.exposure_role != trajectory.exposure_role
                or binding.initial_observation_identity
                != trajectory.transitions[0].current_observation.identity
                or binding.terminal_observation_identity
                != trajectory.transitions[-1].targets.next_observation.identity
            ):
                raise DeploymentTemporalError(
                    "trajectory differs from its lineage binding"
                )
            if binding.transition_identities != tuple(
                item.identity for item in trajectory.transitions
            ):
                raise DeploymentTemporalError(
                    "trajectory decision inventory is incomplete or split"
                )
            if any(
                transition.source_bindings.get("release_identity")
                != lineage_manifest.source_release_identity
                for transition in trajectory.transitions
            ):
                raise DeploymentTemporalError(
                    "trajectory crossed its source release lineage manifest"
                )
        lineages = tuple(item.scenario_lineage_identity for item in trajectories)
        if len(lineages) != len(set(lineages)):
            raise DeploymentTemporalError(
                "each scenario lineage must be one complete trajectory"
            )
        self.exposure_role = exposure_role
        self.trajectories = trajectories
        self.lineage_manifest = lineage_manifest

    def iter_transitions(self):
        for trajectory in self.trajectories:
            yield from trajectory.transitions

    @staticmethod
    def validate_role_isolation(
        readers: tuple["DeploymentTrajectoryReader", ...],
    ) -> None:
        assignments: dict[str, str] = {}
        for reader in readers:
            if not isinstance(reader, DeploymentTrajectoryReader):
                raise DeploymentTemporalError("role isolation requires trajectory readers")
            for trajectory in reader.trajectories:
                prior = assignments.setdefault(
                    trajectory.scenario_lineage_identity, reader.exposure_role
                )
                if prior != reader.exposure_role:
                    raise DeploymentTemporalError(
                        "scenario lineage leaked across exposure roles"
                    )


@dataclass(frozen=True, slots=True)
class TemporalObjectSlot:
    identity: str
    presence_probability: float
    kind: str
    kind_confidence: float
    expected_kind_probability: float
    center_x: float
    center_y: float
    center_available: bool
    motion_x_per_second: float
    motion_y_per_second: float
    motion_available: bool


@dataclass(frozen=True, slots=True)
class TemporalCarrier:
    adapter_identity: str
    observation_identity: str
    tensor: torch.Tensor
    object_slots: tuple[TemporalObjectSlot, ...]
    fixed_step_delta: int | None
    elapsed_seconds: float | None
    diagnostics: Mapping[str, Any]
    symbols: DeploymentFrameRecordSymbols


@dataclass(frozen=True, slots=True)
class TransitionCarriers:
    context: TemporalCarrier
    target: TemporalCarrier
    action: ExecutedAction
    targets: DecisionTargets


class TemporalVisualCarrierAdapter:
    """Build the deployment carrier from current and optional prior agent RGB."""

    identity = (
        "deployment-visual-temporal-carrier-v1:normalized-screen-centers;"
        "normalized-motion-per-second;motion-availability-mask;"
        "frozen-object-and-kind-vocabularies"
    )

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        parser_checkpoint_identity: str,
        temperatures: Mapping[str, float],
        thresholds: Mapping[str, float],
        latent_dim: int,
        max_entities: int,
        object_kind_temperature: float = 1.0,
    ) -> None:
        if len(model.object_vocabulary) > max_entities or latent_dim < 2 + 13 * max_entities:
            raise DeploymentTemporalError(
                "temporal carrier dimensions do not fit the frozen parser vocabulary"
            )
        self.model = model
        self.parser_checkpoint_identity = parser_checkpoint_identity
        self.temperatures = dict(temperatures)
        self.thresholds = dict(thresholds)
        self.latent_dim = latent_dim
        self.max_entities = max_entities
        self.object_kind_temperature = object_kind_temperature

    def _parse(self, observation: AgentObservation) -> dict[str, torch.Tensor]:
        try:
            with Image.open(BytesIO(observation.png)) as opened:
                image = opened.convert("RGB").resize(
                    (self.model.config.image_width, self.model.config.image_height),
                    Image.Resampling.BILINEAR,
                )
                array = np.asarray(image, dtype=np.uint8).copy()
        except (OSError, UnidentifiedImageError) as error:
            raise DeploymentTemporalError("agent observation is not a readable image") from error
        device = next(self.model.parameters()).device
        with torch.no_grad():
            output = self.model(
                torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
            )
        return {
            "presence": torch.sigmoid(
                output["presence_logits"][0] / self.temperatures["object_presence"]
            ).detach().cpu(),
            "centers": output["centers"][0].detach().cpu(),
            "kinds": torch.softmax(
                output["kind_logits"][0] / self.object_kind_temperature, dim=-1
            ).detach().cpu(),
            "relations": torch.sigmoid(
                output["relation_logits"][0]
                / torch.tensor(
                    (
                        self.temperatures["contact"],
                        self.temperatures["supports"],
                    ),
                    device=output["relation_logits"].device,
                )
            ).detach().cpu(),
            "macros": torch.sigmoid(
                output["macro_logits"][0]
                / torch.tensor(
                    (
                        self.temperatures["steady-state"],
                        self.temperatures["structure-unstable"],
                    ),
                    device=output["macro_logits"].device,
                )
            ).detach().cpu(),
        }

    def build(self, context: TemporalObservationContext) -> TemporalCarrier:
        if type(context) is not TemporalObservationContext:
            raise DeploymentTemporalError("carrier adapter requires temporal observations")
        current = self._parse(context.current)
        prior = None if context.prior is None else self._parse(context.prior)
        elapsed = (
            None
            if context.prior is None
            or context.prior.fixed_time_seconds is None
            or context.current.fixed_time_seconds is None
            else context.current.fixed_time_seconds - context.prior.fixed_time_seconds
        )
        step_delta = (
            None
            if context.prior is None
            or context.prior.fixed_step is None
            or context.current.fixed_step is None
            else context.current.fixed_step - context.prior.fixed_step
        )
        values = [float(context.prior is not None), 0.0 if elapsed is None else float(elapsed)]
        slots = []
        for index, slot_identity in enumerate(self.model.object_vocabulary):
            presence = float(current["presence"][index])
            center_x, center_y = (float(value) for value in current["centers"][index])
            kind_probabilities = current["kinds"][index]
            kind_index = int(kind_probabilities.argmax())
            kind = OBJECT_KIND_VOCABULARY[kind_index]
            expected_kind = slot_identity.split(":", 1)[0]
            expected_index = (
                OBJECT_KIND_VOCABULARY.index(expected_kind)
                if expected_kind in OBJECT_KIND_VOCABULARY[:-1]
                else len(OBJECT_KIND_VOCABULARY) - 1
            )
            center_available = presence >= self.thresholds["object_presence"]
            motion_available = bool(
                prior is not None
                and float(prior["presence"][index]) >= self.thresholds["object_presence"]
                and center_available
            )
            motion_x = 0.0
            motion_y = 0.0
            if motion_available:
                assert elapsed is not None
                motion_x = (center_x - float(prior["centers"][index, 0])) / elapsed
                motion_y = (center_y - float(prior["centers"][index, 1])) / elapsed
            slot = TemporalObjectSlot(
                identity=slot_identity,
                presence_probability=presence,
                kind=kind,
                kind_confidence=float(kind_probabilities[kind_index]),
                expected_kind_probability=float(kind_probabilities[expected_index]),
                center_x=center_x if center_available else 0.0,
                center_y=center_y if center_available else 0.0,
                center_available=center_available,
                motion_x_per_second=motion_x,
                motion_y_per_second=motion_y,
                motion_available=motion_available,
            )
            slots.append(slot)
            values.extend((
                slot.presence_probability,
                1.0,
                kind_index / (len(OBJECT_KIND_VOCABULARY) - 1),
                slot.kind_confidence,
                1.0,
                slot.center_x,
                slot.center_y,
                float(slot.center_available),
                slot.motion_x_per_second,
                slot.motion_y_per_second,
                float(slot.motion_available),
                slot.expected_kind_probability,
                1.0,
            ))
        values.extend((0.0,) * ((self.max_entities - len(slots)) * len(ENTITY_FEATURES)))
        values.extend((0.0,) * (self.latent_dim - len(values)))
        present = tuple(
            float(current["presence"][index]) >= self.thresholds["object_presence"]
            for index in range(len(self.model.object_vocabulary))
        )
        relation_values = {}
        for predicate_index, predicate in enumerate(("contact", "supports")):
            selected = []
            for first in range(len(self.model.object_vocabulary)):
                for second in range(len(self.model.object_vocabulary)):
                    if first == second or not present[first] or not present[second]:
                        continue
                    if (
                        float(current["relations"][first, second, predicate_index])
                        < self.thresholds[predicate]
                    ):
                        continue
                    if predicate == "contact" and first > second:
                        continue
                    selected.append((
                        "runtime:" + self.model.object_vocabulary[first],
                        "runtime:" + self.model.object_vocabulary[second],
                    ))
            relation_values[predicate] = RelationTransitionValue(
                "available", tuple(selected)
            )
        macro_values = {
            predicate: BooleanTransitionValue(
                "available",
                float(current["macros"][index]) >= self.thresholds[predicate],
            )
            for index, predicate in enumerate(("steady-state", "structure-unstable"))
        }
        symbols = DeploymentFrameRecordSymbols(
            context.current.identity,
            relation_values["contact"],
            relation_values["supports"],
            macro_values["steady-state"],
            macro_values["structure-unstable"],
        )
        diagnostics = {
            "parser_checkpoint_identity": self.parser_checkpoint_identity,
            "carrier_adapter_identity": self.identity,
            "entity_features": ENTITY_FEATURES,
            "motion_units": "normalized_screen_center_per_second",
            "fixed_time_identity_available": (
                context.current.fixed_step is not None
                and context.current.fixed_time_seconds is not None
            ),
            "object_presence_probabilities": tuple(
                float(value) for value in current["presence"]
            ),
            "object_centers": tuple(
                tuple(float(value) for value in row) for row in current["centers"]
            ),
            "object_kind_probabilities": tuple(
                tuple(float(value) for value in row) for row in current["kinds"]
            ),
            "steady_state_probability": float(current["macros"][0]),
            "steady_state_thresholded": macro_values["steady-state"].value,
            "structure_unstable_probability": float(current["macros"][1]),
            "structure_unstable_thresholded": macro_values["structure-unstable"].value,
        }
        return TemporalCarrier(
            adapter_identity=self.identity,
            observation_identity=context.current.identity,
            tensor=torch.tensor(values, dtype=torch.float32),
            object_slots=tuple(slots),
            fixed_step_delta=step_delta,
            elapsed_seconds=elapsed,
            diagnostics=diagnostics,
            symbols=symbols,
        )


def build_transition_carriers(
    transition: DecisionTransition,
    adapter: TemporalVisualCarrierAdapter,
) -> TransitionCarriers:
    """Build deployment-identical context and next-observation training carriers."""
    if type(transition) is not DecisionTransition or not isinstance(
        adapter, TemporalVisualCarrierAdapter
    ):
        raise DeploymentTemporalError(
            "transition carriers require the deployment transition and adapter"
        )
    context = adapter.build(transition.inference.observations)
    target = adapter.build(TemporalObservationContext(
        transition.current_observation,
        transition.targets.next_observation,
    ))
    return TransitionCarriers(context, target, transition.action, transition.targets)


__all__ = [
    "AgentObservation",
    "DecisionInference",
    "DecisionTargets",
    "DecisionTransition",
    "DeploymentTemporalError",
    "DeploymentFrameRecordSymbols",
    "DeploymentTrajectory",
    "DeploymentTrajectoryReader",
    "ENTITY_FEATURES",
    "OBJECT_KIND_VOCABULARY",
    "TemporalCarrier",
    "TemporalObjectSlot",
    "TemporalObservationContext",
    "TemporalVisualCarrierAdapter",
    "TransitionCarriers",
    "TrajectoryLineageBinding",
    "TrajectoryLineageManifest",
    "ExecutedAction",
    "build_transition_carriers",
]
