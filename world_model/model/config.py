"""Validated configuration identities for the BG-NS-JEPA backbone.

Every configuration object is an immutable dataclass that validates in
``__post_init__`` and exposes a stable declared ``identity``.  Manifests record
these semantic identities directly rather than deriving integrity values.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Final

from world_model.data.types import ContractValueError

@unique
class Abstraction(StrEnum):
    """The representational axis ``alpha`` of the joint controller (proposal 2.2)."""

    CONTINUOUS = "continuous"
    MICRO = "micro"
    MACRO = "macro"


ABSTRACTION_ORDER: Final = (Abstraction.CONTINUOUS, Abstraction.MICRO, Abstraction.MACRO)
MICRO_TRANSITION_INPUTS: Final = ("contact", "supports")
MACRO_TRANSITION_INPUTS: Final = ("steady-state", "structure-unstable")


def abstraction_index(abstraction: Abstraction) -> int:
    """Return the stable embedding index of an abstraction."""
    return ABSTRACTION_ORDER.index(abstraction)


def coerce_abstraction(value: object) -> Abstraction:
    """Return the Abstraction member for a member or one of its declared values."""
    if isinstance(value, Abstraction):
        return value
    if isinstance(value, str):
        try:
            return Abstraction(value)
        except ValueError as error:
            raise ContractValueError("abstraction", f"unsupported value {value!r}") from error
    raise ContractValueError("abstraction", "must be an Abstraction or a declared value")


def identity(value: object) -> str:
    """Return a plain namespaced identity from declared semantic fields."""
    if type(value) is not tuple or not value or type(value[0]) is not str:
        raise ContractValueError("identity", "must begin with a string namespace")
    namespace, *fields = value
    declared = json.dumps(
        fields, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )
    return f"{namespace}:{declared}"


def _require_positive_integer(value: int, field: str) -> None:
    if type(value) is not int or value <= 0:
        raise ContractValueError(field, "must be a positive integer")


def _require_nonnegative_integer(value: int, field: str) -> None:
    if type(value) is not int or value < 0:
        raise ContractValueError(field, "must be a nonnegative integer")


def _require_unit_interval(value: float, field: str) -> None:
    if type(value) not in (int, float) or not 0.0 <= float(value) <= 1.0:
        raise ContractValueError(field, "must lie in the closed unit interval")


def _require_nonempty(value: str, field: str) -> None:
    if type(value) is not str or not value.strip():
        raise ContractValueError(field, "must be a nonempty string")


@dataclass(frozen=True, slots=True)
class PredictionPair:
    """One ``(Delta, alpha)`` selection from the joint controller's grid."""

    delta: int
    abstraction: Abstraction

    def __post_init__(self) -> None:
        _require_positive_integer(self.delta, "delta")
        object.__setattr__(self, "abstraction", coerce_abstraction(self.abstraction))

    @property
    def identity(self) -> tuple[int, str]:
        return (self.delta, str(self.abstraction))


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    """Shape and capacity of the context/target encoder."""

    name: str = "conv_gn_v1"
    input_channels: int = 3
    input_height: int = 240
    input_width: int = 320
    stem_channels: int = 64
    stage_channels: tuple[int, ...] = (128, 256, 512)
    blocks_per_stage: int = 2
    group_norm_groups: int = 8
    latent_dim: int = 512
    pool_heads: int = 4

    def __post_init__(self) -> None:
        _require_nonempty(self.name, "encoder name")
        _require_positive_integer(self.input_channels, "input_channels")
        _require_positive_integer(self.input_height, "input_height")
        _require_positive_integer(self.input_width, "input_width")
        _require_positive_integer(self.stem_channels, "stem_channels")
        _require_positive_integer(self.blocks_per_stage, "blocks_per_stage")
        _require_positive_integer(self.group_norm_groups, "group_norm_groups")
        _require_positive_integer(self.latent_dim, "latent_dim")
        _require_positive_integer(self.pool_heads, "pool_heads")
        if self.input_height < 16 or self.input_width < 16:
            raise ContractValueError("encoder input", "must be at least 16 pixels on each axis")
        if type(self.stage_channels) is not tuple or not self.stage_channels:
            raise ContractValueError("stage_channels", "must be a nonempty immutable tuple")
        for channels in self.stage_channels:
            _require_positive_integer(channels, "stage channel count")
        # GroupNorm keeps the encoder batch-independent, which is what makes the
        # EMA/stop-grad target branch well defined.  Divisibility is required.
        for channels in (self.stem_channels, *self.stage_channels):
            if channels % self.group_norm_groups != 0:
                raise ContractValueError(
                    "group_norm_groups", f"must divide every channel count, not {channels}"
                )
        if self.latent_dim % self.pool_heads != 0:
            raise ContractValueError("pool_heads", "must divide latent_dim")

    @property
    def identity(self) -> str:
        return identity(
            (
                "jepa-encoder-config-v1",
                self.name,
                self.input_channels,
                self.input_height,
                self.input_width,
                self.stem_channels,
                self.stage_channels,
                self.blocks_per_stage,
                self.group_norm_groups,
                self.latent_dim,
                self.pool_heads,
            )
        )


@dataclass(frozen=True, slots=True)
class PredictorConfig:
    """Shape and capacity of the dual-output predictor ``F_theta^{Delta,alpha}``."""

    latent_dim: int = 512
    action_dim: int = 5
    hidden_dim: int = 1024
    depth: int = 4
    pair_code_dim: int = 128
    delta_frequency_count: int = 8
    micro_predicate_count: int = len(MICRO_TRANSITION_INPUTS)
    macro_predicate_count: int = len(MACRO_TRANSITION_INPUTS)
    event_type_count: int = 10

    def __post_init__(self) -> None:
        _require_positive_integer(self.latent_dim, "latent_dim")
        _require_positive_integer(self.action_dim, "action_dim")
        _require_positive_integer(self.hidden_dim, "hidden_dim")
        _require_positive_integer(self.depth, "depth")
        _require_positive_integer(self.pair_code_dim, "pair_code_dim")
        _require_positive_integer(self.delta_frequency_count, "delta_frequency_count")
        _require_positive_integer(self.micro_predicate_count, "micro_predicate_count")
        _require_positive_integer(self.macro_predicate_count, "macro_predicate_count")
        _require_positive_integer(self.event_type_count, "event_type_count")

    @property
    def identity(self) -> str:
        return identity(
            (
                "jepa-predictor-config-v2",
                self.latent_dim,
                self.action_dim,
                self.hidden_dim,
                self.depth,
                self.pair_code_dim,
                self.delta_frequency_count,
                self.micro_predicate_count,
                self.macro_predicate_count,
                self.event_type_count,
            )
        )


@dataclass(frozen=True, slots=True)
class JepaConfig:
    """The assembled backbone: encoder, EMA target schedule, and predictor."""

    encoder: EncoderConfig
    predictor: PredictorConfig
    ema_base_momentum: float = 0.996
    ema_final_momentum: float = 1.0

    def __post_init__(self) -> None:
        if type(self.encoder) is not EncoderConfig:
            raise ContractValueError("encoder", "must be an EncoderConfig")
        if type(self.predictor) is not PredictorConfig:
            raise ContractValueError("predictor", "must be a PredictorConfig")
        _require_unit_interval(self.ema_base_momentum, "ema_base_momentum")
        _require_unit_interval(self.ema_final_momentum, "ema_final_momentum")
        if self.ema_base_momentum > self.ema_final_momentum:
            raise ContractValueError(
                "ema_base_momentum", "must not exceed ema_final_momentum"
            )
        if self.encoder.latent_dim != self.predictor.latent_dim:
            raise ContractValueError(
                "latent_dim", "encoder and predictor must agree on the carrier width"
            )

    @property
    def latent_dim(self) -> int:
        return self.encoder.latent_dim

    @property
    def identity(self) -> str:
        return identity(
            (
                "jepa-config-v1",
                self.encoder.identity,
                self.predictor.identity,
                float(self.ema_base_momentum),
                float(self.ema_final_momentum),
            )
        )
