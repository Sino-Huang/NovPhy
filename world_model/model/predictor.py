"""The dual-output predictor ``F_theta^{Delta,alpha}`` (Milestone 1b).

Three structural guarantees hold here, and all are pinned by
``tests/test_world_model_model.py``:

1. **The carrier is always emitted.**  Every ``(Delta, alpha)`` selection
   produces ``z_hat`` of shape ``[B, latent_dim]``.
2. **The carrier is graph-independent of the mode heads.**  ``carrier()`` never
   touches a head, so no head parameter can influence the rollout state.  Head
   gradients still reach the latent — symbols shape ``z`` through the loss —
   but a symbol decode never sits *inside* a rollout step.
3. **Exactly one transition adapter executes.**  Continuous steps retain the
   legacy path; micro and macro steps additionally consume only their selected,
   availability-checked symbolic content before entering the shared predictor.

The ``(Delta, alpha)`` pair conditions a single shared trunk through
AdaLN-Zero-style modulation, from a code that fuses the two axes *jointly*
rather than additively.  This identity signal is separate from the selected
adapter's symbolic content.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from world_model.data.types import ContractValueError
from world_model.model.config import (
    MACRO_TRANSITION_INPUTS,
    MICRO_TRANSITION_INPUTS,
    Abstraction,
    PredictionPair,
    PredictorConfig,
    abstraction_index,
)
from world_model.model.heads import MacroReadout, MacroReadoutHead, MicroReadoutHead


@dataclass(frozen=True, slots=True)
class PredictorOutput:
    """One predictor forward pass.

    ``carrier`` is always present.  ``micro_readout`` and ``macro_readout`` are
    populated only for the abstraction the controller selected, so a caller
    cannot accidentally supervise a mode it did not choose.
    """

    carrier: torch.Tensor
    micro_readout: torch.Tensor | None = None
    macro_readout: MacroReadout | None = None


Relation = tuple[str, str]


def _validate_relations(relations: tuple[Relation, ...], field: str) -> None:
    if type(relations) is not tuple or any(
        type(relation) is not tuple
        or len(relation) != 2
        or any(type(entity) is not str or not entity for entity in relation)
        for relation in relations
    ):
        raise ContractValueError(field, "must contain entity-id relation pairs")


def _availability_kind(availability: str, field: str) -> bool:
    if availability == "available":
        return True
    if isinstance(availability, str) and availability.startswith("unavailable_"):
        return False
    raise ContractValueError(field, "has malformed availability")


@dataclass(frozen=True, slots=True)
class RelationTransitionValue:
    """One relation set with its exact cohort-v2 availability status."""

    availability: str
    relations: tuple[Relation, ...] | None

    def __post_init__(self) -> None:
        available = _availability_kind(self.availability, "relation input")
        if available:
            if self.relations is None:
                raise ContractValueError(
                    "relation input", "available relations must be present"
                )
            _validate_relations(self.relations, "relation input")
        elif self.relations is not None:
            raise ContractValueError(
                "relation input", "unavailable relations must remain unavailable"
            )

    @property
    def available(self) -> bool:
        return self.availability == "available"


@dataclass(frozen=True, slots=True)
class BooleanTransitionValue:
    """One boolean predicate with its exact cohort-v2 availability status."""

    availability: str
    value: bool | None

    def __post_init__(self) -> None:
        available = _availability_kind(self.availability, "boolean input")
        if available and type(self.value) is not bool:
            raise ContractValueError(
                "boolean input", "available predicates must have a boolean value"
            )
        if not available and self.value is not None:
            raise ContractValueError(
                "boolean input", "unavailable predicates must remain unavailable"
            )

    @property
    def available(self) -> bool:
        return self.availability == "available"


@dataclass(frozen=True, slots=True)
class MicroTransitionInput:
    """One cohort-v2 micro state; supports remain supporter-to-supported."""

    frame_record_identity: str
    contact: RelationTransitionValue
    supports: RelationTransitionValue

    def __post_init__(self) -> None:
        if type(self.frame_record_identity) is not str or not self.frame_record_identity:
            raise ContractValueError(
                "micro frame record identity", "must be nonempty"
            )
        if type(self.contact) is not RelationTransitionValue:
            raise ContractValueError("contact", "must be a relation transition value")
        if type(self.supports) is not RelationTransitionValue:
            raise ContractValueError("supports", "must be a relation transition value")


@dataclass(frozen=True, slots=True)
class MacroTransitionInput:
    """One available cohort-v2 macro predicate state."""

    frame_record_identity: str
    steady_state: BooleanTransitionValue
    structure_unstable: BooleanTransitionValue

    def __post_init__(self) -> None:
        if type(self.frame_record_identity) is not str or not self.frame_record_identity:
            raise ContractValueError(
                "macro frame record identity", "must be nonempty"
            )
        if type(self.steady_state) is not BooleanTransitionValue:
            raise ContractValueError(
                "steady-state", "must be a boolean transition value"
            )
        if type(self.structure_unstable) is not BooleanTransitionValue:
            raise ContractValueError(
                "structure-unstable", "must be a boolean transition value"
            )


@dataclass(frozen=True, slots=True)
class MicroTransitionBatch:
    samples: tuple[MicroTransitionInput, ...]

    def __post_init__(self) -> None:
        if type(self.samples) is not tuple or not self.samples or any(
            type(sample) is not MicroTransitionInput for sample in self.samples
        ):
            raise ContractValueError(
                "micro transition batch", "must contain micro transition inputs"
            )


@dataclass(frozen=True, slots=True)
class MacroTransitionBatch:
    samples: tuple[MacroTransitionInput, ...]

    def __post_init__(self) -> None:
        if type(self.samples) is not tuple or not self.samples or any(
            type(sample) is not MacroTransitionInput for sample in self.samples
        ):
            raise ContractValueError(
                "macro transition batch", "must contain macro transition inputs"
            )


ModeTransitionInput = MicroTransitionBatch | MacroTransitionBatch | None


@dataclass(frozen=True, slots=True)
class TransitionRequest:
    """One exclusive pair selection and its mode-specific input batch."""

    pair: PredictionPair
    mode_input: ModeTransitionInput

    def __post_init__(self) -> None:
        if type(self.pair) is not PredictionPair:
            raise ContractValueError("prediction pair", "must be a PredictionPair")
        expected = {
            Abstraction.CONTINUOUS: type(None),
            Abstraction.MICRO: MicroTransitionBatch,
            Abstraction.MACRO: MacroTransitionBatch,
        }[self.pair.abstraction]
        if type(self.mode_input) is not expected:
            raise ContractValueError(
                f"{self.pair.abstraction} transition input",
                "does not match the selected abstraction",
            )


class PairConditioner(nn.Module):
    """Fuse ``(Delta, alpha)`` into one joint adaptive-normalization code."""

    def __init__(self, config: PredictorConfig) -> None:
        super().__init__()
        self._config = config
        self.abstraction_embedding = nn.Embedding(len(Abstraction), config.pair_code_dim)
        delta_dim = 2 * config.delta_frequency_count
        self.fuse = nn.Sequential(
            nn.Linear(delta_dim + config.pair_code_dim, 2 * config.pair_code_dim),
            nn.SiLU(),
            nn.Linear(2 * config.pair_code_dim, config.pair_code_dim),
        )

    @property
    def code_dim(self) -> int:
        return self._config.pair_code_dim

    def delta_features(self, delta: int, device: torch.device) -> torch.Tensor:
        """Return sinusoidal features for a horizon, so unseen Delta interpolate."""
        count = self._config.delta_frequency_count
        exponents = torch.arange(count, dtype=torch.float32, device=device) / count
        frequencies = torch.pow(torch.tensor(10000.0, device=device), -exponents)
        angles = float(delta) * frequencies
        return torch.cat((torch.sin(angles), torch.cos(angles)))

    def code(
        self, pair: PredictionPair, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Return the joint conditioning code, broadcast over the batch."""
        if type(pair) is not PredictionPair:
            raise ContractValueError("prediction pair", "must be a PredictionPair")
        if type(batch_size) is not int or batch_size <= 0:
            raise ContractValueError("batch_size", "must be a positive integer")
        index = torch.tensor(abstraction_index(pair.abstraction), device=device)
        fused = self.fuse(
            torch.cat((self.delta_features(pair.delta, device), self.abstraction_embedding(index)))
        )
        return fused.unsqueeze(0).expand(batch_size, -1)


class FiLMBlock(nn.Module):
    """A residual MLP block with zero-initialized adaptive LayerNorm modulation."""

    def __init__(self, width: int, code_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.modulation = nn.Linear(code_dim, 2 * width)
        self.inner = nn.Linear(width, width)
        self.outer = nn.Linear(width, width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, hidden: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(code).chunk(2, dim=-1)
        modulated = self.norm(hidden) * (1.0 + scale) + shift
        return hidden + self.outer(F.silu(self.inner(modulated)))


class ContinuousTransitionAdapter(nn.Module):
    """Retain the legacy continuous hidden state without symbolic content."""

    def forward(
        self, hidden: torch.Tensor, mode_input: ModeTransitionInput
    ) -> torch.Tensor:
        if mode_input is not None:
            raise ContractValueError(
                "continuous transition input", "must not contain symbolic content"
            )
        return hidden


class MicroTransitionAdapter(nn.Module):
    """Inject available contact and directed-support relation content."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.entity_embedding = nn.Embedding(256, hidden_dim)
        self.contact_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.supporter_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.supported_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.availability_projection = nn.Linear(2, hidden_dim, bias=False)

    def _entity(self, identity: str, device: torch.device) -> torch.Tensor:
        encoded = torch.tensor(tuple(identity.encode("utf-8")), device=device)
        return self.entity_embedding(encoded).mean(dim=0)

    def forward(
        self, hidden: torch.Tensor, mode_input: ModeTransitionInput
    ) -> torch.Tensor:
        if type(mode_input) is not MicroTransitionBatch:
            raise ContractValueError(
                "micro transition input", "must be a micro transition batch"
            )
        if len(mode_input.samples) != hidden.shape[0]:
            raise ContractValueError(
                "micro transition input", "must match the latent batch size"
            )
        encoded_samples = []
        for sample in mode_input.samples:
            entities: dict[str, torch.Tensor] = {}

            def entity(identity: str) -> torch.Tensor:
                if identity not in entities:
                    entities[identity] = self._entity(identity, hidden.device)
                return entities[identity]

            encoded = torch.zeros_like(hidden[0])
            if sample.contact.available:
                assert sample.contact.relations is not None
                for first, second in sample.contact.relations:
                    encoded = encoded + self.contact_projection(
                        entity(first) + entity(second)
                    )
            if sample.supports.available:
                assert sample.supports.relations is not None
                for supporter, supported in sample.supports.relations:
                    encoded = (
                        encoded
                        + self.supporter_projection(entity(supporter))
                        + self.supported_projection(entity(supported))
                    )
            availability = torch.tensor(
                (sample.contact.available, sample.supports.available),
                dtype=hidden.dtype,
                device=hidden.device,
            )
            encoded_samples.append(
                encoded + self.availability_projection(availability)
            )
        return hidden + torch.stack(encoded_samples)


class MacroTransitionAdapter(nn.Module):
    """Inject the two available central macro predicates."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(4, hidden_dim, bias=False)

    def forward(
        self, hidden: torch.Tensor, mode_input: ModeTransitionInput
    ) -> torch.Tensor:
        if type(mode_input) is not MacroTransitionBatch:
            raise ContractValueError(
                "macro transition input", "must be a macro transition batch"
            )
        if len(mode_input.samples) != hidden.shape[0]:
            raise ContractValueError(
                "macro transition input", "must match the latent batch size"
            )
        features = torch.tensor(
            [
                (
                    sample.steady_state.value or False,
                    sample.structure_unstable.value or False,
                    sample.steady_state.available,
                    sample.structure_unstable.available,
                )
                for sample in mode_input.samples
            ],
            dtype=hidden.dtype,
            device=hidden.device,
        )
        return hidden + self.projection(features)


class DualOutputPredictor(nn.Module):
    """Predict the carrier for a selected pair, with mode-gated symbolic readouts."""

    def __init__(
        self,
        config: PredictorConfig,
        *,
        conditioner: nn.Module | None = None,
        micro_head: nn.Module | None = None,
        macro_head: nn.Module | None = None,
        continuous_adapter: nn.Module | None = None,
        micro_adapter: nn.Module | None = None,
        macro_adapter: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if type(config) is not PredictorConfig:
            raise ContractValueError("predictor config", "must be a PredictorConfig")
        self._config = config

        self.conditioner = conditioner if conditioner is not None else PairConditioner(config)
        self.input_projection = nn.Linear(
            config.latent_dim + config.action_dim, config.hidden_dim
        )
        self.blocks = nn.ModuleList(
            FiLMBlock(config.hidden_dim, config.pair_code_dim) for _ in range(config.depth)
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.output_projection = nn.Linear(config.hidden_dim, config.latent_dim)

        self.micro_head = (
            micro_head
            if micro_head is not None
            else MicroReadoutHead(
                config.latent_dim, config.hidden_dim, config.micro_predicate_count
            )
        )
        self.macro_head = (
            macro_head
            if macro_head is not None
            else MacroReadoutHead(
                config.latent_dim,
                config.hidden_dim,
                config.macro_predicate_count,
                config.event_type_count,
            )
        )
        # Construct adapters after the legacy modules so a seeded continuous
        # predictor retains its prior initialization and forward behavior.
        self.continuous_adapter = (
            continuous_adapter
            if continuous_adapter is not None
            else ContinuousTransitionAdapter()
        )
        self.micro_adapter = (
            micro_adapter
            if micro_adapter is not None
            else MicroTransitionAdapter(config.hidden_dim)
        )
        self.macro_adapter = (
            macro_adapter
            if macro_adapter is not None
            else MacroTransitionAdapter(config.hidden_dim)
        )

    @property
    def config(self) -> PredictorConfig:
        return self._config

    def carrier(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        pair: PredictionPair | TransitionRequest,
    ) -> torch.Tensor:
        """Return ``z_hat`` alone — the only quantity a rollout step may carry."""
        selected = self._request(pair)
        selected_pair = selected.pair
        self._validate(latent, action, selected_pair)
        code = self.conditioner.code(selected_pair, latent.shape[0], latent.device)
        hidden = self.input_projection(torch.cat((latent, action), dim=-1))
        if selected_pair.abstraction is Abstraction.CONTINUOUS:
            hidden = self.continuous_adapter(hidden, selected.mode_input)
        elif selected_pair.abstraction is Abstraction.MICRO:
            hidden = self.micro_adapter(hidden, selected.mode_input)
        else:
            hidden = self.macro_adapter(hidden, selected.mode_input)
        for block in self.blocks:
            hidden = block(hidden, code)
        return latent + self.output_projection(self.output_norm(hidden))

    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        pair: PredictionPair | TransitionRequest,
    ) -> PredictorOutput:
        """Return the carrier plus the readout for the selected abstraction only."""
        selected = self._request(pair)
        selected_pair = selected.pair
        carrier = self.carrier(latent, action, selected)
        micro = (
            self.micro_head(carrier)
            if selected_pair.abstraction is Abstraction.MICRO
            else None
        )
        macro = (
            self.macro_head(carrier)
            if selected_pair.abstraction is Abstraction.MACRO
            else None
        )
        return PredictorOutput(carrier=carrier, micro_readout=micro, macro_readout=macro)

    def rollout(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        pairs: Sequence[PredictionPair | TransitionRequest],
    ) -> tuple[torch.Tensor, ...]:
        """Chain carrier to carrier across a pair sequence, touching no head."""
        if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
            raise ContractValueError("requests", "must be a transition request sequence")
        if not pairs:
            raise ContractValueError("requests", "must not be empty")
        carriers: list[torch.Tensor] = []
        current = latent
        for pair in pairs:
            current = self.carrier(current, action, pair)
            carriers.append(current)
        return tuple(carriers)

    @staticmethod
    def _request(request: PredictionPair | TransitionRequest) -> TransitionRequest:
        if type(request) is TransitionRequest:
            return request
        if type(request) is PredictionPair:
            if request.abstraction is not Abstraction.CONTINUOUS:
                raise ContractValueError(
                    f"{request.abstraction} transition input",
                    "must be supplied in a TransitionRequest",
                )
            return TransitionRequest(request, None)
        raise ContractValueError(
            "transition request", "must be a PredictionPair or TransitionRequest"
        )

    def _validate(
        self, latent: torch.Tensor, action: torch.Tensor, pair: PredictionPair
    ) -> None:
        if type(pair) is not PredictionPair:
            raise ContractValueError("prediction pair", "must be a PredictionPair")
        if not isinstance(latent, torch.Tensor):
            raise ContractValueError("latent", "must be a torch tensor")
        if not isinstance(action, torch.Tensor):
            raise ContractValueError("action", "must be a torch tensor")
        if latent.ndim != 2 or latent.shape[1] != self._config.latent_dim:
            raise ContractValueError(
                "latent", f"must have shape [B, {self._config.latent_dim}]"
            )
        if action.ndim != 2 or action.shape[1] != self._config.action_dim:
            raise ContractValueError(
                "action", f"must have shape [B, {self._config.action_dim}]"
            )
        if action.shape[0] != latent.shape[0]:
            raise ContractValueError("action", "must match the latent batch size")
