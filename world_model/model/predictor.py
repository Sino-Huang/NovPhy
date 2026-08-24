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
   legacy path; micro and macro steps additionally consume only their selected
   encoded symbolic content before entering the shared predictor.

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
        self, hidden: torch.Tensor, mode_input: torch.Tensor | None
    ) -> torch.Tensor:
        if mode_input is not None:
            raise ContractValueError(
                "continuous transition input", "must not contain symbolic content"
            )
        return hidden


class SymbolicTransitionAdapter(nn.Module):
    """Inject one selected mode's encoded symbolic content into the shared trunk."""

    def __init__(self, abstraction: Abstraction, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self._abstraction = abstraction
        self._input_dim = input_dim
        self.projection = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(
        self, hidden: torch.Tensor, mode_input: torch.Tensor | None
    ) -> torch.Tensor:
        field = f"{self._abstraction} transition input"
        if not isinstance(mode_input, torch.Tensor):
            raise ContractValueError(field, "must be an encoded torch tensor")
        if mode_input.ndim != 2 or mode_input.shape[1] != self._input_dim:
            raise ContractValueError(
                field, f"must have shape [B, {self._input_dim}]"
            )
        if mode_input.shape[0] != hidden.shape[0]:
            raise ContractValueError(field, "must match the latent batch size")
        return hidden + self.projection(mode_input)


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
            else SymbolicTransitionAdapter(
                Abstraction.MICRO,
                config.micro_predicate_count,
                config.hidden_dim,
            )
        )
        self.macro_adapter = (
            macro_adapter
            if macro_adapter is not None
            else SymbolicTransitionAdapter(
                Abstraction.MACRO,
                config.macro_predicate_count,
                config.hidden_dim,
            )
        )

    @property
    def config(self) -> PredictorConfig:
        return self._config

    def carrier(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        pair: PredictionPair,
        mode_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``z_hat`` alone — the only quantity a rollout step may carry."""
        self._validate(latent, action, pair)
        code = self.conditioner.code(pair, latent.shape[0], latent.device)
        hidden = self.input_projection(torch.cat((latent, action), dim=-1))
        if pair.abstraction is Abstraction.CONTINUOUS:
            hidden = self.continuous_adapter(hidden, mode_input)
        elif pair.abstraction is Abstraction.MICRO:
            hidden = self.micro_adapter(hidden, mode_input)
        else:
            hidden = self.macro_adapter(hidden, mode_input)
        for block in self.blocks:
            hidden = block(hidden, code)
        return latent + self.output_projection(self.output_norm(hidden))

    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        pair: PredictionPair,
        mode_input: torch.Tensor | None = None,
    ) -> PredictorOutput:
        """Return the carrier plus the readout for the selected abstraction only."""
        carrier = self.carrier(latent, action, pair, mode_input)
        micro = self.micro_head(carrier) if pair.abstraction is Abstraction.MICRO else None
        macro = self.macro_head(carrier) if pair.abstraction is Abstraction.MACRO else None
        return PredictorOutput(carrier=carrier, micro_readout=micro, macro_readout=macro)

    def rollout(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        pairs: Sequence[PredictionPair],
        *,
        mode_inputs: Sequence[torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Chain carrier to carrier across a pair sequence, touching no head."""
        if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
            raise ContractValueError("pairs", "must be a sequence of PredictionPair")
        if not pairs:
            raise ContractValueError("pairs", "must not be empty")
        if mode_inputs is None:
            selected_inputs: Sequence[torch.Tensor | None] = (None,) * len(pairs)
        elif (
            not isinstance(mode_inputs, Sequence)
            or isinstance(mode_inputs, (str, bytes))
            or len(mode_inputs) != len(pairs)
        ):
            raise ContractValueError(
                "mode_inputs", "must match the pair sequence length"
            )
        else:
            selected_inputs = mode_inputs
        carriers: list[torch.Tensor] = []
        current = latent
        for pair, mode_input in zip(pairs, selected_inputs, strict=True):
            current = self.carrier(current, action, pair, mode_input)
            carriers.append(current)
        return tuple(carriers)

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
