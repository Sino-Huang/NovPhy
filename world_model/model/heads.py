"""Mode-head readouts and the reliability weight ``omega_psi``.

These heads are the symbolic side of the dual-output predictor.  They read the
carrier and enter only the loss; nothing they produce is ever fed back into a
rollout step.  See ``world_model.model.predictor`` for the structural guarantee.

They are wired and shape-tested now but **inactive during Milestone 1 training**:
the legacy RGB cohort carries no symbolic labels.  They activate through the
``PhysicsSupervisionRequest`` seam once the enriched physics cohort exists.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from world_model.data.types import ContractValueError
from world_model.model.config import Abstraction, coerce_abstraction


@dataclass(frozen=True, slots=True)
class MacroReadout:
    """The macro-event triple ``(S^M, Delta, e)`` of proposal section 4.3."""

    macro_logits: torch.Tensor
    delta_hat: torch.Tensor
    event_logits: torch.Tensor


class MicroReadoutHead(nn.Module):
    """Decode object-level micro predicates ``S^mu`` from the carrier."""

    def __init__(self, latent_dim: int, hidden_dim: int, predicate_count: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, predicate_count),
        )

    def forward(self, carrier: torch.Tensor) -> torch.Tensor:
        return self.body(carrier)


class MacroReadoutHead(nn.Module):
    """Decode the structure-level macro state, its duration, and its event."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        macro_predicate_count: int,
        event_type_count: int,
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
        )
        self.macro_projection = nn.Linear(hidden_dim, macro_predicate_count)
        self.delta_projection = nn.Linear(hidden_dim, 1)
        self.event_projection = nn.Linear(hidden_dim, event_type_count)

    def forward(self, carrier: torch.Tensor) -> MacroReadout:
        hidden = self.body(carrier)
        return MacroReadout(
            macro_logits=self.macro_projection(hidden),
            delta_hat=self.delta_projection(hidden),
            event_logits=self.event_projection(hidden),
        )


def mode_weight(abstraction: Abstraction, reliability: float | torch.Tensor) -> float | torch.Tensor:
    """Return ``omega_psi(h, alpha)`` from proposal section 4.

    ``0`` for a continuous step, the reliability estimate ``r_psi`` for micro
    relational constraints, and ``1`` for selected macro-event supervision.  A
    masked term therefore contributes exactly zero gradient rather than being
    silently dropped from the objective.
    """
    selected = coerce_abstraction(abstraction)
    _require_reliability(reliability)
    if selected is Abstraction.CONTINUOUS:
        return 0.0
    if selected is Abstraction.MACRO:
        return 1.0
    return reliability


def _require_reliability(value: float | torch.Tensor) -> None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            raise ContractValueError("reliability", "must not be empty")
        if bool(((value < 0.0) | (value > 1.0)).any()):
            raise ContractValueError("reliability", "must lie in the closed unit interval")
        return
    if type(value) not in (int, float) or not 0.0 <= float(value) <= 1.0:
        raise ContractValueError("reliability", "must lie in the closed unit interval")
