"""Mode-head readouts and the reliability weight ``omega_psi``.

These heads are the symbolic side of the dual-output predictor.  They read the
carrier and enter only the loss; nothing they produce is ever fed back into a
rollout step.  See ``world_model.model.predictor`` for the structural guarantee.

They remain inactive in the legacy continuous-only training loop. The validated
cohort-v2 reader now supplies symbolic supervision; the micro and macro training
paths activate these heads separately in issues #5 and #6.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from world_model.data.types import ContractValueError
from world_model.model.config import Abstraction, coerce_abstraction


@dataclass(frozen=True, slots=True)
class MacroReadout:
    """The macro-event triple ``(S^M, Delta, e)`` of proposal section 4.3."""

    macro_logits: torch.Tensor
    delta_hat: torch.Tensor
    event_logits: torch.Tensor


class MicroReadoutHead(nn.Module):
    """Decode aggregate and entity-pair micro predicates from the carrier.

    ``forward`` retains the original aggregate-logit contract.  Cohort-v2
    supervision uses :meth:`relation_logits`, which scores the exact entity-ID
    pairs supplied by the oracle target.  Contact queries are symmetric while
    support queries retain supporter-to-supported direction.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, predicate_count: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, predicate_count),
        )
        self.relation_carrier = nn.Linear(latent_dim, hidden_dim)
        self.entity_embedding = nn.Embedding(256, hidden_dim)
        self.contact_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.supporter_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.supported_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.contact_score = nn.Linear(hidden_dim, 1)
        self.support_score = nn.Linear(hidden_dim, 1)

    def forward(self, carrier: torch.Tensor) -> torch.Tensor:
        aggregate = self.body(carrier)
        # Keep the historical masked-head contract: when a selected micro loss
        # is multiplied by zero, every parameter in that head receives an
        # explicit zero gradient rather than an absent gradient. Exact relation
        # supervision below supplies the nonzero path for these parameters.
        relation_zero = sum(
            parameter.sum() * 0.0
            for name, parameter in self.named_parameters()
            if not name.startswith("body.")
        )
        return aggregate + relation_zero

    def _entity(self, identity: str, device: torch.device) -> torch.Tensor:
        if type(identity) is not str or not identity:
            raise ContractValueError("relation query", "entity ids must be nonempty")
        encoded = torch.tensor(tuple(identity.encode("utf-8")), device=device)
        return self.entity_embedding(encoded).mean(dim=0)

    def relation_logits(
        self,
        carrier: torch.Tensor,
        predicate: str,
        queries: tuple[tuple[tuple[str, str], ...], ...],
    ) -> tuple[torch.Tensor, ...]:
        """Score variable-length entity-pair queries for one micro predicate."""
        if (
            not isinstance(carrier, torch.Tensor)
            or carrier.ndim != 2
            or type(queries) is not tuple
            or len(queries) != carrier.shape[0]
        ):
            raise ContractValueError(
                "relation queries", "must match a two-dimensional carrier batch"
            )
        if predicate not in ("contact", "supports"):
            raise ContractValueError(
                "micro predicate", "must be contact or supports"
            )
        carrier_features = self.relation_carrier(carrier)
        results = []
        entities: dict[str, torch.Tensor] = {}

        def entity(identity: str) -> torch.Tensor:
            if identity not in entities:
                entities[identity] = self._entity(identity, carrier.device)
            return entities[identity]

        for index, sample_queries in enumerate(queries):
            if type(sample_queries) is not tuple or any(
                type(query) is not tuple
                or len(query) != 2
                or any(type(entity) is not str or not entity for entity in query)
                for query in sample_queries
            ):
                raise ContractValueError(
                    "relation queries", "must contain entity-id pairs"
                )
            if sample_queries:
                first_entities = torch.stack(
                    tuple(entity(first) for first, _second in sample_queries)
                )
                second_entities = torch.stack(
                    tuple(entity(second) for _first, second in sample_queries)
                )
                if predicate == "contact":
                    relation = self.contact_projection(
                        first_entities + second_entities
                    )
                    score = self.contact_score
                else:
                    relation = (
                        self.supporter_projection(first_entities)
                        + self.supported_projection(second_entities)
                    )
                    score = self.support_score
                results.append(
                    score(
                        F.silu(carrier_features[index].unsqueeze(0) + relation)
                    ).squeeze(-1)
                )
            else:
                results.append(carrier_features[index, :0])
        return tuple(results)


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
