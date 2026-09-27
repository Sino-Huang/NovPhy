"""Deterministic scoring harness shared by every scoring runner from issue #111 on.

Float noise in scoring forced two post-hoc guard revisions (#99 v2, #100 v2). This
module holds the three shared pieces that prevent a third:

- ``deterministic_scoring()``: cuDNN autotuning off, deterministic cuDNN kernels and
  TF32 off for convolutions and matmuls. Wrap every ``--run`` scoring phase and every
  ``--validate`` in it. Training may keep autotuning, but in a separate process from
  scoring (#100 G3: an autotuned training process scored the #96 arms with differently
  rounded kernels, and Delta = 1 rollouts over 225 steps amplified the noise).
- Replication tolerances: ``carrier_replication`` holds non-motion carrier values to
  the parse tolerance and motion values to 2 x tolerance / elapsed seconds (#100 G1: a
  motion value is a center difference divided by the frame interval, which is a few
  ms on the terminal frame of an early-ending shot). ``endpoint_replication`` compares
  parsed quantities, never the 1000x-weighted count (#99 G3: a 1.34e-4 presence
  difference read as 0.134).
- Scoring default: the tie-free cost is primary and the count cost is secondary
  (``PRIMARY_COST``, ``SECONDARY_COST``, ``EndpointCosts``).
"""
from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Final, Iterable, Iterator, Mapping

import torch

from world_model.data.deployment_temporal import ENTITY_FEATURES
from world_model.planning.task_objective import TaskObjective

# carrier layout (TemporalVisualCarrierAdapter.build_from_parsed): column 0 has-prior flag,
# column 1 elapsed seconds between prior and current frame, then 13 values per slot
CARRIER_HEADER: Final = 2
ELAPSED_COLUMN: Final = 1
SLOT_WIDTH: Final = len(ENTITY_FEATURES)
PRESENCE_FEATURE: Final = ENTITY_FEATURES.index("presence_probability")
CENTER_FEATURES: Final = (ENTITY_FEATURES.index("center_x_normalized"),
                          ENTITY_FEATURES.index("center_y_normalized"))
MOTION_FEATURES: Final = (ENTITY_FEATURES.index("motion_x_normalized_per_second"),
                          ENTITY_FEATURES.index("motion_y_normalized_per_second"))

PARSE_TOLERANCE: Final = 1e-3  # parse-batch float noise of one parsed value (#99 smoke, #100 freeze)

PRIMARY_COST: Final = "tie_free"
SECONDARY_COST: Final = "count"
COSTS: Final = (PRIMARY_COST, SECONDARY_COST)
DISPLACEMENT_WEIGHT: Final = 0.1
COST_DEFINITIONS: Final = {
    PRIMARY_COST: (f"pig presence (unclamped) - {DISPLACEMENT_WEIGHT} * Euclidean displacement of the pig "
                   "slot center (normalized screen units) between the start carrier and the endpoint "
                   "carrier; lower is better"),
    SECONDARY_COST: ("frozen TaskObjective: 1000 * clamp(pig presence, 0, 1) + sum over block slots of "
                     "clamp(block presence, 0, 1)"),
}
# what a replication guard compares; the count cost is excluded on purpose
PARSED_QUANTITIES: Final = ("pig_presence", "pig_displacement", PRIMARY_COST)

SCORING_POLICY: Final = {"cudnn_benchmark": False, "cudnn_deterministic": True,
                         "cudnn_allow_tf32": False, "matmul_allow_tf32": False}


def _flags() -> dict[str, bool]:
    return {"cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}


def _set_flags(flags: Mapping[str, bool]) -> None:
    torch.backends.cudnn.benchmark = flags["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = flags["cudnn_deterministic"]
    torch.backends.cudnn.allow_tf32 = flags["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = flags["matmul_allow_tf32"]


@contextmanager
def deterministic_scoring() -> Iterator[dict[str, bool]]:
    """Score under ``SCORING_POLICY``; yields the policy for the record; restores the prior flags.

    Usable as ``with deterministic_scoring() as policy:`` or as a decorator on a mode function."""
    saved = _flags()
    _set_flags(SCORING_POLICY)
    try:
        yield dict(SCORING_POLICY)
    finally:
        _set_flags(saved)


def motion_columns(width: int) -> list[int]:
    """Carrier columns holding motion (center difference / elapsed seconds) values."""
    slots = (width - CARRIER_HEADER) // SLOT_WIDTH
    return [CARRIER_HEADER + SLOT_WIDTH * slot + feature for slot in range(slots) for feature in MOTION_FEATURES]


def carrier_replication(reference: torch.Tensor, fresh: torch.Tensor,
                        tolerance: float = PARSE_TOLERANCE) -> dict:
    """Compare carrier rows (N x width, or one row) of two parses of the same frames.

    Non-motion values must agree within ``tolerance``. A motion value is the difference of
    two parsed centers, each within ``tolerance``, divided by the frame interval (the
    reference row's elapsed column), so it must agree within 2 x tolerance / elapsed;
    the check is made as |delta| x elapsed / 2 <= tolerance, which is 0 where no prior
    frame exists (elapsed 0, motion unavailable)."""
    reference, fresh = torch.atleast_2d(reference).double(), torch.atleast_2d(fresh).double()
    if reference.shape != fresh.shape:
        raise ValueError(f"carrier shapes differ: {tuple(reference.shape)} vs {tuple(fresh.shape)}")
    delta = (reference - fresh).abs()
    motion = torch.zeros(delta.shape[-1], dtype=torch.bool)
    motion[motion_columns(delta.shape[-1])] = True
    elapsed = reference[:, ELAPSED_COLUMN]
    nonmotion = float(delta[:, ~motion].max()) if len(delta) else 0.0
    scaled = float((delta[:, motion] * elapsed[:, None] / 2).max()) if len(delta) else 0.0
    return {"rows": len(delta), "max_abs_nonmotion_delta": nonmotion,
            "max_motion_delta_times_elapsed_over_2": scaled, "tolerance": tolerance,
            "pass": nonmotion <= tolerance and scaled <= tolerance}


def tie_free(presence, displacement):
    """The primary cost; works on floats and on tensors (differentiable)."""
    return presence - DISPLACEMENT_WEIGHT * displacement


class EndpointCosts:
    """Both costs and their parsed quantities from one endpoint carrier and its start carrier."""

    def __init__(self, objective: TaskObjective) -> None:
        if len(objective.pig_slots) != 1:
            raise ValueError("the tie-free cost is declared for exactly one pig slot")
        self.objective = objective
        base = CARRIER_HEADER + SLOT_WIDTH * objective.pig_slots[0]
        self.presence = base + PRESENCE_FEATURE
        self.center = (base + CENTER_FEATURES[0], base + CENTER_FEATURES[1])

    def row(self, ordinal: int, z: torch.Tensor, z0: torch.Tensor) -> dict:
        """A nonfinite endpoint is a retained typed failure (excluded, every value None)."""
        z = z.detach()
        if not bool(torch.isfinite(z).all()):
            return {"ordinal": ordinal, "excluded": True, PRIMARY_COST: None, SECONDARY_COST: None,
                    "pig_presence": None, "pig_displacement": None}
        presence = float(z[self.presence])
        displacement = math.hypot(float(z[self.center[0]]) - float(z0[self.center[0]]),
                                  float(z[self.center[1]]) - float(z0[self.center[1]]))
        return {"ordinal": ordinal, "excluded": False, PRIMARY_COST: tie_free(presence, displacement),
                SECONDARY_COST: float(self.objective(z)), "pig_presence": presence,
                "pig_displacement": displacement}


def endpoint_replication(reference: Iterable[Mapping], fresh: Iterable[Mapping],
                         tolerance: float = PARSE_TOLERANCE) -> dict:
    """Compare ``EndpointCosts`` rows (matched by ordinal) on ``PARSED_QUANTITIES`` only.

    A candidate excluded in one set and scored in the other, or an ordinal present in only
    one set, fails the guard (delta inf)."""
    mine = {row["ordinal"]: row for row in fresh}
    theirs = {row["ordinal"]: row for row in reference}
    worst = dict.fromkeys(PARSED_QUANTITIES, 0.0)
    for ordinal in sorted(mine.keys() | theirs.keys()):
        a, b = theirs.get(ordinal), mine.get(ordinal)
        if a is None or b is None or a["excluded"] != b["excluded"]:
            worst = dict.fromkeys(PARSED_QUANTITIES, math.inf)
            break
        if a["excluded"]:
            continue
        for key in PARSED_QUANTITIES:
            worst[key] = max(worst[key], abs(a[key] - b[key]))
    return {"candidates": len(theirs), "max_abs_delta": worst, "tolerance": tolerance,
            "pass": all(value <= tolerance for value in worst.values())}


__all__ = [
    "COSTS", "COST_DEFINITIONS", "DISPLACEMENT_WEIGHT", "EndpointCosts", "PARSED_QUANTITIES",
    "PARSE_TOLERANCE", "PRIMARY_COST", "SCORING_POLICY", "SECONDARY_COST", "carrier_replication",
    "deterministic_scoring", "endpoint_replication", "motion_columns", "tie_free",
]
