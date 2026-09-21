"""External temporal-adaptation baseline ports onto the frozen carrier (issue #78).

Both ports keep the frozen ~1.8M-parameter ContinuousDynamics carrier
(`world_model.training.matched_dynamics`, capacity contract width 456,
1,862,396 parameters) and add NO new learned component.

TAWM (arXiv 2506.08441; official code github.com/anh-nn01/Time-Aware-World-Model
consulted at pinned commit ffb61f8e2bcdb0030cb4a7175e0b782cdad9af4c, reference
only, never run at its original scale): a shared-parameter continuous predictor
conditioned on the transition length, trained over a mixed-length distribution.
The reference README states the architecture-agnostic recipe as (1) condition
the dynamics on the time step and (2) train over a mixture of time steps.  Our
carrier's HorizonConditioner already realizes length-conditioned shared
dynamics, so the port is the training schedule alone.  Declared deviations,
frozen in the issue-78 plan before any outcome: (i) TAWM's sampling-interval
conditioning is replaced by h-conditioning on the frozen grid {1,5,15}
(observed-frame units; 1 frame = 50 native steps); (ii) TAWM's
LogUniform(delta-t) sampler is replaced by the uniform distribution over that
grid, realized as deterministic round-robin (targets exist only at grid
offsets); (iii) evaluation is (a) fixed per-h and (b) with h chosen by the
existing learned horizon-controller machinery of the repo's `continuous
adaptive` arm (MatchedController(pure=True), DP-utility labels), no new
selector.

VLWM (arXiv 2606.21775): NO public code exists (7-way no-code search recorded
in the frozen issue-78 plan); this is a `paper-fidelity port, no reference code
consulted`, implemented from paper Eqs. 3-6 only.  Eq. 3: direct prediction of
the future latent z_{t+k} conditioned on the current latent and the action
sequence, trained with squared error against the (observed) target latent.
Eq. 4: the conditioning context; we instantiate the minimal L=1 context
{s_t} plus the action sequence, matching the carrier interface.  Eqs. 5-6: the
stage-wise curriculum j(tau) = min(K_max, ceil(K_max * tau / T)) with
p_tau(k) = Uniform({1..j(tau)}), ported over the frozen grid as the ordered
stages {1}, {1,5}, {1,5,15}.  Single-shot mapping (declared deviation): the
paper's action sequence a_{t:t+k-1} is the executed launch action followed by
k-1 null actions; under the carrier's single 5-value action slot the sequence
collapses to the launch action (null actions are the zero vector), and the
sequence length k enters through the carrier's horizon conditioning.  The
paper's stop-gradient target encoder is unnecessary here because targets are
observed carriers, not encoder outputs.
"""
from __future__ import annotations

import fcntl
from contextlib import contextmanager
from collections import Counter

import torch
from torch.nn import functional as F

from world_model.model import Abstraction, PredictionPair
from world_model.training.cnn_hybrid import DIM
from world_model.training.matched_dynamics import ContinuousDynamics

DELTAS = (1, 5, 15)
GPU_LOCK_PATH = "/tmp/novphy-addexp-gpu.lock"


def tawm_horizon(step):
    """Uniform mixed-h schedule: deterministic round-robin over the frozen grid."""
    return DELTAS[step % len(DELTAS)]


def tawm_horizon_counts(steps):
    return {h: steps // len(DELTAS) + (1 if i < steps % len(DELTAS) else 0)
            for i, h in enumerate(DELTAS)}


def vlwm_stage(update, total):
    """VLWM Eq. 5 with K_max = 3 and the frozen grid as stages; update is 1-based."""
    if not 1 <= update <= total:
        raise ValueError("curriculum update index outside the scheduled budget")
    return min(len(DELTAS), -(-len(DELTAS) * update // total))


def vlwm_horizon(step, total):
    """VLWM Eqs. 5-6: uniform over the first j(tau) grid stages, round-robin."""
    eligible = DELTAS[:vlwm_stage(step + 1, total)]
    return eligible[step % len(eligible)]


def vlwm_horizon_counts(total):
    counts = Counter(vlwm_horizon(step, total) for step in range(total))
    return {h: counts[h] for h in DELTAS}


def vlwm_loss(model, z, action, length, k):
    """VLWM Eq. 3 port: direct prediction of z_{t+k} from every valid in-window start.

    The expectation over t of Eq. 3 is realized as all window starts t with a
    complete k-offset target (length >= t + k, the carrier's masking rule); the
    action sequence is the single-shot mapping (launch action; k-1 null actions
    collapse in the carrier's 5-value slot); k conditions the horizon code.
    R(s) latent regularization uses the carrier's standard .01 bound penalty.
    """
    if k not in DELTAS:
        raise ValueError("VLWM predicts only on the frozen grid")
    size, frames, _ = z.shape
    positions = frames - k
    if positions < 1:
        raise ValueError("window shorter than the requested horizon")
    pair = PredictionPair(k, Abstraction.CONTINUOUS)
    contexts = z[:, :positions].reshape(-1, DIM)
    actions = action[:, None].expand(-1, positions, -1).reshape(-1, 5)
    predicted = model.carrier(contexts, actions, pair).reshape(size, positions, DIM)
    error = (predicted - z[:, k:]).square().mean(-1)
    keep = (length >= torch.arange(k, k + positions, device=length.device)[:, None]).T.float()
    if not bool((keep.sum() > 0).all()):
        raise ValueError("VLWM batch lacks a complete k-offset transition")
    bound = (F.relu(predicted.abs() - 2).square().mean(-1)) * keep
    return ((error * keep).sum() / keep.sum()
            + .01 * bound.sum() / keep.sum())


@contextmanager
def gpu_exclusive(path=GPU_LOCK_PATH):
    """Hold an exclusive advisory lock on the shared GPU for a whole measured phase.

    The file descriptor is opened once per process and kept open for the
    process lifetime; only the flock is acquired/released per phase.
    """
    global _GPU_LOCK_FD
    try:
        fd = _GPU_LOCK_FD
    except NameError:
        fd = _GPU_LOCK_FD = open(path, "w")
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        yield fd
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
