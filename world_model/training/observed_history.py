"""Shared causal observation/action memory for the #76 successor candidate.

This module does not accept engine events, object identities or hit counters.
Each row is an observation event, an executed-action event, or padding. An
action being considered by a planner is not an executed-action event.
"""
from dataclasses import dataclass

import torch
from torch import nn

from world_model.training.cnn_hybrid import DIM

HISTORY_DIM = 64
ACTION_DIM = 5


@dataclass(frozen=True)
class HistoryState:
    memory: torch.Tensor
    last_timestamp: torch.Tensor
    seen_event: torch.Tensor


class ObservedHistoryEncoder(nn.Module):
    """One common encoder checkpoint is shared by both independently fitted arms.

    Timestamps are seconds on the agent observation clock. Missing observations
    do not masquerade as zero-valued observations: two masks distinguish them
    from executed actions and padding. An action-only event still updates memory.
    Passing the returned state continues the episode; omitting it starts a new
    episode. Planning branches must not replace the real episode's state.
    """
    def __init__(self, carrier_dim=DIM):
        super().__init__()
        self.carrier_dim = carrier_dim
        self.cell = nn.GRUCell(carrier_dim + ACTION_DIM + 3, HISTORY_DIM)

    def forward(self, carriers, actions, timestamps, observation_mask, action_mask, state=None):
        batch, steps, width = carriers.shape
        if width != self.carrier_dim or actions.shape != (batch, steps, ACTION_DIM):
            raise ValueError(f"history requires {self.carrier_dim}-carriers and five-component executed actions")
        if any(value.shape != (batch, steps) for value in (timestamps, observation_mask, action_mask)):
            raise ValueError("history timestamps and event masks must align with the event sequence")
        if observation_mask.dtype != torch.bool or action_mask.dtype != torch.bool:
            raise ValueError("history event masks must be boolean")
        if state is None:
            memory = carriers.new_zeros(batch, HISTORY_DIM)
            last_timestamp = timestamps.new_zeros(batch)
            seen = torch.zeros(batch, dtype=torch.bool, device=carriers.device)
        else:
            memory, last_timestamp, seen = state.memory, state.last_timestamp, state.seen_event
        memories = []
        for index in range(steps):
            observed, acted = observation_mask[:, index], action_mask[:, index]
            valid = observed | acted
            now = timestamps[:, index]
            if not bool(torch.isfinite(now[valid]).all()):
                raise ValueError("nonfinite agent event timestamp")
            elapsed = torch.where(seen & valid, now - last_timestamp, 0.)
            if bool((elapsed < 0).any()):
                raise ValueError("observed history cannot move backwards in time")
            visible = torch.where(observed[:, None], carriers[:, index], 0.)
            executed = torch.where(acted[:, None], actions[:, index], 0.)
            features = torch.cat((visible, executed, torch.log1p(elapsed)[:, None].to(carriers),
                                  observed[:, None].to(carriers), acted[:, None].to(carriers)), dim=-1)
            if not bool(torch.isfinite(features).all()):
                raise ValueError("nonfinite permitted history input")
            proposed = self.cell(features, memory)
            memory = torch.where(valid[:, None], proposed, memory)
            last_timestamp = torch.where(valid, now, last_timestamp)
            seen = seen | valid
            memories.append(memory)
        sequence = torch.stack(memories, dim=1) if memories else carriers.new_empty(batch, 0, HISTORY_DIM)
        return sequence, HistoryState(memory, last_timestamp, seen)


def history_contract():
    return {"schema": "issue_76_observed_history_v1", "carrier_dim": DIM,
            "memory_dim": HISTORY_DIM, "executed_action_dim": ACTION_DIM,
            "inputs": ["agent_observation_carrier", "executed_action", "agent_timestamp_seconds",
                       "observation_present", "action_executed"],
            "reset": "new gameplay episode only; never reset between shots",
            "missing_observation": "zero masked visual input; action-only events still update; padding holds memory",
            "time": "log1p seconds since previous valid event; first event has zero elapsed time",
            "common_weights": "same frozen encoder checkpoint for both dynamics arms",
            "oracle_hit_counts_allowed": False, "future_observations_allowed": False,
            "fit_executed": False}
