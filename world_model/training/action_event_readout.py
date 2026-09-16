"""Action-conditioned event utility readout and globally balanced training loss."""

import torch
from torch import nn
from torch.nn import functional as F

from world_model.training.event_ranking import CLEAR_INDEX, STOP_KINDS


class ActionConditionedEventReadout(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, action_scale=6.0):
        super().__init__()
        self.action_scale = action_scale
        self.state = nn.Linear(2 * state_dim, hidden_dim)
        self.action = nn.Linear(5, hidden_dim)
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                    nn.Linear(hidden_dim, len(STOP_KINDS)))

    def forward(self, initial, endpoint, action):
        action = torch.cat((action[:, :2] * self.action_scale, action[:, 2:]), dim=1)
        state = F.silu(self.state(torch.cat((initial, endpoint), dim=1)))
        return self.output(state * (1 + self.action(action).tanh()))


def balanced_event_loss(logits, labels, class_weights, groups, paired):
    classification = F.cross_entropy(logits, labels, weight=class_weights,
                                      reduction="none").mean()
    ranking = logits.new_zeros(())
    for indices, admissible in zip(groups, paired, strict=True):
        clear = labels[indices] == CLEAR_INDEX
        if admissible and bool(clear.any()):
            scores = logits[indices, CLEAR_INDEX]
            ranking = ranking + torch.logsumexp(scores, 0) - torch.logsumexp(scores[clear], 0)
    return classification + ranking / len(groups)
