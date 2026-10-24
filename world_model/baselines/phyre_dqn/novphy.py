"""#128 NovPhy adapter for the PHYRE DQN recipe (A1 DQN-px, A2 DQN-car).

The network, FiLM fusion, head and loss are the vendored official code (``vendor/nets.py``, facebookresearch/phyre
@ d0765dd, Apache-2.0). This module only changes what the NovPhy input requires and re-expresses the official
training step, which in ``vendor/neural_agent.py::train`` is a closure over PHYRE's simulation cache:

- **A1** ``A1Net``: the official ``ResNet18FilmAction`` (ResNet-18; action MLP 1 x 256 ReLU on the 4-d ``eb.launch4``
  vector -> FiLM gain and bias on the 256 channels before ``layer4``; max-pool; ``Linear(512, 1)``). ``input_mode
  "rgb"`` replaces the 7-channel one-hot stem convolution by a 3-channel one on the RGB decision frame scaled to
  [0, 1] (uint8 [3, 240, 320], native resolution); ``"onehot7"`` is the unmodified official configuration (G-port).
- **A2** ``A2Net``: the official ``preprocess`` / ``_forward`` / ``ce_loss`` and ``FilmActionNetwork`` on an MLP trunk
  over the 236-d E99 carrier: the ResNet-18 stage ladder (widths b, 2b, 4b, 8b; two residual blocks per stage)
  with every convolution 1 x 1 on a [B, C, 1, 1] tensor, FiLM before the last stage, ``Linear(8b, 1)`` head. ``b`` is
  set so the trainable parameter count matches U (matched) or A1 (native-size).
- Training (``train_indices_sampler``, ``make_optimizer``, ``train_step``) and scoring (``score_actions``) are the
  official loop body, sampler and ``eval_actions`` with the data passed in (one example per fit shot).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from torch import nn

from .vendor import nets
from .vendor.torchvision.models.resnet import conv1x1

# The official dqn_10k recipe: agents/run_experiment.py DQN_BASE_ARGS + agents/offline_agents.py DQNAgent defaults,
# identical to the parsed args in the shipped results.json of every ball_within_template seed.
OFFICIAL = {"network_type": "resnet18", "fusion_place": "last", "action_layers": 1, "action_hidden_size": 256,
            "learning_rate": 3e-4, "train_batch_size": 64, "updates": 100_000, "cosine_scheduler": True,
            "balance_classes": True, "negative_sampling_prob": 1.0, "eval_batch_size": 128,
            "optimizer": "Adam(model.parameters(), lr)", "scheduler": "CosineAnnealingLR(T_max=updates), "
            "scheduler.step() before every optimizer.step()", "sampler_rng": "np.random.RandomState(42)"}
NATIVE_SCHEDULE = {"updates": 100_000, "batch_size": 64}  # A1, A2 native-size
MATCHED_SCHEDULE = {"updates": 9_000, "batch_size": 64}  # U's B1 recipe (A2 matched)
U_PARAMETERS = 1_837_690
MATCHED_TOLERANCE = 0.05
ACTION_DIM = 4  # eb.launch4: [drag_x/480, drag_y/480, release/1000, tap/1000]
CARRIER_DIM = 236
INPUT_MODES = ("rgb", "onehot7")
A2_MATCHED_BASE_WIDTH = 70  # 1 838 781 trainable parameters (U 1 837 690, +0.06 %)
A2_NATIVE_BASE_WIDTH = 179  # 11 333 771 (A1 rgb 11 309 889, +0.21 %)


def trainable_parameters(model):
    """Every trainable parameter (trunk, action MLP, FiLM output layer and head), each counted once."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class A1Net(nets.ResNet18FilmAction):
    """The official ResNet18FilmAction; ``input_mode="rgb"`` takes uint8 [B, 3, H, W] frames (scaled to [0, 1])."""

    def __init__(self, input_mode="rgb", action_size=ACTION_DIM, action_hidden_size=OFFICIAL["action_hidden_size"],
                 action_layers=OFFICIAL["action_layers"], fusion_place=OFFICIAL["fusion_place"]):
        if input_mode not in INPUT_MODES:
            raise ValueError(f"unknown input_mode {input_mode!r}")
        super().__init__(action_size, action_layers=action_layers, action_hidden_size=action_hidden_size,
                         fusion_place=fusion_place)
        self.input_mode = input_mode
        if input_mode == "rgb":
            official = self.stem[0]  # Conv2d(7, 64, 7, stride 2, padding 3, bias False), PyTorch default init
            self.stem[0] = nn.Conv2d(3, official.out_channels, kernel_size=official.kernel_size,
                                     stride=official.stride, padding=official.padding, bias=False)
            del self.embed_weights

    def _image_colors_to_onehot(self, indices):
        if self.input_mode == "onehot7":
            return super()._image_colors_to_onehot(indices)
        return indices.to(device=self.stem[0].weight.device, dtype=torch.float32).div(255.0)


class CarrierBlock(nn.Module):
    """torchvision's BasicBlock (vendor/torchvision/models/resnet.py) with both 3 x 3 convolutions replaced by 1 x 1
    ones: a residual two-layer MLP with BatchNorm on [B, C, 1, 1]; projection shortcut when the width changes."""

    def __init__(self, inplanes, planes):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = (nn.Sequential(conv1x1(inplanes, planes), nn.BatchNorm2d(planes))
                           if inplanes != planes else None)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class A2Net(nets.ResNet18FilmAction):
    """A1's fusion and head on the E99 carrier: the official ``preprocess`` / ``_forward`` / ``ce_loss`` and
    ``FilmActionNetwork`` (inherited) over a CarrierBlock trunk with the ResNet-18 stage ladder."""

    def __init__(self, base_width=A2_MATCHED_BASE_WIDTH, action_size=ACTION_DIM, carrier_dim=CARRIER_DIM,
                 action_hidden_size=OFFICIAL["action_hidden_size"], action_layers=OFFICIAL["action_layers"]):
        nn.Module.__init__(self)  # the official __init__ builds the torchvision trunk; the attributes it sets follow
        widths = (base_width, 2 * base_width, 4 * base_width, 8 * base_width)
        self.stem = nn.Sequential(conv1x1(carrier_dim, widths[0]), nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        stages, inplanes = [], widths[0]
        for width in widths:
            stages.append(nn.Sequential(CarrierBlock(inplanes, width), CarrierBlock(width, width)))
            inplanes = width
        self.stages = nn.ModuleList(stages)
        # torchvision ResNet init for the residual stages (kaiming-normal fan-out, BN 1 / 0); the stem convolution
        # keeps PyTorch's default init like the official (and A1's) stem convolution.
        for m in self.stages.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.last_network = None
        self._action_network = nets.FilmActionNetwork(action_size, widths[2], hidden_size=action_hidden_size,
                                                      num_layers=action_layers)
        self.action_networks = [None, None, None, self._action_network]  # fusion_place "last", as official
        self.reason = nn.Linear(widths[3], 1)
        self.base_width = base_width

    def _image_colors_to_onehot(self, carriers):
        return carriers.to(device=self.reason.weight.device, dtype=torch.float32)[:, :, None, None]


def build_model(arm, variant="primary"):
    """A1 (primary), A2 (primary = matched to U, or native_size = matched to A1)."""
    if arm == "A1" and variant == "primary":
        return A1Net("rgb")
    if arm == "A2" and variant in ("primary", "matched"):
        return A2Net(A2_MATCHED_BASE_WIDTH)
    if arm == "A2" and variant == "native_size":
        return A2Net(A2_NATIVE_BASE_WIDTH)
    raise ValueError(f"unknown arm/variant {arm}/{variant}")


class TrainSet(NamedTuple):
    """The official TrainData layout (neural_agent.compact_simulation_data_to_trainset) without the simulator:
    example i is (observations[task_indices[i]], actions[i], is_solved[i])."""
    task_indices: torch.Tensor  # int64 [N]
    is_solved: torch.Tensor  # bool [N]
    actions: torch.Tensor  # float32 [N, 4]
    observations: torch.Tensor  # [L, ...]: uint8 [L, 3, 240, 320] frames (A1) or float32 [L, 236] carriers (A2)
    levels: tuple  # member of each observation row


def build_trainset(members, launch4, verdicts, anchors):
    """One example per fit shot: the shot's level anchor (``anchors[member]``), its launch4 vector, its verdict."""
    levels = tuple(sorted(set(members)))
    row = {member: i for i, member in enumerate(levels)}
    observations = torch.stack([torch.as_tensor(anchors[member]) for member in levels])
    return TrainSet(task_indices=torch.tensor([row[m] for m in members], dtype=torch.long),
                    is_solved=torch.as_tensor(verdicts, dtype=torch.bool).clone(),
                    actions=torch.as_tensor(launch4, dtype=torch.float32).clone(),
                    observations=observations, levels=levels)


def train_indices_sampler(is_solved, train_batch_size, rng, balance_classes=True, negative_sampling_prob=1.0):
    """neural_agent.train's ``train_indices_sampler`` (vendor/neural_agent.py), with the labels, batch size and the
    numpy RandomState passed in. Balanced batches alternate the extra example of an odd batch between classes."""
    indices = np.arange(len(is_solved))
    if balance_classes:
        solved_mask = is_solved.numpy() > 0
        positive_indices = indices[solved_mask]
        negative_indices = indices[~solved_mask]
        positive_size = train_batch_size // 2
        while True:
            positives = rng.choice(positive_indices, size=positive_size)
            negatives = rng.choice(negative_indices, size=train_batch_size - positive_size)
            positive_size = train_batch_size - positive_size
            yield np.concatenate((positives, negatives))
    elif negative_sampling_prob < 1:
        probs = (is_solved.numpy() * (1.0 - negative_sampling_prob) + negative_sampling_prob)
        probs /= probs.sum()
        while True:
            yield rng.choice(indices, size=train_batch_size, p=probs)
    else:
        while True:
            yield rng.choice(indices, size=train_batch_size)


def make_optimizer(model, learning_rate=OFFICIAL["learning_rate"], updates=NATIVE_SCHEDULE["updates"],
                   cosine_scheduler=True):
    """Adam(model.parameters(), lr) and CosineAnnealingLR(T_max=updates), as neural_agent.train builds them."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates) if cosine_scheduler else None
    return optimizer, scheduler


def train_step(model, optimizer, scheduler, data, batch_indices, device="cpu"):
    """One iteration of neural_agent.train's loop: ``scheduler.step()`` first (so update k runs at the cosine LR of
    step k), train mode, the batch gathered through ``task_indices``, ``model.ce_loss`` (BCE with logits on the
    verdict), backward, ``optimizer.step()``. ``data.observations`` must already be on ``device``."""
    if scheduler is not None:
        scheduler.step()
    model.train()
    batch_task_indices = data.task_indices[batch_indices]
    batch_observations = data.observations[batch_task_indices]
    batch_actions = data.actions[batch_indices].to(device, non_blocking=True)
    batch_is_solved = data.is_solved[batch_indices].to(device, non_blocking=True)
    optimizer.zero_grad()
    loss = model.ce_loss(model(batch_observations, batch_actions), batch_is_solved)
    loss.backward()
    optimizer.step()
    return loss.detach()


@torch.no_grad()
def score_actions(model, observation, actions, batch_size=OFFICIAL["eval_batch_size"]):
    """neural_agent.eval_actions: eval mode, the observation encoded once, success logits of every action."""
    model.eval()
    preprocessed = model.preprocess(observation.unsqueeze(0))
    device = next(model.parameters()).device
    scores = []
    for start in range(0, len(actions), batch_size):
        batch = actions[start:start + batch_size].to(device=device, dtype=torch.float32)
        scores.append(model(None, batch, preprocessed=preprocessed).float().cpu())
    return torch.cat(scores)
