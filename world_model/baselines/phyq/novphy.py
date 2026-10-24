"""#128 A3 PhyQ-RelDQN: NovPhy adapter for Phy-Q's relational DQN (phy-q/benchmark @ 514491c, MIT; ``vendor/``).

Two variants (#127 arm matrix), never in the program set:

- **A3-sym** ``DQNRelationalSymbolic``: input = Phy-Q's symbolic state of the decision state, a 12 x 120 x 160
  binary occupancy image (``SymbolicStateReader.get_symbolic_image``, the dense ``transform_full`` path the official
  ``TrainLearningAgent.py --if_save_local True`` training stores). It is computed by the vendored reader, verbatim
  (incl. its linear colour classifier ``Utils/model``), from the ground-truth JSON our engine's
  ``SymbolicGameState.GetGTJson`` produces at the decision state, rebuilt from the capture's
  ``decision-physics-v1.json`` (``PhysicalSnapshotExporter``: every node is one GT feature, ``screen_polygon`` = its
  first contour, ``object_type`` = its dev-mode label = sprite name) plus the colour map the engine attaches in the
  non-dev GT JSON (``ColorDataLookUp(sprite)`` = the engine's ``StreamingAssets/ColorCodes/BaseMap.json`` entry, no
  noise). Labels are the non-dev ones (``Object``, ``Slingshot``, ``Ground``), as Phy-Q's agents receive them.
- **A3-img** ``DQNRelationalImage``: input = the 320 x 240 RGB decision frame as the Phy-Q client would hold the
  640 x 480 screenshot (2x pixel replication, BGR channel order, [3, 480, 640], ``agent_client.read_image_from_stream``)
  through the official ``DQNRelationalImage.transform`` (transpose, ``CenterCrop((360, 480))``, ``Resize((120, 160))``,
  ``ToTensor``, ``Normalize(0.5, 0.5)``), re-expressed with PIL (novphy has no torchvision; G-port checks equality).

``RelationalDQN`` re-expresses both official ``__init__`` / ``forward`` line by line (same module creation order and
random-number consumption, so a seed gives the official initialisation) on the vendored torchvision v0.4.0 ResNet-18
(``world_model/baselines/phyre_dqn/vendor``, BSD-3) with torchvision 0.9.1's ImageNet weights
(``data/external/phyq/resnet18-5c106cde.pth``). Training is the official ``DQNBase.train_model_loader`` step (the path
``TrainLearningAgent.py`` runs): Q(s, a) gathered from the 180-way head, ``SmoothL1Loss`` against r + gamma *
max Q_target(s') with the bootstrap zeroed on terminal transitions (every fit shot is terminal: the 'passing' reward
= int(did_win) = engine pig removal), gradients clamped to [-1, 1], AdamW lr 3e-4 re-created at every outer
training step (``OuterSteps``), class-balanced sampling with replacement (``WeightedRandomSampler``), batch 32.

Action mapping: Phy-Q's action a in [0, 180) releases the bird at (200 cos(a + 90 deg), 200 sin(a + 90 deg)) px
from the slingshot (``DQNDiscreteAgent.__degToShot``; screen y up, i.e. abs = (sling.X + dx, sling.Y - dy)); our
launch releases at (drag_x, drag_y) px from the slingshot with screen y down. A launch maps to the nearest native
action: a = round(atan2(-drag_y, drag_x) in degrees, mod 360) - 90. Drag length is not expressible (Phy-Q always
pulls 200 px); every fit launch is beyond the 1-unit drag radius (~18 px), so power is saturated in the fit data.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch import nn

from ..phyre_dqn.vendor.torchvision.models.resnet import resnet18
from .vendor.StateReader.SymbolicStateReader import SymbolicStateReader

ROOT = Path(__file__).resolve().parents[3]
VENDOR = Path(__file__).resolve().parent / "vendor"
IMAGENET_WEIGHTS = ROOT / "data/external/phyq/resnet18-5c106cde.pth"
IMAGENET_SHA256 = "5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8"
IMAGENET_URL = "https://download.pytorch.org/models/resnet18-5c106cde.pth"  # torchvision 0.9.1 model_urls

ARMS = {"A3-sym": "symbolic", "A3-img": "image"}
# Utils/Parameters.py (h, w, output, train_batch, gamma, lr, reward_type), TrainLearningAgent.py (optim.AdamW,
# train_time_per_ep schedule, N = training levels // 5), DQNBase.train_model_loader (SmoothL1Loss, clamp_(-1, 1)).
OFFICIAL = {"h": 120, "w": 160, "outputs": 180, "train_batch": 32, "learning_rate": 3e-4, "gamma": 0.99,
            "grad_clamp": 1.0, "train_time_per_ep": 32, "train_time_rise": 1, "reward_type": "passing",
            "optimizer": "torch.optim.AdamW(params, lr) with torch defaults (betas 0.9/0.999, eps 1e-8, "
                         "weight_decay 0.01), re-created at every outer training step"}
SYMBOLIC_CHANNELS = 12
SCREEN = (640, 480)  # Phy-Q screenshot / GT coordinate frame (SymbolicStateReader x_size, y_size)
FRAME = (320, 240)  # the #127 decision frame (s99.image_tensor: bilinear 640x480 -> 320x240)
DECISION_FIXED_STEP = 30000  # m99.DECISION_FIXED_STEP: frame 0 of every shot
PLATFORM_SPRITE = "effects_21"  # Platform.prefab m_Sprite (Sprites/Effects/effects.png.meta fileID 21300082)
BASEMAP = Path("runtime/9001_Data/StreamingAssets/ColorCodes/BaseMap.json")  # ABConstants.colorDataFile


def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# network (DQNRelationalSymbolic / DQNRelationalImage)
# ---------------------------------------------------------------------------

def imagenet_state_dict():
    """torchvision 0.9.1 ``resnet18(pretrained=True)`` weights (legacy tar file: hash-checked, then unpickled)."""
    digest = hashlib.sha256(IMAGENET_WEIGHTS.read_bytes()).hexdigest()
    if digest != IMAGENET_SHA256:
        raise ValueError(f"{IMAGENET_WEIGHTS} sha256 {digest} differs from {IMAGENET_SHA256}")
    return torch.load(IMAGENET_WEIGHTS, map_location="cpu", weights_only=False)


class RelationalDQN(nn.Module):
    """``DQNRelationalSymbolic`` (input_type "symbolic") / ``DQNRelationalImage`` ("image"), official order."""

    def __init__(self, input_type, h=OFFICIAL["h"], w=OFFICIAL["w"], outputs=OFFICIAL["outputs"]):
        super().__init__()
        if input_type not in ("symbolic", "image"):
            raise ValueError(f"unknown input_type {input_type!r}")
        self.input_type = input_type
        self.output_type = "discrete"
        self.H, self.W, self.h, self.w = h, w, h, w
        self.node_size = 64
        self.out_dim = self.outputs = outputs
        self.ch_in = SYMBOLIC_CHANNELS if input_type == "symbolic" else 3
        self.sp_coord_dim = 2
        self.n_heads = 4

        model = resnet18()
        model.load_state_dict(imagenet_state_dict())
        if input_type == "symbolic":
            model.conv1 = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=2, padding=3)
        self.feature_head = torch.nn.Sequential(*(list(model.children())[:-2]))
        with torch.no_grad():  # official shape probe: consumes the RNG and updates the BatchNorm running statistics
            _, out_ch, out_h, out_w = self.feature_head(torch.randn(1, self.ch_in, h, w)).shape

        self.proj_shape = (out_ch + self.sp_coord_dim, self.n_heads * self.node_size)
        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)

        self.N = out_h * out_w
        self.k_lin = nn.Linear(self.node_size, self.N)
        self.q_lin = nn.Linear(self.node_size, self.N)
        self.a_lin = nn.Linear(self.N, self.N)

        self.node_shape = (self.n_heads, self.N, self.node_size)
        self.k_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)

        self.linear1 = nn.Linear(self.n_heads * self.node_size, self.node_size)
        self.norm1 = nn.LayerNorm([self.N, self.node_size], elementwise_affine=False)
        self.linear2 = nn.Linear(self.node_size, self.out_dim)

    def forward(self, x):
        N, Cin, H, W = x.shape
        x = self.feature_head(x)
        with torch.no_grad():
            _, _, cH, cW = x.shape
            xcoords = torch.arange(cW).repeat(cH, 1).float().to(x.device) / cW
            ycoords = torch.arange(cH).repeat(cW, 1).transpose(1, 0).float().to(x.device) / cH
            spatial_coords = torch.stack([xcoords, ycoords], dim=0).to(x.device)
            spatial_coords = spatial_coords.unsqueeze(dim=0).to(x.device)
            spatial_coords = spatial_coords.repeat(N, 1, 1, 1).to(x.device)
        x = torch.cat([x, spatial_coords], dim=1)
        x = x.permute(0, 2, 3, 1)
        x = x.flatten(1, 2)

        K = rearrange(self.k_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        K = self.k_norm(K)
        Q = rearrange(self.q_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        Q = self.q_norm(Q)
        V = rearrange(self.v_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        V = self.v_norm(V)

        A = torch.nn.functional.elu(self.q_lin(Q) + self.k_lin(K))
        A = self.a_lin(A)
        A = torch.nn.functional.softmax(A, dim=3)
        E = torch.einsum('bhfc,bhcd->bhfd', A, V)

        E = rearrange(E, 'b head n d -> b n (head d)')
        E = self.linear1(E)
        E = torch.relu(E)
        E = self.norm1(E)
        E = E.max(dim=1)[0]
        y = self.linear2(E)
        y = torch.nn.functional.elu(y)
        return y


def build_model(arm):
    """The arm's network in the official configuration (the caller seeds torch first)."""
    return RelationalDQN(ARMS[arm])


# ---------------------------------------------------------------------------
# A3-sym input: Phy-Q's symbolic state of the decision state
# ---------------------------------------------------------------------------

_COLORMAPS = {}


def engine_colormaps(attempt):
    """(sha256, {sprite: colormap}) of the BaseMap.json the attempt's engine read (cached per file)."""
    path = Path(attempt) / BASEMAP
    stat = path.stat()
    key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if key not in _COLORMAPS:
        raw = path.read_bytes()
        _COLORMAPS[key] = ("sha256:" + hashlib.sha256(raw).hexdigest(),
                           {sprite["type"]: sprite["colormap"] for sprite in json.loads(raw)["sprites"]})
    return _COLORMAPS[key]


def gt_json(physics, colormaps):
    """The engine's non-dev ground-truth JSON (``SymbolicGameState.GetGTJson(useNoise=false)``) at the decision state.

    ``physics`` is a ``decision-physics-v1.json`` (``PhysicalSnapshotExporter.Capture`` of that JSON in dev mode): each
    node keeps the feature's first contour and its dev label. Non-dev labels are ``Object`` (with the colour map
    ``ColorDataLookUp`` attaches: the sprite's ``BaseMap.json`` entry; ``Platform`` -> sprite ``effects_21``),
    ``Slingshot`` and ``Ground`` (both without colour map, as ``GTObject`` / ``GTGround`` write them)."""
    if physics["fixed_step"] != DECISION_FIXED_STEP:
        raise ValueError(f"decision physics at fixed step {physics['fixed_step']}, expected {DECISION_FIXED_STEP}")
    if physics["coordinates"]["screen_origin"] != "top_left":
        raise ValueError("decision physics screen coordinates are not top-left")
    features = []
    for node in physics["nodes"]:
        kind = node["object_type"]
        contour = [[point["x"], point["y"]] for point in node["screen_polygon"]]
        if kind in ("Ground", "Slingshot"):
            label, colormap = kind, []
        else:
            sprite = PLATFORM_SPRITE if kind == "Platform" else kind
            if sprite not in colormaps:
                raise KeyError(f"sprite {sprite!r} has no engine colour map")
            label = "Object"
            # C# ColorEntry: (int) color, float percent
            colormap = [{"color": int(entry["x"]), "percent": float(np.float32(entry["y"]))}
                        for entry in colormaps[sprite]]
        features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [contour]},
                         "properties": {"id": node["entity_id"], "label": label, "colormap": colormap}})
    return [{"type": "FeatureCollection", "features": features}]


_READER_MODEL = []


def reader_model():
    """(model, target_class) exactly as DQNSymbolicBase.__init__ loads them (from the vendor directory, not the cwd)."""
    if not _READER_MODEL:
        model = np.loadtxt(VENDOR / "Utils/model", delimiter=",")
        with open(VENDOR / "Utils/target_class") as handle:
            target_class = list(map(lambda x: x.replace("\n", ""), handle.readlines()))
        _READER_MODEL.extend((model, target_class))
    return _READER_MODEL[0], _READER_MODEL[1]


@contextlib.contextmanager
def numpy_float_alias():
    """``np.float`` (removed in NumPy 1.24; alias of ``float``) for the vendored reader (VENDOR.json compatibility)."""
    present = "float" in np.__dict__
    if not present:
        np.float = float
    try:
        yield
    finally:
        if not present:
            del np.float


def symbolic_image(gt):
    """``DQNSymbolicBase.transform_full``: the reader's dense 12 x h x w image, stored as uint8 (official memmap)."""
    model, target_class = reader_model()
    with numpy_float_alias():
        image = SymbolicStateReader(gt, model, target_class).get_symbolic_image(h=OFFICIAL["h"], w=OFFICIAL["w"])
    if not np.isin(image, (0.0, 1.0)).all():
        raise ValueError("symbolic image is not binary")
    return torch.from_numpy(image.astype(np.uint8))


def symbolic_state(attempt):
    """uint8 [12, 120, 160]: Phy-Q's symbolic state of the attempt's decision state (frame 0 of its shot)."""
    physics = json.loads((Path(attempt) / "decision-physics-v1.json").read_text())
    return symbolic_image(gt_json(physics, engine_colormaps(attempt)[1]))


def symbolic_anchors(attempts):
    """{member: uint8 [12, 120, 160]} for {member: anchor attempt directory} (any split; evaluation post-freeze)."""
    return {member: symbolic_state(attempt) for member, attempt in sorted(attempts.items())}


SYMBOLIC_SCHEMA = "issue_128_a3_symbolic_anchors_v1"
A3_DATA = ROOT / ".local-artifacts/issue-128-direct-predictors-v1/a3/data"


def split_anchor_attempts(split):
    """({member: attempt dir}, {member: anchor key}) of the fit or policy anchors (never evaluation)."""
    from scripts import external_baselines as eb
    from scripts import run_hjepa_comparison as rc
    if split == "fit":
        keys = eb.fit_anchors()["keys"]
    elif split == "policy":
        keys = eb.policy_anchors()["keys"]
    else:
        raise ValueError("the symbolic store is built for the fit and policy splits only")
    return {m: rc.CAMPAIGN / rc.STREAMS[split] / "attempts" / k for m, k in keys.items()}, dict(keys)


def symbolic_store_path(split):
    return A3_DATA / f"symbolic-{split}.pt"


def build_symbolic_store(split):
    attempts, keys = split_anchor_attempts(split)
    states = symbolic_anchors(attempts)
    payload = {"schema": SYMBOLIC_SCHEMA, "split": split, "states": states, "keys": keys,
               "basemap_sha256": sorted({engine_colormaps(a)[0] for a in attempts.values()}),
               "decision_fixed_step": DECISION_FIXED_STEP, "shape": [SYMBOLIC_CHANNELS, OFFICIAL["h"], OFFICIAL["w"]],
               "adapter_sha256": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = symbolic_store_path(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return payload


def load_symbolic_store(split):
    payload = torch.load(symbolic_store_path(split), map_location="cpu", weights_only=True)
    if payload.get("schema") != SYMBOLIC_SCHEMA or payload.get("split") != split:
        raise ValueError(f"{symbolic_store_path(split)} binding differs")
    return payload


# ---------------------------------------------------------------------------
# A3-img input: the decision frame through the official image transform
# ---------------------------------------------------------------------------

def phyq_screenshot(frame):
    """uint8 [3, 240, 320] RGB decision frame -> uint8 [3, 480, 640] BGR, the layout ``read_image_from_stream``
    returns (``img[:, :, ::-1]`` then ``transpose(2, 0, 1)``) at the engine's 640 x 480 screen (2x replication)."""
    frame = torch.as_tensor(frame)
    if frame.dtype != torch.uint8 or tuple(frame.shape) != (3, FRAME[1], FRAME[0]):
        raise ValueError(f"expected a uint8 [3, {FRAME[1]}, {FRAME[0]}] frame, got {frame.dtype} {tuple(frame.shape)}")
    return frame.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2).flip(0).numpy()


def image_transform(state):
    """``DQNRelationalImage.transform``: ``T.Compose([T.ToPILImage(), T.CenterCrop((360, 480)), T.Resize((h, w)),
    T.ToTensor(), T.Normalize((.5, .5, .5), (.5, .5, .5))])(state.transpose(2, 1, 0))`` on PIL (torchvision's PIL path)."""
    image = Image.fromarray(np.ascontiguousarray(np.asarray(state, dtype=np.uint8).transpose(2, 1, 0)))
    crop_h, crop_w = 360, 480
    width, height = image.size
    top, left = int(round((height - crop_h) / 2.0)), int(round((width - crop_w) / 2.0))
    if top < 0 or left < 0:
        raise ValueError("the official centre crop needs a 640 x 480 screenshot")
    image = image.crop((left, top, left + crop_w, top + crop_h)).resize((OFFICIAL["w"], OFFICIAL["h"]),
                                                                        Image.BILINEAR)
    tensor = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2, 0, 1).float().div(255)
    return tensor.sub(0.5).div(0.5)


def image_state(frame):
    """float32 [3, 120, 160]: the official image input of a decision frame."""
    return image_transform(phyq_screenshot(frame))


def model_input(input_type, observation):
    """float32 network input of one anchor: symbolic uint8 [12, 120, 160] or the uint8 [3, 240, 320] frame."""
    if input_type == "symbolic":
        state = torch.as_tensor(observation)
        if state.dtype != torch.uint8 or tuple(state.shape) != (SYMBOLIC_CHANNELS, OFFICIAL["h"], OFFICIAL["w"]):
            raise ValueError(f"expected a uint8 symbolic state, got {state.dtype} {tuple(state.shape)}")
        return state.float()
    return image_state(observation)


# ---------------------------------------------------------------------------
# action mapping (launch -> Phy-Q action index)
# ---------------------------------------------------------------------------

def action_index(drag_x, drag_y):
    """Nearest native action: Phy-Q releases action a at angle a + 90 deg (dx = 200 cos, dy = 200 sin, screen y up);
    our release offset is (drag_x, drag_y) with screen y down, i.e. Phy-Q (dx, dy) = (drag_x, -drag_y)."""
    degrees = math.degrees(math.atan2(-drag_y, drag_x)) % 360.0
    index = math.floor(degrees + 0.5) - 90
    if not 0 <= index < OFFICIAL["outputs"]:
        raise ValueError(f"launch ({drag_x}, {drag_y}) at {degrees:.2f} deg is outside Phy-Q's [90, 270) deg actions")
    return index


def action_indices(launches):
    """LongTensor [K] of Phy-Q action indices for an ``eb.launch4`` matrix [K, 4] (drag = 480 * launch[:, :2])."""
    unique, inverse = torch.unique(torch.as_tensor(launches)[:, :2], dim=0, return_inverse=True)
    mapped = torch.tensor([action_index(480. * x, 480. * y) for x, y in unique.tolist()], dtype=torch.long)
    return mapped[inverse.view(-1)]


# ---------------------------------------------------------------------------
# training data, optimizer, step (DQNBase.train_model_loader) and scoring
# ---------------------------------------------------------------------------

class PhyqData(NamedTuple):
    inputs: torch.Tensor  # [L, C, 120, 160] per level (sym uint8, img float32)
    obs: torch.Tensor  # long [N]: level of each fit shot
    actions: torch.Tensor  # long [N]: Phy-Q action index of each fit shot
    rewards: torch.Tensor  # float [N]: 'passing' reward = engine verdict
    probabilities: np.ndarray  # float64 [N]: WeightedRandomSampler probabilities
    levels: tuple


def class_balanced_probabilities(rewards):
    """``MemoryDatasetMemmap.update`` weights (1 / class share) normalised as ``WeightedRandomSampler`` draws them;
    uniform when only one class is present (the official loader then shuffles uniformly)."""
    rewards = np.asarray(rewards).astype(np.int64)
    counts = {value: int((rewards == value).sum()) for value in (0, 1)}
    if min(counts.values()) == 0:
        return np.full(len(rewards), 1.0 / len(rewards))
    weights = np.where(rewards == 0, len(rewards) / counts[0], len(rewards) / counts[1])
    return weights / weights.sum()


def runner_trainset(arm, fit, anchors):
    """One example per fit shot: (level anchor input, Phy-Q action of its launch, engine verdict).
    ``anchors``: {member: uint8 [12, 120, 160]} (A3-sym, ``symbolic_anchors``) or {member: uint8 [3, 240, 320]}
    (A3-img, ``eb.fit_anchor_frames()``)."""
    input_type = ARMS[arm]
    levels = tuple(sorted(set(fit.members)))
    missing = [m for m in levels if m not in anchors]
    if missing:
        raise KeyError(f"{len(missing)} fit levels have no anchor input, e.g. {missing[0]}")
    if input_type == "symbolic":
        inputs = torch.stack([model_input(input_type, anchors[m]).to(torch.uint8) for m in levels])
    else:
        inputs = torch.stack([image_state(anchors[m]) for m in levels])
    index = {m: i for i, m in enumerate(levels)}
    verdicts = torch.as_tensor(fit.verdicts)
    return PhyqData(inputs=inputs, obs=torch.tensor([index[m] for m in fit.members], dtype=torch.long),
                    actions=action_indices(fit.launch4), rewards=verdicts.to(torch.float32),
                    probabilities=class_balanced_probabilities(verdicts.numpy()), levels=levels)


def data_to(data, device):
    return data._replace(inputs=data.inputs.to(device), obs=data.obs.to(device), actions=data.actions.to(device),
                         rewards=data.rewards.to(device))


def runner_batch(data, rng):
    """One batch of ``train_batch`` shot indices, i.i.d. with replacement under the class-balanced weights."""
    return rng.choice(len(data.probabilities), size=OFFICIAL["train_batch"], replace=True, p=data.probabilities)


class OuterSteps:
    """A fresh AdamW at every outer training step: ``TrainLearningAgent.py`` calls ``network.train_model_loader(...,
    optimizer=optim.AdamW, lr=lr)`` once per step and it constructs ``AdamW(self.parameters(), lr=lr)``. ``step()`` is
    called once per update, before ``optimizer.step()``; at a reset update it empties the optimizer state."""

    def __init__(self, optimizer, resets=()):
        self.optimizer = optimizer
        self.resets = sorted({int(r) for r in resets})
        self._reset_set = set(self.resets)
        self.update = 0

    def step(self):
        if self.update in self._reset_set:
            self.optimizer.state.clear()
        self.update += 1

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {"update": self.update, "resets": list(self.resets)}

    def load_state_dict(self, state):
        self.update = int(state["update"])
        self.resets = sorted(int(r) for r in state["resets"])
        self._reset_set = set(self.resets)


def make_optimizer(model, updates, learning_rate=OFFICIAL["learning_rate"], resets=()):
    """(AdamW, OuterSteps); ``resets`` = update indices that start an outer step (``recipe(...)["resets"]``)."""
    if any(not 0 < r < updates for r in resets):
        raise ValueError("outer-step resets must lie strictly inside the run")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return optimizer, OuterSteps(optimizer, resets)


def runner_train_step(model, optimizer, scheduler, data, indices, device="cpu"):
    """``DQNBase.train_model_loader`` loop body on one batch of fit shots (all terminal transitions)."""
    idx = torch.as_tensor(np.asarray(indices), dtype=torch.long, device=data.obs.device)
    model.train()
    state = data.inputs[data.obs[idx]].to(device).float()
    action_batch = data.actions[idx].to(device).long().view(-1, 1)
    reward_batch = data.rewards[idx].to(device).float()
    state_action_values = model(state).gather(1, action_batch)
    # every fit shot ends its level (one bird): is_done = True, so the official next_state_values[is_done] = 0.0
    next_state_values = torch.zeros_like(state_action_values)
    expected_state_action_values = (next_state_values * OFFICIAL["gamma"]) + reward_batch.view(-1, 1)
    loss = nn.SmoothL1Loss(reduction='none')(state_action_values, expected_state_action_values)
    processed_loss = torch.mean(loss)
    optimizer.zero_grad()
    processed_loss.backward()
    grad_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in model.parameters()]))
    for param in model.parameters():
        param.grad.data.clamp_(-OFFICIAL["grad_clamp"], OFFICIAL["grad_clamp"])
    scheduler.step()
    optimizer.step()
    return {"loss": processed_loss.detach(), "grad_norm": grad_norm.detach(),
            "q_mean": state_action_values.detach().mean()}


def score_actions(model, observation, launches):
    """float32 [K]: Q(s, a(launch)) for one anchor (eval mode, no grad); ranking cost = -Q."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        q_values = model(model_input(model.input_type, observation).unsqueeze(0).to(device))[0]
    if was_training:
        model.train()
    return q_values[action_indices(launches).to(device)].float().cpu()


# ---------------------------------------------------------------------------
# native recipe (TrainLearningAgent.py outer loop on a fixed fit set)
# ---------------------------------------------------------------------------

def official_outer_steps(levels):
    """``TrainLearningAgent.py`` line 219: ``num_update_steps = len(train_level_list) // 5`` (>= 5 levels, else 50)."""
    return levels // 5 if levels >= 5 else 50


def outer_epochs(steps, start=OFFICIAL["train_time_per_ep"], rise=OFFICIAL["train_time_rise"]):
    """Line 224: ``c.train_time_per_ep = max(1, c.train_time_per_ep - int(step * c.train_time_rise))`` per step."""
    epochs, current = [], start
    for step in range(steps):
        current = max(1, current - int(step * rise))
        epochs.append(current)
    return epochs


# The <= 3 pre-declared configurations (Decision P-other, native recipe).
CONFIGURATIONS = {
    "native-code": {"outer_steps": "official", "epochs": "official"},
    "native-paper": {"outer_steps": "official", "epochs": 10},
    "native-template": {"outer_steps": 16, "epochs": "official"},
}


def recipe(configuration, examples, levels):
    """Updates, batch, LR and the outer-step resets of a configuration on ``examples`` fit shots of ``levels`` levels.
    One epoch = ``examples // train_batch`` batches (``DataLoader(drop_last=True)`` over a ``WeightedRandomSampler``
    with ``num_samples = len(dataset)``)."""
    spec = CONFIGURATIONS[configuration]
    steps = official_outer_steps(levels) if spec["outer_steps"] == "official" else spec["outer_steps"]
    epochs = outer_epochs(steps) if spec["epochs"] == "official" else [spec["epochs"]] * steps
    per_epoch = examples // OFFICIAL["train_batch"]
    starts = np.cumsum([0] + [e * per_epoch for e in epochs]).tolist()
    return {"configuration": configuration, "outer_steps": steps, "epochs": epochs, "batches_per_epoch": per_epoch,
            "updates": int(starts[-1]), "resets": [int(s) for s in starts[1:-1]],
            "batch_size": OFFICIAL["train_batch"], "learning_rate": OFFICIAL["learning_rate"]}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--build-symbolic", choices=("fit", "policy"), action="append", default=[])
    args = parser.parse_args(argv)
    for split in args.build_symbolic:
        payload = build_symbolic_store(split)
        print(f"{symbolic_store_path(split)}: {len(payload['states'])} anchors, basemaps {payload['basemap_sha256']}")


if __name__ == "__main__":
    main()
