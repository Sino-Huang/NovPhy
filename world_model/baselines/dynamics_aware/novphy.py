"""#128 A4 DynDQN: the hand-crafted dynamics-aware DQN (Ahmed et al. 2021, arXiv 2102.10336) on NovPhy.

Model: the vendored official ``ResNetFilmAction`` (``vendor/agents/nets.py``, facebookresearch/DynamicsAware @ eb65576)
in the paper's Table-1 configuration (ResNet-18, FiLM before the last block, action MLP 1 x 256, linear embeddor
512 -> 256, outer-product merging, 20-bin distance classifier). NovPhy input (as A1): the 320 x 240 RGB level-anchor
frame (uint8 / 255, the official stem with 3 input channels instead of the 7-colour one-hot) and ``eb.launch4``.

Loss: the official ``neural_agent.sample_distance_loss`` / ``comparative_loss_single_batch`` and
``rollout_dist.measure_distances`` (time_slice 0), written here so the NovPhy trajectory form plugs in; G-port
(``scripts/b127_dynamics_aware_port_reference.py``) holds the PHYRE configuration of this code to the official code.

Trajectory mapping (Decision P-other on #128): the #123 fit probe labels (engine slot presence [P, 18] and
engine-projected centres [P, 18, 2] at every 15th position) stand in for PHYRE's featurized objects at stride 60:
- time grid: positions 0, 15, ... <= the shot end (the clamped non-multiple end position is dropped, so equal indices
  are equal times, as PHYRE's stride grid); pairs use the joint valid prefix (official T = min(t1, t2));
- objects: the "moving" set is the dynamic slots (bird, block, pig) present at position 0 of rollout_1 (official: the
  red/green/blue/gray = dynamic colours at frame 0 of rollout_1); platforms, slingshot and terrain are static;
- coordinates: isotropic screen units (x / W, y / W; viewport 640 x 480) and alpha = the screen diagonal
  sqrt(1 + 0.75^2) = 1.25 (official: unit-square scene, alpha = 1.414 ~ sqrt 2, the largest possible distance);
- destroyed objects (absent in NovPhy, never in PHYRE): present in exactly one rollout -> distance alpha (maximally
  dissimilar), absent in both -> 0 (identical); the (o, t) still counts in the normaliser.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from world_model.baselines.dynamics_aware.vendor.agents import nets

# Table-1 recipe (06_hand_crafted_test_run_table_1.conf + run_sweep_file base command + offline_agents defaults)
NATIVE = {"updates": 100_000, "train_batch_size": 512, "n_samples_per_task": 8, "learning_rate": 3e-4,
          "cosine_scheduler": True, "balance_classes": True, "network": "resnet18", "fusion_place": "last",
          "action_layers": 1, "action_hidden_size": 256}
OFFICIAL_HP = {"regression_type": "binned", "repr_merging_method": "outer", "distance_cutoff": 1.414,
               "n_regression_bins": 20, "time_slice": 0, "embedding_dim": 256, "embeddor_type": "linear",
               "weight": 1.0}
VIEWPORT = (640, 480)
ASPECT = VIEWPORT[1] / VIEWPORT[0]
NOVPHY_CUTOFF = math.sqrt(1.0 + ASPECT ** 2)  # 1.25
NOVPHY_HP = {**OFFICIAL_HP, "distance_cutoff": NOVPHY_CUTOFF}
POSITION_STRIDE = 15
MAX_POSITION = 285
T_MAX = MAX_POSITION // POSITION_STRIDE + 1  # 20
DYNAMIC_PREFIXES = ("bird:", "block:", "pig:")
PHYRE_MOVING_COLOURS = (8, 9, 10, 12)  # red, green, blue, gray in PHYRE's featurized objects
LABELS = "issue-123-hjepa-comparison-v1/labels/fit"


class A4Net(nets.ResNetFilmAction):
    """The official ResNetFilmAction; ``in_channels=3`` swaps the 7-colour one-hot stem input for RGB / 255 (A1's
    convention), ``in_channels=7`` is the unchanged official model (G-port)."""

    def __init__(self, in_channels=3, action_size=4, hp=None, action_hidden_size=NATIVE["action_hidden_size"]):
        hp = {**NOVPHY_HP, **(hp or {})}
        super().__init__(action_size, action_layers=NATIVE["action_layers"], action_hidden_size=action_hidden_size,
                         fusion_place=NATIVE["fusion_place"], embedding_dim=hp["embedding_dim"],
                         repr_merging_method=hp["repr_merging_method"],
                         n_regressor_outputs=1 if hp["regression_type"] == "mse" else hp["n_regression_bins"],
                         embeddor_type=hp["embeddor_type"], backbone=NATIVE["network"])
        if in_channels not in (3, nets.phyre.NUM_COLORS):
            raise ValueError(f"in_channels must be 3 (RGB) or {nets.phyre.NUM_COLORS} (official one-hot)")
        self.rgb = in_channels == 3
        if self.rgb:
            self.stem[0] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def _image_colors_to_onehot(self, indices):
        if self.rgb:  # uint8 [B, 3, H, W] -> float [0, 1]
            return indices.to(self.embed_weights.device, torch.float32) / 255.0
        return super()._image_colors_to_onehot(indices)


def parameter_count(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ---------------------------------------------------------------------------
# trajectories: [n, T, O, 2] coordinates, [n, T, O] presence, [n, T] valid frames, [n, O] moving objects
# ---------------------------------------------------------------------------

def phyre_trajectories(rollouts, masks):
    """Official PHYRE featurized rollouts [n, T, 65, 14] and padding masks [n, T] (1 = padded) in this form."""
    rollouts = torch.as_tensor(rollouts)
    masks = torch.as_tensor(masks)
    return {"coords": rollouts[..., :2], "present": torch.ones(rollouts.shape[:3], dtype=torch.bool),
            "frame_valid": masks == 0,
            "moving": (rollouts[:, 0, :, list(PHYRE_MOVING_COLOURS)] > 0.5).any(dim=2)}


def novphy_trajectory(entry, vocabulary, t_max=T_MAX):
    """One fit shot's #123 label entry -> the trajectory form on the equal-time 15-position grid (padded to t_max)."""
    positions = torch.as_tensor(entry["positions"])
    keep = (positions % POSITION_STRIDE) == 0
    if not bool(keep[0]) or int(positions[0]) != 0:
        raise ValueError("label positions must start at frame 0")
    presence = torch.as_tensor(entry["presence"])[keep].bool()
    centers = torch.as_tensor(entry["centers"])[keep].float()
    t = presence.shape[0]
    if t > t_max:
        raise ValueError(f"{t} label frames > t_max {t_max}")
    o = len(vocabulary)
    coords = torch.zeros(t_max, o, 2)
    coords[:t, :, 0] = centers[:, :, 0]
    coords[:t, :, 1] = centers[:, :, 1] * ASPECT
    present = torch.zeros(t_max, o, dtype=torch.bool)
    present[:t] = presence
    frame_valid = torch.zeros(t_max, dtype=torch.bool)
    frame_valid[:t] = True
    dynamic = torch.tensor([name.startswith(DYNAMIC_PREFIXES) for name in vocabulary])
    return {"coords": coords, "present": present, "frame_valid": frame_valid, "moving": presence[0] & dynamic}


def stack_trajectories(items):
    return {k: torch.stack([item[k] for item in items]) for k in ("coords", "present", "frame_valid", "moving")}


def select(traj, index):
    return {k: v[index] for k, v in traj.items()}


def load_fit_trajectories(keys, labels_dir=None):
    """The fit shots' trajectories in ``keys`` order from the #123 label shards (fit split only)."""
    from pathlib import Path
    from scripts.run_slot_encoder_fix import vocabulary
    labels_dir = Path(labels_dir) if labels_dir else Path(__file__).resolve().parents[3] / ".local-artifacts" / LABELS
    vocab = vocabulary()
    store = {}
    for path in sorted(labels_dir.glob("labels-*.pt")):
        store.update(torch.load(path, weights_only=True)["shots"])
    missing = [k for k in keys if k not in store]
    if missing:
        raise ValueError(f"{len(missing)} fit shots lack #123 labels, e.g. {missing[:3]}")
    return stack_trajectories([novphy_trajectory(store[k], vocab) for k in keys])


def trajectory_distances(traj, cutoff, time_slice=0):
    """All-pairs normalised clipped distance [n, n] (= 1 - v_{a,a'} of the paper) within one task. Entry [a, b]
    compares rollout b (rollout_1: its frame-0 moving set counts) with rollout a, the official
    ``measure_all_distances`` layout (``cartesian_product``)."""
    coords, present, valid, moving = traj["coords"], traj["present"], traj["frame_valid"], traj["moving"]
    n, t = valid.shape
    joint = valid[None, :, :] & valid[:, None, :]  # [a, b, T]
    if time_slice != 0:
        index = torch.arange(1, t + 1).expand(n, n, t)
        if time_slice > 0:
            joint = joint & (index <= time_slice)
        else:
            joint = joint & (index > joint.sum(dim=2, keepdim=True) + time_slice)
    total = moving[None, :, None, :] & joint[..., None]  # [a, b, T, O]
    dists = ((coords[None, :] - coords[:, None]) ** 2).sum(dim=-1) ** 0.5
    dists = torch.clamp(dists, max=cutoff)
    p1, p2 = present[None, :], present[:, None]
    dists = torch.where(p1 & p2, dists, torch.where(p1 ^ p2, torch.full_like(dists, cutoff), torch.zeros_like(dists)))
    dists = torch.where(total, dists, torch.zeros_like(dists)).sum(dim=(2, 3))
    return dists / (total.float().sum(dim=(2, 3)) * cutoff)


def pair_losses(model, embeddings, distances, hp):
    """Official comparative_loss_single_batch on one task's n embeddings and its [n, n] distances."""
    n = embeddings.shape[0]
    left = embeddings.view(n, 1, -1).expand(n, n, -1).reshape(n * n, -1)
    right = embeddings.view(1, n, -1).expand(n, n, -1).reshape(n * n, -1)
    predictions = model.compute_regressed_distances(left, right)
    distances = distances.reshape(-1).to(embeddings.device).float()
    if hp["regression_type"] == "binned":
        labels = torch.round(distances * (model.n_regressor_outputs - 1)).long()
        return nn.functional.cross_entropy(predictions, labels, reduction="none")
    return nn.functional.mse_loss(predictions.view(-1), distances, reduction="none")


def aux_loss(model, embeddings, traj, n_samples_per_task, hp):
    """Official sample_distance_loss: consecutive groups of n_samples_per_task share a task; mean over all pairs."""
    total, count = 0, 0
    for i in range(0, embeddings.shape[0], n_samples_per_task):
        group = select(traj, slice(i, i + n_samples_per_task))
        losses = pair_losses(model, embeddings[i:i + n_samples_per_task],
                             trajectory_distances(group, hp["distance_cutoff"], hp.get("time_slice", 0)), hp)
        total = total + losses.sum()
        count += losses.numel()
    return total / count


def losses(model, observations, actions, is_solved, traj, n_samples_per_task=NATIVE["n_samples_per_task"], hp=None):
    """The official training objective: BCE on the success logit + weight x hand-crafted aux loss."""
    hp = {**NOVPHY_HP, **(hp or {})}
    logits, embeddings = model(observations, actions, get_embeddings=True)
    ce = model.ce_loss(logits, is_solved).mean()
    aux = aux_loss(model, embeddings, traj, n_samples_per_task, hp)
    return {"logits": logits, "embeddings": embeddings, "ce": ce, "aux": aux, "loss": ce + aux * hp["weight"]}


def make_optimizer(model, updates=NATIVE["updates"], learning_rate=NATIVE["learning_rate"]):
    """Adam + CosineAnnealingLR(T_max=updates), as neural_agent.train."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates)


def train_step(model, optimizer, scheduler, observations, actions, is_solved, traj,
               n_samples_per_task=NATIVE["n_samples_per_task"], hp=None):
    """One official update, in the official order: scheduler.step() first, then forward, zero_grad, backward, step."""
    scheduler.step()
    model.train()
    out = losses(model, observations, actions, is_solved, traj, n_samples_per_task, hp)
    optimizer.zero_grad()
    out["loss"].backward()
    optimizer.step()
    return out


def task_balanced_indices(rng, positive_in_task, negative_in_task, n_tasks_per_batch,
                          n_samples_per_task=NATIVE["n_samples_per_task"]):
    """Official task_balanced_sampler (one batch): n_tasks_per_batch distinct tasks, per task n/2 negatives then n/2
    positives drawn with replacement; a task without positives takes negatives (official fallback; the fit split has
    55 all-fail levels and no all-success level); a task without negatives would take positives (a guard only: the
    official code assumes negatives exist, and no fit level lacks them)."""
    tasks = rng.choice(len(positive_in_task), size=n_tasks_per_batch, replace=False)
    half = n_samples_per_task // 2
    indices = []
    for task in tasks:
        neg, pos = negative_in_task[task], positive_in_task[task]
        first = neg if len(neg) else pos
        second = pos if len(pos) else neg
        indices.append(rng.choice(first, half))
        indices.append(rng.choice(second, half))
    return np.concatenate(indices), tasks


# ---------------------------------------------------------------------------
# runner-facing interface (scripts/run_issue_128_direct_predictors.py)
# ---------------------------------------------------------------------------

def build_model():
    """The A4 network on NovPhy (the runner seeds torch first)."""
    return A4Net(in_channels=3, action_size=4)


def runner_trainset(fit, anchors):
    """Fit-split training data: per-level anchor frames (the PHYRE 'initial scene'), per-shot level index, launch4,
    verdict and trajectory, and the per-level positive / negative shot indices of the task-balanced sampler."""
    levels = sorted(set(fit.members))
    level_of = {m: i for i, m in enumerate(levels)}
    task = torch.tensor([level_of[m] for m in fit.members], dtype=torch.long)
    verdicts = torch.as_tensor(fit.verdicts).bool()
    shots = np.arange(len(fit.keys))
    task_np, verdict_np = task.numpy(), verdicts.numpy()
    positive = [shots[(task_np == t) & verdict_np] for t in range(len(levels))]
    negative = [shots[(task_np == t) & ~verdict_np] for t in range(len(levels))]
    return {"levels": levels, "observations": torch.stack([torch.as_tensor(anchors[m]) for m in levels]),
            "task": task, "actions": torch.as_tensor(fit.launch4).float(), "verdicts": verdicts,
            "traj": load_fit_trajectories(list(fit.keys)), "positive": positive, "negative": negative}


def data_to(data, device):
    moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}
    moved["traj"] = {k: v.to(device) for k, v in data["traj"].items()}
    return moved


def runner_batch(data, rng, n_tasks_per_batch=NATIVE["train_batch_size"] // NATIVE["n_samples_per_task"]):
    """Shot indices of one native batch (64 levels x 8 shots); rng = eb.batch_rng(seed, b)."""
    indices, _ = task_balanced_indices(rng, data["positive"], data["negative"], n_tasks_per_batch)
    return indices


def runner_train_step(model, optimizer, scheduler, data, indices, device):
    """One official update (scheduler.step() first; the runner must not step the scheduler again)."""
    idx = torch.as_tensor(indices, dtype=torch.long, device=data["task"].device)
    out = train_step(model, optimizer, scheduler, data["observations"][data["task"][idx]].to(device),
                     data["actions"][idx].to(device), data["verdicts"][idx].to(device),
                     {k: v.to(device) for k, v in select(data["traj"], idx).items()})
    return {"loss": out["loss"].detach(), "ce": out["ce"].detach(), "aux": out["aux"].detach()}


@torch.no_grad()
def score_actions(model, observation, actions):
    """Success logits [K] of K launch4 actions on one level-anchor frame (neural_agent.eval_actions: the trunk up to
    the fusion point once, then the FiLM block and head per action)."""
    model.eval()
    device = model.embed_weights.device
    preprocessed = model.preprocess(torch.as_tensor(observation).unsqueeze(0).to(device))
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
    features = preprocessed["features"].expand(actions.shape[0], -1, -1, -1)
    return model(None, actions, preprocessed={"features": features}).float()
