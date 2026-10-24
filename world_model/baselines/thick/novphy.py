"""#131 D1 THICK: the NovPhy adapter around the G-port-verified port (``port.py``).

Model: native THICK (``thick_pinpad`` configuration of the vendored ``configs.yaml``: C-RSSM with deter / hidden 256,
32 x 32 discrete latents, a 16-d GateL0RD context, THICKNet 1024 wide with 5 high-level actions, Adam 1e-4) with the
image pathway replaced by THICK's own MLP pathway (``mlp_keys``): the observation is the 236-d E99 carrier (encoder
and decoders ``mlp_layers`` 4 x 400, ``mse`` heads), exactly the official code path for vector observations.

Episodes: one fit shot = 46 observations, the carriers at frames 0, 5, ..., 225 (stride 5; positions past the shot's
end repeat its final carrier, the scene being at rest). Actions follow the upstream transition layout
(``common/driver.py``: ``action[t]`` is the action that led into observation ``t``, index 0 the reset's zero action):
the launch (5-d ``eb.launch5``) is applied at t0, so it sits at index 1, and every later action is the zero no-op.
There is no task reward (``reward`` 0, no early termination, ``pred_discount`` False as VisualPinPad).

Readout (scoring): observe the anchor carrier (t0), then roll the low level 45 steps with the launch at the first
step and no-ops after (``img_step``), deterministically: latent modes and zero GateL0RD noise (upstream
``eval_ctxt_mean: True``; upstream ``ctxt_always_sample: True`` would keep the noise on at evaluation). The
endpoint carrier is the decoder mean at frame 225; the commit points are the rollout steps whose context gates open
(THICK's adaptive Delta). The high level's jump prediction from the post-launch state is an exploratory readout.
"""
from __future__ import annotations

import numpy as np
import torch

from world_model.baselines.thick import port

STRIDE = 5
END_FRAME = 225
POSITIONS = tuple(range(0, END_FRAME + 1, STRIDE))  # 46 observations
ROLLOUT_STEPS = len(POSITIONS) - 1  # 45
CARRIER_DIM = 236
ACT_DIM = 5
LAUNCH_INDEX = 1
CONFIG_NAMES = ("thick_pinpad",)
CONFIG_OVERRIDES = {"encoder.mlp_keys": "carrier", "encoder.cnn_keys": "$^",
                    "decoder.mlp_keys": "carrier", "decoder.cnn_keys": "$^"}
# pre-declared selection grid (#127: <= 3 configurations; THICK's tuning guide: beta_sparse in {1, 5, 10, 25, 50};
# the paper's per-suite values are 1 (VisualPinPad, MiniHack-Corridor), 10 (MiniHack) and 25 (Multiworld))
CONFIGURATIONS = {"sparsity-1": {"loss_scales.ctxt_sparsity": 1.0},
                  "sparsity-10": {"loss_scales.ctxt_sparsity": 10.0},
                  "sparsity-25": {"loss_scales.ctxt_sparsity": 25.0}}
BATCH = 16  # native THICK batch (16 x 50 steps); a NovPhy episode is 46 steps


def obs_space():
    """Upstream obs-space order: the carrier (MLP key), then the scalar keys."""
    return {"carrier": (CARRIER_DIM,), "reward": (), "is_first": (), "is_last": (), "is_terminal": ()}


def build_config(configuration="sparsity-1", overrides=None):
    return port.load_config(CONFIG_NAMES, {**CONFIG_OVERRIDES, **CONFIGURATIONS[configuration], **(overrides or {})})


def build_model(config=None):
    """The THICK world model (low level + high level) for carriers and the continuous 5-d launch."""
    return port.WorldModel(config or build_config(), obs_space(), ACT_DIM, cont_action=True)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def episode_positions(end):
    """Frames 0, 5, ..., 225 of a shot whose last stored position is ``end`` (later positions repeat ``end``)."""
    return [min(p, int(end)) for p in POSITIONS]


def episode_actions(launch5):
    """[B, 46, 5]: zeros with the launch at index 1 (applied at t0, leading into the frame-5 observation).

    THICK's continuous high-level action head is a truncated normal on [-1, 1] (upstream ``NormalizeAction``
    convention); every fit-split launch5 lies inside (drag_x / 480 in [-0.34, -0.02], drag_y / 480 in [0.01, 0.17],
    release 1, tap 0, indicator 1)."""
    launch5 = torch.as_tensor(launch5, dtype=torch.float32)
    if launch5.numel() and float(launch5.abs().max()) > 1.0:
        raise ValueError("launch5 outside THICK's [-1, 1] action range")
    actions = torch.zeros(launch5.shape[0], len(POSITIONS), ACT_DIM)
    actions[:, LAUNCH_INDEX] = launch5
    return actions


def episode_batch(carriers, launch5):
    """WorldModel inputs for carriers [B, 46, 236] and launches [B, 5]."""
    carriers = torch.as_tensor(carriers, dtype=torch.float32)
    b, t = carriers.shape[:2]
    if t != len(POSITIONS) or carriers.shape[2] != CARRIER_DIM:
        raise ValueError(f"carriers {tuple(carriers.shape)} != [B, {len(POSITIONS)}, {CARRIER_DIM}]")
    is_first = torch.zeros(b, t, dtype=torch.bool)
    is_first[:, 0] = True
    is_last = torch.zeros(b, t, dtype=torch.bool)
    is_last[:, -1] = True
    return {"carrier": carriers, "action": episode_actions(launch5), "reward": torch.zeros(b, t),
            "is_first": is_first, "is_last": is_last, "is_terminal": torch.zeros(b, t, dtype=torch.bool)}


def fit_episodes(fit, shots):
    """Episode batch of fit shots (``fit`` = ``eb.FitShots()``; rows of its frozen order)."""
    shots = [int(s) for s in shots]
    pairs = [(s, p) for s in shots for p in episode_positions(fit.ends[s])]
    rows = fit.carrier_rows([s for s, _ in pairs], [p for _, p in pairs])
    carriers = torch.as_tensor(np.asarray(rows), dtype=torch.float32).reshape(len(shots), len(POSITIONS), CARRIER_DIM)
    return episode_batch(carriers, fit.launch5[shots])


def sample_shots(rng, count, batch=BATCH):
    """Shot rows of one training batch (uniform with replacement over the fit shots, as upstream's replay)."""
    return rng.integers(0, count, batch)


class ZeroNoise:
    """Deterministic readout: zero GateL0RD noise; latent samples are never drawn (modes are used)."""

    def normal(self, shape, device=None):
        return torch.zeros(tuple(shape), device=device)

    def uniform(self, shape, device=None):
        raise RuntimeError("the deterministic readout draws no one-hot sample")


@torch.no_grad()
def rollout(model, anchor_carriers, launch5, steps=ROLLOUT_STEPS):
    """Deterministic low-level rollout from anchor carriers [N, 236] under launches [N, 5].

    Returns ``endpoint`` [N, 236] (decoder mean at frame 5 * steps), ``carriers`` [N, steps + 1, 236] (t0 posterior
    reconstruction, then the imagined frames), ``gates`` [N, steps, 16] (the GateL0RD gates of every imagined step),
    ``commit_steps`` (per candidate: rollout steps 1..steps whose context changed), ``update_counts`` [N],
    ``coarse_endpoint`` [N, 236] (the context decoder at the endpoint, exploratory).
    """
    model.eval()
    noise = ZeroNoise()
    rssm = model.rssm
    anchor_carriers = torch.as_tensor(anchor_carriers, dtype=torch.float32)
    n = anchor_carriers.shape[0]
    device = next(model.parameters()).device
    anchor_carriers = anchor_carriers.to(device)
    launch5 = torch.as_tensor(launch5, dtype=torch.float32, device=device)
    embed = model.encoder({"carrier": anchor_carriers})
    state, _ = rssm.obs_step(rssm.initial(n, device), torch.zeros(n, ACT_DIM, device=device), embed,
                             torch.ones(n, dtype=torch.bool, device=device), sample=False, rnn_sample=False,
                             noise=noise)
    decode = lambda s: model.heads["decoder"](rssm.get_feat(s))["carrier"].mode()
    frames, gates = [decode(state)], []
    for step in range(steps):
        action = launch5 if step == 0 else torch.zeros_like(launch5)
        state = rssm.img_step(state, action, sample=False, rnn_sample=False, noise=noise)
        frames.append(decode(state))
        gates.append(state["gates"])
    gates = torch.stack(gates, 1)
    opened = gates.sum(-1) > 0
    commit_steps = [[int(t) + 1 for t in torch.nonzero(row).flatten()] for row in opened.cpu()]
    coarse = model.ctxt_heads["decoder"](rssm.get_coarse_out(state))["carrier"].mode()
    return {"endpoint": frames[-1], "carriers": torch.stack(frames, 1), "gates": gates,
            "commit_steps": commit_steps, "update_counts": opened.sum(1), "coarse_endpoint": coarse,
            "final_state": state}


@torch.no_grad()
def high_level_jump(model, anchor_carriers, launch5):
    """Exploratory: THICK's high-level prediction from the post-launch state (mode of the prior high-level
    action): the predicted latent before the next context change, its decoded coarse carrier and the predicted
    number of steps until it (``thick_time``)."""
    model.eval()
    out = rollout(model, anchor_carriers, launch5, steps=1)
    state = out["final_state"]
    inps = port.gen_thick_inps(state["context"], state["stoch"])
    pred = model.thick_net.sample_step(inps, sample=False, noise=ZeroNoise())
    coarse = torch.cat([port.flatten_logits(pred["thick_stoch"]), state["context"]], -1)
    return {"thick_time": pred["thick_time"].squeeze(-1),
            "carrier": model.ctxt_heads["decoder"](coarse)["carrier"].mode()}
