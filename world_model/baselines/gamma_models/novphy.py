"""#134 T3 γ-model-car: the NovPhy adapter around the vendored official γ-model (``vendor/gamma``, jannerm/gamma-models
@ bb23e06, MIT).

Model (the official architecture defaults; Decision P-other adapter on #134): ``gamma.flows.make_conditional_flow``
with the Colab notebook's ``hidden_dims [256, 256, 256]`` and the function defaults (3 x [ActNorm, invertible 1x1
conv, conditional rational-quadratic spline coupling], K = 16 bins on [-10, 10], logistic prior, no delta
parameterisation) over the 236-d E99 carrier, conditioned on ``{'s': carrier (236), 'a': launch (5)}`` (10 230 696
parameters). Training = the notebook's loop body: ``BootstrapTarget`` (single-step ``N(s', σ = 0.1)`` with weight
``1 - γ`` + the delayed target γ-model at ``(s', a')`` with weight γ), samples drawn from the mixture with the sample
discount (0.9; 0.5 during the 2000-step burn-in, when γ is 0.5 too), the MSE between the model's and the target's
log-densities at those samples, Adam 1e-4 with weight decay 1e-5, Polyak target update τ = 0.005, batch 1024, 50 000
updates. Only γ changes across the pre-declared configurations (``CONFIGURATIONS``).

Transitions (``transition_batch``): one fit shot = the carriers at frames 0, 5, ..., 225 (stride 5, 46 states;
positions past the shot's end repeat its final carrier, as #131) and 46 transitions ``(s_k, a_k) -> s_{k+1}`` for
k = 0..45 with ``s_46 := s_45`` (the frame-225 carrier is absorbing, the same rule as the repeated final carrier):
``a_0`` = the launch (``eb.launch5``), every later action and every next action ``a'`` = the zero no-op (the policy
after the launch). A batch draws transition rows uniformly with replacement (the official ``ReplayPool.sample``) from
the runner's per-batch generator.

Occupancy mapping: a γ-model sample from ``(s_0, a)`` is ``s_k`` with probability ``(1 - γ) γ^(k-1)`` (k ≥ 1; the
occupancy weight on transition k - 1 is ``(1 - γ) γ^(k-1)``); the absorbing frame-225 state carries the tail mass
``γ^44``. ``occupancy_weights`` gives these weights; the score never uses them (the model learns them).

Score (``expected_rows``): the expected tie-free cost (``harness.EndpointCosts``) over S = 64 samples
``x ~ p_γ(· | s_0, a)``; the samples use one fixed base-noise draw (``base_noise``: ``torch.rand`` [S, 236] from a
CPU generator seeded ``SAMPLE_SEED``, pushed through the model's own prior transforms exactly as
``TransformedDistribution.sample``) shared by every candidate (common random numbers). A candidate whose sampling
raises the official inverse-spline assertion, or any of whose S samples is nonfinite, is a retained typed row
(``excluded True``, costs None, ``typed_failure`` set). G-action reads the per-candidate sample mean.

Devices: the official flows create tensors with factory defaults (upstream switches the default tensor type to CUDA);
``build_model`` initialises on CPU from the caller's torch seed, then rebuilds under ``torch.device(device)`` and loads
that state, and every call runs under ``torch.device(device)``. After ``copy.deepcopy`` the target shares the online
model's (parameter-free) prior (torch 2.13 deepcopy shim; VENDOR.json).
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .vendor.gamma.flows import make_conditional_flow
from .vendor.gamma.td.distributions import BootstrapTarget
from .vendor.gamma.td.utils import soft_update_from_to

VENDOR = Path(__file__).resolve().parent / "vendor"
STRIDE = 5
END_FRAME = 225
POSITIONS = tuple(range(0, END_FRAME + 1, STRIDE))  # 46 states
TRANSITIONS_PER_SHOT = len(POSITIONS)  # k = 0..45 (k = 45: the absorbing frame-225 self-transition)
CARRIER_DIM = 236
ACT_DIM = 5
CONDITION_DIMS = {"s": CARRIER_DIM, "a": ACT_DIM}
# the Colab notebook's Args (scripts/gamma-pendulum.ipynb) and make_conditional_flow's defaults
NATIVE = {"hidden_dims": [256, 256, 256], "K": 16, "B": 10, "num_layers": 3, "delta": False, "sigma": 0.1,
          "batch_size": 1024, "lr": 1e-4, "decay": 1e-5, "tau": 0.005, "discount": 0.99, "sample_discount": 0.9,
          "burnin_discount": 0.5, "n_burnin": 2000, "n_steps": 50000}
# pre-declared selection grid (#127: <= 3 configurations; Decision P-other on #134): effective horizons
# 1 / (1 - γ) = 20, 45 and 100 transitions (100, 225 and 500 frames) around the 45-transition cascade; 0.99 is the
# official notebook's discount
CONFIGURATIONS = {"g0.95": {"discount": 0.95}, "g0.978": {"discount": 1.0 - 1.0 / 45.0},
                  "g0.99": {"discount": 0.99}}
SAMPLES = 64
SAMPLE_SEED = 20261025


def config_of(configuration="g0.99", overrides=None):
    config = {**NATIVE, **CONFIGURATIONS[configuration], **(overrides or {})}
    for key in ("discount", "sample_discount", "burnin_discount"):  # BootstrapTarget.sample asserts float
        config[key] = float(config[key])
    return config


def effective_horizon(gamma):
    """Mean of the geometric time offset (transitions): 1 / (1 - γ)."""
    return 1.0 / (1.0 - gamma)


def occupancy_weights(gamma, transitions=TRANSITIONS_PER_SHOT - 1):
    """[46] probability that a sample from (s_0, a) is s_k, k = 1..45 (index k - 1); the last entry carries the
    absorbing tail γ^44 (every k ≥ 45 is the frame-225 carrier)."""
    k = np.arange(1, transitions + 1)
    w = (1.0 - gamma) * gamma ** (k - 1)
    w[-1] = gamma ** (transitions - 1)
    return w


# ---------------------------------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------------------------------

def _make(config, dim, condition_dims):
    return make_conditional_flow(dim=dim, hidden_dims=list(config["hidden_dims"]), condition_dims=dict(condition_dims),
                                 B=config["B"], K=config["K"], num_layers=config["num_layers"], delta=config["delta"])


def build_model(config=None, device="cpu", dim=CARRIER_DIM, condition_dims=None):
    """The official conditional spline flow, initialised on CPU from the caller's torch seed (device-independent
    initial weights), then rebuilt under ``torch.device(device)`` with that state when ``device`` is not the CPU."""
    config = config or config_of()
    condition_dims = condition_dims or CONDITION_DIMS
    model = _make(config, dim, condition_dims)
    if str(device) == "cpu":
        return model
    state = model.state_dict()
    with torch.device(device):
        model = _make(config, dim, condition_dims)
    model.load_state_dict(state)
    return model


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_device(model):
    return next(model.parameters()).device


def copy_target(model):
    """The notebook's ``target_model = copy.deepcopy(model)`` plus the torch 2.13 prior shim."""
    target = copy.deepcopy(model)
    target.prior = model.prior
    return target


def actnorm_flags(model):
    return [bool(getattr(f, "data_dep_init_done", False)) for f in model.flow.flows]


def set_actnorm_flags(model, flags):
    for flow, flag in zip(model.flow.flows, flags):
        if hasattr(flow, "data_dep_init_done"):
            flow.data_dep_init_done = bool(flag)


class Trainer:
    """Model, delayed target, bootstrapped target distribution, Adam and the MSE criterion (the notebook's cells)."""

    def __init__(self, config, device="cpu", dim=CARRIER_DIM, condition_dims=None):
        self.config = config
        self.device = str(device)
        self.model = build_model(config, device, dim, condition_dims)
        self.target = copy_target(self.model)
        self.bootstrap = BootstrapTarget(self.target, config["discount"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"], weight_decay=config["decay"])
        self.criterion = torch.nn.MSELoss()

    def device_context(self):
        return torch.device(self.device) if self.device != "cpu" else contextlib.nullcontext()

    def discounts(self, step):
        """(bootstrap discount, sample discount) of update ``step`` (burn-in: both burnin_discount)."""
        c = self.config
        if step < c["n_burnin"]:
            return c["burnin_discount"], c["burnin_discount"]
        return c["discount"], c["sample_discount"]

    def train_step(self, condition_dict, next_condition_dict, step):
        """The notebook's loop body for update ``step`` on one batch; returns {"loss", "log_prob_target",
        "log_prob_model", "samples"} (detached)."""
        c = self.config
        with self.device_context():
            discount, sample_discount = self.discounts(step)
            self.bootstrap.update_discount(discount)
            self.bootstrap.update_p(next_condition_dict["s"], sigma=c["sigma"])
            samples = self.bootstrap.sample(len(condition_dict["s"]), condition_dict, next_condition_dict,
                                            discount=sample_discount)
            log_prob_target = self.bootstrap.log_prob(samples, condition_dict, next_condition_dict)
            log_prob_model = self.model.log_prob(samples, condition_dict)
            loss = self.criterion(log_prob_model, log_prob_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            soft_update_from_to(self.model, self.target, c["tau"])
        return {"loss": loss.detach(), "log_prob_target": log_prob_target.detach(),
                "log_prob_model": log_prob_model.detach(), "samples": samples}

    def state(self):
        return {"model": self.model.state_dict(), "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(), "model_actnorm": actnorm_flags(self.model),
                "target_actnorm": actnorm_flags(self.target)}

    def load_state(self, state):
        self.model.load_state_dict(state["model"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        set_actnorm_flags(self.model, state["model_actnorm"])
        set_actnorm_flags(self.target, state["target_actnorm"])


# ---------------------------------------------------------------------------------------------------------------
# transitions
# ---------------------------------------------------------------------------------------------------------------

def episode_positions(end):
    """Frames 0, 5, ..., 225 of a shot whose last stored position is ``end`` (later positions repeat ``end``)."""
    return [min(p, int(end)) for p in POSITIONS]


def transition_rows(rng, shots, batch):
    """(shot, k) of ``batch`` transitions drawn uniformly with replacement from shots x 46 (numpy generator)."""
    flat = rng.integers(0, shots * TRANSITIONS_PER_SHOT, size=batch)
    return flat // TRANSITIONS_PER_SHOT, flat % TRANSITIONS_PER_SHOT


def transition_batch(carriers, launch5, shot, k, device="cpu"):
    """Condition dicts of transitions (shot, k) over episodes ``carriers`` [N, 46, 236] / ``launch5`` [N, 5]:
    ``{'s': s_k, 'a': launch if k == 0 else 0}`` and ``{'s': s_{min(k + 1, 45)}, 'a': 0}``."""
    shot = torch.as_tensor(np.asarray(shot), dtype=torch.long)
    k = torch.as_tensor(np.asarray(k), dtype=torch.long)
    s = carriers[shot, k].float()
    s_next = carriers[shot, torch.clamp(k + 1, max=TRANSITIONS_PER_SHOT - 1)].float()
    a = torch.as_tensor(launch5, dtype=torch.float32)[shot] * (k == 0).float()[:, None]
    zeros = torch.zeros(len(shot), ACT_DIM)
    return ({"s": s.to(device), "a": a.to(device)}, {"s": s_next.to(device), "a": zeros.to(device)})


# ---------------------------------------------------------------------------------------------------------------
# resumable training (SIGTERM -> resume.pt at the next update boundary -> exit 75)
# ---------------------------------------------------------------------------------------------------------------

class TrainingPaused(Exception):
    """A stop was requested; resume.pt holds the state after the last completed update."""


def atomic_save(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def rng_state(device):
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    return {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device) if cuda else None}


def set_rng_state(state, device):
    torch.set_rng_state(state["cpu"])
    if state.get("cuda") is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"], device)


def train_loop(trainer, out_dir, steps, batch_fn, should_stop, max_steps=None, log_every=100,
               checkpoint_every=2000):
    """Updates ``trainer.train_step(*batch_fn(step), step)`` for steps [start, steps), resuming from
    ``out_dir/resume.pt`` (model, target, Adam state, ActNorm init flags, torch generators, step, elapsed). The caller
    seeds torch and builds the trainer first. A nonfinite loss, or the official inverse-spline assertion while
    sampling the bootstrapped target, ends the run as a typed failure (``nonfinite_loss`` / ``sampling_assertion``;
    upstream has no check). ``should_stop()`` True -> resume.pt -> ``TrainingPaused``; ``max_steps`` bounds this call
    (tests, throughput). Returns {"start", "stop", "steps_done", "seconds", "elapsed", "failure", "trace"}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume = out_dir / "resume.pt"
    device = trainer.device
    start, elapsed = 0, 0.0
    if resume.is_file():
        state = torch.load(resume, map_location=device, weights_only=False)
        trainer.load_state(state["trainer"])
        set_rng_state(state["rng"], device)
        start, elapsed = state["step"], state["elapsed"]
    stop_at = steps if max_steps is None else min(steps, start + max_steps)
    began, failure, trace, step = time.monotonic(), None, [], start - 1
    with open(out_dir / "train-log.jsonl", "a") as log_file:
        for step in range(start, stop_at):
            try:
                out = trainer.train_step(*batch_fn(step), step)
            except AssertionError as error:  # gamma/flows/spline.py: assert (discriminant >= 0).all()
                failure = {"typed_failure": "sampling_assertion", "step": step, "message": repr(error)}
                break
            except ValueError as error:  # torch.distributions argument validation of nonfinite values
                if "invalid values" not in str(error):
                    raise
                failure = {"typed_failure": "nonfinite_loss", "step": step, "message": str(error).splitlines()[0]}
                break
            loss = float(out["loss"])
            if not np.isfinite(loss):
                failure = {"typed_failure": "nonfinite_loss", "step": step, "loss": str(loss)}
                break
            if step % log_every == 0 or step == steps - 1:
                entry = {"step": step, "seconds": elapsed + time.monotonic() - began, "loss": loss,
                         "log_prob_model_mean": float(out["log_prob_model"].mean()),
                         "log_prob_target_mean": float(out["log_prob_target"].mean())}
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()
                trace.append(entry)
            stop = bool(should_stop()) and step + 1 < steps
            if stop or ((step + 1) % checkpoint_every == 0 and step + 1 < steps):
                atomic_save(resume, {"trainer": trainer.state(), "rng": rng_state(device), "step": step + 1,
                                     "elapsed": elapsed + time.monotonic() - began})
            if stop:
                raise TrainingPaused(f"paused at update {step + 1} of {steps}")
    if failure is None and step + 1 < steps and step + 1 > start:  # bounded call: keep the state for the next one
        atomic_save(resume, {"trainer": trainer.state(), "rng": rng_state(device), "step": step + 1,
                             "elapsed": elapsed + time.monotonic() - began})
    seconds = time.monotonic() - began
    return {"start": start, "stop": step + 1, "steps_done": step + 1 - start, "seconds": seconds,
            "elapsed": elapsed + seconds, "failure": failure, "trace": trace}


# ---------------------------------------------------------------------------------------------------------------
# sampling the occupancy and the expected cost
# ---------------------------------------------------------------------------------------------------------------

def base_noise(samples=SAMPLES, seed=SAMPLE_SEED, dim=CARRIER_DIM):
    """The fixed uniform base draw [S, dim] (CPU generator; device-independent)."""
    return torch.rand(samples, dim, generator=torch.Generator().manual_seed(seed))


def prior_transform(model, u):
    """The prior's sample path on a given uniform draw (TransformedDistribution.sample after Uniform.sample)."""
    base = model.prior.base_dist
    z = base.low + u * (base.high - base.low)
    for transform in model.prior.transforms:
        z = transform(z)
    return z


@torch.no_grad()
def _transform(model, z, condition):
    device = model_device(model)
    with torch.device(device) if device.type != "cpu" else contextlib.nullcontext():
        return model.flow.transform(z, condition)


@torch.no_grad()
def sample_occupancy(model, z0, launch5, u):
    """Samples [B, S, 236] of p_γ(· | s_0 = z0[b], a = launch5[b]) from base draw u [S, 236] (candidate-major rows),
    and a bool [B] of candidates whose sampling raised the official inverse-spline assertion (their rows are NaN)."""
    device = model_device(model)
    b, s = len(z0), len(u)
    z = prior_transform(model, u.to(device))
    cond = {"s": z0.to(device).float().repeat_interleave(s, 0), "a": launch5.to(device).float().repeat_interleave(s, 0)}
    try:
        x = _transform(model, z.repeat(b, 1), cond)
        return x.reshape(b, s, -1), torch.zeros(b, dtype=torch.bool)
    except AssertionError:
        if b == 1:
            return torch.full((1, s, z.shape[1]), float("nan"), device=device), torch.ones(1, dtype=torch.bool)
    halves = [sample_occupancy(model, z0[i:j], launch5[i:j], u) for i, j in ((0, b // 2), (b // 2, b))]
    return torch.cat([h[0] for h in halves]), torch.cat([h[1] for h in halves])


def sample_candidates(model, z0, launch5, u, chunk=64):
    """sample_occupancy over candidates in chunks; CPU tensors (samples [B, S, 236], failed [B])."""
    xs, failed = [], []
    for i in range(0, len(z0), chunk):
        x, f = sample_occupancy(model, z0[i:i + chunk], launch5[i:i + chunk], u)
        xs.append(x.float().cpu())
        failed.append(f)
    return torch.cat(xs), torch.cat(failed)


def _cost_keys():
    from world_model.planning import scoring_harness as harness
    return (harness.PRIMARY_COST, harness.SECONDARY_COST, "pig_presence", "pig_displacement")


def expected_row(ordinal, samples, z0, costs, sampling_failed=False):
    """The expected EndpointCosts row of one candidate over its S samples [S, 236]; a sampling assertion or any
    nonfinite sample -> excluded typed row."""
    keys = _cost_keys()
    s = len(samples)
    if sampling_failed:
        return {"ordinal": ordinal, "excluded": True, **dict.fromkeys(keys), "samples": s, "nonfinite_samples": s,
                "typed_failure": "sampling_assertion"}
    rows = [costs.row(ordinal, x, z0) for x in samples]
    bad = sum(1 for r in rows if r["excluded"])
    if bad:
        return {"ordinal": ordinal, "excluded": True, **dict.fromkeys(keys), "samples": s, "nonfinite_samples": bad,
                "typed_failure": "nonfinite_samples"}
    return {"ordinal": ordinal, "excluded": False, **{k: float(np.mean([r[k] for r in rows])) for k in keys},
            "samples": s, "nonfinite_samples": 0, "typed_failure": None}


def expected_rows(items, samples, z0, costs, failed=None):
    """Rows of candidates ``items`` from their samples [K, S, 236] and the anchor z0 [236]."""
    failed = torch.zeros(len(items), dtype=torch.bool) if failed is None else failed
    return [expected_row(it["ordinal"], samples[i], z0, costs, bool(failed[i])) for i, it in enumerate(items)]


def sample_means(samples):
    """Per-candidate mean carrier [..., 236] over the sample axis (G-action input); NaN where a sample is NaN."""
    return samples.mean(dim=-2)
