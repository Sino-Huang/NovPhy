"""#133 T1 DreamerV3: the NovPhy adapter around the vendored official world model (``vendor/``, NM512/dreamerv3-torch
@ 6ef8646, MIT).

Model: the official ``models.WorldModel`` in the official ``defaults`` configuration (RSSM: deter 512, hidden 512,
32 x 32 discrete latents with 1 % unimix, learned initial state, LayerNorm + SiLU; KL balancing dyn 0.5 / rep 0.1 with
free bits 1.0; Adam 1e-4, eps 1e-8, grad clip 1000; batch 16 x 64) with the official vector-observation pathway of
the ``dmc_proprio`` block (``mlp_keys '.*'``, ``cnn_keys '$^'``): the observation is the 236-d E99 carrier, encoded by
the 5 x 1024 MLP with symlog inputs and reconstructed by the 5 x 1024 MLP decoder with the symlog-MSE output. There is
no task reward: the reward and continue heads are removed (the official optimizer is rebuilt over the remaining
parameters), which is the official model trained with ``reward_head.loss_scale = cont_head.loss_scale = 0`` (G-port).
The training step is the official ``WorldModel._train`` (encoder, ``RSSM.observe`` with sampled posteriors, the
decoder's ``log_prob`` loss, ``RSSM.kl_loss``, ``tools.Optimizer``); the only override is ``preprocess``, which skips
the upstream ``obs['image'] / 255`` when there is no image.

Episodes: one fit shot = 46 observations, the carriers at frames 0, 5, ..., 225 (stride 5; positions past the shot's
end repeat its final carrier, as #131). Actions follow the upstream transition layout (``tools.simulate``:
``action[t]`` led into observation ``t``; index 0 is the reset): the launch (``eb.launch5``) sits at index 1 (applied
at t0) and every later action is the zero no-op. Training rows are drawn as upstream ``tools.sample_episodes`` draws
them (length 64: a random start inside a first episode, then whole episodes appended, ``is_first`` at every segment
start), with the generator of each batch fixed by the runner (resumable).

Readout (scoring): observe the anchor carrier at t0 (``obs_step`` from the initial state), then 45 ``img_step``s with
the launch at the first step and no-ops after, deterministically (latent modes, ``sample=False``); the endpoint carrier
is the decoder's mode (``symexp`` of its output) at frame 225.
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import types
from pathlib import Path

import numpy as np
import torch
import yaml

from .vendor import models, networks, tools

VENDOR = Path(__file__).resolve().parent / "vendor"
STRIDE = 5
END_FRAME = 225
POSITIONS = tuple(range(0, END_FRAME + 1, STRIDE))  # 46 observations
ROLLOUT_STEPS = len(POSITIONS) - 1  # 45
CARRIER_DIM = 236
ACT_DIM = 5
LAUNCH_INDEX = 1
VECTOR_BLOCKS = ("defaults", "dmc_proprio")  # the official vector-observation configuration
SHAPES = {"carrier": (CARRIER_DIM,), "is_first": (), "is_last": (), "is_terminal": ()}
HEADS = ("decoder",)  # no reward / continue heads (offline, task-free data)
# pre-declared selection grid (#127: <= 3 configurations; Decision P-other on #133): the official defaults, a 3x
# larger model learning rate, and DreamerV3's other native knob, a 4x lower replay ratio (fewer updates)
CONFIGURATIONS = {"native": {}, "lr3e-4": {"model_lr": 3e-4}, "rr128": {"train_ratio": 128}}


# ---------------------------------------------------------------------------------------------------------------
# configuration (the official configs.yaml, merged as dreamer.py merges it)
# ---------------------------------------------------------------------------------------------------------------

class _Yaml12Loader(yaml.SafeLoader):
    """YAML 1.2 floats (upstream reads configs.yaml with ruamel.yaml, where ``1e-4`` is a float, not a string)."""


_Yaml12Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*)(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"),
    list("-+0123456789"))


def official_configs():
    return yaml.load((VENDOR / "configs.yaml").read_text(), Loader=_Yaml12Loader)


def recursive_update(base, update):
    """dreamer.py's merge of named config blocks (verbatim semantics)."""
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value


def build_config(overrides=None, blocks=VECTOR_BLOCKS, device="cpu", num_actions=ACT_DIM):
    """The official config (blocks merged in order) with dotted-key overrides, ``num_actions`` and ``device`` (both set
    by dreamer.py's main at run time)."""
    merged = {}
    configs = official_configs()
    for name in blocks:
        recursive_update(merged, configs[name])
    for key, value in (overrides or {}).items():
        target, parts = merged, key.split(".")
        for part in parts[:-1]:
            target = target[part]
        if parts[-1] not in target:
            raise KeyError(f"unknown config key {key}")
        target[parts[-1]] = value
    config = types.SimpleNamespace(**merged)
    config.num_actions = int(num_actions)
    config.device = str(device)
    return config


@contextlib.contextmanager
def mlp_default_device(device):
    """networks.MLP's default ``device='cuda'`` (MultiEncoder / MultiDecoder pass none) only builds a fixed-std tensor
    that the encoder and symlog-MSE decoder paths never read; constructing it fails on a CUDA-less host. While a model
    is built, the default is the run device (P-route; no vendored file is edited)."""
    init = networks.MLP.__init__
    defaults = init.__defaults__
    names = init.__code__.co_varnames[1:init.__code__.co_argcount]
    index = names.index("device") - (len(names) - len(defaults))
    init.__defaults__ = defaults[:index] + (str(device),) + defaults[index + 1:]
    try:
        yield
    finally:
        init.__defaults__ = defaults


def obs_space(shapes):
    """A gym-like observation space (WorldModel reads ``obs_space.spaces[k].shape``)."""
    return types.SimpleNamespace(spaces={k: types.SimpleNamespace(shape=tuple(v)) for k, v in shapes.items()})


class WorldModel(models.WorldModel):
    """The official world model without the heads not listed in ``heads``; the official optimizer
    (``tools.Optimizer``) is rebuilt over the remaining parameters when a head is removed."""

    def __init__(self, shapes, config, heads=HEADS):
        with mlp_default_device(config.device), contextlib.redirect_stdout(io.StringIO()):
            super().__init__(obs_space(shapes), None, 0, config)
        removed = [name for name in list(self.heads) if name not in heads]
        for name in removed:
            del self.heads[name]
        if removed:
            self._model_opt = tools.Optimizer("model", self.parameters(), config.model_lr, config.opt_eps,
                                              config.grad_clip, config.weight_decay, opt=config.opt,
                                              use_amp=self._use_amp)
        self.requires_grad_(False)  # as upstream (agent.requires_grad_(False)); _train enables grads itself

    def preprocess(self, obs):
        """Upstream ``WorldModel.preprocess`` except that ``image / 255`` applies only when an image is present (inputs
        are copied, as upstream's ``torch.tensor``: the official ``obs_step`` zeroes reset actions in place)."""
        obs = {k: (v.detach().to(device=self._config.device, dtype=torch.float32, copy=True) if torch.is_tensor(v)
                   else torch.tensor(v, device=self._config.device, dtype=torch.float32)) for k, v in obs.items()}
        if "image" in obs:
            obs["image"] = obs["image"] / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            obs["discount"] = obs["discount"].unsqueeze(-1)
        assert "is_first" in obs
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs

    def train_step(self, data):
        """One official update (``WorldModel._train``); returns its metrics as floats."""
        _, _, metrics = self._train(data)
        return {k: float(np.asarray(v)) for k, v in metrics.items()}

    def optimizer_state_dict(self):
        return self._model_opt._opt.state_dict()

    def load_optimizer_state_dict(self, state):
        self._model_opt._opt.load_state_dict(state)


def build_model(configuration="native", overrides=None, device="cpu"):
    """The T1 world model (carriers, 5-d launch) at a pre-declared configuration."""
    config = build_config({**CONFIGURATIONS[configuration], **(overrides or {})}, device=device)
    return WorldModel(SHAPES, config).to(device)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def updates_for(dataset_steps, config):
    """DreamerV3's replay ratio over an offline dataset: ``train_ratio`` replayed steps per dataset step, ``batch_size
    x batch_length`` steps per update (dreamer.py: one update every batch_steps / train_ratio agent steps)."""
    return int(round(dataset_steps * config.train_ratio / (config.batch_size * config.batch_length)))


# ---------------------------------------------------------------------------------------------------------------
# resumable training (the official update; SIGTERM -> resume.pt at the next update boundary -> exit 75)
# ---------------------------------------------------------------------------------------------------------------

class TrainingPaused(Exception):
    """A stop was requested; resume.pt holds the state after the last completed update."""


def atomic_save(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def rng_state(device):
    """The global torch generators the official posterior samples draw from (CPU, and the CUDA device's)."""
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    return {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device) if cuda else None}


def set_rng_state(state, device):
    torch.set_rng_state(state["cpu"])
    if state.get("cuda") is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"], device)


def train_loop(model, out_dir, steps, batch_fn, should_stop, device, max_steps=None, log_every=100,
               checkpoint_every=2000):
    """Official updates ``model.train_step(batch_fn(step))`` for steps [start, steps), resuming from
    ``out_dir/resume.pt`` (model, Adam state, torch generators, step, elapsed seconds). The caller seeds torch and
    builds the model first. A nonfinite loss or grad norm ends the run as a typed failure (upstream has no check:
    the update has been applied). ``should_stop()`` True -> resume.pt -> ``TrainingPaused``; ``max_steps`` bounds
    this call (tests, throughput). Returns {"start", "stop", "steps_done", "seconds", "elapsed", "failure", "trace"}."""
    import json
    import time
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume = out_dir / "resume.pt"
    start, elapsed = 0, 0.0
    if resume.is_file():
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        model.load_optimizer_state_dict(state["optimizer"])
        set_rng_state(state["rng"], device)
        start, elapsed = state["step"], state["elapsed"]
    stop_at = steps if max_steps is None else min(steps, start + max_steps)
    began, failure, trace, step = time.monotonic(), None, [], start - 1
    with open(out_dir / "train-log.jsonl", "a") as log_file:
        for step in range(start, stop_at):
            try:
                metrics = model.train_step(batch_fn(step))
            except ValueError as error:  # torch.distributions argument validation of nonfinite logits / means
                if "invalid values" not in str(error):
                    raise
                failure = {"typed_failure": "nonfinite_loss", "step": step, "message": str(error).splitlines()[0]}
                break
            if not (np.isfinite(metrics["model_loss"]) and np.isfinite(metrics["model_grad_norm"])):
                failure = {"typed_failure": "nonfinite_loss", "step": step,
                           "model_loss": str(metrics["model_loss"]), "model_grad_norm": str(metrics["model_grad_norm"])}
                break
            if step % log_every == 0 or step == steps - 1:
                entry = {"step": step, "seconds": elapsed + time.monotonic() - began, **metrics}
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()
                trace.append(entry)
            stop = bool(should_stop()) and step + 1 < steps
            if stop or ((step + 1) % checkpoint_every == 0 and step + 1 < steps):
                atomic_save(resume, {"model": model.state_dict(), "optimizer": model.optimizer_state_dict(),
                                     "rng": rng_state(device), "step": step + 1,
                                     "elapsed": elapsed + time.monotonic() - began})
            if stop:
                raise TrainingPaused(f"paused at update {step + 1} of {steps}")
    seconds = time.monotonic() - began
    return {"start": start, "stop": step + 1, "steps_done": step + 1 - start, "seconds": seconds,
            "elapsed": elapsed + seconds, "failure": failure, "trace": trace}



# ---------------------------------------------------------------------------------------------------------------
# episodes and training rows
# ---------------------------------------------------------------------------------------------------------------

def episode_positions(end):
    """Frames 0, 5, ..., 225 of a shot whose last stored position is ``end`` (later positions repeat ``end``)."""
    return [min(p, int(end)) for p in POSITIONS]


def episode_actions(launch5):
    """[B, 46, 5]: zeros with the launch at index 1 (applied at t0, leading into the frame-5 observation)."""
    launch5 = torch.as_tensor(launch5, dtype=torch.float32)
    actions = torch.zeros(launch5.shape[0], len(POSITIONS), ACT_DIM)
    actions[:, LAUNCH_INDEX] = launch5
    return actions


def sample_segments(rng, lengths, length, batch):
    """Rows of one batch as upstream ``tools.sample_episodes`` builds them, with numpy generator ``rng``: per row a
    list of (episode, start, stop) segments filling ``length`` steps. Episodes are drawn with probability
    proportional to their length (uniform for the equal-length NovPhy episodes); an episode shorter than 2 steps is
    redrawn; the first segment starts at a random index in [0, total - 2], later episodes are appended whole from
    index 0 (cut at ``length``)."""
    lengths = np.asarray(lengths, dtype=np.int64)
    p = lengths / lengths.sum()
    rows = []
    for _ in range(batch):
        segments, size = [], 0
        while size < length:
            episode = int(rng.choice(len(lengths), p=p))
            total = int(lengths[episode])
            if total < 2:
                continue
            start = int(rng.integers(0, total - 1)) if not segments else 0
            stop = min(start + length - size, total)
            segments.append((episode, start, stop))
            size += stop - start
        rows.append(segments)
    return rows


def training_batch(carriers, launch5, rows):
    """Official training inputs [B, L, ...] of segment rows over episodes ``carriers`` [N, 46, 236] / ``launch5``
    [N, 5]: carrier, action, is_first (True at every segment start), is_terminal (False)."""
    launch5 = torch.as_tensor(launch5, dtype=torch.float32)
    parts = {"carrier": [], "action": [], "is_first": []}
    for segments in rows:
        c, a, f = [], [], []
        for episode, start, stop in segments:
            c.append(carriers[episode, start:stop])
            action = torch.zeros(stop - start, ACT_DIM)  # episode_actions(launch5)[episode, start:stop]
            if start <= LAUNCH_INDEX < stop:
                action[LAUNCH_INDEX - start] = launch5[episode]
            a.append(action)
            first = torch.zeros(stop - start)
            first[0] = 1.0
            f.append(first)
        parts["carrier"].append(torch.cat(c))
        parts["action"].append(torch.cat(a))
        parts["is_first"].append(torch.cat(f))
    batch = {k: torch.stack(v) for k, v in parts.items()}
    batch["is_terminal"] = torch.zeros_like(batch["is_first"])
    return batch


# ---------------------------------------------------------------------------------------------------------------
# deterministic readouts
# ---------------------------------------------------------------------------------------------------------------

def _decode(model, feat):
    return model.heads["decoder"](feat)["carrier"].mode()


@torch.no_grad()
def rollout(model, anchor_carriers, launch5, steps=ROLLOUT_STEPS, keep=None):
    """Deterministic rollout from anchor carriers [N, 236] under launches [N, 5]: posterior mode at t0, then ``steps``
    prior modes (launch at the first ``img_step``, no-ops after). Returns ``endpoint`` [N, 236] (decoded carrier at
    frame 5 * steps), ``feat_end`` [N, 1536] (the RSSM feature there) and, for ``keep`` (rollout indices 0..steps),
    ``carriers`` {index: [N, 236]}."""
    model.eval()
    rssm = model.dynamics
    device = next(model.parameters()).device
    anchor_carriers = torch.as_tensor(anchor_carriers, dtype=torch.float32, device=device)
    launch5 = torch.as_tensor(launch5, dtype=torch.float32, device=device)
    n = anchor_carriers.shape[0]
    obs = model.preprocess({"carrier": anchor_carriers[:, None], "is_first": torch.ones(n, 1),
                            "is_terminal": torch.zeros(n, 1)})
    embed = model.encoder(obs)[:, 0]
    state, _ = rssm.obs_step(None, None, embed, obs["is_first"][:, 0], sample=False)
    keep = set(keep or ())
    kept = {0: _decode(model, rssm.get_feat(state))} if 0 in keep else {}
    for step in range(steps):
        action = launch5 if step == 0 else torch.zeros_like(launch5)
        state = rssm.img_step(state, action, sample=False)
        if step + 1 in keep:
            kept[step + 1] = _decode(model, rssm.get_feat(state))
    feat = rssm.get_feat(state)
    return {"endpoint": _decode(model, feat), "feat_end": feat, "carriers": kept, "final_state": state}


@torch.no_grad()
def posterior_features(model, carriers, launch5, index):
    """RSSM features (``get_feat``: one-hot stochastic mode 32 x 32 + deter 512) of the deterministic posterior
    (``obs_step`` with ``sample=False``, the official observe recursion) of episodes [B, 46, 236] at ``index``:
    [B, len(index), 1536]."""
    model.eval()
    rssm = model.dynamics
    device = next(model.parameters()).device
    b, t = carriers.shape[:2]
    is_first = torch.zeros(b, t)
    is_first[:, 0] = 1.0
    obs = model.preprocess({"carrier": carriers, "is_first": is_first, "is_terminal": torch.zeros(b, t)})
    embed = model.encoder(obs)
    actions = episode_actions(launch5).to(device)
    state, feats, wanted = None, [], set(index)
    for step in range(t):
        state, _ = rssm.obs_step(state, actions[:, step], embed[:, step], obs["is_first"][:, step], sample=False)
        if step in wanted:
            feats.append(rssm.get_feat(state))
    return torch.stack(feats, 1)
