"""#133 T1 DreamerV3 G-port: the official NM512/dreamerv3-torch world model, losses and optimizer vs the adapter.

Both sides run in the ``novphy`` env (torch 2.13, CPU, float32), in separate processes (the #129 lesson: one torch
build on both sides, so Adam round-off cannot differ):

``--reference OUT_DIR``
    Imports the official clone (``DREAMERV3_REF``, default /mnt/array/sukaih/novphy-baselines-src/dreamerv3-torch@6ef8646)
    as upstream does (``sys.path`` = the clone; top-level ``models`` / ``networks`` / ``tools``), reads the clone's
    ``configs.yaml`` with YAML 1.2 floats and dreamer.py's block merge, and builds ``models.WorldModel`` per arm.
    Runtime shims only, no file edited: an empty ``torch.utils.tensorboard`` module (tools.py imports SummaryWriter at
    module level for its Logger; tensorboard is not installed and the Logger is never built) and, while a model is
    constructed, networks.MLP's default ``device='cuda'`` set to ``cpu`` (it only builds a fixed-std tensor the
    encoder / symlog-MSE decoder never read; CUDA-less hosts cannot construct it). Data the official code needs but
    no encoder / decoder key selects (``image`` for the vector arms, ``reward`` for the zero-scaled reward head) are
    fed as upstream environments deliver them. Per arm: the seeded initial weights, the fixed batch, then two calls
    of the official ``WorldModel._train`` (torch seeded before each: posterior samples), recording every loss term,
    the embeddings / RSSM states / features, the pre-step gradients (an Adam step pre-hook), the grad norm and the
    parameters and Adam slots after each step; then the deterministic rollout (``obs_step`` / ``img_step`` with
    ``sample=False``, decoder mode) and posterior features the adapter's readouts use.

``--compare OUT_DIR``
    Builds each arm with ``world_model.baselines.dreamerv3.novphy`` (the adapter's config builder, its WorldModel
    subclass without the reward / continue heads for the NovPhy and PushT arms, its batch builder for the NovPhy
    arms), loads the official initial weights (the adapter's keys must be the official keys minus the removed heads),
    repeats the two steps and the readouts, and checks them under #123's tolerances and #123's parameter rule
    (``run_hjepa_comparison.GPORT_TOLERANCE``: after each step the share of elements with |delta| <= 1e-7 is >= 0.999
    and no element moves more than 2 lr + 1e-7 away). Writes ``OUT_DIR/gport.json`` (schema ``issue_127_gport_v1``).

``--run OUT_DIR``: both, as subprocesses.

DreamerV3 has no learning-rate schedule (constant Adam LR): the step-0 LR is the positive LR, so both steps run at the
configured LR (1e-4 native; 3e-4 for the lr3e-4 configuration).

Arms: ``official-vector`` (the official vector configuration with its reward and continue heads at loss scale 1, the
adapter keeping those heads), ``t1-native`` / ``t1-lr3e-4`` (the official vector configuration with
``reward_head.loss_scale = cont_head.loss_scale = 0`` vs the adapter without those heads, on the adapter's NovPhy
batch: carriers, the launch at index 1, upstream sample_episodes rows), ``pusht-64`` (the official dmc_vision
configuration with the 64 x 64 image CNN and the 4-d proprio MLP, 10-d actions, zero-scaled heads, vs the PushT
fidelity model).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get("DREAMERV3_REF", "/mnt/array/sukaih/novphy-baselines-src/dreamerv3-torch@6ef8646"))
COMMIT = "6ef8646d807cd10ce0c88e10a7e943211e7fc44c"
UPSTREAM = "https://github.com/NM512/dreamerv3-torch"
SCHEMA = "issue_127_gport_v1"
VENDOR = ROOT / "world_model/baselines/dreamerv3/vendor"
ADAPTER = ROOT / "world_model/baselines/dreamerv3/novphy.py"
SEED_INIT, SEED_INPUT, SEED_STEP1, SEED_STEP2 = 20261025, 7, 11, 13
NOVPHY_LAUNCHES = [[-0.25, 0.10, 1.0, 0.0, 1.0], [-0.08, 0.01, 1.0, 0.0, 1.0], [-0.17, 0.06, 1.0, 0.0, 1.0],
                   [-0.33, 0.17, 1.0, 0.0, 1.0], [-0.12, 0.11, 1.0, 0.0, 1.0]]
# fixed upstream-style rows (episode, start, stop) of length 64 over 5 episodes of 46 steps: a first segment from a
# start inside an episode, then whole episodes from index 0
T1_ROWS = [[(0, 10, 46), (3, 0, 28)], [(1, 0, 46), (2, 0, 18)], [(4, 44, 46), (0, 0, 46), (1, 0, 16)],
           [(2, 5, 46), (4, 0, 23)]]
VECTOR_SHAPES = {"carrier": (236,), "reward": (), "is_first": (), "is_last": (), "is_terminal": ()}
PUSHT_SHAPES = {"image": (64, 64, 3), "proprio": (4,), "reward": (), "is_first": (), "is_last": (), "is_terminal": ()}
PUSHT_OVERRIDES = {"encoder.mlp_keys": "proprio", "decoder.mlp_keys": "proprio"}
ARMS = {
    "official-vector": {"blocks": ["defaults", "dmc_proprio"], "overrides": {}, "num_actions": 5,
                        "shapes": VECTOR_SHAPES, "adapter_heads": ["decoder", "reward", "cont"], "batch": "vector"},
    "t1-native": {"blocks": ["defaults", "dmc_proprio"], "configuration": "native", "num_actions": 5,
                  "overrides": {"reward_head.loss_scale": 0.0, "cont_head.loss_scale": 0.0},
                  "shapes": VECTOR_SHAPES, "adapter_heads": ["decoder"], "batch": "t1"},
    "t1-lr3e-4": {"blocks": ["defaults", "dmc_proprio"], "configuration": "lr3e-4", "num_actions": 5,
                  "overrides": {"model_lr": 3e-4, "reward_head.loss_scale": 0.0, "cont_head.loss_scale": 0.0},
                  "shapes": VECTOR_SHAPES, "adapter_heads": ["decoder"], "batch": "t1"},
    "pusht-64": {"blocks": ["defaults", "dmc_vision"], "num_actions": 10,
                 "overrides": {**PUSHT_OVERRIDES, "reward_head.loss_scale": 0.0, "cont_head.loss_scale": 0.0},
                 "shapes": PUSHT_SHAPES, "adapter_heads": ["decoder"], "batch": "pusht"},
}
ROLLOUT_ARMS = ("t1-native",)
TOLERANCE = {"loss_relative": 1e-4, "grad_norm_relative": 1e-4, "grad_tensor_relative_l2": 1e-3,
             "grad_floor_of_global_norm": 1e-6, "forward_relative_l2": 1e-3, "forward_max_abs": 1e-4,
             "param_close_abs": 1e-7, "param_close_share": 0.999, "param_max_abs_over_lr": 2.0,
             "readout_max_abs": 1e-5}


def sha256_of(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# fixed inputs (numpy; identical on both sides)
# ---------------------------------------------------------------------------------------------------------------

def t1_episodes():
    """5 episodes of 46 carriers and their launches."""
    rng = np.random.Generator(np.random.PCG64(SEED_INPUT))
    carriers = (rng.standard_normal((5, 46, 236)) * 2.0).astype(np.float32)
    return carriers, np.asarray(NOVPHY_LAUNCHES, dtype=np.float32)


def vector_batch():
    """official-vector: B 3 x L 12 (random carriers / actions / rewards, a reset inside the second row)."""
    rng = np.random.Generator(np.random.PCG64(SEED_INPUT + 1))
    b, t = 3, 12
    is_first = np.zeros((b, t), np.float32)
    is_first[:, 0] = 1.0
    is_first[1, 6] = 1.0
    terminal = np.zeros((b, t), np.float32)
    terminal[2, -1] = 1.0
    return {"carrier": (rng.standard_normal((b, t, 236)) * 2.0).astype(np.float32),
            "action": rng.uniform(-1, 1, (b, t, 5)).astype(np.float32),
            "reward": rng.standard_normal((b, t)).astype(np.float32), "is_first": is_first, "is_terminal": terminal}


def pusht_batch():
    """pusht-64: B 2 x L 6 (uint8-valued 64 x 64 images, normalised proprio and 10-d actions)."""
    rng = np.random.Generator(np.random.PCG64(SEED_INPUT + 2))
    b, t = 2, 6
    is_first = np.zeros((b, t), np.float32)
    is_first[:, 0] = 1.0
    return {"image": rng.integers(0, 256, (b, t, 64, 64, 3)).astype(np.float32),
            "proprio": rng.standard_normal((b, t, 4)).astype(np.float32),
            "action": rng.standard_normal((b, t, 10)).astype(np.float32), "is_first": is_first,
            "is_terminal": np.zeros((b, t), np.float32)}


def reference_t1_batch():
    """The NovPhy batch as upstream builds training rows (tools.sample_episodes with the segments of T1_ROWS):
    episodes hold carrier, action (launch at index 1 = applied at t0; zeros elsewhere), reward 0, is_first at index
    0, is_terminal 0; segments are appended with is_first set at every segment start."""
    carriers, launches = t1_episodes()
    episodes = []
    for i in range(len(carriers)):
        action = np.zeros((46, 5), np.float32)
        action[1] = launches[i]
        first = np.zeros(46, np.float32)
        first[0] = 1.0
        episodes.append({"carrier": carriers[i], "action": action, "reward": np.zeros(46, np.float32),
                         "is_first": first, "is_terminal": np.zeros(46, np.float32)})
    rows = []
    for segments in T1_ROWS:
        ret, size = None, 0
        for episode, start, stop in segments:
            part = {k: v[start:stop].copy() for k, v in episodes[episode].items()}
            if ret is None:
                ret = part
                ret["is_first"][0] = True
            else:
                ret = {k: np.append(ret[k], part[k], axis=0) for k in ret}
                ret["is_first"][size] = True
            size = len(ret["carrier"])
        assert size == 64
        rows.append(ret)
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def official_data(arm):
    spec = ARMS[arm]
    if spec["batch"] == "vector":
        data = vector_batch()
    elif spec["batch"] == "t1":
        data = reference_t1_batch()
    else:
        data = pusht_batch()
        data["reward"] = np.zeros(data["is_first"].shape, np.float32)
    if "image" not in data:  # upstream environments always deliver an image; no key of these arms selects it
        data["image"] = np.zeros(data["is_first"].shape + (1, 1, 3), np.float32)
    return data


# ---------------------------------------------------------------------------------------------------------------
# shared step recorder
# ---------------------------------------------------------------------------------------------------------------

def _record_steps(wm, data, names):
    """Two official updates with the step seeds; returns per-step metrics, forward tensors, pre-step gradients,
    post-step parameters and Adam slots (CPU tensors)."""
    opt = wm._model_opt._opt
    captured = {}

    def hook(optimizer, args, kwargs):
        captured["grads"] = {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
                             for n, p in names.items()}

    handle = opt.register_step_pre_hook(hook)
    steps = []
    for seed in (SEED_STEP1, SEED_STEP2):
        torch.manual_seed(seed)
        post, context, metrics = wm._train({k: v.copy() if isinstance(v, np.ndarray) else v for k, v in data.items()})
        state = {n: p.detach().clone() for n, p in names.items()}
        slots = {}
        for n, p in names.items():
            s = opt.state.get(p, {})
            if s:
                slots[n] = {"exp_avg": s["exp_avg"].clone(), "exp_avg_sq": s["exp_avg_sq"].clone()}
        steps.append({"metrics": {k: float(np.asarray(v)) for k, v in metrics.items()},
                      "forward": {"embed": context["embed"].detach().clone(), "feat": context["feat"].detach().clone(),
                                  "kl": context["kl"].detach().clone(),
                                  **{f"post_{k}": v.detach().clone() for k, v in post.items()}},
                      "grads": captured.pop("grads"), "params": state, "slots": slots,
                      "lr": [g["lr"] for g in opt.param_groups]})
    handle.remove()
    return steps


def _official_rollout(wm, anchors, launches, steps=45):
    rssm = wm.dynamics
    n = anchors.shape[0]
    obs = wm.preprocess({"carrier": anchors[:, None].numpy(), "image": np.zeros((n, 1, 1, 1, 3), np.float32),
                         "is_first": np.ones((n, 1), np.float32), "is_terminal": np.zeros((n, 1), np.float32)})
    embed = wm.encoder(obs)[:, 0]
    state, _ = rssm.obs_step(None, None, embed, obs["is_first"][:, 0], sample=False)
    decode = lambda s: wm.heads["decoder"](rssm.get_feat(s))["carrier"].mode()  # noqa: E731
    frames = [decode(state)]
    for k in range(steps):
        state = rssm.img_step(state, launches if k == 0 else torch.zeros_like(launches), sample=False)
        frames.append(decode(state))
    return torch.stack(frames, 1)


def _official_posterior(wm, carriers, launches, index):
    rssm = wm.dynamics
    b, t = carriers.shape[:2]
    actions = torch.zeros(b, t, 5)
    actions[:, 1] = launches
    first = np.zeros((b, t), np.float32)
    first[:, 0] = 1
    obs = wm.preprocess({"carrier": carriers.numpy(), "image": np.zeros((b, t, 1, 1, 3), np.float32),
                         "is_first": first, "is_terminal": np.zeros((b, t), np.float32)})
    embed = wm.encoder(obs)
    state, feats = None, []
    for step in range(t):
        state, _ = rssm.obs_step(state, actions[:, step], embed[:, step], obs["is_first"][:, step], sample=False)
        if step in index:
            feats.append(rssm.get_feat(state))
    return torch.stack(feats, 1)


# ---------------------------------------------------------------------------------------------------------------
# --reference (official clone)
# ---------------------------------------------------------------------------------------------------------------

def _official_config(configs, spec):
    merged = {}

    def recursive_update(base, update):  # dreamer.py
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    for name in spec["blocks"]:
        recursive_update(merged, configs[name])
    for key, value in spec["overrides"].items():
        target, parts = merged, key.split(".")
        for part in parts[:-1]:
            target = target[part]
        assert parts[-1] in target, key
        target[parts[-1]] = value
    config = types.SimpleNamespace(**merged)
    config.num_actions = spec["num_actions"]
    config.device = "cpu"
    return config


def reference(out_dir):
    import yaml

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stub = types.ModuleType("torch.utils.tensorboard")
    stub.SummaryWriter = None  # shim: tools.Logger (its only user) is never built
    sys.modules["torch.utils.tensorboard"] = stub
    sys.path.insert(0, str(OFFICIAL))
    import models
    import networks

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_implicit_resolver("tag:yaml.org,2002:float",
                                 re.compile(r"^[-+]?(?:[0-9][0-9_]*)(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"),
                                 list("-+0123456789"))
    configs = yaml.load((OFFICIAL / "configs.yaml").read_text(), Loader=Loader)
    init = networks.MLP.__init__
    names_ = init.__code__.co_varnames[1:init.__code__.co_argcount]
    index = names_.index("device") - (len(names_) - len(init.__defaults__))
    meta = {}
    for arm, spec in ARMS.items():
        config = _official_config(configs, spec)
        space = types.SimpleNamespace(spaces={k: types.SimpleNamespace(shape=tuple(v)) for k, v in spec["shapes"].items()})
        defaults = init.__defaults__
        init.__defaults__ = defaults[:index] + ("cpu",) + defaults[index + 1:]
        try:
            torch.manual_seed(SEED_INIT)
            with contextlib.redirect_stdout(io.StringIO()):
                wm = models.WorldModel(space, None, 0, config)
        finally:
            init.__defaults__ = defaults
        wm.requires_grad_(False)
        names = dict(wm.named_parameters())
        initial = {k: v.detach().clone() for k, v in wm.state_dict().items()}
        data = official_data(arm)
        steps = _record_steps(wm, data, names)
        record = {"arm": arm, "initial": initial, "data": {k: torch.as_tensor(v) for k, v in data.items()},
                  "steps": steps, "config": {k: v for k, v in vars(config).items()},
                  "parameters": int(sum(p.numel() for p in wm.parameters())),
                  "optimizer": {"class": type(wm._model_opt._opt).__name__,
                                "defaults": {k: v for k, v in wm._model_opt._opt.defaults.items()
                                             if isinstance(v, (int, float, bool, str))}}}
        if arm in ROLLOUT_ARMS:
            carriers, launches = (torch.as_tensor(x) for x in t1_episodes())
            with torch.no_grad():
                record["rollout"] = _official_rollout(wm, carriers[:, 0], launches)
                record["posterior"] = _official_posterior(wm, carriers, launches, list(range(0, 46, 3)))
        torch.save(record, out_dir / f"reference-{arm}.pt")
        meta[arm] = {"parameters": record["parameters"], "metrics": [s["metrics"] for s in steps]}
        print(f"[reference] {arm}: {record['parameters']} parameters; losses "
              f"{[round(s['metrics']['model_loss'], 4) for s in steps]}", flush=True)
    (out_dir / "reference.json").write_text(json.dumps({
        "torch": torch.__version__, "python": sys.version.split()[0], "official": str(OFFICIAL), "commit": COMMIT,
        "official_sha256": {f: sha256_of(OFFICIAL / f) for f in ("models.py", "networks.py", "tools.py",
                                                                  "configs.yaml")},
        "arms": meta, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2) + "\n")


# ---------------------------------------------------------------------------------------------------------------
# --compare (adapter)
# ---------------------------------------------------------------------------------------------------------------

def _rel_l2(mine, theirs):
    denom = float(theirs.double().norm())
    diff = float((mine.double() - theirs.double()).norm())
    return diff / denom if denom > 0 else diff


def _max_abs(mine, theirs):
    return float((mine.double() - theirs.double()).abs().max()) if mine.numel() else 0.0


def adapter_data(arm, ref):
    """The adapter's input batch: the adapter's own builder for the NovPhy arms; the fixed batch otherwise (the
    keys no adapter encoder / decoder selects are dropped)."""
    from world_model.baselines.dreamerv3 import novphy as dn
    spec = ARMS[arm]
    if spec["batch"] == "t1":
        carriers, launches = t1_episodes()
        return {k: v.numpy() for k, v in dn.training_batch(torch.as_tensor(carriers), torch.as_tensor(launches),
                                                            T1_ROWS).items()}
    data = {k: v.numpy() for k, v in ref["data"].items()}
    drop = {"image"} if "image" not in spec["shapes"] else set()
    if "reward" not in spec["adapter_heads"]:
        drop.add("reward")
    return {k: v for k, v in data.items() if k not in drop}


def compare_arm(arm, ref):
    """The adapter side of one arm, built by the code the runs use: ``dn.build_model(configuration)`` (the runner's
    builder) for the t1 arms, ``b127_dreamerv3_fidelity.build_pusht_model`` for pusht-64, ``dn.WorldModel`` with every
    head for official-vector."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from world_model.baselines.dreamerv3 import novphy as dn
    from scripts import b127_dreamerv3_fidelity as fid
    spec = ARMS[arm]
    tol = TOLERANCE
    torch.manual_seed(SEED_INIT)
    if spec["batch"] == "t1":
        wm = dn.build_model(spec["configuration"], device="cpu")
    elif spec["batch"] == "pusht":
        wm = fid.build_pusht_model("cpu")
    else:
        wm = dn.WorldModel(spec["shapes"], dn.build_config(blocks=tuple(spec["blocks"]),
                                                           num_actions=spec["num_actions"]),
                           heads=tuple(spec["adapter_heads"]))
    config = wm._config
    names = dict(wm.named_parameters())
    official_keys, mine_keys = set(ref["initial"]), set(wm.state_dict())
    removed = sorted(official_keys - mine_keys)
    keys_ok = mine_keys <= official_keys and all(k.split(".")[1] in ("reward", "cont") and k.startswith("heads.")
                                                 for k in removed) and (bool(removed) == (spec["adapter_heads"] ==
                                                                                          ["decoder"]))
    wm.load_state_dict({k: ref["initial"][k] for k in mine_keys})
    data = adapter_data(arm, ref)
    batch_equal = all(torch.equal(torch.as_tensor(data[k]), ref["data"][k].to(torch.as_tensor(data[k]).dtype))
                      for k in data)
    ref_config = {k: v for k, v in ref["config"].items()}
    mine_config = vars(config)
    config_diff = sorted(k for k in set(ref_config) | set(mine_config)
                         if ref_config.get(k) != mine_config.get(k) and not k.endswith("head"))
    head_scales = {k: ref_config[k]["loss_scale"] for k in ("reward_head", "cont_head")}
    steps = _record_steps(wm, data, names)
    out = {"arm": arm, "adapter_heads": spec["adapter_heads"], "removed_official_tensors": removed,
           "state_keys_ok": keys_ok, "batch_equal": batch_equal, "config_differences": config_diff,
           "official_head_loss_scales": head_scales, "official_parameters": ref["parameters"],
           "adapter_parameters": int(sum(p.numel() for p in wm.parameters())),
           "optimizer_class": type(wm._model_opt._opt).__name__,
           "optimizer_defaults_equal": {k: v for k, v in wm._model_opt._opt.defaults.items()
                                        if isinstance(v, (int, float, bool, str))} == ref["optimizer"]["defaults"],
           "steps": []}
    ok = keys_ok and batch_equal and not config_diff and out["optimizer_defaults_equal"]
    for k, (mine, theirs) in enumerate(zip(steps, ref["steps"])):
        loss_keys = [m for m in theirs["metrics"] if m.endswith("_loss") or m in ("kl", "prior_ent", "post_ent")]
        shared_losses = [m for m in loss_keys if m in mine["metrics"]]
        missing = sorted(set(loss_keys) - set(shared_losses))
        # the zero-scaled official heads' own losses are not part of the adapter (their contribution to model_loss
        # is 0 * loss); every shared term must match
        loss_rel = max(abs(mine["metrics"][m] - theirs["metrics"][m]) / max(1.0, abs(theirs["metrics"][m]))
                       for m in shared_losses)
        norm_rel = abs(mine["metrics"]["model_grad_norm"] - theirs["metrics"]["model_grad_norm"]) / max(
            1.0, abs(theirs["metrics"]["model_grad_norm"]))
        fwd_rel = max(_rel_l2(mine["forward"][f], theirs["forward"][f]) for f in theirs["forward"])
        fwd_abs = max(_max_abs(mine["forward"][f], theirs["forward"][f]) for f in theirs["forward"])
        floor = tol["grad_floor_of_global_norm"] * theirs["metrics"]["model_grad_norm"]
        grad_rel = max(_rel_l2(mine["grads"][n], theirs["grads"][n]) for n in mine["grads"]
                       if float(theirs["grads"][n].norm()) > floor)
        null_abs = max([float((mine["grads"][n] - theirs["grads"][n]).norm()) for n in mine["grads"]
                        if float(theirs["grads"][n].norm()) <= floor] or [0.0])
        removed_grad_max = max([float(theirs["grads"][n].abs().max()) for n in theirs["grads"] if n not in names]
                               or [0.0])
        deltas = torch.cat([(mine["params"][n] - theirs["params"][n]).abs().flatten() for n in mine["params"]])
        close_share = float((deltas <= tol["param_close_abs"]).float().mean())
        max_delta = float(deltas.max())
        lr = max(mine["lr"])
        slot_delta = max([max(float((mine["slots"][n][s] - theirs["slots"][n][s]).abs().max())
                              for s in ("exp_avg", "exp_avg_sq")) for n in mine["slots"]] or [0.0])
        removed_moved = max([float((theirs["params"][n] - ref["initial"][n]).abs().max()) for n in theirs["params"]
                             if n not in names] or [0.0])
        moved_share = float(torch.cat([(mine["params"][n] - (ref["initial"][n] if k == 0 else
                                                             ref["steps"][k - 1]["params"][n])).abs().flatten()
                                       for n in mine["params"]]).gt(0).float().mean())
        step = {"step": k + 1, "lr": lr, "loss_terms": shared_losses, "official_only_terms": missing,
                "loss_max_rel": loss_rel, "model_loss": [mine["metrics"]["model_loss"], theirs["metrics"]["model_loss"]],
                "grad_norm": [mine["metrics"]["model_grad_norm"], theirs["metrics"]["model_grad_norm"]],
                "grad_norm_rel": norm_rel, "forward_max_rel_l2": fwd_rel, "forward_max_abs": fwd_abs,
                "grad_max_rel_l2": grad_rel, "null_gradient_max_abs_l2": null_abs,
                "removed_heads_official_grad_max_abs": removed_grad_max,
                "removed_heads_official_param_moved_max_abs": removed_moved,
                "param_close_share": close_share, "param_max_abs_delta": max_delta, "adam_slot_max_abs_delta": slot_delta,
                "adapter_moved_share": moved_share,
                "lr_equal": [round(x, 12) for x in mine["lr"]] == [round(x, 12) for x in theirs["lr"]]}
        step["pass"] = bool(loss_rel <= tol["loss_relative"] and norm_rel <= tol["grad_norm_relative"]
                            and fwd_rel <= tol["forward_relative_l2"] and fwd_abs <= tol["forward_max_abs"]
                            and grad_rel <= tol["grad_tensor_relative_l2"]
                            and close_share >= tol["param_close_share"]
                            and max_delta <= tol["param_max_abs_over_lr"] * lr + tol["param_close_abs"]
                            and step["lr_equal"] and moved_share > 0.5
                            and (removed_grad_max == 0.0 or spec["adapter_heads"] != ["decoder"]))
        ok = ok and step["pass"]
        out["steps"].append(step)
    if "rollout" in ref:
        carriers, launches = (torch.as_tensor(x) for x in t1_episodes())
        wm.load_state_dict({k: v for k, v in ref["steps"][-1]["params"].items() if k in mine_keys}, strict=False)
        mine_roll = dn.rollout(wm, carriers[:, 0], launches, keep=range(46))
        mine_frames = torch.stack([mine_roll["carriers"][i] for i in range(46)], 1)
        out["rollout_max_abs"] = _max_abs(mine_frames, ref["rollout"])
        out["rollout_endpoint_max_abs"] = _max_abs(mine_roll["endpoint"], ref["rollout"][:, -1])
        out["posterior_max_abs"] = _max_abs(dn.posterior_features(wm, carriers, launches, list(range(0, 46, 3))),
                                            ref["posterior"])
        ok = ok and max(out["rollout_max_abs"], out["rollout_endpoint_max_abs"],
                        out["posterior_max_abs"]) <= tol["readout_max_abs"]
    out["pass"] = bool(ok)
    return out


def compare(out_dir):
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "reference.json").read_text())
    arms = {}
    for arm in ARMS:
        ref = torch.load(out_dir / f"reference-{arm}.pt", map_location="cpu", weights_only=False)
        arms[arm] = compare_arm(arm, ref)
        s = arms[arm]["steps"]
        print(f"[compare] {arm}: pass {arms[arm]['pass']}; param share {[x['param_close_share'] for x in s]}; "
              f"max delta {[x['param_max_abs_delta'] for x in s]}", flush=True)
    vendor = json.loads((VENDOR / "VENDOR.json").read_text())
    hashes = {name: sha256_of(VENDOR / name) == entry["vendored_sha256"]
              and sha256_of(OFFICIAL / entry["upstream_path"]) == entry["upstream_sha256"]
              for name, entry in vendor["files"].items()}
    result = {"schema": SCHEMA, "issue": 133, "arm": "T1 DreamerV3-car", "checked_at": time.strftime(
              "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": "cpu float32", "torch": torch.__version__,
              "reference_torch": meta["torch"], "same_torch": meta["torch"] == torch.__version__,
              "upstream": UPSTREAM, "commit": COMMIT, "tolerance": TOLERANCE,
              "G_port_rule": "issue_123_exact (run_hjepa_comparison.GPORT_TOLERANCE param rule) + #123 loss / grad "
                             "tolerances; readouts max abs <= 1e-5",
              "vendor_hashes_match": hashes, "reference_script_sha256": sha256_of(Path(__file__)),
              "adapter_sha256": sha256_of(ADAPTER), "official_sha256": meta["official_sha256"],
              "arms": arms,
              "notes": [
                  "Both sides on the same torch build (novphy, CPU) in separate processes.",
                  "DreamerV3 has a constant LR (no schedule): both steps run at the configured positive LR.",
                  "t1-* / pusht-64: the official model keeps its reward / continue heads at loss scale 0 (the official "
                  "way to train without reward); the adapter removes them. Their gradients are exactly zero, so the "
                  "global grad norm, the clip and Adam's updates of every shared tensor are unaffected.",
                  "Reference shims (no file edited): empty torch.utils.tensorboard module; networks.MLP default "
                  "device 'cpu' while constructing.",
              ]}
    result["pass"] = bool(all(a["pass"] for a in arms.values()) and all(hashes.values()) and result["same_torch"])
    (out_dir / "gport.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[compare] G-port pass {result['pass']}", flush=True)
    return 0 if result["pass"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--reference", metavar="OUT_DIR")
    mode.add_argument("--compare", metavar="OUT_DIR")
    mode.add_argument("--run", metavar="OUT_DIR")
    args = parser.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    if args.reference:
        reference(args.reference)
        return 0
    if args.compare:
        return compare(args.compare)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for flag in ("--reference", "--compare"):
        code = subprocess.run([sys.executable, str(Path(__file__).resolve()), flag, args.run], cwd=ROOT, env=env).returncode
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
