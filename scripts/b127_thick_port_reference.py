"""#131 D1 THICK: the official TF2 code as the G-port reference, and the official CPU smoke.

Two sides, one file (no TF / torch import at module level):

``--reference OUT_DIR`` (env ``b127-thick-tf``: Python 3.8, tensorflow-gpu 2.4.2 on CPU, TFP 0.11, numpy 1.19.5)
    Builds the official ``agent.WorldModel`` (vendored ``world_model/baselines/thick/vendor/thick`` @ 934980a, run
    as upstream does: cwd and ``sys.path`` = that directory) for every G-port arm, eagerly (upstream ``jit: False``),
    float32, CPU. Instrumentation only (no vendored file is edited): ``tf.random.normal`` (the GateL0RD gate noise)
    and ``tfd.OneHotCategorical.sample`` (every one-hot latent / high-level-action sample) draw from a seeded numpy
    stream that is recorded in call order (one-hot samples become Gumbel-max draws from recorded uniforms; the
    sampled distribution is unchanged), ``tf.GradientTape.gradient`` records the gradients the official
    ``common.Optimizer`` computes, and ``THICKNet.thick_loss`` records its inputs / targets / masks. Per arm it
    writes ``reference-<arm>.npz``: the initial weights (under the port's parameter names), the fixed inputs, module
    forwards (encoder, GRU cell, GateL0RD cell with and without noise, C-RSSM observe / imagine, decoders, heads,
    the THICK body), then two calls of ``Agent.train``'s world-model part (``wm.train`` then ``wm.train_thick``)
    at the configured constant LR 1e-4 (THICK has no schedule: its step-0 LR is the positive LR): every loss term,
    the pre-clip gradients, the grad norms, the parameters and Adam slots after each step; plus every recorded draw.

``--compare OUT_DIR`` (env ``novphy``, torch)
    Rebuilds every arm with ``world_model.baselines.thick.port``, loads the official initial weights, replays the
    recorded draws and checks forward tensors, loss terms, gradients and both optimizer steps under the #123
    tolerances; writes ``OUT_DIR/gport.json`` (schema ``issue_127_gport_v1``).

``--official-smoke -- <train.py args>`` (env ``b127-thick-tf``)
    Runs the vendored official ``train.py`` (or another entry point given as the first argument) on CPU. Upstream
    asserts a visible GPU ("To actually train on CPU remove this assert"); the smoke answers that query with a stub
    at run time instead of editing the file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "world_model/baselines/thick/vendor"
OFFICIAL = VENDOR / "thick"
SEED_INPUT, SEED_WARMUP, SEED_NOISE = 20261024, 7, 11
NOVPHY_LAUNCH = [-0.25, 0.10, 1.0, 0.0, 1.0]

# G-port arms: the NovPhy adapter's configuration (MLP carrier pathway, continuous 5-d launch) and the official
# VisualPinPad configuration (CNN image pathway, one-hot actions) used by the G-fid route.
ARMS = {
    "novphy-carrier": {
        "configs": ["thick_pinpad"],
        "overrides": {"encoder.mlp_keys": "carrier", "encoder.cnn_keys": "$^",
                      "decoder.mlp_keys": "carrier", "decoder.cnn_keys": "$^"},
        "obs": {"carrier": [236]}, "action": {"kind": "box", "dim": 5}, "batch": 3, "length": 10},
    "pinpad-image": {
        "configs": ["visualpinpad", "thick_pinpad"],
        "overrides": {"task": "pinpad_four"},
        "obs": {"image": [64, 64, 3]}, "action": {"kind": "onehot", "dim": 5}, "batch": 2, "length": 8},
}
TOLERANCE = {"loss_relative": 1e-4, "grad_norm_relative": 1e-4, "grad_tensor_relative_l2": 1e-3,
             "grad_floor_of_global_norm": 1e-6, "forward_relative_l2": 1e-3, "forward_max_abs": 1e-4,
             "param_close_abs": 1e-7, "param_close_share": 0.999, "param_max_abs_over_lr": 2.0,
             "update_relative_l2": 1e-3}


def sha256_of(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_inputs(spec):
    """The fixed batch (numpy): observations, actions, rewards, flags; identical on both sides."""
    rng = np.random.Generator(np.random.PCG64(SEED_INPUT))
    b, t = spec["batch"], spec["length"]
    data = {}
    for key, shape in spec["obs"].items():
        if len(shape) == 3:
            data[key] = rng.integers(0, 256, (b, t, *shape), dtype=np.uint8)
        else:
            data[key] = rng.standard_normal((b, t, *shape)).astype(np.float32)
    dim = spec["action"]["dim"]
    if spec["action"]["kind"] == "onehot":
        action = np.eye(dim, dtype=np.float32)[rng.integers(0, dim, (b, t))]
        action[:, 0] = 0.0
        reward = np.where(rng.random((b, t)) < 0.15, 10.0, 0.0).astype(np.float32)
    else:
        action = np.zeros((b, t, dim), np.float32)
        action[:, 1] = np.asarray(NOVPHY_LAUNCH, np.float32) + rng.uniform(-0.05, 0.05, (b, dim)).astype(np.float32)
        action[:, 1, 2:] = np.asarray(NOVPHY_LAUNCH[2:], np.float32)
        reward = np.zeros((b, t), np.float32)
    is_first = np.zeros((b, t), bool)
    is_first[:, 0] = True
    data.update({"action": action, "reward": reward, "is_first": is_first,
                 "is_last": np.zeros((b, t), bool), "is_terminal": np.zeros((b, t), bool)})
    probes = {"gru_x": rng.standard_normal((b, 256 + 16)).astype(np.float32),
              "gru_h": rng.standard_normal((b, 256)).astype(np.float32),
              "gate_x": rng.standard_normal((b, 256)).astype(np.float32),
              "gate_c": rng.uniform(-1, 1, (b, 16)).astype(np.float32),
              "feat": rng.standard_normal((b, 4, 32 * 32 + 256 + 16)).astype(np.float32),
              "coarse": rng.standard_normal((b, 4, 32 * 32 + 16)).astype(np.float32),
              "thick_x": rng.standard_normal((b, 32 * 32 + 16 + 5)).astype(np.float32)}
    return data, probes


# ---------------------------------------------------------------------------
# reference side (TF 2.4)
# ---------------------------------------------------------------------------


def _export(tf, common, tfkl, wm):
    """{port parameter name: (tf.Variable, transform)} for every official variable of the world model."""
    out = {}

    def dense(prefix, layer):
        out[f"{prefix}.weight"] = (layer.kernel, "T")
        if layer.bias is not None:
            out[f"{prefix}.bias"] = (layer.bias, None)

    def conv(prefix, layer):
        out[f"{prefix}.weight"] = (layer.kernel, "conv")
        out[f"{prefix}.bias"] = (layer.bias, None)

    def layer_norm(prefix, layer):
        out[f"{prefix}.weight"] = (layer.gamma, None)
        out[f"{prefix}.bias"] = (layer.beta, None)

    def module(prefix, mod):
        for key, obj in getattr(mod, "_modules", {}).items():
            path = f"{prefix}.m.{key}"
            if isinstance(obj, tfkl.Conv2D):  # includes Conv2DTranspose
                conv(path, obj)
            elif isinstance(obj, tfkl.Dense):
                dense(path, obj)
            elif isinstance(obj, common.NormLayer):
                if obj._layer is not None:
                    layer_norm(f"{path}.layer", obj._layer)
            elif isinstance(obj, common.Module):
                module(path, obj)
            else:
                raise TypeError(f"{path}: {type(obj)}")

    def world_model(p, wm):
        module(f"{p}encoder", wm.encoder)
        module(f"{p}rssm", wm.rssm)
        dense(f"{p}rssm.cell.layer", wm.rssm._cell._layer)
        if wm.rssm._cell._norm:
            layer_norm(f"{p}rssm.cell.norm", wm.rssm._cell._norm)
        if hasattr(wm.rssm, "_ctxt_rnn"):
            dense(f"{p}rssm.ctxt_rnn.r_layer", wm.rssm._ctxt_rnn._r_layer)
            dense(f"{p}rssm.ctxt_rnn.g_layer", wm.rssm._ctxt_rnn._g_layer)
            module(f"{p}rssm.ctxt_head", wm.rssm._ctxt_head)
        for name, head in wm.heads.items():
            module(f"{p}heads.{name}", head)
        for name, head in wm.ctxt_heads.items():
            module(f"{p}ctxt_heads.{name}", head)
        if wm.hierarchical:
            module(f"{p}thick_net", wm.thick_net)
            for name, head in wm.thick_net.thick_heads.items():
                module(f"{p}thick_net.thick_heads.{name}", head)

    def actor_critic(p, ac):
        module(f"{p}actor", ac.actor)
        module(f"{p}critic", ac.critic)
        if ac.config.slow_target:
            module(f"{p}target_critic", ac._target_critic)
        if ac.config.thick_dreamer:
            module(f"{p}coarse_critic", ac.coarse_critic)
            if ac.config.slow_target:
                module(f"{p}coarse_target_critic", ac.coarse_target_critic)

    if hasattr(wm, "wm"):  # an agent.Agent
        world_model("wm.", wm.wm)
        actor_critic("task_behavior.", wm._task_behavior)
        expl = wm._expl_behavior
        if expl is not wm._task_behavior:
            actor_critic("expl_behavior.ac.", expl.ac)
            for i, head in enumerate(expl._networks):
                module(f"expl_behavior.networks.{i}", head)
    else:
        world_model("", wm)
    return out


def _value(var, transform):
    value = var.numpy()
    if transform == "T":
        return value.T
    if transform == "conv":  # Keras [kh, kw, in, out] / transpose [kh, kw, out, in] -> torch [out|in, in|out, kh, kw]
        return value.transpose(3, 2, 0, 1)
    return value


def reference(out_dir):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    out_dir = Path(out_dir).resolve()
    os.chdir(OFFICIAL)
    sys.path.insert(0, str(OFFICIAL))
    import gym
    import tensorflow as tf
    from tensorflow.keras import layers as tfkl
    from tensorflow_probability import distributions as tfd
    import ruamel.yaml as yaml
    import common
    import agent
    from common import thick_utils

    tf.config.run_functions_eagerly(True)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = yaml.safe_load((OFFICIAL / "configs.yaml").read_text())

    state = {"rng": None, "phase": None, "draws": None, "grads": None, "thick": None}
    orig_normal = tf.random.normal

    def rec_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None):
        if state["rng"] is None:
            return orig_normal(shape, mean, stddev, dtype=dtype)
        shape = tuple(int(s) for s in np.asarray(shape).reshape(-1))
        z = state["rng"].standard_normal(shape).astype(np.float32)
        state["draws"].append(("normal", z))
        return tf.constant(z, dtype=dtype) * stddev + mean

    def rec_categorical(self, sample_shape=(), seed=None, name="sample", **kwargs):
        # tfd.Independent._sample_n calls the inner distribution with sample_shape = n (= 1 here)
        lead = () if np.ndim(sample_shape) == 1 and len(sample_shape) == 0 else (int(sample_shape),)
        assert lead in ((), (1,)), sample_shape
        logits = tf.convert_to_tensor(self.logits)
        u = state["rng"].random(tuple(logits.shape)).astype(np.float32)
        state["draws"].append(("uniform", u))
        g = -tf.math.log(-tf.math.log(tf.clip_by_value(tf.constant(u), 1e-20, 1.0 - 1e-7)))
        index = tf.argmax(tf.stop_gradient(logits) + g, -1)
        sample = tf.one_hot(index, logits.shape[-1], dtype=self.dtype)
        return tf.reshape(sample, lead + tuple(sample.shape))

    class RecTape(tf.GradientTape):
        def gradient(self, target, sources, *args, **kwargs):
            grads = super().gradient(target, sources, *args, **kwargs)
            if state["grads"] is not None:
                state["grads"].append((list(sources), grads))
            return grads

    orig_thick_loss = thick_utils.THICKNet.thick_loss

    def rec_thick_loss(self, inps, targets, loss_masks):
        if state["thick"] is not None:
            state["thick"].append({"inps": inps.numpy(), "masks": {k: v.numpy() for k, v in loss_masks.items()},
                                   "targets": {k: v.numpy() for k, v in targets.items()}})
        return orig_thick_loss(self, inps, targets, loss_masks)

    tf.random.normal = rec_normal
    tfd.OneHotCategorical.sample = rec_categorical
    tf.GradientTape = RecTape
    thick_utils.THICKNet.thick_loss = rec_thick_loss

    for arm, spec in ARMS.items():
        t0 = time.time()
        config = common.Config(configs["defaults"])
        for name in spec["configs"]:
            config = config.update(configs[name])
        config = config.update(spec["overrides"])
        obs_space = {k: gym.spaces.Box(0, 1, tuple(v), np.float32) for k, v in spec["obs"].items()}
        obs_space.update({k: gym.spaces.Box(0, 1, (), np.float32) for k in ("reward", "is_first", "is_last",
                                                                              "is_terminal")})
        dim = spec["action"]["dim"]
        act_space = gym.spaces.Box(-1, 1, (dim,), np.float32)
        if spec["action"]["kind"] == "onehot":
            act_space.n = dim
        tf.random.set_seed(SEED_WARMUP)
        wm = agent.WorldModel(config, obs_space, act_space, tf.Variable(0, dtype=tf.int64), actor=None)
        data_np, probes = fixed_inputs(spec)
        data = {k: tf.constant(v) for k, v in data_np.items()}
        # warm-up: build every variable (draws from a throwaway stream, not recorded), weights unchanged
        state.update(rng=np.random.Generator(np.random.PCG64(SEED_WARMUP)), draws=[])
        wm.loss(data)
        wm.thick_loss(data, c_rssm=True)
        exported = _export(tf, common, tfkl, wm)
        model_vars = tf.nest.flatten([m.variables for m in [wm.encoder, wm.rssm, *wm.heads.values(),
                                                             *wm.ctxt_heads.values()]])
        thick_vars = tf.nest.flatten([wm.thick_net.variables])
        names = {v.ref(): n for n, (v, _) in exported.items()}
        missing = [v.name for v in model_vars + thick_vars if v.ref() not in names]
        assert not missing, missing
        record = {f"init/{n}": _value(v, tr) for n, (v, tr) in exported.items()}
        record.update({f"input/{k}": v for k, v in data_np.items()})
        record.update({f"probe_in/{k}": v for k, v in probes.items()})
        meta = {"arm": arm, "spec": spec, "config": dict(config.flat), "tf": tf.__version__,
                "model_variables": len(model_vars), "thick_variables": len(thick_vars),
                "model_parameters": int(sum(np.prod(v.shape) for v in model_vars)),
                "thick_parameters": int(sum(np.prod(v.shape) for v in thick_vars)),
                "transforms": {n: tr for n, (_, tr) in exported.items()}}

        def draws_to_record(phase, draws):
            meta.setdefault("draws", {})[phase] = [kind for kind, _ in draws]
            for i, (kind, value) in enumerate(draws):
                record[f"noise/{phase}/{i:05d}"] = value

        # phase "modules": module forwards with the initial weights
        state.update(rng=np.random.Generator(np.random.PCG64(SEED_NOISE)), draws=[])
        pre = wm.preprocess(data)
        embed = wm.encoder(pre)
        record["modules/embed"] = embed.numpy()
        gru_out, _, _ = wm.rssm._cell(tf.constant(probes["gru_x"]), [tf.constant(probes["gru_h"])], sample=True)
        record["modules/gru"] = gru_out.numpy()
        for sample in (False, True):
            gx = tf.constant(probes["gate_x"])
            out, nxt, gates = wm.rssm._ctxt_rnn(gx, [tf.constant(probes["gate_c"])], sample=sample)
            record[f"modules/gatel0rd_{int(sample)}/state"] = nxt[0].numpy()
            record[f"modules/gatel0rd_{int(sample)}/gates"] = gates.numpy()
        post, prior = wm.rssm.observe(embed, pre["action"], pre["is_first"])
        for k, v in post.items():
            record[f"modules/post/{k}"] = v.numpy()
        for k, v in prior.items():
            record[f"modules/prior/{k}"] = v.numpy()
        last = {k: v[:, -1] for k, v in post.items()}
        img_action = tf.concat([pre["action"][:, 1:2], tf.zeros_like(pre["action"][:, 1:4])], 1)
        imag = wm.rssm.imagine(img_action, last)
        for k, v in imag.items():
            record[f"modules/imagine/{k}"] = v.numpy()
        feat, coarse = tf.constant(probes["feat"]), tf.constant(probes["coarse"])
        for key, dist in wm.heads["decoder"](feat).items():
            record[f"modules/decoder/{key}"] = dist.mode().numpy()
        for key, dist in wm.ctxt_heads["decoder"](coarse).items():
            record[f"modules/ctxt_decoder/{key}"] = dist.mode().numpy()
        record["modules/reward"] = wm.heads["reward"](feat).mode().numpy()
        record["modules/ctxt_reward"] = wm.ctxt_heads["reward"](coarse).mode().numpy()
        tx = tf.constant(probes["thick_x"])
        record["modules/thick_body"] = wm.thick_net.forward_thick_body(tx).numpy()
        record["modules/thick_logit"] = wm.thick_net._state_pred_layers(
            wm.thick_net.forward_thick_body(tx))["thick_logit"].numpy()
        record["modules/thick_prior_logit"] = wm.thick_net._hl_act_prior_layer(tx[:, :-5])["hl_prior_logit"].numpy()
        draws_to_record("modules", state["draws"])

        # phases "step1", "step2": Agent.train's world-model part (wm.train, then wm.train_thick)
        for step in (1, 2):
            state.update(rng=np.random.Generator(np.random.PCG64(SEED_NOISE + step)), draws=[], grads=[],
                         thick=[])
            _, outputs, metrics = wm.train(data)
            thick_metrics = wm.train_thick(data)
            prefix = f"step{step}"
            for k in ("embed", "feat", "kl", "ctxt_kl"):
                record[f"{prefix}/out/{k}"] = outputs[k].numpy()
            for k, v in outputs["post"].items():
                record[f"{prefix}/post/{k}"] = v.numpy()
            for k, v in outputs["prior"].items():
                record[f"{prefix}/prior/{k}"] = v.numpy()
            meta[prefix] = {"metrics": {k: float(v) for k, v in metrics.items() if np.ndim(v) == 0},
                            "thick_metrics": {k: float(v) for k, v in thick_metrics.items() if np.ndim(v) == 0}}
            assert len(state["grads"]) == 2, len(state["grads"])
            for group, (sources, grads) in zip(("model", "thick"), state["grads"]):
                meta[prefix][f"{group}_none_grads"] = [names[s.ref()] for s, g in zip(sources, grads) if g is None]
                for s, g in zip(sources, grads):
                    if g is not None:
                        record[f"{prefix}/grad/{names[s.ref()]}"] = _value(tf.convert_to_tensor(g),
                                                                            exported[names[s.ref()]][1])
            captured = state["thick"][0]
            record[f"{prefix}/thick/inps"] = captured["inps"]
            for k, v in captured["targets"].items():
                record[f"{prefix}/thick/targets/{k}"] = v
            for k, v in captured["masks"].items():
                record[f"{prefix}/thick/masks/{k}"] = v
            for n, (v, tr) in exported.items():
                record[f"{prefix}/param/{n}"] = _value(v, tr)
            for opt_name, opt in (("model", wm.model_opt), ("thick", wm.thick_opt)):
                keras = opt._opt
                meta[prefix][f"{opt_name}_iterations"] = int(keras.iterations.numpy())
                meta[prefix][f"{opt_name}_lr"] = float(keras._decayed_lr(tf.float32).numpy())
                for n, (v, tr) in exported.items():
                    try:
                        m_slot, v_slot = keras.get_slot(v, "m"), keras.get_slot(v, "v")
                    except (KeyError, ValueError):
                        continue
                    record[f"{prefix}/slot_m/{n}"] = _value(m_slot, tr)
                    record[f"{prefix}/slot_v/{n}"] = _value(v_slot, tr)
            draws_to_record(prefix, state["draws"])
        state.update(rng=None, draws=None, grads=None, thick=None)
        path = out_dir / f"reference-{arm}.npz"
        np.savez(path, **record)
        meta["seconds"] = time.time() - t0
        (out_dir / f"reference-{arm}.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
        print(f"[b127-thick-reference] {arm}: model {meta['model_parameters']} / thick {meta['thick_parameters']} "
              f"parameters; step-1 model_loss {meta['step1']['metrics']['model_loss']:.6f}, thick loss "
              f"{meta['step1']['thick_metrics']['thick_net_loss']:.6f}; {meta['seconds']:.1f} s", flush=True)
    return 0


# ---------------------------------------------------------------------------
# comparison side (torch port)
# ---------------------------------------------------------------------------


def _rel_l2(mine, theirs):
    import torch
    mine, theirs = torch.as_tensor(mine).double(), torch.as_tensor(theirs).double()
    return float((mine - theirs).norm() / max(float(theirs.norm()), 1e-30))


def _max_abs(mine, theirs):
    import torch
    return float((torch.as_tensor(mine).double() - torch.as_tensor(theirs).double()).abs().max())


def compare_arm(ref_npz, ref_meta):
    import torch
    sys.path.insert(0, str(ROOT))
    from world_model.baselines.thick import port

    tol = TOLERANCE
    torch.set_num_threads(8)
    rec = dict(np.load(ref_npz))
    meta = json.loads(Path(ref_meta).read_text())
    spec = meta["spec"]
    config = port.load_config(spec["configs"], spec["overrides"])
    config_equal = {k: v for k, v in config.flat.items() if meta["config"].get(k) != v
                    and not (isinstance(v, (list, tuple)) and list(v) == meta["config"].get(k))}
    obs_space = {**{k: tuple(v) for k, v in spec["obs"].items()},
                 "reward": (), "is_first": (), "is_last": (), "is_terminal": ()}
    cont = spec["action"]["kind"] == "box"
    wm = port.WorldModel(config, obs_space, spec["action"]["dim"], cont)
    params = dict(wm.named_parameters())
    init = {k[5:]: v for k, v in rec.items() if k.startswith("init/")}
    names_equal = set(params) == set(init)
    with torch.no_grad():
        for name, p in params.items():
            if name in init:
                p.copy_(torch.as_tensor(init[name]))
    data = {k[6:]: torch.as_tensor(v) for k, v in rec.items() if k.startswith("input/")}
    probes = {k[9:]: torch.as_tensor(v) for k, v in rec.items() if k.startswith("probe_in/")}

    def noise(phase):
        kinds = meta["draws"][phase]
        normals = [rec[f"noise/{phase}/{i:05d}"] for i, k in enumerate(kinds) if k == "normal"]
        uniforms = [rec[f"noise/{phase}/{i:05d}"] for i, k in enumerate(kinds) if k == "uniform"]
        return port.ReplayNoise(normals, uniforms)

    forward = {}

    def check(name, mine, key):
        theirs = rec[key]
        mine = mine.detach().numpy() if hasattr(mine, "detach") else np.asarray(mine)
        if mine.shape != theirs.shape:
            forward[name] = {"shape": [list(mine.shape), list(theirs.shape)], "pass": False}
            return
        rel, mx = _rel_l2(mine, theirs), _max_abs(mine, theirs)
        forward[name] = {"relative_l2": rel, "max_abs": mx,
                         "pass": bool(rel <= tol["forward_relative_l2"] or mx <= tol["forward_max_abs"])}

    # module forwards
    nz = noise("modules")
    with torch.no_grad():
        pre = wm.preprocess(data)
        embed = wm.encoder(pre)
        check("encoder", embed, "modules/embed")
        gru, _ = wm.rssm.cell(probes["gru_x"], probes["gru_h"])
        check("gru_cell", gru, "modules/gru")
        for sample in (False, True):
            nxt, gates = wm.rssm.ctxt_rnn(probes["gate_x"], probes["gate_c"], sample, nz)
            check(f"gatel0rd_sample{int(sample)}_state", nxt, f"modules/gatel0rd_{int(sample)}/state")
            check(f"gatel0rd_sample{int(sample)}_gates", gates, f"modules/gatel0rd_{int(sample)}/gates")
        post, prior = wm.rssm.observe(embed, pre["action"], pre["is_first"], noise=nz)
        for k in post:
            check(f"observe_post_{k}", post[k], f"modules/post/{k}")
        for k in prior:
            check(f"observe_prior_{k}", prior[k], f"modules/prior/{k}")
        last = {k: v[:, -1] for k, v in post.items()}
        img_action = torch.cat([pre["action"][:, 1:2], torch.zeros_like(pre["action"][:, 1:4])], 1)
        imag = wm.rssm.imagine(img_action, last, noise=nz)
        for k in imag:
            check(f"imagine_{k}", imag[k], f"modules/imagine/{k}")
        for key, dist in wm.heads["decoder"](probes["feat"]).items():
            check(f"decoder_{key}", dist.mode(), f"modules/decoder/{key}")
        for key, dist in wm.ctxt_heads["decoder"](probes["coarse"]).items():
            check(f"ctxt_decoder_{key}", dist.mode(), f"modules/ctxt_decoder/{key}")
        check("reward_head", wm.heads["reward"](probes["feat"]).mode(), "modules/reward")
        check("ctxt_reward_head", wm.ctxt_heads["reward"](probes["coarse"]).mode(), "modules/ctxt_reward")
        body = wm.thick_net.body(probes["thick_x"])
        check("thick_body", body, "modules/thick_body")
        check("thick_logit", wm.thick_net.state_logits(body), "modules/thick_logit")
        check("thick_hl_prior_logit", wm.thick_net.hl_prior_logits(probes["thick_x"][:, :-5]),
              "modules/thick_prior_logit")
    modules_noise_used = nz.used == {"normal": len(nz.normals), "uniform": len(nz.uniforms)}

    # two training steps
    model_opt, thick_opt = wm.make_optimizers()
    steps = {}
    previous = {n: torch.as_tensor(init[n]) for n in params}
    previous_ref = dict(previous)
    for step in (1, 2):
        prefix = f"step{step}"
        nz = noise(prefix)
        ref_metrics, ref_thick = meta[prefix]["metrics"], meta[prefix]["thick_metrics"]
        model_loss, _, outs, losses = wm.loss(data, noise=nz)
        fwd = {}
        for k in ("embed", "feat", "kl", "ctxt_kl"):
            mine = outs[k].detach().numpy()
            fwd[k] = {"relative_l2": _rel_l2(mine, rec[f"{prefix}/out/{k}"]),
                      "max_abs": _max_abs(mine, rec[f"{prefix}/out/{k}"])}
        for side, d in (("post", outs["post"]), ("prior", outs["prior"])):
            for k, v in d.items():
                mine = v.detach().numpy()
                fwd[f"{side}_{k}"] = {"relative_l2": _rel_l2(mine, rec[f"{prefix}/{side}/{k}"]),
                                      "max_abs": _max_abs(mine, rec[f"{prefix}/{side}/{k}"])}
        fwd_pass = all(v["relative_l2"] <= tol["forward_relative_l2"] or v["max_abs"] <= tol["forward_max_abs"]
                       for v in fwd.values())
        loss_terms = {k: (float(v), ref_metrics[f"{k}_loss"]) for k, v in losses.items()}
        loss_terms["model"] = (float(model_loss), ref_metrics["model_loss"])
        loss_rel = max(abs(a - b) / max(1.0, abs(b)) for a, b in loss_terms.values())
        grads = model_opt.gradients(model_loss)
        gnames = [n for n, _ in model_opt.named]
        model_norm = float(model_opt.apply(grads))
        # high level on the updated model
        thick_total, thick_losses, thick_aux = wm.thick_loss(data, noise=nz)
        thick_terms = {k: (float(v), ref_thick[f"{k}_loss"]) for k, v in thick_losses.items()}
        thick_terms["thick_total"] = (float(thick_total), ref_thick["thick_net_loss"])
        thick_rel = max(abs(a - b) / max(1.0, abs(b)) for a, b in thick_terms.values())
        thick_data = {"inps": _rel_l2(thick_aux["inps"].detach(), rec[f"{prefix}/thick/inps"])}
        for k, v in thick_aux["targets"].items():
            thick_data[f"target_{k}"] = _rel_l2(v.detach(), rec[f"{prefix}/thick/targets/{k}"])
        for k, v in thick_aux["masks"].items():
            thick_data[f"mask_{k}"] = _max_abs(v.detach(), rec[f"{prefix}/thick/masks/{k}"])
        thick_grads = thick_opt.gradients(thick_total)
        tnames = [n for n, _ in thick_opt.named]
        thick_norm = float(thick_opt.apply(thick_grads))
        noise_used = nz.used == {"normal": len(nz.normals), "uniform": len(nz.uniforms)}

        def grad_rule(names_, grads_, group, ref_norm):
            floor = tol["grad_floor_of_global_norm"] * ref_norm
            none_mine = sorted(n for n, g in zip(names_, grads_) if g is None)
            none_ref = sorted(meta[prefix][f"{group}_none_grads"])
            rel, null_abs, nulls = [], [], 0
            for n, g in zip(names_, grads_):
                if g is None:
                    continue
                theirs = torch.as_tensor(rec[f"{prefix}/grad/{n}"])
                if float(theirs.norm()) > floor:
                    rel.append(_rel_l2(g, theirs))
                else:
                    nulls += 1
                    null_abs.append(float((g.double() - theirs.double()).norm()))
            return {"max_relative_l2": max(rel or [0.0]), "null_tensors": nulls,
                    "null_max_abs_l2": max(null_abs or [0.0]), "floor": floor,
                    "none_equal": none_mine == none_ref, "none": none_mine,
                    "pass": bool(max(rel or [0.0]) <= tol["grad_tensor_relative_l2"]
                                 and max(null_abs or [0.0]) <= floor and none_mine == none_ref)}

        model_grad = grad_rule(gnames, grads, "model", ref_metrics["model_grad_norm"])
        thick_grad = grad_rule(tnames, thick_grads, "thick", ref_thick["thick_net_grad_norm"])
        # parameters and Adam slots after the step
        post_params = {n: p.detach().clone() for n, p in wm.named_parameters()}
        ref_params = {n: torch.as_tensor(rec[f"{prefix}/param/{n}"]) for n in post_params}
        deltas = torch.cat([(post_params[n] - ref_params[n]).abs().flatten() for n in post_params])
        close_share = float((deltas <= tol["param_close_abs"]).float().mean())
        lr = max(meta[prefix]["model_lr"], meta[prefix]["thick_lr"])
        update = torch.cat([(post_params[n] - previous[n]).flatten() for n in post_params])
        ref_update = torch.cat([(ref_params[n] - previous_ref[n]).flatten() for n in post_params])
        slots = {}
        for opt_name, opt in (("model", model_opt), ("thick", thick_opt)):
            for slot, mine_list, transform in (("m", opt.m, lambda t: t), ("v", opt.v, torch.sqrt)):
                rels = []
                ref_all = []
                for (n, _), mine in zip(opt.named, mine_list):
                    theirs = transform(torch.as_tensor(rec[f"{prefix}/slot_{slot}/{n}"]).double())
                    ref_all.append(theirs.flatten())
                    rels.append((n, transform(mine.double()), theirs))
                gnorm = float(torch.cat(ref_all).norm())
                floor = tol["grad_floor_of_global_norm"] * gnorm
                above = [_rel_l2(a, b) for _, a, b in rels if float(b.norm()) > floor]
                below = [float((a - b).norm()) for _, a, b in rels if float(b.norm()) <= floor]
                slots[f"{opt_name}_{slot}"] = {"max_relative_l2": max(above or [0.0]),
                                               "null_max_abs_l2": max(below or [0.0]), "floor": floor,
                                               "pass": bool(max(above or [0.0]) <= tol["grad_tensor_relative_l2"]
                                                            and max(below or [0.0]) <= floor)}
            slots[f"{opt_name}_iterations"] = {"port": opt.iterations,
                                               "official": meta[prefix][f"{opt_name}_iterations"],
                                               "pass": opt.iterations == meta[prefix][f"{opt_name}_iterations"]}
        out = {
            "lr": {"model": meta[prefix]["model_lr"], "thick": meta[prefix]["thick_lr"],
                   "port": [model_opt.lr, thick_opt.lr]},
            "forward": fwd, "forward_pass": fwd_pass,
            "loss_terms": {k: {"port": a, "official": b} for k, (a, b) in loss_terms.items()},
            "loss_max_relative_delta": loss_rel,
            "model_grad_norm": model_norm, "official_model_grad_norm": ref_metrics["model_grad_norm"],
            "model_grad_norm_relative_delta": abs(model_norm - ref_metrics["model_grad_norm"])
            / max(ref_metrics["model_grad_norm"], 1e-12),
            "model_gradients": model_grad,
            "thick_data_relative_l2": thick_data,
            "thick_loss_terms": {k: {"port": a, "official": b} for k, (a, b) in thick_terms.items()},
            "thick_loss_max_relative_delta": thick_rel,
            "thick_grad_norm": thick_norm, "official_thick_grad_norm": ref_thick["thick_net_grad_norm"],
            "thick_grad_norm_relative_delta": abs(thick_norm - ref_thick["thick_net_grad_norm"])
            / max(ref_thick["thick_net_grad_norm"], 1e-12),
            "thick_gradients": thick_grad,
            "param_close_share": close_share, "param_max_abs_delta": float(deltas.max()),
            "update_relative_l2": float((update - ref_update).norm() / ref_update.norm()),
            "update_norm": float(update.norm()), "official_update_norm": float(ref_update.norm()),
            "optimizer_slots": slots, "noise_fully_replayed": noise_used,
        }
        out["pass"] = bool(
            fwd_pass and noise_used
            and loss_rel <= tol["loss_relative"] and thick_rel <= tol["loss_relative"]
            and max(thick_data.values()) <= tol["forward_relative_l2"]
            and out["model_grad_norm_relative_delta"] <= tol["grad_norm_relative"]
            and out["thick_grad_norm_relative_delta"] <= tol["grad_norm_relative"]
            and model_grad["pass"] and thick_grad["pass"]
            and close_share >= tol["param_close_share"]
            and out["param_max_abs_delta"] <= tol["param_max_abs_over_lr"] * lr + tol["param_close_abs"]
            and out["update_relative_l2"] <= tol["update_relative_l2"]
            and all(s["pass"] for s in slots.values()) and lr > 0)
        steps[prefix] = out
        previous, previous_ref = post_params, ref_params
    result = {
        "reference": {"npz": str(Path(ref_npz).relative_to(ROOT)), "npz_sha256": sha256_of(ref_npz),
                      "meta_sha256": sha256_of(ref_meta), "tf": meta["tf"]},
        "parameter_names_equal": names_equal,
        "model_parameters": {"port": sum(p.numel() for _, p in wm.model_named_parameters()),
                             "official": meta["model_parameters"]},
        "thick_parameters": {"port": sum(p.numel() for _, p in wm.thick_named_parameters()),
                             "official": meta["thick_parameters"]},
        "config_differences": config_equal,
        "modules": forward, "modules_noise_fully_replayed": modules_noise_used,
        "steps": steps,
    }
    result["pass"] = bool(names_equal and not config_equal and modules_noise_used
                          and all(v["pass"] for v in forward.values())
                          and result["model_parameters"]["port"] == result["model_parameters"]["official"]
                          and result["thick_parameters"]["port"] == result["thick_parameters"]["official"]
                          and all(s["pass"] for s in steps.values()))
    return result


def compare(out_dir):
    import torch
    out_dir = Path(out_dir).resolve()
    vendor = json.loads((VENDOR / "VENDOR.json").read_text())
    hashes = {rel: sha256_of(VENDOR / rel) == e["vendored_sha256"] for rel, e in vendor["files"].items()}
    arms = {arm: compare_arm(out_dir / f"reference-{arm}.npz", out_dir / f"reference-{arm}.json") for arm in ARMS}
    result = {"schema": "issue_127_gport_v1", "issue": 131, "arm": "D1 THICK",
              "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": "cpu float32",
              "torch": torch.__version__, "upstream": vendor["upstream"], "commit": vendor["commit"],
              "tolerance": TOLERANCE, "vendor_hashes_match": hashes,
              "reference_script_sha256": sha256_of(__file__),
              "port_sha256": sha256_of(ROOT / "world_model/baselines/thick/port.py"),
              "arms": arms,
              "notes": ["THICK uses a constant LR (model_opt / thick_opt lr 1e-4, no schedule): the step-0 LR is the "
                        "positive LR, so both recorded steps run at 1e-4; step 2 exercises the Adam moments.",
                        "Official reference eager (upstream jit: False); one-hot samples are Gumbel-max draws from "
                        "recorded uniforms and the GateL0RD noise is a recorded standard normal times "
                        "ctxt_sample_noise, replayed in call order on both sides."]}
    result["pass"] = bool(all(hashes.values()) and all(a["pass"] for a in arms.values()))
    (out_dir / "gport.json").write_text(json.dumps(result, indent=1) + "\n")
    for arm, a in arms.items():
        print(f"[b127-thick-gport] {arm}: pass {a['pass']} (step1 {a['steps']['step1']['pass']}, step2 "
              f"{a['steps']['step2']['pass']}; modules {sum(m['pass'] for m in a['modules'].values())}/"
              f"{len(a['modules'])})", flush=True)
    print(f"[b127-thick-gport] pass {result['pass']}", flush=True)
    return 0 if result["pass"] else 1

# ---------------------------------------------------------------------------
# agent level (route ii of the G-fid plan): THICK Dreamer, DreamerV2, Plan2Explore
# ---------------------------------------------------------------------------

AGENT_ARMS = {
    "thick-dreamer": {"configs": ["thick_pinpad"], "overrides": {"task": "pinpad_four"},
                      "obs": {"image": [64, 64, 3]}, "action": {"kind": "onehot", "dim": 5}, "batch": 2, "length": 6},
    "dreamerv2": {"configs": ["visualpinpad"], "overrides": {"task": "pinpad_four"},
                  "obs": {"image": [64, 64, 3]}, "action": {"kind": "onehot", "dim": 5}, "batch": 2, "length": 6},
    "plan2explore": {"configs": ["visualpinpad", "plan2explore"], "overrides": {"task": "pinpad_four"},
                     "obs": {"image": [64, 64, 3]}, "action": {"kind": "onehot", "dim": 5}, "batch": 2, "length": 6},
}


def _agent_optimizers(agnt):
    """(label, common.Optimizer) in Agent.train's update order."""
    out = [("model", agnt.wm.model_opt)]
    if agnt.wm.hierarchical:
        out.append(("thick", agnt.wm.thick_opt))
    out += [("task_actor", agnt._task_behavior.actor_opt), ("task_critic", agnt._task_behavior.critic_opt)]
    if agnt._expl_behavior is not agnt._task_behavior:
        expl = agnt._expl_behavior
        out += [("expl", expl.opt), ("expl_actor", expl.ac.actor_opt), ("expl_critic", expl.ac.critic_opt)]
    return out


def agent_reference(out_dir):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    out_dir = Path(out_dir).resolve()
    os.chdir(OFFICIAL)
    sys.path.insert(0, str(OFFICIAL))
    import gym
    import tensorflow as tf
    from tensorflow.keras import layers as tfkl
    from tensorflow_probability import distributions as tfd
    import ruamel.yaml as yaml
    import common
    import agent

    tf.config.run_functions_eagerly(True)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = yaml.safe_load((OFFICIAL / "configs.yaml").read_text())
    st = {"rng": None, "draws": None, "grads": None}
    orig_normal = tf.random.normal

    def rec_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None):
        if st["rng"] is None:
            return orig_normal(shape, mean, stddev, dtype=dtype)
        shape = tuple(int(s) for s in np.asarray(shape).reshape(-1))
        z = st["rng"].standard_normal(shape).astype(np.float32)
        st["draws"].append(("normal", z))
        return tf.constant(z, dtype=dtype) * stddev + mean

    def rec_categorical(self, sample_shape=(), seed=None, name="sample", **kwargs):
        lead = () if np.ndim(sample_shape) == 1 and len(sample_shape) == 0 else (int(sample_shape),)
        assert lead in ((), (1,)), sample_shape
        logits = tf.convert_to_tensor(self.logits)
        u = st["rng"].random(tuple(logits.shape)).astype(np.float32)
        st["draws"].append(("uniform", u))
        g = -tf.math.log(-tf.math.log(tf.clip_by_value(tf.constant(u), 1e-20, 1.0 - 1e-7)))
        index = tf.argmax(tf.stop_gradient(logits) + g, -1)
        sample = tf.one_hot(index, logits.shape[-1], dtype=self.dtype)
        return tf.reshape(sample, lead + tuple(sample.shape))

    def rec_normal_sample_n(self, n, seed=None):
        loc, scale = tf.convert_to_tensor(self.loc), tf.convert_to_tensor(self.scale)
        assert int(n) == 1, n
        shape = tuple(tf.broadcast_static_shape(loc.shape, scale.shape))
        z = st["rng"].standard_normal(shape).astype(np.float32)
        st["draws"].append(("normal", z))
        return tf.constant(z)[None] * scale + loc

    class RecTape(tf.GradientTape):
        def gradient(self, target, sources, *args, **kwargs):
            grads = super().gradient(target, sources, *args, **kwargs)
            if st["grads"] is not None:
                st["grads"].append((list(sources), grads))
            return grads

    tf.random.normal = rec_normal
    tfd.OneHotCategorical.sample = rec_categorical
    tfd.Normal._sample_n = rec_normal_sample_n
    tf.GradientTape = RecTape

    for arm, spec in AGENT_ARMS.items():
        t0 = time.time()
        config = common.Config(configs["defaults"])
        for name in spec["configs"]:
            config = config.update(configs[name])
        config = config.update(spec["overrides"])
        obs_space = {"image": gym.spaces.Box(0, 255, (64, 64, 3), np.uint8)}
        obs_space.update({k: gym.spaces.Box(0, 1, (), np.float32) for k in ("reward", "is_first", "is_last",
                                                                              "is_terminal")})
        act_space = gym.spaces.Box(0, 1, (5,), np.float32)
        act_space.n = 5
        tf.random.set_seed(SEED_WARMUP)
        agnt = agent.Agent(config, obs_space, {"action": act_space}, common.Counter(0))
        data_np, _ = fixed_inputs(spec)
        data = {k: tf.constant(v) for k, v in data_np.items()}
        # warm-up: one full Agent.train builds every variable and optimizer slot (draws not recorded)
        st.update(rng=np.random.Generator(np.random.PCG64(SEED_WARMUP)), draws=[], grads=None)
        agnt.train(data)
        exported = _export(tf, common, tfkl, agnt)
        names = {v.ref(): n for n, (v, _) in exported.items()}
        trainable = [v for v in agnt.trainable_variables if v.dtype == tf.float32]  # tfstep / _updates are int64
        missing = [v.name for v in trainable if v.ref() not in names]
        assert not missing, missing
        optimizers = _agent_optimizers(agnt)
        record = {f"init/{n}": _value(v, tr) for n, (v, tr) in exported.items()}
        record.update({f"input/{k}": v for k, v in data_np.items()})
        meta = {"arm": arm, "spec": spec, "config": dict(config.flat), "tf": tf.__version__,
                "parameters": int(sum(np.prod(v.shape) for v in trainable)),
                "variables": len(trainable)}

        def optimizer_state(prefix):
            meta.setdefault(prefix, {})["iterations"] = {}
            for label, opt in optimizers:
                keras = opt._opt
                meta[prefix]["iterations"][label] = int(keras.iterations.numpy())
                meta[prefix].setdefault("lr", {})[label] = float(keras._decayed_lr(tf.float32).numpy())
                for n, (v, tr) in exported.items():
                    try:
                        m_slot, v_slot = keras.get_slot(v, "m"), keras.get_slot(v, "v")
                    except (KeyError, ValueError):
                        continue
                    record[f"{prefix}/slot_m/{label}/{n}"] = _value(m_slot, tr)
                    record[f"{prefix}/slot_v/{label}/{n}"] = _value(v_slot, tr)
            meta[prefix]["updates"] = {"task": int(agnt._task_behavior._updates.numpy())}
            if agnt._expl_behavior is not agnt._task_behavior:
                meta[prefix]["updates"]["expl"] = int(agnt._expl_behavior.ac._updates.numpy())

        def draws_to_record(phase):
            meta.setdefault("draws", {})[phase] = [kind for kind, _ in st["draws"]]
            for i, (kind, value) in enumerate(st["draws"]):
                record[f"noise/{phase}/{i:05d}"] = value

        optimizer_state("init")
        # acting (Agent.policy) from a reset state
        st.update(rng=np.random.Generator(np.random.PCG64(SEED_NOISE)), draws=[])
        obs = {k: v[:, 0] for k, v in data.items() if k != "action"}
        obs["is_first"] = tf.ones_like(obs["is_first"])
        modes = ["train", "eval"] + (["explore"] if agnt._expl_behavior is not agnt._task_behavior else [])
        meta["policy_modes"] = modes
        for mode in modes:
            outputs, (latent, _) = agnt.policy(obs, None, mode=mode)
            record[f"policy/{mode}/action"] = outputs["action"].numpy()
            for k in ("stoch", "deter", "logit"):
                record[f"policy/{mode}/{k}"] = latent[k].numpy()
        draws_to_record("policy")
        for step in (1, 2):
            prefix = f"step{step}"
            st.update(rng=np.random.Generator(np.random.PCG64(SEED_NOISE + step)), draws=[], grads=[])
            _, metrics = agnt.train(data)
            meta.setdefault(prefix, {})["metrics"] = {k: float(v) for k, v in metrics.items() if np.ndim(v) == 0}
            meta[prefix]["grad_calls"] = []
            for i, (sources, grads) in enumerate(st["grads"]):
                meta[prefix]["grad_calls"].append({"names": [names[s.ref()] for s in sources],
                                                   "none": [names[s.ref()] for s, g in zip(sources, grads)
                                                            if g is None]})
                for s, g in zip(sources, grads):
                    if g is not None:
                        record[f"{prefix}/grad{i}/{names[s.ref()]}"] = _value(tf.convert_to_tensor(g),
                                                                               exported[names[s.ref()]][1])
            for n, (v, tr) in exported.items():
                record[f"{prefix}/param/{n}"] = _value(v, tr)
            optimizer_state(prefix)
            draws_to_record(prefix)
        st.update(rng=None, draws=None, grads=None)
        np.savez(out_dir / f"reference-{arm}.npz", **record)
        meta["seconds"] = time.time() - t0
        (out_dir / f"reference-{arm}.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
        print(f"[b127-thick-agent-reference] {arm}: {meta['parameters']} parameters; step-1 model_loss "
              f"{meta['step1']['metrics']['model_loss']:.4f}, actor_loss {meta['step1']['metrics']['actor_loss']:.6f}, "
              f"critic_loss {meta['step1']['metrics']['critic_loss']:.6f}; {len(meta['step1']['grad_calls'])} "
              f"optimizer calls; {meta['seconds']:.1f} s", flush=True)
    return 0


def _torch_optimizers(ag_agent):
    out = [("model", ag_agent.model_opt)]
    if ag_agent.thick_opt is not None:
        out.append(("thick", ag_agent.thick_opt))
    out += [("task_actor", ag_agent.task_behavior.actor_opt), ("task_critic", ag_agent.task_behavior.critic_opt)]
    if not ag_agent.greedy:
        expl = ag_agent.expl_behavior
        out += [("expl", expl.opt), ("expl_actor", expl.ac.actor_opt), ("expl_critic", expl.ac.critic_opt)]
    return out


def agent_compare_arm(ref_npz, ref_meta):
    import torch
    sys.path.insert(0, str(ROOT))
    from world_model.baselines.thick import agent as ag
    from world_model.baselines.thick import port

    tol = TOLERANCE
    torch.set_num_threads(8)
    rec = dict(np.load(ref_npz))
    meta = json.loads(Path(ref_meta).read_text())
    spec = meta["spec"]
    config = port.load_config(spec["configs"], spec["overrides"])
    config_diff = {k: v for k, v in config.flat.items() if meta["config"].get(k) != v
                   and not (isinstance(v, (list, tuple)) and list(v) == meta["config"].get(k))}
    obs_space = {"image": (64, 64, 3), "reward": (), "is_first": (), "is_last": (), "is_terminal": ()}
    model = ag.Agent(config, obs_space, 5)
    params = dict(model.named_parameters())
    init = {k[5:]: v for k, v in rec.items() if k.startswith("init/") and not k.startswith("init/slot_")}
    names_equal = set(params) == set(init)
    name_of = {id(p): n for n, p in params.items()}
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(torch.as_tensor(init[n]))
    optimizers = _torch_optimizers(model)
    labels_equal = [label for label, _ in optimizers] == list(meta["init"]["iterations"])
    for label, opt in optimizers:
        opt.iterations = meta["init"]["iterations"][label]
        for i, (_, p) in enumerate(opt.named):
            n = name_of[id(p)]
            opt.m[i] = torch.as_tensor(rec[f"init/slot_m/{label}/{n}"]).clone()
            opt.v[i] = torch.as_tensor(rec[f"init/slot_v/{label}/{n}"]).clone()
    model.task_behavior.updates = meta["init"]["updates"]["task"]
    if not model.greedy:
        model.expl_behavior.ac.updates = meta["init"]["updates"]["expl"]
    data = {k[6:]: torch.as_tensor(v) for k, v in rec.items() if k.startswith("input/")}

    def noise(phase):
        kinds = meta["draws"][phase]
        return port.ReplayNoise([rec[f"noise/{phase}/{i:05d}"] for i, k in enumerate(kinds) if k == "normal"],
                                [rec[f"noise/{phase}/{i:05d}"] for i, k in enumerate(kinds) if k == "uniform"])

    def fully_used(nz):
        return nz.used == {"normal": len(nz.normals), "uniform": len(nz.uniforms)}

    # acting
    nz = noise("policy")
    obs = {k: v[:, 0] for k, v in data.items() if k != "action"}
    obs["is_first"] = torch.ones_like(obs["is_first"])
    policy = {}
    for mode in meta["policy_modes"]:
        action, (latent, _) = model.policy(obs, None, mode=mode, noise=nz)
        entry = {"action_equal": bool(torch.equal(action, torch.as_tensor(rec[f"policy/{mode}/action"])))}
        for k in ("stoch", "deter", "logit"):
            entry[f"{k}_relative_l2"] = _rel_l2(latent[k], rec[f"policy/{mode}/{k}"])
        entry["pass"] = bool(entry["action_equal"] and max(v for k, v in entry.items() if k.endswith("l2"))
                             <= tol["forward_relative_l2"])
        policy[mode] = entry
    policy_noise = fully_used(nz)

    # two Agent.train steps; gradients captured per optimizer call
    calls = []
    original = port.TFOptimizer.gradients

    def recording(self, loss):
        grads = original(self, loss)
        calls.append((self, grads))
        return grads

    port.TFOptimizer.gradients = recording
    steps = {}
    previous = {n: torch.as_tensor(init[n]) for n in params}
    previous_ref = dict(previous)
    try:
        for step in (1, 2):
            prefix = f"step{step}"
            calls.clear()
            nz = noise(prefix)
            metrics = {k: float(v) for k, v in model.train_step(data, nz).items()}
            ref_metrics = meta[prefix]["metrics"]
            shared = sorted(k for k in metrics if k in ref_metrics and (k.endswith("_loss") or k.endswith("_norm")
                                                                           or k == "critic_target"))
            metric_rel = {k: abs(metrics[k] - ref_metrics[k]) / max(1.0, abs(ref_metrics[k])) for k in shared}
            grad_calls = []
            for i, ((opt, grads), ref_call) in enumerate(zip(calls, meta[prefix]["grad_calls"])):
                mine_names = [name_of[id(p)] for _, p in opt.named]
                ref_norm = math.sqrt(sum(float((torch.as_tensor(rec[f"{prefix}/grad{i}/{n}"]).double() ** 2).sum())
                                         for n in ref_call["names"] if n not in ref_call["none"]))
                floor = tol["grad_floor_of_global_norm"] * ref_norm
                rel, null_abs = [], []
                for n, g in zip(mine_names, grads):
                    if g is None:
                        continue
                    theirs = torch.as_tensor(rec[f"{prefix}/grad{i}/{n}"])
                    (rel if float(theirs.norm()) > floor else null_abs).append(
                        _rel_l2(g, theirs) if float(theirs.norm()) > floor else float((g.double() - theirs).norm()))
                none_mine = sorted(n for n, g in zip(mine_names, grads) if g is None)
                entry = {"optimizer": opt.name, "tensors": len(mine_names),
                         "names_equal": sorted(mine_names) == sorted(ref_call["names"]),
                         "none_equal": none_mine == sorted(ref_call["none"]),
                         "max_relative_l2": max(rel or [0.0]), "null_max_abs_l2": max(null_abs or [0.0]),
                         "official_global_norm": ref_norm}
                entry["pass"] = bool(entry["names_equal"] and entry["none_equal"]
                                     and entry["max_relative_l2"] <= tol["grad_tensor_relative_l2"]
                                     and entry["null_max_abs_l2"] <= floor)
                grad_calls.append(entry)
            post = {n: p.detach().clone() for n, p in model.named_parameters()}
            ref_post = {n: torch.as_tensor(rec[f"{prefix}/param/{n}"]) for n in post}
            deltas = torch.cat([(post[n] - ref_post[n]).abs().flatten() for n in post])
            update = torch.cat([(post[n] - previous[n]).flatten() for n in post])
            ref_update = torch.cat([(ref_post[n] - previous_ref[n]).flatten() for n in post])
            slots = {}
            for label, opt in optimizers:
                for slot, mine_list, transform in (("m", opt.m, lambda t: t), ("v", opt.v, torch.sqrt)):
                    pairs = [(transform(mine.double()),
                              transform(torch.as_tensor(rec[f"{prefix}/slot_{slot}/{label}/{name_of[id(p)]}"])
                                        .double())) for (_, p), mine in zip(opt.named, mine_list)]
                    gnorm = float(torch.cat([b.flatten() for _, b in pairs]).norm())
                    floor = tol["grad_floor_of_global_norm"] * gnorm
                    above = [_rel_l2(a, b) for a, b in pairs if float(b.norm()) > floor]
                    below = [float((a - b).norm()) for a, b in pairs if float(b.norm()) <= floor]
                    slots[f"{label}_{slot}"] = {"max_relative_l2": max(above or [0.0]),
                                                "null_max_abs_l2": max(below or [0.0]),
                                                "pass": bool(max(above or [0.0]) <= tol["grad_tensor_relative_l2"]
                                                             and max(below or [0.0]) <= floor)}
                slots[f"{label}_iterations"] = {"port": opt.iterations,
                                                "official": meta[prefix]["iterations"][label],
                                                "pass": opt.iterations == meta[prefix]["iterations"][label]}
            lr_max = max(meta[prefix]["lr"].values())
            out = {"metrics": {k: {"port": metrics[k], "official": ref_metrics[k], "relative": metric_rel[k]}
                               for k in shared},
                   "metric_max_relative_delta": max(metric_rel.values()),
                   "optimizer_calls": grad_calls, "optimizer_calls_equal": len(calls) == len(meta[prefix]["grad_calls"]),
                   "param_close_share": float((deltas <= tol["param_close_abs"]).float().mean()),
                   "param_max_abs_delta": float(deltas.max()),
                   "update_relative_l2": float((update - ref_update).norm() / ref_update.norm()),
                   "optimizer_slots": slots, "noise_fully_replayed": fully_used(nz),
                   "lr": meta[prefix]["lr"]}
            out["pass"] = bool(out["metric_max_relative_delta"] <= tol["loss_relative"]
                               and out["optimizer_calls_equal"] and all(c["pass"] for c in grad_calls)
                               and out["param_close_share"] >= tol["param_close_share"]
                               and out["param_max_abs_delta"] <= tol["param_max_abs_over_lr"] * lr_max
                               + tol["param_close_abs"]
                               and out["update_relative_l2"] <= tol["update_relative_l2"]
                               and all(s["pass"] for s in slots.values()) and out["noise_fully_replayed"])
            steps[prefix] = out
            previous, previous_ref = post, ref_post
    finally:
        port.TFOptimizer.gradients = original
    result = {"reference": {"npz": str(Path(ref_npz).relative_to(ROOT)), "npz_sha256": sha256_of(ref_npz),
                            "meta_sha256": sha256_of(ref_meta), "tf": meta["tf"]},
              "parameter_names_equal": names_equal, "optimizer_labels_equal": labels_equal,
              "parameters": {"port": sum(p.numel() for p in params.values()), "official": meta["parameters"]},
              "config_differences": config_diff, "policy": policy, "policy_noise_fully_replayed": policy_noise,
              "steps": steps}
    result["pass"] = bool(names_equal and labels_equal and not config_diff and policy_noise
                          and result["parameters"]["port"] == result["parameters"]["official"]
                          and all(p["pass"] for p in policy.values()) and all(s["pass"] for s in steps.values()))
    return result


def agent_compare(out_dir):
    import torch
    out_dir = Path(out_dir).resolve()
    vendor = json.loads((VENDOR / "VENDOR.json").read_text())
    hashes = {rel: sha256_of(VENDOR / rel) == e["vendored_sha256"] for rel, e in vendor["files"].items()}
    arms = {arm: agent_compare_arm(out_dir / f"reference-{arm}.npz", out_dir / f"reference-{arm}.json")
            for arm in AGENT_ARMS}
    result = {"schema": "issue_127_gport_v1", "issue": 131, "arm": "D1 THICK (G-fid agents)",
              "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": "cpu float32",
              "torch": torch.__version__, "upstream": vendor["upstream"], "commit": vendor["commit"],
              "tolerance": TOLERANCE, "vendor_hashes_match": hashes,
              "vendor_edits": {rel: e["edits"] for rel, e in vendor["files"].items() if e["edits"]},
              "reference_script_sha256": sha256_of(__file__),
              "port_sha256": {name: sha256_of(ROOT / f"world_model/baselines/thick/{name}")
                              for name in ("port.py", "agent.py")},
              "modules": ["EnsembleRSSM + c_rssm-False WorldModel (dreamerv2, plan2explore)",
                          "ActorCritic: actor (one-hot, reinforce), critic, slow target critic, StreamNorm, "
                          "lambda return (all arms)",
                          "WorldModel.imagine (all arms); THICK Dreamer thick_imagine, long_term_return, coarse "
                          "critic (thick-dreamer)",
                          "Plan2Explore disagreement ensemble and intrinsic reward (plan2explore)",
                          "Agent.policy acting in train / eval / explore modes",
                          "Agent.train update order (world model, high level, task behaviour, exploration)"],
              "arms": arms,
              "notes": ["Each arm starts from the official state after one warm-up Agent.train (all variables and "
                        "Adam slots built, the slow target critic copied once); weights, slots, iteration and "
                        "target-update counters are loaded into the port, then two recorded Agent.train calls "
                        "are compared (every optimizer: model, high level, actor, critic, exploration ensemble, "
                        "exploration actor / critic).",
                        "Official reference eager; one-hot samples (latents, actions, high-level actions) are "
                        "Gumbel-max draws from recorded uniforms, normal samples (GateL0RD noise, the high level's "
                        "elapsed-time head) recorded standard normals, replayed in call order."]}
    result["pass"] = bool(all(hashes.values()) and all(a["pass"] for a in arms.values()))
    (out_dir / "gport.json").write_text(json.dumps(result, indent=1) + "\n")
    for arm, a in arms.items():
        worst = max(c["max_relative_l2"] for s in a["steps"].values() for c in s["optimizer_calls"])
        print(f"[b127-thick-agent-gport] {arm}: pass {a['pass']} (steps "
              f"{[s['pass'] for s in a['steps'].values()]}, policy {[p['pass'] for p in a['policy'].values()]}, "
              f"worst grad rel L2 {worst:.2e}, worst metric rel "
              f"{max(s['metric_max_relative_delta'] for s in a['steps'].values()):.2e})", flush=True)
    print(f"[b127-thick-agent-gport] pass {result['pass']}", flush=True)
    return 0 if result["pass"] else 1



def official_smoke(argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import runpy
    import tensorflow as tf
    real = tf.config.experimental.list_physical_devices
    tf.config.experimental.list_physical_devices = lambda kind=None: ["cpu-stub"] if kind == "GPU" else real(kind)
    tf.config.experimental.set_memory_growth = lambda *args, **kwargs: None
    entry = argv[0] if argv and argv[0].endswith(".py") else "train.py"
    args = argv[1:] if argv and argv[0].endswith(".py") else argv
    os.chdir(OFFICIAL)
    sys.path.insert(0, str(OFFICIAL))
    sys.argv = [str(OFFICIAL / entry), *args]
    runpy.run_path(sys.argv[0], run_name="__main__")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--official-smoke":
        return official_smoke([a for a in argv[1:] if a != "--"])
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reference", metavar="OUT_DIR")
    group.add_argument("--compare", metavar="OUT_DIR")
    group.add_argument("--agent-reference", metavar="OUT_DIR")
    group.add_argument("--agent-compare", metavar="OUT_DIR")
    args = parser.parse_args(argv)
    if args.agent_reference:
        return agent_reference(args.agent_reference)
    if args.agent_compare:
        return agent_compare(args.agent_compare)
    return reference(args.reference) if args.reference else compare(args.compare)


if __name__ == "__main__":
    sys.exit(main())
