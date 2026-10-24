"""#131 D1: PyTorch port of the official THICK world model (CognitiveModeling/THICK @ 934980a, TensorFlow 2).

Ported from the vendored reference (``vendor/thick``; never imported here): ``common/nets.py`` (``ContextRSSM`` with
the GateL0RD context kernel ``SimpleGateL0RDCell``, ``GRUCell``, ``Encoder``, ``Decoder``, ``MLP``, ``DistLayer``,
``NormLayer``), ``common/dists.py`` (``OneHotDist``, ``TruncNormalDist``), ``common/thick_utils.py`` (``THICKNet``, the
high-level input/target generation ``gen_thick_data``), ``common/tfutils.py`` (``Optimizer``) and ``agent.py``
(``WorldModel.loss``, ``preprocess``, ``preprocess_thick_data``, ``thick_loss``, ``train``, ``train_thick``).

Faithfulness (held to G-port, ``scripts/b127_thick_port_reference.py``): Keras initialisers (glorot-uniform kernels,
zero biases, LayerNormalization epsilon 1e-3), ELU / tanh / sigmoid activations, the straight-through one-hot samples,
the GateL0RD gate ``relu(tanh(g + noise))`` with its straight-through ``ceil`` gate count, KL balancing (``forward``,
``balance``, ``free``, ``free_avg``), the TFP truncated-normal / normal / one-hot log-probabilities, the
``Optimizer`` order (global-norm clip, decoupled weight decay ``var *= 1 - wd`` before the update) and the Keras
Adam update with the epsilon outside the bias correction (``var -= lr*sqrt(1-b2^t)/(1-b1^t) * m / (sqrt(v) + eps)``).
Float32 only (upstream ``precision: 32``). Only the paths THICK uses are ported: discrete latents
(``rssm.discrete > 0``), ``ensemble`` 1, the dists ``mse``, ``pos_clamp_normal``, ``trunc_normal``, ``onehot``.

Every random draw goes through a ``noise`` object (``normal(shape)``, ``uniform(shape)``), so G-port can replay the
reference's draws exactly; ``TorchNoise`` is the default. One-hot samples are Gumbel-max draws from uniform noise
(same distribution as TFP's categorical sampler).
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

UPSTREAM = "https://github.com/CognitiveModeling/THICK"
UPSTREAM_COMMIT = "934980a4a25d2b6baa357fe234b6e841cd3cd101"
VENDOR = Path(__file__).resolve().parent / "vendor"
HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)

# ---------------------------------------------------------------------------
# configuration (the vendored configs.yaml, merged with the vendored common/config.py semantics)
# ---------------------------------------------------------------------------


def _vendored_config_class():
    spec = importlib.util.spec_from_file_location("_thick_vendored_config", VENDOR / "thick/common/config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Config


def _yaml12_loader():
    """PyYAML SafeLoader that also reads YAML-1.2 exponent floats (``1e-4``) as floats, as upstream's ruamel does."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                    |\.[0-9_]+(?:[eE][-+][0-9]+)?
                    |[-+]?\.(?:inf|Inf|INF)
                    |\.(?:nan|NaN|NAN))$""", re.X),
        list("-+0123456789."))
    return Loader


def load_config(names, overrides=None):
    """``defaults`` updated by each named config of the vendored ``configs.yaml`` then by ``overrides`` (as train.py)."""
    import yaml

    configs = yaml.load((VENDOR / "thick/configs.yaml").read_text(), Loader=_yaml12_loader())
    config = _vendored_config_class()(configs["defaults"])
    for name in names:
        config = config.update(configs[name])
    if overrides:
        config = config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# noise sources
# ---------------------------------------------------------------------------


class TorchNoise:
    """Default noise: torch generator draws (standard normal, uniform [0, 1))."""

    def __init__(self, generator=None):
        self.generator = generator

    def normal(self, shape, device=None):
        return torch.randn(tuple(shape), generator=self.generator, device=device)

    def uniform(self, shape, device=None):
        return torch.rand(tuple(shape), generator=self.generator, device=device)


class ReplayNoise:
    """Replays recorded draws in order (G-port): ``normals`` / ``uniforms`` are lists of arrays."""

    def __init__(self, normals, uniforms):
        self.normals, self.uniforms = list(normals), list(uniforms)
        self.used = {"normal": 0, "uniform": 0}

    def _pop(self, kind, queue, shape, device):
        if self.used[kind] >= len(queue):
            raise RuntimeError(f"replay noise exhausted ({kind} #{self.used[kind]})")
        value = torch.as_tensor(np.asarray(queue[self.used[kind]]), dtype=torch.float32, device=device)
        if tuple(value.shape) != tuple(shape):
            raise RuntimeError(f"replay {kind} #{self.used[kind]}: shape {tuple(value.shape)} != {tuple(shape)}")
        self.used[kind] += 1
        return value

    def normal(self, shape, device=None):
        return self._pop("normal", self.normals, shape, device)

    def uniform(self, shape, device=None):
        return self._pop("uniform", self.uniforms, shape, device)


# ---------------------------------------------------------------------------
# distributions (TFP semantics, float32)
# ---------------------------------------------------------------------------

UNIFORM_LOW, UNIFORM_HIGH = 1e-20, 1.0 - 1e-7


def gumbel_from_uniform(u):
    """Gumbel noise from uniform draws; the clamp is shared with the G-port reference's categorical sampler."""
    u = u.clamp(UNIFORM_LOW, UNIFORM_HIGH)
    return -torch.log(-torch.log(u))


class OneHot:
    """``common.OneHotDist`` (tfd.OneHotCategorical with a straight-through sample)."""

    def __init__(self, logits):
        self.logits = logits

    @property
    def probs(self):
        return torch.softmax(self.logits, -1)

    def mode(self):
        return F.one_hot(self.logits.argmax(-1), self.logits.shape[-1]).to(self.logits.dtype)

    def sample(self, noise):
        index = (self.logits.detach() + gumbel_from_uniform(noise.uniform(self.logits.shape, self.logits.device))).argmax(-1)
        sample = F.one_hot(index, self.logits.shape[-1]).to(self.logits.dtype)
        probs = self.probs
        return sample + (probs - probs.detach())

    def log_prob(self, x):
        # -softmax_cross_entropy_with_logits(labels=stop_gradient(x), logits)
        return (x.detach() * torch.log_softmax(self.logits, -1)).sum(-1)

    def entropy(self):
        logp = torch.log_softmax(self.logits, -1)
        return -(torch.softmax(self.logits, -1) * logp).sum(-1)

    def get_logits(self, shape=None):
        """Upstream ``OneHotDist.get_logits``: ``_pad`` ignores its argument and returns the probabilities (kept)."""
        probs = self.probs
        while shape is not None and probs.dim() < len(shape):
            probs = probs[None]
        return probs


def kl_onehot(a_logits, b_logits):
    """TFP ``_kl_categorical_categorical``: sum softmax(a) * (log_softmax(a) - log_softmax(b)) over the last axis."""
    return (torch.softmax(a_logits, -1) * (torch.log_softmax(a_logits, -1) - torch.log_softmax(b_logits, -1))).sum(-1)


def _sum_last(x, ndims):
    return x.sum(dim=tuple(range(-ndims, 0))) if ndims else x


class IndependentNormal:
    """``tfd.Independent(tfd.Normal(loc, scale), ndims)``."""

    def __init__(self, loc, scale, ndims):
        self.loc, self.scale, self.ndims = loc, scale, ndims

    def log_prob(self, x):
        scale = torch.as_tensor(self.scale, dtype=self.loc.dtype, device=self.loc.device)
        log_unnormalized = -0.5 * (x / scale - self.loc / scale) ** 2
        log_normalization = HALF_LOG_2PI + torch.log(scale)
        return _sum_last(log_unnormalized - log_normalization, self.ndims)

    def mode(self):
        return self.loc

    mean = mode

    def sample(self, noise):
        return self.loc + self.scale * noise.normal(self.loc.shape, self.loc.device)


def ndtr(x):
    """TFP ``special_math.ndtr``."""
    half_sqrt_2 = 0.5 * math.sqrt(2.0)
    w = x * half_sqrt_2
    z = w.abs()
    y = torch.where(z < half_sqrt_2, 1.0 + torch.erf(w), torch.where(w > 0.0, 2.0 - torch.erfc(z), torch.erfc(z)))
    return 0.5 * y


def _log_ndtr_lower(x, series_order=3):
    x_2 = x * x
    log_scale = -0.5 * x_2 - torch.log(-x) - HALF_LOG_2PI
    even_sum = torch.zeros_like(x)
    odd_sum = torch.zeros_like(x)
    x_2n = x_2
    for n in range(1, series_order + 1):
        y = float(np.prod(np.arange(2 * n - 1, 1, -2))) / x_2n
        if n % 2:
            odd_sum = odd_sum + y
        else:
            even_sum = even_sum + y
        x_2n = x_2n * x_2
    return log_scale + torch.log(1.0 + even_sum - odd_sum)


def log_ndtr(x):
    """TFP ``special_math.log_ndtr`` (float32 segments -10 / 5, series order 3)."""
    lower, upper = -10.0, 5.0
    return torch.where(x > upper, -ndtr(-x),
                       torch.where(x > lower, torch.log(ndtr(torch.clamp(x, min=lower))),
                                   _log_ndtr_lower(torch.clamp(x, max=lower))))


def log_sub_exp(x, y):
    larger, smaller = torch.maximum(x, y), torch.minimum(x, y)
    d = torch.clamp(larger - smaller, min=0.0).abs()
    log1mexp = torch.where(d < math.log(2.0), torch.log(-torch.expm1(-d)), torch.log1p(-torch.exp(-d)))
    return larger + log1mexp


class IndependentTruncNormal:
    """``tfd.Independent(common.TruncNormalDist(loc, scale, low, high), 1)`` (TFP 0.11 log-probability)."""

    def __init__(self, loc, scale, low=-1.0, high=1.0, clip=1e-6, ndims=1):
        self.loc, self.scale, self.low, self.high, self.clip, self.ndims = loc, scale, low, high, clip, ndims

    def log_prob(self, x):
        std_low, std_high = (self.low - self.loc) / self.scale, (self.high - self.loc) / self.scale
        log_normalizer = log_sub_exp(log_ndtr(std_high), log_ndtr(std_low))
        log_prob = -(0.5 * ((x - self.loc) / self.scale) ** 2 + HALF_LOG_2PI + torch.log(self.scale) + log_normalizer)
        log_prob = torch.where((x > self.high) | (x < self.low), torch.full_like(log_prob, -math.inf), log_prob)
        return _sum_last(log_prob, self.ndims)

    def mode(self):
        return self.loc


# ---------------------------------------------------------------------------
# layers (Keras semantics)
# ---------------------------------------------------------------------------


def get_act(name):
    if name == "none":
        return lambda x: x
    if name == "mish":
        return lambda x: x * torch.tanh(F.softplus(x))
    return {"elu": F.elu, "tanh": torch.tanh, "relu": F.relu, "sigmoid": torch.sigmoid,
            "softplus": F.softplus}[name]


class Dense(nn.Linear):
    """``tfkl.Dense``: glorot-uniform kernel, zero bias (weight = kernel.T)."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(int(in_features), int(out_features), bias=bias)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class NormLayer(nn.Module):
    def __init__(self, name, dim):
        super().__init__()
        if name == "none":
            self.layer = None
        elif name == "layer":
            self.layer = nn.LayerNorm(dim, eps=1e-3)
        else:
            raise NotImplementedError(name)

    def forward(self, x):
        return x if self.layer is None else self.layer(x)


def _shape(shape):
    return (shape,) if isinstance(shape, int) else tuple(shape)


class DistLayer(nn.Module):
    """``common.DistLayer`` (registry ``out`` / ``std``)."""

    NORMALS = ("normal", "tanh_normal", "trunc_normal", "clamp_normal", "pos_clamp_normal")

    def __init__(self, in_dim, shape, dist="mse", min_std=0.1, init_std=0.0):
        super().__init__()
        self.shape, self.dist, self.min_std, self.init_std = _shape(shape), dist, min_std, init_std
        size = int(np.prod(self.shape))
        self.m = nn.ModuleDict({"out": Dense(in_dim, size)})
        if dist in self.NORMALS:
            self.m["std"] = Dense(in_dim, size)

    def forward(self, x):
        lead = x.shape[:-1]
        out = self.m["out"](x).reshape(*lead, *self.shape)
        if self.dist in self.NORMALS:
            std = self.m["std"](x).reshape(*lead, *self.shape)
        if self.dist == "mse":
            return IndependentNormal(out, 1.0, len(self.shape))
        if self.dist == "pos_clamp_normal":
            return IndependentNormal(F.relu(out), F.elu(std) + 1.0 + self.min_std, len(self.shape))
        if self.dist == "clamp_normal":
            return IndependentNormal(out, F.elu(std) + 1.0 + self.min_std, len(self.shape))
        if self.dist == "trunc_normal":
            std = 2 * torch.sigmoid((std + self.init_std) / 2) + self.min_std
            return IndependentTruncNormal(torch.tanh(out), std, -1.0, 1.0, ndims=1)
        if self.dist == "onehot":
            return OneHot(out)
        raise NotImplementedError(self.dist)


class MLP(nn.Module):
    """``common.MLP`` (registry ``dense{i}{name}``, ``norm{i}{name}``, ``out{name}``)."""

    def __init__(self, in_dim, shape, layers, units, act="elu", norm="none", name="", **out):
        super().__init__()
        self.layers, self.name, self.act = layers, name, get_act(act)
        self.m = nn.ModuleDict()
        dim = in_dim
        for i in range(layers):
            self.m[f"dense{i}{name}"] = Dense(dim, units)
            self.m[f"norm{i}{name}"] = NormLayer(norm, units)
            dim = units
        self.m[f"out{name}"] = DistLayer(dim, shape, **out)

    def forward(self, features):
        x = features.reshape(-1, features.shape[-1])
        for i in range(self.layers):
            x = self.act(self.m[f"norm{i}{self.name}"](self.m[f"dense{i}{self.name}"](x)))
        x = x.reshape(*features.shape[:-1], x.shape[-1])
        return self.m[f"out{self.name}"](x)


class GRUCell(nn.Module):
    """``common.GRUCell`` (update bias -1; ``use_bias = norm is not None`` is always True upstream)."""

    def __init__(self, inp_dim, size, norm=False, act="tanh", update_bias=-1):
        super().__init__()
        self.size, self.act, self.update_bias = size, get_act(act), update_bias
        self.layer = Dense(inp_dim + size, 3 * size, bias=norm is not None)
        self.norm = nn.LayerNorm(3 * size, eps=1e-3) if norm else None

    def forward(self, inputs, state):
        parts = self.layer(torch.cat([inputs, state], -1))
        if self.norm is not None:
            parts = self.norm(parts)
        reset, cand, update = torch.split(parts, self.size, -1)
        reset = torch.sigmoid(reset)
        cand = self.act(reset * cand)
        update = torch.sigmoid(update + self.update_bias)
        output = update * cand + (1 - update) * state
        return output, update.detach()


class GateL0RDCell(nn.Module):
    """``common.SimpleGateL0RDCell`` (headless: output = next state), the THICK context kernel.

    update_gate = relu(tanh(g([x, c]) + noise)), noise ~ N(0, sample_sd) when sampling (or ``always_sample``);
    c' = update_gate * tanh(r([x, c])) + (1 - update_gate) * c; the returned gates are ceil(update_gate) with a
    straight-through gradient (the L0 count the sparsity loss penalises).
    """

    def __init__(self, inp_dim, size, act="tanh", sample_sd=0.05, always_sample=False, headless=True):
        super().__init__()
        if not headless:
            raise NotImplementedError("THICK uses the headless GateL0RD cell")
        self.size, self.act, self.sample_sd, self.always_sample = size, get_act(act), sample_sd, always_sample
        self.r_layer = Dense(inp_dim + size, size)
        self.g_layer = Dense(inp_dim + size, size)

    def forward(self, inputs, state, sample, noise):
        overall = torch.cat([inputs, state], -1)
        cand = self.act(self.r_layer(overall))
        update = self.g_layer(overall)
        if sample or self.always_sample:
            gate_noise = noise.normal(update.shape, update.device) * self.sample_sd + 0.0
            update_gate = F.relu(torch.tanh(update + gate_noise))
        else:
            update_gate = F.relu(torch.tanh(update))
        next_state = update_gate * cand + (1 - update_gate) * state
        gates = torch.ceil(update_gate).detach() + (update_gate - update_gate.detach())
        return next_state, gates


# ---------------------------------------------------------------------------
# encoder / decoder
# ---------------------------------------------------------------------------


def _keys(shapes, pattern, ndim):
    return [k for k, v in shapes.items() if re.match(pattern, k) and len(v) == ndim]


class Encoder(nn.Module):
    """``common.Encoder`` (registry ``conv{i}`` / ``convnorm{i}`` / ``dense{i}`` / ``densenorm{i}``); NHWC images."""

    def __init__(self, shapes, cnn_keys=r".*", mlp_keys=r".*", act="elu", norm="none", cnn_depth=48,
                 cnn_kernels=(4, 4, 4, 4), mlp_layers=(400, 400, 400, 400)):
        super().__init__()
        self.shapes = dict(shapes)
        self.cnn_keys, self.mlp_keys = _keys(shapes, cnn_keys, 3), _keys(shapes, mlp_keys, 1)
        self.act, self.cnn_kernels, self.mlp_layers = get_act(act), tuple(cnn_kernels), tuple(mlp_layers)
        self.m = nn.ModuleDict()
        self.out_dim = 0
        if self.cnn_keys:
            h, w = self.shapes[self.cnn_keys[0]][:2]
            channels = sum(self.shapes[k][-1] for k in self.cnn_keys)
            for i, kernel in enumerate(self.cnn_kernels):
                depth = 2 ** i * cnn_depth
                self.m[f"conv{i}"] = nn.Conv2d(channels, depth, kernel, 2)
                nn.init.xavier_uniform_(self.m[f"conv{i}"].weight)
                nn.init.zeros_(self.m[f"conv{i}"].bias)
                self.m[f"convnorm{i}"] = NormLayer(norm, depth)
                channels, h, w = depth, (h - kernel) // 2 + 1, (w - kernel) // 2 + 1
            self.out_dim += channels * h * w
        if self.mlp_keys:
            dim = sum(self.shapes[k][0] for k in self.mlp_keys)
            for i, width in enumerate(self.mlp_layers):
                self.m[f"dense{i}"] = Dense(dim, width)
                self.m[f"densenorm{i}"] = NormLayer(norm, width)
                dim = width
            self.out_dim += dim

    def forward(self, data):
        key, shape = next(iter(self.shapes.items()))
        batch_dims = data[key].shape[:data[key].dim() - len(shape)]
        outputs = []
        if self.cnn_keys:
            x = torch.cat([data[k].reshape(-1, *data[k].shape[len(batch_dims):]) for k in self.cnn_keys], -1)
            x = x.float().permute(0, 3, 1, 2)
            for i in range(len(self.cnn_kernels)):
                x = self.m[f"conv{i}"](x)
                if self.m[f"convnorm{i}"].layer is not None:
                    x = self.m[f"convnorm{i}"](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                x = self.act(x)
            outputs.append(x.permute(0, 2, 3, 1).reshape(x.shape[0], -1))
        if self.mlp_keys:
            x = torch.cat([data[k].reshape(-1, *data[k].shape[len(batch_dims):]) for k in self.mlp_keys], -1)
            x = x.float()
            for i in range(len(self.mlp_layers)):
                x = self.act(self.m[f"densenorm{i}"](self.m[f"dense{i}"](x)))
            outputs.append(x)
        output = torch.cat(outputs, -1)
        return output.reshape(*batch_dims, output.shape[-1])


class Decoder(nn.Module):
    """``common.Decoder`` (registry ``convin{name}``, ``conv{i}{name}``, ``dense{i}{name}``, ``dense_{key}{name}``)."""

    def __init__(self, shapes, in_dim, cnn_keys=r".*", mlp_keys=r".*", act="elu", norm="none", cnn_depth=48,
                 cnn_kernels=(4, 4, 4, 4), mlp_layers=(400, 400, 400, 400), name=""):
        super().__init__()
        self.shapes = dict(shapes)
        self.cnn_keys, self.mlp_keys = _keys(shapes, cnn_keys, 3), _keys(shapes, mlp_keys, 1)
        self.act, self.cnn_depth, self.cnn_kernels = get_act(act), cnn_depth, tuple(cnn_kernels)
        self.mlp_layers, self.name = tuple(mlp_layers), name
        self.m = nn.ModuleDict()
        if self.cnn_keys:
            channels = {k: self.shapes[k][-1] for k in self.cnn_keys}
            self.m["convin" + name] = Dense(in_dim, 32 * cnn_depth)
            cin = 32 * cnn_depth
            for i, kernel in enumerate(self.cnn_kernels):
                depth = 2 ** (len(self.cnn_kernels) - i - 2) * cnn_depth
                normname = norm
                if i == len(self.cnn_kernels) - 1:
                    depth, normname = sum(channels.values()), "none"
                conv = nn.ConvTranspose2d(cin, depth, kernel, 2)
                nn.init.xavier_uniform_(conv.weight)
                nn.init.zeros_(conv.bias)
                self.m[f"conv{i}{name}"] = conv
                self.m[f"convnorm{i}{name}"] = NormLayer(normname, depth)
                cin = depth
        if self.mlp_keys:
            dim = in_dim
            for i, width in enumerate(self.mlp_layers):
                self.m[f"dense{i}{name}"] = Dense(dim, width)
                self.m[f"densenorm{i}{name}"] = NormLayer(norm, width)
                dim = width
            for key in self.mlp_keys:
                self.m[f"dense_{key}{name}"] = DistLayer(dim, self.shapes[key])

    def forward(self, features):
        outputs = {}
        if self.cnn_keys:
            channels = {k: self.shapes[k][-1] for k in self.cnn_keys}
            x = self.m["convin" + self.name](features).reshape(-1, 32 * self.cnn_depth, 1, 1)
            for i in range(len(self.cnn_kernels)):
                last = i == len(self.cnn_kernels) - 1
                x = self.m[f"conv{i}{self.name}"](x)
                if self.m[f"convnorm{i}{self.name}"].layer is not None:
                    x = self.m[f"convnorm{i}{self.name}"](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                if not last:
                    x = self.act(x)
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(*features.shape[:-1], *x.shape[1:])
            means = torch.split(x, list(channels.values()), -1)
            outputs.update({key: IndependentNormal(mean, 1.0, 3) for key, mean in zip(channels, means)})
        if self.mlp_keys:
            x = features
            for i in range(len(self.mlp_layers)):
                x = self.act(self.m[f"densenorm{i}{self.name}"](self.m[f"dense{i}{self.name}"](x)))
            for key in self.mlp_keys:
                outputs[key] = self.m[f"dense_{key}{self.name}"](x)
        return outputs


# ---------------------------------------------------------------------------
# the context RSSM (C-RSSM): GateL0RD context kernel + GRU + discrete stochastic state
# ---------------------------------------------------------------------------


class ContextRSSM(nn.Module):
    """``common.ContextRSSM`` (discrete latents, ensemble 1)."""

    def __init__(self, embed_dim, act_dim, ensemble=1, stoch=32, deter=200, context=16, hidden=200, discrete=32,
                 act="elu", norm="none", std_act="softplus", min_std=0.1, rnn_norm="layer_norm",
                 ctxt_rnn_type="SimpleGateL0RD", ctxt_sample_noise=0.05, ctxt_rnn_out_size=-1,
                 ctxt_always_sample=False):
        super().__init__()
        if not discrete or ensemble != 1 or ctxt_rnn_type != "SimpleGateL0RD":
            raise NotImplementedError("ported: discrete latents, ensemble 1, SimpleGateL0RD")
        self.stoch, self.discrete, self.deter, self.context, self.hidden = stoch, discrete, deter, context, hidden
        self.act_dim, self.act = act_dim, get_act(act)
        flat = stoch * discrete
        self.cell = GRUCell(hidden + context, deter, norm=(rnn_norm == "layer_norm"))
        self.ctxt_rnn = GateL0RDCell(hidden, context, sample_sd=ctxt_sample_noise, always_sample=ctxt_always_sample)
        self.ctxt_head = MLP(flat + act_dim + context, deter, layers=3, units=hidden, name="contextout")
        self.m = nn.ModuleDict({
            "img_in": Dense(flat + act_dim, hidden), "img_in_norm": NormLayer(norm, hidden),
            "img_out_0": Dense(deter + context, hidden), "img_out_norm_0": NormLayer(norm, hidden),
            "img_dist_0": Dense(hidden, flat),
            "ctxt_img_out_0": Dense(flat + deter, hidden), "ctxt_img_out_norm_0": NormLayer(norm, hidden),
            "ctxt_img_dist_0": Dense(hidden, flat),
            "obs_out": Dense(context + deter + embed_dim, hidden), "obs_out_norm": NormLayer(norm, hidden),
            "obs_dist": Dense(hidden, flat),
        })

    @property
    def feat_dim(self):
        return self.stoch * self.discrete + self.deter + self.context

    @property
    def coarse_dim(self):
        return self.stoch * self.discrete + self.context

    def initial(self, batch, device=None):
        z = lambda *s: torch.zeros(batch, *s, device=device)
        return {"logit": z(self.stoch, self.discrete), "stoch": z(self.stoch, self.discrete),
                "gates": z(self.deter), "context": z(self.context), "deter": z(self.deter),
                "ctxt_logit": z(self.stoch, self.discrete), "ctxt_stoch": z(self.stoch, self.discrete),
                "prev_stoch": z(self.stoch * self.discrete), "prev_action": z(self.act_dim)}

    def _flat(self, x):
        return x.reshape(*x.shape[:-2], self.stoch * self.discrete)

    def _stats(self, x, prefix=""):
        x = self.act(self.m[f"{prefix}img_out_norm_0"](self.m[f"{prefix}img_out_0"](x)))
        x = self.m[f"{prefix}img_dist_0"](x)
        return x.reshape(*x.shape[:-1], self.stoch, self.discrete)

    def ctxt_out(self, prev_stoch, prev_action, context):
        """``_get_coarse_out_internal``: [prev_stoch, mode of the context head(prev_stoch, prev_action, context)]."""
        out = self.ctxt_head(torch.cat([prev_stoch, prev_action, context], -1)).mode()
        return torch.cat([prev_stoch, out], -1)

    def img_step(self, prev_state, prev_action, sample=True, rnn_sample=True, noise=None):
        prev_stoch = self._flat(prev_state["stoch"])
        x = self.act(self.m["img_in_norm"](self.m["img_in"](torch.cat([prev_stoch, prev_action], -1))))
        context, gates = self.ctxt_rnn(x, prev_state["context"], rnn_sample, noise)
        deter, _ = self.cell(torch.cat([x, context], -1), prev_state["deter"])
        logit = self._stats(torch.cat([deter, context], -1))
        dist = OneHot(logit)
        stoch = dist.sample(noise) if sample else dist.mode()
        ctxt_logit = self._stats(self.ctxt_out(prev_stoch, prev_action, context), prefix="ctxt_")
        ctxt_dist = OneHot(ctxt_logit)
        ctxt_stoch = ctxt_dist.sample(noise) if sample else ctxt_dist.mode()
        return {"stoch": stoch, "deter": deter, "context": context, "gates": gates, "logit": logit,
                "ctxt_logit": ctxt_logit, "ctxt_stoch": ctxt_stoch, "prev_stoch": prev_stoch,
                "prev_action": prev_action}

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True, rnn_sample=True, noise=None):
        keep = 1.0 - is_first.float()
        mask = lambda x: keep.reshape(-1, *([1] * (x.dim() - 1))) * x
        prev_state = {k: mask(v) for k, v in prev_state.items()}
        prev_action = mask(prev_action)
        prior = self.img_step(prev_state, prev_action, sample, rnn_sample, noise)
        x = torch.cat([prior["context"], prior["deter"], embed], -1)
        x = self.act(self.m["obs_out_norm"](self.m["obs_out"](x)))
        x = self.m["obs_dist"](x)
        logit = x.reshape(*x.shape[:-1], self.stoch, self.discrete)
        dist = OneHot(logit)
        stoch = dist.sample(noise) if sample else dist.mode()
        ctxt_stoch = dist.sample(noise) if sample else dist.mode()
        post = {"stoch": stoch, "deter": prior["deter"], "context": prior["context"], "gates": prior["gates"],
                "prev_stoch": prior["prev_stoch"], "prev_action": prior["prev_action"], "logit": logit,
                "ctxt_logit": logit, "ctxt_stoch": ctxt_stoch}
        return post, prior

    def observe(self, embed, action, is_first, state=None, sample=True, rnn_sample=True, noise=None):
        if state is None:
            state = self.initial(action.shape[0], action.device)
        posts, priors = [], []
        for t in range(action.shape[1]):
            state, prior = self.obs_step(state, action[:, t], embed[:, t], is_first[:, t], sample, rnn_sample,
                                         noise)
            posts.append(state)
            priors.append(prior)
        stack = lambda seq: {k: torch.stack([s[k] for s in seq], 1) for k in seq[0]}
        return stack(posts), stack(priors)

    def imagine(self, action, state, sample=True, rnn_sample=True, noise=None):
        priors = []
        for t in range(action.shape[1]):
            state = self.img_step(state, action[:, t], sample, rnn_sample, noise)
            priors.append(state)
        return {k: torch.stack([s[k] for s in priors], 1) for k in priors[0]}

    def get_feat(self, state):
        return torch.cat([self._flat(state["stoch"]), state["deter"], state["context"]], -1)

    def get_coarse_out(self, state):
        return torch.cat([self._flat(state["stoch"]), state["context"]], -1)

    def get_coarse_pred(self, state):
        """[ctxt_stoch, context] when the state carries ``ctxt_stoch`` (every img/obs step output does)."""
        return torch.cat([self._flat(state["ctxt_stoch"]), state["context"]], -1)

    def kl_loss(self, post, prior, forward, balance, free, free_avg, prefix=""):
        lhs, rhs = (prior, post) if forward else (post, prior)
        mix = balance if forward else (1 - balance)
        a, b = lhs[f"{prefix}logit"], rhs[f"{prefix}logit"]
        kl = lambda p, q: kl_onehot(p, q).sum(-1)
        if balance == 0.5:
            value = kl(a, b)
            return torch.clamp(value, min=free).mean(), value
        value = value_lhs = kl(a, b.detach())
        value_rhs = kl(a.detach(), b)
        if free_avg:
            loss_lhs, loss_rhs = torch.clamp(value_lhs.mean(), min=free), torch.clamp(value_rhs.mean(), min=free)
        else:
            loss_lhs, loss_rhs = torch.clamp(value_lhs, min=free).mean(), torch.clamp(value_rhs, min=free).mean()
        return mix * loss_lhs + (1 - mix) * loss_rhs, value


# ---------------------------------------------------------------------------
# THICK's high level (common/thick_utils.py)
# ---------------------------------------------------------------------------


def flatten_logits(x):
    return x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])


def unflatten_logits(x):
    side = int(math.sqrt(x.shape[-1]))
    return x.reshape(*x.shape[:-1], side, side)


def gen_thick_inps(latents, stoch):
    """[flat stoch, latent] (discrete)."""
    return torch.cat([flatten_logits(stoch).to(latents.dtype), latents], -1)


def gen_thick_segment_data(gates, is_terminal, thick_segment_crit):
    if "ends" in thick_segment_crit:
        return torch.cat([gates, is_terminal.to(gates.dtype)[..., None]], -1)
    return gates


def thick_data_gen_step(x_t, g_t, targets, t, g_mask=None):
    """Upstream ``thick_data_gen_step`` (time-major [S, B, D] targets)."""
    t_s, t_b, t_d = targets.shape
    g_clamp = torch.clamp(g_t.sum(-1, keepdim=True), 0, 1)
    g_act = g_clamp.repeat_interleave(x_t.shape[-1], -1)
    g_seq_t = g_act[None].repeat_interleave(t + 1, 0)
    ones = torch.ones(1, t_b, t_d, dtype=x_t.dtype, device=x_t.device)
    g_mask_prime = ones if t == 0 else torch.cat([g_mask, ones], 0)
    updates = g_seq_t * g_mask_prime
    # upstream pads the updates with t_S - (t + 1) zero rows (t never reaches t_S in gen_thick_data)
    padded = torch.cat([updates, torch.zeros(t_s - (t + 1), t_b, t_d, dtype=x_t.dtype, device=x_t.device)], 0)
    x_seq = x_t[None].repeat_interleave(t_s, 0)
    updated_targets = (1 - padded) * targets + padded * x_seq
    updated_g_mask = (1 - updates) * g_mask_prime
    return updated_targets, updated_g_mask


def gen_thick_data(inp_states, tar_states, tar_actions, tar_latent, tar_rewards, tar_stoch, gs, discount=0.99):
    """Upstream ``gen_thick_data`` (the second definition, the one in effect); inputs time-major [S, B, D]."""
    x_s, x_b, x_d = inp_states.shape
    y_d = tar_states.shape[-1]
    a_d, l_d, s_d = tar_actions.shape[-1], tar_latent.shape[-1], tar_stoch.shape[-1]
    dtype, device = tar_states.dtype, tar_states.device
    full = lambda d, v: torch.full((x_s - 1, x_b, d), float(v), dtype=dtype, device=device)
    target_states, target_actions, target_times = full(y_d, 99), full(a_d, 0), full(1, 99)
    target_latent, target_stoch = full(l_d, 99), full(s_d, 99)
    m_state = m_action = m_time = m_latent = m_stoch = None
    for t in range(x_s - 1):
        g = gs[t + 1]
        target_states, m_state = thick_data_gen_step(tar_states[t], g, target_states, t, m_state)
        target_actions, m_action = thick_data_gen_step(tar_actions[t + 1], g, target_actions, t, m_action)
        target_times, m_time = thick_data_gen_step(torch.full((x_b, 1), float(t), dtype=dtype, device=device),
                                                   g, target_times, t, m_time)
        target_latent, m_latent = thick_data_gen_step(tar_latent[t + 1], g, target_latent, t, m_latent)
        target_stoch, m_stoch = thick_data_gen_step(tar_stoch[t], g, target_stoch, t, m_stoch)
    seq_targets = {"state": target_states, "action": target_actions, "time": target_times,
                   "latent": target_latent, "stoch": target_stoch}
    discounts = (1.0 - torch.clamp(gs[1:].sum(-1, keepdim=True), 0, 1)) * discount
    returns, last = [], torch.zeros_like(discounts[-1])
    for index in reversed(range(x_s - 1)):
        last = tar_rewards[:-1][index] + discounts[index] * last
        returns.append(last)
    inter_returns = torch.stack(returns[::-1], 0)
    n = (x_s - 1) * x_b
    t_asc = torch.arange(x_s - 1, device=device)[:, None, None].expand(x_s - 1, x_b, 1).to(dtype)
    targets = {
        "thick_tar_states": target_states.reshape(n, y_d), "thick_act": target_actions.reshape(n, a_d),
        "thick_time": (target_times - t_asc).reshape(n, 1), "thick_latent": target_latent.reshape(n, l_d),
        "thick_inter_returns": inter_returns.reshape(n, 1), "thick_stoch": target_stoch.reshape(n, s_d)}
    inps = inp_states[:x_s - 1].reshape(n, x_d)
    mask = lambda m, d: 1.0 - torch.clamp(m.reshape(n, d).float().sum(-1, keepdim=True), 0, 1)
    masks = {"state": mask(m_state, y_d), "action": mask(m_action, a_d), "time": mask(m_time, 1)}
    return inps, targets, masks, seq_targets


class THICKNet(nn.Module):
    """``THICKNet``: the high level (state at the next context change, action, elapsed time, inter-return)."""

    def __init__(self, inp_dim, stoch, hidden, discrete, action_dim, cont_action, config, act="elu", int_layers=3,
                 min_std=0.1):
        super().__init__()
        if not discrete:
            raise NotImplementedError("ported: discrete latents")
        self.stoch, self.hidden, self.discrete, self.action_dim = stoch, hidden, discrete, action_dim
        self.cont_action, self.config, self.act, self.int_layers = cont_action, config, get_act(act), int_layers
        self.hl_act_dim = int(config.thick_hl_act_dim)
        readout = config.thick_readouts
        if readout.sample_skip:
            raise NotImplementedError("ported: thick_readouts.sample_skip False")
        self.m = nn.ModuleDict()
        dim = inp_dim + max(self.hl_act_dim, 0)
        for i in range(int_layers):
            self.m[f"thick_layer{i}"] = Dense(dim, hidden)
            dim = hidden
        self.m["thick_logit_pre"] = Dense(hidden, hidden)
        self.m["thick_logit_out"] = Dense(hidden, stoch * discrete)
        if self.hl_act_dim > 0:
            self.m["hl_act_prior_pre"] = Dense(inp_dim, hidden)
            self.m["hl_act_prior_out"] = Dense(hidden, self.hl_act_dim)
            post_dim = inp_dim + (inp_dim - stoch * discrete) + stoch * discrete
            self.m["hl_act_post_pre"] = Dense(post_dim, hidden)
            self.m["hl_act_post_out"] = Dense(hidden, self.hl_act_dim)
        self.thick_heads = nn.ModuleDict({
            "action": MLP(hidden, [action_dim], readout.layers, readout.units, name="thick_action",
                          dist="trunc_normal" if cont_action else "onehot"),
            "time": MLP(hidden, [1], readout.layers, readout.units, name="thick_state", dist="pos_clamp_normal"),
            "reward": MLP(hidden, [1], readout.layers, readout.units, name="thick_reward", dist="mse")})

    def body(self, x):
        for i in range(self.int_layers):
            x = self.act(self.m[f"thick_layer{i}"](x))
        return x

    def state_logits(self, x):
        x = self.m["thick_logit_out"](self.act(self.m["thick_logit_pre"](x)))
        return x.reshape(*x.shape[:-1], self.stoch, self.discrete)

    def hl_prior_logits(self, inps):
        return self.m["hl_act_prior_out"](self.act(self.m["hl_act_prior_pre"](inps)))

    def hl_post_logits(self, inps_posterior):
        return self.m["hl_act_post_out"](self.act(self.m["hl_act_post_pre"](inps_posterior)))

    def sample_step(self, inps, sample=True, noise=None, hl_act=None):
        """One high-level prediction (prior high-level action unless ``hl_act`` is given)."""
        if self.hl_act_dim > 0:
            if hl_act is None:
                prior = OneHot(self.hl_prior_logits(inps))
                hl_act = prior.sample(noise) if sample else prior.mode()
            full = torch.cat([inps, hl_act.to(inps.dtype)], -1)
        else:
            full = inps
        feat = self.body(full)
        logit = self.state_logits(feat)
        dist = OneHot(logit)
        return {"thick_stoch": dist.sample(noise) if sample else dist.mode(), "thick_logit": logit,
                "thick_act": self.thick_heads["action"](feat).mode(),
                "thick_time": self.thick_heads["time"](feat).mode(),
                "reward": self.thick_heads["reward"](feat).mode(), "hl_act": hl_act}

    def kl_loss(self, post_logits, prior_logits, loss_mask, forward, balance, free, free_avg, independent):
        lhs, rhs = (prior_logits, post_logits) if forward else (post_logits, prior_logits)
        mix = balance if forward else (1 - balance)
        kl = (lambda p, q: kl_onehot(p, q).sum(-1)) if independent else kl_onehot
        if balance == 0.5:
            value = kl(lhs, rhs) * loss_mask
            return torch.clamp(value, min=free).mean(), value
        value_lhs = kl(lhs, rhs.detach()) * loss_mask
        value_rhs = kl(lhs.detach(), rhs) * loss_mask
        if free_avg:
            loss_lhs, loss_rhs = torch.clamp(value_lhs.mean(), min=free), torch.clamp(value_rhs.mean(), min=free)
        else:
            loss_lhs, loss_rhs = torch.clamp(value_lhs, min=free).mean(), torch.clamp(value_rhs, min=free).mean()
        return mix * loss_lhs + (1 - mix) * loss_rhs, value_lhs

    def thick_loss(self, inps, targets, loss_masks, noise=None):
        cfg = self.config
        losses, metrics = {}, {}
        loss_mask = loss_masks["time"].float().squeeze(-1)
        if self.hl_act_dim > 0:
            inps_post = torch.cat([inps, targets["thick_latent"].to(inps.dtype), targets["thick_stoch"].to(inps.dtype)],
                                  -1)
            post_logit, prior_logit = self.hl_post_logits(inps_post), self.hl_prior_logits(inps)
            kl = cfg.thick_hl_act_kl
            losses["thick_hl_act_kl"], _ = self.kl_loss(post_logit, prior_logit, loss_mask, kl.forward, kl.balance,
                                                        kl.free, kl.free_avg, independent=False)
            post_sample = OneHot(post_logit).sample(noise)
            full = torch.cat([inps, post_sample.to(inps.dtype)], -1)
            metrics["hl_act_prior_ent"] = OneHot(prior_logit).entropy()
            metrics["hl_act_post_ent"] = OneHot(post_logit).entropy()
        else:
            full = inps
        feat = self.body(full)
        logit = self.state_logits(feat)
        kl = cfg.thick_kl
        losses["thick_kl"], _ = self.kl_loss(targets["thick_logit"], logit, loss_mask, kl.forward, kl.balance,
                                             kl.free, kl.free_avg, independent=True)
        target_act = targets["thick_act"].float()
        act_dist = self.thick_heads["action"](feat.float())
        if self.cont_action:
            act_loss = -act_dist.log_prob(target_act)
        elif cfg.thick_act_pred_loss == "CCEL_logits":
            act_loss = -(target_act * torch.log_softmax(act_dist.get_logits(target_act.shape), -1)).sum(-1)
        elif cfg.thick_act_pred_loss == "NLL":
            act_loss = -act_dist.log_prob(target_act)
        else:
            raise NotImplementedError(cfg.thick_act_pred_loss)
        losses["thick_act"] = (act_loss * loss_mask).mean()
        time_dist = self.thick_heads["time"](feat.float())
        losses["thick_time"] = -(time_dist.log_prob(targets["thick_time"].float()) * loss_mask).mean()
        reward_dist = self.thick_heads["reward"](feat.float())
        losses["thick_reward"] = -(reward_dist.log_prob(targets["thick_inter_returns"].float()) * loss_mask).mean()
        return losses, metrics


# ---------------------------------------------------------------------------
# optimizer (common/tfutils.py Optimizer, Keras Adam)
# ---------------------------------------------------------------------------


class TFOptimizer:
    """``common.Optimizer`` with Keras (TF 2.4) Adam: global-norm clip, ``var *= 1 - wd`` then the Adam update."""

    def __init__(self, named_params, lr, eps=1e-4, clip=None, wd=None, opt="adam", wd_pattern=r".*",
                 beta_1=0.9, beta_2=0.999, name="opt"):
        if opt != "adam":
            raise NotImplementedError(opt)
        assert wd is None or 0 <= wd < 1
        assert not clip or 1 <= clip
        self.named = list(named_params)
        self.lr, self.eps, self.clip, self.wd, self.wd_pattern = lr, eps, clip, wd, wd_pattern
        self.beta_1, self.beta_2, self.name = beta_1, beta_2, name
        self.iterations = 0
        self.m = [torch.zeros_like(p) for _, p in self.named]
        self.v = [torch.zeros_like(p) for _, p in self.named]

    @property
    def params(self):
        return [p for _, p in self.named]

    def state_dict(self):
        return {"iterations": self.iterations, "m": [t.clone() for t in self.m], "v": [t.clone() for t in self.v],
                "names": [n for n, _ in self.named]}

    def load_state_dict(self, state):
        assert state["names"] == [n for n, _ in self.named]
        self.iterations = int(state["iterations"])
        self.m = [t.clone() for t in state["m"]]
        self.v = [t.clone() for t in state["v"]]

    def gradients(self, loss):
        return torch.autograd.grad(loss, self.params, allow_unused=True)

    @torch.no_grad()
    def apply(self, grads):
        """Clip, decay and update with precomputed gradients; returns the pre-clip global norm."""
        present = [g for g in grads if g is not None]
        norm = torch.sqrt(sum((g.float() ** 2).sum() for g in present))
        if not torch.isfinite(norm):
            raise FloatingPointError(f"{self.name}_norm is not finite")
        if self.clip:
            scale = self.clip * torch.minimum(1.0 / norm, torch.tensor(1.0 / self.clip, device=norm.device))
            grads = [None if g is None else g * scale for g in grads]
        if self.wd:
            for name, p in self.named:
                if re.search(self.wd_pattern, self.name + "/" + name):
                    p.mul_(1 - self.wd)
        self.iterations += 1
        t = torch.tensor(float(self.iterations), device=norm.device)
        b1 = torch.tensor(self.beta_1, device=norm.device)
        b2 = torch.tensor(self.beta_2, device=norm.device)
        alpha = self.lr * torch.sqrt(1 - b2 ** t) / (1 - b1 ** t)
        for (name, p), g, m, v in zip(self.named, grads, self.m, self.v):
            if g is None:
                continue
            m.add_((g - m) * (1 - self.beta_1))
            v.add_((g * g - v) * (1 - self.beta_2))
            p.sub_(alpha * m / (torch.sqrt(v) + self.eps))
        return norm

    def step(self, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{self.name}_loss is not finite")
        return self.apply(self.gradients(loss))


# ---------------------------------------------------------------------------
# the THICK world model (agent.py WorldModel: low level + high level, losses, train / train_thick)
# ---------------------------------------------------------------------------


class WorldModel(nn.Module):
    """``agent.WorldModel`` with ``c_rssm`` (C-RSSM) and ``hierarchical`` (THICKNet), discrete or continuous actions.

    ``obs_space``: {key: shape} in upstream order (e.g. ``carrier: (236,)``, ``reward: ()``, ``is_first: ()``, ...).
    """

    def __init__(self, config, obs_space, act_dim, cont_action):
        super().__init__()
        if not config.c_rssm:
            raise NotImplementedError("ported: the THICK C-RSSM (c_rssm True)")
        if config.pred_discount:
            raise NotImplementedError("ported: pred_discount False (no early termination)")
        self.config, self.act_dim, self.cont_action = config, act_dim, cont_action
        shapes = {k: tuple(v) for k, v in obs_space.items()}
        self.encoder = Encoder(shapes, **config.encoder)
        rssm = dict(config.rssm)
        self.rssm = ContextRSSM(self.encoder.out_dim, act_dim, **rssm, **config.context)
        feat, coarse = self.rssm.feat_dim, self.rssm.coarse_dim
        self.heads = nn.ModuleDict({"decoder": Decoder(shapes, feat, **config.decoder),
                                    "reward": MLP(feat, [], **config.reward_head)})
        self.ctxt_heads = nn.ModuleDict({"decoder": Decoder(shapes, coarse, **config.decoder, name="context"),
                                         "reward": MLP(coarse, [], **config.reward_head, name="context")})
        self.grad_heads = tuple(config.grad_heads)
        self.ctxt_grad_heads = () if config.ctxt_grad_heads == "none" else tuple(config.ctxt_grad_heads)
        self.loss_scales = dict(config.loss_scales)
        self.hierarchical = bool(config.hierarchical)
        if self.hierarchical:
            self.thick_net = THICKNet(coarse, config.rssm.stoch, config.thick_feat, config.rssm.discrete, act_dim,
                                      cont_action, config)

    # -- parameter groups (upstream optimizer module lists) --
    def model_named_parameters(self):
        modules = [("encoder", self.encoder), ("rssm", self.rssm)]
        modules += [(f"heads.{k}", v) for k, v in self.heads.items()]
        modules += [(f"ctxt_heads.{k}", v) for k, v in self.ctxt_heads.items()]
        return [(f"{prefix}.{n}", p) for prefix, module in modules for n, p in module.named_parameters()]

    def thick_named_parameters(self):
        return [(f"thick_net.{n}", p) for n, p in self.thick_net.named_parameters()]

    def make_optimizers(self):
        model = TFOptimizer(self.model_named_parameters(), name="model", **self.config.model_opt)
        thick = TFOptimizer(self.thick_named_parameters(), name="thick_net", **self.config.thick_opt) \
            if self.hierarchical else None
        return model, thick

    def preprocess(self, obs):
        obs = dict(obs)
        for key, value in obs.items():
            if key.startswith("log_"):
                continue
            if value.dtype == torch.uint8:
                obs[key] = value.float() / 255.0 - 0.5
            elif value.dtype in (torch.int32, torch.int64, torch.bool):
                obs[key] = value.float()
        clip = {"identity": lambda x: x, "sign": torch.sign, "tanh": torch.tanh}[self.config.clip_rewards]
        obs["reward"] = clip(obs["reward"].float())
        obs["discount"] = (1.0 - obs["is_terminal"].float()) * self.config.discount
        return obs

    def loss(self, data, state=None, noise=None):
        """``WorldModel.loss``: (model_loss, last_state, outputs, losses)."""
        data = self.preprocess(data)
        embed = self.encoder(data)
        post, prior = self.rssm.observe(embed, data["action"], data["is_first"], state, noise=noise)
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
        ctxt_kl_loss, ctxt_kl_value = self.rssm.kl_loss(post, prior, **self.config.ctxt_kl, prefix="ctxt_")
        gate_activity = post["gates"].mean()
        losses = {"kl": kl_loss, "gate_activity": gate_activity, "ctxt_sparsity": gate_activity,
                  "ctxt_kl": ctxt_kl_loss}
        feat = self.rssm.get_feat(post)
        likes = {}
        for name, head in self.heads.items():
            out = head(feat if name in self.grad_heads else feat.detach())
            dists = out if isinstance(out, dict) else {name: out}
            for key, dist in dists.items():
                likes[key] = dist.log_prob(data[key]).float()
                losses[key] = -likes[key].mean()
        for name, head in self.ctxt_heads.items():
            c_inp = self.rssm.get_coarse_out(post)
            out = head(c_inp if name in self.ctxt_grad_heads else c_inp.detach())
            dists = out if isinstance(out, dict) else {name: out}
            for key, dist in dists.items():
                losses[f"ctxt_{key}"] = -dist.log_prob(data[key]).float().mean()
        model_loss = sum(self.loss_scales.get(k, 1.0) * v for k, v in losses.items())
        outs = {"embed": embed, "feat": feat, "post": post, "prior": prior, "likes": likes, "kl": kl_value,
                "ctxt_kl": ctxt_kl_value}
        last_state = {k: v[:, -1] for k, v in post.items()}
        return model_loss, last_state, outs, losses

    def preprocess_thick_data(self, data, post):
        gates = post["gates"]
        flat_logits, flat_stoch = flatten_logits(post["logit"]), flatten_logits(post["stoch"])
        swap = lambda x: x.transpose(0, 1)
        xs = gen_thick_inps(post["context"], post["stoch"])
        segments = gen_thick_segment_data(gates, data["is_terminal"], tuple(self.config.thick_segment_crit))
        inps, targets, masks, seq_targets = gen_thick_data(
            inp_states=swap(xs).detach(), tar_states=swap(flat_logits).detach(), tar_actions=swap(data["action"]),
            tar_latent=swap(post["context"]).detach(), tar_rewards=swap(data["reward"][..., None]).detach(),
            tar_stoch=swap(flat_stoch).detach(), gs=swap(segments).detach(), discount=self.config.discount)
        targets["thick_logit"] = unflatten_logits(targets["thick_tar_states"])
        return inps, targets, masks, seq_targets

    def thick_loss(self, data, state=None, noise=None):
        data = self.preprocess(data)
        embed = self.encoder(data)
        post, _ = self.rssm.observe(embed, data["action"], data["is_first"], state, noise=noise)
        inps, targets, masks, _ = self.preprocess_thick_data(data, post)
        losses, metrics = self.thick_net.thick_loss(inps, targets, masks, noise=noise)
        scales = dict(self.config.thick_loss_scales)
        total = sum(scales.get(k, 1.0) * v for k, v in losses.items())
        return total, losses, {"inps": inps, "targets": targets, "masks": masks, **metrics}

    def train_step(self, data, model_opt, thick_opt, state=None, noise=None):
        """``Agent.train``'s world-model part: ``wm.train`` then ``wm.train_thick`` (on the updated model)."""
        model_loss, last_state, outs, losses = self.loss(data, state, noise)
        model_norm = model_opt.step(model_loss)
        out = {"model_loss": model_loss.detach(), "losses": {k: v.detach() for k, v in losses.items()},
               "model_grad_norm": model_norm, "state": {k: v.detach() for k, v in last_state.items()}}
        if self.hierarchical:
            thick_total, thick_losses, _ = self.thick_loss(data, state, noise)
            out["thick_grad_norm"] = thick_opt.step(thick_total)
            out["thick_loss"] = thick_total.detach()
            out["thick_losses"] = {k: v.detach() for k, v in thick_losses.items()}
        return out
