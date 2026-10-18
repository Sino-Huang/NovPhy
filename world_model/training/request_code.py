"""Issue #121: request-code arms of the CNN hybrid (fused control, additive, factored-FiLM).

How the request reaches the trunk in ``CNNHybridPredictor.carrier`` (verified, not taken from the ticket):

1. the code ``c = PairConditioner.code(pair)``: ``MLP[phi(Delta); e_alpha]`` (sinusoidal phi, 16 values;
   learned 128-d e_alpha; Linear 144 -> 256, SiLU, Linear 256 -> 128), broadcast over the batch;
2. in each of the three ``FiLMBlock`` s, a zero-initialised ``Linear(128, 2 * 384)`` maps the code to a
   scale gamma and a shift beta: ``norm(h) * (1 + gamma) + beta``;
3. alpha also selects the input adapter (micro / macro symbolic features of the current carrier added to the
   input projection). That path depends on alpha only, never on Delta, and is identical in every arm here.

The arms differ only in steps 1-2:

- ``fused``: ``CNNHybridPredictor`` itself;
- ``additive``: ``c = W_delta phi(Delta) + W_alpha e_alpha`` (128-d, so the FiLM interface is unchanged). The
  nine codes have the rank-2 structure c(Delta, alpha) = u(Delta) + v(alpha), and since each FiLM generator is
  affine, gamma and beta are additive in the two axes as well;
- ``factored``: per-axis FiLM generators in every block, gamma_d, beta_d = G_d(phi(Delta)) and gamma_a,
  beta_a = G_a(e_alpha), composed as ``norm(h) * (1 + gamma_d) * (1 + gamma_a) + beta_d + beta_a``. The codes
  stay per-axis (no code MLP); the modulation may interact multiplicatively in the scale.

The arm classes build the whole ``CNNHybridPredictor`` first (same constructor, so the same RNG draws), then
replace the request path. Under ``torch.manual_seed(seed)`` every trunk, head and adapter parameter therefore
starts equal to the fused member of the same seed; only the request-path parameters differ.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from world_model.model import Abstraction, PredictionPair
from world_model.model.config import abstraction_index
from world_model.model.predictor import FiLMBlock
from world_model.training.cnn_hybrid import CONFIG, PAIRS, CNNHybridPredictor

ARMS = ("fused", "additive", "factored")
DELTAS = tuple(sorted({p.delta for p in PAIRS}))
ALPHAS = tuple(dict.fromkeys(p.abstraction for p in PAIRS))


def delta_features(config, delta: int, device) -> torch.Tensor:
    """``PairConditioner.delta_features`` (sinusoidal features of Delta), the same ops."""
    count = config.delta_frequency_count
    exponents = torch.arange(count, dtype=torch.float32, device=device) / count
    frequencies = torch.pow(torch.tensor(10000.0, device=device), -exponents)
    angles = float(delta) * frequencies
    return torch.cat((torch.sin(angles), torch.cos(angles)))


def _check(pair, batch_size):
    if type(pair) is not PredictionPair:
        raise ValueError("prediction pair must be a PredictionPair")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")


class AdditiveConditioner(nn.Module):
    """``c = W_delta phi(Delta) + W_alpha e_alpha``; one bias (on W_delta)."""

    def __init__(self, config):
        super().__init__()
        self._config = config
        self.abstraction_embedding = nn.Embedding(len(Abstraction), config.pair_code_dim)
        self.delta_projection = nn.Linear(2 * config.delta_frequency_count, config.pair_code_dim)
        self.abstraction_projection = nn.Linear(config.pair_code_dim, config.pair_code_dim, bias=False)

    @property
    def code_dim(self) -> int:
        return self._config.pair_code_dim

    def axis_codes(self, pair: PredictionPair, device) -> tuple[torch.Tensor, torch.Tensor]:
        """(W_delta phi(Delta) + b, W_alpha e_alpha), unbatched."""
        index = torch.tensor(abstraction_index(pair.abstraction), device=device)
        phi = delta_features(self._config, pair.delta, device).to(self.delta_projection.weight.dtype)
        return (self.delta_projection(phi),
                self.abstraction_projection(self.abstraction_embedding(index)))

    def code(self, pair: PredictionPair, batch_size: int, device) -> torch.Tensor:
        _check(pair, batch_size)
        delta, alpha = self.axis_codes(pair, device)
        return (delta + alpha).unsqueeze(0).expand(batch_size, -1)


class FactoredConditioner(nn.Module):
    """Per-axis inputs of the factored FiLM generators: (phi(Delta), e_alpha), each broadcast over the batch."""

    def __init__(self, config):
        super().__init__()
        self._config = config
        self.abstraction_embedding = nn.Embedding(len(Abstraction), config.pair_code_dim)

    def code(self, pair: PredictionPair, batch_size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        _check(pair, batch_size)
        index = torch.tensor(abstraction_index(pair.abstraction), device=device)
        phi = delta_features(self._config, pair.delta, device).to(self.abstraction_embedding.weight.dtype)
        alpha = self.abstraction_embedding(index)
        return phi.unsqueeze(0).expand(batch_size, -1), alpha.unsqueeze(0).expand(batch_size, -1)


class FactoredFiLMBlock(nn.Module):
    """``FiLMBlock`` with per-axis zero-initialised generators: norm(h) (1 + g_d)(1 + g_a) + b_d + b_a.

    ``norm``, ``inner`` and ``outer`` are taken from a constructed ``FiLMBlock`` (its trunk parameters)."""

    def __init__(self, block: FiLMBlock, delta_dim: int, alpha_dim: int):
        super().__init__()
        width = block.norm.normalized_shape[0]
        self.norm = block.norm
        self.inner = block.inner
        self.outer = block.outer
        self.modulation_delta = nn.Linear(delta_dim, 2 * width)
        self.modulation_alpha = nn.Linear(alpha_dim, 2 * width)
        for layer in (self.modulation_delta, self.modulation_alpha):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def film(self, code):
        """(scale_delta, shift_delta, scale_alpha, shift_alpha)."""
        phi, alpha = code
        return (*self.modulation_delta(phi).chunk(2, dim=-1), *self.modulation_alpha(alpha).chunk(2, dim=-1))

    def forward(self, hidden: torch.Tensor, code) -> torch.Tensor:
        scale_d, shift_d, scale_a, shift_a = self.film(code)
        modulated = self.norm(hidden) * ((1.0 + scale_d) * (1.0 + scale_a)) + (shift_d + shift_a)
        return hidden + self.outer(F.silu(self.inner(modulated)))


class AdditiveCodePredictor(CNNHybridPredictor):
    """The CNN hybrid with the additive request code (FiLM blocks unchanged)."""

    def __init__(self, config=CONFIG):
        super().__init__(config)
        self.conditioner = AdditiveConditioner(config)


class FactoredFiLMPredictor(CNNHybridPredictor):
    """The CNN hybrid with per-axis codes and per-axis FiLM generators in every block."""

    def __init__(self, config=CONFIG):
        super().__init__(config)
        self.conditioner = FactoredConditioner(config)
        self.blocks = nn.ModuleList(FactoredFiLMBlock(block, 2 * config.delta_frequency_count, config.pair_code_dim)
                                    for block in self.blocks)


CLASSES = {"fused": CNNHybridPredictor, "additive": AdditiveCodePredictor, "factored": FactoredFiLMPredictor}


def new_predictor(arm: str) -> nn.Module:
    return CLASSES[arm]()


def request_path_names(arm: str) -> tuple[str, ...]:
    """Parameter-name prefixes of the request path (code + FiLM generators)."""
    if arm == "factored":
        return ("conditioner.", *(f"blocks.{b}.modulation_{axis}." for b in range(CONFIG.depth)
                                  for axis in ("delta", "alpha")))
    return ("conditioner.", *(f"blocks.{b}.modulation." for b in range(CONFIG.depth)))


def parameter_split(model: nn.Module, arm: str) -> dict:
    prefixes = request_path_names(arm)
    request = sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefixes))
    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "request_path": request, "trunk_heads_adapters": total - request}


@torch.no_grad()
def film_table(model: nn.Module, arm: str, device="cpu", dtype=torch.float64) -> dict:
    """Effective FiLM parameters of every block at the nine requests: per block, scale gamma and shift beta
    as [len(DELTAS), len(ALPHAS), width] tensors (factored: gamma = (1 + g_d)(1 + g_a) - 1, beta = b_d + b_a);
    and the code itself [Delta, alpha, code] for fused / additive. Computed in ``dtype`` on ``device`` on a
    copy of the model."""
    import copy
    model = copy.deepcopy(model).to(device=device, dtype=dtype).eval()
    gammas = [[[None] * len(ALPHAS) for _ in DELTAS] for _ in model.blocks]
    betas = [[[None] * len(ALPHAS) for _ in DELTAS] for _ in model.blocks]
    codes = [[None] * len(ALPHAS) for _ in DELTAS]
    for i, delta in enumerate(DELTAS):
        for j, alpha in enumerate(ALPHAS):
            pair = PredictionPair(delta, alpha)
            code = model.conditioner.code(pair, 1, device)
            if arm == "factored":
                code = tuple(c.to(dtype) for c in code)
                codes[i][j] = None
            else:
                code = code.to(dtype)
                codes[i][j] = code[0]
            for b, block in enumerate(model.blocks):
                if arm == "factored":
                    scale_d, shift_d, scale_a, shift_a = block.film(code)
                    scale, shift = (1.0 + scale_d) * (1.0 + scale_a) - 1.0, shift_d + shift_a
                else:
                    scale, shift = block.modulation(code).chunk(2, dim=-1)
                gammas[b][i][j], betas[b][i][j] = scale[0], shift[0]

    def stack(grid):
        return torch.stack([torch.stack(row) for row in grid])
    return {"gamma": [stack(g) for g in gammas], "beta": [stack(g) for g in betas],
            "code": None if arm == "factored" else stack(codes)}


def two_way(table: torch.Tensor) -> dict:
    """Two-way decomposition of a [Delta, alpha, d] table of vectors (one observation per cell).

    Sums of squares over the vector dimension: SS_total around the grand mean, SS_Delta and SS_alpha of the
    main effects, SS_interaction = sum ||x_ij - x_i. - x_.j + x_..||^2 (zero iff the table is additive).
    ``alpha_given_delta`` = mean over Delta of the variance over alpha at fixed Delta, relative to the mean
    squared norm of the table (the identity FiLM is gamma = beta = 0); ``delta_given_alpha`` likewise."""
    x = table.double()
    grand = x.mean((0, 1), keepdim=True)
    row = x.mean(1, keepdim=True)
    col = x.mean(0, keepdim=True)
    interaction = x - row - col + grand
    ss_total = float(((x - grand) ** 2).sum())
    ss_delta = float(x.shape[1] * ((row - grand) ** 2).sum())
    ss_alpha = float(x.shape[0] * ((col - grand) ** 2).sum())
    ss_int = float((interaction ** 2).sum())
    energy = float((x ** 2).sum(-1).mean())
    within_delta = float(((x - row) ** 2).sum(-1).mean())
    within_alpha = float(((x - col) ** 2).sum(-1).mean())

    def ratio(a, b):
        return a / b if b > 0 else None
    return {"ss_total": ss_total, "ss_delta": ss_delta, "ss_alpha": ss_alpha, "ss_interaction": ss_int,
            "interaction_share": ratio(ss_int, ss_total),
            "interaction_share_of_alpha_variation": ratio(ss_int, ss_alpha + ss_int),
            "interaction_share_of_delta_variation": ratio(ss_int, ss_delta + ss_int),
            "rms": energy ** 0.5 / x.shape[-1] ** 0.5,
            "alpha_given_delta": ratio(within_delta, energy), "delta_given_alpha": ratio(within_alpha, energy)}


def coupling(model: nn.Module, arm: str) -> dict:
    """Per block: two-way decomposition of the effective gamma and beta tables; and of the code (fused /
    additive)."""
    table = film_table(model, arm)
    return {"blocks": [{"gamma": two_way(g), "beta": two_way(b)} for g, b in zip(table["gamma"], table["beta"])],
            "code": None if table["code"] is None else two_way(table["code"])}
