"""Issue #125: the #123 H-JEPA / LeWM arms on NovPhy plus the paper's inverse-dynamics remedy.

Only the IDM terms change from #123 (H-JEPA §4.3 Eq. 8, App. K; the DROID recipe, App. C.2):

- **IDM head and loss.** The official ``loss.InverseDynamicsModel`` (``MLP(2D -> 256 -> 256 -> d_a)``, ReLU,
  truncated-normal init) inside the official ``loss.InverseDynamicsLoss``
  (``mean ||I(z_t, z_{t+1}) - sg[a_t]||^2``), one per level with a positive coefficient, attached as
  ``jepa.idm_loss_fn`` exactly as ``hjepa_utils.create_world_model`` attaches it. The vendored ``hjepa_forward``
  adds ``idm_coeff * idm_loss_fn(embed_l, action_l)`` to the level loss; ``action_l`` is the level's action
  embedding: at level 1 the embedded launch on the launch transition and the embedded no-op elsewhere, at
  level >= 2 the pooled macro-action. The target is detached by the official loss.
- **Action-SIGReg.** The official ``loss.SIGReg()`` on the macro-action embeddings of every level >= 2
  (``sigreg_action_level{l}``), weight 0.005, added by the vendored ``hjepa_forward``.
- **Why post-construction.** ``create_world_model`` reads the IDM output size from ``predictor.action_dim``,
  which only the DROID ``causal_transformer`` predictor accepts; the #123 arms use ``ARPredictor`` (which takes
  no ``action_dim``). The heads are therefore built by the official classes with the official arguments
  (``state_dim = D``, ``hidden_dim = 256``, ``action_dim`` = the action-embedding width D) after the #123
  model, in level order, and the action-SIGReg modules are added to the loss modules as ``create_world_model``
  adds them. With the same seed the shared parameters start exactly where the #123 arm starts.
- **Weights.** gamma_2 = gamma_3 = 50, action-SIGReg 0.005 (DROID); gamma_1 in {100, 225} selected once per input
  type. LeWM arms have one level: IDM at level 1 only (the DROID LeWM+IDM baseline), no action-SIGReg.
- **Matched carriers.** HJ-car-m-IDM and LeWM-car-m-IDM count the IDM heads: #123's matched sizes with only the
  predictor MLP width reduced (uniformly over levels) to the value whose total, heads included, is nearest the
  Gate-B unit (1 837 690, +-2 %).

The IDM arms are registered in ``novphy.ARMS`` / ``novphy.SIZES`` (in memory; novphy.py is not edited) so that the
#123 loader, geometry, readouts and scoring code handle them unchanged. Their ``model.pt`` uses #123's model
schema with the backbone state (no IDM heads) under ``state_dict``, so #123's ``load_trained`` builds and loads
them; the training-only heads are stored beside it under ``idm_heads``.
"""
from __future__ import annotations

from . import novphy as hn
from .vendor.loss import InverseDynamicsLoss, InverseDynamicsModel, SIGReg

IDM_HIDDEN = 256
GAMMA_UPPER = 50.0
ACTION_SIGREG = 0.005
GAMMA1_GRID = (100.0, 225.0)
GAMMA1_TIE = 100.0  # the DROID value
HEAD_PREFIX = "idm_loss_fn."
GATE_B_UNIT_PARAMETERS = 1837690
PARAMETER_TOLERANCE = 0.02

# #123's matched sizes, predictor MLP width reduced so the total with the IDM heads is nearest the Gate-B unit
SIZES = {
    "matched-idm": {**hn.SIZES["matched"],
                    "level1_predictor": {**hn.SIZES["matched"]["level1_predictor"], "mlp_dim": 96},
                    "upper_predictor": {**hn.SIZES["matched"]["upper_predictor"], "mlp_dim": 96}},
    "matched-lewm-idm": {**hn.SIZES["matched-lewm"],
                         "level1_predictor": {**hn.SIZES["matched-lewm"]["level1_predictor"], "mlp_dim": 384}},
}
# IDM arm -> (#123 arm it extends, size, role)
BASES = {
    "HJ-px-IDM": ("HJ-px", "paper", "primary, faithful + remedy"),
    "HJ-car-m-IDM": ("HJ-car-m", "matched-idm", "primary, controlled + remedy"),
    "LeWM-px-IDM": ("LeWM-px", "paper", "temporal lever under the remedy (pixels)"),
    "LeWM-car-m-IDM": ("LeWM-car-m", "matched-lewm-idm", "temporal lever under the remedy (carriers)"),
}
ARMS = tuple(BASES)
PRIMARY = ("HJ-px-IDM", "HJ-car-m-IDM")


def register():
    """Add the IDM arms and sizes to novphy's tables (idempotent; in memory only)."""
    hn.SIZES.update(SIZES)
    for arm, (base, size, role) in BASES.items():
        hn.ARMS[arm] = {**hn.ARMS[base], "size": size, "role": role}


register()


def coefficients(arm, gamma1):
    """{level: (idm_coeff, action_sigreg_coeff)} of an IDM arm."""
    return {level: (float(gamma1) if level == 1 else GAMMA_UPPER, 0.0 if level == 1 else ACTION_SIGREG)
            for level in range(1, hn.num_levels(arm) + 1)}


def config(arm, lam, gamma1):
    """#123's arm config plus the official per-level IDM keys read by hjepa_forward (idm_coeff, idm.hidden_dim,
    action_sigreg_coeff)."""
    cfg = hn.arm_config(arm, lam)
    for level, (idm, action) in coefficients(arm, gamma1).items():
        level_cfg = cfg[f"level{level}"]
        level_cfg["idm_coeff"] = idm
        level_cfg["idm"] = hn.Cfg({"hidden_dim": IDM_HIDDEN})
        if action > 0:
            level_cfg["action_sigreg_coeff"] = action
    return cfg


def head_dims(model, level):
    """(state_dim, action_dim) of a level's IDM head: the embedding width and the action-embedding width (the
    Embedder's last Linear; create_world_model sets its emb_dim to the embedding width)."""
    return int(model.embed_dims[level]), int(model.get_level(level).action_embed.embed[-1].out_features)


def attach(model, losses, cfg):
    """Attach the official IDM heads and action-SIGReg modules as create_world_model does (level order)."""
    for level in range(1, int(cfg.num_levels) + 1):
        level_cfg = cfg[f"level{level}"]
        if float(level_cfg.get("idm_coeff", 0.0)) > 0:
            state_dim, action_dim = head_dims(model, level)
            model.get_level(level).idm_loss_fn = InverseDynamicsLoss(
                InverseDynamicsModel(state_dim=state_dim, hidden_dim=int(level_cfg.idm.hidden_dim),
                                     action_dim=action_dim))
    for level in range(1, int(cfg.num_levels) + 1):
        if float(cfg[f"level{level}"].get("action_sigreg_coeff", 0.0)) > 0:
            losses[f"sigreg_action_level{level}"] = SIGReg()
    return model, losses


def build_model(arm, lam, gamma1, checkpointing=None):
    """(PlannedHJEPA with IDM heads, loss modules incl. action-SIGReg, cfg)."""
    cfg = config(arm, lam, gamma1)
    model, losses = hn.build_model(arm, lam, checkpointing=checkpointing)
    attach(model, losses, cfg)
    return model, losses, cfg


def is_head(key):
    return HEAD_PREFIX in key


def split_state(state):
    """(backbone state without the IDM heads, IDM head state) of a model state_dict."""
    return ({k: v for k, v in state.items() if not is_head(k)}, {k: v for k, v in state.items() if is_head(k)})


def parameter_counts(arm, lam=hn.SIGREG_LAMBDAS[1], gamma1=GAMMA1_GRID[0]):
    """{backbone, idm_heads, total} parameter counts of an IDM arm."""
    model, _, _ = build_model(arm, lam, gamma1, checkpointing=False)
    heads = sum(p.numel() for name, p in model.named_parameters() if is_head(name))
    total = sum(p.numel() for p in model.parameters())
    return {"backbone": total - heads, "idm_heads": heads, "total": total}


def matched(total):
    return abs(total - GATE_B_UNIT_PARAMETERS) <= PARAMETER_TOLERANCE * GATE_B_UNIT_PARAMETERS
