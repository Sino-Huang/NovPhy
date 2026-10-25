"""Issue #135 G-port: the official DDLP (taldatech/ddlp @ 5686275, run from the read-only clone) vs the NovPhy adapter
(``world_model/baselines/ddlp``), CPU, float32, the same torch build on both sides (env ``b127-ddlp``, torch 2.13).

Two modes, separate processes, one fixed set of cases (``CASES``):

    python scripts/b127_ddlp_port_reference.py reference [CASE ...]   # official side -> REF_DIR/<case>.pt
    python scripts/b127_ddlp_port_reference.py compare   [CASE ...]   # adapter side + comparison -> gport/gport.json

Official side (imports ``models``, ``modules``, ``utils`` from the clone; never the vendored copy). Per case the model
is the official ``ObjectDynamicsDLP`` built with ``train_ddlp.py``'s constructor call from an official config, the
optimizer / scheduler / flags / dynamics-discount lines are ``train_ddlp.py``'s, the loss is ``calc_elbo``. Steps
(``STEPS``): epoch 0 (warm-up + noisy: object-glimpse reconstruction, warm-up priors), epoch 1 (noisy), epoch 2
(regular, dynamics ramp part-way); the StepLR scheduler steps at every epoch boundary, so the three steps run at
lr 2e-4, then gamma x, then gamma^2 x. Then a deterministic rollout: ``model.sample(x, cond_steps=4, num_steps=6,
return_z=True)`` from a static 4-frame context (the NovPhy layout).

``novphy-launch``: the NovPhy model (native config + the launch particle). The official side builds the official
model, replaces its PINT by an official ``DynamicsDLP`` with ``max_particles = K + 2`` and wraps it with an
independently written launch-particle wrapper (``RefLaunch`` below: append the particle, strip it from every output,
re-set it after every rollout step), so the comparison checks the adapter's wrapper and every official component.

Inputs are fixed seeded uniform frames [B, T + 1, 3, 128, 128] and launches; seeds are reset before construction and
before every step on both sides (posterior sampling and PINT dropout 0.1 draw from torch's global generator).

Gate (``issue_123_exact``, #123 ``run_hjepa_comparison.GPORT_TOLERANCE``): per step loss and every loss term relative
delta <= 1e-4, gradient norm relative delta <= 1e-4, per-tensor gradient relative L2 <= 1e-3 (tensors above 1e-6 of
the global norm), the share of trainable elements with |delta param| <= 1e-7 >= 0.999 and max |delta param| <=
2 lr + 1e-7, Adam moments, learning rates, parameter names / counts / initial values equal; the rollout particles and
the forward outputs equal within 1e-5 relative L2.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DDLP_REF = Path(os.environ.get("DDLP_REF", "/mnt/array/sukaih/novphy-baselines-src/ddlp@5686275"))
OUT = ROOT / ".local-artifacts/issue-135-ddlp-v1"
GPORT = OUT / "gport"
REF_DIR = Path(os.environ.get("B135_GPORT_REF", Path.home() / ".cache/novphy-issue-135-gport"))
EXTERNAL = ROOT / "data/external/ddlp/checkpoints"
SEED = 20260908
THREADS = 4
ITER_PER_EPOCH = 20_000
STEPS = [{"epoch": 0, "iteration": 0}, {"epoch": 1, "iteration": 20_003}, {"epoch": 2, "iteration": 45_000}]
ROLLOUT_STEPS = 6
STATIC = 4
CASES = {
    "phyre-released": {"config": "released:ddlp-phyre", "launch": False, "batch": 2,
                       "why": "the released PHYRE-128 checkpoint's hparams.json (paper Table 4; the NovPhy native shape)"},
    "phyre-config": {"config": "clone:configs/phyre.json", "launch": False, "batch": 2,
                     "why": "the repository's configs/phyre.json (train_ddlp.py -d phyre)"},
    "clevrer-released": {"config": "released:ddlp-clevrer", "launch": False, "batch": 1,
                         "why": "the released CLEVRER-128 checkpoint's hparams.json (VGG perceptual loss, K = 12)"},
    "novphy-launch": {"config": "released:ddlp-phyre", "launch": True, "batch": 2,
                      "why": "the NovPhy model: native shape + the launch particle (Decision P-other adapter)"},
    "phyre-released-static": {"config": "released:ddlp-phyre", "launch": False, "batch": 2, "static": True,
                              "why": "train_ddlp.py's static branch (epoch < start_dyn_epoch: forward_dyn False, "
                                     "calc_elbo_static): the #135 readiness model R0"},
}
TOL = {"loss_relative": 1e-4, "grad_norm_relative": 1e-4, "grad_tensor_relative_l2": 1e-3,
       "grad_floor_of_global_norm": 1e-6, "param_close_share": 0.999, "param_close_abs": 1e-7,
       "param_max_abs_over_lr": 2.0, "lr_abs": 1e-12, "output_relative_l2": 1e-5}
FORWARD_KEYS = ("rec", "z", "z_features", "z_bg", "obj_on", "mu_dyn", "logvar_dyn", "mu_features_dyn",
                "obj_on_a_dyn", "mu_bg_dyn")


def load_config(spec: str) -> dict:
    kind, name = spec.split(":", 1)
    path = (EXTERNAL / name / "hparams.json") if kind == "released" else (DDLP_REF / name)
    return json.loads(path.read_text())


def config_path(spec: str) -> Path:
    kind, name = spec.split(":", 1)
    return (EXTERNAL / name / "hparams.json") if kind == "released" else (DDLP_REF / name)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def inputs(case: dict, cfg: dict) -> dict:
    g = torch.Generator().manual_seed(SEED)
    b, t = case["batch"], cfg["timestep_horizon"] + 1
    return {"x": torch.rand(b, t, cfg["ch"], cfg["image_size"], cfg["image_size"], generator=g),
            "launch5": torch.rand(b, 5, generator=g),
            "frames0": torch.rand(b, cfg["ch"], cfg["image_size"], cfg["image_size"], generator=g)}


# ---------------------------------------------------------------------------------------------------------------
# official side
# ---------------------------------------------------------------------------------------------------------------

def official_model(cfg: dict):
    """train_ddlp.py lines 115-125 (the constructor call), verbatim argument mapping."""
    from models import ObjectDynamicsDLP
    c = cfg
    return ObjectDynamicsDLP(cdim=c['ch'], enc_channels=c['enc_channels'], prior_channels=c['prior_channels'],
                             image_size=c['image_size'], n_kp=c['n_kp'], learned_feature_dim=c['learned_feature_dim'],
                             pad_mode=c['pad_mode'], sigma=c['sigma'],
                             dropout=c['dropout'], patch_size=c['patch_size'], n_kp_enc=c['n_kp_enc'],
                             n_kp_prior=c['n_kp_prior'], kp_range=c['kp_range'], kp_activation=c['kp_activation'],
                             anchor_s=c['anchor_s'], use_resblock=c['use_resblock'],
                             timestep_horizon=c['timestep_horizon'], predict_delta=c['predict_delta'],
                             scale_std=c['scale_std'], offset_std=c['offset_std'], obj_on_alpha=c['obj_on_alpha'],
                             obj_on_beta=c['obj_on_beta'], pint_layers=c['pint_layers'], pint_heads=c['pint_heads'],
                             pint_dim=c['pint_dim'], use_correlation_heatmaps=c['use_correlation_heatmaps'],
                             enable_enc_attn=c['enable_enc_attn'], filtering_heuristic=c['filtering_heuristic'])


class RefLaunch(torch.nn.Module):
    """Reference launch-particle wrapper (written independently of the adapter)."""

    def __init__(self, dyn, n_kp, feat_dim):
        super().__init__()
        self.dyn = dyn
        self.launch_map = torch.nn.Linear(5, 6 + feat_dim)
        self.k, self.d = n_kp, feat_dim
        self.launch5 = None

    def _attrs(self):
        a = self.launch_map(self.launch5)  # [B, 6 + d]
        return a[:, 0:2], a[:, 2:4], a[:, 4], a[:, 5:6], a[:, 6:]

    def _append(self, z, s, o, d, f):
        xy, sc, on, dp, ft = self._attrs()
        t = z.shape[1]
        rep = lambda v: v[:, None, None].repeat(1, t, 1, 1)  # noqa: E731
        return (torch.cat([z, rep(xy)], 2), torch.cat([s, rep(sc)], 2), torch.cat([o, on[:, None, None].repeat(1, t, 1)], 2),
                torch.cat([d, rep(dp)], 2), torch.cat([f, rep(ft)], 2))

    def forward(self, z, s, o, d, f, bg):
        out = self.dyn(*self._append(z, s, o, d, f), bg)
        drop = {"mu_bg_features", "logvar_bg_features"}
        return {key: (v if key in drop else v.narrow(2, 0, self.k)) for key, v in out.items()}

    def sample(self, z, s, o, d, f, bg, steps=10, deterministic=False):
        z, s, o, d, f = self._append(z, s, o, d, f)
        xy, sc, on, dp, ft = self._attrs()
        for _ in range(steps):
            z, s, o, d, f, bg = self.dyn.sample(z, s, o, d, f, bg, steps=1, deterministic=deterministic)
            z, s, o, d, f = z.clone(), s.clone(), o.clone(), d.clone(), f.clone()
            z[:, -1, self.k], s[:, -1, self.k], o[:, -1, self.k] = xy, sc, on
            d[:, -1, self.k], f[:, -1, self.k] = dp, ft
        k = self.k
        return z[:, :, :k], s[:, :, :k], o[:, :, :k], d[:, :, :k], f[:, :, :k], bg


def reference(case_names) -> int:
    sys.path.insert(0, str(DDLP_REF))
    import torch.optim as optim
    from modules.dynamics_modules import DynamicsDLP
    from utils.loss_functions import calc_reconstruction_loss, VGGDistance
    torch.set_num_threads(THREADS)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    for name in case_names:
        case = CASES[name]
        cfg = load_config(case["config"])
        data = inputs(case, cfg)
        device = torch.device("cpu")
        torch.manual_seed(SEED)
        model = official_model(cfg)
        if case["launch"]:
            dyn = DynamicsDLP(model.learned_feature_dim, model.bg_learned_feature_dim, hidden_dim=model.pint_dim,
                              projection_dim=model.pint_dim, n_head=model.pint_heads, n_layer=model.pint_layers,
                              block_size=model.timestep_horizon, kp_activation=model.kp_activation,
                              predict_delta=model.predict_delta, max_delta=1.0, positional_bias=True,
                              max_particles=model.n_kp_enc + 2)
            model.dyn_module = RefLaunch(dyn, model.n_kp_enc, model.learned_feature_dim)
        init = {k: v.detach().clone() for k, v in model.state_dict().items()}
        # train_ddlp.py: optimizer, scheduler, loss function, dynamics-warm-up bookkeeping
        recon_loss_type = cfg['recon_loss_type']
        recon_loss_func = VGGDistance(device=device) if recon_loss_type == "vgg" else calc_reconstruction_loss
        optimizer = optim.Adam(model.get_parameters(), lr=cfg['lr'], betas=cfg['adam_betas'], eps=cfg['adam_eps'],
                               weight_decay=cfg['weight_decay'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg['scheduler_gamma'])
        timestep_horizon, warmup_epoch = cfg['timestep_horizon'], cfg['warmup_epoch']
        iter_per_epoch = ITER_PER_EPOCH
        dynamics_warmup_iters = max(warmup_epoch, 1) * max(10_000, iter_per_epoch)
        iter_per_step = dynamics_warmup_iters // timestep_horizon
        max_iterations_per_step = [iter_per_step * (i + 1) for i in range(timestep_horizon)]
        names = {id(p): n for n, p in model.named_parameters()}
        steps, epoch_seen = [], STEPS[0]["epoch"]
        x = data["x"]
        for k, st in enumerate(STEPS):
            epoch, iteration = st["epoch"], st["iteration"]
            while epoch_seen < epoch:
                scheduler.step()
                epoch_seen += 1
            torch.manual_seed(SEED + 1 + k)
            model.train()
            x_prior = x
            noisy = (epoch < (warmup_epoch + 1))
            if case["launch"]:
                model.dyn_module.launch5 = data["launch5"]
            if case.get("static"):
                # train_ddlp.py lines 199-225, the epoch < start_epoch branch
                model_output = model(x, x_prior=x_prior, warmup=(epoch < warmup_epoch), noisy=noisy,
                                     bg_masks_from_fg=False, forward_dyn=False, train_enc_prior=cfg['train_enc_prior'],
                                     num_static_frames=cfg['num_static_frames'])
                discount_factor = 0.0
                discount = discount_factor * torch.ones(size=(timestep_horizon,), device=x.device)
                all_losses = model.calc_elbo_static(x, model_output, warmup=(epoch < warmup_epoch),
                                                    beta_kl=cfg['beta_kl'], beta_dyn=cfg['beta_dyn'],
                                                    beta_rec=cfg['beta_rec'], kl_balance=cfg['kl_balance'],
                                                    dynamic_discount=discount, recon_loss_type=recon_loss_type,
                                                    recon_loss_func=recon_loss_func, noisy=noisy)
            else:
                model_output = model(x, x_prior=x_prior, warmup=(epoch < warmup_epoch), noisy=noisy,
                                     bg_masks_from_fg=False, forward_dyn=True, train_enc_prior=cfg['train_enc_prior'],
                                     num_static_frames=cfg['num_static_frames'])
                discount = iteration / torch.tensor(max_iterations_per_step, device=x.device)
                discount = discount.clamp(0.0, 1.0)
                all_losses = model.calc_elbo(x, model_output, warmup=(epoch < warmup_epoch), beta_kl=cfg['beta_kl'],
                                             beta_dyn=cfg['beta_dyn'], beta_rec=cfg['beta_rec'],
                                             kl_balance=cfg['kl_balance'], dynamic_discount=discount,
                                             recon_loss_type=recon_loss_type, recon_loss_func=recon_loss_func,
                                             beta_dyn_rec=cfg['beta_dyn_rec'], num_static=cfg['num_static_frames'],
                                             noisy=noisy)
            loss = all_losses['loss']
            lrs = [g["lr"] for g in optimizer.param_groups]
            optimizer.zero_grad()
            loss.backward()
            grads = {names[id(p)]: p.grad.detach().clone() for p in model.parameters() if p.grad is not None}
            optimizer.step()
            moments = {names[id(p)]: (s["exp_avg"].clone(), s["exp_avg_sq"].clone())
                       for p, s in optimizer.state.items()}
            steps.append({"losses": {kk: float(v.detach()) for kk, v in all_losses.items() if torch.is_tensor(v)},
                          "grads": grads, "lrs": lrs, "moments": moments,
                          "params": {kk: v.detach().clone() for kk, v in model.state_dict().items()},
                          "forward": {kk: model_output[kk].detach().clone() for kk in FORWARD_KEYS
                                      if torch.is_tensor(model_output.get(kk))}})
        if case["launch"]:
            model.dyn_module.launch5 = None
        model.eval()
        frames0 = data["frames0"]
        xs = torch.cat([frames0[:, None].repeat(1, STATIC, 1, 1, 1),
                        torch.zeros(frames0.shape[0], ROLLOUT_STEPS, *frames0.shape[1:])], 1)
        if case["launch"]:
            model.dyn_module.launch5 = data["launch5"]
        with torch.no_grad():
            _, z_out = model.sample(xs, num_steps=ROLLOUT_STEPS, deterministic=True, cond_steps=STATIC, return_z=True)
        torch.save({"case": name, "torch": torch.__version__, "init": init, "steps": steps,
                    "rollout": {k: v.clone() for k, v in z_out.items()},
                    "optimizer": {"class": type(optimizer).__name__, "defaults": dict(optimizer.defaults)},
                    "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "config_sha256": sha256(config_path(case["config"])), "written_at": time.time()},
                   REF_DIR / f"{name}.pt")
        print(f"reference {name}: losses {[s['losses']['loss'] for s in steps]}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------------------
# adapter side + comparison
# ---------------------------------------------------------------------------------------------------------------

def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    den = float(b.norm())
    return float((a - b).norm()) / den if den > 0 else float((a - b).norm())


def compare_case(name: str) -> dict:
    from world_model.baselines.ddlp import novphy as dd
    case = CASES[name]
    cfg = load_config(case["config"])
    ref = torch.load(REF_DIR / f"{name}.pt", map_location="cpu", weights_only=False)
    data = inputs(case, cfg)
    model = dd.build_model(cfg, launch=case["launch"], seed=SEED)
    sd = model.state_dict()
    init_equal = (set(sd) == set(ref["init"])) and all(torch.equal(sd[k], ref["init"][k]) for k in sd)
    optimizer, scheduler = dd.build_optimizer(model, cfg)
    recon = dd.recon_loss_func(cfg)
    names = {id(p): n for n, p in model.named_parameters()}
    out_steps, epoch_seen, ok = [], STEPS[0]["epoch"], init_equal
    for k, st in enumerate(STEPS):
        while epoch_seen < st["epoch"]:
            scheduler.step()
            epoch_seen += 1
        torch.manual_seed(SEED + 1 + k)
        model.train()
        lrs = [g["lr"] for g in optimizer.param_groups]
        if case.get("static"):
            out, all_losses = dd.losses_static(model, data["x"], st["epoch"], cfg, recon)
        else:
            out, all_losses = dd.losses(model, data["x"], data["launch5"] if case["launch"] else None, st["epoch"],
                                        st["iteration"], ITER_PER_EPOCH, cfg, recon)
        optimizer.zero_grad()
        all_losses["loss"].backward()
        grads = {names[id(p)]: p.grad.detach().clone() for p in model.parameters() if p.grad is not None}
        optimizer.step()
        r = ref["steps"][k]
        terms = {kk: float(v.detach()) for kk, v in all_losses.items() if torch.is_tensor(v)}
        loss_rel = max(abs(terms[kk] - v) / max(1.0, abs(v)) for kk, v in r["losses"].items())
        gnorm = float(torch.sqrt(sum((g.double() ** 2).sum() for g in grads.values())))
        rnorm = float(torch.sqrt(sum((g.double() ** 2).sum() for g in r["grads"].values())))
        floor = TOL["grad_floor_of_global_norm"] * rnorm
        grad_rel = max([rel(grads[kk], v) for kk, v in r["grads"].items() if float(v.norm()) > floor] or [0.0])
        params = model.state_dict()
        deltas = torch.cat([(params[kk].double() - v.double()).abs().flatten() for kk, v in r["params"].items()
                            if v.is_floating_point()])
        close_share = float((deltas <= TOL["param_close_abs"]).double().mean())
        lr_max = max(lrs)
        moments = {names[id(p)]: (s["exp_avg"], s["exp_avg_sq"]) for p, s in optimizer.state.items()}
        moment_rel = max(max(rel(moments[kk][0], v[0]), rel(moments[kk][1], v[1])) for kk, v in r["moments"].items())
        fwd = {kk: rel(out[kk].detach(), v) for kk, v in r["forward"].items()}
        checks = {
            "loss_terms": loss_rel <= TOL["loss_relative"],
            "grad_norm": abs(gnorm - rnorm) / max(rnorm, 1e-30) <= TOL["grad_norm_relative"],
            "grad_tensors": grad_rel <= TOL["grad_tensor_relative_l2"] and set(grads) == set(r["grads"]),
            "param_share": close_share >= TOL["param_close_share"],
            "param_max": float(deltas.max()) <= TOL["param_max_abs_over_lr"] * lr_max + TOL["param_close_abs"],
            "lrs": all(abs(a - b) <= TOL["lr_abs"] for a, b in zip(lrs, r["lrs"])),
            "moments": moment_rel <= TOL["grad_tensor_relative_l2"],
            "forward": all(v <= TOL["output_relative_l2"] for v in fwd.values()),
        }
        step_ok = all(checks.values())
        ok = ok and step_ok
        out_steps.append({"epoch": st["epoch"], "iteration": st["iteration"], "phase": dd.phase(st["epoch"], cfg),
                          "lrs": lrs, "official_lrs": r["lrs"], "loss": terms["loss"],
                          "official_loss": r["losses"]["loss"], "loss_terms": terms, "loss_terms_max_relative_delta": loss_rel,
                          "grad_norm": gnorm, "official_grad_norm": rnorm, "grad_tensor_max_relative_l2": grad_rel,
                          "param_close_share_abs": close_share, "param_max_abs_delta": float(deltas.max()),
                          "adam_moment_max_relative_l2": moment_rel, "forward_relative_l2": fwd,
                          "checks": checks, "pass": step_ok})
    model.eval()
    ro = dd.rollout(model, data["frames0"], data["launch5"] if case["launch"] else None, steps=ROLLOUT_STEPS)
    rmap = {"z": "z_pos", "z_scale": "z_scale", "z_obj_on": "z_obj_on", "z_depth": "z_depth",
            "z_features": "z_features", "z_bg": "z_bg_features"}
    roll = {k: rel(ro[k], ref["rollout"][v]) for k, v in rmap.items()}
    roll_ok = all(v <= TOL["output_relative_l2"] for v in roll.values())
    ok = ok and roll_ok
    return {"case": name, "config": case["config"], "config_sha256": sha256(config_path(case["config"])),
            "why": case["why"], "launch": case["launch"], "batch": case["batch"],
            "trainable_parameters": dd.trainable_parameters(model), "official_trainable_parameters": ref["trainable"],
            "initial_state_equal": init_equal, "optimizer_class": type(optimizer).__name__,
            "official_optimizer_class": ref["optimizer"]["class"],
            "optimizer_defaults_equal": {k: v for k, v in optimizer.defaults.items()} == ref["optimizer"]["defaults"],
            "steps": out_steps, "rollout_relative_l2": roll, "rollout_pass": roll_ok,
            "torch": torch.__version__, "official_torch": ref["torch"], "pass": bool(ok)}


def compare(case_names) -> int:
    torch.set_num_threads(THREADS)
    sys.path.insert(0, str(ROOT))
    path = GPORT / "gport.json"
    record = json.loads(path.read_text()) if path.is_file() else {"cases": {}}
    for name in case_names:
        record["cases"][name] = compare_case(name)
        print(f"compare {name}: pass {record['cases'][name]['pass']}", flush=True)
    vendor = ROOT / "world_model/baselines/ddlp/vendor/VENDOR.json"
    record.update({
        "schema": "issue_127_gport_v1", "issue": 135, "arm_family": "T2 DDLP",
        "commit": "5686275800ee58d9832714af5ed8363c4fb606bc", "official_root": str(DDLP_REF),
        "adapter": sha256(ROOT / "world_model/baselines/ddlp/novphy.py"), "vendor_json": sha256(vendor),
        "runner": sha256(Path(__file__)), "env": "b127-ddlp (both sides; torch " + torch.__version__ + ", CPU, float32, "
                                                 f"{THREADS} threads)",
        "G_port_rule": "issue_123_exact", "tolerance": TOL, "steps_spec": STEPS, "iter_per_epoch": ITER_PER_EPOCH,
        "procedure": __doc__, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    record["G_port"] = bool(record["cases"]) and all(c["pass"] for c in record["cases"].values()) and \
        set(record["cases"]) == set(CASES)
    record["pass"] = record["G_port"]
    GPORT.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    print(json.dumps({"G_port": record["G_port"], "cases": {k: v["pass"] for k, v in record["cases"].items()}}))
    return 0 if record["G_port"] else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("reference", "compare"):
        raise SystemExit(__doc__)
    names = argv[1:] or list(CASES)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    return reference(names) if argv[0] == "reference" else compare(names)


if __name__ == "__main__":
    sys.exit(main())
