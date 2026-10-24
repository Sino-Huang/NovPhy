"""Issue-130 G-port: the official DINO-WM world model vs the NovPhy adapter on fixed CPU float32 inputs.

    --reference OUT   (``b127-dino_wm`` env; the official code at DINO_WM_REF @ 0a9492f, torch 2.7.1)
        Builds the official PushT configuration from the released ``outputs/pusht/hydra.yaml`` exactly as
        ``Trainer.init_models`` does (``hydra.utils.instantiate`` of the encoder, proprio / action encoders, ViT
        predictor and ``VWorldModel``; ``has_decoder`` False, the official predictor-only option: the decoder loss
        only sees ``z.detach()``), the optimizers with the official ``Trainer.init_optimizers``, and takes two
        steps of ``Trainer.train``'s body (forward, zero_grad, backward, predictor and action-encoder steps) on the
        same fixed batch. DINOv2 (frozen) runs inside the official forward. Saves reference.pt.
        CPU-only shim: upstream ``Attention.__init__`` moves the causal mask to 'cuda'; ``generate_mask_matrix`` is
        wrapped so that ``.to('cuda')`` returns the same CPU tensor (values unchanged).
    --compare OUT     (``novphy`` env) Rebuilds the adapter's official configuration
        (``novphy.official_pusht_model``), loads the reference initial state, computes the DINOv2 patch features with
        the adapter's encoder path, takes the same two steps with ``novphy.make_optimizers`` / ``train_step`` and
        writes OUT/gport.json (schema issue_127_gport_v1).

DINO-WM has no LR scheduler: the step-0 LR is already the positive base LR (5e-4); both steps run at it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

import torch

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get("DINO_WM_REF", "/mnt/array/sukaih/novphy-baselines-src/dino_wm@0a9492f"))
EXTERNAL = ROOT / "data/external/dino_wm"
HYDRA_YAML = EXTERNAL / "checkpoints/outputs/pusht/hydra.yaml"
SEED_INIT, SEED_INPUT, SEED_STEP, SEED_STEP2 = 20261024, 7, 11, 13
BATCH, FRAMES = 2, 4  # num_hist 3 + num_pred 1
TOLERANCE = {"loss_relative": 1e-4, "grad_norm_relative": 1e-4, "grad_tensor_relative_l2": 1e-3,
             "grad_floor_of_global_norm": 1e-6, "feature_max_abs": 1e-4, "param_close_share": 0.999,
             "param_close_abs": 1e-7, "param_max_abs_over_lr": 2.0, "moment_relative_l2": 1e-3}
SCHEMA = "issue_127_gport_v1"


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_inputs():
    g = torch.Generator().manual_seed(SEED_INPUT)
    # smooth images in [-1, 1] (the official dataset transform's range), proprio / actions as normalized data
    base = torch.rand(BATCH, FRAMES, 3, 28, 28, generator=g) * 2 - 1
    visual = torch.nn.functional.interpolate(base.flatten(0, 1), size=(224, 224), mode="bilinear",
                                             align_corners=False).view(BATCH, FRAMES, 3, 224, 224)
    proprio = torch.randn(BATCH, FRAMES, 4, generator=g)
    act = torch.randn(BATCH, FRAMES, 10, generator=g)
    return {"visual": visual.contiguous(), "proprio": proprio, "act": act}


def named_state(modules):
    return {f"{name}.{k}": v.detach().clone() for name, m in modules.items() for k, v in m.state_dict().items()}


def named_grads(modules):
    return {f"{name}.{k}": p.grad.detach().clone() for name, m in modules.items()
            for k, p in m.named_parameters() if p.grad is not None}


def optimizer_state(optimizers, modules):
    names = {id(p): f"{name}.{k}" for name, m in modules.items() for k, p in m.named_parameters()}
    out = {}
    for opt in optimizers:
        for p, s in opt.state.items():
            out[names[id(p)]] = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in s.items()}
    return out


# ---------------------------------------------------------------------------
# official side (b127-dino_wm env)
# ---------------------------------------------------------------------------

def reference(out_dir):
    import hydra
    from omegaconf import OmegaConf
    os.environ["TORCH_HOME"] = str(EXTERNAL / "torch_hub")
    sys.path.insert(0, str(OFFICIAL))
    os.chdir(OFFICIAL)
    torch.set_num_threads(8)
    from models import vit

    class _CpuMask:
        def __init__(self, t):
            self.t = t

        def to(self, *args, **kwargs):
            return self.t

    original_mask = vit.generate_mask_matrix
    vit.generate_mask_matrix = lambda npatch, nwindow: _CpuMask(original_mask(npatch, nwindow))
    cfg = OmegaConf.load(HYDRA_YAML)
    OmegaConf.set_struct(cfg, False)
    cfg.has_decoder = False
    encoder = hydra.utils.instantiate(cfg.encoder)
    for p in encoder.parameters():
        p.requires_grad = False
    torch.manual_seed(SEED_INIT)
    proprio_encoder = hydra.utils.instantiate(cfg.proprio_encoder, in_chans=4, emb_dim=cfg.proprio_emb_dim)
    action_encoder = hydra.utils.instantiate(cfg.action_encoder, in_chans=10, emb_dim=cfg.action_emb_dim)
    num_patches = (cfg.img_size // 16) ** 2
    predictor = hydra.utils.instantiate(cfg.predictor, num_patches=num_patches, num_frames=cfg.num_hist,
                                        dim=encoder.emb_dim + (cfg.proprio_emb_dim * cfg.num_proprio_repeat
                                                               + cfg.action_emb_dim * cfg.num_action_repeat)
                                        * cfg.concat_dim)
    model = hydra.utils.instantiate(cfg.model, encoder=encoder, proprio_encoder=proprio_encoder,
                                    action_encoder=action_encoder, predictor=predictor, decoder=None,
                                    proprio_dim=cfg.proprio_emb_dim, action_dim=cfg.action_emb_dim,
                                    concat_dim=cfg.concat_dim, num_action_repeat=cfg.num_action_repeat,
                                    num_proprio_repeat=cfg.num_proprio_repeat)
    modules = {"predictor": predictor, "action_encoder": action_encoder, "proprio_encoder": proprio_encoder}
    init_state = named_state(modules)
    import train as official_train  # Trainer.init_optimizers, run on a namespace with the trainer's attributes
    trainer = types.SimpleNamespace(encoder=encoder, predictor=predictor, action_encoder=action_encoder,
                                    proprio_encoder=proprio_encoder, decoder=None, cfg=cfg,
                                    accelerator=types.SimpleNamespace(prepare=lambda x: x))
    official_train.Trainer.init_optimizers(trainer)
    optimizers = [trainer.predictor_optimizer, trainer.action_encoder_optimizer]
    inputs = fixed_inputs()
    obs = {"visual": inputs["visual"], "proprio": inputs["proprio"]}
    with torch.no_grad():
        features = model.encode_obs(obs)["visual"].clone()
    steps = []
    for seed in (SEED_STEP, SEED_STEP2):
        model.train()
        torch.manual_seed(seed)
        z_out, _, _, loss, components = model(obs, inputs["act"])
        trainer.encoder_optimizer.zero_grad()
        trainer.predictor_optimizer.zero_grad()
        trainer.action_encoder_optimizer.zero_grad()
        loss.backward()
        grads = named_grads(modules)
        trainer.predictor_optimizer.step()
        trainer.action_encoder_optimizer.step()
        steps.append({"seed": seed, "scalars": {k: float(v) for k, v in components.items()},
                      "z_pred": z_out.detach().clone(), "grads": grads,
                      "grad_norm": float(torch.cat([g.flatten() for g in grads.values()]).norm()),
                      "post_state": named_state(modules), "optimizer_state": optimizer_state(optimizers, modules),
                      "lrs": [g["lr"] for opt in optimizers for g in opt.param_groups]})
    defaults = {name: {k: opt.defaults[k] for k in ("lr", "betas", "eps", "weight_decay", "amsgrad")}
                for name, opt in (("predictor", optimizers[0]), ("action_encoder", optimizers[1]))}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "issue_130_gport_reference_v1", "torch": torch.__version__, "official": str(OFFICIAL),
                "hydra_yaml_sha256": sha256_of(HYDRA_YAML), "seeds": [SEED_INIT, SEED_INPUT, SEED_STEP, SEED_STEP2],
                "inputs": inputs, "features": features, "init_state": init_state, "steps": steps,
                "optimizer_defaults": defaults, "encoder_optimizer": type(trainer.encoder_optimizer).__name__,
                "made_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, out_dir / "reference.pt")
    print(f"[issue-130] G-port reference: losses {[s['scalars']['loss'] for s in steps]}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# adapter side (novphy env)
# ---------------------------------------------------------------------------

def rel_l2(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def tensor_rule(mine, theirs, floor):
    above = [rel_l2(mine[k], v) for k, v in theirs.items() if float(v.norm()) > floor]
    below = [float((mine[k] - v).norm()) for k, v in theirs.items() if float(v.norm()) <= floor]
    return max(above or [0.0]), max(below or [0.0]), len(below)


def compare_step(ref, previous_state, model, modules, optimizers, features, inputs, opt_names):
    from world_model.baselines.dino_wm import novphy as dn
    tol = TOLERANCE
    torch.manual_seed(ref["seed"])
    z_pred, loss, components = dn.train_step(model, optimizers, features, inputs["act"], inputs["proprio"])
    grads = {f"{name}.{k}": p.grad.detach().clone() for name, m in modules.items()
             for k, p in m.named_parameters() if p.grad is not None}
    grad_norm = float(torch.cat([g.flatten() for g in grads.values()]).norm())
    scalars = {k: float(v.detach()) for k, v in components.items()}
    loss_rel = max(abs(scalars[k] - v) / max(1.0, abs(v)) for k, v in ref["scalars"].items())
    floor = tol["grad_floor_of_global_norm"] * ref["grad_norm"]
    grad_rel, grad_null, nulls = tensor_rule(grads, ref["grads"], floor)
    post = named_state(modules)
    deltas = torch.cat([(post[k] - v).abs().flatten() for k, v in ref["post_state"].items()])
    lr_max = max(ref["lrs"])
    state = optimizer_state(list(optimizers.values()), modules)
    moments = {}
    for key, fn in (("exp_avg", lambda t: t), ("exp_avg_sq", torch.sqrt)):
        mine = {k: fn(s[key]) for k, s in state.items()}
        theirs = {k: fn(s[key]) for k, s in ref["optimizer_state"].items()}
        m_floor = tol["grad_floor_of_global_norm"] * float(torch.cat([v.flatten() for v in theirs.values()]).norm())
        rel, null_abs, _ = tensor_rule(mine, theirs, m_floor) if set(mine) == set(theirs) else (float("inf"),) * 3
        moments[key] = {"max_relative_l2": rel, "null_max_abs_l2": null_abs,
                        "pass": bool(rel <= tol["moment_relative_l2"] and null_abs <= m_floor)}
    steps_equal = set(state) == set(ref["optimizer_state"]) and all(
        float(state[k]["step"]) == float(v["step"]) for k, v in ref["optimizer_state"].items())
    moved = float(torch.cat([(post[k] - previous_state[k]).abs().flatten() for k in post]).max())
    lrs = [g["lr"] for opt in optimizers.values() for g in opt.param_groups]
    out = {"seed": ref["seed"], "loss": scalars["loss"], "official_loss": ref["scalars"]["loss"],
           "loss_max_relative_delta": loss_rel, "z_pred_max_abs_delta": float((z_pred.detach() - ref["z_pred"]).abs().max()),
           "grad_norm": grad_norm, "official_grad_norm": ref["grad_norm"],
           "grad_norm_relative_delta": abs(grad_norm - ref["grad_norm"]) / ref["grad_norm"],
           "grad_tensor_max_relative_l2": grad_rel, "null_gradient_tensors": nulls, "null_gradient_max_abs_l2": grad_null,
           "grad_tensors": len(grads), "official_grad_tensors": len(ref["grads"]),
           "param_close_share": float((deltas <= tol["param_close_abs"]).float().mean()),
           "param_max_abs_delta": float(deltas.max()), "param_max_movement": moved,
           "lrs": lrs, "official_lrs": ref["lrs"], "optimizer_steps_equal": bool(steps_equal), "moments": moments,
           "optimizer_names": opt_names}
    out["pass"] = bool(set(grads) == set(ref["grads"]) and loss_rel <= tol["loss_relative"]
                       and out["grad_norm_relative_delta"] <= tol["grad_norm_relative"]
                       and grad_rel <= tol["grad_tensor_relative_l2"] and grad_null <= floor
                       and out["param_close_share"] >= tol["param_close_share"]
                       and out["param_max_abs_delta"] <= tol["param_max_abs_over_lr"] * lr_max + tol["param_close_abs"]
                       and lrs == ref["lrs"] and min(lrs) > 0 and steps_equal and moved > 0
                       and all(m["pass"] for m in moments.values()))
    return out, post


def compare(out_dir):
    from world_model.baselines.dino_wm import novphy as dn
    torch.set_num_threads(8)
    out_dir = Path(out_dir)
    ref = torch.load(out_dir / "reference.pt", map_location="cpu", weights_only=False)
    vendor = json.loads((ROOT / "world_model/baselines/dino_wm/vendor/VENDOR.json").read_text())
    hashes = {rel: "sha256:" + sha256_of(ROOT / "world_model/baselines/dino_wm/vendor" / rel) == e["vendored_sha256"]
              for rel, e in vendor["files"].items()}
    model = dn.official_pusht_model()
    modules = {"predictor": model.predictor, "action_encoder": model.action_encoder,
               "proprio_encoder": model.proprio_encoder}
    for name, m in modules.items():
        m.load_state_dict({k[len(name) + 1:]: v for k, v in ref["init_state"].items() if k.startswith(name + ".")})
    encoder = dn.load_dinov2("cpu")
    inputs = ref["inputs"]
    images = dn.official_encoder_input(inputs["visual"].flatten(0, 1))
    features = dn.patch_features(encoder, images).view(BATCH, FRAMES, -1, dn.FEATURE_DIM)
    feature_delta = float((features - ref["features"]).abs().max())
    optimizers = dn.make_optimizers(model)  # defaults only; each arm below builds its own
    defaults = {name: {k: opt.defaults[k] for k in ("lr", "betas", "eps", "weight_decay", "amsgrad")}
                for name, opt in optimizers.items()}
    defaults_equal = all(tuple(v) == tuple(ref["optimizer_defaults"][n][k]) if isinstance(v, tuple)
                         else v == ref["optimizer_defaults"][n][k] for n, d in defaults.items() for k, v in d.items())
    init = {k: v.clone() for k, v in named_state(modules).items()}
    arms = {}
    # "official-pusht": the ported predictor / loss / optimizer on the official encoder's features (the G-port);
    # "official-pusht-own-features": the same steps on the adapter's own DINOv2 features (end-to-end diagnostic;
    # the encoder is the same pinned DINOv2 code under another torch build, checked by feature_max_abs above)
    for arm, feats in (("official-pusht", ref["features"]), ("official-pusht-own-features", features)):
        for name, m in modules.items():
            m.load_state_dict({k[len(name) + 1:]: v for k, v in init.items() if k.startswith(name + ".")})
        optimizers = dn.make_optimizers(model)
        state = named_state(modules)
        steps = []
        for i, step_ref in enumerate(ref["steps"]):
            result, state = compare_step(step_ref, state, model, modules, optimizers, feats, inputs, list(optimizers))
            result["step"] = i
            steps.append(result)
        arms[arm] = {"steps": steps, "pass": all(s["pass"] for s in steps)}
    steps = arms["official-pusht"]["steps"]
    record = {"schema": SCHEMA, "issue": 130, "family": "dino_wm", "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "device": "cpu float32", "tolerance": TOLERANCE,
              "configuration": "official PushT (released outputs/pusht/hydra.yaml): 196 patches x 384, proprio 4 -> 10, "
                               "action 2 x frameskip 5 -> 10, concat_dim 1, num_hist 3, ViT depth 6 / 16 heads / mlp 2048 / "
                               "dropout 0.1; has_decoder False (official predictor-only option)",
              "lr_note": "DINO-WM has no scheduler: step 0 runs at the positive base LR 5e-4; both steps at it",
              "vendor_hashes_match": hashes, "official_torch": ref["torch"], "torch": torch.__version__,
              "reference_sha256": "sha256:" + sha256_of(out_dir / "reference.pt"),
              "feature_max_abs_delta": feature_delta, "features_pass": feature_delta <= TOLERANCE["feature_max_abs"],
              "optimizer_defaults": defaults, "official_optimizer_defaults": ref["optimizer_defaults"],
              "optimizer_defaults_equal": bool(defaults_equal),
              "arms": arms, "decided_by": "official-pusht (own-features arm is a diagnostic)"}
    record["pass"] = bool(all(hashes.values()) and record["features_pass"] and defaults_equal
                          and record["arms"]["official-pusht"]["pass"])
    (out_dir / "gport.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"[issue-130] G-port: {'PASS' if record['pass'] else 'FAIL'} (features {feature_delta:.2e}; "
          + "; ".join(f"{a} step {s['step']} loss {s['loss']:.6f}/{s['official_loss']:.6f} grad {s['grad_tensor_max_relative_l2']:.2e} "
                      f"params {s['param_max_abs_delta']:.2e} close {s['param_close_share']:.5f} pass {s['pass']}"
                      for a, block in arms.items() for s in block["steps"]) + ")", flush=True)
    return 0 if record["pass"] else 1


if __name__ == "__main__":
    mode, target = sys.argv[1], str(Path(sys.argv[2]).resolve())
    if mode == "--reference":
        sys.exit(reference(target))
    if mode == "--compare":
        sys.path.insert(0, str(ROOT))
        sys.exit(compare(target))
    raise SystemExit("usage: b127_dino_wm_port_reference.py --reference|--compare OUT_DIR")
