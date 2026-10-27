"""#138 G-port (Decisions P-route / P-adapter on #138): the official V-JEPA 2 code (facebookresearch/vjepa2 @ 204698b,
the pinned clone, env ``b127-vjepa2``) against the NovPhy adapter (``world_model/baselines/vjepa2/novphy.py``, env
``novphy``) on fixed CPU float32 inputs, both on torch 2.13.0+cu130.

Modes (in order):
1. ``--inputs`` (novphy): 2 fit shots x 4 stride-5 frames (positions 0, 5, 10, 15; uint8 320 x 240 from the #123 fit
   pixel store) with their launches (clip actions: the launch at step 0, zeros after; zero states), fixed random
   7-D actions / states for the released DROID predictor -> ``gport/inputs.pt``.
2. ``--reference`` (b127-vjepa2, the pinned clone on sys.path): the hub entry ``_make_vjepa2_ac_model`` with the
   released ``vjepa2-ac-vitg.pt`` (``torch.hub.load_state_dict_from_url`` served from the local file); the official
   ``VideoTransform`` (``make_transforms`` with the config's data_aug); ``forward_target`` / ``forward_predictions`` /
   ``loss_fn`` / ``train_step`` executed from the source text of ``app/vjepa_droid/train.py`` (nested functions of
   ``main``, extracted with ``ast`` and run unchanged); ``init_opt`` (AdamW groups + WSD / cosine-WD schedules);
   the released predictor's forward; per case (``native``: auto_steps 2, ``ar3``: auto_steps 3) a fresh predictor
   (official constructor under torch seed 1380; num_frames 64 as the hub, Decision P-adapter), its loss and two
   official updates (gradients and parameters captured around ``optimizer.step``); and (``native``) a 7-step rollout
   by ``step_predictor`` of ``notebooks/utils/world_model_wrapper.py`` (pose update replaced by identity: NovPhy has
   no pose) -> ``gport/reference.pt``.
3. ``--compare`` (novphy): the adapter on the same inputs -> ``gport/gport.json`` (schema ``issue_127_gport_v1``;
   #123 tolerances; expected exact on the same torch build).
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CLONE = Path("/mnt/array/sukaih/novphy-baselines-src/vjepa2@204698b")
OUTPUT = ROOT / ".local-artifacts" / "issue-138-vjepa2-ac-v1"
CHECKPOINT = ROOT / "data/external/vjepa2/vjepa2-ac-vitg.pt"
INIT_SEED, DATA_SEED = 1380, 1381
SHOTS = (0, 1)
POSITIONS = (0, 5, 10, 15)
ROLLOUT_STEPS = 7
CASES = {"native": {"auto_steps": 2}, "ar3": {"auto_steps": 3}}
TOLERANCE = {"loss_relative": 1e-4, "grad_norm_relative": 1e-4, "grad_tensor_relative_l2": 1e-3,
             "param_close_abs": 1e-7, "param_close_share": 0.999, "feature_max_abs": 1e-5}
# config data_aug / data of configs/train/vitg16/droid-256px-8f.yaml
TRANSFORM = {"random_horizontal_flip": False, "random_resize_aspect_ratio": [0.75, 1.35],
             "random_resize_scale": [1.777, 1.777], "reprob": 0.0, "auto_augment": False, "motion_shift": False,
             "crop_size": 256}


def sha256(path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------------------------
# --inputs (novphy)
# ---------------------------------------------------------------------------------------------------------------

def make_inputs(output: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from scripts import external_baselines as eb
    from world_model.baselines.vjepa2 import novphy as vn
    shots = eb.FitShots()
    frames = torch.stack([torch.stack([torch.as_tensor(shots.frame(s, p)) for p in POSITIONS]) for s in SHOTS])
    launch5 = torch.as_tensor(shots.launch5)[list(SHOTS)].float()
    g = torch.Generator().manual_seed(DATA_SEED)
    t = len(POSITIONS)
    payload = {"shots": [shots.keys[s] for s in SHOTS], "positions": POSITIONS, "frames": frames,
               "launch5": launch5, "actions": vn.clip_actions(launch5, [0] * len(SHOTS), t),
               "states": vn.clip_states(len(SHOTS), t),
               "actions7": torch.randn(len(SHOTS), t - 1, 7, generator=g) * 0.05,
               "states7": torch.randn(len(SHOTS), t - 1, 7, generator=g)}
    (output / "gport").mkdir(parents=True, exist_ok=True)
    torch.save(payload, output / "gport" / "inputs.pt")
    print(f"inputs: shots {payload['shots']} positions {POSITIONS}")


# ---------------------------------------------------------------------------------------------------------------
# --reference (b127-vjepa2; the official code from the pinned clone)
# ---------------------------------------------------------------------------------------------------------------

def nested_function(path, name):
    """Source of the (possibly nested) function ``name`` in ``path``, dedented (executed unchanged)."""
    text = Path(path).read_text()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return textwrap.dedent(ast.get_source_segment(text, node))
    raise KeyError(f"{name} not in {path}")


def official_function(path, name, namespace):
    exec(compile(nested_function(path, name), f"{path}::{name}", "exec"), namespace)
    return namespace[name]


def _capture(model, optimizer):
    captured, step = [], optimizer.step

    def wrapped(*args, **kwargs):
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        out = step(*args, **kwargs)
        captured.append({"grads": grads, "params": {n: p.detach().clone() for n, p in model.named_parameters()}})
        return out
    optimizer.step = wrapped
    return captured


def reference(output: Path) -> None:
    # the clone's namespace package ``src`` must not lose to the repository's regular package ``src`` (PYTHONPATH)
    sys.path[:] = [str(CLONE)] + [p for p in sys.path if Path(p or ".").resolve() != ROOT]
    torch.set_num_threads(4)
    from app.vjepa_droid.transforms import make_transforms
    from app.vjepa_droid.utils import init_opt
    import src.models.ac_predictor as vit_ac_pred
    from src.hub import backbones
    inputs = torch.load(output / "gport" / "inputs.pt", map_location="cpu", weights_only=False)
    torch.hub.load_state_dict_from_url = lambda url, map_location=None, **kw: torch.load(
        CHECKPOINT, map_location="cpu", mmap=True, weights_only=False)
    encoder, released = backbones._make_vjepa2_ac_model(model_name="vit_ac_giant", img_size=256, pretrained=True)
    encoder.eval()
    released.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    transform = make_transforms(**TRANSFORM)
    clips = torch.stack([transform(f.permute(0, 2, 3, 1).numpy()) for f in inputs["frames"]])  # [B, C, T, H, W]
    b, _, t = clips.shape[:3]
    train_py = CLONE / "app/vjepa_droid/train.py"
    ns = {"torch": torch, "F": F, "batch_size": b, "max_num_frames": t, "normalize_reps": True,
          "target_encoder": encoder, "tokens_per_frame": 256, "loss_exp": 1.0, "mixed_precision": False,
          "dtype": torch.float32, "scaler": None, "clips": clips, "actions": inputs["actions"],
          "states": inputs["states"], "extrinsics": torch.zeros(b, t, 4)}
    forward_target = official_function(train_py, "forward_target", ns)
    with torch.no_grad():
        h = forward_target(clips)
        released_tf = released(h[:, :-256], inputs["actions7"], inputs["states7"])
    out = {"torch": torch.__version__, "clips": clips, "h": h, "released_tf": released_tf, "cases": {}}
    print(f"reference: features {tuple(h.shape)}; released predictor {tuple(released_tf.shape)}")
    for case, spec in CASES.items():
        torch.manual_seed(INIT_SEED)
        predictor = vit_ac_pred.__dict__["vit_ac_predictor"](
            img_size=256, patch_size=16, num_frames=64, tubelet_size=2, embed_dim=encoder.embed_dim,
            predictor_embed_dim=1024, action_embed_dim=5, depth=24, is_frame_causal=True, num_heads=16,
            uniform_power=True, use_rope=True, use_sdpa=True, use_silu=False, wide_silu=True, use_extrinsics=False,
            use_activation_checkpointing=False)
        init = {k: v.clone() for k, v in predictor.state_dict().items()}
        cns = {**ns, "predictor": predictor, "auto_steps": spec["auto_steps"]}
        forward_predictions = official_function(train_py, "forward_predictions", cns)
        loss_fn = official_function(train_py, "loss_fn", cns)
        with torch.no_grad():
            z_tf, z_ar = forward_predictions(h)
            forward_loss = float(loss_fn(z_tf, h) + loss_fn(z_ar, h))
        optimizer, scaler, scheduler, wd_scheduler = init_opt(
            encoder=encoder, predictor=predictor, iterations_per_epoch=300, start_lr=7.5e-5, ref_lr=4.25e-4,
            warmup=15, anneal=15, num_epochs=315, wd=0.04, final_wd=0.04, final_lr=0.0, mixed_precision=False,
            betas=(0.9, 0.999), eps=1e-8)
        captured = _capture(predictor, optimizer)
        cns.update(optimizer=optimizer, scheduler=scheduler, wd_scheduler=wd_scheduler, scaler=scaler)
        train_step = official_function(train_py, "train_step", cns)
        predictor.train()
        steps = []
        for k in range(2):
            loss, jloss, sloss, lr, wd = train_step()
            steps.append({"loss": loss, "jloss": jloss, "sloss": sloss, "lr": lr, "wd": wd, **captured[k]})
        entry = {"init": init, "forward_loss": forward_loss, "steps": steps,
                 "parameters": sum(p.numel() for p in predictor.parameters() if p.requires_grad)}
        if case == "native":
            wrapper = SimpleNamespace(predictor=predictor.eval(), tokens_per_frame=256, normalize_reps=True)
            wns = {"F": F, "self": wrapper, "compute_new_pose": lambda pose, action: pose}
            step_predictor = official_function(CLONE / "notebooks/utils/world_model_wrapper.py", "step_predictor",
                                               wns)
            reps = h[:, None, :256].view(b, 1, 256, -1)
            poses = torch.zeros(b, 1, 5)
            rolled = []
            with torch.no_grad():
                for k in range(ROLLOUT_STEPS):
                    acts = torch.zeros(b, k + 1, 5)
                    acts[:, 0] = inputs["launch5"]
                    nxt, _ = step_predictor(reps, acts, poses)
                    reps = torch.cat([reps, nxt], dim=1)
                    poses = torch.cat([poses, torch.zeros(b, 1, 5)], dim=1)
                    rolled.append(nxt[:, 0])
            entry["rollout"] = torch.stack(rolled)
        out["cases"][case] = entry
        print(f"reference {case}: forward loss {forward_loss:.6f}; steps {[s['loss'] for s in steps]}")
    torch.save(out, output / "gport" / "reference.pt")


# ---------------------------------------------------------------------------------------------------------------
# --compare (novphy; the adapter)
# ---------------------------------------------------------------------------------------------------------------

def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def compare(output: Path) -> int:
    sys.path.insert(0, str(ROOT))
    from world_model.baselines.vjepa2 import novphy as vn
    torch.set_num_threads(4)
    inputs = torch.load(output / "gport" / "inputs.pt", map_location="cpu", weights_only=False)
    ref = torch.load(output / "gport" / "reference.pt", map_location="cpu", mmap=True, weights_only=False)
    frames = inputs["frames"]
    b, t = frames.shape[:2]
    pre = vn.preprocess(frames.flatten(0, 1)).view(b, t, 3, vn.CROP, vn.CROP).permute(0, 2, 1, 3, 4)
    preprocess = {"max_abs": float((pre - ref["clips"]).abs().max()), "equal": bool(torch.equal(pre, ref["clips"]))}
    preprocess["pass"] = preprocess["max_abs"] <= TOLERANCE["feature_max_abs"]
    checkpoint = vn.load_checkpoint()
    encoder, load_msg = vn.load_encoder("cpu", checkpoint)
    h = vn.normalize(vn.encode(encoder, frames.flatten(0, 1), autocast=False)).view(b, t * vn.TOKENS, vn.EMBED_DIM)
    features = {"max_abs": float((h - ref["h"]).abs().max()), "equal": bool(torch.equal(h, ref["h"])),
                "encoder_load": load_msg}
    features["pass"] = features["max_abs"] <= TOLERANCE["feature_max_abs"]
    released = vn.build_predictor(action_dim=7)
    released.load_state_dict(vn._clean_backbone_key(checkpoint["predictor"]), strict=True)
    with torch.no_grad():
        released_tf = released.eval()(h[:, :-vn.TOKENS], inputs["actions7"], inputs["states7"])
    released_check = {"max_abs": float((released_tf - ref["released_tf"]).abs().max()),
                      "load": "strict=True (all 300 tensors)"}
    released_check["pass"] = released_check["max_abs"] <= TOLERANCE["feature_max_abs"]
    del released, encoder, checkpoint
    cases = {}
    for case, spec in CASES.items():
        r = ref["cases"][case]
        torch.manual_seed(INIT_SEED)
        predictor = vn.build_predictor()
        init_equal = (set(predictor.state_dict()) == set(r["init"]) and
                      all(torch.equal(v, r["init"][k]) for k, v in predictor.state_dict().items()))
        with torch.no_grad():
            loss, _, _ = vn.clip_loss(predictor, h, inputs["actions"], inputs["states"], spec["auto_steps"])
        forward = {"loss": [float(loss), r["forward_loss"]], "loss_rel": _rel(float(loss), r["forward_loss"])}
        forward["pass"] = forward["loss_rel"] <= TOLERANCE["loss_relative"]
        optimizer, scheduler, wd_scheduler = vn.build_optimizer(predictor, {})
        captured = _capture(predictor, optimizer)
        predictor.train()
        terms = [vn.train_step(predictor, optimizer, scheduler, wd_scheduler,
                               [(h, inputs["actions"], inputs["states"])], spec["auto_steps"]) for _ in range(2)]
        steps = []
        for k, (term, c, rs) in enumerate(zip(terms, captured, r["steps"])):
            names_equal = set(c["grads"]) == set(rs["grads"])
            grad_rel = max(float((c["grads"][n] - rs["grads"][n]).norm() / rs["grads"][n].norm().clamp_min(1e-12))
                           for n in rs["grads"]) if names_equal else None
            gn = float(torch.stack([g.norm() for g in c["grads"].values()]).norm())
            gn_ref = float(torch.stack([g.norm() for g in rs["grads"].values()]).norm())
            diffs = torch.cat([(c["params"][n] - rs["params"][n]).abs().flatten() for n in rs["params"]])
            close = float((diffs <= TOLERANCE["param_close_abs"]).float().mean())
            entry = {"index": k, "loss": [term["loss"], rs["loss"]], "loss_rel": _rel(term["loss"], rs["loss"]),
                     "jloss": [term["jloss"], rs["jloss"]], "sloss": [term["sloss"], rs["sloss"]],
                     "lr": [term["lr"], rs["lr"]], "lr_equal": term["lr"] == rs["lr"],
                     "wd": [term["wd"], rs["wd"]], "wd_equal": term["wd"] == rs["wd"],
                     "grad_norm": [gn, gn_ref], "grad_norm_rel": _rel(gn, gn_ref), "grad_names_equal": names_equal,
                     "grad_max_rel_l2": grad_rel, "param_max_abs": float(diffs.max()), "param_close_share": close}
            entry["pass"] = bool(entry["loss_rel"] <= TOLERANCE["loss_relative"] and entry["lr_equal"] and
                                 entry["wd_equal"] and entry["grad_norm_rel"] <= TOLERANCE["grad_norm_relative"] and
                                 names_equal and grad_rel <= TOLERANCE["grad_tensor_relative_l2"] and
                                 close >= TOLERANCE["param_close_share"])
            steps.append(entry)
        entry = {"auto_steps": spec["auto_steps"], "init_state_equal": bool(init_equal), "forward": forward,
                 "steps": steps, "parameters": [vn.trainable_parameters(predictor), r["parameters"]]}
        if "rollout" in r:
            keep = tuple(range(1, ROLLOUT_STEPS + 1))
            out = vn.rollout(predictor.eval(), h[:, :vn.TOKENS], inputs["launch5"], steps=ROLLOUT_STEPS, keep=keep)
            pred = torch.stack([out[k] for k in keep])
            entry["rollout"] = {"steps": ROLLOUT_STEPS, "max_abs": float((pred - r["rollout"]).abs().max())}
            entry["rollout"]["pass"] = entry["rollout"]["max_abs"] <= TOLERANCE["feature_max_abs"]
        entry["exact"] = bool(forward["loss_rel"] == 0 and all(s["param_max_abs"] == 0 for s in steps) and
                              entry.get("rollout", {"max_abs": 0})["max_abs"] == 0)
        entry["pass"] = bool(init_equal and forward["pass"] and all(s["pass"] for s in steps) and
                             entry.get("rollout", {"pass": True})["pass"])
        cases[case] = entry
        print(f"compare {case}: pass {entry['pass']} exact {entry['exact']}")
    record = {"schema": "issue_127_gport_v1", "issue": 138, "arm": "T6 V-JEPA 2-AC",
              "G_port_rule": "#123 tolerances (loss / grad-norm relative 1e-4, per-tensor gradient relative L2 1e-3, "
                             "parameters |delta| <= 1e-7 on >= 99.9 % after each of two optimizer steps, LR and WD "
                             "equal) + initial state equal + preprocessing, frozen-encoder features, the released "
                             "predictor's forward and the 7-step step_predictor rollout max abs <= 1e-5; reference = "
                             "the official code (hub _make_vjepa2_ac_model with the released vjepa2-ac-vitg.pt, "
                             "VideoTransform, train.py forward_target / forward_predictions / loss_fn / train_step "
                             "executed from source, init_opt, world_model_wrapper step_predictor) in env b127-vjepa2; "
                             "same torch build; CPU float32 (the official mixed_precision False path)",
              "tolerance": TOLERANCE, "torch": [torch.__version__, ref["torch"]], "preprocess": preprocess,
              "features": features, "released_predictor": released_check, "cases": cases,
              "adapter_sha256": sha256(ROOT / "world_model/baselines/vjepa2/novphy.py"),
              "reference_script_sha256": sha256(Path(__file__)),
              "inputs_sha256": sha256(output / "gport" / "inputs.pt"),
              "reference_sha256": sha256(output / "gport" / "reference.pt"),
              "pass": bool(preprocess["pass"] and features["pass"] and released_check["pass"]
                           and all(c["pass"] for c in cases.values())),
              "written_at": utc_now()}
    write_json(output / "gport" / "gport.json", record)
    print(f"G-port pass {record['pass']}")
    return 0 if record["pass"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inputs", action="store_true")
    mode.add_argument("--reference", action="store_true")
    mode.add_argument("--compare", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    output = Path(args.output)
    if args.inputs:
        make_inputs(output)
        return 0
    if args.reference:
        reference(output)
        return 0
    return compare(output)


if __name__ == "__main__":
    raise SystemExit(main())
