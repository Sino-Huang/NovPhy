"""#136 G-port (Decision P-route on #136): the official Diffusion Forcing code (``DiffusionForcingBase`` under real
Lightning, env ``b127-dforce``) against the NovPhy adapter (``world_model/baselines/diffusion_forcing/novphy.py``,
env ``novphy``) on fixed CPU float32 inputs, both on torch 2.13.0+cu130.

This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research template
[repo](https://github.com/buoyancy99/research-template). By its MIT license, you must keep the above sentence in
`README.md` and the `LICENSE` file to credit the author. (Upstream fork-credit sentence, kept as upstream modules do.)

Modes (in order):
1. ``--inputs`` (novphy): 4 fit shots (carriers at frames 0, 5, ..., 225 and launches, the runner's fit store) and the
   fit-split normalisation -> ``gport/inputs.pt``.
2. ``--reference`` (b127-dforce): per arm (``ts`` = the time-series recipe of ts-native / ts-100k, ``planning``): the
   official module's initial state (torch seed), ``training_step`` (forward: loss and predictions), two optimizer
   steps under ``lightning.Trainer`` (precision 32, the recipe's gradient clip; batch b re-seeds torch with
   STEP_SEED + b at ``on_train_batch_start``): step-0 LR, pre-clip gradient norm, post-clip gradients, loss and
   parameters after each step; three ``ddim_sample_step`` calls (indices 0, 25, 49); the full causal-uncertainty
   ``validation_step`` (``context_frames 1``, ``chunk_size -1``, ``calc_crps_sum 20``) and the official
   ``crps_quantile_sum`` of its paths -> ``gport/reference.pt``.
3. ``--compare`` (novphy): the adapter on the same inputs and seeds -> ``gport/gport.json`` (schema
   ``issue_127_gport_v1``; tolerances of #123; expected exact on the same torch build).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "world_model" / "baselines" / "diffusion_forcing" / "vendor"
OUTPUT = ROOT / ".local-artifacts" / "issue-136-diffusion-forcing-v1"
SHOTS = 4
INIT_SEED, FWD_SEED, STEP_SEED, DDIM_SEED, SAMPLE_SEED = 1360, 1361, 1362, 1363, 1364
DDIM_INDICES = (0, 25, 49)
SAMPLE_COPIES = 20  # >= 20: the 0.05 quantile needs kthvalue index >= 1
ARMS = {"ts": "ts-native", "planning": "planning"}  # ts-100k = ts-native's model and optimizer (fixed steps)
TOLERANCE = {"loss_relative": 1e-4, "grad_norm_relative": 1e-4, "grad_tensor_relative_l2": 1e-3,
             "param_close_abs": 1e-7, "param_close_share": 0.999, "forward_max_abs": 1e-5, "sample_max_abs": 1e-5}


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


def _adapter(standalone=False):
    """The adapter module; ``standalone`` loads it by file location (b127-dforce lacks the world_model package's
    dependencies; the adapter itself needs torch / numpy / einops only)."""
    if standalone:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "b127_dforce_novphy", ROOT / "world_model" / "baselines" / "diffusion_forcing" / "novphy.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    sys.path.insert(0, str(ROOT))
    from world_model.baselines.diffusion_forcing import novphy as dn
    return dn


# ---------------------------------------------------------------------------------------------------------------
# --inputs (novphy)
# ---------------------------------------------------------------------------------------------------------------

def inputs(output=OUTPUT) -> int:
    sys.path.insert(0, str(ROOT))
    from scripts import external_baselines as eb
    from scripts import run_issue_136_diffusion_forcing as runner
    dn = _adapter()
    store = runner.fit_store(eb.FitShots())
    mean, std = dn.normalisation(store["carriers"])
    rows = list(range(SHOTS))
    xs, conditions, nonterminals = dn.sequence_batch(store["carriers"], store["launch5"], rows)
    payload = {"xs": xs, "conditions": conditions, "nonterminals": nonterminals, "mean": mean, "std": std,
               "rows": rows, "written_at": utc_now()}
    path = Path(output) / "gport" / "inputs.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"inputs: {tuple(xs.shape)} -> {path}")
    return 0


# ---------------------------------------------------------------------------------------------------------------
# --reference (b127-dforce: the official code under Lightning)
# ---------------------------------------------------------------------------------------------------------------

def official_cfg(config, mean, std):
    """The DictConfig ``DiffusionForcingBase`` reads (df_base.yaml keys) for the carrier sequence."""
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "debug": False, "lr": config["lr"], "x_shape": [236], "z_shape": [config["z_dim"]], "frame_stack": 1,
        "data_mean": [float(v) for v in mean], "data_std": [float(v) for v in std], "external_cond_dim": 5,
        "context_frames": 1, "weight_decay": config["weight_decay"], "warmup_steps": config["warmup_steps"],
        "gt_first_frame": config["gt_first_frame"], "gt_cond_prob": config["gt_cond_prob"],
        "uncertainty_scale": config["uncertainty_scale"], "chunk_size": -1, "calc_crps_sum": SAMPLE_COPIES,
        "learnable_init_z": False, "optimizer_beta": list(config["betas"]), "diffusion": dict(config["diffusion"])})


def reference(output=OUTPUT) -> int:
    import lightning.pytorch as pl
    sys.path.insert(0, str(VENDOR))
    from algorithms.diffusion_forcing.df_base import DiffusionForcingBase
    from algorithms.common.metrics import crps_quantile_sum
    dn = _adapter(standalone=True)  # configurations only; the official model below is the vendored package
    data = torch.load(Path(output) / "gport" / "inputs.pt", weights_only=False)
    batch = [data["xs"], data["conditions"], data["nonterminals"]]
    record = {"torch": torch.__version__, "lightning": pl.__version__ if hasattr(pl, "__version__") else None,
              "arms": {}}

    class Ref(DiffusionForcingBase):
        trace = None

        def on_train_batch_start(self, batch, batch_idx):
            torch.manual_seed(STEP_SEED + batch_idx)

        def on_before_optimizer_step(self, optimizer):
            grads = [p.grad for p in self.parameters() if p.grad is not None]
            norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g, 2.0) for g in grads]), 2.0)
            self.trace.append({"lr": optimizer.param_groups[0]["lr"], "grad_norm_preclip": float(norm)})

        def on_train_batch_end(self, outputs, batch, batch_idx):
            self.trace[-1].update(
                loss=outputs["loss"].detach().clone(),
                grads={n: p.grad.detach().clone() for n, p in self.named_parameters() if p.grad is not None},
                params={n: p.detach().clone() for n, p in self.named_parameters()})

    for arm, name in ARMS.items():
        config = dn.config_of(name)
        cfg = official_cfg(config, data["mean"], data["std"])
        torch.manual_seed(INIT_SEED)
        algo = Ref(cfg)
        init = {k: v.clone() for k, v in algo.state_dict().items()}
        out = {"init": init, "parameters": sum(p.numel() for p in algo.parameters() if p.requires_grad)}
        # forward + loss with per-step noise levels
        torch.manual_seed(FWD_SEED)
        fwd = algo.training_step([t.clone() for t in batch], 1)
        out["forward"] = {"loss": fwd["loss"].detach(), "xs_pred": fwd["xs_pred"].detach()}
        # two optimizer steps under Lightning
        algo.load_state_dict(init)
        algo.trace = []
        trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=2, precision="32-true", logger=False,
                             gradient_clip_val=config["gradient_clip_val"], enable_checkpointing=False,
                             enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
                             limit_val_batches=0)
        loader = torch.utils.data.DataLoader([[t.clone() for t in batch]] * 2, batch_size=None,
                                             generator=torch.Generator().manual_seed(0))
        trainer.fit(algo, train_dataloaders=loader)
        out["steps"] = algo.trace
        # one DDIM sampling step at three indices (global generator); a fresh module without a trainer reference
        algo = Ref(cfg)
        algo.load_state_dict(init)
        tm = algo.transition_model
        xs, conditions, _, z = algo._preprocess_batch([t.clone() for t in batch])
        torch.manual_seed(DDIM_SEED)
        x, zc, ddim = xs[3].clone(), torch.randn_like(z), []
        with torch.no_grad():
            for index in DDIM_INDICES:
                x, zc = tm.ddim_sample_step(x, zc, conditions[4], index)
                ddim.append({"index": index, "x": x.clone(), "z": zc.clone()})
        out["ddim"] = ddim
        # the full causal-uncertainty prediction loop (validation_step) and its CRPS-sum
        algo.validation_step_outputs.clear()
        torch.manual_seed(SAMPLE_SEED)
        algo.validation_step([t.clone() for t in batch], 0)
        preds, gt = algo.validation_step_outputs[-1]
        crps_preds = preds.view(preds.shape[0], SAMPLE_COPIES, -1, *preds.shape[2:]).float().permute(1, 0, 2, 3)
        crps_gt = gt.view(gt.shape[0], SAMPLE_COPIES, -1, *gt.shape[2:]).float()[:, 0]
        out["sampling"] = {"xs_pred": preds, "crps_sum": crps_quantile_sum(crps_preds[:, 1:], crps_gt[1:])}
        record["arms"][arm] = out
        print(f"reference {arm}: loss {float(out['forward']['loss']):.6f}, steps "
              f"{[float(s['loss']) for s in out['steps']]}, crps {float(out['sampling']['crps_sum']):.6f}")
    torch.save(record, Path(output) / "gport" / "reference.pt")
    return 0


# ---------------------------------------------------------------------------------------------------------------
# --compare (novphy: the adapter)
# ---------------------------------------------------------------------------------------------------------------

def _rel(a, b):
    a, b = float(a), float(b)
    return abs(a - b) / max(abs(b), 1e-12)


def _rel_l2(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-12))


def _max_abs(a, b):
    return float((a.double() - b.double()).abs().max())


def compare(output=OUTPUT) -> int:
    dn = _adapter()
    output = Path(output)
    data = torch.load(output / "gport" / "inputs.pt", weights_only=False)
    ref = torch.load(output / "gport" / "reference.pt", weights_only=False)
    batch = (data["xs"], data["conditions"], data["nonterminals"])
    tol = TOLERANCE
    arms = {}
    for arm, name in ARMS.items():
        config = dn.config_of(name)
        r = ref["arms"][arm]
        torch.manual_seed(INIT_SEED)
        model = dn.build_model(config, data["mean"], data["std"])
        init = model.state_dict()
        init_equal = (set(init) == set(r["init"]) and all(torch.equal(init[k], r["init"][k]) for k in init))
        res = {"configuration": name, "official_parameters": r["parameters"],
               "adapter_parameters": dn.parameter_count(model), "init_state_equal": bool(init_equal)}
        # forward
        torch.manual_seed(FWD_SEED)
        loss, _, preds = dn.training_loss(model, *[t.clone() for t in batch])
        preds = model.unnormalize(preds.detach())
        res["forward"] = {"loss": [float(loss.detach()), float(r["forward"]["loss"])],
                          "loss_rel": _rel(loss, r["forward"]["loss"]),
                          "xs_pred_max_abs": _max_abs(preds, r["forward"]["xs_pred"])}
        res["forward"]["pass"] = bool(res["forward"]["loss_rel"] <= tol["loss_relative"]
                                      and res["forward"]["xs_pred_max_abs"] <= tol["forward_max_abs"])
        # two optimizer steps
        model.load_state_dict(r["init"])
        trainer = dn.Trainer(config, data["mean"], data["std"], "cpu", model=model)
        steps = []
        for b, rs in enumerate(r["steps"]):
            torch.manual_seed(STEP_SEED + b)
            out = trainer.train_step(*[t.clone() for t in batch])
            grads = {f"transition_model.{n}": p.grad for n, p in model.transition_model.named_parameters()
                     if p.grad is not None}
            params = dict(model.named_parameters())
            norm = (float(out["grad_norm"]) if out["grad_norm"] is not None else None)
            deltas = torch.cat([(params[n].detach() - rs["params"][n]).abs().flatten() for n in rs["params"]])
            lr = out["lr"]
            step = {"index": b, "lr": [lr, rs["lr"]], "lr_equal": lr == rs["lr"],
                    "loss": [float(out["loss"]), float(rs["loss"])], "loss_rel": _rel(out["loss"], rs["loss"]),
                    "grad_norm_preclip": [norm, rs["grad_norm_preclip"]],
                    "grad_names_equal": set(grads) == set(rs["grads"]),
                    "grad_max_rel_l2": max(_rel_l2(grads[n], rs["grads"][n]) for n in rs["grads"]),
                    "param_max_abs": float(deltas.max()),
                    "param_close_share": float((deltas <= tol["param_close_abs"]).double().mean())}
            step["grad_norm_rel"] = None if norm is None else _rel(norm, rs["grad_norm_preclip"])
            step["pass"] = bool(step["lr_equal"] and step["loss_rel"] <= tol["loss_relative"]
                                and step["grad_names_equal"]
                                and (norm is None or step["grad_norm_rel"] <= tol["grad_norm_relative"])
                                and step["grad_max_rel_l2"] <= tol["grad_tensor_relative_l2"]
                                and step["param_close_share"] >= tol["param_close_share"])
            steps.append(step)
        res["steps"] = steps
        # DDIM steps
        model.load_state_dict(r["init"])
        model.eval()
        tm = model.transition_model
        xs, conditions, _, z = dn.preprocess(model, *[t.clone() for t in batch])
        torch.manual_seed(DDIM_SEED)
        x, zc, ddim = xs[3].clone(), torch.randn_like(z), []
        pairs = dn.ddim_times(tm)
        with torch.no_grad():
            for rd in r["ddim"]:
                x, zc = dn.ddim_step(tm, x, zc, conditions[4], rd["index"], dn.GlobalNoise(), pairs)
                ddim.append({"index": rd["index"], "x_max_abs": _max_abs(x, rd["x"]), "z_max_abs": _max_abs(zc, rd["z"])})
        res["ddim"] = {"steps": ddim, "pass": all(d["x_max_abs"] <= tol["sample_max_abs"]
                                                  and d["z_max_abs"] <= tol["sample_max_abs"] for d in ddim)}
        # the full sampling loop (rows sample-major, as validation_step's calc_crps_sum expand) and CRPS-sum
        torch.manual_seed(SAMPLE_SEED)
        z0 = data["xs"][:, 0].repeat(SAMPLE_COPIES, 1)
        launch = data["conditions"][:, 1].repeat(SAMPLE_COPIES, 1)
        paths = dn.sample_paths(model, z0, launch, dn.GlobalNoise())
        official_paths = r["sampling"]["xs_pred"]
        preds = paths.view(paths.shape[0], SAMPLE_COPIES, -1, dn.CARRIER_DIM).permute(1, 0, 2, 3)
        gt = data["xs"].permute(1, 0, 2)
        crps = dn.crps_quantile_sum(preds[:, 1:], gt[1:])
        res["sampling"] = {"rows": int(paths.shape[1]), "frames": int(paths.shape[0]),
                           "pyramid_rows": int(dn.pyramid(dn.HORIZON, tm.sampling_timesteps,
                                                          config["uncertainty_scale"]).shape[0]),
                           "paths_max_abs": _max_abs(paths, official_paths),
                           "frame225_max_abs": _max_abs(paths[-1], official_paths[-1]),
                           "crps_sum": [float(crps), float(r["sampling"]["crps_sum"])],
                           "crps_rel": _rel(crps, r["sampling"]["crps_sum"])}
        res["sampling"]["pass"] = bool(res["sampling"]["paths_max_abs"] <= tol["sample_max_abs"]
                                       and res["sampling"]["crps_rel"] <= tol["loss_relative"])
        res["pass"] = bool(init_equal and res["official_parameters"] == res["adapter_parameters"]
                           and res["forward"]["pass"] and all(s["pass"] for s in steps)
                           and res["ddim"]["pass"] and res["sampling"]["pass"])
        res["exact"] = bool(res["forward"]["loss_rel"] == 0 and res["forward"]["xs_pred_max_abs"] == 0
                            and all(s["loss_rel"] == 0 and s["grad_max_rel_l2"] == 0 and s["param_max_abs"] == 0
                                    for s in steps)
                            and all(d["x_max_abs"] == 0 and d["z_max_abs"] == 0 for d in ddim)
                            and res["sampling"]["paths_max_abs"] == 0 and res["sampling"]["crps_rel"] == 0)
        arms[arm] = res
        print(f"compare {arm}: pass {res['pass']} exact {res['exact']}")
    defects = injected_defects(dn, data, ref["arms"]["ts"], batch)
    print(f"injected defects detected: {defects}")
    vendor = json.loads((dn.VENDOR / "VENDOR.json").read_text())
    vendor_ok = {rel: sha256(dn.VENDOR / rel) == entry["vendored_sha256"] for rel, entry in vendor["files"].items()}
    record = {
        "schema": "issue_127_gport_v1", "issue": 136, "arm": "T4 DF-car", "checked_at": utc_now(),
        "device": "cpu float32", "threads": torch.get_num_threads(), "torch": torch.__version__,
        "reference_torch": ref["torch"], "reference_lightning": ref["lightning"],
        "same_torch": torch.__version__ == ref["torch"], "upstream": vendor["upstream"], "branch": vendor["branch"],
        "commit": vendor["commit"], "tolerance": tol,
        "G_port_rule": "#123 tolerances (loss / pre-clip grad-norm relative 1e-4, per-tensor post-clip gradient relative "
                       "L2 1e-3, parameters |delta| <= 1e-7 on >= 99.9 % after each of two optimizer steps, step-0 LR "
                       "equal) + forward predictions, DDIM steps and the full sampling loop max abs <= 1e-5, CRPS-sum "
                       "relative 1e-4; reference = official DiffusionForcingBase under lightning.Trainer (precision "
                       "32) in env b127-dforce; same torch build",
        "vendor_hashes_match": vendor_ok, "inputs_sha256": sha256(output / "gport" / "inputs.pt"),
        "reference_sha256": sha256(output / "gport" / "reference.pt"),
        "reference_script_sha256": sha256(Path(__file__)), "adapter_sha256": sha256(Path(dn.__file__)),
        "not_checked_on_cpu": "16-mixed (fp16 autocast + GradScaler) runs on CUDA only; the adapter follows "
                              "Lightning's MixedPrecisionPlugin order (scale, backward, unscale, clip, step, update)",
        "arms": arms,
        "injected_defects": defects,
        "pass": bool(all(a["pass"] for a in arms.values()) and all(vendor_ok.values())
                     and all(d["detected"] for d in defects.values())),
        "exact": bool(all(a["exact"] for a in arms.values())),
    }
    write_json(output / "gport" / "gport.json", record)
    print(f"G-port pass {record['pass']} (exact {record['exact']})")
    return 0 if record["pass"] else 1


def injected_defects(dn, data, r, batch):
    """The comparison must fail on three port defects (time-series arm): the warm-up applied before the first step
    (step-0 LR / parameters), one noise level for the whole sequence (the forward loss), and the full-sequence noise
    schedule instead of the causal-uncertainty pyramid (sampled paths)."""
    tol = TOLERANCE
    config = dn.config_of(ARMS["ts"])
    model = dn.build_model(config, data["mean"], data["std"])
    model.load_state_dict(r["init"])
    trainer = dn.Trainer(config, data["mean"], data["std"], "cpu", model=model)
    for group in trainer.optimizer.param_groups:
        group["lr"] = config["lr"] / config["warmup_steps"]
    torch.manual_seed(STEP_SEED)
    trainer.train_step(*[t.clone() for t in batch])
    params = dict(model.named_parameters())
    delta = max(float((params[n].detach() - v).abs().max()) for n, v in r["steps"][0]["params"].items())
    out = {"warmup_before_step": {"param_max_abs": delta, "detected": delta > tol["param_close_abs"]}}
    model.load_state_dict(r["init"])
    tm, fixed = model.transition_model, {}
    original = tm.forward

    def one_level(z, x, cond, deterministic_t=None, cum_snr=None):
        fixed.setdefault("t", int(torch.randint(0, tm.num_timesteps, (1,))))
        return original(z, x, cond, deterministic_t=fixed["t"], cum_snr=cum_snr)

    tm.forward = one_level
    torch.manual_seed(FWD_SEED)
    loss = float(dn.training_loss(model, *[t.clone() for t in batch])[0])
    del tm.forward
    rel = _rel(loss, r["forward"]["loss"])
    out["shared_noise_level"] = {"loss_rel": rel, "detected": rel > tol["loss_relative"]}
    model.config = {**config, "uncertainty_scale": 0}
    torch.manual_seed(SAMPLE_SEED)
    paths = dn.sample_paths(model, data["xs"][:, 0].repeat(SAMPLE_COPIES, 1),
                            data["conditions"][:, 1].repeat(SAMPLE_COPIES, 1), dn.GlobalNoise())
    model.config = config
    gap = _max_abs(paths, r["sampling"]["xs_pred"])
    out["full_sequence_noise_schedule"] = {"paths_max_abs": gap, "detected": gap > tol["sample_max_abs"]}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("inputs", "reference", "compare"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    output = Path(args.output)
    if args.inputs:
        return inputs(output)
    if args.reference:
        return reference(output)
    return compare(output)


if __name__ == "__main__":
    raise SystemExit(main())
