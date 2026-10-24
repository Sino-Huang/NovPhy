"""Issue #129 G-port: the official SlotFormer dynamics, loss and optimizer vs the NovPhy adapter (CPU, float32).

Two modes, one fixed set of arms (``ARMS``):

    # official side, in the b127-slotformer env (official code at SLOTFORMER_REF, nerv v0.1.0, torch 2.7.1)
    /srv/home/shua0171/miniconda3/envs/b127-slotformer/bin/python scripts/b127_slotformer_port_reference.py \
        reference .local-artifacts/issue-129-slotformer-v1/gport
    # adapter side + comparison, in the novphy env (vendored code, torch 2.13) -> gport/gport.json
    python scripts/b127_slotformer_port_reference.py compare .local-artifacts/issue-129-slotformer-v1/gport

Reference, per arm. The model is built by the official ``build_model`` from the official PHYRE config
``slotformer_phyre_params-fold0.py``; the NovPhy arms override only the shape fields (num_slots 19, slot_size 13,
the Transformer size, the clip length). The frozen SAVi decoder the official ``SlotFormer`` loads is the released
``savi_phyre_params-fold0/model_30.pth`` for ``official-phyre``. For the 13-wide arms it is a seeded random decoder
of that shape, written by the official ``StoSAVi._build_decoder``; it is frozen and unused (no image loss), as in
the official PHYRE training.

The step runs through the official training code on a bare ``SlotFormerMethod``:
- ``_setup_optimizer`` (``SlotBaseMethod._configure_optimizers`` -> nerv ``BaseMethod._configure_optimizers``) with
  ``max_epochs`` x loader length = ``TOTAL_STEPS``;
- ``_training_step`` (nerv ``_training_step_fp32``: ``_loss_function``, backward, clip if ``clip_grad`` > 0,
  ``optimizer.step``, ``zero_grad``) and ``_training_step_end`` (per-step ``scheduler.step``).
Gradients are read by an optimizer step pre-hook. Step 1 runs at the schedule's step-0 LR (lr / 100). Step 2 runs
after the scheduler is advanced to the end of warm-up (every group at lr).

The NovPhy arms' official inputs are the shots' strided carrier frames, padded by the official PHYRE
``fix_video_len`` with ``vid_len`` as ``PHYRESlotsDataset._read_slots``. They are packed with the slot layout
declared in Decision P-other: 18 carrier slots, then the launch token ``[launch5, 0 x 8]``. The adapter side builds
the same batch through ``novphy.training_batch`` from the raw carriers, so the comparison covers the adapter's
sequence plan and packing too.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import torch

ROOT = Path(__file__).resolve().parents[1]
SLOTFORMER_REF = Path(os.environ.get("SLOTFORMER_REF", "/mnt/array/sukaih/novphy-baselines-src/SlotFormer@10a6d9b"))
PRETRAINED = ROOT / "data/external/slotformer"  # holds pretrained/ (the released weights)
PHYRE_CONFIG = "slotformer/video_prediction/configs/slotformer_phyre_params-fold0.py"
SEED_INIT, SEED_INPUT, SEED_STEP, SEED_STEP2, SEED_DECODER = 20261024, 7, 11, 13, 17
BATCH = 4
TOTAL_STEPS = 9000
MAX_EPOCHS = 50  # the PHYRE config's max_epochs; the loader length is TOTAL_STEPS / MAX_EPOCHS
CARRIER_POSITIONS = 286  # positions 0..285 (the fit store's window)
ENDS = (285, 150, 230, 138)  # last stored position per shot: two full, two short (vid_len truncation)
# Parameter criterion: #123's rule exactly (run_hjepa_comparison.GPORT_TOLERANCE): after each step the share of
# trainable elements with |delta| <= param_close_abs (1e-7, absolute) is >= 0.999 and no element differs by more than
# 2 lr_max + 1e-7 (lr_max = the largest LR after the step). The frozen G-port runs the official sources on the SAME
# torch build as the adapter (2.13, CPU; the venv b127-slotformer-ref213 over the novphy env), which removes the
# torch-version round-off that made the 2.7.1 comparison fail this rule (gport/torch271/, kept as a diagnostic). The
# per-element isclose(rtol 1e-5, atol 1e-6) share is recorded beside it (G_port_isclose_variant), never gating.
TOLERANCE = {"loss_relative": 1e-4, "grad_norm_relative": 1e-4, "tensor_relative_l2": 1e-3,
             "prediction_relative_l2": 1e-3, "param_rtol": 1e-5, "param_atol": 1e-6, "param_close_share": 0.999,
             "param_max_abs_over_lr": 2.0, "param_close_abs": 1e-7}

NATIVE = {"d_model": 256, "num_layers": 8, "num_heads": 8, "ffn_dim": 1024}
MATCHED = {"d_model": 136, "num_layers": 8, "num_heads": 8, "ffn_dim": 544}
ARMS = {
    # the official PHYRE configuration itself (8 slots x 128, 1 burn-in + 10 rollout frames)
    "official-phyre": {"kind": "slots", "num_slots": 8, "slot_size": 128, "frames": 11, "size": None,
                       "vid_len": (11, 7, 11, 4)},
    "b1-native-s15": {"kind": "carriers", "stride": 15, "size": NATIVE},
    "b1-matched-s15": {"kind": "carriers", "stride": 15, "size": MATCHED},
    "b1-native-s5": {"kind": "carriers", "stride": 5, "size": NATIVE},
    # the native success head: SlotFormer's PHYRE readout, official config (8 slots x 128, sel_slots [0, 3] of 11)
    # and the #129 head (19 slots x 13, anchor + predicted end, d_model 128; Decision P-other selection)
    "head-official-phyre": {"kind": "head", "readout": None, "frames": 11, "num_slots": 8, "slot_size": 128},
    "head-b1": {"kind": "head", "readout": dict(num_slots=19, slot_size=13, t_pe="sin", d_model=128, num_layers=4,
                                                num_heads=8, ffn_dim=512, norm_first=True, sel_slots=[0, 1]),
                "frames": 2},
}
READOUT_CONFIG = "slotformer/phyre_planning/configs/readout_phyre_params-fold0.py"
LABELS = (1.0, 0.0, 1.0, 0.0)
PHYRE_SHIM = '''"""b127 shim: the PHYRE package (py3.6 wheels only) is imported unconditionally by SlotFormer's dataset
package; no PHYRE code path runs in the CLEVRER evaluation or the G-port. Any attribute access raises."""


def __getattr__(name):
    raise ImportError(f"phyre is not installed in b127-slotformer (phyre.{name} requested)")
'''


def write_phyre_shim(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "phyre.py").write_text(PHYRE_SHIM)
    return directory


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arm_inputs(arm: str) -> dict:
    """Fixed inputs (seeded): slots + vid_len (official arm) or carriers [B, 286, 236], ends and launches."""
    spec = ARMS[arm]
    g = torch.Generator().manual_seed(SEED_INPUT)
    if spec["kind"] == "slots":
        slots = torch.randn(BATCH, spec["frames"], spec["num_slots"], spec["slot_size"], generator=g)
        return {"slots": slots, "vid_len": torch.tensor(spec["vid_len"])}
    if spec["kind"] == "head" and spec["readout"] is None:
        slots = torch.randn(BATCH, spec["frames"], spec["num_slots"], spec["slot_size"], generator=g)
        return {"slots": slots, "label": torch.tensor(LABELS)}
    carriers = torch.rand(BATCH, CARRIER_POSITIONS, 236, generator=g)
    carriers[:, :, 2 + 8::13] = torch.randn(BATCH, CARRIER_POSITIONS, 18, generator=g)  # motion columns, +-
    carriers[:, :, 2 + 9::13] = torch.randn(BATCH, CARRIER_POSITIONS, 18, generator=g)
    launch = torch.tensor([[-40 / 480, 5 / 480, 1.0, 0.0, 1.0], [-160 / 480, 80 / 480, 1.0, 0.0, 1.0],
                           [-80 / 480, 30 / 480, 1.0, 0.0, 1.0], [-120 / 480, 55 / 480, 1.0, 0.0, 1.0]])
    if spec["kind"] == "head":  # anchor and (predicted) frame-225 carriers + launch
        return {"anchor": carriers[:, 0], "end": carriers[:, 225], "launch5": launch, "label": torch.tensor(LABELS)}
    return {"carriers": carriers, "ends": torch.tensor(ENDS), "launch5": launch}


def frames_of(arm: str) -> int:
    spec = ARMS[arm]
    return spec["frames"] if "frames" in spec else 225 // spec["stride"] + 1


# ---------------------------------------------------------------------------------------------------------------
# reference (official code)
# ---------------------------------------------------------------------------------------------------------------

def official_batch(arm: str, inputs: dict) -> dict:
    """The official data_dict: PHYRE ``fix_video_len`` padding and ``vid_len`` of the strided real frames."""
    if ARMS[arm]["kind"] == "slots":
        return {"slots": inputs["slots"].clone(), "vid_len": inputs["vid_len"].clone()}
    if ARMS[arm]["kind"] == "head":
        if "slots" in inputs:
            return {"slots": inputs["slots"].clone(), "label": inputs["label"].clone()}
        both = torch.stack([inputs["anchor"], inputs["end"]], dim=1)  # [B, 2, 236]
        token = torch.cat([inputs["launch5"], torch.zeros(BATCH, 8)], dim=1)  # the declared launch token layout
        slots = torch.cat([both[:, :, 2:].reshape(BATCH, 2, 18, 13), token[:, None, None].expand(BATCH, 2, 1, 13)],
                          dim=2)
        return {"slots": slots.float(), "label": inputs["label"].clone()}
    from slotformer.base_slots.datasets.phyre import fix_video_len
    stride, frames = ARMS[arm]["stride"], frames_of(arm)
    slots, vid_len = [], []
    for b in range(BATCH):
        end = int(inputs["ends"][b])
        real = inputs["carriers"][b, 0:min(end, 225) + 1:stride].numpy()  # the shot's strided frames
        vid_len.append(min(len(real), frames))
        video = torch.from_numpy(fix_video_len(real, frames))  # [T, 236]
        carrier_slots = video[:, 2:].reshape(frames, 18, 13)
        token = torch.cat([inputs["launch5"][b], torch.zeros(8)])  # the declared launch token layout
        slots.append(torch.cat([carrier_slots, token.expand(frames, 1, 13)], dim=1))
    return {"slots": torch.stack(slots).float(), "vid_len": torch.tensor(vid_len)}


def official_params(arm: str, decoder_ckpt: Path):
    params = importlib.import_module("phyre_params").SlotFormerParams()  # loaded by path in reference()
    spec = ARMS[arm]
    if spec["kind"] == "carriers":
        frames = frames_of(arm)
        params.num_slots, params.slot_size = 19, 13
        params.slot_dict = dict(num_slots=19, slot_size=13)
        params.rollout_dict = dict(params.rollout_dict, num_slots=19, slot_size=13, **spec["size"])
        params.n_sample_frames = frames
        params.video_len = frames
        params.loss_dict = dict(params.loss_dict, rollout_len=frames - params.rollout_dict["history_len"])
        params.dec_dict = dict(params.dec_dict, dec_channels=(13, 64, 64, 64, 64), dec_ckp_path=str(decoder_ckpt))
    else:
        params.dec_dict = dict(params.dec_dict, dec_ckp_path=str(PRETRAINED / params.dec_dict["dec_ckp_path"]))
    return params


def random_decoder(params, path: Path) -> Path:
    """A seeded random frozen decoder of the arm's slot width, built by the official StoSAVi._build_decoder."""
    from slotformer.base_slots.models import StoSAVi
    ns = types.SimpleNamespace(slot_size=params.slot_dict["slot_size"], dec_dict=params.dec_dict,
                               resolution=params.resolution)
    torch.manual_seed(SEED_DECODER)
    StoSAVi._build_decoder(ns)
    state = {f"decoder.{k}": v for k, v in ns.decoder.state_dict().items()}
    state.update({f"decoder_pos_embedding.{k}": v for k, v in ns.decoder_pos_embedding.state_dict().items()})
    torch.save({"state_dict": state}, path)
    return path


def reference(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    write_phyre_shim(out_dir / "shims")
    sys.path[:0] = [str(out_dir / "shims"), str(SLOTFORMER_REF)]
    # the config module names have a '-': load them by path as 'phyre_params' / 'readout_params'
    for name, rel in (("phyre_params", PHYRE_CONFIG), ("readout_params", READOUT_CONFIG)):
        spec = importlib.util.spec_from_file_location(name, SLOTFORMER_REF / rel)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    from slotformer.video_prediction.models import build_model
    from slotformer.video_prediction.method import SlotFormerMethod
    from slotformer.phyre_planning.models import build_model as build_readout
    from slotformer.phyre_planning.method import PHYREReadoutMethod

    torch.set_num_threads(8)
    for arm in ARMS:
        head = ARMS[arm]["kind"] == "head"
        if head:
            params = importlib.import_module("readout_params").SlotFormerParams()
            if ARMS[arm]["readout"] is not None:
                params.readout_dict = dict(ARMS[arm]["readout"])
            torch.manual_seed(SEED_INIT)
            model = build_readout(params)
            method_class, output_key = PHYREReadoutMethod, "logits"
        else:
            params = official_params(arm, out_dir / f"decoder-{arm}.pt")
            if ARMS[arm]["kind"] == "carriers":
                random_decoder(params, out_dir / f"decoder-{arm}.pt")
            torch.manual_seed(SEED_INIT)
            model = build_model(params)
            method_class, output_key = SlotFormerMethod, "pred_slots"
        trainable = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        inputs = arm_inputs(arm)
        batch = official_batch(arm, inputs)
        # forward (eval: dropout off) on the fixed batch
        model.eval()
        with torch.no_grad():
            pred = model(dict(batch))[output_key].clone()
        model.train()

        method = method_class.__new__(method_class)
        torch.nn.Module.__init__(method)  # nerv BaseMethod is an nn.Module; its __init__ needs CUDA + loaders
        method.params, method.device = params, torch.device("cpu")
        method.model = torch.nn.parallel.DataParallel(model, device_ids=[])  # nerv: "only to enable .module"
        params.max_epochs = MAX_EPOCHS
        method.train_loader = range(TOTAL_STEPS // MAX_EPOCHS)
        method.iter_train_loader = method.train_loader
        method._setup_optimizer()
        method.use_fp16, method.grad_accum_steps, method.epoch_it, method.it = False, 1, 0, 0
        method.clip_grad, method.save_iter, method.use_ddp = params.clip_grad, 10 ** 9, False
        if not head:  # the PHYRE config has use_loss_decay False: _training_step_start sets the factor to 1
            model.loss_decay_factor = 1.
        captured = {}

        def snapshot(optimizer, *_):
            captured["grads"] = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                                 if p.grad is not None}

        method.optimizer.register_step_pre_hook(snapshot)
        steps = []
        for index, seed in enumerate((SEED_STEP, SEED_STEP2)):
            if index == 1:  # advance to the end of warm-up: every group at its base (max) LR
                while method.scheduler.step_in_cycle < method.scheduler.warmup_steps:
                    method.scheduler.step()
            lrs_at = [group["lr"] for group in method.optimizer.param_groups]
            torch.manual_seed(seed)
            out = method._training_step(dict(batch))
            method._training_step_end()
            grads = captured.pop("grads")
            steps.append({
                "seed": seed, "lrs_at_step": lrs_at,
                "scalars": {k: float(v) for k, v in out.items() if k.endswith("loss")},
                "grads": grads,
                "grad_norm": float(torch.norm(torch.stack([g.norm() for g in grads.values()]))),
                "post_params": {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad},
                "optimizer_state": {n: {k: (v.detach().clone() if torch.is_tensor(v) else v)
                                        for k, v in method.optimizer.state[p].items()}
                                    for n, p in model.named_parameters() if p in method.optimizer.state},
                "lrs_after_step": [group["lr"] for group in method.optimizer.param_groups],
            })
        optimizer_defaults = {k: v for k, v in method.optimizer.defaults.items()
                              if isinstance(v, (int, float, tuple, bool)) and v is not None}
        torch.save({"arm": arm, "spec": ARMS[arm], "init_trainable": trainable, "inputs": inputs, "batch": batch,
                    "prediction": pred, "steps": steps,
                    "optimizer_class": type(method.optimizer).__name__, "optimizer_defaults": optimizer_defaults,
                    "scheduler": {"class": type(method.scheduler).__name__,
                                  "first_cycle_steps": method.scheduler.first_cycle_steps,
                                  "warmup_steps": method.scheduler.warmup_steps,
                                  "max_lr": list(method.scheduler.base_max_lr),
                                  "min_lr": list(method.scheduler.min_lr)},
                    "trainable_parameters": sum(p.numel() for p in trainable.values()),
                    "torch": torch.__version__, "slotformer_ref": str(SLOTFORMER_REF), "python": sys.executable,
                    "nerv": __import__("nerv").__file__,
                    "seeds": [SEED_INIT, SEED_INPUT, SEED_STEP, SEED_STEP2, SEED_DECODER],
                    "total_steps": TOTAL_STEPS},
                   out_dir / f"reference-{arm}.pt")
        print(f"[b127-slotformer-port-reference] {arm}: loss {steps[0]['scalars']['loss']:.6f} at "
              f"{steps[0]['lrs_at_step']}, step-2 loss {steps[1]['scalars']['loss']:.6f} at "
              f"{steps[1]['lrs_at_step']}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# compare (the NovPhy adapter, vendored code)
# ---------------------------------------------------------------------------------------------------------------

def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def vendor_check() -> dict:
    vendor = ROOT / "world_model/baselines/slotformer/vendor"
    record = json.loads((vendor / "VENDOR.json").read_text())
    rows, ok = {}, True
    for rel, entry in record["files"].items():
        vendored = sha256(vendor / rel)
        row = {"vendored_match": vendored == entry["vendored_sha256"]}
        if entry["upstream_path"]:
            base = SLOTFORMER_REF if "upstream" not in entry else \
                SLOTFORMER_REF.parent / f"nerv@{entry['commit'][:7]}"
            upstream = base / entry["upstream_path"]
            row["upstream_match"] = upstream.exists() and sha256(upstream) == entry["upstream_sha256"]
            row["verbatim"] = (entry["upstream_sha256"] == entry["vendored_sha256"]) == (not entry["edits"])
        ok = ok and all(row.values())
        rows[rel] = row
    return {"files": rows, "pass": ok, "commit": record["commit"]}


def compare(out_dir: Path):
    sys.path.insert(0, str(ROOT))
    from world_model.baselines.slotformer import novphy as sf

    torch.set_num_threads(8)
    arms = {}
    for arm, spec in ARMS.items():
        ref = torch.load(out_dir / f"reference-{arm}.pt", weights_only=False)
        inputs = ref["inputs"]
        head = spec["kind"] == "head"
        if head:
            readout = sf.HEAD_DICT if spec["readout"] is not None else dict(
                num_slots=8, slot_size=128, t_pe="sin", d_model=128, num_layers=4, num_heads=8, ffn_dim=512,
                norm_first=True, sel_slots=[0, 3])  # readout_phyre_params-fold0.py
            model = sf.build_head(readout)
            if "slots" in inputs:
                batch = {"slots": inputs["slots"].clone(), "label": inputs["label"].clone()}
            else:
                batch = {"slots": sf.head_input(inputs["anchor"], inputs["end"], inputs["launch5"]),
                         "label": inputs["label"].clone()}
        elif spec["kind"] == "slots":
            model = sf.SlotFormerE99(size=dict(d_model=256, num_layers=8, num_heads=8, ffn_dim=1024),
                                     num_slots=spec["num_slots"], slot_size=spec["slot_size"],
                                     rollout_len=spec["frames"] - 1)
            batch = {"slots": inputs["slots"].clone(), "vid_len": inputs["vid_len"].clone()}
        else:
            model = sf.SlotFormerE99(size=spec["size"], stride=spec["stride"])
            carriers = inputs["carriers"]

            def carrier_rows(shots, positions, carriers=carriers):
                return carriers[torch.as_tensor(shots), torch.as_tensor(positions)]

            batch = sf.training_batch(carrier_rows, range(BATCH), inputs["ends"], inputs["launch5"], spec["stride"])
        batch_equal = all(torch.equal(batch[k], ref["batch"][k]) for k in ref["batch"])
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        names_equal = sorted(names) == sorted(ref["init_trainable"])
        model.load_state_dict({**model.state_dict(), **ref["init_trainable"]})
        model.eval()
        with torch.no_grad():
            pred = model(dict(batch))["logits" if head else "pred_slots"]
        model.train()
        optimizer, scheduler = (sf.build_optimizer(model, TOTAL_STEPS, sf.HEAD_LR, sf.HEAD_WARMUP_PCT) if head
                                else sf.build_optimizer(model, TOTAL_STEPS))
        total_loss = (lambda b: sf.head_total_loss(model, b)) if head else model.total_loss
        steps = []
        for index, rstep in enumerate(ref["steps"]):
            if index == 1:
                while scheduler.step_in_cycle < scheduler.warmup_steps:
                    scheduler.step()
            lrs_at = [group["lr"] for group in optimizer.param_groups]
            torch.manual_seed(rstep["seed"])
            loss, out = total_loss(dict(batch))
            loss.backward()
            grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
            grad_norm = float(torch.norm(torch.stack([g.norm() for g in grads.values()])))
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            post = {n: p.detach() for n, p in model.named_parameters()}
            state = {n: optimizer.state[p] for n, p in model.named_parameters() if p in optimizer.state}
            close = [torch.isclose(post[n], rstep["post_params"][n], rtol=TOLERANCE["param_rtol"],
                                   atol=TOLERANCE["param_atol"]) for n in rstep["post_params"]]
            moments = [relative_l2(state[n][k], rstep["optimizer_state"][n][k])
                       for n in rstep["optimizer_state"] for k in ("exp_avg", "exp_avg_sq")]
            row = {
                "lrs_at_step": lrs_at, "official_lrs_at_step": rstep["lrs_at_step"],
                "lrs_after_step": [g["lr"] for g in optimizer.param_groups],
                "official_lrs_after_step": rstep["lrs_after_step"],
                "loss": float(loss), "official_loss": rstep["scalars"]["loss"],
                "loss_relative_delta": abs(float(loss) - rstep["scalars"]["loss"]) / abs(rstep["scalars"]["loss"]),
                "loss_terms": {k: float(v) for k, v in out.items() if k.endswith("_loss")},
                "grad_norm": grad_norm, "official_grad_norm": rstep["grad_norm"],
                "grad_norm_relative_delta": abs(grad_norm - rstep["grad_norm"]) / rstep["grad_norm"],
                "grad_tensors_equal_set": sorted(grads) == sorted(rstep["grads"]),
                "grad_tensor_max_relative_l2": max(relative_l2(grads[n], rstep["grads"][n]) for n in rstep["grads"]
                                                   if rstep["grads"][n].norm() > 0),
                "param_max_abs_delta": max(float((post[n] - rstep["post_params"][n]).abs().max())
                                           for n in rstep["post_params"]),
                "param_close_share_isclose": float(torch.cat([c.flatten() for c in close]).float().mean()),
                # the #123 rule: share of elements with |delta| <= param_close_abs (absolute)
                "param_close_share_abs": float(torch.cat([((post[n] - rstep["post_params"][n]).abs()
                                                           <= TOLERANCE["param_close_abs"]).flatten()
                                                          for n in rstep["post_params"]]).float().mean()),
                "parameter_mismatch": {
                    "elements": int(sum(int((~c).sum()) for c in close)),
                    # the official gradient magnitude at the mismatched elements, against the global norm
                    "official_grad_max_abs": max([float(rstep["grads"][n][~c].abs().max())
                                                  for n, c in zip(rstep["post_params"], close)
                                                  if n in rstep["grads"] and bool((~c).any())], default=0.0),
                    "official_grad_norm": rstep["grad_norm"]},
                "adam_moment_max_relative_l2": max(moments),
                "param_moved_max_abs": max(float((rstep["post_params"][n] - (ref["init_trainable"][n] if index == 0
                                                  else ref["steps"][0]["post_params"][n])).abs().max())
                                           for n in rstep["post_params"]),
            }
            common = (row["lrs_at_step"] == row["official_lrs_at_step"]
                      and row["lrs_after_step"] == row["official_lrs_after_step"]
                      and row["loss_relative_delta"] <= TOLERANCE["loss_relative"]
                      and row["grad_norm_relative_delta"] <= TOLERANCE["grad_norm_relative"]
                      and row["grad_tensors_equal_set"]
                      and row["grad_tensor_max_relative_l2"] <= TOLERANCE["tensor_relative_l2"]
                      and row["adam_moment_max_relative_l2"] <= TOLERANCE["tensor_relative_l2"]
                      and row["param_max_abs_delta"] <= (TOLERANCE["param_max_abs_over_lr"]
                                                         * max(row["lrs_after_step"]) + TOLERANCE["param_close_abs"])
                      and row["param_moved_max_abs"] > 0)
            row["pass_123_exact"] = bool(common and row["param_close_share_abs"] >= TOLERANCE["param_close_share"])
            row["pass"] = bool(common and row["param_close_share_isclose"] >= TOLERANCE["param_close_share"])
            steps.append(row)
        # diagnostic (not gating): step 2 alone, started from the OFFICIAL step-1 parameters and Adam state, so the
        # step-1 differences do not propagate; tells propagated from per-step float differences
        model.load_state_dict({**model.state_dict(), **ref["steps"][0]["post_params"]})
        optimizer, scheduler = (sf.build_optimizer(model, TOTAL_STEPS, sf.HEAD_LR, sf.HEAD_WARMUP_PCT) if head
                                else sf.build_optimizer(model, TOTAL_STEPS))
        for name, p in model.named_parameters():
            if p.requires_grad:
                optimizer.state[p] = {k: (v.clone() if torch.is_tensor(v) else v)
                                      for k, v in ref["steps"][0]["optimizer_state"][name].items()}
        scheduler.step()
        while scheduler.step_in_cycle < scheduler.warmup_steps:
            scheduler.step()
        model.train()
        torch.manual_seed(ref["steps"][1]["seed"])
        loss2, _ = total_loss(dict(batch))
        loss2.backward()
        rg = ref["steps"][1]["grads"]
        g2 = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        optimizer.step()
        optimizer.zero_grad()
        d2 = torch.cat([(p.detach() - ref["steps"][1]["post_params"][n]).abs().flatten()
                        for n, p in model.named_parameters() if p.requires_grad])
        step2_alone = {"loss_relative_delta": abs(float(loss2) - ref["steps"][1]["scalars"]["loss"])
                       / abs(ref["steps"][1]["scalars"]["loss"]),
                       "grad_tensor_max_relative_l2": max(relative_l2(g2[n], rg[n]) for n in rg if rg[n].norm() > 0),
                       "param_close_share_abs": float((d2 <= TOLERANCE["param_close_abs"]).float().mean()),
                       "param_max_abs_delta": float(d2.max())}
        defaults = {k: v for k, v in optimizer.defaults.items()
                    if isinstance(v, (int, float, tuple, bool)) and v is not None}
        scheduler_equal = (type(scheduler).__name__ == ref["scheduler"]["class"]
                           and scheduler.first_cycle_steps == ref["scheduler"]["first_cycle_steps"]
                           and scheduler.warmup_steps == ref["scheduler"]["warmup_steps"]
                           and list(scheduler.base_max_lr) == ref["scheduler"]["max_lr"]
                           and list(scheduler.min_lr) == ref["scheduler"]["min_lr"])
        record = {
            "arm": arm, "spec": spec, "torch": torch.__version__, "official_torch": ref["torch"],
            "batch_equal": batch_equal, "parameter_names_equal": names_equal,
            "trainable_parameters": sf.trainable_parameters(model),
            "official_trainable_parameters": ref["trainable_parameters"],
            "prediction_relative_l2": relative_l2(pred, ref["prediction"]),
            "prediction_max_abs_delta": float((pred - ref["prediction"]).abs().max()),
            "optimizer_class": type(optimizer).__name__, "official_optimizer_class": ref["optimizer_class"],
            "optimizer_defaults": defaults, "official_optimizer_defaults": ref["optimizer_defaults"],
            "scheduler": ref["scheduler"], "scheduler_equal": scheduler_equal,
            "steps": steps, "step2_from_official_step1_state": step2_alone,
        }
        record_common = (batch_equal and names_equal
                         and record["trainable_parameters"] == record["official_trainable_parameters"]
                         and record["prediction_relative_l2"] <= TOLERANCE["prediction_relative_l2"]
                         and record["optimizer_class"] == record["official_optimizer_class"]
                         and defaults == ref["optimizer_defaults"] and scheduler_equal)
        record["pass_123_exact"] = bool(record_common and all(s["pass_123_exact"] for s in steps))
        record["pass"] = bool(record_common and all(s["pass"] for s in steps))
        arms[arm] = record
        print(f"[b127-slotformer-gport] {arm}: pass_123_exact {record['pass_123_exact']} "
              f"(isclose variant {record['pass']}; loss delta "
              f"{steps[0]['loss_relative_delta']:.2e} / {steps[1]['loss_relative_delta']:.2e}, grad rel L2 "
              f"{steps[0]['grad_tensor_max_relative_l2']:.2e} / {steps[1]['grad_tensor_max_relative_l2']:.2e})",
              flush=True)
    vendor = vendor_check()
    gport = {"schema": "issue_127_gport_v1", "issue": 129, "arm_family": "B1 SlotFormer-E99",
             "upstream": "https://github.com/pairlab/SlotFormer", "commit": vendor["commit"],
             "tolerance": TOLERANCE, "total_steps": TOTAL_STEPS, "batch": BATCH,
             "seeds": {"init": SEED_INIT, "input": SEED_INPUT, "step1": SEED_STEP, "step2": SEED_STEP2,
                       "decoder": SEED_DECODER},
             "procedure": __doc__, "vendor": vendor,
             "references": {arm: sha256(out_dir / f"reference-{arm}.pt") for arm in ARMS},
             "reference_env": sorted({(r["official_torch"]) for r in arms.values()}),
             "reference_python": sorted({torch.load(out_dir / f"reference-{a}.pt", weights_only=False)["python"]
                                         for a in ARMS}),
             "adapter_torch": torch.__version__,
             "script": sha256(Path(__file__)),
             "adapter": sha256(ROOT / "world_model/baselines/slotformer/novphy.py"),
             "arms": arms,
             "criterion": ("G_port (gating) = #123's rule exactly (run_hjepa_comparison.GPORT_TOLERANCE): per step the "
                           "share of trainable elements with |delta param| <= 1e-7 is >= 0.999 and max |delta| <= "
                           "2 lr_max + 1e-7, plus loss / gradient / Adam-moment / LR / optimizer / scheduler / count / "
                           "batch / prediction equality. G_port_isclose_variant (diagnostic, not gating): the same with "
                           "the per-element test isclose(rtol 1e-5, atol 1e-6)."),
             "G_port_isclose_variant": vendor["pass"] and all(a["pass"] for a in arms.values()),
             "G_port": vendor["pass"] and all(a["pass_123_exact"] for a in arms.values())}
    diagnostic = out_dir / "torch271" / "gport.json"
    if diagnostic.is_file():
        diagnostic = diagnostic.resolve()
        old = json.loads(diagnostic.read_text())
        gport["diagnostic_torch_2_7_1"] = {
            "artifact": str(diagnostic.relative_to(ROOT)), "sha256": sha256(diagnostic),
            "reference_torch": "2.7.1+cu128 (env b127-slotformer)",
            "G_port_123_exact": old.get("G_port_123_exact"), "G_port_isclose_variant": old.get("G_port"),
            "note": "the earlier comparison against the official code on torch 2.7.1; kept as a recorded diagnostic"}
    gport["G_port_rule"] = "issue_123_exact"
    (out_dir / "gport.json").write_text(json.dumps(gport, indent=2, sort_keys=True) + "\n")
    print(f"[b127-slotformer-gport] G-port {'PASS' if gport['G_port'] else 'FAIL'} -> {out_dir / 'gport.json'}")
    return 0 if gport["G_port"] else 1


if __name__ == "__main__":
    mode, out = sys.argv[1], Path(sys.argv[2])
    if mode == "reference":
        reference(out)
    elif mode == "compare":
        sys.exit(compare(out))
    else:
        raise SystemExit(f"unknown mode {mode!r} (reference | compare)")
