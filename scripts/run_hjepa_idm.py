"""Issue-125: H-JEPA with its own inverse-dynamics remedy on NovPhy (pre-registered follow-up to #123).

Binding runner module: scripts/run_hjepa_idm.py
Exact validation command: python -u -m scripts.run_hjepa_idm --validate

#125 asks whether H-JEPA with the paper's anti-collapse remedy (inverse-dynamics term + action-SIGReg, H-JEPA §4.3
Eq. 8, App. K, the DROID recipe) ranks the Gate-B grid candidates of the sealed #109 evaluation split better or
worse than the Gate-B unit U. Only the IDM terms change from #123's frozen plan (world_model/hjepa/idm.py); data,
splits, layout, sampler, launch encoding, lambda, optimizer, schedule, readouts, cost, cells, bootstrap and guards are
#123's, imported from scripts/run_hjepa_comparison.py (hash-pinned, not edited) and #123's artefacts. This runner

1. --gport-configs / --gport-reference / --gport (CPU): G-port-IDM. The official code in the hjepa env
   (scripts/hjepa_idm_port_reference.py) and this port (vendored modules + idm.py + #123's optimizer / scheduler)
   on fixed CPU inputs: forward pass with the IDM and action-SIGReg terms, the step-0-LR step and a positive-LR step,
   compared under #123's declared GPORT_TOLERANCE (gport-idm.json);
2. --trigger (GPU, fit split only): the frozen trigger. G-collapse (top-level effective rank >= 0.1 D, within-episode
   variance share >= 0.05) and G-action (endpoint launch sensitivity >= 0.05) of the trained #123 HJ-px and HJ-car-m
   checkpoints (models/<arm>/seed-<s>/model.pt; never resume.pt) on the fit probe frames and the 480 fit anchors,
   with #123's readout code and guard expressions and thresholds; triggered iff any seed of either arm fails;
   trigger.json. No evaluation file is read;
3. --selection (GPU): gamma_1 in {100, 225} per input type (HJ-px-IDM, HJ-car-m-IDM at B1, seed 20260908; #123's
   readouts; policy-split pooled grid AUC under R1; ties at 4 d.p. -> 100);
4. --smoke (GPU): trigger, G-port-IDM, CPU tests, throughput, parameter matching, the gamma_1 choice and the
   projection against the 120 GPU-h cap (and the 48 h / 4 GPU stop rule); smoke.json. --selection and --smoke exit 0
   at once with a recorded not_triggered when trigger.json says false;
5. --prepare freezes plan.json (agent-committed freeze); --train (per GPU queue; SIGTERM -> resume.pt at the next
   optimizer step, exit 75, #123's mechanism); --readouts (#123's fit_readouts); --freeze-readouts;
6. --score, once, under deterministic_scoring(): every IDM arm-seed through #123's score_arm_seed, the wrong-anchor
   control of the primary arms, the U / C* position curves;
7. --publish: records-manifest.json, compute.json, summary.json, findings.md (PI1 / PI2 / SI1-SI8, guards, delta-rule
   tokens, dispositions 1-4; every token is the stop token until --validate binds a passing re-score); --validate.

Every interval is a member-clustered percentile bootstrap (10000 draws, PCG64 7201), DESCRIPTIVE. Engine seconds 0.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import torch

from scripts import run_hjepa_comparison as rc
from world_model.hjepa import idm
from world_model.hjepa import novphy as hn

ROOT = rc.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-125-hjepa-idm-v1"
SOURCE = rc.OUTPUT  # #123: data, trained checkpoints, frozen plan, published records
IDENTITY = "issue-125-hjepa-idm-v1"
SCHEMA_PLAN = "issue_125_plan_v1"
SCHEMA_RECORD = "issue_125_record_v1"
RUNNER = "scripts/run_hjepa_idm.py"
MODULE = "world_model/hjepa/idm.py"
REFERENCE_SCRIPT = "scripts/hjepa_idm_port_reference.py"
TESTS = "tests/test_issue_125_hjepa_idm.py"
VALIDATION_COMMAND = "python -u -m scripts.run_hjepa_idm --validate"
HJEPA_PYTHON = Path.home() / "miniconda3/envs/hjepa/bin/python"
SEEDS = rc.SEEDS
SELECTION_SEED = rc.SELECTION_SEED
TRIGGER_ARMS = ("HJ-px", "HJ-car-m")
WRONG_ANCHOR_ARMS = idm.PRIMARY
NOT_TRIGGERED = "not_triggered"
CAP_GPU_HOURS = 120.0
STOP_WALL_HOURS = 48.0  # round-2 conventions: projected wall over 48 h on 4 GPUs -> stop and report
GPUS = rc.GPUS
NIGHT_HOURS = 12 + 50 / 60  # 18:00 -> 06:50 graceful stop
SMOKE_STEPS = rc.SMOKE_STEPS
GPORT_LAMBDA = 0.18
GPORT_GAMMA1 = 100.0
GPORT_CROPS = {"H3": (0, 2, None), "L1": (None,)}  # level-1 crop at 0: the launch transition is an IDM target
DELTA = rc.DELTA
STOP_TOKEN = rc.STOP_TOKEN
CONTRASTS = {
    "PI1": ("HJ-px-IDM", "U"), "PI2": ("HJ-car-m-IDM", "U"),
    "SI1": ("HJ-px-IDM", "HJ-px"), "SI2": ("HJ-car-m-IDM", "HJ-car-m"),
    "SI3": ("HJ-px-IDM", "LeWM-px-IDM"), "SI4": ("HJ-car-m-IDM", "LeWM-car-m-IDM"),
    "SI5_pixels": ("HJ-px-IDM", "C*"), "SI5_carriers": ("HJ-car-m-IDM", "C*"),
    "SI6_pixels": ("HJ-px-IDM", "P"), "SI6_carriers": ("HJ-car-m-IDM", "P"),
    "SI7_pixels_J": ("HJ-px-IDM", "J"), "SI7_pixels_J_dec": ("HJ-px-IDM", "J_dec"),
    "SI7_carriers_J": ("HJ-car-m-IDM", "J"), "SI7_carriers_J_dec": ("HJ-car-m-IDM", "J_dec"),
}
ISSUE_123_ARMS = ("HJ-px", "HJ-car-m")  # SI1 / SI2 read #123's published records of these arms
ISSUE_123_GLOBAL_GUARDS = ("G_isolation", "G_chronology", "G_scoring_policy", "G_records")  # required in compute.json
REGISTRY = "scripts/post_freeze_amendments.py"  # shared amendment registry (see load_plan, A1)

read_json = rc.read_json
write_json = rc.write_json
json_text = rc.json_text
sha256_of = rc.sha256_of
atomic_torch = rc.atomic_torch
utc_now = rc.utc_now
mark = rc.mark


def log(message):
    print(f"[issue-125] {message}", flush=True)


def source_lambda(input_kind):
    """#123's frozen SIGReg lambda of the input type (0.18 for both)."""
    return float(read_json(SOURCE / "plan.json")["selected_lambda"][input_kind])


# ---------------------------------------------------------------------------
# G-port-IDM (CPU): the official code with the remedy vs this port, #123's protocol and tolerances
# ---------------------------------------------------------------------------

def gport_dir(output):
    return Path(output) / "gport-idm"


def gport_configs(output):
    configs = {}
    for arm in idm.ARMS:
        torch.manual_seed(0)
        model, _, cfg = idm.build_model(arm, GPORT_LAMBDA, GPORT_GAMMA1, checkpointing=False)
        dims = {}
        for level in range(1, int(cfg.num_levels) + 1):
            state_dim, action_dim = idm.head_dims(model, level)
            dims[str(level)] = [state_dim, idm.IDM_HIDDEN, action_dim]
        configs[arm] = {"config": cfg, "idm": dims, "input": hn.ARMS[arm]["input"],
                        "level1_steps": hn.Geometry(arm).level1_steps,
                        "crop_starts": list(GPORT_CROPS[hn.ARMS[arm]["layout"]])}
    path = gport_dir(output) / "configs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_text(configs))
    log(f"G-port-IDM configs written: {path}")
    return 0


def gport_reference(output):
    """The official side, in the hjepa env, CPU only."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run([str(HJEPA_PYTHON), str(ROOT / REFERENCE_SCRIPT), str(gport_dir(output) / "configs.json"),
                           str(gport_dir(output))], env=env, cwd=ROOT)
    return proc.returncode


def gport_arm(reference):
    """#123's gport_arm with the IDM model and config: the step at the step-0 LR (forward incl. the IDM and
    action-SIGReg terms, gradients, clip, AdamW, scheduler), then #123's gport_step2 (the positive-LR step)."""
    arm = reference["arm"]
    geometry = hn.Geometry(arm)
    starts = {level: s for level, s in enumerate(reference["crop_starts"], start=1)}
    needed = geometry.needed(starts)
    model, losses, cfg = idm.build_model(arm, GPORT_LAMBDA, GPORT_GAMMA1)
    model.load_state_dict(reference["init_state"])
    model.plan = (geometry.level1_steps, starts, needed)
    wrapper = hn.Wrapper(model, losses).train()
    x = reference["inputs"]["x"]
    batch = {"pixels_level1": x[:, needed].clone(), "action_level1": reference["inputs"]["action"].clone()}
    optimizer = hn.make_optimizer(model, cfg)
    scheduler = hn.WarmupCosine(optimizer, hn.warmup_steps(reference["total_steps"]), reference["total_steps"])
    torch.manual_seed(reference["seeds"][2])
    output = hn.hjepa_utils.hjepa_forward(wrapper, batch, "fit", cfg, normalize_batch=hn.normalize_uint8_images_on_device)
    scalars = {k: float(v) for k, v in output.items() if "loss" in k and torch.is_tensor(v) and v.ndim == 0}
    output["loss"].backward()
    params = [p for group in optimizer.param_groups for p in group["params"]]
    grads = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
    norm = float(torch.nn.utils.clip_grad_norm_(params, hn.GRAD_CLIP))
    optimizer.step()
    scheduler.step()
    tol = rc.GPORT_TOLERANCE
    official = reference["scalars"]
    loss_rel = max(abs(scalars[k] - v) / max(1.0, abs(v)) for k, v in official.items())
    embed_abs = max(float((output[k].detach() - v).abs().max()) for k, v in reference["embeds"].items()
                    if k in output and output[k].shape == v.shape)
    floor = tol["grad_floor_of_global_norm"] * reference["grad_norm"]
    grad_rel, grad_null_abs, null_tensors = rc.per_tensor_l2(grads, reference["grads"], floor)
    heads = {k: v for k, v in reference["grads"].items() if idm.is_head(k)}
    head_rel, head_null_abs, head_nulls = rc.per_tensor_l2(grads, heads, floor)
    post = model.state_dict()
    deltas = torch.cat([(post[k].float() - v.float()).abs().flatten() for k, v in reference["post_state"].items()
                        if v.is_floating_point() and v.numel() and post[k].shape == v.shape])
    close_share = float((deltas <= tol["param_close_abs"]).float().mean())
    lrs = [group["lr"] for group in optimizer.param_groups]
    lr_max = max(lrs)
    terms = sorted(k for k in official if k.startswith(("idm_loss", "sigreg_loss_action")))
    expected_terms = sorted([f"idm_loss_level{level}" for level in range(1, int(cfg.num_levels) + 1)]
                            + [f"sigreg_loss_action_level{level}" for level in range(2, int(cfg.num_levels) + 1)])
    first = {k: v.detach().clone() for k, v in post.items()}
    group_heads = {group["name"]: sum(1 for p in group["params"] for n, q in model.named_parameters()
                                      if q is p and idm.is_head(n)) for group in optimizer.param_groups}
    out = {"arm": arm, "gamma1": GPORT_GAMMA1, "lambda": GPORT_LAMBDA, "needed_frames": needed,
           "needed_equal": needed == reference["official_needed_frames"], "drawn_crop_starts": reference["drawn"],
           "loss": scalars.get("loss"), "official_loss": official["loss"],
           "idm_terms": {k: {"port": scalars.get(k), "official": official[k]} for k in terms},
           "idm_terms_expected": expected_terms, "idm_terms_present": terms == expected_terms,
           "head_tensors_in_optimizer_groups": group_heads,
           "loss_max_relative_delta": loss_rel, "embed_max_abs_delta": embed_abs,
           "grad_norm": norm, "official_grad_norm": reference["grad_norm"],
           "grad_norm_relative_delta": abs(norm - reference["grad_norm"]) / max(reference["grad_norm"], 1e-12),
           "grad_tensor_max_relative_l2": grad_rel, "null_gradient_tensors": null_tensors,
           "null_gradient_max_abs_l2": grad_null_abs, "null_gradient_floor": floor,
           "head_grad_tensors": len(heads), "head_grad_max_relative_l2": head_rel, "head_null_tensors": head_nulls,
           "param_close_share": close_share, "param_max_abs_delta": float(deltas.max()), "lr_max": lr_max,
           "lrs_after_step": lrs, "official_lrs_after_step": reference["lrs_after_step"],
           "optimizer_defaults": {k: optimizer.defaults[k] for k in ("lr", "betas", "eps", "weight_decay", "amsgrad")},
           "official_optimizer_defaults": {k: reference["optimizer_defaults"].get(k)
                                           for k in ("lr", "betas", "eps", "weight_decay", "amsgrad")},
           "official_torch": reference["torch"], "torch": torch.__version__}
    out["optimizer_equal"] = all(
        tuple(out["optimizer_defaults"][k]) == tuple(out["official_optimizer_defaults"][k] or ())
        if isinstance(out["optimizer_defaults"][k], tuple)
        else out["optimizer_defaults"][k] == out["official_optimizer_defaults"][k] for k in out["optimizer_defaults"])
    heads_in_groups = sum(group_heads.values()) == len([n for n, _ in model.named_parameters() if idm.is_head(n)]) > 0
    out["pass"] = bool(out["needed_equal"] and out["optimizer_equal"] and out["idm_terms_present"] and heads_in_groups
                       and len(heads) > 0 and loss_rel <= tol["loss_relative"] and embed_abs <= tol["embed_max_abs"]
                       and out["grad_norm_relative_delta"] <= tol["grad_norm_relative"]
                       and grad_rel <= tol["grad_tensor_relative_l2"] and grad_null_abs <= floor
                       and close_share >= tol["param_close_share"]
                       and out["param_max_abs_delta"] <= tol["param_max_abs_over_lr"] * lr_max + tol["param_close_abs"]
                       and max(abs(a - b) for a, b in zip(lrs, reference["lrs_after_step"])) <= tol["lr_abs"])
    batch2 = {"pixels_level1": x[:, needed].clone(), "action_level1": reference["inputs"]["action"].clone()}
    out["step2"] = rc.gport_step2(reference["step2"], hn, model, wrapper, cfg, batch2, optimizer, scheduler, params,
                                  first, reference["post_state"], reference["total_steps"])
    step2_terms = {k: v for k, v in reference["step2"]["scalars"].items() if k.startswith(("idm_loss", "sigreg_loss_action"))}
    out["step2"]["idm_terms_official"] = step2_terms
    out["pass"] = bool(out["pass"] and out["step2"]["pass"])
    return out


def gport(output):
    """Compare every reference arm; writes gport-idm.json (passes iff the vendored files match their pins and every
    arm passes both steps)."""
    torch.set_num_threads(8)
    vendor = read_json(ROOT / "world_model/hjepa/vendor/VENDOR.json")
    hashes = {rel: sha256_of(ROOT / "world_model/hjepa/vendor" / rel) == entry["vendored_sha256"]
              for rel, entry in vendor["files"].items()}
    arms = {}
    for arm in idm.ARMS:
        path = gport_dir(output) / f"reference-{arm}.pt"
        reference = torch.load(path, map_location="cpu", weights_only=False)
        arms[arm] = {**gport_arm(reference), "reference_sha256": sha256_of(path)}
        log(f"G-port-IDM {arm}: pass {arms[arm]['pass']} (loss {arms[arm]['loss']:.6f} vs {arms[arm]['official_loss']:.6f}; "
            f"IDM terms {arms[arm]['idm_terms']}; positive-LR step pass {arms[arm]['step2']['pass']})")
    result = {"schema": "issue_125_gport_idm_v1", "checked_at": utc_now(), "tolerance": rc.GPORT_TOLERANCE,
              "tolerance_source": "scripts/run_hjepa_comparison.py GPORT_TOLERANCE (#123's declared tolerances)",
              "official": {"code": "kevinghst/H-JEPA @ 97054ec (hjepa env)", "script": REFERENCE_SCRIPT,
                           "script_sha256": sha256_of(ROOT / REFERENCE_SCRIPT)},
              "port": {"module": MODULE, "module_sha256": sha256_of(ROOT / MODULE)},
              "vendor_hashes_match": hashes, "arms": arms, "device": "cpu float32",
              "pass": bool(all(hashes.values()) and all(a["pass"] for a in arms.values()))}
    write_json(Path(output) / "gport-idm.json", result)
    log(f"G-port-IDM: {'PASS' if result['pass'] else 'FAIL'}")
    return 0 if result["pass"] else 1


# ---------------------------------------------------------------------------
# training: #123's batches, loop and SIGTERM checkpoint (exit 75) with the IDM model
# ---------------------------------------------------------------------------

def run_dir(output, arm, seed):
    return rc.run_dir(output, arm, seed)


def selection_dir(output, arm, gamma1):
    return Path(output) / "selection" / arm / f"gamma1-{gamma1:g}" / f"seed-{SELECTION_SEED}"


def train_run(output, arm, seed, lam, gamma1, steps, batch, out_dir, device="cuda", max_steps=None, source=None):
    """#123's train_run with the IDM model and config (resumable from out_dir/resume.pt; SIGTERM / SIGUSR1 ->
    resume.pt after the current optimizer step and TrainingPaused). model.pt uses #123's model schema: the backbone
    state under state_dict (rc.load_trained loads it), the IDM heads under idm_heads."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "model.pt"
    if final.is_file():
        log(f"{arm} seed {seed}: model.pt exists; skipped")
        return read_json(out_dir / "report.json")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    model, losses, cfg = idm.build_model(arm, lam, gamma1)
    wrapper = hn.Wrapper(model, losses).to(device).train()
    optimizer = hn.make_optimizer(model, cfg)
    scheduler = hn.WarmupCosine(optimizer, hn.warmup_steps(steps), steps)
    start, elapsed = 0, 0.0
    resume = out_dir / "resume.pt"
    state = torch.load(resume, map_location="cpu", weights_only=False) if resume.is_file() else None
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, elapsed = state["step"], state["elapsed"]
        log(f"{arm} seed {seed}: resumed at step {start}")
    source = source or rc.BatchSource(SOURCE, arm, seed, steps, batch)
    sampler = list(range(start, steps if max_steps is None else min(steps, start + max_steps)))
    pixels = hn.ARMS[arm]["input"] == "pixels"
    loader = torch.utils.data.DataLoader(source, batch_size=None, sampler=sampler, pin_memory=pixels,
                                         **(rc.PIXEL_LOADER if pixels else {"num_workers": 2, "prefetch_factor": 4}))
    batches = iter(loader)  # draws the loader's worker base seed from the CPU generator
    if state is not None:
        # restored after that draw (#123 A1), so a resumed run continues the CPU (and CUDA) streams
        torch.set_rng_state(state["cpu_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng"])
    geometry = hn.Geometry(arm)
    log_path = out_dir / "train-log.jsonl"
    began = time.monotonic()
    failure = None
    step = start
    trace = []
    with open(log_path, "a") as log_file:
        for item in batches:
            step = item["b"]
            starts = {level: s for level, s in enumerate(item["starts"], start=1)}
            model.plan = (geometry.level1_steps, starts, item["needed"])
            batch_gpu = {"pixels_level1": item["x"].to(device, non_blocking=True),
                         "action_level1": item["action"].to(device, non_blocking=True)}
            try:
                stats = hn.train_step(wrapper, cfg, optimizer, scheduler, batch_gpu)
            except ValueError as error:
                if "NaN loss" not in str(error):
                    raise
                failure = {"typed_failure": "nonfinite_loss", "step": step, "message": str(error)}
                break
            if not np.isfinite(stats["loss"]):
                failure = {"typed_failure": "nonfinite_loss", "step": step, "message": "nonfinite total loss"}
                break
            if step % rc.LOG_EVERY == 0 or step == steps - 1:
                gc.collect()
                entry = {"step": step, "lr": [g["lr"] for g in optimizer.param_groups],
                         "seconds": elapsed + time.monotonic() - began,
                         "allocated_gib": torch.cuda.memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None,
                         **stats}
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()
                trace.append(entry)
            stop = bool(rc.STOP_REQUESTED) and step + 1 < steps
            if stop or ((step + 1) % rc.CHECKPOINT_EVERY == 0 and step + 1 < steps):
                atomic_torch(resume, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                      "scheduler": scheduler.state_dict(), "cpu_rng": torch.get_rng_state(),
                                      "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                                      "step": step + 1, "elapsed": elapsed + time.monotonic() - began})
            if stop:
                log(f"{arm} seed {seed}: stop signal; resume.pt written at step {step + 1}")
                raise rc.TrainingPaused(f"{arm} seed {seed} paused at step {step + 1} of {steps}")
    seconds = elapsed + time.monotonic() - began
    if failure is None and max_steps is not None and step + 1 < steps:
        return {"partial": True, "steps_done": step + 1 - start, "seconds": time.monotonic() - began}
    complete = failure is None
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
    backbone, heads = idm.split_state(model.state_dict())
    total = sum(p.numel() for p in model.parameters())
    head_count = sum(p.numel() for n, p in model.named_parameters() if idm.is_head(n))
    payload = {"schema": "issue_123_model_v1", "issue": 125, "arm": arm, "seed": seed, "lambda": lam,
               "gamma1": gamma1, "coefficients": idm.coefficients(arm, gamma1), "steps": steps, "batch": batch,
               "complete": complete, "typed_failure": failure, "state_dict": backbone if complete else None,
               "idm_heads": heads if complete else None, "parameters": total, "idm_head_parameters": head_count}
    atomic_torch(final, payload)
    report = {"arm": arm, "seed": seed, "lambda": lam, "gamma1": gamma1, "steps": steps, "batch": batch,
              "complete": complete, "typed_failure": failure, "wall_seconds": seconds, "gpu_seconds": seconds,
              "peak_gib": peak, "model_sha256": sha256_of(final), "parameters": total,
              "idm_head_parameters": head_count, "final": trace[-1] if trace else None, "finished_at": utc_now()}
    write_json(out_dir / "report.json", report)
    if resume.is_file() and complete:
        resume.unlink()
    log(f"{arm} seed {seed} lam {lam} gamma1 {gamma1}: {'complete' if complete else failure['typed_failure']} "
        f"after {seconds:.0f}s")
    return report


def load_trained_idm(path, device="cuda"):
    """(model with IDM heads, payload) of an IDM model.pt (the full trained state), or (None, payload)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "issue_123_model_v1" or payload.get("issue") != 125:
        raise ValueError(f"{path}: not an issue-125 model")
    if not payload["complete"]:
        return None, payload
    model, _, _ = idm.build_model(payload["arm"], payload["lambda"], payload["gamma1"], checkpointing=False)
    model.load_state_dict({**payload["state_dict"], **payload["idm_heads"]})
    return model.to(device).eval(), payload


def install_stop_handlers():
    for signum in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, rc.request_stop)


# ---------------------------------------------------------------------------
# trigger (fit split only; #123's readout code and guard expressions)
# ---------------------------------------------------------------------------

class TriggerNotReady(Exception):
    """A #123 primary arm-seed has no model.pt (training not complete)."""


def trigger_path(output):
    return Path(output) / "trigger.json"


def triggered(output):
    """True / False from trigger.json; raises if the trigger was not evaluated."""
    path = trigger_path(output)
    if not path.is_file():
        raise ValueError("trigger.json missing; run --trigger first")
    return bool(read_json(path)["triggered"])


def record_not_triggered(output, stage):
    path = Path(output) / f"{stage}.json"
    write_json(path, {"schema": "issue_125_stage_v1", "stage": stage, "status": NOT_TRIGGERED, "at": utc_now(),
                      "trigger_sha256": sha256_of(trigger_path(output))})
    log(f"{stage}: {NOT_TRIGGERED} (trigger.json triggered = false); nothing run")
    return 0


def seed_guards(part):
    """#123's per-seed G-collapse / G-action expressions (rc.arm_guards) on a readout-style report. A typed
    failure fails both, as in rc.arm_guards (s.get('G_collapse', False))."""
    if part.get("typed_failure"):
        return {"typed_failure": part["typed_failure"], "G_collapse": False, "G_action": False, "pass": False}
    thresholds = rc.GUARD_THRESHOLDS
    top = part["collapse"][-1]
    dim = top["dim"]
    collapse_ok = (top["effective_rank"] >= thresholds["effective_rank_fraction"] * dim
                   and top["within_episode_share"] >= thresholds["within_episode_share"])
    action_ok = part["action_sensitivity"] >= thresholds["action_sensitivity"]
    return {"effective_rank": top["effective_rank"], "dim": dim, "effective_rank_threshold":
            thresholds["effective_rank_fraction"] * dim, "within_episode_share": top["within_episode_share"],
            "action_sensitivity": part["action_sensitivity"], "G_collapse": bool(collapse_ok),
            "G_action": bool(action_ok), "pass": bool(collapse_ok and action_ok)}


def trigger_decision(parts):
    """{arm: {seed: guards}} and triggered (any seed of any primary #123 arm fails G-collapse or G-action)."""
    arms = {arm: {str(seed): seed_guards(parts[(arm, seed)]) for seed in SEEDS} for arm in TRIGGER_ARMS}
    failing = [f"{arm}/{seed}" for arm, seeds in arms.items() for seed, g in seeds.items()
               if not (g["G_collapse"] and g["G_action"])]
    return arms, failing, bool(failing)


def issue_123_state():
    """What #123 has read of the evaluation split so far (records, evaluation anchors, the position cache)."""
    records = sorted((SOURCE / "records").glob("*.json")) if (SOURCE / "records").is_dir() else []
    state = {"records_scored": len(records), "readouts_frozen": (SOURCE / "readouts-manifest.json").is_file(),
             "evaluation_pixel_anchors_read": rc.pixel_anchors_path(SOURCE).is_file(),
             "evaluation_position_cache_built": rc.eval_cache_path(SOURCE).is_file()}
    state["before_123_evaluation_read"] = bool(not records and not state["evaluation_pixel_anchors_read"]
                                               and not state["evaluation_position_cache_built"])
    return state


def refuse_trigger(output, state, when):
    """Record why the trigger was not evaluated (trigger-refused.json; no trigger.json); exit 1 (scheduler: failed)."""
    write_json(Path(output) / "trigger-refused.json", {
        "schema": "issue_125_trigger_refusal_v1", "identity": IDENTITY, "refused_at": utc_now(), "when": when,
        "rule": ("the trigger runs only before #123's first evaluation-split read (no #123 scoring record, evaluation "
                 "pixel anchors or evaluation position cache), at its start and at its end (ticket 125: trigger "
                 "before evaluation); the scheduler makes 123-score depend on 125-trigger"),
        "issue_123_state": state})
    log(f"trigger refused {when}: #123 has read the evaluation split ({state}); trigger-refused.json written")
    return 1


@torch.no_grad()
def trigger_part(arm, seed, data, device="cuda"):
    """fit_readouts steps 1-2 (rc.fit_readouts lines 'fit probe frames' and 'fit anchors'), on #123's model.pt:
    per-level effective rank and within-episode share on every 15th fit frame, endpoint launch sensitivity on the
    480 fit anchors x 16 grid launches, under deterministic_scoring()."""
    from world_model.planning import scoring_harness as harness
    path = rc.run_dir(SOURCE, arm, seed) / "model.pt"
    if not path.is_file():
        raise TriggerNotReady(f"{arm} seed {seed}: no model.pt (training not complete; resume.pt is never used)")
    began = time.monotonic()
    with harness.deterministic_scoring() as policy:
        model, payload = rc.load_trained(path, device)
        base = {"arm": arm, "seed": seed, "model": str(path.relative_to(ROOT)), "model_sha256": sha256_of(path),
                "scoring_policy": policy, "lambda": payload["lambda"], "steps": payload["steps"]}
        if model is None:
            return {**base, "typed_failure": payload["typed_failure"], "seconds": time.monotonic() - began}
        items, episodes = [], []
        for shot, end in enumerate(data.ends):
            for p in rc.probe_positions(int(end)):
                items.append((shot, p))
                episodes.append(shot)
        targets = data.carrier_rows(np.array([s for s, _ in items]), np.array([p for _, p in items]))
        levels = rc.encode_items(SOURCE, model, arm, "fit", items, targets, device)
        collapse = [{"level": k + 1, "effective_rank": hn.effective_rank(z), "within_episode_share":
                     hn.within_episode_share(z, episodes), "dim": int(z.shape[1])} for k, z in enumerate(levels)]
        anchors = torch.load(rc.rt.anchors_path(rc.RETRAIN), map_location="cpu", weights_only=True)
        members = sorted(anchors["anchors"])
        row_of = {k: i for i, k in enumerate(data.keys)}
        if hn.ARMS[arm]["input"] == "pixels":
            x0 = torch.stack([data.frame(row_of[anchors["frames"][m]], 0) for m in members])
        else:
            x0 = torch.stack([anchors["anchors"][m] for m in members])
        z0 = rc.encode_top(model, x0, device)[-1]
        grid = rc.grid_items()
        launches = torch.stack([rc.wlc_action(item["action"]) for item in grid])
        ends = rc.rollout(model, arm, z0.repeat_interleave(len(grid), 0), launches.repeat(len(members), 1), device)[:, -1]
        sensitivity = hn.action_sensitivity(ends.view(len(members), len(grid), -1))
    return {**base, "typed_failure": None, "fit_frames": len(items), "fit_shots": len(data.keys),
            "fit_anchors": len(members), "grid_launches": len(grid), "collapse": collapse,
            "action_sensitivity": sensitivity, "seconds": time.monotonic() - began}


def trigger(output):
    """Evaluate the frozen trigger once (resumable per arm-seed: parts are written atomically); trigger.json. Refuses
    (trigger-refused.json, exit 1) if #123 has read the evaluation split before it starts or by the time it ends."""
    output = Path(output)
    if trigger_path(output).is_file():
        log(f"trigger.json exists: triggered = {triggered(output)}")
        return 0
    state_before = issue_123_state()
    if not state_before["before_123_evaluation_read"]:
        return refuse_trigger(output, state_before, "at start")
    parts_dir = output / "trigger-parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    data = rc.FitData(SOURCE)
    parts = {}
    for arm in TRIGGER_ARMS:
        for seed in SEEDS:
            path = parts_dir / f"{arm}--seed{seed}.json"
            if not path.is_file():
                part = trigger_part(arm, seed, data)
                write_json(path, part)
                torch.cuda.empty_cache()
                mark(output, f"trigger-{arm}-seed{seed}", part["seconds"], part["seconds"])
                log(f"trigger {arm} seed {seed}: {seed_guards(part)}")
            parts[(arm, seed)] = read_json(path)
    arms, failing, fired = trigger_decision(parts)
    state_after = issue_123_state()
    if not state_after["before_123_evaluation_read"]:
        return refuse_trigger(output, state_after, "at end (#123 evaluation read during the trigger)")
    result = {"schema": "issue_125_trigger_v1", "identity": IDENTITY, "evaluated_at": utc_now(),
              "rule": ("triggered iff at least one #123 primary H-JEPA arm (HJ-px, HJ-car-m) fails G-collapse (top-level "
                       "effective rank < 0.1 D or within-episode variance share < 0.05) or G-action (endpoint launch "
                       "sensitivity < 0.05) in any seed; #123's readout code and guard expressions "
                       "(rc.fit_readouts, rc.arm_guards, rc.GUARD_THRESHOLDS) on the trained model.pt; fit probe "
                       "frames and fit anchors only; a typed training failure fails both guards (rc.arm_guards)"),
              "thresholds": {k: rc.GUARD_THRESHOLDS[k] for k in ("effective_rank_fraction", "within_episode_share",
                                                                  "action_sensitivity")},
              "data": {"split": "fit (issue109-fit)", "frames": "every 15th position 0..min(last, 285) of every fit shot",
                       "anchors": "the 480 #112 retrain fit anchors (frame 0 of each level's first complete candidate)",
                       "launches": "the 16 grid candidates of the frozen training candidate set",
                       "evaluation_files_read": 0},
              "arms": arms, "failing": failing, "triggered": fired,
              "parts": {f"{arm}/{seed}": {"sha256": sha256_of(parts_dir / f"{arm}--seed{seed}.json"),
                                          **{k: parts[(arm, seed)][k] for k in ("model", "model_sha256")}}
                        for arm in TRIGGER_ARMS for seed in SEEDS},
              "issue_123_state_at_start": state_before, "issue_123_state_at_end": state_after}
    write_json(trigger_path(output), result)
    log(f"trigger: triggered = {fired} (failing {failing or 'none'})")
    return 0


# ---------------------------------------------------------------------------
# gamma_1 selection (B1, seed 20260908, policy-split grid AUC under R1)
# ---------------------------------------------------------------------------

def selection_runs():
    """The pre-declared selection: gamma_1 in {100, 225} on HJ-px-IDM / HJ-car-m-IDM at B1, seed 20260908."""
    return [(arm, gamma1) for arm in idm.PRIMARY for gamma1 in idm.GAMMA1_GRID]


def selection(output, gpu, gpus):
    """Train this slot's selection runs and fit their #123 readouts (fit split; policy split for the AUC only)."""
    output = Path(output)
    if not triggered(output):
        return record_not_triggered(output, f"selection-slot{gpu}")
    install_stop_handlers()
    for i, (arm, gamma1) in enumerate(selection_runs()):
        if i % gpus != gpu:
            continue
        if rc.STOP_REQUESTED:
            log(f"selection slot {gpu}: stop signal; exiting before {arm} gamma1 {gamma1:g}")
            return rc.EXIT_PAUSED
        run = selection_dir(output, arm, gamma1)
        lam = source_lambda(hn.ARMS[arm]["input"])
        trained_before = (run / "model.pt").is_file()
        try:
            report = train_run(output, arm, SELECTION_SEED, lam, gamma1, hn.B1["steps"], hn.B1["batch"], run)
        except rc.TrainingPaused as paused:
            log(f"selection slot {gpu}: {paused}")
            return rc.EXIT_PAUSED
        if not trained_before:
            mark(output, f"selection-train-{arm}-gamma1{gamma1:g}", report["wall_seconds"], report["gpu_seconds"])
        if (run / "readouts.json").is_file():
            continue
        if rc.STOP_REQUESTED:
            return rc.EXIT_PAUSED
        t0 = time.monotonic()
        rc.fit_readouts(SOURCE, run)
        mark(output, f"selection-readouts-{arm}-gamma1{gamma1:g}", time.monotonic() - t0, time.monotonic() - t0)
    return 0


def selection_result(output):
    out = {}
    for input_kind, arm in (("pixels", "HJ-px-IDM"), ("carriers", "HJ-car-m-IDM")):
        aucs = {}
        for gamma1 in idm.GAMMA1_GRID:
            report = read_json(selection_dir(output, arm, gamma1) / "readouts.json")
            aucs[f"{gamma1:g}"] = None if report.get("typed_failure") else report["policy_auc"]["R1"]
        valid = {k: v for k, v in aucs.items() if v is not None}
        best = max(valid.values()) if valid else None
        tied = [float(k) for k, v in valid.items() if round(v, 4) == round(best, 4)] if valid else []
        chosen = None if not tied else (idm.GAMMA1_TIE if idm.GAMMA1_TIE in tied else max(tied))
        out[input_kind] = {"arm": arm, "policy_grid_auc_R1": aucs, "selected_gamma1": chosen,
                           "rule": ("largest policy-split pooled grid AUC under R1 at B1, seed 20260908; ties (4 d.p.) "
                                    "-> 100 (the DROID value); applies to every IDM arm of the input type")}
    return out


# ---------------------------------------------------------------------------
# smoke (pre-freeze; fit / policy only)
# ---------------------------------------------------------------------------

def measure_throughput(arm, gamma1, batch, steps, device="cuda"):
    """Seconds per optimizer step of a scratch IDM run (nothing kept): median of the steps after warm-up."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    lam = source_lambda(hn.ARMS[arm]["input"])
    torch.manual_seed(0)
    model, losses, cfg = idm.build_model(arm, lam, gamma1)
    wrapper = hn.Wrapper(model, losses).to(device).train()
    optimizer = hn.make_optimizer(model, cfg)
    scheduler = hn.WarmupCosine(optimizer, 10, 100000)
    source = rc.BatchSource(SOURCE, arm, 0, steps, batch)
    pixels = hn.ARMS[arm]["input"] == "pixels"
    loader = torch.utils.data.DataLoader(source, batch_size=None, sampler=list(range(steps)), pin_memory=pixels,
                                         **(rc.PIXEL_LOADER if pixels else {"num_workers": 2, "prefetch_factor": 4}))
    geometry = hn.Geometry(arm)
    times = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    last = time.monotonic()
    for item in loader:
        starts = {level: s for level, s in enumerate(item["starts"], start=1)}
        model.plan = (geometry.level1_steps, starts, item["needed"])
        hn.train_step(wrapper, cfg, optimizer, scheduler, {"pixels_level1": item["x"].to(device),
                                                           "action_level1": item["action"].to(device)})
        torch.cuda.synchronize()
        now = time.monotonic()
        times.append(now - last)
        last = now
    warm = times[len(times) // 4:]
    return {"arm": arm, "batch": batch, "steps_timed": len(warm), "seconds_per_step": float(np.median(warm)),
            "mean_seconds_per_step": float(np.mean(warm)), "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30}


def arm_budgets(data):
    return {arm: hn.steps_for(arm, rc.valid_clips(arm, data.ends)) for arm in idm.ARMS}


def projection(output, throughput, readout_seconds):
    """GPU-h and wall projection of the main #125 work (training, readouts, scoring) plus what #125 has spent."""
    jobs = [(f"{arm}/{seed}", throughput[arm]["projected_gpu_hours_per_seed"]) for arm in idm.ARMS for seed in SEEDS]
    train_hours = sum(h for _, h in jobs)
    readout_hours = readout_seconds * len(jobs) / 3600.0
    scoring_hours = 0.15 * len(jobs) + 1.0  # #123's per-record rule
    _, spent_gpu, _ = rc.ledger_totals(output)
    queues, loads = rc.lpt_queues(jobs)
    total = train_hours + readout_hours + scoring_hours + spent_gpu / 3600.0
    wall = max(loads) + (readout_hours + scoring_hours) / GPUS
    return {"training_gpu_hours": train_hours, "readout_gpu_hours": readout_hours, "scoring_gpu_hours": scoring_hours,
            "spent_issue125_gpu_hours": spent_gpu / 3600.0, "total_gpu_hours": total, "training_queues": queues,
            "training_queue_hours": loads, "projected_wall_hours": wall,
            "projected_nights": int(np.ceil(wall / NIGHT_HOURS)), "night_hours": NIGHT_HOURS,
            "cap_gpu_hours": CAP_GPU_HOURS, "stop_wall_hours": STOP_WALL_HOURS,
            "within_cap": bool(total <= CAP_GPU_HOURS and wall <= STOP_WALL_HOURS)}


def smoke_tests():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(["python", "-m", "pytest", "-q", TESTS], cwd=ROOT, capture_output=True, text=True, env=env)
    return {"command": f'CUDA_VISIBLE_DEVICES="" pytest -q {TESTS}', "exit_code": proc.returncode,
            "tail": proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-400:]}


def accounting():
    counts = {arm: idm.parameter_counts(arm) for arm in idm.ARMS}
    return {"parameters": counts, "gate_b_unit_parameters": idm.GATE_B_UNIT_PARAMETERS,
            "matched": {arm: idm.matched(counts[arm]["total"]) for arm in ("HJ-car-m-IDM", "LeWM-car-m-IDM")},
            "rule": ("carrier IDM arms: total parameters incl. the IDM heads within +-2 % of the Gate-B unit; #123's "
                     "matched sizes with only the predictor MLP width reduced (uniformly over levels) to the value "
                     "whose total is nearest the unit")}


def smoke(output):
    """Pre-freeze smoke (fit / policy only). not_triggered -> exit 0 at once. Writes smoke.json."""
    output = Path(output)
    if not triggered(output):
        return record_not_triggered(output, "smoke")
    began = time.monotonic()
    data = rc.FitData(SOURCE)
    gport_result = read_json(output / "gport-idm.json")
    tests = smoke_tests()
    selected = selection_result(output)
    budgets = arm_budgets(data)
    throughput = {}
    for arm in idm.ARMS:
        steps, batch = budgets[arm]
        gamma1 = selected[hn.ARMS[arm]["input"]]["selected_gamma1"] or idm.GAMMA1_TIE
        measured = measure_throughput(arm, gamma1, batch, SMOKE_STEPS[hn.ARMS[arm]["input"]])
        throughput[arm] = {**measured, "steps": steps, "budget": hn.ARMS[arm]["budget"],
                           "projected_gpu_hours_per_seed": steps * measured["seconds_per_step"] / 3600.0}
        log(f"throughput {arm}: {measured['seconds_per_step']:.3f} s/step at batch {batch}, {steps} steps -> "
            f"{throughput[arm]['projected_gpu_hours_per_seed']:.2f} GPU-h per seed")
        torch.cuda.empty_cache()
    readout_seconds = float(np.mean([read_json(selection_dir(output, arm, g) / "readouts.json").get("seconds", 0.0)
                                     for arm, g in selection_runs()]))
    project = projection(output, throughput, readout_seconds)
    counts = accounting()
    chosen = all(v["selected_gamma1"] is not None for v in selected.values())
    ok = bool(gport_result["pass"] and tests["exit_code"] == 0 and all(counts["matched"].values()) and chosen
              and project["within_cap"])
    evidence = {"schema": "issue_125_smoke_v1", "ok": ok, "at": utc_now(), "trigger_sha256": sha256_of(trigger_path(output)),
                "accounting": counts, "gport_idm": {"pass": gport_result["pass"],
                                                    "arms": {a: r["pass"] for a, r in gport_result["arms"].items()}},
                "tests": tests, "throughput": throughput, "selection": selected,
                "selection_reports": {f"{arm}/gamma1-{g:g}": read_json(selection_dir(output, arm, g) / "readouts.json")
                                      for arm, g in selection_runs()},
                "projection": project, "wall_seconds": time.monotonic() - began}
    write_json(output / "smoke.json", evidence)
    mark(output, "smoke", evidence["wall_seconds"], evidence["wall_seconds"])
    log(f"smoke ok={ok}: projection {project['total_gpu_hours']:.0f} GPU-h, {project['projected_wall_hours']:.0f} h wall "
        f"({project['projected_nights']} nights; cap {CAP_GPU_HOURS:.0f} GPU-h / {STOP_WALL_HOURS:.0f} h); gamma1 "
        f"{ {k: v['selected_gamma1'] for k, v in selected.items()} }")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# plan (frozen before any main-arm training and any #125 evaluation read; the freeze commit is the agent's)
# ---------------------------------------------------------------------------

def frozen_inputs(output):
    output = Path(output)
    names = {"runner": ROOT / RUNNER, "module": ROOT / MODULE, "tests": ROOT / TESTS,
             "port_reference": ROOT / REFERENCE_SCRIPT, "gport_idm": output / "gport-idm.json",
             "smoke": output / "smoke.json", "trigger": trigger_path(output),
             "issue_123_plan": SOURCE / "plan.json", "issue_123_runner": ROOT / rc.RUNNER,
             "issue_123_module": ROOT / rc.MODULE, "package_init": ROOT / "world_model/hjepa/__init__.py",
             "amendments_registry": ROOT / "scripts/post_freeze_amendments.py"}
    for arm, gamma1 in selection_runs():
        names[f"selection_{arm}_{gamma1:g}"] = selection_dir(output, arm, gamma1) / "readouts.json"
    return [{"name": name, "artifact": str(Path(path).relative_to(ROOT)), "sha256": sha256_of(path)}
            for name, path in names.items()]


def frozen_plan(output, frozen_at):
    output = Path(output)
    evidence = read_json(output / "smoke.json")
    source_plan = read_json(SOURCE / "plan.json")
    gamma1 = {kind: evidence["selection"][kind]["selected_gamma1"] for kind in ("pixels", "carriers")}
    data = rc.FitData(SOURCE)
    arms = {}
    for arm in idm.ARMS:
        spec = hn.ARMS[arm]
        steps, batch = hn.steps_for(arm, rc.valid_clips(arm, data.ends))
        lam = source_lambda(spec["input"])
        counts = idm.parameter_counts(arm, lam, gamma1[spec["input"]])
        arms[arm] = {**spec, "extends": idm.BASES[arm][0], "lambda": lam, "gamma1": gamma1[spec["input"]],
                     "coefficients": idm.coefficients(arm, gamma1[spec["input"]]), "steps": steps, "batch": batch,
                     "valid_clips": rc.valid_clips(arm, data.ends), "parameters": counts["total"],
                     "parameter_counts": counts, "config": idm.config(arm, lam, gamma1[spec["input"]]),
                     "projected_gpu_hours_per_seed": evidence["throughput"][arm]["projected_gpu_hours_per_seed"]}
    jobs = [(f"{arm}/{seed}", arms[arm]["projected_gpu_hours_per_seed"]) for arm in arms for seed in SEEDS]
    queues, loads = rc.lpt_queues(jobs)
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "issue": 125, "related_issues": [108, 123, 109, 112, 101],
        "frozen_at": frozen_at, "frozen_before_training_and_evaluation": True, "runner": RUNNER,
        "validation_command": VALIDATION_COMMAND, "engine_seconds": 0,
        "question": ("does H-JEPA with its own inverse-dynamics remedy (IDM + action-SIGReg), reproduced on NovPhy, rank "
                     "the Gate-B grid candidates of the sealed #109 evaluation split better or worse than the Gate-B unit?"),
        "trigger": {k: read_json(trigger_path(output))[k] for k in ("evaluated_at", "triggered", "failing", "arms")},
        "seeds": list(SEEDS), "selected_gamma1": gamma1, "selection": evidence["selection"],
        "inherited_from_123": {"plan_identity": source_plan["identity"], "frozen_at": source_plan["frozen_at"],
                               "selected_lambda": source_plan["selected_lambda"],
                               "unchanged": ["data and splits", "H3-Delta layout", "anchor-mixture sampler",
                                             "launch encoding", "lambda 0.18", "optimizer and schedule", "readouts",
                                             "cost", "cells", "bootstrap", "guards"]},
        "remedy": {"idm_head": "official loss.InverseDynamicsModel MLP(2D -> 256 -> 256 -> D), ReLU, trunc-normal init",
                   "idm_loss": "official loss.InverseDynamicsLoss: mean ||I(z_t, z_t+1) - sg[a_t]||^2 over the level crop",
                   "targets": "level 1: the embedded launch on the launch transition, the embedded no-op elsewhere; "
                              "level >= 2: the pooled macro-action embedding (detached)",
                   "action_sigreg": "official loss.SIGReg() on the macro-action embeddings of levels >= 2, weight 0.005",
                   "gamma": {"level1": "selected from {100, 225} per input type", "upper": idm.GAMMA_UPPER},
                   "lewm": "one level: IDM at level 1 only (DROID LeWM+IDM), no action-SIGReg"},
        "arms": arms, "data_manifest": source_plan["data_manifest"],
        "existing_arms": {"U": "Gate-B unit LC : F-15-macro, published Gate-B records",
                          "C*": "LC : C-F-15, published Gate-B records", "J": "#101 J-pe-1e-04 records",
                          "J_dec": "#101 J-dec-1e-01 records", "P": "the Gate-B no-model action prior",
                          "HJ-px / HJ-car-m": "#123's published records (SI1 / SI2), read at publication, never re-scored"},
        "training": {"queues": queues, "queue_projected_hours": loads, "checkpoint_every": rc.CHECKPOINT_EVERY,
                     "stop": "SIGTERM / SIGUSR1 -> resume.pt after the current optimizer step, exit 75 (#123 A1)",
                     "typed_failure": "a NaN / nonfinite loss ends the run (model.pt complete False); never retried",
                     "recipe": "#123's frozen recipe; objective = vendored hjepa_forward with the IDM keys"},
        "readouts": "#123's fit_readouts unchanged (R1 / R2 / R3 / R4 / P_c, collapse / action diagnostics, policy AUCs)",
        "scoring": "#123's score_arm_seed unchanged on the Gate-B mixed cells; the wrong-anchor control of HJ-px-IDM and "
                   "HJ-car-m-IDM with the 480 fit anchors; U / C* position curves; one process, deterministic_scoring()",
        "contrasts": {name: {"first": a, "second": b} for name, (a, b) in CONTRASTS.items()},
        "wrong_anchor": {"arms": list(WRONG_ANCHOR_ARMS), "token": "SI8 right - wrong, delta rule"},
        "token_rule": f"supported iff mean >= {DELTA} and lower > 0; not_supported_by_this_experiment iff upper < {DELTA}; "
                      f"otherwise {STOP_TOKEN}; primary contrasts in both directions",
        "guards": {"inherited": "every #123 guard per arm (G-collapse, G-action, G-readout, G-nonfinite, G-trained) with "
                                "#123's thresholds; global G-fid (#123 fidelity/gfid.json), G-port (#123), G-isolation, "
                                "G-chronology, G-scoring-policy, G-records",
                   "G_port_IDM": "gport-idm.json pass",
                   "issue_123_arms": "SI1 / SI2 also need #123's published guard pass of HJ-px / HJ-car-m, #123's "
                                     "G-records (validated publication) and every global guard #123 published "
                                     "(G-isolation, G-chronology, G-scoring-policy and G-records required)",
                   "failure": "a failed guard turns every token involving the arm into readiness_or_precision_insufficient",
                   "thresholds": rc.GUARD_THRESHOLDS},
        "dispositions": {
            "1": ("H-JEPA with its inverse-dynamics remedy ranks better than our Gate-B model (scoped) iff PI1 or PI2 is "
                  "supported in the HJ - U direction, G-fid, G-port-IDM and the arm's guards pass, and SI8 is supported "
                  "for that arm; triggers the HJ-LC follow-up of #123"),
            "2": ("our Gate-B model ranks better than H-JEPA with its remedy (scoped) iff U - HJ-px-IDM and "
                  "U - HJ-car-m-IDM are both supported and every guard passes; scope: #123's plus 'with the paper's "
                  "IDM + action-SIGReg remedy, gamma_1 selected on the policy split'"),
            "3": "otherwise per-contrast tokens only (delta-rule wording)",
            "4": ("if the IDM arms still fail G-collapse / G-action: the paper's own remedy does not prevent collapse on "
                  "single-launch, action-sparse data (reported with the diagnostics)")},
        "caps": {"gpu_hours": CAP_GPU_HOURS, "stop_wall_hours": STOP_WALL_HOURS, "stop": "over the cap: stop and report"},
        "inputs": frozen_inputs(output),
        "smoke_evidence": {k: evidence[k] for k in ("ok", "at", "accounting", "gport_idm", "tests", "projection")},
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY or not plan["smoke_evidence"]["ok"]:
        raise ValueError("plan.json is not the issue-125 protocol with a passed smoke")
    from scripts import post_freeze_amendments as amendments
    for entry in plan["inputs"] + plan["data_manifest"]:
        if entry["artifact"] == REGISTRY:
            continue  # A1: the shared append-only registry that matches() reads; its frozen hash stays in plan.json
        if not amendments.matches(entry["artifact"], entry["sha256"], sha256_of(ROOT / entry["artifact"])):
            raise ValueError(f"frozen input {entry['artifact']} changed after the freeze")
    return plan


def prepare(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        load_plan(output)
        log("existing frozen plan validated")
        return 0
    if not triggered(output):
        raise ValueError("trigger.json says not triggered; #125 closes with the trigger record (no freeze)")
    if (output / "models").exists() or (output / "records").exists():
        raise ValueError("main-arm models or records exist before the freeze; refusing")
    evidence = read_json(output / "smoke.json")
    if not evidence.get("ok"):
        raise ValueError("smoke failed or projects over the cap; STOP RULE: no freeze")
    plan = frozen_plan(output, utc_now())
    (output / "plan.json").write_text(json_text(plan))
    load_plan(output)
    log(f"frozen plan published: gamma1 {plan['selected_gamma1']}, queues {plan['training']['queue_projected_hours']}")
    return 0


def train(output, gpu):
    """This GPU's frozen training queue; SIGTERM / SIGUSR1 -> resume.pt and exit 75; a relaunch resumes there."""
    plan = load_plan(output)
    install_stop_handlers()
    for name in plan["training"]["queues"][gpu]:
        if rc.STOP_REQUESTED:
            log(f"queue {gpu}: stop signal; exiting before {name}")
            return rc.EXIT_PAUSED
        arm, seed = name.split("/")
        spec = plan["arms"][arm]
        run = run_dir(output, arm, int(seed))
        finished_before = (run / "model.pt").is_file()  # A1: a relaunch must not add a finished run's seconds again
        try:
            report = train_run(output, arm, int(seed), spec["lambda"], spec["gamma1"], spec["steps"], spec["batch"], run)
        except rc.TrainingPaused as paused:
            log(f"queue {gpu}: {paused}")
            return rc.EXIT_PAUSED
        if not report.get("partial") and not finished_before:
            mark(output, f"train-{arm}-seed{seed}", report["wall_seconds"], report["gpu_seconds"])
    return 0


def readouts(output, gpu, gpus=GPUS):
    """#123's fit_readouts for every main arm-seed of this GPU (round robin); fit data from #123's stores."""
    plan = load_plan(output)
    install_stop_handlers()
    names = [f"{arm}/{seed}" for arm in plan["arms"] for seed in SEEDS]
    data, policy, labels = rc.FitData(SOURCE), rc.policy_split(SOURCE), rc.fit_labels(SOURCE)
    for i, name in enumerate(names):
        if i % gpus != gpu:
            continue
        arm, seed = name.split("/")
        run = run_dir(output, arm, int(seed))
        if (run / "readouts.json").is_file():
            continue
        if rc.STOP_REQUESTED:
            return rc.EXIT_PAUSED
        t0 = time.monotonic()
        rc.fit_readouts(SOURCE, run, data=data, policy=policy, labels=labels)
        mark(output, f"readouts-{arm}-seed{seed}", time.monotonic() - t0, time.monotonic() - t0)
    return 0


def freeze_readouts(output):
    plan = load_plan(output)
    path = Path(output) / "readouts-manifest.json"
    if path.is_file():
        raise ValueError("readouts are already frozen")
    entries = {}
    for arm in plan["arms"]:
        for seed in SEEDS:
            run = run_dir(output, arm, seed)
            report = read_json(run / "readouts.json")
            has = rc.readouts_path(run).is_file()
            entries[f"{arm}/{seed}"] = {"model": str((run / "model.pt").relative_to(ROOT)),
                                        "model_sha256": sha256_of(run / "model.pt"),
                                        "readouts": str(rc.readouts_path(run).relative_to(ROOT)) if has else None,
                                        "readouts_sha256": sha256_of(rc.readouts_path(run)) if has else None,
                                        "report_sha256": sha256_of(run / "readouts.json"),
                                        "typed_failure": report.get("typed_failure")}
    write_json(path, {"schema": "issue_125_readouts_manifest_v1", "plan_identity": IDENTITY, "frozen_at": utc_now(),
                      "entries": entries})
    log(f"readouts frozen: {len(entries)} arm-seeds")
    return 0


# ---------------------------------------------------------------------------
# scoring: once, after the readout freeze, under deterministic_scoring()
# ---------------------------------------------------------------------------

@torch.no_grad()
def wrong_anchor_rows(output, arm, seed, costs, device="cuda"):
    """#123's wrong-anchor control (rc.score_arm_seed): the 16 grid launches from every one of the 480 fit anchors."""
    run = run_dir(output, arm, seed)
    model, _ = rc.load_trained(run / "model.pt", device)
    probes = rc.load_readouts(run, device)
    fit_store = torch.load(rc.rt.anchors_path(rc.RETRAIN), map_location="cpu", weights_only=True)
    fit_members = sorted(fit_store["anchors"])
    if hn.ARMS[arm]["input"] == "pixels":
        data = rc.FitData(SOURCE)
        row_of = {k: i for i, k in enumerate(data.keys)}
        fx = torch.stack([data.frame(row_of[fit_store["frames"][m]], 0) for m in fit_members])
    else:
        fx = torch.stack([fit_store["anchors"][m] for m in fit_members])
    fz0 = rc.encode_top(model, fx, device)[-1]
    grid = rc.grid_items()
    launches = torch.stack([rc.wlc_action(item["action"]) for item in grid])
    w_end = rc.rollout(model, arm, fz0.repeat_interleave(len(grid), 0), launches.repeat(len(fit_members), 1),
                       device)[:, -1].to(device)
    w_carrier = probes["R1"](w_end).cpu()
    w_anchor = probes["R1"](fz0.to(device)).cpu()
    return {member: {"rows": [costs.row(item["ordinal"], w_carrier[a * len(grid) + j], w_anchor[a])
                              for j, item in enumerate(grid)]} for a, member in enumerate(fit_members)}


def score_arm_seed(output, plan, manifest, arm, seed, cells, anchors, cache, costs, device="cuda"):
    """#123's score_arm_seed on the IDM arm-seed (its wrong-anchor list is #123's), then this ticket's wrong-anchor
    control for the primary IDM arms; the record carries this plan's schema and identity."""
    began = time.monotonic()
    record = rc.score_arm_seed(output, plan, manifest, arm, seed, cells, anchors, cache, costs, device)
    record.update({"schema": SCHEMA_RECORD, "plan_identity": IDENTITY})
    if arm in WRONG_ANCHOR_ARMS and not record["typed_failure"]:
        record["wrong_anchor"] = wrong_anchor_rows(output, arm, seed, costs, device)
    record["seconds"] = time.monotonic() - began
    return record


def score(output):
    from world_model.planning import scoring_harness as harness
    from scripts import run_lookahead_control as wlc
    output = Path(output)
    plan = load_plan(output)
    manifest = rc.load_manifest(output)
    began = time.monotonic()
    _, cells = rc.evaluation_set()
    if not rc.eval_cache_path(output).is_file():
        t0 = time.monotonic()
        rc.build_eval_cache(output, cells)
        mark(output, "score-eval-cache", time.monotonic() - t0, 0.0, cpu_workers=16)
    cache = torch.load(rc.eval_cache_path(output), map_location="cpu", weights_only=False)
    with harness.deterministic_scoring() as policy:
        if not rc.pixel_anchors_path(output).is_file():
            rc.read_pixel_anchors(output, policy)
        carriers = torch.load(rc.GATE_B / "data" / "anchors-evaluation.pt", map_location="cpu", weights_only=True)
        anchors = {"carriers": carriers["anchors"],
                   "pixels": torch.load(rc.pixel_anchors_path(output), map_location="cpu", weights_only=True)["frames"]}
        costs = harness.EndpointCosts(wlc.load_objective())
        for arm in plan["arms"]:
            for seed in SEEDS:
                path = rc.record_path(output, arm, seed)
                if path.is_file():
                    continue
                t0 = time.monotonic()
                record = score_arm_seed(output, plan, manifest, arm, seed, cells, anchors, cache, costs)
                record.update({"scoring_policy": policy, "scored_at": utc_now()})
                write_json(path, record)
                torch.cuda.empty_cache()
                mark(output, f"score-{arm}-seed{seed}", time.monotonic() - t0, time.monotonic() - t0)
                log(f"scored {arm} seed {seed} ({time.monotonic() - t0:.0f}s)")
        for name in ("unit", "c_star"):
            for seed in SEEDS:
                path = rc.curve_path(output, name, seed)
                if not path.is_file():
                    record = rc.score_reference_curves(output, name, seed, cells, cache)
                    record.update({"scoring_policy": policy, "scored_at": utc_now(), "schema": SCHEMA_RECORD,
                                   "plan_identity": IDENTITY})
                    write_json(path, record)
    log(f"scoring complete ({time.monotonic() - began:.0f}s)")
    return 0


# ---------------------------------------------------------------------------
# publication and validation
# ---------------------------------------------------------------------------

VALIDATION_RECEIPT = "validation.json"
PUBLISHED = ("compute.json", "summary.json", "findings.md")


def records_manifest(output):
    files = sorted((Path(output) / "records").glob("*.json"))
    return {"schema": "issue_125_records_manifest_v1", "plan_identity": IDENTITY,
            "records": [{"artifact": str(p.relative_to(ROOT)), "sha256": sha256_of(p), "bytes": p.stat().st_size}
                        for p in files]}


def records_guard(output):
    output = Path(output)
    manifest_path, receipt_path = output / "records-manifest.json", output / VALIDATION_RECEIPT
    if not manifest_path.is_file() or not receipt_path.is_file():
        return {"pass": False, "receipt": None}
    receipt = read_json(receipt_path)
    bound = bool(receipt.get("plan_identity") == IDENTITY
                 and receipt.get("records_manifest_sha256") == sha256_of(manifest_path)
                 and read_json(manifest_path) == records_manifest(output))
    spot = receipt.get("spot_check")
    return {"pass": bool(bound and receipt.get("pass") is True and spot is not None and spot.get("pass") is True),
            "receipt": VALIDATION_RECEIPT, "bound_to_current_records": bound,
            "records_manifest_sha256": receipt.get("records_manifest_sha256"), "spot_check": spot}


def write_receipt(output, spot, error=None):
    output = Path(output)
    manifest_path = output / "records-manifest.json"
    write_json(output / VALIDATION_RECEIPT, {
        "schema": "issue_125_validation_receipt_v1", "plan_identity": IDENTITY,
        "records_manifest_sha256": sha256_of(manifest_path) if manifest_path.is_file() else None,
        "spot_check": spot, "error": error, "validated_at": utc_now(),
        "pass": bool(error is None and spot is not None and spot["pass"])})


def issue_123_inputs():
    """#123's published records of HJ-px / HJ-car-m and their guards (SI1 / SI2); never re-scored. An arm passes only
    if its #123 arm guards, #123's G-records (validation.json, re-checked now) and every global guard #123 published in
    compute.json pass; G-isolation, G-chronology, G-scoring-policy and G-records must be among them."""
    compute = read_json(SOURCE / "compute.json")
    validated = rc.records_guard(SOURCE)
    published = compute["guards"]["global"]
    missing = [name for name in ISSUE_123_GLOBAL_GUARDS if name not in published]
    if missing:
        raise ValueError(f"#123 compute.json does not publish the global guards {missing}")
    global_123 = {name: bool(guard["pass"]) for name, guard in published.items()}
    records = {(arm, seed): read_json(rc.record_path(SOURCE, arm, seed)) for arm in ISSUE_123_ARMS for seed in SEEDS}
    for (arm, seed), record in records.items():
        if record.get("schema") != rc.SCHEMA_RECORD or record.get("plan_identity") != rc.IDENTITY:
            raise ValueError(f"#123 record {arm}/{seed} binding differs")
    guards = {arm: {"pass": bool(compute["guards"]["arms"][arm]["pass"] and validated["pass"]
                                 and all(global_123.values())),
                    "issue_123_arm_guards_pass": compute["guards"]["arms"][arm]["pass"],
                    "issue_123_G_records": validated["pass"], "issue_123_global_guards": global_123,
                    "source": "#123 compute.json (guards.arms, guards.global) and validation.json"}
              for arm in ISSUE_123_ARMS}
    return records, guards


def global_guards(output, plan, records, manifest):
    from scripts import run_gate_b as gb
    from world_model.planning import scoring_harness as harness
    gfid = SOURCE / "fidelity" / "gfid.json"
    fid = read_json(gfid) if gfid.is_file() else {"G_fid": None}
    gport_123 = read_json(rc.gport_dir(SOURCE) / "gport.json")
    gport_123_a2 = rc.gport_a2_dir(SOURCE) / "gport.json"
    gport_123_a2 = read_json(gport_123_a2) if gport_123_a2.is_file() else {"pass": None}
    gport_idm = read_json(Path(output) / "gport-idm.json")
    roster = {level["identity"] for level in gb.evaluation_levels()}
    fit_roster = {level["identity"] for level in rc.rt.fit_levels()}
    scored = [r for r in records.values() if not r["typed_failure"]]
    pixel_store = torch.load(rc.pixel_anchors_path(output), map_location="cpu", weights_only=True)
    isolation = (all(set(r["inventories"]["grid"]) <= roster for r in scored)
                 and all(set(r["wrong_anchor"] or {}) <= fit_roster for r in scored)
                 and all(k.startswith("issue-109-sc-fit") for k in rc.FitData(SOURCE).keys))
    first_scored = min(r["scored_at"] for r in records.values())
    trigger = read_json(trigger_path(output))
    chronology = (trigger["evaluated_at"] <= plan["frozen_at"] <= manifest["frozen_at"] <= pixel_store["read_at"]
                  <= first_scored)
    policies = all(r.get("scoring_policy") == harness.SCORING_POLICY for r in records.values())
    return {"G_fid": {"pass": fid["G_fid"] is True, "value": fid["G_fid"]},
            "G_port": {"pass": bool(gport_123["pass"] and gport_123_a2["pass"] is True), "source": "#123 gport"},
            "G_port_IDM": {"pass": bool(gport_idm["pass"])},
            "G_isolation": {"pass": bool(isolation)},
            "G_chronology": {"pass": bool(chronology), "trigger_evaluated_at": trigger["evaluated_at"],
                             "plan_frozen_at": plan["frozen_at"], "readouts_frozen_at": manifest["frozen_at"],
                             "anchors_read_at": pixel_store["read_at"], "first_record_scored_at": first_scored},
            "G_scoring_policy": {"pass": bool(policies)},
            "G_records": records_guard(output)}


def grid_contrasts(rows, guards_arm, global_ok):
    """PI1, PI2, SI1-SI7 on the grid rows; each token gated by the global guards and every involved arm's guards."""
    from scripts import run_decision_chain_attribution as m99
    contrasts = {}
    for label, (first, second) in CONTRASTS.items():
        paired = m99.paired(rows, [(1, first), (-1, second)])
        involved = [a for a in (rc.arm_of(first), rc.arm_of(second)) if a in guards_arm]
        guarded = bool(global_ok and all(guards_arm[a]["pass"] for a in involved))
        forward, backward = rc.token_pair(paired["estimate"], guarded)
        contrasts[label] = {"first": first, "second": second, **paired, "token": forward,
                            "reverse_token": backward, "guards_pass": guarded}
    return contrasts


def wrong_anchor_table(cells, records, guards_arm, global_ok):
    """SI8 (right - wrong anchor) per primary IDM arm; #123's S10 computation."""
    from scripts import run_decision_dynamics_controls as c2
    wrong = {}
    for arm in WRONG_ANCHOR_ARMS:
        wrows = []
        for member, entry in sorted(cells["grid"]["mixed"].items()):
            for seed in SEEDS:
                record = records[(arm, seed)]
                if record["typed_failure"]:
                    wrows.append({"member": member, "seed": seed, "views": {arm: None}})
                    continue
                right = record["inventories"]["grid"][member]["rows"]
                others = [record["wrong_anchor"][a]["rows"] for a in sorted(record["wrong_anchor"])]
                aucs = [a for a in (c2.auc(entry, w) for w in others) if a is not None]
                rhos = [r for r in (c2.spearman_rows(right, w) for w in others) if r is not None]
                wrows.append({"member": member, "seed": seed, "views": {arm: {
                    "right": c2.auc(entry, right), "wrong": float(np.mean(aucs)) if aucs else None,
                    "stability": float(np.mean(rhos)) if rhos else None}}})
        diff = c2.difference_block(wrows, [(1, arm, "right"), (-1, arm, "wrong")])
        guarded = bool(global_ok and guards_arm[arm]["pass"])
        wrong[arm] = {"right": c2.field_block(wrows, arm, "right"), "wrong": c2.field_block(wrows, arm, "wrong"),
                      "right_minus_wrong": diff, "stability": c2.field_block(wrows, arm, "stability"),
                      "token": c2.token(diff) if guarded else STOP_TOKEN, "guards_pass": guarded,
                      "wrong_anchors": len(records[(arm, SEEDS[0])]["wrong_anchor"] or {})}
    return wrong


def dispositions(grid, guards_arm):
    c = grid["contrasts"]
    wrong = grid["SI8_wrong_anchor"]
    hj_better = [arm for label, arm in (("PI1", "HJ-px-IDM"), ("PI2", "HJ-car-m-IDM"))
                 if c[label]["token"] == "supported" and wrong[arm]["token"] == "supported"]
    ours_better = c["PI1"]["reverse_token"] == "supported" and c["PI2"]["reverse_token"] == "supported"
    if hj_better:
        outcome = {"disposition": 1, "label": "H-JEPA with its inverse-dynamics remedy ranks better than our Gate-B "
                                              "model (scoped)", "arms": hj_better,
                   "branch": "open the HJ-LC follow-up described in #123"}
    elif ours_better:
        outcome = {"disposition": 2, "label": "Our Gate-B model ranks better than H-JEPA with its remedy (scoped)",
                   "scope": ("#109 sealed evaluation split; Gate-B mixed grid cells (311 x 3 seeds); tie-free cost; readout "
                             "R1; budgets B3 (pixels) and B1 (carriers); layout H3-Delta; the Gate-B unit LC : F-15-macro; "
                             "with the paper's IDM + action-SIGReg remedy, gamma_1 selected on the policy split"),
                   "branch": "scoped claim recorded for the manuscript"}
    else:
        outcome = {"disposition": 3, "label": "per-contrast tokens only", "branch": "no scoped ranking claim"}
    still = {arm: {"G_collapse": guards_arm[arm]["G_collapse"], "G_action": guards_arm[arm]["G_action"]}
             for arm in idm.PRIMARY}
    outcome["disposition_4_remedy_does_not_prevent_collapse"] = bool(
        any(not (g["G_collapse"] and g["G_action"]) for g in still.values()))
    outcome["disposition_4_arms"] = still
    return outcome


def compute_tables(output, plan):
    from scripts import run_decision_chain_attribution as m99
    from scripts import run_gate_b as gb
    from scripts import run_decision_relevant_dynamics as v1
    from world_model.planning import scoring_harness as harness
    output = Path(output)
    manifest = rc.load_manifest(output)
    _, cells = rc.evaluation_set()
    records = {(arm, seed): read_json(rc.record_path(output, arm, seed)) for arm in plan["arms"] for seed in SEEDS}
    for (arm, seed), record in records.items():
        if record.get("schema") != SCHEMA_RECORD or record.get("plan_identity") != IDENTITY:
            raise ValueError(f"record {arm}/{seed} binding differs")
    records_123, guards_123 = issue_123_inputs()
    references = {}
    for seed in SEEDS:
        references[("unit", seed)] = rc.gate_b_rows("unit", seed)
        references[("c_star", seed)] = rc.gate_b_rows("c_star", seed)
        for label, arm in rc.J_ARMS.items():
            references[(label, seed)] = read_json(rc.ISSUE_101 / "records" / f"{arm}--seed{seed}.json")
    fit = read_json(rc.RETRAIN / "fit-verdicts.json")["branches"]
    guards_global = global_guards(output, plan, records, manifest)
    guards_arm = rc.arm_guards(output, plan, records)
    global_ok = all(g["pass"] for g in guards_global.values())
    gating = {**guards_arm, **guards_123}
    view_plan = {"arms": [*plan["arms"], *ISSUE_123_ARMS]}
    all_records = {**records, **records_123}
    rankers = ["U", "C*", "P", "J", "J_dec", *plan["arms"], *ISSUE_123_ARMS,
               *[f"{a}:{r}" for a in plan["arms"] for r in ("R2", "R3", "R4")]]
    tables = {}
    for inventory in rc.EVAL_INVENTORIES:
        rows = rc.view_rows(view_plan, cells, inventory, all_records, references, gb.prior_stats(fit, inventory))
        block = {"cells": len(rows), "members": len({r["member"] for r in rows}), "rankers": {}}
        for name in rankers:
            ranker = v1.ranker_block(rows, name)
            ranker["per_seed"] = ({str(s): m99.block_of([r for r in rows if r["seed"] == s], name) for s in SEEDS}
                                  if name != "P" else None)
            ranker["top1_rate"] = ranker["top1_hits"] / ranker["cells"] if ranker["cells"] else None
            chances = [r["views"][name]["chance_top1"] for r in rows if r["views"][name]["chance_top1"] is not None]
            ranker["chance_top1"] = float(np.mean(chances)) if chances else None
            block["rankers"][name] = ranker
        if inventory == "grid":
            block["contrasts"] = grid_contrasts(rows, gating, global_ok)
            block["SI8_wrong_anchor"] = wrong_anchor_table(cells, records, guards_arm, global_ok)
        tables[inventory] = block
    positions = rc.position_table(output, plan, records, cells)
    entries, gpu, wall = rc.ledger_totals(output)
    return {"schema": "issue_125_compute_v1", "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "selected_gamma1": plan["selected_gamma1"], "trigger": plan["trigger"],
            "guards": {"global": guards_global, "arms": guards_arm, "issue_123_arms": guards_123},
            "inventories": tables, "positions": positions, "disposition": dispositions(tables["grid"], guards_arm),
            "records": {"arm_seeds": len(records), "typed_failures": {f"{a}/{s}": r["typed_failure"]
                                                                     for (a, s), r in records.items() if r["typed_failure"]}},
            "compute": {"issue_125_phases": entries, "issue_125_gpu_seconds": gpu, "issue_125_wall_seconds": wall,
                        "total_gpu_hours": gpu / 3600.0, "cap_gpu_hours": CAP_GPU_HOURS, "engine_seconds": 0,
                        "hardware": "4 x RTX 5090 (32 GB), night window 18:00-07:00 Australia/Melbourne"},
            "primary_cost": harness.PRIMARY_COST}


def findings_md(plan, compute):
    fmt_block, fmt_auc, fmt_num = rc.fmt_block, rc.fmt_auc, rc.fmt_num
    grid = compute["inventories"]["grid"]
    d = compute["disposition"]
    lines = ["# Issue #125 findings: H-JEPA with its inverse-dynamics remedy vs the Gate-B unit (sealed #109 split)", "",
             f"Plan frozen {plan['frozen_at']}; trigger evaluated {plan['trigger']['evaluated_at']} (failing "
             f"{', '.join(plan['trigger']['failing'])}); selected gamma_1 {plan['selected_gamma1']}; lambda 0.18 (#123). "
             "Intervals are member (level)-clustered percentile bootstraps (10 000 draws, PCG64 7201), DESCRIPTIVE. "
             "delta = 0.02.", "",
             f"**Disposition {d['disposition']}: {d['label']}** (outcome branch: {d['branch']}).", "",
             f"Disposition 4 (the remedy does not prevent collapse): {d['disposition_4_remedy_does_not_prevent_collapse']} "
             f"({d['disposition_4_arms']}).", "", "## Guards", "", "| guard | pass |", "| --- | --- |"]
    for name, g in compute["guards"]["global"].items():
        lines.append(f"| {name} | {g['pass']} |")
    lines += ["", "| arm | role | params | G-collapse | G-action | G-readout (policy ceiling) | G-nonfinite (R1 / R2) | "
              "trained | pass |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm, g in compute["guards"]["arms"].items():
        lines.append(f"| {arm} | {g['role']} | {g['parameters']:,} | {g['G_collapse']} | {g['G_action']} | "
                     f"{g['G_readout']} ({fmt_num(g['policy_ceiling_auc'])}) | {g['G_nonfinite']} "
                     f"({fmt_num(g['nonfinite_shares']['R1'], 3)} / {fmt_num(g['nonfinite_shares']['R2'], 3)}) | "
                     f"{g['G_trained']} | {g['pass']} |")
    for arm, g in compute["guards"]["issue_123_arms"].items():
        lines.append(f"| {arm} (#123, SI1 / SI2) | | | | | | | | {g['pass']} |")
    lines += ["", "| arm | seed | erank / D | chi | sensitivity | policy R1 AUC | policy ceiling |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for arm, g in compute["guards"]["arms"].items():
        for seed, s in g["seeds"].items():
            if "typed_failure" in s:
                lines.append(f"| {arm} | {seed} | typed failure {s['typed_failure']} | | | | |")
                continue
            lines.append(f"| {arm} | {seed} | {s['effective_rank']:.1f} / {s['dim']} | {s['within_episode_share']:.3f} | "
                         f"{s['action_sensitivity']:.3g} | {fmt_num(s['policy_r1_auc'])} | {fmt_num(s['policy_ceiling_auc'])} |")
    lines += ["", "## Grid rankers (311 mixed cells x 3 seeds)", "",
              "| ranker | pooled AUC | seed 20260908 | seed 20260909 | seed 20260910 | top-1 | chance top-1 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in grid["rankers"].items():
        seeds = r["per_seed"]
        per = " | ".join(fmt_auc(seeds.get(str(s))) for s in SEEDS) if seeds else "seed-free | | "
        lines.append(f"| {name} | {fmt_auc(r['auc'])} | {per} | {fmt_num(r['top1_rate'], 3)} | {fmt_num(r['chance_top1'], 3)} |")
    lines += ["", "## Contrasts (grid, paired per (level, seed))", "",
              "| contrast | first - second | estimate | token (first - second) | token (second - first) | guards |",
              "| --- | --- | --- | --- | --- | --- |"]
    for label, c in grid["contrasts"].items():
        lines.append(f"| {label} | {c['first']} - {c['second']} | {fmt_block(c['estimate'])} | {c['token']} | "
                     f"{c['reverse_token']} | {c['guards_pass']} |")
    lines += ["", "SI8 wrong-anchor control (right - wrong, 480 fit anchors):", "",
              "| arm | right | wrong | right - wrong | token |", "| --- | --- | --- | --- | --- |"]
    for arm, w in grid["SI8_wrong_anchor"].items():
        lines.append(f"| {arm} | {fmt_auc(w['right'])} | {fmt_auc(w['wrong'])} | {fmt_block(w['right_minus_wrong'])} | "
                     f"{w['token']} |")
    lines += ["", "## Descriptive inventories (no tokens)", "", "| ranker | offset | angle | power |", "| --- | --- | --- | --- |"]
    for name in grid["rankers"]:
        lines.append(f"| {name} | " + " | ".join(fmt_auc(compute["inventories"][i]["rankers"][name]["auc"])
                                                for i in ("offset", "angle", "power")) + " |")
    lines += ["", "## Engine position MSE (#100 metric; P_c for the IDM arms, predicted carriers for U / C*)", "",
              "| model | " + " | ".join(f"t = {t}" for t in rc.POSITION_TIMES) + " | floor t = 225 |",
              "| --- | " + " | ".join("---" for _ in rc.POSITION_TIMES) + " | --- |"]
    for name, row in compute["positions"].items():
        lines.append(f"| {name} | " + " | ".join(fmt_auc(row[f'pred_{t}']) for t in rc.POSITION_TIMES)
                     + f" | {fmt_auc(row['floor_225'])} |")
    lines += ["", "## Compute", "", f"- issue-125 GPU seconds {compute['compute']['issue_125_gpu_seconds']:.0f} "
              f"({compute['compute']['total_gpu_hours']:.1f} GPU-h of active compute; cap {CAP_GPU_HOURS:.0f}); engine "
              "seconds 0."]
    return "\n".join(lines) + "\n"


def summary(plan, compute):
    grid = compute["inventories"]["grid"]
    return {"schema": "issue_125_summary_v1", "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "disposition": compute["disposition"], "selected_gamma1": plan["selected_gamma1"],
            "grid_auc": {name: r["auc"] for name, r in grid["rankers"].items()},
            "contrasts": {label: {"estimate": c["estimate"], "token": c["token"], "reverse_token": c["reverse_token"]}
                          for label, c in grid["contrasts"].items()},
            "SI8": {arm: w["token"] for arm, w in grid["SI8_wrong_anchor"].items()},
            "guards": {"global": {k: g["pass"] for k, g in compute["guards"]["global"].items()},
                       "arms": {a: g["pass"] for a, g in compute["guards"]["arms"].items()},
                       "issue_123_arms": {a: g["pass"] for a, g in compute["guards"]["issue_123_arms"].items()}},
            "engine_seconds": 0}


def rendered(plan, compute):
    return {"compute.json": json_text(compute), "summary.json": json_text(summary(plan, compute)),
            "findings.md": findings_md(plan, compute)}


def write_publication(output, plan):
    from world_model.planning import scoring_harness as harness
    with harness.deterministic_scoring():
        compute = compute_tables(output, plan)
    for name, text in rendered(plan, compute).items():
        (Path(output) / name).write_text(text)
    return compute


def publish(output):
    """records-manifest.json and the tables; every token is the stop token until --validate binds a passing
    re-score to this records manifest (G-records)."""
    output = Path(output)
    plan = load_plan(output)
    write_json(output / "records-manifest.json", records_manifest(output))
    compute = write_publication(output, plan)
    log(f"published: disposition {compute['disposition']['disposition']} ({compute['disposition']['label']}); "
        f"G-records {compute['guards']['global']['G_records']['pass']}")
    return 0


def spot_check(output, plan):
    """Re-score the first complete record of HJ-car-m-IDM (then HJ-px-IDM) and compare its R1 grid rows and its
    wrong-anchor rows."""
    from world_model.planning import scoring_harness as harness
    from scripts import run_lookahead_control as wlc
    manifest = rc.load_manifest(output)
    _, cells = rc.evaluation_set()
    cache = torch.load(rc.eval_cache_path(output), map_location="cpu", weights_only=False)
    carriers = torch.load(rc.GATE_B / "data" / "anchors-evaluation.pt", map_location="cpu", weights_only=True)
    anchors = {"carriers": carriers["anchors"],
               "pixels": torch.load(rc.pixel_anchors_path(output), map_location="cpu", weights_only=True)["frames"]}
    costs = harness.EndpointCosts(wlc.load_objective())
    for arm in ("HJ-car-m-IDM", "HJ-px-IDM"):
        for seed in SEEDS:
            record = read_json(rc.record_path(output, arm, seed))
            if record["typed_failure"]:
                continue
            with harness.deterministic_scoring():
                fresh = score_arm_seed(output, plan, manifest, arm, seed, cells, anchors, cache, costs)
            worst, passed = dict.fromkeys(harness.PARSED_QUANTITIES, 0.0), True
            pairs = [(block["rows"], fresh["inventories"]["grid"][member]["rows"])
                     for member, block in record["inventories"]["grid"].items()]
            pairs += [(block["rows"], fresh["wrong_anchor"][member]["rows"])
                      for member, block in (record["wrong_anchor"] or {}).items()]
            for old, new in pairs:
                check = harness.endpoint_replication(old, new)
                passed &= check["pass"]
                for k in worst:
                    worst[k] = max(worst[k], check["max_abs_delta"][k])
            return {"arm": arm, "seed": seed, "cells": len(record["inventories"]["grid"]),
                    "wrong_anchors": len(record["wrong_anchor"] or {}), "max_abs_delta": worst, "pass": bool(passed)}
    raise ValueError("no complete HJ-car-m-IDM / HJ-px-IDM record to re-score")


def validate(output):
    """Re-check every record hash, byte-compare every published table with its recomputation, re-score one record;
    validation.json binds the records manifest. Any failure withdraws the publication, writes a failed receipt,
    re-derives it with every token the stop token and exits non-zero."""
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    spot = None
    try:
        manifest = read_json(output / "records-manifest.json")
        for entry in manifest["records"]:
            if sha256_of(ROOT / entry["artifact"]) != entry["sha256"]:
                raise ValueError(f"record {entry['artifact']} differs from records-manifest.json")
        if manifest != records_manifest(output):
            raise ValueError("records-manifest.json does not list exactly the current records")
        from world_model.planning import scoring_harness as harness
        with harness.deterministic_scoring():
            compute = compute_tables(output, plan)
        for name, text in rendered(plan, compute).items():
            if (output / name).read_bytes() != text.encode():
                raise ValueError(f"published {name} differs from the recomputation")
        spot = spot_check(output, plan)
        if not spot["pass"]:
            raise ValueError(f"re-scoring does not replicate the retained record: {spot}")
    except Exception as error:
        for name in PUBLISHED:
            (output / name).unlink(missing_ok=True)
        write_receipt(output, spot, f"{type(error).__name__}: {error}")
        write_publication(output, plan)
        log(f"validation FAILED: publication withdrawn and re-derived with every token {STOP_TOKEN}")
        raise
    write_receipt(output, spot)
    compute = write_publication(output, plan)
    if not compute["guards"]["global"]["G_records"]["pass"]:
        raise ValueError("the validation receipt does not pass G-records")
    log(f"validation passed ({time.monotonic() - began:.0f}s): {len(manifest['records'])} records hash-checked, tables "
        f"re-derived, {spot['arm']} seed {spot['seed']} re-scored (max delta {spot['max_abs_delta']}); disposition "
        f"{compute['disposition']['disposition']}")
    return 0


MODES = {"trigger": trigger, "smoke": smoke, "prepare": prepare, "freeze-readouts": freeze_readouts, "score": score,
         "publish": publish, "validate": validate, "gport-configs": gport_configs, "gport-reference": gport_reference,
         "gport": gport}
GPU_MODES = {"train": train, "readouts": readouts}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for mode in (*MODES, *GPU_MODES, "selection"):
        parser.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="queue / slot index (run with CUDA_VISIBLE_DEVICES set)")
    parser.add_argument("--workers", type=int, default=1, help="selection slots")
    args = parser.parse_args()
    if args.selection:
        return selection(OUTPUT, args.gpu, args.workers)
    for mode, fn in GPU_MODES.items():
        if getattr(args, mode.replace("-", "_")):
            return fn(OUTPUT, args.gpu)
    for mode, fn in MODES.items():
        if getattr(args, mode.replace("-", "_")):
            return fn(OUTPUT)
    parser.error("no mode given")


if __name__ == "__main__":
    raise SystemExit(main())
