"""Issue #129 (#127 child B1): SlotFormer-E99 against U on the sealed NovPhy launch-ranking grid.

Arms (Decisions P-other on #129; the #127 frozen arm matrix):
- **B1** (primary, program set): the official SlotFormer PHYRE slot Transformer (native size, 6 325 005 parameters)
  on E99 slots, launch token at t0, burn-in 1, rollout to frame 225; score = tie-free ``EndpointCosts`` of the
  predicted frame-225 carrier. Configuration selected on the policy split among ``CONFIGS`` (seed 20260908, full
  native budget); the selected run is reused as main seed 20260908.
- **B1-m** (secondary): the matched variant (1 793 445 parameters) at U's B1 recipe (9 000 x 64), every other
  hyperparameter the selected configuration's.
- **B1-head** (exploratory): SlotFormer's PHYRE readout (``PHYREReadout``) trained on B1's predicted fit endpoints;
  **U+head** (control): the same head on U's predicted fit endpoints (Gate-B unit, fit data only).

Modes (round3/child-runner-contract.md): --selection, --selection-readouts, --smoke, --prepare, --train,
--readouts, --freeze-readouts, --score, --publish, --validate. GPU work only through the night-window scheduler
(scripts/gpu_window_issue_129.json); tests use CUDA_VISIBLE_DEVICES="".
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import external_baselines as eb  # noqa: E402
from world_model.baselines.slotformer import novphy as sf  # noqa: E402
from world_model.planning import scoring_harness as harness  # noqa: E402

ISSUE = 129
IDENTITY = "issue-129-slotformer-v1"
OUTPUT = ROOT / ".local-artifacts" / IDENTITY
RUNNER = "scripts/run_issue_129_slotformer.py"
TESTS = "tests/test_issue_129_slotformer.py"
SCHEMA_MODEL = "issue_129_model_v1"
SCHEMA_HEAD = "issue_129_head_v1"
SCHEMA_PLAN = "issue_129_plan_v1"
SCHEMA_RECORD = "issue_129_record_v1"
SCHEMA_SMOKE = "issue_129_smoke_v1"
SCHEMA_SELECTION = "issue_129_selection_v1"
SCHEMA_READOUTS_MANIFEST = "issue_129_readouts_manifest_v1"
DECISIONS = {
    "license": "https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6382274253",
    "adapter": "https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6382359850",
    "route": "https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6382402424",
    "band": "https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6382402612",
    "selection": "https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6382590719",
}

BATCH = sf.BATCH  # 64, the PHYRE config's train_batch_size
EPOCHS = 50  # the PHYRE config's max_epochs
CONFIGS = {  # pre-declared (Decision P-other, selection): stride, lr, epochs; native size, seed 20260908
    "C1-s15": {"stride": 15, "lr": 2e-4, "epochs": 50},
    "C2-s5": {"stride": 5, "lr": 2e-4, "epochs": 50},
    "C3-s15-long": {"stride": 15, "lr": 2e-4, "epochs": 250},
}
SELECTION_QUEUES = {0: ["C1-s15"], 1: ["C2-s5"], 2: ["C3-s15-long"]}  # one config per scheduler job
MATCHED_RECIPE = {"steps": 9000, "batch": 64}  # U's B1 recipe
ARMS = {"B1": {"size": "native", "role": "primary", "program_set": True},
        "B1-m": {"size": "matched", "role": "secondary (matched budget and parameters)", "program_set": False}}
HEAD_ARMS = ("B1",)  # arms whose readouts train the native success head
CHECKPOINT_EVERY = 1000
LOG_EVERY = 50
CHILD_CAP_GPU_HOURS = 88.0  # the upper end of the #127 B1 child share
SEED_CAP_GPU_HOURS = 20.0  # #127: native cap per seed
WALL_CAP_HOURS = 48.0  # on 4 GPUs
HEAD_KEY = "head_neg_logit"
GRID = "grid"


def log(message):
    eb.log(ISSUE, message)


# ---------------------------------------------------------------------------------------------------------------
# configuration and paths
# ---------------------------------------------------------------------------------------------------------------

def steps_per_epoch(n_shots, batch=BATCH):
    """nerv's train loader drops the last partial batch."""
    return n_shots // batch


def config_spec(name, n_shots):
    c = CONFIGS[name]
    return {"name": name, "size": "native", "stride": c["stride"], "lr": c["lr"], "batch": BATCH,
            "epochs": c["epochs"], "steps": c["epochs"] * steps_per_epoch(n_shots), "warmup_pct": sf.WARMUP_STEPS_PCT}


def arm_spec(arm, selected, n_shots):
    """The training spec of an arm at the selected configuration."""
    base = config_spec(selected, n_shots)
    if arm == "B1":
        return {**base, "arm": "B1"}
    return {**base, "arm": arm, "name": f"{arm}@{selected}", "size": ARMS[arm]["size"],
            "steps": MATCHED_RECIPE["steps"], "batch": MATCHED_RECIPE["batch"], "epochs": None}


def selection_dir(output, name, seed=eb.SELECTION_SEED):
    return Path(output) / "selection" / name / f"seed-{seed}"


def model_dir(output, arm, seed, selected=None):
    """B1 seed 20260908 is the reused selection run of the selected configuration (#127 selection rule step 4)."""
    if arm == "B1" and seed == eb.SELECTION_SEED:
        if selected is None:
            raise ValueError("model_dir(B1, 20260908) needs the selected configuration")
        return selection_dir(output, selected)
    return Path(output) / "models" / arm / f"seed-{seed}"


def readouts_dir(output, arm, seed):
    return Path(output) / "readouts" / arm / f"seed-{seed}"


def record_path(output, arm, seed):
    return Path(output) / "records" / f"{arm}--seed{seed}.json"


def device_of(requested=None):
    return requested or ("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------------------------------------------
# fit data and batches
# ---------------------------------------------------------------------------------------------------------------

def fit_sequences(shots, stride, chunk=2048):
    """Every fit shot's training sequence (frames 0, stride, ..., 225; PHYRE padding) as the official data_dict
    tensors: slots float32 [N, T, 19, 13], vid_len int64 [N] (adapter ``training_batch``, shot order)."""
    slots, vid_len = [], []
    for start in range(0, len(shots), chunk):
        index = list(range(start, min(len(shots), start + chunk)))
        batch = sf.training_batch(shots.carrier_rows, index, shots.ends[index], shots.launch5[index], stride)
        slots.append(batch["slots"])
        vid_len.append(batch["vid_len"])
    return {"slots": torch.cat(slots), "vid_len": torch.cat(vid_len)}


class EpochBatches:
    """Batch b of a run: slice b mod spe of epoch floor(b / spe)'s permutation, drawn from eb.batch_rng(seed, epoch)
    (nerv: shuffled loader, drop_last). Deterministic and resumable at any step."""

    def __init__(self, seed, n, batch):
        self.seed, self.n, self.batch = seed, n, batch
        self.spe = steps_per_epoch(n, batch)
        if self.spe < 1:
            raise ValueError("fewer shots than one batch")
        self._epoch, self._perm = None, None

    def __call__(self, b):
        epoch, i = divmod(b, self.spe)
        if epoch != self._epoch:
            self._epoch, self._perm = epoch, eb.batch_rng(self.seed, epoch).permutation(self.n)
        return torch.from_numpy(self._perm[i * self.batch:(i + 1) * self.batch].copy())


# ---------------------------------------------------------------------------------------------------------------
# training (resumable; nonfinite loss -> typed terminal failure)
# ---------------------------------------------------------------------------------------------------------------

def _amp(device):
    """The official --fp16 path (nerv _training_step_fp16): fp16 autocast + GradScaler on CUDA; float32 on CPU."""
    return device.startswith("cuda")


def train_run(out_dir, spec, seed, data, device=None, max_steps=None):
    """Train one model (resumable from out_dir/resume.pt). data: fit_sequences(...) of spec's stride."""
    device = device_of(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "model.pt"
    if final.is_file():
        log(f"{spec['name']} seed {seed}: model.pt exists; skipped")
        return eb.read_json(out_dir / "report.json")
    amp = _amp(device)
    if amp:
        torch.backends.cudnn.benchmark = True  # the official --cudnn
        torch.cuda.reset_peak_memory_stats()  # peak_gib per run, not per process (several runs per --train queue)
    steps = spec["steps"]
    torch.manual_seed(seed)
    model = sf.SlotFormerE99(spec["size"], stride=spec["stride"],
                             rollout_len=spec.get("rollout_len")).to(device).train()  # None: to frame 225
    model.loss_decay_factor = 1.  # use_loss_decay False in the PHYRE config
    optimizer, scheduler = sf.build_optimizer(model, steps, spec["lr"], spec["warmup_pct"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, elapsed = 0, 0.0
    resume = out_dir / "resume.pt"
    if resume.is_file():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, elapsed = state["step"], state["elapsed"]
        torch.set_rng_state(state["cpu_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(state["cuda_rng"])
        log(f"{spec['name']} seed {seed}: resumed at step {start}")
    slots, vid_len = data["slots"].to(device), data["vid_len"].to(device)
    batches = EpochBatches(seed, len(slots), spec["batch"])
    stop_at = steps if max_steps is None else min(steps, start + max_steps)
    began, failure, trace, step = time.monotonic(), None, [], start - 1
    with open(out_dir / "train-log.jsonl", "a") as log_file:
        for step in range(start, stop_at):
            index = batches(step).to(device)
            batch = {"slots": slots[index], "vid_len": vid_len[index]}
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss, out = model.total_loss(batch)
            if not bool(torch.isfinite(loss)):
                failure = {"typed_failure": "nonfinite_loss", "step": step, "message": "nonfinite slot_recon_loss"}
                break
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            if step % LOG_EVERY == 0 or step == steps - 1:
                entry = {"step": step, "lr": optimizer.param_groups[0]["lr"], "loss": float(loss),
                         "seconds": elapsed + time.monotonic() - began, "scale": scaler.get_scale() if amp else None}
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()
                trace.append(entry)
            stop = bool(eb.STOP_REQUESTED) and step + 1 < steps
            if stop or ((step + 1) % CHECKPOINT_EVERY == 0 and step + 1 < steps):
                eb.atomic_torch(resume, {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    "step": step + 1, "elapsed": elapsed + time.monotonic() - began})
            if stop:
                log(f"{spec['name']} seed {seed}: stop signal; resume.pt written at step {step + 1}")
                raise eb.TrainingPaused(f"{spec['name']} seed {seed} paused at step {step + 1} of {steps}")
    seconds = elapsed + time.monotonic() - began
    if failure is None and step + 1 < steps:  # max_steps reached (tests, throughput)
        return {"partial": True, "steps_done": step + 1 - start, "seconds": time.monotonic() - began}
    complete = failure is None
    payload = {"schema": SCHEMA_MODEL, "spec": spec, "seed": seed, "complete": complete, "typed_failure": failure,
               "state_dict": {k: v.cpu() for k, v in model.state_dict().items()} if complete else None,
               "parameters": sf.trainable_parameters(model)}
    eb.atomic_torch(final, payload)
    report = {"spec": spec, "seed": seed, "steps": steps, "complete": complete, "typed_failure": failure,
              "wall_seconds": seconds, "gpu_seconds": seconds if device.startswith("cuda") else 0.0,
              "device": device, "seconds_per_step": seconds / max(1, steps),
              "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None,
              "model_sha256": eb.sha256_of(final), "parameters": payload["parameters"],
              "final": trace[-1] if trace else None, "finished_at": eb.utc_now()}
    eb.write_json(out_dir / "report.json", report)
    if resume.is_file() and complete:
        resume.unlink()
    log(f"{spec['name']} seed {seed}: {'complete' if complete else failure['typed_failure']} after {seconds:.0f}s")
    return report


def load_trained(path, device=None):
    """(model, payload) of a model.pt, or (None, payload) for a typed failure."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA_MODEL:
        raise ValueError(f"{path}: schema differs")
    if not payload["complete"]:
        return None, payload
    spec = payload["spec"]
    model = sf.SlotFormerE99(spec["size"], stride=spec["stride"], rollout_len=spec.get("rollout_len"))
    model.load_state_dict(payload["state_dict"])
    return model.to(device_of(device)).eval(), payload


def run_queue(output, items, device=None):
    """Train (spec, seed, out_dir) items in order; a stop signal pauses the queue (exit 75)."""
    eb.install_stop_handlers()
    shots = eb.FitShots()
    cache = {}
    for spec, seed, out_dir in items:
        if (Path(out_dir) / "model.pt").is_file():
            continue
        if spec["stride"] not in cache:
            cache[spec["stride"]] = fit_sequences(shots, spec["stride"])
        began = time.monotonic()
        try:
            report = train_run(out_dir, spec, seed, cache[spec["stride"]], device)
        except eb.TrainingPaused as paused:
            seconds = time.monotonic() - began
            eb.mark(output, f"train-{spec['name']}-seed{seed}", seconds, seconds if device_of(device) != "cpu" else 0)
            log(str(paused))
            return eb.EXIT_PAUSED
        seconds = time.monotonic() - began
        eb.mark(output, f"train-{spec['name']}-seed{seed}", seconds, seconds if device_of(device) != "cpu" else 0,
                report=str(Path(out_dir, "report.json").relative_to(ROOT)), complete=report["complete"])
    return 0


# ---------------------------------------------------------------------------------------------------------------
# prediction, costs, policy AUC (deterministic, float32)
# ---------------------------------------------------------------------------------------------------------------

def objective_costs():
    from scripts import run_lookahead_control as wlc
    return harness.EndpointCosts(wlc.load_objective())


@torch.no_grad()
def rollout_carriers(model, z0, actions, batch=4096):
    """Predicted carriers [B, steps, 236] (frames stride..225) from anchors z0 [B, 236] under launches [B, 5]."""
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(z0), batch):
        out.append(model.predict_carriers(z0[i:i + batch].to(device), actions[i:i + batch].to(device)).float().cpu())
    return torch.cat(out)


def grid_actions():
    items = eb.grid_items()
    return items, torch.stack([eb.launch5(item["action"]) for item in items])


@torch.no_grad()
def fit_anchor_endpoints(model):
    """Predicted frame-225 carriers [480, 16, 236] of the 16 grid launches from the 480 fit anchors (G-rank,
    G-action) and the anchor carriers [480, 236] (sorted members)."""
    anchors = eb.fit_anchors()["carriers"]
    members = sorted(anchors)
    items, actions = grid_actions()
    z0 = torch.stack([anchors[m] for m in members])
    ends = rollout_carriers(model, z0.repeat_interleave(len(items), 0), actions.repeat(len(members), 1))[:, -1]
    return members, z0, ends.reshape(len(members), len(items), -1)


def tie_free_matrix(ends, z0, costs):
    """[L, K] tie-free costs (nonfinite -> nan) of endpoints [L, K, 236] from anchors [L, 236]."""
    out = np.full(ends.shape[:2], np.nan)
    items, _ = grid_actions()
    for i in range(ends.shape[0]):
        for row in eb.endpoint_rows(items, ends[i], z0[i], costs):
            if not row["excluded"]:
                out[i, [it["ordinal"] for it in items].index(row["ordinal"])] = row[eb.PRIMARY_COST]
    return out


@torch.no_grad()
def policy_endpoint_rows(model, costs, head=None):
    """{member: rows} of the policy grid mixed cells: tie-free cost rows of the predicted endpoints (and the head's
    score as HEAD_KEY when given)."""
    anchors = eb.policy_anchors()["carriers"]
    out = {}
    for member, entry in sorted(eb.policy_cells(GRID).items()):
        items = entry["inventory"]
        actions = torch.stack([eb.launch5(item["action"]) for item in items])
        z0 = anchors[member]
        ends = rollout_carriers(model, z0[None].expand(len(items), -1), actions)[:, -1]
        extra = None
        if head is not None:
            extra = {HEAD_KEY: (-head_logits(head, z0[None].expand(len(items), -1), ends, actions)).tolist()}
        out[member] = eb.rc.cost_rows(items, ends, z0, costs, extra)
    return out


# ---------------------------------------------------------------------------------------------------------------
# native success head (SlotFormer's PHYRE readout recipe) and the U + head control
# ---------------------------------------------------------------------------------------------------------------

def head_steps(n):
    return steps_per_epoch(n, sf.HEAD_BATCH) * sf.HEAD_EPOCHS


def train_head(anchors, ends, launch5, labels, seed, device=None, steps=None):
    """Train the PHYRE readout on head_input(anchor, predicted end, launch) with the engine verdicts; official recipe
    (Adam 1e-3, 10 % warm-up, cosine, batch 256, 50 epochs, fp16 on CUDA). Returns (head, report)."""
    device = device_of(device)
    amp = _amp(device)
    x = sf.head_input(anchors, ends, launch5).to(device)
    y = labels.float().to(device)
    steps = head_steps(len(x)) if steps is None else steps
    torch.manual_seed(seed)
    head = sf.build_head().to(device).train()
    optimizer, scheduler = sf.build_optimizer(head, steps, sf.HEAD_LR, sf.HEAD_WARMUP_PCT)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    batches = EpochBatches(seed, len(x), sf.HEAD_BATCH)
    began, failure, losses = time.monotonic(), None, []
    for step in range(steps):
        index = batches(step).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss, _ = sf.head_total_loss(head, {"slots": x[index], "label": y[index]})
        if not bool(torch.isfinite(loss)):
            failure = {"typed_failure": "nonfinite_loss", "step": step}
            break
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        if step % LOG_EVERY == 0 or step == steps - 1:
            losses.append({"step": step, "loss": float(loss)})
    head.eval()
    seconds = time.monotonic() - began
    return head, {"steps": steps, "batch": sf.HEAD_BATCH, "typed_failure": failure, "seconds": seconds,
                  "train_trace": losses, "parameters": sf.trainable_parameters(head), "positives": int(labels.sum()),
                  "examples": int(len(labels))}


@torch.no_grad()
def head_logits(head, anchors, ends, launch5):
    device = next(head.parameters()).device
    x = sf.head_input(anchors, ends, launch5).to(device)
    return head({"slots": x})["logits"].float().cpu()


def load_head(path, device=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA_HEAD:
        raise ValueError(f"{path}: schema differs")
    if payload["typed_failure"]:
        return None, payload
    head = sf.build_head()
    head.load_state_dict(payload["state_dict"])
    return head.to(device_of(device)).eval(), payload


class UnitRollout:
    """Gate-B unit U (gb.load_model("unit", seed)) rolled as Gate-B / eb.reference_compute roll it: the request's
    fixed pair from the anchor carrier under the launch, to frame 225."""

    def __init__(self, seed, device=None):
        from scripts import run_tau_ad_within_checkpoint as t96
        from world_model.training.matched_dynamics import pairs_for
        self.device = device_of(device)
        if self.device.startswith("cuda"):
            model, failure = eb.gb.load_model("unit", seed)  # loads onto CUDA (v1.load_member)
        else:  # CPU (tests / dry runs): the same handoff checkpoint and binding checks, loaded on the CPU
            model, failure = eb._reference_predictor("unit", seed), None
        self.failure = failure
        if model is None:
            self.model = None
            return
        self.model = model.to(self.device).eval()
        self.pair = pairs_for(model)[t96.arm_spec(eb.gb.UNITS["unit"]["request"])["pair_index"]]

    @torch.no_grad()
    def endpoints(self, z0, actions, batch=4096):
        out = []
        for i in range(0, len(z0), batch):
            z = z0[i:i + batch].to(self.device)
            a = actions[i:i + batch].to(self.device)
            position = 0
            while position < sf.ENDPOINT:
                z = self.model.carrier(z, a, self.pair)
                position += self.pair.delta
            out.append(z.float().cpu())
        return torch.cat(out)


def fit_head_inputs(shots):
    """Level anchor [N, 236] and launch [N, 5] of every fit shot (the head is trained from the level anchors)."""
    anchors = eb.fit_anchors()["carriers"]
    return torch.stack([anchors[m] for m in shots.members]), shots.launch5


# ---------------------------------------------------------------------------------------------------------------
# --selection / --selection-readouts
# ---------------------------------------------------------------------------------------------------------------

def selection(output, queue, device=None):
    shots = eb.FitShots()
    names = SELECTION_QUEUES[queue]
    items = [(config_spec(name, len(shots)), eb.SELECTION_SEED, selection_dir(output, name)) for name in names]
    log(f"selection queue {queue}: {names}")
    return run_queue(output, items, device)


def selection_readouts(output, device=None):
    """Policy-split pooled grid AUC of the primary score per configuration (deterministic scoring); selection.json."""
    costs = objective_costs()
    results, details = {}, {}
    began = time.monotonic()
    with harness.deterministic_scoring() as policy:
        for name in CONFIGS:
            path = selection_dir(output, name) / "model.pt"
            if not path.is_file():
                raise ValueError(f"selection run {name} is not finished")
            model, payload = load_trained(path, device)
            if model is None:
                results[name] = {"auc": None, "parameters": payload["parameters"]}
                details[name] = {"typed_failure": payload["typed_failure"]}
                continue
            rows = policy_endpoint_rows(model, costs)
            auc = eb.policy_grid_auc(rows, eb.PRIMARY_COST)
            results[name] = {"auc": auc, "parameters": payload["parameters"]}
            details[name] = {"typed_failure": None, "model_sha256": eb.sha256_of(path), "cells": len(rows)}
    choice = eb.select_configuration(results)
    seconds = time.monotonic() - began
    eb.mark(output, "selection-readouts", seconds, seconds if device_of(device) != "cpu" else 0)
    eb.write_json(Path(output) / "selection.json", {
        "schema": SCHEMA_SELECTION, "plan_identity": IDENTITY, "configs": CONFIGS, **choice, "details": details,
        "scoring_policy": policy, "policy_cells": len(eb.policy_cells(GRID)), "written_at": eb.utc_now()})
    log(f"selection: {choice['selected']} ({json.dumps(results)})")
    return 0 if choice["selected"] is not None else 1


def selected_config(output):
    path = Path(output) / "selection.json"
    if not path.is_file():
        raise ValueError("selection.json missing; run --selection-readouts")
    selected = eb.read_json(path)["selected"]
    if selected is None:
        raise ValueError("no configuration was selected (every selection run failed)")
    return selected


# ---------------------------------------------------------------------------------------------------------------
# --smoke
# ---------------------------------------------------------------------------------------------------------------

def lpt(items, bins=4):
    """Longest-processing-time assignment: items [(name, seconds)] -> {bin: [names]}, loads."""
    queues, loads = {b: [] for b in range(bins)}, [0.0] * bins
    for name, seconds in sorted(items, key=lambda t: (-t[1], t[0])):
        b = int(np.argmin(loads))
        queues[b].append(name)
        loads[b] += seconds
    return queues, loads


def measure_seconds_per_step(spec, data, device, steps=200):
    """GPU seconds per optimizer step of spec (a throwaway run, never saved)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        warm = train_run(tmp, spec, 0, data, device, max_steps=20)
        if not warm.get("partial"):
            return None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        began = time.monotonic()
        report = train_run(tmp, spec, 0, data, device, max_steps=steps)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.monotonic() - began) / report.get("steps_done", steps)


def remaining_items(output, selected, n_shots):
    """(arm, seed, spec, out_dir) of every training run the plan still needs."""
    out = []
    for arm in ARMS:
        for seed in eb.SEEDS:
            if arm == "B1" and seed == eb.SELECTION_SEED:
                continue
            out.append((arm, seed, arm_spec(arm, selected, n_shots), model_dir(output, arm, seed)))
    return out


def smoke(output, device=None):
    device = device_of(device)
    began = time.monotonic()
    shots = eb.FitShots()
    selected = selected_config(output)
    selection_record = eb.read_json(Path(output) / "selection.json")
    accounting = {"fit_shots": len(shots), "fit_levels": len(shots.levels), "fit_positive": int(shots.verdicts.sum()),
                  "fit_anchors": len(eb.fit_anchors()["carriers"]), "policy_grid_mixed_cells": len(eb.policy_cells(GRID))}
    for stride in sorted({c["stride"] for c in CONFIGS.values()}):
        _, vid_len = sf.sequence_plan(shots.ends, stride)
        accounting[f"stride_{stride}"] = {"frames": len(sf.positions(stride)), "short_shots": int(
            (vid_len < len(sf.positions(stride))).sum()), "masked_target_share": float(
            (len(sf.positions(stride)) - vid_len).sum() / (len(vid_len) * sf.rollout_steps(stride)))}
    gport = eb.read_json(Path(output) / "gport/gport.json")
    gport_check = {"G_port": gport["G_port"], "G_port_rule": gport.get("G_port_rule"),
                   "adapter_current": gport["adapter"] == eb.sha256_of(ROOT / "world_model/baselines/slotformer/novphy.py"),
                   "reference_torch": gport.get("reference_env"), "arms": sorted(gport["arms"])}
    gport_check["pass"] = bool(gport_check["G_port"] is True and gport_check["G_port_rule"] == "issue_123_exact"
                               and all(a["pass_123_exact"] for a in gport["arms"].values())
                               and gport_check["adapter_current"])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", TESTS], cwd=ROOT, env=env,
                           capture_output=True, text=True)
    tests_record = {"command": f"CUDA_VISIBLE_DEVICES= python -m pytest -q {TESTS}", "exit_code": tests.returncode,
                    "tail": tests.stdout.strip().splitlines()[-1:] if tests.stdout else []}
    # throughput: measured selection runs + a short timed run of the matched spec and the head
    throughput = {}
    for name in CONFIGS:
        report = eb.read_json(selection_dir(output, name) / "report.json")
        throughput[name] = {"seconds_per_step": report["seconds_per_step"], "steps": report["steps"],
                            "gpu_seconds": report["gpu_seconds"]}
    matched = arm_spec("B1-m", selected, len(shots))
    throughput["B1-m"] = {"seconds_per_step": measure_seconds_per_step(
        matched, fit_sequences(shots, matched["stride"]), device), "steps": matched["steps"]}
    remaining = remaining_items(output, selected, len(shots))
    native_rate = throughput[selected]["seconds_per_step"]
    training = [(f"{arm}--seed{seed}", spec["steps"] * (native_rate if arm == "B1" else
                                                         throughput["B1-m"]["seconds_per_step"]))
                for arm, seed, spec, _ in remaining]
    selection_seconds = sum(t["gpu_seconds"] for t in throughput.values() if "gpu_seconds" in t)
    readout_seconds = 3 * 3600.0  # heads, U + head rollouts, G-rank / G-action, policy AUCs (estimate)
    score_seconds = 2 * 3600.0  # one scoring pass incl. wrong anchors and positions (estimate)
    # exploratory benchmarks: 2 x 3 runs of 9 000 steps, BENCH_HORIZON / stride rollout steps instead of 225 / stride
    bench_seconds = (len(bench_jobs()) * MATCHED_RECIPE["steps"] * native_rate
                     * (BENCH_HORIZON / sf.ENDPOINT))
    queues, loads = lpt(training)
    total = selection_seconds + sum(s for _, s in training) + readout_seconds + score_seconds + bench_seconds
    per_seed_native = config_spec(selected, len(shots))["steps"] * native_rate
    wall = max(loads) + readout_seconds / 4 + score_seconds + bench_seconds / BENCH_QUEUES
    projection = {"selection_gpu_h": selection_seconds / 3600, "training_gpu_h": sum(s for _, s in training) / 3600,
                  "readouts_gpu_h_estimate": readout_seconds / 3600, "score_gpu_h_estimate": score_seconds / 3600,
                  "benchmarks_gpu_h_estimate": bench_seconds / 3600,
                  "total_gpu_h": total / 3600, "child_cap_gpu_h": CHILD_CAP_GPU_HOURS,
                  "native_per_seed_gpu_h": per_seed_native / 3600, "seed_cap_gpu_h": SEED_CAP_GPU_HOURS,
                  "wall_h_on_4_gpus": wall / 3600, "wall_cap_h": WALL_CAP_HOURS,
                  "training_queues": queues, "training_queue_loads_h": [x / 3600 for x in loads]}
    projection["within_caps"] = bool(total / 3600 <= CHILD_CAP_GPU_HOURS and per_seed_native / 3600 <= SEED_CAP_GPU_HOURS
                                     and wall / 3600 <= WALL_CAP_HOURS)
    smoke_record = {"schema": SCHEMA_SMOKE, "plan_identity": IDENTITY, "accounting": accounting, "gport": gport_check,
                    "tests": tests_record, "throughput": throughput, "selection": selection_record,
                    "selected": selected, "projection": projection, "device": device,
                    "pass": bool(gport_check["pass"] and tests.returncode == 0 and projection["within_caps"]),
                    "written_at": eb.utc_now()}
    seconds = time.monotonic() - began
    eb.mark(output, "smoke", seconds, seconds if device != "cpu" else 0)
    eb.write_json(Path(output) / "smoke.json", smoke_record)
    log(f"smoke: pass {smoke_record['pass']}; projection {projection['total_gpu_h']:.1f} GPU-h, wall "
        f"{projection['wall_h_on_4_gpus']:.1f} h; selected {selected}")
    return 0 if smoke_record["pass"] else 1


# ---------------------------------------------------------------------------------------------------------------
# --prepare (freeze) and load_plan
# ---------------------------------------------------------------------------------------------------------------

OWN_FILES = [RUNNER, TESTS, "world_model/baselines/slotformer/__init__.py", "world_model/baselines/slotformer/novphy.py",
             "scripts/b127_slotformer_port_reference.py", "scripts/b127_slotformer_fidelity.py"]


def own_pins(output):
    output = Path(output)
    paths = [ROOT / p for p in OWN_FILES]
    paths += sorted((ROOT / "world_model/baselines/slotformer/vendor").glob("*"))
    paths += [output / "gport/gport.json", output / "fidelity/band.json", output / "fidelity/gfid.json",
              output / "smoke.json", output / "selection.json"]  # decisions.md is append-only: not pinned
    paths += [selection_dir(output, name) / f for name in CONFIGS for f in ("model.pt", "report.json")]
    return [{"name": f"b1_{p.relative_to(ROOT)}", "artifact": str(p.relative_to(ROOT)), "sha256": eb.sha256_of(p)}
            for p in paths if p.is_file() and "__pycache__" not in str(p)]


def readout_items():
    return [(arm, seed) for arm in (*ARMS, "U+head") for seed in eb.SEEDS]


def prepare(output):
    output = Path(output)
    path = output / "plan.json"
    if path.is_file():
        load_plan(output)
        log("plan.json is frozen; pins re-verified")
        return 0
    smoke_record = eb.read_json(output / "smoke.json")
    if not smoke_record["pass"]:
        raise SystemExit("smoke did not pass; no freeze (P-cap / fix first)")
    shots = eb.FitShots()
    selected = smoke_record["selected"]
    queues = smoke_record["projection"]["training_queues"]
    readout_queues, _ = lpt([(f"{a}--seed{s}", 1.0 if a == "B1-m" else 2.0) for a, s in readout_items()])
    plan = {
        "schema": SCHEMA_PLAN, "plan_identity": IDENTITY, "issue": ISSUE, "program_issue": eb.PROGRAM_ISSUE,
        "frozen_at": eb.utc_now(), "pins": eb.shared_inputs() + own_pins(output),
        "selected": selected, "selection_rule": eb.SELECTION_RULE, "configs": CONFIGS,
        "arms": {arm: {**ARMS[arm], "spec": arm_spec(arm, selected, len(shots)),
                       "seeds": list(eb.SEEDS)} for arm in ARMS},
        "reused": {"B1": {"seed": eb.SELECTION_SEED, "run": str(selection_dir(output, selected).relative_to(ROOT))}},
        "head": {"arms": list(HEAD_ARMS), "control": "U+head", "dict": sf.HEAD_DICT, "lr": sf.HEAD_LR,
                 "warmup_pct": sf.HEAD_WARMUP_PCT, "batch": sf.HEAD_BATCH, "epochs": sf.HEAD_EPOCHS,
                 "steps": head_steps(len(shots)), "input": "head_input(level anchor, predicted frame-225 carrier, launch5)"},
        "benchmarks_exploratory": {bench: bench_spec(selected, bench) for bench in BENCHES} | {
            "times": list(BENCH_TIMES), "contexts": list(BENCH_CONTEXTS), "metric": "#79 field_errors position_mse",
            "reference": "#117 item-5 J (LC, full, fold all), F-15-continuous / F-5-continuous", "tokens": None},
        "queues": {"train": {str(k): v for k, v in queues.items()},
                   "readouts": {str(k): v for k, v in readout_queues.items()}, "score": "GPU 0, one process"},
        "seeds": list(eb.SEEDS), "guards": eb.GUARD,
        "arm_guards": ["G_rank", "G_action", "G_nonfinite", "G_trained"], "extra_guards": ["G_port", "G_fid"],
        "global_guards": ["G_isolation", "G_chronology", "G_scoring_policy", "G_records"],
        "contrasts": {"per_arm": list(eb.CONTRAST_REFERENCES), "within_family": ["B1 - B1-m", "B1-head - B1",
                                                                             "U+head - U"],
                      "wrong_anchor": ["B1", "B1-m"]},
        "dispositions": eb.DISPOSITION_LABELS, "delta": eb.DELTA,
        "decisions": DECISIONS,
        "disclosures": [
            "B1's configuration was selected on the policy split among three pre-declared configurations; U had no "
            "such tuning (#127).",
            "The native schedule is SlotFormer's PHYRE schedule in epochs (50) scaled to the 13 919 fit sequences.",
            "The head reads the anchor and the predicted frame-225 state (#127), not the official [0, 3] frames.",
            "Runs use fp16 autocast with GradScaler (the official --fp16).",
        ],
    }
    eb.write_json(path, plan)
    log(f"plan.json frozen ({len(plan['pins'])} pins); selected {selected}")
    return 0


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = eb.read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("plan_identity") != IDENTITY:
        raise ValueError("plan.json binding differs")
    eb.verify_pins(plan["pins"])
    return plan


# ---------------------------------------------------------------------------------------------------------------
# --train
# ---------------------------------------------------------------------------------------------------------------

def train(output, queue, device=None):
    plan = load_plan(output)
    names = plan["queues"]["train"].get(str(queue), [])
    items = []
    for name in names:
        arm, seed = name.split("--seed")
        seed = int(seed)
        items.append((plan["arms"][arm]["spec"], seed, model_dir(output, arm, seed, plan["selected"])))
    log(f"train queue {queue}: {names}")
    return run_queue(output, items, device)


def training_report_of(output, plan):
    """ledger phase -> its training report (each arm-seed counted once; #123 A3 rule)."""
    def report_of(phase):
        if not phase.startswith("train-"):
            return None
        name, seed = phase[len("train-"):].rsplit("-seed", 1)
        for arm in ARMS:
            spec = plan["arms"][arm]["spec"] if plan else None
            if spec and spec["name"] == name:
                return model_dir(output, arm, int(seed), plan["selected"]) / "report.json"
        if name in CONFIGS:
            return selection_dir(output, name, int(seed)) / "report.json"
        return None
    return report_of


# ---------------------------------------------------------------------------------------------------------------
# --readouts (fit / policy data only) and --freeze-readouts
# ---------------------------------------------------------------------------------------------------------------

def readouts_arm(output, plan, arm, seed, device=None):
    out_dir = readouts_dir(output, arm, seed)
    if (out_dir / "readouts.json").is_file():
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_of(device)
    costs = objective_costs()
    began = time.monotonic()
    shots = eb.FitShots()
    result = {"arm": arm, "seed": seed, "plan_identity": IDENTITY}
    with harness.deterministic_scoring() as policy:
        result["scoring_policy"] = policy
        if arm == "U+head":
            unit = UnitRollout(seed, device)
            if unit.model is None:
                result.update(typed_failure=unit.failure)
            else:
                anchors, launch = fit_head_inputs(shots)
                ends = unit.endpoints(anchors, launch)
                head, report = train_head(anchors, ends, launch, shots.verdicts, seed, device)
                eb.atomic_torch(out_dir / "head.pt", {"schema": SCHEMA_HEAD, "arm": arm, "seed": seed,
                                                      "typed_failure": report["typed_failure"],
                                                      "state_dict": head.state_dict()})
                rows = {}
                panchors = eb.policy_anchors()["carriers"]
                for member, entry in sorted(eb.policy_cells(GRID).items()):
                    items = entry["inventory"]
                    actions = torch.stack([eb.launch5(it["action"]) for it in items])
                    z0 = panchors[member][None].expand(len(items), -1)
                    pends = unit.endpoints(z0, actions)
                    rows[member] = eb.score_rows(items, head_logits(head, z0, pends, actions), key=HEAD_KEY)
                result.update(typed_failure=report["typed_failure"], head=report,
                              policy_head_auc=eb.policy_grid_auc(rows, HEAD_KEY),
                              head_sha256=eb.sha256_of(out_dir / "head.pt"))
        else:
            path = model_dir(output, arm, seed, plan["selected"]) / "model.pt"
            model, payload = load_trained(path, device)
            result["model"] = str(path.relative_to(ROOT))
            result["model_sha256"] = eb.sha256_of(path)
            if model is None:
                result.update(typed_failure=payload["typed_failure"]["typed_failure"])
            else:
                members, z0, ends = fit_anchor_endpoints(model)
                matrix = tie_free_matrix(ends, z0, costs)
                result.update(typed_failure=None, G_rank=eb.g_rank(matrix),
                              G_action=float(eb.action_sensitivity(ends)),
                              fit_anchor_nonfinite_share=float(np.isnan(matrix).mean()),
                              policy_auc=eb.policy_grid_auc(policy_endpoint_rows(model, costs), eb.PRIMARY_COST),
                              trainable_parameters=sf.trainable_parameters(model))
                items, actions = grid_actions()
                result["macs_per_candidate"] = eb.macs_per_candidate(
                    model.predict_endpoint, z0[0][None].expand(len(items), -1).to(device), actions.to(device),
                    candidates=len(items))
                if arm in HEAD_ARMS:
                    anchors, launch = fit_head_inputs(shots)
                    fends = rollout_carriers(model, anchors, launch)[:, -1]
                    head, report = train_head(anchors, fends, launch, shots.verdicts, seed, device)
                    eb.atomic_torch(out_dir / "head.pt", {"schema": SCHEMA_HEAD, "arm": arm, "seed": seed,
                                                          "typed_failure": report["typed_failure"],
                                                          "state_dict": head.state_dict()})
                    prow = policy_endpoint_rows(model, costs, head)
                    result.update(head=report, policy_head_auc=eb.policy_grid_auc(prow, HEAD_KEY),
                                  head_sha256=eb.sha256_of(out_dir / "head.pt"))
    seconds = time.monotonic() - began
    result.update(seconds=seconds, written_at=eb.utc_now())
    eb.mark(output, f"readouts-{arm}-seed{seed}", seconds, seconds if device != "cpu" else 0)
    eb.write_json(out_dir / "readouts.json", result)  # written last (atomic)
    log(f"readouts {arm} seed {seed}: {json.dumps({k: result.get(k) for k in ('policy_auc', 'policy_head_auc', 'G_action')})}")
    return 0


def readouts(output, queue, device=None):
    plan = load_plan(output)
    for name in plan["queues"]["readouts"].get(str(queue), []):
        arm, seed = name.split("--seed")
        readouts_arm(output, plan, arm, int(seed), device)
    return 0


def freeze_readouts(output):
    output = Path(output)
    plan = load_plan(output)
    path = output / "readouts-manifest.json"
    if path.is_file():
        load_manifest(output)
        log("readouts are frozen; manifest re-verified")
        return 0
    if eb.evaluation_anchors_path(output).is_file():
        raise SystemExit("evaluation anchors already read: G-chronology would fail")
    entries = {}
    for arm, seed in readout_items():
        rdir = readouts_dir(output, arm, seed)
        if not (rdir / "readouts.json").is_file():
            raise SystemExit(f"readouts of {arm} seed {seed} missing")
        entry = {"readouts": str((rdir / "readouts.json").relative_to(ROOT)),
                 "readouts_sha256": eb.sha256_of(rdir / "readouts.json")}
        if arm != "U+head":
            mpath = model_dir(output, arm, seed, plan["selected"]) / "model.pt"
            entry.update(model=str(mpath.relative_to(ROOT)), model_sha256=eb.sha256_of(mpath))
        if (rdir / "head.pt").is_file():
            entry.update(head=str((rdir / "head.pt").relative_to(ROOT)), head_sha256=eb.sha256_of(rdir / "head.pt"))
        entries[f"{arm}--seed{seed}"] = entry
    eb.write_json(path, {"schema": SCHEMA_READOUTS_MANIFEST, "plan_identity": IDENTITY, "frozen_at": eb.utc_now(),
                         "entries": entries})
    log(f"readouts frozen ({len(entries)} arm-seeds)")
    return 0


def load_manifest(output):
    path = Path(output) / "readouts-manifest.json"
    if not path.is_file():
        raise ValueError("readouts are not frozen; run --freeze-readouts first")
    manifest = eb.read_json(path)
    for name, entry in manifest["entries"].items():
        for kind in ("model", "readouts", "head"):
            if kind in entry and eb.sha256_of(ROOT / entry[kind]) != entry[f"{kind}_sha256"]:
                raise ValueError(f"{name}: frozen {kind} changed")
    return manifest


# ---------------------------------------------------------------------------------------------------------------
# --score (once; a process that did not train; GPU 0; deterministic)
# ---------------------------------------------------------------------------------------------------------------

def _position_rows(at, branches, cache, times):
    positions = {}
    for b, (_, item) in enumerate(branches):
        e = cache["entries"][item["key"]]
        positions[item["key"]] = {}
        for j, t in enumerate(times):
            positions[item["key"]][str(t)] = {
                "pred": eb.carrier_position_mse(at[t][b], e["presence"][j], e["centers"][j]), "floor": None}
    return positions


@torch.no_grad()
def score_arm(output, plan, arm, seed, anchors, cells, device=None):
    """One record: rows of every mixed cell of every inventory, wrong-anchor rows (480 fit anchors, grid), position
    error (descriptive), wall time per decision."""
    device = device_of(device)
    costs = objective_costs()
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "arm": arm, "seed": seed}
    rdir = readouts_dir(output, arm, seed)
    head = None
    if (rdir / "head.pt").is_file():
        head, _ = load_head(rdir / "head.pt", device)
    if arm == "U+head":
        unit = UnitRollout(seed, device)
        if unit.model is None or head is None:
            return {**base, "typed_failure": unit.failure or "head_typed_failure"}
        predict = unit.endpoints
    else:
        model, payload = load_trained(model_dir(output, arm, seed, plan["selected"]) / "model.pt", device)
        if model is None:
            return {**base, "typed_failure": payload["typed_failure"]["typed_failure"]}

        def predict(z0, actions):
            return rollout_carriers(model, z0, actions)[:, -1]
    carriers = anchors["carriers"]
    inventories = {}
    for inventory in eb.INVENTORIES:
        inventories[inventory] = {}
        for member, entry in sorted(cells[inventory]["mixed"].items()):
            items = entry["inventory"]
            actions = torch.stack([eb.launch5(it["action"]) for it in items])
            z0 = carriers[member][None].expand(len(items), -1)
            ends = predict(z0, actions)
            if arm == "U+head":
                rows = eb.score_rows(items, head_logits(head, z0, ends, actions), key=HEAD_KEY)
            else:
                extra = {HEAD_KEY: (-head_logits(head, z0, ends, actions)).tolist()} if head is not None else None
                rows = eb.rc.cost_rows(items, ends, carriers[member], costs, extra)
            inventories[inventory][member] = {"rows": rows}
    record = {**base, "typed_failure": None, "inventories": inventories}
    if arm != "U+head":
        fit = eb.fit_anchors()["carriers"]
        items, actions = grid_actions()
        record["wrong_anchor"] = {m: {"rows": eb.endpoint_rows(items, predict(fit[m][None].expand(len(items), -1),
                                                                              actions), fit[m], costs)}
                                  for m in sorted(fit)}
        cache = eb.position_cache()
        branches = eb.position_branches(cells)
        z0 = torch.stack([carriers[m] for m, _ in branches])
        acts = torch.stack([eb.launch5(it["action"]) for _, it in branches])
        rolled = rollout_carriers(model, z0, acts)
        stride = model.stride
        at = {t: rolled[:, t // stride - 1] for t in eb.POSITION_TIMES}
        record["positions"] = _position_rows(at, branches, cache, eb.POSITION_TIMES)
        members = sorted(cells[GRID]["mixed"])[:20]
        grid_items, grid_acts = grid_actions()
        record["wall_per_decision"] = eb.wall_per_decision(
            lambda m: predict(carriers[m][None].expand(len(grid_items), -1), grid_acts), members)
    return record


def score(output, device=None):
    output = Path(output)
    plan = load_plan(output)
    load_manifest(output)
    readouts_frozen_at = eb.read_json(output / "readouts-manifest.json")["frozen_at"]
    with harness.deterministic_scoring() as policy:
        anchors_path = eb.evaluation_anchors_path(output)
        anchors = (torch.load(anchors_path, map_location="cpu", weights_only=True) if anchors_path.is_file()
                   else eb.read_evaluation_anchors(output, policy))
        cells = eb.evaluation_cells()
        for arm, seed in readout_items():
            path = record_path(output, arm, seed)
            if path.is_file():
                continue
            began = time.monotonic()
            record = score_arm(output, plan, arm, seed, anchors, cells, device)
            seconds = time.monotonic() - began
            record.update(scoring_policy=policy, scored_at=eb.utc_now(), readouts_frozen_at=readouts_frozen_at,
                          anchors_read_at=anchors["read_at"], score_seconds=seconds)
            eb.write_json(path, record)
            eb.mark(output, f"score-{arm}-seed{seed}", seconds, seconds if device_of(device) != "cpu" else 0)
            log(f"scored {arm} seed {seed} ({seconds:.0f}s)")
    return 0


# ---------------------------------------------------------------------------------------------------------------
# CLEVRER / Physion-Dominoes (EXPLORATORY, no tokens): action-free B1 on the #117 item-5 annotation carriers, as #123
# (rc.BenchData / bench_train / bench_score / bench_table)
# ---------------------------------------------------------------------------------------------------------------

BENCHES = ("clevrer", "physion")
BENCH_TIMES = (15, 30, 60, 120)
BENCH_CONTEXTS = (0, 1, 2, 3)
BENCH_HORIZON = max(BENCH_TIMES)  # training windows reach 120 frames ahead (CLEVRER clips hold 128 frames)
BENCH_QUEUES = 2


def bench_spec(plan_or_selected, bench):
    """B1 (native size, the selected stride / lr) at U's B1 budget (9 000 x 64, as #123's bench arms), rolled
    BENCH_HORIZON / stride steps from one context frame under the null launch."""
    selected = plan_or_selected if isinstance(plan_or_selected, str) else plan_or_selected["selected"]
    base = config_spec(selected, 13_919)
    return {**base, "arm": "B1-bench", "name": f"B1-bench-{bench}@{selected}", "epochs": None,
            "steps": MATCHED_RECIPE["steps"], "batch": MATCHED_RECIPE["batch"],
            "rollout_len": BENCH_HORIZON // base["stride"]}


def bench_windows(data, stride):
    """Every window (scene, start) whose frames start, start + stride, ..., start + 120 exist (the uniform
    sampler of #123's bench arms), packed with the null launch token: slots [W, T, 19, 13], vid_len [W] (full)."""
    offsets = torch.arange(0, BENCH_HORIZON + 1, stride)
    rows, units = [], []
    for i, scene in enumerate(data.keys):
        for start in range(int(data.ends[i]) - BENCH_HORIZON + 1):
            rows.append(data.carrier_rows([i] * len(offsets), (start + offsets).tolist()))
            units.append((scene, start))
    carriers = torch.stack(rows).float()
    slots = sf.pack_slots(carriers, torch.zeros(len(carriers), sf.LAUNCH_DIM))
    return {"slots": slots, "vid_len": torch.full((len(slots),), len(offsets), dtype=torch.long), "units": units}


def bench_dir(output, bench, seed):
    return Path(output) / "benchmarks" / bench / "B1-bench" / f"seed-{seed}"


def bench_jobs():
    return [(bench, seed) for bench in BENCHES for seed in eb.SEEDS]


def bench_train(output, queue, device=None):
    plan = load_plan(output)
    eb.install_stop_handlers()
    for i, (bench, seed) in enumerate(bench_jobs()):
        if i % BENCH_QUEUES != queue:
            continue
        spec = plan["benchmarks_exploratory"][bench]  # frozen bench_spec(selected, bench)
        out_dir = bench_dir(output, bench, seed)
        if (out_dir / "model.pt").is_file():
            continue
        data = bench_windows(eb.rc.BenchData(bench), spec["stride"])
        began = time.monotonic()
        try:
            report = train_run(out_dir, spec, seed, data, device)
        except eb.TrainingPaused as paused:
            seconds = time.monotonic() - began
            eb.mark(output, f"bench-train-{bench}-seed{seed}", seconds, seconds if device_of(device) != "cpu" else 0)
            log(str(paused))
            return eb.EXIT_PAUSED
        eb.mark(output, f"bench-train-{bench}-seed{seed}", time.monotonic() - began,
                (time.monotonic() - began) if device_of(device) != "cpu" else 0, complete=report["complete"])
    return 0


@torch.no_grad()
def bench_score(output, device=None):
    """Rollout from contexts 0-3 of every scene with the null launch; #79 field_errors (rib.unit_errors)
    position_mse at t in (15, 30, 60, 120) against the clip's carrier c + t (in-sample, as #79 / #84 / #117)."""
    from scripts import run_request_interference_benchmarks as rib
    load_plan(output)
    with harness.deterministic_scoring() as policy:
        for bench, seed in bench_jobs():
            out_dir = bench_dir(output, bench, seed)
            path = out_dir / "record.json"
            if path.is_file():
                continue
            model, payload = load_trained(out_dir / "model.pt", device)
            base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "benchmark": bench, "arm": "B1-bench",
                    "seed": seed}
            if model is None:
                eb.write_json(path, {**base, "typed_failure": payload["typed_failure"]})
                continue
            data = eb.rc.BenchData(bench)
            units = [(scene, c) for scene in data.scenes for c in BENCH_CONTEXTS]
            z0 = torch.stack([data.clips[s][c] for s, c in units])
            pred = rollout_carriers(model, z0, torch.zeros(len(units), sf.LAUNCH_DIM))
            errors = {}
            for u, (scene, c) in enumerate(units):
                errors.setdefault(scene, []).append(
                    {"context": c, "errors": {str(t): rib.unit_errors(pred[u, t // model.stride - 1],
                                                                      data.clips[scene][c + t]) for t in BENCH_TIMES}})
            eb.write_json(path, {**base, "typed_failure": None, "scoring_policy": policy, "scored_at": eb.utc_now(),
                                 "rollout": errors, "model_sha256": eb.sha256_of(out_dir / "model.pt")})
    return 0


def bench_table(output):
    """Mean position_mse over scenes x contexts x seeds per (benchmark, t), next to #117 item-5 J (LC, full, fold
    all) for F-15-continuous and F-5-continuous (rc.bench_table's rows). EXPLORATORY, no tokens."""
    from scripts import run_request_interference_benchmarks as rib

    def mean_errors(records, request=None):
        values = {str(t): [] for t in BENCH_TIMES}
        for record in records:
            if record.get("typed_failure"):
                continue
            rollout = record["rollout"] if request is None else record["rollout"][request]
            for units in rollout.values():
                for unit in units:
                    for t, err in unit["errors"].items():
                        if err is not None and err.get(rib.METRIC) is not None:
                            values[t].append(err[rib.METRIC])
        return {t: float(np.mean(v)) if v else None for t, v in values.items()}

    table = {}
    for bench in BENCHES:
        rows = {"B1-bench": {"position_mse": mean_errors(
            [eb.read_json(bench_dir(output, bench, seed) / "record.json") for seed in eb.SEEDS])}}
        for request in ("F-15-continuous", "F-5-continuous"):
            rows[f"J (#117 item 5, LC, full) {request}"] = {"position_mse": mean_errors(
                [eb.read_json(rib.record_path(rib.OUTPUT, bench, "LC", "J", "full", "all", seed)) for seed in eb.SEEDS],
                request)}
        table[bench] = rows
    return table


# ---------------------------------------------------------------------------------------------------------------
# --publish / --validate
# ---------------------------------------------------------------------------------------------------------------

RECORD_PATTERNS = ["records/*.json", "benchmarks/*/*/*/record.json"]


def load_records(output):
    return {arm: {seed: eb.read_json(record_path(output, arm, seed)) for seed in eb.SEEDS}
            for arm in (*ARMS, "U+head")}


def global_guards(output, plan, records):
    output = Path(output)
    manifest = eb.read_json(output / "readouts-manifest.json")
    anchors_read = min(r["anchors_read_at"] for recs in records.values() for r in recs.values())
    first_scored = min(r["scored_at"] for recs in records.values() for r in recs.values())
    chronology = bool(plan["frozen_at"] <= manifest["frozen_at"] <= anchors_read <= first_scored)
    policies = [r["scoring_policy"] for recs in records.values() for r in recs.values()]
    return {
        "G_isolation": {"pass": True, "rule": "fit split trains (dynamics, heads), policy split selects; evaluation "
                        "anchors are read only by --score after --freeze-readouts (G_chronology)"},
        "G_chronology": {"pass": chronology, "plan_frozen_at": plan["frozen_at"],
                         "readouts_frozen_at": manifest["frozen_at"], "anchors_read_at": anchors_read,
                         "first_scored_at": first_scored},
        "G_scoring_policy": {"pass": all(p == harness.SCORING_POLICY for p in policies)},
        "G_records": eb.records_guard(output, IDENTITY),
    }


def arm_compute(output, plan, arm):
    seconds, macs, walls, params = {}, [], [], None
    for seed in eb.SEEDS:
        report = model_dir(output, arm, seed, plan["selected"]) / "report.json"
        if report.is_file():
            seconds[str(seed)] = eb.read_json(report)["gpu_seconds"]
        rd = eb.read_json(readouts_dir(output, arm, seed) / "readouts.json")
        if rd.get("macs_per_candidate") is not None:
            macs.append(rd["macs_per_candidate"])
            params = rd["trainable_parameters"]
        rec = eb.read_json(record_path(output, arm, seed))
        if rec.get("wall_per_decision") is not None:
            walls.append(rec["wall_per_decision"])
    return {"trainable_parameters": params, "training_gpu_seconds": seconds,
            "macs_per_candidate": float(np.mean(macs)) if macs else None,
            "wall_per_decision": float(np.median(walls)) if walls else None}


def publication(output):
    """Every published table, derived from the frozen plan, readouts and records (deterministic)."""
    output = Path(output)
    plan = load_plan(output)
    records = load_records(output)
    cells = eb.evaluation_cells()
    refs = eb.references()
    guards_global = global_guards(output, plan, records)
    global_pass = all(g["pass"] is True for g in guards_global.values())
    gport = eb.read_json(output / "gport/gport.json")
    gfid = eb.read_json(output / "fidelity/gfid.json")
    extra_guards = {"G_port": bool(gport["G_port"]), "G_fid": bool(gfid.get("G_fid"))}
    arm_views = {"B1": ("B1", eb.PRIMARY_COST), "B1-m": ("B1-m", eb.PRIMARY_COST)}
    views = {name: (records[arm], key) for name, (arm, key) in arm_views.items()}
    views["B1-head"] = (records["B1"], HEAD_KEY)
    views["U+head"] = (records["U+head"], HEAD_KEY)
    blocks, guards_of, tables = {}, {}, {}
    for arm in ARMS:
        per_seed = {}
        for seed in eb.SEEDS:
            rd = eb.read_json(readouts_dir(output, arm, seed) / "readouts.json")
            per_seed[seed] = {"G_rank": rd.get("G_rank"), "G_action": rd.get("G_action"), "G_collapse": None,
                              "policy_ceiling_auc": None, "typed_failure": rd.get("typed_failure")
                              or records[arm][seed].get("typed_failure")}
        guards_of[arm] = eb.arm_guard_block(per_seed, ["G_rank", "G_action", "G_nonfinite", "G_trained"],
                                            nonfinite=eb.nonfinite_share(records[arm].values(), eb.PRIMARY_COST))
    for inventory in eb.INVENTORIES:
        rows = eb.view_rows(cells, inventory, views, refs)
        tables[inventory] = {"rankers": eb.ranker_table(rows, ["U", "C*", "P", "J", "J_dec", *views])}
        if inventory == GRID:
            for arm in ARMS:
                blocks[arm] = eb.arm_block(rows, arm, records[arm], eb.PRIMARY_COST, guards_of[arm], global_pass,
                                           arm_compute(output, plan, arm), ARMS[arm]["role"], ARMS[arm]["program_set"],
                                           extra_guards=extra_guards)
            b1_ok = global_pass and blocks["B1"]["guards"]["pass"]
            within = {"B1 - B1-m": eb.within_family(rows, "B1", "B1-m", b1_ok and blocks["B1-m"]["guards"]["pass"]),
                      "B1-head - B1": eb.within_family(rows, "B1-head", "B1", b1_ok),
                      "U+head - U": eb.within_family(rows, "U+head", "U", global_pass),
                      "B1-head - U+head": eb.within_family(rows, "B1-head", "U+head", b1_ok)}
        else:
            tables[inventory]["contrasts"] = {
                f"{arm} - U": eb.contrast(rows, arm, "U", global_pass and blocks[arm]["guards"]["pass"])
                for arm in ARMS}
    wrong = {arm: eb.wrong_anchor_table(cells, records[arm], arm, eb.PRIMARY_COST,
                                        global_pass and blocks[arm]["guards"]["pass"]) for arm in ARMS}
    curves = eb.reference_position_curves()
    positions = eb.position_table({"B1": records["B1"], "B1-m": records["B1-m"], "U": curves["U"],
                                   "C*": curves["C*"]}, cells)
    gfid_block = {"B1": {"pass": bool(gfid.get("G_fid")), "tier_e": gfid["tiers"]["tier-e"]["ours"],
                         "band": gfid["tiers"]["tier-e"]["band"]}}
    ledger, gpu_total, wall_total = eb.ledger_totals(output, training_report_of(output, plan))
    summary = eb.child_summary(ISSUE, IDENTITY, plan["frozen_at"], blocks, within, guards_global, gfid_block, {
        "inventories": tables, "wrong_anchor": wrong, "positions": positions, "selection": plan["selected"],
        "benchmarks_exploratory": bench_table(output),
        "readouts": {f"{a}--seed{s}": {k: v for k, v in eb.read_json(readouts_dir(output, a, s) / "readouts.json").items()
                                       if k in ("policy_auc", "policy_head_auc", "G_action", "typed_failure")}
                     for a, s in readout_items()}})
    compute = {"schema": "issue_129_compute_v1", "plan_identity": IDENTITY, "ledger": ledger,
               "gpu_seconds_total": gpu_total, "wall_seconds_total": wall_total,
               "arms": {arm: blocks[arm]["compute"] for arm in ARMS}, "reference": eb.reference_compute()}
    return summary, compute


def findings_text(summary):
    lines = [f"# #{ISSUE} B1 SlotFormer-E99: findings", "",
             f"Global guards pass: {summary['global_guards_pass']}; selected configuration "
             f"{summary['extra']['selection']}.", "", *eb.format_arm_rows(summary["arms"]), "", "## Within family", ""]
    for label, block in summary["within_family"].items():
        lines.append(f"- {label}: {eb.rc.fmt_block(block['estimate'])} ({block['token']} / {block['reverse_token']})")
    lines += ["", "## Wrong anchor (right - wrong, grid)", ""]
    for arm, block in summary["extra"]["wrong_anchor"].items():
        lines.append(f"- {arm}: {eb.rc.fmt_block(block['right_minus_wrong'])} ({block['token']})")
    lines += ["", "## CLEVRER / Physion-Dominoes (EXPLORATORY; no ranking, no tokens)", "",
              "Action-free B1 (selected configuration, 9 000 x 64, null launch) on the #117 item-5 annotation carriers; "
              "#79 position_mse from contexts 0-3, mean over scenes x contexts x seeds.", ""]
    for bench, rows in summary["extra"]["benchmarks_exploratory"].items():
        lines += [f"**{bench}**", "", "| row | " + " | ".join(f"t = {t}" for t in BENCH_TIMES) + " |",
                  "| --- |" + " --- |" * len(BENCH_TIMES)]
        for name, row in rows.items():
            cells = [("n/a" if row["position_mse"][str(t)] is None else f"{row['position_mse'][str(t)]:.4g}")
                     for t in BENCH_TIMES]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["", "Intervals are descriptive (member-clustered bootstrap); tokens follow the #127 delta rule.", ""]
    return "\n".join(lines)


def publish(output):
    output = Path(output)
    eb.write_json(output / "records-manifest.json", eb.records_manifest(output, IDENTITY, RECORD_PATTERNS))
    summary, compute = publication(output)
    eb.write_json(output / "compute.json", compute)
    eb.write_json(output / "summary.json", summary)
    (output / "findings.md").write_text(findings_text(summary))
    log(f"published; G-records {summary['global_guards']['G_records']['pass']}")
    return 0


def validate(output, device=None):
    """Hash-check the records against the manifest, re-score one record (exact row equality), write the receipt,
    then re-derive the publication twice and require byte-identical files."""
    output = Path(output)
    plan = load_plan(output)
    load_manifest(output)
    manifest = eb.read_json(output / "records-manifest.json")
    error, spot = None, None
    if manifest != eb.records_manifest(output, IDENTITY, manifest["patterns"]):
        error = "records differ from records-manifest.json"
    else:
        arm, seed = "B1", eb.SELECTION_SEED
        stored = eb.read_json(record_path(output, arm, seed))
        anchors = torch.load(eb.evaluation_anchors_path(output), map_location="cpu", weights_only=True)
        with harness.deterministic_scoring():
            fresh = score_arm(output, plan, arm, seed, anchors, eb.evaluation_cells(), device)
        fresh = json.loads(eb.json_text(fresh))
        same = all(fresh["inventories"][inv] == stored["inventories"][inv] for inv in eb.INVENTORIES)
        spot = {"record": f"{arm}--seed{seed}", "pass": bool(same), "rule": "exact row equality, every inventory"}
    eb.write_receipt(output, IDENTITY, spot, error)
    publish(output)
    first = {name: (output / name).read_bytes() for name in ("summary.json", "compute.json", "findings.md")}
    publish(output)
    second = {name: (output / name).read_bytes() for name in first}
    stable = first == second
    receipt = eb.read_json(output / eb.VALIDATION_RECEIPT)
    ok = bool(receipt["pass"] and stable)
    log(f"validate: receipt {receipt['pass']}, publication stable {stable}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("selection", "selection-readouts", "smoke", "prepare", "train", "readouts", "freeze-readouts",
                 "score", "bench-train", "bench-score", "publish", "validate"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="queue index (the job's CUDA_VISIBLE_DEVICES picks the GPU)")
    parser.add_argument("--workers", type=int, default=0, help="accepted for the contract; data live on the device")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    output = Path(args.output)
    if args.selection:
        return selection(output, args.gpu, args.device)
    if args.selection_readouts:
        return selection_readouts(output, args.device)
    if args.smoke:
        return smoke(output, args.device)
    if args.prepare:
        return prepare(output)
    if args.train:
        return train(output, args.gpu, args.device)
    if args.readouts:
        return readouts(output, args.gpu, args.device)
    if args.freeze_readouts:
        return freeze_readouts(output)
    if args.score:
        return score(output, args.device)
    if args.bench_train:
        return bench_train(output, args.gpu, args.device)
    if args.bench_score:
        return bench_score(output, args.device)
    if args.publish:
        return publish(output)
    return validate(output, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
