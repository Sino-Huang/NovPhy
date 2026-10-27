"""Issue #136 (#127 coverage arm T4, rule P-coverage; Tier-C if D1 - U or D1 - J_dec is supported): Diffusion Forcing
on E99 carrier sequences against U on the sealed NovPhy launch-ranking grid.

Arm (Decisions P-route / P-adapter on #136, see DECISIONS):
- **T4 DF-car** (coverage, secondary; never in the program claim): the official Diffusion Forcing low-dimensional
  transition model (``DiffusionTransitionModel`` / ``TransitionMlp``, vendored from buoyancy99/diffusion-forcing
  ``paper`` @ dd65516; adapter ``world_model/baselines/diffusion_forcing/novphy.py``) trained with independent per-step
  noise levels on fit-shot carrier sequences at frame stride 5 (46 frames 0..225; the launch conditions the
  transition out of t0, no-ops after), sampled with the causal-uncertainty pyramid over the 45-step horizon from the
  anchor carrier (the context frame). Score = expected tie-free ``EndpointCosts`` cost of the frame-225 carrier over
  S = 32 sampled paths (common random numbers across candidates); a candidate with a nonfinite sample is a retained
  typed row. Configuration selected on the policy split among three pre-declared ones (``dn.CONFIGURATIONS``:
  ts-native = the time-series recipe with its early stopping, ts-100k = the same recipe for a fixed 100 000 updates,
  planning = the maze-planning recipe; seed 20260908, full native budget); the selected run is reused as main seed
  20260908.
- No native success head; no U + head control; no matched-budget arm; G-collapse is not in the ticket's guard list
  (scores are computed in carrier space); position error is descriptive (sample-mean carriers at 15/30/60/150/225).

Gates (#136): ``--selection`` and ``--train`` refuse without the orchestrator's ``funded.json`` (Decision P-coverage
funding), a passing ``gport/gport.json`` and a passing ``fidelity/gfid.json`` (validation ladder: Table 2 Exchange
first); ``--score`` refuses unless G-fid passed. ``--timing`` (a GPU probe, <= 1 GPU-h) needs no funded.json but its own
posted decision before it is enqueued. Without plan.json, ``--publish`` writes the not-run child summary from the
orchestrator's ``not-run.json`` (``{"typed_reason": "cap" | "fidelity_not_reproduced", "decision": url}``).

Modes (round3/child-runner-contract.md): --project (CPU, pre-funding GPU-h projection), --timing, --selection,
--selection-readouts, --smoke, --prepare, --train, --readouts, --freeze-readouts, --score, --score-novelty
(P-novelty, descriptive), --publish, --validate. GPU work only through the night-window scheduler
(scripts/gpu_window_issue_136*.json, prepared, not enqueued before funding); tests use CUDA_VISIBLE_DEVICES="".
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
from world_model.baselines.diffusion_forcing import novphy as dn  # noqa: E402
from world_model.planning import scoring_harness as harness  # noqa: E402

ISSUE = 136
IDENTITY = "issue-136-diffusion-forcing-v1"
OUTPUT = ROOT / ".local-artifacts" / IDENTITY
RUNNER = "scripts/run_issue_136_diffusion_forcing.py"
TESTS = "tests/test_issue_136_diffusion_forcing.py"
SCHEMA_MODEL = "issue_136_model_v1"
SCHEMA_PLAN = "issue_136_plan_v1"
SCHEMA_RECORD = "issue_136_record_v1"
SCHEMA_SMOKE = "issue_136_smoke_v1"
SCHEMA_SELECTION = "issue_136_selection_v1"
SCHEMA_READOUTS_MANIFEST = "issue_136_readouts_manifest_v1"
SCHEMA_PROJECTION = "issue_136_projection_v1"
SCHEMA_TIMING = "issue_136_timing_v1"
DECISIONS = {
    "coverage": "https://github.com/Sino-Huang/NovPhy/issues/127#issuecomment-6412711831",
    "license": "https://github.com/Sino-Huang/NovPhy/issues/136#issuecomment-6412925640",
    "route": "https://github.com/Sino-Huang/NovPhy/issues/136#issuecomment-6412971124",
    "band": "https://github.com/Sino-Huang/NovPhy/issues/136#issuecomment-6413164397",
    "adapter": "https://github.com/Sino-Huang/NovPhy/issues/136#issuecomment-6413489211",
    "band_addendum": "https://github.com/Sino-Huang/NovPhy/issues/136#issuecomment-6413855182",
    "gfid_result": "https://github.com/Sino-Huang/NovPhy/issues/136#issuecomment-6436060220",
}

CONFIGS = dict(dn.CONFIGURATIONS)  # ts-native / ts-100k / planning (pre-declared)
SELECTION_QUEUES = {0: ["ts-native"], 1: ["ts-100k"], 2: ["planning"]}  # one configuration per scheduler job
FIT_SHOTS = 13_919  # eb.FitShots() (13 920 minus one typed-failure branch); checked by --smoke
ARMS = {"T4": {"size": "native", "role": "coverage (secondary, P-coverage / Tier-C; never in the program claim)",
               "program_set": False}}
ARM = "T4"
POSITION_INDEX = [t // dn.STRIDE for t in eb.POSITION_TIMES]  # frames 15, 30, 60, 150, 225 -> path indices
CHECKPOINT_EVERY = 2000
LOG_EVERY = 100
ROWS_PER_CALL = 128  # candidate rows per sampler call (x S samples); fixed order -> reproducible records
CHILD_CAP_GPU_HOURS = 184.0  # the upper end of the #127 coverage estimate for Diffusion Forcing (P-coverage funds)
SEED_CAP_GPU_HOURS = 24.0  # #127: native cap per seed
WALL_CAP_HOURS = 48.0  # on 4 GPUs
GRID = "grid"
ARM_CHECKS = ["G_rank", "G_action", "G_nonfinite", "G_trained"]
TRAINING_PRECISION = {"dtype": "16-mixed on CUDA (fp16 autocast + GradScaler; upstream precision 16-mixed), fp32 on "
                               "CPU", "flags": "torch.set_float32_matmul_precision('high') as upstream main.py"}


def log(message):
    eb.log(ISSUE, message)


def device_of(requested=None):
    return requested or ("cuda" if torch.cuda.is_available() else "cpu")


def rel(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


# ---------------------------------------------------------------------------------------------------------------
# configurations and paths
# ---------------------------------------------------------------------------------------------------------------

def config_spec(name):
    config = dn.config_of(name)
    return {"name": name, "arm": ARM, "size": "native", "configuration": name, "config": config,
            "steps": config["steps"], "batch": config["batch_size"], "early_stopping": config["early_stopping"],
            "samples": dn.SAMPLES, "sample_seed": dn.SAMPLE_SEED}


def arm_spec(arm, selected):
    if arm != ARM:
        raise KeyError(arm)
    return config_spec(selected)


def selection_dir(output, name, seed=eb.SELECTION_SEED):
    return Path(output) / "selection" / name / f"seed-{seed}"


def model_dir(output, arm, seed, selected=None):
    """T4 seed 20260908 is the reused selection run of the selected configuration (#127 selection rule step 4)."""
    if arm == ARM and seed == eb.SELECTION_SEED:
        if selected is None:
            raise ValueError("model_dir(T4, 20260908) needs the selected configuration")
        return selection_dir(output, selected)
    return Path(output) / "models" / arm / f"seed-{seed}"


def readouts_dir(output, arm, seed):
    return Path(output) / "readouts" / arm / f"seed-{seed}"


def record_path(output, arm, seed):
    return Path(output) / "records" / f"{arm}--seed{seed}.json"


# ---------------------------------------------------------------------------------------------------------------
# fit sequences
# ---------------------------------------------------------------------------------------------------------------

def fit_store(shots, chunk=1024):
    """Every fit shot's sequence: carriers [N, 46, 236] float32 (frames 0, 5, ..., 225; positions past the end repeat
    the final carrier), launch5 [N, 5] and the fit-split normalisation (mean / std [236]), frozen shot order (CPU)."""
    carriers = []
    for start in range(0, len(shots.keys), chunk):
        index = range(start, min(len(shots.keys), start + chunk))
        pairs = [(s, p) for s in index for p in dn.episode_positions(shots.ends[s])]
        rows = shots.carrier_rows([s for s, _ in pairs], [p for _, p in pairs])
        carriers.append(torch.as_tensor(np.asarray(rows), dtype=torch.float32).reshape(
            len(index), dn.FRAMES, dn.CARRIER_DIM))
    carriers = torch.cat(carriers)
    mean, std = dn.normalisation(carriers)
    return {"carriers": carriers, "launch5": torch.as_tensor(shots.launch5, dtype=torch.float32), "mean": mean,
            "std": std}


def training_batch(data, step, seed, spec, device="cpu"):
    """Batch ``step`` of a run: fit-shot sequences drawn from eb.batch_rng(seed, step) (uniform with replacement)."""
    rows = dn.batch_rows(eb.batch_rng(seed, step), len(data["launch5"]), spec["batch"])
    return dn.sequence_batch(data["carriers"], data["launch5"], rows, device)


# ---------------------------------------------------------------------------------------------------------------
# training (resumable; nonfinite loss -> typed terminal failure)
# ---------------------------------------------------------------------------------------------------------------

def train_run(out_dir, spec, seed, data, device=None, max_steps=None):
    """Train one model (resumable from out_dir/resume.pt; the official update via dn.train_loop). Batch b =
    training_batch(data, b, seed, spec); the noise levels / noise draw from the global torch generators seeded with
    ``seed`` (checkpointed). ts-native: early stopping on the fixed validation subset, best checkpoint kept."""
    device = device_of(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "model.pt"
    if final.is_file():
        log(f"{spec['name']} seed {seed}: model.pt exists; skipped")
        return eb.read_json(out_dir / "report.json")
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")  # upstream main.py
    steps, early = spec["steps"], spec["early_stopping"]
    torch.manual_seed(seed)
    trainer = dn.Trainer(spec["config"], data["mean"], data["std"], device)
    validate = None
    if early is not None:
        rows = dn.validation_rows(len(data["launch5"]), early["validation_shots"])
        validate = lambda: dn.validation_crps(trainer.model, data["carriers"], data["launch5"], rows,  # noqa: E731
                                              early["samples"], autocast=trainer.mixed)
    try:
        out = dn.train_loop(trainer, out_dir, steps, lambda b: training_batch(data, b, seed, spec, device),
                            lambda: bool(eb.STOP_REQUESTED), validate=validate, early=early, max_steps=max_steps,
                            log_every=LOG_EVERY, checkpoint_every=CHECKPOINT_EVERY)
    except dn.TrainingPaused as paused:
        log(f"{spec['name']} seed {seed}: stop signal; {paused}")
        raise eb.TrainingPaused(f"{spec['name']} seed {seed} {paused}") from None
    if out["failure"] is None and not out["finished"]:  # max_steps reached (tests, throughput)
        return {"partial": True, "steps_done": out["steps_done"], "seconds": out["seconds"]}
    failure = out["failure"]
    if failure is None and early is not None:
        if out["early"]["best_step"] is None:
            failure = {"typed_failure": "nonfinite_validation", "step": out["stop"]}
        else:  # ModelCheckpoint(save_top_k=1): the best validation checkpoint
            trainer.model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device,
                                                     weights_only=False)["model"])
    complete = failure is None
    precision = {**TRAINING_PRECISION, "device": device, "mixed": trainer.mixed,
                 "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                 "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    model = trainer.model
    payload = {"schema": SCHEMA_MODEL, "spec": spec, "seed": seed, "complete": complete, "typed_failure": failure,
               "state_dict": {k: v.cpu() for k, v in model.state_dict().items()} if complete else None,
               "mean": data["mean"], "std": data["std"], "parameters": dn.parameter_count(model),
               "precision": precision, "steps_run": out["stop"], "early_stopping": out["early"] if early else None}
    eb.atomic_torch(final, payload)
    seconds = out["elapsed"]
    report = {"spec": spec, "seed": seed, "steps": steps, "steps_run": out["stop"], "complete": complete,
              "typed_failure": failure, "wall_seconds": seconds,
              "gpu_seconds": seconds if device.startswith("cuda") else 0.0, "device": device,
              "seconds_per_step": seconds / max(1, out["stop"]),
              "early_stopping": ({k: out["early"][k] for k in ("best", "best_step", "stopped_early")}
                                 | {"validations": len(out["early"]["history"])}) if early else None,
              "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None,
              "model_sha256": eb.sha256_of(final), "parameters": payload["parameters"],
              "final": out["trace"][-1] if out["trace"] else None, "precision": precision, "finished_at": eb.utc_now()}
    eb.write_json(out_dir / "report.json", report)
    if (out_dir / "resume.pt").is_file() and complete:
        (out_dir / "resume.pt").unlink()
    log(f"{spec['name']} seed {seed}: {'complete' if complete else failure['typed_failure']} after "
        f"{out['stop']} updates, {seconds:.0f}s")
    return report


def load_trained(path, device=None):
    """(model, payload) of a model.pt, or (None, payload) for a typed failure."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA_MODEL:
        raise ValueError(f"{path}: schema differs")
    if not payload["complete"]:
        return None, payload
    model = dn.build_model(payload["spec"]["config"], payload["mean"], payload["std"], device_of(device))
    model.load_state_dict(payload["state_dict"])
    return model.eval(), payload


def run_queue(output, items, device=None):
    """Train (spec, seed, out_dir) items in order; a stop signal pauses the queue (exit 75)."""
    eb.install_stop_handlers()
    data = None
    for spec, seed, out_dir in items:
        if (Path(out_dir) / "model.pt").is_file():
            continue
        data = data or fit_store(eb.FitShots())
        began = time.monotonic()
        try:
            report = train_run(out_dir, spec, seed, data, device)
        except eb.TrainingPaused as paused:
            seconds = time.monotonic() - began
            eb.mark(output, f"train-{spec['name']}-seed{seed}", seconds, seconds if device_of(device) != "cpu" else 0)
            log(str(paused))
            return eb.EXIT_PAUSED
        seconds = time.monotonic() - began
        eb.mark(output, f"train-{spec['name']}-seed{seed}", seconds, seconds if device_of(device) != "cpu" else 0,
                report=rel(Path(out_dir, "report.json")), complete=report["complete"])
    return 0


# ---------------------------------------------------------------------------------------------------------------
# prediction (sampled paths), expected costs, policy AUC
# ---------------------------------------------------------------------------------------------------------------

def objective_costs():
    from scripts import run_lookahead_control as wlc
    return harness.EndpointCosts(wlc.load_objective())


def grid_actions():
    items = eb.grid_items()
    return items, torch.stack([eb.launch5(item["action"]) for item in items])


@torch.no_grad()
def row_samples(model, z0, launch5, keep=(dn.HORIZON,), rows_per_call=ROWS_PER_CALL):
    """Sampled paths of candidate rows (anchors z0 [R, 236], launches [R, 5]) in fixed chunks of ``rows_per_call``;
    {path index: [R, S, 236] CPU} (common random numbers: every row sees the same S noise sequences)."""
    noise = dn.CommonNoise(dn.SAMPLES)
    out = {k: [] for k in keep}
    for i in range(0, len(z0), rows_per_call):
        paths = dn.candidate_paths(model, z0[i:i + rows_per_call], launch5[i:i + rows_per_call], dn.SAMPLES, noise,
                                   keep)
        for k in keep:
            out[k].append(paths[k])
    return {k: torch.cat(v) for k, v in out.items()}


@torch.no_grad()
def levels_rows(model, levels, costs):
    """Rows of several levels [(items, z0 [236])] (the primary scoring path, batched over levels in order; shared by
    policy AUCs, G-rank, --score and --score-novelty). Returns (rows per level, frame-225 samples per level)."""
    z0 = torch.cat([z[None].expand(len(items), -1) for items, z in levels]) if levels else torch.empty(0, 236)
    acts = torch.cat([torch.stack([eb.launch5(it["action"]) for it in items]) for items, _ in levels])
    ends = row_samples(model, z0, acts)[dn.HORIZON]
    out, start = [], 0
    for items, z in levels:
        samples = ends[start:start + len(items)]
        out.append((dn.expected_rows(items, samples, z, costs), samples))
        start += len(items)
    return out


@torch.no_grad()
def level_rows(model, items, z0, costs):
    """Rows of one level's candidates from its carrier anchor (levels_rows of one level)."""
    return levels_rows(model, [(items, z0)], costs)[0][0]


@torch.no_grad()
def fit_anchor_readouts(model, costs):
    """480 fit anchors x 16 grid launches: the tie-free expected-cost matrix [L, 16] (excluded -> nan) and the
    per-candidate sample-mean frame-225 carriers [L, 16, 236] (G-action)."""
    anchors = eb.fit_anchors()["carriers"]
    members = sorted(anchors)
    items, _ = grid_actions()
    out = levels_rows(model, [(items, anchors[m]) for m in members], costs)
    matrix = np.full((len(members), len(items)), np.nan)
    means = torch.empty(len(members), len(items), dn.CARRIER_DIM)
    typed = 0
    for i, (rows, samples) in enumerate(out):
        for j, row in enumerate(rows):
            if row["excluded"]:
                typed += 1
            else:
                matrix[i, j] = row[eb.PRIMARY_COST]
        means[i] = dn.sample_means(samples)
    return members, matrix, means, typed


@torch.no_grad()
def policy_rows(model, costs):
    """{member: rows} of the policy grid mixed cells."""
    anchors = eb.policy_anchors()["carriers"]
    cells = sorted(eb.policy_cells(GRID).items())
    out = levels_rows(model, [(entry["inventory"], anchors[m]) for m, entry in cells], costs)
    return {m: rows for (m, _), (rows, _) in zip(cells, out)}


# ---------------------------------------------------------------------------------------------------------------
# gates (#136)
# ---------------------------------------------------------------------------------------------------------------

def funded(output):
    """The orchestrator's funding marker (Decision P-coverage funding): present, a JSON object, not ``funded: false``."""
    path = Path(output) / "funded.json"
    if not path.is_file():
        return False
    try:
        marker = eb.read_json(path)
    except ValueError:
        return False
    return isinstance(marker, dict) and marker.get("funded", True) is True


def gport_pass(output):
    path = Path(output) / "gport/gport.json"
    return bool(path.is_file() and eb.read_json(path).get("pass") is True)


def gfid_pass(output):
    path = Path(output) / "fidelity/gfid.json"
    return bool(path.is_file() and eb.read_json(path).get("pass") is True)


def readiness(output):
    """Funding marker, G-port pass and G-fid pass (validation ladder: Table 2 Exchange first)."""
    out = {"funded.json": funded(output), "gport/gport.json": gport_pass(output),
           "fidelity/gfid.json": gfid_pass(output)}
    out["pass"] = all(out.values())
    return out


# ---------------------------------------------------------------------------------------------------------------
# --selection / --selection-readouts
# ---------------------------------------------------------------------------------------------------------------

def selection(output, queue, device=None):
    gate = readiness(output)
    if not gate["pass"]:
        log(f"selection refused: readiness gate {gate}; nothing trained")
        return 1
    names = SELECTION_QUEUES[queue]
    items = [(config_spec(name), eb.SELECTION_SEED, selection_dir(output, name)) for name in names]
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
            rows = policy_rows(model, costs)
            results[name] = {"auc": eb.policy_grid_auc(rows, eb.PRIMARY_COST), "parameters": payload["parameters"]}
            details[name] = {"typed_failure": None, "model_sha256": eb.sha256_of(path), "cells": len(rows),
                             "steps_run": payload["steps_run"],
                             "typed_rows": sum(r["excluded"] for rs in rows.values() for r in rs)}
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
# --project (CPU, before funding), --timing (GPU probe) and --smoke
# ---------------------------------------------------------------------------------------------------------------

def lpt(items, bins=4):
    """Longest-processing-time assignment: items [(name, seconds)] -> {bin: [names]}, loads."""
    queues, loads = {b: [] for b in range(bins)}, [0.0] * bins
    for name, seconds in sorted(items, key=lambda t: (-t[1], t[0])):
        b = int(np.argmin(loads))
        queues[b].append(name)
        loads[b] += seconds
    return queues, loads


def remaining_items(output, selected):
    """(arm, seed, spec, out_dir) of every training run the plan still needs."""
    return [(ARM, seed, arm_spec(ARM, selected), model_dir(output, ARM, seed)) for seed in eb.SEEDS
            if seed != eb.SELECTION_SEED]


# decision counts (levels scored with 16 candidates x S samples); fit / policy counts are exact, evaluation /
# novelty counts are metadata: the #129 publication's mixed-level counts (grid 311, offset 76, angle 78, power 47)
# and 29 novelty cells x 8 levels x 2 sides (round-3 NoveltyScout) as an upper bound [INFERENCE]
FIT_ANCHORS = 480
EVALUATION_LEVELS = 311 + 76 + 78 + 47
POSITION_LEVELS = 311  # one path set per complete grid candidate of the grid mixed cells (~311 x 16 rows)
NOVELTY_LEVELS = 29 * 8 * 2


def decision_counts(policy_cells):
    """Levels scored per stage: readouts per arm-seed (fit anchors + policy cells), selection readouts (3 x policy
    cells), scoring per seed (evaluation levels + wrong anchors + positions + novelty)."""
    return {"readouts_per_arm_seed": FIT_ANCHORS + policy_cells, "selection_readouts": len(CONFIGS) * policy_cells,
            "score_per_seed": EVALUATION_LEVELS + FIT_ANCHORS + POSITION_LEVELS + NOVELTY_LEVELS}


def child_projection(rates, fidelity_gpu_h, seconds_per_decision, policy_cells, validation_seconds):
    """GPU-h of everything left. ``rates`` {configuration: GPU s/update}; ``validation_seconds`` = one ts-native
    early-stopping validation pass (every 100 updates; upper bound: no early stop before the 200 000-update cap);
    ``seconds_per_decision`` = one level (16 candidates x S paths); ``fidelity_gpu_h`` = the Exchange G-fid run."""
    run_s = {}
    for name in CONFIGS:
        spec = config_spec(name)
        validations = spec["steps"] // spec["early_stopping"]["epoch_batches"] if spec["early_stopping"] else 0
        run_s[name] = spec["steps"] * rates[name] + validations * validation_seconds
    counts = decision_counts(policy_cells)
    per_seed = max(run_s.values())  # the selected configuration is unknown: the most expensive one
    training = [(f"{ARM}--seed{seed}", per_seed) for seed in eb.SEEDS if seed != eb.SELECTION_SEED]
    selro_s = counts["selection_readouts"] * seconds_per_decision
    readout_s = counts["readouts_per_arm_seed"] * len(eb.SEEDS) * seconds_per_decision
    score_s = counts["score_per_seed"] * len(eb.SEEDS) * seconds_per_decision
    queues, loads = lpt(training)
    fid_s = fidelity_gpu_h * 3600
    total = fid_s + sum(run_s.values()) + sum(s for _, s in training) + selro_s + readout_s + score_s
    wall = fid_s + max(run_s.values()) + selro_s + max(loads) + readout_s / 4 + score_s
    out = {"steps": {name: config_spec(name)["steps"] for name in CONFIGS},
           "gpu_seconds_per_update": rates, "validation_seconds": validation_seconds,
           "seconds_per_decision": seconds_per_decision, "decision_counts": counts, "policy_cells": policy_cells,
           "fidelity_gpu_h": fidelity_gpu_h, "selection_gpu_h": {k: v / 3600 for k, v in run_s.items()},
           "selection_readouts_gpu_h": selro_s / 3600, "training_gpu_h": sum(s for _, s in training) / 3600,
           "readouts_gpu_h": readout_s / 3600, "score_gpu_h": score_s / 3600, "total_gpu_h": total / 3600,
           "child_cap_gpu_h": CHILD_CAP_GPU_HOURS, "native_per_seed_gpu_h": per_seed / 3600,
           "seed_cap_gpu_h": SEED_CAP_GPU_HOURS, "wall_h_on_4_gpus_serial_ladder": wall / 3600,
           "wall_cap_h": WALL_CAP_HOURS, "training_queues": queues,
           "training_queue_loads_h": [x / 3600 for x in loads]}
    out["within_caps"] = bool(total / 3600 <= CHILD_CAP_GPU_HOURS and per_seed / 3600 <= SEED_CAP_GPU_HOURS
                              and wall / 3600 <= WALL_CAP_HOURS)
    return out


# the Exchange G-fid run (official pipeline): 60-frame batches of 32 sequences (8 series, 20 covariates); a validation
# (and the test) samples 5 windows x 100 paths x 100 (df_prediction.validation_step repeats the batch calc_crps_sum
# times and df_base.validation_step repeats it again: 50 000 rows) through 30 context + 30 x 51 AR DDIM calls
GFID = {"x_dim": 8, "cond_dim": 20, "frames": 60, "batch": 32, "rows": 50_000, "calls": 30 + 30 * 51,
        "max_updates": 200_000, "validations": 2001}


def fidelity_seconds(update_seconds, validation_seconds):
    """The Exchange G-fid upper bound: no early stop before the 200 000-update cap (2000 validations + the test)."""
    return GFID["max_updates"] * update_seconds + GFID["validations"] * validation_seconds


_VIEW_OPS = {"view", "_unsafe_view", "t", "expand", "permute", "reshape", "transpose", "unsqueeze", "squeeze",
             "slice", "select", "alias", "detach", "as_strided", "split", "chunk", "unbind", "lift_fresh",
             "_to_copy", "empty", "empty_like", "zeros", "zeros_like", "ones_like", "full", "scalar_tensor",
             "split_with_sizes", "is_same_size"}


def op_and_flop_count(fn):
    """(non-view aten ops dispatched, FLOPs) of ``fn()`` on CPU: the kernel-launch and compute proxies of the GPU
    projection model [INFERENCE]."""
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils.flop_counter import FlopCounterMode

    class Count(TorchDispatchMode):
        ops = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func.overloadpacket.__name__ not in _VIEW_OPS:
                Count.ops += 1
            return func(*args, **(kwargs or {}))

    flops = FlopCounterMode(display=False)
    with flops, Count():
        fn()
    return Count.ops, flops.get_total_flops()


def _synthetic_data(shots=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    carriers = torch.rand(shots, dn.FRAMES, dn.CARRIER_DIM, generator=g)
    carriers[:, :, 118:] = 0.0
    mean, std = dn.normalisation(carriers)
    return {"carriers": carriers, "launch5": torch.tensor([[-0.2, 0.1, 1.0, 0.0, 1.0]]).repeat(shots, 1),
            "mean": mean, "std": std}


def cpu_measurements(repeats=2):
    """CPU (4 threads) s/update at each configuration's native batch, op / FLOP counts of one update and of one sampler
    call (S x 16 rows per level; ROWS_PER_CALL x S rows per chunk), synthetic carrier-shaped batches."""
    data = _synthetic_data()
    out = {}
    for name in ("ts-native", "planning"):
        spec = config_spec(name)
        torch.manual_seed(0)
        trainer = dn.Trainer(spec["config"], data["mean"], data["std"], "cpu")
        batch = training_batch(data, 0, 0, spec)
        trainer.train_step(*batch)
        began = time.monotonic()
        for step in range(1, repeats + 1):
            trainer.train_step(*training_batch(data, step, 0, spec))
        seconds = (time.monotonic() - began) / repeats
        small = dn.sequence_batch(data["carriers"], data["launch5"], np.arange(8))
        ops, _ = op_and_flop_count(lambda: dn.training_loss(trainer.model, *small)[0].backward())
        _, flops = op_and_flop_count(lambda: dn.training_loss(trainer.model, *batch)[0].backward())
        model, tm = trainer.model.eval(), trainer.model.transition_model
        pairs = dn.ddim_times(tm)

        def call(rows):
            x = torch.randn(rows, dn.CARRIER_DIM)
            z = torch.zeros(rows, spec["config"]["z_dim"])
            with torch.no_grad():
                dn.ddim_step(tm, x, z, torch.zeros(rows, dn.ACT_DIM), 10, dn.GlobalNoise(), pairs)
        call_ops, _ = op_and_flop_count(lambda: call(8))
        _, call_flops_row = op_and_flop_count(lambda: call(dn.SAMPLES))
        out[name] = {"cpu_seconds_per_update": seconds, "batch": spec["batch"], "update_ops": ops,
                     "update_flops": flops, "parameters": dn.parameter_count(model), "sampler_call_ops": call_ops,
                     "sampler_call_flops_per_row": call_flops_row / dn.SAMPLES}
    out["ts-100k"] = out["ts-native"]
    return out


# GPU model [INFERENCE]: time = launch latency x ops + FLOPs / throughput (RTX 5090; fp16 autocast for training,
# fp32 without TF32 for scoring under deterministic_scoring()); lower = max of the two terms at the fast end, upper =
# their sum at the slow end
GPU_MODEL = {"lower": {"launch_us": 5.0, "train_tflops": 150.0, "score_tflops": 60.0, "combine": "max"},
             "upper": {"launch_us": 15.0, "train_tflops": 40.0, "score_tflops": 20.0, "combine": "sum"}}
CALLS_PER_PATH = 1 + dn.HORIZON * (1 + 50 + dn.HORIZON)  # context + 95 pyramid rows x 45 frames (sampler calls)
VALIDATION_ROWS = dn.EARLY_STOPPING["validation_shots"] * dn.EARLY_STOPPING["samples"]


def gpu_model_rates(cpu, model):
    def t(ops, flops, tflops):
        latency, compute = ops * model["launch_us"] * 1e-6, flops / (tflops * 1e12)
        return max(latency, compute) if model["combine"] == "max" else latency + compute
    rates = {name: t(m["update_ops"], m["update_flops"], model["train_tflops"]) for name, m in cpu.items()}
    ts = cpu["ts-native"]

    def sampler(m, rows, tflops, calls=CALLS_PER_PATH):
        return calls * t(m["sampler_call_ops"], m["sampler_call_flops_per_row"] * rows, tflops)
    validation = sampler(ts, VALIDATION_ROWS, model["train_tflops"])  # validation under autocast (16-mixed)
    worst = max(cpu.values(), key=lambda m: m["sampler_call_flops_per_row"])
    per_decision = sampler(worst, ROWS_PER_CALL * dn.SAMPLES, model["score_tflops"]) / (ROWS_PER_CALL / 16)
    # G-fid: the time-series model on 60 frames (ops / FLOPs x 60 / 46) and the official 50 000-row validation
    # (FLOPs per row of the 236-d carrier model: an upper bound for 8 series)
    fid = {"update": rates["ts-native"] * GFID["frames"] / dn.FRAMES,
           "validation": sampler(ts, GFID["rows"], model["train_tflops"], GFID["calls"])}
    return rates, validation, per_decision, fid


def policy_cell_count():
    return len(eb.policy_cells(GRID))


def project(output, policy_cells=None, cpu=None):
    """projection.json (CPU, before funding): when timing.json (GPU) exists, the measured projection (primary);
    always the [INFERENCE] CPU model: op (kernel-launch) and FLOP counts of one update / one sampler call, measured
    on CPU, through GPU_MODEL's lower / upper ends, plus the CPU s/update for reference."""
    output = Path(output)
    cpu = cpu or cpu_measurements()
    policy_cells = policy_cell_count() if policy_cells is None else policy_cells
    result = {"schema": SCHEMA_PROJECTION, "plan_identity": IDENTITY,
              "label": "[INFERENCE] CPU micro-benchmark (synthetic carrier-shaped batches, 4 threads): op and FLOP "
                       "counts of one update / one sampler call mapped to an RTX 5090 by GPU_MODEL (lower: max(ops x "
                       "5 us, FLOPs / 150 TFLOP/s fp16 or 60 fp32); upper: ops x 15 us + FLOPs / 40 fp16 or 20 fp32); "
                       "ts-native assumes no early stop before the 200 000-update cap",
              "cpu": cpu, "threads": torch.get_num_threads(), "gpu_model": GPU_MODEL, "policy_cells": policy_cells,
              "calls_per_path": CALLS_PER_PATH, "measured_at": eb.utc_now()}
    for end, model in GPU_MODEL.items():
        rates, validation, per_decision, fid = gpu_model_rates(cpu, model)
        fid_h = fidelity_seconds(fid["update"], fid["validation"]) / 3600
        result[f"projection_{end}"] = {**child_projection(rates, fid_h, per_decision, policy_cells, validation),
                                       "fidelity_rates": fid}
    result["primary"] = "projection_upper"
    timing_path = output / "timing.json"
    if timing_path.is_file():
        record = eb.read_json(timing_path)
        if str(record.get("device", "")).startswith("cuda"):
            result["projection_measured"] = measured_projection(record, policy_cells)
            result["timing_sha256"] = eb.sha256_of(timing_path)
            result["primary"] = "projection_measured"
    eb.write_json(output / "projection.json", result)
    lo, hi = result["projection_lower"], result["projection_upper"]
    log(f"projection [INFERENCE]: {lo['total_gpu_h']:.1f}-{hi['total_gpu_h']:.1f} GPU-h, wall "
        f"{lo['wall_h_on_4_gpus_serial_ladder']:.1f}-{hi['wall_h_on_4_gpus_serial_ladder']:.1f} h; within caps "
        f"{lo['within_caps']} / {hi['within_caps']}; primary {result['primary']}")
    return 0


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _rss_gib():
    import resource
    current = None
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            current = int(line.split()[1]) / 2 ** 20
    return current, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def _time_updates(spec, data, updates, device):
    """Steady-state s/update of ``updates`` native updates (chunks of 25, the first chunk dropped) on fit batches."""
    torch.manual_seed(eb.SELECTION_SEED)
    trainer = dn.Trainer(spec["config"], data["mean"], data["std"], device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    chunks, losses, start = [], [], time.monotonic()
    for step in range(updates):
        losses.append(float(trainer.train_step(*training_batch(data, step, eb.SELECTION_SEED, spec, device))["loss"]))
        if (step + 1) % 25 == 0 or step + 1 == updates:
            _sync(device)
            now = time.monotonic()
            chunks.append({"updates": (step % 25) + 1, "seconds": now - start})
            start = now
    steady = [c["seconds"] / c["updates"] for c in chunks[1:]] or [c["seconds"] / c["updates"] for c in chunks]
    out = {"parameters": dn.parameter_count(trainer.model), "batch": spec["batch"], "updates": updates,
           "chunks": chunks, "seconds_per_update_steady_median": float(np.median(steady)),
           "first_loss": losses[0], "last_loss": losses[-1], "finite": bool(np.all(np.isfinite(losses))),
           "mixed": trainer.mixed}
    if device.startswith("cuda"):
        out.update(peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                   peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30)
    return trainer, out


def timing(output, updates=300, planning_updates=100, device=None, data=None, decisions=16, gfid_updates=100,
           gfid_rows=GFID["rows"]):
    """GPU timing probe (no funded.json; its own posted decision before enqueue; <= 1 GPU-h): ``updates`` official
    updates of the time-series model (ts-native / ts-100k: same model and batch) and ``planning_updates`` of the
    planning model on fit-split batches, one ts-native early-stopping validation pass, the scoring cost of
    ``decisions`` fit anchors (16 candidates x S paths, ROWS_PER_CALL chunks, deterministic scoring) per model, and the
    Exchange G-fid shape (gfid_timing). Writes timing.json and a ledger entry; keeps no model."""
    device = device_of(device)
    output = Path(output)
    began = time.monotonic()
    result = {"schema": SCHEMA_TIMING, "plan_identity": IDENTITY, "decision": "P-coverage timing probe (#136)",
              "device": device, "device_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
              "torch": torch.__version__}
    t = time.monotonic()
    data = data if data is not None else fit_store(eb.FitShots())
    result["fit_store_seconds"] = time.monotonic() - t
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
    costs = objective_costs()
    anchors = eb.fit_anchors()["carriers"]
    members = sorted(anchors)[:decisions]
    items, _ = grid_actions()
    result["configs"] = {}
    for name, count in (("ts-native", updates), ("planning", planning_updates)):
        spec = config_spec(name)
        trainer, out = _time_updates(spec, data, count, device)
        if name == "ts-native":
            rows = dn.validation_rows(len(data["launch5"]), spec["early_stopping"]["validation_shots"])
            _sync(device)
            t = time.monotonic()
            value = dn.validation_crps(trainer.model, data["carriers"], data["launch5"], rows,
                                       spec["early_stopping"]["samples"], autocast=trainer.mixed)
            _sync(device)
            out["validation"] = {"seconds": time.monotonic() - t, "crps_sum": value, "rows": VALIDATION_ROWS}
        model = trainer.model.eval()
        trainer = None
        with harness.deterministic_scoring():
            _sync(device)
            t = time.monotonic()
            levels_rows(model, [(items, anchors[m]) for m in members], costs)
            _sync(device)
            seconds = time.monotonic() - t
        out["scoring"] = {"decisions": len(members), "samples": dn.SAMPLES, "rows_per_call": ROWS_PER_CALL,
                          "seconds": seconds, "seconds_per_decision": seconds / len(members)}
        result["configs"][name] = out
        model = None
    result["gfid"] = gfid_timing(device, gfid_updates, gfid_rows)
    result["host_rss_gib"], result["host_peak_rss_gib"] = _rss_gib()
    result["seconds"] = time.monotonic() - began
    result["written_at"] = eb.utc_now()
    eb.write_json(output / "timing.json", result)
    eb.mark(output, "timing", result["seconds"], result["seconds"] if device.startswith("cuda") else 0.0)
    log("timing: " + ", ".join(f"{k} {v['seconds_per_update_steady_median']:.4f} s/update, "
                               f"{v['scoring']['seconds_per_decision']:.3f} s/decision"
                               for k, v in result["configs"].items())
        + f"; validation {result['configs']['ts-native']['validation']['seconds']:.1f} s ({result['seconds']:.0f} s)")
    projection_path = output / "projection.json"
    if device.startswith("cuda") and projection_path.is_file():  # the measured projection becomes primary
        projection = eb.read_json(projection_path)
        projection.update(projection_measured=measured_projection(result, projection["policy_cells"]),
                          primary="projection_measured", timing_sha256=eb.sha256_of(output / "timing.json"))
        eb.write_json(projection_path, projection)
    return 0


@torch.no_grad()
def _gfid_calls(tm, rows, calls, device, mixed):
    pairs = dn.ddim_times(tm)
    x = torch.randn(rows, GFID["x_dim"], device=device)
    z = torch.zeros(rows, tm.z_shape[0], device=device)
    c = torch.zeros(rows, GFID["cond_dim"], device=device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=mixed):
        dn.ddim_step(tm, x, z, c, 10, dn.GlobalNoise(), pairs)  # warm-up call
        _sync(device)
        t = time.monotonic()
        for _ in range(calls):
            x, z = dn.ddim_step(tm, x, z, c, 10, dn.GlobalNoise(), pairs)
        _sync(device)
    return (time.monotonic() - t) / calls


def gfid_timing(device, updates=100, rows=GFID["rows"], calls=10):
    """The Exchange G-fid model at the official shape (vendored DiffusionTransitionModel, x 8 series, 20 covariates,
    time-series recipe; synthetic inputs): steady-state s/update on 60-frame batches of 32 (the df_base training step
    and Lightning's 16-mixed optimizer step, as dn.Trainer) and s per DDIM sampler call at ``rows`` rows (16-mixed, as
    the official validation)."""
    config = dn.config_of("ts-native")
    torch.manual_seed(0)
    tm = dn.vendor_models().DiffusionTransitionModel([GFID["x_dim"]], [config["z_dim"]], GFID["cond_dim"],
                                                     dn.diffusion_cfg(config)).to(device)
    params = list(tm.parameters())
    optimizer = torch.optim.AdamW(params, lr=config["lr"], weight_decay=config["weight_decay"])
    mixed = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda") if mixed else None
    xs = torch.randn(GFID["frames"], GFID["batch"], GFID["x_dim"], device=device)
    cs = torch.randn(GFID["frames"], GFID["batch"], GFID["cond_dim"], device=device)
    chunks, start = [], time.monotonic()
    for step in range(updates):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=mixed):
            z, losses = torch.zeros(GFID["batch"], config["z_dim"], device=device), []
            for t in range(GFID["frames"]):
                z, _, loss, _ = tm(z, xs[t], cs[t])
                losses.append(loss)
            loss = torch.stack(losses).mean()
        (scaler.scale(loss) if scaler else loss).backward()
        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, config["gradient_clip_val"])
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if (step + 1) % 25 == 0 or step + 1 == updates:
            _sync(device)
            now = time.monotonic()
            chunks.append({"updates": (step % 25) + 1, "seconds": now - start})
            start = now
    steady = [c["seconds"] / c["updates"] for c in chunks[1:]] or [c["seconds"] / c["updates"] for c in chunks]
    return {"shape": GFID, "updates": updates, "chunks": chunks, "seconds_per_update_steady_median": float(np.median(
        steady)), "rows": rows, "seconds_per_sampler_call": _gfid_calls(tm, rows, calls, device, mixed),
        "parameters": sum(p.numel() for p in params)}


def measured_projection(timing_record, policy_cells):
    """The child projection from timing.json: measured steady-state s/update per configuration, the measured
    ts-native validation pass, the slower configuration's measured s/decision; the Exchange G-fid from the measured
    official-shape update and sampler call (fidelity_seconds: no early stop before the cap, an upper bound)."""
    configs = timing_record["configs"]
    rates = {"ts-native": configs["ts-native"]["seconds_per_update_steady_median"],
             "ts-100k": configs["ts-native"]["seconds_per_update_steady_median"],
             "planning": configs["planning"]["seconds_per_update_steady_median"]}
    validation = configs["ts-native"]["validation"]["seconds"]
    per_decision = max(c["scoring"]["seconds_per_decision"] for c in configs.values())
    g = timing_record["gfid"]
    fid = {"update": g["seconds_per_update_steady_median"],
           "validation": g["seconds_per_sampler_call"] * GFID["rows"] / g["rows"] * GFID["calls"]}
    fid_h = fidelity_seconds(fid["update"], fid["validation"]) / 3600
    return {"child": child_projection(rates, fid_h, per_decision, policy_cells, validation),
            "gpu_seconds_per_update": rates, "validation_seconds": validation, "seconds_per_decision": per_decision,
            "fidelity_rates": fid, "source": "timing.json (measured)"}


def smoke(output, device=None):
    device = device_of(device)
    output = Path(output)
    shots = eb.FitShots()
    selected = selected_config(output)
    selection_record = eb.read_json(output / "selection.json")
    launch = torch.as_tensor(shots.launch5)
    accounting = {"fit_shots": len(shots), "fit_shots_expected": FIT_SHOTS, "fit_levels": len(shots.levels),
                  "fit_positive": int(shots.verdicts.sum()), "fit_anchors": len(eb.fit_anchors()["carriers"]),
                  "policy_grid_mixed_cells": len(eb.policy_cells(GRID)), "frames": dn.FRAMES,
                  "shots_ending_before_225": int((np.asarray(shots.ends) < dn.END_FRAME).sum()),
                  "launch5_abs_max": float(launch.abs().max())}
    gport = eb.read_json(output / "gport/gport.json")
    adapter = eb.sha256_of(ROOT / "world_model/baselines/diffusion_forcing/novphy.py").split(":")[-1]
    gport_check = {"pass": gport.get("pass") is True,
                   "adapter_current": gport.get("adapter_sha256", "").split(":")[-1] == adapter}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", TESTS], cwd=ROOT, env=env,
                           capture_output=True, text=True)
    tests_record = {"command": f"CUDA_VISIBLE_DEVICES= python -m pytest -q {TESTS}", "exit_code": tests.returncode,
                    "tail": tests.stdout.strip().splitlines()[-1:] if tests.stdout else []}
    throughput = {}
    for name in CONFIGS:
        report = eb.read_json(selection_dir(output, name) / "report.json")
        throughput[name] = {"seconds_per_step": report["seconds_per_step"], "steps": report["steps"],
                            "steps_run": report["steps_run"], "gpu_seconds": report["gpu_seconds"]}
    gfid = eb.read_json(output / "fidelity/gfid.json")
    remaining = remaining_items(output, selected)
    rate = throughput[selected]["seconds_per_step"]  # includes the early-stopping validations of ts-native
    training = [(f"{arm}--seed{seed}", spec["steps"] * rate) for arm, seed, spec, _ in remaining]  # cap: upper bound
    selection_seconds = sum(t["gpu_seconds"] for t in throughput.values())
    timing_record = eb.read_json(output / "timing.json") if (output / "timing.json").is_file() else None
    per_decision = (max(c["scoring"]["seconds_per_decision"] for c in timing_record["configs"].values())
                    if timing_record else 2.0)
    counts = decision_counts(accounting["policy_grid_mixed_cells"])
    readout_seconds = counts["readouts_per_arm_seed"] * len(eb.SEEDS) * per_decision
    score_seconds = counts["score_per_seed"] * len(eb.SEEDS) * per_decision
    queues, loads = lpt(training)
    total = selection_seconds + sum(s for _, s in training) + readout_seconds + score_seconds
    per_seed = arm_spec(ARM, selected)["steps"] * rate
    wall = max(loads) + readout_seconds / 4 + score_seconds
    projection = {"fidelity_gpu_h_spent": gfid.get("compute", {}).get("gpu_seconds", 0.0) / 3600,
                  "selection_gpu_h": selection_seconds / 3600, "training_gpu_h": sum(s for _, s in training) / 3600,
                  "readouts_gpu_h_estimate": readout_seconds / 3600, "score_gpu_h_estimate": score_seconds / 3600,
                  "seconds_per_decision": per_decision, "decision_counts": counts,
                  "total_gpu_h": total / 3600, "child_cap_gpu_h": CHILD_CAP_GPU_HOURS,
                  "native_per_seed_gpu_h": per_seed / 3600, "seed_cap_gpu_h": SEED_CAP_GPU_HOURS,
                  "wall_h_on_4_gpus": wall / 3600, "wall_cap_h": WALL_CAP_HOURS, "training_queues": queues,
                  "training_queue_loads_h": [x / 3600 for x in loads]}
    projection["within_caps"] = bool(total / 3600 <= CHILD_CAP_GPU_HOURS and per_seed / 3600 <= SEED_CAP_GPU_HOURS
                                     and wall / 3600 <= WALL_CAP_HOURS)
    smoke_record = {"schema": SCHEMA_SMOKE, "plan_identity": IDENTITY, "accounting": accounting, "gport": gport_check,
                    "readiness": readiness(output), "tests": tests_record, "throughput": throughput,
                    "selection": selection_record, "selected": selected, "projection": projection, "device": device,
                    "pass": bool(gport_check["pass"] and gport_check["adapter_current"] and tests.returncode == 0
                                 and projection["within_caps"] and accounting["fit_shots"] == FIT_SHOTS
                                 and accounting["launch5_abs_max"] <= 1.0),
                    "written_at": eb.utc_now()}
    eb.write_json(output / "smoke.json", smoke_record)
    log(f"smoke: pass {smoke_record['pass']}; projection {projection['total_gpu_h']:.1f} GPU-h, wall "
        f"{projection['wall_h_on_4_gpus']:.1f} h; selected {selected}")
    return 0 if smoke_record["pass"] else 1


# ---------------------------------------------------------------------------------------------------------------
# --prepare (freeze) and load_plan
# ---------------------------------------------------------------------------------------------------------------

OWN_FILES = [RUNNER, TESTS, "world_model/baselines/diffusion_forcing/__init__.py",
             "world_model/baselines/diffusion_forcing/novphy.py", "scripts/b127_dforce_port_reference.py",
             "scripts/b127_dforce_fidelity.py"]


def own_pins(output):
    output = Path(output)
    paths = [ROOT / p for p in OWN_FILES]
    paths += sorted(p for p in (ROOT / "world_model/baselines/diffusion_forcing/vendor").rglob("*")
                    if p.is_file() and "__pycache__" not in str(p))
    paths += [output / "gport/gport.json", output / "fidelity/band.json", output / "fidelity/gfid.json",
              ROOT / "data/external/diffusion_forcing/MANIFEST.json",
              output / "smoke.json", output / "selection.json", output / "funded.json"]  # decisions.md: append-only
    paths += [selection_dir(output, name) / f for name in CONFIGS for f in ("model.pt", "report.json")]
    return [{"name": f"t4_{p.relative_to(ROOT)}", "artifact": str(p.relative_to(ROOT)), "sha256": eb.sha256_of(p)}
            for p in paths if p.is_file()]


def readout_items():
    return [(arm, seed) for arm in ARMS for seed in eb.SEEDS]


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
    selected = smoke_record["selected"]
    queues = smoke_record["projection"]["training_queues"]
    readout_queues, _ = lpt([(f"{a}--seed{s}", 1.0) for a, s in readout_items()])
    plan = {
        "schema": SCHEMA_PLAN, "plan_identity": IDENTITY, "issue": ISSUE, "program_issue": eb.PROGRAM_ISSUE,
        "frozen_at": eb.utc_now(), "pins": eb.shared_inputs() + own_pins(output),
        "selected": selected, "selection_rule": eb.SELECTION_RULE, "configs": CONFIGS,
        "arms": {arm: {**ARMS[arm], "spec": arm_spec(arm, selected), "seeds": list(eb.SEEDS)} for arm in ARMS},
        "training_precision": TRAINING_PRECISION, "scoring_policy": harness.SCORING_POLICY,
        "reused": {ARM: {"seed": eb.SELECTION_SEED, "run": str(selection_dir(output, selected).relative_to(ROOT))}},
        "readout": {"score": f"expected tie-free EndpointCosts cost of the frame-225 carrier over S = {dn.SAMPLES} "
                             "paths sampled with the causal-uncertainty pyramid from the anchor carrier (common random "
                             f"numbers: CPU generator seed {dn.SAMPLE_SEED}, shared by every candidate); any nonfinite "
                             "sample -> excluded typed row",
                    "action_sensitivity": "eb.action_sensitivity on the per-candidate sample-mean frame-225 carriers, "
                                          "480 fit anchors x 16 grid launches",
                    "collapse": "not in the ticket's guard list (scores in carrier space)",
                    "positions": "sample-mean carriers at frames 15, 30, 60, 150, 225 (descriptive)"},
        "queues": {"train": {str(k): v for k, v in queues.items()},
                   "readouts": {str(k): v for k, v in readout_queues.items()}, "score": "GPU 0, one process"},
        "seeds": list(eb.SEEDS), "guards": eb.GUARD, "arm_guards": ARM_CHECKS, "extra_guards": ["G_port", "G_fid"],
        "global_guards": ["G_isolation", "G_chronology", "G_scoring_policy", "G_records"],
        "contrasts": {"per_arm": list(eb.CONTRAST_REFERENCES), "within_family": [], "wrong_anchor": list(ARMS)},
        "dispositions": eb.DISPOSITION_LABELS, "delta": eb.DELTA, "decisions": DECISIONS,
        "disclosures": [
            "T4 is a coverage arm (P-coverage; Tier-C if triggered): secondary, reported, never in the program claim.",
            "T4's configuration was selected on the policy split among three pre-declared ones (ts-native, ts-100k, "
            "planning); U had no such tuning (#127).",
            "The time-series recipe samples autoregressively in the paper (chunk 1); T4 samples the whole 45-step "
            "horizon with the causal-uncertainty pyramid (the planning experiments' schedule; ticket).",
            "ts-native's early stopping validates on 16 fixed fit shots x 100 paths (the official validation set is a "
            "random subset of the training series); the best validation checkpoint is the model.",
            "The planning recipe runs with frame_stack 1 (46 frames are not divisible by 10) and the launch through "
            "the external-condition input (the maze recipe puts actions in x and has no condition input).",
            "16-mixed training on CUDA as upstream; the G-port checks the fp32 path on CPU.",
            "The score is a Monte-Carlo expectation over 32 sampled paths with common random numbers.",
            "No native success head, no U + head control; G-collapse is not in the ticket's guard list.",
            "Scoring waits for G-fid pass (Table 2 Exchange); a G-fid failure drops T4 (fidelity_not_reproduced).",
            "P-novelty: after the primary scoring pass, T4 (if its guards pass) is scored on the #110 novelty levels "
            "from the E99 v1 anchor parse (--score-novelty); descriptive only, no tokens, never in a disposition.",
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
    if not funded(output):
        log("train refused: funded.json absent; nothing trained")
        return 1
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
        if name in CONFIGS:
            seed = int(seed)
            if seed == eb.SELECTION_SEED:
                return selection_dir(output, name, seed) / "report.json"
            if plan and name == plan["selected"]:
                return model_dir(output, ARM, seed, plan["selected"]) / "report.json"
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
    result = {"arm": arm, "seed": seed, "plan_identity": IDENTITY}
    path = model_dir(output, arm, seed, plan["selected"]) / "model.pt"
    with harness.deterministic_scoring() as policy:
        result["scoring_policy"] = policy
        model, payload = load_trained(path, device)
        result["model"] = rel(path)
        result["model_sha256"] = eb.sha256_of(path)
        if model is None:
            result.update(typed_failure=payload["typed_failure"]["typed_failure"])
        else:
            members, matrix, means, typed = fit_anchor_readouts(model, costs)
            items, actions = grid_actions()
            result.update(typed_failure=None, G_rank=eb.g_rank(matrix), G_action=float(eb.action_sensitivity(means)),
                          G_collapse=None, fit_anchor_nonfinite_share=float(np.isnan(matrix).mean()),
                          fit_anchor_typed_rows=typed, trainable_parameters=dn.parameter_count(model))
            prow = policy_rows(model, costs)
            result["policy_auc"] = eb.policy_grid_auc(prow, eb.PRIMARY_COST)
            anchors = eb.fit_anchors()["carriers"]
            z0 = anchors[members[0]][None].expand(len(items), -1)
            result["macs_per_candidate"] = eb.macs_per_candidate(
                lambda a, b: dn.candidate_paths(model, a, b, dn.SAMPLES), z0, actions, candidates=len(items))
    seconds = time.monotonic() - began
    result.update(seconds=seconds, written_at=eb.utc_now())
    eb.mark(output, f"readouts-{arm}-seed{seed}", seconds, seconds if device != "cpu" else 0)
    eb.write_json(out_dir / "readouts.json", result)  # written last (atomic)
    log(f"readouts {arm} seed {seed}: {json.dumps({k: result.get(k) for k in ('policy_auc', 'G_action')})}")
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
        mpath = model_dir(output, arm, seed, plan["selected"]) / "model.pt"
        entries[f"{arm}--seed{seed}"] = {"readouts": rel(rdir / "readouts.json"),
                                         "readouts_sha256": eb.sha256_of(rdir / "readouts.json"),
                                         "model": rel(mpath), "model_sha256": eb.sha256_of(mpath)}
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
        for kind in ("model", "readouts"):
            if kind in entry and eb.sha256_of(ROOT / entry[kind]) != entry[f"{kind}_sha256"]:
                raise ValueError(f"{name}: frozen {kind} changed")
    return manifest


# ---------------------------------------------------------------------------------------------------------------
# --score (once; a process that did not train; GPU 0; deterministic; only after G-fid passed)
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
    error (descriptive; sample-mean carriers), wall time per decision."""
    device = device_of(device)
    costs = objective_costs()
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "arm": arm, "seed": seed}
    model, payload = load_trained(model_dir(output, arm, seed, plan["selected"]) / "model.pt", device)
    if model is None:
        return {**base, "typed_failure": payload["typed_failure"]["typed_failure"]}
    carriers = anchors["carriers"]
    inventories = {}
    for inventory in eb.INVENTORIES:
        mixed = sorted(cells[inventory]["mixed"].items())
        out = levels_rows(model, [(entry["inventory"], carriers[m]) for m, entry in mixed], costs)
        inventories[inventory] = {m: {"rows": rows} for (m, _), (rows, _) in zip(mixed, out)}
    record = {**base, "typed_failure": None, "inventories": inventories}
    fit = eb.fit_anchors()["carriers"]
    items, _ = grid_actions()
    wrong = levels_rows(model, [(items, fit[m]) for m in sorted(fit)], costs)
    record["wrong_anchor"] = {m: {"rows": rows} for m, (rows, _) in zip(sorted(fit), wrong)}
    cache = eb.position_cache()
    branches = eb.position_branches(cells)
    z0 = torch.stack([carriers[m] for m, _ in branches])
    acts = torch.stack([eb.launch5(it["action"]) for _, it in branches])
    paths = row_samples(model, z0, acts, keep=tuple(POSITION_INDEX))
    at = {t: dn.sample_means(paths[t // dn.STRIDE]) for t in eb.POSITION_TIMES}
    record["positions"] = _position_rows(at, branches, cache, eb.POSITION_TIMES)
    members = sorted(cells[GRID]["mixed"])[:20]
    record["wall_per_decision"] = eb.wall_per_decision(lambda m: level_rows(model, items, carriers[m], costs), members)
    return record


def score(output, device=None):
    output = Path(output)
    plan = load_plan(output)
    load_manifest(output)
    if not gfid_pass(output):
        log("score refused: fidelity/gfid.json has not passed (#136 gate); evaluation not read")
        return 1
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
# --score-novelty (P-novelty, DESCRIPTIVE; once, after --score; GPU 0; deterministic; resumable per record)
# ---------------------------------------------------------------------------------------------------------------

SCHEMA_NOVELTY_RECORD = "issue_136_novelty_record_v1"
NOVELTY_ARMS = tuple(ARMS)


def novelty_dir(output):
    return Path(output) / "novelty"


def novelty_record_path(output, arm, seed):
    return novelty_dir(output) / "records" / f"{arm}--seed{seed}.json"


def novelty_rows(model, cells, carriers, costs):
    """{level: {"rows"}} of the mixed novelty levels (grid) from each level's carrier anchor (levels_rows)."""
    levels = sorted(cells["mixed"])
    out = levels_rows(model, [(cells["levels"][lv]["inventory"], carriers[lv]) for lv in levels], costs)
    return {lv: {"rows": rows} for lv, (rows, _) in zip(levels, out)}


def score_novelty(output, device=None, cells=None):
    """P-novelty: every guard-passing arm, every seed, on the mixed novelty levels (grid) from each level's carrier
    anchor (the primary scoring path, levels_rows); refuses before the primary records and the readouts freeze exist;
    never touches a primary record."""
    output = Path(output)
    plan = load_plan(output)
    load_manifest(output)
    missing = [f"{a}--seed{s}" for a, s in readout_items() if not record_path(output, a, s).is_file()]
    if missing:
        log(f"score-novelty refused: primary records missing ({missing})")
        return 1
    records = load_records(output)
    guards_of, extra = arm_guards(output, records), guard_extras(output)
    status = {arm: {"guards_pass": arm_guard_pass(guards_of[arm], extra),
                    "failed": sorted(k for k, v in {**guards_of[arm], **extra}.items()
                                     if k.startswith("G_") and v is not True)} for arm in NOVELTY_ARMS}
    costs = objective_costs()
    with harness.deterministic_scoring() as policy:
        cells = eb.novelty_cells() if cells is None else cells
        path = eb.novelty_anchors_path(output)
        anchors = (torch.load(path, map_location="cpu", weights_only=True) if path.is_file()
                   else eb.read_novelty_anchors(output, policy, cells))
        eb.novelty_reference_records(output, policy, cells, anchors["carriers"])
        eb.write_json(novelty_dir(output) / "arms.json", {"schema": "issue_136_novelty_arms_v1", "arms": status,
                                                          "anchors_read_at": anchors["read_at"]})
        for arm in NOVELTY_ARMS:
            if not status[arm]["guards_pass"]:
                log(f"score-novelty: {arm} not scored (failed guards {status[arm]['failed']})")
                continue
            for seed in eb.SEEDS:
                rpath = novelty_record_path(output, arm, seed)
                if rpath.is_file():
                    continue
                began = time.monotonic()
                mpath = model_dir(output, arm, seed, plan["selected"]) / "model.pt"
                model, _ = load_trained(mpath, device)
                grid = novelty_rows(model, cells, anchors["carriers"], costs)
                seconds = time.monotonic() - began
                eb.write_json(rpath, {"schema": SCHEMA_NOVELTY_RECORD, "arm": arm, "seed": seed,
                                      "model_sha256": eb.sha256_of(mpath), "inventories": {GRID: grid},
                                      "scoring_policy": policy, "scored_at": eb.utc_now(),
                                      "anchors_read_at": anchors["read_at"]})
                eb.mark(output, f"score-novelty-{arm}-seed{seed}", seconds,
                        seconds if device_of(device) != "cpu" else 0)
                log(f"novelty scored {arm} seed {seed} ({seconds:.0f}s)")
    return 0


def novelty_section(output, records, cells=None):
    """DESCRIPTIVE novelty table (no tokens, never in a disposition) when the novelty pass has run, else None."""
    output = Path(output)
    arms_path = novelty_dir(output) / "arms.json"
    if not arms_path.is_file():
        return None
    status = eb.read_json(arms_path)["arms"]
    scored = [arm for arm in NOVELTY_ARMS if status[arm]["guards_pass"]]
    cells = eb.novelty_cells() if cells is None else cells
    anchors_read = torch.load(eb.novelty_anchors_path(output), map_location="cpu", weights_only=True)["read_at"]
    views = {arm: ({seed: eb.read_json(novelty_record_path(output, arm, seed)) for seed in eb.SEEDS},
                   eb.PRIMARY_COST) for arm in scored}
    refs = eb.novelty_reference_records(output, None, cells)  # read back (written once by --score-novelty)
    last_primary = max(r["scored_at"] for recs in records.values() for r in recs.values())
    return {"label": "DESCRIPTIVE (P-novelty)", "scored_arms": scored,
            "not_scored": {arm: status[arm]["failed"] for arm in NOVELTY_ARMS if arm not in scored},
            "chronology": {"pass": bool(anchors_read >= last_primary), "novelty_anchors_read_at": anchors_read,
                           "last_primary_scored_at": last_primary},
            "table": eb.novelty_table({**views, **refs}, cells)}


# ---------------------------------------------------------------------------------------------------------------
# --publish / --validate
# ---------------------------------------------------------------------------------------------------------------

RECORD_PATTERNS = ["records/*.json", "novelty/arms.json", "novelty/records/*.json", "novelty/references/*.json"]


def load_records(output):
    return {arm: {seed: eb.read_json(record_path(output, arm, seed)) for seed in eb.SEEDS} for arm in ARMS}


def global_guards(output, plan, records):
    output = Path(output)
    manifest = eb.read_json(output / "readouts-manifest.json")
    anchors_read = min(r["anchors_read_at"] for recs in records.values() for r in recs.values())
    first_scored = min(r["scored_at"] for recs in records.values() for r in recs.values())
    chronology = bool(plan["frozen_at"] <= manifest["frozen_at"] <= anchors_read <= first_scored)
    policies = [r["scoring_policy"] for recs in records.values() for r in recs.values()]
    return {
        "G_isolation": {"pass": True, "rule": "fit split trains, policy split selects; evaluation anchors are read "
                        "only by --score after --freeze-readouts (G_chronology) and after G-fid passed"},
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


def guard_extras(output):
    return {"G_port": gport_pass(output), "G_fid": gfid_pass(output)}


def arm_guards(output, records):
    """{arm: eb.arm_guard_block} of the arm checks from the readouts and the primary records (publication and
    --score-novelty use this one computation)."""
    guards_of = {}
    for arm in ARMS:
        per_seed = {}
        for seed in eb.SEEDS:
            rd = eb.read_json(readouts_dir(output, arm, seed) / "readouts.json")
            per_seed[seed] = {"G_rank": rd.get("G_rank"), "G_action": rd.get("G_action"), "G_collapse": None,
                              "policy_ceiling_auc": None,
                              "typed_failure": rd.get("typed_failure") or records[arm][seed].get("typed_failure")}
        guards_of[arm] = eb.arm_guard_block(per_seed, ARM_CHECKS,
                                            nonfinite=eb.nonfinite_share(records[arm].values(), eb.PRIMARY_COST))
    return guards_of


def arm_guard_pass(guards, extra_guards):
    """eb.arm_block's arm pass: every G_* entry of the arm guards and the extra guards (G_port, G_fid) is True."""
    merged = {**guards, **extra_guards}
    names = [k for k in merged if k.startswith("G_")]
    return bool(names) and all(merged[k] is True for k in names)


def publication(output):
    """Every published table, derived from the frozen plan, readouts and records (deterministic)."""
    output = Path(output)
    plan = load_plan(output)
    records = load_records(output)
    cells = eb.evaluation_cells()
    refs = eb.references()
    guards_global = global_guards(output, plan, records)
    global_pass = all(g["pass"] is True for g in guards_global.values())
    extra_guards = guard_extras(output)
    views = {arm: (records[arm], eb.PRIMARY_COST) for arm in ARMS}
    blocks, tables = {}, {}
    guards_of = arm_guards(output, records)
    for inventory in eb.INVENTORIES:
        rows = eb.view_rows(cells, inventory, views, refs)
        tables[inventory] = {"rankers": eb.ranker_table(rows, ["U", "C*", "P", "J", "J_dec", *views])}
        if inventory == GRID:
            for arm in ARMS:
                blocks[arm] = eb.arm_block(rows, arm, records[arm], eb.PRIMARY_COST, guards_of[arm], global_pass,
                                           arm_compute(output, plan, arm), ARMS[arm]["role"], ARMS[arm]["program_set"],
                                           extra_guards=extra_guards)
        else:
            tables[inventory]["contrasts"] = {
                f"{arm} - U": eb.contrast(rows, arm, "U", global_pass and blocks[arm]["guards"]["pass"])
                for arm in ARMS}
    wrong = {arm: eb.wrong_anchor_table(cells, records[arm], arm, eb.PRIMARY_COST,
                                        global_pass and blocks[arm]["guards"]["pass"]) for arm in ARMS}
    curves = eb.reference_position_curves()
    positions = eb.position_table({ARM: records[ARM], "U": curves["U"], "C*": curves["C*"]}, cells)
    gfid_path = output / "fidelity/gfid.json"
    gfid_block = {ARM: {"pass": extra_guards["G_fid"], "record": rel(gfid_path),
                        "sha256": eb.sha256_of(gfid_path) if gfid_path.is_file() else None}}
    ledger, gpu_total, wall_total = eb.ledger_totals(output, training_report_of(output, plan))
    summary = eb.child_summary(ISSUE, IDENTITY, plan["frozen_at"], blocks, {}, guards_global, gfid_block, {
        "inventories": tables, "wrong_anchor": wrong, "positions": positions, "selection": plan["selected"],
        "coverage_arm": True,
        "readouts": {f"{a}--seed{s}": {k: v for k, v in eb.read_json(readouts_dir(output, a, s) / "readouts.json").items()
                                       if k in ("policy_auc", "G_action", "fit_anchor_typed_rows", "typed_failure")}
                     for a, s in readout_items()},
        "novelty": novelty_section(output, records),
        "native_success_head": None, "u_plus_head": "not applicable: no reward head (task-free sequence model)"})
    compute = {"schema": "issue_136_compute_v1", "plan_identity": IDENTITY, "ledger": ledger,
               "gpu_seconds_total": gpu_total, "wall_seconds_total": wall_total,
               "arms": {arm: blocks[arm]["compute"] for arm in ARMS}, "reference": eb.reference_compute()}
    return summary, compute


def findings_text(summary):
    lines = [f"# #{ISSUE} T4 Diffusion Forcing (coverage arm): findings", "",
             f"Global guards pass: {summary['global_guards_pass']}; selected configuration "
             f"{summary['extra']['selection']}.", "", *eb.format_arm_rows(summary["arms"]), "",
             "## Wrong anchor (right - wrong, grid)", ""]
    for arm, block in summary["extra"]["wrong_anchor"].items():
        lines.append(f"- {arm}: {eb.rc.fmt_block(block['right_minus_wrong'])} ({block['token']})")
    novelty = summary["extra"]["novelty"]
    if novelty is not None:
        lines += ["", f"## Novelty pairs ({novelty['table']['label']})", "",
                  f"Not scored (guards): {novelty['not_scored'] or 'none'}; novelty anchors read after the primary "
                  f"records: {novelty['chronology']['pass']}.", "", *eb.format_novelty_rows(novelty["table"])]
    lines += ["", "T4 is a coverage arm (P-coverage): never in the program claim. Intervals are descriptive "
              "(member-clustered bootstrap); tokens follow the #127 delta rule.", ""]
    return "\n".join(lines)


def not_run_reason(output):
    """The typed reason T4 is not run (no plan.json): the orchestrator's not-run.json marker (e.g. {"typed_reason":
    "cap", "decision": url}); None while T4 may still run."""
    marker = Path(output) / "not-run.json"
    return eb.read_json(marker)["typed_reason"] if marker.is_file() else None


def publish_not_run(output, reason):
    """summary.json of a T4 that was not run (typed): no arm block, extra.pending_arms {T4: reason} (listed as not run
    by the program summary), the G-port / G-fid / projection records that exist."""
    output = Path(output)
    gfid_path = output / "fidelity/gfid.json"
    projection_path = output / "projection.json"
    gfid_block = {ARM: {"pass": gfid_pass(output), "record": rel(gfid_path),
                        "sha256": eb.sha256_of(gfid_path) if gfid_path.is_file() else None}}
    projection = eb.read_json(projection_path) if projection_path.is_file() else None
    if projection is not None:
        primary = projection[projection["primary"]]
        primary = primary.get("child", primary)  # projection_measured nests the child projection
        projection = {"primary": projection["primary"], "total_gpu_h": primary["total_gpu_h"],
                      "sha256": eb.sha256_of(projection_path)}
    summary = eb.child_summary(ISSUE, IDENTITY, None, {}, {}, {}, gfid_block, {
        "pending_arms": {ARM: reason}, "coverage_arm": True, "G_port": gport_pass(output), "projection": projection,
        "not_run_marker": eb.read_json(output / "not-run.json"), "decisions": DECISIONS})
    eb.write_json(output / "summary.json", summary)
    (output / "findings.md").write_text(
        f"# #{ISSUE} T4 Diffusion Forcing (coverage arm): not run\n\nTyped reason: `{reason}`. G-port "
        f"{gport_pass(output)}; G-fid {gfid_pass(output)}. No records; T4 is listed as not run in the program "
        "summary.\n")
    log(f"published not-run summary ({reason})")
    return 0


def publish(output):
    output = Path(output)
    if not (output / "plan.json").is_file():
        reason = not_run_reason(output)
        if reason is None:
            raise SystemExit("no plan.json and no typed not-run reason: nothing to publish")
        return publish_not_run(output, reason)
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
        arm, seed = ARM, eb.SELECTION_SEED
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
    for mode in ("project", "timing", "selection", "selection-readouts", "smoke", "prepare", "train", "readouts",
                 "freeze-readouts", "score", "score-novelty", "publish", "validate"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="queue index (the job's CUDA_VISIBLE_DEVICES picks the GPU)")
    parser.add_argument("--workers", type=int, default=0, help="accepted for the contract; data live in memory")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--timing-updates", type=int, default=300, help="--timing: time-series updates")
    parser.add_argument("--timing-planning-updates", type=int, default=100, help="--timing: planning updates")
    args = parser.parse_args(argv)
    output = Path(args.output)
    torch.set_num_threads(int(os.environ.get("B127_T4_CPU_THREADS", "4")))  # shared host: avoid oversubscription
    if args.timing:
        return timing(output, args.timing_updates, args.timing_planning_updates, device=args.device)
    if args.project:
        return project(output)
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
    if args.score_novelty:
        return score_novelty(output, args.device)
    if args.publish:
        return publish(output)
    return validate(output, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
