"""Issue #134 (#127 coverage arm T3, rule P-coverage; Tier-C if D1 - U or D1 - J_dec is supported): γ-models on E99
carriers against U on the sealed NovPhy launch-ranking grid.

Arm (Decision P-other adapter on #134, see DECISIONS["adapter"]):
- **T3** (coverage, secondary; never in the program claim): the official jannerm/gamma-models conditional
  neural-spline-flow γ-model (vendored @ bb23e06; adapter ``world_model/baselines/gamma_models/novphy.py``) at the
  official architecture defaults (10 230 696 parameters), trained with the official γ-bootstrapped log-density
  regression (the Colab notebook's loop body: batch 1024, Adam 1e-4 / wd 1e-5, τ 0.005, σ 0.1, sample discount 0.9,
  2000 burn-in updates at γ 0.5, 50 000 updates) on fit-shot carrier transitions at frame stride 5 (the launch on the
  first transition, no-ops after, the frame-225 carrier absorbing). Score = expected tie-free ``EndpointCosts`` cost
  over S = 64 samples of p_γ(· | s0, a) (one fixed base-noise draw shared by every candidate); a candidate with a
  nonfinite sample or a sampling assertion is a retained typed row. Configuration = γ, selected on the policy split
  among three pre-declared values (``gn.CONFIGURATIONS``: 0.95, 1 - 1/45, 0.99; seed 20260908, full budget); the
  selected run is reused as main seed 20260908.
- No native success head; no U + head control; no matched-budget arm; no latent (G-collapse not applicable); the
  occupancy has no timestep (position error not applicable).

Gates (#134): ``--selection`` and ``--train`` refuse without the orchestrator's ``funded.json`` (Decision P-coverage
funding), a passing ``gport/gport.json`` and a passing ``fidelity/gfid.json`` (pendulum Fig. 4, CPU); ``--score``
refuses unless G-fid passed. ``--timing`` (a GPU probe, <= 0.5 GPU-h) needs no funded.json but its own posted decision
before it is enqueued.

Modes (round3/child-runner-contract.md): --project (CPU, pre-funding GPU-h projection), --timing, --selection,
--selection-readouts, --smoke, --prepare, --train, --readouts, --freeze-readouts, --score, --score-novelty
(P-novelty, descriptive), --publish, --validate. GPU work only through the night-window scheduler
(scripts/gpu_window_issue_134.json, prepared, not enqueued before funding); tests use CUDA_VISIBLE_DEVICES="".
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
from world_model.baselines.gamma_models import novphy as gn  # noqa: E402
from world_model.planning import scoring_harness as harness  # noqa: E402

ISSUE = 134
IDENTITY = "issue-134-gamma-models-v1"
OUTPUT = ROOT / ".local-artifacts" / IDENTITY
RUNNER = "scripts/run_issue_134_gamma_models.py"
TESTS = "tests/test_issue_134_gamma_models.py"
SCHEMA_MODEL = "issue_134_model_v1"
SCHEMA_PLAN = "issue_134_plan_v1"
SCHEMA_RECORD = "issue_134_record_v1"
SCHEMA_SMOKE = "issue_134_smoke_v1"
SCHEMA_SELECTION = "issue_134_selection_v1"
SCHEMA_READOUTS_MANIFEST = "issue_134_readouts_manifest_v1"
SCHEMA_PROJECTION = "issue_134_projection_v1"
SCHEMA_TIMING = "issue_134_timing_v1"
DECISIONS = {
    "coverage": "https://github.com/Sino-Huang/NovPhy/issues/134#issuecomment-6390745463",
    "license": "https://github.com/Sino-Huang/NovPhy/issues/134#issuecomment-6390841265",
    "band": "https://github.com/Sino-Huang/NovPhy/issues/134#issuecomment-6390881384",
    "adapter": "https://github.com/Sino-Huang/NovPhy/issues/134#issuecomment-6391005347",
}

CONFIGS = dict(gn.CONFIGURATIONS)  # g0.95 / g0.978 / g0.99 (pre-declared)
SELECTION_QUEUES = {0: ["g0.95"], 1: ["g0.978"], 2: ["g0.99"]}  # one configuration per scheduler job
FIT_SHOTS = 13_919  # eb.FitShots() (13 920 minus one typed-failure branch); checked by --smoke
ARMS = {"T3": {"size": "native", "role": "coverage (secondary, P-coverage / Tier-C; never in the program claim)",
               "program_set": False}}
ARM = "T3"
CHECKPOINT_EVERY = 2000
LOG_EVERY = 100
CHILD_CAP_GPU_HOURS = 12.0  # the upper end of the #127 coverage estimate for γ-models (P-coverage funds the projection)
SEED_CAP_GPU_HOURS = 24.0  # #127: native cap per seed
WALL_CAP_HOURS = 48.0  # on 4 GPUs
GRID = "grid"
ARM_CHECKS = ["G_rank", "G_action", "G_nonfinite", "G_trained"]  # no latent: G-collapse not applicable
TRAINING_PRECISION = {"dtype": "float32", "flags": "torch defaults (upstream float32)"}


def log(message):
    eb.log(ISSUE, message)


def device_of(requested=None):
    return requested or ("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------------------------------------------
# configurations and paths
# ---------------------------------------------------------------------------------------------------------------

def config_spec(name):
    config = gn.config_of(name)
    return {"name": name, "arm": ARM, "size": "native", "configuration": name, "config": config,
            "steps": config["n_steps"], "batch": config["batch_size"], "samples": gn.SAMPLES,
            "sample_seed": gn.SAMPLE_SEED}


def arm_spec(arm, selected):
    if arm != ARM:
        raise KeyError(arm)
    return config_spec(selected)


def selection_dir(output, name, seed=eb.SELECTION_SEED):
    return Path(output) / "selection" / name / f"seed-{seed}"


def model_dir(output, arm, seed, selected=None):
    """T3 seed 20260908 is the reused selection run of the selected configuration (#127 selection rule step 4)."""
    if arm == ARM and seed == eb.SELECTION_SEED:
        if selected is None:
            raise ValueError("model_dir(T3, 20260908) needs the selected configuration")
        return selection_dir(output, selected)
    return Path(output) / "models" / arm / f"seed-{seed}"


def readouts_dir(output, arm, seed):
    return Path(output) / "readouts" / arm / f"seed-{seed}"


def record_path(output, arm, seed):
    return Path(output) / "records" / f"{arm}--seed{seed}.json"


# ---------------------------------------------------------------------------------------------------------------
# fit transitions
# ---------------------------------------------------------------------------------------------------------------

def fit_store(shots, chunk=1024):
    """Every fit shot's states: carriers [N, 46, 236] float32 (frames 0, 5, ..., 225; positions past the end repeat
    the final carrier) and launch5 [N, 5], in the frozen shot order (CPU)."""
    carriers = []
    for start in range(0, len(shots.keys), chunk):
        index = range(start, min(len(shots.keys), start + chunk))
        pairs = [(s, p) for s in index for p in gn.episode_positions(shots.ends[s])]
        rows = shots.carrier_rows([s for s, _ in pairs], [p for _, p in pairs])
        carriers.append(torch.as_tensor(np.asarray(rows), dtype=torch.float32).reshape(
            len(index), len(gn.POSITIONS), gn.CARRIER_DIM))
    return {"carriers": torch.cat(carriers), "launch5": torch.as_tensor(shots.launch5, dtype=torch.float32)}


def training_batch(data, step, seed, spec, device="cpu"):
    """Batch ``step`` of a run: transition rows drawn from eb.batch_rng(seed, step) (uniform with replacement)."""
    shot, k = gn.transition_rows(eb.batch_rng(seed, step), len(data["launch5"]), spec["batch"])
    return gn.transition_batch(data["carriers"], data["launch5"], shot, k, device)


# ---------------------------------------------------------------------------------------------------------------
# training (resumable; nonfinite loss / sampling assertion -> typed terminal failure)
# ---------------------------------------------------------------------------------------------------------------

def train_run(out_dir, spec, seed, data, device=None, max_steps=None):
    """Train one γ-model (resumable from out_dir/resume.pt; the official update via gn.train_loop). Batch b =
    training_batch(data, b, seed, spec); the bootstrapped-target samples draw from the global torch generators seeded
    with ``seed`` (checkpointed)."""
    device = device_of(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "model.pt"
    if final.is_file():
        log(f"{spec['name']} seed {seed}: model.pt exists; skipped")
        return eb.read_json(out_dir / "report.json")
    steps = spec["steps"]
    torch.manual_seed(seed)
    trainer = gn.Trainer(spec["config"], device)
    try:
        out = gn.train_loop(trainer, out_dir, steps, lambda b: training_batch(data, b, seed, spec, device),
                            lambda: bool(eb.STOP_REQUESTED), max_steps=max_steps, log_every=LOG_EVERY,
                            checkpoint_every=CHECKPOINT_EVERY)
    except gn.TrainingPaused as paused:
        log(f"{spec['name']} seed {seed}: stop signal; {paused}")
        raise eb.TrainingPaused(f"{spec['name']} seed {seed} {paused}") from None
    if out["failure"] is None and out["stop"] < steps:  # max_steps reached (tests, throughput)
        return {"partial": True, "steps_done": out["steps_done"], "seconds": out["seconds"]}
    complete = out["failure"] is None
    precision = {**TRAINING_PRECISION, "device": device,
                 "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                 "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    model = trainer.model
    payload = {"schema": SCHEMA_MODEL, "spec": spec, "seed": seed, "complete": complete, "typed_failure": out["failure"],
               "state_dict": {k: v.cpu() for k, v in model.state_dict().items()} if complete else None,
               "actnorm": gn.actnorm_flags(model), "parameters": gn.parameter_count(model), "precision": precision}
    eb.atomic_torch(final, payload)
    seconds = out["elapsed"]
    report = {"spec": spec, "seed": seed, "steps": steps, "complete": complete, "typed_failure": out["failure"],
              "wall_seconds": seconds, "gpu_seconds": seconds if device.startswith("cuda") else 0.0,
              "device": device, "seconds_per_step": seconds / max(1, steps),
              "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None,
              "model_sha256": eb.sha256_of(final), "parameters": payload["parameters"],
              "final": out["trace"][-1] if out["trace"] else None, "precision": precision, "finished_at": eb.utc_now()}
    eb.write_json(out_dir / "report.json", report)
    if (out_dir / "resume.pt").is_file() and complete:
        (out_dir / "resume.pt").unlink()
    log(f"{spec['name']} seed {seed}: {'complete' if complete else out['failure']['typed_failure']} after "
        f"{seconds:.0f}s")
    return report


def load_trained(path, device=None):
    """(model, payload) of a model.pt, or (None, payload) for a typed failure."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA_MODEL:
        raise ValueError(f"{path}: schema differs")
    if not payload["complete"]:
        return None, payload
    model = gn.build_model(payload["spec"]["config"], device_of(device))
    model.load_state_dict(payload["state_dict"])
    gn.set_actnorm_flags(model, payload["actnorm"])
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
                report=str(Path(out_dir, "report.json").relative_to(ROOT)) if Path(out_dir).is_relative_to(ROOT)
                else str(Path(out_dir, "report.json")), complete=report["complete"])
    return 0


# ---------------------------------------------------------------------------------------------------------------
# prediction (occupancy samples), expected costs, policy AUC
# ---------------------------------------------------------------------------------------------------------------

def objective_costs():
    from scripts import run_lookahead_control as wlc
    return harness.EndpointCosts(wlc.load_objective())


def grid_actions():
    items = eb.grid_items()
    return items, torch.stack([eb.launch5(item["action"]) for item in items])


@torch.no_grad()
def candidate_samples(model, items, z0):
    """Occupancy samples [K, S, 236] (CPU) of a level's candidates from its anchor carrier z0 [236] and the
    candidates whose sampling raised the official assertion."""
    actions = torch.stack([eb.launch5(it["action"]) for it in items])
    return gn.sample_candidates(model, z0[None].expand(len(items), -1), actions, gn.base_noise())


@torch.no_grad()
def level_rows(model, items, z0, costs):
    """Rows of one level's candidates from its carrier anchor: the expected tie-free cost over the S occupancy samples
    (the primary scoring path; shared by policy AUCs, --score and --score-novelty)."""
    samples, failed = candidate_samples(model, items, z0)
    return gn.expected_rows(items, samples, z0, costs, failed)


@torch.no_grad()
def fit_anchor_readouts(model, costs):
    """480 fit anchors x 16 grid launches: the tie-free expected-cost matrix [L, 16] (excluded -> nan), the
    per-candidate sample means [L, 16, 236] (G-action) and the typed-row count."""
    anchors = eb.fit_anchors()["carriers"]
    members = sorted(anchors)
    items, _ = grid_actions()
    matrix = np.full((len(members), len(items)), np.nan)
    means = torch.empty(len(members), len(items), gn.CARRIER_DIM)
    typed = 0
    for i, member in enumerate(members):
        samples, failed = candidate_samples(model, items, anchors[member])
        for j, row in enumerate(gn.expected_rows(items, samples, anchors[member], costs, failed)):
            if row["excluded"]:
                typed += 1
            else:
                matrix[i, j] = row[eb.PRIMARY_COST]
        means[i] = gn.sample_means(samples)
    return members, matrix, means, typed


@torch.no_grad()
def policy_rows(model, costs):
    """{member: rows} of the policy grid mixed cells."""
    anchors = eb.policy_anchors()["carriers"]
    return {member: level_rows(model, entry["inventory"], anchors[member], costs)
            for member, entry in sorted(eb.policy_cells(GRID).items())}


# ---------------------------------------------------------------------------------------------------------------
# gates (#134)
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
    """Funding marker, G-port pass and G-fid pass (validation ladder: the pendulum fidelity first)."""
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


READOUT_HOURS_PER_ARM_SEED = 0.1  # G-rank / G-action on 480 x 16 x 64 samples + policy AUC (estimate)
SCORE_HOURS = 0.5  # one scoring pass of 3 arm-seeds incl. wrong anchors and novelty (estimate)


def child_projection(per_update_gpu_s, fidelity_gpu_h=0.0, readout_h=READOUT_HOURS_PER_ARM_SEED, score_h=SCORE_HOURS):
    """GPU-h of everything left from per-configuration GPU s/update: three selection runs, two more seeds at the
    selected configuration, readouts, scoring (the pendulum fidelity runs on CPU: 0 GPU-h)."""
    steps = {name: config_spec(name)["steps"] for name in CONFIGS}
    selection_s = {name: steps[name] * per_update_gpu_s[name] for name in CONFIGS}
    rate = max(per_update_gpu_s.values())
    main_steps = max(steps.values())
    training = [(f"{ARM}--seed{seed}", main_steps * rate) for seed in eb.SEEDS if seed != eb.SELECTION_SEED]
    readout_s = readout_h * 3600 * len(eb.SEEDS)
    score_s = score_h * 3600
    queues, loads = lpt(training)
    total = sum(selection_s.values()) + sum(s for _, s in training) + readout_s + score_s + fidelity_gpu_h * 3600
    wall = max(selection_s.values()) + max(loads) + readout_s / 4 + score_s + fidelity_gpu_h * 3600
    per_seed = main_steps * rate
    out = {"steps": steps, "selection_gpu_h": {k: v / 3600 for k, v in selection_s.items()},
           "training_gpu_h": sum(s for _, s in training) / 3600, "readouts_gpu_h_estimate": readout_s / 3600,
           "score_gpu_h_estimate": score_s / 3600, "fidelity_gpu_h": fidelity_gpu_h, "total_gpu_h": total / 3600,
           "child_cap_gpu_h": CHILD_CAP_GPU_HOURS, "native_per_seed_gpu_h": per_seed / 3600,
           "seed_cap_gpu_h": SEED_CAP_GPU_HOURS, "wall_h_on_4_gpus_serial_ladder": wall / 3600,
           "wall_cap_h": WALL_CAP_HOURS, "training_queues": queues, "training_queue_loads_h": [x / 3600 for x in loads]}
    out["within_caps"] = bool(total / 3600 <= CHILD_CAP_GPU_HOURS and per_seed / 3600 <= SEED_CAP_GPU_HOURS
                              and wall / 3600 <= WALL_CAP_HOURS)
    return out


def _synthetic_data(shots=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    carriers = torch.rand(shots, len(gn.POSITIONS), gn.CARRIER_DIM, generator=g)
    carriers[:, :, 118:] = 0.0
    return {"carriers": carriers, "launch5": torch.tensor([[-0.2, 0.1, 1.0, 0.0, 1.0]]).repeat(shots, 1)}


def cpu_seconds_per_update(configuration="g0.99", repeats=3):
    """CPU s/update of T3 (synthetic carrier-shaped transitions; native batch 1024), the burn-in update excluded."""
    data = _synthetic_data()
    spec = config_spec(configuration)
    torch.manual_seed(0)
    trainer = gn.Trainer(spec["config"], "cpu")
    trainer.train_step(*training_batch(data, 0, 0, spec), 0)
    began = time.monotonic()
    for step in range(1, repeats + 1):
        trainer.train_step(*training_batch(data, step, 0, spec), spec["config"]["n_burnin"] + step)
    return (time.monotonic() - began) / repeats


GPU_OVER_CPU = {"lower": 0.03, "upper": 0.3637298005808285}  # [INFERENCE] see project()


def project(output):
    """projection.json (CPU, before funding): when timing.json (GPU) exists, the measured projection (primary);
    always the [INFERENCE] CPU model: T3 CPU s/update scaled by a GPU/CPU ratio range — lower 0.03 (dense 10 M-parameter
    flow at batch 1024: ~100 GFLOP/update, launch-bound on a GPU at ~30 ms [INFERENCE]), upper 0.364 (#131's measured
    THICK ratio, as #133's upper model). The pendulum G-fid runs on CPU (0 GPU-h; fidelity/projection.json)."""
    output = Path(output)
    cpu = cpu_seconds_per_update()
    proj = {name: child_projection({c: cpu * ratio for c in CONFIGS}) for name, ratio in GPU_OVER_CPU.items()}
    fidelity_path = output / "fidelity" / "projection.json"
    result = {"schema": SCHEMA_PROJECTION, "plan_identity": IDENTITY,
              "label": "[INFERENCE] CPU micro-benchmark (synthetic carrier-shaped batches of 1024, 4 threads) scaled "
                       "by an assumed GPU/CPU per-update ratio range (lower 0.03: launch-bound dense flow; upper 0.364: "
                       "#131's measured THICK ratio)",
              "t3_cpu_seconds_per_update": cpu, "threads": torch.get_num_threads(), "gpu_over_cpu": GPU_OVER_CPU,
              "parameters": gn.parameter_count(gn.build_model()), "projection_lower": proj["lower"],
              "projection_upper": proj["upper"], "primary": "projection_upper",
              "fidelity": eb.read_json(fidelity_path) if fidelity_path.is_file() else None,
              "fidelity_note": "pendulum G-fid runs on CPU in daytime (0 GPU-h)", "measured_at": eb.utc_now()}
    timing_path = output / "timing.json"
    if timing_path.is_file():
        record = eb.read_json(timing_path)
        if str(record.get("device", "")).startswith("cuda"):
            result["projection_measured"] = measured_projection(record)
            result["timing_sha256"] = eb.sha256_of(timing_path)
            result["primary"] = "projection_measured"
    eb.write_json(output / "projection.json", result)
    log(f"projection [INFERENCE]: {proj['lower']['total_gpu_h']:.1f}-{proj['upper']['total_gpu_h']:.1f} GPU-h "
        f"(CPU {cpu:.3f} s/update); within caps {proj['lower']['within_caps']} / {proj['upper']['within_caps']}")
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


def timing(output, updates=500, device=None, data=None, decisions=8):
    """GPU timing probe (no funded.json; its own posted decision before enqueue; <= 0.5 GPU-h): ``updates`` official
    updates of the native T3 model (g0.99; every configuration's update costs the same) on the fit split (the runner's
    own batches; the first burn-in updates then post-burn-in ones), then the scoring cost of ``decisions`` fit anchors
    (16 candidates x 64 samples, expected costs). Writes timing.json and a ledger entry; keeps no model."""
    device = device_of(device)
    output = Path(output)
    began = time.monotonic()
    result = {"schema": SCHEMA_TIMING, "plan_identity": IDENTITY, "decision": "P-coverage timing probe (#134)",
              "device": device, "device_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
              "torch": torch.__version__, "updates": updates}
    t = time.monotonic()
    data = data if data is not None else fit_store(eb.FitShots())
    result["fit_store_seconds"] = time.monotonic() - t
    spec = config_spec("g0.99")
    torch.manual_seed(eb.SELECTION_SEED)
    trainer = gn.Trainer(spec["config"], device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    chunks, losses, batch_seconds, start = [], [], 0.0, time.monotonic()
    burnin = spec["config"]["n_burnin"]
    for step in range(updates):
        index = step if step < updates // 2 else burnin + step  # half burn-in, half post-burn-in updates
        t = time.monotonic()
        batch = training_batch(data, step, eb.SELECTION_SEED, spec, device)
        batch_seconds += time.monotonic() - t
        losses.append(float(trainer.train_step(*batch, index)["loss"]))
        if (step + 1) % 50 == 0 or step + 1 == updates:
            _sync(device)
            now = time.monotonic()
            chunks.append({"updates": (step % 50) + 1, "seconds": now - start})
            start = now
    steady = [c["seconds"] / c["updates"] for c in chunks[1:]] or [c["seconds"] / c["updates"] for c in chunks]
    result["t3"] = {"parameters": gn.parameter_count(trainer.model), "batch": spec["batch"], "chunks": chunks,
                    "seconds_per_update_steady_median": float(np.median(steady)),
                    "batch_build_seconds_per_update": batch_seconds / updates, "first_loss": losses[0],
                    "last_loss": losses[-1], "finite": bool(np.all(np.isfinite(losses)))}
    if device.startswith("cuda"):
        result["t3"].update(peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                            peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30)
    model = trainer.model.eval()
    trainer = data = None
    costs = objective_costs()
    anchors = eb.fit_anchors()["carriers"]
    members = sorted(anchors)[:decisions]
    items, _ = grid_actions()
    walls = []
    for member in members:
        _sync(device)
        t = time.monotonic()
        level_rows(model, items, anchors[member], costs)
        _sync(device)
        walls.append(time.monotonic() - t)
    result["scoring"] = {"decisions": len(members), "samples": gn.SAMPLES, "seconds_per_decision_median":
                         float(np.median(walls[1:] or walls)), "seconds": walls}
    result["host_rss_gib"], result["host_peak_rss_gib"] = _rss_gib()
    result["seconds"] = time.monotonic() - began
    result["written_at"] = eb.utc_now()
    eb.write_json(output / "timing.json", result)
    eb.mark(output, "timing", result["seconds"], result["seconds"] if device.startswith("cuda") else 0.0)
    log(f"timing: T3 {result['t3']['seconds_per_update_steady_median']:.4f} s/update, "
        f"{result['scoring']['seconds_per_decision_median']:.3f} s/decision ({result['seconds']:.0f} s total)")
    projection_path = output / "projection.json"
    if device.startswith("cuda") and projection_path.is_file():  # the measured projection becomes primary
        projection = eb.read_json(projection_path)
        projection.update(projection_measured=measured_projection(result), primary="projection_measured",
                          timing_sha256=eb.sha256_of(output / "timing.json"))
        eb.write_json(projection_path, projection)
    return 0


def measured_projection(timing_record, levels_per_pass=None):
    """The child projection from timing.json: measured steady-state s/update; readouts / scoring from the measured
    s/decision (480 fit anchors + the policy cells per arm-seed; scoring: the declared SCORE_HOURS upper estimate unless
    ``levels_per_pass`` is given)."""
    rate = timing_record["t3"]["seconds_per_update_steady_median"]
    per_decision = timing_record["scoring"]["seconds_per_decision_median"]
    readout_h = (480 + 200) * per_decision / 3600  # fit anchors + policy grid cells (<= 200) [INFERENCE]
    score_h = SCORE_HOURS if levels_per_pass is None else levels_per_pass * per_decision / 3600
    return {"child": child_projection({name: rate for name in CONFIGS}, 0.0, readout_h, score_h),
            "t3_seconds_per_update": rate, "seconds_per_decision": per_decision, "source": "timing.json (measured)"}


def smoke(output, device=None):
    device = device_of(device)
    output = Path(output)
    shots = eb.FitShots()
    selected = selected_config(output)
    selection_record = eb.read_json(output / "selection.json")
    launch = torch.as_tensor(shots.launch5)
    accounting = {"fit_shots": len(shots), "fit_shots_expected": FIT_SHOTS, "fit_levels": len(shots.levels),
                  "fit_positive": int(shots.verdicts.sum()), "fit_anchors": len(eb.fit_anchors()["carriers"]),
                  "policy_grid_mixed_cells": len(eb.policy_cells(GRID)), "episode_states": len(gn.POSITIONS),
                  "transitions": len(shots) * gn.TRANSITIONS_PER_SHOT,
                  "shots_ending_before_225": int((np.asarray(shots.ends) < gn.END_FRAME).sum()),
                  "launch5_abs_max": float(launch.abs().max())}
    gport = eb.read_json(output / "gport/gport.json")
    adapter = eb.sha256_of(ROOT / "world_model/baselines/gamma_models/novphy.py").split(":")[-1]
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
                            "gpu_seconds": report["gpu_seconds"]}
    gfid = eb.read_json(output / "fidelity/gfid.json")
    remaining = remaining_items(output, selected)
    rate = throughput[selected]["seconds_per_step"]
    training = [(f"{arm}--seed{seed}", spec["steps"] * rate) for arm, seed, spec, _ in remaining]
    selection_seconds = sum(t["gpu_seconds"] for t in throughput.values())
    readout_seconds = READOUT_HOURS_PER_ARM_SEED * 3600 * len(eb.SEEDS)
    score_seconds = SCORE_HOURS * 3600
    queues, loads = lpt(training)
    total = selection_seconds + sum(s for _, s in training) + readout_seconds + score_seconds
    per_seed = arm_spec(ARM, selected)["steps"] * rate
    wall = max(loads) + readout_seconds / 4 + score_seconds
    projection = {"fidelity_gpu_h_spent": gfid.get("compute", {}).get("gpu_seconds", 0.0) / 3600,
                  "fidelity_cpu_h_spent": gfid.get("compute", {}).get("cpu_seconds_wall", 0.0) / 3600,
                  "selection_gpu_h": selection_seconds / 3600, "training_gpu_h": sum(s for _, s in training) / 3600,
                  "readouts_gpu_h_estimate": readout_seconds / 3600, "score_gpu_h_estimate": score_seconds / 3600,
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

OWN_FILES = [RUNNER, TESTS, "world_model/baselines/gamma_models/__init__.py",
             "world_model/baselines/gamma_models/novphy.py", "scripts/b127_gamma_models_port_reference.py",
             "scripts/b127_gamma_models_fidelity.py"]


def own_pins(output):
    output = Path(output)
    paths = [ROOT / p for p in OWN_FILES]
    paths += sorted(p for p in (ROOT / "world_model/baselines/gamma_models/vendor").rglob("*") if p.is_file())
    paths += [output / "gport/gport.json", output / "fidelity/band.json", output / "fidelity/gfid.json",
              output / "fidelity/digitised/fig4.json", output / "fidelity/digitised/fig4-maps.npz",
              ROOT / "data/external/gamma_models/MANIFEST.json",
              output / "smoke.json", output / "selection.json", output / "funded.json"]  # decisions.md: append-only
    paths += [selection_dir(output, name) / f for name in CONFIGS for f in ("model.pt", "report.json")]
    return [{"name": f"t3_{p.relative_to(ROOT)}", "artifact": str(p.relative_to(ROOT)), "sha256": eb.sha256_of(p)}
            for p in paths if p.is_file() and "__pycache__" not in str(p)]


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
        "readout": {"score": f"expected tie-free EndpointCosts cost over S = {gn.SAMPLES} samples of p_γ(· | s0, a) "
                             f"(flow inverse of one fixed uniform draw, seed {gn.SAMPLE_SEED}, shared by every "
                             "candidate); a sampling assertion or any nonfinite sample -> excluded typed row",
                    "action_sensitivity": "eb.action_sensitivity on the per-candidate sample-mean carriers, 480 fit "
                                          "anchors x 16 grid launches",
                    "collapse": "not applicable (no latent: the flow is a bijection of the carrier)",
                    "positions": "not applicable (the occupancy has no timestep)"},
        "queues": {"train": {str(k): v for k, v in queues.items()},
                   "readouts": {str(k): v for k, v in readout_queues.items()}, "score": "GPU 0, one process"},
        "seeds": list(eb.SEEDS), "guards": eb.GUARD, "arm_guards": ARM_CHECKS, "extra_guards": ["G_port", "G_fid"],
        "global_guards": ["G_isolation", "G_chronology", "G_scoring_policy", "G_records"],
        "contrasts": {"per_arm": list(eb.CONTRAST_REFERENCES), "within_family": [], "wrong_anchor": list(ARMS)},
        "dispositions": eb.DISPOSITION_LABELS, "delta": eb.DELTA, "decisions": DECISIONS,
        "disclosures": [
            "T3 is a coverage arm (P-coverage; Tier-C if triggered): secondary, reported, never in the program claim.",
            "T3's γ was selected on the policy split among three pre-declared values (0.95, 1 - 1/45, 0.99); U had no "
            "such tuning (#127).",
            "Training follows the official Colab notebook (50 000 updates of 1024 transitions); upstream trains on a "
            "201 000-transition pendulum pool, here 640 274 fit transitions (13 919 shots x 46).",
            "The frame-225 carrier is absorbing (s_46 := s_45): the occupancy's tail mass γ^44 sits on it.",
            "The score is a Monte-Carlo expectation over 64 samples with common random numbers; the paper's value "
            "estimate is the same expectation of a reward (Eq. 6).",
            "No native success head (no reward model), no U + head control, no G-collapse (no latent), no position "
            "error (no timestep).",
            "Scoring waits for G-fid pass (pendulum Fig. 4); a G-fid failure drops T3 (fidelity_not_reproduced).",
            "P-novelty: after the primary scoring pass, T3 (if its guards pass) is scored on the #110 novelty levels "
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
        result["model"] = str(path.relative_to(ROOT))
        result["model_sha256"] = eb.sha256_of(path)
        if model is None:
            result.update(typed_failure=payload["typed_failure"]["typed_failure"])
        else:
            members, matrix, means, typed = fit_anchor_readouts(model, costs)
            items, actions = grid_actions()
            result.update(typed_failure=None, G_rank=eb.g_rank(matrix), G_action=float(eb.action_sensitivity(means)),
                          G_collapse=None, fit_anchor_nonfinite_share=float(np.isnan(matrix).mean()),
                          fit_anchor_typed_rows=typed, trainable_parameters=gn.parameter_count(model))
            prow = policy_rows(model, costs)
            result["policy_auc"] = eb.policy_grid_auc(prow, eb.PRIMARY_COST)
            anchors = eb.fit_anchors()["carriers"]
            z0 = anchors[members[0]][None].expand(len(items), -1)
            result["macs_per_candidate"] = eb.macs_per_candidate(
                lambda a, b: gn.sample_occupancy(model, a, b, gn.base_noise()), z0, actions, candidates=len(items))
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
        entries[f"{arm}--seed{seed}"] = {"readouts": str((rdir / "readouts.json").relative_to(ROOT)),
                                         "readouts_sha256": eb.sha256_of(rdir / "readouts.json"),
                                         "model": str(mpath.relative_to(ROOT)), "model_sha256": eb.sha256_of(mpath)}
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

@torch.no_grad()
def score_arm(output, plan, arm, seed, anchors, cells, device=None):
    """One record: rows of every mixed cell of every inventory, wrong-anchor rows (480 fit anchors, grid), wall time per
    decision (position error: not applicable)."""
    device = device_of(device)
    costs = objective_costs()
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "arm": arm, "seed": seed}
    model, payload = load_trained(model_dir(output, arm, seed, plan["selected"]) / "model.pt", device)
    if model is None:
        return {**base, "typed_failure": payload["typed_failure"]["typed_failure"]}
    carriers = anchors["carriers"]
    inventories = {}
    for inventory in eb.INVENTORIES:
        inventories[inventory] = {member: {"rows": level_rows(model, entry["inventory"], carriers[member], costs)}
                                  for member, entry in sorted(cells[inventory]["mixed"].items())}
    record = {**base, "typed_failure": None, "inventories": inventories}
    fit = eb.fit_anchors()["carriers"]
    items, _ = grid_actions()
    record["wrong_anchor"] = {m: {"rows": level_rows(model, items, fit[m], costs)} for m in sorted(fit)}
    record["positions"] = None
    members = sorted(cells[GRID]["mixed"])[:20]
    record["wall_per_decision"] = eb.wall_per_decision(lambda m: level_rows(model, items, carriers[m], costs), members)
    return record


def score(output, device=None):
    output = Path(output)
    plan = load_plan(output)
    load_manifest(output)
    if not gfid_pass(output):
        log("score refused: fidelity/gfid.json has not passed (#134 gate); evaluation not read")
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

SCHEMA_NOVELTY_RECORD = "issue_134_novelty_record_v1"
NOVELTY_ARMS = tuple(ARMS)


def novelty_dir(output):
    return Path(output) / "novelty"


def novelty_record_path(output, arm, seed):
    return novelty_dir(output) / "records" / f"{arm}--seed{seed}.json"


def score_novelty(output, device=None, cells=None):
    """P-novelty: every guard-passing arm, every seed, on the mixed novelty levels (grid) from each level's carrier
    anchor (the primary scoring path, level_rows); refuses before the primary records and the readouts freeze exist;
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
        eb.write_json(novelty_dir(output) / "arms.json", {"schema": "issue_134_novelty_arms_v1", "arms": status,
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
                grid = {level: {"rows": level_rows(model, cells["levels"][level]["inventory"],
                                                   anchors["carriers"][level], costs)}
                        for level in sorted(cells["mixed"])}
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
    gfid_path = output / "fidelity/gfid.json"
    gfid_block = {ARM: {"pass": extra_guards["G_fid"], "record": str(gfid_path.relative_to(ROOT)),
                        "sha256": eb.sha256_of(gfid_path) if gfid_path.is_file() else None}}
    ledger, gpu_total, wall_total = eb.ledger_totals(output, training_report_of(output, plan))
    summary = eb.child_summary(ISSUE, IDENTITY, plan["frozen_at"], blocks, {}, guards_global, gfid_block, {
        "inventories": tables, "wrong_anchor": wrong,
        "positions": "not applicable: a γ-model predicts a discounted state occupancy without timesteps",
        "selection": plan["selected"], "coverage_arm": True,
        "readouts": {f"{a}--seed{s}": {k: v for k, v in eb.read_json(readouts_dir(output, a, s) / "readouts.json").items()
                                       if k in ("policy_auc", "G_action", "fit_anchor_typed_rows", "typed_failure")}
                     for a, s in readout_items()},
        "novelty": novelty_section(output, records),
        "native_success_head": None, "u_plus_head": "not applicable: no reward model (task-free occupancy model)"})
    compute = {"schema": "issue_134_compute_v1", "plan_identity": IDENTITY, "ledger": ledger,
               "gpu_seconds_total": gpu_total, "wall_seconds_total": wall_total,
               "arms": {arm: blocks[arm]["compute"] for arm in ARMS}, "reference": eb.reference_compute()}
    return summary, compute


def findings_text(summary):
    lines = [f"# #{ISSUE} T3 γ-models (coverage arm): findings", "",
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
    lines += ["", "T3 is a coverage arm (P-coverage): never in the program claim. Intervals are descriptive "
              "(member-clustered bootstrap); tokens follow the #127 delta rule.", ""]
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
    parser.add_argument("--timing-updates", type=int, default=500, help="--timing: updates")
    args = parser.parse_args(argv)
    output = Path(args.output)
    torch.set_num_threads(int(os.environ.get("B127_T3_CPU_THREADS", "4")))  # shared host: avoid oversubscription
    if args.timing:
        return timing(output, args.timing_updates, device=args.device)
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
