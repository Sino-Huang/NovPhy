"""Issue #140 (#127 child U-native): U trained to A4's training GPU-h, scored as U's Gate-B records.

The #127 budget safeguard fired for A4 (#128: A4 - U supported at 2 434.73 GPU-s per seed, 1.88 x U's B1 training
compute of 1 294.19 GPU-s). This child is the U-native control (Decision P-adapter, 2026-10-29, on #140):

- **U-native** (role control, not in the program set): the Gate-B unit F-15-macro (arm LC, family hybrid, fold
  all) trained by the frozen #112 v3 retrain code. A fresh copy of ``rt.implementation()`` runs the v1
  ``train_group`` unchanged. It uses the same data (the retrain's #109 fit payload), objective, optimiser and
  hyperparameters. Only four things change, each a step count scaled by f = 2435.0 / 1294.19:
  - updates S' = 9 x round(1000 f) = 16 929 (whole 9-request cycles);
  - the long-horizon curriculum ``every`` 500 -> 941;
  - the ranking ramp ``start`` 1000 -> 1881;
  - the ranking ramp ``length`` 1000 -> 1881.

  Seeds 20260908-10. No selection.
- **Scoring**: U's own Gate-B path, ``gb.score_record``, with the U-native checkpoint substituted for the handoff
  checkpoint. It writes every mixed cell of every inventory from its Gate-B anchor and the wrong-anchor rows from
  the 480 fit anchors. Position error comes from ``rc.score_reference_curves`` (U's #123 curve rule) and is
  descriptive. Validation follows ``gb.spot_checks`` (``harness.endpoint_replication``).
- **Contrasts**: the standard X - U / - C* / - P / - J_dec rows; ``extra.u_native``: **A4 - U-native** on #128's
  published A4 records (both directions, guarded) and U-native - U (descriptive).

Modes (round3/child-runner-contract.md, the ones that apply): --gport (CPU), --prepare, --train --gpu g,
--readouts --gpu g, --freeze-readouts, --score, --publish, --validate. GPU work only through the night-window
scheduler (scripts/gpu_window_issue_140.json); tests use CUDA_VISIBLE_DEVICES="".
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import json
import math
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import external_baselines as eb  # noqa: E402
from scripts import run_decision_dynamics_retrain as rt  # noqa: E402
from scripts import run_decision_relevant_dynamics as v1  # noqa: E402
from scripts import run_gate_b as gb  # noqa: E402
from scripts import run_lookahead_control as wlc  # noqa: E402
from scripts import run_slot_encoder_fix as s99  # noqa: E402
from scripts import run_tau_ad_within_checkpoint as t96  # noqa: E402
from world_model.planning import scoring_harness as harness  # noqa: E402

ISSUE = 140
IDENTITY = "issue-140-u-native-v1"
OUTPUT = ROOT / ".local-artifacts" / IDENTITY
RUNNER = "scripts/run_issue_140_u_native.py"
TESTS = "tests/test_issue_140_u_native.py"
SCHEMA_PLAN = "issue_140_plan_v1"
SCHEMA_PREDICTOR = "issue_140_predictor_v1"
SCHEMA_GROUP = "issue_140_training_group_v1"
SCHEMA_UPDATES = "issue_140_updates_v1"
SCHEMA_REPORT = "issue_140_training_report_v1"
SCHEMA_READOUTS = "issue_140_readouts_v1"
SCHEMA_READOUTS_MANIFEST = "issue_140_readouts_manifest_v1"
SCHEMA_RECORD = "issue_140_record_v1"
SCHEMA_GPORT = "issue_127_gport_v1"
SCHEMA_COMPUTE = "issue_140_compute_v1"
DECISION = "https://github.com/Sino-Huang/NovPhy/issues/140#issuecomment-6479886189"
PROGRAM_DECISION = "https://github.com/Sino-Huang/NovPhy/issues/127#issuecomment-6479617332"

VIEW = "U-native"
ROLE = "control"
UNIT = "unit"  # gb.UNITS key of the Gate-B unit
FAMILY = gb.UNITS[UNIT]["family"]  # hybrid
REQUEST = gb.UNITS[UNIT]["request"]  # F-15-macro
ARM = rt.ARM  # LC
GRID = "grid"

# Budget scaling (Decision P-adapter): #128's published A4 mean per-seed training GPU-s over eb's U reference.
A4_GPU_S = 2435.0
U_GPU_S = 1294.19
FACTOR = A4_GPU_S / U_GPU_S
U_STEPS = rt.STEPS  # 9000
REQUEST_CYCLE = 9  # hybrid request of update k: t96.FIXED_ARMS[k % 9]


def scaled(count):
    """round(count x FACTOR), half up (no constant here sits on a half)."""
    return int(math.floor(count * FACTOR + 0.5))


STEPS = REQUEST_CYCLE * scaled(U_STEPS // REQUEST_CYCLE)  # 9 x 1881 = 16 929
SKIP_CAP = scaled(rt.SKIP_CAP)  # G2 (descriptive, as in the retrain): 45 -> 85
GPORT_UPDATES = 18  # two request cycles; the schedules first differ at update 500

TRAIN_QUEUES = {2: [20260908, 20260910], 3: [20260909]}  # physical GPU index -> seeds (GPU 0 carries #137)
READOUT_QUEUES = {3: list(eb.SEEDS)}
ARM_GUARDS = ["G_rank", "G_action", "G_nonfinite", "G_trained"]
EXTRA_GUARDS = ["G_port"]
GLOBAL_GUARDS = ["G_isolation", "G_chronology", "G_scoring_policy", "G_records"]
A128 = ROOT / ".local-artifacts" / "issue-128-direct-predictors-v1"
A4 = "A4"
RECORD_PATTERNS = ["records/*.json"]
CODE_FILES = ["scripts/run_decision_dynamics_versions.py", "scripts/run_request_interference.py",
              "scripts/run_lookahead_control.py", "world_model/training/decision_dynamics.py",
              "world_model/training/stable_decision_dynamics.py", "world_model/training/cnn_hybrid.py",
              "world_model/model/__init__.py"]


def log(message):
    eb.log(ISSUE, message)


# ---------------------------------------------------------------------------------------------------------------
# the U-native implementation: a fresh copy of the #112 v3 retrain implementation, budget-scaled
# ---------------------------------------------------------------------------------------------------------------

_IMPLEMENTATIONS = []


def schedule_constants(reference=None):
    """{name: (U's value, U-native's value)} of every step-indexed constant of the frozen train_group."""
    reference = reference or rt.implementation()
    return {"long_every": (reference.LONG["every"], scaled(reference.LONG["every"])),
            "rank_start": (reference.RANK["start"], scaled(reference.RANK["start"])),
            "rank_ramp": (reference.RANK["ramp"], scaled(reference.RANK["ramp"]))}


@functools.lru_cache(maxsize=None)
def implementation(steps=STEPS):
    """``rt.implementation()`` re-executed into a new module, so the retrain's own cached copy is untouched. On
    top of it: this child's output, identity and schemas; ``steps`` updates (through ``recipe()``, which
    ``train_group`` and ``load_member`` read); the stretched schedule constants; and a progress-checkpoint hook
    that raises TrainingPaused right after the progress state a stop signal requested has been written."""
    impl = rt.implementation.__wrapped__()
    constants = schedule_constants()
    impl.OUTPUT = OUTPUT
    impl.DEVICE = v1.DEVICE  # "cuda"; "cpu" when created inside on_device("cpu") (restored on exit)
    impl.IDENTITY = IDENTITY
    impl.SCHEMA_PREDICTOR = SCHEMA_PREDICTOR
    impl.SCHEMA_GROUP = SCHEMA_GROUP
    impl.RUNNER = RUNNER
    impl.LONG = {**impl.LONG, "every": constants["long_every"][1]}
    impl.RANK = {**impl.RANK, "start": constants["rank_start"][1], "ramp": constants["rank_ramp"][1]}
    impl.recipe = lambda: {**s99.training_recipe(), "steps": steps}
    write_state = impl.atomic_torch

    def atomic_torch(path, payload):
        write_state(path, payload)
        if eb.STOP_REQUESTED and Path(path).name.endswith(".progress.pt"):
            raise eb.TrainingPaused(f"stop signal: progress written at update {payload['step']}")

    impl.atomic_torch = atomic_torch
    _IMPLEMENTATIONS.append(impl)
    return impl


def schedule(impl, step):
    """(long horizon, ranking weight) of update `step` under an implementation's constants (train_group's rule)."""
    horizon = impl.dd.horizon_at(step, first=impl.LONG["first"], last=impl.LONG["last"],
                                 increment=impl.LONG["increment"], every=impl.LONG["every"])
    weight = impl.RANK["weight"] * impl.dd.ramp_at(step, start=impl.RANK["start"], length=impl.RANK["ramp"])
    return horizon, weight


def schedule_summary(native, reference, steps):
    first_difference = next((k for k in range(steps) if schedule(native, k) != schedule(reference, k)), None)
    full = next(k for k in range(steps) if schedule(native, k)[0] == native.LONG["last"])
    ramp_on = next(k for k in range(steps) if schedule(native, k)[1] > 0)
    ramp_full = next(k for k in range(steps) if schedule(native, k)[1] == native.RANK["weight"])
    return {"first_update_differing_from_U": first_difference, "horizon_reaches_225_at": full,
            "horizon_225_fraction": full / steps, "rank_weight_positive_from": ramp_on,
            "rank_weight_full_from": ramp_full}


def request_pause(signum, frame):
    """SIGTERM / SIGUSR1: record the stop (eb) and make train_group write its progress state after the current
    update; the progress hook then raises TrainingPaused (exit 75)."""
    eb.request_stop(signum, frame)
    for impl in _IMPLEMENTATIONS:
        impl.PROGRESS_EVERY = 1


def install_stop_handlers():
    for signum in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, request_pause)


@contextlib.contextmanager
def on_device(device, *extra):
    """CPU runs (G-port, tests): the frozen code hard-codes DEVICE = "cuda" and calls torch.cuda.synchronize; point
    every module's DEVICE at the CPU, make synchronize a no-op and run torch.compile'd functions eagerly. A GPU
    run changes nothing."""
    if device != "cpu":
        yield device
        return
    modules = [v1, rt, gb, eb.m99, rt.implementation(), *_IMPLEMENTATIONS, *extra]
    saved = [(m, m.DEVICE) for m in modules]
    synchronize, disabled = torch.cuda.synchronize, torch._dynamo.config.disable
    for m in modules:
        m.DEVICE = "cpu"
    torch.cuda.synchronize = lambda *args, **kwargs: None
    torch._dynamo.config.disable = True
    try:
        yield device
    finally:
        for m, value in saved:
            m.DEVICE = value
        for m in _IMPLEMENTATIONS:  # created inside the context
            if all(m is not s for s, _ in saved):
                m.DEVICE = v1.DEVICE
        torch.cuda.synchronize = synchronize
        torch._dynamo.config.disable = disabled


def device_of(device=None):
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------------------------------------------

def seed_index(seed):
    return list(eb.SEEDS).index(seed)


def predictor_path(output, seed, steps=STEPS):
    return implementation(steps).member_path(output, ARM, FAMILY, "all", seed)


def group_report_path(output, seed):
    return v1.group_path(output, ARM, FAMILY, seed_index(seed))


def updates_path(output, seed):
    return rt.updates_path(output, FAMILY, seed_index(seed))


def report_path(output, seed):
    return Path(output) / "runs" / f"{VIEW}--seed{seed}" / "report.json"


def readouts_path(output, seed):
    return Path(output) / "runs" / f"{VIEW}--seed{seed}" / "readouts.json"


def record_path(output, seed):
    return Path(output) / "records" / f"{VIEW}--seed{seed}.json"


def train_phase(seed):
    return f"train-{VIEW}-seed{seed}"


def rel(path):
    return str(Path(path).relative_to(ROOT))


# ---------------------------------------------------------------------------------------------------------------
# --gport (CPU): the U-native code reproduces U's B1 retrain code bit-exactly for the first updates
# ---------------------------------------------------------------------------------------------------------------

def _tensor_deltas(first, second):
    out = {}
    for name in first:
        a, b = first[name].double(), second[name].double()
        out[name] = {"equal": bool(torch.equal(first[name], second[name])),
                     "max_abs_delta": float((a - b).abs().max()) if a.numel() else 0.0}
    return out


def _optimizer_tensors(ensemble):
    state = ensemble.optimizer.state_dict()["state"]
    return {f"{index}.{key}": value.detach().cpu() for index, entry in sorted(state.items())
            for key, value in sorted(entry.items()) if torch.is_tensor(value)}


def gport_compare(updates=GPORT_UPDATES, seed=eb.SELECTION_SEED, native=None, data=None):
    """U's B1 retrain code (``rt.implementation()``, the code that trained the Gate-B checkpoints) vs the U-native
    implementation: both run ``train_group`` for `updates` updates of seed `seed` on the retrain's fit payload, on
    the CPU, eagerly, without saving. Parameters, AdamW state, every update's #77 local loss and pre-clip gradient
    norm, and the skip counts must be bitwise equal."""
    native = native or implementation(STEPS)
    reference = rt.implementation()
    with on_device("cpu", native):
        if data is None:
            data = rt.IndexedTrainData(rt.load_training_data(rt.OUTPUT), FAMILY)
        scratch = OUTPUT / "gport" / "scratch"
        _, ours = native.train_group(scratch, ARM, FAMILY, 0, [("all", seed)], data, steps=updates, save=False)
        _, theirs = reference.train_group(scratch, ARM, FAMILY, 0, [("all", seed)], data, steps=updates, save=False)
    params = _tensor_deltas({k: v.detach().cpu() for k, v in theirs.params.items()},
                            {k: v.detach().cpu() for k, v in ours.params.items()})
    moments = _tensor_deltas(_optimizer_tensors(theirs), _optimizer_tensors(ours))
    logs = _tensor_deltas(theirs.log_tensors(), ours.log_tensors())
    same_code = native.train_group.__code__.co_code == reference.train_group.__code__.co_code
    checks = {"parameters_equal": all(d["equal"] for d in params.values()),
              "adamw_state_equal": bool(moments) and all(d["equal"] for d in moments.values()),
              "per_update_logs_equal": all(d["equal"] for d in logs.values()),
              "skipped_equal": bool(torch.equal(theirs.skipped.cpu(), ours.skipped.cpu())),
              "logged_updates": int(len(ours.log["local"])) == updates,
              "same_train_group_bytecode": bool(same_code)}
    return {"updates": updates, "seed": seed, "checks": checks,
            "max_abs_delta": {"parameters": max(d["max_abs_delta"] for d in params.values()),
                              "adamw_state": max(d["max_abs_delta"] for d in moments.values()),
                              "logs": max(d["max_abs_delta"] for d in logs.values())},
            "pass": all(checks.values())}


def gport(output):
    output = Path(output)
    began = time.monotonic()
    result = gport_compare()
    native, reference = implementation(STEPS), rt.implementation()
    record = {
        "schema": SCHEMA_GPORT, "issue": ISSUE, "plan_identity": IDENTITY, "written_at": eb.utc_now(),
        "arms": {VIEW: {**result, "tolerance": "bitwise (torch.equal)"}},
        "G_port": bool(result["pass"]), "G_port_rule": "bitwise_first_updates_same_code_path",
        "rule": (f"U's B1 retrain code (rt.implementation().train_group) and the U-native implementation run the "
                 f"first {GPORT_UPDATES} updates of seed {eb.SELECTION_SEED} on the retrain's #109 fit payload "
                 "(CPU, eager: torch.compile is a kernel transformation, not a code-path change); parameters, AdamW "
                 "state, per-update local loss / gradient norm and skip counts are bitwise equal, and both "
                 "train_group functions have identical bytecode"),
        "schedule": {"constants": {k: list(v) for k, v in schedule_constants().items()},
                     **schedule_summary(native, reference, STEPS),
                     "u": schedule_summary(reference, reference, U_STEPS)},
        "training_data": {"path": rel(rt.data_path(rt.OUTPUT)), "pool_sizes": eb.read_json(
            v1.group_path(rt.OUTPUT, ARM, FAMILY, 0))["pool_sizes"][0]},
        "seconds": time.monotonic() - began,
    }
    eb.write_json(output / "gport" / "gport.json", record)
    log(f"G-port {'pass' if record['G_port'] else 'FAIL'}: {json.dumps(result['checks'])}")
    return 0 if record["G_port"] else 1


# ---------------------------------------------------------------------------------------------------------------
# --prepare (freeze) and the plan
# ---------------------------------------------------------------------------------------------------------------

def u_reference():
    """eb's U reference: the #112 v3 retrain training-group reports (seed -> GPU-s, steps)."""
    seeds = eb._reference_training(UNIT)
    return {"per_seed": {s: e["training_gpu_seconds"] for s, e in seeds.items()},
            "mean": float(np.mean([e["training_gpu_seconds"] for e in seeds.values()])),
            "steps": sorted({e["steps"] for e in seeds.values()}),
            "parameters_per_member": sorted({e["parameters_per_member"] for e in seeds.values()})}


def a4_record_paths():
    return {seed: A128 / "records" / f"{A4}--seed{seed}.json" for seed in eb.SEEDS}


def own_pins(output):
    output = Path(output)
    paths = [ROOT / RUNNER, ROOT / TESTS, output / "gport" / "gport.json", *[ROOT / p for p in CODE_FILES],
             rt.OUTPUT / "plan.json", rt.OUTPUT / "compute.json", s99.DYNAMICS / "plan.json",
             *a4_record_paths().values()]
    return [{"name": f"u_native_{rel(p)}", "artifact": rel(p), "sha256": eb.sha256_of(p)} for p in paths]


def projection():
    """Training GPU-s from U's measured rate (the retrain reports) x S'; scoring / readouts from Gate-B's ledger."""
    u = u_reference()
    per_update = u["mean"] / U_STEPS
    per_seed = per_update * STEPS
    return {"u_seconds_per_update": per_update, "steps": STEPS, "training_gpu_s_per_seed": per_seed,
            "training_gpu_h": 3 * per_seed / 3600, "training_wall_h_on_queues": {
                str(g): len(seeds) * per_seed / 3600 for g, seeds in TRAIN_QUEUES.items()},
            "readouts_score_validate_gpu_h": 0.5}


def prepare(output):
    output = Path(output)
    path = output / "plan.json"
    if path.is_file():
        load_plan(output)
        log("plan.json is frozen; pins re-verified")
        return 0
    gport_record = eb.read_json(output / "gport" / "gport.json")
    if not gport_record["G_port"] or gport_record["arms"][VIEW]["updates"] != GPORT_UPDATES:
        raise SystemExit("G-port did not pass; no freeze")
    handoff = eb.read_json(gb.HANDOFF)
    u = u_reference()
    a128 = eb.read_json(A128 / "summary.json")["arms"][A4]
    reference = rt.implementation()
    recipe = s99.training_recipe()
    plan = {
        "schema": SCHEMA_PLAN, "plan_identity": IDENTITY, "issue": ISSUE, "program_issue": eb.PROGRAM_ISSUE,
        "frozen_at": eb.utc_now(), "pins": eb.shared_inputs() + own_pins(output),
        "decision": DECISION, "program_decision": PROGRAM_DECISION,
        "arms": {VIEW: {"role": ROLE, "program_set": False, "unit": {"name": UNIT, **gb.UNITS[UNIT], "arm": ARM,
                                                                     "fold": "all"},
                        "seeds": list(eb.SEEDS), "primary_cost_key": eb.PRIMARY_COST}},
        "recipe": {
            "code": ("a fresh copy of scripts.run_decision_dynamics_retrain.implementation() (the frozen v3 "
                     "train_group of run_decision_relevant_dynamics, SkippingEnsemble patience 10, truncation 45, "
                     "logging ensemble); output / identity / schemas / steps / three schedule constants differ"),
            "u_steps": U_STEPS, "steps": STEPS, "update_ratio": STEPS / U_STEPS,
            "factor": {"a4_gpu_s": A4_GPU_S, "u_gpu_s": U_GPU_S, "value": FACTOR,
                       "rule": "S' = 9 x round(1000 x f) (whole request cycles); constant c -> round(c x f)"},
            "schedule": {k: {"u": a, "u_native": b} for k, (a, b) in schedule_constants(reference).items()},
            "kept": {"request_cycle": REQUEST_CYCLE, "trace_every": reference.TRACE_EVERY,
                     "progress_every": reference.PROGRESS_EVERY, "long": reference.LONG, "rank": reference.RANK},
            "optimiser": {k: recipe[k] for k in ("batch_size", "learning_rate", "weight_decay", "grad_clip")},
            "skip_cap_scaled": SKIP_CAP,
            "end_state": {"rule": "rt.excursion (#117) over the full update log; usable iff complete and no excursion",
                          "ratio": rt.EXCURSION_RATIO, "last": rt.EXCURSION_LAST, "spike_norm": rt.SPIKE_NORM,
                          "spike_window": rt.SPIKE_WINDOW},
        },
        "training_data": {"path": rel(rt.data_path(rt.OUTPUT)), "sha256": handoff["training_data"]["sha256"],
                          "source": "the #112 v3 retrain Gate-B handoff (training_data.sha256)"},
        "u_reference": u,
        "a4_reference": {"training_gpu_seconds": a128["compute"]["training_gpu_seconds"],
                         "mean": float(np.mean(list(a128["compute"]["training_gpu_seconds"].values()))),
                         "records": {str(s): rel(p) for s, p in a4_record_paths().items()},
                         "primary_cost_key": a128["primary_cost_key"]},
        "queues": {"train": {str(g): s for g, s in TRAIN_QUEUES.items()},
                   "readouts": {str(g): s for g, s in READOUT_QUEUES.items()}, "score": "GPU 0, one process"},
        "projection": projection(), "seeds": list(eb.SEEDS), "guards": eb.GUARD, "arm_guards": ARM_GUARDS,
        "extra_guards": EXTRA_GUARDS, "global_guards": GLOBAL_GUARDS, "skip_cap": SKIP_CAP,
        "contrasts": {"per_arm": list(eb.CONTRAST_REFERENCES), "extra.u_native": [
            "A4 - U-native (token and reverse token; guarded by both arms' guards, both children's global guards and "
            "#128's G-records; A4 records hash-checked against #128's records manifest)",
            "U-native - U (descriptive)"], "wrong_anchor": [VIEW]},
        "dispositions": eb.DISPOSITION_LABELS, "delta": eb.DELTA,
        "reading": ("A4 - U-native supported -> A4's budget qualification is dropped (an architectural reading is "
                    "permitted); otherwise A4's wording stays budget-qualified; reported beside #128's table"),
        "disclosures": [
            "No selection: U's configuration is fixed by Gate-B; only the update count and the step-indexed "
            "schedules change.",
            "U-native's realised training GPU-s differs from A4's 2 434.73 (measured, published per seed with "
            "both ratios).",
            "G-port runs on the CPU with torch.compile disabled in both runs; GPU kernels are not bitwise "
            "deterministic run to run.",
            "A stop signal writes train_group's own progress state after the current update; a resumed run "
            "recompiles (the warm-up is inside the measured GPU-s, as in U's reports).",
        ],
    }
    eb.write_json(path, plan)
    log(f"plan.json frozen ({len(plan['pins'])} pins); {STEPS} updates per seed")
    return 0


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --gport and --prepare first")
    plan = eb.read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("plan_identity") != IDENTITY:
        raise ValueError("plan.json binding differs")
    eb.verify_pins(plan["pins"])
    return plan


# ---------------------------------------------------------------------------------------------------------------
# --train --gpu g
# ---------------------------------------------------------------------------------------------------------------

def verify_training_data(plan):
    if eb.sha256_of(ROOT / plan["training_data"]["path"]) != plan["training_data"]["sha256"]:
        raise ValueError("the retrain training data differ from the handoff hash")


def training_data():
    return rt.IndexedTrainData(rt.load_training_data(rt.OUTPUT), FAMILY)


def train_seed(output, plan, seed, data, device="cuda"):
    """One seed through the frozen train_group (resumable from its progress state); then the per-update log and
    the report (training GPU-s = train_group's own measure, counted once; #123 A3)."""
    output = Path(output)
    steps = plan["recipe"]["steps"]
    impl = implementation(steps)
    index = seed_index(seed)
    phase = train_phase(seed)
    gpu = device != "cpu"
    began = time.monotonic()
    try:
        group, ensemble = impl.train_group(output, ARM, FAMILY, index, [("all", seed)], data)
    except eb.TrainingPaused:
        wall = time.monotonic() - began
        eb.mark(output, phase, wall, wall if gpu else 0.0, paused_at=eb.utc_now())
        raise
    eb.atomic_torch(updates_path(output, seed), {
        "schema": SCHEMA_UPDATES, "plan_identity": IDENTITY, "family": FAMILY, "index": index,
        "members": [("all", seed)], "requests": rt.step_requests(FAMILY, steps), "skipped": ensemble.skipped.cpu(),
        "logs": ensemble.log_tensors()})
    del ensemble
    if gpu:
        torch.cuda.empty_cache()
    wall = time.monotonic() - began
    eb.mark(output, phase, wall, wall if gpu else 0.0)
    ledger = eb.read_json(eb.rc.ledger_dir(output) / f"{phase}.json")
    member = group["members"][0]
    seconds = group["gpu_seconds"]
    report = {"schema": SCHEMA_REPORT, "plan_identity": IDENTITY, "arm": VIEW, "seed": seed, "steps": group["steps"],
              "gpu_seconds": seconds, "wall_seconds": ledger["wall_seconds"], "complete": member["complete"],
              "retired_at": member["retired_at"], "parameters_per_member": group["parameters_per_member"],
              "peak_memory_gb": group["peak_memory_gb"], "pool_sizes": group["pool_sizes"][0],
              "ratio_to_u": seconds / plan["u_reference"]["mean"], "ratio_to_a4": seconds / plan["a4_reference"]["mean"],
              "group_report": rel(group_report_path(output, seed)),
              "group_report_sha256": eb.sha256_of(group_report_path(output, seed)),
              "predictor": rel(predictor_path(output, seed, steps)),
              "predictor_sha256": eb.sha256_of(predictor_path(output, seed, steps)),
              "updates": rel(updates_path(output, seed)), "updates_sha256": eb.sha256_of(updates_path(output, seed)),
              "written_at": eb.utc_now()}
    eb.write_json(report_path(output, seed), report)
    log(f"trained {VIEW} seed {seed}: {steps} updates, {seconds:.0f} GPU-s ({report['ratio_to_u']:.3f} x U, "
        f"{report['ratio_to_a4']:.3f} x A4); complete {member['complete']}")
    return report


def train(output, queue, device=None):
    output = Path(output)
    plan = load_plan(output)
    device = device_of(device)
    implementation(plan["recipe"]["steps"])  # exists before a stop signal can arrive (request_pause reaches it)
    install_stop_handlers()
    seeds = [s for s in plan["queues"]["train"].get(str(queue), []) if not report_path(output, s).is_file()]
    log(f"train queue {queue}: {seeds}")
    if not seeds:
        return 0
    verify_training_data(plan)
    torch.backends.cudnn.benchmark = True  # as rt.train
    with on_device(device):
        data = training_data()
        for seed in seeds:
            if eb.STOP_REQUESTED:
                log(f"stop signal; exiting before seed {seed}")
                return eb.EXIT_PAUSED
            try:
                train_seed(output, plan, seed, data, device)
            except eb.TrainingPaused as paused:
                log(f"seed {seed} paused ({paused})")
                return eb.EXIT_PAUSED
    return 0


def training_report_of(output):
    def report_of(phase):
        for seed in eb.SEEDS:
            if phase == train_phase(seed):
                return report_path(output, seed)
        return None
    return report_of


# ---------------------------------------------------------------------------------------------------------------
# --readouts --gpu g (fit / policy data only) and --freeze-readouts
# ---------------------------------------------------------------------------------------------------------------

def end_state(output, plan, seed):
    """U's retrain guards on one seed's update log: usable iff complete and no #117 end-state excursion (the
    handoff rule; otherwise a typed failure); G2 with the scaled skip cap (descriptive)."""
    report = eb.read_json(report_path(output, seed))
    updates = torch.load(updates_path(output, seed), map_location="cpu", weights_only=True)
    local, norms = updates["logs"]["local"][:, 0].double(), updates["logs"]["grad_norm"][:, 0].double()
    excursion = rt.excursion(local, norms, updates["requests"]) if report["complete"] else None
    skipped = int(updates["skipped"][0])
    usable = bool(report["complete"] and not excursion["excursion"])
    return {"complete": report["complete"], "retired_at": report["retired_at"], "skipped": skipped,
            "logged_updates": len(local), "excursion": excursion,
            "G2_stability": bool(report["complete"] and skipped <= plan["skip_cap"]
                                 and len(local) == plan["recipe"]["steps"]),
            "usable": usable,
            "typed_failure": None if usable else ("retired" if not report["complete"] else "final_checkpoint_excursion")}


def load_native(output, plan, seed):
    return implementation(plan["recipe"]["steps"]).load_member(output, ARM, FAMILY, "all", seed)


@torch.no_grad()
def readouts_seed(output, plan, seed, device=None):
    output = Path(output)
    path = readouts_path(output, seed)
    if path.is_file():
        return 0
    device = device_of(device)
    began = time.monotonic()
    result = {"schema": SCHEMA_READOUTS, "plan_identity": IDENTITY, "arm": VIEW, "seed": seed,
              "end_state": end_state(output, plan, seed)}
    result["typed_failure"] = result["end_state"]["typed_failure"]
    with on_device(device), harness.deterministic_scoring() as policy:
        result["scoring_policy"] = policy
        if result["typed_failure"] is None:
            model = load_native(output, plan, seed)
            costs = harness.EndpointCosts(wlc.load_objective())
            items = eb.grid_items()
            actions = eb.m99.action_batch(items)
            spec = t96.arm_spec(REQUEST)
            fit = eb.fit_anchors()["carriers"]
            members = sorted(fit)
            matrix, ends = [], []
            for member in members:
                rows = gb.request_block(model, REQUEST, fit[member], items, costs)["rows"]
                matrix.append([np.nan if r["excluded"] else r[eb.PRIMARY_COST] for r in rows])
                z_end, _ = eb.m99.rollout_endpoints(model, None, fit[member].to(gb.DEVICE), actions, spec)
                ends.append(z_end.cpu())
            matrix = np.asarray(matrix, dtype=float)
            panchors = eb.policy_anchors()["carriers"]
            policy_rows = {member: gb.request_block(model, REQUEST, panchors[member], entry["inventory"], costs)["rows"]
                           for member, entry in sorted(eb.policy_cells(GRID).items())}
            from world_model.training.matched_dynamics import pairs_for
            pair = pairs_for(model)[spec["pair_index"]]
            z0 = fit[members[0]].to(gb.DEVICE)
            result.update(
                G_rank=eb.g_rank(matrix), G_action=float(eb.action_sensitivity(torch.stack(ends))),
                fit_anchor_nonfinite_share=float(np.isnan(matrix).mean()),
                policy_auc=eb.policy_grid_auc(policy_rows, eb.PRIMARY_COST),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                macs_per_candidate=eb.macs_per_candidate(eb._reference_rollout, model, pair, z0,
                                                         actions.to(gb.DEVICE), candidates=len(items)),
                model=rel(predictor_path(output, seed, plan["recipe"]["steps"])),
                model_sha256=eb.sha256_of(predictor_path(output, seed, plan["recipe"]["steps"])))
    seconds = time.monotonic() - began
    result.update(seconds=seconds, written_at=eb.utc_now())
    eb.mark(output, f"readouts-{VIEW}-seed{seed}", seconds, seconds if device != "cpu" else 0.0)
    eb.write_json(path, result)  # written last (atomic)
    log(f"readouts seed {seed}: " + json.dumps({k: result.get(k) for k in ("typed_failure", "G_action", "policy_auc")}))
    return 0


def readouts(output, queue, device=None):
    plan = load_plan(output)
    for seed in plan["queues"]["readouts"].get(str(queue), []):
        readouts_seed(output, plan, seed, device)
    return 0


def manifest_entry(output, plan, seed):
    files = {"report": report_path(output, seed), "updates": updates_path(output, seed),
             "group_report": group_report_path(output, seed),
             "model": predictor_path(output, seed, plan["recipe"]["steps"]), "readouts": readouts_path(output, seed)}
    return {k: {"artifact": rel(p), "sha256": eb.sha256_of(p)} for k, p in files.items()}


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
    missing = [s for s in eb.SEEDS if not readouts_path(output, s).is_file()]
    if missing:
        raise SystemExit(f"readouts missing for seeds {missing}")
    eb.write_json(path, {"schema": SCHEMA_READOUTS_MANIFEST, "plan_identity": IDENTITY, "frozen_at": eb.utc_now(),
                         "entries": {f"{VIEW}--seed{s}": manifest_entry(output, plan, s) for s in eb.SEEDS}})
    log(f"readouts frozen ({len(eb.SEEDS)} seeds)")
    return 0


def load_manifest(output):
    path = Path(output) / "readouts-manifest.json"
    if not path.is_file():
        raise ValueError("readouts are not frozen; run --freeze-readouts first")
    manifest = eb.read_json(path)
    for name, entry in manifest["entries"].items():
        for kind, item in entry.items():
            if eb.sha256_of(ROOT / item["artifact"]) != item["sha256"]:
                raise ValueError(f"{name}: frozen {kind} changed")
    return manifest


# ---------------------------------------------------------------------------------------------------------------
# --score (once; a process that did not train; GPU 0; deterministic): U's own Gate-B scoring path
# ---------------------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def as_gate_b_unit(model, path):
    """Substitute the U-native checkpoint for the Gate-B handoff checkpoint of U inside gb (and so inside
    gb.score_record / rc.score_reference_curves), restoring gb afterwards."""
    saved = gb.load_model, gb.member_path

    def load_model(name, seed):
        if name != UNIT:
            raise ValueError(f"only the unit is substituted, not {name}")
        return model, None

    gb.load_model = load_model
    gb.member_path = lambda name, seed: path
    try:
        yield
    finally:
        gb.load_model, gb.member_path = saved


def score_positions(output, model, path, seed, cells, device):
    """POST-FREEZE. Predicted-carrier position error at eb.POSITION_TIMES on the grid mixed-cell branches (U's #123
    curve rule, rc.score_reference_curves, from the Gate-B anchors)."""
    with as_gate_b_unit(model, path):
        return eb.rc.score_reference_curves(output, UNIT, seed, cells, eb.position_cache(), device)["positions"]


@torch.no_grad()
def score_seed(output, plan, seed, anchors, cells, device=None):
    """One record exactly as U's Gate-B record (gb.score_record: every mixed cell of every inventory from its Gate-B
    anchor, the wrong-anchor rows from the 480 fit anchors), plus position error and wall time per decision."""
    device = device_of(device)
    readout = eb.read_json(readouts_path(output, seed))
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "arm": VIEW, "seed": seed, "engine_seconds": 0}
    if readout["typed_failure"] is not None:
        return {**base, "typed_failure": readout["typed_failure"], "inventories": {}, "wrong_anchor": None}
    path = predictor_path(output, seed, plan["recipe"]["steps"])
    model = load_native(output, plan, seed)
    costs = harness.EndpointCosts(wlc.load_objective())
    carriers = anchors["carriers"]
    with as_gate_b_unit(model, path), harness.deterministic_scoring() as policy:
        record = gb.score_record(cells, carriers, gb.fit_anchors(), costs, UNIT, seed, policy)
    items = eb.grid_items()  # = gb.inventory_items("grid") (the same 16 launches)
    members = sorted(cells[GRID]["mixed"])[:20]
    with harness.deterministic_scoring():
        wall = eb.wall_per_decision(lambda m: gb.request_block(model, REQUEST, carriers[m], items, costs), members)
        positions = score_positions(output, model, path, seed, cells, device)
    gate_b_cell = record.pop("cell")
    return {**record, **base, "gate_b_schema": record["schema"], "cell": {**gate_b_cell, "name": VIEW},
            "positions": positions, "wall_per_decision": wall}


def score(output, device=None):
    output = Path(output)
    plan = load_plan(output)
    manifest = load_manifest(output)
    device = device_of(device)
    with on_device(device):
        with harness.deterministic_scoring() as policy:
            path = eb.evaluation_anchors_path(output)
            anchors = (torch.load(path, map_location="cpu", weights_only=True) if path.is_file()
                       else eb.read_evaluation_anchors(output, policy))
        cells = eb.evaluation_cells()
        for seed in eb.SEEDS:
            target = record_path(output, seed)
            if target.is_file():
                continue
            began = time.monotonic()
            record = score_seed(output, plan, seed, anchors, cells, device)
            seconds = time.monotonic() - began
            record.update(scoring_policy=record.get("scoring_policy", policy), scored_at=record.get(
                "scored_at", eb.utc_now()), readouts_frozen_at=manifest["frozen_at"],
                anchors_read_at=anchors["read_at"], score_seconds=seconds)
            eb.write_json(target, record)
            eb.mark(output, f"score-{VIEW}-seed{seed}", seconds, seconds if device != "cpu" else 0.0)
            log(f"scored seed {seed} ({seconds:.0f}s)" + (f"; typed failure {record['typed_failure']}"
                                                          if record["typed_failure"] else ""))
    return 0


# ---------------------------------------------------------------------------------------------------------------
# --publish / --validate
# ---------------------------------------------------------------------------------------------------------------

def load_records(output):
    return {seed: eb.read_json(record_path(output, seed)) for seed in eb.SEEDS}


def global_guards(output, plan, records):
    output = Path(output)
    manifest = eb.read_json(output / "readouts-manifest.json")
    anchors_read = min(r["anchors_read_at"] for r in records.values())
    first_scored = min(r["scored_at"] for r in records.values())
    return {
        "G_isolation": {"pass": True, "rule": ("the fit split trains (the retrain payload); the policy split gives "
                                               "readouts only; evaluation anchors are read only by --score after "
                                               "--freeze-readouts (G_chronology)")},
        "G_chronology": {"pass": bool(plan["frozen_at"] <= manifest["frozen_at"] <= anchors_read <= first_scored),
                         "plan_frozen_at": plan["frozen_at"], "readouts_frozen_at": manifest["frozen_at"],
                         "anchors_read_at": anchors_read, "first_scored_at": first_scored},
        "G_scoring_policy": {"pass": all(r["scoring_policy"] == harness.SCORING_POLICY for r in records.values())},
        "G_records": eb.records_guard(output, IDENTITY),
    }


def arm_guards(output, records):
    output = Path(output)
    per_seed = {}
    for seed in eb.SEEDS:
        rd = eb.read_json(readouts_path(output, seed))
        per_seed[seed] = {"G_rank": rd.get("G_rank"), "G_action": rd.get("G_action"), "G_collapse": None,
                          "policy_ceiling_auc": None,
                          "typed_failure": rd.get("typed_failure") or records[seed].get("typed_failure")}
    guards = eb.arm_guard_block(per_seed, ARM_GUARDS, nonfinite=eb.nonfinite_share(records.values(), eb.PRIMARY_COST))
    gport_record = eb.read_json(output / "gport" / "gport.json")
    return guards, {"G_port": bool(gport_record["G_port"])}


def arm_compute(output, records):
    seconds, macs, params = {}, [], None
    for seed in eb.SEEDS:
        seconds[str(seed)] = eb.read_json(report_path(output, seed))["gpu_seconds"]
        rd = eb.read_json(readouts_path(output, seed))
        if rd.get("macs_per_candidate") is not None:
            macs.append(rd["macs_per_candidate"])
            params = rd["trainable_parameters"]
    walls = [r["wall_per_decision"] for r in records.values() if r.get("wall_per_decision") is not None]
    return {"trainable_parameters": params, "training_gpu_seconds": seconds,
            "macs_per_candidate": float(np.mean(macs)) if macs else None,
            "wall_per_decision": float(np.median(walls)) if walls else None}


def a4_source():
    """#128's published A4: records hash-checked against #128's records manifest, its G-records, guards and
    compute (read only by --publish, after this child's records exist)."""
    summary = eb.load_child_summary(A128 / "summary.json")
    block = summary["arms"][A4]
    manifest = {e["artifact"]: e["sha256"] for e in eb.read_json(A128 / "records-manifest.json")["records"]}
    records, hashes = {}, {}
    for seed, path in a4_record_paths().items():
        hashes[str(seed)] = bool(manifest.get(rel(path)) == eb.sha256_of(path))
        records[seed] = eb.read_json(path)
    g_records = eb.records_guard(A128, summary["identity"])
    return {"records": records, "key": block["primary_cost_key"], "guards_pass": block["guards_pass"],
            "arm_guards_pass": block["guards"]["pass"], "global_guards_pass": summary["global_guards_pass"],
            "records_guard": bool(g_records["pass"]), "records_in_manifest": hashes,
            "training_gpu_seconds": block["compute"]["training_gpu_seconds"],
            "summary_sha256": eb.sha256_of(A128 / "summary.json"),
            "records_manifest_sha256": eb.sha256_of(A128 / "records-manifest.json")}


def a4_guarded(a4):
    return bool(a4["guards_pass"] and a4["arm_guards_pass"] and a4["global_guards_pass"] and a4["records_guard"]
                and all(a4["records_in_manifest"].values()))


def descriptive(block, label):
    return {"label": label, "first": block["first"], "second": block["second"], "estimate": block["estimate"],
            "cells": block.get("cells"), "descriptive": True,
            "note": "what the extra budget buys (U-native at A4's training GPU-h vs U at B1); no token"}


def u_native_extra(rows, a4, native_guarded, budget):
    """extra.u_native: A4 - U-native (eb.contrast: token = A4 - U-native, reverse_token = U-native - A4), guarded by
    U-native's guards (global and arm) and A4's published guards / G-records / record hashes; U-native - U
    descriptive; the budget ratios; the mechanical reading."""
    guarded = bool(native_guarded and a4_guarded(a4))
    block = eb.contrast(rows, A4, VIEW, guarded)
    supported = block["token"] == "supported"
    return {
        "A4 - U-native": {"label": "A4 - U-native", **block,
                          "guards": {"u_native": bool(native_guarded), "a4": a4_guarded(a4),
                                     "a4_detail": {k: a4[k] for k in ("guards_pass", "arm_guards_pass",
                                                                      "global_guards_pass", "records_guard",
                                                                      "records_in_manifest")}}},
        "U-native - U": descriptive(eb.contrast(rows, VIEW, "U", native_guarded), "U-native - U"),
        "budget": budget,
        "reading": ("A4 - U-native supported: A4's budget qualification is dropped (an architectural reading is "
                    "permitted)" if supported else
                    "A4 - U-native not supported: A4's wording stays budget-qualified"),
        "a4_source": {"summary_sha256": a4["summary_sha256"], "records_manifest_sha256": a4["records_manifest_sha256"],
                      "key": a4["key"]},
    }


def budget_block(plan, compute, a4):
    seconds = {s: float(v) for s, v in compute["training_gpu_seconds"].items()}
    mean = float(np.mean(list(seconds.values())))
    u_mean = plan["u_reference"]["mean"]
    a4_mean = float(np.mean([float(v) for v in a4["training_gpu_seconds"].values()]))
    return {"steps": plan["recipe"]["steps"], "u_steps": plan["recipe"]["u_steps"],
            "update_ratio": plan["recipe"]["update_ratio"], "factor": plan["recipe"]["factor"]["value"],
            "u_native_training_gpu_s": seconds, "u_native_mean": mean, "u_mean": u_mean, "a4_mean": a4_mean,
            "ratio_to_u": {s: v / u_mean for s, v in seconds.items()}, "ratio_to_u_mean": mean / u_mean,
            "ratio_to_a4": {s: v / a4_mean for s, v in seconds.items()}, "ratio_to_a4_mean": mean / a4_mean}


def reference_curves():
    return eb.reference_position_curves()


def publication(output):
    """Every published table, derived from the frozen plan, readouts and records (deterministic)."""
    output = Path(output)
    plan = load_plan(output)
    records = load_records(output)
    cells = eb.evaluation_cells()
    refs = eb.references()
    a4 = a4_source()
    guards_global = global_guards(output, plan, records)
    global_pass = all(g["pass"] is True for g in guards_global.values())
    guards, extra_guards = arm_guards(output, records)
    compute = arm_compute(output, records)
    views = {VIEW: (records, eb.PRIMARY_COST), A4: (a4["records"], a4["key"])}
    tables, block, extra = {}, None, None
    for inventory in eb.INVENTORIES:
        rows = eb.view_rows(cells, inventory, views, refs)
        tables[inventory] = {"rankers": eb.ranker_table(rows, ["U", "C*", "P", "J", "J_dec", VIEW, A4])}
        if inventory == GRID:
            block = eb.arm_block(rows, VIEW, records, eb.PRIMARY_COST, guards, global_pass, compute, ROLE, False,
                                 extra_guards=extra_guards)
            extra = u_native_extra(rows, a4, block["guards_pass"], budget_block(plan, compute, a4))
        else:
            tables[inventory]["contrasts"] = {
                f"{VIEW} - U": eb.contrast(rows, VIEW, "U", block["guards_pass"]),
                f"{A4} - {VIEW}": eb.contrast(rows, A4, VIEW, bool(block["guards_pass"] and a4_guarded(a4)))}
    wrong = {VIEW: eb.wrong_anchor_table(cells, records, VIEW, eb.PRIMARY_COST, block["guards_pass"])}
    positions = eb.position_table({VIEW: records, "U": reference_curves()["U"]}, cells)
    readouts_view = {f"{VIEW}--seed{s}": {k: v for k, v in eb.read_json(readouts_path(output, s)).items()
                                          if k in ("typed_failure", "G_action", "policy_auc", "end_state",
                                                   "fit_anchor_nonfinite_share")} for s in eb.SEEDS}
    summary = eb.child_summary(ISSUE, IDENTITY, plan["frozen_at"], {VIEW: block}, {}, guards_global, {}, {
        "u_native": extra, "inventories": tables, "wrong_anchor": wrong, "positions": positions,
        "readouts": readouts_view, "recipe": plan["recipe"], "decision": plan["decision"]})
    ledger, gpu_total, wall_total = eb.ledger_totals(output, training_report_of(output))
    compute_json = {"schema": SCHEMA_COMPUTE, "plan_identity": IDENTITY, "ledger": ledger,
                    "gpu_seconds_total": gpu_total, "wall_seconds_total": wall_total, "arms": {VIEW: compute},
                    "budget": extra["budget"], "projection": plan["projection"]}
    return summary, compute_json


def findings_text(summary):
    u = summary["extra"]["u_native"]
    a4u, unu, budget = u["A4 - U-native"], u["U-native - U"], u["budget"]
    lines = [f"# #{ISSUE} U-native (budget-safeguard control): findings", "",
             f"Global guards pass: {summary['global_guards_pass']}. U-native = U's #112 v3 retrain recipe at "
             f"{budget['steps']:,} updates ({budget['update_ratio']:.3f} x U's {budget['u_steps']:,}); decision "
             f"{summary['extra']['decision']}.", "", *eb.format_arm_rows(summary["arms"]), "",
             "## A4 - U-native (grid; the control's purpose)", "",
             f"- A4 - U-native: {eb.rc.fmt_block(a4u['estimate'])}; token {a4u['token']} (A4 - U-native), reverse "
             f"{a4u['reverse_token']} (U-native - A4); guards U-native {a4u['guards']['u_native']}, A4 "
             f"{a4u['guards']['a4']}",
             f"- Reading: {u['reading']}.",
             f"- U-native - U (descriptive): {eb.rc.fmt_block(unu['estimate'])}", "",
             "## Training budget (measured)", "",
             "| seed | U-native GPU-s | / U (1 294.19) | / A4 |", "| --- | --- | --- | --- |"]
    for seed, seconds in budget["u_native_training_gpu_s"].items():
        lines.append(f"| {seed} | {seconds:.1f} | {budget['ratio_to_u'][seed]:.3f} | {budget['ratio_to_a4'][seed]:.3f} |")
    lines.append(f"| mean | {budget['u_native_mean']:.1f} | {budget['ratio_to_u_mean']:.3f} | "
                 f"{budget['ratio_to_a4_mean']:.3f} |")
    lines += ["", "## Wrong anchor (right - wrong, grid)", ""]
    for arm, entry in summary["extra"]["wrong_anchor"].items():
        lines.append(f"- {arm}: {eb.rc.fmt_block(entry['right_minus_wrong'])} ({entry['token']})")
    lines += ["", "## Position error (descriptive; pooled mean, member-clustered interval)", "",
              "| ranker | " + " | ".join(f"t = {t}" for t in eb.POSITION_TIMES) + " |",
              "| --- |" + " --- |" * len(eb.POSITION_TIMES)]
    for name, row in summary["extra"]["positions"].items():
        lines.append(f"| {name} | " + " | ".join(eb.rc.fmt_auc(row[f'pred_{t}']) for t in eb.POSITION_TIMES) + " |")
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


def spot_check(output, plan, device=None):
    """gb.spot_checks on the first usable seed: re-score every grid cell in this process and compare the stored
    rows by harness.endpoint_replication."""
    records = load_records(output)
    seed = next((s for s in eb.SEEDS if records[s].get("typed_failure") is None), None)
    if seed is None:
        return {"record": None, "pass": False, "rule": "no usable record"}
    anchors = torch.load(eb.evaluation_anchors_path(output), map_location="cpu", weights_only=True)["carriers"]
    cells = eb.evaluation_cells()
    costs = harness.EndpointCosts(wlc.load_objective())
    worst = dict.fromkeys(harness.PARSED_QUANTITIES, 0.0)
    passed = True
    with on_device(device_of(device)), harness.deterministic_scoring():
        model = load_native(output, plan, seed)
        for member, entry in sorted(cells[GRID]["mixed"].items()):
            fresh = gb.request_block(model, REQUEST, anchors[member], entry["inventory"], costs)
            check = harness.endpoint_replication(records[seed]["inventories"][GRID][member]["rows"], fresh["rows"])
            passed &= check["pass"]
            for k in worst:
                worst[k] = max(worst[k], check["max_abs_delta"][k])
    return {"record": f"{VIEW}--seed{seed}", "max_abs_delta": worst, "pass": bool(passed),
            "rule": "gb.spot_checks: every grid cell re-scored, harness.endpoint_replication"}


def validate(output, device=None):
    """Hash-check the records against the manifest, re-score one record, write the receipt, then re-derive the
    publication twice and require byte-identical files."""
    output = Path(output)
    plan = load_plan(output)
    load_manifest(output)
    manifest = eb.read_json(output / "records-manifest.json")
    error, spot = None, None
    if manifest != eb.records_manifest(output, IDENTITY, manifest["patterns"]):
        error = "records differ from records-manifest.json"
    else:
        spot = spot_check(output, plan, device)
    eb.write_receipt(output, IDENTITY, spot, error)
    publish(output)
    first = {name: (output / name).read_bytes() for name in ("summary.json", "compute.json", "findings.md")}
    publish(output)
    stable = first == {name: (output / name).read_bytes() for name in first}
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
    for mode in ("gport", "prepare", "train", "readouts", "freeze-readouts", "score", "publish", "validate"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="queue index = physical GPU (the job's CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    output = Path(args.output)
    if args.gport:
        return gport(output)
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
    if args.publish:
        return publish(output)
    return validate(output, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
