"""Issue-116: request-set scaling (denser and continuous Delta, larger |R|, DP-labelling and policy cost),
scored once on the #104 sealed #109 evaluation split.

Binding runner module: scripts/run_request_set_scaling.py
Exact validation command: python -u -m scripts.run_request_set_scaling --validate

The request set R was hard-coded in three places (``cnn_hybrid.PAIRS`` / ``carrier``, the controller's 9-way
layer, the DP's 9-column cost tensor). ``world_model/training/request_sets.py`` declares R:

1. Generalised R (Work 1). R_9 reproduces the #112 LC path bit for bit: training (the retrain's
   ``train_group``), the Gate-B / #101 fixed-request records and the #101 prediction-error DP labels (smoke
   replication guard; guard G5 re-checks every R_9 fixed-request row of the evaluation split).
2. Frozen sets (Work 2): R_9 = {1, 5, 15} x alpha, R_18 = {1, 2, 3, 5, 10, 15} x alpha, R_30 = {1, 2, 3, 5,
   8, 10, 15, 25, 45, 75} x alpha, R_cont = {1..75} x alpha. One LC hybrid per set (#112 v3 LC recipe on
   the #109 fit payload, fold 'all', the three #112 seeds) under two budgets: equal total updates (9000) and
   equal updates per request (1000 x |R|). R_9 under both budgets is the Gate-B joint itself.
3. DP scaling (Work 3): exact prediction-error DP labels (ICLR eq. dp, lambda 1e-4) on the #109 policy pool
   with 225-frame windows; labelling GPU s, peak memory and label entropy against O(N L |R|); for R_cont the
   exact DP against a log-spaced pruned DP (label disagreement, value gap).
4. Policy scaling (Work 4): flat |R|-way, factorised Delta x alpha and (R_cont) ordinal-Delta x alpha heads
   at matched parameters per set, #101 pe recipe (2 DAgger rounds x 1800 steps); label accuracy on the
   evaluation grid branches, rollout error at t = 225 (#100 engine_position_mse), per-inventory ranking AUC,
   MACs per candidate and per state, Delta histograms.
5. Controls (Work 5): the no-model action prior and the wrong-anchor control in every ranking table.
6. Sub-frame Delta (< 50 native steps) is ``unsupported`` (Work 6): one carrier frame is the capture stride.

Modes: --dry-run --smoke --prepare --train --label --score --publish --validate. --train/--label/--score take
--workers N --worker K (one process per GPU; CUDA_VISIBLE_DEVICES set by the caller). Zero engine seconds.
Intervals: level-clustered percentile bootstraps (10000 draws, PCG64 7201), DESCRIPTIVE; delta = 0.02.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import hashlib
import json
from pathlib import Path
import shutil
import time
import types

import numpy as np
import torch

from scripts import post_freeze_amendments as amendments
from scripts import run_decision_aware_policy as p101
from scripts import run_decision_chain_attribution as m99
from scripts import run_decision_dynamics_controls as c2
from scripts import run_decision_dynamics_retrain as rt
from scripts import run_decision_dynamics_versions as versions
from scripts import run_decision_relevant_dynamics as v1
from scripts import run_gate_b as gb
from scripts import run_request_interference_ranking as rk
from scripts import run_slot_encoder_fix as s99
from scripts import run_tau_ad_within_checkpoint as t96
from world_model.model import Abstraction
from world_model.planning import scoring_harness as harness
from world_model.training import request_policy as rp
from world_model.training import request_sets as rs
from world_model.training.cnn_hybrid import PAIRS, CNNHybridPredictor
from world_model.training.matched_dynamics import linear_macs

ROOT = v1.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-116-request-set-scaling-v1"
RETRAIN = rt.OUTPUT
GATE_B = gb.OUTPUT
POLICY = p101.OUTPUT
TRUTH = ROOT / ".local-artifacts/issue-115-predicate-feedback-v1/data/engine-truth.pt"
TRUTH_SCHEMA = "issue_115_engine_truth_v1"

IDENTITY = "issue-116-request-set-scaling-v1"
SCHEMA_PLAN = "issue_116_plan_v1"
SCHEMA_SMOKE = "issue_116_smoke_v1"
SCHEMA_PREDICTOR = "issue_116_predictor_v1"
SCHEMA_TRAINED = "issue_116_trained_v1"
SCHEMA_UPDATES = "issue_116_updates_v1"
SCHEMA_LABELS = "issue_116_labels_v1"
SCHEMA_POLICY = "issue_116_policy_v1"
SCHEMA_RECORD = "issue_116_record_v1"
SCHEMA_MANIFEST = "issue_116_records_manifest_v1"
SCHEMA_COMPUTE = "issue_116_compute_v1"
SCHEMA_REPORT = "issue_116_report_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_request_set_scaling --validate"
RUNNER = "scripts/run_request_set_scaling.py"
MODULE = "world_model/training/request_sets.py"

DEVICE = "cuda"
SEEDS = tuple(t96.SEEDS)
INVENTORIES = gb.INVENTORIES
PRIMARY = gb.GATE_INVENTORY
PRIMARY_COST = harness.PRIMARY_COST
DELTA = v1.DELTA
STOP_TOKEN = v1.STOP_TOKEN
NONFINITE_SHARE_MAX = v1.NONFINITE_SHARE_MAX
ENDPOINT = t96.ENDPOINT
SETS = tuple(rs.DELTA_GRIDS)
BASELINE = "R_9"
BUDGETS = rs.BUDGETS
CONFIGS = (BASELINE, *(f"{s}/{b}" for s in SETS[1:] for b in BUDGETS))
TRAINED_CONFIGS = CONFIGS[1:]
HEADS = {s: ("flat", "factorised", "ordinal") if s == "R_cont" else ("flat", "factorised") for s in SETS}
PRIMARY_HEAD = "flat"
FLAT_WIDTH = 128
REMAINING_FEATURE = "clamp60"
LAMBDA = rp.ICLR_COMPUTE_WEIGHT
LABEL_WINDOW = ENDPOINT
TIMING_WINDOW = 60
DP_CHUNK = 256
PRUNED_DELTAS = rs.pruned_deltas()
SKIP_CAP = rt.SKIP_CAP
WORKERS = 4
SMOKE_STEPS = 12
SMOKE_WINDOWS = 256
TIMING_DELTAS = (1, 2, 3, 5, 8, 10, 15, 25, 45, 75)
BOOTSTRAP_SEED = 7201
GPU_CAP_SECONDS = 192 * 3600.0     # 48 h on 4 GPUs (the owner's limit)
ELAPSED_LIMIT_HOURS = 48.0
STABILITY_TOLERANCE = 1e-12
SUBFRAME = {"token": "unsupported",
            "reason": ("one carrier frame is 50 native engine steps (the #104 / #109 capture stride, frames 30000 + "
                       "50 i); a Delta below one frame has no observed target in any existing capture, and a finer-"
                       "stride capture is outside the frozen #109 / #104 plan (an owner decision, not this ticket)")}

read_json = v1.read_json
write_json = v1.write_json
json_text = v1.json_text
sha256_of = v1.sha256_of
atomic_torch = v1.atomic_torch


def write_record(path, value):
    """Records are compact JSON (sorted keys, no whitespace): the R_cont fixed records hold 225 requests."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def log(message):
    print(f"[issue-116] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# configurations: a request set and a budget (R_9: the Gate-B joint under both budgets)
# ---------------------------------------------------------------------------

def config_set(config):
    return rs.request_set(config.split("/")[0])


def config_budget(config):
    return config.split("/")[1] if "/" in config else None


def config_of(set_name, budget):
    return BASELINE if set_name == BASELINE else f"{set_name}/{budget}"


def tag(config):
    return config.replace("/", "--")


def heads_of(config):
    return HEADS[config_set(config).name]


def config_updates(config):
    budget = config_budget(config)
    return rs.schedule(config_set(config), budget or "total")


def member_skip_cap(config):
    return rs.skip_cap(len(config_updates(config)), SKIP_CAP)


def parse_task(key):
    parts = key.split("|")
    return (parts[0], int(parts[1]), *parts[2:])


# ---------------------------------------------------------------------------
# paths and ledger
# ---------------------------------------------------------------------------

def member_path(output, config, seed):
    return Path(output) / "models" / tag(config) / f"seed-{seed}" / "predictor.pt"


def progress_path(output, config, seed):
    return Path(output) / "models" / tag(config) / f"seed-{seed}" / "progress.pt"


def trained_path(output, config, seed):
    return Path(output) / "training" / f"{tag(config)}--seed{seed}.json"


def updates_path(output, config, seed):
    return Path(output) / "updates" / f"{tag(config)}--seed{seed}.pt"


def labels_path(output, config, seed):
    return Path(output) / "labels" / f"{tag(config)}--seed{seed}.json"


def round0_path(output, config, seed):
    return Path(output) / "labels" / f"{tag(config)}--seed{seed}--round0.pt"


def policy_path(output, config, head, seed):
    return Path(output) / "policies" / tag(config) / head / f"seed-{seed}" / "policy.pt"


def record_path(output, config, part, seed):
    return Path(output) / "records" / f"{tag(config)}--{part}--seed{seed}.json"


def ledger_path(output, phase):
    return Path(output) / "ledger" / f"{phase}.json"


def manifest_path(output):
    return Path(output) / "records-manifest.json"


def mark(output, phase, wall, gpu, **extra):
    path = ledger_path(output, phase)
    block = read_json(path) if path.is_file() else {"wall_seconds": 0.0, "gpu_seconds": 0.0}
    write_json(path, {**block, "wall_seconds": block["wall_seconds"] + wall, "gpu_seconds": block["gpu_seconds"] + gpu,
                      "last_at": utc_now(), **extra})
    total = sum(read_json(p)["gpu_seconds"] for p in (Path(output) / "ledger").glob("*.json"))
    if total > GPU_CAP_SECONDS:
        raise ValueError(f"{STOP_TOKEN}: GPU-seconds cap exceeded")


# ---------------------------------------------------------------------------
# training data: the retrain's indexed #109 fit payload with Delta-extended local windows
# ---------------------------------------------------------------------------

class ScalingTrainData(rt.IndexedTrainData):
    """``rt.IndexedTrainData`` (the Gate-B joint's data object); ``windows(..., delta)`` gathers the
    ``request_sets.local_window(delta)``-transition window at the same drawn window start when 60 frames hold
    fewer than four transitions of Delta (the same index draw, so every generator stream is unchanged)."""

    def __init__(self, payload):
        super().__init__(payload, "hybrid")
        self.window_shot = payload["window_shot"].to(DEVICE)
        self.window_start = self.window_positions[:, 0] - self.offsets[self.window_shot]

    def windows(self, pools, generators, count, delta=None):
        if delta is None or rs.local_window(delta) == s99.WINDOW_LENGTH:
            return super().windows(pools, generators, count)
        index = self._draw(pools, "window", generators, count)
        shot = self.window_shot[index]
        positions, length = rs.window_positions(self.window_start[index], self.offsets[shot], self.lasts[shot], delta)
        out = {"z": self.carriers[positions], "action": self.window_action[index], "length": length}
        out.update({k: v[positions] for k, v in self.symbols.items()})
        return out


def expected_payload_sha():
    return read_json(RETRAIN / "compute.json")["data"]["training_data_sha256"]


def training_recipe():
    return s99.training_recipe()


def v3_blocks():
    v3 = versions.implementation(rt.VERSION)
    return v3.LONG, v3.RANK


@functools.lru_cache(maxsize=None)
def pig_columns():
    return v1.pig_columns()


def train_config(output, config, seed, data, **kwargs):
    long, rank = v3_blocks()
    presence, center = pig_columns()
    return rs.train_member(config_set(config), config_updates(config), data, seed, recipe=training_recipe(),
                           long=long, rank=rank, presence=presence, center=center, device=DEVICE, log=log, **kwargs)


# ---------------------------------------------------------------------------
# training stage
# ---------------------------------------------------------------------------

def train(output, workers, worker):
    output = Path(output)
    plan = load_plan(output)
    if workers != plan["execution"]["workers"]:
        raise ValueError(f"the frozen assignment is for {plan['execution']['workers']} workers")
    mine = [parse_task(k) for k in plan["execution"]["training"]["assignment"][str(worker)]]
    mine = [(c, s) for c, s in mine if not trained_path(output, c, s).is_file()]
    if not mine:
        return 0
    torch.backends.cudnn.benchmark = True
    payload_sha = sha256_of(rt.data_path(RETRAIN))
    if payload_sha != plan["data"]["training_data_sha256"]:
        raise ValueError("the retrain training payload differs from the frozen hash")
    data = ScalingTrainData(rt.load_training_data(RETRAIN))
    for config, seed in mine:
        began = time.monotonic()
        progress = progress_path(output, config, seed)
        progress.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.reset_peak_memory_stats()
        report, ensemble, logs = train_config(output, config, seed, data, progress=progress)
        updates = config_updates(config)
        atomic_torch(updates_path(output, config, seed), {
            "schema": SCHEMA_UPDATES, "plan_identity": IDENTITY, "config": config, "seed": seed,
            "schedule": torch.tensor(updates, dtype=torch.int16), **logs})
        atomic_torch(member_path(output, config, seed), {
            "schema": SCHEMA_PREDICTOR, "plan_identity": IDENTITY, "config": config, "seed": seed,
            "pairs": [list(p.identity) for p in config_set(config).pairs], "updates": len(updates),
            "complete": report["complete"], "retired_at": report["retired_at"], "model": ensemble.member_state(0)})
        progress.unlink(missing_ok=True)
        del ensemble
        torch.cuda.empty_cache()
        write_json(trained_path(output, config, seed), {
            "schema": SCHEMA_TRAINED, "plan_identity": IDENTITY, "config": config, "seed": seed,
            "trained_at": utc_now(), "training_data_sha256": payload_sha, "device": torch.cuda.get_device_name(0),
            "sha256": sha256_of(member_path(output, config, seed)), **report})
        mark(output, f"train-{tag(config)}-seed{seed}", time.monotonic() - began, report["gpu_seconds"], worker=worker)
        log(f"worker {worker}: trained {config} seed {seed} ({len(updates)} updates, {report['gpu_seconds']:.0f}s); "
            f"complete {report['complete']}, skipped {report['skipped']}")
    return 0


def training_complete(output):
    return all(trained_path(output, c, s).is_file() for c in TRAINED_CONFIGS for s in SEEDS)


def usability(config, complete, skipped, local, norms, requests, expected):
    """Retrain rule over the member's own schedule: complete, at most the scaled skip cap (45 per 9000
    updates), every update logged, no #117 end-of-training excursion."""
    logged = int(torch.isfinite(local).sum()) if complete else len(local)
    end_state = rt.excursion(local, norms, requests) if complete else None
    cap = member_skip_cap(config)
    usable = bool(complete and skipped <= cap and logged == expected and not end_state["excursion"])
    failure = (None if usable else "retired" if not complete else "skip_cap" if skipped > cap
               else "incomplete_log" if logged != expected else "final_checkpoint_excursion")
    return {"complete": bool(complete), "skipped": int(skipped), "skip_cap": cap, "logged_updates": logged,
            "end_state": end_state, "usable": usable, "typed_failure": failure}


@functools.lru_cache(maxsize=None)
def member_state(output, config, seed):
    if config == BASELINE:
        entry = next(c for c in gb.checkpoints("unit") if c["seed"] == seed)
        return {"config": config, "seed": seed, "usable": bool(entry["usable"]), "source": "Gate-B handoff",
                "sha256": entry["sha256"], "typed_failure": None if entry["usable"] else "unusable_in_handoff",
                "updates": rt.STEPS}
    trained = read_json(trained_path(output, config, seed))
    updates = torch.load(updates_path(output, config, seed), map_location="cpu", weights_only=True)
    request_set_ = config_set(config)
    schedule = updates["schedule"].long().tolist()
    names = [request_set_.request_name(k) for k in schedule]
    local, norms = updates["local"].double(), updates["grad_norm"].double()
    state = usability(config, trained["complete"], int(updates["skipped"].sum()), local, norms, names,
                      len(config_updates(config)))
    return {"config": config, "seed": seed, **state, "schedule_matches": schedule == config_updates(config),
            "retired_at": trained["retired_at"], "gpu_seconds": trained["gpu_seconds"],
            "peak_memory_gb": trained["peak_memory_gb"], "parameters": trained["parameters"],
            "updates": trained["updates"], "trained_at": trained["trained_at"], "sha256": trained["sha256"],
            "training_data_sha256": trained["training_data_sha256"],
            "first_update_seconds_total": float(sum(trained["first_update_seconds"].values()))}


def load_member(output, config, seed):
    """(predictor over the config's set, typed failure, sha256)."""
    state = member_state(Path(output), config, seed)
    if state["typed_failure"]:
        return None, state["typed_failure"], None
    if config == BASELINE:
        model, failure = gb.load_model("unit", seed)  # hash-checked against the Gate-B handoff
        if model is None:
            return None, failure, None
        return rs.load_predictor(config_set(config), model.state_dict(), DEVICE), None, state["sha256"]
    path = member_path(output, config, seed)
    if sha256_of(path) != state["sha256"]:
        raise ValueError(f"{config} seed {seed}: checkpoint differs from its training record")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("schema") != SCHEMA_PREDICTOR or payload.get("plan_identity") != IDENTITY
            or (payload["config"], payload["seed"]) != (config, seed)):
        raise ValueError(f"predictor {config}/{seed} binding differs")
    return rs.load_predictor(config_set(config), payload["model"], DEVICE), None, state["sha256"]


# ---------------------------------------------------------------------------
# DP labelling and policy training on the #109 policy pool (225-frame windows)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def pool_windows():
    """The #101 pool (#109 policy split, 3480 shots, E99 carriers) as start-0 windows of LABEL_WINDOW frames
    (rp.first_window rule); level units in roster order (p101.window_units)."""
    pool = p101.load_pool(POLICY)
    windows = []
    for key in pool["shot_keys"]:
        shot = p101.read_shot(POLICY, key)
        windows.append(rp.first_window(shot["carriers"], shot["action"], shot["end"], LABEL_WINDOW))
    z = torch.stack([w["z"] for w in windows])
    action = torch.stack([w["action"] for w in windows])
    length = torch.stack([w["length"] for w in windows]).long()
    units = [[i for i, m in enumerate(pool["shot_members"]) if m == level] for level in pool["levels"]]
    units = [u for u in units if u]
    digest = hashlib.sha256()
    for tensor in (z, action, length):
        digest.update(tensor.contiguous().numpy().tobytes())
    return {"z": z, "action": action, "length": length, "units": units, "levels": list(pool["levels"]),
            "members": list(pool["shot_members"]), "keys": list(pool["shot_keys"]),
            "sha256": "sha256:" + digest.hexdigest()}


def unit_rows(windows_rows, units):
    return [{k: torch.cat([windows_rows[i][k] for i in unit]) for k in ("z", "action", "remaining", "labels")}
            for unit in units]


def flat_reference(request_set_):
    """The set's flat policy structure: the MAC term of every head's DP labels (labels depend on the
    predictor and R, not on the head)."""
    return rp.RequestPolicy(request_set_.pairs, FLAT_WIDTH, REMAINING_FEATURE)


@functools.lru_cache(maxsize=None)
def head_width(head, request_set_):
    if head == "flat":
        return FLAT_WIDTH
    return rs.matched_width(head, request_set_, rs.parameter_count(flat_reference(request_set_)), REMAINING_FEATURE)


def evaluations(length, pairs, frames):
    """Carrier evaluations of one exact DP pass: per window and request, the F - Delta positions."""
    return int(len(length)) * sum(max(0, frames - p.delta) for p in pairs)


def timed(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    out = fn()
    torch.cuda.synchronize()
    return out, time.monotonic() - began, torch.cuda.max_memory_allocated() / 1e9


def label_stats(labels, length, request_set_):
    counts = rs.label_histogram(labels, length, request_set_.size)
    deltas = [sum(counts[i] for i in range(request_set_.size) if request_set_.delta_of(i) == d)
              for d in request_set_.deltas]
    alphas = [sum(counts[i] for i in range(request_set_.size) if i % 3 == a) for a in range(3)]
    return {"counts": counts, "delta_counts": deltas, "alpha_counts": alphas, "entropy_bits": rs.entropy_bits(counts),
            "entropy_max_bits": float(np.log2(request_set_.size)), "delta_entropy_bits": rs.entropy_bits(deltas),
            "positions": int(sum(counts))}


def save_policy(output, config, head, seed, policy, extra):
    atomic_torch(policy_path(output, config, head, seed), {
        "schema": SCHEMA_POLICY, "plan_identity": IDENTITY, "config": config, "head": head, "seed": seed,
        "pairs": [list(p.identity) for p in policy.pairs], "width": policy_width(policy),
        "remaining_feature": policy.remaining_feature, "model": policy.state_dict(), **extra})


def policy_width(policy):
    return policy.layers[0].out_features if hasattr(policy, "layers") else policy.trunk[0].out_features


def load_policy(output, config, head, seed, expected_sha=None):
    path = policy_path(output, config, head, seed)
    if expected_sha is not None and sha256_of(path) != expected_sha:
        raise ValueError(f"policy {config}/{head}/{seed} differs from its label report")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("schema") != SCHEMA_POLICY or payload.get("plan_identity") != IDENTITY
            or (payload["config"], payload["head"], payload["seed"]) != (config, head, seed)):
        raise ValueError(f"policy {config}/{head}/{seed} binding differs")
    policy = rs.build_head(head, config_set(config), payload["width"], payload["remaining_feature"])
    policy.load_state_dict(payload["model"], strict=True)
    return policy.to(DEVICE).eval()


def policy_accuracy(policy, rows, chunk=1 << 15):
    """Share of rows whose greedy request equals the label (exact, Delta, alpha)."""
    request_set_ = policy.request_set if hasattr(policy, "request_set") else None
    deltas = torch.tensor([p.delta for p in policy.pairs])
    hits = torch.zeros(3, dtype=torch.long)
    total = 0
    with torch.no_grad():
        for unit in rows:
            for b in range(0, len(unit["labels"]), chunk):
                choice = policy(unit["z"][b:b + chunk].to(DEVICE), unit["action"][b:b + chunk].to(DEVICE),
                                unit["remaining"][b:b + chunk].to(DEVICE)).argmax(-1).cpu()
                label = unit["labels"][b:b + chunk]
                hits += torch.stack([(choice == label).sum(), (deltas[choice] == deltas[label]).sum(),
                                     (choice % 3 == label % 3).sum()])
                total += len(label)
    del request_set_
    return {"exact": float(hits[0]) / total, "delta": float(hits[1]) / total, "alpha": float(hits[2]) / total,
            "rows": total}


def label_task(output, config, seed, pool, predictor=None):
    """Exact round-0 labels, the L = 60 timing pass, the pruned DP (R_cont), and every head's two #101 pe
    rounds (round 1 on its own DAgger contexts)."""
    request_set_ = config_set(config)
    pairs = request_set_.pairs
    began = time.monotonic()
    model, failure, sha = predictor or load_member(output, config, seed)
    base = {"schema": SCHEMA_LABELS, "plan_identity": IDENTITY, "config": config, "seed": seed, "predictor_sha256": sha,
            "pool_sha256": pool["sha256"]}
    if model is None:
        return {**base, "typed_failure": failure, "labelled_at": utc_now()}
    z, action, length = pool["z"], pool["action"], pool["length"]
    frames = z.shape[1]
    mac = flat_reference(request_set_)
    report = {**base, "typed_failure": None, "windows": len(z), "window_frames": frames, "requests": request_set_.size,
              "chunk": DP_CHUNK, "lambda": LAMBDA}
    with harness.deterministic_scoring() as scoring_policy:
        (labels0, values0), seconds, peak = timed(lambda: rs.window_labels(
            model, pairs, z, action, length, compute_weight=LAMBDA, mac_policy=mac, chunk=DP_CHUNK))
        report["round0"] = {"gpu_seconds": seconds, "peak_memory_gb": peak,
                            "evaluations": evaluations(length, pairs, frames), **label_stats(labels0, length, request_set_)}
        atomic_torch(round0_path(output, config, seed), {"labels": labels0.to(torch.int16), "values": values0})
        short = length.clamp(max=TIMING_WINDOW)
        (labels60, _), seconds, peak = timed(lambda: rs.window_labels(
            model, pairs, z[:, :TIMING_WINDOW + 1].contiguous(), action, short, compute_weight=LAMBDA, mac_policy=mac,
            chunk=DP_CHUNK))
        report["timing_window_60"] = {"gpu_seconds": seconds, "peak_memory_gb": peak,
                                      "evaluations": evaluations(short, pairs, TIMING_WINDOW + 1),
                                      **label_stats(labels60, short, request_set_)}
        if request_set_.name == "R_cont":
            subset = rs.sub_requests(request_set_, PRUNED_DELTAS)
            (labels_p, values_p), seconds, peak = timed(lambda: rs.window_labels(
                model, pairs, z, action, length, compute_weight=LAMBDA, mac_policy=mac, chunk=DP_CHUNK, subset=subset))
            deltas = torch.tensor([p.delta for p in pairs])
            mask = torch.arange(labels0.shape[1])[None] < length[:, None]
            exact_d, pruned_d = deltas[labels0][mask].double(), deltas[labels_p][mask].double()
            report["pruned"] = {
                "deltas": list(PRUNED_DELTAS), "requests": len(subset), "gpu_seconds": seconds, "peak_memory_gb": peak,
                "evaluations": evaluations(length, [pairs[i] for i in subset], frames),
                "disagreement": rs.disagreement(labels0, labels_p, length),
                "delta_disagreement": float((exact_d != pruned_d).double().mean()),
                "alpha_disagreement": float(((labels0 % 3) != (labels_p % 3))[mask].double().mean()),
                "exact_label_on_grid_share": float(torch.isin(exact_d, torch.tensor(PRUNED_DELTAS).double())
                                                   .double().mean()),
                "mean_abs_log_delta_ratio": float((exact_d.log() - pruned_d.log()).abs().mean()),
                "value_gap": {"mean_relative": float(((values_p - values0) / values0).mean()),
                              "max_relative": float(((values_p - values0) / values0).max()),
                              "min_relative": float(((values_p - values0) / values0).min())},
                **label_stats(labels_p, length, request_set_)}
        rows0 = unit_rows(rs.window_rows(z, action, length, labels0), pool["units"])
        report["heads"] = {}
        for head in HEADS[request_set_.name]:
            width = head_width(head, request_set_)
            policy = rs.new_head(head, request_set_, seed, width=width, remaining_feature=REMAINING_FEATURE, device=DEVICE)
            head_began = time.monotonic()
            history = [rows0]
            losses0, seconds0 = timed(lambda: rp.distil_round(policy, rp.aggregate_rounds(history), 0, seed=seed,
                                                               device=DEVICE))[:2]
            contexts, dagger_seconds, dagger_peak = timed(lambda: rs.batched_dagger(model, policy, z, action, length,
                                                                                    chunk=DP_CHUNK))
            (labels1, _), seconds1, peak1 = timed(lambda: rs.window_labels(
                model, pairs, z, action, length, contexts=contexts, compute_weight=LAMBDA, mac_policy=mac,
                chunk=DP_CHUNK))
            rows1 = unit_rows(rs.window_rows(contexts, action, length, labels1), pool["units"])
            history.append(rows1)
            losses1, seconds_d1 = timed(lambda: rp.distil_round(policy, rp.aggregate_rounds(history), 1, seed=seed,
                                                                 device=DEVICE))[:2]
            policy.eval()
            save_policy(output, config, head, seed, policy, {"scoring_policy": scoring_policy})
            report["heads"][head] = {
                "width": width, "parameters": rs.parameter_count(policy), "macs_per_call": rp.policy_macs(policy),
                "final_loss_0": float(np.mean(losses0[-60:])), "final_loss_1": float(np.mean(losses1[-60:])),
                "round1": {"gpu_seconds": seconds1, "peak_memory_gb": peak1,
                           "evaluations": evaluations(length, pairs, frames), **label_stats(labels1, length, request_set_)},
                "dagger_seconds": dagger_seconds, "dagger_peak_memory_gb": dagger_peak,
                "distil_seconds": [seconds0, seconds_d1],
                "label_change_round1_vs_round0": rs.disagreement(labels0, labels1, length),
                "in_sample_accuracy": {"round0_rows": policy_accuracy(policy, rows0),
                                       "round1_rows": policy_accuracy(policy, rows1)},
                "wall_seconds": time.monotonic() - head_began,
                "policy_sha256": sha256_of(policy_path(output, config, head, seed))}
    report["scoring_policy"] = scoring_policy
    report["wall_seconds"] = time.monotonic() - began
    report["labelled_at"] = utc_now()
    return report


def label(output, workers, worker):
    output = Path(output)
    plan = load_plan(output)
    if workers != plan["execution"]["workers"]:
        raise ValueError(f"the frozen assignment is for {plan['execution']['workers']} workers")
    if not training_complete(output):
        raise ValueError("training is not complete; label after every predictor is trained")
    mine = [parse_task(k) for k in plan["execution"]["labelling"]["assignment"][str(worker)]]
    mine = [(c, s) for c, s in mine if not labels_path(output, c, s).is_file()]
    if not mine:
        return 0
    pool = pool_windows()
    if pool["sha256"] != plan["data"]["policy_pool_windows_sha256"]:
        raise ValueError("the policy-pool windows differ from the frozen hash")
    for config, seed in mine:
        began = time.monotonic()
        report = label_task(output, config, seed, pool)
        write_json(labels_path(output, config, seed), report)
        torch.cuda.empty_cache()
        wall = time.monotonic() - began
        mark(output, f"label-{tag(config)}-seed{seed}", wall, wall, worker=worker)
        log(f"worker {worker}: labelled {config} seed {seed} ({wall:.0f}s)"
            + (f"; typed failure {report['typed_failure']}" if report["typed_failure"] else ""))
    return 0


def labelling_complete(output):
    return all(labels_path(output, c, s).is_file() for c in CONFIGS for s in SEEDS)


def read_labels(output, config, seed):
    report = read_json(labels_path(output, config, seed))
    if (report.get("schema") != SCHEMA_LABELS or report.get("plan_identity") != IDENTITY
            or (report["config"], report["seed"]) != (config, seed)):
        raise ValueError(f"label report {config}/{seed} binding differs")
    return report


# ---------------------------------------------------------------------------
# scoring (deterministic_scoring; processes that did not train)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def truth_store():
    store = torch.load(TRUTH, map_location="cpu", weights_only=True, mmap=True)
    if store.get("schema") != TRUTH_SCHEMA:
        raise ValueError("the #115 engine-truth store binding differs")
    index = {k: i for i, k in enumerate(store["keys"])}
    return {"index": index, "presence": store["presence"].clone(), "centers": store["centers"].clone(),
            "grid_keys": list(store["grid_keys"]),
            "grid_last": store["last"][torch.tensor([index[k] for k in store["grid_keys"]])].clone(),
            "carriers": store["carriers"]}


def engine_error(z, store, key):
    if not bool(torch.isfinite(z).all()):
        return None
    i = store["index"][key]
    return v1.r100.engine_position_mse(z.tolist(), {"presence": store["presence"][i][None],
                                                    "centers": store["centers"][i][None]}, 0)


def compact(rows):
    return [None if row["excluded"] else [row[PRIMARY_COST], row[harness.SECONDARY_COST]] for row in rows]


def expand(items, values):
    """Rows (m99.view / c2 form) from a compact block."""
    return [{"ordinal": item["ordinal"], "excluded": value is None, PRIMARY_COST: None if value is None else value[0],
             harness.SECONDARY_COST: None if value is None else value[1]} for item, value in zip(items, values)]


def rollout_block(model, holder, spec, z0, items, costs, store=None, schedules=False):
    """One cell: ``request_sets.rollout_endpoints`` (rp / m99 op for op) from z0 over the items' actions;
    ``holder`` is the policy, or a namespace carrying R for the schedule kind."""
    actions = m99.action_batch(items)
    z_end, timing = rs.rollout_endpoints(model, holder, z0.to(DEVICE), actions, spec, select=t96.select_pairs,
                                         record_schedules=schedules)
    rows = [costs.row(item["ordinal"], z_end[slot].cpu(), z0.cpu()) for slot, item in enumerate(items)]
    out = {"rows": compact(rows), "compute": [timing["linear_macs"], timing["transition_calls"],
                                              timing["controller_calls"], len(items)]}
    if schedules:
        out["schedules"] = [t96.rle(s) for s in timing["schedules"]]
    if store is not None:
        out["engine_mse"] = [engine_error(z_end[slot].cpu(), store, item["key"]) for slot, item in enumerate(items)]
    return out


def fixed_record(output, cells, anchors, config, seed, scoring_policy, predictor=None, store=None):
    request_set_ = config_set(config)
    model, failure, sha = predictor or load_member(output, config, seed)
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "scoring_policy": scoring_policy, "scored_at": utc_now(),
            "config": config, "part": "fixed", "seed": seed, "engine_seconds": 0, "typed_failure": failure,
            "predictor_sha256": sha}
    if model is None:
        return {**base, "requests": {}}
    began = time.monotonic()
    store = truth_store() if store is None else (store or None)
    costs = p101.endpoint_costs()
    holder = types.SimpleNamespace(pairs=request_set_.pairs)
    requests = {}
    for k in range(request_set_.size):
        schedule = rs.endpoint_schedule(request_set_.pairs, k)
        spec = {"kind": "schedule", "schedule": schedule}
        inventories = {i: {} for i in INVENTORIES}
        for inventory, member, entry in p101.ordered_cells(cells):
            inventories[inventory][member] = rollout_block(model, holder, spec, anchors[member], entry["inventory"],
                                                           costs, store if inventory == PRIMARY else None)
        requests[request_set_.request_name(k)] = {"schedule": t96.rle(schedule), "inventories": inventories}
    torch.cuda.synchronize()
    return {**base, "requests": requests, "seconds": time.monotonic() - began}


def eval_label_inputs(cells):
    store = truth_store()
    actions = {}
    for _, entry in sorted(cells[PRIMARY]["mixed"].items()):
        batch = m99.action_batch(entry["inventory"]).cpu()
        for item, a in zip(entry["inventory"], batch):
            actions[item["key"]] = a
    keys = store["grid_keys"]
    return keys, store["carriers"], torch.stack([actions[k] for k in keys]), store["grid_last"].long()


def evaluation_accuracy(model, policies, request_set_, inputs):
    """Exact DP labels (observed contexts, the evaluation grid branches' E99 parses 0..min(225, last)) and each
    head's greedy request at every labelled position (remaining = frames to the window end)."""
    keys, z, action, length = inputs
    mac = flat_reference(request_set_)
    (labels, _), seconds, peak = timed(lambda: rs.window_labels(model, request_set_.pairs, z, action, length,
                                                                compute_weight=LAMBDA, mac_policy=mac, chunk=DP_CHUNK))
    deltas = torch.tensor(request_set_.deltas)[torch.arange(request_set_.size) // 3]
    out = {"dp_seconds": seconds, "dp_peak_memory_gb": peak, "branches": len(keys),
           **{k: v for k, v in label_stats(labels, length, request_set_).items()}, "heads": {}}
    for head, policy in policies.items():
        per_key = {}
        with torch.no_grad():
            for b in range(0, len(keys), DP_CHUNK):
                for i in range(b, min(b + DP_CHUNK, len(keys))):
                    n = int(length[i])
                    choice = policy(z[i, :n].to(DEVICE), action[i].reshape(1, 5).expand(n, -1).to(DEVICE),
                                    torch.arange(n, 0, -1, device=DEVICE)).argmax(-1).cpu()
                    label_ = labels[i, :n]
                    per_key[keys[i]] = [n, int((choice == label_).sum()), int((deltas[choice] == deltas[label_]).sum()),
                                        int((choice % 3 == label_ % 3).sum())]
        out["heads"][head] = per_key
    return out


def policies_record(output, cells, anchors, config, seed, scoring_policy, predictor=None, store=None,
                    accuracy_inputs=None):
    request_set_ = config_set(config)
    model, failure, sha = predictor or load_member(output, config, seed)
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "scoring_policy": scoring_policy, "scored_at": utc_now(),
            "config": config, "part": "policies", "seed": seed, "engine_seconds": 0, "typed_failure": failure,
            "predictor_sha256": sha}
    if model is None:
        return {**base, "heads": {}}
    began = time.monotonic()
    labels = read_labels(output, config, seed)
    store = truth_store() if store is None else (store or None)
    costs = p101.endpoint_costs()
    policies = {head: load_policy(output, config, head, seed, labels["heads"][head]["policy_sha256"])
                for head in heads_of(config)}
    heads = {}
    for head, policy in policies.items():
        inventories = {i: {} for i in INVENTORIES}
        for inventory, member, entry in p101.ordered_cells(cells):
            inventories[inventory][member] = rollout_block(model, policy, {"kind": "joint"}, anchors[member],
                                                           entry["inventory"], costs,
                                                           store if inventory == PRIMARY else None, schedules=True)
        heads[head] = {"policy_sha256": labels["heads"][head]["policy_sha256"], "inventories": inventories}
    items = gb.inventory_items(PRIMARY)
    fit_anchors = gb.fit_anchors()
    wrong = {member: rollout_block(model, policies[PRIMARY_HEAD], {"kind": "joint"}, fit_anchors[member], items,
                                   costs)["rows"] for member in sorted(fit_anchors)}
    accuracy = evaluation_accuracy(model, policies, request_set_, accuracy_inputs or eval_label_inputs(cells))
    torch.cuda.synchronize()
    return {**base, "heads": heads, "wrong_anchor": wrong, "label_accuracy": accuracy,
            "seconds": time.monotonic() - began}


PARTS = ("fixed", "policies")


def score_tasks():
    return [f"{config}|{seed}|{part}" for config in CONFIGS for seed in SEEDS for part in PARTS]


def score(output, workers, worker):
    output = Path(output)
    plan = load_plan(output)
    if workers != plan["execution"]["workers"]:
        raise ValueError(f"the frozen assignment is for {plan['execution']['workers']} workers")
    if not (training_complete(output) and labelling_complete(output)):
        raise ValueError("training or labelling is not complete; score after every policy is trained")
    _, cells = gb.evaluation_set()
    anchors = gb.load_anchors(GATE_B)["anchors"]
    for key in plan["execution"]["scoring"]["assignment"][str(worker)]:
        config, seed, part = parse_task(key)
        path = record_path(output, config, part, seed)
        if path.is_file():
            continue
        began = time.monotonic()
        with harness.deterministic_scoring() as scoring_policy:
            record = (fixed_record if part == "fixed" else policies_record)(output, cells, anchors, config, seed,
                                                                         scoring_policy)
        write_record(path, record)
        torch.cuda.empty_cache()
        wall = time.monotonic() - began
        mark(output, f"score-{tag(config)}-{part}-seed{seed}", wall, wall, worker=worker)
        log(f"worker {worker}: scored {config} {part} seed {seed} ({wall:.0f}s)"
            + (f"; typed failure {record['typed_failure']}" if record["typed_failure"] else ""))
    return 0


def scoring_complete(output):
    return all(record_path(output, config, part, seed).is_file()
               for config in CONFIGS for seed in SEEDS for part in PARTS)


def read_record(output, config, part, seed):
    record = read_json(record_path(output, config, part, seed))
    if (record.get("schema") != SCHEMA_RECORD or record.get("plan_identity") != IDENTITY
            or (record["config"], record["part"], record["seed"]) != (config, part, seed)):
        raise ValueError(f"record {config}/{part}/{seed} binding differs")
    return record


def load_records(output):
    return {(config, part, seed): read_record(output, config, part, seed)
            for config in CONFIGS for part in PARTS for seed in SEEDS}


# ---------------------------------------------------------------------------
# statistics (pure: records, label reports, member states, cells -> tables)
# ---------------------------------------------------------------------------

def fixed_key(config, request):
    return f"{config}:{request}"


def head_key(config, head):
    return f"{config}:J-{head}"


def r9_requests():
    return [rs.request_set(BASELINE).request_name(k) for k in range(len(PAIRS))]


def set_requests(config):
    request_set_ = config_set(config)
    return [request_set_.request_name(k) for k in range(request_set_.size)]


def view_rows(records, cells, inventory, stats, fixed=True):
    rows = []
    for member, entry in sorted(cells[inventory]["mixed"].items()):
        items = entry["inventory"]
        prior = m99.view(entry, gb.prior_rows(entry, stats), PRIMARY_COST)
        for seed in SEEDS:
            views = {"prior": prior}
            for config in CONFIGS:
                record = records[(config, "policies", seed)]
                for head in heads_of(config):
                    found = None if record["typed_failure"] else expand(
                        items, record["heads"][head]["inventories"][inventory][member]["rows"])
                    views[head_key(config, head)] = v1.cost_view(entry, found, PRIMARY_COST)
                if fixed:
                    record = records[(config, "fixed", seed)]
                    for request in set_requests(config):
                        found = None if record["typed_failure"] else expand(
                            items, record["requests"][request]["inventories"][inventory][member]["rows"])
                        views[fixed_key(config, request)] = v1.cost_view(entry, found, PRIMARY_COST)
            rows.append({"member": member, "seed": seed, "views": views})
    return rows


def contrast(rows, pieces, token=c2.token):
    block = m99.paired(rows, pieces)
    return {**block, "token": token(block["estimate"])}


def mean_pieces(config, requests, weight=1.0):
    return [(weight / len(requests), fixed_key(config, r)) for r in requests]


def auc_mean_block(rows, pieces):
    return m99.paired(rows, pieces)


def member_mean(rows, key):
    return t96.member_mean([{"member": r["member"], "auc": r["views"][key]["auc"]} for r in rows], "auc")


def primary_contrast_name(config):
    return f"{config} - {BASELINE}"


def ranking_tables(rows, inventory):
    """Rankers (J heads, fixed request means, best-in-set fixed), contrasts (primary: J-flat of each set and
    budget - J-flat of R_9; heads; fixed R_9-request mean; model - prior)."""
    out = {"cells": len(rows), "members": len({r["member"] for r in rows}), "rankers": {}, "fixed_means": {},
           "contrasts": {}, "heads": {}, "prior_contrasts": {}}
    out["rankers"]["prior"] = v1.ranker_block(rows, "prior")
    for config in CONFIGS:
        for head in heads_of(config):
            out["rankers"][head_key(config, head)] = v1.ranker_block(rows, head_key(config, head))
        out["fixed_means"][config] = {
            "R_9_requests": auc_mean_block(rows, mean_pieces(config, r9_requests()))["estimate"],
            "all_requests": auc_mean_block(rows, mean_pieces(config, set_requests(config)))["estimate"]}
        means = {r: member_mean(rows, fixed_key(config, r)) for r in set_requests(config)}
        best = max(set_requests(config), key=lambda r: (-np.inf if means[r] is None else means[r],
                                                       -set_requests(config).index(r)))
        out["fixed_means"][config]["hindsight"] = {"request": best, "member_mean": means[best],
                                                   "auc": m99.block_of(rows, fixed_key(config, best))}
        if inventory == PRIMARY:
            out["fixed_means"][config]["per_request_member_mean"] = means
        for head in heads_of(config):
            if head != PRIMARY_HEAD:
                out["heads"][f"{config}: J-{head} - J-{PRIMARY_HEAD}"] = contrast(
                    rows, [(1, head_key(config, head)), (-1, head_key(config, PRIMARY_HEAD))])
        out["prior_contrasts"][f"{config}: J-{PRIMARY_HEAD} - prior"] = contrast(
            rows, [(1, head_key(config, PRIMARY_HEAD)), (-1, "prior")])
    for config in TRAINED_CONFIGS:
        out["contrasts"][primary_contrast_name(config)] = {
            "J-flat": contrast(rows, [(1, head_key(config, PRIMARY_HEAD)), (-1, head_key(BASELINE, PRIMARY_HEAD))]),
            "fixed_R_9_requests": contrast(rows, mean_pieces(config, r9_requests())
                                           + mean_pieces(BASELINE, r9_requests(), -1.0))}
    return out


def per_level_values(records, cells, config, part, extract):
    """{member: seed-mean of the per-(member, seed) value} over the grid mixed cells; extract(result) -> value."""
    out = {}
    for member in sorted(cells[PRIMARY]["mixed"]):
        values = []
        for seed in SEEDS:
            record = records[(config, part, seed)]
            if record["typed_failure"]:
                continue
            value = extract(record, member)
            if value is not None:
                values.append(value)
        out[member] = float(np.mean(values)) if values else None
    return out


def level_block(per_level):
    common = [m for m, v in per_level.items() if v is not None]
    return v1.bootstrap([per_level[m] for m in common], common)


def paired_level_block(first, second):
    common = [m for m in sorted(first) if first[m] is not None and second.get(m) is not None]
    return v1.bootstrap([first[m] - second[m] for m in common], common)


def branch_mean(values):
    defined = [v for v in values if v is not None]
    return float(np.mean(defined)) if defined else None


def rollout_tables(records, cells):
    """#100 engine_position_mse at t = 225 on grid: branch mean per (level, seed), seed mean per level, level
    bootstrap; J heads, the fixed R_9-request mean and the per-Delta mean of the fixed requests."""
    out = {"heads": {}, "fixed_R_9_requests": {}, "per_delta": {}, "versus_R_9": {}}
    for config in CONFIGS:
        for head in heads_of(config):
            per = per_level_values(records, cells, config, "policies", lambda r, m, h=head: branch_mean(
                r["heads"][h]["inventories"][PRIMARY][m]["engine_mse"]))
            out["heads"][head_key(config, head)] = {"error": level_block(per), "per_level": per}
        per = per_level_values(records, cells, config, "fixed", lambda r, m: branch_mean(
            [branch_mean(r["requests"][q]["inventories"][PRIMARY][m]["engine_mse"]) for q in r9_requests()]))
        out["fixed_R_9_requests"][config] = {"error": level_block(per), "per_level": per}
        request_set_ = config_set(config)
        curve = {}
        for d in request_set_.deltas:
            names = [f"F-{d}-{a}" for a in rs.ALPHAS]
            values = []
            for seed in SEEDS:
                record = records[(config, "fixed", seed)]
                if record["typed_failure"]:
                    continue
                values += [v for q in names for m in sorted(cells[PRIMARY]["mixed"])
                           for v in record["requests"][q]["inventories"][PRIMARY][m]["engine_mse"] if v is not None]
            curve[str(d)] = float(np.mean(values)) if values else None
        out["per_delta"][config] = curve
    for config in TRAINED_CONFIGS:
        out["versus_R_9"][config] = {
            "J-flat": paired_level_block(out["heads"][head_key(config, PRIMARY_HEAD)]["per_level"],
                                         out["heads"][head_key(BASELINE, PRIMARY_HEAD)]["per_level"]),
            "fixed_R_9_requests": paired_level_block(out["fixed_R_9_requests"][config]["per_level"],
                                                     out["fixed_R_9_requests"][BASELINE]["per_level"])}
    for section in ("heads", "fixed_R_9_requests"):
        for entry in out[section].values():
            entry.pop("per_level")
    return out


def accuracy_tables(records, cells):
    """Label accuracy of every head on the evaluation grid branches (DP labels of the head's own predictor):
    per branch share of positions, mean per level and seed, seed mean per level, level bootstrap."""
    branches_of = {m: [item["key"] for item in e["inventory"]] for m, e in cells[PRIMARY]["mixed"].items()}
    out = {}
    for config in CONFIGS:
        for head in heads_of(config):
            entry = {}
            for k, field in enumerate(("exact", "delta", "alpha")):
                per = per_level_values(records, cells, config, "policies", lambda r, m, h=head, k=k: branch_mean(
                    [r["label_accuracy"]["heads"][h][key][k + 1] / r["label_accuracy"]["heads"][h][key][0]
                     for key in branches_of[m]]))
                entry[field] = level_block(per)
            out[head_key(config, head)] = entry
    return out


def compute_of(records, cells, config, head=None, inventory=PRIMARY):
    """Mean linear MACs, transitions and policy calls per candidate rollout (grid; pooled over seeds)."""
    totals = np.zeros(4)
    for seed in SEEDS:
        record = records[(config, "policies" if head else "fixed", seed)]
        if record["typed_failure"]:
            continue
        if head:
            blocks = record["heads"][head]["inventories"][inventory].values()
        else:
            blocks = [b for q in r9_requests() for b in record["requests"][q]["inventories"][inventory].values()]
        for block in blocks:
            totals += np.asarray(block["compute"], dtype=float)
    if not totals[3]:
        return None
    return {"macs_per_candidate": totals[0] / totals[3], "transitions_per_candidate": totals[1] / totals[3],
            "policy_calls_per_candidate": totals[2] / totals[3]}


def delta_histograms(records, cells, config, head):
    request_set_ = config_set(config)
    counts = [0] * request_set_.size
    for seed in SEEDS:
        record = records[(config, "policies", seed)]
        if record["typed_failure"]:
            continue
        for block in record["heads"][head]["inventories"][PRIMARY].values():
            for text in block["schedules"]:
                for index in t96.unrle(text):
                    counts[index] += 1
    deltas = {str(d): sum(counts[i] for i in range(request_set_.size) if request_set_.delta_of(i) == d)
              for d in request_set_.deltas}
    alphas = {str(a): sum(counts[i] for i in range(request_set_.size) if i % 3 == k) for k, a in enumerate(rs.ALPHAS)}
    total = sum(counts)
    steps = [int(d) for d, c in deltas.items() for _ in range(c)]
    return {"delta_counts": deltas, "alpha_counts": alphas, "steps": total,
            "modal": request_set_.request_name(max(range(request_set_.size), key=lambda k: (counts[k], -k))),
            "delta_quantiles": [float(q) for q in np.quantile(steps, (0.1, 0.5, 0.9))] if steps else None,
            "delta_entropy_bits": rs.entropy_bits(list(deltas.values()))}


def macs_tables(reports, records, cells):
    out = {}
    for config in CONFIGS:
        request_set_ = config_set(config)
        usable = [reports[(config, s)] for s in SEEDS if not reports[(config, s)]["typed_failure"]]
        model_macs = {str(a): linear_macs(CNNHybridPredictor(), rs.request_set(BASELINE).pairs[k])
                      for k, a in enumerate(rs.ALPHAS)}
        out[config] = {"predictor_macs_per_transition": model_macs,
                       "fixed_R_9_requests": compute_of(records, cells, config),
                       "heads": {head: {"per_candidate": compute_of(records, cells, config, head),
                                        "policy_macs_per_call": usable[0]["heads"][head]["macs_per_call"] if usable
                                        else rp.policy_macs(rs.build_head(head, request_set_,
                                                                          head_width(head, request_set_))),
                                        "parameters": rs.parameter_count(rs.build_head(head, request_set_,
                                                                                       head_width(head, request_set_))),
                                        "width": head_width(head, request_set_),
                                        "delta_histogram": delta_histograms(records, cells, config, head)}
                                 for head in heads_of(config)}}
    return out


def dp_tables(reports):
    """Labelling cost per set and budget: round 0 exact (L = 225), the L = 60 timing pass, every head's round 1
    and (R_cont) the pruned DP; seconds per carrier evaluation and the per-unit ratio to R_9."""
    out = {}
    units = []
    for config in CONFIGS:
        per_seed = {}
        windows = None
        for seed in SEEDS:
            report = reports[(config, seed)]
            if report["typed_failure"]:
                per_seed[str(seed)] = {"typed_failure": report["typed_failure"]}
                continue
            entry = {key: {k: report[key][k] for k in ("gpu_seconds", "peak_memory_gb", "evaluations", "entropy_bits",
                                                       "entropy_max_bits", "delta_entropy_bits", "positions")}
                     for key in ("round0", "timing_window_60")}
            entry["round0"]["delta_counts"] = report["round0"]["delta_counts"]
            entry["round0"]["alpha_counts"] = report["round0"]["alpha_counts"]
            entry["heads"] = {h: {"round1_seconds": v["round1"]["gpu_seconds"], "dagger_seconds": v["dagger_seconds"],
                                  "distil_seconds": v["distil_seconds"],
                                  "label_change_round1_vs_round0": v["label_change_round1_vs_round0"],
                                  "in_sample_accuracy": v["in_sample_accuracy"]}
                              for h, v in report["heads"].items()}
            if "pruned" in report:
                entry["pruned"] = {k: v for k, v in report["pruned"].items() if k not in ("counts",)}
            for key in ("round0", "timing_window_60"):
                entry[key]["seconds_per_evaluation"] = entry[key]["gpu_seconds"] / entry[key]["evaluations"]
                units.append({"config": config, "seed": seed, "pass": key, "requests": report["requests"],
                              "evaluations": entry[key]["evaluations"], "gpu_seconds": entry[key]["gpu_seconds"]})
            per_seed[str(seed)] = entry
            windows = report["windows"]
        out[config] = {"requests": config_set(config).size, "windows": windows, "per_seed": per_seed}
        good = [e for e in per_seed.values() if "round0" in e]
        if good:
            out[config]["round0_mean"] = {k: float(np.mean([e["round0"][k] for e in good]))
                                          for k in ("gpu_seconds", "peak_memory_gb", "entropy_bits",
                                                    "seconds_per_evaluation", "delta_entropy_bits")}
    baseline = out[BASELINE].get("round0_mean", {}).get("seconds_per_evaluation")
    for config in CONFIGS:
        mean = out[config].get("round0_mean")
        if mean and baseline:
            mean["per_unit_ratio_to_R_9"] = mean["seconds_per_evaluation"] / baseline
    x = np.array([u["evaluations"] for u in units], dtype=float)
    y = np.array([u["gpu_seconds"] for u in units], dtype=float)
    fit = None
    if len(x) > 2:
        slope, intercept = np.polyfit(x, y, 1)
        predicted = slope * x + intercept
        fit = {"seconds_per_evaluation": float(slope), "intercept_seconds": float(intercept),
               "r2": float(1 - ((y - predicted) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
               "log_log_slope_vs_requests_round0": float(np.polyfit(
                   np.log([u["requests"] for u in units if u["pass"] == "round0"]),
                   np.log([u["gpu_seconds"] for u in units if u["pass"] == "round0"]), 1)[0]),
               "passes": len(units)}
    return {"configs": out, "linear_fit": fit, "passes": units}


def training_table(output):
    members = [member_state(Path(output), c, s) for c in CONFIGS for s in SEEDS]
    return {"members": members, "typed_failures": [{"config": m["config"], "seed": m["seed"],
                                                    "typed_failure": m["typed_failure"]}
                                                   for m in members if m["typed_failure"]]}


def disposition_ranking(grid, guards_ok):
    """Pre-declared: per (set, budget) the grid J-flat - J-flat(R_9) delta superiority token; overall supported
    iff some set is supported under both budgets; not_supported_by_this_experiment iff every (set, budget) is;
    otherwise insufficient; a failed guard turns every token into STOP_TOKEN."""
    tokens = {config: grid["contrasts"][primary_contrast_name(config)]["J-flat"]["token"] for config in TRAINED_CONFIGS}
    if not guards_ok:
        return {"tokens": {c: STOP_TOKEN for c in tokens}, "token": STOP_TOKEN}
    sets_both = [s for s in SETS[1:] if all(tokens[f"{s}/{b}"] == "supported" for b in BUDGETS)]
    if sets_both:
        token = "supported"
    elif all(t == "not_supported_by_this_experiment" for t in tokens.values()):
        token = "not_supported_by_this_experiment"
    else:
        token = STOP_TOKEN
    return {"tokens": tokens, "token": token, "sets_supported_under_both_budgets": sets_both}


TRACTABLE_SECONDS = 3600.0
TRACTABLE_UNIT_RATIO = 2.0


def disposition_dp(dp, guards_ok):
    """Pre-declared: the exact DP labeller stays tractable at |R| = 225 iff every usable R_cont predictor's exact
    round-0 pass over the policy pool (N = 3480, L = 225) completed on one RTX 5090 within TRACTABLE_SECONDS GPU
    seconds and its seconds per carrier evaluation is at most TRACTABLE_UNIT_RATIO x the R_9 mean (no
    super-linear growth in |R|); not_supported_by_this_experiment iff any fails; insufficient if no R_cont
    predictor is usable or a guard fails."""
    baseline = dp["configs"][BASELINE].get("round0_mean", {}).get("seconds_per_evaluation")
    checks = []
    for config in ("R_cont/total", "R_cont/request"):
        for seed, entry in dp["configs"][config]["per_seed"].items():
            if "round0" not in entry:
                continue
            ratio = entry["round0"]["seconds_per_evaluation"] / baseline if baseline else None
            checks.append({"config": config, "seed": int(seed), "gpu_seconds": entry["round0"]["gpu_seconds"],
                           "peak_memory_gb": entry["round0"]["peak_memory_gb"], "unit_ratio_to_R_9": ratio,
                           "ok": bool(entry["round0"]["gpu_seconds"] <= TRACTABLE_SECONDS and ratio is not None
                                      and ratio <= TRACTABLE_UNIT_RATIO)})
    if not guards_ok or not checks:
        token = STOP_TOKEN
    else:
        token = "supported" if all(c["ok"] for c in checks) else "not_supported_by_this_experiment"
    return {"checks": checks, "token": token}


def replication(records):
    """G5: every R_9 fixed-request row (tie-free and count cost) equals the Gate-B unit records (F-15-macro) and
    the #101 fixed-request records, every inventory and seed."""
    _, cells = gb.evaluation_set()
    same, blocks = True, 0
    for seed in SEEDS:
        record = records[(BASELINE, "fixed", seed)]
        for request in r9_requests():
            reference = (read_json(gb.record_path(GATE_B, "unit", seed))["inventories"]
                         if request == gb.UNITS["unit"]["request"] else p101.read_record(POLICY, request, seed)["inventories"])
            for inventory in INVENTORIES:
                for member, result in reference[inventory].items():
                    mine = record["requests"][request]["inventories"][inventory][member]["rows"]
                    same &= mine == compact(result["rows"])
                    blocks += 1
    return {"blocks": blocks, "pass": bool(same)}


def anchor_table(records, cells):
    rows = {}
    for config in CONFIGS:
        cell_rows = []
        for member, entry in sorted(cells[PRIMARY]["mixed"].items()):
            for seed in SEEDS:
                record = records[(config, "policies", seed)]
                if record["typed_failure"]:
                    cell_rows.append({"member": member, "seed": seed, "views": {"J": None}})
                    continue
                items = gb.inventory_items(PRIMARY)
                right = expand(entry["inventory"], record["heads"][PRIMARY_HEAD]["inventories"][PRIMARY][member]["rows"])
                wrong = {a: {"rows": expand(items, v)} for a, v in record["wrong_anchor"].items()}
                summary = rk.anchor_summary(entry, right, rk.wrong_matrix(wrong))
                cell_rows.append({"member": member, "seed": seed, "views": {"J": summary}})
        difference = c2.difference_block(cell_rows, [(1, "J", "right"), (-1, "J", "wrong")])
        rows[config] = {"right": c2.field_block(cell_rows, "J", "right"), "wrong": c2.field_block(cell_rows, "J", "wrong"),
                        "right_minus_wrong": difference, "stability": c2.field_block(cell_rows, "J", "stability"),
                        "token": c2.token(difference)}
    return {"arm": f"J-{PRIMARY_HEAD}", "wrong_anchors": len(gb.fit_anchors()), "configs": rows}


def invalidate(obj):
    if isinstance(obj, dict):
        return {k: STOP_TOKEN if k == "token" else invalidate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [invalidate(v) for v in obj]
    return obj


def guard_block(output, plan, records, shares, label_reports):
    members = training_table(output)["members"]
    eval_roster = {lv["identity"] for lv in gb.evaluation_levels()}
    fit_roster = {lv["identity"] for lv in rt.fit_levels()}
    pool = p101.load_pool(POLICY)
    g1 = (not (set(rt.load_training_data(RETRAIN)["shot_members"]) & eval_roster)
          and not (set(pool["levels"]) & (eval_roster | fit_roster)) and set(gb.fit_anchors()) <= fit_roster)
    policies = [r["scoring_policy"] for r in records.values()] + [r["scoring_policy"] for r in label_reports
                                                                 if not r["typed_failure"]]
    g2 = all(p == harness.SCORING_POLICY for p in policies)
    stamps_train = [m["trained_at"] for m in members if m["config"] != BASELINE]
    stamps_label = [r["labelled_at"] for r in label_reports]
    stamps_score = [r["scored_at"] for r in records.values()]
    g3 = plan["frozen_at"] <= min(stamps_train) and max(stamps_train + stamps_label) <= min(stamps_score)
    g4 = (all(r["predictor_sha256"] == member_state(Path(output), r["config"], r["seed"])["sha256"]
              for r in records.values() if not r["typed_failure"])
          and all(r["pool_sha256"] == plan["data"]["policy_pool_windows_sha256"] for r in label_reports)
          and all(m["training_data_sha256"] == plan["data"]["training_data_sha256"] for m in members
                  if m["config"] != BASELINE)
          and all(records[(r["config"], "policies", r["seed"])]["heads"][h]["policy_sha256"] == v["policy_sha256"]
                  for r in label_reports if not r["typed_failure"] for h, v in r["heads"].items()))
    g5 = replication(records)
    g6 = all(m["schedule_matches"] and m["logged_updates"] == m["updates"] for m in members
             if m["config"] != BASELINE and m["complete"])
    g7 = all(v is not None and v <= NONFINITE_SHARE_MAX for v in shares.values())
    guards = {"G1_isolation": bool(g1), "G2_scoring_policy": bool(g2), "G3_chronology": bool(g3),
              "G4_binding": bool(g4), "G5_replication": g5, "G6_training_integrity": bool(g6),
              "G7_nonfinite": {"shares": shares, "pass": bool(g7)}}
    ok = all(v if isinstance(v, bool) else v["pass"] for v in guards.values())
    return guards, ok


def compute_accounting(output):
    ledger = {p.stem: read_json(p) for p in sorted((Path(output) / "ledger").glob("*.json"))}
    phases = {}
    for name, block in ledger.items():
        kind = name.split("-")[0]
        phases.setdefault(kind, 0.0)
        phases[kind] += block["gpu_seconds"]
    return {"gpu_seconds": sum(b["gpu_seconds"] for b in ledger.values()), "by_phase": phases,
            "hardware": "4 x NVIDIA GeForce RTX 5090 (not the #85 RTX 3090)", "engine_seconds": 0}


def statistics(records, cells, stats):
    """The record-only tables (rankings per inventory, wrong-anchor, rollout error, label accuracy)."""
    inventories = {i: ranking_tables(view_rows(records, cells, i, stats[i]), i) for i in INVENTORIES}
    return inventories, anchor_table(records, cells), rollout_tables(records, cells), accuracy_tables(records, cells)


def assemble(plan, records, reports, cells, stats, guards, ok, training, compute):
    """Every published table from records, label reports, guard results, training and compute accounting."""
    inventories, wrong, rollout, accuracy = statistics(records, cells, stats)
    grid = inventories[PRIMARY]
    dp = dp_tables(reports)
    rankings = disposition_ranking(grid, ok)
    tractable = disposition_dp(dp, ok)
    if not ok:
        inventories, wrong = invalidate(inventories), invalidate(wrong)
    return {"schema": SCHEMA_COMPUTE, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "dispositions": {"larger_or_continuous_delta_improves_ranking": rankings,
                             "dp_labeller_tractable_at_225": tractable,
                             "sub_frame_delta": SUBFRAME},
            "guards": guards, "guards_pass": ok, "inventories": inventories, "wrong_anchor": wrong,
            "rollout_error": rollout, "label_accuracy": accuracy, "macs": macs_tables(reports, records, cells),
            "dp_scaling": dp, "training": training, "compute": compute}


def compute_tables(output, plan):
    output = Path(output)
    _, cells = gb.evaluation_set()
    records = load_records(output)
    reports = {(c, s): read_labels(output, c, s) for c in CONFIGS for s in SEEDS}
    fit = read_json(gb.FIT_VERDICTS)["branches"]
    stats = {i: gb.prior_stats(fit, i) for i in INVENTORIES}
    grid_rows = view_rows(records, cells, PRIMARY, stats[PRIMARY], fixed=False)
    shares = {c: v1.ranker_block(grid_rows, head_key(c, PRIMARY_HEAD))["nonfinite_share"] for c in CONFIGS}
    guards, ok = guard_block(output, plan, records, shares, list(reports.values()))
    return assemble(plan, records, reports, cells, stats, guards, ok, training_table(output),
                    compute_accounting(output))


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def num(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def est(block, digits=3):
    if block is None:
        return "n/a"
    low, high = block["interval"]
    return f"{block['mean']:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]"


def level(block, digits=3):
    if block is None:
        return "n/a"
    low, high = block["interval"]
    return f"{block['mean']:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def auc_cell(table, key, digits=3):
    ranker = table["rankers"].get(key)
    return "n/a" if ranker is None else level(ranker["auc"], digits)


def gmacs(value):
    return "n/a" if value is None else f"{value / 1e6:.1f}M"


def delta_summary(histogram):
    q = histogram["delta_quantiles"]
    return "n/a" if q is None else f"median {q[1]:g} (10-90 %: {q[0]:g}-{q[2]:g}), modal {histogram['modal']}"


def findings_md(plan, compute):
    grid = compute["inventories"][PRIMARY]
    dispositions = compute["dispositions"]
    ranking = dispositions["larger_or_continuous_delta_improves_ranking"]
    tractable = dispositions["dp_labeller_tractable_at_225"]
    dp = compute["dp_scaling"]["configs"]
    rollout = compute["rollout_error"]
    accuracy = compute["label_accuracy"]
    macs = compute["macs"]
    lines = []
    add = lines.append
    add("# Issue #116: request-set scaling (findings)")
    add("")
    add(f"Plan frozen {plan['frozen_at']}; scored once on the #104 sealed #109 evaluation split; tie-free cost; "
        "level-clustered percentile bootstraps (10 000 draws, PCG64 7201), DESCRIPTIVE; delta = 0.02; zero engine "
        "seconds. Grid is primary; offset, angle, power, rollout error, label accuracy and MACs are DESCRIPTIVE.")
    add("")
    add(f"Guards pass: {compute['guards_pass']}.")
    add("")
    add("## Dispositions (pre-declared)")
    add("")
    add(f"1. **A larger or continuous Delta set improves ranking over R_9: `{ranking['token']}`.** Rule: per set and "
        "budget the grid J-flat AUC minus R_9's J-flat (paired by level and seed) with the delta superiority token; "
        "supported iff some set is supported under both budgets; not_supported_by_this_experiment iff every set and "
        "budget is.")
    add("")
    add("| set / budget | J-flat grid AUC | minus R_9 | token |")
    add("| --- | --- | --- | --- |")
    add(f"| R_9 (Gate-B joint, both budgets) | {auc_cell(grid, head_key(BASELINE, PRIMARY_HEAD))} | - | - |")
    for config in TRAINED_CONFIGS:
        block = grid["contrasts"][primary_contrast_name(config)]["J-flat"]
        add(f"| {config} | {auc_cell(grid, head_key(config, PRIMARY_HEAD))} | {est(block['estimate'])} | "
            f"{ranking['tokens'][config]} |")
    add("")
    add(f"2. **The DP labeller stays tractable at |R| = 225: `{tractable['token']}`.** Rule: every usable R_cont "
        f"predictor's exact round-0 pass over the policy pool (N = 3480 windows, L = 225 frames) within "
        f"{TRACTABLE_SECONDS:.0f} GPU s on one RTX 5090 and at most {TRACTABLE_UNIT_RATIO:g}x R_9's seconds per carrier "
        "evaluation.")
    add("")
    add("| R_cont predictor | GPU s | peak GB | s per evaluation / R_9 | ok |")
    add("| --- | --- | --- | --- | --- |")
    for check in tractable["checks"]:
        add(f"| {check['config']} seed {check['seed']} | {check['gpu_seconds']:.1f} | {check['peak_memory_gb']:.2f} | "
            f"{num(check['unit_ratio_to_R_9'], 2)} | {check['ok']} |")
    add("")
    add(f"3. **Sub-frame Delta: `{SUBFRAME['token']}`.** {SUBFRAME['reason']}.")
    add("")
    add("## Scaling table (grid; per set and budget)")
    add("")
    add("| set / budget | updates | DP round-0 GPU s | peak GB | label entropy (bits / max) | J-flat label accuracy "
        "(eval) | J-flat rollout error t=225 | J-flat grid AUC | J-flat MACs / candidate |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for config in CONFIGS:
        mean = dp[config].get("round0_mean", {})
        entropy = (f"{mean['entropy_bits']:.2f} / {np.log2(config_set(config).size):.2f}" if mean else "n/a")
        flat = macs[config]["heads"][PRIMARY_HEAD]["per_candidate"]
        add(f"| {config} | {len(config_updates(config))} | {num(mean.get('gpu_seconds'), 1)} | "
            f"{num(mean.get('peak_memory_gb'), 2)} | {entropy} | "
            f"{level(accuracy[head_key(config, PRIMARY_HEAD)]['exact'])} | "
            f"{level(rollout['heads'][head_key(config, PRIMARY_HEAD)]['error'], 4)} | "
            f"{auc_cell(grid, head_key(config, PRIMARY_HEAD))} | {gmacs(flat and flat['macs_per_candidate'])} |")
    add("")
    add("## Ranking AUC per inventory (J heads, fixed-request means, controls)")
    add("")
    add("| ranker | " + " | ".join(INVENTORIES) + " |")
    add("| --- |" + " --- |" * len(INVENTORIES))
    for config in CONFIGS:
        for head in heads_of(config):
            add(f"| {config} J-{head} | " + " | ".join(auc_cell(compute["inventories"][i], head_key(config, head))
                                                      for i in INVENTORIES) + " |")
        add(f"| {config} fixed, R_9-request mean | " + " | ".join(
            level(compute["inventories"][i]["fixed_means"][config]["R_9_requests"]) for i in INVENTORIES) + " |")
        add(f"| {config} fixed, all-request mean | " + " | ".join(
            level(compute["inventories"][i]["fixed_means"][config]["all_requests"]) for i in INVENTORIES) + " |")
    add("| no-model action prior | " + " | ".join(auc_cell(compute["inventories"][i], "prior") for i in INVENTORIES)
        + " |")
    add("")
    add("Fixed requests follow the endpoint rule (floor(225 / Delta) transitions, then the same-alpha fallback to "
        "exactly 225). Best fixed request in hindsight (grid, member mean; selection on the evaluation split, "
        "DESCRIPTIVE):")
    add("")
    for config in CONFIGS:
        hindsight = grid["fixed_means"][config]["hindsight"]
        add(f"- {config}: {hindsight['request']} {level(hindsight['auc'])}")
    add("")
    add("### Contrasts against R_9 (grid; paired by level and seed)")
    add("")
    add("| set / budget | J-flat - R_9 J-flat | fixed R_9-request mean - R_9 | rollout error J-flat - R_9 | rollout "
        "error fixed R_9 requests - R_9 |")
    add("| --- | --- | --- | --- | --- |")
    for config in TRAINED_CONFIGS:
        contrasts = grid["contrasts"][primary_contrast_name(config)]
        versus = rollout["versus_R_9"][config]
        add(f"| {config} | {est(contrasts['J-flat']['estimate'])} | {est(contrasts['fixed_R_9_requests']['estimate'])} "
            f"| {est(versus['J-flat'], 4)} | {est(versus['fixed_R_9_requests'], 4)} |")
    add("")
    add("## Policy-head comparison (per set)")
    add("")
    add("| set / budget | head | width | parameters | policy MACs / call | label accuracy exact / Delta / alpha "
        "(eval) | grid AUC | minus flat | rollout error t=225 | MACs / candidate | Delta chosen |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for config in CONFIGS:
        for head in heads_of(config):
            entry = macs[config]["heads"][head]
            acc = accuracy[head_key(config, head)]
            minus = ("-" if head == PRIMARY_HEAD else
                     est(grid["heads"][f"{config}: J-{head} - J-{PRIMARY_HEAD}"]["estimate"]))
            per = entry["per_candidate"]
            add(f"| {config} | {head} | {entry['width']} | {entry['parameters']} | {entry['policy_macs_per_call']} | "
                f"{num(acc['exact'] and acc['exact']['mean'])} / {num(acc['delta'] and acc['delta']['mean'])} / "
                f"{num(acc['alpha'] and acc['alpha']['mean'])} | {auc_cell(grid, head_key(config, head))} | {minus} | "
                f"{level(rollout['heads'][head_key(config, head)]['error'], 4)} | "
                f"{gmacs(per and per['macs_per_candidate'])} | {delta_summary(entry['delta_histogram'])} |")
    add("")
    add("Predictor MACs per transition: " + ", ".join(
        f"{a} {gmacs(v)}" for a, v in macs[BASELINE]["predictor_macs_per_transition"].items()) + " (Delta-independent).")
    add("")
    add("## Controls")
    add("")
    add("| set / budget | J-flat - prior (grid) | token | right - wrong anchor | token | stability |")
    add("| --- | --- | --- | --- | --- | --- |")
    for config in CONFIGS:
        prior = grid["prior_contrasts"][f"{config}: J-{PRIMARY_HEAD} - prior"]
        wrong = compute["wrong_anchor"]["configs"][config]
        add(f"| {config} | {est(prior['estimate'])} | {prior['token']} | {est(wrong['right_minus_wrong'])} | "
            f"{wrong['token']} | {num(wrong['stability'] and wrong['stability']['mean'], 2)} |")
    add("")
    add("## DP labelling cost")
    add("")
    add("| set / budget | requests | round-0 s (L = 225) | L = 60 s | s per 10^6 evaluations (L = 225) | ratio to R_9 "
        "| Delta entropy (bits) |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for config in CONFIGS:
        seeds_ = [e for e in dp[config]["per_seed"].values() if "round0" in e]
        mean = dp[config].get("round0_mean", {})
        short = float(np.mean([e["timing_window_60"]["gpu_seconds"] for e in seeds_])) if seeds_ else None
        add(f"| {config} | {dp[config]['requests']} | {num(mean.get('gpu_seconds'), 1)} | {num(short, 1)} | "
            f"{num(mean['seconds_per_evaluation'] * 1e6 if mean else None, 3)} | "
            f"{num(mean.get('per_unit_ratio_to_R_9'), 2)} | {num(mean.get('delta_entropy_bits'), 2)} |")
    fit = compute["dp_scaling"]["linear_fit"]
    if fit:
        add("")
        add(f"Linear fit over every exact pass: {fit['seconds_per_evaluation'] * 1e6:.3f} s per 10^6 carrier evaluations "
            f"+ {fit['intercept_seconds']:.1f} s (R^2 {fit['r2']:.3f}, {fit['passes']} passes); log-log slope of the "
            f"round-0 seconds against |R|: {fit['log_log_slope_vs_requests_round0']:.2f}.")
    add("")
    add("Exact vs pruned DP (R_cont; candidate Delta restricted to the log-spaced grid "
        f"{list(PRUNED_DELTAS)} x alpha):")
    add("")
    add("| predictor | label disagreement | Delta disagreement | alpha disagreement | exact label on the grid | "
        "mean relative value gap | pruned GPU s |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for config in ("R_cont/total", "R_cont/request"):
        for seed, entry in dp[config]["per_seed"].items():
            if "pruned" not in entry:
                continue
            p = entry["pruned"]
            add(f"| {config} seed {seed} | {p['disagreement']:.3f} | {p['delta_disagreement']:.3f} | "
                f"{p['alpha_disagreement']:.3f} | {p['exact_label_on_grid_share']:.3f} | "
                f"{p['value_gap']['mean_relative']:+.4f} | {p['gpu_seconds']:.1f} |")
    add("")
    add("## Training")
    add("")
    add("| set / budget | seed | updates | GPU s | skipped (cap) | usable | typed failure |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for m in compute["training"]["members"]:
        if m["config"] == BASELINE:
            add(f"| {BASELINE} | {m['seed']} | {m['updates']} | Gate-B handoff | - | {m['usable']} | "
                f"{m['typed_failure'] or '-'} |")
            continue
        add(f"| {m['config']} | {m['seed']} | {m['updates']} | {m['gpu_seconds']:.0f} | {m['skipped']} "
            f"({m['skip_cap']}) | {m['usable']} | {m['typed_failure'] or '-'} |")
    add("")
    add("## Guards")
    add("")
    for name, value in compute["guards"].items():
        add(f"- {name}: {value if isinstance(value, bool) else value['pass']}")
    add("")
    add(f"Compute: {compute['compute']['gpu_seconds']:.0f} GPU s ("
        + ", ".join(f"{k} {v:.0f}" for k, v in sorted(compute["compute"]["by_phase"].items()))
        + f"); {compute['compute']['hardware']}; engine seconds 0.")
    add("")
    return "\n".join(lines)


def summary(plan, compute):
    grid = compute["inventories"][PRIMARY]
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "dispositions": {k: v["token"] for k, v in compute["dispositions"].items()},
            "ranking_tokens": compute["dispositions"]["larger_or_continuous_delta_improves_ranking"]["tokens"],
            "guards_pass": compute["guards_pass"],
            "grid_J_flat_auc": {c: grid["rankers"][head_key(c, PRIMARY_HEAD)]["auc"] for c in CONFIGS},
            "grid_J_flat_minus_R_9": {c: grid["contrasts"][primary_contrast_name(c)]["J-flat"]["estimate"]
                                      for c in TRAINED_CONFIGS},
            "prior_grid_auc": grid["rankers"]["prior"]["auc"],
            "dp_round0_gpu_seconds": {c: compute["dp_scaling"]["configs"][c].get("round0_mean", {}).get("gpu_seconds")
                                      for c in CONFIGS},
            "typed_failures": compute["training"]["typed_failures"],
            "gpu_seconds": compute["compute"]["gpu_seconds"], "engine_seconds": 0,
            "claim_boundary": plan["claim_boundary"]}


def manifest(output):
    """Content hashes of every raw record, label report, round-0 label file and update log."""
    output = Path(output)
    expected = sorted([record_path(output, c, p, s) for c in CONFIGS for p in PARTS for s in SEEDS])
    present = sorted(output.glob("records/*.json"))
    if expected != present:
        raise ValueError(f"records/ holds {len(present)} files, expected exactly the {len(expected)} named records")
    files = present + sorted(output.glob("labels/*")) + sorted(output.glob("updates/*.pt"))
    return {"schema": SCHEMA_MANIFEST, "identity": IDENTITY, "count": len(files),
            "files": {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha256_of(p)} for p in files}}


def rendered(plan, compute, output):
    return {"compute.json": json_text(compute), "summary.json": json_text(summary(plan, compute)),
            "findings.md": findings_md(plan, compute), "records-manifest.json": json_text(manifest(output))}


# ---------------------------------------------------------------------------
# smoke (pre-freeze; fit payload, policy pool and published Gate-B / #101 / #115 records only; nothing kept)
# ---------------------------------------------------------------------------

def smoke_sets():
    """Request sets, schedules, budgets, the curriculum clock, the endpoint rule and the pruned grid."""
    r9 = rs.request_set(BASELINE)
    sizes = {n: rs.request_set(n).size for n in SETS}
    names_ok = [r9.request_name(k) for k in range(r9.size)] == list(t96.FIXED_ARMS)
    schedule_ok = ([r9.request_name(k) for k in rs.schedule(r9, "total")] == rt.step_requests("hybrid")
                   and rs.schedule(r9, "request") == rs.schedule(r9, "total"))
    clock_ok = all(rs.curriculum_step(u, rs.REFERENCE_UPDATES) == u for u in range(rs.REFERENCE_UPDATES))
    endpoint = {}
    endpoint_ok = True
    for name in SETS:
        request_set_ = rs.request_set(name)
        nondividing = []
        for k, pair in enumerate(request_set_.pairs):
            schedule = rs.endpoint_schedule(request_set_.pairs, k)
            endpoint_ok &= sum(request_set_.pairs[j].delta for j in schedule) == ENDPOINT
            endpoint_ok &= all(request_set_.pairs[j].abstraction == pair.abstraction for j in schedule)
            if ENDPOINT % pair.delta == 0:
                endpoint_ok &= schedule == [k] * (ENDPOINT // pair.delta)
            elif pair.abstraction == Abstraction.CONTINUOUS:
                nondividing.append({"delta": pair.delta, "transitions": len(schedule),
                                    "fallback": [request_set_.pairs[j].delta for j in schedule[ENDPOINT // pair.delta:]]})
        endpoint[name] = {"nondividing": nondividing}
    counts = {c: len(config_updates(c)) for c in CONFIGS}
    ok = bool(names_ok and schedule_ok and clock_ok and endpoint_ok
              and sizes == {"R_9": 9, "R_18": 18, "R_30": 30, "R_cont": 225}
              and counts == {"R_9": 9000, "R_18/total": 9000, "R_18/request": 18000, "R_30/total": 9000,
                             "R_30/request": 30000, "R_cont/total": 9000, "R_cont/request": 225000})
    return {"control": "request_sets_schedules_clock_endpoint_rule", "sizes": sizes, "update_counts": counts,
            "pruned_deltas": list(PRUNED_DELTAS), "endpoint_rule": endpoint, "ok": ok}


def smoke_predictor():
    """The generalised predictor class loads the Gate-B checkpoint; carrier equals CNNHybridPredictor's bit for
    bit on every R_9 pair; R_9 rejects a pair outside R_9; R_cont accepts Delta = 75."""
    seed = SEEDS[0]
    reference, _ = gb.load_model("unit", seed)
    anchors = gb.fit_anchors()
    z = torch.stack([anchors[m] for m in sorted(anchors)[:64]]).to(DEVICE)
    action = m99.action_batch(gb.inventory_items(PRIMARY))[torch.arange(64) % 16]
    ours = {name: rs.load_predictor(rs.request_set(name), reference.state_dict(), DEVICE) for name in ("R_9", "R_cont")}
    with harness.deterministic_scoring(), torch.no_grad():
        equal = all(torch.equal(reference.carrier(z, action, p), ours["R_9"].carrier(z, action, p)) for p in PAIRS)
        equal_cont = all(torch.equal(reference.carrier(z, action, p), ours["R_cont"].carrier(z, action, p))
                         for p in PAIRS)
        try:
            ours["R_9"].carrier(z, action, rs.request_set("R_18").pairs[3])
            rejects = False
        except ValueError:
            rejects = True
        far = ours["R_cont"].carrier(z, action, rs.request_set("R_cont").pairs[-1])
    same_state = all(torch.equal(a, b) for a, b in zip(reference.state_dict().values(), ours["R_9"].state_dict().values()))
    return {"control": "generalised_predictor_equals_cnn_hybrid_on_R_9", "carrier_equal": bool(equal),
            "carrier_equal_in_R_cont_class": bool(equal_cont), "rejects_outside_R": rejects,
            "delta_75_finite": bool(torch.isfinite(far).all()), "state_dict_equal": bool(same_state),
            "ok": bool(equal and equal_cont and rejects and torch.isfinite(far).all() and same_state)}


def smoke_windows(data):
    """``request_sets.window_positions`` at Delta <= 15 reproduces the payload's windows; extended windows hold
    four local transitions of Delta where the shot allows."""
    shot = data.window_shot
    positions, length = rs.window_positions(data.window_start, data.offsets[shot], data.lasts[shot], 15)
    same = bool(torch.equal(positions, data.window_positions) and torch.equal(length, data.window_length))
    extended = {}
    for delta in (16, 25, 45, 75):
        _, length_d = rs.window_positions(data.window_start, data.offsets[shot], data.lasts[shot], delta)
        steps = torch.clamp(length_d // delta, max=rs.LOCAL_STEPS)
        extended[str(delta)] = {"window": rs.local_window(delta),
                                "local_steps_histogram": torch.bincount(steps, minlength=5).tolist()}
    return {"control": "window_rule_reproduces_the_payload_windows", "same": same, "extended": extended, "ok": same}


def max_delta(first, second):
    return max(float((first.params[k] - second.params[k]).abs().max()) for k in first.params)


def smoke_training(data):
    """R_9 through ``request_sets.train_member`` equals the retrain's ``train_group`` (the Gate-B joint's training
    path) for SMOKE_STEPS updates at the real schedule and at full cost: parameters and the logged local losses
    and pre-clip norms the excursion rule reads (the replication guard, training side)."""
    retrain = rt.implementation()
    seed = SEEDS[0]
    checks = []
    for smoke_full in (False, True):
        _, ours, logs = train_config(None, BASELINE, seed, data, stop=SMOKE_STEPS, smoke_full=smoke_full)
        _, reference = retrain.train_group(OUTPUT / "smoke-scratch", rt.ARM, "hybrid", 0, [("all", seed)], data,
                                           steps=SMOKE_STEPS, save=False, smoke_full=smoke_full)
        theirs = reference.log_tensors()
        delta = max_delta(ours, reference)
        same_logs = (torch.equal(logs["local"][:SMOKE_STEPS].float(), theirs["local"][:, 0].float())
                     and torch.equal(logs["grad_norm"][:SMOKE_STEPS].float(), theirs["grad_norm"][:, 0].float()))
        checks.append({"control": f"R_9_training_equals_retrain_train_group_{'full_cost' if smoke_full else 'schedule'}",
                       "max_abs_parameter_delta": delta, "logs_equal": bool(same_logs),
                       "ok": bool(delta == 0.0 and same_logs)})
        del ours, reference
        torch.cuda.empty_cache()
    return checks


def smoke_usability():
    compute = read_json(RETRAIN / "compute.json")
    same = True
    for m in compute["members"]:
        updates = torch.load(rt.updates_path(RETRAIN, m["family"], m["index"]), map_location="cpu", weights_only=True)
        local, norms = updates["logs"]["local"][:, 0].double(), updates["logs"]["grad_norm"][:, 0].double()
        state = usability(BASELINE, m["complete"], int(updates["skipped"][0]), local, norms, updates["requests"],
                          rt.STEPS)
        same &= state["end_state"] == m["end_state"] and state["usable"] == m["usable"]
    return {"control": "usability_rule_reproduces_the_retrain_member_table", "members": len(compute["members"]),
            "ok": bool(same)}


def smoke_general_training(data):
    """R_cont at full cost on (Delta, micro) for every timing Delta (non-dividing Delta, extended windows, the
    fallback ranking schedule): finite, no skip; per-update seconds and the first-update (compile) seconds.
    R_30 Delta = 75 at clock 0 (no long term, no ranking term) and at clock 4500."""
    seed = SEEDS[0]
    long, rank = v3_blocks()
    presence, center = pig_columns()
    cont = rs.request_set("R_cont")
    updates = [cont.index(rs.PredictionPair(d, Abstraction.MICRO)) for d in TIMING_DELTAS for _ in range(3)]
    report, ensemble, logs = rs.train_member(cont, updates, data, seed, recipe=training_recipe(), long=long, rank=rank,
                                             presence=presence, center=center, smoke_full=True, step_times=True)
    del ensemble
    r30 = rs.request_set("R_30")
    k75 = r30.index(rs.PredictionPair(75, Abstraction.CONTINUOUS))
    report30, ensemble, logs30 = rs.train_member(r30, [k75, k75], data, seed, recipe=training_recipe(), long=long,
                                                 rank=rank, presence=presence, center=center)
    del ensemble
    torch.cuda.empty_cache()
    steady, first = {}, {}
    for entry in report["update_seconds"]:
        delta = int(entry["request"].split("-")[1])
        (first if entry["first"] else steady).setdefault(delta, []).append(entry["seconds"])
    timing = {"steady_seconds": {str(d): float(np.mean(v)) for d, v in steady.items()},
              "first_update_seconds": {str(d): float(np.mean(v)) for d, v in first.items()},
              "peak_memory_gb": report["peak_memory_gb"]}
    ok = bool(report["complete"] and report["skipped"] == 0 and torch.isfinite(logs["local"]).all()
              and report30["complete"] and report30["skipped"] == 0 and torch.isfinite(logs30["local"]).all())
    return {"control": "R_cont_and_R_30_train_finite_at_every_timing_delta", "skipped": report["skipped"],
            "r30_delta75_trace": report30["trace"], "ok": ok}, timing


def gate_b_rows(seed):
    return read_json(gb.record_path(GATE_B, "unit", seed))


def smoke_scoring():
    """Scoring path (replication guard, evaluation side, published models only): the generalised fixed-request
    rollout of the R_9 predictor reproduces the Gate-B unit records (F-15-macro, every inventory and seed, and
    the 480 wrong-anchor blocks) and the #101 fixed-request records of the other eight requests (grid, first
    seed); the #101 J (pe, 1e-4) policy through the generalised path reproduces its grid rows and schedules; the
    #115 truth store indexes the Gate-B grid branches and its engine metric reproduces #115's F-15-micro values."""
    _, cells = gb.evaluation_set()
    anchors = gb.load_anchors(GATE_B)["anchors"]
    fit_anchors = gb.fit_anchors()
    costs = p101.endpoint_costs()
    items = gb.inventory_items(PRIMARY)
    r9 = rs.request_set(BASELINE)
    holder = types.SimpleNamespace(pairs=r9.pairs)
    store = truth_store()
    grid_keys = [item["key"] for _, entry in sorted(cells[PRIMARY]["mixed"].items()) for item in entry["inventory"]]
    checks, seconds = [], []
    unit = gb.UNITS["unit"]["request"]
    with harness.deterministic_scoring() as scoring_policy:
        same = True
        for seed in SEEDS:
            gate = gate_b_rows(seed)
            model, _, _ = load_member(OUTPUT, BASELINE, seed)
            spec = {"kind": "schedule", "schedule": rs.endpoint_schedule(r9.pairs, t96.FIXED_ARMS.index(unit))}
            torch.cuda.synchronize()
            began = time.monotonic()
            for inventory, member, entry in p101.ordered_cells(cells):
                fresh = rollout_block(model, holder, spec, anchors[member], entry["inventory"], costs)
                same &= fresh["rows"] == compact(gate["inventories"][inventory][member]["rows"])
            torch.cuda.synchronize()
            seconds.append(time.monotonic() - began)
            for member in sorted(fit_anchors):
                fresh = rollout_block(model, holder, spec, fit_anchors[member], items, costs)
                same &= fresh["rows"] == compact(gate["wrong_anchor"][member]["rows"])
        checks.append({"control": "R_9_fixed_rows_reproduce_gate_b_unit_records", "ok": bool(same)})
        seed = SEEDS[0]
        model, _, _ = load_member(OUTPUT, BASELINE, seed)
        same = True
        per_request = {}
        for k, request in enumerate(t96.FIXED_ARMS):
            if request == unit:
                continue
            reference = p101.read_record(POLICY, request, seed)["inventories"][PRIMARY]
            spec = {"kind": "schedule", "schedule": rs.endpoint_schedule(r9.pairs, k)}
            torch.cuda.synchronize()
            began = time.monotonic()
            for member, entry in sorted(cells[PRIMARY]["mixed"].items()):
                fresh = rollout_block(model, holder, spec, anchors[member], entry["inventory"], costs)
                same &= fresh["rows"] == compact(reference[member]["rows"])
            torch.cuda.synchronize()
            per_request[request] = time.monotonic() - began
        checks.append({"control": "R_9_fixed_rows_reproduce_issue_101_records_grid", "ok": bool(same)})
        policy = p101.load_policy(POLICY, "pe", LAMBDA, seed)
        reference = p101.read_record(POLICY, p101.J_ICLR, seed)["inventories"][PRIMARY]
        same = True
        torch.cuda.synchronize()
        began = time.monotonic()
        for member, entry in sorted(cells[PRIMARY]["mixed"].items()):
            fresh = rollout_block(model, policy, {"kind": "joint"}, anchors[member], entry["inventory"], costs,
                                  schedules=True)
            same &= (fresh["rows"] == compact(reference[member]["rows"])
                     and fresh["schedules"] == [reference[member]["schedules"][str(i["ordinal"])]
                                                for i in entry["inventory"]])
        torch.cuda.synchronize()
        j_seconds = time.monotonic() - began
        checks.append({"control": "issue_101_J_reproduces_through_the_generalised_rollout", "ok": bool(same)})
        published = read_json(TRUTH.parents[1] / "records" / f"gateb--default--F-15-micro--seed{seed}.json")
        k = t96.FIXED_ARMS.index("F-15-micro")
        spec = {"kind": "schedule", "schedule": rs.endpoint_schedule(r9.pairs, k)}
        worst = 0.0
        for member, entry in sorted(cells[PRIMARY]["mixed"].items()):
            fresh = rollout_block(model, holder, spec, anchors[member], entry["inventory"], costs, store)
            for mine, theirs in zip(fresh["engine_mse"], published["inventories"][PRIMARY][member]["engine_mse"]):
                worst = max(worst, abs(mine - theirs)) if mine is not None and theirs is not None else (
                    worst if mine is None and theirs is None else float("inf"))
        checks.append({"control": "truth_store_indexes_grid_branches_and_reproduces_issue_115_engine_mse",
                       "grid_keys_equal": store["grid_keys"] == grid_keys, "max_abs_engine_mse_delta": worst,
                       "ok": bool(store["grid_keys"] == grid_keys and worst <= 1e-9)})
    return checks, {"F_15_macro_seconds_per_seed": seconds, "grid_seconds_per_request": per_request,
                    "J_grid_seconds": j_seconds, "scoring_policy": scoring_policy}


def smoke_dp():
    """DP replication guard: on four policy levels (60-frame windows, the #101 pool) with the Gate-B predictor and
    the #101 J (pe, 1e-4) policy, ``window_labels`` (chunk 1) with ``batched_dagger`` (chunk 1) reproduces
    ``request_policy.window_rows`` bit for bit in both rounds; chunked labelling agreement; the pruned DP
    chooses only grid requests and never beats the exact optimum; DP timings at L = 225 for R_9 and R_cont."""
    seed = SEEDS[0]
    pool = p101.load_pool(POLICY)
    units = p101.window_units(pool)[:4]
    windows = [w for unit in units for w in unit]
    model, _, _ = load_member(OUTPUT, BASELINE, seed)
    original, _ = gb.load_model("unit", seed)
    policy = p101.load_policy(POLICY, "pe", LAMBDA, seed)
    z = torch.stack([w["z"] for w in windows])
    action = torch.stack([w["action"] for w in windows])
    length = torch.stack([w["length"] for w in windows]).long()
    checks = []
    with harness.deterministic_scoring():
        for aggregated in (False, True):
            reference = rp.window_rows(original, policy, windows, aggregated=aggregated, compute_weight=LAMBDA,
                                       device=DEVICE)
            contexts = rs.batched_dagger(model, policy, z, action, length, chunk=1) if aggregated else None
            labels, _ = rs.window_labels(model, PAIRS, z, action, length, contexts=contexts, compute_weight=LAMBDA,
                                         mac_policy=policy, chunk=1)
            rows = rs.window_rows(z if contexts is None else contexts, action, length, labels)
            ours = {k: torch.cat([r[k] for r in rows]) for k in ("z", "action", "remaining", "labels")}
            same = all(torch.equal(ours[k], reference[k]) for k in ours)
            chunked, _ = rs.window_labels(model, PAIRS, z, action, length, contexts=contexts, compute_weight=LAMBDA,
                                          mac_policy=policy, chunk=DP_CHUNK)
            if aggregated:
                chunked_contexts = rs.batched_dagger(model, policy, z, action, length, chunk=DP_CHUNK)
                chunked_dagger, _ = rs.window_labels(model, PAIRS, z, action, length, contexts=chunked_contexts,
                                                     compute_weight=LAMBDA, mac_policy=policy, chunk=DP_CHUNK)
            agreement = 1 - rs.disagreement(labels, chunked, length)
            checks.append({"control": f"dp_labels_reproduce_issue_101_window_rows_{'dagger' if aggregated else 'observed'}",
                           "rows": int(len(reference["labels"])), "bit_equal": bool(same),
                           "chunked_agreement": agreement,
                           "chunked_dagger_agreement": (1 - rs.disagreement(labels, chunked_dagger, length))
                           if aggregated else None,
                           "ok": bool(same and agreement >= 0.99)})
        windows225 = pool_windows()
        timing = {}
        cont = rs.request_set("R_cont")
        for name in ("R_9", "R_cont"):
            request_set_ = rs.request_set(name)
            predictor = rs.load_predictor(request_set_, original.state_dict(), DEVICE)
            mac = flat_reference(request_set_)
            zs, actions, lengths = (windows225["z"][:SMOKE_WINDOWS], windows225["action"][:SMOKE_WINDOWS],
                                    windows225["length"][:SMOKE_WINDOWS])
            (labels, values), seconds, peak = timed(lambda: rs.window_labels(
                predictor, request_set_.pairs, zs, actions, lengths, compute_weight=LAMBDA, mac_policy=mac, chunk=DP_CHUNK))
            timing[name] = {"seconds": seconds, "peak_memory_gb": peak,
                            "evaluations": evaluations(lengths, request_set_.pairs, zs.shape[1])}
            if name == "R_cont":
                subset = rs.sub_requests(cont, PRUNED_DELTAS)
                (pruned, pruned_values), seconds, _ = timed(lambda: rs.window_labels(
                    predictor, cont.pairs, zs, actions, lengths, compute_weight=LAMBDA, mac_policy=mac, chunk=DP_CHUNK,
                    subset=subset))
                mask = torch.arange(pruned.shape[1])[None] < lengths[:, None]
                on_grid = bool(torch.isin(pruned[mask], torch.tensor(subset)).all())
                no_better = bool((pruned_values >= values - 1e-6 * values.abs()).all())
                timing["pruned"] = {"seconds": seconds, "disagreement": rs.disagreement(labels, pruned, lengths)}
                checks.append({"control": "pruned_dp_on_grid_and_never_below_the_exact_optimum", "on_grid": on_grid,
                               "no_better": no_better, "ok": on_grid and no_better})
    return checks, timing


def smoke_heads():
    """Heads: matched parameters per set, normalized joint log-probabilities over the feasible requests, the
    feasibility mask, a few finite distillation steps on random rows."""
    out = {}
    ok = True
    generator = torch.Generator().manual_seed(11601)
    for name in SETS:
        request_set_ = rs.request_set(name)
        flat_params = rs.parameter_count(flat_reference(request_set_))
        out[name] = {}
        for head in HEADS[name]:
            width = head_width(head, request_set_)
            policy = rs.new_head(head, request_set_, SEEDS[0], width=width, device=DEVICE)
            params = rs.parameter_count(policy)
            step = (rs.head_parameters(head, request_set_, width + 1) - params) if head != "flat" else 0
            z = torch.randn(64, 236, generator=generator).to(DEVICE)
            action = torch.rand(64, 5, generator=generator).to(DEVICE)
            remaining = torch.randint(1, ENDPOINT + 1, (64,), generator=generator).to(DEVICE)
            scores = policy(z, action, remaining)
            deltas = torch.tensor([p.delta for p in request_set_.pairs], device=DEVICE)
            feasible = deltas[None] <= remaining[:, None]
            masked = bool(torch.isinf(scores[~feasible]).all() and torch.isfinite(scores[feasible]).all())
            normalized = head == "flat" or bool((torch.logsumexp(scores, -1).abs() < 1e-4).all())
            labels = torch.multinomial(feasible.float(), 1, generator=torch.Generator(device=DEVICE).manual_seed(1))[:, 0]
            rows = [{"z": z.cpu(), "action": action.cpu(), "remaining": remaining.cpu(), "labels": labels.cpu()}]
            losses = rp.distil_round(policy, rows, 0, seed=SEEDS[0], steps=20, device=DEVICE)
            matched = head == "flat" or abs(params - flat_params) <= abs(step)
            finite = bool(np.isfinite(losses).all())
            out[name][head] = {"width": width, "parameters": params, "flat_parameters": flat_params,
                               "macs_per_call": rp.policy_macs(policy), "masked": masked, "normalized": normalized,
                               "matched": bool(matched), "first_loss": losses[0], "last_loss": losses[-1]}
            ok &= masked and normalized and matched and finite
    return {"control": "policy_heads_matched_masked_normalized", "heads": out, "ok": bool(ok)}


def stand_in_records():
    """Records built from published rows only: every configuration's fixed requests from the Gate-B unit
    (F-15-macro) and #101 fixed records (requests outside R_9 stand in with F-15-macro), every head from the
    Gate-B unit rows with the constant F-15-macro schedule, the wrong-anchor rows of the Gate-B unit record,
    zero engine errors and full label accuracy; one R_18/request seed as a typed failure."""
    _, cells = gb.evaluation_set()
    unit = gb.UNITS["unit"]["request"]
    records = {}
    for seed in SEEDS:
        gate = gate_b_rows(seed)
        published = {r: (gate["inventories"] if r == unit else p101.read_record(POLICY, r, seed)["inventories"])
                     for r in t96.FIXED_ARMS}
        grid_keys = {m: [i["key"] for i in e["inventory"]] for m, e in cells[PRIMARY]["mixed"].items()}
        for config in CONFIGS:
            request_set_ = config_set(config)
            requests = {}
            for k in range(request_set_.size):
                name = request_set_.request_name(k)
                source = published.get(name, published[unit])
                requests[name] = {"inventories": {i: {m: {"rows": compact(source[i][m]["rows"]),
                                                          "engine_mse": [0.0] * len(source[i][m]["rows"]),
                                                          "compute": [0, 0, 0, len(source[i][m]["rows"])]}
                                                      for m in source[i]} for i in INVENTORIES}}
            index = request_set_.index(rs.PredictionPair(15, Abstraction.MACRO))
            heads = {head: {"inventories": {i: {m: {"rows": compact(gate["inventories"][i][m]["rows"]),
                                                    "schedules": [t96.rle([index] * 15)] * len(gate["inventories"][i][m]["rows"]),
                                                    "engine_mse": [0.0] * len(gate["inventories"][i][m]["rows"]),
                                                    "compute": [0, 0, 0, len(gate["inventories"][i][m]["rows"])]}
                                                for m in gate["inventories"][i]} for i in INVENTORIES}}
                     for head in heads_of(config)}
            accuracy = {"heads": {head: {key: [1, 1, 1, 1] for keys in grid_keys.values() for key in keys}
                                  for head in heads_of(config)}}
            failed = config == "R_18/request" and seed == SEEDS[0]
            records[(config, "fixed", seed)] = {"typed_failure": "stand_in" if failed else None, "requests": requests}
            records[(config, "policies", seed)] = {
                "typed_failure": "stand_in" if failed else None, "heads": heads, "label_accuracy": accuracy,
                "wrong_anchor": {a: compact(v["rows"]) for a, v in gate["wrong_anchor"].items()}}
    return records


def stand_in_reports():
    """Label reports of the rehearsal: synthetic timings and counts with the published structure."""
    out = {}
    for config in CONFIGS:
        request_set_ = config_set(config)
        for seed in SEEDS:
            stats = {"counts": [1] * request_set_.size, "delta_counts": [3] * len(request_set_.deltas),
                     "alpha_counts": [len(request_set_.deltas)] * 3, "entropy_bits": 1.0, "entropy_max_bits": 2.0,
                     "delta_entropy_bits": 1.0, "positions": request_set_.size}
            passes = {"gpu_seconds": 1.0 + request_set_.size / 100, "peak_memory_gb": 1.0,
                      "evaluations": 1000 * request_set_.size, **stats}
            report = {"typed_failure": None, "windows": 3480, "requests": request_set_.size, "round0": passes,
                      "timing_window_60": passes,
                      "heads": {h: {"round1": passes, "dagger_seconds": 1.0, "distil_seconds": [1.0, 1.0],
                                    "label_change_round1_vs_round0": 0.1, "macs_per_call": 1,
                                    "in_sample_accuracy": {"round0_rows": {"exact": 0.5}, "round1_rows": {"exact": 0.5}}}
                                for h in heads_of(config)}}
            if request_set_.name == "R_cont":
                report["pruned"] = {**passes, "deltas": list(PRUNED_DELTAS), "requests": 33, "disagreement": 0.1,
                                    "delta_disagreement": 0.1, "alpha_disagreement": 0.0,
                                    "exact_label_on_grid_share": 0.9, "mean_abs_log_delta_ratio": 0.1,
                                    "value_gap": {"mean_relative": 0.01, "max_relative": 0.02, "min_relative": 0.0}}
            out[(config, seed)] = report
    return out


def smoke_statistics():
    """Statistics rehearsal on published rows: R_9's J-flat stand-in reproduces the Gate-B pooled grid AUC, the
    prior reproduces Gate B, the wrong-anchor table reproduces Gate B's, stand-in contrasts are zero, a typed
    failure drops its seed; the disposition rules, every table, the summary and the findings render."""
    _, cells = gb.evaluation_set()
    records = stand_in_records()
    fit = read_json(gb.FIT_VERDICTS)["branches"]
    stats = {i: gb.prior_stats(fit, i) for i in INVENTORIES}
    training = {"members": [{"config": c, "seed": s, "updates": len(config_updates(c)), "gpu_seconds": 1.0, "skipped": 0,
                             "skip_cap": member_skip_cap(c), "usable": True, "typed_failure": None}
                            for c in CONFIGS for s in SEEDS], "typed_failures": []}
    plan = {"frozen_at": "rehearsal", "claim_boundary": "rehearsal"}
    compute = assemble(plan, records, stand_in_reports(), cells, stats, {"G0_rehearsal": True}, True, training,
                       {"gpu_seconds": 0.0, "by_phase": {}, "hardware": "-", "engine_seconds": 0})
    inventories, wrong = compute["inventories"], compute["wrong_anchor"]
    grid = inventories[PRIMARY]
    gate = read_json(GATE_B / "compute.json")
    auc_same = grid["rankers"][head_key(BASELINE, PRIMARY_HEAD)]["auc"] == gate["gate_b"]["pooled_grid_auc"]
    prior_same = all(inventories[i]["rankers"]["prior"]["auc"] == gate["inventories"][i]["rankers"]["prior"]["auc"]
                     for i in INVENTORIES)
    reference = gate["controls"]["wrong_anchor"]
    ours = wrong["configs"][BASELINE]
    exact = all(ours[k] == reference[k] for k in ("right", "wrong", "right_minus_wrong"))
    stability = max(abs(a - b) for a, b in zip([ours["stability"]["mean"], *ours["stability"]["interval"]],
                                               [reference["stability"]["mean"], *reference["stability"]["interval"]]))
    zero = [grid["contrasts"][primary_contrast_name(c)]["J-flat"]["estimate"] for c in TRAINED_CONFIGS]
    zero_ok = all(max(abs(v) for v in (b["mean"], *b["interval"])) <= STABILITY_TOLERANCE for b in zero)
    dropped = grid["contrasts"][primary_contrast_name("R_18/request")]["J-flat"]["cells"]
    drop_ok = dropped == grid["cells"] - grid["members"]
    ranking = compute["dispositions"]["larger_or_continuous_delta_improves_ranking"]["token"]
    tractable = compute["dispositions"]["dp_labeller_tractable_at_225"]["token"]
    json_text(compute)
    json_text(summary(plan, compute))
    findings_md(plan, compute)
    return {"control": "statistics_rehearsal_reproduces_gate_b", "unit_auc": auc_same, "prior": prior_same,
            "wrong_anchor_exact": exact, "stability_max_abs_delta": stability, "stand_in_contrasts_zero": zero_ok,
            "typed_failure_drops_one_seed": drop_ok, "ranking_rule_on_zero_contrasts": ranking,
            "dp_rule_on_stand_in": tractable,
            "ok": bool(auc_same and prior_same and exact and stability <= STABILITY_TOLERANCE and zero_ok and drop_ok
                       and ranking == "not_supported_by_this_experiment" and tractable == "supported")}


def smoke_pipeline():
    """The label and scoring stages end to end on training data only: the policy-pool windows of the first two
    levels with a mixed cell, the Gate-B seed-1 weights in the R_9 and R_cont predictor classes, every head's two
    rounds; fixed and policies records on those levels' policy-split cells (anchors of the #101 pool), label
    accuracy on their windows; nothing kept."""
    scratch = OUTPUT / "smoke-scratch" / "pipeline"
    seed = SEEDS[0]
    original, _ = gb.load_model("unit", seed)
    full = pool_windows()
    entries = p101.policy_cell_entries()
    levels = []
    for member, _ in entries:
        if member not in levels:
            levels.append(member)
    levels = levels[:2]
    units = [unit for unit in full["units"] if full["members"][unit[0]] in levels]
    index = [i for unit in units for i in unit]
    offsets = np.cumsum([0] + [len(u) for u in units])
    pool = {"z": full["z"][index], "action": full["action"][index], "length": full["length"][index],
            "units": [list(range(offsets[k], offsets[k + 1])) for k in range(len(units))], "sha256": full["sha256"]}
    cells = {i: {"mixed": {}} for i in INVENTORIES}
    for (member, inventory), entry in entries.items():
        if member in levels:
            cells[inventory]["mixed"][member] = entry
    anchors = p101.load_pool(POLICY)["anchors"]
    keys = [full["keys"][i] for i in index]
    accuracy_inputs = (keys, pool["z"], pool["action"], pool["length"])
    checks, reports = [], {}
    for config in ("R_9", "R_cont/total"):
        model = rs.load_predictor(config_set(config), original.state_dict(), DEVICE)
        report = label_task(scratch, config, seed, pool, predictor=(model, None, "smoke"))
        write_json(labels_path(scratch, config, seed), report)
        with harness.deterministic_scoring() as scoring_policy:
            fixed = fixed_record(scratch, cells, anchors, config, seed, scoring_policy, predictor=(model, None, "smoke"),
                                 store=False)
            policies = policies_record(scratch, cells, anchors, config, seed, scoring_policy,
                                       predictor=(model, None, "smoke"), store=False, accuracy_inputs=accuracy_inputs)
        json.dumps(fixed)
        json.dumps(policies)
        heads = heads_of(config)
        cells_scored = sum(len(cells[i]["mixed"]) for i in INVENTORIES)
        ok = (set(report["heads"]) == set(heads) and ("pruned" in report) == (config != BASELINE)
              and len(fixed["requests"]) == config_set(config).size and set(policies["heads"]) == set(heads)
              and all(sum(len(policies["heads"][h]["inventories"][i]) for i in INVENTORIES) == cells_scored
                      for h in heads)
              and len(policies["wrong_anchor"]) == len(gb.fit_anchors())
              and all(len(policies["label_accuracy"]["heads"][h]) == len(index) for h in heads))
        checks.append({"control": f"label_and_score_stages_run_end_to_end_{tag(config)}", "windows": len(index),
                       "cells": cells_scored, "heads": {h: {k: report["heads"][h][k] for k in
                                                            ("width", "parameters", "final_loss_0", "final_loss_1",
                                                             "label_change_round1_vs_round0")} for h in heads},
                       "label_seconds": report["wall_seconds"], "fixed_seconds": fixed["seconds"],
                       "policies_seconds": policies["seconds"], "ok": bool(ok)})
        iterations = sum(len(rs.endpoint_schedule(config_set(config).pairs, k)) for k in range(config_set(config).size))
        reports[config] = {**report, "smoke_fixed_seconds_per_cell_iteration": fixed["seconds"] / (cells_scored
                                                                                                   * iterations)}
    shutil.rmtree(scratch, ignore_errors=True)
    return checks, reports


def smoke_isolation():
    handoff = gb.handoff()
    usable = [c["seed"] for c in gb.checkpoints("unit") if c["usable"]]
    hashes = all(sha256_of(gb.member_path("unit", c["seed"])) == c["sha256"] for c in gb.checkpoints("unit"))
    eval_roster = {lv["identity"] for lv in gb.evaluation_levels()}
    fit_roster = {lv["identity"] for lv in rt.fit_levels()}
    payload = rt.load_training_data(RETRAIN)
    pool = p101.load_pool(POLICY)
    ok = (tuple(usable) == SEEDS and hashes and not (set(payload["shot_members"]) & eval_roster)
          and not (set(pool["levels"]) & (eval_roster | fit_roster)))
    return {"control": "handoff_seeds_hashes_and_split_isolation", "handoff_token": handoff["unit"].get("token"),
            "usable_seeds": usable, "hashes": hashes, "policy_levels": len(pool["levels"]), "ok": bool(ok)}


def lpt(costs, workers):
    """Longest-processing-time assignment: {worker: [keys]}, projected seconds per worker."""
    loads = [0.0] * workers
    out = {str(w): [] for w in range(workers)}
    for key, cost in sorted(costs, key=lambda kv: (-kv[1], kv[0])):
        w = min(range(workers), key=lambda k: (loads[k], k))
        out[str(w)].append(key)
        loads[w] += cost
    return out, loads


def projection(training, pipeline, scoring):
    """Upper-bound projection: training at the full-cost update time (seconds(Delta) = a + b / Delta fitted on the
    timing Deltas) plus the first-update (compile) overhead per request; labelling from the pipeline smoke's label
    reports (R_9 and R_cont; DP seconds scaled to the full pool, linear in |R|, plus distillation per head);
    scoring from the smoke's seconds per (cell, transition iteration), every policy step at Delta = 1 (upper
    bound)."""
    deltas = np.array([int(d) for d in training["steady_seconds"]], dtype=float)
    values = np.array(list(training["steady_seconds"].values()))
    b, a = np.polyfit(1 / deltas, values, 1)
    compile_overhead = float(np.mean([training["first_update_seconds"][d] - training["steady_seconds"][d]
                                      for d in training["steady_seconds"]]))
    train_costs = []
    for config in TRAINED_CONFIGS:
        request_set_ = config_set(config)
        updates = config_updates(config)
        per = sum(a + b / request_set_.pairs[k].delta for k in updates) + compile_overhead * request_set_.size
        train_costs += [(f"{config}|{seed}", per) for seed in SEEDS]

    def label_seconds(report):
        windows = report["windows"]
        dp = report["round0"]["gpu_seconds"] + report["timing_window_60"]["gpu_seconds"]
        dp += report.get("pruned", {}).get("gpu_seconds", 0.0)
        dp += sum(h["dagger_seconds"] + h["round1"]["gpu_seconds"] for h in report["heads"].values())
        distil = sum(sum(h["distil_seconds"]) for h in report["heads"].values())
        return dp * len(pool_windows()["length"]) / windows, distil

    measured = {name: label_seconds(pipeline[name]) for name in ("R_9", "R_cont/total")}
    label_costs = []
    for config in CONFIGS:
        request_set_ = config_set(config)
        fraction = (request_set_.size - 9) / (225 - 9)
        dp = measured["R_9"][0] + fraction * (measured["R_cont/total"][0] - measured["R_9"][0])
        distil = measured["R_cont/total"][1] / 3 * len(HEADS[request_set_.name])
        label_costs += [(f"{config}|{seed}", dp + distil + 300.0) for seed in SEEDS]
    per_iteration = max(float(np.mean(scoring["F_15_macro_seconds_per_seed"])) / (512 * (ENDPOINT // 15)),
                        *(pipeline[n]["smoke_fixed_seconds_per_cell_iteration"] for n in ("R_9", "R_cont/total")))
    score_costs = []
    for config in CONFIGS:
        request_set_ = config_set(config)
        iterations = sum(len(rs.endpoint_schedule(request_set_.pairs, k)) for k in range(request_set_.size))
        fixed = per_iteration * 512 * iterations
        heads = len(HEADS[request_set_.name])
        policies = per_iteration * 2 * ENDPOINT * (512 * heads + 480) + 600.0
        score_costs += [(f"{config}|{seed}|fixed", fixed) for seed in SEEDS]
        score_costs += [(f"{config}|{seed}|policies", policies) for seed in SEEDS]
    stages = {}
    for name, costs in (("training", train_costs), ("labelling", label_costs), ("scoring", score_costs)):
        assignment, loads = lpt(costs, WORKERS)
        stages[name] = {"assignment": assignment, "projected_seconds_per_worker": loads,
                        "projected_gpu_seconds": float(sum(c for _, c in costs))}
    stages["seconds_per_update_model"] = {"a": float(a), "b_over_delta": float(b), "compile_overhead": compile_overhead}
    stages["projected_elapsed_seconds"] = float(sum(max(stages[n]["projected_seconds_per_worker"])
                                                    for n in ("training", "labelling", "scoring")))
    return stages


def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the smoke is a pre-freeze control")
    began = time.monotonic()
    torch.backends.cudnn.benchmark = True
    checks = [smoke_sets(), smoke_isolation(), smoke_usability(), smoke_heads(), smoke_predictor()]
    data = ScalingTrainData(rt.load_training_data(RETRAIN))
    checks.append(smoke_windows(data))
    checks += smoke_training(data)
    general, training = smoke_general_training(data)
    checks.append(general)
    del data
    torch.cuda.empty_cache()
    dp_checks, dp_timing = smoke_dp()
    checks += dp_checks
    scoring_checks, scoring = smoke_scoring()
    checks += scoring_checks
    checks.append(smoke_statistics())
    pipeline_checks, pipeline = smoke_pipeline()
    checks += pipeline_checks
    projected = projection(training, pipeline, scoring)
    within = projected["projected_elapsed_seconds"] <= ELAPSED_LIMIT_HOURS * 3600
    checks.append({"control": "projection_within_48h_on_4_gpus", "projected_elapsed_seconds":
                   projected["projected_elapsed_seconds"], "ok": bool(within)})
    ok = all(c["ok"] for c in checks)
    shutil.rmtree(output / "smoke-scratch", ignore_errors=True)
    evidence = {"schema": SCHEMA_SMOKE, "identity": IDENTITY, "run_at": utc_now(), "checks": checks,
                "training_timing": training, "dp_timing": dp_timing,
                "scoring_timing": {k: v for k, v in scoring.items() if k != "scoring_policy"},
                "projection": projected, "device": torch.cuda.get_device_name(0),
                "pool_windows_sha256": pool_windows()["sha256"], "wall_seconds": time.monotonic() - began,
                "reading": ("infrastructure assertions only; nothing kept; no statistic of a new model output; the only "
                            "evaluation rows computed reproduce published Gate-B, #101 and #115 records"), "ok": ok}
    write_json(output / "smoke.json", evidence)
    for check in checks:
        log(f"  {check['control']}: {check['ok']}")
    log(f"smoke {'PASSED' if ok else 'FAILED'}; projected elapsed {projected['projected_elapsed_seconds'] / 3600:.1f} h "
        f"on {WORKERS} GPUs (training GPU {projected['training']['projected_gpu_seconds'] / 3600:.1f} h, labelling "
        f"{projected['labelling']['projected_gpu_seconds'] / 3600:.1f} h, scoring "
        f"{projected['scoring']['projected_gpu_seconds'] / 3600:.1f} h)")
    if not ok:
        raise ValueError("smoke control differs; STOP RULE: abort before the freeze")
    return 0


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def input_bindings(output):
    names = {
        "issue_112_retrain_handoff": gb.HANDOFF, "issue_112_retrain_plan": RETRAIN / "plan.json",
        "issue_112_retrain_compute": RETRAIN / "compute.json", "issue_112_fit_verdicts": gb.FIT_VERDICTS,
        "issue_112_fit_anchors": rt.anchors_path(RETRAIN),
        **{f"issue_112_retrain_updates_{f}_{i}": rt.updates_path(RETRAIN, f, i) for f, i, _ in rt.groups()},
        "gate_b_plan": GATE_B / "plan.json", "gate_b_compute": GATE_B / "compute.json",
        "gate_b_anchors": gb.anchors_path(GATE_B),
        **{f"gate_b_record_unit_{s}": gb.record_path(GATE_B, "unit", s) for s in SEEDS},
        "issue_101_plan": POLICY / "plan.json", "issue_101_pool": p101.pool_path(POLICY),
        **{f"issue_101_record_{r}_{s}": p101.record_path(POLICY, r, s) for r in (*t96.FIXED_ARMS, p101.J_ICLR)
           for s in SEEDS},
        **{f"issue_101_policy_pe_{s}": p101.policy_path(POLICY, "pe", LAMBDA, s) for s in SEEDS},
        "issue_115_engine_truth": TRUTH, "issue_115_plan": TRUTH.parents[1] / "plan.json",
        "issue_117_ranking_runner": ROOT / rk.RUNNER,
        "gate_b_runner": ROOT / gb.RUNNER, "issue_112_retrain_runner": ROOT / rt.RUNNER,
        "issue_112_v1_runner": ROOT / v1.RUNNER, "issue_112_versions_runner": ROOT / versions.RUNNER,
        "issue_101_runner": ROOT / p101.RUNNER,
        "issue_99_attribution_runner": ROOT / "scripts/run_decision_chain_attribution.py",
        "issue_112_controls_runner": ROOT / "scripts/run_decision_dynamics_controls.py",
        "issue_96_runner": ROOT / "scripts/run_tau_ad_within_checkpoint.py",
        "issue_99_encoder": s99.OUTPUT / "encoder" / "encoder.pt",
        "issue_109_cohort_plan": rt.COHORT, "issue_104_plan": rt.POWER_PLAN,
        "issue_104_campaign_branches": rt.CAMPAIGN / "branches.json",
        "scoring_harness": ROOT / "world_model/planning/scoring_harness.py",
        "decision_dynamics": ROOT / "world_model/training/decision_dynamics.py",
        "stable_decision_dynamics": ROOT / "world_model/training/stable_decision_dynamics.py",
        "request_policy": ROOT / "world_model/training/request_policy.py",
        "matched_dynamics": ROOT / "world_model/training/matched_dynamics.py",
        "cnn_hybrid": ROOT / "world_model/training/cnn_hybrid.py", "predictor": ROOT / "world_model/model/predictor.py",
        "request_sets": ROOT / MODULE, "smoke": Path(output) / "smoke.json",
    }
    names.update({f"checkpoint_unit_{c['seed']}": gb.member_path("unit", c["seed"]) for c in gb.checkpoints("unit")})
    return [{"name": name, "artifact": str(Path(path).relative_to(ROOT)), "sha256": sha256_of(path)}
            for name, path in names.items()]


def frozen_rules():
    return {
        "disposition_ranking": (
            "'a larger or continuous Delta set improves ranking over R_9': per set R in {R_18, R_30, R_cont} and budget "
            "b in {total, request}, the grid tie-free AUC of the flat-head policy J-flat(R, b) minus J-flat(R_9), paired "
            "by (level, seed), level-clustered bootstrap; delta superiority token (mean >= 0.02 and lower > 0 "
            "supported; upper < 0.02 not_supported_by_this_experiment; else readiness_or_precision_insufficient). "
            "Overall: supported iff some set is supported under both budgets; not_supported_by_this_experiment iff "
            "every (set, budget) token is not_supported_by_this_experiment; otherwise "
            "readiness_or_precision_insufficient"),
        "disposition_dp": (
            f"'the DP labeller stays tractable at |R| = 225': supported iff every usable R_cont predictor's exact "
            f"round-0 prediction-error DP over the #109 policy pool (N = 3480 start-0 windows, L = 225 frames, "
            f"chunk {DP_CHUNK}) completes on one RTX 5090 within {TRACTABLE_SECONDS:.0f} GPU s and its seconds per "
            f"carrier evaluation are at most {TRACTABLE_UNIT_RATIO:g} x the R_9 mean (no super-linear growth in |R|); "
            "not_supported_by_this_experiment iff any R_cont pass fails either; readiness_or_precision_insufficient "
            "if no R_cont predictor is usable"),
        "guard_failure": f"a failed guard turns every token into {STOP_TOKEN}",
        "secondary": ("DESCRIPTIVE with the same delta token reported: heads (J-factorised / J-ordinal - J-flat per "
                      "set), the fixed R_9-request mean of each set - R_9, J-flat - prior, right - wrong anchor; offset, "
                      "angle and power; rollout error, label accuracy, MACs, Delta histograms, DP cost"),
        "pairing": ("cells (level, seed): seed s of every configuration pairs with Gate-B seed s; a typed-failure member "
                    "drops its seed's cells from every contrast that reads it; the bootstrap resamples levels"),
        "sub_frame_delta": SUBFRAME,
    }


def frozen_plan(output, frozen_at, smoke_evidence):
    branches, cells = gb.evaluation_set()
    long, rank = v3_blocks()
    widths = {s: {h: head_width(h, rs.request_set(s)) for h in HEADS[s]} for s in SETS}
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "issue": 116,
        "related_issues": [101, 104, 108, 109, 111, 112, 114, 117, 119, 121],
        "role": ("#116 Work 1-6 in full, including the ranking comparison across request sets, confirmatory on the #104 "
                 "sealed #109 evaluation split after Gate B passed"),
        "frozen_at": frozen_at, "frozen_before_training_and_scoring": True,
        "validation_command": VALIDATION_COMMAND, "runner": RUNNER, "runner_sha256_at_freeze": sha256_of(ROOT / RUNNER),
        "module": MODULE, "module_sha256_at_freeze": sha256_of(ROOT / MODULE), "engine_seconds": 0,
        "question": ("what happens to rollout error, ranking, DP labelling cost and policy accuracy when the request set "
                     "grows beyond R_9 = {1, 5, 15} x alpha, up to an effectively continuous Delta?"),
        "request_sets": {s: {"deltas": list(rs.DELTA_GRIDS[s]), "alphas": [str(a) for a in rs.ALPHAS],
                             "size": rs.request_set(s).size, "order": "Delta-major, alpha continuous/micro/macro "
                                                                       "(R_9 = cnn_hybrid.PAIRS)"} for s in SETS},
        "generalisation": {
            "module": MODULE,
            "predictor": ("request_sets.predictor_class(R): CNNHybridPredictor whose carrier accepts exactly R; same "
                          "parameters (1 837 690), state dict and ops; PairConditioner encodes any integer Delta"),
            "replication_guard": ("smoke: R_9 training equals the retrain train_group bit for bit (schedule and full "
                                  "cost, 12 updates); the R_9 fixed-request rows equal the Gate-B unit and #101 records; "
                                  "the #101 J reproduces through the generalised rollout; the DP labels equal "
                                  "request_policy.window_rows in both DAgger rounds; G5 re-checks every R_9 fixed row")},
        "training": {
            "recipe": ("#112 v3 LC (the Gate-B recipe): #77 local loss, long-horizon recursion to the curriculum "
                       "horizon, decision-contrastive ranking, truncation every 45 transitions, skip-on-nonfinite "
                       "ensemble (patience 10); fold 'all', one member per (configuration, seed), seeds "
                       f"{list(SEEDS)}"),
            "recipe_constants": {"s99": training_recipe(), "long": long, "rank": rank},
            "schedule": "round robin over R in declared order (R_9: PAIRS[step % 9], the retrain schedule)",
            "budgets": {"total": "9000 updates (the Gate-B count): 9000 / |R| updates per request",
                        "request": "1000 x |R| updates (1000 per request, the Gate-B joint's per-request count)",
                        "update_counts": {c: len(config_updates(c)) for c in CONFIGS},
                        "R_9": "both budgets are the Gate-B joint (9000 updates, 1000 per request): not retrained"},
            "curriculum_clock": ("update u of an n-update run reads the horizon curriculum and the ranking ramp at "
                                 "u * 9000 // n (identity at 9000 updates)"),
            "local_window": ("an update of Delta reads a window of max(60, 4 Delta) transitions at the drawn window "
                             "start (same index draw), so every Delta keeps four local transitions where the shot "
                             "allows; Delta <= 15 is the payload window"),
            "long_horizon": "floor(horizon / Delta) transitions of the trained pair; none when Delta > horizon",
            "endpoint_rule": ("fixed requests (ranking term in training and every fixed-request scoring rollout): "
                              "floor(225 / Delta) transitions of the request, then the #101 same-alpha fallback (the "
                              "largest Delta' in R with Delta' <= remaining) until exactly 225; Delta | 225 is the "
                              "fixed request itself. Policies mask Delta > remaining"),
            "usable": (f"complete, at most {SKIP_CAP} skipped updates per 9000 (scaled to the member's updates), every "
                       "update logged, no #117 end-of-training excursion (rt.excursion over its own schedule); an "
                       "unusable member is a typed terminal failure, never retried or replaced, and gets no policy"),
            "data": {"payload": str(rt.data_path(RETRAIN).relative_to(ROOT)),
                     "definition": "the retrain's #109 fit payload (13 919 shots, 680 ranking cells)"}},
        "dp_labels": {
            "kind": "prediction error (ICLR eq. dp; #101 pe): c(t, i) = Delta_i x one-step carrier MSE + lambda x "
                    "work(pair_i, flat policy of R) / work(15, continuous)",
            "lambda": LAMBDA,
            "pool": "the #109 policy split as built by #101 (3480 shots, 120 levels, E99 carriers)",
            "window": (f"start-0 windows of {LABEL_WINDOW} frames (rp.first_window rule, terminal absorption), the "
                       "deployment horizon, so every Delta <= 75 is feasible"),
            "mac_term": "the set's flat policy (width 128) for every head, so round-0 labels depend on (predictor, R) only",
            "chunk": DP_CHUNK, "timing_pass": f"exact round-0 labels on the first {TIMING_WINDOW} frames (cost only)",
            "pruned": {"deltas": list(PRUNED_DELTAS),
                       "rule": "round-half-up of 12 geometric points from 1 to 75; candidate requests = grid x alpha; "
                               "R_cont only; disagreement with the exact labels on every labelled position"}},
        "policies": {
            "heads": {s: list(HEADS[s]) for s in SETS},
            "definitions": {"flat": "request_policy.RequestPolicy(R, 128, clamp60): |R|-way (the current design)",
                            "factorised": "trunk -> Delta head x alpha head; joint log-probabilities",
                            "ordinal": "trunk -> cumulative-logit Delta head (one score, ordered thresholds) x alpha head"},
            "widths": widths,
            "parameters": {s: {h: rs.head_parameters(h, rs.request_set(s), widths[s][h]) for h in HEADS[s]}
                           for s in SETS},
            "matching": "factorised / ordinal hidden width chosen so the parameter count is closest to the set's flat head",
            "recipe": ("#101 pe: torch.manual_seed(seed + 7700) init; round 0 on observed-context labels; round 1 on the "
                       "head's own DAgger contexts; rp.distil_round (2 x 1800 steps, batch 128, AdamW 1e-3, unit = "
                       "policy level); cross-entropy on the joint scores (sum of factor NLLs for factorised / ordinal)"),
            "remaining_feature": REMAINING_FEATURE, "seed_pairing": "policy seed s uses predictor seed s"},
        "evaluation": {
            "split": "the #109 evaluation split as in Gate B (stream issue109-evaluation, mixed cells)",
            "accounting": gb.evaluation_accounting(branches, cells), "anchors": "the Gate-B E99 anchor store",
            "primary_inventory": PRIMARY, "descriptive": list(gb.DESCRIPTIVE_INVENTORIES),
            "arms": ("per configuration and seed: every fixed request of R (endpoint rule) and every head's greedy "
                     "policy J (joint argmax); the wrong-anchor control of J-flat (16 grid candidates from the 480 fit "
                     "anchors)"),
            "cost": "tie_free (EndpointCosts.row against the anchor); cell AUC m99.view",
            "rollout_error": ("#100 engine_position_mse of every mixed grid candidate's endpoint (t = 225) against its "
                              "own branch at min(225, last frame), the #115 engine-truth store (hash-bound)"),
            "label_accuracy": ("exact DP labels of the head's predictor on every grid evaluation branch (E99 parses "
                               "0..min(225, last) from the #115 store, observed contexts, remaining = frames to the "
                               "window end); share of positions whose greedy request equals the label (exact, Delta, "
                               "alpha); branch mean, level mean over seeds, level bootstrap"),
            "macs": "linear MACs per candidate rollout (predictor + policy), policy MACs per call, Delta histograms"},
        "decision_rule": frozen_rules(),
        "guards": {"G1_isolation": "fit payload members, policy-pool levels and evaluation levels are disjoint; wrong "
                                   "anchors are fit levels",
                   "G2_scoring_policy": "every record and every label report carries the #111 SCORING_POLICY",
                   "G3_chronology": "frozen_at <= every training; every training and labelling <= the first score",
                   "G4_binding": ("predictor hashes = training records (R_9: handoff); policy hashes = label reports; "
                                  "training payload, policy-pool windows and truth store = the frozen hashes"),
                   "G5_replication": "every R_9 fixed-request row equals the Gate-B unit / #101 records (all seeds and "
                                     "inventories)",
                   "G6_training_integrity": "every member's schedule is its frozen schedule and every update is logged",
                   "G7_nonfinite": f"every J-flat grid nonfinite share <= {NONFINITE_SHARE_MAX}",
                   "failure": f"a failed guard turns every token into {STOP_TOKEN}"},
        "data": {"training_data_sha256": expected_payload_sha(),
                 "policy_pool_windows_sha256": smoke_evidence["pool_windows_sha256"],
                 "truth_store_sha256": sha256_of(TRUTH)},
        "execution": {"workers": WORKERS, "training": smoke_evidence["projection"]["training"],
                      "labelling": smoke_evidence["projection"]["labelling"],
                      "scoring": smoke_evidence["projection"]["scoring"],
                      "rule": ("each stage launched once (one process per GPU, worker PIDs recorded under logs/), then "
                               "waited on; no relaunch over live workers; a crashed worker resumes its own assignment "
                               "(training resumes from its progress checkpoint)")},
        "caps": {"gpu_seconds": GPU_CAP_SECONDS, "elapsed_limit_hours": ELAPSED_LIMIT_HOURS},
        "records": ("records, label files, update logs and checkpoints stay local; records-manifest.json (sha256 of every "
                    "record, label report, round-0 label file and update log) is committed"),
        "prohibited": ["reading evaluation captures for training or selection", "re-scoring after publication",
                       "changing a set, budget, head, rule or recipe after the freeze",
                       "retrying or replacing a typed failure"],
        "disclosure": ("Known before the freeze: Gate B (unit 0.7444, C* 0.7834, prior 0.7203 on grid); #101 (J 0.706, "
                       "fixed F-15-continuous 0.762, F-15-macro 0.744 on grid); #117 (full-budget Delta = 5 / 15 "
                       "specialists rank above the joint by 0.05-0.14; Delta = 1 joint rows barely use the anchor). No "
                       "predictor or policy over a set other than R_9 existed, and no #109 policy trained on 225-frame "
                       "labels. The smoke rescored only published models on the evaluation cells (Gate-B unit, #101 "
                       "fixed requests and J, #115 F-15-micro) to reproduce their records. Hardware: RTX 5090 instead "
                       "of the #85 RTX 3090."),
        "smoke_evidence": smoke_evidence,
        "inputs": input_bindings(output),
        "claim_boundary": ("one look at the sealed #109 evaluation split; LC (#112 v3) recipe only; prediction-error DP "
                           "labels (lambda 1e-4) only; grid primary; tie-free cost; DESCRIPTIVE intervals; NovPhy only; "
                           "sub-frame Delta unsupported"),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-116 protocol")
    if not plan.get("frozen_before_training_and_scoring") or not plan["smoke_evidence"].get("ok"):
        raise ValueError("plan.json does not declare a passed-smoke freeze")
    for entry in plan["inputs"]:
        if not amendments.matches(entry["artifact"], entry["sha256"], sha256_of(ROOT / entry["artifact"])):
            raise ValueError(f"frozen input {entry['name']} changed after the freeze")
    return plan


def prepare(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        load_plan(output)
        log("existing frozen plan validated")
        return 0
    for name in ("records", "models", "training", "labels", "policies", "updates"):
        if (output / name).exists():
            raise ValueError(f"{name} exist before the freeze; refusing")
    evidence = read_json(output / "smoke.json")
    if not evidence.get("ok"):
        raise ValueError("smoke failed; STOP RULE: no freeze")
    plan = frozen_plan(output, utc_now(), evidence)
    (output / "plan.json").write_bytes(json_text(plan).encode())
    load_plan(output)
    log("frozen plan published")
    return 0


# ---------------------------------------------------------------------------
# publication and validation
# ---------------------------------------------------------------------------

def publish(output):
    output = Path(output)
    plan = load_plan(output)
    if not scoring_complete(output):
        raise ValueError("scoring is not complete")
    compute = compute_tables(output, plan)
    for name, text in rendered(plan, compute, output).items():
        (output / name).write_bytes(text.encode())
    log(f"published: {summary(plan, compute)['dispositions']}; guards pass {compute['guards_pass']}")
    return 0


def spot_checks(output):
    """Re-score in this process (exact rows): R_cont/total J-flat and R_18/request F-2-macro on every grid cell
    (first usable seed)."""
    _, cells = gb.evaluation_set()
    anchors = gb.load_anchors(GATE_B)["anchors"]
    costs = p101.endpoint_costs()
    results = {}
    with harness.deterministic_scoring():
        for config, arm in (("R_cont/total", "J-flat"), ("R_18/request", "F-2-macro")):
            seed = next((s for s in SEEDS if not read_record(output, config, "fixed", s)["typed_failure"]), None)
            if seed is None:
                results[f"{config}:{arm}"] = None
                continue
            model, _, _ = load_member(output, config, seed)
            request_set_ = config_set(config)
            same = True
            if arm == "J-flat":
                record = read_record(output, config, "policies", seed)
                policy = load_policy(output, config, PRIMARY_HEAD, seed, record["heads"][PRIMARY_HEAD]["policy_sha256"])
                for member, entry in sorted(cells[PRIMARY]["mixed"].items()):
                    fresh = rollout_block(model, policy, {"kind": "joint"}, anchors[member], entry["inventory"], costs,
                                          schedules=True)
                    kept = record["heads"][PRIMARY_HEAD]["inventories"][PRIMARY][member]
                    same &= fresh["rows"] == kept["rows"] and fresh["schedules"] == kept["schedules"]
            else:
                record = read_record(output, config, "fixed", seed)
                k = set_requests(config).index(arm)
                spec = {"kind": "schedule", "schedule": rs.endpoint_schedule(request_set_.pairs, k)}
                holder = types.SimpleNamespace(pairs=request_set_.pairs)
                for member, entry in sorted(cells[PRIMARY]["mixed"].items()):
                    fresh = rollout_block(model, holder, spec, anchors[member], entry["inventory"], costs)
                    same &= fresh["rows"] == record["requests"][arm]["inventories"][PRIMARY][member]["rows"]
            results[f"{config}:{arm}"] = {"seed": seed, "pass": bool(same)}
    return {"checks": results, "pass": all(v is None or v["pass"] for v in results.values())}


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    compute = compute_tables(output, plan)
    for name, text in rendered(plan, compute, output).items():
        if (output / name).read_bytes() != text.encode():
            raise ValueError(f"published {name} differs from the recomputation")
    spot = spot_checks(output)
    if not spot["pass"]:
        raise ValueError(f"deterministic re-scoring does not replicate the retained records: {spot}")
    log(f"validation passed ({time.monotonic() - began:.0f}s): tables, summary, findings and the manifest re-derived "
        f"and byte-compared; re-scored {spot['checks']}; dispositions {summary(plan, compute)['dispositions']}")
    return 0


def dry_run(output):
    _, cells = gb.evaluation_set()
    log(f"output {Path(output)}; mixed cells { {i: len(cells[i]['mixed']) for i in INVENTORIES} }")
    for config in CONFIGS:
        request_set_ = config_set(config)
        log(f"  {config}: |R| {request_set_.size}, updates {len(config_updates(config))}, heads {heads_of(config)} "
            f"widths { {h: head_width(h, request_set_) for h in heads_of(config)} }")
    log(f"  pruned Delta grid {list(PRUNED_DELTAS)}; plan present {(Path(output) / 'plan.json').is_file()}")
    return 0


MODES = {"dry-run": dry_run, "smoke": smoke, "prepare": prepare, "train": train, "label": label, "score": score,
         "publish": publish, "validate": validate}
SHARDED = {"train", "label", "score"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker", type=int, default=0)
    args = parser.parse_args()
    mode = next(name for name in MODES if getattr(args, name.replace("-", "_")))
    try:
        if mode in SHARDED:
            return MODES[mode](OUTPUT, args.workers, args.worker)
        return MODES[mode](OUTPUT)
    except (ValueError, OSError, KeyError) as error:
        log(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
