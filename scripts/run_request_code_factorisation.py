"""Issue-121: request-code factorisation (fused MLP code vs additive vs factored-FiLM), scored once on the #104
sealed #109 evaluation split.

Binding runner module: scripts/run_request_code_factorisation.py
Exact validation command: python -u -m scripts.run_request_code_factorisation --validate

The CNN hybrid conditions its shared trunk on the request r = (Delta, alpha) through one 128-d code,
c = MLP[phi(Delta); e_alpha] (``PairConditioner``), which feeds zero-initialised FiLM scale / shift generators
in all three trunk blocks (``cnn_hybrid.CNNHybridPredictor.carrier``). alpha also selects the input adapter;
that path is the same in every arm. This runner ablates the code (``world_model/training/request_code.py``):

1. arms: (a) fused, the Gate-B LC hybrid checkpoints (the control; the training path of the new arms,
   configured as fused, reproduces the retrain's ``train_group`` bit for bit in the smoke), (b) additive,
   c = W_delta phi(Delta) + W_alpha e_alpha, (c) factored-FiLM, per-axis FiLM generators composed as
   (1 + g_d)(1 + g_a) and b_d + b_a. (b) and (c) are trained under the #112 v3 LC recipe (the Gate-B recipe:
   9000 updates, fold 'all', seeds 20260908/09/10) on the #109 fit split (the retrain payload, hash-checked);
2. the coupling diagnostic: per block, a two-way decomposition of the FiLM scale and shift over the nine
   requests of every trained model (fused: the Gate-B checkpoints; no training);
3. per arm: rollout error (#100 engine_position_mse, every mixed grid candidate, endpoints t = 15, 30, 60, 150,
   225) of the nine fixed requests; per-inventory ranking AUC of the nine fixed requests and of a J trained per
   arm (#101 pe DP, lambda 1e-4, the #109 policy pool; fused: the #101 J); linear MACs;
4. the no-model action prior and the wrong-anchor control in every ranking table.

Modes: --dry-run --smoke --prepare --train --truth --score --publish --validate. --train and --score take
--workers N --worker K (one process per GPU; CUDA_VISIBLE_DEVICES set by the caller). Zero engine seconds.
Intervals: level-clustered percentile bootstrap (10000 draws, PCG64 7201), DESCRIPTIVE; delta = 0.02.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import functools
import importlib.util
import math
import multiprocessing
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
from scripts import run_request_interference as r117
from scripts import run_slot_encoder_fix as s99
from scripts import run_tau_ad_within_checkpoint as t96
from world_model.planning import scoring_harness as harness
from world_model.training import request_code as rc
from world_model.training import request_policy as rp
from world_model.training.cnn_hybrid import PAIRS, linear_macs

ROOT = v1.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-121-request-code-v1"
GATE_B = gb.OUTPUT
RETRAIN = rt.OUTPUT
POLICY = p101.OUTPUT
TRUTH115 = ROOT / ".local-artifacts/issue-115-predicate-feedback-v1"

IDENTITY = "issue-121-request-code-v1"
SCHEMA_PLAN = "issue_121_request_code_plan_v1"
SCHEMA_SMOKE = "issue_121_request_code_smoke_v1"
SCHEMA_PREDICTOR = "issue_121_predictor_v1"
SCHEMA_GROUP = "issue_121_training_group_v1"
SCHEMA_UPDATES = "issue_121_updates_v1"
SCHEMA_TRAINED = "issue_121_trained_v1"
SCHEMA_POLICY = "issue_121_request_policy_v1"
SCHEMA_TRUTH = "issue_121_engine_truth_v1"
SCHEMA_RECORD = "issue_121_record_v1"
SCHEMA_COMPUTE = "issue_121_compute_v1"
SCHEMA_REPORT = "issue_121_report_v1"
SCHEMA_MANIFEST = "issue_121_records_manifest_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_request_code_factorisation --validate"
RUNNER = "scripts/run_request_code_factorisation.py"
MODULE = "world_model/training/request_code.py"

DEVICE = "cuda"
ARM = rt.ARM  # the #112 v3 LC recipe
SEEDS = tuple(t96.SEEDS)
INVENTORIES = gb.INVENTORIES
PRIMARY = gb.GATE_INVENTORY
PRIMARY_COST = harness.PRIMARY_COST
DELTA = v1.DELTA
STOP_TOKEN = v1.STOP_TOKEN
NONFINITE_SHARE_MAX = v1.NONFINITE_SHARE_MAX
MIN_CLUSTERS = r117.MIN_CLUSTERS
ENDPOINT = t96.ENDPOINT
TIMES = v1.TIMES                       # (15, 30, 60, 150, 225): the #112 / #117 rollout endpoints
STEPS = rt.STEPS
SKIP_CAP = rt.SKIP_CAP
ICLR_LAMBDA = p101.ICLR_LAMBDA
CODE_ARMS = rc.ARMS                    # fused, additive, factored
CONTROL = "fused"
TRAINED = ("additive", "factored")
REQUESTS = t96.FIXED_ARMS              # F-{1,5,15}-{continuous,micro,macro}, the order of cnn_hybrid.PAIRS
RANKERS = (*REQUESTS, "J")
COMPARISONS = (("fused", "additive"), ("fused", "factored"), ("factored", "additive"))
CLAIM = r117.CLAIM                     # 1.5x, the #117 criterion
TRUTH_WORKERS = 24
SMOKE_STEPS = 12
SMOKE_WARMUP = 9
SMOKE_CELLS = 3
GPU_CAP_SECONDS = 24 * 3600.0
WALL_CAP_SECONDS = 48 * 3600.0         # summed over processes
ELAPSED_LIMIT_HOURS = 48.0             # the owner's limit on 4 GPUs: beyond it the ticket finishes blocked
COST_WEIGHT = {"F-1": 3.0, "J": 2.5, "F-5": 1.5, "wrong": 1.0, "F-15": 0.5}

read_json = v1.read_json
write_json = v1.write_json
json_text = v1.json_text
sha256_of = v1.sha256_of
atomic_torch = v1.atomic_torch
fmt_interval = v1.fmt_interval
bootstrap = v1.bootstrap


def log(message):
    print(f"[issue-121] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# paths and ledger
# ---------------------------------------------------------------------------

def updates_path(output, arm, seed):
    return Path(output) / "training" / f"updates--{arm}--seed{seed}.pt"


def trained_path(output, arm, seed):
    return Path(output) / "training" / f"trained--{arm}--seed{seed}.json"


def policy_path(output, arm, seed):
    return Path(output) / "models" / "policies" / arm / f"seed-{seed}" / "policy.pt"


def policy_report_path(output, arm, seed):
    return Path(output) / "models" / "policies" / arm / f"seed-{seed}" / "report.json"


def truth_path(output):
    return Path(output) / "data" / "engine-truth.pt"


def record_path(output, arm, ranker, seed):
    return Path(output) / "records" / f"{arm}--{ranker}--seed{seed}.json"


def wrong_path(output, arm, seed):
    return Path(output) / "records" / f"wrong--{arm}--J--seed{seed}.json"


def ledger_path(output, phase):
    return Path(output) / "ledger" / f"{phase}.json"


def mark(output, phase, wall, gpu, **extra):
    path = ledger_path(output, phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    block = read_json(path) if path.is_file() else {"wall_seconds": 0.0, "gpu_seconds": 0.0}
    write_json(path, {**block, "wall_seconds": block["wall_seconds"] + wall, "gpu_seconds": block["gpu_seconds"] + gpu,
                      "last_at": utc_now(), **extra})
    total = [read_json(p) for p in sorted((Path(output) / "ledger").glob("*.json"))]
    if (sum(b["gpu_seconds"] for b in total) > GPU_CAP_SECONDS
            or sum(b["wall_seconds"] for b in total) > WALL_CAP_SECONDS):
        raise ValueError(f"{STOP_TOKEN}: compute cap exceeded")


# ---------------------------------------------------------------------------
# training: the #112 v3 LC recipe (rt.implementation) with the arm's predictor class
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def implementation(arm):
    """A private copy of the v1 runner carrying the frozen v3 spec exactly as ``rt.implementation()`` (LC arm,
    v3 LONG / RANK, truncated recursion, skipping + logging ensemble, ``rt.logged_local_loss``, PAIRS[step % 9]),
    bound to this output, whose ``s99.new_predictor`` builds the arm's class (``request_code.new_predictor``).
    The arm name is the "family" of the member paths; the step schedule is the hybrid one."""
    v3 = versions.implementation(rt.VERSION)
    loader = importlib.util.spec_from_file_location(f"issue121_implementation_{arm}", ROOT / v1.RUNNER)
    impl = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(impl)
    impl.OUTPUT = OUTPUT
    impl.IDENTITY = IDENTITY
    impl.SCHEMA_PREDICTOR = SCHEMA_PREDICTOR
    impl.SCHEMA_GROUP = SCHEMA_GROUP
    impl.SCHEMA_DATA = rt.SCHEMA_DATA
    impl.RUNNER = RUNNER
    impl.MODEL_MODULE = MODULE
    impl.ARMS = (ARM,)
    impl.ARM_DEFINITIONS = {ARM: v3.ARM_DEFINITIONS[ARM]}
    impl.LONG = v3.LONG
    impl.RANK = v3.RANK
    impl.dd = types.SimpleNamespace(Ensemble=rt.logging_ensemble(v3.dd.Ensemble), local_loss=rt.logged_local_loss,
                                    horizon_at=v3.dd.horizon_at, ramp_at=v3.dd.ramp_at,
                                    long_and_rank_loss=v3.dd.long_and_rank_loss)
    impl.step_pair = lambda family, step: v1.step_pair("hybrid", step)
    predictors = types.SimpleNamespace(**vars(s99))
    predictors.new_predictor = lambda family: rc.new_predictor(arm)
    impl.s99 = predictors
    return impl


def member_path(output, arm, seed):
    return implementation(arm).member_path(output, ARM, arm, "all", seed)


def payload_hash_ok():
    expected = read_json(RETRAIN / "compute.json")["data"]["training_data_sha256"]
    return sha256_of(rt.data_path(RETRAIN)) == expected


def train_member(output, arm, seed, data):
    impl = implementation(arm)
    began = time.monotonic()
    report, ensemble = impl.train_group(output, ARM, arm, SEEDS.index(seed), [("all", seed)], data)
    atomic_torch(updates_path(output, arm, seed), {
        "schema": SCHEMA_UPDATES, "plan_identity": IDENTITY, "arm": arm, "seed": seed,
        "requests": rt.step_requests("hybrid"), "skipped": ensemble.skipped.cpu(), "logs": ensemble.log_tensors()})
    del ensemble
    torch.cuda.empty_cache()
    write_json(trained_path(output, arm, seed), {
        "schema": SCHEMA_TRAINED, "plan_identity": IDENTITY, "arm": arm, "seed": seed, "trained_at": utc_now(),
        "gpu_seconds": report["gpu_seconds"], "parameters": report["parameters_per_member"],
        "parameter_split": rc.parameter_split(rc.new_predictor(arm), arm),
        "sha256": sha256_of(member_path(output, arm, seed))})
    mark(output, f"train-{arm}-seed{seed}", time.monotonic() - began, report["gpu_seconds"])
    log(f"trained {arm} seed {seed}: {report['gpu_seconds']:.0f}s")


def member_state(output, arm, seed):
    """Training accounting and usability of one member (retrain rule: complete, at most SKIP_CAP skipped
    updates, no #117 end-of-training excursion)."""
    report = read_json(implementation(arm).group_path(output, ARM, arm, SEEDS.index(seed)))
    updates = torch.load(updates_path(output, arm, seed), map_location="cpu", weights_only=True)
    trained = read_json(trained_path(output, arm, seed))
    member = report["members"][0]
    local, norms = updates["logs"]["local"][:, 0].double(), updates["logs"]["grad_norm"][:, 0].double()
    end_state = rt.excursion(local, norms, updates["requests"]) if member["complete"] else None
    skipped = int(updates["skipped"][0])
    usable = bool(member["complete"] and skipped <= SKIP_CAP and len(local) == STEPS and not end_state["excursion"])
    last = report["trace"][-1]
    return {"arm": arm, "seed": seed, "complete": member["complete"], "retired_at": member["retired_at"],
            "skipped": skipped, "logged_updates": len(local), "gpu_seconds": report["gpu_seconds"],
            "parameters": trained["parameters"], "parameter_split": trained["parameter_split"],
            "peak_memory_gb": report["peak_memory_gb"],
            "final_trace": {k: last[k] for k in ("loss", "local", "long", "rank", "grad_norm")},
            "end_state": end_state, "usable": usable, "trained_at": trained["trained_at"], "sha256": trained["sha256"],
            "typed_failure": None if usable else ("retired" if not member["complete"] else
                                                  "skip_cap" if skipped > SKIP_CAP else "final_checkpoint_excursion")}


def train_policy(output, arm, seed, model):
    """The ICLR request policy J of this predictor: the #101 pe recipe at lambda 1e-4 on the #109 policy pool
    (two DAgger rounds of the #77 prediction-error DP labels, distillation), then the policy-split evaluation."""
    began = time.monotonic()
    pool = p101.load_pool(POLICY)
    with harness.deterministic_scoring() as scoring_policy:
        policy = rp.new_policy(PAIRS, seed, device=DEVICE)
        history, stats = [], {}
        for round_index in range(rp.DISTIL_RECIPE["rounds"]):
            rows = [rp.window_rows(model, policy, unit, aggregated=bool(round_index), compute_weight=ICLR_LAMBDA,
                                   device=DEVICE) for unit in p101.window_units(pool)]
            history.append(rows)
            losses = rp.distil_round(policy, rp.aggregate_rounds(history), round_index, seed=seed, device=DEVICE)
            stats[f"final_loss_{round_index}"] = float(np.mean(losses[-60:]))
            stats[f"rows_{round_index}"] = int(sum(len(u["labels"]) for u in rows))
        policy.eval()
        evaluation = p101.policy_split_evaluation(model, policy, pool)
    path = policy_path(output, arm, seed)
    atomic_torch(path, {"schema": SCHEMA_POLICY, "plan_identity": IDENTITY, "arm": arm, "seed": seed,
                        "lambda": ICLR_LAMBDA, "pairs": [str(p.identity) for p in policy.pairs],
                        "width": policy.layers[0].out_features, "remaining_feature": policy.remaining_feature,
                        "scoring_policy": scoring_policy, "model": policy.state_dict()})
    wall = time.monotonic() - began
    write_json(policy_report_path(output, arm, seed), {
        "schema": SCHEMA_POLICY, "plan_identity": IDENTITY, "arm": arm, "seed": seed, "lambda": ICLR_LAMBDA,
        "trained_at": utc_now(), "policy_sha256": sha256_of(path),
        "predictor_sha256": sha256_of(member_path(output, arm, seed)), "stats": stats,
        "policy_split": evaluation, "wall_seconds": wall})
    mark(output, f"policy-{arm}-seed{seed}", wall, wall)
    log(f"J for {arm} seed {seed}: {wall:.0f}s; policy-split AUC {evaluation['auc_pooled']:.4f}")


def train(output, workers, worker):
    output = Path(output)
    load_plan(output)
    torch.backends.cudnn.benchmark = True
    tasks = [(arm, seed) for seed in SEEDS for arm in TRAINED]
    mine = [t for k, t in enumerate(tasks) if k % workers == worker]
    if not payload_hash_ok():
        raise ValueError("the retrain training payload differs from its published hash")
    data = None
    for arm, seed in mine:
        if not trained_path(output, arm, seed).is_file():
            if data is None:
                data = rt.IndexedTrainData(rt.load_training_data(RETRAIN), "hybrid")
            train_member(output, arm, seed, data)
        if policy_report_path(output, arm, seed).is_file():
            continue
        state = member_state(output, arm, seed)
        if state["typed_failure"]:
            log(f"{arm} seed {seed}: typed failure {state['typed_failure']}; no policy")
            continue
        model = implementation(arm).load_member(output, ARM, arm, "all", seed)
        train_policy(output, arm, seed, model)
        del model
        torch.cuda.empty_cache()
    return 0


def training_complete(output):
    return all(trained_path(output, a, s).is_file() for a in TRAINED for s in SEEDS) and all(
        policy_report_path(output, a, s).is_file() or member_state(output, a, s)["typed_failure"]
        for a in TRAINED for s in SEEDS)


# ---------------------------------------------------------------------------
# engine truth: presence and projected centres of every mixed grid branch at min(t, last), t in TIMES
# ---------------------------------------------------------------------------

def truth_worker(attempt):
    """Process-pool worker: the #99 engine labels (s99.shot_labels) at positions min(t, last) for t in TIMES."""
    torch.set_num_threads(1)
    _, _, frames, _ = s99.shot_frames(attempt)
    segment = read_json(Path(attempt) / "shot-1" / "segment.json")
    last = min(ENDPOINT, len(frames) - 1)
    labels = s99.shot_labels(segment, frames, [min(t, last) for t in TIMES], s99.vocabulary())
    return {"last": last, "frames": len(frames), "presence": labels["presence"].bool(),
            "centers": labels["centers"].float()}


def grid_keys(cells):
    return sorted({item["key"] for entry in cells[PRIMARY]["mixed"].values() for item in entry["inventory"]})


def truth(output):
    output = Path(output)
    load_plan(output)
    if truth_path(output).is_file():
        log("engine truth exists")
        return 0
    began = time.monotonic()
    branches, cells = gb.evaluation_set()
    attempts = {b["key"]: b["attempt"] for b in branches}
    keys = grid_keys(cells)
    with ProcessPoolExecutor(TRUTH_WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        values = list(pool.map(truth_worker, [attempts[k] for k in keys], chunksize=4))
    store = {"schema": SCHEMA_TRUTH, "plan_identity": IDENTITY, "built_at": utc_now(), "keys": keys,
             "times": list(TIMES), "last": torch.tensor([v["last"] for v in values]),
             "frames": torch.tensor([v["frames"] for v in values]),
             "presence": torch.stack([v["presence"] for v in values]),
             "centers": torch.stack([v["centers"] for v in values]),
             "source": "s99.shot_labels on the #104 evaluation captures (engine native traces), positions min(t, last)"}
    atomic_torch(truth_path(output), store)
    wall = time.monotonic() - began
    mark(output, "truth", wall, 0.0, branches=len(keys), processes=TRUTH_WORKERS)
    log(f"engine truth: {len(keys)} grid branches ({wall:.0f}s)")
    return 0


@functools.lru_cache(maxsize=None)
def load_truth(output):
    store = torch.load(truth_path(output), map_location="cpu", weights_only=True)
    if store.get("schema") != SCHEMA_TRUTH or store.get("plan_identity") != IDENTITY:
        raise ValueError("engine truth binding differs")
    store["index"] = {key: i for i, key in enumerate(store["keys"])}
    return store


def engine_error(z, store, key, t):
    """#100 engine_position_mse of one rollout carrier at endpoint t against its branch at min(t, last)."""
    if not bool(torch.isfinite(z).all()):
        return None
    i, k = store["index"][key], TIMES.index(t)
    return v1.r100.engine_position_mse(z.tolist(), {"presence": store["presence"][i, k][None],
                                                    "centers": store["centers"][i, k][None]}, 0)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def load_predictor(output, arm, seed):
    """(model, typed failure)."""
    if arm == CONTROL:
        return p101.load_predictor(seed), None
    state = member_state(output, arm, seed)
    if state["typed_failure"]:
        return None, state["typed_failure"]
    if sha256_of(member_path(output, arm, seed)) != state["sha256"]:
        raise ValueError(f"{arm} seed {seed}: checkpoint differs from its training report")
    return implementation(arm).load_member(output, ARM, arm, "all", seed), None


def load_j(output, arm, seed):
    if arm == CONTROL:
        return p101.load_policy(POLICY, "pe", ICLR_LAMBDA, seed)
    payload = torch.load(policy_path(output, arm, seed), map_location="cpu", weights_only=True)
    report = read_json(policy_report_path(output, arm, seed))
    if (payload.get("schema") != SCHEMA_POLICY or payload["arm"] != arm or payload["seed"] != seed
            or sha256_of(policy_path(output, arm, seed)) != report["policy_sha256"]):
        raise ValueError(f"policy {arm}/{seed} binding differs")
    policy = rp.RequestPolicy(PAIRS, payload["width"], payload["remaining_feature"])
    policy.load_state_dict(payload["model"], strict=True)
    return policy.to(DEVICE).eval()


def predictor_sha(output, arm, seed):
    return sha256_of(gb.member_path("unit", seed)) if arm == CONTROL else sha256_of(member_path(output, arm, seed))


def policy_sha(output, arm, seed):
    return (sha256_of(p101.policy_path(POLICY, "pe", ICLR_LAMBDA, seed)) if arm == CONTROL
            else sha256_of(policy_path(output, arm, seed)))


# ---------------------------------------------------------------------------
# scoring (deterministic_scoring, processes that did not train)
# ---------------------------------------------------------------------------

def spec_of(ranker):
    return {"kind": "joint"} if ranker == "J" else t96.arm_spec(ranker)


def score_block(model, policy, ranker, z0, items, costs, store=None):
    """The ranking rows of one (anchor, inventory) block at t = 225 (``rp.rollout_endpoints``, the #101 path); on
    grid (``store``) the engine error of every candidate at every endpoint in TIMES (fixed requests: the same
    rollout stopped at t; J: t = 225 only, its remaining-time input is defined for the 225 endpoint)."""
    actions = m99.action_batch(items)
    z0 = z0.to(DEVICE)
    spec = spec_of(ranker)
    z_end, timing = rp.rollout_endpoints(model, policy, z0, actions, spec, select=t96.select_pairs,
                                         record_schedules=ranker == "J")
    out = {"rows": [costs.row(item["ordinal"], z_end[slot].cpu(), z0.cpu()) for slot, item in enumerate(items)],
           "candidates": len(items), **{k: v for k, v in timing.items() if k != "schedules"}}
    if ranker == "J":
        out["schedules"] = {str(item["ordinal"]): t96.rle(timing["schedules"][k]) for k, item in enumerate(items)}
    if store is not None:
        errors = {}
        for t in (TIMES if ranker != "J" else (ENDPOINT,)):
            z_t = z_end if t == ENDPOINT else rp.rollout_endpoints(model, None, z0, actions, spec,
                                                                   select=t96.select_pairs, endpoint=t)[0]
            errors[str(t)] = [engine_error(z_t[slot].cpu(), store, item["key"], t) for slot, item in enumerate(items)]
        out["engine_mse"] = errors
    return out


def score_record(output, cells, anchors, store, arm, ranker, seed, policy_state):
    costs = p101.endpoint_costs()
    began = time.monotonic()
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "scoring_policy": policy_state, "scored_at": utc_now(),
            "arm": arm, "ranker": ranker, "seed": seed, "engine_seconds": 0}
    model, failure = load_predictor(output, arm, seed)
    if model is None:
        return {**base, "typed_failure": failure, "inventories": {}}
    policy = load_j(output, arm, seed) if ranker == "J" else None
    inventories = {i: {} for i in INVENTORIES}
    for inventory, member, entry in p101.ordered_cells(cells):
        inventories[inventory][member] = score_block(model, policy, ranker, anchors[member], entry["inventory"], costs,
                                                     store if inventory == PRIMARY else None)
    torch.cuda.synchronize()
    return {**base, "typed_failure": None, "predictor_sha256": predictor_sha(output, arm, seed),
            "policy_sha256": policy_sha(output, arm, seed) if ranker == "J" else None,
            "inventories": inventories, "seconds": time.monotonic() - began}


def wrong_record(output, arm, seed, policy_state):
    """J from every fit-split anchor over the shared grid candidates (Gate-B wrong-anchor rule)."""
    costs = p101.endpoint_costs()
    began = time.monotonic()
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "scoring_policy": policy_state, "scored_at": utc_now(),
            "arm": arm, "ranker": "wrong-J", "seed": seed, "engine_seconds": 0}
    model, failure = load_predictor(output, arm, seed)
    if model is None:
        return {**base, "typed_failure": failure, "wrong_anchor": None}
    policy = load_j(output, arm, seed)
    items = gb.inventory_items(PRIMARY)
    fit = gb.fit_anchors()
    rows = {member: score_block(model, policy, "J", fit[member], items, costs) for member in sorted(fit)}
    return {**base, "typed_failure": None, "wrong_anchor": rows, "seconds": time.monotonic() - began}


def score_tasks():
    """(kind, arm, ranker, seed), heaviest first so that the shards balance."""
    tasks = [("right", arm, ranker) for arm in CODE_ARMS for ranker in RANKERS]
    tasks += [("wrong", arm, "J") for arm in CODE_ARMS]

    def weight(task):
        kind, _, ranker = task
        return COST_WEIGHT["wrong"] if kind == "wrong" else COST_WEIGHT[ranker if ranker == "J"
                                                                         else ranker.rsplit("-", 1)[0]]
    ordered = sorted(tasks, key=lambda t: -weight(t))
    return [(*t, s) for t in ordered for s in SEEDS]


def task_path(output, task):
    kind, arm, ranker, seed = task
    return record_path(output, arm, ranker, seed) if kind == "right" else wrong_path(output, arm, seed)


def score(output, workers, worker):
    output = Path(output)
    load_plan(output)
    if not training_complete(output):
        raise ValueError("training or policies incomplete; score after --train")
    if not truth_path(output).is_file():
        raise ValueError("engine truth missing; run --truth first")
    _, cells = gb.evaluation_set()
    anchors = gb.load_anchors(GATE_B)["anchors"]
    store = load_truth(output)
    for task in score_tasks()[worker::workers]:
        path = task_path(output, task)
        if path.is_file():
            continue
        kind, arm, ranker, seed = task
        t0 = time.monotonic()
        with harness.deterministic_scoring() as policy_state:
            record = (score_record(output, cells, anchors, store, arm, ranker, seed, policy_state) if kind == "right"
                      else wrong_record(output, arm, seed, policy_state))
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, record)
        torch.cuda.empty_cache()
        wall = time.monotonic() - t0
        mark(output, f"score-worker{worker}", wall, wall)
        log(f"worker {worker}: {path.stem} ({wall:.0f}s)"
            + (f"; typed failure {record['typed_failure']}" if record["typed_failure"] else ""))
    return 0


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def read_record(path):
    record = read_json(path)
    if record.get("plan_identity") != IDENTITY or record.get("schema") != SCHEMA_RECORD:
        raise ValueError(f"record {path} binding differs")
    return record


def load_records(output):
    return {task: read_record(task_path(output, task)) for task in score_tasks()}


def key_of(arm, ranker):
    return f"{arm}|{ranker}"


def found_rows(record, inventory, member):
    return None if record["typed_failure"] else record["inventories"][inventory][member]["rows"]


def view_rows(records, cells, inventory, stats):
    """One row per (level, seed) cell: views of every (arm, ranker) and of the prior."""
    rows = []
    for member, entry in sorted(cells[inventory]["mixed"].items()):
        prior = m99.view(entry, gb.prior_rows(entry, stats), PRIMARY_COST)
        for seed in SEEDS:
            views = {key_of(a, r): v1.cost_view(entry, found_rows(records[("right", a, r, seed)], inventory, member),
                                                PRIMARY_COST) for a in CODE_ARMS for r in RANKERS}
            views["prior"] = prior
            rows.append({"member": member, "seed": seed, "views": views})
    return rows


def contrast(rows, pieces):
    block = m99.paired(rows, pieces)
    return {**block, "token": c2.token(block["estimate"])}


def mean_pieces(arm, weight=1.0):
    return [(weight / len(REQUESTS), key_of(arm, r)) for r in REQUESTS]


def ranking_table(rows):
    arms = {}
    for arm in CODE_ARMS:
        arms[arm] = {r: v1.ranker_block(rows, key_of(arm, r)) for r in RANKERS}
        arms[arm]["request_mean"] = m99.paired(rows, mean_pieces(arm))["estimate"]
    contrasts = {}
    for first, second in COMPARISONS:
        contrasts[f"{first}-{second}"] = {
            "request_mean": contrast(rows, mean_pieces(first) + mean_pieces(second, -1.0)),
            "per_request": {r: contrast(rows, [(1, key_of(first, r)), (-1, key_of(second, r))]) for r in RANKERS}}
    prior = {arm: {"J": contrast(rows, [(1, key_of(arm, "J")), (-1, "prior")]),
                   "request_mean": contrast(rows, mean_pieces(arm) + [(-1, "prior")])} for arm in CODE_ARMS}
    return {"cells": len(rows), "arms": arms, "prior": v1.ranker_block(rows, "prior"), "contrasts": contrasts,
            "versus_prior": prior}


def wrong_table(records, cells, arm):
    rows = []
    for member, entry in sorted(cells[PRIMARY]["mixed"].items()):
        for seed in SEEDS:
            record = records[("wrong", arm, "J", seed)]
            right_rows = found_rows(records[("right", arm, "J", seed)], PRIMARY, member)
            if record["typed_failure"] or right_rows is None:
                rows.append({"member": member, "seed": seed, "views": {"J": None}})
                continue
            wrong = [record["wrong_anchor"][a]["rows"] for a in sorted(record["wrong_anchor"])]
            wrong_auc = [a for a in (c2.auc(entry, w) for w in wrong) if a is not None]
            rhos = [r for r in (c2.spearman_rows(right_rows, w) for w in wrong) if r is not None]
            rows.append({"member": member, "seed": seed, "views": {"J": {
                "right": c2.auc(entry, right_rows), "wrong": float(np.mean(wrong_auc)) if wrong_auc else None,
                "stability": float(np.mean(rhos)) if rhos else None}}})
    difference = c2.difference_block(rows, [(1, "J", "right"), (-1, "J", "wrong")])
    return {"cells": len(rows), "wrong_anchors": len(gb.fit_anchors()), "right": c2.field_block(rows, "J", "right"),
            "wrong": c2.field_block(rows, "J", "wrong"), "right_minus_wrong": difference,
            "stability": c2.field_block(rows, "J", "stability"), "token": c2.token(difference)}


def error_cells(records, cells, arm, ranker, t):
    """{(level, seed): mean engine error over the cell's candidates with a defined value} (None: typed failure or
    no defined value), and the count of nonfinite endpoints."""
    out, nonfinite = {}, 0
    for member in sorted(cells[PRIMARY]["mixed"]):
        for seed in SEEDS:
            record = records[("right", arm, ranker, seed)]
            if record["typed_failure"]:
                out[(member, seed)] = None
                continue
            block = record["inventories"][PRIMARY][member]
            values = [v for v in block["engine_mse"][str(t)] if v is not None]
            nonfinite += sum(1 for v in block["engine_mse"][str(t)] if v is None)
            out[(member, seed)] = float(np.mean(values)) if values else None
    return out, nonfinite


def level_values(per_cell, seeds_by_level=None):
    levels = {}
    for (member, seed), value in per_cell.items():
        if value is None or (seeds_by_level is not None and seed not in seeds_by_level.get(member, ())):
            continue
        levels.setdefault(member, []).append(value)
    return {m: float(np.mean(v)) for m, v in levels.items()}


def ratio_block(first, second):
    """Per level the log of (first / second) over the seeds defined in both arms; bootstrap over levels;
    geometric ratio (the #117 ratio construction, levels as clusters)."""
    common = {}
    for (member, seed), value in first.items():
        other = second.get((member, seed))
        if value is not None and other is not None and value > 0 and other > 0:
            common.setdefault(member, set()).add(seed)
    a, b = level_values(first, common), level_values(second, common)
    members = sorted(common)
    block = bootstrap([math.log(a[m] / b[m]) for m in members], members)
    if block is None:
        return None
    return {"ratio": math.exp(block["mean"]), "interval": [math.exp(v) for v in block["interval"]],
            "log_mean": block["mean"], "log_interval": block["interval"], "clusters": block["clusters"],
            "label": block["label"]}


def claim_token(block):
    """#117: 'supported' (costs no more than CLAIM x) iff the upper bound < CLAIM; 'not_supported_by_this_experiment'
    iff the lower bound > CLAIM; otherwise (or fewer than MIN_CLUSTERS levels) STOP_TOKEN."""
    if block is None or block["clusters"] < MIN_CLUSTERS:
        return STOP_TOKEN
    if block["interval"][1] < CLAIM:
        return "supported"
    if block["interval"][0] > CLAIM:
        return "not_supported_by_this_experiment"
    return STOP_TOKEN


def rollout_tables(records, cells):
    """#100 engine_position_mse on every mixed grid candidate: per (level, seed) the candidate mean, per level
    the seed mean, level bootstrap; ratios versus fused per request and endpoint."""
    cache = {}
    for arm in CODE_ARMS:
        for ranker in RANKERS:
            for t in (TIMES if ranker != "J" else (ENDPOINT,)):
                cache[(arm, ranker, t)] = error_cells(records, cells, arm, ranker, t)
    errors = {}
    for (arm, ranker, t), (per_cell, nonfinite) in cache.items():
        levels = level_values(per_cell)
        errors[f"{arm}|{ranker}|{t}"] = {"error": bootstrap([levels[m] for m in sorted(levels)], sorted(levels)),
                                         "nonfinite_endpoints": nonfinite}
    ratios = {}
    for arm in TRAINED:
        for ranker in RANKERS:
            for t in (TIMES if ranker != "J" else (ENDPOINT,)):
                block = ratio_block(cache[(arm, ranker, t)][0], cache[(CONTROL, ranker, t)][0])
                ratios[f"{arm}/{CONTROL}|{ranker}|{t}"] = {**(block or {}), "token": claim_token(block)}
    return {"metric": "engine_position_mse (#100) at min(t, last frame), every mixed grid candidate",
            "times": list(TIMES), "errors": errors, "ratios": ratios}


def macs_table(records, cells):
    """Linear MACs (cnn_hybrid.linear_macs accounting: executed linear operators per call, the request code and
    FiLM generators counted per row as executed) per predictor call per request, and per candidate-state (one
    candidate rollout to t = 225, policy calls included for J), pooled over grid blocks and seeds."""
    out = {}
    for arm in CODE_ARMS:
        model = rc.new_predictor(arm)
        per_call = {name: linear_macs(model, p) for name, p in zip(t96.PAIR_NAMES, PAIRS)}
        per_state = {}
        for ranker in RANKERS:
            macs = candidates = 0
            for seed in SEEDS:
                record = records[("right", arm, ranker, seed)]
                if record["typed_failure"]:
                    continue
                for member in cells[PRIMARY]["mixed"]:
                    block = record["inventories"][PRIMARY][member]
                    macs += block["linear_macs"]
                    candidates += block["candidates"]
            per_state[ranker] = macs / candidates if candidates else None
        out[arm] = {"predictor_macs_per_call": per_call, "macs_per_candidate_state": per_state,
                    "parameters": rc.parameter_split(model, arm)}
    return out


def coupling_tables(output):
    """Coupling diagnostic on every usable model: per block, the two-way decomposition (Delta x alpha) of the
    effective FiLM scale gamma and shift beta at the nine requests (float64, CPU); seed means."""
    out = {}
    for arm in CODE_ARMS:
        seeds = {}
        for seed in SEEDS:
            model, failure = load_predictor(output, arm, seed)
            seeds[str(seed)] = None if model is None else rc.coupling(model.cpu(), arm)
            del model
        usable = [v for v in seeds.values() if v is not None]
        mean = None
        if usable:
            fields = ("interaction_share", "interaction_share_of_alpha_variation",
                      "interaction_share_of_delta_variation", "rms", "alpha_given_delta", "delta_given_alpha")
            mean = {"blocks": [{part: {f: float(np.mean([u["blocks"][b][part][f] for u in usable]))
                                       for f in fields} for part in ("gamma", "beta")}
                               for b in range(len(usable[0]["blocks"]))],
                    "code": None if usable[0]["code"] is None else
                    {f: float(np.mean([u["code"][f] for u in usable])) for f in fields}}
        out[arm] = {"seeds": seeds, "mean": mean}
    return out


def training_table(output, arm):
    if arm == CONTROL:
        return {"source": "Gate-B handoff checkpoints (#112 v3 retrain, LC hybrid); J = the #101 J",
                "usable_seeds": list(SEEDS),
                "J_policy_split_auc": {str(s): read_json(p101.report_path(POLICY, "pe", ICLR_LAMBDA, s))["policy_split"]
                                       ["auc_pooled"] for s in SEEDS}}
    members = [member_state(output, arm, s) for s in SEEDS]
    return {"members": members, "usable_seeds": [m["seed"] for m in members if m["usable"]],
            "J_policy_split_auc": {str(s): read_json(policy_report_path(output, arm, s))["policy_split"]["auc_pooled"]
                                   for s in SEEDS if policy_report_path(output, arm, s).is_file()}}


def dispositions(ranking, rollout, guards_ok):
    """Pre-declared (frozen_rules)."""
    per = {r: rollout["ratios"][f"additive/{CONTROL}|{r}|{ENDPOINT}"]["token"] if guards_ok else STOP_TOKEN
           for r in REQUESTS}
    tokens = set(per.values())
    if not guards_ok:
        first = STOP_TOKEN
    elif tokens == {"supported"}:
        first = "supported"
    elif "not_supported_by_this_experiment" in tokens:
        first = "not_supported_by_this_experiment"
    else:
        first = STOP_TOKEN
    primary = ranking[PRIMARY]["contrasts"]["fused-additive"]["request_mean"]
    second = primary["token"] if guards_ok else STOP_TOKEN
    return {
        "additive_within_1_5x": {
            "claim": (f"an additive request code matches the fused code within {CLAIM}x per-request rollout error at "
                      f"t = {ENDPOINT}"),
            "token": first, "per_request": per,
            "ratios": {r: rollout["ratios"][f"additive/{CONTROL}|{r}|{ENDPOINT}"].get("ratio") for r in REQUESTS}},
        "interaction_decision_value": {
            "claim": "axis interaction in the request code carries measurable decision value",
            "token": second, "contrast": "fused - additive, grid, nine-fixed-request mean AUC",
            "estimate": primary["estimate"],
            "J": ranking[PRIMARY]["contrasts"]["fused-additive"]["per_request"]["J"]["estimate"]}}


def replication(records):
    """Fused rows (and J schedules and linear MACs) equal the published #101 records (nine fixed requests and the
    #101 J) on every inventory, cell and seed."""
    same, checked = True, 0
    for ranker in RANKERS:
        source = p101.J_ICLR if ranker == "J" else ranker
        for seed in SEEDS:
            reference = p101.read_record(POLICY, source, seed)["inventories"]
            mine = records[("right", CONTROL, ranker, seed)]["inventories"]
            for inventory in INVENTORIES:
                for member, result in reference[inventory].items():
                    ours = mine[inventory][member]
                    same &= ours["rows"] == result["rows"] and ours["linear_macs"] == result["linear_macs"]
                    if "schedules" in result:
                        same &= ours["schedules"] == result["schedules"]
                    checked += 1
    return {"blocks": checked, "pass": bool(same)}


def truth_check(output, records):
    """The store's t = 225 labels equal the #115 engine-truth store (presence and projected centres at
    min(225, last)) on every grid key, and the fused fixed micro requests' t = 225 engine errors equal the #115
    Gate-B default-feedback records."""
    mine = load_truth(output)
    other = torch.load(TRUTH115 / "data" / "engine-truth.pt", map_location="cpu", weights_only=True, mmap=True)
    index = {k: i for i, k in enumerate(other["keys"])}
    k225 = TIMES.index(ENDPOINT)
    rows = torch.tensor([index[k] for k in mine["keys"]])
    store_same = bool(torch.equal(mine["presence"][:, k225], other["presence"][rows])
                      and torch.equal(mine["centers"][:, k225], other["centers"][rows])
                      and torch.equal(mine["last"], other["last"][rows]))
    metric_same, checked = True, 0
    for ranker in ("F-1-micro", "F-5-micro", "F-15-micro"):
        for seed in SEEDS:
            reference = read_json(TRUTH115 / "records" / f"gateb--default--{ranker}--seed{seed}.json")
            for member, result in reference["inventories"][PRIMARY].items():
                ours = records[("right", CONTROL, ranker, seed)]["inventories"][PRIMARY][member]["engine_mse"][str(ENDPOINT)]
                metric_same &= ours == result["engine_mse"]
                checked += 1
    return {"store_equals_115_at_225": store_same, "engine_mse_blocks_checked": checked,
            "engine_mse_equals_115": bool(metric_same), "pass": bool(store_same and metric_same)}


def records_manifest(output):
    output = Path(output)
    paths = (sorted((output / "records").glob("*.json")) + sorted((output / "training").glob("updates--*.pt"))
             + [truth_path(output)])
    return {"schema": SCHEMA_MANIFEST, "identity": IDENTITY,
            "files": {str(p.relative_to(output)): sha256_of(p) for p in paths}}


GUARD_FAILURE_NOTE = (f"a failed guard turns every token of this publication into {STOP_TOKEN}: both dispositions, "
                      "the contrast and ratio tokens of every table and the wrong-anchor tokens are not interpretable")


def invalidate(obj):
    """Every token below ``obj`` -> STOP_TOKEN (a failed guard)."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "token":
                obj[key] = STOP_TOKEN
            else:
                invalidate(value)
    elif isinstance(obj, list):
        for value in obj:
            invalidate(value)


def guard_block(output, plan, records, rows, cells):
    payload = rt.load_training_data(RETRAIN)
    eval_roster = {lv["identity"] for lv in gb.evaluation_levels()}
    fit_roster = {lv["identity"] for lv in rt.fit_levels()}
    store_meta = torch.load(truth_path(output), map_location="cpu", weights_only=True, mmap=True)
    scored_at = sorted(r["scored_at"] for r in records.values())
    members = [member_state(output, a, s) for a in TRAINED for s in SEEDS]
    trained_at = sorted(m["trained_at"] for m in members)
    policy_reports = [read_json(policy_report_path(output, a, s)) for a in TRAINED for s in SEEDS
                      if policy_report_path(output, a, s).is_file()]
    policies_at = sorted(r["trained_at"] for r in policy_reports)
    hashes = (all(m["sha256"] == sha256_of(member_path(output, m["arm"], m["seed"])) for m in members)
              and all(r["policy_sha256"] == sha256_of(policy_path(output, r["arm"], r["seed"])) for r in policy_reports)
              and all(r["predictor_sha256"] == sha256_of(member_path(output, r["arm"], r["seed"]))
                      for r in policy_reports)
              and all(c["sha256"] == sha256_of(gb.member_path("unit", c["seed"])) for c in gb.checkpoints("unit")))
    smoke_replication = next(c for c in plan["smoke_evidence"]["checks"]
                             if c["control"] == "fused_training_path_equals_retrain")
    nonfinite = v1.ranker_block(rows[PRIMARY], key_of(CONTROL, "J"))["nonfinite_share"]
    scoring_replication = replication(records)
    return {
        "G1_isolation": {"pass": bool(set(payload["shot_members"]) <= fit_roster and not (fit_roster & eval_roster)
                                      and all(set(cells[i]["mixed"]) <= eval_roster for i in INVENTORIES))},
        "G2_scoring_policy": {"pass": bool(all(r["scoring_policy"] == harness.SCORING_POLICY
                                               for r in records.values()))},
        "G3_chronology": {"plan_frozen_at": plan["frozen_at"], "first_trained_at": trained_at[0],
                          "last_trained_at": trained_at[-1], "last_policy_at": policies_at[-1] if policies_at else None,
                          "truth_built_at": store_meta["built_at"], "first_scored_at": scored_at[0],
                          "pass": bool(plan["frozen_at"] <= trained_at[0] and trained_at[-1] <= scored_at[0]
                                       and (not policies_at or policies_at[-1] <= scored_at[0])
                                       and plan["frozen_at"] <= store_meta["built_at"] <= scored_at[0])},
        "G4_checkpoint_binding": {"pass": bool(hashes)},
        "G5_replication": {"training_path": {k: smoke_replication[k] for k in
                                             ("updates", "max_parameter_delta", "local_losses_equal", "ok")},
                           "scoring": scoring_replication,
                           "pass": bool(smoke_replication["ok"] and scoring_replication["pass"])},
        "G6_engine_truth": truth_check(output, records),
        "G7_nonfinite": {"fused_J_grid_share": nonfinite,
                         "pass": bool(nonfinite is not None and nonfinite <= NONFINITE_SHARE_MAX)},
    }


def compute_tables(output, plan):
    output = Path(output)
    branches, cells = gb.evaluation_set()
    records = load_records(output)
    fit = read_json(gb.FIT_VERDICTS)["branches"]
    stats = {i: gb.prior_stats(fit, i) for i in INVENTORIES}
    rows = {i: view_rows(records, cells, i, stats[i]) for i in INVENTORIES}
    ranking = {i: ranking_table(rows[i]) for i in INVENTORIES}
    wrong = {arm: wrong_table(records, cells, arm) for arm in CODE_ARMS}
    rollout = rollout_tables(records, cells)
    with harness.deterministic_scoring():
        coupling = coupling_tables(output)
    guards = guard_block(output, plan, records, rows, cells)
    guards_ok = all(g["pass"] for g in guards.values())
    failed = sorted(k for k, g in guards.items() if not g["pass"])
    phases = {p.stem: read_json(p) for p in sorted((output / "ledger").glob("*.json"))}
    smoke_ = read_json(output / "smoke.json")
    all_phases = {"smoke": {"wall_seconds": smoke_["wall_seconds"], "gpu_seconds": smoke_["wall_seconds"]},
                  **{k: {"wall_seconds": v["wall_seconds"], "gpu_seconds": v["gpu_seconds"]} for k, v in phases.items()}}
    compute = {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
        "evaluation": gb.evaluation_accounting(branches, cells),
        "guards": guards, "dispositions": dispositions(ranking, rollout, guards_ok),
        "ranking": ranking, "wrong_anchor": wrong, "rollout": rollout, "macs": macs_table(records, cells),
        "coupling": coupling, "training": {arm: training_table(output, arm) for arm in CODE_ARMS},
        "compute": {"phases": all_phases, "gpu_seconds": sum(b["gpu_seconds"] for b in all_phases.values()),
                    "wall_seconds": sum(b["wall_seconds"] for b in all_phases.values()), "engine_seconds": 0,
                    "hardware": "4 x RTX 5090 (one process per GPU)"},
    }
    if failed:
        compute["guard_failure"] = {"failed_guards": failed, "note": GUARD_FAILURE_NOTE}
        for key in ("ranking", "wrong_anchor", "rollout"):
            invalidate(compute[key])
    return compute


# ---------------------------------------------------------------------------
# renderings
# ---------------------------------------------------------------------------

def num(value, digits=3):
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else format(value, f".{digits}f")


def cell(block, digits=3):
    if not block:
        return "n/a"
    return f"{num(block['mean'], digits)} [{num(block['interval'][0], digits)}, {num(block['interval'][1], digits)}]"


def signed(block, digits=3):
    if not block:
        return "n/a"
    lo, hi = block["interval"]
    return f"{block['mean']:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def ratio_cell(block):
    if not block or block.get("ratio") is None:
        return "n/a"
    return f"{block['ratio']:.2f} [{block['interval'][0]:.2f}, {block['interval'][1]:.2f}]"


def short(token):
    return {"supported": "sup.", "not_supported_by_this_experiment": "no", STOP_TOKEN: "insuff."}.get(token, token)


def auc_mean(block):
    return block["auc"]["mean"] if block and block.get("auc") else None


def findings_md(plan, compute):
    lines = []
    add = lines.append
    d = compute["dispositions"]
    add("# Issue #121: request-code factorisation (fused vs additive vs factored-FiLM)")
    add("")
    add(f"Frozen {plan['frozen_at']}; runner `{RUNNER}`; module `{MODULE}`. Scored once on the #104 sealed #109 "
        "evaluation split (Gate-B cells and anchors), under `deterministic_scoring()`, tie-free cost. Intervals: "
        "level-clustered percentile bootstrap (10 000 draws, PCG64 7201), DESCRIPTIVE; delta = 0.02. Engine "
        "seconds 0.")
    add("")
    if "guard_failure" in compute:
        add(f"**Guard failure: {', '.join(compute['guard_failure']['failed_guards'])}.** "
            f"{compute['guard_failure']['note']}.")
        add("")
    add("## Dispositions (pre-declared)")
    add("")
    add("| claim | token | evidence |")
    add("| --- | --- | --- |")
    first = d["additive_within_1_5x"]
    add(f"| {first['claim']} | `{first['token']}` | additive / fused per request at t = {ENDPOINT}: "
        + ", ".join(f"{r} {num(first['ratios'][r], 2)} ({short(first['per_request'][r])})" for r in REQUESTS) + " |")
    second = d["interaction_decision_value"]
    add(f"| {second['claim']} | `{second['token']}` | {second['contrast']}: {signed(second['estimate'])}; "
        f"J (descriptive): {signed(second['J'])} |")
    add("")
    ratios, errors = compute["rollout"]["ratios"], compute["rollout"]["errors"]
    add(f"## Rollout error at t = {ENDPOINT} (#100 engine_position_mse, every mixed grid candidate)")
    add("")
    add("| request | fused | additive | factored | additive / fused | factored / fused |")
    add("| --- | --- | --- | --- | --- | --- |")
    for r in RANKERS:
        add(f"| {r} | " + " | ".join(cell(errors[f'{a}|{r}|{ENDPOINT}']['error'], 4) for a in CODE_ARMS) + " | "
            + " | ".join(f"{ratio_cell(ratios[f'{a}/{CONTROL}|{r}|{ENDPOINT}'])} "
                         f"({short(ratios[f'{a}/{CONTROL}|{r}|{ENDPOINT}']['token'])})" for a in TRAINED) + " |")
    add("")
    add("J is each arm's own policy (scheduled requests); its row is DESCRIPTIVE and does not enter the 1.5x rule.")
    add("")
    add("### Every endpoint (mean error; ratio to fused)")
    add("")
    add("| request | arm | " + " | ".join(f"t = {t}" for t in TIMES) + " |")
    add("|" + " --- |" * (2 + len(TIMES)))
    for r in REQUESTS:
        for a in CODE_ARMS:
            cells_ = []
            for t in TIMES:
                e = errors[f"{a}|{r}|{t}"]["error"]
                text = num(e and e["mean"], 4)
                if a != CONTROL:
                    text += f" ({num(ratios[f'{a}/{CONTROL}|{r}|{t}'].get('ratio'), 2)}x)"
                cells_.append(text)
            add(f"| {r} | {a} | " + " | ".join(cells_) + " |")
    add("")
    for inventory in INVENTORIES:
        table = compute["ranking"][inventory]
        label = "primary" if inventory == PRIMARY else "DESCRIPTIVE"
        add(f"## Ranking AUC, {inventory} ({label}; {table['cells']} level x seed cells)")
        add("")
        add("| ranker | fused | additive | factored | fused − additive | fused − factored | factored − additive |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for r in RANKERS:
            add(f"| {r} | " + " | ".join(cell(table['arms'][a][r]['auc']) for a in CODE_ARMS) + " | "
                + " | ".join(f"{signed(table['contrasts'][f'{x}-{y}']['per_request'][r]['estimate'])} "
                             f"({short(table['contrasts'][f'{x}-{y}']['per_request'][r]['token'])})"
                             for x, y in COMPARISONS) + " |")
        add("| nine-request mean | " + " | ".join(cell(table['arms'][a]['request_mean']) for a in CODE_ARMS) + " | "
            + " | ".join(f"{signed(table['contrasts'][f'{x}-{y}']['request_mean']['estimate'])} "
                         f"({short(table['contrasts'][f'{x}-{y}']['request_mean']['token'])})" for x, y in COMPARISONS)
            + " |")
        add(f"| no-model prior | {cell(table['prior']['auc'])} | | | | | |")
        add("")
        add("| arm | J − prior | request mean − prior |")
        add("| --- | --- | --- |")
        for a in CODE_ARMS:
            vp = table["versus_prior"][a]
            add(f"| {a} | {signed(vp['J']['estimate'])} ({short(vp['J']['token'])}) | "
                f"{signed(vp['request_mean']['estimate'])} ({short(vp['request_mean']['token'])}) |")
        add("")
    anchors = compute["wrong_anchor"][CONTROL]["wrong_anchors"]
    add(f"## Wrong-anchor control (J, grid; {anchors} fit-split anchors x the grid candidates)")
    add("")
    add("| arm | right | wrong | right − wrong | stability (Spearman) |")
    add("| --- | --- | --- | --- | --- |")
    for a in CODE_ARMS:
        w = compute["wrong_anchor"][a]
        add(f"| {a} | {cell(w['right'])} | {cell(w['wrong'])} | {signed(w['right_minus_wrong'])} "
            f"({short(w['token'])}) | {cell(w['stability'])} |")
    add("")
    add("## Coupling diagnostic (per block; seed means)")
    add("")
    add("Two-way decomposition of the effective FiLM scale gamma and shift beta over the 3 x 3 requests. "
        "Interaction share = SS_interaction / SS_total (0 iff additive in Delta and alpha). Var over alpha = mean "
        "variance over alpha at fixed Delta relative to the mean squared norm (the identity FiLM is gamma = beta = 0); "
        "var over Delta likewise. rms = root mean square entry.")
    add("")
    add("| arm | block | gamma interaction share | gamma var over alpha | gamma var over Delta | gamma rms | "
        "beta interaction share | beta var over alpha | beta var over Delta | beta rms |")
    add("|" + " --- |" * 10)
    for a in CODE_ARMS:
        mean = compute["coupling"][a]["mean"]
        if mean is None:
            add(f"| {a} | n/a | | | | | | | | |")
            continue
        for b, block in enumerate(mean["blocks"]):
            g, be = block["gamma"], block["beta"]
            add(f"| {a} | {b} | {num(g['interaction_share'], 4)} | {num(g['alpha_given_delta'], 4)} | "
                f"{num(g['delta_given_alpha'], 4)} | {num(g['rms'], 4)} | {num(be['interaction_share'], 4)} | "
                f"{num(be['alpha_given_delta'], 4)} | {num(be['delta_given_alpha'], 4)} | {num(be['rms'], 4)} |")
    add("")
    add("Per seed (fused, the Gate-B checkpoints): gamma / beta interaction share per block:")
    add("")
    for seed, value in compute["coupling"][CONTROL]["seeds"].items():
        add(f"- seed {seed}: " + "; ".join(
            f"block {b} {num(x['gamma']['interaction_share'], 4)} / {num(x['beta']['interaction_share'], 4)}"
            for b, x in enumerate(value["blocks"])) + f"; code {num(value['code']['interaction_share'], 4)}")
    add("")
    add("## MACs and parameters")
    add("")
    add("Linear MACs (#103 / #114 accounting). Predictor MACs per call are Delta-independent.")
    add("")
    add("| arm | parameters total | request path | MACs per call: continuous | micro | macro |")
    add("|" + " --- |" * 6)
    for a in CODE_ARMS:
        m = compute["macs"][a]
        call = m["predictor_macs_per_call"]
        add(f"| {a} | {m['parameters']['total']:,} | {m['parameters']['request_path']:,} | "
            + " | ".join(f"{call[f'1-{alpha}']:,}" for alpha in t96.ALPHAS) + " |")
    add("")
    add("MACs per candidate-state (one candidate rollout to t = 225; J includes its policy calls), grid, millions:")
    add("")
    add("| ranker | " + " | ".join(CODE_ARMS) + " |")
    add("|" + " --- |" * (1 + len(CODE_ARMS)))
    for r in RANKERS:
        add(f"| {r} | " + " | ".join(num(compute['macs'][a]['macs_per_candidate_state'][r]
                                         and compute['macs'][a]['macs_per_candidate_state'][r] / 1e6, 2)
                                     for a in CODE_ARMS) + " |")
    add("")
    add("## Training")
    add("")
    add("| arm | seed | usable | skipped | GPU s | final local loss | J policy-split AUC |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for a in TRAINED:
        t = compute["training"][a]
        for m in t["members"]:
            add(f"| {a} | {m['seed']} | {m['usable']}{'' if m['usable'] else ' (' + m['typed_failure'] + ')'} | "
                f"{m['skipped']} | {m['gpu_seconds']:.0f} | {num(m['final_trace']['local'][0], 4)} | "
                f"{num(t['J_policy_split_auc'].get(str(m['seed'])), 4)} |")
    add("")
    add("## Guards")
    add("")
    add("| guard | pass |")
    add("| --- | --- |")
    for k, g in compute["guards"].items():
        add(f"| {k} | {g['pass']} |")
    add("")
    c = compute["compute"]
    add(f"## Compute\n\nGPU seconds {c['gpu_seconds']:.0f}; wall seconds (summed over processes) "
        f"{c['wall_seconds']:.0f}; engine seconds 0; {c['hardware']} (the #85 protocol named one RTX 3090).")
    add("")
    return "\n".join(lines)


def summary(plan, compute):
    d = compute["dispositions"]
    grid = compute["ranking"][PRIMARY]
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "dispositions": {k: v["token"] for k, v in d.items()},
            "additive_over_fused_t225": d["additive_within_1_5x"]["ratios"],
            "grid_request_mean_auc": {a: grid["arms"][a]["request_mean"] for a in CODE_ARMS},
            "grid_J_auc": {a: grid["arms"][a]["J"]["auc"] for a in CODE_ARMS},
            "grid_contrasts_request_mean": {k: v["request_mean"]["estimate"] for k, v in grid["contrasts"].items()},
            "prior_grid": grid["prior"]["auc"],
            "guards": {k: g["pass"] for k, g in compute["guards"].items()},
            **({"guard_failure": compute["guard_failure"]} if "guard_failure" in compute else {}),
            "gpu_seconds": compute["compute"]["gpu_seconds"], "engine_seconds": 0}


def rendered(plan, compute, output=OUTPUT):
    return {"compute.json": json_text(compute), "summary.json": json_text(summary(plan, compute)),
            "findings.md": findings_md(plan, compute), "records-manifest.json": json_text(records_manifest(output))}


# ---------------------------------------------------------------------------
# smoke (pre-freeze; fit split and the #101 policies only)
# ---------------------------------------------------------------------------

def max_delta(first, second):
    return max(float((first[k] - second[k]).abs().max()) for k in first)


def smoke_training_path(scratch, data):
    """Replication guard: this runner's training path configured as fused equals the retrain's train_group bit
    for bit (parameters after SMOKE_STEPS full-cost updates and every logged local loss)."""
    torch.backends.cudnn.benchmark = True
    _, a = implementation("fused").train_group(scratch, ARM, "fused", 0, [("all", SEEDS[0])], data,
                                               steps=SMOKE_STEPS, save=False, smoke_full=True)
    params_a = {k: v.detach().clone() for k, v in a.params.items()}
    local_a = a.log_tensors()["local"].clone()
    del a
    torch.cuda.empty_cache()
    _, b = rt.implementation().train_group(scratch, ARM, "hybrid", 0, [("all", SEEDS[0])], data,
                                           steps=SMOKE_STEPS, save=False, smoke_full=True)
    delta = max_delta(params_a, b.params)
    same_local = bool(torch.equal(local_a, b.log_tensors()["local"]))
    same_names = sorted(params_a) == sorted(b.params)
    del b
    torch.cuda.empty_cache()
    return {"control": "fused_training_path_equals_retrain", "updates": SMOKE_STEPS, "max_parameter_delta": delta,
            "local_losses_equal": same_local, "parameter_names_equal": same_names,
            "ok": bool(delta == 0.0 and same_local and same_names)}


def smoke_arms(scratch, data):
    """A full-cost training smoke per new arm (warm-up excluded from the timing)."""
    out, timings = [], {}
    for arm in TRAINED:
        torch.cuda.reset_peak_memory_stats()
        result, ensemble = implementation(arm).train_group(scratch, ARM, arm, 0, [("all", SEEDS[0])], data,
                                                           steps=SMOKE_WARMUP + SMOKE_STEPS, save=False,
                                                           smoke_full=True, timing_after=SMOKE_WARMUP)
        finite = bool(torch.isfinite(torch.tensor(result["trace"][-1]["loss"])).all())
        timings[arm] = result["timed_seconds_per_update"]
        out.append({"arm": arm, "seconds_per_update_full_cost": result["timed_seconds_per_update"],
                    "parameters": result["parameters_per_member"], "peak_memory_gb": result["peak_memory_gb"],
                    "final_loss_finite": finite, "skipped": int(ensemble.skipped[0])})
        del ensemble
        torch.cuda.empty_cache()
    return {"control": "arm_training", "arms": out, "ok": all(r["final_loss_finite"] for r in out)}, timings


def smoke_structure():
    """Arm construction: the fused class is the Gate-B class; trunk / head / adapter initialisation of the new
    arms equals the fused member of the same seed; the additive code is rank-2 (zero interaction); the factored
    block composes (1 + g_d)(1 + g_a), b_d + b_a; parameter split per arm."""
    torch.manual_seed(SEEDS[0])
    fused = rc.new_predictor("fused")
    out = {"fused_is_cnn_hybrid": type(fused).__name__ == "CNNHybridPredictor"}
    reference = fused.state_dict()
    for arm in TRAINED:
        torch.manual_seed(SEEDS[0])
        model = rc.new_predictor(arm)
        prefixes = rc.request_path_names(arm)
        shared = {k: v for k, v in model.state_dict().items() if not k.startswith(prefixes)}
        out[f"{arm}_trunk_init_equals_fused"] = all(
            torch.equal(v, reference[k]) for k, v in shared.items()) and len(shared) == len(
            [k for k in reference if not k.startswith(rc.request_path_names("fused"))])
        out[f"{arm}_parameters"] = rc.parameter_split(model, arm)
    out["fused_parameters"] = rc.parameter_split(fused, "fused")
    torch.manual_seed(1)
    additive = rc.new_predictor("additive")
    for block in additive.blocks:
        torch.nn.init.normal_(block.modulation.weight, std=0.1)
    shares = [b["gamma"]["interaction_share"] for b in rc.coupling(additive, "additive")["blocks"]]
    out["additive_interaction_share_max"] = max(shares)
    torch.manual_seed(2)
    factored = rc.new_predictor("factored")
    for block in factored.blocks:
        for layer in (block.modulation_delta, block.modulation_alpha):
            torch.nn.init.normal_(layer.weight, std=0.1)
            torch.nn.init.normal_(layer.bias, std=0.1)
    h = torch.randn(4, 384)
    code = factored.conditioner.code(PAIRS[4], 4, "cpu")
    block = factored.blocks[0]
    sd, bd, sa, ba = block.film(code)
    manual = h + block.outer(torch.nn.functional.silu(block.inner(block.norm(h) * ((1 + sd) * (1 + sa)) + (bd + ba))))
    out["factored_composition_max_error"] = float((block(h, code) - manual).abs().max())
    out["factored_gamma_interaction_share_block0"] = rc.coupling(factored, "factored")["blocks"][0]["gamma"][
        "interaction_share"]
    out["ok"] = bool(out["fused_is_cnn_hybrid"] and all(out[f"{a}_trunk_init_equals_fused"] for a in TRAINED)
                     and out["additive_interaction_share_max"] < 1e-12 and out["factored_composition_max_error"] == 0.0
                     and out["factored_gamma_interaction_share_block0"] > 1e-6)
    out["control"] = "arm_structure"
    return out


def smoke_policy():
    """Round-0 J labels of this runner's path equal #101's on the first policy units (Gate-B seed 0), and the
    label path runs on a (fresh) factored model; projected label-round time."""
    pool = p101.load_pool(POLICY)
    model = p101.load_predictor(SEEDS[0])
    reference = torch.load(p101.group_root(POLICY, "pe", ICLR_LAMBDA, SEEDS[0]) / "labels-0.pt", map_location="cpu",
                           weights_only=True)["units"]
    units = p101.window_units(pool)
    began = time.monotonic()
    with harness.deterministic_scoring():
        policy = rp.new_policy(PAIRS, SEEDS[0], device=DEVICE)
        mine = [rp.window_rows(model, policy, unit, aggregated=False, compute_weight=ICLR_LAMBDA, device=DEVICE)
                for unit in units[:2]]
        seconds = (time.monotonic() - began) / 2 * len(units)
        torch.manual_seed(SEEDS[0])
        other = rc.new_predictor("factored").to(DEVICE).eval()
        runs = rp.window_rows(other, policy, units[0], aggregated=False, compute_weight=ICLR_LAMBDA, device=DEVICE)
    same = all(torch.equal(mine[u][k].cpu(), reference[u][k].cpu()) for u in range(2) for k in mine[u])
    return {"control": "policy_labels_equal_101", "units": 2, "projected_label_round_seconds": seconds,
            "factored_label_rows": int(len(runs["labels"])), "ok": bool(same and len(runs["labels"]) > 0)}


def fit_grid_cells():
    return gb.mixed_cells(gb.inventory_entries(rt.fit_levels(), rt.complete_branches(), rt.fit_verdicts(), PRIMARY))


def smoke_scoring():
    """On fit grid cells (Gate-B seed 0, the #101 J): this runner's scoring block equals the #101 block (rows, J
    schedules, linear MACs) for every ranker; a fixed-request rollout stopped at t equals t / Delta full-batch
    carrier calls; the truth worker and the engine error at every t run on the first cell's fit branches
    (``score_block`` with a store); timing per ranker and per truth branch."""
    cells = sorted(fit_grid_cells().items())[:SMOKE_CELLS]
    anchors = gb.fit_anchors()
    attempts = {b["key"]: b["attempt"] for b in rt.complete_branches()}
    model = p101.load_predictor(SEEDS[0])
    policy = p101.load_policy(POLICY, "pe", ICLR_LAMBDA, SEEDS[0])
    costs = p101.endpoint_costs()
    first_member, first_entry = cells[0]
    keys = sorted({item["key"] for item in first_entry["inventory"]})
    began = time.monotonic()
    with ProcessPoolExecutor(len(keys), mp_context=multiprocessing.get_context("spawn")) as pool:
        values = list(pool.map(truth_worker, [attempts[k] for k in keys]))
    truth_seconds = time.monotonic() - began  # one branch per process: the wall is the per-branch time
    store = {"index": {k: i for i, k in enumerate(keys)}, "presence": torch.stack([v["presence"] for v in values]),
             "centers": torch.stack([v["centers"] for v in values])}
    timing, same, prefix, errors = {}, True, True, []
    with harness.deterministic_scoring():
        for ranker in RANKERS:
            began = time.monotonic()
            for member, entry in cells:
                items = entry["inventory"]
                use = policy if ranker == "J" else None
                mine = score_block(model, use, ranker, anchors[member], items, costs,
                                   store if member == first_member else None)
                theirs = p101.block(model, use, spec_of(ranker), anchors[member], items, costs, ranker == "J")
                same &= all(mine[k] == theirs[k] for k in ("rows", "linear_macs", "transition_calls", "candidates"))
                same &= ranker != "J" or mine["schedules"] == theirs["schedules"]
                if member == first_member:
                    errors += [e for t in TIMES for e in mine["engine_mse"].get(str(t), [])]
                if ranker != "J" and member == first_member:
                    pair = PAIRS[REQUESTS.index(ranker)]
                    actions = m99.action_batch(items)
                    z0 = anchors[member].to(DEVICE)
                    stopped = rp.rollout_endpoints(model, None, z0, actions, spec_of(ranker), select=t96.select_pairs,
                                                   endpoint=60)[0]
                    with torch.no_grad():
                        z = z0[None].expand(len(items), -1).clone()
                        for _ in range(60 // pair.delta):
                            z = model.carrier(z, actions, pair)
                    prefix &= bool(torch.equal(stopped, z))
            timing[ranker] = (time.monotonic() - began) / len(cells)
    finite = bool(errors) and all(e is not None and math.isfinite(e) for e in errors)
    return {"control": "scoring_path_fit", "cells": len(cells), "seconds_per_cell": timing,
            "block_equals_101": bool(same), "stopped_rollout_equals_carrier_loop": bool(prefix),
            "truth_seconds_per_branch": truth_seconds, "truth_branches": len(keys),
            "engine_errors": len(errors), "engine_error_range": [min(errors), max(errors)] if finite else None,
            "ok": bool(same and prefix and finite)}


def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the smoke is a pre-freeze control")
    output.mkdir(parents=True, exist_ok=True)
    scratch = output / "smoke-scratch"
    shutil.rmtree(scratch, ignore_errors=True)
    began = time.monotonic()
    if not payload_hash_ok():
        raise ValueError("the retrain training payload differs from its published hash")
    data = rt.IndexedTrainData(rt.load_training_data(RETRAIN), "hybrid")
    checks = [smoke_training_path(scratch, data)]
    arms, timings = smoke_arms(scratch, data)
    checks.append(arms)
    del data
    torch.cuda.empty_cache()
    checks.append(smoke_structure())
    checks.append(smoke_policy())
    checks.append(smoke_scoring())
    gate = read_json(GATE_B / "summary.json")
    checks.append({"control": "inputs_bound", "gate_b": gate["gate_b"], "J": p101.J_ICLR,
                   "ok": bool(gate["gate_b"] == "pass" and p101.J_ICLR == "J-pe-1e-04")})
    ok = all(c["ok"] for c in checks)
    wall = time.monotonic() - began
    evidence = {"schema": SCHEMA_SMOKE, "identity": IDENTITY, "run_at": utc_now(), "checks": checks,
                "projection": projection(checks, timings), "wall_seconds": wall,
                "reading": ("fit-split payload, cells, anchors and branches; the Gate-B checkpoints and #101 policies; "
                            "no evaluation-split capture or model output"), "ok": ok}
    write_json(output / "smoke.json", evidence)
    shutil.rmtree(scratch, ignore_errors=True)
    log(f"smoke {'PASSED' if ok else 'FAILED'} ({wall:.0f}s); projection {evidence['projection']}")
    for check in checks:
        log(f"  {check['control']}: {check['ok']}")
    if not ok:
        raise ValueError("smoke control differs; STOP RULE: no freeze")
    return 0


def projection(checks, timings):
    """GPU seconds: training at the full-cost update time (an upper bound), policies at the measured label time x 2
    rounds + distillation, scoring from the smoke's per-cell timings over the evaluation cells (x 2.2 on grid for
    the fixed requests' stopped rollouts), the CPU truth build (concurrent with training) from the per-branch time
    over TRUTH_WORKERS processes. Elapsed: the six members (and their J) in two waves on 4 GPUs, then scoring on 4."""
    per_update = float(np.mean(list(timings.values())))
    label_round = next(c for c in checks if c["control"] == "policy_labels_equal_101")["projected_label_round_seconds"]
    per_member = STEPS * per_update + 2 * label_round + 120.0
    members = len(TRAINED) * len(SEEDS)
    scoring_check = next(c for c in checks if c["control"] == "scoring_path_fit")
    per_cell = scoring_check["seconds_per_cell"]
    levels, grid, grid_branches = 512, 311, 4976  # Gate-B mixed cells (311 / 76 / 78 / 47); #115 grid branches
    scoring = 0.0
    for ranker in RANKERS:
        factor = 1.0 if ranker == "J" else (levels + 1.2 * grid) / levels
        scoring += per_cell[ranker] * levels * factor * len(CODE_ARMS) * len(SEEDS)
    scoring += per_cell["J"] * 480 * len(CODE_ARMS) * len(SEEDS)
    truth_ = scoring_check["truth_seconds_per_branch"] * grid_branches / TRUTH_WORKERS
    elapsed = math.ceil(members / 4) * per_member + scoring / 4
    return {"training_and_policies_gpu_seconds_upper": per_member * members, "scoring_gpu_seconds": scoring,
            "truth_wall_seconds_cpu": truth_, "total_gpu_seconds": per_member * members + scoring,
            "elapsed_hours_on_4_gpus": elapsed / 3600, "within_limit": elapsed / 3600 <= ELAPSED_LIMIT_HOURS,
            "note": "training projected at the full-cost (horizon 225 + ranking) update time: an upper bound"}


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def input_bindings(output):
    names = {
        "issue_112_retrain_handoff": gb.HANDOFF, "issue_112_fit_verdicts": gb.FIT_VERDICTS,
        "issue_112_fit_anchors": rt.anchors_path(RETRAIN), "issue_112_retrain_compute": RETRAIN / "compute.json",
        "issue_112_retrain_plan": RETRAIN / "plan.json",
        "gate_b_plan": GATE_B / "plan.json", "gate_b_summary": GATE_B / "summary.json",
        "gate_b_anchors": gb.anchors_path(GATE_B),
        "issue_101_plan": POLICY / "plan.json", "issue_101_selection": POLICY / "selection.json",
        "issue_101_pool": p101.pool_path(POLICY),
        **{f"issue_101_policy_pe_1e-04_{s}": p101.policy_path(POLICY, "pe", ICLR_LAMBDA, s) for s in SEEDS},
        **{f"issue_101_report_pe_1e-04_{s}": p101.report_path(POLICY, "pe", ICLR_LAMBDA, s) for s in SEEDS},
        "issue_115_engine_truth": TRUTH115 / "data" / "engine-truth.pt",
        "issue_115_plan": TRUTH115 / "plan.json",
        "gate_b_runner": ROOT / gb.RUNNER, "issue_112_retrain_runner": ROOT / rt.RUNNER,
        "issue_112_v1_runner": ROOT / v1.RUNNER, "issue_112_versions_runner": ROOT / versions.RUNNER,
        "issue_101_runner": ROOT / p101.RUNNER, "issue_117_runner": ROOT / "scripts/run_request_interference.py",
        "issue_99_attribution_runner": ROOT / "scripts/run_decision_chain_attribution.py",
        "issue_96_runner": ROOT / "scripts/run_tau_ad_within_checkpoint.py",
        "issue_100_runner": ROOT / "scripts/run_readout_fidelity_jepa.py",
        "issue_99_runner": ROOT / "scripts/run_slot_encoder_fix.py",
        "issue_109_cohort_plan": rt.COHORT, "issue_104_plan": rt.POWER_PLAN,
        "issue_104_campaign_branches": rt.CAMPAIGN / "branches.json",
        "scoring_harness": ROOT / "world_model/planning/scoring_harness.py",
        "request_policy_module": ROOT / p101.MODULE,
        "predictor_module": ROOT / "world_model/model/predictor.py",
        "matched_dynamics": ROOT / "world_model/training/matched_dynamics.py",
        "decision_dynamics": ROOT / "world_model/training/decision_dynamics.py",
        "stable_decision_dynamics": ROOT / "world_model/training/stable_decision_dynamics.py",
        "cnn_hybrid": ROOT / "world_model/training/cnn_hybrid.py",
        "smoke": Path(output) / "smoke.json",
    }
    names.update({f"checkpoint_unit_{c['seed']}": gb.member_path("unit", c["seed"]) for c in gb.checkpoints("unit")})
    return [{"name": name, "artifact": str(Path(path).relative_to(ROOT)), "sha256": sha256_of(path)}
            for name, path in names.items()]


def frozen_rules():
    return {
        "disposition_1_rollout": (
            f"'an additive request code matches the fused code within {CLAIM}x per-request rollout error at t = "
            f"{ENDPOINT}' (aligned with #117): per fixed request r, the geometric ratio additive / fused of the #100 "
            f"engine_position_mse at t = {ENDPOINT} on the mixed grid candidates (per level: candidate mean per "
            "(level, seed), mean over the seeds usable in both arms, log ratio; bootstrap over levels). Per request: "
            f"'supported' iff the upper bound < {CLAIM}; 'not_supported_by_this_experiment' iff the lower bound > "
            f"{CLAIM}; otherwise (or < {MIN_CLUSTERS} levels) {STOP_TOKEN}. Overall: supported iff all nine "
            f"requests are supported; not_supported_by_this_experiment iff any request is; otherwise {STOP_TOKEN}"),
        "disposition_2_ranking": (
            "'axis interaction in the request code carries measurable decision value': grid, paired by (level, seed), "
            "level-clustered, D = AUC(fused) - AUC(additive) of the nine-fixed-request mean (each request 1/9). "
            f"delta rule: supported iff mean >= {DELTA} and lower bound > 0; not_supported_by_this_experiment iff "
            f"upper bound < {DELTA}; otherwise {STOP_TOKEN}. The J contrast (each arm's own J) is reported beside "
            "it with its own token and does not enter the disposition"),
        "guards": f"a failed guard turns every token (both dispositions and every table) into {STOP_TOKEN}",
        "contrast_tokens": (f"delta rule, DESCRIPTIVE: supported iff mean >= {DELTA} with lower bound > 0; "
                            f"not_supported_by_this_experiment iff upper < {DELTA}; otherwise {STOP_TOKEN}"),
        "ratio_tokens": "the disposition-1 per-request rule at every endpoint and for factored / fused (DESCRIPTIVE)",
        "rollout_error": ("#100 engine_position_mse of every mixed grid candidate at t in "
                          f"{list(TIMES)} (fixed requests: the same rollout stopped at t; J at {ENDPOINT} only) "
                          "against its own branch at min(t, last frame); candidate mean per (level, seed), seed mean "
                          "per level, level bootstrap"),
        "coupling": ("per block, the effective FiLM scale gamma and shift beta at the nine requests (factored: "
                     "gamma = (1 + g_d)(1 + g_a) - 1, beta = b_d + b_a), two-way decomposition over (Delta, alpha): "
                     "interaction share SS_int / SS_total, the interaction share of the alpha (Delta) variation, the "
                     "variance over alpha at fixed Delta (and vice versa) relative to the mean squared norm (identity "
                     "FiLM = 0), rms; float64 on CPU; seed means; fused = the Gate-B checkpoints; no training"),
        "macs": ("cnn_hybrid.linear_macs (#103 / #114 accounting: executed linear operators per predictor call; "
                 "code and FiLM generators counted per row) per call per request, and per candidate-state (one "
                 "candidate rollout to t = 225, policy calls included), pooled over grid blocks and seeds"),
    }


def frozen_plan(output, frozen_at, smoke_evidence):
    branches, cells = gb.evaluation_set()
    structure = next(c for c in smoke_evidence["checks"] if c["control"] == "arm_structure")
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "issue": 121,
        "related_issues": [85, 96, 99, 100, 101, 103, 104, 105, 108, 109, 111, 112, 114, 116, 117],
        "role": ("request-code ablation of the LC hybrid: fused (control) vs additive vs factored-FiLM; confirmatory "
                 "on the sealed #109 evaluation split (Gate B passed); rollout error stands regardless"),
        "frozen_at": frozen_at, "frozen_before_training_and_scoring": True,
        "validation_command": VALIDATION_COMMAND, "runner": RUNNER, "runner_sha256_at_freeze": sha256_of(ROOT / RUNNER),
        "modules": {"request_code": {"path": MODULE, "sha256_at_freeze": sha256_of(ROOT / MODULE)}},
        "engine_seconds": 0,
        "question": ("does the fused MLP request code re-couple horizon and description, and does the coupling "
                     "carry rollout accuracy or decision value that an additive code lacks?"),
        "request_path_verified": (
            "CNNHybridPredictor.carrier: code = PairConditioner.code(pair) = MLP[phi(Delta) (16 sinusoids); e_alpha "
            "(128)] (144 -> 256 SiLU -> 128), broadcast over the batch; in each of the 3 FiLMBlocks a zero-initialised "
            "Linear(128, 768) gives scale gamma / shift beta: norm(h)(1 + gamma) + beta. alpha also selects the input "
            "adapter (micro / macro symbolic features of the current carrier added to the input projection); that "
            "path depends on alpha only and is identical in every arm"),
        "arms": {
            "fused": "the Gate-B LC hybrid handoff checkpoints (CNNHybridPredictor); J = the #101 J (pe, 1e-4)",
            "additive": ("AdditiveCodePredictor: c = W_delta phi(Delta) + b + W_alpha e_alpha (Linear 16 -> 128 with "
                         "bias, Embedding 3 x 128, Linear 128 -> 128 no bias); FiLM blocks unchanged"),
            "factored": ("FactoredFiLMPredictor: per block G_d: Linear(16, 768) on phi(Delta), G_a: Linear(128, 768) "
                         "on e_alpha (Embedding 3 x 128), both zero-initialised; norm(h)(1 + g_d)(1 + g_a) + b_d + b_a; "
                         "no code MLP"),
            "initialisation": ("torch.manual_seed(seed); the CNNHybridPredictor constructor first, then the request "
                               "path is replaced: trunk, heads and adapters start equal to the fused member of the "
                               "same seed (smoke-checked)"),
            "parameters": {a: structure[f"{a}_parameters"] for a in CODE_ARMS},
            "rankers": {"fixed": list(REQUESTS), "J": ("#101 pe DP lambda 1e-4 policy trained for each new arm on the "
                                                       "#109 policy pool (2 DAgger rounds, distillation)")},
            "controls": {"prior": "no-model action prior (Gate-B rule, fit-verdicts.json)",
                         "wrong_anchor": "each arm's J from the 480 fit-split anchors over the 16 grid candidates"}},
        "training": {
            "recipe": ("#112 v3 LC exactly as the Gate-B retrain (rt.implementation; only s99.new_predictor returns "
                       "the arm's class): #77 local loss + long-horizon supervision to 225 + decision-contrastive "
                       "ranking loss, truncation 45, skip-on-nonfinite, AdamW lr 1e-4 wd 1e-4, clip 1, batch 64, "
                       f"{STEPS} updates (the matched budget), PAIRS[step % 9], fold 'all', one member per group"),
            "data": "the v3 retrain payload (#109 fit split), hash-checked",
            "seeds": list(SEEDS),
            "replication_guard": ("smoke: the training path configured as fused equals rt.implementation().train_group "
                                  f"bit for bit after {SMOKE_STEPS} full-cost updates (parameters, every local loss)"),
            "usable": f"complete, at most {SKIP_CAP} skipped updates, no #117 end-of-training excursion (rt.excursion)",
            "typed_failures": "an unusable member is a typed terminal failure: not retrained, no policy, its cells drop",
            "policy": ("#101 pe recipe: rp.new_policy(seed), 2 DAgger rounds of rp.window_rows (lambda 1e-4) on the "
                       "#101 pool, rp.distil_round, under deterministic_scoring")},
        "evaluation": {"split": "the #109 evaluation split as in Gate B (stream issue109-evaluation, mixed cells)",
                       "accounting": gb.evaluation_accounting(branches, cells),
                       "anchors": "the Gate-B E99 anchor store", "primary_inventory": PRIMARY,
                       "descriptive": list(gb.DESCRIPTIVE_INVENTORIES),
                       "cost": "tie_free (EndpointCosts.row against the anchor)",
                       "cell_auc": "m99.view; cells (level, seed); bootstrap over levels (10000 draws, PCG64 7201)",
                       "rollout": "rp.rollout_endpoints (the #101 path; fixed requests stopped at each t on grid)",
                       "engine_truth": ("built after the freeze: s99.shot_labels of every mixed grid branch at "
                                        f"positions min(t, last), t in {list(TIMES)}")},
        "decision_rule": frozen_rules(),
        "guards": {"G1_isolation": "payload on fit levels; scored cells on evaluation levels; disjoint rosters",
                   "G2_scoring_policy": "every record carries the #111 SCORING_POLICY",
                   "G3_chronology": "frozen_at <= training <= policies <= first score; truth after the freeze",
                   "G4_checkpoint_binding": "Gate-B, trained member and policy hashes equal their reports",
                   "G5_replication": ("the smoke training-path replication passed, and the fused rows (and J "
                                      "schedules and linear MACs) equal the #101 records on every inventory, cell "
                                      "and seed"),
                   "G6_engine_truth": ("the t = 225 truth equals the #115 store and the fused fixed micro t = 225 "
                                       "engine errors equal the #115 Gate-B default records"),
                   "G7_nonfinite": f"fused J grid nonfinite share <= {NONFINITE_SHARE_MAX}",
                   "failure": f"a failed guard turns every token into {STOP_TOKEN}"},
        "prohibited": ["reading evaluation captures for training or selection", "re-scoring after publication",
                       "changing an arm, rule or recipe after the freeze", "retrying or replacing a typed failure"],
        "caps": {"gpu_seconds": GPU_CAP_SECONDS, "wall_seconds_summed_over_processes": WALL_CAP_SECONDS,
                 "projection": smoke_evidence["projection"],
                 "scope_rule": (f"if the projected elapsed time on 4 GPUs exceeds {ELAPSED_LIMIT_HOURS} h, finish "
                                "blocked with the projection; never cut scope")},
        "disclosure": ("Known before the freeze: Gate B (unit 0.7444, prior 0.7203 on grid), #101 (J grid 0.706), "
                       "#114, #105, #115 (Gate-B rollout error t = 225 F-1/5/15-micro 0.030 / 0.038 / 0.042), #116 "
                       "(the 9000-update recipe looks undertrained on the fit split), #117 (1.5x rule; joint vs "
                       "specialists). No additive or factored model existed; no coupling statistic of the Gate-B "
                       "checkpoints was computed before the freeze. Hardware: RTX 5090 instead of the #85 RTX 3090. "
                       "Pre-freeze: the analysis and rendering code was exercised on synthetic random rows over "
                       "fit-split cells posing as the evaluation set (no model output, no evaluation capture; random-"
                       "init models for the coupling code); a first smoke passed on a revision that differed in the "
                       "findings layout, the guard-failure propagation into the per-request disposition tokens and "
                       "this disclosure; a second passed on a revision that differed only in this disclosure; the "
                       "smoke was rerun on the frozen runner (logs/smoke-first-revision.log, "
                       "logs/smoke-second-revision.log, logs/smoke.log)."),
        "smoke_evidence": smoke_evidence,
        "inputs": input_bindings(output),
        "claim_boundary": ("one look at the sealed #109 evaluation split; grid ranking primary, other inventories "
                           "DESCRIPTIVE; rollout error stands regardless of Gate B; tie-free cost; DESCRIPTIVE "
                           "intervals; NovPhy E99 carriers, the LC hybrid at the Gate-B budget only"),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-121 protocol")
    if not plan.get("frozen_before_training_and_scoring") or not plan["smoke_evidence"].get("ok"):
        raise ValueError("plan.json does not declare a passed-smoke freeze")
    for entry in plan["inputs"]:
        if not amendments.matches(entry["artifact"], entry["sha256"], sha256_of(ROOT / entry["artifact"])):
            raise ValueError(f"frozen input {entry['name']} changed after the freeze")
    for name, module in plan["modules"].items():
        if sha256_of(ROOT / module["path"]) != module["sha256_at_freeze"]:
            raise ValueError(f"module {name} changed after the freeze")
    return plan


def prepare(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        load_plan(output)
        log("existing frozen plan validated")
        return 0
    for name in ("records", "models", "data", "training"):
        if (output / name).exists():
            raise ValueError(f"{name} exist before the freeze; refusing")
    evidence = read_json(output / "smoke.json")
    if not evidence.get("ok"):
        raise ValueError("smoke failed; STOP RULE: no freeze")
    if not evidence["projection"]["within_limit"]:
        raise ValueError(f"projection {evidence['projection']['elapsed_hours_on_4_gpus']:.1f} h exceeds the limit")
    plan = frozen_plan(output, utc_now(), evidence)
    (output / "plan.json").write_text(json_text(plan))
    load_plan(output)
    log("frozen plan published")
    return 0


# ---------------------------------------------------------------------------
# publication and validation
# ---------------------------------------------------------------------------

def publish(output):
    output = Path(output)
    plan = load_plan(output)
    compute = compute_tables(output, plan)
    for name, text in rendered(plan, compute, output).items():
        (output / name).write_text(text)
    log(f"published: { {k: v['token'] for k, v in compute['dispositions'].items()} }")
    return 0


def spot_checks(output):
    """Re-score, in this process, the first seed of additive F-1-macro (rows and engine errors at every t) and
    factored J (rows and schedules) on every grid cell."""
    load_plan(output)
    _, cells = gb.evaluation_set()
    anchors = gb.load_anchors(GATE_B)["anchors"]
    store = load_truth(output)
    seed = SEEDS[0]
    grid = {**{i: {"mixed": {}} for i in INVENTORIES}, PRIMARY: cells[PRIMARY]}
    results = {}
    with harness.deterministic_scoring() as policy_state:
        for arm, ranker in (("additive", "F-1-macro"), ("factored", "J")):
            kept = read_record(record_path(output, arm, ranker, seed))
            if kept["typed_failure"]:
                results[f"{arm}|{ranker}"] = None
                continue
            fresh = score_record(output, grid, anchors, store, arm, ranker, seed, policy_state)
            fields = ("rows", "engine_mse", "schedules", "linear_macs")
            results[f"{arm}|{ranker}"] = all(
                fresh["inventories"][PRIMARY][m].get(f) == kept["inventories"][PRIMARY][m].get(f)
                for m in cells[PRIMARY]["mixed"] for f in fields)
    return {"seed": seed, "checks": results, "pass": all(v is not False for v in results.values())}


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
    log(f"validation passed ({time.monotonic() - began:.0f}s): tables re-derived and byte-compared; re-scored "
        f"{spot['checks']} seed {spot['seed']}")
    return 0


def dry_run(output):
    _, cells = gb.evaluation_set()
    log(f"output {Path(output)}; arms {CODE_ARMS} (trained {TRAINED} x {len(SEEDS)} seeds); "
        f"score tasks {len(score_tasks())}")
    log(f"mixed cells { {i: len(cells[i]['mixed']) for i in INVENTORIES} }; grid branches {len(grid_keys(cells))}")
    for arm in CODE_ARMS:
        log(f"  {arm}: {rc.parameter_split(rc.new_predictor(arm), arm)}")
    return 0


MODES = {"dry-run": dry_run, "smoke": smoke, "prepare": prepare, "train": train, "truth": truth, "score": score,
         "publish": publish, "validate": validate}
SHARDED = ("train", "score")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker", type=int, default=0)
    args = parser.parse_args()
    mode = next(m for m in MODES if getattr(args, m.replace("-", "_")))
    if mode in SHARDED:
        return MODES[mode](OUTPUT, args.workers, args.worker)
    return MODES[mode](OUTPUT)


if __name__ == "__main__":
    raise SystemExit(main())
