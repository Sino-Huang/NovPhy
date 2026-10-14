"""Issue-105: independent axes vs hierarchy, scored once on the #104 sealed #109 evaluation split.

Binding runner module: scripts/run_hierarchy_baselines.py
Exact validation command: python -u -m scripts.run_hierarchy_baselines --validate

#105 asks whether horizon (Delta) and description (alpha) add ranking value as independent request
coordinates over hierarchies that tie each time scale to one abstraction level (H-JEPA, CW-VAE, MSPred,
Director). This runner

1. restricts the argmax of the #101 request policies (J = the ICLR prediction-error policy, J_dec = the
   decision-valued policy, both on the Gate-B LC hybrid predictors) to tied request sets with the #114
   selector kinds (``world_model/training/request_selectors.py``): J_diag {(1,cont),(5,micro),(15,macro)},
   J_monotone (alpha may only coarsen as Delta grows, given the previous request) and the anti-diagonal
   control {(1,macro),(5,micro),(15,cont)}; zero training;
2. trains a three-level hierarchical latent predictor (``world_model/training/hierarchical_dynamics.py``:
   level k predicts only at (Delta_k, alpha_k) on the diagonal, top-down coupled) on the same 236-d E99
   carrier with the #112 v3 LC recipe (the Gate-B predictor's recipe, matched in parameters and updates)
   on the #112 v3 retrain's #109 fit-split training payload, 3 seeds (``--train``);
3. scores the #78 external temporal-adaptation baselines (TAWM, VLWM; their published checkpoints, on
   their own #71 carrier parsed from the same evaluation anchor frames) where the interface allows;
4. scores once, in processes that did not train, under ``scoring_harness.deterministic_scoring()``:
   ranking AUC (tie-free cost) on the Gate-B cells and the #100 engine-referenced rollout error at
   t = 225 on every grid candidate, with the nine fixed requests, C*, the no-model action prior and the
   wrong-anchor control (``--anchors``, ``--truth``, ``--score``);
5. publishes compute.json, summary.json and findings.md.

Modes: --dry-run --smoke --prepare --train --anchors --truth --score --publish --validate. --train and
--score take --workers N --worker K (one process per GPU; CUDA_VISIBLE_DEVICES set by the caller). Zero
engine seconds. Intervals: member (level)-clustered percentile bootstrap (10000 draws, PCG64 7201),
DESCRIPTIVE.
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
import time
import types

import numpy as np
import torch

from scripts import run_decision_aware_policy as p101
from scripts import run_decision_chain_attribution as m99
from scripts import run_decision_dynamics_controls as c2
from scripts import run_decision_dynamics_retrain as rt
from scripts import run_decision_dynamics_versions as versions
from scripts import run_decision_relevant_dynamics as v1
from scripts import run_external_temporal_baselines as e78
from scripts import run_gate_b as gb
from scripts import run_lookahead_control as wlc
from scripts import run_readout_fidelity_jepa as r100
from scripts import run_slot_encoder_fix as s99
from scripts import run_tau_ad_within_checkpoint as t96
from world_model.planning import scoring_harness as harness
from world_model.training import hierarchical_dynamics as hd
from world_model.training import request_selectors as rs
from world_model.training.cnn_hybrid import PAIRS
from world_model.training.matched_dynamics import ContinuousDynamics, MatchedController, linear_macs

ROOT = v1.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-105-hierarchy-baselines-v1"
GATE_B = gb.OUTPUT
RETRAIN = rt.OUTPUT
POLICY = p101.OUTPUT
EXTERNAL = e78.OUTPUT

IDENTITY = "issue-105-hierarchy-baselines-v1"
SCHEMA_PLAN = "issue_105_hierarchy_baselines_plan_v1"
SCHEMA_SMOKE = "issue_105_hierarchy_baselines_smoke_v1"
SCHEMA_PREDICTOR = "issue_105_hierarchical_predictor_v1"
SCHEMA_GROUP = "issue_105_hierarchical_training_group_v1"
SCHEMA_DATA = rt.SCHEMA_DATA
SCHEMA_UPDATES = "issue_105_hierarchical_updates_v1"
SCHEMA_TRAINED = "issue_105_hierarchical_trained_v1"
SCHEMA_ANCHORS = "issue_105_f77_anchors_v1"
SCHEMA_TRUTH = "issue_105_engine_truth_v1"
SCHEMA_RECORD = "issue_105_record_v1"
SCHEMA_COMPUTE = "issue_105_compute_v1"
SCHEMA_REPORT = "issue_105_report_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_hierarchy_baselines --validate"
RUNNER = "scripts/run_hierarchy_baselines.py"
MODULE = "world_model/training/hierarchical_dynamics.py"
SELECTORS = "world_model/training/request_selectors.py"

DEVICE = "cuda"
SEEDS = tuple(t96.SEEDS)
INVENTORIES = gb.INVENTORIES
PRIMARY = gb.GATE_INVENTORY
PRIMARY_COST = harness.PRIMARY_COST
DELTA = v1.DELTA
STOP_TOKEN = v1.STOP_TOKEN
NONFINITE_SHARE_MAX = v1.NONFINITE_SHARE_MAX
ENDPOINT = t96.ENDPOINT
FIXED_ARMS = t96.FIXED_ARMS
C_STAR = p101.C_STAR
ARM = rt.ARM  # the #112 v3 LC recipe
FAMILY = "hierarchical"
STEPS = rt.STEPS
SKIP_CAP = rt.SKIP_CAP
SMOKE_STEPS = 12
SMOKE_CELLS = 3
TRUTH_WORKERS = 16
GPU_CAP_SECONDS = 12 * 3600.0
WALL_CAP_SECONDS = 24 * 3600.0  # summed over processes (2 GPUs: ~12 h elapsed at most)

POLICIES = {"J": p101.J_ICLR}  # J_dec: the #101 selection (dec, lambda 1e-1), resolved by policy_arms()
RESTRICTIONS = {"": None, "_diag": "diagonal", "_monotone": "monotone", "_anti": "anti_diagonal"}
POLICY_ARMS = tuple(f"{base}{suffix}" for base in ("J", "J_dec") for suffix in RESTRICTIONS)
H_ARMS = tuple(f"H-{p.delta}-{p.abstraction.value}" for p in hd.LEVEL_PAIRS)
EXTERNAL_FIXED = tuple(f"{arm.upper()}-{h}" for arm in e78.ARMS for h in e78.HORIZONS)
TAWM_ADAPTIVE = "TAWM-adaptive"
EXTERNAL_ARMS = (*EXTERNAL_FIXED, TAWM_ADAPTIVE)
ARMS = (*FIXED_ARMS, C_STAR, *POLICY_ARMS, *H_ARMS, *EXTERNAL_ARMS)
WRONG_ARMS = ("J_diag", *H_ARMS)
SCHEDULED = POLICY_ARMS + (TAWM_ADAPTIVE,)
UNSUPPORTED = {
    "VLWM-adaptive": ("#78 trained a horizon controller for TAWM only (its adaptive evaluation arm); no VLWM "
                      "controller exists"),
    "TAWM/VLWM diagonal, monotone and anti-diagonal arms": ("the #78 ports are continuous-only (ContinuousDynamics, "
                                                            "horizon conditioning only): no description axis alpha"),
    "TAWM/VLWM wrong-anchor control": ("their carrier is the #71 parser's; the fit-split wrong-anchor pool exists "
                                       "only as E99 carriers"),
}
REHEARSAL = {
    "gpu_seconds": 1092.0, "wall_seconds": 1092.0,
    "what": ("pre-freeze rehearsal of every post-freeze mode (prepare, train at 12 updates, anchors, truth, score, "
             "publish, validate) in a throwaway output on 24 #109 fit levels (14 grid / 19 angle mixed cells) posing "
             "as the evaluation set; the scratch output was deleted. It exercised code paths only, on training-split "
             "cells with an untrained (12-update) hierarchical model; no evaluation-split capture was read"),
}

read_json = v1.read_json
write_json = v1.write_json
json_text = v1.json_text
sha256_of = v1.sha256_of
atomic_torch = v1.atomic_torch
fmt_interval = v1.fmt_interval


def log(message):
    print(f"[issue-105] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# the hierarchical predictor under the frozen #112 v3 LC recipe
# ---------------------------------------------------------------------------

def level_request(step):
    """Update ``step`` trains level step % 3 (the v1 PAIRS[step % 9] rule over the three levels)."""
    return H_ARMS[step % len(hd.LEVEL_PAIRS)]


@functools.lru_cache(maxsize=None)
def implementation():
    """A private copy of the v1 runner carrying the frozen v3 spec exactly as ``rt.implementation()``
    (LC arm, v3 LONG/RANK, truncated recursion, skipping + logging ensemble), bound to this output, with
    the hierarchical predictor as the model and its three level requests as the update schedule."""
    v3 = versions.implementation(rt.VERSION)
    loader = importlib.util.spec_from_file_location("issue105_hierarchical_implementation", ROOT / v1.RUNNER)
    impl = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(impl)
    impl.OUTPUT = OUTPUT
    impl.IDENTITY = IDENTITY
    impl.SCHEMA_PREDICTOR = SCHEMA_PREDICTOR
    impl.SCHEMA_GROUP = SCHEMA_GROUP
    impl.SCHEMA_DATA = SCHEMA_DATA
    impl.RUNNER = RUNNER
    impl.MODEL_MODULE = MODULE
    impl.ARMS = (ARM,)
    impl.ARM_DEFINITIONS = {ARM: v3.ARM_DEFINITIONS[ARM]}
    impl.LONG = v3.LONG
    impl.RANK = v3.RANK
    impl.dd = types.SimpleNamespace(Ensemble=rt.logging_ensemble(v3.dd.Ensemble), local_loss=rt.logged_local_loss,
                                    horizon_at=v3.dd.horizon_at, ramp_at=v3.dd.ramp_at,
                                    long_and_rank_loss=v3.dd.long_and_rank_loss)
    impl.step_pair = lambda family, step: hd.LEVEL_PAIRS[step % len(hd.LEVEL_PAIRS)]
    predictors = types.SimpleNamespace(**vars(s99))
    predictors.new_predictor = lambda family: hd.HierarchicalPredictor()
    impl.s99 = predictors
    return impl


def member_path(output, seed):
    return implementation().member_path(output, ARM, FAMILY, "all", seed)


def updates_path(output, seed):
    return Path(output) / "training" / f"updates--seed{seed}.pt"


def trained_path(output, seed):
    return Path(output) / "training" / f"trained--seed{seed}.json"


def ledger_path(output, phase):
    return Path(output) / "ledger" / f"{phase}.json"


def record_path(output, arm, seed):
    return Path(output) / "records" / f"{arm}--seed{seed}.json"


def wrong_path(output, arm, seed):
    return Path(output) / "records" / f"wrong--{arm}--seed{seed}.json"


def f77_anchors_path(output):
    return Path(output) / "data" / "anchors-f77-evaluation.pt"


def truth_path(output):
    return Path(output) / "data" / "engine-truth-grid.pt"


def mark(output, phase, wall, gpu, **extra):
    path = ledger_path(output, phase)
    block = read_json(path) if path.is_file() else {"wall_seconds": 0.0, "gpu_seconds": 0.0}
    write_json(path, {**block, "wall_seconds": block["wall_seconds"] + wall, "gpu_seconds": block["gpu_seconds"] + gpu,
                      "last_at": utc_now(), **extra})
    total = [read_json(p) for p in sorted((Path(output) / "ledger").glob("*.json"))]
    if (sum(b["gpu_seconds"] for b in total) > GPU_CAP_SECONDS
            or sum(b["wall_seconds"] for b in total) > WALL_CAP_SECONDS):
        raise ValueError(f"{STOP_TOKEN}: compute cap exceeded")


def training_data():
    """The #112 v3 retrain's #109 fit-split payload (read-only, mmap), checked against its published hash."""
    payload = rt.load_training_data(RETRAIN)
    return payload


def train(output, workers, worker):
    output = Path(output)
    load_plan(output)
    impl = implementation()
    torch.backends.cudnn.benchmark = True
    mine = [(i, s) for i, s in enumerate(SEEDS) if i % workers == worker and not trained_path(output, s).is_file()]
    if not mine:
        return 0
    expected = read_json(RETRAIN / "compute.json")["data"]["training_data_sha256"]
    if sha256_of(rt.data_path(RETRAIN)) != expected:
        raise ValueError("the retrain training payload differs from its published hash")
    data = rt.IndexedTrainData(training_data(), "hybrid")
    for index, seed in mine:
        began = time.monotonic()
        report, ensemble = impl.train_group(output, ARM, FAMILY, index, [("all", seed)], data)
        atomic_torch(updates_path(output, seed), {
            "schema": SCHEMA_UPDATES, "plan_identity": IDENTITY, "seed": seed,
            "requests": [level_request(k) for k in range(STEPS)], "skipped": ensemble.skipped.cpu(),
            "logs": ensemble.log_tensors()})
        del ensemble
        torch.cuda.empty_cache()
        write_json(trained_path(output, seed), {"schema": SCHEMA_TRAINED, "plan_identity": IDENTITY, "seed": seed,
                                                "trained_at": utc_now(), "gpu_seconds": report["gpu_seconds"],
                                                "parameters": report["parameters_per_member"],
                                                "sha256": sha256_of(member_path(output, seed))})
        mark(output, f"train-seed{seed}", time.monotonic() - began, report["gpu_seconds"])
        log(f"trained hierarchical seed {seed}: {report['gpu_seconds']:.0f}s")
    return 0


def member_state(output, seed):
    """Training accounting of one hierarchical member and its usability (retrain rule: complete, at most
    SKIP_CAP skips, no #117 end-of-training excursion)."""
    report = read_json(implementation().group_path(output, ARM, FAMILY, SEEDS.index(seed)))
    updates = torch.load(updates_path(output, seed), map_location="cpu", weights_only=True)
    trained = read_json(trained_path(output, seed))
    member = report["members"][0]
    local, norms = updates["logs"]["local"][:, 0].double(), updates["logs"]["grad_norm"][:, 0].double()
    requests = updates["requests"]
    end_state = rt.excursion(local, norms, requests) if member["complete"] else None
    skipped = int(updates["skipped"][0])
    usable = bool(member["complete"] and skipped <= SKIP_CAP and len(local) == STEPS and not end_state["excursion"])
    last = report["trace"][-1]
    return {"seed": seed, "complete": member["complete"], "retired_at": member["retired_at"], "skipped": skipped,
            "logged_updates": len(local), "gpu_seconds": report["gpu_seconds"], "parameters": trained["parameters"],
            "peak_memory_gb": report["peak_memory_gb"], "pool_sizes": report["pool_sizes"][0],
            "final_trace": {k: last[k] for k in ("loss", "local", "long", "rank", "grad_norm")},
            "end_state": end_state, "usable": usable, "trained_at": trained["trained_at"],
            "sha256": trained["sha256"],
            "typed_failure": None if usable else ("retired" if not member["complete"] else
                                                  "skip_cap" if skipped > SKIP_CAP else "final_checkpoint_excursion")}


def load_hierarchical(output, seed):
    """The trained member, or (None, typed failure)."""
    state = member_state(output, seed)
    if state["typed_failure"]:
        return None, state["typed_failure"]
    if sha256_of(member_path(output, seed)) != state["sha256"]:
        raise ValueError(f"hierarchical seed {seed}: checkpoint differs from its training report")
    return implementation().load_member(output, ARM, FAMILY, "all", seed), None


# ---------------------------------------------------------------------------
# models of every arm
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def policy_arms():
    """Restricted arm -> (#101 policy arm, kind, lambda)."""
    selection = p101.load_selection(POLICY)
    bases = {"J": p101.J_ICLR, "J_dec": selection["kinds"]["dec"]["arm"]}
    out = {}
    for base, source in bases.items():
        kind, lam, _ = p101.arm_policy(source, selection)
        for suffix in RESTRICTIONS:
            out[f"{base}{suffix}"] = (source, kind, lam)
    return out


def restriction_spec(arm):
    base = "J_dec" if arm.startswith("J_dec") else "J"
    name = RESTRICTIONS[arm[len(base):]]
    if name is None:
        return {"kind": "joint"}
    return {"diagonal": rs.diagonal_spec(PAIRS), "anti_diagonal": rs.anti_diagonal_spec(PAIRS),
            "monotone": rs.monotone_spec()}[name]


def external_cell(seed, arm):
    return EXTERNAL / f"seed-{seed}" / arm


@functools.lru_cache(maxsize=None)
def external_plan():
    return read_json(EXTERNAL / "plan.json")


def load_external(seed, arm):
    """A #78 predictor (``tawm`` / ``vlwm``) with its binding and budget checks (e78.load_predictor rules)."""
    plan = external_plan()
    value = torch.load(external_cell(seed, arm) / "predictor.pt", map_location="cpu", weights_only=True)
    e78.check_binding(value, e78.binding(plan, seed, arm, "predictor"))
    model = ContinuousDynamics(plan["capacity"]["continuous_width"])
    model.load_state_dict(value["model"], strict=True)
    if value["step"] != plan["training"]["steps"] or value["pair_counts"] != e78.horizon_counts(plan, arm):
        raise ValueError(f"incomplete #78 {arm} predictor")
    return model.to(DEVICE).eval()


def load_external_controller(seed):
    plan = external_plan()
    value = torch.load(external_cell(seed, "tawm") / "controller.pt", map_location="cpu", weights_only=True)
    e78.check_binding(value, e78.binding(plan, seed, "tawm", "controller"))
    if value["round"] != 1 or value["step"] != plan["controller"]["steps_per_round"]:
        raise ValueError("the #78 TAWM controller is incomplete")
    controller = MatchedController(pure=True)
    controller.load_state_dict(value["model"], strict=True)
    return controller.to(DEVICE).eval()


def external_files():
    return {f"external_{arm}_{s}_{c}": external_cell(s, arm) / f"{c}.pt"
            for s in SEEDS for arm, c in (("tawm", "predictor"), ("vlwm", "predictor"), ("tawm", "controller"))}


def arm_setup(output, arm, seed):
    """(model, policy, spec, carrier, extra) of an arm; model None = typed failure (extra['typed_failure'])."""
    if arm in FIXED_ARMS or arm == C_STAR:
        name = "c_star" if arm == C_STAR else "unit"
        return p101.load_predictor(seed, name), None, t96.arm_spec(arm), "E99", {
            "predictor_sha256": sha256_of(gb.member_path(name, seed))}
    if arm in POLICY_ARMS:
        source, kind, lam = policy_arms()[arm]
        return p101.load_predictor(seed), p101.load_policy(POLICY, kind, lam, seed), restriction_spec(arm), "E99", {
            "predictor_sha256": sha256_of(gb.member_path("unit", seed)),
            "policy": {"arm": source, "kind": kind, "lambda": lam,
                       "sha256": sha256_of(p101.policy_path(POLICY, kind, lam, seed))}}
    if arm in H_ARMS:
        model, failure = load_hierarchical(output, seed)
        pair = hd.LEVEL_PAIRS[H_ARMS.index(arm)]
        return model, None, {"kind": "level", "pair": pair}, "E99", {
            "typed_failure": failure, "predictor_sha256": None if model is None else sha256_of(member_path(output, seed))}
    if arm in EXTERNAL_FIXED:
        name, horizon = arm.split("-")
        return load_external(seed, name.lower()), None, t96.arm_spec(f"C-F-{horizon}"), "F77", {
            "predictor_sha256": sha256_of(external_cell(seed, name.lower()) / "predictor.pt")}
    if arm == TAWM_ADAPTIVE:
        return load_external(seed, "tawm"), load_external_controller(seed), {"kind": "joint"}, "F77", {
            "predictor_sha256": sha256_of(external_cell(seed, "tawm") / "predictor.pt"),
            "controller_sha256": sha256_of(external_cell(seed, "tawm") / "controller.pt")}
    raise ValueError(f"unknown arm {arm}")


# ---------------------------------------------------------------------------
# anchors (#71 carrier for the #78 ports) and engine truth at t = 225
# ---------------------------------------------------------------------------

def f77_anchor(adapter, attempt):
    """The #71 (F77) batch-1 parse of the same decision frame as ``rt.anchor_carrier``."""
    _, observation_root, frames, _ = s99.shot_frames(attempt)
    frame = frames[0]
    anchor = {"observation_identity": frame["agent_observation"]["identity"], "fixed_step": frame["fixed_step"],
              "fixed_time_seconds": frame["fixed_time_seconds"],
              "frame_path": str(observation_root / frame["agent_observation"]["relative_path"])}
    return wlc.encode_carrier(adapter, anchor, DEVICE).float().cpu()


def anchors(output):
    output = Path(output)
    load_plan(output)
    began = time.monotonic()
    branches, _ = gb.evaluation_set()
    first = rt.anchor_branches(gb.complete_branches(branches))
    with harness.deterministic_scoring() as policy:
        adapter = wlc.load_adapter(DEVICE)
        store = {member: f77_anchor(adapter, b["attempt"]) for member, b in first.items()}
    atomic_torch(f77_anchors_path(output), {"schema": SCHEMA_ANCHORS, "plan_identity": IDENTITY,
                                            "scoring_policy": policy, "parsed_at": utc_now(), "anchors": store,
                                            "frames": {m: b["key"] for m, b in first.items()}})
    wall = time.monotonic() - began
    mark(output, "anchors-f77", wall, wall, levels=len(store))
    log(f"F77 anchors parsed: {len(store)} levels ({wall:.0f}s)")
    return 0


def load_f77_anchors(output):
    store = torch.load(f77_anchors_path(output), map_location="cpu", weights_only=True)
    if store.get("schema") != SCHEMA_ANCHORS or store.get("plan_identity") != IDENTITY:
        raise ValueError("F77 anchor store binding differs")
    return store


UNAVAILABLE_MICRO = {"predicates": {name: {"availability": "unavailable"} for name in ("contact", "supports")}}
UNAVAILABLE_MACRO = {"predicates": {name: {"availability": "unavailable"}
                                    for name in ("steady-state", "structure-unstable")}}


def engine_truth(attempt):
    """Process-pool worker: the #99 engine label fields presence and projected centers
    (``run_issue_70_parser_repair.targets``, the ``s99.shot_labels`` row) of the shot at observed position
    min(225, last frame); the relation / macro fields are not built (they do not enter the metric)."""
    from scripts.canonical_native_trace import NativeTrace
    from scripts.run_issue_70_parser_repair import targets
    torch.set_num_threads(1)
    _, _, frames, _ = s99.shot_frames(attempt)
    segment = read_json(Path(attempt) / "shot-1" / "segment.json")
    position = min(ENDPOINT, len(frames) - 1)
    trace = NativeTrace(segment["native_root"])
    if [f["fixed_step"] for f in trace.manifest["frame_records"]] != [f["fixed_step"] for f in frames]:
        raise ValueError("native physics and observations do not share exact endpoints")
    step = frames[position]["fixed_step"]
    sample = next(s for chunk in trace.chunks() for s in chunk["fixed_step_samples"] if s["fixed_step"] == step)
    names = s99.vocabulary()
    slots = set(names)
    filtered = {**sample, "entities": [e for e in sample["entities"] if e["scenario_object_id"] in slots]}
    row = targets(filtered, frames[position]["capture_metadata"], UNAVAILABLE_MICRO, UNAVAILABLE_MACRO, names)
    return {"position": position, "frame_count": len(frames), "presence": row["presence"].bool(),
            "centers": row["centers"].float()}


def grid_branches(cells):
    """Every candidate of every mixed grid cell (level order, ordinal order)."""
    return [(member, item["key"]) for member, entry in sorted(cells[PRIMARY]["mixed"].items())
            for item in entry["inventory"]]


def truth(output):
    output = Path(output)
    load_plan(output)
    began = time.monotonic()
    branches, cells = gb.evaluation_set()
    attempts = {b["key"]: b["attempt"] for b in branches}
    keys = [key for _, key in grid_branches(cells)]
    with ProcessPoolExecutor(TRUTH_WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        values = list(pool.map(engine_truth, [attempts[k] for k in keys], chunksize=8))
    atomic_torch(truth_path(output), {"schema": SCHEMA_TRUTH, "plan_identity": IDENTITY, "built_at": utc_now(),
                                      "keys": keys, "truth": dict(zip(keys, values))})
    wall = time.monotonic() - began
    mark(output, "truth", wall, 0.0, branches=len(keys), processes=TRUTH_WORKERS)
    log(f"engine truth at t={ENDPOINT}: {len(keys)} grid branches ({wall:.0f}s)")
    return 0


def load_truth(output):
    store = torch.load(truth_path(output), map_location="cpu", weights_only=True)
    if store.get("schema") != SCHEMA_TRUTH or store.get("plan_identity") != IDENTITY:
        raise ValueError("engine truth binding differs")
    return store["truth"]


def engine_error(z, label):
    """#100 ``engine_position_mse`` of one endpoint carrier; None for a nonfinite endpoint (typed failure)
    or when no slot is present in the engine frame."""
    if not bool(torch.isfinite(z).all()):
        return None
    return r100.engine_position_mse(z.tolist(), {"presence": label["presence"][None], "centers": label["centers"][None]},
                                    0)


# ---------------------------------------------------------------------------
# scoring (deterministic_scoring, processes that did not train)
# ---------------------------------------------------------------------------

def rollout(model, policy, spec, z0, actions, schedules):
    if spec["kind"] == "level":
        return hd.fixed_rollout(model, z0.to(DEVICE), actions, spec["pair"], hd.linear_macs(model, spec["pair"]))
    return rs.rollout_endpoints(model, policy, z0.to(DEVICE), actions, spec, select=t96.select_pairs,
                                record_schedules=schedules)


def block(model, policy, spec, z0, items, costs, schedules, truth_map=None):
    actions = m99.action_batch(items)
    z_end, timing = rollout(model, policy, spec, z0, actions, schedules)
    out = {"rows": [costs.row(item["ordinal"], z_end[slot].cpu(), z0.cpu()) for slot, item in enumerate(items)],
           "candidates": len(items), **{k: v for k, v in timing.items() if k != "schedules"}}
    if schedules:
        out["schedules"] = {str(item["ordinal"]): t96.rle(timing["schedules"][k]) for k, item in enumerate(items)}
    if truth_map is not None:
        out["engine_mse"] = [engine_error(z_end[slot].cpu(), truth_map[item["key"]]) for slot, item in enumerate(items)]
    return out


def ordered_cells(cells):
    return p101.ordered_cells(cells)


def score_record(output, cells, anchor_sets, truth_map, arm, seed, policy_state):
    costs = p101.endpoint_costs()
    began = time.monotonic()
    model, policy, spec, carrier, extra = arm_setup(output, arm, seed)
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "scoring_policy": policy_state,
            "scored_at": utc_now(), "arm": arm, "seed": seed, "carrier": carrier, "engine_seconds": 0,
            "typed_failure": extra.pop("typed_failure", None), **extra}
    if model is None:
        return {**base, "inventories": {}}
    schedules = arm in SCHEDULED
    store = anchor_sets[carrier]
    inventories = {i: {} for i in INVENTORIES}
    for inventory, member, entry in ordered_cells(cells):
        inventories[inventory][member] = block(model, policy, spec, store[member], entry["inventory"], costs, schedules,
                                               truth_map if inventory == PRIMARY else None)
    torch.cuda.synchronize()
    return {**base, "inventories": inventories, "seconds": time.monotonic() - began}


def wrong_record(output, arm, seed, policy_state):
    """The arm from every fit-split anchor over the shared grid candidates (Gate-B wrong-anchor rule)."""
    costs = p101.endpoint_costs()
    began = time.monotonic()
    model, policy, spec, _, extra = arm_setup(output, arm, seed)
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "scoring_policy": policy_state,
            "scored_at": utc_now(), "arm": f"wrong--{arm}", "seed": seed, "engine_seconds": 0,
            "typed_failure": extra.pop("typed_failure", None), **extra}
    if model is None:
        return {**base, "wrong_anchor": None}
    items = gb.inventory_items(PRIMARY)
    fit = gb.fit_anchors()
    rows = {member: block(model, policy, spec, fit[member], items, costs, False) for member in sorted(fit)}
    return {**base, "wrong_anchor": rows, "seconds": time.monotonic() - began}


def read_record(output, arm, seed, wrong=False):
    record = read_json((wrong_path if wrong else record_path)(output, arm, seed))
    if record.get("schema") != SCHEMA_RECORD or record.get("plan_identity") != IDENTITY or record["seed"] != seed:
        raise ValueError(f"record {arm}/{seed} binding differs")
    return record


def score_tasks():
    return [(arm, s, False) for arm in ARMS for s in SEEDS] + [(arm, s, True) for arm in WRONG_ARMS for s in SEEDS]


def score(output, workers, worker):
    output = Path(output)
    load_plan(output)
    missing = [s for s in SEEDS if not trained_path(output, s).is_file()]
    if missing:
        raise ValueError(f"hierarchical seeds {missing} are not trained; score after training")
    _, cells = gb.evaluation_set()
    anchor_sets = {"E99": gb.load_anchors(GATE_B)["anchors"], "F77": load_f77_anchors(output)["anchors"]}
    truth_map = load_truth(output)
    for arm, seed, wrong in score_tasks()[worker::workers]:
        path = (wrong_path if wrong else record_path)(output, arm, seed)
        if path.is_file():
            continue
        t0 = time.monotonic()
        with harness.deterministic_scoring() as policy_state:
            record = (wrong_record(output, arm, seed, policy_state) if wrong
                      else score_record(output, cells, anchor_sets, truth_map, arm, seed, policy_state))
        write_json(path, record)
        torch.cuda.empty_cache()
        wall = time.monotonic() - t0
        label = f"wrong--{arm}" if wrong else arm
        mark(output, f"score-{label}-seed{seed}", wall, wall)
        log(f"worker {worker}: scored {label} seed {seed} ({wall:.0f}s)"
            + (f"; typed failure {record['typed_failure']}" if record["typed_failure"] else ""))
    return 0


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def load_records(output):
    return {(arm, s): read_record(output, arm, s) for arm in ARMS for s in SEEDS}


def found_rows(record, inventory, member):
    return None if record["typed_failure"] else record["inventories"][inventory][member]["rows"]


def view_rows(records, cells, inventory, stats):
    rows = []
    for member, entry in sorted(cells[inventory]["mixed"].items()):
        prior = m99.view(entry, gb.prior_rows(entry, stats), PRIMARY_COST)
        for seed in SEEDS:
            views = {arm: v1.cost_view(entry, found_rows(records[(arm, seed)], inventory, member), PRIMARY_COST)
                     for arm in ARMS}
            views["prior"] = prior
            rows.append({"member": member, "seed": seed, "views": views})
    return rows


def member_means(rows):
    return {arm: t96.member_mean([{"member": r["member"], "auc": r["views"][arm]["auc"]} for r in rows], "auc")
            for arm in ARMS}


def cross_fit(means_all, inventory, names):
    """The arm of ``names`` with the best mean member-mean AUC over the other three inventories (t96 F*)."""
    scores = {}
    for arm in names:
        others = [means_all[o][arm] for o in INVENTORIES if o != inventory and means_all[o][arm] is not None]
        scores[arm] = float(np.mean(others)) if others else None
    best = max(names, key=lambda a: (-math.inf if scores[a] is None else scores[a], -names.index(a)))
    return best, scores


def contrast_block(rows, first, second):
    block_ = m99.paired(rows, [(1, first), (-1, second)])
    return {**block_, "first": first, "second": second, "token": c2.token(block_["estimate"])}


def arm_compute(records, inventory, arm):
    results = [r for s in SEEDS if not records[(arm, s)]["typed_failure"]
               for r in records[(arm, s)]["inventories"][inventory].values()]
    candidates = sum(r["candidates"] for r in results)
    return {"mean_macs_per_candidate": sum(r["linear_macs"] for r in results) / candidates if candidates else None,
            "mean_transitions_per_candidate": sum(r["transition_calls"] for r in results) / candidates
            if candidates else None,
            "gpu_seconds": sum(r["gpu_seconds"] for r in results)}


def schedule_diagnostics(records, inventory, arm):
    """Pooled request counts (index into PAIRS, or the 3 continuous pairs for TAWM-adaptive), modal request,
    share of steps on the diagonal, and the share of cells whose candidates' schedules differ."""
    names = t96.PAIR_NAMES if arm != TAWM_ADAPTIVE else tuple(f"C-F-{d}" for d in t96.DELTAS)
    counts = [0] * len(names)
    differing = cells = 0
    for seed in SEEDS:
        for result in records[(arm, seed)]["inventories"][inventory].values():
            texts = list(result["schedules"].values())
            cells += 1
            differing += len(set(texts)) > 1
            for text in texts:
                for index in t96.unrle(text):
                    counts[index] += 1
    total = sum(counts)
    modal = max(range(len(names)), key=lambda k: (counts[k], -k))
    diagonal = sum(counts[k] for k in rs.diagonal_spec(PAIRS)["allowed"]) if arm != TAWM_ADAPTIVE else None
    return {"pair_counts": dict(zip(names, counts)), "modal_pair": names[modal],
            "diagonal_step_share": diagonal / total if diagonal is not None and total else None,
            "non_modal_step_share": 1 - counts[modal] / total if total else None,
            "cells_with_differing_schedules": differing, "cells": cells}


def monotone_violations(records):
    """Steps of the J_monotone / J_dec_monotone schedules that break the monotone rule (must be 0)."""
    table = rs.monotone_table(PAIRS)
    bad = 0
    for arm in ("J_monotone", "J_dec_monotone"):
        for seed in SEEDS:
            for results in records[(arm, seed)]["inventories"].values():
                for result in results.values():
                    for text in result["schedules"].values():
                        sequence = t96.unrle(text)
                        bad += sum(not bool(table[a, b]) for a, b in zip(sequence, sequence[1:]))
    return bad


def restriction_violations(records):
    """Steps of the diagonal / anti-diagonal arms outside their allowed set (must be 0)."""
    bad = 0
    for arm in POLICY_ARMS:
        spec = restriction_spec(arm)
        if spec["kind"] != "restricted":
            continue
        for seed in SEEDS:
            for results in records[(arm, seed)]["inventories"].values():
                for result in results.values():
                    for text in result["schedules"].values():
                        bad += sum(index not in spec["allowed"] for index in t96.unrle(text))
    return bad


PRIMARY_CONTRASTS = ("J-J_diag", "J-J_monotone", "J-H*")


def inventory_tables(records, rows, means_all, inventory):
    f_star, f_scores = cross_fit(means_all, inventory, list(FIXED_ARMS))
    h_star, h_scores = cross_fit(means_all, inventory, list(H_ARMS))
    means = means_all[inventory]
    hindsight = max(FIXED_ARMS, key=lambda a: (-math.inf if means[a] is None else means[a], -FIXED_ARMS.index(a)))
    pairs = {
        "J-J_diag": ("J", "J_diag"), "J-J_monotone": ("J", "J_monotone"), "J-H*": ("J", h_star),
        "J-J_anti": ("J", "J_anti"), "J_diag-J_anti": ("J_diag", "J_anti"), "J_monotone-J_diag": ("J_monotone", "J_diag"),
        "J-F_hindsight": ("J", hindsight), "J-F*": ("J", f_star), "J_diag-F*": ("J_diag", f_star),
        "J-prior": ("J", "prior"), "J-C*": ("J", C_STAR),
        "H*-F*": (h_star, f_star), "H*-J_diag": (h_star, "J_diag"), "H*-prior": (h_star, "prior"),
        **{f"J-{h}": ("J", h) for h in H_ARMS},
        **{f"{h}-F-{h[2:]}": (h, f"F-{h[2:]}") for h in H_ARMS},
        **{f"J-{e}": ("J", e) for e in EXTERNAL_ARMS},
        "J_dec-J_dec_diag": ("J_dec", "J_dec_diag"), "J_dec-J_dec_monotone": ("J_dec", "J_dec_monotone"),
        "J_dec-H*": ("J_dec", h_star), "J_dec-J_dec_anti": ("J_dec", "J_dec_anti"),
        "J_dec_diag-J_dec_anti": ("J_dec_diag", "J_dec_anti"), "J_dec-F_hindsight": ("J_dec", hindsight),
    }
    rankers = {arm: {**v1.ranker_block(rows, arm), **(arm_compute(records, inventory, arm) if arm != "prior" else {})}
               for arm in (*ARMS, "prior")}
    return {"cells": len(rows), "members": len({r["member"] for r in rows}),
            "F_star": f_star, "F_star_scores": f_scores, "F_hindsight": hindsight, "H_star": h_star,
            "H_star_scores": h_scores, "member_mean_auc": means, "rankers": rankers,
            "contrasts": {name: contrast_block(rows, a, b) for name, (a, b) in pairs.items()},
            "schedules": {arm: schedule_diagnostics(records, inventory, arm) for arm in SCHEDULED}}


def rollout_tables(records, cells, h_star):
    """#100 engine_position_mse at t = 225 on every mixed grid candidate: per (level, seed) the mean over
    the level's candidates with a defined value, per level the mean over its available seeds, bootstrap
    over levels; paired contrasts per level (both arms defined). Lower is better."""
    per_level = {arm: {} for arm in ARMS}
    nonfinite = {arm: 0 for arm in ARMS}
    for arm in ARMS:
        for member in sorted(cells[PRIMARY]["mixed"]):
            seeds_ = []
            for seed in SEEDS:
                record = records[(arm, seed)]
                if record["typed_failure"]:
                    seeds_.append(None)
                    continue
                result = record["inventories"][PRIMARY][member]
                values = [v for v in result["engine_mse"] if v is not None]
                nonfinite[arm] += sum(1 for r in result["rows"] if r["excluded"])
                seeds_.append(float(np.mean(values)) if values else None)
            finite = [v for v in seeds_ if v is not None]
            per_level[arm][member] = float(np.mean(finite)) if finite else None
    arms = {arm: {"error": v1.bootstrap([v for v in per_level[arm].values() if v is not None],
                                        [m for m, v in per_level[arm].items() if v is not None]),
                  "nonfinite_endpoints": nonfinite[arm]} for arm in ARMS}

    def paired(first, second):
        common = [m for m in sorted(per_level[first]) if per_level[first][m] is not None
                  and per_level[second][m] is not None]
        return {"first": first, "second": second, "levels": len(common),
                "estimate": v1.bootstrap([per_level[first][m] - per_level[second][m] for m in common], common)}
    contrasts = {"J-J_diag": paired("J", "J_diag"), "J-J_monotone": paired("J", "J_monotone"),
                 "J-H*": paired("J", h_star), "J_diag-H*": paired("J_diag", h_star),
                 **{f"{h}-F-{h[2:]}": paired(h, f"F-{h[2:]}") for h in H_ARMS},
                 **{f"{e}-F-15-continuous": paired(e, "F-15-continuous") for e in EXTERNAL_ARMS}}
    return {"metric": "engine_position_mse (r100) at t = 225, observed frames", "H_star": h_star,
            "arms": arms, "contrasts": contrasts}


def wrong_anchor_tables(output, records, cells):
    out = {}
    for arm in WRONG_ARMS:
        rows = []
        for member, entry in sorted(cells[PRIMARY]["mixed"].items()):
            for seed in SEEDS:
                record = read_record(output, arm, seed, wrong=True)
                right = found_rows(records[(arm, seed)], PRIMARY, member)
                if record["typed_failure"] or right is None:
                    rows.append({"member": member, "seed": seed, "views": {arm: None}})
                    continue
                wrong = [record["wrong_anchor"][a]["rows"] for a in sorted(record["wrong_anchor"])]
                wrong_auc = [a for a in (c2.auc(entry, w) for w in wrong) if a is not None]
                rhos = [r for r in (c2.spearman_rows(right, w) for w in wrong) if r is not None]
                rows.append({"member": member, "seed": seed, "views": {arm: {
                    "right": c2.auc(entry, right), "wrong": float(np.mean(wrong_auc)) if wrong_auc else None,
                    "stability": float(np.mean(rhos)) if rhos else None}}})
        difference = c2.difference_block(rows, [(1, arm, "right"), (-1, arm, "wrong")])
        out[arm] = {"cells": len(rows), "wrong_anchors": len(gb.fit_anchors()),
                    "right": c2.field_block(rows, arm, "right"), "wrong": c2.field_block(rows, arm, "wrong"),
                    "right_minus_wrong": difference, "stability": c2.field_block(rows, arm, "stability"),
                    "token": c2.token(difference)}
    return out


def replication(records):
    """Fixed, C*, J and J_dec rows (and schedules) equal the #101 records on every cell and seed."""
    same, checked = True, 0
    pairs = [(a, a) for a in (*FIXED_ARMS, C_STAR)] + [(a, policy_arms()[a][0]) for a in ("J", "J_dec")]
    for arm, source in pairs:
        for seed in SEEDS:
            reference = p101.read_record(POLICY, source, seed)["inventories"]
            mine = records[(arm, seed)]["inventories"]
            for inventory in INVENTORIES:
                for member, result in reference[inventory].items():
                    same &= mine[inventory][member]["rows"] == result["rows"]
                    if "schedules" in result:
                        same &= mine[inventory][member]["schedules"] == result["schedules"]
                    checked += 1
    return {"blocks": checked, "pass": bool(same)}


def disposition(grid, guards_ok):
    """The pre-declared rule on 'independent axes add value over tied axes' (grid)."""
    tokens = {name: grid["contrasts"][name]["token"] for name in PRIMARY_CONTRASTS}
    if not guards_ok:
        return STOP_TOKEN, tokens
    if all(t == "supported" for t in tokens.values()):
        return "supported", tokens
    if any(t == "not_supported_by_this_experiment" for t in tokens.values()):
        return "not_supported_by_this_experiment", tokens
    return STOP_TOKEN, tokens


def compute_tables(output, plan):
    output = Path(output)
    branches, cells = gb.evaluation_set()
    records = load_records(output)
    fit = read_json(gb.FIT_VERDICTS)["branches"]
    stats = {i: gb.prior_stats(fit, i) for i in INVENTORIES}
    rows = {i: view_rows(records, cells, i, stats[i]) for i in INVENTORIES}
    means = {i: member_means(rows[i]) for i in INVENTORIES}
    tables = {i: inventory_tables(records, rows[i], means, i) for i in INVENTORIES}
    grid = tables[PRIMARY]
    rollout_ = rollout_tables(records, cells, grid["H_star"])
    wrong = wrong_anchor_tables(output, records, cells)
    members = [member_state(output, s) for s in SEEDS]
    payload = rt.load_training_data(RETRAIN)
    eval_roster = {lv["identity"] for lv in gb.evaluation_levels()}
    fit_roster = {lv["identity"] for lv in rt.fit_levels()}
    f77 = load_f77_anchors(output)
    truth_store = torch.load(truth_path(output), map_location="cpu", weights_only=True)
    scored_at = sorted(r["scored_at"] for r in records.values())
    trained_at = sorted(m["trained_at"] for m in members)
    trained = p101.training_table(POLICY)
    policy_hashes = all(
        records[(arm, s)]["policy"]["sha256"]
        == trained[records[(arm, s)]["policy"]["kind"]][p101.tag(records[(arm, s)]["policy"]["lambda"])][str(s)][
            "policy_sha256"] for arm in POLICY_ARMS for s in SEEDS)
    h_hashes = all(records[(arm, s)]["predictor_sha256"] in (None, m["sha256"])
                   for arm in H_ARMS for s, m in zip(SEEDS, members))
    plan_external = {e["name"]: e["sha256"] for e in plan["inputs"] if e["name"].startswith("external_")}
    external_hashes = all(sha256_of(path) == plan_external[name] for name, path in external_files().items())
    primary_nonfinite = {arm: grid["rankers"][arm]["nonfinite_share"] for arm in ("J", "J_diag", "J_monotone",
                                                                               grid["H_star"])}
    guards = {
        "G1_isolation": {"pass": bool(set(payload["shot_members"]) <= fit_roster and not (fit_roster & eval_roster)
                                      and all(set(cells[i]["mixed"]) <= eval_roster for i in INVENTORIES))},
        "G2_scoring_policy": {"pass": bool(all(r["scoring_policy"] == harness.SCORING_POLICY for r in records.values())
                                           and f77["scoring_policy"] == harness.SCORING_POLICY)},
        "G3_chronology": {"plan_frozen_at": plan["frozen_at"], "first_trained_at": trained_at[0],
                          "last_trained_at": trained_at[-1], "f77_anchors_parsed_at": f77["parsed_at"],
                          "truth_built_at": truth_store["built_at"], "first_scored_at": scored_at[0],
                          "pass": bool(plan["frozen_at"] <= trained_at[0] and trained_at[-1] <= scored_at[0]
                                       and plan["frozen_at"] <= f77["parsed_at"] <= scored_at[0]
                                       and plan["frozen_at"] <= truth_store["built_at"] <= scored_at[0])},
        "G4_checkpoint_binding": {"policies": bool(policy_hashes), "hierarchical": bool(h_hashes),
                                  "external": bool(external_hashes),
                                  "pass": bool(policy_hashes and h_hashes and external_hashes)},
        "G5_replication_101": replication(records),
        "G6_restriction_integrity": {"restricted_violations": restriction_violations(records),
                                     "monotone_violations": monotone_violations(records)},
        "G7_nonfinite": {"shares": primary_nonfinite,
                         "pass": all(v is not None and v <= NONFINITE_SHARE_MAX for v in primary_nonfinite.values())},
    }
    g6 = guards["G6_restriction_integrity"]
    g6["pass"] = g6["restricted_violations"] == 0 and g6["monotone_violations"] == 0
    guards_ok = all(g["pass"] for g in guards.values())
    token, tokens = disposition(grid, guards_ok)
    dec_token, dec_tokens = disposition({"contrasts": {
        "J-J_diag": grid["contrasts"]["J_dec-J_dec_diag"], "J-J_monotone": grid["contrasts"]["J_dec-J_dec_monotone"],
        "J-H*": grid["contrasts"]["J_dec-H*"]}}, guards_ok)
    phases = {p.stem: read_json(p) for p in sorted((output / "ledger").glob("*.json"))}
    smoke = read_json(output / "smoke.json")
    rehearsal = plan["rehearsal"]
    all_phases = {"smoke": {"wall_seconds": smoke["wall_seconds"], "gpu_seconds": smoke["wall_seconds"]},
                  "rehearsal": {"wall_seconds": rehearsal["wall_seconds"], "gpu_seconds": rehearsal["gpu_seconds"]},
                  **{k: {"wall_seconds": v["wall_seconds"], "gpu_seconds": v["gpu_seconds"]} for k, v in phases.items()}}
    return {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
        "policies": {arm: {"source": src, "kind": k, "lambda": lam} for arm, (src, k, lam) in policy_arms().items()},
        "evaluation": gb.evaluation_accounting(branches, cells),
        "hierarchical": {"members": members, "parameters": members[0]["parameters"],
                         "hybrid_parameters": plan["hierarchical"]["hybrid_parameters"],
                         "width": hd.WIDTH, "usable_seeds": [m["seed"] for m in members if m["usable"]]},
        "guards": guards,
        "disposition": {"claim": "independent axes add value over tied axes", "token": token,
                        "contrast_tokens": tokens, "rule": plan["decision_rule"]["disposition"],
                        "J_dec_reading": {"token": dec_token, "contrast_tokens": dec_tokens,
                                          "label": "DESCRIPTIVE (same rule on J_dec)"}},
        "inventories": tables, "rollout_error": rollout_, "wrong_anchor": wrong,
        "unsupported": UNSUPPORTED,
        "compute": {"phases": all_phases, "gpu_seconds": sum(b["gpu_seconds"] for b in all_phases.values()),
                    "wall_seconds": sum(b["wall_seconds"] for b in all_phases.values()), "engine_seconds": 0,
                    "hardware": "2 x RTX 5090 (GPUs 2, 3; one process per GPU)"},
    }


# ---------------------------------------------------------------------------
# renderings
# ---------------------------------------------------------------------------

def cell(block_):
    return fmt_interval(block_) if block_ else "n/a"


def num(value, digits=4):
    return "n/a" if value is None else format(value, f".{digits}f")


def gmacs(value):
    return "n/a" if value is None else f"{value / 1e6:.1f}M"


ARM_LABELS = {
    "J": "J (ICLR, #101 pe 1e-4)", "J_diag": "J_diag", "J_monotone": "J_monotone", "J_anti": "J_anti (control)",
    "J_dec": "J_dec (#101 dec 1e-1)", "J_dec_diag": "J_dec_diag", "J_dec_monotone": "J_dec_monotone",
    "J_dec_anti": "J_dec_anti (control)", C_STAR: "C* (LC continuous C-F-15)", "prior": "no-model action prior",
}


def findings_md(plan, compute):
    lines = []
    add = lines.append
    grid = compute["inventories"][PRIMARY]
    d = compute["disposition"]
    add("# Issue #105: independent axes vs hierarchy on the #104 sealed #109 evaluation split")
    add("")
    add(f"Plan frozen {plan['frozen_at']} after a passed smoke, before training and scoring; scored once in processes "
        f"that did not train, under `deterministic_scoring()`. Validation: `{VALIDATION_COMMAND}`. Intervals: "
        "level-clustered percentile bootstrap (10 000 draws), DESCRIPTIVE; margin δ = 0.02; zero engine seconds.")
    add("")
    add(f"**Disposition on \"independent axes add value over tied axes\" (grid): {d['token']}.** "
        + "; ".join(f"{k} {v}" for k, v in d["contrast_tokens"].items()) + f". H* = {grid['H_star']} (cross-fit).")
    add("")
    add("## Acceptance contrasts (AUC difference, paired by level and seed)")
    add("")
    add("| contrast | " + " | ".join(INVENTORIES) + " |")
    add("| --- | " + " | ".join("---" for _ in INVENTORIES) + " |")
    for name in (*PRIMARY_CONTRASTS, "J-J_anti", "J_diag-J_anti", "H*-J_diag", "J-F_hindsight", "J-F*",
                 "J_dec-J_dec_diag", "J_dec-J_dec_monotone", "J_dec-H*", "J_dec-J_dec_anti"):
        cells_ = []
        for inventory in INVENTORIES:
            c = compute["inventories"][inventory]["contrasts"][name]
            cells_.append(f"{cell(c['estimate'])} {c['token']}")
        add(f"| {name} | " + " | ".join(cells_) + " |")
    add("")
    add("H* and F* are cross-fit per inventory (best mean member-mean AUC over the other three inventories); "
        "F_hindsight is the in-inventory best fixed request. Per inventory: "
        + "; ".join(f"{i}: H* {compute['inventories'][i]['H_star']}, F* {compute['inventories'][i]['F_star']}, "
                    f"F_hindsight {compute['inventories'][i]['F_hindsight']}" for i in INVENTORIES) + ".")
    add("")
    add("## Ranking AUC per arm (tie-free cost)")
    add("")
    add("| arm | carrier | " + " | ".join(INVENTORIES) + " | MACs / candidate (grid) | tied pair share (grid) |")
    add("| --- | --- | " + " | ".join("---" for _ in INVENTORIES) + " | --- | --- |")
    for arm in (*ARMS, "prior"):
        carrier = "F77 (#71)" if arm in EXTERNAL_ARMS else ("—" if arm == "prior" else "E99")
        r = grid["rankers"][arm]
        add(f"| {ARM_LABELS.get(arm, arm)} | {carrier} | "
            + " | ".join(cell(compute["inventories"][i]["rankers"][arm]["auc"]) for i in INVENTORIES)
            + f" | {gmacs(r.get('mean_macs_per_candidate'))} | {num(r['tied_pair_share'])} |")
    add("")
    add("| arm (#78) | reading |")
    add("| --- | --- |")
    for name, reason in compute["unsupported"].items():
        add(f"| {name} | unsupported: {reason} |")
    add("")
    add("## Request schedules (grid, pooled over seeds)")
    add("")
    add("| arm | modal request | diagonal step share | non-modal step share | cells with differing schedules |")
    add("| --- | --- | --- | --- | --- |")
    for arm, s in grid["schedules"].items():
        add(f"| {arm} | {s['modal_pair']} | {num(s['diagonal_step_share'], 3)} | {num(s['non_modal_step_share'], 3)} | "
            f"{s['cells_with_differing_schedules']} / {s['cells']} |")
    add("")
    ro = compute["rollout_error"]
    add(f"## Rollout error at t = {ENDPOINT} (grid candidates; {ro['metric']}; lower is better)")
    add("")
    add("| arm | error | nonfinite endpoints |")
    add("| --- | --- | --- |")
    for arm, a in ro["arms"].items():
        add(f"| {ARM_LABELS.get(arm, arm)} | {cell(a['error'])} | {a['nonfinite_endpoints']} |")
    add("")
    add("| contrast (paired by level) | estimate |")
    add("| --- | --- |")
    for name, c in ro["contrasts"].items():
        add(f"| {name} ({c['first']} − {c['second']}) | {cell(c['estimate'])} |")
    add("")
    add("## Controls")
    add("")
    add(f"- No-model action prior (grid): {cell(grid['rankers']['prior']['auc'])}; J − prior "
        f"{cell(grid['contrasts']['J-prior']['estimate'])}; H* − prior {cell(grid['contrasts']['H*-prior']['estimate'])}.")
    for arm, w in compute["wrong_anchor"].items():
        add(f"- Wrong-anchor control of {arm} ({w['wrong_anchors']} fit anchors): right {cell(w['right'])}, wrong "
            f"{cell(w['wrong'])}, right − wrong {cell(w['right_minus_wrong'])} ({w['token']}); stability "
            f"{cell(w['stability'])}.")
    add("")
    h = compute["hierarchical"]
    add("## Hierarchical predictor")
    add("")
    add(f"- Three levels (1, continuous) / (5, micro) / (15, macro), width {h['width']}, {h['parameters']:,} "
        f"parameters vs {h['hybrid_parameters']:,} for the LC hybrid; #112 v3 LC recipe, {STEPS} updates (3000 per "
        "level), #109 fit-split payload of the v3 retrain.")
    for m in h["members"]:
        add(f"- seed {m['seed']}: usable {m['usable']} (complete {m['complete']}, skipped {m['skipped']}, excursion "
            f"{None if m['end_state'] is None else m['end_state']['excursion']}), {m['gpu_seconds']:.0f} GPU s.")
    add("")
    add("## Guards")
    add("")
    for name, guard in compute["guards"].items():
        add(f"- {name}: {guard['pass']}")
    add("")
    c = compute["compute"]
    add(f"Compute: {c['gpu_seconds']:.0f} GPU s, {c['wall_seconds']:.0f} process wall s ({c['hardware']}); 0 engine s.")
    add("")
    return "\n".join(lines)


def summary(plan, compute):
    grid = compute["inventories"][PRIMARY]
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "validation_command": VALIDATION_COMMAND, "disposition": compute["disposition"],
            "acceptance": {i: {name: {"estimate": compute["inventories"][i]["contrasts"][name]["estimate"],
                                      "token": compute["inventories"][i]["contrasts"][name]["token"],
                                      "second": compute["inventories"][i]["contrasts"][name]["second"]}
                               for name in PRIMARY_CONTRASTS} for i in INVENTORIES},
            "grid_auc": {arm: grid["rankers"][arm]["auc"] for arm in (*ARMS, "prior")},
            "H_star": grid["H_star"], "F_hindsight": grid["F_hindsight"], "F_star": grid["F_star"],
            "rollout_error_t225": {arm: a["error"] for arm, a in compute["rollout_error"]["arms"].items()},
            "unsupported": compute["unsupported"],
            "hierarchical": {k: compute["hierarchical"][k] for k in ("parameters", "hybrid_parameters", "width",
                                                                     "usable_seeds")},
            "guards": {k: v["pass"] for k, v in compute["guards"].items()},
            "compute": {k: compute["compute"][k] for k in ("gpu_seconds", "wall_seconds", "engine_seconds")},
            "claim_boundary": plan["claim_boundary"]}


def rendered(plan, compute):
    return {"compute.json": json_text(compute), "summary.json": json_text(summary(plan, compute)),
            "findings.md": findings_md(plan, compute)}


# ---------------------------------------------------------------------------
# smoke (pre-freeze; fit split, #101 / #78 artefacts; no evaluation score)
# ---------------------------------------------------------------------------

def fit_grid_cells():
    return gb.mixed_cells(gb.inventory_entries(rt.fit_levels(), rt.complete_branches(), rt.fit_verdicts(), PRIMARY))


def smoke_hierarchical():
    """Capacity, interface, and a full-cost training smoke (horizon 225 and ranking from update 0)."""
    width, target, counts = hd.matched_width()
    model = hd.HierarchicalPredictor().to(DEVICE)
    z, action = torch.randn(4, 236, device=DEVICE), torch.randn(4, 5, device=DEVICE)
    shapes = all(tuple(model.carrier(z, action, p).shape) == (4, 236) for p in hd.LEVEL_PAIRS)
    rejected = 0
    for pair in PAIRS:
        if pair in hd.LEVEL_OF:
            continue
        try:
            model.carrier(z, action, pair)
        except ValueError:
            rejected += 1
    impl = implementation()
    data = rt.IndexedTrainData(training_data(), "hybrid")
    began = time.monotonic()
    report, ensemble = impl.train_group(OUTPUT, ARM, FAMILY, 0, [("all", SEEDS[0])], data, steps=SMOKE_STEPS,
                                        save=False, smoke_full=True, timing_after=2 * len(hd.LEVEL_PAIRS))
    wall = time.monotonic() - began
    finite = bool(torch.isfinite(torch.tensor(report["trace"][-1]["loss"])).all())
    del ensemble, data
    torch.cuda.empty_cache()
    return {"control": "hierarchical_capacity_interface_training",
            "width": width, "parameters": counts[width], "hybrid_parameters": target,
            "relative_difference": (counts[width] - target) / target, "level_macs": {
                a: hd.linear_macs(model, p) for a, p in zip(H_ARMS, hd.LEVEL_PAIRS)},
            "shapes_ok": shapes, "off_diagonal_rejected": rejected, "smoke_updates": SMOKE_STEPS,
            "smoke_wall_seconds": wall, "timed_seconds_per_update_full_cost": report["timed_seconds_per_update"],
            "peak_memory_gb": report["peak_memory_gb"], "final_loss_finite": finite,
            "ok": bool(width == hd.WIDTH and abs(counts[width] - target) / target <= 0.02 and shapes and rejected == 6
                       and finite)}


def smoke_rollouts():
    """On three fit grid cells: hd.fixed_rollout equals m99.rollout_endpoints for fixed requests of the LC
    hybrid; rs.rollout_endpoints with J equals the #101 rp path; restricted schedules stay in their sets;
    monotone schedules obey the rule."""
    from world_model.training import request_policy as rp
    cells = sorted(fit_grid_cells().items())[:SMOKE_CELLS]
    anchors_ = gb.fit_anchors()
    model = p101.load_predictor(SEEDS[0])
    _, kind, lam = policy_arms()["J"]
    policy = p101.load_policy(POLICY, kind, lam, SEEDS[0])
    table = rs.monotone_table(PAIRS)
    equal = restricted_ok = monotone_ok = True
    for member, entry in cells:
        actions = m99.action_batch(entry["inventory"])
        z0 = anchors_[member].to(DEVICE)
        for arm in ("F-15-macro", "F-1-continuous", "F-5-micro"):
            spec = t96.arm_spec(arm)
            a, ta = m99.rollout_endpoints(model, None, z0, actions, spec)
            b, tb = hd.fixed_rollout(model, z0, actions, PAIRS[spec["pair_index"]],
                                     linear_macs(model, PAIRS[spec["pair_index"]]))
            equal &= bool(torch.equal(a, b)) and ta["linear_macs"] == tb["linear_macs"]
        a, ta = rp.rollout_endpoints(model, policy, z0, actions, {"kind": "joint"}, select=t96.select_pairs,
                                     record_schedules=True)
        b, tb = rs.rollout_endpoints(model, policy, z0, actions, {"kind": "joint"}, select=t96.select_pairs,
                                     record_schedules=True)
        equal &= bool(torch.equal(a, b)) and ta["schedules"] == tb["schedules"]
        for arm in ("J_diag", "J_anti"):
            spec = restriction_spec(arm)
            _, t = rs.rollout_endpoints(model, policy, z0, actions, spec, select=t96.select_pairs, record_schedules=True)
            restricted_ok &= all(i in spec["allowed"] for s in t["schedules"] for i in s)
        _, t = rs.rollout_endpoints(model, policy, z0, actions, restriction_spec("J_monotone"), select=t96.select_pairs,
                                    record_schedules=True)
        monotone_ok &= all(bool(table[x, y]) for s in t["schedules"] for x, y in zip(s, s[1:]))
    return {"control": "rollout_paths_and_restrictions", "cells": len(cells), "fixed_and_joint_equal": bool(equal),
            "restricted_in_set": bool(restricted_ok), "monotone_rule": bool(monotone_ok),
            "ok": bool(equal and restricted_ok and monotone_ok)}


def smoke_truth():
    """engine_truth equals the s99.shot_labels presence / centers rows on three fit grid branches."""
    branches = {b["key"]: b for b in rt.complete_branches()}
    keys = [item["key"] for _, entry in sorted(fit_grid_cells().items())[:1] for item in entry["inventory"][:SMOKE_CELLS]]
    equal = True
    began = time.monotonic()
    for key in keys:
        attempt = branches[key]["attempt"]
        mine = engine_truth(attempt)
        _, _, frames, _ = s99.shot_frames(attempt)
        segment = read_json(Path(attempt) / "shot-1" / "segment.json")
        labels = s99.shot_labels(segment, frames, [mine["position"]], s99.vocabulary())
        equal &= (torch.equal(labels["presence"][0].bool(), mine["presence"])
                  and torch.equal(labels["centers"][0].float(), mine["centers"]))
    return {"control": "engine_truth_equals_shot_labels", "branches": len(keys), "ok": bool(equal),
            "seconds": time.monotonic() - began}


def smoke_external():
    """#78 checkpoints load under their bindings; the F77 parse of fit anchor frames is deterministic; the
    TAWM-adaptive and fixed rollouts run from it."""
    branches = rt.complete_branches()
    first = rt.anchor_branches(branches)
    cells = sorted(fit_grid_cells().items())[:SMOKE_CELLS]
    costs = p101.endpoint_costs()
    began = time.monotonic()
    adapter = wlc.load_adapter(DEVICE)
    parses = {m: f77_anchor(adapter, first[m]["attempt"]) for m, _ in cells}
    again = f77_anchor(adapter, first[cells[0][0]]["attempt"])
    parse_seconds = (time.monotonic() - began) / (len(cells) + 1)
    deterministic = bool(torch.equal(parses[cells[0][0]], again))
    finite = True
    for seed in SEEDS:
        for name in ("tawm", "vlwm"):
            model = load_external(seed, name)
            z, _ = rs.rollout_endpoints(model, None, parses[cells[0][0]].to(DEVICE),
                                        m99.action_batch(cells[0][1]["inventory"]), t96.arm_spec("C-F-5"),
                                        select=t96.select_pairs)
            finite &= bool(torch.isfinite(z).all())
        controller = load_external_controller(seed)
        z, t = rs.rollout_endpoints(load_external(seed, "tawm"), controller, parses[cells[0][0]].to(DEVICE),
                                    m99.action_batch(cells[0][1]["inventory"]), {"kind": "joint"},
                                    select=t96.select_pairs, record_schedules=True)
        finite &= bool(torch.isfinite(z).all())
        rows = [costs.row(i["ordinal"], z[k].cpu(), parses[cells[0][0]]) for k, i in enumerate(cells[0][1]["inventory"])]
        finite &= all(r[PRIMARY_COST] is not None for r in rows)
    return {"control": "external_78_load_parse_rollout", "f77_parse_deterministic": deterministic,
            "f77_parse_seconds": parse_seconds, "rollouts_finite": bool(finite), "ok": bool(deterministic and finite)}


def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the smoke is a pre-freeze control")
    output.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    checks = [smoke_hierarchical()]
    with harness.deterministic_scoring():
        checks.append(smoke_rollouts())
        checks.append(smoke_external())
    checks.append(smoke_truth())
    gate = read_json(GATE_B / "summary.json")
    selection = p101.load_selection(POLICY)
    checks.append({"control": "inputs_bound", "gate_b": gate["gate_b"],
                   "J": p101.J_ICLR, "J_dec": selection["kinds"]["dec"]["arm"],
                   "ok": bool(gate["gate_b"] == "pass" and selection["kinds"]["dec"]["arm"] == "J-dec-1e-01")})
    ok = all(c["ok"] for c in checks)
    wall = time.monotonic() - began
    evidence = {"schema": SCHEMA_SMOKE, "identity": IDENTITY, "run_at": utc_now(), "checks": checks,
                "projection": projection(checks), "wall_seconds": wall,
                "reading": ("fit-split cells, anchors and branches, the #101 policies and the #78 checkpoints; no "
                            "evaluation-split model output"), "ok": ok}
    write_json(output / "smoke.json", evidence)
    log(f"smoke {'PASSED' if ok else 'FAILED'} ({wall:.0f}s)")
    for check in checks:
        log(f"  {check['control']}: {check['ok']}")
    if not ok:
        raise ValueError("smoke control differs; STOP RULE: no freeze")
    return 0


def projection(checks):
    h = checks[0]
    training = STEPS * h["timed_seconds_per_update_full_cost"] * len(SEEDS)
    scoring = len(ARMS) * len(SEEDS) * 60.0 + len(WRONG_ARMS) * len(SEEDS) * 60.0
    total = training + scoring
    return {"training_gpu_seconds_upper": training, "scoring_gpu_seconds_estimate": scoring,
            "total_gpu_seconds": total, "elapsed_hours_on_2_gpus": total / 2 / 3600,
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
        **{f"gate_b_record_{n}_{s}": gb.record_path(GATE_B, n, s) for n in gb.UNITS for s in SEEDS},
        "issue_101_plan": POLICY / "plan.json", "issue_101_selection": POLICY / "selection.json",
        "issue_101_compute": POLICY / "compute.json",
        **{f"issue_101_policy_{k}_{p101.tag(lam)}_{s}": p101.policy_path(POLICY, k, lam, s)
           for k, lam in (("pe", p101.ICLR_LAMBDA), ("dec", 0.1)) for s in SEEDS},
        "issue_78_plan": EXTERNAL / "plan.json", **external_files(),
        "gate_b_runner": ROOT / gb.RUNNER, "issue_112_retrain_runner": ROOT / rt.RUNNER,
        "issue_112_v1_runner": ROOT / v1.RUNNER, "issue_112_versions_runner": ROOT / versions.RUNNER,
        "issue_101_runner": ROOT / p101.RUNNER, "issue_78_runner": ROOT / "scripts/run_external_temporal_baselines.py",
        "issue_99_attribution_runner": ROOT / "scripts/run_decision_chain_attribution.py",
        "issue_96_runner": ROOT / "scripts/run_tau_ad_within_checkpoint.py",
        "issue_100_runner": ROOT / "scripts/run_readout_fidelity_jepa.py",
        "issue_99_encoder": s99.OUTPUT / "encoder" / "encoder.pt",
        "issue_109_cohort_plan": rt.COHORT, "issue_104_plan": rt.POWER_PLAN,
        "issue_104_campaign_branches": rt.CAMPAIGN / "branches.json",
        "scoring_harness": ROOT / "world_model/planning/scoring_harness.py",
        "hierarchical_module": ROOT / MODULE, "request_selectors_module": ROOT / SELECTORS,
        "request_policy_module": ROOT / p101.MODULE, "matched_dynamics": ROOT / "world_model/training/matched_dynamics.py",
        "decision_dynamics": ROOT / "world_model/training/decision_dynamics.py",
        "stable_decision_dynamics": ROOT / "world_model/training/stable_decision_dynamics.py",
        "cnn_hybrid": ROOT / "world_model/training/cnn_hybrid.py",
        "smoke": Path(output) / "smoke.json",
    }
    names.update({f"checkpoint_{n}_{c['seed']}": gb.member_path(n, c["seed"]) for n in gb.UNITS
                  for c in gb.checkpoints(n)})
    return [{"name": name, "artifact": str(Path(path).relative_to(ROOT)), "sha256": sha256_of(path)}
            for name, path in names.items()]


def frozen_rules():
    return {
        "disposition": (
            f"on grid, for 'independent axes add value over tied axes': supported iff J - J_diag, J - J_monotone and "
            f"J - H* are each supported (mean >= {DELTA} with lower bound > 0); not_supported_by_this_experiment iff "
            f"at least one of the three has upper bound < {DELTA}; otherwise {STOP_TOKEN}; any failed guard -> "
            f"{STOP_TOKEN}. The same rule on J_dec is reported DESCRIPTIVE"),
        "H_star": ("the hierarchical level request (H-1-continuous, H-5-micro, H-15-macro) with the best mean member-mean "
                   "AUC over the other three inventories (the t96 F* cross-fit rule; ties to the lower level)"),
        "contrast_tokens": (f"supported iff mean >= {DELTA} with lower bound > 0; not_supported_by_this_experiment iff "
                            f"upper < {DELTA}; otherwise {STOP_TOKEN}; offset, angle, power DESCRIPTIVE"),
        "rollout_error": ("DESCRIPTIVE: #100 engine_position_mse at t = 225 on every mixed grid candidate, branch mean "
                          "per (level, seed), seed mean per level, bootstrap over levels; paired per level"),
    }


def frozen_plan(output, frozen_at, smoke_evidence):
    branches, cells = gb.evaluation_set()
    h = smoke_evidence["checks"][0]
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "issue": 105,
        "related_issues": [78, 96, 100, 101, 104, 108, 109, 111, 112, 114],
        "role": ("hierarchy-constrained request arms, a trained hierarchical predictor and the #78 external baselines "
                 "against the independent-axes request policy J; scored once on the sealed #109 evaluation split "
                 "(confirmatory: Gate B passed)"),
        "frozen_at": frozen_at, "frozen_before_training_and_scoring": True,
        "validation_command": VALIDATION_COMMAND, "runner": RUNNER, "runner_sha256_at_freeze": sha256_of(ROOT / RUNNER),
        "modules": {"hierarchical": {"path": MODULE, "sha256_at_freeze": sha256_of(ROOT / MODULE)},
                    "request_selectors": {"path": SELECTORS, "sha256_at_freeze": sha256_of(ROOT / SELECTORS),
                                          "owner": "#114"}},
        "engine_seconds": 0,
        "question": ("do horizon and description add ranking value as independent request coordinates (J) over "
                     "request sets and models that tie each time scale to one abstraction level?"),
        "arms": {
            "fixed": list(FIXED_ARMS), "c_star": C_STAR,
            "policies": {"J": f"#101 {p101.J_ICLR} (ICLR prediction-error DP, lambda 1e-4), Gate-B LC hybrid predictor",
                         "J_dec": "#101 J-dec-1e-01 (decision-valued DP, the #101 selection)"},
            "restrictions": {
                "": "unrestricted joint argmax over the 9 requests",
                "_diag": "argmax restricted to {(1,cont),(5,micro),(15,macro)} (rs.diagonal_spec)",
                "_monotone": ("path-dependent (rs.monotone_spec): request (D, a) allowed after the row's previous request "
                              "(Dp, ap) iff sign(D - Dp) * sign(a - ap) >= 0 with alpha order cont < micro < macro "
                              "(alpha may only coarsen as Delta grows); the first request is free"),
                "_anti": "control: argmax restricted to {(1,macro),(5,micro),(15,cont)} (rs.anti_diagonal_spec)"},
            "hierarchical": list(H_ARMS),
            "external": {"fixed": list(EXTERNAL_FIXED), "adaptive": TAWM_ADAPTIVE,
                         "carrier": ("the #71 repaired-adapter (F77) batch-1 parse of the same decision frame as the "
                                     "E99 anchor; the #78 checkpoints (trained on #77 N1 F77 carriers) as published"),
                         "unsupported": UNSUPPORTED},
            "controls": {"prior": "no-model action prior (Gate-B rule, fit-verdicts.json)",
                         "wrong_anchor": f"{list(WRONG_ARMS)} from the 480 fit-split anchors (Gate-B rule, grid)",
                         "anti_diagonal": "J_anti / J_dec_anti"},
        },
        "hierarchical": {
            "module": MODULE, "levels": [str(p) for p in hd.LEVEL_PAIRS], "width": hd.WIDTH,
            "parameters": h["parameters"], "hybrid_parameters": h["hybrid_parameters"],
            "relative_difference": h["relative_difference"],
            "architecture": ("three residual levels (Linear(241, w), 3 FiLM blocks, LayerNorm, Linear(w, 236)); level 1 "
                             "adds the micro head + adapter, level 2 the macro head + adapter; top-down coupling: level "
                             "k's FiLM code = Linear(236, 128) of the displacement level k+1 predicts from the same state; "
                             "the top level uses a learned constant code"),
            "recipe": ("#112 v3 LC exactly as the Gate-B retrain (rt.implementation): #77 local loss + long-horizon "
                       "supervision to 225 + decision-contrastive ranking loss, truncation 45, skip-on-nonfinite "
                       "(patience 10), AdamW lr 1e-4 wd 1e-4, clip 1, batch 64, 9000 updates; update k trains level "
                       "k % 3 (3000 updates per level; the hybrid trains 1000 per request); init torch.manual_seed(seed)"),
            "data": "the v3 retrain payload (#109 fit split: 13 919 shots, 680 ranking cells), hash-checked",
            "seeds": list(SEEDS),
            "usable": f"complete, at most {SKIP_CAP} skipped updates, no #117 end-of-training excursion (rt.excursion)",
            "decision": ("the matched baseline is the Gate-B predictor (#112 v3 LC hybrid), replacing the ticket's #77 "
                         "hybrid: it is the predictor every confirmatory reading of #108 uses"),
        },
        "evaluation": {"split": "the #109 evaluation split as in Gate B (stream issue109-evaluation, mixed cells)",
                       "accounting": gb.evaluation_accounting(branches, cells),
                       "anchors": "the Gate-B E99 anchor store; F77 parse of the same frames for the #78 ports",
                       "primary_inventory": PRIMARY, "descriptive": list(gb.DESCRIPTIVE_INVENTORIES),
                       "cost": "tie_free (EndpointCosts.row against the anchor)",
                       "cell_auc": "m99.view; cells (level, seed); bootstrap over levels (10000 draws, PCG64 7201)",
                       "rollout_error": ("engine truth: presence and projected centres (#99 targets) of each mixed grid "
                                         "candidate's own branch at observed position min(225, last frame), "
                                         "built after the freeze")},
        "decision_rule": frozen_rules(),
        "guards": {"G1_isolation": "training payload members are fit levels; scored cells are evaluation levels",
                   "G2_scoring_policy": "every record and the F77 anchor store carry the #111 SCORING_POLICY",
                   "G3_chronology": "frozen_at <= training <= first score; F77 anchors and truth after the freeze",
                   "G4_checkpoint_binding": "#101 policy, hierarchical and #78 checkpoint hashes",
                   "G5_replication_101": "fixed, C*, J and J_dec rows (and schedules) equal the #101 records exactly",
                   "G6_restriction_integrity": "restricted schedules stay in their sets; monotone schedules obey the rule",
                   "G7_nonfinite": f"J, J_diag, J_monotone and H* grid nonfinite share <= {NONFINITE_SHARE_MAX}",
                   "failure": f"a failed guard turns the disposition into {STOP_TOKEN}"},
        "prohibited": ["reading evaluation captures for training or selection", "re-scoring after publication",
                       "changing an arm, rule or recipe after the freeze", "retrying or replacing a typed failure"],
        "caps": {"gpu_seconds": GPU_CAP_SECONDS, "wall_seconds_summed_over_processes": WALL_CAP_SECONDS,
                 "projection": smoke_evidence["projection"],
                 "scope_rule": ("if the projected elapsed wall on 2 GPUs exceeds 24 h, cut to the acceptance minimum "
                                "(J arms, the hierarchical model, controls) and record why on #105")},
        "disclosure": ("Known before the freeze: Gate B (unit 0.7444, C* 0.7834, prior 0.7203 on grid), every #101 "
                       "result (J grid 0.706, J_dec 0.746, F_hindsight F-15-continuous 0.762, J 79 % (1, macro)), #96 "
                       "per-arm AUCs and every #78/#100/#112 result. No restricted-argmax arm, hierarchical model or "
                       "#78 port output on an evaluation capture existed. Hardware: RTX 5090 instead of the #85 RTX "
                       "3090. A pre-freeze rehearsal ran every post-freeze mode on fit-split cells (see rehearsal)."),
        "rehearsal": REHEARSAL,
        "smoke_evidence": smoke_evidence,
        "inputs": input_bindings(output),
        "claim_boundary": ("one look at the sealed #109 evaluation split; grid ranking primary, other inventories and "
                           "rollout error descriptive; tie-free cost; DESCRIPTIVE intervals; the hierarchical model is one "
                           "matched instance of a tied-axes model, not every hierarchy; no closed-loop claim"),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-105 protocol")
    if not plan.get("frozen_before_training_and_scoring") or not plan["smoke_evidence"].get("ok"):
        raise ValueError("plan.json does not declare a passed-smoke freeze")
    for entry in plan["inputs"]:
        if sha256_of(ROOT / entry["artifact"]) != entry["sha256"]:
            raise ValueError(f"frozen input {entry['name']} changed after the freeze")
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
    for name, text in rendered(plan, compute).items():
        (output / name).write_text(text)
    log(f"published: disposition {compute['disposition']['token']} {compute['disposition']['contrast_tokens']}")
    return 0


def spot_checks(output):
    """Re-score J_diag and the H* level, first seed, on every grid cell in this process (exact rows)."""
    plan = load_plan(output)
    del plan
    _, cells = gb.evaluation_set()
    anchors_ = gb.load_anchors(GATE_B)["anchors"]
    truth_map = load_truth(output)
    costs = p101.endpoint_costs()
    seed = SEEDS[0]
    results = {}
    with harness.deterministic_scoring():
        for arm in ("J_diag", "J_monotone", *H_ARMS):
            record = read_record(output, arm, seed)
            if record["typed_failure"]:
                results[arm] = None
                continue
            model, policy, spec, _, _ = arm_setup(output, arm, seed)
            same = True
            for member, entry in sorted(cells[PRIMARY]["mixed"].items()):
                fresh = block(model, policy, spec, anchors_[member], entry["inventory"], costs, arm in SCHEDULED,
                              truth_map)
                kept = record["inventories"][PRIMARY][member]
                same &= fresh["rows"] == kept["rows"] and fresh["engine_mse"] == kept["engine_mse"]
                if arm in SCHEDULED:
                    same &= fresh["schedules"] == kept["schedules"]
            results[arm] = bool(same)
    return {"seed": seed, "arms": results, "pass": all(v is not False for v in results.values())}


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    compute = compute_tables(output, plan)
    for name, text in rendered(plan, compute).items():
        if (output / name).read_bytes() != text.encode():
            raise ValueError(f"published {name} differs from the recomputation")
    spot = spot_checks(output)
    if not spot["pass"]:
        raise ValueError(f"deterministic re-scoring does not replicate the retained records: {spot}")
    log(f"validation passed ({time.monotonic() - began:.0f}s): tables re-derived and byte-compared; re-scored "
        f"{spot['arms']} seed {spot['seed']}; disposition {compute['disposition']['token']}")
    return 0


def dry_run(output):
    _, cells = gb.evaluation_set()
    log(f"output {Path(output)}; arms {len(ARMS)} x seeds {len(SEEDS)} + wrong-anchor {len(WRONG_ARMS)} x {len(SEEDS)}")
    mixed = {i: len(cells[i]["mixed"]) for i in INVENTORIES}
    log(f"policy arms {policy_arms()}; mixed cells {mixed}")
    log(f"grid rollout branches {len(grid_branches(cells))}; hierarchical width {hd.WIDTH}")
    return 0


MODES = {"dry-run": dry_run, "smoke": smoke, "prepare": prepare, "train": train, "anchors": anchors, "truth": truth,
         "score": score, "publish": publish, "validate": validate}
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
