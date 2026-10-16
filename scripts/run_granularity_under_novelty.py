"""Issue-107: granularity under novelty -- does request selection transfer across NovPhy novelty levels?

Binding runner module: scripts/run_granularity_under_novelty.py
Exact validation command: python -u -m scripts.run_granularity_under_novelty --validate

Inputs are the #110 contract-v2 results (handoff_107.json: supported cells, flags, v2 encoders, scene
heads, zero-shot and few-shot LC hybrid predictors, the v2 anchor rule, the issue110-evaluation grid
inventories with engine verdicts) and the #101 request-policy recipe (world_model/training/
request_policy.py, executed as a private copy at the 370-value v2 carrier; the module bytes are unchanged).

1. Normal request policies. ICLR J (prediction-error DP, lambda 1e-4) and decision-aware J_dec
   (decision-valued DP, lambda 1e-1, the #101 selection) are trained with the #101 recipe on the #109
   policy split (normal N1 data, stream issue109-policy) against the zero-shot v2 world (encoder, scene
   head, LC hybrid predictor of the same seed) and frozen (``--data``, ``--assemble``, ``--train-normal``).
2. Few-shot adaptation on the level-k novel fit sides (``--train-fewshot``):
   - policy only: the normal policy retuned with the #101 recipe, zero-shot world frozen ("policy");
   - predictor only: the #110 few-shot-k world (encoder, scene head, predictor), normal policy frozen
     ("predictor");
   - both: the normal policy retuned with the #101 recipe against the few-shot-k world ("both").
3. One score (``--score``, processes that did not train, deterministic_scoring): the nine fixed requests,
   J and J_dec of every adaptation arm, from the v2 anchor of every evaluation level of every in-scope
   cell (issue110-evaluation, grid), plus the N1 reference on the #109 evaluation split (Gate-B grid levels).
4. ``--publish``: ranking AUC per arm and novelty type (zero-shot and few-shot never pooled), request
   shares and their shift vs the paired normal sides and N1, wrong-anchor and no-model prior controls,
   the pre-declared disposition, compute.json / summary.json / findings.md / records-manifest.json.
5. ``--share-supplement`` (post-freeze, phase-A review F2; DESCRIPTIVE, outcome-free): the frozen policies'
   request rollouts on the supported cells the frozen scope excluded, for request shares on every supported
   cell (supplement/, own ledger and manifest); no verdict, AUC or token. Runs after the publication.

Modes: --dry-run --smoke --prepare --data --assemble --train-normal --train-fewshot --score
--share-supplement --publish --validate. --data, --train-normal, --train-fewshot and --score take --workers N
--worker K (one process per GPU; CUDA_VISIBLE_DEVICES set by the caller). Zero engine seconds. Intervals:
member (evaluation level)-clustered percentile bootstrap (10000 draws, PCG64 7201), DESCRIPTIVE.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import functools
import importlib.util
import json
import math
from pathlib import Path
import time
import types

import numpy as np
import torch

from scripts import post_freeze_amendments as amendments
from scripts import run_novelty_coverage as nc

ROOT = nc.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-107-granularity-under-novelty-v1"
NC = nc.OUTPUT
HANDOFF = NC / "handoff_107.json"
DAP = ROOT / ".local-artifacts/issue-101-decision-aware-policy-v1"
GATE_B = ROOT / ".local-artifacts/issue-108-gate-b-v1"
IDENTITY = "issue-107-granularity-under-novelty-v1"
SCHEMA_PLAN = "issue_107_plan_v1"
SCHEMA_SMOKE = "issue_107_smoke_v1"
SCHEMA_SHOT = "issue_107_prefix_shot_v1"
SCHEMA_POOL = "issue_107_policy_pool_v1"
SCHEMA_EVAL = "issue_107_evaluation_anchors_v1"
SCHEMA_POLICY = "issue_107_request_policy_v1"
SCHEMA_REPORT_TRAIN = "issue_107_training_report_v1"
SCHEMA_RECORD = "issue_107_record_v1"
SCHEMA_COMPUTE = "issue_107_compute_v1"
SCHEMA_SUMMARY = "issue_107_summary_v1"
SCHEMA_MANIFEST = "issue_107_record_manifest_v1"
SCHEMA_SUPPLEMENT_EVAL = "issue_107_share_supplement_anchors_v1"
SCHEMA_SUPPLEMENT_RECORD = "issue_107_share_supplement_record_v1"
SCHEMA_SUPPLEMENT_MANIFEST = "issue_107_share_supplement_manifest_v1"
SUPPLEMENT = "supplement"                    # post-freeze request-share supplement (phase-A review F2)
SUPPLEMENT_MANIFEST = "supplement/manifest.json"
RUNNER = "scripts/run_granularity_under_novelty.py"
MODULE = "world_model/training/request_policy.py"
VALIDATION_COMMAND = "python -u -m scripts.run_granularity_under_novelty --validate"

DEVICE = "cuda"
SEEDS = nc.SEEDS
ZERO = nc.ZERO
KINDS = ("pe", "dec")
LAMBDA = {"pe": 1e-4, "dec": 1e-1}           # ICLR lambda; the #101 dec selection
ARM_OF_KIND = {"pe": "J", "dec": "J_dec"}
VARIANTS = ("N", "P", "B")                    # normal, policy-only retune, both retuned
VARIANT_NAMES = {"N": "normal (trained on N1, frozen)", "P": "policy-only few-shot retune (zero-shot world)",
                 "B": "policy retune on the few-shot world (both)"}
WINDOW_LENGTH = 60
PREFIX_FRAMES = 64                            # stored positions 0..min(end, 63) of the #110 whole-shot parse
PAIR_NAMES = tuple(f"{d}-{a}" for d in (1, 5, 15) for a in ("continuous", "micro", "macro"))
FIXED_ARMS = tuple(f"F-{name}" for name in PAIR_NAMES)
UNIT_REQUEST = "F-15-macro"                   # the Gate-B unit request (pre-declared fixed comparator)
BASE_CONTINUATION = "15-macro"
PRIMARY_COST = "tie_free"
DELTA = 0.02
STOP_TOKEN = "readiness_or_precision_insufficient"
GPU_CAP_SECONDS = 4 * 48 * 3600.0
WALL_CAP_SECONDS = 4 * 48 * 3600.0
NOVELTY_TYPES = {0: "normal (paired normal sides)", 1: "appearance (pink pig/blocks)", 2: "fan (bird push)",
                 3: "stronger air turbulence", 4: "magnet", 5: "right slingshot (mirrored layout)",
                 6: "inverse gravity", 7: "inverse air turbulence", 8: "storm"}
SMOKE_FS_LEVEL = 7


def log(message):
    print(f"[issue-107] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


read_json = nc.read_json
json_text = nc.json_text
write_json = nc.write_json
atomic_torch = nc.atomic_torch
sha256_of = nc.sha256_of


@functools.lru_cache(maxsize=None)
def heavy():
    """GPU/publication modules, imported on first use (worker entry points stay light)."""
    from scripts import run_decision_aware_policy as dap
    from scripts import run_decision_chain_attribution as m99
    from scripts import run_decision_dynamics_retrain as rt
    from scripts import run_decision_relevant_dynamics as v1
    from scripts import run_gate_b as gb
    from scripts import run_lookahead_control as wlc
    from scripts import run_tau_ad_within_checkpoint as t96
    from world_model.planning import scoring_harness as harness
    from world_model.training import novelty_v2 as nv
    from world_model.training import request_policy as rp
    from world_model.training.cnn_hybrid import PAIRS
    return types.SimpleNamespace(dap=dap, m99=m99, rt=rt, v1=v1, gb=gb, wlc=wlc, t96=t96, harness=harness, nv=nv,
                                 rp=rp, PAIRS=PAIRS)


@functools.lru_cache(maxsize=None)
def policy_module():
    """``request_policy.py`` executed as a private module whose carrier width is the v2 370 values
    (``DIM`` is read at call time; every other line is the #101 module, bytes unchanged)."""
    module = private_module("issue107_request_policy_v2")
    module.DIM = heavy().nv.DIM
    return module


def private_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tag(lam):
    return f"{lam:.0e}"


# ---------------------------------------------------------------------------
# scope (the #110 handoff)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def handoff():
    return read_json(HANDOFF)


def cell_scope(cell):
    """In scope iff the cell has mixed-verdict evaluation levels and is not rollout-error-only."""
    entry = handoff()["cells"][cell]
    if any(flag.startswith("rollout_error_only") for flag in entry["flags"]):
        return False, "rollout_error_only: degenerate single-shot pig-removal verdicts (multi-hit pigs)"
    if entry["headroom"]["mixed_levels"] == 0:
        return False, "no_mixed_verdict_evaluation_level: no decision headroom on the evaluation grid"
    return True, None


def in_scope_cells():
    return [cell for cell in handoff()["supported_cells"] if cell_scope(cell)[0]]


def supplement_cells():
    """Supported cells outside the frozen scope: no AUC (no mixed level, or rollout-error only), but the
    frozen policies' request shares are outcome-free (post-freeze DESCRIPTIVE supplement, phase-A F2)."""
    return [cell for cell in handoff()["supported_cells"] if not cell_scope(cell)[0]]


def supplement_groups():
    return tuple(f"L{k}" for k in sorted({cell_level(c) for c in supplement_cells()}))


def scope_table():
    out = {}
    for cell in handoff()["supported_cells"]:
        ok, reason = cell_scope(cell)
        out[cell] = {"in_scope": ok, "token": None if ok else STOP_TOKEN, "reason": reason,
                     "mixed_levels": handoff()["cells"][cell]["headroom"]["mixed_levels"],
                     "evaluation_levels": handoff()["cells"][cell]["headroom"]["evaluation_levels"]}
    return out


def cell_level(cell):
    return int(cell.split("/")[0])


@functools.lru_cache(maxsize=None)
def novel_levels():
    return tuple(sorted({cell_level(c) for c in in_scope_cells() if cell_level(c) > 0}))


def groups():
    """Evaluation groups: n1 (#109 evaluation split), L0 (paired normal sides), Lk (novel sides)."""
    return ("n1", "L0", *(f"L{k}" for k in novel_levels()))


def group_level(group):
    return None if group == "n1" else int(group[1:])


def worlds_of(group):
    return ("zs", "fs") if group_level(group) else ("zs",)


def condition_of(world, k):
    return ZERO if world == "zs" else f"few-shot-{k}"


def names_of(group, world):
    """Scored arms of a group x world: nine fixed requests, the normal J / J_dec and, on novel groups,
    the retuned policies of the world (P on the zero-shot world, B on the few-shot world)."""
    out = [*FIXED_ARMS, "N-pe", "N-dec"]
    if group_level(group):
        out += ["P-pe", "P-dec"] if world == "zs" else ["B-pe", "B-dec"]
    return out


# ---------------------------------------------------------------------------
# membership (outcome-free order)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def usable():
    return tuple(nc.usable_branches(NC))


@functools.lru_cache(maxsize=None)
def eval_levels(group):
    """level -> meta and grid entry (items in grid-ordinal order, verdicts), in #110 membership order;
    issue110-evaluation branches of in-scope cells only."""
    if group == "n1":
        return n1_levels()
    return levels_of(group, frozenset(in_scope_cells()))


@functools.lru_cache(maxsize=None)
def supplement_levels(group):
    """The same, over the supported cells the frozen scope excluded (the request-share supplement)."""
    return levels_of(group, frozenset(supplement_cells()))


def levels_of(group, scope):
    out = {}
    for b in usable():
        if b["stream"] != nc.STREAM_EVAL or b["cell"] not in scope:
            continue
        k = 0 if b["side"] == "normal" else b["novelty_level"]
        if f"L{k}" != group:
            continue
        ordinal = nc.grid_ordinal(b)
        if ordinal is None:
            continue
        level = out.setdefault(b["level"], {"cell": b["cell"], "pair": b["pair"], "ordinal": b["ordinal"],
                                            "side": b["side"], "novelty_level": b["novelty_level"],
                                            "pair_level": b["pair_level"], "items": [], "verdicts": {}})
        level["items"].append({"ordinal": ordinal, "action": b["action"], "key": b["key"], "suffix": b["suffix"]})
        level["verdicts"][ordinal] = bool(b["pig_removed"])
    for level in out.values():
        level["items"].sort(key=lambda item: item["ordinal"])
        level["anchor_key"] = min(level["items"], key=lambda item: item["suffix"])["key"]
    return out


@functools.lru_cache(maxsize=None)
def n1_levels():
    """#109 evaluation split, grid (Gate-B cells): every level with a grid entry; anchor = the Gate-B
    anchor branch (first complete candidate in suffix order)."""
    h = heavy()
    branches, cells = h.gb.evaluation_set()
    first = h.rt.anchor_branches([b for b in branches if b["status"] == "complete"])
    out = {}
    for member, entry in cells[h.gb.GATE_INVENTORY]["all"].items():
        out[member] = {"cell": f"n1/{first[member]['family']}", "pair": None, "ordinal": None, "side": "normal",
                       "novelty_level": 0, "pair_level": None,
                       "items": [{"ordinal": it["ordinal"], "action": it["action"], "key": it["key"]}
                                 for it in entry["inventory"]],
                       "verdicts": dict(entry["verdicts"]), "anchor_key": first[member]["key"],
                       "anchor_attempt": first[member]["attempt"], "anchor_action": first[member]["candidate"]["action"]}
    return out


def mixed(level):
    return set(level["verdicts"].values()) == {True, False}


def entry_of(level):
    return {"inventory": level["items"], "verdicts": level["verdicts"]}


def fewshot_branches(k):
    """Level-k novel fit sides (the #110 few-shot data), usable branches, membership order."""
    return [b for b in usable() if nc.few_training(b, k)]


def policy_branches():
    return heavy().dap.complete_branches()


def shot_tasks():
    """Prefix parses: every complete #109 policy-split branch, then every N1 anchor branch."""
    tasks = [{"key": b["key"], "member": b["member"], "attempt": b["attempt"], "action": b["candidate"]["action"],
              "role": "policy"} for b in policy_branches()]
    for member, level in n1_levels().items():
        tasks.append({"key": level["anchor_key"], "member": member, "attempt": level["anchor_attempt"],
                      "action": level["anchor_action"], "role": "n1-anchor"})
    return tasks


# ---------------------------------------------------------------------------
# paths and ledger
# ---------------------------------------------------------------------------

def shot_path(output, key):
    return Path(output) / "shots" / f"{key}.pt"


def pool_name(variant, k):
    return "normal" if variant == "N" else f"{'zs' if variant == 'P' else 'fs'}-L{k}"


def pool_path(output, name):
    return Path(output) / "data" / f"pool-{name}.pt"


def eval_path(output, group, world):
    return Path(output) / "data" / f"eval-{group}-{world}.pt"


def group_root(output, variant, k, kind, seed):
    scope = "normal" if variant == "N" else f"{variant}-L{k}"
    return Path(output) / "models" / scope / kind / f"seed-{seed}"


def policy_path(output, variant, k, kind, seed):
    return group_root(output, variant, k, kind, seed) / "policy.pt"


def report_path(output, variant, k, kind, seed):
    return group_root(output, variant, k, kind, seed) / "report.json"


def record_path(output, group, world, name, seed):
    return Path(output) / "records" / group / world / f"{name}--seed{seed}.json"


def ledger_path(output, phase):
    return Path(output) / "ledger" / f"{phase}.json"


def mark(output, phase, wall, gpu, **extra):
    write_json(ledger_path(output, phase), {"phase": phase, "wall_seconds": wall, "gpu_seconds": gpu,
                                            "at": utc_now(), **extra})
    totals = ledger_totals(output)
    if totals["gpu_seconds"] > GPU_CAP_SECONDS or totals["wall_seconds"] > WALL_CAP_SECONDS:
        raise ValueError(f"{STOP_TOKEN}: compute cap exceeded ({totals})")


def ledger_phases(output):
    root = Path(output) / "ledger"
    return {p.stem: read_json(p) for p in sorted(root.glob("*.json"))} if root.is_dir() else {}


def ledger_totals(output):
    phases = ledger_phases(output)
    return {"gpu_seconds": sum(b["gpu_seconds"] for b in phases.values()),
            "wall_seconds": sum(b["wall_seconds"] for b in phases.values())}


# ---------------------------------------------------------------------------
# worlds (hash-checked against the #110 handoff), costs, policies
# ---------------------------------------------------------------------------

def handoff_checkpoint(group, name):
    entry = handoff()["checkpoints"][group][name]
    if sha256_of(ROOT / entry["path"]) != entry["sha256"]:
        raise ValueError(f"checkpoint {group}/{name} differs from the #110 handoff")
    return entry


def predictor_key(condition, seed):
    return f"{condition}/hybrid/seed{seed}"


def load_world(condition, seed):
    """The condition's LC hybrid predictor (hash and usability from the handoff)."""
    entry = handoff_checkpoint("predictors", predictor_key(condition, seed))
    if not entry["usable"]:
        raise ValueError(f"predictor {condition}/{seed} is not usable")
    model = nc.load_member(NC, condition, "hybrid", seed)
    if model is None:
        raise ValueError(f"predictor {condition}/{seed} incomplete")
    return model


def cost_row(ordinal, z, z0):
    """The #111 tie-free cost at the v2 pig columns (EndpointCosts.row semantics; a nonfinite
    endpoint is a retained typed failure, every value None)."""
    nv, harness = heavy().nv, heavy().harness
    z = z.detach()
    if not bool(torch.isfinite(z).all()):
        return {"ordinal": ordinal, "excluded": True, PRIMARY_COST: None, "pig_presence": None,
                "pig_displacement": None}
    presence = float(z[nv.PIG_PRESENCE])
    displacement = math.hypot(float(z[nv.PIG_CENTER[0]]) - float(z0[nv.PIG_CENTER[0]]),
                              float(z[nv.PIG_CENTER[1]]) - float(z0[nv.PIG_CENTER[1]]))
    return {"ordinal": ordinal, "excluded": False, PRIMARY_COST: harness.tie_free(presence, displacement),
            "pig_presence": presence, "pig_displacement": displacement}


def tie_free_tensor(z_end, z0):
    nv = heavy().nv
    displacement = torch.hypot(z_end[:, nv.PIG_CENTER[0]] - z0[:, nv.PIG_CENTER[0]],
                               z_end[:, nv.PIG_CENTER[1]] - z0[:, nv.PIG_CENTER[1]])
    return heavy().harness.tie_free(z_end[:, nv.PIG_PRESENCE], displacement)


def save_policy(output, variant, k, kind, seed, policy, extra):
    atomic_torch(policy_path(output, variant, k, kind, seed), {
        "schema": SCHEMA_POLICY, "plan_identity": IDENTITY, "variant": variant, "level": k, "kind": kind,
        "lambda": LAMBDA[kind], "seed": seed, "pairs": [str(p.identity) for p in policy.pairs],
        "width": policy.layers[0].out_features, "remaining_feature": policy.remaining_feature,
        "model": policy.state_dict(), **extra})


def load_policy(output, variant, k, kind, seed):
    payload = torch.load(policy_path(output, variant, k, kind, seed), map_location="cpu", weights_only=True)
    if (payload.get("schema") != SCHEMA_POLICY or payload.get("plan_identity") != IDENTITY
            or (payload["variant"], payload["level"], payload["kind"], payload["seed"]) != (variant, k, kind, seed)):
        raise ValueError(f"policy {variant}/{k}/{kind}/{seed} binding differs")
    R = policy_module()
    policy = R.RequestPolicy(heavy().PAIRS, payload["width"], payload["remaining_feature"])
    if payload["pairs"] != [str(p.identity) for p in policy.pairs]:
        raise ValueError("policy request set differs")
    policy.load_state_dict(payload["model"], strict=True)
    return policy.to(DEVICE).eval()


def load_report(output, variant, k, kind, seed):
    report = read_json(report_path(output, variant, k, kind, seed))
    if report.get("schema") != SCHEMA_REPORT_TRAIN or report.get("plan_identity") != IDENTITY:
        raise ValueError("training report binding differs")
    return report


# ---------------------------------------------------------------------------
# data: prefix parses (#109 policy split, N1 anchors) under the zero-shot encoder and scene head
# ---------------------------------------------------------------------------

def parse_prefix(encoder, scene_head, attempt):
    """The #110 carriers rule (every position 0..end parsed in batches of 64, v2 carrier, scene head
    over the whole shot), kept on positions 0..min(end, 63): bit-identical to the #110 store."""
    _, frames, segment = nc.labels.shot_frames(attempt)
    end = min(len(frames) - 1, nc.labels.MAX_POSITION)
    images = nc.shot_images(attempt, end)
    times = torch.tensor([frames[p]["fixed_time_seconds"] for p in range(end + 1)], dtype=torch.float64)
    values, _ = nc.base_carriers(encoder, images, times)
    values = scene_head.fill(values.to(DEVICE)).cpu()
    return values[:min(end, PREFIX_FRAMES - 1) + 1].clone(), end, nc.labels.action_of(segment)


def zero_world_parsers():
    handoff_checkpoint("encoders", ZERO)
    handoff_checkpoint("scene_heads", ZERO)
    return nc.load_encoder(NC, ZERO)[0], nc.load_scene(str(NC), ZERO)


def data(output, workers, worker):
    output = Path(output)
    load_plan(output)
    began = time.monotonic()
    shard = shot_tasks()[worker::workers]
    pending = [t for t in shard if not shot_path(output, t["key"]).is_file()]
    log(f"worker {worker}/{workers}: {len(pending)} of {len(shard)} prefix parses")
    wlc = heavy().wlc
    with torch.no_grad():
        encoder, scene_head = zero_world_parsers()
        for done, task in enumerate(pending, 1):
            carriers, end, action = parse_prefix(encoder, scene_head, task["attempt"])
            if not torch.equal(action, wlc.action_tensor(task["action"], "cpu")[0]):
                raise ValueError(f"executed action of {task['key']} differs from its inventory action")
            atomic_torch(shot_path(output, task["key"]), {
                "schema": SCHEMA_SHOT, "plan_identity": IDENTITY, "key": task["key"], "member": task["member"],
                "role": task["role"], "end": end, "action": action, "carriers": carriers})
            if done % 200 == 0:
                elapsed = time.monotonic() - began
                log(f"worker {worker}: {done}/{len(pending)}, {elapsed:.0f}s, eta {elapsed / done * (len(pending) - done):.0f}s")
    wall = time.monotonic() - began
    mark(output, f"data-worker{worker}", wall, wall, shots=len(pending))
    return 0


def read_shot(output, key):
    entry = torch.load(shot_path(output, key), map_location="cpu", weights_only=True)
    if entry.get("schema") != SCHEMA_SHOT or entry.get("plan_identity") != IDENTITY or entry["key"] != key:
        raise ValueError(f"shot {key} binding differs")
    return entry


# ---------------------------------------------------------------------------
# assemble: policy pools and evaluation anchors
# ---------------------------------------------------------------------------

def normal_pool(output):
    h = heavy()
    R = policy_module()
    branches = policy_branches()
    first = h.rt.anchor_branches(branches)
    shots = [read_shot(output, b["key"]) for b in branches]
    by_key = {s["key"]: s for s in shots}
    anchors = {member: by_key[b["key"]]["carriers"][0].clone() for member, b in first.items()}
    windows = [R.first_window(s["carriers"], s["action"], s["end"], WINDOW_LENGTH) for s in shots]
    cells = h.dap.policy_cells(h.dap.policy_cell_entries(), anchors)
    return {"levels": [lv["identity"] for lv in h.dap.policy_levels()], "anchors": anchors,
            "anchor_frames": {m: b["key"] for m, b in first.items()},
            "shot_keys": [s["key"] for s in shots], "shot_members": [s["member"] for s in shots],
            "window_z": torch.stack([w["z"] for w in windows]), "window_action": torch.stack([w["action"] for w in windows]),
            "window_length": torch.stack([w["length"] for w in windows]), "cells": cells}


def fewshot_cells(branches, anchors):
    """Per level: the complete grid candidates (grid-ordinal order) with mixed verdicts."""
    h = heavy()
    by_level = defaultdict(list)
    for b in branches:
        by_level[b["level"]].append(b)
    cells = []
    for level, items in by_level.items():
        entries = sorted((nc.grid_ordinal(b), b) for b in items if nc.grid_ordinal(b) is not None)
        verdicts = [bool(b["pig_removed"]) for _, b in entries]
        if set(verdicts) != {True, False}:
            continue
        cells.append({"member": level, "inventory": "grid", "ordinals": [o for o, _ in entries],
                      "z0": anchors[level],
                      "actions": torch.cat([h.wlc.action_tensor(b["action"], "cpu") for _, b in entries]),
                      "verdicts": torch.tensor(verdicts)})
    return cells


def fewshot_pool(world, k):
    R = policy_module()
    condition = condition_of(world, k)
    handoff_checkpoint("encoders", condition)
    handoff_checkpoint("scene_heads", condition)
    branches = fewshot_branches(k)
    wlc = heavy().wlc
    levels, anchors, windows, members, keys = [], {}, [], [], []
    anchor_key = {}
    for b in branches:
        if b["level"] not in anchor_key or b["suffix"] < anchor_key[b["level"]][0]:
            anchor_key[b["level"]] = (b["suffix"], b["key"])
    for b in branches:
        full = nc.full_carriers(NC, condition, b["key"]).cpu()
        action = nc.read_label(NC, b["key"])["action"]
        if not torch.equal(action, wlc.action_tensor(b["action"], "cpu")[0]):
            raise ValueError(f"executed action of {b['key']} differs from its inventory action")
        end = len(full) - 1
        windows.append(R.first_window(full, action, end, WINDOW_LENGTH))
        members.append(b["level"])
        keys.append(b["key"])
        if b["level"] not in levels:
            levels.append(b["level"])
        if anchor_key[b["level"]][1] == b["key"]:
            anchors[b["level"]] = full[0].clone()
    return {"levels": levels, "anchors": anchors, "anchor_frames": {lv: v[1] for lv, v in anchor_key.items()},
            "condition": condition, "shot_keys": keys, "shot_members": members,
            "window_z": torch.stack([w["z"] for w in windows]), "window_action": torch.stack([w["action"] for w in windows]),
            "window_length": torch.stack([w["length"] for w in windows]), "cells": fewshot_cells(branches, anchors)}


def eval_anchors(output, group, world, levels=None):
    levels = eval_levels(group) if levels is None else levels
    if group == "n1":
        anchors = {lv: read_shot(output, meta["anchor_key"])["carriers"][0].clone() for lv, meta in levels.items()}
        condition = ZERO
    else:
        condition = condition_of(world, group_level(group))
        handoff_checkpoint("encoders", condition)
        handoff_checkpoint("scene_heads", condition)
        anchors = {lv: nc.full_carriers(NC, condition, meta["anchor_key"])[0].cpu().clone()
                   for lv, meta in levels.items()}
    return {"levels": list(levels), "anchors": anchors, "condition": condition,
            "anchor_keys": {lv: meta["anchor_key"] for lv, meta in levels.items()}}


def assemble(output):
    output = Path(output)
    load_plan(output)
    began = time.monotonic()
    harness = heavy().harness
    with torch.no_grad(), harness.deterministic_scoring() as scoring_policy:
        jobs = [("normal", lambda: normal_pool(output))]
        for k in novel_levels():
            for world in ("zs", "fs"):
                jobs.append((f"{world}-L{k}", functools.partial(fewshot_pool, world, k)))
        for name, build in jobs:
            path = pool_path(output, name)
            if path.is_file():
                continue
            payload = build()
            atomic_torch(path, {"schema": SCHEMA_POOL, "plan_identity": IDENTITY, "name": name,
                                "scoring_policy": scoring_policy, "assembled_at": utc_now(), **payload})
            log(f"pool {name}: {len(payload['shot_keys'])} shots, {len(payload['levels'])} levels, "
                f"{len(payload['cells'])} mixed cells")
        for group in groups():
            for world in worlds_of(group):
                path = eval_path(output, group, world)
                if path.is_file():
                    continue
                payload = eval_anchors(output, group, world)
                atomic_torch(path, {"schema": SCHEMA_EVAL, "plan_identity": IDENTITY, "group": group, "world": world,
                                    "scoring_policy": scoring_policy, **payload})
                log(f"evaluation anchors {group}/{world}: {len(payload['levels'])} levels")
    wall = time.monotonic() - began
    mark(output, "assemble", wall, wall)
    return 0


@functools.lru_cache(maxsize=None)
def load_pool(output, name):
    payload = torch.load(pool_path(output, name), map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA_POOL or payload.get("plan_identity") != IDENTITY or payload["name"] != name:
        raise ValueError(f"pool {name} binding differs")
    return payload


@functools.lru_cache(maxsize=None)
def load_eval(output, group, world):
    payload = torch.load(eval_path(output, group, world), map_location="cpu", weights_only=True)
    if (payload.get("schema") != SCHEMA_EVAL or payload.get("plan_identity") != IDENTITY
            or (payload["group"], payload["world"]) != (group, world)):
        raise ValueError(f"evaluation anchors {group}/{world} binding differ")
    return payload


# ---------------------------------------------------------------------------
# training (the #101 recipe; one group = variant x level x kind x seed)
# ---------------------------------------------------------------------------

def normal_groups():
    return [("N", None, kind, seed) for kind in KINDS for seed in SEEDS]


def fewshot_groups():
    return [(variant, k, kind, seed) for kind in ("dec", "pe") for variant in ("P", "B")
            for k in novel_levels() for seed in SEEDS]


def label_round(kind, model, policy, pool, round_index):
    h = heavy()
    R = policy_module()
    lam = LAMBDA[kind]
    if kind == "pe":
        return [R.window_rows(model, policy, unit, aggregated=bool(round_index), compute_weight=lam, device=DEVICE)
                for unit in h.dap.window_units(pool)], {}
    units, count = h.dap.cell_units(pool)
    continuation = ({"kind": "fixed", "pair_index": PAIR_NAMES.index(BASE_CONTINUATION)} if round_index == 0
                    else {"kind": "policy"})
    rows = R.decision_rows(model, policy, pool["cells"], continuation=continuation, cost_fn=tie_free_tensor,
                           compute_weight=lam, device=DEVICE)
    stats = {"rows": int(len(rows["labels"])),
             "label_counts": torch.bincount(rows["labels"], minlength=len(PAIR_NAMES)).tolist()}
    return h.dap.split_rows(rows, units, count), stats


def rows_stats(history):
    labels = torch.cat([u["labels"] for u in history])
    return {"rows": int(len(labels)), "label_counts": torch.bincount(labels, minlength=len(PAIR_NAMES)).tolist()}


@torch.no_grad()
def in_sample(model, policy, pool):
    """Greedy-J rollouts on the pool's own mixed cells (training data only): tie-free AUC, MACs per
    candidate rollout, request counts."""
    h = heavy()
    R = policy_module()
    aucs, macs, candidates = [], 0, 0
    counts = [0] * len(PAIR_NAMES)
    for cell in pool["cells"]:
        z0 = cell["z0"]
        z_end, timing = R.rollout_endpoints(model, policy, z0.to(DEVICE), cell["actions"].to(DEVICE), {"kind": "joint"},
                                            select=h.t96.select_pairs, record_schedules=True)
        rows = [cost_row(o, z_end[i].cpu(), z0) for i, o in enumerate(cell["ordinals"])]
        entry = {"verdicts": {o: bool(v) for o, v in zip(cell["ordinals"], cell["verdicts"].tolist())}}
        auc = h.m99.view(entry, rows, PRIMARY_COST)["auc"]
        if auc is not None:
            aucs.append(auc)
        macs += timing["linear_macs"]
        candidates += len(rows)
        for i, c in enumerate(R.schedule_counts(timing["schedules"], len(PAIR_NAMES))):
            counts[i] += c
    return {"auc_mean": float(np.mean(aucs)) if aucs else None, "cells": len(aucs),
            "mean_macs_per_candidate": macs / candidates if candidates else None, "pair_counts": counts}


def train_group(output, variant, k, kind, seed):
    if report_path(output, variant, k, kind, seed).is_file():
        log(f"{variant}/{k}/{kind}/{seed}: done")
        return
    h = heavy()
    R = policy_module()
    root = group_root(output, variant, k, kind, seed)
    began = time.monotonic()
    pool = load_pool(output, pool_name(variant, k))
    condition = ZERO if variant in ("N", "P") else f"few-shot-{k}"
    model = load_world(condition, seed)
    init = None
    stats, failure = {}, None
    with h.harness.deterministic_scoring() as scoring_policy:
        policy = R.new_policy(h.PAIRS, seed, device=DEVICE)
        if variant != "N":
            source = report_path(output, "N", None, kind, seed)
            normal = load_report(output, "N", None, kind, seed)
            if normal.get("typed_failure"):
                failure = f"normal policy {kind}/{seed} is a typed failure"
            else:
                policy.load_state_dict(load_policy(output, "N", None, kind, seed).state_dict())
                init = {"report": str(source.relative_to(ROOT)), "policy_sha256": normal["policy_sha256"]}
        history = []
        try:
            for round_index in range(R.DISTIL_RECIPE["rounds"]) if failure is None else ():
                checkpoint = root / f"round-{round_index}.pt"
                labels_file = root / f"labels-{round_index}.pt"
                if labels_file.is_file():
                    history.append(torch.load(labels_file, map_location="cpu", weights_only=True)["units"])
                else:
                    t0 = time.monotonic()
                    rows, label_stats = label_round(kind, model, policy, pool, round_index)
                    atomic_torch(labels_file, {"units": rows, "stats": label_stats})
                    history.append(rows)
                    stats[f"label_seconds_{round_index}"] = time.monotonic() - t0
                if checkpoint.is_file():
                    policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
                    continue
                t0 = time.monotonic()
                losses = R.distil_round(policy, R.aggregate_rounds(history), round_index, seed=seed, device=DEVICE)
                atomic_torch(checkpoint, {"model": policy.state_dict(), "losses": losses})
                stats[f"distil_seconds_{round_index}"] = time.monotonic() - t0
                stats[f"final_loss_{round_index}"] = float(np.mean(losses[-60:]))
                log(f"{variant}/{k}/{kind}/{seed}: round {round_index} rows "
                    f"{sum(len(u['labels']) for u in history[-1])}, loss {stats[f'final_loss_{round_index}']:.4f}")
            if failure is None:
                stats["labels"] = {f"round_{r}": rows_stats(hist) for r, hist in enumerate(history)}
                policy.eval()
                evaluation = in_sample(model, policy, pool)
        except ValueError as error:
            if "nonfinite" not in str(error):
                raise
            failure = f"typed failure: {error}"
    wall = time.monotonic() - began
    report = {"schema": SCHEMA_REPORT_TRAIN, "plan_identity": IDENTITY, "variant": variant, "level": k, "kind": kind,
              "lambda": LAMBDA[kind], "seed": seed, "pool": pool_name(variant, k), "condition": condition,
              "predictor_sha256": handoff()["checkpoints"]["predictors"][predictor_key(condition, seed)]["sha256"],
              "init": init, "trained_at": utc_now(), "stats": stats, "wall_seconds": wall,
              "scoring_policy": scoring_policy, "typed_failure": failure}
    if failure is None:
        save_policy(output, variant, k, kind, seed, policy, {"scoring_policy": scoring_policy})
        report.update(policy_sha256=sha256_of(policy_path(output, variant, k, kind, seed)), in_sample=evaluation,
                      parameters=sum(p.numel() for p in policy.parameters()))
    write_json(report_path(output, variant, k, kind, seed), report)
    mark(output, f"train-{variant}-L{k}-{kind}-seed{seed}", wall, wall)
    log(f"{variant}/{k}/{kind}/{seed}: {'trained' if failure is None else failure} ({wall:.0f}s)")


def train_normal(output, workers, worker):
    output = Path(output)
    load_plan(output)
    for group in normal_groups()[worker::workers]:
        train_group(output, *group)
    return 0


def train_fewshot(output, workers, worker):
    output = Path(output)
    load_plan(output)
    missing = [g for g in normal_groups() if not report_path(output, *g).is_file()]
    if missing:
        raise ValueError(f"few-shot retuning needs every normal policy ({len(missing)} missing)")
    for group in fewshot_groups()[worker::workers]:
        train_group(output, *group)
    return 0


# ---------------------------------------------------------------------------
# scoring (deterministic_scoring, processes that did not train)
# ---------------------------------------------------------------------------

def block(model, policy, spec, z0, items, schedules):
    h = heavy()
    R = policy_module()
    actions = h.m99.action_batch(items)
    z_end, timing = R.rollout_endpoints(model, policy, z0.to(DEVICE), actions, spec, select=h.t96.select_pairs,
                                        record_schedules=schedules)
    out = {"rows": [cost_row(item["ordinal"], z_end[slot].cpu(), z0.cpu()) for slot, item in enumerate(items)],
           "candidates": len(items), **{k: v for k, v in timing.items() if k != "schedules"}}
    if schedules:
        out["schedules"] = {str(item["ordinal"]): h.t96.rle(timing["schedules"][slot])
                            for slot, item in enumerate(items)}
    return out


def score_tasks():
    return [(group, world, name, seed) for group in groups() for world in worlds_of(group)
            for name in names_of(group, world) for seed in SEEDS]


def policy_of(name, group):
    """(variant, level, kind) of a policy arm name, or None for a fixed request."""
    if name in FIXED_ARMS:
        return None
    variant, kind = name.split("-")
    return variant, (None if variant == "N" else group_level(group)), kind


def score_record(output, group, world, name, seed, model, scoring_policy, store=None, levels=None,
                 schema=SCHEMA_RECORD):
    """One arm x seed on a group's levels (the frozen evaluation store by default; the request-share
    supplement passes its own store, levels and schema)."""
    h = heavy()
    store = load_eval(output, group, world) if store is None else store
    levels = eval_levels(group) if levels is None else levels
    condition = store["condition"]
    base = {"schema": schema, "plan_identity": IDENTITY, "scoring_policy": scoring_policy,
            "scored_at": utc_now(), "group": group, "world": world, "condition": condition, "arm": name,
            "seed": seed, "engine_seconds": 0,
            "predictor_sha256": handoff()["checkpoints"]["predictors"][predictor_key(condition, seed)]["sha256"]}
    began = time.monotonic()
    parts = policy_of(name, group)
    policy, extra = None, {}
    if parts is None:
        spec = h.t96.arm_spec(name)
    else:
        report = load_report(output, *parts, seed)
        if report["typed_failure"]:
            return {**base, "typed_failure": report["typed_failure"], "levels": {}, "seconds": 0.0}
        policy = load_policy(output, *parts, seed)
        spec = {"kind": "joint"}
        extra = {"policy": {"variant": parts[0], "level": parts[1], "kind": parts[2], "lambda": LAMBDA[parts[2]],
                            "sha256": sha256_of(policy_path(output, *parts, seed))}}
    out = {level: block(model, policy, spec, store["anchors"][level], levels[level]["items"], policy is not None)
           for level in store["levels"]}
    torch.cuda.synchronize()
    return {**base, **extra, "typed_failure": None, "levels": out, "seconds": time.monotonic() - began}


def score(output, workers, worker):
    output = Path(output)
    load_plan(output)
    h = heavy()
    missing = [g for g in (*normal_groups(), *fewshot_groups()) if not report_path(output, *g).is_file()]
    if missing:
        raise ValueError(f"scoring needs every training report ({len(missing)} missing)")
    mine = [t for t in score_tasks()[worker::workers] if not record_path(output, *t).is_file()]
    mine.sort(key=lambda t: (condition_of(t[1], group_level(t[0])), t[3], t[0], t[2]))
    log(f"worker {worker}: {len(mine)} records to score")
    loaded = {}
    for group, world, name, seed in mine:
        condition = condition_of(world, group_level(group))
        t0 = time.monotonic()
        with h.harness.deterministic_scoring() as scoring_policy:
            if (condition, seed) not in loaded:
                loaded.clear()
                torch.cuda.empty_cache()
                loaded[(condition, seed)] = load_world(condition, seed)
            record = score_record(output, group, world, name, seed, loaded[(condition, seed)], scoring_policy)
        write_json(record_path(output, group, world, name, seed), record)
        wall = time.monotonic() - t0
        mark(output, f"score-{group}-{world}-{name}-seed{seed}", wall, wall)
        log(f"worker {worker}: scored {group}/{world}/{name}/{seed} ({wall:.0f}s)")
    return 0


def read_record(output, group, world, name, seed):
    record = read_json(record_path(output, group, world, name, seed))
    if (record.get("schema") != SCHEMA_RECORD or record.get("plan_identity") != IDENTITY
            or (record["group"], record["world"], record["arm"], record["seed"]) != (group, world, name, seed)):
        raise ValueError(f"record {group}/{world}/{name}/{seed} binding differs")
    return record


def load_records(output):
    return {t: read_record(output, *t) for t in score_tasks()}


def record_manifest(output):
    output = Path(output)
    files = sorted(output.glob("records/**/*.json"))
    return {"schema": SCHEMA_MANIFEST, "identity": IDENTITY,
            "records": {str(p.relative_to(output)): sha256_of(p) for p in files}}


# request-share supplement (post-freeze, DESCRIPTIVE, outcome-free; phase-A review F2) -----------------

def supplement_tasks():
    """The frozen policy arms (no fixed request: its shares are fixed by construction) on the supplement
    levels; retuned policies only where the frozen scope trained them (levels 1-7)."""
    return [(group, world, name, seed) for group in supplement_groups() for world in worlds_of(group)
            for name in names_of(group, world) if name not in FIXED_ARMS
            and (name.startswith("N-") or group_level(group) in novel_levels()) for seed in SEEDS]


def supplement_eval_path(output, group, world):
    return Path(output) / SUPPLEMENT / "data" / f"eval-{group}-{world}.pt"


def supplement_record_path(output, group, world, name, seed):
    return Path(output) / SUPPLEMENT / "records" / group / world / f"{name}--seed{seed}.json"


def load_supplement_eval(output, group, world):
    payload = torch.load(supplement_eval_path(output, group, world), map_location="cpu", weights_only=True)
    if (payload.get("schema") != SCHEMA_SUPPLEMENT_EVAL or payload.get("plan_identity") != IDENTITY
            or (payload["group"], payload["world"]) != (group, world)):
        raise ValueError(f"supplement anchors {group}/{world} binding differ")
    return payload


def share_supplement(output):
    """Roll the frozen policies out on the supplement levels for their request schedules: the #110
    anchor carriers (handoff-hashed encoder and scene head), the handoff-hashed worlds, the frozen
    policies (hash-bound to their training reports), deterministic_scoring(), zero engine seconds.
    Writes only under supplement/ (its own ledger); existing files are kept, never rewritten."""
    output = Path(output)
    load_plan(output)
    if not (output / "summary.json").is_file():
        raise ValueError("the supplement follows the frozen publication")
    h = heavy()
    root = output / SUPPLEMENT
    for group in supplement_groups():
        for world in worlds_of(group):
            path = supplement_eval_path(output, group, world)
            if path.is_file():
                continue
            began = time.monotonic()
            with torch.no_grad(), h.harness.deterministic_scoring() as scoring_policy:
                payload = eval_anchors(output, group, world, supplement_levels(group))
            atomic_torch(path, {"schema": SCHEMA_SUPPLEMENT_EVAL, "plan_identity": IDENTITY, "group": group,
                                "world": world, "scoring_policy": scoring_policy, **payload})
            wall = time.monotonic() - began
            write_json(root / "ledger" / f"anchors-{group}-{world}.json",
                       {"wall_seconds": wall, "gpu_seconds": wall, "at": utc_now()})
            log(f"supplement anchors {group}/{world}: {len(payload['levels'])} levels")
    loaded = {}
    for group, world, name, seed in supplement_tasks():
        path = supplement_record_path(output, group, world, name, seed)
        if path.is_file():
            continue
        condition = condition_of(world, group_level(group))
        began = time.monotonic()
        with h.harness.deterministic_scoring() as scoring_policy:
            if (condition, seed) not in loaded:
                loaded.clear()
                torch.cuda.empty_cache()
                loaded[(condition, seed)] = load_world(condition, seed)
            record = score_record(output, group, world, name, seed, loaded[(condition, seed)], scoring_policy,
                                  store=load_supplement_eval(output, group, world), levels=supplement_levels(group),
                                  schema=SCHEMA_SUPPLEMENT_RECORD)
        write_json(path, record)
        wall = time.monotonic() - began
        write_json(root / "ledger" / f"score-{group}-{world}-{name}-seed{seed}.json",
                   {"wall_seconds": wall, "gpu_seconds": wall, "at": utc_now()})
        log(f"supplement {group}/{world}/{name}/{seed} ({wall:.1f}s)")
    write_json(root / "manifest.json", supplement_manifest(output))
    return 0


def read_supplement(output, group, world, name, seed):
    record = read_json(supplement_record_path(output, group, world, name, seed))
    if (record.get("schema") != SCHEMA_SUPPLEMENT_RECORD or record.get("plan_identity") != IDENTITY
            or (record["group"], record["world"], record["arm"], record["seed"]) != (group, world, name, seed)):
        raise ValueError(f"supplement record {group}/{world}/{name}/{seed} binding differs")
    return record


def load_supplement(output):
    return {t: read_supplement(output, *t) for t in supplement_tasks()}


def supplement_manifest(output):
    output = Path(output)
    files = sorted((output / SUPPLEMENT).glob("records/**/*.json")) + sorted((output / SUPPLEMENT).glob("data/*.pt"))
    return {"schema": SCHEMA_SUPPLEMENT_MANIFEST, "identity": IDENTITY,
            "files": {str(p.relative_to(output)): sha256_of(p) for p in files}}


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def view_key(world, name):
    """Arm label of a scored (world, arm): zs:* zero-shot world; predictor:* few-shot world with a
    normal policy or fixed request; policy:* / both:* retuned policies."""
    if name in FIXED_ARMS:
        return f"{'zs' if world == 'zs' else 'predictor'}:{name}"
    variant, kind = name.split("-")
    prefix = {"N": "zs" if world == "zs" else "predictor", "P": "policy", "B": "both"}[variant]
    return f"{prefix}:{ARM_OF_KIND[kind]}"


POLICY_KEYS = tuple(f"{p}:{a}" for p in ("zs", "policy", "predictor", "both") for a in ("J", "J_dec"))


def prior_stats(branches):
    """Engine success counts per grid ordinal (mirrored levels share ordinals)."""
    stats = {}
    for b in branches:
        ordinal = nc.grid_ordinal(b)
        if ordinal is None or b["status"] != "complete":
            continue
        s = stats.setdefault(ordinal, [0, 0])
        s[0] += int(bool(b["pig_removed"]))
        s[1] += 1
    return stats


@functools.lru_cache(maxsize=None)
def priors():
    """zs: the normal fit sides (the zero-shot training data of #110); k: the level-k novel fit sides."""
    complete = nc.complete()
    out = {"zs": prior_stats([b for b in complete if nc.zero_training(b)])}
    for k in novel_levels():
        out[k] = prior_stats([b for b in complete if nc.few_training(b, k)])
    return out


def prior_rows(level, stats):
    return [{"ordinal": item["ordinal"], "excluded": False,
             PRIMARY_COST: -(stats[item["ordinal"]][0] / stats[item["ordinal"]][1]) if item["ordinal"] in stats else 0.0}
            for item in level["items"]]


def record_rows(record, level):
    if record["typed_failure"]:
        return None
    return record["levels"][level]["rows"]


def view_rows(records, group):
    """One row per (mixed level, seed): every arm view of the group, plus the priors (seed-free)."""
    h = heavy()
    k = group_level(group)
    rows = []
    for level, meta in eval_levels(group).items():
        if not mixed(meta):
            continue
        entry = entry_of(meta)
        priors_ = {"zs:prior": h.m99.view(entry, prior_rows(meta, priors()["zs"]), PRIMARY_COST)}
        if k:
            priors_["fs:prior"] = h.m99.view(entry, prior_rows(meta, priors()[k]), PRIMARY_COST)
        for seed in SEEDS:
            views = dict(priors_)
            for world in worlds_of(group):
                for name in names_of(group, world):
                    views[view_key(world, name)] = h.v1.cost_view(
                        entry, record_rows(records[(group, world, name, seed)], level), PRIMARY_COST)
            rows.append({"member": level, "seed": seed, "cell": meta["cell"], "group": group, "views": views})
    return rows


def member_mean(rows, key):
    return heavy().t96.member_mean([{"member": r["member"], "auc": r["views"][key]["auc"]} for r in rows
                                    if key in r["views"]], "auc")


def hindsight(rows, prefix):
    means = {arm: member_mean(rows, f"{prefix}:{arm}") for arm in FIXED_ARMS}
    best = max(FIXED_ARMS, key=lambda a: (-math.inf if means[a] is None else means[a], -FIXED_ARMS.index(a)))
    return best, means


def token(block):
    """delta rule: supported iff mean >= delta and lower > 0; not supported iff upper < delta."""
    if block is None:
        return STOP_TOKEN
    low, high = block["interval"]
    if block["mean"] >= DELTA and low > 0:
        return "supported"
    if high < DELTA:
        return "not_supported_by_this_experiment"
    return STOP_TOKEN


def non_inferiority_token(block):
    """supported iff lower > -delta; not supported iff upper < -delta."""
    if block is None:
        return STOP_TOKEN
    low, high = block["interval"]
    if low > -DELTA:
        return "supported"
    if high < -DELTA:
        return "not_supported_by_this_experiment"
    return STOP_TOKEN


def contrast(rows, first, second, rule="delta"):
    out = heavy().m99.paired(rows, [(1, first), (-1, second)])
    return {**out, "first": first, "second": second, "rule": rule,
            "token": (token if rule == "delta" else non_inferiority_token)(out["estimate"])}


def zero_contrasts(rows, f_zs):
    pairs = {}
    for a in ("J_dec", "J"):
        pairs[f"{a} - F_hindsight"] = (f"zs:{a}", f"zs:{f_zs}")
        pairs[f"{a} - {UNIT_REQUEST}"] = (f"zs:{a}", f"zs:{UNIT_REQUEST}")
        pairs[f"{a} - prior"] = (f"zs:{a}", "zs:prior")
    pairs["J_dec - J"] = ("zs:J_dec", "zs:J")
    return {name: contrast(rows, a, b) for name, (a, b) in pairs.items()}


def fewshot_contrasts(rows, f_zs, f_fs):
    out = {}
    for a in ("J_dec", "J"):
        out[f"{a}: policy - zero-shot"] = contrast(rows, f"policy:{a}", f"zs:{a}")
        out[f"{a}: policy - predictor"] = contrast(rows, f"policy:{a}", f"predictor:{a}", rule="non_inferiority")
        out[f"{a}: predictor - zero-shot"] = contrast(rows, f"predictor:{a}", f"zs:{a}")
        out[f"{a}: both - predictor"] = contrast(rows, f"both:{a}", f"predictor:{a}")
        out[f"{a}: both - zero-shot"] = contrast(rows, f"both:{a}", f"zs:{a}")
        out[f"{a}: policy - F_hindsight (zero-shot world)"] = contrast(rows, f"policy:{a}", f"zs:{f_zs}")
        out[f"{a}: predictor - F_hindsight (few-shot world)"] = contrast(rows, f"predictor:{a}", f"predictor:{f_fs}")
        out[f"{a}: both - F_hindsight (few-shot world)"] = contrast(rows, f"both:{a}", f"predictor:{f_fs}")
        out[f"{a}: policy - few-shot prior"] = contrast(rows, f"policy:{a}", "fs:prior")
        out[f"{a}: both - few-shot prior"] = contrast(rows, f"both:{a}", "fs:prior")
    return out


def auc_table(rows, keys):
    h = heavy()
    return {key: {"auc": h.m99.block_of(rows, key), "member_mean": member_mean(rows, key)} for key in keys
            if rows and key in rows[0]["views"]}


def population_block(rows, few):
    """AUC per arm, F_hindsight and contrasts of one population (zero-shot and few-shot separate)."""
    if not rows:
        return None
    f_zs, means_zs = hindsight(rows, "zs")
    zero_keys = (*(f"zs:{a}" for a in FIXED_ARMS), "zs:J", "zs:J_dec", "zs:prior")
    out = {"cells": sorted({r["cell"] for r in rows}), "levels": len({r["member"] for r in rows}), "units": len(rows),
           "zero_shot": {"auc": auc_table(rows, zero_keys), "F_hindsight": f_zs, "fixed_member_means": means_zs,
                         "contrasts": zero_contrasts(rows, f_zs)}}
    if few:
        f_fs, means_fs = hindsight(rows, "predictor")
        few_keys = (*(f"predictor:{a}" for a in FIXED_ARMS),
                    *(f"{p}:{a}" for p in ("policy", "predictor", "both") for a in ("J", "J_dec")), "fs:prior")
        out["few_shot"] = {"auc": auc_table(rows, few_keys), "F_hindsight": f_fs, "fixed_member_means": means_fs,
                           "contrasts": fewshot_contrasts(rows, f_zs, f_fs)}
    return out


# request shares --------------------------------------------------------------

def level_shares(record, level):
    if record["typed_failure"]:
        return None
    counts = [0] * len(PAIR_NAMES)
    for text in record["levels"][level]["schedules"].values():
        for index in heavy().t96.unrle(text):
            counts[index] += 1
    total = sum(counts)
    return [c / total for c in counts] if total else None


def share_rows(records, group, levels=None):
    """(level, seed) -> {policy key: share vector} over every evaluation level of the group (``levels``:
    the frozen scope's by default); arms without a record (no such policy) are skipped."""
    out = []
    for level, meta in (eval_levels(group) if levels is None else levels).items():
        for seed in SEEDS:
            shares = {}
            for world in worlds_of(group):
                for name in names_of(group, world):
                    if name in FIXED_ARMS or (group, world, name, seed) not in records:
                        continue
                    shares[view_key(world, name)] = level_shares(records[(group, world, name, seed)], level)
            out.append({"member": level, "seed": seed, "cell": meta["cell"], "meta": meta, "shares": shares})
    return out


def marginals(vector):
    deltas = [sum(vector[3 * d + a] for a in range(3)) for d in range(3)]
    alphas = [sum(vector[3 * d + a] for d in range(3)) for a in range(3)]
    return deltas, alphas


def vector_blocks(values, clusters):
    boot = heavy().v1.bootstrap
    if not values:
        return None
    return [boot([v[i] for v in values], clusters) for i in range(len(values[0]))]


def share_block(rows, key, reference=None):
    """Mean request shares of ``key`` (and, with a reference mapping row -> reference share vector,
    the paired shift and its total-variation size)."""
    boot = heavy().v1.bootstrap
    values, clusters, shifts, shift_clusters = [], [], [], []
    for r in rows:
        vector = r["shares"].get(key)
        if vector is None:
            continue
        values.append(vector)
        clusters.append(r["member"])
        if reference is not None:
            base = reference(r)
            if base is not None:
                shifts.append([a - b for a, b in zip(vector, base)])
                shift_clusters.append(r["member"])
    if not values:
        return None
    out = {"units": len(values), "levels": len(set(clusters)), "shares": vector_blocks(values, clusters),
           "delta_shares": vector_blocks([marginals(v)[0] for v in values], clusters),
           "alpha_shares": vector_blocks([marginals(v)[1] for v in values], clusters)}
    if reference is not None and shifts:
        out["shift"] = vector_blocks(shifts, shift_clusters)
        out["delta_shift"] = vector_blocks([marginals(s)[0] for s in shifts], shift_clusters)
        out["alpha_shift"] = vector_blocks([marginals(s)[1] for s in shifts], shift_clusters)
        out["total_variation"] = boot([0.5 * sum(abs(x) for x in s) for s in shifts], shift_clusters)
        out["paired_units"] = len(shifts)
        out["moves_toward"] = [PAIR_NAMES[i] for i, b in enumerate(out["shift"]) if b["interval"][0] > 0]
        out["moves_away"] = [PAIR_NAMES[i] for i, b in enumerate(out["shift"]) if b["interval"][1] < 0]
    return out


def normal_reference(normal_rows):
    """(pair, ordinal, seed) -> the normal policy's shares on the paired normal side (zero-shot world)."""
    out = {}
    for r in normal_rows:
        meta = r["meta"]
        for kind_key in ("zs:J", "zs:J_dec"):
            out[(meta["pair"], meta["ordinal"], r["seed"], kind_key)] = r["shares"].get(kind_key)
    return out


def share_tables(records):
    rows = {g: share_rows(records, g) for g in groups()}
    reference = normal_reference(rows["L0"])

    def ref_for(kind):
        return lambda r: reference.get((r["meta"]["pair"], r["meta"]["ordinal"], r["seed"], f"zs:{kind}"))

    out = {"n1": {k: share_block(rows["n1"], f"zs:{k}") for k in ("J", "J_dec")},
           "L0": {k: share_block(rows["L0"], f"zs:{k}") for k in ("J", "J_dec")}}
    novel_rows = [r for k in novel_levels() for r in rows[f"L{k}"]]
    for name, population in ((f"L{k}", rows[f"L{k}"]) for k in novel_levels()):
        out[name] = {key: share_block(population, key, ref_for(key.split(":")[1])) for key in POLICY_KEYS}
    out["novel"] = {key: share_block(novel_rows, key, ref_for(key.split(":")[1])) for key in POLICY_KEYS}
    out["cells"] = {}
    for cell in sorted({r["cell"] for g in groups() if g != "n1" for r in rows[g]}):
        population = [r for g in groups() if g != "n1" for r in rows[g] if r["cell"] == cell]
        keys = ("zs:J", "zs:J_dec") if cell_level(cell) == 0 else POLICY_KEYS
        out["cells"][cell] = {key: share_block(population, key, None if cell_level(cell) == 0
                                               else ref_for(key.split(":")[1])) for key in keys}
    return out


SUPPLEMENT_LABEL = (
    "post-freeze DESCRIPTIVE supplement (phase-A review F2), outcome-free: request shares of the frozen policies on "
    "every #110-supported cell. The frozen scope's records give the in-scope cells; the supported cells the frozen "
    "scope excluded (no mixed-verdict level, or rollout-error only; all of level 8) were rolled out after the "
    "publication with the same frozen policies, worlds and anchor rule, zero engine seconds, under "
    "deterministic_scoring(), for request shares only (supplement/). No verdict is read, no AUC, contrast or token is "
    "formed, and the frozen tables, scope and dispositions are unchanged. Policy-only and both-retuned policies "
    "exist only for the levels the frozen scope trained them on (1-7)")


def supplement_share_tables(records, supplement):
    """Request shares on every supported cell: in-scope cells from the retained records, the other
    supported cells from the supplement records; per cell, per novelty level and pooled novel 1-8."""
    rows = {g: share_rows(records, g) for g in groups() if g != "n1"}
    for g in supplement_groups():
        rows[g] = rows.get(g, []) + share_rows(supplement, g, supplement_levels(g))
    reference = normal_reference(rows["L0"])

    def ref_for(kind):
        return lambda r: reference.get((r["meta"]["pair"], r["meta"]["ordinal"], r["seed"], f"zs:{kind}"))

    def blocks(population, normal):
        if normal:
            return {key: share_block(population, key) for key in ("zs:J", "zs:J_dec")}
        return {key: share_block(population, key, ref_for(key.split(":")[1])) for key in POLICY_KEYS}

    names = sorted(rows, key=group_level)
    excluded = set(supplement_cells())
    out = {"label": SUPPLEMENT_LABEL, "levels": {}, "cells": {},
           "source": {cell: "supplement" if cell in excluded else "retained" for cell in handoff()["supported_cells"]}}
    for name in names:
        out["levels"][name] = blocks(rows[name], group_level(name) == 0)
    out["levels"]["novel"] = blocks([r for name in names if group_level(name) for r in rows[name]], False)
    for cell in handoff()["supported_cells"]:
        population = [r for r in rows[f"L{cell_level(cell)}"] if r["cell"] == cell]
        out["cells"][cell] = blocks(population, cell_level(cell) == 0)
    return out


def issue101_reference():
    """The #101 J / J_dec request counts on the #109 evaluation split, grid (E99 Gate-B predictor;
    external reference, different carrier)."""
    compute = read_json(DAP / "compute.json")
    arms = compute["primary_arms"]
    grid = compute["inventories"]["grid"]
    out = {}
    for label, arm in (("J", arms["J"]), ("J_dec", arms["J_dec"])):
        counts = grid["schedules"][arm]["pair_counts"]
        total = sum(counts)
        out[label] = {"arm": arm, "shares": [c / total for c in counts], "auc": grid["rankers"][arm]["auc"]}
    return out


# wrong anchor ------------------------------------------------------------------

def wrong_anchor_rows(records, group, keys):
    """Per mixed target level and seed: the arm's AUC from its own anchor (right) and the mean AUC of
    the target's verdicts against the rollouts from every other level's anchor of the group (wrong)."""
    h = heavy()
    levels = eval_levels(group)
    by_key = {view_key(w, n): (w, n) for w in worlds_of(group) for n in names_of(group, w) if n not in FIXED_ARMS}
    rows = []
    for level, meta in levels.items():
        if not mixed(meta):
            continue
        entry = entry_of(meta)
        for seed in SEEDS:
            views = {}
            for key in keys:
                if key not in by_key:
                    continue
                record = records[(group, *by_key[key], seed)]
                if record["typed_failure"]:
                    views[key] = None
                    continue
                right = h.m99.view(entry, record["levels"][level]["rows"], PRIMARY_COST)["auc"]
                wrong = [h.m99.view(entry, record["levels"][other]["rows"], PRIMARY_COST)["auc"]
                         for other in levels if other != level]
                wrong = [w for w in wrong if w is not None]
                views[key] = {"right": right, "wrong": float(np.mean(wrong)) if wrong else None,
                              "wrong_anchors": len(wrong)}
            rows.append({"member": level, "seed": seed, "cell": meta["cell"], "views": views})
    return rows


def wrong_block(rows, key):
    if not rows or key not in rows[0]["views"]:
        return None
    boot = heavy().v1.bootstrap

    def field(name):
        values = [(r["views"][key][name], r["member"]) for r in rows
                  if r["views"][key] is not None and r["views"][key][name] is not None]
        return boot([v for v, _ in values], [m for _, m in values]) if values else None
    diff = [(r["views"][key]["right"] - r["views"][key]["wrong"], r["member"]) for r in rows
            if r["views"][key] is not None and None not in (r["views"][key]["right"], r["views"][key]["wrong"])]
    block = boot([v for v, _ in diff], [m for _, m in diff]) if diff else None
    return {"right": field("right"), "wrong": field("wrong"), "right_minus_wrong": block, "token": token(block)}


def wrong_tables(records):
    out = {}
    novel_rows = []
    for group in groups():
        keys = ("zs:J", "zs:J_dec") if not group_level(group) else POLICY_KEYS
        rows = wrong_anchor_rows(records, group, keys)
        out[group] = {key: wrong_block(rows, key) for key in keys}
        if group_level(group):
            novel_rows += rows
    out["novel"] = {key: wrong_block(novel_rows, key) for key in POLICY_KEYS}
    return out


# training and compute ----------------------------------------------------------

def training_table(output):
    out = {}
    for group in (*normal_groups(), *fewshot_groups()):
        report = load_report(output, *group)
        variant, k, kind, seed = group
        out.setdefault(pool_name(variant, k), {}).setdefault(kind, {})[str(seed)] = {
            "typed_failure": report["typed_failure"], "wall_seconds": report["wall_seconds"],
            "policy_sha256": report.get("policy_sha256"), "in_sample": report.get("in_sample"),
            "labels": report["stats"].get("labels"), "init": report["init"],
            "final_loss": [report["stats"].get(f"final_loss_{r}") for r in range(2)]}
    return out


def adaptation_cost(output):
    """Few-shot retune GPU seconds per (level, seed): the policy retune (this ticket, P variant,
    labels + distillation + in-sample check) vs the #110 predictor retune of the same level and seed
    (hybrid member training; the shared encoder and scene-head adaptation reported separately)."""
    ledger = NC / "ledger"
    rows = []
    for k in novel_levels():
        encoder = read_json(ledger / f"encoder-few-shot-{k}.json")["gpu_seconds"]
        scene = read_json(ledger / f"scene-few-shot-{k}.json")["gpu_seconds"]
        for seed in SEEDS:
            predictor = read_json(ledger / f"train-few-shot-{k}-hybrid-seed{seed}.json")["gpu_seconds"]
            rows.append({"level": k, "seed": seed, "predictor_member_gpu_seconds": predictor,
                         "encoder_gpu_seconds": encoder, "scene_head_gpu_seconds": scene,
                         **{f"policy_{kind}_gpu_seconds": load_report(output, "P", k, kind, seed)["wall_seconds"]
                            for kind in KINDS},
                         **{f"both_policy_{kind}_gpu_seconds": load_report(output, "B", k, kind, seed)["wall_seconds"]
                            for kind in KINDS}})
    shared = ("encoder_gpu_seconds", "scene_head_gpu_seconds")  # one adaptation per level, shared by the seeds
    totals = {key: sum(r[key] for r in rows if key not in shared or r["seed"] == SEEDS[0])
              for key in rows[0] if key.endswith("gpu_seconds")}
    parameters = {"policy": load_report(output, "N", None, "dec", SEEDS[0]).get("parameters"),
                  "predictor": nc.nv().parameter_count(nc.nv().HybridPredictorV2())}
    return {"rows": rows, "totals": totals, "parameters": parameters,
            "cheap": {kind: totals[f"policy_{kind}_gpu_seconds"] < totals["predictor_member_gpu_seconds"]
                      for kind in KINDS}}


def invalidate(obj):
    """A failed guard turns every token into readiness_or_precision_insufficient (frozen guards.failure);
    estimates, AUCs and shares are kept."""
    if isinstance(obj, dict):
        return {k: STOP_TOKEN if k == "token" else invalidate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [invalidate(v) for v in obj]
    return obj


def disposition(populations, cost, guards_ok):
    """Pre-declared: 'request selection is a cheap adaptation surface under novelty' on the pooled
    novel levels, J_dec primary (J secondary, same rule)."""
    out = {}
    for arm in ("J_dec", "J"):
        block = populations["novel"]["few_shot"]["contrasts"]
        gain = block[f"{arm}: policy - zero-shot"]
        parity = block[f"{arm}: policy - predictor"]
        kind = "dec" if arm == "J_dec" else "pe"
        cheap = cost["cheap"][kind]
        tokens = (gain["token"], parity["token"])
        if not guards_ok:
            final = STOP_TOKEN
        elif tokens == ("supported", "supported") and cheap:
            final = "supported"
        elif "not_supported_by_this_experiment" in tokens or not cheap:
            final = "not_supported_by_this_experiment"
        else:
            final = STOP_TOKEN
        out[arm] = {"gain_token": gain["token"] if guards_ok else STOP_TOKEN, "gain": gain["estimate"],
                    "parity_token": parity["token"] if guards_ok else STOP_TOKEN, "parity": parity["estimate"],
                    "cheap": cheap, "token": final, "role": "primary" if arm == "J_dec" else "secondary"}
    return out


def guard_checks(output, plan, records, training):
    h = heavy()
    policy_roster = {lv["identity"] for lv in h.dap.policy_levels()}
    fit_roster = {lv["identity"] for lv in h.rt.fit_levels()}
    eval_roster = {lv["identity"] for lv in h.gb.evaluation_levels()}
    pool = load_pool(output, "normal")
    fs_ok = True
    for k in novel_levels():
        for world in ("zs", "fs"):
            payload = load_pool(output, f"{world}-L{k}")
            keys = set(payload["shot_keys"])
            fs_ok &= keys == {b["key"] for b in fewshot_branches(k)}
            fs_ok &= all(b["stream"] == nc.STREAM_FIT and b["side"] == "novel" and b["novelty_level"] == k
                         for b in usable() if b["key"] in keys)
    scored = {lv for g in groups() if g != "n1" for lv in eval_levels(g)}
    training_levels = {b["level"] for b in usable() if b["stream"] != nc.STREAM_EVAL}
    trained = sorted(load_report(output, *g)["trained_at"] for g in (*normal_groups(), *fewshot_groups()))
    scored_at = sorted(r["scored_at"] for r in records.values())
    reports = {(g[0], g[1], g[2], g[3]): load_report(output, *g) for g in (*normal_groups(), *fewshot_groups())}
    binding = all(r["predictor_sha256"] == handoff()["checkpoints"]["predictors"][predictor_key(r["condition"], r["seed"])]["sha256"]
                  for r in records.values())
    for (group, world, name, seed), record in records.items():
        parts = policy_of(name, group)
        if parts is None or record["typed_failure"]:
            continue
        binding &= record["policy"]["sha256"] == reports[(*parts, seed)]["policy_sha256"]
    for (variant, k, kind, seed), report in reports.items():
        if variant != "N" and report["init"] is not None:
            binding &= report["init"]["policy_sha256"] == reports[("N", None, kind, seed)]["policy_sha256"]
    handoff_rows = {row["key"]: row["pig_removed"] for cell in in_scope_cells()
                    for row in handoff()["evaluation_inventories"]["by_cell"][cell]}
    mine = {item["key"]: meta["verdicts"][item["ordinal"]] for g in groups() if g != "n1"
            for meta in eval_levels(g).values() for item in meta["items"]}
    smoke = read_json(Path(output) / "smoke.json")
    return {
        "G1_isolation": {"pass": bool(set(pool["levels"]) == policy_roster
                                      and not (policy_roster & (fit_roster | eval_roster)) and fs_ok
                                      and not (scored & training_levels)
                                      and set(eval_levels("n1")) <= eval_roster)},
        "G2_scoring_policy": {"pass": all(r["scoring_policy"] == h.harness.SCORING_POLICY for r in records.values())},
        "G3_chronology": {"plan_frozen_at": plan["frozen_at"], "first_trained_at": trained[0],
                          "last_trained_at": trained[-1], "first_scored_at": scored_at[0],
                          "pass": plan["frozen_at"] <= trained[0] and trained[-1] <= scored_at[0]},
        "G4_checkpoint_binding": {"pass": bool(binding)},
        "G5_prefix_parse_replication": {"pass": bool(smoke["prefix_replication"]["pass"]),
                                        "evidence": "smoke.json prefix_replication; re-checked by --validate"},
        "G6_handoff_inventories": {"pass": handoff_rows == mine, "branches": len(mine)},
    }


def compute_tables(output, plan):
    output = Path(output)
    records = load_records(output)
    rows = {g: view_rows(records, g) for g in groups()}
    populations = {g: population_block(rows[g], bool(group_level(g))) for g in groups()}
    novel_rows = [r for k in novel_levels() for r in rows[f"L{k}"]]
    populations["novel"] = population_block(novel_rows, True)
    cells = {}
    for cell in in_scope_cells():
        population = [r for g in groups() if g != "n1" for r in rows[g] if r["cell"] == cell]
        cells[cell] = population_block(population, cell_level(cell) > 0)
    training = training_table(output)
    cost = adaptation_cost(output)
    guards = guard_checks(output, plan, records, training)
    guards_ok = all(g["pass"] for g in guards.values())
    dispositions = disposition(populations, cost, guards_ok)
    wrong = wrong_tables(records)
    if not guards_ok:  # frozen guards.failure: every token, not only the disposition
        populations, cells, wrong = invalidate(populations), invalidate(cells), invalidate(wrong)
    supplement = load_supplement(output)
    supplement_ledger = [read_json(p) for p in sorted((output / SUPPLEMENT / "ledger").glob("*.json"))]
    phases = ledger_phases(output)
    smoke = read_json(output / "smoke.json")
    all_phases = {"smoke": {"wall_seconds": smoke["wall_seconds"], "gpu_seconds": smoke["wall_seconds"]},
                  **{k: {"wall_seconds": v["wall_seconds"], "gpu_seconds": v["gpu_seconds"]} for k, v in phases.items()}}
    stage = defaultdict(lambda: {"wall_seconds": 0.0, "gpu_seconds": 0.0})
    for name, block in all_phases.items():
        key = ("train-normal" if name.startswith("train-N-") else "train-fewshot" if name.startswith("train-")
               else name.split("-")[0])
        stage[key]["wall_seconds"] += block["wall_seconds"]
        stage[key]["gpu_seconds"] += block["gpu_seconds"]
    tokens_scope = scope_table()
    return {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
        "scope": tokens_scope, "groups": {g: {"levels": len(eval_levels(g)),
                                              "mixed_levels": sum(mixed(m) for m in eval_levels(g).values())}
                                          for g in groups()},
        "guards": guards, "disposition": dispositions,
        "populations": populations, "cells": cells,
        "shares": share_tables(records), "issue101_reference": issue101_reference(),
        "shares_supported_cells": {
            **supplement_share_tables(records, supplement),
            "records": len(supplement), "manifest": SUPPLEMENT_MANIFEST,
            "compute": {"gpu_seconds": sum(b["gpu_seconds"] for b in supplement_ledger),
                        "wall_seconds": sum(b["wall_seconds"] for b in supplement_ledger), "engine_seconds": 0}},
        "wrong_anchor": wrong, "training": training, "adaptation_cost": cost,
        "typed_failures": sorted(f"{g}/{w}/{n}/{s}" for (g, w, n, s), r in records.items() if r["typed_failure"]),
        "compute": {"stages": dict(sorted(stage.items())),
                    "gpu_seconds": sum(b["gpu_seconds"] for b in all_phases.values()),
                    "wall_seconds": sum(b["wall_seconds"] for b in all_phases.values()), "engine_seconds": 0,
                    "hardware": "4 x RTX 5090 (one process per GPU)"},
    }


# ---------------------------------------------------------------------------
# renderings
# ---------------------------------------------------------------------------

def fmt(value, digits=3):
    return "n/a" if value is None else format(value, f".{digits}f")


def iv(block, digits=3):
    if block is None:
        return "n/a"
    return f"{fmt(block['mean'], digits)} [{fmt(block['interval'][0], digits)}, {fmt(block['interval'][1], digits)}]"


def pct(block):
    return "n/a" if block is None else f"{100 * block['mean']:.0f}"


def signed_pct(block):
    if block is None:
        return "n/a"
    return f"{100 * block['mean']:+.0f} [{100 * block['interval'][0]:+.0f}, {100 * block['interval'][1]:+.0f}]"


def auc_of(population, section, key):
    if population is None or section not in population:
        return None
    entry = population[section]["auc"].get(key)
    return None if entry is None else entry["auc"]


def population_label(name):
    if name == "n1":
        return "N1 (#109 evaluation)"
    if name == "novel":
        return "novel 1-7 pooled"
    k = group_level(name)
    return f"{k}: {NOVELTY_TYPES[k]}"


def findings_md(plan, compute):
    lines = []
    add = lines.append
    pops = compute["populations"]
    order = ["n1", "L0", *(f"L{k}" for k in novel_levels()), "novel"]
    novel_order = [*(f"L{k}" for k in novel_levels()), "novel"]
    d = compute["disposition"]
    add("# Issue #107: granularity under novelty (request selection on the #110 contract-v2 cells)")
    add("")
    add(f"Frozen {plan['frozen_at']}. One score. Intervals: member (evaluation level)-clustered bootstrap, 10 000 "
        f"draws, DESCRIPTIVE. Margin delta = {DELTA}. Zero-shot and few-shot are separate sections; no estimate pools "
        "their rows. Zero engine seconds.")
    add("")
    add("## Disposition: 'request selection is a cheap adaptation surface under novelty'")
    add("")
    add("| policy | role | policy-only retune - zero-shot | token | policy-only - predictor-only (non-inferiority) | token | "
        "policy retune cheaper | disposition |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for arm in ("J_dec", "J"):
        x = d[arm]
        add(f"| {arm} | {x['role']} | {iv(x['gain'])} | {x['gain_token']} | {iv(x['parity'])} | {x['parity_token']} | "
            f"{'yes' if x['cheap'] else 'no'} | **{x['token']}** |")
    add("")
    add(f"Rule: {plan['decision_rule']['disposition']}")
    add("")
    add("Guards: " + ", ".join(f"{k} {'pass' if v['pass'] else 'FAIL'}" for k, v in compute["guards"].items()) + ".")
    add("")
    add("## Scope (45 #110-supported cells)")
    add("")
    out_scope = {c: v for c, v in compute["scope"].items() if not v["in_scope"]}
    add(f"In scope: {sum(v['in_scope'] for v in compute['scope'].values())} cells with mixed-verdict evaluation levels. "
        f"Out of scope ({len(out_scope)} cells, token `{STOP_TOKEN}`):")
    add("")
    add("| cell | reason |")
    add("| --- | --- |")
    for cell, v in out_scope.items():
        add(f"| {cell} | {v['reason']} |")
    add("")
    add("## Zero-shot (normal-trained world and policies, frozen): ranking AUC per arm")
    add("")
    add("| novelty type | levels (mixed) | J | J_dec | F-15-macro | F_hindsight | no-model prior |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for name in order:
        p = pops[name]
        f = p["zero_shot"]["F_hindsight"]
        add(f"| {population_label(name)} | {p['levels']} | {iv(auc_of(p, 'zero_shot', 'zs:J'))} | "
            f"{iv(auc_of(p, 'zero_shot', 'zs:J_dec'))} | {iv(auc_of(p, 'zero_shot', 'zs:F-15-macro'))} | "
            f"{f}: {iv(auc_of(p, 'zero_shot', f'zs:{f}'))} | {iv(auc_of(p, 'zero_shot', 'zs:prior'))} |")
    add("")
    add("Nine fixed requests, zero-shot world (AUC):")
    add("")
    add("| novelty type | " + " | ".join(FIXED_ARMS) + " |")
    add("| --- |" + " --- |" * len(FIXED_ARMS))
    for name in order:
        p = pops[name]
        add(f"| {population_label(name)} | " + " | ".join(iv(auc_of(p, "zero_shot", f"zs:{a}")) for a in FIXED_ARMS) + " |")
    add("")
    add("Zero-shot contrasts (paired by level and seed):")
    add("")
    names = list(pops["L0"]["zero_shot"]["contrasts"])
    add("| novelty type | " + " | ".join(names) + " |")
    add("| --- |" + " --- |" * len(names))
    for name in order:
        c = pops[name]["zero_shot"]["contrasts"]
        add(f"| {population_label(name)} | " + " | ".join(f"{iv(c[n]['estimate'])} {short(c[n]['token'])}"
                                                           for n in names) + " |")
    add("")
    add("## Few-shot (level-k novel fit sides): ranking AUC per arm")
    add("")
    add("| novelty type | levels (mixed) | J policy-only | J_dec policy-only | J predictor-only | J_dec predictor-only | "
        "J both | J_dec both | F_hindsight (few-shot world) | few-shot prior |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in novel_order:
        p = pops[name]
        f = p["few_shot"]["F_hindsight"]
        cells_ = [iv(auc_of(p, "few_shot", f"{v}:{a}")) for v in ("policy", "predictor", "both") for a in ("J", "J_dec")]
        add(f"| {population_label(name)} | {p['levels']} | " + " | ".join(cells_) +
            f" | {f}: {iv(auc_of(p, 'few_shot', f'predictor:{f}'))} | {iv(auc_of(p, 'few_shot', 'fs:prior'))} |")
    add("")
    add("Nine fixed requests, few-shot world (AUC):")
    add("")
    add("| novelty type | " + " | ".join(FIXED_ARMS) + " |")
    add("| --- |" + " --- |" * len(FIXED_ARMS))
    for name in novel_order:
        p = pops[name]
        add(f"| {population_label(name)} | " + " | ".join(iv(auc_of(p, "few_shot", f"predictor:{a}"))
                                                           for a in FIXED_ARMS) + " |")
    add("")
    add("Few-shot contrasts, J_dec (paired by level and seed; `policy - predictor` uses the non-inferiority rule):")
    add("")
    for arm in ("J_dec", "J"):
        names = [n for n in pops["novel"]["few_shot"]["contrasts"] if n.startswith(f"{arm}:")]
        add(f"{arm}:")
        add("")
        add("| novelty type | " + " | ".join(n.split(": ", 1)[1] for n in names) + " |")
        add("| --- |" + " --- |" * len(names))
        for name in novel_order:
            c = pops[name]["few_shot"]["contrasts"]
            add(f"| {population_label(name)} | " + " | ".join(f"{iv(c[n]['estimate'])} {short(c[n]['token'])}"
                                                               for n in names) + " |")
        add("")
    add("Token marks: S supported, N not_supported_by_this_experiment, I readiness_or_precision_insufficient "
        "(DESCRIPTIVE per novelty type; the disposition uses the pooled novel row only).")
    add("")
    add("## Request shares (% of rollout steps) and the shift toward/away from the paired normal operating point")
    add("")
    add("Reference: the same kind's normal policy on the paired normal side (same pair and ordinal), zero-shot world. "
        "Shares use every evaluation level of the in-scope cells (outcome-free).")
    add("")
    shares = compute["shares"]
    ref = compute["issue101_reference"]
    add("| population | arm | " + " | ".join(PAIR_NAMES) + " | TV shift vs paired normal | moves toward | moves away |")
    add("| --- | --- |" + " --- |" * len(PAIR_NAMES) + " --- | --- | --- |")
    for kind in ("J", "J_dec"):
        add(f"| #101 N1 (E99 Gate-B predictor) | {kind} | " + " | ".join(f"{100 * s:.0f}" for s in ref[kind]["shares"])
            + " | | | |")
        for name in ("n1", "L0"):
            b = shares[name][kind]
            add(f"| {population_label(name)} | zs:{kind} | " + " | ".join(pct(x) for x in b["shares"]) + " | | | |")
    for name in novel_order:
        for key in POLICY_KEYS:
            b = shares[name][key]
            if b is None:
                continue
            add(f"| {population_label(name)} | {key} | " + " | ".join(pct(x) for x in b["shares"]) +
                f" | {iv(b.get('total_variation'), 2)} | {', '.join(b.get('moves_toward', [])) or '-'} | "
                f"{', '.join(b.get('moves_away', [])) or '-'} |")
    add("")
    add("Shift of the horizon marginal (percentage points, Delta = 1 / 5 / 15) vs the paired normal side:")
    add("")
    add("| population | arm | Delta 1 | Delta 5 | Delta 15 | continuous | micro | macro |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in novel_order:
        for key in POLICY_KEYS:
            b = shares[name][key]
            if b is None or "delta_shift" not in b:
                continue
            add(f"| {population_label(name)} | {key} | " + " | ".join(signed_pct(x) for x in b["delta_shift"]) + " | "
                + " | ".join(signed_pct(x) for x in b["alpha_shift"]) + " |")
    add("")
    s = compute["shares_supported_cells"]
    add("## Request shares on every supported cell (post-freeze DESCRIPTIVE supplement)")
    add("")
    add(f"{s['label']}. {s['records']} supplement records, {s['compute']['gpu_seconds']:.0f} GPU s, 0 engine s; "
        f"hashes in `{s['manifest']}`. Horizon (Delta 1 / 5 / 15) and description (continuous / micro / macro) "
        "marginals, % of rollout steps; shift vs the same kind's normal policy on the paired normal side.")
    add("")
    add("| novelty type | arm | units | Delta 1 | Delta 5 | Delta 15 | continuous | micro | macro | TV shift vs paired "
        "normal | moves toward | moves away |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, blocks in s["levels"].items():
        for key, b in blocks.items():
            if b is None:
                continue
            label = "novel 1-8 pooled (every supported cell)" if name == "novel" else population_label(name)
            add(f"| {label} | {key} | {b['units']} | " + " | ".join(pct(x) for x in b["delta_shares"]) + " | "
                + " | ".join(pct(x) for x in b["alpha_shares"]) + f" | {iv(b.get('total_variation'), 2)} | "
                f"{', '.join(b.get('moves_toward', [])) or '-'} | {', '.join(b.get('moves_away', [])) or '-'} |")
    add("")
    add("Per cell, zero-shot J_dec and J (source: retained frozen-scope records or the supplement):")
    add("")
    add("| cell | source | arm | Delta 1 | Delta 5 | Delta 15 | continuous | micro | macro |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for cell, blocks in s["cells"].items():
        for key in ("zs:J_dec", "zs:J"):
            b = blocks.get(key)
            if b is None:
                continue
            add(f"| {cell} | {s['source'][cell]} | {key} | " + " | ".join(pct(x) for x in b["delta_shares"]) + " | "
                + " | ".join(pct(x) for x in b["alpha_shares"]) + " |")
    add("")
    add("## Wrong-anchor control (right - wrong anchor AUC)")
    add("")
    add("| population | " + " | ".join(POLICY_KEYS) + " |")
    add("| --- |" + " --- |" * len(POLICY_KEYS))
    wrong = compute["wrong_anchor"]
    for name in order:
        add(f"| {population_label(name)} | " + " | ".join(
            iv(wrong[name][k]["right_minus_wrong"]) if wrong[name].get(k) else "-" for k in POLICY_KEYS) + " |")
    add("")
    add("## Per cell (J_dec AUC)")
    add("")
    add("| cell | levels | zero-shot | policy-only | predictor-only | both | F_hindsight zero-shot |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for cell, p in compute["cells"].items():
        if p is None:
            continue
        few = "few_shot" in p
        add(f"| {cell} | {p['levels']} | {iv(auc_of(p, 'zero_shot', 'zs:J_dec'))} | "
            f"{iv(auc_of(p, 'few_shot', 'policy:J_dec')) if few else '-'} | "
            f"{iv(auc_of(p, 'few_shot', 'predictor:J_dec')) if few else '-'} | "
            f"{iv(auc_of(p, 'few_shot', 'both:J_dec')) if few else '-'} | "
            f"{p['zero_shot']['F_hindsight']}: {iv(auc_of(p, 'zero_shot', 'zs:' + p['zero_shot']['F_hindsight']))} |")
    add("")
    add("## Adaptation cost (GPU s, summed over levels 1-7 x 3 seeds; the shared encoder and scene-head "
        "adaptations once per level)")
    add("")
    cost = compute["adaptation_cost"]
    for key, value in cost["totals"].items():
        add(f"- {key}: {value:.0f}")
    add(f"- parameters: policy {cost['parameters']['policy']}, LC hybrid predictor {cost['parameters']['predictor']}")
    add("")
    c = compute["compute"]
    add(f"## Compute: {c['gpu_seconds']:.0f} GPU s, {c['wall_seconds']:.0f} process-wall s, 0 engine s ({c['hardware']})")
    add("")
    add("| stage | GPU s | wall s |")
    add("| --- | --- | --- |")
    for name, b in c["stages"].items():
        add(f"| {name} | {b['gpu_seconds']:.0f} | {b['wall_seconds']:.0f} |")
    add("")
    if compute["typed_failures"]:
        add("Typed failures (recorded, not retried): " + ", ".join(compute["typed_failures"]))
        add("")
    return "\n".join(lines)


def short(token_):
    return {"supported": "S", "not_supported_by_this_experiment": "N", STOP_TOKEN: "I"}[token_]


def summary(plan, compute):
    pops = compute["populations"]
    return {"schema": SCHEMA_SUMMARY, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "validation_command": VALIDATION_COMMAND, "disposition": compute["disposition"],
            "guards": {k: v["pass"] for k, v in compute["guards"].items()},
            "in_scope_cells": in_scope_cells(),
            "out_of_scope": {c: v["reason"] for c, v in compute["scope"].items() if not v["in_scope"]},
            "auc": {name: {"zero_shot": {k: (v["auc"] or {}).get("mean") for k, v in p["zero_shot"]["auc"].items()},
                           **({"few_shot": {k: (v["auc"] or {}).get("mean") for k, v in p["few_shot"]["auc"].items()}}
                              if "few_shot" in p else {})}
                    for name, p in pops.items()},
            "compute": {k: compute["compute"][k] for k in ("gpu_seconds", "wall_seconds", "engine_seconds")},
            "typed_failures": compute["typed_failures"],
            "shares_supported_cells": {
                "label": compute["shares_supported_cells"]["label"],
                "source": compute["shares_supported_cells"]["source"],
                "levels": {name: {key: {"delta": [b["mean"] for b in v["delta_shares"]],
                                        "alpha": [b["mean"] for b in v["alpha_shares"]]}
                                  for key, v in blocks.items() if v is not None}
                           for name, blocks in compute["shares_supported_cells"]["levels"].items()}}}


def rendered(plan, compute):
    return {"compute.json": json_text(compute), "summary.json": json_text(summary(plan, compute)),
            "findings.md": findings_md(plan, compute) + "\n"}


# ---------------------------------------------------------------------------
# smoke (pre-freeze; fit / policy data only, no evaluation-split model output)
# ---------------------------------------------------------------------------

def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the smoke is a pre-freeze control")
    began = time.monotonic()
    h = heavy()
    R = policy_module()
    checks, timings = {}, {}

    # 1. accounting and handoff consistency (membership only)
    scope = scope_table()
    accounting = {"in_scope_cells": in_scope_cells(), "novel_levels": list(novel_levels()),
                  "groups": {g: {"levels": len(eval_levels(g)), "mixed_levels": sum(mixed(m) for m in eval_levels(g).values())}
                             for g in groups()},
                  "policy_pool": h.dap.pool_accounting(),
                  "fewshot_pools": {k: {"shots": len(fewshot_branches(k)),
                                        "levels": len({b["level"] for b in fewshot_branches(k)})}
                                    for k in novel_levels()}}
    handoff_rows = {row["key"]: row["pig_removed"] for cell in in_scope_cells()
                    for row in handoff()["evaluation_inventories"]["by_cell"][cell]}
    mine = {item["key"]: meta["verdicts"][item["ordinal"]] for g in groups() if g != "n1"
            for meta in eval_levels(g).values() for item in meta["items"]}
    checks["handoff_inventories_equal"] = handoff_rows == mine
    checks["handoff_mixed_levels_equal"] = all(
        sum(mixed(m) for g in groups() if g != "n1" for m in eval_levels(g).values() if m["cell"] == cell)
        == scope[cell]["mixed_levels"] for cell in in_scope_cells())
    checks["pig_columns"] = (h.nv.PIG_PRESENCE == handoff()["contract"]["pig_presence_column"]
                             and list(h.nv.PIG_CENTER) == handoff()["contract"]["pig_center_columns"])
    for condition in (ZERO, *(f"few-shot-{k}" for k in novel_levels())):
        handoff_checkpoint("encoders", condition)
        handoff_checkpoint("scene_heads", condition)
        for seed in SEEDS:
            checks.setdefault("handoff_predictors_usable", True)
            checks["handoff_predictors_usable"] &= handoff_checkpoint("predictors", predictor_key(condition, seed))["usable"]
    # paired normal side exists for every novel level
    normals = {(m["pair"], m["ordinal"]) for m in eval_levels("L0").values()}
    novel = [(m["pair"], m["ordinal"]) for k in novel_levels() for m in eval_levels(f"L{k}").values()]
    accounting["novel_levels_with_paired_normal"] = sum(p in normals for p in novel)
    accounting["novel_levels_total"] = len(novel)
    # prior keying: grid ordinal <-> action is one-to-one per slingshot side
    action_by_ordinal = defaultdict(set)
    for b in nc.complete():
        o = nc.grid_ordinal(b)
        if o is not None:
            action_by_ordinal[(b["slingshot_side"], o)].add((b["action"]["drag_x"], b["action"]["drag_y"]))
    checks["prior_grid_ordinal_one_to_one"] = all(len(v) == 1 for v in action_by_ordinal.values())
    checks["prior_mirrored_ordinals"] = all(
        {(-x, y) for x, y in action_by_ordinal[("right", o)]} == action_by_ordinal[("left", o)]
        for (side, o) in action_by_ordinal if side == "right")

    # 2. the private module: at the v1 width it is the #101 module bit for bit; at v2 it accepts 370 values
    v1_copy = private_module("issue107_request_policy_check")
    reference = h.rp.new_policy(h.PAIRS, SEEDS[0], device=DEVICE)
    copy = v1_copy.new_policy(h.PAIRS, SEEDS[0], device=DEVICE)
    generator = torch.Generator().manual_seed(1)
    z = torch.randn(64, h.rp.DIM, generator=generator).to(DEVICE)
    a = torch.randn(64, 5, generator=generator).to(DEVICE)
    remaining = torch.randint(1, 226, (64,), generator=generator).to(DEVICE)
    checks["private_module_equals_101_module"] = (
        all(torch.equal(x, y) for x, y in zip(reference.state_dict().values(), copy.state_dict().values()))
        and torch.equal(reference(z, a, remaining), copy(z, a, remaining)))
    v2_policy = R.new_policy(h.PAIRS, SEEDS[0], device=DEVICE)
    checks["v2_policy_input_width"] = v2_policy.layers[0].in_features == h.nv.DIM + 6
    checks["module_sha256"] = sha256_of(ROOT / MODULE)

    # 3. prefix parses reproduce the #110 carriers (fit shots, zero-shot and few-shot conditions)
    replication = []
    with torch.no_grad():
        samples = [(ZERO, next(b for b in usable() if nc.zero_training(b) and b["stream"] == nc.STREAM_FIT)),
                   (ZERO, fewshot_branches(1)[0]), ("few-shot-1", fewshot_branches(1)[1]),
                   (f"few-shot-{SMOKE_FS_LEVEL}", fewshot_branches(SMOKE_FS_LEVEL)[0])]
        for condition, b in samples:
            encoder = nc.load_encoder(NC, condition)[0]
            scene_head = nc.load_scene(str(NC), condition)
            t0 = time.monotonic()
            mine_, end, action = parse_prefix(encoder, scene_head, b["attempt"])
            timings.setdefault("prefix_parse_seconds", []).append(time.monotonic() - t0)
            stored = nc.full_carriers(NC, condition, b["key"]).cpu()[:len(mine_)]
            replication.append({"key": b["key"], "condition": condition, "positions": len(mine_),
                                "end": end, "equal": bool(torch.equal(mine_, stored)),
                                "max_abs": float((mine_ - stored).abs().max())})
    checks["prefix_replication"] = all(r["equal"] for r in replication)

    # 4. normal pool slice: one policy level with a mixed grid cell (training data only)
    entries = h.dap.policy_cell_entries()
    member = next(m for (m, inv) in entries if inv == "grid")
    level_branches = [b for b in policy_branches() if b["member"] == member]
    encoder, scene_head = zero_world_parsers()
    shots = []
    with torch.no_grad():
        t0 = time.monotonic()
        for b in level_branches:
            carriers, end, action = parse_prefix(encoder, scene_head, b["attempt"])
            checks.setdefault("policy_actions_equal", True)
            checks["policy_actions_equal"] &= bool(torch.equal(action, h.wlc.action_tensor(b["candidate"]["action"], "cpu")[0]))
            shots.append({"key": b["key"], "member": member, "carriers": carriers, "end": end, "action": action})
        timings["policy_shot_parse_seconds"] = (time.monotonic() - t0) / len(level_branches)
    anchor = shots[0]["carriers"][0].clone()
    pool = {"levels": [member], "anchors": {member: anchor}, "shot_keys": [s["key"] for s in shots],
            "shot_members": [member] * len(shots),
            "window_z": torch.stack([R.first_window(s["carriers"], s["action"], s["end"])["z"] for s in shots]),
            "window_action": torch.stack([s["action"] for s in shots]),
            "window_length": torch.stack([R.first_window(s["carriers"], s["action"], s["end"])["length"] for s in shots]),
            "cells": h.dap.policy_cells({key: e for key, e in entries.items() if key[0] == member}, {member: anchor})}
    model = load_world(ZERO, SEEDS[0])
    labels = {}
    with h.harness.deterministic_scoring():
        policy = R.new_policy(h.PAIRS, SEEDS[0], device=DEVICE)
        for kind in KINDS:
            for round_index in (0, 1):
                t0 = time.monotonic()
                rows, stats = label_round(kind, model, policy, pool, round_index)
                elapsed = time.monotonic() - t0
                count = sum(len(u["labels"]) for u in rows)
                labels[f"{kind}-{round_index}"] = {
                    "rows": count, "label_counts": torch.bincount(torch.cat([u["labels"] for u in rows]),
                                                                  minlength=len(PAIR_NAMES)).tolist(),
                    "seconds": elapsed}
                per = len(shots) if kind == "pe" else len(pool["cells"])
                timings[f"{kind}_round{round_index}_seconds_per_{'window' if kind == 'pe' else 'cell'}"] = elapsed / per
                if round_index == 0:
                    t0 = time.monotonic()
                    R.distil_round(policy, rows, 0, seed=SEEDS[0], steps=60, device=DEVICE)
                    timings[f"{kind}_distil_seconds_per_step"] = (time.monotonic() - t0) / 60
        t0 = time.monotonic()
        evaluation = in_sample(model, policy, pool)
        timings["in_sample_seconds_per_cell"] = (time.monotonic() - t0) / max(1, len(pool["cells"]))
    checks["normal_labels_finite"] = all(v["rows"] > 0 for v in labels.values())

    # 5. few-shot pool slice (level SMOKE_FS_LEVEL, few-shot world): first two levels' windows and cells
    fs_level = SMOKE_FS_LEVEL
    fs_branches = fewshot_branches(fs_level)
    with_cells = sorted({b["level"] for b in fs_branches if set(bool(x["pig_removed"]) for x in fs_branches
                                                                  if x["level"] == b["level"]) == {True, False}})
    chosen = with_cells[:2]
    condition = f"few-shot-{fs_level}"
    fs_model = load_world(condition, SEEDS[0])
    subset = [b for b in fs_branches if b["level"] in chosen]
    full = {b["key"]: nc.full_carriers(NC, condition, b["key"]).cpu() for b in subset}
    anchors = {}
    for b in sorted(subset, key=lambda b: b["suffix"]):
        anchors.setdefault(b["level"], full[b["key"]][0].clone())
    windows = [R.first_window(full[b["key"]], nc.read_label(NC, b["key"])["action"], len(full[b["key"]]) - 1)
               for b in subset]
    fs_pool = {"levels": chosen, "anchors": anchors, "shot_keys": [b["key"] for b in subset],
               "shot_members": [b["level"] for b in subset],
               "window_z": torch.stack([w["z"] for w in windows]), "window_action": torch.stack([w["action"] for w in windows]),
               "window_length": torch.stack([w["length"] for w in windows]), "cells": fewshot_cells(subset, anchors)}
    fs_labels = {}
    with h.harness.deterministic_scoring():
        fs_policy = R.new_policy(h.PAIRS, SEEDS[0], device=DEVICE)
        for kind in KINDS:
            for round_index in (0, 1):
                try:
                    rows, _ = label_round(kind, fs_model, fs_policy, fs_pool, round_index)
                    fs_labels[f"{kind}-{round_index}"] = {"rows": sum(len(u["labels"]) for u in rows), "finite": True}
                except ValueError as error:
                    fs_labels[f"{kind}-{round_index}"] = {"finite": False, "error": str(error)}

    # 6. scoring path: fixed request equals the #110 rollout op; joint runs (fit cell, few-shot world)
    cell = fs_pool["cells"][0]
    pair_index = PAIR_NAMES.index("15-macro")
    with h.harness.deterministic_scoring():
        z0 = cell["z0"].to(DEVICE)
        actions = cell["actions"].to(DEVICE)
        fixed, _ = R.rollout_endpoints(fs_model, None, z0, actions, {"kind": "fixed", "pair_index": pair_index},
                                       select=h.t96.select_pairs)
        reference_end = nc.rollout_batch(fs_model, z0[None].expand(len(actions), -1).clone(), actions,
                                         h.PAIRS[pair_index])[225]
        t0 = time.monotonic()
        _, timing = R.rollout_endpoints(fs_model, fs_policy, z0, actions, {"kind": "joint"},
                                        select=h.t96.select_pairs, record_schedules=True)
        timings["joint_block_seconds"] = time.monotonic() - t0
        t0 = time.monotonic()
        R.rollout_endpoints(fs_model, None, z0, actions, {"kind": "fixed", "pair_index": 0}, select=h.t96.select_pairs)
        timings["fixed_1_block_seconds"] = time.monotonic() - t0
    checks["fixed_equals_110_rollout"] = bool(torch.equal(fixed, reference_end))

    projection_ = projection(timings, accounting)
    ok = all(v for k, v in checks.items() if k != "module_sha256") and projection_["elapsed_hours_on_4_gpus"] < 48
    evidence = {"schema": SCHEMA_SMOKE, "identity": IDENTITY, "ok": bool(ok), "at": utc_now(),
                "checks": checks, "accounting": accounting, "scope": scope,
                "prefix_replication": {"pass": checks["prefix_replication"], "samples": replication},
                "labels": labels, "in_sample": evaluation, "fewshot_labels": fs_labels,
                "fewshot_smoke": {"level": fs_level, "levels": chosen, "cells": len(fs_pool["cells"])},
                "timings": timings, "projection": projection_, "wall_seconds": time.monotonic() - began}
    write_json(output / "smoke.json", evidence)
    log(f"smoke {'passed' if ok else 'FAILED'}: {json.dumps(checks)}; projection {projection_['elapsed_hours_on_4_gpus']:.2f} h")
    return 0 if ok else 1


def projection(timings, accounting):
    """Projected GPU seconds per stage (DAgger rounds as measured on the smoke slice)."""
    pool = accounting["policy_pool"]
    cells_normal = sum(pool["mixed_cells"].values())
    shots_normal = pool["complete"]
    n1_levels_ = accounting["groups"]["n1"]["levels"]
    data_ = (shots_normal + n1_levels_) * timings["policy_shot_parse_seconds"]
    pe = lambda windows: windows * (timings["pe_round0_seconds_per_window"] + timings["pe_round1_seconds_per_window"])
    dec = lambda cells: cells * (timings["dec_round0_seconds_per_cell"] + timings["dec_round1_seconds_per_cell"])
    distil = 3600 * max(timings["pe_distil_seconds_per_step"], timings["dec_distil_seconds_per_step"])
    normal = len(SEEDS) * (pe(shots_normal) + dec(cells_normal) + 2 * distil)
    fs_shots = sum(v["shots"] for v in accounting["fewshot_pools"].values())
    fs_cells = 25 * len(novel_levels())  # upper bound: the largest level has 25 mixed fit cells
    fewshot = 2 * len(SEEDS) * (pe(fs_shots) + dec(fs_cells) + 2 * len(novel_levels()) * distil)
    blocks = sum(len(SEEDS) * len(worlds_of(g)) * len(names_of(g, "zs")) * v["levels"]
                 for g, v in accounting["groups"].items())
    scoring = blocks * max(timings["joint_block_seconds"], timings["fixed_1_block_seconds"])
    total = data_ + normal + fewshot + scoring
    return {"data_gpu_seconds": data_, "normal_training_gpu_seconds": normal, "fewshot_training_gpu_seconds": fewshot,
            "scoring_gpu_seconds": scoring, "scoring_blocks": blocks, "total_gpu_seconds": total,
            "elapsed_hours_on_4_gpus": total / 4 / 3600}


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def input_bindings(output):
    h = heavy()
    names = {
        "issue_110_handoff_107": HANDOFF, "issue_110_plan": NC / "plan.json",
        "issue_110_records_manifest": NC / "records-manifest.json", "issue_110_compute": NC / "compute.json",
        "issue_110_runner": ROOT / nc.RUNNER, "novelty_v2": ROOT / "world_model/training/novelty_v2.py",
        "novelty_labels_v2": ROOT / "world_model/data/novelty_labels_v2.py",
        "issue_110_label_failures": NC / "labels" / "failures.json",
        "request_policy_module": ROOT / MODULE, "matched_dynamics": ROOT / "world_model/training/matched_dynamics.py",
        "scoring_harness": ROOT / "world_model/planning/scoring_harness.py",
        "issue_101_runner": ROOT / h.dap.RUNNER, "issue_101_compute": DAP / "compute.json",
        "gate_b_runner": ROOT / h.gb.RUNNER, "issue_112_retrain_runner": ROOT / h.rt.RUNNER,
        "issue_109_cohort_plan": h.rt.COHORT, "issue_104_plan": h.rt.POWER_PLAN,
        "issue_104_campaign_branches": nc.CAMPAIGN / "branches.json",
        "issue_110_design_plan": nc.DESIGN / "plan.json",
        "smoke": Path(output) / "smoke.json",
    }
    for condition in (ZERO, *(f"few-shot-{k}" for k in novel_levels())):
        names[f"encoder_{condition}"] = ROOT / handoff()["checkpoints"]["encoders"][condition]["path"]
        names[f"scene_head_{condition}"] = ROOT / handoff()["checkpoints"]["scene_heads"][condition]["path"]
        for seed in SEEDS:
            names[f"predictor_{condition}_{seed}"] = ROOT / handoff()["checkpoints"]["predictors"][predictor_key(condition, seed)]["path"]
    names = {k: v for k, v in names.items() if Path(v).is_file()}
    return [{"name": name, "artifact": str(Path(path).relative_to(ROOT)), "sha256": sha256_of(path)}
            for name, path in names.items()]


def frozen_rules():
    return {
        "disposition": (
            "'request selection is a cheap adaptation surface under novelty', J_dec primary (J secondary, same rule), "
            "on the pooled novel population (every mixed evaluation level of the in-scope cells of novelty levels "
            f"{list(novel_levels())}, x 3 seeds, member = evaluation level): (a) gain = AUC(J_dec policy-only retune) - "
            f"AUC(J_dec zero-shot), delta rule (supported iff mean >= {DELTA} and lower > 0; not supported iff upper < "
            f"{DELTA}); (b) parity = AUC(J_dec policy-only) - AUC(J_dec predictor-only), non-inferiority (supported "
            f"iff lower > -{DELTA}; not supported iff upper < -{DELTA}); (c) cheap = the summed GPU seconds of the "
            "policy-only retunes (levels x seeds) are below the summed #110 few-shot hybrid predictor member training "
            "GPU seconds of the same levels and seeds. supported iff (a) and (b) are supported and (c) holds; "
            "not_supported_by_this_experiment iff (a) or (b) is not supported or (c) fails; otherwise "
            f"{STOP_TOKEN}. A failed guard turns every token into {STOP_TOKEN}"),
        "secondary": (
            "per novelty type and per cell, every AUC and contrast is DESCRIPTIVE with the same token rules; "
            "zero-shot: J / J_dec - F_hindsight (in-population best fixed request, outcome-selected), - F-15-macro "
            "(pre-declared), - no-model prior, J_dec - J; few-shot: policy - zero-shot, policy - predictor "
            "(non-inferiority), predictor - zero-shot, both - predictor, both - zero-shot, policy / predictor / both - "
            "F_hindsight of their world, - few-shot prior; wrong-anchor right - wrong (delta rule)"),
        "shares": (
            "request share = the share of rollout steps (transitions) choosing each of the nine requests, pooled over "
            "the 16 candidates of a level, per (level, seed); shift = the arm's shares on a novel level minus the "
            "same-kind normal policy's shares on the paired normal side (same pair and ordinal, zero-shot world, same "
            "seed); TV = half the L1 norm of the shift; 'moves toward' = shift interval above 0, 'moves away' = below "
            "0 (DESCRIPTIVE); every evaluation level of the in-scope cells (outcome-free). N1 references: the normal "
            "policies on the #109 evaluation split (same v2 zero-shot world) and the #101 J / J_dec records (E99 "
            "Gate-B predictor)"),
    }


def frozen_plan(output, frozen_at, smoke_evidence):
    rules = frozen_rules()
    R = policy_module()
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "issue": 107,
        "related_issues": [77, 99, 101, 104, 108, 109, 110, 111, 112],
        "role": ("granularity under novelty: frozen normal request policies on the #110 v2 worlds, few-shot policy / "
                 "predictor / both adaptation, request shifts per novelty type, scored once"),
        "frozen_at": frozen_at, "frozen_before_training_and_scoring": True,
        "validation_command": VALIDATION_COMMAND, "runner": RUNNER, "runner_sha256_at_freeze": sha256_of(ROOT / RUNNER),
        "shared_module": {"path": MODULE, "sha256_at_freeze": sha256_of(ROOT / MODULE),
                          "use": "executed as a private module with DIM = 370 (the v2 carrier); bytes unchanged"},
        "engine_seconds": 0,
        "question": ("does the request a normal-trained policy picks, and its ranking value, change on NovPhy novel "
                     "levels, and does retuning only the request policy on the few-shot data recover ranking as well "
                     "as retuning the predictor, at less compute?"),
        "scope": {"rule": ("a #110-supported cell is in scope iff it has mixed-verdict evaluation levels and is not "
                           f"rollout-error-only; every other cell is reported {STOP_TOKEN} with its reason"),
                  "table": smoke_evidence["scope"], "in_scope_cells": in_scope_cells(),
                  "novel_levels": list(novel_levels())},
        "worlds": {
            "zero-shot": "the #110 zero-shot encoder, scene head and LC hybrid predictor (normal fit sides only)",
            "few-shot-k": ("the #110 few-shot-k encoder, scene head and LC hybrid predictor (zero-shot adapted on the "
                           "level-k novel fit sides, 2000 updates) = 'retune only the predictor'"),
            "seeds": list(SEEDS), "pairing": "policy seed s uses predictor seed s",
            "anchor": ("the #110 v2 anchor rule: the world's encoder parse of shot frame 0 of the level's first complete "
                       "candidate in suffix order, scene head over the one-frame history (the #110 stored carriers; "
                       "N1 and policy-split shots parsed here with the #110 carrier rule over the whole shot, "
                       f"positions 0..min(end, {PREFIX_FRAMES - 1}) stored))"),
        },
        "policies": {
            "architecture": "Linear(376,128)-SiLU-Linear(128,9) (the #101 RequestPolicy at the v2 carrier; 242 -> 376 inputs)",
            "kinds": {"pe": f"ICLR prediction-error DP, lambda {LAMBDA['pe']} (J)",
                      "dec": f"decision-valued DP, lambda {LAMBDA['dec']} (J_dec; the #101 selection, no new sweep)"},
            "recipe": {**R.DISTIL_RECIPE, "pe_windows": f"first {WINDOW_LENGTH}-frame window of every pool shot",
                       "dec_cells": "mixed grid (and, on the policy split, angle) cells; continuation F-15-macro in round 0, "
                                    "the round-0 policy in the DAgger round"},
            "normal": ("trained from the #77 initialization on the #109 policy split (stream issue109-policy, 120 levels, "
                       "3480 shots, grid + angle mixed cells) against the zero-shot world; frozen"),
            "few_shot": ("the normal policy of the same kind and seed, retuned with the same recipe (2 DAgger rounds x 1800 "
                         "steps, batch 128, AdamW 1e-3) on the level-k novel fit sides (40 levels x 16 grid candidates; "
                         "pe: every shot's first window; dec: the mixed grid cells): P against the zero-shot world "
                         "(policy only), B against the few-shot-k world (both)"),
            "typed_failures": "a nonfinite label or loss is a typed failure: recorded, never retried; its arms are missing",
        },
        "arms": {
            "zero-shot": {"world": "zero-shot", "arms": [*FIXED_ARMS, "J (N-pe)", "J_dec (N-dec)"],
                          "controls": ["no-model prior from the normal fit sides (cost = - fit success frequency of the "
                                       "grid ordinal)", "wrong anchor"]},
            "few-shot": {"policy-only": "zero-shot world + P policies", "predictor-only": "few-shot-k world + N policies and "
                         "the nine fixed requests", "both": "few-shot-k world + B policies",
                         "controls": ["no-model prior from the level-k novel fit sides", "wrong anchor"]},
            "wrong_anchor": ("per mixed target level and seed: mean AUC of the target's verdicts against the same arm's "
                             "rollouts from every other evaluation level's anchor of the same group and world"),
        },
        "evaluation": {
            "groups": {"n1": "#109 evaluation split, grid levels (Gate-B cells), zero-shot world only (N1 reference)",
                       "L0": "normal sides of the in-scope 0/* cells (paired normal reference)",
                       "Lk": "novel sides of the in-scope level-k cells"},
            "cells": "per evaluation level: complete grid candidates in grid-ordinal order; AUC on mixed levels only",
            "cost": "tie-free (presence - 0.1 x pig-centre displacement) at the v2 pig columns 175 / (180, 181)",
            "cell_auc": "m99.view; units (level, seed); member-clustered bootstrap over levels (10000 draws, PCG64 7201)",
            "never_pooled": "zero-shot and few-shot arms are reported in separate sections; no estimate pools their rows",
        },
        "decision_rule": rules,
        "guards": {"G1_isolation": ("normal pool = the #109 policy roster (disjoint from the #109 fit/evaluation rosters); "
                                    "few-shot pools = level-k issue110-fit novel sides only; no scored level is "
                                    "training data; N1 levels are #109 evaluation levels"),
                   "G2_scoring_policy": "every record carries the #111 SCORING_POLICY",
                   "G3_chronology": "frozen_at <= first training <= last training <= first score",
                   "G4_checkpoint_binding": ("every world checkpoint equals the #110 handoff hash; every scored policy equals "
                                             "its training report; every retune starts from its normal policy's hash"),
                   "G5_prefix_parse_replication": "prefix parses equal the #110 stored carriers bit for bit (smoke)",
                   "G6_handoff_inventories": "scored inventories and verdicts equal the #110 handoff exactly",
                   "failure": f"a failed guard turns every token into {STOP_TOKEN}"},
        "prohibited": ["reading evaluation captures for training or selection", "re-scoring after publication",
                       "changing a lambda, rule or arm after the freeze", "retrying or replacing a typed failure",
                       "pooling zero-shot and few-shot rows"],
        "caps": {"gpu_seconds": GPU_CAP_SECONDS, "wall_seconds_summed_over_processes": WALL_CAP_SECONDS,
                 "projection": smoke_evidence["projection"],
                 "scope_rule": "if the projected elapsed wall exceeds 48 h on 4 GPUs: blocked report, no scope cut"},
        "disclosure": ("Known before the freeze: every #101 / Gate-B / #110 published result (#110 rollout error per cell, "
                       "few-shot level-7 F-1 divergence, identifiability), and the per-cell mixed-level counts of the "
                       "handoff. No request policy has been trained on a v2 world; no policy output on an evaluation "
                       "capture exists. Hardware: RTX 5090 instead of the #85 RTX 3090."),
        "smoke_evidence": smoke_evidence,
        "inputs": input_bindings(output),
        "claim_boundary": ("one look at the issue110-evaluation grid cells (and the #109 evaluation split as N1 reference) "
                           "for request policies trained on normal data and few-shot retuned on the novel fit sides; "
                           "tie-free cost; DESCRIPTIVE intervals; no closed-loop claim"),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-107 protocol")
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
    for name in ("records", "models", "data", "shots"):
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
    write_json(output / "records-manifest.json", record_manifest(output))
    log(f"published: {json.dumps({k: v['token'] for k, v in compute['disposition'].items()})}")
    return 0


def spot_checks(output):
    """Re-score two records and one request-share supplement record in this process (exact rows and
    schedules) and re-parse two prefix shots."""
    h = heavy()
    k = max(novel_levels(), key=lambda k: sum(mixed(m) for m in eval_levels(f"L{k}").values()))
    seed = SEEDS[0]
    out = []
    for group, world, name in ((f"L{k}", "zs", "N-dec"), (f"L{k}", "fs", "B-dec")):
        record = read_record(output, group, world, name, seed)
        with h.harness.deterministic_scoring() as scoring_policy:
            model = load_world(condition_of(world, k), seed)
            fresh = score_record(output, group, world, name, seed, model, scoring_policy)
        same = all(fresh["levels"][lv]["rows"] == record["levels"][lv]["rows"]
                   and fresh["levels"][lv].get("schedules") == record["levels"][lv].get("schedules")
                   for lv in record["levels"])
        out.append({"record": f"{group}/{world}/{name}/{seed}", "pass": bool(same)})
    group, world, name = "L8", "fs", "N-dec"  # a storm cell, few-shot world: request-share supplement
    record = read_supplement(output, group, world, name, seed)
    with h.harness.deterministic_scoring() as scoring_policy:
        model = load_world(condition_of(world, 8), seed)
        fresh = score_record(output, group, world, name, seed, model, scoring_policy,
                             store=load_supplement_eval(output, group, world), levels=supplement_levels(group),
                             schema=SCHEMA_SUPPLEMENT_RECORD)
    out.append({"supplement": f"{group}/{world}/{name}/{seed}",
                "pass": all(fresh["levels"][lv]["rows"] == record["levels"][lv]["rows"]
                            and fresh["levels"][lv]["schedules"] == record["levels"][lv]["schedules"]
                            for lv in record["levels"])})
    tasks = shot_tasks()
    with torch.no_grad():
        encoder, scene_head = zero_world_parsers()
        for task in (tasks[0], tasks[-1]):
            carriers, _, _ = parse_prefix(encoder, scene_head, task["attempt"])
            out.append({"shot": task["key"], "pass": bool(torch.equal(carriers, read_shot(output, task["key"])["carriers"]))})
    return out


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    if read_json(output / "records-manifest.json") != record_manifest(output):
        raise ValueError("raw records differ from records-manifest.json")
    if read_json(output / SUPPLEMENT_MANIFEST) != supplement_manifest(output):
        raise ValueError(f"request-share supplement files differ from {SUPPLEMENT_MANIFEST}")
    compute = compute_tables(output, plan)
    for name, text in rendered(plan, compute).items():
        if (output / name).read_bytes() != text.encode():
            raise ValueError(f"published {name} differs from the recomputation")
    spot = spot_checks(output)
    if not all(s["pass"] for s in spot):
        raise ValueError(f"re-scoring / re-parsing does not replicate the retained records: {spot}")
    log(f"validation passed ({time.monotonic() - began:.0f}s): manifest, tables byte-compared, {len(spot)} spot checks; "
        f"disposition {json.dumps({k: v['token'] for k, v in compute['disposition'].items()})}")
    return 0


def dry_run(output):
    log(f"output {Path(output)}; in-scope cells {len(in_scope_cells())}; novel levels {list(novel_levels())}")
    log(f"groups {groups()}; score tasks {len(score_tasks())}; training groups "
        f"{len(normal_groups())} + {len(fewshot_groups())}")
    return 0


MODES = {"dry-run": dry_run, "smoke": smoke, "prepare": prepare, "data": data, "assemble": assemble,
         "train-normal": train_normal, "train-fewshot": train_fewshot, "score": score,
         "share-supplement": share_supplement, "publish": publish, "validate": validate}
SHARDED = ("data", "train-normal", "train-fewshot", "score")


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
