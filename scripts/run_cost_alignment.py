"""Issue-120: endpoint-cost alignment ablation on the #104 sealed #109 evaluation split, scored once.

Binding runner module: scripts/run_cost_alignment.py
Exact validation command: python -u -m scripts.run_cost_alignment --validate

#120 asks whether the description axis's ranking advantage (macro over continuous fixed requests at matched
Delta) survives a decision cost whose verdict is not aligned with the macro predicates. This runner

1. derives four engine verdicts per executed branch from the retained native traces of the #104 campaign
   (``world_model/data/cost_verdicts.py``): (a) pig removal (control), (b) a block destroyed, (c) structure
   displacement >= 0.5 world units, (d) a bird-structure collision by frame 150; and the #113 settle flag at
   the trace's last frame (``observable_macro.shot_labels`` on the stored #113 summaries) plus the present
   pigs and blocks at that step. The fit stream is derived in the smoke (prior tables, fit prevalence); the
   evaluation stream after the freeze and before any score;
2. measures each verdict's alignment: the correlation with the settle flag at the endpoint and with the
   endpoint presence counts, over every evaluation branch (level-clustered DESCRIPTIVE intervals);
3. rolls out once, under ``deterministic_scoring()``, every candidate of every evaluation level and inventory
   from the Gate-B anchors for: the Gate-B LC hybrid (nine fixed requests, the #101 J (pe, lambda 1e-4) and
   J_dec), the #115 ``ma-ref`` predictor (nine fixed requests; the pre-declared #113-coordination arm) and the
   Gate-B wrong-anchor control (F-15-macro from the 480 fit anchors). Each rollout's frame-225 carrier and its
   last visited carrier at or before frame 150 are kept; the four model costs read the same rollouts;
4. scores every arm, the no-model action prior and the wrong-anchor control under each cost on that cost's
   mixed-verdict cells; the primary contrast is macro - continuous at matched Delta (Delta-mean) on grid,
   member-clustered DESCRIPTIVE intervals, delta = 0.02, tracked across (a)-(d) against the alignment;
5. publishes compute.json, summary.json, findings.md, alignment.svg and records-manifest.json.

Modes: --dry-run --smoke --prepare --derive --score (--workers N --worker k) --publish --validate. Zero
training, zero engine seconds.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import functools
import io
import json
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

from scripts import run_decision_aware_policy as p101
from scripts import run_decision_chain_attribution as m99
from scripts import run_decision_dynamics_controls as c2
from scripts import run_decision_dynamics_retrain as rt
from scripts import run_decision_relevant_dynamics as v1
from scripts import run_gate_b as gb
from scripts import run_lookahead_control as wlc
from scripts import run_predicate_feedback_robustness as p115
from scripts import run_tau_ad_within_checkpoint as t96
from world_model.data import cost_verdicts as cv
from world_model.planning import scoring_harness as harness
from world_model.training import request_policy as rp

ROOT = v1.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-120-cost-alignment-v1"
CAMPAIGN = rt.CAMPAIGN
EVAL_STREAM = gb.STREAM
FIT_STREAM = "issue109-fit"
FIT_VERDICTS = gb.FIT_VERDICTS
SUMMARY_ROOTS = {"evaluation": ROOT / ".local-artifacts/issue-113-temporal-macro-head-v1/summaries/evaluation",
                 "fit": ROOT / ".local-artifacts/issue-113-macro-predicates-v1/summaries/fit"}

IDENTITY = "issue-120-cost-alignment-v1"
SCHEMA_PLAN = "issue_120_cost_alignment_plan_v1"
SCHEMA_SMOKE = "issue_120_cost_alignment_smoke_v1"
SCHEMA_BRANCHES = "issue_120_branches_v1"
SCHEMA_VERDICTS = "issue_120_verdicts_v1"
SCHEMA_RECORD = "issue_120_record_v1"
SCHEMA_CARRIERS = "issue_120_carriers_v1"
SCHEMA_COMPUTE = "issue_120_compute_v1"
SCHEMA_REPORT = "issue_120_report_v1"
SCHEMA_MANIFEST = "issue_120_records_manifest_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_cost_alignment --validate"
RUNNER = "scripts/run_cost_alignment.py"
MODULE = "world_model/data/cost_verdicts.py"

DEVICE = "cuda"
SEEDS = tuple(t96.SEEDS)
INVENTORIES = gb.INVENTORIES
PRIMARY = gb.GATE_INVENTORY
COSTS = cv.COSTS
CONTROL = "a"
ALTERNATIVES = ("b", "c", "d")
FIXED_ARMS = t96.FIXED_ARMS
DELTAS = (1, 5, 15)
POLICY_ARMS = {"J": ("pe", rp.ICLR_COMPUTE_WEIGHT), "J_dec": ("dec", 0.1)}
PREDICTORS = {"gateb": (*FIXED_ARMS, *POLICY_ARMS), "ma-ref": FIXED_ARMS}
WRONG = "wrong-anchor"
WRONG_ARM = "F-15-macro"  # the Gate-B unit's request (Gate-B wrong-anchor rule)
MIN_MIXED_GRID = 42       # the #96 grid mixed-cell count (#120 Work 4)
DELTA = v1.DELTA
STOP_TOKEN = v1.STOP_TOKEN
NONFINITE_SHARE_MAX = v1.NONFINITE_SHARE_MAX
SCAN_WORKERS = 44
GPU_CAP_SECONDS = 12 * 3600.0
WALL_CAP_SECONDS = 48 * 3600.0
SMOKE_CELLS = 3
SMOKE_TRACES = 12
SPOT_BRANCHES = 24

read_json = v1.read_json
write_json = v1.write_json
json_text = v1.json_text
sha256_of = v1.sha256_of
atomic_torch = v1.atomic_torch
fmt_interval = v1.fmt_interval


def log(message):
    print(f"[issue-120] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_compact(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


# ---------------------------------------------------------------------------
# paths and ledger
# ---------------------------------------------------------------------------

def branches_path(output, split):
    """Derived branch quantities: the fit stream under fit/ (smoke, pre-freeze), the evaluation stream under
    data/ (after the freeze, before any score)."""
    return Path(output) / ("fit" if split == "fit" else "data") / f"branches-{split}.json"


def task_name(predictor, arm, seed):
    return f"{predictor}--{arm}--seed{seed}"


def record_path(output, predictor, arm, seed):
    return Path(output) / "records" / f"{task_name(predictor, arm, seed)}.json"


def carriers_path(output, predictor, arm, seed):
    return Path(output) / "carriers" / f"{task_name(predictor, arm, seed)}.pt"


def ledger_path(output, phase):
    return Path(output) / "ledger" / f"{phase}.json"


def ledger_phases(output):
    root = Path(output) / "ledger"
    return {p.stem: read_json(p) for p in sorted(root.glob("*.json"))} if root.is_dir() else {}


def mark(output, phase, wall, gpu, **extra):
    write_json(ledger_path(output, phase), {"phase": phase, "wall_seconds": wall, "gpu_seconds": gpu,
                                            "at": utc_now(), **extra})
    phases = ledger_phases(output)
    if (sum(b["gpu_seconds"] for b in phases.values()) > GPU_CAP_SECONDS
            or sum(b["wall_seconds"] for b in phases.values()) > WALL_CAP_SECONDS):
        raise ValueError(f"{STOP_TOKEN}: compute cap exceeded")


# ---------------------------------------------------------------------------
# branches: the evaluation stream (verdicts, scoring) and the fit stream (priors)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def fit_branches():
    """The frozen fit-split executions (the retrain's fit-verdicts.json: key, level, action, inventories,
    campaign pig_removed)."""
    return read_json(FIT_VERDICTS)["branches"]


def stream_jobs(split):
    """(key, attempt, summary path, campaign pig_removed) of every complete branch of the split."""
    if split == "evaluation":
        rows = gb.stream_rows()
        return [(b["key"], b["attempt"], str(SUMMARY_ROOTS[split] / f"{b['key']}.json.gz"),
                 bool(rows[b["key"]]["pig_removed"]))
                for b in gb.complete_branches(gb.evaluation_branches())]
    return [(b["key"], str(CAMPAIGN / FIT_STREAM / "attempts" / b["key"]),
             str(SUMMARY_ROOTS[split] / f"{b['key']}.json.gz"), bool(b["pig_removed"])) for b in fit_branches()]


def derive_branches(output, split, workers=SCAN_WORKERS):
    """Scan every complete branch of the split once (process pool, fork) and write its quantities, the four
    verdicts, the #113 settle flag at the trace's last frame and the summary hash."""
    jobs = stream_jobs(split)
    began = time.monotonic()
    out, cpu = {}, 0.0
    context = multiprocessing.get_context("fork")
    with context.Pool(workers) as pool:
        for n, (key, record, seconds) in enumerate(pool.imap_unordered(
                cv.record_worker, [(k, a, s) for k, a, s, _ in jobs], chunksize=4), 1):
            out[key] = record
            cpu += seconds
            if n % 1000 == 0:
                log(f"{split}: {n}/{len(jobs)} branches scanned ({time.monotonic() - began:.0f}s)")
    campaign = {k: p for k, _, _, p in jobs}
    branches = {}
    for key in sorted(out):
        record = out[key]
        branches[key] = {**record, "pig_removed": campaign[key],
                         "verdicts": cv.verdicts(record["quantities"], campaign[key])}
    wall = time.monotonic() - began
    payload = {"schema": SCHEMA_BRANCHES, "plan_identity": IDENTITY, "split": split,
               "stream": EVAL_STREAM if split == "evaluation" else FIT_STREAM, "derived_at": utc_now(),
               "module_sha256": sha256_of(ROOT / MODULE), "definitions": cv.DEFINITIONS,
               "branches": branches, "wall_seconds": wall, "cpu_process_seconds": cpu, "workers": workers}
    write_compact(branches_path(output, split), payload)
    return payload


def load_branches(output, split):
    payload = read_json(branches_path(output, split))
    if payload.get("schema") != SCHEMA_BRANCHES or payload.get("plan_identity") != IDENTITY \
            or payload.get("split") != split:
        raise ValueError(f"branch file of {split} binding differs")
    return payload


def verdict_table(payload):
    return {key: b["verdicts"] for key, b in payload["branches"].items()}


# ---------------------------------------------------------------------------
# cells: per level x inventory, the complete candidates; mixed under each cost
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def evaluation_items():
    """inventory -> level -> the level's complete candidates in ordinal order (Gate-B rule; verdict-free)."""
    _, cells = gb.evaluation_set()
    return {i: {m: e["inventory"] for m, e in cells[i]["all"].items()} for i in INVENTORIES}


def cost_entries(items, verdicts, cost):
    """level -> {inventory items, verdicts by ordinal} under one cost."""
    return {member: {"inventory": entry, "verdicts": {item["ordinal"]: bool(verdicts[item["key"]][cost])
                                                      for item in entry}}
            for member, entry in items.items()}


def mixed(entries):
    return {member: entry for member, entry in entries.items() if t96.mixed(entry)}


def fit_items():
    """inventory -> fit level -> its executions in ordinal order (fit-verdicts.json)."""
    out = defaultdict(lambda: defaultdict(list))
    for b in fit_branches():
        for entry in b["inventory_entries"]:
            out[entry["inventory"]][b["level"]].append({"ordinal": entry["ordinal"], "action": b["action"],
                                                         "key": b["key"]})
    return {i: {m: sorted(v, key=lambda item: item["ordinal"]) for m, v in sorted(levels.items())}
            for i, levels in sorted(out.items())}


def mixed_counts(items_by_inventory, verdicts):
    return {cost: {i: len(mixed(cost_entries(items, verdicts, cost))) for i, items in items_by_inventory.items()}
            for cost in COSTS}


# ---------------------------------------------------------------------------
# alignment: verdict vs the #113 settle flag at the endpoint and the endpoint presence counts
# ---------------------------------------------------------------------------

def level_of(key):
    return key.rsplit("-", 1)[0]


def alignment_columns(payload):
    keys = sorted(payload["branches"])
    b = payload["branches"]
    columns = {f"verdict_{c}": np.array([float(b[k]["verdicts"][c]) for k in keys]) for c in COSTS}
    columns["settle_end"] = np.array([float(b[k]["settle_end"]) for k in keys])
    present = [b[k]["quantities"]["present_end"] for k in keys]
    columns["present_pigs"] = np.array([float(p["pig"]) for p in present])
    columns["present_blocks"] = np.array([float(p["block"]) for p in present])
    columns["present_total"] = columns["present_pigs"] + columns["present_blocks"]
    return keys, [level_of(k) for k in keys], columns


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def demeaned(values, clusters):
    values = np.asarray(values, float)
    groups = defaultdict(list)
    for n, c in enumerate(clusters):
        groups[c].append(n)
    out = values.copy()
    for rows in groups.values():
        out[rows] -= values[rows].mean()
    return out


def correlation_block(x, y, clusters):
    """Pooled Pearson r with a level-clustered percentile bootstrap (t96 draws, generator and quantiles),
    DESCRIPTIVE."""
    estimate = pearson(x, y)
    if estimate is None:
        return None
    order = sorted(set(clusters))
    index = {c: n for n, c in enumerate(order)}
    stats = np.zeros((len(order), 6))
    x, y = np.asarray(x, float), np.asarray(y, float)
    for xi, yi, c in zip(x, y, clusters):
        stats[index[c]] += (1.0, xi, yi, xi * xi, yi * yi, xi * yi)
    generator = np.random.default_rng(np.random.PCG64(t96.BOOTSTRAP_SEED))
    draws = generator.integers(0, len(order), size=(t96.BOOTSTRAP_DRAWS, len(order)))
    totals = stats[draws].sum(axis=1)
    n, sx, sy, sxx, syy, sxy = totals.T
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (n * sxy - sx * sy) / np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    r = r[np.isfinite(r)]
    low, high = np.quantile(r, t96.INTERVAL_QUANTILES)
    return {"mean": estimate, "interval": [float(low), float(high)], "clusters": len(order), "units": len(x),
            "draws_defined": int(len(r)), "label": "DESCRIPTIVE"}


ALIGNMENT_TARGETS = ("settle_end", "present_total", "present_pigs", "present_blocks", "verdict_a")


def alignment_table(payload):
    keys, clusters, columns = alignment_columns(payload)
    out = {}
    for cost in COSTS:
        v = columns[f"verdict_{cost}"]
        block = {"branches": len(keys), "prevalence": float(v.mean()), "pooled": {}, "within_level": {}}
        for target in ALIGNMENT_TARGETS:
            block["pooled"][target] = correlation_block(v, columns[target], clusters)
            block["within_level"][target] = pearson(demeaned(v, clusters), demeaned(columns[target], clusters))
        settle = block["pooled"]["settle_end"]
        block["alignment"] = None if settle is None else abs(settle["mean"])
        out[cost] = block
    return {"costs": out, "settle_end_prevalence": float(columns["settle_end"].mean()),
            "primary": "alignment = |pooled Pearson r (phi) between the verdict and the #113 settle flag at the endpoint|"}


# ---------------------------------------------------------------------------
# model costs (one rollout, four readouts)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def objective():
    return wlc.load_objective()


@functools.lru_cache(maxsize=None)
def endpoint_costs():
    return harness.EndpointCosts(objective())


def slot_columns(slot):
    base = harness.CARRIER_HEADER + harness.SLOT_WIDTH * slot
    return (base + harness.PRESENCE_FEATURE, base + harness.CENTER_FEATURES[0], base + harness.CENTER_FEATURES[1])


def slot_displacement(z, z0, slot):
    _, cx, cy = slot_columns(slot)
    return math.hypot(float(z[cx]) - float(z0[cx]), float(z[cy]) - float(z0[cy]))


def cost_row(ordinal, z_end, z_early, early_frame, z0):
    """The four model costs of one candidate (lower = the cost's goal predicted). (a) is the #111 tie-free
    primary (``EndpointCosts.row``); (b) block presence - 0.1 x block displacement; (c) minus the block and pig
    displacement at frame 225; (d) minus the block and pig displacement at the last visited frame <= 150.
    A nonfinite carrier is an excluded row for the costs that read it (value None)."""
    base = endpoint_costs().row(ordinal, z_end, z0)
    obj = objective()
    structure = (*obj.block_slots, *obj.pig_slots)
    row = {"ordinal": ordinal, "excluded": base["excluded"], "early_frame": int(early_frame)}
    if base["excluded"]:
        row.update({c: None for c in ("a", "b", "c")}, pig_presence=None, pig_displacement=None,
                   block_presence=None, block_displacement=None, structure_displacement=None)
    else:
        block_presence = sum(float(z_end[slot_columns(s)[0]]) for s in obj.block_slots)
        block_displacement = sum(slot_displacement(z_end, z0, s) for s in obj.block_slots)
        structure_displacement = sum(slot_displacement(z_end, z0, s) for s in structure)
        row.update(a=base[harness.PRIMARY_COST], b=harness.tie_free(block_presence, block_displacement),
                   c=-structure_displacement, pig_presence=base["pig_presence"],
                   pig_displacement=base["pig_displacement"], block_presence=block_presence,
                   block_displacement=block_displacement, structure_displacement=structure_displacement)
    z_early = z_early.detach()
    if bool(torch.isfinite(z_early).all()):
        early = sum(slot_displacement(z_early, z0, s) for s in structure)
        row.update(d=-early, early_structure_displacement=early, early_excluded=False)
    else:
        row.update(d=None, early_structure_displacement=None, early_excluded=True)
    return row


def rollout(model, policy, z0, actions, spec, *, early=cv.EARLY_FRAMES, endpoint=rp.ENDPOINT,
            record_schedules=False):
    """``request_policy.rollout_endpoints`` op for op (select = t96.select_pairs), plus each row's carrier at its
    last visited position <= ``early`` and that position. The added reads do not touch the rollout."""
    pairs = policy.pairs if policy is not None else rp.pairs_for(model)
    count = len(actions)
    z = z0[None].to(actions.device).expand(count, -1).clone()
    alive = torch.ones(count, dtype=torch.bool, device=z.device)
    positions = torch.zeros(count, dtype=torch.long, device=z.device)
    z_early = z.clone()
    early_at = positions.clone()
    pair_macs = [rp.linear_macs(model, pair) for pair in pairs]
    transition_calls = controller_calls = macs = 0
    trace = []
    iteration = 0
    if z.is_cuda:
        torch.cuda.synchronize()
    began = time.monotonic()
    with torch.no_grad():
        while bool(alive.any()):
            index = alive.nonzero(as_tuple=True)[0]
            if spec["kind"] == "fixed":
                chosen = torch.full((len(index),), spec["pair_index"], dtype=torch.long, device=z.device)
            else:
                logits = policy(z[index], actions[index], endpoint-positions[index])
                chosen = t96.select_pairs(spec["kind"], logits, spec)
                controller_calls += len(index)
            if record_schedules:
                trace.append((index.cpu(), chosen.cpu()))
            z_next = z
            for pair_index, pair in enumerate(pairs):
                rows = index[chosen == pair_index]
                if not len(rows):
                    continue
                z_next = z_next.clone()
                z_next[rows] = model.carrier(z[rows], actions[rows], pair)
                transition_calls += len(rows)
                macs += len(rows)*pair_macs[pair_index]
                positions[rows] += pair.delta
            z = z_next
            reached = positions <= early
            z_early = torch.where(reached[:, None], z, z_early)
            early_at = torch.where(reached, positions, early_at)
            alive = alive & torch.isfinite(z).all(dim=1) & (positions < endpoint)
            iteration += 1
    if z.is_cuda:
        torch.cuda.synchronize()
    macs += controller_calls*rp.policy_macs(policy if spec["kind"] != "fixed" else None)
    timing = {"transition_calls": transition_calls, "controller_calls": controller_calls, "linear_macs": macs,
              "gpu_seconds": time.monotonic()-began}
    if record_schedules:
        sequences = [[] for _ in range(count)]
        for index, chosen in trace:
            for r, c in zip(index.tolist(), chosen.tolist()):
                sequences[r].append(c)
        timing["schedules"] = sequences
    return z, z_early, early_at, timing


# ---------------------------------------------------------------------------
# predictors and policies (hash-checked)
# ---------------------------------------------------------------------------

def predictor_path(predictor, seed):
    return gb.member_path("unit", seed) if predictor == "gateb" else p115.member_path(p115.OUTPUT, "ma-ref", seed)


def expected_predictor_sha(predictor, seed):
    if predictor == "gateb":
        return next(c["sha256"] for c in gb.checkpoints("unit") if c["seed"] == seed)
    return p115.member_state(p115.OUTPUT, "ma-ref", seed)["sha256"]


def load_predictor(predictor, seed):
    """(model, typed failure); a hash mismatch raises (gb.load_model / p115.load_predictor)."""
    if predictor == "gateb":
        return gb.load_model("unit", seed)
    return p115.load_predictor(p115.OUTPUT, "ma-ref", seed)


def policy_file(arm, seed):
    kind, lam = POLICY_ARMS[arm]
    return p101.policy_path(p101.OUTPUT, kind, lam, seed)


def p101_arm(arm):
    return arm if arm in FIXED_ARMS else p101.policy_arm(*POLICY_ARMS[arm])


def expected_policy_sha(arm, seed):
    """The policy hash the #101 record of that arm and seed carries."""
    return read_json(p101.record_path(p101.OUTPUT, p101_arm(arm), seed))["policy"]["sha256"]


# ---------------------------------------------------------------------------
# scoring (deterministic_scoring, processes that did not train)
# ---------------------------------------------------------------------------

def score_block(model, policy, spec, z0, items, schedules):
    actions = m99.action_batch(items)
    z_end, z_early, early_at, timing = rollout(model, policy, z0.to(DEVICE), actions, spec,
                                               record_schedules=schedules)
    z_end, z_early, early_at = z_end.cpu(), z_early.cpu(), early_at.cpu()
    z0 = z0.cpu()
    out = {"rows": [cost_row(item["ordinal"], z_end[k], z_early[k], int(early_at[k]), z0)
                    for k, item in enumerate(items)],
           "candidates": len(items), **{k: v for k, v in timing.items() if k != "schedules"}}
    if schedules:
        out["schedules"] = {str(item["ordinal"]): t96.rle(timing["schedules"][k]) for k, item in enumerate(items)}
    return out, (z_end, z_early, early_at)


def arm_spec(arm):
    return t96.arm_spec(arm) if arm in FIXED_ARMS else {"kind": "joint"}


def score_task(output, anchors, predictor, arm, seed, policy_state):
    """One record: every level x inventory of the evaluation split (frozen order: inventory, level) for an arm,
    or the wrong-anchor control (F-15-macro from every fit anchor over the 16 grid candidates)."""
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "scoring_policy": policy_state,
            "scored_at": utc_now(), "predictor": predictor, "arm": arm, "seed": seed, "engine_seconds": 0}
    began = time.monotonic()
    model, failure = load_predictor(predictor, seed)
    if model is None:
        return {**base, "typed_failure": failure, "inventories": {}, "wrong_anchor": None}, None
    policy, schedules, extra = None, False, {}
    if arm in POLICY_ARMS:
        kind, lam = POLICY_ARMS[arm]
        policy = p101.load_policy(p101.OUTPUT, kind, lam, seed)
        schedules = True
        extra = {"policy": {"kind": kind, "lambda": lam, "arm_101": p101_arm(arm),
                            "sha256": sha256_of(policy_file(arm, seed))}}
    order, z_ends, z_earlies, early_ats = [], [], [], []
    inventories, wrong = {}, None
    if arm == WRONG:
        items = gb.inventory_items(PRIMARY)
        spec = t96.arm_spec(WRONG_ARM)
        wrong = {}
        for member, z0 in sorted(gb.fit_anchors().items()):
            wrong[member], carriers = score_block(model, None, spec, z0, items, False)
            order.append(["wrong", member])
            for store, value in zip((z_ends, z_earlies, early_ats), carriers):
                store.append(value)
        extra["request"] = WRONG_ARM
    else:
        spec = arm_spec(arm)
        for inventory in INVENTORIES:
            inventories[inventory] = {}
            for member, items in sorted(evaluation_items()[inventory].items()):
                inventories[inventory][member], carriers = score_block(model, policy, spec, anchors[member], items,
                                                                       schedules)
                order.append([inventory, member])
                for store, value in zip((z_ends, z_earlies, early_ats), carriers):
                    store.append(value)
    torch.cuda.synchronize()
    record = {**base, **extra, "typed_failure": None, "predictor_sha256": sha256_of(predictor_path(predictor, seed)),
              "inventories": inventories, "wrong_anchor": wrong, "seconds": time.monotonic() - began}
    carriers = {"schema": SCHEMA_CARRIERS, "plan_identity": IDENTITY, "order": order,
                "z_end": torch.cat(z_ends), "z_early": torch.cat(z_earlies), "early_frame": torch.cat(early_ats)}
    return record, carriers


def score_tasks():
    """(predictor, arm, seed), heaviest first (Delta = 1 requests and the policies), then the controls."""
    weight = {arm: (0 if arm in POLICY_ARMS or arm.startswith("F-1-") else 1 if arm.startswith("F-5-") else 2)
              for arms in PREDICTORS.values() for arm in arms}
    tasks = [(p, a, s) for p, arms in PREDICTORS.items() for a in arms for s in SEEDS]
    tasks.sort(key=lambda t: (weight[t[1]], list(PREDICTORS).index(t[0]), t[1], t[2]))
    return tasks + [("gateb", WRONG, s) for s in SEEDS]


def score(output, workers, worker):
    output = Path(output)
    plan = load_plan(output)
    evaluation = load_branches(output, "evaluation")
    if not evaluation["derived_at"] >= plan["frozen_at"]:
        raise ValueError("the evaluation verdicts were not derived after the freeze")
    anchors = gb.load_anchors(gb.OUTPUT)["anchors"]
    for predictor, arm, seed in score_tasks()[worker::workers]:
        if record_path(output, predictor, arm, seed).is_file():
            continue
        t0 = time.monotonic()
        with harness.deterministic_scoring() as policy_state:
            record, carriers = score_task(output, anchors, predictor, arm, seed, policy_state)
        if carriers is not None:
            atomic_torch(carriers_path(output, predictor, arm, seed), carriers)
        write_compact(record_path(output, predictor, arm, seed), record)
        torch.cuda.empty_cache()
        wall = time.monotonic() - t0
        mark(output, f"score-{task_name(predictor, arm, seed)}", wall, wall, worker=worker)
        log(f"worker {worker}: scored {task_name(predictor, arm, seed)} ({wall:.0f}s)"
            + (f"; typed failure {record['typed_failure']}" if record["typed_failure"] else ""))
    return 0


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def read_record(output, predictor, arm, seed):
    record = read_json(record_path(output, predictor, arm, seed))
    if record.get("schema") != SCHEMA_RECORD or record.get("plan_identity") != IDENTITY \
            or (record["predictor"], record["arm"], record["seed"]) != (predictor, arm, seed):
        raise ValueError(f"record {task_name(predictor, arm, seed)} binding differs")
    return record


def load_records(output):
    records = {(p, a, s): read_record(output, p, a, s) for p, arms in PREDICTORS.items() for a in arms for s in SEEDS}
    records.update({("gateb", WRONG, s): read_record(output, "gateb", WRONG, s) for s in SEEDS})
    return records


def view_key(predictor, arm):
    return arm if predictor == "gateb" else f"{predictor}:{arm}"


VIEW_KEYS = tuple(view_key(p, a) for p, arms in PREDICTORS.items() for a in arms)


def action_key(action):
    return gb.action_key(action)


def prior_stats(fit_verdicts, inventory, cost):
    """Engine success counts under the cost of each action over the fit executions in the inventory."""
    stats = {}
    for b in fit_branches():
        if any(e["inventory"] == inventory for e in b["inventory_entries"]):
            s = stats.setdefault(action_key(b["action"]), [0, 0])
            s[0] += int(fit_verdicts[b["key"]][cost])
            s[1] += 1
    return stats


def prior_rows(entry, stats, cost):
    """Gate-B rule under the cost: minus the fit success frequency of the identical action; unseen costs 0."""
    return [{"ordinal": item["ordinal"], "excluded": False,
             cost: -(stats[action_key(item["action"])][0] / stats[action_key(item["action"])][1])
             if action_key(item["action"]) in stats else 0.0} for item in entry["inventory"]]


def view_rows(records, cells, inventory, cost, stats):
    rows = []
    for member, entry in sorted(cells.items()):
        prior = m99.view(entry, prior_rows(entry, stats, cost), cost)
        for seed in SEEDS:
            views = {}
            for predictor, arms in PREDICTORS.items():
                for arm in arms:
                    record = records[(predictor, arm, seed)]
                    found = None if record["typed_failure"] else record["inventories"][inventory][member]["rows"]
                    views[view_key(predictor, arm)] = v1.cost_view(entry, found, cost)
            views["prior"] = prior
            rows.append({"member": member, "seed": seed, "views": views})
    return rows


def macro_minus_continuous(rows, predictor="gateb"):
    """The primary contrast: mean over Delta of (F-Delta-macro - F-Delta-continuous), paired per (level, seed);
    plus each Delta."""
    key = functools.partial(view_key, predictor)
    n = len(DELTAS)
    pieces = [(w / n, key(f"F-{d}-{alpha}")) for d in DELTAS for w, alpha in ((1, "macro"), (-1, "continuous"))]
    out = {"delta_mean": m99.paired(rows, pieces)}
    for d in DELTAS:
        out[f"delta_{d}"] = m99.paired(rows, [(1, key(f"F-{d}-macro")), (-1, key(f"F-{d}-continuous"))])
    for block in out.values():
        block["token"] = c2.token(block["estimate"])
    return out


def wrong_anchor_rows(records, cells, cost):
    rows = []
    for member, entry in sorted(cells.items()):
        for seed in SEEDS:
            right_record, wrong_record = records[("gateb", WRONG_ARM, seed)], records[("gateb", WRONG, seed)]
            if right_record["typed_failure"] or wrong_record["typed_failure"]:
                rows.append({"member": member, "seed": seed, "views": {"unit": None}})
                continue
            right = m99.view(entry, right_record["inventories"][PRIMARY][member]["rows"], cost)["auc"]
            wrong = [m99.view(entry, w["rows"], cost)["auc"] for _, w in sorted(wrong_record["wrong_anchor"].items())]
            wrong = [a for a in wrong if a is not None]
            rows.append({"member": member, "seed": seed, "views": {"unit": {
                "right": right, "wrong": float(np.mean(wrong)) if wrong else None}}})
    return rows


def wrong_anchor_table(rows):
    block = c2.difference_block(rows, [(1, "unit", "right"), (-1, "unit", "wrong")])
    return {"arm": f"gateb {WRONG_ARM}", "cells": len(rows), "right": c2.field_block(rows, "unit", "right"),
            "wrong": c2.field_block(rows, "unit", "wrong"), "right_minus_wrong": block, "token": c2.token(block)}


def best_fixed(rankers, prefix=""):
    means = {arm: (rankers[prefix + arm]["auc"] or {}).get("mean") for arm in FIXED_ARMS}
    order = sorted(FIXED_ARMS, key=lambda a: (-(means[a] if means[a] is not None else -math.inf), FIXED_ARMS.index(a)))
    return {"best": order[0], "order": order, "auc": means,
            "best_macro": next(a for a in order if a.endswith("-macro")),
            "macro_rank": 1 + next(n for n, a in enumerate(order) if a.endswith("-macro"))}


def inventory_table(records, cells, inventory, cost, stats):
    rows = view_rows(records, cells, inventory, cost, stats)
    rankers = {key: v1.ranker_block(rows, key) for key in (*VIEW_KEYS, "prior")}
    return {"cells": len(rows), "members": len(cells), "rankers": rankers,
            "macro_minus_continuous": {p: macro_minus_continuous(rows, p) for p in PREDICTORS},
            "best_fixed": {p: best_fixed(rankers, "" if p == "gateb" else f"{p}:") for p in PREDICTORS},
            "J_minus_prior": {**m99.paired(rows, [(1, "J"), (-1, "prior")])},
            "_rows": rows}


NOT_ALIGNED_MAX = 0.1  # |r| below 0.1: a negligible correlation (Cohen's "small" begins at 0.1)
DISPOSITION_RULES = {
    "absolute": (f"primary: a cost is not aligned iff its alignment (|r| with the settle flag at the endpoint) is "
                 f"below {NOT_ALIGNED_MAX}; c* = the least-aligned such cost in {{b, c, d}} that passes the "
                 "mixed-cell guard"),
    "relative": ("sensitivity: eligible iff the alignment (settle flag) is below the control's; c* = the least "
                 "aligned eligible cost"),
    "combined": ("sensitivity: alignment = max(|r settle flag|, |r presence count|); eligible iff below the control's; "
                 "c* = the least aligned eligible cost"),
}


def disposition(per_cost, guards_pass, rule="absolute"):
    """The pre-declared disposition on "the macro ranking advantage persists under a verdict not aligned with
    the macro predicates". ``per_cost[c]`` holds ``guard_pass`` (>= 42 mixed grid cells), ``alignment``
    (|r| with the settle flag at the endpoint), ``alignment_combined`` (the larger of that and |r| with the
    endpoint presence count) and ``token`` (delta rule on the grid Delta-mean macro - continuous difference).
    ``rule`` picks the eligibility of the non-aligned cost (``DISPOSITION_RULES``; "absolute" is primary)."""
    if not guards_pass:
        return {"rule": rule, "token": STOP_TOKEN, "least_aligned": None, "reason": "a guard failed"}
    control = per_cost[CONTROL]
    if not control["guard_pass"] or control["alignment"] is None:
        return {"rule": rule, "token": STOP_TOKEN, "least_aligned": None,
                "reason": "the control cost fails the mixed-cell guard"}
    field = "alignment_combined" if rule == "combined" else "alignment"
    bound = NOT_ALIGNED_MAX if rule == "absolute" else control[field]
    eligible = [c for c in ALTERNATIVES if per_cost[c]["guard_pass"] and per_cost[c][field] is not None
                and per_cost[c][field] < bound]
    if not eligible:
        return {"rule": rule, "token": STOP_TOKEN, "least_aligned": None,
                "reason": f"no alternative cost passes the mixed-cell guard with alignment below {bound:.3f}"}
    star = min(eligible, key=lambda c: (per_cost[c][field], c))
    a, s = control["token"], per_cost[star]["token"]
    if a == "supported" and s == "supported":
        token, reason = "supported", "macro - continuous >= delta (lower > 0) under the control and the least-aligned cost"
    elif a == "not_supported_by_this_experiment" or s == "not_supported_by_this_experiment":
        token = "not_supported_by_this_experiment"
        reason = ("no macro advantage under the control cost to persist (upper < delta)"
                  if a == "not_supported_by_this_experiment" else
                  "the macro advantage is absent under the least-aligned cost (upper < delta)")
    else:
        token, reason = STOP_TOKEN, "an interval straddles the margin"
    return {"rule": rule, "token": token, "least_aligned": star, "reason": reason, "control_token": a,
            "least_aligned_token": s}


def replication_guard(records, cells_a):
    """G5: under cost (a) on the Gate-B mixed cells the retained rows equal the published records exactly:
    the #101 records of every Gate-B arm (nine fixed, J, J_dec), the Gate-B unit's wrong-anchor rows, and the
    #115 ma-ref default records of the three fixed macro requests."""
    compared = mismatched = 0
    examples = []

    def compare(ours, theirs, label):
        nonlocal compared, mismatched
        theirs = {r["ordinal"]: r[harness.PRIMARY_COST] for r in theirs}
        for row in ours:
            compared += 1
            if row["ordinal"] not in theirs or row["a"] != theirs[row["ordinal"]]:
                mismatched += 1
                if len(examples) < 5:
                    examples.append(label)

    for seed in SEEDS:
        for arm in PREDICTORS["gateb"]:
            published = read_json(p101.record_path(p101.OUTPUT, p101_arm(arm), seed))["inventories"]
            for inventory in INVENTORIES:
                for member in sorted(cells_a[inventory]):
                    compare(records[("gateb", arm, seed)]["inventories"][inventory][member]["rows"],
                            published[inventory][member]["rows"], f"{arm}/{seed}/{inventory}/{member}")
        unit = read_json(gb.record_path(gb.OUTPUT, "unit", seed))
        for member, block in sorted(records[("gateb", WRONG, seed)]["wrong_anchor"].items()):
            compare(block["rows"], unit["wrong_anchor"][member]["rows"], f"wrong/{seed}/{member}")
        for d in DELTAS:
            arm = f"F-{d}-macro"
            path = p115.OUTPUT / "records" / f"ma-ref--default--{arm}--seed{seed}.json"
            published = read_json(path)["inventories"]
            for inventory in INVENTORIES:
                for member in sorted(cells_a[inventory]):
                    compare(records[("ma-ref", arm, seed)]["inventories"][inventory][member]["rows"],
                            published[inventory][member]["rows"], f"ma-ref {arm}/{seed}/{inventory}/{member}")
    return {"rows_compared": compared, "mismatched": mismatched, "examples": examples, "pass": mismatched == 0}


def guards_table(output, plan, records, evaluation, fit, cells_a):
    roster = {level["identity"] for level in gb.evaluation_levels()}
    fit_roster = {level["identity"] for level in rt.fit_levels()}
    keys = set(evaluation["branches"])
    jobs = stream_jobs("evaluation")
    scored_at = sorted(r["scored_at"] for r in records.values())
    channel_eval = sum(cv.pig_removed_from_events(b["quantities"]) != b["pig_removed"]
                       for b in evaluation["branches"].values())
    channel_fit = sum(cv.pig_removed_from_events(b["quantities"]) != b["pig_removed"] for b in fit["branches"].values())
    frames = sum(b["summary_frames"] != b["quantities"]["frames"]
                 for p in (evaluation, fit) for b in p["branches"].values())
    checkpoint = {task_name(p, a, s): r.get("predictor_sha256") == expected_predictor_sha(p, s)
                  for (p, a, s), r in records.items() if not r["typed_failure"]}
    policies = {task_name(p, a, s): r["policy"]["sha256"] == expected_policy_sha(a, s)
                for (p, a, s), r in records.items() if a in POLICY_ARMS and not r["typed_failure"]}
    nonfinite = {}
    for (p, a, s), r in records.items():
        if r["typed_failure"]:
            continue
        blocks = list(r["wrong_anchor"].values()) if a == WRONG else \
            [b for inv in r["inventories"].values() for b in inv.values()]
        rows = [row for b in blocks for row in b["rows"]]
        nonfinite[task_name(p, a, s)] = sum(row["excluded"] or row["early_excluded"] for row in rows) / len(rows)
    wrong_members = {m for (p, a, s), r in records.items() if a == WRONG and r["wrong_anchor"] for m in r["wrong_anchor"]}
    guards = {
        "G1_isolation": {"pass": bool(keys == {k for k, *_ in jobs}
                                      and all(Path(a).parent.parent.name == EVAL_STREAM for _, a, *_ in jobs)
                                      and {level_of(k) for k in keys} <= roster
                                      and {level_of(k) for k in fit["branches"]} <= fit_roster
                                      and wrong_members <= fit_roster)},
        "G2_scoring_policy": {"records": len(records),
                              "pass": all(r["scoring_policy"] == harness.SCORING_POLICY for r in records.values())},
        "G3_chronology": {"plan_frozen_at": plan["frozen_at"], "verdicts_derived_at": evaluation["derived_at"],
                          "first_record_scored_at": scored_at[0],
                          "pass": plan["frozen_at"] <= evaluation["derived_at"] <= scored_at[0]},
        "G4_bindings": {"predictors": sum(checkpoint.values()), "policies": sum(policies.values()),
                        "anchor_store_sha256": sha256_of(gb.anchors_path(gb.OUTPUT)),
                        "pass": all(checkpoint.values()) and all(policies.values())
                        and sha256_of(gb.anchors_path(gb.OUTPUT)) == plan["anchors"]["sha256"]},
        "G5_replication": replication_guard(records, cells_a),
        "G6_verdicts": {"evaluation_branches": len(keys), "pig_channel_mismatch_evaluation": channel_eval,
                        "pig_channel_mismatch_fit": channel_fit, "summary_frame_mismatch": frames,
                        "module_sha256": evaluation["module_sha256"],
                        "pass": channel_eval == 0 and channel_fit == 0 and frames == 0
                        and evaluation["module_sha256"] == plan["module_sha256"]
                        and fit["module_sha256"] == plan["module_sha256"]},
        "G7_nonfinite": {"max_share": max(nonfinite.values()), "limit": NONFINITE_SHARE_MAX,
                         "pass": max(nonfinite.values()) <= NONFINITE_SHARE_MAX},
    }
    return guards


def strip_rows(table):
    return {k: v for k, v in table.items() if k != "_rows"}


def compute_tables(output, plan):
    output = Path(output)
    records = load_records(output)
    evaluation = load_branches(output, "evaluation")
    fit = load_branches(output, "fit")
    verdicts, fit_verdicts = verdict_table(evaluation), verdict_table(fit)
    items = evaluation_items()
    cells = {c: {i: mixed(cost_entries(items[i], verdicts, c)) for i in INVENTORIES} for c in COSTS}
    _, gate_cells = gb.evaluation_set()
    cells_a = {i: gate_cells[i]["mixed"] for i in INVENTORIES}
    if {i: sorted(cells["a"][i]) for i in INVENTORIES} != {i: sorted(cells_a[i]) for i in INVENTORIES}:
        raise ValueError("cost (a) cells differ from the Gate-B cells")
    guards = guards_table(output, plan, records, evaluation, fit, cells_a)
    guards_pass = all(g["pass"] for g in guards.values())
    alignment = alignment_table(evaluation)
    fit_alignment = alignment_table(fit)
    tables, wrong = {}, {}
    for cost in COSTS:
        stats = {i: prior_stats(fit_verdicts, i, cost) for i in INVENTORIES}
        tables[cost] = {i: inventory_table(records, cells[cost][i], i, cost, stats[i]) for i in INVENTORIES}
        wrong[cost] = wrong_anchor_table(wrong_anchor_rows(records, cells[cost][PRIMARY], cost))
    mixed_eval = {c: {i: len(cells[c][i]) for i in INVENTORIES} for c in COSTS}
    per_cost = {}
    for cost in COSTS:
        guard_pass = mixed_eval[cost][PRIMARY] >= MIN_MIXED_GRID
        block = tables[cost][PRIMARY]["macro_minus_continuous"]["gateb"]["delta_mean"]
        token = block["token"] if (guards_pass and guard_pass) else STOP_TOKEN
        presence = alignment["costs"][cost]["pooled"]["present_total"]
        settle = alignment["costs"][cost]["alignment"]
        combined = None if settle is None else max(settle, abs(presence["mean"]) if presence else 0.0)
        per_cost[cost] = {"guard_pass": guard_pass, "mixed_grid_cells": mixed_eval[cost][PRIMARY],
                          "alignment": settle, "alignment_combined": combined, "estimate": block["estimate"],
                          "token": token}
    decision = {**disposition(per_cost, guards_pass), "rule_text": DISPOSITION_RULES["absolute"],
                "sensitivity": {rule: disposition(per_cost, guards_pass, rule) for rule in ("relative", "combined")}}

    def gated(token, cost):
        return token if guards_pass and per_cost[cost]["guard_pass"] else STOP_TOKEN

    contrasts = {cost: {i: {p: {k: {**b, "token": gated(b["token"], cost)}
                                for k, b in tables[cost][i]["macro_minus_continuous"][p].items()}
                            for p in PREDICTORS} for i in INVENTORIES} for cost in COSTS}
    arm113 = contrasts[CONTROL][PRIMARY]["ma-ref"]["delta_mean"]
    phases = ledger_phases(output)
    smoke = read_json(output / "smoke.json")
    compute_phases = {"smoke": {"wall_seconds": smoke["wall_seconds"], "gpu_seconds": smoke["gpu_seconds"],
                                "cpu_process_seconds": smoke["fit_scan"]["cpu_process_seconds"]},
                      "derive_evaluation": {"wall_seconds": evaluation["wall_seconds"], "gpu_seconds": 0.0,
                                            "cpu_process_seconds": evaluation["cpu_process_seconds"]}}
    score_wall = sum(b["wall_seconds"] for k, b in phases.items() if k.startswith("score-"))
    compute_phases["score"] = {"wall_seconds": score_wall, "gpu_seconds": score_wall, "records": len(records)}
    return {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
        "definitions": cv.DEFINITIONS,
        "thresholds": {"blocks_destroyed_min": cv.BLOCKS_DESTROYED_MIN, "displacement_min": cv.DISPLACEMENT_MIN,
                       "early_frames": cv.EARLY_FRAMES},
        "evaluation": {"branches": len(evaluation["branches"]), "levels": len(gb.evaluation_levels()),
                       "levels_per_inventory": {i: len(items[i]) for i in INVENTORIES},
                       "mixed_cells": mixed_eval, "verdicts_sha256": sha256_of(branches_path(output, "evaluation"))},
        "fit": {"branches": len(fit["branches"]), "mixed_cells": mixed_counts(fit_items(), fit_verdicts),
                "alignment": {c: {"prevalence": fit_alignment["costs"][c]["prevalence"],
                                  "alignment": fit_alignment["costs"][c]["alignment"]} for c in COSTS}},
        "alignment": alignment,
        "guards": guards, "guards_pass": guards_pass,
        "mixed_cell_guard": {c: {"mixed_grid_cells": per_cost[c]["mixed_grid_cells"], "minimum": MIN_MIXED_GRID,
                                 "pass": per_cost[c]["guard_pass"]} for c in COSTS},
        "primary": {c: {k: v for k, v in per_cost[c].items()} for c in COSTS},
        "disposition": decision,
        "arm_113": {"predictor": "ma-ref", "cost": CONTROL, "contrast": arm113,
                    "token": gated(arm113["token"], CONTROL)},
        "contrasts": contrasts,
        "inventories": {c: {i: strip_rows(tables[c][i]) for i in INVENTORIES} for c in COSTS},
        "wrong_anchor": {c: {**wrong[c], "token": gated(wrong[c]["token"], c)} for c in COSTS},
        "typed_failures": {task_name(*k): r["typed_failure"] for k, r in records.items() if r["typed_failure"]},
        "compute": {"phases": compute_phases,
                    "gpu_seconds": sum(b["gpu_seconds"] for b in compute_phases.values()),
                    "wall_seconds": sum(b["wall_seconds"] for b in compute_phases.values()),
                    "cpu_process_seconds": sum(b.get("cpu_process_seconds", 0.0) for b in compute_phases.values()),
                    "engine_seconds": 0, "hardware": "4 x RTX 5090 (one scoring process per GPU); 44 CPU scan workers"},
    }


# ---------------------------------------------------------------------------
# renderings
# ---------------------------------------------------------------------------

def num(value, digits=3):
    return "n/a" if value is None else format(value, f".{digits}f")


def signed(value, digits=3):
    return "n/a" if value is None else format(value, f"+.{digits}f")


def interval(block, digits=3):
    if block is None:
        return "n/a"
    return f"{signed(block['mean'], digits)} [{signed(block['interval'][0], digits)}, {signed(block['interval'][1], digits)}]"


def auc_cell(block, digits=3):
    if block is None:
        return "n/a"
    return f"{block['mean']:.{digits}f} [{block['interval'][0]:.{digits}f}, {block['interval'][1]:.{digits}f}]"


def figure_svg(compute):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.rcParams["svg.hashsalt"] = IDENTITY
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, target, label in ((axes[0], "settle_end", "|r| verdict vs #113 settle flag at the endpoint"),
                              (axes[1], "present_total", "|r| verdict vs endpoint presence count (pigs + blocks)")):
        for predictor, marker, colour in (("gateb", "o", "tab:blue"), ("ma-ref", "s", "tab:orange")):
            for cost in COSTS:
                r = compute["alignment"]["costs"][cost]["pooled"][target]
                block = compute["contrasts"][cost][PRIMARY][predictor]["delta_mean"]["estimate"]
                if r is None or block is None:
                    continue
                x = abs(r["mean"])
                ax.errorbar([x], [block["mean"]], yerr=[[block["mean"] - block["interval"][0]],
                                                         [block["interval"][1] - block["mean"]]],
                            fmt=marker, color=colour, capsize=3,
                            label=("Gate-B LC hybrid" if predictor == "gateb" else "#115 ma-ref (#113 predicates)")
                            if cost == "a" else None)
                ax.annotate(f"({cost})", (x, block["mean"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.axhline(0.0, color="grey", lw=0.8)
        ax.axhline(DELTA, color="grey", lw=0.8, ls="--")
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("grid AUC: macro - continuous (Delta-mean)", fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle("#120: macro - continuous fixed-request difference vs measured verdict alignment", fontsize=9)
    fig.tight_layout()
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", metadata={"Date": None})
    plt.close(fig)
    return buffer.getvalue()


def findings_md(plan, compute):
    lines = []
    add = lines.append
    al = compute["alignment"]["costs"]
    add("# Issue #120: endpoint-cost alignment ablation on the #104 sealed #109 evaluation split")
    add("")
    add(f"Frozen {plan['frozen_at']}; validation `{VALIDATION_COMMAND}`. Zero engine seconds, zero training. Intervals "
        "are level-clustered percentile bootstraps (10 000 draws, PCG64 7201), DESCRIPTIVE. Margin delta = 0.02.")
    add("")
    d = compute["disposition"]
    add(f"## Disposition: **{d['token']}**")
    add("")
    add("\"The macro ranking advantage persists under a verdict not aligned with the macro predicates\": "
        f"{d['token']} ({d['reason']}; least-aligned eligible cost: {d['least_aligned']}). Rule: {d['rule_text']}.")
    add("")
    for rule, other in d["sensitivity"].items():
        add(f"- sensitivity, {rule} rule: {other['token']} ({other['reason']}; c* = {other['least_aligned']})")
    add("")
    add("## Frozen costs and verdicts")
    add("")
    for cost in COSTS:
        add(f"- ({cost}) {cv.COST_NAMES[cost]}: {cv.DEFINITIONS[cost]}")
    add("")
    add("Mixed-verdict cells (levels) per inventory; guard: fewer than 42 mixed grid cells gives "
        f"`{STOP_TOKEN}` for that cost.")
    add("")
    add("| cost | eval prevalence | grid | offset | angle | power | guard | fit prevalence | fit grid | fit angle |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for cost in COSTS:
        m = compute["evaluation"]["mixed_cells"][cost]
        f = compute["fit"]["mixed_cells"][cost]
        add(f"| ({cost}) | {al[cost]['prevalence']:.3f} | {m['grid']} | {m['offset']} | {m['angle']} | {m['power']} | "
            f"{'pass' if compute['mixed_cell_guard'][cost]['pass'] else 'fail'} | "
            f"{compute['fit']['alignment'][cost]['prevalence']:.3f} | {f.get('grid', 0)} | {f.get('angle', 0)} |")
    add("")
    add("## Alignment (item 2; every evaluation branch)")
    add("")
    add(f"Pooled Pearson r (phi for binary pairs) over {al['a']['branches']} branches; within-level r in brackets "
        f"after demeaning by level. Settle flag at the endpoint prevalence "
        f"{compute['alignment']['settle_end_prevalence']:.3f}.")
    add("")
    add("| cost | settle flag at endpoint | presence count (pigs + blocks) | present pigs | present blocks | verdict (a) |")
    add("| --- | --- | --- | --- | --- | --- |")
    for cost in COSTS:
        cells = []
        for target in ALIGNMENT_TARGETS:
            block = al[cost]["pooled"][target]
            within = al[cost]["within_level"][target]
            cells.append(f"{interval(block)} ({signed(within)})" if block else "n/a")
        add(f"| ({cost}) | " + " | ".join(cells) + " |")
    add("")
    add("## Primary: macro - continuous at matched Delta (grid, Gate-B LC hybrid)")
    add("")
    add("| cost | alignment | Delta-mean | Delta 1 | Delta 5 | Delta 15 | token |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for cost in COSTS:
        c = compute["contrasts"][cost][PRIMARY]["gateb"]
        add(f"| ({cost}) | {num(compute['primary'][cost]['alignment'])} | {interval(c['delta_mean']['estimate'])} | "
            f"{interval(c['delta_1']['estimate'])} | {interval(c['delta_5']['estimate'])} | "
            f"{interval(c['delta_15']['estimate'])} | {c['delta_mean']['token']} |")
    add("")
    a113 = compute["arm_113"]
    add(f"#113-coordination arm (pre-declared): `ma-ref` fixed macro requests under cost (a), Delta-mean macro - "
        f"continuous {interval(a113['contrast']['estimate'])} -> {a113['token']}.")
    add("")
    add("| cost | ma-ref Delta-mean | Delta 1 | Delta 5 | Delta 15 |")
    add("| --- | --- | --- | --- | --- |")
    for cost in COSTS:
        c = compute["contrasts"][cost][PRIMARY]["ma-ref"]
        add(f"| ({cost}) | {interval(c['delta_mean']['estimate'])} | {interval(c['delta_1']['estimate'])} | "
            f"{interval(c['delta_5']['estimate'])} | {interval(c['delta_15']['estimate'])} |")
    add("")
    add("## Ranking under each cost (pooled AUC, same rollouts)")
    add("")
    for inventory in INVENTORIES:
        add(f"### {inventory}{'' if inventory == PRIMARY else ' (DESCRIPTIVE)'}")
        add("")
        add("| arm | " + " | ".join(f"({c})" for c in COSTS) + " |")
        add("| --- | " + " | ".join("---" for _ in COSTS) + " |")
        for key in (*VIEW_KEYS, "prior"):
            add(f"| {key} | " + " | ".join(auc_cell(compute["inventories"][c][inventory]["rankers"][key]["auc"])
                                          for c in COSTS) + " |")
        add("| best fixed (Gate-B) | " + " | ".join(
            f"{compute['inventories'][c][inventory]['best_fixed']['gateb']['best']} (macro rank "
            f"{compute['inventories'][c][inventory]['best_fixed']['gateb']['macro_rank']})" for c in COSTS) + " |")
        add("| macro - cont Delta-mean | " + " | ".join(
            interval(compute["contrasts"][c][inventory]["gateb"]["delta_mean"]["estimate"]) for c in COSTS) + " |")
        add("| cells (levels x 3) | " + " | ".join(str(compute["inventories"][c][inventory]["cells"]) for c in COSTS)
            + " |")
        add("")
    add("## Wrong-anchor control (grid, Gate-B F-15-macro from the 480 fit anchors)")
    add("")
    add("| cost | right | wrong | right - wrong | token |")
    add("| --- | --- | --- | --- | --- |")
    for cost in COSTS:
        w = compute["wrong_anchor"][cost]
        add(f"| ({cost}) | {auc_cell(w['right'])} | {auc_cell(w['wrong'])} | {interval(w['right_minus_wrong'])} | "
            f"{w['token']} |")
    add("")
    add("## Guards")
    add("")
    for name, guard in compute["guards"].items():
        add(f"- {name}: {'pass' if guard['pass'] else 'FAIL'}")
    add(f"- a failed guard turns every token into `{STOP_TOKEN}`; all guards pass: {compute['guards_pass']}")
    add(f"- typed failures: {compute['typed_failures'] or 'none'}")
    add("")
    add("## Compute")
    add("")
    c = compute["compute"]
    add(f"GPU s {c['gpu_seconds']:.0f}; summed process wall s {c['wall_seconds']:.0f}; CPU process s (trace scans) "
        f"{c['cpu_process_seconds']:.0f}; engine s 0. {c['hardware']}.")
    add("")
    return "\n".join(lines)


def summary(plan, compute):
    return {
        "schema": SCHEMA_REPORT, "identity": IDENTITY, "issue": 120, "frozen_at": plan["frozen_at"],
        "disposition": compute["disposition"],
        "primary": {c: {"alignment": compute["primary"][c]["alignment"],
                        "macro_minus_continuous": compute["primary"][c]["estimate"],
                        "token": compute["primary"][c]["token"],
                        "mixed_grid_cells": compute["primary"][c]["mixed_grid_cells"]} for c in COSTS},
        "arm_113": {"estimate": compute["arm_113"]["contrast"]["estimate"], "token": compute["arm_113"]["token"]},
        "mixed_cells": compute["evaluation"]["mixed_cells"],
        "guards_pass": compute["guards_pass"],
        "compute": {k: compute["compute"][k] for k in ("gpu_seconds", "wall_seconds", "cpu_process_seconds",
                                                       "engine_seconds")},
        "validation_command": VALIDATION_COMMAND,
    }


def record_manifest(output):
    output = Path(output)
    expected = [record_path(output, p, a, s) for p, arms in PREDICTORS.items() for a in arms for s in SEEDS]
    expected += [record_path(output, "gateb", WRONG, s) for s in SEEDS]
    present = sorted(output.glob("records/*.json"))
    if sorted(expected) != present:
        raise ValueError(f"records/ holds {len(present)} files, expected exactly {len(expected)}")
    files = present + sorted(output.glob("carriers/*.pt")) + [branches_path(output, "evaluation"),
                                                              branches_path(output, "fit")]
    return {"schema": SCHEMA_MANIFEST, "identity": IDENTITY, "count": len(files),
            "files": {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha256_of(p)} for p in files}}


def rendered(output, plan, compute):
    return {"compute.json": json_text(compute), "summary.json": json_text(summary(plan, compute)),
            "findings.md": findings_md(plan, compute), "alignment.svg": figure_svg(compute),
            "records-manifest.json": json_text(record_manifest(output))}


# ---------------------------------------------------------------------------
# smoke, freeze, derive
# ---------------------------------------------------------------------------

def smoke_trace_checks(keys):
    """The events tail parse equals the full chunk parse on every chunk of a few fit traces."""
    import gzip
    checked = chunks = 0
    ok = True
    for key in keys:
        root = cv.native_root(CAMPAIGN / FIT_STREAM / "attempts" / key)
        manifest = read_json(root / "native-manifest.json")
        for descriptor in manifest["chunks"]:
            with gzip.open(root / descriptor["path"], "rb") as stream:
                raw = stream.read()
            ok &= cv.chunk_events(raw) == json.loads(raw)["events"]
            chunks += 1
        checked += 1
    return {"traces": checked, "chunks": chunks, "pass": bool(ok)}


def smoke_rollouts():
    """On the first fit grid cells (fit anchors, the 16 shared grid candidates): ``rollout`` equals
    ``request_policy.rollout_endpoints`` at frame 225 for fixed requests and both policies, the early carrier of a
    fixed request equals a rollout to 150, and the ma-ref predictor carries the nine requests. Timings feed the
    projection."""
    items = gb.inventory_items(PRIMARY)
    anchors = sorted(gb.fit_anchors().items())[:SMOKE_CELLS]
    seed = SEEDS[0]
    out = {"cells": [m for m, _ in anchors], "checks": {}, "timings": {}}
    ok = True
    with harness.deterministic_scoring():
        for predictor in PREDICTORS:
            model, failure = load_predictor(predictor, seed)
            if model is None:
                raise ValueError(f"{predictor} seed {seed}: {failure}")
            pairs = rp.pairs_for(model)
            out["checks"][f"{predictor}_pairs"] = [str(p.identity) for p in pairs]
            ok &= len(pairs) == len(FIXED_ARMS)
            for arm in PREDICTORS[predictor]:
                policy = None
                if arm in POLICY_ARMS:
                    policy = p101.load_policy(p101.OUTPUT, *POLICY_ARMS[arm], seed)
                spec = arm_spec(arm)
                equal = early_equal = True
                seconds = rows = 0.0
                for member, z0 in anchors:
                    actions = m99.action_batch(items)
                    t0 = time.monotonic()
                    z, z_early, early_at, timing = rollout(model, policy, z0.to(DEVICE), actions, spec)
                    seconds += time.monotonic() - t0
                    rows += len(items)
                    ref, _ = rp.rollout_endpoints(model, policy, z0.to(DEVICE), actions, spec, select=t96.select_pairs)
                    equal &= bool(torch.equal(torch.nan_to_num(z), torch.nan_to_num(ref)))
                    if arm in FIXED_ARMS:
                        ref_early, _ = rp.rollout_endpoints(model, None, z0.to(DEVICE), actions, spec,
                                                            select=t96.select_pairs, endpoint=cv.EARLY_FRAMES)
                        early_equal &= bool(torch.equal(torch.nan_to_num(z_early), torch.nan_to_num(ref_early)))
                        early_equal &= bool((early_at == cv.EARLY_FRAMES).all())
                    else:
                        early_equal &= bool((early_at <= cv.EARLY_FRAMES).all())
                out["checks"][f"{predictor}:{arm}"] = {"endpoint_equal": equal, "early_equal": early_equal}
                out["timings"][f"{predictor}:{arm}"] = seconds / rows
                ok &= equal and early_equal
            del model
            torch.cuda.empty_cache()
    out["pass"] = bool(ok)
    return out


def projection(timings):
    candidates = sum(len(v) for i in INVENTORIES for v in evaluation_items()[i].values())
    per_task = {f"{p}:{a}": timings[f"{p}:{a}"] * candidates for p, arms in PREDICTORS.items() for a in arms}
    wrong = timings[f"gateb:{WRONG_ARM}"] * len(gb.fit_anchors()) * len(gb.inventory_items(PRIMARY))
    total = sum(per_task.values()) * len(SEEDS) + wrong * len(SEEDS)
    return {"candidates_per_record": candidates, "gpu_seconds_per_record": per_task, "wrong_anchor_record": wrong,
            "gpu_seconds_total": total, "elapsed_hours_on_4_gpus": total / 4 / 3600.0,
            "note": "batch-size effects ignored (smoke cells hold 16 candidates; evaluation cells 13-20)"}


def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json exists; the smoke runs before the freeze")
    began = time.monotonic()
    keys = [b["key"] for b in fit_branches()]
    trace = smoke_trace_checks(keys[::len(keys) // SMOKE_TRACES][:SMOKE_TRACES])
    log(f"trace checks: {trace}")
    if branches_path(output, "fit").is_file():
        fit = load_branches(output, "fit")
    else:
        fit = derive_branches(output, "fit")
    log(f"fit stream derived: {len(fit['branches'])} branches")
    fit_verdicts = verdict_table(fit)
    channel = sum(cv.pig_removed_from_events(b["quantities"]) != b["pig_removed"] for b in fit["branches"].values())
    frames = sum(b["summary_frames"] != b["quantities"]["frames"] for b in fit["branches"].values())
    fit_alignment = alignment_table(fit)
    t0 = time.monotonic()
    rollouts = smoke_rollouts()
    gpu = time.monotonic() - t0
    cost_check = smoke_cost_rows()
    evidence = {
        "schema": SCHEMA_SMOKE, "identity": IDENTITY, "smoked_at": utc_now(),
        "trace_tail_parse": trace,
        "fit_scan": {"branches": len(fit["branches"]), "sha256": sha256_of(branches_path(output, "fit")),
                     "wall_seconds": fit["wall_seconds"], "cpu_process_seconds": fit["cpu_process_seconds"],
                     "pig_channel_mismatch": channel, "summary_frame_mismatch": frames,
                     "prevalence": {c: fit_alignment["costs"][c]["prevalence"] for c in COSTS},
                     "mixed_cells": mixed_counts(fit_items(), fit_verdicts),
                     "alignment": {c: {t: (fit_alignment["costs"][c]["pooled"][t] or {}).get("mean")
                                       for t in ALIGNMENT_TARGETS} for c in COSTS}},
        "rollouts": rollouts, "cost_rows": cost_check,
        "projection": projection(rollouts["timings"]),
        "gpu_seconds": gpu, "wall_seconds": time.monotonic() - began,
    }
    evidence["ok"] = bool(trace["pass"] and channel == 0 and frames == 0 and rollouts["pass"] and cost_check["pass"]
                          and evidence["projection"]["elapsed_hours_on_4_gpus"] <= 48.0)
    write_json(output / "smoke.json", evidence)
    log(f"smoke {'passed' if evidence['ok'] else 'FAILED'}: projection "
        f"{evidence['projection']['elapsed_hours_on_4_gpus']:.2f} h on 4 GPUs; fit mixed cells "
        f"{evidence['fit_scan']['mixed_cells']}")
    return 0 if evidence["ok"] else 1


def smoke_cost_rows():
    """``cost_row`` (a) equals EndpointCosts.row's tie-free value on a fit anchor and a perturbed carrier."""
    z0 = next(iter(sorted(gb.fit_anchors().items())))[1]
    z = z0.clone()
    z[harness.CARRIER_HEADER + harness.SLOT_WIDTH * 4 + harness.CENTER_FEATURES[0]] += 0.25
    row = cost_row(0, z, z, 150, z0)
    reference = endpoint_costs().row(0, z, z0)
    return {"a_equals_tie_free": row["a"] == reference[harness.PRIMARY_COST], "c": row["c"], "d": row["d"],
            "pass": row["a"] == reference[harness.PRIMARY_COST] and abs(row["c"] + 0.25) < 1e-6
            and row["c"] == row["d"]}


def input_bindings(output):
    names = {
        "issue_108_gate_b_plan": gb.OUTPUT / "plan.json", "issue_108_gate_b_anchors": gb.anchors_path(gb.OUTPUT),
        "issue_112_retrain_handoff": gb.HANDOFF, "issue_112_fit_verdicts": FIT_VERDICTS,
        "issue_112_fit_anchors": rt.anchors_path(rt.OUTPUT),
        "issue_101_plan": p101.OUTPUT / "plan.json", "issue_101_selection": p101.selection_path(p101.OUTPUT),
        "issue_115_plan": p115.OUTPUT / "plan.json",
        "issue_109_cohort_plan": rt.COHORT, "issue_104_campaign_branches": CAMPAIGN / "branches.json",
        "issue_113_summaries_manifest": ROOT / ".local-artifacts/issue-113-macro-predicates-v1/summaries-manifest.json",
        "issue_113_label_module": ROOT / "world_model/data/observable_macro.py",
        "scoring_harness": ROOT / "world_model/planning/scoring_harness.py",
        "request_policy": ROOT / "world_model/training/request_policy.py",
        "cost_verdicts": ROOT / MODULE, "fit_branches": branches_path(output, "fit"), "smoke": Path(output) / "smoke.json",
    }
    for seed in SEEDS:
        names[f"gateb_{seed}"] = predictor_path("gateb", seed)
        names[f"ma_ref_{seed}"] = predictor_path("ma-ref", seed)
        for arm in POLICY_ARMS:
            names[f"policy_{arm}_{seed}"] = policy_file(arm, seed)
    return [{"name": name, "artifact": str(Path(path).relative_to(ROOT)), "sha256": sha256_of(path)}
            for name, path in names.items()]


def frozen_rules():
    return {
        "costs": {c: {"name": cv.COST_NAMES[c], "definition": cv.DEFINITIONS[c]} for c in COSTS},
        "thresholds": {"blocks_destroyed_min": cv.BLOCKS_DESTROYED_MIN, "displacement_min_world_units":
                       cv.DISPLACEMENT_MIN, "early_frames": cv.EARLY_FRAMES},
        "verdict_source": ("retained native traces of the executed #104 campaign branches (cost_verdicts."
                           "branch_quantities); (a) the campaign's pig_removed; the evaluation stream is derived after "
                           "the freeze and before the first score, the fit stream in the smoke (priors only)"),
        "cells": ("per level x inventory, the complete candidates in ordinal order (Gate-B rule); under each cost the "
                  "mixed-verdict cells of that cost's verdict; cost (a) cells equal the Gate-B cells"),
        "alignment": ("per cost over every complete evaluation branch: Pearson r (phi) of the verdict with (i) the #113 "
                      "settle flag at the trace's last frame (settle_1, observable_macro.shot_labels on the stored #113 "
                      "summaries) and (ii) the present pigs + blocks at the trace's last step (also each kind and "
                      "verdict (a)); level-clustered bootstrap; within-level r reported beside. The alignment of a cost "
                      "is |pooled r with the settle flag|"),
        "rollouts": ("one rollout per (predictor, arm, seed, level, inventory) from the Gate-B anchor to frame 225 "
                     "under deterministic_scoring(); the frame-225 carrier and the last visited carrier at or before "
                     "frame 150 feed the four model costs (cost_verdicts.DEFINITIONS)"),
        "arms": {"gateb": list(PREDICTORS["gateb"]), "ma-ref": list(PREDICTORS["ma-ref"]),
                 "controls": ["no-model action prior under each cost's fit verdicts (Gate-B rule)",
                              f"wrong-anchor: Gate-B {WRONG_ARM} from the 480 fit anchors over the 16 grid candidates"]},
        "primary": ("per cost, grid: mean over Delta in {1, 5, 15} of AUC(F-Delta-macro) - AUC(F-Delta-continuous) of "
                    "the Gate-B LC hybrid, paired per (level, seed), level-clustered bootstrap; token by the delta rule "
                    f"(supported iff mean >= {DELTA} and lower > 0; not_supported_by_this_experiment iff upper < "
                    f"{DELTA}; otherwise {STOP_TOKEN}); each Delta and the other inventories DESCRIPTIVE"),
        "arm_113": ("the single pre-declared #113-coordination arm: #120 item 3(a) under #113's redefined macro "
                    "predicates = the #115 ma-ref predictor's fixed requests under cost (a), same contrast and rule"),
        "mixed_cell_guard": (f"a cost with fewer than {MIN_MIXED_GRID} mixed grid cells (levels) is "
                             f"{STOP_TOKEN} for every token of that cost; it is reported, not dropped"),
        "disposition": ("\"the macro ranking advantage persists under a verdict not aligned with the macro predicates\": "
                        f"{DISPOSITION_RULES['absolute']}. supported iff the primary token is supported under (a) and "
                        "under c*; not_supported_by_this_experiment iff either is not_supported_by_this_experiment; "
                        f"otherwise (or without an eligible c*) {STOP_TOKEN}. Reported beside it, as sensitivity "
                        f"readings with the same token logic: {DISPOSITION_RULES['relative']}; "
                        f"{DISPOSITION_RULES['combined']}"),
        "guards": {"G1_isolation": "verdicts and scores from the evaluation stream only; fit only as priors; wrong "
                                   "anchors from the fit roster",
                   "G2_scoring_policy": "every record carries the #111 SCORING_POLICY",
                   "G3_chronology": "plan frozen_at <= evaluation verdicts derived_at <= first scored_at",
                   "G4_bindings": "predictor, policy and anchor-store hashes equal their published bindings",
                   "G5_replication": ("under (a) on the Gate-B cells every retained row equals the #101 record of the "
                                      "arm, the Gate-B wrong-anchor rows and the #115 ma-ref default macro records "
                                      "exactly"),
                   "G6_verdicts": ("the event-derived #85 pig channel equals the campaign's pig_removed on every "
                                   "branch; the #113 summary frame counts equal the traces'; module bytes frozen"),
                   "G7_nonfinite": f"at most {NONFINITE_SHARE_MAX:.0%} excluded rows per record",
                   "failure": f"a failed guard turns every token into {STOP_TOKEN}"},
    }


def frozen_plan(output, frozen_at, evidence):
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "issue": 120,
        "related_issues": [96, 101, 104, 108, 109, 111, 113, 115], "frozen_at": frozen_at,
        "frozen_before_evaluation_scoring": True, "validation_command": VALIDATION_COMMAND, "runner": RUNNER,
        "runner_sha256_at_freeze": sha256_of(ROOT / RUNNER), "module_sha256": sha256_of(ROOT / MODULE),
        "engine_seconds": 0, "training": "none",
        "question": ("does the macro - continuous fixed-request ranking difference survive a decision cost whose verdict "
                     "is not aligned with the macro predicates?"),
        "evaluation": {"split": f"#109 evaluation split as captured by the #104 campaign stream {EVAL_STREAM}",
                       "superseded": ("the ticket's held-out lineages 007/008/014/016 (#99/#100 era) are superseded "
                                      "by the sealed #109 evaluation split (Gate B passed; orchestrator decision)"),
                       "predictors": "Gate-B LC hybrid handoff seeds (gateb) and the #115 ma-ref members",
                       "fit_stream": "read only for the prior tables and the fit prevalence (fit/branches-fit.json)",
                       "excluded": ["issue109-policy", "issue110-*"]},
        "anchors": {"path": str(gb.anchors_path(gb.OUTPUT).relative_to(ROOT)),
                    "sha256": sha256_of(gb.anchors_path(gb.OUTPUT))},
        "rules": frozen_rules(),
        "bootstrap": {"draws": t96.BOOTSTRAP_DRAWS, "generator": f"PCG64 {t96.BOOTSTRAP_SEED}",
                      "quantiles": list(t96.INTERVAL_QUANTILES), "cluster": "level", "label": "DESCRIPTIVE"},
        "seeds": list(SEEDS), "delta": DELTA,
        "caps": {"gpu_seconds": GPU_CAP_SECONDS, "wall_seconds": WALL_CAP_SECONDS, "stop": STOP_TOKEN},
        "prohibited": ["training", "engine execution", "re-scoring after publication",
                       "changing a cost, threshold, arm, rule or guard after the freeze",
                       "retrying or replacing a typed failure"],
        "disclosure": ("Known before the freeze: Gate B, #101, #115 and #96 results under cost (a) (on this predictor "
                       "F-15-continuous 0.762 > F-15-macro 0.744 on grid, #101); the thresholds of (b)-(d) were set "
                       "from a 400-600-shot exploration of the fit stream (pig displacement was degenerate there: "
                       "0.25 % of fit shots move the pig >= 0.01, so (c) uses the summed block and pig displacement); "
                       "the full fit-stream prevalence, mixed-cell counts and alignments (smoke). The disposition's "
                       "eligibility rule was set after the smoke showed that on the fit stream every alternative "
                       "verdict correlates more with the settle flag than the control does (|r| a 0.059, b 0.096, "
                       "c 0.085, d 0.139), so a rule relative to the control would leave no eligible cost; the absolute "
                       "rule (|r| < 0.1) is primary and the relative and combined rules are reported beside it. No "
                       "evaluation verdict under (b)-(d) and no model output under any new cost had been computed. "
                       "Hardware: RTX 5090."),
        "smoke_evidence": evidence,
        "inputs": input_bindings(output),
        "claim_boundary": ("ranking AUC of fixed requests under four engine verdicts derived from the same retained "
                           "branches, on one predictor family; DESCRIPTIVE intervals; no closed-loop claim"),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY or not plan["smoke_evidence"].get("ok"):
        raise ValueError("plan.json is not a passed-smoke #120 freeze")
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
    for name in ("records", "data", "carriers"):
        if (output / name).exists():
            raise ValueError(f"{name} exists before the freeze; refusing")
    evidence = read_json(output / "smoke.json")
    if not evidence.get("ok"):
        raise ValueError("smoke failed; STOP RULE: no freeze")
    plan = frozen_plan(output, utc_now(), evidence)
    (output / "plan.json").write_text(json_text(plan))
    load_plan(output)
    log(f"frozen plan published at {plan['frozen_at']}")
    return 0


def derive(output):
    """The evaluation-stream verdicts, once, after the freeze and before any score."""
    output = Path(output)
    load_plan(output)
    if branches_path(output, "evaluation").is_file():
        raise ValueError("the evaluation verdicts exist; they are derived once")
    if (output / "records").exists():
        raise ValueError("records exist before the evaluation verdicts; refusing")
    payload = derive_branches(output, "evaluation")
    verdicts = verdict_table(payload)
    counts = mixed_counts(evaluation_items(), verdicts)
    write_json(output / "verdicts.json", {
        "schema": SCHEMA_VERDICTS, "identity": IDENTITY, "derived_at": payload["derived_at"],
        "branches": len(verdicts), "sha256": sha256_of(branches_path(output, "evaluation")),
        "prevalence": {c: sum(v[c] for v in verdicts.values()) / len(verdicts) for c in COSTS},
        "mixed_cells": counts,
        "mixed_cell_guard": {c: counts[c][PRIMARY] >= MIN_MIXED_GRID for c in COSTS},
        "wall_seconds": payload["wall_seconds"], "cpu_process_seconds": payload["cpu_process_seconds"]})
    log(f"evaluation verdicts derived: mixed cells {counts}")
    return 0


# ---------------------------------------------------------------------------
# publication and validation
# ---------------------------------------------------------------------------

def publish(output):
    output = Path(output)
    plan = load_plan(output)
    compute = compute_tables(output, plan)
    for name, text in rendered(output, plan, compute).items():
        (output / name).write_text(text)
    log(f"published: disposition {compute['disposition']['token']}")
    return 0


def spot_checks(output, records):
    """Re-score Gate-B F-15-macro and J (seed 1) on every grid level and re-derive a spread of evaluation branches."""
    anchors = gb.load_anchors(gb.OUTPUT)["anchors"]
    seed = SEEDS[0]
    differing = 0
    with harness.deterministic_scoring():
        model, _ = load_predictor("gateb", seed)
        for arm in (WRONG_ARM, "J"):
            policy = p101.load_policy(p101.OUTPUT, *POLICY_ARMS[arm], seed) if arm in POLICY_ARMS else None
            stored = records[("gateb", arm, seed)]["inventories"][PRIMARY]
            for member, items in sorted(evaluation_items()[PRIMARY].items()):
                fresh, _ = score_block(model, policy, arm_spec(arm), anchors[member], items, False)
                differing += fresh["rows"] != stored[member]["rows"]
    evaluation = load_branches(output, "evaluation")["branches"]
    keys = sorted(evaluation)[::len(evaluation) // SPOT_BRANCHES][:SPOT_BRANCHES]
    jobs = {k: (a, s) for k, a, s, _ in stream_jobs("evaluation")}
    rederived = sum(cv.record_worker((k, *jobs[k]))[1] != {x: evaluation[k][x] for x in
                                                            ("quantities", "settle_end", "summary_sha256",
                                                             "summary_frames")} for k in keys)
    return {"rescored_levels_differing": differing, "rederived_branches": len(keys),
            "rederived_differing": rederived, "pass": differing == 0 and rederived == 0}


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    compute = compute_tables(output, plan)
    for name, text in rendered(output, plan, compute).items():
        if (output / name).read_bytes() != text.encode():
            raise ValueError(f"published {name} differs from the recomputation")
    spot = spot_checks(output, load_records(output))
    if not spot["pass"]:
        raise ValueError(f"spot checks failed: {spot}")
    log(f"validation passed ({time.monotonic() - began:.0f}s): tables re-derived and byte-compared; {spot}")
    return 0


def dry_run(output):
    log(f"output {output}; tasks {len(score_tasks())}; seeds {SEEDS}; evaluation levels per inventory "
        f"{ {i: len(v) for i, v in evaluation_items().items()} }")
    return 0


MODES = {"dry-run": dry_run, "smoke": smoke, "prepare": prepare, "derive": derive, "score": score,
         "publish": publish, "validate": validate}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker", type=int, default=0)
    args = parser.parse_args()
    mode = next(m for m in MODES if getattr(args, m.replace("-", "_")))
    if mode == "score":
        return score(OUTPUT, args.workers, args.worker)
    return MODES[mode](OUTPUT)


if __name__ == "__main__":
    raise SystemExit(main())
