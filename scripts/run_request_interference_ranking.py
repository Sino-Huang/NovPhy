"""Issue-117 ranking (confirmatory): per-request ranking AUC of the joint predictor against
single-request specialists and per-Delta models on the #104 sealed #109 evaluation split.

Binding runner module: scripts/run_request_interference_ranking.py
Exact validation command: python -u -m scripts.run_request_interference_ranking --validate

#117 measured per-request ranking EXPLORATORY (leave-one-lineage-out on the 12 #77 lineages and
the exposed held-out 007/008/014/016). Gate B passed, so the ranking part becomes confirmatory on
the sealed split. This runner

1. trains, on the #109 fit split (the Gate-B retrain payload, hash-checked), the nine
   single-request specialists S-* and the three per-Delta models D1/D5/D15 with the #117 arm
   construction (``run_request_interference.train_unit`` in a private copy bound to this output)
   under the LC3 recipe (#112 v3 LC, the Gate-B recipe), fold 'all', the three #112 seeds, one
   member per group as in the retrain. Two budgets: the per-request budget (exactly the joint
   run's updates of the arm's requests, each at its joint step; primary) and the full budget
   (9000 updates; DESCRIPTIVE). A member is usable under the retrain rule: complete, at most 45
   skipped updates, no #117 end-of-training excursion (``run_decision_dynamics_retrain.excursion``).
   An unusable member is a typed terminal failure, never retried or replaced;
2. scores once, under deterministic_scoring(), in processes that did not train: the joint model
   (the Gate-B handoff LC hybrid checkpoints) at each of its nine requests and every specialist /
   per-Delta model at its requests. Each is scored on every mixed cell of the Gate-B evaluation set
   from the cell's E99 anchor, and on the grid candidates from every fit-split anchor (the
   wrong-anchor control);
3. reports per-request grid AUC joint - specialist and joint - per-Delta, request means, the
   asymmetry by Delta and by description, the no-model action prior and the wrong-anchor
   controls. It also reports the pre-declared disposition on "sharing all weights does not cost
   per-request ranking on the sealed split".

Modes: --dry-run --smoke --prepare --train --score --publish --validate. --train and --score take
--workers N --worker K (one process per GPU, CUDA_VISIBLE_DEVICES set by the caller). Zero engine
seconds. Intervals: member (level)-clustered percentile bootstraps (10000 draws, PCG64 7201),
DESCRIPTIVE.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import importlib.util
from pathlib import Path
import shutil
import time

import numpy as np
import torch

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
from world_model.training import request_interference as ri
from world_model.training.cnn_hybrid import PAIRS

ROOT = v1.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-117-ranking-confirmatory-v1"
RETRAIN = rt.OUTPUT
GATE_B = gb.OUTPUT
POLICY = p101.OUTPUT
EXPLORATORY = r117.OUTPUT

IDENTITY = "issue-117-ranking-confirmatory-v1"
SCHEMA_PLAN = "issue_117_ranking_plan_v1"
SCHEMA_SMOKE = "issue_117_ranking_smoke_v1"
SCHEMA_PREDICTOR = "issue_117_ranking_predictor_v1"
SCHEMA_UNIT = "issue_117_ranking_training_unit_v1"
SCHEMA_TRAINED = "issue_117_ranking_trained_v1"
SCHEMA_RECORD = "issue_117_ranking_record_v1"
SCHEMA_MANIFEST = "issue_117_ranking_manifest_v1"
SCHEMA_COMPUTE = "issue_117_ranking_compute_v1"
SCHEMA_REPORT = "issue_117_ranking_report_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_request_interference_ranking --validate"
RUNNER = "scripts/run_request_interference_ranking.py"

DEVICE = "cuda"
SEEDS = tuple(t96.SEEDS)
INVENTORIES = gb.INVENTORIES
PRIMARY = gb.GATE_INVENTORY
PRIMARY_COST = harness.PRIMARY_COST
DELTA = v1.DELTA
STOP_TOKEN = v1.STOP_TOKEN
NONFINITE_SHARE_MAX = v1.NONFINITE_SHARE_MAX
MIN_CLUSTERS = r117.MIN_CLUSTERS
ENDPOINT = t96.ENDPOINT
RECIPE = r117.GATE_B_RECIPE  # LC3: the #112 v3 LC recipe (the Gate-B recipe)
REQUESTS = r117.REQUESTS
GATE_B_UNIT_REQUEST = r117.GATE_B_UNIT_REQUEST
JOINT = "J"
SPECIALIST_ARMS = tuple(arm for arm in r117.ARMS if arm != JOINT)
BUDGETS = ri.BUDGETS
PRIMARY_BUDGET = "request"
TRAINED = tuple((arm, budget) for arm in SPECIALIST_ARMS for budget in BUDGETS)
SCORED_UNITS = (JOINT, *(f"{arm}/{budget}" for arm, budget in TRAINED))
KINDS = ("J", "S", "D")
COMPARISONS = (("J", "S"), ("J", "D"), ("D", "S"))
GROUPS = r117.GROUPS
SKIP_CAP = rt.SKIP_CAP
WORKERS = 4
SMOKE_STEPS = 12
SMOKE_WARMUP = 9
SMOKE_TIMED = 36
GROUP_OVERHEAD_SECONDS = 90.0  # compile and setup per training group (projection only)
GPU_CAP_SECONDS = 40 * 3600.0
WALL_CAP_SECONDS = 24 * 3600.0  # elapsed, 4 GPUs
STABILITY_TOLERANCE = 1e-12

read_json = v1.read_json
write_json = v1.write_json
json_text = v1.json_text
sha256_of = v1.sha256_of
fmt_interval = v1.fmt_interval


def log(message):
    print(f"[issue-117-ranking] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# the #117 arm construction, bound to this output
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def implementation():
    """A private copy of the #117 runner (``train_unit``, schedules, member files) with this
    runner's output, identity and schemas. The training code is the #117 code, unchanged."""
    loader = importlib.util.spec_from_file_location("issue117_ranking_implementation", ROOT / r117.RUNNER)
    impl = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(impl)
    impl.OUTPUT = OUTPUT
    impl.IDENTITY = IDENTITY
    impl.SCHEMA_PREDICTOR = SCHEMA_PREDICTOR
    impl.SCHEMA_UNIT = SCHEMA_UNIT
    impl.RUNNER = RUNNER
    return impl


def retrain_data(payload):
    """#117 ``Data`` over the retrain's indexed #109 fit payload (``rt.IndexedTrainData``, the
    data object the Gate-B joint was trained on). Fold 'all' only; the item-3 probes are off."""
    impl = implementation()

    class RetrainData(impl.Data):
        def __init__(self, payload_):
            self.base = rt.IndexedTrainData(payload_, "hybrid")
            self.window_members = self.base.window_members
            self.long_members = self.base.long_members
            self.cell_members = self.base.cell_members

        def lineage(self, fold):
            if fold != "all":
                raise ValueError("the confirmatory specialists train fold 'all' only")
            return None

        def cells_pad(self):
            return 0

    return RetrainData(payload)


def expected_payload_sha():
    return read_json(RETRAIN / "compute.json")["data"]["training_data_sha256"]


def retrain_payload():
    if sha256_of(rt.data_path(RETRAIN)) != expected_payload_sha():
        raise ValueError("the retrain training payload differs from its published hash")
    return rt.load_training_data(RETRAIN)


def unit_name(unit):
    return unit.replace("/", "--")


def unit_arm_budget(unit):
    arm, budget = unit.split("/")
    return arm, budget


def unit_requests(unit):
    return list(REQUESTS) if unit == JOINT else r117.arm_requests(unit_arm_budget(unit)[0])


def kind_unit(kind, request, budget):
    if kind == "J":
        return JOINT
    arm = r117.specialist_arm(request) if kind == "S" else r117.delta_arm(request)
    return f"{arm}/{budget}"


def view_key(unit, request):
    return f"{unit}:{request}"


def groups():
    """One member per group, as the retrain: (arm, budget, index, seed); index = seed position."""
    return [(arm, budget, index, seed) for arm, budget in TRAINED for index, seed in enumerate(SEEDS)]


def group_key(arm, budget, seed):
    return f"{arm}/{budget}/{seed}"


def parse_group(key):
    arm, budget, seed = key.split("/")
    return arm, budget, SEEDS.index(int(seed)), int(seed)


# ---------------------------------------------------------------------------
# paths and ledger
# ---------------------------------------------------------------------------

def trained_path(output, arm, budget, seed):
    return Path(output) / "training" / f"trained--{arm}--{budget}--seed{seed}.json"


def record_path(output, unit, seed):
    return Path(output) / "records" / f"{unit_name(unit)}--seed{seed}.json"


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
# usability (the retrain rule) of a trained member
# ---------------------------------------------------------------------------

def usability(complete, skipped, local, norms, requests, expected):
    """Retrain rule: usable iff complete, at most SKIP_CAP skipped updates, every update logged and
    no #117 end-of-training excursion (rt.excursion over the member's own schedule)."""
    end_state = rt.excursion(local, norms, requests) if complete else None
    usable = bool(complete and skipped <= SKIP_CAP and len(local) == expected and not end_state["excursion"])
    failure = (None if usable else "retired" if not complete else "skip_cap" if skipped > SKIP_CAP
               else "incomplete_log" if len(local) != expected else "final_checkpoint_excursion")
    return {"complete": bool(complete), "skipped": int(skipped), "logged_updates": len(local), "end_state": end_state,
            "usable": usable, "typed_failure": failure}


def member_state(output, arm, budget, seed):
    impl = implementation()
    index = SEEDS.index(seed)
    report = read_json(impl.unit_path(output, RECIPE, arm, budget, index))
    updates = torch.load(impl.updates_path(output, RECIPE, arm, budget, index), map_location="cpu", weights_only=True)
    trained = read_json(trained_path(output, arm, budget, seed))
    member = report["members"][0]
    schedule = [tuple(s) for s in updates["schedule"]]
    local, norms = updates["logs"]["local"][:, 0].double(), updates["logs"]["grad_norm"][:, 0].double()
    state = usability(member["complete"], member["skipped"], local, norms, [REQUESTS[p] for p, _ in schedule],
                      len(impl.schedule_of(arm, budget)))
    return {"arm": arm, "budget": budget, "seed": seed, **state,
            "schedule_matches": schedule == [tuple(s) for s in impl.schedule_of(arm, budget)],
            "retired_at": member["retired_at_update"], "retired_request": member["retired_request"],
            "gpu_seconds": report["gpu_seconds"], "peak_memory_gb": report["peak_memory_gb"],
            "parameters": report["parameters_per_member"], "updates": report["updates"],
            "nonfinite_updates": sum(e["count"] for e in report["nonfinite_events"]),
            "trained_at": trained["trained_at"], "sha256": trained["sha256"],
            "training_data_sha256": trained["training_data_sha256"]}


# ---------------------------------------------------------------------------
# training (its own invocations; torch.compile)
# ---------------------------------------------------------------------------

def train(output, workers, worker):
    output = Path(output)
    plan = load_plan(output)
    if workers != plan["execution"]["workers"]:
        raise ValueError(f"the frozen assignment is for {plan['execution']['workers']} workers")
    impl = implementation()
    torch.backends.cudnn.benchmark = True
    mine = []
    for key in plan["execution"]["training"]["assignment"][str(worker)]:
        arm, budget, _, seed = parse_group(key)
        if not trained_path(output, arm, budget, seed).is_file():
            mine.append(key)
    if not mine:
        return 0
    payload_sha = sha256_of(rt.data_path(RETRAIN))
    if payload_sha != plan["data"]["training_data_sha256"]:
        raise ValueError("the retrain training payload differs from the frozen hash")
    data = retrain_data(rt.load_training_data(RETRAIN))
    for key in mine:
        arm, budget, index, seed = parse_group(key)
        began = time.monotonic()
        report, ensemble = impl.train_unit(output, RECIPE, arm, budget, index, [("all", seed)], data, probe_every=0)
        del ensemble
        torch.cuda.empty_cache()
        write_json(trained_path(output, arm, budget, seed), {
            "schema": SCHEMA_TRAINED, "plan_identity": IDENTITY, "arm": arm, "budget": budget, "seed": seed,
            "trained_at": utc_now(), "gpu_seconds": report["gpu_seconds"], "updates": report["updates"],
            "training_data_sha256": payload_sha,
            "sha256": sha256_of(impl.member_path(output, RECIPE, arm, budget, "all", seed))})
        mark(output, f"train-{arm}-{budget}-seed{seed}", time.monotonic() - began, report["gpu_seconds"],
             worker=worker, device=torch.cuda.get_device_name(0))
        log(f"worker {worker}: trained {key} ({report['updates']} updates): {report['gpu_seconds']:.0f}s; "
            f"complete {report['members'][0]['complete']}, skipped {report['members'][0]['skipped']}")
    return 0


def training_complete(output):
    return all(trained_path(output, arm, budget, seed).is_file() for arm, budget, _, seed in groups())


# ---------------------------------------------------------------------------
# scoring (deterministic_scoring; processes that did not train)
# ---------------------------------------------------------------------------

def unit_model(output, unit, seed):
    """(model, typed failure, checkpoint sha256) of a scored unit."""
    if unit == JOINT:
        model, failure = gb.load_model("unit", seed)  # hash-checked against the Gate-B handoff
        return model, failure, sha256_of(gb.member_path("unit", seed))
    arm, budget = unit_arm_budget(unit)
    state = member_state(output, arm, budget, seed)
    if state["typed_failure"]:
        return None, state["typed_failure"], None
    path = implementation().member_path(output, RECIPE, arm, budget, "all", seed)
    if sha256_of(path) != state["sha256"]:
        raise ValueError(f"{unit} seed {seed}: checkpoint differs from its training record")
    model = implementation().load_member(output, RECIPE, arm, budget, "all", seed)
    return model, None, state["sha256"]


def score_record(output, cells, anchors, fit_anchors, costs, unit, seed, policy):
    """Rows of every mixed cell of every inventory from its own anchor, and the grid candidates
    from every fit-split anchor (wrong-anchor control), for each request of the unit."""
    began = time.monotonic()
    model, failure, sha = unit_model(output, unit, seed)
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "scoring_policy": policy, "scored_at": utc_now(),
            "unit": unit, "recipe": RECIPE, "seed": seed, "engine_seconds": 0, "typed_failure": failure,
            "checkpoint_sha256": sha}
    if model is None:
        return {**base, "requests": {}}
    items = gb.inventory_items(PRIMARY)
    requests = {}
    for request in unit_requests(unit):
        inventories = {inventory: {} for inventory in INVENTORIES}
        for inventory, member, entry in p101.ordered_cells(cells):
            inventories[inventory][member] = gb.request_block(model, request, anchors[member], entry["inventory"], costs)
        wrong = {member: gb.request_block(model, request, fit_anchors[member], items, costs)
                 for member in sorted(fit_anchors)}
        requests[request] = {"inventories": inventories, "wrong_anchor": wrong}
    torch.cuda.synchronize()
    return {**base, "requests": requests, "seconds": time.monotonic() - began}


def read_record(output, unit, seed):
    record = read_json(record_path(output, unit, seed))
    if (record.get("schema") != SCHEMA_RECORD or record.get("plan_identity") != IDENTITY or record["unit"] != unit
            or record["seed"] != seed):
        raise ValueError(f"record {unit}/{seed} binding differs")
    return record


def score(output, workers, worker):
    output = Path(output)
    plan = load_plan(output)
    if workers != plan["execution"]["workers"]:
        raise ValueError(f"the frozen assignment is for {plan['execution']['workers']} workers")
    if not training_complete(output):
        raise ValueError("training is not complete; score after every specialist is trained")
    _, cells = gb.evaluation_set()
    anchors = gb.load_anchors(GATE_B)["anchors"]
    fit_anchors = gb.fit_anchors()
    costs = p101.endpoint_costs()
    for key in plan["execution"]["scoring"]["assignment"][str(worker)]:
        unit, seed = key.split("|")
        seed = int(seed)
        path = record_path(output, unit, seed)
        if path.is_file():
            continue
        began = time.monotonic()
        with harness.deterministic_scoring() as policy:
            record = score_record(output, cells, anchors, fit_anchors, costs, unit, seed, policy)
        write_json(path, record)
        torch.cuda.empty_cache()
        wall = time.monotonic() - began
        mark(output, f"score-{unit_name(unit)}-seed{seed}", wall, wall, worker=worker)
        log(f"worker {worker}: scored {unit} seed {seed} ({wall:.0f}s)"
            + (f"; typed failure {record['typed_failure']}" if record["typed_failure"] else ""))
    return 0


def scoring_complete(output):
    return all(record_path(output, unit, seed).is_file() for unit in SCORED_UNITS for seed in SEEDS)


# ---------------------------------------------------------------------------
# wrong-anchor statistics (vectorized c2.auc / c2.spearman_rows over the fit anchors)
# ---------------------------------------------------------------------------

def wrong_matrix(wrong):
    """The fit-anchor cost matrix [anchors, ordinals] of one record request (anchors sorted), or
    None when a row is excluded or nonfinite (the caller then uses the per-anchor loop)."""
    anchors = sorted(wrong)
    ordinals = [row["ordinal"] for row in wrong[anchors[0]]["rows"]]
    rows = [wrong[a]["rows"] for a in anchors]
    if any([row["ordinal"] for row in r] != ordinals for r in rows):
        raise ValueError("wrong-anchor rows differ in their ordinals")
    if any(row["excluded"] or row[PRIMARY_COST] is None for r in rows for row in r):
        return {"rows": rows, "costs": None}
    return {"rows": rows, "index": {o: k for k, o in enumerate(ordinals)},
            "costs": np.asarray([[row[PRIMARY_COST] for row in r] for r in rows], dtype=float)}


def spearman_vector(x, matrix):
    """v1.spearman(x, row) for every row of ``matrix`` (None where undefined)."""
    from scipy.stats import rankdata
    if len(x) < 3 or len(set(x)) < 2:
        return [None] * len(matrix)
    rx = rankdata(x)
    ry = rankdata(matrix, axis=1)
    dx = rx - rx.mean()
    dy = ry - ry.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):  # constant rows: undefined, reported as None below
        rho = (dy @ dx) / np.sqrt((dx @ dx) * (dy * dy).sum(axis=1))
    distinct = matrix.max(axis=1) > matrix.min(axis=1)
    return [float(r) if ok else None for r, ok in zip(rho, distinct)]


def anchor_summary(entry, right, matrix):
    """{right, wrong, stability} of one cell: the right-anchor AUC, the mean AUC from every fit
    anchor and the mean Spearman between right and wrong candidate costs (Gate-B anchor_rows)."""
    verdicts = entry["verdicts"]
    if matrix["costs"] is None or any(o not in matrix["index"] for o in verdicts):
        wrong_auc = [a for a in (c2.auc(entry, w) for w in matrix["rows"]) if a is not None]
        rhos = [r for r in (c2.spearman_rows(right, w) for w in matrix["rows"]) if r is not None]
    else:
        costs = matrix["costs"]
        pos = [matrix["index"][o] for o in verdicts if verdicts[o]]
        neg = [matrix["index"][o] for o in verdicts if not verdicts[o]]
        if pos and neg:
            p, n = costs[:, pos][:, :, None], costs[:, neg][:, None, :]
            wrong_auc = list(((p < n).sum(axis=(1, 2)) + 0.5 * (p == n).sum(axis=(1, 2))) / (len(pos) * len(neg)))
        else:
            wrong_auc = []
        x = {r["ordinal"]: r[PRIMARY_COST] for r in right if not r["excluded"]}
        common = sorted(o for o in x if o in matrix["index"])
        rhos = [r for r in spearman_vector([x[o] for o in common], costs[:, [matrix["index"][o] for o in common]])
                if r is not None]
    return {"right": c2.auc(entry, right), "wrong": float(np.mean(wrong_auc)) if wrong_auc else None,
            "stability": float(np.mean(rhos)) if rhos else None}


# ---------------------------------------------------------------------------
# statistics (pure: records, member states, cells -> tables)
# ---------------------------------------------------------------------------

def found_rows(record, request, inventory, member):
    return None if record["typed_failure"] else record["requests"][request]["inventories"][inventory][member]["rows"]


def view_rows(records, cells, inventory, stats):
    """One row per (level, seed): a view per (unit, request) of that seed and the seed-free prior."""
    rows = []
    for member, entry in sorted(cells[inventory]["mixed"].items()):
        prior = m99.view(entry, gb.prior_rows(entry, stats), PRIMARY_COST)
        for seed in SEEDS:
            views = {view_key(unit, request): v1.cost_view(entry, found_rows(records[(unit, seed)], request, inventory,
                                                                           member), PRIMARY_COST)
                     for unit in SCORED_UNITS for request in unit_requests(unit)}
            views["prior"] = prior
            rows.append({"member": member, "seed": seed, "views": views})
    return rows


def noninferiority(estimate):
    """delta rule for 'the first ranks no worse than the second': supported iff the lower bound of the
    paired difference > -DELTA; not_supported_by_this_experiment iff the upper bound < -DELTA;
    otherwise (or fewer than MIN_CLUSTERS levels) readiness_or_precision_insufficient."""
    if estimate is None or estimate["clusters"] < MIN_CLUSTERS:
        return STOP_TOKEN
    low, high = estimate["interval"]
    if low > -DELTA:
        return "supported"
    if high < -DELTA:
        return "not_supported_by_this_experiment"
    return STOP_TOKEN


def contrast(rows, pieces):
    block = m99.paired(rows, pieces)
    estimate = block["estimate"]
    return {**block, "token": noninferiority(estimate),
            "upper_below_zero": bool(estimate is not None and estimate["interval"][1] < 0)}


def mean_pieces(first, second, requests, budget, weight=1.0):
    n = len(requests)
    return ([(weight / n, view_key(kind_unit(first, r, budget), r)) for r in requests]
            + [(-weight / n, view_key(kind_unit(second, r, budget), r)) for r in requests])


def contrast_tables(rows, budget):
    out = {}
    for first, second in COMPARISONS:
        out[f"{first}-{second}"] = {
            "per_request": {r: contrast(rows, mean_pieces(first, second, [r], budget)) for r in REQUESTS},
            "groups": {g: contrast(rows, mean_pieces(first, second, requests, budget))
                       for g, requests in GROUPS.items()},
            "delta1_minus_delta15": m99.paired(rows, mean_pieces(first, second, GROUPS["delta_1"], budget)
                                               + mean_pieces(first, second, GROUPS["delta_15"], budget, -1.0)),
            "continuous_minus_macro": m99.paired(rows, mean_pieces(first, second, GROUPS["alpha_continuous"], budget)
                                                 + mean_pieces(first, second, GROUPS["alpha_macro"], budget, -1.0))}
    return out


def kind_mean_pieces(kind, budget):
    return [(1 / len(REQUESTS), view_key(kind_unit(kind, r, budget), r)) for r in REQUESTS]


def ranking_tables(rows):
    rankers = {view_key(unit, r): v1.ranker_block(rows, view_key(unit, r))
               for unit in SCORED_UNITS for r in unit_requests(unit)}
    rankers["prior"] = v1.ranker_block(rows, "prior")
    out = {"cells": len(rows), "members": len({r["member"] for r in rows}), "rankers": rankers,
           "request_mean": {}, "minus_prior": {}, "contrasts": {}}
    for budget in BUDGETS:
        out["contrasts"][budget] = contrast_tables(rows, budget)
        for kind in KINDS:
            if kind == "J" and budget != PRIMARY_BUDGET:
                continue
            name = kind if kind == "J" else f"{kind}/{budget}"
            out["request_mean"][name] = m99.paired(rows, kind_mean_pieces(kind, budget))
            block = m99.paired(rows, kind_mean_pieces(kind, budget) + [(-1, "prior")])
            out["minus_prior"][name] = {**block, "token": c2.token(block["estimate"]),
                                        "per_request": {r: m99.paired(rows, [(1, view_key(kind_unit(kind, r, budget), r)),
                                                                             (-1, "prior")]) for r in REQUESTS}}
    return out


def anchor_rows(records, cells):
    rows = []
    matrices = {(unit, seed, r): None if records[(unit, seed)]["typed_failure"]
                else wrong_matrix(records[(unit, seed)]["requests"][r]["wrong_anchor"])
                for unit in SCORED_UNITS for seed in SEEDS for r in unit_requests(unit)}
    for member, entry in sorted(cells[PRIMARY]["mixed"].items()):
        for seed in SEEDS:
            views = {}
            for unit in SCORED_UNITS:
                for r in unit_requests(unit):
                    matrix = matrices[(unit, seed, r)]
                    right = found_rows(records[(unit, seed)], r, PRIMARY, member)
                    views[view_key(unit, r)] = None if matrix is None else anchor_summary(entry, right, matrix)
            rows.append({"member": member, "seed": seed, "views": views})
    return rows


def anchor_block(rows, pieces_right, pieces_wrong):
    difference = c2.difference_block(rows, [(w, k, "right") for w, k in pieces_right]
                                     + [(-w, k, "wrong") for w, k in pieces_wrong])
    return {"right": c2.difference_block(rows, [(w, k, "right") for w, k in pieces_right]),
            "wrong": c2.difference_block(rows, [(w, k, "wrong") for w, k in pieces_wrong]),
            "right_minus_wrong": difference,
            "stability": c2.difference_block(rows, [(w, k, "stability") for w, k in pieces_right]),
            "token": c2.token(difference)}


def anchor_tables(rows, wrong_anchors):
    units = {view_key(unit, r): anchor_block(rows, [(1, view_key(unit, r))], [(1, view_key(unit, r))])
             for unit in SCORED_UNITS for r in unit_requests(unit)}
    kinds = {}
    for budget in BUDGETS:
        for kind in KINDS:
            if kind == "J" and budget != PRIMARY_BUDGET:
                continue
            pieces = kind_mean_pieces(kind, budget)
            kinds[kind if kind == "J" else f"{kind}/{budget}"] = anchor_block(rows, pieces, pieces)
    return {"cells": len(rows), "wrong_anchors": wrong_anchors, "units": units, "request_mean": kinds}


def statistics(records, cells, stats, wrong_anchors):
    tables = {i: ranking_tables(view_rows(records, cells, i, stats[i])) for i in INVENTORIES}
    return tables, anchor_tables(anchor_rows(records, cells), wrong_anchors)


def disposition(grid, guards_ok):
    """Pre-declared rule on 'sharing all weights does not cost per-request ranking on the sealed split'
    (the #117 LOLO T2 analogue): grid, LC3, per-request budget, joint - specialist."""
    js = grid["contrasts"][PRIMARY_BUDGET]["J-S"]
    blocks = {"request_mean": js["groups"]["all"], GATE_B_UNIT_REQUEST: js["per_request"][GATE_B_UNIT_REQUEST]}
    tokens = {name: block["token"] for name, block in blocks.items()}
    t2 = sorted(name for name, block in blocks.items() if block["upper_below_zero"])
    if not guards_ok:
        token = STOP_TOKEN
    elif all(t == "supported" for t in tokens.values()):
        token = "supported"
    elif any(t == "not_supported_by_this_experiment" for t in tokens.values()):
        token = "not_supported_by_this_experiment"
    else:
        token = STOP_TOKEN
    return {"token": token, "contrast_tokens": tokens,
            "estimates": {name: block["estimate"] for name, block in blocks.items()},
            "t2_reading": {"upper_bound_below_zero": t2, "cost_detected": bool(t2)}}


def invalidate(obj):
    """A failed guard turns every token into readiness_or_precision_insufficient."""
    if isinstance(obj, dict):
        return {k: (STOP_TOKEN if k in ("token", "contrast_tokens") and not isinstance(v, dict)
                    else {kk: STOP_TOKEN for kk in v} if k == "contrast_tokens" else invalidate(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [invalidate(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# guards and compute
# ---------------------------------------------------------------------------

def replication(records):
    """The joint's rows equal the Gate-B unit records (F-15-macro, every inventory and the wrong-anchor
    rows) and the #101 fixed-request records (nine requests, every inventory), every seed."""
    same, blocks = True, 0
    for seed in SEEDS:
        mine = records[(JOINT, seed)]
        gate = read_json(gb.record_path(GATE_B, "unit", seed))
        unit = mine["requests"][GATE_B_UNIT_REQUEST]
        for inventory in INVENTORIES:
            for member, block in gate["inventories"][inventory].items():
                same &= unit["inventories"][inventory][member]["rows"] == block["rows"]
                blocks += 1
        for member, block in gate["wrong_anchor"].items():
            same &= unit["wrong_anchor"][member]["rows"] == block["rows"]
            blocks += 1
        for request in REQUESTS:
            reference = p101.read_record(POLICY, request, seed)["inventories"]
            for inventory in INVENTORIES:
                for member, block in reference[inventory].items():
                    same &= mine["requests"][request]["inventories"][inventory][member]["rows"] == block["rows"]
                    blocks += 1
    return {"blocks": blocks, "pass": bool(same)}


def load_records(output):
    return {(unit, seed): read_record(output, unit, seed) for unit in SCORED_UNITS for seed in SEEDS}


def manifest(output):
    output = Path(output)
    files = sorted(output.glob("records/*.json")) + sorted(output.glob("updates/*.pt"))
    return {"schema": SCHEMA_MANIFEST, "identity": IDENTITY,
            "files": {str(p.relative_to(output)): sha256_of(p) for p in files}}


def compute_tables(output, plan):
    output = Path(output)
    branches, cells = gb.evaluation_set()
    records = load_records(output)
    states = [member_state(output, arm, budget, seed) for arm, budget, _, seed in groups()]
    fit = read_json(gb.FIT_VERDICTS)["branches"]
    stats = {i: gb.prior_stats(fit, i) for i in INVENTORIES}
    fit_anchors = gb.fit_anchors()
    tables, anchors = statistics(records, cells, stats, len(fit_anchors))
    payload = rt.load_training_data(RETRAIN)
    eval_roster = {lv["identity"] for lv in gb.evaluation_levels()}
    fit_roster = {lv["identity"] for lv in rt.fit_levels()}
    scored_at = sorted(r["scored_at"] for r in records.values())
    trained_at = sorted(s["trained_at"] for s in states)
    handoff = {c["seed"]: c["sha256"] for c in gb.checkpoints("unit")}
    by_state = {(f"{s['arm']}/{s['budget']}", s["seed"]): s for s in states}
    hashes = {f"{u}/{s}": (r["checkpoint_sha256"] == (handoff[s] if u == JOINT else by_state[(u, s)]["sha256"]))
              for (u, s), r in records.items() if not r["typed_failure"]}
    grid = tables[PRIMARY]
    joint_nonfinite = {r: grid["rankers"][view_key(JOINT, r)]["nonfinite_share"] for r in REQUESTS}
    guards = {
        "G1_isolation": {"pass": bool(set(payload["shot_members"]) <= fit_roster and not (fit_roster & eval_roster)
                                      and all(Path(b["attempt"]).parent.parent.name == gb.STREAM
                                              and b["member"] in eval_roster for b in branches)
                                      and all(set(cells[i]["mixed"]) <= eval_roster for i in INVENTORIES)
                                      and set(fit_anchors) <= fit_roster)},
        "G2_scoring_policy": {"records": len(records),
                              "pass": all(r["scoring_policy"] == harness.SCORING_POLICY for r in records.values())},
        "G3_chronology": {"plan_frozen_at": plan["frozen_at"], "first_trained_at": trained_at[0],
                          "last_trained_at": trained_at[-1], "first_scored_at": scored_at[0],
                          "pass": bool(plan["frozen_at"] <= trained_at[0] and trained_at[-1] <= scored_at[0])},
        "G4_checkpoint_binding": {"records": hashes,
                                  "training_data": all(s["training_data_sha256"] == plan["data"]["training_data_sha256"]
                                                       for s in states),
                                  "pass": bool(all(hashes.values()) and all(
                                      s["training_data_sha256"] == plan["data"]["training_data_sha256"]
                                      for s in states))},
        "G5_replication": replication(records),
        "G6_training_integrity": {"pass": bool(all(s["schedule_matches"] and s["updates"] == s["logged_updates"]
                                                   for s in states))},
        "G7_nonfinite": {"shares": joint_nonfinite,
                         "pass": all(v is not None and v <= NONFINITE_SHARE_MAX for v in joint_nonfinite.values())},
    }
    guards_ok = all(g["pass"] for g in guards.values())
    result = {"inventories": tables, "wrong_anchor": anchors, "disposition": disposition(grid, guards_ok)}
    if not guards_ok:
        result = invalidate(result)
    phases = {p.stem: read_json(p) for p in sorted((output / "ledger").glob("*.json"))}
    smoke = read_json(output / "smoke.json")
    all_phases = {"smoke": {"wall_seconds": smoke["wall_seconds"], "gpu_seconds": smoke["wall_seconds"]},
                  **{k: {"wall_seconds": v["wall_seconds"], "gpu_seconds": v["gpu_seconds"]} for k, v in phases.items()}}
    training = {"members": states,
                "usable": {f"{a}/{b}": [s["seed"] for s in states if s["arm"] == a and s["budget"] == b and s["usable"]]
                           for a, b in TRAINED},
                "typed_failures": [{"unit": f"{s['arm']}/{s['budget']}", "seed": s["seed"],
                                    "typed_failure": s["typed_failure"],
                                    "requests_elevated": (s["end_state"] or {}).get("requests_elevated")}
                                   for s in states if s["typed_failure"]]}
    return {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
        "evaluation": gb.evaluation_accounting(branches, cells),
        "training": training,
        "guards": guards, "guards_pass": guards_ok,
        **result,
        "compute": {"phases": all_phases, "gpu_seconds": sum(b["gpu_seconds"] for b in all_phases.values()),
                    "wall_seconds_summed": sum(b["wall_seconds"] for b in all_phases.values()),
                    "training_gpu_seconds": sum(s["gpu_seconds"] for s in states),
                    "scoring_seconds": sum(v["wall_seconds"] for k, v in phases.items() if k.startswith("score-")),
                    "engine_seconds": 0, "hardware": "4 x RTX 5090 (one process per GPU)"},
    }


# ---------------------------------------------------------------------------
# renderings
# ---------------------------------------------------------------------------

def num(value, digits=4):
    return "n/a" if value is None else format(value, f".{digits}f")


def est(block):
    return fmt_interval(block["estimate"] if block and "estimate" in block else block)


def auc_of(grid, unit, request):
    return fmt_interval(grid["rankers"][view_key(unit, request)]["auc"])


def short(request):
    return request[2:]


def budget_table(add, tables, inventory, budget):
    t = tables[inventory]
    js, jd = t["contrasts"][budget]["J-S"], t["contrasts"][budget]["J-D"]
    add(f"| request | J | S ({budget}) | J − S | token | D ({budget}) | J − D | token |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in REQUESTS:
        add(f"| {short(r)} | {auc_of(t, JOINT, r)} | {auc_of(t, kind_unit('S', r, budget), r)} | "
            f"{est(js['per_request'][r])} | {js['per_request'][r]['token']} | "
            f"{auc_of(t, kind_unit('D', r, budget), r)} | {est(jd['per_request'][r])} | {jd['per_request'][r]['token']} |")
    add(f"| **request mean** | {est(t['request_mean'][JOINT])} | {est(t['request_mean'][f'S/{budget}'])} | "
        f"**{est(js['groups']['all'])}** | **{js['groups']['all']['token']}** | {est(t['request_mean'][f'D/{budget}'])} | "
        f"**{est(jd['groups']['all'])}** | **{jd['groups']['all']['token']}** |")
    add("")


def asymmetry_table(add, tables, budget):
    c = tables[PRIMARY]["contrasts"][budget]
    add("| group | J − S | J − D | D − S |")
    add("| --- | --- | --- | --- |")
    for g in GROUPS:
        add(f"| {g} | {est(c['J-S']['groups'][g])} | {est(c['J-D']['groups'][g])} | {est(c['D-S']['groups'][g])} |")
    for name, label in (("delta1_minus_delta15", "Δ=1 − Δ=15"), ("continuous_minus_macro", "continuous − macro")):
        add(f"| {label} | {est(c['J-S'][name])} | {est(c['J-D'][name])} | {est(c['D-S'][name])} |")
    add("")


def findings_md(plan, compute):
    lines = []
    add = lines.append
    tables, d = compute["inventories"], compute["disposition"]
    grid = tables[PRIMARY]
    add("# Issue #117 ranking (confirmatory): joint vs specialist on the #104 sealed split")
    add("")
    add(f"Frozen {plan['frozen_at']}; validation `{VALIDATION_COMMAND}`. Recipe {RECIPE} (#112 v3 LC, the Gate-B recipe); "
        f"joint = the Gate-B handoff LC hybrid (3 seeds); specialists trained here on the same #109 fit payload. "
        f"Grid primary ({grid['members']} mixed levels × {len(SEEDS)} seeds); offset, angle and power DESCRIPTIVE. "
        f"Intervals: level-clustered percentile bootstraps (DESCRIPTIVE). δ = {DELTA}. Engine seconds 0.")
    add("")
    add("## Disposition")
    add("")
    add(f"Claim: {plan['decision_rule']['claim']}")
    add("")
    add(f"Rule: {plan['decision_rule']['disposition']}")
    add("")
    for name, token in d["contrast_tokens"].items():
        add(f"- {name}: J − S {fmt_interval(d['estimates'][name])} → `{token}`")
    add(f"- #117 T2 reading (upper bound < 0): {d['t2_reading']['upper_bound_below_zero'] or 'none'}")
    add(f"- Guards pass: {compute['guards_pass']}")
    add("")
    add(f"**Disposition: `{d['token']}`**")
    add("")
    add("## Per-request grid AUC, per-request budget (primary; token = δ non-inferiority of J)")
    add("")
    budget_table(add, tables, PRIMARY, PRIMARY_BUDGET)
    add("## Asymmetry (grid, per-request budget)")
    add("")
    asymmetry_table(add, tables, PRIMARY_BUDGET)
    add("## Full budget (DESCRIPTIVE; 9000 updates per specialist / per-Δ model)")
    add("")
    budget_table(add, tables, PRIMARY, "full")
    asymmetry_table(add, tables, "full")
    add("## Controls (grid)")
    add("")
    add(f"No-model action prior (Gate-B rule, fit-verdicts.json): AUC {fmt_interval(grid['rankers']['prior']['auc'])}.")
    add("")
    add("| ranker (request mean) | AUC | − prior | token (δ superiority) | wrong-anchor AUC | right − wrong | stability |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    wa = compute["wrong_anchor"]["request_mean"]
    for name in grid["request_mean"]:
        mp = grid["minus_prior"][name]
        add(f"| {name} | {est(grid['request_mean'][name])} | {est(mp)} | {mp['token']} | "
            f"{fmt_interval(wa[name]['wrong'])} | {fmt_interval(wa[name]['right_minus_wrong'])} | "
            f"{fmt_interval(wa[name]['stability'])} |")
    add("")
    add(f"Wrong anchors: the {compute['wrong_anchor']['wrong_anchors']} fit-split anchors over the 16 shared grid "
        "candidates. Per request (per-request budget):")
    add("")
    add("| request | J right − wrong | S right − wrong | D right − wrong | J − prior | S − prior |")
    add("| --- | --- | --- | --- | --- | --- |")
    units = compute["wrong_anchor"]["units"]
    for r in REQUESTS:
        add(f"| {short(r)} | {fmt_interval(units[view_key(JOINT, r)]['right_minus_wrong'])} | "
            f"{fmt_interval(units[view_key(kind_unit('S', r, PRIMARY_BUDGET), r)]['right_minus_wrong'])} | "
            f"{fmt_interval(units[view_key(kind_unit('D', r, PRIMARY_BUDGET), r)]['right_minus_wrong'])} | "
            f"{est(grid['minus_prior'][JOINT]['per_request'][r])} | "
            f"{est(grid['minus_prior'][f'S/{PRIMARY_BUDGET}']['per_request'][r])} |")
    add("")
    add("## Descriptive inventories (request mean)")
    add("")
    add("| inventory | cells | prior | J | J − S (request) | J − D (request) | J − S (full) | J − D (full) |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for i in INVENTORIES:
        t = tables[i]
        c = t["contrasts"]
        add(f"| {i} | {t['cells']} | {fmt_interval(t['rankers']['prior']['auc'])} | {est(t['request_mean'][JOINT])} | "
            f"{est(c['request']['J-S']['groups']['all'])} | {est(c['request']['J-D']['groups']['all'])} | "
            f"{est(c['full']['J-S']['groups']['all'])} | {est(c['full']['J-D']['groups']['all'])} |")
    add("")
    add("## Training (LC3, #109 fit payload, fold 'all')")
    add("")
    add("| unit | updates | usable seeds | typed failures | GPU s |")
    add("| --- | --- | --- | --- | --- |")
    members = compute["training"]["members"]
    for arm, budget in TRAINED:
        ms = [m for m in members if m["arm"] == arm and m["budget"] == budget]
        fails = [f"{m['seed']}: {m['typed_failure']}" for m in ms if m["typed_failure"]]
        add(f"| {arm}/{budget} | {ms[0]['updates']} | {len([m for m in ms if m['usable']])}/{len(ms)} | "
            f"{'; '.join(fails) or '—'} | {sum(m['gpu_seconds'] for m in ms):.0f} |")
    add("")
    add("Typed failures are terminal (never retried or replaced); a cell enters a paired contrast only when every "
        "piece of it is defined, so a failed seed drops that seed's cells from the contrasts that read it.")
    add("")
    add("## Guards")
    add("")
    for name, g in compute["guards"].items():
        add(f"- {name}: {g['pass']}")
    add("")
    c = compute["compute"]
    add("## Compute")
    add("")
    add(f"GPU seconds {c['gpu_seconds']:.0f} (training {c['training_gpu_seconds']:.0f}, scoring {c['scoring_seconds']:.0f}); "
        f"engine seconds 0; {c['hardware']}.")
    add("")
    return "\n".join(lines)


def summary(plan, compute):
    grid = compute["inventories"][PRIMARY]
    c = grid["contrasts"]
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "validation_command": VALIDATION_COMMAND, "recipe": RECIPE,
            "disposition": compute["disposition"], "guards_pass": compute["guards_pass"],
            "grid_request_mean": {f"{k}/{b}": {"estimate": c[b][k]["groups"]["all"]["estimate"],
                                               "token": c[b][k]["groups"]["all"]["token"]}
                                  for b in BUDGETS for k in ("J-S", "J-D")},
            "grid_per_request_J_minus_S": {r: {"estimate": c[PRIMARY_BUDGET]["J-S"]["per_request"][r]["estimate"],
                                               "token": c[PRIMARY_BUDGET]["J-S"]["per_request"][r]["token"]}
                                           for r in REQUESTS},
            "prior_grid_auc": grid["rankers"]["prior"]["auc"],
            "typed_failures": compute["training"]["typed_failures"],
            "gpu_seconds": compute["compute"]["gpu_seconds"], "engine_seconds": 0,
            "claim_boundary": plan["claim_boundary"]}


def rendered(plan, compute, output):
    return {"compute.json": json_text(compute), "summary.json": json_text(summary(plan, compute)),
            "findings.md": findings_md(plan, compute), "records-manifest.json": json_text(manifest(output))}


# ---------------------------------------------------------------------------
# smoke (pre-freeze; fit payload and published Gate-B / #101 records only; nothing kept)
# ---------------------------------------------------------------------------

def max_delta(first, second):
    return max(float((first.params[k] - second.params[k]).abs().max()) for k in first.params)


def smoke_schedules():
    impl = implementation()
    joint = impl.schedule_of("J", "full")
    retrain_requests = rt.step_requests("hybrid")
    ok = (joint == [(k % 9, k) for k in range(ri.STEPS)] and [REQUESTS[p] for p, _ in joint] == retrain_requests
          and all(PAIRS[p] == v1.step_pair("hybrid", k) for p, k in joint))
    counts = {arm: {b: len(impl.schedule_of(arm, b)) for b in BUDGETS} for arm in SPECIALIST_ARMS}
    restricted = all(impl.schedule_of(arm, "request") == [(p, g) for p, g in joint if p in r117.ARMS[arm]]
                     for arm in SPECIALIST_ARMS)
    sizes = all(counts[a]["request"] == 1000 * len(r117.ARMS[a]) and counts[a]["full"] == ri.STEPS for a in counts)
    return {"control": "schedules: the joint schedule is the retrain's; per-request = the joint run's updates of the "
                       "arm's requests at their joint steps; full = 9000",
            "update_counts": counts, "ok": bool(ok and restricted and sizes)}


def smoke_training(scratch):
    """The #117 train_unit on the retrain payload equals the retrain's train_group (the Gate-B joint's
    training path) for SMOKE_STEPS updates, at the real schedule and at full cost, including the logged
    local losses and pre-clip norms the excursion rule reads; a specialist trains finite; timings."""
    impl = implementation()
    retrain = rt.implementation()
    torch.backends.cudnn.benchmark = True
    payload = retrain_payload()
    data = retrain_data(payload)
    reference_data = data.base  # the same IndexedTrainData object; generators are per call
    members = [("all", SEEDS[0])]
    checks = []
    for smoke_full in (False, True):
        ours, ensemble = impl.train_unit(scratch, RECIPE, "J", "full", 0, members, data, steps=SMOKE_STEPS,
                                         save=True, probe_every=0, smoke_full=smoke_full)
        logs = torch.load(impl.updates_path(scratch, RECIPE, "J", "full", 0), weights_only=True)["logs"]
        _, reference = retrain.train_group(scratch / "reference", rt.ARM, "hybrid", 0, members, reference_data,
                                           steps=SMOKE_STEPS, save=False, smoke_full=smoke_full)
        theirs = reference.log_tensors()
        delta = max_delta(ensemble, reference)
        same_logs = (torch.equal(logs["local"][:, 0].float(), theirs["local"][:, 0].float())
                     and torch.equal(logs["grad_norm"][:, 0].float(), theirs["grad_norm"][:, 0].float()))
        checks.append({"control": f"train_unit_equals_retrain_train_group_{'full_cost' if smoke_full else 'schedule'}",
                       "max_abs_parameter_delta": delta, "logs_equal": bool(same_logs),
                       "ok": bool(delta == 0.0 and same_logs)})
        del ensemble, reference
        shutil.rmtree(scratch, ignore_errors=True)
        torch.cuda.empty_cache()
    report, _ = impl.train_unit(scratch, RECIPE, "S-1-continuous", "full", 0, members, data, steps=SMOKE_STEPS,
                                save=False, probe_every=0, smoke_full=True)
    checks.append({"control": "specialist_trains_finite_at_full_cost", "skipped": report["members"][0]["skipped"],
                   "ok": bool(report["members"][0]["complete"] and report["members"][0]["skipped"] == 0)})
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    timed, _ = impl.train_unit(scratch, RECIPE, "J", "full", 0, members, data, steps=SMOKE_WARMUP + SMOKE_TIMED,
                               save=False, probe_every=0, smoke_full=True, timing_after=SMOKE_WARMUP, step_times=True)
    per_pair = {}
    for entry in timed["update_times"][SMOKE_WARMUP:]:
        per_pair.setdefault(entry["pair"], []).append(entry["seconds"])
    seconds = {str(p): float(np.median(v)) for p, v in sorted(per_pair.items())}
    warmup = sum(e["seconds"] for e in timed["update_times"][:SMOKE_WARMUP])
    shutil.rmtree(scratch, ignore_errors=True)
    return checks, {"seconds_per_update_full_cost": seconds, "warmup_seconds": warmup,
                    "peak_memory_gb": timed["peak_memory_gb"]}


def smoke_usability():
    """The usability rule applied here reproduces the retrain's published member table (end state, usable)."""
    compute = read_json(RETRAIN / "compute.json")
    same = True
    for m in compute["members"]:
        updates = torch.load(rt.updates_path(RETRAIN, m["family"], m["index"]), map_location="cpu", weights_only=True)
        local, norms = updates["logs"]["local"][:, 0].double(), updates["logs"]["grad_norm"][:, 0].double()
        state = usability(m["complete"], int(updates["skipped"][0]), local, norms, updates["requests"], rt.STEPS)
        same &= state["end_state"] == m["end_state"] and state["usable"] == m["usable"]
    return {"control": "usability_rule_reproduces_the_retrain_member_table", "members": len(compute["members"]),
            "ok": bool(same)}


def joint_records_from_published():
    """Stand-in records built from published outputs only: the joint's nine requests from the #101 fixed
    records, F-15-macro from the Gate-B unit record, wrong-anchor rows from the Gate-B unit record."""
    out = {}
    for seed in SEEDS:
        gate = read_json(gb.record_path(GATE_B, "unit", seed))
        requests = {}
        for r in REQUESTS:
            inventories = (gate["inventories"] if r == GATE_B_UNIT_REQUEST
                           else p101.read_record(POLICY, r, seed)["inventories"])
            requests[r] = {"inventories": inventories, "wrong_anchor": gate["wrong_anchor"]}
        out[seed] = {"typed_failure": None, "requests": requests}
    return out


def smoke_scoring():
    """Scoring path: the joint's F-15-macro rescored here reproduces the Gate-B unit records exactly (every
    mixed cell of every inventory and the 480 wrong-anchor blocks, every seed). Statistics rehearsal on
    published rows only: every unit stands in as the joint (so every contrast is 0), one specialist seed as a
    typed failure; the pooled AUC, the prior and the wrong-anchor table reproduce Gate B."""
    _, cells = gb.evaluation_set()
    anchors = gb.load_anchors(GATE_B)["anchors"]
    fit_anchors = gb.fit_anchors()
    costs = p101.endpoint_costs()
    items = gb.inventory_items(PRIMARY)
    checks, seconds = [], []
    same = True
    with harness.deterministic_scoring() as policy:
        for seed in SEEDS:
            gate = read_json(gb.record_path(GATE_B, "unit", seed))
            model, failure = gb.load_model("unit", seed)
            began = time.monotonic()
            for inventory, member, entry in p101.ordered_cells(cells):
                fresh = gb.request_block(model, GATE_B_UNIT_REQUEST, anchors[member], entry["inventory"], costs)
                same &= fresh["rows"] == gate["inventories"][inventory][member]["rows"]
            for member in sorted(fit_anchors):
                fresh = gb.request_block(model, GATE_B_UNIT_REQUEST, fit_anchors[member], items, costs)
                same &= fresh["rows"] == gate["wrong_anchor"][member]["rows"]
            torch.cuda.synchronize()
            seconds.append(time.monotonic() - began)
    checks.append({"control": "joint_scoring_reproduces_gate_b_unit_rows", "policy": policy, "ok": bool(same)})
    joint = joint_records_from_published()
    records = {}
    for unit in SCORED_UNITS:
        for seed in SEEDS:
            records[(unit, seed)] = {"typed_failure": None,
                                     "requests": {r: joint[seed]["requests"][r] for r in unit_requests(unit)}}
    records[("S-1-continuous/request", SEEDS[0])] = {"typed_failure": "final_checkpoint_excursion", "requests": {}}
    fit = read_json(gb.FIT_VERDICTS)["branches"]
    stats = {i: gb.prior_stats(fit, i) for i in INVENTORIES}
    tables, wrong = statistics(records, cells, stats, len(fit_anchors))
    gate_compute = read_json(GATE_B / "compute.json")
    grid = tables[PRIMARY]
    auc_same = grid["rankers"][view_key(JOINT, GATE_B_UNIT_REQUEST)]["auc"] == gate_compute["gate_b"]["pooled_grid_auc"]
    prior_same = all(tables[i]["rankers"]["prior"]["auc"] == gate_compute["inventories"][i]["rankers"]["prior"]["auc"]
                     for i in INVENTORIES)
    reference = gate_compute["controls"]["wrong_anchor"]
    ours = wrong["units"][view_key(JOINT, GATE_B_UNIT_REQUEST)]
    exact = all(ours[k] == reference[k] for k in ("right", "wrong", "right_minus_wrong"))
    stability = max(abs(a - b) for a, b in zip([ours["stability"]["mean"], *ours["stability"]["interval"]],
                                               [reference["stability"]["mean"], *reference["stability"]["interval"]]))
    zero = grid["contrasts"][PRIMARY_BUDGET]["J-S"]["groups"]["all"]["estimate"]
    zero_ok = max(abs(v) for v in (zero["mean"], *zero["interval"])) <= STABILITY_TOLERANCE  # 1/9-weighted sums
    failed_drop = grid["contrasts"][PRIMARY_BUDGET]["J-S"]["per_request"]["F-1-continuous"]["cells"]
    rehearsal = {"inventories": tables, "wrong_anchor": wrong, "disposition": disposition(grid, True),
                 "guards_pass": True, "guards": {},
                 "training": {"members": [{"arm": a, "budget": b, "seed": s, "updates": 0, "usable": True,
                                           "typed_failure": None, "gpu_seconds": 0.0} for a, b in TRAINED for s in SEEDS]},
                 "compute": {"gpu_seconds": 0, "training_gpu_seconds": 0, "scoring_seconds": 0, "hardware": "-"}}
    json_text(rehearsal)
    findings_md({"frozen_at": "rehearsal", "decision_rule": frozen_rules()}, rehearsal)
    checks.append({"control": "statistics_rehearsal_reproduces_gate_b", "unit_auc": auc_same, "prior": prior_same,
                   "wrong_anchor_exact": exact, "stability_max_abs_delta": stability,
                   "stand_in_contrast_is_zero": zero_ok, "stand_in_contrast": zero,
                   "typed_failure_drops_one_seed": failed_drop == grid["cells"] - grid["members"],
                   "ok": bool(auc_same and prior_same and exact and stability <= STABILITY_TOLERANCE
                              and zero_ok and failed_drop == grid["cells"] - grid["members"])})
    return checks, {"joint_f15_macro_seconds_per_seed": seconds}


def smoke_isolation():
    h = gb.handoff()
    usable = [c["seed"] for c in gb.checkpoints("unit") if c["usable"]]
    hashes = all(sha256_of(gb.member_path("unit", c["seed"])) == c["sha256"] for c in gb.checkpoints("unit"))
    eval_roster = {lv["identity"] for lv in gb.evaluation_levels()}
    payload = rt.load_training_data(RETRAIN)
    return {"control": "handoff_seeds_usable_hashes_and_fit_isolation", "handoff_token": h["unit"].get("token"),
            "usable_seeds": usable, "hashes": hashes,
            "ok": bool(tuple(usable) == SEEDS and hashes and not (set(payload["shot_members"]) & eval_roster))}


def projection(timing, scoring):
    impl = implementation()
    per_pair = {int(p): s for p, s in timing["seconds_per_update_full_cost"].items()}
    train_costs = [(group_key(arm, budget, seed), GROUP_OVERHEAD_SECONDS
                    + sum(per_pair[p] for p, _ in impl.schedule_of(arm, budget)))
                   for arm, budget, _, seed in groups()]
    per_transition = float(np.mean(scoring["joint_f15_macro_seconds_per_seed"])) / (ENDPOINT // 15)
    score_costs = [(f"{unit}|{seed}", sum(per_transition * ENDPOINT / r117.request_delta(r) for r in unit_requests(unit)))
                   for unit in SCORED_UNITS for seed in SEEDS]
    train_assignment, train_loads = lpt(train_costs, WORKERS)
    score_assignment, score_loads = lpt(score_costs, WORKERS)
    return {"training": {"assignment": train_assignment, "projected_seconds_per_worker": train_loads,
                         "projected_gpu_seconds": sum(c for _, c in train_costs)},
            "scoring": {"assignment": score_assignment, "projected_seconds_per_worker": score_loads,
                        "projected_gpu_seconds": sum(c for _, c in score_costs)},
            "projected_elapsed_seconds": max(train_loads) + max(score_loads),
            "note": ("training at the full-cost update time (horizon 225 and ranking from update 0, an upper bound) "
                     f"plus {GROUP_OVERHEAD_SECONDS:.0f} s per group; scoring from the smoke's joint F-15-macro time "
                     "scaled by 225 / Delta transitions")}


def lpt(costs, workers):
    """Longest-processing-time assignment: {worker: [keys]}, projected seconds per worker."""
    loads = [0.0] * workers
    out = {str(w): [] for w in range(workers)}
    for key, cost in sorted(costs, key=lambda kv: (-kv[1], kv[0])):
        w = min(range(workers), key=lambda k: (loads[k], k))
        out[str(w)].append(key)
        loads[w] += cost
    return out, loads


def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the smoke is a pre-freeze control")
    began = time.monotonic()
    scratch = output / "smoke-scratch"
    shutil.rmtree(scratch, ignore_errors=True)
    checks = [smoke_schedules(), smoke_isolation(), smoke_usability()]
    training_checks, timing = smoke_training(scratch)
    checks += training_checks
    scoring_checks, scoring = smoke_scoring()
    checks += scoring_checks
    projected = projection(timing, scoring)
    within = projected["projected_elapsed_seconds"] <= WALL_CAP_SECONDS
    checks.append({"control": "full_budget_fits_24h_on_4_gpus", "projected_elapsed_seconds":
                   projected["projected_elapsed_seconds"], "ok": bool(within)})
    ok = all(c["ok"] for c in checks)
    shutil.rmtree(scratch, ignore_errors=True)
    evidence = {"schema": SCHEMA_SMOKE, "identity": IDENTITY, "run_at": utc_now(), "checks": checks,
                "timing": timing, "scoring_timing": scoring, "projection": projected,
                "device": torch.cuda.get_device_name(0), "wall_seconds": time.monotonic() - began,
                "reading": ("infrastructure assertions only; nothing kept; no statistic of a new model output; the "
                            "only evaluation rows computed reproduce the published Gate-B unit records"), "ok": ok}
    write_json(output / "smoke.json", evidence)
    for check in checks:
        log(f"  {check['control']}: {check['ok']}")
    log(f"smoke {'PASSED' if ok else 'FAILED'}; projected elapsed {projected['projected_elapsed_seconds'] / 3600:.1f} h "
        f"on {WORKERS} GPUs (training GPU {projected['training']['projected_gpu_seconds'] / 3600:.1f} h, scoring GPU "
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
        **{f"gate_b_record_{n}_{s}": gb.record_path(GATE_B, n, s) for n in gb.UNITS for s in SEEDS},
        "issue_101_plan": POLICY / "plan.json",
        **{f"issue_101_record_{r}_{s}": p101.record_path(POLICY, r, s) for r in REQUESTS for s in SEEDS},
        "issue_117_plan": EXPLORATORY / "plan.json", "issue_117_compute": EXPLORATORY / "compute.json",
        "issue_117_runner": ROOT / r117.RUNNER, "issue_117_module": ROOT / r117.MODULE,
        "gate_b_runner": ROOT / gb.RUNNER, "issue_112_retrain_runner": ROOT / rt.RUNNER,
        "issue_112_v1_runner": ROOT / v1.RUNNER, "issue_112_versions_runner": ROOT / versions.RUNNER,
        "issue_101_runner": ROOT / p101.RUNNER, "issue_99_attribution_runner": ROOT / "scripts/run_decision_chain_attribution.py",
        "issue_112_controls_runner": ROOT / "scripts/run_decision_dynamics_controls.py",
        "issue_96_runner": ROOT / "scripts/run_tau_ad_within_checkpoint.py",
        "issue_99_encoder": s99.OUTPUT / "encoder" / "encoder.pt",
        "issue_109_cohort_plan": rt.COHORT, "issue_104_plan": rt.POWER_PLAN,
        "issue_104_campaign_branches": rt.CAMPAIGN / "branches.json",
        "scoring_harness": ROOT / "world_model/planning/scoring_harness.py",
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
        "claim": ("sharing all weights does not cost per-request ranking on the sealed split: the request-mean grid "
                  "AUC joint - specialist, and that of the Gate-B unit F-15-macro, under LC3 at the per-request "
                  "budget (the #117 LOLO T2 analogue, now confirmatory on the #104 sealed #109 evaluation split)"),
        "disposition": (f"supported iff both the request-mean and the F-15-macro grid AUC J - S have lower bound > "
                        f"-{DELTA} (non-inferiority at delta); not_supported_by_this_experiment iff either has upper "
                        f"bound < -{DELTA}; otherwise {STOP_TOKEN}; any failed guard turns every token into "
                        f"{STOP_TOKEN}. The #117 T2 condition (upper bound < 0) is reported next to the token"),
        "contrast_tokens": (f"every paired joint - specialist / joint - per-Delta / per-Delta - specialist AUC "
                            f"difference carries the same delta non-inferiority token (first no worse than second: "
                            f"lower > -{DELTA} supported; upper < -{DELTA} not_supported_by_this_experiment; else "
                            f"{STOP_TOKEN}; fewer than {MIN_CLUSTERS} levels {STOP_TOKEN})"),
        "secondary": ("J - D (per-Delta) on grid at the per-request budget: same rule, reported, not part of the "
                      "disposition. Full budget, offset / angle / power, the asymmetry by Delta and by description and "
                      "the controls: DESCRIPTIVE"),
        "pairing": ("cells (level, seed): the joint seed s is paired with the specialist / per-Delta seed s; a cell "
                    "enters a contrast only when every piece is defined (a typed-failure member drops its seed's "
                    "cells from the contrasts that read it); the bootstrap resamples levels"),
        "controls": ("no-model action prior: Gate-B rule over fit-verdicts.json; request-mean model - prior with the "
                     f"delta superiority token (mean >= {DELTA} and lower > 0 supported; upper < {DELTA} "
                     "not_supported_by_this_experiment). Wrong-anchor: every scored (unit, request) over the 16 grid "
                     "candidates from the 480 fit-split anchors (Gate-B rule): right, mean wrong, right - wrong (same "
                     "superiority token), Spearman stability"),
    }


def frozen_plan(output, frozen_at, smoke_evidence):
    branches, cells = gb.evaluation_set()
    impl = implementation()
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "issue": 117,
        "related_issues": [104, 108, 109, 111, 112, 116],
        "role": ("the ranking part of #117 items 1-2, confirmatory on the #104 sealed #109 evaluation split after "
                 "Gate B passed: per-request ranking AUC of the joint predictor against single-request specialists "
                 "and per-Delta models trained on the same #109 fit payload"),
        "frozen_at": frozen_at, "frozen_before_training_and_scoring": True,
        "validation_command": VALIDATION_COMMAND, "runner": RUNNER, "runner_sha256_at_freeze": sha256_of(ROOT / RUNNER),
        "engine_seconds": 0,
        "question": ("does sharing every weight across the nine requests cost a request's ranking of candidate launches "
                     "on the sealed split, against a specialist that makes the same updates of that request?"),
        "recipe": {"name": RECIPE, "definition": r117.RECIPES[RECIPE]["definition"],
                   "implementation": ("scripts.run_request_interference.train_unit (the #117 arm construction) in a "
                                      "private copy bound to this output; the smoke shows it equals the retrain's "
                                      "train_group for the joint arm bit for bit"),
                   "item3_probes": "off (training dynamics are #117 item 3, published)"},
        "arms": {"joint": ("J: the Gate-B handoff LC hybrid checkpoints (the #112 v3 retrain on the #109 fit split), "
                           "scored at each of its nine requests"),
                 **{arm: r117.ARM_DEFINITIONS[arm] for arm in SPECIALIST_ARMS},
                 "parameters": "every arm is a CNNHybridPredictor with 1 837 690 parameters"},
        "budgets": {**{b: r117.BUDGET_DEFINITIONS[b] for b in BUDGETS}, "primary": PRIMARY_BUDGET,
                    "update_counts": {arm: {b: len(impl.schedule_of(arm, b)) for b in BUDGETS} for arm in SPECIALIST_ARMS},
                    "full_budget_included": ("the smoke projects the full budget within 24 h elapsed on 4 GPUs; it is "
                                             "reported DESCRIPTIVE")},
        "data": {"payload": str(rt.data_path(RETRAIN).relative_to(ROOT)), "training_data_sha256": expected_payload_sha(),
                 "definition": "the retrain's #109 fit payload (13 919 shots, 680 ranking cells), fold 'all'"},
        "seeds": list(SEEDS),
        "members": "one member per group (as the retrain): 12 arms x 2 budgets x 3 seeds = 72 members",
        "usable": (f"a member is usable iff complete, at most {SKIP_CAP} skipped updates, every update logged and no "
                   "#117 end-of-training excursion (rt.excursion over its own schedule, declared before training); an "
                   "unusable member is a typed terminal failure (retired / skip_cap / final_checkpoint_excursion), "
                   "never retried or replaced; the joint uses the handoff usability (all three seeds usable)"),
        "evaluation": {"split": "the #109 evaluation split as in Gate B (stream issue109-evaluation, mixed cells)",
                       "accounting": gb.evaluation_accounting(branches, cells),
                       "anchors": "the Gate-B E99 anchor store", "primary_inventory": PRIMARY,
                       "descriptive": list(gb.DESCRIPTIVE_INVENTORIES),
                       "cost": "tie_free (EndpointCosts.row against the anchor; gb.request_block)",
                       "cell_auc": "m99.view; cells (level, seed); bootstrap over levels (10000 draws, PCG64 7201)"},
        "decision_rule": frozen_rules(),
        "guards": {"G1_isolation": "training payload members are fit levels; scored cells are evaluation levels; wrong "
                                   "anchors are fit levels",
                   "G2_scoring_policy": "every record carries the #111 SCORING_POLICY",
                   "G3_chronology": "frozen_at <= every training <= the first score",
                   "G4_checkpoint_binding": "joint hashes = handoff; specialist hashes = training records; training "
                                            "payload hash = the frozen hash",
                   "G5_replication": ("the joint's rows equal the Gate-B unit records (F-15-macro, every inventory and "
                                      "the wrong-anchor rows) and the #101 fixed-request records (nine requests)"),
                   "G6_training_integrity": "every member's schedule is its frozen schedule and every update is logged",
                   "G7_nonfinite": f"the joint's nine grid nonfinite shares <= {NONFINITE_SHARE_MAX}",
                   "failure": f"a failed guard turns every token into {STOP_TOKEN}"},
        "execution": {"workers": WORKERS, "training": smoke_evidence["projection"]["training"],
                      "scoring": smoke_evidence["projection"]["scoring"],
                      "rule": "each stage launched once (one process per GPU, PIDs recorded under logs/); no relaunch "
                              "over live workers; a crashed worker resumes its own assignment"},
        "caps": {"gpu_seconds": GPU_CAP_SECONDS, "elapsed_wall_seconds": WALL_CAP_SECONDS},
        "records": "records stay local (about 2-20 MB each); records-manifest.json (sha256 of every record and update "
                   "log) is committed",
        "prohibited": ["reading evaluation captures for training or selection", "re-scoring after publication",
                       "changing an arm, rule or recipe after the freeze", "retrying or replacing a typed failure"],
        "disclosure": ("Known before the freeze: Gate B (unit 0.7444, C* 0.7834, prior 0.7203 on grid), the #101 "
                       "per-request joint rows on this split (published by #101/#103), and every #117 LOLO / held-out "
                       "result (LC3 request-mean J - S +0.006 [-0.050, +0.062], F-15-macro +0.076). No specialist or "
                       "per-Delta model trained on the #109 fit split existed. The smoke rescored the joint's "
                       "F-15-macro on the evaluation cells only to reproduce the published Gate-B rows. Hardware: RTX "
                       "5090 instead of the #85 RTX 3090."),
        "smoke_evidence": smoke_evidence,
        "inputs": input_bindings(output),
        "claim_boundary": ("one look at the sealed #109 evaluation split; LC3 only (the #77 recipe B is not trained "
                           "here); grid primary, other inventories, the full budget and the controls DESCRIPTIVE; "
                           "tie-free cost; DESCRIPTIVE intervals; NovPhy only (item 5 deferred)"),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-117 ranking protocol")
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
    for name in ("records", "models", "training", "updates"):
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
    log(f"published: disposition {compute['disposition']['token']} {compute['disposition']['contrast_tokens']}; "
        f"guards pass {compute['guards_pass']}")
    return 0


def spot_checks(output):
    """Re-score the first usable seed of S-15-macro, D15 and S-1-continuous (per-request budget) on every grid cell
    in this process (exact rows)."""
    _, cells = gb.evaluation_set()
    anchors = gb.load_anchors(GATE_B)["anchors"]
    costs = p101.endpoint_costs()
    results = {}
    with harness.deterministic_scoring():
        for unit in ("S-15-macro/request", "D15/request", "S-1-continuous/request"):
            seed = next((s for s in SEEDS if not read_record(output, unit, s)["typed_failure"]), None)
            if seed is None:
                results[unit] = None
                continue
            record = read_record(output, unit, seed)
            model, _, _ = unit_model(output, unit, seed)
            same = True
            for request in unit_requests(unit):
                for member, entry in sorted(cells[PRIMARY]["mixed"].items()):
                    fresh = gb.request_block(model, request, anchors[member], entry["inventory"], costs)
                    same &= fresh["rows"] == record["requests"][request]["inventories"][PRIMARY][member]["rows"]
            results[unit] = {"seed": seed, "pass": bool(same)}
    return {"units": results, "pass": all(v is None or v["pass"] for v in results.values())}


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
    log(f"validation passed ({time.monotonic() - began:.0f}s): tables, summary, findings and the records manifest "
        f"re-derived and byte-compared; re-scored {spot['units']}; disposition {compute['disposition']['token']}")
    return 0


def dry_run(output):
    impl = implementation()
    _, cells = gb.evaluation_set()
    log(f"output {Path(output)}; mixed cells { {i: len(cells[i]['mixed']) for i in INVENTORIES} }")
    for arm in SPECIALIST_ARMS:
        log(f"  {arm}: requests {r117.arm_requests(arm)}; updates "
            f"{ {b: len(impl.schedule_of(arm, b)) for b in BUDGETS} }")
    log(f"  groups {len(groups())}; scored units {len(SCORED_UNITS)} x {len(SEEDS)} seeds; plan present "
        f"{(Path(output) / 'plan.json').is_file()}")
    return 0


MODES = {"dry-run": dry_run, "smoke": smoke, "prepare": prepare, "train": train, "score": score, "publish": publish,
         "validate": validate}
SHARDED = {"train", "score"}


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
