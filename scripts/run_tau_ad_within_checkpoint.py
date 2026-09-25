"""Issue-96 ADD-EXP tau_ad-W: within-checkpoint selection-schedule contrast.

Binding runner module: scripts/run_tau_ad_within_checkpoint.py
Exact validation command: python -u -m scripts.run_tau_ad_within_checkpoint --validate

Question: on the frozen #77 N1 hybrid checkpoints, does per-decision joint
(Delta, alpha) selection order engine-truth outcomes better than the best fixed
pair, holding checkpoint, anchor, inventory, endpoint (225), cost, tie rule,
typed-failure handling and verdicts identical so that only the per-step pair
schedule (step, carrier, remaining) -> pair varies?

Decision-only, zero engine seconds: no rendering, no capture, no retraining.
Every candidate of four retained engine-verdicted inventories (#93 drag grid
[primary], #93 offset sweep, #87 angle sweep, #94 launch-power sweep) is rolled
from the member's sealed anchor carrier to the 225-step endpoint under 18
schedules (14 hybrid-checkpoint arms, 4 continuous-checkpoint arms) and ranked
by the frozen TaskObjective count cost of the endpoint carrier.

Modes (frozen-protocol chronology, binding shared rule):
- --dry-run   no-write structural inventory (inventories, members, anchors,
              arms, cell counts); computes NO statistic and writes nothing.
- --smoke     pre-freeze controls: G1 replication of two #92 WP1b cells
              (hybrid-adaptive-e225 and hybrid-fixed-h15-e225, seed 20260908,
              issue-77-n1-001-a00: chosen ordinal identical, selected cost within
              1e-3), G2 anchor sha256 equality for every member, checkpoint
              identity equality against the #87 plan bindings, and schedule
              mechanics of all 18 arms on the G1 cell (no verdict join, no
              ranking retained); writes smoke.json.
- --prepare   freezes plan.json (arms, estimands, guards, decision rules, caps,
              disclosure, embedded smoke evidence) before any scoring run.
- --run       phase 1 (F x9, J, M, C-F x3, C-J), then the outcome-free J
              step-pair frequencies (frequencies.json), then phase 2 (T, D, OL);
              ledger-resumable; retained records are never recomputed.
- --publish   compute.json / summary.json / comparisons.csv / findings.md.
- --validate  re-derives frequencies, OL schedules and every published table
              from the retained records and byte-compares; no rollout.

All writes stay under the issue root (.local-artifacts/issue-96-tau-ad-within-
checkpoint-v1). Verdict vocabulary: supported / not_supported_by_this_experiment
/ readiness_or_precision_insufficient. Every interval is DESCRIPTIVE.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from scripts import run_launch_power_probe as p94
from scripts import run_lookahead_control as wlc
from scripts import run_second_parameterization_probe as p93
from scripts import run_selection_validity_restatement as wp1

ROOT = wlc.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-96-tau-ad-within-checkpoint-v1"
DYNAMICS = wlc.DYNAMICS
LOOKAHEAD = wlc.OUTPUT

IDENTITY = "issue-96-tau-ad-within-checkpoint-v1"
SCHEMA_PLAN = "issue_96_tau_ad_within_checkpoint_plan_v1"
SCHEMA_SMOKE = "issue_96_tau_ad_within_checkpoint_smoke_v1"
SCHEMA_RECORD = "issue_96_tau_ad_within_checkpoint_decision_v1"
SCHEMA_LEDGER = "issue_96_tau_ad_within_checkpoint_ledger_v1"
SCHEMA_FREQUENCIES = "issue_96_tau_ad_within_checkpoint_frequencies_v1"
SCHEMA_COMPUTE = "issue_96_tau_ad_within_checkpoint_compute_v1"
SCHEMA_REPORT = "issue_96_tau_ad_within_checkpoint_report_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_tau_ad_within_checkpoint --validate"

DEVICE = "cuda"
SEEDS = wlc.SEEDS
ENDPOINT = 225
INVENTORIES = ("grid", "offset", "angle", "power")
PRIMARY = "grid"
INVENTORY_SOURCES = {
    "grid": "#93 Arm B 4x4 Cartesian drag grid (16 candidates; #93 oracle verdicts)",
    "offset": "#93 Arm A offset angle sweep at radius 80 px (11 candidates; #93 oracle verdicts)",
    "angle": ("#87 N1 radius-80 launch-angle sweep of the member's canonical state (9-13 "
              "candidates; #87+#89 union verdict table)"),
    "power": ("#94 angle x launch-power sweep (20 candidates; #94 oracle verdicts; out of "
              "training support per #95, secondary only)"),
}

DELTA = 0.02
HALF_WIDTH_MAX = 0.05
NONFINITE_SHARE_MAX = 0.10
TIE_SHARE_MAX = 0.20
NONMODAL_SHARE_MIN = 0.10
RANK_DIVERGENCE_SHARE_MIN = 0.25
COST_TOLERANCE = 1e-3
OL_SEED = 7202
BOOTSTRAP_SEED = p94.BOOTSTRAP_SEED
BOOTSTRAP_DRAWS = p94.BOOTSTRAP_DRAWS
INTERVAL_QUANTILES = p94.INTERVAL_QUANTILES
GPU_CAP_SECONDS = 3 * 3600.0
WALL_CAP_SECONDS = 6 * 3600.0
STOP_TOKEN = "readiness_or_precision_insufficient"
DISPOSITION_TOKENS = ("supported", "not_supported_by_this_experiment", STOP_TOKEN)

G1_STATE = "issue-77-n1-001-a00"
G1_MEMBER = "issue-77-n1-001"
G1_SEED = 20260908
G1_CELLS = {"J": "decision--hybrid-adaptive-e225--seed20260908--issue-77-n1-001-a00",
            "F-15-continuous": "decision--hybrid-fixed-h15-e225--seed20260908--issue-77-n1-001-a00"}
G1_EXPECTED_J = {"chosen_ordinal": 5, "selected_predicted_cost": 2.564507}

PLACEHOLDER_MARKERS = ("TBD", "placeholder", "to be frozen", "XXX", "FIXME")

read_json = wlc.read_json
write_json = wlc.write_json
json_text = wlc.json_text
sha256_of = wlc.sha256_of
bootstrap = p94.bootstrap
cell_auc = p94.cell_auc
fmt = p94.fmt
fmt_interval = p94.fmt_interval


def log(message):
    print(f"[issue-96-tau-ad-within-checkpoint] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# pairs and arms
# ---------------------------------------------------------------------------

def check_pair_order():
    """PAIR_NAMES / DELTAS must match the frozen PAIRS (delta-major) and CONTINUOUS_PAIRS."""
    from world_model.training.cnn_hybrid import PAIRS
    from world_model.training.matched_dynamics import CONTINUOUS_PAIRS
    if tuple(f"{p.delta}-{p.abstraction}" for p in PAIRS) != PAIR_NAMES:
        raise ValueError("PAIRS order differs from the frozen delta-major pair order")
    if tuple(p.delta for p in CONTINUOUS_PAIRS) != DELTAS:
        raise ValueError("CONTINUOUS_PAIRS order differs from the frozen horizon order")


PAIR_NAMES = tuple(f"{d}-{a}" for d in (1, 5, 15) for a in ("continuous", "micro", "macro"))
DELTAS = (1, 5, 15)
ALPHAS = ("continuous", "micro", "macro")
FIXED_ARMS = tuple(f"F-{name}" for name in PAIR_NAMES)
CONTINUOUS_FIXED_ARMS = ("C-F-1", "C-F-5", "C-F-15")
PHASE_1 = (*FIXED_ARMS, "J", "M", *CONTINUOUS_FIXED_ARMS, "C-J")
PHASE_2 = ("T", "D", "OL")
ARMS = (*FIXED_ARMS, "J", "M", "T", "D", "OL", *CONTINUOUS_FIXED_ARMS, "C-J")
CONTROLLER_ARMS = ("J", "M", "T", "D", "C-J")

ARM_DEFINITIONS = {
    **{f"F-{name}": f"hybrid checkpoint, fixed pair ({name.replace('-', ', ')}) at every step"
       for name in PAIR_NAMES},
    "J": ("hybrid checkpoint, joint argmax over the nine MatchedController logits at every "
          "step (logits with Delta > remaining masked to -inf inside the controller; "
          "argmax ties to the lower PAIRS index) = #92 hybrid-adaptive-e225"),
    "M": ("hybrid checkpoint, factorized selector from the same logits, no training: "
          "Delta* = argmax_Delta logsumexp_alpha L[Delta, alpha], alpha* = argmax_alpha "
          "logsumexp_Delta L[Delta, alpha]; pair (Delta*, alpha*)"),
    "T": ("hybrid checkpoint, temporal-only: alpha fixed at alpha_T, Delta = argmax_Delta "
          "L[Delta, alpha_T]; alpha_T = the modal alpha of J's pooled per-seed step-pair "
          "frequencies (outcome-free)"),
    "D": ("hybrid checkpoint, description-only: Delta fixed at Delta_D, alpha = argmax_alpha "
          "L[Delta_D, alpha]; Delta_D = the modal Delta of J's pooled per-seed step-pair "
          "frequencies (outcome-free; 225 is divisible by every Delta so Delta_D is always "
          "feasible)"),
    "OL": ("hybrid checkpoint, open-loop: pairs drawn i.i.d. from J's pooled per-seed "
           "step-pair frequencies renormalized over the pairs feasible at the remaining "
           "horizon (uniform over feasible pairs if their pooled mass is zero); one draw "
           "sequence per cell shared by every candidate; one PCG64(7202) stream consumed in "
           "the frozen cell order (inventory, member, seed)"),
    **{f"C-F-{d}": f"continuous checkpoint, fixed pair ({d}, continuous) at every step"
       for d in DELTAS},
    "C-J": ("continuous checkpoint, joint argmax over its three MatchedController logits "
            "(horizon-only, complete control set) = #92 continuous-adaptive-e225"),
}


def arm_family(arm):
    return "continuous" if arm.startswith("C-") else "hybrid"


# ---------------------------------------------------------------------------
# universe: inventories, members, verdicts, anchors (no statistic)
# ---------------------------------------------------------------------------

def load_sources():
    plan93 = p93.load_plan(p93.OUTPUT)
    plan94 = p94.load_plan(p94.OUTPUT)
    plan87, decisions87, oracles87 = wp1.load_issue_87()
    summary89, records89 = wp1.load_issue_89()
    verdicts87, _, _, _ = wp1.build_verdicts(plan87, oracles87, records89)
    return {"plan93": plan93, "plan94": plan94, "plan87": plan87, "decisions87": decisions87,
            "oracles87": oracles87, "summary89": summary89, "records89": records89,
            "verdicts87": verdicts87,
            "oracle93": p93.load_oracle_records(p93.OUTPUT, plan93),
            "oracle94": p94.load_oracle_records(p94.OUTPUT, plan94)}


def first_removal_step(record):
    channel = (record or {}).get("engine_channel") or {}
    events = channel.get("pig_removed_events") or []
    if not channel.get("pig_removed") or not events:
        return None
    return min(event["fixed_step"] for event in events)


def angle_removal_steps(sources):
    """#87/#89 engine-channel pig-removal steps (the #92 WP1b coverage construction)."""
    best = {}
    for records in (sources["oracles87"], sources["records89"]):
        for identity in sorted(records):
            record = records[identity]
            step = first_removal_step(record)
            key = (record["cell"]["state"], record["cell"]["ordinal"])
            if step is not None and key not in best:
                best[key] = step
    return best


def build_universe(sources):
    plan93, plan94, plan87 = sources["plan93"], sources["plan94"], sources["plan87"]
    members = [state["member"] for state in plan93["states"]]
    canonical = {state["member"]: state["canonical_state"] for state in plan93["states"]}
    states87 = {state["identity"]: state for state in plan87["states"]}
    per89 = sources["summary89"]["ceiling_reestimate"]["per_state"]
    removal87 = angle_removal_steps(sources)
    universe = {inventory: {} for inventory in INVENTORIES}
    excluded = []
    for member in members:
        for arm in ("grid", "offset"):
            items = plan93["inventories"][arm]["members"][member]
            verdicts, failures, removal = {}, {}, {}
            for item in items:
                record = sources["oracle93"].get(p93.oracle_identity(member, arm, item["ordinal"]))
                if record is not None and record["outcome"] is not None:
                    verdicts[item["ordinal"]] = bool(record["outcome"]["first_shot_success"])
                    removal[item["ordinal"]] = first_removal_step(record)
                else:
                    failures[item["ordinal"]] = None if record is None else record.get("failure_kind")
            universe[arm][member] = {"state": canonical[member], "inventory": items,
                                     "verdicts": verdicts, "oracle_failures": failures,
                                     "removal_steps": removal}
        items = plan94["inventory"]["members"][member]
        verdicts, failures, removal = {}, {}, {}
        for item in items:
            record = sources["oracle94"].get(p94.oracle_identity(member, item["ordinal"]))
            if record is not None and record["outcome"] is not None:
                verdicts[item["ordinal"]] = bool(record["outcome"]["first_shot_success"])
                removal[item["ordinal"]] = first_removal_step(record)
            else:
                failures[item["ordinal"]] = None if record is None else record.get("failure_kind")
        universe["power"][member] = {"state": canonical[member], "inventory": items,
                                     "verdicts": verdicts, "oracle_failures": failures,
                                     "removal_steps": removal}
        state = canonical[member]
        if per89[state]["typed_unmeasurable"]:
            excluded.append({"inventory": "angle", "member": member, "state": state,
                             "rule": ("#89 frozen typed-unmeasurable declaration (no executed "
                                      "oracle anchor); inherited, not re-opened")})
            continue
        items = states87[state]["inventory"]
        verdicts, failures, removal = {}, {}, {}
        for item in items:
            key = (state, item["ordinal"])
            if key in sources["verdicts87"]:
                verdicts[item["ordinal"]] = sources["verdicts87"][key]
                removal[item["ordinal"]] = removal87.get(key)
            else:
                failures[item["ordinal"]] = "unresolved_in_union_table"
        universe["angle"][member] = {"state": state, "inventory": items, "verdicts": verdicts,
                                     "oracle_failures": failures, "removal_steps": removal}
    return members, universe, excluded


def mixed(entry):
    values = set(entry["verdicts"].values())
    return values == {True, False}


def scheduled_cells(universe):
    """Frozen cell order: inventory, member, seed; arms in ARMS order inside a cell."""
    return [(inventory, member, seed) for inventory in INVENTORIES
            for member in sorted(universe[inventory]) for seed in SEEDS]


def record_identity(inventory, member, seed, arm):
    return f"decision--{inventory}--{member}--seed{seed}--{arm}"


def record_path(output, inventory, member, seed, arm):
    return Path(output) / "records" / f"{record_identity(inventory, member, seed, arm)}.json"


def member_anchors(sources, members):
    """p94.frozen_anchor (the #93 anchor_retention rule) plus G2 equality evidence."""
    anchors, evidence = {}, []
    decisions93 = p93.load_decisions(p93.OUTPUT, sources["plan93"])
    decisions94 = p94.load_decisions(p94.OUTPUT, sources["plan94"])
    canonical = {state["member"]: state["canonical_state"] for state in sources["plan93"]["states"]}
    for member in members:
        anchor = p94.frozen_anchor(member)
        file_sha = sha256_of(anchor["frame_path"])
        used93 = sorted({record["anchor"]["sha256"] for identity, record in decisions93.items()
                         if record["cell"]["member"] == member and record["anchor"]})
        used94 = sorted({record["anchor"]["sha256"] for identity, record in decisions94.items()
                         if record["cell"]["member"] == member and record["anchor"]})
        used87 = sorted({record["anchor"]["sha256"] for record in sources["decisions87"].values()
                         if record["state_identity"] == canonical[member]
                         and not record.get("failure_kind")})
        ok = (file_sha == anchor["sha256"] and used93 == [anchor["sha256"]]
              and used94 == [anchor["sha256"]] and used87 in ([], [anchor["sha256"]]))
        evidence.append({"member": member, "source": anchor["source"], "cell": anchor["cell"],
                         "sha256_recorded": anchor["sha256"], "sha256_file": file_sha,
                         "issue93_decision_anchor_sha256": used93,
                         "issue94_decision_anchor_sha256": used94,
                         "issue87_canonical_state_anchor_sha256": used87, "ok": ok})
        anchors[member] = anchor
    return anchors, evidence


def checkpoint_identities(plan87):
    rows = []
    for seed in SEEDS:
        for family in ("hybrid", "continuous"):
            bound = plan87["dynamics_checkpoints"][str(seed)][family]
            predictor = Path(bound["path"])
            controller = DYNAMICS / f"seed-{seed}" / family / "controller.pt"
            rows.append({"seed": seed, "family": family, "predictor": str(predictor),
                         "predictor_sha256": sha256_of(predictor),
                         "predictor_identity_issue87": bound["identity"],
                         "controller": str(controller), "controller_sha256": sha256_of(controller),
                         "ok": sha256_of(predictor) == bound["identity"]})
    return rows


# ---------------------------------------------------------------------------
# schedules and the batched rollout
# ---------------------------------------------------------------------------

def rle(sequence):
    runs = []
    for value in sequence:
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([value, 1])
    return " ".join(f"{value}x{count}" for value, count in runs)


def unrle(text):
    sequence = []
    for token in text.split():
        value, count = token.split("x")
        sequence.extend([int(value)] * int(count))
    return sequence


def select_pairs(kind, logits, spec):
    if kind == "joint":
        return logits.argmax(-1)
    grid = logits.reshape(len(logits), len(DELTAS), len(ALPHAS))
    if kind == "factorized":
        delta = torch.logsumexp(grid, 2).argmax(1)
        alpha = torch.logsumexp(grid, 1).argmax(1)
        return delta * len(ALPHAS) + alpha
    if kind == "temporal":
        alpha = spec["alpha_index"]
        return grid[:, :, alpha].argmax(1) * len(ALPHAS) + alpha
    if kind == "description":
        delta = spec["delta_index"]
        return delta * len(ALPHAS) + grid[:, delta, :].argmax(1)
    raise ValueError(f"unknown schedule kind {kind}")


def arm_spec(arm, coordinates=None, schedule=None):
    if arm.startswith("F-"):
        return {"kind": "fixed", "pair_index": PAIR_NAMES.index(arm[2:])}
    if arm.startswith("C-F-"):
        return {"kind": "fixed", "pair_index": DELTAS.index(int(arm[4:]))}
    if arm in ("J", "C-J"):
        return {"kind": "joint"}
    if arm == "M":
        return {"kind": "factorized"}
    if arm == "T":
        return {"kind": "temporal", "alpha_index": coordinates["alpha_T_index"]}
    if arm == "D":
        return {"kind": "description", "delta_index": coordinates["delta_D_index"]}
    if arm == "OL":
        return {"kind": "schedule", "schedule": schedule}
    raise ValueError(f"unknown arm {arm}")


def rollout(model, controller, objective, carrier, inventory, spec, device):
    """Batched decision rollout to ENDPOINT; mirrors wlc.rollout_candidates exactly
    (residual carrier steps grouped by pair, nonfinite rows retired as typed
    failures) with the per-step pair map supplied by the arm's schedule spec."""
    from world_model.training.matched_dynamics import linear_macs, pairs_for
    pairs = pairs_for(model)
    count = len(inventory)
    z = carrier[None].to(device).expand(count, -1).clone()
    actions = torch.cat([wlc.action_tensor(item["action"], device) for item in inventory], 0)
    alive = torch.ones(count, dtype=torch.bool, device=z.device)
    positions = torch.zeros(count, dtype=torch.long, device=z.device)
    sequences = [[] for _ in inventory]
    pair_macs = [linear_macs(model, pair) for pair in pairs]
    controller_macs = (sum(m.in_features * m.out_features for m in controller.modules()
                           if isinstance(m, torch.nn.Linear)) if controller is not None else 0)
    transition_calls = controller_calls = macs = 0
    iteration = 0
    torch.cuda.synchronize()
    began = time.monotonic()
    with torch.no_grad():
        while bool(alive.any()):
            index = alive.nonzero(as_tuple=True)[0]
            if spec["kind"] == "fixed":
                chosen = torch.full((len(index),), spec["pair_index"], dtype=torch.long,
                                    device=z.device)
            elif spec["kind"] == "schedule":
                chosen = torch.full((len(index),), spec["schedule"][iteration], dtype=torch.long,
                                    device=z.device)
            else:
                remaining = ENDPOINT - positions[index]
                logits = controller(z[index], actions[index], remaining)
                chosen = select_pairs(spec["kind"], logits, spec)
                controller_calls += len(index)
            z_next = z
            for pair_index, pair in enumerate(pairs):
                rows = index[chosen == pair_index]
                if not len(rows):
                    continue
                z_next = z_next.clone()
                z_next[rows] = model.carrier(z[rows], actions[rows], pair)
                transition_calls += len(rows)
                macs += len(rows) * pair_macs[pair_index]
                for row in rows.tolist():
                    sequences[row].append(pair_index)
                positions[rows] += pair.delta
            z = z_next
            alive = alive & torch.isfinite(z).all(dim=1) & (positions < ENDPOINT)
            iteration += 1
    torch.cuda.synchronize()
    gpu_seconds = time.monotonic() - began
    macs += controller_calls * controller_macs
    ranking = []
    for slot, item in enumerate(inventory):
        failed = not bool(torch.isfinite(z[slot]).all())
        ranking.append({"ordinal": item["ordinal"], "branch_identity": item["branch_identity"],
                        "predicted_cost": None if failed else float(objective(z[slot])),
                        "excluded": failed,
                        "steps": sum(pairs[k].delta for k in sequences[slot])})
    return ranking, sequences, {"transition_calls": transition_calls,
                                "controller_calls": controller_calls,
                                "linear_macs": macs, "gpu_seconds": gpu_seconds}


def choose(ranking):
    return wlc.choose(ranking)


class Stack:
    """Lazily loaded frozen models, controllers and anchor carriers."""

    def __init__(self, sources, anchors):
        self.sources = sources
        self.anchors = anchors
        self.adapter = wlc.load_adapter(DEVICE)
        self.objective = wlc.load_objective()
        self.models, self.controllers, self.carriers = {}, {}, {}
        check_pair_order()

    def model(self, seed, family):
        key = (seed, family)
        if key not in self.models:
            self.models[key] = wlc.load_predictor(self.sources["plan87"], seed, family, DEVICE)
        return self.models[key]

    def controller(self, seed, family):
        key = (seed, family)
        if key not in self.controllers:
            self.controllers[key] = wlc.load_controller(seed, family, DEVICE)
        return self.controllers[key]

    def carrier(self, member):
        if member not in self.carriers:
            anchor = self.anchors[member]
            if sha256_of(anchor["frame_path"]) != anchor["sha256"]:
                raise ValueError(f"{STOP_TOKEN}: anchor of {member} changed")
            self.carriers[member] = wlc.encode_carrier(self.adapter, anchor, DEVICE)
        return self.carriers[member]

    def run(self, arm, seed, member, inventory, coordinates=None, schedule=None):
        family = arm_family(arm)
        spec = arm_spec(arm, coordinates, schedule)
        controller = self.controller(seed, family) if arm in CONTROLLER_ARMS else None
        return rollout(self.model(seed, family), controller, self.objective,
                       self.carrier(member), inventory, spec, DEVICE)


# ---------------------------------------------------------------------------
# outcome-free J frequencies, T/D coordinates, OL schedules
# ---------------------------------------------------------------------------

def j_frequencies(output, universe):
    counts = {seed: [0] * len(PAIR_NAMES) for seed in SEEDS}
    for inventory, member, seed in scheduled_cells(universe):
        record = read_record(output, inventory, member, seed, "J")
        for text in record["decision"]["schedules"].values():
            for index in unrle(text):
                counts[seed][index] += 1
    coordinates = {}
    for seed in SEEDS:
        by_alpha = [sum(counts[seed][d * len(ALPHAS) + a] for d in range(len(DELTAS)))
                    for a in range(len(ALPHAS))]
        by_delta = [sum(counts[seed][d * len(ALPHAS) + a] for a in range(len(ALPHAS)))
                    for d in range(len(DELTAS))]
        alpha = max(range(len(ALPHAS)), key=lambda a: (by_alpha[a], -a))
        delta = max(range(len(DELTAS)), key=lambda d: (by_delta[d], -d))
        coordinates[str(seed)] = {"alpha_T_index": alpha, "alpha_T": ALPHAS[alpha],
                                  "delta_D_index": delta, "delta_D": DELTAS[delta],
                                  "alpha_marginal_steps": by_alpha,
                                  "delta_marginal_steps": by_delta}
    return {"schema": SCHEMA_FREQUENCIES, "identity": IDENTITY,
            "rule": ("J step-pair counts pooled per seed over every scheduled cell of all four "
                     "inventories and every candidate (outcome-free: controller traces only); "
                     "alpha_T / Delta_D = modal marginal, ties to the lower index"),
            "pairs": list(PAIR_NAMES),
            "counts": {str(seed): counts[seed] for seed in SEEDS},
            "coordinates": coordinates}


def ol_schedules(universe, frequencies):
    from world_model.training.cnn_hybrid import PAIRS
    generator = np.random.Generator(np.random.PCG64(OL_SEED))
    schedules = {}
    for inventory, member, seed in scheduled_cells(universe):
        weights = frequencies["counts"][str(seed)]
        position, sequence = 0, []
        while position < ENDPOINT:
            feasible = [k for k, pair in enumerate(PAIRS) if pair.delta <= ENDPOINT - position]
            mass = np.array([weights[k] for k in feasible], dtype=float)
            if mass.sum() == 0:
                mass = np.ones(len(feasible))
            pick = feasible[int(generator.choice(len(feasible), p=mass / mass.sum()))]
            sequence.append(pick)
            position += PAIRS[pick].delta
        schedules[(inventory, member, seed)] = sequence
    return schedules


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

def read_record(output, inventory, member, seed, arm):
    path = record_path(output, inventory, member, seed, arm)
    record = read_json(path)
    if record.get("schema") != SCHEMA_RECORD or record.get("plan_identity") != IDENTITY:
        raise ValueError(f"record {path} binding differs")
    return record


def make_record(plan, anchors, inventory, member, seed, arm, entry, ranking, sequences, timing,
                wall, extra):
    chosen, failure = choose(ranking)
    schedules = None
    if arm in CONTROLLER_ARMS:
        schedules = {str(row["ordinal"]): rle(sequence)
                     for row, sequence in zip(ranking, sequences)}
    pair_counts = Counter(index for sequence in sequences for index in sequence)
    names = PAIR_NAMES if arm_family(arm) == "hybrid" else tuple(f"{d}-continuous" for d in DELTAS)
    anchor = anchors[member]
    return {
        "schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "plan_version": plan["version"],
        "cell": {"identity": record_identity(inventory, member, seed, arm), "inventory": inventory,
                 "member": member, "state": entry["state"], "seed": seed, "arm": arm,
                 "family": arm_family(arm), "endpoint": ENDPOINT},
        "anchor": {"sha256": anchor["sha256"], "source": anchor["source"], "cell": anchor["cell"]},
        "decision": {
            "candidate_count": len(ranking), "chosen_ordinal": chosen, "failure": failure,
            "selected_predicted_cost": next((row["predicted_cost"] for row in ranking
                                             if row["ordinal"] == chosen), None),
            "ranking": ranking, "schedules": schedules,
            "pair_step_counts": {names[k]: pair_counts[k] for k in sorted(pair_counts)},
            **extra, **timing},
        "wall_seconds": wall, "engine_seconds": 0, "issue_64_authorized": False,
    }


# ---------------------------------------------------------------------------
# smoke (pre-freeze)
# ---------------------------------------------------------------------------

def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the smoke is a pre-freeze control")
    sources = load_sources()
    members, universe, _ = build_universe(sources)
    anchors, anchor_evidence = member_anchors(sources, members)
    checkpoints = checkpoint_identities(sources["plan87"])
    stack = Stack(sources, anchors)
    entry = universe["angle"][G1_MEMBER]
    if entry["state"] != G1_STATE:
        raise ValueError("G1 state is not the canonical #87 state of member 001")
    checks = []
    fresh = {}
    for arm, identity in G1_CELLS.items():
        ranking, sequences, timing = stack.run(arm, G1_SEED, G1_MEMBER, entry["inventory"])
        chosen, _ = choose(ranking)
        cost = next(row["predicted_cost"] for row in ranking if row["ordinal"] == chosen)
        retained = read_json(LOOKAHEAD / "records" / f"{identity}.json")["decision"]
        ok = (chosen == retained["chosen_ordinal"]
              and abs(cost - retained["selected_predicted_cost"]) <= COST_TOLERANCE)
        if arm == "J":
            ok = ok and chosen == G1_EXPECTED_J["chosen_ordinal"] and abs(
                cost - G1_EXPECTED_J["selected_predicted_cost"]) <= COST_TOLERANCE
        fresh[arm] = sequences
        checks.append({"control": f"G1_replication_{arm}", "reference_record": identity,
                       "chosen_ordinal_fresh": chosen,
                       "chosen_ordinal_retained": retained["chosen_ordinal"],
                       "selected_cost_fresh": cost,
                       "selected_cost_retained": retained["selected_predicted_cost"],
                       "abs_cost_delta": abs(cost - retained["selected_predicted_cost"]),
                       "tolerance": COST_TOLERANCE, "gpu_seconds": timing["gpu_seconds"], "ok": ok})
        log(f"smoke G1 {arm}: chosen {chosen} vs retained {retained['chosen_ordinal']}; "
            f"|cost delta| {abs(cost - retained['selected_predicted_cost']):.2e}: {ok}")
    checks.append({"control": "G2_anchor_sha256_equality", "members": anchor_evidence,
                   "ok": all(row["ok"] for row in anchor_evidence)})
    checks.append({"control": "checkpoint_identity_equality", "checkpoints": checkpoints,
                   "ok": all(row["ok"] for row in checkpoints)})
    # schedule mechanics of every arm on the G1 cell; smoke-only T/D/OL inputs from the
    # fresh G1 J trace; no ranking is retained and no verdict is joined
    counts = Counter(index for sequence in fresh["J"] for index in sequence)
    weights = [counts[k] for k in range(len(PAIR_NAMES))]
    smoke_frequencies = {"counts": {str(seed): weights for seed in SEEDS}}
    by_alpha = [sum(weights[d * 3 + a] for d in range(3)) for a in range(3)]
    by_delta = [sum(weights[d * 3 + a] for a in range(3)) for d in range(3)]
    coordinates = {"alpha_T_index": max(range(3), key=lambda a: (by_alpha[a], -a)),
                   "delta_D_index": max(range(3), key=lambda d: (by_delta[d], -d))}
    schedule = ol_schedules({"grid": {G1_MEMBER: None}, "offset": {}, "angle": {}, "power": {}},
                            smoke_frequencies)[("grid", G1_MEMBER, G1_SEED)]
    mechanics = []
    for arm in ARMS:
        ranking, sequences, timing = stack.run(arm, G1_SEED, G1_MEMBER, entry["inventory"],
                                               coordinates, schedule)
        finite = [row for row in ranking if not row["excluded"]]
        steps_ok = all(row["steps"] == ENDPOINT for row in finite)
        names = PAIR_NAMES if arm_family(arm) == "hybrid" else tuple(f"{d}-continuous" for d in DELTAS)
        histogram = Counter(names[k] for sequence in sequences for k in sequence)
        mechanics.append({"arm": arm, "candidates": len(ranking), "nonfinite": len(ranking) - len(finite),
                          "every_finite_candidate_reaches_endpoint": steps_ok,
                          "pair_step_histogram": dict(sorted(histogram.items())),
                          "transition_calls": timing["transition_calls"],
                          "controller_calls": timing["controller_calls"],
                          "linear_macs": timing["linear_macs"],
                          "gpu_seconds": timing["gpu_seconds"], "ok": steps_ok})
    checks.append({"control": "schedule_mechanics_all_arms", "cell": f"angle/{G1_MEMBER}/seed{G1_SEED}",
                   "smoke_only_inputs": {"j_counts": weights, **coordinates,
                                         "ol_schedule": rle(schedule)},
                   "arms": mechanics, "ok": all(row["ok"] for row in mechanics)})
    # factorized-selector unit check against a brute-force python reference
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(64, 9, generator=generator)
    logits[:8, 6:] = -torch.inf
    logits[8:16, 3:] = -torch.inf
    reference = []
    for row in logits.tolist():
        grid = [[row[d * 3 + a] for a in range(3)] for d in range(3)]
        lse = lambda values: (-math.inf if all(v == -math.inf for v in values)
                              else math.log(sum(math.exp(v) for v in values if v != -math.inf)))
        delta = max(range(3), key=lambda d: (lse(grid[d]), -d))
        alpha = max(range(3), key=lambda a: (lse([grid[d][a] for d in range(3)]), -a))
        reference.append(delta * 3 + alpha)
    factorized_ok = select_pairs("factorized", logits, {}).tolist() == reference
    checks.append({"control": "factorized_selector_reference", "rows": 64, "ok": factorized_ok})
    projection = sum(row["gpu_seconds"] for row in mechanics) * len(scheduled_cells(universe))
    ok = all(check["ok"] for check in checks)
    evidence = {"schema": SCHEMA_SMOKE, "identity": IDENTITY, "run_at": utc_now(), "device": DEVICE,
                "checks": checks, "projected_run_gpu_seconds": projection,
                "reading": ("infrastructure assertions only; no AUC, top-1 or other outcome "
                            "statistic of this contrast was computed"), "ok": ok}
    write_json(output / "smoke.json", evidence)
    log(f"smoke {'PASSED' if ok else 'FAILED'}; projected run gpu {projection:.0f}s")
    if not ok:
        raise ValueError("smoke control differs; STOP RULE: abort before the freeze")
    return 0


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def input_bindings(output):
    names = {
        "issue_87_plan": wlc.EIGHTY_SEVEN / "plan.json",
        "issue_89_summary": wlc.EIGHTY_NINE / "summary.json",
        "issue_92_lookahead_plan": LOOKAHEAD / "plan.json",
        "issue_92_g1_reference_J": LOOKAHEAD / "records" / f"{G1_CELLS['J']}.json",
        "issue_92_g1_reference_F15": LOOKAHEAD / "records" / f"{G1_CELLS['F-15-continuous']}.json",
        "issue_93_plan": p93.OUTPUT / "plan.json",
        "issue_93_anchor_retention": p93.OUTPUT / "anchor_retention.json",
        "issue_93_summary": p93.OUTPUT / "summary.json",
        "issue_94_plan": p94.OUTPUT / "plan.json",
        "issue_94_summary": p94.OUTPUT / "summary.json",
        "issue_77_dynamics_plan": DYNAMICS / "plan.json",
        "smoke": Path(output) / "smoke.json",
    }
    return [{"name": name, "artifact": str(path), "sha256": sha256_of(path)}
            for name, path in names.items()]


def frozen_plan(output, frozen_at, smoke_evidence, universe, members, excluded):
    cells = scheduled_cells(universe)
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "role": "terminal",
        "frozen_at": frozen_at, "frozen_before_scoring_run": True, "issue": 96,
        "issue_64_authorized": False, "validation_command": VALIDATION_COMMAND,
        "runner": "scripts/run_tau_ad_within_checkpoint.py",
        "runner_sha256_at_freeze": sha256_of(ROOT / "scripts/run_tau_ad_within_checkpoint.py"),
        "engine_seconds": 0,
        "question": ("on the frozen #77 N1 hybrid checkpoints, does per-decision joint (Delta, "
                     "alpha) selection order engine-truth outcomes better than the best fixed pair, "
                     "holding checkpoint, anchors, inventory, endpoint, cost and verdicts identical "
                     "so that only the per-step pair schedule varies?"),
        "disclosure": ("The pre-freeze Oracle review read #92's J step-pair frequencies "
                       "(horizon_trace, trace-only, outcome-free) and F*'s cross-fit inventories "
                       "are chosen from published fixed-arm results; no outcome statistic of this "
                       "contrast was computed before freeze. The #92 G1 reference records are read "
                       "for chosen_ordinal and selected_predicted_cost only."),
        "held_identical": {
            "checkpoints": ("per-seed predictor.pt bound by issue-87 plan.json dynamics_checkpoints "
                            "(sha256 equality verified at smoke); the same controller.pt for J/M/T/D "
                            "and for OL's frequencies"),
            "anchors": ("the member's sealed anchor carrier by the #93 anchor_retention rule "
                        "(run_launch_power_probe.frozen_anchor), sha256-checked at every load; "
                        "one anchor per member for every inventory and arm"),
            "inventory_and_normalization": ("the frozen source inventories; action tensor "
                                            "(drag_x/480, drag_y/480, release/1000, tap/1000, 1)"),
            "endpoint": ENDPOINT, "cost": "the frozen TaskObjective count cost of the endpoint carrier",
            "tie_rule": "argmin predicted cost, ties to the lower ordinal",
            "typed_failures": ("a nonfinite carrier is a retained typed failure (excluded from the "
                               "ranking, inherits no verdict), never retried"),
            "verdicts": ("#93 oracle table (grid, offset), #94 oracle table (power), #87+#89 union "
                         "table (angle); keyed by candidate identity"),
            "varies": "only the map (step, carrier, remaining) -> pair",
            "compute": "differs by schedule; reported, not matched",
        },
        "universe": {
            "inventories": INVENTORY_SOURCES, "primary": PRIMARY,
            "members": {inventory: sorted(universe[inventory]) for inventory in INVENTORIES},
            "mixed_members": {inventory: sorted(m for m, e in universe[inventory].items() if mixed(e))
                              for inventory in INVENTORIES},
            "excluded": excluded, "seeds": list(SEEDS), "arms": list(ARMS),
            "arm_definitions": ARM_DEFINITIONS, "phase_1": list(PHASE_1), "phase_2": list(PHASE_2),
            "pairs_order": list(PAIR_NAMES),
            "scheduled_cells": len(cells), "records": len(cells) * len(ARMS),
            "mixed_rule": ("a (inventory, member) is mixed iff its retained verdicts hold at least "
                           "one success and one failure; cells = mixed member x 3 seeds"),
        },
        "estimands": {
            "cell_auc": ("within-cell P(successful candidate has lower predicted cost than failed "
                         "candidate), ties 0.5, over admissible candidates (retained verdict AND "
                         "finite predicted cost); the tie-correct definition "
                         "run_launch_power_probe.cell_auc is used because "
                         "run_cross_pool_selection_audit.cell_auc scores a tied pair as -0.5 when "
                         "the failure precedes the success in cost-then-ordinal order (verified: "
                         "tie fail-first -> -0.5, tie success-first -> 0.5)"),
            "primary": ("theta_grid = member-clustered mean over mixed grid cells of AUC_J - AUC_F*, "
                        "F* = the cross-fitted best fixed pair"),
            "cross_fit": ("F*(I) = the F(Delta, alpha) with the highest mean, over the other three "
                          "inventories, of the inventory's member-clustered mean AUC (inventories "
                          "equally weighted); ties by PAIRS order; C-F*(I) likewise over C-F"),
            "F_hindsight": "the F pair with the highest member-clustered mean AUC on I itself (in-sample)",
            "contrast_cells": ("cells where both arms' AUCs are defined; undefined cells counted "
                               "and reported"),
            "interval": (f"member-clustered percentile bootstrap, {BOOTSTRAP_DRAWS} draws, PCG64 seed "
                         f"{BOOTSTRAP_SEED}, quantiles {list(INTERVAL_QUANTILES)}, DESCRIPTIVE"),
            "secondaries": ["J - F_hindsight", "J - F(1, macro)", "J - F(1, continuous)", "J - M",
                            "J - OL", "J - T", "J - D", "C-J - C-F*",
                            "top-1 J - F* (pooled descriptive, power note)",
                            "fixed-pair spread max_F - min_F per cell"],
            "per_inventory": "every estimand per inventory, never pooled across inventories",
            "coverage": ("per inventory: successful slots of mixed members whose first engine-channel "
                         "pig-removal step lies within 225 agent frames of the decision frame "
                         "((fixed_step - 30000)/50 <= 225)"),
            "compute_accounting": "per arm: transition calls, controller calls, linear MACs, GPU seconds",
            "power_note": ("top-1 over ~42 grid cells against chance ~0.06-0.10 cannot resolve "
                           "differences below ~0.15; descriptive only"),
        },
        "margin": {"delta": DELTA, "unit": "within-cell AUC difference"},
        "guards": {
            "G1": ("replication: fresh J and F(15, continuous) on angle/issue-77-n1-001 seed 20260908 "
                   "reproduce #92 decision--hybrid-adaptive-e225 (chosen 5, cost 2.564507) and "
                   "decision--hybrid-fixed-h15-e225: ordinal identical, cost within 1e-3 (smoke and "
                   "retained run records)"),
            "G2": "anchor sha256 equality (file == recorded == #93/#94/#87 decision anchors)",
            "G3": (f"verdict resolution 1.0 on admissible candidates of mixed grid cells for every "
                   f"arm; nonfinite typed-failure share <= {NONFINITE_SHARE_MAX}"),
            "G4": f"<= {TIE_SHARE_MAX} of mixed grid cells have J or F* fully tied",
            "G5": (f"J's non-modal step share >= {NONMODAL_SHARE_MIN} on mixed grid cells AND "
                   f"Kendall tau-b(J, F(modal)) < 1 on >= {RANK_DIVERGENCE_SHARE_MIN} of mixed grid "
                   "cells (modal = J's modal step pair over mixed grid cells)"),
            "G6": f"primary interval half-width <= {HALF_WIDTH_MAX}",
            "caps": f"run GPU seconds <= {GPU_CAP_SECONDS}, run wall seconds <= {WALL_CAP_SECONDS}",
            "failure": f"any guard failure -> {STOP_TOKEN}",
        },
        "decision_rule": {
            "supported": "all guards pass AND theta_hat >= delta AND 95% lower bound > 0",
            "not_supported_by_this_experiment": "all guards pass AND 95% upper bound < delta",
            "readiness_or_precision_insufficient": "otherwise",
            "secondary_tokens": ("J - M and J - OL on the grid: the same three-token rule and delta; "
                                 "guards G1-G5 and caps as evaluated for the primary, G6 applied to "
                                 "the secondary's own interval"),
            "replication": ("J - F* on offset/angle/power reported verbatim with the count meeting "
                            "theta >= delta AND lower > 0; does not change the token"),
            "interpretation": ("if the member-mean fixed-pair spread on the grid is < delta, a "
                               "not_supported result is written as 'the selection schedule is "
                               "near-inert on this checkpoint', not as 'adaptivity does not help'"),
        },
        "licenses": {
            "supported": ("on this checkpoint, per-decision joint selection orders engine-truth "
                          "outcomes better than the best fixed pair by >= 0.02 within-cell AUC - an "
                          "ordering claim, decision-only, scoped to the frozen N1 membership; top-1 "
                          "stays descriptive and underpowered; jointness/state-dependence claims "
                          "require J - M / J - OL to pass"),
            "not_supported_healthy_spread": ("per-decision selection does not help on this checkpoint, "
                                             "which strengthens the existing measurement story"),
            "near_inert": "no selection-based claim in either direction",
            "insufficient": ("tau_ad stays unmeasured-in-effect; the five 'unmeasured' manuscript "
                             "sentences are still corrected to the accurate three-reading statement"),
        },
        "prohibited": ["computing any AUC or top-1 statistic of this contrast from the #92 "
                       "hybrid-adaptive records before freeze",
                       "adding endpoint 600 or a pooled estimate after seeing results (no e600 "
                       "column is declared)", "retrying typed failures",
                       "re-opening any prior disposition (#15, #87, #89, #90, #92-#95)"],
        "caps": {"gpu_seconds": GPU_CAP_SECONDS, "wall_seconds": WALL_CAP_SECONDS, "engine_seconds": 0,
                 "stop": STOP_TOKEN},
        "smoke_evidence": smoke_evidence,
        "inputs": input_bindings(output),
        "claim_boundary": ("decision-only rollouts of the frozen #77 N1 checkpoints from the sealed "
                           "member anchors to endpoint 225 on four retained engine-verdicted "
                           "inventories; the controller selects the prediction pair from a policy "
                           "trained by dynamic programming on the same frozen corpus, never from "
                           "task outcomes; the record bounds decision-time pair-schedule adaptivity "
                           "on this checkpoint, not policy learning; no engine access, rendering or "
                           "retraining; prior dispositions are inputs and are not re-opened; "
                           "inventories are never pooled; every interval is DESCRIPTIVE"),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-96 protocol")
    if not plan.get("frozen_before_scoring_run") or not plan["smoke_evidence"].get("ok"):
        raise ValueError("plan.json does not declare a passed-smoke pre-scoring freeze")
    blob = json_text({k: v for k, v in plan.items() if k != "smoke_evidence"})
    for marker in PLACEHOLDER_MARKERS:
        if marker in blob:
            raise ValueError(f"frozen plan contains a missing-value marker {marker!r}")
    for entry in plan["inputs"]:
        if entry["name"] != "smoke" and sha256_of(ROOT / entry["artifact"]) != entry["sha256"]:
            raise ValueError(f"frozen input {entry['name']} changed after the freeze")
    return plan


def prepare(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        load_plan(output)
        log("existing frozen plan validated")
        return 0
    if (output / "records").exists():
        raise ValueError("records exist before the freeze; refusing")
    evidence = read_json(output / "smoke.json")
    if not evidence.get("ok"):
        raise ValueError("smoke failed; STOP RULE: no freeze")
    sources = load_sources()
    members, universe, excluded = build_universe(sources)
    plan = frozen_plan(output, utc_now(), evidence, universe, members, excluded)
    (output / "plan.json").write_text(json_text(plan))
    load_plan(output)
    log(f"frozen plan published: {plan['universe']['records']} arm-cell records scheduled; no "
        "outcome statistic computed")
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def load_ledger(output):
    path = Path(output) / "ledger.json"
    if path.is_file():
        return read_json(path)
    return {"schema": SCHEMA_LEDGER, "identity": IDENTITY, "status": "running", "records": 0,
            "gpu_seconds_elapsed": 0.0, "wall_seconds_elapsed": 0.0}


def run(output):
    output = Path(output)
    plan = load_plan(output)
    sources = load_sources()
    members, universe, _ = build_universe(sources)
    if len(scheduled_cells(universe)) * len(ARMS) != plan["universe"]["records"]:
        raise ValueError("universe differs from the frozen plan")
    anchors, evidence = member_anchors(sources, members)
    if not all(row["ok"] for row in evidence):
        raise ValueError(f"{STOP_TOKEN}: anchor sha256 equality failed")
    stack = Stack(sources, anchors)
    ledger = load_ledger(output)
    began = time.monotonic()
    wall_base = ledger["wall_seconds_elapsed"]

    def execute(arms, coordinates=None, schedules=None):
        for inventory, member, seed in scheduled_cells(universe):
            entry = universe[inventory][member]
            for arm in arms:
                path = record_path(output, inventory, member, seed, arm)
                if path.is_file():
                    read_record(output, inventory, member, seed, arm)
                    continue
                if (ledger["gpu_seconds_elapsed"] > GPU_CAP_SECONDS
                        or wall_base + time.monotonic() - began > WALL_CAP_SECONDS):
                    ledger["status"] = "cap_exceeded"
                    write_json(output / "ledger.json", ledger)
                    raise ValueError(f"{STOP_TOKEN}: compute cap exceeded")
                started = time.monotonic()
                cell_coordinates = None if coordinates is None else coordinates[str(seed)]
                schedule = None if schedules is None else schedules[(inventory, member, seed)]
                ranking, sequences, timing = stack.run(arm, seed, member, entry["inventory"],
                                                       cell_coordinates, schedule)
                extra = {}
                if arm == "T":
                    extra = {"alpha_T": cell_coordinates["alpha_T"]}
                elif arm == "D":
                    extra = {"delta_D": cell_coordinates["delta_D"]}
                elif arm == "OL":
                    extra = {"ol_schedule": rle(schedule)}
                record = make_record(plan, anchors, inventory, member, seed, arm, entry, ranking,
                                     sequences, timing, time.monotonic() - started, extra)
                write_json(path, record)
                ledger["records"] += 1
                ledger["gpu_seconds_elapsed"] += timing["gpu_seconds"]
                ledger["wall_seconds_elapsed"] = wall_base + time.monotonic() - began
                if ledger["records"] % 100 == 0:
                    write_json(output / "ledger.json", ledger)
                    log(f"records {ledger['records']}/{plan['universe']['records']}; gpu "
                        f"{ledger['gpu_seconds_elapsed']:.0f}s; wall {ledger['wall_seconds_elapsed']:.0f}s")

    execute(PHASE_1)
    frequencies = j_frequencies(output, universe)
    frequency_path = output / "frequencies.json"
    if frequency_path.is_file() and read_json(frequency_path) != json.loads(json_text(frequencies)):
        raise ValueError("retained frequencies.json differs from the J records")
    write_json(frequency_path, frequencies)
    log(f"J frequencies per seed: {frequencies['counts']}; coordinates "
        f"{ {s: (c['alpha_T'], c['delta_D']) for s, c in frequencies['coordinates'].items()} }")
    execute(PHASE_2, frequencies["coordinates"], ol_schedules(universe, frequencies))
    ledger["status"] = "terminal"
    ledger["wall_seconds_elapsed"] = wall_base + time.monotonic() - began
    write_json(output / "ledger.json", ledger)
    log(f"run complete: {ledger['records']} records; gpu {ledger['gpu_seconds_elapsed']:.0f}s; wall "
        f"{ledger['wall_seconds_elapsed']:.0f}s")
    return 0


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def kendall(x, y):
    from scipy.stats import kendalltau
    if len(x) < 2:
        return None
    value = kendalltau(x, y).statistic
    return None if value is None or math.isnan(value) else float(value)


def cell_view(entry, record):
    costs = {row["ordinal"]: row["predicted_cost"] for row in record["decision"]["ranking"]}
    verdicts = entry["verdicts"]
    admissible = sorted(((o, costs[o]) for o in verdicts if costs.get(o) is not None),
                        key=lambda pair: (pair[1], pair[0]))
    pos = [c for o, c in admissible if verdicts[o]]
    neg = [c for o, c in admissible if not verdicts[o]]
    return {"resolved": sum(1 for o in verdicts if o in costs), "verdict_slots": len(verdicts),
            "nonfinite": sum(1 for o in verdicts if o in costs and costs[o] is None),
            "admissible": admissible,
            "auc": cell_auc(pos, neg) if pos and neg else None,
            "top1_hit": bool(verdicts[admissible[0][0]]) if admissible else None,
            "chance_top1": len(pos) / len(admissible) if admissible else None,
            "all_tied": len({c for _, c in admissible}) <= 1}


def member_mean(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        if row[key] is not None:
            grouped[row["member"]].append(row[key])
    if not grouped:
        return None
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def contrast(rows, first, second, key="auc"):
    values, clusters, undefined = [], [], 0
    for row in rows:
        a, b = row["arms"][first][key], row["arms"][second][key]
        if a is None or b is None:
            undefined += 1
            continue
        values.append(float(a) - float(b))
        clusters.append(row["member"])
    block = bootstrap(values, clusters)
    if block is not None:
        block["half_width"] = (block["interval"][1] - block["interval"][0]) / 2
    return {"first": first, "second": second, "cells": len(values), "undefined_cells": undefined,
            "members": len(set(clusters)), "estimate": block}


def token(block, guards_ok):
    if not guards_ok or block is None:
        return STOP_TOKEN
    low, high = block["interval"]
    if block["mean"] >= DELTA and low > 0:
        return "supported"
    if high < DELTA:
        return "not_supported_by_this_experiment"
    return STOP_TOKEN


def inventory_rows(output, universe, inventory):
    rows = []
    for member in sorted(universe[inventory]):
        entry = universe[inventory][member]
        if not mixed(entry):
            continue
        for seed in SEEDS:
            arms = {}
            for arm in ARMS:
                record = read_record(output, inventory, member, seed, arm)
                arms[arm] = cell_view(entry, record)
                if arm == "J":
                    arms[arm]["schedules"] = record["decision"]["schedules"]
                    arms[arm]["costs"] = {row["ordinal"]: row["predicted_cost"]
                                          for row in record["decision"]["ranking"]}
                if arm.startswith("F-"):
                    arms[arm]["costs"] = {row["ordinal"]: row["predicted_cost"]
                                          for row in record["decision"]["ranking"]}
            rows.append({"member": member, "seed": seed, "arms": arms})
    return rows


def coverage(universe, inventory):
    slots = reached = unmeasured = 0
    for member, entry in universe[inventory].items():
        if not mixed(entry):
            continue
        for ordinal, hit in entry["verdicts"].items():
            if not hit:
                continue
            slots += 1
            step = entry["removal_steps"].get(ordinal)
            if step is None:
                unmeasured += 1
            elif (step - p93.DECISION_FIXED_STEP) / p93.NATIVE_STRIDE <= ENDPOINT:
                reached += 1
    return {"successful_slots": slots, "reached_within_225": reached,
            "resolution_step_unmeasured": unmeasured}


def compute_tables(output, plan):
    output = Path(output)
    sources = load_sources()
    members, universe, _ = build_universe(sources)
    frequencies = j_frequencies(output, universe)
    if read_json(output / "frequencies.json") != json.loads(json_text(frequencies)):
        raise ValueError("frequencies.json differs from the retained J records")
    schedules = ol_schedules(universe, frequencies)
    records = 0
    accounting = {arm: {"records": 0, "transition_calls": 0, "controller_calls": 0,
                        "linear_macs": 0, "gpu_seconds": 0.0} for arm in ARMS}
    nonfinite_all = {inventory: {"nonfinite": 0, "candidates": 0} for inventory in INVENTORIES}
    for inventory, member, seed in scheduled_cells(universe):
        for arm in ARMS:
            record = read_record(output, inventory, member, seed, arm)
            decision = record["decision"]
            if arm == "OL" and decision["ol_schedule"] != rle(schedules[(inventory, member, seed)]):
                raise ValueError(f"OL schedule of {inventory}/{member}/{seed} differs from the frozen draw")
            if arm == "T" and decision["alpha_T"] != frequencies["coordinates"][str(seed)]["alpha_T"]:
                raise ValueError("T coordinate differs from frequencies.json")
            if arm == "D" and decision["delta_D"] != frequencies["coordinates"][str(seed)]["delta_D"]:
                raise ValueError("D coordinate differs from frequencies.json")
            block = accounting[arm]
            block["records"] += 1
            for key in ("transition_calls", "controller_calls", "linear_macs", "gpu_seconds"):
                block[key] += decision[key]
            nonfinite_all[inventory]["candidates"] += decision["candidate_count"]
            nonfinite_all[inventory]["nonfinite"] += sum(row["excluded"] for row in decision["ranking"])
            records += 1
    if records != plan["universe"]["records"]:
        raise ValueError(f"record count {records} differs from the frozen {plan['universe']['records']}")
    ledger = read_json(output / "ledger.json")

    tables = {}
    member_means = {}
    for inventory in INVENTORIES:
        rows = inventory_rows(output, universe, inventory)
        tables[inventory] = rows
        member_means[inventory] = {arm: member_mean([{"member": r["member"], "auc": r["arms"][arm]["auc"]}
                                                     for r in rows], "auc") for arm in ARMS}

    def cross_fit(inventory, arms):
        scores = {}
        for arm in arms:
            others = [member_means[other][arm] for other in INVENTORIES
                      if other != inventory and member_means[other][arm] is not None]
            scores[arm] = float(np.mean(others)) if others else None
        best = max(arms, key=lambda arm: (-math.inf if scores[arm] is None else scores[arm],
                                          -arms.index(arm)))
        return best, scores

    per_inventory = {}
    for inventory in INVENTORIES:
        rows = tables[inventory]
        f_star, f_scores = cross_fit(inventory, list(FIXED_ARMS))
        c_star, c_scores = cross_fit(inventory, list(CONTINUOUS_FIXED_ARMS))
        hindsight = max(FIXED_ARMS, key=lambda arm: (
            -math.inf if member_means[inventory][arm] is None else member_means[inventory][arm],
            -FIXED_ARMS.index(arm)))
        contrasts = {
            "J-F*": contrast(rows, "J", f_star),
            "J-F_hindsight": contrast(rows, "J", hindsight),
            "J-F(1,macro)": contrast(rows, "J", "F-1-macro"),
            "J-F(1,continuous)": contrast(rows, "J", "F-1-continuous"),
            "J-M": contrast(rows, "J", "M"), "J-OL": contrast(rows, "J", "OL"),
            "J-T": contrast(rows, "J", "T"), "J-D": contrast(rows, "J", "D"),
            "C-J-C-F*": contrast(rows, "C-J", c_star),
        }
        top1 = contrast(rows, "J", f_star, key="top1_hit")
        spread_values = []
        for row in rows:
            aucs = [row["arms"][arm]["auc"] for arm in FIXED_ARMS if row["arms"][arm]["auc"] is not None]
            if aucs:
                spread_values.append({"member": row["member"], "spread": max(aucs) - min(aucs)})
        spread = bootstrap([r["spread"] for r in spread_values], [r["member"] for r in spread_values])
        arm_auc = {arm: bootstrap([r["arms"][arm]["auc"] for r in rows if r["arms"][arm]["auc"] is not None],
                                  [r["member"] for r in rows if r["arms"][arm]["auc"] is not None])
                   for arm in ARMS}
        hits = {arm: sum(1 for r in rows if r["arms"][arm]["top1_hit"]) for arm in ("J", f_star)}
        chance = [r["arms"]["J"]["chance_top1"] for r in rows if r["arms"]["J"]["chance_top1"] is not None]
        # guard ingredients
        resolved = sum(r["arms"][arm]["resolved"] for r in rows for arm in ARMS)
        slots = sum(r["arms"][arm]["verdict_slots"] for r in rows for arm in ARMS)
        nonfinite = sum(r["arms"][arm]["nonfinite"] for r in rows for arm in ARMS)
        tied = sum(1 for r in rows if r["arms"]["J"]["all_tied"] or r["arms"][f_star]["all_tied"])
        steps = Counter()
        for r in rows:
            for text in r["arms"]["J"]["schedules"].values():
                steps.update(unrle(text))
        total_steps = sum(steps.values())
        modal_index = max(range(len(PAIR_NAMES)), key=lambda k: (steps[k], -k)) if total_steps else 0
        modal_arm = f"F-{PAIR_NAMES[modal_index]}"
        divergent, taus = 0, []
        for r in rows:
            j_costs, f_costs = r["arms"]["J"]["costs"], r["arms"][modal_arm]["costs"]
            common = sorted(o for o in r["arms"]["J"]["costs"]
                            if o in universe[inventory][r["member"]]["verdicts"]
                            and j_costs[o] is not None and f_costs.get(o) is not None)
            tau = kendall([j_costs[o] for o in common], [f_costs[o] for o in common])
            taus.append(tau)
            divergent += int(tau is not None and tau < 1.0 - 1e-12)
        per_inventory[inventory] = {
            "source": INVENTORY_SOURCES[inventory],
            "members": len(universe[inventory]),
            "mixed_members": sorted({r["member"] for r in rows}),
            "mixed_cells": len(rows), "coverage": coverage(universe, inventory),
            "F_star": f_star, "F_star_cross_fit_scores": f_scores,
            "C_F_star": c_star, "C_F_star_cross_fit_scores": c_scores,
            "F_hindsight": hindsight, "arm_member_mean_auc": member_means[inventory],
            "arm_auc": arm_auc, "contrasts": contrasts,
            "top1": {"J_hits": hits["J"], "F_star_hits": hits[f_star], "cells": len(rows),
                     "chance_mean": float(np.mean(chance)) if chance else None,
                     "difference": top1,
                     "power_note": plan["estimands"]["power_note"]},
            "fixed_pair_spread": spread,
            "guard_inputs": {
                "verdict_resolution": resolved / slots if slots else None,
                "nonfinite_share": nonfinite / slots if slots else None,
                "nonfinite_all_candidates": nonfinite_all[inventory],
                "tied_cells": tied, "tied_share": tied / len(rows) if rows else None,
                "J_modal_pair": PAIR_NAMES[modal_index],
                "J_step_counts": {PAIR_NAMES[k]: steps[k] for k in sorted(steps)},
                "J_nonmodal_step_share": 1 - steps[modal_index] / total_steps if total_steps else None,
                "kendall_tau_J_vs_F_modal": taus,
                "rank_divergent_cells": divergent,
                "rank_divergent_share": divergent / len(rows) if rows else None,
            },
            "rows": [{"member": r["member"], "seed": r["seed"],
                      "auc": {arm: r["arms"][arm]["auc"] for arm in ARMS},
                      "top1_hit": {arm: r["arms"][arm]["top1_hit"] for arm in ARMS},
                      "admissible": len(r["arms"]["J"]["admissible"])} for r in rows],
        }

    grid = per_inventory[PRIMARY]
    gi = grid["guard_inputs"]
    primary = grid["contrasts"]["J-F*"]["estimate"]
    smoke_g1 = [c for c in plan["smoke_evidence"]["checks"] if c["control"].startswith("G1_")]
    g1_run = []
    for arm, identity in G1_CELLS.items():
        retained = read_json(LOOKAHEAD / "records" / f"{identity}.json")["decision"]
        mine = read_record(output, "angle", G1_MEMBER, G1_SEED, arm)["decision"]
        g1_run.append({"arm": arm, "reference": identity, "chosen": mine["chosen_ordinal"],
                       "reference_chosen": retained["chosen_ordinal"],
                       "abs_cost_delta": abs(mine["selected_predicted_cost"]
                                             - retained["selected_predicted_cost"]),
                       "ok": (mine["chosen_ordinal"] == retained["chosen_ordinal"]
                              and abs(mine["selected_predicted_cost"]
                                      - retained["selected_predicted_cost"]) <= COST_TOLERANCE)})
    _, anchor_evidence = member_anchors(sources, members)
    guards = {
        "G1": {"observed": {"smoke": [{k: c[k] for k in ("control", "chosen_ordinal_fresh",
                                                          "chosen_ordinal_retained", "abs_cost_delta")}
                                      for c in smoke_g1], "run_records": g1_run},
               "pass": all(c["ok"] for c in smoke_g1) and all(r["ok"] for r in g1_run)},
        "G2": {"observed": {"members": len(anchor_evidence),
                            "equal": sum(r["ok"] for r in anchor_evidence)},
               "pass": all(r["ok"] for r in anchor_evidence)},
        "G3": {"observed": {"verdict_resolution": gi["verdict_resolution"],
                            "nonfinite_share": gi["nonfinite_share"]},
               "pass": gi["verdict_resolution"] == 1.0 and gi["nonfinite_share"] <= NONFINITE_SHARE_MAX},
        "G4": {"observed": {"tied_cells": gi["tied_cells"], "cells": grid["mixed_cells"],
                            "tied_share": gi["tied_share"]},
               "pass": gi["tied_share"] is not None and gi["tied_share"] <= TIE_SHARE_MAX},
        "G5": {"observed": {"J_modal_pair": gi["J_modal_pair"],
                            "J_nonmodal_step_share": gi["J_nonmodal_step_share"],
                            "rank_divergent_share": gi["rank_divergent_share"]},
               "pass": (gi["J_nonmodal_step_share"] is not None
                        and gi["J_nonmodal_step_share"] >= NONMODAL_SHARE_MIN
                        and gi["rank_divergent_share"] >= RANK_DIVERGENCE_SHARE_MIN)},
        "G6": {"observed": {"half_width": None if primary is None else primary["half_width"]},
               "pass": primary is not None and primary["half_width"] <= HALF_WIDTH_MAX},
        "caps": {"observed": {"gpu_seconds": ledger["gpu_seconds_elapsed"],
                              "wall_seconds": ledger["wall_seconds_elapsed"],
                              "status": ledger["status"]},
                 "pass": (ledger["status"] == "terminal"
                          and ledger["gpu_seconds_elapsed"] <= GPU_CAP_SECONDS
                          and ledger["wall_seconds_elapsed"] <= WALL_CAP_SECONDS)},
    }
    guards_ok = all(g["pass"] for g in guards.values())
    base_ok = all(guards[g]["pass"] for g in ("G1", "G2", "G3", "G4", "G5", "caps"))
    primary_token = token(primary, guards_ok)
    secondary_tokens = {}
    for name in ("J-M", "J-OL"):
        block = grid["contrasts"][name]["estimate"]
        own = block is not None and block["half_width"] <= HALF_WIDTH_MAX
        secondary_tokens[name] = {"token": token(block, base_ok and own),
                                  "own_half_width": None if block is None else block["half_width"],
                                  "own_half_width_pass": own}
    replication = {}
    for inventory in INVENTORIES:
        if inventory == PRIMARY:
            continue
        block = per_inventory[inventory]["contrasts"]["J-F*"]["estimate"]
        replication[inventory] = {"estimate": block, "meets_rule": bool(
            block is not None and block["mean"] >= DELTA and block["interval"][0] > 0)}
    spread_mean = grid["fixed_pair_spread"]["mean"] if grid["fixed_pair_spread"] else None
    near_inert = spread_mean is not None and spread_mean < DELTA
    if primary_token == "supported":
        reading = plan["licenses"]["supported"]
    elif primary_token == "not_supported_by_this_experiment":
        reading = (("the selection schedule is near-inert on this checkpoint; "
                    + plan["licenses"]["near_inert"]) if near_inert
                   else plan["licenses"]["not_supported_healthy_spread"])
    else:
        reading = plan["licenses"]["insufficient"]
    return {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY,
        "structure": {"records": records, "scheduled_cells": len(scheduled_cells(universe)),
                      "arms": len(ARMS), "mixed_cells": {i: per_inventory[i]["mixed_cells"]
                                                         for i in INVENTORIES}},
        "frequencies": frequencies,
        "guards": guards, "guards_pass": guards_ok,
        "primary": {"estimand": "theta_grid = AUC_J - AUC_F* over mixed grid cells",
                    "F_star": grid["F_star"], "estimate": primary, "delta": DELTA,
                    "token": primary_token},
        "secondary_tokens": secondary_tokens,
        "replication": {"inventories": replication,
                        "count": sum(r["meets_rule"] for r in replication.values()),
                        "of": len(replication)},
        "fixed_pair_spread_grid_member_mean": spread_mean, "near_inert": near_inert,
        "reading": reading,
        "inventories": per_inventory,
        "compute": {"per_arm": accounting, "run_gpu_seconds": ledger["gpu_seconds_elapsed"],
                    "run_wall_seconds": ledger["wall_seconds_elapsed"], "engine_seconds": 0,
                    "smoke_projected_gpu_seconds": plan["smoke_evidence"]["projected_run_gpu_seconds"]},
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

CONTRAST_ORDER = ("J-F*", "J-F_hindsight", "J-F(1,macro)", "J-F(1,continuous)", "J-M", "J-OL",
                  "J-T", "J-D", "C-J-C-F*")


def contrast_label(inventory_block, name):
    if name == "J-F*":
        return f"J - F* [{inventory_block['F_star']}]"
    if name == "J-F_hindsight":
        return f"J - F_hindsight [{inventory_block['F_hindsight']}]"
    if name == "C-J-C-F*":
        return f"C-J - C-F* [{inventory_block['C_F_star']}]"
    return name.replace("-", " - ", 1)


def comparisons_csv(compute):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["id", "inventory", "statistic", "value", "interval_low", "interval_high",
                     "interval_label", "cells", "members", "detail"])

    def put(identity, inventory, statistic, value, block=None, cells="", members="", detail=""):
        writer.writerow([identity, inventory, statistic, "" if value is None else value,
                         block["interval"][0] if block else "", block["interval"][1] if block else "",
                         "DESCRIPTIVE" if block else "", cells, members, detail])

    p = compute["primary"]
    put("primary_theta_grid", PRIMARY, f"AUC_J - AUC_F* (F* = {p['F_star']})",
        p["estimate"] and p["estimate"]["mean"], p["estimate"],
        compute["inventories"][PRIMARY]["contrasts"]["J-F*"]["cells"],
        compute["inventories"][PRIMARY]["contrasts"]["J-F*"]["members"], f"token {p['token']}")
    for name, block in compute["secondary_tokens"].items():
        put(f"token_{name}", PRIMARY, f"{name} token", block["token"])
    for guard, block in compute["guards"].items():
        put(f"guard_{guard}", "", f"{guard} pass", block["pass"],
            detail=json.dumps(block["observed"], sort_keys=True))
    for inventory in INVENTORIES:
        inv = compute["inventories"][inventory]
        for name in CONTRAST_ORDER:
            c = inv["contrasts"][name]
            put(f"{inventory}_{name}", inventory, contrast_label(inv, name),
                c["estimate"] and c["estimate"]["mean"], c["estimate"], c["cells"], c["members"],
                f"undefined {c['undefined_cells']}")
        t = inv["top1"]
        put(f"{inventory}_top1_J-F*", inventory, f"top-1 J - F* [{inv['F_star']}] (pooled descriptive)",
            t["difference"]["estimate"] and t["difference"]["estimate"]["mean"], t["difference"]["estimate"],
            t["cells"], len(inv["mixed_members"]),
            f"J {t['J_hits']}/{t['cells']}; F* {t['F_star_hits']}/{t['cells']}; chance {t['chance_mean']}")
        s = inv["fixed_pair_spread"]
        put(f"{inventory}_spread", inventory, "fixed-pair spread max_F - min_F", s and s["mean"], s,
            inv["mixed_cells"], len(inv["mixed_members"]))
        for arm in ARMS:
            b = inv["arm_auc"][arm]
            put(f"{inventory}_auc_{arm}", inventory, f"AUC {arm}", b and b["mean"], b,
                b["units"] if b else 0, b["clusters"] if b else 0)
        cov = inv["coverage"]
        put(f"{inventory}_coverage", inventory, "successful slots resolved within endpoint 225",
            cov["reached_within_225"], detail=json.dumps(cov, sort_keys=True))
    for arm, block in compute["compute"]["per_arm"].items():
        put(f"compute_{arm}", "", f"compute {arm}", block["gpu_seconds"],
            detail=json.dumps({k: v for k, v in block.items() if k != "gpu_seconds"}, sort_keys=True))
    return buffer.getvalue()


def findings_md(plan, compute):
    lines = []
    add = lines.append
    p = compute["primary"]
    grid = compute["inventories"][PRIMARY]
    add("# Issue-96 tau_ad-W: within-checkpoint selection-schedule contrast — findings")
    add("")
    add(f"- identity `{IDENTITY}`, plan v1 frozen {plan['frozen_at']} before any scoring run")
    add(f"- validation command: `{VALIDATION_COMMAND}`; zero engine seconds; endpoint {ENDPOINT}; "
        "every interval DESCRIPTIVE (member-clustered, 10000 draws, PCG64 7201)")
    add(f"- disclosure: {plan['disclosure']}")
    add("")
    add("## Disposition")
    add("")
    add(f"- **primary theta_grid = AUC_J - AUC_F\\* (F\\* = {p['F_star']}): "
        f"{fmt_interval(p['estimate'])} over {grid['contrasts']['J-F*']['cells']} cells / "
        f"{grid['contrasts']['J-F*']['members']} members -> {p['token']}** (delta {DELTA})")
    for name, block in compute["secondary_tokens"].items():
        est = grid["contrasts"][name]["estimate"]
        add(f"- secondary {name} (grid): {fmt_interval(est)} -> **{block['token']}**")
    add(f"- fixed-pair spread on the grid (member mean of per-cell max_F - min_F): "
        f"{fmt(compute['fixed_pair_spread_grid_member_mean'])}; near-inert rule triggered: "
        f"{compute['near_inert']}")
    add(f"- replication of J - F\\* on the secondary inventories (theta >= delta AND lower > 0): "
        f"{compute['replication']['count']}/{compute['replication']['of']} "
        + ", ".join(f"{i} {fmt_interval(r['estimate'])}"
                    for i, r in compute["replication"]["inventories"].items()))
    add(f"- reading: {compute['reading']}")
    add("")
    add("## Guards")
    add("")
    add("| guard | pass | observed |")
    add("|---|---|---|")
    for guard, block in compute["guards"].items():
        add(f"| {guard} | {block['pass']} | `{json.dumps(block['observed'], sort_keys=True)}` |")
    add("")
    add("## Contrasts per inventory (never pooled; AUC difference, member-clustered)")
    add("")
    add("| inventory | mixed members / cells | " + " | ".join(
        contrast_label({"F_star": "F*", "F_hindsight": "F_h", "C_F_star": "C-F*"}, n)
        for n in CONTRAST_ORDER) + " |")
    add("|---|---|" + "---|" * len(CONTRAST_ORDER))
    for inventory in INVENTORIES:
        inv = compute["inventories"][inventory]
        add(f"| {inventory} | {len(inv['mixed_members'])} / {inv['mixed_cells']} | " + " | ".join(
            fmt_interval(inv["contrasts"][n]["estimate"]) for n in CONTRAST_ORDER) + " |")
    add("")
    add("| inventory | F* (cross-fit) | F_hindsight | C-F* | AUC J | AUC F* | top-1 J / F* / chance | "
        "fixed-pair spread | coverage (reached/successful slots) | J modal pair, non-modal share |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for inventory in INVENTORIES:
        inv = compute["inventories"][inventory]
        t = inv["top1"]
        cov = inv["coverage"]
        gi = inv["guard_inputs"]
        add(f"| {inventory} | {inv['F_star']} | {inv['F_hindsight']} | {inv['C_F_star']} | "
            f"{fmt_interval(inv['arm_auc']['J'])} | {fmt_interval(inv['arm_auc'][inv['F_star']])} | "
            f"{t['J_hits']}/{t['cells']} / {t['F_star_hits']}/{t['cells']} / {fmt(t['chance_mean'])} | "
            f"{fmt_interval(inv['fixed_pair_spread'])} | {cov['reached_within_225']}/{cov['successful_slots']} | "
            f"{gi['J_modal_pair']}, {fmt(gi['J_nonmodal_step_share'])} |")
    add("")
    add(f"- top-1 power note: {plan['estimands']['power_note']}")
    add("")
    add("## Per-arm member-mean AUC (mixed cells)")
    add("")
    add("| arm | " + " | ".join(INVENTORIES) + " |")
    add("|---|" + "---|" * len(INVENTORIES))
    for arm in ARMS:
        add(f"| {arm} | " + " | ".join(fmt(compute["inventories"][i]["arm_member_mean_auc"][arm])
                                       for i in INVENTORIES) + " |")
    add("")
    add("## Schedules (outcome-free)")
    add("")
    for seed, block in compute["frequencies"]["coordinates"].items():
        add(f"- seed {seed}: J step counts {dict(zip(PAIR_NAMES, compute['frequencies']['counts'][seed]))}; "
            f"alpha_T {block['alpha_T']}, Delta_D {block['delta_D']}")
    add("")
    add("## Compute (reported, not matched)")
    add("")
    add("| arm | records | transition calls | controller calls | linear MACs | GPU s |")
    add("|---|---|---|---|---|---|")
    for arm, block in compute["compute"]["per_arm"].items():
        add(f"| {arm} | {block['records']} | {block['transition_calls']} | {block['controller_calls']} | "
            f"{block['linear_macs']} | {block['gpu_seconds']:.1f} |")
    add("")
    add(f"- run GPU {compute['compute']['run_gpu_seconds']:.1f} s, wall "
        f"{compute['compute']['run_wall_seconds']:.1f} s (caps {GPU_CAP_SECONDS:.0f} / "
        f"{WALL_CAP_SECONDS:.0f} s); engine seconds 0")
    add("")
    add("## Claim boundary")
    add("")
    add(plan["claim_boundary"] + ".")
    add("")
    return "\n".join(lines)


def summary(plan, compute):
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "validation_command": VALIDATION_COMMAND, "question": plan["question"],
            "disclosure": plan["disclosure"],
            "primary": compute["primary"], "secondary_tokens": compute["secondary_tokens"],
            "guards": compute["guards"], "guards_pass": compute["guards_pass"],
            "replication": compute["replication"],
            "fixed_pair_spread_grid_member_mean": compute["fixed_pair_spread_grid_member_mean"],
            "near_inert": compute["near_inert"], "reading": compute["reading"],
            "inventories": {inventory: {k: v for k, v in block.items() if k not in ("rows",)}
                            | {"guard_inputs": {k: v for k, v in block["guard_inputs"].items()
                                                if k != "kendall_tau_J_vs_F_modal"}}
                            for inventory, block in compute["inventories"].items()},
            "frequencies": compute["frequencies"], "compute": compute["compute"],
            "claim_boundary": plan["claim_boundary"], "issue_64_authorized": False}


def rendered(plan, compute):
    return {"summary.json": json_text(summary(plan, compute)), "findings.md": findings_md(plan, compute),
            "comparisons.csv": comparisons_csv(compute)}


def publish(output):
    output = Path(output)
    plan = load_plan(output)
    compute = compute_tables(output, plan)
    write_json(output / "compute.json", compute)
    for name, text in rendered(plan, compute).items():
        (output / name).write_bytes(text.encode())
    log(f"published: primary {compute['primary']['token']} "
        f"({fmt_interval(compute['primary']['estimate'])}); guards pass {compute['guards_pass']}")
    return 0


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    fresh = compute_tables(output, plan)
    if read_json(output / "compute.json") != json.loads(json_text(fresh)):
        raise ValueError("compute.json differs from the fresh recomputation")
    for name, text in rendered(plan, fresh).items():
        if (output / name).read_bytes() != text.encode():
            raise ValueError(f"published {name} differs from the recomputation")
    log(f"exact recomputation validation passed: {fresh['structure']['records']} records; "
        "frequencies, OL schedules, compute.json, summary.json, comparisons.csv and findings.md "
        f"re-derived and byte-compared; primary {fresh['primary']['token']} "
        f"({time.monotonic() - began:.1f}s)")
    return 0


def dry_run(output):
    sources = load_sources()
    members, universe, excluded = build_universe(sources)
    log(f"dry run (no write, no statistic); output root {Path(output)}")
    for inventory in INVENTORIES:
        entries = universe[inventory]
        candidates = Counter(len(entry["inventory"]) for entry in entries.values())
        verdicts = sum(len(entry["verdicts"]) for entry in entries.values())
        mixed_members = sum(mixed(entry) for entry in entries.values())
        log(f"  {inventory}: {len(entries)} members, candidates {dict(candidates)}, "
            f"{verdicts} verdict-bearing slots, {mixed_members} mixed members -> "
            f"{mixed_members * len(SEEDS)} mixed cells; {INVENTORY_SOURCES[inventory]}")
    for row in excluded:
        log(f"  excluded {row['inventory']}/{row['member']}: {row['rule']}")
    cells = scheduled_cells(universe)
    log(f"  arms {len(ARMS)}: {', '.join(ARMS)}")
    log(f"  scheduled cells {len(cells)} x {len(ARMS)} arms = {len(cells) * len(ARMS)} records; "
        f"endpoint {ENDPOINT}; seeds {list(SEEDS)}")
    log(f"  plan present: {(Path(output) / 'plan.json').is_file()}; records present: "
        f"{len(list((Path(output) / 'records').glob('*.json'))) if (Path(output) / 'records').is_dir() else 0}")
    return 0


MODES = {"dry-run": (dry_run, False), "smoke": (smoke, True), "prepare": (prepare, False),
         "run": (run, True), "publish": (publish, False), "validate": (validate, False)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    mode = next(name for name in MODES if getattr(args, name.replace("-", "_")))
    function, locked = MODES[mode]
    try:
        if locked:
            with wlc.GPULock():
                return function(args.output)
        return function(args.output)
    except (ValueError, OSError, KeyError) as error:
        log(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
