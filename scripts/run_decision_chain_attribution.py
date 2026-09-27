"""Issue-99 step 1-2: stage-wise attribution of the launch-ranking loss and a tie-free cost.

Binding runner module: scripts/run_decision_chain_attribution.py
Exact validation command: python -u -m scripts.run_decision_chain_attribution --validate

Question: on the four engine-verdicted #96 inventories (#93 grid [primary], #93
offset, #87+#89 angle, #94 power), where along the decision chain
    engine state -> perception -> carrier -> predictor rollout -> cost
is the ranking signal lost? Five rankers replace one stage at a time with engine
truth; every candidate launch is scored under the frozen TaskObjective count cost
and under a tie-free continuous cost read from the same endpoint carrier:

- a   engine endpoint: the engine state of the executed shot at rollout position 225
      (fixed step 41250; terminal absorption) projected into the carrier -> cost.
      The ceiling of the cost itself.
- ap  perceived endpoint: the agent frame of the executed shot at position 225 parsed
      by the frozen slot parser -> cost (perception at the endpoint only).
- d   teacher-forced: every predictor input is the parsed carrier of the executed
      shot at the request's own positions; only the last step is predicted (removes
      compounding; uses post-launch frames, so it is a diagnostic, not a ranker).
- c   current: the parsed sealed anchor -> predictor rollout to 225 -> cost (#96).
- b   engine anchor: the engine state at position 0 projected into the carrier ->
      predictor rollout to 225 -> cost (removes perception error at the anchor).

Requests: the nine hybrid fixed pairs F-(Delta, alpha), the three continuous fixed
horizons C-F-Delta, and the frozen controllers J (hybrid) and C-J (continuous), on
the frozen #77 N1 checkpoints of seeds 20260908/09/10. Cohorts: all 15 #96
members and the held-out members 007/008/014/016 (no #77 predictor or controller
was fit on them), never pooled with each other; inventories never pooled.

Zero engine seconds: every executed shot is a retained #87/#89/#93/#94 oracle
execution; nothing is rendered, captured or retrained.

Modes (frozen-protocol chronology, binding shared rule):
- --dry-run   no-write structural inventory; no statistic.
- --smoke     pre-freeze controls on member issue-77-n1-001 (no verdict joined,
              no ranking statistic): stage-c replication of the #96 records,
              engine projection geometry against the parsed anchor, anchor-frame
              identity, teacher-forcing mechanics, extraction/rollout timing.
- --prepare   freezes plan.json before any scoring run.
- --run       outcome-free evidence extraction (parsed and engine carriers of every
              verdicted executed shot), then endpoint records (a, ap) and model
              records (b, c, d); ledger-resumable; retained records are never
              recomputed.
- --publish   compute.json / summary.json / comparisons.csv / findings.md.
- --validate  re-derives the endpoint costs from the retained evidence tensors and
              every published table from the retained records; byte-compares.

Every interval is DESCRIPTIVE (member-clustered percentile bootstrap, 10000 draws,
PCG64 7201). Verdict vocabulary: supported / not_supported_by_this_experiment /
readiness_or_precision_insufficient.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
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
from scripts import run_tau_ad_within_checkpoint as t96

ROOT = t96.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-99-decision-chain-attribution-v1"
DYNAMICS = t96.DYNAMICS
SOURCE96 = t96.OUTPUT

IDENTITY = "issue-99-decision-chain-attribution-v1"
SCHEMA_PLAN = "issue_99_decision_chain_attribution_plan_v1"
SCHEMA_SMOKE = "issue_99_decision_chain_attribution_smoke_v1"
SCHEMA_EVIDENCE = "issue_99_decision_chain_evidence_v1"
SCHEMA_ENDPOINT = "issue_99_decision_chain_endpoint_record_v1"
SCHEMA_MODEL = "issue_99_decision_chain_model_record_v1"
SCHEMA_LEDGER = "issue_99_decision_chain_attribution_ledger_v1"
SCHEMA_COMPUTE = "issue_99_decision_chain_attribution_compute_v1"
SCHEMA_REPORT = "issue_99_decision_chain_attribution_report_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_decision_chain_attribution --validate"
RUNNER = "scripts/run_decision_chain_attribution.py"

DEVICE = "cuda"
SEEDS = t96.SEEDS
ENDPOINT = t96.ENDPOINT
INVENTORIES = t96.INVENTORIES
PRIMARY = t96.PRIMARY
DECISION_FIXED_STEP = p93.DECISION_FIXED_STEP
NATIVE_STRIDE = p93.NATIVE_STRIDE

STAGES = ("a", "ap", "d", "c", "b")
ENDPOINT_STAGES = ("a", "ap")
MODEL_STAGES = ("d", "c", "b")
STAGE_DEFINITIONS = {
    "a": ("engine endpoint: engine state of the executed shot at position 225 (fixed step "
          "41250, terminal absorption) projected into the carrier (presence = active entity "
          "with a body, centers through the frame's world_to_observation transform, kinds "
          "one-hot of the declared slot kind; prior position 224 for motion) -> cost"),
    "ap": ("perceived endpoint: the executed shot's agent frame at position 225 (prior 224; "
           "terminal absorption) parsed by the frozen issue-70 slot parser -> cost"),
    "d": ("teacher-forced predictor: the request's pair schedule walks the executed shot's "
          "parsed carriers (fixed pair Delta: input at position 225 - Delta; J / C-J: the "
          "controller chooses on the parsed carrier at each visited position); only the "
          "final step to 225 is predicted -> cost"),
    "c": ("current: the member's sealed parsed anchor carrier -> request rollout to 225 -> "
          "cost (the #96 arm, recomputed)"),
    "b": ("engine anchor: the engine state at position 0 of the member's executed shots "
          "projected into the carrier -> request rollout to 225 -> cost"),
}
REQUESTS = (*t96.FIXED_ARMS, *t96.CONTINUOUS_FIXED_ARMS, "J", "C-J")
COSTS = ("tie_free", "count")
PRIMARY_COST = "tie_free"
DISPLACEMENT_WEIGHT = 0.1
COST_DEFINITIONS = {
    "count": ("frozen TaskObjective: 1000 * clamp(pig presence, 0, 1) + sum over block slots "
              "of clamp(block presence, 0, 1)"),
    "tie_free": (f"pig presence (unclamped) - {DISPLACEMENT_WEIGHT} * Euclidean displacement of "
                 "the pig slot center (normalized screen units) between the stage's start "
                 "carrier (position 0) and the endpoint carrier; lower is better"),
}
HELD_OUT = ("issue-77-n1-007", "issue-77-n1-008", "issue-77-n1-014", "issue-77-n1-016")
COHORTS = ("all", "held_out")

TIE_SHARE_TARGET = 0.10
TARGET_AUC = 0.65
TARGET_LOWER = 0.55
DROP_MARGIN = 0.02
NONFINITE_SHARE_MAX = 0.10
ENGINE_ANCHOR_TOLERANCE = 1e-6
PROJECTION_CENTER_TOLERANCE = 0.05
PARSE_BATCH_TOLERANCE = 1e-3  # cuDNN batch-size float differences between batch-1 and batch-64 parses
COST_TOLERANCE = t96.COST_TOLERANCE
GPU_CAP_SECONDS = 3 * 3600.0
WALL_CAP_SECONDS = 6 * 3600.0
STOP_TOKEN = t96.STOP_TOKEN
SMOKE_MEMBER = t96.G1_MEMBER
SMOKE_SEED = t96.G1_SEED
PARSE_BATCH = 64
PLACEHOLDER_MARKERS = (*t96.PLACEHOLDER_MARKERS, "~", "e.g.")

read_json = t96.read_json
write_json = t96.write_json
json_text = t96.json_text
sha256_of = t96.sha256_of
bootstrap = t96.bootstrap
cell_auc = t96.cell_auc
fmt = t96.fmt
fmt_interval = t96.fmt_interval


def log(message):
    print(f"[issue-99-decision-chain] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# universe (no statistic)
# ---------------------------------------------------------------------------

def fit_lineages():
    lineages = read_json(DYNAMICS / "plan.json")["n1_lineages"]
    return sorted(lineages["predictor"]), sorted(lineages["controller"])


def cohort_members(members):
    predictor, controller = fit_lineages()
    fitted = set(predictor) | set(controller)
    held = sorted(m for m in members if m not in fitted)
    if tuple(held) != HELD_OUT:
        raise ValueError(f"held-out members {held} differ from the declared {HELD_OUT}")
    return {"all": sorted(members), "held_out": list(HELD_OUT)}


def trace_record(sources, inventory, member, entry, ordinal):
    """The oracle record whose execution carries the candidate's verdict."""
    if inventory in ("grid", "offset"):
        return sources["oracle93"][p93.oracle_identity(member, inventory, ordinal)]
    if inventory == "power":
        return sources["oracle94"][p94.oracle_identity(member, ordinal)]
    key = (entry["state"], ordinal)
    # #89 completion verdicts supersede #87 (run_selection_validity_restatement.build_verdicts)
    for records in (sources["records89"], sources["oracles87"]):
        for identity in sorted(records):
            record = records[identity]
            if ((record["cell"]["state"], record["cell"]["ordinal"]) == key
                    and record.get("outcome") is not None):
                return record
    raise ValueError(f"no verdict-bearing angle record for {key}")


def candidates(universe):
    """Frozen order of verdict-bearing candidates: inventory, member, ordinal."""
    return [(inventory, member, ordinal) for inventory in INVENTORIES
            for member in sorted(universe[inventory])
            for ordinal in sorted(universe[inventory][member]["verdicts"])]


def candidate_key(inventory, member, ordinal):
    return f"{inventory}--{member}--o{ordinal:02d}"


def evidence_path(output, inventory, member, ordinal):
    return Path(output) / "evidence" / f"{candidate_key(inventory, member, ordinal)}.pt"


def endpoint_path(output, inventory, member):
    return Path(output) / "records" / f"endpoint--{inventory}--{member}.json"


def model_path(output, inventory, member, seed):
    return Path(output) / "records" / f"model--{inventory}--{member}--seed{seed}.json"


def scheduled_members(universe):
    return [(inventory, member) for inventory in INVENTORIES for member in sorted(universe[inventory])]


def scheduled_cells(universe):
    return [(inventory, member, seed) for inventory, member in scheduled_members(universe)
            for seed in SEEDS]


# ---------------------------------------------------------------------------
# evidence extraction: parsed and engine-projected carriers of an executed shot
# ---------------------------------------------------------------------------

UNAVAILABLE_MICRO = {"predicates": {"contact": {"availability": "unavailable"},
                                    "supports": {"availability": "unavailable"}}}
UNAVAILABLE_MACRO = {"predicates": {"steady-state": {"availability": "unavailable"},
                                    "structure-unstable": {"availability": "unavailable"}}}


def observation_frames(record):
    native_root = Path(record["execution"]["native_root"])
    observation_root = native_root.parent.parent / "shot-1" / "observation-trace"
    manifest = read_json(observation_root / "observation_trace_manifest.json")
    frames = manifest["frame_records"]
    if (frames[0]["fixed_step"] != DECISION_FIXED_STEP
            or any(f["fixed_step"] != DECISION_FIXED_STEP + NATIVE_STRIDE * i
                   for i, f in enumerate(frames[:-1]))
            or not (DECISION_FIXED_STEP + NATIVE_STRIDE * (len(frames) - 2)
                    < frames[-1]["fixed_step"]
                    <= DECISION_FIXED_STEP + NATIVE_STRIDE * (len(frames) - 1))):
        raise ValueError(f"observation cadence differs for {observation_root}")
    return native_root, observation_root, frames


def agent_observation(observation_root, frame, with_bytes=True):
    from world_model.data.deployment_temporal import AgentObservation
    ref = frame["agent_observation"]
    data = (observation_root / ref["relative_path"]).read_bytes() if with_bytes else b""
    return AgentObservation(ref["identity"], frame["fixed_step"], frame["fixed_time_seconds"],
                            data, "agent")


def engine_samples(native_root, steps):
    from scripts.canonical_native_trace import NativeTrace
    wanted, found = set(steps), {}
    last = max(wanted)
    for chunk in NativeTrace(native_root).chunks():
        for sample in chunk["fixed_step_samples"]:
            if sample["fixed_step"] in wanted:
                found[sample["fixed_step"]] = sample
        if chunk["fixed_step_samples"][-1]["fixed_step"] >= last:
            break
    if set(found) != wanted:
        raise ValueError(f"engine trace {native_root} lacks steps {sorted(wanted - set(found))}")
    return found


def engine_parsed(sample, metadata, vocabulary):
    """The parser-output dict an ideal perception would emit for this engine state."""
    from scripts.run_issue_70_parser_repair import targets
    from world_model.data.deployment_temporal import OBJECT_KIND_VOCABULARY
    slots = set(vocabulary)
    filtered = {**sample, "entities": [e for e in sample["entities"]
                                       if e["scenario_object_id"] in slots]}
    truth = targets(filtered, metadata, UNAVAILABLE_MICRO, UNAVAILABLE_MACRO, vocabulary)
    kinds = torch.zeros(len(vocabulary), len(OBJECT_KIND_VOCABULARY))
    for index, name in enumerate(vocabulary):
        kind = name.split(":", 1)[0]
        kinds[index, OBJECT_KIND_VOCABULARY.index(kind) if kind in OBJECT_KIND_VOCABULARY[:-1]
              else len(OBJECT_KIND_VOCABULARY) - 1] = 1.0
    return {"presence": truth["presence"].float(), "centers": truth["centers"].float(),
            "kinds": kinds, "relations": torch.zeros(len(vocabulary), len(vocabulary), 2),
            "macros": torch.zeros(2)}


def build(adapter, context_prior, context_current, current, prior):
    from world_model.data.deployment_temporal import TemporalObservationContext
    return adapter.build_from_parsed(TemporalObservationContext(context_prior, context_current),
                                     current, prior).tensor.float().cpu()


def extract(adapter, vocabulary, record):
    """Outcome-free carriers of one executed shot.

    parsed[k], k = 0..225: the parsed carrier at position min(k, last) with prior
    position - 1 (terminal absorption, the #77/#85 parse_offset_carrier semantics).
    engine_start / engine_end: engine-projected carriers at positions 0 and
    min(225, last) (prior position - 1 for motion)."""
    native_root, observation_root, frames = observation_frames(record)
    last = len(frames) - 1
    end = min(ENDPOINT, last)
    positions = list(range(end + 1))
    parsed = {}
    calls = 0
    with torch.no_grad():
        for begin in range(0, len(positions), PARSE_BATCH):
            chunk = positions[begin:begin + PARSE_BATCH]
            values = adapter.parse_batch(tuple(agent_observation(observation_root, frames[p])
                                               for p in chunk))
            parsed.update(zip(chunk, values, strict=True))
            calls += len(chunk)
        light = {p: agent_observation(observation_root, frames[p], with_bytes=False)
                 for p in positions}
        carriers = [build(adapter, None if p == 0 else light[p - 1], light[p], parsed[p],
                          None if p == 0 else parsed[p - 1]) for p in positions]
        carriers.extend([carriers[-1]] * (ENDPOINT - end))
        steps = sorted({frames[0]["fixed_step"], frames[end]["fixed_step"],
                        frames[max(0, end - 1)]["fixed_step"]})
        samples = engine_samples(native_root, steps)
        engine = {p: engine_parsed(samples[frames[p]["fixed_step"]], frames[p]["capture_metadata"],
                                   vocabulary)
                  for p in sorted({0, max(0, end - 1), end})}
        engine_start = build(adapter, None, light[0], engine[0], None)
        engine_end = build(adapter, None if end == 0 else light[end - 1], light[end], engine[end],
                           None if end == 0 else engine[end - 1])
    frame0 = (observation_root / frames[0]["agent_observation"]["relative_path"]).read_bytes()
    return {"schema": SCHEMA_EVIDENCE, "plan_identity": IDENTITY,
            "source": record["cell"]["identity"], "native_root": str(native_root),
            "frame_count": len(frames), "last_index": last, "end_index": end,
            "absorbed": end < ENDPOINT, "frame0_sha256": f"sha256:{sha256(frame0).hexdigest()}",
            "parse_calls": calls, "parsed": torch.stack(carriers),
            "engine_start": engine_start, "engine_end": engine_end}


def load_evidence(output, inventory, member, ordinal):
    evidence = torch.load(evidence_path(output, inventory, member, ordinal), map_location="cpu",
                          weights_only=True)
    if evidence.get("schema") != SCHEMA_EVIDENCE or evidence.get("plan_identity") != IDENTITY:
        raise ValueError(f"evidence {candidate_key(inventory, member, ordinal)} binding differs")
    return evidence


# ---------------------------------------------------------------------------
# costs
# ---------------------------------------------------------------------------

class Costs:
    def __init__(self, objective):
        self.objective = objective
        if len(objective.pig_slots) != 1:
            raise ValueError("the tie-free cost is declared for exactly one pig slot")
        base = 2 + 13 * objective.pig_slots[0]
        self.presence, self.center = base, (base + 5, base + 6)

    def row(self, ordinal, z, z0):
        """Both costs of one endpoint carrier; nonfinite -> retained typed failure."""
        z = z.detach()
        if not bool(torch.isfinite(z).all()):
            return {"ordinal": ordinal, "excluded": True, "count": None, "tie_free": None,
                    "pig_presence": None, "pig_displacement": None}
        presence = float(z[self.presence])
        displacement = math.hypot(float(z[self.center[0]]) - float(z0[self.center[0]]),
                                  float(z[self.center[1]]) - float(z0[self.center[1]]))
        return {"ordinal": ordinal, "excluded": False, "count": float(self.objective(z)),
                "tie_free": tie_free(presence, displacement), "pig_presence": presence,
                "pig_displacement": displacement}


def tie_free(presence, displacement):
    return presence - DISPLACEMENT_WEIGHT * displacement


# ---------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------

def macs_of(controller):
    return (sum(m.in_features * m.out_features for m in controller.modules()
                if isinstance(m, torch.nn.Linear)) if controller is not None else 0)


def rollout_endpoints(model, controller, z0, actions, spec):
    """t96.rollout verbatim (same ops, same order) returning the endpoint carriers."""
    from world_model.training.matched_dynamics import linear_macs, pairs_for
    pairs = pairs_for(model)
    count = len(actions)
    z = z0[None].to(DEVICE).expand(count, -1).clone()
    alive = torch.ones(count, dtype=torch.bool, device=z.device)
    positions = torch.zeros(count, dtype=torch.long, device=z.device)
    pair_macs = [linear_macs(model, pair) for pair in pairs]
    transition_calls = controller_calls = macs = 0
    torch.cuda.synchronize()
    began = time.monotonic()
    with torch.no_grad():
        while bool(alive.any()):
            index = alive.nonzero(as_tuple=True)[0]
            if spec["kind"] == "fixed":
                chosen = torch.full((len(index),), spec["pair_index"], dtype=torch.long,
                                    device=z.device)
            else:
                logits = controller(z[index], actions[index], ENDPOINT - positions[index])
                chosen = t96.select_pairs(spec["kind"], logits, spec)
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
                positions[rows] += pair.delta
            z = z_next
            alive = alive & torch.isfinite(z).all(dim=1) & (positions < ENDPOINT)
    torch.cuda.synchronize()
    macs += controller_calls * macs_of(controller)
    return z, {"transition_calls": transition_calls, "controller_calls": controller_calls,
               "linear_macs": macs, "gpu_seconds": time.monotonic() - began}


def teacher_forced(model, controller, z_true, actions, spec):
    """Endpoint prediction whose every input is the executed shot's parsed carrier.

    z_true [N, 226, D]: parsed carriers at positions 0..225. Fixed pair Delta: one
    step from position 225 - Delta. J / C-J: the controller chooses on the parsed
    carrier at every visited position; the step that reaches 225 is predicted."""
    from world_model.training.matched_dynamics import linear_macs, pairs_for
    pairs = pairs_for(model)
    count = len(actions)
    z_true = z_true.to(DEVICE)
    z_end = torch.full((count, z_true.shape[-1]), float("nan"), device=DEVICE)
    positions = torch.zeros(count, dtype=torch.long, device=DEVICE)
    alive = torch.ones(count, dtype=torch.bool, device=DEVICE)
    pair_macs = [linear_macs(model, pair) for pair in pairs]
    transition_calls = controller_calls = macs = 0
    torch.cuda.synchronize()
    began = time.monotonic()
    with torch.no_grad():
        if spec["kind"] == "fixed":
            pair = pairs[spec["pair_index"]]
            z_end = model.carrier(z_true[:, ENDPOINT - pair.delta], actions, pair)
            transition_calls, macs = count, count * pair_macs[spec["pair_index"]]
            positions[:] = ENDPOINT
        else:
            while bool(alive.any()):
                index = alive.nonzero(as_tuple=True)[0]
                z = z_true[index, positions[index]]
                logits = controller(z, actions[index], ENDPOINT - positions[index])
                chosen = t96.select_pairs(spec["kind"], logits, spec)
                controller_calls += len(index)
                for pair_index, pair in enumerate(pairs):
                    mask = chosen == pair_index
                    rows = index[mask]
                    if not len(rows):
                        continue
                    final = rows[positions[rows] + pair.delta == ENDPOINT]
                    if len(final):
                        z_end[final] = model.carrier(z_true[final, positions[final]],
                                                     actions[final], pair)
                        transition_calls += len(final)
                        macs += len(final) * pair_macs[pair_index]
                    positions[rows] += pair.delta
                alive = positions < ENDPOINT
    torch.cuda.synchronize()
    if not bool((positions == ENDPOINT).all()):
        raise ValueError("teacher-forced schedule overshoots the endpoint")
    macs += controller_calls * macs_of(controller)
    return z_end, {"transition_calls": transition_calls, "controller_calls": controller_calls,
                   "linear_macs": macs, "gpu_seconds": time.monotonic() - began}


def action_batch(inventory_items):
    return torch.cat([wlc.action_tensor(item["action"], DEVICE) for item in inventory_items], 0)


class Stack(t96.Stack):
    """t96.Stack plus the frozen objective wrapped in both costs."""

    def __init__(self, sources, anchors):
        super().__init__(sources, anchors)
        self.costs = Costs(self.objective)

    def models_for(self, request, seed):
        family = t96.arm_family(request)
        controller = self.controller(seed, family) if request in ("J", "C-J") else None
        return self.model(seed, family), controller


# ---------------------------------------------------------------------------
# smoke (pre-freeze; no verdict joined, no ranking statistic)
# ---------------------------------------------------------------------------

def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the smoke is a pre-freeze control")
    sources = t96.load_sources()
    members, universe, _ = t96.build_universe(sources)
    cohort_members(members)
    anchors, anchor_evidence = t96.member_anchors(sources, members)
    stack = Stack(sources, anchors)
    vocabulary = read_json(DYNAMICS / "plan.json")["contract"]["vocabulary"]
    checks = []
    # 1. stage-c replication of the #96 records on every smoke-member inventory, seed 20260908
    deltas = []
    for inventory in INVENTORIES:
        if SMOKE_MEMBER not in universe[inventory]:
            continue
        items = universe[inventory][SMOKE_MEMBER]["inventory"]
        actions = action_batch(items)
        z0 = stack.carrier(SMOKE_MEMBER)
        for request in REQUESTS:
            model, controller = stack.models_for(request, SMOKE_SEED)
            z_end, _ = rollout_endpoints(model, controller, z0, actions, t96.arm_spec(request))
            rows = [stack.costs.row(item["ordinal"], z_end[slot], z0) for slot, item in enumerate(items)]
            retained = t96.read_record(SOURCE96, inventory, SMOKE_MEMBER, SMOKE_SEED, request)
            reference = {row["ordinal"]: row["predicted_cost"] for row in retained["decision"]["ranking"]}
            for row in rows:
                if (row["count"] is None) != (reference[row["ordinal"]] is None):
                    deltas.append(math.inf)
                elif row["count"] is not None:
                    deltas.append(abs(row["count"] - reference[row["ordinal"]]))
    checks.append({"control": "stage_c_replicates_issue_96_records", "member": SMOKE_MEMBER,
                   "seed": SMOKE_SEED, "requests": list(REQUESTS), "candidates": len(deltas),
                   "max_abs_count_delta": max(deltas), "tolerance": COST_TOLERANCE,
                   "ok": max(deltas) <= COST_TOLERANCE})
    # 2. evidence extraction on the smoke member's grid shots (outcome-free)
    grid = universe[PRIMARY][SMOKE_MEMBER]
    extracted, walls = [], []
    for ordinal in sorted(grid["verdicts"])[:3]:
        record = trace_record(sources, PRIMARY, SMOKE_MEMBER, grid, ordinal)
        began = time.monotonic()
        extracted.append(extract(stack.adapter, vocabulary, record))
        walls.append(time.monotonic() - began)
    anchor = anchors[SMOKE_MEMBER]
    frame0_ok = all(e["frame0_sha256"] == anchor["sha256"] for e in extracted)
    checks.append({"control": "frame0_equals_sealed_anchor", "shots": len(extracted),
                   "anchor_sha256": anchor["sha256"],
                   "frame0_sha256": [e["frame0_sha256"] for e in extracted], "ok": frame0_ok})
    parsed_anchor = stack.carrier(SMOKE_MEMBER).float().cpu()
    parsed0_delta = max(float((e["parsed"][0] - parsed_anchor).abs().max()) for e in extracted)
    checks.append({"control": "parsed_position0_equals_anchor_carrier", "max_abs_delta": parsed0_delta,
                   "tolerance": PARSE_BATCH_TOLERANCE, "ok": parsed0_delta <= PARSE_BATCH_TOLERANCE})
    engine0 = extracted[0]["engine_start"]
    engine_spread = max(float((e["engine_start"] - engine0).abs().max()) for e in extracted)
    agreement = []
    for index, name in enumerate(vocabulary):
        base = 2 + 13 * index
        engine_present = float(engine0[base]) >= 0.5
        parsed_present = float(parsed_anchor[base]) >= 0.5
        distance = (math.hypot(float(engine0[base + 5] - parsed_anchor[base + 5]),
                               float(engine0[base + 6] - parsed_anchor[base + 6]))
                    if engine_present and parsed_present else None)
        agreement.append({"slot": name, "engine_presence": float(engine0[base]),
                          "parsed_presence": float(parsed_anchor[base]),
                          "center_distance": distance})
    # projection geometry: the slot the parser is most confident about must land on the
    # engine-projected center (perception errors on other slots are the measured quantity)
    confident = max((row for row in agreement if row["center_distance"] is not None),
                    key=lambda row: row["parsed_presence"], default=None)
    geometry_ok = (confident is not None
                   and confident["center_distance"] <= PROJECTION_CENTER_TOLERANCE)
    checks.append({"control": "engine_projection_geometry_at_anchor", "member": SMOKE_MEMBER,
                   "slots": agreement, "most_confident_parsed_slot": confident,
                   "engine_start_spread_across_shots": engine_spread,
                   "center_tolerance": PROJECTION_CENTER_TOLERANCE,
                   "ok": geometry_ok and engine_spread <= ENGINE_ANCHOR_TOLERANCE})
    # 3. teacher-forcing mechanics (every row reaches 225, fixed pairs consume one step)
    z_true = torch.stack([e["parsed"] for e in extracted])
    items = [next(i for i in grid["inventory"] if i["ordinal"] == o) for o in sorted(grid["verdicts"])[:3]]
    actions = action_batch(items)
    mechanics = []
    for request in REQUESTS:
        model, controller = stack.models_for(request, SMOKE_SEED)
        z_end, timing = teacher_forced(model, controller, z_true, actions, t96.arm_spec(request))
        mechanics.append({"request": request, "finite": int(torch.isfinite(z_end).all(dim=1).sum()),
                          "transition_calls": timing["transition_calls"],
                          "controller_calls": timing["controller_calls"],
                          "ok": timing["transition_calls"] == len(items)})
    checks.append({"control": "teacher_forcing_mechanics", "requests": mechanics,
                   "ok": all(row["ok"] for row in mechanics)})
    shots = len(candidates(universe))
    projection = {"shots": shots, "extraction_seconds_per_shot": float(np.mean(walls)),
                  "projected_extraction_seconds": float(np.mean(walls)) * shots}
    ok = all(check["ok"] for check in checks)
    evidence = {"schema": SCHEMA_SMOKE, "identity": IDENTITY, "run_at": utc_now(), "device": DEVICE,
                "checks": checks, "projection": projection,
                "anchor_evidence_ok": all(row["ok"] for row in anchor_evidence),
                "reading": ("infrastructure assertions only; no verdict was joined and no AUC, tie "
                            "or top-1 statistic was computed"), "ok": ok}
    write_json(output / "smoke.json", evidence)
    log(f"smoke {'PASSED' if ok else 'FAILED'}; extraction {projection['extraction_seconds_per_shot']:.1f}"
        f" s/shot, projected {projection['projected_extraction_seconds']:.0f} s for {shots} shots")
    for check in checks:
        log(f"  {check['control']}: {check['ok']}")
    if not ok:
        raise ValueError("smoke control differs; STOP RULE: abort before the freeze")
    return 0


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def input_bindings(output):
    names = {
        "issue_96_plan": SOURCE96 / "plan.json",
        "issue_96_summary": SOURCE96 / "summary.json",
        "issue_96_runner": ROOT / "scripts/run_tau_ad_within_checkpoint.py",
        "issue_93_plan": p93.OUTPUT / "plan.json",
        "issue_93_anchor_retention": p93.OUTPUT / "anchor_retention.json",
        "issue_94_plan": p94.OUTPUT / "plan.json",
        "issue_87_plan": wlc.EIGHTY_SEVEN / "plan.json",
        "issue_89_summary": wlc.EIGHTY_NINE / "summary.json",
        "issue_77_dynamics_plan": DYNAMICS / "plan.json",
        "smoke": Path(output) / "smoke.json",
    }
    return [{"name": name, "artifact": str(Path(path).relative_to(ROOT)), "sha256": sha256_of(path)}
            for name, path in names.items()]


def frozen_plan(output, frozen_at, smoke_evidence, universe, members):
    cohorts = cohort_members(members)
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "role": "terminal",
        "issue": 99, "frozen_at": frozen_at, "frozen_before_scoring_run": True,
        "issue_64_authorized": False, "validation_command": VALIDATION_COMMAND, "runner": RUNNER,
        "runner_sha256_at_freeze": sha256_of(ROOT / RUNNER), "engine_seconds": 0,
        "question": ("where along engine state -> perception -> carrier -> predictor rollout -> cost "
                     "is the launch-ranking signal lost on the four engine-verdicted #96 "
                     "inventories, and does a tie-free cost read from the same endpoint carrier "
                     "remove the count cost's ties?"),
        "disclosure": ("Known before freeze and not re-derived here: the #96 stage-c count-cost AUCs "
                       "of every request (published), the #109 held-out re-score of the same "
                       "records, and the #85 finding that the parsed-endpoint count cost "
                       "disagrees with engine truth in closed loop (offset 600, 42/47 proxy-only "
                       "successes). No stage a/ap/b/d statistic and no tie-free statistic was "
                       "computed before freeze; the smoke joins no verdict."),
        "universe": {
            "inventories": t96.INVENTORY_SOURCES, "primary": PRIMARY,
            "members": {i: sorted(universe[i]) for i in INVENTORIES},
            "mixed_members": {i: sorted(m for m, e in universe[i].items() if t96.mixed(e))
                              for i in INVENTORIES},
            "cohorts": cohorts,
            "cohort_rule": ("held_out = #96 members outside issue-77-n1-dynamics-v1 n1_lineages "
                            "predictor and controller pools; all = every #96 member; reported "
                            "side by side, never pooled"),
            "verdict_bearing_shots": len(candidates(universe)),
            "seeds": list(SEEDS), "requests": list(REQUESTS), "stages": list(STAGES),
            "endpoint_records": len(scheduled_members(universe)),
            "model_records": len(scheduled_cells(universe)),
            "mixed_rule": ("a (inventory, member) is mixed iff its retained verdicts hold at least "
                           "one success and one failure; cells = mixed member x 3 seeds (seed-free "
                           "stages a/ap repeat the member's value in each seed cell, so "
                           "member-clustered means are unchanged)"),
        },
        "stages": STAGE_DEFINITIONS,
        "costs": COST_DEFINITIONS, "primary_cost": PRIMARY_COST,
        "cost_rules": {"tie_rule": "argmin cost, ties to the lower ordinal (top-1)",
                       "typed_failures": ("a nonfinite endpoint carrier is a retained typed failure "
                                          "(excluded, inherits no verdict), never retried"),
                       "start_carrier": ("a and b: the engine-projected position-0 carrier; ap, "
                                         "c and d: the parsed position-0 carrier")},
        "held_identical": {
            "checkpoints": "the #96 per-seed predictor.pt / controller.pt (issue-87 bindings)",
            "anchors": "the #96 sealed parsed anchors (run_launch_power_probe.frozen_anchor)",
            "engine_anchor": ("the engine-projected position-0 carrier of the member's first "
                              "verdict-bearing executed shot in frozen order (inventory, ordinal); "
                              f"every other shot of the member must agree within "
                              f"{ENGINE_ANCHOR_TOLERANCE}"),
            "inventories_and_verdicts": "t96.build_universe, unchanged",
            "endpoint": ENDPOINT, "positions": "position k = fixed step 30000 + 50 k",
        },
        "estimands": {
            "cell_auc": ("within-cell P(successful candidate has lower cost than failed candidate), "
                         "ties 0.5, over admissible candidates (verdict AND finite cost) "
                         "(run_launch_power_probe.cell_auc)"),
            "stage_auc": ("per cohort x inventory x cost x stage x request: member-clustered "
                          "percentile bootstrap of the cell AUCs over mixed cells"),
            "attribution": ("paired per cell, member-clustered: endpoint_perception = a - ap; "
                            "one_step_dynamics(r) = ap - d(r); compounding(r) = d(r) - c(r); "
                            "anchor_perception(r) = b(r) - c(r); total(r) = a - c(r); request-mean "
                            "versions average the 14 requests within a cell first"),
            "ties": ("tied cell = every admissible cost of the cell equal (the #96 G4 definition); "
                     "tied pair share = success/failure pairs with equal cost over all such pairs"),
            "top1": "argmin-cost candidate succeeds; pooled count over cells with chance reported",
            "interval": (f"member-clustered percentile bootstrap, {t96.BOOTSTRAP_DRAWS} draws, PCG64 "
                         f"seed {t96.BOOTSTRAP_SEED}, quantiles {list(t96.INTERVAL_QUANTILES)}, "
                         "DESCRIPTIVE"),
            "perception_fidelity": ("outcome-free, per cohort: parser presence (>= 0.5) against "
                                    "engine presence per slot kind at positions 0 and 225 over "
                                    "every verdict-bearing shot; accuracy and mean absolute "
                                    "presence error"),
            "compute_accounting": ("per request x stage: records, candidates, transition calls, "
                                   "controller calls, linear MACs, GPU seconds, active parameters; "
                                   "per record wall seconds; evidence extraction parse calls and "
                                   "wall seconds"),
        },
        "gate_a": {
            "scope": f"grid, cohort all, cost {PRIMARY_COST}, request-mean",
            "drops": {"cost": "1 - AUC(a)",
                      "perception": "max(endpoint_perception, request-mean anchor_perception)",
                      "dynamics": "request-mean (AUC(ap) - AUC(c)) = one_step_dynamics + compounding"},
            "dominant": "argmax of the three point drops; ties in the order dynamics, perception, cost",
            "fix": {"dynamics": ("retrain the predictor with a decision-aware auxiliary loss "
                                 "(pairwise ranking of candidate endpoints on fit lineages)"),
                    "perception": "improve the slot encoder",
                    "cost": "redefine the decision cost before any model work"},
            "held_out": "the same drops on the held-out cohort are reported, not used for the choice",
        },
        "decision_rule": {
            "Q1_attribution": (f"supported iff guards pass AND the dominant drop's lower bound > 0; "
                               f"not_supported_by_this_experiment iff guards pass AND the dominant "
                               f"drop's upper bound < {DROP_MARGIN}; otherwise {STOP_TOKEN}"),
            "Q2_tie_free_cost": (f"supported iff guards pass AND at stage c on mixed grid cells "
                                 f"(cohort all) every request's tied-cell share under the tie-free "
                                 f"cost is < {TIE_SHARE_TARGET}; not_supported_by_this_experiment "
                                 f"iff guards pass and some request is >= {TIE_SHARE_TARGET}; "
                                 f"otherwise {STOP_TOKEN}"),
            "baseline_target": (f"reported, not a token: any request at stage c, held-out grid, "
                                f"tie-free cost with AUC >= {TARGET_AUC} and lower bound > "
                                f"{TARGET_LOWER}; the issue's target token is decided after step 3"),
        },
        "guards": {
            "G1": (f"stage-c count cost reproduces every #96 record of the 14 requests within "
                   f"{COST_TOLERANCE} (same nonfinite exclusions)"),
            "G2": "every executed shot's position-0 agent frame is byte-identical to the member's sealed anchor",
            "G3": f"engine-projected position-0 carriers agree across a member's shots within {ENGINE_ANCHOR_TOLERANCE}",
            "G4": (f"verdict resolution 1.0 and nonfinite typed-failure share <= "
                   f"{NONFINITE_SHARE_MAX} on mixed grid cells for every stage and request"),
            "caps": f"run GPU seconds <= {GPU_CAP_SECONDS}, run wall seconds <= {WALL_CAP_SECONDS}",
            "failure": f"any guard failure -> {STOP_TOKEN} for Q1 and Q2",
        },
        "prohibited": ["tuning the tie-free cost weight after any verdict join",
                       "adding stages, requests or cohorts after scoring",
                       "retrying typed failures", "re-opening any prior disposition (#15-#98, #109)"],
        "caps": {"gpu_seconds": GPU_CAP_SECONDS, "wall_seconds": WALL_CAP_SECONDS,
                 "engine_seconds": 0, "stop": STOP_TOKEN},
        "smoke_evidence": smoke_evidence,
        "inputs": input_bindings(output),
        "claim_boundary": ("decision-only re-scoring of retained engine-verdicted executions and the "
                           "frozen #77 N1 checkpoints; stage d reads post-launch frames and is a "
                           "diagnostic, not a decision-time ranker; development/exposed N1 members "
                           "only (held-out = not fit by #77 predictors or controllers, still "
                           "exposed through earlier tickets); no engine access, rendering or "
                           "retraining; inventories and cohorts are never pooled; every interval "
                           "is DESCRIPTIVE"),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-99 protocol")
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
    if (output / "records").exists() or (output / "evidence").exists():
        raise ValueError("records exist before the freeze; refusing")
    evidence = read_json(output / "smoke.json")
    if not evidence.get("ok"):
        raise ValueError("smoke failed; STOP RULE: no freeze")
    sources = t96.load_sources()
    members, universe, _ = t96.build_universe(sources)
    plan = frozen_plan(output, utc_now(), evidence, universe, members)
    (output / "plan.json").write_text(json_text(plan))
    load_plan(output)
    log(f"frozen plan published: {plan['universe']['verdict_bearing_shots']} shots, "
        f"{plan['universe']['endpoint_records']} endpoint + {plan['universe']['model_records']} "
        "model records scheduled; no outcome statistic computed")
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def load_ledger(output):
    path = Path(output) / "ledger.json"
    if path.is_file():
        return read_json(path)
    return {"schema": SCHEMA_LEDGER, "identity": IDENTITY, "status": "running", "evidence": 0,
            "endpoint_records": 0, "model_records": 0, "gpu_seconds_elapsed": 0.0,
            "extraction_seconds_elapsed": 0.0, "wall_seconds_elapsed": 0.0}


def engine_anchors(output, universe):
    """Member engine anchor = first shot in frozen order; spread over the member's shots."""
    anchors, spread = {}, {}
    for inventory, member, ordinal in candidates(universe):
        start = load_evidence(output, inventory, member, ordinal)["engine_start"]
        if member not in anchors:
            anchors[member] = {"carrier": start, "source": candidate_key(inventory, member, ordinal)}
            spread[member] = 0.0
        else:
            spread[member] = max(spread[member],
                                 float((start - anchors[member]["carrier"]).abs().max()))
    return anchors, spread


def endpoint_record(output, universe, costs, inventory, member):
    entry = universe[inventory][member]
    stages = {"a": [], "ap": []}
    shots = {}
    for ordinal in sorted(entry["verdicts"]):
        evidence = load_evidence(output, inventory, member, ordinal)
        stages["a"].append(costs.row(ordinal, evidence["engine_end"], evidence["engine_start"]))
        stages["ap"].append(costs.row(ordinal, evidence["parsed"][ENDPOINT], evidence["parsed"][0]))
        shots[str(ordinal)] = {k: evidence[k] for k in ("source", "frame_count", "last_index",
                                                        "end_index", "absorbed", "frame0_sha256",
                                                        "parse_calls")}
    return {"schema": SCHEMA_ENDPOINT, "plan_identity": IDENTITY,
            "cell": {"inventory": inventory, "member": member, "state": entry["state"]},
            "stages": stages, "shots": shots, "engine_seconds": 0}


def model_record(output, universe, stack, engine_anchor, inventory, member, seed):
    entry = universe[inventory][member]
    items = entry["inventory"]
    actions = action_batch(items)
    parsed_anchor = stack.carrier(member).float()
    verdicted = sorted(entry["verdicts"])
    forced_items = [next(item for item in items if item["ordinal"] == o) for o in verdicted]
    forced_actions = action_batch(forced_items)
    evidence = [load_evidence(output, inventory, member, o) for o in verdicted]
    z_true = torch.stack([e["parsed"] for e in evidence])
    stages = {stage: {} for stage in MODEL_STAGES}
    for request in REQUESTS:
        model, controller = stack.models_for(request, seed)
        spec = t96.arm_spec(request)
        for stage, z0 in (("c", parsed_anchor), ("b", engine_anchor)):
            z_end, timing = rollout_endpoints(model, controller, z0.to(DEVICE), actions, spec)
            rows = [stack.costs.row(item["ordinal"], z_end[slot].cpu(), z0.cpu())
                    for slot, item in enumerate(items)]
            stages[stage][request] = {"rows": rows, "candidates": len(items), **timing}
        z_end, timing = teacher_forced(model, controller, z_true, forced_actions, spec)
        rows = [stack.costs.row(o, z_end[slot].cpu(), evidence[slot]["parsed"][0])
                for slot, o in enumerate(verdicted)]
        stages["d"][request] = {"rows": rows, "candidates": len(verdicted), **timing}
    return {"schema": SCHEMA_MODEL, "plan_identity": IDENTITY,
            "cell": {"inventory": inventory, "member": member, "state": entry["state"], "seed": seed},
            "stages": stages, "engine_seconds": 0}


def run(output):
    output = Path(output)
    plan = load_plan(output)
    sources = t96.load_sources()
    members, universe, _ = t96.build_universe(sources)
    if len(candidates(universe)) != plan["universe"]["verdict_bearing_shots"]:
        raise ValueError("universe differs from the frozen plan")
    anchors, anchor_evidence = t96.member_anchors(sources, members)
    if not all(row["ok"] for row in anchor_evidence):
        raise ValueError(f"{STOP_TOKEN}: anchor sha256 equality failed")
    stack = Stack(sources, anchors)
    vocabulary = read_json(DYNAMICS / "plan.json")["contract"]["vocabulary"]
    ledger = load_ledger(output)
    began = time.monotonic()
    wall_base = ledger["wall_seconds_elapsed"]

    def check_caps():
        ledger["wall_seconds_elapsed"] = wall_base + time.monotonic() - began
        if (ledger["gpu_seconds_elapsed"] > GPU_CAP_SECONDS
                or ledger["wall_seconds_elapsed"] > WALL_CAP_SECONDS):
            ledger["status"] = "cap_exceeded"
            write_json(output / "ledger.json", ledger)
            raise ValueError(f"{STOP_TOKEN}: compute cap exceeded")

    shots = candidates(universe)
    pending = [c for c in shots if not evidence_path(output, *c).is_file()]
    log(f"evidence: {len(shots) - len(pending)}/{len(shots)} retained; extracting {len(pending)}")
    extraction_began = time.monotonic()
    for done, (inventory, member, ordinal) in enumerate(pending, 1):
        check_caps()
        started = time.monotonic()
        entry = universe[inventory][member]
        evidence = extract(stack.adapter, vocabulary,
                           trace_record(sources, inventory, member, entry, ordinal))
        evidence["wall_seconds"] = time.monotonic() - started
        path = evidence_path(output, inventory, member, ordinal)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(evidence, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
        ledger["evidence"] += 1
        ledger["extraction_seconds_elapsed"] += evidence["wall_seconds"]
        if done % 20 == 0 or done == len(pending):
            elapsed = time.monotonic() - extraction_began
            write_json(output / "ledger.json", ledger)
            log(f"evidence {done}/{len(pending)}; {elapsed:.0f}s elapsed; eta "
                f"{elapsed / done * (len(pending) - done):.0f}s")

    engine, spread = engine_anchors(output, universe)
    write_json(output / "engine_anchors.json",
               {"schema": "issue_99_engine_anchor_evidence_v1", "plan_identity": IDENTITY,
                "members": {m: {"source": engine[m]["source"], "max_abs_spread": spread[m]}
                            for m in sorted(engine)}})
    for inventory, member in scheduled_members(universe):
        path = endpoint_path(output, inventory, member)
        if path.is_file():
            continue
        started = time.monotonic()
        record = endpoint_record(output, universe, stack.costs, inventory, member)
        record["wall_seconds"] = time.monotonic() - started
        write_json(path, record)
        ledger["endpoint_records"] += 1
    cells = scheduled_cells(universe)
    model_began = time.monotonic()
    pending = [c for c in cells if not model_path(output, *c).is_file()]
    for done, (inventory, member, seed) in enumerate(pending, 1):
        check_caps()
        started = time.monotonic()
        record = model_record(output, universe, stack, engine[member]["carrier"], inventory, member,
                              seed)
        record["wall_seconds"] = time.monotonic() - started
        write_json(model_path(output, inventory, member, seed), record)
        ledger["model_records"] += 1
        ledger["gpu_seconds_elapsed"] += sum(block["gpu_seconds"] for stage in record["stages"].values()
                                             for block in stage.values())
        if done % 10 == 0 or done == len(pending):
            elapsed = time.monotonic() - model_began
            ledger["wall_seconds_elapsed"] = wall_base + time.monotonic() - began
            write_json(output / "ledger.json", ledger)
            log(f"model records {done}/{len(pending)}; gpu {ledger['gpu_seconds_elapsed']:.0f}s; "
                f"eta {elapsed / done * (len(pending) - done):.0f}s")
    ledger["status"] = "terminal"
    ledger["wall_seconds_elapsed"] = wall_base + time.monotonic() - began
    write_json(output / "ledger.json", ledger)
    log(f"run complete: {ledger['evidence']} evidence, {ledger['endpoint_records']} endpoint, "
        f"{ledger['model_records']} model records; gpu {ledger['gpu_seconds_elapsed']:.0f}s; wall "
        f"{ledger['wall_seconds_elapsed']:.0f}s")
    return 0


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def read_record(path, schema):
    record = read_json(path)
    if record.get("schema") != schema or record.get("plan_identity") != IDENTITY:
        raise ValueError(f"record {path} binding differs")
    return record


def view(entry, rows, cost):
    """One cell under one cost: AUC, ties, top-1 (t96.cell_view semantics)."""
    values = {row["ordinal"]: row[cost] for row in rows}
    verdicts = entry["verdicts"]
    admissible = sorted(((o, values[o]) for o in verdicts if values.get(o) is not None),
                        key=lambda pair: (pair[1], pair[0]))
    pos = [c for o, c in admissible if verdicts[o]]
    neg = [c for o, c in admissible if not verdicts[o]]
    return {"resolved": sum(1 for o in verdicts if o in values), "verdict_slots": len(verdicts),
            "nonfinite": sum(1 for o in verdicts if o in values and values[o] is None),
            "auc": cell_auc(pos, neg) if pos and neg else None,
            "top1_hit": bool(verdicts[admissible[0][0]]) if admissible else None,
            "chance_top1": len(pos) / len(admissible) if admissible else None,
            "all_tied": len({c for _, c in admissible}) <= 1,
            "pairs": len(pos) * len(neg), "tied_pairs": sum(1 for p in pos for n in neg if p == n)}


def ranker_keys():
    return [("a", None), ("ap", None)] + [(stage, request) for stage in MODEL_STAGES
                                          for request in REQUESTS]


def ranker_name(stage, request):
    return stage if request is None else f"{stage}:{request}"


def cell_rows(output, universe, inventory, members):
    rows = []
    for member in members:
        entry = universe[inventory].get(member)
        if entry is None or not t96.mixed(entry):
            continue
        endpoint = read_record(endpoint_path(output, inventory, member), SCHEMA_ENDPOINT)
        for seed in SEEDS:
            model = read_record(model_path(output, inventory, member, seed), SCHEMA_MODEL)
            views = {}
            for cost in COSTS:
                for stage, request in ranker_keys():
                    rows_ = (endpoint["stages"][stage] if request is None
                             else model["stages"][stage][request]["rows"])
                    views[(cost, stage, request)] = view(entry, rows_, cost)
            rows.append({"member": member, "seed": seed, "views": views})
    return rows


def block_of(rows, key, field="auc"):
    values = [(r["views"][key][field], r["member"]) for r in rows if r["views"][key][field] is not None]
    return bootstrap([float(v) for v, _ in values], [m for _, m in values])


def paired(rows, pieces):
    """Member-clustered bootstrap of a per-cell linear combination of AUCs.

    pieces: list of (weight, key); a cell enters only if every piece is defined."""
    values, clusters = [], []
    for r in rows:
        parts = [r["views"][key]["auc"] for _, key in pieces]
        if any(p is None for p in parts):
            continue
        values.append(float(sum(w * p for (w, _), p in zip(pieces, parts))))
        clusters.append(r["member"])
    block = bootstrap(values, clusters)
    return {"cells": len(values), "members": len(set(clusters)), "estimate": block}


def attribution(rows, cost):
    n = len(REQUESTS)
    a, ap = (cost, "a", None), (cost, "ap", None)

    def mean_over(stage_pieces):
        return [(w / n, (cost, stage, request)) for request in REQUESTS for w, stage in stage_pieces]

    out = {"endpoint_perception": paired(rows, [(1, a), (-1, ap)]),
           "cost_ceiling_auc": paired(rows, [(1, a)]),
           "request_mean": {
               "one_step_dynamics": paired(rows, [(1, ap)] + mean_over([(-1, "d")])),
               "compounding": paired(rows, mean_over([(1, "d"), (-1, "c")])),
               "anchor_perception": paired(rows, mean_over([(1, "b"), (-1, "c")])),
               "dynamics": paired(rows, [(1, ap)] + mean_over([(-1, "c")])),
               "total": paired(rows, [(1, a)] + mean_over([(-1, "c")]))},
           "per_request": {}}
    for request in REQUESTS:
        d, c, b = (cost, "d", request), (cost, "c", request), (cost, "b", request)
        out["per_request"][request] = {
            "one_step_dynamics": paired(rows, [(1, ap), (-1, d)]),
            "compounding": paired(rows, [(1, d), (-1, c)]),
            "anchor_perception": paired(rows, [(1, b), (-1, c)]),
            "total": paired(rows, [(1, a), (-1, c)])}
    return out


def gate_a(block):
    ceiling = block["cost_ceiling_auc"]["estimate"]
    endpoint = block["endpoint_perception"]["estimate"]
    anchor = block["request_mean"]["anchor_perception"]["estimate"]
    dynamics = block["request_mean"]["dynamics"]["estimate"]
    if None in (ceiling, endpoint, anchor, dynamics):
        return None
    cost_drop = {"mean": 1 - ceiling["mean"],
                 "interval": [1 - ceiling["interval"][1], 1 - ceiling["interval"][0]]}
    perception_source = ("endpoint_perception" if endpoint["mean"] >= anchor["mean"]
                         else "anchor_perception")
    perception = endpoint if perception_source == "endpoint_perception" else anchor
    drops = {"dynamics": {"mean": dynamics["mean"], "interval": dynamics["interval"]},
             "perception": {"mean": perception["mean"], "interval": perception["interval"],
                            "source": perception_source},
             "cost": cost_drop}
    order = ("dynamics", "perception", "cost")
    dominant = max(order, key=lambda k: (drops[k]["mean"], -order.index(k)))
    return {"drops": drops, "dominant": dominant}


def cohort_tables(output, universe, members):
    tables = {}
    for inventory in INVENTORIES:
        rows = cell_rows(output, universe, inventory, members)
        rankers = {}
        for cost in COSTS:
            for stage, request in ranker_keys():
                key = (cost, stage, request)
                views = [r["views"][key] for r in rows]
                pairs = sum(v["pairs"] for v in views)
                chance = [v["chance_top1"] for v in views if v["chance_top1"] is not None]
                rankers.setdefault(cost, {})[ranker_name(stage, request)] = {
                    "auc": block_of(rows, key),
                    "tied_cells": sum(v["all_tied"] for v in views),
                    "tied_share": sum(v["all_tied"] for v in views) / len(views) if views else None,
                    "tied_pair_share": sum(v["tied_pairs"] for v in views) / pairs if pairs else None,
                    "top1_hits": sum(1 for v in views if v["top1_hit"]),
                    "chance_top1_mean": float(np.mean(chance)) if chance else None,
                    "verdict_resolution": (sum(v["resolved"] for v in views)
                                           / sum(v["verdict_slots"] for v in views)) if views else None,
                    "nonfinite_share": (sum(v["nonfinite"] for v in views)
                                        / sum(v["verdict_slots"] for v in views)) if views else None,
                }
        tables[inventory] = {
            "mixed_members": sorted({r["member"] for r in rows}), "mixed_cells": len(rows),
            "rankers": rankers,
            "attribution": {cost: attribution(rows, cost) for cost in COSTS},
            "rows": [{"member": r["member"], "seed": r["seed"],
                      "auc": {cost: {ranker_name(s, q): r["views"][(cost, s, q)]["auc"]
                                     for s, q in ranker_keys()} for cost in COSTS}} for r in rows],
        }
        for cost in COSTS:
            tables[inventory]["attribution"][cost]["gate_a"] = gate_a(tables[inventory]["attribution"][cost])
    return tables


def perception_fidelity(output, universe, cohorts):
    """Outcome-free parser-vs-engine presence agreement at positions 0 and 225, per slot kind."""
    vocabulary = read_json(DYNAMICS / "plan.json")["contract"]["vocabulary"]
    kinds = sorted({name.split(":", 1)[0] for name in vocabulary})
    out = {}
    for cohort in COHORTS:
        tally = {when: {kind: {"slots": 0, "agree": 0, "engine_present": 0, "parsed_present": 0,
                               "abs_error_sum": 0.0} for kind in kinds} for when in ("start", "end")}
        shots = 0
        for inventory, member, ordinal in candidates(universe):
            if member not in cohorts[cohort]:
                continue
            shots += 1
            evidence = load_evidence(output, inventory, member, ordinal)
            pairs = {"start": (evidence["parsed"][0], evidence["engine_start"]),
                     "end": (evidence["parsed"][ENDPOINT], evidence["engine_end"])}
            for when, (parsed, engine) in pairs.items():
                for index, name in enumerate(vocabulary):
                    block = tally[when][name.split(":", 1)[0]]
                    p, e = float(parsed[2 + 13 * index]), float(engine[2 + 13 * index])
                    block["slots"] += 1
                    block["agree"] += int((p >= 0.5) == (e >= 0.5))
                    block["engine_present"] += int(e >= 0.5)
                    block["parsed_present"] += int(p >= 0.5)
                    block["abs_error_sum"] += abs(p - e)
        for when in tally.values():
            for block in when.values():
                block["accuracy"] = block["agree"] / block["slots"] if block["slots"] else None
                block["mean_abs_presence_error"] = (block.pop("abs_error_sum") / block["slots"]
                                                    if block["slots"] else None)
        out[cohort] = {"shots": shots, **tally}
    return out


def replication_guard(output, universe):
    worst, mismatched, compared = 0.0, 0, 0
    for inventory, member, seed in scheduled_cells(universe):
        model = read_record(model_path(output, inventory, member, seed), SCHEMA_MODEL)
        for request in REQUESTS:
            retained = t96.read_record(SOURCE96, inventory, member, seed, request)
            reference = {row["ordinal"]: row["predicted_cost"] for row in retained["decision"]["ranking"]}
            for row in model["stages"]["c"][request]["rows"]:
                compared += 1
                if (row["count"] is None) != (reference[row["ordinal"]] is None):
                    mismatched += 1
                elif row["count"] is not None:
                    worst = max(worst, abs(row["count"] - reference[row["ordinal"]]))
    return {"compared": compared, "exclusion_mismatches": mismatched, "max_abs_count_delta": worst,
            "tolerance": COST_TOLERANCE}


def accounting(output, universe, stack_params):
    per = {f"{stage}:{request}": {"records": 0, "candidates": 0, "transition_calls": 0,
                                  "controller_calls": 0, "linear_macs": 0, "gpu_seconds": 0.0}
           for stage in MODEL_STAGES for request in REQUESTS}
    walls, extraction = [], {"shots": 0, "parse_calls": 0, "wall_seconds": 0.0}
    for inventory, member, seed in scheduled_cells(universe):
        model = read_record(model_path(output, inventory, member, seed), SCHEMA_MODEL)
        walls.append(model["wall_seconds"])
        for stage in MODEL_STAGES:
            for request in REQUESTS:
                block, source = per[f"{stage}:{request}"], model["stages"][stage][request]
                block["records"] += 1
                for key in ("candidates", "transition_calls", "controller_calls", "linear_macs",
                            "gpu_seconds"):
                    block[key] += source[key]
    for inventory, member in scheduled_members(universe):
        endpoint = read_record(endpoint_path(output, inventory, member), SCHEMA_ENDPOINT)
        for shot in endpoint["shots"].values():
            extraction["shots"] += 1
            extraction["parse_calls"] += shot["parse_calls"]
    for request in REQUESTS:
        for stage in MODEL_STAGES:
            per[f"{stage}:{request}"]["active_parameters_per_transition"] = stack_params[
                t96.arm_family(request)]
    return {"per_ranker": per, "model_record_wall_seconds": {
        "records": len(walls), "sum": float(sum(walls)), "mean": float(np.mean(walls)),
        "max": float(max(walls))}, "extraction": extraction}


def parameter_counts():
    from world_model.training.cnn_hybrid import CNNHybridPredictor
    from world_model.training.matched_dynamics import ContinuousDynamics, parameter_count
    width = read_json(DYNAMICS / "plan.json")["capacity"]["continuous_width"]
    return {"hybrid": parameter_count(CNNHybridPredictor()),
            "continuous": parameter_count(ContinuousDynamics(width))}


def token_q1(gate, guards_ok):
    if not guards_ok or gate is None:
        return STOP_TOKEN
    low, high = gate["drops"][gate["dominant"]]["interval"]
    if low > 0:
        return "supported"
    if high < DROP_MARGIN:
        return "not_supported_by_this_experiment"
    return STOP_TOKEN


def compute_tables(output, plan):
    output = Path(output)
    sources = t96.load_sources()
    members, universe, _ = t96.build_universe(sources)
    cohorts = cohort_members(members)
    ledger = read_json(output / "ledger.json")
    per_cohort = {cohort: cohort_tables(output, universe, cohorts[cohort]) for cohort in COHORTS}
    replication = replication_guard(output, universe)
    anchor_file = read_json(output / "engine_anchors.json")
    frame0_mismatch = []
    anchors = {m: p94.frozen_anchor(m)["sha256"] for m in members}
    for inventory, member in scheduled_members(universe):
        endpoint = read_record(endpoint_path(output, inventory, member), SCHEMA_ENDPOINT)
        for ordinal, shot in endpoint["shots"].items():
            if shot["frame0_sha256"] != anchors[member]:
                frame0_mismatch.append(f"{inventory}/{member}/o{ordinal}")
    grid_all = per_cohort["all"][PRIMARY]["rankers"]
    worst_nonfinite = max(block["nonfinite_share"] for cost in COSTS
                          for block in grid_all[cost].values())
    worst_resolution = min(block["verdict_resolution"] for cost in COSTS
                           for block in grid_all[cost].values())
    spread = max(v["max_abs_spread"] for v in anchor_file["members"].values())
    guards = {
        "G1": {"observed": replication,
               "pass": replication["exclusion_mismatches"] == 0
               and replication["max_abs_count_delta"] <= COST_TOLERANCE},
        "G2": {"observed": {"mismatched_shots": frame0_mismatch}, "pass": not frame0_mismatch},
        "G3": {"observed": {"max_abs_spread": spread}, "pass": spread <= ENGINE_ANCHOR_TOLERANCE},
        "G4": {"observed": {"worst_verdict_resolution": worst_resolution,
                            "worst_nonfinite_share": worst_nonfinite},
               "pass": worst_resolution == 1.0 and worst_nonfinite <= NONFINITE_SHARE_MAX},
        "caps": {"observed": {"gpu_seconds": ledger["gpu_seconds_elapsed"],
                              "wall_seconds": ledger["wall_seconds_elapsed"],
                              "status": ledger["status"]},
                 "pass": (ledger["status"] == "terminal"
                          and ledger["gpu_seconds_elapsed"] <= GPU_CAP_SECONDS
                          and ledger["wall_seconds_elapsed"] <= WALL_CAP_SECONDS)},
    }
    guards_ok = all(g["pass"] for g in guards.values())
    gate = per_cohort["all"][PRIMARY]["attribution"][PRIMARY_COST]["gate_a"]
    q1 = token_q1(gate, guards_ok)
    tie_shares = {request: grid_all[PRIMARY_COST][f"c:{request}"]["tied_share"] for request in REQUESTS}
    count_ties = {request: grid_all["count"][f"c:{request}"]["tied_share"] for request in REQUESTS}
    q2 = (STOP_TOKEN if not guards_ok else "supported"
          if all(v < TIE_SHARE_TARGET for v in tie_shares.values())
          else "not_supported_by_this_experiment")
    held_grid = per_cohort["held_out"][PRIMARY]["rankers"][PRIMARY_COST]
    baseline = []
    for request in REQUESTS:
        block = held_grid[f"c:{request}"]["auc"]
        baseline.append({"request": request, "auc": block,
                         "meets": bool(block and block["mean"] >= TARGET_AUC
                                       and block["interval"][0] > TARGET_LOWER)})
    return {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY,
        "structure": {"shots": len(candidates(universe)),
                      "endpoint_records": len(scheduled_members(universe)),
                      "model_records": len(scheduled_cells(universe)),
                      "cohorts": cohorts},
        "guards": guards, "guards_pass": guards_ok,
        "Q1_attribution": {"gate_a": gate, "token": q1,
                           "fix": plan["gate_a"]["fix"][gate["dominant"]] if gate else None,
                           "held_out_gate_a": per_cohort["held_out"][PRIMARY]["attribution"][PRIMARY_COST]["gate_a"]},
        "Q2_tie_free_cost": {"tied_share_tie_free": tie_shares, "tied_share_count": count_ties,
                             "target": TIE_SHARE_TARGET, "token": q2},
        "baseline_target": {"scope": "stage c, held-out grid, tie-free cost", "requests": baseline,
                            "met": any(row["meets"] for row in baseline)},
        "cohorts": per_cohort,
        "perception_fidelity": perception_fidelity(output, universe, cohorts),
        "compute": {**accounting(output, universe, parameter_counts()),
                    "run_gpu_seconds": ledger["gpu_seconds_elapsed"],
                    "run_wall_seconds": ledger["wall_seconds_elapsed"],
                    "extraction_seconds": ledger["extraction_seconds_elapsed"], "engine_seconds": 0},
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def comparisons_csv(compute):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["id", "cohort", "inventory", "cost", "statistic", "value", "interval_low",
                     "interval_high", "interval_label", "cells", "members", "detail"])

    def put(identity, cohort, inventory, cost, statistic, value, block=None, cells="", members="",
            detail=""):
        writer.writerow([identity, cohort, inventory, cost, statistic, "" if value is None else value,
                         block["interval"][0] if block else "", block["interval"][1] if block else "",
                         "DESCRIPTIVE" if block else "", cells, members, detail])

    put("Q1_token", "all", PRIMARY, PRIMARY_COST, "attribution token", compute["Q1_attribution"]["token"],
        detail=json.dumps(compute["Q1_attribution"]["gate_a"], sort_keys=True))
    put("Q2_token", "all", PRIMARY, PRIMARY_COST, "tie-free cost token",
        compute["Q2_tie_free_cost"]["token"],
        detail=json.dumps(compute["Q2_tie_free_cost"]["tied_share_tie_free"], sort_keys=True))
    put("baseline_target_met", "held_out", PRIMARY, PRIMARY_COST, "baseline target met (stage c)",
        compute["baseline_target"]["met"])
    for guard, block in compute["guards"].items():
        put(f"guard_{guard}", "", "", "", f"{guard} pass", block["pass"],
            detail=json.dumps(block["observed"], sort_keys=True))
    for cohort, tables in compute["cohorts"].items():
        for inventory in INVENTORIES:
            inv = tables[inventory]
            for cost in COSTS:
                for name, block in inv["rankers"][cost].items():
                    auc = block["auc"]
                    put(f"{cohort}_{inventory}_{cost}_auc_{name}", cohort, inventory, cost,
                        f"AUC {name}", auc and auc["mean"], auc, auc["units"] if auc else 0,
                        auc["clusters"] if auc else 0,
                        json.dumps({k: block[k] for k in ("tied_cells", "tied_share", "tied_pair_share",
                                                          "top1_hits", "chance_top1_mean")},
                                   sort_keys=True))
                att = inv["attribution"][cost]
                for name in ("cost_ceiling_auc", "endpoint_perception"):
                    c = att[name]
                    put(f"{cohort}_{inventory}_{cost}_{name}", cohort, inventory, cost, name,
                        c["estimate"] and c["estimate"]["mean"], c["estimate"], c["cells"], c["members"])
                for name, c in att["request_mean"].items():
                    put(f"{cohort}_{inventory}_{cost}_mean_{name}", cohort, inventory, cost,
                        f"request-mean {name}", c["estimate"] and c["estimate"]["mean"], c["estimate"],
                        c["cells"], c["members"])
                for request, parts in att["per_request"].items():
                    for name, c in parts.items():
                        put(f"{cohort}_{inventory}_{cost}_{request}_{name}", cohort, inventory, cost,
                            f"{name} [{request}]", c["estimate"] and c["estimate"]["mean"],
                            c["estimate"], c["cells"], c["members"])
    for name, block in compute["compute"]["per_ranker"].items():
        put(f"compute_{name}", "", "", "", f"compute {name}", block["gpu_seconds"],
            detail=json.dumps({k: v for k, v in block.items() if k != "gpu_seconds"}, sort_keys=True))
    return buffer.getvalue()


def stage_table(add, tables, cost):
    add("| inventory | cells / members | a (engine endpoint) | ap (perceived endpoint) |")
    add("|---|---|---|---|")
    for inventory in INVENTORIES:
        inv = tables[inventory]
        r = inv["rankers"][cost]
        add(f"| {inventory} | {inv['mixed_cells']} / {len(inv['mixed_members'])} | "
            f"{fmt_interval(r['a']['auc'])} | {fmt_interval(r['ap']['auc'])} |")
    add("")
    add("| request | " + " | ".join(f"{i} {s}" for i in INVENTORIES for s in MODEL_STAGES) + " |")
    add("|---|" + "---|" * (len(INVENTORIES) * len(MODEL_STAGES)))
    for request in REQUESTS:
        add(f"| {request} | " + " | ".join(
            fmt_interval(tables[i]["rankers"][cost][f"{s}:{request}"]["auc"])
            for i in INVENTORIES for s in MODEL_STAGES) + " |")


def findings_md(plan, compute):
    lines = []
    add = lines.append
    q1, q2 = compute["Q1_attribution"], compute["Q2_tie_free_cost"]
    add("# Issue-99 decision-chain attribution and tie-free cost — findings")
    add("")
    add(f"- identity `{IDENTITY}`, plan v1 frozen {plan['frozen_at']} before any scoring run")
    add(f"- validation command: `{VALIDATION_COMMAND}`; zero engine seconds; endpoint {ENDPOINT}; "
        "every interval DESCRIPTIVE (member-clustered, 10000 draws, PCG64 7201)")
    add(f"- disclosure: {plan['disclosure']}")
    add(f"- tie-free cost (frozen): {COST_DEFINITIONS['tie_free']}")
    add("")
    add("## Dispositions")
    add("")
    gate = q1["gate_a"]
    if gate:
        drops = ", ".join(f"{k} {fmt(v['mean'])} [{fmt(v['interval'][0])}, {fmt(v['interval'][1])}]"
                          + (f" ({v['source']})" if "source" in v else "")
                          for k, v in gate["drops"].items())
        add(f"- **Q1 attribution (grid, all members, tie-free, request-mean): dominant stage "
            f"{gate['dominant']} -> {q1['token']}**; drops {drops}")
        add(f"- Gate A fix: {q1['fix']}")
    held = q1["held_out_gate_a"]
    if held:
        add(f"- held-out Gate A (reported, not used): dominant {held['dominant']}; "
            + ", ".join(f"{k} {fmt(v['mean'])}" for k, v in held["drops"].items()))
    add(f"- **Q2 tie-free cost: {q2['token']}**; stage-c tied-cell share on mixed grid cells "
        f"(tie-free / count): " + ", ".join(
            f"{r} {fmt(q2['tied_share_tie_free'][r], 3)}/{fmt(q2['tied_share_count'][r], 3)}"
            for r in REQUESTS))
    met = [row["request"] for row in compute["baseline_target"]["requests"] if row["meets"]]
    add(f"- baseline target (stage c, held-out grid, tie-free: AUC >= {TARGET_AUC}, lower > "
        f"{TARGET_LOWER}): {'met by ' + ', '.join(met) if met else 'not met by any request'}")
    add("")
    add("## Guards")
    add("")
    add("| guard | pass | observed |")
    add("|---|---|---|")
    for guard, block in compute["guards"].items():
        add(f"| {guard} | {block['pass']} | `{json.dumps(block['observed'], sort_keys=True)}` |")
    add("")
    for cohort in COHORTS:
        tables = compute["cohorts"][cohort]
        for cost in COSTS:
            add(f"## Stage AUC — cohort {cohort}, cost {cost}")
            add("")
            stage_table(add, tables, cost)
            add("")
            add(f"### Attribution (paired, request-mean) — cohort {cohort}, cost {cost}")
            add("")
            add("| inventory | 1 - AUC(a) ceiling loss | a - ap endpoint perception | ap - d one-step | "
                "d - c compounding | b - c anchor perception | ap - c dynamics | a - c total | Gate A |")
            add("|---|---|---|---|---|---|---|---|---|")
            for inventory in INVENTORIES:
                att = tables[inventory]["attribution"][cost]
                ceiling = att["cost_ceiling_auc"]["estimate"]
                loss = (None if ceiling is None else
                        {"mean": 1 - ceiling["mean"],
                         "interval": [1 - ceiling["interval"][1], 1 - ceiling["interval"][0]]})
                m = att["request_mean"]
                add(f"| {inventory} | {fmt_interval(loss)} | "
                    f"{fmt_interval(att['endpoint_perception']['estimate'])} | "
                    f"{fmt_interval(m['one_step_dynamics']['estimate'])} | "
                    f"{fmt_interval(m['compounding']['estimate'])} | "
                    f"{fmt_interval(m['anchor_perception']['estimate'])} | "
                    f"{fmt_interval(m['dynamics']['estimate'])} | {fmt_interval(m['total']['estimate'])} | "
                    f"{att['gate_a']['dominant'] if att['gate_a'] else 'n/a'} |")
            add("")
            add(f"### Ties and top-1 — cohort {cohort}, cost {cost} (grid)")
            add("")
            add("| ranker | tied cells | tied share | tied pair share | top-1 hits / cells | chance |")
            add("|---|---|---|---|---|---|")
            grid = tables[PRIMARY]
            for name, block in grid["rankers"][cost].items():
                add(f"| {name} | {block['tied_cells']} | {fmt(block['tied_share'], 3)} | "
                    f"{fmt(block['tied_pair_share'], 3)} | {block['top1_hits']}/{grid['mixed_cells']} | "
                    f"{fmt(block['chance_top1_mean'], 3)} |")
            add("")
    add("## Perception fidelity (outcome-free; parser presence >= 0.5 vs engine presence)")
    add("")
    add("| cohort | position | kind | slots | accuracy | engine present | parsed present | "
        "mean abs presence error |")
    add("|---|---|---|---|---|---|---|---|")
    for cohort, block in compute["perception_fidelity"].items():
        for when in ("start", "end"):
            for kind, row in block[when].items():
                add(f"| {cohort} ({block['shots']} shots) | {when} | {kind} | {row['slots']} | "
                    f"{fmt(row['accuracy'], 3)} | {row['engine_present']} | {row['parsed_present']} | "
                    f"{fmt(row['mean_abs_presence_error'], 3)} |")
    add("")
    add("## Compute (reported, not matched)")
    add("")
    add("| ranker | records | candidates | transition calls | controller calls | linear MACs | "
        "active params / transition | GPU s |")
    add("|---|---|---|---|---|---|---|---|")
    for name, block in compute["compute"]["per_ranker"].items():
        add(f"| {name} | {block['records']} | {block['candidates']} | {block['transition_calls']} | "
            f"{block['controller_calls']} | {block['linear_macs']} | "
            f"{block['active_parameters_per_transition']} | {block['gpu_seconds']:.1f} |")
    add("")
    c = compute["compute"]
    add(f"- evidence extraction: {c['extraction']['shots']} shots, {c['extraction']['parse_calls']} "
        f"parser calls, {c['extraction_seconds']:.1f} s; model records wall sum "
        f"{c['model_record_wall_seconds']['sum']:.1f} s (mean {c['model_record_wall_seconds']['mean']:.2f}"
        f" s per member-seed state)")
    add(f"- run GPU {c['run_gpu_seconds']:.1f} s, wall {c['run_wall_seconds']:.1f} s (caps "
        f"{GPU_CAP_SECONDS:.0f} / {WALL_CAP_SECONDS:.0f} s); engine seconds 0")
    add("")
    add("## Claim boundary")
    add("")
    add(plan["claim_boundary"] + ".")
    add("")
    return "\n".join(lines)


def summary(plan, compute):
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "validation_command": VALIDATION_COMMAND, "question": plan["question"],
            "disclosure": plan["disclosure"], "costs": COST_DEFINITIONS,
            "guards": compute["guards"], "guards_pass": compute["guards_pass"],
            "Q1_attribution": compute["Q1_attribution"], "Q2_tie_free_cost": compute["Q2_tie_free_cost"],
            "baseline_target": compute["baseline_target"],
            "perception_fidelity": compute["perception_fidelity"],
            "cohorts": {cohort: {inventory: {k: v for k, v in block.items() if k != "rows"}
                                 for inventory, block in tables.items()}
                        for cohort, tables in compute["cohorts"].items()},
            "compute": compute["compute"], "claim_boundary": plan["claim_boundary"],
            "issue_64_authorized": False}


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
    log(f"published: Q1 {compute['Q1_attribution']['token']} (dominant "
        f"{compute['Q1_attribution']['gate_a'] and compute['Q1_attribution']['gate_a']['dominant']}); "
        f"Q2 {compute['Q2_tie_free_cost']['token']}; guards pass {compute['guards_pass']}")
    return 0


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    sources = t96.load_sources()
    members, universe, _ = t96.build_universe(sources)
    costs = Costs(wlc.load_objective())
    for inventory, member in scheduled_members(universe):
        fresh = endpoint_record(output, universe, costs, inventory, member)
        retained = read_record(endpoint_path(output, inventory, member), SCHEMA_ENDPOINT)
        if json.loads(json_text(fresh["stages"])) != retained["stages"]:
            raise ValueError(f"endpoint record {inventory}/{member} differs from its evidence")
    for inventory, member, seed in scheduled_cells(universe):
        record = read_record(model_path(output, inventory, member, seed), SCHEMA_MODEL)
        for stage in record["stages"].values():
            for block in stage.values():
                for row in block["rows"]:
                    if not row["excluded"] and row["tie_free"] != tie_free(row["pig_presence"],
                                                                           row["pig_displacement"]):
                        raise ValueError("retained tie-free cost differs from its components")
    fresh = compute_tables(output, plan)
    if read_json(output / "compute.json") != json.loads(json_text(fresh)):
        raise ValueError("compute.json differs from the fresh recomputation")
    for name, text in rendered(plan, fresh).items():
        if (output / name).read_bytes() != text.encode():
            raise ValueError(f"published {name} differs from the recomputation")
    log(f"validation passed: endpoint costs re-derived from {fresh['structure']['shots']} evidence "
        f"tensors; {fresh['structure']['model_records']} model records; compute.json, summary.json, "
        f"comparisons.csv, findings.md byte-compared ({time.monotonic() - began:.1f}s)")
    return 0


def dry_run(output):
    sources = t96.load_sources()
    members, universe, excluded = t96.build_universe(sources)
    cohorts = cohort_members(members)
    log(f"dry run (no write, no statistic); output root {Path(output)}")
    for inventory in INVENTORIES:
        entries = universe[inventory]
        shots = sum(len(e["verdicts"]) for e in entries.values())
        mixed = sorted(m for m, e in entries.items() if t96.mixed(e))
        held = [m for m in mixed if m in cohorts["held_out"]]
        log(f"  {inventory}: {len(entries)} members, {shots} verdict-bearing shots, {len(mixed)} mixed "
            f"members ({len(held)} held-out: {held})")
    log(f"  stages {list(STAGES)}; requests {len(REQUESTS)}; costs {list(COSTS)}; seeds {list(SEEDS)}")
    log(f"  shots {len(candidates(universe))}; endpoint records {len(scheduled_members(universe))}; "
        f"model records {len(scheduled_cells(universe))}")
    for row in excluded:
        log(f"  excluded {row['inventory']}/{row['member']}: {row['rule']}")
    log(f"  plan present: {(Path(output) / 'plan.json').is_file()}")
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
