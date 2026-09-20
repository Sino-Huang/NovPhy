"""Issue-77 N1 dynamics diagnostic: fixed-pair curves + ranked-action regret.

Small-project mode (plan §0): no authorization gates, no audit cycles. Kept
discipline, declared here before any model outcome was seen (2026-09-20):

- Evaluation states: the N1 campaign's held_out_evaluation lineages with at
  least one coverage-admissible branch, sorted by identity; never fitted.
- Candidates: the lineage's 13 fixed actions, restricted to coverage-admissible
  branches. Dropped candidates are reported; regret is computed over the
  admissible subset and is never silently treated as worst-case.
- Context: the observed agent frame at the shot launch (shot-1 frame 0) of the
  lowest-ordinal admissible branch, parsed once with the frozen issue-70
  adapter, exactly as training window first carriers are built.
- Endpoints (observed-frame units; one frame = 50 native steps): recursive and
  local errors at t in {15, 30, 60, 150, 225, 600}; the task endpoint is 600
  (the full observed window, 30000 native steps). Most N1 branches are
  right-censored at the window limit, so the t=600 cost is an END-OF-WINDOW
  cost, not a settled cost; this is stated wherever the number appears.
- Systems: continuous_h{1,5,15} and hybrid_{continuous,micro,macro}_h{1,5,15}
  from the frozen issue-77 N1 dynamics checkpoints; ordinal09 prior (candidate
  index 8) and uniform-random regret retained as no-model baselines.
- Action encoding at evaluation equals training: drag/480, true 1000 ms hold,
  tap/1000, bias 1; the frozen #62 bounds check is not applied to N1 actions.
- Uncertainty: paired bootstrap over states (10000 draws, #72 tooling seed
  7201), seed differences averaged before resampling; descriptive only -
  with at most four independent held-out states no confirmatory claim is made.
- Claims are limited to normal mechanics (plan §4.5). Zero-shot and few-shot
  novelty claims belong to stage N2 and are not made here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import io
import math
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import torch

from scripts import run_issue_71_hybrid_readiness as old
from scripts import run_issue_72_matched_grid as grid
from scripts import run_issue_74_matched_dynamics as base
from scripts import run_issue_77_n1_train as train
from scripts.canonical_native_trace import NativeTrace
from world_model.data.deployment_temporal import AgentObservation, TemporalObservationContext
from world_model.model import Abstraction, PredictionPair

ROOT = base.ROOT
CAMPAIGN = train.CAMPAIGN
DYNAMICS = train.OUTPUT
OUTPUT = ROOT / ".local-artifacts/issue-77-n1-diagnostic-v1"
SCHEMA = "issue_77_n1_diagnostic_v1"
IDENTITY = "issue-77-n1-diagnostic-v1"
TIMES = (15, 30, 60, 150, 225, 600)
TASK_TIME = 600
HORIZONS = (1, 5, 15)
SEEDS = train.SEEDS
SYSTEMS = {f"continuous_h{h}": ("continuous", PredictionPair(h, Abstraction.CONTINUOUS)) for h in HORIZONS}
SYSTEMS.update({f"hybrid_{str(mode)}_h{h}": ("hybrid", PredictionPair(h, mode))
                for h in HORIZONS for mode in Abstraction})
PRIOR_ORDINAL = 8
FIELDS = {"presence": ([0], 1), "kind": ([2, 11], 4), "position": ([5, 6], 7), "motion": ([8, 9], 10)}
FILES = tuple(dict.fromkeys((*base.FILES, "scripts/run_issue_77_n1_train.py",
                             "scripts/run_issue_77_n1_diagnostic.py")))
read, write = base.read, base.write


def log(message):
    print(f"[issue-77-n1-diag] {message}", flush=True)


def dynamics_args(args):
    return argparse.Namespace(output=args.dynamics, issue71=train.old.ROOT,
                              parser=train.old.repair.ROOT, device=args.device)


def definitions():
    return {"times": list(TIMES), "task_time": TASK_TIME, "horizons": list(HORIZONS),
            "seeds": list(SEEDS), "endpoint": TASK_TIME,
            "endpoint_semantics": "observed-frame units (1 frame = 50 native steps); t=600 is end-of-window, "
                                  "right-censored branches included, NOT a settled cost",
            "systems": {name: {"arm": arm, "horizon": p.delta, "mode": str(p.abstraction)}
                        for name, (arm, p) in SYSTEMS.items()},
            "membership": "held_out_evaluation lineages with >=1 coverage-admissible branch, sorted by identity",
            "candidates": "coverage-admissible fixed actions only; dropped candidates reported, never worst-cased",
            "terminal_absorption": "offsets beyond an early-terminal branch's last frame reuse the terminal "
                                   "frame's carrier (stable-terminal absorption); absorbed offsets recorded",
            "aggregate_error": "unmasked mean squared error of all 236 carrier values, as #74",
            "recursive": "one shared initial carrier; predictions feed next transition; no truth resets",
            "local": "observed context at t-h -> t, evaluation only; shared initial carrier when t-h=0",
            "action_tensor": "drag/480, true 1000 ms hold, tap/1000, 1; #62 bounds check not applied",
            "field_masks": "target availability only; presence[0]/mask1, kind[2,11]/mask4, position[5,6]/mask7, motion[8,9]/mask10",
            "aggregate_error": "unmasked mean squared error of all 236 carrier values, as #74",
            "task": "predicted count cost at t=600 versus actual end-of-window count cost; not a local MSE",
            "prior": "candidate index 8 (ordinal09), no-model baseline; uniform-random expected regret retained",
            "typed_failure_absorption": "failed branches excluded from candidates; reported in coverage summary",
            "bootstrap": {"draws": 10000, "seed": 7201, "unit": "state",
                          "note": "seed differences averaged before resampling; descriptive development only"},
            "claims": "normal-mechanics breadth only; no zero-shot/few-shot novelty claim (stage N2)"}


def candidate_source(branch, plan_branches):
    declared = plan_branches[branch["identity"]]
    result = read(CAMPAIGN / "results" / f"{branch['identity']}.json")
    segment = result["segments"][0]
    executed = segment["action"]
    if (result["member_identity"] != branch["identity"] or result["complete"] is not True
            or list(executed["drag_release"]) != [declared["action"]["drag_x"], declared["action"]["drag_y"]]
            or executed["release_time"] != declared["action"]["release_time_ms"]
            or executed["tap_time"] != declared["action"]["tap_time_ms"]):
        raise ValueError("diagnostic candidate source differs from frozen campaign membership")
    summary = segment["summary"]
    if summary["first_fixed_step"] != 30000 or summary["frame_count"] < 2:
        raise ValueError("N1 candidate observation window differs from the declared contract")
    terminal = summary.get("terminal_evidence")
    return {"identity": branch["identity"], "status": "accepted",
            "ordinal": branch["candidate_ordinal"],
            "action": declared["action"],
            "attempt": f"attempts/{branch['identity']}",
            "capture_id": segment["capture_id"],
            "native_root": segment["native_root"],
            "observation_manifest": segment["observation_manifest"],
            "censored": summary["censored"],
            "terminal_reason": None if terminal is None else terminal["reason"],
            "first_fixed_step": 30000, "last_offset": summary["frame_count"] - 1,
            "frame_count": summary["frame_count"]}


def select_samples(coverage, campaign_plan):
    plan_branches = {b["identity"]: b for b in campaign_plan["branches"]}
    samples = []
    for member in sorted(campaign_plan["members"], key=lambda m: m["identity"]):
        if member["study_role"] != "held_out_evaluation":
            continue
        admissible = [b for b in sorted(campaign_plan["branches"], key=lambda b: b["candidate_ordinal"])
                      if b["source_member_identity"] == member["identity"]
                      and coverage["branches"][b["identity"]]["status"] == "admissible"]
        dropped = [b["identity"] for b in campaign_plan["branches"]
                   if b["source_member_identity"] == member["identity"]
                   and coverage["branches"][b["identity"]]["status"] != "admissible"]
        if not admissible:
            samples.append({"state": member, "sources": [], "dropped_candidates": dropped,
                            "unsupported": "zero coverage-admissible branches"})
            continue
        samples.append({"state": member, "dropped_candidates": dropped,
                        "sources": [candidate_source(b, plan_branches) for b in admissible]})
    if not any(s["sources"] for s in samples):
        raise ValueError("no evaluable held-out N1 states in coverage")
    return samples


def make_plan(args):
    dynamics = train.load_plan(dynamics_args(args))
    coverage = read(CAMPAIGN / "coverage.json")
    if coverage["schema"] != "issue_77_n1_coverage_v1" or coverage["coverage_complete"] is not True:
        raise ValueError("requires completed issue-77 N1 campaign coverage")
    campaign_plan = read(CAMPAIGN / "plan.json")
    samples = select_samples(coverage, campaign_plan)
    return {"schema": SCHEMA, "identity": IDENTITY, "definitions": definitions(),
            "dynamics": str(args.dynamics.resolve()), "dynamics_plan_identity": dynamics["identity"],
            "contract": dynamics["contract"],
            "n1_campaign": {"root": str(CAMPAIGN), "identity": campaign_plan["identity"],
                            "coverage_status_counts": coverage["status_counts"]},
            "samples": samples,
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_text": {p: (ROOT / p).read_text() for p in FILES},
            "archived_release": False, "fresh_evaluation_opened": False,
            "final_evaluation_opened": False, "issue_64_authorized": False}


def load_plan(args):
    plan = read(args.output / "plan.json")
    current = make_plan(args)
    current["source_revision"] = plan["source_revision"]
    if plan != current:
        raise ValueError("frozen issue-77 N1 diagnostic source/membership changed; preserve and explicitly version")
    return plan


# ---------------------------------------------------------------- targets

def needed_offsets():
    return sorted({0, *TIMES, *(t - h for t in TIMES for h in HORIZONS)})


def target_path(args, state):
    return args.output / "targets" / f"state-{state['ordinal']:03d}.json"


def curve_path(args, seed, state, system, ordinal):
    return args.output / "fixed" / f"seed-{seed}" / f"state-{state['ordinal']:03d}" / system / \
        f"candidate-{ordinal:02d}.json"


def branch_events(source):
    relevant = ("bird_launched", "collision", "entity_death", "entity_destroyed",
                "stable_entered", "stable_exited", "level_clear")
    events = {k: [] for k in relevant}
    pig_removed = pig_contact = block_contact = False
    for chunk in NativeTrace(source["native_root"]).chunks():
        for event in chunk["events"]:
            name = event["event_type"]
            if name in events:
                events[name].append(event["fixed_step"] - source["first_fixed_step"])
            participants = " ".join(event.get("participants", []))
            if name in ("entity_death", "entity_destroyed") and "pig" in participants:
                pig_removed = True
            if name == "collision" and "pig" in participants:
                pig_contact = True
            if name == "collision" and "block" in participants:
                block_contact = True
    return {"events": events, "interaction_coverage": {
        "pig_removed": pig_removed, "pig_contact": pig_contact, "block_contact": block_contact}}
def target_candidate(args, source, adapter, vocabulary):
    root = CAMPAIGN / source["attempt"] / "shot-1"
    obsroot = root / "observation-trace"
    manifest = read(obsroot / "observation_trace_manifest.json")
    if manifest["identity"] != source["observation_manifest"]:
        raise ValueError("target observation source differs")
    frames = manifest["frame_records"]
    if (frames[0]["fixed_step"] != 30000
            or any(f["fixed_step"] != 30000 + 50 * i for i, f in enumerate(frames[:-1]))
            or not (30000 + 50 * (len(frames) - 2) < frames[-1]["fixed_step"]
                    <= 30000 + 50 * (len(frames) - 1))):
        raise ValueError("target observation cadence differs")
    last_index = len(frames) - 1
    absorbed = sorted(offset for offset in needed_offsets() if offset > last_index)
    positions = sorted({max(0, min(offset, last_index) - 1) for offset in needed_offsets()}
                       | {min(offset, last_index) for offset in needed_offsets()})
    refs = {i: frames[i] for i in positions}
    observations, parsed = {}, {}
    parse_calls = parse_images = 0
    began = time.monotonic()

    def observation(i):
        if i not in observations:
            ref = refs[i]["agent_observation"]
            observations[i] = AgentObservation(ref["identity"], refs[i]["fixed_step"],
                                               refs[i]["fixed_time_seconds"],
                                               (obsroot / ref["relative_path"]).read_bytes(), "agent")
        return observations[i]

    def at(p):
        nonlocal parse_calls, parse_images
        chosen = sorted({max(0, p - 1), p})
        if any(i not in parsed for i in chosen):
            values = adapter.parse_batch(tuple(observation(i) for i in chosen))
            parsed.update(zip(chosen, values, strict=True))
            parse_calls += 1
            parse_images += len(chosen)
        context = TemporalObservationContext(None if p == 0 else observation(p - 1), observation(p))
        return adapter.build_from_parsed(context, parsed[p], None if p == 0 else parsed[max(0, p - 1)]).tensor

    carriers = {str(offset): at(min(offset, last_index)).tolist() for offset in needed_offsets()}
    grid.synchronize(args.device)
    return {"source": source, "status": "available",
            "carriers": carriers,
            "absorbed_offsets": absorbed,
            "timing": {**branch_events(source), "terminal_reason": source["terminal_reason"],
                       "censored": source["censored"], "last_offset": source["last_offset"]},
            "target_parser": {"calls": parse_calls, "image_examples": parse_images,
                              "wall_seconds": time.monotonic() - began,
                              "scope": "evaluation target construction only, not deployment latency"}}


def check_targets(value, sample, plan):
    if (value["plan_identity"] != plan["identity"] or value["state"] != sample["state"]["identity"]
            or len(value["context"]) != 236):
        raise ValueError("target state/parser binding differs")
    if value["dropped_candidates"] != sample["dropped_candidates"]:
        raise ValueError("target dropped-candidate accounting differs")
    for target, source in zip(value["candidates"], sample["sources"], strict=True):
        if target["source"]["identity"] != source["identity"] or target["status"] != "available":
            raise ValueError("target candidate binding differs")
        for offset in needed_offsets():
            vector = target["carriers"].get(str(offset))
            if vector is None or len(vector) != 236 or not all(math.isfinite(x) for x in vector):
                raise ValueError("target availability/representation differs")


def prepare_targets(args, plan):
    adapter = old.repair.load_repaired_adapter(old.repair.ROOT, args.device)
    vocabulary = plan["contract"]["vocabulary"]
    began = time.monotonic()
    evaluable = [s for s in plan["samples"] if s["sources"]]
    for n, sample in enumerate(evaluable, 1):
        state = sample["state"]
        path = target_path(args, state)
        if path.exists():
            check_targets(read(path), sample, plan)
            log(f"targets state={n}/{len(evaluable)} cached")
            continue
        reference = sample["sources"][0]
        obsroot = CAMPAIGN / reference["attempt"] / "shot-1" / "observation-trace"
        manifest = read(obsroot / "observation_trace_manifest.json")
        frame0 = manifest["frame_records"][0]
        started = time.monotonic()
        observation = AgentObservation(frame0["agent_observation"]["identity"], frame0["fixed_step"],
                                       frame0["fixed_time_seconds"],
                                       (obsroot / frame0["agent_observation"]["relative_path"]).read_bytes(),
                                       "agent")
        parsed = adapter.parse_batch((observation,))[0]
        context = adapter.build_from_parsed(TemporalObservationContext(None, observation), parsed, None).tensor
        perception = {"wall_seconds": time.monotonic() - started, "reference_branch": reference["identity"],
                      "scope": "one B=1 decision-frame perception per state"}
        targets = []
        for source in sample["sources"]:
            log(f"targets state={n}/{len(evaluable)} candidate={source['ordinal']} start")
            targets.append(target_candidate(args, source, adapter, vocabulary))
        result = {"plan_identity": plan["identity"], "state": state["identity"],
                  "context": context.tolist(), "perception": perception,
                  "dropped_candidates": sample["dropped_candidates"], "candidates": targets}
        check_targets(result, sample, plan)
        write(path, result)
        elapsed = time.monotonic() - began
        log(f"targets state={n}/{len(evaluable)} complete elapsed={elapsed:.1f}s "
            f"eta={elapsed/n*(len(evaluable)-n):.1f}s")


# ---------------------------------------------------------------- curves

def n1_action_tensor(action, device):
    return torch.tensor([[action["drag_x"] / 480., action["drag_y"] / 480.,
                          action["release_time_ms"] / 1000., action["tap_time_ms"] / 1000., 1.]],
                        device=device)


@torch.no_grad()
def fixed_curve(model, context, action, pair):
    """Deployment boundary: current carrier/action/pair only, never observed targets."""
    device = next(model.parameters()).device
    if context.shape != (236,) or pair not in base.pairs_for(model):
        raise ValueError("incompatible fixed carrier/horizon/mode")
    z = context[None].to(device)
    a = n1_action_tensor(action, device)
    output, elapsed, segments, transition_wall = {}, 0, [], 0.
    mac = base.work(model, pair)
    for target in TIMES:
        grid.synchronize(device)
        began = time.monotonic()
        while elapsed < target:
            z = model.carrier(z, a, pair)
            elapsed += pair.delta
            if not bool(torch.isfinite(z).all()):
                return {"outputs": output, "failure": "nonfinite_recursive_carrier",
                        "completed_steps": elapsed, "segments": segments,
                        "linear_macs": mac * (elapsed // pair.delta),
                        "transition_wall_seconds": transition_wall + time.monotonic() - began}
        grid.synchronize(device)
        duration = time.monotonic() - began
        transition_wall += duration
        output[str(target)] = z[0].cpu().tolist()
        segments.append({"endpoint": target,
                         "transition_calls": (target - (segments[-1]["endpoint"] if segments else 0)) // pair.delta,
                         "wall_seconds": duration})
    return {"outputs": output, "failure": None, "completed_steps": elapsed, "segments": segments,
            "linear_macs": mac * (elapsed // pair.delta), "transition_wall_seconds": transition_wall}


@torch.no_grad()
def local_predictions(model, target, context, action, pair):
    """Observed-context diagnostics only; never used to choose an action."""
    device = next(model.parameters()).device
    a = n1_action_tensor(action, device)
    outputs, calls, unavailable = {}, 0, {}
    grid.synchronize(device)
    began = time.monotonic()
    for t in TIMES:
        value = context.tolist() if t == pair.delta else target["carriers"].get(str(t - pair.delta))
        if value is None or target["carriers"].get(str(t)) is None:
            outputs[str(t)] = None
            unavailable[str(t)] = "unavailable_observed_context_or_target"
            continue
        predicted = model.carrier(torch.tensor([value], device=device), a, pair)
        calls += 1
        outputs[str(t)] = predicted[0].cpu().tolist() if bool(torch.isfinite(predicted).all()) else None
        if outputs[str(t)] is None:
            unavailable[str(t)] = "nonfinite_local_prediction"
    grid.synchronize(device)
    return {"outputs": outputs, "unavailable": unavailable, "calls": calls,
            "linear_macs": calls * base.work(model, pair),
            "wall_seconds": time.monotonic() - began,
            "scope": "evaluation local-error probe, excluded from deployment work"}


def fixed_record(args, plan, sample, target_state, seed, system, source, model):
    arm, pair = SYSTEMS[system]
    context = torch.tensor(target_state["context"])
    prediction = fixed_curve(model, context, source["action"], pair)
    target = next(t for t in target_state["candidates"] if t["source"]["ordinal"] == source["ordinal"])
    local = local_predictions(model, target, context, source["action"], pair)
    objective = base.TaskObjective.from_vocabulary(plan["contract"]["vocabulary"])
    cost = None if prediction["failure"] else objective(torch.tensor(prediction["outputs"][str(TASK_TIME)]))
    return {"plan_identity": plan["identity"], "seed": seed, "system": system,
            "state": sample["state"]["identity"], "candidate_identity": source["identity"],
            "ordinal": source["ordinal"], "action": source["action"],
            "prediction": prediction, "local": local, "cost": cost,
            "work_scope": "instrumented fixed-rollout wall with prefix synchronization; "
                          "target parsing/local probes separate; MACs not full FLOPs"}


def check_curve(record, plan, sample, seed, system, source, model):
    _, pair = SYSTEMS[system]
    if any(record[k] != v for k, v in {"plan_identity": plan["identity"], "seed": seed, "system": system,
                                       "state": sample["state"]["identity"],
                                       "candidate_identity": source["identity"],
                                       "ordinal": source["ordinal"], "action": source["action"]}.items()):
        raise ValueError("fixed diagnostic checkpoint/state/action binding differs")
    pred = record["prediction"]
    elapsed = pred["completed_steps"]
    if pred["linear_macs"] != base.work(model, pair) * (elapsed // pair.delta) or elapsed % pair.delta:
        raise ValueError("fixed execution work differs")
    if not pred["failure"]:
        if elapsed != TASK_TIME or set(pred["outputs"]) != {str(t) for t in TIMES}:
            raise ValueError("missing common physical endpoint")
        for values in pred["outputs"].values():
            if len(values) != 236 or not all(math.isfinite(v) for v in values):
                raise ValueError("invalid predicted carrier")
        prior = 0
        for segment, t in zip(pred["segments"], TIMES, strict=True):
            if segment["endpoint"] != t or segment["transition_calls"] != (t - prior) // pair.delta:
                raise ValueError("fixed prefix execution timeline differs")
            prior = t
        objective = base.TaskObjective.from_vocabulary(plan["contract"]["vocabulary"])
        if objective(torch.tensor(pred["outputs"][str(TASK_TIME)])) != record["cost"]:
            raise ValueError("predicted task cost differs from endpoint carrier")
    elif record["cost"] is not None:
        raise ValueError("failed rollout has a task cost")
    if record["local"]["linear_macs"] != record["local"]["calls"] * base.work(model, pair):
        raise ValueError("local diagnostic work differs")
    if set(record["local"]["outputs"]) != {str(t) for t in TIMES}:
        raise ValueError("local endpoints differ")


def stored_curve(args, plan, sample, targets, seed, system, source, model):
    path = curve_path(args, seed, sample["state"], system, source["ordinal"])
    if path.exists():
        record = read(path)
    else:
        record = fixed_record(args, plan, sample, targets, seed, system, source, model)
        check_curve(record, plan, sample, seed, system, source, model)
        write(path, record)
    check_curve(record, plan, sample, seed, system, source, model)
    return record


def run_diagnostic(args, plan):
    inventory = artifact_inventory(args, plan)
    if (inventory["target_states"] == inventory["expected_target_states"]
            and inventory["fixed_candidate_records"] == inventory["expected_fixed_candidate_records"]):
        log("complete diagnostic inventory already exists; no new inference; use --publish/--validate")
        return
    prepare_targets(args, plan)
    dynamics = train.load_plan(dynamics_args(args))
    evaluable = [s for s in plan["samples"] if s["sources"]]
    total = sum(len(s["sources"]) for s in evaluable) * len(SEEDS) * len(SYSTEMS)
    began = time.monotonic()
    completed = 0
    for seed in SEEDS:
        models = {a: train.load_predictor(dynamics_args(args), dynamics, seed, a)[0].requires_grad_(False)
                  for a in base.ARMS}
        for n, sample in enumerate(evaluable, 1):
            targets = read(target_path(args, sample["state"]))
            check_targets(targets, sample, plan)
            for system, (arm, pair) in SYSTEMS.items():
                for source in sample["sources"]:
                    record = stored_curve(args, plan, sample, targets, seed, system, source, models[arm])
                    completed += 1
                    elapsed = time.monotonic() - began
                    log(f"diagnostic={completed}/{total} seed={seed} state={n}/{len(evaluable)} "
                        f"system={system} candidate={source['ordinal']} "
                        f"failure={record['prediction']['failure']} elapsed={elapsed:.1f}s "
                        f"eta={elapsed/completed*(total-completed):.1f}s")
    log("fixed-pair diagnostics complete")


# ---------------------------------------------------------------- metrics


def field_errors(prediction, target, objective):
    if prediction is None or target is None:
        return None
    p, t = torch.tensor(prediction), torch.tensor(target)
    if p.shape != (236,) or t.shape != (236,):
        raise ValueError("field diagnostic requires 236-value carriers")
    slots_p, slots_t = p[2:].reshape(18, 13), t[2:].reshape(18, 13)
    result = {"carrier_mse": float((p - t).square().mean()),
              "history_available_mse": float((p[0] - t[0]).square()),
              "prior_elapsed_seconds_mse": float((p[1] - t[1]).square()),
              "availability_bits_mse": float((slots_p[:, [1, 4, 7, 10, 12]]
                                              - slots_t[:, [1, 4, 7, 10, 12]]).square().mean())}
    for field, (columns, mask_column) in FIELDS.items():
        mask = slots_t[:, mask_column] > .5
        squared = (slots_p[:, columns] - slots_t[:, columns]).square()
        result[field + "_available_values"] = int(mask.sum()) * len(columns)
        result[field + "_mse"] = float(squared[mask].mean()) if bool(mask.any()) else None
    pc, tc = objective.counts(p), objective.counts(t)
    result["parsed_pig_count_absolute_error"] = abs(pc[0] - tc[0])
    result["parsed_block_count_absolute_error"] = abs(pc[1] - tc[1])
    if any(isinstance(v, float) and not math.isfinite(v) for v in result.values()):
        raise ValueError("nonfinite error statistic; do not serialize as null or favorable missing data")
    return result

def average_errors(values, *, unit="candidates"):
    present = [v for v in values if v is not None and "carrier_mse" in v]
    if not present:
        return {"available_" + unit: 0}
    result = {"available_" + unit: len(present)}
    for key in present[0]:
        items = [v[key] for v in present if v.get(key) is not None]
        output_key = "mean_" + key if key == "available_candidates" else key
        result[output_key] = float(np.mean(items)) if items else None
    return result


def outcome_flags(timing):
    coverage = timing["interaction_coverage"]
    return {"pig_removed": coverage["pig_removed"], "level_clear": timing["terminal_reason"] == "level_clear",
            "pig_contact": coverage["pig_contact"], "block_contact": coverage["block_contact"],
            "stable_transition_observed": bool(timing["events"]["stable_entered"]
                                               or timing["events"]["stable_exited"]),
            "censored": timing["censored"]}


def task_metrics(costs, realized):
    """costs: predicted per candidate; realized: actual end-of-window count costs."""
    candidates = [{"accepted": True, "realized_count_cost": r} for r in realized]
    metrics = grid.ranked(costs, candidates)
    selected = metrics["selected"]
    best = min(realized)
    selected_cost = realized[selected] if selected is not None else 1e9
    return {**metrics, "selected_ordinal": None if selected is None else selected,
                "realized_count_cost": selected_cost,
                "excess_count_cost_over_best": selected_cost - best}


def aggregate_states(rows):
    return {"states": len(rows),
            "mean_regret": float(np.mean([r["task"]["regret"] for r in rows])),
            "top1_fraction": float(np.mean([r["task"]["top1"] for r in rows])),
            "top3_fraction": float(np.mean([r["task"]["top3"] for r in rows])),
            "prediction_failure_states": sum(r["task"]["prediction_failure"] for r in rows),
            "local_prediction_failures": sum(r.get("local_prediction_failures", 0) for r in rows),
            "predicted_all_tied_states": sum(r["task"]["all_tied"] for r in rows),
            "mean_realized_count_cost": float(np.mean([r["task"]["realized_count_cost"] for r in rows])),
            "transition_linear_macs": sum(r["work"]["linear_macs"] for r in rows),
            "executed_calls": {k: sum(r["work"].get(k, 0) for r in rows) for k in
                               ("controller_calls", "transition_calls", "symbol_decoder_calls",
                                "symbol_adapter_calls", "local_transition_calls")},
            "local_probe_linear_macs": sum(r["work"].get("local_linear_macs", 0) for r in rows),
            "local_probe_wall_seconds": sum(r["work"].get("local_wall_seconds", 0.) for r in rows),
            "mean_initial_perception_seconds": float(np.mean([r["work"].get("perception_seconds", 0.)
                                                              for r in rows])),
            "mean_instrumented_model_seconds_per_state": float(np.mean([r["work"]["wall_seconds"]
                                                                        for r in rows])),
            "curves": {str(t): average_errors([r["recursive"][str(t)] for r in rows if str(t) in r["recursive"]],
                                              unit="states") for t in TIMES},
            "local_curves": {str(t): average_errors([r["local"][str(t)] for r in rows if str(t) in r["local"]],
                                                    unit="states") for t in TIMES}}


def artifact_inventory(args, plan):
    evaluable = [s for s in plan["samples"] if s["sources"]]
    expected = sum(len(s["sources"]) for s in evaluable) * len(SEEDS) * len(SYSTEMS)
    targets = sum(target_path(args, s["state"]).exists() for s in evaluable)
    curves = sum(curve_path(args, seed, s["state"], name, source["ordinal"]).exists()
                 for seed in SEEDS for s in evaluable for name in SYSTEMS for source in s["sources"])
    return {"evaluable_states": len(evaluable), "unsupported_states": len(plan["samples"]) - len(evaluable),
            "target_states": targets, "expected_target_states": len(evaluable),
            "fixed_candidate_records": curves, "expected_fixed_candidate_records": expected}


def summarize_contrasts(per_state):
    pairs = []
    for h in HORIZONS:
        pairs.append((f"hybrid_continuous_h{h}", f"continuous_h{h}", "training_effect"))
        for mode in ("micro", "macro"):
            pairs.append((f"hybrid_{mode}_h{h}", f"hybrid_continuous_h{h}", "symbolic_execution"))
    result = []
    for tested, reference, kind in pairs:
        ids = sorted(per_state[str(SEEDS[0])][tested])
        differences = [float(np.mean([per_state[str(seed)][reference][s]["task"]["regret"]
                                      - per_state[str(seed)][tested][s]["task"]["regret"]
                                      for seed in SEEDS])) for s in ids]
        curves = {}
        for t in TIMES:
            paired = []
            for state in ids:
                across_seeds = []
                for seed in SEEDS:
                    a = per_state[str(seed)][tested][state]["recursive"].get(str(t), {}).get("carrier_mse")
                    b = per_state[str(seed)][reference][state]["recursive"].get(str(t), {}).get("carrier_mse")
                    if a is not None and b is not None:
                        across_seeds.append(b - a)
                if len(across_seeds) == len(SEEDS):
                    paired.append(float(np.mean(across_seeds)))
            curves[str(t)] = {"paired_states": len(paired),
                              **(grid.paired_interval(paired) if paired else {})}
        result.append({"tested": tested, "reference": reference, "kind": kind,
                       "positive_is_improvement": True,
                       "regret": grid.paired_interval(differences), "carrier_mse": curves,
                       "scope": "descriptive held-out development, not multiplicity-adjusted confirmation"})
    return result


def publication(args, plan):
    inv = artifact_inventory(args, plan)
    complete = (inv["target_states"] == inv["expected_target_states"]
                and inv["fixed_candidate_records"] == inv["expected_fixed_candidate_records"])
    common = {"schema": "issue_77_n1_diagnostic_report_v1", "identity": "issue-77-n1-diagnostic-report-v1",
              "plan_identity": plan["identity"], "inventory": inv, "diagnostics_complete": complete,
              "definitions": plan["definitions"],
              "claim_boundary": "normal-mechanics breadth only; zero-shot/few-shot novelty evaluation is stage N2",
              "limitations": ["shared semantically supervised CNN, not end-to-end symbol-free comparison",
                              "t=600 is end-of-window cost on mostly right-censored branches, not settled cost",
                              "at most four independent held-out states; bootstrap intervals are descriptive",
                              "typed branch failures excluded from candidates and reported, never worst-cased",
                              "linear MACs not full FLOPs or matched total work"],
              "archived_release": False, "fresh_evaluation_opened": False,
              "final_evaluation_opened": False, "issue_64_authorized": False}
    if not complete:
        return common
    dynamics = train.load_plan(dynamics_args(args))
    objective = base.TaskObjective.from_vocabulary(plan["contract"]["vocabulary"])
    evaluable = [s for s in plan["samples"] if s["sources"]]
    all_states, summary, headroom = {}, {}, []
    target_cost = 0.
    for sample in evaluable:
        target_state = read(target_path(args, sample["state"]))
        check_targets(target_state, sample, plan)
        target_cost += sum(t["target_parser"]["wall_seconds"] for t in target_state["candidates"])
    for seed in SEEDS:
        models = {a: train.load_predictor(dynamics_args(args), dynamics, seed, a)[0] for a in base.ARMS}
        seed_rows = {name: {} for name in SYSTEMS}
        for n, sample in enumerate(evaluable, 1):
            state = sample["state"]
            targets = read(target_path(args, state))
            realized = [objective(torch.tensor(t["carriers"][str(TASK_TIME)])) for t in targets["candidates"]]
            flags = [outcome_flags(t["timing"]) for t in targets["candidates"]]
            if seed == SEEDS[0]:
                candidates = [{"accepted": True, "realized_count_cost": r} for r in realized]
                regrets, informative = grid.normalized_regrets(candidates)
                prior = next((i for i, t in enumerate(targets["candidates"])
                              if t["source"]["ordinal"] == PRIOR_ORDINAL), None)
                headroom.append({"state": state["identity"], "family": state["generator_family"],
                                 "candidates": len(realized), "informative": informative,
                                 "prior_ordinal09_regret": None if prior is None else regrets[prior],
                                 "uniform_expected_regret": float(np.mean(regrets)),
                                 "dropped_candidates": len(sample["dropped_candidates"]),
                                 "opportunities": {k: any(f[k] for f in flags) for k in
                                                   ("pig_removed", "pig_contact", "block_contact")}})
            for system, (arm, pair) in SYSTEMS.items():
                records = [read(curve_path(args, seed, state, system, s["ordinal"]))
                           for s in sample["sources"]]
                for record, source in zip(records, sample["sources"], strict=True):
                    check_curve(record, plan, sample, seed, system, source, models[arm])
                recursive, local = {}, {}
                for t in TIMES:
                    recursive[str(t)] = average_errors(
                        [field_errors(r["prediction"]["outputs"].get(str(t)), target["carriers"].get(str(t)),
                                      objective)
                         for r, target in zip(records, targets["candidates"], strict=True)])
                    local[str(t)] = average_errors(
                        [field_errors(r["local"]["outputs"].get(str(t)), target["carriers"].get(str(t)),
                                      objective)
                         for r, target in zip(records, targets["candidates"], strict=True)])
                seed_rows[system][state["identity"]] = {
                    "family": state["generator_family"],
                    "task": task_metrics([r["cost"] for r in records], realized),
                    "local_prediction_failures": sum(v == "nonfinite_local_prediction"
                                                     for r in records for v in r["local"]["unavailable"].values()),
                    "recursive": recursive, "local": local,
                    "work": {"linear_macs": sum(r["prediction"]["linear_macs"] for r in records),
                             "wall_seconds": sum(r["prediction"]["transition_wall_seconds"] for r in records),
                             "local_linear_macs": sum(r["local"]["linear_macs"] for r in records),
                             "local_transition_calls": sum(r["local"]["calls"] for r in records),
                             "local_wall_seconds": sum(r["local"]["wall_seconds"] for r in records),
                             "controller_calls": 0,
                             "transition_calls": sum(r["prediction"]["completed_steps"] // pair.delta
                                                     for r in records),
                             "symbol_decoder_calls": sum(r["prediction"]["completed_steps"] // pair.delta
                                                         for r in records)
                             if pair.abstraction != Abstraction.CONTINUOUS else 0,
                             "symbol_adapter_calls": sum(r["prediction"]["completed_steps"] // pair.delta
                                                         for r in records)
                             if pair.abstraction != Abstraction.CONTINUOUS else 0,
                             "perception_seconds": targets["perception"]["wall_seconds"]}}
            log(f"publication seed={seed} state={n}/{len(evaluable)} validated")
        all_states[str(seed)] = seed_rows
        summary[str(seed)] = {name: aggregate_states(list(rows.values()))
                              for name, rows in seed_rows.items()}
    contrasts = summarize_contrasts(all_states)
    return {**common, "per_seed": summary, "per_state": all_states, "contrasts": contrasts,
            "headroom": headroom, "target_parsing_seconds": target_cost,
            "independent_state_count": len(evaluable),
            "source_family_counts": dict(Counter(s["state"]["generator_family"] for s in evaluable)),
            "mean_prior_regret": float(np.mean([r["prior_ordinal09_regret"] for r in headroom
                                                if r["prior_ordinal09_regret"] is not None]))
            if any(r["prior_ordinal09_regret"] is not None for r in headroom) else None,
            "mean_uniform_expected_regret": float(np.mean([r["uniform_expected_regret"]
                                                           for r in headroom]))}


def compact_report(result):
    return {k: v for k, v in result.items() if k not in ("per_state",)}


def comparisons_csv(result):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(("seed", "system", "elapsed_observed_frames", "kind", "metric", "value", "available_states"))
    for seed, systems in result.get("per_seed", {}).items():
        for name, system in systems.items():
            for kind, key in (("recursive", "curves"), ("local", "local_curves")):
                for t, metrics in system[key].items():
                    for metric, value in metrics.items():
                        if metric != "available_states":
                            writer.writerow((seed, name, t, kind, metric, value,
                                             metrics["available_states"]))
    writer.writerow(())
    writer.writerow(("seed", "system", "mean_regret", "top1_fraction", "top3_fraction",
                     "prediction_failure_states", "transition_linear_macs"))
    for seed, systems in result.get("per_seed", {}).items():
        for name, system in systems.items():
            writer.writerow((seed, name, system["mean_regret"], system["top1_fraction"],
                             system["top3_fraction"], system["prediction_failure_states"],
                             system["transition_linear_macs"]))
    writer.writerow(())
    writer.writerow(("contrast_tested", "reference", "kind", "mean_regret_difference",
                     "interval_low", "interval_high"))
    for contrast in result.get("contrasts", []):
        interval = contrast["regret"]["descriptive_95_percent_interval"]
        writer.writerow((contrast["tested"], contrast["reference"], contrast["kind"],
                         contrast["regret"]["mean"], interval[0], interval[1]))
    return stream.getvalue()


def findings_md(result):
    lines = ["# Issue-77 N1 dynamics diagnostic - findings", ""]
    lines.append(f"Diagnostics complete: {result['diagnostics_complete']}. "
                 f"Claim boundary: {result['claim_boundary']}.")
    lines.append("")
    lines.append("Declared endpoint semantics: " + result["definitions"]["endpoint_semantics"])
    lines.append("")
    if not result.get("per_seed"):
        lines.append("Diagnostics incomplete; no outcome numbers reported.")
        return "\n".join(lines) + "\n"
    lines.append("## Headroom")
    lines.append("")
    lines.append("| State | Family | Candidates | Informative | Prior(ordinal09) regret | Uniform regret | Dropped |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in result["headroom"]:
        prior = "n/a" if row["prior_ordinal09_regret"] is None else f"{row['prior_ordinal09_regret']:.4f}"
        lines.append(f"| {row['state']} | {row['family']} | {row['candidates']} | {row['informative']} | "
                     f"{prior} | {row['uniform_expected_regret']:.4f} | {row['dropped_candidates']} |")
    lines.append("")
    lines.append("## Action ranking (mean over held-out states; seeds listed individually)")
    lines.append("")
    lines.append("| System | Seed | Mean regret | Top1 | Top3 | Prediction failures |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for seed, systems in result["per_seed"].items():
        for name, system in systems.items():
            lines.append(f"| {name} | {seed} | {system['mean_regret']:.4f} | {system['top1_fraction']:.3f} | "
                         f"{system['top3_fraction']:.3f} | {system['prediction_failure_states']} |")
    lines.append("")
    lines.append("## Paired contrasts (positive favors the tested system)")
    lines.append("")
    lines.append("| Tested | Reference | Kind | Mean regret difference | Descriptive 95% interval |")
    lines.append("| --- | --- | --- | --- | --- |")
    for contrast in result["contrasts"]:
        interval = contrast["regret"]["descriptive_95_percent_interval"]
        lines.append(f"| {contrast['tested']} | {contrast['reference']} | {contrast['kind']} | "
                     f"{contrast['regret']['mean']:+.4f} | [{interval[0]:+.4f}, {interval[1]:+.4f}] |")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for item in result["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def smoke(args):
    plan = make_plan(args)
    dynamics = train.load_plan(dynamics_args(args))
    began = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="novphy-issue77-n1-diag-smoke-") as directory:
        small = copy.copy(args)
        small.output = Path(directory)
        plan = copy.deepcopy(plan)
        plan["identity"] += ":smoke"
        chosen = [s for s in plan["samples"] if s["sources"]][:1]
        if not chosen:
            raise ValueError("smoke needs at least one evaluable state")
        chosen[0]["sources"] = chosen[0]["sources"][:2]
        plan["samples"] = chosen
        prepare_targets(small, plan)
        prepare_targets(small, plan)  # target-cache resume, no repeat image parse
        seed = SEEDS[0]
        models = {a: train.load_predictor(dynamics_args(args), dynamics, seed, a)[0].requires_grad_(False)
                  for a in base.ARMS}
        sample = chosen[0]
        targets = read(target_path(small, sample["state"]))
        objective = base.TaskObjective.from_vocabulary(plan["contract"]["vocabulary"])
        records = []
        for system in SYSTEMS:
            arm, _ = SYSTEMS[system]
            for source in sample["sources"]:
                record = fixed_record(small, plan, sample, targets, seed, system, source, models[arm])
                check_curve(record, plan, sample, seed, system, source, models[arm])
                path = curve_path(small, seed, sample["state"], system, source["ordinal"])
                write(path, record)
                check_curve(read(path), plan, sample, seed, system, source, models[arm])
                target = next(t for t in targets["candidates"]
                              if t["source"]["ordinal"] == source["ordinal"])
                errors = {str(t): field_errors(record["prediction"]["outputs"].get(str(t)),
                                               target["carriers"].get(str(t)), objective) for t in TIMES}
                records.append({"state": sample["state"]["identity"], "system": system,
                                "ordinal": source["ordinal"],
                                "failure": record["prediction"]["failure"], "errors": errors})
                log(f"smoke state={sample['state']['identity']} system={system} "
                    f"candidate={source['ordinal']} complete")
        report = publication(small, plan)
        if report["diagnostics_complete"]:
            raise ValueError("smoke must not masquerade as complete diagnostics")
        write(small.output / "result.json", report)
        if read(small.output / "result.json") != report:
            raise ValueError("report JSON round trip differs")
        result = {"schema": "issue_77_n1_diagnostic_smoke_v1", "production_evidence": False,
                  "records": records, "wall_seconds": time.monotonic() - began,
                  "memory": base.memory(args.device)}
    write(args.output / "smoke.json", result)
    log(f"real smoke complete cells={len(records)} wall={result['wall_seconds']:.1f}s; "
        "no full diagnostic run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "smoke-test", "prepare", "run-diagnostic", "publish", "validate"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dynamics", type=Path, default=DYNAMICS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        if args.dry_run:
            plan = make_plan(args)
            evaluable = [s for s in plan["samples"] if s["sources"]]
            log(f"no-write dry-run: evaluable_states={len(evaluable)} "
                f"candidates={sum(len(s['sources']) for s in evaluable)} systems={len(SYSTEMS)} seeds=3")
            return 0
        if args.smoke_test:
            smoke(args)
            return 0
        if args.prepare:
            if (args.output / "plan.json").exists():
                load_plan(args)
            else:
                write(args.output / "plan.json", make_plan(args))
            log("N1 diagnostic protocol frozen; no images scored")
            return 0
        plan = load_plan(args)
        if args.run_diagnostic:
            run_diagnostic(args, plan)
        elif args.publish:
            result = publication(args, plan)
            write(args.output / "summary.json", compact_report(result))
            (args.output / "comparisons.csv").write_text(comparisons_csv(result))
            (args.output / "findings.md").write_text(findings_md(result))
            log(f"published diagnostics_complete={result['diagnostics_complete']}")
        else:
            result = publication(args, plan)
            if (compact_report(result) != read(args.output / "summary.json")
                    or comparisons_csv(result).encode("utf-8") != (args.output / "comparisons.csv").read_bytes()
                    or findings_md(result) != (args.output / "findings.md").read_text()):
                raise ValueError("published N1 diagnostic tables differ from bound source evidence")
            log("exact saved-evidence validation passed")
        return 0
    except (ValueError, OSError) as error:
        log(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
