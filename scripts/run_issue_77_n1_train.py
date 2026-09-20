"""Issue-77 N1: expand frozen issue-74 matched dynamics training with N1 shards.

Small-project mode (plan §0): no authorization gates. Both arms always train with
identical budgets, identical lineage groups and identical random minibatches per
paired seed, exactly as in the frozen issue-74 recipe; the only change is the
fitting/controller pools gaining the issue-77 N1 training-role lineages.

Declared data decisions (pinned 2026-09-20, before any model outcome was seen):
- Corpus: the frozen issue-71 3,000 lineage shards (read-only, unchanged) plus new
  issue-77 N1 shards built from `.local-artifacts/issue-77-n1-v1` capture results.
- Roles: N1 lineages with exposure_role == 'training' only. Predictor-role
  lineages join the predictor fitting pool; controller-role lineages join the
  controller supervision pool. held_out_evaluation lineages are never fitted.
- Windows: N1 observation traces hold 601 agent frames at native stride 50
  (steps 30000..60000). Windows are 60 observed frames long with the four
  uniform starts of the issue-71 builder; pair deltas (1, 5, 15) therefore act
  on observed-frame indices for N1 data (native steps 50, 250, 750), while they
  act on native steps for the frozen issue-62 data. Both arms see identical
  data, so the pairing is unaffected; the unit difference is declared here.
- Action tensor: true executed action, including the N1 1000 ms hold
  (native_history_data.action_context precedent), not the frozen #62 bounds
  constant used by issue-66 _action_tensor.
- Symbolic targets: entities outside the frozen 18-slot contract vocabulary
  (the novelty generator's world:ground_extension scenery) are skipped, the
  native_history_data.native_labels precedent; targets are derived with the
  unchanged issue-70 repair.targets builder on the filtered samples.
- Micro labels: cohort-v2 contact/support projection over the 601 observed
  samples. Macro labels: engine stability events establish steady state and
  structure change uses the immediately preceding native step's support set
  (the native_history_data.native_labels semantics for this capture stack;
  the cohort-v2 debounced re-derivation rejects 30001-step N1 windows),
  emitted in the issue-70 repair.targets predicate format.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import resource
import subprocess
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from scripts import run_issue_71_hybrid_readiness as old
from scripts import run_issue_72_matched_grid as grid
from scripts import run_issue_74_matched_dynamics as base
from scripts.cohort_v2_micro_relations import derive_capture_micro_relations
from scripts.canonical_native_trace import NativeTrace
from scripts.physics_capture_v2 import PhysicsCaptureV2
from world_model.data.deployment_temporal import AgentObservation, TemporalObservationContext
from world_model.model import Abstraction, PredictionPair
from world_model.training.cnn_hybrid import CNNHybridPredictor, PAIRS, training_loss
from world_model.training.matched_dynamics import (
    ContinuousDynamics, CONTINUOUS_PAIRS, MatchedController, capacity_contract,
    continuous_loss, controller_labels, parameter_count, pairs_for, rollout, work,
    active_capacity,
)

ROOT = base.ROOT
CAMPAIGN = ROOT / ".local-artifacts/issue-77-n1-v1"
OUTPUT = ROOT / ".local-artifacts/issue-77-n1-dynamics-v1"
SEEDS = base.SEEDS
ARMS = base.ARMS
SCHEMA = "issue_77_n1_dynamics_v1"
IDENTITY = "issue-77-n1-dynamics-v1"
FILES = tuple(dict.fromkeys((*base.FILES, "scripts/run_issue_77_n1_train.py")))
read, write = base.read, base.write


def log(message):
    print(f"[issue-77-n1] {message}", flush=True)


memory = base.memory
source_args = base.source_args

def n1_lineages():
    """Deterministic training-role pools from the frozen N1 coverage.

    Typed-failure absorption (declared 2026-09-20, before any model outcome):
    a training-role lineage joins its pool when at least one of its branches is
    coverage-admissible; shards are built from admissible branches only.
    Lineages with zero admissible branches are recorded under 'dropped' and
    reported as unsupported-for-training, never silently skipped.
    """
    coverage = read(CAMPAIGN / "coverage.json")
    if coverage["schema"] != "issue_77_n1_coverage_v1" or coverage["coverage_complete"] is not True:
        raise ValueError("requires completed issue-77 N1 campaign coverage")
    campaign_plan = read(CAMPAIGN / "plan.json")
    predictor, controller, dropped = [], [], []
    records = {}
    for member in sorted(campaign_plan["members"], key=lambda m: m["ordinal"]):
        if member["exposure_role"] != "training":
            continue
        admissible = [b["identity"] for b in campaign_plan["branches"]
                      if b["source_member_identity"] == member["identity"]
                      and coverage["branches"][b["identity"]]["status"] == "admissible"]
        record = {"identity": member["identity"], "ordinal": member["ordinal"],
                  "generator_family": member["generator_family"],
                  "exposure_role": member["exposure_role"],
                  "fit_partition": member["fit_partition"],
                  "branches": admissible, "scheduled_branches": 13,
                  "dropped_branches": 13 - len(admissible)}
        if not admissible:
            dropped.append(record)
            continue
        records[member["identity"]] = record
        (predictor if member["fit_partition"] == "predictor" else controller).append(member["identity"])
    if not predictor or not controller:
        raise ValueError("N1 campaign coverage leaves an empty predictor/controller pool")
    return {"predictor": predictor, "controller": controller,
            "records": records, "dropped": dropped}


def make_plan(args):
    source = old.load_plan(source_args(args))
    if read(args.issue71 / "data.json")["lineages"] != 3000:
        raise ValueError("requires completed #71 training shards")
    pools = n1_lineages()
    campaign_plan = read(CAMPAIGN / "plan.json")
    return {"schema": SCHEMA, "identity": IDENTITY,
            "issue71": str(args.issue71.resolve()), "parser": str(args.parser.resolve()),
            "source_plan_identity": source["identity"], "contract": source["contract"],
            "source_release_identity": source["source_release_identity"],
            "n1_campaign": {"root": str(CAMPAIGN), "identity": campaign_plan["identity"],
                            "coverage_complete": True,
                            "training_role_lineages": len(pools["predictor"]) + len(pools["controller"]),
                            "dropped_lineages": pools["dropped"],
                            "admissible_shots": sum(len(r["branches"]) for r in pools["records"].values()),
                            "typed_failure_absorption": "lineage joins pool with >=1 admissible branch; "
                                                        "zero-admissible lineages recorded as dropped"},
            "n1_lineages": pools,
            "capacity": capacity_contract(), "seeds": list(SEEDS),
            "training": {"steps": 9000, "batch_size": 64, "learning_rate": .0001,
                         "weight_decay": .0001, "grad_clip": 1., "unroll": 4,
                         "schedule": "#71 three fitting lineages per nine steps over the 2400-entry frozen pool "
                                     "plus the N1 predictor pool (membership frozen in n1_lineages); identical "
                                     "random minibatches per paired seed",
                         "horizons": "hybrid PAIRS[step%9]; continuous uses same horizon, all 9000 updates; "
                                     "N1 windows are observed-frame units (stride 50 native steps)",
                         "recipe": "local MSE + recursive MSE + .01 carrier bound penalty; hybrid adds .1 symbolic losses",
                         "selection": "last update, no best-validation checkpoint",
                         "fit_lineages": 2400 + len(pools["predictor"]),
                         "controller_lineages": 600 + len(pools["controller"]),
                         "n1_action_tensor": "true executed action incl. 1000 ms hold; frozen #62 data unchanged",
                         "n1_symbolic_scope": "18 contract slots; world:ground_extension scenery skipped",
                         "parser_overlap": "shared frozen CNN saw all 3000 training lineages; none of calibration used for fitting"},
            "controller": {"steps_per_round": 1800, "batch_size": 128, "learning_rate": .001,
                           "rounds": 2, "compute_weight": .0001,
                           "coverage": "first window per shot, both arms; #75 later-window intervention NOT applied",
                           "utility": "duration times continuous carrier MSE + own continuous-h15-normalized linear MAC cost + DP continuation",
                           "aggregation": "one round; predicted visited contexts, unchanged training-only continuous targets",
                           "capacity_rule": "hybrid width128; nearest continuous width to its parameter count is131; tolerance0.2%; no dead output logits",
                           "capacity": {a: parameter_count(MatchedController(a == 'continuous')) for a in ARMS}},
            "systems": {"continuous": "independently initialized dynamics; no symbolic heads/adapters/labels",
                        "hybrid": "new matched hybrid fit; exclusive learned mode per step",
                        "comparison": "paired seeds, identical expanded corpus, identical update budgets"},
            "claim_boundary": "normal-mechanics breadth only (plan §4.5); no novelty-generalization claim (that is N2); "
                              "shared semantically supervised CNN; pure continuous baseline has no symbolic modules",
            "cost_estimates_not_measurements": {"new_capture_count": 0, "new_disk_gib_upper_planning": 2,
                                               "cpu_ram_gib_planning": 3, "cuda_ram_gib_planning": 2,
                                               "training_and_labels_gpu_hours_planning": 5},
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_text": {p: (ROOT / p).read_text() for p in FILES},
            "archived_release": False, "fresh_evaluation_opened": False,
            "final_evaluation_opened": False, "issue_64_authorized": False}


def load_plan(args):
    plan = read(args.output / "plan.json")
    current = make_plan(args)
    current["source_revision"] = plan["source_revision"]
    if plan != current:
        raise ValueError("frozen issue-77 N1 source/settings differ; retain old output and explicitly version any change")
    return plan


# ---------------------------------------------------------------- N1 shards

def n1_shard_path(args, identity, pools):
    return args.output / "shards" / f"lineage-n1-{pools['records'][identity]['ordinal']:03d}.pt"


def _capture_shim(segment, samples, events):
    manifest = json.loads((Path(segment["native_root"]) / "native-manifest.json").read_text())
    return PhysicsCaptureV2(
        capture_id=manifest["capture_id"], shot_id=manifest["shot_id"],
        configured_fixed_step_capture_stride=50, source_bindings=segment["source_bindings"],
        record={"fixed_step_samples": samples, "events": events})


def build_n1_shot(attempt_root, segment, shot_ordinal, adapter, vocabulary, device):
    """One N1 branch -> issue-71-format windows over the observed frames."""
    shot_root = Path(attempt_root) / "shot-1"
    manifest = read(shot_root / "observation-trace" / "observation_trace_manifest.json")
    frames = manifest["frame_records"]
    if (len(frames) < 2 or frames[0]["fixed_step"] != 30000
            or any(f["fixed_step"] != 30000 + 50 * i for i, f in enumerate(frames[:-1]))
            or not (30000 + 50 * (len(frames) - 2) < frames[-1]["fixed_step"]
                    <= 30000 + 50 * (len(frames) - 1))):
        raise ValueError("N1 shot observation cadence differs from the declared 30000+50i contract")
    trace = NativeTrace(segment["native_root"])
    native_manifest = trace.manifest
    summary = segment["summary"]
    if [f["fixed_step"] for f in native_manifest["frame_records"]] != [f["fixed_step"] for f in frames]:
        raise ValueError("native physics and observations do not share exact endpoints")
    if (native_manifest["first_fixed_step"] != summary["first_fixed_step"]
            or native_manifest["last_fixed_step"] != summary["last_fixed_step"]
            or native_manifest["sample_count"] != summary["sample_count"]):
        raise ValueError("N1 segment summary differs from the native trace")
    frame_steps = {f["fixed_step"] for f in frames}
    # One chunk pass: native stability events establish steady state; frame-step
    # samples are retained for micro/targets; macro labels follow the
    # native_history_data.native_labels semantics (steady from the latest
    # engine stability event; structure-unstable from the immediately
    # preceding native step's support set), emitted in the issue-70
    # repair.targets predicate format.
    samples = []
    macro = {}
    steady = None
    previous_supports = None
    for chunk in trace.chunks():
        stability = {event["fixed_step"]: event["event_type"] == "stable_entered"
                     for event in chunk["events"]
                     if event["event_type"] in ("stable_entered", "stable_exited")}
        for sample in chunk["fixed_step_samples"]:
            step = sample["fixed_step"]
            if step in stability:
                steady = stability[step]
            supports = {(s["supporter_entity_id"], s["supported_entity_id"])
                        for s in sample["supports"]}
            if step in frame_steps:
                steady_label = {"value": None if steady is None else bool(steady),
                                "availability": "available" if steady is not None
                                else "unavailable_incomplete_debounce_window"}
                unstable_available = steady is not None and previous_supports is not None
                unstable_label = {"value": (steady is False and previous_supports is not None
                                            and supports != previous_supports)
                                  if unstable_available else None,
                                  "availability": "available" if unstable_available
                                  else ("unavailable_steady_state" if steady is None
                                        else "unavailable_no_predecessor")}
                macro[step] = {"fixed_step": step,
                               "predicates": {"steady-state": steady_label,
                                              "structure-unstable": unstable_label}}
                samples.append(sample)
            previous_supports = supports
    if [s["fixed_step"] for s in samples] != [f["fixed_step"] for f in frames]:
        raise ValueError("N1 native samples and observed frames are misaligned")
    source_reference = f"issue-77-n1:{segment['capture_id']}"
    bundle = f"issue-77-n1:{segment['identity']}"
    micro = derive_capture_micro_relations(
        _capture_shim(segment, samples, []), source_reference=source_reference,
        source_capture_bundle_identity=bundle)["labels"]
    slots = set(vocabulary)
    last = len(frames) - 1
    starts = sorted({round(i * max(0, last - 60) / 3) for i in range(4)})
    needed = sorted({p for start in starts for p in range(max(0, start - 1), min(start + 60, last) + 1)})
    parsed = {}
    observation_root = shot_root / "observation-trace"
    for begin in range(0, len(needed), 32):
        positions = needed[begin:begin + 32]
        observations = []
        for p in positions:
            ref = frames[p]["agent_observation"]
            observations.append(AgentObservation(ref["identity"], frames[p]["fixed_step"],
                                                 frames[p]["fixed_time_seconds"],
                                                 (observation_root / ref["relative_path"]).read_bytes(), "agent"))
        parsed.update(zip(positions, adapter.parse_batch(tuple(observations)), strict=True))

    def observation(p):
        ref = frames[p]["agent_observation"]
        return AgentObservation(ref["identity"], frames[p]["fixed_step"],
                                frames[p]["fixed_time_seconds"], b"", "agent")

    action = segment["action"]
    x, y = action["drag_release"]
    hold_ms = action.get("release_time", action.get("release_time_ms", 1000))
    a = torch.tensor((x / 480., y / 480., hold_ms / 1000., 0., 1.), dtype=torch.float32)
    windows = []
    for start in starts:
        length = min(60, last - start)
        carriers, targets = [], []
        for p in range(start, start + length + 1):
            context = TemporalObservationContext(None if p == 0 else observation(p - 1), observation(p))
            carriers.append(adapter.build_from_parsed(context, parsed[p], None if p == 0 else parsed[p - 1]).tensor)
            sample = samples[p]
            filtered = {**sample, "entities": [e for e in sample["entities"]
                                               if e["scenario_object_id"] in slots]}
            targets.append(old.repair.targets(filtered, frames[p]["capture_metadata"], micro[p],
                                              macro[sample["fixed_step"]], vocabulary))
        while len(carriers) < 61:
            carriers.append(carriers[-1])
            targets.append(targets[-1])
        windows.append({"z": torch.stack(carriers), "action": a, "length": torch.tensor(length),
                        **{k: torch.stack([t[k] for t in targets]).bool()
                           for k in ("relations", "relation_mask", "macros", "macro_mask")},
                        "shot": shot_ordinal, "start": start})
    return windows


def build_n1_shards(args, plan):
    """Derive training-role lineage shards from coverage-admissible branches."""
    pools = plan["n1_lineages"]
    adapter = old.repair.load_repaired_adapter(args.parser, args.device)
    vocabulary = plan["contract"]["vocabulary"]
    campaign_plan = read(CAMPAIGN / "plan.json")
    branch_actions = {b["identity"]: b["action"] for b in campaign_plan["branches"]}
    began = time.monotonic()
    total = 0
    total_lineages = len(pools["predictor"]) + len(pools["controller"])
    for ordinal, identity in enumerate([*pools["predictor"], *pools["controller"]], 1):
        record = pools["records"][identity]
        path = n1_shard_path(args, identity, pools)
        if path.exists():
            data = load_shard_entry(args, plan, identity)
            total += len(data["z"])
            log(f"shard lineage={ordinal}/{total_lineages} {identity} cached")
            continue
        windows = []
        for shot_ordinal, branch in enumerate(record["branches"]):
            result = read(CAMPAIGN / "results" / f"{branch}.json")
            segment = result["segments"][0]
            executed = segment["action"]
            declared = branch_actions[branch]
            if (result["member_identity"] != branch or result["complete"] is not True
                    or list(executed["drag_release"]) != [declared["drag_x"], declared["drag_y"]]
                    or executed["release_time"] != declared["release_time_ms"]
                    or executed["tap_time"] != declared["tap_time_ms"]):
                raise ValueError("N1 shard source result differs from the frozen campaign membership")
            windows.extend(build_n1_shot(CAMPAIGN / "attempts" / branch, segment, shot_ordinal,
                                         adapter, vocabulary, args.device))
            log(f"shard lineage={ordinal}/{total_lineages} {identity} branch={branch} windows={len(windows)}")
        shard = {"schema": SCHEMA, "contract": plan["contract"], "record": record,
                 "windows": [{"shot": w["shot"], "start": w["start"]} for w in windows],
                 "tensors": {k: torch.stack([w[k] for w in windows])
                             for k in ("z", "action", "length", "relations", "relation_mask", "macros", "macro_mask")}}
        shard["tensors"]["relations_mask"] = shard["tensors"].pop("relation_mask")
        shard["tensors"]["macros_mask"] = shard["tensors"].pop("macro_mask")
        old.repair.atomic_torch(path, shard)
        total += len(shard["tensors"]["z"])
        elapsed = time.monotonic() - began
        log(f"shard lineage={ordinal}/{total_lineages} {identity} complete windows={len(shard['tensors']['z'])} "
            f"elapsed={elapsed:.1f}s eta={elapsed/ordinal*(total_lineages-ordinal):.1f}s")
    write(args.output / "n1-data.json", {"plan_identity": plan["identity"], "lineages": total_lineages,
                                         "windows": total, "wall_seconds": time.monotonic() - began})


# ---------------------------------------------------------------- pool plumbing

def fitting_pool(plan):
    return tuple([i for i in range(1, 3001) if i % 5] + plan["n1_lineages"]["predictor"])


def controller_pool(plan):
    return tuple(list(range(5, 3001, 5)) + plan["n1_lineages"]["controller"])


def fitting_lineage_group(plan, step):
    pool = fitting_pool(plan)
    return tuple(pool[((step // 9) * 3 + j) % len(pool)] for j in range(3))


def load_shard_entry(args, plan, entry, source=None, pure=False):
    if isinstance(entry, int):
        data = old.load_shard(source_args(args), source, entry)
    else:
        value = torch.load(n1_shard_path(args, entry, plan["n1_lineages"]),
                           map_location="cpu", weights_only=True)
        if value["contract"] != plan["contract"] or value["record"] != plan["n1_lineages"]["records"][entry]:
            raise ValueError("N1 shard source/role/representation differs")
        data = value["tensors"]
    return {k: data[k] for k in ("z", "action", "length")} if pure else data


def shard(args, plan, source, entry, pure=False):
    return load_shard_entry(args, plan, entry, source, pure)


def first_windows(args, plan, entry):
    if isinstance(entry, int):
        raw = torch.load(old.shard_path(source_args(args), entry), map_location="cpu", weights_only=True)
    else:
        raw = torch.load(n1_shard_path(args, entry, plan["n1_lineages"]),
                         map_location="cpu", weights_only=True)
    seen, selected = set(), []
    for i, window in enumerate(raw["windows"]):
        if window["shot"] not in seen:
            seen.add(window["shot"])
            selected.append(i)
    return selected


def label_path(root, round_index, entry):
    name = f"lineage-{entry:04d}.pt" if isinstance(entry, int) else \
        f"lineage-n1-{int(entry.rsplit('-', 1)[1]):03d}.pt"
    return root / f"labels-{round_index}" / name


# ---------------------------------------------------------------- training

cell = base.cell
new_model = base.new_model
binding = base.binding
check_binding = base.check_binding
loss_for = base.loss_for


def train_cell(args, plan, source, seed, arm, stop_after=None):
    """Resume optimizer/sampler exactly. stop_after is for bounded smoke/tests only."""
    root = cell(args, seed, arm)
    target = root / "predictor.pt"
    if target.exists():
        load_predictor(args, plan, seed, arm)
        log(f"seed={seed} arm={arm} predictor reused")
        return
    torch.manual_seed(seed)
    model = new_model(plan, arm).to(args.device)
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    recipe = plan["training"]
    optim = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"],
                              weight_decay=recipe["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    expected = binding(plan, seed, arm, "predictor")
    start, prior, gradients, counts = 0, 0., set(), Counter()
    progress = root / "predictor-progress.pt"
    if progress.exists():
        saved = torch.load(progress, map_location="cpu", weights_only=True)
        check_binding(saved, expected)
        model.load_state_dict(saved["model"], strict=True)
        optim.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"])
        start, prior = saved["step"], saved["wall_seconds"]
        gradients, counts = set(saved["gradient_parameters"]), Counter(saved["pair_counts"])
    began = time.monotonic()
    cached = None
    finish = min(recipe["steps"], stop_after if stop_after is not None else recipe["steps"])
    for step in range(start, finish):
        group = fitting_lineage_group(plan, step)
        if group != cached:
            rows = [shard(args, plan, source, i, arm == "continuous") for i in group]
            data = {k: torch.cat([r[k] for r in rows]) for k in rows[0]}
            cached = group
        batch = old.sample(data, recipe["batch_size"], generator, args.device)
        selected = PAIRS[step % 9]
        pair = PredictionPair(selected.delta, Abstraction.CONTINUOUS) if arm == "continuous" else selected
        optim.zero_grad(set_to_none=True)
        loss = loss_for(model, batch, pair)
        if not bool(torch.isfinite(loss)):
            raise ValueError(f"nonfinite training loss seed={seed} arm={arm} step={step+1}")
        loss.backward()
        gradients.update(n for n, p in model.named_parameters() if p.grad is not None and bool((p.grad != 0).any()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["grad_clip"])
        optim.step()
        counts[str(pair.identity)] += 1
        if (step + 1) % 90 == 0 or step + 1 == finish:
            elapsed = prior + time.monotonic() - began
            saved = {"binding": expected, "model": model.state_dict(), "optimizer": optim.state_dict(),
                     "generator": generator.get_state(), "step": step + 1, "wall_seconds": elapsed,
                     "gradient_parameters": sorted(gradients), "pair_counts": dict(counts),
                     "changed_parameters": [n for n, p in model.named_parameters()
                                            if not torch.equal(p.detach().cpu(), initial[n])],
                     "optimizer_examples": (step + 1) * recipe["batch_size"], "memory": memory(args.device)}
            old.repair.atomic_torch(progress, saved)
            log(f"train seed={seed} arm={arm} step={step+1}/{recipe['steps']} loss={float(loss.detach()):.6f} "
                f"elapsed={elapsed:.1f}s memory={saved['memory']}")
    if finish == recipe["steps"]:
        old.repair.atomic_torch(target, saved)


load_predictor = base.load_predictor


def label_lineage(args, plan, source, model, controller, entry, aggregated):
    data = shard(args, plan, source, entry, pure=True)
    records = []
    for position in first_windows(args, plan, entry):
        batch = {k: v[position:position + 1].to(args.device) for k, v in data.items()}
        z, a, length = batch["z"], batch["action"], batch["length"]
        contexts = z.clone()
        if aggregated:
            current, t = z[:, 0], 0
            with torch.no_grad():
                while t < int(length[0]):
                    pair = controller.pairs[int(controller(current, a, length - t).argmax(-1))]
                    current = model.carrier(current, a, pair)
                    t += pair.delta
                    contexts[:, t] = current
        labels = controller_labels(model, controller, z, a, length, contexts)
        n = int(length[0])
        records.append({"z": contexts[0, :n].cpu(), "action": a.expand(n, -1).cpu(),
                        "remaining": torch.arange(n, 0, -1), "labels": labels[0, :n].cpu()})
    return {k: torch.cat([r[k] for r in records]) for k in records[0]}


def train_control(args, plan, source, seed, arm, indices=None):
    indices = controller_pool(plan) if indices is None else indices
    root = cell(args, seed, arm)
    model, _ = load_predictor(args, plan, seed, arm)
    if (root / "controller.pt").exists():
        load_control(args, plan, seed, arm)
        log(f"seed={seed} arm={arm} controller reused")
        return
    torch.manual_seed(seed + 7700)
    control = MatchedController(arm == "continuous").to(args.device)
    initial = {n: p.detach().cpu().clone() for n, p in control.named_parameters()}
    expected = binding(plan, seed, arm, "controller")
    recipe = plan["controller"]
    gradients = set()
    for round_index in (0, 1):
        checkpoint = root / f"controller-round-{round_index}.pt"
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            check_binding(saved, expected)
            control.load_state_dict(saved["model"], strict=True)
            gradients.update(saved["gradient_parameters"])
            continue
        began = time.monotonic()
        for ordinal, entry in enumerate(indices, 1):
            path = label_path(root, round_index, entry)
            record = (source["records"][entry - 1] if isinstance(entry, int)
                      else plan["n1_lineages"]["records"][entry])
            if not path.exists():
                started = time.monotonic()
                tensors = label_lineage(args, plan, source, model, control, entry, bool(round_index))
                wall = time.monotonic() - started
                old.repair.atomic_torch(path, {"binding": expected, "record": record,
                                               "round": round_index, "tensors": tensors,
                                               "wall_seconds": wall})
            else:
                cached = torch.load(path, weights_only=True, map_location="cpu")
                check_binding(cached, expected)
                if cached["record"] != record or cached["round"] != round_index:
                    raise ValueError("controller labels source/round differs")
            elapsed = time.monotonic() - began
            log(f"labels seed={seed} arm={arm} round={round_index+1}/2 lineage={ordinal}/{len(indices)} "
                f"elapsed={elapsed:.1f}s eta={elapsed/ordinal*(len(indices)-ordinal):.1f}s")
        optim = torch.optim.AdamW(control.parameters(), lr=recipe["learning_rate"])
        generator = torch.Generator().manual_seed(seed + 7700 + round_index)
        progress = root / f"controller-progress-{round_index}.pt"
        start, prior = 0, 0.
        if progress.exists():
            saved = torch.load(progress, weights_only=True, map_location="cpu")
            check_binding(saved, expected)
            control.load_state_dict(saved["model"], strict=True)
            optim.load_state_dict(saved["optimizer"])
            generator.set_state(saved["generator"])
            start, prior = saved["step"], saved["wall_seconds"]
            gradients.update(saved["gradient_parameters"])
        began = time.monotonic()
        for step in range(start, recipe["steps_per_round"]):
            entry = indices[step % len(indices)]
            rows = [torch.load(label_path(root, r, entry), weights_only=True, map_location="cpu")["tensors"]
                    for r in range(round_index + 1)]
            data = {k: torch.cat([r[k] for r in rows]) for k in rows[0]}
            batch = old.sample(data, recipe["batch_size"], generator, args.device)
            loss = F.cross_entropy(control(batch["z"], batch["action"], batch["remaining"]), batch["labels"])
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite controller loss")
            optim.zero_grad(set_to_none=True)
            loss.backward()
            gradients.update(n for n, p in control.named_parameters()
                             if p.grad is not None and bool((p.grad != 0).any()))
            optim.step()
            if (step + 1) % 60 == 0 or step + 1 == recipe["steps_per_round"]:
                elapsed = prior + time.monotonic() - began
                saved = {"binding": expected, "model": control.state_dict(), "optimizer": optim.state_dict(),
                         "generator": generator.get_state(), "round": round_index, "step": step + 1,
                         "wall_seconds": elapsed, "gradient_parameters": sorted(gradients),
                         "changed_parameters": [n for n, p in control.named_parameters()
                                                if not torch.equal(p.detach().cpu(), initial[n])],
                         "memory": memory(args.device)}
                old.repair.atomic_torch(progress, saved)
                log(f"controller seed={seed} arm={arm} round={round_index+1}/2 "
                    f"step={step+1}/{recipe['steps_per_round']} elapsed={elapsed:.1f}s")
        old.repair.atomic_torch(checkpoint, saved)
    old.repair.atomic_torch(root / "controller.pt", saved)


load_control = base.load_control


def smoke(args):
    plan = make_plan(args)
    source = old.load_plan(source_args(args))
    plan["training"]["steps"] = 18
    plan["controller"]["steps_per_round"] = 6
    began = time.monotonic()
    result = {"production_evidence": False, "cells": {}}
    with tempfile.TemporaryDirectory(prefix="novphy-issue77-n1-smoke-") as temporary:
        test = copy.copy(args)
        test.output = Path(temporary)
        for arm in ARMS:
            train_cell(test, plan, source, SEEDS[0], arm, stop_after=9)
            train_cell(test, plan, source, SEEDS[0], arm, stop_after=18)
            saved = torch.load(cell(test, SEEDS[0], arm) / "predictor-progress.pt",
                               weights_only=True, map_location="cpu")
            model = new_model(plan, arm).to(args.device)
            model.load_state_dict(saved["model"], strict=True)
            train_control(test, plan, source, SEEDS[0], arm, indices=(5,))
            train_control(test, plan, source, SEEDS[0], arm, indices=(5,))
            control, _ = load_control(test, plan, SEEDS[0], arm)
            names = set(dict(model.named_parameters()))
            if set(saved["gradient_parameters"]) != names or set(saved["changed_parameters"]) != names:
                raise ValueError("smoke found untrained parameter tensors")
            z = torch.zeros(1, 236, device=args.device)
            a = torch.zeros(1, 5, device=args.device)
            pair = PAIRS[0] if arm == "hybrid" else PredictionPair(PAIRS[0].delta, Abstraction.CONTINUOUS)
            out, trace = rollout(model, control, z, a, 30, pair)
            if sum(t["horizon"] for t in trace) != 30:
                raise ValueError("smoke rollout did not reach the requested horizon")
            result["cells"][arm] = {"resume_step": saved["step"], "gradient_tensors": len(names),
                                    "rollout_segments": len(trace)}
    result.update({"wall_seconds": time.monotonic() - began, "memory": memory(args.device),
                   "capacity": plan["capacity"]})
    write(args.output / "smoke.json", result)
    log(f"real smoke complete wall={result['wall_seconds']:.1f}s; temporary weights removed, no production training")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "smoke-test", "prepare", "build-shards", "train"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--issue71", type=Path, default=old.ROOT)
    parser.add_argument("--parser", type=Path, default=old.repair.ROOT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        if args.dry_run:
            plan = make_plan(args)
            log(f"dry run: seeds={plan['seeds']} six new fits; pools 2409/603; capacity={plan['capacity']}")
            log(f"estimates (not measured full run)={plan['cost_estimates_not_measurements']}; no files written")
            return 0
        if args.smoke_test:
            smoke(args)
            return 0
        if args.prepare:
            if (args.output / "plan.json").exists():
                load_plan(args)
            else:
                write(args.output / "plan.json", make_plan(args))
            log("plan frozen; no corpus copy, no training started")
            return 0
        plan = load_plan(args)
        if args.build_shards:
            build_n1_shards(args, plan)
            log(f"N1 shards complete memory={memory(args.device)}")
        elif args.train:
            source = old.load_plan(source_args(args))
            if not (args.output / "n1-data.json").exists():
                raise ValueError("run --build-shards before --train")
            for seed in SEEDS:
                for arm in ARMS:
                    log(f"training cell seed={seed} arm={arm} start")
                    train_cell(args, plan, source, seed, arm)
                    train_control(args, plan, source, seed, arm)
            log(f"six matched fits/controllers complete memory={memory(args.device)}")
        return 0
    except (ValueError, OSError) as error:
        log(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
