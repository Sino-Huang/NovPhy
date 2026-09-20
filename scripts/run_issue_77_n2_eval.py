"""Issue-77 N2: matched normal/novel appearance evaluation (zero-shot + few-shot).

Small-project mode (plan section 0): no authorization gates. Declared before any
model outcome was seen (2026-09-20):

- Pairs and sides (pinned membership):
  * type010101: normal = the frozen R3 bounded-transfer corpus
    (`.local-artifacts/issue-76-bounded-transfer-v1`) held_out_model_selection
    lineages, the first eight by identity with at least one coverage-admissible
    branch. R3 lineages were never gradient-fitted; the #75 development
    model-selection scoring optimism on them is retained and declared.
    novel = issue-77 N2 campaign held_out_evaluation lineages (ordinals 7, 8).
  * type010102: normal = the issue-77-n2n campaign's eight evaluation-only
    lineages (fresh, never used elsewhere; plan section 5.2's "normal side
    already exists" holds only for type010101 via the R3 corpus - the issue-62
    cohort used behavior-policy actions, not the fixed 13-action grid).
    novel = issue-77 N2 campaign held_out_evaluation lineages (ordinals 14, 16).
- Candidates: each state's 13 fixed actions restricted to coverage-admissible
  branches; dropped candidates are reported and never treated as worst case.
  The three campaigns' action grids are verified identical at plan freeze.
- Conditions: zero-shot = the frozen issue-77 N1 dynamics checkpoints (fitted
  on the issue-71 corpus plus N1 rolling/sliding only; never on appearance
  novelty). few-shot = the same checkpoints adapted on the N2 novel
  training-role predictor lineages: 2000 steps, batch 64, AdamW lr 1e-4,
  weight decay 1e-4, grad clip 1.0, the unchanged N1 loss/horizon schedule,
  paired initialization RNG seeds 764400001..764400003 (one per paired seed
  position, identical across arms); identical minibatch draws per paired seed.
  Controllers are not adapted: the evaluation uses fixed systems only.
- Systems/endpoints/uncertainty: identical to the issue-77 N1 diagnostic
  (12 fixed systems, t in {15,30,60,150,225,600}, task cost at t=600
  end-of-window, bootstrap 10000 draws seed 7201, seed differences averaged
  within state before resampling; descriptive only).
- Zero-shot and few-shot are reported separately and never mixed (plan
  section 0). The appearance cell doubles as the perception control (plan
  section 5.5): it probes the shared CNN parser, not dynamics. The cross-side
  change in the hybrid-minus-continuous effect uses a two-sample bootstrap
  over each side's independent states (sides hold different lineages).
  Claims are limited to the two appearance cells.
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
from scripts import run_issue_77_n1_diagnostic as diag
from world_model.data.deployment_temporal import AgentObservation, TemporalObservationContext
from world_model.model import Abstraction, PredictionPair

ROOT = base.ROOT
N2_CAMPAIGN = ROOT / ".local-artifacts/issue-77-n2-appearance-v1"
N2N_CAMPAIGN = ROOT / ".local-artifacts/issue-77-n2n-v1"
R3_CAMPAIGN = ROOT / ".local-artifacts/issue-76-bounded-transfer-v1"
DYNAMICS = train.OUTPUT
OUTPUT = ROOT / ".local-artifacts/issue-77-n2-eval-v1"
SCHEMA = "issue_77_n2_eval_v1"
IDENTITY = "issue-77-n2-eval-v1"
TIMES = diag.TIMES
TASK_TIME = diag.TASK_TIME
HORIZONS = diag.HORIZONS
SEEDS = train.SEEDS
ADAPT_SEEDS = (764400001, 764400002, 764400003)
ARMS = base.ARMS
SYSTEMS = diag.SYSTEMS
CONDITIONS = ("zero-shot", "few-shot")
ADAPTATION = {"steps": 2000, "batch_size": 64, "learning_rate": .0001, "weight_decay": .0001,
              "grad_clip": 1., "pool": "N2 novel training-role predictor lineages with >=1 admissible branch",
              "schedule": "three lineages per nine steps over the adaptation pool; PAIRS[step%9] horizons; "
                          "continuous arm uses the same horizon in continuous mode; unchanged N1 loss",
              "paired_seeds": list(ADAPT_SEEDS),
              "controllers": "not adapted; fixed systems only",
              "selection": "last update, no best-validation checkpoint"}
R3_NORMAL_LINEAGES = 8
FILES = tuple(dict.fromkeys((*diag.FILES, "scripts/run_issue_77_n2_eval.py")))
read, write = base.read, base.write


def log(message):
    print(f"[issue-77-n2-eval] {message}", flush=True)


def dynamics_args(args):
    return diag.dynamics_args(args)


# ---------------------------------------------------------------- membership

def _coverage(root, schema, require_complete):
    coverage = read(root / "coverage.json")
    if coverage.get("schema") != schema:
        raise ValueError(f"unexpected coverage schema for {root.name}")
    if require_complete and coverage.get("coverage_complete") is not True:
        raise ValueError(f"requires completed campaign coverage: {root.name}")
    return coverage


def _member_samples(campaign_root, campaign_plan, coverage, members):
    """State dicts with coverage-admissible 13-action candidates (N1 diagnostic shape)."""
    plan_branches = {b["identity"]: b for b in campaign_plan["branches"]}
    samples = []
    for member in members:
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
                        "sources": [_candidate_source(campaign_root, b, plan_branches) for b in admissible]})
    return samples


def _candidate_source(campaign_root, branch, plan_branches):
    declared = plan_branches[branch["identity"]]
    result = read(campaign_root / "results" / f"{branch['identity']}.json")
    segment = result["segments"][0]
    executed = segment["action"]
    if (result["member_identity"] != branch["identity"] or result["complete"] is not True
            or list(executed["drag_release"]) != [declared["action"]["drag_x"], declared["action"]["drag_y"]]
            or executed["release_time"] != declared["action"]["release_time_ms"]
            or executed["tap_time"] != declared["action"]["tap_time_ms"]):
        raise ValueError("N2 candidate source differs from frozen campaign membership")
    summary = segment["summary"]
    if summary["first_fixed_step"] != 30000 or summary["frame_count"] < 2:
        raise ValueError("N2 candidate observation window differs from the declared contract")
    terminal = summary.get("terminal_evidence")
    return {"identity": branch["identity"], "status": "accepted",
            "ordinal": branch["candidate_ordinal"],
            "action": declared["action"],
            "campaign_root": str(campaign_root),
            "attempt": f"attempts/{branch['identity']}",
            "capture_id": segment["capture_id"],
            "native_root": segment["native_root"],
            "observation_manifest": segment["observation_manifest"],
            "terminal_reason": None if terminal is None else terminal["reason"],
            "censored": summary["censored"],
            "first_fixed_step": 30000, "last_offset": summary["frame_count"] - 1,
            "frame_count": summary["frame_count"]}

def r3_normal_members():
    """First eight R3 type010101 held_out_model_selection lineages by identity
    with at least one coverage-admissible branch (pinned rule)."""
    plan = read(R3_CAMPAIGN / "plan.json")
    coverage = _coverage(R3_CAMPAIGN, "issue_76_bounded_transfer_coverage_v1", False)
    members = sorted((m for m in plan["members"]
                      if m["generator_family"] == "type010101"
                      and m.get("study_role") == "held_out_model_selection"),
                     key=lambda m: m["identity"])
    chosen = []
    for member in members:
        admissible = any(coverage["branches"][b["identity"]]["status"] == "admissible"
                         for b in plan["branches"] if b["source_member_identity"] == member["identity"])
        if admissible:
            chosen.append(member)
        if len(chosen) == R3_NORMAL_LINEAGES:
            break
    if len(chosen) < R3_NORMAL_LINEAGES:
        raise ValueError("R3 corpus holds fewer than eight admissible held-out type010101 lineages")
    return chosen, plan, coverage


def n2_novel_members():
    plan = read(N2_CAMPAIGN / "plan.json")
    coverage = _coverage(N2_CAMPAIGN, "issue_77_n2_coverage_v1", True)
    held_out = sorted((m for m in plan["members"] if m["study_role"] == "held_out_evaluation"),
                      key=lambda m: m["identity"])
    adaptation = sorted((m for m in plan["members"]
                         if m["exposure_role"] == "training" and m["fit_partition"] == "predictor"),
                        key=lambda m: m["ordinal"])
    admissible_adaptation, dropped_adaptation = [], []
    for member in adaptation:
        admissible = any(coverage["branches"][b["identity"]]["status"] == "admissible"
                         for b in plan["branches"] if b["source_member_identity"] == member["identity"])
        (admissible_adaptation if admissible else dropped_adaptation).append(member)
    if not admissible_adaptation:
        raise ValueError("N2 coverage leaves an empty adaptation pool")
    return held_out, admissible_adaptation, dropped_adaptation, plan, coverage
def n2n_normal_members():
    plan = read(N2N_CAMPAIGN / "plan.json")
    coverage = _coverage(N2N_CAMPAIGN, "issue_77_n2n_coverage_v1", True)
    members = sorted(plan["members"], key=lambda m: m["identity"])
    if any(m["study_role"] != "held_out_evaluation" for m in members):
        raise ValueError("issue-77-n2n campaign membership is not evaluation-only")
    return members, plan, coverage


def _action_grids(*plans):
    grids = []
    for plan in plans:
        grid_actions = {}
        for branch in plan["branches"]:
            a = branch["action"]
            grid_actions[str(branch["candidate_ordinal"])] = {k: a[k] for k in
                                                              ("drag_x", "drag_y", "release_time_ms", "tap_time_ms")}
        grids.append(grid_actions)
    first = grids[0]
    if sorted(first, key=int) != [str(i) for i in range(13)] or any(g != first for g in grids[1:]):
        raise ValueError("paired appearance campaigns hold different fixed action grids")
    return first


def select_samples():
    """Pinned pair/side membership with coverage-admissible candidates."""
    r3_members, r3_plan, r3_coverage = r3_normal_members()
    novel_held_out, adaptation, dropped_adaptation, n2_plan, n2_coverage = n2_novel_members()
    n2n_members, n2n_plan, n2n_coverage = n2n_normal_members()
    grid_actions = _action_grids(r3_plan, n2_plan, n2n_plan)
    sides = []
    sides.append({"pair": "type010101", "side": "normal", "corpus": "issue-76-bounded-transfer-v1",
                  "samples": _member_samples(R3_CAMPAIGN, r3_plan, r3_coverage, r3_members)})
    sides.append({"pair": "type010101", "side": "novel", "corpus": "issue-77-n2-appearance-v1",
                  "samples": _member_samples(N2_CAMPAIGN, n2_plan, n2_coverage,
                                             [m for m in novel_held_out
                                              if m["generator_family"] == "type010101"])})
    sides.append({"pair": "type010102", "side": "normal", "corpus": "issue-77-n2n-v1",
                  "samples": _member_samples(N2N_CAMPAIGN, n2n_plan, n2n_coverage, n2n_members)})
    sides.append({"pair": "type010102", "side": "novel", "corpus": "issue-77-n2-appearance-v1",
                  "samples": _member_samples(N2_CAMPAIGN, n2_plan, n2_coverage,
                                             [m for m in novel_held_out
                                              if m["generator_family"] == "type010102"])})
    if not any(s["sources"] for side in sides for s in side["samples"]):
        raise ValueError("no evaluable N2 states in coverage")
    adaptation_records = {}
    for member in adaptation:
        admissible = [b["identity"] for b in n2_plan["branches"]
                      if b["source_member_identity"] == member["identity"]
                      and n2_coverage["branches"][b["identity"]]["status"] == "admissible"]
        adaptation_records[member["identity"]] = {
            "identity": member["identity"], "ordinal": member["ordinal"],
            "generator_family": member["generator_family"],
            "exposure_role": member["exposure_role"], "fit_partition": member["fit_partition"],
            "branches": admissible, "scheduled_branches": 13,
            "dropped_branches": 13 - len(admissible)}
    adaptation_summary = {"lineages": [m["identity"] for m in adaptation],
                          "dropped_lineages": [m["identity"] for m in dropped_adaptation],
                          "members": adaptation, "records": adaptation_records}
    return sides, adaptation_summary, grid_actions


def make_plan(args):
    dynamics = train.load_plan(dynamics_args(args))
    sides, adaptation, grid_actions = select_samples()
    return {"schema": SCHEMA, "identity": IDENTITY,
            "definitions": {**diag.definitions(),
                            "terminal_absorption": "offsets beyond an early-terminal branch's last frame reuse "
                                                   "the terminal frame's carrier; absorbed offsets recorded",
                            "conditions": {"zero-shot": "frozen issue-77 N1 dynamics checkpoints; no novelty fitting",
                                           "few-shot": "N1 checkpoints adapted on N2 novel predictor lineages"},
                            "adaptation": ADAPTATION,
                            "cross_side_effect_change": "two-sample bootstrap (10000 draws, seed 7201) over each "
                                                        "side's independent states; sides hold different lineages",
                            "claims": "two appearance cells only; appearance is the perception control"},
            "dynamics": str(args.dynamics.resolve()), "dynamics_plan_identity": dynamics["identity"],
            "contract": dynamics["contract"], "capacity": dynamics["capacity"],
            "corpora": {"r3": str(R3_CAMPAIGN), "n2": str(N2_CAMPAIGN), "n2n": str(N2N_CAMPAIGN)},
            "action_grid": grid_actions,
            "sides": sides,
            "adaptation": {**ADAPTATION, **adaptation},
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_text": {p: (ROOT / p).read_text() for p in FILES},
            "archived_release": False, "fresh_evaluation_opened": False,
            "final_evaluation_opened": False, "issue_64_authorized": False}


def load_plan(args):
    plan = read(args.output / "plan.json")
    current = make_plan(args)
    current["source_revision"] = plan["source_revision"]
    if plan != current:
        raise ValueError("frozen issue-77 N2 evaluation source/membership changed; preserve and explicitly version")
    return plan


# ---------------------------------------------------------------- adaptation shards

def adaptation_shard_path(args, plan, identity):
    member = next(m for m in plan["adaptation"]["members"] if m["identity"] == identity)
    return args.output / "shards" / f"lineage-n2-{member['ordinal']:03d}.pt"


def build_novel_shards(args, plan):
    """Derive few-shot adaptation shards from the N2 novel predictor lineages."""
    adapter = old.repair.load_repaired_adapter(train.old.repair.ROOT, args.device)
    vocabulary = plan["contract"]["vocabulary"]
    n2_plan = read(N2_CAMPAIGN / "plan.json")
    coverage = _coverage(N2_CAMPAIGN, "issue_77_n2_coverage_v1", True)
    branch_actions = {b["identity"]: b["action"] for b in n2_plan["branches"]}
    members = {m["identity"]: m for m in n2_plan["members"]}
    began = time.monotonic()
    total = 0
    lineages = plan["adaptation"]["lineages"]
    for ordinal, identity in enumerate(lineages, 1):
        member = members[identity]
        path = adaptation_shard_path(args, plan, identity)
        if path.exists():
            cached = torch.load(path, map_location="cpu", weights_only=True)
            if cached["record"] != plan["adaptation"]["records"][identity]:
                raise ValueError("cached adaptation shard membership differs")
            total += len(cached["tensors"]["z"])
            log(f"shard lineage={ordinal}/{len(lineages)} {identity} cached")
            continue
        admissible = [b["identity"] for b in n2_plan["branches"]
                      if b["source_member_identity"] == identity
                      and coverage["branches"][b["identity"]]["status"] == "admissible"]
        windows = []
        for shot_ordinal, branch in enumerate(admissible):
            result = read(N2_CAMPAIGN / "results" / f"{branch}.json")
            segment = result["segments"][0]
            executed = segment["action"]
            declared = branch_actions[branch]
            if (result["member_identity"] != branch or result["complete"] is not True
                    or list(executed["drag_release"]) != [declared["drag_x"], declared["drag_y"]]
                    or executed["release_time"] != declared["release_time_ms"]
                    or executed["tap_time"] != declared["tap_time_ms"]):
                raise ValueError("N2 shard source result differs from the frozen campaign membership")
            windows.extend(train.build_n1_shot(N2_CAMPAIGN / "attempts" / branch, segment, shot_ordinal,
                                               adapter, vocabulary, args.device))
        record = {"identity": identity, "ordinal": member["ordinal"],
                  "generator_family": member["generator_family"],
                  "exposure_role": member["exposure_role"], "fit_partition": member["fit_partition"],
                  "branches": admissible, "scheduled_branches": 13,
                  "dropped_branches": 13 - len(admissible)}
        if record != plan["adaptation"]["records"][identity]:
            raise ValueError("adaptation shard record differs from the frozen plan")
        shard = {"schema": SCHEMA, "contract": plan["contract"], "record": record,
                 "windows": [{"shot": w["shot"], "start": w["start"]} for w in windows],
                 "tensors": {k: torch.stack([w[k] for w in windows])
                             for k in ("z", "action", "length", "relations", "relation_mask", "macros", "macro_mask")}}
        shard["tensors"]["relations_mask"] = shard["tensors"].pop("relation_mask")
        shard["tensors"]["macros_mask"] = shard["tensors"].pop("macro_mask")
        old.repair.atomic_torch(path, shard)
        total += len(shard["tensors"]["z"])
        elapsed = time.monotonic() - began
        log(f"shard lineage={ordinal}/{len(lineages)} {identity} complete windows={len(shard['tensors']['z'])} "
            f"elapsed={elapsed:.1f}s eta={elapsed/max(ordinal,1)*(len(lineages)-ordinal):.1f}s")
    write(args.output / "n2-shards.json", {"plan_identity": plan["identity"], "lineages": len(lineages),
                                           "windows": total, "wall_seconds": time.monotonic() - began})


def load_adaptation_shard(args, plan, identity, pure=False):
    value = torch.load(adaptation_shard_path(args, plan, identity), map_location="cpu", weights_only=True)
    if value["contract"] != plan["contract"] or value["record"] != plan["adaptation"]["records"][identity]:
        raise ValueError("adaptation shard source/role/representation differs")
    data = value["tensors"]
    return {k: data[k] for k in ("z", "action", "length")} if pure else data


# ---------------------------------------------------------------- adaptation

def adapt_binding(plan, train_seed, adapt_seed, arm):
    return {"plan_identity": plan["identity"], "contract": plan["contract"], "capacity": plan["capacity"],
            "dynamics_plan_identity": plan["dynamics_plan_identity"],
            "seed": adapt_seed, "base_seed": train_seed, "arm": arm, "component": "adapted-predictor",
            "adaptation": {k: v for k, v in plan["adaptation"].items()
                           if k in ADAPTATION or k == "paired_seeds"}}


def adapt_cell(args, plan, train_seed, adapt_seed, arm, stop_after=None):
    """Few-shot adaptation of one N1 checkpoint; resume exactly. stop_after is for smoke/tests only."""
    root = args.output / "adapt" / f"seed-{train_seed}" / arm
    target = root / "adapted.pt"
    if target.exists():
        load_adapted(args, plan, train_seed, arm)
        log(f"seed={train_seed} arm={arm} adapted predictor reused")
        return
    torch.manual_seed(adapt_seed)
    dynamics_plan = train.load_plan(dynamics_args(args))
    model, _ = train.load_predictor(dynamics_args(args), dynamics_plan, train_seed, arm)
    model = model.to(args.device).train()
    recipe = plan["adaptation"]
    optim = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"],
                              weight_decay=recipe["weight_decay"])
    generator = torch.Generator().manual_seed(adapt_seed)
    expected = adapt_binding(plan, train_seed, adapt_seed, arm)
    pool = tuple(recipe["lineages"])
    start, prior, gradients, counts = 0, 0., set(), Counter()
    progress = root / "adapted-progress.pt"
    if progress.exists():
        saved = torch.load(progress, map_location="cpu", weights_only=True)
        base.check_binding(saved, expected)
        model.load_state_dict(saved["model"], strict=True)
        optim.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"])
        start, prior = saved["step"], saved["wall_seconds"]
        gradients, counts = set(saved["gradient_parameters"]), Counter(saved["pair_counts"])
    began = time.monotonic()
    cached = None
    finish = min(recipe["steps"], stop_after if stop_after is not None else recipe["steps"])
    for step in range(start, finish):
        group = tuple(pool[((step // 9) * 3 + j) % len(pool)] for j in range(3))
        if group != cached:
            rows = [load_adaptation_shard(args, plan, i, arm == "continuous") for i in group]
            data = {k: torch.cat([r[k] for r in rows]) for k in rows[0]}
            cached = group
        batch = old.sample(data, recipe["batch_size"], generator, args.device)
        selected = train.PAIRS[step % 9]
        pair = PredictionPair(selected.delta, Abstraction.CONTINUOUS) if arm == "continuous" else selected
        optim.zero_grad(set_to_none=True)
        loss = train.loss_for(model, batch, pair)
        if not bool(torch.isfinite(loss)):
            raise ValueError(f"nonfinite adaptation loss seed={train_seed} arm={arm} step={step+1}")
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
                     "optimizer_examples": (step + 1) * recipe["batch_size"],
                     "memory": base.memory(args.device)}
            old.repair.atomic_torch(progress, saved)
            log(f"adapt seed={train_seed} arm={arm} step={step+1}/{recipe['steps']} "
                f"loss={float(loss.detach()):.6f} elapsed={elapsed:.1f}s")
    if finish == recipe["steps"]:
        old.repair.atomic_torch(target, saved)


def load_adapted(args, plan, train_seed, arm):
    adapt_seed = ADAPT_SEEDS[SEEDS.index(train_seed)]
    root = args.output / "adapt" / f"seed-{train_seed}" / arm
    value = torch.load(root / "adapted.pt", map_location="cpu", weights_only=True)
    base.check_binding(value, adapt_binding(plan, train_seed, adapt_seed, arm))
    model = base.new_model(plan, arm).to(args.device)
    model.load_state_dict(value["model"], strict=True)
    names = set(dict(model.named_parameters()))
    if (value["step"] != plan["adaptation"]["steps"]
            or sum(value["pair_counts"].values()) != plan["adaptation"]["steps"]
            or set(value["gradient_parameters"]) != names):
        raise ValueError("incomplete adapted predictor budget/coverage/gradients")
    return model.eval(), value


# ---------------------------------------------------------------- targets

def needed_offsets():
    return diag.needed_offsets()


def target_path(args, state):
    return args.output / "targets" / f"state-{state['identity']}.json"


def curve_path(args, condition, seed, state, system, ordinal):
    return args.output / "fixed" / condition / f"seed-{seed}" / f"state-{state['identity']}" / system / \
        f"candidate-{ordinal:02d}.json"


def target_candidate(args, source, adapter, vocabulary):
    campaign_root = Path(source["campaign_root"])
    root = campaign_root / source["attempt"] / "shot-1"
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
            "timing": {**diag.branch_events(source), "terminal_reason": source["terminal_reason"],
                       "censored": source["censored"], "last_offset": source["last_offset"]},
            "target_parser": {"calls": parse_calls, "image_examples": parse_images,
                              "wall_seconds": time.monotonic() - began,
                              "scope": "evaluation target construction only, not deployment latency"}}


def all_samples(plan):
    return [s for side in plan["sides"] for s in side["samples"]]


def evaluable_samples(plan):
    return [s for s in all_samples(plan) if s["sources"]]


def prepare_targets(args, plan):
    adapter = old.repair.load_repaired_adapter(train.old.repair.ROOT, args.device)
    began = time.monotonic()
    evaluable = evaluable_samples(plan)
    for n, sample in enumerate(evaluable, 1):
        state = sample["state"]
        path = target_path(args, state)
        if path.exists():
            diag.check_targets(read(path), sample, plan)
            log(f"targets state={n}/{len(evaluable)} cached")
            continue
        reference = sample["sources"][0]
        obsroot = Path(reference["campaign_root"]) / reference["attempt"] / "shot-1" / "observation-trace"
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
            targets.append(target_candidate(args, source, adapter, plan["contract"]["vocabulary"]))
        result = {"plan_identity": plan["identity"], "state": state["identity"],
                  "context": context.tolist(), "perception": perception,
                  "dropped_candidates": sample["dropped_candidates"], "candidates": targets}
        diag.check_targets(result, sample, plan)
        write(path, result)
        elapsed = time.monotonic() - began
        log(f"targets state={n}/{len(evaluable)} complete elapsed={elapsed:.1f}s "
            f"eta={elapsed/n*(len(evaluable)-n):.1f}s")


# ---------------------------------------------------------------- evaluation

def condition_models(args, plan, dynamics_plan, condition, train_seed):
    if condition == "zero-shot":
        return {a: train.load_predictor(dynamics_args(args), dynamics_plan, train_seed, a)[0]
                .requires_grad_(False) for a in ARMS}
    return {a: load_adapted(args, plan, train_seed, a)[0].requires_grad_(False) for a in ARMS}


def stored_curve(args, plan, sample, targets, condition, seed, system, source, model):
    path = curve_path(args, condition, seed, sample["state"], system, source["ordinal"])
    if path.exists():
        record = read(path)
    else:
        record = diag.fixed_record(args, plan, sample, targets, seed, system, source, model)
        record["condition"] = condition
        diag.check_curve(record, plan, sample, seed, system, source, model)
        write(path, record)
    if record.get("condition") != condition:
        raise ValueError("fixed record condition binding differs")
    diag.check_curve(record, plan, sample, seed, system, source, model)
    return record


def artifact_inventory(args, plan):
    evaluable = evaluable_samples(plan)
    per_condition = sum(len(s["sources"]) for s in evaluable) * len(SEEDS) * len(SYSTEMS)
    targets = sum(target_path(args, s["state"]).exists() for s in evaluable)
    curves = sum(curve_path(args, condition, seed, s["state"], name, source["ordinal"]).exists()
                 for condition in CONDITIONS for seed in SEEDS
                 for s in evaluable for name in SYSTEMS for source in s["sources"])
    adapted = sum((args.output / "adapt" / f"seed-{seed}" / arm / "adapted.pt").exists()
                  for seed in SEEDS for arm in ARMS)
    return {"evaluable_states": len(evaluable),
            "unsupported_states": len(all_samples(plan)) - len(evaluable),
            "target_states": targets, "expected_target_states": len(evaluable),
            "adapted_predictors": adapted, "expected_adapted_predictors": len(SEEDS) * len(ARMS),
            "fixed_candidate_records": curves,
            "expected_fixed_candidate_records": per_condition * len(CONDITIONS)}


def run_evaluation(args, plan):
    inventory = artifact_inventory(args, plan)
    if (inventory["target_states"] == inventory["expected_target_states"]
            and inventory["adapted_predictors"] == inventory["expected_adapted_predictors"]
            and inventory["fixed_candidate_records"] == inventory["expected_fixed_candidate_records"]):
        log("complete evaluation inventory already exists; no new inference; use --publish/--validate")
        return
    prepare_targets(args, plan)
    dynamics_plan = train.load_plan(dynamics_args(args))
    evaluable = evaluable_samples(plan)
    total = sum(len(s["sources"]) for s in evaluable) * len(SEEDS) * len(SYSTEMS) * len(CONDITIONS)
    began = time.monotonic()
    completed = 0
    for condition in CONDITIONS:
        if condition == "few-shot" and inventory["adapted_predictors"] != inventory["expected_adapted_predictors"]:
            raise ValueError("run --adapt before few-shot evaluation")
        for seed in SEEDS:
            models = condition_models(args, plan, dynamics_plan, condition, seed)
            for n, sample in enumerate(evaluable, 1):
                targets = read(target_path(args, sample["state"]))
                diag.check_targets(targets, sample, plan)
                for system, (arm, pair) in SYSTEMS.items():
                    for source in sample["sources"]:
                        record = stored_curve(args, plan, sample, targets, condition, seed, system,
                                              source, models[arm])
                        completed += 1
                        elapsed = time.monotonic() - began
                        log(f"eval={completed}/{total} condition={condition} seed={seed} "
                            f"state={n}/{len(evaluable)} system={system} candidate={source['ordinal']} "
                            f"failure={record['prediction']['failure']} elapsed={elapsed:.1f}s "
                            f"eta={elapsed/completed*(total-completed):.1f}s")
    log("zero-shot and few-shot fixed-pair evaluations complete")


# ---------------------------------------------------------------- publication

def two_sample_interval(novel_values, normal_values, draws=10000, seed=7201):
    """Descriptive two-sample bootstrap over independent state sets (sides differ)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(novel_values, dtype=float), np.asarray(normal_values, dtype=float)
    estimates = [float(rng.choice(a, size=len(a), replace=True).mean()
                       - rng.choice(b, size=len(b), replace=True).mean()) for _ in range(draws)]
    low, high = np.percentile(estimates, (2.5, 97.5))
    return {"mean": float(a.mean() - b.mean()), "descriptive_95_percent_interval": [float(low), float(high)],
            "draws": draws, "seed": seed, "unit": "state",
            "note": "two-sample bootstrap; sides hold different lineages; descriptive only"}


def side_rows(args, plan, condition):
    """diag-publication-shaped per_state rows per pair/side for one condition."""
    dynamics_plan = train.load_plan(dynamics_args(args))
    objective = base.TaskObjective.from_vocabulary(plan["contract"]["vocabulary"])
    per_side, headroom = {}, []
    for side in plan["sides"]:
        key = f"{side['pair']}:{side['side']}"
        per_state = {}
        evaluable = [s for s in side["samples"] if s["sources"]]
        for seed in SEEDS:
            models = condition_models(args, plan, dynamics_plan, condition, seed)
            rows = {name: {} for name in SYSTEMS}
            for sample in evaluable:
                state = sample["state"]
                targets = read(target_path(args, state))
                diag.check_targets(targets, sample, plan)
                realized = [objective(torch.tensor(t["carriers"][str(TASK_TIME)])) for t in targets["candidates"]]
                flags = [diag.outcome_flags(t["timing"]) for t in targets["candidates"]]
                if seed == SEEDS[0]:
                    candidates = [{"accepted": True, "realized_count_cost": r} for r in realized]
                    regrets, informative = grid.normalized_regrets(candidates)
                    prior = next((i for i, t in enumerate(targets["candidates"])
                                  if t["source"]["ordinal"] == diag.PRIOR_ORDINAL), None)
                    headroom.append({"pair": side["pair"], "side": side["side"],
                                     "state": state["identity"], "family": state["generator_family"],
                                     "candidates": len(realized), "informative": informative,
                                     "prior_ordinal09_regret": None if prior is None else regrets[prior],
                                     "uniform_expected_regret": float(np.mean(regrets)),
                                     "dropped_candidates": len(sample["dropped_candidates"]),
                                     "opportunities": {k: any(f[k] for f in flags) for k in
                                                       ("pig_removed", "pig_contact", "block_contact")}})
                for system, (arm, pair) in SYSTEMS.items():
                    records = [read(curve_path(args, condition, seed, state, system, s["ordinal"]))
                               for s in sample["sources"]]
                    for record, source in zip(records, sample["sources"], strict=True):
                        if record.get("condition") != condition:
                            raise ValueError("curve condition binding differs")
                        diag.check_curve(record, plan, sample, seed, system, source, models[arm])
                    recursive, local = {}, {}
                    for t in TIMES:
                        recursive[str(t)] = diag.average_errors(
                            [diag.field_errors(r["prediction"]["outputs"].get(str(t)),
                                               target["carriers"].get(str(t)), objective)
                             for r, target in zip(records, targets["candidates"], strict=True)])
                        local[str(t)] = diag.average_errors(
                            [diag.field_errors(r["local"]["outputs"].get(str(t)),
                                               target["carriers"].get(str(t)), objective)
                             for r, target in zip(records, targets["candidates"], strict=True)])
                    rows[system][state["identity"]] = {
                        "family": state["generator_family"],
                        "task": diag.task_metrics([r["cost"] for r in records], realized),
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
            per_state[str(seed)] = rows
        per_side[key] = per_state
    return per_side, headroom


def condition_report(args, plan, condition):
    per_side, headroom = side_rows(args, plan, condition)
    summary, contrasts = {}, []
    for key, per_state in per_side.items():
        summary[key] = {str(seed): {name: diag.aggregate_states(list(rows.values()))
                                    for name, rows in systems.items()}
                        for seed, systems in per_state.items()}
        for contrast in diag.summarize_contrasts(per_state):
            contrasts.append({"condition": condition, "side": key, **contrast})
    # Paired few-shot/zero-shot and cross-side effect changes are computed by the caller.
    return {"per_side": summary, "per_side_state": per_side, "contrasts": contrasts,
            "headroom": headroom}


def effect_differences(per_state, tested, reference):
    """Per-state hybrid-minus-continuous style effect: mean over seeds of reference-tested regret."""
    ids = sorted(per_state[str(SEEDS[0])][tested])
    return {s: float(np.mean([per_state[str(seed)][reference][s]["task"]["regret"]
                              - per_state[str(seed)][tested][s]["task"]["regret"]
                              for seed in SEEDS])) for s in ids}


def publication(args, plan):
    inv = artifact_inventory(args, plan)
    complete = (inv["target_states"] == inv["expected_target_states"]
                and inv["adapted_predictors"] == inv["expected_adapted_predictors"]
                and inv["fixed_candidate_records"] == inv["expected_fixed_candidate_records"])
    common = {"schema": "issue_77_n2_eval_report_v1", "identity": "issue-77-n2-eval-report-v1",
              "plan_identity": plan["identity"], "inventory": inv, "evaluation_complete": complete,
              "definitions": plan["definitions"], "adaptation": plan["adaptation"],
              "claim_boundary": "two appearance cells only; appearance is the perception control; "
                                "zero-shot and few-shot reported separately and never mixed",
              "limitations": ["shared semantically supervised CNN, not end-to-end symbol-free comparison",
                              "t=600 is end-of-window cost on mostly right-censored branches, not settled cost",
                              "two independent held-out novel lineages per family; intervals descriptive only",
                              "R3 normal-side lineages retain #75 development model-selection scoring optimism",
                              "typed branch failures excluded from candidates and reported, never worst-cased",
                              "controllers not adapted; fixed systems only",
                              "linear MACs not full FLOPs or matched total work"],
              "archived_release": False, "fresh_evaluation_opened": False,
              "final_evaluation_opened": False, "issue_64_authorized": False}
    if not complete:
        return common
    conditions = {condition: condition_report(args, plan, condition) for condition in CONDITIONS}
    effect_changes, adaptation_changes = [], []
    for pair in ("type010101", "type010102"):
        novel_key, normal_key = f"{pair}:novel", f"{pair}:normal"
        for condition in CONDITIONS:
            per_side = conditions[condition]["per_side_state"]
            for contrast_kind, tested, reference in (
                    ("training_effect_h1", "hybrid_continuous_h1", "continuous_h1"),
                    ("training_effect_h15", "hybrid_continuous_h15", "continuous_h15"),
                    ("symbolic_macro_h15", "hybrid_macro_h15", "hybrid_continuous_h15")):
                novel = effect_differences(per_side[novel_key], tested, reference)
                normal = effect_differences(per_side[normal_key], tested, reference)
                effect_changes.append({
                    "pair": pair, "condition": condition, "tested": tested, "reference": reference,
                    "kind": "cross_side_effect_change", "positive_is_improvement": True,
                    "novel_states": len(novel), "normal_states": len(normal),
                    "novel_effect_mean": float(np.mean(list(novel.values()))),
                    "normal_effect_mean": float(np.mean(list(normal.values()))),
                    "effect_change": two_sample_interval(list(novel.values()), list(normal.values())),
                    "scope": "descriptive two-sample bootstrap; sides hold different lineages"})
        # Few-shot minus zero-shot on the novel side, paired per state.
        per_side_zero = conditions["zero-shot"]["per_side_state"][novel_key]
        per_side_few = conditions["few-shot"]["per_side_state"][novel_key]
        for system in SYSTEMS:
            ids = sorted(per_side_zero[str(SEEDS[0])][system])
            differences = []
            for s in ids:
                across = [per_side_few[str(seed)][system][s]["task"]["regret"]
                          - per_side_zero[str(seed)][system][s]["task"]["regret"]
                          for seed in SEEDS]
                differences.append(float(np.mean(across)))
            adaptation_changes.append({
                "pair": pair, "side": "novel", "system": system,
                "kind": "few_shot_minus_zero_shot_regret", "positive_is_improvement": False,
                "paired_states": len(ids),
                "regret_change": grid.paired_interval(differences),
                "scope": "paired per state, seeds averaged within state; negative favors adaptation"})
    headroom = conditions["zero-shot"]["headroom"]
    return {**common,
            "conditions": {c: {"per_side": v["per_side"], "contrasts": v["contrasts"]}
                           for c, v in conditions.items()},
            "headroom": headroom,
            "cross_side_effect_changes": effect_changes,
            "adaptation_changes": adaptation_changes,
            "independent_state_counts": {f"{s['pair']}:{s['side']}": sum(1 for x in s["samples"] if x["sources"])
                                         for s in plan["sides"]}}


def compact_report(result):
    return {k: v for k, v in result.items() if k != "per_side_state"}


def comparisons_csv(result):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(("section", "condition", "side_or_pair", "tested", "reference", "kind",
                     "mean_difference", "interval_low", "interval_high", "states"))
    for condition, payload in result.get("conditions", {}).items():
        for contrast in payload["contrasts"]:
            interval = contrast["regret"]["descriptive_95_percent_interval"]
            writer.writerow(("within_side", condition, contrast["side"], contrast["tested"],
                             contrast["reference"], contrast["kind"], contrast["regret"]["mean"],
                             interval[0], interval[1], contrast["regret"].get("paired_states")))
    for change in result.get("cross_side_effect_changes", []):
        interval = change["effect_change"]["descriptive_95_percent_interval"]
        writer.writerow(("cross_side_effect_change", change["condition"], change["pair"],
                         change["tested"], change["reference"], change["kind"],
                         change["effect_change"]["mean"], interval[0], interval[1],
                         f"novel={change['novel_states']},normal={change['normal_states']}"))
    for change in result.get("adaptation_changes", []):
        interval = change["regret_change"]["descriptive_95_percent_interval"]
        writer.writerow(("adaptation", "few-shot_minus_zero-shot", change["pair"], change["system"],
                         "zero-shot", change["kind"], change["regret_change"]["mean"],
                         interval[0], interval[1], change["paired_states"]))
    return stream.getvalue()


def _interval_text(value):
    if not value or "mean" not in value:
        return "unavailable"
    low, high = value["descriptive_95_percent_interval"]
    return f"{value['mean']:+.4f} [{low:+.4f}, {high:+.4f}]"


def findings_md(result):
    lines = ["# Issue-77 N2 appearance evaluation - findings", ""]
    lines.append(f"Evaluation complete: {result['evaluation_complete']}. "
                 f"Claim boundary: {result['claim_boundary']}.")
    lines.append("")
    if not result.get("conditions"):
        lines.append("Evaluation incomplete; no outcome numbers reported.")
        return "\n".join(lines) + "\n"
    lines.append("## Zero-shot (frozen N1 checkpoints; no novelty fitting)")
    lines.append("")
    lines.append("| Pair | Side | System | Seed-mean regret (per seed) |")
    lines.append("| --- | --- | --- | --- |")
    for key, seeds in result["conditions"]["zero-shot"]["per_side"].items():
        pair, side = key.split(":")
        for name in SYSTEMS:
            values = "; ".join(f"{seed}={systems[name]['mean_regret']:.4f}" for seed, systems in seeds.items())
            lines.append(f"| {pair} | {side} | {name} | {values} |")
    lines.append("")
    lines.append("## Few-shot (adapted on novel predictor lineages; budgets in summary.json)")
    lines.append("")
    lines.append("| Pair | Side | System | Seed-mean regret (per seed) |")
    lines.append("| --- | --- | --- | --- |")
    for key, seeds in result["conditions"]["few-shot"]["per_side"].items():
        pair, side = key.split(":")
        for name in SYSTEMS:
            values = "; ".join(f"{seed}={systems[name]['mean_regret']:.4f}" for seed, systems in seeds.items())
            lines.append(f"| {pair} | {side} | {name} | {values} |")
    lines.append("")
    lines.append("## Cross-side change in the hybrid-minus-continuous effect")
    lines.append("")
    lines.append("Positive effect = reference regret minus tested regret (better hybrid). "
                 "Change = novel-side effect minus normal-side effect; negative means the hybrid "
                 "advantage shrinks on the novelty. Two-sample bootstrap over each side's "
                 "independent states; descriptive only.")
    lines.append("")
    lines.append("| Pair | Condition | Tested | Reference | Novel effect | Normal effect | Change [95%] |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for change in result["cross_side_effect_changes"]:
        lines.append(f"| {change['pair']} | {change['condition']} | {change['tested']} | {change['reference']} | "
                     f"{change['novel_effect_mean']:+.4f} | {change['normal_effect_mean']:+.4f} | "
                     f"{_interval_text(change['effect_change'])} |")
    lines.append("")
    lines.append("## Adaptation change on the novel side (few-shot minus zero-shot)")
    lines.append("")
    lines.append("| Pair | System | Regret change [95%] (negative favors adaptation) |")
    lines.append("| --- | --- | --- |")
    for change in result["adaptation_changes"]:
        lines.append(f"| {change['pair']} | {change['system']} | {_interval_text(change['regret_change'])} |")
    lines.append("")
    lines.append("## Headroom (zero-shot condition targets)")
    lines.append("")
    lines.append("| Pair | Side | State | Candidates | Informative | Prior(ordinal09) regret | Uniform regret | Dropped |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in result["headroom"]:
        prior = "n/a" if row["prior_ordinal09_regret"] is None else f"{row['prior_ordinal09_regret']:.4f}"
        lines.append(f"| {row['pair']} | {row['side']} | {row['state']} | {row['candidates']} | "
                     f"{row['informative']} | {prior} | {row['uniform_expected_regret']:.4f} | "
                     f"{row['dropped_candidates']} |")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for item in result["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- smoke

def smoke(args):
    plan = make_plan(args)
    began = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="novphy-issue77-n2-eval-smoke-") as directory:
        small = copy.copy(args)
        small.output = Path(directory)
        plan = copy.deepcopy(plan)
        plan["identity"] += ":smoke"
        plan["adaptation"]["steps"] = 8
        for side in plan["sides"]:
            chosen = [s for s in side["samples"] if s["sources"]][:1]
            if chosen:
                chosen[0]["sources"] = chosen[0]["sources"][:2]
            side["samples"] = chosen
        dynamics_plan = train.load_plan(dynamics_args(args))
        prepare_targets(small, plan)
        seed = SEEDS[0]
        for arm in ARMS:
            adapt_cell(small, plan, seed, ADAPT_SEEDS[0], arm, stop_after=4)
            adapt_cell(small, plan, seed, ADAPT_SEEDS[0], arm, stop_after=8)
        saved = torch.load(small.output / "adapt" / f"seed-{seed}" / "continuous" / "adapted-progress.pt",
                           weights_only=True, map_location="cpu")
        if saved["step"] != 8:
            raise ValueError("adaptation resume did not reach the smoke budget")
        records = 0
        for condition in CONDITIONS:
            models = condition_models(small, plan, dynamics_plan, condition, seed)
            for side in plan["sides"]:
                for sample in side["samples"]:
                    if not sample["sources"]:
                        continue
                    targets = read(target_path(small, sample["state"]))
                    for system in ("continuous_h15", "hybrid_macro_h15"):
                        arm, _ = SYSTEMS[system]
                        for source in sample["sources"]:
                            record = stored_curve(small, plan, sample, targets, condition, seed,
                                                  system, source, models[arm])
                            records += 1
        result = {"schema": "issue_77_n2_eval_smoke_v1", "production_evidence": False,
                  "records": records, "adaptation_resume_step": saved["step"],
                  "wall_seconds": time.monotonic() - began, "memory": base.memory(args.device)}
    write(args.output / "smoke.json", result)
    log(f"real smoke complete records={records} wall={result['wall_seconds']:.1f}s; no production evaluation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "smoke-test", "prepare", "build-shards", "adapt",
                 "run-evaluation", "publish", "validate"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dynamics", type=Path, default=DYNAMICS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        if args.dry_run:
            plan = make_plan(args)
            evaluable = evaluable_samples(plan)
            log(f"no-write dry-run: sides={len(plan['sides'])} evaluable_states={len(evaluable)} "
                f"candidates={sum(len(s['sources']) for s in evaluable)} systems={len(SYSTEMS)} "
                f"conditions=2 seeds=3 adaptation_lineages={len(plan['adaptation']['lineages'])}")
            return 0
        if args.smoke_test:
            smoke(args)
            return 0
        if args.prepare:
            if (args.output / "plan.json").exists():
                load_plan(args)
            else:
                write(args.output / "plan.json", make_plan(args))
            log("N2 appearance evaluation protocol frozen; no images scored")
            return 0
        plan = load_plan(args)
        if args.build_shards:
            build_novel_shards(args, plan)
            log(f"adaptation shards complete memory={base.memory(args.device)}")
        elif args.adapt:
            if not (args.output / "n2-shards.json").exists():
                raise ValueError("run --build-shards before --adapt")
            for seed in SEEDS:
                for arm in ARMS:
                    log(f"adaptation cell seed={seed} arm={arm} start")
                    adapt_cell(args, plan, seed, ADAPT_SEEDS[SEEDS.index(seed)], arm)
            log(f"six adapted predictors complete memory={base.memory(args.device)}")
        elif args.run_evaluation:
            run_evaluation(args, plan)
        elif args.publish:
            result = publication(args, plan)
            write(args.output / "summary.json", compact_report(result))
            (args.output / "comparisons.csv").write_text(comparisons_csv(result))
            (args.output / "findings.md").write_text(findings_md(result))
            log(f"published evaluation_complete={result['evaluation_complete']}")
        else:
            result = publication(args, plan)
            if (compact_report(result) != read(args.output / "summary.json")
                    or comparisons_csv(result).encode("utf-8") != (args.output / "comparisons.csv").read_bytes()
                    or findings_md(result) != (args.output / "findings.md").read_text()):
                raise ValueError("published N2 evaluation tables differ from bound source evidence")
            log("exact saved-evidence validation passed")
        return 0
    except (ValueError, OSError) as error:
        log(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
