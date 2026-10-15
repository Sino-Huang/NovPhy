"""Issue-102: closed-loop evaluation - chosen launches executed live in the #104 player.

Binding runner module: scripts/run_closed_loop_evaluation.py
Exact validation command: python -u -m scripts.run_closed_loop_evaluation --validate

Every #109 / #110 level holds a single bird, so a level ends after one shot. The user chose
**retry-from-start with live engine execution** (binding):
- shot k resets the same level to its start state;
- before each shot the planner ranks the frozen grid inventory (16 launches) from the level's decision
  frame with its model or rule, excludes the candidates already executed and failed in this episode, and
  picks the best remaining one;
- the chosen launch is executed live in the #104 player (``capture_pipeline_v2.run_campaign``); recorded
  #104 outcomes are never the result (they are a declared cross-check and the oracle's inventory);
- an episode ends at the first pass (engine pig removal, the verdict every ranking ticket uses) or when
  the shot budget (3) is spent.
The state never changes between shots, so "replan" amounts to excluding failed candidates; the
planner re-ranks at every shot anyway, so per-decision wall time and MACs are measured per shot.

Episode sharing (pre-declared): execution is deterministic (#104 160/160 replay), so one live execution
per distinct level x candidate chosen by any arm, seed or shot is shared by every episode choosing it.

Arms: the nine fixed requests on the Gate-B hybrid LC predictor (F-15-macro = Gate-B unit), C* (C-F-15),
horizon-only adaptive C-J and ICLR J at the ICLR lambda 1e-4 (#114 / #101), decision-aware J_dec (#101
selection, lambda 1e-1), the no-model action prior (Gate-B rule), uniform random over the remaining
candidates and the oracle ceiling (recorded-inventory passes, confirmed live for the executed ones).

Sets: N1 = the 400 #109 evaluation levels (#104 stream issue109-evaluation; the 311 levels with a
recorded passing grid candidate are a declared subset); N2-appearance = the 40 #110 evaluation
appearance-novelty levels (novelty level 1, novel side), reported separately.

Modes: --dry-run --smoke --prepare --anchors --decide --execute --gallery --publish --validate --status.
--decide takes --shot K --workers N --worker W (one process per GPU, CUDA_VISIBLE_DEVICES set by the
caller); --execute takes --shot K (16 isolated engine workers, the #104 limits).
Intervals: level-clustered percentile bootstrap over levels x seeds (10000 draws, PCG64 7201), DESCRIPTIVE.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import functools
import hashlib
import html
import io
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from scripts import capture_pipeline_v2 as pipeline
from scripts import prepare_power_cohort as cohort
from scripts import run_decision_aware_policy as dap
from scripts import run_decision_dynamics_controls as c2
from scripts import run_decision_dynamics_retrain as rt
from scripts import run_decision_relevant_dynamics as v1
from scripts import run_engine_truth_frontier as etf
from scripts import run_gate_b as gb
from scripts import run_launch_power_probe as p94
from scripts import run_power_cohort_campaign as rpc
from scripts import run_slot_encoder_fix as s99
from scripts import run_tau_ad_within_checkpoint as t96
from world_model.planning import scoring_harness as harness

ROOT = v1.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-102-closed-loop-v1"
GALLERY = ROOT / "data/issue-102-closed-loop"
CAMPAIGN = rt.CAMPAIGN

IDENTITY = "issue-102-closed-loop-v1"
SCHEMA_PLAN = "issue_102_closed_loop_plan_v1"
SCHEMA_SMOKE = "issue_102_closed_loop_smoke_v1"
SCHEMA_DECISION = "issue_102_decision_record_v1"
SCHEMA_VERDICT = "issue_102_live_verdict_v1"
SCHEMA_ANCHORS = "issue_102_n2_anchors_v1"
SCHEMA_COMPUTE = "issue_102_compute_v1"
SCHEMA_REPORT = "issue_102_report_v1"
SCHEMA_MANIFEST = "issue_102_record_manifest_v1"
SCHEMA_GALLERY = "issue_102_episode_gallery_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_closed_loop_evaluation --validate"
RUNNER = "scripts/run_closed_loop_evaluation.py"

SEEDS = tuple(dap.SEEDS)
SHOT_BUDGET = 3
INVENTORY = gb.GATE_INVENTORY  # grid: the 16 launches every level shares
PRIMARY_COST = harness.PRIMARY_COST
PASS_FIELD = "pig_removed"  # #85 engine channel, the verdict of every #104 / Gate-B ranking cell
DELTA = v1.DELTA
STOP_TOKEN = v1.STOP_TOKEN
TOLERANCE = harness.PARSE_TOLERANCE
N1, N1_MIXED, N2 = "N1", "N1-mixed", "N2-appearance"
SETS = (N1, N2)
TABLE_SETS = (N1, N1_MIXED, N2)
N1_STREAM = gb.STREAM
N2_STREAM = "issue110-evaluation"
N2_NOVELTY = 1  # #110 contract: novelty level 1 = appearance

UNIT = "F-15-macro"  # Gate-B unit
C_STAR = dap.C_STAR
CJ_ICLR = etf.ICLR_CJ  # CJ-1e-04
J_ICLR = etf.ICLR_J  # J-1e-04
J_DEC = "J-dec-1e-01"  # the #101 selected decision-aware arm
MODEL_ARMS = (*t96.FIXED_ARMS, C_STAR, CJ_ICLR, J_ICLR, J_DEC)
PRIOR, RANDOM, ORACLE = "prior", "random", "oracle"
RULE_ARMS = (PRIOR, RANDOM, ORACLE)
ARMS = (*MODEL_ARMS, *RULE_ARMS)
SEED_FREE = (PRIOR, ORACLE)
ADAPTIVE = (CJ_ICLR, J_ICLR, J_DEC)
GROUPS = {**{arm: "fixed request (hybrid LC)" for arm in t96.FIXED_ARMS}, C_STAR: "C* fixed (15, continuous)",
          CJ_ICLR: "horizon-only adaptive C-J (lambda 1e-4)", J_ICLR: "ICLR J (lambda 1e-4)",
          J_DEC: "decision-aware J_dec (lambda 1e-1)", PRIOR: "no-model action prior",
          RANDOM: "uniform random", ORACLE: "oracle ceiling"}
RANDOM_KEY = 102  # random arm stream: PCG64([RANDOM_KEY, seed, level index])
CONTRAST_PAIRS = ((J_ICLR, CJ_ICLR), (J_DEC, J_ICLR), (UNIT, C_STAR), (PRIOR, RANDOM))

GPU_CAP_SECONDS = 4 * 3600.0
ENGINE_WALL_CAP_SECONDS = 48 * 3600.0  # summed run_campaign wall on 16 workers
ARTIFACT_CAP_BYTES = 1_500_000_000_000
AMORTIZED_BRANCH_SECONDS = 11.95  # #104 amortized branch wall at 16 workers (projection only)
WEBM_WORKERS = 12

SMOKE_SEED = SEEDS[0]
SMOKE_MODEL_ARMS = (UNIT, CJ_ICLR, J_ICLR, J_DEC)
SMOKE_CELLS = 2  # N1 mixed grid cells re-scored per model arm against the retained source rows
SMOKE_N1_STREAM = "issue109-fit"
SMOKE_N2_STREAM = "issue110-fit"

read_json = v1.read_json
write_json = v1.write_json
json_text = v1.json_text
sha256_of = v1.sha256_of
atomic_torch = v1.atomic_torch
fmt_interval = v1.fmt_interval


def log(message):
    print(f"[issue-102] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seed_tag(seed):
    return "seedfree" if seed is None else f"seed{seed}"


def arm_seeds(arm, seeds=SEEDS):
    return (None,) if arm in SEED_FREE else tuple(seeds)


def execution_id(level, suffix):
    return f"issue-102-{level}-{suffix}"


# ---------------------------------------------------------------------------
# levels, inventories, recorded #104 rows (oracle inventory and cross-check only)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def campaign_rows():
    return {row["identity"]: row for row in read_json(CAMPAIGN / "branches.json")}


POWER_PLAN_PATH_REWRITES = (".local-artifacts/issue-96-tau-ad-within-checkpoint-v1/plan.json",
                            ".local-artifacts/issue-104-capture-dev/summary.json")


@functools.lru_cache(maxsize=None)
def power_plan():
    """The frozen #104 plan. ``cohort.load_plan`` re-hashes its inputs; exactly two of them differ from the
    #104-frozen bytes, and only by machine-local home-directory path rewrites (disclosed in plan.json); any
    other changed input refuses."""
    plan = cohort.read(cohort.OUTPUT / "plan.json")
    changed = [item["artifact"] for item in plan["inputs"]
               if cohort.file_sha256(ROOT / item["artifact"]) != item["sha256"]]
    if set(changed) - set(POWER_PLAN_PATH_REWRITES):
        raise ValueError(f"#104 inputs changed beyond the disclosed path rewrites: {changed}")
    return plan


@functools.lru_cache(maxsize=None)
def cohort_plans():
    return cohort.read(cohort.COHORT_109 / "plan.json"), cohort.read(cohort.COHORT_110 / "plan.json")


@functools.lru_cache(maxsize=None)
def dispatch_index():
    """level identity -> (stream, level dict, base cluster) of the frozen #104 campaign rows."""
    p109, p110 = cohort_plans()
    return {level["identity"]: (stream, level, cluster)
            for stream, level, cluster, _ in cohort.stream_levels(p109, p110, power_plan()["depth"])}


def level_entry(set_name, stream, items):
    rows = campaign_rows()
    recorded = {}
    for item in items:
        row = rows[item["key"]]
        recorded[item["ordinal"]] = {"identity": row["identity"], "status": row["status"],
                                     "pig_removed": row.get("pig_removed"), "stop_kind": row.get("stop_kind")}
    verdicts = {o: bool(r["pig_removed"]) for o, r in recorded.items() if r["status"] == "complete"}
    return {"set": set_name, "stream": stream, "inventory": items, "verdicts": verdicts, "recorded": recorded,
            "recorded_any_pass": any(verdicts.values())}


def n110_items(level):
    side = level["slingshot_side"]
    items = []
    for candidate in cohort_plans()[1]["candidates"]["by_slingshot_side"][side]:
        entry = next(e for e in candidate["inventory_entries"] if e["inventory"] == INVENTORY)
        items.append({"ordinal": entry["ordinal"], "action": dict(candidate["action"]),
                      "key": f"{level['identity']}-{candidate['suffix']}", "suffix": candidate["suffix"]})
    return sorted(items, key=lambda item: item["ordinal"])


def n110_levels(stream):
    return sorted(identity for identity, (s, level, _) in dispatch_index().items()
                  if s == stream and level["novelty_level"] == N2_NOVELTY and "-novel-" in identity)


@functools.lru_cache(maxsize=None)
def levels():
    """Frozen level set: N1 (400 #109 evaluation levels) and N2-appearance (40 #110 evaluation levels)."""
    out = {}
    _, cells = gb.evaluation_set()
    for member, entry in sorted(cells[INVENTORY]["all"].items()):
        items = [{"ordinal": i["ordinal"], "action": dict(i["action"]), "key": i["key"],
                  "suffix": i["key"].rsplit("-", 1)[1]} for i in entry["inventory"]]
        out[member] = level_entry(N1, N1_STREAM, items)
    for member in n110_levels(N2_STREAM):
        out[member] = level_entry(N2, N2_STREAM, n110_items(dispatch_index()[member][1]))
    return out


def set_members(level_map, set_name):
    if set_name == N1_MIXED:
        return [m for m, e in level_map.items() if e["set"] == N1 and e["recorded_any_pass"]]
    return [m for m, e in level_map.items() if e["set"] == set_name]


@functools.lru_cache(maxsize=None)
def smoke_levels():
    """Pre-freeze smoke levels (no evaluation level): the first #109 fit level of each family with a
    recorded passing and failing grid candidate, and the first #110 fit appearance level likewise."""
    fit = read_json(gb.FIT_VERDICTS)["branches"]
    by_level = {}
    for branch in fit:
        entry = next((e for e in branch["inventory_entries"] if e["inventory"] == INVENTORY), None)
        if entry is not None:
            by_level.setdefault(branch["level"], []).append(
                {"ordinal": entry["ordinal"], "action": dict(branch["action"]), "key": branch["key"],
                 "suffix": branch["suffix"]})
    out, families = {}, set()
    for level in sorted(by_level):
        stream, record, _ = dispatch_index()[level]
        if stream != SMOKE_N1_STREAM or record["generator_family"] in families:
            continue
        entry = level_entry(N1, stream, sorted(by_level[level], key=lambda item: item["ordinal"]))
        if len(entry["inventory"]) == 16 and t96.mixed(entry):
            out[level] = entry
            families.add(record["generator_family"])
    for level in n110_levels(SMOKE_N2_STREAM):
        entry = level_entry(N2, SMOKE_N2_STREAM, n110_items(dispatch_index()[level][1]))
        if t96.mixed(entry):
            out[level] = entry
            break
    return out


# ---------------------------------------------------------------------------
# anchors (the decision frame the planner ranks from)
# ---------------------------------------------------------------------------

def first_attempt(entry):
    """The level's first complete recorded candidate in suffix order (the Gate-B anchor rule)."""
    for item in sorted(entry["inventory"], key=lambda item: item["suffix"]):
        if entry["recorded"][item["ordinal"]]["status"] == "complete":
            return CAMPAIGN / entry["stream"] / "attempts" / item["key"]
    raise ValueError("level has no complete recorded candidate")


def n2_anchor_path(output):
    return Path(output) / "data" / "anchors-n2.pt"


def parse_anchors(path, members, entries):
    began = time.monotonic()
    with harness.deterministic_scoring() as policy:
        adapter = s99.load_adapter(s99.OUTPUT, "E")
        frames = {m: str(first_attempt(entries[m])) for m in members}
        anchors = {m: rt.anchor_carrier(adapter, frames[m]) for m in members}
    atomic_torch(path, {"schema": SCHEMA_ANCHORS, "plan_identity": IDENTITY, "scoring_policy": policy,
                        "parsed_at": utc_now(), "anchors": anchors,
                        "frames": {m: Path(f).name for m, f in frames.items()}})
    return time.monotonic() - began


def load_n2_anchors(path):
    store = torch.load(path, map_location="cpu", weights_only=True)
    if store.get("schema") != SCHEMA_ANCHORS or store.get("plan_identity") != IDENTITY:
        raise ValueError("N2 anchor store binding differs")
    return store


# ---------------------------------------------------------------------------
# run contexts (main campaign and pre-freeze smoke share every function)
# ---------------------------------------------------------------------------

def main_context(output=OUTPUT):
    output = Path(output)
    model = [(arm, seed) for arm in MODEL_ARMS for seed in SEEDS]
    rule = [(arm, seed) for arm in RULE_ARMS for seed in arm_seeds(arm)]
    return {"name": "main", "output": output, "engine": output / "engine", "gallery": GALLERY,
            "levels": levels(), "model_jobs": model, "rule_jobs": rule,
            "anchors": lambda: {**gb.load_anchors(gb.OUTPUT)["anchors"],
                                **load_n2_anchors(n2_anchor_path(output))["anchors"]}}


def smoke_context(output=OUTPUT):
    root = Path(output) / "smoke"
    model = [(arm, SMOKE_SEED) for arm in SMOKE_MODEL_ARMS]
    rule = [(PRIOR, None), (RANDOM, SMOKE_SEED), (ORACLE, None)]
    fit = gb.fit_anchors()

    def anchors():
        store = load_n2_anchors(n2_anchor_path(root))["anchors"]
        return {m: (fit[m] if m in fit else store[m]) for m in smoke_levels()}
    return {"name": "smoke", "output": root, "engine": root / "engine", "gallery": GALLERY / "smoke",
            "levels": smoke_levels(), "model_jobs": model, "rule_jobs": rule, "anchors": anchors}


def jobs(ctx):
    return ctx["model_jobs"] + ctx["rule_jobs"]


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------

def decision_path(output, shot, arm, seed):
    return Path(output) / "records" / "decisions" / f"shot{shot}" / f"{arm}--{seed_tag(seed)}.json"


def verdict_path(output, execution):
    return Path(output) / "records" / "verdicts" / f"{execution}.json"


def load_decisions(ctx):
    out = {}
    for shot in range(1, SHOT_BUDGET + 1):
        for arm, seed in jobs(ctx):
            path = decision_path(ctx["output"], shot, arm, seed)
            if path.is_file():
                record = read_json(path)
                if (record.get("schema") != SCHEMA_DECISION or record.get("plan_identity") != IDENTITY
                        or (record["arm"], record["seed"], record["shot"]) != (arm, seed, shot)):
                    raise ValueError(f"decision record {path.name} binding differs")
                out[(shot, arm, seed)] = record
    return out


def load_verdicts(output):
    root = Path(output) / "records" / "verdicts"
    out = {}
    for path in sorted(root.glob("*.json")) if root.is_dir() else ():
        record = read_json(path)
        if record.get("schema") != SCHEMA_VERDICT or record.get("plan_identity") != IDENTITY:
            raise ValueError(f"verdict record {path.name} binding differs")
        out[record["execution"]] = record
    return out


def passed(verdict):
    return verdict["status"] == "complete" and bool(verdict[PASS_FIELD])


def open_levels(ctx, arm, seed, shot, decisions, verdicts):
    """Levels whose episode is still running at ``shot``: every earlier shot was executed, completed
    and failed. Value: the failed (excluded) ordinals in shot order."""
    out = {}
    for level in ctx["levels"]:
        excluded, running = [], True
        for earlier in range(1, shot):
            choice = decisions[(earlier, arm, seed)]["levels"].get(level)
            if choice is None:
                running = False
                break
            verdict = verdicts[choice["execution"]]
            if verdict["status"] != "complete" or verdict[PASS_FIELD]:
                running = False
                break
            excluded.append(choice["chosen"])
        if running:
            out[level] = excluded
    return out


def model_ranking(rows):
    """Lowest tie-free cost first (ties to the lower ordinal); nonfinite (excluded) endpoints last."""
    admissible = sorted((r[PRIMARY_COST], r["ordinal"]) for r in rows
                        if not r["excluded"] and r[PRIMARY_COST] is not None)
    rest = sorted(r["ordinal"] for r in rows if r["excluded"] or r[PRIMARY_COST] is None)
    return [o for _, o in admissible] + rest


def choose(ranking, excluded):
    return next((o for o in ranking if o not in excluded), None)


def random_ranking(seed, level_index, ordinals):
    """Uniform random order of the inventory (one permutation per seed and level); taking the first
    non-excluded entry is uniform sampling over the remaining candidates without replacement."""
    generator = np.random.Generator(np.random.PCG64([RANDOM_KEY, int(seed), int(level_index)]))
    return [int(o) for o in generator.permutation(np.array(sorted(ordinals)))]


def oracle_ranking(entry):
    """Recorded-inventory passes (#104), lowest ordinal first; empty when no recorded candidate passes."""
    return sorted(o for o, v in entry["verdicts"].items() if v)


def request_names(arm):
    policy = arm["policy"]
    if policy is None:
        return None
    return [f"{p.delta}-{p.abstraction.value}" for p in policy.pairs]


def fixed_request(pid):
    return "15-continuous" if pid == C_STAR else pid[2:]


def request_counts(names, text):
    counts = {}
    for index in t96.unrle(text):
        counts[names[index]] = counts.get(names[index], 0) + 1
    return dict(sorted(counts.items()))


def item_of(entry, ordinal):
    return next(item for item in entry["inventory"] if item["ordinal"] == ordinal)


def choice_block(level, entry, excluded, ranking, chosen):
    item = item_of(entry, chosen)
    return {"excluded": list(excluded), "ranking": ranking, "chosen": chosen, "suffix": item["suffix"],
            "execution": execution_id(level, item["suffix"])}


def model_decisions(ctx, pid, seed, shot, open_, policy_state):
    entry_ = etf.entry_of(pid)
    began = time.monotonic()
    arm = etf.load_arm(entry_, seed, None)
    anchors = {"E99": ctx["anchors"]()}
    costs = dap.endpoint_costs()
    names = request_names(arm)
    base = {"schema": SCHEMA_DECISION, "plan_identity": IDENTITY, "context": ctx["name"], "arm": pid,
            "group": GROUPS[pid], "seed": seed, "shot": shot, "scoring_policy": policy_state,
            "decided_at": utc_now(), "binding": arm["binding"], "source": entry_["source"],
            "source_arm": entry_["arm"], "request_names": names,
            "fixed_request": None if names else fixed_request(pid), "device": torch.cuda.get_device_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "engine_seconds": 0}
    members = sorted(open_)
    out = {}
    if members:
        warm = members[0]
        etf.run_cell(arm, INVENTORY, warm, ctx["levels"][warm], anchors, costs, seed)  # warm-up, discarded
    for level in members:
        entry = ctx["levels"][level]
        excluded = open_[level]
        start = time.monotonic()
        block = etf.run_cell(arm, INVENTORY, level, entry, anchors, costs, seed)
        ranking = model_ranking(block["rows"])
        chosen = choose(ranking, excluded)
        seconds = time.monotonic() - start
        decision = {**choice_block(level, entry, excluded, ranking, chosen),
                    "rows_sha256": digest(block["rows"]), "rollout_seconds": block["gpu_seconds"],
                    "decision_seconds": seconds, "linear_macs": block["linear_macs"],
                    "transition_calls": block["transition_calls"], "controller_calls": block["controller_calls"],
                    "candidates": block["candidates"]}
        if shot == 1:
            decision["rows"] = block["rows"]
        if names:
            schedules = block["schedules"]
            decision["chosen_schedule"] = schedules[str(chosen)]
            decision["requests_chosen"] = request_counts(names, schedules[str(chosen)])
            total = {}
            for text in schedules.values():
                for name, count in request_counts(names, text).items():
                    total[name] = total.get(name, 0) + count
            decision["requests_all"] = dict(sorted(total.items()))
        out[level] = decision
    torch.cuda.synchronize()
    return {**base, "warmup": members[0] if members else None, "levels": out, "no_candidate": [],
            "seconds": time.monotonic() - began}


@functools.lru_cache(maxsize=None)
def prior_stats():
    return gb.prior_stats(read_json(gb.FIT_VERDICTS)["branches"], INVENTORY)


def rule_decisions(ctx, pid, seed, shot, open_, policy_state):
    began = time.monotonic()
    order = list(ctx["levels"])
    out, none = {}, []
    for level in sorted(open_):
        entry = ctx["levels"][level]
        excluded = open_[level]
        start = time.monotonic()
        extra = {}
        if pid == PRIOR:
            rows = gb.prior_rows(entry, prior_stats())
            ranking = model_ranking(rows)
            extra = {"rows_sha256": digest(rows)}
            if shot == 1:
                extra["rows"] = rows
        elif pid == RANDOM:
            ranking = random_ranking(seed, order.index(level), [i["ordinal"] for i in entry["inventory"]])
        else:
            ranking = oracle_ranking(entry)
        chosen = choose(ranking, excluded)
        seconds = time.monotonic() - start
        if chosen is None:
            none.append(level)
            continue
        out[level] = {**choice_block(level, entry, excluded, ranking, chosen), **extra, "rollout_seconds": 0.0,
                      "decision_seconds": seconds, "linear_macs": 0, "transition_calls": 0, "controller_calls": 0,
                      "candidates": len(entry["inventory"])}
    return {"schema": SCHEMA_DECISION, "plan_identity": IDENTITY, "context": ctx["name"], "arm": pid,
            "group": GROUPS[pid], "seed": seed, "shot": shot, "scoring_policy": policy_state,
            "decided_at": utc_now(), "binding": {}, "source": "rule", "source_arm": pid, "request_names": None,
            "fixed_request": None, "engine_seconds": 0, "levels": out, "no_candidate": none,
            "seconds": time.monotonic() - began}


def check_previous_shot(ctx, shot, decisions, verdicts):
    if shot < 1 or shot > SHOT_BUDGET:
        raise ValueError(f"shot {shot} outside the budget 1..{SHOT_BUDGET}")
    for earlier in range(1, shot):
        for arm, seed in jobs(ctx):
            record = decisions.get((earlier, arm, seed))
            if record is None:
                raise ValueError(f"shot {earlier} decision {arm}/{seed_tag(seed)} missing")
            missing = [c["execution"] for c in record["levels"].values() if c["execution"] not in verdicts]
            if missing:
                raise ValueError(f"shot {earlier}: {len(missing)} chosen executions have no live verdict yet")


def decide_jobs(ctx, shot, selected, worker, ledger_output):
    decisions, verdicts = load_decisions(ctx), load_verdicts(ctx["output"])
    check_previous_shot(ctx, shot, decisions, verdicts)
    for pid, seed in selected:
        path = decision_path(ctx["output"], shot, pid, seed)
        if path.is_file():
            continue
        totals = ledger_totals(ledger_output)
        if totals["gpu_seconds"] > GPU_CAP_SECONDS:
            log(f"worker {worker}: GPU cap reached; stopping ({totals})")
            return 1
        open_ = open_levels(ctx, pid, seed, shot, decisions, verdicts)
        began = time.monotonic()
        with harness.deterministic_scoring() as policy_state:
            if pid in MODEL_ARMS:
                record = model_decisions(ctx, pid, seed, shot, open_, policy_state)
            else:
                record = rule_decisions(ctx, pid, seed, shot, open_, policy_state)
        write_json(path, record)
        if pid in MODEL_ARMS:
            torch.cuda.empty_cache()
        wall = time.monotonic() - began
        mark(ledger_output, f"{ctx['name']}-decide-shot{shot}-{pid}-{seed_tag(seed)}", wall,
             wall if pid in MODEL_ARMS else 0.0, worker=worker, decisions=len(record["levels"]))
        log(f"worker {worker}: shot {shot} {pid} {seed_tag(seed)}: {len(record['levels'])} decisions ({wall:.0f}s)")
    return 0


# ---------------------------------------------------------------------------
# live execution (#104 player, capture_pipeline_v2) and verdicts
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def templates():
    return cohort.templates_109(), cohort.templates_110()


def dispatch_record(level, item):
    stream, record_level, cluster = dispatch_index()[level]
    by_109, by_110 = cohort.actions_by_suffix(*cohort_plans())
    t109, t110 = templates()
    if stream.startswith("issue110"):
        candidate = by_110[record_level["slingshot_side"]][item["suffix"]]
        template = t110[record_level["template"]]
    else:
        candidate = by_109[item["suffix"]]
        template = t109[record_level["generator_family"]]
    if candidate["action"] != item["action"]:
        raise ValueError(f"{level} {item['suffix']}: dispatch action differs from the planner's candidate")
    record = cohort.dispatch_record(stream, record_level, cluster, candidate, template)
    record["identity"] = execution_id(level, item["suffix"])
    return record


def pin_directory(directory):
    """Create ``directory`` on one mergerfs branch (most free bytes) so player hard links stay intra-branch
    (#104 pin_stream); returns the branch, or None off mergerfs."""
    directory = Path(os.path.abspath(directory))
    found = rpc.mergerfs_branches(ROOT)
    if found is None:
        directory.mkdir(parents=True, exist_ok=True)
        return None
    mount, branches = found
    relative = directory.relative_to(mount)
    existing = [b for b in branches if (b / relative).is_dir()]
    if existing:
        return str(existing[0])
    branch = max(branches, key=lambda b: (shutil.disk_usage(b).free, str(b)))
    (branch / relative).mkdir(parents=True, exist_ok=True)
    if not directory.is_dir():
        raise RuntimeError(f"{directory}: pinned branch directory is not visible through the mount")
    return str(branch)


def engine_preflight():
    """#104 bindings the live executions depend on: the capture pipeline and the player assembly must equal the
    frozen #104 bytes. The Mesa libraries were rebuilt on this host from the same Mesa 26.1.2 / LLVM 22.1.6
    sources (embedded install paths differ), so their hashes are recorded and the differing files disclosed;
    capture_pipeline_v2 checks the renderer string of every execution, and every verdict record compares the
    live decision and final frames with the recorded #104 frames byte for byte."""
    plan = power_plan()
    if cohort.file_sha256(cohort.PIPELINE) != plan["pipeline"]["sha256"]:
        raise ValueError("capture pipeline differs from the frozen #104 pipeline")
    if cohort.player_identity()["assembly_sha256"] != plan["player"]["assembly_sha256"]:
        raise ValueError("campaign player differs from the frozen #104 player")
    current = cohort.renderer_block()["library_sha256"]
    frozen = plan["renderer"]["library_sha256"]
    return {"pipeline_sha256": plan["pipeline"]["sha256"], "player_assembly_sha256": plan["player"]["assembly_sha256"],
            "renderer_library_sha256": current,
            "renderer_libraries_differing_from_104": sorted(k for k in set(current) | set(frozen)
                                                            if current.get(k) != frozen.get(k))}


def prepare_engine(engine):
    plan = power_plan()
    binding = engine_preflight()
    branch = pin_directory(engine)
    if not (engine / "player").exists():
        shutil.copytree(cohort.OUTPUT / "player", engine / "player", symlinks=True)
    if cohort.file_sha256(engine / "player" / rpc.ASSEMBLY) != "sha256:" + plan["player"]["assembly_sha256"]:
        raise ValueError("engine player copy differs from the frozen #104 player")
    return {**binding, "branch": branch}


def engine_limits(engine):
    used = sum(read_json(p)["artifact_bytes_added"] for p in (engine / "run-log").glob("*.json")) \
        if (engine / "run-log").is_dir() else 0
    return {**power_plan()["limits"], "artifact_bytes": ARTIFACT_CAP_BYTES - used}


def chosen_executions(ctx, shot, decisions):
    out = {}
    for arm, seed in jobs(ctx):
        record = decisions.get((shot, arm, seed))
        if record is None:
            raise ValueError(f"shot {shot} decision {arm}/{seed_tag(seed)} missing")
        for level, choice in record["levels"].items():
            out[choice["execution"]] = (level, choice["chosen"])
    return dict(sorted(out.items()))


def frame_bytes(attempt):
    """sha256 of the first (decision) and last agent frame of a retained shot."""
    frames = agent_frames(attempt)
    if not frames or not frames[0].is_file() or not frames[-1].is_file():
        return None
    return {"frames": len(frames), "first": sha256_of(frames[0]), "last": sha256_of(frames[-1])}


def verdict_core(args):
    """Live verdict of one execution (``capture_pipeline_v2.branch_outcome`` over the retained shot) and the
    byte comparison of its decision / final frames with the recorded #104 attempt of the same candidate."""
    engine, execution, recorded_attempt = args
    entry = pipeline.branch_entry(engine, execution)
    out = {"status": entry["status"], "failure_class": entry.get("failure_class"), "failure": entry.get("failure"),
           "wall_seconds": entry.get("wall_seconds"), "stop_kind": None, "pig_removed": None,
           "native_clear": None, "channel": None, "renderer_ok": None, "shot_capture_seconds": None,
           "stage_seconds": None, "frames": None, "native_trace": None}
    if entry["status"] == "complete":
        result = read_json(Path(engine) / "results" / f"{execution}.json")
        outcome = pipeline.branch_outcome(result["segments"][0])
        live = frame_bytes(Path(engine) / "attempts" / execution)
        recorded = frame_bytes(recorded_attempt)
        out.update(stop_kind=outcome["stop_kind"], pig_removed=outcome["pig_removed"],
                   native_clear=outcome["native_clear"], channel=outcome["channel"],
                   renderer_ok=result.get("renderer") == power_plan()["limits"]["renderer"],
                   shot_capture_seconds=result["stage_seconds"].get("shot_capture"),
                   stage_seconds=result["stage_seconds"],
                   frames={"live": live, "recorded": recorded,
                           "decision_frame_equal": bool(live and recorded and live["first"] == recorded["first"]),
                           "final_frame_equal": bool(live and recorded and live["last"] == recorded["last"]),
                           "frame_count_equal": bool(live and recorded and live["frames"] == recorded["frames"])})
        recorded_result = Path(recorded_attempt).parent.parent / "results" / f"{Path(recorded_attempt).name}.json"
        live_trace = native_digest(result["segments"][0]["native_root"])
        recorded_trace = (native_digest(read_json(recorded_result)["segments"][0]["native_root"])
                          if recorded_result.is_file() else None)
        out["native_trace"] = {"live": live_trace, "recorded": recorded_trace,
                               "equal": bool(live_trace and recorded_trace and live_trace == recorded_trace)}
    return out


def native_digest(native_root):
    """sha256 over the decompressed native chunks (every physics step and event of the shot) in manifest order,
    with the run-specific capture / shot identifiers replaced by placeholders."""
    import gzip
    root = Path(native_root)
    if not (root / "native-manifest.json").is_file():
        return None
    manifest = read_json(root / "native-manifest.json")
    total = hashlib.sha256()
    for chunk in manifest["chunks"]:
        with gzip.open(root / chunk["path"], "rb") as handle:
            data = handle.read()
        for key in ("capture_id", "shot_id"):
            data = data.replace(str(manifest[key]).encode(), key.upper().encode())
        total.update(data)
    return {"chunks": len(manifest["chunks"]), "sha256": total.hexdigest()}


def frame_check(adapter, engine, execution, anchor):
    """E99 parse of the live decision frame (shot frame 0) against the anchor the planner ranked from."""
    attempt = Path(engine) / "attempts" / execution
    live = rt.anchor_carrier(adapter, attempt)
    _, _, frames, _ = s99.shot_frames(attempt)
    return {"observation_identity": frames[0]["agent_observation"]["identity"],
            "max_abs_diff": float((live - anchor.float()).abs().max())}


def observe(ctx, executions):
    """Verdict records for executed launches lacking one; returns (wall, gpu) seconds."""
    engine = ctx["engine"]
    pending = {e: lv for e, lv in executions.items() if not verdict_path(ctx["output"], e).is_file()
               and pipeline.branch_entry(engine, e)["status"] != "unattempted"}
    if not pending:
        return 0.0, 0.0
    began = time.monotonic()
    with multiprocessing.get_context("fork").Pool(16) as pool:
        cores = dict(zip(pending, pool.map(verdict_core, [
            (str(engine), e, str(CAMPAIGN / ctx["levels"][lv[0]]["stream"] / "attempts"
                                 / item_of(ctx["levels"][lv[0]], lv[1])["key"])) for e, lv in pending.items()])))
    gpu_began = time.monotonic()
    anchors = ctx["anchors"]()
    with harness.deterministic_scoring() as policy:
        adapter = s99.load_adapter(s99.OUTPUT, "E")
        for execution, (level, ordinal) in pending.items():
            core = cores[execution]
            entry = ctx["levels"][level]
            recorded = entry["recorded"][ordinal]
            agreement = None
            if core["status"] == "complete" and recorded["status"] == "complete":
                agreement = {"pig_removed": core["pig_removed"] == recorded["pig_removed"],
                             "stop_kind": core["stop_kind"] == recorded["stop_kind"]}
            frame = None
            if core["status"] == "complete":
                frame = {**frame_check(adapter, engine, execution, anchors[level]), "scoring_policy": policy}
            item = item_of(entry, ordinal)
            write_json(verdict_path(ctx["output"], execution), {
                "schema": SCHEMA_VERDICT, "plan_identity": IDENTITY, "context": ctx["name"], "execution": execution,
                "level": level, "set": entry["set"], "stream": entry["stream"], "ordinal": ordinal,
                "suffix": item["suffix"], "action": item["action"], **core, "recorded": recorded,
                "agreement": agreement, "decision_frame": frame, "observed_at": utc_now()})
    gpu = time.monotonic() - gpu_began
    return time.monotonic() - began, gpu


def engine_totals(output):
    """Campaign wall so far (raw run-logs; includes interrupted launches)."""
    root = Path(output) / "engine" / "run-log"
    return sum(read_json(p)["wall_seconds"] for p in root.glob("*.json")) if root.is_dir() else 0.0


def execute_shot(ctx, shot, ledger_output):
    decisions = load_decisions(ctx)
    executions = chosen_executions(ctx, shot, decisions)
    engine = ctx["engine"]
    pending = {e: lv for e, lv in executions.items()
               if pipeline.branch_entry(engine, e)["status"] == "unattempted"}
    projected = engine_totals(ledger_output) + len(pending) * AMORTIZED_BRANCH_SECONDS
    if ctx["name"] == "main" and projected > ENGINE_WALL_CAP_SECONDS:
        write_json(Path(ledger_output) / "blocked.json", {"shot": shot, "pending": len(pending),
                                                          "projected_campaign_wall_seconds": projected,
                                                          "cap_seconds": ENGINE_WALL_CAP_SECONDS, "at": utc_now()})
        log(f"shot {shot}: projected engine wall {projected / 3600:.1f} h exceeds the cap; blocked")
        return 3
    began = time.monotonic()
    binding = prepare_engine(engine)
    records = [dispatch_record(level, item_of(ctx["levels"][level], ordinal))
               for _, (level, ordinal) in pending.items()]
    campaign_wall = 0.0
    if records:
        log(f"{ctx['name']} shot {shot}: {len(records)} new live executions of {len(executions)} chosen "
            f"(shared: {len(executions) - len(records)})")
        campaign_wall = pipeline.run_campaign(engine, IDENTITY, records, engine_limits(engine),
                                              f"{IDENTITY}:{ctx['name']}:shot{shot}")
    observe_wall, gpu = observe(ctx, executions)
    engine_seconds = sum(read_json(engine / "receipts" / f"{e}.json").get("wall_seconds") or 0.0
                         for e in pending if (engine / "receipts" / f"{e}.json").is_file())
    wall = time.monotonic() - began
    mark(ledger_output, f"{ctx['name']}-execute-shot{shot}", wall, gpu, campaign_wall_seconds=campaign_wall,
         engine_seconds=engine_seconds, observe_seconds=observe_wall, executions=len(records),
         chosen=len(executions), placement=binding["branch"])
    log(f"{ctx['name']} shot {shot}: executed {len(records)} in {campaign_wall:.0f}s campaign wall")
    return 0


# ---------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------

def episodes(ctx, decisions, verdicts):
    """Every (arm, seed, level) episode: its shots (live verdicts) and outcome."""
    out = []
    for arm, seed in jobs(ctx):
        for level, entry in ctx["levels"].items():
            shots, outcome = [], "failed"
            for shot in range(1, SHOT_BUDGET + 1):
                record = decisions.get((shot, arm, seed))
                if record is None:
                    outcome = "incomplete"
                    break
                choice = record["levels"].get(level)
                if choice is None:
                    outcome = "no_candidate" if level in record["no_candidate"] else "failed"
                    break
                verdict = verdicts.get(choice["execution"])
                if verdict is None:
                    outcome = "incomplete"
                    break
                recorded = entry["recorded"][choice["chosen"]]
                shots.append({"shot": shot, "ordinal": choice["chosen"], "suffix": choice["suffix"],
                              "execution": choice["execution"], "status": verdict["status"],
                              "pass": passed(verdict), "native_clear": bool(verdict["native_clear"]),
                              "recorded_pass": recorded["pig_removed"] if recorded["status"] == "complete" else None,
                              "decision_seconds": choice["decision_seconds"],
                              "rollout_seconds": choice["rollout_seconds"], "linear_macs": choice["linear_macs"],
                              "candidates": choice["candidates"], "controller_calls": choice["controller_calls"],
                              "transition_calls": choice["transition_calls"],
                              "requests_chosen": choice.get("requests_chosen"),
                              "requests_all": choice.get("requests_all"), "rows_sha256": choice.get("rows_sha256")})
                if verdict["status"] != "complete":
                    outcome = "typed_failure"
                    break
                if passed(verdict):
                    outcome = "passed"
                    break
            out.append({"arm": arm, "seed": seed, "level": level, "set": entry["set"], "shots": shots,
                        "outcome": outcome})
    return out


def recorded_replay(ctx, decisions, arm, seed, level):
    """Declared cross-check only: the outcome the same planner would get if the recorded #104 verdicts
    were the engine (rankings re-used, exclusion re-simulated). Never the result."""
    entry = ctx["levels"][level]
    record = decisions.get((1, arm, seed))
    choice = record["levels"].get(level) if record else None
    if choice is None:
        return {"pass_shot": None}
    ranking = choice["ranking"]
    excluded = []
    for shot in range(1, SHOT_BUDGET + 1):
        pick = choose(ranking, excluded)
        if pick is None:
            break
        if entry["verdicts"].get(pick):
            return {"pass_shot": shot}
        excluded.append(pick)
    return {"pass_shot": None}


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def episode_view(ep, replay):
    shots = ep["shots"]
    pass_shot = next((s["shot"] for s in shots if s["pass"]), None)
    view = {f"pass{k}": float(pass_shot is not None and pass_shot <= k) for k in range(1, SHOT_BUDGET + 1)}
    view.update(shots_to_pass=None if pass_shot is None else float(pass_shot), shots_executed=float(len(shots)),
                typed_failure=float(ep["outcome"] == "typed_failure"),
                native_clear3=float(any(s["native_clear"] for s in shots)),
                recorded_pass3=float(replay["pass_shot"] is not None),
                replay_differs=float((replay["pass_shot"] is not None) != (pass_shot is not None)))
    return view


def cell_rows(ctx, eps, decisions, members):
    """One row per (level, seed); seed-free arms (prior, oracle) repeat their episode per seed (Gate-B
    weighting); the random arm's seed s is paired with the model seed s."""
    index = {(e["arm"], e["seed"], e["level"]): e for e in eps}
    rows = []
    for level in members:
        for seed in SEEDS:
            views = {}
            for arm in ARMS:
                arm_seed = None if arm in SEED_FREE else seed
                ep = index[(arm, arm_seed, level)]
                views[arm] = episode_view(ep, recorded_replay(ctx, decisions, arm, arm_seed, level))
            rows.append({"member": level, "seed": seed, "views": views})
    return rows


def field_block(rows, arm, field):
    values = [(r["views"][arm][field], r["member"]) for r in rows if r["views"][arm][field] is not None]
    return p94.bootstrap([v for v, _ in values], [m for _, m in values])


def contrast(rows, a, b, field, guards_ok):
    block = c2.difference_block(rows, [(1, a, field), (-1, b, field)])
    return {"minuend": a, "subtrahend": b, "field": field, "estimate": block,
            "token": c2.token(block) if guards_ok else STOP_TOKEN}


def compute_block(eps, arm):
    shots = [s for e in eps if e["arm"] == arm for s in e["shots"]]
    if not shots:
        return None
    per_shot = {}
    for k in range(1, SHOT_BUDGET + 1):
        chosen = [s for s in shots if s["shot"] == k]
        per_shot[str(k)] = {"decisions": len(chosen),
                            "mean_decision_seconds": float(np.mean([s["decision_seconds"] for s in chosen]))
                            if chosen else None}
    return {"decisions": len(shots),
            "mean_decision_seconds": float(np.mean([s["decision_seconds"] for s in shots])),
            "median_decision_seconds": float(np.median([s["decision_seconds"] for s in shots])),
            "mean_rollout_seconds": float(np.mean([s["rollout_seconds"] for s in shots])),
            "mean_linear_macs_per_decision": float(np.mean([s["linear_macs"] for s in shots])),
            "mean_linear_macs_per_candidate": float(np.mean([s["linear_macs"] / s["candidates"] for s in shots])),
            "mean_transition_calls_per_decision": float(np.mean([s["transition_calls"] for s in shots])),
            "mean_controller_calls_per_decision": float(np.mean([s["controller_calls"] for s in shots])),
            "per_shot": per_shot}


def request_mix(eps, arm):
    """Per shot index: shares of imagined rollout steps per request (Delta-description) for the chosen
    candidate, and over every ranked candidate, pooled over the arm's decisions at that shot."""
    out = {}
    for k in range(1, SHOT_BUDGET + 1):
        chosen, everything, decisions = {}, {}, 0
        for e in eps:
            for s in e["shots"]:
                if e["arm"] != arm or s["shot"] != k or s["requests_chosen"] is None:
                    continue
                decisions += 1
                for target, source in ((chosen, s["requests_chosen"]), (everything, s["requests_all"])):
                    for name, count in source.items():
                        target[name] = target.get(name, 0) + count
        if not decisions:
            out[str(k)] = None
            continue
        out[str(k)] = {"decisions": decisions,
                       "chosen": {n: c / sum(chosen.values()) for n, c in sorted(chosen.items())},
                       "all_candidates": {n: c / sum(everything.values()) for n, c in sorted(everything.items())}}
    return out


def set_table(ctx, eps, decisions, set_name, guards_ok):
    members = set_members(ctx["levels"], set_name)
    rows = cell_rows(ctx, eps, decisions, members)
    member_set = set(members)
    scoped = [e for e in eps if e["level"] in member_set]
    arms = {}
    for arm in ARMS:
        arm_eps = [e for e in scoped if e["arm"] == arm]
        dist = {str(k): sum(1 for e in arm_eps if e["outcome"] == "passed" and len(e["shots"]) == k)
                for k in range(1, SHOT_BUDGET + 1)}
        arms[arm] = {"group": GROUPS[arm], "episodes": len(arm_eps),
                     **{f"pass{k}": field_block(rows, arm, f"pass{k}") for k in range(1, SHOT_BUDGET + 1)},
                     "shots_to_pass": field_block(rows, arm, "shots_to_pass"),
                     "shots_executed": field_block(rows, arm, "shots_executed"),
                     "native_clear3": field_block(rows, arm, "native_clear3"),
                     "recorded_pass3": field_block(rows, arm, "recorded_pass3"),
                     "replay_differs_cells": int(sum(r["views"][arm]["replay_differs"] for r in rows)),
                     "passed_at_shot": dist,
                     "outcomes": {o: sum(1 for e in arm_eps if e["outcome"] == o)
                                  for o in ("passed", "failed", "typed_failure", "no_candidate", "incomplete")},
                     "compute": compute_block(arm_eps, arm),
                     "request_mix": request_mix(arm_eps, arm) if arm in ADAPTIVE else None,
                     "fixed_request": (fixed_request(arm) if arm in MODEL_ARMS and arm not in ADAPTIVE else None)}
    contrasts = {"vs_prior": {a: contrast(rows, a, PRIOR, "pass3", guards_ok) for a in ARMS if a != PRIOR},
                 "vs_random": {a: contrast(rows, a, RANDOM, "pass3", guards_ok) for a in ARMS if a != RANDOM},
                 "declared": [contrast(rows, a, b, "pass3", guards_ok) for a, b in CONTRAST_PAIRS]}
    return {"set": set_name, "levels": len(members), "cells": len(rows), "arms": arms, "contrasts": contrasts,
            "recorded_ceiling": float(np.mean([ctx["levels"][m]["recorded_any_pass"] for m in members]))
            if members else None}


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def guards(ctx, plan, decisions, verdicts, eps):
    out = {}
    records = list(decisions.values())
    out["G1_scoring_policy"] = {"records": len(records), "pass": bool(records) and all(
        r["scoring_policy"] == harness.SCORING_POLICY for r in records)}
    # G2 decision determinism: re-ranking at shots 2/3 reproduces shot-1 rows; N1 recorded-pass cells
    # reproduce the retained #101 / #114 grid rows of the same arm and seed
    mismatch, checked, retained_checked, retained_mismatch = [], 0, 0, []
    for (shot, arm, seed), record in decisions.items():
        if arm not in MODEL_ARMS and arm != PRIOR:
            continue
        first = decisions[(1, arm, seed)]["levels"]
        for level, choice in record["levels"].items():
            checked += 1
            if choice["rows_sha256"] != first[level]["rows_sha256"] or choice["ranking"] != first[level]["ranking"]:
                mismatch.append([arm, seed, shot, level])
    if ctx["name"] == "main":
        mixed = set(set_members(ctx["levels"], N1_MIXED))
        for arm in MODEL_ARMS:
            entry_ = etf.entry_of(arm)
            for seed in SEEDS:
                stored = etf.retained(entry_["source"], entry_["arm"], seed)["inventories"][INVENTORY]
                for level, choice in decisions[(1, arm, seed)]["levels"].items():
                    if level in mixed:
                        retained_checked += 1
                        if choice["rows"] != stored[level]["rows"]:
                            retained_mismatch.append([arm, seed, level])
    out["G2_decision_determinism"] = {"rerank_checked": checked, "rerank_mismatches": mismatch[:20],
                                      "retained_checked": retained_checked,
                                      "retained_mismatches": retained_mismatch[:20],
                                      "pass": checked > 0 and not mismatch and not retained_mismatch
                                      and (ctx["name"] != "main" or retained_checked > 0)}
    first_decision = min(r["decided_at"] for r in records)
    markers = sorted(read_json(p)["dispatched_at_utc"] for p in (ctx["engine"] / "markers").glob("*.json"))
    frozen = plan["frozen_at"] if plan else None
    out["G3_chronology"] = {"frozen_at": frozen, "first_decision": first_decision,
                            "first_dispatch": markers[0] if markers else None,
                            "pass": bool(markers) and (frozen is None or frozen <= first_decision)
                            and first_decision <= markers[0]}
    frames = [v["decision_frame"]["max_abs_diff"] for v in verdicts.values() if v["decision_frame"]]
    out["G4_decision_frame"] = {"complete_executions": len(frames),
                                "max_abs_diff": max(frames) if frames else None, "tolerance": TOLERANCE,
                                "scoring_policy": all(v["decision_frame"]["scoring_policy"] == harness.SCORING_POLICY
                                                      for v in verdicts.values() if v["decision_frame"]),
                                "pass": bool(frames) and max(frames) <= TOLERANCE}
    chosen = {s["execution"] for e in eps for s in e["shots"]}
    results = sorted(p.stem for p in (ctx["engine"] / "results").glob("*.json"))
    attempts = sorted(p.name for p in (ctx["engine"] / "attempts").iterdir()) if (ctx["engine"] / "attempts").is_dir() else []
    repeated = [[e["arm"], e["seed"], e["level"]] for e in eps
                if len({s["ordinal"] for s in e["shots"]}) != len(e["shots"])]
    incomplete = [[e["arm"], e["seed"], e["level"]] for e in eps if e["outcome"] == "incomplete"]
    out["G5_execution_integrity"] = {
        "executions": len(results), "chosen_executions": len(chosen),
        "unchosen_results": sorted(set(results) - chosen)[:20], "chosen_without_verdict": sorted(chosen - set(verdicts))[:20],
        "attempts_without_result": sorted(set(attempts) - set(results))[:20],
        "repeated_candidate_episodes": repeated[:20], "incomplete_episodes": len(incomplete),
        "pass": bool(results) and set(results) == chosen and chosen <= set(verdicts)
        and set(attempts) == set(results) and not repeated and not incomplete}
    out["all_pass"] = all(block["pass"] for block in out.values() if isinstance(block, dict))
    return out


def cross_check(verdicts):
    """Declared determinism cross-check: live verdict vs the recorded #104 verdict of the same level and
    candidate (never the result)."""
    complete = [v for v in verdicts.values() if v["agreement"] is not None]
    disagree = [{"execution": v["execution"], "live": [v["pig_removed"], v["stop_kind"]],
                 "recorded": [v["recorded"]["pig_removed"], v["recorded"]["stop_kind"]]}
                for v in complete if not (v["agreement"]["pig_removed"] and v["agreement"]["stop_kind"])]
    by_set = {}
    for v in verdicts.values():
        block = by_set.setdefault(v["set"], {"executions": 0, "complete": 0, "typed_failures": 0, "pass": 0,
                                             "pig_removed_agree": 0, "stop_kind_agree": 0, "compared": 0})
        block["executions"] += 1
        block["complete"] += v["status"] == "complete"
        block["typed_failures"] += v["status"] != "complete"
        block["pass"] += passed(v)
        if v["agreement"] is not None:
            block["compared"] += 1
            block["pig_removed_agree"] += v["agreement"]["pig_removed"]
            block["stop_kind_agree"] += v["agreement"]["stop_kind"]
    return {"compared": len(complete), "pig_removed_agree": sum(v["agreement"]["pig_removed"] for v in complete),
            "stop_kind_agree": sum(v["agreement"]["stop_kind"] for v in complete),
            "disagreements": disagree[:50], "disagreement_count": len(disagree), "by_set": by_set,
            "typed_failures": sorted([v["execution"], v["failure_class"]] for v in verdicts.values()
                                     if v["status"] != "complete"),
            "frames": {key: sum(1 for v in verdicts.values() if v["frames"] and v["frames"][key])
                       for key in ("decision_frame_equal", "final_frame_equal", "frame_count_equal")},
            "renderer_ok": sum(1 for v in verdicts.values() if v["renderer_ok"]),
            "native_trace_equal": sum(1 for v in verdicts.values() if v["native_trace"] and v["native_trace"]["equal"])}


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------

def ledger_path(output, phase):
    return Path(output) / "ledger" / f"{phase}.json"


def ledger_phases(output):
    root = Path(output) / "ledger"
    return {p.stem: read_json(p) for p in sorted(root.glob("*.json"))} if root.is_dir() else {}


def mark(output, phase, wall, gpu, **extra):
    """Ledger entry per phase; a repeated phase (a resumed stage) accumulates its seconds."""
    path = ledger_path(output, phase)
    old = read_json(path) if path.is_file() else {}
    summed = {k: old.get(k, 0.0) + v for k, v in (("wall_seconds", wall), ("gpu_seconds", gpu),
                                                  *((k, extra.pop(k)) for k in ("campaign_wall_seconds",
                                                                                "engine_seconds", "observe_seconds",
                                                                                "executions")
                                                    if k in extra))}
    write_json(path, {**old, "phase": phase, **summed, "at": utc_now(), "runs": old.get("runs", 0) + 1, **extra})


def ledger_totals(output):
    phases = ledger_phases(output)
    return {"gpu_seconds": sum(b["gpu_seconds"] for b in phases.values()),
            "wall_seconds": sum(b["wall_seconds"] for b in phases.values()),
            "engine_seconds": sum(b.get("engine_seconds", 0.0) for b in phases.values()),
            "campaign_wall_seconds": sum(b.get("campaign_wall_seconds", 0.0) for b in phases.values())}


# ---------------------------------------------------------------------------
# gallery (data/issue-102-closed-loop: frames / WebM per live execution, episodes in manifest.json)
# ---------------------------------------------------------------------------

def agent_frames(attempt):
    trace = Path(attempt) / "shot-1" / "observation-trace"
    manifest = trace / "observation_trace_manifest.json"
    if not manifest.is_file():
        return []
    return [trace / f["agent_observation"]["relative_path"] for f in read_json(manifest)["frame_records"]]


def render_one(args):
    from scripts.run_issue_62_successor_cohort import _encode_agent_frames_webm
    attempt, cell = Path(args[0]), Path(args[1])
    frames = [f for f in agent_frames(attempt) if f.is_file()]
    cell.mkdir(parents=True, exist_ok=True)
    if not frames:
        return "no agent frames retained"
    for name, source in (("decision.png", frames[0]), ("final.png", frames[-1])):
        if not (cell / name).is_file():
            shutil.copyfile(source, cell / name)
    if not (cell / "shot.webm").is_file():
        try:
            _encode_agent_frames_webm(frames, cell / "shot.webm")
        except Exception as error:  # recorded per cell, never retried silently
            return f"{type(error).__name__}: {error}"
    return None


def gallery_manifest(ctx, eps, verdicts, errors):
    root = ctx["gallery"]
    executions = []
    for execution, v in sorted(verdicts.items()):
        cell = root / "cells" / execution
        files = {name: {"bytes": (cell / name).stat().st_size, "sha256": sha256_of(cell / name)}
                 for name in ("decision.png", "final.png", "shot.webm") if (cell / name).is_file()}
        executions.append({"execution": execution, "level": v["level"], "set": v["set"], "ordinal": v["ordinal"],
                           "suffix": v["suffix"], "action": v["action"], "status": v["status"],
                           "failure_class": v["failure_class"], "pass": passed(v), "stop_kind": v["stop_kind"],
                           "recorded": v["recorded"], "files": files, "render_error": errors.get(execution)})
    return {"schema": SCHEMA_GALLERY, "identity": IDENTITY, "context": ctx["name"],
            "playback_fps": 50, "frames": "agent observation frames of the live shot (shot-1/observation-trace)",
            "pass_criterion": PASS_FIELD, "shot_budget": SHOT_BUDGET,
            "sharing": "one live execution per distinct level x candidate; episodes reference executions",
            "executions": executions,
            "episodes": [{"arm": e["arm"], "seed": e["seed"], "level": e["level"], "set": e["set"],
                          "outcome": e["outcome"],
                          "shots": [{"shot": s["shot"], "suffix": s["suffix"], "execution": s["execution"],
                                     "pass": s["pass"], "status": s["status"]} for s in e["shots"]]} for e in eps]}


def mark_of(shot):
    return "pass" if shot["pass"] else ("typed failure" if shot["status"] != "complete" else "fail")


def level_page(level, entry, executions, eps):
    lines = [f"<!doctype html><meta charset='utf-8'><title>{html.escape(level)}</title>",
             f"<h1>{html.escape(level)} ({entry['set']})</h1><p><a href='../index.html'>index</a> - recorded #104 "
             f"passing grid candidates: {sum(entry['verdicts'].values())}/16</p><h2>Live executions</h2>"]
    for x in executions:
        cell = f"../cells/{x['execution']}"
        verdict = "PASS" if x["pass"] else (f"typed failure ({x['failure_class']})" if x["status"] != "complete"
                                             else "fail")
        lines.append(f"<div style='display:inline-block;margin:4px'><b>{x['suffix']}</b> {verdict} "
                     f"({x['stop_kind']}; recorded pig_removed {x['recorded']['pig_removed']})<br>"
                     + (f"<video controls preload='none' width='320' poster='{cell}/decision.png' "
                        f"src='{cell}/shot.webm'></video>" if "shot.webm" in x["files"] else "no video")
                     + "</div>")
    lines.append("<h2>Episodes</h2><table border='1'><tr><th>arm</th><th>seed</th><th>shots</th><th>outcome</th></tr>")
    for e in eps:
        shots = " / ".join(f"{s['suffix']} {mark_of(s)}" for s in e["shots"])
        lines.append(f"<tr><td>{e['arm']}</td><td>{e['seed']}</td><td>{shots}</td><td>{e['outcome']}</td></tr>")
    lines.append("</table>")
    return "\n".join(lines) + "\n"


def write_gallery(ctx):
    decisions, verdicts = load_decisions(ctx), load_verdicts(ctx["output"])
    eps = episodes(ctx, decisions, verdicts)
    root = ctx["gallery"]
    tasks = [(str(ctx["engine"] / "attempts" / e), str(root / "cells" / e)) for e in sorted(verdicts)
             if (ctx["engine"] / "attempts" / e).is_dir()]
    began = time.monotonic()
    with multiprocessing.get_context("fork").Pool(WEBM_WORKERS) as pool:
        results = pool.map(render_one, tasks, chunksize=4)
    errors = {Path(cell).name: error for (_, cell), error in zip(tasks, results) if error}
    for execution in verdicts:
        if not (ctx["engine"] / "attempts" / execution).is_dir():
            errors[execution] = "no attempt directory"
    write_json(root / "render-errors.json", dict(sorted(errors.items())))
    manifest = gallery_manifest(ctx, eps, verdicts, errors)
    (root / "manifest.json").write_text(json_text(manifest))
    write_pages(ctx, manifest)
    log(f"gallery {root}: {len(tasks)} executions, {len(errors)} render errors ({time.monotonic() - began:.0f}s)")
    return time.monotonic() - began


def write_pages(ctx, manifest):
    root = ctx["gallery"]
    by_level = {}
    for x in manifest["executions"]:
        by_level.setdefault(x["level"], []).append(x)
    eps_by_level = {}
    for e in manifest["episodes"]:
        eps_by_level.setdefault(e["level"], []).append(e)
    (root / "levels").mkdir(parents=True, exist_ok=True)
    index = ["<!doctype html><meta charset='utf-8'><title>#102 closed-loop episodes</title>",
             "<h1>#102 closed-loop episodes (retry-from-start, live #104 player)</h1>",
             f"<p>{len(manifest['executions'])} live executions, {len(manifest['episodes'])} episodes; "
             "failed and typed-failure attempts included.</p>",
             "<table border='1'><tr><th>level</th><th>set</th><th>live executions</th><th>live passes</th></tr>"]
    for level in ctx["levels"]:
        xs = sorted(by_level.get(level, []), key=lambda x: x["ordinal"])
        (root / "levels" / f"{level}.html").write_text(
            level_page(level, ctx["levels"][level], xs, eps_by_level.get(level, [])))
        index.append(f"<tr><td><a href='levels/{level}.html'>{level}</a></td><td>{ctx['levels'][level]['set']}</td>"
                     f"<td>{len(xs)}</td><td>{sum(x['pass'] for x in xs)}</td></tr>")
    index.append("</table>")
    (root / "index.html").write_text("\n".join(index) + "\n")


# ---------------------------------------------------------------------------
# publication tables and renderings
# ---------------------------------------------------------------------------

def record_manifest(ctx):
    output, engine = ctx["output"], ctx["engine"]
    files = []

    def add(path, role):
        files.append({"path": str(Path(path).relative_to(ROOT)), "role": role, "bytes": Path(path).stat().st_size,
                      "sha256": sha256_of(path)})
    for path in sorted((output / "records" / "decisions").rglob("*.json")):
        add(path, "decision")
    for path in sorted((output / "records" / "verdicts").glob("*.json")):
        add(path, "live_verdict")
    for sub, role in (("results", "engine_result"), ("receipts", "engine_receipt"), ("run-log", "engine_run_log")):
        for path in sorted((engine / sub).glob("*.json")):
            add(path, role)
    add(n2_anchor_path(output), "n2_anchor_store")
    if posthoc_path(output).is_file():
        add(posthoc_path(output), "posthoc_decision_frames")
    add(gb.anchors_path(gb.OUTPUT), "n1_anchor_store")
    add(CAMPAIGN / "branches.json", "recorded_104_rows")
    if (ctx["gallery"] / "manifest.json").is_file():
        add(ctx["gallery"] / "manifest.json", "gallery_manifest")
    return {"schema": SCHEMA_MANIFEST, "identity": IDENTITY, "files": files, "count": len(files),
            "roles": {role: sum(1 for f in files if f["role"] == role) for role in sorted({f["role"] for f in files})}}


def engine_accounting(engine, totals):
    """Engine compute from the raw campaign run-logs (campaign wall, 16 workers) and receipts (per-execution worker
    wall). An interrupted launch writes its run-log but no ledger entry, so the raw files are authoritative."""
    runs = [read_json(p) for p in sorted((engine / "run-log").glob("*.json"))]
    receipts = [read_json(p) for p in sorted((engine / "receipts").glob("*.json"))]
    return {"engine_seconds": sum(r.get("wall_seconds") or 0.0 for r in receipts),
            "engine_campaign_wall_seconds": sum(r["wall_seconds"] for r in runs),
            "engine_campaign_runs": [{k: r[k] for k in ("wall_seconds", "attempts", "interrupted", "finished_at_utc",
                                                        "artifact_bytes_added")} for r in runs],
            "artifact_bytes_added": sum(r["artifact_bytes_added"] for r in runs),
            "ledger_engine_seconds": totals["engine_seconds"],
            "ledger_campaign_wall_seconds": totals["campaign_wall_seconds"]}


def compute_tables(ctx, plan):
    decisions, verdicts = load_decisions(ctx), load_verdicts(ctx["output"])
    eps = episodes(ctx, decisions, verdicts)
    guard = guards(ctx, plan, decisions, verdicts, eps)
    ok = guard["all_pass"]
    tables = {s: set_table(ctx, eps, decisions, s, ok) for s in TABLE_SETS}
    totals = ledger_totals(ctx["output"])
    smoke_totals = ledger_totals(ctx["output"] / "smoke")
    distinct = {}
    for v in verdicts.values():
        distinct.setdefault(v["set"], set()).add(v["level"])
    return {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
        "semantics": plan["semantics"], "pass_criterion": PASS_FIELD, "shot_budget": SHOT_BUDGET,
        "tables": tables, "guards": guard, "cross_check": cross_check(verdicts),
        "posthoc_decision_frames": (read_json(posthoc_path(ctx["output"])) if posthoc_path(ctx["output"]).is_file()
                                    else None),
        "executions": {"live": len(verdicts), "levels_touched": {k: len(v) for k, v in distinct.items()},
                       "episodes": len(eps),
                       "chosen_choices": sum(len(s) for s in (e["shots"] for e in eps))},
        "compute": {"gpu_seconds": totals["gpu_seconds"], "wall_seconds": totals["wall_seconds"],
                    **engine_accounting(ctx["engine"], totals),
                    "smoke": smoke_totals, "phases": ledger_phases(ctx["output"]),
                    "caps": {"gpu_seconds": GPU_CAP_SECONDS, "engine_campaign_wall_seconds": ENGINE_WALL_CAP_SECONDS,
                             "artifact_bytes": ARTIFACT_CAP_BYTES},
                    "hardware": "4x RTX 5090 (decisions, one process per GPU) and 48 CPU threads (16 engine "
                                "workers, llvmpipe); the #85 rules name an RTX 3090"},
    }


def num(value, digits=3):
    return "n/a" if value is None else format(value, f".{digits}f")


def cell(block, digits=3):
    if block is None:
        return "n/a"
    return f"{block['mean']:.{digits}f} [{block['interval'][0]:.{digits}f}, {block['interval'][1]:.{digits}f}]"


def macs(value):
    return "0" if not value else f"{value / 1e6:.1f}M"


def arm_line(table, arm):
    a = table["arms"][arm]
    c = a["compute"] or {}
    vp = table["contrasts"]["vs_prior"].get(arm)
    vr = table["contrasts"]["vs_random"].get(arm)
    return (f"| {arm} | {a['group']} | {cell(a['pass1'])} | {cell(a['pass2'])} | **{cell(a['pass3'])}** | "
            f"{cell(a['shots_to_pass'], 2)} | {num(c.get('mean_decision_seconds'), 4)} | "
            f"{macs(c.get('mean_linear_macs_per_decision'))} | "
            f"{cell(vp['estimate']) + ' ' + vp['token'] if vp else '-'} | "
            f"{cell(vr['estimate']) + ' ' + vr['token'] if vr else '-'} |")


def table_md(add, table):
    add(f"Levels {table['levels']}, cells (level x seed) {table['cells']}; recorded-inventory ceiling "
        f"{num(table['recorded_ceiling'])}.")
    add("")
    add("| Arm | Group | pass@1 | pass@2 | pass@3 | shots to pass (passed) | decision s | MACs / decision | "
        "pass@3 - prior | pass@3 - random |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for arm in ARMS:
        add(arm_line(table, arm))
    add("")


def findings_md(plan, compute):
    lines = []
    add = lines.append
    g = compute["guards"]
    add("# Issue #102 - closed-loop evaluation (retry-from-start, live #104 player)")
    add("")
    add(f"Frozen {compute['frozen_at']}. Pass = engine pig removal (`{PASS_FIELD}`), shot budget {SHOT_BUDGET}. "
        "Every shot resets the single-bird level; the planner re-ranks the 16-launch grid from the level's "
        "decision frame and excludes candidates already executed and failed. The state never changes between "
        "shots, so replanning amounts to excluding failed candidates. Every chosen launch was executed live; "
        "one execution per distinct level x candidate is shared by the episodes choosing it (pre-declared). "
        "Intervals: level-clustered bootstrap over levels x seeds, DESCRIPTIVE; contrast tokens use the delta "
        f"rule (delta = {DELTA}) on pass@3 differences.")
    add("")
    add(f"Guards: {'all pass' if g['all_pass'] else 'FAILED - every token is ' + STOP_TOKEN}.")
    add("")
    for set_name in TABLE_SETS:
        add(f"## {set_name}")
        add("")
        table_md(add, compute["tables"][set_name])
    add("## Declared contrasts (pass@3)")
    add("")
    add("| Set | Contrast | Estimate | Token |")
    add("| --- | --- | --- | --- |")
    for set_name in TABLE_SETS:
        for c in compute["tables"][set_name]["contrasts"]["declared"]:
            add(f"| {set_name} | {c['minuend']} - {c['subtrahend']} | {cell(c['estimate'])} | {c['token']} |")
    add("")
    add("## Request mix per shot (adaptive arms, N1; share of imagined rollout steps of the chosen candidate)")
    add("")
    add("| Arm | Shot | Decisions | Chosen-candidate mix | All-candidate mix |")
    add("| --- | --- | --- | --- | --- |")
    for arm in ADAPTIVE:
        for k, block in compute["tables"][N1]["arms"][arm]["request_mix"].items():
            if block is None:
                continue
            mix = lambda d: ", ".join(f"{n} {v:.2f}" for n, v in sorted(d.items(), key=lambda p: -p[1]) if v >= 0.005)
            add(f"| {arm} | {k} | {block['decisions']} | {mix(block['chosen'])} | {mix(block['all_candidates'])} |")
    add("")
    add("Fixed arms request one (Delta, description) pair at every step: " + ", ".join(
        f"{a} = {compute['tables'][N1]['arms'][a]['fixed_request']}" for a in MODEL_ARMS if a not in ADAPTIVE) + ".")
    add("")
    add("## Determinism cross-check (declared; never the result)")
    add("")
    x = compute["cross_check"]
    add(f"Live vs recorded #104 verdict of the same level and candidate: pig_removed {x['pig_removed_agree']}/"
        f"{x['compared']}, stop kind {x['stop_kind_agree']}/{x['compared']}; typed engine failures "
        f"{len(x['typed_failures'])}. Frames vs the recorded attempt of the same candidate (byte-equal): decision "
        f"frame {x['frames']['decision_frame_equal']}, final frame {x['frames']['final_frame_equal']}, frame count "
        f"{x['frames']['frame_count_equal']}; renderer string as frozen {x['renderer_ok']}; native physics trace "
        f"(every step and event, run identifiers masked) equal {x['native_trace_equal']}.")
    add("")
    add("| Set | Arm | live pass@3 | recorded-replay pass@3 | cells whose outcome differs |")
    add("| --- | --- | --- | --- | --- |")
    for set_name in SETS:
        for arm in ARMS:
            a = compute["tables"][set_name]["arms"][arm]
            add(f"| {set_name} | {arm} | {cell(a['pass3'])} | {cell(a['recorded_pass3'])} | "
                f"{a['replay_differs_cells']} |")
    add("")
    p = compute["posthoc_decision_frames"]
    if p:
        add("## POST-HOC: decision-frame sensitivity (declared after G4 failed; changes no token)")
        add("")
        add(f"Levels whose live decision frame parsed beyond the tolerance: {', '.join(p['levels'])}. Re-ranking "
            f"from each distinct live decision frame with every model arm and seed keeps the first pick in "
            f"{p['same_first']}/{p['comparisons']} and the first three picks in {p['same_first_three']}/"
            f"{p['comparisons']} comparisons.")
        add("")
    add("## Guards")
    add("")
    for name, block in g.items():
        if isinstance(block, dict):
            add(f"- {name}: {'pass' if block['pass'] else 'FAIL'} "
                f"({json.dumps({k: v for k, v in block.items() if k != 'pass' and not isinstance(v, list)}, sort_keys=True)})")
    add("")
    c = compute["compute"]
    add("## Compute")
    add("")
    add(f"GPU {c['gpu_seconds']:.0f} s; stage wall {c['wall_seconds']:.0f} s; engine (per-execution worker wall) "
        f"{c['engine_seconds']:.0f} s; engine campaign wall {c['engine_campaign_wall_seconds']:.0f} s on 16 workers; "
        f"live executions {compute['executions']['live']}. Smoke (pre-freeze): GPU {c['smoke']['gpu_seconds']:.0f} s, "
        f"engine {c['smoke']['engine_seconds']:.0f} s, campaign wall {c['smoke']['campaign_wall_seconds']:.0f} s.")
    add("")
    return "\n".join(lines) + "\n"


def summary(plan, compute):
    def brief(set_name):
        t = compute["tables"][set_name]
        return {arm: {"pass3": t["arms"][arm]["pass3"], "pass1": t["arms"][arm]["pass1"],
                      "shots_to_pass": t["arms"][arm]["shots_to_pass"],
                      "minus_prior": t["contrasts"]["vs_prior"].get(arm),
                      "minus_random": t["contrasts"]["vs_random"].get(arm)} for arm in ARMS}
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "validation_command": VALIDATION_COMMAND, "guards_pass": compute["guards"]["all_pass"],
            "sets": {s: brief(s) for s in TABLE_SETS},
            "declared": {s: compute["tables"][s]["contrasts"]["declared"] for s in TABLE_SETS},
            "cross_check": {k: compute["cross_check"][k] for k in ("compared", "pig_removed_agree", "stop_kind_agree",
                                                                   "disagreement_count")},
            "executions": compute["executions"],
            "compute": {k: compute["compute"][k] for k in ("gpu_seconds", "wall_seconds", "engine_seconds",
                                                          "engine_campaign_wall_seconds")}}


CSV_FIELDS = ("set", "arm", "group", "episodes", "pass1", "pass1_low", "pass1_high", "pass2", "pass2_low",
              "pass2_high", "pass3", "pass3_low", "pass3_high", "shots_to_pass", "shots_to_pass_low",
              "shots_to_pass_high", "native_clear3", "recorded_pass3", "mean_decision_seconds",
              "mean_rollout_seconds", "mean_linear_macs_per_decision", "mean_linear_macs_per_candidate",
              "minus_prior", "minus_prior_low", "minus_prior_high", "minus_prior_token", "minus_random",
              "minus_random_low", "minus_random_high", "minus_random_token")


def table_csv(compute):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for set_name in TABLE_SETS:
        t = compute["tables"][set_name]
        for arm in ARMS:
            a = t["arms"][arm]
            row = {"set": set_name, "arm": arm, "group": a["group"], "episodes": a["episodes"]}
            for field in ("pass1", "pass2", "pass3", "shots_to_pass"):
                block = a[field]
                row[field] = num(block and block["mean"], 4)
                row[f"{field}_low"] = num(block and block["interval"][0], 4)
                row[f"{field}_high"] = num(block and block["interval"][1], 4)
            row["native_clear3"] = num(a["native_clear3"] and a["native_clear3"]["mean"], 4)
            row["recorded_pass3"] = num(a["recorded_pass3"] and a["recorded_pass3"]["mean"], 4)
            c = a["compute"] or {}
            row["mean_decision_seconds"] = num(c.get("mean_decision_seconds"), 5)
            row["mean_rollout_seconds"] = num(c.get("mean_rollout_seconds"), 5)
            row["mean_linear_macs_per_decision"] = num(c.get("mean_linear_macs_per_decision"), 0)
            row["mean_linear_macs_per_candidate"] = num(c.get("mean_linear_macs_per_candidate"), 0)
            for key, name in (("vs_prior", "minus_prior"), ("vs_random", "minus_random")):
                con = t["contrasts"][key].get(arm)
                est = con["estimate"] if con else None
                row[name] = num(est and est["mean"], 4)
                row[f"{name}_low"] = num(est and est["interval"][0], 4)
                row[f"{name}_high"] = num(est and est["interval"][1], 4)
                row[f"{name}_token"] = con["token"] if con else ""
            writer.writerow(row)
    return buffer.getvalue()


def rendered(ctx, plan, compute):
    return {"compute.json": json_text(compute), "summary.json": json_text(summary(plan, compute)),
            "findings.md": findings_md(plan, compute), "closed-loop-table.csv": table_csv(compute),
            "manifest.json": json_text(record_manifest(ctx))}


# ---------------------------------------------------------------------------
# smoke (pre-freeze, fit-split levels only)
# ---------------------------------------------------------------------------

def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the smoke is a pre-freeze control")
    ctx = smoke_context(output)
    root = ctx["output"]
    began = time.monotonic()
    if not n2_anchor_path(root).is_file():
        n2 = [m for m, e in ctx["levels"].items() if e["set"] == N2]
        wall = parse_anchors(n2_anchor_path(root), n2, ctx["levels"])
        mark(root, "smoke-anchors", wall, wall)
    # (c) the scoring path reproduces the retained source rows on N1 evaluation grid cells (no new outcome)
    retained_checks = []
    if not ledger_path(root, "smoke-retained").is_file():
        start = time.monotonic()
        _, cells = gb.evaluation_set()
        sample = sorted(cells[INVENTORY]["mixed"])[:SMOKE_CELLS]
        anchors = {"E99": gb.load_anchors(gb.OUTPUT)["anchors"]}
        with harness.deterministic_scoring():
            for pid in MODEL_ARMS:
                entry_ = etf.entry_of(pid)
                arm = etf.load_arm(entry_, SMOKE_SEED, None)
                stored = etf.retained(entry_["source"], entry_["arm"], SMOKE_SEED)["inventories"][INVENTORY]
                for member in sample:
                    block = etf.run_cell(arm, INVENTORY, member, cells[INVENTORY]["mixed"][member], anchors,
                                         dap.endpoint_costs(), SMOKE_SEED)
                    retained_checks.append({"arm": pid, "member": member, "equal": block["rows"] == stored[member]["rows"]})
                torch.cuda.empty_cache()
        wall = time.monotonic() - start
        mark(root, "smoke-retained", wall, wall, checks=retained_checks)
    retained_checks = read_json(ledger_path(root, "smoke-retained"))["checks"]
    for shot in range(1, SHOT_BUDGET + 1):
        if decide_jobs(ctx, shot, jobs(ctx), 0, root):
            raise RuntimeError("smoke decisions stopped at the GPU cap")
        if execute_shot(ctx, shot, root):
            raise RuntimeError("smoke execution blocked")
    gallery_wall = write_gallery(ctx)
    mark(root, "smoke-gallery", gallery_wall, 0.0)
    decisions, verdicts = load_decisions(ctx), load_verdicts(root)
    eps = episodes(ctx, decisions, verdicts)
    guard = guards(ctx, None, decisions, verdicts, eps)
    x = cross_check(verdicts)
    manifest = read_json(ctx["gallery"] / "manifest.json")
    rendered_ok = all("shot.webm" in e["files"] and "decision.png" in e["files"] for e in manifest["executions"]
                      if e["status"] == "complete")
    checks = {
        "live_equals_recorded": {"compared": x["compared"], "pig_removed_agree": x["pig_removed_agree"],
                                 "stop_kind_agree": x["stop_kind_agree"], "disagreements": x["disagreements"],
                                 "typed_failures": x["typed_failures"],
                                 "native_trace_equal": x["native_trace_equal"],
                                 "pass": x["compared"] == len(verdicts) and x["disagreement_count"] == 0
                                 and x["native_trace_equal"] == x["compared"]},
        "decisions_under_deterministic_scoring": guard["G1_scoring_policy"],
        "retained_rows_reproduced": {"checks": retained_checks,
                                     "pass": bool(retained_checks) and all(c["equal"] for c in retained_checks)},
        "rerank_determinism": guard["G2_decision_determinism"],
        "decision_frame": guard["G4_decision_frame"],
        "execution_integrity": guard["G5_execution_integrity"],
        "exclusion_exercised": {"episodes_with_two_or_more_shots": sum(1 for e in eps if len(e["shots"]) >= 2),
                                "pass": any(len(e["shots"]) >= 2 for e in eps)},
        "gallery_rendered": {"executions": len(manifest["executions"]), "pass": rendered_ok},
        "frames_vs_recorded_reported": {**x["frames"], "renderer_ok": x["renderer_ok"], "pass": True},
    }
    totals = ledger_totals(root)
    record = {"schema": SCHEMA_SMOKE, "identity": IDENTITY, "ran_at": utc_now(),
              "levels": {m: {"set": e["set"], "stream": e["stream"],
                             "recorded_passing": sorted(o for o, v in e["verdicts"].items() if v)}
                         for m, e in ctx["levels"].items()},
              "arms": [[a, s] for a, s in jobs(ctx)], "checks": checks,
              "episodes": [{"arm": e["arm"], "seed": e["seed"], "level": e["level"], "outcome": e["outcome"],
                            "shots": [[s["suffix"], s["pass"]] for s in e["shots"]]} for e in eps],
              "live_executions": len(verdicts), "compute": totals,
              "wall_seconds_this_call": time.monotonic() - began,
              "pass": all(c["pass"] for c in checks.values())}
    write_json(output / "smoke.json", record)
    log(f"smoke {'passed' if record['pass'] else 'FAILED'}: {len(verdicts)} live executions; "
        f"{json.dumps({k: c['pass'] for k, c in checks.items()})}")
    return 0 if record["pass"] else 1


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def input_bindings():
    names = {"issue_104_branches": CAMPAIGN / "branches.json", "issue_104_plan": cohort.OUTPUT / "plan.json",
             "issue_109_cohort_plan": cohort.COHORT_109 / "plan.json",
             "issue_110_cohort_plan": cohort.COHORT_110 / "plan.json",
             "gate_b_anchor_store": gb.anchors_path(gb.OUTPUT), "gate_b_handoff": gb.HANDOFF,
             "fit_verdicts": gb.FIT_VERDICTS, "issue_101_selection": dap.selection_path(dap.OUTPUT),
             "capture_pipeline": cohort.PIPELINE, "runner": ROOT / RUNNER}
    return [{"name": n, "artifact": str(p.relative_to(ROOT)), "sha256": sha256_of(p)} for n, p in names.items()]


def projection(level_map):
    worst = sum(len(e["inventory"]) for e in level_map.values())
    return {"worst_case_executions": worst,
            "worst_case_campaign_wall_hours": worst * AMORTIZED_BRANCH_SECONDS / 3600,
            "amortized_branch_seconds": AMORTIZED_BRANCH_SECONDS,
            "rule": "worst case = every grid candidate of every level executed once (sharing caps executions at "
                    "levels x 16); #104 amortized branch wall at 16 workers",
            "within_cap": worst * AMORTIZED_BRANCH_SECONDS <= ENGINE_WALL_CAP_SECONDS}


def plan_rules():
    return {
        "semantics": {
            "episode": ("retry-from-start: every shot k resets the same single-bird level to its start state; "
                        "before each shot the planner ranks the frozen 16-launch grid from the level's decision frame "
                        "with its model or rule, excludes candidates already executed and failed in the episode, "
                        "and picks the best remaining one; the launch is executed live in the #104 player"),
            "replan_disclosure": ("the state never changes between shots, so replanning amounts to excluding failed "
                                  "candidates; the model re-ranks at every shot (re-ranking identity is guard G2)"),
            "stop": f"first pass, or the shot budget {SHOT_BUDGET} spent, or a typed engine failure",
            "pass": f"live engine pig removal (`{PASS_FIELD}`, #85 channel via capture_pipeline_v2.branch_outcome)",
            "native_clear": "level-clear terminal reported descriptively (pass@3 under native_clear)",
            "typed_failure": ("a typed engine failure of a chosen launch is recorded, never retried; the episode ends "
                              "as typed_failure and counts as not passed"),
            "sharing": ("pre-declared: one live execution per distinct level x candidate chosen by any arm, seed or "
                        "shot; every episode choosing it reads the same live verdict (execution is deterministic: "
                        "#104 replay 160/160)"),
            "recorded_outcomes": ("the recorded #104 verdicts are only (a) the oracle's inventory and (b) a declared "
                                  "cross-check (live vs recorded verdict; recorded-replay pass@3); never the result"),
            "decision_frame": ("N1: the Gate-B anchor store (E99 parse of shot frame 0 of the level's first complete "
                               "recorded candidate); N2: the same rule parsed here after the freeze; guard G4 checks "
                               "the live shot frame 0 of every complete execution against it"),
            "ranking": ("lowest #111 tie-free endpoint cost over the 16 candidates (ties to the lower ordinal; "
                        "nonfinite endpoints last); prior: Gate-B rule (minus the fit-split success frequency of the "
                        "identical action), ties to the lower ordinal; random: one PCG64([102, seed, level index]) "
                        "permutation per seed and level, first non-excluded entry (uniform without replacement); "
                        "oracle: recorded passing candidates, lowest ordinal first, no shot when none is recorded"),
            "per_decision_compute": ("CUDA-synchronized rollout seconds (block gpu_seconds) and host decision seconds "
                                     "of ranking all 16 candidates, one process per RTX 5090, warm-up level discarded "
                                     "per arm x seed x shot; linear MACs per decision = the rollout's linear MACs "
                                     "(#96/#101 accounting)"),
            "request_mix": ("adaptive arms: shares of imagined rollout steps per (Delta, description) request of the "
                            "chosen candidate, and of all 16 ranked candidates, pooled per shot index"),
        },
        "arms": {arm: {"group": GROUPS[arm], "seeds": list(arm_seeds(arm)) if arm not in SEED_FREE else ["seed-free"],
                       "source": etf.entry_of(arm)["source"] if arm in MODEL_ARMS else "rule",
                       "source_arm": etf.entry_of(arm)["arm"] if arm in MODEL_ARMS else arm} for arm in ARMS},
        "statistics": {
            "cells": "(level, seed); seed-free arms repeat per seed; random seed s pairs with model seed s",
            "bootstrap": "level-clustered percentile bootstrap, 10000 draws, PCG64 7201 (p94.bootstrap), DESCRIPTIVE",
            "metrics": ["pass@1", "pass@2", "pass@3", "shots to pass (passed episodes)", "shots executed",
                        "native-clear pass@3", "per-decision seconds / MACs / calls", "request mix per shot"],
            "contrasts": {"every arm - prior": "pass@3", "every arm - random": "pass@3",
                          "declared": [f"{a} - {b}" for a, b in CONTRAST_PAIRS]},
            "token_rule": (f"delta = {DELTA} on the pass@3 difference: supported iff mean >= delta and lower bound > 0; "
                           "not_supported_by_this_experiment iff upper bound < delta; otherwise "
                           f"{STOP_TOKEN} (c2.token)"),
            "sets": {N1: "400 #109 evaluation levels (primary)",
                     N1_MIXED: "the N1 levels with at least one recorded passing grid candidate (declared subset)",
                     N2: "40 #110 evaluation appearance levels (novelty level 1, novel side), reported separately"},
        },
        "guards": {"G1": "scoring policy on every decision record", "G2": "re-ranking identity across shots and "
                   "retained #101/#114 grid rows on N1-mixed shot-1 cells", "G3": "chronology: freeze <= first "
                   "decision <= first dispatch", "G4": "live decision frame E99 parse within the parse tolerance of "
                   "the anchor", "G5": "execution integrity: one live execution per chosen level x candidate, a "
                   "verdict for each, no candidate repeated in an episode, no incomplete episode",
                   "failure": "a failed guard turns every token into readiness_or_precision_insufficient"},
        "execution": {"limits": "#104 plan limits verbatim (16 isolated workers, port blocks from 29100, cold-start "
                                "gate, no retries) except the artifact cap", "artifact_cap_bytes": ARTIFACT_CAP_BYTES,
                      "engine_campaign_wall_cap_seconds": ENGINE_WALL_CAP_SECONDS, "gpu_cap_seconds": GPU_CAP_SECONDS,
                      "workers": 16, "launch": "each stage launched once with recorded PIDs and waited on; never "
                                               "relaunched over live workers"},
    }


def frozen_plan(output, smoke_record):
    level_map = levels()
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "issue": 102, "frozen_at": utc_now(),
        "validation_command": VALIDATION_COMMAND, "runner": RUNNER,
        "levels": {s: set_members(level_map, s) for s in TABLE_SETS},
        "level_counts": {s: len(set_members(level_map, s)) for s in TABLE_SETS},
        "inventory": {"name": INVENTORY, "candidates": 16,
                      "actions": [i["action"] for i in next(iter(level_map.values()))["inventory"]]},
        "shot_budget": SHOT_BUDGET, "pass_criterion": PASS_FIELD, "seeds": list(SEEDS),
        "projection": projection(level_map), "inputs": input_bindings(),
        "smoke": {"path": "smoke.json", "sha256": sha256_of(Path(output) / "smoke.json"), "pass": smoke_record["pass"],
                  "live_executions": smoke_record["live_executions"]},
        "known_before_freeze": ("every published #101/#103/#105/#114 grid ranking on the N1 mixed levels (including "
                                "#103 top-3 hit, which equals recorded-replay pass@3 under this semantics) and the #104 "
                                "verdict yields; no model output on the 89 N1 levels without a recorded pass or on any "
                                "#110 evaluation level; no live #102 execution on an evaluation level"),
        "disclosures": {
            "issue_104_input_paths": ("cohort.load_plan's input re-hash fails on exactly "
                                      f"{list(POWER_PLAN_PATH_REWRITES)}; their diffs against the committed bytes are "
                                      "machine-local home-directory path rewrites only, so the #104 plan is read "
                                      "directly and any other changed input refuses"),
            "renderer_libraries": {**{k: v for k, v in engine_preflight().items()
                                      if k == "renderer_libraries_differing_from_104"},
                                   "note": ("the Mesa 26.1.2 / LLVM 22.1.6 build under ~/.cache/novphy-mesa was rebuilt "
                                            "on this host (2026-10-14; embedded install paths differ); the renderer "
                                            "string is checked per execution and decision / final frames are compared "
                                            "byte for byte with the recorded #104 frames (reported)")},
            "hardware": "RTX 5090 instead of the #85 RTX 3090",
        },
        **plan_rules(),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan binding differs")
    changed = [i["name"] for i in plan["inputs"] if i["name"] != "runner"
               and sha256_of(ROOT / i["artifact"]) != i["sha256"]]
    if changed:
        raise ValueError(f"frozen inputs changed after the freeze: {changed}")
    return plan


def prepare(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        load_plan(output)
        log("plan already frozen")
        return 0
    smoke_record = read_json(output / "smoke.json")
    if not smoke_record["pass"]:
        raise ValueError("the smoke did not pass; no freeze")
    plan = frozen_plan(output, smoke_record)
    write_json(output / "plan.json", plan)
    log(f"frozen {plan['frozen_at']}: {plan['level_counts']}, projection {plan['projection']}")
    return 0


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

def anchors(output):
    output = Path(output)
    load_plan(output)
    path = n2_anchor_path(output)
    if path.is_file():
        log("N2 anchors already parsed")
        return 0
    members = set_members(levels(), N2)
    wall = parse_anchors(path, members, levels())
    mark(output, "anchors-n2", wall, wall, levels=len(members))
    log(f"parsed {len(members)} N2 anchors ({wall:.0f}s)")
    return 0


def decide(output, shot, workers, worker):
    output = Path(output)
    load_plan(output)
    ctx = main_context(output)
    if not n2_anchor_path(output).is_file():
        raise ValueError("N2 anchors missing; run --anchors first")
    selected = ctx["model_jobs"][worker::workers] + (ctx["rule_jobs"] if worker == 0 else [])
    return decide_jobs(ctx, shot, selected, worker, output)


def execute(output, shot):
    output = Path(output)
    load_plan(output)
    return execute_shot(main_context(output), shot, output)


def gallery(output):
    output = Path(output)
    load_plan(output)
    wall = write_gallery(main_context(output))
    mark(output, "gallery", wall, 0.0)
    return 0


def publish(output):
    output = Path(output)
    plan = load_plan(output)
    ctx = main_context(output)
    compute = compute_tables(ctx, plan)
    for name, text in rendered(ctx, plan, compute).items():
        (output / name).write_text(text)
    log(f"published; guards {'pass' if compute['guards']['all_pass'] else 'FAIL'}")
    return 0


def posthoc_path(output):
    return Path(output) / "records" / "posthoc-decision-frames.json"


def posthoc_frames(output):
    """POST-HOC (declared after G4 failed; changes no token): on every level where a live decision frame parsed more
    than the tolerance away from the planner's anchor, re-rank the grid from each distinct live decision frame with
    every model arm x seed and report whether the first three picks (the episode's possible shots; exclusion
    follows the ranking order) or the first pick would change."""
    output = Path(output)
    load_plan(output)
    ctx = main_context(output)
    decisions, verdicts = load_decisions(ctx), load_verdicts(output)
    exceed = sorted({v["level"] for v in verdicts.values()
                     if v["decision_frame"] and v["decision_frame"]["max_abs_diff"] > TOLERANCE})
    began = time.monotonic()
    rows = []
    with harness.deterministic_scoring() as policy:
        adapter = s99.load_adapter(s99.OUTPUT, "E")
        frames = {}
        for level in exceed:
            groups = {}
            for v in sorted(verdicts.values(), key=lambda v: v["execution"]):
                if v["level"] == level and v["decision_frame"]:
                    groups.setdefault(v["frames"]["live"]["first"], []).append(v)
            frames[level] = [{"frame_sha256": sha, "executions": [v["execution"] for v in group],
                              "max_abs_diff": group[0]["decision_frame"]["max_abs_diff"],
                              "z": rt.anchor_carrier(adapter, ctx["engine"] / "attempts" / group[0]["execution"])}
                             for sha, group in sorted(groups.items())]
        costs = dap.endpoint_costs()
        for pid in MODEL_ARMS:
            for seed in SEEDS:
                arm = etf.load_arm(etf.entry_of(pid), seed, None)
                for level in exceed:
                    base = decisions[(1, pid, seed)]["levels"][level]["ranking"]
                    for frame in frames[level]:
                        block = etf.run_cell(arm, INVENTORY, level, ctx["levels"][level], {"E99": {level: frame["z"]}},
                                             costs, seed)
                        ranking = model_ranking(block["rows"])
                        rows.append({"arm": pid, "seed": seed, "level": level, "frame_sha256": frame["frame_sha256"],
                                     "max_abs_diff": frame["max_abs_diff"], "same_first": ranking[0] == base[0],
                                     "same_first_three": ranking[:SHOT_BUDGET] == base[:SHOT_BUDGET]})
                torch.cuda.empty_cache()
    record = {"schema": "issue_102_posthoc_decision_frames_v1", "plan_identity": IDENTITY, "label": "POST-HOC",
              "scoring_policy": policy, "computed_at": utc_now(), "levels": exceed,
              "frames": {lv: [{k: f[k] for k in ("frame_sha256", "executions", "max_abs_diff")} for f in fs]
                         for lv, fs in frames.items()},
              "rows": rows, "same_first": sum(r["same_first"] for r in rows),
              "same_first_three": sum(r["same_first_three"] for r in rows), "comparisons": len(rows)}
    write_json(posthoc_path(output), record)
    wall = time.monotonic() - began
    mark(output, "posthoc-decision-frames", wall, wall, comparisons=len(rows))
    log(f"post-hoc decision frames: {record['same_first_three']}/{len(rows)} same first-three picks, "
        f"{record['same_first']}/{len(rows)} same first pick ({wall:.0f}s)")
    return 0


def spot_checks(ctx):
    """Re-decide one arm x seed on the first N1 levels and recompute live verdicts of a sample (exact)."""
    decisions = load_decisions(ctx)
    record = decisions[(1, UNIT, SEEDS[0])]
    sample = sorted(record["levels"])[:20]
    anchors_ = {"E99": ctx["anchors"]()}
    with harness.deterministic_scoring():
        arm = etf.load_arm(etf.entry_of(UNIT), SEEDS[0], None)
        same = all(digest(etf.run_cell(arm, INVENTORY, level, ctx["levels"][level], anchors_, dap.endpoint_costs(),
                                       SEEDS[0])["rows"]) == record["levels"][level]["rows_sha256"]
                   for level in sample)
    verdicts = load_verdicts(ctx["output"])
    picks = sorted(verdicts)[:: max(1, len(verdicts) // 12)][:12]
    fresh = [verdict_core((str(ctx["engine"]), e, str(CAMPAIGN / verdicts[e]["stream"] / "attempts"
                                                      / verdicts[e]["recorded"]["identity"]))) for e in picks]
    verdict_same = all(f[k] == verdicts[e][k] for f, e in zip(fresh, picks)
                       for k in ("status", "pig_removed", "stop_kind", "frames"))
    return {"rerank_levels": len(sample), "rerank_equal": same, "verdicts_rechecked": len(picks),
            "verdicts_equal": verdict_same, "pass": bool(same and verdict_same)}


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    ctx = main_context(output)
    compute = compute_tables(ctx, plan)
    problems = [name for name, text in rendered(ctx, plan, compute).items()
                if not (output / name).is_file() or (output / name).read_text() != text]
    decisions, verdicts = load_decisions(ctx), load_verdicts(output)
    eps = episodes(ctx, decisions, verdicts)
    gallery_path = ctx["gallery"] / "manifest.json"
    errors = read_json(ctx["gallery"] / "render-errors.json") if (ctx["gallery"] / "render-errors.json").is_file() else {}
    if not gallery_path.is_file() or gallery_path.read_text() != json_text(gallery_manifest(ctx, eps, verdicts, errors)):
        problems.append("gallery manifest.json")
    checks = spot_checks(ctx)
    if not checks["pass"]:
        problems.append(f"spot checks {checks}")
    if not compute["guards"]["all_pass"]:
        log("note: guards failed; tokens are readiness_or_precision_insufficient")
    if problems:
        log(f"validation FAILED: {problems}")
        return 1
    log(f"validation passed ({time.monotonic() - began:.0f}s): {checks}")
    return 0


def dry_run(output):
    level_map = levels()
    counts = {s: len(set_members(level_map, s)) for s in TABLE_SETS}
    ctx = main_context(output)
    log(f"levels {counts}; arms {len(ARMS)}; jobs {len(jobs(ctx))}; smoke levels {list(smoke_levels())}")
    log(f"projection {projection(level_map)}")
    return 0


def status(output):
    output = Path(output)
    ctx = main_context(output)
    decisions, verdicts = load_decisions(ctx), load_verdicts(output)
    for shot in range(1, SHOT_BUDGET + 1):
        made = [k for k in decisions if k[0] == shot]
        if made:
            chosen = chosen_executions(ctx, shot, decisions) if len(made) == len(jobs(ctx)) else {}
            log(f"shot {shot}: {len(made)}/{len(jobs(ctx))} decision records; chosen executions {len(chosen)}, "
                f"with verdict {sum(1 for e in chosen if e in verdicts)}")
    log(f"verdicts {len(verdicts)}; ledger {ledger_totals(output)}")
    return 0


MODES = {"dry-run": dry_run, "smoke": smoke, "prepare": prepare, "anchors": anchors, "decide": decide,
         "execute": execute, "gallery": gallery, "posthoc-frames": posthoc_frames, "publish": publish,
         "validate": validate, "status": status}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--shot", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker", type=int, default=0)
    args = parser.parse_args()
    mode = next(m for m in MODES if getattr(args, m.replace("-", "_")))
    if mode == "decide":
        return decide(OUTPUT, args.shot, args.workers, args.worker)
    if mode == "execute":
        return execute(OUTPUT, args.shot)
    return MODES[mode](OUTPUT)


if __name__ == "__main__":
    raise SystemExit(main())
