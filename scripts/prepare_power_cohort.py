"""Issue #104: freeze the joint #104 + #110 capture-campaign plan (power cohort).

One frozen plan merges three design artifacts before any capture:

* the #109 sealed cohort (rolling type010103 / sliding type010105; fit, policy and
  evaluation rosters, deduplicated candidate inventories);
* the #110 novelty cells (45 scenario x novelty cells through 40 paired templates;
  smoke, fit and evaluation rosters; contract v2; mirrored right-slingshot grid);
* the #113 capture-window spec (run to engine rest plus a 50-frame tail, 12 s cap).

Work item 1 (power analysis) is computed here from the #96 member-level records
(zero engine seconds) and fixes the evaluation depth by a pre-declared rule. The plan
also fixes the capture player (the #113 window rebuild), the renderer (Mesa 26.1.2 /
LLVM 22.1.6 llvmpipe, the renderer of every retained frame), campaign limits, dispatch
order, stop rules, the rendered smoke gate (executed by ``scripts.run_power_cohort_smoke``)
and the Gate-B binding of the #112 v3 handoff.

Modes: ``--dry-run`` (derive, print, write nothing), ``--prepare`` (freeze),
``--validate`` (re-derive every field and re-check every binding).
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = "issue-104-power-cohort-v1"
SCHEMA = "issue_104_power_cohort_plan_v1"
OUTPUT = ROOT / ".local-artifacts" / IDENTITY
VALIDATION_COMMAND = "python -u -m scripts.prepare_power_cohort --validate"
SMOKE_RUNNER = "scripts/run_power_cohort_smoke.py"

COHORT_109 = ROOT / ".local-artifacts/issue-109-sealed-cohort-v1"
COHORT_110 = ROOT / ".local-artifacts/issue-110-novelty-coverage-v1"
WINDOW_113 = ROOT / ".local-artifacts/issue-113-capture-window-v1/spec.json"
SUMMARY_96 = ROOT / ".local-artifacts/issue-96-tau-ad-within-checkpoint-v1/summary.json"
PLAN_96 = ROOT / ".local-artifacts/issue-96-tau-ad-within-checkpoint-v1/plan.json"
HELDOUT_109 = ROOT / ".local-artifacts/issue-109-heldout-rescore-v1/summary.json"
GATE_B = ROOT / ".local-artifacts/issue-112-decision-dynamics-v3/gate_b_handoff.json"
SMOKE_109 = ROOT / ".local-artifacts/issue-109-capture-smoke-v1"
DEV = ROOT / ".local-artifacts/issue-104-capture-dev/summary.json"
PIPELINE = ROOT / "scripts/capture_pipeline_v2.py"
MESA_PREFIX = Path.home() / ".cache/novphy-mesa/mesa-26.1.2-llvm22.1.6"
MESA_FILES = ("lib/libGLX_mesa.so.0.0.0", "lib/libgallium-26.1.2.so", "lib/dri/libdril_dri.so")
LLVM_LIBRARY = Path.home() / ".cache/novphy-mesa/env/lib/libLLVM.so.22.1"

DELTAS = (0.02, 0.05)
Z = 1.959964
PRIMARY = ("grid", "J-F*")
SECONDARY_INVENTORIES = ("offset", "angle", "power")
PRIMARY_DELTA = 0.02
SECONDARY_DELTA = 0.05
DEPTH_ROUNDING = 20
FIT_DEPTH, POLICY_DEPTH = 240, 60                 # full #109 fit / policy rosters per family
NOVELTY_PREFIX = 8                                 # #110 recommended floor, fit and evaluation
WORKERS = 16
COLD_START_SECONDS = 11.0                          # boot + menu inside the cold-start gate (dev runs: 11.0 s)
ARTIFACT_BYTES_PER_BRANCH = 100_000_000            # dev runs ~99 MB unique per branch (tail included)
REPLAY_PER_STOP_KIND = 2
SMOKE_CANDIDATES = ("c05", "c10")
ROLES = {"issue109-evaluation": "final_evaluation", "issue110-evaluation": "final_evaluation",
         "issue110-fit": "training", "issue109-fit": "training", "issue109-policy": "training",
         "smoke-novelty": "calibration", "smoke-replay": None}
TOKENS = ("supported", "not_supported_by_this_experiment", "readiness_or_precision_insufficient")
RENDERER = {"renderer": "llvmpipe (LLVM 22.1.6, 256 bits)", "version": "4.6 (Core Profile) Mesa 26.1.2"}


def log(message):
    print(f"[{IDENTITY}] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def file_sha256(path):
    return "sha256:" + sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return "sha256:" + sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def relative(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


# ------------------------------------------------------------ power analysis

def power_analysis():
    """Members needed per inventory for a member-clustered half-width <= delta (#104 work item 1).

    Two estimators from the #96 member-level records: (a) the observed bootstrap half-width
    scaled by 1/sqrt(members): m = m_obs (h_obs / delta)^2; (b) the normal approximation on
    member means: m = (z sd / delta)^2. The design uses the larger of the two.
    """
    import numpy as np
    from scripts import run_tau_ad_within_checkpoint as t96
    sources = t96.load_sources()
    _, universe, _ = t96.build_universe(sources)
    summary = read(SUMMARY_96)
    held_out = read(HELDOUT_109)["cohorts"]["held_out"]
    table, shares = {}, {}
    for inventory in t96.INVENTORIES:
        rows = t96.inventory_rows(t96.OUTPUT, universe, inventory)
        block = {}
        for name, contrast in sorted(summary["inventories"][inventory]["contrasts"].items()):
            per_member = {}
            for row in rows:
                first = row["arms"][contrast["first"]]["auc"]
                second = row["arms"][contrast["second"]]["auc"]
                if first is not None and second is not None:
                    per_member.setdefault(row["member"], []).append(first - second)
            means = [float(np.mean(values)) for values in per_member.values()]
            members, spread = len(means), float(np.std(means, ddof=1))
            half = contrast["estimate"]["half_width"]
            if members != contrast["members"]:
                raise ValueError(f"{inventory} {name}: member count differs from the #96 summary")
            block[name] = {
                "first": contrast["first"], "second": contrast["second"], "members": members,
                "cells": contrast["cells"], "estimate": round(contrast["estimate"]["mean"], 6),
                "half_width": round(half, 6), "member_sd": round(spread, 6),
                "required_mixed_members": {str(delta): {
                    "bootstrap_scaling": math.ceil(members * (half / delta) ** 2),
                    "normal_approximation": math.ceil((Z * spread / delta) ** 2),
                    "design": max(math.ceil(members * (half / delta) ** 2), math.ceil((Z * spread / delta) ** 2)),
                } for delta in DELTAS},
            }
        table[inventory] = block
        entries = universe[inventory]
        development = [m for m in entries if m not in held_out]
        holdout = [m for m in entries if m in held_out]
        share = {"development": {"members": len(development),
                                 "mixed": sum(t96.mixed(entries[m]) for m in development)},
                 "held_out": {"members": len(holdout), "mixed": sum(t96.mixed(entries[m]) for m in holdout)}}
        for value in share.values():
            value["share"] = round(value["mixed"] / value["members"], 4)
        share["conservative"] = min(share["development"]["share"], share["held_out"]["share"])
        shares[inventory] = share
    return {"method": ("required mixed members for a member-clustered half-width <= delta; (a) bootstrap scaling "
                       "m = m_obs (h_obs / delta)^2, (b) normal approximation m = (1.96 sd(member means) / delta)^2; "
                       "design = max(a, b)"),
            "source": {"summary": relative(SUMMARY_96), "records": "issue-96 records (member x seed x arm)",
                       "interval": "#96 member-clustered percentile bootstrap, 10000 draws, PCG64 7201"},
            "contrasts": table, "mixed_shares": shares,
            "held_out_members": held_out,
            "label": "DESCRIPTIVE planning numbers; development variance; not a verdict"}


def round_up(value, step=DEPTH_ROUNDING):
    return int(math.ceil(value / step) * step)


def depth(power, roster=200):
    """Pre-declared depth rule (levels per family, two families)."""
    grid = power["contrasts"]["grid"]
    grid_need = max(item["required_mixed_members"][str(PRIMARY_DELTA)]["design"] for item in grid.values())
    grid_share = power["mixed_shares"]["grid"]["conservative"]
    grid_levels = min(roster, round_up(grid_need / (2 * grid_share)))
    secondary = {}
    for inventory in SECONDARY_INVENTORIES:
        need = max(item["required_mixed_members"][str(SECONDARY_DELTA)]["design"]
                   for item in power["contrasts"][inventory].values())
        share = power["mixed_shares"][inventory]["conservative"]
        secondary[inventory] = {"required_mixed_members": need, "conservative_share": share,
                                "levels_per_family": math.ceil(need / (2 * share))}
    union_levels = min(roster, round_up(max(item["levels_per_family"] for item in secondary.values())))
    primary_need = grid[PRIMARY[1]]["required_mixed_members"][str(PRIMARY_DELTA)]["design"]
    return {
        "rule": ("grid (primary, 16 candidates): the deepest roster prefix needed for every #96 grid contrast at "
                 f"delta {PRIMARY_DELTA}; union (angle + grid + offset + power, 60 candidates): the prefix needed for "
                 f"every #96 contrast of offset, angle and power at delta {SECONDARY_DELTA}; levels per family = "
                 "required mixed members / (2 families x conservative mixed share), rounded up to 20, capped at the "
                 "roster; conservative share = min(development, held-out) member share"),
        "grid": {"required_mixed_members_max": grid_need, "primary_required_mixed_members": primary_need,
                 "conservative_share": grid_share, "levels_per_family": grid_levels},
        "secondary": secondary,
        "union_levels_per_family": union_levels,
        "expected_mixed_members": {
            "grid": round(2 * grid_levels * grid_share, 1),
            **{inventory: round(2 * union_levels * value["conservative_share"], 1)
               for inventory, value in secondary.items()}},
        "fit_levels_per_family": FIT_DEPTH, "policy_levels_per_family": POLICY_DEPTH,
        "fit_policy_rationale": ("no power target; the full #109 fit (240) and policy (60) rosters, the "
                                 "Gate-B retrain and controller data, fit the storage budget"),
        "novelty_levels_per_split_per_side": NOVELTY_PREFIX,
        "novelty_rationale": ("the #110 recommended floor: >= 8 clusters per cell interval and 40 levels per "
                              "novelty level for #107; storage keeps a > 2 TiB margin at this depth"),
    }


# ----------------------------------------------------------------- streams

def candidate_suffixes(plan109):
    sets = plan109["candidate_inventories"]["sets"]
    evaluation = sets["evaluation"]
    return {"union": [c["suffix"] for c in evaluation["candidates"]],
            "grid": list(evaluation["membership"]["grid"]),
            "training": [c["suffix"] for c in sets["training"]["candidates"]]}


def stream_levels(plan109, plan110, depths):
    """Ordered (stream, level side, base cluster, suffix list) rows; dispatch is in this order."""
    suffixes = candidate_suffixes(plan109)
    families = list(plan109["splits"]["evaluation"])
    rows = []
    for ordinal in range(1, 201):
        for family in families:
            level = plan109["splits"]["evaluation"][family][ordinal - 1]
            if ordinal <= depths["grid"]["levels_per_family"]:
                chosen = suffixes["union"] if ordinal <= depths["union_levels_per_family"] else suffixes["grid"]
                rows.append(("issue109-evaluation", level, level["identity"], chosen))
    pairs = [pair["pair"] for pair in plan110["capture_membership"]["pairs"]]
    for split in ("evaluation", "fit"):
        for ordinal in range(1, NOVELTY_PREFIX + 1):
            for pair in pairs:
                level = plan110["rosters"][split][pair][ordinal - 1]
                for side in ("normal", "novel"):
                    item = level[side]
                    rows.append((f"issue110-{split}", {**item, "engine_seed": level["engine_seed"],
                                                      "generation_seed": level["generation_seed"],
                                                      "ordinal": ordinal},
                                 f"issue-110-{split}-{item['generator_family']}-{ordinal:03}",
                                 [c["suffix"] for c in plan110["candidates"]["by_slingshot_side"][item["slingshot_side"]]]))
    for split, count in (("fit", FIT_DEPTH), ("policy", POLICY_DEPTH)):
        for ordinal in range(1, count + 1):
            for family in families:
                level = plan109["splits"][split][family][ordinal - 1]
                rows.append((f"issue109-{split}", level, level["identity"], suffixes["training"]))
    return rows


def actions_by_suffix(plan109, plan110):
    by_109 = {c["suffix"]: c for c in plan109["candidate_inventories"]["sets"]["evaluation"]["candidates"]}
    by_110 = {side: {c["suffix"]: c for c in items}
              for side, items in plan110["candidates"]["by_slingshot_side"].items()}
    return by_109, by_110


def branch_rows(plan109, plan110, depths):
    by_109, by_110 = actions_by_suffix(plan109, plan110)
    rows = []
    for stream, level, cluster, suffixes in stream_levels(plan109, plan110, depths):
        for suffix in suffixes:
            candidate = (by_110[level["slingshot_side"]] if stream.startswith("issue110") else by_109)[suffix]
            rows.append({"stream": stream, "identity": f"{level['identity']}-{suffix}",
                         "level_identity": level["identity"], "base_cluster": cluster,
                         "engine_seed": level["engine_seed"], "action": candidate["action"]})
    return rows


def membership(plan109, plan110, depths):
    rows = branch_rows(plan109, plan110, depths)
    identities = [row["identity"] for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("campaign branch identities are not unique")
    streams = {}
    for row in rows:
        entry = streams.setdefault(row["stream"], {"branches": 0, "levels": set(), "role": ROLES[row["stream"]]})
        entry["branches"] += 1
        entry["levels"].add(row["level_identity"])
    return {"order": list(streams),
            "streams": {name: {"branches": value["branches"], "levels": len(value["levels"]),
                               "exposure_role": value["role"]} for name, value in streams.items()},
            "branches": len(rows), "levels": len({row["level_identity"] for row in rows}),
            "branch_digest": digest(rows),
            "identity_rule": "<level identity>-<candidate suffix>; one identity per (level, engine seed, action)",
            "dispatch": ("streams in order; inside a stream ordinal-major (every family / pair at ordinal k before "
                         "k + 1; #110 sides normal then novel); candidates in suffix order; a stopped campaign "
                         "therefore keeps every stream balanced across families, pairs and cells")}


# --------------------------------------------------------------- dispatch

def templates_109():
    from scripts import prepare_issue_77_n1 as n1
    return n1.template_sources()


def templates_110():
    from scripts import prepare_novelty_coverage as nc
    return {entry["template"]: entry for entry in nc.load_inventory()["templates"]}


def level_files(level, stream):
    root = COHORT_110 if stream.startswith(("issue110", "smoke-novelty")) else COHORT_109
    xml = (root / level["xml_path"]).read_text()
    scenario_text = (root / level["scenario_path"]).read_text()
    if sha256(xml.encode()).hexdigest() != level["xml_sha256"]:
        raise ValueError(f"{level['identity']}: level xml differs from its frozen sha256")
    scenario = json.loads(scenario_text)
    from scripts import prepare_issue_77_n1 as n1
    if n1._digest(scenario) != level["scenario_sha256"]:
        raise ValueError(f"{level['identity']}: scenario differs from its frozen sha256")
    return xml, scenario


def dispatch_record(stream, level, cluster, candidate, template):
    xml, scenario = level_files(level, stream)
    return {"identity": f"{level['identity']}-{candidate['suffix']}", "source_member_identity": level["identity"],
            "stream": stream, "base_cluster": cluster, "exposure_role": ROLES[stream],
            "generation_seed": level["generation_seed"], "engine_seed": level["engine_seed"],
            "novelty_level": level["novelty_level"], "generator_family": level["generator_family"],
            "template": template, "xml": xml, "scenario": scenario,
            "generated_slots": list(level["generated_slots"]), "actions": [dict(candidate["action"])],
            "candidate_suffix": candidate["suffix"], "inventory_entries": candidate["inventory_entries"],
            "maximum_shots": 1}


def campaign_records(plan, streams=None):
    """Dispatch records of the frozen campaign, in dispatch order (optionally some streams only)."""
    plan109, plan110 = read(COHORT_109 / "plan.json"), read(COHORT_110 / "plan.json")
    by_109, by_110 = actions_by_suffix(plan109, plan110)
    t109, t110 = templates_109(), templates_110()
    for stream, level, cluster, suffixes in stream_levels(plan109, plan110, plan["depth"]):
        if streams is not None and stream not in streams:
            continue
        for suffix in suffixes:
            if stream.startswith("issue110"):
                yield dispatch_record(stream, level, cluster, by_110[level["slingshot_side"]][suffix],
                                      t110[level["template"]])
            else:
                yield dispatch_record(stream, level, cluster, by_109[suffix], t109[level["generator_family"]])


# ------------------------------------------------------------------ smoke

def replay_branches():
    """Exposed #77 N1 branches re-captured for renderer, window and determinism evidence."""
    outcomes = read(SMOKE_109 / "outcomes.json")["outcomes"]
    chosen = []
    for kind in ("stable_without_clear", "right_censored", "native_fail", "native_clear"):
        items = sorted(identity for identity, value in outcomes.items() if value["stop_kind"] == kind)
        dev = set()
        if DEV.is_file():
            for run in read(DEV)["runs"].values():
                dev |= set(run["branch_identities"])
        chosen += [identity for identity in items if identity not in dev][:REPLAY_PER_STOP_KIND]
    return chosen


def smoke_records(plan):
    """The frozen smoke: #110 smoke roster x c05, c10 (both sides) + the exposed replay branches."""
    plan110 = read(COHORT_110 / "plan.json")
    by_110 = {side: {c["suffix"]: c for c in items}
              for side, items in plan110["candidates"]["by_slingshot_side"].items()}
    t110 = templates_110()
    for pair in plan110["capture_membership"]["pairs"]:
        level = plan110["rosters"]["smoke"][pair["pair"]][0]
        for side in ("normal", "novel"):
            item = {**level[side], "engine_seed": level["engine_seed"], "generation_seed": level["generation_seed"]}
            for suffix in SMOKE_CANDIDATES:
                yield dispatch_record("smoke-novelty", item, f"issue-110-smoke-{item['generator_family']}-001",
                                      by_110[item["slingshot_side"]][suffix], t110[item["template"]])
    from scripts import run_capture_reliability_smoke as smoke109
    records = {record["identity"]: record for record in smoke109.records()}
    for identity in plan["smoke"]["replay_branches"]:
        yield {**records[identity], "stream": "smoke-replay"}


def smoke_spec():
    plan110 = read(COHORT_110 / "plan.json")
    novelty = list(plan110["smoke_branches"])
    replay = replay_branches()
    return {
        "runner": SMOKE_RUNNER,
        "validation_command": "python -u -m scripts.run_power_cohort_smoke --validate",
        "output": ".local-artifacts/issue-104-capture-smoke-v1",
        "media": "data/issue-104-capture-smoke",
        "novelty_branches": novelty, "replay_branches": replay,
        "branches": len(novelty) + len(replay),
        "membership_rule": ("#110 smoke roster (1 paired level per pair, both sides) x candidates c05, c10 = the "
                            "#110 frozen smoke gate; plus 2 exposed #77 N1 branches per retained #109 stop kind "
                            "(first in identity order, excluding the engineering-run branches) re-captured against "
                            "the retained #109 smoke"),
        "workers": WORKERS,
        "gate": {
            "S1_capture_reliability": "typed capture failures over all smoke branches <= 0.05 (the #109 gate)",
            "S2_renderer": f"every complete branch logs the frozen renderer {RENDERER} (enforced per branch)",
            "S3_pixel_identity": ("every replay branch's decision frame (paused-before.png) equals the retained #109 "
                                  "frame outside the animated Bird/Pig boxes"),
            "S4_window_rule": ("every complete branch declares (30000, 2500); every trace whose first post-launch "
                               "stable_entered lies at or before the cap ends with rest_tail_complete exactly "
                               "2500 steps after its last stable_entered, or with level_clear / level_fail / "
                               "censoring after a stable_exited; every other trace ends as before (clear, fail, or "
                               "censored exactly at the cap)"),
        },
        "novelty_template_criteria": list(plan110["protocols"]["smoke_gate"]["pass_criteria_per_template"]),
        "novelty_template_rules": {
            "force_sign": ("over contact-free consecutive native steps (no contact ids on either step) where force_on "
                           "is active, the median projection of the residual acceleration (velocity change / 0.0004 s "
                           "minus gravity x gravity_scale) on the unit force direction is > 0; a template with no such "
                           "step is reported 'not exercised', not failed"),
            "gravity": "world gravity_vector of the first native sample: (0, 6) on novelty level 6, (0, -9.8) otherwise",
            "right_slingshot": "bird_launched payload launch_velocity x < 0 on every right-slingshot branch",
            "slots": "every generated slot (novelty:0000 included) is in the initial trace (pipeline check)",
        },
        "failure_effect": ("S1-S4 failing: the campaign is not launched (readiness_or_precision_insufficient). A #110 "
                           "template failing a criterion makes its cell unsupported with that typed reason; the "
                           "campaign dispatches no branch of an unsupported cell and records them as "
                           "unsupported_by_smoke"),
        "reported_not_gated": ["storm onset inside the window (level 8)", "force-on prevalence per template",
                               "settle frames (post-rest frames) and collapse (non-bird support change) per level type",
                               "replay native-trace prefix identity and stop kinds vs the retained #109 smoke",
                               "throughput (amortized s/branch, branch wall, unique bytes) and the campaign "
                               "budget re-projected with it"],
        "retry_policy": "technical_retries 0; every scheduled smoke branch executed once or retained as a typed failure",
    }


# ------------------------------------------------------------------ player

def player_identity():
    identity = read(OUTPUT / "player-identity.json") if (OUTPUT / "player-identity.json").is_file() else None
    if identity is None:
        raise ValueError("assemble the campaign player first (prepare does it)")
    actual = sha256((OUTPUT / "player/9001_Data/Managed/Assembly-CSharp.dll").read_bytes()).hexdigest()
    if actual != identity["assembly_sha256"]:
        raise ValueError("campaign player assembly differs from its identity record")
    return identity


def renderer_block():
    files = {name: file_sha256(MESA_PREFIX / name) for name in MESA_FILES}
    files["libLLVM.so.22.1"] = file_sha256(LLVM_LIBRARY)
    return {
        "expected": RENDERER,
        "engine_environment": {"LD_LIBRARY_PATH": str(MESA_PREFIX / "lib"),
                               "LIBGL_DRIVERS_PATH": str(MESA_PREFIX / "lib/dri"),
                               "__GLX_VENDOR_LIBRARY_NAME": "mesa"},
        "library_sha256": files,
        "build": "scripts/build_capture_mesa.sh (Mesa 26.1.2 source sha256 bac2bca9..., conda-forge llvmdev 22.1.6)",
        "reason": ("every retained frame (#77, #87, #89, #93, #94, #109) was rendered by llvmpipe (LLVM 22.1.6) / "
                   "Mesa 26.1.2; the server's Mesa 25.2.8 / LLVM 20.1.2 differs on 230-872 edge pixels of every "
                   "decision frame; the local build reproduces the retained decision frames byte for byte "
                   "(engineering runs, dev summary)"),
    }


# ------------------------------------------------------------------ budget

def budget(members, dev):
    branches = members["branches"]
    wall = dev["runs"]["dev1"]["branch_wall_excluding_gate_wait_mean"]
    amortized = max(COLD_START_SECONDS, wall / WORKERS)
    total_bytes = branches * ARTIFACT_BYTES_PER_BRANCH
    return {
        "workers": WORKERS,
        "throughput_model": ("engine cold starts are serialized (cold-start gate), so amortized seconds per branch "
                             "= max(cold-start critical section, branch wall / workers)"),
        "cold_start_critical_section_seconds": COLD_START_SECONDS,
        "branch_wall_seconds_excluding_gate": wall,
        "amortized_seconds_per_branch": round(amortized, 2),
        "artifact_bytes_per_branch": ARTIFACT_BYTES_PER_BRANCH,
        "branches": branches,
        "projected_wall_hours": round(branches * amortized / 3600, 1),
        "projected_worker_hours": round(branches * wall / 3600, 1),
        "projected_artifact_tib": round(total_bytes / 2**40, 2),
        "per_stream": {name: {"branches": value["branches"],
                              "wall_hours": round(value["branches"] * amortized / 3600, 1),
                              "artifact_tib": round(value["branches"] * ARTIFACT_BYTES_PER_BRANCH / 2**40, 3)}
                       for name, value in members["streams"].items()},
        "source": "EXPLORATORY engineering runs (dev summary); the rendered smoke re-measures and re-projects",
    }


def limits(members, renderer, window):
    from scripts import capture_pipeline_v2 as pipeline
    return {**pipeline.DEFAULT_LIMITS,
            "workers": WORKERS, "attempt_seconds": 1800, "worker_cpu_rss_mib": 4096,
            "aggregate_cpu_rss_mib": 65536, "minimum_free_bytes": 500 * 2**30,
            "artifact_bytes": int(1.25 * members["branches"] * ARTIFACT_BYTES_PER_BRANCH),
            "decision_fixed_step": 30000,
            "native_window": {"maximum_shot_steps": window["rule"]["pre_rest_cap_steps"],
                              "rest_tail_steps": window["rule"]["rest_tail_steps"]},
            "engine_environment": renderer["engine_environment"], "renderer": renderer["expected"]}


# ------------------------------------------------------------------- plan

def inputs():
    items = [("issue109_sealed_cohort_plan", COHORT_109 / "plan.json"),
             ("issue110_novelty_coverage_plan", COHORT_110 / "plan.json"),
             ("issue113_capture_window_spec", WINDOW_113),
             ("issue96_summary", SUMMARY_96), ("issue96_plan", PLAN_96),
             ("issue109_heldout_rescore_summary", HELDOUT_109),
             ("issue109_capture_smoke_outcomes", SMOKE_109 / "outcomes.json"),
             ("issue112_v3_gate_b_handoff", GATE_B),
             ("issue104_capture_dev_summary", DEV),
             ("capture_pipeline_v2", PIPELINE),
             ("window_player_stage", ROOT / "scripts/issue_104_window_player.py"),
             ("mesa_build_script", ROOT / "scripts/build_capture_mesa.sh"),
             ("canonical_native_trace", ROOT / "scripts/canonical_native_trace.py"),
             ("native_segment_trace", ROOT / "scripts/native_segment_trace.py"),
             ("issue76_expansion", ROOT / "scripts/issue_76_expansion.py")]
    return [{"name": name, "artifact": relative(path), "sha256": file_sha256(path)} for name, path in items]


def gate_b():
    handoff = read(GATE_B)
    return {
        "handoff": relative(GATE_B), "handoff_sha256": file_sha256(GATE_B),
        "plan_identity": handoff["plan_identity"],
        "unit": {"arm": handoff["recipe"]["arm"], "request": handoff["request"], "family": handoff["family"],
                 "encoder": handoff["encoder"]["identity"], "usable_seeds": handoff["usable_seeds"]},
        "target": handoff["gate"],
        "scope_rule": handoff["scope"]["rule"],
        "pre_declared_contrast": handoff["scope"]["baseline_contrast"],
        "reported_controls": handoff["scope"]["reported_controls"],
        "controls_runner": "scripts/run_decision_dynamics_controls.py",
        "evaluation": ("the #109 evaluation split captured by this campaign, grid inventory (16 candidates), mixed "
                       "cells only, tie-free cost, scoring_harness.deterministic_scoring() in a process that did "
                       "not train; never fitted"),
        "retrain": handoff["retrain"] + "; a retrain is a separately frozen follow-up, not part of this plan",
        "status": "bound, not executed: Gate B runs after the campaign on its frozen evaluation captures",
    }


def derive(player=True):
    plan109, plan110 = read(COHORT_109 / "plan.json"), read(COHORT_110 / "plan.json")
    window = read(WINDOW_113)
    dev = read(DEV)
    power = power_analysis()
    depths = depth(power)
    members = membership(plan109, plan110, depths)
    renderer = renderer_block()
    budget_block = budget(members, dev)
    plan = {
        "schema": SCHEMA, "identity": IDENTITY, "issue": 104, "version": 1, "role": "terminal",
        "validation_command": VALIDATION_COMMAND, "frozen_before_any_capture": True,
        "engine_seconds_before_freeze": ("0 on any cohort level; engineering runs touched exposed #77 N1 "
                                         "branches only (EXPLORATORY, dev summary)"),
        "inputs": inputs(),
        "power": power,
        "depth": depths,
        "membership": members,
        "cohorts": {
            "issue109": {"plan": relative(COHORT_109 / "plan.json"), "families": list(plan109["splits"]["evaluation"]),
                         "candidate_sets": {name: len(value) for name, value in candidate_suffixes(plan109).items()},
                         "evaluation_rule": "evaluation captures are never fitted; #96 contrasts rerun on them only"},
            "issue110": {"plan": relative(COHORT_110 / "plan.json"), "pairs": len(plan110["capture_membership"]["pairs"]),
                         "cells": plan110["cell_counts"], "contract": plan110["contract"]["identity"],
                         "cell_rule": ("all 45 cells are captured through the 40 pairs unless the smoke makes a cell "
                                       "unsupported; zero-shot (normal fit) and few-shot (novel fit) never pooled"),
                         "fit_merge": ("training pool = #109 fit split + #110 fit normal sides (zero-shot) ; the "
                                       "#110 novel fit sides are few-shot adaptation data only"),
                         "flags": ["multiple-forces cells (9) carry no single-shot pig-removal headroom",
                                   "storm onset needs the first bird destroyed; reported by the smoke"]},
        },
        "window": {"spec": relative(WINDOW_113), "spec_sha256": file_sha256(WINDOW_113), "rule": window["rule"],
                   "merge": "the #113 rule is the campaign's engine window (limits.native_window, player rebuild)"},
        "player": player_identity() if player else "assembled by --prepare",
        "renderer": renderer,
        "pipeline": {"module": relative(PIPELINE), "sha256": file_sha256(PIPELINE),
                     "identity": "capture-pipeline-v2"},
        "limits": limits(members, renderer, window),
        "budget": budget_block,
        "stop_rules": {
            "S1_failure_rate": "after >= 200 dispatched branches, cumulative typed-failure share > 0.05 stops the campaign",
            "S2_storage": "free bytes on the output filesystem < 500 GiB (limits.minimum_free_bytes)",
            "S3_artifact_cap": "accounted unique bytes > limits.artifact_bytes (1.25 x projection)",
            "S4_wall_cap_hours": round(2 * budget_block["projected_wall_hours"], 1),
            "effect": ("a stop is terminal: every undispatched branch is recorded as campaign_stopped:<rule>; no "
                       "outcome-conditioned retry, replacement or re-freeze"),
        },
        "smoke": smoke_spec(),
        "gate_b": gate_b(),
        "hardware": ("4x RTX 5090 server (48 threads); capture renders on the CPU (llvmpipe), no GPU; every later GPU "
                     "stage runs on one device (CUDA_VISIBLE_DEVICES pinned): the #85 single-RTX-3090 gate is "
                     "carried as a single-GPU gate on this server (disclosed deviation)"),
        "disclosures": [
            "SERVER_SETUP.md byte-rewrote /p/Project/NovPhy in frozen records (owner decision); the #109 sealed-cohort "
            "--validate now fails on its #96 input hash. Inputs here are bound to the current bytes; level files are "
            "re-checked against their frozen sha256 at every dispatch",
            "physics is not bit-identical across machines: engineering re-captures on this server agree with the "
            "retained #109 smoke on decision frames and on 16/20 event sequences, but differ in float digits from "
            "the bird's first collision; within this server the parent and window players agree on 19/20 full "
            "native prefixes",
            "the #113 tail is cut by level_fail when the launched bird dies after rest (single-bird levels); "
            "engineering runs kept 1.7-46 tail frames; the smoke reports the realized settle frames",
        ],
        "tokens": list(TOKENS),
        "claim_boundary": ("design freeze and rendered smoke only; no outcome of any cohort level is read before "
                           "the campaign; prior dispositions (#15-#113) are read-only; nothing edits the ICLR 2026 "
                           "submission"),
    }
    return plan


def findings(plan):
    power, depths, members, budget_block = plan["power"], plan["depth"], plan["membership"], plan["budget"]
    lines = [f"# #104 joint cohort plan (`{IDENTITY}`)", "",
             f"Frozen {plan.get('frozen_at', '(unfrozen)')}, before any capture. `{VALIDATION_COMMAND}`.", "",
             "## Power (from #96 member-level records; required mixed members, design = max of two estimators)", "",
             "| inventory | contrast | members | estimate | half-width | sd | delta 0.02 | delta 0.05 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for inventory, block in power["contrasts"].items():
        for name, row in block.items():
            need = row["required_mixed_members"]
            lines.append(f"| {inventory} | {name} | {row['members']} | {row['estimate']:+.4f} | {row['half_width']:.4f} | "
                         f"{row['member_sd']:.4f} | {need['0.02']['design']} | {need['0.05']['design']} |")
    lines += ["", "| inventory | development mixed | held-out mixed | conservative share |", "| --- | --- | --- | --- |"]
    for inventory, share in power["mixed_shares"].items():
        lines.append(f"| {inventory} | {share['development']['mixed']}/{share['development']['members']} | "
                     f"{share['held_out']['mixed']}/{share['held_out']['members']} | {share['conservative']} |")
    lines += ["", "## Depth", "", f"- rule: {depths['rule']}",
              f"- grid: {depths['grid']['levels_per_family']} levels per family (needs {depths['grid']['required_mixed_members_max']} "
              f"mixed members; primary J-F* needs {depths['grid']['primary_required_mixed_members']})",
              f"- union inventories: first {depths['union_levels_per_family']} levels per family",
              f"- expected mixed members: {depths['expected_mixed_members']}",
              f"- fit {depths['fit_levels_per_family']} / policy {depths['policy_levels_per_family']} per family; "
              f"#110 {depths['novelty_levels_per_split_per_side']} per split per side", "",
              "## Membership (dispatch order)", "", "| stream | role | levels | branches | wall h | TiB |",
              "| --- | --- | --- | --- | --- | --- |"]
    for name in members["order"]:
        stream, cost = members["streams"][name], budget_block["per_stream"][name]
        lines.append(f"| {name} | {stream['exposure_role']} | {stream['levels']} | {stream['branches']} | "
                     f"{cost['wall_hours']} | {cost['artifact_tib']} |")
    lines += [f"| total | | {members['levels']} | {members['branches']} | {budget_block['projected_wall_hours']} | "
              f"{budget_block['projected_artifact_tib']} |", "",
              f"Workers {budget_block['workers']}; amortized {budget_block['amortized_seconds_per_branch']} s/branch "
              f"(cold-start bound); worker-hours {budget_block['projected_worker_hours']}.", "",
              "## Window (#113)", ""] + [f"- {key}: {value}" for key, value in plan["window"]["rule"].items()
                                          if key in ("name", "pre_rest_cap_steps", "rest_tail_frames", "tail",
                                                     "tail_cancel", "clear_and_fail", "censoring")]
    lines += ["", "## Capture stack", "",
              f"- player assembly `{plan['player']['assembly_sha256'][:16]}...` (parent 82db3f42 + #113 window)",
              f"- renderer {plan['renderer']['expected']}", f"- pipeline `{plan['pipeline']['sha256'][:23]}...`",
              "", "## Smoke gate", ""] + [f"- **{key}**: {value}" for key, value in plan["smoke"]["gate"].items()]
    lines += [f"- membership: {plan['smoke']['branches']} branches ({len(plan['smoke']['novelty_branches'])} novelty + "
              f"{len(plan['smoke']['replay_branches'])} replay)", "", "## Gate B (bound, not executed)", "",
              f"- unit {plan['gate_b']['unit']}", f"- target: {plan['gate_b']['target']}",
              f"- controls: {plan['gate_b']['reported_controls']}", "", "## Stop rules", ""]
    lines += [f"- {key}: {value}" for key, value in plan["stop_rules"].items()]
    lines += ["", "## Disclosures", ""] + [f"- {item}" for item in plan["disclosures"]] + [""]
    return "\n".join(lines)


def prepare():
    if (OUTPUT / "plan.json").exists():
        raise ValueError("plan.json is frozen; refusing to overwrite")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not (OUTPUT / "player").exists():
        from scripts import issue_104_window_player as window_player
        window_player.assemble(OUTPUT / "player")
    plan = derive()
    plan["frozen_at"] = utc_now()
    (OUTPUT / "plan.json").write_text(json_text(plan))
    (OUTPUT / "findings.md").write_text(findings(read(OUTPUT / "plan.json")))
    log(f"frozen {plan['frozen_at']}: {plan['membership']['branches']} branches, smoke {plan['smoke']['branches']}")


def load_plan():
    plan = read(OUTPUT / "plan.json")
    changed = [item["artifact"] for item in plan["inputs"] if file_sha256(ROOT / item["artifact"]) != item["sha256"]]
    if changed:
        raise ValueError(f"frozen inputs changed after the freeze: {changed}")
    return plan


def validate():
    frozen = read(OUTPUT / "plan.json")
    problems = []
    plan = derive()
    plan["frozen_at"] = frozen["frozen_at"]
    for key in sorted(set(plan) | set(frozen)):
        if plan.get(key) != frozen.get(key):
            problems.append(key)
    if (OUTPUT / "findings.md").read_text() != findings(frozen):
        problems.append("findings.md")
    report = {"identity": IDENTITY, "branches": frozen["membership"]["branches"], "problems": problems,
              "validated": not problems}
    print(json_text(report), end="")
    return not problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="derive and print the plan summary; writes nothing")
    mode.add_argument("--prepare", action="store_true", help="assemble the player and freeze the plan")
    mode.add_argument("--validate", action="store_true", help="re-derive the plan and re-check every binding")
    args = parser.parse_args()
    if args.dry_run:
        plan = derive(player=(OUTPUT / "player-identity.json").is_file())
        print(json_text({"membership": plan["membership"], "depth": plan["depth"], "budget": plan["budget"],
                         "smoke_branches": plan["smoke"]["branches"]}), end="")
    elif args.prepare:
        prepare()
    elif not validate():
        sys.exit(1)


if __name__ == "__main__":
    main()
