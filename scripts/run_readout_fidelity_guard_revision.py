"""Issue-100 v2: guard revision of the v1 readout-fidelity / JEPA publication (terminal).

Binding runner module: scripts/run_readout_fidelity_guard_revision.py
Exact validation command: python -u -m scripts.run_readout_fidelity_guard_revision --validate

The v1 publication (issue-100-readout-fidelity-jepa-v1) returned
readiness_or_precision_insufficient for Q2 (readout target) and Q4 (J - T, J - D)
because two replication guards failed for reasons unrelated to what they protect:

- G1 demanded every F77 audit carrier value within 1e-3 of the #77 shard carrier. It
  protects "the audit reads the carriers the #77 model was trained on". Every
  exceedance is a motion value (normalized center / elapsed seconds): parser batch-size
  float noise of the centers, divided by the frame interval (0.087 s, and as little as
  a few ms on the terminal frame of early-ending shots), reached 0.036.
- G3 demanded that the F77 arm records reproduce every #96 count cost within 1e-3. It
  protects "the arms here are the #96 arms". The run process had enabled cuDNN autotuning
  (cudnn.benchmark) for JEPA training, so its anchor parses used differently rounded
  convolution kernels; recursive Delta = 1 rollouts over 225 steps and the count cost's
  1000x pig weight amplified that float noise (worst 87.7). A fresh process with the
  same code reproduces the worst cells exactly.

v2 replaces exactly these two guard definitions, adds a deterministic replication of the
F77 arms (the evidence for revised G3) and one descriptive explanation of the probe's
macro numbers, and re-derives the tokens from the retained v1 records under the
unchanged v1 decision rules. v1 stays byte-identical and is re-validated by its own
--validate at every v2 publish and validate.

v2 was frozen AFTER every v1 outcome was known; that is disclosed in plan.json and in
every rendering. The revised tolerances come from the v1 parse tolerance (1e-3, frozen
before any outcome) and the carrier definition, not from any F1, AUC or contrast.
v2 is terminal: no further guard revision of #100.

Modes: --dry-run (no write, no statistic) --prepare --run --publish --validate.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from scripts import run_decision_chain_attribution as m99
from scripts import run_lookahead_control as wlc
from scripts import run_readout_fidelity_jepa as r100
from scripts import run_tau_ad_within_checkpoint as t96

ROOT = t96.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-100-guard-revision-v2"
V1 = r100.OUTPUT
IDENTITY = "issue-100-guard-revision-v2"
SCHEMA_PLAN = "issue_100_guard_revision_plan_v2"
SCHEMA_REPLICATION = "issue_100_guard_revision_replication_v2"
SCHEMA_PROBE = "issue_100_guard_revision_probe_v2"
SCHEMA_COMPUTE = "issue_100_guard_revision_compute_v2"
SCHEMA_REPORT = "issue_100_guard_revision_report_v2"
VALIDATION_COMMAND = "python -u -m scripts.run_readout_fidelity_guard_revision --validate"
RUNNER = "scripts/run_readout_fidelity_guard_revision.py"

STOP_TOKEN = r100.STOP_TOKEN
TOLERANCE = r100.PARSE_TOLERANCE
COST_TOLERANCE = r100.COST_TOLERANCE
PUBLISHED = ("plan.json", "smoke.json", "ledger.json", "compute.json", "summary.json", "findings.md",
             "comparisons.csv", "frequencies--F77.json", "frequencies--E99.json", "frequencies--JEPA.json",
             "records/probe.json")
RUNNERS = {"v1_runner": r100.RUNNER, "model_module": r100.MODEL_MODULE}
MOTION_FEATURES = (8, 9)  # ENTITY_FEATURES motion_x/y_normalized_per_second

read_json = t96.read_json
write_json = t96.write_json
json_text = t96.json_text
sha256_of = t96.sha256_of
fmt = t96.fmt
fmt_interval = t96.fmt_interval

GUARD_CHANGES = {
    "G1": {
        "v1": f"F77 audit carriers replicate every valid #77 shard carrier value within {TOLERANCE}",
        "v2": (f"every non-motion carrier value within {TOLERANCE}, and every motion value within "
               f"2 x {TOLERANCE} / elapsed seconds of its frame (a motion value is the difference of two "
               f"parsed centers, each within the {TOLERANCE} parse tolerance, divided by the frame interval)"),
        "reason": ("the guard protects 'the audit reads the #77 training carriers'; every v1 exceedance was a "
                   "motion value, where center float noise from a different parse batch is divided by the frame "
                   "interval (terminal frames of early-ending shots have intervals of a few milliseconds)"),
    },
    "G3": {
        "v1": (f"F77 arms replicate the #96 records (count cost of every candidate of every F/J/M/T/D record) "
               f"within {COST_TOLERANCE}, and the F77 T/D coordinates equal #96's"),
        "v2": (f"a fresh deterministic process (cudnn.benchmark off, cudnn.deterministic on, no prior GPU work) "
               f"running the v1 arm code for F77 on every #96 cell reproduces every #96 count cost within "
               f"{COST_TOLERANCE}, and the F77 T/D coordinates equal #96's (v1 observation, unchanged)"),
        "reason": ("the guard protects 'these arms are the #96 arms'; the v1 run process had turned on cuDNN "
                   "autotuning for JEPA training, so its anchor parses used differently rounded kernels and "
                   "recursive Delta = 1 rollouts plus the count cost's 1000x pig weight amplified the noise. "
                   "The v1 records stay the scored records; their divergence from #96 is reported descriptively"),
    },
}


def log(message):
    print(f"[issue-100-guard-revision] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def bindings():
    rows = [{"name": f"v1/{file}", "artifact": str((V1 / file).relative_to(ROOT)), "sha256": sha256_of(V1 / file)}
            for file in PUBLISHED]
    for name, path in RUNNERS.items():
        rows.append({"name": f"runner/{name}", "artifact": path, "sha256": sha256_of(ROOT / path)})
    return rows


def check_bindings(plan):
    for entry in plan["inputs"]:
        if sha256_of(ROOT / entry["artifact"]) != entry["sha256"]:
            raise ValueError(f"v1 artifact {entry['artifact']} changed after the v2 freeze")


# ---------------------------------------------------------------------------
# revised guards
# ---------------------------------------------------------------------------

def motion_columns():
    return sorted(2 + 13 * slot + feature for slot in range(18) for feature in MOTION_FEATURES)


def carrier_replication(shard_rows, fresh_rows, elapsed):
    """(max |delta| over non-motion values, max |delta| x elapsed / 2 over motion values)."""
    delta = (shard_rows - fresh_rows).abs()
    motion = torch.zeros(delta.shape[-1], dtype=torch.bool)
    motion[motion_columns()] = True
    nonmotion = float(delta[:, ~motion].max()) if len(delta) else 0.0
    scaled = float((delta[:, motion] * elapsed[:, None] / 2).max()) if len(delta) else 0.0
    return nonmotion, scaled


def revised_g1():
    shots = [s for s in r100.audit_shots() if s["cohort"] == "fit"]
    worst_nonmotion, worst_scaled, worst_raw, exceed = 0.0, 0.0, 0.0, 0
    for shot in shots:
        carriers = r100.load_carriers(V1, "F77", shot["branch"])
        labels = r100.load_labels(V1, shot["branch"])
        shard = torch.load(r100.DYNAMICS / "shards" / f"lineage-n1-{int(shot['member'].rsplit('-', 1)[1]):03d}.pt",
                           map_location="cpu", weights_only=True)
        branch_index = shard["record"]["branches"].index(shot["branch"])
        for w, window in enumerate(shard["windows"]):
            if window["shot"] != branch_index:
                continue
            length = int(shard["tensors"]["length"][w])
            rows = shard["tensors"]["z"][w, :length + 1]
            span = slice(window["start"], window["start"] + length + 1)
            nonmotion, scaled = carrier_replication(rows, carriers[span], labels["elapsed"][span].float())
            raw = float((rows - carriers[span]).abs().max())
            worst_nonmotion, worst_scaled = max(worst_nonmotion, nonmotion), max(worst_scaled, scaled)
            worst_raw = max(worst_raw, raw)
            exceed += int(raw > TOLERANCE)
    return {"observed": {"max_abs_nonmotion_delta": worst_nonmotion,
                         "max_motion_delta_times_elapsed_over_2": worst_scaled,
                         "v1_statistic_max_abs_delta": worst_raw, "windows_above_v1_tolerance": exceed,
                         "tolerance": TOLERANCE},
            "pass": worst_nonmotion <= TOLERANCE and worst_scaled <= TOLERANCE}


def replication_path(output, inventory, member, seed):
    return Path(output) / "records" / f"replication--F77--{inventory}--{member}--seed{seed}.json"


def replicate(output):
    """Deterministic fresh-process F77 arms (the v1 arm code) on every #96 cell."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    _, universe, _ = r100.fit_data()
    stack = r100.ArmStack(V1)
    coordinates = read_json(V1 / "frequencies--F77.json")["coordinates"]
    cells = t96.scheduled_cells(universe)
    progress = r100.Progress("replication F77", len(cells))
    gpu = 0.0
    for inventory, member, seed in cells:
        path = replication_path(output, inventory, member, seed)
        if path.is_file():
            progress.step(60)
            continue
        record = r100.arms_record(stack, "F77", seed, inventory, member, universe[inventory][member], r100.ARMS,
                                  coordinates[str(seed)])
        gpu += sum(block["gpu_seconds"] for block in record["arms"].values())
        write_json(path, {"schema": SCHEMA_REPLICATION, "plan_identity": IDENTITY,
                          "cell": record["cell"], "cudnn": {"benchmark": False, "deterministic": True},
                          "counts": {arm: {str(row["ordinal"]): row["count"] for row in block["rows"]}
                                     for arm, block in record["arms"].items()},
                          "gpu_seconds": sum(block["gpu_seconds"] for block in record["arms"].values()),
                          "engine_seconds": 0})
        progress.step(60)
    return gpu


def revised_g3(output, universe, v1_compute):
    worst, checked = 0.0, 0
    for inventory, member, seed in t96.scheduled_cells(universe):
        fresh = read_json(replication_path(output, inventory, member, seed))
        if fresh.get("schema") != SCHEMA_REPLICATION or fresh.get("plan_identity") != IDENTITY:
            raise ValueError("replication record binding differs")
        for arm in r100.ARMS:
            retained = t96.read_record(t96.OUTPUT, inventory, member, seed, arm)["decision"]["ranking"]
            for row in retained:
                mine = fresh["counts"][arm].get(str(row["ordinal"]))
                if (row["predicted_cost"] is None) != (mine is None):
                    worst = math.inf
                    continue
                if mine is None:
                    continue
                worst = max(worst, abs(mine - row["predicted_cost"]))
                checked += 1
    coordinates_equal = v1_compute["guards"]["G3"]["observed"]["coordinates_equal"]
    return {"observed": {"max_abs_count_cost_delta_fresh_vs_96": worst, "candidates_checked": checked,
                         "coordinates_equal": coordinates_equal,
                         "v1_run_max_abs_count_cost_delta": v1_compute["guards"]["G3"]["observed"][
                             "max_abs_count_cost_delta"], "tolerance": COST_TOLERANCE},
            "pass": worst <= COST_TOLERANCE and coordinates_equal}


def noise_impact(universe):
    """Descriptive: how far the v1 run's F77 records diverge from #96 in per-cell AUC."""
    per_arm = {}
    for arm in r100.ARMS:
        deltas, cost_deltas = [], []
        for inventory, member, seed in t96.scheduled_cells(universe):
            entry = universe[inventory][member]
            if not t96.mixed(entry):
                continue
            blocks = {}
            for phase in (1, 2):
                blocks.update(read_json(r100.arms_path(V1, phase, "F77", inventory, member, seed))["arms"])
            retained = t96.read_record(t96.OUTPUT, inventory, member, seed, arm)["decision"]["ranking"]
            theirs = [{"ordinal": row["ordinal"], "count": row["predicted_cost"]} for row in retained]
            a = m99.view(entry, blocks[arm]["rows"], "count")["auc"]
            b = m99.view(entry, theirs, "count")["auc"]
            if a is not None and b is not None:
                deltas.append(abs(a - b))
            mine = {row["ordinal"]: row["count"] for row in blocks[arm]["rows"]}
            cost_deltas.extend(abs(mine[row["ordinal"]] - row["predicted_cost"]) for row in retained
                               if row["predicted_cost"] is not None and mine.get(row["ordinal"]) is not None)
        per_arm[arm] = {"mixed_cells": len(deltas), "cells_with_auc_change": int(sum(d > 1e-12 for d in deltas)),
                        "max_abs_auc_change": float(max(deltas)) if deltas else None,
                        "mean_abs_auc_change": float(np.mean(deltas)) if deltas else None,
                        "candidates_above_cost_tolerance": int(sum(d > COST_TOLERANCE for d in cost_deltas)),
                        "candidates": len(cost_deltas)}
    return per_arm


def probe_path(output):
    return Path(output) / "records" / "probe-explanation.json"


@torch.no_grad()
def probe_explanation(output):
    """Descriptive: the probe's frame-80 macro values from a no-prior (single-frame) carrier."""
    from world_model.data.deployment_temporal import AgentObservation, TemporalObservationContext
    from scripts import run_slot_encoder_fix as s99
    torch.backends.cudnn.benchmark = False
    adapter = wlc.load_adapter(r100.DEVICE)
    _, root, frames, _ = s99.shot_frames(r100.CAMPAIGN / "attempts" / r100.PROBE["capture"])
    frame = frames[r100.PROBE["frame"]]
    observation = AgentObservation(frame["agent_observation"]["identity"], frame["fixed_step"],
                                   frame["fixed_time_seconds"],
                                   (root / frame["agent_observation"]["relative_path"]).read_bytes(), "agent")
    single = adapter.build_from_parsed(TemporalObservationContext(None, observation),
                                       adapter.parse_batch((observation,))[0], None).tensor[None].to(r100.DEVICE)
    rows = {}
    for seed in r100.SEEDS:
        predictor = r100.load_hybrid(V1, "F77", seed)
        micro, macro = r100.gated(predictor, single)
        rows[str(seed)] = {"gated_macro_no_prior": [float(v) for v in macro[0]],
                           "max_gated_micro_no_prior": float(micro.max())}
    write_json(probe_path(output), {"schema": SCHEMA_PROBE, "plan_identity": IDENTITY,
                                    "capture": r100.PROBE["capture"], "frame": r100.PROBE["frame"],
                                    "carrier": "single-frame parse, no prior frame (motion unavailable)",
                                    "per_seed": rows, "engine_seconds": 0})


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------

def compute(output, plan):
    check_bindings(plan)
    r100.validate(V1)
    v1 = read_json(V1 / "compute.json")
    _, universe, _ = r100.fit_data()
    g1 = revised_g1()
    g3 = revised_g3(output, universe, v1)
    guards = {}
    for name, block in v1["guards"].items():
        revised = {"G1": g1, "G3": g3}.get(name)
        guards[name] = {"v1_pass": block["pass"], "v1_observed": block["observed"],
                        "v2_pass": revised["pass"] if revised else block["pass"],
                        "v2_observed": revised["observed"] if revised else block["observed"],
                        "revised": revised is not None}
    v1_ok = v1["guards_pass"]
    v2_ok = all(g["v2_pass"] for g in guards.values())
    target = v1["readout_target"]
    q2_v2 = STOP_TOKEN if not v2_ok else ("supported" if target["met_by"] else "not_supported_by_this_experiment")
    alpha = v1["arms"]["alpha_effect"]
    q4 = {inventory: {name: {"estimate": block["estimate"], "cells": block["cells"], "members": block["members"],
                             "v1_token": block["token"], "v2_token": r100.contrast_token(block["estimate"], v2_ok)}
                      for name, block in row.items()} for inventory, row in alpha["inventories"].items()}
    probe = read_json(probe_path(output))
    replication_gpu = sum(read_json(replication_path(output, i, m, s))["gpu_seconds"]
                          for i, m, s in t96.scheduled_cells(universe))
    return {"schema": SCHEMA_COMPUTE, "identity": IDENTITY, "revision_of": r100.IDENTITY,
            "guards": guards, "guards_pass": {"v1": v1_ok, "v2": v2_ok},
            "Q2_readout_target": {"met_by": target["met_by"], "better_encoder": target["better_encoder"],
                                  "encoders": {e: {k: row[k] for k in ("macro_f1", "micro_edge_f1", "score", "meets")}
                                               for e, row in target["encoders"].items()},
                                  "v1_token": target["token"], "v2_token": q2_v2},
            "Q4_alpha_effect": {"encoder": alpha["encoder"], "cohort": alpha["cohort"], "cost": alpha["cost"],
                                "inventories": q4},
            "noise_impact_v1_run_vs_96": noise_impact(universe),
            "probe_explanation": {"claimed": {"steady": r100.PROBE["claims"]["C4_macro_steady_frame80"],
                                              "unstable": r100.PROBE["claims"]["C5_macro_unstable_frame80"]},
                                  "v1_observed_with_prior": v1["probe"]["record"]["gated_macro_frame"],
                                  "no_prior": probe["per_seed"]},
            "compute": {"replication_gpu_seconds": replication_gpu, "engine_seconds": 0}}


def frozen_plan(frozen_at):
    v1_plan = read_json(V1 / "plan.json")
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 2, "role": "terminal", "issue": 100,
        "frozen_at": frozen_at, "frozen_after_v1_publication": True, "issue_64_authorized": False,
        "revision_of": r100.IDENTITY, "validation_command": VALIDATION_COMMAND, "runner": RUNNER,
        "runner_sha256_at_freeze": sha256_of(ROOT / RUNNER), "engine_seconds": 0,
        "disclosure": ("v2 was frozen after the v1 publication, with every v1 outcome known (all readout F1s, "
                       "rollout errors, arm AUCs, contrasts and the v1 tokens). The v2 guard definitions come from "
                       "the diagnoses of the two v1 guard failures (motion-value noise; cuDNN autotuning in the "
                       "run process) and from the parse tolerance the v1 plan froze before any outcome; no F1, "
                       "AUC or contrast entered their definition. v2 changes no estimand, readout, threshold, "
                       "cohort, record or decision rule; the scored records stay the v1 records; v1 is retained "
                       "byte-identical (sha256-bound below)."),
        "changes": GUARD_CHANGES,
        "added_evidence": {"replication": ("F77 arm records recomputed in a fresh deterministic process on every "
                                           "#96 cell; evidence for revised G3 only, never scored"),
                           "probe_explanation": ("descriptive: F77 gated macro at probe frame 80 from a single-frame "
                                                 "carrier without prior frame; no verdict changes")},
        "unchanged": {"readout_target": v1_plan["readout_target"], "arms_rule": v1_plan["arms"]["rule"],
                      "guards": v1_plan["guards"], "records": "every v1 record and checkpoint, read-only"},
        "token_rule": ("each question's v2 token is its frozen v1 rule evaluated with the v2 guard set; a failing "
                       "v2 guard still forces readiness_or_precision_insufficient"),
        "terminal": "v2 is the final guard revision of #100; no v3",
        "inputs": bindings(),
        "claim_boundary": v1_plan["claim_boundary"],
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-100 v2 protocol")
    blob = json_text({k: v for k, v in plan.items() if k != "disclosure"})
    for marker in m99.PLACEHOLDER_MARKERS:
        if marker in blob:
            raise ValueError(f"frozen plan contains a missing-value marker {marker!r}")
    return plan


def prepare(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        load_plan(output)
        log("existing frozen v2 plan validated")
        return 0
    if (output / "records").exists() or (output / "compute.json").exists():
        raise ValueError("v2 records or publications exist before the freeze; refusing")
    output.mkdir(parents=True, exist_ok=True)
    plan = frozen_plan(utc_now())
    (output / "plan.json").write_text(json_text(plan))
    load_plan(output)
    log(f"frozen v2 plan published: {len(plan['inputs'])} v1 artifacts sha256-bound; no statistic computed")
    return 0


def run(output):
    output = Path(output)
    plan = load_plan(output)
    check_bindings(plan)
    began = time.monotonic()
    gpu = replicate(output)
    if not probe_path(output).is_file():
        probe_explanation(output)
    log(f"v2 run complete: replication gpu {gpu:.0f}s; wall {time.monotonic() - began:.0f}s")
    return 0


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def findings_md(plan, c):
    lines = []
    add = lines.append
    add("# Issue-100 v2: guard revision of the readout-fidelity / JEPA publication — findings")
    add("")
    add(f"- identity `{IDENTITY}`, revision of `{r100.IDENTITY}`; v2 frozen {plan['frozen_at']}, **after** the v1 "
        f"publication")
    add(f"- validation command: `{VALIDATION_COMMAND}` (runs the v1 `--validate` and re-derives every v2 table)")
    add(f"- disclosure: {plan['disclosure']}")
    add("")
    add("## Guards")
    add("")
    add("| guard | v1 pass | v2 pass | revised | v2 observed |")
    add("|---|---|---|---|---|")
    for name, g in c["guards"].items():
        add(f"| {name} | {g['v1_pass']} | {g['v2_pass']} | {g['revised']} | "
            f"`{json.dumps(g['v2_observed'], sort_keys=True)[:400]}` |")
    add("")
    for name, change in GUARD_CHANGES.items():
        add(f"- **{name}** v1: {change['v1']}. v2: {change['v2']}. Why: {change['reason']}.")
    add("")
    q2 = c["Q2_readout_target"]
    add("## Dispositions")
    add("")
    add(f"- **Q2 readout target (held-out, macro F1 >= {r100.MACRO_F1_TARGET} and micro edge F1 >= "
        f"{r100.MICRO_F1_TARGET}): v1 {q2['v1_token']} → v2 {q2['v2_token']}**; met by "
        f"{', '.join(q2['met_by']) or 'none'}; better encoder {q2['better_encoder']}")
    add("")
    add("| encoder | macro F1 | micro edge F1 | meets |")
    add("|---|---|---|---|")
    for encoder, row in q2["encoders"].items():
        add(f"| {encoder} | {fmt_interval(row['macro_f1'])} | {fmt_interval(row['micro_edge_f1'])} | {row['meets']} |")
    add("")
    q4 = c["Q4_alpha_effect"]
    add(f"**Q4 (better encoder {q4['encoder']}, cohort {q4['cohort']}, cost {q4['cost']})**")
    add("")
    add("| inventory | contrast | estimate | cells / members | v1 token | v2 token |")
    add("|---|---|---|---|---|---|")
    for inventory, row in q4["inventories"].items():
        for name, block in row.items():
            add(f"| {inventory} | {name} | {fmt_interval(block['estimate'])} | {block['cells']} / {block['members']} | "
                f"{block['v1_token']} | {block['v2_token']} |")
    add("")
    add("Limits: the held-out cohort has 2-3 mixed member clusters per inventory (a percentile bootstrap over "
        "3 clusters has 10 distinct resamples); no multiplicity adjustment across the 8 contrasts.")
    add("")
    add("## Noise impact of the v1 run records (F77 vs #96, count cost; descriptive)")
    add("")
    add("| arm | mixed cells | cells whose AUC changed | max abs AUC change | mean abs AUC change | candidates above "
        f"{COST_TOLERANCE} / candidates |")
    add("|---|---|---|---|---|---|")
    for arm, row in c["noise_impact_v1_run_vs_96"].items():
        add(f"| {arm} | {row['mixed_cells']} | {row['cells_with_auc_change']} | {fmt(row['max_abs_auc_change'])} | "
            f"{fmt(row['mean_abs_auc_change'])} | {row['candidates_above_cost_tolerance']} / {row['candidates']} |")
    add("")
    p = c["probe_explanation"]
    add("## Probe macro numbers (descriptive)")
    add("")
    add(f"- claimed (2026-09-25 probe): steady {p['claimed']['steady']}, unstable {p['claimed']['unstable']}; v1 "
        f"(deployment carrier, prior frame 79): {[round(v, 4) for v in p['v1_observed_with_prior']]}")
    for seed, row in p["no_prior"].items():
        add(f"- seed {seed}, single-frame carrier without prior: gated macro "
            f"{[round(v, 4) for v in row['gated_macro_no_prior']]}")
    add("")
    add(f"Compute: replication {c['compute']['replication_gpu_seconds']:.0f} GPU s; engine seconds 0.")
    add("")
    add("## Claim boundary")
    add("")
    add(plan["claim_boundary"] + ".")
    add("")
    return "\n".join(lines)


def comparisons_csv(c):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["id", "statistic", "value", "interval_low", "interval_high", "interval_label", "detail"])
    for name, g in c["guards"].items():
        writer.writerow([f"guard_{name}", "v1 pass / v2 pass", f"{g['v1_pass']}/{g['v2_pass']}", "", "", "",
                         json.dumps(g["v2_observed"], sort_keys=True)])
    q2 = c["Q2_readout_target"]
    writer.writerow(["Q2_token", "v1 / v2", f"{q2['v1_token']}/{q2['v2_token']}", "", "", "", json.dumps(q2["met_by"])])
    for inventory, row in c["Q4_alpha_effect"]["inventories"].items():
        for name, block in row.items():
            e = block["estimate"]
            writer.writerow([f"Q4_{inventory}_{name}", "estimate", e and e["mean"], e and e["interval"][0],
                             e and e["interval"][1], "DESCRIPTIVE" if e else "",
                             f"{block['v1_token']}/{block['v2_token']}"])
    for arm, row in c["noise_impact_v1_run_vs_96"].items():
        writer.writerow([f"noise_{arm}", "max abs AUC change", row["max_abs_auc_change"], "", "", "",
                         json.dumps(row, sort_keys=True)])
    return buffer.getvalue()


def summary(plan, c):
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "validation_command": VALIDATION_COMMAND, "disclosure": plan["disclosure"], "changes": GUARD_CHANGES,
            **{k: c[k] for k in ("guards", "guards_pass", "Q2_readout_target", "Q4_alpha_effect",
                                 "noise_impact_v1_run_vs_96", "probe_explanation", "compute")},
            "claim_boundary": plan["claim_boundary"], "issue_64_authorized": False}


def rendered(plan, c):
    return {"summary.json": json_text(summary(plan, c)), "findings.md": findings_md(plan, c),
            "comparisons.csv": comparisons_csv(c)}


def publish(output):
    output = Path(output)
    plan = load_plan(output)
    c = compute(output, plan)
    write_json(output / "compute.json", c)
    for name, text in rendered(plan, c).items():
        (output / name).write_bytes(text.encode())
    log(f"published v2: Q2 {c['Q2_readout_target']['v2_token']}; Q4 "
        f"{ {i: {n: b['v2_token'] for n, b in r.items()} for i, r in c['Q4_alpha_effect']['inventories'].items()} }")
    return 0


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    fresh = compute(output, plan)
    if read_json(output / "compute.json") != json.loads(json_text(fresh)):
        raise ValueError("compute.json differs from the fresh recomputation")
    for name, text in rendered(plan, fresh).items():
        if (output / name).read_bytes() != text.encode():
            raise ValueError(f"published {name} differs from the recomputation")
    log(f"v2 validation passed: v1 bindings, v1 --validate, revised guards and tokens re-derived and "
        f"byte-compared ({time.monotonic() - began:.1f}s)")
    return 0


def dry_run(output):
    log(f"dry run (no write, no statistic); output root {Path(output)}")
    for entry in bindings():
        log(f"  bind {entry['artifact']} {entry['sha256'][:16]}")
    for name, change in GUARD_CHANGES.items():
        log(f"  revise {name}: {change['v2']}")
    log(f"  plan present: {(Path(output) / 'plan.json').is_file()}")
    return 0


MODES = {"dry-run": (dry_run, False), "prepare": (prepare, False), "run": (run, True),
         "publish": (publish, True), "validate": (validate, True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
