"""Issue #104: the bounded rendered smoke of the frozen joint #104 + #110 campaign plan.

Membership, gate and limits come from ``.local-artifacts/issue-104-power-cohort-v1/plan.json``
(``smoke``, ``limits``): the #110 smoke roster (40 pairs x both sides x c05, c10) and
exposed #77 N1 replay branches, captured once each on the frozen player (#113 window),
renderer (Mesa 26.1.2 / LLVM 22.1.6 llvmpipe) and pipeline. No cohort fit or evaluation
level is touched; nothing is fitted or scored.

Modes: ``--dry-run`` (build every smoke record, re-materialize it and check it against
its frozen level files; writes nothing), ``--run`` (capture; resumable, no retries),
``--publish`` (gates, per-template #110 verdicts, window/settle/collapse evidence,
throughput, budget re-projection, gallery), ``--validate`` (recompute and compare).
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import gzip
import html
from io import StringIO
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from scripts import capture_pipeline_v2 as pipeline
from scripts import prepare_power_cohort as cohort

ROOT = cohort.ROOT
IDENTITY = "issue-104-capture-smoke-v1"
OUTPUT = ROOT / ".local-artifacts" / IDENTITY
MEDIA = ROOT / "data" / "issue-104-capture-smoke"
RETAINED = cohort.SMOKE_109
VALIDATION_COMMAND = "python -u -m scripts.run_power_cohort_smoke --validate"
SCHEMA = "issue_104_capture_smoke_report_v1"
PUBLISHED = ("summary.json", "branches.json", "comparisons.csv", "findings.md")
FIXED_DELTA = 0.0004
STRIDE = 50
STANDARD_GRAVITY = (0.0, -9.8)
INVERSE_GRAVITY = (0.0, 6.0)
ANALYSIS_PROCESSES = 24


def log(message):
    print(f"[{IDENTITY}] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def read(path):
    return json.loads(Path(path).read_text())


def load():
    plan = cohort.load_plan()
    records = list(cohort.smoke_records(plan))
    expected = plan["smoke"]["novelty_branches"] + plan["smoke"]["replay_branches"]
    if [record["identity"] for record in records] != expected:
        raise ValueError("smoke records differ from the frozen smoke membership")
    return plan, records


# ------------------------------------------------------------------ dry run

def dry_run():
    """No-write check: every record re-materializes to its frozen scenario and xml."""
    from scripts import issue_76_expansion as files
    plan, records = load()
    problems = []
    with tempfile.TemporaryDirectory() as scratch:
        for index, record in enumerate(records):
            target = Path(scratch) / str(index)
            target.mkdir()
            generated, scenario = files.materialize(record, record["template"], target)
            if scenario.to_dict() != record["scenario"]:
                problems.append(f"{record['identity']}: scenario re-materialization differs")
            if record["stream"] == "smoke-novelty" and generated.xml_content.decode("utf-8") != record["xml"]:
                problems.append(f"{record['identity']}: xml re-materialization differs")
            side = "right" if float(ET.fromstring(record["xml"]).find("Slingshot").attrib["x"]) > 0 else "left"
            if (record["actions"][0]["drag_x"] > 0) != (side == "right"):
                problems.append(f"{record['identity']}: drag_x sign does not pull away from the targets")
    counts = Counter(record["stream"] for record in records)
    print(json_text({"records": len(records), "streams": dict(counts), "problems": problems,
                     "limits": {key: plan["limits"][key] for key in ("workers", "native_window", "renderer")},
                     "writes": "none"}), end="")
    return not problems


# ---------------------------------------------------------------------- run

def run():
    plan, records = load()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = OUTPUT / "run-manifest.json"
    binding = {"cohort_plan": cohort.relative(cohort.OUTPUT / "plan.json"),
               "cohort_plan_sha256": cohort.file_sha256(cohort.OUTPUT / "plan.json"),
               "pipeline_sha256": cohort.file_sha256(cohort.PIPELINE), "runner_sha256": cohort.file_sha256(__file__),
               "branches": [record["identity"] for record in records]}
    if manifest.is_file():
        if {k: v for k, v in read(manifest).items() if k != "started_at"} != binding:
            raise ValueError("smoke run manifest differs from the frozen plan bindings")
    else:
        manifest.write_text(json_text({**binding, "started_at": utc_now()}))
    if plan["pipeline"]["sha256"] != binding["pipeline_sha256"]:
        raise ValueError("capture pipeline differs from the frozen pipeline")
    if not (OUTPUT / "player").exists():
        shutil.copytree(cohort.OUTPUT / "player", OUTPUT / "player", symlinks=True)
    limits = {**plan["limits"], "workers": plan["smoke"]["workers"]}
    wall = pipeline.run_campaign(OUTPUT, IDENTITY, records, limits, IDENTITY)
    log(f"smoke pass finished in {wall / 60:.1f} min")


# ----------------------------------------------------------------- analysis

def chunks(root, manifest):
    for descriptor in manifest["chunks"]:
        with gzip.open(Path(root) / descriptor["path"], "rb") as stream:
            yield json.loads(stream.read())


def level_source(record):
    """(contract source, materials, slingshot side) of a record's level xml."""
    from world_model.data import novelty_contract as contract
    tree = ET.fromstring(record["xml"])
    materials = {node.attrib["scenarioObjectId"]: node.attrib.get("material") or None
                 for node in tree.iter() if "scenarioObjectId" in node.attrib}
    side = contract.slingshot_side(float(tree.find("Slingshot").attrib["x"]))
    return contract.source_from_scenario(tree), materials, side


def window_check(manifest, events):
    """The #113 rule applied to one complete trace: (consistent, explanation)."""
    first = manifest["first_fixed_step"]
    cap, tail = manifest.get("maximum_shot_steps"), manifest.get("rest_tail_steps")
    terminal = manifest["terminal_evidence"]
    last = manifest["last_fixed_step"]
    stability = [(step, kind) for step, kind, _ in events if kind in ("stable_entered", "stable_exited")]
    entered = [step for step, kind in stability if kind == "stable_entered" and step - first <= cap]
    if (cap, tail) != (30000, 2500):
        return False, f"declared window {(cap, tail)}"
    if terminal is None:
        if manifest["failure"] != "native_time_window_limit":
            return False, f"failed trace {manifest['failure']}"
        if last - first == cap:
            return True, "censored at the cap"
        exited = [step for step, kind in stability if kind == "stable_exited" and step - first > cap]
        return (bool(exited) and cap < last - first <= cap + tail), "censored after a tail cancelled past the cap"
    reason = terminal["reason"]
    if reason == "rest_tail_complete":
        return (bool(entered) and last == entered[-1] + tail), "rest tail complete"
    if reason in ("level_clear", "level_fail"):
        return True, f"{reason}{' inside a tail' if entered else ''}"
    return False, f"terminal {reason} is not allowed under the rule"


def force_sign(root, manifest, events, source, materials):
    """Median projection of residual acceleration on active force directions (contact-free steps)."""
    from world_model.data import novelty_contract as contract
    if source is None or contract.LAW_OF_TYPE.get(source.type) in (None, "global_gravity"):
        return None
    launched = set()
    storm = contract.storm_onset_step([{"event_type": kind, "fixed_step": step, "participants": who}
                                       for step, kind, who in events],
                                      [f"runtime:bird:{i:04}" for i in range(4)])
    launches = {step: who for step, kind, who in events if kind == "bird_launched"}
    projections, previous = [], None
    for chunk in chunks(root, manifest):
        for sample in chunk["fixed_step_samples"]:
            step = sample["fixed_step"]
            for who in launches.get(step, ()):
                launched.add(who.replace("runtime:", ""))
            bodies = {body.slot: body for body in contract.bodies_from_sample(sample, materials, launched, source.slot)}
            free = {e["scenario_object_id"]: (not e["contact_ids"]) for e in sample["entities"]}
            scales = {e["scenario_object_id"]: (e.get("body") or {}).get("gravity_scale") for e in sample["entities"]}
            if previous is not None:
                old_bodies, old_free, gravity = previous
                for slot, body in bodies.items():
                    before = old_bodies.get(slot)
                    if before is None or not (free.get(slot) and old_free.get(slot)) or scales.get(slot) is None:
                        continue
                    force = contract.force_on(source, before, storm_active=storm is not None and step - 1 >= storm)
                    if not force:
                        continue
                    norm = math.hypot(*force)
                    residual = contract.residual_acceleration(before, body, gravity, scales[slot], FIXED_DELTA)
                    projections.append((residual[0] * force[0] + residual[1] * force[1]) / norm)
            previous = (bodies, free, sample["world"]["gravity_vector"])
    if not projections:
        return {"verdict": "not_exercised", "steps": 0}
    median = statistics.median(projections)
    return {"verdict": "agree" if median > 0 else "disagree", "steps": len(projections),
            "median_projection": round(median, 4)}


def analyze(record, plan):
    """Per-branch evidence from the retained result and native trace (no outcome used for any gate)."""
    result = read(OUTPUT / "results" / f"{record['identity']}.json")
    row = {"identity": record["identity"], "stream": record["stream"],
           "level": record["source_member_identity"], "novelty_level": record["novelty_level"],
           "family": record["generator_family"], "complete": bool(result["complete"]),
           "failure": result.get("failure"), "failure_class": result.get("failure_class"),
           "renderer": result.get("renderer"), "stop_kind": result.get("stop_kind"),
           "wall_seconds": result.get("wall_seconds"), "stage_seconds": result.get("stage_seconds")}
    if not row["complete"]:
        return row
    segment = result["segments"][0]
    root = Path(segment["native_root"])
    manifest = read(root / "native-manifest.json")
    events = []
    for chunk in chunks(root, manifest):
        events += [(e["fixed_step"], e["event_type"], tuple(e["participants"])) for e in chunk["events"]]
    first = manifest["first_fixed_step"]
    consistent, explanation = window_check(manifest, events)
    frames = [f["fixed_step"] for f in manifest["frame_records"]]
    entered = [step for step, kind, _ in events if kind == "stable_entered"]
    rest_frames = sum(1 for step in frames if entered and step >= entered[-1])
    source, materials, side = level_source(record)
    first_sample = next(chunks(root, manifest))["fixed_step_samples"][0]
    supports = []
    for chunk in chunks(root, manifest):
        for sample in chunk["fixed_step_samples"]:
            if (sample["fixed_step"] - first) % STRIDE == 0:
                supports.append(frozenset((s["supporter_entity_id"], s["supported_entity_id"])
                                          for s in sample["supports"]
                                          if "bird:" not in s["supporter_entity_id"]
                                          and "bird:" not in s["supported_entity_id"]))
    launch_event = None
    for chunk in chunks(root, manifest):
        launch_event = next((e for e in chunk["events"] if e["event_type"] == "bird_launched"), launch_event)
        if launch_event:
            break
    storm = None
    if source is not None and source.type == "Storm":
        from world_model.data import novelty_contract as contract
        onset = contract.storm_onset_step([{"event_type": kind, "fixed_step": step, "participants": who}
                                           for step, kind, who in events],
                                          [f"runtime:bird:{i:04}" for i in range(4)])
        storm = None if onset is None else onset - first
    row.update(
        terminal=(manifest["terminal_evidence"] or {}).get("reason") or manifest["failure"],
        steps=manifest["last_fixed_step"] - first, frames=len(frames),
        window=[manifest.get("maximum_shot_steps"), manifest.get("rest_tail_steps")],
        window_consistent=consistent, window_explanation=explanation,
        stable_entered_steps=[step - first for step in entered], rest_frames=rest_frames,
        support_change_frames=sum(supports[t] != supports[t - 1] for t in range(1, len(supports))),
        gravity=first_sample["world"]["gravity_vector"], slingshot_side=side,
        launch_velocity=launch_event["payload"].get("launch_velocity") if launch_event else None,
        novelty_in_trace=any(e["scenario_object_id"] == "novelty:0000" for e in first_sample["entities"]),
        novelty_authored="novelty:0000" in record["generated_slots"],
        storm_onset_step=storm, force_sign=force_sign(root, manifest, events, source, materials),
        source_type=None if source is None else source.type)
    return row


def replay_row(record):
    """Re-capture of an exposed branch vs the retained #109 smoke (pixels, stop kind, native prefix)."""
    identity = record["identity"]
    new, old = read(OUTPUT / "results" / f"{identity}.json"), read(RETAINED / "results" / f"{identity}.json")
    row = {"identity": identity, "complete": bool(new["complete"]), "retained_stop_kind": old["stop_kind"],
           "stop_kind": new.get("stop_kind")}
    if not new["complete"]:
        return row
    boxes = pipeline.character_boxes(read(OUTPUT / "attempts" / identity / "decision-physics-v1.json"))
    retained_png = (RETAINED / "attempts" / identity / "paused-before.png").read_bytes()
    new_png = (OUTPUT / "attempts" / identity / "paused-before.png").read_bytes()
    try:
        row["decision_frame"] = pipeline.render_invariance(retained_png, new_png, boxes)
        row["decision_frame_equal"] = True
    except ValueError as error:
        row["decision_frame"], row["decision_frame_equal"] = str(error), False
    oroot, nroot = Path(old["segments"][0]["native_root"]), Path(new["segments"][0]["native_root"])
    omanifest, nmanifest = read(oroot / "native-manifest.json"), read(nroot / "native-manifest.json")
    last = min(omanifest["last_fixed_step"], nmanifest["last_fixed_step"])

    def prefix(root, manifest):
        for chunk in chunks(root, manifest):
            for sample in chunk["fixed_step_samples"]:
                if sample["fixed_step"] > last:
                    return
                yield sample
    row["first_native_difference_step"] = next(
        (a["fixed_step"] - omanifest["first_fixed_step"] for a, b in zip(prefix(oroot, omanifest), prefix(nroot, nmanifest))
         if a != b), None)
    return row


# ------------------------------------------------------------------ publish

def throughput(plan, rows):
    runs = [read(path) for path in sorted((OUTPUT / "run-log").glob("*.json"))]
    complete = [row for row in rows if row["complete"]]
    walls = [row["wall_seconds"] for row in complete]
    gate = [row["stage_seconds"].get("engine_boot_connect", 0) + row["stage_seconds"].get("menu_to_playing", 0)
            for row in complete]
    seen = set()
    pipeline._attempt_bytes(OUTPUT / "player", seen)
    sizes = [pipeline._attempt_bytes(OUTPUT / "attempts" / row["identity"], seen) for row in complete]
    campaign_wall = sum(run["wall_seconds"] for run in runs)
    amortized = campaign_wall / len(rows)
    per_branch = statistics.mean(sizes)
    branches = plan["membership"]["branches"]
    return {"workers": plan["smoke"]["workers"], "campaign_wall_seconds": round(campaign_wall, 1),
            "amortized_seconds_per_branch": round(amortized, 2),
            "branch_wall_mean": round(statistics.mean(walls), 1),
            "branch_wall_p90": round(sorted(walls)[int(0.9 * (len(walls) - 1))], 1),
            "cold_start_critical_section_mean": round(statistics.mean(gate), 2),
            "shot_capture_mean": round(statistics.mean(row["stage_seconds"]["shot_capture"] for row in complete), 1),
            "artifact_bytes_per_branch_mean": round(per_branch),
            "label": "single smoke pass including the cold-start ramp; DESCRIPTIVE",
            "campaign_reprojection": {
                "branches": branches,
                "wall_hours_at_smoke_rate": round(branches * amortized / 3600, 1),
                "wall_hours_cold_start_bound": round(branches * max(statistics.mean(gate),
                                                                   statistics.mean(walls) / plan["smoke"]["workers"]) / 3600, 1),
                "artifact_tib": round(branches * per_branch / 2**40, 2),
                "frozen_projection": {"wall_hours": plan["budget"]["projected_wall_hours"],
                                      "artifact_tib": plan["budget"]["projected_artifact_tib"]}}}


def template_verdicts(rows):
    """#110 per-template smoke criteria; failing T1-T4 makes the template's cell unsupported."""
    by_level = defaultdict(list)
    for row in rows:
        if row["stream"] == "smoke-novelty":
            by_level[row["level"]].append(row)
    verdicts = {}
    for level, items in sorted(by_level.items()):
        reasons = []
        if not all(item["complete"] for item in items):
            reasons.append("typed_capture_failure: " + ", ".join(sorted({str(i["failure_class"]) for i in items
                                                                       if not i["complete"]})))
        complete = [item for item in items if item["complete"]]
        if any(item["novelty_authored"] and not item["novelty_in_trace"] for item in complete):
            reasons.append("novelty_entity_missing_from_trace")
        expected = INVERSE_GRAVITY if items[0]["novelty_level"] == 6 else STANDARD_GRAVITY
        if any(any(abs(a - b) > 1e-4 for a, b in zip(item["gravity"], expected)) for item in complete):
            reasons.append(f"gravity_differs_from_{list(expected)}")
        if any(item["slingshot_side"] == "right" and not (item["launch_velocity"] and item["launch_velocity"][0] < 0)
               for item in complete):
            reasons.append("right_slingshot_launch_not_leftward")
        signs = [item["force_sign"] for item in complete if item["force_sign"] is not None]
        sign = ("not_applicable" if not signs else "disagree" if any(s["verdict"] == "disagree" for s in signs)
                else "agree" if any(s["verdict"] == "agree" for s in signs) else "not_exercised")
        verdicts[level] = {"branches": len(items), "complete": len(complete), "novelty_level": items[0]["novelty_level"],
                           "family": items[0]["family"], "unsupported_reasons": reasons,
                           "supported": not reasons, "force_sign": sign,
                           "force_sign_effect": ("labeler disagreement: repaired in the derivation before any "
                                                 "training (#110 rule); not a cell failure") if sign == "disagree" else None,
                           "storm_onset_in_window": [item["storm_onset_step"] is not None for item in complete
                                                     if item["source_type"] == "Storm"] or None}
    return verdicts


def cell_readiness(verdicts):
    """45 cells: a pair covers its novel cell; the 5 normal cells are covered by the normal sides."""
    plan110 = read(cohort.COHORT_110 / "plan.json")
    cells = {}
    for pair in plan110["capture_membership"]["pairs"]:
        family, level = pair["family"], pair["novelty_level"]
        for side, cell_level in (("normal", 0), ("novel", level)):
            key = f"{pair['scenario']}|novelty_level_{cell_level}"
            identity = f"issue-110-smoke-{family}-{side}-001"
            entry = cells.setdefault(key, {"templates": [], "reasons": []})
            entry["templates"].append(identity)
            entry["reasons"] += [f"{identity}: {r}" for r in verdicts[identity]["unsupported_reasons"]]
    return {key: {"status": "supported" if not value["reasons"] else "unsupported",
                  "templates": len(value["templates"]), "reasons": value["reasons"]}
            for key, value in sorted(cells.items())}


def per_level_type(rows):
    table = defaultdict(lambda: {"branches": 0, "complete": 0, "stop_kinds": Counter(), "terminals": Counter(),
                                 "rest_frames": 0, "shots_with_rest": 0, "collapse_shots": 0})
    for row in rows:
        if row["stream"] != "smoke-novelty":
            continue
        entry = table[f"novelty_level_{row['novelty_level']}"]
        entry["branches"] += 1
        if not row["complete"]:
            continue
        entry["complete"] += 1
        entry["stop_kinds"][row["stop_kind"]] += 1
        entry["terminals"][row["terminal"]] += 1
        entry["rest_frames"] += row["rest_frames"]
        entry["shots_with_rest"] += row["rest_frames"] > 0
        entry["collapse_shots"] += row["support_change_frames"] > 0
    return {key: {**value, "stop_kinds": dict(value["stop_kinds"]), "terminals": dict(value["terminals"])}
            for key, value in sorted(table.items())}


def build():
    plan, records = load()
    with ProcessPoolExecutor(ANALYSIS_PROCESSES) as pool:
        rows = list(pool.map(analyze, records, [plan] * len(records)))
        replay = list(pool.map(replay_row, [record for record in records if record["stream"] == "smoke-replay"]))
    failures = [row for row in rows if not row["complete"]]
    complete = [row for row in rows if row["complete"]]
    verdicts = template_verdicts(rows)
    gates = {
        "S1_capture_reliability": {"typed_failures": len(failures), "branches": len(rows),
                                   "rate": round(len(failures) / len(rows), 4),
                                   "classes": dict(Counter(row["failure_class"] for row in failures)),
                                   "pass": len(failures) / len(rows) <= 0.05},
        "S2_renderer": {"mismatched": [row["identity"] for row in complete if row["renderer"] != plan["renderer"]["expected"]],
                        "renderer_mismatch_failures": sum(row["failure_class"] == "renderer_mismatch" for row in failures)},
        "S3_pixel_identity": {"equal": sum(bool(row.get("decision_frame_equal")) for row in replay),
                              "replay_branches": len(replay)},
        "S4_window_rule": {"violations": [row["identity"] for row in complete if not row["window_consistent"]],
                           "explanations": dict(Counter(row["window_explanation"] for row in complete))},
    }
    gates["S2_renderer"]["pass"] = not gates["S2_renderer"]["mismatched"] and not gates["S2_renderer"]["renderer_mismatch_failures"]
    gates["S3_pixel_identity"]["pass"] = gates["S3_pixel_identity"]["equal"] == len(replay) and bool(replay)
    gates["S4_window_rule"]["pass"] = not gates["S4_window_rule"]["violations"]
    launch = all(gate["pass"] for gate in gates.values())
    cells = cell_readiness(verdicts)
    summary = {
        "schema": SCHEMA, "identity": IDENTITY, "cohort_plan": cohort.relative(cohort.OUTPUT / "plan.json"),
        "cohort_plan_sha256": cohort.file_sha256(cohort.OUTPUT / "plan.json"),
        "validation_command": VALIDATION_COMMAND,
        "gates": gates,
        "campaign_launch": {"token": "supported" if launch else "readiness_or_precision_insufficient",
                            "statement": ("every smoke gate passed: the frozen campaign may launch" if launch else
                                          "a smoke gate failed: the frozen campaign is not launched")},
        "novelty_templates": verdicts,
        "cells": {"supported": sum(c["status"] == "supported" for c in cells.values()),
                  "unsupported": sum(c["status"] == "unsupported" for c in cells.values()), "table": cells},
        "per_level_type": per_level_type(rows),
        "replay": {"rows": replay,
                   "stop_kind_agreement": sum(r.get("stop_kind") == r["retained_stop_kind"] for r in replay if r["complete"]),
                   "native_prefix_identical": sum(r.get("first_native_difference_step") is None for r in replay if r["complete"])},
        "throughput": throughput(plan, rows),
        "engine_seconds": round(sum(row["wall_seconds"] or 0 for row in rows), 1),
        "claim_boundary": ("rendered smoke of the frozen plan: capture reliability, renderer, window rule and #110 "
                           "runtime readiness; no cohort level fitted or scored; outcomes are not read by any gate"),
    }
    return summary, rows


def comparisons_csv(summary):
    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["id", "value", "detail"])
    for name, gate in summary["gates"].items():
        writer.writerow([name, gate["pass"], json.dumps({k: v for k, v in gate.items() if k != "pass"}, sort_keys=True)])
    for level, verdict in summary["novelty_templates"].items():
        writer.writerow([f"template:{level}", verdict["supported"], ";".join(verdict["unsupported_reasons"])
                         or verdict["force_sign"]])
    for name, value in summary["per_level_type"].items():
        writer.writerow([f"level_type:{name}", value["complete"], json.dumps(value, sort_keys=True)])
    return buffer.getvalue()


def findings(summary):
    gates, through = summary["gates"], summary["throughput"]
    lines = [f"# #104 rendered smoke (`{IDENTITY}`)", "", f"`{VALIDATION_COMMAND}`; cohort plan "
             f"`{summary['cohort_plan_sha256'][:23]}...`.", "",
             f"**Campaign launch: `{summary['campaign_launch']['token']}`** — {summary['campaign_launch']['statement']}.",
             "", "| gate | pass | evidence |", "| --- | --- | --- |"]
    for name, gate in gates.items():
        lines.append(f"| {name} | {gate['pass']} | {json.dumps({k: v for k, v in gate.items() if k != 'pass'})} |")
    lines += ["", f"Cells: {summary['cells']['supported']} supported, {summary['cells']['unsupported']} unsupported "
              "(#110 per-template criteria).", "", "| level type | branches | complete | terminals | shots with rest "
              "frames | rest frames | collapse shots |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, value in summary["per_level_type"].items():
        lines.append(f"| {name} | {value['branches']} | {value['complete']} | {value['terminals']} | "
                     f"{value['shots_with_rest']} | {value['rest_frames']} | {value['collapse_shots']} |")
    lines += ["", "| template | supported | reasons | force sign | storm onset in window |", "| --- | --- | --- | --- | --- |"]
    for level, verdict in summary["novelty_templates"].items():
        lines.append(f"| {level} | {verdict['supported']} | {'; '.join(verdict['unsupported_reasons'])} | "
                     f"{verdict['force_sign']} | {verdict['storm_onset_in_window']} |")
    unsupported = {k: v for k, v in summary["cells"]["table"].items() if v["status"] != "supported"}
    if unsupported:
        lines += ["", "Unsupported cells:", ""] + [f"- {k}: {'; '.join(v['reasons'])}" for k, v in unsupported.items()]
    replay = summary["replay"]
    lines += ["", f"Replay vs retained #109 smoke: decision frames equal {gates['S3_pixel_identity']['equal']}/"
              f"{gates['S3_pixel_identity']['replay_branches']}; stop kinds equal {replay['stop_kind_agreement']}; "
              f"native prefixes identical {replay['native_prefix_identical']} (cross-machine float divergence after "
              "the first collision is disclosed in the plan).", "",
              f"Throughput ({through['workers']} workers): {through['amortized_seconds_per_branch']} s/branch amortized "
              f"(single pass incl. ramp), branch wall mean {through['branch_wall_mean']} s (p90 {through['branch_wall_p90']}), "
              f"cold-start section {through['cold_start_critical_section_mean']} s, {through['artifact_bytes_per_branch_mean'] / 1e6:.1f} MB/branch.",
              f"Campaign re-projection: {through['campaign_reprojection']}.", "",
              f"Engine seconds (sum of branch walls): {summary['engine_seconds']}.", ""]
    return "\n".join(lines)


def media_entry(row):
    identity = row["identity"]
    entry = {"identity": identity, "stream": row["stream"], "complete": row["complete"],
             "failure_class": row["failure_class"], "terminal": row.get("terminal"), "frames": {}}
    target = MEDIA / identity
    target.mkdir(exist_ok=True)
    decision = OUTPUT / "attempts" / identity / "paused-before.png"
    if decision.is_file():
        shutil.copy2(decision, target / "decision.png")
        entry["frames"]["decision"] = f"{identity}/decision.png"
    result = read(OUTPUT / "results" / f"{identity}.json")
    if row["complete"]:
        root = Path(result["segments"][0]["native_root"])
        frames = sorted(root.glob("frame_*.png"))
        shutil.copy2(frames[-1], target / "final.png")
        entry["frames"]["final"] = f"{identity}/final.png"
        if row["stable_entered_steps"]:
            index = min(len(frames) - 1, math.ceil(row["stable_entered_steps"][-1] / STRIDE))
            shutil.copy2(frames[index], target / "rest.png")
            entry["frames"]["rest"] = f"{identity}/rest.png"
        video = target / "shot.webm"
        if not video.is_file():
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "25", "-pattern_type", "glob",
                            "-i", str(root / "frame_*.png"), "-c:v", "libvpx", "-b:v", "600k", "-an",
                            str(video)], check=True)
        entry["frames"]["video"] = f"{identity}/shot.webm"
    return entry


def media(rows, records):
    """Decision, first-rest and final frames plus a WebM per branch; manifest and gallery."""
    MEDIA.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(ANALYSIS_PROCESSES) as pool:
        entries = list(pool.map(media_entry, rows))
    manifest = {"schema": "issue_104_capture_smoke_media_v1", "identity": IDENTITY,
                "label": "reviewable media of every scheduled smoke branch, failed ones included", "branches": entries}
    (MEDIA / "manifest.json").write_text(json_text(manifest))
    body = []
    for entry in entries:
        cells = "".join(f'<td><img src="{html.escape(entry["frames"][name])}" width="256"></td>'
                        if name in entry["frames"] else "<td>&mdash;</td>" for name in ("decision", "rest", "final"))
        video = (f'<td><video src="{html.escape(entry["frames"]["video"])}" width="256" controls></video></td>'
                 if "video" in entry["frames"] else "<td>&mdash;</td>")
        body.append(f"<tr><td>{html.escape(entry['identity'])}</td><td>{entry['stream']}</td>"
                    f"<td>{entry['terminal'] or entry['failure_class']}</td>{cells}{video}</tr>")
    (MEDIA / "index.html").write_text(
        "<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>issue-104 capture smoke</title><style>"
        "body{font-family:sans-serif}td{vertical-align:top;padding:4px;border-bottom:1px solid #ccc}</style></head>"
        "<body>\n<h1>Issue #104 rendered smoke: decision, first rest, final frames and shot video</h1>\n<table><tr>"
        "<th>branch</th><th>stream</th><th>terminal</th><th>decision</th><th>rest</th><th>final</th><th>video</th>"
        "</tr>\n" + "\n".join(body) + "\n</table></body></html>\n")
    return manifest


def outputs(summary, rows):
    return {"summary.json": json_text(summary), "branches.json": json_text(rows),
            "comparisons.csv": comparisons_csv(summary), "findings.md": findings(summary)}


def publish():
    summary, rows = build()
    for name, text in outputs(summary, rows).items():
        (OUTPUT / name).write_text(text)
    _, records = load()
    media(rows, records)
    log(f"published: launch {summary['campaign_launch']['token']}; cells {summary['cells']['supported']} supported")


def validate():
    summary, rows = build()
    problems = [name for name, text in outputs(summary, rows).items() if (OUTPUT / name).read_text() != text]
    manifest = read(MEDIA / "manifest.json")
    missing = [path for entry in manifest["branches"] for path in entry["frames"].values() if not (MEDIA / path).is_file()]
    if missing:
        problems.append(f"missing media: {missing[:5]}")
    if len(manifest["branches"]) != len(rows):
        problems.append("media manifest does not cover every scheduled branch")
    print(json_text({"identity": IDENTITY, "problems": problems, "validated": not problems}), end="")
    return not problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "run", "publish", "validate"):
        mode.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        ok = dry_run()
    elif args.run:
        run()
        ok = True
    elif args.publish:
        publish()
        ok = True
    else:
        ok = validate()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
