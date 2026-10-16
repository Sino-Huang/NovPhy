"""Issue #113 items 1 and 3: observable macro predicates and a temporal macro head.

Binding runner module: scripts/run_observable_macro.py
Exact validation command: python -u -m scripts.run_observable_macro --validate

Item 1 (output ``.local-artifacts/issue-113-macro-predicates-v1``, its own freeze): the macro
predicates redefined per request horizon Delta in {1, 5, 15} (``world_model/data/observable_macro.py``):
settle with an explicit debounce and structure change as the debounced support-set difference
between the request's anchor and landing frames. The debounce is chosen by a declared rule on the
#104 fit stream and the retained N1, #93 and #94 captures only; no evaluation-split trace is read.
Labels are recomputed on the retained captures and the #104 fit and policy streams, with prevalence
per phase and per level type.

Item-1 modes: --scan (definition-free trace summaries; --sources), --select (debounce rule),
--smoke1, --prepare1 (freeze), --run1 (policy summaries, prevalence records), --publish1.

Item 3 (output ``.local-artifacts/issue-113-temporal-macro-head-v1``, its own freeze after item 1's):
a macro head with stride-Delta temporal context (``world_model/training/temporal_macro_head.py``)
trained on the #104 fit stream with a class-balanced loss, selected by level-grouped leave-out
against the same-capacity single-frame head on the new labels and the current Gate-B ``macro_head``
as-is, and scored once on the #109 evaluation stream: macro F1 per redefined predicate on engine
carriers (teacher-forced) and on Gate-B predicted carriers, with #100-style readout counts and
phases (the #113 item-4 readout target).

Item-3 modes: --smoke3, --prepare3 (freeze), --carriers (sharded: --workers 4 --worker k),
--scan-evaluation, --assemble --split fit|evaluation, --cv (sharded), --select3, --final (sharded),
--score (sharded), --publish3. Training and scoring run under the #111 deterministic_scoring()
policy. --validate checks every published item and re-scores retained heads.

Zero engine seconds: every number is read from retained native traces and captured frames.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import functools
import hashlib
import io
import json
import math
import multiprocessing
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import unquote

import numpy as np

from world_model.data import observable_macro as om

ROOT = Path(__file__).resolve().parents[1]
OUTPUT1 = ROOT / ".local-artifacts/issue-113-macro-predicates-v1"
CAMPAIGN = ROOT / ".local-artifacts/issue-104-power-cohort-campaign-v1"
N1 = ROOT / ".local-artifacts/issue-77-n1-v1"
ORACLE = {"i93": ROOT / "data/issue-93-second-parameterization/attempts",
          "i94": ROOT / "data/issue-94-launch-power/attempts"}
CAPTURE_WINDOW = ROOT / ".local-artifacts/issue-113-capture-window-v1/spec.json"

IDENTITY1 = om.IDENTITY
SCHEMA_SELECTION = "issue_113_debounce_selection_v1"
SCHEMA_PLAN1 = "issue_113_macro_predicates_plan_v1"
SCHEMA_SMOKE1 = "issue_113_macro_predicates_smoke_v1"
SCHEMA_PREVALENCE = "issue_113_prevalence_record_v1"
SCHEMA_MANIFEST = "issue_113_summaries_manifest_v1"
SCHEMA_COMPUTE1 = "issue_113_macro_predicates_compute_v1"
SCHEMA_SUMMARY1 = "issue_113_macro_predicates_summary_v1"
RUNNER = "scripts/run_observable_macro.py"
LABEL_MODULE = "world_model/data/observable_macro.py"
VALIDATION_COMMAND = "python -u -m scripts.run_observable_macro --validate"

STREAMS = {"fit": "issue109-fit", "policy": "issue109-policy", "evaluation": "issue109-evaluation"}
SELECTION_SOURCES = ("fit", "n1", "i93", "i94")       # the debounce rule reads these only
ITEM1_SOURCES = ("n1", "i93", "i94", "fit", "policy")  # labels and prevalence (never the evaluation split)
SOURCE_NAMES = {"fit": "#104 stream issue109-fit (#109 fit split)",
                "policy": "#104 stream issue109-policy (#109 policy split)",
                "evaluation": "#104 stream issue109-evaluation (#109 held-out split)",
                "n1": "#77 N1 captures (issue-77-n1-v1, complete shots)",
                "i93": "#93 oracle captures (grid + offset)", "i94": "#94 oracle captures (angle x power)"}
SCAN_PROCESSES = 44
VALIDATE_RESCAN = 6          # first shots per source (frozen order) re-scanned by --validate
SMOKE_SHOTS = 4


def log(message):
    print(f"[issue-113] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def read_json(path):
    with open(path) as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json_text(value))
    temporary.replace(path)


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def relative(path):
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------------------
# shot inventories (frozen order: sorted keys)
# ---------------------------------------------------------------------------

LEVEL_TYPE = re.compile(r"novelty_level_(\d+)\W+(type\d{6})")


def level_type_of(source_bindings):
    text = json.dumps(source_bindings)
    for _ in range(4):
        text = unquote(text)
    match = LEVEL_TYPE.search(text)
    if match is None:
        raise ValueError("no template level type in the source bindings")
    return f"novelty_level_{match.group(1)}/{match.group(2)}"


def single_capture(attempt):
    roots = sorted(Path(attempt).glob("aligned/capture-v2:*"))
    roots = [r for r in roots if (r / "native-manifest.json").is_file()]
    if len(roots) != 1:
        raise ValueError(f"{attempt}: expected one native trace, found {len(roots)}")
    return roots[0]


@functools.lru_cache(maxsize=None)
def campaign_rows():
    return read_json(CAMPAIGN / "branches.json")


@functools.lru_cache(maxsize=None)
def shots(source):
    """[{source, key, level, level_type, attempt, native_root}] in sorted-key order."""
    out = []
    if source in STREAMS:
        stream = STREAMS[source]
        for row in campaign_rows():
            if row["stream"] != stream or row["status"] != "complete":
                continue
            attempt = CAMPAIGN / stream / "attempts" / row["identity"]
            out.append({"source": source, "key": row["identity"], "level": row["level"],
                        "level_type": f"novelty_level_0/{row['level_type']}", "attempt": relative(attempt)})
    elif source == "n1":
        for path in sorted((N1 / "results").glob("*.json")):
            result = read_json(path)
            if result.get("failure") or not result.get("segments"):
                continue
            segment = result["segments"][0]
            out.append({"source": source, "key": path.stem, "level": result["member_identity"].rsplit("-", 1)[0],
                        "level_type": level_type_of(segment["source_bindings"]),
                        "attempt": relative(N1 / "attempts" / path.stem)})
    elif source in ORACLE:
        for attempt in sorted(ORACLE[source].iterdir()):
            if not list(attempt.glob("aligned/capture-v2:*/native-manifest.json")):
                continue  # typed failures of #93/#94 hold no trace
            segment = read_json(attempt / "shot-1" / "segment.json")
            out.append({"source": source, "key": attempt.name, "level": attempt.name.split("--")[1],
                        "level_type": level_type_of(segment["source_bindings"]), "attempt": relative(attempt)})
    else:
        raise ValueError(f"unknown source {source}")
    out.sort(key=lambda s: s["key"])
    return tuple(out)


def native_root(shot):
    return single_capture(ROOT / shot["attempt"])


def summary_path(output, source, key):
    return Path(output) / "summaries" / source / f"{key}.json.gz"


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------

def mark(output, phase, wall, gpu=0.0, cpu=0.0, **extra):
    path = Path(output) / "ledger" / f"{phase}.json"
    previous = read_json(path) if path.is_file() else {"wall_seconds": 0.0, "gpu_seconds": 0.0, "cpu_seconds": 0.0}
    write_json(path, {**previous, **extra, "wall_seconds": previous["wall_seconds"] + wall,
                      "gpu_seconds": previous["gpu_seconds"] + gpu, "cpu_seconds": previous["cpu_seconds"] + cpu,
                      "engine_seconds": 0, "last_at": utc_now()})


def ledger(output):
    return {p.stem: read_json(p) for p in sorted((Path(output) / "ledger").glob("*.json"))}


# ---------------------------------------------------------------------------
# scan (definition-free summaries)
# ---------------------------------------------------------------------------

def scan(output, sources, processes=SCAN_PROCESSES):
    if "evaluation" in sources and Path(output) == OUTPUT1:
        raise ValueError("item 1 never reads the evaluation split")
    for source in sources:
        rows = shots(source)
        jobs = [(str(native_root(s)), str(summary_path(output, source, s["key"]))) for s in rows
                if not summary_path(output, source, s["key"]).is_file()]
        log(f"scan {source}: {len(jobs)} of {len(rows)} traces to summarize")
        if not jobs:
            continue
        began, cpu, done = time.monotonic(), 0.0, 0
        with ProcessPoolExecutor(processes, mp_context=multiprocessing.get_context("spawn")) as pool:
            for _, _, seconds in pool.map(om.summary_worker, jobs, chunksize=4):
                cpu += seconds
                done += 1
                if done % 1000 == 0:
                    log(f"scan {source}: {done}/{len(jobs)} ({time.monotonic() - began:.0f}s)")
        mark(output, f"scan-{source}", time.monotonic() - began, cpu=cpu, traces=len(rows))
    return 0


def summaries(output, source):
    for shot in shots(source):
        yield shot, om.load_summary(summary_path(output, source, shot["key"]))


def manifest_rows(output, sources):
    return {source: {s["key"]: sha256_of(summary_path(output, source, s["key"])) for s in shots(source)}
            for source in sources}


# ---------------------------------------------------------------------------
# debounce selection (pre-freeze; fit + retained only)
# ---------------------------------------------------------------------------

def select(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("item 1 is frozen; the selection is part of the frozen plan")
    began = time.monotonic()
    pooled = [summary for source in SELECTION_SOURCES for _, summary in summaries(output, source)]
    chosen, table = om.select_debounce(pooled)
    per_source = {}
    for source in SELECTION_SOURCES:
        _, rows = om.select_debounce([s for _, s in summaries(output, source)])
        per_source[source] = rows
    record = {"schema": SCHEMA_SELECTION, "identity": IDENTITY1, "selected_at": utc_now(),
              "rule": om.DEFINITIONS["debounce_rule"], "candidates": list(om.DEBOUNCE_CANDIDATES),
              "relapse_horizon_frames": om.RELAPSE_HORIZON, "relapse_max": om.RELAPSE_MAX,
              "sources": {s: {"name": SOURCE_NAMES[s], "shots": len(shots(s))} for s in SELECTION_SOURCES},
              "pooled": table, "per_source": per_source, "chosen_debounce_frames": chosen,
              "exploration_disclosure": ("before the rule was written, a 400-shot random sample of the fit stream "
                                         "was scanned to size the relapse rates (fit stream only)")}
    write_json(output / "selection.json", record)
    mark(output, "select", time.monotonic() - began)
    log(f"debounce: {chosen} frames; pooled table {[(r['k'], r['rate']) for r in table]}")
    return 0


# ---------------------------------------------------------------------------
# prevalence (Acceptance 1)
# ---------------------------------------------------------------------------

def empty_block():
    return {"shots": 0, "frames": 0, "phase_frames": [0] * len(om.PHASES),
            "available": [[[0] * len(om.DELTAS) for _ in om.PREDICATES] for _ in range(len(om.PHASES))],
            "positives": [[[0] * len(om.DELTAS) for _ in om.PREDICATES] for _ in range(len(om.PHASES))],
            "shots_with_positive": [[0] * len(om.DELTAS) for _ in om.PREDICATES]}


def add_shot(block, labels):
    values, available, phase = labels["values"], labels["available"], labels["phase"]
    block["shots"] += 1
    block["frames"] += int(labels["frames"])
    for index in range(len(om.PHASES)):
        rows = phase == index
        block["phase_frames"][index] += int(rows.sum())
        for p in range(len(om.PREDICATES)):
            for d in range(len(om.DELTAS)):
                block["available"][index][p][d] += int(available[rows, p, d].sum())
                block["positives"][index][p][d] += int((values[rows, p, d] & available[rows, p, d]).sum())
    for p in range(len(om.PREDICATES)):
        for d in range(len(om.DELTAS)):
            block["shots_with_positive"][p][d] += int(bool((values[:, p, d] & available[:, p, d]).any()))


def prevalence_record(output, source, k):
    blocks = defaultdict(empty_block)
    terminals = Counter()
    onsets = Counter()
    for shot, summary in summaries(output, source):
        labels = om.shot_labels(summary, k)
        add_shot(blocks[shot["level_type"]], labels)
        terminals[f"{shot['level_type']}|{summary['terminal']}"] += 1
        onsets[f"{shot['level_type']}|contact={labels['contact_onset_frame'] is not None}"
               f"|settled={labels['settled_onset_frame'] is not None}"] += 1
    return {"schema": SCHEMA_PREVALENCE, "identity": IDENTITY1, "source": source, "name": SOURCE_NAMES[source],
            "debounce_frames": k, "deltas": list(om.DELTAS), "predicates": list(om.PREDICATES),
            "phases": list(om.PHASES), "level_types": {t: blocks[t] for t in sorted(blocks)},
            "terminals": dict(sorted(terminals.items())), "onsets": dict(sorted(onsets.items()))}


def record_path(output, source):
    return Path(output) / "records" / f"prevalence--{source}.json"


# ---------------------------------------------------------------------------
# smoke, plan, run (item 1)
# ---------------------------------------------------------------------------

def smoke1(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("item 1 is frozen")
    selection = read_json(output / "selection.json")
    k = selection["chosen_debounce_frames"]
    checks = []
    began = time.monotonic()
    timings = []
    for source in ("fit", "policy", "n1", "i93", "i94"):
        for shot in shots(source)[:SMOKE_SHOTS]:
            t0 = time.monotonic()
            summary = om.trace_summary(native_root(shot))
            timings.append(time.monotonic() - t0)
            stored = summary_path(output, source, shot["key"])
            if stored.is_file():
                checks.append({"check": "rescan_equals_stored", "source": source, "key": shot["key"],
                               "ok": om.load_summary(stored) == json.loads(json.dumps(summary))})
            labels = om.shot_labels(summary, k)
            frames = labels["frames"]
            ok = (labels["values"].shape == (frames, len(om.PREDICATES), len(om.DELTAS))
                  and not (labels["values"] & ~labels["available"]).any()
                  and all(not labels["available"][:max(k, d), 0, j].any() for j, d in enumerate(om.DELTAS))
                  and all(not labels["available"][:d, 1, j].any() for j, d in enumerate(om.DELTAS)))
            checks.append({"check": "label_shapes_and_masks", "source": source, "key": shot["key"], "ok": bool(ok),
                           "frames": frames, "terminal": summary["terminal"],
                           "positives": labels["values"].sum((0,)).tolist()})
    # synthetic checks of the rules
    synthetic = {"schema": om.SCHEMA_SUMMARY, "frame_steps": [100 + 50 * f for f in range(40)],
                 "moving": [0] * 5 + [1] * 10 + [0] * 25, "launch_step": 150, "bird_removed_step": 300,
                 "support_index": [0] * 20 + [1] * 20, "support_sets": [[], ["a>b"]], "contact_onset_step": 400}
    labels = om.shot_labels(synthetic, 3, (1, 5))
    settle = labels["values"][:, 0, 0]
    change = labels["values"][:, 1, 1]
    checks.append({"check": "synthetic_settle", "ok": settle.tolist() == [False] * 17 + [True] * 23})
    checks.append({"check": "synthetic_change", "ok": change.tolist() == [False] * 20 + [True] * 5 + [False] * 15})
    checks.append({"check": "synthetic_phase", "ok": labels["phase"].tolist() == [0] * 6 + [1] * 11 + [2] * 23})
    record = {"schema": SCHEMA_SMOKE1, "identity": IDENTITY1, "at": utc_now(), "debounce_frames": k,
              "checks": checks, "ok": all(c["ok"] for c in checks),
              "seconds_per_trace": {"mean": float(np.mean(timings)), "max": float(np.max(timings))},
              "projection": {"policy_traces": len(shots("policy")),
                             "wall_seconds_at_44_processes": len(shots("policy")) * float(np.mean(timings)) / 44}}
    write_json(output / "smoke.json", record)
    mark(output, "smoke1", time.monotonic() - began)
    log(f"smoke1 ok={record['ok']}")
    return 0 if record["ok"] else 1


def frozen_plan1(output, frozen_at):
    selection = read_json(output / "selection.json")
    k = selection["chosen_debounce_frames"]
    return {
        "schema": SCHEMA_PLAN1, "identity": IDENTITY1, "frozen_at": frozen_at, "issue": 113, "item": 1,
        "parent": 108, "git_head_at_freeze": git_head(),
        "question": ("redefine the macro predicates so they are observable at their own request horizon: settle "
                     "with an explicit debounce, structure change as the per-Delta aggregated support-set "
                     "difference with no single-step spike, declared per Delta in {1, 5, 15}"),
        "definitions": {**om.DEFINITIONS, "debounce_frames": k, "deltas": list(om.DELTAS),
                        "predicates": list(om.PREDICATES), "phases": list(om.PHASES),
                        "rest_speed": om.REST_SPEED, "rest_angular_deg": om.REST_ANGULAR_DEG,
                        "settle_window_frames": {str(d): max(k, d) for d in om.DELTAS}},
        "label_module": {"path": LABEL_MODULE, "sha256": sha256_of(ROOT / LABEL_MODULE),
                         "frozen_constant": f"DEBOUNCE_FRAMES = {k}"},
        "selection": {"path": relative(output / "selection.json"), "sha256": sha256_of(output / "selection.json"),
                      "sources": list(SELECTION_SOURCES), "chosen_debounce_frames": k},
        "smoke": {"path": relative(output / "smoke.json"), "sha256": sha256_of(output / "smoke.json")},
        "runner_at_freeze": {"path": RUNNER, "sha256": sha256_of(ROOT / RUNNER),
                             "note": ("informational: the item-3 modes are added to this runner after this "
                                      "freeze and the item-3 plan pins it; --validate regenerates every item-1 "
                                      "record byte for byte from the pinned label module")},
        "inputs": [{"path": relative(p), "sha256": sha256_of(p)} for p in (CAMPAIGN / "branches.json",
                                                                          CAPTURE_WINDOW)],
        "summaries_manifest": {"path": relative(output / "summaries-manifest.json"),
                               "sha256": sha256_of(output / "summaries-manifest.json"),
                               "sources": list(SELECTION_SOURCES)},
        "sources": {s: {"name": SOURCE_NAMES[s], "shots": len(shots(s))} for s in ITEM1_SOURCES},
        "outputs": {"prevalence": ("per source x level type x phase x predicate x Delta: available frames, positive "
                                   "frames, shots with a positive; terminal reasons; onsets"),
                    "labels": ("not stored: every label is a pure function of a stored summary "
                               "(observable_macro.shot_labels(load_summary(path)))")},
        "prohibited": ["reading any issue109-evaluation trace or label before the item-3 freeze",
                       "changing a definition after this freeze", "retrying typed failures"],
        "engine_seconds": 0,
    }


def prepare1(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("item 1 is already frozen")
    smoke = read_json(output / "smoke.json")
    selection = read_json(output / "selection.json")
    if not smoke["ok"] or smoke["debounce_frames"] != selection["chosen_debounce_frames"]:
        raise ValueError("smoke failed or ran on another debounce")
    if om.DEBOUNCE_FRAMES != selection["chosen_debounce_frames"]:
        raise ValueError("freeze observable_macro.DEBOUNCE_FRAMES to the selected debounce first")
    write_json(output / "summaries-manifest.json", {"schema": SCHEMA_MANIFEST, "identity": IDENTITY1,
                                                    "summaries": manifest_rows(output, SELECTION_SOURCES)})
    plan = frozen_plan1(output, utc_now())
    write_json(output / "plan.json", plan)
    log(f"item 1 frozen at {plan['frozen_at']} (debounce {om.DEBOUNCE_FRAMES})")
    return 0


def load_plan1(output):
    plan = read_json(Path(output) / "plan.json")
    if plan.get("schema") != SCHEMA_PLAN1 or plan.get("identity") != IDENTITY1:
        raise ValueError("item-1 plan binding differs")
    problems = []
    if sha256_of(ROOT / LABEL_MODULE) != plan["label_module"]["sha256"]:
        problems.append("label module drifted after the item-1 freeze")
    for entry in plan["inputs"] + [plan["selection"], plan["smoke"], plan["summaries_manifest"]]:
        if sha256_of(ROOT / entry["path"]) != entry["sha256"]:
            problems.append(f"{entry['path']} drifted")
    if om.DEBOUNCE_FRAMES != plan["definitions"]["debounce_frames"]:
        problems.append("debounce differs from the frozen value")
    if problems:
        raise ValueError("; ".join(problems))
    return plan


def run1(output):
    output = Path(output)
    plan = load_plan1(output)
    scan(output, ("policy",))
    began = time.monotonic()
    for source in ITEM1_SOURCES:
        write_json(record_path(output, source), prevalence_record(output, source, plan["definitions"]["debounce_frames"]))
    write_json(output / "summaries-manifest-run.json", {"schema": SCHEMA_MANIFEST, "identity": IDENTITY1,
                                                        "summaries": manifest_rows(output, ("policy",))})
    mark(output, "prevalence", time.monotonic() - began)
    return 0


# ---------------------------------------------------------------------------
# publication (item 1)
# ---------------------------------------------------------------------------

def ratio(num, den):
    return None if den == 0 else num / den


def totals(block, phase=None, predicate=0, delta=0):
    phases = range(len(om.PHASES)) if phase is None else (phase,)
    available = sum(block["available"][i][predicate][delta] for i in phases)
    positives = sum(block["positives"][i][predicate][delta] for i in phases)
    return positives, available


def prevalence_rows(records):
    rows = []
    for source, record in records.items():
        for level_type, block in record["level_types"].items():
            for p, predicate in enumerate(om.PREDICATES):
                for d, delta in enumerate(om.DELTAS):
                    for phase in ("all", *om.PHASES):
                        index = None if phase == "all" else om.PHASES.index(phase)
                        positives, available = totals(block, index, p, d)
                        rows.append({"source": source, "level_type": level_type, "predicate": predicate,
                                     "delta": delta, "phase": phase, "shots": block["shots"],
                                     "phase_frames": (block["frames"] if index is None
                                                      else block["phase_frames"][index]),
                                     "available": available, "positives": positives,
                                     "prevalence": ratio(positives, available),
                                     "shots_with_positive": block["shots_with_positive"][p][d] if index is None else ""})
    return rows


def prevalence_csv(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if v is None else (f"{v:.6f}" if isinstance(v, float) else v)) for k, v in row.items()})
    return buffer.getvalue()


def pct(value):
    return "n/a" if value is None else f"{100 * value:.2f}%"


def compute1(output):
    entries = ledger(output)
    return {"schema": SCHEMA_COMPUTE1, "identity": IDENTITY1, "phases": entries,
            "wall_seconds": sum(e["wall_seconds"] for e in entries.values()),
            "cpu_process_seconds": sum(e["cpu_seconds"] for e in entries.values()),
            "gpu_seconds": sum(e["gpu_seconds"] for e in entries.values()), "engine_seconds": 0,
            "hardware": "48 CPU threads (scan: 44 processes); no GPU used by item 1"}


def findings1(plan, records, selection, compute):
    k = plan["definitions"]["debounce_frames"]
    lines = [f"# #113 item 1: observable macro predicates (`{IDENTITY1}`)", "",
             f"Frozen {plan['frozen_at']} before any head training. Label module `{LABEL_MODULE}` "
             f"(`DEBOUNCE_FRAMES = {k}`). Zero engine seconds.", "", "## Definitions (frozen)", ""]
    for name in ("rest", "settle", "structure-change", "label_index", "phases", "relation_to_110"):
        lines.append(f"- **{name}**: {plan['definitions'][name]}")
    lines += ["", f"Settle window D = max({k}, Delta): " + ", ".join(
        f"Delta {d}: {w} frames" for d, w in plan["definitions"]["settle_window_frames"].items()), "",
        "## Debounce selection (fit stream + retained captures only)", "",
        f"Rule: {selection['rule']}.", "", "| k | settle frames (with 15-frame look-ahead) | relapsed | rate |",
        "|---|---|---|---|"]
    for row in selection["pooled"]:
        lines.append(f"| {row['k']} | {row['frames']} | {row['relapsed']} | {pct(row['rate'])} |")
    lines += ["", f"Chosen: **{k} frames**.", "", "## Prevalence (all phases; positives / available frames)", "",
              "| source | level type | shots | " + " | ".join(f"{p} Δ{d}" for p in om.PREDICATES for d in om.DELTAS)
              + " |", "|---|---|---|" + "---|" * (len(om.PREDICATES) * len(om.DELTAS))]
    for source, record in records.items():
        for level_type, block in record["level_types"].items():
            cells = []
            for p in range(len(om.PREDICATES)):
                for d in range(len(om.DELTAS)):
                    positives, available = totals(block, None, p, d)
                    cells.append(f"{positives} / {available} ({pct(ratio(positives, available))})")
            lines.append(f"| {source} | {level_type} | {block['shots']} | " + " | ".join(cells) + " |")
    lines += ["", "## Prevalence per phase (Delta = 1 / 5 / 15 positives, pooled over level types)", "",
              "| source | phase | frames | " + " | ".join(om.PREDICATES) + " |", "|---|---|---|---|---|"]
    for source, record in records.items():
        for index, phase in enumerate(om.PHASES):
            frames = sum(b["phase_frames"][index] for b in record["level_types"].values())
            cells = []
            for p in range(len(om.PREDICATES)):
                cells.append(" / ".join(str(sum(totals(b, index, p, d)[0] for b in record["level_types"].values()))
                                        for d in range(len(om.DELTAS))))
            lines.append(f"| {source} | {phase} | {frames} | " + " | ".join(cells) + " |")
    lines += ["", "Full table (per level type x phase x predicate x Delta, with available frames and shots with a "
              "positive): `prevalence.csv`.", "",
              f"Compute: {compute['cpu_process_seconds']:.0f} CPU process-seconds, {compute['wall_seconds']:.0f} s "
              "wall, 0 GPU s, 0 engine s.", ""]
    return "\n".join(lines)


def rendered1(output):
    output = Path(output)
    plan = load_plan1(output)
    records = {s: read_json(record_path(output, s)) for s in ITEM1_SOURCES}
    selection = read_json(output / "selection.json")
    compute = compute1(output)
    rows = prevalence_rows(records)
    summary = {"schema": SCHEMA_SUMMARY1, "identity": IDENTITY1, "frozen_at": plan["frozen_at"],
               "debounce_frames": plan["definitions"]["debounce_frames"], "deltas": list(om.DELTAS),
               "prevalence_all_phases": {
                   s: {lt: {f"{p}|{d}": dict(zip(("positives", "available"), totals(b, None, i, j)))
                            for i, p in enumerate(om.PREDICATES) for j, d in enumerate(om.DELTAS)}
                       for lt, b in r["level_types"].items()} for s, r in records.items()},
               "records": {s: sha256_of(record_path(output, s)) for s in ITEM1_SOURCES},
               "label_api": "world_model/data/observable_macro.py (see docs/issue-113-observable-macro.md)",
               "engine_seconds": 0}
    return {"compute.json": json_text(compute), "summary.json": json_text(summary),
            "prevalence.csv": prevalence_csv(rows), "findings.md": findings1(plan, records, selection, compute)}


def publish1(output):
    output = Path(output)
    for name, text in rendered1(output).items():
        (output / name).write_text(text)
    log("item 1 published")
    return 0


def validate1(output):
    output = Path(output)
    problems = []
    plan = load_plan1(output)
    k = plan["definitions"]["debounce_frames"]
    stored = read_json(output / "summaries-manifest.json")["summaries"]
    stored.update(read_json(output / "summaries-manifest-run.json")["summaries"])
    current = manifest_rows(output, ITEM1_SOURCES)
    if current != stored:
        problems.append("summary bytes differ from the manifests")
    for source in ITEM1_SOURCES:
        for shot in shots(source)[:VALIDATE_RESCAN]:
            fresh = json.loads(json.dumps(om.trace_summary(native_root(shot))))
            if fresh != om.load_summary(summary_path(output, source, shot["key"])):
                problems.append(f"re-scan of {source}/{shot['key']} differs")
        if json_text(prevalence_record(output, source, k)) != record_path(output, source).read_text():
            problems.append(f"prevalence record {source} differs")
    for name, text in rendered1(output).items():
        if (output / name).read_text() != text:
            problems.append(f"{name} differs")
    return problems



# ---------------------------------------------------------------------------
# item 3: the temporal macro head (own freeze, after the item-1 freeze)
# ---------------------------------------------------------------------------

OUTPUT3 = ROOT / ".local-artifacts/issue-113-temporal-macro-head-v1"
IDENTITY3 = "issue-113-temporal-macro-head-v1"
SCHEMA_PLAN3 = "issue_113_temporal_head_plan_v1"
SCHEMA_SMOKE3 = "issue_113_temporal_head_smoke_v1"
SCHEMA_CARRIERS = "issue_113_carriers_v1"
SCHEMA_DATA = "issue_113_head_data_v1"
SCHEMA_HEAD = "issue_113_head_v1"
SCHEMA_RECORD = "issue_113_readout_record_v1"
SCHEMA_SELECT3 = "issue_113_head_selection_v1"
SCHEMA_MANIFEST3 = "issue_113_records_manifest_v1"
SCHEMA_COMPUTE3 = "issue_113_temporal_head_compute_v1"
SCHEMA_SUMMARY3 = "issue_113_temporal_head_summary_v1"
HEAD_MODULE = "world_model/training/temporal_macro_head.py"
RETRAIN = ROOT / ".local-artifacts/issue-112-v3-retrain-v1"
HANDOFF = RETRAIN / "gate_b_handoff.json"
E99_ENCODER = ROOT / ".local-artifacts/issue-99-slot-encoder-fix-v1/encoder/encoder.pt"

DEVICE = "cuda"
SEEDS = (20260908, 20260909, 20260910)        # final heads; head seed i is paired with Gate-B seed i
CV_SEED = SEEDS[0]
SINGLE = 1                                     # K = 1: the same-capacity single-frame head on the new labels
HISTORIES = (2, 3, 4)                          # temporal candidates, selected by grouped leave-out
MIXES = (0.0, 0.5)                             # share of each batch read from Gate-B one-step predictions
CONFIGS = tuple((k, mix) for k in (SINGLE, *HISTORIES) for mix in MIXES)
FOLDS = 5
FOLD_SEED = 11301
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)
AS_IS_THRESHOLD = 0.5                          # the #100 READ_THRESHOLD on the gated macro value
G1_ROW_RATE = 0.9                              # G1: share of fit rows (0..285) within the harness tolerance of v3
G1_TEXT = ("the fresh fit parse (deterministic_scoring policy) agrees with the v3 retrain carriers on positions 0..285: "
           f">= {G1_ROW_RATE} of rows within the scoring_harness parse tolerance (a pipeline error fails every row; "
           "the v3 data mode ran under torch's default flags, cuDNN TF32 allowed, so slots near a presence or kind "
           "boundary differ numerically: the smoke saw 95 % of rows within tolerance and one kind-index flip)")
TRAIN = {"steps": 6000, "batch_per_delta": 2048, "lr": 1e-3, "weight_decay": 1e-4, "schedule": "cosine to 0",
         "normalisation_rows": 200000, "optimizer": "AdamW"}
SOURCES3 = ("engine", "predicted", "rollout")
SOURCE_TEXT = {
    "engine": "teacher-forced: the history and the landing carrier are E99 parses of the observed frames",
    "predicted": ("one Delta-step Gate-B prediction (pair (Delta, macro)) from the observed carrier at the anchor "
                  "f - Delta; the history before the landing is observed (the #100 'after one Delta prediction' "
                  "audit form; the first step of a Delta-step rollout)"),
    "rollout": ("open-loop Gate-B rollout from the decision frame at stride Delta (pair (Delta, macro)); landing "
                "frames n * Delta up to the shot's last frame; the history is the rollout's own earlier landings "
                "(the #112 deployment form; DESCRIPTIVE)"),
}
HEADS = ("temporal", "single", "as_is")
CONTRASTS = (("temporal", "as_is", "temporal head - current head (Acceptance 3/4)"),
             ("temporal", "single", "temporal-context effect (same labels, same width)"),
             ("single", "as_is", "label effect (same single-frame architecture)"))
PRIMARY_SOURCES = ("engine", "predicted")
MARGIN = 0.05
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 11302
REFERENCE_F1 = 0.7                             # the #100 macro readout bar, reported for reference only
STOP_TOKEN = "readiness_or_precision_insufficient"
TOKENS = ("supported", "not_supported_by_this_experiment", STOP_TOKEN)
GPU_CAP_SECONDS = 40 * 3600.0
WALL_CAP_SECONDS = 48 * 3600.0
PARSE_WORKERS = 4
SCORE_CHUNK = 1 << 19


def heavy():
    """Item-3 imports (torch, the E99 parser, the Gate-B loader); never imported by item-1 workers."""
    import torch
    from scripts import run_decision_dynamics_retrain as rt
    from scripts import run_slot_encoder_fix as s99
    from world_model.planning import scoring_harness as harness
    from world_model.training import temporal_macro_head as tmh
    return torch, rt, s99, harness, tmh


def split_shots(split):
    return shots(split)


def carrier_path(output, split, key):
    return Path(output) / "carriers" / split / f"{key}.pt"


def data_path(output, split):
    return Path(output) / "data" / f"{split}.pt"


def head_path(output, name):
    return Path(output) / "models" / f"{name}.pt"


def record_path3(output, name):
    return Path(output) / "records" / f"{name}.json"


def config_name(k, mix):
    return f"K{k}-mix{round(100 * mix)}"


def head_name(stage, config, seed, fold=None):
    return f"{stage}--{config}--seed{seed}" + ("" if fold is None else f"--fold{fold}")


def summary_root(split):
    return OUTPUT1 if split == "fit" else OUTPUT3


def frames_of(split, key):
    return len(om.load_summary(summary_path(summary_root(split), split, key))["frame_steps"])


# ------------------------------------------------------------ plan binding

def load_plan3(output):
    plan = read_json(Path(output) / "plan.json")
    if plan.get("schema") != SCHEMA_PLAN3 or plan.get("identity") != IDENTITY3:
        raise ValueError("item-3 plan binding differs")
    load_plan1(OUTPUT1)
    problems = [f"{e['path']} drifted" for e in plan["inputs"] + [plan["smoke"]]
                if sha256_of(ROOT / e["path"]) != e["sha256"]]
    if problems:
        raise ValueError("; ".join(problems))
    return plan


# ------------------------------------------------------------ carriers (GPU)

def carrier_jobs(split):
    """Every complete shot of the split, parsed whole (positions 0..last) in one deterministic pipeline."""
    return [shot["key"] for shot in split_shots(split)]


def v3_comparison(fresh, reference, tolerance):
    """Fresh parse vs the v3 retrain carriers on their positions: per-row harness replication counts."""
    torch, _, _, harness, _ = heavy()
    reference, fresh = reference.double(), fresh[:len(reference)].double()
    delta = (reference - fresh).abs()
    motion = torch.zeros(delta.shape[1], dtype=torch.bool)
    motion[harness.motion_columns(delta.shape[1])] = True
    nonmotion = delta[:, ~motion].amax(1)
    scaled = (delta[:, motion] * reference[:, harness.ELAPSED_COLUMN, None] / 2).amax(1)
    rows_pass = (nonmotion <= tolerance) & (scaled <= tolerance)
    return {"rows": len(delta), "rows_pass": int(rows_pass.sum()), "max_abs_nonmotion_delta": float(nonmotion.max()),
            "max_motion_scaled": float(scaled.max())}


def build_carriers(output, split, key, adapter, attempt):
    torch, rt, s99, harness, _ = heavy()
    frames = len(s99.shot_frames(str(ROOT / attempt))[2])
    carriers = s99.carriers_for(adapter, str(ROOT / attempt), list(range(frames)))
    segment = read_json(ROOT / attempt / "shot-1" / "segment.json")
    entry = {"schema": SCHEMA_CARRIERS, "identity": IDENTITY3, "key": key, "split": split, "frames": frames,
             "action": s99.action_of(segment), "carriers": carriers}
    if split == "fit":
        entry["v3_comparison"] = v3_comparison(carriers, rt.read_shot(rt.OUTPUT, key)["carriers"],
                                               harness.PARSE_TOLERANCE)
    path = carrier_path(output, split, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(entry, path.with_name(path.name + ".tmp"))
    path.with_name(path.name + ".tmp").replace(path)
    return entry


def carriers_mode(output, workers, worker, splits=("fit", "evaluation")):
    output = Path(output)
    load_plan3(output)
    torch, _, s99, harness, _ = heavy()
    began = time.monotonic()
    adapter = s99.load_adapter(s99.OUTPUT, "E")
    attempts = {s["key"]: s["attempt"] for split in splits for s in split_shots(split)}
    done = 0
    mark(output, f"carriers-start-worker{worker}", 0.0, started_at=utc_now())
    with harness.deterministic_scoring():
        for split in splits:
            jobs = carrier_jobs(split)[worker::workers]
            pending = [key for key in jobs if not carrier_path(output, split, key).is_file()]
            log(f"carriers {split} worker {worker}: {len(pending)} of {len(jobs)} shots")
            for key in pending:
                build_carriers(output, split, key, adapter, attempts[key])
                done += 1
                if done % 250 == 0:
                    log(f"carriers worker {worker}: {done} shots ({time.monotonic() - began:.0f}s)")
    wall = time.monotonic() - began
    mark(output, f"carriers-worker{worker}", wall, gpu=wall, shots=done)
    return 0


def scan_evaluation(output):
    output = Path(output)
    plan = load_plan3(output)
    if utc_now() <= plan["frozen_at"]:
        raise ValueError("the evaluation split is read only after the item-3 freeze")
    mark(output, "scan-evaluation-start", 0.0, started_at=utc_now())
    scan(output, ("evaluation",))
    return 0


# ------------------------------------------------------------ assembled data

def shot_carriers(output, split, key):
    torch, *_ = heavy()
    entry = torch.load(carrier_path(output, split, key), map_location="cpu", weights_only=True)
    if entry["key"] != key or entry["schema"] != SCHEMA_CARRIERS or entry["split"] != split:
        raise ValueError(f"carriers of {split}/{key} bound elsewhere")
    return entry["carriers"], entry["action"], entry.get("v3_comparison")


def v3_summary(rows):
    """G1 statistics over the fit shots: row pass rate at the harness tolerance and per-shot maxima."""
    if not rows:
        return {"shots": 0}
    out = {"shots": len(rows), "rows": sum(r["rows"] for r in rows), "rows_pass": sum(r["rows_pass"] for r in rows),
           "max_abs_nonmotion_delta": max(r["max_abs_nonmotion_delta"] for r in rows),
           "max_motion_scaled": max(r["max_motion_scaled"] for r in rows)}
    out["row_pass_rate"] = out["rows_pass"] / out["rows"]
    out["pass"] = out["row_pass_rate"] >= G1_ROW_RATE
    return out


def assemble(output, split):
    output = Path(output)
    load_plan3(output)
    return _assemble(output, split, split_shots(split))


def _assemble(output, split, rows):
    """One tensor file per split: carriers [N, 236] (all shots, all frames), per-row shot start, position,
    level and labels (frozen observable_macro definitions), per-shot action; plus a small meta JSON."""
    torch, *_ = heavy()
    began = time.monotonic()
    levels = sorted({s["level"] for s in rows})
    level_index = {level: i for i, level in enumerate(levels)}
    carriers, values, available, phase, starts, positions, shot_level, actions = [], [], [], [], [], [], [], []
    replication, mismatched = [], []
    offset = 0
    for i, shot in enumerate(rows):
        summary = om.load_summary(summary_path(summary_root(split), split, shot["key"]))
        labels = om.shot_labels(summary)
        z, action, check = shot_carriers(output, split, shot["key"])
        if len(z) != labels["frames"]:
            mismatched.append(shot["key"])
            continue
        if check is not None:
            replication.append(check)
        count = len(z)
        carriers.append(z.float())
        values.append(torch.from_numpy(labels["values"]))
        available.append(torch.from_numpy(labels["available"]))
        phase.append(torch.from_numpy(labels["phase"]).to(torch.int8))
        starts.append(torch.full((count,), offset, dtype=torch.int64))
        positions.append(torch.arange(count, dtype=torch.int64))
        shot_level.append(level_index[shot["level"]])
        actions.append(action.float())
        offset += count
        if (i + 1) % 2000 == 0:
            log(f"assemble {split}: {i + 1}/{len(rows)} shots")
    shot_of_row = torch.cat([torch.full((len(c),), j, dtype=torch.int32) for j, c in enumerate(carriers)])
    payload = {"schema": SCHEMA_DATA, "identity": IDENTITY3, "split": split, "levels": levels,
               "keys": [s["key"] for s in rows if s["key"] not in set(mismatched)],
               "carriers": torch.cat(carriers), "values": torch.cat(values), "available": torch.cat(available),
               "phase": torch.cat(phase), "start": torch.cat(starts), "position": torch.cat(positions),
               "shot": shot_of_row, "shot_level": torch.tensor(shot_level, dtype=torch.int64),
               "action": torch.stack(actions), "mismatched": mismatched,
               "v3_comparison": v3_summary(replication)}
    path = data_path(output, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path.with_name(path.name + ".tmp"))
    path.with_name(path.name + ".tmp").replace(path)
    write_json(path.with_name(f"{split}.meta.json"), {
        "schema": SCHEMA_DATA, "identity": IDENTITY3, "split": split, "rows": offset, "shots": len(payload["keys"]),
        "levels": len(levels), "mismatched": len(mismatched), "mismatched_keys": mismatched,
        "v3_comparison": payload["v3_comparison"], "sha256": sha256_of(path),
        "positives": {f"{p}|{d}": int((payload["values"][:, i, j] & payload["available"][:, i, j]).sum())
                      for i, p in enumerate(om.PREDICATES) for j, d in enumerate(om.DELTAS)},
        "available": {f"{p}|{d}": int(payload["available"][:, i, j].sum())
                      for i, p in enumerate(om.PREDICATES) for j, d in enumerate(om.DELTAS)}})
    mark(output, f"assemble-{split}", time.monotonic() - began, rows=offset, shots=len(payload["keys"]),
         mismatched=len(mismatched))
    log(f"assembled {split}: {offset} rows, {len(payload['keys'])} shots, {len(mismatched)} mismatched")
    return 0


def load_data(output, split, device=DEVICE):
    torch, *_ = heavy()
    payload = torch.load(data_path(output, split), map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA_DATA or payload.get("split") != split:
        raise ValueError(f"data {split} binding differs")
    for name in ("carriers", "values", "available", "phase", "start", "position", "shot", "shot_level", "action"):
        payload[name] = payload[name].to(device)
    payload["row_level"] = payload["shot_level"][payload["shot"].long()]
    return payload


# ------------------------------------------------------------ folds

def folds(levels):
    """Pre-declared grouped leave-out: levels are the groups; per family, the sorted levels are permuted
    by PCG64(FOLD_SEED) and dealt round-robin into FOLDS folds."""
    out = {}
    families = sorted({level.split("-")[-2] for level in levels})
    for family in families:
        members = sorted(level for level in levels if level.split("-")[-2] == family)
        order = np.random.Generator(np.random.PCG64(FOLD_SEED)).permutation(len(members))
        for rank, index in enumerate(order.tolist()):
            out[members[index]] = rank % FOLDS
    return out


# ------------------------------------------------------------ training

def train_head(data, train_rows, k, seed, mix, steps=None):
    """Class-balanced training of one head (K = k) on the rows where ``train_rows`` is true. With ``mix`` > 0,
    that share of each batch reads the paired Gate-B member's one-step prediction of the landing carrier
    (pair (Delta, macro) from the observed anchor f - Delta) instead of the observed one, so the head is
    trained on the carriers its interface also reads inside a rollout."""
    torch, _, _, harness, tmh = heavy()
    from world_model.model import Abstraction, PredictionPair
    steps = TRAIN["steps"] if steps is None else steps
    began = time.monotonic()
    model = gate_b(seed) if mix > 0 else None
    torch.manual_seed(seed)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    head = tmh.TemporalMacroHead(k).to(DEVICE)
    candidates, weights, samples = {}, {}, {}
    for d, delta in enumerate(om.DELTAS):
        rows = torch.nonzero(train_rows & data["available"][:, :, d].any(1)).flatten()
        candidates[delta] = rows
        weights[delta] = tmh.pos_weights(data["values"][rows, :, d], data["available"][rows, :, d]).to(DEVICE)
        pick = rows[torch.randint(len(rows), (min(TRAIN["normalisation_rows"], len(rows)),), device=DEVICE,
                                  generator=generator)]
        samples[delta] = tmh.TemporalMacroHead.raw_features(
            tmh.gather_history(data["carriers"], data["start"][pick], data["position"][pick], delta, k))
    tmh.fit_normalisation(head, samples)
    del samples
    optimizer = torch.optim.AdamW(head.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    trace = []
    predicted_rows = int(round(mix * TRAIN["batch_per_delta"]))
    with harness.deterministic_scoring():
        for step in range(steps):
            losses = []
            for d, delta in enumerate(om.DELTAS):
                rows = candidates[delta]
                pick = rows[torch.randint(len(rows), (TRAIN["batch_per_delta"],), device=DEVICE, generator=generator)]
                history = tmh.gather_history(data["carriers"], data["start"][pick], data["position"][pick], delta, k)
                if predicted_rows:
                    part = pick[:predicted_rows]
                    with torch.no_grad():
                        landing = model.carrier(data["carriers"][data["start"][part] + data["position"][part] - delta],
                                                data["action"][data["shot"][part].long()],
                                                PredictionPair(delta, Abstraction.MACRO))
                    history = history.clone()
                    history[:predicted_rows, -1] = landing
                losses.append(tmh.balanced_loss(head(history, delta), data["values"][pick, :, d],
                                                data["available"][pick, :, d], weights[delta]))
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            if step % 250 == 0 or step == steps - 1:
                trace.append({"step": step, "loss": float(loss.detach())})
    torch.cuda.synchronize()
    return head.eval(), {"steps": steps, "seconds": time.monotonic() - began, "loss_trace": trace,
                         "pos_weight": {str(d): weights[d].tolist() for d in om.DELTAS},
                         "training_rows": {str(d): len(candidates[d]) for d in om.DELTAS}}


def save_head(output, name, head, k, seed, extra):
    torch, *_ = heavy()
    path = head_path(output, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA_HEAD, "identity": IDENTITY3, "name": name, "history": k, "seed": seed,
               "state": {key: value.cpu() for key, value in head.state_dict().items()}, **extra}
    torch.save(payload, path.with_name(path.name + ".tmp"))
    path.with_name(path.name + ".tmp").replace(path)


def load_head(output, name):
    torch, _, _, _, tmh = heavy()
    payload = torch.load(head_path(output, name), map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA_HEAD or payload.get("name") != name:
        raise ValueError(f"head {name} binding differs")
    head = tmh.TemporalMacroHead(payload["history"])
    head.load_state_dict(payload["state"], strict=True)
    return head.to(DEVICE).eval(), payload


def gate_b(seed):
    _, rt, *_ = heavy()
    model = rt.implementation().load_member(rt.OUTPUT, "LC", "hybrid", "all", seed)
    if model is None:
        raise ValueError(f"Gate-B member {seed} is retired")
    return model


def gate_b_check():
    handoff = read_json(HANDOFF)
    rows = []
    for entry in handoff["unit"]["checkpoints"]:
        actual = sha256_of(ROOT / entry["path"])
        rows.append({"seed": entry["seed"], "path": entry["path"], "sha256": actual, "ok": actual == entry["sha256"]
                     and entry["usable"]})
    return {"checkpoints": rows, "pass": all(r["ok"] for r in rows) and len(rows) == len(SEEDS)}


# ------------------------------------------------------------ readout (always under deterministic_scoring)

def confusion(prob, values, available, level, phase, levels, thresholds):
    """[levels, 2, phases, T, 4] counts (tp, fp, fn, tn) of prob >= threshold."""
    torch, *_ = heavy()
    th = torch.tensor(thresholds, device=prob.device)
    T = len(thresholds)
    out = torch.zeros(levels * 2 * len(om.PHASES) * T * 4, dtype=torch.int64, device=prob.device)
    predicate = torch.arange(2, device=prob.device)[None, :, None]
    t_index = torch.arange(T, device=prob.device)[None, None, :]
    for begin in range(0, len(prob), SCORE_CHUNK):
        sl = slice(begin, begin + SCORE_CHUNK)
        pred = prob[sl, :, None] >= th
        truth = values[sl, :, None]
        cell = (~pred).long() * 2 + (~truth).long()
        key = ((((level[sl, None, None] * 2 + predicate) * len(om.PHASES) + phase[sl, None, None].long()) * T
                + t_index) * 4 + cell)
        mask = available[sl, :, None].expand_as(pred)
        out += torch.bincount(key[mask], minlength=out.numel())
    return out.reshape(levels, 2, len(om.PHASES), T, 4)


def landing_rows(data, delta):
    """Rows a Delta request can land on (an observed anchor f - Delta exists)."""
    torch, *_ = heavy()
    return torch.nonzero(data["position"] >= delta).flatten()


def as_is_values(model, z):
    """The current head as-is: gated values of steady-state / structure-unstable (#100 ``gated``)."""
    from world_model.model import Abstraction
    return model.symbolic_features(z, Abstraction.MACRO)[:, :2]


def readout(data, head_kind, head, model, source, delta, d):
    """(rows, probabilities [R, 2]) of one head on one source at one Delta."""
    torch, _, _, _, tmh = heavy()
    from world_model.model import Abstraction, PredictionPair
    pair = PredictionPair(delta, Abstraction.MACRO)
    k = 1 if head_kind == "as_is" else head.history
    if source in ("engine", "predicted"):
        rows = landing_rows(data, delta)
        probs = []
        for begin in range(0, len(rows), SCORE_CHUNK):
            r = rows[begin:begin + SCORE_CHUNK]
            history = tmh.gather_history(data["carriers"], data["start"][r], data["position"][r], delta, max(k, 1))
            if source == "predicted":
                anchor = data["carriers"][data["start"][r] + data["position"][r] - delta]
                action = data["action"][data["shot"][r].long()]
                history = history.clone()
                history[:, -1] = model.carrier(anchor, action, pair)
            if head_kind == "as_is":
                probs.append(as_is_values(model, history[:, -1]))
            else:
                probs.append(torch.sigmoid(head(history, delta)))
        return rows, torch.cat(probs)
    # open-loop rollout from the decision frame, chunked over shots
    shot_count = len(data["keys"])
    first = data["start"][torch.nonzero(data["position"] == 0).flatten()]
    lengths = torch.bincount(data["shot"].long(), minlength=shot_count)
    all_rows, probs = [], []
    for begin in range(0, shot_count, 2048):
        shots_ = torch.arange(begin, min(begin + 2048, shot_count), device=DEVICE)
        z = data["carriers"][first[shots_]]
        action = data["action"][shots_]
        steps = int((lengths[shots_].max() - 1) // delta)
        chain = [z]
        for _ in range(steps):
            chain.append(model.carrier(chain[-1], action, pair))
        chain = torch.stack(chain, 1)                                     # [S, n + 1, D]
        for n in range(1, steps + 1):
            live = torch.nonzero(n * delta <= lengths[shots_] - 1).flatten()
            if len(live) == 0:
                continue
            index = torch.arange(n - k + 1, n + 1, device=DEVICE).clamp_min(0)
            history = chain[live][:, index]
            rows = first[shots_[live]] + n * delta
            if head_kind == "as_is":
                probs.append(as_is_values(model, history[:, -1]))
            else:
                probs.append(torch.sigmoid(head(history, delta)))
            all_rows.append(rows)
    return torch.cat(all_rows), torch.cat(probs)


def record_counts(data, head_kind, head, model, source, thresholds):
    torch, *_ = heavy()
    counts = {}
    finite = True
    with torch.no_grad():
        for d, delta in enumerate(om.DELTAS):
            rows, prob = readout(data, head_kind, head, model, source, delta, d)
            finite = finite and bool(torch.isfinite(prob).all())
            block = confusion(prob, data["values"][rows, :, d], data["available"][rows, :, d],
                              data["row_level"][rows], data["phase"][rows], len(data["levels"]), thresholds)
            counts[str(delta)] = block.cpu().tolist()
    return counts, finite


def write_record(output, name, record):
    path = record_path3(output, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def score_record(output, data, split, head_kind, name, seed, source, thresholds, fold=None):
    """One readout record (counts per level x predicate x phase x threshold x cell, per Delta)."""
    torch, _, _, harness, _ = heavy()
    with harness.deterministic_scoring() as policy:
        model = gate_b(seed)
        head = None if head_kind == "as_is" else load_head(output, name)[0]
        counts, finite = record_counts(data, head_kind, head, model, source, thresholds)
    return {"schema": SCHEMA_RECORD, "identity": IDENTITY3, "split": split, "head": head_kind, "model": name,
            "gate_b_seed": seed, "source": source, "fold": fold, "thresholds": list(thresholds),
            "levels": data["levels"], "deltas": list(om.DELTAS), "phases": list(om.PHASES),
            "predicates": list(om.PREDICATES), "cells": ["tp", "fp", "fn", "tn"], "finite": finite,
            "scoring_policy": policy, "counts": counts}


# ------------------------------------------------------------ grouped leave-out (fit)

def cv_tasks():
    tasks = [("as_is", None, None, None)]
    for k, mix in CONFIGS:
        for fold in range(FOLDS):
            tasks.append(("head", k, mix, fold))
    return tasks


def cv_name(task):
    kind, k, mix, fold = task
    return "cv--as_is" if kind == "as_is" else head_name("cv", config_name(k, mix), CV_SEED, fold)


def cv_mode(output, workers, worker):
    output = Path(output)
    load_plan3(output)
    torch, *_ = heavy()
    began = time.monotonic()
    data = load_data(output, "fit")
    fold_of = folds(data["levels"])
    level_fold = torch.tensor([fold_of[level] for level in data["levels"]], device=DEVICE)
    row_fold = level_fold[data["row_level"]]
    trained = 0
    for task in cv_tasks()[worker::workers]:
        name = cv_name(task)
        kind, k, mix, fold = task
        if all(record_path3(output, f"{name}--{s}").is_file() for s in ("engine", "predicted")):
            continue
        if kind == "head" and not head_path(output, name).is_file():
            head, report = train_head(data, row_fold != fold, k, CV_SEED, mix)
            save_head(output, name, head, k, CV_SEED, {"mix": mix, "fold": fold, "report": report})
            trained += 1
        subset = held_subset(data, row_fold == fold) if kind == "head" else data
        head_kind = "as_is" if kind == "as_is" else ("single" if k == SINGLE else "temporal")
        for source in ("engine", "predicted"):
            record = score_record(output, subset, "fit", head_kind, name, CV_SEED, source,
                                  (AS_IS_THRESHOLD,) if kind == "as_is" else THRESHOLDS, fold)
            write_record(output, f"{name}--{source}", record)
        log(f"cv worker {worker}: {name} done ({time.monotonic() - began:.0f}s)")
    wall = time.monotonic() - began
    mark(output, f"cv-worker{worker}", wall, gpu=wall, trained=trained)
    return 0


def held_subset(data, rows_mask):
    """The rows of whole shots in the held fold, keeping row indices into the full carrier table (the history
    of a held row stays inside its own shot)."""
    torch, *_ = heavy()
    rows = torch.nonzero(rows_mask).flatten()
    subset = dict(data)
    keep = {"values", "available", "phase", "start", "position", "shot", "row_level"}
    for name in keep:
        subset[name] = data[name][rows]
    subset["rows_in_full"] = rows
    return subset


# ------------------------------------------------------------ selection (fit only)

def f1_of(c):
    """c [..., 4] counts -> F1 (0 when tp + fp + fn = 0)."""
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def record_array(record):
    """[deltas, levels, 2, phases, T, 4] int array."""
    return np.array([record["counts"][str(d)] for d in om.DELTAS], dtype=np.int64)


def cv_counts(output, config, source):
    total = None
    for fold in range(FOLDS):
        array = record_array(read_json(record_path3(output, f"{head_name('cv', config, CV_SEED, fold)}--{source}")))
        total = array if total is None else total + array
    return total


def threshold_source(source):
    """The threshold set a source reads: the engine set for engine carriers, the predicted set otherwise."""
    return "engine" if source == "engine" else "predicted"


def select3(output):
    output = Path(output)
    load_plan3(output)
    began = time.monotonic()
    rows = {}
    for k, mix in CONFIGS:
        config = config_name(k, mix)
        row = {"history": k, "mix": mix, "thresholds": {}, "cv_macro_f1": {}, "cv_macro_f1_at_0.5": {}}
        for source in ("engine", "predicted"):
            f1 = f1_of(cv_counts(output, config, source).sum((1, 3)))       # [deltas, 2, T]
            best = f1.argmax(-1)                                            # first maximum: the smaller threshold
            row["thresholds"][source] = {f"{om.PREDICATES[p]}|{delta}": THRESHOLDS[int(best[d, p])]
                                         for d, delta in enumerate(om.DELTAS) for p in range(2)}
            row["cv_macro_f1"][source] = float(np.mean([f1[d, p, best[d, p]] for d in range(len(om.DELTAS))
                                                        for p in range(2)]))
            row["cv_macro_f1_at_0.5"][source] = float(f1[..., THRESHOLDS.index(0.5)].mean())
        row["score"] = (row["cv_macro_f1"]["engine"] + row["cv_macro_f1"]["predicted"]) / 2
        rows[config] = row

    def best(candidates):
        return max(candidates, key=lambda c: (rows[c]["score"], -rows[c]["history"], -rows[c]["mix"]))
    temporal = best([config_name(k, m) for k, m in CONFIGS if k != SINGLE])
    single = best([config_name(k, m) for k, m in CONFIGS if k == SINGLE])
    record = {"schema": SCHEMA_SELECT3, "identity": IDENTITY3, "selected_at": utc_now(), "rule": SELECTION_RULE,
              "candidates": rows, "selected": {"temporal": temporal, "single": single},
              "thresholds": {"temporal": rows[temporal]["thresholds"], "single": rows[single]["thresholds"],
                             "as_is": AS_IS_THRESHOLD}}
    write_json(output / "selection.json", record)
    mark(output, "select", time.monotonic() - began)
    log(f"selected temporal {temporal}, single {single}")
    return 0


SELECTION_RULE = ("for every candidate (K, mix): per source (engine, predicted) and per (predicate, Delta) the "
                  f"threshold is the first maximiser of the pooled out-of-fold F1 over {list(THRESHOLDS)}; the "
                  "candidate's score is the mean of its engine and predicted macro F1 (mean over predicates x Delta, "
                  "every phase) at those thresholds. Temporal head: the best K in {2, 3, 4} x mix in {0, 0.5}; "
                  "single-frame head: the best mix for K = 1; ties to the smaller K, then the smaller mix. Fit split "
                  "only; the rollout source reads the predicted thresholds.")


# ------------------------------------------------------------ final heads and held-out scoring

def final_tasks(output):
    selected = read_json(Path(output) / "selection.json")["selected"]
    out = []
    for kind in ("temporal", "single"):
        k, mix = next((k, m) for k, m in CONFIGS if config_name(k, m) == selected[kind])
        out += [(kind, k, mix, seed) for seed in SEEDS]
    return out


def final_mode(output, workers, worker):
    output = Path(output)
    load_plan3(output)
    torch, *_ = heavy()
    began = time.monotonic()
    data = load_data(output, "fit")
    every = torch.ones(len(data["position"]), dtype=torch.bool, device=DEVICE)
    trained = 0
    for kind, k, mix, seed in final_tasks(output)[worker::workers]:
        name = head_name("final", config_name(k, mix), seed)
        if head_path(output, name).is_file():
            continue
        head, report = train_head(data, every, k, seed, mix)
        save_head(output, name, head, k, seed, {"mix": mix, "fold": None, "report": report})
        trained += 1
    wall = time.monotonic() - began
    mark(output, f"final-worker{worker}", wall, gpu=wall, trained=trained)
    return 0


def heldout_tasks(output):
    selected = read_json(Path(output) / "selection.json")["selected"]
    tasks = []
    for seed in SEEDS:
        for source in SOURCES3:
            for kind in ("temporal", "single"):
                tasks.append((kind, head_name("final", selected[kind], seed), seed, source))
            tasks.append(("as_is", f"gate-b--seed{seed}", seed, source))
    return tasks


def heldout_name(task):
    kind, name, seed, source = task
    return f"heldout--{kind}--seed{seed}--{source}"


def score_mode(output, workers, worker):
    output = Path(output)
    plan = load_plan3(output)
    if not gate_b_check()["pass"]:
        raise ValueError("Gate-B checkpoints differ from the handoff")
    began = time.monotonic()
    data = load_data(output, "evaluation")
    replayed = None
    for task in heldout_tasks(output)[worker::workers]:
        kind, name, seed, source = task
        out = heldout_name(task)
        if record_path3(output, out).is_file():
            continue
        thresholds = (AS_IS_THRESHOLD,) if kind == "as_is" else THRESHOLDS
        record = score_record(output, data, "evaluation", kind, name, seed, source, thresholds)
        write_record(output, out, record)
        if replayed is None:   # G5: the worker's first record is scored twice
            again = score_record(output, data, "evaluation", kind, name, seed, source, thresholds)
            replayed = {"record": out, "equal": again["counts"] == record["counts"]}
            write_json(output / "replay" / f"worker{worker}.json", replayed)
        log(f"score worker {worker}: {out}")
    del plan
    wall = time.monotonic() - began
    mark(output, f"score-worker{worker}", wall, gpu=wall)
    return 0


# ------------------------------------------------------------ analysis (publication)

def selected_counts(record, thresholds):
    """[levels, deltas, 2, phases, 4] counts at the per-(predicate, Delta) threshold (as-is: its only one)."""
    array = record_array(record)                                          # [deltas, levels, 2, phases, T, 4]
    out = np.zeros(array.shape[1:2] + (len(om.DELTAS), 2) + array.shape[3:4] + (4,), dtype=np.int64)
    for d, delta in enumerate(om.DELTAS):
        for p, predicate in enumerate(om.PREDICATES):
            t = 0 if len(record["thresholds"]) == 1 else record["thresholds"].index(thresholds[f"{predicate}|{delta}"])
            out[:, d, p] = array[d, :, p, :, t]
    return out


def draws(levels):
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    return generator.multinomial(levels, np.full(levels, 1.0 / levels), size=BOOTSTRAP_DRAWS).astype(np.float64)


def f1_block(stack, weights, phase=None):
    """stack [S, L, deltas, 2, phases, 4]; -> point [deltas, 2], draws [B, deltas, 2] (seed means)."""
    counts = stack.sum(4) if phase is None else stack[..., phase, :]
    point = f1_of(counts.sum(1).astype(np.float64)).mean(0)
    boot = np.einsum("bl,sldpc->bsdpc", weights, counts.astype(np.float64))
    return point, f1_of(boot).mean(1)


def interval(point, boot):
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"estimate": round(float(point), 6), "interval": [round(float(low), 6), round(float(high), 6)],
            "label": "DESCRIPTIVE level-clustered percentile bootstrap"}


def stacks(output, split, selection):
    """{(head, source): [S, L, deltas, 2, phases, 4]} of the scored records at their frozen thresholds."""
    out = {}
    if split == "fit":
        for config, row in selection["candidates"].items():
            for source in ("engine", "predicted"):
                total = None
                for fold in range(FOLDS):
                    record = read_json(record_path3(output, f"{head_name('cv', config, CV_SEED, fold)}--{source}"))
                    part = selected_counts(record, row["thresholds"][source])
                    total = part if total is None else total + part
                out[(config, source)] = total[None]
        for source in ("engine", "predicted"):
            out[("as_is", source)] = selected_counts(read_json(record_path3(output, f"cv--as_is--{source}")),
                                                     None)[None]
        return out
    for kind in HEADS:
        for source in SOURCES3:
            parts = []
            for seed in SEEDS:
                record = read_json(record_path3(output, f"heldout--{kind}--seed{seed}--{source}"))
                parts.append(selected_counts(record, None if kind == "as_is"
                                             else selection["thresholds"][kind][threshold_source(source)]))
            out[(kind, source)] = np.stack(parts)
    return out


def readout_table(stack_map, levels):
    weights = draws(levels)
    table = {}
    for (head, source), stack in stack_map.items():
        for phase_name in ("all", *om.PHASES):
            phase = None if phase_name == "all" else om.PHASES.index(phase_name)
            point, boot = f1_block(stack, weights, phase)
            counts = (stack.sum(4) if phase is None else stack[..., phase, :]).sum(1).mean(0)   # [deltas, 2, 4]
            block = {"macro_f1": interval(point.mean(), boot.mean((1, 2)))}
            for d, delta in enumerate(om.DELTAS):
                for p, predicate in enumerate(om.PREDICATES):
                    tp, fp, fn, tn = counts[d, p].tolist()
                    block[f"{predicate}|{delta}"] = {**interval(point[d, p], boot[:, d, p]),
                                                     "engine_positives": tp + fn, "predicted_positives": tp + fp,
                                                     "tp": tp, "fp": fp, "fn": fn, "tn": tn}
            table[f"{head}|{source}|{phase_name}"] = block
    return table


def contrasts(stack_map, levels, guards_pass, sources, tokens=True):
    weights = draws(levels)
    out = {}
    for source in sources:
        for first, second, meaning in CONTRASTS:
            a_point, a_boot = f1_block(stack_map[(first, source)], weights)
            b_point, b_boot = f1_block(stack_map[(second, source)], weights)
            point = float(a_point.mean() - b_point.mean())
            boot = a_boot.mean((1, 2)) - b_boot.mean((1, 2))
            block = {**interval(point, boot), "meaning": meaning, "margin": MARGIN}
            if not tokens or source not in PRIMARY_SOURCES:
                block["token"] = None
            elif not guards_pass:
                block["token"] = STOP_TOKEN
            else:
                block["token"] = ("supported" if block["interval"][0] > 0 and point >= MARGIN
                                  else "not_supported_by_this_experiment")
            out[f"{first}-{second}|{source}"] = block
    return out


def guards(output, plan):
    output = Path(output)
    fit = torch_payload_meta(output, "fit")
    evaluation = torch_payload_meta(output, "evaluation")
    entries = ledger(output)
    scan_start = entries.get("scan-evaluation-start", {}).get("started_at")
    carrier_start = min((e.get("started_at") for name, e in entries.items()
                         if name.startswith("carriers-start") and e.get("started_at")), default=None)
    replays = [read_json(p) for p in sorted((output / "replay").glob("*.json"))]
    records = [read_json(p) for p in sorted((output / "records").glob("heldout--*.json"))]
    rows = {
        "G1_carrier_replication": {"text": G1_TEXT,
                                   **fit["v3_comparison"], "ok": bool(fit["v3_comparison"].get("pass"))},
        "G2_gate_b_checkpoints": {"text": "the Gate-B unit checkpoints match the handoff sha256", **gate_b_check(),
                                  "ok": gate_b_check()["pass"]},
        "G3_evaluation_isolation": {"text": "no evaluation-split trace, carrier or label was read before the item-3 "
                                            "freeze", "frozen_at": plan["frozen_at"], "scan_started_at": scan_start,
                                    "carriers_started_at": carrier_start,
                                    "ok": bool(scan_start and carrier_start and scan_start > plan["frozen_at"]
                                               and carrier_start > plan["frozen_at"])},
        "G4_frame_alignment": {"text": "every shot's carrier rows equal its trace-summary frames",
                               "fit_mismatched": fit["mismatched"], "evaluation_mismatched": evaluation["mismatched"],
                               "ok": fit["mismatched"] == 0 and evaluation["mismatched"] == 0},
        "G5_deterministic_rescore": {"text": "each scoring worker's first record scored twice is identical",
                                     "replays": replays, "ok": bool(replays) and all(r["equal"] for r in replays)},
        "G6_finite": {"text": "every held-out readout is finite", "ok": bool(records) and all(r["finite"]
                                                                                            for r in records)},
        "G7_compute_caps": {"text": f"summed GPU s <= {GPU_CAP_SECONDS:.0f} and summed process wall s <= "
                                    f"{WALL_CAP_SECONDS:.0f}",
                            "gpu_seconds": sum(e["gpu_seconds"] for e in entries.values()),
                            "wall_seconds": sum(e["wall_seconds"] for e in entries.values()),
                            "ok": (sum(e["gpu_seconds"] for e in entries.values()) <= GPU_CAP_SECONDS
                                   and sum(e["wall_seconds"] for e in entries.values()) <= WALL_CAP_SECONDS)},
    }
    return rows, all(r["ok"] for r in rows.values())


def torch_payload_meta(output, split):
    path = Path(output) / "data" / f"{split}.meta.json"
    return read_json(path)


def compute3(output):
    entries = ledger(output)
    return {"schema": SCHEMA_COMPUTE3, "identity": IDENTITY3, "phases": entries,
            "gpu_seconds": sum(e["gpu_seconds"] for e in entries.values()),
            "wall_seconds": sum(e["wall_seconds"] for e in entries.values()),
            "cpu_process_seconds": sum(e["cpu_seconds"] for e in entries.values()), "engine_seconds": 0,
            "caps": {"gpu_seconds": GPU_CAP_SECONDS, "wall_seconds": WALL_CAP_SECONDS},
            "hardware": "4x RTX 5090 (32 GB) instead of the #85 RTX 3090; 48 CPU threads"}


def analysis(output):
    output = Path(output)
    plan = load_plan3(output)
    selection = read_json(output / "selection.json")
    guard_rows, guards_pass = guards(output, plan)
    fit_levels = len(read_json(record_path3(output, "cv--as_is--engine"))["levels"])
    held_levels = len(read_json(record_path3(output, f"heldout--as_is--seed{SEEDS[0]}--engine"))["levels"])
    cv_stack = stacks(output, "fit", selection)
    held_stack = stacks(output, "evaluation", selection)
    named = {selection["selected"]["temporal"]: "temporal", selection["selected"]["single"]: "single"}
    cv_map = {(named.get(h, h), s): v for (h, s), v in cv_stack.items()}
    return {"plan": plan, "selection": selection, "guards": guard_rows, "guards_pass": guards_pass,
            "cv": readout_table(cv_stack, fit_levels),
            "cv_contrasts": contrasts(cv_map, fit_levels, guards_pass, ("engine", "predicted"), tokens=False),
            "heldout": readout_table(held_stack, held_levels),
            "heldout_contrasts": contrasts(held_stack, held_levels, guards_pass, SOURCES3),
            "levels": {"fit": fit_levels, "evaluation": held_levels}}


def fmt_block(block):
    low, high = block["interval"]
    return f"{block['estimate']:.3f} [{low:.3f}, {high:.3f}]"


def findings3(result, compute):
    plan, selection = result["plan"], result["selection"]
    selected = selection["selected"]
    lines = [f"# #113 item 3: temporal macro head (`{IDENTITY3}`)", "",
             f"Frozen {plan['frozen_at']} (item-1 definitions frozen {plan['item1_frozen_at']}). Zero engine seconds. "
             "Every interval is a level-clustered percentile bootstrap (10 000 draws, PCG64 "
             f"{BOOTSTRAP_SEED}), DESCRIPTIVE.", "",
             f"Selected by grouped leave-out: temporal head **{selected['temporal']}**, single-frame head "
             f"**{selected['single']}** (K = stride-Delta history length; mix = share of training rows read from "
             f"Gate-B one-step predictions). Current head: threshold {AS_IS_THRESHOLD}. Thresholds per source "
             "(engine / predicted) in `selection.json`.", "",
             "## Guards", ""]
    for name, row in result["guards"].items():
        lines.append(f"- {name}: {'pass' if row['ok'] else 'FAIL'} — {row['text']}")
    if not result["guards_pass"]:
        lines += ["", f"A guard failed: every token is `{STOP_TOKEN}` and no comparison is interpretable."]
    columns = [f"{p} Δ{d}" for p in om.PREDICATES for d in om.DELTAS]
    for title, key, heads, sources in (
            ("Grouped leave-out (fit split, 5 level-grouped folds; heads seed "
             f"{CV_SEED}; current head and predictor in-sample)", "cv",
             [config_name(k, m) for k, m in CONFIGS] + ["as_is"], ("engine", "predicted")),
            ("Held-out (#109 evaluation split, scored once; seed mean of 3)", "heldout", list(HEADS), SOURCES3)):
        lines += ["", f"## {title}: F1 per redefined predicate", "",
                  "| head | source | macro F1 | " + " | ".join(columns) + " |", "|---|---|---|" + "---|" * len(columns)]
        for head in heads:
            for source in sources:
                block = result[key][f"{head}|{source}|all"]
                cells = [f"{block[f'{p}|{d}']['estimate']:.3f}" for p in om.PREDICATES for d in om.DELTAS]
                lines.append(f"| {head} | {source} | {fmt_block(block['macro_f1'])} | " + " | ".join(cells) + " |")
    lines += ["", "## Held-out contrasts (macro F1 difference, paired level bootstrap; margin "
              f"{MARGIN})", "", "| contrast | source | difference | token |", "|---|---|---|---|"]
    for name, block in result["heldout_contrasts"].items():
        contrast, source = name.split("|")
        lines.append(f"| {contrast} ({block['meaning']}) | {source} | {fmt_block(block)} | {block['token'] or 'descriptive'} |")
    lines += ["", "## Grouped leave-out contrasts (descriptive)", "", "| contrast | source | difference |",
              "|---|---|---|"]
    for name, block in result["cv_contrasts"].items():
        contrast, source = name.split("|")
        lines.append(f"| {contrast} | {source} | {fmt_block(block)} |")
    lines += ["", "## Item-4 readout target: held-out readout counts per phase (temporal head; seed mean)", "",
              "| source | phase | predicate | Δ | engine positives | predicted positives | F1 |", "|---|---|---|---|---|---|---|"]
    for source in SOURCES3:
        for phase in ("all", *om.PHASES):
            block = result["heldout"][f"temporal|{source}|{phase}"]
            for p in om.PREDICATES:
                for d in om.DELTAS:
                    cell = block[f"{p}|{d}"]
                    lines.append(f"| {source} | {phase} | {p} | {d} | {cell['engine_positives']:.0f} | "
                                 f"{cell['predicted_positives']:.1f} | {cell['estimate']:.3f} |")
    lines += ["", f"Reference: the #100 macro readout bar was F1 >= {REFERENCE_F1}; it is reported, not tested.", "",
              f"Compute: {compute['gpu_seconds']:.0f} GPU s, {compute['wall_seconds']:.0f} s summed process wall, "
              f"{compute['cpu_process_seconds']:.0f} CPU process-s, 0 engine s.", ""]
    return "\n".join(lines)


def readout_csv(result):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["split", "head", "source", "phase", "predicate", "delta", "f1", "low", "high", "engine_positives",
                     "predicted_positives", "tp", "fp", "fn", "tn"])
    for split in ("cv", "heldout"):
        for key, block in result[split].items():
            head, source, phase = key.split("|")
            writer.writerow([split, head, source, phase, "macro", "", block["macro_f1"]["estimate"],
                             *block["macro_f1"]["interval"], "", "", "", "", "", ""])
            for p in om.PREDICATES:
                for d in om.DELTAS:
                    c = block[f"{p}|{d}"]
                    writer.writerow([split, head, source, phase, p, d, c["estimate"], *c["interval"],
                                     c["engine_positives"], c["predicted_positives"], c["tp"], c["fp"], c["fn"], c["tn"]])
    return buffer.getvalue()


def records_manifest(output):
    return {"schema": SCHEMA_MANIFEST3, "identity": IDENTITY3,
            "records": {p.name: sha256_of(p) for p in sorted((Path(output) / "records").glob("*.json"))},
            "heads": {p.name: sha256_of(p) for p in sorted((Path(output) / "models").glob("*.pt"))}}


def rendered3(output):
    result = analysis(output)
    compute = compute3(output)
    summary = {"schema": SCHEMA_SUMMARY3, "identity": IDENTITY3, "frozen_at": result["plan"]["frozen_at"],
               "selected": result["selection"]["selected"],
               "thresholds": result["selection"]["thresholds"], "guards_pass": result["guards_pass"],
               "guards": {k: v["ok"] for k, v in result["guards"].items()},
               "grouped_leave_out": {k: v["macro_f1"] for k, v in result["cv"].items() if k.endswith("|all")},
               "heldout": {k: {"macro_f1": v["macro_f1"],
                               **{f"{p}|{d}": {"estimate": v[f"{p}|{d}"]["estimate"],
                                               "interval": v[f"{p}|{d}"]["interval"]}
                                  for p in om.PREDICATES for d in om.DELTAS}}
                           for k, v in result["heldout"].items() if k.endswith("|all")},
               "heldout_contrasts": result["heldout_contrasts"], "item4_readout_target": (
                   "macro F1 per redefined predicate on the #104 held-out split (#109 evaluation stream): the "
                   "held-out rows of this summary; temporal head, engine source = teacher-forced"),
               "engine_seconds": 0}
    return {"compute.json": json_text(compute), "summary.json": json_text(summary),
            "readout.csv": readout_csv(result), "findings.md": findings3(result, compute),
            "records-manifest.json": json_text(records_manifest(output))}


def publish3(output):
    output = Path(output)
    for name, text in rendered3(output).items():
        (output / name).write_text(text)
    log("item 3 published")
    return 0


VALIDATE_RESCORE = (("temporal", 0, "engine"), ("as_is", 0, "predicted"))


def validate3(output):
    """Records manifest, publication bytes, and a fresh deterministic re-score of retained heads."""
    output = Path(output)
    problems = []
    load_plan3(output)
    if json_text(records_manifest(output)) != (output / "records-manifest.json").read_text():
        problems.append("records or heads differ from records-manifest.json")
    for name, text in rendered3(output).items():
        if (output / name).read_text() != text:
            problems.append(f"{name} differs")
    selection = read_json(output / "selection.json")
    data = load_data(output, "evaluation")
    for kind, index, source in VALIDATE_RESCORE:
        seed = SEEDS[index]
        name = (head_name("final", selection["selected"]["temporal"], seed) if kind == "temporal"
                else f"gate-b--seed{seed}")
        thresholds = (AS_IS_THRESHOLD,) if kind == "as_is" else THRESHOLDS
        fresh = score_record(output, data, "evaluation", kind, name, seed, source, thresholds)
        stored = read_json(record_path3(output, f"heldout--{kind}--seed{seed}--{source}"))
        if fresh["counts"] != stored["counts"]:
            problems.append(f"re-score of heldout--{kind}--seed{seed}--{source} differs")
        else:
            log(f"re-scored heldout--{kind}--seed{seed}--{source}: identical")
    fit = load_data(output, "fit")
    fold_of = folds(fit["levels"])
    import torch
    level_fold = torch.tensor([fold_of[level] for level in fit["levels"]], device=DEVICE)
    name = head_name("cv", selection["selected"]["temporal"], CV_SEED, 0)
    subset = held_subset(fit, level_fold[fit["row_level"]] == 0)
    fresh = score_record(output, subset, "fit", "temporal", name, CV_SEED, "engine", THRESHOLDS, 0)
    if fresh["counts"] != read_json(record_path3(output, f"{name}--engine"))["counts"]:
        problems.append(f"re-score of {name}--engine differs")
    return problems


# ------------------------------------------------------------ smoke and freeze (item 3)

SMOKE3_SHOTS = 24


def smoke3(output):
    """Pre-freeze pipeline smoke on fit shots only (scratch directory, removed afterwards): whole-shot
    deterministic carriers with the v3 comparison (G1), assembly, a short training of a temporal and a
    single-frame head, every readout source, a replayed score, and the compute projection. No
    evaluation-split byte is read."""
    import shutil
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("item 3 is frozen")
    load_plan1(OUTPUT1)
    torch, _, s99, harness, _ = heavy()
    scratch = output / "smoke-scratch"
    shutil.rmtree(scratch, ignore_errors=True)
    began = time.monotonic()
    checks = []
    gate = gate_b_check()
    checks.append({"check": "gate_b_checkpoints", "ok": gate["pass"]})
    fit = list(split_shots("fit"))
    chosen = fit[::max(1, len(fit) // SMOKE3_SHOTS)][:SMOKE3_SHOTS]
    adapter = s99.load_adapter(s99.OUTPUT, "E")
    parsed_frames, t0 = 0, time.monotonic()
    comparisons = []
    with harness.deterministic_scoring():
        for shot in chosen:
            entry = build_carriers(scratch, "fit", shot["key"], adapter, shot["attempt"])
            parsed_frames += len(entry["carriers"])
            comparisons.append(entry["v3_comparison"])
    torch.cuda.synchronize()
    parse_rate = parsed_frames / (time.monotonic() - t0)
    g1 = v3_summary(comparisons)
    checks.append({"check": "G1_v3_comparison", "ok": bool(g1["pass"]), **g1})
    _assemble(scratch, "fit", chosen)
    data = load_data(scratch, "fit")
    every = torch.ones(len(data["position"]), dtype=torch.bool, device=DEVICE)
    steps = 100
    timings = {}
    smoke_configs = ((SINGLE, MIXES[0]), (max(HISTORIES), MIXES[-1]))
    for k, mix in smoke_configs:
        t0 = time.monotonic()
        head, report = train_head(data, every, k, CV_SEED, mix, steps=steps)
        timings[f"train_K{k}_seconds_per_step"] = (time.monotonic() - t0) / steps
        name = head_name("smoke", config_name(k, mix), CV_SEED)
        save_head(scratch, name, head, k, CV_SEED, {"mix": mix, "report": report})
        checks.append({"check": f"training_{config_name(k, mix)}_finite",
                       "ok": all(math.isfinite(r["loss"]) for r in report["loss_trace"])})
    rows = len(data["position"])
    for kind, config in (("temporal", config_name(*smoke_configs[1])), ("single", config_name(*smoke_configs[0])),
                         ("as_is", None)):
        name = f"gate-b--seed{CV_SEED}" if kind == "as_is" else head_name("smoke", config, CV_SEED)
        for source in SOURCES3:
            t0 = time.monotonic()
            first = score_record(scratch, data, "fit", kind, name, CV_SEED, source,
                                 (AS_IS_THRESHOLD,) if kind == "as_is" else THRESHOLDS)
            timings[f"score_{kind}_{source}_seconds_per_row"] = (time.monotonic() - t0) / rows
            second = score_record(scratch, data, "fit", kind, name, CV_SEED, source,
                                  (AS_IS_THRESHOLD,) if kind == "as_is" else THRESHOLDS)
            checks.append({"check": f"replay_{kind}_{source}", "ok": first["counts"] == second["counts"]
                           and first["finite"]})
    mean_frames = float(np.mean([frames_of("fit", s["key"]) for s in fit]))
    evaluation_shots = sum(1 for r in campaign_rows() if r["stream"] == STREAMS["evaluation"] and r["status"] == "complete")
    evaluation_frames = evaluation_shots * mean_frames
    fit_rows = sum(frames_of("fit", s["key"]) for s in fit)
    trainings = len(CONFIGS) * FOLDS + 2 * len(SEEDS)
    train_seconds = trainings * TRAIN["steps"] * max(timings[f"train_K{max(HISTORIES)}_seconds_per_step"],
                                                     timings[f"train_K{SINGLE}_seconds_per_step"])
    score_per_row = sum(v for key, v in timings.items() if key.startswith("score_"))
    projection = {
        "parse_frames_per_second_per_gpu": parse_rate, "fit_frames": fit_rows,
        "evaluation_frames_estimate": evaluation_frames, "evaluation_shots": evaluation_shots,
        "carrier_gpu_seconds": (fit_rows + evaluation_frames) / parse_rate,
        "training_gpu_seconds": train_seconds,
        "scoring_gpu_seconds": score_per_row * (evaluation_frames * len(SEEDS) + fit_rows * 2),
    }
    projection["total_gpu_seconds"] = (projection["carrier_gpu_seconds"] + projection["training_gpu_seconds"]
                                       + projection["scoring_gpu_seconds"])
    projection["wall_seconds_on_4_gpus"] = projection["total_gpu_seconds"] / 4
    projection["within_48h_on_4_gpus"] = projection["wall_seconds_on_4_gpus"] < 48 * 3600
    record = {"schema": SCHEMA_SMOKE3, "identity": IDENTITY3, "at": utc_now(), "shots": [s["key"] for s in chosen],
              "rows": rows, "checks": checks, "ok": all(c["ok"] for c in checks), "timings": timings,
              "projection": projection, "kept": "nothing but this record (scratch removed; fit split only)"}
    write_json(output / "smoke.json", record)
    shutil.rmtree(scratch, ignore_errors=True)
    mark(output, "smoke3", time.monotonic() - began, gpu=time.monotonic() - began)
    log(f"smoke3 ok={record['ok']} projection {projection}")
    return 0 if record["ok"] else 1


def frozen_plan3(output, frozen_at):
    from world_model.training import temporal_macro_head as tmh
    plan1 = load_plan1(OUTPUT1)
    handoff = read_json(HANDOFF)
    pinned = [RUNNER, HEAD_MODULE, LABEL_MODULE, relative(OUTPUT1 / "plan.json"), relative(HANDOFF),
              relative(E99_ENCODER), relative(CAMPAIGN / "branches.json"),
              *(c["path"] for c in handoff["unit"]["checkpoints"])]
    return {
        "schema": SCHEMA_PLAN3, "identity": IDENTITY3, "frozen_at": frozen_at, "issue": 113, "item": 3,
        "item1_frozen_at": plan1["frozen_at"], "git_head_at_freeze": git_head(),
        "question": ("does a macro head with temporal context read the frozen observable macro predicates from E99 "
                     "carriers, engine-parsed and predicted by the Gate-B predictor, better than the current "
                     "single-frame head; and what is the held-out macro F1 per redefined predicate (the #113 item-4 "
                     "readout target)?"),
        "owner_decisions": ["item 1 (frozen predicates) runs first and is frozen before any head training",
                            "item 3's held-out score uses item 4's pre-declared readout target: macro F1 per "
                            "redefined predicate on the #104 held-out split (#109 evaluation stream), with the #100 "
                            "audit machinery (readout counts and phases)"],
        "labels": {"module": LABEL_MODULE, "definitions": plan1["definitions"]},
        "interface": tmh.__doc__,
        "heads": {"temporal": (f"TemporalMacroHead(K), K in {list(HISTORIES)} (stride-Delta history) x training mix in "
                               f"{list(MIXES)}, selected by grouped leave-out"),
                  "single": ("TemporalMacroHead(1): the same-capacity single-frame head trained on the new labels, "
                             f"training mix in {list(MIXES)} selected by the same rule"),
                  "as_is": ("the Gate-B member's own macro_head unchanged: gated steady-state value read as settle at "
                            "every Delta, gated structure-unstable value read as structure change at every Delta, "
                            f"threshold {AS_IS_THRESHOLD} (#100 READ_THRESHOLD)")},
        "head_architecture": {"hidden": tmh.HIDDEN, "clip": tmh.CLIP, "per_delta_mlp": True,
                              "loss": "class-balanced BCE (pos_weight = negatives / positives per predicate x Delta)"},
        "training": {**TRAIN, "data": ("every frame of every complete #104 issue109-fit shot (13 919): this ticket's "
                                       "deterministic E99 parse (scoring_harness policy, s99.carriers_for) of "
                                       "positions 0..last; the v3 retrain carriers (0..285) are the G1 reference"),
                     "seeds": {"grouped_leave_out": CV_SEED, "final": list(SEEDS)},
                     "determinism": "training and scoring run under scoring_harness.deterministic_scoring()"},
        "grouped_leave_out": {"groups": "levels (480 fit levels)", "folds": FOLDS,
                              "assignment": (f"per family, sorted levels permuted by PCG64({FOLD_SEED}) and dealt "
                                             "round-robin"),
                              "candidates": {"temporal": [config_name(k, m) for k, m in CONFIGS if k != SINGLE],
                                             "single": [config_name(k, m) for k, m in CONFIGS if k == SINGLE]},
                              "thresholds": list(THRESHOLDS),
                              "rule": SELECTION_RULE,
                              "mix": ("share of each training batch whose landing carrier is the paired Gate-B "
                                      "member's one-step prediction from the observed anchor (CV: seed "
                                      f"{CV_SEED}; final head seed i: Gate-B seed i)"),
                              "current_head": ("in-sample on fit (the Gate-B predictor was trained on the fit split); "
                                               "the predicted source uses the in-sample Gate-B seed "
                                               f"{CV_SEED}")},
        "scoring": {"split": "#109 evaluation stream issue109-evaluation (all complete shots, every frame), scored "
                             "once after the final heads exist", "sources": SOURCE_TEXT,
                    "pairing": "head seed i with Gate-B unit seed i (20260908..10); seed-mean estimates",
                    "readout": "tp/fp/fn/tn per level x predicate x Delta x phase (pre_contact, cascade, settled)",
                    "target": ("item 4: macro F1 per redefined predicate (F1 per predicate x Delta and their mean), "
                               "temporal head, engine source primary; predicted and rollout sources reported"),
                    "reference": f"#100 macro bar F1 >= {REFERENCE_F1}, reported only"},
        "contrasts": [{"first": a, "second": b, "meaning": m} for a, b, m in CONTRASTS],
        "decision_rule": {"statistic": "held-out macro F1 difference (seed means), paired level-clustered bootstrap, "
                                       f"{BOOTSTRAP_DRAWS} draws PCG64({BOOTSTRAP_SEED}), DESCRIPTIVE",
                          "margin": MARGIN, "sources_with_tokens": list(PRIMARY_SOURCES),
                          "supported": "every guard passes AND lower bound > 0 AND point >= margin",
                          "not_supported_by_this_experiment": "every guard passes and not supported",
                          STOP_TOKEN: "any guard fails (then every token is this one)"},
        "guards": [f"G1 {G1_TEXT}",
                   "G2 Gate-B unit checkpoints equal the handoff sha256", "G3 no evaluation-split byte read before "
                   "this freeze (scan and carrier start times after frozen_at)", "G4 carrier rows equal summary frames "
                   "for every shot", "G5 each scoring worker's first record replays identically",
                   "G6 every held-out readout finite", "G7 summed GPU and process wall seconds within the caps"],
        "caps": {"gpu_seconds": GPU_CAP_SECONDS, "wall_seconds": WALL_CAP_SECONDS, "stop": STOP_TOKEN},
        "order": ["--carriers (4 GPU workers: fit and evaluation, whole shots)", "--scan-evaluation",
                  "--assemble fit", "--cv (4 workers)", "--select3", "--final (4 workers)", "--assemble evaluation",
                  "--score (4 workers)", "--publish3"],
        "smoke": {"path": relative(output / "smoke.json"), "sha256": sha256_of(output / "smoke.json")},
        "inputs": [{"path": p, "sha256": sha256_of(ROOT / p)} for p in pinned],
        "prohibited": ["reading evaluation-split labels before this freeze", "changing a recipe, rule or threshold "
                       "grid after this freeze", "retrying typed failures", "relaunching over live workers"],
        "engine_seconds": 0,
    }


def prepare3(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("item 3 is already frozen")
    smoke = read_json(output / "smoke.json")
    if not smoke["ok"] or not smoke["projection"]["within_48h_on_4_gpus"]:
        raise ValueError("smoke failed or the projection exceeds 48 h on 4 GPUs")
    plan = frozen_plan3(output, utc_now())
    write_json(output / "plan.json", plan)
    log(f"item 3 frozen at {plan['frozen_at']}")
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def validate(_):
    problems = validate1(OUTPUT1)
    if (OUTPUT3 / "summary.json").is_file():
        problems += validate3(OUTPUT3)
    for problem in problems:
        log(f"VALIDATE: {problem}")
    log("validate: " + ("ok" if not problems else f"{len(problems)} problem(s)"))
    return 0 if not problems else 1


MODES = {"scan": None, "select": select, "smoke1": smoke1, "prepare1": prepare1, "run1": run1,
         "publish1": publish1, "smoke3": smoke3, "prepare3": prepare3, "carriers": None,
         "scan-evaluation": scan_evaluation, "assemble": None, "cv": None, "select3": select3, "final": None,
         "score": None, "publish3": publish3, "validate": validate}
SHARDED = {"carriers": carriers_mode, "cv": cv_mode, "final": final_mode, "score": score_mode}
ITEM3 = {"smoke3", "prepare3", "carriers", "scan-evaluation", "assemble", "cv", "select3", "final", "score",
         "publish3"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        group.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--sources", default=",".join(SELECTION_SOURCES))
    parser.add_argument("--split", choices=("fit", "evaluation"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker", type=int, default=0)
    args = parser.parse_args()
    mode = next(m for m in MODES if getattr(args, m.replace("-", "_")))
    output = OUTPUT3 if mode in ITEM3 else OUTPUT1
    if mode == "scan":
        return scan(OUTPUT1, tuple(args.sources.split(",")))
    if mode == "assemble":
        return assemble(output, args.split)
    if mode in SHARDED:
        return SHARDED[mode](output, args.workers, args.worker)
    return MODES[mode](output)


if __name__ == "__main__":
    sys.exit(main())
