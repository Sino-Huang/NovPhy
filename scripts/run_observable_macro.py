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
--validate checks every published item.

Zero engine seconds: every number is read from retained native traces.
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
# entry point
# ---------------------------------------------------------------------------

def validate(_):
    problems = validate1(OUTPUT1)
    for problem in problems:
        log(f"VALIDATE: {problem}")
    log("validate: " + ("ok" if not problems else f"{len(problems)} problem(s)"))
    return 0 if not problems else 1


MODES = {"scan": None, "select": select, "smoke1": smoke1, "prepare1": prepare1, "run1": run1,
         "publish1": publish1, "validate": validate}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        group.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--sources", default=",".join(SELECTION_SOURCES))
    args = parser.parse_args()
    mode = next(m for m in MODES if getattr(args, m.replace("-", "_")))
    if mode == "scan":
        return scan(OUTPUT1, tuple(args.sources.split(",")))
    return MODES[mode](OUTPUT1)


if __name__ == "__main__":
    sys.exit(main())
