"""Issue #110 results: contract-v2 representation coverage on the #104 campaign captures.

Modes (one launch per stage; sharded modes take ``--workers 4 --worker k`` with one process per
GPU, ``CUDA_VISIBLE_DEVICES=k``):

* ``--repair``   fan labeler repair evidence (residual acceleration, smoke + fit fan captures)
* ``--smoke``    pre-freeze controls (label equality with #99/#112, model shapes, per-Delta
                 symbols, tiny train/score/probe passes, cost projection); nothing kept
* ``--prepare``  freeze ``plan.json``
* ``--labels``   v2 labels (and encoder images) for every complete branch (CPU pool)
* ``--encoder``  zero-shot E99-recipe encoder on normal fit images (one GPU)
* ``--fewshot-encoder`` per-novelty-level encoder adaptation (sharded)
* ``--carriers`` parse every shot with the encoders it needs (sharded)
* ``--scene``    zero-shot and few-shot scene heads (one GPU)
* ``--assemble`` training payloads (zero-shot normal; few-shot per novelty level)
* ``--train``    zero-shot #112 v3 LC members (sharded); ``--fewshot-train`` few-shot members
* ``--score``    rollout-error records on issue110-evaluation (sharded)
* ``--probe``    identifiability probes (sharded)
* ``--publish`` / ``--validate``

Worker entry points stay light: heavy runner modules are imported inside functions, and the
label pool runs ``world_model.data.novelty_labels_v2.label_worker``.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
import functools
import hashlib
import importlib.util
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import time
import types

import numpy as np
import torch

from world_model.data import novelty_contract as contract
from world_model.data import novelty_labels_v2 as labels

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".local-artifacts/issue-110-novelty-coverage-results-v1"
DESIGN = ROOT / ".local-artifacts/issue-110-novelty-coverage-v1"
CAMPAIGN = ROOT / ".local-artifacts/issue-104-power-cohort-campaign-v1"
SMOKE104 = ROOT / ".local-artifacts/issue-104-capture-smoke-v1"
COHORT109 = ROOT / ".local-artifacts/issue-109-sealed-cohort-v1"
RETRAIN = ROOT / ".local-artifacts/issue-112-v3-retrain-v1"
WINDOW113 = ROOT / ".local-artifacts/issue-113-capture-window-v1/spec.json"
RECOVERED = ROOT / ".local-artifacts/issue-76-canonical-recovery-v1/ExportedProject/Assets"

IDENTITY = "issue-110-novelty-coverage-results-v1"
SCHEMA_PLAN = "issue_110_results_plan_v1"
SCHEMA_SMOKE = "issue_110_results_smoke_v1"
SCHEMA_REPAIR = "issue_110_fan_labeler_repair_v1"
SCHEMA_ENCODER = "issue_110_v2_encoder_v1"
SCHEMA_SCENE = "issue_110_v2_scene_head_v1"
SCHEMA_CARRIERS = "issue_110_v2_carriers_v1"
SCHEMA_DATA = "issue_110_v2_training_data_v1"
SCHEMA_PREDICTOR = "issue_110_v2_predictor_v1"
SCHEMA_GROUP = "issue_110_v2_training_group_v1"
SCHEMA_UPDATES = "issue_110_v2_updates_v1"
SCHEMA_ROLLOUT = "issue_110_rollout_record_v1"
SCHEMA_PROBE = "issue_110_probe_record_v1"
SCHEMA_COMPUTE = "issue_110_results_compute_v1"
SCHEMA_HANDOFF = "issue_110_handoff_107_v1"
RUNNER = "scripts/run_novelty_coverage.py"
VALIDATION_COMMAND = "python -u -m scripts.run_novelty_coverage --validate"

STREAM_FIT, STREAM_EVAL, STREAM_109 = "issue110-fit", "issue110-evaluation", "issue109-fit"
ZERO = "zero-shot"
LEVELS = tuple(range(1, 9))
FAMILIES = ("hybrid", "continuous")
ARM = "LC"
SEEDS = (20260908, 20260909, 20260910)
TIMES = (15, 60, 225)
REQUESTS = tuple(f"F-{d}-{a}" for d in (1, 5, 15) for a in ("continuous", "micro", "macro"))
CONTINUOUS_REQUESTS = ("C-F-1", "C-F-5", "C-F-15")
ZERO_IMAGE_ORDINALS = (0, 5, 10, 15)
ENCODER = {"image_height": 240, "image_width": 320, "hidden_dim": 128, "epochs": 12, "batch_size": 64,
           "learning_rate": 1e-3, "weight_decay": 1e-4, "seed": 9900001, "attention_layers": 2,
           "attention_heads": 4, "grad_clip": 1.0}
FEW_SHOT = {"updates": 2000, "batch_size": 64, "learning_rate": 1e-4, "weight_decay": 1e-4, "grad_clip": 1.0,
            "seed": 11000001, "predictor_schedule": "the LC end state (long horizon 225, full ranking weight)"}
SCENE_RECIPE = {"updates": 2000, "batch_size": 64, "learning_rate": 1e-3, "weight_decay": 1e-4, "grad_clip": 1.0,
                "seed": 11000002, "history": 15, "hidden": 128}
PROBE = {"hidden": 64, "updates": 2000, "batch_size": 256, "learning_rate": 1e-3, "seed": 1100001,
         "ks": (1, 5, 15), "primary_k": 15, "threshold": 0.75, "lower": 0.65, "minimum_class_frames": 20,
         "bins": 1024, "bootstrap": "level-clustered, 10 000 draws on score-quantile bins; draws lacking a class "
                                    "are discarded"}
FORCE_ON_KINDS = {"Fan": ("bird", "block", "pig"), "turbulence": ("bird", "block")}  # bodies the law can act on
BOOTSTRAP = {"draws": 10000, "seed": 7201, "quantiles": (0.025, 0.975), "label": "DESCRIPTIVE"}
LABEL_PROCESSES = 44
PARSE_THREADS = 10
PARSE_BATCH = 64
STOP_TOKEN = "readiness_or_precision_insufficient"
GPU_CAP_SECONDS = 4 * 40 * 3600.0
WALL_CAP_SECONDS = 4 * 48 * 3600.0
DEVICE = "cuda"


def log(message):
    print(f"[issue-110-results] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path):
    with open(path) as stream:
        return json.load(stream)


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json_text(value))
    temporary.replace(path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def sha256_text(text):
    return f"sha256:{hashlib.sha256(text.encode()).hexdigest()}"


@functools.lru_cache(maxsize=None)
def nv():
    from world_model.training import novelty_v2
    return novelty_v2


def heavy():
    """The #112/#99 runner modules (imported only by GPU/publication modes)."""
    from scripts import run_decision_dynamics_retrain as rt
    from scripts import run_decision_dynamics_versions as versions
    from scripts import run_decision_relevant_dynamics as v1
    from scripts import run_slot_encoder_fix as s99
    from world_model.planning import scoring_harness as harness
    return types.SimpleNamespace(rt=rt, versions=versions, v1=v1, s99=s99, harness=harness)


# ---------------------------------------------------------------------------
# membership (outcome-free order: stream, pair, ordinal, side, candidate suffix)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def design_plan():
    return read_json(DESIGN / "plan.json")


@functools.lru_cache(maxsize=None)
def cohort109():
    return read_json(COHORT109 / "plan.json")


@functools.lru_cache(maxsize=None)
def campaign_rows():
    return {row["identity"]: row for row in read_json(CAMPAIGN / "branches.json")
            if row["stream"] in (STREAM_FIT, STREAM_EVAL, STREAM_109)}


@functools.lru_cache(maxsize=None)
def pair_cells():
    """pair -> (novelty level of its novel side, scenario); from the frozen #110 cells."""
    out = {}
    for cell in design_plan()["cells"]:
        for pair in cell["captured_by_pairs"]:
            if cell["novelty_level"] == 0:
                out.setdefault(pair, [None, cell["scenario"]])[1] = cell["scenario"]
            else:
                out.setdefault(pair, [cell["novelty_level"], cell["scenario"]])[0] = cell["novelty_level"]
    return {pair: tuple(value) for pair, value in out.items()}


def cells():
    """The 45 cells in frozen order: novelty level, then the design plan's order."""
    return [c["cell"] for c in sorted(design_plan()["cells"], key=lambda c: c["novelty_level"])]


def cell_of(pair, side):
    level, scenario = pair_cells()[pair]
    return f"{0 if side == 'normal' else level}/{scenario}"


def candidates_for(side):
    return design_plan()["candidates"]["by_slingshot_side"][side]


@functools.lru_cache(maxsize=None)
def branches():
    rows = campaign_rows()
    out = []
    for split, stream in (("fit", STREAM_FIT), ("evaluation", STREAM_EVAL)):
        roster = design_plan()["rosters"][split]
        for pair in sorted(roster):
            for entry in roster[pair]:
                for side in ("normal", "novel"):
                    level = entry[side]
                    for candidate in candidates_for(level["slingshot_side"]):
                        key = f"{level['identity']}-{candidate['suffix']}"
                        if key not in rows:
                            continue  # ordinal not captured (the campaign captured ordinals 1..8)
                        row = rows[key]
                        out.append({"key": key, "stream": stream, "split": split, "level": level["identity"],
                                    "pair": pair, "ordinal": entry["ordinal"], "side": side,
                                    "novelty_level": level["novelty_level"], "pair_level": pair_cells()[pair][0],
                                    "cell": cell_of(pair, side), "template": level["template"],
                                    "slingshot_side": level["slingshot_side"], "suffix": candidate["suffix"],
                                    "candidate_ordinal": candidate["ordinal"], "action": candidate["action"],
                                    "inventory_entries": candidate["inventory_entries"],
                                    "status": row["status"], "failure_class": row.get("failure_class"),
                                    "pig_removed": row.get("pig_removed"), "stop_kind": row.get("stop_kind"),
                                    "attempt": str(CAMPAIGN / stream / "attempts" / key),
                                    "xml": str(DESIGN / level["xml_path"])})
    cohort = cohort109()
    training = cohort["candidate_inventories"]["sets"]["training"]["candidates"]
    for family in ("type010103", "type010105"):
        for level in cohort["splits"]["fit"][family]:
            for candidate in training:
                key = f"{level['identity']}-{candidate['suffix']}"
                row = rows[key]
                pair = f"pair-{family}"
                out.append({"key": key, "stream": STREAM_109, "split": "fit", "level": level["identity"],
                            "pair": pair, "ordinal": level["ordinal"], "side": "normal", "novelty_level": 0,
                            "pair_level": pair_cells()[pair][0], "cell": cell_of(pair, "normal"),
                            "template": f"issue-109/{family}", "slingshot_side": "left",
                            "suffix": candidate["suffix"], "candidate_ordinal": candidate["ordinal"],
                            "action": candidate["action"], "inventory_entries": candidate["inventory_entries"],
                            "status": row["status"], "failure_class": row.get("failure_class"),
                            "pig_removed": row.get("pig_removed"), "stop_kind": row.get("stop_kind"),
                            "attempt": str(CAMPAIGN / STREAM_109 / "attempts" / key),
                            "xml": str(COHORT109 / level["xml_path"])})
    if len(out) != len(rows):
        raise ValueError(f"membership {len(out)} differs from the campaign rows {len(rows)}")
    return tuple(out)


def complete():
    return [b for b in branches() if b["status"] == "complete"]


def grid_ordinal(branch):
    return next((e["ordinal"] for e in branch["inventory_entries"] if e["inventory"] == "grid"), None)


def zero_image_keys():
    """Encoder images (zero-shot): normal fit shots at grid ordinals 0, 5, 10, 15."""
    return [b["key"] for b in complete() if b["split"] == "fit" and b["side"] == "normal"
            and grid_ordinal(b) in ZERO_IMAGE_ORDINALS]


def few_image_keys(level):
    """Encoder images (few-shot, level k): every complete novel fit shot of level k."""
    return [b["key"] for b in complete() if b["stream"] == STREAM_FIT and b["side"] == "novel"
            and b["novelty_level"] == level]


def zero_training(branch):
    return branch["split"] == "fit" and branch["side"] == "normal"


def few_training(branch, level):
    return branch["stream"] == STREAM_FIT and branch["side"] == "novel" and branch["novelty_level"] == level


def conditions_of(branch):
    """Carrier conditions a shot needs: zero-shot always; few-shot k for level-k novel fit shots and
    every evaluation shot of a level-k pair (both sides, for the novel - normal contrast)."""
    out = [ZERO]
    if branch["stream"] == STREAM_FIT and branch["side"] == "novel":
        out.append(f"few-shot-{branch['novelty_level']}")
    if branch["stream"] == STREAM_EVAL:
        out.append(f"few-shot-{branch['pair_level']}")
    return out


def condition_levels():
    return (ZERO, *(f"few-shot-{k}" for k in LEVELS))


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def label_path(output, key):
    return Path(output) / "labels" / f"{key}.pt"


def encoder_path(output, condition):
    return Path(output) / "encoders" / f"{condition}.pt"


def scene_path(output, condition):
    return Path(output) / "scene" / f"{condition}.pt"


def carrier_path(output, condition, key):
    return Path(output) / "carriers" / condition / f"{key}.pt"


def data_path(output, condition):
    return Path(output) / "data" / f"{condition}.pt"


def updates_path(output, condition, family, seed):
    return Path(output) / "updates" / f"{condition}--{family}--seed{seed}.pt"


def rollout_path(output, condition, level):
    return Path(output) / "records" / "rollout" / condition / f"{level}.json"


def probe_path(output, task):
    return Path(output) / "records" / "probe" / f"{task}.json"


def ledger_path(output, phase):
    return Path(output) / "ledger" / f"{phase}.json"


def mark(output, phase, wall, gpu, **extra):
    path = ledger_path(output, phase)
    block = read_json(path) if path.is_file() else {"wall_seconds": 0.0, "gpu_seconds": 0.0}
    write_json(path, {**block, "wall_seconds": block["wall_seconds"] + wall, "gpu_seconds": block["gpu_seconds"] + gpu,
                      "last_at": utc_now(), **extra})
    total = [read_json(p) for p in sorted((Path(output) / "ledger").glob("*.json"))]
    if (sum(b["gpu_seconds"] for b in total) > GPU_CAP_SECONDS
            or sum(b["wall_seconds"] for b in total) > WALL_CAP_SECONDS):
        raise ValueError(f"{STOP_TOKEN}: compute cap exceeded")


def load_plan(output):
    """The frozen plan, refusing on drift of any pinned input (mirrors Gate B; the runner itself is
    excluded as in #102: its post-freeze edits are reporting-only and disclosed), of smoke.json or of
    the labeler-repair record (the G2 input)."""
    output = Path(output)
    path = output / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --repair, --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan binding differs")
    changed = [e["name"] for e in plan["inputs"] if e["name"] != "runner"
               and sha256_of(ROOT / e["artifact"]) != e["sha256"]]
    if sha256_of(output / "smoke.json") != plan["smoke_sha256"]:
        changed.append("smoke.json")
    if sha256_of(output / "repair" / "fan-repair.json") != plan["repair_sha256"]:
        changed.append("repair/fan-repair.json")
    if changed:
        raise ValueError(f"frozen inputs changed after the freeze: {changed}")
    return plan


def read_label(output, key):
    value = torch.load(label_path(output, key), map_location="cpu", weights_only=True, mmap=True)
    if value.get("schema") != labels.SCHEMA_LABELS:
        raise ValueError(f"label {key} schema differs")
    return value


def label_failures(output):
    path = Path(output) / "labels" / "failures.json"
    return read_json(path)["failures"] if path.is_file() else {}


def usable_branches(output):
    failed = label_failures(output)
    return [b for b in complete() if b["key"] not in failed]


# ---------------------------------------------------------------------------
# fan labeler repair (pre-freeze; outcome-free)
# ---------------------------------------------------------------------------

def residual_job(job):
    attempt, xml = job
    torch.set_num_threads(1)
    result = labels.residual_check(attempt, Path(xml).read_text())
    return {name: {"steps": len(b["projections"]),
                   "median_projection": statistics.median(b["projections"]) if b["projections"] else None,
                   **{k: b[k] for k in ("tp", "fp", "fn", "tn")}} for name, b in result.items()}


def fan_sources():
    """The attached scripts of the Fan prefab (recovered player), the repair's source evidence."""
    prefab = (RECOVERED / "Resources/prefabs/gameworld/benchmarknovelties/Fan.prefab").read_text()
    guids = {}
    for meta in (RECOVERED / "Scripts/Assembly-CSharp").glob("*.cs.meta"):
        text = meta.read_text()
        guids[text.split("guid: ")[1].split()[0]] = meta.name.removesuffix(".cs.meta")
    scripts = []
    for block in prefab.split("--- !u!114 ")[1:]:
        enabled = "m_Enabled: 1" in block.split("m_Script")[0]
        guid = block.split("guid: ")[1].split(",")[0]
        fields = {line.split(":")[0].strip(): line.split(":")[1].strip() for line in block.splitlines()
                  if line.strip().split(":")[0] in ("fanForce", "windWidth", "windHeight")}
        scripts.append({"script": guids.get(guid, guid), "enabled": enabled, "serialized": fields})
    fan_cs = (RECOVERED / "Scripts/Assembly-CSharp/Fan.cs").read_text()
    literals = ["item.AddForce(base.transform.right * fanForce);",
                "boxCollider2D.size = new Vector2(windWidth * 2f, windHeight);",
                'if (collision.gameObject.tag != "Platform")']
    return {"prefab": "Resources/prefabs/gameworld/benchmarknovelties/Fan.prefab",
            "prefab_sha256": sha256_text(prefab), "attached_scripts": scripts,
            "fan_cs_sha256": sha256_text(fan_cs), "fan_cs_literals": {l: l in fan_cs for l in literals}}


def repair(output):
    """Residual acceleration of contact-free bodies vs the legacy and repaired fan labels on every
    fan capture of the #104 smoke and of the issue110-fit stream (novel sides of pair-type0102xx)."""
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the labeler repair is a pre-freeze step")
    began = time.monotonic()
    smoke = sorted((SMOKE104 / "attempts").glob("issue-110-smoke-type0102*-novel-*"))
    jobs = [(str(a), str(DESIGN / "levels" / (a.name.rsplit("-", 1)[0] + ".xml"))) for a in smoke]
    fit = [b for b in complete() if b["stream"] == STREAM_FIT and b["side"] == "novel" and b["novelty_level"] == 2]
    jobs += [(b["attempt"], b["xml"]) for b in fit]
    names = [("smoke", a.name) for a in smoke] + [("fit", b["key"]) for b in fit]
    with ProcessPoolExecutor(LABEL_PROCESSES, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(residual_job, jobs, chunksize=2))
    rows = [{"source": s, "branch": n, **{f"{lab}_{k}": v for lab, r in result.items() for k, v in r.items()}}
            for (s, n), result in zip(names, results)]
    summary = {}
    for source in ("smoke", "fit"):
        for template in sorted({r["branch"].split("-")[3] for r in rows if r["source"] == source}):
            subset = [r for r in rows if r["source"] == source and r["branch"].split("-")[3] == template]
            block = {}
            for lab in ("legacy", "repaired"):
                exercised = [r for r in subset if r[f"{lab}_steps"]]
                medians = [r[f"{lab}_median_projection"] for r in exercised]
                tp, fp, fn = (sum(r[f"{lab}_{k}"] for r in subset) for k in ("tp", "fp", "fn"))
                block[lab] = {"branches_exercised": len(exercised), "branches": len(subset),
                              "median_of_branch_medians": statistics.median(medians) if medians else None,
                              "verdict": ("not_exercised" if not medians else
                                          "agree" if statistics.median(medians) > 0 else "disagree"),
                              "labelled_steps": sum(r[f"{lab}_steps"] for r in subset),
                              "precision": tp / (tp + fp) if tp + fp else None,
                              "recall": tp / (tp + fn) if tp + fn else None}
            summary[f"{source}/{template}"] = block
    wall = time.monotonic() - began
    value = {"schema": SCHEMA_REPAIR, "plan_identity": IDENTITY, "created_at": utc_now(),
             "rule": "#110 smoke gate: a labeler disagreement is repaired in the derivation before any training",
             "law": labels.FAN_LAW_REPAIRED, "sources": fan_sources(), "summary": summary, "rows": rows,
             "anomalous_definition": "|residual acceleration| > 10 m/s^2 on a contact-free dynamic body",
             "outcome_free": True, "wall_seconds": wall, "engine_seconds": 0}
    write_json(output / "repair" / "fan-repair.json", value)
    mark(output, "repair", wall, 0.0, branches=len(jobs))
    for key, block in summary.items():
        log(f"{key}: legacy {block['legacy']['verdict']} / repaired {block['repaired']['verdict']} "
            f"(precision {block['repaired']['precision']}, recall {block['repaired']['recall']})")
    return 0


# ---------------------------------------------------------------------------
# labels (CPU pool)
# ---------------------------------------------------------------------------

def image_keys():
    keys = set(zero_image_keys())
    for level in LEVELS:
        keys |= set(few_image_keys(level))
    return keys


def run_labels(output):
    output = Path(output)
    load_plan(output)
    began = time.monotonic()
    images = image_keys()
    pending = [b for b in complete() if not label_path(output, b["key"]).is_file()]
    failures = label_failures(output)
    pending = [b for b in pending if b["key"] not in failures]
    log(f"labels: {len(pending)} of {len(complete())} complete branches to label ({len(images)} with images)")
    jobs = [(b["attempt"], b["xml"], str(label_path(output, b["key"])), b["key"] in images) for b in pending]
    done = 0
    with ProcessPoolExecutor(LABEL_PROCESSES, mp_context=multiprocessing.get_context("spawn")) as pool:
        for (target, error), branch in zip(pool.map(labels.label_worker, jobs, chunksize=4), pending):
            done += 1
            if error is not None:
                failures[branch["key"]] = error
                log(f"typed label failure {branch['key']}: {error}")
            if done % 1000 == 0:
                elapsed = time.monotonic() - began
                log(f"labels {done}/{len(pending)}: {elapsed:.0f}s, eta {elapsed / done * (len(pending) - done):.0f}s")
    write_json(output / "labels" / "failures.json", {"failures": failures})
    wall = time.monotonic() - began
    mark(output, "labels", wall, 0.0, shots=len(pending), failures=len(failures))
    return 0


# ---------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------

def encoder_config():
    from world_model.training.cohort_v2_visual_parser import CohortV2VisualParserConfig
    return CohortV2VisualParserConfig(seed=ENCODER["seed"], image_height=ENCODER["image_height"],
                                      image_width=ENCODER["image_width"], hidden_dim=ENCODER["hidden_dim"],
                                      epochs=ENCODER["epochs"], batch_size=ENCODER["batch_size"],
                                      learning_rate=ENCODER["learning_rate"], weight_decay=ENCODER["weight_decay"],
                                      device=DEVICE)


def new_encoder():
    return nv().SpatialSlotParserV2(encoder_config(), layers=ENCODER["attention_layers"],
                                    heads=ENCODER["attention_heads"])


def image_rows(shot):
    """Encoder targets at a shot's image positions."""
    p = shot["image_positions"]
    presence = shot["presence"][p]
    return {"images": shot["images"], "presence": presence.float(), "centers": shot["centers"][p],
            "kinds": shot["kinds"][p].long(), "relations": labels.unpack(shot["relations"][p]),
            "relation_mask": labels.relation_mask(presence, shot["relation_available"][p]),
            "macros": shot["macros"][p].float(), "macro_mask": shot["macro_mask"][p]}


def shot_stream(output, keys, generator, threads=8):
    """Shot-shuffled image rows, loaded ahead by a thread pool."""
    order = [keys[i] for i in torch.randperm(len(keys), generator=generator).tolist()]
    with ThreadPoolExecutor(threads) as pool:
        futures = [pool.submit(lambda k: image_rows(read_label(output, k)), k) for k in order[:threads * 2]]
        for index in range(len(order)):
            row = futures[index].result()
            futures[index] = None
            if index + threads * 2 < len(order):
                futures.append(pool.submit(lambda k: image_rows(read_label(output, k)), order[index + threads * 2]))
            yield row


def encoder_batches(output, keys, batch_size, generator):
    """The s99 ``encoder_batches`` pattern: 8-batch shuffle buffers of shot-shuffled frames."""
    pending, count = [], 0
    for row in shot_stream(output, keys, generator):
        pending.append(row)
        count += len(row["images"])
        if count >= batch_size * 8:
            merged = {k: torch.cat([t[k] for t in pending]) for k in pending[0]}
            order = torch.randperm(count, generator=generator)
            for start in range(0, count - batch_size + 1, batch_size):
                yield {k: v[order[start:start + batch_size]] for k, v in merged.items()}
            pending, count = [], 0
    if pending:
        yield {k: torch.cat([t[k] for t in pending]) for k in pending[0]}


def presence_weights(output, keys):
    total, positive = 0, torch.zeros(labels.SLOTS)
    for key in keys:
        shot = read_label(output, key)
        presence = shot["presence"][shot["image_positions"]].float()
        total += len(presence)
        positive += presence.sum(0)
    return total, positive, nv().class_balance_weights(total, positive)


def train_encoder(output, keys, target, steps_limit=None):
    """E99 recipe (12 epochs, batch 64, AdamW lr 1e-3, wd 1e-4, clip 1.0, last epoch kept) on v2 targets."""
    total, positive, weights = presence_weights(output, keys)
    torch.manual_seed(ENCODER["seed"])
    model = new_encoder().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ENCODER["learning_rate"], weight_decay=ENCODER["weight_decay"])
    progress = Path(target).with_name(Path(target).stem + "-progress.pt")
    first, reports, gpu = 0, [], 0.0
    if progress.is_file() and not steps_limit:
        saved = torch.load(progress, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        first, reports, gpu = saved["epoch"], saved["reports"], saved["gpu_seconds"]
    epochs = 1 if steps_limit else ENCODER["epochs"]
    steps = 0
    weights = weights.to(DEVICE)
    for epoch in range(first, epochs):
        model.train()
        generator = torch.Generator().manual_seed(ENCODER["seed"] + epoch)
        began, seen, loss_sum = time.monotonic(), 0, 0.0
        for batch in encoder_batches(output, keys, ENCODER["batch_size"], generator):
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = nv().parser_loss(model, batch, weights)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite encoder loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ENCODER["grad_clip"])
            optimizer.step()
            seen += len(batch["images"])
            loss_sum += float(loss.detach()) * len(batch["images"])
            steps += 1
            if steps_limit and steps >= steps_limit:
                break
        torch.cuda.synchronize()
        wall = time.monotonic() - began
        gpu += wall
        reports.append({"epoch": epoch + 1, "frames": seen, "mean_loss": loss_sum / max(seen, 1), "wall_seconds": wall})
        log(f"encoder epoch {epoch + 1}/{epochs}: frames {seen}, loss {loss_sum / max(seen, 1):.5f}, {wall:.0f}s")
        if not steps_limit:
            atomic_torch(progress, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                    "epoch": epoch + 1, "reports": reports, "gpu_seconds": gpu})
    payload = {"schema": SCHEMA_ENCODER, "plan_identity": IDENTITY, "condition": ZERO,
               "architecture": model.architecture_identity, "recipe": ENCODER,
               "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "reports": reports,
               "gpu_seconds": gpu, "training_frames": total, "training_shots": len(keys),
               "presence_positive": positive.tolist(), "presence_weights": weights.cpu().tolist(),
               "parameters": sum(p.numel() for p in model.parameters()), "complete": not steps_limit}
    if not steps_limit:
        atomic_torch(target, payload)
        progress.unlink(missing_ok=True)
    return payload


def load_encoder(output, condition):
    payload = torch.load(encoder_path(output, condition), map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA_ENCODER or payload.get("plan_identity") != IDENTITY \
            or payload.get("condition") != condition or not payload.get("complete"):
        raise ValueError(f"encoder {condition} binding differs")
    model = new_encoder()
    model.load_state_dict(payload["model"], strict=True)
    return model.to(DEVICE).eval(), payload


def encoder(output):
    output = Path(output)
    load_plan(output)
    if encoder_path(output, ZERO).is_file():
        return 0
    began = time.monotonic()
    keys = [k for k in zero_image_keys() if k not in label_failures(output)]
    payload = train_encoder(output, keys, encoder_path(output, ZERO))
    mark(output, "encoder-zero-shot", time.monotonic() - began, payload["gpu_seconds"], frames=payload["training_frames"])
    return 0


def adapt_encoder(output, level, steps=None):
    """Few-shot encoder: the zero-shot encoder adapted for 2000 updates (batch 64, AdamW lr 1e-4,
    wd 1e-4, clip 1.0) on the level's novel fit frames; the zero-shot presence weights."""
    base, zero = load_encoder(output, ZERO)
    model = base.train()
    weights = torch.tensor(zero["presence_weights"], device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FEW_SHOT["learning_rate"], weight_decay=FEW_SHOT["weight_decay"])
    keys = [k for k in few_image_keys(level) if k not in label_failures(output)]
    total = steps or FEW_SHOT["updates"]
    began, done, epoch, losses = time.monotonic(), 0, 0, []
    while done < total:
        generator = torch.Generator().manual_seed(FEW_SHOT["seed"] + 1000 * level + epoch)
        for batch in encoder_batches(output, keys, FEW_SHOT["batch_size"], generator):
            if len(batch["images"]) < FEW_SHOT["batch_size"]:
                continue
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = nv().parser_loss(model, batch, weights)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite few-shot encoder loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), FEW_SHOT["grad_clip"])
            optimizer.step()
            losses.append(float(loss.detach()))
            done += 1
            if done >= total:
                break
        epoch += 1
    torch.cuda.synchronize()
    gpu = time.monotonic() - began
    return {"schema": SCHEMA_ENCODER, "plan_identity": IDENTITY, "condition": f"few-shot-{level}",
            "architecture": model.architecture_identity, "recipe": FEW_SHOT, "initialization": ZERO,
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "gpu_seconds": gpu,
            "training_shots": len(keys), "updates": done, "epochs_started": epoch,
            "loss_first_100": float(np.mean(losses[:100])), "loss_last_100": float(np.mean(losses[-100:])),
            "presence_weights": zero["presence_weights"], "complete": steps is None}


def fewshot_encoder(output, workers, worker):
    output = Path(output)
    load_plan(output)
    for level in LEVELS[worker::workers]:
        target = encoder_path(output, f"few-shot-{level}")
        if target.is_file():
            continue
        began = time.monotonic()
        payload = adapt_encoder(output, level)
        atomic_torch(target, payload)
        mark(output, f"encoder-few-shot-{level}", time.monotonic() - began, payload["gpu_seconds"])
        log(f"few-shot encoder level {level}: loss {payload['loss_first_100']:.4f} -> {payload['loss_last_100']:.4f}")
    return 0


# ---------------------------------------------------------------------------
# carriers
# ---------------------------------------------------------------------------

def shot_images(attempt, end):
    observation_root, frames, _ = labels.shot_frames(attempt)
    paths = [observation_root / frames[p]["agent_observation"]["relative_path"] for p in range(end + 1)]
    with ThreadPoolExecutor(PARSE_THREADS) as pool:
        return torch.stack(list(pool.map(labels.image_tensor, paths)))


def base_carriers(model, images, times):
    """Parse a shot's frames (batches of 64) and build its v2 carriers (scene columns zero)."""
    outputs = []
    for start in range(0, len(images), PARSE_BATCH):
        outputs.append(nv().parsed(model, images[start:start + PARSE_BATCH].to(DEVICE)))
    parsed = {k: torch.cat([o[k] for o in outputs]) for k in outputs[0]}
    return nv().carrier_values(parsed["presence"], parsed["centers"], parsed["kinds"], times).cpu(), parsed


def carriers(output, workers, worker):
    output = Path(output)
    load_plan(output)
    began = time.monotonic()
    shard = usable_branches(output)[worker::workers]
    needed = [(b, [c for c in conditions_of(b) if not carrier_path(output, c, b["key"]).is_file()]) for b in shard]
    needed = [(b, c) for b, c in needed if c]
    models = {}
    log(f"worker {worker}: {len(needed)} of {len(shard)} shots to parse")
    with torch.no_grad():
        for done, (branch, conditions) in enumerate(needed, 1):
            shot = read_label(output, branch["key"])
            images = shot_images(branch["attempt"], shot["end"])
            for condition in conditions:
                if condition not in models:
                    models[condition] = load_encoder(output, condition)[0]
                values, parsed = base_carriers(models[condition], images, shot["times"])
                atomic_torch(carrier_path(output, condition, branch["key"]),
                             {"schema": SCHEMA_CARRIERS, "plan_identity": IDENTITY, "condition": condition,
                              "key": branch["key"], "carriers": values,
                              "decision_parse": {k: v[0].cpu() for k, v in parsed.items()}})
            if done % 500 == 0:
                elapsed = time.monotonic() - began
                log(f"worker {worker}: {done}/{len(needed)} shots, {elapsed:.0f}s, eta {elapsed / done * (len(needed) - done):.0f}s")
    wall = time.monotonic() - began
    mark(output, f"carriers-worker{worker}", wall, wall, shots=len(needed))
    return 0


def read_carriers(output, condition, key):
    value = torch.load(carrier_path(output, condition, key), map_location="cpu", weights_only=True)
    if value.get("schema") != SCHEMA_CARRIERS or value.get("condition") != condition or value.get("key") != key:
        raise ValueError(f"carriers {condition}/{key} binding differs")
    return value["carriers"]


# ---------------------------------------------------------------------------
# scene heads
# ---------------------------------------------------------------------------

def scene_training_keys(condition):
    if condition == ZERO:
        return [b["key"] for b in complete() if zero_training(b)]
    level = int(condition.rsplit("-", 1)[1])
    return [b["key"] for b in complete() if few_training(b, level)]


def train_scene(output, condition, steps=None):
    keys = [k for k in scene_training_keys(condition) if k not in label_failures(output)]
    carriers_all, targets, starts, offset = [], [], [], 0
    for key in keys:
        values = read_carriers(output, condition, key)
        carriers_all.append(values)
        targets.append(read_label(output, key)["scene"])
        starts.append(torch.full((len(values),), offset, dtype=torch.long))
        offset += len(values)
    z = torch.cat(carriers_all).to(DEVICE)
    y = torch.cat(targets).to(DEVICE)
    floor = torch.cat(starts).to(DEVICE)
    torch.manual_seed(SCENE_RECIPE["seed"])
    head = nv().SceneHead(SCENE_RECIPE["history"], SCENE_RECIPE["hidden"]).to(DEVICE)
    recipe = dict(SCENE_RECIPE)
    if condition != ZERO:
        state = torch.load(scene_path(output, ZERO), map_location="cpu", weights_only=True)
        head.load_state_dict(state["model"])
        recipe.update(learning_rate=FEW_SHOT["learning_rate"], seed=FEW_SHOT["seed"])
    optimizer = torch.optim.AdamW(head.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    generator = torch.Generator(device=DEVICE).manual_seed(recipe["seed"])
    total = steps or recipe["updates"]
    began, losses = time.monotonic(), []
    for _ in range(total):
        index = torch.randint(len(z), (recipe["batch_size"],), device=DEVICE, generator=generator)
        loss = nv().scene_loss(head, nv().scene_inputs(z, index, floor[index], head.k), y[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), recipe["grad_clip"])
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    return {"schema": SCHEMA_SCENE, "plan_identity": IDENTITY, "condition": condition, "recipe": recipe,
            "model": {k: v.detach().cpu() for k, v in head.state_dict().items()}, "shots": len(keys),
            "positions": len(z), "gpu_seconds": time.monotonic() - began,
            "loss_first_100": float(np.mean(losses[:100])), "loss_last_100": float(np.mean(losses[-100:])),
            "complete": steps is None}


def scene(output):
    output = Path(output)
    load_plan(output)
    for condition in condition_levels():
        if scene_path(output, condition).is_file():
            continue
        began = time.monotonic()
        payload = train_scene(output, condition)
        atomic_torch(scene_path(output, condition), payload)
        mark(output, f"scene-{condition}", time.monotonic() - began, payload["gpu_seconds"])
        log(f"scene head {condition}: loss {payload['loss_first_100']:.4f} -> {payload['loss_last_100']:.4f}")
    return 0


@functools.lru_cache(maxsize=None)
def load_scene(output, condition):
    payload = torch.load(scene_path(output, condition), map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA_SCENE or payload.get("condition") != condition or not payload["complete"]:
        raise ValueError(f"scene head {condition} binding differs")
    head = nv().SceneHead(SCENE_RECIPE["history"], SCENE_RECIPE["hidden"])
    head.load_state_dict(payload["model"])
    return head.to(DEVICE).eval()


def full_carriers(output, condition, key):
    """The v2 carriers of a shot under a condition: its encoder's parse plus its scene head."""
    values = read_carriers(output, condition, key).to(DEVICE)
    return load_scene(str(output), condition).fill(values)


# ---------------------------------------------------------------------------
# training data (the #112 retrain payload layout at v2, per-Delta symbols gathered at batch time)
# ---------------------------------------------------------------------------

def ranking_cells(level_branches, anchors, shots):
    """Every training level x inventory whose complete candidates hold both a pig removal and a
    failure; candidates in inventory-ordinal order (the #112 retrain rule)."""
    pad = 20
    cells_out = []
    for level, items in level_branches.items():
        for inventory in ("grid", "angle"):
            entries = sorted((e["ordinal"], b) for b in items for e in b["inventory_entries"] if e["inventory"] == inventory)
            values = [bool(b["pig_removed"]) for _, b in entries]
            if set(values) != {True, False}:
                continue
            action = torch.zeros(pad, 5)
            verdict = torch.full((pad,), -1, dtype=torch.long)
            action[:len(entries)] = torch.stack([shots[b["key"]]["action"] for _, b in entries])
            verdict[:len(entries)] = torch.tensor([int(v) for v in values])
            cells_out.append({"member": level, "inventory": inventory, "ordinals": [o for o, _ in entries],
                              "z0": anchors[level], "action": action, "verdict": verdict})
    return cells_out


def assemble_condition(output, condition):
    from world_model.training.cnn_hybrid import PAIRS  # noqa: F401  (import check)
    h = heavy()
    if condition == ZERO:
        chosen = [b for b in usable_branches(output) if zero_training(b)]
    else:
        level = int(condition.rsplit("-", 1)[1])
        chosen = [b for b in usable_branches(output) if few_training(b, level)]
    shots, anchors, by_level = {}, {}, defaultdict(list)
    window_positions, window_shot, window_length, long_windows, offsets, lasts = [], [], [], [], [], []
    carriers_all, presence, rel, mask, macros, macro_mask, actions = [], [], [], [], [], [], []
    steps = torch.arange(h.s99.WINDOW_LENGTH + 1)
    total = 0
    for index, branch in enumerate(chosen):
        shot = read_label(output, branch["key"])
        values = full_carriers(output, condition, branch["key"]).cpu()
        end = int(shot["end"])
        if len(values) != end + 1:
            raise ValueError(f"carriers of {branch['key']} do not cover positions 0..end")
        from scripts import run_lookahead_control as wlc
        expected = wlc.action_tensor(branch["action"], "cpu")[0]
        if not torch.equal(shot["action"], expected):
            raise ValueError(f"executed action of {branch['key']} differs from its inventory action")
        shots[branch["key"]] = {"action": shot["action"]}
        anchors.setdefault(branch["level"], values[0].clone())
        by_level[branch["level"]].append(branch)
        for start in h.s99.WINDOW_STARTS:
            if start >= end:
                continue
            length = min(h.s99.WINDOW_LENGTH, end - start)
            window_positions.append(total + start + torch.minimum(steps, torch.tensor(length)))
            window_shot.append(index)
            window_length.append(length)
        long_windows.extend((index, start) for start in h.v1.LONG["starts"] if start < end)
        offsets.append(total)
        lasts.append(end)
        total += end + 1
        carriers_all.append(values)
        presence.append(shot["presence"])
        rel.append(shot["relations"])
        mask.append(labels.pack(labels.relation_mask(shot["presence"], shot["relation_available"])))
        macros.append(shot["macros"])
        macro_mask.append(shot["macro_mask"])
        actions.append(shot["action"])
    cells_list = ranking_cells(by_level, anchors, shots)

    def stacked(key, empty):
        return torch.stack([c[key] for c in cells_list]) if cells_list else empty
    return {"schema": SCHEMA_DATA, "plan_identity": IDENTITY, "condition": condition,
            "carriers": torch.cat(carriers_all), "presence": torch.cat(presence), "relations": torch.cat(rel),
            "relation_mask": torch.cat(mask), "macros": torch.cat(macros), "macro_mask": torch.cat(macro_mask),
            "window_positions": torch.stack(window_positions), "window_shot": torch.tensor(window_shot),
            "window_length": torch.tensor(window_length), "offsets": torch.tensor(offsets),
            "lasts": torch.tensor(lasts), "shot_actions": torch.stack(actions),
            "long_windows": torch.tensor(long_windows), "shot_keys": [b["key"] for b in chosen],
            "shot_members": [b["level"] for b in chosen],
            "cell_members": [c["member"] for c in cells_list], "cell_inventories": [c["inventory"] for c in cells_list],
            "cell_ordinals": [c["ordinals"] for c in cells_list],
            "cell_z0": stacked("z0", torch.empty(0, nv().DIM)),
            "cell_action": stacked("action", torch.empty(0, 20, 5)),
            "cell_verdict": stacked("verdict", torch.empty(0, 20, dtype=torch.long))}


def assemble(output):
    output = Path(output)
    load_plan(output)
    h = heavy()
    for condition in condition_levels():
        if data_path(output, condition).is_file():
            continue
        began = time.monotonic()
        with h.harness.deterministic_scoring():
            payload = assemble_condition(output, condition)
        atomic_torch(data_path(output, condition), payload)
        mark(output, f"assemble-{condition}", time.monotonic() - began, 0.0, shots=len(payload["shot_keys"]),
             positions=len(payload["carriers"]), windows=len(payload["window_positions"]),
             cells=len(payload["cell_members"]))
        log(f"assembled {condition}: {len(payload['shot_keys'])} shots, {len(payload['carriers'])} positions, "
            f"{len(payload['cell_members'])} ranking cells")
    return 0


def load_data(output, condition):
    payload = torch.load(data_path(output, condition), map_location="cpu", weights_only=True, mmap=True)
    if payload.get("schema") != SCHEMA_DATA or payload.get("condition") != condition:
        raise ValueError(f"training data {condition} binding differs")
    return payload


def train_data_class():
    v1 = heavy().v1

    class TrainDataV2(v1.TrainData):
        """v1.TrainData over an indexed v2 payload. Windows gather per-position carriers; the
        hybrid's symbols are gathered for the update's Delta at batch time under the #110 per-Delta
        rule: the label at window position j is the OR over frames (j - Delta, j] (clamped to the
        shot's first frame), and so is its mask."""

        def __init__(self, payload, family):
            self.hybrid = family == "hybrid"
            self.carriers = payload["carriers"].to(DEVICE)
            self.window_positions = payload["window_positions"].to(DEVICE)
            self.window_length = payload["window_length"].to(DEVICE)
            self.shot_members = payload["shot_members"]
            self.window_members = [self.shot_members[i] for i in payload["window_shot"].tolist()]
            self.offsets = payload["offsets"].to(DEVICE)
            self.window_floor = self.offsets[payload["window_shot"].to(DEVICE)]
            self.lasts = payload["lasts"].to(DEVICE)
            self.shot_actions = payload["shot_actions"].to(DEVICE)
            self.window_action = self.shot_actions[payload["window_shot"].to(DEVICE)]
            self.long = payload["long_windows"].to(DEVICE)
            self.long_members = [self.shot_members[i] for i in payload["long_windows"][:, 0].tolist()]
            self.cell_members = payload["cell_members"]
            self.cell_z0 = payload["cell_z0"].to(DEVICE)
            self.cell_action = payload["cell_action"].to(DEVICE)
            self.cell_verdict = payload["cell_verdict"].to(DEVICE)
            if self.hybrid:
                self.packed = {k: payload[k].to(DEVICE) for k in ("relations", "relation_mask", "macros", "macro_mask")}

        def windows(self, pools, generators, count):
            index = self._draw(pools, "window", generators, count)
            positions = self.window_positions[index]
            return {"z": self.carriers[positions], "action": self.window_action[index],
                    "length": self.window_length[index], "positions": positions, "floor": self.window_floor[index]}

        def symbols(self, batch, delta):
            positions, floor = batch["positions"], batch["floor"]
            out = {}
            for name, key in (("relations", "relations"), ("relations_mask", "relation_mask"),
                              ("macros", "macros"), ("macros_mask", "macro_mask")):
                source = self.packed[key]
                value = None
                for shift in range(delta):
                    index = torch.maximum(positions - shift, floor[..., None])
                    item = source[index]
                    value = item if value is None else value | item
                out[name] = labels.unpack(value) if name.startswith("relations") else value
            return out

        def rank_batch(self, pools, generators, count):
            if not any(len(pool["cell"]) for pool in pools):  # no mixed-verdict cell: the ranking term is 0
                members = len(pools)
                return {"z0": torch.zeros(members, count, self.carriers.shape[1], device=DEVICE),
                        "action": torch.zeros(members, count, 20, 5, device=DEVICE),
                        "verdict": torch.full((members, count, 20), -1, dtype=torch.long, device=DEVICE)}
            return super().rank_batch(pools, generators, count)
    return TrainDataV2


# ---------------------------------------------------------------------------
# predictors (the frozen #112 v3 LC recipe, retrain implementation, at v2)
# ---------------------------------------------------------------------------

def new_predictor(family):
    if family == "continuous":
        return nv().ContinuousDynamicsV2(nv().continuous_width()["width"])
    return nv().HybridPredictorV2()


def implementation(data, init=None):
    """A private copy of the #112 v1 runner carrying the frozen v3 spec (the retrain's logging
    ensemble, truncated recursion, LONG/RANK), bound to this output, with the v2 predictors, the
    v2 pig columns and the per-Delta symbol gather. ``init``: (family, seed) -> state dict
    (few-shot adaptation starts from the zero-shot member)."""
    h = heavy()
    v3 = h.versions.implementation(3)
    loader = importlib.util.spec_from_file_location("issue110_v2_implementation", ROOT / h.v1.RUNNER)
    impl = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(impl)
    impl.OUTPUT = OUTPUT
    impl.IDENTITY = IDENTITY
    impl.SCHEMA_PREDICTOR = SCHEMA_PREDICTOR
    impl.SCHEMA_GROUP = SCHEMA_GROUP
    impl.SCHEMA_DATA = SCHEMA_DATA
    impl.RUNNER = RUNNER
    impl.MODEL_MODULE = "world_model/training/novelty_v2.py"
    impl.ARMS = (ARM,)
    impl.ARM_DEFINITIONS = {ARM: v3.ARM_DEFINITIONS[ARM]}
    impl.LONG = v3.LONG
    impl.RANK = v3.RANK
    from world_model.model import Abstraction

    def local_loss(ensemble, batch, pair):
        if ensemble.hybrid and pair.abstraction != Abstraction.CONTINUOUS:
            batch = {**batch, **data.symbols(batch, pair.delta)}
        value = h.versions.dd.local_loss(ensemble, batch, pair)
        ensemble.log["local"].append(value.detach())
        return value
    impl.dd = types.SimpleNamespace(Ensemble=h.rt.logging_ensemble(v3.dd.Ensemble), local_loss=local_loss,
                                    horizon_at=v3.dd.horizon_at, ramp_at=v3.dd.ramp_at,
                                    long_and_rank_loss=v3.dd.long_and_rank_loss)
    state = {"seed": None}

    def make(family):
        model = new_predictor(family)
        if init is not None:
            model.load_state_dict(init(family, state["seed"]), strict=True)
        return model
    impl.s99 = types.SimpleNamespace(**{k: getattr(h.s99, k) for k in dir(h.s99) if not k.startswith("__")})
    impl.s99.new_predictor = make
    impl.pig_columns = lambda: (nv().PIG_PRESENCE, nv().PIG_CENTER)
    impl._issue110_state = state
    return impl


def member_file(output, condition, family, seed):
    return Path(output) / "models" / ARM / family / f"fold-{condition}" / f"seed-{seed}" / "predictor.pt"


def train_member(output, condition, family, seed, data, *, steps=None, save=True, timing_after=None, full=None):
    h = heavy()
    if condition == ZERO:
        impl = implementation(data)
        total, smoke_full = steps, False
    else:
        def init(fam, s):
            return torch.load(member_file(output, ZERO, fam, s), map_location="cpu", weights_only=True)["model"]
        impl = implementation(data, init)
        total, smoke_full = steps or FEW_SHOT["updates"], True
    impl._issue110_state["seed"] = seed
    index = f"{condition}-seed{seed}"
    report, ensemble = impl.train_group(output, ARM, family, index, [(condition, seed)], data, steps=total,
                                        save=save, smoke_full=smoke_full if full is None else full,
                                        timing_after=timing_after)
    requests = h.rt.step_requests(family, report["steps"])
    return report, ensemble, requests


def training_groups(condition):
    return [(condition, family, seed) for family in FAMILIES for seed in SEEDS]


def run_training(output, workers, worker, conditions):
    output = Path(output)
    load_plan(output)
    torch.backends.cudnn.benchmark = True
    groups = [g for c in conditions for g in training_groups(c)]
    mine = [g for k, g in enumerate(groups) if k % workers == worker and not updates_path(output, *g).is_file()]
    if conditions != (ZERO,):
        missing = [g for g in training_groups(ZERO) if not updates_path(output, *g).is_file()]
        if missing:
            raise ValueError("few-shot training needs every zero-shot member")
    TrainDataV2 = train_data_class()
    current = None
    for condition, family, seed in mine:
        if current is None or current[:2] != (condition, family):
            current = None
            torch.cuda.empty_cache()
            current = (condition, family, TrainDataV2(load_data(output, condition), family))
        began = time.monotonic()
        report, ensemble, requests = train_member(output, condition, family, seed, current[2])
        atomic_torch(updates_path(output, condition, family, seed), {
            "schema": SCHEMA_UPDATES, "plan_identity": IDENTITY, "condition": condition, "family": family,
            "seed": seed, "requests": requests, "skipped": ensemble.skipped.cpu(), "logs": ensemble.log_tensors(),
            "report": {k: v for k, v in report.items() if k != "trace"}, "trace": report["trace"]})
        del ensemble
        torch.cuda.empty_cache()
        mark(output, f"train-{condition}-{family}-seed{seed}", time.monotonic() - began, report["gpu_seconds"])
        log(f"trained {condition}/{family}/seed {seed}: {report['gpu_seconds']:.0f}s")
    return 0


def train(output, workers, worker):
    return run_training(output, workers, worker, (ZERO,))


def fewshot_train(output, workers, worker):
    return run_training(output, workers, worker, tuple(f"few-shot-{k}" for k in LEVELS))


def member_status(output, condition, family, seed):
    """Usable = complete (no retirement), skips within the v3 cap, and no end-of-training excursion
    (the #112 retrain / #117 rule)."""
    h = heavy()
    updates = torch.load(updates_path(output, condition, family, seed), map_location="cpu", weights_only=True)
    payload = torch.load(member_file(output, condition, family, seed), map_location="cpu", weights_only=True,
                         mmap=True)
    local = updates["logs"]["local"][:, 0].double()
    norms = updates["logs"]["grad_norm"][:, 0].double()
    end_state = h.rt.excursion(local, norms, updates["requests"]) if payload["complete"] else None
    skips = int(updates["skipped"][0])
    usable = bool(payload["complete"]) and skips <= h.rt.SKIP_CAP and not end_state["excursion"]
    return {"condition": condition, "family": family, "seed": seed, "complete": bool(payload["complete"]),
            "retired_at": payload["retired_at"], "skips": skips, "end_state": end_state, "usable": usable,
            "steps": payload["steps"], "gpu_seconds": updates["report"]["gpu_seconds"],
            "parameters": updates["report"]["parameters_per_member"],
            "peak_memory_gb": updates["report"]["peak_memory_gb"]}


def load_member(output, condition, family, seed):
    payload = torch.load(member_file(output, condition, family, seed), map_location="cpu", weights_only=True)
    expected = 9000 if condition == ZERO else FEW_SHOT["updates"]
    if (payload.get("schema") != SCHEMA_PREDICTOR or payload.get("plan_identity") != IDENTITY
            or (payload["family"], payload["fold"], payload["seed"]) != (family, condition, seed)
            or payload["steps"] != expected):
        raise ValueError(f"predictor {condition}/{family}/{seed} binding differs")
    if not payload["complete"]:
        return None
    model = new_predictor(family)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(DEVICE).eval()


# ---------------------------------------------------------------------------
# rollout scoring (issue110-evaluation; under deterministic_scoring)
# ---------------------------------------------------------------------------

def request_pairs():
    from world_model.training.cnn_hybrid import PAIRS
    from world_model.training.matched_dynamics import CONTINUOUS_PAIRS
    out = {r: PAIRS[i] for i, r in enumerate(REQUESTS)}
    out.update({r: CONTINUOUS_PAIRS[i] for i, r in enumerate(CONTINUOUS_REQUESTS)})
    return out


def level_record(output, condition, level, members, policy):
    """Every complete evaluation branch of one level under one condition: masked carrier MSE and
    position MSE of each request's rollout (and the persistence reference) at 15 / 60 / 225."""
    items = [b for b in usable_branches(output) if b["level"] == level]
    shots = [read_label(output, b["key"]) for b in items]
    anchors = torch.stack([full_carriers(output, condition, b["key"])[0] for b in items])
    actions = torch.stack([s["action"] for s in shots]).to(DEVICE)
    targets = {t: [] for t in TIMES}
    presence = {t: [] for t in TIMES}
    for shot in shots:
        engine = nv().engine_carriers(shot).to(DEVICE)
        for t in TIMES:
            q = min(t, int(shot["end"]))
            targets[t].append(engine[q])
            presence[t].append(shot["presence"][q].to(DEVICE))
    targets = {t: torch.stack(v) for t, v in targets.items()}
    presence = {t: torch.stack(v) for t, v in presence.items()}
    pairs = request_pairs()
    errors = defaultdict(dict)

    def score(name, predicted):
        for t in TIMES:
            value = predicted[t]
            finite = torch.isfinite(value).all(-1)
            masked, position = nv().masked_errors(torch.nan_to_num(value), targets[t], presence[t])
            for i, branch in enumerate(items):
                ok = bool(finite[i]) and math.isfinite(float(masked[i])) and math.isfinite(float(position[i]))
                errors[branch["key"]].setdefault(name, {})[str(t)] = (
                    {"masked_carrier_mse": float(masked[i]), "position_mse": float(position[i])} if ok else None)
    score("persistence", {t: anchors.clone() for t in TIMES})
    for (family, seed), model in members.items():
        if model is None:
            continue
        requests = REQUESTS if family == "hybrid" else CONTINUOUS_REQUESTS
        for request in requests:
            score(f"{request}/seed{seed}", rollout_batch(model, anchors, actions, pairs[request]))
    branch = items[0]
    return {"schema": SCHEMA_ROLLOUT, "plan_identity": IDENTITY, "scoring_policy": policy, "condition": condition,
            "level": level, "pair": branch["pair"], "ordinal": branch["ordinal"], "side": branch["side"],
            "cell": branch["cell"], "novelty_level": branch["novelty_level"],
            "members": {f"{f}/seed{s}": m is not None for (f, s), m in members.items()},
            "branches": [{"key": b["key"], "suffix": b["suffix"], "end": int(s["end"]), "errors": errors[b["key"]]}
                         for b, s in zip(items, shots)], "engine_seconds": 0}


@torch.no_grad()
def rollout_batch(model, anchors, actions, pair):
    """Delta-step recursion from each branch's anchor under its action; carriers at TIMES."""
    z = anchors.clone()
    out, elapsed = {}, 0
    while elapsed < max(TIMES):
        z = model.carrier(z, actions, pair)
        elapsed += pair.delta
        if elapsed in TIMES:
            out[elapsed] = z.clone()
    return out


def evaluation_levels():
    seen = {}
    for b in complete():
        if b["stream"] == STREAM_EVAL:
            seen.setdefault(b["level"], b)
    return list(seen.values())


def score_tasks():
    tasks = []
    for branch in evaluation_levels():
        tasks.append((ZERO, branch["level"]))
        tasks.append((f"few-shot-{branch['pair_level']}", branch["level"]))
    return tasks


def score(output, workers, worker):
    output = Path(output)
    load_plan(output)
    h = heavy()
    began = time.monotonic()
    mine = sorted((t for k, t in enumerate(score_tasks()) if k % workers == worker
                   and not rollout_path(output, *t).is_file()), key=lambda t: (t[0] != ZERO, t[0]))
    loaded = {}
    with h.harness.deterministic_scoring() as policy:
        for condition, level in mine:
            if condition not in loaded:
                loaded.clear()
                torch.cuda.empty_cache()
                loaded[condition] = {(f, s): load_member(output, condition, f, s) for f in FAMILIES for s in SEEDS}
            write_json(rollout_path(output, condition, level),
                       level_record(output, condition, level, loaded[condition], policy))
    wall = time.monotonic() - began
    mark(output, f"score-worker{worker}", wall, wall, records=len(mine))
    return 0


# ---------------------------------------------------------------------------
# identifiability probes
# ---------------------------------------------------------------------------

def probe_targets():
    """(task, cell, target, templates side filter) per the frozen #110 identifiability targets."""
    tasks = []
    for cell in design_plan()["cells"]:
        k = cell["novelty_level"]
        if k in (2, 3, 4, 7, 8):
            tasks.append({"cell": cell["cell"], "target": "external-force-active", "sides": ("novel",), "pairs": None})
        if k in (2, 3, 7):
            tasks.append({"cell": cell["cell"], "target": "force-on", "sides": ("novel",), "pairs": None})
        if k == 6:
            tasks.append({"cell": cell["cell"], "target": "gravity-inverted", "sides": ("novel", "normal"), "pairs": None})
        if k == 8:
            tasks.append({"cell": cell["cell"], "target": "storm-active", "sides": ("novel",), "pairs": None})
        if k == 0:
            turbulence = [p for p in cell["captured_by_pairs"] if pair_cells()[p][0] in (3, 7)]
            tasks.append({"cell": cell["cell"], "target": "external-force-active", "sides": ("normal",),
                          "pairs": turbulence})
            tasks.append({"cell": cell["cell"], "target": "force-on", "sides": ("normal",), "pairs": turbulence})
    for task in tasks:
        task["task"] = f"{task['cell'].replace('/', '-')}--{task['target']}"
    return tasks


def probe_branches(task):
    out = []
    level_k = int(task["cell"].split("/")[0])
    for b in complete():
        if b["stream"] not in (STREAM_FIT, STREAM_EVAL) or b["side"] not in task["sides"]:
            continue
        if task["pairs"] is not None:
            if b["pair"] not in task["pairs"]:
                continue
        elif b["pair"] != next(p for c in design_plan()["cells"] if c["cell"] == task["cell"]
                               for p in c["captured_by_pairs"]) or (b["side"] == "novel" and b["novelty_level"] != level_k):
            continue
        out.append(b)
    return out


def frame_targets(shot, target):
    if target == "external-force-active":
        return shot["macros"][:, 2]
    if target == "gravity-inverted":
        return shot["scene"][:, 0] > 0
    if target == "storm-active":
        return shot["scene"][:, 1] > 0.5
    raise ValueError(target)


def auroc(scores, truth):
    """Rank AUROC (ties 0.5)."""
    scores, truth = np.asarray(scores, dtype=np.float64), np.asarray(truth, dtype=bool)
    pos, neg = truth.sum(), (~truth).sum()
    if pos == 0 or neg == 0:
        return None
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[truth].sum() - pos * (pos + 1) / 2) / (pos * neg))


def auroc_bootstrap(scores, truth, clusters):
    """Level-clustered percentile bootstrap of the AUROC on score-quantile bins (10 000 draws)."""
    scores, truth = np.asarray(scores, dtype=np.float64), np.asarray(truth, dtype=bool)
    names = sorted(set(clusters))
    index = {c: i for i, c in enumerate(names)}
    cluster = np.array([index[c] for c in clusters])
    edges = np.unique(np.quantile(scores, np.linspace(0, 1, PROBE["bins"] + 1)[1:-1]))
    bins = np.searchsorted(edges, scores, side="right")
    nb = len(edges) + 1
    pos = np.zeros((len(names), nb))
    neg = np.zeros((len(names), nb))
    np.add.at(pos, (cluster[truth], bins[truth]), 1)
    np.add.at(neg, (cluster[~truth], bins[~truth]), 1)
    generator = np.random.default_rng(np.random.PCG64(BOOTSTRAP["seed"]))
    draws = generator.integers(0, len(names), size=(BOOTSTRAP["draws"], len(names)))
    weights = np.zeros((BOOTSTRAP["draws"], len(names)))
    np.add.at(weights, (np.repeat(np.arange(BOOTSTRAP["draws"]), len(names)), draws.ravel()), 1)
    P, N = weights @ pos, weights @ neg
    below = np.cumsum(N, 1) - N
    with np.errstate(invalid="ignore", divide="ignore"):
        value = ((P * (below + N / 2)).sum(1)) / (P.sum(1) * N.sum(1))
    value = value[np.isfinite(value)]
    if len(value) == 0:
        return None
    low, high = np.quantile(value, BOOTSTRAP["quantiles"])
    return [float(low), float(high)]


def probe_features(output, branch, kind):
    shot = read_label(output, branch["key"])
    if kind == "carrier":
        values = read_carriers(output, ZERO, branch["key"])
        base = torch.cat((values[:, :nv().HEADER], values[:, nv().SLOT0:]), -1)
    else:
        engine = shot["engine"].clone()
        base = torch.cat((shot["presence"].float()[..., None], engine), -1).flatten(1)
    return shot, base


def run_probe_task(output, task, *, updates=None):
    """Leave-one-level-out folds over the cell's fit roster plus a final fold on the whole fit roster
    scored once on the evaluation roster; k in (1, 5, 15); v2 carriers (zero-shot encoder) and the
    engine-feature ceiling (presence, world position and velocity of every slot)."""
    items = probe_branches(task)
    edge = task["target"] == "force-on"
    unit = (lambda b: f"{b['pair']}/{b['ordinal']}") if task["target"] == "gravity-inverted" else (lambda b: b["level"])
    result = {"schema": SCHEMA_PROBE, "plan_identity": IDENTITY, "task": task["task"], "cell": task["cell"],
              "target": task["target"], "sides": list(task["sides"]), "pairs": task["pairs"], "kinds": {},
              "engine_seconds": 0}
    for kind in ("carrier", "engine"):
        bases, parts = [], []
        unit_names = sorted({unit(b) for b in items})
        unit_index = {u: i for i, u in enumerate(unit_names)}
        offset = 0
        for b in items:
            shot, base = probe_features(output, b, kind)
            T = len(base)
            bases.append(base)
            if edge:
                eligible = FORCE_ON_KINDS["Fan" if shot["source_type"] == "Fan" else "turbulence"]
                present = shot["presence"] & torch.tensor([contract.slot_kind(n) in eligible
                                                           for n in contract.VOCABULARY_V2])
                p, s = torch.nonzero(present, as_tuple=True)
                truth_b = shot["forced"][p, s]
            else:
                p = torch.arange(T)
                s = torch.full((T,), -1)
                truth_b = frame_targets(shot, task["target"]).bool()
            n = len(p)
            parts.append(torch.stack((offset + p, torch.full((n,), offset), s, truth_b.long(),
                                      torch.full((n,), unit_index[unit(b)]), torch.full((n,), int(b["split"] == "fit"))), 1))
            offset += T
        base = torch.cat(bases).float().to(DEVICE)
        table = torch.cat(parts).to(DEVICE)
        position, floor, slot, truth, unit_id, fit_mask = (table[:, i] for i in range(6))
        truth, fit_mask = truth.bool(), fit_mask.bool()
        units = [unit_names[i] for i in unit_id.tolist()]
        fit_units = sorted({unit_names[i] for i in unit_id[fit_mask].unique().tolist()})
        folds = [("lolo", u) for u in fit_units] + [("final", None)]
        train_masks = [fit_mask & (unit_id != unit_index[u]) for _, u in folds[:-1]] + [fit_mask]
        test_masks = [unit_id == unit_index[u] for _, u in folds[:-1]] + [~fit_mask]
        block = {}
        for k in PROBE["ks"]:
            torch.manual_seed(PROBE["seed"])
            generator = torch.Generator(device=DEVICE).manual_seed(PROBE["seed"] + k)
            width = (base.shape[1] * (2 * k - 1) if k > 1 else base.shape[1]) + (labels.SLOTS if edge else 0)
            means, stds = [], []
            for train_mask in train_masks:
                rows = base[torch.unique(position[train_mask])]
                means.append(rows.mean(0))
                stds.append(rows.std(0).clamp_min(1e-6))
            probe = nv().BatchedProbe(len(folds), width, PROBE["hidden"]).to(DEVICE)
            optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE["learning_rate"])
            pools = [torch.nonzero(m).flatten() for m in train_masks]
            pos_weight = torch.tensor([float((~truth[p]).sum()) / max(float(truth[p].sum()), 1.0) for p in pools],
                                      device=DEVICE)

            def inputs(rows, f):
                offsets = torch.arange(-k + 1, 1, device=DEVICE)
                index = torch.maximum(position[rows][:, None] + offsets[None], floor[rows][:, None])
                h = (base[index] - means[f]) / stds[f]
                x = h.flatten(1) if k == 1 else torch.cat((h.flatten(1), (h[:, 1:] - h[:, :-1]).flatten(1)), -1)
                if edge:
                    x = torch.cat((x, torch.nn.functional.one_hot(slot[rows], labels.SLOTS).float()), -1)
                return x
            for _ in range(updates or PROBE["updates"]):
                batch = [p[torch.randint(len(p), (PROBE["batch_size"],), device=DEVICE, generator=generator)]
                         for p in pools]
                x = torch.stack([inputs(rows, f) for f, rows in enumerate(batch)])
                y = torch.stack([truth[rows].float() for rows in batch])
                logits = probe(x)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=pos_weight[:, None].expand_as(y), reduction="none").mean(1).sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            scores = {}
            with torch.no_grad():
                for f, test in enumerate(test_masks):
                    rows = torch.nonzero(test).flatten()
                    out = []
                    for start in range(0, len(rows), 4096):
                        chunk = rows[start:start + 4096]
                        x = inputs(chunk, f)
                        out.append((torch.relu(x @ probe.w1[f] + probe.b1[f]) @ probe.w2[f] + probe.b2[f]).squeeze(-1))
                    scores[f] = (rows.cpu(), torch.cat(out).cpu() if out else torch.empty(0))
            lolo_rows = torch.cat([scores[f][0] for f in range(len(folds) - 1)]) if len(folds) > 1 else torch.empty(0)
            lolo_scores = torch.cat([scores[f][1] for f in range(len(folds) - 1)]) if len(folds) > 1 else torch.empty(0)
            entry = {}
            for name, rows, values in (("lolo_fit", lolo_rows, lolo_scores),
                                       ("evaluation", scores[len(folds) - 1][0], scores[len(folds) - 1][1])):
                t = truth[rows.to(DEVICE)].cpu().numpy() if len(rows) else np.zeros(0, dtype=bool)
                c = [units[i] for i in rows.tolist()]
                v = values.numpy()
                entry[name] = {"positives": int(t.sum()), "negatives": int((~t).sum()), "levels": len(set(c)),
                               "auroc": auroc(v, t) if len(v) else None,
                               "interval": auroc_bootstrap(v, t, c) if len(v) and t.any() and (~t).any() else None}
            block[str(k)] = entry
        result["kinds"][kind] = block
    return result


def verdict(entry, ceiling):
    rule = PROBE
    if entry["positives"] < rule["minimum_class_frames"] or entry["negatives"] < rule["minimum_class_frames"] \
            or entry["auroc"] is None or entry["interval"] is None:
        return STOP_TOKEN
    if ceiling is not None and ceiling["auroc"] is not None and ceiling["auroc"] < rule["threshold"]:
        return "label_invalid"
    if entry["auroc"] >= rule["threshold"] and entry["interval"][0] > rule["lower"]:
        return "identifiable"
    if entry["interval"][1] < rule["threshold"]:
        return "not_identifiable"
    return STOP_TOKEN


def probe(output, workers, worker):
    output = Path(output)
    load_plan(output)
    began = time.monotonic()
    mine = [t for k, t in enumerate(probe_targets()) if k % workers == worker and not probe_path(output, t["task"]).is_file()]
    for task in mine:
        started = time.monotonic()
        # phase-A review (B2): any probe written from now on is scored under deterministic_scoring() and
        # records the policy; the 60 retained probe records predate this and carry none (scoring_coverage)
        with heavy().harness.deterministic_scoring() as policy:
            record = run_probe_task(output, task)
        write_json(probe_path(output, task["task"]), {**record, "scoring_policy": policy})
        log(f"probe {task['task']}: {time.monotonic() - started:.0f}s")
    wall = time.monotonic() - began
    mark(output, f"probe-worker{worker}", wall, wall, tasks=len(mine))
    return 0


# ---------------------------------------------------------------------------
# smoke (pre-freeze; nothing kept)
# ---------------------------------------------------------------------------

def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the smoke is a pre-freeze control")
    h = heavy()
    repair_record = read_json(output / "repair" / "fan-repair.json")
    evidence = {"schema": SCHEMA_SMOKE, "plan_identity": IDENTITY, "created_at": utc_now()}
    began = time.monotonic()
    rows = complete()
    evidence["membership"] = {"branches": len(branches()), "complete": len(rows),
                              "by_stream": {s: sum(b["stream"] == s for b in rows) for s in (STREAM_FIT, STREAM_EVAL, STREAM_109)},
                              "zero_image_shots": len(zero_image_keys()),
                              "few_image_shots": {k: len(few_image_keys(k)) for k in LEVELS}}
    # 1. label equality with the #112 retrain shot cache (v1 slots) and s99.shot_labels
    sample = [b for b in rows if b["stream"] == STREAM_109][::700][:20]
    v1_index = [labels.SLOT_INDEX[n] for n in contract.VOCABULARY_V1]
    equal = []
    for b in sample:
        lab = labels.shot_labels(b["attempt"], Path(b["xml"]).read_text())
        ref = torch.load(RETRAIN / "shots" / f"{b['key']}.pt", map_location="cpu", weights_only=True)
        rel = labels.unpack(lab["relations"])[:, v1_index][:, :, v1_index][..., :2]
        msk = labels.relation_mask(lab["presence"], lab["relation_available"])[:, v1_index][:, :, v1_index][..., :2]
        equal.append(bool(torch.equal(rel, ref["relations"]) and torch.equal(msk, ref["relation_mask"])
                          and torch.equal(lab["macros"][:, :2], ref["macros"])
                          and torch.equal(lab["macro_mask"][:, :2], ref["macro_mask"])))
    b = sample[0]
    _, _, frames, segment = h.s99.shot_frames(b["attempt"])
    lab = labels.shot_labels(b["attempt"], Path(b["xml"]).read_text(), images=True)
    v1 = h.s99.shot_labels(segment, frames, list(range(lab["end"] + 1)), h.s99.vocabulary())
    evidence["label_equality"] = {
        "retrain_cache_shots": len(equal), "retrain_cache_equal": sum(equal),
        "s99_presence_equal": bool(torch.equal(lab["presence"][:, v1_index].float(), v1["presence"])),
        "s99_centers_max_abs": float((lab["centers"][:, v1_index] - v1["centers"]).abs().max()),
        "images_equal_s99": bool(torch.equal(lab["images"][0], h.s99.image_tensor(
            Path(b["attempt"]) / "shot-1/observation-trace" / frames[0]["agent_observation"]["relative_path"])))}
    # 2. one label record per novelty level type (novel side, fit) and timing
    per_level, timing = {}, []
    for k in range(0, 9):
        side = "normal" if k == 0 else "novel"
        b = next(x for x in rows if x["stream"] == STREAM_FIT and x["side"] == side and (k == 0 or x["novelty_level"] == k))
        t0 = time.monotonic()
        lab = labels.shot_labels(b["attempt"], Path(b["xml"]).read_text())
        timing.append(time.monotonic() - t0)
        per_level[str(k)] = {"branch": b["key"], "source": lab["source_type"], "outside_v2": lab["outside_v2"],
                             "novelty_present_frames": int(lab["presence"][:, labels.NOVELTY_SLOT].sum()),
                             "force_on_frames": int(lab["forced"].any(1).sum()),
                             "external_force_frames": int(lab["macros"][:, 2].sum()),
                             "gravity_y": float(lab["scene"][0, 0]), "storm_frames": int(lab["scene"][:, 1].sum()),
                             "side": lab["side"], "action": lab["action"].tolist()}
    evidence["labels_per_level"] = per_level
    evidence["label_seconds_per_shot"] = statistics.mean(timing)
    # 3. models: encoder steps, carriers, scene head, per-Delta symbols, predictor updates, scoring, probe
    import tempfile
    with tempfile.TemporaryDirectory(prefix="issue110-smoke-") as tmp:
        tmp = Path(tmp)
        keys = [b["key"] for b in rows if b["stream"] == STREAM_FIT and b["side"] == "normal"][:2]
        keys += [b["key"] for b in rows if b["stream"] == STREAM_FIT and b["side"] == "novel" and b["novelty_level"] == 2][:2]
        for key in keys:
            b = next(x for x in rows if x["key"] == key)
            atomic_torch(label_path(tmp, key), labels.shot_labels(b["attempt"], Path(b["xml"]).read_text(), images=True))
        t0 = time.monotonic()
        payload = train_encoder(tmp, keys, tmp / "enc.pt", steps_limit=1)
        model = new_encoder().to(DEVICE).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        rows64 = {k: torch.cat([image_rows(read_label(tmp, key))[k] for key in keys])[:64].to(DEVICE)
                  for k in image_rows(read_label(tmp, keys[0]))}
        weights = torch.tensor(payload["presence_weights"], device=DEVICE)
        for step in range(13):
            if step == 3:
                torch.cuda.synchronize()
                t0 = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            nv().parser_loss(model, rows64, weights).backward()
            optimizer.step()
        torch.cuda.synchronize()
        encoder_seconds = (time.monotonic() - t0) / 10
        model = new_encoder().to(DEVICE)
        model.load_state_dict(payload["model"])
        model.eval()
        b = next(x for x in rows if x["key"] == keys[2])
        shot = read_label(tmp, keys[2])
        t0 = time.monotonic()
        images = shot_images(b["attempt"], shot["end"])
        values, _ = base_carriers(model, images, shot["times"])
        carrier_seconds = time.monotonic() - t0
        head = nv().SceneHead().to(DEVICE)
        full = head.fill(values.to(DEVICE))
        engine = nv().engine_carriers(shot)
        evidence["carriers"] = {"shape": list(full.shape), "finite": bool(torch.isfinite(full).all()),
                                "engine_shape": list(engine.shape), "seconds_per_shot": carrier_seconds,
                                "pig_presence_column": nv().PIG_PRESENCE,
                                "engine_pig_presence_at_0": float(engine[0, nv().PIG_PRESENCE])}
        # tiny payload through the v2 data class and both families
        fake = {"schema": SCHEMA_DATA, "carriers": full.cpu().repeat(3, 1)[:3 * len(full)],
                "presence": shot["presence"].repeat(3, 1), "relations": shot["relations"].repeat(3, 1),
                "relation_mask": labels.pack(labels.relation_mask(shot["presence"], shot["relation_available"])).repeat(3, 1),
                "macros": shot["macros"].repeat(3, 1), "macro_mask": shot["macro_mask"].repeat(3, 1)}
        T = len(full)
        steps = torch.arange(61)
        wp, ws, wl = [], [], []
        for s in range(3):
            for start in range(0, 226, 15):
                if start >= T - 1:
                    continue
                length = min(60, T - 1 - start)
                wp.append(s * T + start + torch.minimum(steps, torch.tensor(length)))
                ws.append(s)
                wl.append(length)
        fake.update({"window_positions": torch.stack(wp), "window_shot": torch.tensor(ws), "window_length": torch.tensor(wl),
                     "offsets": torch.tensor([0, T, 2 * T]), "lasts": torch.tensor([T - 1] * 3),
                     "shot_actions": shot["action"][None].repeat(3, 1),
                     "long_windows": torch.tensor([(s, 0) for s in range(3)]), "shot_members": ["a", "b", "c"],
                     "cell_members": [], "cell_z0": torch.empty(0, nv().DIM), "cell_action": torch.empty(0, 20, 5),
                     "cell_verdict": torch.empty(0, 20, dtype=torch.long)})
        TrainDataV2 = train_data_class()
        data = TrainDataV2(fake, "hybrid")
        pools = [data.pools("zero-shot")]
        batch = data.windows(pools, [torch.Generator().manual_seed(1)], 4)
        sym = data.symbols(batch, 15)
        # per-Delta rule against window_any on the unpacked shot labels
        dense = labels.unpack(shot["relations"])
        row_positions = batch["positions"][0, 0].cpu()
        expected = labels.window_any(dense, 15)[row_positions % T]
        evidence["per_delta_symbols"] = {"shape": list(sym["relations"].shape),
                                         "equals_window_any": bool(torch.equal(sym["relations"][0, 0].cpu(), expected))}
        reports = {}
        for family in FAMILIES:
            data = TrainDataV2(fake, family)
            report, _, _ = train_member(tmp, ZERO, family, SEEDS[0], data, steps=40, save=False, timing_after=20,
                                        full=True)
            reports[family] = {"seconds_per_update": report["timed_seconds_per_update"],
                               "parameters": report["parameters_per_member"], "peak_memory_gb": report["peak_memory_gb"],
                               "basis": "updates 21-40 of 40 at the most expensive LC update (horizon 225, ranking on)"}
        evidence["predictors"] = {**reports, "continuous_width": nv().continuous_width()}
        # scoring and probe on the tiny data
        hybrid = new_predictor("hybrid").to(DEVICE).eval()
        with h.harness.deterministic_scoring():
            out = rollout_batch(hybrid, full[:1].repeat(2, 1), shot["action"][None].repeat(2, 1).to(DEVICE),
                                request_pairs()["F-1-macro"])
            masked, position = nv().masked_errors(out[225], engine[min(225, T - 1)][None].repeat(2, 1).to(DEVICE),
                                                  shot["presence"][min(225, T - 1)][None].repeat(2, 1).to(DEVICE))
        evidence["scoring"] = {"times": sorted(out), "finite": bool(torch.isfinite(masked).all())}
        evidence["auroc_check"] = {"perfect": auroc([0.1, 0.2, 0.9, 0.8], [0, 0, 1, 1]),
                                   "ties": auroc([0.5, 0.5], [0, 1]),
                                   "bootstrap": auroc_bootstrap(np.r_[np.random.default_rng(0).normal(0, 1, 400),
                                                                      np.random.default_rng(1).normal(1, 1, 400)],
                                                                np.r_[np.zeros(400, bool), np.ones(400, bool)],
                                                                [i % 20 for i in range(800)])}
    # 4. fan repair verdicts and cost projection
    evidence["fan_repair"] = {k: {lab: v[lab]["verdict"] for lab in ("legacy", "repaired")}
                              for k, v in repair_record["summary"].items()}
    shots_all = len(rows)
    evidence["projection"] = {
        "labels_hours": shots_all * evidence["label_seconds_per_shot"] / LABEL_PROCESSES / 3600,
        "encoder_hours": 12 * len(zero_image_keys()) * 58 * encoder_seconds / 64 / 3600,
        "carrier_hours_4gpu": (shots_all + sum(len([c for c in conditions_of(b) if c != ZERO]) for b in rows))
        * carrier_seconds / 4 / 3600,
        "zero_shot_training_hours_4gpu": 2 * 9000 * max(reports[f]["seconds_per_update"] for f in FAMILIES) / 3600,
        "few_shot_training_hours_4gpu": 48 * 2000 * max(reports[f]["seconds_per_update"] for f in FAMILIES) / 4 / 3600,
    }
    evidence["projection"]["total_wall_hours_upper"] = sum(evidence["projection"].values()) + 4.0
    evidence["wall_seconds"] = time.monotonic() - began
    passed = (evidence["label_equality"]["retrain_cache_equal"] == len(equal)
              and evidence["label_equality"]["s99_presence_equal"]
              and evidence["label_equality"]["s99_centers_max_abs"] < 1e-6
              and evidence["per_delta_symbols"]["equals_window_any"] and evidence["carriers"]["finite"]
              and evidence["scoring"]["finite"]
              and all(v["repaired"] in ("agree", "not_exercised") for v in evidence["fan_repair"].values())
              and evidence["projection"]["total_wall_hours_upper"] < 48)
    evidence["passed"] = bool(passed)
    write_json(output / "smoke.json", evidence)
    log(f"smoke {'passed' if passed else 'FAILED'}; projection {evidence['projection']}")
    return 0 if passed else 1


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

INPUTS = {
    "contract_module": "world_model/data/novelty_contract.py",
    "labels_module": "world_model/data/novelty_labels_v2.py",
    "models_module": "world_model/training/novelty_v2.py",
    "spatial_slot_parser": "world_model/training/spatial_slot_parser.py",
    "cnn_hybrid": "world_model/training/cnn_hybrid.py",
    "matched_dynamics": "world_model/training/matched_dynamics.py",
    "decision_dynamics": "world_model/training/decision_dynamics.py",
    "stable_decision_dynamics": "world_model/training/stable_decision_dynamics.py",
    "issue_112_v1_runner": "scripts/run_decision_relevant_dynamics.py",
    "issue_112_versions_runner": "scripts/run_decision_dynamics_versions.py",
    "issue_112_retrain_runner": "scripts/run_decision_dynamics_retrain.py",
    "scoring_harness": "world_model/planning/scoring_harness.py",
    "runner": RUNNER,
    "issue_110_design_plan": ".local-artifacts/issue-110-novelty-coverage-v1/plan.json",
    "issue_104_campaign_branches": ".local-artifacts/issue-104-power-cohort-campaign-v1/branches.json",
    "issue_104_smoke_summary": ".local-artifacts/issue-104-capture-smoke-v1/summary.json",
    "issue_109_cohort_plan": ".local-artifacts/issue-109-sealed-cohort-v1/plan.json",
    "issue_113_window_spec": ".local-artifacts/issue-113-capture-window-v1/spec.json",
}


def input_bindings():
    return [{"name": name, "artifact": path, "sha256": sha256_of(ROOT / path)} for name, path in INPUTS.items()]


def frozen_plan(output, frozen_at, smoke_evidence):
    rows = complete()
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "issue": 110, "frozen_at": frozen_at,
        "validation_command": VALIDATION_COMMAND, "runner": RUNNER,
        "inputs": input_bindings(), "smoke_sha256": sha256_of(Path(output) / "smoke.json"),
        "repair_sha256": sha256_of(Path(output) / "repair" / "fan-repair.json"),
        "question": ("Can contract v2 (28 slots, scene state, novelty entity, symmetric action contract, declared "
                     "level-7 objective) carry every NovPhy scenario x novelty cell through the E99 encoder and the "
                     "#112 v3 LC predictor, and with what rollout error, zero-shot from normal training and "
                     "few-shot per novelty level?"),
        "membership": {"streams": [STREAM_FIT, STREAM_EVAL, STREAM_109], "branches": len(branches()),
                       "complete": len(rows),
                       "order": "stream, pair, roster ordinal, side (normal, novel), candidate suffix",
                       "zero_shot_training": "split fit, normal side: issue110-fit normal sides + issue109-fit",
                       "few_shot_training": "issue110-fit novel sides of the novelty level's five pairs",
                       "evaluation": "issue110-evaluation, both sides, never fitted (zero-shot and few-shot scored)",
                       "zero_shot_encoder_images": {"rule": "normal fit shots at grid ordinals 0, 5, 10, 15; image "
                                                            "stride 5 (E99)", "shots": len(zero_image_keys())},
                       "few_shot_encoder_images": {"rule": "every complete novel fit shot of the level; stride 5",
                                                   "shots": {str(k): len(few_image_keys(k)) for k in LEVELS}}},
        "labeler_repair": {"law": labels.FAN_LAW_REPAIRED,
                           "evidence": "repair/fan-repair.json (residual acceleration, smoke + fit fan captures)",
                           "frozen_contract_untouched": "world_model/data/novelty_contract.py is unchanged; the "
                                                        "repaired law lives in novelty_labels_v2.py"},
        "labels": {"module": INPUTS["labels_module"], "relations": list(labels.RELATIONS), "macros": list(labels.MACROS),
                   "scene_targets": ["gravity_y / 9.8", "storm-active (first destroyed bird <= step, Storm levels)"],
                   "per_delta_rule": ("a predicate requested at Delta is true when it holds at any frame in (t, t + "
                                      "Delta]; the label of a prediction landing at window position j is the OR of "
                                      "the per-frame labels at j - Delta + 1 .. j (clamped to the shot's first "
                                      "frame); its mask is the OR of the per-frame masks"),
                   "v1_equivalence": "on the v1 slots the v2 labels equal the #99 / #112 labels (smoke)",
                   "approximations": {**contract.LABEL_APPROXIMATIONS, "fan": labels.FAN_LAW_REPAIRED["label"]}},
        "encoder": {"architecture": "SpatialSlotParserV2 (E99 SpatialSlotParser, 28 queries, 9 kinds, 3 relations, 3 macros)",
                    "zero_shot_recipe": ENCODER, "few_shot_recipe": FEW_SHOT,
                    "carrier": "2 header + 4 scene + 13 x 28 = 370 (TemporalVisualCarrierAdapter features, thresholds 0.5, temperatures 1)",
                    "scene_head": {**SCENE_RECIPE, "inputs": "header + slot columns of the last 15 carriers and their 1-frame differences, "
                                   "left-padded with the shot's first frame",
                                   "outputs": "gravity_y_normalized (MSE), storm_active (BCE, sigmoid)",
                                   "availability": "1 iff the full 15-frame history exists (position >= 14)",
                                   "few_shot": "from the zero-shot head, lr 1e-4, the few-shot batch and updates"}},
        "anchor_rule_v2": ("the v2 carrier of shot frame 0 (decision state, native step 30000): the condition's encoder "
                           "parse with no prior (header [0, 0], no motion), its scene head over the one-frame history "
                           "(scene availability 0)"),
        "predictor": {"recipe": "#112 v3 LC (retrain implementation: logging v3 ensemble, truncation 45, LONG, RANK, "
                                "9000 updates, batch 64, lr 1e-4, wd 1e-4, clip 1.0), group size 1",
                      "families": list(FAMILIES), "seeds": list(SEEDS),
                      "continuous_width": nv().continuous_width(),
                      "ranking_cells": "every training level x inventory with both verdicts (#112 retrain rule); "
                                       "a few-shot level without one has a zero ranking term",
                      "few_shot": FEW_SHOT, "usable": "complete, skips <= 45, no #117 end-of-training excursion"},
        "action_contract": {"bounds": contract.ACTION_BOUNDS_V2, "encoding": contract.ACTION_ENCODING,
                            "level_5": "candidates are the #93 grid mirrored (drag_x -> -drag_x), same suffixes"},
        "objective": contract.TASK_OBJECTIVE,
        "rollout_error": {"requests": list(REQUESTS), "descriptive_requests": list(CONTINUOUS_REQUESTS),
                          "reference": "persistence (the anchor carrier held fixed)", "times": list(TIMES),
                          "anchor": "the v2 anchor rule", "target": "engine-projected v2 carrier at min(t, end)",
                          "metric": ("masked_carrier_mse: mean squared error over the 13 features of every "
                                     "engine-present slot plus gravity_y_normalized and storm_active; "
                                     "position_mse: slot centers only"),
                          "aggregation": "branch -> mean over usable seeds; level -> mean over its complete branches; "
                                         "cell -> level-clustered bootstrap",
                          "conditions": {"zero_shot": "every evaluation level", "few_shot": "levels of novelty-level-k pairs "
                                         "(both sides) under the level-k adaptation"},
                          "pooling": "never pooled across conditions",
                          "contrast": "novel minus paired normal (same pair and ordinal), level-clustered bootstrap"},
        "identifiability": {**PROBE, "targets": probe_targets(),
                            "inputs": "zero-shot v2 carriers (header + slots, no scene columns) of the last k frames + "
                                      "1-frame differences, per-fold standardized; force-on samples (frame, body) over "
                                      "the bodies the law can act on (FORCE_ON_KINDS) and adds the body slot one-hot",
                            "force_on_bodies": FORCE_ON_KINDS,
                            "ceiling": "the same probe on engine features (presence, x, y, vx, vy of every slot)",
                            "folds": "leave one fit level out (gravity-inverted: one pair ordinal, both sides) + a final "
                                     "probe on the whole fit roster scored once on the evaluation roster",
                            "verdict_basis": "the evaluation-roster AUROC at k = 15 (LOLO reported)",
                            "rule": design_plan()["protocols"]["identifiability"]["rule"]},
        "readiness": {"supported": ("smoke gate supported, every captured level fits v2, at least one complete "
                                    "evaluation level, finite zero-shot rollout errors for all nine requests with at "
                                    "least one usable hybrid seed, and every guard holds"),
                      "multiple_forces": "rollout error only (no single-shot decision headroom)"},
        "guards": {"G1_isolation": "no evaluation level in any training or adaptation set; no novel shot in zero-shot training",
                   "G2_labels": "smoke label equality and repaired fan labels agree on every exercised fan template",
                   "G3_training": "every condition keeps >= 1 usable hybrid member",
                   "G4_scoring": "every record under deterministic_scoring; record hashes match the manifest",
                   "failure": "a failed guard turns every token into readiness_or_precision_insufficient"},
        "bootstrap": BOOTSTRAP, "smoke": {"passed": smoke_evidence["passed"], "projection": smoke_evidence["projection"]},
        "hardware": "4x RTX 5090 (the #85 rules name an RTX 3090)",
        "claim_boundary": ("Rollout error and identifiability are DESCRIPTIVE. Zero-shot and few-shot are separate "
                           "conditions, never pooled. No decision or ranking claim is made here (#107)."),
        "engine_seconds": 0,
    }


def prepare(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        load_plan(output)
        return 0
    evidence = read_json(output / "smoke.json")
    if not evidence["passed"]:
        raise ValueError("the smoke failed; no freeze")
    write_json(output / "plan.json", frozen_plan(output, utc_now(), evidence))
    log("plan frozen")
    return 0


# ---------------------------------------------------------------------------
# publication
# ---------------------------------------------------------------------------

def bootstrap(values, clusters):
    grouped = defaultdict(list)
    for value, cluster in zip(values, clusters, strict=True):
        grouped[cluster].append(value)
    if not grouped:
        return None
    means = np.array([np.mean(grouped[k]) for k in sorted(grouped)])
    generator = np.random.default_rng(np.random.PCG64(BOOTSTRAP["seed"]))
    draws = generator.integers(0, len(means), size=(BOOTSTRAP["draws"], len(means)))
    low, high = np.quantile(means[draws].mean(axis=1), BOOTSTRAP["quantiles"])
    return {"mean": float(means.mean()), "interval": [float(low), float(high)], "clusters": len(means),
            "label": BOOTSTRAP["label"]}


def level_values(record, usable, metric="masked_carrier_mse"):
    """{request: {t: level mean}} of a rollout record (branch: mean over usable seeds)."""
    out = defaultdict(dict)
    names = list(REQUESTS) + list(CONTINUOUS_REQUESTS)
    for request in names + ["persistence"]:
        for t in TIMES:
            branch_values, nonfinite = [], 0
            for branch in record["branches"]:
                if request == "persistence":
                    cell = branch["errors"]["persistence"][str(t)]
                    seeds = [] if cell is None else [cell[metric]]
                else:
                    family = "hybrid" if request in REQUESTS else "continuous"
                    seeds = []
                    for seed in SEEDS:
                        if not usable.get((record["condition"], family, seed)):
                            continue
                        cell = branch["errors"].get(f"{request}/seed{seed}", {}).get(str(t))
                        if cell is None:
                            nonfinite += 1
                            continue
                        seeds.append(cell[metric])
                if seeds:
                    branch_values.append(float(np.mean(seeds)))
            out[request][t] = {"value": float(np.mean(branch_values)) if branch_values else None,
                               "branches": len(branch_values), "nonfinite": nonfinite}
    return out


def rollout_tables(output, usable):
    records = {}
    for condition, level in score_tasks():
        records[(condition, level)] = read_json(rollout_path(output, condition, level))
    per = {key: level_values(r, usable) for key, r in records.items()}
    per_pos = {key: level_values(r, usable, "position_mse") for key, r in records.items()}
    normal_of = {(k[0], r["pair"], r["ordinal"]): k for k, r in records.items() if r["side"] == "normal"}
    tables = {}
    for cell in cells():
        level_k = int(cell.split("/")[0])
        rows_cell = {}
        for condition in ([ZERO] if level_k == 0 else [ZERO, f"few-shot-{level_k}"]):
            keys = [k for k, r in records.items() if k[0] == condition and r["cell"] == cell]
            block = {}
            for request in list(REQUESTS) + list(CONTINUOUS_REQUESTS) + ["persistence"]:
                block[request] = {}
                for t in TIMES:
                    vals = [(per[k][request][t]["value"], k[1]) for k in keys if per[k][request][t]["value"] is not None]
                    pos = [(per_pos[k][request][t]["value"], k[1]) for k in keys
                           if per_pos[k][request][t]["value"] is not None]
                    entry = {"masked_carrier_mse": bootstrap([v for v, _ in vals], [c for _, c in vals]),
                             "position_mse": bootstrap([v for v, _ in pos], [c for _, c in pos]),
                             "levels": len(keys), "levels_finite": len(vals),
                             "nonfinite_seed_rollouts": sum(per[k][request][t]["nonfinite"] for k in keys)}
                    if level_k > 0:
                        paired = []
                        for k in keys:
                            r = records[k]
                            mate = normal_of.get((condition, r["pair"], r["ordinal"]))
                            a = per[k][request][t]["value"]
                            b = None if mate is None else per[mate][request][t]["value"]
                            if a is not None and b is not None:
                                paired.append((a - b, f"{r['pair']}/{r['ordinal']}"))
                        entry["novel_minus_normal"] = bootstrap([v for v, _ in paired], [c for _, c in paired])
                    block[request][str(t)] = entry
            rows_cell[condition] = block
        tables[cell] = rows_cell
    return tables, records


def static_fit(output):
    """Per template (80): captured levels, levels whose every active bodied trace entity is a v2 slot,
    v1 fit (no presence outside the 18 v1 slots), and the zero-shot parse at the decision frame."""
    v1_index = torch.tensor([labels.SLOT_INDEX[n] for n in contract.VOCABULARY_V1])
    others = torch.tensor([i for i in range(labels.SLOTS) if i not in set(v1_index.tolist())])
    rows = defaultdict(lambda: {"levels": set(), "outside_v2": set(), "outside_v1": set(), "presence_correct": 0,
                                "presence_slots": 0, "center_sq": 0.0, "center_n": 0})
    failed = label_failures(output)
    for b in complete():
        if b["stream"] == STREAM_109 or b["key"] in failed:
            continue
        template = b["template"]
        shot = read_label(output, b["key"])
        entry = rows[template]
        entry["levels"].add(b["level"])
        if shot["outside_v2"]:
            entry["outside_v2"].add(b["level"])
        if bool(shot["presence"][:, others].any()):
            entry["outside_v1"].add(b["level"])
        if b["stream"] == STREAM_EVAL:
            parse = torch.load(carrier_path(output, ZERO, b["key"]), map_location="cpu", weights_only=True)["decision_parse"]
            truth = shot["presence"][0]
            guess = parse["presence"] >= 0.5
            entry["presence_correct"] += int((guess == truth).sum())
            entry["presence_slots"] += labels.SLOTS
            if truth.any():
                entry["center_sq"] += float(((parse["centers"][truth] - shot["centers"][0][truth]) ** 2).sum(-1).sum())
                entry["center_n"] += int(truth.sum())
    out = []
    for template in sorted(rows):
        e = rows[template]
        out.append({"template": template, "levels_captured": len(e["levels"]),
                    "levels_fit_v2": len(e["levels"] - e["outside_v2"]),
                    "levels_fit_v1": len(e["levels"] - e["outside_v1"]),
                    "v2_static_fit": len(e["outside_v2"]) == 0,
                    "decision_frame_presence_accuracy": e["presence_correct"] / max(e["presence_slots"], 1),
                    "decision_frame_center_rmse": math.sqrt(e["center_sq"] / e["center_n"]) if e["center_n"] else None})
    return out


def headroom():
    """Per cell: evaluation levels and mixed-verdict levels (grid) -- #107's decision headroom."""
    out = defaultdict(lambda: {"levels": set(), "mixed": 0, "branches": 0, "pig_removed": 0})
    by_level = defaultdict(list)
    for b in complete():
        if b["stream"] == STREAM_EVAL:
            by_level[(b["cell"], b["level"])].append(bool(b["pig_removed"]))
    for (cell, level), values in by_level.items():
        out[cell]["levels"].add(level)
        out[cell]["branches"] += len(values)
        out[cell]["pig_removed"] += sum(values)
        out[cell]["mixed"] += set(values) == {True, False}
    return {c: {"evaluation_levels": len(v["levels"]), "complete_branches": v["branches"],
                "pig_removed": v["pig_removed"], "mixed_levels": v["mixed"]} for c, v in out.items()}


def probe_tables(output):
    out = {}
    for task in probe_targets():
        record = read_json(probe_path(output, task["task"]))
        block = {}
        for k in PROBE["ks"]:
            carrier = record["kinds"]["carrier"][str(k)]
            engine = record["kinds"]["engine"][str(k)]
            block[str(k)] = {"carrier": carrier, "engine": engine,
                             "verdict": verdict(carrier["evaluation"], engine["evaluation"])}
        out[task["task"]] = {"cell": task["cell"], "target": task["target"], "by_k": block,
                             "verdict": block[str(PROBE["primary_k"])]["verdict"]}
    return out


def invalidate(obj):
    """A failed guard turns every verdict into readiness_or_precision_insufficient (frozen guards.failure),
    including each probe's by-k verdicts; AUROCs, intervals and counts are kept."""
    if isinstance(obj, dict):
        return {k: STOP_TOKEN if k == "verdict" else invalidate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [invalidate(v) for v in obj]
    return obj


PROBE_SCORING_NOTE = (
    "the identifiability probes (MLP probes trained and scored in-process by --probe) ran without "
    "deterministic_scoring() and their records carry no scoring policy; the frozen G4 check iterates the rollout "
    "records only, so the plan's G4 text ('every record under deterministic_scoring') does not hold for the probe "
    "records. The probe models were not retained, so the verdicts cannot be re-scored without retraining, which a "
    "post-freeze change may not do; they are reported as recorded. A fresh process importing this runner runs with "
    "matmul TF32 off (as SCORING_POLICY) and cuDNN deterministic off / cuDNN TF32 on; the probes use no cuDNN "
    "operation. Borderline verdicts are not guaranteed to reproduce. G4 and every token are unchanged")


def scoring_coverage(output, compute):
    """Which records the frozen G4 scoring-policy check covers (phase-A review B2)."""
    policy = heavy().harness.SCORING_POLICY
    probes = [read_json(probe_path(output, t["task"])) for t in probe_targets()]
    return {"rollout_records": len(score_tasks()), "rollout_policy_checked_by_G4": True,
            "G4_scoring": compute["guards"]["G4_scoring"], "probe_records": len(probes),
            "probe_records_with_scoring_policy": sum(r.get("scoring_policy") == policy for r in probes),
            "probe_policy_checked_by_G4": False, "note": PROBE_SCORING_NOTE}



def compute_tables(output, plan):
    output = Path(output)
    usable_rows = [member_status(output, c, f, s) for c in condition_levels() for f in FAMILIES for s in SEEDS]
    usable = {(r["condition"], r["family"], r["seed"]): r["usable"] for r in usable_rows}
    tables, records = rollout_tables(output, usable)
    fit = static_fit(output)
    room = headroom()
    probes = probe_tables(output)
    failures = label_failures(output)
    smoke104 = read_json(SMOKE104 / "summary.json")["cells"]["table"]
    repair_record = read_json(output / "repair" / "fan-repair.json")
    # guards
    training_keys = set()
    for condition in condition_levels():
        payload = load_data(output, condition)
        training_keys |= set(payload["shot_keys"])
        side = {b["key"]: b["side"] for b in complete()}
        if condition == ZERO and any(side[k] == "novel" for k in payload["shot_keys"]):
            raise ValueError("a novel shot entered the zero-shot training data")
    evaluation_keys = {b["key"] for b in complete() if b["stream"] == STREAM_EVAL}
    images = set(zero_image_keys()) | {k for level in LEVELS for k in few_image_keys(level)}
    g1 = not (training_keys & evaluation_keys) and not (images & evaluation_keys)
    g2 = all(v["repaired"]["verdict"] in ("agree", "not_exercised") for v in repair_record["summary"].values()) \
        and read_json(output / "smoke.json")["passed"]
    g3 = all(any(usable[(c, "hybrid", s)] for s in SEEDS) for c in condition_levels())
    policies = all(r["scoring_policy"] == heavy().harness.SCORING_POLICY for r in records.values())
    manifest = record_manifest(output)
    g4 = policies and manifest_matches(output, manifest)
    guards = {"G1_isolation": g1, "G2_labels": bool(g2), "G3_training": g3, "G4_scoring": bool(g4)}
    guards_ok = all(guards.values())
    if not guards_ok:
        probes = invalidate(probes)
    # readiness
    template_fit = {r["template"]: r for r in fit}
    readiness = []
    for cell in cells():
        level_k, scenario = int(cell.split("/")[0]), cell.split("/")[1]
        design_cell = next(c for c in design_plan()["cells"] if c["cell"] == cell)
        templates = sorted({b["template"] for b in complete() if b["cell"] == cell and b["stream"] != STREAM_109})
        reasons = []
        smoke_entry = smoke104.get(f"{scenario}|novelty_level_{level_k}", {})
        if smoke_entry.get("status") != "supported":
            reasons.append("smoke_gate_unsupported")
        if not all(template_fit[t]["v2_static_fit"] for t in templates):
            reasons.append("static_fit_outside_v2")
        if room.get(cell, {}).get("evaluation_levels", 0) == 0:
            reasons.append("no_complete_evaluation_level")
        zero = tables[cell][ZERO]
        if any(zero[r][str(t)]["masked_carrier_mse"] is None for r in REQUESTS for t in TIMES):
            reasons.append("no_finite_zero_shot_rollout")
        if not guards_ok:
            reasons.append("guard_failed")
        status = STOP_TOKEN if not guards_ok else "supported" if not reasons else "unsupported"
        flags = []
        if scenario == "multiple_forces":
            flags.append("rollout_error_only: degenerate single-shot pig-removal verdicts (multi-hit pigs)")
        if room.get(cell, {}).get("mixed_levels", 0) == 0:
            flags.append("no_mixed_verdict_evaluation_level")
        cell_failures = sum(1 for b in complete() if b["cell"] == cell and b["key"] in failures)
        readiness.append({"cell": cell, "novelty_level": level_k, "scenario": scenario, "status": status,
                          "reasons": reasons, "flags": flags, "templates": templates,
                          "visibility": design_cell["visibility"], "force_laws": design_cell["force_laws"],
                          "headroom": room.get(cell), "label_failures": cell_failures,
                          "identifiability": {t["target"]: probes[t["task"]]["verdict"] if guards_ok else STOP_TOKEN
                                              for t in probe_targets() if t["cell"] == cell},
                          "few_shot": None if level_k == 0 else f"few-shot-{level_k}"})
    ledger = {p.stem: read_json(p) for p in sorted((output / "ledger").glob("*.json"))}
    return {"schema": SCHEMA_COMPUTE, "identity": IDENTITY, "frozen_at": plan["frozen_at"], "guards": guards,
            "guards_ok": guards_ok, "members": usable_rows, "readiness": readiness, "rollout": tables,
            "static_fit": fit, "identifiability": probes, "headroom": room,
            "label_failures": failures, "repair": repair_record["summary"],
            "record_manifest_sha256": sha256_text(json_text(manifest)),
            "compute": {"phases": ledger, "gpu_seconds": sum(b["gpu_seconds"] for b in ledger.values()),
                        "wall_seconds": sum(b["wall_seconds"] for b in ledger.values()), "engine_seconds": 0}}


def record_manifest(output):
    output = Path(output)
    files = sorted(output.glob("records/**/*.json"))
    return {"schema": "issue_110_record_manifest_v1", "records": {str(p.relative_to(output)): sha256_of(p) for p in files}}


def manifest_matches(output, manifest):
    path = Path(output) / "records-manifest.json"
    return not path.is_file() or read_json(path) == manifest


def checkpoint(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_of(path)}


def handoff(output, plan, compute):
    output = Path(output)
    token = "supported" if compute["guards_ok"] else STOP_TOKEN
    supported = [r for r in compute["readiness"] if r["status"] == "supported"]
    evaluation = defaultdict(list)
    for b in complete():
        if b["stream"] == STREAM_EVAL:
            evaluation[b["cell"]].append({"key": b["key"], "level": b["level"], "pair": b["pair"], "ordinal": b["ordinal"],
                                          "side": b["side"], "suffix": b["suffix"], "action": b["action"],
                                          "inventory": "grid", "pig_removed": bool(b["pig_removed"])})
    return {
        "schema": SCHEMA_HANDOFF, "identity": IDENTITY, "frozen_at": plan["frozen_at"], "consumer": 107, "token": token,
        "supported_cells": [r["cell"] for r in supported],
        "unsupported_cells": [{"cell": r["cell"], "reasons": r["reasons"]} for r in compute["readiness"]
                              if r["status"] == "unsupported"],
        **({"insufficient_cells": [{"cell": r["cell"], "reasons": r["reasons"]} for r in compute["readiness"]
                                   if r["status"] == STOP_TOKEN]} if not compute["guards_ok"] else {}),
        "cells": {r["cell"]: {"flags": r["flags"], "headroom": r["headroom"], "identifiability": r["identifiability"],
                              "templates": r["templates"], "few_shot_condition": r["few_shot"]}
                  for r in compute["readiness"]},
        "contract": {"identity": contract.CONTRACT_IDENTITY, "vocabulary": list(contract.VOCABULARY_V2),
                     "carrier_dim": contract.CARRIER_DIM_V2, "scene_fields": list(contract.SCENE_FIELDS),
                     "kinds": list(contract.KIND_VOCABULARY_V2), "relations": list(labels.RELATIONS),
                     "macros": list(labels.MACROS), "pig_presence_column": nv().PIG_PRESENCE,
                     "pig_center_columns": list(nv().PIG_CENTER), "action_encoding": contract.ACTION_ENCODING,
                     "action_bounds": contract.ACTION_BOUNDS_V2, "objective": contract.TASK_OBJECTIVE,
                     "labeler_repair": labels.FAN_LAW_REPAIRED},
        "anchor_rule_v2": plan["anchor_rule_v2"],
        "loading": {"encoder": "scripts.run_novelty_coverage.load_encoder(OUTPUT, condition)",
                    "scene_head": "scripts.run_novelty_coverage.load_scene(str(OUTPUT), condition)",
                    "predictor": "scripts.run_novelty_coverage.load_member(OUTPUT, condition, family, seed)",
                    "carriers": "scripts.run_novelty_coverage.full_carriers(OUTPUT, condition, key)"},
        "checkpoints": {
            "encoders": {c: checkpoint(encoder_path(output, c)) for c in condition_levels()},
            "scene_heads": {c: checkpoint(scene_path(output, c)) for c in condition_levels()},
            "predictors": {f"{r['condition']}/{r['family']}/seed{r['seed']}":
                           {**checkpoint(member_file(output, r["condition"], r["family"], r["seed"])), "usable": r["usable"]}
                           for r in compute["members"]}},
        "conditions": {"zero-shot": "encoder, scene head and predictors fitted on normal fit data only",
                       **{f"few-shot-{k}": f"zero-shot adapted on the novelty-level-{k} novel fit sides" for k in LEVELS}},
        "evaluation_inventories": {"stream": STREAM_EVAL, "inventory": "grid (16, #93 Arm B; mirrored on the right "
                                   "slingshot)", "verdict": "engine pig removal (#85 channel, branches.json)",
                                   "by_cell": {c: evaluation[c] for c in cells()}},
        "multiple_forces": "rollout error only; no single-shot decision headroom",
        "records_manifest": "records-manifest.json",
    }


def fmt(value, digits=4):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}" if abs(value) >= 1e-3 or value == 0 else f"{value:.2e}"


def interval_text(block, digits=4):
    if block is None:
        return "n/a"
    return f"{fmt(block['mean'], digits)} [{fmt(block['interval'][0], digits)}, {fmt(block['interval'][1], digits)}]"


def findings_md(plan, compute, coverage, encoders):
    lines = []
    add = lines.append
    add("# #110 results: contract-v2 coverage of all 45 NovPhy cells")
    add("")
    add(f"`{VALIDATION_COMMAND}`; plan frozen {plan['frozen_at']}. Guards: "
        + ", ".join(f"{k} {'pass' if v else 'FAIL'}" for k, v in compute["guards"].items()) + ".")
    add("")
    supported = sum(r["status"] == "supported" for r in compute["readiness"])
    add(f"**Readiness: {supported} / 45 cells supported.**")
    add("")
    add("| cell | status | reasons | flags | eval levels | mixed levels | identifiability (k = 15) |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for r in compute["readiness"]:
        ident = "; ".join(f"{t}: {v}" for t, v in r["identifiability"].items()) or "not required"
        room = r["headroom"] or {}
        add(f"| {r['cell']} | {r['status']} | {', '.join(r['reasons']) or '-'} | "
            f"{'; '.join(f.split(':')[0] for f in r['flags']) or '-'} | {room.get('evaluation_levels', 0)} | "
            f"{room.get('mixed_levels', 0)} | {ident} |")
    add("")
    add("## Rollout error at 225 frames (masked carrier MSE, level-clustered bootstrap)")
    add("")
    add("Zero-shot (normal training) and few-shot (level adaptation) are separate conditions; never pooled.")
    add("")
    header = "| cell | condition | " + " | ".join(REQUESTS) + " | persistence |"
    add(header)
    add("|" + " --- |" * (3 + len(REQUESTS)))
    for cell, block in compute["rollout"].items():
        for condition, rows in block.items():
            add(f"| {cell} | {condition} | " + " | ".join(fmt((rows[r]['225']['masked_carrier_mse'] or {}).get('mean'))
                                                         for r in REQUESTS)
                + f" | {fmt((rows['persistence']['225']['masked_carrier_mse'] or {}).get('mean'))} |")
    add("")
    add("Full tables (15 / 60 / 225 frames, intervals, position MSE, novel - normal contrasts, C-F-* rows): "
        "`rollout-error.csv` and `compute.json`.")
    add("")
    add("## Static fit (80 templates)")
    add("")
    fit = compute["static_fit"]
    add(f"v2: {sum(r['v2_static_fit'] for r in fit)} / {len(fit)} templates fit on every captured level; "
        f"v1: {sum(r['levels_fit_v1'] == r['levels_captured'] for r in fit)} / {len(fit)}.")
    add("")
    add("## Fan labeler repair")
    add("")
    for key, block in compute["repair"].items():
        add(f"- {key}: legacy {block['legacy']['verdict']} (median {fmt(block['legacy']['median_of_branch_medians'], 1)}),"
            f" repaired {block['repaired']['verdict']} (median {fmt(block['repaired']['median_of_branch_medians'], 1)}, "
            f"precision {fmt(block['repaired']['precision'], 3)}, recall {fmt(block['repaired']['recall'], 3)})")
    add("")
    add("## Members")
    add("")
    add("| condition | family | seed | usable | skips | excursion | GPU s |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for m in compute["members"]:
        add(f"| {m['condition']} | {m['family']} | {m['seed']} | {m['usable']} | {m['skips']} | "
            f"{(m['end_state'] or {}).get('excursion')} | {m['gpu_seconds']:.0f} |")
    add("")
    add("## Encoder training (`encoder-training.json`, read from the checkpoints)")
    add("")
    zero = encoders["encoders"][ZERO]["reports"]
    add(f"- {ZERO}: {len(zero)} epochs, mean loss {zero[0]['mean_loss']:.5f} (epoch 1) → {zero[-1]['mean_loss']:.5f} "
        f"(epoch {zero[-1]['epoch']}).")
    for condition, e in encoders["encoders"].items():
        if condition != ZERO:
            add(f"- {condition}: {e['updates']} updates from {e['initialization']}, loss (mean of 100 updates) "
                f"{e['loss_first_100']:.5f} → {e['loss_last_100']:.5f}.")
    add("")
    add("## Scoring-policy coverage (phase-A review)")
    add("")
    add(f"G4 checks the scoring policy of the {coverage['rollout_records']} rollout records; "
        f"{coverage['probe_records_with_scoring_policy']} of the {coverage['probe_records']} identifiability probe "
        f"records carry one. Note: {coverage['note']}.")
    add("")
    c = compute["compute"]
    add(f"Compute: {c['gpu_seconds']:.0f} GPU s, {c['wall_seconds']:.0f} process wall s, 0 engine s.")
    add("")
    return "\n".join(lines)


def rollout_csv(compute):
    buffer = io.StringIO()
    import csv
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["cell", "condition", "request", "frames", "metric", "mean", "low", "high", "levels",
                     "contrast_mean", "contrast_low", "contrast_high"])
    for cell, block in compute["rollout"].items():
        for condition, rows in block.items():
            for request, by_t in rows.items():
                for t, entry in by_t.items():
                    for metric in ("masked_carrier_mse", "position_mse"):
                        b = entry[metric]
                        contrast = entry.get("novel_minus_normal") if metric == "masked_carrier_mse" else None
                        writer.writerow([cell, condition, request, t, metric,
                                         *(("", "", "") if b is None else (repr(b["mean"]), repr(b["interval"][0]),
                                                                          repr(b["interval"][1]))),
                                         entry["levels_finite"],
                                         *(("", "", "") if contrast is None else (repr(contrast["mean"]),
                                                                                 repr(contrast["interval"][0]),
                                                                                 repr(contrast["interval"][1])))])
    return buffer.getvalue()


def readiness_csv(compute):
    buffer = io.StringIO()
    import csv
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["cell", "status", "reasons", "flags", "evaluation_levels", "mixed_levels", "identifiability"])
    for r in compute["readiness"]:
        writer.writerow([r["cell"], r["status"], ";".join(r["reasons"]), ";".join(r["flags"]),
                         (r["headroom"] or {}).get("evaluation_levels", 0), (r["headroom"] or {}).get("mixed_levels", 0),
                         ";".join(f"{k}={v}" for k, v in r["identifiability"].items())])
    return buffer.getvalue()


def summary(plan, compute, coverage):
    return {"schema": "issue_110_results_summary_v1", "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "validation_command": VALIDATION_COMMAND, "guards": compute["guards"],
            "supported_cells": sum(r["status"] == "supported" for r in compute["readiness"]),
            "unsupported": [{"cell": r["cell"], "reasons": r["reasons"]} for r in compute["readiness"]
                            if r["status"] == "unsupported"],
            **({"insufficient": [{"cell": r["cell"], "reasons": r["reasons"]} for r in compute["readiness"]
                                 if r["status"] == STOP_TOKEN]} if not compute["guards_ok"] else {}),
            "identifiability": {k: v["verdict"] for k, v in compute["identifiability"].items()},
            "scoring_coverage": coverage,
            "claim_boundary": plan["claim_boundary"], "engine_seconds": 0}


def encoder_training(output):
    """Training summaries stored in the encoder checkpoints (no weights): the zero-shot E99-recipe epoch
    reports and each few-shot adaptation's first/last-100-update losses (phase-A review: a committed
    source for the encoder-loss numbers quoted in the #110 document)."""
    out = {}
    for condition in condition_levels():
        path = encoder_path(output, condition)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        keep = ("reports", "gpu_seconds", "training_frames", "training_shots", "parameters", "updates",
                "epochs_started", "loss_first_100", "loss_last_100", "initialization", "complete")
        out[condition] = {"checkpoint": checkpoint(path), **{k: payload[k] for k in keep if k in payload}}
    return {"schema": "issue_110_encoder_training_v1", "identity": IDENTITY, "encoders": out}


def rendered(output, plan, compute):
    coverage = scoring_coverage(output, compute)
    encoders = encoder_training(output)
    return {"compute.json": json_text(compute), "summary.json": json_text(summary(plan, compute, coverage)),
            "findings.md": findings_md(plan, compute, coverage, encoders), "rollout-error.csv": rollout_csv(compute),
            "readiness.csv": readiness_csv(compute), "handoff_107.json": json_text(handoff(output, plan, compute)),
            "records-manifest.json": json_text(record_manifest(output)),
            "encoder-training.json": json_text(encoders)}


def publish(output):
    output = Path(output)
    plan = load_plan(output)
    began = time.monotonic()
    with heavy().harness.deterministic_scoring():
        compute = compute_tables(output, plan)
        files = rendered(output, plan, compute)
    for name, text in files.items():
        (output / name).write_text(text)
    log(f"publish wall {time.monotonic() - began:.0f}s")
    log(f"published: {compute['guards']}; supported {sum(r['status'] == 'supported' for r in compute['readiness'])}/45")
    return 0


def validate(output):
    """Re-derive every published file from the retained records (byte for byte), check the record
    manifest, and re-score two rollout records in this process (the identifiability probes are not
    re-scored: their models were not retained, and re-running a probe retrains it)."""
    output = Path(output)
    plan = load_plan(output)
    stored = {name: (output / name).read_text() for name in ("records-manifest.json",)}
    if json.loads(stored["records-manifest.json"]) != record_manifest(output):
        log("record manifest differs")
        return 1
    h = heavy()
    with h.harness.deterministic_scoring() as policy:
        compute = compute_tables(output, plan)
        files = rendered(output, plan, compute)
        mismatched = [name for name, text in files.items() if (output / name).read_text() != text]
        tasks = [score_tasks()[0], score_tasks()[-1]]
        rescored = []
        for condition, level in tasks:
            members = {(f, s): load_member(output, condition, f, s) for f in FAMILIES for s in SEEDS}
            fresh = level_record(output, condition, level, members, policy)
            rescored.append(fresh == read_json(rollout_path(output, condition, level)))
    ok = not mismatched and all(rescored)
    log(f"validate: mismatched {mismatched}, rescored equal {rescored}")
    return 0 if ok else 1


def dry_run(output):
    rows = branches()
    log(f"branches {len(rows)}, complete {len(complete())}; zero images {len(zero_image_keys())}; "
        f"few images {[len(few_image_keys(k)) for k in LEVELS]}; score tasks {len(score_tasks())}; "
        f"probe tasks {len(probe_targets())}")
    return 0


MODES = {"dry-run": dry_run, "repair": repair, "smoke": smoke, "prepare": prepare, "labels": run_labels,
         "encoder": encoder, "fewshot-encoder": fewshot_encoder, "carriers": carriers, "scene": scene,
         "assemble": assemble, "train": train, "fewshot-train": fewshot_train, "score": score, "probe": probe,
         "publish": publish, "validate": validate}
SHARDED = {"fewshot-encoder", "carriers", "train", "fewshot-train", "score", "probe"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        group.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker", type=int, default=0)
    args = parser.parse_args()
    mode = next(m for m in MODES if getattr(args, m.replace("-", "_")))
    log(f"mode {mode} pid {os.getpid()}" + (f" worker {args.worker}/{args.workers}" if mode in SHARDED else ""))
    if mode in SHARDED:
        return MODES[mode](OUTPUT, args.workers, args.worker)
    return MODES[mode](OUTPUT)


if __name__ == "__main__":
    raise SystemExit(main())
