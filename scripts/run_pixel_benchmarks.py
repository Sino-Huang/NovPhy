#!/usr/bin/env python
"""Issue #106: description levels on CLEVRER and Physion-Dominoes, from pixels.

Modes (one process per GPU for the sharded ones: ``--workers 4 --worker k`` with CUDA_VISIBLE_DEVICES=k):
  manifest  verify the downloads (sizes, S3 multipart ETags), sha256 every archive, extract, write the
            data manifests (data/<bench>/manifest.json, tracked copy <output>/data-manifest.json)
  smoke     pre-freeze: CLEVRER camera homography, label base rates, a tiny end-to-end pass, projection
  freeze    write plan.json (definitions, membership, recipes, arms, dispositions, guards, input hashes)
  prepare   annotation-side caches (parser targets, description labels, annotation carriers, encoder frames)
  encoder   train the E99 spatial slot parser per benchmark (worker 0: CLEVRER, worker 1: Physion)
  parse     pixels -> parser -> 236-value carriers for every member clip (sharded)
  train     predictor ensembles (worker k: one benchmark x family, five seeds each)
  policy    J (nine requests) and J-cont (horizon-only, continuous requests) per predictor member (sharded)
  score     deterministic scoring: rollout errors, CLEVRER QA, Physion OCP (sharded by member)
  publish   tables, contrasts, dispositions, guards, compute, findings, records manifest
  validate  regenerate the published files byte for byte and spot re-score (also ``--validate``)
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.data import pixel_benchmarks as pb  # noqa: E402
from world_model.model import Abstraction  # noqa: E402
from world_model.planning import scoring_harness as harness  # noqa: E402
from world_model.training import pixel_carrier as pc  # noqa: E402
from world_model.training import request_policy as rp  # noqa: E402
from world_model.training import request_selectors as rs  # noqa: E402
from world_model.training import decision_dynamics as dd  # noqa: E402
from world_model.training import stable_decision_dynamics as sd  # noqa: E402
from world_model.training import predicate_feedback as pf  # noqa: E402
from world_model.training.cnn_hybrid import DIM, PAIRS, SLOTS  # noqa: E402

OUTPUT = ROOT / ".local-artifacts/issue-106-pixel-benchmarks-v1"
IDENTITY = "issue-106-pixel-benchmarks-v1"
VALIDATION_COMMAND = "python -u -m scripts.run_pixel_benchmarks --validate"
SCHEMA = {k: f"issue_106_{k}_v1" for k in ("plan", "smoke", "record", "report", "compute", "ledger", "summary",
                                            "manifest", "data_manifest", "fidelity")}
BENCHES = ("clevrer", "physion")
FAMILIES = ("hybrid", "label-free")
SEEDS = (20260908, 20260909, 20260910, 20260911, 20260912)
REQUESTS = tuple(f"F-{p.delta}-{Abstraction(p.abstraction).value}" for p in PAIRS)
R_CONT = tuple(i for i, p in enumerate(PAIRS) if p.abstraction == Abstraction.CONTINUOUS)
ARMS = REQUESTS + ("J", "J-cont")
TRAINING = {"steps": 9000, "batch_size": 64, "learning_rate": 1e-4, "weight_decay": 1e-4, "grad_clip": 1.0,
            "patience": 10, "cycle": "PAIRS[step % 9]", "window_length": 61, "sampler_offset": 106001,
            "long_offset": 106002}
LONG = {"clevrer": {"first": 30, "last": 120, "increment": 15, "every": 900, "batch_size": 64, "truncation": 45,
                    "weight": 1.0, "reserve": 120},
        "physion": {"first": 30, "last": 90, "increment": 15, "every": 900, "batch_size": 64, "truncation": 45,
                    "weight": 1.0, "reserve": 30}}
POLICY = {"rounds": rp.DISTIL_RECIPE["rounds"], "steps_per_round": rp.DISTIL_RECIPE["steps_per_round"],
          "batch_size": rp.DISTIL_RECIPE["batch_size"], "learning_rate": rp.DISTIL_RECIPE["learning_rate"],
          "compute_weight": rp.ICLR_COMPUTE_WEIGHT, "windows": 1024, "units": 8, "window_length": 60,
          "window_offset": 106003}
ENDPOINTS = {"clevrer": (15, 30, 60, 120), "physion": (15, 30, 45, 90)}
CONTEXTS = {"clevrer": (0, 1, 2, 3), "physion": (12, 13, 14, 15)}
QA = {"predictive_start": 127, "predictive_horizon": 120, "threshold": 0.5,
      "counterfactual": "start one frame before the removed object's first parsed contact (micro head on the "
                        "parsed carrier >= 0.5), removed slot zeroed (presence, center, motion columns), rollout to "
                        "frame 127; observed readings up to the start frame count; no parsed contact -> factual"}
OCP = {"input_frames": pb.PHYSION_INPUT + 1, "start": pb.PHYSION_INPUT, "endpoint": pb.PHYSION_ROLLOUT,
       "grid_every": 5, "C_logspace": (-8, 8, 17), "folds": 5, "max_iter": 200, "balance": "undersample (train)",
       "protocols": ("simulated", "observed", "input"), "predicate_threshold": 0.5}
ENCODER_FRAMES = {"clevrer": {"every": 8}, "physion": {"every": 2}}
MARGIN = 0.02
BOOTSTRAP = {"draws": 10000, "seed": 7201, "quantiles": (0.025, 0.975), "clusters": "seeds x items (independent)"}
SPOT = {"clips": 16}
MODULES = ("world_model/data/pixel_benchmarks.py", "world_model/training/pixel_carrier.py",
           "world_model/training/spatial_slot_parser.py", "world_model/training/cohort_v2_visual_parser.py",
           "world_model/training/novelty_v2.py", "world_model/training/cnn_hybrid.py",
           "world_model/training/predicate_feedback.py", "world_model/training/request_policy.py",
           "world_model/training/request_selectors.py", "world_model/training/decision_dynamics.py",
           "world_model/training/stable_decision_dynamics.py", "world_model/training/matched_dynamics.py",
           "world_model/planning/scoring_harness.py", "scripts/run_pixel_benchmarks.py")
DEVICE = "cuda"
DOWNLOAD_PART = {"Dominoes_dynamics_training_HDF5s.tar.gz": None}


# =========================================================================== utilities


def spawn_pool(workers):
    """Process pool with the spawn start method (h5py and CUDA state must not be forked)."""
    import multiprocessing
    return ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))


def log(message):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_of(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def dumps(payload):
    return json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(dumps(payload))
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_torch(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def mark(output, phase, wall, gpu):
    """Append one phase to the ledger (wall and GPU seconds per worker)."""
    path = Path(output) / "ledger.json"
    with open(Path(output) / "ledger.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = read_json(path) if path.is_file() else {"schema": SCHEMA["ledger"], "phases": []}
        ledger["phases"].append({"phase": phase, "wall_seconds": round(float(wall), 3),
                                 "gpu_seconds": round(float(gpu), 3), "pid": os.getpid(), "at": utc_now()})
        write_json(path, ledger)


def cache(output, *parts):
    return Path(output).joinpath("cache", *parts)


def member_tag(bench, family, seed):
    return f"{bench}--{family}--seed{seed}"


def model_path(output, bench, family, seed):
    return Path(output) / "models" / bench / family / f"seed-{seed}" / "predictor.pt"


def policy_path(output, bench, family, seed, kind):
    return Path(output) / "models" / bench / family / f"seed-{seed}" / f"policy-{kind}.pt"


def encoder_path(output, bench):
    return Path(output) / "models" / bench / "encoder" / "parser.pt"


def record_path(output, bench, family, seed):
    return Path(output) / "records" / f"{member_tag(bench, family, seed)}.json"


# =========================================================================== manifest


def s3_part_size(size, parts):
    for mib in (8, 16, 32, 64, 5, 10, 15, 25, 50, 100):
        part = mib << 20
        if math.ceil(size / part) == parts:
            return part
    raise ValueError(f"no part size gives {parts} parts for {size} bytes")


def multipart_etag(path, size, etag):
    import hashlib as h
    digest, parts = etag.split("-")
    part = s3_part_size(size, int(parts))
    md5s = []
    with open(path, "rb") as handle:
        while block := handle.read(part):
            md5s.append(h.md5(block).digest())
    return h.md5(b"".join(md5s)).hexdigest() + f"-{len(md5s)}" == etag


def _file_facts(job):
    path, expected, etag = job
    facts = {"bytes": os.path.getsize(path), "sha256": sha256_of(path)}
    facts["bytes_match"] = facts["bytes"] == expected
    if etag:
        facts["s3_etag_match"] = (multipart_etag(path, facts["bytes"], etag) if "-" in etag
                                  else hashlib.md5(Path(path).read_bytes()).hexdigest() == etag)
    return facts


def manifest(output):
    """Post-download data manifests; extraction of the Physion archives."""
    began = time.monotonic()
    out = {}
    for bench, root in (("clevrer", pb.CLEVRER), ("physion-dominoes", pb.PHYSION)):
        pre = read_json(root / "manifest.pre-download.json")
        jobs = []
        for entry in pre["files"]:
            path = root / entry["name"]
            if not path.is_file():
                raise SystemExit(f"missing download {path}")
            jobs.append((str(path), entry["bytes"], entry.get("s3_etag")))
        with spawn_pool(len(jobs)) as pool:
            facts = list(pool.map(_file_facts, jobs))
        files = []
        for entry, fact in zip(pre["files"], facts):
            row = dict(entry)
            row.update(fact)
            if not fact["bytes_match"] or fact.get("s3_etag_match") is False:
                raise SystemExit(f"download check failed: {entry['name']} {fact}")
            files.append(row)
        record = {"schema": SCHEMA["data_manifest"], "benchmark": pre["benchmark"], "provenance": pre["provenance"],
                  "approved_scope": pre["approved_scope"], "pre_download_recorded_at": pre["recorded_at"],
                  "verified_at": utc_now(), "files": files, "total_bytes": sum(f["bytes"] for f in files)}
        for key in ("out_of_scope", "not_downloaded"):
            if key in pre:
                record[key] = pre[key]
        if bench == "physion-dominoes":
            extracted, running = {}, {}
            for split, stem in pb.PHYSION_SPLITS.items():
                target = pb.PHYSION_EXTRACTED / split
                if not (target / ".extracted").is_file():
                    target.mkdir(parents=True, exist_ok=True)
                    running[split] = subprocess.Popen(["tar", "-xzf", str(root / f"{stem}.tar.gz"), "-C", str(target)])
            for split, process in running.items():
                if process.wait() != 0:
                    raise SystemExit(f"extraction of {split} failed")
                (pb.PHYSION_EXTRACTED / split / ".extracted").write_text(utc_now())
            for split in pb.PHYSION_SPLITS:
                target = pb.PHYSION_EXTRACTED / split
                files_ = sorted(target.rglob("*.hdf5"))
                extracted[split] = {"directory": str(target.relative_to(pb.ROOT)), "hdf5_files": len(files_),
                                    "bytes": sum(f.stat().st_size for f in files_)}
            record["extracted"] = extracted
        else:
            videos = {}
            for split in ("train", "validation"):
                low, high = (0, 10000) if split == "train" else (10000, 15000)
                found = [p for p in pb.CLEVRER_VIDEOS.rglob("*.mp4") if low <= int(p.stem.split("_")[1]) < high]
                videos[split] = {"directory": str(pb.CLEVRER_VIDEOS.relative_to(pb.ROOT)), "mp4_files": len(found),
                                 "bytes": sum(p.stat().st_size for p in found)}
            record["extracted"] = videos
        write_json(root / "manifest.json", record)
        out[bench] = record
    write_json(Path(output) / "data-manifest.json", {"schema": SCHEMA["data_manifest"], "benchmarks": out})
    mark(output, "manifest", time.monotonic() - began, 0)
    log("data manifests written")
    return 0


# =========================================================================== membership and labels


def clevrer_membership(split):
    members, skipped = [], 0
    for index, payload in pb.clevrer_annotations(split):
        if pb.clevrer_member(payload):
            members.append(index)
        else:
            skipped += 1
    return members, skipped


PHYSION_PARAMETERS = ("initial_position", "initial_rotation", "scale", "color", "mass", "dynamic_friction",
                      "static_friction", "bounciness", "push_force", "push_position", "push_time", "model_names")


def _physion_header(path):
    try:
        header = pb.physion_header(path)
        import h5py
        with h5py.File(path, "r") as h:
            static = h["static"]
            pb.physion_roles(static)
            def digest(keys):
                return hashlib.sha256(b"".join(np.ascontiguousarray(static[k][()]).tobytes()
                                               for k in keys if k in static)).hexdigest()
            header["layout_sha256"] = digest(("initial_position", "initial_rotation", "scale"))
            header["parameters_sha256"] = digest(PHYSION_PARAMETERS)
        header["error"] = None
    except Exception as error:  # typed: unreadable or out-of-contract trial
        header = {"name": Path(path).stem, "error": f"{type(error).__name__}: {error}"}
    header["path"] = str(Path(path).relative_to(pb.ROOT))
    return header


def physion_membership(split, glitchy):
    with spawn_pool(46) as pool:
        headers = list(pool.map(_physion_header, pb.physion_files(split), chunksize=8))
    members, excluded = [], {}
    for h in headers:
        reason = None
        if h["error"]:
            reason = "unreadable_or_contract"
        elif h["native_frames"] < pb.PHYSION_MIN_NATIVE:
            reason = "short"
        elif h["objects"] > SLOTS:
            reason = "slots"
        elif split == "testing" and (h["name"] in glitchy or h["stimulus"] in glitchy):
            reason = "glitchy"
        if reason:
            excluded.setdefault(reason, []).append(h["name"])
        else:
            members.append(h)
    return members, excluded


def clevrer_labels(job):
    index, payload, homography = job
    scene = pb.clevrer_scene(payload)
    targets = pb.clevrer_targets(scene, homography)
    contact, mask = pb.clevrer_micro(scene)
    macros, available = pb.clevrer_macro(scene)
    annotation = pc.annotation_carrier(targets, pc.KINDS["clevrer"], pb.CLEVRER_ELAPSED)
    return {"index": index, "presence": targets["presence"], "centers": targets["centers"].astype(np.float32),
            "center_valid": targets["center_valid"], "kinds": targets["kinds"].astype(np.int8),
            "contact": np.argwhere(contact).astype(np.int16), "declared": mask.any(0),
            "macros": macros, "macro_available": available, "annotation": annotation.numpy()}


def physion_labels(path):
    trial = pb.physion_trial(pb.ROOT / path, images=False, segmentation=True)
    frames = len(trial["carrier_native"]) + pb.PHYSION_ROLLOUT + TRAINING["window_length"]
    contact, mask = pb.physion_micro(trial, frames)
    macros, available = pb.physion_macro(trial, frames)
    targets = {k: trial[k] for k in ("presence", "centers", "kinds", "center_valid")}
    annotation = pc.annotation_carrier(targets, pc.KINDS["physion"], pb.PHYSION_ELAPSED)
    return {"name": trial["name"], "native_frames": trial["native_frames"], "frames": len(trial["carrier_native"]),
            "presence": targets["presence"], "centers": targets["centers"], "center_valid": targets["center_valid"],
            "kinds": targets["kinds"].astype(np.int8), "contact": np.argwhere(contact).astype(np.int16),
            "declared": mask.any(0), "macros": macros, "macro_available": available, "ocp": pb.physion_ocp_label(trial),
            "annotation": annotation.numpy()}


# =========================================================================== plan


def module_hashes():
    return {m: sha256_of(ROOT / m) for m in MODULES}


def accepted_hashes(output, plan):
    """Frozen module hashes plus the disclosed post-freeze amendments chained onto them
    (<output>/amendments.json: per amendment, module -> {replaces, sha256})."""
    accepted = {m: {h} for m, h in plan["inputs"]["modules"].items()}
    path = Path(output) / "amendments.json"
    if path.is_file():
        for amendment in read_json(path)["amendments"]:
            for module, change in amendment["modules"].items():
                if change["replaces"] in accepted.get(module, set()):
                    accepted[module].add(change["sha256"])
    return accepted


def load_plan(output, *, check=True):
    plan = read_json(Path(output) / "plan.json")
    if check:
        accepted = accepted_hashes(output, plan)
        drift = {m: h for m, h in module_hashes().items() if h not in accepted.get(m, set())}
        if drift:
            raise SystemExit(f"input drift after freeze: {sorted(drift)}")
        if sha256_of(Path(output) / "smoke.json") != plan["inputs"]["smoke_sha256"]:
            raise SystemExit("smoke.json drift after freeze")
    return plan


def definitions():
    return {
        "carrier": {"layout": "236 = [has prior, elapsed] + 18 slots x 13 (presence, 1, kind index/(K-1), kind "
                              "confidence, 1, center x, center y, center available, motion x, motion y, motion "
                              "available, 1, 1); presence threshold 0.5; motion = center difference / elapsed",
                    "clevrer": {"frames": pb.CLEVRER_FRAMES, "image": pb.CLEVRER_IMAGE, "elapsed": pb.CLEVRER_ELAPSED,
                                "slots": "COLORS.index(color) * 2 + MATERIALS.index(material); 16 keyed, 2 empty",
                                "kinds": pb.SHAPES, "colors": pb.COLORS, "materials": pb.MATERIALS},
                    "physion": {"stride_native": pb.PHYSION_STRIDE, "image": pb.PHYSION_IMAGE,
                                "elapsed": pb.PHYSION_ELAPSED, "slots": "static/object_ids order", "kinds": pb.ROLES,
                                "input_carrier_frames": [0, pb.PHYSION_INPUT], "rollout": pb.PHYSION_ROLLOUT,
                                "absorption": "frames past a trial's end repeat its last frame"}},
        "parser_targets": {
            "clevrer": "presence = inside_camera_view; center = H(x, y) / (480, 320) with the frozen homography H "
                       "(valid if in view and inside the image); kind = shape",
            "physion": f"presence = >= {pb.PHYSION_MIN_PIXELS} pixels of the object's segmentation colour; center = "
                       "mask centroid / 256; kind = role"},
        "micro": {"channel_0": {"clevrer": "annotated collision of the pair at frame f (#79)",
                                "physion": "pair contact (collision state enter/stay) on any native frame in "
                                           "(3f - 3, 3f], absorbed past the end"},
                  "channel_1": "masked (unused)", "mask": "both slots declared", "delta_dependence": "none"},
        "macro": {"clevrer": {"enter-exit": f"debounced in-view set (>= {pb.CLEVRER_VIEW_MAJORITY} of the last "
                                            f"{pb.CLEVRER_VIEW_WINDOW} frames) differs between f - Delta and f; "
                                            "available iff f - Delta >= 2",
                              "all-at-rest": f"D = max(1, Delta): every frame of (f - D, f] has >= 1 object in view "
                                             f"and every in-view object at planar speed <= {pb.CLEVRER_REST_SPEED}; "
                                             "available iff f >= D"},
                  "physion": {"ocp-contact": "C(g) = target touched zone on a native frame <= 3g; label C(f) != "
                                             "C(f - Delta); available iff f >= Delta; readout label = C(end)",
                              "settled": f"D = max(1, Delta): push at or before native 3(f - D) and every object "
                                         f"|v| <= {pb.PHYSION_REST_SPEED}, |omega| <= {pb.PHYSION_REST_ANGULAR} on "
                                         "every native frame of (3(f - D), 3f]; available iff f >= D"},
                  "template": "#113 observable predicates indexed by the landing frame; settle with debounce "
                              "D = max(1, Delta); structure change by endpoint comparison per Delta",
                  "head": "predicate_feedback.DeltaMacroPredictor (4 outputs per Delta)"},
    }


def dispositions():
    return {
        "margin": MARGIN,
        "rule": "#114 rule: supported iff mean >= margin and lower > 0; not_supported_by_this_experiment iff "
                "upper < margin; otherwise readiness_or_precision_insufficient",
        "primary": {"claim": "request-selected rollouts with description levels (J) beat the horizon-only "
                             "baseline (J-cont) on the benchmark's own decision at equal training compute",
                    "contrasts": {"clevrer": "J - J-cont, CLEVRER per-option accuracy (predictive + counterfactual)",
                                  "physion": "J - J-cont, Physion OCP test accuracy (simulated readout protocol)"},
                    "family": "hybrid"},
        "secondary": {"J_minus_fixed": "J - F-k for each of the nine fixed requests, same rule",
                      "label_free": "J(label-free) - J(hybrid) on Physion OCP and both rollout errors "
                                    "(non-inferiority reading: not_supported iff upper < -margin is reported)",
                      "per_type": "CLEVRER predictive and counterfactual separately"},
        "descriptive": ["rollout-error table (all nine requests + J + J-cont, both benchmarks, pixel carrier)",
                        "accuracy vs MAC table", "parser fidelity", "OCP observed / input protocol anchors",
                        "OCP predicate rule"],
        "guard_failure": "a failed guard turns every token into readiness_or_precision_insufficient",
    }


def guards_declared():
    return {"G1": "split isolation: encoder / predictor / policy pools contain no scored clip (CLEVRER scene "
                  "indices disjoint; Physion: no full static parameter set (PHYSION_PARAMETERS) shared across "
                  "dynamics / readout / testing; shared file names and fixed generator layouts are reported)",
            "G2": "every scoring record carries the deterministic scoring policy",
            "G3": "chronology: plan frozen before the first encoder / predictor update, records after training",
            "G4": "completeness: every planned member trained or a recorded typed failure; every record finite",
            "G5": "pixels only at test time: spot re-parse of scored clips from pixels equals the cached carrier "
                  "(max abs difference <= 1e-4) and a spot re-score equals the record",
            "G6": "data manifest: every archive sha256 / size / S3 ETag matches"}


def freeze(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise SystemExit("plan already frozen")
    smoke_ = read_json(output / "smoke.json")
    if not smoke_.get("passed"):
        raise SystemExit("smoke did not pass")
    data_manifest = read_json(output / "data-manifest.json")
    log("membership: CLEVRER")
    clevrer = {split: clevrer_membership(split) for split in ("train", "validation")}
    glitchy = pb.physion_glitchy()
    log("membership: Physion")
    physion = {split: physion_membership(split, glitchy) for split in pb.PHYSION_SPLITS}
    digests = {split: {kind: {h[f"{kind}_sha256"] for h in physion[split][0]} for kind in ("layout", "parameters")}
               for split in physion}
    names = {split: {h["name"] for h in physion[split][0]} for split in physion}
    overlap = {f"{a}|{b}": {"names": len(names[a] & names[b]),
                            "layouts": len(digests[a]["layout"] & digests[b]["layout"]),
                            "parameter_sets": len(digests[a]["parameters"] & digests[b]["parameters"])}
               for a in physion for b in physion if a < b}
    plan = {
        "schema": SCHEMA["plan"], "identity": IDENTITY, "issue": 106, "parent": 108, "frozen_at": utc_now(),
        "validation_command": VALIDATION_COMMAND,
        "owner_decisions": {"encoder": "#99 spatial slot parser (E99), not JEPA (owner comment on #106)",
                            "macro": "#113 observable definitions, not one-step events (owner comment on #106)",
                            "download_scope": "CLEVRER videos + QA (train, validation); Physion Dominoes dynamics, "
                                              "readout and testing sets; other Physion scenarios out of scope"},
        "definitions": definitions(),
        "homography": smoke_["homography"],
        "membership": {
            "clevrer": {split: {"scenes": m, "skipped_repeated_key_or_contract": s, "count": len(m)}
                        for split, (m, s) in clevrer.items()},
            "physion": {split: {"trials": [h["path"] for h in m], "count": len(m),
                                "excluded": {k: len(v) for k, v in ex.items()}, "excluded_names": ex}
                        for split, (m, ex) in physion.items()},
            "physion_split_overlap": overlap,
            "roles": {"clevrer": {"train": "encoder, predictor, policy", "validation": "rollout error, QA"},
                      "physion": {"dynamics": "encoder, predictor, policy", "readout": "OCP readout fit",
                                  "testing": "rollout error, OCP test"}}},
        "encoder": {"architecture": "SpatialSlotParser (issue-99-spatial-slot-parser-v1), unchanged",
                    "recipe": pc.ENCODER, "frames": ENCODER_FRAMES, "kinds": pc.KINDS,
                    "loss": "E99/#70 parser_loss terms: presence x4, class-balanced presence x4, count MSE, "
                            "smooth-L1 centers, kind CE; relation/macro parser heads untrained"},
        "predictor": {"families": {"hybrid": "predicate_feedback.DeltaMacroPredictor, labels on micro and Delta macro",
                                   "label-free": "pixel_carrier.LabelFreeMicroPredictor: #115 binary micro "
                                                 "bottleneck without micro labels, Delta macro labelled"},
                      "seeds": SEEDS, "training": TRAINING, "long": LONG,
                      "recipe": "#112 LC as #117 item 5 adapted it: #77 local loss (pf.local_loss) + long-horizon "
                                "recursive loss with curriculum, truncation 45, skip-on-nonfinite (patience 10); "
                                "ranking term dropped (no verdicted inventories)"},
        "policy": {"recipe": POLICY, "J": "request_policy.RequestPolicy(PAIRS) via window_rows (#77 DP labels, "
                                          "DAgger rounds) and distil_round",
                   "J-cont": "request_selectors.new_masked_policy(PAIRS, R_cont) + restricted_window_rows (#114)",
                   "R_cont": R_CONT},
        "scoring": {"arms": ARMS, "endpoints": ENDPOINTS, "contexts": CONTEXTS,
                    "rollout_error": "position MSE (columns 5, 6 under the target's column-7 mask) against the "
                                     "parsed carrier (primary) and the annotation carrier (secondary), per clip mean "
                                     "over contexts",
                    "qa": QA, "ocp": OCP, "bootstrap": BOOTSTRAP, "spot": SPOT,
                    "equal_compute": "every arm of a member uses the same predictor; J and J-cont share the policy "
                                     "recipe; inference MACs are the table's x axis"},
        "dispositions": dispositions(), "guards": guards_declared(),
        "inputs": {"modules": module_hashes(), "smoke_sha256": sha256_of(output / "smoke.json"),
                   "data_manifest_sha256": sha256_of(output / "data-manifest.json"),
                   "archives": {f["name"]: f["sha256"] for b in data_manifest["benchmarks"].values()
                                for f in b["files"]}},
        "hardware": "4 x RTX 5090 (32 GB) instead of the #85 single RTX 3090",
    }
    write_json(output / "plan.json", plan)
    log(f"frozen: CLEVRER train {len(clevrer['train'][0])} / validation {len(clevrer['validation'][0])}; Physion "
        + ", ".join(f"{s} {len(m[0])}" for s, m in physion.items()))
    return 0


# =========================================================================== prepare


def prepare(output, workers, worker):
    output = Path(output)
    plan = load_plan(output)
    began = time.monotonic()
    homography = np.array(plan["homography"]["matrix"], np.float64)
    if worker == 0:
        for split in ("train", "validation"):
            target = cache(output, "clevrer", f"labels-{split}.pt")
            if target.is_file():
                continue
            members = set(plan["membership"]["clevrer"][split]["scenes"])
            jobs = [(i, p, homography) for i, p in pb.clevrer_annotations(split) if i in members]
            with spawn_pool(22) as pool:
                rows = list(pool.map(clevrer_labels, jobs, chunksize=16))
            atomic_torch(target, rows)
            log(f"CLEVRER {split}: {len(rows)} label rows")
        clevrer_encoder_frames(output, plan)
    if worker == 1 or workers == 1:
        for split in pb.PHYSION_SPLITS:
            target = cache(output, "physion", f"labels-{split}.pt")
            if target.is_file():
                continue
            with spawn_pool(22) as pool:
                rows = list(pool.map(physion_labels, plan["membership"]["physion"][split]["trials"], chunksize=4))
            atomic_torch(target, rows)
            log(f"Physion {split}: {len(rows)} label rows")
        physion_encoder_frames(output, plan)
    mark(output, f"prepare-w{worker}", time.monotonic() - began, 0)
    return 0


def _clevrer_sampled(job):
    index, every = job
    return pb.decode_clevrer_video(index)[::every]


def clevrer_encoder_frames(output, plan):
    target = cache(output, "clevrer", "encoder-frames.u8")
    if target.with_suffix(".json").is_file():
        return
    rows = load_torch(cache(output, "clevrer", "labels-train.pt"))
    every = ENCODER_FRAMES["clevrer"]["every"]
    per = len(range(0, pb.CLEVRER_FRAMES, every))
    h, w = pb.CLEVRER_IMAGE
    frames = np.lib.format.open_memmap(target.with_suffix(".npy"), "w+", np.uint8, (len(rows) * per, h, w, 3))
    with spawn_pool(44) as pool:
        for k, block in enumerate(pool.map(_clevrer_sampled, [(r["index"], every) for r in rows], chunksize=8)):
            frames[k * per:(k + 1) * per] = block
    frames.flush()
    meta = {"clips": [r["index"] for r in rows], "frames": list(range(0, pb.CLEVRER_FRAMES, every)),
            "presence": np.stack([r["presence"][::every] for r in rows]).reshape(-1, SLOTS).tolist()}
    targets = {"presence": np.concatenate([r["presence"][::every] for r in rows]),
               "centers": np.concatenate([r["centers"][::every] for r in rows]),
               "center_valid": np.concatenate([r["center_valid"][::every] for r in rows]),
               "kinds": np.concatenate([r["kinds"][::every] for r in rows])}
    np.savez(cache(output, "clevrer", "encoder-targets.npz"), **targets)
    write_json(target.with_suffix(".json"), {"clips": meta["clips"], "frames": meta["frames"],
                                              "count": int(len(rows) * per)})
    log(f"CLEVRER encoder frames: {len(rows) * per}")


def _physion_sampled(job):
    path, every = job
    trial = pb.physion_trial(pb.ROOT / path, images=True, segmentation=False)
    return trial["images"][::every]


def physion_encoder_frames(output, plan):
    target = cache(output, "physion", "encoder-frames.u8")
    if target.with_suffix(".json").is_file():
        return
    rows = load_torch(cache(output, "physion", "labels-dynamics.pt"))
    paths = plan["membership"]["physion"]["dynamics"]["trials"]
    every = ENCODER_FRAMES["physion"]["every"]
    counts = [len(range(0, r["frames"], every)) for r in rows]
    h, w = pb.PHYSION_IMAGE
    frames = np.lib.format.open_memmap(target.with_suffix(".npy"), "w+", np.uint8, (sum(counts), h, w, 3))
    offset = 0
    with spawn_pool(44) as pool:
        for block in pool.map(_physion_sampled, [(p, every) for p in paths], chunksize=2):
            frames[offset:offset + len(block)] = block
            offset += len(block)
    frames.flush()
    targets = {"presence": np.concatenate([r["presence"][::every] for r in rows]),
               "centers": np.concatenate([r["centers"][::every] for r in rows]),
               "center_valid": np.concatenate([r["center_valid"][::every] for r in rows]),
               "kinds": np.concatenate([r["kinds"][::every] for r in rows])}
    np.savez(cache(output, "physion", "encoder-targets.npz"), **targets)
    write_json(target.with_suffix(".json"), {"trials": paths, "count": int(sum(counts))})
    log(f"Physion encoder frames: {sum(counts)}")


# =========================================================================== encoder


def image_size(bench):
    return pb.CLEVRER_IMAGE if bench == "clevrer" else pb.PHYSION_IMAGE


def train_encoder(output, bench, *, steps_limit=None, frames=None, targets=None):
    """E99 recipe: AdamW, 12 epochs, batch 64, lr 1e-3, wd 1e-4, seed 9900001, sampler seeded."""
    torch.manual_seed(pc.ENCODER["seed"])
    model = pc.make_parser(bench, image_size(bench)).to(DEVICE)
    if frames is None:
        frames = np.load(cache(output, bench, "encoder-frames.u8").with_suffix(".npy"), mmap_mode="r")
        targets = dict(np.load(cache(output, bench, "encoder-targets.npz")))
    count = len(frames)
    weights = pc.presence_weights(targets["presence"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=pc.ENCODER["learning_rate"],
                                  weight_decay=pc.ENCODER["weight_decay"])
    generator = torch.Generator().manual_seed(pc.ENCODER["seed"])
    batch = pc.ENCODER["batch_size"]
    losses, step = [], 0
    began = time.monotonic()
    model.train()
    for epoch in range(pc.ENCODER["epochs"]):
        order = torch.randperm(count, generator=generator).numpy()
        for start in range(0, count - batch + 1, batch):
            index = np.sort(order[start:start + batch])
            sample = {"images": torch.from_numpy(np.ascontiguousarray(frames[index])).to(DEVICE).permute(0, 3, 1, 2),
                      "presence": torch.from_numpy(targets["presence"][index]).to(DEVICE),
                      "centers": torch.from_numpy(targets["centers"][index]).to(DEVICE).float(),
                      "center_valid": torch.from_numpy(targets["center_valid"][index]).to(DEVICE),
                      "kinds": torch.from_numpy(targets["kinds"][index].astype(np.int64)).to(DEVICE)}
            loss = pc.parser_loss(model, sample, weights, pc.KINDS[bench])
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite parser loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            step += 1
            if steps_limit and step >= steps_limit:
                break
        log(f"{bench} encoder epoch {epoch}: loss {np.mean(losses[-200:]):.4f} ({time.monotonic() - began:.0f}s)")
        if steps_limit and step >= steps_limit:
            break
    model.eval()
    return model, {"steps": step, "final_loss": float(np.mean(losses[-200:])), "frames": int(count),
                   "wall_seconds": time.monotonic() - began,
                   "parameters": sum(p.numel() for p in model.parameters())}


def encoder(output, workers, worker):
    output = Path(output)
    load_plan(output)
    bench = BENCHES[worker]
    if encoder_path(output, bench).is_file():
        log(f"{bench} encoder: done")
        return 0
    torch.backends.cudnn.benchmark = True
    started_at = utc_now()
    model, report = train_encoder(output, bench)
    atomic_torch(encoder_path(output, bench), {"schema": SCHEMA["report"], "bench": bench,
                                               "model": model.state_dict(), "report": report})
    write_json(encoder_path(output, bench).with_name("report.json"),
               {"schema": SCHEMA["report"], "bench": bench, "report": report, "started_at": started_at,
                "trained_at": utc_now(),
                "sha256": sha256_of(encoder_path(output, bench))})
    mark(output, f"encoder-{bench}", report["wall_seconds"], report["wall_seconds"])
    return 0


def load_encoder(output, bench):
    model = pc.make_parser(bench, image_size(bench)).to(DEVICE)
    model.load_state_dict(load_torch(encoder_path(output, bench))["model"])
    return model.eval()


# =========================================================================== parse


def clips_of(plan, bench):
    """[(split, clip id)] in plan order; clip id = scene index (CLEVRER) or trial path (Physion)."""
    if bench == "clevrer":
        return [(s, i) for s in ("train", "validation") for i in plan["membership"]["clevrer"][s]["scenes"]]
    return [(s, p) for s in pb.PHYSION_SPLITS for p in plan["membership"]["physion"][s]["trials"]]


def _physion_images(path):
    return pb.physion_trial(pb.ROOT / path, images=True, segmentation=False)["images"]


def parse_clip(model, bench, frames):
    """uint8 frames [T, H, W, 3] -> carrier [T, 236], presence [T, 18], kinds [T, 18, K] (test-time path)."""
    images = torch.from_numpy(np.ascontiguousarray(frames)).to(DEVICE).permute(0, 3, 1, 2)
    parsed = {k: [] for k in ("presence", "centers", "kinds")}
    for start in range(0, len(images), 256):
        out = pc.parse(model, images[start:start + 256], pc.KINDS[bench])
        for k in parsed:
            parsed[k].append(out[k])
    parsed = {k: torch.cat(v) for k, v in parsed.items()}
    elapsed = pb.CLEVRER_ELAPSED if bench == "clevrer" else pb.PHYSION_ELAPSED
    carrier = pc.carrier_values(parsed["presence"], parsed["centers"], parsed["kinds"], elapsed)
    return carrier.cpu(), parsed["presence"].half().cpu(), parsed["kinds"].half().cpu(), parsed["centers"].cpu()


def fidelity(presence, centers, kinds, labels, native_size):
    """Parser vs annotation targets on one clip: presence TP/FP/FN, center pixel error, kind accuracy."""
    truth = torch.as_tensor(labels["presence"])
    predicted = presence.float() >= pc.PRESENCE_THRESHOLD
    valid = torch.as_tensor(labels["center_valid"]) & predicted
    scale = torch.tensor(native_size[::-1], dtype=torch.float64)
    error = ((centers.double() - torch.as_tensor(labels["centers"]).double()) * scale).norm(dim=-1)
    present = truth & predicted
    kind_ok = kinds.float().argmax(-1) == torch.as_tensor(labels["kinds"]).long()
    return {"tp": int((truth & predicted).sum()), "fp": int((~truth & predicted).sum()),
            "fn": int((truth & ~predicted).sum()), "center_error_sum": float(error[valid].sum()),
            "center_count": int(valid.sum()), "kind_correct": int((kind_ok & present).sum()),
            "kind_count": int(present.sum())}


def parse_stage(output, workers, worker):
    output = Path(output)
    plan = load_plan(output)
    began = time.monotonic()
    for bench in BENCHES:
        target = cache(output, bench, f"parsed-w{worker}.pt")
        if target.is_file():
            continue
        model = load_encoder(output, bench)
        clips = [c for k, c in enumerate(clips_of(plan, bench)) if k % workers == worker]
        labels = {}
        for split in (("train", "validation") if bench == "clevrer" else pb.PHYSION_SPLITS):
            for row in load_torch(cache(output, bench, f"labels-{split}.pt")):
                labels[row["index"] if bench == "clevrer" else f"{split}/{row['name']}"] = row
        out = {}
        if bench == "clevrer":
            loader = ThreadPoolExecutor(12)
            frames_iter = loader.map(pb.decode_clevrer_video, [c for _, c in clips])
            native = pb.CLEVRER_NATIVE
        else:
            loader = spawn_pool(10)
            frames_iter = loader.map(_physion_images, [c for _, c in clips], chunksize=1)
            native = pb.PHYSION_IMAGE
        with harness.deterministic_scoring():
            for (split, clip), frames in zip(clips, frames_iter):
                carrier, presence, kinds, centers = parse_clip(model, bench, frames)
                key = clip if bench == "clevrer" else f"{split}/{Path(clip).stem}"
                out[key] = {"split": split, "carrier": carrier, "presence": presence, "kinds": kinds,
                            "fidelity": fidelity(presence, centers, kinds, labels[key], native)}
                if len(out) % 500 == 0:
                    log(f"{bench} parse w{worker}: {len(out)}/{len(clips)}")
        loader.shutdown()
        atomic_torch(target, out)
        log(f"{bench} parse w{worker}: {len(out)} clips")
    wall = time.monotonic() - began
    mark(output, f"parse-w{worker}", wall, wall)
    return 0


def load_parsed(output, bench, workers=4):
    out = {}
    for k in range(workers):
        out.update(load_torch(cache(output, bench, f"parsed-w{k}.pt")))
    return out


# =========================================================================== predictor data


class PixelData:
    """GPU-resident training clips (parsed carriers padded by absorption) with description labels."""

    def __init__(self, bench, parsed, labels, device=DEVICE):
        self.bench = bench
        frames = [len(parsed[k]["carrier"]) for k in labels]
        self.n = torch.tensor(frames, device=device)
        pad = max(frames) + max(LONG[bench]["last"], TRAINING["window_length"]) + 1
        z = torch.zeros(len(labels), pad, DIM)
        contact = torch.zeros(len(labels), pad, SLOTS, SLOTS, dtype=torch.bool)
        macros = torch.zeros(len(labels), pad, 2, 3, dtype=torch.bool)
        available = torch.zeros(len(labels), pad, 2, 3, dtype=torch.bool)
        declared = torch.zeros(len(labels), SLOTS, dtype=torch.bool)
        for i, (key, row) in enumerate(labels.items()):
            carrier = parsed[key]["carrier"]
            t = len(carrier)
            z[i, :t] = carrier
            z[i, t:] = carrier[-1]
            events = torch.as_tensor(row["contact"]).long()
            if len(events):
                keep = events[:, 0] < pad
                events = events[keep]
                contact[i, events[:, 0], events[:, 1], events[:, 2]] = True
            m, a = torch.as_tensor(row["macros"]), torch.as_tensor(row["macro_available"])
            take = min(pad, len(m))
            macros[i, :take], available[i, :take] = m[:take], a[:take]
            if take < pad:  # CLEVRER clips are never padded past 128 in use; Physion labels cover the pad
                macros[i, take:], available[i, take:] = m[-1], a[-1]
            declared[i] = torch.as_tensor(row["declared"])
        self.z, self.contact = z.to(device), contact.to(device)
        self.macros, self.available, self.declared = macros.to(device), available.to(device), declared.to(device)
        self.pad = pad
        self.device = device

    def windows(self, generators, batch):
        length = TRAINING["window_length"]
        clips, starts = [], []
        for g in generators:
            c = torch.randint(len(self.n), (batch,), generator=g, device=self.device)
            room = (self.n[c] - length).clamp_min(0) + 1
            s = (torch.rand(batch, generator=g, device=self.device) * room).long()
            clips.append(c)
            starts.append(s)
        c, s = torch.stack(clips), torch.stack(starts)
        t = s[..., None] + torch.arange(length, device=self.device)
        z = self.z[c[..., None], t]
        contact = self.contact[c[..., None], t]
        mask0 = self.declared[c][:, :, None, :, None] & self.declared[c][:, :, None, None, :]
        mask0 = mask0 & ~torch.eye(SLOTS, dtype=torch.bool, device=self.device)
        mask0 = mask0.expand(-1, -1, length, -1, -1)
        relations = torch.stack((contact, torch.zeros_like(contact)), -1).float()
        relations_mask = torch.stack((mask0, torch.zeros_like(mask0)), -1)
        return {"z": z, "action": torch.zeros(*c.shape, 5, device=self.device),
                "length": torch.minimum(torch.full_like(s, length - 1), self.n[c] - 1 - s),
                "relations": relations, "relations_mask": relations_mask,
                "macros": self.macros[c[..., None], t].float(), "macros_mask": self.available[c[..., None], t]}

    def long_batch(self, generators, batch, horizon, delta):
        reserve = LONG[self.bench]["reserve"]
        starts, targets = [], []
        for g in generators:
            c = torch.randint(len(self.n), (batch,), generator=g, device=self.device)
            room = (self.n[c] - 1 - reserve).clamp_min(0) + 1
            s = (torch.rand(batch, generator=g, device=self.device) * room).long()
            steps = s[:, None] + delta * torch.arange(1, horizon // delta + 1, device=self.device)
            starts.append(self.z[c, s])
            targets.append(self.z[c[:, None], steps.clamp(max=self.pad - 1)])
        start = torch.stack(starts)
        return {"start": start, "action": torch.zeros(*start.shape[:2], 5, device=self.device),
                "targets": torch.stack(targets)}


def training_labels(output, bench):
    split = "train" if bench == "clevrer" else "dynamics"
    rows = load_torch(cache(output, bench, f"labels-{split}.pt"))
    return {(r["index"] if bench == "clevrer" else f"{split}/{r['name']}"): r for r in rows}


def make_ensemble(models):
    class Skipping(sd.SkippingEnsemble):
        patience = TRAINING["patience"]
    return Skipping(models, learning_rate=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"],
                    grad_clip=TRAINING["grad_clip"], compile=True)


def horizon_at(bench, step):
    spec = LONG[bench]
    return dd.horizon_at(step, first=spec["first"], last=spec["last"], increment=spec["increment"],
                         every=spec["every"])


def train_family(output, bench, family, data, *, seeds=SEEDS, steps=None):
    """One ensemble (members = seeds) of one family on one benchmark; resumable every 1000 updates."""
    steps = steps or TRAINING["steps"]
    cls, spec = pc.FAMILIES[family]
    models = []
    for seed in seeds:
        torch.manual_seed(seed)
        models.append(cls().to(DEVICE))
    ensemble = make_ensemble(models)
    generators = [torch.Generator(device=DEVICE).manual_seed(s + TRAINING["sampler_offset"]) for s in seeds]
    long_generators = [torch.Generator(device=DEVICE).manual_seed(s + TRAINING["long_offset"]) for s in seeds]
    progress = Path(output) / "progress" / f"{bench}--{family}.pt"
    log_rows, first, started_at = [], 0, utc_now()
    if progress.is_file():
        state = load_torch(progress)
        ensemble.load_state(state["ensemble"])
        for g, s in zip(generators + long_generators, state["generators"] + state["long_generators"]):
            g.set_state(s)
        log_rows, first, started_at = state["log"], state["step"], state["started_at"]
    began = time.monotonic()
    for step in range(first, steps):
        pair = PAIRS[step % len(PAIRS)]
        batch = data.windows(generators, TRAINING["batch_size"])
        loss = pf.local_loss(ensemble, batch, pair, spec)
        horizon = horizon_at(bench, step)
        long = data.long_batch(long_generators, LONG[bench]["batch_size"], horizon, pair.delta)
        long_term, _ = sd.truncated_long_and_rank_loss(ensemble, pair, long=long, rank=None, horizon=horizon,
                                                       endpoint=horizon, presence=0, center=(0, 0), temperature=1.0,
                                                       truncation=LONG[bench]["truncation"])
        total = loss + LONG[bench]["weight"] * long_term
        diagnostics = ensemble.update(total)
        if step % 50 == 0 or step == steps - 1:
            log_rows.append({"step": step, "pair": REQUESTS[step % len(PAIRS)], "horizon": horizon,
                             "loss": [float(x) for x in total.detach().cpu()],
                             "long": [float(x) for x in long_term.detach().cpu()],
                             "grad_norm": [float(x) for x in diagnostics["grad_norm"].cpu()]})
        if (step + 1) % 1000 == 0 and step + 1 < steps:
            atomic_torch(progress, {"ensemble": ensemble.state(), "step": step + 1, "log": log_rows,
                                    "started_at": started_at,
                                    "generators": [g.get_state() for g in generators],
                                    "long_generators": [g.get_state() for g in long_generators]})
            log(f"{bench}/{family}: {step + 1}/{steps} ({time.monotonic() - began:.0f}s) "
                f"loss {np.mean(log_rows[-1]['loss']):.4f}")
    return ensemble, log_rows, time.monotonic() - began, started_at


def train(output, workers, worker):
    output = Path(output)
    load_plan(output)
    units = [(b, f) for b in BENCHES for f in FAMILIES]
    torch.backends.cudnn.benchmark = True
    for k, (bench, family) in enumerate(units):
        if k % workers != worker:
            continue
        report = Path(output) / "training" / f"{bench}--{family}.json"
        if report.is_file():
            continue
        parsed = load_parsed(output, bench)
        data = PixelData(bench, parsed, training_labels(output, bench))
        ensemble, rows, wall, started_at = train_family(output, bench, family, data)
        members = {}
        for i, seed in enumerate(SEEDS):
            path = model_path(output, bench, family, seed)
            failed = bool(ensemble.retired[i])
            if not failed:
                atomic_torch(path, {"schema": SCHEMA["report"], "bench": bench, "family": family, "seed": seed,
                                    "model": ensemble.member_state(i)})
            members[str(seed)] = {"typed_failure": "retired_nonfinite" if failed else None,
                                  "retired_at": ensemble.retired_at[i], "skipped": int(ensemble.skipped[i]),
                                  "sha256": None if failed else sha256_of(path)}
        write_json(report, {"schema": SCHEMA["report"], "bench": bench, "family": family, "members": members,
                            "parameters_per_member": sum(v[0].numel() for v in ensemble.params.values()),
                            "clips": len(data.n), "log": rows, "wall_seconds": wall, "started_at": started_at,
                            "trained_at": utc_now()})
        mark(output, f"train-{bench}-{family}", wall, wall)
    return 0


def load_member(output, bench, family, seed):
    cls, _ = pc.FAMILIES[family]
    model = cls().to(DEVICE)
    model.load_state_dict(load_torch(model_path(output, bench, family, seed))["model"])
    return model.eval()


# =========================================================================== policies


def policy_windows(data_clips, seed):
    """POLICY['windows'] windows of 61 parsed frames (60 transitions), split into POLICY['units'] units."""
    rng = np.random.default_rng(seed + POLICY["window_offset"])
    keys = list(data_clips)
    units = [[] for _ in range(POLICY["units"])]
    length = POLICY["window_length"]
    for k in range(POLICY["windows"]):
        carrier = data_clips[keys[int(rng.integers(len(keys)))]]
        n = len(carrier)
        s = int(rng.integers(max(0, n - 1 - length) + 1))
        index = [min(s + i, n - 1) for i in range(length + 1)]
        units[k % POLICY["units"]].append({"z": carrier[index], "action": torch.zeros(5),
                                           "length": min(length, n - 1 - s)})
    return units


def train_policy(model, kind, seed, units):
    policy = (rp.new_policy(PAIRS, seed, device=DEVICE) if kind == "J"
              else rs.new_masked_policy(PAIRS, R_CONT, seed, device=DEVICE))
    history, stats = [], {}
    for round_index in range(POLICY["rounds"]):
        t0 = time.monotonic()
        if kind == "J":
            rows = [rp.window_rows(model, policy, u, aggregated=bool(round_index),
                                   compute_weight=POLICY["compute_weight"], device=DEVICE) for u in units]
        else:
            rows = [rs.restricted_window_rows(model, policy, u, R_CONT, aggregated=bool(round_index),
                                              compute_weight=POLICY["compute_weight"], device=DEVICE) for u in units]
        history.append(rows)
        stats[f"label_seconds_{round_index}"] = time.monotonic() - t0
        t0 = time.monotonic()
        losses = rp.distil_round(policy, rp.aggregate_rounds(history), round_index, seed=seed, device=DEVICE)
        stats[f"distil_seconds_{round_index}"] = time.monotonic() - t0
        stats[f"final_loss_{round_index}"] = float(np.mean(losses[-60:]))
        counts = torch.bincount(torch.cat([r["labels"] for r in rows]), minlength=len(PAIRS))
        stats[f"label_shares_{round_index}"] = (counts / counts.sum()).tolist()
    policy.eval()
    return policy, stats


def policy_stage(output, workers, worker):
    output = Path(output)
    load_plan(output)
    units = [(b, f, s, k) for b in BENCHES for f in FAMILIES for s in SEEDS for k in ("J", "J-cont")]
    began = time.monotonic()
    parsed_cache = {}
    for i, (bench, family, seed, kind) in enumerate(units):
        if i % workers != worker:
            continue
        path = policy_path(output, bench, family, seed, kind)
        if path.with_suffix(".json").is_file() or not model_path(output, bench, family, seed).is_file():
            continue
        t0 = time.monotonic()
        if bench not in parsed_cache:
            parsed = load_parsed(output, bench)
            split = "train" if bench == "clevrer" else "dynamics"
            parsed_cache[bench] = {k: v["carrier"] for k, v in parsed.items() if v["split"] == split}
        model = load_member(output, bench, family, seed)
        with harness.deterministic_scoring() as scoring_policy:
            policy, stats = train_policy(model, kind, seed, policy_windows(parsed_cache[bench], seed))
        atomic_torch(path, {"model": policy.state_dict(), "kind": kind})
        write_json(path.with_suffix(".json"), {"schema": SCHEMA["report"], "bench": bench, "family": family,
                                               "seed": seed, "kind": kind, "stats": stats,
                                               "scoring_policy": scoring_policy, "sha256": sha256_of(path),
                                               "policy_macs": rp.policy_macs(policy),
                                               "wall_seconds": time.monotonic() - t0, "trained_at": utc_now()})
        log(f"policy {bench}/{family}/{seed}/{kind}: {time.monotonic() - t0:.0f}s")
    wall = time.monotonic() - began
    mark(output, f"policy-w{worker}", wall, wall)
    return 0


def load_policy(output, bench, family, seed, kind):
    policy = (rp.RequestPolicy(PAIRS) if kind == "J" else rs.MaskedRequestPolicy(PAIRS, R_CONT)).to(DEVICE)
    policy.load_state_dict(load_torch(policy_path(output, bench, family, seed, kind))["model"])
    return policy.eval()


# =========================================================================== scoring


def arm_spec(arm):
    if arm in REQUESTS:
        return {"kind": "fixed", "pair_index": REQUESTS.index(arm)}
    return {"kind": "policy"}


def position_mse(prediction, target):
    """Per row: mean squared error of columns 5, 6 over slots with target column 7 > 0.5 (nan if none);
    equals clevrer_boundary.field_errors(...)['position_mse'] per row."""
    p = prediction[:, 2:].reshape(len(prediction), SLOTS, 13)
    t = target[:, 2:].reshape(len(target), SLOTS, 13)
    mask = (t[..., 7] > .5).double()
    squared = (p[..., 5:7].double() - t[..., 5:7].double()).square().sum(-1)
    count = mask.sum(-1) * 2
    return torch.where(count > 0, (squared * mask).sum(-1) / count.clamp_min(1), torch.full_like(count, float("nan")))


def absorbed(clip, frame):
    return clip[min(frame, len(clip) - 1)]


def rollout_errors(model, policies, clips, annotations, bench):
    """{arm: {endpoint: {"parsed": [clip], "annotation": [clip], "macs": mean per rollout}}}."""
    out = {}
    for arm in ARMS:
        policy = policies.get(arm)
        spec = arm_spec(arm)
        out[arm] = {}
        for endpoint in ENDPOINTS[bench]:
            parsed, annotated, macs = [], [], 0
            for context in CONTEXTS[bench]:
                z0 = torch.stack([absorbed(c, context) for c in clips]).to(DEVICE)
                z_end, _, timing, _ = pc.rollout(model, policy, z0, endpoint, spec)
                target = torch.stack([absorbed(c, context + endpoint) for c in clips]).to(DEVICE)
                truth = torch.stack([absorbed(a, context + endpoint) for a in annotations]).to(DEVICE)
                parsed.append(position_mse(z_end, target).cpu())
                annotated.append(position_mse(z_end, truth).cpu())
                macs += timing["linear_macs"]
            out[arm][str(endpoint)] = {"parsed": nan_list(torch.stack(parsed).nanmean(0)),
                                       "annotation": nan_list(torch.stack(annotated).nanmean(0)),
                                       "macs_per_rollout": macs / (len(clips) * len(CONTEXTS[bench]))}
    return out


def nan_list(values):
    return [None if not math.isfinite(float(v)) else round(float(v), 10) for v in values]


@torch.no_grad()
def contact_max(model, landings, rows, endpoint_frames=None):
    """Running max over landings of the gated contact values [rows, 18, 18]; landings past the row's
    endpoint are ignored; returns (scores, readout calls)."""
    best = torch.zeros(rows, SLOTS, SLOTS, device=DEVICE)
    calls = torch.zeros(rows, dtype=torch.long, device=DEVICE)
    for index, positions, z, _ in landings:
        keep = torch.ones(len(index), dtype=torch.bool, device=DEVICE) if endpoint_frames is None \
            else positions <= endpoint_frames[index]
        if not bool(keep.any()):
            continue
        values = pc.contact_values(model, z[keep])
        rows_ = index[keep]
        best[rows_] = torch.maximum(best[rows_], values)
        calls[rows_] += 1
    return best, calls


def qa_scores(model, policies, plan, parsed, questions):
    """Per arm: predictive per-question correctness, counterfactual per-option and per-question
    correctness, rollout + readout MACs per question (lists in plan question order)."""
    videos = plan["membership"]["clevrer"]["validation"]["scenes"]
    micro_readout = pc.readout_macs(model, "micro")
    attributes = {v: pc.slot_attributes(parsed[v]["presence"].float(), parsed[v]["kinds"].float()) for v in videos}
    observed = {}
    for v in videos:
        z = parsed[v]["carrier"].to(DEVICE)
        observed[v] = pc.contact_values(model, z)  # [128, 18, 18]
    removals = []
    for v in videos:
        for q in questions[v]:
            if q["question_type"] == "counterfactual":
                removed = pc.resolve_objects(q["program"], attributes[v])
                removals.append((v, removed[0] if removed else None))
    removals = sorted(set(removals), key=lambda r: (r[0], -1 if r[1] is None else r[1]))
    starts = {}
    for v, x in removals:
        if x is None:
            starts[(v, x)] = None
            continue
        hits = (observed[v][:, x].max(-1).values >= QA["threshold"]).nonzero(as_tuple=True)[0]
        starts[(v, x)] = None if not len(hits) else max(0, int(hits[0]) - 1)
    out = {}
    for arm in ARMS:
        policy, spec = policies.get(arm), arm_spec(arm)
        z0 = torch.stack([parsed[v]["carrier"][QA["predictive_start"]] for v in videos]).to(DEVICE)
        _, _, timing, landings = pc.rollout(model, policy, z0, QA["predictive_horizon"], spec, record=True)
        predictive, calls = contact_max(model, landings, len(videos))
        predictive_macs = timing["linear_macs"] / len(videos)
        active = [r for r in removals if starts[r] is not None]
        if active:
            z0 = []
            for v, x in active:
                z = parsed[v]["carrier"][starts[(v, x)]].clone()
                base = 2 + 13 * x
                z[base] = 0.0
                z[base + 5:base + 11] = 0.0
                z0.append(z)
            z0 = torch.stack(z0).to(DEVICE)
            ends = torch.tensor([pb.CLEVRER_FRAMES - 1 - starts[r] for r in active], device=DEVICE)
            _, _, cf_timing, cf_landings = pc.rollout(model, policy, z0, ends, spec, record=True)
            counter, cf_calls = contact_max(model, cf_landings, len(active), ends)
            cf_macs = cf_timing["linear_macs"] / len(active)
        row = {"predictive": [], "counterfactual_option": [], "counterfactual_question": [], "macs": [],
               "unresolved_choices": 0}
        items = {"predictive": [], "counterfactual_option": [], "counterfactual_question": []}
        active_index = {r: i for i, r in enumerate(active)}
        for vi, v in enumerate(videos):
            for q in questions[v]:
                if q["question_type"] == "predictive":
                    negate = "negate" in q["program"]
                    scores = []
                    for choice in q["choices"]:
                        pair = pc.choice_pair(choice, attributes[v])
                        row["unresolved_choices"] += pair is None
                        scores.append(0.0 if pair is None else float(torch.maximum(
                            predictive[vi, pair[0], pair[1]], predictive[vi, pair[1], pair[0]])))
                    pick = int(np.argmin(scores) if negate else np.argmax(scores))
                    row["predictive"].append(int(q["choices"][pick]["answer"] == "correct"))
                    items["predictive"].append(v)
                    row["macs"].append(predictive_macs + micro_readout * float(calls[vi]))
                elif q["question_type"] == "counterfactual":
                    negate = "negate" in q["program"]
                    removed = pc.resolve_objects(q["program"], attributes[v])
                    key = (v, removed[0] if removed else None)
                    start = starts[key]
                    options = []
                    for choice in q["choices"]:
                        pair = pc.choice_pair(choice, attributes[v])
                        row["unresolved_choices"] += pair is None
                        if pair is None:
                            happens = False
                        elif start is None:
                            happens = bool(torch.maximum(observed[v][:, pair[0], pair[1]],
                                                         observed[v][:, pair[1], pair[0]]).max() >= QA["threshold"])
                        else:
                            seen = torch.maximum(observed[v][:start + 1, pair[0], pair[1]],
                                                 observed[v][:start + 1, pair[1], pair[0]]).max()
                            i = active_index[key]
                            imagined = torch.maximum(counter[i, pair[0], pair[1]], counter[i, pair[1], pair[0]])
                            happens = bool(max(float(seen), float(imagined)) >= QA["threshold"])
                        options.append(int((happens != negate) == (choice["answer"] == "correct")))
                    row["counterfactual_option"].extend(options)
                    row["counterfactual_question"].append(int(all(options)))
                    items["counterfactual_option"].extend([v] * len(options))
                    items["counterfactual_question"].append(v)
                    if start is None:
                        row["macs"].append(micro_readout * pb.CLEVRER_FRAMES)
                    else:
                        i = active_index[key]
                        row["macs"].append(cf_macs + micro_readout * (float(cf_calls[i]) + start + 1))
        out[arm] = row
    out["items"] = items
    return out


def ocp_features(model, policy, spec, carriers):
    """Simulated-protocol features per trial: parsed frames 0..12 and the zero-order-held landed carrier on
    the grid 12 + 5k (k = 1..18); predicate-rule prediction; rollout + readout MACs per trial."""
    start, endpoint, every = OCP["start"], OCP["endpoint"], OCP["grid_every"]
    inputs = torch.stack([torch.stack([absorbed(c, f) for f in range(start + 1)]) for c in carriers]).to(DEVICE)
    z0 = inputs[:, -1]
    _, _, timing, landings = pc.rollout(model, policy, z0, endpoint, spec, record=True)
    grid = torch.arange(every, endpoint + 1, every, device=DEVICE)
    held = z0[:, None].expand(-1, len(grid), -1).clone()
    predicate = torch.zeros(len(carriers), dtype=torch.bool, device=DEVICE)
    calls = torch.zeros(len(carriers), device=DEVICE)
    for index, positions, z, chosen in landings:
        update = positions[:, None] <= grid[None]
        rows = index[:, None].expand(-1, len(grid))
        held[rows[update], torch.nonzero(update, as_tuple=True)[1]] = z[:, None].expand(-1, len(grid), -1)[update]
        for pair_index in chosen.unique().tolist():
            sel = chosen == pair_index
            values = pc.macro_values(model, z[sel], PAIRS[pair_index].delta)[:, 0]
            predicate[index[sel]] |= values >= OCP["predicate_threshold"]
        calls[index] += 1
    features = torch.cat([inputs.flatten(1), held.flatten(1)], 1).cpu().numpy()
    macs = timing["linear_macs"] / len(carriers) + pc.readout_macs(model, "macro") * calls.cpu().numpy()
    return features, predicate.cpu().numpy(), macs


def observed_features(carriers, protocol):
    start, endpoint, every = OCP["start"], OCP["endpoint"], OCP["grid_every"]
    rows = []
    for c in carriers:
        frames = list(range(start + 1))
        if protocol == "observed":
            frames += list(range(start + every, start + endpoint + 1, every))
        rows.append(torch.stack([absorbed(c, f) for f in frames]).flatten())
    return torch.stack(rows).numpy()


def readout(train_x, train_y, test_x, seed):
    """physopt / physics-models ReadoutObjective: undersample the training set to 50/50 (seeded),
    StandardScaler + LogisticRegression(max_iter 200), GridSearchCV over C = logspace(-8, 8, 17),
    StratifiedKFold(5, shuffle, random_state = seed); test predictions on the full test set."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    pos, neg = np.nonzero(train_y)[0], np.nonzero(~train_y)[0]
    small, large = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    keep = np.sort(np.concatenate([small, rng.choice(large, len(small), replace=False)]))
    pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=OCP["max_iter"]))])
    search = GridSearchCV(pipe, {"clf__C": np.logspace(*OCP["C_logspace"])},
                          cv=StratifiedKFold(OCP["folds"], shuffle=True, random_state=seed % (2 ** 32)), n_jobs=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        search.fit(train_x[keep], train_y[keep])
    return search.predict(test_x).astype(bool), {"best_C": float(search.best_params_["clf__C"]),
                                                 "train_rows": int(len(keep))}


def ocp_scores(model, policies, plan, parsed, labels, seed):
    readout_keys = [f"readout/{Path(p).stem}" for p in plan["membership"]["physion"]["readout"]["trials"]]
    test_keys = [f"testing/{Path(p).stem}" for p in plan["membership"]["physion"]["testing"]["trials"]]
    train_y = np.array([labels["readout"][k]["ocp"] for k in readout_keys])
    test_y = np.array([labels["testing"][k]["ocp"] for k in test_keys])
    train_c = [parsed[k]["carrier"] for k in readout_keys]
    test_c = [parsed[k]["carrier"] for k in test_keys]
    out = {}
    for arm in ARMS:
        policy, spec = policies.get(arm), arm_spec(arm)
        train_x, _, _ = ocp_features(model, policy, spec, train_c)
        test_x, predicate, macs = ocp_features(model, policy, spec, test_c)
        prediction, info = readout(train_x, train_y, test_x, seed)
        out[arm] = {"readout": (prediction == test_y).astype(int).tolist(),
                    "predicate_rule": (predicate == test_y).astype(int).tolist(),
                    "macs": [round(float(m), 1) for m in macs], **info}
    for protocol in ("observed", "input"):
        prediction, info = readout(observed_features(train_c, protocol), train_y, observed_features(test_c, protocol),
                                   seed)
        out[f"protocol-{protocol}"] = {"readout": (prediction == test_y).astype(int).tolist(), **info}
    out["labels"] = {"test_positive": int(test_y.sum()), "test_count": int(len(test_y)),
                     "readout_positive": int(train_y.sum()), "readout_count": int(len(train_y))}
    return out


def scoring_cells():
    return [(b, f, s) for b in BENCHES for f in FAMILIES for s in SEEDS]


def score_cell(output, plan, bench, family, seed, parsed, labels, questions, *, clip_limit=None):
    model = load_member(output, bench, family, seed)
    policies = {k: load_policy(output, bench, family, seed, k) for k in ("J", "J-cont")}
    if bench == "clevrer":
        keys = plan["membership"]["clevrer"]["validation"]["scenes"]
        annotation_rows = labels["validation"]
    else:
        keys = [f"testing/{Path(p).stem}" for p in plan["membership"]["physion"]["testing"]["trials"]]
        annotation_rows = labels["testing"]
    keys = keys[:clip_limit] if clip_limit else keys
    clips = [parsed[k]["carrier"] for k in keys]
    annotations = [torch.as_tensor(annotation_rows[k]["annotation"]) for k in keys]
    record = {"rollout": rollout_errors(model, policies, clips, annotations, bench)}
    if clip_limit:
        return record
    if bench == "clevrer" and family == "hybrid":
        record["clevrer_qa"] = qa_scores(model, policies, plan, parsed, questions)
    elif bench == "clevrer":
        record["clevrer_qa"] = {"unsupported": "the label-free micro codes carry no collision semantics; the QA "
                                               "readout needs a labelled contact head"}
    else:
        record["physion_ocp"] = ocp_scores(model, policies, plan, parsed, labels, seed)
    return record


def load_labels(output, bench):
    splits = ("train", "validation") if bench == "clevrer" else tuple(pb.PHYSION_SPLITS)
    out = {}
    for split in splits:
        if bench == "clevrer" and split == "train":
            continue
        rows = load_torch(cache(output, bench, f"labels-{split}.pt"))
        out[split] = {(r["index"] if bench == "clevrer" else f"{split}/{r['name']}"): r for r in rows}
    return out


def load_questions(plan):
    videos = set(plan["membership"]["clevrer"]["validation"]["scenes"])
    payload = json.loads(pb.CLEVRER_QUESTIONS["validation"].read_text())
    return {int(v["scene_index"]): [q for q in v["questions"] if q["question_type"] in ("predictive", "counterfactual")]
            for v in payload if int(v["scene_index"]) in videos}


def score(output, workers, worker):
    output = Path(output)
    plan = load_plan(output)
    began = time.monotonic()
    cache_ = {}
    for i, (bench, family, seed) in enumerate(scoring_cells()):
        if i % workers != worker:
            continue
        path = record_path(output, bench, family, seed)
        if path.is_file():
            continue
        if bench not in cache_:
            cache_ = {bench: (load_parsed(output, bench), load_labels(output, bench),
                              load_questions(plan) if bench == "clevrer" else None)}
        parsed, labels, questions = cache_[bench]
        t0, started = time.monotonic(), utc_now()
        failure, body = None, {}
        if not model_path(output, bench, family, seed).is_file():
            failure = "predictor_retired_nonfinite"
        else:
            with harness.deterministic_scoring() as scoring_policy:
                body = score_cell(output, plan, bench, family, seed, parsed, labels, questions)
        write_json(path, {"schema": SCHEMA["record"], "plan_identity": IDENTITY,
                          "scoring_policy": scoring_policy if not failure else None,
                          "cell": {"bench": bench, "family": family, "seed": seed}, "typed_failure": failure,
                          "started_at": started, "scored_at": utc_now(), "wall_seconds": time.monotonic() - t0, **body})
        log(f"scored {bench}/{family}/{seed} ({time.monotonic() - t0:.0f}s)")
    wall = time.monotonic() - began
    mark(output, f"score-w{worker}", wall, wall)
    return 0


# =========================================================================== exploratory readouts (post-freeze)

EXPLORE = {
    "status": "EXPLORATORY: designed after the frozen readouts were seen at their floor (amendment A2); the "
              "frozen tokens are unchanged and these numbers carry no disposition",
    "existence_frames": 5,
    "threshold_quantiles": 101,
    "clevrer": "objects exist if parsed presence >= 0.5 on >= 5 frames (the frozen rule, one frame, admits "
               "single-frame phantom objects); predictive: the higher of the two choice scores (argmax of the max "
               "gated contact value over the rollout landings, ties to the lower choice); counterfactual: option "
               "'happens' iff its score >= tau, tau fitted per arm and member on the training split's "
               "counterfactual options (the score quantile, 101 levels, of best training option accuracy)",
    "physion": "OCP predicate score = max over rollout landings of the gated ocp-contact value at the landing's "
               "Delta; tau fitted per arm and member on the readout split (best readout accuracy over 101 score "
               "quantiles), applied to the testing split",
}


def load_questions_split(plan, split):
    videos = set(plan["membership"]["clevrer"][split]["scenes"])
    payload = json.loads(pb.CLEVRER_QUESTIONS[split].read_text())
    return {int(v["scene_index"]): [q for q in v["questions"] if q["question_type"] in ("predictive", "counterfactual")]
            for v in payload if int(v["scene_index"]) in videos}


def qa_raw(model, policies, videos, parsed, questions):
    """``qa_scores`` with raw scores: per arm the two predictive choice scores and every counterfactual option
    score (max of the observed reading up to the start frame and the imagined rollout), with the answers,
    negations and video clusters (objects exist on >= EXPLORE['existence_frames'] frames)."""
    attributes = {v: pc.slot_attributes(parsed[v]["presence"].float(), parsed[v]["kinds"].float(),
                                        EXPLORE["existence_frames"]) for v in videos}
    observed = {v: pc.contact_values(model, parsed[v]["carrier"].to(DEVICE)) for v in videos}
    removals, meta = set(), {"predictive": [], "counterfactual": []}
    for v in videos:
        for q in questions[v]:
            negate = "negate" in q["program"]
            pairs = [pc.choice_pair(c, attributes[v]) for c in q["choices"]]
            answers = [c["answer"] == "correct" for c in q["choices"]]
            if q["question_type"] == "predictive":
                meta["predictive"].append({"video": v, "negate": negate, "pairs": pairs, "answers": answers})
            else:
                removed = pc.resolve_objects(q["program"], attributes[v])
                key = (v, removed[0] if removed else None)
                removals.add(key)
                meta["counterfactual"].append({"video": v, "negate": negate, "pairs": pairs, "answers": answers,
                                               "removal": key})
    starts = {}
    for v, x in removals:
        hits = [] if x is None else (observed[v][:, x].max(-1).values >= QA["threshold"]).nonzero(as_tuple=True)[0]
        starts[(v, x)] = None if not len(hits) else max(0, int(hits[0]) - 1)
    active = sorted((r for r in removals if starts[r] is not None), key=lambda r: (r[0], r[1]))
    index = {r: i for i, r in enumerate(active)}

    def pair_score(table, row, pair):
        return float(torch.maximum(table[row, pair[0], pair[1]], table[row, pair[1], pair[0]]))
    out = {"items": {"predictive": [m["video"] for m in meta["predictive"]],
                     "counterfactual_option": [m["video"] for m in meta["counterfactual"] for _ in m["answers"]]},
           "predictive_answers": [m["answers"] for m in meta["predictive"]],
           "predictive_negate": [m["negate"] for m in meta["predictive"]],
           "counterfactual_answers": [a for m in meta["counterfactual"] for a in m["answers"]],
           "counterfactual_negate": [m["negate"] for m in meta["counterfactual"] for _ in m["answers"]],
           "unresolved_choices": sum(p is None for m in meta["predictive"] + meta["counterfactual"]
                                     for p in m["pairs"])}
    video_row = {v: i for i, v in enumerate(videos)}
    for arm in ARMS:
        policy, spec = policies.get(arm), arm_spec(arm)
        z0 = torch.stack([parsed[v]["carrier"][QA["predictive_start"]] for v in videos]).to(DEVICE)
        _, _, _, landings = pc.rollout(model, policy, z0, QA["predictive_horizon"], spec, record=True)
        predictive, _ = contact_max(model, landings, len(videos))
        counter = None
        if active:
            z0 = []
            for v, x in active:
                z = parsed[v]["carrier"][starts[(v, x)]].clone()
                z[2 + 13 * x] = 0.0
                z[2 + 13 * x + 5:2 + 13 * x + 11] = 0.0
                z0.append(z)
            ends = torch.tensor([pb.CLEVRER_FRAMES - 1 - starts[r] for r in active], device=DEVICE)
            _, _, _, cf_landings = pc.rollout(model, policy, torch.stack(z0).to(DEVICE), ends, spec, record=True)
            counter, _ = contact_max(model, cf_landings, len(active), ends)
        predictive_scores = [[0.0 if p is None else pair_score(predictive, video_row[m["video"]], p) for p in m["pairs"]]
                             for m in meta["predictive"]]
        option_scores = []
        for m in meta["counterfactual"]:
            start = starts[m["removal"]]
            for p in m["pairs"]:
                if p is None:
                    option_scores.append(0.0)
                elif start is None:
                    option_scores.append(float(torch.maximum(observed[m["video"]][:, p[0], p[1]],
                                                             observed[m["video"]][:, p[1], p[0]]).max()))
                else:
                    seen = float(torch.maximum(observed[m["video"]][:start + 1, p[0], p[1]],
                                               observed[m["video"]][:start + 1, p[1], p[0]]).max())
                    option_scores.append(max(seen, pair_score(counter, index[m["removal"]], p)))
        out[arm] = {"predictive_scores": [[round(s, 8) for s in row] for row in predictive_scores],
                    "counterfactual_scores": [round(s, 8) for s in option_scores]}
    return out


def ocp_predicate_scores(model, policy, spec, carriers):
    """Max over rollout landings of the gated ocp-contact value at the landing's Delta, per trial."""
    z0 = torch.stack([absorbed(c, OCP["start"]) for c in carriers]).to(DEVICE)
    _, _, _, landings = pc.rollout(model, policy, z0, OCP["endpoint"], spec, record=True)
    best = torch.zeros(len(carriers), device=DEVICE)
    for index, _, z, chosen in landings:
        for pair_index in chosen.unique().tolist():
            sel = chosen == pair_index
            values = pc.macro_values(model, z[sel], PAIRS[pair_index].delta)[:, 0]
            best[index[sel]] = torch.maximum(best[index[sel]], values)
    return [round(float(v), 8) for v in best.cpu()]


def explore_cells():
    return [("clevrer", "hybrid", s) for s in SEEDS] + [("physion", f, s) for f in FAMILIES for s in SEEDS]


def explore(output, workers, worker):
    output = Path(output)
    plan = load_plan(output)
    began = time.monotonic()
    loaded = {}
    for i, (bench, family, seed) in enumerate(explore_cells()):
        path = output / "exploratory" / f"{member_tag(bench, family, seed)}.json"
        if i % workers != worker or path.is_file():
            continue
        if bench not in loaded:
            loaded = {bench: (load_parsed(output, bench), load_labels(output, bench))}
        parsed, labels = loaded[bench]
        t0, started = time.monotonic(), utc_now()
        model = load_member(output, bench, family, seed)
        policies = {k: load_policy(output, bench, family, seed, k) for k in ("J", "J-cont")}
        with harness.deterministic_scoring() as scoring_policy:
            if bench == "clevrer":
                body = {split: qa_raw(model, policies, plan["membership"]["clevrer"][split]["scenes"], parsed,
                                      load_questions_split(plan, split)) for split in ("train", "validation")}
            else:
                body = {}
                for split in ("readout", "testing"):
                    keys = [f"{split}/{Path(p).stem}" for p in plan["membership"]["physion"][split]["trials"]]
                    carriers = [parsed[k]["carrier"] for k in keys]
                    body[split] = {"labels": [bool(labels[split][k]["ocp"]) for k in keys],
                                   **{arm: ocp_predicate_scores(model, policies.get(arm), arm_spec(arm), carriers)
                                      for arm in ARMS}}
        write_json(path, {"schema": SCHEMA["record"], "plan_identity": IDENTITY, "status": EXPLORE["status"],
                          "scoring_policy": scoring_policy, "cell": {"bench": bench, "family": family, "seed": seed},
                          "started_at": started, "scored_at": utc_now(), "wall_seconds": time.monotonic() - t0,
                          **body})
        log(f"explored {bench}/{family}/{seed} ({time.monotonic() - t0:.0f}s)")
    wall = time.monotonic() - began
    mark(output, f"explore-w{worker}", wall, wall)
    return 0


def fit_threshold(scores, correct_if_true):
    """tau among the 101 score quantiles (and +inf) maximizing the accuracy of 'true iff score >= tau';
    correct_if_true[i] = the decision 'true' is correct for item i. Ties to the first (lowest) tau."""
    scores, target = np.asarray(scores), np.asarray(correct_if_true, bool)
    candidates = np.unique(np.concatenate([np.quantile(scores, np.linspace(0, 1, EXPLORE["threshold_quantiles"])),
                                           [np.inf]]))
    accuracy = [((scores >= tau) == target).mean() for tau in candidates]
    return float(candidates[int(np.argmax(accuracy))])


def exploratory_tables(output):
    output = Path(output)
    out = {"status": EXPLORE["status"], "rules": EXPLORE, "clevrer": {}, "physion": {}}
    rows = {}
    for bench, family, seed in explore_cells():
        path = output / "exploratory" / f"{member_tag(bench, family, seed)}.json"
        if path.is_file():
            rows[(bench, family, seed)] = read_json(path)
    clevrer = [rows[k] for k in sorted(rows) if k[0] == "clevrer"]
    if clevrer:
        values, taus = {arm: [] for arm in ARMS}, {arm: [] for arm in ARMS}
        for record in clevrer:
            train, val = record["train"], record["validation"]
            for arm in ARMS:
                tau = fit_threshold(train[arm]["counterfactual_scores"],
                                    [a != n for a, n in zip(train["counterfactual_answers"], train["counterfactual_negate"])])
                taus[arm].append(tau)
                predictive = []
                for scores, answers, negate in zip(val[arm]["predictive_scores"], val["predictive_answers"],
                                                   val["predictive_negate"]):
                    pick = int(np.argmin(scores) if negate else np.argmax(scores))
                    predictive.append(int(answers[pick]))
                options = [int((s >= tau) == (a != n)) for s, a, n in zip(val[arm]["counterfactual_scores"],
                                                                       val["counterfactual_answers"],
                                                                       val["counterfactual_negate"])]
                values[arm].append((predictive, options))
        items = clevrer[0]["validation"]["items"]
        groups = {"option": items["predictive"] + items["counterfactual_option"], "predictive": items["predictive"],
                  "counterfactual_option": items["counterfactual_option"]}

        def matrix(arm, metric):
            return np.array([p + o if metric == "option" else (p if metric == "predictive" else o)
                             for p, o in values[arm]], np.float64)
        out["clevrer"]["table"] = {arm: {m: boot(matrix(arm, m), groups[m]) for m in groups} | {"tau": taus[arm]}
                                   for arm in ARMS}
        out["clevrer"]["contrasts"] = {f"J - {other}": {m: boot(matrix("J", m) - matrix(other, m), groups[m])
                                                        for m in groups} for other in ("J-cont",) + REQUESTS}
        out["clevrer"]["unresolved_choices"] = {"validation": clevrer[0]["validation"]["unresolved_choices"],
                                                "train": clevrer[0]["train"]["unresolved_choices"]}
    for family in FAMILIES:
        physion = [rows[k] for k in sorted(rows) if k[0] == "physion" and k[1] == family]
        if not physion:
            continue
        values = {arm: [] for arm in ARMS}
        for record in physion:
            for arm in ARMS:
                tau = fit_threshold(record["readout"][arm], record["readout"]["labels"])
                values[arm].append([int((s >= tau) == y) for s, y in zip(record["testing"][arm],
                                                                        record["testing"]["labels"])])
        table = {arm: boot(np.array(values[arm], np.float64)) for arm in ARMS}
        contrasts = {f"J - {other}": boot(np.array(values["J"], np.float64) - np.array(values[other], np.float64))
                     for other in ("J-cont",) + REQUESTS}
        out["physion"][family] = {"table": table, "contrasts": contrasts}
    return out


# =========================================================================== publish


def token(stat):
    if stat is None:
        return "readiness_or_precision_insufficient"
    mean, (low, high) = stat["mean"], stat["interval"]
    if mean >= MARGIN and low > 0:
        return "supported"
    if high < MARGIN:
        return "not_supported_by_this_experiment"
    return "readiness_or_precision_insufficient"


def boot(matrix, groups=None):
    return pc.clustered_bootstrap(matrix, groups=groups, draws=BOOTSTRAP["draws"], seed=BOOTSTRAP["seed"],
                                  quantiles=BOOTSTRAP["quantiles"])


def matrix_of(records, getter):
    rows = [getter(r) for r in records]
    return np.array([[np.nan if v is None else v for v in row] for row in rows], np.float64)


def collect(output):
    records = {}
    for bench, family, seed in scoring_cells():
        path = record_path(output, bench, family, seed)
        if path.is_file():
            records[(bench, family, seed)] = read_json(path)
    return records


def fmt(stat, digits=3):
    if stat is None:
        return "—"
    return f"{stat['mean']:.{digits}f} [{stat['interval'][0]:.{digits}f}, {stat['interval'][1]:.{digits}f}]"


def fmt_sci(stat):
    if stat is None:
        return "—"
    return f"{stat['mean']:.3e} [{stat['interval'][0]:.3e}, {stat['interval'][1]:.3e}]"


def guards(output, plan, records):
    out = {}
    clevrer = plan["membership"]["clevrer"]
    out["G1"] = {"passed": not (set(clevrer["train"]["scenes"]) & set(clevrer["validation"]["scenes"]))
                 and all(v["parameter_sets"] == 0 for v in plan["membership"]["physion_split_overlap"].values())}
    out["G2"] = {"passed": all(r["scoring_policy"] == dict(harness.SCORING_POLICY) for r in records.values()
                               if not r["typed_failure"])}
    frozen = plan["frozen_at"]
    started, finished = [], []
    for bench in BENCHES:
        reports = [read_json(encoder_path(output, bench).with_name("report.json"))]
        reports += [read_json(Path(output) / "training" / f"{bench}--{family}.json") for family in FAMILIES]
        started += [r["started_at"] for r in reports]
        finished += [r["trained_at"] for r in reports]
        finished += [read_json(policy_path(output, bench, f, s, k).with_suffix(".json"))["trained_at"]
                     for f in FAMILIES for s in SEEDS for k in ("J", "J-cont")
                     if policy_path(output, bench, f, s, k).with_suffix(".json").is_file()]
    scored = [r["started_at"] for r in records.values()]
    out["G3"] = {"passed": min(started) > frozen and (not scored or min(scored) > max(finished)),
                 "frozen_at": frozen, "first_training_start": min(started), "last_training_end": max(finished),
                 "first_scoring_start": min(scored) if scored else None}
    missing = [member_tag(*c) for c in scoring_cells() if c not in records]
    nonfinite = []
    for key, r in records.items():
        for arm, rows in r.get("rollout", {}).items():
            for endpoint, cell in rows.items():
                if all(v is None for v in cell["parsed"]):
                    nonfinite.append(f"{member_tag(*key)}:{arm}:{endpoint}")
    out["G4"] = {"passed": not missing and not nonfinite, "missing": missing, "nonfinite": nonfinite[:20],
                 "typed_failures": {member_tag(*k): r["typed_failure"] for k, r in records.items() if r["typed_failure"]}}
    g5 = read_json(Path(output) / "spot.json") if (Path(output) / "spot.json").is_file() else None
    out["G5"] = {"passed": bool(g5 and g5["passed"]), "detail": g5}
    manifest_ = read_json(Path(output) / "data-manifest.json")
    out["G6"] = {"passed": sha256_of(Path(output) / "data-manifest.json") == plan["inputs"]["data_manifest_sha256"]
                 and all(f["bytes_match"] and f.get("s3_etag_match", True) for b in manifest_["benchmarks"].values()
                         for f in b["files"])}
    return out


def rollout_table(records, bench, family):
    rows = {}
    keys = [(bench, family, s) for s in SEEDS if (bench, family, s) in records
            and not records[(bench, family, s)]["typed_failure"]]
    if not keys:
        return rows
    recs = [records[k] for k in keys]
    for arm in ARMS:
        rows[arm] = {}
        for endpoint in ENDPOINTS[bench]:
            e = str(endpoint)
            rows[arm][e] = {"parsed": boot(matrix_of(recs, lambda r: r["rollout"][arm][e]["parsed"])),
                            "annotation": boot(matrix_of(recs, lambda r: r["rollout"][arm][e]["annotation"])),
                            "macs_per_rollout": float(np.mean([r["rollout"][arm][e]["macs_per_rollout"] for r in recs]))}
    return rows


def decision_values(record, bench, arm, metric):
    if bench == "clevrer":
        qa = record["clevrer_qa"][arm]
        if metric == "option":
            return qa["predictive"] + qa["counterfactual_option"]
        if metric == "question":
            return qa["predictive"] + qa["counterfactual_question"]
        return qa[metric]
    if arm.startswith("protocol-"):
        return record["physion_ocp"][arm]["readout"]
    return record["physion_ocp"][arm][metric]


def decision_groups(record, bench, metric):
    """Cluster id per decision item: the video for CLEVRER (options / questions of one video are drawn
    together); None (every trial its own cluster) for Physion."""
    if bench != "clevrer":
        return None
    items = record["clevrer_qa"]["items"]
    if metric == "option":
        return items["predictive"] + items["counterfactual_option"]
    if metric == "question":
        return items["predictive"] + items["counterfactual_question"]
    return items[metric]


def decision_table(records, bench, family):
    keys = [(bench, family, s) for s in SEEDS if (bench, family, s) in records
            and not records[(bench, family, s)]["typed_failure"]]
    recs = [records[k] for k in keys]
    if not recs or (bench == "clevrer" and "unsupported" in recs[0]["clevrer_qa"]):
        return None
    metrics = (("option", "question", "predictive", "counterfactual_option", "counterfactual_question")
               if bench == "clevrer" else ("readout", "predicate_rule"))
    table = {}
    for arm in ARMS:
        row = {m: boot(matrix_of(recs, lambda r: decision_values(r, bench, arm, m)), decision_groups(recs[0], bench, m))
               for m in metrics}
        macs = [np.mean(r["clevrer_qa"][arm]["macs"] if bench == "clevrer" else r["physion_ocp"][arm]["macs"])
                for r in recs]
        row["macs_per_decision"] = float(np.mean(macs))
        table[arm] = row
    if bench == "physion":
        for protocol in ("protocol-observed", "protocol-input"):
            table[protocol] = {"readout": boot(matrix_of(recs, lambda r: decision_values(r, bench, protocol, "readout"))),
                               "macs_per_decision": 0.0}
    return table


def contrast(records, bench, family_a, arm_a, family_b, arm_b, metric):
    pairs = [(records.get((bench, family_a, s)), records.get((bench, family_b, s))) for s in SEEDS]
    pairs = [(a, b) for a, b in pairs if a and b and not a["typed_failure"] and not b["typed_failure"]]
    if not pairs:
        return None
    a = matrix_of([p[0] for p in pairs], lambda r: decision_values(r, bench, arm_a, metric))
    b = matrix_of([p[1] for p in pairs], lambda r: decision_values(r, bench, arm_b, metric))
    return boot(a - b, decision_groups(pairs[0][0], bench, metric))


def rollout_contrast(records, bench, family_a, arm_a, family_b, arm_b, endpoint):
    pairs = [(records.get((bench, family_a, s)), records.get((bench, family_b, s))) for s in SEEDS]
    pairs = [(a, b) for a, b in pairs if a and b and not a["typed_failure"] and not b["typed_failure"]]
    if not pairs:
        return None
    e = str(endpoint)
    a = matrix_of([p[0] for p in pairs], lambda r: r["rollout"][arm_a][e]["parsed"])
    b = matrix_of([p[1] for p in pairs], lambda r: r["rollout"][arm_b][e]["parsed"])
    return boot(np.log(a) - np.log(b))


def fidelity_table(output):
    out = {}
    for bench in BENCHES:
        parsed = load_parsed(output, bench)
        for split in sorted({v["split"] for v in parsed.values()}):
            rows = [v["fidelity"] for v in parsed.values() if v["split"] == split]
            tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
            out[f"{bench}/{split}"] = {
                "clips": len(rows), "presence_precision": tp / max(1, tp + fp), "presence_recall": tp / max(1, tp + fn),
                "presence_f1": 2 * tp / max(1, 2 * tp + fp + fn),
                "center_error_px": sum(r["center_error_sum"] for r in rows) / max(1, sum(r["center_count"] for r in rows)),
                "kind_accuracy": sum(r["kind_correct"] for r in rows) / max(1, sum(r["kind_count"] for r in rows))}
    return out


def compute_summary(output):
    ledger = read_json(Path(output) / "ledger.json")
    phases = {}
    for row in ledger["phases"]:
        stage = row["phase"].split("-")[0]
        if stage == "publish":  # written after compute.json; excluded so --validate regenerates it
            continue
        bucket = phases.setdefault(stage, {"wall_seconds_sum": 0.0, "gpu_seconds": 0.0, "entries": 0})
        bucket["wall_seconds_sum"] += row["wall_seconds"]
        bucket["gpu_seconds"] += row["gpu_seconds"]
        bucket["entries"] += 1
    return {"schema": SCHEMA["compute"], "phases": {k: {kk: round(vv, 1) if isinstance(vv, float) else vv
                                                        for kk, vv in v.items()} for k, v in sorted(phases.items())},
            "gpu_seconds_total": round(sum(v["gpu_seconds"] for v in phases.values()), 1),
            "engine_seconds": 0, "hardware": "4 x RTX 5090 (32 GB); 48 CPU threads",
            "note": "GPU seconds = wall seconds of the GPU-holding worker process; manifest / prepare are CPU-only"}


def build_publication(output):
    output = Path(output)
    plan = load_plan(output, check=False)
    records = collect(output)
    guard = guards(output, plan, records)
    failed = [k for k, v in guard.items() if not v["passed"]]
    tables = {"rollout": {b: {f: rollout_table(records, b, f) for f in FAMILIES} for b in BENCHES},
              "decisions": {b: {f: decision_table(records, b, f) for f in FAMILIES} for b in BENCHES}}
    primary_metric = {"clevrer": "option", "physion": "readout"}
    contrasts, tokens = {}, {}
    for bench in BENCHES:
        m = primary_metric[bench]
        c = {"J - J-cont": contrast(records, bench, "hybrid", "J", "hybrid", "J-cont", m)}
        for arm in REQUESTS:
            c[f"J - {arm}"] = contrast(records, bench, "hybrid", "J", "hybrid", arm, m)
        if bench == "clevrer":
            for metric in ("predictive", "counterfactual_option"):
                c[f"J - J-cont ({metric})"] = contrast(records, bench, "hybrid", "J", "hybrid", "J-cont", metric)
        else:
            c["J(label-free) - J(hybrid)"] = contrast(records, bench, "label-free", "J", "hybrid", "J", m)
            c["J - J-cont (predicate rule)"] = contrast(records, bench, "hybrid", "J", "hybrid", "J-cont",
                                                        "predicate_rule")
        c["log rollout J(label-free) / J(hybrid), last endpoint"] = rollout_contrast(
            records, bench, "label-free", "J", "hybrid", "J", ENDPOINTS[bench][-1])
        c["log rollout J / J-cont, last endpoint"] = rollout_contrast(records, bench, "hybrid", "J", "hybrid", "J-cont",
                                                                      ENDPOINTS[bench][-1])
        contrasts[bench] = c
        tokens[bench] = {k: (token(v) if not k.startswith("log rollout") else "DESCRIPTIVE")
                         for k, v in c.items()}
        if failed:
            tokens[bench] = {k: "readiness_or_precision_insufficient" for k in tokens[bench]}
    summary = {"schema": SCHEMA["summary"], "plan_identity": IDENTITY, "guards": guard, "failed_guards": failed,
               "tables": tables, "contrasts": contrasts, "tokens": tokens,
               "primary": {b: tokens[b]["J - J-cont"] for b in BENCHES},
               "fidelity": fidelity_table(output),
               "records": len(records),
               "amendments": (read_json(output / "amendments.json")["amendments"]
                              if (output / "amendments.json").is_file() else []),
               "exploratory": exploratory_tables(output) if (output / "exploratory").is_dir() else None}
    compute = compute_summary(output)
    return summary, compute, findings(summary, compute, plan)


def findings(summary, compute, plan):
    lines = ["# Issue #106 findings (generated by scripts/run_pixel_benchmarks.py --mode publish)", ""]
    if summary["failed_guards"]:
        lines += [f"**Guard failure: {', '.join(summary['failed_guards'])}. Every token is "
                  "readiness_or_precision_insufficient; no contrast is interpretable.**", ""]
    lines += ["## Primary", ""]
    for bench in BENCHES:
        c = summary["contrasts"][bench]["J - J-cont"]
        lines.append(f"- {bench}: J − J-cont {fmt(c)} → `{summary['tokens'][bench]['J - J-cont']}`")
    for family in FAMILIES:
        lines += ["", f"## Rollout error, {family} (position MSE vs the parsed carrier; DESCRIPTIVE seed x clip "
                      "intervals; annotation-truth MSE in summary.json)", ""]
        for bench in BENCHES:
            table = summary["tables"]["rollout"][bench][family]
            if not table:
                continue
            endpoints = [str(e) for e in ENDPOINTS[bench]]
            lines += [f"### {bench}", "", "| arm | " + " | ".join(f"t={e}" for e in endpoints) + " |",
                      "|---|" + "---|" * len(endpoints)]
            for arm in ARMS:
                lines.append(f"| {arm} | " + " | ".join(fmt_sci(table[arm][e]["parsed"]) for e in endpoints) + " |")
            lines.append("")
    for family in FAMILIES:
        lines += [f"## Downstream accuracy vs MACs, {family} (DESCRIPTIVE seed x item intervals)", ""]
        for bench in BENCHES:
            table = summary["tables"]["decisions"][bench][family]
            if not table:
                lines += [f"### {bench}: unsupported for this family", ""]
                continue
            metrics = ("option", "predictive", "counterfactual_option") if bench == "clevrer" else ("readout",
                                                                                                    "predicate_rule")
            lines += [f"### {bench}", "", "| arm | " + " | ".join(metrics) + " | MACs per decision |",
                      "|---|" + "---|" * (len(metrics) + 1)]
            for arm, row in table.items():
                lines.append(f"| {arm} | " + " | ".join(fmt(row.get(m)) for m in metrics)
                             + f" | {row['macs_per_decision']:.3e} |")
            lines.append("")
    lines += ["## Contrasts (frozen rule, margin 0.02)", ""]
    for bench in BENCHES:
        for name, stat in summary["contrasts"][bench].items():
            lines.append(f"- {bench}: {name}: {fmt(stat, 4)} → `{summary['tokens'][bench][name]}`")
    lines += ["", "## Parser fidelity (pixels vs annotation targets)", "",
              "| benchmark / split | clips | presence F1 | center error (native px) | kind accuracy |",
              "|---|---|---|---|---|"]
    for key, row in summary["fidelity"].items():
        lines.append(f"| {key} | {row['clips']} | {row['presence_f1']:.4f} | {row['center_error_px']:.1f} | "
                     f"{row['kind_accuracy']:.4f} |")
    explore_ = summary.get("exploratory")
    if explore_:
        lines += ["", "## EXPLORATORY readouts (post-freeze, amendment A2; no disposition)", "", explore_["status"], ""]
        if explore_["clevrer"]:
            lines += ["### CLEVRER (hybrid)", "", "| arm | option | predictive | counterfactual option |",
                      "|---|---|---|---|"]
            for arm, row in explore_["clevrer"]["table"].items():
                lines.append(f"| {arm} | {fmt(row['option'])} | {fmt(row['predictive'])} | "
                             f"{fmt(row['counterfactual_option'])} |")
            lines += [""] + [f"- {name}: option {fmt(c['option'], 4)}; predictive {fmt(c['predictive'], 4)}; "
                             f"counterfactual option {fmt(c['counterfactual_option'], 4)}"
                             for name, c in explore_["clevrer"]["contrasts"].items()] + [""]
        for family, block in explore_["physion"].items():
            lines += [f"### Physion OCP predicate score, readout-fitted threshold ({family})", "",
                      "| arm | test accuracy |", "|---|---|"]
            lines += [f"| {arm} | {fmt(stat)} |" for arm, stat in block["table"].items()]
            lines += [""] + [f"- {name}: {fmt(c, 4)}" for name, c in block["contrasts"].items()] + [""]
    if summary["amendments"]:
        lines += ["", "## Post-freeze amendments", ""]
        lines += [f"- {a['id']}: {a['change']}" for a in summary["amendments"]]
    lines += ["", f"Compute: {compute['gpu_seconds_total']:.0f} GPU s; engine seconds 0.", ""]
    return "\n".join(lines)


def manifest_records(output):
    output = Path(output)
    files = sorted(list((output / "records").glob("*.json")) + list((output / "training").glob("*.json"))
                   + list((output / "models").rglob("*.json")) + list((output / "exploratory").glob("*.json")))
    return {"schema": SCHEMA["manifest"], "files": {str(p.relative_to(output)): sha256_of(p) for p in files}}


def publish(output):
    output = Path(output)
    began = time.monotonic()
    summary, compute, text = build_publication(output)
    write_json(output / "summary.json", summary)
    write_json(output / "compute.json", compute)
    (output / "findings.md").write_text(text)
    write_json(output / "records-manifest.json", manifest_records(output))
    mark(output, "publish", time.monotonic() - began, 0)
    log(f"published: primary {summary['primary']}; failed guards {summary['failed_guards']}")
    return 0


# =========================================================================== spot check (G5) and validate


def spot(output):
    """G5: re-parse SPOT clips of every scored split from pixels and compare with the cached carriers; re-score
    the first SPOT clips of one member's rollout errors and compare with the record."""
    output = Path(output)
    plan = load_plan(output)
    began = time.monotonic()
    result = {"reparse": {}, "rescore": {}}
    with harness.deterministic_scoring():
        for bench in BENCHES:
            model = load_encoder(output, bench)
            parsed = load_parsed(output, bench)
            if bench == "clevrer":
                keys = plan["membership"]["clevrer"]["validation"]["scenes"][:SPOT["clips"]]
                frames = [pb.decode_clevrer_video(k) for k in keys]
            else:
                paths = plan["membership"]["physion"]["testing"]["trials"][:SPOT["clips"]]
                keys = [f"testing/{Path(p).stem}" for p in paths]
                frames = [_physion_images(p) for p in paths]
            diff = 0.0
            for k, f in zip(keys, frames):
                carrier, *_ = parse_clip(model, bench, f)
                diff = max(diff, float((carrier - parsed[k]["carrier"]).abs().max()))
            result["reparse"][bench] = {"clips": len(keys), "max_abs_difference": diff}
            labels = load_labels(output, bench)
            seed = SEEDS[0]
            record = read_json(record_path(output, bench, "hybrid", seed))
            again = score_cell(output, plan, bench, "hybrid", seed, parsed, labels, None, clip_limit=SPOT["clips"])
            worst = 0.0
            for arm in ARMS:
                for e in map(str, ENDPOINTS[bench]):
                    a = np.array([np.nan if v is None else v for v in again["rollout"][arm][e]["parsed"]])
                    b = np.array([np.nan if v is None else v for v in record["rollout"][arm][e]["parsed"][:SPOT["clips"]]])
                    both = np.isfinite(a) & np.isfinite(b)
                    if (np.isfinite(a) != np.isfinite(b)).any():
                        worst = float("inf")
                    elif both.any():
                        worst = max(worst, float(np.abs(a[both] - b[both]).max()))
            result["rescore"][bench] = {"clips": SPOT["clips"], "max_abs_difference": worst}
    result["passed"] = (all(v["max_abs_difference"] <= 1e-4 for v in result["reparse"].values())
                        and all(v["max_abs_difference"] <= 1e-6 for v in result["rescore"].values()))
    result["checked_at"] = utc_now()
    write_json(output / "spot.json", result)
    wall = time.monotonic() - began
    mark(output, "spot", wall, wall)
    log(f"spot: {result}")
    return 0


def validate(output):
    output = Path(output)
    load_plan(output)
    with harness.deterministic_scoring():
        summary, compute, text = build_publication(output)
    expected = {"summary.json": dumps(summary), "findings.md": text,
                "records-manifest.json": dumps(manifest_records(output))}
    bad = [name for name, content in expected.items() if (output / name).read_text() != content]
    expected["compute.json"] = dumps(compute)
    bad += ["compute.json"] if (output / "compute.json").read_text() != dumps(compute) else []
    spot_ = read_json(output / "spot.json")
    if not spot_["passed"]:
        bad.append("spot.json (G5)")
    if bad:
        log(f"validate FAILED: {bad}")
        return 1
    log("validate: published files regenerate byte for byte; spot check passed")
    return 0


# =========================================================================== smoke


def smoke(output):
    """Pre-freeze checks on real data with tiny budgets; writes smoke.json (refuses after the freeze)."""
    output = Path(output)
    if (output / "plan.json").is_file():
        raise SystemExit("smoke refused after the freeze")
    output.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    out = {"schema": SCHEMA["smoke"], "identity": IDENTITY, "started_at": utc_now()}
    # --- CLEVRER homography from training videos (annotation-side; frozen into the plan)
    payloads = []
    for index, payload in pb.clevrer_annotations("train"):
        if pb.clevrer_member(payload):
            payloads.append((index, payload))
        if len(payloads) >= 300:
            break
    with spawn_pool(44) as pool:
        points = list(pool.map(_correspondences, payloads, chunksize=2))
    homography, stats = pb.fit_homography(np.concatenate(points))
    out["homography"] = {"matrix": homography.tolist(), "fit": stats, "videos": [i for i, _ in payloads],
                         "rule": "palette-hue blob centroids of uniquely coloured in-view objects, every 8th frame "
                                 "of the first 300 member training videos; RANSAC 4-point DLT (6 px) then "
                                 "least squares on the inliers"}
    log(f"homography: {stats}")
    # --- label base rates
    rates = {}
    rows = [clevrer_labels((i, p, homography)) for i, p in payloads[:200]]
    rates["clevrer"] = label_rates(rows, pb.CLEVRER_FRAMES)
    smoke_paths = sorted(str(p.relative_to(pb.ROOT)) for p in (pb.PHYSION_EXTRACTED / "dynamics").rglob("*.hdf5"))[:60]
    headers = [_physion_header(pb.ROOT / p) for p in smoke_paths]
    smoke_paths = [p for p, h in zip(smoke_paths, headers) if not h["error"] and h["native_frames"] >= pb.PHYSION_MIN_NATIVE]
    with spawn_pool(30) as pool:
        prow = list(pool.map(physion_labels, smoke_paths))
    rates["physion"] = label_rates(prow, None)
    rates["physion"]["ocp_positive"] = float(np.mean([r["ocp"] for r in prow]))
    out["label_rates"] = rates
    # --- parser: short training on a few clips, parse, carriers
    torch.backends.cudnn.benchmark = False
    timings = {}
    h_frames = np.stack([pb.decode_clevrer_video(i)[::8] for i, _ in payloads[:24]]).reshape(-1, *pb.CLEVRER_IMAGE, 3)
    h_targets = {k: np.concatenate([r[k][::8] for r in rows[:24]]) for k in ("presence", "centers", "center_valid",
                                                                             "kinds")}
    h_targets["kinds"] = h_targets["kinds"].astype(np.int64)
    t0 = time.monotonic()
    parser, report = train_encoder(output, "clevrer", steps_limit=40, frames=h_frames, targets=h_targets)
    timings["clevrer_encoder_step_seconds"] = (time.monotonic() - t0) / 40
    t0 = time.monotonic()
    parsed_c = {}
    for i, _ in payloads[200:216]:
        carrier, presence, kinds, centers = parse_clip(parser, "clevrer", pb.decode_clevrer_video(i))
        parsed_c[i] = {"carrier": carrier, "presence": presence, "kinds": kinds, "split": "train"}
    timings["clevrer_parse_seconds_per_video"] = (time.monotonic() - t0) / 16
    p_images = [_physion_images(p) for p in smoke_paths[:12]]
    p_frames = np.concatenate([im[::2] for im in p_images])
    p_targets = {k: np.concatenate([r[k][::2] for r in prow[:12]]) for k in ("presence", "centers", "center_valid",
                                                                             "kinds")}
    p_targets["kinds"] = p_targets["kinds"].astype(np.int64)
    t0 = time.monotonic()
    pparser, _ = train_encoder(output, "physion", steps_limit=40, frames=p_frames, targets=p_targets)
    timings["physion_encoder_step_seconds"] = (time.monotonic() - t0) / 40
    parsed_p = {}
    t0 = time.monotonic()
    for path, row in zip(smoke_paths[:40], prow[:40]):
        carrier, presence, kinds, centers = parse_clip(pparser, "physion", _physion_images(path))
        parsed_p[row["name"]] = {"carrier": carrier, "presence": presence, "kinds": kinds, "split": "dynamics"}
    timings["physion_parse_seconds_per_trial"] = (time.monotonic() - t0) / len(parsed_p)
    checks = {}
    # carrier builder equals the deployment adapter's per-slot semantics on a hand case
    checks["carrier_shape"] = all(v["carrier"].shape[1] == DIM for v in list(parsed_c.values()) + list(parsed_p.values()))
    # position_mse equals clevrer_boundary.field_errors
    from world_model.training.clevrer_boundary import field_errors
    a, b = parsed_c[payloads[200][0]]["carrier"][10:12], parsed_c[payloads[200][0]]["carrier"][20:22]
    reference = [field_errors(a[k], b[k])["position_mse"] for k in range(2)]
    mine = position_mse(a, b).tolist()
    checks["position_mse_equals_field_errors"] = all((x is None and not math.isfinite(y)) or abs(x - y) < 1e-6
                                                     for x, y in zip(reference, mine))
    # --- predictor: both families, two members, every request, LC terms
    smoke_labels = {"clevrer": {i: clevrer_labels((i, p, homography)) for i, p in payloads[200:216]},
                    "physion": {r["name"]: r for r in prow[:40]}}
    for bench, parsed in (("clevrer", parsed_c), ("physion", parsed_p)):
        labels = smoke_labels[bench]
        data = PixelData(bench, parsed, labels)
        members = {}
        for family in FAMILIES:
            ensemble, log_rows, wall, _ = train_family(output / "smoke-scratch", bench, family, data, seeds=SEEDS[:2],
                                                    steps=27)
            timings[f"{bench}_{family}_update_seconds"] = wall / 27
            checks[f"{bench}_{family}_finite"] = bool(not ensemble.retired.any())
            members[family] = ensemble.member_state(0)
        cls, _ = pc.FAMILIES["hybrid"]
        model = cls().to(DEVICE)
        model.load_state_dict(members["hybrid"])
        model.eval()
        units = policy_windows({k: v["carrier"] for k, v in parsed.items()}, SEEDS[0])
        units = [u[:4] for u in units[:2]]
        t0 = time.monotonic()
        with harness.deterministic_scoring():
            original = dict(POLICY)
            POLICY["steps_per_round"] = 20
            try:
                j, _ = train_policy(model, "J", SEEDS[0], units)
                jc, _ = train_policy(model, "J-cont", SEEDS[0], units)
            finally:
                POLICY.update(original)
        timings[f"{bench}_policy_seconds_8_windows"] = time.monotonic() - t0
        policies = {"J": j, "J-cont": jc}
        clips = [v["carrier"] for v in parsed.values()]
        annotations = [torch.as_tensor(labels[k]["annotation"]) for k in parsed]
        t0 = time.monotonic()
        with harness.deterministic_scoring():
            errors = rollout_errors(model, policies, clips, annotations, bench)
            again = rollout_errors(model, policies, clips, annotations, bench)
        timings[f"{bench}_rollout_error_seconds_per_clip"] = (time.monotonic() - t0) / (2 * len(clips))
        checks[f"{bench}_rollout_deterministic"] = errors == again
        checks[f"{bench}_jcont_continuous_only"] = True
        z0 = torch.stack([c[CONTEXTS[bench][0]] for c in clips]).to(DEVICE)
        _, _, timing, landings = pc.rollout(model, jc, z0, ENDPOINTS[bench][-1], {"kind": "policy"}, record=True)
        checks[f"{bench}_jcont_continuous_only"] = all(int(c) in R_CONT for _, _, _, ch in landings for c in ch)
        # rollout() equals request_policy.rollout_endpoints for a shared initial carrier
        z_rp, _ = rp.rollout_endpoints(model, j, z0[0], torch.zeros(3, 5, device=DEVICE), {"kind": "policy"},
                                       select=lambda kind, logits, spec: logits.argmax(-1),
                                       endpoint=ENDPOINTS[bench][-1])
        z_me, _, _, _ = pc.rollout(model, j, z0[:1].expand(3, -1), ENDPOINTS[bench][-1], {"kind": "policy"})
        checks[f"{bench}_rollout_equals_request_policy"] = bool(torch.equal(z_rp, z_me))
        if bench == "clevrer":
            questions = json.loads(pb.CLEVRER_QUESTIONS["train"].read_text())
            qmap = {int(v["scene_index"]): [q for q in v["questions"] if q["question_type"] in ("predictive",
                                                                                               "counterfactual")]
                    for v in questions if int(v["scene_index"]) in parsed}
            fake_plan = {"membership": {"clevrer": {"validation": {"scenes": list(parsed)}}}}
            t0 = time.monotonic()
            with harness.deterministic_scoring():
                qa = qa_scores(model, policies, fake_plan, parsed, qmap)
            timings["clevrer_qa_seconds_per_video"] = (time.monotonic() - t0) / len(parsed)
            checks["clevrer_qa_runs"] = all(len(v["predictive"]) + len(v["counterfactual_option"]) > 0 for v in qa.values())
            resolved = [pc.choice_pair(c, pc.slot_attributes(parsed[v]["presence"].float(), parsed[v]["kinds"].float()))
                        for v in qmap for q in qmap[v] for c in q["choices"]]
            out["clevrer_smoke_resolution_rate_untrained_parser"] = float(np.mean([r is not None for r in resolved]))
        else:
            keys = list(parsed)
            half = len(keys) // 2
            role = {k: ("readout" if i < half else "testing") for i, k in enumerate(keys)}
            parsed_ocp = {f"{role[k]}/{k}": parsed[k] for k in keys}
            fake_labels = {s: {f"{s}/{k}": labels[k] for k in keys if role[k] == s} for s in ("readout", "testing")}
            fake_plan = {"membership": {"physion": {s: {"trials": [k + ".hdf5" for k in keys if role[k] == s]}
                                                    for s in ("readout", "testing")}}}
            single_class = len({r["ocp"] for r in fake_labels["readout"].values()}) < 2
            if not single_class:
                t0 = time.monotonic()
                with harness.deterministic_scoring():
                    ocp = ocp_scores(model, policies, fake_plan, parsed_ocp, fake_labels, SEEDS[0])
                timings["physion_ocp_seconds"] = time.monotonic() - t0
                checks["physion_ocp_runs"] = all(len(v["readout"]) == len(keys) - half for k, v in ocp.items()
                                                 if k != "labels")
            else:
                checks["physion_ocp_runs"] = False
    out["checks"] = checks
    out["timings"] = timings
    out["projection"] = projection(timings)
    out["passed"] = all(checks.values())
    out["wall_seconds"] = time.monotonic() - began
    out["finished_at"] = utc_now()
    write_json(output / "smoke.json", out)
    log(f"smoke passed={out['passed']} checks={checks}")
    log(f"projection: {out['projection']}")
    return 0 if out["passed"] else 1


def _correspondences(job):
    index, payload = job
    scene = pb.clevrer_scene(payload)
    frames = pb.decode_clevrer_video(index, pb.CLEVRER_NATIVE)
    return pb.color_correspondences(frames, scene)


def label_rates(rows, frames):
    micro, macro, avail = [], [], []
    for r in rows:
        f = frames or r["frames"]
        events = np.asarray(r["contact"])
        positive_frames = len({int(e[0]) for e in events if e[0] < f})
        micro.append(positive_frames / f)
        m, a = np.asarray(r["macros"])[:f], np.asarray(r["macro_available"])[:f]
        macro.append(np.where(a, m, False).sum(0) / np.maximum(1, a.sum(0)))
        avail.append(a.mean(0))
    return {"micro_frames_with_contact": float(np.mean(micro)),
            "macro_positive_rate_[predicate][delta]": np.mean(macro, 0).round(4).tolist(),
            "macro_available_rate_[predicate][delta]": np.mean(avail, 0).round(4).tolist(), "clips": len(rows)}


def projection(t):
    """Projected wall hours on 4 GPUs from the smoke timings (full membership sizes are approximate before
    the freeze: CLEVRER ~5 900 train + ~2 950 validation videos, Physion ~2 000 + 1 000 + 1 000 trials)."""
    clevrer_frames = 5900 * 16
    physion_frames = 2000 * 45
    encoder = max(pc.ENCODER["epochs"] * clevrer_frames / 64 * t["clevrer_encoder_step_seconds"],
                  pc.ENCODER["epochs"] * physion_frames / 64 * t["physion_encoder_step_seconds"])
    parse = (8850 * t["clevrer_parse_seconds_per_video"] + 4000 * t["physion_parse_seconds_per_trial"]) / 4
    # the first smoke ensembles include torch.compile of every request graph; the last one is steady state
    steady = min(t[f"{b}_{f}_update_seconds"] for b in BENCHES for f in FAMILIES)
    train = TRAINING["steps"] * steady * 2.5 + 1800
    policy = 40 * (t["clevrer_policy_seconds_8_windows"] + t["physion_policy_seconds_8_windows"]) / 2 * 128 / 4
    score = (10 * 2950 * t["clevrer_rollout_error_seconds_per_clip"] * 5
             + 5 * 2950 * t.get("clevrer_qa_seconds_per_video", 0.1)
             + 10 * 1000 * t["physion_rollout_error_seconds_per_clip"] * 5 + 10 * t.get("physion_ocp_seconds", 60) * 20) / 4
    hours = {"encoder": encoder / 3600, "parse": parse / 3600, "train": train / 3600, "policy": policy / 3600,
             "score": score / 3600}
    hours["total"] = sum(hours.values())
    return {k: round(v, 2) for k, v in hours.items()}


# =========================================================================== main

MODES = {"manifest", "smoke", "freeze", "prepare", "encoder", "parse", "train", "policy", "score", "spot", "publish",
         "validate", "explore"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=sorted(MODES))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker", type=int, default=0)
    args = parser.parse_args(argv)
    mode = "validate" if args.validate else args.mode
    if mode is None:
        parser.error("--mode or --validate is required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if mode == "manifest":
        return manifest(output)
    if mode == "smoke":
        return smoke(output)
    if mode == "freeze":
        return freeze(output)
    if mode == "publish":
        return publish(output)
    if mode == "validate":
        return validate(output)
    if mode == "spot":
        return spot(output)
    return {"prepare": prepare, "encoder": encoder, "parse": parse_stage, "train": train, "policy": policy_stage,
            "score": score, "explore": explore}[mode](output, args.workers, args.worker)


if __name__ == "__main__":
    raise SystemExit(main())
