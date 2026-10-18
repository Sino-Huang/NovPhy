"""Issue-115: predicate-feedback robustness, scored once on the #104 sealed #109 evaluation split.

Binding runner module: scripts/run_predicate_feedback_robustness.py
Exact validation command: python -u -m scripts.run_predicate_feedback_robustness --validate

The CNN hybrid's description requests read their own predicate heads at runtime (``symbolic_features`` of
the current carrier through ``micro_adapter`` / ``macro_adapter``); engine labels enter only through the
symbolic training loss. This runner measures how much the description axis depends on (a) the feedback
quality at runtime and (b) engine supervision of the heads:

1. inference-time interventions on the frozen predictors (micro: the Gate-B LC hybrid; macro: the LC hybrid
   retrained with the #113 observable macro predicates, ``ma-ref``): default, oracle (engine labels of the
   executed branch at the matching frame), symmetric flips of the oracle at p in {0.05, 0.1, 0.2, 0.4},
   Gaussian noise on the head logits calibrated on the fit split to 0.8 / 0.6 / 0.4 / 0.2 of the default
   feedback F1, removed (zero adapter input), shuffled (another candidate's feedback); macro adds the #113
   temporal head (K4-mix50) as the feedback source;
2. retrains under the #112 v3 LC recipe (matched parameters and updates, 3 seeds) with engine labels of
   the description flipped at the item-1 rates;
3. low-label (10 / 25 % of fit levels), a learned perception head (trained on the 25 % levels, frozen, its
   readings as targets) and a label-free discrete bottleneck (binary codes, no symbolic loss);
4. items 1-3 on macro with the #113 predicates (Delta-indexed macro head) and temporal head;
5. the no-model action prior and the wrong-anchor control in every ranking table.

Every predictor gets its own ICLR request policy J (#101 pe DP, lambda 1e-4, the #109 policy pool); T is the
horizon-only restriction of J (T-continuous, #114) and D the description-only restriction at J's modal Delta
on the policy split (#96 rule). Scored under ``deterministic_scoring()`` in processes that did not train:
tie-free AUC per inventory on the Gate-B cells, the #100 engine_position_mse at t = 225 per request (grid),
and the #100 readout counts of the delivered feedback and of the heads.

Modes: --dry-run --smoke --prepare --truth --heads --train --calibrate --score --publish --validate.
--heads/--train/--calibrate/--score take --workers N --worker K (one process per GPU; CUDA_VISIBLE_DEVICES
set by the caller). Zero engine seconds. Intervals: level-clustered percentile bootstrap (10000 draws,
PCG64 7201), DESCRIPTIVE; delta = 0.02.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import functools
import hashlib
import importlib.util
import io
import math
import multiprocessing
from pathlib import Path
import shutil
import time
import types

import numpy as np
import torch

from scripts import post_freeze_amendments as amendments
from scripts import run_decision_aware_policy as p101
from scripts import run_decision_chain_attribution as m99
from scripts import run_decision_dynamics_controls as c2
from scripts import run_decision_dynamics_retrain as rt
from scripts import run_decision_dynamics_versions as versions
from scripts import run_decision_relevant_dynamics as v1
from scripts import run_gate_b as gb
from scripts import run_observable_macro as om113
from scripts import run_slot_encoder_fix as s99
from scripts import run_tau_ad_within_checkpoint as t96
from world_model.data import observable_macro as om
from world_model.model import Abstraction, PredictionPair
from world_model.planning import scoring_harness as harness
from world_model.training import predicate_feedback as pf
from world_model.training import request_policy as rp
from world_model.training import request_selectors as rs
from world_model.training import temporal_macro_head as tmh
from world_model.training.cnn_hybrid import PAIRS, SLOTS

ROOT = v1.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-115-predicate-feedback-v1"
GATE_B = gb.OUTPUT
RETRAIN = rt.OUTPUT
POLICY = p101.OUTPUT
MACRO113 = ROOT / ".local-artifacts/issue-113-temporal-macro-head-v1"

IDENTITY = "issue-115-predicate-feedback-v1"
SCHEMA_PLAN = "issue_115_predicate_feedback_plan_v1"
SCHEMA_SMOKE = "issue_115_predicate_feedback_smoke_v1"
SCHEMA_PREDICTOR = "issue_115_predictor_v1"
SCHEMA_GROUP = "issue_115_training_group_v1"
SCHEMA_UPDATES = "issue_115_updates_v1"
SCHEMA_TRAINED = "issue_115_trained_v1"
SCHEMA_POLICY = "issue_115_request_policy_v1"
SCHEMA_HEAD = "issue_115_perception_head_v1"
SCHEMA_PSEUDO = "issue_115_pseudo_macro_labels_v1"
SCHEMA_TRUTH = "issue_115_engine_truth_v1"
SCHEMA_CALIBRATION = "issue_115_calibration_v1"
SCHEMA_RECORD = "issue_115_record_v1"
SCHEMA_READOUT = "issue_115_readout_v1"
SCHEMA_COMPUTE = "issue_115_compute_v1"
SCHEMA_REPORT = "issue_115_report_v1"
SCHEMA_MANIFEST = "issue_115_records_manifest_v1"
SCHEMA_HANDOFF = "issue_115_handoff_106_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_predicate_feedback_robustness --validate"
RUNNER = "scripts/run_predicate_feedback_robustness.py"
MODULE = "world_model/training/predicate_feedback.py"

DEVICE = "cuda"
ARM = rt.ARM  # the #112 v3 LC recipe
SEEDS = tuple(t96.SEEDS)
INVENTORIES = gb.INVENTORIES
PRIMARY = gb.GATE_INVENTORY
PRIMARY_COST = harness.PRIMARY_COST
DELTA = v1.DELTA
STOP_TOKEN = v1.STOP_TOKEN
NONFINITE_SHARE_MAX = v1.NONFINITE_SHARE_MAX
ENDPOINT = t96.ENDPOINT
DELTAS = pf.DELTAS
STEPS = rt.STEPS
SKIP_CAP = rt.SKIP_CAP
ICLR_LAMBDA = p101.ICLR_LAMBDA
FLIPS = (0.05, 0.1, 0.2, 0.4)
LOW_K = (10, 25)
HEAD_K = 25
LABEL_SEED = 11501
GAUSS_FRACTIONS = (0.8, 0.6, 0.4, 0.2)
CAL_CELLS = 24
CAL_ITERATIONS = 12
THRESHOLDS = om113.THRESHOLDS
HEAD_TRAIN = {"steps": 6000, "batch": 4096, "lr": 1e-3, "weight_decay": 1e-4, "schedule": "cosine to 0",
              "optimizer": "AdamW", "threshold_rows": 200000}
MACRO_HEAD = {"history": 4, "mix": 0.5, "recipe": "the #113 train_head recipe (selected K4-mix50)"}
TEMPORAL_FEEDBACK = "K4-mix50"
TRUTH_WORKERS = 40
SMOKE_STEPS = 12
SMOKE_CELLS = 3
BOOTSTRAP_SEED = 7201
BOOTSTRAP_DRAWS = 10000
GPU_CAP_SECONDS = 80 * 3600.0
WALL_CAP_SECONDS = 120 * 3600.0  # summed over processes
ELAPSED_LIMIT_HOURS = 48.0       # the owner's limit on 4 GPUs: beyond it the ticket finishes blocked

read_json = v1.read_json
write_json = v1.write_json
json_text = v1.json_text
sha256_of = v1.sha256_of
atomic_torch = v1.atomic_torch
fmt_interval = v1.fmt_interval


def log(message):
    print(f"[issue-115] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# predictors, interventions, arms
# ---------------------------------------------------------------------------

def tag(p):
    return f"{round(p * 100):02d}"


def _predictors():
    out = {"gateb": {"description": "micro", "class": "hybrid", "labels": ["engine"], "trained": False,
                     "text": "the Gate-B LC hybrid (handoff checkpoints, #112 v3 retrain on the #109 fit split)"}}
    for description, prefix, base_class, free_class in (("micro", "mi", "hybrid", "binary_micro"),
                                                        ("macro", "ma", "delta_macro", "binary_delta_macro")):
        if description == "macro":
            out["ma-ref"] = {"description": "macro", "class": base_class, "labels": ["engine"], "trained": True,
                             "text": "LC hybrid with the Delta-indexed macro head on the #113 observable predicates"}
        for p in FLIPS:
            out[f"{prefix}-flip{tag(p)}"] = {"description": description, "class": base_class, "labels": ["flip", p],
                                             "trained": True,
                                             "text": f"engine {description} labels with symmetric flips at p = {p}"}
        for k in LOW_K:
            out[f"{prefix}-low{k}"] = {"description": description, "class": base_class, "labels": ["low", k],
                                       "trained": True,
                                       "text": f"{description} heads supervised on the labels of {k} % of fit levels"}
        out[f"{prefix}-learned"] = {"description": description, "class": base_class, "labels": ["learned", HEAD_K],
                                    "trained": True,
                                    "text": (f"{description} targets = readings of a perception head trained on the "
                                             f"{HEAD_K} % labelled levels and frozen")}
        out[f"{prefix}-free"] = {"description": description, "class": free_class, "labels": ["none"],
                                 "trained": True,
                                 "text": f"label-free binary {description} bottleneck (no {description} symbolic loss)"}
    return out


PREDICTORS = _predictors()
TRAINED = tuple(name for name, entry in PREDICTORS.items() if entry["trained"])
REFERENCE = {"micro": "gateb", "macro": "ma-ref"}
PREFIX = {"micro": "mi", "macro": "ma"}
ITEM2 = {d: (REFERENCE[d], *(f"{PREFIX[d]}-flip{tag(p)}" for p in FLIPS)) for d in REFERENCE}
ITEM3 = {d: (REFERENCE[d], *(f"{PREFIX[d]}-low{k}" for k in LOW_K), f"{PREFIX[d]}-learned", f"{PREFIX[d]}-free")
         for d in REFERENCE}
FREE = {d: f"{PREFIX[d]}-free" for d in REFERENCE}
GAUSS = tuple(f"gauss{round(100 * f)}" for f in GAUSS_FRACTIONS)
INTERVENTIONS = {"micro": ("default", "oracle", *(f"flip{tag(p)}" for p in FLIPS), *GAUSS, "removed", "shuffled")}
INTERVENTIONS["macro"] = (*INTERVENTIONS["micro"], "temporal")


def arms_for(description):
    return ("J", "T", "D", *(f"F-{d}-{description}" for d in DELTAS))


def fixed_arms(description):
    return tuple(f"F-{d}-{description}" for d in DELTAS)


SCHEDULED = ("J", "T", "D")
COST_WEIGHT = {"J": 3.0, "T": 2.0, "D": 3.0, "F-1": 3.0, "F-5": 1.0, "F-15": 0.3, "wrong": 1.0, "readout": 0.5}


def symbolic_spec(name):
    entry = PREDICTORS[name]
    kind, mode = entry["labels"][0], entry["description"]
    kwargs = {"macro_delta": entry["class"] in ("delta_macro", "binary_delta_macro")}
    if kind == "none":
        kwargs[mode] = "none"
    if kind == "low":
        kwargs["weighted"] = frozenset({mode})
    return pf.SymbolicSpec(**kwargs)


def feedback_parameters(fb):
    if fb.startswith("flip"):
        return {"kind": "flip", "p": FLIPS[[f"flip{tag(p)}" for p in FLIPS].index(fb)]}
    if fb.startswith("gauss"):
        return {"kind": "gauss", "fraction": GAUSS_FRACTIONS[GAUSS.index(fb)]}
    return {"kind": fb}


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def updates_path(output, name, seed):
    return Path(output) / "training" / f"updates--{name}--seed{seed}.pt"


def trained_path(output, name, seed):
    return Path(output) / "training" / f"trained--{name}--seed{seed}.json"


def policy_path(output, name, seed):
    return Path(output) / "models" / "policies" / name / f"seed-{seed}" / "policy.pt"


def policy_report_path(output, name, seed):
    return Path(output) / "models" / "policies" / name / f"seed-{seed}" / "report.json"


def head_path(output, kind, seed):
    return Path(output) / "models" / "heads" / f"{kind}--seed{seed}.pt"


def head_report_path(output, kind, seed):
    return Path(output) / "models" / "heads" / f"{kind}--seed{seed}.json"


def pseudo_path(output, seed):
    return Path(output) / "data" / f"pseudo-macro--seed{seed}.pt"


def truth_path(output):
    return Path(output) / "data" / "engine-truth.pt"


def calibration_path(output, name, seed):
    return Path(output) / "calibration" / f"{name}--seed{seed}.json"


def record_path(output, name, fb, arm, seed):
    return Path(output) / "records" / f"{name}--{fb}--{arm}--seed{seed}.json"


def wrong_path(output, name, seed):
    return Path(output) / "records" / f"wrong--{name}--J--seed{seed}.json"


def readout_path(output, name, seed):
    return Path(output) / "records" / f"readout--{name}--seed{seed}.json"


def ledger_path(output, phase):
    return Path(output) / "ledger" / f"{phase}.json"


def mark(output, phase, wall, gpu, **extra):
    path = ledger_path(output, phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    block = read_json(path) if path.is_file() else {"wall_seconds": 0.0, "gpu_seconds": 0.0}
    write_json(path, {**block, "wall_seconds": block["wall_seconds"] + wall, "gpu_seconds": block["gpu_seconds"] + gpu,
                      "last_at": utc_now(), **extra})
    total = [read_json(p) for p in sorted((Path(output) / "ledger").glob("*.json"))]
    if (sum(b["gpu_seconds"] for b in total) > GPU_CAP_SECONDS
            or sum(b["wall_seconds"] for b in total) > WALL_CAP_SECONDS):
        raise ValueError(f"{STOP_TOKEN}: compute cap exceeded")


def block_seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:15], 16)


# ---------------------------------------------------------------------------
# label variants on the #109 fit split (training payload of the #112 v3 retrain)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def labelled_levels(k):
    """Pre-declared nested subsets: per generator family, the sorted fit levels permuted by
    PCG64(LABEL_SEED); the first ceil(k % x n) of each family are labelled."""
    out = []
    by_family = {}
    for level in rt.fit_levels():
        by_family.setdefault(level["generator_family"], []).append(level["identity"])
    for family in sorted(by_family):
        members = sorted(by_family[family])
        order = np.random.Generator(np.random.PCG64(LABEL_SEED)).permutation(len(members))
        out += [members[i] for i in order[:math.ceil(k / 100 * len(members))].tolist()]
    return frozenset(out)


def position_members(payload):
    """[N] index of each payload position's shot."""
    sizes = payload["lasts"] + 1
    return torch.repeat_interleave(torch.arange(len(sizes)), sizes)


def labelled_positions(payload, k):
    subset = labelled_levels(k)
    flags = torch.tensor([m in subset for m in payload["shot_members"]])
    return flags[position_members(payload)]


@functools.lru_cache(maxsize=None)
def macro113(split):
    """The #113 frozen labels of every frame (values / available [rows, 2, 3]) with each shot's first row."""
    data = torch.load(MACRO113 / "data" / f"{split}.pt", map_location="cpu", weights_only=True, mmap=True)
    first = torch.nonzero(data["position"] == 0).flatten()
    starts = {key: int(data["start"][row]) for key, row in zip(data["keys"], first.tolist(), strict=True)}
    return {"values": data["values"].clone(), "available": data["available"].clone(), "starts": starts,
            "carriers": data["carriers"], "position": data["position"], "start": data["start"],
            "shot": data["shot"], "keys": data["keys"], "levels": data["levels"], "shot_level": data["shot_level"]}


def payload_rows113(payload):
    data = macro113("fit")
    return torch.cat([data["starts"][key] + torch.arange(int(last) + 1)
                      for key, last in zip(payload["shot_keys"], payload["lasts"].tolist(), strict=True)])


def macro_payload_labels(payload):
    rows = payload_rows113(payload)
    data = macro113("fit")
    return data["values"][rows], data["available"][rows]


def label_counts_micro(labels, mask, reference, reference_mask, chunk=1 << 18):
    """#100 counts of ``labels`` (as 0/1 readings) on ``mask`` positions against the engine reference."""
    total = torch.zeros(2, 3, dtype=torch.long)
    for begin in range(0, len(labels), chunk):
        part = slice(begin, begin + chunk)
        values = (labels[part] & mask[part]).to(DEVICE).float()
        total += pf.micro_counts(values, reference[part].to(DEVICE), reference_mask[part].to(DEVICE)).cpu()
    return total


def label_counts_macro(values, reference, available):
    counts = torch.zeros(len(DELTAS), 2, 3, dtype=torch.long)
    for d in range(len(DELTAS)):
        counts[d] = pf.macro_counts(values[..., d].float(), reference[..., d], available[..., d])
    return counts


def variant_symbols(output, name, seed, payload):
    """(symbols dict for VariantTrainData, label report) of one predictor arm and seed."""
    entry = PREDICTORS[name]
    kind, description = entry["labels"][0], entry["description"]
    relations, relation_mask = payload["relations"], payload["relation_mask"]
    symbols = {"relations": relations, "relations_mask": relation_mask}
    if description == "macro":
        values, available = macro_payload_labels(payload)
        symbols.update({"macros": values, "macros_mask": available})
    else:
        symbols.update({"macros": payload["macros"], "macros_mask": payload["macro_mask"]})
    report = {"kind": kind, "description": description}
    generator = torch.Generator().manual_seed(block_seed("labels", name, seed))
    if kind == "flip":
        p = entry["labels"][1]
        if description == "micro":
            flipped, tally = pf.flip_relations(relations, relation_mask, p, generator)
            symbols["relations"] = flipped
            counts = label_counts_micro(flipped, relation_mask, relations, relation_mask)
        else:
            flipped, tally = pf.flip_values(symbols["macros"], symbols["macros_mask"], p, generator)
            counts = label_counts_macro(flipped, symbols["macros"], symbols["macros_mask"])
            symbols["macros"] = flipped
        report.update({"p": p, **tally, "realized_rate": tally["flipped"] / tally["eligible"],
                       "label_counts": counts.tolist(), "label_f1": float(np.nanmean(pf.f1_from(counts).numpy()))})
    elif kind == "low":
        k = entry["labels"][1]
        flags = labelled_positions(payload, k)
        symbols[f"{description}_labelled"] = flags
        report.update({"k": k, "labelled_levels": len(labelled_levels(k)), "labelled_positions": int(flags.sum()),
                       "positions": len(flags)})
    elif kind == "learned":
        if description == "micro":
            head, info = load_head(output, "micro", seed)
            thresholds = info["thresholds"]
            learned, learned_mask = micro_pseudo_labels(head, payload["carriers"], thresholds)
            counts = label_counts_micro(learned, learned_mask, relations, relation_mask)
            symbols["relations"], symbols["relations_mask"] = learned, learned_mask
        else:
            store = torch.load(pseudo_path(output, seed), map_location="cpu", weights_only=True)
            if store.get("schema") != SCHEMA_PSEUDO or store["seed"] != seed:
                raise ValueError("pseudo macro labels binding differs")
            learned = store["values"]
            counts = label_counts_macro(learned, symbols["macros"], symbols["macros_mask"])
            symbols["macros"] = learned
            info = read_json(head_report_path(output, "macro", seed))
        report.update({"head_sha256": info.get("sha256"), "label_counts": counts.tolist(),
                       "label_f1": float(np.nanmean(pf.f1_from(counts).numpy()))})
    elif kind == "none":
        report["labels_read"] = False
    digest = hashlib.sha256()
    for key in sorted(symbols):
        digest.update(key.encode())
        digest.update(symbols[key].contiguous().numpy().data)
    report["symbols_sha256"] = "sha256:" + digest.hexdigest()
    return symbols, report


class VariantTrainData(rt.IndexedTrainData):
    """``rt.IndexedTrainData`` (windows gathered from per-position carriers) with given label tensors;
    extra per-position tensors (``<mode>_labelled``) are gathered with the windows."""

    def __init__(self, payload, symbols):
        super().__init__(payload, "continuous")
        self.hybrid = True
        self.symbols = {k: v.to(DEVICE) for k, v in symbols.items()}


# ---------------------------------------------------------------------------
# training: the #112 v3 LC recipe (rt.implementation) with the arm's predictor class and label use
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def implementation(name):
    """A private copy of the v1 runner carrying the frozen v3 spec exactly as ``rt.implementation()``
    (LC arm, v3 LONG/RANK, truncated recursion, skipping + logging ensemble, PAIRS[step % 9] schedule),
    bound to this output, with the arm's predictor class and ``pf.local_loss`` under its symbolic spec
    (DEFAULT_SPEC = decision_dynamics.local_loss op for op)."""
    v3 = versions.implementation(rt.VERSION)
    loader = importlib.util.spec_from_file_location(f"issue115_implementation_{name}", ROOT / v1.RUNNER)
    impl = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(impl)
    impl.OUTPUT = OUTPUT
    impl.IDENTITY = IDENTITY
    impl.SCHEMA_PREDICTOR = SCHEMA_PREDICTOR
    impl.SCHEMA_GROUP = SCHEMA_GROUP
    impl.SCHEMA_DATA = rt.SCHEMA_DATA
    impl.RUNNER = RUNNER
    impl.MODEL_MODULE = MODULE
    impl.ARMS = (ARM,)
    impl.ARM_DEFINITIONS = {ARM: v3.ARM_DEFINITIONS[ARM]}
    impl.LONG = v3.LONG
    impl.RANK = v3.RANK
    spec = symbolic_spec(name) if name in PREDICTORS else pf.DEFAULT_SPEC

    def logged_local_loss(ensemble, batch, pair):
        value = pf.local_loss(ensemble, batch, pair, spec)
        ensemble.log["local"].append(value.detach())
        return value

    impl.dd = types.SimpleNamespace(Ensemble=rt.logging_ensemble(v3.dd.Ensemble), local_loss=logged_local_loss,
                                    horizon_at=v3.dd.horizon_at, ramp_at=v3.dd.ramp_at,
                                    long_and_rank_loss=v3.dd.long_and_rank_loss)
    impl.step_pair = lambda family, step: v1.step_pair("hybrid", step)
    cls = pf.PREDICTOR_CLASSES[PREDICTORS[name]["class"] if name in PREDICTORS else "hybrid"]
    predictors = types.SimpleNamespace(**vars(s99))
    predictors.new_predictor = lambda family: cls()
    impl.s99 = predictors
    return impl


def member_path(output, name, seed):
    return implementation(name).member_path(output, ARM, name, "all", seed)


def payload_hash_ok():
    expected = read_json(RETRAIN / "compute.json")["data"]["training_data_sha256"]
    return sha256_of(rt.data_path(RETRAIN)) == expected


def train_member(output, name, seed, payload):
    impl = implementation(name)
    began = time.monotonic()
    symbols, label_report = variant_symbols(output, name, seed, payload)
    data = VariantTrainData(payload, symbols)
    del symbols
    report, ensemble = impl.train_group(output, ARM, name, SEEDS.index(seed), [("all", seed)], data)
    atomic_torch(updates_path(output, name, seed), {
        "schema": SCHEMA_UPDATES, "plan_identity": IDENTITY, "name": name, "seed": seed,
        "requests": rt.step_requests("hybrid"), "skipped": ensemble.skipped.cpu(), "logs": ensemble.log_tensors()})
    del ensemble, data
    torch.cuda.empty_cache()
    write_json(trained_path(output, name, seed), {
        "schema": SCHEMA_TRAINED, "plan_identity": IDENTITY, "name": name, "seed": seed, "trained_at": utc_now(),
        "gpu_seconds": report["gpu_seconds"], "parameters": report["parameters_per_member"],
        "labels": label_report, "sha256": sha256_of(member_path(output, name, seed))})
    mark(output, f"train-{name}-seed{seed}", time.monotonic() - began, report["gpu_seconds"])
    log(f"trained {name} seed {seed}: {report['gpu_seconds']:.0f}s")


def member_state(output, name, seed):
    """Training accounting and usability of one member (retrain rule: complete, at most SKIP_CAP skipped
    updates, no #117 end-of-training excursion)."""
    report = read_json(implementation(name).group_path(output, ARM, name, SEEDS.index(seed)))
    updates = torch.load(updates_path(output, name, seed), map_location="cpu", weights_only=True)
    trained = read_json(trained_path(output, name, seed))
    member = report["members"][0]
    local, norms = updates["logs"]["local"][:, 0].double(), updates["logs"]["grad_norm"][:, 0].double()
    requests = updates["requests"]
    end_state = rt.excursion(local, norms, requests) if member["complete"] else None
    skipped = int(updates["skipped"][0])
    usable = bool(member["complete"] and skipped <= SKIP_CAP and len(local) == STEPS and not end_state["excursion"])
    last = report["trace"][-1]
    return {"name": name, "seed": seed, "complete": member["complete"], "retired_at": member["retired_at"],
            "skipped": skipped, "logged_updates": len(local), "gpu_seconds": report["gpu_seconds"],
            "parameters": trained["parameters"], "peak_memory_gb": report["peak_memory_gb"],
            "final_trace": {k: last[k] for k in ("loss", "local", "long", "rank", "grad_norm")},
            "end_state": end_state, "usable": usable, "trained_at": trained["trained_at"], "sha256": trained["sha256"],
            "labels": trained["labels"],
            "typed_failure": None if usable else ("retired" if not member["complete"] else
                                                  "skip_cap" if skipped > SKIP_CAP else "final_checkpoint_excursion")}


def train_policy(output, name, seed, model):
    """The ICLR request policy J for this predictor: the #101 pe recipe at lambda 1e-4 on the #109 policy
    pool (two DAgger rounds of the #77 prediction-error DP labels, distillation), then the policy-split
    evaluation that fixes the D arm's Delta (#96 modal rule)."""
    began = time.monotonic()
    pool = p101.load_pool(POLICY)
    with harness.deterministic_scoring() as scoring_policy:
        policy = rp.new_policy(PAIRS, seed, device=DEVICE)
        history, stats = [], {}
        for round_index in range(rp.DISTIL_RECIPE["rounds"]):
            rows = [rp.window_rows(model, policy, unit, aggregated=bool(round_index), compute_weight=ICLR_LAMBDA,
                                   device=DEVICE) for unit in p101.window_units(pool)]
            history.append(rows)
            losses = rp.distil_round(policy, rp.aggregate_rounds(history), round_index, seed=seed, device=DEVICE)
            stats[f"final_loss_{round_index}"] = float(np.mean(losses[-60:]))
            stats[f"rows_{round_index}"] = int(sum(len(u["labels"]) for u in rows))
        policy.eval()
        evaluation = p101.policy_split_evaluation(model, policy, pool)
    path = policy_path(output, name, seed)
    atomic_torch(path, {"schema": SCHEMA_POLICY, "plan_identity": IDENTITY, "name": name, "seed": seed,
                        "lambda": ICLR_LAMBDA, "pairs": [str(p.identity) for p in policy.pairs],
                        "width": policy.layers[0].out_features, "remaining_feature": policy.remaining_feature,
                        "scoring_policy": scoring_policy, "model": policy.state_dict()})
    wall = time.monotonic() - began
    write_json(policy_report_path(output, name, seed), {
        "schema": SCHEMA_POLICY, "plan_identity": IDENTITY, "name": name, "seed": seed, "lambda": ICLR_LAMBDA,
        "trained_at": utc_now(), "policy_sha256": sha256_of(path), "predictor_sha256": sha256_of(member_path(output, name,
                                                                                                         seed)),
        "stats": stats, "policy_split": evaluation, "wall_seconds": wall})
    mark(output, f"policy-{name}-seed{seed}", wall, wall)
    log(f"J for {name} seed {seed}: {wall:.0f}s; policy-split AUC {evaluation['auc_pooled']:.4f}")


def train(output, workers, worker):
    output = Path(output)
    load_plan(output)
    torch.backends.cudnn.benchmark = True
    tasks = [(name, seed) for name in TRAINED for seed in SEEDS]
    mine = [t for k, t in enumerate(tasks) if k % workers == worker]
    if any(PREDICTORS[n]["labels"][0] == "learned" and not head_ready(output, PREDICTORS[n]["description"], s)
           for n, s in mine):
        raise ValueError("perception heads are not trained; run --heads first")
    if not payload_hash_ok():
        raise ValueError("the retrain training payload differs from its published hash")
    payload = rt.load_training_data(RETRAIN)
    for name, seed in mine:
        if not trained_path(output, name, seed).is_file():
            train_member(output, name, seed, payload)
        if policy_report_path(output, name, seed).is_file():
            continue
        state = member_state(output, name, seed)
        if state["typed_failure"]:
            log(f"{name} seed {seed}: typed failure {state['typed_failure']}; no policy")
            continue
        model = implementation(name).load_member(output, ARM, name, "all", seed)
        train_policy(output, name, seed, model)
        del model
        torch.cuda.empty_cache()
    return 0


# ---------------------------------------------------------------------------
# perception heads (item 3 (ii)): trained on the HEAD_K % labelled fit levels, frozen
# ---------------------------------------------------------------------------

def best_threshold(probability, truth, region):
    """The THRESHOLDS value with the highest in-sample F1 (ties to the lower threshold)."""
    scores = []
    for value in THRESHOLDS:
        p, t = (probability >= value)[region], truth[region]
        tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    best = max(range(len(THRESHOLDS)), key=lambda i: (scores[i], -i))
    return THRESHOLDS[best], dict(zip(map(str, THRESHOLDS), scores))


def train_micro_head(payload, seed, steps=None):
    steps = HEAD_TRAIN["steps"] if steps is None else steps
    rows = torch.nonzero(labelled_positions(payload, HEAD_K)).flatten()
    carriers = payload["carriers"][rows].to(DEVICE)
    labels = payload["relations"][rows].to(DEVICE)
    masks = payload["relation_mask"][rows].to(DEVICE)
    began = time.monotonic()
    with harness.deterministic_scoring():
        torch.manual_seed(seed)
        head = pf.MicroPerceptionHead().to(DEVICE)
        generator = torch.Generator().manual_seed(seed)
        optimizer = torch.optim.AdamW(head.parameters(), lr=HEAD_TRAIN["lr"], weight_decay=HEAD_TRAIN["weight_decay"])
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
        trace = []
        for step in range(steps):
            pick = torch.randint(len(rows), (HEAD_TRAIN["batch"],), generator=generator).to(DEVICE)
            loss = pf.perception_loss(head, carriers[pick], labels[pick], masks[pick])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            if step % 500 == 0 or step == steps - 1:
                trace.append({"step": step, "loss": float(loss.detach())})
        head.eval()
        pick = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed + 1))[
            :HEAD_TRAIN["threshold_rows"]].to(DEVICE)
        with torch.no_grad():
            logits, _ = head(carriers[pick])
        probability = logits.sigmoid().cpu()
        relations, mask = labels[pick].cpu(), masks[pick].cpu()
        del carriers, labels, masks
        upper = pf.upper_triangle()
        contact, contact_scores = best_threshold(probability[..., 0], relations[..., 0], mask[..., 0] & upper)
        supports, supports_scores = best_threshold(probability[..., 1], relations[..., 1], mask[..., 1])
    return head, {"steps": steps, "seconds": time.monotonic() - began, "loss_trace": trace,
                  "labelled_levels": len(labelled_levels(HEAD_K)), "labelled_positions": len(rows),
                  "thresholds": [contact, supports], "threshold_f1": {"contact": contact_scores,
                                                                      "supports": supports_scores}}


def micro_pseudo_labels(head, carriers, thresholds, chunk=1 << 17):
    relations, masks = [], []
    with harness.deterministic_scoring(), torch.no_grad():
        for begin in range(0, len(carriers), chunk):
            r, m = pf.perception_labels(head, carriers[begin:begin + chunk].to(DEVICE), thresholds)
            relations.append(r.cpu())
            masks.append(m.cpu())
    return torch.cat(relations), torch.cat(masks)


def macro_head_thresholds(data, head, train_rows):
    """Per (predicate, Delta): the in-sample best threshold on the labelled rows' engine (parsed) readings."""
    out, scores = torch.zeros(2, len(DELTAS)), {}
    with torch.no_grad():
        for d, delta in enumerate(DELTAS):
            rows, probability = om113.readout(data, "temporal", head, None, "engine", delta, d)
            keep = train_rows[rows]
            for k, name in enumerate(om.PREDICATES):
                region = data["available"][rows, k, d] & keep
                value, table = best_threshold(probability[:, k].cpu(), data["values"][rows, k, d].cpu(),
                                              region.cpu())
                out[k, d] = value
                scores[f"{name}|{delta}"] = table
    return out, scores


def macro_pseudo_labels(data, head, thresholds, rows, chunk=1 << 18):
    values = torch.zeros(len(rows), 2, len(DELTAS), dtype=torch.bool)
    with harness.deterministic_scoring(), torch.no_grad():
        for d, delta in enumerate(DELTAS):
            for begin in range(0, len(rows), chunk):
                r = rows[begin:begin + chunk].to(DEVICE)
                history = tmh.gather_history(data["carriers"], data["start"][r], data["position"][r], delta,
                                             head.history)
                probability = torch.sigmoid(head(history, delta))
                values[begin:begin + chunk, :, d] = (probability >= thresholds[:, d].to(DEVICE)).cpu()
    return values


def head_ready(output, kind, seed):
    return head_report_path(output, kind, seed).is_file()


def save_head(output, kind, seed, head, extra):
    path = head_path(output, kind, seed)
    atomic_torch(path, {"schema": SCHEMA_HEAD, "plan_identity": IDENTITY, "kind": kind, "seed": seed,
                        "state": {k: v.cpu() for k, v in head.state_dict().items()},
                        "history": getattr(head, "history", None), **{k: extra[k] for k in ("thresholds",)}})
    write_json(head_report_path(output, kind, seed), {"schema": SCHEMA_HEAD, "plan_identity": IDENTITY, "kind": kind,
                                                      "seed": seed, "trained_at": utc_now(),
                                                      "sha256": sha256_of(path), **extra})


def load_head(output, kind, seed):
    payload = torch.load(head_path(output, kind, seed), map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA_HEAD or payload["kind"] != kind or payload["seed"] != seed:
        raise ValueError(f"head {kind}/{seed} binding differs")
    report = read_json(head_report_path(output, kind, seed))
    if sha256_of(head_path(output, kind, seed)) != report["sha256"]:
        raise ValueError(f"head {kind}/{seed} differs from its report")
    head = pf.MicroPerceptionHead() if kind == "micro" else tmh.TemporalMacroHead(payload["history"])
    head.load_state_dict(payload["state"], strict=True)
    thresholds = payload["thresholds"]
    if kind == "macro":
        thresholds = torch.tensor(thresholds)
    return head.to(DEVICE).eval(), {**report, "thresholds": thresholds}


def heads(output, workers, worker):
    output = Path(output)
    load_plan(output)
    tasks = [("micro", s) for s in SEEDS] + [("macro", s) for s in SEEDS]
    mine = [t for k, t in enumerate(tasks) if k % workers == worker and not head_ready(output, *t)]
    for kind, seed in mine:
        began = time.monotonic()
        if kind == "micro":
            payload = rt.load_training_data(RETRAIN)
            head, extra = train_micro_head(payload, seed)
            save_head(output, kind, seed, head, extra)
        else:
            data = om113.load_data(MACRO113, "fit")
            level_index = {level: i for i, level in enumerate(data["levels"])}
            subset = torch.tensor([level_index[m] for m in sorted(labelled_levels(HEAD_K))], device=DEVICE)
            train_rows = torch.isin(data["row_level"], subset)
            head, report = om113.train_head(data, train_rows, MACRO_HEAD["history"], seed, MACRO_HEAD["mix"])
            with harness.deterministic_scoring():
                thresholds, scores = macro_head_thresholds(data, head, train_rows)
            payload = rt.load_training_data(RETRAIN)
            rows = payload_rows113(payload)
            values = macro_pseudo_labels(data, head, thresholds, rows)
            atomic_torch(pseudo_path(output, seed), {"schema": SCHEMA_PSEUDO, "plan_identity": IDENTITY, "seed": seed,
                                                     "values": values})
            extra = {"report": report, "labelled_levels": len(labelled_levels(HEAD_K)),
                     "labelled_rows": int(train_rows.sum()), "thresholds": thresholds.tolist(),
                     "threshold_f1": scores, "pseudo_sha256": sha256_of(pseudo_path(output, seed))}
            save_head(output, kind, seed, head, extra)
            del data
        torch.cuda.empty_cache()
        wall = time.monotonic() - began
        mark(output, f"head-{kind}-seed{seed}", wall, wall)
        log(f"perception head {kind} seed {seed}: {wall:.0f}s")
    return 0


# ---------------------------------------------------------------------------
# engine truth of the evaluation branches (built once after the freeze)
# ---------------------------------------------------------------------------

def truth_worker(attempt):
    """Process-pool worker: the #99 engine labels (s99.shot_labels) of positions 0..min(225, last frame):
    relations / relation_mask, and presence / projected centres at the last of them (the #100 metric)."""
    torch.set_num_threads(1)
    _, _, frames, _ = s99.shot_frames(attempt)
    segment = read_json(Path(attempt) / "shot-1" / "segment.json")
    last = min(ENDPOINT, len(frames) - 1)
    labels = s99.shot_labels(segment, frames, list(range(last + 1)), s99.vocabulary())
    return {"last": last, "frames": len(frames), "relations": labels["relations"].bool(),
            "relation_mask": labels["relation_mask"].bool(), "presence": labels["presence"][last].bool(),
            "centers": labels["centers"][last].float()}


def padded(tensor, length):
    if len(tensor) >= length:
        return tensor[:length]
    return torch.cat((tensor, tensor[-1:].expand(length - len(tensor), *tensor.shape[1:])))


def mixed_keys(cells):
    keys = []
    for inventory in INVENTORIES:
        for _, entry in sorted(cells[inventory]["mixed"].items()):
            keys += [item["key"] for item in entry["inventory"] if item["key"] not in keys]
    return sorted(set(keys))


def grid_keys(cells):
    return [item["key"] for _, entry in sorted(cells[PRIMARY]["mixed"].items()) for item in entry["inventory"]]


def truth(output):
    output = Path(output)
    load_plan(output)
    if truth_path(output).is_file():
        log("engine truth exists")
        return 0
    began = time.monotonic()
    branches, cells = gb.evaluation_set()
    attempts = {b["key"]: b["attempt"] for b in branches}
    keys = mixed_keys(cells)
    with ProcessPoolExecutor(TRUTH_WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        values = list(pool.map(truth_worker, [attempts[k] for k in keys], chunksize=4))
    length = ENDPOINT + 1
    data = macro113("evaluation")
    macro_values, macro_available = [], []
    for key, value in zip(keys, values, strict=True):
        rows = data["starts"][key] + torch.arange(value["last"] + 1)
        macro_values.append(padded(data["values"][rows], length))
        macro_available.append(padded(data["available"][rows], length))
    grid = grid_keys(cells)
    by_key = dict(zip(keys, values))
    carriers = torch.stack([padded(data["carriers"][data["starts"][k] + torch.arange(by_key[k]["last"] + 1)]
                                   .float(), length) for k in grid])
    store = {"schema": SCHEMA_TRUTH, "plan_identity": IDENTITY, "built_at": utc_now(), "keys": keys,
             "last": torch.tensor([v["last"] for v in values]), "frames": torch.tensor([v["frames"] for v in values]),
             "relations": torch.stack([padded(v["relations"], length) for v in values]),
             "relation_mask": torch.stack([padded(v["relation_mask"], length) for v in values]),
             "presence": torch.stack([v["presence"] for v in values]),
             "centers": torch.stack([v["centers"] for v in values]),
             "macro_values": torch.stack(macro_values), "macro_available": torch.stack(macro_available),
             "grid_keys": grid, "carriers": carriers,
             "sources": {"micro": "s99.shot_labels on the #104 evaluation captures (engine native traces)",
                         "macro": "the #113 frozen observable predicates (data/evaluation.pt)",
                         "carriers": "the #113 deterministic E99 parses of the evaluation frames (data/evaluation.pt)"}}
    atomic_torch(truth_path(output), store)
    wall = time.monotonic() - began
    mark(output, "truth", wall, 0.0, branches=len(keys), processes=TRUTH_WORKERS)
    log(f"engine truth: {len(keys)} branches, {len(grid)} grid carriers ({wall:.0f}s)")
    return 0


@functools.lru_cache(maxsize=None)
def load_truth(output):
    store = torch.load(truth_path(output), map_location="cpu", weights_only=True)
    if store.get("schema") != SCHEMA_TRUTH or store.get("plan_identity") != IDENTITY:
        raise ValueError("engine truth binding differs")
    store["index"] = {key: i for i, key in enumerate(store["keys"])}
    store["grid_index"] = {key: i for i, key in enumerate(store["grid_keys"])}
    return store


def block_truth(store, keys):
    index = torch.tensor([store["index"][k] for k in keys])
    return pf.BlockTruth(store["last"][index].to(DEVICE), store["relations"][index].to(DEVICE),
                         store["relation_mask"][index].to(DEVICE), store["macro_values"][index].to(DEVICE),
                         store["macro_available"][index].to(DEVICE))


def engine_error(z, store, key):
    if not bool(torch.isfinite(z).all()):
        return None
    i = store["index"][key]
    return v1.r100.engine_position_mse(z.tolist(), {"presence": store["presence"][i][None],
                                                    "centers": store["centers"][i][None]}, 0)


# ---------------------------------------------------------------------------
# models of every condition
# ---------------------------------------------------------------------------

def load_predictor(output, name, seed):
    """(model, typed failure)."""
    if name == "gateb":
        return p101.load_predictor(seed), None
    state = member_state(output, name, seed)
    if state["typed_failure"]:
        return None, state["typed_failure"]
    if sha256_of(member_path(output, name, seed)) != state["sha256"]:
        raise ValueError(f"{name} seed {seed}: checkpoint differs from its training report")
    return implementation(name).load_member(output, ARM, name, "all", seed), None


def load_j(output, name, seed):
    if name == "gateb":
        return p101.load_policy(POLICY, "pe", ICLR_LAMBDA, seed)
    payload = torch.load(policy_path(output, name, seed), map_location="cpu", weights_only=True)
    report = read_json(policy_report_path(output, name, seed))
    if (payload.get("schema") != SCHEMA_POLICY or payload["name"] != name or payload["seed"] != seed
            or sha256_of(policy_path(output, name, seed)) != report["policy_sha256"]):
        raise ValueError(f"policy {name}/{seed} binding differs")
    policy = rp.RequestPolicy(PAIRS, payload["width"], payload["remaining_feature"])
    policy.load_state_dict(payload["model"], strict=True)
    return policy.to(DEVICE).eval()


def j_pair_counts(output, name, seed):
    if name == "gateb":
        return read_json(p101.report_path(POLICY, "pe", ICLR_LAMBDA, seed))["policy_split"]["pair_counts"]
    return read_json(policy_report_path(output, name, seed))["policy_split"]["pair_counts"]


def delta_d(output, name, seed):
    """#96 rule: J's modal Delta (pooled over descriptions) on the policy split; ties to the lower Delta."""
    counts = j_pair_counts(output, name, seed)
    per_delta = [sum(counts[t96.PAIR_NAMES.index(f"{d}-{a}")] for a in t96.ALPHAS) for d in DELTAS]
    return DELTAS[max(range(len(DELTAS)), key=lambda i: (per_delta[i], -i))]


def arm_spec(arm, delta):
    if arm == "J":
        return {"kind": "joint"}
    if arm == "T":
        return rs.temporal_spec(PAIRS, "continuous")
    if arm == "D":
        return rs.description_spec(PAIRS, delta)
    return t96.arm_spec(arm)


def calibration(output, name, seed):
    record = read_json(calibration_path(output, name, seed))
    if record.get("schema") != SCHEMA_CALIBRATION or record["name"] != name or record["seed"] != seed:
        raise ValueError(f"calibration {name}/{seed} binding differs")
    return record


@functools.lru_cache(maxsize=None)
def temporal_thresholds():
    selected = read_json(MACRO113 / "selection.json")["candidates"][TEMPORAL_FEEDBACK]["thresholds"]["predicted"]
    return torch.tensor([[selected[f"{p}|{d}"] for d in DELTAS] for p in om.PREDICATES])


def temporal_head(seed):
    head, _ = om113.load_head(MACRO113, om113.head_name("final", TEMPORAL_FEEDBACK, seed))
    return head


def make_feedback(output, name, fb, seed, description, keys, store, context, place):
    """The block's Feedback (always with the branches' engine labels, for the readout counts)."""
    params = feedback_parameters(fb)
    mode = Abstraction(description)
    common = {"truth": block_truth(store, keys), "device": DEVICE}
    generator = torch.Generator(device=DEVICE).manual_seed(block_seed(name, fb, seed, *place))
    if params["kind"] == "flip":
        return pf.Feedback("flip", mode, p=params["p"], generator=generator, **common)
    if params["kind"] == "gauss":
        return pf.Feedback("gauss", mode, sigma=context["sigma"][fb], generator=generator, **common)
    if params["kind"] == "shuffled":
        partner = (torch.arange(len(keys)) + 1) % len(keys)
        return pf.Feedback("shuffled", mode, partner=partner.to(DEVICE), **common)
    if params["kind"] == "temporal":
        return pf.Feedback("temporal", mode, temporal=context["temporal"], thresholds=context["thresholds"].to(DEVICE),
                           rows=len(keys), **common)
    return pf.Feedback(params["kind"], mode, **common)


# ---------------------------------------------------------------------------
# scoring (deterministic_scoring, processes that did not train)
# ---------------------------------------------------------------------------

def rle_schedules(items, timing):
    return {str(item["ordinal"]): t96.rle(timing["schedules"][k]) for k, item in enumerate(items)}


def score_block(model, policy, spec, z0, items, costs, schedules, feedback=None, store=None):
    actions = m99.action_batch(items)
    z_end, timing = pf.rollout_endpoints(model, policy, z0.to(DEVICE), actions, spec, select=t96.select_pairs,
                                         feedback=feedback, record_schedules=schedules)
    out = {"rows": [costs.row(item["ordinal"], z_end[slot].cpu(), z0.cpu()) for slot, item in enumerate(items)],
           "candidates": len(items), **{k: v for k, v in timing.items() if k != "schedules"}}
    if schedules:
        out["schedules"] = rle_schedules(items, timing)
    if store is not None:
        out["engine_mse"] = [engine_error(z_end[slot].cpu(), store, item["key"]) for slot, item in enumerate(items)]
    if feedback is not None:
        out["feedback"] = feedback.summary()
    return out


def score_record(output, cells, anchors, store, name, fb, arm, seed, policy_state):
    costs = p101.endpoint_costs()
    began = time.monotonic()
    description = PREDICTORS[name]["description"]
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "scoring_policy": policy_state, "scored_at": utc_now(),
            "name": name, "feedback": fb, "arm": arm, "seed": seed, "engine_seconds": 0}
    model, failure = load_predictor(output, name, seed)
    context = {}
    if model is not None and feedback_parameters(fb)["kind"] == "gauss":
        cal = calibration(output, name, seed)
        if cal["typed_failure"]:
            failure = cal["typed_failure"]
        else:
            context["sigma"] = {g: cal["targets"][g]["sigma"] for g in GAUSS}
    if model is None or failure:
        return {**base, "typed_failure": failure, "inventories": {}}
    if fb == "temporal":
        context["temporal"], context["thresholds"] = temporal_head(seed), temporal_thresholds()
    policy = load_j(output, name, seed) if arm in SCHEDULED else None
    delta = delta_d(output, name, seed)
    spec = arm_spec(arm, delta)
    inventories = {i: {} for i in INVENTORIES}
    for inventory, member, entry in p101.ordered_cells(cells):
        keys = [item["key"] for item in entry["inventory"]]
        feedback = make_feedback(output, name, fb, seed, description, keys, store, context, (arm, inventory, member))
        inventories[inventory][member] = score_block(model, policy, spec, anchors[member], entry["inventory"], costs,
                                                     arm in SCHEDULED, feedback,
                                                     store if inventory == PRIMARY else None)
    torch.cuda.synchronize()
    return {**base, "typed_failure": None, "delta_D": delta, "spec": {k: (list(v) if isinstance(v, tuple) else v)
                                                                       for k, v in spec.items()},
            "predictor_sha256": sha256_of(gb.member_path("unit", seed)) if name == "gateb"
            else sha256_of(member_path(output, name, seed)),
            "policy_sha256": None if policy is None else (
                sha256_of(p101.policy_path(POLICY, "pe", ICLR_LAMBDA, seed)) if name == "gateb"
                else sha256_of(policy_path(output, name, seed))),
            "inventories": inventories, "seconds": time.monotonic() - began}


def wrong_record(output, name, seed, policy_state):
    """J from every fit-split anchor over the shared grid candidates (Gate-B wrong-anchor rule)."""
    costs = p101.endpoint_costs()
    began = time.monotonic()
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "scoring_policy": policy_state, "scored_at": utc_now(),
            "name": name, "feedback": "default", "arm": "wrong-J", "seed": seed, "engine_seconds": 0}
    model, failure = load_predictor(output, name, seed)
    if model is None:
        return {**base, "typed_failure": failure, "wrong_anchor": None}
    policy = load_j(output, name, seed)
    items = gb.inventory_items(PRIMARY)
    fit = gb.fit_anchors()
    rows = {member: score_block(model, policy, {"kind": "joint"}, fit[member], items, costs, False)
            for member in sorted(fit)}
    return {**base, "typed_failure": None, "wrong_anchor": rows, "seconds": time.monotonic() - began}


def readout_counts_block(model, description, z, labels):
    """#100 counts per source on a branch's parsed carriers z [T, D] (positions 0..last): parsed and one
    Delta-step predictions from the parsed f - Delta. micro: {source: [2, 3]}; macro: {source: [2, 3]} per
    Delta column (parsed-<Delta>, predicted-<Delta>)."""
    out = {}
    last = labels["last"]
    action = labels["action"]
    if description == "micro":
        relations, mask = labels["relations"], labels["relation_mask"]
        pair = PredictionPair(1, Abstraction.MICRO)
        values = pf.own_features(model, z, pair)[:, :-2].reshape(len(z), SLOTS, SLOTS, 2)
        out["parsed"] = pf.micro_counts(values, relations, mask)
        for delta in DELTAS:
            if last < delta:
                out[f"predicted-{delta}"] = torch.zeros(2, 3, dtype=torch.long, device=z.device)
                continue
            pair = PredictionPair(delta, Abstraction.MICRO)
            predicted = model.carrier(z[:last + 1 - delta], action.expand(last + 1 - delta, -1), pair)
            values = pf.own_features(model, predicted, pair)[:, :-2].reshape(len(predicted), SLOTS, SLOTS, 2)
            out[f"predicted-{delta}"] = pf.micro_counts(values, relations[delta:last + 1], mask[delta:last + 1])
        return out
    truth_values, available = labels["macro_values"], labels["macro_available"]
    for d, delta in enumerate(DELTAS):
        pair = PredictionPair(delta, Abstraction.MACRO)
        values = pf.own_features(model, z, pair)[:, :2]
        out[f"parsed-{delta}"] = pf.macro_counts(values, truth_values[:, :, d], available[:, :, d])
        if last < delta:
            out[f"predicted-{delta}"] = torch.zeros(2, 3, dtype=torch.long, device=z.device)
            continue
        predicted = model.carrier(z[:last + 1 - delta], action.expand(last + 1 - delta, -1), pair)
        values = pf.own_features(model, predicted, pair)[:, :2]
        out[f"predicted-{delta}"] = pf.macro_counts(values, truth_values[delta:last + 1, :, d],
                                                    available[delta:last + 1, :, d])
    return out


def branch_labels(store, key, action):
    i = store["index"][key]
    last = int(store["last"][i])
    return {"last": last, "action": action,
            "relations": store["relations"][i][:last + 1].to(DEVICE),
            "relation_mask": store["relation_mask"][i][:last + 1].to(DEVICE),
            "macro_values": store["macro_values"][i][:last + 1].to(DEVICE),
            "macro_available": store["macro_available"][i][:last + 1].to(DEVICE)}


def readout_record(output, cells, store, name, seed, policy_state):
    """Readout counts of the predictor's own heads per grid level (parsed and one-step predicted), or of a
    perception head (``head-micro`` / ``head-macro``: parsed, its frozen thresholds)."""
    began = time.monotonic()
    base = {"schema": SCHEMA_READOUT, "plan_identity": IDENTITY, "scoring_policy": policy_state,
            "scored_at": utc_now(), "name": name, "seed": seed, "engine_seconds": 0}
    head = None
    if name.startswith("head-"):
        kind = name.split("-", 1)[1]
        head, info = load_head(output, kind, seed)
        description = kind
        model = None
    else:
        model, failure = load_predictor(output, name, seed)
        if model is None:
            return {**base, "typed_failure": failure, "levels": {}}
        description = PREDICTORS[name]["description"]
    levels = {}
    with torch.no_grad():
        for member, entry in sorted(cells[PRIMARY]["mixed"].items()):
            total = {}
            for item in entry["inventory"]:
                labels = branch_labels(store, item["key"], m99.action_batch([item]))
                z = store["carriers"][store["grid_index"][item["key"]]][:labels["last"] + 1].to(DEVICE)
                if head is None:
                    part = readout_counts_block(model, description, z, labels)
                elif description == "micro":
                    logits, _ = head(z)
                    probability = logits.sigmoid()
                    values = torch.stack([(probability[..., c] >= info["thresholds"][c]).float() for c in range(2)], -1)
                    part = {"parsed": pf.micro_counts(values, labels["relations"], labels["relation_mask"])}
                else:
                    part = {}
                    for d, delta in enumerate(DELTAS):
                        positions = torch.arange(labels["last"] + 1, device=DEVICE)
                        history = tmh.gather_history(z, torch.zeros_like(positions), positions, delta, head.history)
                        probability = torch.sigmoid(head(history, delta))
                        values = (probability >= info["thresholds"][:, d].to(DEVICE)).float()
                        part[f"parsed-{delta}"] = pf.macro_counts(values, labels["macro_values"][:, :, d],
                                                                  labels["macro_available"][:, :, d])
                for key, value in part.items():
                    total[key] = total.get(key, 0) + value
            levels[member] = {key: value.cpu().tolist() for key, value in total.items()}
    torch.cuda.synchronize()
    return {**base, "typed_failure": None, "description": description, "levels": levels,
            "seconds": time.monotonic() - began}


def score_tasks():
    """(kind, name, feedback, arm, seed), heaviest first so that the shards balance."""
    tasks = []
    for description, base in REFERENCE.items():
        for fb in INTERVENTIONS[description]:
            for arm in arms_for(description):
                if arm == "T" and fb != "default":
                    continue
                tasks.append(("right", base, fb, arm))
    for name in TRAINED:
        if name in REFERENCE.values():
            continue
        tasks += [("right", name, "default", arm) for arm in arms_for(PREDICTORS[name]["description"])]
    tasks += [("wrong", name, "default", "J") for name in PREDICTORS]
    tasks += [("readout", name, None, None) for name in (*PREDICTORS, "head-micro", "head-macro")]

    def weight(task):
        kind, _, _, arm = task
        if kind != "right":
            return COST_WEIGHT[kind]
        return COST_WEIGHT[arm if arm in COST_WEIGHT else arm.rsplit("-", 1)[0]]
    ordered = sorted(tasks, key=lambda t: -weight(t))
    return [(*t, s) for t in ordered for s in SEEDS]


def task_path(output, task):
    kind, name, fb, arm, seed = task
    if kind == "right":
        return record_path(output, name, fb, arm, seed)
    if kind == "wrong":
        return wrong_path(output, name, seed)
    return readout_path(output, name, seed)


def score(output, workers, worker):
    output = Path(output)
    load_plan(output)
    missing = [(n, s) for n in TRAINED for s in SEEDS if not trained_path(output, n, s).is_file()]
    if missing:
        raise ValueError(f"members {missing[:4]} are not trained; score after training")
    missing = [(n, s) for n in REFERENCE.values() for s in SEEDS if not calibration_path(output, n, s).is_file()]
    if missing:
        raise ValueError(f"calibration {missing} missing; run --calibrate first")
    _, cells = gb.evaluation_set()
    anchors = gb.load_anchors(GATE_B)["anchors"]
    store = load_truth(output)
    for task in score_tasks()[worker::workers]:
        path = task_path(output, task)
        if path.is_file():
            continue
        kind, name, fb, arm, seed = task
        t0 = time.monotonic()
        with harness.deterministic_scoring() as policy_state:
            if kind == "right":
                record = score_record(output, cells, anchors, store, name, fb, arm, seed, policy_state)
            elif kind == "wrong":
                record = wrong_record(output, name, seed, policy_state)
            else:
                record = readout_record(output, cells, store, name, seed, policy_state)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, record)
        torch.cuda.empty_cache()
        wall = time.monotonic() - t0
        mark(output, f"score-worker{worker}", wall, wall)
        log(f"worker {worker}: {path.stem} ({wall:.0f}s)"
            + (f"; typed failure {record['typed_failure']}" if record["typed_failure"] else ""))
    return 0


# ---------------------------------------------------------------------------
# calibration of the Gaussian logit noise (fit split only, before any evaluation score)
# ---------------------------------------------------------------------------

def fit_grid_cells():
    return gb.mixed_cells(gb.inventory_entries(rt.fit_levels(), rt.complete_branches(), rt.fit_verdicts(), PRIMARY))


def fit_truth(keys):
    """BlockTruth of fit branches: micro from the retrain shot cache, macro from the #113 fit labels."""
    length = ENDPOINT + 1
    data = macro113("fit")
    relations, masks, values, available, last = [], [], [], [], []
    for key in keys:
        shot = rt.read_shot(RETRAIN, key)
        end = min(int(shot["end"]), ENDPOINT)
        relations.append(padded(shot["relations"][:end + 1].bool(), length))
        masks.append(padded(shot["relation_mask"][:end + 1].bool(), length))
        rows = data["starts"][key] + torch.arange(end + 1)
        values.append(padded(data["values"][rows], length))
        available.append(padded(data["available"][rows], length))
        last.append(end)
    return pf.BlockTruth(torch.tensor(last).to(DEVICE), torch.stack(relations).to(DEVICE),
                         torch.stack(masks).to(DEVICE), torch.stack(values).to(DEVICE),
                         torch.stack(available).to(DEVICE))


def feedback_f1(counts):
    """Mean over the two channels of the pooled F1 (Delta columns summed)."""
    return float(np.nanmean(pf.f1_from(torch.as_tensor(counts).sum(0)).numpy()))


def calibration_f1(model, description, cells, anchors, truths, sigma, seed):
    """Delivered feedback F1 over the fixed description requests on the calibration cells (sigma None =
    the model's own feedback)."""
    mode = Abstraction(description)
    total = torch.zeros(len(DELTAS), 2, 3, dtype=torch.long, device=DEVICE)
    for member, entry in cells:
        for delta in DELTAS:
            spec = t96.arm_spec(f"F-{delta}-{description}")
            if sigma is None:
                feedback = pf.Feedback("default", mode, truth=truths[member], device=DEVICE)
            else:
                generator = torch.Generator(device=DEVICE).manual_seed(block_seed("calibration", seed, member, delta))
                feedback = pf.Feedback("gauss", mode, truth=truths[member], sigma=sigma, generator=generator,
                                       device=DEVICE)
            pf.rollout_endpoints(model, None, anchors[member].to(DEVICE), m99.action_batch(entry["inventory"]), spec,
                                 feedback=feedback)
            total += feedback.counts
    return feedback_f1(total.cpu()), total.cpu().tolist()


def calibrate_one(output, name, seed, cells_count=CAL_CELLS, iterations=CAL_ITERATIONS):
    began = time.monotonic()
    description = PREDICTORS[name]["description"]
    base = {"schema": SCHEMA_CALIBRATION, "plan_identity": IDENTITY, "name": name, "seed": seed,
            "description": description, "calibrated_at": utc_now()}
    model, failure = load_predictor(output, name, seed)
    if model is None:
        return {**base, "typed_failure": failure}
    cells = sorted(fit_grid_cells().items())[:cells_count]
    anchors = gb.fit_anchors()
    truths = {member: fit_truth([item["key"] for item in entry["inventory"]]) for member, entry in cells}
    with harness.deterministic_scoring() as policy_state:
        default, default_counts = calibration_f1(model, description, cells, anchors, truths, None, seed)
        memo = {}

        def at(sigma):
            if sigma not in memo:
                memo[sigma] = calibration_f1(model, description, cells, anchors, truths, sigma, seed)[0]
            return memo[sigma]
        targets = {}
        for g, fraction in zip(GAUSS, GAUSS_FRACTIONS):
            result = pf.bisect_log(at, fraction * default, iterations=iterations)
            targets[g] = {"fraction": fraction, "target_f1": fraction * default, **result}
    return {**base, "typed_failure": None, "scoring_policy": policy_state, "cells": [m for m, _ in cells],
            "default_f1": default, "default_counts": default_counts, "targets": targets,
            "seconds": time.monotonic() - began}


def calibrate(output, workers, worker):
    output = Path(output)
    load_plan(output)
    tasks = [(name, seed) for name in REFERENCE.values() for seed in SEEDS]
    for name, seed in tasks[worker::workers]:
        path = calibration_path(output, name, seed)
        if path.is_file():
            continue
        if name != "gateb" and not trained_path(output, name, seed).is_file():
            raise ValueError(f"{name} seed {seed} is not trained; calibrate after training")
        began = time.monotonic()
        record = calibrate_one(output, name, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, record)
        wall = time.monotonic() - began
        mark(output, f"calibrate-{name}-seed{seed}", wall, wall)
        log(f"calibrated {name} seed {seed} ({wall:.0f}s): default F1 {record.get('default_f1')}")
    return 0


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def read_record(path):
    record = read_json(path)
    if record.get("plan_identity") != IDENTITY or record.get("schema") not in (SCHEMA_RECORD, SCHEMA_READOUT):
        raise ValueError(f"record {path} binding differs")
    return record


def load_records(output):
    records = {}
    for task in score_tasks():
        records[task] = read_record(task_path(output, task))
    return records


def key_of(name, fb, arm):
    return f"{name}|{fb}|{arm}"


def right_records(records):
    return {(name, fb, arm, seed): record for (kind, name, fb, arm, seed), record in records.items() if kind == "right"}


def found_rows(record, inventory, member):
    return None if record["typed_failure"] else record["inventories"][inventory][member]["rows"]


def view_rows(right, cells, inventory, stats):
    keys = sorted({(n, f, a) for n, f, a, _ in right})
    rows = []
    for member, entry in sorted(cells[inventory]["mixed"].items()):
        prior = m99.view(entry, gb.prior_rows(entry, stats), PRIMARY_COST)
        for seed in SEEDS:
            views = {key_of(n, f, a): v1.cost_view(entry, found_rows(right[(n, f, a, seed)], inventory, member),
                                                   PRIMARY_COST) for n, f, a in keys}
            for description, base in REFERENCE.items():
                for fb in INTERVENTIONS[description][1:]:
                    views[key_of(base, fb, "T")] = views[key_of(base, "default", "T")]
            views["prior"] = prior
            rows.append({"member": member, "seed": seed, "views": views})
    return rows


def contrast(rows, pieces):
    block = m99.paired(rows, pieces)
    return {**block, "token": c2.token(block["estimate"])}


def noninferiority(block):
    """supported iff lower > -delta; not_supported_by_this_experiment iff upper < -delta; otherwise
    readiness_or_precision_insufficient."""
    if block is None:
        return STOP_TOKEN
    low, high = block["interval"]
    if low > -DELTA:
        return "supported"
    if high < -DELTA:
        return "not_supported_by_this_experiment"
    return STOP_TOKEN


def condition_contrasts(rows, name, fb, description):
    k = functools.partial(key_of, name, fb)
    fixed = fixed_arms(description)
    out = {"J-T": contrast(rows, [(1, k("J")), (-1, k("T"))]),
           "J-D": contrast(rows, [(1, k("J")), (-1, k("D"))]),
           "J-prior": contrast(rows, [(1, k("J")), (-1, "prior")]),
           "F_mean-prior": contrast(rows, [*((1 / 3, k(a)) for a in fixed), (-1, "prior")])}
    return out


def versus(rows, first, second, description, fb_first="default", fb_second="default"):
    """first - second for J, (J - T), D and the request mean of the fixed description requests."""
    a, b = functools.partial(key_of, first, fb_first), functools.partial(key_of, second, fb_second)
    fixed = fixed_arms(description)
    return {"J": contrast(rows, [(1, a("J")), (-1, b("J"))]),
            "J-T": contrast(rows, [(1, a("J")), (-1, a("T")), (-1, b("J")), (1, b("T"))]),
            "D": contrast(rows, [(1, a("D")), (-1, b("D"))]),
            "F_mean": contrast(rows, [*((1 / 3, a(x)) for x in fixed), *((-1 / 3, b(x)) for x in fixed)]),
            **{x: contrast(rows, [(1, a(x)), (-1, b(x))]) for x in fixed}}


def rankers(rows, name, fb, description):
    return {arm: v1.ranker_block(rows, key_of(name, fb, arm)) for arm in arms_for(description)}


def f1_bootstrap(per_level):
    """Pooled F1 (mean over channels of F1 of summed counts) with a level bootstrap: per_level [L, ..., 2, 3]."""
    counts = np.asarray(per_level, dtype=np.int64)
    if not len(counts):
        return None

    def f1(summed):
        tp, fp, fn = summed[..., 0], summed[..., 1], summed[..., 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            value = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
        return np.nanmean(value.reshape(*value.shape[:1], -1) if value.ndim > 1 else value, axis=-1)
    flat = counts.reshape(len(counts), -1, 2, 3).sum(1)
    point = float(f1(flat.sum(0)[None])[0])
    generator = np.random.default_rng(np.random.PCG64(BOOTSTRAP_SEED))
    draws = generator.integers(0, len(flat), size=(BOOTSTRAP_DRAWS, len(flat)))
    values = []
    for begin in range(0, BOOTSTRAP_DRAWS, 500):
        values.append(f1(flat[draws[begin:begin + 500]].sum(1)))
    low, high = np.nanquantile(np.concatenate(values), (0.025, 0.975))
    channels = flat.sum(0)
    tp, fp, fn = channels[:, 0], channels[:, 1], channels[:, 2]
    per_channel = [float(2 * a / (2 * a + b + c)) if 2 * a + b + c else None for a, b, c in zip(tp, fp, fn)]
    return {"mean": point, "interval": [float(low), float(high)], "clusters": len(flat), "per_channel": per_channel,
            "counts": channels.tolist(), "label": "DESCRIPTIVE"}


def feedback_table(right, cells, name, fb, arms):
    """Delivered-feedback F1 on grid, pooled over ``arms`` and seeds, level bootstrap."""
    per_level = []
    calls = clamped = flipped = eligible = nonzero = 0
    for member in sorted(cells[PRIMARY]["mixed"]):
        total = np.zeros((len(DELTAS), 2, 3), dtype=np.int64)
        for arm in arms:
            for seed in SEEDS:
                record = right[(name, fb, arm, seed)]
                if record["typed_failure"]:
                    continue
                summary = record["inventories"][PRIMARY][member]["feedback"]
                total += np.asarray(summary["counts"], dtype=np.int64)
                calls += sum(summary["calls"])
                clamped += summary["clamped_calls"]
                flipped += summary["flipped"]
                eligible += summary["eligible"]
                nonzero += summary["removed_nonzero"]
        per_level.append(total)
    return {"f1": f1_bootstrap(per_level), "calls": calls, "clamped_calls": clamped, "flipped": flipped,
            "eligible": eligible, "realized_flip_rate": flipped / eligible if eligible else None,
            "removed_nonzero": nonzero, "arms": list(arms)}


def rollout_per_level(right, cells, name, fb, arm):
    out = {}
    nonfinite = 0
    for member in sorted(cells[PRIMARY]["mixed"]):
        seeds_ = []
        for seed in SEEDS:
            record = right[(name, fb, arm, seed)]
            if record["typed_failure"]:
                continue
            result = record["inventories"][PRIMARY][member]
            values = [v for v in result["engine_mse"] if v is not None]
            nonfinite += sum(1 for r in result["rows"] if r["excluded"])
            if values:
                seeds_.append(float(np.mean(values)))
        out[member] = float(np.mean(seeds_)) if seeds_ else None
    return out, nonfinite


def rollout_block(per_level):
    common = [m for m, v in per_level.items() if v is not None]
    return v1.bootstrap([per_level[m] for m in common], common)


def rollout_versus(first, second):
    common = [m for m in sorted(first) if first[m] is not None and second.get(m) is not None]
    return v1.bootstrap([first[m] - second[m] for m in common], common)


def rollout_table(right, cells, name, fb, description, reference=None):
    """#100 engine_position_mse at t = 225 per arm on grid (branch mean per (level, seed), seed mean per
    level, level bootstrap); with ``reference`` = (name, fb) the paired difference per arm."""
    out = {}
    for arm in arms_for(description):
        source = (name, "default", "T") if arm == "T" else (name, fb, arm)
        per_level, nonfinite = rollout_per_level(right, cells, *source)
        entry = {"error": rollout_block(per_level), "nonfinite_endpoints": nonfinite}
        if reference is not None:
            ref_source = (reference[0], "default", "T") if arm == "T" else (*reference, arm)
            entry["minus_reference"] = rollout_versus(per_level, rollout_per_level(right, cells, *ref_source)[0])
        out[arm] = entry
    return out


def readout_table(records, name):
    """Readout F1 of a predictor's own heads on grid (parsed, one-step predicted) with level bootstrap."""
    per_seed = [records[("readout", name, None, None, s)] for s in SEEDS]
    usable = [r for r in per_seed if not r["typed_failure"]]
    if not usable:
        return None
    sources = sorted(usable[0]["levels"][next(iter(usable[0]["levels"]))])
    out = {}
    members = sorted(usable[0]["levels"])
    for source in sources:
        per_level = [np.sum([np.asarray(r["levels"][m][source]) for r in usable], axis=0) for m in members]
        out[source] = f1_bootstrap(per_level)
    if PREDICTORS.get(name, {}).get("description", name.split("-")[-1]) == "macro" or name == "head-macro":
        for kind in ("parsed", "predicted"):
            keys = [f"{kind}-{d}" for d in DELTAS if f"{kind}-{d}" in sources]
            if keys:
                per_level = [np.stack([np.sum([np.asarray(r["levels"][m][k]) for r in usable], axis=0) for k in keys])
                             for m in members]
                out[kind] = f1_bootstrap(per_level)
    else:
        keys = [f"predicted-{d}" for d in DELTAS if f"predicted-{d}" in sources]
        if keys:
            per_level = [np.stack([np.sum([np.asarray(r["levels"][m][k]) for r in usable], axis=0) for k in keys])
                         for m in members]
            out["predicted"] = f1_bootstrap(per_level)
    out["seeds"] = len(usable)
    return out


def wrong_table(records, right, cells, name):
    rows = []
    for member, entry in sorted(cells[PRIMARY]["mixed"].items()):
        for seed in SEEDS:
            record = records[("wrong", name, "default", "J", seed)]
            right_rows = found_rows(right[(name, "default", "J", seed)], PRIMARY, member)
            if record["typed_failure"] or right_rows is None:
                rows.append({"member": member, "seed": seed, "views": {"J": None}})
                continue
            wrong = [record["wrong_anchor"][a]["rows"] for a in sorted(record["wrong_anchor"])]
            wrong_auc = [a for a in (c2.auc(entry, w) for w in wrong) if a is not None]
            rhos = [r for r in (c2.spearman_rows(right_rows, w) for w in wrong) if r is not None]
            rows.append({"member": member, "seed": seed, "views": {"J": {
                "right": c2.auc(entry, right_rows), "wrong": float(np.mean(wrong_auc)) if wrong_auc else None,
                "stability": float(np.mean(rhos)) if rhos else None}}})
    difference = c2.difference_block(rows, [(1, "J", "right"), (-1, "J", "wrong")])
    return {"cells": len(rows), "wrong_anchors": len(gb.fit_anchors()), "right": c2.field_block(rows, "J", "right"),
            "wrong": c2.field_block(rows, "J", "wrong"), "right_minus_wrong": difference,
            "stability": c2.field_block(rows, "J", "stability"), "token": c2.token(difference)}


def training_table(output, name):
    if name == "gateb":
        return {"source": "Gate-B handoff (#112 v3 retrain)", "usable_seeds": list(SEEDS)}
    members = [member_state(output, name, s) for s in SEEDS]
    labels = [m["labels"] for m in members]
    f1 = [entry.get("label_f1") for entry in labels if entry.get("label_f1") is not None]
    return {"members": members, "usable_seeds": [m["seed"] for m in members if m["usable"]],
            "label_f1": float(np.mean(f1)) if f1 else None,
            "realized_flip_rate": float(np.mean([e["realized_rate"] for e in labels])) if labels[0].get("realized_rate")
            is not None else None,
            "J_policy_split_auc": {str(s): read_json(policy_report_path(output, name, s))["policy_split"]["auc_pooled"]
                                   for s in SEEDS if policy_report_path(output, name, s).is_file()},
            "delta_D": {str(s): delta_d(output, name, s) for s in SEEDS if policy_report_path(output, name, s).is_file()}}


def dose_response(right, cells, rows, description):
    base = REFERENCE[description]
    out = {}
    for fb in INTERVENTIONS[description]:
        out[fb] = {
            "parameters": feedback_parameters(fb),
            "feedback_fixed": feedback_table(right, cells, base, fb, fixed_arms(description)),
            "feedback_J": feedback_table(right, cells, base, fb, ("J",)),
            "rankers": {i: rankers(rows[i], base, fb, description) for i in INVENTORIES},
            "contrasts": {i: condition_contrasts(rows[i], base, fb, description) for i in INVENTORIES},
            "versus_default": ({i: versus(rows[i], base, base, description, fb_first=fb) for i in INVENTORIES}
                               if fb != "default" else None),
            "rollout": rollout_table(right, cells, base, fb, description,
                                     None if fb == "default" else (base, "default")),
        }
    return out


def predictor_table(output, records, right, cells, rows, name, description):
    ref = REFERENCE[description]
    return {"training": training_table(output, name), "readout": readout_table(records, name),
            "feedback_fixed": feedback_table(right, cells, name, "default", fixed_arms(description)),
            "rankers": {i: rankers(rows[i], name, "default", description) for i in INVENTORIES},
            "contrasts": {i: condition_contrasts(rows[i], name, "default", description) for i in INVENTORIES},
            "versus_reference": ({i: versus(rows[i], name, ref, description) for i in INVENTORIES}
                                 if name != ref else None),
            "rollout": rollout_table(right, cells, name, "default", description,
                                     None if name == ref else (ref, "default"))}


def disposition(rows_grid, description, guards_ok):
    """The pre-declared rule on 'the description axis keeps its value without engine-labelled predicates'
    (grid): label-free bottleneck (item 3 iii) vs the engine-labelled reference."""
    free, ref = FREE[description], REFERENCE[description]
    block = versus(rows_grid, free, ref, description)
    pieces = {"J-T": block["J-T"], "F_mean": block["F_mean"]}
    tokens = {k: noninferiority(v["estimate"]) for k, v in pieces.items()}
    if not guards_ok:
        token = STOP_TOKEN
    elif all(t == "supported" for t in tokens.values()):
        token = "supported"
    elif any(t == "not_supported_by_this_experiment" for t in tokens.values()):
        token = "not_supported_by_this_experiment"
    else:
        token = STOP_TOKEN
    reference_value = contrast(rows_grid, [(1, key_of(ref, "default", "J")), (-1, key_of(ref, "default", "T"))])
    return {"claim": f"the description axis keeps its value without engine-labelled predicates ({description})",
            "label_free": free, "reference": ref, "token": token, "contrast_tokens": tokens,
            "contrasts": {k: v["estimate"] for k, v in pieces.items()},
            "J_label_free_minus_reference": block["J"]["estimate"],
            "reference_description_value_J_minus_T": {**reference_value["estimate"], "token": reference_value["token"]}
            if reference_value["estimate"] else None}


def replication(right):
    """Gate-B default-feedback rows (and schedules) equal the published #101 (J, fixed micro requests) and
    #114 (T-continuous, D-Delta) records on every cell and seed."""
    from scripts import run_horizon_vs_joint as h114
    same, checked = True, 0
    sources = {"J": lambda s: p101.read_record(POLICY, p101.J_ICLR, s),
               **{a: (lambda a: lambda s: p101.read_record(POLICY, a, s))(a) for a in fixed_arms("micro")},
               "T": lambda s: h114.read_record(h114.OUTPUT, "T-continuous", s)}
    for arm, source in sources.items():
        for seed in SEEDS:
            reference = source(seed)["inventories"]
            mine = right[("gateb", "default", arm, seed)]["inventories"]
            for inventory in INVENTORIES:
                for member, result in reference[inventory].items():
                    same &= mine[inventory][member]["rows"] == result["rows"]
                    if "schedules" in result:
                        same &= mine[inventory][member]["schedules"] == result["schedules"]
                    checked += 1
    for seed in SEEDS:
        delta = right[("gateb", "default", "D", seed)]["delta_D"]
        reference = h114.read_record(h114.OUTPUT, f"D-{delta}", seed)["inventories"]
        mine = right[("gateb", "default", "D", seed)]["inventories"]
        for inventory in INVENTORIES:
            for member, result in reference[inventory].items():
                same &= mine[inventory][member]["rows"] == result["rows"]
                same &= mine[inventory][member]["schedules"] == result["schedules"]
                checked += 1
    return {"blocks": checked, "pass": bool(same)}


def intervention_integrity(dose):
    """Oracle feedback reads exactly the engine labels (F1 1, fp = fn = 0); removed delivers zeros; the
    realized flip rates lie within 5 binomial SDs of p."""
    out = {}
    ok = True
    for description, table in dose.items():
        oracle = table["oracle"]["feedback_fixed"]["f1"]
        counts = np.asarray(oracle["counts"])
        oracle_ok = bool(counts[:, 1].sum() == 0 and counts[:, 2].sum() == 0
                         and table["oracle"]["feedback_fixed"]["calls"] > 0)
        removed_ok = table["removed"]["feedback_fixed"]["removed_nonzero"] == 0
        flips = {}
        for p in FLIPS:
            entry = table[f"flip{tag(p)}"]["feedback_fixed"]
            n = entry["eligible"]
            flips[str(p)] = {"realized": entry["realized_flip_rate"],
                             "ok": bool(n and abs(entry["realized_flip_rate"] - p) <= 5 * math.sqrt(p * (1 - p) / n))}
        flips_ok = all(f["ok"] for f in flips.values())
        out[description] = {"oracle_exact": oracle_ok, "removed_zero": removed_ok, "flip_rates": flips}
        ok &= oracle_ok and removed_ok and flips_ok
    out["pass"] = bool(ok)
    return out


def records_manifest(output):
    paths = sorted((Path(output) / "records").glob("*.json")) + sorted((Path(output) / "training").glob("updates--*.pt"))
    return {"schema": SCHEMA_MANIFEST, "identity": IDENTITY,
            "files": {str(p.relative_to(output)): sha256_of(p) for p in paths}}


GUARD_FAILURE_NOTE = (f"a failed guard turns every token of this publication into {STOP_TOKEN}: the dispositions "
                      "(overall, contrast and reference J - T tokens), the contrast tokens of every table and the "
                      "wrong-anchor tokens are not interpretable")


def invalidate(obj):
    """Every token below ``obj`` -> STOP_TOKEN (a failed guard): each ``token`` field and every value of a
    ``contrast_tokens`` map (phase-A and phase-B/C guard propagation)."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "token":
                obj[key] = STOP_TOKEN
            elif key == "contrast_tokens":
                obj[key] = {k: STOP_TOKEN for k in value}
            else:
                invalidate(value)
    elif isinstance(obj, list):
        for value in obj:
            invalidate(value)


def apply_guard_failure(compute):
    """The frozen rule on the assembled publication: a failed guard turns every token into STOP_TOKEN (estimates
    unchanged); with every guard passing nothing changes."""
    failed = sorted(k for k, g in compute["guards"].items() if not g["pass"])
    if failed:
        compute["guard_failure"] = {"failed_guards": failed, "note": GUARD_FAILURE_NOTE}
        for key in ("dispositions", "dose_response", "predictor_tables", "wrong_anchor"):
            invalidate(compute[key])
    return compute


def compute_tables(output, plan):
    output = Path(output)
    branches, cells = gb.evaluation_set()
    records = load_records(output)
    right = right_records(records)
    fit = read_json(gb.FIT_VERDICTS)["branches"]
    stats = {i: gb.prior_stats(fit, i) for i in INVENTORIES}
    rows = {i: view_rows(right, cells, i, stats[i]) for i in INVENTORIES}
    prior = {i: v1.ranker_block(rows[i], "prior") for i in INVENTORIES}
    dose = {d: dose_response(right, cells, rows, d) for d in REFERENCE}
    predictors = {d: {name: predictor_table(output, records, right, cells, rows, name, d)
                      for name in dict.fromkeys((*ITEM2[d], *ITEM3[d]))} for d in REFERENCE}
    wrong = {name: wrong_table(records, right, cells, name) for name in PREDICTORS}
    heads_readout = {kind: readout_table(records, f"head-{kind}") for kind in REFERENCE}
    calibrations = {name: [calibration(output, name, s) for s in SEEDS] for name in REFERENCE.values()}
    guards = guard_block(output, plan, records, right, cells, dose, calibrations)
    guards_ok = all(g["pass"] for g in guards.values())
    dispositions = {d: disposition(rows[PRIMARY], d, guards_ok) for d in REFERENCE}
    phases = {p.stem: read_json(p) for p in sorted((output / "ledger").glob("*.json"))}
    smoke_ = read_json(output / "smoke.json")
    all_phases = {"smoke": {"wall_seconds": smoke_["wall_seconds"], "gpu_seconds": smoke_["wall_seconds"]},
                  **{k: {"wall_seconds": v["wall_seconds"], "gpu_seconds": v["gpu_seconds"]} for k, v in phases.items()}}
    compute = {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
        "evaluation": gb.evaluation_accounting(branches, cells),
        "predictors": {name: {k: v for k, v in entry.items() if k != "trained"} for name, entry in PREDICTORS.items()},
        "guards": guards, "dispositions": dispositions,
        "dose_response": dose, "predictor_tables": predictors, "prior": prior, "wrong_anchor": wrong,
        "perception_heads": heads_readout,
        "calibration": {name: [{k: c.get(k) for k in ("seed", "default_f1", "typed_failure")}
                               | {"sigma": {g: c["targets"][g]["sigma"] for g in GAUSS} if not c["typed_failure"] else None,
                                  "reached": {g: c["targets"][g]["reached"] for g in GAUSS}
                                  if not c["typed_failure"] else None,
                                  "calibrated_f1": {g: c["targets"][g]["f1"] for g in GAUSS}
                                  if not c["typed_failure"] else None} for c in cs]
                        for name, cs in calibrations.items()},
        "compute": {"phases": all_phases, "gpu_seconds": sum(b["gpu_seconds"] for b in all_phases.values()),
                    "wall_seconds": sum(b["wall_seconds"] for b in all_phases.values()), "engine_seconds": 0,
                    "hardware": "4 x RTX 5090 (one process per GPU)"},
    }
    return apply_guard_failure(compute)


def guard_block(output, plan, records, right, cells, dose, calibrations):
    payload = rt.load_training_data(RETRAIN)
    eval_roster = {lv["identity"] for lv in gb.evaluation_levels()}
    fit_roster = {lv["identity"] for lv in rt.fit_levels()}
    store_meta = torch.load(truth_path(output), map_location="cpu", weights_only=True, mmap=True)
    scored_at = sorted(r["scored_at"] for r in records.values())
    members = [member_state(output, n, s) for n in TRAINED for s in SEEDS]
    trained_at = sorted(m["trained_at"] for m in members)
    head_reports = [read_json(head_report_path(output, k, s)) for k in REFERENCE for s in SEEDS]
    calibrated_at = sorted(c["calibrated_at"] for cs in calibrations.values() for c in cs)
    policy_reports = [read_json(policy_report_path(output, n, s)) for n in TRAINED for s in SEEDS
                      if policy_report_path(output, n, s).is_file()]
    hashes = (all(m["sha256"] == sha256_of(member_path(output, m["name"], m["seed"])) for m in members)
              and all(r["policy_sha256"] == sha256_of(policy_path(output, r["name"], r["seed"])) for r in policy_reports)
              and all(r["sha256"] == sha256_of(head_path(output, r["kind"], r["seed"])) for r in head_reports)
              and all(c["sha256"] == sha256_of(gb.member_path("unit", c["seed"])) for c in gb.checkpoints("unit")))
    nonfinite = {d: dose[d]["default"]["rankers"][PRIMARY]["J"]["nonfinite_share"] for d in REFERENCE}
    integrity = intervention_integrity(dose)
    free_labels = all(not m["labels"].get("labels_read", True) for m in members if PREDICTORS[m["name"]]["labels"][0]
                      == "none")
    labelled_ok = all(labelled_levels(k) <= fit_roster for k in LOW_K)
    guards = {
        "G1_isolation": {"pass": bool(set(payload["shot_members"]) <= fit_roster and not (fit_roster & eval_roster)
                                      and all(set(cells[i]["mixed"]) <= eval_roster for i in INVENTORIES)
                                      and labelled_ok)},
        "G2_scoring_policy": {"pass": bool(all(r["scoring_policy"] == harness.SCORING_POLICY for r in records.values())
                                           and all(c.get("scoring_policy", harness.SCORING_POLICY)
                                                   == harness.SCORING_POLICY for cs in calibrations.values() for c in cs))},
        "G3_chronology": {"plan_frozen_at": plan["frozen_at"], "first_trained_at": trained_at[0],
                          "last_trained_at": trained_at[-1], "truth_built_at": store_meta["built_at"],
                          "last_calibrated_at": calibrated_at[-1], "first_scored_at": scored_at[0],
                          "pass": bool(plan["frozen_at"] <= min(trained_at[0], *(h["trained_at"] for h in head_reports))
                                       and trained_at[-1] <= scored_at[0] and calibrated_at[-1] <= scored_at[0]
                                       and plan["frozen_at"] <= store_meta["built_at"] <= scored_at[0])},
        "G4_checkpoint_binding": {"pass": bool(hashes)},
        "G5_replication": replication(right),
        "G6_intervention_integrity": integrity,
        "G7_nonfinite": {"shares": nonfinite,
                         "pass": all(v is not None and v <= NONFINITE_SHARE_MAX for v in nonfinite.values())},
        "G8_label_integrity": {"label_free_read_no_labels": free_labels, "labelled_subsets_in_fit": labelled_ok,
                               "pass": bool(free_labels and labelled_ok)},
    }
    return guards


# ---------------------------------------------------------------------------
# renderings
# ---------------------------------------------------------------------------

def cell(block):
    return fmt_interval(block) if block else "n/a"


def num(value, digits=3):
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else format(value, f".{digits}f")


def auc_of(table, inventory, arm):
    return table["rankers"][inventory][arm]["auc"]


def signed(block):
    if not block:
        return "n/a"
    m, (lo, hi) = block["mean"], block["interval"]
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def token_cell(entry):
    return f"{signed(entry['estimate'])} {entry['token'].replace('readiness_or_precision_insufficient', 'insufficient').replace('not_supported_by_this_experiment', 'not supported')}"


def figure_svg(compute):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.rcParams["svg.hashsalt"] = IDENTITY
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for row, description in enumerate(REFERENCE):
        dose = compute["dose_response"][description]
        for col, (label, getter) in enumerate((
                ("grid AUC", lambda t, arm: (t["rankers"][PRIMARY][arm]["auc"] or {}).get("mean")),
                ("rollout error t=225", lambda t, arm: (t["rollout"][arm]["error"] or {}).get("mean")))):
            ax = axes[row, col]
            for arm, marker in (("J", "o"), (f"F-15-{description}", "s"), (f"F-1-{description}", "^")):
                xs, ys, names = [], [], []
                for fb, table in dose.items():
                    f1 = (table["feedback_fixed"]["f1"] or {}).get("mean")
                    y = getter(table, arm)
                    if f1 is None or y is None or (isinstance(f1, float) and math.isnan(f1)):
                        continue
                    xs.append(f1)
                    ys.append(y)
                    names.append(fb)
                ax.scatter(xs, ys, marker=marker, label=arm)
                for x, y, n in zip(xs, ys, names):
                    if arm == "J":
                        ax.annotate(n, (x, y), fontsize=6)
            ax.set_xlabel(f"delivered {description} feedback F1 (fixed requests, grid)")
            ax.set_ylabel(label)
            ax.set_title(f"{description}: {label}")
            ax.legend(fontsize=7)
    fig.tight_layout()
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", metadata={"Date": None})
    plt.close(fig)
    return buffer.getvalue()


def findings_md(plan, compute):
    lines = []
    add = lines.append
    add("# Issue #115: predicate-feedback robustness on the #104 sealed #109 evaluation split")
    add("")
    if "guard_failure" in compute:
        add(f"**Guard failure ({', '.join(compute['guard_failure']['failed_guards'])}): "
            f"{compute['guard_failure']['note']}.**")
        add("")
    add("Intervals: level-clustered percentile bootstrap (10 000 draws, PCG64 7201), DESCRIPTIVE. delta = 0.02. "
        "Tokens in tables: delta superiority rule (supported iff mean >= delta and lower > 0; not supported iff "
        "upper < delta). Zero engine seconds.")
    add("")
    add("## Dispositions (pre-declared, grid)")
    add("")
    add("| description | token | (J-T) label-free - reference | F mean label-free - reference | J label-free - reference "
        "| reference J - T |")
    add("| --- | --- | --- | --- | --- | --- |")
    for d, entry in compute["dispositions"].items():
        add(f"| {d} | `{entry['token']}` | {signed(entry['contrasts']['J-T'])} ({entry['contrast_tokens']['J-T']}) | "
            f"{signed(entry['contrasts']['F_mean'])} ({entry['contrast_tokens']['F_mean']}) | "
            f"{signed(entry['J_label_free_minus_reference'])} | "
            f"{signed(entry['reference_description_value_J_minus_T'])} "
            f"({(entry['reference_description_value_J_minus_T'] or {}).get('token')}) |")
    add("")
    add(f"Rule: {plan['decision_rule']['disposition']}")
    add("")
    for description in REFERENCE:
        dose = compute["dose_response"][description]
        add(f"## Item 1 dose-response ({description}, {REFERENCE[description]})")
        add("")
        add("| feedback | delivered F1 (fixed) | delivered F1 (J) | J | T | D | " +
            " | ".join(fixed_arms(description)) + " | J - T | J - D | J - default | F mean - default |")
        add("| --- " * (3 + len(arms_for(description)) + 4) + "|")
        for fb, table in dose.items():
            g = table["rankers"][PRIMARY]
            c = table["contrasts"][PRIMARY]
            vs = table["versus_default"][PRIMARY] if table["versus_default"] else None
            add(f"| {fb} | {cell(table['feedback_fixed']['f1'])} | {cell(table['feedback_J']['f1'])} | "
                + " | ".join(num((g[a]["auc"] or {}).get("mean")) for a in arms_for(description))
                + f" | {token_cell(c['J-T'])} | {token_cell(c['J-D'])} | "
                + (f"{token_cell(vs['J'])} | {token_cell(vs['F_mean'])} |" if vs else "— | — |"))
        add("")
        add("Grid AUC per arm; feedback F1 is the #100 readout of the delivered adapter input (gated value >= 0.5) "
            "against the executed branch's engine labels at the matching frame. T does not read feedback.")
        add("")
        add(f"Rollout error at t = 225 (grid, #100 engine_position_mse), {description}:")
        add("")
        add("| feedback | " + " | ".join(arms_for(description)) + " |")
        add("| --- " * (1 + len(arms_for(description))) + "|")
        for fb, table in dose.items():
            add(f"| {fb} | " + " | ".join(num((table["rollout"][a]["error"] or {}).get("mean"), 4)
                                          for a in arms_for(description)) + " |")
        add("")
        add(f"Per-inventory AUC of J (offset, angle, power DESCRIPTIVE), {description}:")
        add("")
        add("| feedback | " + " | ".join(INVENTORIES) + " | J - T per inventory | J - D per inventory |")
        add("| --- " * (3 + len(INVENTORIES)) + "|")
        for fb, table in dose.items():
            add(f"| {fb} | " + " | ".join(cell(auc_of(table, i, "J")) for i in INVENTORIES) + " | "
                + "; ".join(signed(table["contrasts"][i]["J-T"]["estimate"]) for i in INVENTORIES) + " | "
                + "; ".join(signed(table["contrasts"][i]["J-D"]["estimate"]) for i in INVENTORIES) + " |")
        add("")
    for item, groups in (("Item 2 training-label noise", ITEM2), ("Item 3 learned and label-free predicates", ITEM3)):
        for description, names in groups.items():
            add(f"## {item} ({description})")
            add("")
            add("| predictor | label F1 | readout parsed | readout predicted | delivered F1 | J | T | D | F mean "
                "| J - T | J - D | J - ref | (J-T) - ref | F mean - ref | rollout J | rollout F-15 |")
            add("| --- " * 16 + "|")
            for name in names:
                t = compute["predictor_tables"][description][name]
                g = t["rankers"][PRIMARY]
                c = t["contrasts"][PRIMARY]
                vs = t["versus_reference"][PRIMARY] if t["versus_reference"] else None
                readout = t["readout"] or {}
                fmean = [((g[a]["auc"] or {}).get("mean")) for a in fixed_arms(description)]
                add(f"| {name} | {num(t['training'].get('label_f1'))} | {cell(readout.get('parsed'))} | "
                    f"{cell(readout.get('predicted'))} | {cell(t['feedback_fixed']['f1'])} | "
                    + " | ".join(num((g[a]["auc"] or {}).get("mean")) for a in ("J", "T", "D"))
                    + f" | {num(float(np.mean(fmean)) if all(v is not None for v in fmean) else None)}"
                    + f" | {token_cell(c['J-T'])} | {token_cell(c['J-D'])} | "
                    + (f"{token_cell(vs['J'])} | {token_cell(vs['J-T'])} | {token_cell(vs['F_mean'])} | " if vs
                       else "— | — | — | ")
                    + f"{num((t['rollout']['J']['error'] or {}).get('mean'), 4)} | "
                    + f"{num((t['rollout'][f'F-15-{description}']['error'] or {}).get('mean'), 4)} |")
            add("")
            add(f"Per-inventory J - T and J - D ({description}):")
            add("")
            add("| predictor | " + " | ".join(f"J - T {i}" for i in INVENTORIES) + " | "
                + " | ".join(f"J - D {i}" for i in INVENTORIES) + " |")
            add("| --- " * (1 + 2 * len(INVENTORIES)) + "|")
            for name in names:
                t = compute["predictor_tables"][description][name]
                add(f"| {name} | " + " | ".join(signed(t["contrasts"][i]["J-T"]["estimate"]) for i in INVENTORIES)
                    + " | " + " | ".join(signed(t["contrasts"][i]["J-D"]["estimate"]) for i in INVENTORIES) + " |")
            add("")
    add("## Per-inventory ranking AUC (grid primary; offset, angle, power DESCRIPTIVE)")
    add("")
    for description in REFERENCE:
        conditions = [(f"{REFERENCE[description]} / {fb}", compute["dose_response"][description][fb])
                      for fb in INTERVENTIONS[description]]
        conditions += [(f"{name} / default", compute["predictor_tables"][description][name])
                       for name in dict.fromkeys((*ITEM2[description], *ITEM3[description]))
                       if name != REFERENCE[description]]
        for inventory in INVENTORIES:
            add(f"### {description}, {inventory} (no-model prior {cell(compute['prior'][inventory]['auc'])})")
            add("")
            add("| condition | " + " | ".join(arms_for(description)) + " | J - T | J - D | J - prior |")
            add("| --- " * (4 + len(arms_for(description))) + "|")
            for label, table in conditions:
                g = table["rankers"][inventory]
                c = table["contrasts"][inventory]
                add(f"| {label} | " + " | ".join(num((g[a]["auc"] or {}).get("mean")) for a in arms_for(description))
                    + f" | {token_cell(c['J-T'])} | {token_cell(c['J-D'])} | {token_cell(c['J-prior'])} |")
            add("")
    add("## Controls (every ranking table above)")
    add("")
    add("| inventory | no-model action prior AUC |")
    add("| --- | --- |")
    for i in INVENTORIES:
        add(f"| {i} | {cell(compute['prior'][i]['auc'])} |")
    add("")
    add("| predictor (J, default feedback) | right AUC | wrong-anchor AUC | right - wrong | stability |")
    add("| --- | --- | --- | --- | --- |")
    for name, w in compute["wrong_anchor"].items():
        add(f"| {name} | {cell(w['right'])} | {cell(w['wrong'])} | {signed(w['right_minus_wrong'])} ({w['token']}) | "
            f"{cell(w['stability'])} |")
    add("")
    add("## Perception heads (item 3 ii, readout on grid branches, parsed carriers)")
    add("")
    for kind, table in compute["perception_heads"].items():
        if table:
            add(f"- {kind}: " + "; ".join(f"{k} {cell(v)}" for k, v in table.items() if isinstance(v, dict)))
    add("")
    add("## Calibration (Gaussian logit noise, fit split)")
    add("")
    for name, rows_ in compute["calibration"].items():
        for r in rows_:
            add(f"- {name} seed {r['seed']}: default F1 {num(r['default_f1'])}; sigma "
                + ", ".join(f"{g} {num(s, 4)} (F1 {num(r['calibrated_f1'][g])}, reached {r['reached'][g]})"
                            for g, s in (r["sigma"] or {}).items()))
    add("")
    add("## Guards")
    add("")
    for k, g in compute["guards"].items():
        add(f"- {k}: {'pass' if g['pass'] else 'FAIL'}")
    add("")
    add(f"Compute: {compute['compute']['gpu_seconds']:.0f} GPU s, {compute['compute']['wall_seconds']:.0f} process "
        f"wall s, 0 engine s.")
    add("")
    return "\n".join(lines)


def summary(plan, compute):
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "validation_command": VALIDATION_COMMAND, "dispositions": compute["dispositions"],
            "guards": {k: v["pass"] for k, v in compute["guards"].items()},
            **({"guard_failure": compute["guard_failure"]} if "guard_failure" in compute else {}),
            "dose_response_grid": {d: {fb: {"feedback_f1": (t["feedback_fixed"]["f1"] or {}).get("mean"),
                                            "J": (t["rankers"][PRIMARY]["J"]["auc"] or {}).get("mean"),
                                            "J-T": t["contrasts"][PRIMARY]["J-T"]["estimate"],
                                            "rollout_J": (t["rollout"]["J"]["error"] or {}).get("mean")}
                                       for fb, t in compute["dose_response"][d].items()} for d in REFERENCE},
            "predictors_grid": {d: {n: {"J": (t["rankers"][PRIMARY]["J"]["auc"] or {}).get("mean"),
                                        "J-T": t["contrasts"][PRIMARY]["J-T"]["estimate"],
                                        "J-D": t["contrasts"][PRIMARY]["J-D"]["estimate"]}
                                    for n, t in compute["predictor_tables"][d].items()} for d in REFERENCE},
            "compute": {k: compute["compute"][k] for k in ("gpu_seconds", "wall_seconds", "engine_seconds")},
            "claim_boundary": plan["claim_boundary"]}


def handoff_106(plan, compute):
    """Written only when a label-free disposition is supported (item 3 iii keeps the description axis)."""
    kept = [d for d, entry in compute["dispositions"].items() if entry["token"] == "supported"]
    if not kept:
        return None
    return {"schema": SCHEMA_HANDOFF, "identity": IDENTITY, "frozen_at": plan["frozen_at"], "to": 106,
            "descriptions": kept,
            "arm": {d: {"predictor": FREE[d], "class": PREDICTORS[FREE[d]]["class"],
                        "recipe": plan["training"]["recipe"], "spec": plan["training"]["arms"][FREE[d]],
                        "checkpoints": [{"seed": m["seed"], "usable": m["usable"], "sha256": m["sha256"],
                                         "path": str(member_path(OUTPUT, FREE[d], m["seed"]).relative_to(ROOT))}
                                        for m in compute["predictor_tables"][d][FREE[d]]["training"]["members"]]}
                    for d in kept},
            "dispositions": {d: compute["dispositions"][d] for d in kept},
            "note": ("the label-free discrete bottleneck (binary codes trained only through the prediction loss) is "
                     "offered to #106 as an extra description arm for CLEVRER / Physion; #106 does not wait for it")}


def rendered(plan, compute, output=OUTPUT):
    out = {"compute.json": json_text(compute), "summary.json": json_text(summary(plan, compute)),
           "findings.md": findings_md(plan, compute), "dose-response.svg": figure_svg(compute),
           "records-manifest.json": json_text(records_manifest(output))}
    handoff = handoff_106(plan, compute)
    if handoff is not None:
        out["handoff_106.json"] = json_text(handoff)
    return out


# ---------------------------------------------------------------------------
# smoke (pre-freeze; fit split and published artefacts only; no evaluation capture read)
# ---------------------------------------------------------------------------

def smoke_training_path(scratch, payload):
    """The default spec through this runner's path equals the retrain's train_group bit for bit."""
    torch.backends.cudnn.benchmark = True
    base = {"relations": payload["relations"], "relations_mask": payload["relation_mask"],
            "macros": payload["macros"], "macros_mask": payload["macro_mask"]}
    data = VariantTrainData(payload, base)
    _, a = implementation("smoke-default").train_group(scratch, ARM, "smoke-default", 0, [("all", SEEDS[0])], data,
                                                       steps=SMOKE_STEPS, save=False, smoke_full=True)
    params_a = {k: v.detach().clone() for k, v in a.params.items()}
    local_a = a.log_tensors()["local"].clone()
    del a, data
    torch.cuda.empty_cache()
    data = rt.IndexedTrainData(payload, "hybrid")
    _, b = rt.implementation().train_group(scratch, ARM, "hybrid", 0, [("all", SEEDS[0])], data,
                                           steps=SMOKE_STEPS, save=False, smoke_full=True)
    delta = max(float((params_a[k] - b.params[k]).abs().max()) for k in params_a)
    same_local = bool(torch.equal(local_a, b.log_tensors()["local"]))
    del b, data
    torch.cuda.empty_cache()
    return {"control": "default_training_path_equals_retrain", "updates": SMOKE_STEPS, "max_parameter_delta": delta,
            "local_losses_equal": same_local, "ok": delta == 0.0 and same_local}


def smoke_variants(scratch, payload):
    """Every label construction (seed 0) and a full-cost training smoke per predictor class / spec."""
    out = []
    timings = {}
    for name in ("mi-flip05", "mi-low10", "mi-learned", "mi-free", "ma-ref", "ma-flip05", "ma-low10", "ma-learned",
                 "ma-free"):
        began = time.monotonic()
        symbols, report = variant_symbols(scratch, name, SEEDS[0], payload)
        build = time.monotonic() - began
        data = VariantTrainData(payload, symbols)
        del symbols
        torch.cuda.reset_peak_memory_stats()
        result, ensemble = implementation(name).train_group(scratch, ARM, name, 0, [("all", SEEDS[0])], data,
                                                            steps=9 + SMOKE_STEPS, save=False, smoke_full=True,
                                                            timing_after=9)
        finite = bool(torch.isfinite(torch.tensor(result["trace"][-1]["loss"])).all())
        timings[name] = result["timed_seconds_per_update"]
        out.append({"name": name, "label_report": {k: v for k, v in report.items() if k != "label_counts"},
                    "build_seconds": build, "seconds_per_update_full_cost": result["timed_seconds_per_update"],
                    "parameters": result["parameters_per_member"], "peak_memory_gb": result["peak_memory_gb"],
                    "final_loss_finite": finite})
        del ensemble, data
        torch.cuda.empty_cache()
    hybrid = sum(p.numel() for p in pf.CNNHybridPredictor().parameters())
    ok = all(r["final_loss_finite"] for r in out)
    flip = next(r for r in out if r["name"] == "mi-flip05")["label_report"]
    ok &= abs(flip["realized_rate"] - 0.05) < 0.001
    ok &= all(r["label_report"].get("labels_read", True) is False for r in out if r["name"].endswith("free"))
    return {"control": "variant_labels_and_training", "arms": out, "hybrid_parameters": hybrid, "ok": bool(ok)}, timings


def smoke_heads(scratch, payload):
    """Short head trainings (micro and macro) with threshold selection and pseudo-labels."""
    began = time.monotonic()
    head, extra = train_micro_head(payload, SEEDS[0], steps=50)
    save_head(scratch, "micro", SEEDS[0], head, extra)
    relations, mask = micro_pseudo_labels(head, payload["carriers"][:4096], extra["thresholds"])
    micro_ok = relations.shape == (4096, SLOTS, SLOTS, 2) and bool(torch.equal(relations[..., 0],
                                                                                relations[..., 0].transpose(1, 2)))
    micro_seconds = time.monotonic() - began
    began = time.monotonic()
    data = om113.load_data(MACRO113, "fit")
    level_index = {level: i for i, level in enumerate(data["levels"])}
    subset = torch.tensor([level_index[m] for m in sorted(labelled_levels(HEAD_K))], device=DEVICE)
    train_rows = torch.isin(data["row_level"], subset)
    mhead, report = om113.train_head(data, train_rows, MACRO_HEAD["history"], SEEDS[0], MACRO_HEAD["mix"], steps=20)
    with harness.deterministic_scoring():
        thresholds, _ = macro_head_thresholds(data, mhead, train_rows)
    rows = payload_rows113(payload)
    values = macro_pseudo_labels(data, mhead, thresholds, rows[:50000])
    available_rule = all(bool(torch.equal(data["available"][rows[:200000], :, d].cpu(),
                                          pf.macro_available(data["position"][rows[:200000]].cpu(), delta)))
                         for d, delta in enumerate(DELTAS))
    atomic_torch(pseudo_path(scratch, SEEDS[0]), {"schema": SCHEMA_PSEUDO, "plan_identity": IDENTITY, "seed": SEEDS[0],
                                                  "values": torch.zeros(len(rows), 2, len(DELTAS), dtype=torch.bool)})
    save_head(scratch, "macro", SEEDS[0], mhead, {"report": report, "thresholds": thresholds.tolist()})
    del data
    torch.cuda.empty_cache()
    return {"control": "perception_heads", "micro_seconds_50_steps": micro_seconds,
            "macro_seconds_20_steps": time.monotonic() - began, "micro_thresholds": extra["thresholds"],
            "macro_thresholds": thresholds.tolist(), "macro_rows_mapped": len(rows),
            "payload_positions": int((payload["lasts"] + 1).sum()), "macro_available_is_positional": available_rule,
            "pseudo_shape": list(values.shape),
            "ok": bool(micro_ok and available_rule and len(rows) == int((payload["lasts"] + 1).sum()))}


def smoke_policy(payload):
    """Round-0 J labels of this runner's path equal #101's on the first policy units (Gate-B seed 0)."""
    pool = p101.load_pool(POLICY)
    model = p101.load_predictor(SEEDS[0])
    reference = torch.load(p101.group_root(POLICY, "pe", ICLR_LAMBDA, SEEDS[0]) / "labels-0.pt", map_location="cpu",
                           weights_only=True)["units"]
    began = time.monotonic()
    with harness.deterministic_scoring():
        policy = rp.new_policy(PAIRS, SEEDS[0], device=DEVICE)
        mine = [rp.window_rows(model, policy, unit, aggregated=False, compute_weight=ICLR_LAMBDA, device=DEVICE)
                for unit in p101.window_units(pool)[:2]]
    seconds = (time.monotonic() - began) / 2 * len(p101.window_units(pool))
    same = all(torch.equal(mine[u][k].cpu(), reference[u][k].cpu()) for u in range(2) for k in mine[u])
    return {"control": "policy_labels_equal_101", "units": 2, "projected_label_round_seconds": seconds, "ok": bool(same)}


def smoke_rollouts():
    """On fit grid cells: default feedback through ``pf.rollout_endpoints`` equals ``rs.rollout_endpoints``
    (J, T, D, fixed micro); every intervention runs; oracle F1 is 1; removed is zero; flips hit p."""
    cells = sorted(fit_grid_cells().items())[:SMOKE_CELLS]
    anchors = gb.fit_anchors()
    model = p101.load_predictor(SEEDS[0])
    policy = p101.load_policy(POLICY, "pe", ICLR_LAMBDA, SEEDS[0])
    equal = True
    interventions = {}
    with harness.deterministic_scoring():
        for member, entry in cells:
            actions = m99.action_batch(entry["inventory"])
            z0 = anchors[member].to(DEVICE)
            truth_ = fit_truth([item["key"] for item in entry["inventory"]])
            for arm in arms_for("micro"):
                spec = arm_spec(arm, 1)
                use = policy if arm in SCHEDULED else None
                a, ta = rs.rollout_endpoints(model, use, z0, actions, spec, select=t96.select_pairs,
                                             record_schedules=True)
                feedback = pf.Feedback("default", Abstraction.MICRO, truth=truth_, device=DEVICE)
                b, tb = pf.rollout_endpoints(model, use, z0, actions, spec, select=t96.select_pairs, feedback=feedback,
                                             record_schedules=True)
                equal &= bool(torch.equal(a, b)) and ta["schedules"] == tb["schedules"]
            for fb in INTERVENTIONS["micro"]:
                params = feedback_parameters(fb)
                generator = torch.Generator(device=DEVICE).manual_seed(block_seed("smoke", fb, member))
                kwargs = {"truth": truth_, "device": DEVICE}
                if params["kind"] == "flip":
                    feedback = pf.Feedback("flip", Abstraction.MICRO, p=params["p"], generator=generator, **kwargs)
                elif params["kind"] == "gauss":
                    feedback = pf.Feedback("gauss", Abstraction.MICRO, sigma=params["fraction"], generator=generator,
                                           **kwargs)
                elif params["kind"] == "shuffled":
                    partner = ((torch.arange(len(actions)) + 1) % len(actions)).to(DEVICE)
                    feedback = pf.Feedback("shuffled", Abstraction.MICRO, partner=partner, **kwargs)
                else:
                    feedback = pf.Feedback(params["kind"], Abstraction.MICRO, **kwargs)
                z, _ = pf.rollout_endpoints(model, policy, z0, actions, {"kind": "joint"}, select=t96.select_pairs,
                                            feedback=feedback)
                z2, _ = pf.rollout_endpoints(model, None, z0, actions, t96.arm_spec("F-5-micro"), feedback=feedback)
                entry_ = interventions.setdefault(fb, {"counts": torch.zeros(len(DELTAS), 2, 3, dtype=torch.long),
                                                       "finite": True, "flipped": 0, "eligible": 0, "nonzero": 0})
                entry_["counts"] += feedback.counts.cpu()
                entry_["finite"] &= bool(torch.isfinite(z).all() and torch.isfinite(z2).all())
                entry_["flipped"] += feedback.flipped
                entry_["eligible"] += feedback.eligible
                entry_["nonzero"] += feedback.nonzero
    summary_ = {fb: {"f1": feedback_f1(e["counts"]), "finite": e["finite"],
                     "flip_rate": e["flipped"] / e["eligible"] if e["eligible"] else None, "removed_nonzero": e["nonzero"]}
                for fb, e in interventions.items()}
    oracle = interventions["oracle"]["counts"].sum(0)
    ok = (equal and int(oracle[:, 1].sum()) == 0 and int(oracle[:, 2].sum()) == 0
          and summary_["removed"]["removed_nonzero"] == 0 and all(s["finite"] for s in summary_.values()))
    return {"control": "feedback_rollouts_fit", "cells": len(cells), "default_equals_selectors": bool(equal),
            "interventions": summary_, "ok": bool(ok)}


def smoke_macro_rollouts(scratch):
    """Macro interventions (incl. the #113 temporal head) on a Delta-macro predictor (smoke weights)."""
    cells = sorted(fit_grid_cells().items())[:2]
    anchors = gb.fit_anchors()
    torch.manual_seed(0)
    model = pf.DeltaMacroPredictor().to(DEVICE).eval()
    head, thresholds = temporal_head(SEEDS[0]), temporal_thresholds().to(DEVICE)
    out = {}
    with harness.deterministic_scoring():
        for member, entry in cells:
            actions = m99.action_batch(entry["inventory"])
            truth_ = fit_truth([item["key"] for item in entry["inventory"]])
            for fb in INTERVENTIONS["macro"]:
                params = feedback_parameters(fb)
                generator = torch.Generator(device=DEVICE).manual_seed(block_seed("smoke-macro", fb, member))
                kwargs = {"truth": truth_, "device": DEVICE}
                if params["kind"] == "flip":
                    feedback = pf.Feedback("flip", Abstraction.MACRO, p=params["p"], generator=generator, **kwargs)
                elif params["kind"] == "gauss":
                    feedback = pf.Feedback("gauss", Abstraction.MACRO, sigma=1.0, generator=generator, **kwargs)
                elif params["kind"] == "shuffled":
                    partner = ((torch.arange(len(actions)) + 1) % len(actions)).to(DEVICE)
                    feedback = pf.Feedback("shuffled", Abstraction.MACRO, partner=partner, **kwargs)
                elif params["kind"] == "temporal":
                    feedback = pf.Feedback("temporal", Abstraction.MACRO, temporal=head, thresholds=thresholds,
                                           rows=len(actions), **kwargs)
                else:
                    feedback = pf.Feedback(params["kind"], Abstraction.MACRO, **kwargs)
                for delta in DELTAS:
                    z, _ = pf.rollout_endpoints(model, None, anchors[member].to(DEVICE), actions,
                                                t96.arm_spec(f"F-{delta}-macro"), feedback=feedback)
                    out.setdefault(fb, True)
                    out[fb] &= bool(torch.isfinite(z).all())
            a, _ = rs.rollout_endpoints(model, None, anchors[member].to(DEVICE), actions, t96.arm_spec("F-5-macro"))
            b, _ = pf.rollout_endpoints(model, None, anchors[member].to(DEVICE), actions, t96.arm_spec("F-5-macro"),
                                        feedback=pf.Feedback("default", Abstraction.MACRO, truth=truth_, device=DEVICE))
            out["default_equals_selectors"] = out.get("default_equals_selectors", True) and bool(torch.equal(a, b))
    return {"control": "macro_feedback_rollouts_fit", "results": out, "ok": all(out.values())}


def smoke_truth():
    """truth_worker equals the retrain shot cache's #99 relations on fit branches."""
    branches = {b["key"]: b for b in rt.complete_branches()}
    keys = [item["key"] for _, entry in sorted(fit_grid_cells().items())[:1] for item in entry["inventory"][:2]]
    began = time.monotonic()
    same = True
    for key in keys:
        mine = truth_worker(branches[key]["attempt"])
        shot = rt.read_shot(RETRAIN, key)
        last = mine["last"]
        same &= (torch.equal(mine["relations"], shot["relations"][:last + 1].bool())
                 and torch.equal(mine["relation_mask"], shot["relation_mask"][:last + 1].bool()))
    return {"control": "truth_worker_equals_shot_cache", "branches": len(keys),
            "seconds_per_shot": (time.monotonic() - began) / len(keys), "ok": bool(same)}


def smoke_readout(scratch):
    """Readout counts on fit branches (shot-cache carriers posing as parsed carriers) run for both
    descriptions; the micro counts equal r100.readout_counts' contact/supports on the same rows."""
    cells = sorted(fit_grid_cells().items())[:1]
    model = p101.load_predictor(SEEDS[0])
    truth_ = None
    same = True
    with harness.deterministic_scoring(), torch.no_grad():
        for member, entry in cells:
            item = entry["inventory"][0]
            shot = rt.read_shot(RETRAIN, item["key"])
            last = min(int(shot["end"]), ENDPOINT)
            truth_ = fit_truth([item["key"]])
            labels = {"last": last, "action": m99.action_batch([item]),
                      "relations": truth_.relations[0, :last + 1], "relation_mask": truth_.relation_mask[0, :last + 1],
                      "macro_values": truth_.macro_values[0, :last + 1],
                      "macro_available": truth_.macro_available[0, :last + 1]}
            z = shot["carriers"][:last + 1].float().to(DEVICE)
            part = readout_counts_block(model, "micro", z, labels)
            reference, _, _ = v1.r100.readout_counts(model, z, {
                "phase": torch.zeros(last + 1, dtype=torch.long), "presence": torch.zeros(last + 1, SLOTS, dtype=torch.bool),
                "relations": labels["relations"].cpu(), "relation_mask": labels["relation_mask"].cpu(),
                "macros": torch.zeros(last + 1, 2, dtype=torch.bool), "macro_mask": torch.zeros(last + 1, 2, dtype=torch.bool)},
                list(range(last + 1)), s99.vocabulary())
            mine = part["parsed"].cpu().tolist()
            same &= mine[0] == reference["contact"][0][:3] and mine[1] == reference["supports"][0][:3]
            macro_model = pf.DeltaMacroPredictor().to(DEVICE).eval()
            macro_part = readout_counts_block(macro_model, "macro", z, labels)
    return {"control": "readout_counts_equal_100", "ok": bool(same and len(macro_part) == 2 * len(DELTAS))}


def smoke_calibration():
    """The bisection on Gate-B seed 0 (2 cells, 3 iterations) runs and decreases F1 with sigma."""
    began = time.monotonic()
    model = p101.load_predictor(SEEDS[0])
    cells = sorted(fit_grid_cells().items())[:2]
    anchors = gb.fit_anchors()
    truths = {member: fit_truth([item["key"] for item in entry["inventory"]]) for member, entry in cells}
    with harness.deterministic_scoring():
        default, _ = calibration_f1(model, "micro", cells, anchors, truths, None, SEEDS[0])
        low, _ = calibration_f1(model, "micro", cells, anchors, truths, 0.01, SEEDS[0])
        high, _ = calibration_f1(model, "micro", cells, anchors, truths, 100.0, SEEDS[0])
    seconds = (time.monotonic() - began) / 3
    return {"control": "calibration_curve", "default_f1": default, "f1_sigma_0.01": low, "f1_sigma_100": high,
            "seconds_per_evaluation_2_cells": seconds, "ok": bool(high < default)}


def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the smoke is a pre-freeze control")
    output.mkdir(parents=True, exist_ok=True)
    scratch = output / "smoke-scratch"
    shutil.rmtree(scratch, ignore_errors=True)
    began = time.monotonic()
    if not payload_hash_ok():
        raise ValueError("the retrain training payload differs from its published hash")
    payload = rt.load_training_data(RETRAIN)
    checks = [smoke_training_path(scratch, payload), smoke_heads(scratch, payload)]
    variants, timings = smoke_variants(scratch, payload)
    checks.append(variants)
    checks.append(smoke_policy(payload))
    with harness.deterministic_scoring():
        checks.append(smoke_rollouts())
        checks.append(smoke_macro_rollouts(scratch))
    checks.append(smoke_truth())
    checks.append(smoke_readout(scratch))
    checks.append(smoke_calibration())
    gate = read_json(GATE_B / "summary.json")
    checks.append({"control": "inputs_bound", "gate_b": gate["gate_b"], "J": p101.J_ICLR,
                   "ok": bool(gate["gate_b"] == "pass" and p101.J_ICLR == "J-pe-1e-04")})
    ok = all(c["ok"] for c in checks)
    wall = time.monotonic() - began
    evidence = {"schema": SCHEMA_SMOKE, "identity": IDENTITY, "run_at": utc_now(), "checks": checks,
                "projection": projection(checks, timings), "wall_seconds": wall,
                "reading": ("fit-split cells, anchors, shots and labels, the #101 policies, the #113 heads; no "
                            "evaluation-split capture or model output"), "ok": ok}
    write_json(output / "smoke.json", evidence)
    shutil.rmtree(scratch, ignore_errors=True)
    log(f"smoke {'PASSED' if ok else 'FAILED'} ({wall:.0f}s); projection {evidence['projection']}")
    for check in checks:
        log(f"  {check['control']}: {check['ok']}")
    if not ok:
        raise ValueError("smoke control differs; STOP RULE: no freeze")
    return 0


def projection(checks, timings):
    """GPU seconds: training at the full-cost update time (upper bound), policies at the measured label
    time x 2 rounds + distillation, scoring from the #101/#105 per-record ledgers, calibration from the
    smoke evaluation time."""
    per_update = float(np.mean(list(timings.values())))
    training = STEPS * per_update * len(TRAINED) * len(SEEDS)
    label_round = next(c for c in checks if c["control"] == "policy_labels_equal_101")["projected_label_round_seconds"]
    policies = (2 * label_round + 60.0) * len(TRAINED) * len(SEEDS)
    per_arm = {"J": 80.0, "T": 80.0, "D": 80.0, "F-1": 95.0, "F-5": 35.0, "F-15": 8.0, "wrong": 40.0,
               "readout": 60.0}
    scoring = 0.0
    for kind, name, fb, arm, seed in score_tasks():
        if kind == "right":
            scoring += 1.3 * per_arm[arm if arm in per_arm else arm.rsplit("-", 1)[0]]
        else:
            scoring += per_arm[kind]
    cal = next(c for c in checks if c["control"] == "calibration_curve")["seconds_per_evaluation_2_cells"]
    calibration_ = cal * CAL_CELLS / 2 * (2 + CAL_ITERATIONS) * len(GAUSS) * len(REFERENCE) * len(SEEDS)
    heads_ = 6 * 600.0
    total = training + policies + scoring + calibration_ + heads_
    return {"training_gpu_seconds_upper": training, "policies_gpu_seconds": policies,
            "scoring_gpu_seconds": scoring, "calibration_gpu_seconds": calibration_, "heads_gpu_seconds": heads_,
            "total_gpu_seconds": total, "elapsed_hours_on_4_gpus": total / 4 / 3600,
            "within_limit": total / 4 / 3600 <= ELAPSED_LIMIT_HOURS,
            "note": "training projected at the full-cost (horizon 225 + ranking) update time: an upper bound"}


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def input_bindings(output):
    from scripts import run_horizon_vs_joint as h114
    names = {
        "issue_112_retrain_handoff": gb.HANDOFF, "issue_112_fit_verdicts": gb.FIT_VERDICTS,
        "issue_112_fit_anchors": rt.anchors_path(RETRAIN), "issue_112_retrain_compute": RETRAIN / "compute.json",
        "issue_112_retrain_plan": RETRAIN / "plan.json",
        "gate_b_plan": GATE_B / "plan.json", "gate_b_summary": GATE_B / "summary.json",
        "gate_b_anchors": gb.anchors_path(GATE_B),
        "issue_101_plan": POLICY / "plan.json", "issue_101_selection": POLICY / "selection.json",
        "issue_101_pool": p101.pool_path(POLICY),
        **{f"issue_101_policy_pe_1e-04_{s}": p101.policy_path(POLICY, "pe", ICLR_LAMBDA, s) for s in SEEDS},
        **{f"issue_101_report_pe_1e-04_{s}": p101.report_path(POLICY, "pe", ICLR_LAMBDA, s) for s in SEEDS},
        "issue_114_plan": h114.OUTPUT / "plan.json",
        "issue_113_head_plan": MACRO113 / "plan.json", "issue_113_selection": MACRO113 / "selection.json",
        "issue_113_fit_meta": MACRO113 / "data" / "fit.meta.json",
        "issue_113_evaluation_meta": MACRO113 / "data" / "evaluation.meta.json",
        **{f"issue_113_head_{s}": om113.head_path(MACRO113, om113.head_name("final", TEMPORAL_FEEDBACK, s))
           for s in SEEDS},
        "gate_b_runner": ROOT / gb.RUNNER, "issue_112_retrain_runner": ROOT / rt.RUNNER,
        "issue_112_v1_runner": ROOT / v1.RUNNER, "issue_112_versions_runner": ROOT / versions.RUNNER,
        "issue_101_runner": ROOT / p101.RUNNER, "issue_113_runner": ROOT / om113.RUNNER,
        "issue_114_runner": ROOT / "scripts/run_horizon_vs_joint.py",
        "issue_99_attribution_runner": ROOT / "scripts/run_decision_chain_attribution.py",
        "issue_96_runner": ROOT / "scripts/run_tau_ad_within_checkpoint.py",
        "issue_100_runner": ROOT / "scripts/run_readout_fidelity_jepa.py",
        "issue_99_runner": ROOT / "scripts/run_slot_encoder_fix.py",
        "issue_109_cohort_plan": rt.COHORT, "issue_104_plan": rt.POWER_PLAN,
        "issue_104_campaign_branches": rt.CAMPAIGN / "branches.json",
        "scoring_harness": ROOT / "world_model/planning/scoring_harness.py",
        "predicate_feedback_module": ROOT / MODULE,
        "request_selectors_module": ROOT / "world_model/training/request_selectors.py",
        "request_policy_module": ROOT / p101.MODULE,
        "temporal_head_module": ROOT / om113.HEAD_MODULE, "observable_macro_module": ROOT / om113.LABEL_MODULE,
        "matched_dynamics": ROOT / "world_model/training/matched_dynamics.py",
        "decision_dynamics": ROOT / "world_model/training/decision_dynamics.py",
        "stable_decision_dynamics": ROOT / "world_model/training/stable_decision_dynamics.py",
        "cnn_hybrid": ROOT / "world_model/training/cnn_hybrid.py",
        "smoke": Path(output) / "smoke.json",
    }
    names.update({f"checkpoint_unit_{c['seed']}": gb.member_path("unit", c["seed"]) for c in gb.checkpoints("unit")})
    return [{"name": name, "artifact": str(Path(path).relative_to(ROOT)), "sha256": sha256_of(path)}
            for name, path in names.items()]


def frozen_rules():
    return {
        "disposition": (
            "per description, on grid, for 'the description axis keeps its value without engine-labelled predicates': "
            "label-free bottleneck predictor (mi-free / ma-free) vs the engine-labelled reference (Gate-B / ma-ref), "
            "paired by (level, seed), member-clustered: C1 = (J - T)_free - (J - T)_ref and C2 = the mean over "
            "Delta in {1,5,15} of F-Delta-alpha_free - F-Delta-alpha_ref; supported iff both lower bounds > -delta "
            "(non-inferiority at delta = 0.02); not_supported_by_this_experiment iff either upper bound < -delta; "
            f"otherwise {STOP_TOKEN}; any failed guard -> {STOP_TOKEN}. The reference's own J - T (delta "
            "superiority token) and J_free - J_ref are reported beside it"),
        "contrast_tokens": (f"delta superiority, DESCRIPTIVE: supported iff mean >= {DELTA} with lower bound > 0; "
                            f"not_supported_by_this_experiment iff upper < {DELTA}; otherwise {STOP_TOKEN}; a failed "
                            f"guard turns every token into {STOP_TOKEN}"),
        "rollout_error": ("DESCRIPTIVE: #100 engine_position_mse at t = 225 on every mixed grid candidate's own branch; "
                          "branch mean per (level, seed), seed mean per level, level bootstrap; paired per level"),
        "feedback_f1": ("the #100 readout (gated value >= 0.5; contact on the upper triangle, supports directed; macro "
                        "per predicate on available frames) of the adapter input actually delivered at every "
                        "description request of the arm's rollouts, against the executed branch's engine labels at "
                        "the matching frame (clamped to the branch's last frame); pooled counts, mean over the two "
                        "channels; level bootstrap"),
        "readout_f1": ("the #100 audit form on the grid branches of the evaluation split: the predictor's own head on "
                       "the E99 parse of every frame 0..min(225, last) (parsed) and after one Delta-step prediction "
                       "from the parsed frame f - Delta (predicted); macro on the #113 predicates of the request's "
                       "Delta column"),
        "D_arm": "D = description-only restriction of J at J's modal Delta on the policy split (#96 rule, ties lower)",
        "T_arm": "T = horizon-only restriction of J (T-continuous, #114): no description request, no feedback",
        "gauss_calibration": (f"sigma per target fraction {list(GAUSS_FRACTIONS)} of the default delivered F1, bisection "
                              f"in log10 sigma on [1e-2, 1e2] ({CAL_ITERATIONS} halvings), on the fixed description "
                              f"requests from the first {CAL_CELLS} fit-split grid cells (sorted level), fit engine "
                              "labels; a target outside the reachable range takes the nearer end (flagged)"),
    }


def frozen_plan(output, frozen_at, smoke_evidence):
    branches, cells = gb.evaluation_set()
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "issue": 115,
        "related_issues": [85, 96, 99, 100, 101, 104, 106, 108, 109, 111, 112, 113, 114, 117],
        "role": ("predicate-feedback interventions, label-noise / low-label / learned / label-free retrains of the LC "
                 "hybrid, micro and macro; ranking confirmatory on the sealed #109 evaluation split (Gate B passed)"),
        "frozen_at": frozen_at, "frozen_before_training_and_scoring": True,
        "validation_command": VALIDATION_COMMAND, "runner": RUNNER, "runner_sha256_at_freeze": sha256_of(ROOT / RUNNER),
        "modules": {"predicate_feedback": {"path": MODULE, "sha256_at_freeze": sha256_of(ROOT / MODULE)}},
        "engine_seconds": 0,
        "question": ("how much does the description axis depend on the quality of the predicates fed back at runtime "
                     "and on engine supervision of the predicate heads, up to heads trained with no engine labels?"),
        "predictors": PREDICTORS,
        "interventions": {
            "micro_on": "gateb", "macro_on": "ma-ref", "kinds": INTERVENTIONS,
            "definitions": {
                "default": "the model's own heads (symbolic_features of the current carrier)",
                "oracle": ("engine labels of the executed candidate branch at the matching frame (micro: #99 relations "
                           "& relation_mask, channel availability = any labelled pair; macro: the #113 predicate of "
                           "the request's Delta and its positional availability), through the same adapter"),
                "flipXX": "the oracle with symmetric flips at p = XX/100 (contact per unordered pair), drawn per call",
                "gaussXX": "the model's logits + sigma N(0,1) (contact noise symmetric), sigma calibrated to XX % of "
                           "the default delivered F1 on the fit split",
                "removed": "adapter input zero", "shuffled": "the model's feedback of the next candidate (ordinal, "
                           "cyclic) of the same anchor at the same iteration",
                "temporal": (f"macro: the #113 temporal head {TEMPORAL_FEEDBACK} (seed-paired) on the row's own "
                             "stride-Delta rollout history (latest visited state at or before each history frame), "
                             "binary at the #113 selected predicted-source thresholds, positional availability")},
            "random_streams": "per (predictor, feedback, seed, arm, inventory, level): CUDA generator seeded by sha256"},
        "arms": {"J": "#101 pe DP lambda 1e-4 policy trained for each predictor (Gate-B: the #101 J itself)",
                 "T": frozen_rules()["T_arm"], "D": frozen_rules()["D_arm"],
                 "fixed": "F-{1,5,15}-alpha of the predictor's description",
                 "controls": {"prior": "no-model action prior (Gate-B rule, fit-verdicts.json)",
                              "wrong_anchor": "J from the 480 fit-split anchors over the 16 grid candidates"}},
        "training": {
            "recipe": ("#112 v3 LC exactly as the Gate-B retrain (rt.implementation): #77 local loss + long-horizon "
                       "supervision to 225 + decision-contrastive ranking loss, truncation 45, skip-on-nonfinite "
                       "(patience 10), AdamW lr 1e-4 wd 1e-4, clip 1, batch 64, 9000 updates, PAIRS[step % 9], "
                       "init torch.manual_seed(seed), fold 'all', one member per group; symbolic loss weight 0.1, "
                       "pos_weight 5 (pf.local_loss = decision_dynamics.local_loss under the default spec)"),
            "data": "the v3 retrain payload (#109 fit split: 13 919 shots, 680 ranking cells), hash-checked",
            "seeds": list(SEEDS),
            "arms": {name: {"class": e["class"], "labels": e["labels"],
                            "spec": {"micro": symbolic_spec(name).micro, "macro": symbolic_spec(name).macro,
                                     "macro_delta": symbolic_spec(name).macro_delta,
                                     "weighted": sorted(symbolic_spec(name).weighted)}}
                     for name, e in PREDICTORS.items() if e["trained"]},
            "label_constructions": {
                "flip": ("symmetric flips of the engine labels on labelled entries at rate p, one draw per (arm, seed) "
                         "from torch.Generator(sha256('labels', arm, seed)); micro contact per unordered pair"),
                "low": (f"the description's symbolic terms (labels and availability) weighted by the position's level "
                        f"being in the labelled subset: per generator family the sorted fit levels permuted by "
                        f"PCG64({LABEL_SEED}), first ceil(k % x n) labelled (nested)"),
                "learned": (f"micro: MicroPerceptionHead (the micro head's architecture) trained on the {HEAD_K} % "
                            f"subset's v3 carriers and engine labels ({HEAD_TRAIN}), thresholds per channel = "
                            "in-sample best F1 over the #113 grid; targets = readings on every payload position, mask "
                            "= both slots present in the carrier and the head's channel availability. macro: the #113 "
                            f"temporal head recipe ({MACRO_HEAD}) on the {HEAD_K} % subset rows of the #113 fit data, "
                            "thresholds per (predicate, Delta) = in-sample best engine F1; targets on every payload "
                            "position, positional availability"),
                "none": "no symbolic loss for the description; binary straight-through codes (pf.binary_ste)"},
            "macro_labels": ("the #113 frozen observable predicates (settle_Delta, structure-change_Delta) of every "
                             "payload position, Delta-indexed macro head (4 outputs per Delta: +1032 parameters)"),
            "policy": ("#101 pe recipe: rp.new_policy(seed), 2 DAgger rounds of rp.window_rows (lambda 1e-4) on the "
                       "#101 pool, rp.distil_round, under deterministic_scoring; policy-split evaluation fixes D"),
            "usable": f"complete, at most {SKIP_CAP} skipped updates, no #117 end-of-training excursion (rt.excursion)",
            "typed_failures": "an unusable member is a typed terminal failure: not retrained, no policy, its cells drop",
        },
        "evaluation": {"split": "the #109 evaluation split as in Gate B (stream issue109-evaluation, mixed cells)",
                       "accounting": gb.evaluation_accounting(branches, cells),
                       "anchors": "the Gate-B E99 anchor store", "primary_inventory": PRIMARY,
                       "descriptive": list(gb.DESCRIPTIVE_INVENTORIES),
                       "cost": "tie_free (EndpointCosts.row against the anchor)",
                       "cell_auc": "m99.view; cells (level, seed); bootstrap over levels (10000 draws, PCG64 7201)",
                       "engine_truth": ("built after the freeze: s99.shot_labels of every mixed-cell branch at positions "
                                        "0..min(225, last); #113 predicates and carriers of the same frames")},
        "decision_rule": frozen_rules(),
        "guards": {"G1_isolation": "payload / labelled subsets / perception heads on fit levels; scored cells evaluation",
                   "G2_scoring_policy": "every record and calibration carries the #111 SCORING_POLICY",
                   "G3_chronology": "frozen_at <= heads, training <= calibration <= first score; truth after the freeze",
                   "G4_checkpoint_binding": "Gate-B, trained member, policy and head hashes equal their reports",
                   "G5_replication": ("Gate-B default-feedback J, fixed micro, T-continuous and D rows (and schedules) "
                                      "equal the #101 / #114 records exactly"),
                   "G6_intervention_integrity": ("oracle delivered F1 exactly 1 (fp = fn = 0), removed delivers zeros, "
                                                 "realized flip rates within 5 binomial SDs of p"),
                   "G7_nonfinite": f"Gate-B and ma-ref default J grid nonfinite share <= {NONFINITE_SHARE_MAX}",
                   "G8_label_integrity": "label-free arms read no labels; labelled subsets inside the fit roster",
                   "failure": f"a failed guard turns every token into {STOP_TOKEN}"},
        "prohibited": ["reading evaluation captures for training, calibration or selection", "re-scoring after "
                       "publication", "changing an arm, rule or recipe after the freeze",
                       "retrying or replacing a typed failure"],
        "caps": {"gpu_seconds": GPU_CAP_SECONDS, "wall_seconds_summed_over_processes": WALL_CAP_SECONDS,
                 "projection": smoke_evidence["projection"],
                 "scope_rule": (f"if the projected elapsed time on 4 GPUs exceeds {ELAPSED_LIMIT_HOURS} h, finish blocked "
                                "with the projection; never cut scope")},
        "disclosure": ("Known before the freeze: Gate B (unit 0.7444, prior 0.7203 on grid), #101 (J grid 0.706, "
                       "79 % (1, macro)), #114 (J - T-continuous +0.003 grid), #117, #105, #100 (E99 micro edge F1 "
                       "0.989, macro 0) and #113 (temporal head held-out macro F1 0.157 engine / 0.113 predicted / "
                       "0.015 rollout, current head 0.000). No intervention, retrain or label-free output on an "
                       "evaluation capture existed. Hardware: RTX 5090 instead of the #85 RTX 3090. Pre-freeze: "
                       "the analysis and rendering code (compute_tables, findings, figure, handoff) was exercised on "
                       "synthetic random rows over fit-split cells posing as the evaluation set (no model output, no "
                       "evaluation capture); a first smoke passed on a runner revision that differed only in the "
                       "findings layout and the guard-function split, and was rerun on the frozen runner."),
        "smoke_evidence": smoke_evidence,
        "inputs": input_bindings(output),
        "claim_boundary": ("one look at the sealed #109 evaluation split; grid ranking primary, other inventories, "
                           "rollout error and readouts descriptive; tie-free cost; DESCRIPTIVE intervals; NovPhy only; "
                           "the label-free arm is one discrete bottleneck, not every unsupervised predicate learner"),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-115 protocol")
    if not plan.get("frozen_before_training_and_scoring") or not plan["smoke_evidence"].get("ok"):
        raise ValueError("plan.json does not declare a passed-smoke freeze")
    for entry in plan["inputs"]:
        if not amendments.matches(entry["artifact"], entry["sha256"], sha256_of(ROOT / entry["artifact"])):
            raise ValueError(f"frozen input {entry['name']} changed after the freeze")
    for name, module in plan["modules"].items():
        if sha256_of(ROOT / module["path"]) != module["sha256_at_freeze"]:
            raise ValueError(f"module {name} changed after the freeze")
    return plan


def prepare(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        load_plan(output)
        log("existing frozen plan validated")
        return 0
    for name in ("records", "models", "data", "training", "calibration"):
        if (output / name).exists():
            raise ValueError(f"{name} exist before the freeze; refusing")
    evidence = read_json(output / "smoke.json")
    if not evidence.get("ok"):
        raise ValueError("smoke failed; STOP RULE: no freeze")
    if not evidence["projection"]["within_limit"]:
        raise ValueError(f"projection {evidence['projection']['elapsed_hours_on_4_gpus']:.1f} h exceeds the limit")
    plan = frozen_plan(output, utc_now(), evidence)
    (output / "plan.json").write_text(json_text(plan))
    load_plan(output)
    log("frozen plan published")
    return 0


# ---------------------------------------------------------------------------
# publication and validation
# ---------------------------------------------------------------------------

def publish(output):
    output = Path(output)
    plan = load_plan(output)
    compute = compute_tables(output, plan)
    for name, text in rendered(plan, compute, output).items():
        (output / name).write_text(text)
    log(f"published: { {d: t['token'] for d, t in compute['dispositions'].items()} }")
    return 0


def spot_checks(output):
    """Re-score, in this process, the first seed of: mi-free J, the micro oracle and flip20 F-5-micro, and
    the macro temporal F-15-macro on every grid cell (exact rows, feedback counts); and one readout."""
    load_plan(output)
    _, cells = gb.evaluation_set()
    anchors = gb.load_anchors(GATE_B)["anchors"]
    store = load_truth(output)
    seed = SEEDS[0]
    grid = {PRIMARY: cells[PRIMARY]}
    results = {}
    with harness.deterministic_scoring() as policy_state:
        for name, fb, arm in (("mi-free", "default", "J"), ("gateb", "oracle", "F-5-micro"),
                              ("gateb", "flip20", "F-5-micro"), ("ma-ref", "temporal", "F-15-macro")):
            kept = read_record(record_path(output, name, fb, arm, seed))
            if kept["typed_failure"]:
                results[f"{name}|{fb}|{arm}"] = None
                continue
            fresh = score_record(output, {**{i: {"mixed": {}} for i in INVENTORIES}, **grid}, anchors, store, name, fb,
                                 arm, seed, policy_state)
            same = all(fresh["inventories"][PRIMARY][m]["rows"] == kept["inventories"][PRIMARY][m]["rows"]
                       and fresh["inventories"][PRIMARY][m]["feedback"] == kept["inventories"][PRIMARY][m]["feedback"]
                       for m in grid[PRIMARY]["mixed"])
            results[f"{name}|{fb}|{arm}"] = bool(same)
        kept = read_record(readout_path(output, "mi-learned", seed))
        fresh = readout_record(output, cells, store, "mi-learned", seed, policy_state)
        results["readout|mi-learned"] = None if kept["typed_failure"] else fresh["levels"] == kept["levels"]
    return {"seed": seed, "checks": results, "pass": all(v is not False for v in results.values())}


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    compute = compute_tables(output, plan)
    for name, text in rendered(plan, compute, output).items():
        if (output / name).read_bytes() != text.encode():
            raise ValueError(f"published {name} differs from the recomputation")
    spot = spot_checks(output)
    if not spot["pass"]:
        raise ValueError(f"deterministic re-scoring does not replicate the retained records: {spot}")
    log(f"validation passed ({time.monotonic() - began:.0f}s): tables re-derived and byte-compared; re-scored "
        f"{spot['checks']} seed {spot['seed']}")
    return 0


def dry_run(output):
    _, cells = gb.evaluation_set()
    tasks = score_tasks()
    log(f"output {Path(output)}; predictors {len(PREDICTORS)} ({len(TRAINED)} trained x {len(SEEDS)} seeds); "
        f"score tasks {len(tasks)}")
    log(f"mixed cells { {i: len(cells[i]['mixed']) for i in INVENTORIES} }; branches {len(mixed_keys(cells))}")
    log(f"labelled levels: { {k: len(labelled_levels(k)) for k in LOW_K} }")
    return 0


MODES = {"dry-run": dry_run, "smoke": smoke, "prepare": prepare, "truth": truth, "heads": heads, "train": train,
         "calibrate": calibrate, "score": score, "publish": publish, "validate": validate}
SHARDED = ("heads", "train", "calibrate", "score")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker", type=int, default=0)
    args = parser.parse_args()
    mode = next(m for m in MODES if getattr(args, m.replace("-", "_")))
    if mode in SHARDED:
        return MODES[mode](OUTPUT, args.workers, args.worker)
    return MODES[mode](OUTPUT)


if __name__ == "__main__":
    raise SystemExit(main())
