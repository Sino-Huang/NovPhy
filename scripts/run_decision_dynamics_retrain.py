"""Issue-112 v3 retrain on the #104 fit split: the Gate-B checkpoints.

Binding runner module: scripts/run_decision_dynamics_retrain.py
Exact validation command: python -u -m scripts.run_decision_dynamics_retrain --validate

The #112 v3 recipe (arm LC: #77 loss + long-horizon supervision to 225 + decision-contrastive
loss, truncated back-propagation every 45 transitions, skip-on-nonfinite) was selected by
leave-one-lineage-out on the 12 exposed #77 N1 lineages, and #117 kept it (no v4). Gate B
scores it on the #104 sealed evaluation split, after a retrain on the new fit split
(#112 closing correction; #104 plan ``gate_b.retrain``). This runner

1. builds the training data of the #109 fit split captured by the #104 campaign: the E99
   carriers and #99 labels of every complete branch, the #99 windows and #112 long windows,
   each level's E99 anchor (batch-1 parse of its decision frame) and the mixed grid / angle
   ranking cells with the campaign's engine verdicts;
2. runs the frozen v3 ``train_group`` (a private copy of the v1 implementation with the v3
   spec, exactly as ``run_decision_dynamics_versions.implementation(3)``) for the Gate-B unit
   (LC, hybrid) and the pre-declared matched baseline C* (LC, continuous), fold 'all', the
   three #112 seeds, 9000 updates each, logging every update's #77 local loss and pre-clip
   gradient norm;
3. publishes the training accounting, the stability guards (no retirement; the #117
   end-of-training excursion rule declared before training) and a Gate-B handoff that
   supersedes the v3 handoff.

Nothing is selected or scored here: the recipe and the Gate-B unit are frozen by #112 v3, and
no evaluation-split, policy-split or #110 capture is read.

Modes: --dry-run --smoke --prepare --data --assemble --train --publish --validate. --data and
--train take --workers N --worker K (one process per GPU, CUDA_VISIBLE_DEVICES set by the
caller). Zero engine seconds.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import functools
import importlib.util
import multiprocessing
from pathlib import Path
import shutil
import time
import types

import numpy as np
import torch

from scripts import run_decision_dynamics_versions as versions
from scripts import run_decision_relevant_dynamics as v1
from scripts import run_lookahead_control as wlc
from scripts import run_request_interference as r117
from scripts import run_slot_encoder_fix as s99
from scripts import run_tau_ad_within_checkpoint as t96
from world_model.planning import scoring_harness as harness

ROOT = v1.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-112-v3-retrain-v1"
CAMPAIGN = ROOT / ".local-artifacts/issue-104-power-cohort-campaign-v1"
COHORT = ROOT / ".local-artifacts/issue-109-sealed-cohort-v1/plan.json"
POWER_PLAN = ROOT / ".local-artifacts/issue-104-power-cohort-v1/plan.json"
STREAM = "issue109-fit"

IDENTITY = "issue-112-v3-retrain-v1"
SCHEMA_PLAN = "issue_112_v3_retrain_plan_v1"
SCHEMA_SMOKE = "issue_112_v3_retrain_smoke_v1"
SCHEMA_SHOT = "issue_112_v3_retrain_shot_v1"
SCHEMA_DATA = "issue_112_v3_retrain_training_data_v1"
SCHEMA_PREDICTOR = "issue_112_v3_retrain_predictor_v1"
SCHEMA_GROUP = "issue_112_v3_retrain_training_group_v1"
SCHEMA_UPDATES = "issue_112_v3_retrain_updates_v1"
SCHEMA_COMPUTE = "issue_112_v3_retrain_compute_v1"
SCHEMA_REPORT = "issue_112_v3_retrain_report_v1"
SCHEMA_HANDOFF = "issue_112_gate_b_handoff_v3_retrain"
VALIDATION_COMMAND = "python -u -m scripts.run_decision_dynamics_retrain --validate"
RUNNER = "scripts/run_decision_dynamics_retrain.py"

DEVICE = "cuda"
VERSION = 3
ARM = "LC"
FAMILIES = ("hybrid", "continuous")
UNITS = {"hybrid": "F-15-macro", "continuous": "C-F-15"}  # the Gate-B unit and C* (v3 baseline_contrast)
SEEDS = v1.SEEDS
FIT_INVENTORIES = ("grid", "angle")
STEPS = versions.STEPS
SKIP_CAP = versions.VERSIONS[VERSION]["skip_cap"]
EXCURSION_RATIO = r117.EXCURSION_RATIO
EXCURSION_LAST = r117.EXCURSION_LAST
SPIKE_NORM = r117.SPIKE_NORM
SPIKE_WINDOW = r117.SPIKE_WINDOW
LABEL_WORKERS = 10
GPU_CAP_SECONDS = 16 * 3600.0  # summed over processes; a data process holds its GPU for its whole wall
WALL_CAP_SECONDS = 30 * 3600.0  # summed over processes
STOP_TOKEN = v1.STOP_TOKEN
SMOKE_STEPS = 12

read_json = v1.read_json
write_json = v1.write_json
json_text = v1.json_text
sha256_of = v1.sha256_of
atomic_torch = v1.atomic_torch


def log(message):
    print(f"[issue-112-retrain] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# the frozen v3 implementation, bound to this output
# ---------------------------------------------------------------------------

def logging_ensemble(base):
    """The v3 ensemble class, recording every update's per-member pre-clip gradient norm and skip
    flag, and (through ``logged_local_loss``) its #77 local loss; the record is part of the
    resumable state. Values only: no computation of the update changes."""

    class LoggingEnsemble(base):
        def __init__(self, models, **kwargs):
            super().__init__(models, **kwargs)
            self.log = {"local": [], "grad_norm": [], "skipped": []}

        def update(self, loss):
            report = super().update(loss)
            self.log["grad_norm"].append(report["grad_norm"].float())
            self.log["skipped"].append(report["skipped"])
            return report

        def log_tensors(self):
            return {k: torch.stack(v).cpu() if v else torch.empty(0, self.size) for k, v in self.log.items()}

        def state(self):
            return {**super().state(), "log": self.log_tensors()}

        def load_state(self, state):
            super().load_state(state)
            self.log = {k: list(v.to(self.device).unbind(0)) for k, v in state["log"].items()}

    return LoggingEnsemble


def logged_local_loss(ensemble, batch, pair):
    value = versions.dd.local_loss(ensemble, batch, pair)
    ensemble.log["local"].append(value.detach())
    return value


@functools.lru_cache(maxsize=None)
def implementation():
    """A private copy of the v1 runner carrying the frozen v3 spec (arms, loss constants, stable
    ensemble and truncated recursion taken from ``versions.implementation(3)``), with this
    runner's output, identity and schemas and the logging ensemble."""
    v3 = versions.implementation(VERSION)
    loader = importlib.util.spec_from_file_location("issue112_retrain_implementation", ROOT / v1.RUNNER)
    impl = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(impl)
    impl.OUTPUT = OUTPUT
    impl.IDENTITY = IDENTITY
    impl.SCHEMA_PREDICTOR = SCHEMA_PREDICTOR
    impl.SCHEMA_GROUP = SCHEMA_GROUP
    impl.SCHEMA_DATA = SCHEMA_DATA
    impl.RUNNER = RUNNER
    impl.MODEL_MODULE = versions.STABLE_MODULE
    impl.ARMS = (ARM,)
    impl.ARM_DEFINITIONS = {ARM: v3.ARM_DEFINITIONS[ARM]}
    impl.LONG = v3.LONG
    impl.RANK = v3.RANK
    impl.dd = types.SimpleNamespace(Ensemble=logging_ensemble(v3.dd.Ensemble), local_loss=logged_local_loss,
                                    horizon_at=v3.dd.horizon_at, ramp_at=v3.dd.ramp_at,
                                    long_and_rank_loss=v3.dd.long_and_rank_loss)
    return impl


def v3_plan():
    return read_json(versions.implementation(VERSION).OUTPUT / "plan.json")


# ---------------------------------------------------------------------------
# membership (the #109 fit split as captured by the #104 campaign)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def cohort():
    return read_json(COHORT)


def families():
    return read_json(POWER_PLAN)["cohorts"]["issue109"]["families"]


def candidates():
    return cohort()["candidate_inventories"]["sets"]["training"]["candidates"]


def fit_levels():
    return [level for family in families() for level in cohort()["splits"]["fit"][family]]


@functools.lru_cache(maxsize=None)
def stream_rows():
    """The campaign report rows of the fit stream only (other streams are dropped at load)."""
    return {row["identity"]: row for row in read_json(CAMPAIGN / "branches.json") if row["stream"] == STREAM}


def fit_branches():
    """Frozen order: families, roster order, candidate suffix order. Outcome-free."""
    rows = stream_rows()
    out = []
    for level in fit_levels():
        for candidate in candidates():
            key = f"{level['identity']}-{candidate['suffix']}"
            out.append({"key": key, "member": level["identity"], "family": level["generator_family"],
                        "candidate": candidate, "status": rows[key]["status"],
                        "failure_class": rows[key]["failure_class"],
                        "attempt": str(CAMPAIGN / STREAM / "attempts" / key)})
    if len(out) != len(rows):
        raise ValueError("the fit-stream rows differ from the #109 fit roster x training candidates")
    return out


def complete_branches():
    return [b for b in fit_branches() if b["status"] == "complete"]


def fit_verdicts():
    """Engine pig-removal verdict of every complete fit branch (the #104 report, #85 channel)."""
    return {key: bool(row["pig_removed"]) for key, row in stream_rows().items() if row["status"] == "complete"}


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def shot_path(output, key):
    return Path(output) / "shots" / f"{key}.pt"


def anchors_path(output):
    return Path(output) / "data" / "anchors-fit.pt"


def data_path(output):
    return Path(output) / "data" / "training-data.pt"


def updates_path(output, family, index):
    return Path(output) / "updates" / f"{ARM}--{family}--group{index}.pt"


def ledger_path(output, phase):
    return Path(output) / "ledger" / f"{phase}.json"


def verdicts_path(output):
    return Path(output) / "fit-verdicts.json"


# ---------------------------------------------------------------------------
# shot data (outcome-free): #99 labels on CPU, E99 carriers on GPU
# ---------------------------------------------------------------------------

def shot_labels(attempt):
    """Process-pool worker: the #99 shot labels (s99.shot_labels) the windows use, and the action."""
    torch.set_num_threads(1)
    _, _, frames, _ = s99.shot_frames(attempt)
    segment = read_json(Path(attempt) / "shot-1" / "segment.json")
    end = min(len(frames) - 1, s99.MAX_POSITION)
    labels = s99.shot_labels(segment, frames, list(range(end + 1)), s99.vocabulary())
    return {"end": end, "frame_count": len(frames), "action": s99.action_of(segment),
            "relations": labels["relations"].bool(), "relation_mask": labels["relation_mask"].bool(),
            "macros": labels["macros"].bool(), "macro_mask": labels["macro_mask"].bool()}


def shot_entry(branch, labels, adapter):
    carriers = s99.carriers_for(adapter, branch["attempt"], list(range(labels["end"] + 1)))
    return {"schema": SCHEMA_SHOT, "plan_identity": IDENTITY, "key": branch["key"], "member": branch["member"],
            **labels, "carriers": carriers}


def anchor_carrier(adapter, attempt):
    """E99 batch-1 parse (wlc.encode_carrier) of the decision frame: shot frame 0 at the decision step."""
    _, observation_root, frames, _ = s99.shot_frames(attempt)
    frame = frames[0]
    anchor = {"observation_identity": frame["agent_observation"]["identity"], "fixed_step": frame["fixed_step"],
              "fixed_time_seconds": frame["fixed_time_seconds"],
              "frame_path": str(observation_root / frame["agent_observation"]["relative_path"])}
    return wlc.encode_carrier(adapter, anchor, DEVICE).float().cpu()


def anchor_branches(branches):
    """Per level, its first complete candidate in suffix order (its decision frame is the anchor)."""
    out = {}
    for branch in branches:
        out.setdefault(branch["member"], branch)
    return out


# ---------------------------------------------------------------------------
# training data: the #99 windows as position indices into the carriers
# ---------------------------------------------------------------------------

def build_payload(shots, cells):
    """shots (frozen order): key, member, end, action, carriers [end+1, D] and the label
    tensors per position; cells: member, inventory, ordinals, z0, action [pad, 5], verdict [pad].

    Window w of shot s at start t holds positions min(t + i, t + length), i = 0..60 (s99.windows),
    stored as indices into the concatenated carriers and labels, so a gathered batch equals the
    materialized #112 window tensors."""
    window_positions, window_shot, window_length, long_windows, offsets, lasts = [], [], [], [], [], []
    steps = torch.arange(s99.WINDOW_LENGTH + 1)
    total = 0
    for index, shot in enumerate(shots):
        end = int(shot["end"])
        if len(shot["carriers"]) != end + 1:
            raise ValueError(f"carriers of {shot['key']} do not cover positions 0..end")
        for start in s99.WINDOW_STARTS:
            if start >= end:
                continue
            length = min(s99.WINDOW_LENGTH, end - start)
            window_positions.append(total + start + torch.minimum(steps, torch.tensor(length)))
            window_shot.append(index)
            window_length.append(length)
        long_windows.extend((index, start) for start in v1.LONG["starts"] if start < end)
        offsets.append(total)
        lasts.append(end)
        total += end + 1

    def cat(key):
        return torch.cat([shot[key] for shot in shots])

    def stacked(key, empty):
        return torch.stack([c[key] for c in cells]) if cells else empty

    return {"schema": SCHEMA_DATA, "plan_identity": IDENTITY,
            "carriers": cat("carriers"), "relations": cat("relations"), "relation_mask": cat("relation_mask"),
            "macros": cat("macros"), "macro_mask": cat("macro_mask"),
            "window_positions": torch.stack(window_positions), "window_shot": torch.tensor(window_shot),
            "window_length": torch.tensor(window_length), "offsets": torch.tensor(offsets),
            "lasts": torch.tensor(lasts), "shot_actions": torch.stack([shot["action"] for shot in shots]),
            "long_windows": torch.tensor(long_windows), "shot_keys": [shot["key"] for shot in shots],
            "shot_members": [shot["member"] for shot in shots],
            "cell_members": [c["member"] for c in cells], "cell_inventories": [c["inventory"] for c in cells],
            "cell_ordinals": [c["ordinals"] for c in cells],
            "cell_z0": stacked("z0", torch.empty(0, 236)),
            "cell_action": stacked("action", torch.empty(0, v1.RANK["candidate_pad"], 5)),
            "cell_verdict": stacked("verdict", torch.empty(0, v1.RANK["candidate_pad"], dtype=torch.long))}


class IndexedTrainData(v1.TrainData):
    """v1.TrainData over an indexed payload: windows are gathered from the per-position carriers
    and labels; pools, long windows and ranking cells are the v1 code."""

    def __init__(self, payload, family):
        self.hybrid = family == "hybrid"
        self.carriers = payload["carriers"].to(DEVICE)
        self.window_positions = payload["window_positions"].to(DEVICE)
        self.window_length = payload["window_length"].to(DEVICE)
        self.shot_members = payload["shot_members"]
        self.window_members = [self.shot_members[i] for i in payload["window_shot"].tolist()]
        self.offsets = payload["offsets"].to(DEVICE)
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
            self.symbols = {"relations": payload["relations"].to(DEVICE),
                            "relations_mask": payload["relation_mask"].to(DEVICE),
                            "macros": payload["macros"].to(DEVICE), "macros_mask": payload["macro_mask"].to(DEVICE)}

    def windows(self, pools, generators, count):
        index = self._draw(pools, "window", generators, count)
        positions = self.window_positions[index]
        out = {"z": self.carriers[positions], "action": self.window_action[index], "length": self.window_length[index]}
        if self.hybrid:
            out.update({k: v[positions] for k, v in self.symbols.items()})
        return out


def ranking_cells(branches, verdicts, anchors):
    """Every fit level x inventory (grid, angle) whose verdict-bearing candidates hold both a pig
    removal and a failure (t96.mixed); candidates in inventory-ordinal order."""
    by_level = {}
    for branch in branches:
        by_level.setdefault(branch["member"], []).append(branch)
    cells = []
    for level in fit_levels():
        for inventory in FIT_INVENTORIES:
            items = sorted((entry["ordinal"], branch) for branch in by_level.get(level["identity"], [])
                           for entry in branch["candidate"]["inventory_entries"] if entry["inventory"] == inventory)
            values = [verdicts[branch["key"]] for _, branch in items]
            if set(values) != {True, False}:
                continue
            if len(items) > v1.RANK["candidate_pad"]:
                raise ValueError("a ranking cell exceeds the candidate pad")
            action = torch.zeros(v1.RANK["candidate_pad"], 5)
            verdict = torch.full((v1.RANK["candidate_pad"],), -1, dtype=torch.long)
            action[:len(items)] = torch.cat([wlc.action_tensor(b["candidate"]["action"], "cpu") for _, b in items])
            verdict[:len(items)] = torch.tensor([int(v) for v in values])
            cells.append({"member": level["identity"], "inventory": inventory, "ordinals": [o for o, _ in items],
                          "z0": anchors[level["identity"]], "action": action, "verdict": verdict})
    return cells


def load_training_data(output):
    payload = torch.load(data_path(output), map_location="cpu", weights_only=True, mmap=True)
    if payload.get("schema") != SCHEMA_DATA or payload.get("plan_identity") != IDENTITY:
        raise ValueError("training data binding differs")
    return payload


# ---------------------------------------------------------------------------
# training groups and the end-of-training excursion rule
# ---------------------------------------------------------------------------

def groups():
    """One member per group: (family, index, seed); index is the seed position."""
    return [(family, index, seed) for family in FAMILIES for index, seed in enumerate(SEEDS)]


def step_requests(family, steps=STEPS):
    """The request each update trains (v1.step_pair: PAIRS[step % 9])."""
    out = []
    for step in range(steps):
        pair = v1.step_pair(family, step)
        out.append(t96.FIXED_ARMS[step % 9] if family == "hybrid" else f"C-F-{pair.delta}")
    return out


def excursion(local, norms, requests):
    """#117 end-state rule (declared here before training): a member ends in an excursion when,
    for some request, the median #77 local loss of its last EXCURSION_LAST updates of that
    request exceeds EXCURSION_RATIO x the median over that request's updates in [0.6 n, 0.95 n).
    local, norms: [n] per update of one member; requests: [n] request names."""
    total = len(requests)
    ratios = {}
    for request in sorted(set(requests)):
        rows = torch.tensor([i for i, r in enumerate(requests) if r == request])
        reference = local[rows[(rows >= int(0.6 * total)) & (rows < int(0.95 * total))]]
        reference = reference[torch.isfinite(reference)]
        ratios[request] = float(local[rows[-EXCURSION_LAST:]].median() / reference.median())
    window = torch.arange(max(0, total - SPIKE_WINDOW), total)
    spikes = window[norms[window] > SPIKE_NORM]
    first = int(spikes[0]) if len(spikes) else None
    return {"ratios": ratios, "excursion": max(ratios.values()) >= EXCURSION_RATIO,
            "requests_elevated": sorted(r for r, v in ratios.items() if v >= EXCURSION_RATIO),
            "first_spike_update": first, "first_spike_request": None if first is None else requests[first],
            "first_spike_norm": None if first is None else float(norms[first]),
            "spike_requests": sorted({requests[k] for k in spikes.tolist()})}


# ---------------------------------------------------------------------------
# ledger (one file per phase and worker; several processes run at once)
# ---------------------------------------------------------------------------

def mark(output, phase, wall, gpu, **extra):
    path = ledger_path(output, phase)
    block = read_json(path) if path.is_file() else {"wall_seconds": 0.0, "gpu_seconds": 0.0}
    write_json(path, {**block, "wall_seconds": block["wall_seconds"] + wall, "gpu_seconds": block["gpu_seconds"] + gpu,
                      "last_at": utc_now(), **extra})
    total = [read_json(p) for p in sorted((Path(output) / "ledger").glob("*.json"))]
    if (sum(b["gpu_seconds"] for b in total) > GPU_CAP_SECONDS
            or sum(b["wall_seconds"] for b in total) > WALL_CAP_SECONDS):
        raise ValueError(f"{STOP_TOKEN}: compute cap exceeded")


# ---------------------------------------------------------------------------
# smoke (pre-freeze; nothing kept, no statistic, no verdict joined to a model output)
# ---------------------------------------------------------------------------

def v3_shots(payload):
    """The per-position shots of the v3 training data, recovered from its materialized windows
    (every position 0..end lies in some window; the gather check below closes the loop)."""
    shots = []
    starts = {}
    for w, s in enumerate(payload["window_shot"].tolist()):
        starts.setdefault(s, []).append(w)
    for s, key in enumerate(payload["shot_keys"]):
        end, offset = int(payload["lasts"][s]), int(payload["offsets"][s])
        labels = {"relations": torch.zeros(end + 1, 18, 18, 2, dtype=torch.bool),
                  "relation_mask": torch.zeros(end + 1, 18, 18, 2, dtype=torch.bool),
                  "macros": torch.zeros(end + 1, 2, dtype=torch.bool), "macro_mask": torch.zeros(end + 1, 2, dtype=torch.bool)}
        for w, start in zip(starts[s], [t for t in s99.WINDOW_STARTS if t < end], strict=True):
            length = int(payload["length"][w])
            index = [min(start + i, start + length) for i in range(s99.WINDOW_LENGTH + 1)]
            for ours, theirs in (("relations", "relations"), ("relation_mask", "relations_mask"),
                                 ("macros", "macros"), ("macro_mask", "macros_mask")):
                labels[ours][index] = payload[theirs][w]
        shots.append({"key": key, "member": payload["shot_members"][s], "end": end,
                      "action": payload["shot_actions"][s], "carriers": payload["carriers"][offset:offset + end + 1],
                      **labels})
    return shots


def v3_cells(payload):
    return [{"member": m, "inventory": i, "ordinals": o, "z0": z, "action": a, "verdict": v}
            for m, i, o, z, a, v in zip(payload["cell_members"], payload["cell_inventories"], payload["cell_ordinals"],
                                        payload["cell_z0"], payload["cell_action"], payload["cell_verdict"])]


def gathered_windows_equal(indexed, materialized):
    positions = indexed["window_positions"]
    pairs = (("carriers", "z"), ("relations", "relations"), ("relation_mask", "relations_mask"),
             ("macros", "macros"), ("macro_mask", "macros_mask"))
    same = all(torch.equal(indexed[ours][positions], materialized[theirs]) for ours, theirs in pairs)
    same &= torch.equal(indexed["shot_actions"][indexed["window_shot"]], materialized["action"])
    same &= torch.equal(indexed["window_length"], materialized["length"])
    same &= all(torch.equal(indexed[k], materialized[k]) for k in ("window_shot", "offsets", "lasts", "long_windows",
                                                                   "cell_z0", "cell_action", "cell_verdict"))
    return bool(same)


def batches_equal(first, second, family, fold):
    """Same generators, same draws: the window, long and ranking batches of both data objects."""
    pools = [first.pools(fold)], [second.pools(fold)]
    generators = [[torch.Generator().manual_seed(s)] for s in (7, 7)]
    for step in range(60):
        pair = v1.step_pair(family, step)
        a, b = first.windows(pools[0], generators[0], 64), second.windows(pools[1], generators[1], 64)
        if a.keys() != b.keys() or not all(torch.equal(a[k], b[k]) for k in a):
            return False
        a, b = (first.long_batch(pools[0], generators[0], 8, v1.ENDPOINT, pair.delta),
                second.long_batch(pools[1], generators[1], 8, v1.ENDPOINT, pair.delta))
        if not all(torch.equal(a[k], b[k]) for k in a):
            return False
        a, b = first.rank_batch(pools[0], generators[0], 2), second.rank_batch(pools[1], generators[1], 2)
        if not all(torch.equal(a[k], b[k]) for k in a):
            return False
    return True


def max_parameter_delta(first, second):
    return max(float((first.params[k] - second.params[k]).abs().max()) for k in first.params)


def full_size_payload(sizes, family):
    """A payload of the full training-data shapes (random values) for the memory and time projection."""
    positions = sum(e + 1 for e in sizes)
    generator = torch.Generator().manual_seed(0)
    windows = [(i, t) for i, e in enumerate(sizes) for t in s99.WINDOW_STARTS if t < e]
    offsets = torch.tensor(np.concatenate([[0], np.cumsum([e + 1 for e in sizes])[:-1]]))
    starts = torch.tensor([t for _, t in windows])
    shot_of = torch.tensor([i for i, _ in windows])
    lengths = torch.minimum(torch.tensor(sizes)[shot_of] - starts, torch.tensor(s99.WINDOW_LENGTH))
    steps = torch.arange(s99.WINDOW_LENGTH + 1)
    hybrid = family == "hybrid"
    cells = 700
    return {"carriers": torch.rand(positions, 236, generator=generator) * 0.1,
            "relations": torch.zeros(positions if hybrid else 1, 18, 18, 2, dtype=torch.bool),
            "relation_mask": torch.ones(positions if hybrid else 1, 18, 18, 2, dtype=torch.bool),
            "macros": torch.zeros(positions if hybrid else 1, 2, dtype=torch.bool),
            "macro_mask": torch.ones(positions if hybrid else 1, 2, dtype=torch.bool),
            "window_positions": offsets[shot_of, None] + starts[:, None] + torch.minimum(steps, lengths[:, None]),
            "window_shot": shot_of, "window_length": lengths, "offsets": offsets, "lasts": torch.tensor(sizes),
            "shot_actions": torch.rand(len(sizes), 5, generator=generator),
            "long_windows": torch.tensor([(i, t) for i, e in enumerate(sizes) for t in v1.LONG["starts"] if t < e]),
            "shot_members": [f"level-{i // 29}" for i in range(len(sizes))],
            "cell_members": [f"level-{i}" for i in range(cells)],
            "cell_z0": torch.rand(cells, 236, generator=generator) * 0.1,
            "cell_action": torch.rand(cells, v1.RANK["candidate_pad"], 5, generator=generator),
            "cell_verdict": torch.tensor([[1, 0] * 8 + [-1] * 4] * cells)}


def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the smoke is a pre-freeze control")
    scratch = output / "smoke-scratch"
    shutil.rmtree(scratch, ignore_errors=True)
    impl, v3 = implementation(), versions.implementation(VERSION)
    checks, timings, memory = [], {}, {}
    torch.backends.cudnn.benchmark = True

    # 1. the indexed data equals the v3 training data; training on it equals v3's train_group
    reference = torch.load(v1.data_path(v3.OUTPUT), map_location="cpu", weights_only=True)
    indexed = build_payload(v3_shots(reference), v3_cells(reference))
    checks.append({"control": "indexed_windows_equal_v3_training_data", "windows": len(indexed["window_positions"]),
                   "positions": len(indexed["carriers"]), "ok": gathered_windows_equal(indexed, reference)})
    fold = reference["shot_members"][0]
    for family in FAMILIES:
        ours, theirs = IndexedTrainData(indexed, family), v3.TrainData(reference, family)
        checks.append({"control": f"{family}_batches_equal_v3", "folds": ["all", fold],
                       "ok": batches_equal(ours, theirs, family, "all") and batches_equal(ours, theirs, family, fold)})
        members = [("all", SEEDS[0])]
        _, a = impl.train_group(scratch, ARM, family, 0, members, ours, steps=SMOKE_STEPS, save=False, smoke_full=True)
        _, b = v3.train_group(scratch, ARM, family, 0, members, theirs, steps=SMOKE_STEPS, save=False, smoke_full=True)
        delta = max_parameter_delta(a, b)
        logs = a.log_tensors()
        checks.append({"control": f"{family}_train_group_equals_v3", "updates": SMOKE_STEPS, "max_parameter_delta": delta,
                       "logged_updates": {k: len(v) for k, v in logs.items()},
                       "ok": delta == 0.0 and all(len(v) == SMOKE_STEPS for v in logs.values())})
        del ours, theirs, a, b
        torch.cuda.empty_cache()
    del reference, indexed

    # 2. new-split shots: labels, carriers, anchors (two levels, three candidates each)
    branches = complete_branches()
    chosen = [b for b in branches if b["member"] in {branches[0]["member"], branches[-1]["member"]}]
    chosen = [b for b in chosen if b["candidate"]["suffix"] in ("c00", "c01", "c02")]
    adapter = s99.load_adapter(s99.OUTPUT, "E")
    began = time.monotonic()
    labels = [shot_labels(b["attempt"]) for b in chosen]
    timings["label_seconds_per_shot"] = (time.monotonic() - began) / len(chosen)
    began = time.monotonic()
    entries = [shot_entry(b, l, adapter) for b, l in zip(chosen, labels)]
    timings["carrier_seconds_per_shot"] = (time.monotonic() - began) / len(chosen)
    checks.append({"control": "new_shots_cover_positions", "shots": len(entries),
                   "ends": [int(e["end"]) for e in entries],
                   "ok": all(len(e["carriers"]) == e["end"] + 1 == len(e["relations"]) for e in entries)})
    checks.append({"control": "executed_action_equals_inventory_action",
                   "ok": all(torch.equal(e["action"], wlc.action_tensor(b["candidate"]["action"], "cpu")[0])
                             for b, e in zip(chosen, entries))})
    with harness.deterministic_scoring():
        anchors = {b["member"]: anchor_carrier(adapter, b["attempt"]) for b in anchor_branches(chosen).values()}
        again = anchor_carrier(adapter, chosen[0]["attempt"])
    first = anchor_branches(chosen)
    replication = [harness.carrier_replication(anchors[m], entries[chosen.index(b)]["carriers"][0])
                   for m, b in first.items()]
    checks.append({"control": "anchor_parse_deterministic", "ok": bool(torch.equal(again, anchors[chosen[0]["member"]]))})
    checks.append({"control": "anchor_equals_batched_frame0_carrier", "replication": replication,
                   "ok": all(r["pass"] for r in replication)})
    payload = build_payload(entries, [])
    checks.append({"control": "new_windows_built", "windows": len(payload["window_positions"]),
                   "long_windows": len(payload["long_windows"]), "ok": len(payload["window_positions"]) > 0})

    # 3. isolation: every training branch is a complete #109 fit-stream capture of a fit-roster level
    roster = {level["identity"] for level in fit_levels()}
    checks.append({"control": "fit_stream_isolation", "branches": len(branches),
                   "typed_failures_excluded": len(fit_branches()) - len(branches),
                   "ok": all(b["member"] in roster and Path(b["attempt"]).parent.parent.name == STREAM
                             for b in branches)})

    # 4. full-size memory and time per family (one member per group; the most expensive update)
    sizes = [min(read_json(CAMPAIGN / STREAM / "results" / f"{b['key']}.json")["segments"][0]["frame_count"] - 1,
                 s99.MAX_POSITION) for b in branches]
    projection = {}
    for family in FAMILIES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        data = IndexedTrainData(full_size_payload(sizes, family), family)
        report, _ = impl.train_group(scratch, ARM, family, 0, [("all", SEEDS[0])], data, steps=9 + SMOKE_STEPS,
                                     save=False, smoke_full=True, timing_after=9)
        memory[family] = report["peak_memory_gb"]
        timings[f"{family}_seconds_per_update"] = report["timed_seconds_per_update"]
        projection[family] = {"peak_gb": report["peak_memory_gb"],
                              "training_seconds_upper_per_member": STEPS * report["timed_seconds_per_update"]}
        del data
    data_seconds = len(branches) * timings["label_seconds_per_shot"] / (4 * LABEL_WORKERS)
    training_seconds = sum(p["training_seconds_upper_per_member"] for p in projection.values()) * len(SEEDS)
    windows = sum(sum(1 for t in s99.WINDOW_STARTS if t < e) for e in sizes)
    ok = all(c["ok"] for c in checks)
    evidence = {"schema": SCHEMA_SMOKE, "identity": IDENTITY, "run_at": utc_now(), "checks": checks,
                "timings": timings, "memory_gb": memory,
                "projection": {"branches": len(branches), "positions": sum(e + 1 for e in sizes), "windows": windows,
                               "data_wall_seconds_4_processes": data_seconds,
                               "training_gpu_seconds_upper": training_seconds, **projection},
                "reading": "infrastructure assertions only; nothing kept; no statistic; no verdict read", "ok": ok}
    write_json(output / "smoke.json", evidence)
    shutil.rmtree(scratch, ignore_errors=True)
    log(f"smoke {'PASSED' if ok else 'FAILED'}; projected data {data_seconds / 3600:.2f} h wall, training "
        f"{training_seconds / 3600:.2f} GPU h (upper)")
    for check in checks:
        log(f"  {check['control']}: {check['ok']}")
    if not ok:
        raise ValueError("smoke control differs; STOP RULE: abort before the freeze")
    return 0


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def input_bindings(output):
    v3 = versions.implementation(VERSION).OUTPUT
    names = {
        "issue_112_v3_plan": v3 / "plan.json", "issue_112_v3_compute": v3 / "compute.json",
        "issue_112_v3_handoff": v3 / "gate_b_handoff.json",
        "issue_112_v1_runner": ROOT / v1.RUNNER, "issue_112_versions_runner": ROOT / versions.RUNNER,
        "issue_112_v1_module": ROOT / v1.MODEL_MODULE, "issue_112_v3_module": ROOT / versions.STABLE_MODULE,
        "issue_117_compute": r117.OUTPUT / "compute.json",
        "issue_77_dynamics_plan": t96.DYNAMICS / "plan.json",
        "issue_99_encoder": s99.OUTPUT / "encoder" / "encoder.pt",
        "issue_109_cohort_plan": COHORT, "issue_104_plan": POWER_PLAN,
        "issue_104_campaign_branches": CAMPAIGN / "branches.json",
        "issue_104_campaign_summary": CAMPAIGN / "summary.json",
        "scoring_harness": ROOT / "world_model/planning/scoring_harness.py",
        "smoke": Path(output) / "smoke.json",
    }
    return [{"name": name, "artifact": str(Path(path).relative_to(ROOT)), "sha256": sha256_of(path)}
            for name, path in names.items()]


def frozen_plan(output, frozen_at, smoke_evidence):
    v3 = v3_plan()
    branches = fit_branches()
    complete = [b for b in branches if b["status"] == "complete"]
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "issue": 112, "parent_issue": 108,
        "role": ("retrain of the frozen #112 v3 recipe on the #104 fit split (the Gate-B checkpoints); no selection, "
                 "no scoring, no statistic"),
        "frozen_at": frozen_at, "frozen_before_data_and_training": True, "issue_64_authorized": False,
        "validation_command": VALIDATION_COMMAND, "runner": RUNNER, "runner_sha256_at_freeze": sha256_of(ROOT / RUNNER),
        "engine_seconds": 0,
        "question": ("train the frozen #112 v3 Gate-B unit (LC : F-15-macro, hybrid, E99) and its pre-declared matched "
                     "baseline C* (LC : C-F-15, continuous) on the #109 fit split captured by the #104 campaign, and "
                     "hand the checkpoints to Gate B"),
        "recipe": {"source": "issue-112-decision-dynamics-v3 plan.json (training block, version_spec)",
                   "arm": ARM, "definition": versions.implementation(VERSION).ARM_DEFINITIONS[ARM],
                   "training": v3["training"], "version_spec": v3["version_spec"],
                   "implementation": ("scripts.run_decision_dynamics_retrain.implementation(): a private copy of the "
                                      "v1 runner with the v3 arms, loss constants, SkippingEnsemble (patience 10) and "
                                      "truncated recursion (45) taken from versions.implementation(3); train_group "
                                      "is the v1 function, unchanged"),
                   "unchanged": ("losses, weights, curriculum, ramp, truncation, skip rule, optimizer, batch sizes, "
                                 "9000 updates, initialization, seeds, encoder")},
        "units": {"gate_b": {"arm": ARM, "family": "hybrid", "request": UNITS["hybrid"], "encoder": "E99"},
                  "c_star": {"arm": ARM, "family": "continuous", "request": UNITS["continuous"],
                             "source": "v3 compute.json baseline_contrast.c_star (the largest-S continuous unit)"}},
        "data": {
            "split": ("#109 fit split (issue-109-sealed-cohort-v1, splits.fit: 240 levels per family, families "
                      f"{families()}), captured by the #104 campaign stream {STREAM} (capture_pipeline_v2, #113 window "
                      "rule, Mesa 26.1.2 renderer)"),
            "levels": len(fit_levels()), "branches": len(branches), "complete_branches": len(complete),
            "typed_failures_excluded": [{"key": b["key"], "failure_class": b["failure_class"]}
                                        for b in branches if b["status"] != "complete"],
            "shot_order": "families, roster order, candidate suffix order (c00..c28); complete branches only",
            "carriers": ("E99 (the frozen #99 spatial slot parser, s99.load_adapter(s99.OUTPUT, 'E')) parse of every "
                         "observation 0..min(last, 285), s99.carriers_for (prior position - 1)"),
            "labels": "s99.shot_labels (the #99 micro relations and macro labels per observed position)",
            "windows": ("s99.windows: starts 0..225 step 15 below the shot's last position, 60 transitions, terminal "
                        "padding; stored as position indices into the carriers and labels (IndexedTrainData), equal "
                        "to the materialized v3 window tensors (smoke)"),
            "long_windows": "v3 LONG starts 0, 15, 30, 45, 60 below the shot's last position",
            "anchors": ("per level: E99 batch-1 parse (wlc.encode_carrier) of the decision frame, shot frame 0 of its "
                        "first complete candidate in suffix order, under deterministic_scoring()"),
            "ranking_cells": ("every fit level x inventory in (grid 16, angle 13) whose verdict-bearing candidates "
                              "hold both a pig removal and a failure (t96.mixed); candidates in inventory-ordinal "
                              "order, padded to 20; z0 the level anchor; actions the inventory actions"),
            "verdicts": (f"engine pig-removal verdict of every complete {STREAM} branch from the #104 campaign report "
                         "(branches.json, #85 channel); rows of other streams are dropped at load"),
            "excluded": {
                "issue109-evaluation, issue110-evaluation": "final evaluation; never fitted",
                "issue109-policy": "controller (policy) training split (#109 split_roles)",
                "issue110-fit": ("the #104 fit_merge pools the #110 normal fit sides with #109 fit, but the Gate-B "
                                 "recipe fixes the E99 encoder, whose 18-slot contract v1 covers the #110 normal "
                                 "cells on only 25 of 40 templates (#110 doc); those levels need the contract-v2 "
                                 "encoder of the #107 track"),
                "issue-77 N1 lineages": "the old fit split (exposed by #112 selection); replaced by the new split",
            },
        },
        "training": {
            "groups": [{"family": f, "index": i, "seed": s, "fold": "all"} for f, i, s in groups()],
            "folds": "fold 'all' only (every fit level); no leave-one-out folds: nothing is selected",
            "group_size": 1,
            "execution": ("one member per train_group call, torch.compile'd vmapped transition; groups spread over "
                          "the 4 GPUs (one process per GPU); training process only"),
            "logging": ("every update: the #77 local loss, the pre-clip gradient norm and the skip flag per member "
                        "(values only)"),
        },
        "guards": {
            "G1_isolation": ("every training shot, long window and ranking cell belongs to a complete branch of a "
                             f"#109 fit-roster level in stream {STREAM}; every ranking-cell action equals its shot's "
                             "executed action"),
            "G2_stability": f"every member reached {STEPS} updates without retirement and skipped at most {SKIP_CAP}",
            "G3_end_state": (f"#117 end-state rule, declared before training: a member ends in an excursion when, for "
                             f"some request, the median #77 local loss of its last {EXCURSION_LAST} updates of that "
                             f"request exceeds {EXCURSION_RATIO:g} x the median of that request's updates in "
                             f"[0.6 n, 0.95 n); the first spike is the first of its last {SPIKE_WINDOW} updates "
                             f"with a pre-clip norm above {SPIKE_NORM:g}"),
        },
        "decision_rule": {
            "usable_checkpoint": ("a member is usable at Gate B iff it is complete (G2) and does not end in an "
                                  "excursion (G3); an unusable member is a typed terminal failure "
                                  "(retired / final_checkpoint_excursion), never retried or replaced"),
            "handoff": ("Gate-B handoff ready (supported) iff G1 holds and the Gate-B unit and C* each keep at least "
                        "one usable seed; otherwise readiness_or_precision_insufficient"),
        },
        "prohibited": ["reading any evaluation-split, policy-split or #110 capture", "scoring any model",
                       "changing the recipe, the units, the data rule or a guard after the freeze",
                       "retrying or replacing a failed member"],
        "caps": {"gpu_seconds": GPU_CAP_SECONDS, "wall_seconds": WALL_CAP_SECONDS, "stop": STOP_TOKEN},
        "disclosure": ("Known before the freeze: every #112 v1/v2/v3 and #117 result, the #104 campaign accounting and "
                       "its mixed-verdict yield per level type (fit grid 365 / 480, angle 314 / 480). The recipe keeps "
                       "the #77 update budget of 9000 updates although the new split holds about 19x the v3 windows "
                       "and 25x its ranking cells; changing it would be a new version. Hardware: 4 x RTX 5090 "
                       "(the #85 single-RTX-3090 gate is carried as one GPU per member). The SERVER_SETUP path "
                       "rewrite changed bytes of #77/#96 plans; inputs here bind the current bytes. An external vLLM "
                       "job used the GPUs while the fit stream was captured (no effect on 160/160 replays, #104)."),
        "smoke_evidence": smoke_evidence,
        "inputs": input_bindings(output),
        "claim_boundary": ("training of a frozen recipe on retained fit-split captures; no statistic, no ranking, no "
                           "held-out access; Gate B is the test"),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-112 retrain protocol")
    if not plan.get("frozen_before_data_and_training") or not plan["smoke_evidence"].get("ok"):
        raise ValueError("plan.json does not declare a passed-smoke freeze")
    for entry in plan["inputs"]:
        if entry["name"] != "smoke" and sha256_of(ROOT / entry["artifact"]) != entry["sha256"]:
            raise ValueError(f"frozen input {entry['name']} changed after the freeze")
    return plan


def prepare(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        load_plan(output)
        log("existing frozen plan validated")
        return 0
    for name in ("shots", "data", "models", "training"):
        if (output / name).exists():
            raise ValueError(f"{name} exist before the freeze; refusing")
    evidence = read_json(output / "smoke.json")
    if not evidence.get("ok"):
        raise ValueError("smoke failed; STOP RULE: no freeze")
    plan = frozen_plan(output, utc_now(), evidence)
    (output / "plan.json").write_text(json_text(plan))
    load_plan(output)
    log(f"frozen plan published: {plan['data']['complete_branches']} fit branches of {plan['data']['levels']} levels")
    return 0


# ---------------------------------------------------------------------------
# modes: data, assemble, train
# ---------------------------------------------------------------------------

def data(output, workers, worker):
    output = Path(output)
    load_plan(output)
    began = time.monotonic()
    shard = complete_branches()[worker::workers]
    pending = [b for b in shard if not shot_path(output, b["key"]).is_file()]
    log(f"worker {worker}/{workers}: {len(pending)} of {len(shard)} shots to build")
    adapter = s99.load_adapter(s99.OUTPUT, "E")
    with ProcessPoolExecutor(LABEL_WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(shot_labels, b["attempt"]) for b in pending]
        for done, (branch, future) in enumerate(zip(pending, futures), 1):
            atomic_torch(shot_path(output, branch["key"]), shot_entry(branch, future.result(), adapter))
            if done % 200 == 0:
                elapsed = time.monotonic() - began
                log(f"worker {worker}: {done}/{len(pending)} shots, {elapsed:.0f}s, "
                    f"eta {elapsed / done * (len(pending) - done):.0f}s")
    wall = time.monotonic() - began
    mark(output, f"data-worker{worker}", wall, wall, shots=len(shard))
    log(f"worker {worker}: shots ready ({wall:.0f}s)")
    return 0


def read_shot(output, key):
    entry = torch.load(shot_path(output, key), map_location="cpu", weights_only=True)
    if entry.get("schema") != SCHEMA_SHOT or entry.get("plan_identity") != IDENTITY or entry["key"] != key:
        raise ValueError(f"shot {key} binding differs")
    return entry


def assemble(output):
    output = Path(output)
    load_plan(output)
    began = time.monotonic()
    branches = complete_branches()
    with harness.deterministic_scoring() as policy:
        adapter = s99.load_adapter(s99.OUTPUT, "E")
        anchors = {member: anchor_carrier(adapter, b["attempt"]) for member, b in anchor_branches(branches).items()}
    atomic_torch(anchors_path(output), {"scoring_policy": policy, "anchors": anchors,
                                        "frames": {m: b["key"] for m, b in anchor_branches(branches).items()}})
    verdicts = fit_verdicts()
    shots = [read_shot(output, b["key"]) for b in branches]
    actions_agree = all(torch.equal(s["action"], wlc.action_tensor(b["candidate"]["action"], "cpu")[0])
                        for s, b in zip(shots, branches))
    if not actions_agree:
        raise ValueError("an executed action differs from its inventory action")
    cells = ranking_cells(branches, verdicts, anchors)
    atomic_torch(data_path(output), build_payload(shots, cells))
    write_json(verdicts_path(output), {
        "schema": "issue_112_v3_retrain_fit_verdicts_v1", "plan_identity": IDENTITY, "stream": STREAM,
        "source": "issue-104-power-cohort-campaign-v1/branches.json (fit-stream rows)",
        "branches": [{"key": b["key"], "level": b["member"], "family": b["family"], "suffix": b["candidate"]["suffix"],
                      "action": b["candidate"]["action"], "inventory_entries": b["candidate"]["inventory_entries"],
                      "pig_removed": verdicts[b["key"]]} for b in branches]})
    wall = time.monotonic() - began
    mark(output, "assemble", wall, 0.0, shots=len(shots), cells=len(cells))
    log(f"training data assembled: {len(shots)} shots, {len(cells)} ranking cells ({wall:.0f}s)")
    return 0


def train(output, workers, worker):
    output = Path(output)
    load_plan(output)
    impl = implementation()
    torch.backends.cudnn.benchmark = True
    mine = [g for k, g in enumerate(groups()) if k % workers == worker
            and not updates_path(output, g[0], g[1]).is_file()]
    if not mine:
        return 0
    payload = load_training_data(output)
    current = None
    for family, index, seed in mine:
        if current is None or current[0] != family:
            current = None
            torch.cuda.empty_cache()
            current = (family, IndexedTrainData(payload, family))
        began = time.monotonic()
        report, ensemble = impl.train_group(output, ARM, family, index, [("all", seed)], current[1])
        atomic_torch(updates_path(output, family, index), {
            "schema": SCHEMA_UPDATES, "plan_identity": IDENTITY, "family": family, "index": index,
            "members": [("all", seed)], "requests": step_requests(family), "skipped": ensemble.skipped.cpu(),
            "logs": ensemble.log_tensors()})
        del ensemble
        torch.cuda.empty_cache()
        mark(output, f"train-{family}-group{index}", time.monotonic() - began, report["gpu_seconds"])
        log(f"trained {ARM}/{family}/group{index} (seed {seed}): {report['gpu_seconds']:.0f}s")
    return 0


# ---------------------------------------------------------------------------
# publication
# ---------------------------------------------------------------------------

def member_table(output):
    impl = implementation()
    rows = []
    for family, index, seed in groups():
        report = read_json(impl.group_path(output, ARM, family, index))
        updates = torch.load(updates_path(output, family, index), map_location="cpu", weights_only=True)
        member = report["members"][0]
        local, norms = updates["logs"]["local"][:, 0].double(), updates["logs"]["grad_norm"][:, 0].double()
        requests = updates["requests"]
        end_state = excursion(local, norms, requests) if member["complete"] else None
        tail = {r: float(np.median([float(local[i]) for i, q in enumerate(requests) if q == r][-100:]))
                for r in sorted(set(requests))}
        last = report["trace"][-1]
        rows.append({"family": family, "seed": seed, "index": index, "complete": member["complete"],
                     "retired_at": member["retired_at"], "skipped": int(updates["skipped"][0]),
                     "logged_updates": len(local), "gpu_seconds": report["gpu_seconds"],
                     "peak_memory_gb": report["peak_memory_gb"], "pool_sizes": report["pool_sizes"][0],
                     "final_trace": {k: last[k] for k in ("loss", "local", "long", "rank", "grad_norm")},
                     "median_local_last_100_per_request": tail,
                     "max_pre_clip_norm": float(norms[torch.isfinite(norms)].max()),
                     "max_pre_clip_norm_update": int(torch.nan_to_num(norms, nan=-1.0, posinf=-1.0).argmax()),
                     "updates_above_spike_norm": {r: sum(1 for i, q in enumerate(requests)
                                                         if q == r and float(norms[i]) > SPIKE_NORM)
                                                  for r in sorted(set(requests))},
                     "spike_updates_after_500": [i + 1 for i in (norms > SPIKE_NORM).nonzero().flatten().tolist()
                                                 if i >= 500],
                     "end_state": end_state,
                     "sha256": sha256_of(impl.member_path(output, ARM, family, "all", seed))})
    return rows


def data_summary(output):
    payload = load_training_data(output)
    branches = fit_branches()
    roster = {level["identity"] for level in fit_levels()}
    complete = {b["key"] for b in branches if b["status"] == "complete"}
    cells = Counter(payload["cell_inventories"])
    g1 = (set(payload["shot_keys"]) == complete and set(payload["shot_members"]) <= roster
          and set(payload["cell_members"]) <= roster)
    return {"levels": len(roster), "branches": len(branches), "shots": len(payload["shot_keys"]),
            "typed_failures_excluded": len(branches) - len(complete), "positions": len(payload["carriers"]),
            "windows": len(payload["window_positions"]), "long_windows": len(payload["long_windows"]),
            "ranking_cells": dict(sorted(cells.items())), "ranking_levels": len(set(payload["cell_members"])),
            "v3_reference": {"shots": 745, "windows": 11780, "long_windows": 3725, "ranking_cells": 27},
            "training_data_sha256": sha256_of(data_path(output)), "G1": g1}


def compute_tables(output, plan):
    output = Path(output)
    members = member_table(output)
    data_block = data_summary(output)
    ledger = {p.stem: read_json(p) for p in sorted((output / "ledger").glob("*.json"))}
    g2 = all(m["complete"] and m["skipped"] <= SKIP_CAP and m["logged_updates"] == STEPS for m in members)
    for m in members:
        m["usable"] = bool(m["complete"] and not m["end_state"]["excursion"])
        m["typed_failure"] = (None if m["usable"] else "retired" if not m["complete"] else "final_checkpoint_excursion")
    usable = {family: [m["seed"] for m in members if m["family"] == family and m["usable"]] for family in FAMILIES}
    ready = data_block["G1"] and all(usable[f] for f in FAMILIES)
    return {"schema": SCHEMA_COMPUTE, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "units": plan["units"], "data": data_block, "members": members,
            "guards": {"G1_isolation": data_block["G1"], "G2_stability": g2,
                       "G3_end_state": {f"{m['family']}/{m['seed']}": bool(m["complete"] and not m["end_state"]["excursion"])
                                        for m in members}},
            "usable_seeds": usable,
            "handoff": {"token": "supported" if ready else STOP_TOKEN, "rule": plan["decision_rule"]["handoff"]},
            "compute": {"phases": ledger, "gpu_seconds": sum(b["gpu_seconds"] for b in ledger.values()),
                        "wall_seconds": sum(b["wall_seconds"] for b in ledger.values()), "engine_seconds": 0}}


def handoff(output, plan, compute):
    impl = implementation()
    v3_handoff = read_json(versions.implementation(VERSION).OUTPUT / "gate_b_handoff.json")

    def entries(family):
        return [{"seed": m["seed"], "path": str(impl.member_path(output, ARM, family, "all", m["seed"]).relative_to(ROOT)),
                 "sha256": m["sha256"], "complete": m["complete"], "usable": m["usable"],
                 "typed_failure": m["typed_failure"]} for m in compute["members"] if m["family"] == family]

    return {"schema": SCHEMA_HANDOFF, "plan_identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "supersedes": ".local-artifacts/issue-112-decision-dynamics-v3/gate_b_handoff.json",
            "gate": v3_handoff["gate"], "recipe": plan["recipe"],
            "unit": {**plan["units"]["gate_b"], "checkpoints": entries("hybrid"),
                     "usable_seeds": compute["usable_seeds"]["hybrid"]},
            "c_star": {**plan["units"]["c_star"], "checkpoints": entries("continuous"),
                       "usable_seeds": compute["usable_seeds"]["continuous"]},
            "load_checkpoint": ("scripts.run_decision_dynamics_retrain.implementation().load_member(OUTPUT, 'LC', "
                                "family, 'all', seed)"),
            "encoder": v3_handoff["encoder"],
            "anchors": ("evaluation levels must use the anchor rule of this training data: E99 batch-1 parse "
                        "(wlc.encode_carrier) of the level's decision frame (shot frame 0 of its first complete "
                        "candidate in suffix order)"),
            "scoring": v3_handoff["scoring"], "seed_aggregation": v3_handoff["seed_aggregation"],
            "scope": versions.GATE_B_SCOPE,
            "fit_verdicts": str(verdicts_path(output).relative_to(ROOT)),
            "training_data": {"path": str(data_path(output).relative_to(ROOT)),
                              "sha256": compute["data"]["training_data_sha256"]},
            "token": compute["handoff"]["token"]}


def fmt(value, digits=4):
    if value is None:
        return "n/a"
    return format(value, f".{digits}f") if abs(value) >= 1e-3 or value == 0 else f"{value:.2e}"


def findings_md(plan, compute):
    lines = []
    add = lines.append
    d = compute["data"]
    add("# Issue #112 v3 retrain on the #104 fit split (Gate-B checkpoints)")
    add("")
    add(f"Frozen {plan['frozen_at']} before any data build or training. Validation: `{VALIDATION_COMMAND}`.")
    add("Nothing is selected or scored here; no evaluation, policy or #110 capture is read.")
    add("")
    add("## Training data")
    add("")
    add("| quantity | retrain (#109 fit split) | v3 (12 #77 N1 lineages) |")
    add("| --- | --- | --- |")
    add(f"| levels | {d['levels']} | 12 |")
    add(f"| shots (complete branches) | {d['shots']} ({d['typed_failures_excluded']} typed failure excluded) | 745 |")
    add(f"| windows | {d['windows']} | {d['v3_reference']['windows']} |")
    add(f"| long windows | {d['long_windows']} | {d['v3_reference']['long_windows']} |")
    add(f"| ranking cells | {sum(d['ranking_cells'].values())} ({', '.join(f'{k} {v}' for k, v in d['ranking_cells'].items())}) "
        f"over {d['ranking_levels']} levels | {d['v3_reference']['ranking_cells']} |")
    add(f"| carrier positions | {d['positions']} | 208886 |")
    add("")
    add("## Members (arm LC, fold 'all', 9000 updates)")
    add("")
    add("| unit | seed | complete | skipped | max pre-clip norm (update, request) | final local / long / rank | end state "
        "| usable |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for m in compute["members"]:
        unit = f"{m['family']} {UNITS[m['family']]}"
        trace = m["final_trace"]
        terms = " / ".join(fmt(trace[k][0]) if trace[k] is not None else "n/a" for k in ("local", "long", "rank"))
        state = m["end_state"]
        if state is None:
            end = f"retired at {m['retired_at']}"
        elif state["excursion"]:
            end = f"excursion ({', '.join(state['requests_elevated'])})"
        else:
            end = f"no excursion (max ratio {max(state['ratios'].values()):.2f})"
        peak = m["max_pre_clip_norm_update"]
        add(f"| {unit} | {m['seed']} | {m['complete']} | {m['skipped']} | {m['max_pre_clip_norm']:.3g} ({peak + 1}, "
            f"{step_requests(m['family'])[peak]}) | {terms} | "
            f"{end} | {m['usable']} |")
    add("")
    add(f"Updates with a pre-clip gradient norm above {SPIKE_NORM:g} (clipped to the recipe's grad_clip; per request "
        "with any):")
    add("")
    for m in compute["members"]:
        spikes = {r: n for r, n in m["updates_above_spike_norm"].items() if n}
        late = m["spike_updates_after_500"]
        add(f"- {m['family']} {m['seed']}: {sum(spikes.values())} of {m['logged_updates']}"
            + (f" ({', '.join(f'{r} {n}' for r, n in spikes.items())})" if spikes else "")
            + (f"; after update 500: {len(late)}, updates {late[0]}–{late[-1]}" if late else "; none after update 500"))
    add("")
    add("Median #77 local loss over each request's last 100 updates:")
    add("")
    for family in FAMILIES:
        rows = [m for m in compute["members"] if m["family"] == family]
        requests = list(rows[0]["median_local_last_100_per_request"])
        add(f"| {family} seed | " + " | ".join(requests) + " |")
        add("| --- | " + " | ".join("---" for _ in requests) + " |")
        for m in rows:
            add(f"| {m['seed']} | " + " | ".join(fmt(m["median_local_last_100_per_request"][r]) for r in requests) + " |")
        add("")
    g = compute["guards"]
    add("## Guards and handoff")
    add("")
    add(f"- G1 isolation (fit-stream branches of fit-roster levels only): {g['G1_isolation']}")
    add(f"- G2 stability (no retirement, at most {SKIP_CAP} skipped updates): {g['G2_stability']}")
    add(f"- G3 end state (#117 excursion rule): {g['G3_end_state']}")
    add(f"- Usable seeds: Gate-B unit {compute['usable_seeds']['hybrid']}, C* {compute['usable_seeds']['continuous']}")
    add(f"- Gate-B handoff: **{compute['handoff']['token']}** (`gate_b_handoff.json`, supersedes the v3 handoff)")
    add("")
    c = compute["compute"]
    add(f"Compute: {c['gpu_seconds']:.0f} GPU s, {c['wall_seconds']:.0f} wall s over all phases and workers; "
        "0 engine seconds.")
    add("")
    return "\n".join(lines)


def summary(plan, compute):
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "validation_command": VALIDATION_COMMAND, "units": plan["units"], "guards": compute["guards"],
            "usable_seeds": compute["usable_seeds"], "handoff_token": compute["handoff"]["token"],
            "claim_boundary": plan["claim_boundary"], "engine_seconds": 0}


def rendered(plan, compute):
    return {"compute.json": json_text(compute), "summary.json": json_text(summary(plan, compute)),
            "findings.md": findings_md(plan, compute), "gate_b_handoff.json": json_text(handoff(OUTPUT, plan, compute))}


def training_complete(output):
    impl = implementation()
    return all(impl.group_path(output, ARM, f, i).is_file() and updates_path(output, f, i).is_file()
               for f, i, _ in groups())


def publish(output):
    output = Path(output)
    plan = load_plan(output)
    if not training_complete(output):
        raise ValueError("training is not complete")
    for name, text in rendered(plan, compute_tables(output, plan)).items():
        (output / name).write_text(text)
    log("published compute.json, summary.json, findings.md, gate_b_handoff.json")
    return 0


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    compute = compute_tables(output, plan)
    for name, text in rendered(plan, compute).items():
        if (output / name).read_text() != text:
            raise ValueError(f"{name} differs from its recomputation")
    impl = implementation()
    for family, index, seed in groups():
        if impl.load_member(output, ARM, family, "all", seed) is None:
            log(f"  {family}/{seed}: retired (typed failure)")
    # stored data against a fresh parse: one shot's carriers and its level anchor
    branch = complete_branches()[0]
    stored = read_shot(output, branch["key"])
    anchors = torch.load(anchors_path(output), map_location="cpu", weights_only=True)["anchors"]
    with harness.deterministic_scoring():
        adapter = s99.load_adapter(s99.OUTPUT, "E")
        fresh = s99.carriers_for(adapter, branch["attempt"], list(range(stored["end"] + 1)))
        anchor = anchor_carrier(adapter, branch["attempt"])
    checks = [harness.carrier_replication(stored["carriers"], fresh),
              harness.carrier_replication(anchors[branch["member"]], anchor)]
    if not all(c["pass"] for c in checks):
        raise ValueError(f"stored carriers differ from a fresh parse: {checks}")
    log(f"validation passed ({time.monotonic() - began:.0f}s): handoff {compute['handoff']['token']}, "
        f"usable seeds {compute['usable_seeds']}")
    return 0


def dry_run(output):
    branches = fit_branches()
    complete = [b for b in branches if b["status"] == "complete"]
    log(f"output {Path(output)}; levels {len(fit_levels())}; branches {len(branches)}, complete {len(complete)}")
    log(f"groups {groups()}; units {UNITS}; v3 arm definition: {implementation().ARM_DEFINITIONS[ARM]}")
    return 0


MODES = {"dry-run": dry_run, "smoke": smoke, "prepare": prepare, "data": data, "assemble": assemble,
         "train": train, "publish": publish, "validate": validate}
SHARDED = {"data", "train"}


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
