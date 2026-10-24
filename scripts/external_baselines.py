"""Issue-127: the shared protocol module of the external-baselines program (imported as ``eb``).

Binding for every #127 child runner (#128 A, #129 B1, #130 C1, #131 D1); the contract is
``novphy-spec-notes/round3/SHARED_PROTOCOL_API.md``. Arm-agnostic: no model code, no vendored imports. The hash-pinned
upstream runners are reused by import only (``rc`` #123, ``rt`` #112 v3 retrain, ``v1``, ``gb`` Gate B, ``m99``,
``c2``, ``s99``, the scoring harness); where #123 hard-codes its own identity, output paths or H-JEPA arm names, this
module carries a parameterised equivalent. ``world_model.hjepa.novphy`` (``hn``) is imported lazily, for its
diagnostic helpers only.

Data: the #109 fit split (training), the policy split (selection, readout ceilings) and, only after an arm's
model / readout freeze, the evaluation split (``evaluation_cells``, ``read_evaluation_anchors``). Every interval is a
member (level)-clustered percentile bootstrap (10 000 draws, PCG64 7201), DESCRIPTIVE.

Shared data products (``--build-shared``, CPU): ``SHARED/fit-anchor-frames.pt`` and ``SHARED/policy-anchor-frames.pt``
(frame 0 of every fit / policy anchor branch from the #123 pixel stores) and ``SHARED/MANIFEST.json``.
"""
from __future__ import annotations

import argparse
import functools
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch

from scripts import run_decision_chain_attribution as m99
from scripts import run_decision_dynamics_controls as c2
from scripts import run_decision_dynamics_retrain as rt
from scripts import run_decision_relevant_dynamics as v1
from scripts import run_gate_b as gb
from scripts import run_hjepa_comparison as rc
from scripts import run_slot_encoder_fix as s99
from world_model.planning import scoring_harness as harness

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

ROOT = rc.ROOT
PROGRAM_ISSUE = 127
MODULE = "scripts/external_baselines.py"
SHARED = ROOT / ".local-artifacts/issue-127-shared-v1"
SEEDS = tuple(rc.SEEDS)  # 20260908, 20260909, 20260910
SELECTION_SEED = SEEDS[0]
DELTA = v1.DELTA  # 0.02
STOP_TOKEN = v1.STOP_TOKEN  # readiness_or_precision_insufficient
INVENTORIES = tuple(rc.EVAL_INVENTORIES)  # grid, offset, angle, power
POSITION_TIMES = tuple(rc.POSITION_TIMES)  # 15, 30, 60, 150, 225
EXIT_PAUSED = rc.EXIT_PAUSED  # 75, EX_TEMPFAIL
PRIMARY_COST = harness.PRIMARY_COST  # tie_free
GUARD = {
    "g_rank_tie_rel": 1e-6,
    "g_rank_mean_tied_share_max": 0.5,
    "g_rank_fully_tied_level_share_max": 0.10,
    "action_sensitivity_min": 0.05,
    "effective_rank_fraction_min": 0.1,
    "within_episode_share_min": 0.05,
    "readout_ceiling_auc_min": 0.65,
    "nonfinite_share_max": 0.10,
}
PROGRAM_SET = ("A1", "A2", "A4", "B1", "C1-dino", "D1")
DISPOSITION_LABELS = {
    1: "X ranks launches better than U on the sealed NovPhy grid (scoped)",
    2: "U ranks launches better than X (scoped)",
    3: "No AUC difference of 0.02 or more",
    4: "per-contrast tokens only",
}
HJ123 = rc.OUTPUT  # the #123 data stores (pixels, fit carriers, policy data)
RETRAIN = rt.OUTPUT
GATE_B = rc.GATE_B
ISSUE_101 = rc.ISSUE_101
J_ARMS = dict(rc.J_ARMS)
SCHEMA_FIT_FRAMES = "issue_127_fit_anchor_frames_v1"
SCHEMA_POLICY_FRAMES = "issue_127_policy_anchor_frames_v1"
SCHEMA_SHARED_MANIFEST = "issue_127_shared_manifest_v1"
SCHEMA_EVAL_ANCHORS = "issue_127_eval_anchors_v1"
SCHEMA_RECORDS_MANIFEST = "issue_127_records_manifest_v1"
SCHEMA_RECEIPT = "issue_127_validation_receipt_v1"
VALIDATION_RECEIPT = rc.VALIDATION_RECEIPT  # validation.json

# ---------------------------------------------------------------------------
# training plumbing (#123 A1 pattern: SIGTERM / SIGUSR1 -> checkpoint after the current step -> exit 75)
# ---------------------------------------------------------------------------

read_json = rc.read_json
write_json = rc.write_json
json_text = rc.json_text
sha256_of = rc.sha256_of
atomic_torch = rc.atomic_torch
utc_now = rc.utc_now
STOP_REQUESTED = rc.STOP_REQUESTED  # signal numbers received; a training loop checks it at every step boundary
request_stop = rc.request_stop
TrainingPaused = rc.TrainingPaused


def log(issue, message):
    print(f"[issue-{issue}] {message}", flush=True)


def install_stop_handlers():
    for signum in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, request_stop)


def batch_rng(seed, b):
    """The generator of batch b of a run (PCG64([seed, b]); deterministic and resumable at any step)."""
    return np.random.Generator(np.random.PCG64([seed, b]))


# ---------------------------------------------------------------------------
# launch encodings and the grid
# ---------------------------------------------------------------------------

def launch4(action):
    """[drag_x/480, drag_y/480, release_time_ms/1000, tap_time_ms/1000] (the A1 / A2 / A4 action-MLP input)."""
    return torch.tensor([action["drag_x"] / 480., action["drag_y"] / 480., action["release_time_ms"] / 1000.,
                         action["tap_time_ms"] / 1000.], dtype=torch.float32)


def launch5(action):
    """U's launch vector [.., .., .., .., 1] (rc.wlc_action = wlc.action_tensor); world-model adapters use it."""
    return rc.wlc_action(action)


def grid_items():
    """The 16 grid {ordinal, action} items (drag_x in {-40..-160} x drag_y in {5..80}, release 1000 ms)."""
    return rc.grid_items()


# ---------------------------------------------------------------------------
# fit split (training data; never evaluation)
# ---------------------------------------------------------------------------

class FitShots:
    """The frozen fit shot order of rc.FitData (= rt.complete_branches(): 13 919 complete of 13 920 scheduled
    branches, 480 levels) with launch encodings and the engine verdicts."""

    def __init__(self):
        self._data = rc.FitData(HJ123)
        branches = rt.complete_branches()
        if [b["key"] for b in branches] != self._data.keys:
            raise ValueError("the #123 fit carrier store and rt.complete_branches() disagree on the shot order")
        self.keys = list(self._data.keys)
        self.members = list(self._data.members)
        self.ends = self._data.ends
        self.row_of = {key: i for i, key in enumerate(self.keys)}
        self.actions = [dict(b["candidate"]["action"]) for b in branches]
        self.launch4 = torch.stack([launch4(a) for a in self.actions])
        self.launch5 = torch.stack([launch5(a) for a in self.actions])
        if not torch.equal(self.launch5, self._data.actions):
            raise ValueError("launch5 differs from the #123 fit store's launch vectors")
        fit = read_json(rt.verdicts_path(RETRAIN))["branches"]
        if [b["key"] for b in fit] != self.keys:
            raise ValueError("retrain fit-verdicts.json and the fit shot order disagree")
        campaign = rt.fit_verdicts()  # the campaign branches.json fit-stream rows
        retrain = [bool(b["pig_removed"]) for b in fit]
        if retrain != [campaign[key] for key in self.keys]:
            raise ValueError("retrain fit-verdicts.json and campaign branches.json disagree on a verdict")
        self.verdicts = torch.tensor(retrain, dtype=torch.bool)
        self.levels = sorted(set(self.members))

    def __len__(self):
        return len(self.keys)

    def carrier_rows(self, shots, positions):
        """E99 carriers float32 [N, 236] of (shot, position) pairs, positions 0..end (the #123 fit memmap)."""
        return self._data.carrier_rows(shots, positions)

    def frame(self, shot, position):
        """uint8 [3, 240, 320] frame of the #123 fit pixel store (s99.image_tensor of the shot's frame)."""
        return self._data.frame(shot, position)


def fit_anchors():
    """The 480 Gate-B wrong-anchor pool: {"carriers": {member: float[236]}, "keys": {member: anchor branch key}}."""
    store = torch.load(rt.anchors_path(RETRAIN), map_location="cpu", weights_only=True)
    return {"carriers": store["anchors"], "keys": dict(store["frames"])}


def _load_product(name, schema):
    path = SHARED / name
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run python -m scripts.external_baselines --build-shared")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != schema:
        raise ValueError(f"{path} binding differs")
    return payload


def fit_anchor_frames():
    """{member: uint8 [3, 240, 320]}: frame 0 of each fit anchor branch (SHARED/fit-anchor-frames.pt)."""
    return _load_product("fit-anchor-frames.pt", SCHEMA_FIT_FRAMES)["frames"]


# ---------------------------------------------------------------------------
# policy split (selection and readout ceilings only)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _policy_data():
    return torch.load(rc.policy_data_path(HJ123), map_location="cpu", weights_only=True)


@functools.lru_cache(maxsize=None)
def _policy_cells(inventory):
    entries = gb.inventory_entries(rc.split_levels("policy"), rc.split_branches("policy"), rc.split_verdicts("policy"),
                                   inventory)
    return {m: e for m, e in entries.items() if set(e["verdicts"].values()) == {True, False}}


def policy_cells(inventory="grid"):
    """Mixed policy cells under the Gate-B rule: {member: {"inventory": [{ordinal, action, key}],
    "verdicts": {ordinal: bool}}} (rc.policy_split's rule, any inventory)."""
    return _policy_cells(inventory)


def policy_anchors():
    """{"carriers": {member: float[236]} (E99 batch-1 parses), "keys": {member: anchor branch key}} (#123 policy.pt)."""
    data = _policy_data()
    return {"carriers": data["anchors"], "keys": dict(data["anchor_frames"])}


def policy_anchor_frames():
    """{member: uint8 [3, 240, 320]}: frame 0 of each policy anchor branch (SHARED/policy-anchor-frames.pt)."""
    return _load_product("policy-anchor-frames.pt", SCHEMA_POLICY_FRAMES)["frames"]


def policy_shots():
    """{key: {"end", "frame_count", "positions" (every 15th, clamped to end), "carriers", "action" (launch5)}}."""
    return _policy_data()["shots"]


def policy_grid_auc(rows_by_member, key):
    """Pooled mean cell AUC over the policy grid mixed cells (rc.pooled_auc); None if no cell has an AUC."""
    return rc.pooled_auc(policy_cells("grid"), rows_by_member, key)


SELECTION_RULE = ("largest policy-split pooled grid AUC at 4 d.p.; ties -> fewer trainable parameters -> "
                  "declaration order; a configuration without an AUC is never selected")


def select_configuration(results):
    """results: {name: {"auc": float | None, "parameters": int}} in declaration order."""
    eligible = [(i, name, r) for i, (name, r) in enumerate(results.items())
                if r["auc"] is not None and math.isfinite(r["auc"])]
    selected = min(eligible, key=lambda t: (-round(t[2]["auc"], 4), t[2]["parameters"], t[0]))[1] if eligible else None
    return {"selected": selected, "rule": SELECTION_RULE, "results": results}


# ---------------------------------------------------------------------------
# rows (records keep one row per candidate)
# ---------------------------------------------------------------------------

def score_rows(items, scores, key="neg_score", extras=None):
    """Direct predictors: row value = -score (a higher logit / Q ranks first under m99.view, lower cost first);
    a nonfinite score is a retained typed failure (excluded True, value None). extras: {key: [N]} per row."""
    rows = []
    for slot, item in enumerate(items):
        score = float(scores[slot])
        finite = math.isfinite(score)
        row = {"ordinal": item["ordinal"], "excluded": not finite, "score": score if finite else None,
               key: -score if finite else None}
        for name, values in (extras or {}).items():
            value = values[slot]
            row[name] = None if not finite or value is None or not math.isfinite(float(value)) else float(value)
        rows.append(row)
    return rows


def endpoint_rows(items, z_end, z0, costs):
    """harness.EndpointCosts.row per candidate (tie-free primary) of predicted endpoint carriers (rc.cost_rows)."""
    return rc.cost_rows(items, z_end, z0, costs)


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def g_rank(costs):
    """G-rank on per-level candidate costs [L, K] (480 fit anchors x 16 grid launches). Within a level a pair is
    tied iff |s_i - s_j| <= 1e-6 * max(1, max_k |s_k|) (max over the level's finite values); a pair with a
    nonfinite value counts as tied (it carries no order); a level is fully tied iff every pair is tied (zero
    spread included). Pass iff the mean tied-pair share <= 0.5 and the fully-tied level share <= 0.10."""
    c = np.asarray(costs, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] < 2:
        raise ValueError("g_rank expects costs [levels, candidates >= 2]")
    upper = np.triu_indices(c.shape[1], 1)
    shares, full = [], 0
    for row in c:
        finite = np.isfinite(row)
        scale = max(1.0, float(np.abs(row[finite]).max())) if finite.any() else 1.0
        tolerance = GUARD["g_rank_tie_rel"] * scale
        both = (finite[:, None] & finite[None, :])[upper]
        with np.errstate(invalid="ignore"):
            close = (np.abs(row[:, None] - row[None, :]) <= tolerance)[upper]
        tied = close | ~both
        shares.append(float(tied.mean()))
        full += bool(tied.all())
    mean_share = float(np.mean(shares))
    full_share = full / len(c)
    return {"levels": int(len(c)), "mean_tied_pair_share": mean_share, "fully_tied_level_share": full_share,
            "pass": bool(mean_share <= GUARD["g_rank_mean_tied_share_max"]
                         and full_share <= GUARD["g_rank_fully_tied_level_share_max"])}


def action_sensitivity(endpoints):
    """hn.action_sensitivity (endpoints [A, K, D]): mean ||z_i - z_j|| / ||z_i|| over anchors, ordered pairs."""
    from world_model.hjepa import novphy as hn
    return hn.action_sensitivity(endpoints)


def effective_rank(z):
    from world_model.hjepa import novphy as hn
    return hn.effective_rank(z)


def within_episode_share(z, episodes):
    from world_model.hjepa import novphy as hn
    return hn.within_episode_share(z, episodes)


def nonfinite_share(records, key, inventory="grid"):
    """Share of the inventory's verdict slots, over the scored (non-typed-failure) records, whose value under key is
    None or nonfinite (rc.nonfinite_shares, one key); None when no record was scored."""
    slots = bad = 0
    for record in records:
        if record.get("typed_failure"):
            continue
        for cell in record["inventories"][inventory].values():
            for row in cell["rows"]:
                slots += 1
                bad += row.get(key) is None or not np.isfinite(row[key])
    return bad / slots if slots else None


GUARD_CHECKS = ("G_rank", "G_action", "G_collapse", "G_readout", "G_nonfinite", "G_trained")
SEED_GUARDS = ("G_rank", "G_action", "G_collapse")


def _seed_guards(entry):
    if entry is None or entry.get("typed_failure"):
        return dict.fromkeys(SEED_GUARDS, False)
    rank, action, collapse = entry.get("G_rank"), entry.get("G_action"), entry.get("G_collapse")
    return {"G_rank": bool(rank is not None and rank["pass"]),
            "G_action": bool(action is not None and action >= GUARD["action_sensitivity_min"]),
            "G_collapse": bool(collapse is not None
                               and collapse["effective_rank"] >= GUARD["effective_rank_fraction_min"] * collapse["dim"]
                               and collapse["within_episode_share"] >= GUARD["within_episode_share_min"])}


def arm_guard_block(per_seed, checks, nonfinite=None):
    """Per-arm guards with the GUARD thresholds. per_seed: {seed: {"G_rank": g_rank dict, "G_action": float | None,
    "G_collapse": {"effective_rank", "dim", "within_episode_share"} | None, "policy_ceiling_auc": float | None,
    "typed_failure": reason | None}}; checks: subset of GUARD_CHECKS. G_rank / G_action / G_collapse must pass on
    every seed (a missing or typed-failure seed fails them); G_readout on the mean policy ceiling over the scored
    seeds (#123); G_nonfinite on nonfinite = nonfinite_share(the arm's scored records, its key); G_trained iff every
    seed is present without a typed failure."""
    unknown = set(checks) - set(GUARD_CHECKS)
    if unknown:
        raise ValueError(f"unknown guard checks {sorted(unknown)}")
    seeds, ceilings = {}, []
    for seed in SEEDS:
        entry = per_seed.get(seed, per_seed.get(str(seed)))
        failure = "missing" if entry is None else entry.get("typed_failure")
        seeds[str(seed)] = {**_seed_guards(entry), "typed_failure": failure}
        if failure is None and entry.get("policy_ceiling_auc") is not None:
            ceilings.append(entry["policy_ceiling_auc"])
    ceiling = float(np.mean(ceilings)) if ceilings else None
    guards = {}
    for check in checks:
        if check in SEED_GUARDS:
            guards[check] = all(s[check] for s in seeds.values())
        elif check == "G_readout":
            guards[check] = bool(ceiling is not None and ceiling >= GUARD["readout_ceiling_auc_min"])
        elif check == "G_nonfinite":
            guards[check] = bool(nonfinite is not None and nonfinite <= GUARD["nonfinite_share_max"])
        else:
            guards[check] = all(s["typed_failure"] is None for s in seeds.values())
    return {"checks": list(checks), "seeds": seeds, "policy_ceiling_auc": ceiling, "nonfinite_share": nonfinite,
            **guards, "pass": all(guards.values())}


# ---------------------------------------------------------------------------
# evaluation (only after the arm's model / readout freeze)
# ---------------------------------------------------------------------------

def evaluation_cells():
    """inventory -> {"all", "mixed"} of the sealed #109 evaluation split (gb.evaluation_set()[1])."""
    return gb.evaluation_set()[1]


def evaluation_anchors_path(output):
    return Path(output) / "data" / "anchors-evaluation.pt"


def read_evaluation_anchors(output, scoring_policy):
    """output/data/anchors-evaluation.pt: the Gate-B anchor carriers and frame 0 of the same anchor branches
    (s99.image_tensor; exactly rc.read_pixel_anchors). Call only after the arm's model / readout freeze."""
    store = gb.load_anchors(gb.OUTPUT)
    frames = {}
    for member, key in sorted(store["frames"].items()):
        _, observation_root, shot, _ = s99.shot_frames(rc.eval_attempt(key))
        frames[member] = s99.image_tensor(observation_root / shot[0]["agent_observation"]["relative_path"])
    payload = {"schema": SCHEMA_EVAL_ANCHORS, "carriers": store["anchors"], "frames": frames,
               "keys": dict(store["frames"]), "read_at": utc_now(), "scoring_policy": scoring_policy}
    atomic_torch(evaluation_anchors_path(output), payload)
    return payload


# ---------------------------------------------------------------------------
# position error (DESCRIPTIVE; evaluation data: post-freeze only)
# ---------------------------------------------------------------------------

def position_cache():
    """POST-FREEZE ONLY (evaluation content). The #123 evaluation positions cache: per grid-mixed-cell branch key,
    engine presence bool [5, 18] / centres [5, 18, 2], true frames (zstd) and #113 carriers at POSITION_TIMES."""
    cache = torch.load(rc.eval_cache_path(HJ123), map_location="cpu", weights_only=True)
    if cache.get("schema") != "issue_123_eval_positions_v1" or list(cache["times"]) != list(POSITION_TIMES):
        raise ValueError("the #123 evaluation positions cache binding differs")
    return cache


def position_branches(cells):
    """(member, item) of every complete grid candidate of every mixed grid cell, frozen order (rc)."""
    return rc.position_branches(cells)


def decode_frame(blob):
    return rc.decode_frame(blob)


def carrier_position_mse(pred, presence, centers):
    """Position error of a predicted carrier [236] (the #123 score_reference_curves rule): slot centres
    pred[2:2+13*18].reshape(18, 13)[:, 5:7] against the engine centres on engine-present slots; None if the carrier
    is nonfinite or no slot is present."""
    if not bool(torch.isfinite(pred).all()) or not bool(presence.any()):
        return None
    p = pred[2:2 + 13 * 18].reshape(18, 13)[:, 5:7]
    return float((p[presence] - centers[presence]).square().mean())


def reference_position_curves():
    """POST-FREEZE ONLY. {"U" / "C*": {seed: #123 curves record}} (predicted-carrier position error of U / C*)."""
    return {label: {seed: read_json(rc.curve_path(HJ123, name, seed)) for seed in SEEDS}
            for label, name in (("U", "unit"), ("C*", "c_star"))}


def position_table(records_by_name, cells):
    """POST-FREEZE ONLY. {name: {"pred_<t>" / "floor_<t>": member-clustered bootstrap block | None}} over the
    records' "positions" {branch key: {str(t): {"pred", "floor"}}} (rc.position_table; a missing floor is None).
    records_by_name: {name: {seed: record}}; typed-failure records are skipped."""
    member_of = {item["key"]: member for member, item in position_branches(cells)}
    out = {}
    for name, records in records_by_name.items():
        row = {}
        for t in POSITION_TIMES:
            for kind in ("pred", "floor"):
                values, clusters = [], []
                for seed in SEEDS:
                    record = records[seed]
                    if record.get("typed_failure"):
                        continue
                    for key, curve in record["positions"].items():
                        value = curve[str(t)].get(kind)
                        if value is not None:
                            values.append(value)
                            clusters.append(member_of[key])
                row[f"{kind}_{t}"] = m99.bootstrap(values, clusters) if values else None
        out[name] = row
    return out


# ---------------------------------------------------------------------------
# P-novelty (owner decision D6; DESCRIPTIVE only, never in a disposition; evaluation data: post-freeze only).
# Population: the #110 issue110-evaluation levels of #107's frozen in-scope cells (both sides of every level pair),
# grid inventory (16 launches, mirrored on level 5 as #110 captured it), AUC on mixed-verdict levels only.
# ---------------------------------------------------------------------------

SCHEMA_NOVELTY_ANCHORS = "issue_127_novelty_anchors_v1"
SCHEMA_NOVELTY_REFERENCE = "issue_127_novelty_reference_v1"
NOVELTY_REFERENCES = (("U", "unit"), ("C*", "c_star"))


def _g107():
    from scripts import run_granularity_under_novelty as g107
    return g107


def novelty_cells():
    """POST-FREEZE ONLY (reads evaluation verdicts). {"levels": {level: {"cell", "pair", "pair_ordinal", "side",
    "novelty_level", "pair_level", "inventory": [{ordinal, action, key}], "verdicts": {ordinal: bool}, "anchor_key",
    "attempt"}}, "in_scope_cells": [...], "mixed": [levels]}: #107's frozen scope (in_scope_cells, levels_of over its
    groups L0, L1..L7; usable #110 branches), its anchor rule (the level's first usable grid candidate in suffix
    order) and its mixed rule. pair_level = the novelty level of the pair's novel side (also on the normal side)."""
    from scripts import run_novelty_coverage as nc
    g107 = _g107()
    scope = g107.in_scope_cells()
    levels = {}
    for group in g107.groups():
        if group == "n1":
            continue
        for level, meta in g107.levels_of(group, frozenset(scope)).items():
            levels[level] = {"cell": meta["cell"], "pair": meta["pair"], "pair_ordinal": meta["ordinal"],
                             "side": meta["side"], "novelty_level": meta["novelty_level"],
                             "pair_level": meta["pair_level"],
                             "inventory": [{"ordinal": i["ordinal"], "action": i["action"], "key": i["key"]}
                                           for i in meta["items"]],
                             "verdicts": dict(meta["verdicts"]), "anchor_key": meta["anchor_key"],
                             "attempt": str(nc.CAMPAIGN / nc.STREAM_EVAL / "attempts" / meta["anchor_key"])}
    return {"levels": levels, "in_scope_cells": list(scope),
            "mixed": sorted(level for level, meta in levels.items() if g107.mixed(meta))}


def novelty_anchors_path(output):
    return Path(output) / "data" / "anchors-novelty.pt"


def read_novelty_anchors(output, scoring_policy, cells=None):
    """POST-FREEZE ONLY; call inside harness.deterministic_scoring() (scoring_policy = its yielded policy). Recomputes
    every novelty anchor (no reuse of the #102 anchors-n2.pt subset, for uniformity): carriers = the E99 batch-1
    parse rt.anchor_carrier(s99.load_adapter(s99.OUTPUT, "E"), attempt) (the #102 parse_anchors pattern), frames =
    s99.image_tensor of the same shot frame 0, physics_paths = the attempt's decision-physics-v1.json. Writes
    output/data/anchors-novelty.pt; G-chronology reads read_at."""
    cells = novelty_cells() if cells is None else cells
    adapter = s99.load_adapter(s99.OUTPUT, "E")
    carriers, frames, physics, keys = {}, {}, {}, {}
    for level, meta in sorted(cells["levels"].items()):
        _, observation_root, shot, _ = s99.shot_frames(meta["attempt"])
        carriers[level] = rt.anchor_carrier(adapter, meta["attempt"])
        frames[level] = s99.image_tensor(observation_root / shot[0]["agent_observation"]["relative_path"])
        physics[level] = str(Path(meta["attempt"]) / "decision-physics-v1.json")
        keys[level] = meta["anchor_key"]
    payload = {"schema": SCHEMA_NOVELTY_ANCHORS, "carriers": carriers, "frames": frames, "physics_paths": physics,
               "keys": keys, "read_at": utc_now(), "scoring_policy": scoring_policy}
    atomic_torch(novelty_anchors_path(output), payload)
    return payload


def novelty_prior_rows(entry, stats):
    """P on a novelty level: gb.prior_rows with each candidate's launch replaced by the #109 grid launch of the same
    grid ordinal (identical to gb.prior_rows on unmirrored levels; on the mirrored level-5 grid the fit split never
    executed the mirrored launch, so the rule is applied to the launch of the same grid position)."""
    canonical = {item["ordinal"]: item["action"] for item in grid_items()}
    return gb.prior_rows({"inventory": [{**item, "action": canonical[item["ordinal"]]} for item in entry["inventory"]]},
                         stats)


def novelty_reference_dir(output):
    return Path(output) / "novelty" / "references"


def _novelty_reference_record(name, seed, request, scoring_policy, levels, anchors, costs, load):
    base = {"schema": SCHEMA_NOVELTY_REFERENCE, "reference": name, "seed": seed, "request": request,
            "scoring_policy": scoring_policy, "engine_seconds": 0}
    model, failure = load(name, seed)
    if model is None:
        return {**base, "typed_failure": failure, "checkpoint_sha256": None, "inventories": {"grid": {}}}
    grid = {level: {"rows": gb.request_block(model, request, anchors[level], meta["inventory"], costs)["rows"]}
            for level, meta in levels}
    return {**base, "typed_failure": None, "checkpoint_sha256": sha256_of(gb.member_path(name, seed)),
            "inventories": {"grid": grid}}


def novelty_reference_records(output, scoring_policy, cells=None, anchors=None, load=None):
    """POST-FREEZE ONLY; inside harness.deterministic_scoring(). U / C* per seed (the Gate-B handoff checkpoints via
    gb.load_model, rolled out exactly as Gate-B scores: gb.request_block from the E99 novelty anchor, tie-free cost)
    and P (novelty_prior_rows over gb.prior_stats of the retrain fit verdicts) on the mixed novelty levels. Written
    once to output/novelty/references/ (existing files are read back, never rewritten); no timestamps, so every
    child's files are byte-identical. Returns {"U" / "C*" / "P": ({seed: record}, "tie_free")} for novelty_table
    (P is seed-free: the same record per seed, the Gate-B cell weighting)."""
    from scripts import run_lookahead_control as wlc
    root = novelty_reference_dir(output)
    cells = novelty_cells() if cells is None else cells
    levels = [(level, cells["levels"][level]) for level in cells["mixed"]]
    if anchors is None:
        anchors = torch.load(novelty_anchors_path(output), map_location="cpu", weights_only=True)["carriers"]
    load = gb.load_model if load is None else load
    costs = harness.EndpointCosts(wlc.load_objective())
    out = {}
    for label, name in NOVELTY_REFERENCES:
        records = {}
        for seed in SEEDS:
            path = root / f"{name}--seed{seed}.json"
            if not path.is_file():
                write_json(path, _novelty_reference_record(name, seed, gb.UNITS[name]["request"], scoring_policy,
                                                           levels, anchors, costs, load))
            records[seed] = read_json(path)
        out[label] = (records, PRIMARY_COST)
    path = root / "prior.json"
    if not path.is_file():
        stats = gb.prior_stats(read_json(rt.verdicts_path(RETRAIN))["branches"], "grid")
        write_json(path, {"schema": SCHEMA_NOVELTY_REFERENCE, "reference": "prior", "typed_failure": None,
                          "inventories": {"grid": {level: {"rows": novelty_prior_rows(meta, stats)}
                                                   for level, meta in levels}}})
    prior = read_json(path)
    out["P"] = ({seed: prior for seed in SEEDS}, PRIMARY_COST)
    return out


def _unit(meta):
    return f"{meta['pair']}/{meta['pair_ordinal']}"


def novelty_table(records_by_name, cells):
    """POST-FREEZE ONLY; DESCRIPTIVE, no tokens. records_by_name: {name: ({seed: record}, row key)} (records hold
    "inventories": {"grid": {level: {"rows"}}}). Per ranker, over mixed levels x seeds (v1.cost_view):
    novel[k] (novel sides at novelty level k), normal_paired[k] (the normal sides paired with an in-population level-k
    novel side), novel_pooled (1-7), normal_paired_pooled, normal_all (every in-scope normal side, #107 L0) - each
    level-clustered (m99.block_of); paired[k] / paired_pooled: novel - paired normal AUC, one row per (level pair,
    seed) with both sides mixed, clustered by level pair (m99.paired). Bootstrap 10 000 draws, PCG64 7201."""
    levels = cells["levels"]
    novel_of = {_unit(m): level for level, m in levels.items() if m["side"] == "novel"}
    normal_of = {_unit(m): level for level, m in levels.items() if m["side"] == "normal"}
    mixed = set(cells["mixed"])
    ks = sorted({m["novelty_level"] for m in levels.values() if m["side"] == "novel"})
    views = {}
    for level in sorted(mixed):
        meta = levels[level]
        entry = {"inventory": meta["inventory"], "verdicts": meta["verdicts"]}
        for seed in SEEDS:
            views[(level, seed)] = {name: v1.cost_view(entry, _found(records[seed], "grid", level), key)
                                    for name, (records, key) in records_by_name.items()}

    def rows_of(selected):
        return [{"member": level, "seed": seed, "views": views[(level, seed)]}
                for level in sorted(selected & mixed) for seed in SEEDS]

    novel = {k: {lv for lv, m in levels.items() if m["side"] == "novel" and m["novelty_level"] == k} for k in ks}
    paired_normal = {k: {normal_of[_unit(levels[lv])] for lv in novel[k] if _unit(levels[lv]) in normal_of} for k in ks}
    all_normal = {lv for lv, m in levels.items() if m["side"] == "normal"}

    def pair_rows(selected):
        out = []
        for lv in sorted(selected):
            unit = _unit(levels[lv])
            partner = normal_of.get(unit)
            if lv not in mixed or partner not in mixed:
                continue
            for seed in SEEDS:
                out.append({"member": unit, "seed": seed,
                            "views": {(name, side): views[(level, seed)][name] for name in records_by_name
                                      for side, level in (("novel", lv), ("normal", partner))}})
        return out

    pooled_novel = set().union(*novel.values()) if ks else set()
    pooled_normal = set().union(*paired_normal.values()) if ks else set()
    groups = {"novel": {str(k): rows_of(novel[k]) for k in ks}, "normal_paired": {str(k): rows_of(paired_normal[k])
                                                                                 for k in ks}}
    pooled = {"novel_pooled": rows_of(pooled_novel), "normal_paired_pooled": rows_of(pooled_normal),
              "normal_all": rows_of(all_normal)}
    prows = {str(k): pair_rows(novel[k]) for k in ks}
    prows_pooled = pair_rows(pooled_novel)
    rankers = {}
    for name in records_by_name:
        block = {section: {k: m99.block_of(rows, name) for k, rows in by_k.items()} for section, by_k in groups.items()}
        block.update({section: m99.block_of(rows, name) for section, rows in pooled.items()})
        block["paired"] = {k: m99.paired(rows, [(1, (name, "novel")), (-1, (name, "normal"))])
                           for k, rows in prows.items()}
        block["paired_pooled"] = m99.paired(prows_pooled, [(1, (name, "novel")), (-1, (name, "normal"))])
        rankers[name] = block
    return {"label": "DESCRIPTIVE (P-novelty; no tokens, never in a disposition)", "novelty_levels": ks,
            "population": {"levels": len(levels), "mixed_levels": len(mixed), "in_scope_cells": len(cells["in_scope_cells"]),
                           "level_pairs": len(novel_of), "novel_mixed": {str(k): len(novel[k] & mixed) for k in ks},
                           "paired_units": {k: len(rows) // len(SEEDS) for k, rows in prows.items()}},
            "rankers": rankers}


NOVELTY_TABLE_HEADER = ("ranker", "novelty level", "novel AUC [CI]", "paired normal AUC [CI]",
                        "novel − normal [CI]", "paired level pairs")


def format_novelty_rows(table):
    """Markdown lines of the P-novelty table: per ranker one row per novelty level, the pooled 1-7 row and the
    all-normal (L0) row (rc.fmt_auc / rc.fmt_block)."""
    lines = ["| " + " | ".join(NOVELTY_TABLE_HEADER) + " |",
             "| " + " | ".join("---" for _ in NOVELTY_TABLE_HEADER) + " |"]
    for name, block in table["rankers"].items():
        entries = [(str(k), block["novel"][str(k)], block["normal_paired"][str(k)], block["paired"][str(k)])
                   for k in table["novelty_levels"]]
        entries.append((f"{table['novelty_levels'][0]}–{table['novelty_levels'][-1]} pooled" if table["novelty_levels"]
                        else "pooled", block["novel_pooled"], block["normal_paired_pooled"], block["paired_pooled"]))
        for label, novel, normal, paired in entries:
            lines.append("| " + " | ".join((name, label, rc.fmt_auc(novel), rc.fmt_auc(normal),
                                            rc.fmt_block(paired["estimate"]), str(paired["members"]))) + " |")
        lines.append("| " + " | ".join((name, "0 (all normal sides)", "n/a", rc.fmt_auc(block["normal_all"]), "n/a",
                                        "n/a")) + " |")
    return lines


# ---------------------------------------------------------------------------
# statistics, tokens, contrasts, dispositions
# ---------------------------------------------------------------------------

def references():
    """{"U" / "C*": {seed: Gate-B record}, "J" / "J_dec": {seed: #101 record}, "prior": {inventory: gb.prior_stats
    over the retrain fit-verdicts.json}}."""
    out = {"U": {}, "C*": {}, "J": {}, "J_dec": {}}
    for seed in SEEDS:
        out["U"][seed] = rc.gate_b_rows("unit", seed)
        out["C*"][seed] = rc.gate_b_rows("c_star", seed)
        for label, arm in J_ARMS.items():
            out[label][seed] = read_json(ISSUE_101 / "records" / f"{arm}--seed{seed}.json")
    fit = read_json(rt.verdicts_path(RETRAIN))["branches"]
    out["prior"] = {inventory: gb.prior_stats(fit, inventory) for inventory in INVENTORIES}
    return out


def _found(record, inventory, member):
    return None if record.get("typed_failure") else record["inventories"][inventory][member]["rows"]


def view_rows(cells, inventory, arms, references):
    """One row per (member, seed) of the inventory's mixed cells; views U, C*, P, J, J_dec (tie-free) and every
    arm view. arms: {view_name: ({seed: record}, row key)} (rc.view_rows, parameterised)."""
    rows = []
    for member, entry in sorted(cells[inventory]["mixed"].items()):
        prior = m99.view(entry, gb.prior_rows(entry, references["prior"][inventory]), PRIMARY_COST)
        for seed in SEEDS:
            views = {"P": prior}
            for label in ("U", "C*", "J", "J_dec"):
                views[label] = v1.cost_view(entry, _found(references[label][seed], inventory, member), PRIMARY_COST)
            for name, (records, key) in arms.items():
                views[name] = v1.cost_view(entry, _found(records[seed], inventory, member), key)
            rows.append({"member": member, "seed": seed, "views": views})
    return rows


def ranker_table(rows, names):
    """Per named ranker: pooled AUC block (v1.ranker_block), per-seed blocks (None for the seed-free P), top-1 rate
    and chance top-1 (the #123 compute.json ranker block)."""
    out = {"cells": len(rows), "members": len({r["member"] for r in rows}), "rankers": {}}
    for name in names:
        ranker = v1.ranker_block(rows, name)
        ranker["per_seed"] = ({str(s): m99.block_of([r for r in rows if r["seed"] == s], name) for s in SEEDS}
                              if name != "P" else None)
        ranker["top1_rate"] = ranker["top1_hits"] / ranker["cells"] if ranker["cells"] else None
        chances = [r["views"][name]["chance_top1"] for r in rows if r["views"][name]["chance_top1"] is not None]
        ranker["chance_top1"] = float(np.mean(chances)) if chances else None
        out["rankers"][name] = ranker
    return out


def contrast(rows, first, second, guarded):
    """first - second (m99.paired); token / reverse_token by c2.token when guarded, else the stop token; an absent
    estimate is the stop token (rc.grid_contrasts per contrast)."""
    paired = m99.paired(rows, [(1, first), (-1, second)])
    forward, backward = rc.token_pair(paired["estimate"], guarded)
    return {"first": first, "second": second, **paired, "token": forward, "reverse_token": backward,
            "guards_pass": bool(guarded)}


def _spearman_rows(a, b, key):
    """c2.spearman_rows under any row key."""
    x = {r["ordinal"]: r[key] for r in a if not r["excluded"]}
    y = {r["ordinal"]: r[key] for r in b if not r["excluded"]}
    common = sorted(x.keys() & y.keys())
    return v1.spearman([x[o] for o in common], [y[o] for o in common])


def wrong_anchor_table(cells, records, arm, key, guarded):
    """Right - wrong anchor (480 fit anchors) of one arm on the grid mixed cells, as #123 S10 (rc.wrong_anchor_table
    for one arm, any row key). records: {seed: record} with "wrong_anchor": {fit member: {"rows"}}."""
    wrows = []
    for member, entry in sorted(cells["grid"]["mixed"].items()):
        for seed in SEEDS:
            record = records[seed]
            if record.get("typed_failure"):
                wrows.append({"member": member, "seed": seed, "views": {arm: None}})
                continue
            right = record["inventories"]["grid"][member]["rows"]
            others = [record["wrong_anchor"][a]["rows"] for a in sorted(record["wrong_anchor"])]
            aucs = [a for a in (m99.view(entry, w, key)["auc"] for w in others) if a is not None]
            rhos = [r for r in (_spearman_rows(right, w, key) for w in others) if r is not None]
            wrows.append({"member": member, "seed": seed, "views": {arm: {
                "right": m99.view(entry, right, key)["auc"], "wrong": float(np.mean(aucs)) if aucs else None,
                "stability": float(np.mean(rhos)) if rhos else None}}})
    diff = c2.difference_block(wrows, [(1, arm, "right"), (-1, arm, "wrong")])
    return {"right": c2.field_block(wrows, arm, "right"), "wrong": c2.field_block(wrows, arm, "wrong"),
            "right_minus_wrong": diff, "stability": c2.field_block(wrows, arm, "stability"),
            "token": c2.token(diff) if guarded else STOP_TOKEN, "guards_pass": bool(guarded),
            "wrong_anchors": len(records[SEEDS[0]].get("wrong_anchor") or {})}


def disposition(xu, guards_pass):
    """xu: the X - U contrast (contrast(rows, X, "U", ...)). 1 iff X - U supported and the guards pass; 2 iff U - X
    supported and the guards pass; 3 iff the guards pass and the X - U interval lies inside (-delta, +delta);
    otherwise 4 (per-contrast tokens only). The tokens are recomputed from the estimate (c2.token)."""
    block = xu["estimate"]
    n = 4
    if guards_pass and block is not None:
        low, high = block["interval"]
        if c2.token(block) == "supported":
            n = 1
        elif c2.token(rc.reverse(block)) == "supported":
            n = 2
        elif -DELTA < low and high < DELTA:
            n = 3
    return {"disposition": n, "label": DISPOSITION_LABELS[n], "guards_pass": bool(guards_pass), "x_minus_u": block}


def program_claim(per_arm):
    """Intersection-union over PROGRAM_SET. per_arm: {arm: {"disposition": int, "guards_pass": bool,
    "gfid_pass": bool, "dropped": typed reason | None}}. Holds iff every member is present, not dropped, passes its
    guards and G-fid and reaches disposition 2; every other case vetoes it (reasons listed)."""
    vetoes = []
    for arm in PROGRAM_SET:
        entry = per_arm.get(arm)
        if entry is None:
            vetoes.append({"arm": arm, "reason": "not_run"})
            continue
        if entry.get("dropped"):
            vetoes.append({"arm": arm, "reason": f"dropped: {entry['dropped']}"})
        if entry.get("guards_pass") is not True:
            vetoes.append({"arm": arm, "reason": "guard_failed"})
        if entry.get("gfid_pass") is not True:
            vetoes.append({"arm": arm, "reason": "g_fid_failed"})
        if entry.get("disposition") != 2:
            vetoes.append({"arm": arm, "reason": f"disposition_{entry.get('disposition')}"})
    return {"members": list(PROGRAM_SET), "holds": not vetoes, "vetoes": vetoes,
            "disposition_1_arms": [a for a in PROGRAM_SET if (per_arm.get(a) or {}).get("disposition") == 1],
            "claim": "U ranks launches better than every baseline in this set on the sealed NovPhy grid"}


def budget_safeguard(xu_token, x_training_gpu_s, u_training_gpu_s):
    """The #123 U-B3 rule: fires iff X - U is supported and X's training compute exceeds U's B1 training compute
    (budget-qualified wording; U-native runs before any architectural reading). x None (not trained) never fires."""
    fires = bool(xu_token == "supported" and x_training_gpu_s is not None and x_training_gpu_s > u_training_gpu_s)
    ratio = x_training_gpu_s / u_training_gpu_s if x_training_gpu_s is not None and u_training_gpu_s else None
    return {"fires": fires, "ratio": ratio, "x_training_gpu_s": x_training_gpu_s,
            "u_training_gpu_s": u_training_gpu_s,
            "wording": (f"X at its native budget ({ratio:.2f} x U's training GPU-h) ranks better than U at B1"
                        if fires else None)}


# ---------------------------------------------------------------------------
# per-arm publication block and the child summary (every child publishes the same fields)
# ---------------------------------------------------------------------------

CHILD_SUMMARY_SCHEMA = "issue_127_child_summary_v1"
CONTRAST_REFERENCES = {"X-U": "U", "X-C*": "C*", "X-P": "P", "X-J_dec": "J_dec"}
COMPUTE_FIELDS = ("trainable_parameters", "training_gpu_seconds", "macs_per_candidate", "wall_per_decision")
ARM_BLOCK_FIELDS = ("view", "role", "program_set", "primary_cost_key", "guards", "global_guards_pass", "guards_pass",
                    "typed_failures", "nonfinite_share", "ranker", "contrasts", "disposition", "budget_safeguard",
                    "compute", "dropped")


def arm_block(rows, view, arm_records, key, guards, global_guards_pass, compute, role, program_set, extra_guards=None,
              dropped=None):
    """The standard per-arm block. rows: view_rows(...) with the arm under `view`; arm_records: {seed: record};
    key: the arm's primary row key; guards: arm_guard_block(...) (or any dict of G_* booleans); extra_guards: further
    G_* booleans (e.g. G_port, G_fid). The arm passes iff every G_* entry is True; every contrast (X - U, X - C*,
    X - P, X - J_dec) is guarded by the global guards AND the arm's; the disposition is read from X - U. compute:
    {"trainable_parameters", "training_gpu_seconds": {seed: s}, "macs_per_candidate", "wall_per_decision"}; the
    budget safeguard compares X's mean per-seed training GPU-s with U's (the #112 v3 retrain reports). dropped:
    typed reason or None."""
    missing = [f for f in COMPUTE_FIELDS if f not in compute]
    if missing:
        raise ValueError(f"compute block lacks {missing}")
    merged = {**guards, **(extra_guards or {})}
    names = [k for k in merged if k.startswith("G_")]
    arm_pass = bool(names) and all(merged[k] is True for k in names)
    guarded = bool(global_guards_pass) and arm_pass
    contrasts = {label: contrast(rows, view, reference, guarded) for label, reference in CONTRAST_REFERENCES.items()}
    seconds = [float(s) for s in compute["training_gpu_seconds"].values()]
    u_seconds = float(np.mean([s["training_gpu_seconds"] for s in _reference_training("unit").values()]))
    return {"view": view, "role": role, "program_set": bool(program_set), "primary_cost_key": key,
            "guards": {**merged, "pass": arm_pass}, "global_guards_pass": bool(global_guards_pass),
            "guards_pass": guarded,
            "typed_failures": {str(seed): record.get("typed_failure") for seed, record in arm_records.items()},
            "nonfinite_share": nonfinite_share(arm_records.values(), key),
            "ranker": ranker_table(rows, [view])["rankers"][view], "contrasts": contrasts,
            "disposition": disposition(contrasts["X-U"], guarded),
            "budget_safeguard": budget_safeguard(contrasts["X-U"]["token"], float(np.mean(seconds)) if seconds else None,
                                                 u_seconds),
            "compute": compute, "dropped": dropped}


def within_family(rows, first, second, guarded):
    """A within-family contrast (e.g. A2 - A1): contrast(...) with its label."""
    return {"label": f"{first} - {second}", **contrast(rows, first, second, guarded)}


def child_summary(issue, identity, frozen_at, arms_blocks, within_family, global_guards, gfid, extra,
                  engine_seconds=0):
    """The summary.json every child writes. arms_blocks: {view: arm_block}; within_family: {label: within_family
    block}; global_guards: {name: {"pass": bool, ...}}; gfid: {view: {"pass": bool, ...}} per arm. A program-set
    arm's view must be its PROGRAM_SET name; program_claim_entries feed program_claim()."""
    global_pass = all(g["pass"] is True for g in global_guards.values())
    for view, block in arms_blocks.items():
        if block["view"] != view or block["global_guards_pass"] != global_pass:
            raise ValueError(f"arm block {view} is not bound to this child's view name / global guards")
        if block["program_set"] and view not in PROGRAM_SET:
            raise ValueError(f"program-set arm {view} is not named as in PROGRAM_SET")
    members = [view for view, block in arms_blocks.items() if block["program_set"]]
    entries = {view: {"disposition": arms_blocks[view]["disposition"]["disposition"],
                      "guards_pass": arms_blocks[view]["guards_pass"],
                      "gfid_pass": (gfid.get(view) or {}).get("pass") is True, "dropped": arms_blocks[view]["dropped"]}
               for view in members}
    return {"schema": CHILD_SUMMARY_SCHEMA, "program_issue": PROGRAM_ISSUE, "issue": issue, "identity": identity,
            "frozen_at": frozen_at, "global_guards": global_guards, "global_guards_pass": global_pass, "gfid": gfid,
            "arms": arms_blocks, "within_family": within_family, "program_set_members": members,
            "program_claim_entries": entries, "engine_seconds": engine_seconds, "extra": extra}


def load_child_summary(path):
    """A child's summary.json, validated against CHILD_SUMMARY_SCHEMA."""
    summary = read_json(path)
    if summary.get("schema") != CHILD_SUMMARY_SCHEMA or summary.get("program_issue") != PROGRAM_ISSUE:
        raise ValueError(f"{path} is not an {CHILD_SUMMARY_SCHEMA} summary")
    required = ("issue", "identity", "frozen_at", "global_guards", "global_guards_pass", "gfid", "arms",
                "within_family", "program_set_members", "program_claim_entries", "engine_seconds", "extra")
    lacking = [k for k in required if k not in summary]
    lacking += [f"arms.{v}.{f}" for v, b in summary.get("arms", {}).items() for f in ARM_BLOCK_FIELDS if f not in b]
    if lacking:
        raise ValueError(f"{path} lacks {lacking}")
    if sorted(summary["program_claim_entries"]) != sorted(summary["program_set_members"]):
        raise ValueError(f"{path}: program claim entries and program-set members differ")
    return summary


ARM_TABLE_HEADER = ("arm", "role", "params", "guards", "grid AUC [CI]", "X − U [CI]", "token X−U", "token U−X",
                    "disposition")


def format_arm_rows(blocks):
    """Markdown lines of the per-arm table (header, separator, one row per arm block) with rc.fmt_auc /
    rc.fmt_block; blocks: {view: arm_block} or a sequence of arm blocks."""
    lines = ["| " + " | ".join(ARM_TABLE_HEADER) + " |", "| " + " | ".join("---" for _ in ARM_TABLE_HEADER) + " |"]
    for block in blocks.values() if isinstance(blocks, dict) else blocks:
        failed = [k for k, v in block["guards"].items() if k.startswith("G_") and v is not True]
        guards = "pass" if block["guards"]["pass"] else "fail" + (f" ({', '.join(failed)})" if failed else "")
        if not block["global_guards_pass"]:
            guards += "; global fail"
        xu = block["contrasts"]["X-U"]
        disp = str(block["disposition"]["disposition"]) + (f" (dropped: {block['dropped']})" if block["dropped"] else "")
        role = block["role"] + (" (program set)" if block["program_set"] else "")
        params = block["compute"]["trainable_parameters"]
        cells = (block["view"], role, "n/a" if params is None else f"{params:,}", guards,
                 rc.fmt_auc(block["ranker"]["auc"]), rc.fmt_block(xu["estimate"]), xu["token"], xu["reverse_token"],
                 disp)
        lines.append("| " + " | ".join(cells) + " |")
    return lines


# ---------------------------------------------------------------------------
# compute ledger
# ---------------------------------------------------------------------------

mark = rc.mark  # output/ledger/<phase>.json, wall / GPU seconds accumulated


def ledger_totals(output, report_of):
    """Ledger phases with their GPU / wall seconds and the two totals (rc.ledger_totals, A3 rule). report_of(phase)
    -> path of the arm-seed's training report.json for a training phase, else None; an existing report's seconds
    replace the phase's ledger seconds (kept as ledger_gpu_seconds / ledger_wall_seconds when they differ)."""
    entries = {}
    ledger = rc.ledger_dir(output)
    for path in sorted(ledger.glob("*.json")) if ledger.is_dir() else ():
        entry = read_json(path)
        report = report_of(path.stem)
        if report is not None and Path(report).is_file():
            once = read_json(report)
            if (once["gpu_seconds"], once["wall_seconds"]) != (entry["gpu_seconds"], entry["wall_seconds"]):
                entry = {**entry, "gpu_seconds": once["gpu_seconds"], "wall_seconds": once["wall_seconds"],
                         "ledger_gpu_seconds": entry["gpu_seconds"], "ledger_wall_seconds": entry["wall_seconds"]}
        entries[path.stem] = entry
    return entries, sum(e["gpu_seconds"] for e in entries.values()), sum(e["wall_seconds"] for e in entries.values())


def macs_per_candidate(fn, *inputs, candidates):
    """MACs per candidate of fn(*inputs) scoring `candidates` candidates: FlopCounterMode FLOPs / 2 / candidates
    (matmul / conv / attention ops; elementwise work is not counted)."""
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        fn(*inputs)
    return counter.get_total_flops() / 2 / candidates


def _synchronize():
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def wall_per_decision(fn, decisions):
    """Median seconds of fn(decision) over decisions (one decision = one anchor's 16 grid candidates), CUDA
    synchronised before and after each call."""
    seconds = []
    for decision in decisions:
        _synchronize()
        began = time.perf_counter()
        fn(decision)
        _synchronize()
        seconds.append(time.perf_counter() - began)
    return float(np.median(seconds))


def _reference_predictor(name, seed):
    """The Gate-B handoff checkpoint of U ("unit") / C* ("c_star") on the CPU (rt load_member's binding checks)."""
    entry = next(c for c in gb.checkpoints(name) if c["seed"] == seed)
    path = gb.member_path(name, seed)
    if not entry["usable"] or sha256_of(path) != entry["sha256"]:
        raise ValueError(f"{name} seed {seed}: checkpoint unusable or differs from the handoff hash")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    family = gb.UNITS[name]["family"]
    if (payload["arm"], payload["family"], payload["fold"], payload["seed"]) != (rt.ARM, family, "all", seed):
        raise ValueError(f"predictor {name}/{seed} binding differs")
    model = s99.new_predictor(family)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval()


def _reference_rollout(model, pair, z0, actions):
    """The fixed-pair rollout of m99.rollout_endpoints (launch at the anchor, steps of pair.delta to frame 225)."""
    z = z0[None].expand(len(actions), -1).clone()
    position = 0
    while position < m99.ENDPOINT:
        z = model.carrier(z, actions, pair)
        position += pair.delta
    return z


def _reference_training(name):
    """{seed: {"training_gpu_seconds", "steps", "parameters_per_member", "report"}} of U ("unit") / C* ("c_star")
    from the #112 v3 retrain training-group reports (one complete member per seed)."""
    impl = rt.implementation()
    family = gb.UNITS[name]["family"]
    seeds = {}
    for index, seed in enumerate(SEEDS):
        path = impl.group_path(RETRAIN, rt.ARM, family, index)
        report = read_json(path)
        if [m["seed"] for m in report["members"]] != [seed] or not report["members"][0]["complete"]:
            raise ValueError(f"{path} is not the complete seed-{seed} group")
        seeds[str(seed)] = {"training_gpu_seconds": report["gpu_seconds"], "steps": report["steps"],
                            "parameters_per_member": report["parameters_per_member"],
                            "report": str(path.relative_to(ROOT))}
    return seeds


def reference_compute():
    """U / C*: trainable parameters, per-seed training GPU / wall seconds (the #112 v3 retrain training-group
    reports, one per seed), and inference MACs per candidate (CPU FlopCounterMode over the rollout from a fit
    anchor to frame 225 for the 16 grid candidates; m99's explicit linear-MAC count beside it)."""
    from scripts import run_tau_ad_within_checkpoint as t96
    from world_model.training.matched_dynamics import linear_macs, pairs_for
    anchors = fit_anchors()["carriers"]
    z0 = anchors[sorted(anchors)[0]]
    actions = torch.stack([launch5(item["action"]) for item in grid_items()])
    out = {}
    for label, name in (("U", "unit"), ("C*", "c_star")):
        request = gb.UNITS[name]["request"]
        seeds = _reference_training(name)
        model = _reference_predictor(name, SELECTION_SEED)
        pair = pairs_for(model)[t96.arm_spec(request)["pair_index"]]
        steps = math.ceil(m99.ENDPOINT / pair.delta)
        out[label] = {"name": name, "request": request, "model_class": type(model).__name__,
                      "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                      "training": seeds,
                      "training_gpu_seconds_mean": float(np.mean([s["training_gpu_seconds"] for s in seeds.values()])),
                      "rollout": {"delta": pair.delta, "steps": steps, "endpoint": m99.ENDPOINT},
                      "inference_macs_per_candidate": macs_per_candidate(
                          _reference_rollout, model, pair, z0, actions, candidates=len(actions)),
                      "linear_macs_per_candidate": linear_macs(model, pair) * steps}
    return out


# ---------------------------------------------------------------------------
# records, manifest, validation (G-records; #123 A2 semantics, parameterised identity and record patterns)
# ---------------------------------------------------------------------------

def records_manifest(output, identity, patterns):
    """Content hashes of every record matching the glob patterns (relative to output), in pattern order."""
    output = Path(output)
    files = [p for pattern in patterns for p in sorted(output.glob(pattern))]
    return {"schema": SCHEMA_RECORDS_MANIFEST, "plan_identity": identity, "patterns": list(patterns),
            "records": [{"artifact": str(p.relative_to(ROOT)), "sha256": sha256_of(p), "bytes": p.stat().st_size}
                        for p in files]}


def write_receipt(output, identity, spot, error=None):
    """The --validate receipt, bound to the sha256 of the records-manifest.json it checked."""
    output = Path(output)
    manifest_path = output / "records-manifest.json"
    write_json(output / VALIDATION_RECEIPT, {
        "schema": SCHEMA_RECEIPT, "plan_identity": identity,
        "records_manifest_sha256": sha256_of(manifest_path) if manifest_path.is_file() else None,
        "spot_check": spot, "error": error, "validated_at": utc_now(),
        "pass": bool(error is None and spot is not None and spot["pass"])})


def records_guard(output, identity):
    """G-records: passes only if the --validate receipt binds the current records-manifest.json (by sha256), that
    manifest lists exactly the current records (its own patterns), and the receipt's re-score replicated."""
    output = Path(output)
    manifest_path, receipt_path = output / "records-manifest.json", output / VALIDATION_RECEIPT
    if not manifest_path.is_file() or not receipt_path.is_file():
        return {"pass": False, "receipt": None}
    receipt = read_json(receipt_path)
    manifest = read_json(manifest_path)
    bound = bool(receipt.get("plan_identity") == identity
                 and receipt.get("records_manifest_sha256") == sha256_of(manifest_path)
                 and manifest == records_manifest(output, identity, manifest.get("patterns", [])))
    spot = receipt.get("spot_check")
    return {"pass": bool(bound and receipt.get("pass") is True and spot is not None and spot.get("pass") is True),
            "receipt": VALIDATION_RECEIPT, "bound_to_current_records": bound,
            "records_manifest_sha256": receipt.get("records_manifest_sha256"), "spot_check": spot}


# ---------------------------------------------------------------------------
# pins
# ---------------------------------------------------------------------------

def _input_paths():
    pixels = {split: rc.pixel_dir(HJ123, split) for split in ("fit", "policy")}
    names = {
        "module": ROOT / MODULE,
        "scoring_harness": ROOT / "world_model/planning/scoring_harness.py",
        "issue_123_runner": ROOT / rc.RUNNER, "retrain_runner": ROOT / rt.RUNNER, "v1_runner": ROOT / v1.RUNNER,
        "gate_b_runner": ROOT / gb.RUNNER, "chain_attribution_runner": ROOT / "scripts/run_decision_chain_attribution.py",
        "controls_runner": ROOT / "scripts/run_decision_dynamics_controls.py",
        "slot_encoder_fix_runner": ROOT / "scripts/run_slot_encoder_fix.py",
        "hjepa_module": ROOT / "world_model/hjepa/novphy.py",
        # the modules that define the pre-registered cell AUC and bootstrap (p94.cell_auc / p94.bootstrap,
        # 10 000 draws, PCG64 7201, reached through t96 -> m99 / v1 / c2) and U / C*'s reference rollout
        "bootstrap_runner": ROOT / "scripts/run_launch_power_probe.py",
        "tau_runner": ROOT / "scripts/run_tau_ad_within_checkpoint.py",
        "matched_dynamics_module": ROOT / "world_model/training/matched_dynamics.py",
    }
    for split, base in pixels.items():
        names[f"issue_123_pixels_{split}_index"] = base / "index.npz"
        names[f"issue_123_pixels_{split}_manifest"] = base / "MANIFEST.json"
        for path in sorted(base.glob("shard-*.jsonl")):
            names[f"issue_123_pixels_{split}_{path.stem}"] = path
    carriers = rc.carriers_dir(HJ123)
    names.update({
        "issue_123_fit_index": carriers / "fit-index.npz", "issue_123_fit_carriers": carriers / "fit-carriers.bin",
        "issue_123_policy_data": rc.policy_data_path(HJ123),
        "retrain_fit_anchors": rt.anchors_path(RETRAIN), "retrain_fit_verdicts": rt.verdicts_path(RETRAIN),
        "retrain_handoff": gb.HANDOFF, "cohort_plan": rt.COHORT, "power_plan": rt.POWER_PLAN,
        "campaign_branches": rt.CAMPAIGN / "branches.json",
        "gate_b_plan": GATE_B / "plan.json", "gate_b_summary": GATE_B / "summary.json",
        "gate_b_anchors": GATE_B / "data" / "anchors-evaluation.pt",
        "issue_101_plan": ISSUE_101 / "plan.json", "issue_99_encoder": s99.OUTPUT / "encoder" / "encoder.pt",
    })
    impl = rt.implementation()
    for name in ("unit", "c_star"):
        family = gb.UNITS[name]["family"]
        for index, seed in enumerate(SEEDS):
            names[f"gate_b_record_{name}_{seed}"] = GATE_B / "records" / f"{name}--seed{seed}.json"
            names[f"retrain_training_{family}_{seed}"] = impl.group_path(RETRAIN, rt.ARM, family, index)
        names[f"retrain_predictor_{name}_{SELECTION_SEED}"] = gb.member_path(name, SELECTION_SEED)
    for label, arm in J_ARMS.items():
        for seed in SEEDS:
            names[f"issue_101_record_{label}_{seed}"] = ISSUE_101 / "records" / f"{arm}--seed{seed}.json"
    names["issue_123_eval_positions"] = rc.eval_cache_path(HJ123)
    for name in ("unit", "c_star"):
        for seed in SEEDS:
            names[f"issue_123_curves_{name}_{seed}"] = rc.curve_path(HJ123, name, seed)
    from scripts import run_novelty_coverage as nc
    g107 = _g107()
    names.update({
        "issue_107_runner": ROOT / g107.RUNNER, "issue_110_runner": ROOT / "scripts/run_novelty_coverage.py",
        "issue_110_novelty_contract": ROOT / "world_model/data/novelty_contract.py",
        "issue_110_novelty_labels_v2": ROOT / "world_model/data/novelty_labels_v2.py",
        "issue_110_handoff_107": g107.HANDOFF, "issue_110_design_plan": nc.DESIGN / "plan.json",
        "issue_110_label_failures": nc.OUTPUT / "labels" / "failures.json",
    })
    for product in ("fit-anchor-frames.pt", "policy-anchor-frames.pt", "MANIFEST.json"):
        names[f"shared_{product}"] = SHARED / product
    return names


def shared_inputs():
    """[{name, artifact, sha256}] of this module and every upstream file it reads (hashing the Gate-B anchor store
    reads no content)."""
    return [{"name": name, "artifact": str(Path(path).relative_to(ROOT)), "sha256": sha256_of(path)}
            for name, path in _input_paths().items()]


def verify_pins(entries):
    """Raise on drift of any pinned input unless post_freeze_amendments.matches accepts the change."""
    from scripts import post_freeze_amendments as amendments
    for entry in entries:
        actual = sha256_of(ROOT / entry["artifact"])
        if not amendments.matches(entry["artifact"], entry["sha256"], actual):
            raise ValueError(f"frozen input {entry['artifact']} changed after the freeze")


# ---------------------------------------------------------------------------
# shared data products (CPU)
# ---------------------------------------------------------------------------

def build_shared():
    """Frame 0 of every fit / policy anchor branch from the #123 pixel stores, re-read and compared with a fresh
    rc.FitData / rc.PixelStore; MANIFEST.json with sha256, bytes, built_at and the source pins."""
    began = time.monotonic()
    shots = FitShots()
    fit = fit_anchors()["keys"]
    frames = {member: shots.frame(shots.row_of[key], 0).contiguous() for member, key in sorted(fit.items())}
    atomic_torch(SHARED / "fit-anchor-frames.pt", {"schema": SCHEMA_FIT_FRAMES, "keys": dict(sorted(fit.items())),
                                                   "frames": frames})
    policy_keys = policy_anchors()["keys"]
    store = rc.PixelStore(HJ123, "policy")
    pframes = {member: store.frame(store.row[key], 0).contiguous() for member, key in sorted(policy_keys.items())}
    atomic_torch(SHARED / "policy-anchor-frames.pt", {"schema": SCHEMA_POLICY_FRAMES,
                                                      "keys": dict(sorted(policy_keys.items())), "frames": pframes})
    fresh, fresh_store = rc.FitData(HJ123), rc.PixelStore(HJ123, "policy")
    fresh_row = {key: i for i, key in enumerate(fresh.keys)}
    saved, psaved = fit_anchor_frames(), policy_anchor_frames()
    if sorted(saved) != sorted(fit) or len(saved) != 480:
        raise ValueError("fit anchor frames: member set differs")
    if sorted(psaved) != sorted(policy_keys) or len(psaved) != 120:
        raise ValueError("policy anchor frames: member set differs")
    for member, key in fit.items():
        if not torch.equal(saved[member], fresh.frame(fresh_row[key], 0)):
            raise ValueError(f"fit anchor frame of {member} differs from rc.FitData.frame")
    for member, key in policy_keys.items():
        if not torch.equal(psaved[member], fresh_store.frame(fresh_store.row[key], 0)):
            raise ValueError(f"policy anchor frame of {member} differs from the #123 policy pixel store")
    sources = {"retrain_fit_anchors": rt.anchors_path(RETRAIN), "issue_123_policy_data": rc.policy_data_path(HJ123),
               "issue_123_fit_index": rc.carriers_dir(HJ123) / "fit-index.npz"}
    for split in ("fit", "policy"):
        sources[f"issue_123_pixels_{split}_index"] = rc.pixel_dir(HJ123, split) / "index.npz"
        sources[f"issue_123_pixels_{split}_manifest"] = rc.pixel_dir(HJ123, split) / "MANIFEST.json"
    products = []
    for name, count in (("fit-anchor-frames.pt", len(saved)), ("policy-anchor-frames.pt", len(psaved))):
        path = SHARED / name
        products.append({"artifact": str(path.relative_to(ROOT)), "sha256": sha256_of(path),
                         "bytes": path.stat().st_size, "anchors": count, "shape": [3, 240, 320], "dtype": "uint8"})
    write_json(SHARED / "MANIFEST.json", {
        "schema": SCHEMA_SHARED_MANIFEST, "built_at": utc_now(), "builder": MODULE,
        "builder_sha256": sha256_of(ROOT / MODULE), "products": products,
        "rule": ("frame 0 of each anchor branch (fit: rt.anchors_path frames; policy: #123 policy.pt anchor_frames) "
                 "from the #123 pixel stores; every saved frame equals a fresh rc.FitData / rc.PixelStore read"),
        "sources": [{"name": n, "artifact": str(p.relative_to(ROOT)), "sha256": sha256_of(p),
                     "bytes": p.stat().st_size} for n, p in sources.items()]})
    log(PROGRAM_ISSUE, f"shared products: {len(saved)} fit + {len(psaved)} policy anchor frames "
                       f"({time.monotonic() - began:.0f}s)")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--build-shared", action="store_true", help="write the SHARED data products (CPU)")
    modes.add_argument("--reference-compute", action="store_true", help="print reference_compute() (CPU)")
    args = parser.parse_args()
    if args.build_shared:
        return build_shared()
    print(json_text(reference_compute()), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
