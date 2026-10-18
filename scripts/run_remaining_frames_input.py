"""Issue-119: remaining-frames input saturation of the request policy, scored once on the #104 sealed #109
evaluation split.

Binding runner module: scripts/run_remaining_frames_input.py
Exact validation command: python -u -m scripts.run_remaining_frames_input --validate

The request policy reads ``min(kappa, 60) / 60``, so on a 225-frame evaluation rollout its time input is
1.0 for the first 165 frames, while its DP labels see the window end. This runner

1. trains the #101 J recipe (prediction-error DP, lambda 1e-4, the #109 policy-split pool of #101, LC
   hybrid Gate-B predictors, seeds 20260908/09/10, 2 DAgger rounds x 1800 steps, the 242 -> 128 -> 9
   MLP and its initialization) with four frozen input variants (``world_model/training/
   remaining_input.py``): (a) min(kappa, 60)/60 (control; must reproduce the #101 J bit for bit),
   (b) min(kappa, 225)/225, (c) log1p(kappa)/log1p(225), (d) (a) plus the elapsed frames min(k, 225)/225;
2. crosses the input with the DP label window as a separate pre-declared 2 x 2 factor: {a, b} x {first
   60-frame window (#101), start-0 window reaching the 225-frame endpoint};
3. selects v* in {b, c, d} on the policy split (seed-mean pooled policy-split AUC, training data) and
   repeats it once under the #101 decision-valued labels (dec, lambda 1e-1): J_dec(v*);
4. reports label accuracy, the request mix against the true remaining frames, the per-inventory AUC of
   every variant, the nine fixed requests (#101 records), F_hindsight, the no-model action prior and
   each new policy's wrong-anchor control, MACs, and the pre-declared disposition.

Modes: --dry-run --smoke --prepare --assemble --train --select --train-dec --accuracy --score --publish
--validate. --train, --train-dec, --accuracy and --score take --workers N --worker K (one process per GPU;
CUDA_VISIBLE_DEVICES set by the caller). Zero engine seconds. Intervals: member (level)-clustered
percentile bootstrap (10000 draws, PCG64 7201), DESCRIPTIVE.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from scripts import post_freeze_amendments as amendments
from scripts import run_decision_aware_policy as dap
from scripts import run_decision_chain_attribution as m99
from scripts import run_decision_dynamics_controls as c2
from scripts import run_decision_dynamics_retrain as rt
from scripts import run_decision_relevant_dynamics as v1
from scripts import run_gate_b as gb
from scripts import run_tau_ad_within_checkpoint as t96
from world_model.planning import scoring_harness as harness
from world_model.training import remaining_input as ri
from world_model.training import request_policy as rp
from world_model.training.cnn_hybrid import PAIRS

ROOT = v1.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-119-remaining-frames-v1"
P101 = dap.OUTPUT
GATE_B = gb.OUTPUT

IDENTITY = "issue-119-remaining-frames-v1"
SCHEMA_PLAN = "issue_119_remaining_frames_plan_v1"
SCHEMA_SMOKE = "issue_119_remaining_frames_smoke_v1"
SCHEMA_WINDOWS = "issue_119_windows_225_v1"
SCHEMA_POLICY = "issue_119_request_policy_v1"
SCHEMA_REPORT_TRAIN = "issue_119_training_report_v1"
SCHEMA_FAILURE = "issue_119_typed_failure_v1"
SCHEMA_SELECTION = "issue_119_selection_v1"
SCHEMA_ACCURACY = "issue_119_label_accuracy_v1"
SCHEMA_RECORD = "issue_119_record_v1"
SCHEMA_COMPUTE = "issue_119_compute_v1"
SCHEMA_MANIFEST = "issue_119_record_manifest_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_remaining_frames_input --validate"
RUNNER = "scripts/run_remaining_frames_input.py"
MODULE = "world_model/training/remaining_input.py"

DEVICE = "cuda"
SEEDS = dap.SEEDS
INVENTORIES = dap.INVENTORIES
PRIMARY = dap.PRIMARY
PRIMARY_COST = harness.PRIMARY_COST
DELTA = v1.DELTA
STOP_TOKEN = v1.STOP_TOKEN
FIXED_ARMS = t96.FIXED_ARMS
LAMBDA_PE = rp.ICLR_COMPUTE_WEIGHT
LAMBDA_DEC = 0.1  # the #101 selected decision-label lambda (J_dec), checked against #101 selection.json
SHORT, LONG = 60, rp.ENDPOINT
CONFIGS = (("a", SHORT), ("b", SHORT), ("c", SHORT), ("d", SHORT), ("a", LONG), ("b", LONG))
CANDIDATES = ("b", "c", "d")  # v* eligibility (declared tie order)
REFERENCE_J = dap.policy_arm("pe", LAMBDA_PE)  # #101 J record / policy (replication target)
REFERENCE_JDEC = dap.policy_arm("dec", LAMBDA_DEC)  # #101 J_dec (variant a under decision labels)
JDEC_A = "Jdec-a"
GPU_CAP_SECONDS = 12 * 3600.0
WALL_CAP_SECONDS = 24 * 3600.0  # summed over processes
SMOKE_UNITS = 4

read_json = v1.read_json
write_json = v1.write_json
json_text = v1.json_text
sha256_of = v1.sha256_of
atomic_torch = v1.atomic_torch
fmt_interval = v1.fmt_interval
PAIR_NAMES = t96.PAIR_NAMES


def log(message):
    print(f"[issue-119] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# groups and arms
# ---------------------------------------------------------------------------

def pe_arm(variant, window):
    return f"J-{variant}-L{window}"


def dec_arm(variant):
    return f"Jdec-{variant}"


PE_ARMS = tuple(pe_arm(v, w) for v, w in CONFIGS)
J_CONTROL = pe_arm("a", SHORT)


def phase1_groups():
    """(kind, variant, window, seed), seed-major so the four L225 groups per seed spread over workers."""
    return [("pe", v, w, s) for s in SEEDS for v, w in CONFIGS]


def phase2_groups(selection):
    return [("dec", selection["v_star"], None, s) for s in SEEDS]


def all_groups(selection):
    return phase1_groups() + phase2_groups(selection)


def group_arm(group):
    kind, variant, window, _ = group
    return pe_arm(variant, window) if kind == "pe" else dec_arm(variant)


def new_arms(selection):
    return (*PE_ARMS, dec_arm(selection["v_star"]))


def arm_group(arm, seed):
    if arm.startswith("Jdec-"):
        return ("dec", arm[5:], None, seed)
    _, variant, window = arm.split("-")
    return ("pe", variant, int(window[1:]), seed)


def table_arms(selection):
    """Every ranked arm: the nine fixed requests and #101 J_dec (reused records), the new policies."""
    return (*FIXED_ARMS, *new_arms(selection), JDEC_A)


# ---------------------------------------------------------------------------
# paths and ledger
# ---------------------------------------------------------------------------

def group_root(output, group):
    kind, variant, window, seed = group
    name = f"pe-{variant}-L{window}" if kind == "pe" else f"dec-{variant}"
    return Path(output) / "models" / name / f"seed-{seed}"


def policy_path(output, group):
    return group_root(output, group) / "policy.pt"


def report_path(output, group):
    return group_root(output, group) / "report.json"


def failure_path(output, group):
    return group_root(output, group) / "failure.json"


def windows_path(output):
    return Path(output) / "data" / "windows-225.pt"


def selection_path(output):
    return Path(output) / "selection.json"


def accuracy_path(output, name, seed):
    return Path(output) / "accuracy" / f"{name}--seed{seed}.json"


def record_path(output, arm, seed):
    return Path(output) / "records" / f"{arm}--seed{seed}.json"


def ledger_path(output, phase):
    return Path(output) / "ledger" / f"{phase}.json"


def ledger_phases(output):
    root = Path(output) / "ledger"
    return {p.stem: read_json(p) for p in sorted(root.glob("*.json"))} if root.is_dir() else {}


def mark(output, phase, wall, gpu, **extra):
    write_json(ledger_path(output, phase), {"phase": phase, "wall_seconds": wall, "gpu_seconds": gpu,
                                            "at": utc_now(), **extra})
    phases = ledger_phases(output)
    totals = {"gpu_seconds": sum(b["gpu_seconds"] for b in phases.values()),
              "wall_seconds": sum(b["wall_seconds"] for b in phases.values())}
    if totals["gpu_seconds"] > GPU_CAP_SECONDS or totals["wall_seconds"] > WALL_CAP_SECONDS:
        raise ValueError(f"{STOP_TOKEN}: compute cap exceeded ({totals})")


# ---------------------------------------------------------------------------
# label windows
# ---------------------------------------------------------------------------

def assemble(output):
    """Start-0 windows reaching the 225-frame endpoint of every #101 pool shot (same shot order)."""
    output = Path(output)
    load_plan(output)
    if windows_path(output).is_file():
        log("windows already assembled")
        return 0
    began = time.monotonic()
    pool = dap.load_pool(P101)
    windows, shots = [], []
    for key in pool["shot_keys"]:
        shot = dap.read_shot(P101, key)
        windows.append(rp.first_window(shot["carriers"], shot["action"], shot["end"], LONG))
        shots.append({"key": key, "sha256": sha256_of(dap.shot_path(P101, key)), "end": int(shot["end"])})
    atomic_torch(windows_path(output), {
        "schema": SCHEMA_WINDOWS, "plan_identity": IDENTITY, "assembled_at": utc_now(), "declared_length": LONG,
        "levels": pool["levels"], "shot_keys": pool["shot_keys"], "shot_members": pool["shot_members"],
        "shots": shots, "window_z": torch.stack([w["z"] for w in windows]),
        "window_action": torch.stack([w["action"] for w in windows]),
        "window_length": torch.stack([w["length"] for w in windows])})
    wall = time.monotonic()-began
    mark(output, "assemble", wall, 0.0, shots=len(windows))
    log(f"assembled {len(windows)} 225-frame windows ({wall:.0f}s)")
    return 0


def load_windows(output):
    payload = torch.load(windows_path(output), map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA_WINDOWS or payload.get("plan_identity") != IDENTITY:
        raise ValueError("225-frame window binding differs")
    return payload


def verify_window_shots(windows):
    """windows-225.pt embeds the sha256 of every #101 pool shot it was assembled from (the plan, like #101's, does
    not pin the shot files); every one is recomputed and any difference, or a shot order other than the pool's,
    refuses the publication (phase-B/C review)."""
    keys = [s["key"] for s in windows["shots"]]
    changed = [s["key"] for s in windows["shots"] if sha256_of(dap.shot_path(P101, s["key"])) != s["sha256"]]
    if changed or keys != list(windows["shot_keys"]):
        raise ValueError(f"225-frame window sources changed since assembly: {len(changed)} shot hashes differ "
                         f"({changed[:3]}), shot order equal: {keys == list(windows['shot_keys'])}")
    return len(keys)


def window_units(output, window):
    """Per policy level (roster order) its shots' windows: the #101 first 60-frame windows or the 225 ones."""
    return dap.window_units(dap.load_pool(P101) if window == SHORT else load_windows(output))


def window_accounting(payload):
    lengths = payload["window_length"]
    return {"windows": int(len(lengths)), "reaching_225": int((lengths == LONG).sum()),
            "mean_length": float(lengths.float().mean()), "min_length": int(lengths.min()),
            "label_rows": int(lengths.sum())}


# ---------------------------------------------------------------------------
# policies and training
# ---------------------------------------------------------------------------

def save_policy(output, group, policy, extra):
    kind, variant, window, seed = group
    atomic_torch(policy_path(output, group), {
        "schema": SCHEMA_POLICY, "plan_identity": IDENTITY, "kind": kind, "variant": variant, "window": window,
        "seed": seed, "lambda": LAMBDA_PE if kind == "pe" else LAMBDA_DEC,
        "pairs": [str(p.identity) for p in policy.pairs], "width": policy.layers[0].out_features,
        "model": policy.state_dict(), **extra})


def load_policy(output, group):
    payload = torch.load(policy_path(output, group), map_location="cpu", weights_only=True)
    kind, variant, window, seed = group
    if (payload.get("schema") != SCHEMA_POLICY or payload.get("plan_identity") != IDENTITY
            or (payload["kind"], payload["variant"], payload["window"], payload["seed"]) != group):
        raise ValueError(f"policy {group} binding differs")
    policy = ri.build_policy(variant, PAIRS, payload["width"])
    if payload["pairs"] != [str(p.identity) for p in policy.pairs]:
        raise ValueError("policy request set differs")
    policy.load_state_dict(payload["model"], strict=True)
    return policy.to(DEVICE).eval()


def decision_units(model, policy, pool, round_index):
    """#101 decision-valued rows (lambda 1e-1, F-15-macro then the round-0 policy), per training-cell
    unit, with the elapsed column (= position)."""
    units, count = dap.cell_units(pool)
    continuation = ({"kind": "fixed", "pair_index": PAIR_NAMES.index(dap.BASE_CONTINUATION[2:])}
                    if round_index == 0 else {"kind": "policy"})
    rows = ri.decision_elapsed(rp.decision_rows(model, policy, pool["cells"], continuation=continuation,
                                                cost_fn=dap.tie_free_tensor, compute_weight=LAMBDA_DEC,
                                                device=DEVICE))
    unit_of = torch.tensor(units)[rows["cell"]]
    return [{k: rows[k][unit_of == u] for k in ri.ROW_KEYS} for u in range(count)]


def rows_stats(units):
    labels = torch.cat([u["labels"] for u in units])
    return {"rows": int(len(labels)), "label_counts": torch.bincount(labels, minlength=len(PAIRS)).tolist()}


def train_group(output, group):
    """One group; resumable per round (labels, checkpoint)."""
    kind, variant, window, seed = group
    root = group_root(output, group)
    began = time.monotonic()
    pool = dap.load_pool(P101)
    units = window_units(output, window) if kind == "pe" else None
    model = dap.load_predictor(seed)
    with harness.deterministic_scoring() as scoring_policy:
        policy = ri.new_policy(variant, PAIRS, seed, device=DEVICE)
        stats, history = {}, []
        for round_index in range(rp.DISTIL_RECIPE["rounds"]):
            checkpoint = root / f"round-{round_index}.pt"
            labels = root / f"labels-{round_index}.pt"
            if labels.is_file():
                history.append(torch.load(labels, map_location="cpu", weights_only=True)["units"])
            else:
                t0 = time.monotonic()
                if kind == "pe":
                    rows = [ri.window_rows(model, policy, unit, aggregated=bool(round_index), compute_weight=LAMBDA_PE,
                                           device=DEVICE) for unit in units]
                else:
                    rows = decision_units(model, policy, pool, round_index)
                atomic_torch(labels, {"units": rows})
                history.append(rows)
                stats[f"label_seconds_{round_index}"] = time.monotonic()-t0
            if checkpoint.is_file():
                policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
                continue
            t0 = time.monotonic()
            losses = ri.distil_round(policy, ri.aggregate_rounds(history), round_index, seed=seed, device=DEVICE)
            atomic_torch(checkpoint, {"model": policy.state_dict(), "losses": losses})
            stats[f"distil_seconds_{round_index}"] = time.monotonic()-t0
            stats[f"final_loss_{round_index}"] = float(np.mean(losses[-60:]))
            log(f"{group_arm(group)}/{seed}: round {round_index} rows {sum(len(u['labels']) for u in history[-1])}, "
                f"loss {stats[f'final_loss_{round_index}']:.4f}")
        stats["labels"] = {f"round_{r}": rows_stats(h) for r, h in enumerate(history)}
        policy.eval()
        evaluation = dap.policy_split_evaluation(model, policy, pool)
    save_policy(output, group, policy, {"scoring_policy": scoring_policy})
    wall = time.monotonic()-began
    report = {"schema": SCHEMA_REPORT_TRAIN, "plan_identity": IDENTITY, "group": list(group),
              "arm": group_arm(group), "policy_sha256": sha256_of(policy_path(output, group)),
              "trained_at": utc_now(), "predictor_sha256": sha256_of(gb.member_path("unit", seed)),
              "policy_macs": rp.policy_macs(policy), "stats": stats, "policy_split": evaluation,
              "wall_seconds": wall}
    write_json(report_path(output, group), report)
    return report, wall


def run_group(output, group):
    """Train a group once; a raised error is a typed terminal failure (recorded, never retried)."""
    if report_path(output, group).is_file() or failure_path(output, group).is_file():
        log(f"{group_arm(group)}/{group[3]}: already terminal")
        return
    began = time.monotonic()
    try:
        report, wall = train_group(output, group)
    except (ValueError, RuntimeError) as error:
        wall = time.monotonic()-began
        write_json(failure_path(output, group), {"schema": SCHEMA_FAILURE, "plan_identity": IDENTITY,
                                                 "group": list(group), "type": type(error).__name__,
                                                 "message": str(error), "at": utc_now(), "wall_seconds": wall})
        mark(output, f"train-{group_arm(group)}-seed{group[3]}", wall, wall, typed_failure=type(error).__name__)
        log(f"{group_arm(group)}/{group[3]}: TYPED FAILURE {type(error).__name__}: {error}")
        return
    mark(output, f"train-{group_arm(group)}-seed{group[3]}", wall, wall)
    log(f"{group_arm(group)}/{group[3]}: trained ({wall:.0f}s); policy-split AUC "
        f"{report['policy_split']['auc_pooled']:.4f}, MACs/candidate {report['policy_split']['mean_macs_per_candidate']:.3g}")


def train(output, workers, worker):
    output = Path(output)
    load_plan(output)
    load_windows(output)
    for group in phase1_groups()[worker::workers]:
        run_group(output, group)
    return 0


def train_dec(output, workers, worker):
    output = Path(output)
    load_plan(output)
    selection = load_selection(output)
    for group in phase2_groups(selection)[worker::workers]:
        run_group(output, group)
    return 0


def load_report(output, group):
    report = read_json(report_path(output, group))
    if (report.get("schema") != SCHEMA_REPORT_TRAIN or report.get("plan_identity") != IDENTITY
            or tuple(report["group"]) != group):
        raise ValueError(f"training report {group} binding differs")
    return report


def terminal(output, group):
    """'report', 'failure' or None."""
    return "report" if report_path(output, group).is_file() else "failure" if failure_path(output, group).is_file() else None


# ---------------------------------------------------------------------------
# selection of v* (policy split only)
# ---------------------------------------------------------------------------

def selection_table(output):
    """v* = the variant in {b, c, d} (L60 labels) with the largest seed-mean policy-split pooled AUC;
    a variant with a typed training failure is ineligible; ties in the order b, c, d."""
    scores = {}
    for variant in CANDIDATES:
        groups = [("pe", variant, SHORT, s) for s in SEEDS]
        if any(terminal(output, g) != "report" for g in groups):
            scores[variant] = None
            continue
        scores[variant] = float(np.mean([load_report(output, g)["policy_split"]["auc_pooled"] for g in groups]))
    eligible = [v for v in CANDIDATES if scores[v] is not None]
    if not eligible:
        raise ValueError(f"{STOP_TOKEN}: no eligible input variant")
    best = max(eligible, key=lambda v: (scores[v], -CANDIDATES.index(v)))
    control = [("pe", "a", SHORT, s) for s in SEEDS]
    reference = (float(np.mean([load_report(output, g)["policy_split"]["auc_pooled"] for g in control]))
                 if all(terminal(output, g) == "report" for g in control) else None)
    return {"v_star": best, "policy_split_auc": scores, "control_a_policy_split_auc": reference}


def select(output):
    output = Path(output)
    load_plan(output)
    if selection_path(output).is_file():
        log(f"selection exists: {load_selection(output)['v_star']}")
        return 0
    missing = [g for g in phase1_groups() if terminal(output, g) is None]
    if missing:
        raise ValueError(f"{len(missing)} phase-1 groups not terminal; selection follows training")
    if any(record_path(output, a, s).is_file() for a in PE_ARMS for s in SEEDS):
        raise ValueError("records exist before the selection; refusing")
    table = selection_table(output)
    write_json(selection_path(output), {"schema": SCHEMA_SELECTION, "plan_identity": IDENTITY,
                                        "selected_at": utc_now(), **table})
    log(f"selected v* = {table['v_star']} ({table['policy_split_auc']})")
    return 0


def load_selection(output):
    selection = read_json(selection_path(output))
    if selection.get("schema") != SCHEMA_SELECTION or selection.get("plan_identity") != IDENTITY:
        raise ValueError("selection binding differs")
    return selection


# ---------------------------------------------------------------------------
# label accuracy (policy split, training data)
# ---------------------------------------------------------------------------

def round_rows(output, group, round_index):
    units = torch.load(group_root(output, group) / f"labels-{round_index}.pt", map_location="cpu",
                       weights_only=True)["units"]
    return {k: torch.cat([u[k] for u in units]) for k in units[0]}


def reference_jdec_rows(seed):
    """#101 J_dec's own training rows (both rounds; variant a reads no elapsed column)."""
    root = dap.group_root(P101, "dec", LAMBDA_DEC, seed)
    units = [torch.load(root / f"labels-{r}.pt", map_location="cpu", weights_only=True)["units"] for r in (0, 1)]
    rows = [torch.cat([u[k] for rounds in units for u in rounds]) for k in ("z", "action", "remaining", "labels")]
    return dict(zip(("z", "action", "remaining", "labels"), rows))


def accuracy_tasks(selection):
    return ([(group_arm(g), g[3]) for g in all_groups(selection)] + [(JDEC_A, s) for s in SEEDS])


def accuracy_record(output, arm, seed):
    """Agreement with: L60 = round-0 (observed-context) labels of the first 60-frame windows; L225 =
    round-0 labels of the 225-frame windows (both from the variant-a groups of the seed); own = the
    policy's own aggregated training rows (in-sample fit)."""
    sets = {"L60": round_rows(output, ("pe", "a", SHORT, seed), 0), "L225": round_rows(output, ("pe", "a", LONG, seed), 0)}
    if arm == JDEC_A:
        policy = dap.load_policy(P101, "dec", LAMBDA_DEC, seed)
        own = reference_jdec_rows(seed)
    else:
        group = arm_group(arm, seed)
        policy = load_policy(output, group)
        own = [round_rows(output, group, r) for r in (0, 1)]
        own = {k: torch.cat([o[k] for o in own]) for k in ri.ROW_KEYS}
    out = {name: ri.label_accuracy(policy, rows, device=DEVICE) for name, rows in sets.items()}
    out["own"] = ri.label_accuracy(policy, own, device=DEVICE)
    return out


def accuracy(output, workers, worker):
    output = Path(output)
    load_plan(output)
    selection = load_selection(output)
    missing = [g for g in all_groups(selection) if terminal(output, g) is None]
    if missing:
        raise ValueError(f"{len(missing)} groups not terminal; accuracy follows training")
    for arm, seed in accuracy_tasks(selection)[worker::workers]:
        path = accuracy_path(output, arm, seed)
        if path.is_file():
            continue
        if arm != JDEC_A and terminal(output, arm_group(arm, seed)) != "report":
            write_json(path, {"schema": SCHEMA_ACCURACY, "plan_identity": IDENTITY, "arm": arm, "seed": seed,
                              "typed_failure": True})
            continue
        t0 = time.monotonic()
        with harness.deterministic_scoring() as policy_state:
            result = accuracy_record(output, arm, seed)
        write_json(path, {"schema": SCHEMA_ACCURACY, "plan_identity": IDENTITY, "arm": arm, "seed": seed,
                          "scoring_policy": policy_state, "typed_failure": False, **result})
        wall = time.monotonic()-t0
        mark(output, f"accuracy-{arm}-seed{seed}", wall, wall)
        log(f"worker {worker}: accuracy {arm}/{seed}: L60 {result['L60']['accuracy']:.4f} "
            f"L225 {result['L225']['accuracy']:.4f} own {result['own']['accuracy']:.4f}")
    return 0


def read_accuracy(output, arm, seed):
    record = read_json(accuracy_path(output, arm, seed))
    if record.get("schema") != SCHEMA_ACCURACY or record.get("plan_identity") != IDENTITY:
        raise ValueError(f"accuracy {arm}/{seed} binding differs")
    return record


# ---------------------------------------------------------------------------
# scoring (deterministic_scoring, processes that did not train)
# ---------------------------------------------------------------------------

def failed_record(base, cells, failure):
    """Every cell of an arm whose policy is a typed training failure: no rows."""
    inventories = {i: {} for i in INVENTORIES}
    for inventory, member, entry in dap.ordered_cells(cells):
        inventories[inventory][member] = {"rows": None, "candidates": len(entry["inventory"]), "transition_calls": 0,
                                          "controller_calls": 0, "linear_macs": 0, "gpu_seconds": 0.0}
    return {**base, "typed_failure": failure, "inventories": inventories}


def score_record(output, cells, anchors, arm, seed, policy_state):
    costs = dap.endpoint_costs()
    group = arm_group(arm, seed)
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "scoring_policy": policy_state,
            "scored_at": utc_now(), "arm": arm, "seed": seed, "engine_seconds": 0,
            "predictor_sha256": sha256_of(gb.member_path("unit", seed))}
    if terminal(output, group) != "report":
        return failed_record(base, cells, read_json(failure_path(output, group)))
    began = time.monotonic()
    model = dap.load_predictor(seed)
    policy = load_policy(output, group)
    inventories = {i: {} for i in INVENTORIES}
    for inventory, member, entry in dap.ordered_cells(cells):
        inventories[inventory][member] = dap.block(model, policy, {"kind": "joint"}, anchors[member],
                                                   entry["inventory"], costs, True)
    torch.cuda.synchronize()
    return {**base, "policy": {"group": list(group), "sha256": sha256_of(policy_path(output, group))},
            "inventories": inventories, "seconds": time.monotonic()-began}


def wrong_record(output, arm, seed, policy_state):
    """The arm's policy from every fit-split anchor over the shared grid candidates (Gate-B rule)."""
    costs = dap.endpoint_costs()
    group = arm_group(arm, seed)
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "scoring_policy": policy_state,
            "scored_at": utc_now(), "arm": f"wrong-{arm}", "seed": seed, "engine_seconds": 0}
    if terminal(output, group) != "report":
        return {**base, "typed_failure": read_json(failure_path(output, group)), "wrong_anchor": None}
    model = dap.load_predictor(seed)
    policy = load_policy(output, group)
    items = gb.inventory_items(PRIMARY)
    anchors = gb.fit_anchors()
    began = time.monotonic()
    rows = {member: dap.block(model, policy, {"kind": "joint"}, anchors[member], items, costs, False)
            for member in sorted(anchors)}
    return {**base, "policy": {"group": list(group), "sha256": sha256_of(policy_path(output, group))},
            "wrong_anchor": rows, "seconds": time.monotonic()-began}


def read_record(output, arm, seed):
    record = read_json(record_path(output, arm, seed))
    if record.get("schema") != SCHEMA_RECORD or record.get("plan_identity") != IDENTITY or record["seed"] != seed:
        raise ValueError(f"record {arm}/{seed} binding differs")
    return record


def score_tasks(selection):
    arms = new_arms(selection)
    return [(arm, s) for arm in arms for s in SEEDS] + [(f"wrong-{arm}", s) for arm in arms for s in SEEDS]


def score(output, workers, worker):
    output = Path(output)
    load_plan(output)
    selection = load_selection(output)
    missing = [g for g in all_groups(selection) if terminal(output, g) is None]
    if missing:
        raise ValueError(f"{len(missing)} training groups not terminal; scoring follows training")
    _, cells = gb.evaluation_set()
    anchors = gb.load_anchors(GATE_B)["anchors"]
    for arm, seed in score_tasks(selection)[worker::workers]:
        if record_path(output, arm, seed).is_file():
            continue
        t0 = time.monotonic()
        with harness.deterministic_scoring() as policy_state:
            record = (wrong_record(output, arm[6:], seed, policy_state) if arm.startswith("wrong-")
                      else score_record(output, cells, anchors, arm, seed, policy_state))
        write_json(record_path(output, arm, seed), record)
        torch.cuda.empty_cache()
        wall = time.monotonic()-t0
        mark(output, f"score-{arm}-seed{seed}", wall, wall)
        log(f"worker {worker}: scored {arm} seed {seed} ({wall:.0f}s)")
    return 0


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def reused_names():
    """#101 records read here (fixed requests, the replication target J, the J_dec reference)."""
    return (*FIXED_ARMS, REFERENCE_J, REFERENCE_JDEC)


def reused_record_check():
    """Every reused #101 record matches the committed #101 raw-record manifest."""
    manifest = read_json(P101 / "records-manifest.json")["records"]
    checked = {}
    for arm in reused_names():
        for seed in SEEDS:
            path = dap.record_path(P101, arm, seed)
            entry = manifest[str(path.relative_to(P101))]
            checked[f"{arm}/{seed}"] = sha256_of(path) == entry["sha256"]
    return {"records": len(checked), "pass": all(checked.values())}


def load_records(output, selection):
    records = {(arm, s): read_record(output, arm, s) for arm in new_arms(selection) for s in SEEDS}
    for arm in FIXED_ARMS:
        for s in SEEDS:
            records[(arm, s)] = dap.read_record(P101, arm, s)
    for s in SEEDS:
        records[(JDEC_A, s)] = dap.read_record(P101, REFERENCE_JDEC, s)
    return records


def view_rows(records, cells, inventory, stats, arms):
    rows = []
    for member, entry in sorted(cells[inventory]["mixed"].items()):
        prior = m99.view(entry, gb.prior_rows(entry, stats), PRIMARY_COST)
        for seed in SEEDS:
            views = {arm: v1.cost_view(entry, records[(arm, seed)]["inventories"][inventory][member]["rows"],
                                       PRIMARY_COST) for arm in arms}
            views["prior"] = prior
            rows.append({"member": member, "seed": seed, "views": views})
    return rows


def member_means(rows, arms):
    return {arm: t96.member_mean([{"member": r["member"], "auc": r["views"][arm]["auc"]} for r in rows], "auc")
            for arm in arms}


def arm_compute(records, inventory, arm):
    results = [records[(arm, s)]["inventories"][inventory][m] for s in SEEDS
               for m in records[(arm, s)]["inventories"][inventory]]
    results = [r for r in results if r["rows"] is not None]
    candidates = sum(r["candidates"] for r in results)
    return {"mean_macs_per_candidate": sum(r["linear_macs"] for r in results)/candidates if candidates else None,
            "mean_transitions_per_candidate": sum(r["transition_calls"] for r in results)/candidates if candidates else None,
            "mean_policy_calls_per_candidate": sum(r["controller_calls"] for r in results)/candidates if candidates else None,
            "gpu_seconds": sum(r["gpu_seconds"] for r in results)}


def kappa_table(records, inventories, arm):
    """Request mix against the true remaining frames: per kappa bin, step counts per request and the
    Delta / alpha shares, pooled over seeds, cells and candidates of ``inventories``."""
    counts = [[0]*len(PAIRS) for _ in ri.KAPPA_BINS]
    for seed in SEEDS:
        for inventory in inventories:
            for result in records[(arm, seed)]["inventories"][inventory].values():
                if result["rows"] is None:
                    continue
                part = ri.kappa_mix([t96.unrle(text) for text in result["schedules"].values()], PAIRS)
                for b in range(len(ri.KAPPA_BINS)):
                    for k in range(len(PAIRS)):
                        counts[b][k] += part[b][k]

    def shares(row):
        total = sum(row)
        if not total:
            return {"steps": 0, **{f"delta_{d}": None for d in t96.DELTAS}, **{a: None for a in t96.ALPHAS}}
        return {"steps": total,
                **{f"delta_{d}": sum(row[3*i:3*i+3])/total for i, d in enumerate(t96.DELTAS)},
                **{a: sum(row[j::3])/total for j, a in enumerate(t96.ALPHAS)}}

    bins = [{"kappa": list(b), "pair_counts": row, **shares(row)} for b, row in zip(ri.KAPPA_BINS, counts)]
    early = shares(counts[0])
    late = shares([counts[3][k]+counts[4][k] for k in range(len(PAIRS))])
    shift = {key: (late[key]-early[key]) if late[key] is not None and early[key] is not None else None
             for key in ("delta_1", "delta_5", "delta_15", "continuous", "micro", "macro")}
    return {"bins": bins, "late_minus_early": shift,
            "early": "kappa 166-225 (the saturated first 60 frames)", "late": "kappa 1-60"}


def contrast_block(rows, pieces, label):
    block = m99.paired(rows, pieces)
    return {**block, "contrast": label, "pieces": [[w, k] for w, k in pieces], "token": c2.token(block["estimate"])}


def vs_j_names(selection):
    """arm -> name of its contrast against J = J(a, L60)."""
    names = {pe_arm(v, SHORT): f"J({v},L60)-J(a,L60)" for v in CANDIDATES}
    names.update({pe_arm("a", LONG): "J(a,L225)-J(a,L60)", pe_arm("b", LONG): "J(b,L225)-J(a,L60)",
                  dec_arm(selection["v_star"]): "J_dec(v*)-J", JDEC_A: "J_dec(a)-J"})
    return names


def contrast_specs(selection, hindsight):
    """name -> pieces. Primary: J(v*, L60) - J(a, L60) = (J(v*) - F_hindsight) - (J - F_hindsight)."""
    star = selection["v_star"]
    jd = dec_arm(star)
    specs = {"primary: J(v*)-J": [(1, pe_arm(star, SHORT)), (-1, J_CONTROL)]}
    for arm, name in vs_j_names(selection).items():
        specs[name] = [(1, arm), (-1, J_CONTROL)]
    specs["J(b,L225)-J(b,L60)"] = [(1, pe_arm("b", LONG)), (-1, pe_arm("b", SHORT))]
    specs["J(b,L225)-J(a,L225)"] = [(1, pe_arm("b", LONG)), (-1, pe_arm("a", LONG))]
    specs["interaction input x window"] = [(1, pe_arm("b", LONG)), (-1, pe_arm("a", LONG)), (-1, pe_arm("b", SHORT)),
                                           (1, J_CONTROL)]
    specs["J_dec(v*)-J_dec(a)"] = [(1, jd), (-1, JDEC_A)]
    specs["J_dec(v*)-J(v*,L60)"] = [(1, jd), (-1, pe_arm(star, SHORT))]
    for arm in (*new_arms(selection), JDEC_A):
        specs[f"{arm}-F_hindsight"] = [(1, arm), (-1, hindsight)]
        specs[f"{arm}-prior"] = [(1, arm), (-1, "prior")]
    return specs


def inventory_tables(records, rows, inventory, selection):
    arms = table_arms(selection)
    means = member_means(rows, arms)
    hindsight = max(FIXED_ARMS, key=lambda a: (-math.inf if means[a] is None else means[a], -FIXED_ARMS.index(a)))
    rankers = {arm: {**v1.ranker_block(rows, arm), **(arm_compute(records, inventory, arm) if arm != "prior" else {})}
               for arm in (*arms, "prior")}
    return {"cells": len(rows), "members": len({r["member"] for r in rows}), "F_hindsight": hindsight,
            "member_mean_auc": means, "rankers": rankers,
            "contrasts": {name: contrast_block(rows, pieces, name)
                          for name, pieces in contrast_specs(selection, hindsight).items()},
            "request_mix_by_kappa": {arm: kappa_table(records, (inventory,), arm)
                                     for arm in (*new_arms(selection), JDEC_A)}}


def wrong_anchor_table(output, records, cells, arm):
    wrong_records = {s: read_record(output, f"wrong-{arm}", s)["wrong_anchor"] for s in SEEDS}
    rows = []
    for member, entry in sorted(cells[PRIMARY]["mixed"].items()):
        for seed in SEEDS:
            right = records[(arm, seed)]["inventories"][PRIMARY][member]["rows"]
            if right is None or wrong_records[seed] is None:
                continue
            wrong = [wrong_records[seed][a]["rows"] for a in sorted(wrong_records[seed])]
            wrong_auc = [a for a in (c2.auc(entry, w) for w in wrong) if a is not None]
            rhos = [r for r in (c2.spearman_rows(right, w) for w in wrong) if r is not None]
            rows.append({"member": member, "seed": seed, "views": {"x": {
                "right": c2.auc(entry, right), "wrong": float(np.mean(wrong_auc)) if wrong_auc else None,
                "stability": float(np.mean(rhos)) if rhos else None}}})
    block = c2.difference_block(rows, [(1, "x", "right"), (-1, "x", "wrong")])
    return {"arm": arm, "cells": len(rows), "wrong_anchors": len(gb.fit_anchors()),
            "right": c2.field_block(rows, "x", "right"), "wrong": c2.field_block(rows, "x", "wrong"),
            "right_minus_wrong": block, "stability": c2.field_block(rows, "x", "stability"), "token": c2.token(block)}


def invalidate(obj):
    """A failed guard turns every token into readiness_or_precision_insufficient (frozen guards.failure)."""
    if isinstance(obj, dict):
        return {k: STOP_TOKEN if k == "token" else invalidate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [invalidate(v) for v in obj]
    return obj


def replication_guard(output, records):
    """Variant a (L60) reproduces the #101 J (pe, lambda 1e-4) bit for bit: both label rounds, both
    round checkpoints and the final policy state dict, and every scored block (rows, schedules, calls,
    MACs) of every inventory, cell and seed."""
    out = {"labels": True, "checkpoints": True, "policy": True, "blocks": 0, "records": True}
    for seed in SEEDS:
        group = ("pe", "a", SHORT, seed)
        if terminal(output, group) != "report":
            out.update(labels=False, checkpoints=False, policy=False, records=False)
            continue
        mine_root, theirs_root = group_root(output, group), dap.group_root(P101, "pe", LAMBDA_PE, seed)
        for r in (0, 1):
            mine = torch.load(mine_root / f"labels-{r}.pt", map_location="cpu", weights_only=True)["units"]
            theirs = torch.load(theirs_root / f"labels-{r}.pt", map_location="cpu", weights_only=True)["units"]
            out["labels"] &= len(mine) == len(theirs) and all(
                torch.equal(a[k], b[k]) for a, b in zip(mine, theirs) for k in ("z", "action", "remaining", "labels"))
            mine = torch.load(mine_root / f"round-{r}.pt", map_location="cpu", weights_only=True)["model"]
            theirs = torch.load(theirs_root / f"round-{r}.pt", map_location="cpu", weights_only=True)["model"]
            out["checkpoints"] &= mine.keys() == theirs.keys() and all(torch.equal(mine[k], theirs[k]) for k in mine)
        mine = torch.load(policy_path(output, group), map_location="cpu", weights_only=True)["model"]
        theirs = torch.load(dap.policy_path(P101, "pe", LAMBDA_PE, seed), map_location="cpu", weights_only=True)["model"]
        out["policy"] &= mine.keys() == theirs.keys() and all(torch.equal(mine[k], theirs[k]) for k in mine)
        reference = dap.read_record(P101, REFERENCE_J, seed)["inventories"]
        for inventory in INVENTORIES:
            for member, result in reference[inventory].items():
                ours = records[(J_CONTROL, seed)]["inventories"][inventory][member]
                out["records"] &= all(ours[k] == result[k] for k in ("rows", "schedules", "candidates",
                                                                        "transition_calls", "controller_calls",
                                                                        "linear_macs"))
                out["blocks"] += 1
    out["pass"] = bool(out["labels"] and out["checkpoints"] and out["policy"] and out["records"])
    return out


def training_table(output, selection):
    out = {}
    for group in all_groups(selection):
        arm, seed = group_arm(group), str(group[3])
        if terminal(output, group) != "report":
            out.setdefault(arm, {})[seed] = {"typed_failure": read_json(failure_path(output, group))}
            continue
        report = load_report(output, group)
        out.setdefault(arm, {})[seed] = {
            "policy_split": report["policy_split"], "labels": report["stats"]["labels"],
            "final_loss": [report["stats"].get(f"final_loss_{r}") for r in (0, 1)],
            "policy_macs": report["policy_macs"], "wall_seconds": report["wall_seconds"],
            "policy_sha256": report["policy_sha256"]}
    return out


def accuracy_table(output, selection):
    """Per arm: seed means of the label accuracy against L60 / L225 / own rows and per kappa bin (L225)."""
    out = {}
    for arm in (*new_arms(selection), JDEC_A):
        records = [read_accuracy(output, arm, s) for s in SEEDS]
        valid = [r for r in records if not r["typed_failure"]]
        entry = {"seeds": len(valid)}
        for name in ("L60", "L225", "own"):
            entry[name] = float(np.mean([r[name]["accuracy"] for r in valid])) if valid else None
        entry["L225_by_kappa"] = [{"kappa": list(b), "accuracy": float(np.mean([r["L225"]["by_kappa"][i]["accuracy"]
                                                                                for r in valid])) if valid else None}
                                  for i, b in enumerate(ri.KAPPA_BINS)]
        entry["L60_rows"] = valid[0]["L60"]["rows"] if valid else None
        entry["L225_rows"] = valid[0]["L225"]["rows"] if valid else None
        out[arm] = entry
    return out


def record_manifest(output, selection):
    """Content hashes of every raw record the publication is derived from (scored records, wrong-anchor
    records, label-accuracy records)."""
    output = Path(output)
    paths = ([record_path(output, a, s) for a, s in score_tasks(selection)]
             + [accuracy_path(output, a, s) for a, s in accuracy_tasks(selection)])
    present = sorted([*output.glob("records/*.json"), *output.glob("accuracy/*.json")])
    if sorted(paths) != present:
        raise ValueError(f"records/ and accuracy/ hold {len(present)} files, expected exactly the {len(paths)} named")
    return {"schema": SCHEMA_MANIFEST, "identity": IDENTITY, "count": len(paths),
            "records": {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha256_of(p)}
                        for p in present}}


def compute_tables(output, plan):
    output = Path(output)
    selection = load_selection(output)
    branches, cells = gb.evaluation_set()
    records = load_records(output, selection)
    arms = table_arms(selection)
    fit = read_json(gb.FIT_VERDICTS)["branches"]
    stats = {i: gb.prior_stats(fit, i) for i in INVENTORIES}
    rows = {i: view_rows(records, cells, i, stats[i], arms) for i in INVENTORIES}
    tables = {i: inventory_tables(records, rows[i], i, selection) for i in INVENTORIES}
    pooled_mix = {arm: kappa_table(records, INVENTORIES, arm) for arm in (*new_arms(selection), JDEC_A)}
    wrong = {arm: wrong_anchor_table(output, records, cells, arm) for arm in new_arms(selection)}
    training = training_table(output, selection)
    label_accuracy = accuracy_table(output, selection)
    pool = dap.load_pool(P101)
    windows = load_windows(output)
    verify_window_shots(windows)
    eval_roster = {lv["identity"] for lv in gb.evaluation_levels()}
    fit_roster = {lv["identity"] for lv in rt.fit_levels()}
    policy_roster = {lv["identity"] for lv in dap.policy_levels()}
    new_records = [records[(a, s)] for a in new_arms(selection) for s in SEEDS]
    wrong_records = [read_record(output, f"wrong-{a}", s) for a in new_arms(selection) for s in SEEDS]
    hashes_ok = all(r["policy"]["sha256"] == training[r["arm"].removeprefix("wrong-")][str(r["seed"])]["policy_sha256"]
                    for r in new_records + wrong_records if "policy" in r)
    reports = [load_report(output, g) for g in all_groups(selection) if terminal(output, g) == "report"]
    phase1 = sorted(load_report(output, g)["trained_at"] for g in phase1_groups() if terminal(output, g) == "report")
    phase2 = sorted(load_report(output, g)["trained_at"] for g in phase2_groups(selection) if terminal(output, g) == "report")
    scored_at = sorted(r["scored_at"] for r in new_records + wrong_records)
    guards = {
        "G1_isolation": {"pass": bool(set(pool["levels"]) == policy_roster and windows["shot_keys"] == pool["shot_keys"]
                                      and not (policy_roster & (eval_roster | fit_roster))
                                      and all(set(cells[i]["mixed"]) <= eval_roster for i in INVENTORIES))},
        "G2_scoring_policy": {"pass": all(r["scoring_policy"] == harness.SCORING_POLICY for r in new_records + wrong_records)
                              and all(read_accuracy(output, a, s).get("scoring_policy", harness.SCORING_POLICY)
                                      == harness.SCORING_POLICY for a, s in accuracy_tasks(selection))},
        "G3_chronology": {"plan_frozen_at": plan["frozen_at"], "first_trained_at": phase1[0] if phase1 else None,
                          "last_phase1_trained_at": phase1[-1] if phase1 else None,
                          "selected_at": selection["selected_at"],
                          "first_dec_trained_at": phase2[0] if phase2 else None,
                          "last_trained_at": max(r["trained_at"] for r in reports), "first_scored_at": scored_at[0],
                          "pass": bool(phase1 and plan["frozen_at"] <= phase1[0] and phase1[-1] <= selection["selected_at"]
                                       and (not phase2 or selection["selected_at"] <= phase2[0])
                                       and max(r["trained_at"] for r in reports) <= scored_at[0])},
        "G4_checkpoint_binding": {"pass": bool(hashes_ok)},
        "G5_replication_101_J": replication_guard(output, records),
        "G6_reused_101_records": reused_record_check(),
        "G7_selection": {"pass": selection_table(output) == {k: selection[k] for k in
                                                             ("v_star", "policy_split_auc", "control_a_policy_split_auc")}},
    }
    guards_ok = all(g["pass"] for g in guards.values())
    if not guards_ok:
        tables, wrong = invalidate(tables), invalidate(wrong)
    grid = tables[PRIMARY]
    primary = grid["contrasts"]["primary: J(v*)-J"]
    phases = ledger_phases(output)
    smoke = read_json(output / "smoke.json")
    all_phases = {"smoke": {"wall_seconds": smoke["wall_seconds"], "gpu_seconds": smoke["wall_seconds"]},
                  **{k: {"wall_seconds": v["wall_seconds"], "gpu_seconds": v["gpu_seconds"]} for k, v in phases.items()}}
    failures = sorted(f"{group_arm(g)}/{g[3]}" for g in all_groups(selection) if terminal(output, g) == "failure")
    return {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
        "selection": {k: selection[k] for k in ("v_star", "policy_split_auc", "control_a_policy_split_auc")},
        "evaluation": gb.evaluation_accounting(branches, cells), "pool": plan["pool"]["accounting"],
        "windows_225": window_accounting(windows), "guards": guards, "typed_failures": failures,
        "primary": {"claim": plan["decision_rule"]["claim"],
                    "contrast": f"(J(v*) - F_hindsight) - (J - F_hindsight) = J({selection['v_star']}, L60) - J(a, L60), grid",
                    "v_star": selection["v_star"], "estimate": primary["estimate"], "cells": primary["cells"],
                    "members": primary["members"], "F_hindsight": grid["F_hindsight"],
                    "token": c2.token(primary["estimate"]) if guards_ok else STOP_TOKEN,
                    "rule": plan["decision_rule"]["primary"]},
        "policy_macs": {arm: rp.policy_macs(ri.build_policy("a" if arm == JDEC_A else arm.split("-")[1], PAIRS))
                        for arm in (*new_arms(selection), JDEC_A)},
        "wrong_anchor_101_jdec": read_json(P101 / "compute.json")["wrong_anchor"]["right_minus_wrong"],
        "inventories": tables, "request_mix_by_kappa_all_inventories": pooled_mix, "wrong_anchor": wrong,
        "label_accuracy": label_accuracy, "training": training,
        "compute": {"phases": all_phases, "gpu_seconds": sum(b["gpu_seconds"] for b in all_phases.values()),
                    "wall_seconds": sum(b["wall_seconds"] for b in all_phases.values()), "engine_seconds": 0,
                    "hardware": "4 x RTX 5090 (one process per GPU)"},
    }


# ---------------------------------------------------------------------------
# renderings
# ---------------------------------------------------------------------------

def cell(block):
    return fmt_interval(block) if block else "n/a"


def num(value, digits=4):
    return "n/a" if value is None else format(value, f".{digits}f")


def pct(value):
    return "n/a" if value is None else f"{100*value:.1f}%"


def mmacs(value):
    return "n/a" if value is None else f"{value/1e6:.1f}M"


def arm_label(arm):
    if arm == JDEC_A:
        return "J_dec(a) (#101 J_dec)"
    if arm.startswith("Jdec-"):
        return f"J_dec({arm[5:]})"
    if arm.startswith("J-"):
        _, variant, window = arm.split("-")
        return f"J({variant}, {window})" + (" = J" if arm == J_CONTROL else "")
    return arm


def findings_md(plan, compute):
    lines = []
    add = lines.append
    grid = compute["inventories"][PRIMARY]
    p = compute["primary"]
    star = p["v_star"]
    arms_new = (*PE_ARMS, dec_arm(star))
    add("# Issue #119: remaining-frames input saturation (#104 sealed #109 evaluation split)")
    add("")
    if not all(g["pass"] for g in compute["guards"].values()):
        failed = ", ".join(n for n, g in compute["guards"].items() if not g["pass"])
        add(f"**Guard failure ({failed}): every token is {STOP_TOKEN} (frozen guards.failure).**")
        add("")
    add(f"Primary claim: \"{p['claim']}\".")
    add("")
    add(f"- v* = **{star}** ({ri.VARIANTS[star]['text']}), selected on the policy split before any score "
        f"(seed-mean policy-split AUC: " + ", ".join(f"{v} {num(compute['selection']['policy_split_auc'][v])}"
                                                    for v in CANDIDATES) + f"; control a "
        f"{num(compute['selection']['control_a_policy_split_auc'])}).")
    add(f"- {p['contrast']}: **{cell(p['estimate'])}** over {p['cells']} cells ({p['members']} levels); "
        f"F_hindsight = {p['F_hindsight']}. Token: **{p['token']}**.")
    add(f"- Rule: {p['rule']}")
    add("")
    add(f"Intervals: member (level)-clustered percentile bootstrap, 10000 draws, PCG64 7201, DESCRIPTIVE; δ = {DELTA}. "
        "Tokens follow the δ rule only: supported iff mean ≥ δ and lower > 0; not_supported_by_this_experiment iff "
        f"upper < δ; otherwise {STOP_TOKEN}. Offset, angle and power are DESCRIPTIVE.")
    add("")
    add("## Per-variant table (grid)")
    add("")
    add("| arm | input | labels | label acc. L60 | label acc. L225 | own rows | AUC | − F_hindsight | − J | − prior | "
        "MACs / candidate | policy MACs / state | right − wrong anchor |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for arm in (*arms_new, JDEC_A):
        acc = compute["label_accuracy"][arm]
        rank = grid["rankers"][arm]
        minus_j = grid["contrasts"].get(vs_j_names({"v_star": star}).get(arm, ""))
        variant = "a" if arm == JDEC_A else arm.split("-")[1]
        labels = ("decision (λ 1e-1)" if arm.startswith("Jdec") else f"pe, {arm.split('-')[2]}")
        wrong = compute["wrong_anchor"].get(arm)
        add(f"| {arm_label(arm)} | {variant} | {labels} | {pct(acc['L60'])} | {pct(acc['L225'])} | {pct(acc['own'])} | "
            f"{cell(rank['auc'])} | {cell(grid['contrasts'][f'{arm}-F_hindsight']['estimate'])} | "
            f"{cell(minus_j['estimate']) if minus_j else ('—' if arm == J_CONTROL else 'n/a')} | "
            f"{cell(grid['contrasts'][f'{arm}-prior']['estimate'])} | {mmacs(rank['mean_macs_per_candidate'])} | "
            f"{compute['policy_macs'][arm]} | "
            f"{cell(wrong['right_minus_wrong']) if wrong else cell(compute['wrong_anchor_101_jdec']) + ' (#101)'} |")
    add("")
    add("Label accuracy is argmax agreement with the DP labels on the policy split (training data, in-sample for a "
        "policy's own label kind): L60 = round-0 labels of the first 60-frame windows, L225 = round-0 labels of the "
        "225-frame windows (both from the variant-a groups of the same seed), own = the policy's aggregated training "
        "rows (both DAgger rounds). Seed means.")
    add("")
    add("## Contrasts (AUC difference paired by level and seed)")
    add("")
    names = ["primary: J(v*)-J", *(f"J({v},L60)-J(a,L60)" for v in CANDIDATES), "J(a,L225)-J(a,L60)",
             "J(b,L225)-J(b,L60)", "J(b,L225)-J(a,L60)", "J(b,L225)-J(a,L225)", "interaction input x window",
             "J_dec(v*)-J_dec(a)", "J_dec(v*)-J(v*,L60)", f"{J_CONTROL}-F_hindsight", f"{pe_arm(star, SHORT)}-F_hindsight",
             f"{dec_arm(star)}-F_hindsight", f"{JDEC_A}-F_hindsight"]
    add("| contrast | " + " | ".join(INVENTORIES) + " |")
    add("| --- | " + " | ".join("---" for _ in INVENTORIES) + " |")
    for name in names:
        add(f"| {name} | " + " | ".join(
            f"{cell(compute['inventories'][i]['contrasts'][name]['estimate'])} "
            f"{compute['inventories'][i]['contrasts'][name]['token']}" for i in INVENTORIES) + " |")
    add("")
    add("## Per-inventory AUC")
    add("")
    add("| arm | " + " | ".join(INVENTORIES) + " | MACs / candidate (grid) |")
    add("| --- | " + " | ".join("---" for _ in INVENTORIES) + " | --- |")
    for arm in (*FIXED_ARMS, *arms_new, JDEC_A, "prior"):
        marks = {i: " (F_hindsight)" if compute["inventories"][i]["F_hindsight"] == arm else "" for i in INVENTORIES}
        add(f"| {arm_label(arm) if arm != 'prior' else 'no-model action prior'} | " + " | ".join(
            f"{cell(compute['inventories'][i]['rankers'][arm]['auc'])}{marks[i]}" for i in INVENTORIES)
            + f" | {mmacs(grid['rankers'][arm].get('mean_macs_per_candidate'))} |")
    add("")
    add("## Request mix against the true remaining frames κ (grid; share of policy steps)")
    add("")
    add("| arm | κ bin | steps | Δ=1 | Δ=5 | Δ=15 | continuous | micro | macro |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for arm in (*arms_new, JDEC_A):
        for b in grid["request_mix_by_kappa"][arm]["bins"]:
            add(f"| {arm_label(arm)} | {b['kappa'][0]}–{b['kappa'][1]} | {b['steps']} | {pct(b['delta_1'])} | "
                f"{pct(b['delta_5'])} | {pct(b['delta_15'])} | {pct(b['continuous'])} | {pct(b['micro'])} | "
                f"{pct(b['macro'])} |")
    add("")
    add("Late − early shift (κ 1–60 minus κ 166–225, grid; percentage points): ")
    add("")
    add("| arm | Δ=1 | Δ=5 | Δ=15 | continuous | micro | macro |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for arm in (*arms_new, JDEC_A):
        s = grid["request_mix_by_kappa"][arm]["late_minus_early"]
        add(f"| {arm_label(arm)} | " + " | ".join("n/a" if s[k] is None else f"{100*s[k]:+.1f}"
                                                  for k in ("delta_1", "delta_5", "delta_15", "continuous", "micro",
                                                            "macro")) + " |")
    add("")
    add("## Label accuracy by κ (L225 round-0 labels, seed means)")
    add("")
    add("| arm | " + " | ".join(f"κ {lo}–{hi}" for lo, hi in ri.KAPPA_BINS) + " |")
    add("| --- | " + " | ".join("---" for _ in ri.KAPPA_BINS) + " |")
    for arm in (*arms_new, JDEC_A):
        add(f"| {arm_label(arm)} | " + " | ".join(pct(b["accuracy"]) for b in compute["label_accuracy"][arm]["L225_by_kappa"])
            + " |")
    add("")
    add("## Wrong-anchor control (grid, 480 fit anchors)")
    add("")
    add("| arm | right | wrong | right − wrong | rank stability | token |")
    add("| --- | --- | --- | --- | --- | --- |")
    for arm in arms_new:
        w = compute["wrong_anchor"][arm]
        add(f"| {arm_label(arm)} | {cell(w['right'])} | {cell(w['wrong'])} | {cell(w['right_minus_wrong'])} | "
            f"{cell(w['stability'])} | {w['token']} |")
    add("")
    add("## Guards")
    add("")
    for name, guard in compute["guards"].items():
        add(f"- {name}: {'pass' if guard['pass'] else 'FAIL'}")
    g5 = compute["guards"]["G5_replication_101_J"]
    add(f"  - G5 detail: labels {g5['labels']}, round checkpoints {g5['checkpoints']}, policy {g5['policy']}, "
        f"{g5['blocks']} scored blocks {g5['records']}.")
    add(f"- Typed failures: {', '.join(compute['typed_failures']) if compute['typed_failures'] else 'none'}.")
    add("")
    add("## Compute")
    add("")
    c = compute["compute"]
    add(f"- GPU {c['gpu_seconds']:.0f} s, process wall {c['wall_seconds']:.0f} s, engine 0 s ({c['hardware']}).")
    w = compute["windows_225"]
    add(f"- 225-frame label windows: {w['windows']} shots, {w['reaching_225']} reach frame 225 (mean length "
        f"{w['mean_length']:.1f}, min {w['min_length']}), {w['label_rows']} round-0 label rows per group.")
    add("")
    return "\n".join(lines)


def summary(plan, compute):
    grid = compute["inventories"][PRIMARY]
    star = compute["primary"]["v_star"]
    arms = (*PE_ARMS, dec_arm(star), JDEC_A)
    return {"schema": "issue_119_summary_v1", "identity": IDENTITY, "issue": 119,
            "primary": {k: compute["primary"][k] for k in ("claim", "contrast", "v_star", "estimate", "token", "F_hindsight")},
            "guards_pass": all(g["pass"] for g in compute["guards"].values()),
            "typed_failures": compute["typed_failures"],
            "grid_auc": {arm: grid["rankers"][arm]["auc"]["mean"] if grid["rankers"][arm]["auc"] else None
                         for arm in (*FIXED_ARMS, *arms, "prior")},
            "contrasts": {i: {n: {"estimate": b["estimate"], "token": b["token"]}
                              for n, b in compute["inventories"][i]["contrasts"].items()} for i in INVENTORIES},
            "label_accuracy": {arm: {k: compute["label_accuracy"][arm][k] for k in ("L60", "L225", "own")} for arm in arms},
            "compute": {k: compute["compute"][k] for k in ("gpu_seconds", "wall_seconds", "engine_seconds")}}


def published(output, plan, compute):
    selection = load_selection(output)
    return {"compute.json": json_text(compute), "summary.json": json_text(summary(plan, compute)),
            "findings.md": findings_md(plan, compute),
            "records-manifest.json": json_text(record_manifest(output, selection))}


# ---------------------------------------------------------------------------
# smoke (pre-freeze; training data and #101 artefacts only)
# ---------------------------------------------------------------------------

def replication_smoke(output):
    """Train variant a (L60) seed 20260908 through this runner's training path into a scratch tree and
    compare labels, round checkpoints, the final policy and its policy-split evaluation with #101."""
    scratch = Path(output) / "smoke-replication"
    if scratch.exists():
        shutil.rmtree(scratch)
    group = ("pe", "a", SHORT, SEEDS[0])
    report, wall = train_group(scratch, group)
    theirs_root = dap.group_root(P101, "pe", LAMBDA_PE, SEEDS[0])
    mine_root = group_root(scratch, group)
    same = {}
    for r in (0, 1):
        mine = torch.load(mine_root / f"labels-{r}.pt", map_location="cpu", weights_only=True)["units"]
        theirs = torch.load(theirs_root / f"labels-{r}.pt", map_location="cpu", weights_only=True)["units"]
        same[f"labels_{r}"] = len(mine) == len(theirs) and all(
            torch.equal(a[k], b[k]) for a, b in zip(mine, theirs) for k in ("z", "action", "remaining", "labels"))
        mine = torch.load(mine_root / f"round-{r}.pt", map_location="cpu", weights_only=True)
        theirs = torch.load(theirs_root / f"round-{r}.pt", map_location="cpu", weights_only=True)
        same[f"round_{r}_state"] = all(torch.equal(mine["model"][k], theirs["model"][k]) for k in theirs["model"])
        same[f"round_{r}_losses"] = mine["losses"] == theirs["losses"]
    mine = torch.load(policy_path(scratch, group), map_location="cpu", weights_only=True)["model"]
    theirs = torch.load(dap.policy_path(P101, "pe", LAMBDA_PE, SEEDS[0]), map_location="cpu", weights_only=True)["model"]
    same["policy_state"] = all(torch.equal(mine[k], theirs[k]) for k in theirs)
    same["policy_split_evaluation"] = (report["policy_split"]
                                       == read_json(dap.report_path(P101, "pe", LAMBDA_PE, SEEDS[0]))["policy_split"])
    shutil.rmtree(scratch)
    return {"control": "variant_a_reproduces_101_J_training", "seed": SEEDS[0], "checks": same, "wall_seconds": wall,
            "ok": all(same.values())}


@torch.no_grad()
def input_checks(model):
    """Feature values of a-d, ElapsedRequestPolicy forward and init, the elapsed argument reaching the
    policy in the scoring rollout (= position) and in the window DAgger (= frame since window start)."""
    out = {}
    remaining = torch.tensor([1, 30, 60, 61, 165, 225])
    like = torch.zeros(1)
    features = {v: rp.REMAINING_FEATURES[ri.VARIANTS[v]["remaining_feature"]][1](remaining, like).flatten().tolist()
                for v in ri.VARIANTS}
    out["features"] = {v: [round(x, 6) for x in f] for v, f in features.items()}
    out["b_equals_min_form"] = bool(torch.equal(rp.REMAINING_FEATURES["over225"][1](remaining, like).flatten(),
                                                remaining.clamp(max=225).float()/225))
    a = ri.new_policy("a", PAIRS, SEEDS[0], device=DEVICE)
    reference = rp.new_policy(PAIRS, SEEDS[0], device=DEVICE)
    out["a_init_equals_request_policy"] = all(torch.equal(x, y) for x, y in zip(a.state_dict().values(),
                                                                                  reference.state_dict().values()))
    d = ri.new_policy("d", PAIRS, SEEDS[0], device=DEVICE)
    z = torch.randn(6, 236, device=DEVICE)
    act = torch.randn(6, 5, device=DEVICE)
    rem = remaining.to(DEVICE)
    out["d_default_elapsed"] = bool(torch.equal(d(z, act, rem), d(z, act, rem, ri.ENDPOINT-rem)))
    seen = []
    hook = d.register_forward_hook(lambda module, inputs, output: seen.append(
        (inputs[2].clone(), inputs[3].clone() if len(inputs) > 3 and inputs[3] is not None else None)))

    fit_cells = gb.mixed_cells(gb.inventory_entries(rt.fit_levels(), rt.complete_branches(), rt.fit_verdicts(), PRIMARY))
    member, entry = sorted(fit_cells.items())[0]
    z0 = gb.fit_anchors()[member].to(DEVICE)
    actions = m99.action_batch(entry["inventory"])
    _, timing = rp.rollout_endpoints(model, d, z0, actions, {"kind": "joint"}, select=t96.select_pairs,
                                     record_schedules=True)
    positions = []
    for schedule in timing["schedules"]:
        position, steps = 0, []
        for index in schedule:
            steps.append(position)
            position += PAIRS[index].delta
        positions.append(steps)
    out["rollout_remaining_is_endpoint_minus_position"] = bool(seen) and all(
        e is None and sorted((ri.ENDPOINT-r).tolist()) == sorted(p[j] for p in positions if j < len(p))
        for j, (r, e) in enumerate(seen))
    seen.clear()
    shot = dap.read_shot(P101, dap.load_pool(P101)["shot_keys"][0])
    window = rp.first_window(shot["carriers"], shot["action"], shot["end"], LONG)
    rows = ri.window_rows(model, d, [window], aggregated=True, device=DEVICE)
    hook.remove()
    out["dagger_elapsed_is_window_frame"] = bool(seen) and all(
        e is not None and int(e) + int(r) == int(window["length"]) for r, e in seen)
    out["window_rows_elapsed_column"] = bool(torch.equal(rows["elapsed"] + rows["remaining"],
                                                          torch.full_like(rows["remaining"], int(window["length"]))))
    plain = ri.window_rows(model, a, [window], aggregated=True, device=DEVICE)
    reference_rows = rp.window_rows(model, a, [window], aggregated=True, device=DEVICE)
    out["abc_rows_equal_request_policy"] = all(torch.equal(plain[k], reference_rows[k]) for k in reference_rows)
    ok = all(v for k, v in out.items() if k != "features")
    return {"control": "input_variants", **out, "ok": bool(ok)}


def long_window_timing(model, output):
    """Both label rounds on the 225-frame windows of the first policy levels for variants b and d
    (random-init policies; timing and label mix), and the 60-frame rounds for comparison."""
    pool = dap.load_pool(P101)
    units = dap.window_units(pool)[:SMOKE_UNITS]
    by_member = {}
    for key, member in zip(pool["shot_keys"], pool["shot_members"]):
        by_member.setdefault(member, []).append(key)
    members = [m for m in pool["levels"] if m in by_member][:SMOKE_UNITS]
    long_units = []
    for m in members:
        windows = []
        for key in by_member[m]:
            shot = dap.read_shot(P101, key)
            windows.append(rp.first_window(shot["carriers"], shot["action"], shot["end"], LONG))
        long_units.append(windows)
    out = {"units": len(long_units), "windows": sum(len(u) for u in long_units)}
    for variant in ("b", "d"):
        policy = ri.new_policy(variant, PAIRS, SEEDS[0], device=DEVICE)
        for window, unit_list in (("L60", units), ("L225", long_units)):
            for r in (0, 1):
                torch.cuda.synchronize()
                t0 = time.monotonic()
                rows = [ri.window_rows(model, policy, u, aggregated=bool(r), device=DEVICE) for u in unit_list]
                torch.cuda.synchronize()
                labels = torch.cat([x["labels"] for x in rows])
                out[f"{variant}_{window}_round{r}"] = {
                    "seconds_per_window": (time.monotonic()-t0)/out["windows"], "rows": int(len(labels)),
                    "label_counts": torch.bincount(labels, minlength=len(PAIRS)).tolist()}
    return out


def shot_ends():
    pool = dap.load_pool(P101)
    ends = torch.tensor(pool["shot_ends"])
    lengths = ends.clamp(max=LONG)
    return {"shots": int(len(ends)), "reaching_225": int((lengths == LONG).sum()),
            "mean_length": float(lengths.float().mean()), "min_length": int(lengths.min())}


def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the smoke is a pre-freeze control")
    output.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    checks = [replication_smoke(output)]
    with harness.deterministic_scoring():
        model = dap.load_predictor(SEEDS[0])
        checks.append(input_checks(model))
        timing = long_window_timing(model, output)
    accounting = dap.pool_accounting()
    checks.append({"control": "policy_pool_accounting", **accounting,
                   "ok": bool(accounting["levels"] == 120 and accounting["complete"] == 3480
                              and accounting["disjoint_from_fit_and_evaluation"])})
    store = gb.load_anchors(GATE_B)
    gate = read_json(GATE_B / "summary.json")
    checks.append({"control": "gate_b_inputs_bound", "gate_b": gate["gate_b"], "anchors": len(store["anchors"]),
                   "ok": bool(gate["gate_b"] == "pass" and store["scoring_policy"] == harness.SCORING_POLICY)})
    reused = reused_record_check()
    checks.append({"control": "issue101_records_match_manifest", **reused, "ok": reused["pass"]})
    selection = read_json(P101 / "selection.json")["kinds"]
    checks.append({"control": "issue101_jdec_lambda", "lambda": selection["dec"]["lambda"],
                   "ok": selection["dec"]["lambda"] == LAMBDA_DEC and selection["dec"]["arm"] == REFERENCE_JDEC})
    ends = shot_ends()
    ok = all(c["ok"] for c in checks)
    per_group = read_json(dap.report_path(P101, "pe", LAMBDA_PE, SEEDS[0]))["wall_seconds"]
    jdec_group = read_json(dap.report_path(P101, "dec", LAMBDA_DEC, SEEDS[0]))["wall_seconds"]
    ratio = (timing["b_L225_round1"]["seconds_per_window"]+timing["b_L225_round0"]["seconds_per_window"]) / (
        timing["b_L60_round1"]["seconds_per_window"]+timing["b_L60_round0"]["seconds_per_window"])
    projection = {"issue101_pe_group_seconds": per_group, "issue101_dec_group_seconds": jdec_group,
                  "long_to_short_label_time_ratio": ratio,
                  "training_gpu_seconds": len(SEEDS)*(4*per_group+2*per_group*ratio+jdec_group),
                  "accuracy_gpu_seconds_estimate": 24*60.0,
                  "scoring_gpu_seconds_estimate": 7*len(SEEDS)*(130.0+15.0)}
    projection["total_gpu_seconds"] = (projection["training_gpu_seconds"]+projection["accuracy_gpu_seconds_estimate"]
                                       + projection["scoring_gpu_seconds_estimate"])
    projection["elapsed_hours_on_4_gpus"] = projection["total_gpu_seconds"]/4/3600
    wall = time.monotonic()-began
    evidence = {"schema": SCHEMA_SMOKE, "identity": IDENTITY, "run_at": utc_now(), "checks": checks,
                "label_window_timing": timing, "shot_lengths_225": ends, "pool_accounting": accounting,
                "projection": projection, "wall_seconds": wall,
                "reading": ("#101 artefacts (pool, shots, J policy, labels, records), fit-split anchors and cells, "
                            "policy-split windows (training data); no evaluation-split model output"), "ok": ok}
    write_json(output / "smoke.json", evidence)
    log(f"smoke {'PASSED' if ok else 'FAILED'} ({wall:.0f}s); projection {projection['elapsed_hours_on_4_gpus']:.2f} h "
        "on 4 GPUs")
    for check in checks:
        log(f"  {check['control']}: {check['ok']}")
    if not ok:
        raise ValueError("smoke control differs; STOP RULE: no freeze")
    return 0


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def input_bindings(output):
    names = {
        "issue_112_retrain_handoff": gb.HANDOFF, "issue_112_fit_verdicts": gb.FIT_VERDICTS,
        "issue_112_fit_anchors": rt.anchors_path(rt.OUTPUT),
        "gate_b_plan": GATE_B / "plan.json", "gate_b_summary": GATE_B / "summary.json",
        "gate_b_anchors": gb.anchors_path(GATE_B),
        "issue_101_plan": P101 / "plan.json", "issue_101_compute": P101 / "compute.json",
        "issue_101_selection": P101 / "selection.json", "issue_101_records_manifest": P101 / "records-manifest.json",
        "issue_101_pool": dap.pool_path(P101),
        **{f"issue_101_{kind}_{dap.tag(lam)}_{s}_{what}": path
           for kind, lam in (("pe", LAMBDA_PE), ("dec", LAMBDA_DEC)) for s in SEEDS
           for what, path in (("policy", dap.policy_path(P101, kind, lam, s)),
                              ("report", dap.report_path(P101, kind, lam, s)),
                              *((f"labels_{r}", dap.group_root(P101, kind, lam, s) / f"labels-{r}.pt") for r in (0, 1)),
                              *((f"round_{r}", dap.group_root(P101, kind, lam, s) / f"round-{r}.pt") for r in (0, 1)))},
        **{f"issue_101_record_{arm}_{s}": dap.record_path(P101, arm, s) for arm in reused_names() for s in SEEDS},
        "issue_101_runner": ROOT / dap.RUNNER, "gate_b_runner": ROOT / gb.RUNNER,
        "issue_112_retrain_runner": ROOT / rt.RUNNER, "issue_112_v1_runner": ROOT / v1.RUNNER,
        "issue_99_attribution_runner": ROOT / "scripts/run_decision_chain_attribution.py",
        "issue_96_runner": ROOT / "scripts/run_tau_ad_within_checkpoint.py",
        "issue_109_cohort_plan": rt.COHORT, "issue_104_campaign_branches": dap.CAMPAIGN / "branches.json",
        "scoring_harness": ROOT / "world_model/planning/scoring_harness.py",
        "request_policy_module": ROOT / dap.MODULE, "remaining_input_module": ROOT / MODULE,
        "matched_dynamics": ROOT / "world_model/training/matched_dynamics.py",
        "smoke": Path(output) / "smoke.json",
    }
    names.update({f"checkpoint_unit_{s}": gb.member_path("unit", s) for s in SEEDS})
    return [{"name": name, "artifact": str(Path(path).relative_to(ROOT)), "sha256": sha256_of(path)}
            for name, path in names.items()]


CLAIM = ("the kappa cap costs decision value: an unsaturated remaining-frames input improves J - F_hindsight by >= "
         "delta on the primary inventory")


def frozen_rules():
    return {
        "claim": CLAIM,
        "selection": ("v* in {b, c, d}: the input variant trained on the first 60-frame windows (pe, lambda 1e-4) with "
                      "the largest seed-mean policy-split pooled AUC (dap.policy_split_evaluation: greedy J rollouts "
                      "from each policy level's anchor over the mixed grid and angle policy cells; training data only); "
                      "a variant with a typed training failure is ineligible; ties in the order b, c, d; selected after "
                      "every phase-1 group is terminal and before any evaluation score"),
        "primary": (f"(J(v*) - F_hindsight) - (J - F_hindsight) = J(v*, L60) - J(a, L60) paired grid AUC per (level, "
                    f"seed), F_hindsight common to both (the in-inventory best fixed request of the #101 records); "
                    f"supported iff mean >= {DELTA} with lower bound > 0; not_supported_by_this_experiment iff upper < "
                    f"{DELTA}; otherwise {STOP_TOKEN}; any failed guard (including the variant-a replication guard) -> "
                    f"{STOP_TOKEN}"),
        "secondary": ("each of J(b|c|d, L60) - J(a, L60); the label-window factor J(a, L225) - J(a, L60), J(b, L225) - "
                      "J(b, L60), J(b, L225) - J(a, L60), J(b, L225) - J(a, L225) and the interaction (b225 - a225) - "
                      "(b60 - a60); J_dec(v*) - J_dec(a) (#101 J_dec) and J_dec(v*) - J(v*, L60); every new policy and "
                      "J_dec(a) minus F_hindsight and minus the no-model prior; every inventory with the same token "
                      "rule; offset, angle, power DESCRIPTIVE; label accuracy, request mix vs kappa, MACs descriptive"),
    }


def frozen_plan(output, frozen_at, smoke_evidence):
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "issue": 119,
        "related_issues": [77, 96, 101, 104, 108, 109, 111, 112, 114, 116],
        "role": ("remaining-frames input variants and a DP label-window factor for the request policy; one score on "
                 "the sealed #109 evaluation split"),
        "frozen_at": frozen_at, "frozen_before_training_and_scoring": True,
        "validation_command": VALIDATION_COMMAND, "runner": RUNNER, "runner_sha256_at_freeze": sha256_of(ROOT / RUNNER),
        "module": {"path": MODULE, "sha256_at_freeze": sha256_of(ROOT / MODULE)},
        "engine_seconds": 0,
        "question": ("does the request policy rank the sealed evaluation candidates better when it can see how much of "
                     "the 225-frame rollout remains (unsaturated time input), and when its DP labels plan to the "
                     "225-frame endpoint?"),
        "predictors": {"unit": "LC hybrid fold 'all' (Gate-B handoff checkpoints, hash-checked), seeds "
                               f"{list(SEEDS)}; policy seed s uses predictor seed s"},
        "pool": {"source": ("the #101 pool (issue-101 data/pool.pt and shots/): #109 policy split, 120 levels, 3480 "
                            "shots, E99 carriers at positions 0..min(frames - 1, 285); unit = policy level"),
                 "accounting": smoke_evidence["pool_accounting"]},
        "input_variants": {v: s["text"] for v, s in ri.VARIANTS.items()},
        "input_variant_notes": ("a = request_policy clamp60 (ICLR; RequestPolicy unchanged), b = over225 (kappa / 225; "
                                "kappa <= 225 on every path, so = min(kappa, 225) / 225), c = log1p, d = "
                                "remaining_input.ElapsedRequestPolicy (243 -> 128 -> 9; elapsed k = frames since the "
                                "window start in training windows, = 225 - kappa in rollouts from the anchor). All "
                                "else identical: width 128, init torch.manual_seed(seed + 7700), request set PAIRS"),
        "label_windows": {
            "L60": "the #101 first 60-frame window of every pool shot (rp.first_window(..., 60)), as #101 J",
            "L225": ("the start-0 window reaching the 225-frame endpoint of every pool shot (rp.first_window(..., 225); "
                     "length min(225, shot end), terminal absorption), so input range and label range match. The #112 "
                     "long-horizon supervision windows are predictor-training windows on the fit split (predictor "
                     "only); the policy-split counterpart is the same E99 carrier stream to the 225-frame endpoint "
                     "(as #116 Work 3)"),
            "shot_lengths": smoke_evidence["shot_lengths_225"]},
        "factor_design": ("input variants a, b, c, d on L60 (Work 1); the label-window factor is a separate pre-declared "
                          "2 x 2: input {a, b} x labels {L60, L225} (cells a-L60 and b-L60 shared with Work 1)"),
        "training": {"labels": ("#77 prediction-error DP (request_policy.prediction_error_labels via remaining_input."
                                "window_rows), lambda 1e-4 (ICLR), round 1 on DAgger contexts of the round-0 policy"),
                     "recipe": {**rp.DISTIL_RECIPE, "unit": "policy level, step s -> unit s % 120",
                                "sampler": "CPU Generator(seed + 7700 + round)", "loss": "cross-entropy"},
                     "groups": [f"{pe_arm(v, w)}/{s}" for _, v, w, s in phase1_groups()],
                     "typed_failure": "a training error is recorded as a typed terminal failure, never retried"},
        "decision_arm": {"arm": "J_dec(v*)", "labels": (
            "the #101 decision-valued labels (request_policy.decision_rows, lambda 1e-1 = #101's selected J_dec, "
            f"continuation {dap.BASE_CONTINUATION} in round 0 and the round-0 policy in the DAgger round, softplus "
            "pairwise ranking loss T = 0.1, the #101 mixed policy cells); same recipe; elapsed column = rollout "
            "position"), "reference": f"#101 {REFERENCE_JDEC} (variant a under the same labels)",
            "rule": "one pre-declared arm, not a search: v* from the selection rule"},
        "arms": {"new": [*PE_ARMS, "Jdec-<v*>"], "reused_101": [*FIXED_ARMS, f"{JDEC_A} = {REFERENCE_JDEC}"],
                 "replication_target": f"#101 {REFERENCE_J} (J)",
                 "F_hindsight": "in-inventory best fixed request by member-mean AUC (outcome-selected contrast target)",
                 "controls": ("no-model action prior (Gate-B rule, fit-verdicts.json); wrong-anchor control of every new "
                              "policy (480 fit anchors x grid candidates)")},
        "metrics": {
            "auc": "tie-free cost (EndpointCosts.row against the anchor), m99.view per (level, seed); bootstrap over levels",
            "label_accuracy": ("argmax agreement on the policy split (training data): L60 / L225 round-0 labels of the "
                               "variant-a groups of the same seed, the policy's own aggregated rows, and per kappa bin "
                               f"{[list(b) for b in ri.KAPPA_BINS]} of the L225 rows"),
            "request_mix": (f"per kappa bin {[list(b) for b in ri.KAPPA_BINS]} of the true remaining frames at each "
                            "policy step of the scored rollouts: Delta and alpha shares; late (kappa 1-60) minus early "
                            "(kappa 166-225) shift"),
            "macs": "linear MACs per candidate rollout (predictor calls + policy calls) and policy MACs per call (state)"},
        "evaluation": {"split": "the #109 evaluation split as in Gate B (stream issue109-evaluation, mixed cells)",
                       "anchors": "the Gate-B anchor store", "primary_inventory": PRIMARY,
                       "descriptive": list(gb.DESCRIPTIVE_INVENTORIES),
                       "reading": "confirmatory (Gate B passed): the #104 sealed #109 evaluation split, scored once"},
        "decision_rule": frozen_rules(),
        "guards": {"G1_isolation": "pool = policy roster, disjoint from fit and evaluation rosters; windows = pool shots",
                   "G2_scoring_policy": "every new record carries the #111 SCORING_POLICY",
                   "G3_chronology": ("frozen_at <= phase-1 training <= selection <= J_dec training; all training <= "
                                     "first score"),
                   "G4_checkpoint_binding": "every scored policy's sha256 equals its training report",
                   "G5_replication_101_J": ("variant a (L60) equals the #101 J bit for bit: both label rounds, both round "
                                            "checkpoints, the final policy, and every scored block (rows, schedules, "
                                            "calls, MACs)"),
                   "G6_reused_101_records": "every reused #101 record matches the #101 records-manifest",
                   "G7_selection": "selection.json equals the selection rule re-derived from the training reports",
                   "failure": f"a failed guard turns every token into {STOP_TOKEN}"},
        "prohibited": ["reading evaluation captures for training or selection", "re-scoring after publication",
                       "changing a variant, window, lambda, rule or arm after the freeze",
                       "retrying or replacing a typed failure"],
        "caps": {"gpu_seconds": GPU_CAP_SECONDS, "wall_seconds_summed_over_processes": WALL_CAP_SECONDS,
                 "projection": smoke_evidence["projection"],
                 "scope_rule": "if the projection exceeds 48 h elapsed on 4 GPUs, stop blocked with the projection"},
        "disclosure": ("Known before the freeze: Gate B (unit 0.7444, C* 0.7834, prior 0.7203), the #101 results (J grid "
                       "0.706, J_dec 0.746, F-15-continuous 0.762 = F_hindsight, J - F_hindsight -0.057), #114, #116 "
                       "(R_9 pe DP on 225-frame policy windows) and every #96/#100/#112 result. No policy with input "
                       "b, c or d and no policy on L225 labels has been scored on an evaluation capture. Hardware: RTX "
                       "5090 instead of the #85 RTX 3090. Variant d's MAC term counts its 243-input policy (128 more "
                       "MACs per call), so its labels can differ from a's by that term."),
        "smoke_evidence": smoke_evidence,
        "inputs": input_bindings(output),
        "claim_boundary": ("one look at the sealed #109 evaluation split for policies trained on the #109 policy split; "
                           "grid ranking primary, other inventories descriptive; tie-free cost; DESCRIPTIVE intervals; "
                           "no closed-loop claim"),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-119 protocol")
    if not plan.get("frozen_before_training_and_scoring") or not plan["smoke_evidence"].get("ok"):
        raise ValueError("plan.json does not declare a passed-smoke freeze")
    for entry in plan["inputs"]:
        if not amendments.matches(entry["artifact"], entry["sha256"], sha256_of(ROOT / entry["artifact"])):
            raise ValueError(f"frozen input {entry['name']} changed after the freeze")
    return plan


def prepare(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        load_plan(output)
        log("existing frozen plan validated")
        return 0
    for name in ("records", "models", "data", "accuracy"):
        if (output / name).exists():
            raise ValueError(f"{name} exist before the freeze; refusing")
    evidence = read_json(output / "smoke.json")
    if not evidence.get("ok"):
        raise ValueError("smoke failed; STOP RULE: no freeze")
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
    for name, text in published(output, plan, compute).items():
        (output / name).write_text(text)
    log(f"published: primary {compute['primary']['token']} ({fmt_interval(compute['primary']['estimate'])})")
    return 0


def spot_checks(output):
    """Re-score J(v*, L60) of the first seed on every grid cell (exact rows and schedules) and recompute
    its label-accuracy record."""
    selection = load_selection(output)
    seed = SEEDS[0]
    arm = pe_arm(selection["v_star"], SHORT)
    group = arm_group(arm, seed)
    record = read_record(output, arm, seed)
    _, cells = gb.evaluation_set()
    anchors = gb.load_anchors(GATE_B)["anchors"]
    costs = dap.endpoint_costs()
    same = True
    with harness.deterministic_scoring():
        model = dap.load_predictor(seed)
        policy = load_policy(output, group)
        for member, entry in sorted(cells[PRIMARY]["mixed"].items()):
            fresh = dap.block(model, policy, {"kind": "joint"}, anchors[member], entry["inventory"], costs, True)
            stored = record["inventories"][PRIMARY][member]
            same &= fresh["rows"] == stored["rows"] and fresh["schedules"] == stored["schedules"]
        stored = read_accuracy(output, arm, seed)
        fresh = accuracy_record(output, arm, seed)
        same &= all(fresh[k] == stored[k] for k in fresh)
    return {"arm": arm, "seed": seed, "pass": bool(same)}


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    compute = compute_tables(output, plan)
    for name, text in published(output, plan, compute).items():
        if (output / name).read_bytes() != text.encode():
            raise ValueError(f"published {name} differs from the recomputation")
    spot = spot_checks(output)
    if not spot["pass"]:
        raise ValueError(f"deterministic re-scoring does not replicate the retained records: {spot}")
    log(f"validation passed ({time.monotonic()-began:.0f}s): tables and the record manifest re-derived and "
        f"byte-compared, selection re-derived (G7), {spot['arm']} seed {spot['seed']} re-scored and its label "
        f"accuracy recomputed exactly; primary {compute['primary']['token']}")
    return 0


def dry_run(output):
    log(f"output {Path(output)}; pool {dap.pool_accounting()}")
    log(f"phase-1 groups {len(phase1_groups())}; J_dec groups {len(SEEDS)}; arms {len(PE_ARMS) + 1} x seeds "
        f"{len(SEEDS)} + wrong-anchor")
    return 0


MODES = {"dry-run": dry_run, "smoke": smoke, "prepare": prepare, "assemble": assemble, "train": train,
         "select": select, "train-dec": train_dec, "accuracy": accuracy, "score": score, "publish": publish,
         "validate": validate}
SHARDED = ("train", "train-dec", "accuracy", "score")


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
