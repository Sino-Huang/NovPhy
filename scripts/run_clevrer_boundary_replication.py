"""Issue-79 ADD-EXP: out-of-family replication of the horizon-decay boundary on CLEVRER.

Out-of-family replication test: two arms retrained from scratch on the frozen
CLEVRER annotation-derived carrier (pure continuous vs micro-symbolic hybrid),
three paired seeds, 9000 identical updates. Estimands are recursive state
errors at the frozen endpoint grid, training-effect contrasts per horizon, the
micro symbolic-execution effect within the hybrid checkpoint, and the
contact-activity regime analysis. Planning/ranking regret is NOT an estimand on
CLEVRER (no candidate inventory); the macro arm is unsupported (Phase 0) and
its estimand is dropped. Descriptive paired-bootstrap intervals only.

Modes (mutually exclusive): --dry-run (no writes), --smoke-test (bounded real
one-scene pass, CPU, temporary weights), --prepare (download checks, bounded
parsing, Phase-0 freeze), --train (GPU lock held for the whole phase),
--run-evaluation (GPU lock held for the whole phase), --publish, --validate.
Exact validation command: python -u -m scripts.run_clevrer_boundary_replication --validate
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import fcntl
import hashlib
import io
import json
from pathlib import Path
import resource
import subprocess
import tempfile
import time
from collections import Counter

import numpy as np
import torch

from world_model.model import Abstraction, PredictionPair
from world_model.training import clevrer_boundary as clevrer
from world_model.training.cnn_hybrid import DIM, PAIRS, CNNHybridPredictor, linear_macs, training_loss
from world_model.training.matched_dynamics import (
    ContinuousDynamics, active_capacity, capacity_contract, continuous_loss,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".local-artifacts/issue-79-clevrer-boundary-v1"
DATA = ROOT / "data/clevrer"
ANNOTATIONS = DATA / "annotations"
ZIPS = {"train": ANNOTATIONS / "annotation_train.zip",
        "validation": ANNOTATIONS / "annotation_validation.zip"}
ZIP_FACTS = {
    "train": {"bytes": 156761889,
              "sha256": "e7f9185ad3dde8139b839b00f0f75483374dd0325a3cbec2dac882078497784c"},
    "validation": {"bytes": 78386966,
                   "sha256": "d89e74714dff7a1117fd113b54af99fbbc51583ebd742c788c484d708eb426d2"},
}
SEEDS = (20260908, 20260909, 20260910)
ARMS = ("continuous", "hybrid")
PAIRS_CLEVRER = tuple(p for p in PAIRS if p.abstraction != Abstraction.MACRO)
SYSTEMS = {f"continuous_h{h}": ("continuous", PredictionPair(h, Abstraction.CONTINUOUS))
           for h in clevrer.HORIZONS}
SYSTEMS.update({f"hybrid_continuous_h{h}": ("hybrid", PredictionPair(h, Abstraction.CONTINUOUS))
                for h in clevrer.HORIZONS})
SYSTEMS.update({f"hybrid_micro_h{h}": ("hybrid", PredictionPair(h, Abstraction.MICRO))
                for h in clevrer.HORIZONS})
SCHEMA = "issue_79_clevrer_boundary_v1"
IDENTITY = "issue-79-clevrer-boundary-v1"
FILES = ("scripts/run_clevrer_boundary_replication.py", "world_model/training/clevrer_boundary.py")
GPU_LOCK = Path("/tmp/novphy-addexp-gpu.lock")
GPU_HOURS_ALLOWANCE = 6.0
ARTIFACTS_CAP_GIB = 5
REGIME_MINIMUM_UNITS = 8
REGIME_WINDOW = 30
RELATIVE_MARGIN = .05
UNTRAINED_HYBRID_PARAMETERS = ("macro_head", "macro_adapter")
DISPOSITION_TOKENS = ("supported", "not_supported_by_this_experiment", "readiness_or_precision_insufficient")


def log(message):
    print(f"[issue-79] {message}", flush=True)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def memory(device):
    return {"peak_cpu_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 2 ** 20
            if str(device).startswith("cuda") else 0.}


@contextlib.contextmanager
def gpu_lock():
    """Exclusive single-GPU lock for wall-time-measured phases (open once, hold)."""
    handle = open(GPU_LOCK, "a+b")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield handle
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def paired_interval(values):
    """Paired bootstrap over paired units; DESCRIPTIVE (issue-72 convention, rng 7201)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(7201)
    draws = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return {"mean": float(values.mean()), "paired_units": int(len(values)),
            "descriptive_95_percent_interval": np.quantile(draws, [.025, .975]).tolist()}


# ---------------------------------------------------------------- plan


def data_path(args):
    return args.output / "clevrer-data.json"


def load_data(args):
    data = read(data_path(args))
    if data["schema"] != "issue_79_clevrer_data_v1":
        raise ValueError("clevrer derived-data schema differs")
    if data["zip_facts"] != ZIP_FACTS:
        raise ValueError("CLEVRER annotation archives differ from the verified release")
    return data


def make_plan(args):
    clevrer.check_eval_grid()
    if not data_path(args).exists():
        raise ValueError("requires --prepare: no frozen CLEVRER derived data (Phase 0 blocks training)")
    data = load_data(args)
    capacity = capacity_contract()
    return {
        "schema": SCHEMA, "identity": IDENTITY,
        "category": "ADD-EXP out-of-family replication of the horizon-decay boundary on CLEVRER (issue #79)",
        "data": {
            "source": "official CLEVRER release, data.csail.mit.edu/clevrer, CC0 license",
            "annotation_zips": {name: {"path": str(path.resolve()), **ZIP_FACTS[name]}
                                for name, path in ZIPS.items()},
            "videos_downloaded": False,
            "splits": "train 0-9999 and validation 10000-14999 ship annotations; test ships none",
            "clips": "128 frames, 5 s, 480x320; membership uses validation annotations only",
        },
        "representation": {
            "identity": clevrer.IDENTITY,
            "carrier": "unchanged issue-71 236-value layout: 2 global values + 18 slots x 13 columns",
            "slot_columns": [list(pair) for pair in clevrer.SLOT_COLUMNS],
            "slot_contract": clevrer.SLOT_CONTRACT,
            "kind_vocabulary": list(clevrer.KIND_VOCABULARY),
            "position_scale": clevrer.POS_SCALE,
            "velocity_scale": data["velocity_scale"],
            "speed_tier_edges": data["speed_tier_edges"],
            "prior_elapsed_seconds": clevrer.PRIOR_ELAPSED_SECONDS,
            "null_action": list(clevrer.NULL_ACTION),
            "availability_mask": "inside_camera_view in columns 7 and 10; unavailable fields zeroed; all masked estimands",
            "micro_predicates": {
                "contact": "collision events (frame_id, object_ids) label symmetric channel 0",
                "velocity_bin": f"same speed tier at frozen edges {data['speed_tier_edges']}, symmetric channel 1"},
            "macro_status": "unsupported",
            "macro_note": ("no supports/structure-unstable referent on CLEVRER; macro pairs are never trained"
                           " and never evaluated; the macro estimand is DROPPED, no invented vocabulary"),
            "unmapped_fields": ("vertical location/velocity, orientation matrices and angular velocities are parsed"
                                " into per-scene shards for the parser-contract record with no carrier column referent"),
            "parser_symmetry": ("identical frozen annotation-to-carrier builder for both arms; engine features in,"
                                " predicates out; no learned perception on either side"),
        },
        "membership": data["membership"],
        "capacity": capacity,
        "design": {
            "arms": {
                "continuous": f"ContinuousDynamics at frozen width {capacity['continuous_width']}; no symbolic modules",
                "hybrid": ("CNNHybridPredictor port; micro contact+velocity-bin symbolic conditioning; macro head and"
                           " adapter receive no CLEVRER gradient and are never evaluated (unsupported Phase 0)")},
            "seeds": list(SEEDS),
            "training": {
                "steps": 9000, "batch_size": 64, "learning_rate": .0001, "weight_decay": .0001, "grad_clip": 1.,
                "unroll": 4,
                "pair_cycle": ("step %% %d over frozen issue-71 PAIRS order minus macro: %s; the continuous arm uses"
                               " the same delta with continuous abstraction" %
                               (len(PAIRS_CLEVRER), [list(p.identity) for p in PAIRS_CLEVRER])),
                "identical_minibatches": ("both arms of one seed share the generator seed and the frozen window order;"
                                          " windows draw with one per-seed generator exactly as issue-74"),
                "objective": {
                    "retained": ["local carrier MSE at four requested-horizon transitions",
                                 "recursive MSE along the unrolled carrier",
                                 "0.01 carrier bound penalty relu(|z|-2)^2 (issue-74 weight)"],
                    "hybrid_symbolic": ("0.1 x micro symbolic losses: pos-weight-5 relations BCE over the contact and"
                                        " velocity-bin channels plus availability BCE, at predicted and observed contexts"),
                    "dropped": ("none; the issue-74 objective contains no action-dependent penalty term; the action"
                                " enters only as the constant null-action conditioning input held for both arms")},
                "selection": "last update, no best-validation checkpoint"},
            "endpoint_grid": {
                "endpoints": list(clevrer.ENDPOINTS), "horizons": list(clevrer.HORIZONS),
                "contexts": list(clevrer.EVAL_CONTEXTS),
                "divisibility_rule": ("every endpoint divisible by all horizons and context+endpoint within the"
                                      " 128-frame clip; the NovPhy {15..600} grid does not transfer because 600"
                                      " exceeds the clip; divisibility is the freezing rule")},
            "systems": list(SYSTEMS),
            "estimands": {
                "recursive": ("fixed-pair rollout from the true carrier at each eval context; one shared initial"
                              " carrier, predictions feed the next transition, no truth resets"),
                "metrics": ["position_mse (mask inside_camera_view at target frame)",
                            "velocity_mse (same mask)", "carrier_mse", "presence_mse", "kind_mse"],
                "training_effect": "hybrid fixed continuous pair h vs independently trained continuous arm at h",
                "symbolic_execution": "hybrid fixed micro pair h vs hybrid fixed continuous pair h (same checkpoint)",
                "regime": ("contact-active = at least one collision event inside the early cascade window"
                           f" (context, context+{REGIME_WINDOW}]; strata contrasts of the symbolic-execution effect"
                           f" per horizon and endpoint; minimum {REGIME_MINIMUM_UNITS} units per stratum"),
                "dropped": ("planning/ranking regret is NOT an estimand on CLEVRER (no candidate inventory); no"
                            " shots/success claims; the macro arm is unsupported and its estimand dropped"),
                "zero_shot_pooling": "no zero-shot or adapted conditions exist here; both arms retrained from scratch"},
            "intervals": ("paired bootstrap over paired (scene, context) units, mean-over-seeds reference-minus-tested"
                          " differences, 10000 resamples, rng 7201; DESCRIPTIVE only, no inferential superiority claim"),
            "comparator_disclosure": ("continuous_h5 selection optimism (#74, selection_optimism=true) attaches to the"
                                      " NovPhy family only; here both arms are retrained from scratch with no policy"
                                      " selection and every contrast discloses independent training"),
        },
        "compute": {
            "allowance_gpu_hours": GPU_HOURS_ALLOWANCE,
            "stop_rule": ("if measured training+evaluation wall exceeds 6.0 GPU-hours, stop executing cells, retain"
                          " the remaining cells as typed failures compute_allowance_exceeded, and dispose"
                          " readiness_or_precision_insufficient"),
            "gpu_lock": f"exclusive fcntl.flock on {GPU_LOCK} held for the whole --train and --run-evaluation phases",
            "accounting": ["per-unit rollout wall seconds", "transition step counts per system and endpoint",
                           "declared linear MACs per transition (not FLOPs)", "active parameter counts per arm/pair kind",
                           "training wall seconds per fit"],
            "compute_matching": ("updates, batch size and pair cycle are identical by construction; wall time and"
                                 " active parameters differ (the hybrid executes symbolic heads/adapters) and are"
                                 " recorded, not equalized; inference is NOT equalized"),
            "derived_artifacts_cap_gib": ARTIFACTS_CAP_GIB,
        },
        "decision_rules": {
            "tokens": list(DISPOSITION_TOKENS),
            "q1_signature": {
                "question": "does the NovPhy boundary signature replicate on CLEVRER rollouts",
                "components": [
                    "S1 training-effect position_mse difference (reference minus tested, positive favors hybrid) is positive at h=1, endpoint 15, and exceeds 5% of the continuous-arm mean position_mse at h=1 endpoint 15",
                    "S2 the h=15 training-effect difference is smaller than the h=1 difference at endpoint 15 (separation shrinks or reverses with horizon)",
                    "S3 continuous-arm recursive position_mse grows monotonically across endpoints 15,30,60,120 at every horizon"],
                "disposition_rule": "supported iff S1-S3 all hold; otherwise not_supported_by_this_experiment",
                "incomplete_rule": ("missing cells, typed failures or absent metric values at the rule's endpoints"
                                    " -> readiness_or_precision_insufficient")},
            "q2_regime": {
                "question": "does contact activity predict where the micro symbolic-execution effect changes prediction quality",
                "measure": ("units are strata by the early cascade window: contact-active = at least one collision"
                            f" event inside (context, context+{REGIME_WINDOW}];"
                            " D(endpoint=120, h) = symbolic-execution contrast (reference minus tested, positive"
                            " favors micro) in the contact-active stratum minus the same contrast in the"
                            " contact-still stratum"),
                "direction": "D > 0 means the micro effect is more favorable when early contact is active",
                "practical_margin": ("D must exceed 5% of the contact-still stratum mean position_mse of the hybrid"
                                     " continuous pair at the same h, endpoint 120"),
                "disposition_rule": "supported iff D exceeds the margin at h=5 or h=15; otherwise not_supported_by_this_experiment",
                "precision_rule": f"either stratum below {REGIME_MINIMUM_UNITS} units at endpoint 120 -> readiness_or_precision_insufficient"},
        },
        "claim_boundary": ("mechanism-shape replication only; not 'the negative result generalizes'; a non-replication"
                           " does not falsify the NovPhy boundary and vice versa; the two families are reported"
                           " separately; no CLEVRER leaderboard/VQA or perception claim; no Physion; no new NovPhy"
                           " captures; #64/#65 sealed; prior dispositions unchanged; descriptive intervals only"),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_text": {path: (ROOT / path).read_text() for path in FILES},
        "archived_release": False, "fresh_evaluation_opened": False,
        "final_evaluation_opened": False, "issue_64_authorized": False,
    }


def load_plan(args):
    plan = read(args.output / "plan.json")
    current = make_plan(args)
    current["source_revision"] = plan["source_revision"]
    if plan != current:
        raise ValueError("frozen issue-79 source/settings differ; retain old output and explicitly version any change")
    return plan


# ---------------------------------------------------------------- prepare


def digest(path):
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            sha.update(block)
    return sha.hexdigest()


def verify_zips():
    for name, path in ZIPS.items():
        if not path.exists():
            raise ValueError(f"missing {name} annotation archive at {path}; download from "
                             f"https://data.csail.mit.edu/clevrer/annotations/")
        if path.stat().st_size != ZIP_FACTS[name]["bytes"] or digest(path) != ZIP_FACTS[name]["sha256"]:
            raise ValueError(f"{name} annotation archive does not match the verified release digest")


def prepare(args):
    """Phase 0: bounded parsing, derived facts, membership, and the freeze."""
    verify_zips()
    if (args.output / "plan.json").exists():
        load_plan(args)
        log("frozen plan already present and validated; no re-freeze")
        return
    members, scan = clevrer.select_membership(ZIPS["validation"])
    velocity_scale = clevrer.frozen_velocity_scale(tuple(members))
    tier_edges = list(clevrer.frozen_tier_edges(tuple(members)))
    shards = args.output / "shards"
    membership = []
    for scene in members:
        clip = clevrer.clip_carriers(scene, velocity_scale)
        relations, mask = clevrer.clip_relations(scene, tier_edges)
        windows = clevrer.training_windows(clip, relations, mask)
        atomic_torch(shards / f"scene-{scene.scene_index}.pt", {
            "schema": "issue_79_clevrer_scene_v1", "scene_index": scene.scene_index,
            "video_filename": scene.video_filename, "identity": clevrer.IDENTITY,
            "velocity_scale": velocity_scale, "speed_tier_edges": tier_edges,
            "objects": [dict(obj) for obj in scene.objects],
            "clip": clip, "windows": windows,
            "vertical_location": scene.location[:, :, 2],
            "vertical_velocity": scene.velocity[:, :, 2],
            "orientation": scene.orientation, "angular_velocity": scene.angular_velocity,
            "collision_frames": scene.collision_frames, "collision_pairs": scene.collision_pairs})
        membership.append({"scene_index": scene.scene_index, "video_filename": scene.video_filename,
                           "objects": len(scene.objects), "collisions": int(len(scene.collision_frames)),
                           "object_property": [dict(obj) for obj in scene.objects]})
    data = {"schema": "issue_79_clevrer_data_v1", "identity": "issue-79-clevrer-data-v1",
            "zip_facts": ZIP_FACTS, "velocity_scale": velocity_scale, "speed_tier_edges": tier_edges,
            "membership": {"rule": ("first 12 collision-complete scenes in ascending scene_index within the"
                                    " validation-split annotations; collision-complete = contiguous 128-frame"
                                    " trajectory for every declared object and at least one collision event; no"
                                    " outcome-based selection"),
                           "scenes": membership, "windows_per_scene": clevrer.WINDOWS_PER_SCENE,
                           "training_windows": len(members) * clevrer.WINDOWS_PER_SCENE,
                           "eval_units_per_seed": len(members) * len(clevrer.EVAL_CONTEXTS),
                           "scan": scan}}
    write(data_path(args), data)
    write(args.output / "plan.json", make_plan(args))
    log(f"Phase-0 freeze complete: scenes={[m['scene_index'] for m in membership]} "
        f"velocity_scale={velocity_scale} tier_edges={tier_edges}")


# ---------------------------------------------------------------- shards/pool


def scene_shard(args, plan, scene_index):
    shard = torch.load(args.output / "shards" / f"scene-{scene_index}.pt",
                       map_location="cpu", weights_only=True)
    if shard["schema"] != "issue_79_clevrer_scene_v1" or shard["identity"] != plan["representation"]["identity"] \
            or shard["velocity_scale"] != plan["representation"]["velocity_scale"] \
            or shard["speed_tier_edges"] != plan["representation"]["speed_tier_edges"]:
        raise ValueError(f"scene shard {scene_index} source/representation differs from the frozen plan")
    return shard


_POOLS = {}


def pool(args, plan, arm):
    """Frozen 480-window training pool; identical window order for both arms."""
    if arm in _POOLS:
        return _POOLS[arm]
    stacks = None
    for record in plan["membership"]["scenes"]:
        windows = scene_shard(args, plan, record["scene_index"])["windows"]
        keys = ("z", "action", "length") if arm == "continuous" else \
            ("z", "action", "length", "relations", "relations_mask")
        rows = {k: windows[k] for k in keys}
        stacks = {k: torch.cat([stacks[k], rows[k]]) for k in keys} if stacks else rows
    _POOLS[arm] = stacks
    return stacks


def sample(data, batch_size, generator, device):
    indices = torch.randint(len(data["z"]), (batch_size,), generator=generator)
    return {k: v[indices].to(device) for k, v in data.items()}


def loss_for(model, batch, pair):
    if isinstance(model, ContinuousDynamics):
        return continuous_loss(model, batch["z"], batch["action"], batch["length"], pair)
    return training_loss(model, batch, pair, True)


# ---------------------------------------------------------------- training


def cell(args, seed, arm):
    return args.output / f"seed-{seed}" / arm


def new_model(plan, arm):
    return ContinuousDynamics(plan["capacity"]["continuous_width"]) if arm == "continuous" else CNNHybridPredictor()


def binding(plan, seed, arm, component):
    return {"plan_identity": plan["identity"], "capacity": plan["capacity"], "representation": plan["representation"],
            "membership_scenes": [m["scene_index"] for m in plan["membership"]["scenes"]],
            "design": plan["design"], "seed": seed, "arm": arm, "component": component}


def eval_binding(plan, seed, system):
    return {"plan_identity": plan["identity"], "representation_identity": plan["representation"]["identity"],
            "velocity_scale": plan["representation"]["velocity_scale"],
            "speed_tier_edges": plan["representation"]["speed_tier_edges"],
            "membership_scenes": [m["scene_index"] for m in plan["membership"]["scenes"]],
            "training": {"steps": plan["design"]["training"]["steps"], "seeds": plan["design"]["seeds"]},
            "seed": seed, "system": system}


def check_binding(value, expected):
    if value["binding"] != expected:
        raise ValueError("checkpoint/record source/representation/training binding differs")


def expected_pair_counts(plan, arm):
    """Counts implied by the frozen cycle: each cycle slot trains once per pass."""
    steps = plan["design"]["training"]["steps"]
    counts = Counter()
    for step in range(steps):
        selected = PAIRS_CLEVRER[step % len(PAIRS_CLEVRER)]
        pair = PredictionPair(selected.delta, Abstraction.CONTINUOUS) if arm == "continuous" else selected
        counts[str(pair.identity)] += 1
    return dict(counts)


def expected_gradient_parameters(plan, arm):
    """Hybrid macro head/adapter are declared untrained (unsupported Phase 0)."""
    names = {name for name, _ in new_model(plan, arm).named_parameters()}
    if arm == "hybrid":
        names = {name for name in names if not name.startswith(UNTRAINED_HYBRID_PARAMETERS)}
    return names


def measured_gpu_wall(args):
    """Wall seconds already spent on fits and evaluation cells (the stop-rule budget)."""
    total = 0.
    for seed in SEEDS:
        for arm in ARMS:
            progress = cell(args, seed, arm) / "predictor-progress.pt"
            if progress.exists():
                total += float(torch.load(progress, map_location="cpu", weights_only=True)["wall_seconds"])
    if (args.output / "eval").exists():
        for path in sorted((args.output / "eval").rglob("*.pt")):
            total += float(torch.load(path, map_location="cpu", weights_only=True)["wall_seconds"])
    return total


def train_cell(args, plan, seed, arm, stop_after=None):
    """Resume optimizer/sampler exactly; stop_after is for the bounded smoke only."""
    root = cell(args, seed, arm)
    target = root / "predictor.pt"
    if target.exists():
        load_predictor(args, plan, seed, arm)
        log(f"seed={seed} arm={arm} predictor reused")
        return
    torch.manual_seed(seed)
    model = new_model(plan, arm).to(args.device)
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    recipe = plan["design"]["training"]
    optim = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    expected = binding(plan, seed, arm, "predictor")
    start, prior, gradients, counts, saved = 0, 0., set(), Counter(), None
    progress = root / "predictor-progress.pt"
    if progress.exists():
        saved = torch.load(progress, map_location="cpu", weights_only=True)
        check_binding(saved, expected)
        model.load_state_dict(saved["model"], strict=True)
        optim.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"])
        start, prior = saved["step"], saved["wall_seconds"]
        gradients, counts = set(saved["gradient_parameters"]), Counter(saved["pair_counts"])
    data = pool(args, plan, arm)
    began = time.monotonic()
    finish = min(recipe["steps"], stop_after if stop_after is not None else recipe["steps"])
    for step in range(start, finish):
        batch = sample(data, recipe["batch_size"], generator, args.device)
        selected = PAIRS_CLEVRER[step % len(PAIRS_CLEVRER)]
        pair = PredictionPair(selected.delta, Abstraction.CONTINUOUS) if arm == "continuous" else selected
        optim.zero_grad(set_to_none=True)
        loss = loss_for(model, batch, pair)
        if not bool(torch.isfinite(loss)):
            raise ValueError(f"nonfinite training loss seed={seed} arm={arm} step={step + 1}")
        loss.backward()
        gradients.update(n for n, p in model.named_parameters()
                         if p.grad is not None and bool((p.grad != 0).any()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["grad_clip"])
        optim.step()
        counts[str(pair.identity)] += 1
        if (step + 1) % 90 == 0 or step + 1 == finish:
            elapsed = prior + time.monotonic() - began
            saved = {"binding": expected, "model": model.state_dict(), "optimizer": optim.state_dict(),
                     "generator": generator.get_state(), "step": step + 1, "wall_seconds": elapsed,
                     "gradient_parameters": sorted(gradients), "pair_counts": dict(counts),
                     "changed_parameters": [n for n, p in model.named_parameters()
                                            if not torch.equal(p.detach().cpu(), initial[n])],
                     "optimizer_examples": (step + 1) * recipe["batch_size"], "memory": memory(args.device)}
            atomic_torch(progress, saved)
            log(f"train seed={seed} arm={arm} step={step + 1}/{recipe['steps']} "
                f"loss={float(loss.detach()):.6f} elapsed={elapsed:.1f}s "
                f"eta={elapsed / (step + 1) * (recipe['steps'] - step - 1):.1f}s memory={saved['memory']}")
    if finish == recipe["steps"]:
        if saved is None or saved["step"] != recipe["steps"]:
            raise ValueError(f"seed={seed} arm={arm}: training finished without a final checkpoint")
        if set(saved["gradient_parameters"]) != expected_gradient_parameters(plan, arm):
            raise ValueError(f"seed={seed} arm={arm}: gradient coverage differs from the declared trainable set")
        if set(saved["changed_parameters"]) != expected_gradient_parameters(plan, arm):
            raise ValueError(f"seed={seed} arm={arm}: parameters changed outside the declared trainable set")
        if saved["pair_counts"] != expected_pair_counts(plan, arm):
            raise ValueError(f"seed={seed} arm={arm}: pair coverage differs from the frozen cycle")
        atomic_torch(target, saved)


def load_predictor(args, plan, seed, arm):
    value = torch.load(cell(args, seed, arm) / "predictor.pt", map_location="cpu", weights_only=True)
    check_binding(value, binding(plan, seed, arm, "predictor"))
    model = new_model(plan, arm).to(args.device)
    model.load_state_dict(value["model"], strict=True)
    if value["step"] != plan["design"]["training"]["steps"] or value["pair_counts"] != expected_pair_counts(plan, arm) \
            or set(value["gradient_parameters"]) != expected_gradient_parameters(plan, arm) \
            or set(value["changed_parameters"]) != expected_gradient_parameters(plan, arm):
        raise ValueError("incomplete predictor budget/coverage/gradients/updates")
    return model.eval(), value


def train(args, plan):
    """All six fits under one exclusive GPU lock, with the frozen stop rule."""
    if args.device != "cuda":
        raise ValueError("--train measures wall time on the shared GPU; pass --device cuda")
    with gpu_lock():
        torch.cuda.reset_peak_memory_stats(args.device)
        phase_started = time.monotonic()
        for seed in SEEDS:
            for arm in ARMS:
                spent = measured_gpu_wall(args)
                if spent >= plan["compute"]["allowance_gpu_hours"] * 3600:
                    record_stop(args, f"training stop before seed={seed} arm={arm}: "
                                      f"{spent / 3600:.2f} GPU-h measured exceeds the allowance")
                    return
                train_cell(args, plan, seed, arm)
        log(f"training phase complete wall={time.monotonic() - phase_started:.1f}s memory={memory(args.device)}")


STOP = "stop-rule.json"


def record_stop(args, message):
    write(args.output / STOP, {"schema": "issue_79_stop_rule_v1", "reason": "compute_allowance_exceeded",
                               "measured_gpu_wall_seconds": measured_gpu_wall(args),
                               "allowance_gpu_hours": GPU_HOURS_ALLOWANCE, "message": message})
    log(message)


# ---------------------------------------------------------------- evaluation


def eval_path(args, seed, system, scene_index, context):
    return args.output / "eval" / f"seed-{seed}" / system / f"scene-{scene_index}-t{context}.pt"


@torch.no_grad()
def rollout_errors(model, clip, context, pair, device):
    """Fixed-pair recursive rollout from the true carrier; per-endpoint field errors.

    One shared initial carrier; predictions feed the next transition; no truth
    resets. The rollout walks the endpoint grid cumulatively (15->30->60->120).
    """
    carrier = clip[context].to(device)
    trace, steps, wall = {}, 0, 0.
    null = torch.zeros((1, 5), device=device)
    for endpoint in clevrer.ENDPOINTS:
        while steps < endpoint:
            began = time.monotonic()
            carrier = model.carrier(carrier[None], null, pair)[0]
            wall += time.monotonic() - began
            steps += pair.delta
            if not bool(torch.isfinite(carrier).all()):
                raise ValueError("nonfinite recursive carrier")
        if steps != endpoint:
            raise ValueError("fixed pair does not land exactly on the endpoint")
        trace[str(endpoint)] = {"errors": clevrer.field_errors(carrier.cpu(), clip[context + endpoint]),
                                "steps": steps // pair.delta}
    return {"errors": trace, "rollout_wall_seconds": wall, "transition_calls": steps // pair.delta,
            "linear_macs_per_step": linear_macs(model, pair)}


def active_capacity_counts(model, pair, device):
    """Parameters in executed leaf operators for one carrier, counted once per system."""
    return active_capacity(model, torch.zeros((2, DIM), device=device),
                           torch.zeros((2, 5), device=device), pair)


def run_evaluation(args, plan):
    """Every scheduled (seed, system, scene, context) cell; GPU lock held throughout."""
    if args.device != "cuda":
        raise ValueError("--run-evaluation measures wall time on the shared GPU; pass --device cuda")
    with gpu_lock():
        torch.cuda.reset_peak_memory_stats(args.device)
        phase_started = time.monotonic()
        units = [(record["scene_index"], context) for record in plan["membership"]["scenes"]
                 for context in clevrer.EVAL_CONTEXTS]
        active, clips = {}, {}
        total = len(SEEDS) * len(SYSTEMS) * len(units)
        done, failures = 0, Counter()
        budget = plan["compute"]["allowance_gpu_hours"] * 3600
        spent_before_phase = measured_gpu_wall(args)
        phase_wall = 0.
        for seed in SEEDS:
            models = {arm: load_predictor(args, plan, seed, arm)[0] for arm in ARMS}
            for system, (arm, pair) in SYSTEMS.items():
                if (arm, str(pair.identity)) not in active:
                    active[(arm, str(pair.identity))] = active_capacity_counts(models[arm], pair, args.device)
                for scene_index, context in units:
                    if scene_index not in clips:
                        clips[scene_index] = scene_shard(args, plan, scene_index)["clip"]
                    done += 1
                    if spent_before_phase + phase_wall >= budget:
                        record_stop(args, f"evaluation stop before seed={seed} system={system} "
                                          f"scene={scene_index} context={context}: "
                                          f"{(spent_before_phase + phase_wall) / 3600:.2f} GPU-h measured "
                                          f"exceeds the allowance; remaining cells retained as typed failures")
                        return
                    path = eval_path(args, seed, system, scene_index, context)
                    if path.exists():
                        record = torch.load(path, map_location="cpu", weights_only=True)
                        check_binding(record, eval_binding(plan, seed, system))
                        continue
                    began = time.monotonic()
                    try:
                        result, failure = rollout_errors(models[arm], clips[scene_index], context, pair,
                                                         args.device), None
                    except ValueError as error:
                        result, failure = None, str(error)
                    wall = time.monotonic() - began
                    phase_wall += wall
                    atomic_torch(path, {"binding": eval_binding(plan, seed, system),
                                        "scene_index": scene_index, "context": context,
                                        "horizon": pair.delta, "system": system, "arm": arm,
                                        "pair": list(pair.identity),
                                        "active_parameters": active[(arm, str(pair.identity))],
                                        "result": result, "failure": failure, "wall_seconds": wall})
                    if failure:
                        failures[system] += 1
                    if done % 60 == 0 or done == total:
                        elapsed = time.monotonic() - phase_started
                        log(f"eval cells={done}/{total} elapsed={elapsed:.1f}s "
                            f"eta={elapsed / done * (total - done):.1f}s typed_failures={sum(failures.values())}")
        log(f"evaluation phase complete wall={time.monotonic() - phase_started:.1f}s "
            f"typed_failures={dict(failures)} memory={memory(args.device)}")


# ---------------------------------------------------------------- publication


def average_units(values):
    present = [v for v in values if v and "carrier_mse" in v]
    if not present:
        return {"available_units": 0}
    result = {"available_units": len(present)}
    for key in present[0]:
        items = [v[key] for v in present if v.get(key) is not None]
        result[key] = float(np.mean(items)) if items else None
    return result


def publication(args, plan):
    units = [(record["scene_index"], context) for record in plan["membership"]["scenes"]
             for context in clevrer.EVAL_CONTEXTS]
    expected = len(SEEDS) * len(SYSTEMS) * len(units)
    found, records, failures = 0, {}, Counter()
    for seed in SEEDS:
        for system in SYSTEMS:
            for scene_index, context in units:
                path = eval_path(args, seed, system, scene_index, context)
                if not path.exists():
                    continue
                record = torch.load(path, map_location="cpu", weights_only=True)
                check_binding(record, eval_binding(plan, seed, system))
                if record["scene_index"] != scene_index or record["context"] != context:
                    raise ValueError("evaluation record identity differs from its path")
                found += 1
                records[(seed, system, scene_index, context)] = record
                if record["failure"]:
                    failures[system] += 1
    stop_rule = read(args.output / STOP)["reason"] if (args.output / STOP).exists() else None
    common = {"schema": "issue_79_clevrer_boundary_report_v1", "identity": "issue-79-clevrer-boundary-report-v1",
              "plan_identity": plan["identity"], "definitions": plan["design"],
              "membership": plan["membership"], "representation": plan["representation"],
              "compute": compute_section(args, plan, records), "stop_rule": stop_rule,
              "inventory": {"scheduled_cells": expected, "executed_cells": found,
                            "typed_failure_cells": int(sum(failures.values())),
                            "missing_cells": expected - found},
              "claim_boundary": plan["claim_boundary"],
              "limitations": [
                  "out-of-family replication: mechanism-shape evidence only, no leaderboard/VQA or perception claim",
                  "engine-annotation features in, no visual perception on either arm; shared frozen parser contract",
                  "descriptive paired bootstrap over 48 (scene, context) units per seed; no inferential superiority claim",
                  "inference and symbolic-head work are not equalized across arms; recorded, not matched",
                  "macro arm unsupported on CLEVRER; its estimand dropped (Phase 0)"],
              "archived_release": False, "fresh_evaluation_opened": False,
              "final_evaluation_opened": False, "issue_64_authorized": False}
    if found != expected:
        common["diagnostics_complete"] = False
        common["dispositions"] = {"question_1_signature": "readiness_or_precision_insufficient",
                                  "question_2_regime": "readiness_or_precision_insufficient"}
        return common
    per_seed, per_unit = {}, {}
    for seed in SEEDS:
        seed_rows = {}
        for system, (arm, pair) in SYSTEMS.items():
            rows = {}
            for scene_index, context in units:
                record = records[(seed, system, scene_index, context)]
                rows[(scene_index, context)] = {} if record["failure"] else \
                    {str(endpoint): record["result"]["errors"][str(endpoint)]["errors"]
                     for endpoint in clevrer.ENDPOINTS}
            per_unit[(seed, system)] = rows
            first = records[(seed, system, units[0][0], units[0][1])]
            seed_rows[system] = {
                "curves": {str(endpoint): average_units([rows[unit].get(str(endpoint)) for unit in units])
                           for endpoint in clevrer.ENDPOINTS},
                "typed_failures": int(failures[system]),
                "mean_rollout_wall_seconds": float(np.mean([records[(seed, system, s, c)]["wall_seconds"]
                                                            for s, c in units])),
                "transition_calls_per_endpoint": {str(endpoint): endpoint // pair.delta
                                                  for endpoint in clevrer.ENDPOINTS},
                "linear_macs_per_step": first["result"]["linear_macs_per_step"] if first["result"] else None,
                "active_parameters": first["active_parameters"]}
        per_seed[str(seed)] = seed_rows
    contrasts = summarize_contrasts(per_unit, units)
    regime = regime_analysis(args, plan, per_unit, units)
    dispositions = decide(plan, contrasts, regime, per_seed, sum(failures.values()))
    return {**common, "diagnostics_complete": True, "per_seed": per_seed, "contrasts": contrasts,
            "regime": regime, "dispositions": dispositions}


def error_lookup(per_unit, seeds, system, unit, endpoint, metric):
    values = [per_unit[(seed, system)][unit].get(str(endpoint), {}).get(metric) for seed in seeds]
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def summarize_contrasts(per_unit, units):
    """Paired training-effect and symbolic-execution contrasts per (h, endpoint)."""
    result = []
    for kind, tested_template, reference_template in (
            ("training_effect", "hybrid_continuous_h{h}", "continuous_h{h}"),
            ("symbolic_execution", "hybrid_micro_h{h}", "hybrid_continuous_h{h}")):
        for horizon in clevrer.HORIZONS:
            tested, reference = tested_template.format(h=horizon), reference_template.format(h=horizon)
            for endpoint in clevrer.ENDPOINTS:
                entry = {"kind": kind, "horizon": horizon, "endpoint": endpoint,
                         "tested": tested, "reference": reference, "positive_favors": tested,
                         "scope": "descriptive; arms independently trained; intervals descriptive only"}
                for metric in ("position_mse", "velocity_mse", "carrier_mse"):
                    differences = []
                    for unit in units:
                        across = []
                        for seed in SEEDS:
                            a = error_lookup(per_unit, [seed], reference, unit, endpoint, metric)
                            b = error_lookup(per_unit, [seed], tested, unit, endpoint, metric)
                            if a is not None and b is not None:
                                across.append(a - b)
                        if len(across) == len(SEEDS):
                            differences.append(float(np.mean(across)))
                    entry[metric] = paired_interval(differences) if differences else {"paired_units": 0}
                result.append(entry)
    return result


def regime_analysis(args, plan, per_unit, units):
    """Contact-activity strata of the symbolic-execution effect (question 2)."""
    events = {record["scene_index"]: scene_shard(args, plan, record["scene_index"])["collision_frames"]
              for record in plan["membership"]["scenes"]}
    result = []
    for horizon in clevrer.HORIZONS:
        for endpoint in clevrer.ENDPOINTS:
            strata = {"contact_active": [], "contact_still": []}
            for scene_index, context in units:
                differences = []
                for seed in SEEDS:
                    a = error_lookup(per_unit, [seed], f"hybrid_continuous_h{horizon}", (scene_index, context),
                                     endpoint, "position_mse")
                    b = error_lookup(per_unit, [seed], f"hybrid_micro_h{horizon}", (scene_index, context),
                                     endpoint, "position_mse")
                    if a is not None and b is not None:
                        differences.append(a - b)
                if len(differences) == len(SEEDS):
                    flag = clevrer.interval_active(events[scene_index], context, REGIME_WINDOW)
                    strata["contact_active" if flag else "contact_still"].append(float(np.mean(differences)))
            row = {"horizon": horizon, "endpoint": endpoint,
                   "active": paired_interval(strata["contact_active"]) if strata["contact_active"]
                   else {"paired_units": 0},
                   "still": paired_interval(strata["contact_still"]) if strata["contact_still"]
                   else {"paired_units": 0}}
            if row["active"]["paired_units"] and row["still"]["paired_units"]:
                row["difference_active_minus_still"] = row["active"]["mean"] - row["still"]["mean"]
            result.append(row)
    return result


def decide(plan, contrasts, regime, per_seed, typed_failures):
    """Frozen decision rules from plan['decision_rules']; descriptive tokens only."""
    insufficient = "readiness_or_precision_insufficient"
    if typed_failures:
        return {"question_1_signature": insufficient, "question_2_regime": insufficient,
                "question_1_detail": {"reason": "typed evaluation failures present", "cells": typed_failures},
                "question_2_detail": {"reason": "typed evaluation failures present", "cells": typed_failures}}
    by_key = {(c["kind"], c["horizon"], c["endpoint"]): c for c in contrasts}

    def mean_curve(system, endpoint):
        values = [per_seed[str(seed)][system]["curves"][str(endpoint)].get("position_mse") for seed in SEEDS]
        return float(np.mean(values)) if all(v is not None for v in values) else None

    curves = {h: {str(endpoint): mean_curve(f"continuous_h{h}", endpoint) for endpoint in clevrer.ENDPOINTS}
              for h in clevrer.HORIZONS}
    q1 = {"components": {}, "disposition": "not_supported_by_this_experiment"}
    s1_entry = by_key[("training_effect", 1, 15)]["position_mse"]
    s1 = s1_entry["mean"] if s1_entry.get("paired_units") else None
    margin_base = curves[1]["15"]
    margin = RELATIVE_MARGIN * margin_base if margin_base is not None else None
    q1["components"]["S1"] = {"difference": s1, "margin": margin,
                              "holds": s1 is not None and margin is not None and s1 > 0 and s1 > margin}
    s15_entry = by_key[("training_effect", 15, 15)]["position_mse"]
    s15 = s15_entry["mean"] if s15_entry.get("paired_units") else None
    q1["components"]["S2"] = {"h1_difference": s1, "h15_difference": s15,
                              "holds": s1 is not None and s15 is not None and s15 < s1}
    q1["components"]["S3"] = {"curves": {str(h): curves[h] for h in clevrer.HORIZONS},
                              "holds": all(curves[h][str(a)] is not None and curves[h][str(b)] is not None
                                           and curves[h][str(b)] > curves[h][str(a)]
                                           for h in clevrer.HORIZONS
                                           for a, b in zip(clevrer.ENDPOINTS[:-1], clevrer.ENDPOINTS[1:]))}
    q1["disposition"] = "supported" if all(part.get("holds") for part in q1["components"].values()) \
        else "not_supported_by_this_experiment"
    primary = [row for row in regime if row["endpoint"] == 120]
    q2 = {"rows_at_120": primary, "disposition": "not_supported_by_this_experiment"}
    if any(row["active"]["paired_units"] < REGIME_MINIMUM_UNITS or row["still"]["paired_units"] < REGIME_MINIMUM_UNITS
           for row in primary):
        q2["disposition"] = insufficient
        q2["reason"] = "a contact stratum is below the frozen minimum unit count"
        return {"question_1_signature": q1["disposition"], "question_2_regime": q2["disposition"],
                "question_1_detail": q1, "question_2_detail": q2}
    margins, holds = {}, []
    for row in primary:
        base = mean_curve(f"hybrid_continuous_h{row['horizon']}", 120)
        margin = RELATIVE_MARGIN * base if base is not None else None
        margins[str(row["horizon"])] = margin
        difference = row.get("difference_active_minus_still")
        holds.append(difference is not None and margin is not None and difference > margin)
    q2["margins"] = margins
    q2["disposition"] = "supported" if any(holds) else "not_supported_by_this_experiment"
    return {"question_1_signature": q1["disposition"], "question_2_regime": q2["disposition"],
            "question_1_detail": q1, "question_2_detail": q2}


def compute_section(args, plan, records):
    """Compute accounting: wall, steps, active parameters; inference not equalized."""
    fits = []
    for seed in SEEDS:
        for arm in ARMS:
            path = cell(args, seed, arm) / "predictor.pt"
            if path.exists():
                value = torch.load(path, map_location="cpu", weights_only=True)
                fits.append({"seed": seed, "arm": arm, "wall_seconds": value["wall_seconds"],
                             "steps": value["step"], "optimizer_examples": value["optimizer_examples"],
                             "memory": value["memory"]})
    active = {}
    for (_, system, _, _), record in records.items():
        active[f"{system}|h{record['pair'][0]}|{record['pair'][1]}"] = record["active_parameters"]
    return {"fits": fits, "training_wall_seconds": sum(f["wall_seconds"] for f in fits),
            "evaluation_wall_seconds": sum(record["wall_seconds"] for record in records.values()),
            "allowance_gpu_hours": plan["compute"]["allowance_gpu_hours"],
            "active_parameters_per_system": active,
            "units_per_arm": {arm: sum(1 for (_, _, _, _), record in records.items()
                                       if record["arm"] == arm and not record["failure"]) for arm in ARMS},
            "inference_equalized": False,
            "note": ("candidate counts do not exist on CLEVRER (no planning estimand); units, steps, wall and active"
                     " parameters are reported per arm; inference is not equalized")}


def compact_report(result):
    return {k: v for k, v in result.items() if k not in ("per_unit",)}


def comparisons_csv(result):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(("seed", "system", "endpoint_frames", "kind", "metric", "value", "available_units"))
    for seed, systems in result.get("per_seed", {}).items():
        for name, system in systems.items():
            for endpoint, metrics in system["curves"].items():
                for metric, value in metrics.items():
                    if metric != "available_units":
                        writer.writerow((seed, name, endpoint, "recursive", metric, value,
                                         metrics["available_units"]))
    writer.writerow(())
    writer.writerow(("contrast_tested", "reference", "kind", "horizon", "endpoint_frames", "metric",
                     "mean_difference", "interval_low", "interval_high", "paired_units"))
    for contrast in result.get("contrasts", []):
        for metric in ("position_mse", "velocity_mse", "carrier_mse"):
            entry = contrast[metric]
            if entry.get("paired_units"):
                low, high = entry["descriptive_95_percent_interval"]
                writer.writerow((contrast["tested"], contrast["reference"], contrast["kind"], contrast["horizon"],
                                 contrast["endpoint"], metric, entry["mean"], low, high, entry["paired_units"]))
    writer.writerow(())
    writer.writerow(("kind", "horizon", "endpoint_frames", "stratum", "mean_difference", "interval_low",
                     "interval_high", "paired_units"))
    for row in result.get("regime", []):
        for stratum in ("active", "still"):
            entry = row[stratum]
            if entry.get("paired_units"):
                low, high = entry["descriptive_95_percent_interval"]
                writer.writerow(("symbolic_execution_regime", row["horizon"], row["endpoint"], stratum,
                                 entry["mean"], low, high, entry["paired_units"]))
    writer.writerow(())
    writer.writerow(("seed", "arm", "fit_wall_seconds", "steps", "optimizer_examples"))
    for fit in result.get("compute", {}).get("fits", []):
        writer.writerow((fit["seed"], fit["arm"], fit["wall_seconds"], fit["steps"], fit["optimizer_examples"]))
    return stream.getvalue()


def findings_md(result):
    lines = ["# Issue-79 CLEVRER boundary replication - findings", ""]
    lines.append(f"Diagnostics complete: {result['diagnostics_complete']}. Claim boundary: {result['claim_boundary']}")
    lines.append("")
    lines.append("Endpoint semantics: CLEVRER annotation frames; recursive fixed-pair rollouts from each eval"
                 " context; no planning/ranking-regret estimand and no candidate inventory on CLEVRER.")
    lines.append("")
    if not result.get("per_seed"):
        lines.append("Diagnostics incomplete; no outcome numbers reported.")
        lines.append(f"- inventory: {result['inventory']}")
        if result.get("stop_rule"):
            lines.append(f"- stop rule: {result['stop_rule']}")
        lines.append(f"- dispositions: {result['dispositions']}")
        return "\n".join(lines) + "\n"
    lines.append("## Membership")
    lines.append("")
    lines.append("| Scene | Objects | Collisions | Windows |")
    lines.append("| --- | --- | --- | --- |")
    for scene in result["membership"]["scenes"]:
        lines.append(f"| {scene['scene_index']} | {scene['objects']} | {scene['collisions']} | "
                     f"{result['membership']['windows_per_scene']} |")
    lines.append("")
    lines.append("## Recursive position MSE (mean over units; seeds listed individually)")
    lines.append("")
    lines.append("| System | Seed | e=15 | e=30 | e=60 | e=120 | Typed failures |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for seed, systems in result["per_seed"].items():
        for name, system in systems.items():
            cells = [system["curves"][str(endpoint)].get("position_mse") for endpoint in clevrer.ENDPOINTS]
            cells = ["n/a" if value is None else f"{value:.4f}" for value in cells]
            lines.append(f"| {name} | {seed} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | "
                         f"{system['typed_failures']} |")
    lines.append("")
    lines.append("## Paired contrasts on position MSE (positive favors the tested system; DESCRIPTIVE)")
    lines.append("")
    lines.append("| Tested | Reference | Kind | h | Endpoint | Difference | Descriptive 95% interval |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for contrast in result["contrasts"]:
        entry = contrast["position_mse"]
        if not entry.get("paired_units"):
            continue
        low, high = entry["descriptive_95_percent_interval"]
        lines.append(f"| {contrast['tested']} | {contrast['reference']} | {contrast['kind']} | "
                     f"{contrast['horizon']} | {contrast['endpoint']} | {entry['mean']:+.4f} | "
                     f"[{low:+.4f}, {high:+.4f}] |")
    lines.append("")
    lines.append("## Contact-activity regime of the micro symbolic-execution effect (DESCRIPTIVE)")
    lines.append("")
    lines.append("| h | Endpoint | Active units | Active difference | Still units | Still difference | "
                 "Active-minus-still |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in result["regime"]:
        active, still = row["active"], row["still"]
        difference = row.get("difference_active_minus_still")

        def cell(entry):
            return "n/a" if not entry.get("paired_units") else f"{entry['mean']:+.4f}"
        lines.append(f"| {row['horizon']} | {row['endpoint']} | {active['paired_units']} | "
                     f"{cell(active)} | {still['paired_units']} | {cell(still)} | "
                     + (f"{difference:+.4f}" if difference is not None else "n/a") + " |")
    lines.append("")
    lines.append("## Dispositions")
    lines.append("")
    for key, label in (("question_1_signature", "Q1 signature replication"),
                       ("question_2_regime", "Q2 contact-activity regime")):
        detail = result["dispositions"].get(f"{key.split('_')[0]}_{key.split('_')[1]}_detail", {})
        suffix = f" ({detail['reason']})" if detail.get("reason") else ""
        lines.append(f"- {label}: **{result['dispositions'][key]}**{suffix}")
    lines.append("")
    lines.append("Both families are reported separately: a CLEVRER non-replication does not falsify the NovPhy"
                 " boundary, and vice versa.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for item in result["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def publish(args, plan):
    result = publication(args, plan)
    write(args.output / "summary.json", compact_report(result))
    (args.output / "comparisons.csv").write_text(comparisons_csv(result))
    (args.output / "findings.md").write_text(findings_md(result))
    log(f"published diagnostics_complete={result['diagnostics_complete']} "
        f"dispositions={result['dispositions']}")


# ---------------------------------------------------------------- smoke / dry-run


def smoke(args):
    """Bounded real pass: one scene end-to-end, temporary weights, production_evidence false."""
    began = time.monotonic()
    verify_zips()
    members, _ = clevrer.select_membership(ZIPS["validation"], count=1, scan_limit=64)
    scene = members[0]
    velocity_scale = clevrer.frozen_velocity_scale(tuple(members))
    tier_edges = list(clevrer.frozen_tier_edges(tuple(members)))
    clip = clevrer.clip_carriers(scene, velocity_scale)
    relations, mask = clevrer.clip_relations(scene, tier_edges)
    windows = clevrer.training_windows(clip, relations, mask)
    plan = {"schema": SCHEMA, "identity": IDENTITY + ":smoke", "capacity": capacity_contract(),
            "representation": {"identity": clevrer.IDENTITY, "velocity_scale": velocity_scale,
                               "speed_tier_edges": tier_edges},
            "design": {"training": {"steps": 9000, "batch_size": 16, "learning_rate": .0001,
                                    "weight_decay": .0001, "grad_clip": 1.}}}
    torch.manual_seed(SEEDS[0])
    cells = []
    for arm in ARMS:
        model = new_model(plan, arm)
        optim = torch.optim.AdamW(model.parameters(), lr=.0001)
        pair_counts = Counter()
        loss = None
        for step in range(12):
            batch = sample(windows, 16, torch.Generator().manual_seed(SEEDS[0] + step), "cpu")
            selected = PAIRS_CLEVRER[step % len(PAIRS_CLEVRER)]
            pair = PredictionPair(selected.delta, Abstraction.CONTINUOUS) if arm == "continuous" else selected
            optim.zero_grad(set_to_none=True)
            loss = loss_for(model, batch, pair)
            loss.backward()
            optim.step()
            pair_counts[str(pair.identity)] += 1
        errors = rollout_errors(model.eval(), clip, 0, PAIRS_CLEVRER[0], "cpu")
        cells.append({"arm": arm, "final_loss_finite": bool(torch.isfinite(loss)),
                      "pair_counts": dict(pair_counts), "endpoint_errors": errors["errors"]})
    with tempfile.TemporaryDirectory(prefix="novphy-issue79-smoke-") as directory:
        write(Path(directory) / "smoke-evidence.json", {"cells": cells})
        if read(Path(directory) / "smoke-evidence.json")["cells"] != cells:
            raise ValueError("smoke evidence JSON round trip differs")
    result = {"schema": "issue_79_clevrer_smoke_v1", "production_evidence": False,
              "scene_index": scene.scene_index, "windows": {k: list(v.shape) for k, v in windows.items()},
              "velocity_scale": velocity_scale, "speed_tier_edges": tier_edges,
              "cells": cells, "wall_seconds": time.monotonic() - began, "memory": memory(args.device)}
    write(args.output / "smoke.json", result)
    log(f"real smoke complete scene={scene.scene_index} wall={result['wall_seconds']:.1f}s; "
        "temporary weights removed, no production training")


def dry_run(args):
    """No-write readiness report; exits 0 with or without downloaded data."""
    present = {name: path.exists() for name, path in ZIPS.items()}
    log(f"no-write dry-run: modes=prepare,train,run-evaluation,publish,validate; "
        f"archives_present={present}; output={args.output}")
    if all(present.values()):
        try:
            verify_zips()
            members, scan = clevrer.select_membership(ZIPS["validation"])
            log(f"dry-run membership preview: scenes={[s.scene_index for s in members]} "
                f"scanned={scan['scanned']} windows={len(members) * clevrer.WINDOWS_PER_SCENE} "
                f"units_per_seed={len(members) * len(clevrer.EVAL_CONTEXTS)}")
        except (ValueError, clevrer.ClevrerBoundaryError) as error:
            log(f"dry-run membership preview blocked: {error}")
    else:
        log("dry-run: archives missing; --prepare will download-check, parse and freeze Phase 0")
    log(f"dry-run planned cells: fits={len(SEEDS) * len(ARMS)} steps_each=9000; "
        f"eval_cells={len(SEEDS) * len(SYSTEMS) * 12 * len(clevrer.EVAL_CONTEXTS)}; "
        f"allowance={GPU_HOURS_ALLOWANCE} GPU-h; lock={GPU_LOCK}")
    return 0


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "smoke-test", "prepare", "train", "run-evaluation", "publish", "validate"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        if args.dry_run:
            return dry_run(args)
        if args.smoke_test:
            smoke(args)
            return 0
        if args.prepare:
            prepare(args)
            return 0
        plan = load_plan(args)
        if args.train:
            train(args, plan)
        elif args.run_evaluation:
            run_evaluation(args, plan)
        elif args.publish:
            publish(args, plan)
        else:
            result = publication(args, plan)
            if compact_report(result) != read(args.output / "summary.json") \
                    or comparisons_csv(result).encode("utf-8") != (args.output / "comparisons.csv").read_bytes() \
                    or findings_md(result) != (args.output / "findings.md").read_text():
                raise ValueError("published issue-79 tables differ from bound source evidence")
            log("exact saved-evidence validation passed")
        return 0
    except (ValueError, OSError, clevrer.ClevrerBoundaryError) as error:
        log(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
