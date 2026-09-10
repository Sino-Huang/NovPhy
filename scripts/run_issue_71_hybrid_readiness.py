"""Bounded-memory training and readiness checks for the CNN hybrid candidate."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import tempfile
import time

import torch
from torch.nn import functional as F

from scripts import run_issue_70_parser_repair as repair
from world_model.data.deployment_temporal import AgentObservation, TemporalObservationContext
from world_model.planning.gameplay import SlingshotAction
from world_model.model import DualOutputPredictor, PredictorConfig
from world_model.training.cnn_hybrid import (
    ARCHITECTURE, PAIRS, CNNHybridPredictor, CarrierPairController,
    adaptive_rollout, checkpoint_contract, intervention_audit, linear_macs,
    training_loss, trajectory_labels,
)


ROOT = repair.experiment.ROOT / ".local-artifacts/issue-71-hybrid-readiness-v1"
SCHEMA = "issue_71_hybrid_readiness_v1"
SOURCE_FILES = (
    "scripts/run_issue_71_hybrid_readiness.py", "world_model/training/cnn_hybrid.py",
    "scripts/run_issue_70_parser_repair.py", "world_model/training/cohort_v2_visual_parser.py",
    "world_model/data/deployment_temporal.py", "world_model/model/predictor.py",
    "world_model/model/config.py",
)


def log(message):
    print(f"[issue-71] {message}", flush=True)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_receipt():
    root = repair.experiment.ROOT
    return {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "worktree_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "archived_release": False,
        "source_text": {p: (root / p).read_text() for p in SOURCE_FILES},
    }


def audit_existing(source):
    path = source / "world-models/checkpoints/self-conditioned-h15-u4/seed-20260901.pt"
    payload = torch.load(path, weights_only=True, map_location="cpu")
    spec = payload["metadata"]["spec"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec["seed"])
        initial = DualOutputPredictor(PredictorConfig(**spec["predictor_config"])).state_dict()
    untouched = {group: all(torch.equal(value, initial[name]) for name, value in payload["model_state"].items()
                           if name.startswith(group + "."))
                 for group in ("micro_adapter", "macro_adapter", "micro_head", "macro_head")}
    return {"reusable": ["trained CNN parser", "236-value carrier codec", "accepted training RGB and masked engine labels",
                         "PairConditioner and FiLMBlock implementation", "capture infrastructure"],
            "issue_70_checkpoint": str(path), "issue_70_checkpoint_identity": payload["metadata"]["checkpoint_identity"],
            "issue_70_symbolic_components_equal_seed_initialization": untouched,
            "legacy_hybrid": {"source": "https://github.com/Sino-Huang/NovPhy/issues/15", "slots": 15,
                              "carrier_dimension": 197, "compatible": False, "oracle_symbol_evidence": True},
            "missing": ["CNN-compatible trained symbolic transitions/readouts", "h1/h5 training on CNN carrier",
                        "checkpoint-bound controller", "predicted-carrier symbol refresh"],
            "legacy_mode_graph_and_controller_weights_reused": False}


def make_plan(args):
    previous = repair.load_plan(args.source)
    if len(previous["vocabulary"]) != 18 or len(previous["training_records"]) != 3000:
        raise ValueError("requires the accepted #70 CNN 18-slot / 3000-lineage source")
    if not read(args.source / "experiment/objective-audit.json")["passed"]:
        raise ValueError("#70 CNN perception objective has not passed")
    return {
        "schema": SCHEMA, "identity": "issue-71-hybrid-plan-v1",
        "source": str(args.source.resolve()), "source_plan_identity": previous["identity"],
        "source_release_identity": previous["training_release_identity"],
        "reuse_audit": audit_existing(args.source),
        "contract": checkpoint_contract(previous["vocabulary"], repair.PARSER_ID, repair.CARRIER_ID),
        "records": previous["training_records"],
        "data": {"window_steps": 60, "windows_per_shot": 4, "parser_batch": 32,
                 "sampling": "uniform starts including first and last; every training shot",
                 "controller_lineages": "one-based index divisible by 5; excluded from predictor fit",
                 "roles": ["training"], "predictor_lineages": 2400, "controller_lineage_count": 600},
        "training": {"seed": 20260908, "steps": 9000, "batch_size": 64,
                     "learning_rate": .0001, "weight_decay": .0001, "grad_clip": 1.,
                     "pair_schedule": "nine pairs per group of three lineages; 1000 updates each; every fitting lineage visits every pair",
                     "arms": ["teacher_forced", "corrected"], "unroll_steps": 4,
                     "symbolic_loss_weight": .1, "symbol_positive_weight": 5.,
                     "corrected_extra_loss": "recursive MSE + 0.01 * mean(relu(abs(z)-2)^2)",
                     "matched": ["CNN", "carrier", "training lineages/windows", "capacity", "seed",
                                 "local transition targets", "symbol supervision", "optimizer steps/examples"],
                     "compute_matched": False},
        "controller": {"seed": 7101, "steps": 1800, "batch_size": 128, "learning_rate": .001,
                       "teacher": "duration-weighted trajectory dynamic programming",
                       "compute_weight": .0001, "tie_order": "frozen PAIRS order",
                       "closed_loop_aggregation_rounds": 1,
                       "aggregation_labels": "recompute DP from predicted carriers against aligned training targets",
                       "diagnostics": "training-role only; not independent evaluation"},
        "code": source_receipt(), "final_evaluation_opened": False, "issue_64_authorized": False,
    }


def load_plan(args):
    plan = read(args.output / "plan.json")
    if plan["schema"] != SCHEMA or plan["contract"]["architecture"] != ARCHITECTURE:
        raise ValueError("incompatible issue-71 plan")
    if Path(plan["source"]).resolve() != args.source.resolve():
        raise ValueError("source path differs from the frozen CNN source")
    for path, source in plan["code"]["source_text"].items():
        if (repair.experiment.ROOT / path).read_text() != source:
            raise ValueError(f"implementation changed since freeze: {path}; retain this run and use a new output")
    return plan


def build_shard(args, plan, index, adapter, *, shot_limit=None, window_limit=None):
    """At most one capture JSON and one parser batch of PNG bytes at a time."""
    previous = repair.load_plan(args.source)
    record = plan["records"][index - 1]
    root = Path(previous["training_release"])
    raw = repair.trajectory(root, record, previous["training_release_identity"])
    windows = []
    for shot in raw["shots"][:shot_limit]:
        shot_root, samples, frames, labels = repair.load_shot(root, record, shot)
        if any(f["fixed_step"] != frames[0]["fixed_step"] + i for i, f in enumerate(frames)):
            raise ValueError("hybrid training needs aligned stride-one frames")
        last = len(frames) - 1
        if last < 15:
            raise ValueError("training shot lacks a complete h15 transition; do not silently omit it")
        starts = sorted({round(i * max(0, last - 60) / 3) for i in range(4)})[:window_limit]
        needed = sorted({p for start in starts for p in range(max(0, start - 1), min(start + 60, last) + 1)})
        parsed = {}
        observation_root = shot_root / "observation-trace"
        for start in range(0, len(needed), 32):
            positions = needed[start:start + 32]
            observations = []
            for p in positions:
                f = frames[p]; ref = f["agent_observation"]
                observations.append(AgentObservation(ref["identity"], f["fixed_step"], f["fixed_time_seconds"],
                                                     (observation_root / ref["relative_path"]).read_bytes(), "agent"))
            parsed.update(zip(positions, adapter.parse_batch(tuple(observations)), strict=True))
        # Source projection is irrelevant to symbolic labels, but reuse the
        # existing authoritative target builder rather than inventing semantics.
        def observation(p):
            f = frames[p]
            return AgentObservation(f["agent_observation"]["identity"], f["fixed_step"],
                                    f["fixed_time_seconds"], b"", "agent")
        action = SlingshotAction.from_interface_action(shot["action"]["interface_action"])
        a = repair.experiment.old.probe._action_tensor(action, repair.experiment.old.probe._bounds(), 480)
        for start in starts:
            length = min(60, last - start)
            carriers, targets = [], []
            for p in range(start, start + length + 1):
                context = TemporalObservationContext(None if p == 0 else observation(p - 1), observation(p))
                carriers.append(adapter.build_from_parsed(context, parsed[p], None if p == 0 else parsed[p - 1]).tensor)
                targets.append(repair.targets(samples[p], frames[p]["capture_metadata"], labels["micro"][p], labels["macro"][p],
                                               previous["vocabulary"]))
            while len(carriers) < 61:
                carriers.append(carriers[-1]); targets.append(targets[-1])
            windows.append({"z": torch.stack(carriers), "action": a, "length": torch.tensor(length),
                            **{k: torch.stack([t[k] for t in targets]).bool()
                               for k in ("relations", "relation_mask", "macros", "macro_mask")},
                            "shot": shot["shot_index"], "start": start})
        log(f"prepare lineage={index}/3000 shot={shot['shot_index']} windows={len(starts)} parsed_frames={len(needed)}")
    result = {k: torch.stack([w[k] for w in windows]) for k in ("z", "action", "length", "relations", "relation_mask", "macros", "macro_mask")}
    result["relations_mask"] = result.pop("relation_mask")
    result["macros_mask"] = result.pop("macro_mask")
    return {"schema": SCHEMA, "contract": plan["contract"], "record": record,
            "windows": [{"shot": w["shot"], "start": w["start"]} for w in windows], "tensors": result}


def shard_path(args, index):
    return args.output / "shards" / f"lineage-{index:04d}.pt"


def load_shard(args, plan, index):
    value = torch.load(shard_path(args, index), map_location="cpu", weights_only=True)
    if value["contract"] != plan["contract"] or value["record"] != plan["records"][index - 1]:
        raise ValueError("hybrid shard source/role/representation differs")
    return value["tensors"]


def prepare(args):
    if not (args.output / "plan.json").exists():
        write(args.output / "plan.json", make_plan(args))
    plan = load_plan(args)
    adapter = repair.load_repaired_adapter(args.source, args.device)
    began = time.monotonic()
    total_windows = 0
    for index in range(1, 3001):
        log(f"prepare lineage={index}/3000 start")
        path = shard_path(args, index)
        if path.exists():
            data = load_shard(args, plan, index)
        else:
            shard = build_shard(args, plan, index, adapter)
            repair.atomic_torch(path, shard)
            data = shard["tensors"]
        total_windows += len(data["z"])
        elapsed = time.monotonic() - began
        log(f"prepare lineage={index}/3000 complete elapsed={elapsed:.1f}s eta={elapsed/index*(3000-index):.1f}s")
    write(args.output / "data.json", {"plan_identity": plan["identity"], "lineages": 3000,
                                      "windows": total_windows, "wall_seconds": time.monotonic() - began})


def load_model(args, plan, arm):
    value = torch.load(args.output / f"{arm}.pt", map_location="cpu", weights_only=True)
    if (value["contract"] != plan["contract"] or value["training"] != plan["training"]
            or value["source_release_identity"] != plan["source_release_identity"]
            or value["arm"] != arm or value["step"] != plan["training"]["steps"]):
        raise ValueError("incompatible/incomplete trained hybrid checkpoint")
    model = CNNHybridPredictor().to(args.device)
    model.load_state_dict(value["model"], strict=True)
    return model.eval(), value


def sample(data, batch_size, generator, device):
    indices = torch.randint(len(data["z"]), (batch_size,), generator=generator)
    return {k: v[indices].to(device) for k, v in data.items()}


def fitting_lineage_group(step):
    indices = [i for i in range(1, 3001) if i % 5]
    return tuple(indices[((step // 9) * 3 + j) % len(indices)] for j in range(3))


def train_models(args):
    plan = load_plan(args)
    if read(args.output / "data.json")["lineages"] != 3000:
        raise ValueError("data preparation incomplete")
    recipe = plan["training"]
    for arm in recipe["arms"]:
        if (args.output / f"{arm}.pt").exists():
            load_model(args, plan, arm)
            log(f"train arm={arm} completed checkpoint reused")
            continue
        torch.manual_seed(recipe["seed"])
        model = CNNHybridPredictor().to(args.device)
        initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
        generator = torch.Generator().manual_seed(recipe["seed"])
        progress_path = args.output / f"{arm}-progress.pt"
        start, elapsed_prior, gradients, pair_counts = 0, 0., set(), Counter()
        if progress_path.exists():
            old = torch.load(progress_path, map_location="cpu", weights_only=True)
            if old["contract"] != plan["contract"] or old["training"] != recipe or old["arm"] != arm:
                raise ValueError("training resume contract differs")
            model.load_state_dict(old["model"], strict=True); optimizer.load_state_dict(old["optimizer"])
            generator.set_state(old["generator"])
            start, elapsed_prior = old["step"], old["wall_seconds"]
            if not 0 <= start <= recipe["steps"]:
                raise ValueError("training progress exceeds the frozen budget")
            gradients, pair_counts = set(old["gradient_parameters"]), Counter(old["pair_counts"])
            value = old  # a completed progress checkpoint can be published after interruption
        began = time.monotonic()
        cached_group, data = None, None
        for step in range(start, recipe["steps"]):
            # Same outcome-independent lineage/window schedule in both arms.
            group = step // 9
            if group != cached_group:
                shards = [load_shard(args, plan, index) for index in fitting_lineage_group(step)]
                data = {k: torch.cat([s[k] for s in shards]) for k in shards[0]}
                cached_group = group
            batch = sample(data, recipe["batch_size"], generator, args.device)
            pair = PAIRS[step % len(PAIRS)]
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model, batch, pair, arm == "corrected")
            if not bool(torch.isfinite(loss)):
                raise ValueError(f"nonfinite {arm} loss at step {step+1}")
            loss.backward()
            # Actual task gradients, before clipping/weight decay.
            gradients.update(n for n, p in model.named_parameters()
                             if p.grad is not None and bool((p.grad != 0).any()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["grad_clip"])
            optimizer.step()
            pair_counts[str(pair.identity)] += 1
            if (step + 1) % 90 == 0 or step + 1 == recipe["steps"]:
                elapsed = time.monotonic() - began + elapsed_prior
                log(f"train arm={arm} step={step+1}/{recipe['steps']} loss={float(loss.detach()):.6f} elapsed={elapsed:.1f}s eta={elapsed/(step+1)*(recipe['steps']-step-1):.1f}s")
                value = {"schema": SCHEMA, "identity": f"issue-71-{arm}-seed-{recipe['seed']}",
                         "contract": plan["contract"], "training": recipe, "arm": arm, "step": step + 1,
                         "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "generator": generator.get_state(), "gradient_parameters": sorted(gradients),
                         "pair_counts": dict(pair_counts), "wall_seconds": elapsed,
                         "optimizer_examples": (step + 1) * recipe["batch_size"],
                         "local_transition_example_upper_bound": (step + 1) * recipe["batch_size"] * 4,
                         "forward_transition_example_upper_bound": (step + 1) * recipe["batch_size"] * (7 if arm == "corrected" else 4),
                         "changed_parameters": [n for n, p in model.named_parameters()
                                                if not torch.equal(p.detach().cpu(), initial[n])],
                         "source_release_identity": plan["source_release_identity"]}
                repair.atomic_torch(progress_path, value)
        repair.atomic_torch(args.output / f"{arm}.pt", value)
        log(f"train arm={arm} complete")


def load_controller(args, plan, model_identity):
    value = torch.load(args.output / "controller.pt", map_location="cpu", weights_only=True)
    if (value["contract"] != plan["contract"] or value["predictor_identity"] != model_identity
            or value["recipe"] != plan["controller"] or value["round"] != 1
            or value["steps_this_round"] != plan["controller"]["steps"]):
        raise ValueError("controller is not bound to the trained corrected predictor")
    controller = CarrierPairController().to(args.device)
    controller.load_state_dict(value["model"], strict=True)
    return controller.eval(), value


def generate_labels(args, plan, model, *, controller=None):
    round_index = int(controller is not None)
    began = time.monotonic()
    for ordinal, index in enumerate(range(5, 3001, 5), 1):
        path = args.output / f"controller-labels-{round_index}" / f"lineage-{index:04d}.pt"
        if path.exists():
            cached = torch.load(path, weights_only=True, map_location="cpu")
            if cached["contract"] != plan["contract"] or cached["record"] != plan["records"][index-1] or cached["round"] != round_index:
                raise ValueError("cached controller label source/contract differs")
            continue
        data = load_shard(args, plan, index)
        # Fixed first window per shot, no outcome-based selection.
        shard = torch.load(shard_path(args, index), map_location="cpu", weights_only=True)
        selected = []
        seen = set()
        for position, window in enumerate(shard["windows"]):
            if window["shot"] not in seen:
                selected.append(position); seen.add(window["shot"])
        records = []
        for position in selected:
            batch = {k: v[position:position+1].to(args.device) for k, v in data.items()}
            original = batch["z"]
            if controller is not None:
                predicted = original.clone()
                current = original[:, 0]; t = 0
                with torch.no_grad():
                    while t < int(batch["length"][0]):
                        remaining = int(batch["length"][0]) - t
                        pair = PAIRS[int(controller(current, batch["action"], torch.tensor([remaining], device=args.device)).argmax(-1))]
                        current = model.carrier(current, batch["action"], pair)
                        t += pair.delta
                        predicted[:, t] = current
                # DP uses predicted decision inputs but unchanged true targets.
                labels = aggregation_labels(model, batch, predicted)
                features = predicted
            else:
                labels = trajectory_labels(model, batch, plan["controller"]["compute_weight"])
                features = original
            length = int(batch["length"][0])
            records.append({"z": features[0, :length].cpu(), "action": batch["action"].expand(length, -1).cpu(),
                            "remaining": torch.arange(length, 0, -1), "labels": labels[0, :length].cpu()})
        repair.atomic_torch(path, {"contract": plan["contract"], "record": plan["records"][index-1], "round": round_index,
                                    "tensors": {k: torch.cat([r[k] for r in records]) for k in records[0]}})
        elapsed = time.monotonic() - began
        log(f"controller labels round={round_index} lineage={ordinal}/600 elapsed={elapsed:.1f}s eta={elapsed/ordinal*(600-ordinal):.1f}s")


@torch.no_grad()
def aggregation_labels(model, batch, contexts):
    # Same DP objective as initial labels, changing only decision contexts.
    from world_model.training.cnn_hybrid import trajectory_labels
    return trajectory_labels(model, batch, contexts=contexts)


def train_controller(args):
    plan = load_plan(args)
    model, predictor = load_model(args, plan, "corrected")
    if (args.output / "controller.pt").exists():
        load_controller(args, plan, predictor["identity"])
        log("controller complete checkpoint reused")
        return
    recipe = plan["controller"]
    torch.manual_seed(recipe["seed"])
    controller = CarrierPairController().to(args.device)
    gradients = set()
    for round_index in (0, 1):
        checkpoint = args.output / f"controller-round-{round_index}.pt"
        if checkpoint.exists():
            saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
            if (saved["contract"] != plan["contract"] or saved["recipe"] != recipe
                    or saved["predictor_identity"] != predictor["identity"] or saved["round"] != round_index):
                raise ValueError("controller round binding differs")
            controller.load_state_dict(saved["model"], strict=True)
            gradients.update(saved["gradient_parameters"])
            continue
        generate_labels(args, plan, model, controller=controller if round_index else None)
        optimizer = torch.optim.AdamW(controller.parameters(), lr=recipe["learning_rate"])
        generator = torch.Generator().manual_seed(recipe["seed"] + round_index)
        began = time.monotonic()
        for step in range(recipe["steps"]):
            index = (step % 600 + 1) * 5
            records = []
            for label_round in range(round_index + 1):
                path = args.output / f"controller-labels-{label_round}" / f"lineage-{index:04d}.pt"
                record = torch.load(path, weights_only=True, map_location="cpu")
                if record["contract"] != plan["contract"] or record["record"] != plan["records"][index-1] or record["round"] != label_round:
                    raise ValueError("controller training source differs")
                records.append(record["tensors"])
            data = {k: torch.cat([r[k] for r in records]) for k in records[0]}
            batch = sample(data, recipe["batch_size"], generator, args.device)
            logits = controller(batch["z"], batch["action"], batch["remaining"])
            loss = F.cross_entropy(logits, batch["labels"])
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite controller loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradients.update(n for n, p in controller.named_parameters() if p.grad is not None and bool((p.grad != 0).any()))
            optimizer.step()
            if (step + 1) % 60 == 0:
                elapsed = time.monotonic() - began
                log(f"controller train round={round_index} step={step+1}/{recipe['steps']} loss={float(loss.detach()):.4f} elapsed={elapsed:.1f}s eta={elapsed/(step+1)*(recipe['steps']-step-1):.1f}s")
        saved = {"schema": SCHEMA, "identity": f"issue-71-joint-controller-seed-7101-round-{round_index}",
                 "contract": plan["contract"], "recipe": recipe, "predictor_identity": predictor["identity"],
                 "round": round_index, "steps_this_round": recipe["steps"],
                 "wall_seconds_this_round": time.monotonic() - began,
                 "model": controller.state_dict(), "gradient_parameters": sorted(gradients)}
        repair.atomic_torch(checkpoint, saved)
    repair.atomic_torch(args.output / "controller.pt", saved)


def readiness(args):
    plan = load_plan(args)
    reports = {}
    for arm in plan["training"]["arms"]:
        log(f"validate checkpoint={arm}")
        model, checkpoint = load_model(args, plan, arm)
        parameters = set(dict(model.named_parameters()))
        if set(checkpoint["gradient_parameters"]) != parameters or set(checkpoint["changed_parameters"]) != parameters:
            raise ValueError(f"untrained deployed parameter tensors in {arm}")
        if list(sorted(checkpoint["pair_counts"].values())) != [1000] * 9:
            raise ValueError("pair exposure is incomplete/unbalanced")
        reports[arm] = {"checkpoint_identity": checkpoint["identity"], "wall_seconds": checkpoint["wall_seconds"],
                        "parameter_count": sum(p.numel() for p in model.parameters()), "pair_counts": checkpoint["pair_counts"],
                        **{k: checkpoint[k] for k in ("optimizer_examples", "local_transition_example_upper_bound",
                                                       "forward_transition_example_upper_bound")}}
    model, checkpoint = load_model(args, plan, "corrected")
    controller, control = load_controller(args, plan, checkpoint["identity"])
    if set(control["gradient_parameters"]) != set(dict(controller.named_parameters())):
        raise ValueError("controller contains untrained deployed parameters")
    usage = Counter(); audits = []; traces = []; recursive = []
    symbolic = {name: Counter() for name in ("contact", "supports", "steady-state", "structure-unstable")}
    for ordinal, index in enumerate(range(5, 3001, 125), 1):
        log(f"validate training-role diagnostic={ordinal}/24")
        data = load_shard(args, plan, index)
        z = data["z"][:1, 0].to(args.device); a = data["action"][:1].to(args.device)
        audit = intervention_audit(model, z, a); audits.append(audit)
        if not audit["passed"]:
            raise ValueError("pair intervention is nonfinite or does not affect transition")
        _, trace = adaptive_rollout(model, controller, z, a)
        _, fixed = adaptive_rollout(model, controller, z, a, fixed_pair=PAIRS[6])
        # Same physical endpoint, observed CNN targets; descriptive training-role
        # diagnostics, never engine labels passed to runtime prediction.
        length = int(data["length"][0])
        endpoint = length - length % 15
        adaptive_end, _ = adaptive_rollout(model, controller, z, a, fixed_steps=endpoint)
        fixed_end, _ = adaptive_rollout(model, controller, z, a, fixed_steps=endpoint, fixed_pair=PAIRS[6])
        target = data["z"][:1, endpoint].to(args.device)
        recursive.append({"lineage_index": index, "endpoint_fixed_steps": endpoint,
                          "adaptive_mse": float((adaptive_end-target).square().mean()),
                          "fixed_mse": float((fixed_end-target).square().mean()),
                          "adaptive_max_abs_carrier": float(adaptive_end.abs().max()),
                          "fixed_max_abs_carrier": float(fixed_end.abs().max())})
        with torch.no_grad():
            for mode, names, key in ((PAIRS[1].abstraction, ("contact", "supports"), "relations"),
                                     (PAIRS[2].abstraction, ("steady-state", "structure-unstable"), "macros")):
                logits, _ = model.symbols(data["z"][0, :length+1].to(args.device), mode)
                predictions = logits.cpu() >= 0
                labels = data[key][0, :length+1].bool(); mask = data[key+"_mask"][0, :length+1].bool()
                for channel, name in enumerate(names):
                    pred, true, valid = predictions[..., channel], labels[..., channel], mask[..., channel]
                    symbolic[name].update({"true_positive": int((pred & true & valid).sum()),
                                           "false_positive": int((pred & ~true & valid).sum()),
                                           "false_negative": int((~pred & true & valid).sum()),
                                           "available_labels": int(valid.sum())})
        usage.update(f"{t['requested_horizon']}:{t['mode']}" for t in trace)
        traces.append({"lineage_index": index, "adaptive": trace, "fixed": fixed})
    return {"schema": SCHEMA, "identity": "issue-71-readiness-v1", "readiness_passed": True,
            "scope": "trained executable readiness; not predictive quality or useful adaptation",
            "contract": plan["contract"], "predictors": reports, "controller_identity": control["identity"],
            "fixed_and_adaptive_predictor_identity": checkpoint["identity"],
            "pair_usage": dict(usage), "single_pair_controller": len(usage) == 1,
            "single_description_mode": len({k.split(':')[1] for k in usage}) == 1,
            "interventions": audits, "traces": traces,
            "training_role_recursive_diagnostics": recursive,
            "training_role_symbol_diagnostics": {k: dict(v) for k, v in symbolic.items()},
            "compute": {"linear_macs_per_pair": {str(p.identity): linear_macs(model, p, True) for p in PAIRS},
                        "includes": "trunk, selected decoder/adapter, joint controller linear MACs",
                        "not_full_flops": "nonlinearities, masks and perception excluded; wall time reported separately"},
            "source_release_identity": plan["source_release_identity"],
            "role_provenance": plan["data"], "training_recipe": plan["training"], "controller_recipe": plan["controller"],
            "code_revision": plan["code"]["revision"], "archived_release": False,
            "code_snapshot": str(args.output / "plan.json"), "final_evaluation_opened": False,
            "reuse_audit": plan["reuse_audit"],
            "issue_64_authorized": False, "historical_issue_70_models": "unmatched references, not these controls"}


def smoke(args):
    """Real CNN + real training shot, tiny optimizer budgets, isolated output."""
    plan = make_plan(args)
    began = time.monotonic()
    torch.manual_seed(71)
    adapter = repair.load_repaired_adapter(args.source, args.device)
    shard = build_shard(args, plan, 1, adapter, shot_limit=1, window_limit=1)
    generator = torch.Generator().manual_seed(71)
    model = CNNHybridPredictor().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
    gradients = set()
    for cycle in range(2):
        for pair in PAIRS:
            batch = sample(shard["tensors"], 2, generator, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model, batch, pair, True); loss.backward()
            gradients.update(n for n, p in model.named_parameters() if p.grad is not None and bool((p.grad != 0).any()))
            optimizer.step()
            log(f"smoke cycle={cycle+1}/2 pair={pair.identity} loss={float(loss.detach()):.5f}")
    if gradients != set(dict(model.named_parameters())):
        raise ValueError(f"smoke missing trained gradients: {set(dict(model.named_parameters())) - gradients}")
    batch = sample(shard["tensors"], 1, generator, args.device)
    labels = trajectory_labels(model, batch)
    controller = CarrierPairController().to(args.device)
    optim = torch.optim.AdamW(controller.parameters(), lr=.001)
    length = int(batch["length"][0])
    for _ in range(4):
        optim.zero_grad(set_to_none=True)
        loss = F.cross_entropy(controller(batch["z"][0, :length], batch["action"].expand(length, -1),
                                          torch.arange(length, 0, -1, device=args.device)), labels[0, :length])
        loss.backward(); optim.step()
    aggregation_labels(model, batch, batch["z"].clone())
    audit = intervention_audit(model, batch["z"][:, 0], batch["action"])
    if not audit["passed"]:
        raise ValueError("smoke intervention failed")
    _, trace = adaptive_rollout(model, controller, batch["z"][:, 0], batch["action"], fixed_steps=15)
    with tempfile.TemporaryDirectory(prefix="issue-71-smoke-") as directory:
        path = Path(directory) / "model.pt"
        torch.save(model.state_dict(), path)
        other = CNNHybridPredictor().to(args.device)
        other.load_state_dict(torch.load(path, weights_only=True, map_location=args.device), strict=True)
        if not torch.equal(model.carrier(batch["z"][:, 0], batch["action"], PAIRS[6]),
                           other.carrier(batch["z"][:, 0], batch["action"], PAIRS[6])):
            raise ValueError("checkpoint reload changed prediction")
    result = {"schema": SCHEMA, "smoke_passed": True, "production_evidence": False,
              "real_training_lineages": 1, "optimizer_steps": 18, "intervention": audit,
              "trace": trace, "wall_seconds": time.monotonic() - began,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0}
    write(args.output / "smoke.json", result)
    log(f"real smoke passed elapsed={result['wall_seconds']:.1f}s; not readiness evidence")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "smoke-test", "prepare", "train", "train-controller", "publish", "validate", "run"):
        modes.add_argument("--" + name, action="store_true")
    parser.add_argument("--source", type=Path, default=repair.ROOT)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    try:
        if args.dry_run:
            plan = make_plan(args)
            print(json.dumps({k: v for k, v in plan.items() if k not in ("records", "code")}, indent=2))
            log("no-write dry run passed; requires prepare, two matched fits, controller labels + one aggregation round")
        elif args.smoke_test:
            smoke(args)
        elif args.prepare:
            prepare(args)
        elif args.train:
            train_models(args)
        elif args.train_controller:
            train_controller(args)
        elif args.publish:
            write(args.output / "readiness.json", readiness(args)); log("readiness published; no advancement claim")
        elif args.validate:
            actual = readiness(args)
            if actual != read(args.output / "readiness.json"):
                raise ValueError("published readiness differs from recomputed checkpoint evidence")
            log("exact readiness validation passed; #64 remains unauthorized")
        else:
            for i, (name, action) in enumerate((("prepare", prepare), ("train matched models", train_models),
                                               ("train controller", train_controller)), 1):
                log(f"stage={i}/4 {name}"); action(args)
            log("stage=4/4 publish readiness")
            write(args.output / "readiness.json", readiness(args))
            log("run complete; now run --validate")
    except (ValueError, OSError, KeyError) as error:
        log(f"error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
