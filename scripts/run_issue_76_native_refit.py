"""Operator-run common/history and independent paired native dynamics refit."""
import argparse
from collections import Counter, defaultdict
import gc
from importlib.metadata import version
from pathlib import Path
import tempfile
import time

import torch
from torch.nn import functional as F

from scripts import issue_76_expansion as files
from scripts import run_issue_76_development as collection
from scripts.issue_76_fit_budget import (fitting_budget, fit_updates, FitBudgetExceeded, FitPaused, log)
from scripts.run_issue_70_parser_repair import atomic_torch
from world_model.model import Abstraction
from world_model.training.native_history_model import NativeHistoryDynamics as HistoryDynamics, capacity_contract, PAIRS, CONTINUOUS_PAIRS
from world_model.training.native_history_data import VOCABULARY, VISUAL_DIM, prepare_episode
from world_model.training.native_history_fit import (NativeVisualParser, CommonHistoryFit, NativeHistoryController,
    perception_loss, encode_visual, encode_history, transition_batch, continuous_loss, hybrid_loss,
    controller_targets, rollout, ENDPOINT)
from world_model.training.observed_history import ObservedHistoryEncoder

ROOT = files.ROOT / ".local-artifacts/issue-76-native-refit-v2"
SEEDS = (760930001, 760930002, 760930003)
SOURCES = ("scripts/run_issue_76_native_refit.py", "scripts/issue_76_fit_budget.py",
    "world_model/training/native_history_data.py", "world_model/training/native_history_fit.py",
    "world_model/training/native_history_model.py", "world_model/training/observed_history.py",
    "world_model/training/cohort_v2_visual_parser.py", "world_model/training/cnn_hybrid.py",
    "world_model/training/matched_dynamics.py", "docs/issue-76-native-refit-protocol.md",
    "docs/issue-76-native-capacity-amendment.md", "world_model/model/predictor.py",
    "world_model/model/config.py", "world_model/model/__init__.py",
    "world_model/data/deployment_temporal.py", "scripts/run_issue_70_parser_repair.py")


def pair_for_update(pure, index):
    return CONTINUOUS_PAIRS[index % 3] if pure else PAIRS[(index % 3) * 3 + (index // 3) % 3]


def make_plan():
    data = (collection.load_plan("development") if (collection.DEVELOPMENT / "plan.json").exists()
            else collection.make_plan("development"))
    if any(m["novelty_level"] != 0 or not set(m["generated_slots"]).issubset(VOCABULARY) for m in data["members"]):
        raise ValueError("refit requires the frozen normal, represented-slot development membership")
    training_slots = {s for m in data["members"] if m["exposure_role"] == "training" for s in m["generated_slots"]}
    if training_slots | {"world:landscape:0000", "world:landscape:0001"} != set(VOCABULARY):
        raise ValueError("native vocabulary must come from the training footprint and declared landscape slots")
    with torch.device("meta"):
        visual_count = sum(p.numel() for p in NativeVisualParser().parameters())
        history_count = sum(p.numel() for p in ObservedHistoryEncoder(carrier_dim=VISUAL_DIM).parameters())
        auxiliary_count = sum(p.numel() for p in CommonHistoryFit().auxiliary.parameters())
        controllers = {name: sum(p.numel() for p in NativeHistoryController(pure).parameters())
                       for name, pure in (("hybrid", False), ("pure", True))}
    return {"schema": "issue_76_native_refit_plan_v1", "identity": ROOT.name,
            "collection_identity": data["identity"], "collection_root": str(collection.DEVELOPMENT),
            "members": data["members"], "vocabulary": list(VOCABULARY), "seeds": list(SEEDS),
            "capacity": capacity_contract(), "controller_parameters": controllers,
            "common_parameters": {"visual": visual_count, "history": history_count, "training_only_auxiliary": auxiliary_count},
            "updates": {"visual": 1500, "history": 500, "predictor": 6000, "controller": 2000},
            "limits": {"common_seconds_per_seed": 3600, "predictor_seconds_per_arm_seed": 3600,
                       "controller_diagnostic_seconds_per_arm_seed": 1800, "cpu_rss_mib": 12288,
                       "cuda_allocated_mib": 8192, "working_bytes": 256 * 2**30},
            "minimum_usable_fraction_per_role_family": .9,
            "source_text": {p: (files.ROOT / p).read_text() for p in SOURCES},
            "software": {name: version(name) for name in ("torch", "numpy", "Pillow")},
            "execution_device": "cuda", "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "collection_source_text": data["source_text"], "synthetic": False,
            "fresh_access": False, "gameplay_evidence_supplied_by_offline_diagnostic": False}


def load_plan(root=ROOT):
    saved = files.read(root / "plan.json")
    if saved != make_plan():
        raise ValueError("native refit source/membership/compute freeze changed")
    return saved


def prepare():
    if ROOT.exists():
        raise ValueError("refit already prepared; retain its plan and checkpoints")
    if not (collection.DEVELOPMENT / "plan.json").exists():
        raise ValueError("freeze the development collection before preparing the refit")
    plan = make_plan()
    files.write(ROOT / "plan.json", plan)
    log("refit frozen: common weights once per paired seed; 250 training/50 calibration/50 model selection; no fitting yet")


def working_bytes(root, plan):
    return sum(p.stat().st_size for folder in (root, Path(plan["collection_root"])) for p in folder.rglob("*") if p.is_file())


def check_disk(root, plan):
    size = working_bytes(root, plan)
    if size > plan["limits"]["working_bytes"]:
        raise FitBudgetExceeded("new collection/refit working data reached 256GiB; preserve it and stop")
    return size


def prepare_data(root, plan):
    collection_root = Path(plan["collection_root"])
    capture_budget = files.read(collection_root / "budget.json")
    if not capture_budget["stopped"] and any(not collection.c2.result_path(collection_root, m).exists() for m in plan["members"]):
        raise ValueError("finish the assigned collection or its resource stop before deriving fitting data")
    if (root / "data-index.json").exists():
        value = files.read(root / "data-index.json")
        if value["plan_identity"] != plan["identity"] or [e["member_identity"] for e in value["entries"]] != [m["identity"] for m in plan["members"]]:
            raise ValueError("existing prepared-data inventory differs")
        require_finished_budget(root, "data-preparation")
        log(f"retain completed data preparation; fit-data gate={value['fit_data_gate_passed']}")
        return
    remaining = max(0., 43200 - capture_budget["active_seconds"])
    with fitting_budget(root, "data-preparation", remaining, "cpu") as budget:
        _prepare_data(root, plan, collection_root, budget)


def _prepare_data(root, plan, collection_root, budget):
    entries, started = [], time.monotonic()
    for ordinal, member in enumerate(plan["members"], 1):
        budget.check()
        target = root / "prepared" / (member["identity"] + ".pt")
        path = collection.c2.result_path(collection_root, member)
        result = files.read(path) if path.exists() else None
        if target.exists():
            shard = torch.load(target, map_location="cpu", weights_only=False)
            if shard["plan_identity"] != plan["identity"] or shard["source_result"] != result:
                raise ValueError("prepared data source changed; do not replace the shard")
        else:
            failure = None
            shard = None
            if result:
                try:
                    shard = prepare_episode(collection_root, member, result)
                except (ValueError, OSError) as error:
                    failure = f"{type(error).__name__}: {error}"
                    log(f"preparation failure retained {member['identity']}: {failure}")
            if shard is None:
                shard = {"member_identity": member["identity"], "base_cluster": member["base_cluster"],
                         "exposure_role": member["exposure_role"], "tensors": {}, "references": [],
                         "source_result": result, "segment_ranges": [], "synthetic": False}
            shard["preparation_failure"] = failure
            shard["plan_identity"] = plan["identity"]
            atomic_torch(target, shard)
        frames = len(shard["references"])
        entries.append({"member_identity": member["identity"], "base_cluster": member["base_cluster"],
                        "exposure_role": member["exposure_role"], "family": member["generator_family"],
                        "path": str(target.relative_to(root)), "frames": frames, "usable": frames >= 2,
                        "preparation_failure": shard["preparation_failure"],
                        "collection_complete": bool(result and result["complete"]),
                        "gameplay_success": bool(result and result.get("gameplay_success"))})
        elapsed = time.monotonic() - started
        size = check_disk(root, plan)
        budget.check()
        budget.save()
        log(f"prepare-data member={ordinal}/{len(plan['members'])} frames={frames} elapsed={elapsed:.1f}s ETA={elapsed/ordinal*(len(plan['members'])-ordinal):.1f}s bytes={size}")
    groups = defaultdict(list)
    for entry in entries:
        groups[entry["exposure_role"], entry["family"]].append(entry)
    coverage = [{"role": role, "family": family, "assigned": len(values), "usable": sum(e["usable"] for e in values)}
                for (role, family), values in sorted(groups.items())]
    passed = all(c["usable"] / c["assigned"] >= plan["minimum_usable_fraction_per_role_family"] for c in coverage)
    value = {"schema": "issue_76_native_data_index_v1", "plan_identity": plan["identity"], "entries": entries,
             "coverage": coverage, "fit_data_gate_passed": passed,
             "preparation_budget_key": "data-preparation"}
    target = root / "data-index.json"
    if target.exists():
        prior = files.read(target)
        if any(prior[k] != value[k] for k in ("plan_identity", "entries", "coverage", "fit_data_gate_passed")):
            raise ValueError("prepared data index differs from its existing publication")
    else:
        files.write(target, value)
    log(f"data preparation complete; fit-data gate={passed}; failures/unattempted members retained")


def data_index(root, plan):
    value = files.read(root / "data-index.json")
    if value["plan_identity"] != plan["identity"] or not value["fit_data_gate_passed"]:
        raise ValueError("native refit requires its source-bound per-role/family data coverage gate")
    if [e["member_identity"] for e in value["entries"]] != [m["identity"] for m in plan["members"]]:
        raise ValueError("native data index omitted or reordered assigned members")
    return value


def load_shard(root, entry, plan, *, fitting=False):
    if fitting and entry["exposure_role"] != "training":
        raise ValueError("calibration/model-selection/engineering data cannot enter an optimizer")
    value = torch.load(root / entry["path"], map_location="cpu", weights_only=False)
    if (value["member_identity"] != entry["member_identity"] or value["exposure_role"] != entry["exposure_role"]
            or value["plan_identity"] != plan["identity"] or value["synthetic"] != plan["synthetic"]):
        raise ValueError("native shard identity/role/synthetic binding differs")
    return value


def generator(seed, update):
    return torch.Generator().manual_seed(seed + 1009 * update)


def visual_path(root, seed, entry):
    return root / "visual" / str(seed) / (entry["member_identity"] + ".pt")


def carrier_path(root, seed, entry):
    return root / "carriers" / str(seed) / (entry["member_identity"] + ".pt")


def common_path(root, seed):
    return root / "checkpoints" / f"common-{seed}.pt"


def model_path(root, seed, pure, name):
    return root / "checkpoints" / f"{name}-{'pure' if pure else 'hybrid'}-{seed}.pt"


def train_common(root, plan, device):
    index = data_index(root, plan)
    training = [e for e in index["entries"] if e["exposure_role"] == "training"]
    for seed in plan["seeds"]:
        final = common_path(root, seed)
        if final.exists():
            value = torch.load(final, map_location="cpu", weights_only=False)
            if value["plan_identity"] != plan["identity"] or value["seed"] != seed or value["synthetic"] != plan["synthetic"]:
                raise ValueError("common checkpoint plan differs")
            require_finished_budget(root, f"common-{seed}")
            log(f"retain completed common seed={seed}")
            continue
        torch.manual_seed(seed)
        with fitting_budget(root, f"common-{seed}", plan["limits"]["common_seconds_per_seed"], device) as budget:
            parser = NativeVisualParser().to(device)
            def parser_loss(update):
                entry = training[update % len(training)]
                if not entry["usable"]:
                    return None
                shard = load_shard(root, entry, plan, fitting=True)
                tensors = shard["tensors"]
                rows = torch.randint(entry["frames"], (32,), generator=generator(seed, update))
                return perception_loss(parser, {k: tensors[k][rows].to(device) for k in ("images", "presence", "centers")})
            fit_updates(root / "checkpoints" / f"visual-{seed}.pt", parser, plan_identity=plan["identity"],
                        updates=plan["updates"]["visual"], lr=.001, make_loss=parser_loss, budget=budget, device=device)
            parser.eval().requires_grad_(False)
            for ordinal, entry in enumerate(index["entries"], 1):
                budget.check()
                if not entry["usable"]:
                    continue
                target = visual_path(root, seed, entry)
                if not target.exists():
                    shard = load_shard(root, entry, plan)
                    visual = encode_visual(parser, shard["tensors"])
                    atomic_torch(target, {"plan_identity": plan["identity"], "seed": seed,
                        "member_identity": entry["member_identity"], "visual": visual})
                if ordinal % 10 == 0 or ordinal == len(index["entries"]):
                    budget.save()
                    log(f"common seed={seed} visual-cache={ordinal}/{len(index['entries'])} active={budget.value['active_seconds']:.1f}s")
            common = CommonHistoryFit().to(device)
            def history_loss(update):
                entry = training[update % len(training)]
                if not entry["usable"]:
                    return None
                shard = load_shard(root, entry, plan, fitting=True)
                cached = torch.load(visual_path(root, seed, entry), map_location="cpu", weights_only=False)
                return common.loss(cached["visual"].to(device), shard["tensors"]["timestamps"].to(device), shard["segment_ranges"])
            fit_updates(root / "checkpoints" / f"history-{seed}.pt", common, plan_identity=plan["identity"],
                        updates=plan["updates"]["history"], lr=.001, make_loss=history_loss, budget=budget, device=device)
            common.eval().requires_grad_(False)
            for ordinal, entry in enumerate(index["entries"], 1):
                budget.check()
                if not entry["usable"]:
                    continue
                target = carrier_path(root, seed, entry)
                if not target.exists():
                    shard = load_shard(root, entry, plan)
                    cached = torch.load(visual_path(root, seed, entry), map_location="cpu", weights_only=False)
                    with torch.no_grad():
                        carrier = encode_history(common.history, cached["visual"].to(device),
                            shard["tensors"]["timestamps"].to(device), shard["segment_ranges"]).cpu()
                    atomic_torch(target, {"plan_identity": plan["identity"], "seed": seed,
                        "member_identity": entry["member_identity"], "carrier": carrier})
                if ordinal % 10 == 0 or ordinal == len(index["entries"]):
                    budget.save()
                    log(f"common seed={seed} history-cache={ordinal}/{len(index['entries'])} active={budget.value['active_seconds']:.1f}s")
            budget.check()
            atomic_torch(final, {"schema": "issue_76_native_common_checkpoint_v1", "plan_identity": plan["identity"],
                "seed": seed, "parser": parser.state_dict(), "history": common.history.state_dict(),
                "synthetic": plan["synthetic"], "auxiliary_deployed": False})
        del parser, common
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        check_disk(root, plan)


def load_carrier(root, seed, entry, plan):
    value = torch.load(carrier_path(root, seed, entry), map_location="cpu", weights_only=False)
    if value["plan_identity"] != plan["identity"] or value["seed"] != seed or value["member_identity"] != entry["member_identity"]:
        raise ValueError("common carrier source binding differs")
    return value["carrier"]


def new_predictor(plan, pure, device):
    return HistoryDynamics(pure=pure, width=plan["capacity"]["continuous_width" if pure else "hybrid_width"]).to(device)


def train_predictors(root, plan, device):
    index = data_index(root, plan)
    training = [e for e in index["entries"] if e["exposure_role"] == "training"]
    for seed in plan["seeds"]:
        if not common_path(root, seed).exists():
            raise ValueError("both predictors require the same completed common checkpoint")
        require_finished_budget(root, f"common-{seed}")
        for pure in (False, True):
            torch.manual_seed(seed)
            name = "pure" if pure else "hybrid"
            if retain_completed_fit(root, plan, seed, pure, "predictor"):
                continue
            with fitting_budget(root, f"predictor-{name}-{seed}", plan["limits"]["predictor_seconds_per_arm_seed"], device) as budget:
                model = new_predictor(plan, pure, device)
                def loss(update):
                    entry = training[update % len(training)]
                    if not entry["usable"]:
                        return None
                    shard = load_shard(root, entry, plan, fitting=True)
                    carrier = load_carrier(root, seed, entry, plan)
                    pair = pair_for_update(pure, update)
                    batch = transition_batch(shard, carrier, pair.delta, generator(seed, update), symbolic=not pure)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    if pure:
                        return continuous_loss(model, batch["z"], batch["available"], batch["action"], pair)
                    return hybrid_loss(model, batch, pair)
                fit_updates(model_path(root, seed, pure, "predictor"), model, plan_identity=plan["identity"],
                            updates=plan["updates"]["predictor"], lr=.0003, make_loss=loss, budget=budget, device=device)
            del model
            gc.collect()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        check_disk(root, plan)


def require_finished_budget(root, key):
    value = files.read(root / "budgets" / (key + ".json"))
    if value["stopped"] or value.get("running") or value["active_seconds"] > value["limit_seconds"]:
        raise ValueError(f"{key}: required budget record is unfinished or exceeded")
    return value


def retain_completed_fit(root, plan, seed, pure, phase):
    path = model_path(root, seed, pure, phase)
    if not path.exists():
        return False
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value["plan_identity"] != plan["identity"]:
        raise ValueError("completed fit plan differs")
    if value["complete"]:
        require_finished_budget(root, f"{phase}-{'pure' if pure else 'hybrid'}-{seed}")
        log(f"retain completed {path.stem}; no new optimizer work or allowance")
        return True
    return False


def load_predictor(root, plan, seed, pure, device):
    name = "pure" if pure else "hybrid"
    require_finished_budget(root, f"common-{seed}")
    require_finished_budget(root, f"predictor-{name}-{seed}")
    value = torch.load(model_path(root, seed, pure, "predictor"), map_location=device, weights_only=False)
    if value["plan_identity"] != plan["identity"] or not value["complete"]:
        raise ValueError("predictor is not a completed source-bound paired fit")
    model = new_predictor(plan, pure, device)
    model.load_state_dict(value["model"])
    return model.eval().requires_grad_(False)


def teacher_path(root, seed, pure, entry):
    return root / "teachers" / f"{'pure' if pure else 'hybrid'}-{seed}" / (entry["member_identity"] + ".pt")


def train_controllers(root, plan, device):
    index = data_index(root, plan)
    training = [e for e in index["entries"] if e["exposure_role"] == "training"]
    for seed in plan["seeds"]:
        for pure in (False, True):
            name = "pure" if pure else "hybrid"
            torch.manual_seed(seed)
            if retain_completed_fit(root, plan, seed, pure, "controller"):
                continue
            with fitting_budget(root, f"controller-{name}-{seed}", plan["limits"]["controller_diagnostic_seconds_per_arm_seed"], device) as budget:
                model = load_predictor(root, plan, seed, pure, device)
                for ordinal, entry in enumerate(training, 1):
                    budget.check()
                    if not entry["usable"]:
                        continue
                    target = teacher_path(root, seed, pure, entry)
                    if not target.exists():
                        shard = load_shard(root, entry, plan, fitting=True)
                        carrier = load_carrier(root, seed, entry, plan).to(device)
                        parts = []
                        for segment in shard["segment_ranges"]:
                            start, stop = segment["start"], segment["stop"]
                            value = controller_targets(model, carrier[start:stop],
                                shard["tensors"]["fixed_steps"][start:stop].tolist(), segment["action"].to(device))
                            if value:
                                parts.append({k: v.cpu() for k, v in value.items()})
                        tensors = {k: torch.cat([p[k] for p in parts]) for k in parts[0]} if parts else {}
                        atomic_torch(target, {"plan_identity": plan["identity"], "member_identity": entry["member_identity"],
                            "seed": seed, "pure": pure, "exposure_role": "training", "tensors": tensors})
                    if ordinal % 10 == 0 or ordinal == len(training):
                        budget.save()
                        log(f"controller {name} seed={seed} teacher={ordinal}/{len(training)} active={budget.value['active_seconds']:.1f}s")
                controller = NativeHistoryController(pure).to(device)
                def loss(update):
                    entry = training[update % len(training)]
                    if not entry["usable"]:
                        return None
                    value = torch.load(teacher_path(root, seed, pure, entry), map_location="cpu", weights_only=False)
                    if (value["plan_identity"] != plan["identity"] or value["exposure_role"] != "training"
                            or value["seed"] != seed or value["pure"] != pure):
                        raise ValueError("controller teacher source/role differs")
                    data = value["tensors"]
                    if not data:
                        return None
                    rows = torch.randint(len(data["z"]), (32,), generator=generator(seed, update))
                    batch = {k: v[rows].to(device) for k, v in data.items()}
                    return F.cross_entropy(controller(batch["z"], batch["action"], batch["remaining"]), batch["labels"])
                fit_updates(model_path(root, seed, pure, "controller"), controller, plan_identity=plan["identity"],
                            updates=plan["updates"]["controller"], lr=.001, make_loss=loss, budget=budget, device=device)
            del model, controller
            gc.collect()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        check_disk(root, plan)


def policy_name(pair):
    return f"fixed-{pair.delta}-{pair.abstraction.value}"


@torch.no_grad()
def score_entry(root, plan, seed, entry, model, controller, device):
    row = {"member_identity": entry["member_identity"], "base_cluster": entry["base_cluster"],
           "role": entry["exposure_role"], "family": entry["family"], "available": False,
           "collection_policy_gameplay_success": entry["gameplay_success"], "policies": {}}
    if not entry["usable"]:
        row["unavailable_reason"] = "no_usable_observed_segment"
        return row
    shard = load_shard(root, entry, plan)
    segment = shard["segment_ranges"][0]  # Frozen first-shot anchor, never best-of-segment selection.
    start, stop = segment["start"], segment["stop"]
    steps = shard["tensors"]["fixed_steps"].tolist()
    targets = [i for i in range(start, stop) if steps[i] == steps[start] + ENDPOINT]
    if len(targets) != 1:
        row["unavailable_reason"] = "first_shot_has_no_exact_4_5_second_observed_endpoint"
        return row
    target_index = targets[0]
    carrier = load_carrier(root, seed, entry, plan).to(device)
    initial, target = carrier[start], carrier[target_index]
    truth = shard["tensors"]
    row["available"] = True
    choices = [(policy_name(pair), pair) for pair in model.pairs] + [("adaptive", None)]
    for name, pair in choices:
        began = time.monotonic()
        try:
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            predicted, work = rollout(model, controller, initial, segment["action"].to(device), fixed_pair=pair)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            visual = predicted[:VISUAL_DIM].reshape(-1)
            presence = visual[2::13]
            centers = torch.stack((visual[7::13], visual[8::13]), -1)
            actual_presence = truth["presence"][target_index].to(device)
            present = actual_presence.bool()
            record = {"failure": None, "carrier_mse": float((predicted - target).square().mean()),
                      "visual_mse": float((predicted[:VISUAL_DIM] - target[:VISUAL_DIM]).square().mean()),
                      "memory_mse": float((predicted[VISUAL_DIM:] - target[VISUAL_DIM:]).square().mean()),
                      "presence_mse_to_engine": float((presence - actual_presence).square().mean()),
                      "center_mse_to_engine": float((centers[present] - truth["centers"][target_index].to(device)[present]).square().mean()) if bool(present.any()) else None,
                      "pig_count_absolute_error": float((presence[10].clamp(0, 1) - actual_presence[10]).abs()),
                      "block_count_absolute_error": float((presence[3:10].clamp(0, 1).sum() - actual_presence[3:10].sum()).abs()),
                      "carrier_absolute_bound_excess": float(F.relu(predicted.abs() - 2).max()),
                      "wall_seconds": time.monotonic() - began, "work": work,
                      "linear_macs": sum(w["linear_macs"] for w in work),
                      "mode_counts": dict(Counter(w["mode"] for w in work))}
            # Probe costs are diagnostic-only and never fed to action selection.
            # Pure has no symbolic decoder: do not fabricate its symbolic validity.
            record["symbolic_validity"] = None
            if not model.pure:
                probes = {}
                for mode, label_name, mask_name in ((Abstraction.MICRO, "relations", "relation_mask"),
                                                   (Abstraction.MACRO, "macros", "macro_mask")):
                    logits, availability = model.symbols(predicted[None], mode)
                    labels, mask = truth[label_name][target_index].to(device), truth[mask_name][target_index].to(device)
                    probes[mode.value] = {"available_labels": int(mask.sum()),
                        "brier": float((logits[0].sigmoid()[mask] - labels[mask]).square().mean()) if bool(mask.any()) else None,
                        "availability_probabilities": availability[0].sigmoid().cpu().tolist()}
                record["symbolic_validity"] = {"diagnostic_only": True, "probes": probes}
        except (ValueError, RuntimeError) as error:
            record = {"failure": f"{type(error).__name__}: {error}", "carrier_mse": 1e9,
                      "wall_seconds": time.monotonic() - began}
        row["policies"][name] = record
    row["no_model_reference"] = {"kind": "unchanged_carrier_not_a_gameplay_policy",
                                 "carrier_mse": float((initial - target).square().mean())}
    return row


def diagnose(root, plan, device):
    index = data_index(root, plan)
    entries = [e for e in index["entries"] if e["exposure_role"] in ("calibration", "model_selection")]
    for seed in plan["seeds"]:
        for pure in (False, True):
            name = "pure" if pure else "hybrid"
            folder = root / "diagnostics" / f"{name}-{seed}"
            if all((folder / (e["member_identity"] + ".json")).exists() for e in entries):
                require_finished_budget(root, f"controller-{name}-{seed}")
                log(f"retain completed diagnostic {name} seed={seed}; exact publication checked below")
                continue
            with fitting_budget(root, f"controller-{name}-{seed}", plan["limits"]["controller_diagnostic_seconds_per_arm_seed"], device) as budget:
                checkpoint = torch.load(model_path(root, seed, pure, "controller"), map_location=device, weights_only=False)
                if checkpoint["plan_identity"] != plan["identity"] or not checkpoint["complete"]:
                    raise ValueError("diagnostic requires a completed, frozen controller")
                model = load_predictor(root, plan, seed, pure, device)
                controller = NativeHistoryController(pure).to(device).eval()
                controller.load_state_dict(checkpoint["model"])
                started = time.monotonic()
                completed_now = 0
                for ordinal, entry in enumerate(entries, 1):
                    budget.check()
                    target = root / "diagnostics" / f"{name}-{seed}" / (entry["member_identity"] + ".json")
                    if not target.exists():
                        row = score_entry(root, plan, seed, entry, model, controller, device)
                        row.update(plan_identity=plan["identity"], seed=seed, pure=pure)
                        files.write(target, row)
                        completed_now += 1
                    budget.check()
                    budget.save()
                    eta = (time.monotonic() - started) / max(completed_now, 1) * (len(entries) - ordinal)
                    log(f"diagnostic {name} seed={seed} member={ordinal}/{len(entries)} active={budget.value['active_seconds']:.1f}s ETA={eta:.1f}s")
            del model, controller, checkpoint
            gc.collect()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
    publish_diagnostic(root, plan)


def publish_diagnostic(root, plan, *, validate=False):
    index = data_index(root, plan)
    expected = [e for e in index["entries"] if e["exposure_role"] in ("calibration", "model_selection")]
    rows, summaries = [], []
    for seed in plan["seeds"]:
        common_cost = require_finished_budget(root, f"common-{seed}")
        for pure in (False, True):
            name = "pure" if pure else "hybrid"
            costs = {"common_standalone": common_cost,
                     "predictor": require_finished_budget(root, f"predictor-{name}-{seed}"),
                     "controller_and_diagnostic": require_finished_budget(root, f"controller-{name}-{seed}")}
            local = []
            for entry in expected:
                path = root / "diagnostics" / f"{name}-{seed}" / (entry["member_identity"] + ".json")
                if not path.exists():
                    raise ValueError("diagnostic is incomplete; do not select baselines from a partial run")
                value = files.read(path)
                if value["plan_identity"] != plan["identity"] or value["seed"] != seed or value["pure"] != pure or value["member_identity"] != entry["member_identity"]:
                    raise ValueError("diagnostic membership/source differs")
                local.append(value)
            calibration = [r for r in local if r["role"] == "calibration" and r["available"]]
            pairs = CONTINUOUS_PAIRS if pure else PAIRS
            averages = {policy_name(p): sum(r["policies"][policy_name(p)]["carrier_mse"] for r in calibration) / len(calibration)
                        for p in pairs} if calibration else {}
            strongest = min(averages, key=averages.get) if averages else None
            selected = [r for r in local if r["role"] == "model_selection"]
            available = [r for r in selected if r["available"]]
            means = {policy: sum(r["policies"][policy]["carrier_mse"] for r in available) / len(available)
                     for policy in [*(policy_name(p) for p in pairs), "adaptive"]} if available else {}
            summaries.append({"seed": seed, "method": name, "costs": costs,
                "calibration_assigned": sum(r["role"] == "calibration" for r in local), "calibration_available": len(calibration),
                "strongest_fixed_baseline": strongest, "model_selection_assigned": len(selected),
                "model_selection_available": len(available), "mean_carrier_mse_available_only": means,
                "prediction_failures": sum(bool(p["failure"]) for r in available for p in r["policies"].values()),
                "endpoint_coverage_gate_passed": len(available) >= .9 * len(selected)})
            rows.extend(local)
    value = {"schema": "issue_76_native_refit_diagnostic_v1", "plan_identity": plan["identity"],
             "summaries": summaries, "rows": rows, "all_assigned_diagnostics_completed": True,
             "preprocessing_cost": None if plan["synthetic"] else require_finished_budget(root, "data-preparation"),
             "collection_policy_gameplay_successes": sum(e["gameplay_success"] for e in index["entries"]),
             "collection_policy_assigned_episodes": len(index["entries"]),
             "independent_gameplay_evaluation_performed": False, "action_ranking_power_established": False,
             "pure_symbolic_validity_available": False, "fresh_access_allowed": False,
             "issue_64_authorized": False, "issue_65_authorized": False,
             "disposition": "offline_prerequisites_only_gameplay_validity_archive_and_fresh_design_still_required",
             "synthetic": plan["synthetic"]}
    target = root / "diagnostic-report.json"
    if validate:
        if files.read(target) != value:
            raise ValueError("native diagnostic publication differs from its saved records and costs")
        log(f"diagnostic publication validated: {target}")
        return value
    if target.exists() and files.read(target) != value:
        raise ValueError("do not overwrite a different diagnostic publication")
    files.write(target, value)
    log(f"diagnostic published: {target}; not a supported gameplay/fresh disposition")
    return value


def dry_run():
    plan = make_plan()
    for index in range(18):
        if pair_for_update(True, index).delta != pair_for_update(False, index).delta:
            raise ValueError("paired training horizons differ")
    if plan["common_parameters"] != {"visual": 241418, "history": 69504, "training_only_auxiliary": 101952}:
        raise ValueError("common implementation differs from its prospectively costed architecture")
    log(f"no-fit dry-run passed: members={len(plan['members'])}, seeds={plan['seeds']}, common3h + predictors6h + controllers3h; old checkpoints not loaded")
    log("production data/weights are still required; this dry-run does not certify their future results")


def smoke_test(device):
    from tests.test_native_history_fit import synthetic_shard
    with tempfile.TemporaryDirectory(prefix="novphy-native-refit-synthetic-") as temporary:
        root = Path(temporary)
        plan = make_plan()
        plan.update(identity="native-refit-synthetic-test-only", seeds=[SEEDS[0]], synthetic=True,
                    collection_root=str(root / "synthetic-collection"))
        plan["updates"] = {"visual": 2, "history": 2, "predictor": 9, "controller": 2}
        plan["limits"].update(common_seconds_per_seed=180, predictor_seconds_per_arm_seed=180,
                              controller_diagnostic_seconds_per_arm_seed=180)
        entries, members = [], []
        for ordinal, role in enumerate(("training", "calibration", "model_selection")):
            shard = synthetic_shard(240, seed=760940001 + ordinal)
            identity = f"synthetic-{role}"
            shard.update(member_identity=identity, base_cluster=identity, exposure_role=role, plan_identity=plan["identity"])
            target = root / "prepared" / (identity + ".pt")
            atomic_torch(target, shard)
            entries.append({"member_identity": identity, "base_cluster": identity, "exposure_role": role,
                            "path": str(target.relative_to(root)), "family": "synthetic", "frames": 240,
                            "usable": True, "gameplay_success": False})
            members.append({"identity": identity})
        plan["members"] = members
        files.write(root / "data-index.json", {"plan_identity": plan["identity"], "entries": entries, "fit_data_gate_passed": True})
        log(f"synthetic optimizer/diagnostic smoke starts device={device}; no research lineage is fitted")
        train_common(root, plan, device)
        train_predictors(root, plan, device)
        train_controllers(root, plan, device)
        diagnose(root, plan, device)
        publish_diagnostic(root, plan, validate=True)
        before = {p.name: p.read_bytes() for p in (root / "budgets").glob("*.json")}
        train_common(root, plan, device)
        train_predictors(root, plan, device)
        train_controllers(root, plan, device)
        diagnose(root, plan, device)
        if before != {p.name: p.read_bytes() for p in (root / "budgets").glob("*.json")}:
            raise ValueError("completed-stage resume performed new charged work")
        report = files.read(root / "diagnostic-report.json")
        if not report["all_assigned_diagnostics_completed"] or report["fresh_access_allowed"]:
            raise ValueError("synthetic end-to-end smoke disposition differs")
        receipt = {"schema": "issue_76_native_refit_synthetic_smoke_v1", "device": device,
                   "updates": plan["updates"], "seed": SEEDS[0], "research_fits": 0,
                   "both_independent_arms_completed": True, "diagnostic_rows": len(report["rows"]),
                   "budget_records": {p.stem: files.read(p) for p in (root / "budgets").glob("*.json")},
                   "source_text": plan["source_text"], "completed_resume_replays": 0}
        folder = files.ROOT / "data/issue-76-native-refit-synthetic-smoke"
        ordinal = len(list(folder.glob("attempt-*.json"))) + 1
        target = folder / f"attempt-{ordinal:03}.json"
        files.write(target, receipt)
        log(f"synthetic smoke passed; receipt={target}; temporary synthetic weights/data removed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "prepare", "prepare-data", "fit-common", "fit-predictors", "fit-controllers", "diagnose", "publish", "validate", "smoke-test", "run-refit"):
        modes.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.dry_run:
            dry_run()
        elif args.smoke_test:
            smoke_test(args.device)
        elif args.prepare:
            prepare()
        else:
            plan = load_plan()
            if args.device != plan["execution_device"]:
                raise ValueError("production fitting uses the frozen CUDA device; CPU is available for synthetic tests only")
            if args.prepare_data or args.run_refit:
                prepare_data(ROOT, plan)
            if args.fit_common or args.run_refit:
                train_common(ROOT, plan, args.device)
            if args.fit_predictors or args.run_refit:
                train_predictors(ROOT, plan, args.device)
            if args.fit_controllers or args.run_refit:
                train_controllers(ROOT, plan, args.device)
            if args.diagnose or args.run_refit:
                diagnose(ROOT, plan, args.device)
            if args.publish:
                publish_diagnostic(ROOT, plan)
            if args.validate:
                publish_diagnostic(ROOT, plan, validate=True)
    except FitPaused as error:
        log(str(error))
        raise SystemExit(130)
    except FitBudgetExceeded as error:
        log(f"RESOURCE STOP: {error}; partial evidence/checkpoints retained")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
