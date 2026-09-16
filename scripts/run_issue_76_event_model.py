"""Fit and screen the frozen #76 causal event-ranking models."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from scripts import issue_76_expansion as files
from scripts import run_issue_76_native_refit as fit
from scripts.issue_76_fit_budget import fitting_budget, fit_updates, log
from scripts.run_issue_70_parser_repair import atomic_torch
from scripts.score_issue_76_fixed_development import load_cell
from world_model.training.event_ranking import (
    CLEAR_INDEX,
    STOP_KINDS,
    EventReadout,
    event_loss,
    pair_teacher,
    second_fraction,
)
from world_model.training.cnn_hybrid import linear_macs
from world_model.training.native_history_data import action_context, visual_carriers
from world_model.training.native_history_model import PAIRS, CONTINUOUS_PAIRS, STATE_DIM


ROOT = files.ROOT / ".local-artifacts/issue-76-event-model-v1"
OUTPUT = files.ROOT / "data/issue-76-event-model"
COLLECTION = files.ROOT / "data/issue-76-event-development/collection-validation.json"
PARENT_INVENTORY = files.ROOT / "data/issue-76-fixed-development/inventory.json"
SOURCES = (
    "scripts/run_issue_76_event_model.py",
    "scripts/score_issue_76_fixed_development.py",
    "world_model/training/event_ranking.py",
    "tests/test_issue_76_event_model.py",
    "docs/issue-76-event-model-protocol.md",
)


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if files.read(path) != value:
            raise ValueError(f"different immutable event-model artifact: {path}")
    else:
        files.write(path, value)


def _groups(validation):
    rows = []
    for group_index, group in enumerate(validation["groups"]):
        reference = next(i for i, row in enumerate(group["rows"])
                         if row["original_action_reference"])
        for row_index, row in enumerate(group["rows"]):
            valid = bool(row["capture_contract_validated"])
            rows.append({
                "member_identity": row["member_identity"],
                "base_cluster": group["base_cluster"],
                "exposure_role": group["exposure_role"],
                "family": group["generator_family"],
                "group_index": group_index,
                "candidate_ordinal": row["candidate_ordinal"],
                "paired_ranking_admissible": group["paired_ranking_admissible"],
                "reference_row_in_group": reference,
                "row_in_group": row_index,
                "valid": valid,
                "decision_image_path": row["decision_image_path"] if valid else None,
                "decision_time": row["target"]["decision_time_seconds"] if valid else None,
                "action": row["action"],
                "stop_kind": row["target"]["stop_kind"] if valid else None,
            })
    if len(rows) != 2600 or len({row["member_identity"] for row in rows}) != 2600:
        raise ValueError("event-model source must retain all 2,600 assigned rows")
    return rows


def make_plan():
    validation = files.read(COLLECTION)
    readiness = validation["data_readiness"]
    if (validation["identity"] != "issue-76-event-collection-validation-v1"
            or not validation["all_assigned_records_audited"]
            or not validation["capture_inventory_completed"]
            or validation["collection_resource_failure"] is not None
            or not readiness["data_ready"]
            or not readiness["supports_separate_training_protocol_preparation"]
            or validation["validated_captures"] != 2560):
        raise ValueError("event model requires the completed positive data-readiness audit")
    rows = _groups(validation)
    counts = Counter(row["stop_kind"] for row in rows
                     if row["valid"] and row["exposure_role"] == "training")
    if set(counts) != set(STOP_KINDS):
        raise ValueError("training role lacks a declared event class")
    total = sum(counts.values())
    weights = {name: total / (len(STOP_KINDS) * counts[name]) for name in STOP_KINDS}
    parent = files.read(PARENT_INVENTORY)
    cells = [cell for cell in parent["models"] if cell["recipe"] == "fixed-midpoint"]
    if len(cells) != 6:
        raise ValueError("event stage requires all six frozen fixed-midpoint parent models")
    common = fit.load_plan()
    return {
        "schema": "issue_76_event_model_plan_v1",
        "identity": ROOT.name,
        "collection_validation": str(COLLECTION),
        "collection_validation_identity": validation["identity"],
        "execution_plan_identity": validation["execution_plan_identity"],
        "assigned_records": len(rows),
        "validated_records": sum(row["valid"] for row in rows),
        "role_records": dict(Counter(row["exposure_role"] for row in rows)),
        "paired_groups": dict(Counter(row["exposure_role"] for row in rows
                                      if row["row_in_group"] == 0
                                      and row["paired_ranking_admissible"])),
        "training_class_counts": dict(counts),
        "training_class_weights": weights,
        "collection_records": rows,
        "stop_kinds": list(STOP_KINDS),
        "seeds": list(fit.SEEDS),
        "parent_inventory": str(PARENT_INVENTORY),
        "parent_inventory_identity": parent["identity"],
        "parent_models": cells,
        "common_plan_identity": common["identity"],
        "common_checkpoints": {str(seed): str(fit.common_path(fit.ROOT, seed))
                               for seed in fit.SEEDS},
        "pairs": {
            "hybrid": [fit.policy_name(pair) for pair in PAIRS],
            "pure": [fit.policy_name(pair) for pair in CONTINUOUS_PAIRS],
        },
        "event_readout": {"hidden_dim": 256, "updates": 4500,
                          "learning_rate": .001, "weight_decay": .0001,
                          "batch_unit": "one assigned lineage", "gradient_clip": 1.0},
        "selector": {"updates": 2000, "batch_size": 128, "learning_rate": .001,
                     "weight_decay": .0001, "gradient_clip": 1.0,
                     "remaining_native_steps": 750},
        "limits": {"input_preparation_seconds": 1800,
                   "feature_seconds_per_seed": 3600,
                   "fit_seconds_per_job": 1800,
                   "cpu_rss_mib": 12288, "cuda_allocated_mib": 8192},
        "readiness": {"minimum_additional_clear_hits": 1,
                      "minimum_useful_clear_hits": 1,
                      "minimum_second_mode_fraction": .05,
                      "minimum_second_horizon_fraction": .05,
                      "maximum_hybrid_to_pure_linear_mac_ratio": 1.10},
        "source_text": {name: (files.ROOT / name).read_text() for name in SOURCES},
        "event_model_training_authorized": True,
        "parent_dynamics_updated": False,
        "fresh_access": False,
        "advancement_authorized": False,
    }


def load_plan():
    plan = files.read(ROOT / "plan.json")
    if plan != make_plan():
        raise ValueError("event-model source, data binding, or numeric protocol changed")
    return plan


def prepare():
    if ROOT.exists():
        raise ValueError("event-model stage is already prepared")
    plan = make_plan()
    immutable(ROOT / "plan.json", plan)
    immutable(OUTPUT / "plan.json", plan)
    log("event-model protocol frozen; no fitting or model-selection scoring performed")


def _image(path):
    with Image.open(path) as opened:
        rgb = np.asarray(opened.convert("RGB").resize((96, 64), Image.Resampling.BILINEAR),
                         dtype=np.uint8).copy()
    return torch.from_numpy(rgb).permute(2, 0, 1)


def prepare_inputs(plan):
    target = ROOT / "inputs.pt"
    if target.exists():
        value = torch.load(target, map_location="cpu", weights_only=False)
        if value["plan_identity"] != plan["identity"] or len(value["rows"]) != 2600:
            raise ValueError("existing event inputs differ from the frozen plan")
        fit.require_finished_budget(ROOT, "input-preparation")
        return value
    rows = plan["collection_records"]
    with fitting_budget(ROOT, "input-preparation",
                        plan["limits"]["input_preparation_seconds"], "cpu") as budget:
        images = torch.zeros(len(rows), 3, 64, 96, dtype=torch.uint8)
        actions = torch.stack([action_context(row["action"]) for row in rows])
        labels = torch.full((len(rows),), -1, dtype=torch.long)
        timestamps = torch.zeros(len(rows), dtype=torch.float64)
        for index, row in enumerate(rows):
            budget.check()
            if row["valid"]:
                images[index] = _image(row["decision_image_path"])
                labels[index] = STOP_KINDS.index(row["stop_kind"])
                timestamps[index] = row["decision_time"]
            if (index + 1) % 100 == 0:
                budget.save()
                log(f"event inputs={index + 1}/{len(rows)}")
        group_starts = {}
        for index, row in enumerate(rows):
            group_starts.setdefault(row["group_index"], index)
        source_indices = torch.tensor([
            group_starts[row["group_index"]] + row["reference_row_in_group"]
            if row["paired_ranking_admissible"] else index
            for index, row in enumerate(rows)
        ])
        value = {"schema": "issue_76_event_inputs_v1", "plan_identity": plan["identity"],
                 "rows": rows, "images": images, "actions": actions, "labels": labels,
                 "timestamps": timestamps, "source_indices": source_indices}
        atomic_torch(target, value)
        budget.check()
    return value


def _load_common(plan, seed, device):
    fit.require_finished_budget(fit.ROOT, f"common-{seed}")
    saved = torch.load(plan["common_checkpoints"][str(seed)], map_location=device,
                       weights_only=False)
    if (saved["plan_identity"] != plan["common_plan_identity"] or saved["seed"] != seed
            or saved["synthetic"] or saved["auxiliary_deployed"]):
        raise ValueError("event features require the completed shared common checkpoint")
    parser = fit.NativeVisualParser().to(device).eval().requires_grad_(False)
    parser.load_state_dict(saved["parser"])
    history = fit.ObservedHistoryEncoder(carrier_dim=fit.VISUAL_DIM).to(device).eval().requires_grad_(False)
    history.load_state_dict(saved["history"])
    return parser, history


@torch.no_grad()
def _initial_carriers(parser, history, inputs, device, batch_size=64):
    outputs = []
    for start in range(0, len(inputs["images"]), batch_size):
        stop = min(start + batch_size, len(inputs["images"]))
        images = inputs["images"][start:stop].to(device)
        timestamps = inputs["timestamps"][start:stop].to(device)
        parsed = parser(images)
        visual = torch.stack([
            visual_carriers({key: value[i:i + 1] for key, value in parsed.items()},
                            timestamps[i:i + 1])[0]
            for i in range(stop - start)
        ])
        memory, _ = history(
            carriers=visual[:, None], actions=visual.new_zeros(len(visual), 1, 5),
            timestamps=timestamps[:, None],
            observation_mask=torch.ones(len(visual), 1, dtype=torch.bool, device=device),
            action_mask=torch.zeros(len(visual), 1, dtype=torch.bool, device=device),
        )
        outputs.append(torch.cat((visual, memory[:, 0]), dim=1).cpu())
        log(f"event initial carriers={stop}/{len(inputs['images'])}")
    own = torch.cat(outputs)
    return own[inputs["source_indices"]]


@torch.no_grad()
def _successors(model, initial, actions, device, batch_size=128):
    values = []
    for pair in model.pairs:
        batches = []
        for start in range(0, len(initial), batch_size):
            stop = min(start + batch_size, len(initial))
            batches.append(model.carrier(initial[start:stop].to(device),
                                         actions[start:stop].to(device), pair).cpu())
        values.append(torch.cat(batches))
    return torch.stack(values, dim=1)


def feature_path(seed):
    return ROOT / "features" / f"seed-{seed}.pt"


def prepare_features(plan, device):
    inputs = prepare_inputs(plan)
    inventory = files.read(PARENT_INVENTORY)
    for seed in plan["seeds"]:
        target = feature_path(seed)
        if target.exists():
            value = torch.load(target, map_location="cpu", weights_only=False)
            if value["plan_identity"] != plan["identity"] or value["seed"] != seed:
                raise ValueError("existing event features differ from the frozen plan")
            fit.require_finished_budget(ROOT, f"features-{seed}")
            log(f"retain completed event features seed={seed}")
            continue
        with fitting_budget(ROOT, f"features-{seed}",
                            plan["limits"]["feature_seconds_per_seed"], device) as budget:
            parser, history = _load_common(plan, seed, device)
            initial = _initial_carriers(parser, history, inputs, device)
            successors, sources = {}, {}
            for name, pure in (("hybrid", False), ("pure", True)):
                cell = next(cell for cell in plan["parent_models"]
                            if cell["seed"] == seed and cell["pure"] == pure)
                model, source = load_cell(inventory, cell, device)
                successors[name] = _successors(model, initial, inputs["actions"], device)
                sources[name] = source
                budget.check()
                budget.save()
                log(f"event features seed={seed} arm={name} complete")
                del model
            value = {"schema": "issue_76_event_features_v1", "plan_identity": plan["identity"],
                     "seed": seed, "initial": initial, "actions": inputs["actions"],
                     "labels": inputs["labels"], "rows": inputs["rows"],
                     "successors": successors, "parent_sources": sources,
                     "pair_names": plan["pairs"]}
            atomic_torch(target, value)
            budget.check()
        del parser, history
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def event_path(seed, pure):
    return ROOT / "checkpoints" / f"event-{'pure' if pure else 'hybrid'}-{seed}.pt"


def selector_path(seed, pure):
    return ROOT / "checkpoints" / f"selector-{'pure' if pure else 'hybrid'}-{seed}.pt"


def _role_groups(rows, role):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if row["exposure_role"] == role and row["valid"]:
            groups[row["group_index"]].append(index)
    return [groups[key] for key in sorted(groups)]


def _readout(plan, device):
    return EventReadout(STATE_DIM, hidden_dim=plan["event_readout"]["hidden_dim"]).to(device)


def train_models(plan, device):
    class_weights = torch.tensor([plan["training_class_weights"][name]
                                  for name in STOP_KINDS], device=device)
    for seed in plan["seeds"]:
        features = torch.load(feature_path(seed), map_location="cpu", weights_only=False)
        if features["plan_identity"] != plan["identity"]:
            raise ValueError("event feature plan differs")
        groups = _role_groups(features["rows"], "training")
        training = torch.tensor([index for group in groups for index in group])
        for name, pure in (("hybrid", False), ("pure", True)):
            torch.manual_seed(seed + int(pure))
            pairs = CONTINUOUS_PAIRS if pure else PAIRS
            head = _readout(plan, device)
            with fitting_budget(ROOT, f"event-{name}-{seed}",
                                plan["limits"]["fit_seconds_per_job"], device) as budget:
                def head_loss(update):
                    group = groups[update % len(groups)]
                    pair_index = (update // len(groups)) % len(pairs)
                    indices = torch.tensor(group)
                    logits = head(features["initial"][indices].to(device),
                                  features["successors"][name][indices, pair_index].to(device),
                                  features["actions"][indices].to(device))
                    labels = features["labels"][indices].to(device)
                    return event_loss(logits, labels, class_weights,
                                      paired_ranking=features["rows"][group[0]]["paired_ranking_admissible"])
                fit_updates(event_path(seed, pure), head, plan_identity=plan["identity"],
                            updates=plan["event_readout"]["updates"],
                            lr=plan["event_readout"]["learning_rate"],
                            make_loss=head_loss, budget=budget, device=device)
            saved = torch.load(event_path(seed, pure), map_location=device, weights_only=False)
            if not saved["complete"]:
                raise ValueError("event readout fit is incomplete")
            head.load_state_dict(saved["model"])
            head.eval().requires_grad_(False)
            pair_logits = []
            with torch.no_grad():
                for pair_index in range(len(pairs)):
                    parts = []
                    for indices in training.split(256):
                        parts.append(head(features["initial"][indices].to(device),
                                          features["successors"][name][indices, pair_index].to(device),
                                          features["actions"][indices].to(device)).cpu())
                    pair_logits.append(torch.cat(parts))
            teachers = pair_teacher(torch.stack(pair_logits, dim=1),
                                    features["labels"][training])
            selector = fit.NativeHistoryController(pure).to(device)
            with fitting_budget(ROOT, f"selector-{name}-{seed}",
                                plan["limits"]["fit_seconds_per_job"], device) as budget:
                def selector_loss(update):
                    rows = torch.randint(len(training), (plan["selector"]["batch_size"],),
                                         generator=fit.generator(seed + int(pure), update))
                    indices = training[rows]
                    remaining = torch.full((len(indices),),
                                           plan["selector"]["remaining_native_steps"],
                                           device=device)
                    logits = selector(features["initial"][indices].to(device),
                                      features["actions"][indices].to(device), remaining)
                    return F.cross_entropy(logits, teachers[rows].to(device))
                fit_updates(selector_path(seed, pure), selector, plan_identity=plan["identity"],
                            updates=plan["selector"]["updates"],
                            lr=plan["selector"]["learning_rate"],
                            make_loss=selector_loss, budget=budget, device=device)
            del head, selector, pair_logits, teachers
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()


def _load_trained(plan, seed, pure, device):
    name = "pure" if pure else "hybrid"
    fit.require_finished_budget(ROOT, f"event-{name}-{seed}")
    fit.require_finished_budget(ROOT, f"selector-{name}-{seed}")
    head_saved = torch.load(event_path(seed, pure), map_location=device, weights_only=False)
    selector_saved = torch.load(selector_path(seed, pure), map_location=device, weights_only=False)
    if (head_saved["plan_identity"] != plan["identity"] or not head_saved["complete"]
            or selector_saved["plan_identity"] != plan["identity"] or not selector_saved["complete"]):
        raise ValueError("event readout or selector checkpoint is incomplete")
    head = _readout(plan, device).eval().requires_grad_(False)
    head.load_state_dict(head_saved["model"])
    selector = fit.NativeHistoryController(pure).to(device).eval().requires_grad_(False)
    selector.load_state_dict(selector_saved["model"])
    return head, selector


@torch.no_grad()
def _seed_scores(plan, seed, pure, device):
    name = "pure" if pure else "hybrid"
    features = torch.load(feature_path(seed), map_location="cpu", weights_only=False)
    pairs = CONTINUOUS_PAIRS if pure else PAIRS
    head, selector = _load_trained(plan, seed, pure, device)
    pair_probabilities = []
    for pair_index in range(len(pairs)):
        parts = []
        for indices in torch.arange(len(features["initial"])).split(256):
            logits = head(features["initial"][indices].to(device),
                          features["successors"][name][indices, pair_index].to(device),
                          features["actions"][indices].to(device))
            parts.append(logits.softmax(-1)[:, CLEAR_INDEX].cpu())
        pair_probabilities.append(torch.cat(parts))
    probabilities = torch.stack(pair_probabilities, dim=1)
    choices = []
    for indices in torch.arange(len(features["initial"])).split(256):
        remaining = torch.full((len(indices),), plan["selector"]["remaining_native_steps"],
                               device=device)
        choices.append(selector(features["initial"][indices].to(device),
                                features["actions"][indices].to(device), remaining).argmax(1).cpu())
    choices = torch.cat(choices)
    adaptive = probabilities.gather(1, choices[:, None]).squeeze(1)
    return {"adaptive": adaptive, "fixed": probabilities, "choices": choices,
            "rows": features["rows"], "labels": features["labels"]}


def _metrics(rows, labels, scores, role):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if row["exposure_role"] == role and row["paired_ranking_admissible"]:
            groups[row["group_index"]].append(index)
    records, log_losses = [], []
    for group_index in sorted(groups):
        indices = groups[group_index]
        if not bool(torch.isfinite(scores[indices]).all()):
            raise ValueError(f"nonfinite {role} event score inventory")
        selected = max(indices, key=lambda index: (float(scores[index]), -rows[index]["candidate_ordinal"]))
        clear = [index for index in indices if int(labels[index]) == CLEAR_INDEX]
        for index in indices:
            probability = float(scores[index].clamp(1e-6, 1 - 1e-6))
            target = int(int(labels[index]) == CLEAR_INDEX)
            log_losses.append(-(target * np.log(probability) + (1 - target) * np.log(1 - probability)))
        records.append({"base_cluster": rows[selected]["base_cluster"],
                        "family": rows[selected]["family"],
                        "selected_ordinal": rows[selected]["candidate_ordinal"],
                        "selected_clear": int(labels[selected]) == CLEAR_INDEX,
                        "clear_available": bool(clear),
                        "clear_ordinals": [rows[index]["candidate_ordinal"] for index in clear]})
    hits = sum(record["selected_clear"] for record in records)
    informative = sum(record["clear_available"] for record in records)
    return {"groups": len(records), "informative_groups": informative, "clear_hits": hits,
            "clear_hit_rate_all": hits / len(records) if records else 0.,
            "clear_hit_rate_informative": hits / informative if informative else 0.,
            "binary_log_loss": float(np.mean(log_losses)) if log_losses else None,
            "records": records}


def _prior_scores(rows, labels):
    counts = defaultdict(lambda: defaultdict(lambda: [1, 2]))
    for index, row in enumerate(rows):
        if row["exposure_role"] == "training" and row["valid"]:
            cell = counts[row["family"]][row["candidate_ordinal"]]
            cell[0] += int(int(labels[index]) == CLEAR_INDEX)
            cell[1] += 1
    return torch.tensor([counts[row["family"]][row["candidate_ordinal"]][0]
                         / counts[row["family"]][row["candidate_ordinal"]][1]
                         for row in rows])


def _linear_macs(module):
    return sum(layer.in_features * layer.out_features for layer in module.modules()
               if isinstance(layer, torch.nn.Linear))


def evaluate(plan, device):
    by_arm = {name: [] for name in ("hybrid", "pure")}
    for seed in plan["seeds"]:
        by_arm["hybrid"].append(_seed_scores(plan, seed, False, device))
        by_arm["pure"].append(_seed_scores(plan, seed, True, device))
    rows = by_arm["hybrid"][0]["rows"]
    labels = by_arm["hybrid"][0]["labels"]
    if any(item["rows"] != rows or not torch.equal(item["labels"], labels)
           for values in by_arm.values() for item in values):
        raise ValueError("paired event score inventories differ")
    systems, calibration = {}, {}
    selected_pairs = {}
    for name, pure in (("hybrid", False), ("pure", True)):
        adaptive = torch.stack([item["adaptive"] for item in by_arm[name]]).mean(0)
        fixed = torch.stack([item["fixed"] for item in by_arm[name]]).mean(0)
        pairs = CONTINUOUS_PAIRS if pure else PAIRS
        candidates = []
        for pair_index, pair in enumerate(pairs):
            metric = _metrics(rows, labels, fixed[:, pair_index], "calibration")
            candidates.append(((-metric["clear_hits"], metric["binary_log_loss"], pair_index),
                               pair_index, metric))
        _, pair_index, fixed_calibration = min(candidates)
        selected_pairs[name] = pair_index
        calibration[name] = {"adaptive": _metrics(rows, labels, adaptive, "calibration"),
                             "selected_fixed_pair": fit.policy_name(pairs[pair_index]),
                             "selected_fixed": fixed_calibration}
        systems[f"{name}_adaptive"] = _metrics(rows, labels, adaptive, "model_selection")
        systems[f"{name}_fixed"] = _metrics(rows, labels, fixed[:, pair_index], "model_selection")
    pure_options = []
    for order, policy in enumerate(("adaptive", "fixed")):
        metric = calibration["pure"][policy if policy == "adaptive" else "selected_fixed"]
        pure_options.append(((-metric["clear_hits"], metric["binary_log_loss"], order), policy))
    _, primary_pure_policy = min(pure_options)
    calibration["pure"]["selected_primary_policy"] = primary_pure_policy
    systems["pure_primary"] = systems[f"pure_{primary_pure_policy}"]
    prior = _prior_scores(rows, labels)
    original = torch.tensor([1. if row["candidate_ordinal"] == 0 else 0. for row in rows])
    prior_options = []
    for order, (name, scores) in enumerate((("training_family_prior", prior),
                                            ("original_action", original))):
        metric = _metrics(rows, labels, scores, "calibration")
        prior_options.append(((-metric["clear_hits"], metric["binary_log_loss"], order),
                              name, scores, metric))
    _, prior_name, prior_scores, prior_calibration = min(prior_options)
    calibration["no_model"] = {"selected": prior_name, "metrics": prior_calibration}
    systems["no_model"] = _metrics(rows, labels, prior_scores, "model_selection")

    usage = {}
    total_macs = {}
    inventory = files.read(PARENT_INVENTORY)
    for name, pure in (("hybrid", False), ("pure", True)):
        pairs = CONTINUOUS_PAIRS if pure else PAIRS
        mode_counts, horizon_counts = Counter(), Counter()
        transition_macs = selector_macs = head_macs = 0
        for seed, item in zip(plan["seeds"], by_arm[name], strict=True):
            cell = next(cell for cell in plan["parent_models"]
                        if cell["seed"] == seed and cell["pure"] == pure)
            model, _ = load_cell(inventory, cell, "cpu")
            head, selector = _load_trained(plan, seed, pure, "cpu")
            for index, choice in enumerate(item["choices"].tolist()):
                if (rows[index]["exposure_role"] != "model_selection"
                        or not rows[index]["paired_ranking_admissible"]):
                    continue
                pair = pairs[choice]
                mode_counts[str(pair.abstraction)] += 1
                horizon_counts[str(pair.delta)] += 1
                transition_macs += linear_macs(model, pair)
                selector_macs += _linear_macs(selector)
                head_macs += _linear_macs(head)
        usage[name] = {"modes": dict(mode_counts), "horizons": dict(horizon_counts),
                       "second_mode_fraction": second_fraction(mode_counts),
                       "second_horizon_fraction": second_fraction(horizon_counts)}
        total_macs[name] = {"transition": transition_macs, "selector": selector_macs,
                            "event_readout": head_macs,
                            "total": transition_macs + selector_macs + head_macs}
    ratio = total_macs["hybrid"]["total"] / total_macs["pure"]["total"]
    hybrid = systems["hybrid_adaptive"]["clear_hits"]
    required = plan["readiness"]
    checks = {
        "complete_finite_score_inventory": all(value["groups"] == plan["paired_groups"]["model_selection"]
                                                for value in systems.values()),
        "useful_clear_hits": hybrid >= required["minimum_useful_clear_hits"],
        "beats_primary_pure": hybrid >= systems["pure_primary"]["clear_hits"]
                              + required["minimum_additional_clear_hits"],
        "beats_same_hybrid_fixed": hybrid >= systems["hybrid_fixed"]["clear_hits"]
                                   + required["minimum_additional_clear_hits"],
        "beats_no_model": hybrid >= systems["no_model"]["clear_hits"]
                          + required["minimum_additional_clear_hits"],
        "nondegenerate_mode_use": usage["hybrid"]["second_mode_fraction"]
                                  >= required["minimum_second_mode_fraction"],
        "nondegenerate_horizon_use": usage["hybrid"]["second_horizon_fraction"]
                                     >= required["minimum_second_horizon_fraction"],
        "matched_compute": ratio <= required["maximum_hybrid_to_pure_linear_mac_ratio"],
    }
    report = {"schema": "issue_76_event_model_report_v1", "plan_identity": plan["identity"],
              "collection_validation_identity": plan["collection_validation_identity"],
              "calibration": calibration, "model_selection": systems,
              "primary_pure_comparator": primary_pure_policy,
              "selected_pairs": {name: plan["pairs"][name][index]
                                 for name, index in selected_pairs.items()},
              "adaptive_usage": usage, "deployment_linear_macs": total_macs,
              "hybrid_to_pure_linear_mac_ratio": ratio, "readiness_checks": checks,
              "ready_for_fresh_protocol": all(checks.values()),
              "fresh_access": False, "advancement_authorized": False,
              "interpretation": "development event-ranking readiness only; not fresh gameplay evidence"}
    immutable(ROOT / "report.json", report)
    immutable(OUTPUT / "report.json", report)
    log(f"event-model readiness={report['ready_for_fresh_protocol']} checks={checks}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "prepare", "prepare-features", "train", "evaluate", "run"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.dry_run:
        plan = make_plan()
        print({"assigned_records": plan["assigned_records"],
               "validated_records": plan["validated_records"],
               "seeds": plan["seeds"], "fresh_access": False}, flush=True)
        return
    if args.prepare:
        prepare()
        return
    if args.run and not ROOT.exists():
        prepare()
    plan = load_plan()
    if args.prepare_features or args.run:
        prepare_features(plan, args.device)
    if args.train or args.run:
        train_models(plan, args.device)
    if args.evaluate or args.run:
        evaluate(plan, args.device)


if __name__ == "__main__":
    main()
