"""Conditioned-readout engineering intervention with frozen trajectory features."""

import argparse
from collections import Counter

import torch
from torch.nn import functional as F

from scripts import issue_76_expansion as files
from scripts import run_issue_76_balanced_event as base
from scripts import run_issue_76_balanced_native as native
from scripts import run_issue_76_event_model as old
from scripts import run_issue_76_native_refit as fit
from scripts.issue_76_fit_budget import fitting_budget, log
from scripts.issue_76_scaled_fit import backward_clipped, fit_updates
from scripts.run_issue_70_parser_repair import atomic_torch
from world_model.training.action_event_readout import ActionConditionedEventReadout, balanced_event_loss
from world_model.training.cnn_hybrid import linear_macs
from world_model.training.event_ranking import CLEAR_INDEX, STOP_KINDS, second_fraction
from world_model.training.event_trajectory_ranking import adaptive_trajectory, trajectory_pair_teacher
from world_model.training.native_history_fit import ENDPOINT
from world_model.training.native_history_model import PAIRS, CONTINUOUS_PAIRS, STATE_DIM


ROOT = files.ROOT / ".local-artifacts/issue-76-conditioned-event-engineering-v1"
OUTPUT = files.ROOT / "data/issue-76-conditioned-event-engineering"
SOURCES = (
    "scripts/run_issue_76_conditioned_event.py",
    "world_model/training/action_event_readout.py",
    "tests/test_issue_76_action_event_readout.py",
    "docs/issue-76-conditioned-event-engineering-protocol.md",
    "scripts/run_issue_76_balanced_event.py",
    "world_model/training/event_trajectory_ranking.py",
    "scripts/issue_76_scaled_fit.py",
)


def make_plan():
    source = base.load_plan()
    report = files.read(base.ROOT / "report.json")
    if report["plan_identity"] != source["identity"] or report["engineering_promising"]:
        raise ValueError("conditioned readout requires the retained failed trajectory engineering screen")
    return {
        "schema": "issue_76_conditioned_event_engineering_plan_v1",
        "identity": ROOT.name,
        "feature_source_plan": source,
        "seeds": source["seeds"],
        "pairs": source["pairs"],
        "pair_costs": source["pair_costs"],
        "paired_groups": source["paired_groups"],
        "training_class_weights": source["training_class_weights"],
        "selector_compute_weight": source["selector_compute_weight"],
        "event_readout": {"hidden_dim": 256, "action_scale": 6.0, "updates": 9000,
                          "learning_rate": .001, "batch": "all valid training records",
                          "loss": "globally weighted classification mean plus mean within-lineage ranking"},
        "selector": source["selector"],
        "limits": source["limits"],
        "engineering_screen": source["engineering_screen"],
        "source_text": {name: (files.ROOT / name).read_text() for name in SOURCES},
        "dynamics_or_features_changed": False,
        "conditioning_selected_from_training_fit_only": True,
        "old_model_selection_is_engineering_only": True,
        "readiness_claimed": False,
        "fresh_access": False,
        "advancement_authorized": False,
    }


def load_plan():
    plan = files.read(ROOT / "plan.json")
    if plan != make_plan():
        raise ValueError("conditioned event engineering freeze changed")
    return plan


def prepare():
    if ROOT.exists():
        raise ValueError("conditioned event engineering stage is already prepared")
    plan = make_plan()
    old.immutable(ROOT / "plan.json", plan)
    old.immutable(OUTPUT / "plan.json", plan)
    log("conditioned event engineering protocol frozen; no held-out scoring performed")


def _features(plan, seed):
    return base._features(plan["feature_source_plan"], seed)


def _head(plan, device):
    return ActionConditionedEventReadout(STATE_DIM,
        hidden_dim=plan["event_readout"]["hidden_dim"],
        action_scale=plan["event_readout"]["action_scale"]).to(device)


def _checkpoint(seed, pure, stage):
    return fit.model_path(ROOT, seed, pure, stage)


@torch.no_grad()
def _pair_logits(head, features, name, indices, device):
    return torch.stack([torch.cat([head(features["initial"][part].to(device),
        features["endpoints"][name][part, pair_index].to(device),
        features["actions"][part].to(device)).cpu() for part in indices.split(256)])
        for pair_index in range(features["endpoints"][name].shape[1])], dim=1)


def _training_groups(features, device):
    groups = old._role_groups(features["rows"], "training")
    training = torch.tensor([index for group in groups for index in group])
    position = {int(index): offset for offset, index in enumerate(training)}
    batches = [torch.tensor([position[index] for index in group], device=device) for group in groups]
    paired = [features["rows"][group[0]]["paired_ranking_admissible"] for group in groups]
    return training, batches, paired


def train_models(plan, device):
    weights = torch.tensor([plan["training_class_weights"][name] for name in STOP_KINDS], device=device)
    for seed in plan["seeds"]:
        features = _features(plan, seed)
        training, groups, paired = _training_groups(features, device)
        initial = features["initial"][training].to(device)
        action = features["actions"][training].to(device)
        labels = features["labels"][training].to(device)
        for name, pure in (("hybrid", False), ("pure", True)):
            torch.manual_seed(seed)
            head = _head(plan, device)
            endpoints = features["endpoints"][name][training].to(device)
            if not fit.retain_completed_fit(ROOT, plan, seed, pure, "event"):
                with fitting_budget(ROOT, f"event-{name}-{seed}", plan["limits"]["fit_seconds_per_job"], device) as budget:
                    def head_loss(update):
                        logits = head(initial, endpoints[:, update % endpoints.shape[1]], action)
                        return balanced_event_loss(logits, labels, weights, groups, paired)
                    fit_updates(_checkpoint(seed, pure, "event"), head,
                        plan_identity=plan["identity"], updates=plan["event_readout"]["updates"],
                        lr=plan["event_readout"]["learning_rate"], make_loss=head_loss,
                        budget=budget, device=device, backward_scale=1.0)
            saved = torch.load(_checkpoint(seed, pure, "event"), map_location=device, weights_only=False)
            head.load_state_dict(saved["model"])
            head.eval().requires_grad_(False)
            logits = _pair_logits(head, features, name, training, device)
            teachers = trajectory_pair_teacher(logits, features["labels"][training],
                torch.tensor(plan["pair_costs"][name]), plan["selector_compute_weight"])
            examples = features["teacher_states"][name][training, teachers]
            selector = fit.NativeHistoryController(pure).to(device)
            if not fit.retain_completed_fit(ROOT, plan, seed, pure, "selector"):
                with fitting_budget(ROOT, f"selector-{name}-{seed}", plan["limits"]["fit_seconds_per_job"], device) as budget:
                    def selector_loss(update):
                        generator = fit.generator(seed, update)
                        rows = torch.randint(len(training), (plan["selector"]["batch_size"],), generator=generator)
                        snapshots = torch.randint(len(base.CHECKPOINTS), (len(rows),), generator=generator)
                        indices = training[rows]
                        remaining = (ENDPOINT - torch.tensor(base.CHECKPOINTS)[snapshots]).to(device)
                        logits = selector(examples[rows, snapshots].to(device),
                            features["actions"][indices].to(device), remaining)
                        return F.cross_entropy(logits, teachers[rows].to(device))
                    fit_updates(_checkpoint(seed, pure, "selector"), selector,
                        plan_identity=plan["identity"], updates=plan["selector"]["updates"],
                        lr=plan["selector"]["learning_rate"], make_loss=selector_loss,
                        budget=budget, device=device, backward_scale=1.0)
            del head, selector, endpoints, logits, teachers, examples
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()


def _trained(plan, seed, pure, device):
    name = "pure" if pure else "hybrid"
    values = {}
    for stage in ("event", "selector"):
        fit.require_finished_budget(ROOT, f"{stage}-{name}-{seed}")
        saved = torch.load(_checkpoint(seed, pure, stage), map_location=device, weights_only=False)
        if saved["plan_identity"] != plan["identity"] or not saved["complete"] or saved["failure"]:
            raise ValueError("conditioned event trained checkpoint is incomplete")
        values[stage] = saved
    head = _head(plan, device).eval().requires_grad_(False)
    head.load_state_dict(values["event"]["model"])
    selector = fit.NativeHistoryController(pure).to(device).eval().requires_grad_(False)
    selector.load_state_dict(values["selector"]["model"])
    return head, selector


@torch.no_grad()
def _seed_scores(plan, seed, pure, device):
    name = "pure" if pure else "hybrid"
    path = ROOT / "scores" / f"{name}-{seed}.pt"
    key = f"score-{name}-{seed}"
    failure_path = ROOT / "failures" / (key + ".json")
    if path.exists():
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value["plan_identity"] != plan["identity"]:
            raise ValueError("conditioned event existing scores differ")
        fit.require_finished_budget(ROOT, key)
        return value
    if failure_path.exists():
        raise ValueError("failed conditioned event score job requires a new audit")
    features = _features(plan, seed)
    head, selector = _trained(plan, seed, pure, device)
    model = fit.load_predictor(native.ROOT, plan["feature_source_plan"]["native_plan"], seed, pure, device)
    with fitting_budget(ROOT, key, plan["limits"]["score_seconds_per_arm_seed"], device) as budget:
        try:
            fixed = _pair_logits(head, features, name, torch.arange(len(features["initial"])), device)
            fixed = fixed.softmax(-1)[:, :, CLEAR_INDEX]
            probabilities, choices = [], []
            for indices in torch.arange(len(features["initial"])).split(64):
                initial = features["initial"][indices].to(device)
                actions = features["actions"][indices].to(device)
                endpoint, counts = adaptive_trajectory(model, selector, initial, actions)
                probabilities.append(head(initial, endpoint, actions).softmax(-1)[:, CLEAR_INDEX].cpu())
                choices.append(counts.cpu())
                budget.check()
            value = {"plan_identity": plan["identity"], "seed": seed, "pure": pure,
                     "adaptive": torch.cat(probabilities), "fixed": fixed, "counts": torch.cat(choices)}
            if not bool(torch.isfinite(value["adaptive"]).all()) or not bool(torch.isfinite(fixed).all()):
                raise ValueError("nonfinite conditioned event score inventory")
            atomic_torch(path, value)
        except Exception as error:
            old.immutable(failure_path, {"plan_identity": plan["identity"],
                "failure": f"{type(error).__name__}: {error}"})
            raise
    return value


def evaluate(plan, device):
    by_arm = {name: [_seed_scores(plan, seed, pure, device) for seed in plan["seeds"]]
              for name, pure in (("hybrid", False), ("pure", True))}
    features = _features(plan, plan["seeds"][0])
    rows, labels = features["rows"], features["labels"]
    systems, calibration, selected = {}, {}, {}
    for name, pure in (("hybrid", False), ("pure", True)):
        adaptive = torch.stack([item["adaptive"] for item in by_arm[name]]).mean(0)
        fixed = torch.stack([item["fixed"] for item in by_arm[name]]).mean(0)
        pairs = CONTINUOUS_PAIRS if pure else PAIRS
        options = []
        for index, pair in enumerate(pairs):
            metric = old._metrics(rows, labels, fixed[:, index], "calibration")
            options.append(((-metric["clear_hits"], metric["binary_log_loss"], index), index, metric))
        _, index, fixed_calibration = min(options)
        selected[name] = index
        calibration[name] = {"adaptive": old._metrics(rows, labels, adaptive, "calibration"),
            "selected_fixed_pair": fit.policy_name(pairs[index]), "selected_fixed": fixed_calibration}
        systems[name + "_adaptive"] = old._metrics(rows, labels, adaptive, "model_selection")
        systems[name + "_fixed"] = old._metrics(rows, labels, fixed[:, index], "model_selection")
    options = []
    for order, policy in enumerate(("adaptive", "fixed")):
        metric = calibration["pure"]["adaptive" if policy == "adaptive" else "selected_fixed"]
        options.append(((-metric["clear_hits"], metric["binary_log_loss"], order), policy))
    _, pure_primary = min(options)
    systems["pure_primary"] = systems["pure_" + pure_primary]
    calibration["pure"]["selected_primary_policy"] = pure_primary
    prior = old._prior_scores(rows, labels)
    original = torch.tensor([1. if row["candidate_ordinal"] == 0 else 0. for row in rows])
    options = []
    for order, (name, scores) in enumerate((("training_family_prior", prior), ("original_action", original))):
        metric = old._metrics(rows, labels, scores, "calibration")
        options.append(((-metric["clear_hits"], metric["binary_log_loss"], order), name, scores, metric))
    _, name, scores, metric = min(options)
    calibration["no_model"] = {"selected": name, "metrics": metric}
    systems["no_model"] = old._metrics(rows, labels, scores, "model_selection")

    admissible = torch.tensor([index for index, row in enumerate(rows)
        if row["exposure_role"] == "model_selection" and row["paired_ranking_admissible"]])
    usage, macs = {}, {}
    for name, pure in (("hybrid", False), ("pure", True)):
        modes, horizons = Counter(), Counter()
        adaptive_cost = {"transition": 0, "selector": 0, "event_readout": 0}
        fixed_cost = {"transition": 0, "selector": 0, "event_readout": 0}
        for seed, item in zip(plan["seeds"], by_arm[name], strict=True):
            model = fit.load_predictor(native.ROOT, plan["feature_source_plan"]["native_plan"], seed, pure, "cpu")
            head, selector = _trained(plan, seed, pure, "cpu")
            counts = item["counts"][admissible].sum(0)
            for pair, count in zip(model.pairs, counts.tolist(), strict=True):
                modes[pair.abstraction.value] += count
                horizons[str(pair.delta)] += count
                adaptive_cost["transition"] += count * linear_macs(model, pair)
            adaptive_cost["selector"] += int(counts.sum()) * old._linear_macs(selector)
            adaptive_cost["event_readout"] += len(admissible) * old._linear_macs(head)
            pair = model.pairs[selected[name]]
            fixed_cost["transition"] += len(admissible) * (ENDPOINT // pair.delta) * linear_macs(model, pair)
            fixed_cost["event_readout"] += len(admissible) * old._linear_macs(head)
        usage[name] = {"modes": dict(modes), "horizons": dict(horizons),
            "second_mode_fraction": second_fraction(modes), "second_horizon_fraction": second_fraction(horizons)}
        adaptive_cost["total"] = sum(adaptive_cost.values())
        fixed_cost["total"] = sum(fixed_cost.values())
        macs[name + "_adaptive"] = adaptive_cost
        macs[name + "_fixed"] = fixed_cost
    macs["pure_primary"] = macs["pure_" + pure_primary]
    ratio = macs["hybrid_adaptive"]["total"] / macs["pure_primary"]["total"]
    hybrid, required = systems["hybrid_adaptive"]["clear_hits"], plan["engineering_screen"]
    checks = {
        "complete_finite_score_inventory": all(item["groups"] == plan["paired_groups"]["model_selection"] for item in systems.values()),
        "useful_clear_hits": hybrid >= required["minimum_useful_clear_hits"],
        "beats_primary_pure": hybrid >= systems["pure_primary"]["clear_hits"] + required["minimum_additional_clear_hits"],
        "beats_same_hybrid_fixed": hybrid >= systems["hybrid_fixed"]["clear_hits"] + required["minimum_additional_clear_hits"],
        "beats_no_model": hybrid >= systems["no_model"]["clear_hits"] + required["minimum_additional_clear_hits"],
        "nondegenerate_mode_use": usage["hybrid"]["second_mode_fraction"] >= required["minimum_second_mode_fraction"],
        "nondegenerate_horizon_use": usage["hybrid"]["second_horizon_fraction"] >= required["minimum_second_horizon_fraction"],
        "matched_primary_pure_compute": ratio <= required["maximum_hybrid_to_primary_pure_linear_mac_ratio"],
    }
    report = {"schema": "issue_76_conditioned_event_engineering_report_v1", "plan_identity": plan["identity"],
        "feature_source_plan_identity": plan["feature_source_plan"]["identity"], "calibration": calibration,
        "exposed_model_selection_engineering": systems, "adaptive_usage": usage,
        "deployment_linear_macs": macs, "hybrid_to_primary_pure_linear_mac_ratio": ratio,
        "engineering_checks": checks, "engineering_promising": all(checks.values()),
        "ready_for_fresh_protocol": False, "readiness_claimed": False, "fresh_access": False,
        "advancement_authorized": False,
        "interpretation": "already exposed cohort: conditioned-readout engineering only; disjoint development readiness remains required"}
    old.immutable(ROOT / "report.json", report)
    old.immutable(OUTPUT / "report.json", report)
    log(f"conditioned event engineering promising={report['engineering_promising']} checks={checks}")
    return report


def smoke_test(device):
    plan = make_plan()
    features = _features(plan, plan["seeds"][0])
    training, groups, paired = _training_groups(features, device)
    weights = torch.tensor([plan["training_class_weights"][name] for name in STOP_KINDS], device=device)
    for name in ("hybrid", "pure"):
        head = _head(plan, device)
        logits = head(features["initial"][training].to(device),
            features["endpoints"][name][training, 0].to(device),
            features["actions"][training].to(device))
        loss = balanced_event_loss(logits, features["labels"][training].to(device), weights, groups, paired)
        backward_clipped(loss, head.parameters(), backward_scale=1.0)
        log(f"conditioned event real smoke arm={name} training_records={len(training)} loss={float(loss.detach())}")
    log("conditioned event real smoke passed; no research fitting artifact written")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "smoke-test", "prepare", "train", "evaluate", "run"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.dry_run:
        plan = make_plan()
        print({"identity": plan["identity"], "seeds": plan["seeds"],
               "readout_updates": plan["event_readout"]["updates"], "readiness_claimed": False}, flush=True)
        return
    if args.smoke_test:
        smoke_test(args.device)
        return
    if args.prepare:
        prepare()
        return
    plan = load_plan()
    if args.train or args.run:
        train_models(plan, args.device)
    if args.evaluate or args.run:
        evaluate(plan, args.device)


if __name__ == "__main__":
    main()
