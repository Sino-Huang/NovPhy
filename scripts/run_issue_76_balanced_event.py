"""Full-rollout event-ranking engineering screen on the already exposed cohort."""

import argparse
from collections import Counter

import torch
from torch.nn import functional as F

from scripts import issue_76_expansion as files
from scripts import resume_issue_76_balanced_native_norm as recovery
from scripts import run_issue_76_balanced_native as native
from scripts import run_issue_76_event_model as previous
from scripts import run_issue_76_native_refit as fit
from scripts.issue_76_fit_budget import fitting_budget, log
from scripts.issue_76_scaled_fit import backward_clipped, fit_updates
from scripts.run_issue_70_parser_repair import atomic_torch
from world_model.training.cnn_hybrid import linear_macs
from world_model.training.event_ranking import CLEAR_INDEX, STOP_KINDS, EventReadout, event_loss, second_fraction
from world_model.training.event_trajectory_ranking import (
    adaptive_trajectory,
    fixed_trajectory,
    trajectory_pair_teacher,
)
from world_model.training.native_history_fit import ENDPOINT, REFERENCE_TRANSITION_MACS
from world_model.training.native_history_model import PAIRS, CONTINUOUS_PAIRS, STATE_DIM


ROOT = files.ROOT / ".local-artifacts/issue-76-balanced-event-engineering-v1"
OUTPUT = files.ROOT / "data/issue-76-balanced-event-engineering"
CHECKPOINTS = (0, 3750, 7500, 10500)
SOURCES = (
    "scripts/run_issue_76_balanced_event.py",
    "world_model/training/event_trajectory_ranking.py",
    "tests/test_issue_76_event_trajectory.py",
    "docs/issue-76-balanced-event-engineering-protocol.md",
    "scripts/run_issue_76_event_model.py",
    "world_model/training/event_ranking.py",
    "scripts/issue_76_scaled_fit.py",
)


def _native_binding():
    plan = files.read(recovery.ROOT / "plan.json")
    report = files.read(recovery.ROOT / "report.json")
    if (plan != recovery.make_plan() or report["recovery_plan"] != plan
            or not report["all_jobs_complete"] or not report["parent_report"]["all_jobs_complete"]
            or report["fresh_access"] or report["advancement_authorized"]):
        raise ValueError("balanced event stage requires the completed audited native recovery")
    if files.read(native.ROOT / "plan.json") != plan["parent_plan"]:
        raise ValueError("balanced event native parent binding differs")
    return plan["parent_plan"], plan["identity"]


def make_plan():
    old = previous.load_plan()
    old_report = files.read(previous.ROOT / "report.json")
    if old_report["plan_identity"] != old["identity"] or old_report["ready_for_fresh_protocol"]:
        raise ValueError("balanced engineering stage requires the retained exposed event-stage failure")
    native_plan, recovery_identity = _native_binding()
    fit.require_finished_budget(previous.ROOT, "input-preparation")
    pair_costs = {}
    with torch.device("meta"):
        for name, pure in (("hybrid", False), ("pure", True)):
            model = fit.new_predictor(native_plan, pure, "meta")
            pair_costs[name] = [(ENDPOINT // pair.delta) * linear_macs(model, pair)
                                / REFERENCE_TRANSITION_MACS for pair in model.pairs]
    return {
        "schema": "issue_76_balanced_event_engineering_plan_v1",
        "identity": ROOT.name,
        "input_source_plan_identity": old["identity"],
        "input_source_path": str(previous.ROOT / "inputs.pt"),
        "collection_validation_identity": old["collection_validation_identity"],
        "collection_records": old["collection_records"],
        "paired_groups": old["paired_groups"],
        "training_class_weights": old["training_class_weights"],
        "native_plan": native_plan,
        "native_recovery_plan_identity": recovery_identity,
        "native_recovery_report": str(recovery.ROOT / "report.json"),
        "seeds": native_plan["seeds"],
        "pairs": {"hybrid": [fit.policy_name(pair) for pair in PAIRS],
                  "pure": [fit.policy_name(pair) for pair in CONTINUOUS_PAIRS]},
        "endpoint_native_steps": ENDPOINT,
        "selector_checkpoint_elapsed_steps": list(CHECKPOINTS),
        "pair_costs": pair_costs,
        "selector_compute_weight": .0001,
        "event_readout": {"hidden_dim": 256, "updates": 4500, "learning_rate": .001},
        "selector": {"updates": 2000, "batch_size": 128, "learning_rate": .001},
        "limits": {"feature_seconds_per_seed": 3600, "fit_seconds_per_job": 1800,
                   "score_seconds_per_arm_seed": 3600},
        "engineering_screen": {"minimum_additional_clear_hits": 1,
            "minimum_useful_clear_hits": 1, "minimum_second_mode_fraction": .05,
            "minimum_second_horizon_fraction": .05,
            "maximum_hybrid_to_primary_pure_linear_mac_ratio": 1.10},
        "source_text": {name: (files.ROOT / name).read_text() for name in SOURCES},
        "old_model_selection_is_engineering_only": True,
        "new_dynamics_fitting": False,
        "fresh_access": False,
        "readiness_claimed": False,
        "advancement_authorized": False,
    }


def load_plan():
    plan = files.read(ROOT / "plan.json")
    if plan != make_plan():
        raise ValueError("balanced event engineering freeze changed")
    return plan


def prepare():
    if ROOT.exists():
        raise ValueError("balanced event engineering stage is already prepared")
    plan = make_plan()
    previous.immutable(ROOT / "plan.json", plan)
    previous.immutable(OUTPUT / "plan.json", plan)
    log("balanced event engineering protocol frozen; no readiness scoring performed")


def _inputs(plan):
    value = torch.load(plan["input_source_path"], map_location="cpu", weights_only=False)
    if value["plan_identity"] != plan["input_source_plan_identity"] or value["rows"] != plan["collection_records"]:
        raise ValueError("balanced event input source inventory differs")
    expected = torch.tensor([STOP_KINDS.index(row["stop_kind"]) if row["valid"] else -1
                             for row in value["rows"]])
    if not torch.equal(value["labels"], expected):
        raise ValueError("balanced event input labels differ from validated stop targets")
    return value


def _common(plan, seed, device):
    fit.require_finished_budget(native.ROOT, f"common-{seed}")
    saved = torch.load(fit.common_path(native.ROOT, seed), map_location=device, weights_only=False)
    if saved["plan_identity"] != plan["native_plan"]["identity"] or saved["seed"] != seed:
        raise ValueError("balanced event common checkpoint differs")
    parser = native.load_parser(plan["native_plan"], seed, device)
    history = fit.ObservedHistoryEncoder(carrier_dim=fit.VISUAL_DIM).to(device).eval().requires_grad_(False)
    history.load_state_dict(saved["history"])
    return parser, history


def feature_path(seed):
    return ROOT / "features" / f"seed-{seed}.pt"


def _features(plan, seed):
    value = torch.load(feature_path(seed), map_location="cpu", weights_only=False)
    if value["plan_identity"] != plan["identity"] or value["seed"] != seed:
        raise ValueError("balanced event feature source differs")
    fit.require_finished_budget(ROOT, f"features-{seed}")
    return value


def prepare_features(plan, device):
    inputs = _inputs(plan)
    for seed in plan["seeds"]:
        target = feature_path(seed)
        failure_path = ROOT / "failures" / f"features-{seed}.json"
        if target.exists():
            _features(plan, seed)
            log(f"retain completed balanced event features seed={seed}")
            continue
        if failure_path.exists():
            raise ValueError("failed balanced event feature job requires a new audit")
        with fitting_budget(ROOT, f"features-{seed}", plan["limits"]["feature_seconds_per_seed"], device) as budget:
            try:
                parser, history = _common(plan, seed, device)
                initial = previous._initial_carriers(parser, history, inputs, device)
                if not bool(torch.isfinite(initial).all()):
                    raise ValueError("nonfinite balanced event initial carrier")
                endpoints, states = {}, {}
                for name, pure in (("hybrid", False), ("pure", True)):
                    model = fit.load_predictor(native.ROOT, plan["native_plan"], seed, pure, device)
                    pair_endpoints, pair_states = [], []
                    for pair in model.pairs:
                        ends, snapshots = [], []
                        for indices in torch.arange(len(initial)).split(128):
                            end, state = fixed_trajectory(model, initial[indices].to(device),
                                inputs["actions"][indices].to(device), pair)
                            ends.append(end.cpu())
                            snapshots.append(state.cpu())
                            budget.check()
                        pair_endpoints.append(torch.cat(ends))
                        pair_states.append(torch.cat(snapshots))
                        budget.save()
                        log(f"balanced event features seed={seed} arm={name} pair={fit.policy_name(pair)}")
                    endpoints[name] = torch.stack(pair_endpoints, dim=1)
                    states[name] = torch.stack(pair_states, dim=1)
                    del model
                atomic_torch(target, {"schema": "issue_76_balanced_event_features_v1",
                    "plan_identity": plan["identity"], "seed": seed, "initial": initial,
                    "actions": inputs["actions"], "labels": inputs["labels"], "rows": inputs["rows"],
                    "endpoints": endpoints, "teacher_states": states})
            except Exception as error:
                previous.immutable(failure_path, {"plan_identity": plan["identity"],
                    "seed": seed, "failure": f"{type(error).__name__}: {error}"})
                raise
        del parser, history
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def _head(plan, device):
    return EventReadout(STATE_DIM, hidden_dim=plan["event_readout"]["hidden_dim"]).to(device)


def _checkpoint(seed, pure, name):
    return fit.model_path(ROOT, seed, pure, name)


@torch.no_grad()
def _pair_logits(head, features, name, indices, device):
    values = []
    for pair_index in range(features["endpoints"][name].shape[1]):
        values.append(torch.cat([head(features["initial"][part].to(device),
            features["endpoints"][name][part, pair_index].to(device),
            features["actions"][part].to(device)).cpu() for part in indices.split(256)]))
    return torch.stack(values, dim=1)


def train_models(plan, device):
    weights = torch.tensor([plan["training_class_weights"][name] for name in STOP_KINDS], device=device)
    for seed in plan["seeds"]:
        features = _features(plan, seed)
        groups = previous._role_groups(features["rows"], "training")
        training = torch.tensor([index for group in groups for index in group])
        for name, pure in (("hybrid", False), ("pure", True)):
            torch.manual_seed(seed)
            head = _head(plan, device)
            pairs = CONTINUOUS_PAIRS if pure else PAIRS
            if not fit.retain_completed_fit(ROOT, plan, seed, pure, "event"):
                with fitting_budget(ROOT, f"event-{name}-{seed}", plan["limits"]["fit_seconds_per_job"], device) as budget:
                    def head_loss(update):
                        group = groups[update % len(groups)]
                        indices = torch.tensor(group)
                        pair_index = (update // len(groups)) % len(pairs)
                        logits = head(features["initial"][indices].to(device),
                            features["endpoints"][name][indices, pair_index].to(device),
                            features["actions"][indices].to(device))
                        return event_loss(logits, features["labels"][indices].to(device), weights,
                            paired_ranking=features["rows"][group[0]]["paired_ranking_admissible"])
                    fit_updates(_checkpoint(seed, pure, "event"), head, plan_identity=plan["identity"],
                        updates=plan["event_readout"]["updates"], lr=plan["event_readout"]["learning_rate"],
                        make_loss=head_loss, budget=budget, device=device, backward_scale=1.0)
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
                        snapshots = torch.randint(len(CHECKPOINTS), (len(rows),), generator=generator)
                        indices = training[rows]
                        remaining = (ENDPOINT - torch.tensor(CHECKPOINTS)[snapshots]).to(device)
                        logits = selector(examples[rows, snapshots].to(device),
                            features["actions"][indices].to(device), remaining)
                        return F.cross_entropy(logits, teachers[rows].to(device))
                    fit_updates(_checkpoint(seed, pure, "selector"), selector, plan_identity=plan["identity"],
                        updates=plan["selector"]["updates"], lr=plan["selector"]["learning_rate"],
                        make_loss=selector_loss, budget=budget, device=device, backward_scale=1.0)
            del head, selector, logits, teachers, examples
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()


def _trained(plan, seed, pure, device):
    name = "pure" if pure else "hybrid"
    values = {}
    for stage in ("event", "selector"):
        fit.require_finished_budget(ROOT, f"{stage}-{name}-{seed}")
        saved = torch.load(_checkpoint(seed, pure, stage), map_location=device, weights_only=False)
        if saved["plan_identity"] != plan["identity"] or not saved["complete"] or saved["failure"]:
            raise ValueError("balanced event trained checkpoint is incomplete")
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
            raise ValueError("balanced event existing scores differ")
        fit.require_finished_budget(ROOT, key)
        return value
    if failure_path.exists():
        raise ValueError("failed balanced event score job requires a new audit")
    features = _features(plan, seed)
    head, selector = _trained(plan, seed, pure, device)
    model = fit.load_predictor(native.ROOT, plan["native_plan"], seed, pure, device)
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
                raise ValueError("nonfinite balanced event score inventory")
            atomic_torch(path, value)
        except Exception as error:
            previous.immutable(failure_path, {"plan_identity": plan["identity"],
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
            metric = previous._metrics(rows, labels, fixed[:, index], "calibration")
            options.append(((-metric["clear_hits"], metric["binary_log_loss"], index), index, metric))
        _, index, fixed_calibration = min(options)
        selected[name] = index
        calibration[name] = {"adaptive": previous._metrics(rows, labels, adaptive, "calibration"),
            "selected_fixed_pair": fit.policy_name(pairs[index]), "selected_fixed": fixed_calibration}
        systems[name + "_adaptive"] = previous._metrics(rows, labels, adaptive, "model_selection")
        systems[name + "_fixed"] = previous._metrics(rows, labels, fixed[:, index], "model_selection")
    pure_options = []
    for order, policy in enumerate(("adaptive", "fixed")):
        metric = calibration["pure"]["adaptive" if policy == "adaptive" else "selected_fixed"]
        pure_options.append(((-metric["clear_hits"], metric["binary_log_loss"], order), policy))
    _, pure_primary = min(pure_options)
    systems["pure_primary"] = systems["pure_" + pure_primary]
    calibration["pure"]["selected_primary_policy"] = pure_primary
    prior = previous._prior_scores(rows, labels)
    original = torch.tensor([1. if row["candidate_ordinal"] == 0 else 0. for row in rows])
    options = []
    for order, (name, scores) in enumerate((("training_family_prior", prior), ("original_action", original))):
        metric = previous._metrics(rows, labels, scores, "calibration")
        options.append(((-metric["clear_hits"], metric["binary_log_loss"], order), name, scores, metric))
    _, prior_name, scores, metric = min(options)
    calibration["no_model"] = {"selected": prior_name, "metrics": metric}
    systems["no_model"] = previous._metrics(rows, labels, scores, "model_selection")

    admissible = torch.tensor([index for index, row in enumerate(rows)
        if row["exposure_role"] == "model_selection" and row["paired_ranking_admissible"]])
    usage, macs = {}, {}
    for name, pure in (("hybrid", False), ("pure", True)):
        modes, horizons = Counter(), Counter()
        adaptive_cost = {"transition": 0, "selector": 0, "event_readout": 0}
        fixed_cost = {"transition": 0, "selector": 0, "event_readout": 0}
        for seed, item in zip(plan["seeds"], by_arm[name], strict=True):
            model = fit.load_predictor(native.ROOT, plan["native_plan"], seed, pure, "cpu")
            head, selector = _trained(plan, seed, pure, "cpu")
            counts = item["counts"][admissible].sum(0)
            for pair, count in zip(model.pairs, counts.tolist(), strict=True):
                modes[pair.abstraction.value] += count
                horizons[str(pair.delta)] += count
                adaptive_cost["transition"] += count * linear_macs(model, pair)
            adaptive_cost["selector"] += int(counts.sum()) * previous._linear_macs(selector)
            adaptive_cost["event_readout"] += len(admissible) * previous._linear_macs(head)
            pair = model.pairs[selected[name]]
            fixed_cost["transition"] += len(admissible) * (ENDPOINT // pair.delta) * linear_macs(model, pair)
            fixed_cost["event_readout"] += len(admissible) * previous._linear_macs(head)
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
    report = {"schema": "issue_76_balanced_event_engineering_report_v1", "plan_identity": plan["identity"],
        "native_recovery_plan_identity": plan["native_recovery_plan_identity"], "calibration": calibration,
        "exposed_model_selection_engineering": systems, "adaptive_usage": usage,
        "deployment_linear_macs": macs, "hybrid_to_primary_pure_linear_mac_ratio": ratio,
        "engineering_checks": checks, "engineering_promising": all(checks.values()),
        "ready_for_fresh_protocol": False, "readiness_claimed": False, "fresh_access": False,
        "advancement_authorized": False,
        "interpretation": "already exposed cohort: engineering screen only; new disjoint development readiness is still required"}
    previous.immutable(ROOT / "report.json", report)
    previous.immutable(OUTPUT / "report.json", report)
    log(f"balanced event engineering promising={report['engineering_promising']} checks={checks}")
    return report


def smoke_test(device):
    plan = make_plan()
    inputs = _inputs(plan)
    seed = plan["seeds"][0]
    parser, history = _common(plan, seed, device)
    mini = {"images": inputs["images"][:2], "timestamps": inputs["timestamps"][:2],
            "source_indices": torch.zeros(2, dtype=torch.long)}
    initial = previous._initial_carriers(parser, history, mini, device).to(device)
    actions = inputs["actions"][:2].to(device)
    for pure in (False, True):
        model = fit.load_predictor(native.ROOT, plan["native_plan"], seed, pure, device)
        endpoint, snapshots = fixed_trajectory(model, initial, actions, model.pairs[0])
        selector = fit.NativeHistoryController(pure).to(device).eval()
        adaptive, counts = adaptive_trajectory(model, selector, initial, actions)
        if not bool(torch.isfinite(endpoint).all()) or not bool(torch.isfinite(adaptive).all()):
            raise ValueError("balanced event real smoke trajectory is nonfinite")
        head = _head(plan, device)
        loss = head(initial, endpoint, actions).square().mean()
        backward_clipped(loss, head.parameters(), backward_scale=1.0)
        log(f"balanced event real smoke pure={pure} snapshots={tuple(snapshots.shape)} calls={counts.sum(1).tolist()}")
    log("balanced event real smoke passed; no fitting artifact was written")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "smoke-test", "prepare", "prepare-features", "train", "evaluate", "run"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.dry_run:
        plan = make_plan()
        print({"assigned_records": len(plan["collection_records"]), "endpoint": ENDPOINT,
               "seeds": plan["seeds"], "readiness_claimed": False}, flush=True)
        return
    if args.smoke_test:
        smoke_test(args.device)
        return
    if args.prepare:
        prepare()
        return
    plan = load_plan()
    if args.prepare_features or args.run:
        prepare_features(plan, args.device)
    if args.train or args.run:
        train_models(plan, args.device)
    if args.evaluate or args.run:
        evaluate(plan, args.device)


if __name__ == "__main__":
    main()
