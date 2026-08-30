"""Run the pre-confirmatory calibration and replicate-design study (issue #41)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Final

from scripts.run_cohort_v2_controller import DEFAULT_OUTPUT as DEFAULT_CONTROLLERS
from scripts.run_cohort_v2_controller_aggregation import (
    DEFAULT_OUTPUT as DEFAULT_AGGREGATION,
)
from scripts.run_cohort_v2_pair_measurements import (
    DEFAULT_CHECKPOINT,
    DEFAULT_EVALUATION,
    DEFAULT_RELEASE,
)
from scripts.run_cohort_v2_policy_baselines import DEFAULT_OUTPUT as DEFAULT_BASELINES
from scripts.run_cohort_v2_trajectory_labels import (
    DEFAULT_MEASUREMENTS,
    DEFAULT_OUTPUT as DEFAULT_TRAJECTORY_LABELS,
    _source_inputs,
    issue_8_cost_spec,
)
from world_model.model import Abstraction, PredictionPair
from world_model.training import (
    CohortV2CalibrationRecord,
    CohortV2ControllerDecision,
    CohortV2MicroConfig,
    CohortV2MicroPairScorer,
    CohortV2StressGapRecord,
    analyze_cohort_v2_calibration,
    build_cohort_v2_controller_examples,
    cohort_v2_pair_utility,
    evaluate_cohort_v2_controllers,
    generate_cohort_v2_trajectory_labels,
    git_revision,
    load_cohort_v2_aggregated_controllers,
    load_cohort_v2_controller_checkpoint,
    load_cohort_v2_evaluation,
    validate_cohort_v2_calibration,
    validate_cohort_v2_controller_aggregation,
    validate_cohort_v2_controllers,
    validate_cohort_v2_policy_baselines,
    validate_cohort_v2_reliability_artifact,
    validate_cohort_v2_trajectory_labels,
    write_cohort_v2_calibration,
)
from world_model.training.cohort_v2_symbolic_interfaces import (
    SymbolicInterface,
    load_symbolic_interface_checkpoint,
)
from world_model.training.cohort_v2_reliability import CohortV2ReliabilityConfig
from world_model.training.grid_artifacts import canonical_json_bytes


DEFAULT_RELIABILITY: Final = Path(".local-artifacts/issue-12-reliability")
DEFAULT_SYMBOLIC_INTERFACES: Final = Path(
    ".local-artifacts/issue-13-symbolic-interfaces"
)
DEFAULT_OUTPUT: Final = Path(".local-artifacts/issue-41-calibration")
CANDIDATE: Final = "joint_pair_aggregation_round_1_preintegration"
COMPARATORS: Final = (
    "fixed_pair_h1_continuous",
    "matched_capacity_two_head",
)
ROLE_INFLUENCES: Final = {
    "calibration": "threshold_values",
    "model_selection": "configuration_selection",
}
MISSING_INTEGRATIONS: Final = (
    "Train one full 3x3 shared predictor using the issue-13 retained ordered-flat "
    "micro interface and the issue-12 retained reliability loss gate, using only "
    "training-role learned parameters; then regenerate exhaustive pair evidence, "
    "measurements, trajectory labels, distilled/aggregated controllers, and repeat "
    "this six-rollout calibration analysis before freezing issue #34.",
    "Train source-bound, matched-capacity integrated no-symbol and ordered-flat "
    "without-reliability-gating ablations, then replace the current component-level "
    "stress proxies with their rollout-level calibration degradation records before "
    "freezing issue #34.",
    "Measure continuous-mode accumulation over equal simulated duration by recursively "
    "applying fixed h1, h5, and h15 predictions without future engine carriers or "
    "labels, scoring only against authoritative endpoints, before freezing issue #34.",
)
DOWNSTREAM_WORK: Final = (
    "Learned-feature-parser and frozen-visual-parser stress tests remain downstream of "
    "issue #15; they are not prerequisites for this pre-#34 integrated-model calibration.",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--measurements-root", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--trajectory-labels-root", type=Path, default=DEFAULT_TRAJECTORY_LABELS)
    parser.add_argument("--baselines-root", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--controllers-root", type=Path, default=DEFAULT_CONTROLLERS)
    parser.add_argument("--aggregation-root", type=Path, default=DEFAULT_AGGREGATION)
    parser.add_argument("--reliability-root", type=Path, default=DEFAULT_RELIABILITY)
    parser.add_argument(
        "--symbolic-interfaces-root", type=Path, default=DEFAULT_SYMBOLIC_INTERFACES
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap-seed", type=int, default=20260826)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--compact-report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser


def _paths(args: argparse.Namespace, repository_root: Path) -> dict[str, Path]:
    return {
        name: (repository_root / getattr(args, name)).resolve()
        for name in (
            "release_root",
            "evaluation_root",
            "checkpoint",
            "measurements_root",
            "trajectory_labels_root",
            "baselines_root",
            "controllers_root",
            "aggregation_root",
            "reliability_root",
            "symbolic_interfaces_root",
            "output",
        )
    }


def _load_sources(repository_root: Path, paths: dict[str, Path], device: str):
    print("[load 1/7] validating issue #7 pair evaluation and measurements", flush=True)
    evaluation, measurement, compute_calibration, readers = _source_inputs(
        repository_root,
        paths["release_root"],
        paths["evaluation_root"],
        paths["checkpoint"],
        paths["measurements_root"],
    )
    spec = issue_8_cost_spec(compute_calibration)
    trajectory_manifest = json.loads(
        (paths["trajectory_labels_root"] / "manifest.json").read_bytes()
    )
    print("[load 2/7] validating issue #8 trajectory labels", flush=True)
    trajectory_receipt = validate_cohort_v2_trajectory_labels(
        paths["trajectory_labels_root"],
        evaluation,
        measurement,
        spec,
        implementation_revision=trajectory_manifest["implementation_revision"],
    )
    labels = generate_cohort_v2_trajectory_labels(evaluation, measurement, spec)

    baseline_manifest = json.loads(
        (paths["baselines_root"] / "manifest.json").read_bytes()
    )
    print("[load 3/7] validating issue #9 frozen baseline set", flush=True)
    baseline_receipt = validate_cohort_v2_policy_baselines(
        paths["baselines_root"],
        evaluation,
        measurement,
        spec,
        trajectory_label_artifact_identity=trajectory_receipt.label_artifact_identity,
        derivation_index_identity=baseline_manifest["derivation_index_identity"],
        implementation_revision=baseline_manifest["implementation_revision"],
    )

    controller_manifest = json.loads(
        (paths["controllers_root"] / "manifest.json").read_bytes()
    )
    print("[load 4/7] validating issue #10 controller baseline", flush=True)
    validate_cohort_v2_controllers(
        paths["controllers_root"],
        readers,
        evaluation,
        measurement,
        labels,
        spec,
        trajectory_label_artifact_identity=trajectory_receipt.label_artifact_identity,
        baseline_artifact_identity=baseline_receipt.baseline_artifact_identity,
        derivation_index_identity=controller_manifest["derivation_index_identity"],
        implementation_revision=controller_manifest["implementation_revision"],
    )
    base_models, controller_config, controller_checkpoint_identity = (
        load_cohort_v2_controller_checkpoint(paths["controllers_root"] / "checkpoint.pt")
    )

    print("[load 5/7] validating issue #11 aggregation checkpoint", flush=True)
    aggregation_manifest = validate_cohort_v2_controller_aggregation(
        paths["aggregation_root"]
    )
    aggregation_rounds, aggregation_config = load_cohort_v2_aggregated_controllers(
        paths["aggregation_root"]
    )
    if aggregation_config != controller_config or len(aggregation_rounds) != 1:
        raise ValueError("issue #41 requires the frozen one-round aggregation candidate")

    reliability_manifest_raw = json.loads(
        (paths["reliability_root"] / "manifest.json").read_bytes()
    )
    reliability_config = CohortV2ReliabilityConfig(**reliability_manifest_raw["config"])
    print("[load 6/7] validating issue #12 reliability decision", flush=True)
    reliability_manifest = validate_cohort_v2_reliability_artifact(
        paths["reliability_root"], readers=readers, config=reliability_config
    )

    print("[load 7/7] validating issue #13 symbolic stress sources", flush=True)
    symbolic = {}
    micro_config = CohortV2MicroConfig(seed=20260824, steps=1200, device=device)
    for interface in (SymbolicInterface.NO_SYMBOL, SymbolicInterface.ORDERED_FLAT):
        root = paths["symbolic_interfaces_root"] / str(interface)
        predictor, codec, checkpoint = load_symbolic_interface_checkpoint(
            root / "checkpoint.pt",
            reader=readers[0],
            config=micro_config,
            interface=interface,
            device=device,
        )
        scorer = CohortV2MicroPairScorer(
            predictor, codec, checkpoint, micro_config, readers
        )
        symbolic[str(interface)] = (
            load_cohort_v2_evaluation(
                root / "pair_evaluation",
                readers=readers,
                checkpoint_identity=checkpoint.identity,
                checkpoint_capabilities=scorer.capabilities,
                objective_identity=scorer.objective_identity,
            ),
            checkpoint.identity,
        )
        print(f"[symbolic] validated {interface}", flush=True)
    return {
        "aggregation_manifest": aggregation_manifest,
        "aggregation_rounds": aggregation_rounds,
        "base_models": base_models,
        "baseline_manifest": baseline_manifest,
        "controller_checkpoint_identity": controller_checkpoint_identity,
        "controller_config": controller_config,
        "controller_manifest": controller_manifest,
        "evaluation": evaluation,
        "labels": labels,
        "measurement": measurement,
        "readers": readers,
        "reliability_manifest": reliability_manifest,
        "spec": spec,
        "symbolic": symbolic,
        "trajectory_receipt": trajectory_receipt,
    }


def _fixed_pair_decisions(sources) -> tuple[CohortV2ControllerDecision, ...]:
    evaluation = sources["evaluation"]
    measurement = sources["measurement"]
    spec = sources["spec"]
    pair = PredictionPair(1, Abstraction.CONTINUOUS)
    measured = {item.state_id: item for item in measurement.states}
    labels = {item.state_id: item for item in sources["labels"].labels}
    decisions = []
    for state in evaluation.states:
        if state.exposure_role not in ROLE_INFLUENCES:
            continue
        pair_index = evaluation.grid.pairs.index(pair)
        utility = cohort_v2_pair_utility(
            state,
            state.outcomes[pair_index],
            measured[state.state_id].outcomes[pair_index],
            spec,
        )
        if utility is None:
            raise ValueError(f"fixed comparator unavailable at {state.state_id}")
        decisions.append(CohortV2ControllerDecision(
            controller_id=COMPARATORS[0],
            state_id=state.state_id,
            exposure_role=state.exposure_role,
            scenario_lineage_identity=state.scenario_lineage_identity,
            selected_pair=pair,
            oracle_pair=labels[state.state_id].selected_pair,
            utility_available=True,
            prediction_objective=utility[0],
            endpoint_violation_rate=utility[1],
            policy_compute_per_simulated_frame=utility[2],
            full_compute_per_simulated_frame=utility[3],
            segment_cost=utility[4],
            oracle_segment_cost=labels[state.state_id].segment_cost,
        ))
    return tuple(decisions)


def _group_records(
    decisions: tuple[CohortV2ControllerDecision, ...],
    *,
    configuration_id: str,
    checkpoint_identity: str,
    seed: int,
    sources,
) -> tuple[CohortV2CalibrationRecord, ...]:
    states = {state.state_id: state for state in sources["evaluation"].states}
    rollouts = {
        rollout.attempt_id: rollout
        for reader in sources["readers"]
        for rollout in reader.rollouts
    }
    by_attempt: dict[str, list[CohortV2ControllerDecision]] = {}
    for decision in decisions:
        attempt_id = states[decision.state_id].attempt_id
        by_attempt.setdefault(attempt_id, []).append(decision)
    result = []
    for attempt_id in sorted(by_attempt):
        rows = by_attempt[attempt_id]
        if any(
            value is None
            for row in rows
            for value in (
                row.prediction_objective,
                row.endpoint_violation_rate,
                row.policy_compute_per_simulated_frame,
                row.full_compute_per_simulated_frame,
            )
        ):
            raise ValueError(f"{configuration_id} has unavailable utility at {attempt_id}")
        rollout = rollouts[attempt_id]
        result.append(CohortV2CalibrationRecord(
            configuration_id=configuration_id,
            exposure_role=rows[0].exposure_role,
            attempt_id=attempt_id,
            scenario_lineage_identity=rollout.scenario_lineage_identity,
            coverage_stratum=rollout.coverage_stratum,
            checkpoint_identity=checkpoint_identity,
            seed=seed,
            state_count=len(rows),
            mean_endpoint_prediction_error=mean(
                float(row.prediction_objective) for row in rows
            ),
            mean_endpoint_violation_rate=mean(
                float(row.endpoint_violation_rate) for row in rows
            ),
            mean_policy_compute_per_simulated_frame=mean(
                float(row.policy_compute_per_simulated_frame) for row in rows
            ),
            mean_full_compute_per_simulated_frame=mean(
                float(row.full_compute_per_simulated_frame) for row in rows
            ),
        ))
    return tuple(result)


def _symbolic_stress(sources) -> tuple[CohortV2StressGapRecord, ...]:
    pair = PredictionPair(1, Abstraction.MICRO)
    no_symbol = sources["symbolic"][str(SymbolicInterface.NO_SYMBOL)][0]
    ordered = sources["symbolic"][str(SymbolicInterface.ORDERED_FLAT)][0]
    if tuple(state.state_id for state in no_symbol.states) != tuple(
        state.state_id for state in ordered.states
    ):
        raise ValueError("symbolic stress evaluations use different state scopes")
    rollouts = {
        rollout.attempt_id: rollout
        for reader in sources["readers"]
        for rollout in reader.rollouts
    }
    pair_index = no_symbol.grid.pairs.index(pair)
    gaps: dict[str, list[float]] = {}
    for stressed, reference in zip(no_symbol.states, ordered.states, strict=True):
        if stressed.exposure_role != "calibration":
            continue
        left = stressed.outcomes[pair_index].objective
        right = reference.outcomes[pair_index].objective
        if left is None or right is None:
            raise ValueError("symbolic stress proxy is unavailable")
        gaps.setdefault(stressed.attempt_id, []).append(float(left) - float(right))
    return tuple(
        CohortV2StressGapRecord(
            stress_id="no_symbol_vs_ordered_flat_h1_micro",
            exposure_role="calibration",
            attempt_id=attempt_id,
            scenario_lineage_identity=rollouts[attempt_id].scenario_lineage_identity,
            coverage_stratum=rollouts[attempt_id].coverage_stratum,
            reference_configuration_id="ordered_flat_predicate",
            stressed_configuration_id="no_symbol",
            metric="teacher_forced_h1_micro_total_objective",
            degradation_gap=mean(values),
        )
        for attempt_id, values in sorted(gaps.items())
    )


def _calibration_records(sources):
    config = sources["controller_config"]
    examples = build_cohort_v2_controller_examples(
        sources["readers"],
        sources["labels"],
        config,
        included_roles=("training", "calibration", "model_selection"),
    )
    base = evaluate_cohort_v2_controllers(
        sources["base_models"],
        examples,
        sources["evaluation"],
        sources["measurement"],
        sources["spec"],
        evaluation_roles=("calibration", "model_selection"),
    )
    aggregated = evaluate_cohort_v2_controllers(
        sources["aggregation_rounds"][0],
        examples,
        sources["evaluation"],
        sources["measurement"],
        sources["spec"],
        evaluation_roles=("calibration", "model_selection"),
    )
    records = []
    records.extend(_group_records(
        _fixed_pair_decisions(sources),
        configuration_id=COMPARATORS[0],
        checkpoint_identity=sources["evaluation"].checkpoint_identity,
        seed=20260824,
        sources=sources,
    ))
    records.extend(_group_records(
        tuple(item for item in base.decisions if item.controller_id == "matched_capacity_two_head"),
        configuration_id=COMPARATORS[1],
        checkpoint_identity=sources["controller_checkpoint_identity"],
        seed=config.seed,
        sources=sources,
    ))
    records.extend(_group_records(
        tuple(item for item in aggregated.decisions if item.controller_id == "joint_pair"),
        configuration_id=CANDIDATE,
        checkpoint_identity=sources["aggregation_manifest"]["artifacts"]["checkpoint"]["identity"],
        seed=config.seed,
        sources=sources,
    ))
    oracle_records = _group_records(
        tuple(item for item in base.decisions if item.controller_id == "joint_pair"),
        configuration_id="joint_pair_oracle_state_training_ablation",
        checkpoint_identity=sources["controller_checkpoint_identity"],
        seed=config.seed,
        sources=sources,
    )
    records.extend(oracle_records)

    by_key = {(item.configuration_id, item.attempt_id): item for item in records}
    closed_loop_stress = []
    for attempt_id in sorted(
        item.attempt_id
        for item in records
        if item.configuration_id == CANDIDATE and item.exposure_role == "calibration"
    ):
        candidate = by_key[(CANDIDATE, attempt_id)]
        reference = by_key[("joint_pair_oracle_state_training_ablation", attempt_id)]
        closed_loop_stress.append(CohortV2StressGapRecord(
            stress_id="closed_loop_aggregation_vs_oracle_state_training",
            exposure_role="calibration",
            attempt_id=attempt_id,
            scenario_lineage_identity=candidate.scenario_lineage_identity,
            coverage_stratum=candidate.coverage_stratum,
            reference_configuration_id=reference.configuration_id,
            stressed_configuration_id=CANDIDATE,
            metric="mean_endpoint_prediction_error",
            degradation_gap=(
                candidate.mean_endpoint_prediction_error
                - reference.mean_endpoint_prediction_error
            ),
        ))
    return tuple(records), tuple(closed_loop_stress) + _symbolic_stress(sources)


def _source_bindings(sources, implementation_revision: str) -> dict[str, object]:
    return {
        "authoritative_derivation_index_identity": sources["readers"][0].derivation_identity,
        "baseline_artifact_identity": sources["baseline_manifest"]["baseline_artifact_identity"],
        "calibration_analysis_implementation_revision": implementation_revision,
        "capability_declaration_identity": sources["evaluation"].capability_declaration_identity,
        "controller_artifact_identity": sources["controller_manifest"]["controller_artifact_identity"],
        "controller_checkpoint_identity": sources["controller_checkpoint_identity"],
        "evaluation_identity": sources["evaluation"].identity,
        "grid_identity": sources["evaluation"].grid.identity,
        "measurement_identity": sources["measurement"].identity,
        "partition_identity": sources["evaluation"].partition_identity,
        "predictor_checkpoint_identity": sources["evaluation"].checkpoint_identity,
        "release_identity": sources["evaluation"].release_identity,
        "reliability_artifact_identity": sources["reliability_manifest"]["artifact_identity"],
        "symbolic_checkpoint_identities": {
            key: value[1] for key, value in sources["symbolic"].items()
        },
        "trajectory_label_artifact_identity": sources["trajectory_receipt"].label_artifact_identity,
    }


def _print_report(report: dict[str, object]) -> None:
    proposals = report["proposals_for_issue_34"]
    gain = report["paired_primary_endpoint_gain"]
    print(
        f"[result] comparator={proposals['strongest_comparator_id']} "
        f"replicates={report['independent_calibration_replicates']} "
        f"mean_gain={gain['summary']['mean']:.8f} "
        f"effect_threshold={proposals['practical_effect_threshold_absolute_endpoint_error_reduction']:.8f}",
        flush=True,
    )
    print(
        f"[result] violation_margin={proposals['physical_violation_margin']:.8f} "
        f"compute_budgets={proposals['compute_budgets_policy_units_per_simulated_frame']}",
        flush=True,
    )
    print(f"[disposition] {report['disposition']['status']}", flush=True)
    for item in report["disposition"]["additional_calibration_work_required"]:
        print(f"[additional-work] {item}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.bootstrap_replicates <= 0 or args.bootstrap_seed < 0:
        parser.error("bootstrap settings must be positive and deterministic")
    repository_root = args.repository_root.resolve()
    paths = _paths(args, repository_root)
    implementation_revision = args.implementation_commit
    if implementation_revision is None:
        implementation_revision, dirty = git_revision(str(repository_root))
        if dirty and not args.dry_run and not args.validate:
            parser.error("a dirty worktree requires an explicit --implementation-commit")

    sources = _load_sources(repository_root, paths, args.device)
    bindings = _source_bindings(sources, implementation_revision)
    if args.validate:
        manifest = validate_cohort_v2_calibration(paths["output"])
        expected_bindings = _source_bindings(
            sources, manifest["implementation_revision"]
        )
        if manifest["source_bindings"] != expected_bindings:
            raise ValueError("stored calibration bindings differ from validated sources")
        print(
            f"[validate] passed artifact={manifest['calibration_artifact_identity']}",
            flush=True,
        )
        return 0

    print("[design] freezing candidate and comparator set before calibration", flush=True)
    print(f"[design] candidate={CANDIDATE} comparators={COMPARATORS}", flush=True)
    records, stress = _calibration_records(sources)
    calibration_attempts = sorted({
        item.attempt_id for item in records if item.exposure_role == "calibration"
    })
    for index, attempt_id in enumerate(calibration_attempts, start=1):
        stratum = next(
            item.coverage_stratum
            for item in records
            if item.attempt_id == attempt_id
        )
        print(
            f"[calibration replicate {index}/{len(calibration_attempts)}] "
            f"attempt={attempt_id} stratum={stratum}",
            flush=True,
        )

    report = analyze_cohort_v2_calibration(
        records,
        stress,
        candidate_configuration_id=CANDIDATE,
        eligible_comparator_ids=COMPARATORS,
        source_bindings=bindings,
        missing_integrations=MISSING_INTEGRATIONS,
        downstream_work=DOWNSTREAM_WORK,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    _print_report(report)
    if args.dry_run:
        print("[dry-run] passed; no files written and final evaluation remained sealed", flush=True)
        return 0

    manifest = write_cohort_v2_calibration(
        paths["output"],
        records,
        stress,
        candidate_configuration_id=CANDIDATE,
        eligible_comparator_ids=COMPARATORS,
        source_bindings=bindings,
        missing_integrations=MISSING_INTEGRATIONS,
        implementation_revision=implementation_revision,
        downstream_work=DOWNSTREAM_WORK,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(
        f"[complete] artifact={manifest['calibration_artifact_identity']} "
        f"output={paths['output']}",
        flush=True,
    )
    if args.compact_report is not None:
        compact = {
            "artifact_type": "cohort_v2_preconfirmatory_calibration_summary",
            "calibration_artifact_identity": manifest["calibration_artifact_identity"],
            "disposition": report["disposition"],
            "exposure_audit": report["exposure_audit"],
            "implementation_commit": implementation_revision,
            "independent_calibration_replicates": report["independent_calibration_replicates"],
            "endpoint_scope": report["endpoint_scope"],
            "paired_primary_endpoint_gain": report["paired_primary_endpoint_gain"],
            "paired_physical_violation_increase": report["paired_physical_violation_increase"],
            "proposals_for_issue_34": report["proposals_for_issue_34"],
            "release_identity": sources["evaluation"].release_identity,
            "rerun_commands": [
                "python -u -m scripts.run_cohort_v2_calibration --dry-run",
                "python -u -m scripts.run_cohort_v2_calibration "
                f"--implementation-commit {implementation_revision}",
                "python -u -m scripts.run_cohort_v2_calibration --validate",
            ],
            "schema": "cohort_v2_preconfirmatory_calibration_summary_v1",
            "source_bound_validation": "passed",
            "stress_test_degradation_proxies": report["stress_test_degradation_proxies"],
            "downstream_work": report["downstream_work"],
        }
        target = (repository_root / args.compact_report).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical_json_bytes(compact))
        print(f"[report] {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
