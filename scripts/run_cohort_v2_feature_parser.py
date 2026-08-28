"""Train and evaluate issue #16's source-bound feature predicate parser."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Final

import numpy as np

from scripts.issue_15_final_collection import DEFAULT_SEALED_ROOT
from scripts.cohort_v2_migration_recovery import DEFAULT_MANIFEST
from scripts.run_cohort_v2_confirmatory import _group_controller_decisions
from scripts.run_cohort_v2_integrated import _evaluation_devices
from scripts.run_issue_15_confirmatory_v2 import (
    CAPACITY_COMPACT,
    DEFAULT_COMPACT as ISSUE_15_COMPACT,
    DEFAULT_INTEGRATED,
    DEFAULT_PROTOCOL_ROOT,
    _config,
    _final_reader,
    _load_frozen,
)
from scripts.run_cohort_v2_macro_experiment import DEFAULT_RELEASE
from scripts.run_cohort_v2_trajectory_labels import issue_8_cost_spec
from world_model.data import CohortV2OracleWindowDataset
from world_model.model import Abstraction, PredictionPair
from world_model.training.cohort_v2_confirmatory import CohortV2ConfirmatoryError
from world_model.training.cohort_v2_controller import (
    build_cohort_v2_controller_examples,
    evaluate_cohort_v2_controllers,
)
from world_model.training.cohort_v2_evaluation import (
    CohortV2ParallelExhaustiveEvaluator,
    load_cohort_v2_evaluation,
    write_cohort_v2_evaluation,
)
from world_model.training.cohort_v2_feature_parser import (
    CohortV2FeatureParserConfig,
    CohortV2FeatureParserError,
    LearnedFeatureTransitionRequestBuilder,
    build_feature_parser_model,
    build_feature_parser_role_data,
    calibrate_feature_parser_probabilities,
    calibrate_feature_parser_thresholds,
    feature_parser_metrics,
    load_feature_parser_checkpoint,
    parse_frame_symbols,
    parse_reader_frames,
    parser_coherence,
    save_feature_parser_checkpoint,
    select_feature_parser,
    train_feature_parser,
)
from world_model.training.cohort_v2_integrated import (
    IntegratedVariant,
    integrated_compute_calibration,
    load_cohort_v2_integrated_checkpoint,
)
from world_model.training.cohort_v2_macro import (
    MACRO_CAPABILITIES,
    CohortV2MacroPairScorer,
)
from world_model.training.cohort_v2_measurement import (
    CohortV2ExecutionProfile,
    measure_cohort_v2_evaluation,
    validate_cohort_v2_measurements,
    write_cohort_v2_measurements,
)
from world_model.training.cohort_v2_trajectory_labels import (
    generate_cohort_v2_trajectory_labels,
    validate_cohort_v2_trajectory_labels,
    write_cohort_v2_trajectory_labels,
)
from world_model.training.grid_artifacts import canonical_json_bytes
from world_model.training.manifest import git_revision


DEFAULT_OUTPUT: Final = Path(".local-artifacts/issue-16-feature-parser")
DEFAULT_COMPACT: Final = Path(
    "data/runtime_evidence/issue-16/cohort-v2-feature-parser-stress-summary.json"
)
DEFAULT_ORACLE_OUTPUT: Final = Path(".local-artifacts/issue-15-confirmatory-v2")
DEFAULT_RELIABILITY: Final = Path(".local-artifacts/issue-12-reliability")
EVIDENCE_SCHEMA: Final = "cohort_v2_feature_parser_stress_evidence_v1"
REPORT_SCHEMA: Final = "cohort_v2_feature_parser_stress_report_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--sealed-root", type=Path, default=DEFAULT_SEALED_ROOT)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--integrated-root", type=Path, default=DEFAULT_INTEGRATED)
    parser.add_argument("--reliability-root", type=Path, default=DEFAULT_RELIABILITY)
    parser.add_argument("--oracle-output", type=Path, default=DEFAULT_ORACLE_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compact-report", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--integrated-compact", type=Path, default=CAPACITY_COMPACT)
    parser.add_argument("--oracle-compact", type=Path, default=ISSUE_15_COMPACT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluation-devices", default="auto")
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--score-log-every", type=int, default=250)
    parser.add_argument("--implementation-commit")
    parser.add_argument(
        "--migration-recovery",
        type=Path,
        nargs="?",
        const=DEFAULT_MANIFEST,
        metavar="MANIFEST",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser


def _paths(args: argparse.Namespace, root: Path) -> dict[str, Path]:
    return {
        "release": (root / args.release_root).resolve(),
        "sealed": (root / args.sealed_root).resolve(),
        "protocol": (root / args.protocol_root).resolve(),
        "integrated": (root / args.integrated_root).resolve(),
        "reliability": (root / args.reliability_root).resolve(),
        "oracle": (root / args.oracle_output).resolve(),
        "oracle_compact": (root / args.oracle_compact).resolve(),
        "integrated_compact": (root / args.integrated_compact).resolve(),
        "migration_recovery": (
            None
            if args.migration_recovery is None
            else (root / args.migration_recovery).resolve()
        ),
        "output": (root / args.output).resolve(),
        "compact": (root / args.compact_report).resolve(),
    }


def _parser_configs(device: str, *, dry_run: bool = False) -> tuple[CohortV2FeatureParserConfig, ...]:
    return tuple(
        CohortV2FeatureParserConfig(
            hidden_dim=hidden,
            epochs=1 if dry_run else 15,
            device=device,
        )
        for hidden in ((16,) if dry_run else (32, 64))
    )


def _role_data(readers, *, frame_limit: int | None = None):
    roles = ("training", "calibration", "model_selection")
    values = []
    for index, (reader, role) in enumerate(zip(readers, roles, strict=True), start=1):
        print(f"[parser data {index}/3] role={role}", flush=True)
        data = build_feature_parser_role_data(
            reader, expected_role=role, frame_limit=frame_limit
        )
        print(
            f"[parser data {index}/3] frames="
            f"{sum(data.available_frame_counts.values())} "
            f"relation_queries={sum(len(data.labels[p]) for p in ('contact', 'supports'))}",
            flush=True,
        )
        values.append(data)
    return tuple(values)


def _train_and_freeze(
    frozen,
    paths,
    implementation_commit: str,
    *,
    device: str,
):
    training, calibration, model_selection = _role_data(frozen["readers"])
    candidates = []
    for index, config in enumerate(_parser_configs(device), start=1):
        print(
            f"[parser candidate {index}/2] hidden={config.hidden_dim} "
            f"epochs={config.epochs} seed={config.seed}",
            flush=True,
        )
        model = build_feature_parser_model(config, training)
        train_feature_parser(model, training, progress=lambda value: print(value, flush=True))
        candidates.append(model)
    selected, selection = select_feature_parser(candidates, model_selection)
    print(
        f"[parser selection] hidden={selected.config.hidden_dim} "
        f"mean_nll={next(row['mean_predicate_negative_log_likelihood'] for row in selection if row['selected']):.6f}",
        flush=True,
    )
    temperatures = calibrate_feature_parser_probabilities(selected, calibration)
    thresholds = calibrate_feature_parser_thresholds(
        selected, calibration, temperatures
    )
    calibration_metrics = feature_parser_metrics(
        selected, calibration, thresholds, temperatures
    )
    print(
        f"[parser calibration] temperatures={temperatures} thresholds={thresholds}",
        flush=True,
    )
    checkpoint = save_feature_parser_checkpoint(
        paths["output"] / "parser",
        selected,
        temperatures,
        thresholds,
        role_data=(training, calibration, model_selection),
        readers=frozen["readers"],
        model_selection=selection,
        calibration_metrics=calibration_metrics,
        implementation_revision=implementation_commit,
    )
    print(f"[parser frozen] checkpoint={checkpoint.identity}", flush=True)
    return selected, checkpoint, selection, calibration_metrics


def _evaluate_learned_symbols(paths, frozen, args, final_reader, builder):
    devices = _evaluation_devices(args)
    total = sum(len(item.frame_records) - 1 for item in final_reader.rollouts) * 9
    scorers = []
    for device in devices:
        predictor, codec, checkpoint = load_cohort_v2_integrated_checkpoint(
            paths["integrated"]
            / "models/integrated_ordered_flat_reliability_gated/checkpoint.pt",
            reader=frozen["readers"][0],
            config=_config(device),
            variant=IntegratedVariant.CANDIDATE,
            reliability_artifact_identity=str(frozen["reliability_artifact_identity"]),
            device=device,
        )
        scorers.append(CohortV2MacroPairScorer(
            predictor,
            codec,
            checkpoint,
            _config(device),
            (final_reader,),
            transition_request_builder=builder,
            transition_request_identity=builder.identity,
            progress_every=args.score_log_every,
            progress_total=total,
            worker_name=f"learned-symbol:{device}",
        ))
    print(
        f"[evaluate] learned-symbol exhaustive 3x3 grid devices={devices} "
        f"batch={args.evaluation_batch_size}",
        flush=True,
    )
    return (
        CohortV2ParallelExhaustiveEvaluator(
            tuple(scorers), batch_size=args.evaluation_batch_size
        ).evaluate((final_reader,)),
        scorers[0].objective_identity,
    )


def _oracle_evaluation(paths, final_reader):
    manifest = json.loads((paths["oracle"] / "pair_evaluation/manifest.json").read_bytes())
    return load_cohort_v2_evaluation(
        paths["oracle"] / "pair_evaluation",
        readers=(final_reader,),
        checkpoint_identity=manifest["checkpoint_identity"],
        checkpoint_capabilities=frozenset(manifest["checkpoint_capabilities"]),
        objective_identity=manifest["objective_identity"],
    )


def _controller_rows_and_gaps(
    frozen,
    final_reader,
    learned_evaluation,
    learned_measurement,
    learned_labels,
    oracle_evaluation,
    oracle_measurement,
    oracle_labels,
):
    models = (frozen["candidate_model"], frozen["two_head_model"])
    learned_examples = build_cohort_v2_controller_examples(
        (final_reader,),
        learned_labels,
        frozen["controller_config"],
        included_roles=("final_evaluation",),
    )
    oracle_examples = build_cohort_v2_controller_examples(
        (final_reader,),
        oracle_labels,
        frozen["controller_config"],
        included_roles=("final_evaluation",),
    )
    spec = issue_8_cost_spec(
        integrated_compute_calibration(frozen["config"], IntegratedVariant.CANDIDATE)
    )
    learned = evaluate_cohort_v2_controllers(
        models,
        learned_examples,
        learned_evaluation,
        learned_measurement,
        spec,
        evaluation_roles=("final_evaluation",),
    )
    oracle = evaluate_cohort_v2_controllers(
        models,
        oracle_examples,
        oracle_evaluation,
        oracle_measurement,
        spec,
        evaluation_roles=("final_evaluation",),
    )
    learned_decisions = {
        item.state_id: item
        for item in learned.decisions
        if item.controller_id == "joint_pair"
    }
    oracle_decisions = {
        item.state_id: item
        for item in oracle.decisions
        if item.controller_id == "joint_pair"
    }
    if set(learned_decisions) != set(oracle_decisions):
        raise CohortV2FeatureParserError("learned and oracle controller states differ")
    pairs = [
        (learned_decisions[state_id].selected_pair, oracle_decisions[state_id].selected_pair)
        for state_id in sorted(learned_decisions)
    ]
    learned_teacher = {item.state_id: item.selected_pair for item in learned_labels.labels}
    oracle_teacher = {item.state_id: item.selected_pair for item in oracle_labels.labels}
    teacher_pairs = [
        (learned_teacher[state_id], oracle_teacher[state_id])
        for state_id in sorted(learned_teacher)
    ]
    controller_gaps = {
        "state_count": len(pairs),
        "pair_disagreement_rate": mean(first != second for first, second in pairs),
        "horizon_disagreement_rate": mean(first.delta != second.delta for first, second in pairs),
        "description_mode_disagreement_rate": mean(
            first.abstraction != second.abstraction for first, second in pairs
        ),
        "trajectory_teacher_pair_disagreement_rate": mean(
            first != second for first, second in teacher_pairs
        ),
        "trajectory_teacher_horizon_disagreement_rate": mean(
            first.delta != second.delta for first, second in teacher_pairs
        ),
        "trajectory_teacher_description_mode_disagreement_rate": mean(
            first.abstraction != second.abstraction for first, second in teacher_pairs
        ),
    }
    controller_compute = integrated_compute_calibration(
        frozen["config"],
        IntegratedVariant.CANDIDATE,
        controller_config=frozen["controller_config"],
    ).controller_per_decision
    aggregation_checkpoint = frozen["aggregation_manifest"]["artifacts"]["checkpoint"]["identity"]
    rows = _group_controller_decisions(
        learned.decisions,
        controller_id="joint_pair",
        configuration_id="integrated_aggregated_joint_controller",
        checkpoint_identity=aggregation_checkpoint,
        seed=10,
        evaluation=learned_evaluation,
        final_reader=final_reader,
        controller_compute=controller_compute,
    )
    oracle_rows = _group_controller_decisions(
        oracle.decisions,
        controller_id="joint_pair",
        configuration_id="integrated_aggregated_joint_controller",
        checkpoint_identity=aggregation_checkpoint,
        seed=10,
        evaluation=oracle_evaluation,
        final_reader=final_reader,
        controller_compute=controller_compute,
    )
    return rows, oracle_rows, controller_gaps


def _bootstrap(values: list[float], *, seed: int, replicates: int = 10000) -> dict[str, Any]:
    if len(values) != 6:
        raise CohortV2FeatureParserError("stress bootstrap requires six paired rollouts")
    generator = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    samples = array[generator.integers(0, len(array), size=(replicates, len(array)))].mean(axis=1)
    return {
        "paired_rollout_values": values,
        "mean": float(array.mean()),
        "two_sided_95_percent_interval": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "one_sided_97_5_percent_upper_bound": float(np.quantile(samples, 0.975)),
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
    }


def _stress_report(
    protocol,
    parser_rows,
    oracle_rows,
    *,
    parser_metrics,
    calibration_metrics,
    model_selection,
    coherence,
    controller_gaps,
    source_bindings,
):
    parser_by_attempt = {item["attempt_id"]: item for item in parser_rows}
    oracle_by_attempt = {item["attempt_id"]: item for item in oracle_rows}
    if set(parser_by_attempt) != set(oracle_by_attempt) or len(parser_by_attempt) != 6:
        raise CohortV2FeatureParserError("stress rows do not cover the same six rollouts")
    matrix = protocol["experiment_matrix"]["learned_feature_symbol_stress_issue_16"]
    budget_results = []
    for comparison in matrix["comparisons"]:
        endpoint_values = []
        violation_values = []
        rows = []
        for attempt_id in sorted(parser_by_attempt):
            parser = parser_by_attempt[attempt_id]
            oracle = oracle_by_attempt[attempt_id]
            endpoint = (
                parser["mean_endpoint_prediction_error"]
                - oracle["mean_endpoint_prediction_error"]
            )
            violation = (
                parser["mean_endpoint_violation_rate"]
                - oracle["mean_endpoint_violation_rate"]
            )
            endpoint_values.append(endpoint)
            violation_values.append(violation)
            rows.append({
                "attempt_id": attempt_id,
                "coverage_stratum": parser["coverage_stratum"],
                "parser_endpoint_error": parser["mean_endpoint_prediction_error"],
                "oracle_endpoint_error": oracle["mean_endpoint_prediction_error"],
                "endpoint_degradation": endpoint,
                "parser_violation_rate": parser["mean_endpoint_violation_rate"],
                "oracle_violation_rate": oracle["mean_endpoint_violation_rate"],
                "violation_degradation": violation,
            })
        endpoint = _bootstrap(endpoint_values, seed=comparison["endpoint_bootstrap_seed"])
        violation = _bootstrap(violation_values, seed=comparison["violation_bootstrap_seed"])
        endpoint_passed = (
            endpoint["one_sided_97_5_percent_upper_bound"]
            <= comparison["practical_effect_threshold_absolute_endpoint_error_reduction"]
        )
        violation_passed = (
            violation["one_sided_97_5_percent_upper_bound"]
            <= comparison["physical_violation_margin"]
        )
        budget_results.append({
            "budget": comparison["budget"],
            "endpoint": endpoint,
            "violation": violation,
            "practical_effect_threshold": comparison[
                "practical_effect_threshold_absolute_endpoint_error_reduction"
            ],
            "physical_violation_margin": comparison["physical_violation_margin"],
            "endpoint_rule_passed": endpoint_passed,
            "violation_rule_passed": violation_passed,
            "budget_rule_passed": endpoint_passed and violation_passed,
            "rollouts": rows,
        })
    decision = "supported" if any(item["budget_rule_passed"] for item in budget_results) else "not_supported_by_this_experiment"
    sensitivity = []
    for item in budget_results:
        for threshold_scale in (0.5, 1.5):
            for margin_scale in (0.0, 2.0):
                sensitivity.append({
                    "budget": item["budget"],
                    "threshold_scale": threshold_scale,
                    "margin_scale": margin_scale,
                    "passed": (
                        item["endpoint"]["one_sided_97_5_percent_upper_bound"]
                        <= item["practical_effect_threshold"] * threshold_scale
                        and item["violation"]["one_sided_97_5_percent_upper_bound"]
                        <= item["physical_violation_margin"] * margin_scale
                    ),
                })
    return {
        "schema": REPORT_SCHEMA,
        "protocol_identity": protocol["artifact_identity"],
        "decision": decision,
        "decision_rationale": (
            "The learned feature parser is non-materially degrading only when a frozen "
            "budget passes both upper-bound rules. This stress result cannot change or "
            "rescue issue #15."
        ),
        "budget_results": budget_results,
        "predicate_metrics": parser_metrics,
        "calibration_metrics": calibration_metrics,
        "model_selection": model_selection,
        "coherence": coherence,
        "controller_degradation": controller_gaps,
        "sensitivity": sensitivity,
        "failed_missing_or_excluded_runs": [],
        "source_bindings": source_bindings,
    }


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CohortV2FeatureParserError(f"immutable output already exists: {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)


def _write_evidence(root: Path, report: dict[str, Any], implementation_commit: str) -> dict[str, Any]:
    evidence = root / "evidence"
    _write_json(evidence / "report.json", report)
    artifacts = {
        "report": {
            "path": "report.json",
            "sha256": _sha256(evidence / "report.json"),
        },
        "parser_manifest": {
            "path": "../parser/manifest.json",
            "sha256": _sha256(root / "parser/manifest.json"),
        },
        "pair_evaluation_manifest": {
            "path": "../pair_evaluation/manifest.json",
            "sha256": _sha256(root / "pair_evaluation/manifest.json"),
        },
        "pair_measurement_manifest": {
            "path": "../pair_measurements/manifest.json",
            "sha256": _sha256(root / "pair_measurements/manifest.json"),
        },
        "trajectory_label_manifest": {
            "path": "../trajectory_labels/manifest.json",
            "sha256": _sha256(root / "trajectory_labels/manifest.json"),
        },
    }
    identity_payload = {
        "schema": EVIDENCE_SCHEMA,
        "implementation_revision": implementation_commit,
        "protocol_identity": report["protocol_identity"],
        "artifacts": artifacts,
    }
    manifest = {
        **identity_payload,
        "artifact_identity": (
            "cohort-v2-feature-parser-stress-v1:sha256:"
            + hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()
        ),
    }
    _write_json(evidence / "manifest.json", manifest)
    return manifest


def _compact(manifest, report, implementation_commit: str) -> dict[str, Any]:
    return {
        "schema": "cohort_v2_feature_parser_stress_summary_v1",
        "artifact_identity": manifest["artifact_identity"],
        "implementation_commit": implementation_commit,
        "protocol_identity": report["protocol_identity"],
        "decision": report["decision"],
        "decision_rationale": report["decision_rationale"],
        "budget_results": [
            {
                key: value
                for key, value in row.items()
                if key != "rollouts"
            }
            for row in report["budget_results"]
        ],
        "predicate_metrics": report["predicate_metrics"],
        "coherence": report["coherence"],
        "controller_degradation": report["controller_degradation"],
        "failed_missing_or_excluded_runs": report["failed_missing_or_excluded_runs"],
        "claim_boundary": (
            "This is a feature-parser stress result for the frozen issue-15 system; "
            "it cannot change or rescue issue #15 and makes no visual-perception claim."
        ),
        "source_bindings": {
            "feature_parser_checkpoint_identity": report["source_bindings"][
                "feature_parser_checkpoint_identity"
            ],
            "oracle_evidence_artifact_identity": report["source_bindings"][
                "oracle_evidence_artifact_identity"
            ],
            "sealed_bundle_identity": report["source_bindings"]["sealed_bundle_identity"],
        },
        "rerun_commands": [
            "python -u -m scripts.run_cohort_v2_feature_parser --dry-run",
            "python -u -m scripts.run_cohort_v2_feature_parser "
            f"--implementation-commit {implementation_commit}",
            "python -u -m scripts.run_cohort_v2_feature_parser --validate",
        ],
    }


def _production(paths, frozen, args, implementation_commit: str) -> int:
    if paths["output"].exists() or paths["compact"].exists():
        raise CohortV2FeatureParserError("immutable issue-16 output already exists")
    model, checkpoint, selection, calibration_metrics = _train_and_freeze(
        frozen,
        paths,
        implementation_commit,
        device=args.device,
    )
    print("[access 1/2] parser frozen before replacement final bundle read", flush=True)
    final_reader = _final_reader(paths)
    print(
        f"[access 2/2] final rollouts={len(final_reader.rollouts)} "
        f"frames={sum(len(item.frame_records) for item in final_reader.rollouts)}",
        flush=True,
    )
    final_data = build_feature_parser_role_data(
        final_reader, expected_role="final_evaluation"
    )
    final_metrics = feature_parser_metrics(
        model, final_data, checkpoint.thresholds, checkpoint.temperatures
    )
    parsed = parse_reader_frames(
        model,
        final_reader,
        checkpoint.temperatures,
        checkpoint.thresholds,
        progress=lambda value: print(value, flush=True),
    )
    coherence = parser_coherence(final_reader, parsed)
    builder = LearnedFeatureTransitionRequestBuilder(
        parsed, checkpoint.identity, final_reader.partition_identity
    )
    learned_evaluation, learned_objective_identity = _evaluate_learned_symbols(
        paths, frozen, args, final_reader, builder
    )
    write_cohort_v2_evaluation(
        paths["output"] / "pair_evaluation",
        learned_evaluation,
        readers=(final_reader,),
    )
    compute = integrated_compute_calibration(
        frozen["config"], IntegratedVariant.CANDIDATE
    )
    profile = CohortV2ExecutionProfile(False, True)
    write_cohort_v2_measurements(
        paths["output"] / "pair_measurements",
        learned_evaluation,
        readers=(final_reader,),
        calibration=compute,
        profile=profile,
    )
    learned_measurement = measure_cohort_v2_evaluation(
        learned_evaluation, (final_reader,), compute, profile
    )
    spec = issue_8_cost_spec(compute)
    write_cohort_v2_trajectory_labels(
        paths["output"] / "trajectory_labels",
        learned_evaluation,
        learned_measurement,
        spec,
        implementation_revision=implementation_commit,
    )
    learned_labels = generate_cohort_v2_trajectory_labels(
        learned_evaluation, learned_measurement, spec
    )
    print("[compare] loading source-validated oracle issue-15 evaluation", flush=True)
    oracle_evaluation = _oracle_evaluation(paths, final_reader)
    oracle_measurement = measure_cohort_v2_evaluation(
        oracle_evaluation, (final_reader,), compute, profile
    )
    oracle_labels = generate_cohort_v2_trajectory_labels(
        oracle_evaluation, oracle_measurement, spec
    )
    parser_rows, oracle_rows, controller_gaps = _controller_rows_and_gaps(
        frozen,
        final_reader,
        learned_evaluation,
        learned_measurement,
        learned_labels,
        oracle_evaluation,
        oracle_measurement,
        oracle_labels,
    )
    oracle_evidence = json.loads((paths["oracle"] / "evidence/manifest.json").read_bytes())
    oracle_compact = json.loads(paths["oracle_compact"].read_bytes())
    if oracle_compact["artifact_identity"] != oracle_evidence["artifact_identity"]:
        raise CohortV2FeatureParserError("oracle issue-15 compact binding differs")
    source_bindings = {
        "feature_parser_checkpoint_identity": checkpoint.identity,
        "learned_evaluation_identity": learned_evaluation.identity,
        "learned_objective_identity": learned_objective_identity,
        "oracle_evaluation_identity": oracle_evaluation.identity,
        "oracle_evidence_artifact_identity": oracle_evidence["artifact_identity"],
        "issue_15_capacity_calibration_identity": frozen["capacity_calibration_identity"],
        "candidate_checkpoint_identity": frozen["checkpoint"].identity,
        "controller_checkpoint_identity": frozen["aggregation_manifest"]["artifacts"][
            "checkpoint"
        ]["identity"],
        "release_identity": final_reader.release_identity,
        "partition_identity": final_reader.partition_identity,
        "sealed_bundle_identity": final_reader.sealed_bundle_identity,
        "final_access_audit": final_reader.access_audit,
    }
    report = _stress_report(
        frozen["protocol"],
        parser_rows,
        oracle_rows,
        parser_metrics=final_metrics,
        calibration_metrics=calibration_metrics,
        model_selection=selection,
        coherence=coherence,
        controller_gaps=controller_gaps,
        source_bindings=source_bindings,
    )
    manifest = _write_evidence(paths["output"], report, implementation_commit)
    compact = _compact(manifest, report, implementation_commit)
    _write_json(paths["compact"], compact)
    print(f"[decision] {report['decision']}", flush=True)
    print(f"[complete] artifact={manifest['artifact_identity']}", flush=True)
    print(f"[report] {paths['compact']}", flush=True)
    return 0


def _validate_evidence(root: Path, protocol_identity: str) -> dict[str, Any]:
    try:
        manifest_bytes = (root / "evidence/manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise CohortV2FeatureParserError(f"cannot load issue-16 evidence: {error}") from error
    if (
        canonical_json_bytes(manifest) != manifest_bytes
        or manifest.get("schema") != EVIDENCE_SCHEMA
        or manifest.get("protocol_identity") != protocol_identity
    ):
        raise CohortV2FeatureParserError("issue-16 evidence manifest is malformed")
    identity_payload = {
        key: manifest[key]
        for key in ("schema", "implementation_revision", "protocol_identity", "artifacts")
    }
    expected_identity = (
        "cohort-v2-feature-parser-stress-v1:sha256:"
        + hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()
    )
    if manifest.get("artifact_identity") != expected_identity:
        raise CohortV2FeatureParserError("issue-16 artifact identity is stale")
    for reference in manifest["artifacts"].values():
        path = (root / "evidence" / reference["path"]).resolve()
        if _sha256(path) != reference["sha256"]:
            raise CohortV2FeatureParserError("issue-16 evidence member differs")
    return manifest


def _validate(paths, frozen) -> int:
    model, checkpoint, _parser_manifest = load_feature_parser_checkpoint(
        paths["output"] / "parser",
        readers=frozen["readers"],
        device="cpu",
    )
    final_reader = _final_reader(paths)
    parsed = parse_reader_frames(
        model, final_reader, checkpoint.temperatures, checkpoint.thresholds
    )
    builder = LearnedFeatureTransitionRequestBuilder(
        parsed, checkpoint.identity, final_reader.partition_identity
    )
    scorer = CohortV2MacroPairScorer(
        frozen["predictor"],
        frozen["codec"],
        frozen["checkpoint"],
        frozen["config"],
        (final_reader,),
        transition_request_builder=builder,
        transition_request_identity=builder.identity,
    )
    evaluation = load_cohort_v2_evaluation(
        paths["output"] / "pair_evaluation",
        readers=(final_reader,),
        checkpoint_identity=frozen["checkpoint"].identity,
        checkpoint_capabilities=MACRO_CAPABILITIES,
        objective_identity=scorer.objective_identity,
    )
    compute = integrated_compute_calibration(
        frozen["config"], IntegratedVariant.CANDIDATE
    )
    profile = CohortV2ExecutionProfile(False, True)
    validate_cohort_v2_measurements(
        paths["output"] / "pair_measurements",
        evaluation,
        readers=(final_reader,),
        calibration=compute,
        profile=profile,
    )
    measurement = measure_cohort_v2_evaluation(
        evaluation, (final_reader,), compute, profile
    )
    label_manifest = json.loads((
        paths["output"] / "trajectory_labels/manifest.json"
    ).read_bytes())
    validate_cohort_v2_trajectory_labels(
        paths["output"] / "trajectory_labels",
        evaluation,
        measurement,
        issue_8_cost_spec(compute),
        implementation_revision=label_manifest["implementation_revision"],
    )
    manifest = _validate_evidence(paths["output"], frozen["protocol"]["artifact_identity"])
    report = json.loads((paths["output"] / "evidence/report.json").read_bytes())
    compact = json.loads(paths["compact"].read_bytes())
    expected = _compact(manifest, report, manifest["implementation_revision"])
    if compact != expected:
        raise CohortV2FeatureParserError("stored issue-16 compact report differs")
    print(
        f"[validate] exact issue-16 validation passed artifact={manifest['artifact_identity']}",
        flush=True,
    )
    return 0


def _dry_run(frozen) -> int:
    training, calibration, model_selection = _role_data(
        frozen["readers"], frame_limit=8
    )
    config = _parser_configs("cpu", dry_run=True)[0]
    model = build_feature_parser_model(config, training)
    train_feature_parser(model, training, progress=lambda value: print(value, flush=True))
    selected, _selection = select_feature_parser((model,), model_selection)
    temperatures = calibrate_feature_parser_probabilities(selected, calibration)
    thresholds = calibrate_feature_parser_thresholds(
        selected, calibration, temperatures
    )
    reader = frozen["readers"][1]
    rollout = reader.rollouts[0]
    parsed = {
        frame.identity: parse_frame_symbols(
            selected, frame, temperatures, thresholds
        )
        for frame in rollout.frame_records[:8]
    }
    builder = LearnedFeatureTransitionRequestBuilder(
        parsed, "dry-run-checkpoint", reader.partition_identity
    )
    windows = tuple(
        window
        for window in CohortV2OracleWindowDataset(
            reader, requested_horizons=(1,)
        )
        if window.context.identity in parsed
    )
    for abstraction in Abstraction:
        request = builder(PredictionPair(1, abstraction), (windows[0],))
        print(
            f"[dry-run request] h1/{abstraction} input={type(request.mode_input).__name__}",
            flush=True,
        )
    scorer = CohortV2MacroPairScorer(
        frozen["predictor"],
        frozen["codec"],
        frozen["checkpoint"],
        frozen["config"],
        frozen["readers"],
        transition_request_builder=builder,
        transition_request_identity=builder.identity,
    )
    value = scorer.objective(windows[0], PredictionPair(1, Abstraction.MICRO))
    print(f"[dry-run score] learned h1/micro objective={value:.8f}", flush=True)
    print("[dry-run access] final bundle unopened", flush=True)
    print("[dry-run] no files written", flush=True)
    print(
        "[actual command] python -u -m scripts.run_cohort_v2_feature_parser "
        "--implementation-commit <IMPLEMENTATION_COMMIT>",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.dry_run and args.validate:
        parser.error("--dry-run and --validate are mutually exclusive")
    if args.score_log_every <= 0 or args.evaluation_batch_size <= 0:
        parser.error("progress interval and evaluation batch size must be positive")
    root = args.repository_root.resolve()
    paths = _paths(args, root)
    implementation = args.implementation_commit
    if implementation is None:
        implementation, dirty = git_revision(str(root))
        if dirty and not args.dry_run and not args.validate:
            parser.error("a dirty worktree requires --implementation-commit")
    print(
        "[design] issue-16 four-predicate feature parser; training/model-selection/"
        "calibration roles remain separate",
        flush=True,
    )
    frozen = _load_frozen(
        root,
        paths,
        "cpu" if args.dry_run or args.validate else args.device,
    )
    print(
        "[design] frozen capacity-15 predictor/controller; final rollouts=6; "
        "parser seed=20260828",
        flush=True,
    )
    if args.dry_run:
        return _dry_run(frozen)
    if args.validate:
        return _validate(paths, frozen)
    return _production(paths, frozen, args, str(implementation))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CohortV2FeatureParserError, CohortV2ConfirmatoryError) as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(2) from error
