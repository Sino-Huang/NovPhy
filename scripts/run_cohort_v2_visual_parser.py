"""Train and evaluate issue #17's frozen-encoder visual predicate parser."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any, Final, Mapping

import numpy as np

from scripts.issue_15_final_collection import DEFAULT_SEALED_ROOT
from scripts.cohort_v2_migration_recovery import DEFAULT_MANIFEST
from scripts.run_cohort_v2_feature_parser import (
    DEFAULT_OUTPUT as FEATURE_OUTPUT,
    DEFAULT_RELIABILITY,
    _controller_rows_and_gaps,
)
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
from world_model.data import (
    CohortV2AlignedObservationReader,
    CohortV2OracleWindowDataset,
)
from world_model.model import Abstraction, PredictionPair
from world_model.training.cohort_v2_confirmatory import CohortV2ConfirmatoryError
from world_model.training.cohort_v2 import build_cohort_v2_transition_request
from world_model.training.cohort_v2_evaluation import (
    CohortV2ParallelExhaustiveEvaluator,
    load_cohort_v2_evaluation,
    write_cohort_v2_evaluation,
)
from world_model.training.cohort_v2_feature_parser import (
    LearnedFeatureTransitionRequestBuilder,
    build_feature_parser_role_data,
    feature_parser_metrics,
    load_feature_parser_checkpoint,
    parse_reader_frames,
    parser_coherence,
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
from world_model.training.cohort_v2_visual_parser import (
    CohortV2VisualParserConfig,
    CohortV2VisualParserError,
    LearnedVisualTransitionRequestBuilder,
    build_visual_parser_model,
    build_visual_parser_role_data,
    calibrate_visual_parser,
    load_visual_parser_checkpoint,
    parse_visual_reader_frames,
    save_visual_parser_checkpoint,
    select_visual_parser,
    train_visual_parser,
    visual_object_vocabulary,
    visual_parser_metrics,
)
from world_model.training.grid_artifacts import canonical_json_bytes
from world_model.training.manifest import git_revision


DEFAULT_ALIGNED: Final = Path(".local-artifacts/issue-59-aligned-observation-release")
DEFAULT_OUTPUT: Final = Path(".local-artifacts/issue-17-visual-parser")
DEFAULT_COMPACT: Final = Path(
    "data/runtime_evidence/issue-17/cohort-v2-visual-parser-stress-summary.json"
)
EVIDENCE_SCHEMA: Final = "cohort_v2_visual_parser_stress_evidence_v1"
REPORT_SCHEMA: Final = "cohort_v2_visual_parser_stress_report_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--sealed-root", type=Path, default=DEFAULT_SEALED_ROOT)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--integrated-root", type=Path, default=DEFAULT_INTEGRATED)
    parser.add_argument("--reliability-root", type=Path, default=DEFAULT_RELIABILITY)
    parser.add_argument("--feature-output", type=Path, default=FEATURE_OUTPUT)
    parser.add_argument("--aligned-root", type=Path, default=DEFAULT_ALIGNED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compact-report", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--integrated-compact", type=Path, default=CAPACITY_COMPACT)
    parser.add_argument("--oracle-compact", type=Path, default=ISSUE_15_COMPACT)
    parser.add_argument(
        "--feature-compact",
        type=Path,
        default=Path(
            "data/runtime_evidence/issue-16/"
            "cohort-v2-feature-parser-stress-summary.json"
        ),
    )
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
        "feature": (root / args.feature_output).resolve(),
        "feature_compact": (root / args.feature_compact).resolve(),
        "oracle_compact": (root / args.oracle_compact).resolve(),
        "integrated_compact": (root / args.integrated_compact).resolve(),
        "migration_recovery": (
            None
            if args.migration_recovery is None
            else (root / args.migration_recovery).resolve()
        ),
        "aligned": (root / args.aligned_root).resolve(),
        "output": (root / args.output).resolve(),
        "compact": (root / args.compact_report).resolve(),
    }


def _aligned_public(frozen: Mapping[str, Any], path: Path):
    return tuple(
        CohortV2AlignedObservationReader(path, source_reader=reader)
        for reader in frozen["readers"]
    )


def _configs(device: str, *, dry_run: bool = False):
    return tuple(
        CohortV2VisualParserConfig(
            hidden_dim=hidden,
            epochs=1 if dry_run else 20,
            device=device,
        )
        for hidden in ((32,) if dry_run else (64, 128))
    )


def _role_data(readers, config, vocabulary, *, frame_limit=None):
    values = []
    for index, (reader, role) in enumerate(zip(
        readers, ("training", "calibration", "model_selection"), strict=True
    ), start=1):
        print(f"[visual data {index}/3] role={role}", flush=True)
        value = build_visual_parser_role_data(
            reader, config, expected_role=role,
            object_vocabulary=vocabulary, frame_limit=frame_limit,
        )
        print(
            f"[visual data {index}/3] frames={len(value.frame_identities)} "
            f"objects={int(value.presence.sum())} "
            f"relation_targets={int(value.relation_mask.sum())}",
            flush=True,
        )
        values.append(value)
    return tuple(values)


def _train_and_freeze(frozen, paths, aligned, implementation, device):
    vocabulary = visual_object_vocabulary(aligned)
    base_config = _configs(device)[0]
    training, calibration, selection_data = _role_data(
        aligned, base_config, vocabulary
    )
    candidates = []
    role_data_by_hidden = {}
    for index, config in enumerate(_configs(device), start=1):
        print(
            f"[visual candidate {index}/2] hidden={config.hidden_dim} "
            f"epochs={config.epochs} seed={config.seed}", flush=True,
        )
        if config.image_height == base_config.image_height and config.image_width == base_config.image_width:
            role_data = (training, calibration, selection_data)
        else:
            role_data = _role_data(aligned, config, vocabulary)
        model = build_visual_parser_model(config, role_data[0])
        train_visual_parser(model, role_data[0], progress=lambda value: print(value, flush=True))
        candidates.append(model)
        role_data_by_hidden[config.hidden_dim] = role_data
    selected, selection = select_visual_parser(candidates, selection_data)
    selected_data = role_data_by_hidden[selected.config.hidden_dim]
    print(
        f"[visual selection] hidden={selected.config.hidden_dim} "
        f"mean_nll={next(row['mean_selection_negative_log_likelihood'] for row in selection if row['selected']):.6f}",
        flush=True,
    )
    temperatures, thresholds, kind_temperature = calibrate_visual_parser(
        selected, selected_data[1]
    )
    calibration_metrics = visual_parser_metrics(
        selected, selected_data[1], temperatures, thresholds, kind_temperature
    )
    print(
        f"[visual calibration] temperatures={temperatures} thresholds={thresholds} "
        f"kind_temperature={kind_temperature}", flush=True,
    )
    checkpoint = save_visual_parser_checkpoint(
        paths["output"] / "parser", selected, temperatures, thresholds,
        kind_temperature, role_data=selected_data, readers=aligned,
        model_selection=selection, calibration_metrics=calibration_metrics,
        implementation_revision=implementation,
    )
    print(f"[visual frozen] checkpoint={checkpoint.identity}", flush=True)
    return selected, checkpoint, selection, calibration_metrics


def _scorers(paths, frozen, args, reader, builder, name):
    devices = _evaluation_devices(args)
    total = sum(len(item.frame_records) - 1 for item in reader.rollouts) * 9
    scorers = []
    for device in devices:
        predictor, codec, checkpoint = load_cohort_v2_integrated_checkpoint(
            paths["integrated"] / "models/integrated_ordered_flat_reliability_gated/checkpoint.pt",
            reader=frozen["readers"][0], config=_config(device),
            variant=IntegratedVariant.CANDIDATE,
            reliability_artifact_identity=str(frozen["reliability_artifact_identity"]),
            device=device,
        )
        scorers.append(CohortV2MacroPairScorer(
            predictor, codec, checkpoint, _config(device), (reader,),
            transition_request_builder=builder,
            transition_request_identity=getattr(
                builder, "identity", "cohort-v2-oracle-symbol-input-v1"
            ),
            progress_every=args.score_log_every,
            progress_total=total,
            worker_name=f"{name}:{device}",
        ))
    return tuple(scorers), devices


def _evaluate(paths, frozen, args, reader, builder, name):
    actual_builder = build_cohort_v2_transition_request if builder is None else builder
    scorers, devices = _scorers(paths, frozen, args, reader, actual_builder, name)
    print(
        f"[evaluate {name}] exhaustive 3x3 grid devices={devices} "
        f"batch={args.evaluation_batch_size}", flush=True,
    )
    result = CohortV2ParallelExhaustiveEvaluator(
        scorers, batch_size=args.evaluation_batch_size
    ).evaluate((reader,))
    return result, scorers[0].objective_identity


def _write_stack(root, evaluation, reader, frozen, implementation):
    write_cohort_v2_evaluation(root / "pair_evaluation", evaluation, readers=(reader,))
    compute = integrated_compute_calibration(frozen["config"], IntegratedVariant.CANDIDATE)
    profile = CohortV2ExecutionProfile(False, True)
    write_cohort_v2_measurements(
        root / "pair_measurements", evaluation, readers=(reader,),
        calibration=compute, profile=profile,
    )
    measurement = measure_cohort_v2_evaluation(
        evaluation, (reader,), compute, profile
    )
    spec = issue_8_cost_spec(compute)
    write_cohort_v2_trajectory_labels(
        root / "trajectory_labels", evaluation, measurement, spec,
        implementation_revision=implementation,
    )
    labels = generate_cohort_v2_trajectory_labels(evaluation, measurement, spec)
    return measurement, labels


def _bootstrap(values, seed, replicates=10000):
    if len(values) != 6:
        raise CohortV2VisualParserError("visual bootstrap requires six paired rollouts")
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    samples = array[
        generator.integers(0, len(array), size=(replicates, len(array)))
    ].mean(axis=1)
    return {
        "paired_rollout_values": list(values),
        "mean": float(array.mean()),
        "two_sided_95_percent_interval": [
            float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))
        ],
        "one_sided_97_5_percent_upper_bound": float(np.quantile(samples, 0.975)),
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
    }


def _paired_effects(candidate_rows, reference_rows, comparisons):
    candidate = {item["attempt_id"]: item for item in candidate_rows}
    reference = {item["attempt_id"]: item for item in reference_rows}
    if set(candidate) != set(reference) or len(candidate) != 6:
        raise CohortV2VisualParserError("visual stress rows do not cover six rollouts")
    results = []
    for comparison in comparisons:
        endpoint_values = [
            candidate[key]["mean_endpoint_prediction_error"]
            - reference[key]["mean_endpoint_prediction_error"]
            for key in sorted(candidate)
        ]
        violation_values = [
            candidate[key]["mean_endpoint_violation_rate"]
            - reference[key]["mean_endpoint_violation_rate"]
            for key in sorted(candidate)
        ]
        endpoint = _bootstrap(endpoint_values, comparison["endpoint_bootstrap_seed"])
        violation = _bootstrap(violation_values, comparison["violation_bootstrap_seed"])
        endpoint_passed = endpoint["one_sided_97_5_percent_upper_bound"] <= comparison[
            "practical_effect_threshold_absolute_endpoint_error_reduction"
        ]
        violation_passed = violation["one_sided_97_5_percent_upper_bound"] <= comparison[
            "physical_violation_margin"
        ]
        results.append({
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
            "sensitivity": [
                {
                    "threshold_scale": threshold_scale,
                    "margin_scale": margin_scale,
                    "passed": (
                        endpoint["one_sided_97_5_percent_upper_bound"]
                        <= comparison[
                            "practical_effect_threshold_absolute_endpoint_error_reduction"
                        ] * threshold_scale
                        and violation["one_sided_97_5_percent_upper_bound"]
                        <= comparison["physical_violation_margin"] * margin_scale
                    ),
                }
                for threshold_scale in (0.5, 1.5)
                for margin_scale in (0.0, 2.0)
            ],
        })
    return results


def _metric_deltas(visual, feature):
    keys = (
        "agreement", "precision", "recall", "f1", "brier_score",
        "negative_log_likelihood", "expected_calibration_error_10_bin",
    )
    return {
        predicate: {
            key: visual[predicate][key] - feature[predicate][key]
            for key in keys
        }
        for predicate in ("contact", "supports", "steady-state", "structure-unstable")
    }


def _write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CohortV2VisualParserError(f"immutable output exists: {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)


def _sha256(path: Path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_evidence(root, report, implementation):
    evidence = root / "evidence"
    _write_json(evidence / "report.json", report)
    artifacts = {}
    for name, path in (
        ("report", evidence / "report.json"),
        ("parser_manifest", root / "parser/manifest.json"),
        ("visual_evaluation", root / "visual/pair_evaluation/manifest.json"),
        ("oracle_evaluation", root / "oracle/pair_evaluation/manifest.json"),
        ("feature_evaluation", root / "feature/pair_evaluation/manifest.json"),
    ):
        artifacts[name] = {
            "path": Path(os.path.relpath(path, evidence)).as_posix(),
            "sha256": _sha256(path),
        }
    payload = {
        "schema": EVIDENCE_SCHEMA,
        "implementation_revision": implementation,
        "protocol_identity": report["protocol_identity"],
        "artifacts": artifacts,
    }
    manifest = {
        **payload,
        "artifact_identity": "cohort-v2-visual-parser-stress-v1:sha256:"
        + hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
    }
    _write_json(evidence / "manifest.json", manifest)
    return manifest


def _compact(manifest, report, implementation):
    return {
        "schema": "cohort_v2_visual_parser_stress_summary_v1",
        "artifact_identity": manifest["artifact_identity"],
        "implementation_commit": implementation,
        "protocol_identity": report["protocol_identity"],
        "decision": report["decision"],
        "decision_rationale": report["decision_rationale"],
        "visual_vs_oracle": report["visual_vs_oracle"],
        "visual_vs_feature": report["visual_vs_feature"],
        "predicate_metrics": report["predicate_metrics"],
        "object_metrics": report["object_metrics"],
        "coherence": report["coherence"],
        "controller_degradation": report["controller_degradation"],
        "source_bindings": report["source_bindings"],
        "failed_missing_or_excluded_runs": report["failed_missing_or_excluded_runs"],
        "claim_boundary": (
            "Instance-held-out visual predicate stress only; no perception novelty, "
            "template holdout, unseen physics, planning, or gameplay claim."
        ),
        "rerun_commands": [
            "python -u -m scripts.run_cohort_v2_visual_parser --dry-run",
            "python -u -m scripts.run_cohort_v2_visual_parser "
            f"--implementation-commit {implementation}",
            "python -u -m scripts.run_cohort_v2_visual_parser --validate",
        ],
    }


def _production(paths, frozen, args, implementation):
    if paths["output"].exists() or paths["compact"].exists():
        raise CohortV2VisualParserError("immutable issue-17 output already exists")
    aligned_public = _aligned_public(frozen, paths["aligned"])
    model, checkpoint, selection, calibration_metrics = _train_and_freeze(
        frozen, paths, aligned_public, implementation, args.device
    )
    print("[access 1/2] visual parser frozen before sealed aligned partition read", flush=True)
    source_final = _final_reader(paths)
    final_reader = CohortV2AlignedObservationReader(
        paths["aligned"], source_reader=source_final
    )
    print(
        f"[access 2/2] final rollouts={len(final_reader.rollouts)} "
        f"frames={sum(len(item.frame_records) for item in final_reader.rollouts)}",
        flush=True,
    )
    final_data = build_visual_parser_role_data(
        final_reader, model.config, expected_role="final_evaluation",
        object_vocabulary=model.object_vocabulary,
    )
    visual_metrics = visual_parser_metrics(
        model, final_data, checkpoint.temperatures, checkpoint.thresholds,
        checkpoint.object_kind_temperature,
    )
    visual_parsed = parse_visual_reader_frames(
        model, final_reader, checkpoint.temperatures, checkpoint.thresholds,
        progress=lambda value: print(value, flush=True),
    )
    visual_coherence = parser_coherence(final_reader, visual_parsed)
    visual_builder = LearnedVisualTransitionRequestBuilder(
        visual_parsed, checkpoint.identity, final_reader.release_identity
    )

    feature_model, feature_checkpoint, _ = load_feature_parser_checkpoint(
        paths["feature"] / "parser", readers=frozen["readers"], device=args.device
    )
    feature_data = build_feature_parser_role_data(
        final_reader, expected_role="final_evaluation"
    )
    feature_metrics = feature_parser_metrics(
        feature_model, feature_data, feature_checkpoint.thresholds,
        feature_checkpoint.temperatures,
    )
    feature_parsed = parse_reader_frames(
        feature_model, final_reader, feature_checkpoint.temperatures,
        feature_checkpoint.thresholds,
        progress=lambda value: print(value.replace("[parser", "[feature infer"), flush=True),
    )
    feature_builder = LearnedFeatureTransitionRequestBuilder(
        feature_parsed, feature_checkpoint.identity, final_reader.release_identity
    )

    visual_evaluation, visual_objective = _evaluate(
        paths, frozen, args, final_reader, visual_builder, "visual"
    )
    oracle_evaluation, oracle_objective = _evaluate(
        paths, frozen, args, final_reader, None, "oracle"
    )
    feature_evaluation, feature_objective = _evaluate(
        paths, frozen, args, final_reader, feature_builder, "feature"
    )
    visual_measurement, visual_labels = _write_stack(
        paths["output"] / "visual", visual_evaluation, final_reader, frozen, implementation
    )
    oracle_measurement, oracle_labels = _write_stack(
        paths["output"] / "oracle", oracle_evaluation, final_reader, frozen, implementation
    )
    feature_measurement, feature_labels = _write_stack(
        paths["output"] / "feature", feature_evaluation, final_reader, frozen, implementation
    )
    visual_rows, oracle_rows, visual_oracle_controller = _controller_rows_and_gaps(
        frozen, final_reader, visual_evaluation, visual_measurement, visual_labels,
        oracle_evaluation, oracle_measurement, oracle_labels,
    )
    visual_rows_again, feature_rows, visual_feature_controller = _controller_rows_and_gaps(
        frozen, final_reader, visual_evaluation, visual_measurement, visual_labels,
        feature_evaluation, feature_measurement, feature_labels,
    )
    if visual_rows_again != visual_rows:
        raise CohortV2VisualParserError("visual controller rows changed across references")
    matrix = frozen["protocol"]["experiment_matrix"]["frozen_visual_symbol_stress_issue_17"]
    visual_vs_oracle = _paired_effects(
        visual_rows, oracle_rows, matrix["comparisons"]
    )
    visual_vs_feature = _paired_effects(
        visual_rows, feature_rows, matrix["comparisons"]
    )
    decision = (
        "supported" if any(item["budget_rule_passed"] for item in visual_vs_oracle)
        else "not_supported_by_this_experiment"
    )
    issue15 = json.loads(paths["oracle_compact"].read_bytes())
    issue16 = json.loads(paths["feature_compact"].read_bytes())
    report = {
        "schema": REPORT_SCHEMA,
        "protocol_identity": frozen["protocol"]["artifact_identity"],
        "decision": decision,
        "decision_rationale": (
            "The visual parser is non-materially degrading only if a frozen budget "
            "passes both upper-bound rules relative to the oracle system. This stress "
            "result cannot change or rescue issue #15."
        ),
        "local_teacher_forced_scope": (
            "All parser-conditioned endpoint estimates are local teacher-forced scores. "
            "Fixed-h15 recursive continuous diagnostics remain the issue-15 result and "
            "do not consume symbolic parser outputs."
        ),
        "visual_vs_oracle": visual_vs_oracle,
        "visual_vs_feature": {
            "budget_results": visual_vs_feature,
            "predicate_metric_deltas_visual_minus_feature": _metric_deltas(
                visual_metrics, feature_metrics
            ),
            "controller_degradation": visual_feature_controller,
        },
        "predicate_metrics": {
            key: visual_metrics[key] for key in (
                "contact", "supports", "steady-state", "structure-unstable"
            )
        },
        "object_metrics": {
            "presence": visual_metrics["object_presence"],
            "alignment": visual_metrics["object_alignment"],
            "kind_attribute": visual_metrics["object_kind"],
        },
        "calibration_metrics": calibration_metrics,
        "model_selection": selection,
        "coherence": visual_coherence,
        "controller_degradation": visual_oracle_controller,
        "failed_missing_or_excluded_runs": [],
        "source_bindings": {
            "visual_parser_checkpoint_identity": checkpoint.identity,
            "visual_encoder_identity": model.encoder.identity,
            "visual_encoder_frozen": True,
            "aligned_release_identity": final_reader.release_identity,
            "aligned_release_access_audit": getattr(final_reader, "access_audit", None),
            "feature_parser_checkpoint_identity": feature_checkpoint.identity,
            "issue_16_artifact_identity": issue16["artifact_identity"],
            "issue_15_artifact_identity": issue15["artifact_identity"],
            "visual_evaluation_identity": visual_evaluation.identity,
            "visual_objective_identity": visual_objective,
            "oracle_evaluation_identity": oracle_evaluation.identity,
            "oracle_objective_identity": oracle_objective,
            "feature_evaluation_identity": feature_evaluation.identity,
            "feature_objective_identity": feature_objective,
            "release_identity": final_reader.release_identity,
            "partition_identity": final_reader.partition_identity,
        },
    }
    manifest = _write_evidence(paths["output"], report, implementation)
    _write_json(paths["compact"], _compact(manifest, report, implementation))
    print(f"[decision] {decision}", flush=True)
    print(f"[complete] artifact={manifest['artifact_identity']}", flush=True)
    print(f"[report] {paths['compact']}", flush=True)
    return 0


def _load_evaluation_stack(root, reader, scorer, frozen):
    evaluation = load_cohort_v2_evaluation(
        root / "pair_evaluation", readers=(reader,),
        checkpoint_identity=frozen["checkpoint"].identity,
        checkpoint_capabilities=MACRO_CAPABILITIES,
        objective_identity=scorer.objective_identity,
    )
    compute = integrated_compute_calibration(frozen["config"], IntegratedVariant.CANDIDATE)
    profile = CohortV2ExecutionProfile(False, True)
    validate_cohort_v2_measurements(
        root / "pair_measurements", evaluation, readers=(reader,),
        calibration=compute, profile=profile,
    )
    measurement = measure_cohort_v2_evaluation(evaluation, (reader,), compute, profile)
    label_manifest = json.loads((root / "trajectory_labels/manifest.json").read_bytes())
    validate_cohort_v2_trajectory_labels(
        root / "trajectory_labels", evaluation, measurement,
        issue_8_cost_spec(compute),
        implementation_revision=label_manifest["implementation_revision"],
    )
    return evaluation


def _validate(paths, frozen):
    aligned_public = _aligned_public(frozen, paths["aligned"])
    model, checkpoint, _ = load_visual_parser_checkpoint(
        paths["output"] / "parser", readers=aligned_public, device="cpu"
    )
    source_final = _final_reader(paths)
    final_reader = CohortV2AlignedObservationReader(
        paths["aligned"], source_reader=source_final
    )
    visual_parsed = parse_visual_reader_frames(
        model, final_reader, checkpoint.temperatures, checkpoint.thresholds
    )
    visual_builder = LearnedVisualTransitionRequestBuilder(
        visual_parsed, checkpoint.identity, final_reader.release_identity
    )
    feature_model, feature_checkpoint, _ = load_feature_parser_checkpoint(
        paths["feature"] / "parser", readers=frozen["readers"], device="cpu"
    )
    feature_parsed = parse_reader_frames(
        feature_model, final_reader, feature_checkpoint.temperatures,
        feature_checkpoint.thresholds,
    )
    feature_builder = LearnedFeatureTransitionRequestBuilder(
        feature_parsed, feature_checkpoint.identity, final_reader.release_identity
    )
    scorers = {}
    for name, builder in (("visual", visual_builder), ("oracle", None), ("feature", feature_builder)):
        actual = builder
        if actual is None:
            actual = build_cohort_v2_transition_request
        scorers[name] = CohortV2MacroPairScorer(
            frozen["predictor"], frozen["codec"], frozen["checkpoint"],
            frozen["config"], (final_reader,),
            transition_request_builder=actual,
            transition_request_identity=(
                "cohort-v2-oracle-symbol-input-v1" if builder is None else builder.identity
            ),
        )
        _load_evaluation_stack(
            paths["output"] / name, final_reader, scorers[name], frozen
        )
    manifest_raw = (paths["output"] / "evidence/manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if canonical_json_bytes(manifest) != manifest_raw or manifest.get("schema") != EVIDENCE_SCHEMA:
        raise CohortV2VisualParserError("issue-17 evidence manifest is malformed")
    payload = {
        key: manifest[key]
        for key in ("schema", "implementation_revision", "protocol_identity", "artifacts")
    }
    expected_identity = "cohort-v2-visual-parser-stress-v1:sha256:" + hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    if manifest["artifact_identity"] != expected_identity:
        raise CohortV2VisualParserError("issue-17 evidence identity differs")
    for reference in manifest["artifacts"].values():
        path = (paths["output"] / "evidence" / reference["path"]).resolve()
        if _sha256(path) != reference["sha256"]:
            raise CohortV2VisualParserError("issue-17 evidence member differs")
    report = json.loads((paths["output"] / "evidence/report.json").read_bytes())
    compact = json.loads(paths["compact"].read_bytes())
    if compact != _compact(manifest, report, manifest["implementation_revision"]):
        raise CohortV2VisualParserError("issue-17 compact report differs")
    print(
        f"[validate] exact issue-17 validation passed artifact={manifest['artifact_identity']}",
        flush=True,
    )
    return 0


def _dry_run(paths, frozen):
    aligned = _aligned_public(frozen, paths["aligned"])
    vocabulary = visual_object_vocabulary(aligned)
    config = _configs("cpu", dry_run=True)[0]
    training, calibration, selection_data = _role_data(
        aligned, config, vocabulary, frame_limit=8
    )
    model = build_visual_parser_model(config, training)
    train_visual_parser(model, training, progress=lambda value: print(value, flush=True))
    selected, _ = select_visual_parser((model,), selection_data)
    temperatures, thresholds, _ = calibrate_visual_parser(selected, calibration)
    reader = aligned[2]
    rollout = reader.rollouts[0]
    frames = rollout.frame_records[:8]
    parsed = {}
    for frame in frames:
        from world_model.training.cohort_v2_visual_parser import _image_tensor, parse_visual_frame_symbols
        image = _image_tensor(
            reader.load_frame_observation(rollout, frame, observation_role="agent"), config
        )
        parsed[frame.identity] = parse_visual_frame_symbols(
            selected, image, frame, temperatures, thresholds
        )
    builder = LearnedVisualTransitionRequestBuilder(
        parsed, "dry-run-visual-checkpoint", reader.release_identity
    )
    windows = tuple(
        item for item in CohortV2OracleWindowDataset(reader, requested_horizons=(1,))
        if item.context.identity in parsed
    )
    scorer = CohortV2MacroPairScorer(
        frozen["predictor"], frozen["codec"], frozen["checkpoint"],
        frozen["config"], frozen["readers"],
        transition_request_builder=builder,
        transition_request_identity=builder.identity,
    )
    value = scorer.objective(windows[0], PredictionPair(1, Abstraction.MICRO))
    print(f"[dry-run score] visual h1/micro objective={value:.8f}", flush=True)
    print("[dry-run access] sealed-final aligned partition unopened", flush=True)
    print("[dry-run] no files written", flush=True)
    print(
        "[actual command] python -u -m scripts.run_cohort_v2_visual_parser "
        "--implementation-commit <IMPLEMENTATION_COMMIT>", flush=True,
    )
    return 0


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.dry_run and args.validate:
        raise CohortV2VisualParserError("--dry-run and --validate are mutually exclusive")
    if args.score_log_every <= 0 or args.evaluation_batch_size <= 0:
        raise CohortV2VisualParserError("progress interval and batch size must be positive")
    root = args.repository_root.resolve()
    paths = _paths(args, root)
    implementation = args.implementation_commit
    if implementation is None:
        implementation, dirty = git_revision(str(root))
        if dirty and not args.dry_run and not args.validate:
            raise CohortV2VisualParserError("dirty production requires --implementation-commit")
    print(
        "[design] issue-17 frozen RGB+Sobel encoder; supervised object slots, "
        "kind attributes, relations, and macro events", flush=True,
    )
    frozen = _load_frozen(
        root, paths, "cpu" if args.dry_run or args.validate else args.device
    )
    print(
        "[design] seed=20260829; issue-15 oracle primary; issue-16 feature "
        "parser secondary; sealed aligned final remains deferred", flush=True,
    )
    if args.dry_run:
        return _dry_run(paths, frozen)
    if args.validate:
        return _validate(paths, frozen)
    return _production(paths, frozen, args, str(implementation))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CohortV2VisualParserError, CohortV2ConfirmatoryError) as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(2) from error
