"""Run issue #15's capacity-correct confirmatory experiment on seed 4505."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

from scripts.issue_15_amended_protocol import load_frozen_bundle
from scripts.cohort_v2_migration_recovery import DEFAULT_MANIFEST
from scripts.issue_15_final_collection import (
    DEFAULT_SEALED_ROOT,
    Issue15ConfirmatoryV2Reader,
)
from scripts.run_cohort_v2_confirmatory import (
    _atomic_write,
    _compact_summary,
    _group_controller_decisions,
    _group_fixed_pair,
    _records,
)
from scripts.run_cohort_v2_integrated import (
    DEFAULT_RELIABILITY,
    _evaluation_devices,
    _readers,
)
from scripts.run_cohort_v2_macro_experiment import DEFAULT_RELEASE
from scripts.run_cohort_v2_trajectory_labels import issue_8_cost_spec
from world_model.data import CohortV2OracleWindowDataset
from world_model.model import Abstraction, PredictionPair
from world_model.training.cohort_v2_aggregation import (
    load_cohort_v2_aggregated_controllers,
    validate_cohort_v2_controller_aggregation,
)
from world_model.training.cohort_v2_confirmatory import (
    CANDIDATE_ID,
    CohortV2ConfirmatoryError,
    analyze_cohort_v2_confirmatory,
    audit_final_entity_capacity,
    validate_cohort_v2_confirmatory_evidence,
    write_cohort_v2_confirmatory_evidence,
)
from world_model.training.cohort_v2_controller import (
    build_cohort_v2_controller_examples,
    evaluate_cohort_v2_controllers,
    load_cohort_v2_controller_checkpoint,
)
from world_model.training.cohort_v2_evaluation import (
    CohortV2ParallelExhaustiveEvaluator,
    load_cohort_v2_evaluation,
    validate_cohort_v2_evaluation,
    write_cohort_v2_evaluation,
)
from world_model.training.cohort_v2_integrated import (
    IntegratedVariant,
    integrated_compute_calibration,
    load_cohort_v2_integrated_checkpoint,
    recursive_continuous_rollouts,
    validate_integrated_evidence,
)
from world_model.training.cohort_v2_macro import (
    MACRO_CAPABILITIES,
    CohortV2MacroConfig,
    CohortV2MacroPairScorer,
)
from world_model.training.cohort_v2_measurement import (
    CohortV2ExecutionProfile,
    measure_cohort_v2_evaluation,
    validate_cohort_v2_measurements,
    write_cohort_v2_measurements,
)
from world_model.training.cohort_v2_reliability import (
    CohortV2ReliabilityConfig,
    load_cohort_v2_reliability_estimator,
)
from world_model.training.cohort_v2_trajectory_labels import (
    generate_cohort_v2_trajectory_labels,
    validate_cohort_v2_trajectory_labels,
    write_cohort_v2_trajectory_labels,
)
from world_model.training.grid_artifacts import canonical_json_bytes
from world_model.training.manifest import git_revision


DEFAULT_PROTOCOL_ROOT: Final = Path("data/runtime_evidence/issue-15-amendment-v2")
DEFAULT_INTEGRATED: Final = Path(".local-artifacts/issue-15-capacity-integrated")
DEFAULT_OUTPUT: Final = Path(".local-artifacts/issue-15-confirmatory-v2")
DEFAULT_COMPACT: Final = Path(
    "data/runtime_evidence/issue-15/"
    "cohort-v2-oracle-symbol-confirmatory-v2-summary.json"
)
CAPACITY_COMPACT: Final = Path(
    "data/runtime_evidence/issue-15/capacity-integrated-calibration-summary.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--sealed-root", type=Path, default=DEFAULT_SEALED_ROOT)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--integrated-root", type=Path, default=DEFAULT_INTEGRATED)
    parser.add_argument("--reliability-root", type=Path, default=DEFAULT_RELIABILITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compact-report", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--integrated-compact", type=Path, default=CAPACITY_COMPACT)
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
        "output": (root / args.output).resolve(),
        "compact": (root / args.compact_report).resolve(),
        "integrated_compact": (root / args.integrated_compact).resolve(),
        "migration_recovery": (
            None
            if args.migration_recovery is None
            else (root / args.migration_recovery).resolve()
        ),
    }


def _config(device: str = "cuda:0") -> CohortV2MacroConfig:
    return CohortV2MacroConfig(
        seed=20260824,
        steps=1800,
        batch_size=32,
        learning_rate=3e-4,
        symbolic_weight=1.0,
        device=device,
        latent_dim=197,
        max_entities=15,
    )


def _load_frozen(root: Path, paths: dict[str, Path], device: str):
    _plan, protocol = load_frozen_bundle(paths["protocol"])
    readers = _readers(
        root,
        paths["release"],
        migration_recovery_authority=paths.get("migration_recovery"),
    )
    reliability_raw = json.loads(
        (paths["reliability"] / "manifest.json").read_bytes()
    )
    reliability_config = CohortV2ReliabilityConfig(**reliability_raw["config"])
    _estimator, reliability_manifest = load_cohort_v2_reliability_estimator(
        paths["reliability"],
        readers=readers,
        config=reliability_config,
        device="cpu",
    )
    config = _config(device)
    checkpoint_path = (
        paths["integrated"]
        / "models/integrated_ordered_flat_reliability_gated/checkpoint.pt"
    )
    predictor, codec, checkpoint = load_cohort_v2_integrated_checkpoint(
        checkpoint_path,
        reader=readers[0],
        config=config,
        variant=IntegratedVariant.CANDIDATE,
        reliability_artifact_identity=str(reliability_manifest["artifact_identity"]),
        device=device,
    )
    aggregation_root = paths["integrated"] / "candidate_pipeline/aggregation"
    controller_root = paths["integrated"] / "candidate_pipeline/controllers"
    aggregation_manifest = validate_cohort_v2_controller_aggregation(
        aggregation_root
    )
    aggregated_rounds, controller_config = load_cohort_v2_aggregated_controllers(
        aggregation_root
    )
    base_models, base_config, base_checkpoint_identity = (
        load_cohort_v2_controller_checkpoint(controller_root / "checkpoint.pt")
    )
    if base_config != controller_config or len(aggregated_rounds) != 1:
        raise CohortV2ConfirmatoryError("capacity controller sources differ")
    baseline_manifest = json.loads((
        paths["integrated"]
        / "candidate_pipeline/policy_baselines/manifest.json"
    ).read_bytes())
    if baseline_manifest["selected_configurations"]["fixed_pair"] != {
        "abstraction": "continuous",
        "requested_horizon": 1,
    }:
        raise CohortV2ConfirmatoryError("fixed-pair comparator is not h1/continuous")
    evidence = validate_integrated_evidence(paths["integrated"] / "evidence")
    report = json.loads(
        (paths["integrated"] / "evidence/report.json").read_bytes()
    )
    compact = json.loads(paths["integrated_compact"].read_bytes())
    if (
        compact.get("design") != "issue-15-capacity"
        or compact.get("artifact_identity") != evidence["artifact_identity"]
        or report["source_bindings"]["aggregation_artifact_identity"]
        != aggregation_manifest["aggregation_artifact_identity"]
        or report["source_bindings"]["baseline_artifact_identity"]
        != baseline_manifest["baseline_artifact_identity"]
    ):
        raise CohortV2ConfirmatoryError("capacity calibration bindings differ")
    return {
        "protocol": protocol,
        "readers": readers,
        "config": config,
        "predictor": predictor,
        "codec": codec,
        "checkpoint": checkpoint,
        "controller_config": controller_config,
        "candidate_model": aggregated_rounds[0][0],
        "two_head_model": base_models[1],
        "two_head_checkpoint_identity": base_checkpoint_identity,
        "aggregation_manifest": aggregation_manifest,
        "baseline_manifest": baseline_manifest,
        "reliability_artifact_identity": reliability_manifest["artifact_identity"],
        "capacity_calibration_identity": evidence["artifact_identity"],
    }


def _compact(manifest, report, protocol, implementation_commit: str):
    value = _compact_summary(manifest, report, protocol, implementation_commit)
    value.update({
        "schema": "cohort_v2_oracle_symbol_confirmatory_summary_v2",
        "scientific_claim_boundary": (
            "A negative result means not supported by this fixed experiment; it is "
            "not a general unsupported or impossibility claim."
        ),
        "rerun_commands": [
            "python -u -m scripts.run_issue_15_confirmatory_v2 --dry-run",
            "python -u -m scripts.run_issue_15_confirmatory_v2 "
            f"--implementation-commit {implementation_commit}",
            "python -u -m scripts.run_issue_15_confirmatory_v2 --validate",
        ],
    })
    return value


def _final_reader(paths: dict[str, Path]) -> Issue15ConfirmatoryV2Reader:
    return Issue15ConfirmatoryV2Reader(
        paths["sealed"], plan_root=paths["protocol"]
    )


def _dry_run(frozen) -> int:
    scorer = CohortV2MacroPairScorer(
        frozen["predictor"],
        frozen["codec"],
        frozen["checkpoint"],
        frozen["config"],
        frozen["readers"],
    )
    probe = CohortV2OracleWindowDataset.ingestion_smoke(frozen["readers"][1])[0]
    value = scorer.objective(probe, PredictionPair(1, Abstraction.CONTINUOUS))
    print(
        f"[dry-run score] public calibration h1/continuous objective={value:.8f}",
        flush=True,
    )
    print("[dry-run access] replacement final bundle unopened", flush=True)
    print("[dry-run] no files written", flush=True)
    print(
        "[actual command] python -u -m scripts.run_issue_15_confirmatory_v2 "
        "--implementation-commit <IMPLEMENTATION_COMMIT>",
        flush=True,
    )
    return 0


def _evaluate_final(paths, frozen, args, final_reader):
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
            reliability_artifact_identity=str(
                frozen["reliability_artifact_identity"]
            ),
            device=device,
        )
        scorers.append(CohortV2MacroPairScorer(
            predictor,
            codec,
            checkpoint,
            _config(device),
            (final_reader,),
            progress_every=args.score_log_every,
            progress_total=total,
            worker_name=f"final:{device}",
        ))
    print(
        f"[evaluate] exhaustive final 3x3 grid devices={devices} "
        f"batch={args.evaluation_batch_size}",
        flush=True,
    )
    return CohortV2ParallelExhaustiveEvaluator(
        tuple(scorers), batch_size=args.evaluation_batch_size
    ).evaluate((final_reader,))


def _production(paths, frozen, args, implementation_commit: str) -> int:
    if paths["output"].exists() or paths["compact"].exists():
        raise CohortV2ConfirmatoryError("immutable issue-15 v2 output already exists")
    print("[access 1/2] validating sealed seed-4505 collection and audit", flush=True)
    final_reader = _final_reader(paths)
    attempts = frozen["protocol"]["replicate_and_seed_policy"]["fixed_attempt_ids"]
    if {item.attempt_id for item in final_reader.rollouts} != set(attempts):
        raise CohortV2ConfirmatoryError("final attempts differ from protocol v2")
    print(
        f"[access 2/2] final rollouts={len(final_reader.rollouts)} "
        f"frame_records={sum(len(item.frame_records) for item in final_reader.rollouts)}",
        flush=True,
    )
    capacity_audit = audit_final_entity_capacity(
        final_reader, max_entities=frozen["config"].max_entities
    )
    if any(item["passed"] is False for item in capacity_audit):
        raise CohortV2ConfirmatoryError("capacity-15 candidate failed pre-evaluation audit")
    print(
        "[capacity] passed slots=15 maximum="
        f"{max(item['maximum_entity_count'] for item in capacity_audit)}",
        flush=True,
    )
    evaluation = _evaluate_final(paths, frozen, args, final_reader)
    write_cohort_v2_evaluation(
        paths["output"] / "pair_evaluation",
        evaluation,
        readers=(final_reader,),
    )
    compute = integrated_compute_calibration(
        frozen["config"], IntegratedVariant.CANDIDATE
    )
    profile = CohortV2ExecutionProfile(False, True)
    write_cohort_v2_measurements(
        paths["output"] / "pair_measurements",
        evaluation,
        readers=(final_reader,),
        calibration=compute,
        profile=profile,
    )
    measurement = measure_cohort_v2_evaluation(
        evaluation, (final_reader,), compute, profile
    )
    print(
        f"[evaluate complete] states={len(evaluation.states)} "
        f"available={evaluation.available_count}",
        flush=True,
    )
    spec = issue_8_cost_spec(compute)
    labels = generate_cohort_v2_trajectory_labels(evaluation, measurement, spec)
    label_receipt = write_cohort_v2_trajectory_labels(
        paths["output"] / "trajectory_labels",
        evaluation,
        measurement,
        spec,
        implementation_revision=implementation_commit,
    )
    examples = build_cohort_v2_controller_examples(
        (final_reader,),
        labels,
        frozen["controller_config"],
        included_roles=("final_evaluation",),
    )
    controller_result = evaluate_cohort_v2_controllers(
        (frozen["candidate_model"], frozen["two_head_model"]),
        examples,
        evaluation,
        measurement,
        spec,
        evaluation_roles=("final_evaluation",),
    )
    controller_compute = integrated_compute_calibration(
        frozen["config"],
        IntegratedVariant.CANDIDATE,
        controller_config=frozen["controller_config"],
    ).controller_per_decision
    aggregation_checkpoint = frozen["aggregation_manifest"]["artifacts"][
        "checkpoint"
    ]["identity"]
    grouped = [
        *_group_controller_decisions(
            controller_result.decisions,
            controller_id="joint_pair",
            configuration_id=CANDIDATE_ID,
            checkpoint_identity=aggregation_checkpoint,
            seed=10,
            evaluation=evaluation,
            final_reader=final_reader,
            controller_compute=controller_compute,
        ),
        *_group_controller_decisions(
            controller_result.decisions,
            controller_id="matched_capacity_two_head",
            configuration_id="matched_capacity_two_head",
            checkpoint_identity=frozen["two_head_checkpoint_identity"],
            seed=10,
            evaluation=evaluation,
            final_reader=final_reader,
            controller_compute=controller_compute,
        ),
        *_group_fixed_pair(evaluation, measurement, final_reader),
    ]
    records = _records(
        tuple(grouped),
        frozen["protocol"],
        implementation_commit,
        final_reader,
    )
    print("[recursive] fixed h15 complete-rollout diagnostics", flush=True)
    recursive = recursive_continuous_rollouts(
        frozen["predictor"],
        frozen["codec"],
        frozen["checkpoint"].identity,
        (final_reader,),
        compute,
        requested_horizons=(15,),
    )
    sealed_manifest = json.loads(
        (paths["sealed"] / "sealed-bundle-manifest.json").read_bytes()
    )
    source_bindings = {
        "access_audit": final_reader.access_audit,
        "access_manifest_identity": final_reader.access_audit[
            "workflow_manifest_identity"
        ],
        "aggregation_artifact_identity": frozen["aggregation_manifest"][
            "aggregation_artifact_identity"
        ],
        "baseline_artifact_identity": frozen["baseline_manifest"][
            "baseline_artifact_identity"
        ],
        "candidate_checkpoint_identity": frozen["checkpoint"].identity,
        "capacity_calibration_identity": frozen["capacity_calibration_identity"],
        "collection_implementation_commit": sealed_manifest[
            "collection_implementation_commit"
        ],
        "evaluation_identity": evaluation.identity,
        "measurement_identity": measurement.identity,
        "trajectory_label_artifact_identity": (
            label_receipt.label_artifact_identity
        ),
        "sealed_bundle_identity": final_reader.sealed_bundle_identity,
    }
    report = analyze_cohort_v2_confirmatory(
        records,
        recursive,
        frozen["protocol"],
        source_bindings=source_bindings,
        capacity_audit=capacity_audit,
    )
    manifest = write_cohort_v2_confirmatory_evidence(
        paths["output"] / "evidence",
        records,
        recursive,
        report,
        implementation_revision=implementation_commit,
        capacity_audit=capacity_audit,
    )
    compact = _compact(
        manifest, report, frozen["protocol"], implementation_commit
    )
    _atomic_write(paths["compact"], compact)
    print(f"[decision] {report['decision']}", flush=True)
    print(f"[complete] artifact={manifest['artifact_identity']}", flush=True)
    print(f"[report] {paths['compact']}", flush=True)
    return 0


def _validate(paths, frozen) -> int:
    final_reader = _final_reader(paths)
    scorer = CohortV2MacroPairScorer(
        frozen["predictor"],
        frozen["codec"],
        frozen["checkpoint"],
        frozen["config"],
        (final_reader,),
    )
    evaluation_root = paths["output"] / "pair_evaluation"
    validate_cohort_v2_evaluation(
        evaluation_root,
        readers=(final_reader,),
        checkpoint_identity=frozen["checkpoint"].identity,
        checkpoint_capabilities=MACRO_CAPABILITIES,
        objective_identity=scorer.objective_identity,
    )
    evaluation = load_cohort_v2_evaluation(
        evaluation_root,
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
    manifest = validate_cohort_v2_confirmatory_evidence(
        paths["output"] / "evidence", frozen["protocol"]
    )
    report = json.loads(
        (paths["output"] / "evidence/report.json").read_bytes()
    )
    compact = json.loads(paths["compact"].read_bytes())
    expected = _compact(
        manifest,
        report,
        frozen["protocol"],
        manifest["implementation_revision"],
    )
    if compact != expected:
        raise CohortV2ConfirmatoryError("stored v2 compact report differs")
    print(
        f"[validate] exact v2 validation passed artifact={manifest['artifact_identity']}",
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
        "[design] amended protocol v2, capacity-15 candidate, seed-4505 final",
        flush=True,
    )
    frozen = _load_frozen(
        root,
        paths,
        "cpu" if args.dry_run or args.validate else args.device,
    )
    print(
        "[design] candidate=integrated_aggregated_joint_controller "
        "comparators=(matched_capacity_two_head,fixed_pair) final_rollouts=6",
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
    except CohortV2ConfirmatoryError as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(2) from error
