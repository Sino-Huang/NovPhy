"""Run the frozen oracle-symbol confirmatory experiment (issue #15)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Final

from scripts.cohort_v2_partition import CohortV2PartitionExposureManifest
from scripts.cohort_v2_statistical_protocol import build_protocol, load_protocol
from scripts.final_evaluation_access import (
    FinalEvaluationWorkflowAccessManifest,
    audit_final_evaluation_workflow_access,
    authorize_final_evaluation_workflow_access,
)
from scripts.run_cohort_v2_integrated import (
    DEFAULT_RELIABILITY,
    _evaluation_devices,
    _readers,
)
from scripts.run_cohort_v2_macro_experiment import DEFAULT_RELEASE
from scripts.run_cohort_v2_trajectory_labels import issue_8_cost_spec
from world_model.data import (
    CohortV2FinalEvaluationReader,
    CohortV2OracleWindowDataset,
)
from world_model.model import Abstraction, PredictionPair
from world_model.training.cohort_v2_aggregation import (
    load_cohort_v2_aggregated_controllers,
    validate_cohort_v2_controller_aggregation,
)
from world_model.training.cohort_v2_confirmatory import (
    CANDIDATE_ID,
    CohortV2ConfirmatoryError,
    CohortV2ConfirmatoryRecord,
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
    write_cohort_v2_trajectory_labels,
)
from world_model.training.grid_artifacts import canonical_json_bytes
from world_model.training.manifest import git_revision


DEFAULT_PROTOCOL: Final = Path(
    "data/runtime_evidence/issue-34/cohort-v2-prospective-statistical-protocol-v1.json"
)
DEFAULT_SEALED: Final = Path(
    ".local-artifacts/issue-53-mixed-termination-final-release-v5"
)
DEFAULT_INTEGRATED: Final = Path(".local-artifacts/issue-58-integrated")
DEFAULT_OUTPUT: Final = Path(".local-artifacts/issue-15-confirmatory")
DEFAULT_COMPACT: Final = Path(
    "data/runtime_evidence/issue-15/cohort-v2-oracle-symbol-confirmatory-summary.json"
)
DEFAULT_PLAN: Final = Path("data/runtime_evidence/issue-53-plan-v5")
CONSUMER_ID: Final = "issue-15-oracle-symbol-confirmatory-v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--sealed-root", type=Path, default=DEFAULT_SEALED)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--integrated-root", type=Path, default=DEFAULT_INTEGRATED)
    parser.add_argument("--reliability-root", type=Path, default=DEFAULT_RELIABILITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compact-report", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluation-devices", default="auto")
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--score-log-every", type=int, default=250)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--finalize-failure", action="store_true")
    parser.add_argument("--rewrite-compact", action="store_true")
    return parser


def _paths(args: argparse.Namespace, root: Path) -> dict[str, Path]:
    return {
        "release": (root / args.release_root).resolve(),
        "sealed": (root / args.sealed_root).resolve(),
        "protocol": (root / args.protocol).resolve(),
        "plan": (root / args.plan_root).resolve(),
        "integrated": (root / args.integrated_root).resolve(),
        "reliability": (root / args.reliability_root).resolve(),
        "output": (root / args.output).resolve(),
        "compact": (root / args.compact_report).resolve(),
    }


def _config() -> CohortV2MacroConfig:
    return CohortV2MacroConfig(
        seed=20260824,
        steps=1800,
        batch_size=32,
        learning_rate=3e-4,
        symbolic_weight=1.0,
        device="cuda:0",
    )


def _load_frozen_sources(root: Path, paths: dict[str, Path], device: str):
    protocol = load_protocol(paths["protocol"])
    expected_protocol = build_protocol(
        root, implementation_commit=protocol["implementation_commit"]
    )
    if canonical_json_bytes(protocol) != canonical_json_bytes(expected_protocol):
        raise CohortV2ConfirmatoryError("issue-34 protocol differs from its frozen sources")
    readers = _readers(root, paths["release"])
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
    config = _config()
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
    aggregation_manifest = validate_cohort_v2_controller_aggregation(aggregation_root)
    aggregated_rounds, controller_config = load_cohort_v2_aggregated_controllers(
        aggregation_root
    )
    base_models, base_config, base_checkpoint_identity = (
        load_cohort_v2_controller_checkpoint(controller_root / "checkpoint.pt")
    )
    if base_config != controller_config or len(aggregated_rounds) != 1:
        raise CohortV2ConfirmatoryError("issue-58 frozen controller sources differ")
    baseline_manifest = json.loads(
        (paths["integrated"] / "candidate_pipeline/policy_baselines/manifest.json").read_bytes()
    )
    if baseline_manifest["selected_configurations"]["fixed_pair"] != {
        "abstraction": "continuous",
        "requested_horizon": 1,
    }:
        raise CohortV2ConfirmatoryError("issue-58 fixed-pair comparator is not frozen h1/continuous")
    integrated_evidence = validate_integrated_evidence(paths["integrated"] / "evidence")
    integrated_report = json.loads(
        (paths["integrated"] / "evidence/report.json").read_bytes()
    )
    compact = json.loads(
        (root / "data/runtime_evidence/issue-58/cohort-v2-integrated-calibration-summary.json").read_bytes()
    )
    if (
        compact.get("artifact_identity") != integrated_evidence["artifact_identity"]
        or integrated_report["source_bindings"]["baseline_artifact_identity"]
        != baseline_manifest["baseline_artifact_identity"]
        or integrated_report["source_bindings"]["aggregation_artifact_identity"]
        != aggregation_manifest["aggregation_artifact_identity"]
    ):
        raise CohortV2ConfirmatoryError("issue-58 evidence source bindings differ")
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
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)


def _identity_digest(value: object) -> str:
    return f"sha256:{hashlib.sha256(str(value).encode('utf-8')).hexdigest()}"


def _compact_summary(manifest, report, protocol, implementation_commit: str):
    audit = report["source_bindings"]["access_audit"]
    bindings = report["source_bindings"]
    compact = {
        "schema": "cohort_v2_oracle_symbol_confirmatory_summary_v1",
        "artifact_identity": manifest["artifact_identity"],
        "implementation_commit": implementation_commit,
        "protocol_identity": protocol["artifact_identity"],
        "decision": report["decision"],
        "decision_rationale": report["decision_rationale"],
        "budget_decisions": report["budget_decisions"],
        "fixed_h15_complete_rollout": report["fixed_h15_complete_rollout"],
        "failed_missing_or_excluded_runs": report["failed_missing_or_excluded_runs"],
        "final_access_audit": {
            "schema": audit["schema"],
            "authorization_state": audit["authorization_state"],
            "authorization_identity": audit["authorization_identity"],
            "observed_access_count": audit["observed_access_count"],
            "passed": audit["passed"],
            "workflow_identity": audit["workflow_identity"],
            "partition_identity_sha256": _identity_digest(audit["partition_identity"]),
            "workflow_manifest_identity_sha256": _identity_digest(
                audit["workflow_manifest_identity"]
            ),
        },
        "source_bindings": {
            "access_manifest_identity_sha256": _identity_digest(
                bindings["access_manifest_identity"]
            ),
            "candidate_checkpoint_identity_sha256": _identity_digest(
                bindings["candidate_checkpoint_identity"]
            ),
            "failed_implementation_commit": bindings.get(
                "failed_implementation_commit"
            ),
            "failure_exception": bindings.get("failure_exception"),
            "failure_phase": bindings.get("failure_phase"),
            "finalization_implementation_commit": bindings.get(
                "finalization_implementation_commit"
            ),
            "sealed_bundle_identity": bindings["sealed_bundle_identity"],
            "status": "full exact bindings retained in validated ignored evidence",
        },
        "rerun_commands": [
            "python -u -m scripts.run_cohort_v2_confirmatory --validate",
        ],
    }
    if bindings.get("failed_implementation_commit") is not None:
        compact["failed_implementation_commit"] = bindings[
            "failed_implementation_commit"
        ]
    return compact


def _authorize_final_access(paths, protocol, implementation_commit: str):
    pending = FinalEvaluationWorkflowAccessManifest.from_dict(json.loads(
        (paths["plan"] / "final-evaluation-workflow-access-manifest.json").read_bytes()
    ))
    consumer = protocol["final_evaluation_access"]["later_consumer_identity"]
    if consumer != CONSUMER_ID or pending.authorization_state != "pending":
        raise CohortV2ConfirmatoryError("final workflow or issue-15 consumer differs")
    accessed_at = _utc_now()
    authorization_identity = (
        f"{CONSUMER_ID}:{protocol['artifact_identity']}:{implementation_commit}"
    )
    authorized = authorize_final_evaluation_workflow_access(
        pending,
        authorization_identity=authorization_identity,
        authorized_at=accessed_at,
    )
    artifact = authorized.authorized_artifacts[0]
    observed = {
        "workflow_identity": authorized.workflow_identity,
        "operator_identity": authorized.operator_identity,
        "artifact_identity": artifact.artifact_identity,
        "source_scenario_lineage_identities": list(
            artifact.source_scenario_lineage_identities
        ),
        "accessed_at": accessed_at,
        "authorization_identity": authorization_identity,
        "consumer_exposure_role": "final_evaluation",
    }
    partition = CohortV2PartitionExposureManifest.from_dict(json.loads(
        (paths["release"] / "partition-exposure-manifest.json").read_bytes()
    ))
    audit = audit_final_evaluation_workflow_access(
        partition, authorized, observed_accesses=[observed]
    )
    _atomic_write(paths["output"] / "authorized-final-access-manifest.json", authorized.to_dict())
    _atomic_write(paths["output"] / "final-access-audit.json", audit)
    return authorized, observed, audit


def _group_controller_decisions(
    decisions,
    *,
    controller_id: str,
    configuration_id: str,
    checkpoint_identity: str,
    seed: int,
    evaluation,
    final_reader,
    controller_compute: float,
):
    states = {state.state_id: state for state in evaluation.states}
    rollouts = {item.attempt_id: item for item in final_reader.rollouts}
    grouped = {}
    for decision in decisions:
        if decision.controller_id == controller_id:
            state = states[decision.state_id]
            grouped.setdefault(state.attempt_id, []).append((decision, state))
    rows = []
    for attempt_id in sorted(grouped):
        selected = grouped[attempt_id]
        policy_compute = []
        full_compute = []
        weighted_errors = []
        error_weights = []
        for decision, state in selected:
            pair_index = evaluation.grid.pairs.index(decision.selected_pair)
            effective = state.outcomes[pair_index].effective_horizon
            extra = controller_compute / effective
            policy_compute.append(decision.policy_compute_per_simulated_frame + extra)
            full_compute.append(decision.full_compute_per_simulated_frame + extra)
            weighted_errors.append(decision.prediction_objective * effective)
            error_weights.append(effective)
        rollout = rollouts[attempt_id]
        rows.append({
            "configuration_id": configuration_id,
            "checkpoint_identity": checkpoint_identity,
            "seed": seed,
            "attempt_id": attempt_id,
            "coverage_stratum": rollout.coverage_stratum,
            "state_count": len(selected),
            "mean_endpoint_prediction_error": sum(weighted_errors) / sum(error_weights),
            "mean_endpoint_violation_rate": mean(
                item[0].endpoint_violation_rate for item in selected
            ),
            "mean_policy_compute_per_simulated_frame": mean(policy_compute),
            "mean_full_compute_per_simulated_frame": mean(full_compute),
        })
    return tuple(rows)


def _group_fixed_pair(evaluation, measurement, final_reader):
    pair = PredictionPair(1, Abstraction.CONTINUOUS)
    measured = {item.state_id: item for item in measurement.states}
    rollouts = {item.attempt_id: item for item in final_reader.rollouts}
    grouped = {}
    for state in evaluation.states:
        index = evaluation.grid.pairs.index(pair)
        outcome = state.outcomes[index]
        pair_measurement = measured[state.state_id].outcomes[index]
        if not outcome.available or pair_measurement.endpoint_plausibility is None:
            raise CohortV2ConfirmatoryError("frozen fixed pair is unavailable")
        grouped.setdefault(state.attempt_id, []).append((outcome, pair_measurement))
    rows = []
    for attempt_id in sorted(grouped):
        selected = grouped[attempt_id]
        rollout = rollouts[attempt_id]
        rows.append({
            "configuration_id": "fixed_pair",
            "checkpoint_identity": evaluation.checkpoint_identity,
            "seed": 20260824,
            "attempt_id": attempt_id,
            "coverage_stratum": rollout.coverage_stratum,
            "state_count": len(selected),
            "mean_endpoint_prediction_error": mean(item[0].objective for item in selected),
            "mean_endpoint_violation_rate": mean(
                item[1].endpoint_plausibility.violation_rate for item in selected
            ),
            "mean_policy_compute_per_simulated_frame": mean(
                item[1].compute.policy_dependent_per_simulated_frame for item in selected
            ),
            "mean_full_compute_per_simulated_frame": mean(
                item[1].compute.full_end_to_end_per_simulated_frame for item in selected
            ),
        })
    return tuple(rows)


def _records(source, protocol, implementation_commit: str, final_reader):
    by_configuration = {
        item["configuration_id"]: [] for item in source
    }
    for item in source:
        by_configuration[item["configuration_id"]].append(item)
    result = []
    comparisons = protocol["experiment_matrix"]["confirmatory_oracle_symbol_issue_15"]["comparisons"]
    for comparison in comparisons:
        budget = float(comparison["budget"])
        comparator_id = comparison["strongest_comparator_id"]
        for role, configuration_id in (
            ("candidate", CANDIDATE_ID),
            ("comparator", comparator_id),
        ):
            for row in by_configuration[configuration_id]:
                result.append(CohortV2ConfirmatoryRecord(
                    protocol_identity=protocol["artifact_identity"],
                    release_identity=final_reader.release_identity,
                    partition_identity=final_reader.partition_identity,
                    code_revision=implementation_commit,
                    comparison_role=role,
                    budget=budget,
                    exposure_role="final_evaluation",
                    **row,
                ))
    return tuple(result)


def _dry_run(paths, frozen, args) -> int:
    scorer = CohortV2MacroPairScorer(
        frozen["predictor"], frozen["codec"], frozen["checkpoint"],
        frozen["config"], frozen["readers"],
    )
    probe = CohortV2OracleWindowDataset.ingestion_smoke(frozen["readers"][1])[0]
    value = scorer.objective(probe, PredictionPair(1, Abstraction.CONTINUOUS))
    print(f"[dry-run score] public calibration h1/continuous objective={value:.8f}", flush=True)
    print("[dry-run access] no authorization written; sealed final bundle unopened", flush=True)
    print("[dry-run] no files written", flush=True)
    print(
        "[actual command] python -u -m scripts.run_cohort_v2_confirmatory "
        "--implementation-commit <IMPLEMENTATION_COMMIT>",
        flush=True,
    )
    return 0


def _finalize_capacity_failure(
    paths,
    frozen,
    final_reader,
    access_audit,
    capacity_audit,
    *,
    implementation_commit: str,
    failed_implementation_commit: str,
) -> int:
    source_bindings = {
        "access_audit": access_audit,
        "access_manifest_identity": access_audit["workflow_manifest_identity"],
        "candidate_checkpoint_identity": frozen["checkpoint"].identity,
        "failed_implementation_commit": failed_implementation_commit,
        "failure_exception": (
            "CohortV2MicroError: engine state exceeds the declared entity slots"
        ),
        "failure_phase": "pre_evaluation_input_encoding",
        "finalization_implementation_commit": implementation_commit,
        "sealed_bundle_identity": final_reader.sealed_bundle_identity,
    }
    report = analyze_cohort_v2_confirmatory(
        (),
        (),
        frozen["protocol"],
        source_bindings=source_bindings,
        capacity_audit=capacity_audit,
    )
    manifest = write_cohort_v2_confirmatory_evidence(
        paths["output"] / "evidence",
        (),
        (),
        report,
        implementation_revision=implementation_commit,
        capacity_audit=capacity_audit,
    )
    compact = _compact_summary(
        manifest, report, frozen["protocol"], implementation_commit
    )
    _atomic_write(paths["compact"], compact)
    print(
        "[failure] frozen codec slots=12 final maximum=15; "
        "all six candidate replicates retained as failed",
        flush=True,
    )
    print("[decision] central hypothesis unsupported", flush=True)
    print(f"[complete] artifact={manifest['artifact_identity']}", flush=True)
    print(f"[report] {paths['compact']}", flush=True)
    return 0


def _finalize_existing_failure(root, paths, frozen, implementation_commit: str) -> int:
    if (
        not paths["output"].is_dir()
        or (paths["output"] / "evidence").exists()
        or (paths["output"] / "pair_evaluation").exists()
        or paths["compact"].exists()
    ):
        raise CohortV2ConfirmatoryError(
            "failure finalization requires only the partial access-audit output"
        )
    authorized = FinalEvaluationWorkflowAccessManifest.from_dict(json.loads(
        (paths["output"] / "authorized-final-access-manifest.json").read_bytes()
    ))
    access_audit = json.loads(
        (paths["output"] / "final-access-audit.json").read_bytes()
    )
    if authorized.authorization_state != "authorized" or authorized.authorized_at is None:
        raise CohortV2ConfirmatoryError("partial final access was not authorized")
    artifact = authorized.authorized_artifacts[0]
    observed = {
        "workflow_identity": authorized.workflow_identity,
        "operator_identity": authorized.operator_identity,
        "artifact_identity": artifact.artifact_identity,
        "source_scenario_lineage_identities": list(
            artifact.source_scenario_lineage_identities
        ),
        "accessed_at": authorized.authorized_at,
        "authorization_identity": authorized.authorization_identity,
        "consumer_exposure_role": "final_evaluation",
    }
    print("[finalize 1/3] revalidating the recorded authorized access", flush=True)
    final_reader = CohortV2FinalEvaluationReader(
        paths["release"],
        paths["sealed"],
        capability_declaration_path=(
            root / "docs/data_contracts/cohort_v2_capabilities_v1.json"
        ),
        production_plan_root=paths["plan"],
        access_manifest=authorized,
        observed_accesses=[observed],
    )
    if final_reader.access_audit != access_audit:
        raise CohortV2ConfirmatoryError("recorded access audit differs")
    print("[finalize 2/3] auditing carrier entity capacity only", flush=True)
    capacity_audit = audit_final_entity_capacity(
        final_reader, max_entities=frozen["config"].max_entities
    )
    if not capacity_audit or any(item["passed"] for item in capacity_audit):
        raise CohortV2ConfirmatoryError(
            "recorded failure is not reproduced on every frozen final attempt"
        )
    fixed_attempts = set(
        frozen["protocol"]["replicate_and_seed_policy"]["fixed_attempt_ids"]
    )
    if {item["attempt_id"] for item in capacity_audit} != fixed_attempts:
        raise CohortV2ConfirmatoryError("capacity audit attempt inventory differs")
    failed_commit = str(authorized.authorization_identity).rsplit(":", 1)[-1]
    print("[finalize 3/3] applying the frozen failed-run rule", flush=True)
    return _finalize_capacity_failure(
        paths,
        frozen,
        final_reader,
        access_audit,
        capacity_audit,
        implementation_commit=implementation_commit,
        failed_implementation_commit=failed_commit,
    )


def _production(root, paths, frozen, args, implementation_commit: str) -> int:
    if paths["output"].exists() or paths["compact"].exists():
        raise CohortV2ConfirmatoryError("immutable issue-15 output already exists")
    print("[access 1/3] creating the protocol-bound authorized derivative", flush=True)
    authorized, observed, access_audit = _authorize_final_access(
        paths, frozen["protocol"], implementation_commit
    )
    print("[access 2/3] access audit passed before sealed data read", flush=True)
    final_reader = CohortV2FinalEvaluationReader(
        paths["release"], paths["sealed"],
        capability_declaration_path=root / "docs/data_contracts/cohort_v2_capabilities_v1.json",
        production_plan_root=paths["plan"],
        access_manifest=authorized,
        observed_accesses=[observed],
    )
    fixed_attempts = frozen["protocol"]["replicate_and_seed_policy"]["fixed_attempt_ids"]
    if {item.attempt_id for item in final_reader.rollouts} != set(fixed_attempts):
        raise CohortV2ConfirmatoryError("opened final attempts differ from issue-34")
    print(
        f"[access 3/3] authorized final rollouts={len(final_reader.rollouts)} "
        f"frame_records={sum(len(item.frame_records) for item in final_reader.rollouts)}",
        flush=True,
    )
    capacity_audit = audit_final_entity_capacity(
        final_reader, max_entities=frozen["config"].max_entities
    )
    if any(item["passed"] is False for item in capacity_audit):
        return _finalize_capacity_failure(
            paths,
            frozen,
            final_reader,
            access_audit,
            capacity_audit,
            implementation_commit=implementation_commit,
            failed_implementation_commit=implementation_commit,
        )

    scorers = []
    total = sum(len(item.frame_records) - 1 for item in final_reader.rollouts) * 9
    devices = _evaluation_devices(args)
    for device in devices:
        predictor, codec, checkpoint = load_cohort_v2_integrated_checkpoint(
            paths["integrated"] / "models/integrated_ordered_flat_reliability_gated/checkpoint.pt",
            reader=frozen["readers"][0],
            config=_config(),
            variant=IntegratedVariant.CANDIDATE,
            reliability_artifact_identity=str(frozen["reliability_artifact_identity"]),
            device=device,
        )
        scorers.append(CohortV2MacroPairScorer(
            predictor, codec, checkpoint, _config(), (final_reader,),
            progress_every=args.score_log_every,
            progress_total=total,
            worker_name=f"final:{device}",
        ))
    print(
        f"[evaluate] exhaustive final 3x3 grid devices={devices} "
        f"batch={args.evaluation_batch_size}",
        flush=True,
    )
    evaluation = CohortV2ParallelExhaustiveEvaluator(
        tuple(scorers), batch_size=args.evaluation_batch_size
    ).evaluate((final_reader,))
    write_cohort_v2_evaluation(
        paths["output"] / "pair_evaluation", evaluation, readers=(final_reader,)
    )
    compute = integrated_compute_calibration(
        frozen["config"], IntegratedVariant.CANDIDATE
    )
    profile = CohortV2ExecutionProfile(False, True)
    write_cohort_v2_measurements(
        paths["output"] / "pair_measurements", evaluation,
        readers=(final_reader,), calibration=compute, profile=profile,
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
        paths["output"] / "trajectory_labels", evaluation, measurement, spec,
        implementation_revision=implementation_commit,
    )
    examples = build_cohort_v2_controller_examples(
        (final_reader,), labels, frozen["controller_config"],
        included_roles=("final_evaluation",),
    )
    controller_result = evaluate_cohort_v2_controllers(
        (frozen["candidate_model"], frozen["two_head_model"]),
        examples, evaluation, measurement, spec,
        evaluation_roles=("final_evaluation",),
    )
    controller_compute = integrated_compute_calibration(
        frozen["config"], IntegratedVariant.CANDIDATE,
        controller_config=frozen["controller_config"],
    ).controller_per_decision
    aggregation_checkpoint = frozen["aggregation_manifest"]["artifacts"]["checkpoint"]["identity"]
    grouped = [
        *_group_controller_decisions(
            controller_result.decisions,
            controller_id="joint_pair", configuration_id=CANDIDATE_ID,
            checkpoint_identity=aggregation_checkpoint, seed=10,
            evaluation=evaluation, final_reader=final_reader,
            controller_compute=controller_compute,
        ),
        *_group_controller_decisions(
            controller_result.decisions,
            controller_id="matched_capacity_two_head",
            configuration_id="matched_capacity_two_head",
            checkpoint_identity=frozen["two_head_checkpoint_identity"], seed=10,
            evaluation=evaluation, final_reader=final_reader,
            controller_compute=controller_compute,
        ),
        *_group_fixed_pair(evaluation, measurement, final_reader),
    ]
    records = _records(
        tuple(grouped), frozen["protocol"], implementation_commit, final_reader
    )
    print("[recursive] fixed h15 complete-rollout diagnostics", flush=True)
    recursive = recursive_continuous_rollouts(
        frozen["predictor"], frozen["codec"], frozen["checkpoint"].identity,
        (final_reader,), compute, requested_horizons=(15,),
    )
    source_bindings = {
        "access_audit": access_audit,
        "access_manifest_identity": authorized.identity,
        "aggregation_artifact_identity": frozen["aggregation_manifest"]["aggregation_artifact_identity"],
        "baseline_artifact_identity": frozen["baseline_manifest"]["baseline_artifact_identity"],
        "candidate_checkpoint_identity": frozen["checkpoint"].identity,
        "evaluation_identity": evaluation.identity,
        "measurement_identity": measurement.identity,
        "trajectory_label_artifact_identity": label_receipt.label_artifact_identity,
        "sealed_bundle_identity": final_reader.sealed_bundle_identity,
    }
    report = analyze_cohort_v2_confirmatory(
        records, recursive, frozen["protocol"], source_bindings=source_bindings
    )
    manifest = write_cohort_v2_confirmatory_evidence(
        paths["output"] / "evidence", records, recursive, report,
        implementation_revision=implementation_commit,
    )
    compact = _compact_summary(
        manifest, report, frozen["protocol"], implementation_commit
    )
    _atomic_write(paths["compact"], compact)
    print(f"[decision] central hypothesis {report['decision']}", flush=True)
    print(f"[complete] artifact={manifest['artifact_identity']}", flush=True)
    print(f"[report] {paths['compact']}", flush=True)
    return 0


def _validate(paths, frozen) -> int:
    manifest = validate_cohort_v2_confirmatory_evidence(
        paths["output"] / "evidence", frozen["protocol"]
    )
    authorized = FinalEvaluationWorkflowAccessManifest.from_dict(json.loads(
        (paths["output"] / "authorized-final-access-manifest.json").read_bytes()
    ))
    audit = json.loads((paths["output"] / "final-access-audit.json").read_bytes())
    compact = json.loads(paths["compact"].read_bytes())
    report = json.loads(
        (paths["output"] / "evidence/report.json").read_bytes()
    )
    expected_compact = _compact_summary(
        manifest,
        report,
        frozen["protocol"],
        manifest["implementation_revision"],
    )
    if (
        authorized.authorization_state != "authorized"
        or audit.get("passed") is not True
        or audit.get("workflow_manifest_identity") != authorized.identity
        or compact.get("artifact_identity") != manifest["artifact_identity"]
        or compact.get("protocol_identity") != frozen["protocol"]["artifact_identity"]
        or compact != expected_compact
    ):
        raise CohortV2ConfirmatoryError("stored issue-15 evidence bindings differ")
    print(
        f"[validate] exact confirmatory validation passed artifact={manifest['artifact_identity']}",
        flush=True,
    )
    return 0


def _rewrite_compact(paths, frozen) -> int:
    manifest = validate_cohort_v2_confirmatory_evidence(
        paths["output"] / "evidence", frozen["protocol"]
    )
    report = json.loads(
        (paths["output"] / "evidence/report.json").read_bytes()
    )
    compact = _compact_summary(
        manifest,
        report,
        frozen["protocol"],
        manifest["implementation_revision"],
    )
    _atomic_write(paths["compact"], compact)
    print(f"[compact] rewrote {paths['compact']}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if sum((
        args.dry_run, args.validate, args.finalize_failure, args.rewrite_compact
    )) > 1:
        parser.error(
            "--dry-run, --validate, --finalize-failure, and --rewrite-compact "
            "are mutually exclusive"
        )
    if args.score_log_every <= 0 or args.evaluation_batch_size <= 0:
        parser.error("progress interval and evaluation batch size must be positive")
    root = args.repository_root.resolve()
    paths = _paths(args, root)
    implementation = args.implementation_commit
    if implementation is None:
        implementation, dirty = git_revision(str(root))
        if dirty and not args.dry_run and not args.validate:
            parser.error("a dirty worktree requires --implementation-commit")
    print("[design] loading frozen issue-34 protocol and issue-58 model sources", flush=True)
    frozen = _load_frozen_sources(
        root,
        paths,
        "cpu" if (
            args.dry_run or args.validate or args.finalize_failure
            or args.rewrite_compact
        ) else args.device,
    )
    print(
        "[design] candidate=integrated_aggregated_joint_controller "
        "comparators=(matched_capacity_two_head,fixed_pair) final_rollouts=6",
        flush=True,
    )
    if args.dry_run:
        return _dry_run(paths, frozen, args)
    if args.validate:
        return _validate(paths, frozen)
    if args.finalize_failure:
        return _finalize_existing_failure(root, paths, frozen, implementation)
    if args.rewrite_compact:
        return _rewrite_compact(paths, frozen)
    return _production(root, paths, frozen, args, implementation)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CohortV2ConfirmatoryError as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(2) from error
