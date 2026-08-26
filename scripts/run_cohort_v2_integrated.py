"""Train and calibrate the integrated BG-NS-JEPA candidate (issue #58)."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Final

import torch

from scripts.run_cohort_v2_macro_experiment import (
    DEFAULT_RELEASE,
    _evaluation_devices,
    _readers,
)
from scripts.run_cohort_v2_trajectory_labels import issue_8_cost_spec
from world_model.training.cohort_v2_aggregation import (
    CohortV2AggregationConfig,
    run_cohort_v2_controller_aggregation,
    validate_cohort_v2_controller_aggregation,
    write_cohort_v2_controller_aggregation,
)
from world_model.training.cohort_v2_baselines import (
    generate_cohort_v2_policy_baselines,
    validate_cohort_v2_policy_baselines,
    write_cohort_v2_policy_baselines,
)
from world_model.training.cohort_v2_calibration import CohortV2CalibrationRecord
from world_model.training.cohort_v2_controller import (
    CohortV2ControllerConfig,
    build_cohort_v2_controller_examples,
    evaluate_cohort_v2_controllers,
    load_cohort_v2_controller_checkpoint,
    validate_cohort_v2_controllers,
    write_cohort_v2_controllers,
)
from world_model.training.cohort_v2_evaluation import (
    CohortV2ParallelExhaustiveEvaluator,
    load_cohort_v2_evaluation,
    validate_cohort_v2_evaluation,
    write_cohort_v2_evaluation,
)
from world_model.training.cohort_v2_integrated import (
    CohortV2IntegratedError,
    IntegratedVariant,
    analyze_integrated_calibration,
    build_integrated_trainer,
    integrated_compute_calibration,
    load_cohort_v2_integrated_checkpoint,
    recursive_continuous_rollouts,
    save_cohort_v2_integrated_checkpoint,
    validate_integrated_evidence,
    write_integrated_evidence,
)
from world_model.training.cohort_v2_macro import (
    MACRO_CAPABILITIES,
    MACRO_PAIRS,
    CohortV2MacroCheckpoint,
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
from world_model.training.loop import seed_all
from world_model.training.manifest import git_revision


DEFAULT_RELIABILITY: Final = Path(".local-artifacts/issue-12-reliability")
DEFAULT_OUTPUT: Final = Path(".local-artifacts/issue-58-integrated")
DEFAULT_COMPACT_REPORT: Final = Path(
    "data/runtime_evidence/issue-58/cohort-v2-integrated-calibration-summary.json"
)
CANDIDATE_ID: Final = "integrated_aggregated_joint_controller"
BASELINE_IDS: Final = (
    "fixed_pair",
    "temporal_only",
    "description_only",
    "uniformly_marginalized_independent_axes",
)
COMPARATOR_IDS: Final = (*BASELINE_IDS, "matched_capacity_two_head")


def _short_identity(value: object) -> str:
    return f"sha256:{hashlib.sha256(str(value).encode('utf-8')).hexdigest()[:16]}"


def _compact_source_bindings(
    evidence_manifest: dict[str, object],
    *,
    implementation_revision: str,
    release_identity: str,
) -> dict[str, object]:
    return {
        "full_evidence_artifact_identity": evidence_manifest["artifact_identity"],
        "full_report_identity": evidence_manifest["analysis_identity"],
        "implementation_revision": implementation_revision,
        "release_identity": release_identity,
        "status": "full source bindings retained in validated ignored evidence",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--reliability-root", type=Path, default=DEFAULT_RELIABILITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compact-report", type=Path, default=DEFAULT_COMPACT_REPORT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluation-devices", default="auto")
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--symbolic-weight", type=float, default=1.0)
    parser.add_argument("--controller-seed", type=int, default=10)
    parser.add_argument("--aggregation-rounds", type=int, default=1)
    parser.add_argument("--bootstrap-seed", type=int, default=20260826)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--score-log-every", type=int, default=250)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser


def _config(args: argparse.Namespace) -> CohortV2MacroConfig:
    return CohortV2MacroConfig(
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        symbolic_weight=args.symbolic_weight,
        device=args.device,
    )


def _frozen_design(args: argparse.Namespace) -> None:
    expected = {
        "seed": 20260824,
        "steps": 1800,
        "batch_size": 32,
        "learning_rate": 3e-4,
        "symbolic_weight": 1.0,
        "controller_seed": 10,
        "aggregation_rounds": 1,
        "bootstrap_seed": 20260826,
    }
    actual = {key: getattr(args, key) for key in expected}
    if not args.dry_run and actual != expected:
        raise CohortV2IntegratedError(
            f"production must use the frozen issue-58 design: {expected}"
        )


def _paths(args: argparse.Namespace, root: Path) -> dict[str, Path]:
    return {
        "release": (root / args.release_root).resolve(),
        "reliability": (root / args.reliability_root).resolve(),
        "output": (root / args.output).resolve(),
        "compact": (root / args.compact_report).resolve(),
    }


def _variant_root(output: Path, variant: IntegratedVariant) -> Path:
    return output / "models" / str(variant)


def _score_probe(trainer, variant: IntegratedVariant) -> dict[str, float]:
    checkpoint = CohortV2MacroCheckpoint(
        Path("dry-run"),
        f"dry-run:{variant}",
        trainer.step_count,
        tuple((pair.identity, trainer.pair_counts[pair]) for pair in MACRO_PAIRS),
    )
    scorer = CohortV2MacroPairScorer(
        trainer.predictor,
        trainer.codec,
        checkpoint,
        trainer.config,
        (trainer.data.reader,),
    )
    values = {}
    for pair in MACRO_PAIRS:
        window = trainer.data.pools[pair][0]
        values[pair.identity] = scorer.objective(window, pair)
    return values


def _synthetic_dry_records(readers) -> tuple[CohortV2CalibrationRecord, ...]:
    result = []
    configurations = (CANDIDATE_ID, *COMPARATOR_IDS)
    for role_index, reader in enumerate(readers[1:], start=1):
        for rollout_index, rollout in enumerate(reader.rollouts):
            for config_index, configuration_id in enumerate(configurations):
                result.append(CohortV2CalibrationRecord(
                    configuration_id=configuration_id,
                    exposure_role=rollout.exposure_role,
                    attempt_id=rollout.attempt_id,
                    scenario_lineage_identity=rollout.scenario_lineage_identity,
                    coverage_stratum=rollout.coverage_stratum,
                    checkpoint_identity=f"dry-run:{configuration_id}",
                    seed=10,
                    state_count=1,
                    mean_endpoint_prediction_error=(
                        0.01 + config_index * 0.001 + role_index * 0.0001
                        + rollout_index * 0.00001
                    ),
                    mean_endpoint_violation_rate=0.0,
                    mean_policy_compute_per_simulated_frame=100.0 + config_index,
                    mean_full_compute_per_simulated_frame=100.0 + config_index,
                ))
    return tuple(result)


def _dry_run(args, readers, reliability_estimator, reliability_config) -> int:
    config = replace(_config(args), steps=9, batch_size=min(2, args.batch_size))
    parameter_counts = set()
    candidate = None
    for index, variant in enumerate(IntegratedVariant, start=1):
        print(f"[dry-run model {index}/3] {variant}", flush=True)
        seed_all(config.seed)
        trainer = build_integrated_trainer(
            readers[0],
            config,
            variant,
            reliability_estimator=reliability_estimator,
            reliability_config=reliability_config,
            reliability_readers=readers,
        )
        for step in range(config.steps):
            latest = trainer.train_step()
            print(
                f"[dry-run train {variant} {step + 1}/{config.steps}] "
                f"pair=h{latest.pair.delta}/{latest.pair.abstraction} "
                f"loss={latest.total_loss:.6f}",
                flush=True,
            )
        if set(trainer.pair_counts.values()) != {1}:
            raise CohortV2IntegratedError("dry-run did not visit every pair exactly once")
        probe = _score_probe(trainer, variant)
        print(
            f"[dry-run score {variant}] pairs={len(probe)} "
            f"mean={mean(probe.values()):.6f}",
            flush=True,
        )
        parameter_counts.add(sum(p.numel() for p in trainer.predictor.parameters()))
        if variant is IntegratedVariant.CANDIDATE:
            candidate = trainer
    if len(parameter_counts) != 1 or candidate is None:
        raise CohortV2IntegratedError("integrated stress variants are not matched capacity")
    compute = integrated_compute_calibration(config, IntegratedVariant.CANDIDATE)
    recursive = recursive_continuous_rollouts(
        candidate.predictor,
        candidate.codec,
        "dry-run:candidate",
        readers[1:],
        compute,
    )
    report = analyze_integrated_calibration(
        _synthetic_dry_records(readers),
        recursive,
        candidate_configuration_id=CANDIDATE_ID,
        comparator_configuration_ids=COMPARATOR_IDS,
        source_bindings={"dry_run": True},
        bootstrap_seed=args.bootstrap_seed,
    )
    print(
        f"[dry-run recursive] records={len(recursive)} "
        f"selected_horizon={report['complete_rollout_metrics']['strongest_horizon_selected_on_model_selection']}",
        flush=True,
    )
    print("[dry-run] no files written; final evaluation remained sealed", flush=True)
    print(
        "[actual command] python -u -m scripts.run_cohort_v2_integrated "
        "--implementation-commit <IMPLEMENTATION_COMMIT>",
        flush=True,
    )
    return 0


def _train_variant(
    variant: IntegratedVariant,
    config: CohortV2MacroConfig,
    readers,
    reliability_estimator,
    reliability_config,
    reliability_artifact_identity: str,
    root: Path,
    args,
):
    seed_all(config.seed)
    trainer = build_integrated_trainer(
        readers[0],
        config,
        variant,
        reliability_estimator=reliability_estimator,
        reliability_config=reliability_config,
        reliability_readers=readers,
    )
    print(
        f"[train {variant}] steps={config.steps} exactly={config.steps // 9}/pair "
        f"batch={config.batch_size} device={config.device}",
        flush=True,
    )
    for step in range(config.steps):
        latest = trainer.train_step()
        if step == 0 or step + 1 == config.steps or (step + 1) % args.log_every == 0:
            print(
                f"[train {variant} {step + 1}/{config.steps}] "
                f"pair=h{latest.pair.delta}/{latest.pair.abstraction} "
                f"total={latest.total_loss:.6f} carrier={latest.carrier_loss:.6f} "
                f"micro={latest.micro_loss:.6f} macro={latest.macro_loss:.6f} "
                f"lr={latest.learning_rate:.2e}",
                flush=True,
            )
    checkpoint = save_cohort_v2_integrated_checkpoint(
        root / "checkpoint.pt",
        trainer,
        variant,
        reliability_artifact_identity=reliability_artifact_identity,
    )
    parameter_count = sum(p.numel() for p in trainer.predictor.parameters())
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scorers = []
    total = sum(
        len(rollout.frame_records) - 1
        for reader in readers for rollout in reader.rollouts
    ) * len(MACRO_PAIRS)
    for device in _evaluation_devices(args):
        predictor, codec, loaded = load_cohort_v2_integrated_checkpoint(
            root / "checkpoint.pt",
            reader=readers[0],
            config=config,
            variant=variant,
            reliability_artifact_identity=reliability_artifact_identity,
            device=device,
        )
        scorers.append(CohortV2MacroPairScorer(
            predictor,
            codec,
            loaded,
            config,
            readers,
            progress_every=args.score_log_every,
            progress_total=total,
            worker_name=f"{variant}:{device}",
        ))
    print(
        f"[evaluate {variant}] exhaustive 3x3 grid devices={_evaluation_devices(args)} "
        f"batch={args.evaluation_batch_size}",
        flush=True,
    )
    evaluation = CohortV2ParallelExhaustiveEvaluator(
        tuple(scorers), batch_size=args.evaluation_batch_size
    ).evaluate(readers)
    receipt = write_cohort_v2_evaluation(
        root / "pair_evaluation", evaluation, readers=readers
    )
    compute = integrated_compute_calibration(config, variant)
    profile = CohortV2ExecutionProfile(
        controller_executed=False, shared_perception_executed=True
    )
    measurement_receipt = write_cohort_v2_measurements(
        root / "pair_measurements",
        evaluation,
        readers=readers,
        calibration=compute,
        profile=profile,
    )
    measurement = measure_cohort_v2_evaluation(
        evaluation, readers, compute, profile
    )
    print(
        f"[model complete {variant}] states={receipt.state_count} "
        f"available={receipt.available_count} measurement={measurement_receipt.measurement_identity}",
        flush=True,
    )
    del scorers
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return checkpoint, parameter_count, evaluation, measurement, compute


def _group_decisions(
    decisions,
    *,
    configuration_id: str,
    checkpoint_identity: str,
    seed: int,
    evaluation,
    readers,
    controller_compute_per_decision: float = 0.0,
) -> tuple[CohortV2CalibrationRecord, ...]:
    states = {state.state_id: state for state in evaluation.states}
    rollouts = {
        rollout.attempt_id: rollout
        for reader in readers for rollout in reader.rollouts
    }
    grouped = {}
    for decision in decisions:
        state = states[decision.state_id]
        grouped.setdefault(state.attempt_id, []).append((decision, state))
    result = []
    for attempt_id in sorted(grouped):
        rows = grouped[attempt_id]
        if rows[0][0].exposure_role not in ("calibration", "model_selection"):
            continue
        policy_compute = []
        full_compute = []
        for decision, state in rows:
            pair_index = evaluation.grid.pairs.index(decision.selected_pair)
            effective = state.outcomes[pair_index].effective_horizon
            extra = controller_compute_per_decision / effective
            policy_compute.append(float(decision.policy_compute_per_simulated_frame) + extra)
            full_compute.append(float(decision.full_compute_per_simulated_frame) + extra)
        rollout = rollouts[attempt_id]
        result.append(CohortV2CalibrationRecord(
            configuration_id=configuration_id,
            exposure_role=rows[0][0].exposure_role,
            attempt_id=attempt_id,
            scenario_lineage_identity=rollout.scenario_lineage_identity,
            coverage_stratum=rollout.coverage_stratum,
            checkpoint_identity=checkpoint_identity,
            seed=seed,
            state_count=len(rows),
            mean_endpoint_prediction_error=mean(
                float(row[0].prediction_objective) for row in rows
            ),
            mean_endpoint_violation_rate=mean(
                float(row[0].endpoint_violation_rate) for row in rows
            ),
            mean_policy_compute_per_simulated_frame=mean(policy_compute),
            mean_full_compute_per_simulated_frame=mean(full_compute),
        ))
    return tuple(result)


def _stress_ablations(evaluations) -> dict[str, object]:
    candidate = evaluations[IntegratedVariant.CANDIDATE]
    candidate_states = {state.state_id: state for state in candidate.states}
    result = {}
    for stress_id, variant in (
        ("integrated_no_symbol", IntegratedVariant.NO_SYMBOL),
        ("ordered_flat_without_reliability_gate", IntegratedVariant.UNGATED),
    ):
        gaps = {}
        for stressed in evaluations[variant].states:
            if stressed.exposure_role != "calibration":
                continue
            reference = candidate_states[stressed.state_id]
            paired = tuple(
                float(left.objective) - float(right.objective)
                for left, right in zip(stressed.outcomes, reference.outcomes, strict=True)
                if left.objective is not None and right.objective is not None
            )
            gaps.setdefault(stressed.attempt_id, []).extend(paired)
        rollout_gaps = {
            attempt_id: mean(values) for attempt_id, values in sorted(gaps.items())
        }
        result[stress_id] = {
            "analysis_unit": "complete_calibration_rollout",
            "local_teacher_forced_degradation_gaps": rollout_gaps,
            "mean_degradation_gap": mean(rollout_gaps.values()),
            "reference": str(IntegratedVariant.CANDIDATE),
            "stressed": str(variant),
        }
    return result


def _production(args, paths, readers, reliability_estimator, reliability_config, reliability_manifest):
    output = paths["output"]
    if output.exists():
        raise CohortV2IntegratedError(f"immutable output already exists: {output}")
    config = _config(args)
    reliability_identity = str(reliability_manifest["artifact_identity"])
    checkpoints = {}
    evaluations = {}
    measurements = {}
    compute = {}
    parameter_counts = set()
    for index, variant in enumerate(IntegratedVariant, start=1):
        print(f"[model {index}/3] {variant}", flush=True)
        values = _train_variant(
            variant,
            config,
            readers,
            reliability_estimator,
            reliability_config,
            reliability_identity,
            _variant_root(output, variant),
            args,
        )
        checkpoints[variant], parameter_count, evaluations[variant], measurements[variant], compute[variant] = values
        parameter_counts.add(parameter_count)
    if len(parameter_counts) != 1:
        raise CohortV2IntegratedError("integrated variants are not matched capacity")

    candidate_evaluation = evaluations[IntegratedVariant.CANDIDATE]
    candidate_measurement = measurements[IntegratedVariant.CANDIDATE]
    candidate_compute = compute[IntegratedVariant.CANDIDATE]
    spec = issue_8_cost_spec(candidate_compute)
    downstream = output / "candidate_pipeline"
    print("[pipeline 1/4] trajectory-optimal labels", flush=True)
    trajectory_receipt = write_cohort_v2_trajectory_labels(
        downstream / "trajectory_labels",
        candidate_evaluation,
        candidate_measurement,
        spec,
        implementation_revision=args.implementation_commit,
    )
    labels = generate_cohort_v2_trajectory_labels(
        candidate_evaluation, candidate_measurement, spec
    )
    print("[pipeline 2/4] four issue-9 policy baselines", flush=True)
    baseline_receipt = write_cohort_v2_policy_baselines(
        downstream / "policy_baselines",
        candidate_evaluation,
        candidate_measurement,
        spec,
        trajectory_label_artifact_identity=trajectory_receipt.label_artifact_identity,
        derivation_index_identity=readers[0].derivation_identity,
        implementation_revision=args.implementation_commit,
    )
    baseline_result = generate_cohort_v2_policy_baselines(
        candidate_evaluation,
        candidate_measurement,
        spec,
        trajectory_label_artifact_identity=trajectory_receipt.label_artifact_identity,
        derivation_index_identity=readers[0].derivation_identity,
    )
    controller_config = CohortV2ControllerConfig(seed=args.controller_seed)
    print("[pipeline 3/4] matched joint and two-head controllers", flush=True)
    controller_receipt = write_cohort_v2_controllers(
        downstream / "controllers",
        readers,
        candidate_evaluation,
        candidate_measurement,
        labels,
        spec,
        controller_config,
        trajectory_label_artifact_identity=trajectory_receipt.label_artifact_identity,
        baseline_artifact_identity=baseline_receipt.baseline_artifact_identity,
        derivation_index_identity=readers[0].derivation_identity,
        implementation_revision=args.implementation_commit,
        progress=lambda message: print(message, flush=True),
    )
    base_models, loaded_controller_config, controller_checkpoint_identity = (
        load_cohort_v2_controller_checkpoint(downstream / "controllers/checkpoint.pt")
    )
    if loaded_controller_config != controller_config:
        raise CohortV2IntegratedError("stored controller configuration differs")
    candidate_predictor, candidate_codec, candidate_checkpoint = (
        load_cohort_v2_integrated_checkpoint(
            _variant_root(output, IntegratedVariant.CANDIDATE) / "checkpoint.pt",
            reader=readers[0],
            config=config,
            variant=IntegratedVariant.CANDIDATE,
            reliability_artifact_identity=reliability_identity,
            device=args.device,
        )
    )
    print("[pipeline 4/4] one closed-loop aggregation round", flush=True)
    aggregation_config = CohortV2AggregationConfig(args.aggregation_rounds)
    aggregation_run = run_cohort_v2_controller_aggregation(
        readers,
        candidate_evaluation,
        candidate_measurement,
        labels,
        spec,
        candidate_predictor,
        candidate_codec,
        base_models,
        controller_config,
        aggregation_config,
        progress=lambda message: print(message, flush=True),
    )
    aggregation_manifest = write_cohort_v2_controller_aggregation(
        downstream / "aggregation",
        aggregation_run,
        controller_config,
        aggregation_config,
        candidate_evaluation,
        candidate_measurement,
        spec,
        source_controller_artifact_identity=controller_receipt.controller_artifact_identity,
        source_controller_checkpoint_identity=controller_checkpoint_identity,
        source_predictor_checkpoint_identity=candidate_checkpoint.identity,
        trajectory_label_artifact_identity=trajectory_receipt.label_artifact_identity,
        derivation_index_identity=readers[0].derivation_identity,
        implementation_revision=args.implementation_commit,
    )

    print("[calibration] building rollout-level local records", flush=True)
    records = []
    for policy_id in BASELINE_IDS:
        records.extend(_group_decisions(
            tuple(
                item for item in baseline_result.decisions if item.policy_id == policy_id
            ),
            configuration_id=policy_id,
            checkpoint_identity=candidate_checkpoint.identity,
            seed=config.seed,
            evaluation=candidate_evaluation,
            readers=readers,
        ))
    examples = build_cohort_v2_controller_examples(
        readers,
        labels,
        controller_config,
        included_roles=("training", "calibration", "model_selection"),
    )
    base_result = evaluate_cohort_v2_controllers(
        base_models,
        examples,
        candidate_evaluation,
        candidate_measurement,
        spec,
        evaluation_roles=("calibration", "model_selection"),
    )
    controller_compute = integrated_compute_calibration(
        config,
        IntegratedVariant.CANDIDATE,
        controller_config=controller_config,
    ).controller_per_decision
    records.extend(_group_decisions(
        tuple(
            item for item in base_result.decisions
            if item.controller_id == "matched_capacity_two_head"
        ),
        configuration_id="matched_capacity_two_head",
        checkpoint_identity=controller_checkpoint_identity,
        seed=controller_config.seed,
        evaluation=candidate_evaluation,
        readers=readers,
        controller_compute_per_decision=controller_compute,
    ))
    aggregated_result = evaluate_cohort_v2_controllers(
        aggregation_run.round_models[0],
        examples,
        candidate_evaluation,
        candidate_measurement,
        spec,
        evaluation_roles=("calibration", "model_selection"),
    )
    records.extend(_group_decisions(
        tuple(
            item for item in aggregated_result.decisions
            if item.controller_id == "joint_pair"
        ),
        configuration_id=CANDIDATE_ID,
        checkpoint_identity=str(
            aggregation_manifest["artifacts"]["checkpoint"]["identity"]
        ),
        seed=controller_config.seed,
        evaluation=candidate_evaluation,
        readers=readers,
        controller_compute_per_decision=controller_compute,
    ))
    records = tuple(records)
    for index, attempt_id in enumerate(sorted({
        item.attempt_id for item in records if item.exposure_role == "calibration"
    }), start=1):
        print(f"[calibration rollout {index}/6] attempt={attempt_id}", flush=True)

    print("[recursive] fixed h1/h5/h15 over equal complete-rollout duration", flush=True)
    recursive = recursive_continuous_rollouts(
        candidate_predictor,
        candidate_codec,
        candidate_checkpoint.identity,
        readers[1:],
        candidate_compute,
    )
    for role in ("calibration", "model_selection"):
        print(
            f"[recursive {role}] records={sum(item.exposure_role == role for item in recursive)}",
            flush=True,
        )
    stress = _stress_ablations(evaluations)
    bindings = {
        "aggregation_artifact_identity": aggregation_manifest["aggregation_artifact_identity"],
        "baseline_artifact_identity": baseline_receipt.baseline_artifact_identity,
        "candidate_checkpoint_identity": candidate_checkpoint.identity,
        "capability_declaration_identity": candidate_evaluation.capability_declaration_identity,
        "controller_artifact_identity": controller_receipt.controller_artifact_identity,
        "derivation_index_identity": readers[0].derivation_identity,
        "evaluation_identities": {
            str(variant): evaluation.identity
            for variant, evaluation in evaluations.items()
        },
        "measurement_identities": {
            str(variant): measurement.identity
            for variant, measurement in measurements.items()
        },
        "partition_identity": candidate_evaluation.partition_identity,
        "release_identity": candidate_evaluation.release_identity,
        "reliability_artifact_identity": reliability_identity,
        "trajectory_label_artifact_identity": trajectory_receipt.label_artifact_identity,
        "variant_checkpoint_identities": {
            str(variant): checkpoint.identity
            for variant, checkpoint in checkpoints.items()
        },
    }
    report = analyze_integrated_calibration(
        records,
        recursive,
        candidate_configuration_id=CANDIDATE_ID,
        comparator_configuration_ids=COMPARATOR_IDS,
        source_bindings=bindings,
        stress_ablations=stress,
        bootstrap_seed=args.bootstrap_seed,
    )
    evidence_manifest = write_integrated_evidence(
        output / "evidence",
        records,
        recursive,
        report,
        implementation_revision=args.implementation_commit,
    )
    compact = {
        "artifact_identity": evidence_manifest["artifact_identity"],
        "artifact_type": "cohort_v2_integrated_model_calibration_summary",
        "disposition": report["disposition"],
        "exposure_audit": report["exposure_audit"],
        "implementation_commit": args.implementation_commit,
        "independent_calibration_rollouts": report[
            "independent_calibration_rollouts"
        ],
        "local_teacher_forced_metrics": report["local_teacher_forced_metrics"],
        "complete_rollout_metrics": report["complete_rollout_metrics"],
        "proposals_for_issue_34": report["proposals_for_issue_34"],
        "sensitivity": report["sensitivity"],
        "matched_capacity_parameter_count": next(iter(parameter_counts)),
        "recursive_physical_violation_status": "unavailable",
        "release_identity": candidate_evaluation.release_identity,
        "rerun_commands": [
            "python -u -m scripts.run_cohort_v2_integrated --dry-run",
            "python -u -m scripts.run_cohort_v2_integrated "
            f"--implementation-commit {args.implementation_commit}",
            "python -u -m scripts.run_cohort_v2_integrated --validate",
        ],
        "schema": "cohort_v2_integrated_model_calibration_summary_v1",
        "source_bindings": _compact_source_bindings(
            evidence_manifest,
            implementation_revision=args.implementation_commit,
            release_identity=candidate_evaluation.release_identity,
        ),
        "stress_ablations": stress,
    }
    paths["compact"].parent.mkdir(parents=True, exist_ok=True)
    paths["compact"].write_bytes(canonical_json_bytes(compact))
    print(
        f"[disposition] {report['disposition']['status']} "
        f"artifact={evidence_manifest['artifact_identity']}",
        flush=True,
    )
    print(f"[report] {paths['compact']}", flush=True)
    return 0


def _validate(args, paths, readers, reliability_manifest) -> int:
    config = _config(args)
    output = paths["output"]
    reliability_identity = str(reliability_manifest["artifact_identity"])
    parameter_counts = set()
    loaded = {}
    for index, variant in enumerate(IntegratedVariant, start=1):
        root = _variant_root(output, variant)
        predictor, codec, checkpoint = load_cohort_v2_integrated_checkpoint(
            root / "checkpoint.pt",
            reader=readers[0],
            config=config,
            variant=variant,
            reliability_artifact_identity=reliability_identity,
            device=args.device,
        )
        parameter_counts.add(sum(p.numel() for p in predictor.parameters()))
        scorer = CohortV2MacroPairScorer(
            predictor, codec, checkpoint, config, readers
        )
        validate_cohort_v2_evaluation(
            root / "pair_evaluation",
            readers=readers,
            checkpoint_identity=checkpoint.identity,
            checkpoint_capabilities=MACRO_CAPABILITIES,
            objective_identity=scorer.objective_identity,
        )
        evaluation = load_cohort_v2_evaluation(
            root / "pair_evaluation",
            readers=readers,
            checkpoint_identity=checkpoint.identity,
            checkpoint_capabilities=MACRO_CAPABILITIES,
            objective_identity=scorer.objective_identity,
        )
        compute = integrated_compute_calibration(config, variant)
        profile = CohortV2ExecutionProfile(False, True)
        validate_cohort_v2_measurements(
            root / "pair_measurements",
            evaluation,
            readers=readers,
            calibration=compute,
            profile=profile,
        )
        measurement = measure_cohort_v2_evaluation(
            evaluation, readers, compute, profile
        )
        loaded[variant] = (predictor, codec, checkpoint, evaluation, measurement, compute)
        print(f"[validate model {index}/3] {variant}", flush=True)
    if len(parameter_counts) != 1:
        raise CohortV2IntegratedError("validated variants differ in capacity")
    candidate = loaded[IntegratedVariant.CANDIDATE]
    downstream = output / "candidate_pipeline"
    spec = issue_8_cost_spec(candidate[5])
    labels_manifest = json.loads(
        (downstream / "trajectory_labels/manifest.json").read_bytes()
    )
    trajectory_receipt = validate_cohort_v2_trajectory_labels(
        downstream / "trajectory_labels",
        candidate[3],
        candidate[4],
        spec,
        implementation_revision=labels_manifest["implementation_revision"],
    )
    labels = generate_cohort_v2_trajectory_labels(candidate[3], candidate[4], spec)
    baseline_manifest = json.loads(
        (downstream / "policy_baselines/manifest.json").read_bytes()
    )
    baseline_receipt = validate_cohort_v2_policy_baselines(
        downstream / "policy_baselines",
        candidate[3],
        candidate[4],
        spec,
        trajectory_label_artifact_identity=trajectory_receipt.label_artifact_identity,
        derivation_index_identity=baseline_manifest["derivation_index_identity"],
        implementation_revision=baseline_manifest["implementation_revision"],
    )
    controller_manifest = json.loads(
        (downstream / "controllers/manifest.json").read_bytes()
    )
    validate_cohort_v2_controllers(
        downstream / "controllers",
        readers,
        candidate[3],
        candidate[4],
        labels,
        spec,
        trajectory_label_artifact_identity=trajectory_receipt.label_artifact_identity,
        baseline_artifact_identity=baseline_receipt.baseline_artifact_identity,
        derivation_index_identity=controller_manifest["derivation_index_identity"],
        implementation_revision=controller_manifest["implementation_revision"],
    )
    validate_cohort_v2_controller_aggregation(downstream / "aggregation")
    evidence = validate_integrated_evidence(output / "evidence")
    compact = json.loads(paths["compact"].read_bytes())
    expected_compact_bindings = _compact_source_bindings(
        evidence,
        implementation_revision=compact["implementation_commit"],
        release_identity=compact["release_identity"],
    )
    if (
        compact.get("artifact_identity") != evidence["artifact_identity"]
        or compact.get("source_bindings") != expected_compact_bindings
    ):
        raise CohortV2IntegratedError("compact report differs from full evidence")
    print(
        f"[validate] exact source-bound validation passed artifact={evidence['artifact_identity']}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _frozen_design(args)
    if args.log_every <= 0 or args.score_log_every <= 0:
        parser.error("progress intervals must be positive")
    root = args.repository_root.resolve()
    paths = _paths(args, root)
    implementation = args.implementation_commit
    if implementation is None:
        implementation, dirty = git_revision(str(root))
        if dirty and not args.dry_run and not args.validate:
            parser.error("a dirty worktree requires --implementation-commit")
    args.implementation_commit = implementation
    print("[design] frozen issue-58 candidate, ablations, seeds, and schedule", flush=True)
    print(
        f"[design] candidate={CANDIDATE_ID} comparators={COMPARATOR_IDS} "
        f"variants={tuple(str(value) for value in IntegratedVariant)}",
        flush=True,
    )
    print("[load] validating public training/calibration/model-selection readers", flush=True)
    readers = _readers(root, paths["release"])
    reliability_manifest_raw = json.loads(
        (paths["reliability"] / "manifest.json").read_bytes()
    )
    reliability_config = CohortV2ReliabilityConfig(**reliability_manifest_raw["config"])
    reliability_estimator, reliability_manifest = (
        load_cohort_v2_reliability_estimator(
            paths["reliability"],
            readers=readers,
            config=reliability_config,
            device="cpu",
        )
    )
    print(
        "[load] issue-12 reliability artifact="
        f"{_short_identity(reliability_manifest['artifact_identity'])}",
        flush=True,
    )
    if args.dry_run:
        return _dry_run(args, readers, reliability_estimator, reliability_config)
    if args.validate:
        return _validate(args, paths, readers, reliability_manifest)
    return _production(
        args,
        paths,
        readers,
        reliability_estimator,
        reliability_config,
        reliability_manifest,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CohortV2IntegratedError as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(2) from error
