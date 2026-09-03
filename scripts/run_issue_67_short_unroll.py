"""Run issue #67's short-unrolled h15 deployment-carrier training."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Final

import torch

from world_model.data.deployment_temporal import TemporalVisualCarrierAdapter
from world_model.model import PredictorConfig
from world_model.training.lineage_scaling import (
    CarrierKind,
    CarrierLineage,
    ContinuousTransitionExample,
    LineageScalingError,
    load_carrier_lineage_bundle,
)
from world_model.training.short_unroll import (
    ShortUnrollTrainingReport,
    ShortUnrollTrainingSpec,
    build_short_unroll_windows,
    evaluate_recursive_carrier,
    load_short_unroll_checkpoint,
    save_short_unroll_checkpoint,
    train_short_unroll_predictor,
)


REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_ISSUE_63_OUTPUT: Final = (
    REPOSITORY_ROOT / ".local-artifacts/issue-63-matched-experiment-v1"
)
DEFAULT_OUTPUT: Final = (
    REPOSITORY_ROOT / ".local-artifacts/issue-67-short-unroll-h15-v1"
)
DEFAULT_TRAINING_BUNDLE: Final = (
    DEFAULT_ISSUE_63_OUTPUT / "carrier-bundles/training-deployment.pt"
)
DEFAULT_CALIBRATION_BUNDLE: Final = (
    DEFAULT_ISSUE_63_OUTPUT / "carrier-bundles/calibration-deployment.pt"
)
DEFAULT_SUMMARY: Final = (
    REPOSITORY_ROOT / "data/runtime_evidence/issue-67/short-unroll-summary-v1.json"
)
TRAINING_SEEDS: Final = (20260901, 20260902, 20260903)
CONFIGURATIONS: Final = (
    ("teacher-forced-h15", 1, 1.0, 0.0, 0.0),
    ("self-conditioned-h15-u2", 2, 1.0, 1.0, 0.01),
    ("self-conditioned-h15-u4", 4, 1.0, 1.0, 0.01),
)


def _log(message: str) -> None:
    print(f"[issue-67] {message}", flush=True)


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise LineageScalingError(f"cannot read issue-67 artifact: {path}") from error
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise LineageScalingError(f"issue-67 artifact is not canonical: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise LineageScalingError(f"issue-67 artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        raise LineageScalingError(f"stale issue-67 temporary exists: {temporary}")
    temporary.write_bytes(_canonical_bytes(value))
    os.replace(temporary, target)


def _predictor_config() -> PredictorConfig:
    return PredictorConfig(latent_dim=197, action_dim=5, hidden_dim=384, depth=3)


def _specs(
    *,
    optimizer_example_budget: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    carrier_bound: float,
    lineage_manifest_reference: str,
    predictor_config: PredictorConfig | None = None,
    seeds: tuple[int, ...] = TRAINING_SEEDS,
) -> tuple[ShortUnrollTrainingSpec, ...]:
    config = _predictor_config() if predictor_config is None else predictor_config
    return tuple(
        ShortUnrollTrainingSpec(
            name=name,
            unroll_steps=steps,
            local_loss_weight=local_weight,
            unrolled_loss_weight=unrolled_weight,
            carrier_bound=carrier_bound,
            carrier_bound_loss_weight=bound_weight,
            optimizer_example_budget=optimizer_example_budget,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            seed=seed,
            carrier_identity=TemporalVisualCarrierAdapter.identity,
            lineage_manifest_reference=lineage_manifest_reference,
            predictor_config=config,
        )
        for name, steps, local_weight, unrolled_weight, bound_weight in CONFIGURATIONS
        for seed in seeds
    )


def _spec_from_payload(raw: dict[str, Any]) -> ShortUnrollTrainingSpec:
    value = dict(raw)
    value["predictor_config"] = PredictorConfig(**value["predictor_config"])
    return ShortUnrollTrainingSpec(**value)


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    training_bundle = str(args.training_bundle.resolve())
    specs = _specs(
        optimizer_example_budget=args.optimizer_example_budget,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        carrier_bound=args.carrier_bound,
        lineage_manifest_reference=training_bundle,
    )
    return {
        "schema": "issue_67_short_unroll_plan_v1",
        "identity": "issue-67-short-unroll-plan-v1:full-deployment-h15-u1-u2-u4",
        "training_bundle": training_bundle,
        "calibration_bundle": str(args.calibration_bundle.resolve()),
        "configuration_count": len(specs),
        "configurations": [asdict(spec) for spec in specs],
        "common_optimizer_example_budget": args.optimizer_example_budget,
        "optimizer_budget_basis": (
            "twice issue-63's 4,000,000-example budget; exceeds its approximately "
            "2.5 effective full-scale epochs"
        ),
        "h15_deployment_primary": True,
        "h1_diagnostic_only": True,
        "prediction_clamping": False,
        "final_evaluation_opened": False,
    }


def _load_plan(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], tuple[ShortUnrollTrainingSpec, ...]]:
    path = args.output / "plan.json"
    plan = _read_json(path)
    if (
        plan.get("schema") != "issue_67_short_unroll_plan_v1"
        or plan.get("identity")
        != "issue-67-short-unroll-plan-v1:full-deployment-h15-u1-u2-u4"
        or plan.get("final_evaluation_opened") is not False
    ):
        raise LineageScalingError("issue-67 plan is invalid")
    specs = tuple(_spec_from_payload(raw) for raw in plan["configurations"])
    if len(specs) != 9 or tuple(spec.seed for spec in specs) != tuple(
        seed for _configuration in CONFIGURATIONS for seed in TRAINING_SEEDS
    ):
        raise LineageScalingError("issue-67 plan cell inventory differs")
    return plan, specs


def _prepare(args: argparse.Namespace) -> int:
    plan = _plan(args)
    path = args.output / "plan.json"
    if path.exists():
        if _read_json(path) != plan:
            raise LineageScalingError("existing issue-67 plan differs")
        _log(f"validated existing plan configurations={plan['configuration_count']}")
    else:
        _write_json(path, plan)
        _log(
            f"plan frozen configurations={plan['configuration_count']} "
            f"example_budget={plan['common_optimizer_example_budget']}"
        )
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    return 0


def _load_bundle_with_heartbeat(path: Path) -> tuple[CarrierLineage, ...]:
    started = time.monotonic()
    _log(f"loading legacy carrier bundle path={path}")
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(load_carrier_lineage_bundle, path)
        while True:
            try:
                lineages = future.result(timeout=10.0)
                break
            except TimeoutError:
                _log(
                    f"loading legacy carrier bundle elapsed_seconds="
                    f"{time.monotonic() - started:.0f}"
                )
    _log(
        f"legacy carrier bundle loaded lineages={len(lineages)} "
        f"elapsed_seconds={time.monotonic() - started:.1f}"
    )
    return lineages


def _compact_lineages(
    lineages: tuple[CarrierLineage, ...],
) -> tuple[CarrierLineage, ...]:
    """Drop #63's recursively expanded identity strings while retaining tensors."""

    role = lineages[0].exposure_role
    carrier_identity = lineages[0].carrier_identity
    result = []
    for lineage_ordinal, lineage in enumerate(lineages, start=1):
        transitions = tuple(
            ContinuousTransitionExample(
                identity=(
                    f"issue-67-{role}-l{lineage_ordinal:04d}-"
                    f"d{transition.decision_index}-h{transition.horizon}"
                ),
                context=transition.context,
                action=transition.action,
                target=transition.target,
                physical_diagnostics={},
                decision_index=transition.decision_index,
                horizon=transition.horizon,
                target_decision_index=transition.target_decision_index,
            )
            for transition in lineage.transitions
        )
        result.append(CarrierLineage(
            trajectory_identity=f"issue-67-{role}-trajectory-{lineage_ordinal:04d}",
            scenario_lineage_identity=f"issue-67-{role}-lineage-{lineage_ordinal:04d}",
            exposure_role=role,
            source_release_identity="issue-62-successor-cohort-v4",
            carrier=CarrierKind.DEPLOYMENT,
            carrier_identity=carrier_identity,
            transitions=transitions,
            complete=True,
            decision_count=lineage.decision_count,
            segment_end_positions=lineage.segment_ends,
        ))
        if lineage_ordinal % 100 == 0 or lineage_ordinal == len(lineages):
            _log(
                f"compacting legacy carrier bundle lineages="
                f"{lineage_ordinal}/{len(lineages)}"
            )
    compact = tuple(result)
    _log(
        f"legacy recursive identities replaced with ordinal labels "
        f"lineages={len(compact)}"
    )
    return compact


def _checkpoint_path(output: Path, spec: ShortUnrollTrainingSpec) -> Path:
    return output / "checkpoints" / spec.name / f"seed-{spec.seed}.pt"


def _training_report_path(output: Path, spec: ShortUnrollTrainingSpec) -> Path:
    return output / "checkpoints" / spec.name / f"seed-{spec.seed}.training.json"


def _training_payload(report: ShortUnrollTrainingReport) -> dict[str, Any]:
    values = asdict(report)
    values["failures"] = list(report.failures)
    return {
        "schema": "issue_67_short_unroll_training_report_v1",
        **values,
        "final_evaluation_opened": False,
    }


def _validate_existing_cell(
    output: Path, spec: ShortUnrollTrainingSpec
) -> bool:
    checkpoint_path = _checkpoint_path(output, spec)
    if not checkpoint_path.exists():
        return False
    _model, report = load_short_unroll_checkpoint(
        checkpoint_path, spec, device="cpu"
    )
    expected = _training_payload(report)
    report_path = _training_report_path(output, spec)
    if report_path.exists():
        if _read_json(report_path) != expected:
            raise LineageScalingError("existing issue-67 training report differs")
    else:
        _write_json(report_path, expected)
    return True


def _train(args: argparse.Namespace) -> int:
    plan, specs = _load_plan(args)
    pending = []
    for index, spec in enumerate(specs, start=1):
        if _validate_existing_cell(args.output, spec):
            _log(f"train cell={index}/{len(specs)} validated existing")
        else:
            pending.append((index, spec))
    if not pending:
        _log("training matrix already complete")
        return 0

    raw_lineages = _load_bundle_with_heartbeat(Path(plan["training_bundle"]))
    lineages = _compact_lineages(raw_lineages)
    del raw_lineages
    gc.collect()
    _log(
        f"legacy bundle memory released peak_rss_mib="
        f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0:.1f}"
    )
    for index, spec in pending:
        windows = build_short_unroll_windows(
            lineages, unroll_steps=spec.unroll_steps
        )
        _log(
            f"train cell={index}/{len(specs)} configuration={spec.name} "
            f"seed={spec.seed} windows={len(windows)} device={args.device}"
        )
        model, report = train_short_unroll_predictor(
            spec, lineages, device=args.device, progress=_log
        )
        checkpoint_path = _checkpoint_path(args.output, spec)
        temporary = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        save_short_unroll_checkpoint(temporary, model, report)
        os.replace(temporary, checkpoint_path)
        _write_json(_training_report_path(args.output, spec), _training_payload(report))
        _log(
            f"trained cell={index}/{len(specs)} configuration={spec.name} "
            f"seed={spec.seed} examples={report.optimizer_examples} "
            f"effective_epochs={report.effective_epochs:.3f} "
            f"seconds={report.wall_seconds:.1f}"
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _log(f"training matrix complete checkpoints={len(specs)}")
    return 0


def _score_path(output: Path, spec: ShortUnrollTrainingSpec) -> Path:
    return output / "calibration" / spec.name / f"seed-{spec.seed}.score.json"


def _score_calibration(args: argparse.Namespace) -> int:
    plan, specs = _load_plan(args)
    raw_lineages = _load_bundle_with_heartbeat(Path(plan["calibration_bundle"]))
    lineages = _compact_lineages(raw_lineages)
    del raw_lineages
    gc.collect()
    for index, spec in enumerate(specs, start=1):
        path = _score_path(args.output, spec)
        if path.exists():
            existing = _read_json(path)
            if (
                existing.get("schema")
                != "issue_67_recursive_calibration_score_v1"
                or existing.get("checkpoint_identity") != spec.checkpoint_identity
                or existing.get("final_evaluation_opened") is not False
            ):
                raise LineageScalingError("existing issue-67 calibration score differs")
            _log(f"score cell={index}/{len(specs)} validated existing")
            continue
        _log(
            f"score cell={index}/{len(specs)} configuration={spec.name} "
            f"seed={spec.seed} device={args.device}"
        )
        model, training = load_short_unroll_checkpoint(
            _checkpoint_path(args.output, spec), spec, device=args.device
        )
        h1 = evaluate_recursive_carrier(
            model,
            lineages,
            horizon=1,
            carrier_bound=spec.carrier_bound,
            progress=_log,
        )
        h15 = evaluate_recursive_carrier(
            model,
            lineages,
            horizon=15,
            carrier_bound=spec.carrier_bound,
            progress=_log,
        )
        payload = {
            "schema": "issue_67_recursive_calibration_score_v1",
            "checkpoint_identity": spec.checkpoint_identity,
            "configuration": asdict(spec),
            "training_compute": asdict(training),
            "h1_diagnostic": asdict(h1),
            "h15_deployment_primary": asdict(h15),
            "prediction_clamping": False,
            "final_evaluation_opened": False,
        }
        _write_json(path, payload)
        _log(
            f"scored cell={index}/{len(specs)} h15_auc={h15.error_auc} "
            f"h15_bound_excess={h15.mean_absolute_carrier_bound_excess}"
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


def _validate(args: argparse.Namespace) -> int:
    plan, specs = _load_plan(args)
    records = []
    for index, spec in enumerate(specs, start=1):
        _log(
            f"validate checkpoint={index}/{len(specs)} "
            f"configuration={spec.name} seed={spec.seed}"
        )
        _model, report = load_short_unroll_checkpoint(
            _checkpoint_path(args.output, spec), spec, device="cpu"
        )
        if _read_json(
            _training_report_path(args.output, spec)
        ) != _training_payload(report):
            raise LineageScalingError("issue-67 training report validation failed")
        records.append({
            "checkpoint_identity": spec.checkpoint_identity,
            "configuration": spec.name,
            "seed": spec.seed,
            "optimizer_examples": report.optimizer_examples,
            "optimizer_steps": report.optimizer_steps,
            "effective_epochs": report.effective_epochs,
            "device": report.device,
        })
    result = {
        "schema": "issue_67_short_unroll_validation_v1",
        "plan_identity": plan["identity"],
        "validated_checkpoint_count": len(records),
        "checkpoints": records,
        "existing_issue_63_artifacts_modified": False,
        "final_evaluation_opened": False,
    }
    if args.summary.exists():
        if _read_json(args.summary) != result:
            raise LineageScalingError("existing issue-67 validation summary differs")
    else:
        _write_json(args.summary, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


def _fixture_lineages(role: str, *, count: int) -> tuple[CarrierLineage, ...]:
    action = torch.tensor((-0.2, 0.1, 0.6, 0.0, 1.0), dtype=torch.float32)
    result = []
    for lineage_ordinal in range(1, count + 1):
        states = tuple(
            torch.tensor(
                (
                    lineage_ordinal / 10.0 + position / 200.0,
                    math.sin(position / 20.0),
                    math.cos(position / 20.0),
                    position / 100.0,
                ),
                dtype=torch.float32,
            )
            for position in range(61)
        )
        transitions = tuple(
            ContinuousTransitionExample(
                identity=f"dry-{role}-l{lineage_ordinal}-d{position}-h{horizon}",
                context=states[position],
                action=action,
                target=states[min(position + horizon, 60)],
                physical_diagnostics={},
                decision_index=position,
                horizon=horizon,
                target_decision_index=min(position + horizon, 60),
            )
            for horizon in (1, 15)
            for position in range(0, 60, horizon)
        )
        result.append(CarrierLineage(
            trajectory_identity=f"dry-{role}-trajectory-{lineage_ordinal}",
            scenario_lineage_identity=f"dry-{role}-lineage-{lineage_ordinal}",
            exposure_role=role,
            source_release_identity="issue-67-dry-run-release",
            carrier=CarrierKind.DEPLOYMENT,
            carrier_identity=TemporalVisualCarrierAdapter.identity,
            transitions=transitions,
            complete=True,
            decision_count=60,
            segment_end_positions=(60,),
        ))
    return tuple(result)


def _dry_run(_args: argparse.Namespace) -> int:
    training = _fixture_lineages("training", count=2)
    calibration = _fixture_lineages("calibration", count=1)
    predictor = PredictorConfig(
        latent_dim=4,
        action_dim=5,
        hidden_dim=8,
        depth=1,
        pair_code_dim=4,
        delta_frequency_count=2,
    )
    specs = _specs(
        optimizer_example_budget=4,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        carrier_bound=2.0,
        lineage_manifest_reference="issue-67-dry-run-training-manifest-v1",
        predictor_config=predictor,
        seeds=(67,),
    )
    reports = []
    for index, spec in enumerate(specs, start=1):
        windows = build_short_unroll_windows(
            training, unroll_steps=spec.unroll_steps
        )
        _log(
            f"dry-run validate configuration={index}/{len(specs)} "
            f"name={spec.name} windows={len(windows)}"
        )
        model, training_report = train_short_unroll_predictor(
            spec, training, device="cpu", progress=_log
        )
        h1 = evaluate_recursive_carrier(
            model,
            calibration,
            horizon=1,
            carrier_bound=spec.carrier_bound,
            progress=_log,
        )
        h15 = evaluate_recursive_carrier(
            model,
            calibration,
            horizon=15,
            carrier_bound=spec.carrier_bound,
            progress=_log,
        )
        _log(
            f"dry-run validated configuration={index}/{len(specs)} "
            f"h1_steps={len(h1.step_curve)} h15_steps={len(h15.step_curve)}"
        )
        reports.append({
            "training": asdict(training_report),
            "h1_diagnostic": asdict(h1),
            "h15_deployment_primary": asdict(h15),
        })
    result = {
        "schema": "issue_67_short_unroll_dry_run_v1",
        "configuration_count": len(specs),
        "reports": reports,
        "complete_non_final_training_lineages": len(training),
        "complete_non_final_calibration_lineages": len(calibration),
        "self_conditioned_unrolls": True,
        "prediction_clamping": False,
        "files_written": False,
        "unity_accessed": False,
        "final_evaluation_opened": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--score-calibration", action="store_true")
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-bundle", type=Path, default=DEFAULT_TRAINING_BUNDLE)
    parser.add_argument(
        "--calibration-bundle", type=Path, default=DEFAULT_CALIBRATION_BUNDLE
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--optimizer-example-budget", type=int, default=8_000_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--carrier-bound", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    args = _parser().parse_args(argv)
    args.output = args.output.resolve()
    args.training_bundle = args.training_bundle.resolve()
    args.calibration_bundle = args.calibration_bundle.resolve()
    args.summary = args.summary.resolve()
    if args.dry_run:
        return _dry_run(args)
    if args.prepare:
        return _prepare(args)
    if args.train:
        return _train(args)
    if args.score_calibration:
        return _score_calibration(args)
    return _validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
