"""Generate issue #8 trajectory-optimal labels from accepted pair costs."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Final

import torch

from scripts.run_cohort_v2_pair_measurements import (
    DEFAULT_CHECKPOINT,
    DEFAULT_EVALUATION,
    DEFAULT_RELEASE,
    ISSUE_6_PREDICTOR_CONFIG,
    _readers,
    issue_7_compute_calibration,
)
from world_model.training import (
    CohortV2ComputeCalibration,
    CohortV2ExecutionProfile,
    CohortV2TrajectoryCostSpec,
    generate_cohort_v2_myopic_ablation_labels,
    generate_cohort_v2_trajectory_labels,
    load_cohort_v2_evaluation,
    measure_cohort_v2_evaluation,
    validate_cohort_v2_measurements,
    validate_cohort_v2_trajectory_labels,
    write_cohort_v2_trajectory_labels,
)
from world_model.training.grid_artifacts import canonical_json_bytes


DEFAULT_MEASUREMENTS: Final = Path(".local-artifacts/issue-7-pair-measurements")
DEFAULT_OUTPUT: Final = Path(".local-artifacts/issue-8-trajectory-labels")


def issue_8_cost_spec(
    calibration: CohortV2ComputeCalibration,
) -> CohortV2TrajectoryCostSpec:
    """Return the declared primary controller-teacher objective.

    The accepted continuous transition cost is the reference unit. Unit physical
    and compute weights are the primary setting; later frontier work can declare
    additional immutable specs without changing these labels.
    """
    return CohortV2TrajectoryCostSpec(
        physical_violation_weight=1.0,
        compute_weight=1.0,
        compute_reference=calibration.transition_per_decision,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--measurements-root", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--compact-report", type=Path)
    parser.add_argument("--implementation-commit")
    return parser


def _source_inputs(
    repository_root: Path,
    release_root: Path,
    evaluation_root: Path,
    checkpoint_path: Path,
    measurements_root: Path,
):
    readers = _readers(repository_root, release_root)
    evaluation_manifest = __import__("json").loads(
        (evaluation_root / "manifest.json").read_bytes()
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("checkpoint_identity")
        != evaluation_manifest.get("checkpoint_identity")
        or checkpoint.get("predictor_config_identity")
        != ISSUE_6_PREDICTOR_CONFIG.identity
        or sorted(checkpoint.get("capabilities", ()))
        != evaluation_manifest.get("checkpoint_capabilities")
    ):
        raise ValueError("issue #6 checkpoint metadata differs from its evaluation")
    evaluation = load_cohort_v2_evaluation(
        evaluation_root,
        readers=readers,
        checkpoint_identity=evaluation_manifest["checkpoint_identity"],
        checkpoint_capabilities=frozenset(
            evaluation_manifest["checkpoint_capabilities"]
        ),
        objective_identity=evaluation_manifest["objective_identity"],
    )
    calibration = issue_7_compute_calibration()
    profile = CohortV2ExecutionProfile(
        controller_executed=False,
        shared_perception_executed=True,
    )
    validate_cohort_v2_measurements(
        measurements_root,
        evaluation,
        readers=readers,
        calibration=calibration,
        profile=profile,
    )
    measurement = measure_cohort_v2_evaluation(
        evaluation, readers, calibration, profile
    )
    return evaluation, measurement, calibration


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.compact_report is not None and not args.implementation_commit:
        parser.error("--compact-report requires --implementation-commit")
    repository_root = args.repository_root.resolve()
    release_root = (repository_root / args.release_root).resolve()
    evaluation_root = (repository_root / args.evaluation_root).resolve()
    checkpoint_path = (repository_root / args.checkpoint).resolve()
    measurements_root = (repository_root / args.measurements_root).resolve()
    output = (repository_root / args.output).resolve()

    evaluation, measurement, calibration = _source_inputs(
        repository_root,
        release_root,
        evaluation_root,
        checkpoint_path,
        measurements_root,
    )
    spec = issue_8_cost_spec(calibration)
    trajectory = generate_cohort_v2_trajectory_labels(
        evaluation, measurement, spec
    )
    myopic = generate_cohort_v2_myopic_ablation_labels(
        evaluation, measurement, spec
    )
    difference_count = sum(
        left.selected_pair != right.selected_pair
        for left, right in zip(trajectory.labels, myopic.labels, strict=True)
    )
    print(
        f"[label] states={len(trajectory.labels)} "
        f"trajectory_myopic_differences={difference_count}",
        flush=True,
    )
    if args.dry_run:
        print("[dry-run] no files written", flush=True)
        return 0

    if args.validate:
        receipt = validate_cohort_v2_trajectory_labels(
            output, evaluation, measurement, spec
        )
    else:
        receipt = write_cohort_v2_trajectory_labels(
            output, evaluation, measurement, spec
        )
    print(
        f"[complete] labels={receipt.label_count} output={output}",
        flush=True,
    )
    if args.compact_report is not None:
        selected = Counter(label.selected_pair.identity for label in trajectory.labels)
        report = {
            "artifact_type": "cohort_v2_trajectory_label_summary",
            "checkpoint_identity": evaluation.checkpoint_identity,
            "cost_spec": asdict(spec),
            "cost_spec_identity": spec.identity,
            "evaluation_identity": evaluation.identity,
            "implementation_commit": args.implementation_commit,
            "label_artifact_identity": receipt.label_artifact_identity,
            "label_count": receipt.label_count,
            "measurement_identity": measurement.identity,
            "myopic_ablation_generated_separately": True,
            "objective_identity": evaluation.objective_identity,
            "records_identity": receipt.records_identity,
            "release_identity": evaluation.release_identity,
            "rerun_commands": [
                "python -u -m scripts.run_cohort_v2_trajectory_labels --dry-run",
                "python -u -m scripts.run_cohort_v2_trajectory_labels",
                "python -u -m scripts.run_cohort_v2_trajectory_labels --validate",
            ],
            "schema": "cohort_v2_trajectory_label_summary_v1",
            "selected_pair_counts": {
                f"h{horizon}/{abstraction}": count
                for (horizon, abstraction), count in sorted(selected.items())
            },
            "source_bound_validation": "passed",
            "teacher": trajectory.teacher,
            "trajectory_myopic_difference_count": difference_count,
        }
        report_path = (repository_root / args.compact_report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(canonical_json_bytes(report))
        print(f"[report] {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
