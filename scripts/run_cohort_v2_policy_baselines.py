"""Generate issue #9 policy baselines from accepted issue #7/#8 evidence."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Final

from scripts.run_cohort_v2_pair_measurements import (
    DEFAULT_CHECKPOINT,
    DEFAULT_EVALUATION,
    DEFAULT_RELEASE,
)
from scripts.run_cohort_v2_trajectory_labels import (
    DEFAULT_MEASUREMENTS,
    DEFAULT_OUTPUT as DEFAULT_TRAJECTORY_LABELS,
    _source_inputs,
    issue_8_cost_spec,
)
from world_model.training import (
    generate_cohort_v2_policy_baselines,
    git_revision,
    validate_cohort_v2_policy_baselines,
    validate_cohort_v2_trajectory_labels,
    write_cohort_v2_policy_baselines,
)
from world_model.training.grid_artifacts import canonical_json_bytes


DEFAULT_OUTPUT: Final = Path(".local-artifacts/issue-9-policy-baselines")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--measurements-root", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument(
        "--trajectory-labels-root", type=Path, default=DEFAULT_TRAJECTORY_LABELS
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--compact-report", type=Path)
    parser.add_argument("--implementation-commit")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    release_root = (repository_root / args.release_root).resolve()
    evaluation_root = (repository_root / args.evaluation_root).resolve()
    checkpoint_path = (repository_root / args.checkpoint).resolve()
    measurements_root = (repository_root / args.measurements_root).resolve()
    trajectory_root = (repository_root / args.trajectory_labels_root).resolve()
    output = (repository_root / args.output).resolve()
    implementation_revision = args.implementation_commit
    if implementation_revision is None:
        implementation_revision, dirty = git_revision(str(repository_root))
        if dirty and not args.dry_run:
            parser.error(
                "a dirty worktree requires an explicit --implementation-commit"
            )

    print("[load] validating issue #7 pair evidence", flush=True)
    evaluation, measurement, calibration, readers = _source_inputs(
        repository_root,
        release_root,
        evaluation_root,
        checkpoint_path,
        measurements_root,
    )
    spec = issue_8_cost_spec(calibration)
    trajectory_manifest = json.loads((trajectory_root / "manifest.json").read_bytes())
    print("[load] validating issue #8 trajectory labels", flush=True)
    trajectory_receipt = validate_cohort_v2_trajectory_labels(
        trajectory_root,
        evaluation,
        measurement,
        spec,
        implementation_revision=trajectory_manifest["implementation_revision"],
    )
    derivations = {reader.derivation_identity for reader in readers}
    if len(derivations) != 1:
        raise ValueError("public exposure roles do not share one derivation index")
    derivation_identity = next(iter(derivations))
    result = generate_cohort_v2_policy_baselines(
        evaluation,
        measurement,
        spec,
        trajectory_label_artifact_identity=(
            trajectory_receipt.label_artifact_identity
        ),
        derivation_index_identity=derivation_identity,
    )
    comparison_state_count = len(result.decisions) // 4
    print(
        f"[baseline] comparison_states={comparison_state_count} "
        f"decisions={len(result.decisions)}",
        flush=True,
    )
    print(
        "[select] "
        + " ".join(
            f"{key}={value}"
            for key, value in result.selected_configurations.items()
            if key != "configuration_selection_role"
        ),
        flush=True,
    )
    if args.dry_run:
        print("[dry-run] no files written", flush=True)
        return 0

    kwargs = {
        "trajectory_label_artifact_identity": (
            trajectory_receipt.label_artifact_identity
        ),
        "derivation_index_identity": derivation_identity,
        "implementation_revision": implementation_revision,
    }
    if args.validate:
        receipt = validate_cohort_v2_policy_baselines(
            output, evaluation, measurement, spec, **kwargs
        )
    else:
        receipt = write_cohort_v2_policy_baselines(
            output, evaluation, measurement, spec, **kwargs
        )
    print(
        f"[complete] comparison_states={receipt.comparison_state_count} "
        f"decisions={receipt.decision_count} output={output}",
        flush=True,
    )
    if args.compact_report is not None:
        report = {
            "artifact_type": "cohort_v2_policy_baseline_summary",
            "baseline_artifact_identity": receipt.baseline_artifact_identity,
            "comparison_state_count": receipt.comparison_state_count,
            "comparison_state_set_identity": receipt.comparison_state_set_identity,
            "decision_count": receipt.decision_count,
            "derivation_index_identity": derivation_identity,
            "evaluation_identity": evaluation.identity,
            "implementation_commit": implementation_revision,
            "measurement_identity": measurement.identity,
            "policies": [score.policy_id for score in result.scores[:4]],
            "frontiers": {
                role: list(policy_ids)
                for role, policy_ids in result.frontiers.items()
            },
            "release_identity": evaluation.release_identity,
            "rerun_commands": [
                "python -u -m scripts.run_cohort_v2_policy_baselines --dry-run",
                "python -u -m scripts.run_cohort_v2_policy_baselines "
                f"--implementation-commit {implementation_revision}",
                "python -u -m scripts.run_cohort_v2_policy_baselines --validate "
                f"--implementation-commit {implementation_revision}",
            ],
            "schema": "cohort_v2_policy_baseline_summary_v1",
            "scores": [asdict(score) for score in result.scores],
            "selected_configurations": result.selected_configurations,
            "source_bound_validation": "passed",
            "trajectory_label_artifact_identity": (
                trajectory_receipt.label_artifact_identity
            ),
        }
        report_path = (repository_root / args.compact_report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(canonical_json_bytes(report))
        print(f"[report] {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
