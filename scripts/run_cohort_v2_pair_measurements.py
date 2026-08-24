"""Measure issue #7 plausibility and compute from the accepted issue #6 scores."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Final

import torch

from world_model.data import CohortV2ReleaseReader
from world_model.model import ABSTRACTION_ORDER, PredictorConfig
from world_model.training import (
    COHORT_V2_HORIZONS,
    CohortV2ComputeCalibration,
    CohortV2ExecutionProfile,
    PHYSICAL_VIOLATION_ENDPOINT_QUANTITIES,
    load_cohort_v2_evaluation,
    measure_cohort_v2_evaluation,
    validate_cohort_v2_measurements,
    write_cohort_v2_measurements,
)
from world_model.training.grid_artifacts import canonical_json_bytes


DEFAULT_RELEASE: Final = Path("data/runtime_evidence/issue-53-mixed-termination-v5")
DEFAULT_EVALUATION: Final = Path(
    ".local-artifacts/issue-6-macro-experiment/pair_evaluation"
)
DEFAULT_CHECKPOINT: Final = Path(".local-artifacts/issue-6-macro-experiment/checkpoint.pt")
DEFAULT_OUTPUT: Final = Path(".local-artifacts/issue-7-pair-measurements")
ROLE_INFLUENCE: Final = (
    ("training", "learned_parameters"),
    ("calibration", "threshold_values"),
    ("model_selection", "configuration_selection"),
)
ISSUE_6_PREDICTOR_CONFIG: Final = PredictorConfig(
    latent_dim=192,
    hidden_dim=384,
    depth=3,
    event_type_count=2,
)


def issue_7_compute_calibration() -> CohortV2ComputeCalibration:
    """Return the declared MAC accounting for the accepted issue #6 predictor."""
    config = ISSUE_6_PREDICTOR_CONFIG
    delta_features = 2 * config.delta_frequency_count
    fused_width = 2 * config.pair_code_dim
    conditioner = (
        (delta_features + config.pair_code_dim) * fused_width
        + fused_width * config.pair_code_dim
    )
    transition = (
        conditioner
        + (config.latent_dim + config.action_dim) * config.hidden_dim
        + config.depth
        * (
            config.pair_code_dim * 2 * config.hidden_dim
            + 2 * config.hidden_dim * config.hidden_dim
        )
        + config.hidden_dim * config.latent_dim
    )
    return CohortV2ComputeCalibration(
        authority=(
            "issue-7-declared-mac-accounting-v1:"
            + ISSUE_6_PREDICTOR_CONFIG.identity
        ),
        unit="multiply_accumulate",
        controller_per_decision=0.0,
        continuous_adapter_per_decision=0.0,
        micro_adapter_per_decision=2 * config.hidden_dim,
        macro_adapter_per_decision=4 * config.hidden_dim,
        micro_graph_base_per_decision=0.0,
        micro_graph_per_entity=0.0,
        micro_graph_per_contact=config.hidden_dim * config.hidden_dim,
        micro_graph_per_support=2 * config.hidden_dim * config.hidden_dim,
        transition_per_decision=transition,
        continuous_readout_per_decision=0.0,
        micro_readout_per_decision=(
            config.latent_dim * config.hidden_dim
            + config.hidden_dim * config.micro_predicate_count
        ),
        macro_readout_per_decision=(
            config.latent_dim * config.hidden_dim
            + config.hidden_dim
            * (config.macro_predicate_count + 1 + config.event_type_count)
        ),
        # The accepted oracle state codec has no learned multiply-accumulates.
        shared_initial_perception_per_rollout=0.0,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--compact-report", type=Path)
    parser.add_argument("--implementation-commit")
    return parser


def _readers(repository_root: Path, release_root: Path):
    declaration = repository_root / "docs/data_contracts/cohort_v2_capabilities_v1.json"
    production_plan = repository_root / "data/runtime_evidence/issue-53-plan-v5"
    readers = []
    for role, influence in ROLE_INFLUENCE:
        readers.append(CohortV2ReleaseReader(
            release_root,
            capability_declaration_path=declaration,
            production_plan_root=production_plan,
            workflow_kind=role,
            influence=influence,
        ))
    return tuple(readers)


def _measurement_summary(measured) -> dict[str, object]:
    modes = tuple(str(mode) for mode in ABSTRACTION_ORDER)
    availability = {
        mode: sum(
            outcome.compute is not None
            for state in measured.states
            for outcome in state.outcomes
            if str(outcome.pair.abstraction) == mode
        )
        for mode in modes
    }
    matched = {mode: [] for mode in modes}
    for state in measured.states:
        for horizon in COHORT_V2_HORIZONS:
            outcomes = {
                str(outcome.pair.abstraction): outcome
                for outcome in state.outcomes
                if outcome.pair.delta == horizon and outcome.compute is not None
            }
            if set(outcomes) != set(modes):
                continue
            targets = {
                (
                    outcome.effective_horizon,
                    outcome.target_frame_record_identity,
                )
                for outcome in outcomes.values()
            }
            plausibility = {
                outcome.endpoint_plausibility for outcome in outcomes.values()
            }
            if len(targets) != 1 or len(plausibility) != 1:
                raise ValueError("compared modes do not share one declared endpoint")
            for mode in modes:
                matched[mode].append(outcomes[mode])
    if not matched["macro"]:
        raise ValueError("issue #7 has no three-mode endpoint comparison scope")

    by_mode = {}
    for mode in modes:
        outcomes = matched[mode]
        available_values = sum(
            outcome.endpoint_plausibility.available_value_count for outcome in outcomes
        )
        unavailable_values = sum(
            outcome.endpoint_plausibility.unavailable_value_count for outcome in outcomes
        )
        violations = sum(
            outcome.endpoint_plausibility.violation_count for outcome in outcomes
        )
        by_mode[mode] = {
            "endpoint_available_value_count": available_values,
            "endpoint_unavailable_value_count": unavailable_values,
            "endpoint_violation_count": violations,
            "endpoint_violation_rate": (
                None if available_values == 0 else violations / available_values
            ),
            "mean_full_end_to_end_per_simulated_frame": sum(
                outcome.compute.full_end_to_end_per_simulated_frame
                for outcome in outcomes
            ) / len(outcomes),
            "mean_policy_dependent_per_simulated_frame": sum(
                outcome.compute.policy_dependent_per_simulated_frame
                for outcome in outcomes
            ) / len(outcomes),
        }
    return {
        "availability_by_mode": availability,
        "comparable_endpoint_scope": {
            "group_count": len(matched["macro"]),
            "matching_key": [
                "state_id",
                "requested_horizon",
                "effective_horizon",
                "target_frame_record_identity",
            ],
            "mode_summary": by_mode,
        },
    }


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.compact_report is not None and not args.implementation_commit:
        parser.error("--compact-report requires --implementation-commit")
    repository_root = args.repository_root.resolve()
    release_root = (repository_root / args.release_root).resolve()
    evaluation_root = (repository_root / args.evaluation_root).resolve()
    checkpoint_path = (repository_root / args.checkpoint).resolve()
    output = (repository_root / args.output).resolve()

    readers = _readers(repository_root, release_root)
    evaluation_manifest = json.loads((evaluation_root / "manifest.json").read_bytes())
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
    measured = measure_cohort_v2_evaluation(
        evaluation, readers, calibration, profile
    )
    mode_summary = _measurement_summary(measured)
    print(
        f"[measure] states={len(measured.states)} "
        f"outcomes={sum(len(state.outcomes) for state in measured.states)}",
        flush=True,
    )
    if args.dry_run:
        print("[dry-run] no files written", flush=True)
        return 0

    if args.validate:
        receipt = validate_cohort_v2_measurements(
            output,
            evaluation,
            readers=readers,
            calibration=calibration,
            profile=profile,
        )
    else:
        receipt = write_cohort_v2_measurements(
            output,
            evaluation,
            readers=readers,
            calibration=calibration,
            profile=profile,
        )
    print(
        f"[complete] available={receipt.available_count} "
        f"unavailable={receipt.unavailable_count} output={output}",
        flush=True,
    )
    if args.compact_report is not None:
        report = {
            "artifact_type": "cohort_v2_pair_measurement_summary",
            "available_count": receipt.available_count,
            "compute_calibration": asdict(calibration),
            "compute_calibration_identity": receipt.compute_calibration_identity,
            "endpoint_quantities": list(PHYSICAL_VIOLATION_ENDPOINT_QUANTITIES),
            "endpoint_measurement_kind": "source_endpoint_violation_incidence",
            "evaluation_identity": receipt.evaluation_identity,
            "execution_profile": profile.canonical,
            "execution_profile_identity": receipt.execution_profile_identity,
            "implementation_commit": args.implementation_commit,
            "measurement_identity": receipt.measurement_identity,
            "mode_summary": mode_summary,
            "model_objectives_rerun": False,
            "outcome_count": receipt.outcome_count,
            "records_identity": receipt.records_identity,
            "rerun_commands": [
                "python -u -m scripts.run_cohort_v2_pair_measurements --dry-run",
                "python -u -m scripts.run_cohort_v2_pair_measurements",
                "python -u -m scripts.run_cohort_v2_pair_measurements --validate",
            ],
            "schema": "cohort_v2_pair_measurement_summary_v1",
            "source_bound_validation": "passed",
            "state_count": receipt.state_count,
            "unavailable_count": receipt.unavailable_count,
        }
        report_path = (repository_root / args.compact_report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(canonical_json_bytes(report))
        print(f"[report] {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
