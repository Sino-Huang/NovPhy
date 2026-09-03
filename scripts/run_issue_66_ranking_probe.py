"""Run the issue-66 broad-probe and deterministic h15 ensemble diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
from typing import Any, Final

import torch
from torch.nn import functional as F

from world_model.data.successor_cohort import ACTION_BOUNDS
from world_model.model import DualOutputPredictor, PredictorConfig
from world_model.planning import SlingshotAction, SlingshotActionBounds
from world_model.training.action_ranking_probe import (
    BROAD_ACTION_DESIGN_ID,
    ActionRankingProbeError,
    EnsembleCheckpointBinding,
    aggregate_ensemble_ranking,
    broad_action_candidates,
    calibrate_disagreement_penalty,
    score_action_ranking_member,
    summarize_ranking_diversity,
    validate_ensemble_checkpoint_bindings,
)
from world_model.training.lineage_scaling import (
    ActionCandidate,
    ActionRankingState,
    CarrierKind,
    LineageScalingProtocol,
    TrainingCell,
    load_action_ranking_bundle,
    load_lineage_scaling_protocol,
)


REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_ISSUE_63_OUTPUT: Final = (
    REPOSITORY_ROOT / ".local-artifacts/issue-63-matched-experiment-v1"
)


def _log(message: str) -> None:
    print(f"[issue-66] {message}", flush=True)


def _bounds() -> SlingshotActionBounds:
    return SlingshotActionBounds(
        tuple(ACTION_BOUNDS["drag_x"]),
        tuple(ACTION_BOUNDS["drag_y"]),
        tuple(ACTION_BOUNDS["tap_time_ms"]),
        int(ACTION_BOUNDS["release_time_ms"]),
    )


def _action_tensor(
    action: SlingshotAction,
    bounds: SlingshotActionBounds,
    frame_height: int,
) -> torch.Tensor:
    return torch.tensor((
        action.drag_x / float(frame_height),
        action.drag_y / float(frame_height),
        bounds.release_time_ms / 1000.0,
        action.tap_time_ms / 1000.0,
        1.0,
    ), dtype=torch.float32)


def _checkpoint_path(output: Path, cell: TrainingCell) -> Path:
    return (
        output
        / "checkpoints"
        / cell.scale_name
        / cell.carrier.value
        / f"seed-{cell.seed}.pt"
    )


def _expected_full_deployment_cells(
    protocol: LineageScalingProtocol,
) -> tuple[TrainingCell, ...]:
    cells = tuple(
        cell
        for cell in protocol.cells
        if cell.scale_name == "full" and cell.carrier is CarrierKind.DEPLOYMENT
    )
    if len(cells) != 3:
        raise ActionRankingProbeError(
            "protocol does not contain three full deployment checkpoints"
        )
    return cells


def _load_checkpoint_without_rehashing(
    path: Path,
    protocol: LineageScalingProtocol,
    expected_cell: TrainingCell,
    *,
    device: str,
) -> tuple[DualOutputPredictor, EnsembleCheckpointBinding]:
    """Load already-validated #63 weights without recomputing their 2 GB digest."""

    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if payload["schema"] != "lineage_scaled_continuous_checkpoint_v1":
            raise KeyError("schema")
        raw = payload["metadata"]
        cell = TrainingCell(
            str(raw["cell"]["scale_name"]),
            CarrierKind(raw["cell"]["carrier"]),
            int(raw["cell"]["seed"]),
        )
        config = PredictorConfig(**raw["predictor_config"])
        available_horizons = tuple(
            int(horizon) for horizon, _count in raw["available_horizon_counts"]
        )
        if (
            cell != expected_cell
            or raw["protocol_identity"] != protocol.identity
            or raw["carrier_identity"] != protocol.carrier_identity(cell.carrier)
            or config != protocol.predictor_config
            or 15 not in available_horizons
        ):
            raise ActionRankingProbeError(
                "checkpoint differs from its matched full deployment cell"
            )
        model = DualOutputPredictor(config)
        model.load_state_dict(payload["model_state"], strict=True)
        binding = EnsembleCheckpointBinding(
            checkpoint_identity=str(raw["identity"]),
            protocol_identity=str(raw["protocol_identity"]),
            carrier_identity=str(raw["carrier_identity"]),
            scale_name=cell.scale_name,
            carrier=cell.carrier.value,
            seed=cell.seed,
            available_horizons=available_horizons,
        )
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        if isinstance(error, ActionRankingProbeError):
            raise
        raise ActionRankingProbeError(
            f"cannot load deterministic ensemble checkpoint: {path}: {error}"
        ) from error
    return model.to(torch.device(device)), binding


def _diversity_payload(states: tuple[ActionRankingState, ...]) -> dict[str, Any]:
    report = summarize_ranking_diversity(states)
    tie_histogram = Counter(report.best_action_tie_sizes)
    return {
        **asdict(report),
        "best_action_tie_size_counts": {
            str(size): count for size, count in sorted(tie_histogram.items())
        },
    }


def _evaluation_payload(evaluation: Any) -> dict[str, Any]:
    payload = asdict(evaluation)
    for state in payload["states"]:
        state.pop("candidate_set_identity")
    payload["candidate_set_match_validated"] = True
    return payload


def _design_payload() -> dict[str, Any]:
    candidates = broad_action_candidates("source-state", _bounds())
    return {
        "identity": BROAD_ACTION_DESIGN_ID,
        "source_bound": True,
        "outcome_independent_membership": True,
        "candidate_count": len(candidates),
        "strata": [
            {
                "ordinal": item.ordinal,
                "action_stratum": item.action_stratum,
                "drag_x": item.action.drag_x,
                "drag_y": item.action.drag_y,
                "tap_time_ms": item.action.tap_time_ms,
            }
            for item in candidates
        ],
    }


def _ranking_cost(
    state: ActionRankingState,
    _candidate: ActionCandidate,
    predicted: torch.Tensor,
) -> float:
    if state.cost_target is None:
        raise ActionRankingProbeError("ranking state has no frozen cost target")
    return float(F.mse_loss(predicted, state.cost_target))


def _diagnose_existing(args: argparse.Namespace) -> int:
    output = args.issue_63_output.resolve()
    protocol = load_lineage_scaling_protocol(output / "protocol.json")
    role_states = {
        role: load_action_ranking_bundle(
            output / "ranking-bundles" / f"{role}-deployment.pt"
        )
        for role in ("calibration", "model_selection")
    }
    if args.state_limit is not None:
        role_states = {
            role: states[: args.state_limit] for role, states in role_states.items()
        }
    _log("population diversity is reported before model comparison")
    diversity = {}
    for role, states in role_states.items():
        diversity[role] = _diversity_payload(states)
        _log(
            f"diversity role={role} states={len(states)} "
            f"candidates={sum(len(state.candidates) for state in states)} "
            f"all_tied={diversity[role]['all_tied_state_count']} "
            f"pig_discordant="
            f"{diversity[role]['pig_removal_discordant_state_count']} "
            f"failures={diversity[role]['candidate_failure_count']}"
        )
    expected_cells = _expected_full_deployment_cells(protocol)
    expected_paths = {
        _checkpoint_path(output, cell).resolve(): cell for cell in expected_cells
    }
    supplied = tuple(path.resolve() for path in args.checkpoints)
    if len(set(supplied)) != 3 or set(supplied) != set(expected_paths):
        raise ActionRankingProbeError(
            "explicit checkpoints must be the three #63 full/deployment seeds"
        )
    predictions: dict[str, list[Any]] = {
        "calibration": [],
        "model_selection": [],
    }
    for model_index, path in enumerate(supplied, start=1):
        cell = expected_paths[path]
        _log(
            f"load model={model_index}/3 seed={cell.seed} path={path} "
            "integrity=existing-exact-validation no_rehash=true"
        )
        model, binding = _load_checkpoint_without_rehashing(
            path, protocol, cell, device=args.device
        )
        for role, states in role_states.items():
            _log(
                f"score model={model_index}/3 seed={cell.seed} role={role} "
                f"states={len(states)}"
            )
            predictions[role].append(score_action_ranking_member(
                model,
                binding,
                states,
                predicted_cost=_ranking_cost,
                progress=_log,
            ))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for role in predictions:
        validate_ensemble_checkpoint_bindings(
            tuple(item.checkpoint for item in predictions[role])
        )
    calibration = calibrate_disagreement_penalty(
        role_states["calibration"], tuple(predictions["calibration"])
    )
    _log(
        "calibration selected disagreement_penalty="
        f"{calibration.selected_penalty}"
    )
    evaluations = {
        role: aggregate_ensemble_ranking(
            states,
            tuple(predictions[role]),
            disagreement_penalty=calibration.selected_penalty,
        )
        for role, states in role_states.items()
    }
    report = {
        "schema": "issue_66_deterministic_ensemble_diagnostic_v1",
        "method": "deterministic_seed_ensemble_pets_lite_diagnostic",
        "evidence_status": "exploratory_only",
        "source_evidence": "issue_63_opened_non_final_roles",
        "may_support_advancement": False,
        "candidate_design_for_issue_68": _design_payload(),
        "evaluated_candidate_design": "issue_63_local_five_candidate_probe",
        "population_diversity": diversity,
        "disagreement_calibration": asdict(calibration),
        "ranking": {
            role: _evaluation_payload(evaluation)
            for role, evaluation in evaluations.items()
        },
        "candidate_evaluator_seam": {
            "class": "DeterministicEnsembleCandidateEvaluator",
            "cem_changed": False,
            "member_total_cost_retains_physical_and_rollout_penalties": True,
        },
        "live_collection_audit_contract": {
            "required": True,
            "root": "data/issue-68-ranking-audit",
            "artifacts": "candidate WebM videos plus frame gallery",
        },
        "files_written": False,
        "unity_accessed": False,
        "final_evaluation_opened": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


class _DryRunPredictor(torch.nn.Module):
    def __init__(self, seed_offset: float) -> None:
        super().__init__()
        self.device_anchor = torch.nn.Parameter(torch.zeros(()))
        self.seed_offset = seed_offset

    def carrier(
        self,
        carrier: torch.Tensor,
        action: torch.Tensor,
        _pair: Any,
    ) -> torch.Tensor:
        return carrier + action[:, :1] * 0.01 + self.seed_offset


def _dry_states() -> tuple[ActionRankingState, ...]:
    bounds = _bounds()
    states = []
    for state_index in range(2):
        broad = broad_action_candidates(f"dry-state-{state_index + 1}", bounds)
        candidates = tuple(
            ActionCandidate(
                item.identity,
                _action_tensor(item.action, bounds, 480),
                float(
                    (1000 + item.ordinal)
                    if state_index == 0
                    else ((item.ordinal % 2) * 1000 + item.ordinal)
                ),
                item.action,
            )
            for item in broad
        )
        states.append(ActionRankingState(
            identity=f"dry-state-{state_index + 1}",
            scenario_lineage_identity=f"dry-lineage-{state_index + 1}",
            trajectory_identity=f"dry-trajectory-{state_index + 1}",
            decision_transition_identity=f"dry-transition-{state_index + 1}",
            exposure_role="calibration",
            carrier=CarrierKind.DEPLOYMENT,
            carrier_identity="dry-deployment-carrier",
            context=torch.zeros(197),
            action_bounds=bounds,
            frame_height=480,
            candidates=candidates,
            cost_target=torch.zeros(197),
        ))
    return tuple(states)


def _dry_run() -> int:
    states = _dry_states()
    predictions = []
    for model_index, offset in enumerate((0.0, 0.001, -0.001), start=1):
        binding = EnsembleCheckpointBinding(
            checkpoint_identity=f"dry-checkpoint-{model_index}",
            protocol_identity="dry-protocol",
            carrier_identity="dry-deployment-carrier",
            scale_name="full",
            carrier="deployment",
            seed=model_index,
            available_horizons=(1, 15),
        )
        predictions.append(score_action_ranking_member(
            _DryRunPredictor(offset),
            binding,
            states,
            recursive_steps=2,
            predicted_cost=_ranking_cost,
            progress=_log,
        ))
    calibration = calibrate_disagreement_penalty(states, tuple(predictions))
    evaluation = aggregate_ensemble_ranking(
        states,
        tuple(reversed(predictions)),
        disagreement_penalty=calibration.selected_penalty,
    )
    report = {
        "schema": "issue_66_no_write_dry_run_v1",
        "passed": evaluation.evaluated_state_count == len(states),
        "candidate_design": _design_payload(),
        "diversity": _diversity_payload(states),
        "ensemble": _evaluation_payload(evaluation),
        "files_written": False,
        "unity_accessed": False,
        "final_evaluation_opened": False,
        "actual_command": (
            "python -u -m scripts.run_issue_66_ranking_probe "
            "--diagnose-existing "
            "--checkpoint .local-artifacts/issue-63-matched-experiment-v1/"
            "checkpoints/full/deployment/seed-20260901.pt "
            "--checkpoint .local-artifacts/issue-63-matched-experiment-v1/"
            "checkpoints/full/deployment/seed-20260902.pt "
            "--checkpoint .local-artifacts/issue-63-matched-experiment-v1/"
            "checkpoints/full/deployment/seed-20260903.pt "
            "2>&1 | tee -a data/issue-66-exploratory-diagnostic.log"
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["passed"]:
        raise ActionRankingProbeError("issue-66 dry run did not evaluate every state")
    _log("dry-run passed files_written=0 unity=unopened final_evaluation=unopened")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--diagnose-existing", action="store_true")
    parser.add_argument(
        "--issue-63-output", type=Path, default=DEFAULT_ISSUE_63_OUTPUT
    )
    parser.add_argument(
        "--checkpoint", dest="checkpoints", action="append", type=Path
    )
    parser.add_argument("--state-limit", type=int)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    args = _parser().parse_args(argv)
    if args.state_limit is not None and args.state_limit <= 0:
        raise ActionRankingProbeError("state limit must be positive")
    if args.dry_run:
        return _dry_run()
    if args.checkpoints is None or len(args.checkpoints) != 3:
        raise ActionRankingProbeError(
            "--diagnose-existing requires exactly three --checkpoint paths"
        )
    return _diagnose_existing(args)


if __name__ == "__main__":
    raise SystemExit(main())
