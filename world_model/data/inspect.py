"""Read-only machine-readable health reporting for rollout catalogs."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from scripts.rollout_artifacts import (
    EpisodeSummary,
    EpisodeValidationMode,
    validate_rollout_episode,
)
from world_model.data.catalog_errors import RequiredCapabilityError
from world_model.data.catalog_plan import episode_contract_from_manifest
from world_model.data.types import (
    LEGACY_RGB_V1,
    PHYSICS_CAPTURE_V1,
    CaptureContractDescriptor,
    ContractValueError,
    TemporalWindowRequest,
)


class InspectionError(ValueError):
    """Raised when a requested report cannot be produced honestly."""


class SplitReport(TypedDict):
    accepted_episodes: int
    rejected_episodes: int
    accepted_shots: int
    window_count: int
    windows_feasible: bool
    rejection_counts: dict[str, int]
    frame_length_histogram: dict[str, int]
    novelty_level_composition: "CompositionReport"
    scenario_type_composition: "CompositionReport"


class CompositionReport(TypedDict):
    status: Literal["unavailable"]
    counts: dict[str, int]


class CaptureContractReport(TypedDict):
    contract_name: str
    contract_version: str
    declared_capabilities: list[str]
    supported_capabilities: list[str]


class TemporalRequestReport(TypedDict):
    prediction_steps: int
    stride_frames: int
    horizon_frames: int


class InspectionReport(TypedDict):
    root: str
    capture_contract: CaptureContractReport
    temporal_request: TemporalRequestReport | None
    splits: dict[str, SplitReport]
    physics_coverage: NotRequired[dict]


def _contract_for_name(name: str) -> CaptureContractDescriptor:
    if name == LEGACY_RGB_V1.contract_name:
        return LEGACY_RGB_V1
    if name == PHYSICS_CAPTURE_V1.contract_name:
        return PHYSICS_CAPTURE_V1
    raise InspectionError(f"unsupported capture contract: {name}")


def _split_report(root: Path, split: str, request: TemporalWindowRequest | None) -> SplitReport:
    lengths = Counter[str]()
    shots = 0
    windows = 0
    rejection_counts = Counter[str]()
    split_root = root / split
    candidates = tuple(sorted(
        child
        for child in split_root.iterdir()
        if not child.is_symlink() and child.is_dir()
    )) if split_root.is_dir() else ()
    validation_results = ()
    if candidates:
        with ThreadPoolExecutor(max_workers=min(32, len(candidates))) as executor:
            contracts = tuple(executor.map(episode_contract_from_manifest, candidates))
            summary_validator = partial(
                validate_rollout_episode,
                mode=EpisodeValidationMode.CANONICAL_SUMMARY,
            )
            validation_results = tuple(executor.map(summary_validator, candidates, contracts))
    accepted_episodes = 0
    for result in validation_results:
        if not isinstance(result, EpisodeSummary):
            rejection_counts[str(result.code)] += 1
            continue
        accepted_episodes += 1
        for shot in result.episode.shots:
            shots += 1
            length = shot.frame_count
            lengths[str(length)] += 1
            if request is not None:
                windows += max(0, length - request.horizon_frames)
    return SplitReport(
        accepted_episodes=accepted_episodes,
        rejected_episodes=sum(rejection_counts.values()),
        accepted_shots=shots,
        window_count=windows,
        windows_feasible=windows > 0 if request is not None else accepted_episodes > 0,
        rejection_counts=dict(sorted(rejection_counts.items())),
        frame_length_histogram=dict(sorted(lengths.items())),
        novelty_level_composition={"status": "unavailable", "counts": {}},
        scenario_type_composition={"status": "unavailable", "counts": {}},
    )


def inspect_root(
    root: Path,
    splits: Sequence[str],
    capture_contract: CaptureContractDescriptor,
    required_capabilities: Sequence[str] = (),
    request: TemporalWindowRequest | None = None,
    include_physics_coverage: bool = False,
) -> InspectionReport:
    """Build a read-only report from immutable catalog snapshots."""
    if not splits:
        raise InspectionError("at least one split is required")
    declared = set(capture_contract.declared_capabilities)
    for capability in required_capabilities:
        if capability not in declared:
            raise RequiredCapabilityError(
                capability=capability,
                contract_name=capture_contract.contract_name,
            )
    reports: dict[str, SplitReport] = {}
    for split in splits:
        reports[split] = _split_report(root, split, request)
    report: InspectionReport = {
        "root": str(root),
        "capture_contract": {
            "contract_name": capture_contract.contract_name,
            "contract_version": capture_contract.contract_version,
            "declared_capabilities": list(capture_contract.declared_capabilities),
            "supported_capabilities": list(LEGACY_RGB_V1.declared_capabilities),
        },
        "temporal_request": None if request is None else {
            "prediction_steps": request.prediction_steps,
            "stride_frames": request.stride_frames,
            "horizon_frames": request.horizon_frames,
        },
        "splits": reports,
    }
    if include_physics_coverage:
        if capture_contract.contract_name != PHYSICS_CAPTURE_V1.contract_name:
            raise InspectionError("physics coverage requires the physics_capture_v1 contract")
        from world_model.data.physics_health import physics_coverage_report

        report["physics_coverage"] = dict(physics_coverage_report(root, splits))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only rollout dataset health report")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", required=True, choices=("train", "dev", "test"))
    parser.add_argument("--capture-contract", default="legacy_rgb_v1")
    parser.add_argument("--required-capability", action="append", default=[])
    parser.add_argument("--prediction-steps", type=int)
    parser.add_argument("--stride-frames", type=int)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--physics-coverage", action="store_true", help="Include physics capture, derived-label, oracle-gate, and regime-coverage health")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the inspection CLI without modifying the raw rollout root."""
    args = _parser().parse_args(argv)
    try:
        contract = _contract_for_name(args.capture_contract)
        if (args.prediction_steps is None) != (args.stride_frames is None):
            raise InspectionError("prediction steps and stride frames must be supplied together")
        request = None if args.prediction_steps is None else TemporalWindowRequest(args.prediction_steps, args.stride_frames)
        report = inspect_root(args.root, args.splits, contract, args.required_capability, request, args.physics_coverage)
        if any(
            item["accepted_episodes"] == 0
            for item in report["splits"].values()
        ):
            raise InspectionError("requested split has zero accepted episodes")
    except (ContractValueError, InspectionError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True) if args.json else json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
