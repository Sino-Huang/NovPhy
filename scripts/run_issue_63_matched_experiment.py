"""Run issue #63's frozen carrier-alignment x training-coverage experiment."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any, Final

import torch
from torch.nn import functional as F

from scripts.cohort_v2_migration_recovery import DEFAULT_MANIFEST
from scripts.run_cohort_v2_macro_experiment import DEFAULT_RELEASE, _readers
from scripts.run_issue_61_lineage_scaling import (
    DEFAULT_ALIGNED,
    DEFAULT_VISUAL_PARSER,
)
from scripts.run_issue_62_successor_cohort import (
    DEFAULT_RELEASE as DEFAULT_SUCCESSOR_RELEASE,
    STAGE_ROOT,
    _collect_lineage_attempt,
)
from scripts.smoke_physics_capture import (
    archive_details,
    start_display,
    terminate,
)
from world_model.data import CohortV2AlignedObservationReader
from world_model.data.deployment_temporal import (
    AgentObservation,
    TemporalObservationContext,
    TrajectoryLineageBinding,
    TrajectoryLineageManifest,
)
from world_model.data.successor_cohort import (
    ACTION_BOUNDS,
    PUBLIC_ROLES,
    RELEASE_SCHEMA,
    _load_shot,
    release_identity_for_plan,
    validate_successor_plan,
)
from world_model.model import DualOutputPredictor, PredictorConfig, identity
from world_model.planning import SlingshotAction, SlingshotActionBounds
from world_model.planning.gameplay import VisualPlanningObservationAdapter
from world_model.training.cohort_v2_micro import CohortV2StateCodec
from world_model.training.cohort_v2_visual_parser import (
    load_visual_parser_checkpoint,
)
from world_model.training.lineage_scaling import (
    ActionCandidate,
    ActionRankingState,
    CarrierKind,
    CarrierLineage,
    ContinuousTransitionExample,
    FrozenLineageScale,
    FrozenRankingState,
    LineageScalingError,
    LineageScalingProtocol,
    TrainingCell,
    evaluate_action_ranking,
    evaluate_continuous_prediction,
    load_action_ranking_bundle,
    load_carrier_lineage_bundle,
    load_lineage_scaled_checkpoint,
    load_lineage_scaling_protocol,
    save_action_ranking_bundle,
    save_carrier_lineage_bundle,
    save_lineage_scaled_checkpoint,
    save_lineage_scaling_protocol,
    train_continuous_predictor,
    validate_matched_action_ranking_states,
    validate_matched_carrier_lineages,
    validate_evaluation_lineages,
)
from world_model.training.manifest import git_revision


REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT: Final = (
    REPOSITORY_ROOT / ".local-artifacts/issue-63-matched-experiment-v1"
)
DEFAULT_SUMMARY: Final = (
    REPOSITORY_ROOT
    / "data/runtime_evidence/issue-63/matched-carrier-scaling-summary-v1.json"
)
TRAINING_SCALE_COUNTS: Final = (6, 200, 1000, 3000)
TRAINING_SEEDS: Final = (20260901, 20260902, 20260903)
RANKING_STATES_PER_ROLE: Final = 24
OPTIMIZER_EXAMPLE_BUDGET: Final = 4_000_000
PARSER_BATCH_SIZE: Final = 128
RANKING_RECURSIVE_H15_STEPS: Final = 15
PRIMARY_MARGIN: Final = 0.05
PHYSICAL_REGRESSION_MARGIN: Final = 0.01
BOOTSTRAP_RESAMPLES: Final = 10_000
BOOTSTRAP_SEED: Final = 20260904


def _log(message: str) -> None:
    print(f"[issue-63] {message}", flush=True)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
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


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise LineageScalingError(f"cannot load {label}: {path}") from error
    if not isinstance(value, dict):
        raise LineageScalingError(f"{label} must be an object")
    if raw != _canonical_bytes(value):
        raise LineageScalingError(f"{label} is not canonical: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise LineageScalingError(f"immutable output already exists: {path}")
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        stream.write(_canonical_bytes(value))
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _design_identity(design: Mapping[str, Any]) -> str:
    return identity(
        (
            "issue-63-matched-experiment-design-v1",
            design["protocol_identity"],
            design["source_release_identity"],
            design["implementation_revision"],
            tuple(design["training_scale_counts"]),
            tuple(design["training_seeds"]),
            tuple(
                (
                    spec["identity"],
                    spec["legal_candidate_set_identity"],
                )
                for spec in design["ranking_states"]
            ),
            tuple(sorted(design["decision_rule"].items())),
        )
    )


def _decision_freeze_identity(freeze: Mapping[str, Any]) -> str:
    return identity(
        (
            "issue-63-calibration-decision-freeze-v1",
            freeze["design_identity"],
            tuple(
                sorted(
                    (key, tuple(sorted(value.items())))
                    for key, value in freeze["calibration_summary"].items()
                )
            ),
        )
    )


def _results_identity(results: Mapping[str, Any]) -> str:
    selected = results["selected_deployment_configuration"]
    return identity(
        (
            "issue-63-matched-experiment-results-v1",
            results["design_identity"],
            results["decision_freeze_identity"],
            results["analysis"]["decision"],
            None if selected is None else tuple(sorted(selected.items())),
        )
    )


def _cell_score_identity(score: Mapping[str, Any]) -> str:
    return identity(
        (
            "issue-63-matched-cell-score-v1",
            {key: value for key, value in score.items() if key != "identity"},
        )
    )


def _paths(output: Path) -> dict[str, Path]:
    root = Path(output).resolve()
    return {
        "root": root,
        "protocol": root / "protocol.json",
        "design": root / "design.json",
        "bundles": root / "carrier-bundles",
        "ranking": root / "ranking-bundles",
        "checkpoints": root / "checkpoints",
        "calibration": root / "scores/calibration",
        "model_selection": root / "scores/model-selection",
        "decision_freeze": root / "decision-freeze.json",
        "evidence": root / "evidence",
        "results": root / "evidence/results.json",
    }


def _bundle_path(paths: Mapping[str, Path], role: str, carrier: CarrierKind) -> Path:
    return paths["bundles"] / f"{role}-{carrier.value}.pt"


def _ranking_path(paths: Mapping[str, Path], role: str, carrier: CarrierKind) -> Path:
    return paths["ranking"] / f"{role}-{carrier.value}.pt"


def _checkpoint_path(paths: Mapping[str, Path], cell: TrainingCell) -> Path:
    return (
        paths["checkpoints"]
        / cell.scale_name
        / cell.carrier.value
        / f"seed-{cell.seed}.pt"
    )


def _training_report_path(
    paths: Mapping[str, Path], cell: TrainingCell
) -> Path:
    return _checkpoint_path(paths, cell).with_suffix(".training.json")


def _score_path(paths: Mapping[str, Path], role: str, cell: TrainingCell) -> Path:
    parent = (
        paths["calibration"] if role == "calibration" else paths["model_selection"]
    )
    return parent / cell.scale_name / cell.carrier.value / f"seed-{cell.seed}.json"


def _bounds() -> SlingshotActionBounds:
    return SlingshotActionBounds(
        drag_x=tuple(ACTION_BOUNDS["drag_x"]),
        drag_y=tuple(ACTION_BOUNDS["drag_y"]),
        tap_time_ms=tuple(ACTION_BOUNDS["tap_time_ms"]),
        release_time_ms=int(ACTION_BOUNDS["release_time_ms"]),
    )


def _action_tensor(action: Mapping[str, Any]) -> torch.Tensor:
    engine = action["engine_relative_action"]
    drag = engine["drag_delta_canvas_pixels"]
    return torch.tensor(
        (
            float(drag[0]) / 480.0,
            float(drag[1]) / 480.0,
            float(engine["hold_milliseconds"]) / 1000.0,
            float(engine["tap_time_milliseconds"]) / 1000.0,
            1.0,
        ),
        dtype=torch.float32,
    )


def _physical_diagnostics(labels: Mapping[str, Any]) -> dict[str, float | bool | None]:
    result: dict[str, float | bool | None] = {}
    for name, raw in labels.items():
        if isinstance(raw, Mapping) and type(raw.get("value")) in (
            bool,
            int,
            float,
            type(None),
        ):
            result[name] = raw.get("value")
    return result


def _predicted_physical_diagnostics(carrier: torch.Tensor) -> dict[str, float]:
    return {
        "carrier_bound_excess": float(
            torch.relu(carrier.abs() - 2.0).pow(2).mean()
        )
    }


def _release_inventory(
    release_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, tuple[dict[str, Any], ...]]]:
    root = Path(release_root).resolve()
    manifest = _load_json(root / "manifest.json", "issue-62 release manifest")
    plan = validate_successor_plan(
        _load_json(root / "production-plan.json", "issue-62 production plan")
    )
    expected_identity = release_identity_for_plan(plan["identity"])
    if (
        manifest.get("schema") != RELEASE_SCHEMA
        or manifest.get("identity") != expected_identity
        or manifest.get("production_plan_identity") != plan["identity"]
        or manifest.get("included_roles") != list(PUBLIC_ROLES)
        or manifest.get("final_evaluation_collected") is not False
        or manifest.get("passed") is not True
    ):
        raise LineageScalingError("issue-62 release boundary differs")
    by_slot = {
        item["slot_identity"]: item for item in manifest.get("trajectories", ())
    }
    roles: dict[str, tuple[dict[str, Any], ...]] = {}
    for role in PUBLIC_ROLES:
        records = tuple(
            by_slot[slot["slot_identity"]]
            for slot in plan["lineages"]
            if slot["exposure_role"] == role
        )
        if (
            len(records) != plan["role_counts"][role]
            or any(record["exposure_role"] != role for record in records)
        ):
            raise LineageScalingError(f"issue-62 {role} inventory differs")
        roles[role] = records
    return manifest, plan, roles


def _load_deployment_adapter(
    *, device: str
) -> VisualPlanningObservationAdapter:
    source_readers = _readers(
        REPOSITORY_ROOT,
        DEFAULT_RELEASE,
        migration_recovery_authority=DEFAULT_MANIFEST,
    )
    aligned_readers = tuple(
        CohortV2AlignedObservationReader(DEFAULT_ALIGNED, source_reader=reader)
        for reader in source_readers
    )
    model, checkpoint, _manifest = load_visual_parser_checkpoint(
        DEFAULT_VISUAL_PARSER,
        readers=aligned_readers,
        device=device,
    )
    model.eval()
    return VisualPlanningObservationAdapter(
        model,
        parser_checkpoint_identity=checkpoint.identity,
        temperatures=checkpoint.temperatures,
        thresholds=checkpoint.thresholds,
        object_kind_temperature=checkpoint.object_kind_temperature,
        latent_dim=197,
        max_entities=15,
    )


def _observation(shot: Mapping[str, Any], fixed_step: int) -> AgentObservation:
    metadata = shot["observations"].get(fixed_step)
    if not isinstance(metadata, Mapping):
        raise LineageScalingError("aligned observation metadata is missing")
    relative = metadata["agent_observation"]["relative_path"]
    path = shot["observation_root"] / relative
    try:
        png = path.read_bytes()
    except OSError as error:
        raise LineageScalingError(f"cannot read agent observation {path}") from error
    return AgentObservation(
        identity=str(metadata["agent_observation"]["identity"]),
        fixed_step=fixed_step,
        fixed_time_seconds=float(metadata["fixed_time_seconds"]),
        png=png,
        observation_role="agent",
    )


def _parse_observations(
    adapter: VisualPlanningObservationAdapter,
    observations: tuple[AgentObservation, ...],
) -> tuple[dict[str, torch.Tensor], ...]:
    parsed: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(observations), PARSER_BATCH_SIZE):
        parsed.extend(
            adapter.parse_batch(observations[start : start + PARSER_BATCH_SIZE])
        )
    return tuple(parsed)


def _continuous_identity(
    trajectory_identity: str,
    shot_index: int,
    start_fixed_step: int,
    target_fixed_step: int,
    horizon: int,
) -> str:
    return identity(
        (
            "issue-63-continuous-window-v1",
            trajectory_identity,
            shot_index,
            start_fixed_step,
            target_fixed_step,
            horizon,
        )
    )


def _build_lineage_pair(
    release_root: Path,
    release_identity: str,
    record: Mapping[str, Any],
    adapter: VisualPlanningObservationAdapter,
    source_codec: CohortV2StateCodec,
    *,
    maximum_intervals_per_shot: int | None = None,
    maximum_shots: int | None = None,
) -> tuple[CarrierLineage, CarrierLineage]:
    trajectory_root = Path(release_root) / str(record["path"])
    raw = _load_json(trajectory_root / "trajectory.json", "successor trajectory")
    role = str(record["exposure_role"])
    if (
        raw.get("trajectory_identity") != record.get("trajectory_identity")
        or raw.get("scenario_lineage_identity")
        != record.get("scenario_lineage_identity")
        or raw.get("release_identity") != release_identity
        or raw.get("exposure_role") != role
        or raw.get("complete") is not True
    ):
        raise LineageScalingError("successor trajectory inventory binding differs")
    source_transitions: list[ContinuousTransitionExample] = []
    deployment_transitions: list[ContinuousTransitionExample] = []
    segment_ends: list[int] = []
    position_offset = 0
    shots = raw["shots"] if maximum_shots is None else raw["shots"][:maximum_shots]
    for shot_index, raw_shot in enumerate(shots):
        shot = _load_shot(
            trajectory_root,
            raw_shot,
            release_identity=release_identity,
            role=role,
        )
        fixed_steps = tuple(sorted(shot["frames"]))
        if maximum_intervals_per_shot is not None:
            fixed_steps = fixed_steps[: maximum_intervals_per_shot + 1]
        if len(fixed_steps) < 2:
            raise LineageScalingError("successor shot has no continuous interval")
        observations = tuple(_observation(shot, step) for step in fixed_steps)
        parsed = _parse_observations(adapter, observations)
        source_carriers = tuple(
            source_codec.encode(shot["frames"][step]) for step in fixed_steps
        )
        deployment_carriers = tuple(
            adapter.build_from_parsed(
                TemporalObservationContext(
                    None if index == 0 else observations[index - 1],
                    observation,
                ),
                parsed[index],
                None if index == 0 else parsed[index - 1],
            ).tensor
            for index, observation in enumerate(observations)
        )
        action = _action_tensor(raw_shot["action"])
        interval_count = len(fixed_steps) - 1
        for horizon in (1, 15):
            for local_position in range(0, interval_count, horizon):
                target_position = min(local_position + horizon, interval_count)
                transition_identity = _continuous_identity(
                    str(raw["trajectory_identity"]),
                    shot_index,
                    fixed_steps[local_position],
                    fixed_steps[target_position],
                    horizon,
                )
                common = {
                    "identity": transition_identity,
                    "action": action,
                    "physical_diagnostics": _physical_diagnostics(
                        shot["frames"][fixed_steps[target_position]].labels
                    ),
                    "decision_index": position_offset + local_position,
                    "horizon": horizon,
                    "target_decision_index": position_offset + target_position,
                }
                source_transitions.append(
                    ContinuousTransitionExample(
                        **common,
                        context=source_carriers[local_position],
                        target=source_carriers[target_position],
                    )
                )
                deployment_transitions.append(
                    ContinuousTransitionExample(
                        **common,
                        context=deployment_carriers[local_position],
                        target=deployment_carriers[target_position],
                    )
                )
        position_offset += interval_count
        segment_ends.append(position_offset)
    common_lineage = {
        "trajectory_identity": str(raw["trajectory_identity"]),
        "scenario_lineage_identity": str(raw["scenario_lineage_identity"]),
        "exposure_role": role,
        "source_release_identity": release_identity,
        "complete": True,
        "decision_count": position_offset,
        "segment_end_positions": tuple(segment_ends),
    }
    return (
        CarrierLineage(
            **common_lineage,
            carrier=CarrierKind.SOURCE,
            carrier_identity=source_codec.identity,
            transitions=tuple(source_transitions),
        ),
        CarrierLineage(
            **common_lineage,
            carrier=CarrierKind.DEPLOYMENT,
            carrier_identity=adapter.identity,
            transitions=tuple(deployment_transitions),
        ),
    )


def _binding(lineage: CarrierLineage) -> TrajectoryLineageBinding:
    return TrajectoryLineageBinding(
        trajectory_identity=lineage.trajectory_identity,
        scenario_lineage_identity=lineage.scenario_lineage_identity,
        exposure_role=lineage.exposure_role,
        transition_identities=tuple(item.identity for item in lineage.transitions),
        initial_observation_identity=f"{lineage.trajectory_identity}:continuous-start",
        terminal_observation_identity=f"{lineage.trajectory_identity}:continuous-end",
    )


def _atomic_save_bundle(path: Path, lineages: tuple[CarrierLineage, ...]) -> None:
    if path.exists():
        raise LineageScalingError(f"carrier bundle already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    save_carrier_lineage_bundle(temporary, lineages)
    os.replace(temporary, path)


def _candidate_actions(raw_action: Mapping[str, Any]) -> tuple[SlingshotAction, ...]:
    engine = raw_action["engine_relative_action"]
    drag_x, drag_y = (int(value) for value in engine["drag_delta_canvas_pixels"])
    tap = int(engine["tap_time_milliseconds"])
    bounds = _bounds()
    proposals = (
        (drag_x, drag_y, tap),
        (drag_x - 20, drag_y, tap),
        (drag_x + 20, drag_y, tap),
        (drag_x, drag_y - 20, tap),
        (drag_x, drag_y + 20, tap),
        (drag_x, drag_y, tap - 200),
        (drag_x, drag_y, tap + 200),
        (drag_x - 40, drag_y, tap),
        (drag_x + 40, drag_y, tap),
    )
    selected: list[SlingshotAction] = []
    seen: set[tuple[int, int, int]] = set()
    for x, y, tap_time in proposals:
        action = SlingshotAction(
            min(max(x, bounds.drag_x[0]), bounds.drag_x[1]),
            min(max(y, bounds.drag_y[0]), bounds.drag_y[1]),
            min(max(tap_time, bounds.tap_time_ms[0]), bounds.tap_time_ms[1]),
        )
        key = (action.drag_x, action.drag_y, action.tap_time_ms)
        if key not in seen:
            seen.add(key)
            selected.append(action)
        if len(selected) == 5:
            break
    if len(selected) != 5:
        raise LineageScalingError("cannot freeze five distinct legal ranking actions")
    return tuple(selected)


def _ranking_spec(
    release_root: Path,
    record: Mapping[str, Any],
    binding: TrajectoryLineageBinding,
) -> dict[str, Any]:
    raw = _load_json(
        Path(release_root) / str(record["path"]) / "trajectory.json",
        "ranking source trajectory",
    )
    state_identity = identity(
        (
            "issue-63-ranking-state-v1",
            record["exposure_role"],
            record["trajectory_identity"],
            0,
        )
    )
    actions = _candidate_actions(raw["shots"][0]["action"])
    candidates = tuple(
        ActionCandidate(
            identity(
                (
                    "issue-63-ranking-candidate-v1",
                    state_identity,
                    index,
                    action.drag_x,
                    action.drag_y,
                    action.tap_time_ms,
                )
            ),
            torch.tensor(
                (
                    action.drag_x / 480.0,
                    action.drag_y / 480.0,
                    _bounds().release_time_ms / 1000.0,
                    action.tap_time_ms / 1000.0,
                    1.0,
                ),
                dtype=torch.float32,
            ),
            0.0,
            action,
        )
        for index, action in enumerate(actions)
    )
    probe = ActionRankingState(
        identity=state_identity,
        scenario_lineage_identity=str(record["scenario_lineage_identity"]),
        trajectory_identity=str(record["trajectory_identity"]),
        decision_transition_identity=binding.transition_identities[0],
        exposure_role=str(record["exposure_role"]),
        carrier=CarrierKind.SOURCE,
        carrier_identity="issue-63-legal-candidate-probe",
        context=torch.zeros(197),
        candidates=candidates,
        action_bounds=_bounds(),
        frame_height=480,
        cost_target=torch.zeros(197),
    )
    return {
        "identity": state_identity,
        "scenario_lineage_identity": record["scenario_lineage_identity"],
        "trajectory_identity": record["trajectory_identity"],
        "decision_transition_identity": binding.transition_identities[0],
        "exposure_role": record["exposure_role"],
        "release_path": record["path"],
        "shot_index": 0,
        "legal_candidate_set_identity": probe.legal_candidate_set_identity,
        "candidates": [
            {
                "identity": candidate.identity,
                "drag_x": candidate.interface_action.drag_x,
                "drag_y": candidate.interface_action.drag_y,
                "tap_time_ms": candidate.interface_action.tap_time_ms,
            }
            for candidate in candidates
        ],
    }


def _prepare(args: argparse.Namespace) -> int:
    paths = _paths(args.output)
    if paths["protocol"].exists() or paths["design"].exists():
        raise LineageScalingError("issue-63 design is already frozen")
    implementation_revision, dirty = git_revision(str(REPOSITORY_ROOT))
    if dirty:
        raise LineageScalingError(
            "issue-63 production design requires a clean implementation revision"
        )
    manifest, plan, role_records = _release_inventory(args.successor_release)
    if tuple(plan["nested_training_scale_counts"]) != TRAINING_SCALE_COUNTS:
        raise LineageScalingError("issue-62 nested training ladder differs")
    _log(
        f"prepare source={manifest['identity']} roles="
        + ",".join(f"{role}:{len(role_records[role])}" for role in PUBLIC_ROLES)
    )
    adapter = _load_deployment_adapter(device=args.device)
    source_codec = CohortV2StateCodec(latent_dim=197, max_entities=15)
    bindings_by_role: dict[str, tuple[TrajectoryLineageBinding, ...]] = {}
    for role in PUBLIC_ROLES:
        source_path = _bundle_path(paths, role, CarrierKind.SOURCE)
        deployment_path = _bundle_path(paths, role, CarrierKind.DEPLOYMENT)
        if source_path.exists() or deployment_path.exists():
            if not source_path.is_file() or not deployment_path.is_file():
                raise LineageScalingError(
                    f"prepared {role} carrier bundle pair is incomplete"
                )
            source_existing = load_carrier_lineage_bundle(source_path)
            deployment_existing = load_carrier_lineage_bundle(deployment_path)
            if (
                len(source_existing) != len(role_records[role])
                or len(deployment_existing) != len(role_records[role])
                or tuple(item.scenario_lineage_identity for item in source_existing)
                != tuple(record["scenario_lineage_identity"] for record in role_records[role])
                or tuple(item.scenario_lineage_identity for item in deployment_existing)
                != tuple(record["scenario_lineage_identity"] for record in role_records[role])
            ):
                raise LineageScalingError(f"prepared {role} bundles differ")
            bindings_by_role[role] = tuple(
                _binding(item) for item in source_existing
            )
            _log(f"prepared role={role} validated existing bundles")
            continue
        source_lineages: list[CarrierLineage] = []
        deployment_lineages: list[CarrierLineage] = []
        total = len(role_records[role])
        for index, record in enumerate(role_records[role], start=1):
            source, deployment = _build_lineage_pair(
                args.successor_release,
                str(manifest["identity"]),
                record,
                adapter,
                source_codec,
            )
            source_lineages.append(source)
            deployment_lineages.append(deployment)
            if index == total or index % 10 == 0:
                _log(
                    f"prepare role={role} lineages={index}/{total} "
                    f"windows={sum(len(item.transitions) for item in source_lineages)}"
                )
        source_tuple = tuple(source_lineages)
        deployment_tuple = tuple(deployment_lineages)
        if tuple(item.identity for item in source_tuple[0].transitions) != tuple(
            item.identity for item in deployment_tuple[0].transitions
        ):
            raise LineageScalingError("prepared carrier windows differ")
        _atomic_save_bundle(
            source_path, source_tuple
        )
        _atomic_save_bundle(
            deployment_path, deployment_tuple
        )
        bindings_by_role[role] = tuple(_binding(item) for item in source_tuple)
        _log(f"prepared role={role} bundles written")

    training_manifest = TrajectoryLineageManifest.create(
        str(manifest["identity"]), bindings_by_role["training"]
    )
    scales = tuple(
        FrozenLineageScale.from_manifest(
            "full" if count == TRAINING_SCALE_COUNTS[-1] else (
                "six" if count == 6 else str(count)
            ),
            TrajectoryLineageManifest.create(
                str(manifest["identity"]), training_manifest.bindings[:count]
            ),
        )
        for count in TRAINING_SCALE_COUNTS
    )
    ranking_specs = tuple(
        _ranking_spec(
            args.successor_release,
            record,
            bindings_by_role[role][index],
        )
        for role in ("calibration", "model_selection")
        for index, record in enumerate(
            role_records[role][:RANKING_STATES_PER_ROLE]
        )
    )
    protocol = LineageScalingProtocol(
        training_scales=scales,
        evaluation_manifests=(
            TrajectoryLineageManifest.create(
                str(manifest["identity"]), bindings_by_role["calibration"]
            ),
            TrajectoryLineageManifest.create(
                str(manifest["identity"]), bindings_by_role["model_selection"]
            ),
        ),
        ranking_states=tuple(
            FrozenRankingState(
                identity=str(spec["identity"]),
                scenario_lineage_identity=str(spec["scenario_lineage_identity"]),
                trajectory_identity=str(spec["trajectory_identity"]),
                decision_transition_identity=str(
                    spec["decision_transition_identity"]
                ),
                exposure_role=str(spec["exposure_role"]),
                legal_candidate_set_identity=str(
                    spec["legal_candidate_set_identity"]
                ),
            )
            for spec in ranking_specs
        ),
        training_seeds=TRAINING_SEEDS,
        training_horizons=(1, 15),
        optimizer_example_budget=OPTIMIZER_EXAMPLE_BUDGET,
        batch_size=512,
        learning_rate=1e-4,
        weight_decay=1e-4,
        grad_clip=1.0,
        predictor_config=PredictorConfig(
            latent_dim=197,
            action_dim=5,
            hidden_dim=384,
            depth=3,
        ),
        source_max_entities=15,
        source_carrier_identity=source_codec.identity,
        deployment_carrier_identity=adapter.identity,
        configuration_basis="prospectively_frozen",
    )
    save_lineage_scaling_protocol(paths["protocol"], protocol)
    design = {
        "schema": "issue_63_matched_experiment_design_v1",
        "protocol_identity": protocol.identity,
        "source_release_identity": manifest["identity"],
        "source_production_plan_identity": plan["identity"],
        "implementation_revision": implementation_revision,
        "training_scale_counts": list(TRAINING_SCALE_COUNTS),
        "primary_scales": ["six", "full"],
        "training_seeds": list(TRAINING_SEEDS),
        "training_horizons": [1, 15],
        "ranking_states_per_role": RANKING_STATES_PER_ROLE,
        "ranking_recursive_h15_steps": RANKING_RECURSIVE_H15_STEPS,
        "ranking_states": list(ranking_specs),
        "decision_rule": {
            "primary_metrics": [
                "recursive_h1_auc",
                "recursive_h15_auc",
                "mean_top_action_regret",
            ],
            "minimum_relative_improvement": PRIMARY_MARGIN,
            "simultaneous_interval": "paired-seed percentile bootstrap; Bonferroni 3 metrics",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "physical_regression_absolute_margin": PHYSICAL_REGRESSION_MARGIN,
            "failed_score_treatment": (
                "worst; any nonfinite, execution, or ranking failure in a full "
                "deployment cell rejects advancement"
            ),
            "advancement": (
                "supported only when full deployment exceeds six deployment on all "
                "three co-primary metrics by the margin and simultaneous lower bounds, "
                "without a physical or execution regression"
            ),
            "supported_checkpoint_selection": (
                "among the three frozen full-deployment seeds, minimize the unweighted "
                "sum of model-selection normalized recursive h1 AUC, normalized "
                "recursive h15 AUC, and mean top-action regret"
            ),
        },
        "historical_issue_15_role": "external_reference_only",
        "rerun_commands": [
            (
                "python -u -m scripts.run_issue_63_matched_experiment --prepare "
                "2>&1 | tee -a data/issue-63-prepare.log"
            ),
            (
                "python -u -m scripts.run_issue_63_matched_experiment "
                "--collect-calibration-ranking --start-display 2>&1 | tee -a "
                "data/issue-63-calibration-ranking.log"
            ),
            (
                "python -u -m scripts.run_issue_63_matched_experiment --train "
                "2>&1 | tee -a data/issue-63-train.log"
            ),
            (
                "python -u -m scripts.run_issue_63_matched_experiment "
                "--score-calibration 2>&1 | tee -a data/issue-63-calibration.log"
            ),
            (
                "python -u -m scripts.run_issue_63_matched_experiment "
                "--freeze-decision 2>&1 | tee -a data/issue-63-freeze.log"
            ),
            (
                "python -u -m scripts.run_issue_63_matched_experiment "
                "--collect-model-selection-ranking --start-display 2>&1 | "
                "tee -a data/issue-63-model-selection-ranking.log"
            ),
            (
                "python -u -m scripts.run_issue_63_matched_experiment "
                "--score-model-selection 2>&1 | tee -a "
                "data/issue-63-model-selection.log"
            ),
            (
                "python -u -m scripts.run_issue_63_matched_experiment --publish "
                "2>&1 | tee -a data/issue-63-publish.log"
            ),
            (
                "python -u -m scripts.run_issue_63_matched_experiment --validate "
                "2>&1 | tee -a data/issue-63-validate.log"
            ),
        ],
        "final_evaluation_opened": False,
    }
    design["identity"] = _design_identity(design)
    _write_json(paths["design"], design)
    _log(
        f"design frozen identity={design['identity']} protocol={protocol.identity} "
        f"cells={len(protocol.cells)} final_evaluation=unopened"
    )
    return 0


def _load_frozen(paths: Mapping[str, Path]) -> tuple[LineageScalingProtocol, dict[str, Any]]:
    protocol = load_lineage_scaling_protocol(paths["protocol"])
    design = _load_json(paths["design"], "issue-63 frozen design")
    if (
        design.get("schema") != "issue_63_matched_experiment_design_v1"
        or design.get("protocol_identity") != protocol.identity
        or design.get("identity") != _design_identity(design)
        or design.get("final_evaluation_opened") is not False
    ):
        raise LineageScalingError("issue-63 frozen design differs")
    return protocol, design


def _train(args: argparse.Namespace) -> int:
    paths = _paths(args.output)
    protocol, _design = _load_frozen(paths)
    _log("loading matched full training carrier bundles")
    source = load_carrier_lineage_bundle(
        _bundle_path(paths, "training", CarrierKind.SOURCE)
    )
    deployment = load_carrier_lineage_bundle(
        _bundle_path(paths, "training", CarrierKind.DEPLOYMENT)
    )
    alignment = validate_matched_carrier_lineages(protocol, source, deployment)
    _log(
        f"matched carriers validated lineages={alignment['lineage_count']} "
        f"windows={alignment['transition_count']}"
    )
    by_carrier = {
        CarrierKind.SOURCE: source,
        CarrierKind.DEPLOYMENT: deployment,
    }
    for index, cell in enumerate(protocol.cells, start=1):
        checkpoint_path = _checkpoint_path(paths, cell)
        if checkpoint_path.exists():
            _model, metadata = load_lineage_scaled_checkpoint(
                checkpoint_path,
                protocol,
                expected_cell=cell,
                device="cpu",
            )
            training_record = _load_json(
                _training_report_path(paths, cell), "training compute report"
            )
            if (
                training_record.get("checkpoint_identity") != metadata.identity
                or training_record.get("cell_identity") != cell.identity
                or training_record.get("final_evaluation_opened") is not False
            ):
                raise LineageScalingError("training compute report differs")
            _log(f"train cell={index}/{len(protocol.cells)} validated existing")
            continue
        scale = protocol.scale(cell.scale_name)
        membership = set(scale.lineage_identities)
        selected = tuple(
            lineage
            for lineage in by_carrier[cell.carrier]
            if lineage.scenario_lineage_identity in membership
        )
        _log(
            f"train cell={index}/{len(protocol.cells)} scale={cell.scale_name} "
            f"carrier={cell.carrier.value} seed={cell.seed}"
        )
        model, report = train_continuous_predictor(
            protocol,
            cell,
            selected,
            device=args.device,
            progress=_log,
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = save_lineage_scaled_checkpoint(checkpoint_path, model, report)
        _write_json(
            _training_report_path(paths, cell),
            {
                "schema": "issue_63_training_compute_report_v1",
                "protocol_identity": protocol.identity,
                "cell_identity": cell.identity,
                "checkpoint_identity": metadata.identity,
                "optimizer_examples": report.optimizer_examples,
                "optimizer_steps": report.optimizer_steps,
                "epochs": report.epochs,
                "final_loss": report.final_loss,
                "wall_seconds": report.wall_seconds,
                "parameter_count": report.parameter_count,
                "device": args.device,
                "final_evaluation_opened": False,
            },
        )
        _log(
            f"trained cell={index}/{len(protocol.cells)} "
            f"loss={report.final_loss:.8f} seconds={report.wall_seconds:.1f}"
        )
    _log(f"training matrix complete checkpoints={len(protocol.cells)}")
    return 0


def _realized_goal_cost(frame: Any) -> float:
    entities = frame.engine_state.get("entities")
    if not isinstance(entities, tuple):
        raise LineageScalingError("ranking outcome engine entities are missing")
    active_pigs = 0
    active_blocks = 0
    for entity in entities:
        if not isinstance(entity, Mapping) or entity.get("lifecycle") != "active":
            continue
        scenario_id = str(entity.get("scenario_object_id", ""))
        if scenario_id.startswith("pig:"):
            active_pigs += 1
        elif scenario_id.startswith("block:"):
            active_blocks += 1
    return float(active_pigs * 1000 + active_blocks)


def _ranking_outcome(
    trajectory_root: Path,
    *,
    release_identity: str,
    adapter: VisualPlanningObservationAdapter,
    source_codec: CohortV2StateCodec,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    raw = _load_json(trajectory_root / "trajectory.json", "ranking trajectory")
    role = str(raw["exposure_role"])
    raw_shot = raw["shots"][-1]
    shot = _load_shot(
        trajectory_root,
        raw_shot,
        release_identity=release_identity,
        role=role,
    )
    fixed_steps = tuple(sorted(shot["frames"]))
    if len(fixed_steps) < 2:
        raise LineageScalingError("ranking outcome has no temporal endpoint")
    endpoint_steps = fixed_steps[-2:]
    observations = tuple(_observation(shot, step) for step in endpoint_steps)
    parsed = adapter.parse_batch(observations)
    deployment = adapter.build_from_parsed(
        TemporalObservationContext(observations[0], observations[1]),
        parsed[1],
        parsed[0],
    ).tensor
    frame = shot["frames"][endpoint_steps[-1]]
    return source_codec.encode(frame), deployment, _realized_goal_cost(frame)


def _ranking_branch_slot(
    original: Mapping[str, Any],
    state_spec: Mapping[str, Any],
    candidate_spec: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **original,
        "slot_identity": identity(
            (
                "issue-63-ranking-branch-slot-v1",
                state_spec["identity"],
                candidate_spec["identity"],
            )
        ),
        "behavior_policy": "issue_63_frozen_ranking_candidates",
        "planned_actions": [
            {
                "identity": candidate_spec["identity"],
                "action_stratum": "issue63_frozen_candidate",
                "selection_mode": "frozen_relative",
                "drag_x": candidate_spec["drag_x"],
                "drag_y": candidate_spec["drag_y"],
                "tap_time_ms": candidate_spec["tap_time_ms"],
            }
        ],
    }


def _collect_ranking_branch(
    paths: Mapping[str, Path],
    original_slot: Mapping[str, Any],
    state_spec: Mapping[str, Any],
    candidate_spec: Mapping[str, Any],
    *,
    game: Path,
    release_identity: str,
    speed: int,
    headless: bool,
) -> Path | None:
    branch_root = (
        paths["root"]
        / "ranking-collection/branches"
        / str(state_spec["identity"])
        / str(candidate_spec["identity"])
    )
    failure_path = branch_root.parent / f"{candidate_spec['identity']}.failure.json"
    if (branch_root / "trajectory.json").is_file():
        return branch_root
    if failure_path.is_file():
        failure = _load_json(failure_path, "ranking collection failure")
        if (
            failure.get("state_identity") != state_spec["identity"]
            or failure.get("candidate_identity") != candidate_spec["identity"]
        ):
            raise LineageScalingError("ranking failure record differs")
        return None
    if branch_root.exists():
        raise LineageScalingError(
            f"incomplete ranking branch requires inspection: {branch_root}"
        )
    branch_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        record = _collect_lineage_attempt(
            _ranking_branch_slot(original_slot, state_spec, candidate_spec),
            branch_root,
            game,
            release_identity=release_identity,
            speed=speed,
            headless=headless,
        )
        if record["scenario_lineage_identity"] != state_spec[
            "scenario_lineage_identity"
        ]:
            raise LineageScalingError(
                "ranking replay changed the frozen scenario lineage"
            )
        return branch_root
    except Exception as error:
        failure = {
            "schema": "issue_63_ranking_candidate_failure_v1",
            "state_identity": state_spec["identity"],
            "candidate_identity": candidate_spec["identity"],
            "error_type": type(error).__name__,
            "error": str(error),
            "failed_run_treatment": "worst_cost",
            "final_evaluation_opened": False,
        }
        _write_json(failure_path, failure)
        _log(
            f"ranking candidate failed state={state_spec['identity']} "
            f"candidate={candidate_spec['identity']} error={type(error).__name__}"
        )
        return None


def _atomic_save_ranking(
    path: Path, states: tuple[ActionRankingState, ...]
) -> None:
    if path.exists():
        raise LineageScalingError(f"ranking bundle already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    save_action_ranking_bundle(temporary, states)
    os.replace(temporary, path)


def _collect_ranking(args: argparse.Namespace, role: str) -> int:
    paths = _paths(args.output)
    protocol, design = _load_frozen(paths)
    if role not in ("calibration", "model_selection"):
        raise LineageScalingError("ranking collection role is unsupported")
    if role == "model_selection" and not paths["decision_freeze"].is_file():
        raise LineageScalingError(
            "freeze the calibration-only decision rule before model-selection ranking"
        )
    manifest, plan, role_records = _release_inventory(args.successor_release)
    if manifest["identity"] != design["source_release_identity"]:
        raise LineageScalingError("ranking collector source release differs")
    slots_by_identity = {
        slot["slot_identity"]: slot for slot in plan["lineages"]
    }
    records_by_trajectory = {
        record["trajectory_identity"]: record
        for role in PUBLIC_ROLES
        for record in role_records[role]
    }
    adapter = _load_deployment_adapter(device=args.device)
    codec = CohortV2StateCodec(latent_dim=197, max_entities=15)
    game = paths["root"] / "ranking-collection/game-runtime"
    if not game.exists():
        _log("extracting accepted issue-62 player for ranking replays")
        archive_details(STAGE_ROOT, game)
    display_process = None
    prior_display = os.environ.get("DISPLAY")
    try:
        if args.start_display:
            display, display_process = start_display(
                paths["root"] / "ranking-collection/display.log"
            )
            os.environ["DISPLAY"] = display
            _log(f"ranking display started DISPLAY={display}")
        total_candidates = sum(
            spec["exposure_role"] == role for spec in design["ranking_states"]
        ) * 4
        completed_candidates = 0
        for role in (role,):
            source_ranking_path = _ranking_path(
                paths, role, CarrierKind.SOURCE
            )
            deployment_ranking_path = _ranking_path(
                paths, role, CarrierKind.DEPLOYMENT
            )
            if source_ranking_path.exists() or deployment_ranking_path.exists():
                if (
                    not source_ranking_path.is_file()
                    or not deployment_ranking_path.is_file()
                ):
                    raise LineageScalingError(
                        f"ranking bundle pair is incomplete for {role}"
                    )
                validate_matched_action_ranking_states(
                    protocol,
                    load_action_ranking_bundle(source_ranking_path),
                    load_action_ranking_bundle(deployment_ranking_path),
                )
                _log(f"ranking bundles validated existing role={role}")
                continue
            source_bundle = load_carrier_lineage_bundle(
                _bundle_path(paths, role, CarrierKind.SOURCE)
            )
            deployment_bundle = load_carrier_lineage_bundle(
                _bundle_path(paths, role, CarrierKind.DEPLOYMENT)
            )
            source_by_trajectory = {
                lineage.trajectory_identity: lineage for lineage in source_bundle
            }
            deployment_by_trajectory = {
                lineage.trajectory_identity: lineage for lineage in deployment_bundle
            }
            source_states: list[ActionRankingState] = []
            deployment_states: list[ActionRankingState] = []
            role_specs = tuple(
                spec
                for spec in design["ranking_states"]
                if spec["exposure_role"] == role
            )
            for state_index, state_spec in enumerate(role_specs, start=1):
                record = records_by_trajectory[state_spec["trajectory_identity"]]
                original_slot = slots_by_identity[record["slot_identity"]]
                outcomes: list[
                    tuple[torch.Tensor, torch.Tensor, float] | None
                ] = []
                for candidate_index, candidate_spec in enumerate(
                    state_spec["candidates"]
                ):
                    if candidate_index == 0:
                        trajectory_root = (
                            args.successor_release / str(state_spec["release_path"])
                        )
                    else:
                        trajectory_root = _collect_ranking_branch(
                            paths,
                            original_slot,
                            state_spec,
                            candidate_spec,
                            game=game,
                            release_identity=str(manifest["identity"]),
                            speed=args.speed,
                            headless=not args.start_display,
                        )
                        completed_candidates += 1
                        _log(
                            f"ranking replay progress={completed_candidates}/"
                            f"{total_candidates} role={role} state={state_index}/"
                            f"{len(role_specs)} candidate={candidate_index + 1}/5"
                        )
                    outcomes.append(
                        None
                        if trajectory_root is None
                        else _ranking_outcome(
                            trajectory_root,
                            release_identity=str(manifest["identity"]),
                            adapter=adapter,
                            source_codec=codec,
                        )
                    )
                successful = tuple(
                    (index, outcome)
                    for index, outcome in enumerate(outcomes)
                    if outcome is not None
                )
                if not successful:
                    raise LineageScalingError(
                        "ranking state has no successful realized candidate"
                    )
                best_index, best_outcome = min(
                    successful,
                    key=lambda item: (item[1][2], item[0]),
                )
                candidates = tuple(
                    ActionCandidate(
                        identity=str(candidate_spec["identity"]),
                        action=torch.tensor(
                            (
                                float(candidate_spec["drag_x"]) / 480.0,
                                float(candidate_spec["drag_y"]) / 480.0,
                                _bounds().release_time_ms / 1000.0,
                                float(candidate_spec["tap_time_ms"]) / 1000.0,
                                1.0,
                            ),
                            dtype=torch.float32,
                        ),
                        realized_cost=(
                            1_000_000_000.0
                            if outcomes[index] is None
                            else outcomes[index][2]
                        ),
                        interface_action=SlingshotAction(
                            int(candidate_spec["drag_x"]),
                            int(candidate_spec["drag_y"]),
                            int(candidate_spec["tap_time_ms"]),
                        ),
                    )
                    for index, candidate_spec in enumerate(state_spec["candidates"])
                )
                source_lineage = source_by_trajectory[
                    state_spec["trajectory_identity"]
                ]
                deployment_lineage = deployment_by_trajectory[
                    state_spec["trajectory_identity"]
                ]
                source_transition = next(
                    item
                    for item in source_lineage.transitions
                    if item.identity == state_spec["decision_transition_identity"]
                )
                deployment_transition = next(
                    item
                    for item in deployment_lineage.transitions
                    if item.identity == state_spec["decision_transition_identity"]
                )
                common = {
                    "identity": str(state_spec["identity"]),
                    "scenario_lineage_identity": str(
                        state_spec["scenario_lineage_identity"]
                    ),
                    "trajectory_identity": str(state_spec["trajectory_identity"]),
                    "decision_transition_identity": str(
                        state_spec["decision_transition_identity"]
                    ),
                    "exposure_role": role,
                    "candidates": candidates,
                    "action_bounds": _bounds(),
                    "frame_height": 480,
                }
                source_state = ActionRankingState(
                    **common,
                    carrier=CarrierKind.SOURCE,
                    carrier_identity=protocol.source_carrier_identity,
                    context=source_transition.context,
                    cost_target=best_outcome[0],
                )
                deployment_state = ActionRankingState(
                    **common,
                    carrier=CarrierKind.DEPLOYMENT,
                    carrier_identity=protocol.deployment_carrier_identity,
                    context=deployment_transition.context,
                    cost_target=best_outcome[1],
                )
                if (
                    source_state.legal_candidate_set_identity
                    != state_spec["legal_candidate_set_identity"]
                    or deployment_state.legal_candidate_set_identity
                    != state_spec["legal_candidate_set_identity"]
                ):
                    raise LineageScalingError(
                        "realized ranking actions differ from their frozen legal set"
                    )
                source_states.append(source_state)
                deployment_states.append(deployment_state)
                _log(
                    f"ranking state complete role={role} state={state_index}/"
                    f"{len(role_specs)} best_candidate={best_index + 1}"
                )
            _atomic_save_ranking(
                source_ranking_path,
                tuple(source_states),
            )
            _atomic_save_ranking(
                deployment_ranking_path,
                tuple(deployment_states),
            )
            validate_matched_action_ranking_states(
                protocol, tuple(source_states), tuple(deployment_states)
            )
            _log(f"ranking bundles written role={role} states={len(source_states)}")
    finally:
        terminate(display_process)
        if prior_display is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = prior_display
    _log(
        f"ranking collection complete role={role} "
        f"states={total_candidates // 4} "
        "final_evaluation=unopened"
    )
    return 0


def _null_variance(lineages: tuple[CarrierLineage, ...], horizon: int) -> float:
    targets = torch.stack(
        tuple(
            transition.target
            for lineage in lineages
            for transition in lineage.transitions
            if transition.horizon == horizon
        )
    )
    mean = targets.mean(dim=0, keepdim=True)
    return max(float(F.mse_loss(targets, mean.expand_as(targets))), 1e-12)


def _score_role(args: argparse.Namespace, role: str) -> int:
    paths = _paths(args.output)
    protocol, _design = _load_frozen(paths)
    if role == "model_selection" and not paths["decision_freeze"].is_file():
        raise LineageScalingError(
            "freeze the calibration-only decision rule before model-selection scoring"
        )
    source_lineages = load_carrier_lineage_bundle(
        _bundle_path(paths, role, CarrierKind.SOURCE)
    )
    deployment_lineages = load_carrier_lineage_bundle(
        _bundle_path(paths, role, CarrierKind.DEPLOYMENT)
    )
    source_ranking = load_action_ranking_bundle(
        _ranking_path(paths, role, CarrierKind.SOURCE)
    )
    deployment_ranking = load_action_ranking_bundle(
        _ranking_path(paths, role, CarrierKind.DEPLOYMENT)
    )
    validate_matched_action_ranking_states(
        protocol, source_ranking, deployment_ranking
    )
    validate_evaluation_lineages(
        protocol, source_lineages, carrier=CarrierKind.SOURCE
    )
    validate_evaluation_lineages(
        protocol, deployment_lineages, carrier=CarrierKind.DEPLOYMENT
    )
    lineages_by_carrier = {
        CarrierKind.SOURCE: source_lineages,
        CarrierKind.DEPLOYMENT: deployment_lineages,
    }
    ranking_by_carrier = {
        CarrierKind.SOURCE: source_ranking,
        CarrierKind.DEPLOYMENT: deployment_ranking,
    }
    nulls = {
        carrier: {
            horizon: _null_variance(lineages, horizon)
            for horizon in (1, 15)
        }
        for carrier, lineages in lineages_by_carrier.items()
    }
    for index, cell in enumerate(protocol.cells, start=1):
        output_path = _score_path(paths, role, cell)
        if output_path.exists():
            existing = _load_json(output_path, "cell score")
            if (
                existing.get("protocol_identity") != protocol.identity
                or existing.get("cell_identity") != cell.identity
                or existing.get("evaluation_role") != role
                or existing.get("identity") != _cell_score_identity(existing)
            ):
                raise LineageScalingError("existing cell score differs")
            _log(f"score {role} cell={index}/{len(protocol.cells)} validated existing")
            continue
        model, checkpoint = load_lineage_scaled_checkpoint(
            _checkpoint_path(paths, cell),
            protocol,
            expected_cell=cell,
            device=args.device,
        )
        training_compute = _load_json(
            _training_report_path(paths, cell), "training compute report"
        )
        if training_compute.get("checkpoint_identity") != checkpoint.identity:
            raise LineageScalingError("training compute report differs")
        _log(
            f"score {role} cell={index}/{len(protocol.cells)} "
            f"scale={cell.scale_name} carrier={cell.carrier.value} seed={cell.seed}"
        )
        prediction = evaluate_continuous_prediction(
            model,
            lineages_by_carrier[cell.carrier],
            horizons=(1, 15),
            physical_diagnostic=_predicted_physical_diagnostics,
            progress=_log,
        )

        def predicted_cost(
            state: ActionRankingState,
            _candidate: ActionCandidate,
            predicted: torch.Tensor,
        ) -> float:
            assert state.cost_target is not None
            return float(F.mse_loss(predicted, state.cost_target))

        ranking = evaluate_action_ranking(
            model,
            ranking_by_carrier[cell.carrier],
            horizon=15,
            recursive_steps=RANKING_RECURSIVE_H15_STEPS,
            predicted_cost=predicted_cost,
            progress=_log,
        )
        recursive = {item.horizon: item for item in prediction.recursive}
        selected_actions = []
        states_by_identity = {
            state.identity: state for state in ranking_by_carrier[cell.carrier]
        }
        for item in ranking.states:
            state = states_by_identity[item.state_identity]
            candidate = next(
                candidate
                for candidate in state.candidates
                if candidate.identity == item.selected_candidate_identity
            )
            selected_actions.append(
                (
                    candidate.interface_action.drag_x,
                    candidate.interface_action.drag_y,
                    candidate.interface_action.tap_time_ms,
                )
            )
        payload = {
            "schema": "issue_63_matched_cell_score_v1",
            "protocol_identity": protocol.identity,
            "checkpoint_identity": checkpoint.identity,
            "cell_identity": cell.identity,
            "scale": cell.scale_name,
            "carrier": cell.carrier.value,
            "seed": cell.seed,
            "evaluation_role": role,
            "lineage_count": len(lineages_by_carrier[cell.carrier]),
            "local_mse_diagnostic": prediction.local_mse,
            "local_by_horizon_diagnostic": dict(prediction.local_by_horizon),
            "recursive": {
                str(horizon): {
                    "mean_mse": recursive[horizon].mean_mse,
                    "error_auc": recursive[horizon].error_auc,
                    "normalized_error_auc": (
                        None
                        if recursive[horizon].error_auc is None
                        else recursive[horizon].error_auc
                        / nulls[cell.carrier][horizon]
                    ),
                    "evaluated_transitions": recursive[horizon].evaluated_transitions,
                }
                for horizon in (1, 15)
            },
            "ranking": {
                "state_count": ranking.state_count,
                "mean_top_action_regret": ranking.mean_top_action_regret,
                "distinct_selected_actions": len(set(selected_actions)),
                "states": [asdict(item) for item in ranking.states],
                "execution_failures": list(ranking.execution_failures),
                "model_evaluations": ranking.model_evaluations,
                "wall_seconds": ranking.wall_seconds,
            },
            "physical": {
                "target": dict(prediction.target_physical_diagnostics),
                "predicted": dict(prediction.predicted_physical_diagnostics),
            },
            "failures": {
                "nonfinite": prediction.nonfinite_failures,
                "execution": list(prediction.execution_failures),
            },
            "compute": {
                "prediction_model_evaluations": prediction.model_evaluations,
                "prediction_wall_seconds": prediction.wall_seconds,
                "training_optimizer_examples": checkpoint.optimizer_examples,
                "training_optimizer_steps": checkpoint.optimizer_steps,
                "training_epochs": checkpoint.epochs,
                "training_wall_seconds": training_compute["wall_seconds"],
                "parameter_count": checkpoint.parameter_count,
            },
            "final_evaluation_opened": False,
        }
        payload["identity"] = _cell_score_identity(payload)
        _write_json(output_path, payload)
        _log(f"score {role} cell={index}/{len(protocol.cells)} written")
    _log(f"{role} scoring complete cells={len(protocol.cells)}")
    return 0


def _score_inventory(
    paths: Mapping[str, Path], protocol: LineageScalingProtocol, role: str
) -> dict[TrainingCell, dict[str, Any]]:
    result = {}
    for cell in protocol.cells:
        score = _load_json(_score_path(paths, role, cell), f"{role} score")
        if (
            score.get("schema") != "issue_63_matched_cell_score_v1"
            or score.get("protocol_identity") != protocol.identity
            or score.get("cell_identity") != cell.identity
            or score.get("evaluation_role") != role
            or score.get("identity") != _cell_score_identity(score)
            or score.get("final_evaluation_opened") is not False
        ):
            raise LineageScalingError(f"{role} score inventory differs")
        result[cell] = score
    return result


def _freeze_decision(args: argparse.Namespace) -> int:
    paths = _paths(args.output)
    protocol, design = _load_frozen(paths)
    if paths["model_selection"].exists() and any(paths["model_selection"].rglob("*.json")):
        raise LineageScalingError(
            "model-selection scores were opened before the decision freeze"
        )
    if any(
        _ranking_path(paths, "model_selection", carrier).exists()
        for carrier in (CarrierKind.SOURCE, CarrierKind.DEPLOYMENT)
    ):
        raise LineageScalingError(
            "model-selection ranking outcomes were opened before the decision freeze"
        )
    scores = _score_inventory(paths, protocol, "calibration")
    summary = {
        cell.identity: {
            "recursive_h1_auc": scores[cell]["recursive"]["1"]["error_auc"],
            "recursive_h15_auc": scores[cell]["recursive"]["15"]["error_auc"],
            "mean_top_action_regret": scores[cell]["ranking"][
                "mean_top_action_regret"
            ],
            "failure_count": scores[cell]["failures"]["nonfinite"]
            + len(scores[cell]["failures"]["execution"])
            + len(scores[cell]["ranking"]["execution_failures"]),
        }
        for cell in protocol.cells
    }
    freeze = {
        "schema": "issue_63_calibration_decision_freeze_v1",
        "design_identity": design["identity"],
        "protocol_identity": protocol.identity,
        "decision_rule": design["decision_rule"],
        "calibration_cell_count": len(scores),
        "calibration_summary": summary,
        "model_selection_opened": False,
        "final_evaluation_opened": False,
    }
    summary["identity"] = identity((
        "issue-63-matched-experiment-summary-v1",
        summary["results_identity"],
        summary["decision"],
        None
        if selected is None
        else tuple(sorted(selected.items())),
    ))
    freeze["identity"] = _decision_freeze_identity(freeze)
    _write_json(paths["decision_freeze"], freeze)
    _log(
        f"decision frozen identity={freeze['identity']} "
        f"calibration_cells={len(scores)} model_selection=unopened"
    )
    return 0


def _paired_bootstrap_interval(
    values: tuple[float, ...], *, confidence: float
) -> tuple[float, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise LineageScalingError("bootstrap values are incomplete")
    generator = random.Random(BOOTSTRAP_SEED)
    samples = sorted(
        sum(generator.choice(values) for _ in values) / len(values)
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    tail = (1.0 - confidence) / 2.0
    lower = samples[min(int(tail * len(samples)), len(samples) - 1)]
    upper = samples[min(int((1.0 - tail) * len(samples)), len(samples) - 1)]
    return lower, upper


def _metric(score: Mapping[str, Any], name: str) -> float:
    if name == "recursive_h1_auc":
        value = score["recursive"]["1"]["error_auc"]
    elif name == "recursive_h15_auc":
        value = score["recursive"]["15"]["error_auc"]
    elif name == "mean_top_action_regret":
        value = score["ranking"]["mean_top_action_regret"]
    else:
        raise LineageScalingError(f"unsupported issue-63 metric {name}")
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise LineageScalingError(f"issue-63 metric {name} is unavailable")
    return float(value)


def _analyze(
    protocol: LineageScalingProtocol,
    scores: Mapping[TrainingCell, Mapping[str, Any]],
) -> dict[str, Any]:
    confidence = 1.0 - 0.05 / 3.0
    primary = {}
    passes = []
    for name in (
        "recursive_h1_auc",
        "recursive_h15_auc",
        "mean_top_action_regret",
    ):
        improvements = []
        for seed in protocol.training_seeds:
            small = scores[TrainingCell("six", CarrierKind.DEPLOYMENT, seed)]
            full = scores[TrainingCell("full", CarrierKind.DEPLOYMENT, seed)]
            baseline = _metric(small, name)
            candidate = _metric(full, name)
            improvements.append((baseline - candidate) / max(abs(baseline), 1e-12))
        interval = _paired_bootstrap_interval(
            tuple(improvements), confidence=confidence
        )
        passed = (
            sum(improvements) / len(improvements) >= PRIMARY_MARGIN
            and interval[0] >= PRIMARY_MARGIN
        )
        passes.append(passed)
        primary[name] = {
            "per_seed_relative_improvement": improvements,
            "mean_relative_improvement": sum(improvements) / len(improvements),
            "simultaneous_interval": list(interval),
            "passed": passed,
        }
    small_failures = 0
    full_failures = 0
    small_physical = []
    full_physical = []
    for seed in protocol.training_seeds:
        for scale, failures, physical in (
            ("six", "small", small_physical),
            ("full", "full", full_physical),
        ):
            score = scores[TrainingCell(scale, CarrierKind.DEPLOYMENT, seed)]
            count = score["failures"]["nonfinite"] + len(
                score["failures"]["execution"]
            ) + len(score["ranking"]["execution_failures"])
            if failures == "small":
                small_failures += count
            else:
                full_failures += count
            physical.append(
                float(score["physical"]["predicted"].get("carrier_bound_excess", 0.0))
            )
    physical_regression = (
        sum(full_physical) / len(full_physical)
        > sum(small_physical) / len(small_physical) + PHYSICAL_REGRESSION_MARGIN
    )
    failure_regression = full_failures > 0
    contrasts = {}
    for scale in ("six", "full"):
        for name, source_key in (
            ("recursive_h1_auc", "1"),
            ("recursive_h15_auc", "15"),
        ):
            values = []
            for seed in protocol.training_seeds:
                source = scores[TrainingCell(scale, CarrierKind.SOURCE, seed)]
                deployment = scores[
                    TrainingCell(scale, CarrierKind.DEPLOYMENT, seed)
                ]
                source_value = float(
                    source["recursive"][source_key]["normalized_error_auc"]
                )
                deployment_value = float(
                    deployment["recursive"][source_key]["normalized_error_auc"]
                )
                values.append(
                    (source_value - deployment_value)
                    / max(abs(source_value), 1e-12)
                )
            contrasts[f"{scale}:{name}"] = {
                "deployment_relative_improvement": values,
                "mean": sum(values) / len(values),
                "paired_seed_95_interval": list(
                    _paired_bootstrap_interval(
                        tuple(values), confidence=0.95
                    )
                ),
            }
    source_h15_improvements = []
    for seed in protocol.training_seeds:
        small_source = scores[
            TrainingCell("six", CarrierKind.SOURCE, seed)
        ]
        full_source = scores[
            TrainingCell("full", CarrierKind.SOURCE, seed)
        ]
        baseline = _metric(small_source, "recursive_h15_auc")
        candidate = _metric(full_source, "recursive_h15_auc")
        source_h15_improvements.append(
            (baseline - candidate) / max(abs(baseline), 1e-12)
        )
    data_effect_source = sum(source_h15_improvements) / len(
        source_h15_improvements
    )
    data_effect_deployment = primary["recursive_h15_auc"][
        "mean_relative_improvement"
    ]
    deployment_h15_improvements = primary["recursive_h15_auc"][
        "per_seed_relative_improvement"
    ]
    interaction_values = tuple(
        deployment - source
        for deployment, source in zip(
            deployment_h15_improvements,
            source_h15_improvements,
            strict=True,
        )
    )
    final_rung_improvements = []
    scale_names = {scale.name for scale in protocol.training_scales}
    prior_scale_name = "1000" if "1000" in scale_names else "six"
    for seed in protocol.training_seeds:
        prior = scores[
            TrainingCell(prior_scale_name, CarrierKind.DEPLOYMENT, seed)
        ]
        full = scores[
            TrainingCell("full", CarrierKind.DEPLOYMENT, seed)
        ]
        prior_value = float(
            prior["recursive"]["15"]["normalized_error_auc"]
        )
        full_value = float(
            full["recursive"]["15"]["normalized_error_auc"]
        )
        final_rung_improvements.append(
            (prior_value - full_value) / max(abs(prior_value), 1e-12)
        )
    final_rung_mean = sum(final_rung_improvements) / len(
        final_rung_improvements
    )
    supported = all(passes) and not physical_regression and not failure_regression
    return {
        "primary_data_scale_effect": primary,
        "carrier_contrasts": contrasts,
        "interaction": {
            "source_h15_data_effect": data_effect_source,
            "deployment_h15_data_effect": data_effect_deployment,
            "deployment_minus_source_h15_relative_effect": (
                data_effect_deployment - data_effect_source
            ),
            "paired_seed_95_interval": list(
                _paired_bootstrap_interval(
                    interaction_values, confidence=0.95
                )
            ),
        },
        "physical_regression": physical_regression,
        "failure_regression": failure_regression,
        "deployment_failure_counts": {
            "six": small_failures,
            "full": full_failures,
        },
        "training_scale_adequacy": {
            "maximum_training_lineages": 3000,
            "full_vs_1000_h15_relative_improvement": final_rung_improvements,
            "mean": final_rung_mean,
            "disposition": (
                "resource_limited_non_saturated"
                if final_rung_mean >= PRIMARY_MARGIN
                else "no_declared_final_rung_gain"
            ),
        },
        "decision": (
            "supported" if supported else "not_supported_by_this_experiment"
        ),
    }


def _svg_learning_curve(
    protocol: LineageScalingProtocol,
    scores: Mapping[TrainingCell, Mapping[str, Any]],
) -> str:
    width, height = 760, 420
    left, top, plot_width, plot_height = 70, 35, 650, 320
    series = {}
    for carrier in (CarrierKind.SOURCE, CarrierKind.DEPLOYMENT):
        values = []
        for scale in protocol.training_scales:
            points = [
                float(
                    scores[TrainingCell(scale.name, carrier, seed)]["recursive"]["15"]
                    ["normalized_error_auc"]
                )
                for seed in protocol.training_seeds
            ]
            values.append(sum(points) / len(points))
        series[carrier] = values
    maximum = max(value for values in series.values() for value in values) or 1.0
    xs = [
        left + index * plot_width / (len(protocol.training_scales) - 1)
        for index in range(len(protocol.training_scales))
    ]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>',
        '<text x="380" y="18" text-anchor="middle">Issue 63 normalized recursive h15 AUC</text>',
    ]
    for carrier, color in (
        (CarrierKind.SOURCE, "#d95f02"),
        (CarrierKind.DEPLOYMENT, "#1b9e77"),
    ):
        points = [
            (x, top + plot_height * (1.0 - value / maximum))
            for x, value in zip(xs, series[carrier], strict=True)
        ]
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="'
            + " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            + '"/>'
        )
        for x, y in points:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        lines.append(
            f'<text x="{left + plot_width - 5}" y="{points[-1][1] - 8:.1f}" '
            f'text-anchor="end" fill="{color}">{carrier.value}</text>'
        )
    for x, scale in zip(xs, protocol.training_scales, strict=True):
        lines.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 24}" text-anchor="middle">{len(scale.lineage_identities)}</text>'
        )
    lines.append('<text x="395" y="405" text-anchor="middle">independent training lineages</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _svg_primary_effects(analysis: Mapping[str, Any]) -> str:
    metrics = (
        "recursive_h1_auc",
        "recursive_h15_auc",
        "mean_top_action_regret",
    )
    labels = ("recursive h1 AUC", "recursive h15 AUC", "ranking regret")
    values = tuple(
        float(
            analysis["primary_data_scale_effect"][metric][
                "mean_relative_improvement"
            ]
        )
        for metric in metrics
    )
    width, height = 760, 360
    zero = 250
    scale = 180
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="380" y="24" text-anchor="middle">Full vs six deployment-carrier effect</text>',
        f'<line x1="{zero}" y1="45" x2="{zero}" y2="315" stroke="black"/>',
    ]
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = 90 + index * 85
        x2 = zero + max(min(value, 1.0), -1.0) * scale
        color = "#1b9e77" if value >= PRIMARY_MARGIN else "#d95f02"
        lines.extend((
            f'<text x="235" y="{y + 5}" text-anchor="end">{label}</text>',
            f'<line x1="{zero}" y1="{y}" x2="{x2:.1f}" y2="{y}" stroke="{color}" stroke-width="18"/>',
            f'<text x="{x2 + (8 if value >= 0 else -8):.1f}" y="{y + 5}" text-anchor="{("start" if value >= 0 else "end")}">{value:.3f}</text>',
        ))
    lines.extend((
        '<text x="70" y="345">worse</text>',
        '<text x="690" y="345" text-anchor="end">better</text>',
        "</svg>",
    ))
    return "\n".join(lines) + "\n"


def _publish(args: argparse.Namespace) -> int:
    paths = _paths(args.output)
    protocol, design = _load_frozen(paths)
    freeze = _load_json(paths["decision_freeze"], "decision freeze")
    if freeze.get("protocol_identity") != protocol.identity:
        raise LineageScalingError("decision freeze differs")
    scores = _score_inventory(paths, protocol, "model_selection")
    analysis = _analyze(protocol, scores)
    supported = analysis["decision"] == "supported"
    selected = None
    if supported:
        candidates = tuple(
            TrainingCell("full", CarrierKind.DEPLOYMENT, seed)
            for seed in protocol.training_seeds
        )
        selected_cell = min(
            candidates,
            key=lambda cell: (
                float(scores[cell]["recursive"]["1"]["normalized_error_auc"])
                + float(scores[cell]["recursive"]["15"]["normalized_error_auc"])
                + float(scores[cell]["ranking"]["mean_top_action_regret"])
            ),
        )
        checkpoint = load_lineage_scaled_checkpoint(
            _checkpoint_path(paths, selected_cell),
            protocol,
            expected_cell=selected_cell,
            device="cpu",
        )[1]
        selected = {
            "cell_identity": selected_cell.identity,
            "checkpoint": str(_checkpoint_path(paths, selected_cell)),
            "checkpoint_identity": checkpoint.identity,
            "carrier": "deployment",
            "controller_mode": "continuous-h15",
            "controller_checkpoint": None,
            "ranking_recursive_h15_steps": RANKING_RECURSIVE_H15_STEPS,
        }
    results = {
        "schema": "issue_63_matched_experiment_results_v1",
        "design_identity": design["identity"],
        "protocol_identity": protocol.identity,
        "decision_freeze_identity": freeze["identity"],
        "model_selection_cell_count": len(scores),
        "analysis": analysis,
        "selected_deployment_configuration": selected,
        "historical_issue_15_role": "external_reference_only",
        "final_evaluation_opened": False,
    }
    results["identity"] = _results_identity(results)
    paths["evidence"].mkdir(parents=True, exist_ok=True)
    _write_json(paths["results"], results)
    table_path = paths["evidence"] / "per-seed-model-selection.csv"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "scale",
                "carrier",
                "seed",
                "local_mse_diagnostic",
                "recursive_h1_auc",
                "recursive_h15_auc",
                "normalized_recursive_h1_auc",
                "normalized_recursive_h15_auc",
                "mean_top_action_regret",
                "distinct_selected_actions",
                "nonfinite_failures",
                "execution_failures",
                "prediction_model_evaluations",
                "prediction_wall_seconds",
                "training_wall_seconds",
            )
        )
        for cell in protocol.cells:
            score = scores[cell]
            writer.writerow(
                (
                    cell.scale_name,
                    cell.carrier.value,
                    cell.seed,
                    score["local_mse_diagnostic"],
                    score["recursive"]["1"]["error_auc"],
                    score["recursive"]["15"]["error_auc"],
                    score["recursive"]["1"]["normalized_error_auc"],
                    score["recursive"]["15"]["normalized_error_auc"],
                    score["ranking"]["mean_top_action_regret"],
                    score["ranking"]["distinct_selected_actions"],
                    score["failures"]["nonfinite"],
                    len(score["failures"]["execution"])
                    + len(score["ranking"]["execution_failures"]),
                    score["compute"]["prediction_model_evaluations"],
                    score["compute"]["prediction_wall_seconds"],
                    score["compute"]["training_wall_seconds"],
                )
            )
    (paths["evidence"] / "learning-curve.svg").write_text(
        _svg_learning_curve(protocol, scores), encoding="utf-8"
    )
    (paths["evidence"] / "primary-effects.svg").write_text(
        _svg_primary_effects(analysis), encoding="utf-8"
    )
    aggregate_path = paths["evidence"] / "aggregate-contrasts.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("contrast", "metric", "estimate", "passed"))
        for metric, result in analysis["primary_data_scale_effect"].items():
            writer.writerow((
                "full-vs-six:deployment",
                metric,
                result["mean_relative_improvement"],
                result["passed"],
            ))
        for key, result in analysis["carrier_contrasts"].items():
            scale, metric = key.split(":", 1)
            writer.writerow((
                f"deployment-vs-source:{scale}",
                metric,
                result["mean"],
                "diagnostic",
            ))
    summary = {
        "schema": "issue_63_matched_experiment_summary_v1",
        "results_identity": results["identity"],
        "decision": analysis["decision"],
        "training_lineages": TRAINING_SCALE_COUNTS[-1],
        "training_scales": list(TRAINING_SCALE_COUNTS),
        "training_seeds": list(TRAINING_SEEDS),
        "matched_primary_cells": 12,
        "all_learning_curve_cells": len(protocol.cells),
        "selected_deployment_configuration": selected,
        "final_evaluation_opened": False,
    }
    _write_json(args.summary, summary)
    _log(
        f"published decision={analysis['decision']} results={results['identity']} "
        f"final_evaluation=unopened"
    )
    return 0


def _validate(args: argparse.Namespace) -> int:
    paths = _paths(args.output)
    protocol, design = _load_frozen(paths)
    for role in PUBLIC_ROLES:
        _log(f"validate carrier bundles role={role}")
        source = load_carrier_lineage_bundle(
            _bundle_path(paths, role, CarrierKind.SOURCE)
        )
        deployment = load_carrier_lineage_bundle(
            _bundle_path(paths, role, CarrierKind.DEPLOYMENT)
        )
        if role == "training":
            validate_matched_carrier_lineages(protocol, source, deployment)
        else:
            validate_evaluation_lineages(
                protocol, source, carrier=CarrierKind.SOURCE
            )
            validate_evaluation_lineages(
                protocol, deployment, carrier=CarrierKind.DEPLOYMENT
            )
    for index, cell in enumerate(protocol.cells, start=1):
        _model, checkpoint = load_lineage_scaled_checkpoint(
            _checkpoint_path(paths, cell),
            protocol,
            expected_cell=cell,
            device="cpu",
        )
        training_compute = _load_json(
            _training_report_path(paths, cell), "training compute report"
        )
        if (
            training_compute.get("cell_identity") != cell.identity
            or training_compute.get("checkpoint_identity") != checkpoint.identity
            or training_compute.get("final_evaluation_opened") is not False
        ):
            raise LineageScalingError("training compute report differs")
        if index == len(protocol.cells) or index % 4 == 0:
            _log(f"validate checkpoints={index}/{len(protocol.cells)}")
    for role in ("calibration", "model_selection"):
        validate_matched_action_ranking_states(
            protocol,
            load_action_ranking_bundle(
                _ranking_path(paths, role, CarrierKind.SOURCE)
            ),
            load_action_ranking_bundle(
                _ranking_path(paths, role, CarrierKind.DEPLOYMENT)
            ),
        )
        _score_inventory(paths, protocol, role)
        _log(f"validate scores role={role} cells={len(protocol.cells)}")
    freeze = _load_json(paths["decision_freeze"], "decision freeze")
    results = _load_json(paths["results"], "experiment results")
    summary = _load_json(args.summary, "compact issue-63 summary")
    if (
        freeze.get("protocol_identity") != protocol.identity
        or freeze.get("identity") != _decision_freeze_identity(freeze)
        or results.get("design_identity") != design["identity"]
        or results.get("decision_freeze_identity") != freeze.get("identity")
        or results.get("identity") != _results_identity(results)
        or results.get("final_evaluation_opened") is not False
        or summary.get("results_identity") != results.get("identity")
        or summary.get("decision") != results["analysis"]["decision"]
        or summary.get("identity")
        != identity((
            "issue-63-matched-experiment-summary-v1",
            summary.get("results_identity"),
            summary.get("decision"),
            None
            if summary.get("selected_deployment_configuration") is None
            else tuple(sorted(summary["selected_deployment_configuration"].items())),
        ))
        or summary.get("final_evaluation_opened") is not False
    ):
        raise LineageScalingError("issue-63 published evidence differs")
    _log(
        f"exact validation passed cells={len(protocol.cells)} "
        f"decision={results['analysis']['decision']} final_evaluation=unopened"
    )
    return 0


def _dry_run(args: argparse.Namespace) -> int:
    manifest, plan, role_records = _release_inventory(args.successor_release)
    adapter = _load_deployment_adapter(device=args.device)
    codec = CohortV2StateCodec(latent_dim=197, max_entities=15)
    pairs = {
        role: tuple(
            _build_lineage_pair(
                args.successor_release,
                str(manifest["identity"]),
                record,
                adapter,
                codec,
                maximum_intervals_per_shot=15,
                maximum_shots=1,
            )
            for record in role_records[role][
                : (8 if role == "training" else 1)
            ]
        )
        for role in PUBLIC_ROLES
    }
    training_source = tuple(pair[0] for pair in pairs["training"])
    training_deployment = tuple(pair[1] for pair in pairs["training"])
    training_bindings = tuple(_binding(item) for item in training_source)
    evaluation_manifests = tuple(
        TrajectoryLineageManifest.create(
            str(manifest["identity"]),
            tuple(_binding(pair[0]) for pair in pairs[role]),
        )
        for role in ("calibration", "model_selection")
    )
    ranking_states = []
    for role, evaluation_manifest in zip(
        ("calibration", "model_selection"), evaluation_manifests, strict=True
    ):
        record = role_records[role][0]
        spec = _ranking_spec(
            args.successor_release, record, evaluation_manifest.bindings[0]
        )
        ranking_states.append(
            FrozenRankingState(
                identity=str(spec["identity"]),
                scenario_lineage_identity=str(spec["scenario_lineage_identity"]),
                trajectory_identity=str(spec["trajectory_identity"]),
                decision_transition_identity=str(spec["decision_transition_identity"]),
                exposure_role=role,
                legal_candidate_set_identity=str(spec["legal_candidate_set_identity"]),
            )
        )
    protocol = LineageScalingProtocol(
        training_scales=(
            FrozenLineageScale.from_manifest(
                "six",
                TrajectoryLineageManifest.create(
                    str(manifest["identity"]), training_bindings[:6]
                ),
            ),
            FrozenLineageScale.from_manifest(
                "full",
                TrajectoryLineageManifest.create(
                    str(manifest["identity"]), training_bindings
                ),
            ),
        ),
        evaluation_manifests=evaluation_manifests,
        ranking_states=tuple(ranking_states),
        training_seeds=TRAINING_SEEDS,
        training_horizons=(1, 15),
        optimizer_example_budget=8,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        predictor_config=PredictorConfig(
            latent_dim=197, action_dim=5, hidden_dim=32, depth=1
        ),
        source_max_entities=15,
        source_carrier_identity=codec.identity,
        deployment_carrier_identity=adapter.identity,
        configuration_basis="prospectively_frozen",
    )
    validate_matched_carrier_lineages(
        protocol, training_source, training_deployment
    )
    for index, cell in enumerate(protocol.primary_cells, start=1):
        pool = training_source if cell.carrier is CarrierKind.SOURCE else training_deployment
        membership = set(protocol.scale(cell.scale_name).lineage_identities)
        selected = tuple(
            lineage
            for lineage in pool
            if lineage.scenario_lineage_identity in membership
        )
        model, _report = train_continuous_predictor(
            protocol, cell, selected, device=args.device
        )
        if index == len(protocol.primary_cells):
            evaluation = evaluate_continuous_prediction(
                model,
                (pairs["calibration"][0][1],),
                horizons=(1, 15),
            )
    calibration_spec = _ranking_spec(
        args.successor_release,
        role_records["calibration"][0],
        evaluation_manifests[0].bindings[0],
    )
    calibration_transition = pairs["calibration"][0][1].transitions[0]
    ranking_candidates = tuple(
        ActionCandidate(
            identity=str(candidate["identity"]),
            action=torch.tensor((
                float(candidate["drag_x"]) / 480.0,
                float(candidate["drag_y"]) / 480.0,
                _bounds().release_time_ms / 1000.0,
                float(candidate["tap_time_ms"]) / 1000.0,
                1.0,
            )),
            realized_cost=float(candidate_index),
            interface_action=SlingshotAction(
                int(candidate["drag_x"]),
                int(candidate["drag_y"]),
                int(candidate["tap_time_ms"]),
            ),
        )
        for candidate_index, candidate in enumerate(calibration_spec["candidates"])
    )
    ranking_state = ActionRankingState(
        identity=str(calibration_spec["identity"]),
        scenario_lineage_identity=str(
            calibration_spec["scenario_lineage_identity"]
        ),
        trajectory_identity=str(calibration_spec["trajectory_identity"]),
        decision_transition_identity=str(
            calibration_spec["decision_transition_identity"]
        ),
        exposure_role="calibration",
        carrier=CarrierKind.DEPLOYMENT,
        carrier_identity=protocol.deployment_carrier_identity,
        context=calibration_transition.context,
        candidates=ranking_candidates,
        action_bounds=_bounds(),
        frame_height=480,
        cost_target=calibration_transition.target,
    )
    ranking = evaluate_action_ranking(
        model,
        (ranking_state,),
        horizon=15,
        recursive_steps=2,
        predicted_cost=lambda state, _candidate, predicted: float(
            F.mse_loss(predicted, state.cost_target)
        ),
    )
    calibration_record = role_records["calibration"][0]
    original_slot = next(
        slot
        for slot in plan["lineages"]
        if slot["slot_identity"] == calibration_record["slot_identity"]
    )
    branch_slot = _ranking_branch_slot(
        original_slot,
        calibration_spec,
        calibration_spec["candidates"][1],
    )
    if (
        len(branch_slot["planned_actions"]) != 1
        or ranking.model_evaluations != len(ranking_candidates) * 2
    ):
        raise LineageScalingError("ranking dry-run path is incomplete")
    _log(
        f"dry-run passed real_lineages={sum(len(value) for value in pairs.values())} "
        f"matched_cells={len(protocol.primary_cells)} "
        f"recursive_horizons={tuple(item.horizon for item in evaluation.recursive)} "
        f"ranking_candidates={len(ranking_candidates)} "
        "files_written=0 final_evaluation=unopened"
    )
    _log(
        "production sequence: --prepare; --collect-calibration-ranking "
        "--start-display; --train; --score-calibration; --freeze-decision; "
        "--collect-model-selection-ranking --start-display; "
        "--score-model-selection; --publish; --validate"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--collect-calibration-ranking", action="store_true")
    mode.add_argument("--collect-model-selection-ranking", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--score-calibration", action="store_true")
    mode.add_argument("--freeze-decision", action="store_true")
    mode.add_argument("--score-model-selection", action="store_true")
    mode.add_argument("--publish", action="store_true")
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--successor-release", type=Path, default=DEFAULT_SUCCESSOR_RELEASE
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start-display", action="store_true")
    parser.add_argument("--speed", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    args = _parser().parse_args(argv)
    args.output = args.output.resolve()
    args.successor_release = args.successor_release.resolve()
    args.summary = args.summary.resolve()
    if args.speed <= 0:
        raise LineageScalingError("ranking collection speed must be positive")
    if args.dry_run:
        return _dry_run(args)
    if args.prepare:
        return _prepare(args)
    if args.collect_calibration_ranking:
        return _collect_ranking(args, "calibration")
    if args.collect_model_selection_ranking:
        return _collect_ranking(args, "model_selection")
    if args.train:
        return _train(args)
    if args.score_calibration:
        return _score_role(args, "calibration")
    if args.freeze_decision:
        return _freeze_decision(args)
    if args.score_model_selection:
        return _score_role(args, "model_selection")
    if args.publish:
        return _publish(args)
    return _validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
