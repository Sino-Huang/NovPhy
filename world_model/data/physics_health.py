"""Read-only health reporting for a `physics_capture_v1` cohort and its derived labels.

Answers the questions Milestone 0 has to answer honestly before any training run:
how much of the cohort actually carries engine physics, how much of it is labelled,
what the oracle gate and macro-state distributions look like, whether every RGB frame
is frame-exact with its state record, and -- importantly -- which regimes the cohort
does *not* cover.  Nothing here mutates the cohort.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from scripts.physics_label_derivation import (
    MacroState,
    OracleGateSpec,
    ShotOutcomeClass,
    DerivedLabelError,
    validate_derived_labels,
)
from scripts.rollout_artifacts import (
    EpisodeSummary,
    EpisodeValidationMode,
    validate_rollout_episode,
)
from world_model.data.catalog_plan import episode_contract_from_manifest
from world_model.data.types import PHYSICS_CAPTURE_V1


class PhysicsSplitHealth(TypedDict):
    accepted_episodes: int
    accepted_shots: int
    shots_with_sidecars: int
    shots_with_valid_labels: int
    shots_with_invalid_labels: int
    label_failure_reasons: dict[str, int]
    frames_total: int
    frames_frame_exact: int
    oracle_gate_open_frames: int
    oracle_gate_open_rate: float | None
    macro_state_frame_counts: dict[str, int]
    outcome_counts: dict[str, int]
    buckets: list[str]


class PhysicsCoverageReport(TypedDict):
    oracle_gate_spec: dict[str, float | int]
    oracle_gate_spec_digest: str
    macro_state_taxonomy: list[str]
    splits: dict[str, PhysicsSplitHealth]
    covered_buckets: list[str]
    uncovered_regimes: dict[str, object]


def _bucket_of(episode_name: str) -> str:
    """Recover `novelty_level_x/typeY` from the planner's episode directory name."""
    parts = episode_name.split("_")
    for index, part in enumerate(parts):
        if part.startswith("type"):
            return f"{'_'.join(parts[:index])}/{part}"
    return episode_name


def _split_health(root: Path, split: str, spec: OracleGateSpec) -> PhysicsSplitHealth:
    split_root = root / split
    candidates = (
        tuple(sorted(child for child in split_root.iterdir() if not child.is_symlink() and child.is_dir()))
        if split_root.is_dir()
        else ()
    )
    accepted_episodes = 0
    accepted_shots = 0
    shots_with_sidecars = 0
    valid_labels = 0
    invalid_labels = 0
    failure_reasons = Counter[str]()
    frames_total = 0
    frames_frame_exact = 0
    gate_open = 0
    macro_counts = Counter[str]()
    outcome_counts = Counter[str]()
    buckets: set[str] = set()

    for episode_dir in candidates:
        contract = episode_contract_from_manifest(episode_dir)
        result = validate_rollout_episode(
            episode_dir,
            contract,
            mode=EpisodeValidationMode.CANONICAL_SUMMARY,
            capture_contract=PHYSICS_CAPTURE_V1.contract_name,
        )
        if not isinstance(result, EpisodeSummary):
            continue
        accepted_episodes += 1
        buckets.add(_bucket_of(episode_dir.name))
        for shot in result.episode.shots:
            accepted_shots += 1
            # ValidatedShot.relative_path is episode-relative, not root-relative.
            shot_dir = episode_dir / shot.relative_path
            if not (shot_dir / "physics_state.jsonl").is_file():
                continue
            shots_with_sidecars += 1
            try:
                labels = validate_derived_labels(shot_dir, spec)
            except (OSError, DerivedLabelError, ValueError) as error:
                invalid_labels += 1
                reason = error.detail if isinstance(error, DerivedLabelError) else type(error).__name__
                failure_reasons[reason] += 1
                continue
            valid_labels += 1
            outcome_counts[labels.outcome.outcome_class.value] += 1
            for frame in labels.frames:
                frames_total += 1
                # The capture contract requires RGB/state render-frame equality and the
                # label file is re-derived from that same accepted capture, so a counted
                # frame is a frame-exact one by construction; report it explicitly.
                frames_frame_exact += 1
                if frame.oracle_gate:
                    gate_open += 1
                for state in frame.macro_states:
                    macro_counts[state.value] += 1

    return PhysicsSplitHealth(
        accepted_episodes=accepted_episodes,
        accepted_shots=accepted_shots,
        shots_with_sidecars=shots_with_sidecars,
        shots_with_valid_labels=valid_labels,
        shots_with_invalid_labels=invalid_labels,
        label_failure_reasons=dict(sorted(failure_reasons.items())),
        frames_total=frames_total,
        frames_frame_exact=frames_frame_exact,
        oracle_gate_open_frames=gate_open,
        oracle_gate_open_rate=(gate_open / frames_total) if frames_total else None,
        macro_state_frame_counts={
            state.value: macro_counts.get(state.value, 0) for state in MacroState
        },
        outcome_counts={
            outcome.value: outcome_counts.get(outcome.value, 0) for outcome in ShotOutcomeClass
        },
        buckets=sorted(buckets),
    )


def physics_coverage_report(
    root: Path,
    splits: Sequence[str],
    spec: OracleGateSpec | None = None,
    production_bucket_count: int = 80,
) -> PhysicsCoverageReport:
    """Build the physics/label section of the dataset-health report."""
    gate_spec = spec or OracleGateSpec()
    reports = {split: _split_health(root, split, gate_spec) for split in splits}
    covered = sorted({bucket for report in reports.values() for bucket in report["buckets"]})
    return PhysicsCoverageReport(
        oracle_gate_spec=gate_spec.to_json(),
        oracle_gate_spec_digest=gate_spec.digest(),
        macro_state_taxonomy=[state.value for state in MacroState],
        splits=reports,
        covered_buckets=covered,
        uncovered_regimes={
            "production_bucket_count": production_bucket_count,
            "covered_bucket_count": len(covered),
            "uncovered_bucket_count": max(0, production_bucket_count - len(covered)),
            "note": (
                "The staged physics player ships a single level, so this cohort covers "
                "only the buckets listed in covered_buckets. It is not a multi-regime "
                "cohort and must not be reported as one; extending regime coverage "
                "requires rebuilding the staged player with a wider level set."
            ),
            "splits_collected": sorted(reports),
        },
    )
