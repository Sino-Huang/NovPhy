"""Publish and verify an immutable representative cohort release."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

from scripts.cohort_partition import (
    CohortPartitionManifest,
    audit_cohort_partition_manifest,
    load_cohort_partition_manifest,
)
from scripts.collection_plan import PLAN_COPY_FILENAME, REPORT_FILENAME, load_collection_plan
from scripts.physics_capture_contract import load_physics_capture
from scripts.physics_macro_labels import (
    MACRO_LABEL_SIDECAR,
    derive_macro_labels_for_shot,
    validate_macro_labels,
    write_macro_label_file,
)
from scripts.physics_relational_supervision import (
    RelationalAvailability,
    validate_relational_supervision,
)
from scripts.production_plan import PRODUCTION_PLAN_COPY_FILENAME, load_production_plan
from scripts.rollout_artifacts import validate_physics_shot_artifact
from scripts.scenario_manifest import ScenarioManifest
from world_model.data.supervision import PhysicsSupervisionRequest, read_physics_shot

RELEASE_SCHEMA: Final = "representative_cohort_release_v1"
DERIVATION_SCHEMA: Final = "authoritative_cohort_derivations_v1"
PUBLICATION_SCHEMA: Final = "representative_cohort_publication_v1"
NAMESPACES: Final = {
    RELEASE_SCHEMA: "representative-cohort-release-v1",
    DERIVATION_SCHEMA: "authoritative-cohort-derivations-v1",
    PUBLICATION_SCHEMA: "representative-cohort-publication-v1",
}
INGESTION_READERS: Final = (
    "cohort_partition_manifest_v1",
    "scenario_manifest_v1",
    "physics_capture_v1",
    "physics_macro_labels_v1",
    "physics_relational_supervision_v1",
    "world_model_physics_supervision",
)


@dataclass(frozen=True, slots=True)
class CohortIngestionEvidence:
    publication_identity: str
    cohort_release_identity: str
    cohort_release_version: int
    derivation_identity: str
    derivation_version: int
    partition_identity: str
    partition_version: int
    readers: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    available_capabilities: Mapping[str, str]
    unavailable_capabilities: Mapping[str, str]
    rollout_count: int
    frame_count: int
    event_count: int
    identity_aligned_frame_count: int
    fixed_step_aligned_event_count: int
    unavailable_relational_label_count: int
    terminal_observation_count: int
    macro_frame_count: int
    relational_frame_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "cohort_ingestion_evidence_v1",
            "publication_identity": self.publication_identity,
            "cohort_release": {
                "identity": self.cohort_release_identity,
                "version": self.cohort_release_version,
            },
            "authoritative_derivations": {
                "identity": self.derivation_identity,
                "version": self.derivation_version,
            },
            "partition": {
                "identity": self.partition_identity,
                "version": self.partition_version,
            },
            "readers": list(self.readers),
            "required_capabilities": list(self.required_capabilities),
            "available_capabilities": dict(self.available_capabilities),
            "unavailable_capabilities": dict(self.unavailable_capabilities),
            "counts": {
                "rollouts": self.rollout_count,
                "frames": self.frame_count,
                "events": self.event_count,
                "identity_aligned_frames": self.identity_aligned_frame_count,
                "fixed_step_aligned_events": self.fixed_step_aligned_event_count,
                "terminal_observations": self.terminal_observation_count,
                "macro_frames": self.macro_frame_count,
                "relational_frames": self.relational_frame_count,
                "unavailable_relational_labels": self.unavailable_relational_label_count,
            },
        }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def _identity(payload: Mapping[str, Any]) -> str:
    content = dict(payload)
    content.pop("identity", None)
    return f"{NAMESPACES[str(payload['schema'])]}:sha256:{sha256(_canonical_json(content)).hexdigest()}"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load {name} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _write(path: Path, value: Any) -> Path:
    data = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Immutable artifact differs: {path}")
        return path
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _copy(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise ValueError(f"Immutable copy differs: {destination}")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as handle:
        handle.write(source.read_bytes())
        temporary = Path(handle.name)
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _artifact(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Attempt has no artifact path")
    source = Path(value)
    if source.is_absolute():
        resolved = source.resolve()
    else:
        candidates = (root / source, source)
        resolved = next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[0].resolve())
    if root != resolved and root not in resolved.parents:
        raise ValueError("Attempt artifact is outside production output")
    return resolved


def _raw_inventory(root: Path, shot: Path) -> list[dict[str, Any]]:
    paths = [shot / name for name in ("metadata.json", "physics_state.jsonl", "physics_events.jsonl")]
    frames = shot / "frames"
    if frames.is_dir():
        paths += [path for path in sorted(frames.rglob("*")) if path.is_file()]
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError(f"Primary rollout has missing or non-regular files: {shot}")
    return [
        {"path": path.relative_to(root).as_posix(), "sha256": _digest(path), "size_bytes": path.stat().st_size}
        for path in paths
    ]


def _quality(root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    ledger, planned = report.get("attempt_ledger"), report.get("planned_slots")
    if not isinstance(ledger, list) or not isinstance(planned, list):
        raise ValueError("Collection report lacks plan and attempt accounting")
    counts = {name: report.get(f"{name}_count") for name in ("accepted", "rejected", "failed", "quarantined")}
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values()):
        raise ValueError("Collection counts are invalid")
    if counts["accepted"] + counts["rejected"] + counts["failed"] != len(ledger):
        raise ValueError("Collection report does not account for every attempt")
    if report.get("unmet_slots") or report.get("realized_coverage_shortfalls"):
        raise ValueError("Production has unmet slots or coverage shortfalls")
    normalized = []
    for entry in ledger:
        item = dict(entry)
        for field in ("artifact_path", "quarantine_path", "failure_manifest_path"):
            if isinstance(item.get(field), str) and item[field]:
                item[field] = _artifact(root, item[field]).relative_to(root).as_posix()
        normalized.append(item)
    return {
        "schema": "production_quality_report_v1",
        "plan_identity": report.get("plan_identity"),
        "plan_version": report.get("plan_version"),
        "planned_rollout_count": len(planned),
        "attempted_rollout_count": len(ledger),
        "counts": counts,
        "attempt_ledger": normalized,
        "realized_coverage_stratum_counts": report.get("realized_coverage_stratum_counts"),
        "unmet_slots": [], "coverage_shortfalls": [], "systematic_exporter_defects": [],
    }


def _prior_history(release_root: Path, sources: Mapping[str, Path]) -> list[dict[str, Any]]:
    history = []
    for name, source in sorted(sources.items()):
        data = _load(Path(source), "prior execution report")
        copied = _copy(Path(source), release_root / "prior_executions" / f"{name}.json")
        history.append({
            "name": name,
            "schema": data.get("schema"),
            "path": copied.relative_to(release_root).as_posix(),
            "sha256": _digest(copied),
            "attempt_count": len(data.get("attempt_ledger", [])),
            "accepted_count": data.get("accepted_count", 0),
            "rejected_count": data.get("rejected_count", 0),
            "failed_count": data.get("failed_count", 1 if data.get("status") == "failed" else 0),
            "quarantined_count": data.get("quarantined_count", 0),
        })
    return history


def _derivations(root: Path, release_root: Path, release_id: str, rollouts: Sequence[Mapping[str, Any]], available: Mapping[str, str], unavailable: Mapping[str, str]) -> tuple[Path, dict[str, Any]]:
    artifacts = []
    for rollout in rollouts:
        attempt_id, shot = str(rollout["attempt_id"]), root / str(rollout["path"])
        destination = release_root / "derivations" / attempt_id
        destination.mkdir(parents=True, exist_ok=True)
        labels = derive_macro_labels_for_shot(shot)
        macro = destination / MACRO_LABEL_SIDECAR
        if macro.exists() and macro.read_text(encoding="utf-8") != labels.to_jsonl():
            raise ValueError(f"Immutable macro derivation differs: {macro}")
        if not macro.exists():
            write_macro_label_file(labels, macro)
        artifacts.append({"attempt_id": attempt_id, "kind": "physics_macro_labels_v1", "path": macro.relative_to(release_root).as_posix(), "sha256": _digest(macro), "accepted_predicates": ["steady-state", "structure-unstable"], "excluded_predicates": ["cascade-active", "collapsed", "pigs-cleared"]})
        relational = shot / "physics_relational_supervision.jsonl"
        if not relational.is_file():
            raise ValueError(f"Accepted rollout lacks relational supervision: {shot}")
        copied = _copy(relational, destination / relational.name)
        artifacts.append({"attempt_id": attempt_id, "kind": "physics_relational_supervision_v1", "path": copied.relative_to(release_root).as_posix(), "sha256": _digest(copied)})
    payload = {"schema": DERIVATION_SCHEMA, "identity": "", "derivation_version": 1, "source_cohort_release_identity": release_id, "available_capabilities": dict(sorted(available.items())), "unavailable_capabilities": dict(sorted(unavailable.items())), "artifacts": artifacts}
    payload["identity"] = _identity(payload)
    path = release_root / f"authoritative_derivations_{payload['identity'].rsplit(':', 1)[-1]}_v1.json"
    return _write(path, payload), payload


def publish_cohort_release(output_dir: Path, *, partition_manifest_path: Path, scenario_manifest_paths: Mapping[str, Path], release_version: int, code_revision: str, available_capabilities: Mapping[str, str], unavailable_capabilities: Mapping[str, str], prior_execution_paths: Mapping[str, Path] | None = None) -> Path:
    if isinstance(release_version, bool) or not isinstance(release_version, int) or release_version <= 0 or not code_revision:
        raise ValueError("Release version and code revision are required")
    root = Path(output_dir).resolve()
    loaded_plan = load_collection_plan(root / PLAN_COPY_FILENAME)
    production_plan = load_production_plan(root / PRODUCTION_PLAN_COPY_FILENAME)
    report = _load(root / REPORT_FILENAME, "collection report")
    if production_plan.source_collection_plan["identity"] != loaded_plan.plan.identity or report.get("plan_identity") != loaded_plan.plan.identity:
        raise ValueError("Production artifacts are not bound to the frozen collection plan")
    partition = CohortPartitionManifest.from_dict(_load(Path(partition_manifest_path), "partition manifest"))
    lineages = [scenario.scenario_manifest_projection["scenario_lineage_identity"] for scenario in loaded_plan.plan.scenarios]
    audit_cohort_partition_manifest(partition, admitted_scenario_lineage_identities=lineages, admitted_provenance_records=[record.to_dict() for record in partition.provenance_records])

    release_root = root / "release"
    partition_copy = _copy(Path(partition_manifest_path), release_root / "partition_manifest.json")
    if set(scenario_manifest_paths) != {scenario.scenario_id for scenario in loaded_plan.plan.scenarios}:
        raise ValueError("Scenario manifest sources must exactly match the plan")
    scenarios = []
    for scenario in loaded_plan.plan.scenarios:
        source = Path(scenario_manifest_paths[scenario.scenario_id])
        if _load(source, "scenario manifest") != scenario.scenario_manifest_projection["scenario_manifest"]:
            raise ValueError(f"Scenario manifest differs from plan: {scenario.scenario_id}")
        copied = _copy(source, release_root / "scenario_manifests" / f"{scenario.scenario_id}.json")
        scenarios.append({"scenario_id": scenario.scenario_id, "scenario_lineage_identity": scenario.scenario_manifest_projection["scenario_lineage_identity"], "path": copied.relative_to(release_root).as_posix(), "sha256": _digest(copied)})

    quality = _quality(root, report)
    quality["prior_executions"] = _prior_history(release_root, prior_execution_paths or {})
    quality_path = _write(release_root / "production_quality_report.json", quality)
    rollouts = []
    for entry in report["attempt_ledger"]:
        if entry.get("status") == "accepted" and entry.get("disposition") == "accept":
            shot = _artifact(root, entry.get("artifact_path"))
            validate_physics_shot_artifact(shot)
            rollouts.append({"attempt_id": entry["attempt_id"], "scenario_id": entry["scenario_id"], "intervention_id": entry["intervention_id"], "path": shot.relative_to(root).as_posix(), "files": _raw_inventory(root, shot)})
    if not rollouts or len(rollouts) != quality["counts"]["accepted"]:
        raise ValueError("Accepted rollout inventory does not match accounting")

    release = {
        "schema": RELEASE_SCHEMA, "identity": "", "release_version": release_version, "code_revision": code_revision,
        "source_pilot_report_identity": production_plan.source_pilot_report["identity"],
        "production_plan": {"identity": production_plan.identity, "version": production_plan.plan_version, "path": PRODUCTION_PLAN_COPY_FILENAME, "sha256": _digest(root / PRODUCTION_PLAN_COPY_FILENAME)},
        "collection_plan": {"identity": loaded_plan.plan.identity, "version": loaded_plan.plan.plan_version, "path": PLAN_COPY_FILENAME, "sha256": _digest(root / PLAN_COPY_FILENAME)},
        "partition_manifest": {"identity": partition.identity, "version": partition.partition_version, "path": partition_copy.relative_to(root).as_posix(), "sha256": _digest(partition_copy)},
        "scenario_manifests": scenarios, "primary_rollouts": rollouts,
        "quality_report": {"path": quality_path.relative_to(root).as_posix(), "sha256": _digest(quality_path)},
    }
    release["identity"] = _identity(release)
    release_path = _write(release_root / f"cohort_release_{release['identity'].rsplit(':', 1)[-1]}_v{release_version}.json", release)
    derivation_path, derivation = _derivations(root, release_root, release["identity"], rollouts, available_capabilities, unavailable_capabilities)
    publication = {"schema": PUBLICATION_SCHEMA, "identity": "", "cohort_release": {"identity": release["identity"], "path": release_path.relative_to(root).as_posix(), "sha256": _digest(release_path)}, "authoritative_derivations": {"identity": derivation["identity"], "path": derivation_path.relative_to(root).as_posix(), "sha256": _digest(derivation_path)}}
    publication["identity"] = _identity(publication)
    return _write(release_root / f"cohort_publication_{publication['identity'].rsplit(':', 1)[-1]}_v{release_version}.json", publication)


def verify_cohort_publication(path: Path) -> dict[str, Any]:
    publication_path = Path(path).resolve()
    publication = _load(publication_path, "cohort publication")
    if publication.get("schema") != PUBLICATION_SCHEMA or publication.get("identity") != _identity(publication):
        raise ValueError("Cohort publication identity is stale or unsupported")
    root = publication_path.parent.parent
    for field, schema in (("cohort_release", RELEASE_SCHEMA), ("authoritative_derivations", DERIVATION_SCHEMA)):
        reference = publication.get(field)
        artifact_path = root / str(reference.get("path")) if isinstance(reference, dict) else root
        if not isinstance(reference, dict) or _digest(artifact_path) != reference.get("sha256"):
            raise ValueError(f"Cohort publication {field} digest mismatch")
        artifact = _load(artifact_path, field)
        if artifact.get("schema") != schema or artifact.get("identity") != _identity(artifact) or artifact.get("identity") != reference.get("identity"):
            raise ValueError(f"Cohort publication {field} identity mismatch")
    return publication


def _published_path(root: Path, path_value: Any, name: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{name} path is missing")
    path = (root / path_value).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"{name} is outside the cohort publication")
    return path


def _published_file(root: Path, path_value: Any, name: str) -> Path:
    path = _published_path(root, path_value, name)
    if not path.is_file():
        raise ValueError(f"{name} is missing: {path}")
    return path


def _verify_file_reference(root: Path, reference: Any, name: str, *, check_size: bool = False) -> Path:
    if not isinstance(reference, Mapping):
        raise ValueError(f"{name} reference must be an object")
    path = _published_file(root, reference.get("path"), name)
    if _digest(path) != reference.get("sha256"):
        raise ValueError(f"{name} digest mismatch")
    if check_size and path.stat().st_size != reference.get("size_bytes"):
        raise ValueError(f"{name} size mismatch")
    return path


def ingest_cohort_publication(
    path: Path,
    *,
    required_capabilities: tuple[str, ...] = (),
) -> CohortIngestionEvidence:
    """Read one immutable cohort release through every required public reader."""
    if (
        type(required_capabilities) is not tuple
        or len(required_capabilities) != len(set(required_capabilities))
        or not all(isinstance(item, str) and item for item in required_capabilities)
    ):
        raise ValueError("required_capabilities must be a unique tuple of nonempty strings")

    publication_path = Path(path).resolve()
    publication = verify_cohort_publication(publication_path)
    root = publication_path.parent.parent.resolve()
    release_path = _verify_file_reference(root, publication["cohort_release"], "cohort release")
    derivation_path = _verify_file_reference(
        root, publication["authoritative_derivations"], "authoritative derivations"
    )
    release = _load(release_path, "cohort release")
    derivations = _load(derivation_path, "authoritative derivations")

    if derivations.get("source_cohort_release_identity") != release.get("identity"):
        raise ValueError("Authoritative derivations are bound to another cohort release")
    available = derivations.get("available_capabilities")
    unavailable = derivations.get("unavailable_capabilities")
    if (
        not isinstance(available, dict)
        or not isinstance(unavailable, dict)
        or not all(isinstance(key, str) and isinstance(value, str) for key, value in available.items())
        or not all(isinstance(key, str) and isinstance(value, str) for key, value in unavailable.items())
        or set(available) & set(unavailable)
    ):
        raise ValueError("Cohort capability declarations are malformed")
    missing = tuple(item for item in required_capabilities if item not in available)
    if missing:
        raise ValueError(f"Required cohort capabilities are unavailable: {missing!r}")

    partition_reference = release.get("partition_manifest")
    partition_path = _verify_file_reference(root, partition_reference, "cohort partition")
    partition = load_cohort_partition_manifest(partition_path)
    if (
        not isinstance(partition_reference, Mapping)
        or partition.identity != partition_reference.get("identity")
        or partition.partition_version != partition_reference.get("version")
    ):
        raise ValueError("Cohort partition identity or version mismatch")

    scenario_references = release.get("scenario_manifests")
    if not isinstance(scenario_references, list) or not scenario_references:
        raise ValueError("Cohort release has no scenario manifest inventory")
    partition_lineages = {
        str(entry.scenario_manifest_projection["scenario_lineage_identity"])
        for entry in partition.entries
    }
    scenario_ids: set[str] = set()
    scenario_lineages: set[str] = set()
    for reference in scenario_references:
        if not isinstance(reference, Mapping):
            raise ValueError("Published scenario manifest reference must be an object")
        scenario_id = reference.get("scenario_id")
        lineage = reference.get("scenario_lineage_identity")
        if (
            not isinstance(scenario_id, str)
            or not scenario_id
            or scenario_id in scenario_ids
            or not isinstance(lineage, str)
            or not lineage
            or lineage in scenario_lineages
        ):
            raise ValueError("Published scenario identity is missing or duplicated")
        scenario_path = _verify_file_reference(
            release_path.parent, reference, f"scenario manifest {scenario_id}"
        )
        scenario = ScenarioManifest.from_dict(_load(scenario_path, f"scenario manifest {scenario_id}"))
        if scenario.scenario_lineage.identity != lineage:
            raise ValueError(f"Published scenario lineage identity mismatch for {scenario_id}")
        scenario_ids.add(scenario_id)
        scenario_lineages.add(lineage)
    if scenario_lineages != partition_lineages:
        raise ValueError("Published scenario lineages do not match the cohort partition")

    rollouts = release.get("primary_rollouts")
    artifacts = derivations.get("artifacts")
    if not isinstance(rollouts, list) or not rollouts or not isinstance(artifacts, list):
        raise ValueError("Cohort release has no complete rollout and derivation inventories")
    derivation_by_attempt: dict[tuple[str, str], Path] = {}
    for reference in artifacts:
        if not isinstance(reference, Mapping):
            raise ValueError("Authoritative derivation reference must be an object")
        attempt_id, kind = reference.get("attempt_id"), reference.get("kind")
        if not isinstance(attempt_id, str) or kind not in {
            "physics_macro_labels_v1",
            "physics_relational_supervision_v1",
        }:
            raise ValueError("Authoritative derivation reference is malformed or unsupported")
        expected_fields = {"attempt_id", "kind", "path", "sha256"}
        if kind == "physics_macro_labels_v1":
            expected_fields.update({"accepted_predicates", "excluded_predicates"})
        if set(reference) != expected_fields:
            raise ValueError(
                "Authoritative derivation reference is incomplete or contains unknown fields"
            )
        key = (attempt_id, str(kind))
        if key in derivation_by_attempt:
            raise ValueError("Authoritative derivation references are duplicated")
        derivation_by_attempt[key] = _verify_file_reference(
            derivation_path.parent, reference, f"{kind} for {attempt_id}"
        )

    frame_count = event_count = terminal_count = macro_count = relational_count = 0
    identity_count = aligned_event_count = unavailable_label_count = 0
    seen_attempts: set[str] = set()
    for rollout in rollouts:
        if not isinstance(rollout, Mapping):
            raise ValueError("Primary rollout reference must be an object")
        attempt_id = rollout.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id or attempt_id in seen_attempts:
            raise ValueError("Primary rollout attempt identity is missing or duplicated")
        seen_attempts.add(attempt_id)
        if rollout.get("scenario_id") not in scenario_ids:
            raise ValueError(f"Primary rollout {attempt_id} has an unknown scenario identity")
        shot = _published_path(root, rollout.get("path"), f"primary rollout {attempt_id}")
        if not shot.is_dir():
            raise ValueError(f"Primary rollout is not a directory: {shot}")
        inventory = rollout.get("files")
        if not isinstance(inventory, list) or not inventory:
            raise ValueError(f"Primary rollout {attempt_id} has no file inventory")
        for reference in inventory:
            _verify_file_reference(
                root, reference, f"primary rollout file for {attempt_id}", check_size=True
            )

        capture = load_physics_capture(
            shot / "physics_state.jsonl", shot / "physics_events.jsonl"
        )
        frame_paths = tuple(state.rgb_frame.relative_path for state in capture.states)
        frames = read_physics_shot(
            shot,
            str(capture.states[0].clock.shot_id),
            frame_paths,
            PhysicsSupervisionRequest(include_raw_contacts=True, include_events=True),
        )
        metadata = _load(shot / "metadata.json", f"metadata for {attempt_id}")
        if (
            len(frames) != len(capture.states)
            or not frames
            or frames[-1].fixed_step != metadata.get("terminal_state_fixed_step")
        ):
            raise ValueError(f"Primary rollout {attempt_id} lost its terminal observation")

        macro = validate_macro_labels(
            shot, derivation_by_attempt.get((attempt_id, "physics_macro_labels_v1"))
        )
        relational = validate_relational_supervision(
            shot, derivation_by_attempt.get((attempt_id, "physics_relational_supervision_v1"))
        )
        if len(macro.frames) != len(frames) or len(relational.frames) != len(frames):
            raise ValueError(f"Authoritative sidecars do not align with rollout {attempt_id}")
        source_identities = tuple(
            (
                str(state.clock.capture_id),
                str(state.clock.shot_id),
                state.clock.sequence,
                state.clock.render_frame,
                state.clock.fixed_step,
                state.rgb_frame.relative_path,
            )
            for state in capture.states
        )
        macro_identities = tuple(
            (
                label.identity.capture_id,
                label.identity.shot_id,
                label.identity.state_sequence,
                label.identity.render_frame,
                label.identity.fixed_step,
                label.identity.rgb_relative_path,
            )
            for label in macro.frames
        )
        relational_identities = tuple(
            (
                label.identity.capture_id,
                label.identity.shot_id,
                label.identity.state_sequence,
                label.identity.render_frame,
                label.identity.fixed_step,
                label.identity.rgb_relative_path,
            )
            for label in relational.frames
        )
        reader_identities = tuple(
            (str(frame.shot_id), frame.render_frame, frame.fixed_step) for frame in frames
        )
        capture_reader_identities = tuple(
            (str(state.clock.shot_id), state.clock.render_frame, state.clock.fixed_step)
            for state in capture.states
        )
        if (
            source_identities != macro_identities
            or source_identities != relational_identities
            or reader_identities != capture_reader_identities
        ):
            raise ValueError(f"Authoritative identities changed during ingestion for {attempt_id}")

        source_events = tuple(
            (str(event.event_id), event.clock.fixed_step) for event in capture.events
        )
        exposed_events = tuple(
            (str(event.event_id), event.fixed_step)
            for frame in frames
            for event in frame.events
        )
        if exposed_events != source_events:
            raise ValueError(f"Fixed-step event alignment changed during ingestion for {attempt_id}")

        unavailable_label_count += sum(
            label.availability is not RelationalAvailability.AVAILABLE
            for frame in relational.frames
            for label in (
                *frame.supports,
                frame.physical_regime_eligibility,
                frame.model_relative_micro_relation_usefulness,
            )
        )
        identity_count += len(source_identities)
        aligned_event_count += len(source_events)
        frame_count += len(frames)
        event_count += len(capture.events)
        terminal_count += 1
        macro_count += len(macro.frames)
        relational_count += len(relational.frames)

    expected_keys = {
        (attempt_id, kind)
        for attempt_id in seen_attempts
        for kind in ("physics_macro_labels_v1", "physics_relational_supervision_v1")
    }
    if set(derivation_by_attempt) != expected_keys:
        raise ValueError("Authoritative derivation inventory does not match primary rollouts")

    release_version = release.get("release_version")
    derivation_version = derivations.get("derivation_version")
    if (
        isinstance(release_version, bool)
        or not isinstance(release_version, int)
        or isinstance(derivation_version, bool)
        or not isinstance(derivation_version, int)
    ):
        raise ValueError("Cohort release or derivation version is malformed")
    return CohortIngestionEvidence(
        publication_identity=str(publication["identity"]),
        cohort_release_identity=str(release["identity"]),
        cohort_release_version=release_version,
        derivation_identity=str(derivations["identity"]),
        derivation_version=derivation_version,
        partition_identity=partition.identity,
        partition_version=partition.partition_version,
        readers=INGESTION_READERS,
        required_capabilities=required_capabilities,
        unavailable_capabilities=MappingProxyType(dict(sorted(unavailable.items()))),
        rollout_count=len(rollouts),
        frame_count=frame_count,
        event_count=event_count,
        identity_aligned_frame_count=identity_count,
        fixed_step_aligned_event_count=aligned_event_count,
        available_capabilities=MappingProxyType(dict(sorted(available.items()))),
        unavailable_relational_label_count=unavailable_label_count,
        terminal_observation_count=terminal_count,
        macro_frame_count=macro_count,
        relational_frame_count=relational_count,
    )


def write_cohort_ingestion_evidence(
    evidence: CohortIngestionEvidence, path: Path
) -> Path:
    if type(evidence) is not CohortIngestionEvidence:
        raise ValueError("evidence must be CohortIngestionEvidence")
    return _write(Path(path), evidence.to_dict())


def _pairs(values: Sequence[str], option: str) -> dict[str, str]:
    result = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or not item or key in result:
            raise ValueError(f"{option} values must use unique NAME=VALUE pairs")
        result[key] = item
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish an immutable representative cohort release")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--partition-manifest", required=True, type=Path)
    parser.add_argument("--scenario-manifest", action="append", required=True, metavar="SCENARIO_ID=PATH")
    parser.add_argument("--release-version", required=True, type=int)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--available-capability", action="append", default=[])
    parser.add_argument("--unavailable-capability", action="append", default=[])
    parser.add_argument("--prior-execution", action="append", default=[], metavar="NAME=PATH")
    args = parser.parse_args(argv)
    try:
        path = publish_cohort_release(args.output_dir, partition_manifest_path=args.partition_manifest, scenario_manifest_paths={key: Path(value) for key, value in _pairs(args.scenario_manifest, "--scenario-manifest").items()}, release_version=args.release_version, code_revision=args.code_revision, available_capabilities=_pairs(args.available_capability, "--available-capability"), unavailable_capabilities=_pairs(args.unavailable_capability, "--unavailable-capability"), prior_execution_paths={key: Path(value) for key, value in _pairs(args.prior_execution, "--prior-execution").items()})
        publication = verify_cohort_publication(path)
    except (OSError, ValueError) as error:
        print(str(error), file=os.sys.stderr)
        return 2
    print(json.dumps({"publication": str(path), "identity": publication["identity"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
