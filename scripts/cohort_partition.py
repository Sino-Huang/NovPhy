"""Deterministic, provenance-bound cohort partition manifests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Literal, TypeAlias
from urllib.parse import quote

from scripts.scenario_manifest import (
    SCENARIO_MANIFEST_PROJECTION_FIELDS,
    load_scenario_manifest_projection,
    require_research_eligible,
)


SCHEMA = "cohort_partition_manifest_v1"
IDENTITY_NAMESPACE = "cohort-partition-manifest-v1"
ExposureRole: TypeAlias = Literal[
    "training",
    "calibration",
    "model_selection",
    "final_evaluation",
]
SplitRegime: TypeAlias = Literal["instance_held_out", "template_held_out"]
ArtifactKind: TypeAlias = Literal[
    "derivation_artifact",
    "generation_seed",
    "intervention",
    "observation_configuration",
    "observation_variant",
    "replay",
    "rerun",
]

EXPOSURE_ROLES: tuple[ExposureRole, ...] = (
    "training",
    "calibration",
    "model_selection",
    "final_evaluation",
)
SPLIT_REGIMES: tuple[SplitRegime, ...] = ("instance_held_out", "template_held_out")
PROJECTION_FIELDS = ("scenario_manifest", *SCENARIO_MANIFEST_PROJECTION_FIELDS)
ENTRY_FIELDS = {"dataset_partition", "exposure_role", *PROJECTION_FIELDS}
PROVENANCE_FIELDS = {
    "artifact_kind",
    "artifact_identity",
    "consumer_scenario_lineage_identity",
    "source_scenario_lineage_identities",
}
ARTIFACT_KINDS: tuple[ArtifactKind, ...] = (
    "derivation_artifact",
    "generation_seed",
    "intervention",
    "observation_configuration",
    "observation_variant",
    "replay",
    "rerun",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _identity(*keys: Any) -> str:
    return ":".join(
        (IDENTITY_NAMESPACE, *(quote(str(key), safe="-._~") for key in keys))
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must use string keys")
    return value


def _require_exact_keys(data: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    mapping = _require_mapping(data, name)
    if set(mapping) != fields:
        raise ValueError(f"{name} is incomplete or contains unknown fields")
    return mapping


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _require_sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence")
    return value


def _freeze_json(value: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    frozen = _freeze(value)
    try:
        _canonical_json(_thaw(frozen))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain JSON-compatible values") from exc
    return frozen


@dataclass(frozen=True, slots=True)
class CohortPartitionEntry:
    dataset_partition: str
    exposure_role: ExposureRole
    scenario_manifest_projection: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_partition": self.dataset_partition,
            "exposure_role": self.exposure_role,
            **_thaw(self.scenario_manifest_projection),
        }


@dataclass(frozen=True, slots=True)
class CohortPartitionProvenanceRecord:
    artifact_kind: ArtifactKind
    artifact_identity: str
    consumer_scenario_lineage_identity: str
    source_scenario_lineage_identities: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_identity": self.artifact_identity,
            "consumer_scenario_lineage_identity": self.consumer_scenario_lineage_identity,
            "source_scenario_lineage_identities": list(self.source_scenario_lineage_identities),
        }


@dataclass(frozen=True, slots=True)
class CohortPartitionManifest:
    schema: Literal["cohort_partition_manifest_v1"]
    partition_version: int
    identity: str
    split_regime: SplitRegime
    held_out_roles: tuple[ExposureRole, ...]
    entries: tuple[CohortPartitionEntry, ...]
    provenance_records: tuple[CohortPartitionProvenanceRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "partition_version": self.partition_version,
            "identity": self.identity,
            "split_regime": self.split_regime,
            "held_out_roles": list(self.held_out_roles),
            "entries": [entry.to_dict() for entry in self.entries],
            "provenance_records": [record.to_dict() for record in self.provenance_records],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CohortPartitionManifest":
        raw = _require_exact_keys(
            data,
            {
                "schema",
                "partition_version",
                "identity",
                "split_regime",
                "held_out_roles",
                "entries",
                "provenance_records",
            },
            "Cohort partition manifest",
        )
        identity = _require_nonempty_string(raw["identity"], "Cohort partition manifest identity")
        manifest = _normalize_manifest(
            schema=raw["schema"],
            partition_version=raw["partition_version"],
            identity=identity,
            split_regime=raw["split_regime"],
            held_out_roles=raw["held_out_roles"],
            entries=raw["entries"],
            provenance_records=raw["provenance_records"],
        )
        return manifest


def _parse_entry(data: Any) -> CohortPartitionEntry:
    raw = _require_exact_keys(data, ENTRY_FIELDS, "Cohort partition entry")
    partition = _require_nonempty_string(raw["dataset_partition"], "dataset_partition")
    role = raw["exposure_role"]
    if role not in EXPOSURE_ROLES:
        raise ValueError(f"Cohort partition exposure role is unknown: {role}")

    projection = {key: raw[key] for key in PROJECTION_FIELDS}
    manifest, _ = load_scenario_manifest_projection(projection, required=True)
    if manifest is None:
        raise ValueError("Cohort partition entry requires a scenario manifest")
    require_research_eligible(manifest, "cohort partition")
    return CohortPartitionEntry(
        partition,
        role,
        _freeze_json(projection, "scenario manifest projection"),
    )


def _parse_provenance_record(data: Any) -> CohortPartitionProvenanceRecord:
    raw = _require_exact_keys(data, PROVENANCE_FIELDS, "Cohort partition provenance record")
    kind = raw["artifact_kind"]
    if kind not in ARTIFACT_KINDS:
        raise ValueError(f"Cohort partition artifact kind is unknown: {kind}")
    artifact_identity = _require_nonempty_string(raw["artifact_identity"], "artifact_identity")
    consumer = _require_nonempty_string(
        raw["consumer_scenario_lineage_identity"],
        "consumer_scenario_lineage_identity",
    )
    sources = _require_list(raw["source_scenario_lineage_identities"], "source_scenario_lineage_identities")
    if not sources:
        raise ValueError("source_scenario_lineage_identities must be nonempty")
    normalized_sources = tuple(
        sorted(
            _require_nonempty_string(source, "source_scenario_lineage_identity")
            for source in sources
        )
    )
    if len(normalized_sources) != len(set(normalized_sources)):
        raise ValueError("source scenario lineage identities must be unique")
    return CohortPartitionProvenanceRecord(kind, artifact_identity, consumer, normalized_sources)


def _normalize_manifest(
    *,
    schema: Any,
    partition_version: Any,
    identity: str,
    split_regime: Any,
    held_out_roles: Any,
    entries: Any,
    provenance_records: Any,
) -> CohortPartitionManifest:
    if schema != SCHEMA:
        raise ValueError(f"Unsupported cohort partition manifest schema: {schema}")
    if isinstance(partition_version, bool) or not isinstance(partition_version, int) or partition_version <= 0:
        raise ValueError("Cohort partition version must be a positive integer")
    if split_regime not in SPLIT_REGIMES:
        raise ValueError(f"Unsupported cohort partition split regime: {split_regime}")

    roles = _require_list(held_out_roles, "held_out_roles")
    normalized_roles = tuple(sorted(_require_nonempty_string(role, "held_out_role") for role in roles))
    if len(normalized_roles) != len(set(normalized_roles)):
        raise ValueError("held_out_roles must be unique")
    if any(role not in EXPOSURE_ROLES for role in normalized_roles):
        raise ValueError("held_out_roles contains an unknown exposure role")
    if split_regime == "instance_held_out" and normalized_roles:
        raise ValueError("instance_held_out requires held_out_roles to be empty")
    if split_regime == "template_held_out" and (
        not normalized_roles or len(normalized_roles) == len(EXPOSURE_ROLES)
    ):
        raise ValueError("template_held_out requires a nonempty proper subset of exposure roles")

    raw_entries = _require_list(entries, "entries")
    if not raw_entries:
        raise ValueError("Cohort partition manifest requires at least one entry")
    normalized_entries = tuple(sorted((_parse_entry(item) for item in raw_entries), key=_entry_sort_key))

    raw_records = _require_list(provenance_records, "provenance_records")
    normalized_records = tuple(
        sorted((_parse_provenance_record(item) for item in raw_records), key=_provenance_sort_key)
    )
    manifest = CohortPartitionManifest(
        SCHEMA,
        partition_version,
        identity,
        split_regime,
        normalized_roles,
        normalized_entries,
        normalized_records,
    )
    _validate_invariants(manifest)
    return manifest


def _entry_sort_key(entry: CohortPartitionEntry) -> bytes:
    return _canonical_json(entry.to_dict())


def _provenance_sort_key(record: CohortPartitionProvenanceRecord) -> bytes:
    return _canonical_json(record.to_dict())


def _normalize_admitted_lineage_inventory(value: Sequence[str]) -> tuple[str, ...]:
    identities = _require_sequence(value, "admitted_scenario_lineage_identities")
    if not identities:
        raise ValueError("admitted scenario lineage inventory must be nonempty")
    normalized = tuple(
        sorted(
            _require_nonempty_string(identity, "admitted_scenario_lineage_identity")
            for identity in identities
        )
    )
    if len(normalized) != len(set(normalized)):
        raise ValueError("admitted scenario lineage inventory must be unique")
    return normalized


def _normalize_admitted_provenance_inventory(
    value: Sequence[Mapping[str, Any]],
) -> tuple[CohortPartitionProvenanceRecord, ...]:
    records = _require_sequence(value, "admitted_provenance_records")
    normalized = tuple(sorted((_parse_provenance_record(item) for item in records), key=_provenance_sort_key))
    artifact_identities = [record.artifact_identity for record in normalized]
    if len(artifact_identities) != len(set(artifact_identities)):
        raise ValueError("admitted provenance artifact identities must be unique")
    return normalized


def _identity_payload(manifest: CohortPartitionManifest) -> dict[str, Any]:
    payload = manifest.to_dict()
    del payload["identity"]
    return payload


def _manifest_identity(manifest: CohortPartitionManifest) -> str:
    return _identity(
        manifest.partition_version,
        manifest.split_regime,
        *(
            entry.scenario_manifest_projection["scenario_lineage_identity"]
            for entry in manifest.entries
        ),
    )


def _validate_invariants(manifest: CohortPartitionManifest) -> None:
    lineage_entries: dict[str, CohortPartitionEntry] = {}
    partition_roles: dict[str, str] = {}
    level_assignments: dict[str, tuple[str, str]] = {}

    for item in manifest.entries:
        projection = item.scenario_manifest_projection
        lineage = projection["scenario_lineage_identity"]
        if lineage in lineage_entries:
            raise ValueError("scenario lineage identities must be unique")
        lineage_entries[lineage] = item

        prior_role = partition_roles.setdefault(item.dataset_partition, item.exposure_role)
        if prior_role != item.exposure_role:
            raise ValueError("each dataset partition must map to one exposure role")

        level_instance = projection["level_instance_identity"]
        assignment = (item.dataset_partition, item.exposure_role)
        prior_assignment = level_assignments.setdefault(level_instance, assignment)
        if prior_assignment != assignment:
            raise ValueError("a level instance must not map across roles or partitions")

    if manifest.split_regime == "template_held_out":
        template_sides: dict[str, set[bool]] = {}
        held_out = set(manifest.held_out_roles)
        for item in manifest.entries:
            template = item.scenario_manifest_projection["scenario_template_identity"]
            if not isinstance(template, str) or not template:
                raise ValueError("template-held-out entries require an available template identity")
            template_sides.setdefault(template, set()).add(item.exposure_role in held_out)
        if any(len(sides) > 1 for sides in template_sides.values()):
            raise ValueError("scenario template crosses the template-held-out boundary")

    artifact_identities: set[str] = set()
    for record in manifest.provenance_records:
        if record.artifact_identity in artifact_identities:
            raise ValueError("provenance artifact identities must be unique")
        artifact_identities.add(record.artifact_identity)

        consumer = lineage_entries.get(record.consumer_scenario_lineage_identity)
        if consumer is None:
            raise ValueError("provenance consumer scenario lineage identity is unknown")
        expected_assignment = (consumer.dataset_partition, consumer.exposure_role)
        for source_lineage in record.source_scenario_lineage_identities:
            source = lineage_entries.get(source_lineage)
            if source is None:
                raise ValueError("provenance source scenario lineage identity is unknown")
            if (source.dataset_partition, source.exposure_role) != expected_assignment:
                raise ValueError(
                    "provenance sources must match the consumer's same dataset partition and exposure role"
                )


def create_cohort_partition_manifest(
    *,
    partition_version: int,
    split_regime: SplitRegime,
    held_out_roles: Sequence[ExposureRole],
    entries: Sequence[Mapping[str, Any]],
    provenance_records: Sequence[Mapping[str, Any]],
) -> CohortPartitionManifest:
    manifest = _normalize_manifest(
        schema=SCHEMA,
        partition_version=partition_version,
        identity="",
        split_regime=split_regime,
        held_out_roles=list(held_out_roles),
        entries=list(entries),
        provenance_records=list(provenance_records),
    )
    return replace(manifest, identity=_manifest_identity(manifest))


def audit_cohort_partition_manifest(
    manifest: CohortPartitionManifest,
    *,
    admitted_scenario_lineage_identities: Sequence[str],
    admitted_provenance_records: Sequence[Mapping[str, Any]],
) -> None:
    """Require complete admitted lineage and provenance inventories for a manifest."""
    if not isinstance(manifest, CohortPartitionManifest):
        raise ValueError("audit_cohort_partition_manifest requires a CohortPartitionManifest")
    validated = CohortPartitionManifest.from_dict(manifest.to_dict())

    admitted_lineages = _normalize_admitted_lineage_inventory(admitted_scenario_lineage_identities)
    manifest_lineages = tuple(
        sorted(entry.scenario_manifest_projection["scenario_lineage_identity"] for entry in validated.entries)
    )
    if admitted_lineages != manifest_lineages:
        raise ValueError("admitted scenario lineage inventory mismatch")

    admitted_records = _normalize_admitted_provenance_inventory(admitted_provenance_records)
    _validate_invariants(replace(validated, provenance_records=admitted_records))
    if admitted_records != validated.provenance_records:
        raise ValueError("admitted provenance inventory mismatch")


def write_cohort_partition_manifest(manifest: CohortPartitionManifest, path: Path) -> Path:
    if not isinstance(manifest, CohortPartitionManifest):
        raise ValueError("write_cohort_partition_manifest requires a CohortPartitionManifest")
    validated = CohortPartitionManifest.from_dict(manifest.to_dict())
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(validated.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, target)
    return target


def load_cohort_partition_manifest(path: Path) -> CohortPartitionManifest:
    target = Path(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load cohort partition manifest {target}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("Cohort partition manifest root must be an object")
    return CohortPartitionManifest.from_dict(data)
