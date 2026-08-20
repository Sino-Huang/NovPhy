"""Fail-closed final-evaluation access manifests and audits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Literal
from urllib.parse import quote

from scripts.cohort_partition import CohortPartitionManifest


SCHEMA = "final_evaluation_access_manifest_v1"
IDENTITY_NAMESPACE = "final-evaluation-access-manifest-v1"
ORDINARY_WORKFLOW_KINDS = frozenset((
    "training",
    "calibration",
    "model_selection",
    "pilot",
))


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _identity(value: Mapping[str, Any]) -> str:
    return ":".join((
        IDENTITY_NAMESPACE,
        str(value["access_version"]),
        quote(str(value["partition_identity"]), safe="-._~"),
        quote(str(value["workflow_identity"]), safe="-._~"),
    ))


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list")
    normalized = tuple(sorted(_string(item, name) for item in value))
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must be unique")
    return normalized


@dataclass(frozen=True, slots=True)
class AuthorizedFinalArtifact:
    artifact_kind: str
    artifact_identity: str
    source_scenario_lineage_identities: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_identity": self.artifact_identity,
            "source_scenario_lineage_identities": list(
                self.source_scenario_lineage_identities
            ),
        }


@dataclass(frozen=True, slots=True)
class FinalEvaluationAccessManifest:
    schema: Literal["final_evaluation_access_manifest_v1"]
    access_version: int
    identity: str
    partition_identity: str
    workflow_identity: str
    final_evaluation_lineage_identities: tuple[str, ...]
    authorized_artifacts: tuple[AuthorizedFinalArtifact, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "access_version": self.access_version,
            "identity": self.identity,
            "partition_identity": self.partition_identity,
            "workflow_identity": self.workflow_identity,
            "final_evaluation_lineage_identities": list(
                self.final_evaluation_lineage_identities
            ),
            "authorized_artifacts": [
                artifact.to_dict() for artifact in self.authorized_artifacts
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FinalEvaluationAccessManifest":
        fields = {
            "schema",
            "access_version",
            "identity",
            "partition_identity",
            "workflow_identity",
            "final_evaluation_lineage_identities",
            "authorized_artifacts",
        }
        if not isinstance(data, Mapping) or set(data) != fields:
            raise ValueError("Final-evaluation access manifest fields are invalid")
        if data["schema"] != SCHEMA:
            raise ValueError("Unsupported final-evaluation access manifest schema")
        version = data["access_version"]
        if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
            raise ValueError("Final-evaluation access version must be positive")
        raw_artifacts = data["authorized_artifacts"]
        if not isinstance(raw_artifacts, list):
            raise ValueError("authorized_artifacts must be a list")
        artifacts = []
        for item in raw_artifacts:
            if not isinstance(item, Mapping) or set(item) != {
                "artifact_kind",
                "artifact_identity",
                "source_scenario_lineage_identities",
            }:
                raise ValueError("Authorized final artifact fields are invalid")
            artifacts.append(
                AuthorizedFinalArtifact(
                    _string(item["artifact_kind"], "artifact_kind"),
                    _string(item["artifact_identity"], "artifact_identity"),
                    _strings(
                        item["source_scenario_lineage_identities"],
                        "source_scenario_lineage_identities",
                    ),
                )
            )
        artifacts.sort(key=lambda item: _canonical_json(item.to_dict()))
        if len({item.artifact_identity for item in artifacts}) != len(artifacts):
            raise ValueError("Authorized final artifact identities must be unique")
        manifest = cls(
            SCHEMA,
            version,
            _string(data["identity"], "identity"),
            _string(data["partition_identity"], "partition_identity"),
            _string(data["workflow_identity"], "workflow_identity"),
            _strings(
                data["final_evaluation_lineage_identities"],
                "final_evaluation_lineage_identities",
            ),
            tuple(artifacts),
        )
        return manifest


def create_final_evaluation_access_manifest(
    partition: CohortPartitionManifest,
    *,
    access_version: int,
    workflow_identity: str,
) -> FinalEvaluationAccessManifest:
    validated = CohortPartitionManifest.from_dict(partition.to_dict())
    final_lineages = tuple(
        sorted(
            str(entry.scenario_manifest_projection["scenario_lineage_identity"])
            for entry in validated.entries
            if entry.exposure_role == "final_evaluation"
        )
    )
    if not final_lineages:
        raise ValueError("Partition has no final_evaluation lineages")
    final_set = set(final_lineages)
    role_by_lineage = {
        str(entry.scenario_manifest_projection["scenario_lineage_identity"]):
        entry.exposure_role
        for entry in validated.entries
    }
    authorized = []
    for record in validated.provenance_records:
        sources = set(record.source_scenario_lineage_identities)
        if not sources & final_set:
            continue
        if not sources <= final_set:
            raise ValueError(
                "Final-evaluation provenance cannot mix final and non-final sources"
            )
        if role_by_lineage.get(record.consumer_scenario_lineage_identity) != "final_evaluation":
            raise ValueError(
                "Final-evaluation provenance consumer must have final_evaluation role"
            )
        authorized.append(
            AuthorizedFinalArtifact(
                record.artifact_kind,
                record.artifact_identity,
                record.source_scenario_lineage_identities,
            )
        )
    authorized.sort(key=lambda item: _canonical_json(item.to_dict()))
    payload = {
        "schema": SCHEMA,
        "access_version": access_version,
        "identity": "",
        "partition_identity": validated.identity,
        "workflow_identity": _string(workflow_identity, "workflow_identity"),
        "final_evaluation_lineage_identities": list(final_lineages),
        "authorized_artifacts": [item.to_dict() for item in authorized],
    }
    if isinstance(access_version, bool) or not isinstance(access_version, int) or access_version <= 0:
        raise ValueError("Final-evaluation access version must be positive")
    payload["identity"] = _identity(payload)
    return FinalEvaluationAccessManifest.from_dict(payload)


def audit_ordinary_workflow_access(
    partition: CohortPartitionManifest,
    *,
    workflow_kind: str,
    observed_scenario_lineage_identities: Sequence[str],
    observed_artifact_accesses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed if an ordinary workflow reads final or undeclared data."""
    if workflow_kind not in ORDINARY_WORKFLOW_KINDS:
        raise ValueError("Ordinary workflow kind is unknown")
    validated = CohortPartitionManifest.from_dict(partition.to_dict())
    role_by_lineage = {
        str(entry.scenario_manifest_projection["scenario_lineage_identity"]):
        entry.exposure_role
        for entry in validated.entries
    }
    observed_lineages = _strings(
        observed_scenario_lineage_identities,
        "observed_scenario_lineage_identities",
    )
    for lineage in observed_lineages:
        role = role_by_lineage.get(lineage)
        if role is None:
            raise ValueError("Ordinary workflow referenced an undeclared lineage")
        if role == "final_evaluation":
            raise ValueError("Ordinary workflow has final_evaluation lineage access")

    provenance_by_artifact = {
        record.artifact_identity: record
        for record in validated.provenance_records
    }
    observed_artifacts = []
    for raw in observed_artifact_accesses:
        fields = {
            "artifact_identity",
            "source_scenario_lineage_identities",
        }
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise ValueError("Observed ordinary artifact access fields are invalid")
        artifact_identity = _string(raw["artifact_identity"], "artifact_identity")
        observed_artifacts.append(artifact_identity)
        record = provenance_by_artifact.get(artifact_identity)
        if record is None:
            raise ValueError("Ordinary workflow referenced an undeclared artifact")
        sources = _strings(
            raw["source_scenario_lineage_identities"],
            "source_scenario_lineage_identities",
        )
        if sources != record.source_scenario_lineage_identities:
            raise ValueError("Ordinary workflow artifact source provenance differs")
        if any(role_by_lineage[source] == "final_evaluation" for source in sources):
            raise ValueError("Ordinary workflow artifact uses final_evaluation data")
    if len(observed_artifacts) != len(set(observed_artifacts)):
        raise ValueError("Observed ordinary artifact access is duplicated")
    return {
        "schema": "ordinary_workflow_access_audit_v1",
        "partition_identity": validated.identity,
        "workflow_kind": workflow_kind,
        "observed_lineage_count": len(observed_lineages),
        "observed_artifact_count": len(observed_artifacts),
        "passed": True,
    }


def audit_final_evaluation_access(
    partition: CohortPartitionManifest,
    access: FinalEvaluationAccessManifest,
    *,
    observed_accesses: Sequence[Mapping[str, Any]],
    ordinary_workflow_lineage_identities: Sequence[str] = (),
) -> dict[str, Any]:
    validated_partition = CohortPartitionManifest.from_dict(partition.to_dict())
    validated_access = FinalEvaluationAccessManifest.from_dict(access.to_dict())
    if validated_access.partition_identity != validated_partition.identity:
        raise ValueError("Final-evaluation access manifest targets a different partition")
    final_lineages = {
        str(entry.scenario_manifest_projection["scenario_lineage_identity"])
        for entry in validated_partition.entries
        if entry.exposure_role == "final_evaluation"
    }
    if set(validated_access.final_evaluation_lineage_identities) != final_lineages:
        raise ValueError("Final-evaluation lineage inventory is stale")
    ordinary = set(_strings(
        ordinary_workflow_lineage_identities,
        "ordinary_workflow_lineage_identities",
    ))
    if ordinary & final_lineages:
        raise ValueError("Ordinary workflow has final_evaluation lineage access")

    authorized = {
        item.artifact_identity: item
        for item in validated_access.authorized_artifacts
    }
    observed_identities = []
    for raw in observed_accesses:
        fields = {
            "artifact_identity",
            "workflow_identity",
            "consumer_exposure_role",
            "source_scenario_lineage_identities",
        }
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise ValueError("Observed final-evaluation access fields are invalid")
        artifact_identity = _string(raw["artifact_identity"], "artifact_identity")
        observed_identities.append(artifact_identity)
        if raw["consumer_exposure_role"] != "final_evaluation":
            raise ValueError("Final-evaluation data reached an ordinary workflow")
        if raw["workflow_identity"] != validated_access.workflow_identity:
            raise ValueError("Final-evaluation access used an undeclared workflow")
        artifact = authorized.get(artifact_identity)
        if artifact is None:
            raise ValueError("Final-evaluation access referenced an undeclared artifact")
        sources = _strings(
            raw["source_scenario_lineage_identities"],
            "source_scenario_lineage_identities",
        )
        if sources != artifact.source_scenario_lineage_identities:
            raise ValueError("Final-evaluation access source provenance differs")
    if len(observed_identities) != len(set(observed_identities)):
        raise ValueError("Observed final-evaluation artifact access is duplicated")
    return {
        "schema": "final_evaluation_access_audit_v1",
        "partition_identity": validated_partition.identity,
        "access_manifest_identity": validated_access.identity,
        "workflow_identity": validated_access.workflow_identity,
        "authorized_artifact_count": len(authorized),
        "observed_artifact_count": len(observed_identities),
        "passed": True,
    }


def write_final_evaluation_access_manifest(
    manifest: FinalEvaluationAccessManifest,
    path: Path,
) -> Path:
    validated = FinalEvaluationAccessManifest.from_dict(manifest.to_dict())
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(validated.to_dict(), indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, target)
    return target


def load_final_evaluation_access_manifest(path: Path) -> FinalEvaluationAccessManifest:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load final-evaluation access manifest: {error}") from error
    if not isinstance(data, Mapping):
        raise ValueError("Final-evaluation access manifest root must be an object")
    return FinalEvaluationAccessManifest.from_dict(data)
