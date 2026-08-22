"""Public sealed projection of the central-v2 exposure partition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Any, Literal, TypeAlias
from urllib.parse import quote

from scripts.cohort_v2_scenarios import ScenarioInventoryEntry


SCHEMA = "cohort_v2_partition_exposure_manifest_v1"
IDENTITY_NAMESPACE = "cohort-v2-partition-exposure-manifest-v1"
SPLIT_REGIME = "instance_held_out"
QUOTA_SCOPE = "partition_lineage_membership"
PRODUCTION_QUOTA_STATUS = "deferred_to_representative_pilot_and_collection_plan"
TEMPLATE_POLICY = "reuse_allowed_audit_only_no_template_held_out_claim"
PROVENANCE_ARTIFACT_KINDS = frozenset(
    (
        "derivation_artifact",
        "generation_seed",
        "intervention",
        "observation_configuration",
        "observation_variant",
        "replay",
        "rerun",
    )
)
CENTRAL_EVIDENCE_FLOOR = {
    "minimum_level_instances": 2,
    "minimum_non_final_scenario_lineages": 2,
    "minimum_scenario_templates": 2,
}

ExposureRole: TypeAlias = Literal[
    "training", "calibration", "model_selection", "final_evaluation"
]
InventoryState: TypeAlias = Literal["planned_non_final", "sealed_final"]

EXPOSURE_ROLES: tuple[ExposureRole, ...] = (
    "training",
    "calibration",
    "model_selection",
    "final_evaluation",
)

ROLE_PERMISSIONS: Mapping[ExposureRole, tuple[str, ...]] = {
    "training": ("learned_parameters",),
    "calibration": ("pilot_values", "threshold_values", "tolerance_values"),
    "model_selection": ("configuration_selection",),
    "final_evaluation": ("frozen_final_metrics_after_authorization",),
}


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class CohortV2PartitionExposureEntry:
    dataset_partition: str
    exposure_role: ExposureRole
    inventory_state: InventoryState
    lineage_quota: int
    may_influence: tuple[str, ...]
    scenario_manifest_identity: str
    benchmark_condition_identity: str
    scenario_template_identity: str
    level_instance_identity: str
    scenario_specification_identity: str
    scenario_lineage_identity: str
    declared_initial_engine_state_identity: str
    scenario_manifest_reference: str | None = None
    sealed_scenario_manifest_reference: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = {
            "dataset_partition": self.dataset_partition,
            "exposure_role": self.exposure_role,
            "inventory_state": self.inventory_state,
            "lineage_quota": self.lineage_quota,
            "may_influence": list(self.may_influence),
            "scenario_manifest_identity": self.scenario_manifest_identity,
            "benchmark_condition_identity": self.benchmark_condition_identity,
            "scenario_template_identity": self.scenario_template_identity,
            "level_instance_identity": self.level_instance_identity,
            "scenario_specification_identity": self.scenario_specification_identity,
            "scenario_lineage_identity": self.scenario_lineage_identity,
            "declared_initial_engine_state_identity": (
                self.declared_initial_engine_state_identity
            ),
        }
        if self.exposure_role == "final_evaluation":
            value["sealed_scenario_manifest_reference"] = (
                self.sealed_scenario_manifest_reference
            )
        else:
            value["scenario_manifest_reference"] = self.scenario_manifest_reference
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CohortV2PartitionExposureEntry":
        common = {
            "dataset_partition",
            "exposure_role",
            "inventory_state",
            "lineage_quota",
            "may_influence",
            "scenario_manifest_identity",
            "benchmark_condition_identity",
            "scenario_template_identity",
            "level_instance_identity",
            "scenario_specification_identity",
            "scenario_lineage_identity",
            "declared_initial_engine_state_identity",
        }
        if not isinstance(value, Mapping) or not common.issubset(value):
            raise ValueError("Cohort-v2 partition entry fields are invalid")
        role = value["exposure_role"]
        if role not in EXPOSURE_ROLES:
            raise ValueError("Cohort-v2 partition exposure role is unknown")
        reference_field = (
            "sealed_scenario_manifest_reference"
            if role == "final_evaluation"
            else "scenario_manifest_reference"
        )
        if set(value) != common | {reference_field}:
            raise ValueError("Cohort-v2 partition entry fields are invalid")
        expected_state = (
            "sealed_final" if role == "final_evaluation" else "planned_non_final"
        )
        if value["inventory_state"] != expected_state:
            raise ValueError("Cohort-v2 partition entry has an invalid sealing state")
        expected_partition = f"central-v2-{role.replace('_', '-')}"
        if value["dataset_partition"] != expected_partition:
            raise ValueError("Cohort-v2 partition name does not match its exposure role")
        raw_permissions = value["may_influence"]
        if not isinstance(raw_permissions, list) or tuple(raw_permissions) != ROLE_PERMISSIONS[role]:
            raise ValueError("Cohort-v2 exposure permissions differ from the frozen role")
        identity_fields = common - {
            "exposure_role",
            "inventory_state",
            "lineage_quota",
            "may_influence",
        }
        identities = {
            field: _string(value[field], field)
            for field in identity_fields
        }
        reference = _string(value[reference_field], reference_field)
        return cls(
            dataset_partition=identities["dataset_partition"],
            exposure_role=role,
            inventory_state=expected_state,
            lineage_quota=_positive_integer(value["lineage_quota"], "lineage_quota"),
            may_influence=ROLE_PERMISSIONS[role],
            scenario_manifest_identity=identities["scenario_manifest_identity"],
            benchmark_condition_identity=identities["benchmark_condition_identity"],
            scenario_template_identity=identities["scenario_template_identity"],
            level_instance_identity=identities["level_instance_identity"],
            scenario_specification_identity=identities["scenario_specification_identity"],
            scenario_lineage_identity=identities["scenario_lineage_identity"],
            declared_initial_engine_state_identity=identities[
                "declared_initial_engine_state_identity"
            ],
            scenario_manifest_reference=(
                reference if role != "final_evaluation" else None
            ),
            sealed_scenario_manifest_reference=(
                reference if role == "final_evaluation" else None
            ),
        )


@dataclass(frozen=True, slots=True)
class CohortV2PartitionExposureManifest:
    schema: Literal["cohort_v2_partition_exposure_manifest_v1"]
    partition_version: int
    identity: str
    source_inventory_identity: str
    source_inventory_review_url: str
    split_regime: Literal["instance_held_out"]
    quota_scope: Literal["partition_lineage_membership"]
    production_quota_status: Literal[
        "deferred_to_representative_pilot_and_collection_plan"
    ]
    central_evidence_floor: Mapping[str, int]
    template_policy: Literal["reuse_allowed_audit_only_no_template_held_out_claim"]
    entries: tuple[CohortV2PartitionExposureEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "partition_version": self.partition_version,
            "identity": self.identity,
            "source_inventory_identity": self.source_inventory_identity,
            "source_inventory_review_url": self.source_inventory_review_url,
            "split_regime": self.split_regime,
            "quota_scope": self.quota_scope,
            "production_quota_status": self.production_quota_status,
            "central_evidence_floor": dict(self.central_evidence_floor),
            "template_policy": self.template_policy,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CohortV2PartitionExposureManifest":
        fields = {
            "schema",
            "partition_version",
            "identity",
            "source_inventory_identity",
            "source_inventory_review_url",
            "split_regime",
            "quota_scope",
            "production_quota_status",
            "central_evidence_floor",
            "template_policy",
            "entries",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("Cohort-v2 partition manifest fields are invalid")
        if (
            value["schema"] != SCHEMA
            or value["split_regime"] != SPLIT_REGIME
            or value["quota_scope"] != QUOTA_SCOPE
            or value["production_quota_status"] != PRODUCTION_QUOTA_STATUS
            or value["template_policy"] != TEMPLATE_POLICY
        ):
            raise ValueError("Cohort-v2 partition manifest metadata is invalid")
        if value["central_evidence_floor"] != CENTRAL_EVIDENCE_FLOOR:
            raise ValueError("Cohort-v2 partition central evidence floor is invalid")
        raw_entries = value["entries"]
        if not isinstance(raw_entries, list):
            raise ValueError("Cohort-v2 partition entries must be a list")
        entries = tuple(CohortV2PartitionExposureEntry.from_dict(item) for item in raw_entries)
        _validate_entries(entries)
        ordered = tuple(
            sorted(entries, key=lambda item: EXPOSURE_ROLES.index(item.exposure_role))
        )
        if entries != ordered:
            raise ValueError("Cohort-v2 partition entries are not in role order")
        version = _positive_integer(value["partition_version"], "partition_version")
        source_inventory_identity = _string(
            value["source_inventory_identity"], "source_inventory_identity"
        )
        expected_identity = _manifest_identity(
            partition_version=version,
            source_inventory_identity=source_inventory_identity,
            entries=entries,
        )
        if value["identity"] != expected_identity:
            raise ValueError("Cohort-v2 partition manifest identity is stale")
        return cls(
            schema=SCHEMA,
            partition_version=version,
            identity=expected_identity,
            source_inventory_identity=source_inventory_identity,
            source_inventory_review_url=_string(
                value["source_inventory_review_url"], "source_inventory_review_url"
            ),
            split_regime=SPLIT_REGIME,
            quota_scope=QUOTA_SCOPE,
            production_quota_status=PRODUCTION_QUOTA_STATUS,
            central_evidence_floor=MappingProxyType(dict(CENTRAL_EVIDENCE_FLOOR)),
            template_policy=TEMPLATE_POLICY,
            entries=entries,
        )


def _validate_entries(entries: Sequence[CohortV2PartitionExposureEntry]) -> None:
    if {entry.exposure_role for entry in entries} != set(EXPOSURE_ROLES):
        raise ValueError("Cohort-v2 partition must contain every exposure role")
    lineages = [entry.scenario_lineage_identity for entry in entries]
    instances = [entry.level_instance_identity for entry in entries]
    if len(lineages) != len(set(lineages)):
        raise ValueError("Cohort-v2 partition scenario lineages must be unique")
    if len(instances) != len(set(instances)):
        raise ValueError("Cohort-v2 partition level instances must be unique")
    for role in EXPOSURE_ROLES:
        assigned = sum(entry.exposure_role == role for entry in entries)
        quota = next(entry.lineage_quota for entry in entries if entry.exposure_role == role)
        if assigned != quota:
            raise ValueError("Cohort-v2 partition lineage quota does not match membership")
    non_final = [
        entry for entry in entries if entry.exposure_role != "final_evaluation"
    ]
    if (
        len(non_final)
        < CENTRAL_EVIDENCE_FLOOR["minimum_non_final_scenario_lineages"]
        or len({entry.level_instance_identity for entry in non_final})
        < CENTRAL_EVIDENCE_FLOOR["minimum_level_instances"]
        or len({entry.scenario_template_identity for entry in non_final})
        < CENTRAL_EVIDENCE_FLOOR["minimum_scenario_templates"]
    ):
        raise ValueError("Cohort-v2 partition does not meet the central evidence floor")


def _manifest_identity(
    *,
    partition_version: int,
    source_inventory_identity: str,
    entries: Sequence[CohortV2PartitionExposureEntry],
) -> str:
    return ":".join((
        IDENTITY_NAMESPACE,
        str(partition_version),
        quote(source_inventory_identity, safe="-._~"),
        *(
            quote(
                json.dumps(
                    entry.to_dict(),
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                safe="-._~",
            )
            for entry in entries
        ),
    ))


def create_cohort_v2_partition_exposure_manifest(
    *,
    partition_version: int,
    source_inventory_identity: str,
    source_inventory_review_url: str,
    inventory_entries: Sequence[ScenarioInventoryEntry],
    lineage_quotas: Mapping[str, int],
) -> CohortV2PartitionExposureManifest:
    normalized_version = _positive_integer(partition_version, "partition_version")
    normalized_source_inventory_identity = _string(
        source_inventory_identity, "source_inventory_identity"
    )
    if set(lineage_quotas) != set(EXPOSURE_ROLES):
        raise ValueError("Cohort-v2 lineage quotas must cover every exposure role")
    source_entries = tuple(
        ScenarioInventoryEntry.from_dict(entry.to_dict()) for entry in inventory_entries
    )
    entries = []
    for role in EXPOSURE_ROLES:
        role_entries = [entry for entry in source_entries if entry.exposure_role == role]
        quota = _positive_integer(lineage_quotas[role], f"{role} lineage quota")
        if len(role_entries) != quota:
            raise ValueError("Cohort-v2 lineage quota does not match source inventory")
        for source in role_entries:
            value = {
                "dataset_partition": f"central-v2-{role.replace('_', '-')}",
                "exposure_role": role,
                "inventory_state": source.inventory_state,
                "lineage_quota": quota,
                "may_influence": list(ROLE_PERMISSIONS[role]),
                "scenario_manifest_identity": source.scenario_manifest_identity,
                "benchmark_condition_identity": source.benchmark_condition_identity,
                "scenario_template_identity": source.scenario_template_identity,
                "level_instance_identity": source.level_instance_identity,
                "scenario_specification_identity": source.scenario_specification_identity,
                "scenario_lineage_identity": source.scenario_lineage_identity,
                "declared_initial_engine_state_identity": (
                    source.declared_initial_engine_state_identity
                ),
            }
            if role == "final_evaluation":
                value["sealed_scenario_manifest_reference"] = (
                    source.sealed_scenario_manifest_reference
                )
            else:
                value["scenario_manifest_reference"] = source.scenario_manifest_reference
            entries.append(CohortV2PartitionExposureEntry.from_dict(value))
    manifest = CohortV2PartitionExposureManifest(
        schema=SCHEMA,
        partition_version=normalized_version,
        identity=_manifest_identity(
            partition_version=normalized_version,
            source_inventory_identity=normalized_source_inventory_identity,
            entries=entries,
        ),
        source_inventory_identity=normalized_source_inventory_identity,
        source_inventory_review_url=_string(
            source_inventory_review_url, "source_inventory_review_url"
        ),
        split_regime=SPLIT_REGIME,
        quota_scope=QUOTA_SCOPE,
        production_quota_status=PRODUCTION_QUOTA_STATUS,
        central_evidence_floor=MappingProxyType(dict(CENTRAL_EVIDENCE_FLOOR)),
        template_policy=TEMPLATE_POLICY,
        entries=tuple(entries),
    )
    return CohortV2PartitionExposureManifest.from_dict(manifest.to_dict())


def audit_cohort_v2_partition_exposure(
    manifest: CohortV2PartitionExposureManifest,
    *,
    declared_provenance_records: Sequence[Mapping[str, Any]],
    observed_artifact_identities: Sequence[str],
) -> dict[str, Any]:
    """Audit lineage inheritance while recording, but not holding out, templates."""
    validated = CohortV2PartitionExposureManifest.from_dict(manifest.to_dict())
    by_lineage = {
        entry.scenario_lineage_identity: entry for entry in validated.entries
    }
    record_fields = {
        "artifact_kind",
        "artifact_identity",
        "source_scenario_lineage_identity",
        "level_instance_identity",
        "scenario_template_identity",
        "dataset_partition",
        "exposure_role",
    }
    artifacts: set[str] = set()
    for record in declared_provenance_records:
        if not isinstance(record, Mapping) or set(record) != record_fields:
            raise ValueError("Cohort-v2 provenance record fields are invalid")
        kind = record["artifact_kind"]
        if kind not in PROVENANCE_ARTIFACT_KINDS:
            raise ValueError("Cohort-v2 provenance artifact kind is unknown")
        artifact = _string(record["artifact_identity"], "artifact_identity")
        if artifact in artifacts:
            raise ValueError("Cohort-v2 provenance artifact identities must be unique")
        artifacts.add(artifact)
        lineage = _string(
            record["source_scenario_lineage_identity"],
            "source_scenario_lineage_identity",
        )
        source = by_lineage.get(lineage)
        if source is None:
            raise ValueError("Cohort-v2 provenance references an undeclared lineage")
        inherited = {
            "level_instance_identity": source.level_instance_identity,
            "scenario_template_identity": source.scenario_template_identity,
            "dataset_partition": source.dataset_partition,
            "exposure_role": source.exposure_role,
        }
        if any(record[field] != expected for field, expected in inherited.items()):
            raise ValueError(
                "Cohort-v2 replay, derivation, and observation variants must inherit "
                "their source lineage role and partition"
            )

    observed = tuple(
        _string(identity, "observed_artifact_identity")
        for identity in observed_artifact_identities
    )
    if len(observed) != len(set(observed)):
        raise ValueError("Observed cohort-v2 artifact identities must be unique")
    if not set(observed) <= artifacts:
        raise ValueError("Observed cohort-v2 artifact has undeclared provenance")

    template_roles: dict[str, set[str]] = {}
    for entry in validated.entries:
        template_roles.setdefault(entry.scenario_template_identity, set()).add(
            entry.exposure_role
        )
    return {
        "schema": "cohort_v2_lineage_template_leakage_audit_v1",
        "partition_identity": validated.identity,
        "scenario_lineage_count": len(validated.entries),
        "level_instance_count": len(validated.entries),
        "scenario_template_count": len(template_roles),
        "non_final_scenario_lineage_count": len([
            entry for entry in validated.entries
            if entry.exposure_role != "final_evaluation"
        ]),
        "non_final_level_instance_count": len({
            entry.level_instance_identity for entry in validated.entries
            if entry.exposure_role != "final_evaluation"
        }),
        "non_final_scenario_template_count": len({
            entry.scenario_template_identity for entry in validated.entries
            if entry.exposure_role != "final_evaluation"
        }),
        "central_evidence_floor_satisfied": True,
        "shared_scenario_template_identities": sorted(
            template
            for template, roles in template_roles.items()
            if len(roles) > 1
        ),
        "declared_provenance_record_count": len(artifacts),
        "observed_artifact_count": len(observed),
        "template_reuse_allowed": True,
        "template_held_out_claim": False,
        "template_held_out_score": False,
        "passed": True,
    }


def audit_cohort_v2_workflow_influence(
    manifest: CohortV2PartitionExposureManifest,
    *,
    workflow_kind: str,
    influence: str,
    declared_provenance_records: Sequence[Mapping[str, Any]],
    observed_scenario_lineage_identities: Sequence[str],
    observed_artifact_identities: Sequence[str],
) -> dict[str, Any]:
    """Require a non-final workflow to use only its role's data and permission."""
    if workflow_kind not in EXPOSURE_ROLES[:-1]:
        raise ValueError("Cohort-v2 ordinary workflow kind is unknown")
    if influence not in ROLE_PERMISSIONS[workflow_kind]:
        raise ValueError("Cohort-v2 workflow influence is not a frozen role permission")
    validated = CohortV2PartitionExposureManifest.from_dict(manifest.to_dict())
    base_audit = audit_cohort_v2_partition_exposure(
        validated,
        declared_provenance_records=declared_provenance_records,
        observed_artifact_identities=observed_artifact_identities,
    )
    by_lineage = {
        entry.scenario_lineage_identity: entry for entry in validated.entries
    }
    observed_lineages = tuple(
        _string(identity, "observed_scenario_lineage_identity")
        for identity in observed_scenario_lineage_identities
    )
    if len(observed_lineages) != len(set(observed_lineages)):
        raise ValueError("Observed cohort-v2 scenario lineages must be unique")
    for lineage in observed_lineages:
        entry = by_lineage.get(lineage)
        if entry is None:
            raise ValueError("Cohort-v2 workflow references an undeclared lineage")
        if entry.exposure_role != workflow_kind:
            raise ValueError("Cohort-v2 workflow crossed its exposure role")
    by_artifact = {
        record["artifact_identity"]: record
        for record in declared_provenance_records
    }
    for artifact_identity in observed_artifact_identities:
        if by_artifact[artifact_identity]["exposure_role"] != workflow_kind:
            raise ValueError("Cohort-v2 workflow artifact crossed its exposure role")
    return {
        "schema": "cohort_v2_workflow_influence_audit_v1",
        "partition_identity": validated.identity,
        "workflow_kind": workflow_kind,
        "influence": influence,
        "observed_lineage_count": len(observed_lineages),
        "observed_artifact_count": base_audit["observed_artifact_count"],
        "passed": True,
    }
