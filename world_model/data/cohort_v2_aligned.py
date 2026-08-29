"""Reader for issue #59's fixed-step-aligned observation recollection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scripts.cohort_v2_macro_semantics import validate_capture_macro_derivation
from scripts.cohort_v2_micro_relations import (
    validate_capture_micro_relation_derivation,
)
from scripts.cohort_v2_physical_violations import (
    validate_capture_physical_violation_derivation,
)
from scripts.observation_trace import (
    validate_observation_exposure_boundaries,
    validate_observation_trace,
)
from scripts.physics_capture_v2 import load_physics_capture_v2
from world_model.data.cohort_v2 import (
    CohortV2IngestionError,
    CohortV2ReleaseReader,
    CohortV2Rollout,
    _freeze,
)
from world_model.data.deployment_temporal import AgentObservation


RELEASE_IDENTITY = "cohort-v2-aligned-observation-release-v1:issue-59"
SCHEMA = "cohort_v2_aligned_observation_release_v1"
PARTITION_SCHEMA = "cohort_v2_aligned_observation_partition_v1"
ROLE_ORDER = ("training", "calibration", "model_selection", "final_evaluation")


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise CohortV2IngestionError(f"Aligned release {label} is malformed") from error
    if not isinstance(value, dict):
        raise CohortV2IngestionError(f"Aligned release {label} is not an object")
    return value


class CohortV2AlignedObservationReader(CohortV2ReleaseReader):
    """Expose one issue-59 role with an agent image for every central frame."""

    def __init__(
        self,
        release_root: Path,
        *,
        source_reader: CohortV2ReleaseReader,
    ) -> None:
        if not isinstance(source_reader, CohortV2ReleaseReader):
            raise CohortV2IngestionError(
                "Aligned observations require a validated source-role reader"
            )
        roles = {item.exposure_role for item in source_reader.rollouts}
        if len(roles) != 1 or next(iter(roles)) not in ROLE_ORDER:
            raise CohortV2IngestionError("Aligned source reader crosses exposure roles")
        role = next(iter(roles))
        root = Path(release_root).resolve()
        release = _load(root / "manifest.json", "manifest")
        if (
            release.get("schema") != SCHEMA
            or release.get("identity") != RELEASE_IDENTITY
            or release.get("passed") is not True
            or release.get("role_counts") != {name: 6 for name in ROLE_ORDER}
        ):
            raise CohortV2IngestionError("Aligned release identity or disposition is stale")
        partition_name = "sealed_final" if role == "final_evaluation" else "public"
        reference = release.get("partitions", {}).get(partition_name)
        if not isinstance(reference, dict):
            raise CohortV2IngestionError("Aligned release partition is missing")
        if role == "final_evaluation" and (
            reference.get("ordinary_workflow_access") is not False
            or not hasattr(source_reader, "access_audit")
        ):
            raise CohortV2IngestionError(
                "Aligned final observations require an authorized final reader"
            )
        partition_root = root / str(reference.get("path"))
        manifest = _load(partition_root / "manifest.json", "partition manifest")
        included_roles = (
            ("final_evaluation",)
            if role == "final_evaluation"
            else ROLE_ORDER[:3]
        )
        if (
            manifest.get("schema") != PARTITION_SCHEMA
            or manifest.get("identity") != reference.get("identity")
            or manifest.get("release_identity") != RELEASE_IDENTITY
            or manifest.get("included_roles") != list(included_roles)
            or manifest.get("role_counts")
            != {name: 6 for name in included_roles}
            or manifest.get("passed") is not True
        ):
            raise CohortV2IngestionError(
                "Aligned release partition identity or disposition is stale"
            )
        records = {
            item["attempt_id"]: item
            for item in manifest.get("records", ())
            if item.get("exposure_role") == role
        }
        source_rollouts = {item.attempt_id: item for item in source_reader.rollouts}
        if set(records) != set(source_rollouts) or len(records) != 6:
            raise CohortV2IngestionError(
                "Aligned release does not preserve the source-role attempt inventory"
            )

        self._root = partition_root
        self._source_reader = source_reader
        self._source_rollouts = source_rollouts
        self._workflow_kind = role
        self._enforce_expected_termination = False
        self._aligned_observation_roots = {}
        self._observation_references = {}
        self._observation_records = {}
        self.release_identity = RELEASE_IDENTITY
        self.capability_declaration_identity = (
            source_reader.capability_declaration_identity
        )
        self.derivation_identity = RELEASE_IDENTITY
        self.partition_identity = source_reader.partition_identity
        if hasattr(source_reader, "access_audit"):
            self.access_audit = source_reader.access_audit
        if hasattr(source_reader, "sealed_bundle_identity"):
            self.sealed_bundle_identity = source_reader.sealed_bundle_identity

        rollouts = []
        manifests = []
        for attempt_id in sorted(records):
            record = records[attempt_id]
            source = source_rollouts[attempt_id]
            rollout_root = partition_root / record["rollout_path"]
            capture = load_physics_capture_v2(
                rollout_root / "physics_capture_v2.json"
            )
            if (
                capture.capture_id != record["capture_id"]
                or capture.source_bindings["rollout_id"] != attempt_id
                or capture.source_bindings["scenario_lineage_id"]
                != source.scenario_lineage_identity
                or record["coverage_stratum"] != source.coverage_stratum
                or record["scenario_lineage_identity"]
                != source.scenario_lineage_identity
            ):
                raise CohortV2IngestionError(
                    "Aligned rollout crossed its frozen source binding"
                )
            references = {item["kind"]: item for item in record["derivations"]}
            if set(references) != {"micro", "macro", "physical-violations"}:
                raise CohortV2IngestionError(
                    "Aligned rollout lacks exact central derivations"
                )
            derivations = {
                kind: _load(partition_root / item["path"], f"{kind} derivation")
                for kind, item in references.items()
            }
            for kind, item in references.items():
                if derivations[kind].get("identity") != item["identity"]:
                    raise CohortV2IngestionError(
                        "Aligned derivation identity differs from its manifest"
                    )
            source_reference = (
                Path(record["rollout_path"]) / "physics_capture_v2.json"
            ).as_posix()
            validate_capture_micro_relation_derivation(
                derivations["micro"],
                capture,
                source_reference=source_reference,
                source_capture_bundle_identity=RELEASE_IDENTITY,
            )
            validate_capture_macro_derivation(
                derivations["macro"],
                capture,
                source_reference=source_reference,
                source_capture_bundle_identity=RELEASE_IDENTITY,
            )
            validate_capture_physical_violation_derivation(
                derivations["physical-violations"],
                capture,
                source_reference=source_reference,
                source_capture_bundle_identity=RELEASE_IDENTITY,
            )
            observation_root = rollout_root / "observation-trace"
            observation = validate_observation_trace(observation_root)
            manifests.append(observation)
            bindings = observation["source_bindings"]
            if (
                observation["identity"] != record["observation_manifest_identity"]
                or observation["exposure_role"] != role
                or bindings["rollout_identity"] != attempt_id
                or bindings["source_scenario_lineage_identity"]
                != source.scenario_lineage_identity
            ):
                raise CohortV2IngestionError(
                    "Aligned observation crossed its frozen source binding"
                )
            frames = self._frame_records(capture, derivations)
            observations = {
                item["fixed_step"]: item for item in observation["frame_records"]
            }
            if (
                tuple(item.fixed_step for item in frames)
                != tuple(observations)
                or any(
                    item["capture_metadata"]["capture_id"] != capture.capture_id
                    for item in observations.values()
                )
            ):
                raise CohortV2IngestionError(
                    "Aligned observations do not exactly bind central frames"
                )
            for fixed_step, item in observations.items():
                self._observation_references[(attempt_id, fixed_step)] = (
                    observation_root,
                    item["identity"],
                )
                self._observation_records[(attempt_id, fixed_step)] = _freeze(item)
            rollouts.append(CohortV2Rollout(
                attempt_id=attempt_id,
                exposure_role=role,
                coverage_stratum=source.coverage_stratum,
                scenario_lineage_identity=source.scenario_lineage_identity,
                intervention=_freeze(source.intervention),
                agent_observation_identity=source.agent_observation_identity,
                agent_observation_fixed_step=source.agent_observation_fixed_step,
                frame_records=frames,
            ))
        validate_observation_exposure_boundaries(manifests)
        self.rollouts = tuple(rollouts)

    def load_observation(
        self, rollout: CohortV2Rollout, *, observation_role: str
    ) -> bytes:
        """Preserve the frozen source-rollout observation for existing consumers."""
        source = self._source_rollouts.get(rollout.attempt_id)
        if source is None:
            raise CohortV2IngestionError(
                "Aligned rollout is absent from its frozen source reader"
            )
        return self._source_reader.load_observation(
            source, observation_role=observation_role
        )

    def frame_observation_metadata(
        self, rollout: CohortV2Rollout, frame_record: Any
    ) -> Mapping[str, Any]:
        """Return synchronized camera metadata for supervision derivation only."""
        item = self._observation_records.get(
            (rollout.attempt_id, frame_record.fixed_step)
        )
        if item is None:
            raise CohortV2IngestionError(
                "Aligned observation metadata is missing for the requested frame record"
            )
        return item["capture_metadata"]

    def agent_observation_identity(
        self, rollout: CohortV2Rollout, frame_record: Any
    ) -> str:
        """Return the deployment observation identity without exposing canonical data."""
        item = self._observation_records.get(
            (rollout.attempt_id, frame_record.fixed_step)
        )
        if item is None:
            raise CohortV2IngestionError(
                "Aligned observation identity is missing for the requested frame record"
            )
        return str(item["agent_observation"]["identity"])

    def load_agent_observation(
        self, rollout: CohortV2Rollout, frame_record: Any
    ) -> AgentObservation:
        """Load one deployment-only observation with its exact simulation identities."""
        metadata = self.frame_observation_metadata(rollout, frame_record)
        return AgentObservation(
            identity=self.agent_observation_identity(rollout, frame_record),
            fixed_step=frame_record.fixed_step,
            fixed_time_seconds=float(metadata["fixed_time_seconds"]),
            png=self.load_frame_observation(
                rollout, frame_record, observation_role="agent"
            ),
            observation_role="agent",
        )


__all__ = ["CohortV2AlignedObservationReader"]
