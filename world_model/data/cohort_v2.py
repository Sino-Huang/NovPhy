"""Fail-closed model-facing reader for the immutable central cohort-v2 release."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Protocol

from scripts.cohort_v2_macro_semantics import (
    DERIVATION_SPEC_IDENTITY as MACRO_SPEC_IDENTITY,
    validate_capture_macro_derivation,
)
from scripts.cohort_v2_micro_relations import (
    DERIVATION_SPEC_IDENTITY as MICRO_SPEC_IDENTITY,
    validate_capture_micro_relation_derivation,
)
from scripts.cohort_v2_partition import (
    CohortV2PartitionExposureManifest,
    EXPOSURE_ROLES,
    ROLE_PERMISSIONS,
)
from scripts.cohort_v2_physical_violations import (
    DERIVATION_SPEC_IDENTITY as VIOLATION_SPEC_IDENTITY,
    validate_capture_physical_violation_derivation,
)
from scripts.cohort_v2_release import CENTRAL_LABELS as RELEASE_CENTRAL_LABELS, V5_CONTRACT
from scripts.cohort_v2_release import validate_published_issue_53_evidence
from scripts.final_evaluation_access import FinalEvaluationWorkflowAccessManifest
from scripts.observation_trace import (
    load_observation_bytes,
    validate_observation_exposure_boundaries,
    validate_observation_trace,
)
from scripts.physics_capture_v2 import load_physics_capture_v2


CENTRAL_LABELS: Final = tuple(RELEASE_CENTRAL_LABELS)
CENTRAL_STRATA: Final = (
    "no-contact/miss",
    "collision",
    "persistent support",
    "support change",
    "destruction",
    "stability transitions",
)
CAPABILITY_DECLARATION_IDENTITY: Final = "cohort-v2-capabilities-v1"
ACCEPTED_LABELS: Final = {
    "contact": MICRO_SPEC_IDENTITY,
    "supports": MICRO_SPEC_IDENTITY,
    "steady-state": MACRO_SPEC_IDENTITY,
    "structure-unstable": MACRO_SPEC_IDENTITY,
    "excess_penetration": VIOLATION_SPEC_IDENTITY,
    "unsupported_stationary_or_floating_body": VIOLATION_SPEC_IDENTITY,
}


class CohortV2IngestionError(ValueError):
    """The requested consumer cannot safely ingest the declared release."""


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CohortV2IngestionError(f"Cannot load {label}: {error}") from error
    if not isinstance(value, dict):
        raise CohortV2IngestionError(f"{label} must be an object")
    return value


def _fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise CohortV2IngestionError(f"{label} envelope is malformed")


def _path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CohortV2IngestionError(f"{label} path is missing")
    path = (root / value).resolve()
    if root != path and root not in path.parents:
        raise CohortV2IngestionError(f"{label} path leaves the release")
    return path


def _inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class CohortV2CentralFrame:
    identity: str
    capture_id: str
    state_id: str
    fixed_step: int
    capture_stride: int
    engine_state: Mapping[str, Any]
    events: tuple[Mapping[str, Any], ...]
    labels: Mapping[str, Any]
    terminal: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class CohortV2Rollout:
    attempt_id: str
    exposure_role: str
    coverage_stratum: str
    scenario_lineage_identity: str
    intervention: Mapping[str, Any]
    agent_observation_identity: str
    agent_observation_fixed_step: int
    frames: tuple[CohortV2CentralFrame, ...]


@dataclass(frozen=True, slots=True)
class CohortV2OracleWindow:
    source_release_identity: str
    exposure_role: str
    attempt_id: str
    scenario_lineage_identity: str
    intervention: Mapping[str, Any]
    context: CohortV2CentralFrame
    target: CohortV2CentralFrame
    agent_observation: bytes


@dataclass(frozen=True, slots=True)
class CohortV2EndpointScore:
    endpoint_count: int
    scored_value_count: int
    correct_value_count: int
    unavailable_value_count: int
    relation_record_count: int


@dataclass(frozen=True, slots=True)
class CohortV2FinalAccessReceipt:
    release_identity: str
    sealed_bundle_identity: str
    workflow_identity: str
    authorization_identity: str
    authorization_state: str
    observed_access_count: int
    passed: bool


class CohortV2EndpointPredictor(Protocol):
    def __call__(self, frame: CohortV2CentralFrame) -> Mapping[str, Any]: ...


class CohortV2ReleaseReader:
    """Validate and expose one permitted non-final role from release v5."""

    def __init__(
        self,
        release_root: Path,
        *,
        capability_declaration_path: Path,
        workflow_kind: str,
        influence: str,
        requested_capabilities: tuple[str, ...] = CENTRAL_LABELS,
    ) -> None:
        if workflow_kind not in EXPOSURE_ROLES[:-1]:
            raise CohortV2IngestionError("Ordinary readers cannot access sealed final artifacts")
        if influence not in ROLE_PERMISSIONS[workflow_kind]:
            raise CohortV2IngestionError("Workflow influence is not permitted for its exposure role")
        if (
            type(requested_capabilities) is not tuple
            or len(requested_capabilities) != len(set(requested_capabilities))
            or set(requested_capabilities) != set(CENTRAL_LABELS)
        ):
            raise CohortV2IngestionError(
                "Requested capabilities must be exactly the accepted central labels"
            )
        self._root = Path(release_root).resolve()
        self._workflow_kind = workflow_kind
        self._observation_references: dict[str, tuple[Path, str]] = {}
        try:
            self._validate_capability_declaration(Path(capability_declaration_path))
            release, derivations, collection, partition, ledger = self._load_envelopes()
            self.release_identity = release["identity"]
            self.derivation_identity = derivations["identity"]
            self.partition_identity = partition.identity
            self.rollouts = self._read_role(
                release, derivations, collection, partition, ledger
            )
        except CohortV2IngestionError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise CohortV2IngestionError(f"Cohort-v2 ingestion rejected: {error}") from error

    @staticmethod
    def _validate_capability_declaration(path: Path) -> None:
        declaration = _load(path, "cohort-v2 capability declaration")
        if (
            declaration.get("schema") != "cohort_v2_capabilities_v1"
            or declaration.get("identity") != CAPABILITY_DECLARATION_IDENTITY
            or declaration.get("scope_name") != "central_v2"
        ):
            raise CohortV2IngestionError("Central capability declaration identity is stale")
        required = declaration.get("capabilities", {}).get("required_central", {})
        declared_labels = {
            *required.get("micro_labels", []),
            *required.get("macro_labels", []),
            *required.get("violation_labels", []),
        }
        if declared_labels != set(CENTRAL_LABELS):
            raise CohortV2IngestionError("Central capability declaration omits or promotes labels")
        if tuple(required.get("coverage_strata", ())) != CENTRAL_STRATA:
            raise CohortV2IngestionError("Central coverage declaration differs")
        if tuple(required.get("exposure_roles", ())) != EXPOSURE_ROLES:
            raise CohortV2IngestionError("Central exposure-role declaration differs")

    def _load_envelopes(self) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        CohortV2PartitionExposureManifest,
        dict[str, Any],
    ]:
        bundle = _load(self._root / "bundle-manifest.json", "release bundle")
        _fields(bundle, {
            "schema", "identity", "publication_identity", "cohort_release_identity",
            "authoritative_derivation_index_identity",
            "sealed_final_evaluation_bundle_identity", "artifacts", "passed",
        }, "release bundle")
        if (
            bundle["schema"] != V5_CONTRACT.schema("issue_53_cohort_v2_release_bundle")
            or bundle["identity"] != V5_CONTRACT.bundle_identity
            or bundle["publication_identity"] != V5_CONTRACT.publication_identity
            or bundle["cohort_release_identity"] != V5_CONTRACT.release_identity
            or bundle["authoritative_derivation_index_identity"]
            != V5_CONTRACT.derivation_index_identity
            or bundle["passed"] is not True
        ):
            raise CohortV2IngestionError("Release bundle identity or disposition is stale")
        members = [
            item["path"]
            for item in _inventory(self._root)
            if item["path"] != "bundle-manifest.json"
        ]
        if bundle["artifacts"] != members:
            raise CohortV2IngestionError("Release bundle membership is stale")

        publication = _load(self._root / "cohort-v2-publication.json", "publication")
        _fields(publication, {
            "schema", "identity", "cohort_release_identity", "cohort_release_path",
            "authoritative_derivation_index_identity",
            "sealed_final_evaluation_bundle_identity", "disposition",
        }, "publication")
        if (
            publication["schema"] != V5_CONTRACT.schema("representative_cohort_v2_publication")
            or publication["identity"] != V5_CONTRACT.publication_identity
            or publication["cohort_release_identity"] != V5_CONTRACT.release_identity
            or publication["cohort_release_path"] != "cohort-v2-release.json"
            or publication["authoritative_derivation_index_identity"]
            != V5_CONTRACT.derivation_index_identity
            or publication["disposition"] != "complete"
        ):
            raise CohortV2IngestionError("Publication envelope is stale")

        release = _load(self._root / publication["cohort_release_path"], "cohort-v2 release")
        _fields(release, {
            "schema", "release_version", "identity", "source_pilot_report_identity",
            "collection_plan", "production_parameter_plan", "partition_manifest",
            "capability_declaration_identity", "scenario_inventory",
            "attempt_accounting_path", "quality_report_path", "player_provenance_path",
            "production_replay", "primary_rollouts", "authoritative_derivation_index",
            "sealed_final_evaluation", "disposition",
        }, "cohort-v2 release")
        if (
            release.get("schema") != V5_CONTRACT.schema("representative_cohort_v2_release")
            or release.get("identity") != V5_CONTRACT.release_identity
            or release.get("release_version") != 5
            or release.get("disposition") != "complete"
            or release.get("capability_declaration_identity")
            != CAPABILITY_DECLARATION_IDENTITY
        ):
            raise CohortV2IngestionError("Cohort-v2 release envelope is stale")
        derivation_reference = release.get("authoritative_derivation_index")
        if not isinstance(derivation_reference, Mapping):
            raise CohortV2IngestionError("Authoritative derivation reference is malformed")
        derivation_path = _path(
            self._root, derivation_reference.get("path"), "authoritative derivation index"
        )
        derivations = _load(derivation_path, "authoritative derivation index")
        _fields(derivations, {
            "schema", "identity", "source_cohort_release_identity",
            "accepted_labels", "artifacts", "sealed_final_evaluation_bundle_identity",
        }, "authoritative derivation index")
        if (
            derivations.get("schema")
            != V5_CONTRACT.schema("cohort_v2_authoritative_derivation_index")
            or derivations.get("identity") != V5_CONTRACT.derivation_index_identity
            or derivation_reference.get("identity") != derivations["identity"]
            or derivations.get("source_cohort_release_identity") != release["identity"]
            or derivations.get("accepted_labels") != ACCEPTED_LABELS
        ):
            raise CohortV2IngestionError("Authoritative derivation envelope is stale")
        primary_rollouts = release.get("primary_rollouts")
        artifacts = derivations.get("artifacts")
        if not isinstance(primary_rollouts, list) or not isinstance(artifacts, list):
            raise CohortV2IngestionError("Release rollout or derivation inventory is malformed")
        role_counts = {
            role: sum(item.get("exposure_role") == role for item in primary_rollouts)
            for role in EXPOSURE_ROLES
        }
        if (
            len(primary_rollouts) != 18
            or role_counts != {
                "training": 6,
                "calibration": 6,
                "model_selection": 6,
                "final_evaluation": 0,
            }
            or len({item.get("attempt_id") for item in primary_rollouts}) != 18
            or len(artifacts) != 54
            or len({
                (item.get("attempt_id"), item.get("kind")) for item in artifacts
            }) != 54
        ):
            raise CohortV2IngestionError("Release rollout or derivation membership is stale")

        collection_reference = release.get("collection_plan")
        partition_reference = release.get("partition_manifest")
        if not isinstance(collection_reference, Mapping) or not isinstance(partition_reference, Mapping):
            raise CohortV2IngestionError("Release plan or partition reference is malformed")
        collection = _load(
            _path(self._root, collection_reference.get("path"), "collection plan"),
            "collection plan",
        )
        if (
            collection_reference.get("identity") != collection.get("identity")
            or collection.get("identity") != V5_CONTRACT.collection_identity
        ):
            raise CohortV2IngestionError("Collection plan binding is stale")
        partition_value = _load(
            _path(self._root, partition_reference.get("path"), "partition manifest"),
            "partition manifest",
        )
        partition = CohortV2PartitionExposureManifest.from_dict(partition_value)
        if partition_reference.get("identity") != partition.identity:
            raise CohortV2IngestionError("Partition binding is stale")
        ledger = _load(
            _path(self._root, release.get("attempt_accounting_path"), "attempt accounting"),
            "attempt accounting",
        )
        return release, derivations, collection, partition, ledger

    def _read_role(
        self,
        release: Mapping[str, Any],
        derivations: Mapping[str, Any],
        collection: Mapping[str, Any],
        partition: CohortV2PartitionExposureManifest,
        ledger: Mapping[str, Any],
    ) -> tuple[CohortV2Rollout, ...]:
        role_entry = next(
            entry for entry in partition.entries if entry.exposure_role == self._workflow_kind
        )
        assignment = next(
            item for item in collection["assignments"]
            if item["exposure_role"] == self._workflow_kind
        )
        if (
            assignment["scenario_lineage_identity"] != role_entry.scenario_lineage_identity
            or tuple(assignment["intervention_ids"]) != tuple(
                item["id"] for item in collection["interventions"]
            )
        ):
            raise CohortV2IngestionError("Collection assignment crossed its partition")
        interventions = {item["id"]: item for item in collection["interventions"]}
        ledger_by_attempt = {
            item["attempt_id"]: item
            for item in ledger["attempt_ledger"]
            if item["exposure_role"] == self._workflow_kind
        }
        rollout_references = [
            item for item in release["primary_rollouts"]
            if item["exposure_role"] == self._workflow_kind
        ]
        if len(rollout_references) != len(CENTRAL_STRATA) or len(ledger_by_attempt) != len(CENTRAL_STRATA):
            raise CohortV2IngestionError("Exposure role lacks its six accepted rollout assignments")

        derivation_by_attempt: dict[str, dict[str, Mapping[str, Any]]] = {}
        for reference in derivations["artifacts"]:
            if reference["exposure_role"] == self._workflow_kind:
                derivation_by_attempt.setdefault(reference["attempt_id"], {})[
                    reference["kind"]
                ] = reference

        manifests = []
        rollouts = []
        for reference in rollout_references:
            attempt_id = reference["attempt_id"]
            ledger_entry = ledger_by_attempt.get(attempt_id)
            if ledger_entry is None or ledger_entry.get("status") != "accepted":
                raise CohortV2IngestionError("Released rollout is not an accepted planned attempt")
            if reference.get("capture_id") is None:
                raise CohortV2IngestionError("Released rollout has no capture identity")
            rollout_root = _path(self._root, reference.get("path"), f"rollout {attempt_id}")
            if not rollout_root.is_dir() or reference.get("files") != _inventory(rollout_root):
                raise CohortV2IngestionError("Primary rollout inventory is stale")
            capture_path = rollout_root / "physics_capture_v2.json"
            capture = load_physics_capture_v2(capture_path)
            if (
                capture.capture_id != reference["capture_id"]
                or capture.source_bindings["rollout_id"] != attempt_id
                or capture.source_bindings["scenario_lineage_id"]
                != role_entry.scenario_lineage_identity
                or capture.source_bindings["level_instance_id"]
                != role_entry.level_instance_identity
                or capture.source_bindings["scenario_template_id"]
                != role_entry.scenario_template_identity
                or capture.source_bindings["intervention_id"]
                != ledger_entry["intervention_identity"]
                or capture.record["terminal_evidence"]["reason"]
                != ledger_entry["terminal_reason"]
                or ledger_entry["terminal_reason"] != ledger_entry["expected_termination"]
            ):
                raise CohortV2IngestionError("Primary rollout source binding is stale")
            intervention = interventions[ledger_entry["intervention_id"]]
            if ledger_entry["intended_coverage_stratum"] != intervention["intended_coverage_stratum"]:
                raise CohortV2IngestionError("Intervention coverage binding is stale")

            references = derivation_by_attempt.get(attempt_id, {})
            if set(references) != {"micro", "macro", "physical-violations"}:
                raise CohortV2IngestionError("Primary rollout lacks exact central derivations")
            source_reference = f"{reference['path']}/physics_capture_v2.json"
            loaded_derivations = {
                kind: _load(
                    _path(self._root, item["path"], f"{kind} derivation"),
                    f"{kind} derivation",
                )
                for kind, item in references.items()
            }
            for kind, item in references.items():
                if loaded_derivations[kind].get("identity") != item.get("identity"):
                    raise CohortV2IngestionError("Derivation identity differs from its index")
            validate_capture_micro_relation_derivation(
                loaded_derivations["micro"], capture,
                source_reference=source_reference,
                source_capture_bundle_identity=release["identity"],
            )
            validate_capture_macro_derivation(
                loaded_derivations["macro"], capture,
                source_reference=source_reference,
                source_capture_bundle_identity=release["identity"],
            )
            validate_capture_physical_violation_derivation(
                loaded_derivations["physical-violations"], capture,
                source_reference=source_reference,
                source_capture_bundle_identity=release["identity"],
            )

            observation_root = rollout_root / "observation-trace"
            manifest = validate_observation_trace(observation_root)
            manifests.append(manifest)
            bindings = manifest["source_bindings"]
            if (
                manifest["exposure_role"] != self._workflow_kind
                or bindings["rollout_identity"] != attempt_id
                or bindings["source_scenario_lineage_identity"]
                != role_entry.scenario_lineage_identity
                or bindings["level_instance_identity"] != role_entry.level_instance_identity
                or bindings["scenario_template_identity"] != role_entry.scenario_template_identity
            ):
                raise CohortV2IngestionError("Observation trace crossed its release binding")
            observations = {
                item["fixed_step"]: item for item in manifest["frame_records"]
            }
            if len(observations) != 1:
                raise CohortV2IngestionError(
                    "Each released rollout must expose its declared agent observation"
                )
            observation = next(iter(observations.values()))
            first_capture_step = capture.record["fixed_step_samples"][0]["fixed_step"]
            if observation["fixed_step"] > first_capture_step:
                raise CohortV2IngestionError(
                    "Agent observation occurs after the intervention trace begins"
                )
            frames = self._frames(capture, loaded_derivations)
            self._observation_references[attempt_id] = (
                observation_root,
                observation["identity"],
            )
            rollouts.append(CohortV2Rollout(
                attempt_id=attempt_id,
                exposure_role=self._workflow_kind,
                coverage_stratum=ledger_entry["intended_coverage_stratum"],
                scenario_lineage_identity=role_entry.scenario_lineage_identity,
                intervention=_freeze({
                    "id": intervention["id"],
                    "identity": ledger_entry["intervention_identity"],
                    "interface_action": intervention["interface_action"],
                    "engine_relative_action": intervention["engine_relative_action"],
                }),
                agent_observation_identity=observation["agent_observation"]["identity"],
                agent_observation_fixed_step=observation["fixed_step"],
                frames=frames,
            ))
        validate_observation_exposure_boundaries(manifests)
        if {item.coverage_stratum for item in rollouts} != set(CENTRAL_STRATA):
            raise CohortV2IngestionError("Exposure role does not cover every central stratum")
        return tuple(rollouts)

    def _frames(
        self,
        capture: Any,
        derivations: Mapping[str, Mapping[str, Any]],
    ) -> tuple[CohortV2CentralFrame, ...]:
        samples = capture.record["fixed_step_samples"]
        capture_frames = capture.record["frame_records"]
        micro = derivations["micro"]["labels"]
        macro = derivations["macro"]["labels"]
        violations = derivations["physical-violations"]["labels"]
        step_sequences = [
            tuple(item["fixed_step"] for item in values)
            for values in (samples, capture_frames, micro, macro, violations)
        ]
        if len(set(step_sequences)) != 1:
            raise CohortV2IngestionError("Primary and derived fixed-step sequences are misaligned")
        events_by_step: dict[int, list[Mapping[str, Any]]] = {}
        for event in capture.record["events"]:
            events_by_step.setdefault(event["fixed_step"], []).append(event)
        terminal = capture.record["terminal_evidence"]
        if terminal["fixed_step"] != step_sequences[0][-1]:
            raise CohortV2IngestionError("Terminal record is not aligned to the last fixed step")
        frames = []
        for sample, frame, micro_label, macro_label, violation_label in zip(
            samples, capture_frames, micro, macro, violations, strict=True
        ):
            step = sample["fixed_step"]
            identity = f"cohort-v2-central-frame-v1:{capture.capture_id}:{frame['state_id']}"
            labels = {
                "contact": micro_label["predicates"]["contact"],
                "supports": micro_label["predicates"]["supports"],
                "steady-state": macro_label["predicates"]["steady-state"],
                "structure-unstable": macro_label["predicates"]["structure-unstable"],
                "excess_penetration": violation_label["predicates"]["excess_penetration"],
                "unsupported_stationary_or_floating_body": violation_label["predicates"][
                    "unsupported_stationary_or_floating_body"
                ],
            }
            frames.append(CohortV2CentralFrame(
                identity=identity,
                capture_id=capture.capture_id,
                state_id=frame["state_id"],
                fixed_step=step,
                capture_stride=capture.configured_fixed_step_capture_stride,
                engine_state=_freeze(sample),
                events=tuple(_freeze(item) for item in events_by_step.get(step, ())),
                labels=_freeze(labels),
                terminal=_freeze(terminal) if step == terminal["fixed_step"] else None,
            ))
        return tuple(frames)

    def load_observation(
        self, rollout: CohortV2Rollout, *, observation_role: str
    ) -> bytes:
        reference = self._observation_references.get(rollout.attempt_id)
        if reference is None:
            raise CohortV2IngestionError("Rollout has no synchronized observation")
        root, frame_record_identity = reference
        try:
            return load_observation_bytes(
                root,
                frame_record_identity=frame_record_identity,
                observation_role=observation_role,
                workflow_kind=self._workflow_kind,
                purpose="model_input",
            )
        except ValueError as error:
            raise CohortV2IngestionError(str(error)) from error


class CohortV2OracleWindowDataset(Sequence[CohortV2OracleWindow]):
    """Observation-backed one-step oracle-symbol windows derived from a role reader."""

    def __init__(self, reader: CohortV2ReleaseReader) -> None:
        self._examples = tuple(
            CohortV2OracleWindow(
                source_release_identity=reader.release_identity,
                exposure_role=rollout.exposure_role,
                attempt_id=rollout.attempt_id,
                scenario_lineage_identity=rollout.scenario_lineage_identity,
                intervention=rollout.intervention,
                context=rollout.frames[0],
                target=rollout.frames[1],
                agent_observation=reader.load_observation(
                    rollout, observation_role="agent"
                ),
            )
            for rollout in reader.rollouts
        )
        if len(self._examples) != len(reader.rollouts):
            raise CohortV2IngestionError(
                "Every admitted rollout must yield one observation-backed training window"
            )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> CohortV2OracleWindow:
        return self._examples[index]


def _score_boolean_label(
    label: Mapping[str, Any], prediction: Any
) -> tuple[int, int, int]:
    availability = label.get("availability")
    value = label.get("value")
    if availability == "available":
        if type(value) is not bool or type(prediction) is not bool:
            raise CohortV2IngestionError("Available endpoint labels require boolean predictions")
        return 1, int(value == prediction), 0
    if not isinstance(availability, str) or not availability.startswith("unavailable_"):
        raise CohortV2IngestionError("Endpoint label availability is malformed")
    if value is not None:
        raise CohortV2IngestionError("Unavailable endpoint label was converted to a value")
    return 0, 0, 1


def score_cohort_v2_endpoints(
    reader: CohortV2ReleaseReader,
    predictor: CohortV2EndpointPredictor,
) -> CohortV2EndpointScore:
    """Score role endpoints without treating unavailable oracle values as negatives."""
    scored = correct = unavailable = relations = 0
    prediction_fields = {
        "steady-state",
        "structure-unstable",
        "excess_penetration",
        "unsupported_stationary_or_floating_body",
    }
    for rollout in reader.rollouts:
        endpoint = rollout.frames[-1]
        if endpoint.terminal is None or set(endpoint.labels) != set(CENTRAL_LABELS):
            raise CohortV2IngestionError("Endpoint central tuple is incomplete")
        for predicate in ("contact", "supports"):
            relation = endpoint.labels[predicate]
            if relation.get("availability") != "available" or not isinstance(
                relation.get("relations"), tuple
            ):
                raise CohortV2IngestionError("Endpoint relation label is malformed")
            relations += 1
        prediction = predictor(endpoint)
        if not isinstance(prediction, Mapping) or set(prediction) != prediction_fields:
            raise CohortV2IngestionError("Endpoint predictor returned a partial central prediction")
        for predicate in (
            "steady-state", "structure-unstable", "excess_penetration"
        ):
            counts = _score_boolean_label(
                endpoint.labels[predicate], prediction[predicate]
            )
            scored += counts[0]
            correct += counts[1]
            unavailable += counts[2]
        unsupported = endpoint.labels["unsupported_stationary_or_floating_body"]
        unsupported_prediction = prediction[
            "unsupported_stationary_or_floating_body"
        ]
        if not isinstance(unsupported, tuple) or not isinstance(
            unsupported_prediction, Mapping
        ):
            raise CohortV2IngestionError("Unsupported-body endpoint prediction is malformed")
        entity_ids = tuple(item.get("entity_id") for item in unsupported)
        if (
            not all(isinstance(item, str) and item for item in entity_ids)
            or len(set(entity_ids)) != len(entity_ids)
            or set(unsupported_prediction) != set(entity_ids)
        ):
            raise CohortV2IngestionError("Unsupported-body endpoint prediction is partial")
        for item in unsupported:
            counts = _score_boolean_label(
                item, unsupported_prediction[item["entity_id"]]
            )
            scored += counts[0]
            correct += counts[1]
            unavailable += counts[2]
    return CohortV2EndpointScore(
        endpoint_count=len(reader.rollouts),
        scored_value_count=scored,
        correct_value_count=correct,
        unavailable_value_count=unavailable,
        relation_record_count=relations,
    )


def probe_cohort_v2_final_access(
    public_release_root: Path,
    sealed_release_root: Path,
) -> CohortV2FinalAccessReceipt:
    """Validate the authorized workflow audit without exposing sealed examples."""
    public = Path(public_release_root).resolve()
    sealed = Path(sealed_release_root).resolve()
    partition = CohortV2PartitionExposureManifest.from_dict(
        _load(public / "partition-exposure-manifest.json", "partition manifest")
    )
    workflow = FinalEvaluationWorkflowAccessManifest.from_dict(
        _load(sealed / "authorized-final-access-manifest.json", "authorized final access")
    )
    if workflow.partition_identity != partition.identity:
        raise CohortV2IngestionError("Final access workflow targets another partition")
    if workflow.authorization_state != "authorized":
        raise CohortV2IngestionError("Final access workflow is not authorized")
    audit = _load(sealed / "final-access-audit.json", "final access audit")
    _fields(audit, {
        "schema", "workflow_manifest_identity", "workflow_identity",
        "operator_identity", "partition_identity", "authorization_state",
        "authorization_identity", "observed_access_count", "passed",
    }, "final access audit")
    if (
        audit["schema"] != "final_evaluation_workflow_access_audit_v1"
        or audit["workflow_manifest_identity"] != workflow.identity
        or audit["workflow_identity"] != workflow.workflow_identity
        or audit["operator_identity"] != workflow.operator_identity
        or audit["partition_identity"] != partition.identity
        or audit["authorization_state"] != "authorized"
        or audit["authorization_identity"] != workflow.authorization_identity
        or audit["passed"] is not True
        or not isinstance(audit["observed_access_count"], int)
        or audit["observed_access_count"] <= 0
    ):
        raise CohortV2IngestionError("Final access audit is stale or incomplete")
    sealed_manifest = _load(sealed / "sealed-bundle-manifest.json", "sealed bundle")
    if (
        sealed_manifest.get("identity") != V5_CONTRACT.sealed_bundle_identity
        or sealed_manifest.get("ordinary_workflow_access") is not False
        or sealed_manifest.get("passed") is not True
        or sealed_manifest.get("authorized_workflow_identity")
        != workflow.identity
    ):
        raise CohortV2IngestionError("Sealed final boundary is stale")
    validation = validate_published_issue_53_evidence(public, sealed)
    if validation.get("passed") is not True:
        raise CohortV2IngestionError("Authorized final release validation failed")
    assert workflow.authorization_identity is not None
    return CohortV2FinalAccessReceipt(
        release_identity=V5_CONTRACT.release_identity,
        sealed_bundle_identity=V5_CONTRACT.sealed_bundle_identity,
        workflow_identity=workflow.workflow_identity,
        authorization_identity=workflow.authorization_identity,
        authorization_state=workflow.authorization_state,
        observed_access_count=audit["observed_access_count"],
        passed=True,
    )


__all__ = [
    "CENTRAL_LABELS",
    "CohortV2CentralFrame",
    "CohortV2EndpointScore",
    "CohortV2FinalAccessReceipt",
    "CohortV2IngestionError",
    "CohortV2OracleWindow",
    "CohortV2OracleWindowDataset",
    "CohortV2ReleaseReader",
    "CohortV2Rollout",
    "probe_cohort_v2_final_access",
    "score_cohort_v2_endpoints",
]
