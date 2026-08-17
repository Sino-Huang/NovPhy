"""Frozen, deterministic collection-plan artifacts and their runtime execution seam.

``runtime`` may be a callable accepting :class:`RuntimeInput` or an object with
an ``execute(RuntimeInput)`` method. It returns a :class:`RuntimeResult` or an
equivalent mapping. Accepted results may declare zero or more realized coverage
strata. Rejected results require a reason; failed results require a reason
and failure code. The executor never changes a frozen action or adds a slot to
fill a realized-coverage outcome.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Literal, Protocol

from scripts.scenario_manifest import (
    SCENARIO_MANIFEST_PROJECTION_FIELDS,
    load_scenario_manifest_projection,
)
from scripts.collect_rollouts import normalize_action_to_game


SCHEMA = "collection_plan_v1"
PLAN_IDENTITY_NAMESPACE = "collection-plan-v1"
INTERVENTION_IDENTITY_NAMESPACE = "collection-plan-intervention-v1"
SCENARIO_IDENTITY_NAMESPACE = "collection-plan-scenario-v1"
ATTEMPT_IDENTITY_NAMESPACE = "collection-plan-attempt-v1"
REPORT_FILENAME = "collection_plan_report.json"
PLAN_COPY_FILENAME = "collection_plan.json"

REQUIRED_COVERAGE_STRATA = (
    "no-contact/miss",
    "collision",
    "persistent support",
    "support change",
    "destruction",
    "pig removal",
    "explosion",
    "stability transitions",
    "level clear",
    "level fail",
)
INTERVENTION_SOURCES = (
    "geometry_stratified",
    "targeted_rare",
    "benchmark_agent_replay",
)
ACTION_MAPPING_VERSION = "science-birds-slingshot-relative-v1"
TRANSIENT_FAILURE_CODES = (
    "engine_start_timeout",
    "transport_unavailable",
    "capture_temporarily_unavailable",
)
EXPOSURE_ROLES = (
    "training",
    "calibration",
    "model_selection",
    "final_evaluation",
)
SCENARIO_PROJECTION_FIELDS = ("scenario_manifest", *SCENARIO_MANIFEST_PROJECTION_FIELDS)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _identity(namespace: str, value: Any) -> str:
    return f"{namespace}:sha256:{sha256(_canonical_json(value)).hexdigest()}"


def _freeze(value: Any) -> Any:
    """Copy JSON-shaped input into recursively immutable containers."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _json_value(value: Any, field: str) -> Any:
    try:
        _canonical_json(_thaw(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain JSON-compatible values") from exc
    return _freeze(value)


def _require_object(value: Any, field: str, *, nonempty: bool = False) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or (nonempty and not value):
        qualifier = " a nonempty object" if nonempty else " an object"
        raise ValueError(f"{field} must be{qualifier}")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must use string keys")
    return value


def _require_exact_keys(data: Mapping[str, Any], fields: set[str], name: str) -> None:
    if set(data) != fields:
        raise ValueError(f"{name} is incomplete or contains unknown fields")


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int
    transient_failure_codes: tuple[str, ...]
    stopping_rule: Literal["execute_all_interventions"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "transient_failure_codes": list(self.transient_failure_codes),
            "stopping_rule": self.stopping_rule,
        }


@dataclass(frozen=True, slots=True)
class NegativeSpecification:
    cap: int
    intervention_ids: tuple[str, ...]
    semantic_justification: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cap": self.cap,
            "intervention_ids": list(self.intervention_ids),
            "semantic_justification": self.semantic_justification,
        }


@dataclass(frozen=True, slots=True)
class CollectionIntervention:
    id: str
    ordinal: int
    intended_coverage_stratum: str
    source: str
    interface_action: Mapping[str, Any]
    engine_relative_action: Mapping[str, Any]
    mapping_version: str
    slingshot_reference: Mapping[str, Any]
    source_provenance: Mapping[str, Any]

    @property
    def identity(self) -> str:
        return _identity(INTERVENTION_IDENTITY_NAMESPACE, self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ordinal": self.ordinal,
            "intended_coverage_stratum": self.intended_coverage_stratum,
            "source": self.source,
            "interface_action": _thaw(self.interface_action),
            "engine_relative_action": _thaw(self.engine_relative_action),
            "mapping_version": self.mapping_version,
            "slingshot_reference": _thaw(self.slingshot_reference),
            "source_provenance": _thaw(self.source_provenance),
        }


@dataclass(frozen=True, slots=True)
class CollectionScenario:
    scenario_id: str
    exposure_role: str
    scenario_manifest_projection: Mapping[str, Any]
    expected_initial_engine_state_identity: str
    retry_policy: RetryPolicy
    negative_specification: NegativeSpecification
    interventions: tuple[CollectionIntervention, ...]
    source_dispositions: Mapping[str, Mapping[str, Any]]
    coverage_strata: Mapping[str, Mapping[str, Any]]

    @property
    def identity(self) -> str:
        return _identity(SCENARIO_IDENTITY_NAMESPACE, self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "exposure_role": self.exposure_role,
            **_thaw(self.scenario_manifest_projection),
            "expected_initial_engine_state_identity": self.expected_initial_engine_state_identity,
            "retry_policy": self.retry_policy.to_dict(),
            "negative_specification": self.negative_specification.to_dict(),
            "interventions": [intervention.to_dict() for intervention in self.interventions],
            "source_dispositions": _thaw(self.source_dispositions),
            "coverage_strata": _thaw(self.coverage_strata),
        }


@dataclass(frozen=True, slots=True)
class CollectionPlan:
    schema: Literal["collection_plan_v1"]
    plan_version: int
    identity: str
    scenarios: tuple[CollectionScenario, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_version": self.plan_version,
            "identity": self.identity,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CollectionPlan":
        _require_exact_keys(data, {"schema", "plan_version", "identity", "scenarios"}, "Collection plan")
        if data["schema"] != SCHEMA:
            raise ValueError(f"Unsupported collection plan schema: {data['schema']}")
        plan_version = data["plan_version"]
        if isinstance(plan_version, bool) or not isinstance(plan_version, int) or plan_version <= 0:
            raise ValueError("Collection plan version must be a positive integer")
        scenarios_data = _require_list(data["scenarios"], "Collection plan scenarios")
        if not scenarios_data:
            raise ValueError("Collection plan requires at least one scenario")
        scenarios = tuple(_scenario_from_dict(item) for item in scenarios_data)
        scenario_ids = [scenario.scenario_id for scenario in scenarios]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("Collection plan scenario IDs must be unique")
        identity = _require_nonempty_string(data["identity"], "Collection plan identity")
        plan = cls(SCHEMA, plan_version, identity, scenarios)
        expected_identity = _plan_identity(plan_version, scenarios)
        if identity != expected_identity:
            raise ValueError("Collection plan identity is stale")
        return plan


@dataclass(frozen=True, slots=True)
class LoadedCollectionPlan:
    plan: CollectionPlan
    path: Path
    original_bytes: bytes


@dataclass(frozen=True, slots=True)
class RuntimeInput:
    """Immutable input passed to one runtime attempt."""

    plan_identity: str
    plan_version: int
    scenario_id: str
    scenario_identity: str
    intervention_id: str
    intervention_identity: str
    attempt_id: str
    attempt_number: int
    expected_initial_engine_state_identity: str
    interface_action: Mapping[str, Any]
    engine_relative_action: Mapping[str, Any]
    mapping_version: str
    slingshot_reference: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    status: Literal["accepted", "rejected", "failed"]
    reason: str | None = None
    failure_code: str | None = None
    realized_coverage_strata: tuple[str, ...] = ()
    eligible: bool = True
    artifact_path: str | None = None
    quarantine_path: str | None = None
    failure_manifest_path: str | None = None


@dataclass(frozen=True, slots=True)
class _AttemptDisposition:
    artifact_disposition: Literal["accepted", "quarantined"]
    retry_decision: Literal["none", "retry", "stop"]
    failure_class: Literal["none", "transient", "permanent"]
    disposition: Literal["accept", "retry", "quarantine"]
    disposition_reason: str


class _RuntimeCallbackError(Exception):
    pass


class CollectionPlanRuntime(Protocol):
    def __call__(self, request: RuntimeInput) -> RuntimeResult | Mapping[str, Any]: ...


def _retry_policy_from_dict(data: Any) -> RetryPolicy:
    policy = _require_object(data, "Retry policy")
    _require_exact_keys(policy, {"max_attempts", "transient_failure_codes", "stopping_rule"}, "Retry policy")
    max_attempts = policy["max_attempts"]
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts <= 0:
        raise ValueError("Retry policy max_attempts must be a positive integer")
    codes = _require_list(policy["transient_failure_codes"], "Retry policy transient_failure_codes")
    if any(not isinstance(code, str) or not code for code in codes) or len(codes) != len(set(codes)):
        raise ValueError("Retry policy transient_failure_codes must be unique nonempty strings")
    if any(code not in TRANSIENT_FAILURE_CODES for code in codes):
        raise ValueError("Retry policy contains a non-transient failure code")
    if policy["stopping_rule"] != "execute_all_interventions":
        raise ValueError("Retry policy stopping_rule must be execute_all_interventions")
    return RetryPolicy(max_attempts, tuple(codes), "execute_all_interventions")


def _negative_specification_from_dict(
    data: Any,
    interventions: tuple[CollectionIntervention, ...],
) -> NegativeSpecification:
    specification = _require_object(data, "Negative specification")
    _require_exact_keys(
        specification,
        {"cap", "intervention_ids", "semantic_justification"},
        "Negative specification",
    )
    cap = specification["cap"]
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
        raise ValueError("Negative specification cap must be a nonnegative integer")
    ids = _require_list(specification["intervention_ids"], "Negative specification intervention_ids")
    if any(not isinstance(identifier, str) or not identifier for identifier in ids) or len(ids) != len(set(ids)):
        raise ValueError("Negative specification intervention_ids must be unique nonempty strings")
    negative_ids = {
        intervention.id
        for intervention in interventions
        if intervention.intended_coverage_stratum == "no-contact/miss"
    }
    if set(ids) != negative_ids:
        raise ValueError("Negative specification must list exactly the no-contact/miss interventions")
    if len(ids) > cap:
        raise ValueError("Negative specification interventions exceed cap")
    semantic_justification = _require_nonempty_string(
        specification["semantic_justification"],
        "Negative specification semantic_justification",
    )
    return NegativeSpecification(cap, tuple(ids), semantic_justification)


def _point(value: Any, field: str) -> list[int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{field} must contain exactly two integers")
    return list(value)


def _validate_action_mapping(
    interface_action: Mapping[str, Any],
    engine_relative_action: Mapping[str, Any],
    mapping_version: str,
    slingshot_reference: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        interface_action,
        {
            "action_type",
            "coordinate_frame",
            "drag_start",
            "drag_release",
            "tapTime",
            "releaseTime",
            "frame_height",
            "socket_command",
        },
        "Collection intervention interface_action",
    )
    _require_exact_keys(
        slingshot_reference,
        {"gameX", "gameY"},
        "Collection intervention slingshot_reference",
    )
    _require_exact_keys(
        engine_relative_action,
        {"coordinate_frame", "release_offset", "release_point", "tap_time_ms", "release_time_ms"},
        "Collection intervention engine_relative_action",
    )
    if mapping_version != ACTION_MAPPING_VERSION:
        raise ValueError(f"Collection intervention mapping_version must be {ACTION_MAPPING_VERSION}")
    if interface_action["action_type"] != "drag_hold_release":
        raise ValueError("Collection intervention action_type must be drag_hold_release")
    if interface_action["coordinate_frame"] != "slingshot_relative":
        raise ValueError("Collection intervention interface_action must be slingshot_relative")
    drag_start = _point(interface_action["drag_start"], "Collection intervention drag_start")
    drag_release = _point(interface_action["drag_release"], "Collection intervention drag_release")
    game_x = slingshot_reference["gameX"]
    game_y = slingshot_reference["gameY"]
    if any(isinstance(item, bool) or not isinstance(item, int) for item in (game_x, game_y)):
        raise ValueError("Collection intervention slingshot_reference values must be integers")
    if drag_start != [game_x, game_y]:
        raise ValueError("Collection intervention drag_start must equal the frozen slingshot reference")
    tap_time = interface_action["tapTime"]
    release_time = interface_action["releaseTime"]
    if isinstance(tap_time, bool) or not isinstance(tap_time, int) or tap_time < 0:
        raise ValueError("Collection intervention tapTime must be a nonnegative integer")
    if isinstance(release_time, bool) or not isinstance(release_time, int) or release_time <= 0:
        raise ValueError("Collection intervention releaseTime must be a positive integer")
    frame_height = interface_action["frame_height"]
    if isinstance(frame_height, bool) or not isinstance(frame_height, int) or frame_height <= 0:
        raise ValueError("Collection intervention frame_height must be a positive integer")
    normalized = normalize_action_to_game(dict(interface_action))
    socket_command = _require_object(
        interface_action["socket_command"],
        "Collection intervention socket_command",
    )
    _require_exact_keys(
        socket_command,
        {"x", "y", "tapTime", "releaseTime"},
        "Collection intervention socket_command",
    )
    expected_socket_command = {
        "x": normalized["gameX"],
        "y": max(0, frame_height - 1 - normalized["gameY"]),
        "tapTime": tap_time,
        "releaseTime": release_time,
    }
    if _thaw(socket_command) != expected_socket_command:
        raise ValueError("Collection intervention socket_command does not match interface_action")
    expected_engine_action = {
        "coordinate_frame": "slingshot_relative",
        "release_offset": drag_release,
        "release_point": [normalized["gameX"], normalized["gameY"]],
        "tap_time_ms": tap_time,
        "release_time_ms": release_time,
    }
    if _thaw(engine_relative_action) != expected_engine_action:
        raise ValueError("Collection intervention engine_relative_action does not match interface_action")


def _validate_source_provenance(
    source: str,
    value: Mapping[str, Any],
    intended_coverage_stratum: str,
) -> None:
    fields = {
        "geometry_stratified": {"scenario_geometry_identity", "stratum", "feasibility_rule"},
        "targeted_rare": {"target_stratum", "selection_rule"},
        "benchmark_agent_replay": {"agent_identity", "trace_identity", "action_index"},
    }[source]
    _require_exact_keys(value, fields, f"Collection intervention {source} provenance")
    for field in fields - {"action_index"}:
        _require_nonempty_string(value[field], f"Collection intervention provenance {field}")
    if source == "benchmark_agent_replay":
        action_index = value["action_index"]
        if isinstance(action_index, bool) or not isinstance(action_index, int) or action_index < 0:
            raise ValueError("Collection intervention replay action_index must be a nonnegative integer")
    if source == "targeted_rare" and value["target_stratum"] != intended_coverage_stratum:
        raise ValueError("Collection intervention rare target_stratum must match intended_coverage_stratum")


def _intervention_from_dict(data: Any) -> CollectionIntervention:
    intervention = _require_object(data, "Collection intervention")
    fields = {
        "id",
        "ordinal",
        "intended_coverage_stratum",
        "source",
        "interface_action",
        "engine_relative_action",
        "mapping_version",
        "slingshot_reference",
        "source_provenance",
    }
    _require_exact_keys(intervention, fields, "Collection intervention")
    ordinal = intervention["ordinal"]
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal <= 0:
        raise ValueError("Collection intervention ordinal must be a positive integer")
    stratum = _require_nonempty_string(intervention["intended_coverage_stratum"], "Collection intervention intended_coverage_stratum")
    if stratum not in REQUIRED_COVERAGE_STRATA:
        raise ValueError("Collection intervention has unknown intended coverage stratum")
    source = _require_nonempty_string(intervention["source"], "Collection intervention source")
    if source not in INTERVENTION_SOURCES:
        raise ValueError("Collection intervention has unknown source")
    interface_action = _json_value(_require_object(intervention["interface_action"], "Collection intervention interface_action", nonempty=True), "Collection intervention interface_action")
    engine_relative_action = _json_value(_require_object(intervention["engine_relative_action"], "Collection intervention engine_relative_action", nonempty=True), "Collection intervention engine_relative_action")
    mapping_version = _require_nonempty_string(intervention["mapping_version"], "Collection intervention mapping_version")
    slingshot_reference = _json_value(_require_object(intervention["slingshot_reference"], "Collection intervention slingshot_reference", nonempty=True), "Collection intervention slingshot_reference")
    source_provenance = _json_value(_require_object(intervention["source_provenance"], "Collection intervention source_provenance", nonempty=True), "Collection intervention source_provenance")
    _validate_action_mapping(interface_action, engine_relative_action, mapping_version, slingshot_reference)
    _validate_source_provenance(source, source_provenance, stratum)
    return CollectionIntervention(
        id=_require_nonempty_string(intervention["id"], "Collection intervention id"),
        ordinal=ordinal,
        intended_coverage_stratum=stratum,
        source=source,
        interface_action=interface_action,
        engine_relative_action=engine_relative_action,
        mapping_version=mapping_version,
        slingshot_reference=slingshot_reference,
        source_provenance=source_provenance,
    )


def _validate_source_dispositions(data: Any, interventions: tuple[CollectionIntervention, ...]) -> Mapping[str, Mapping[str, Any]]:
    dispositions = _require_object(data, "Source dispositions")
    _require_exact_keys(dispositions, set(INTERVENTION_SOURCES), "Source dispositions")
    by_source = {source: [item for item in interventions if item.source == source] for source in INTERVENTION_SOURCES}
    normalized: dict[str, Mapping[str, Any]] = {}
    for source in INTERVENTION_SOURCES:
        disposition = _require_object(dispositions[source], f"Source disposition {source}")
        status = disposition.get("status")
        if source in {"geometry_stratified", "targeted_rare"}:
            _require_exact_keys(disposition, {"status"}, f"Source disposition {source}")
            if status != "included" or not by_source[source]:
                raise ValueError(f"Source {source} must be included with an intervention")
        elif status == "included":
            _require_exact_keys(disposition, {"status"}, "Source disposition benchmark_agent_replay")
            if not by_source[source]:
                raise ValueError("Included benchmark_agent_replay requires an intervention")
        elif status == "unavailable":
            _require_exact_keys(disposition, {"status", "rationale"}, "Source disposition benchmark_agent_replay")
            _require_nonempty_string(disposition["rationale"], "Benchmark replay unavailable rationale")
            if by_source[source]:
                raise ValueError("Unavailable benchmark_agent_replay cannot have interventions")
        else:
            raise ValueError("benchmark_agent_replay must be included or unavailable")
        normalized[source] = _freeze(dict(disposition))
    return MappingProxyType(normalized)


def _validate_coverage_strata(data: Any, interventions: tuple[CollectionIntervention, ...]) -> Mapping[str, Mapping[str, Any]]:
    dispositions = _require_object(data, "Coverage strata")
    _require_exact_keys(dispositions, set(REQUIRED_COVERAGE_STRATA), "Coverage strata")
    by_id = {intervention.id: intervention for intervention in interventions}
    normalized: dict[str, Mapping[str, Any]] = {}
    for stratum in REQUIRED_COVERAGE_STRATA:
        disposition = _require_object(dispositions[stratum], f"Coverage stratum {stratum}")
        status = disposition.get("status")
        matching_ids = {
            intervention.id
            for intervention in interventions
            if intervention.intended_coverage_stratum == stratum
        }
        if status == "targeted":
            _require_exact_keys(disposition, {"status", "intervention_ids"}, f"Coverage stratum {stratum}")
            ids = _require_list(disposition["intervention_ids"], f"Coverage stratum {stratum} intervention_ids")
            if not ids or any(not isinstance(identifier, str) or not identifier for identifier in ids) or len(ids) != len(set(ids)):
                raise ValueError(f"Coverage stratum {stratum} must reference unique interventions")
            if set(ids) != matching_ids or any(identifier not in by_id for identifier in ids):
                raise ValueError(f"Coverage stratum {stratum} must reference exactly its planned interventions")
        elif status == "inapplicable":
            _require_exact_keys(disposition, {"status", "rationale"}, f"Coverage stratum {stratum}")
            _require_nonempty_string(disposition["rationale"], f"Coverage stratum {stratum} rationale")
            if matching_ids:
                raise ValueError(f"Inapplicable coverage stratum {stratum} cannot have planned interventions")
        else:
            raise ValueError(f"Coverage stratum {stratum} must be targeted or inapplicable")
        normalized[stratum] = _freeze(dict(disposition))
    return MappingProxyType(normalized)


def _scenario_from_dict(data: Any) -> CollectionScenario:
    scenario = _require_object(data, "Collection scenario")
    fields = {
        "scenario_id",
        "exposure_role",
        *SCENARIO_PROJECTION_FIELDS,
        "expected_initial_engine_state_identity",
        "retry_policy",
        "negative_specification",
        "interventions",
        "source_dispositions",
        "coverage_strata",
    }
    _require_exact_keys(scenario, fields, "Collection scenario")
    projection = {key: scenario[key] for key in SCENARIO_PROJECTION_FIELDS}
    manifest, _ = load_scenario_manifest_projection(projection, required=True)
    assert manifest is not None
    expected_initial_identity = _require_nonempty_string(
        scenario["expected_initial_engine_state_identity"],
        "Collection scenario expected_initial_engine_state_identity",
    )
    exposure_role = _require_nonempty_string(scenario["exposure_role"], "Collection scenario exposure_role")
    if exposure_role not in EXPOSURE_ROLES:
        raise ValueError("Collection scenario exposure_role is unknown")
    intervention_data = _require_list(scenario["interventions"], "Collection scenario interventions")
    if not intervention_data:
        raise ValueError("Collection scenario requires interventions")
    interventions = tuple(_intervention_from_dict(item) for item in intervention_data)
    ids = [intervention.id for intervention in interventions]
    if len(ids) != len(set(ids)):
        raise ValueError("Collection intervention IDs must be unique")
    if [intervention.ordinal for intervention in interventions] != list(range(1, len(interventions) + 1)):
        raise ValueError("Collection intervention ordinals must be contiguous and ordered")
    return CollectionScenario(
        scenario_id=_require_nonempty_string(scenario["scenario_id"], "Collection scenario id"),
        exposure_role=exposure_role,
        scenario_manifest_projection=_freeze(_json_value(projection, "Collection scenario manifest projection")),
        expected_initial_engine_state_identity=expected_initial_identity,
        retry_policy=_retry_policy_from_dict(scenario["retry_policy"]),
        negative_specification=_negative_specification_from_dict(scenario["negative_specification"], interventions),
        interventions=interventions,
        source_dispositions=_validate_source_dispositions(scenario["source_dispositions"], interventions),
        coverage_strata=_validate_coverage_strata(scenario["coverage_strata"], interventions),
    )


def _plan_identity(plan_version: int, scenarios: tuple[CollectionScenario, ...]) -> str:
    return _identity(
        PLAN_IDENTITY_NAMESPACE,
        {"schema": SCHEMA, "plan_version": plan_version, "scenarios": [scenario.to_dict() for scenario in scenarios]},
    )


def create_collection_plan(*, plan_version: int, scenarios: Sequence[Mapping[str, Any]]) -> CollectionPlan:
    """Build a validated plan whose identity excludes the identity field itself."""
    draft = {"schema": SCHEMA, "plan_version": plan_version, "identity": "draft", "scenarios": list(scenarios)}
    _require_exact_keys(draft, {"schema", "plan_version", "identity", "scenarios"}, "Collection plan")
    if isinstance(plan_version, bool) or not isinstance(plan_version, int) or plan_version <= 0:
        raise ValueError("Collection plan version must be a positive integer")
    scenario_values = tuple(_scenario_from_dict(item) for item in draft["scenarios"])
    if not scenario_values:
        raise ValueError("Collection plan requires at least one scenario")
    scenario_ids = [scenario.scenario_id for scenario in scenario_values]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("Collection plan scenario IDs must be unique")
    return CollectionPlan(SCHEMA, plan_version, _plan_identity(plan_version, scenario_values), scenario_values)


def write_collection_plan(plan: CollectionPlan, path: Path) -> Path:
    """Atomically publish a canonical JSON collection plan."""
    if not isinstance(plan, CollectionPlan):
        raise ValueError("write_collection_plan requires a CollectionPlan")
    CollectionPlan.from_dict(plan.to_dict())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(plan.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return _write_bytes_atomic(path, content.encode("utf-8"))


def load_collection_plan(path: Path) -> LoadedCollectionPlan:
    path = Path(path)
    try:
        original_bytes = path.read_bytes()
        data = json.loads(original_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load collection plan {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("Collection plan root must be an object")
    return LoadedCollectionPlan(CollectionPlan.from_dict(data), path, original_bytes)


def assert_plan_unchanged(loaded: LoadedCollectionPlan, path: Path) -> None:
    """Require an on-disk plan still to be the exact bytes loaded for execution."""
    if not isinstance(loaded, LoadedCollectionPlan):
        raise ValueError("assert_plan_unchanged requires a LoadedCollectionPlan")
    try:
        current_bytes = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot read collection plan {path}: {exc}") from exc
    if current_bytes != loaded.original_bytes:
        raise ValueError("Collection plan bytes changed since loading")


def _result_from_runtime(value: RuntimeResult | Mapping[str, Any]) -> RuntimeResult:
    if isinstance(value, RuntimeResult):
        result = value
    else:
        data = _require_object(value, "Runtime result")
        _require_exact_keys(
            data,
            {
                "status",
                "reason",
                "failure_code",
                "realized_coverage_strata",
                "eligible",
                "artifact_path",
                "quarantine_path",
                "failure_manifest_path",
            },
            "Runtime result",
        )
        coverage = _require_list(data["realized_coverage_strata"], "Runtime result realized_coverage_strata")
        result = RuntimeResult(
            data["status"],
            data["reason"],
            data["failure_code"],
            tuple(coverage),
            data["eligible"],
            data["artifact_path"],
            data["quarantine_path"],
            data["failure_manifest_path"],
        )
    if result.status not in {"accepted", "rejected", "failed"}:
        raise ValueError("Runtime result has unknown status")
    if not isinstance(result.eligible, bool):
        raise ValueError("Runtime result eligible must be a boolean")
    for field, path in (
        ("artifact_path", result.artifact_path),
        ("quarantine_path", result.quarantine_path),
        ("failure_manifest_path", result.failure_manifest_path),
    ):
        if path is not None and (not isinstance(path, str) or not path):
            raise ValueError(f"Runtime result {field} must be a nonempty string when provided")
    coverage = result.realized_coverage_strata
    if not isinstance(coverage, tuple) or any(stratum not in REQUIRED_COVERAGE_STRATA for stratum in coverage) or len(coverage) != len(set(coverage)):
        raise ValueError("Runtime result realized_coverage_strata must be a unique subset of required strata")
    if result.status == "accepted":
        if result.reason is not None or result.failure_code is not None:
            raise ValueError("Accepted runtime result cannot report reason or failure code")
        if not result.eligible:
            raise ValueError("Accepted runtime result requires eligible=True")
        if not isinstance(result.artifact_path, str) or not result.artifact_path:
            raise ValueError("Accepted runtime result requires a nonempty artifact_path")
    elif (
        not isinstance(result.reason, str)
        or not result.reason
        or not isinstance(result.failure_code, str)
        or not result.failure_code
        or not isinstance(result.quarantine_path, str)
        or not result.quarantine_path
        or not isinstance(result.failure_manifest_path, str)
        or not result.failure_manifest_path
        or coverage
    ):
        raise ValueError(
            "Rejected or failed runtime result requires nonempty reason, failure_code, "
            "quarantine_path, and failure_manifest_path and no coverage"
        )
    return result


def _call_runtime(runtime: CollectionPlanRuntime, request: RuntimeInput) -> RuntimeResult:
    if not callable(runtime):
        raise ValueError("Collection plan runtime must be callable")
    try:
        value = runtime(request)
    except Exception as exc:
        raise _RuntimeCallbackError from exc
    return _result_from_runtime(value)


def _disposition_for_result(result: RuntimeResult, policy: RetryPolicy, attempt_number: int) -> _AttemptDisposition:
    if result.status == "accepted":
        return _AttemptDisposition("accepted", "none", "none", "accept", "accepted")
    if result.status == "rejected":
        return _AttemptDisposition("quarantined", "stop", "permanent", "quarantine", "rejected")
    if result.failure_code == "runtime_exception":
        return _AttemptDisposition("quarantined", "stop", "permanent", "quarantine", "runtime_exception")
    if result.failure_code in policy.transient_failure_codes:
        if attempt_number < policy.max_attempts:
            return _AttemptDisposition("quarantined", "retry", "transient", "retry", "transient_failure")
        return _AttemptDisposition("quarantined", "stop", "transient", "quarantine", "retry_exhausted")
    return _AttemptDisposition("quarantined", "stop", "permanent", "quarantine", "permanent_failure")


def _failure_manifest_data(
    request: RuntimeInput,
    result: RuntimeResult,
    disposition: _AttemptDisposition,
) -> dict[str, Any]:
    return {
        "schema": "collection_plan_failure_manifest_v1",
        "plan_identity": request.plan_identity,
        "plan_version": request.plan_version,
        "scenario_id": request.scenario_id,
        "scenario_identity": request.scenario_identity,
        "intervention_id": request.intervention_id,
        "intervention_identity": request.intervention_identity,
        "attempt_id": request.attempt_id,
        "attempt_number": request.attempt_number,
        "status": result.status,
        "reason": result.reason,
        "failure_code": result.failure_code,
        "artifact_disposition": disposition.artifact_disposition,
        "failure_class": disposition.failure_class,
        "retry_decision": disposition.retry_decision,
        "quarantine_path": result.quarantine_path,
        "failure_manifest_path": result.failure_manifest_path,
    }


def _write_failure_manifest(path: Path, data: Mapping[str, Any]) -> None:
    content = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes_atomic(path, content.encode("utf-8"))


def _record_failure_manifest(
    output_dir: Path,
    request: RuntimeInput,
    result: RuntimeResult,
    disposition: _AttemptDisposition,
) -> None:
    assert isinstance(result.failure_manifest_path, str)
    manifest_path = Path(result.failure_manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = output_dir / manifest_path
    try:
        adapter_data: dict[str, Any] = {}
        if manifest_path.exists():
            if not manifest_path.is_file():
                raise ValueError("Failure manifest path is not a file")
            adapter_data = dict(_require_object(json.loads(manifest_path.read_text(encoding="utf-8")), "Failure manifest"))
        adapter_data.update(_failure_manifest_data(request, result, disposition))
        _write_failure_manifest(manifest_path, adapter_data)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Cannot record failure manifest {result.failure_manifest_path}") from exc


def _runtime_exception_result(request: RuntimeInput, error: Exception) -> RuntimeResult:
    quarantine_path = Path("quarantine") / request.attempt_id
    failure_manifest_path = quarantine_path / "failure_manifest.json"
    return RuntimeResult(
        "failed",
        reason=str(error) or error.__class__.__name__,
        failure_code="runtime_exception",
        eligible=False,
        quarantine_path=str(quarantine_path),
        failure_manifest_path=str(failure_manifest_path),
    )


def _record_fallback_failure_manifest(
    output_dir: Path,
    request: RuntimeInput,
    result: RuntimeResult,
    disposition: _AttemptDisposition,
) -> None:
    assert isinstance(result.quarantine_path, str)
    assert isinstance(result.failure_manifest_path, str)
    quarantine_path = output_dir / result.quarantine_path
    manifest_path = output_dir / result.failure_manifest_path
    if manifest_path.parent != quarantine_path or manifest_path.name != "failure_manifest.json":
        raise ValueError("Fallback failure manifest must be inside its quarantine directory")
    temporary: Path | None = None
    try:
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        if quarantine_path.exists():
            raise ValueError("Fallback quarantine directory already exists")
        temporary = Path(tempfile.mkdtemp(prefix=".fallback-", dir=quarantine_path.parent))
        _write_failure_manifest(temporary / manifest_path.name, _failure_manifest_data(request, result, disposition))
        os.replace(temporary, quarantine_path)
    except (OSError, ValueError) as exc:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise ValueError(f"Cannot record fallback failure manifest {result.failure_manifest_path}") from exc


def _write_report(report: dict[str, Any], path: Path) -> Path:
    content = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return _write_bytes_atomic(path, content.encode("utf-8"))


def _write_bytes_atomic(path: Path, content: bytes) -> Path:
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return path


def execute_collection_plan(loaded: LoadedCollectionPlan, runtime: CollectionPlanRuntime, output_dir: Path) -> dict[str, Any]:
    """Execute every frozen intervention in artifact order and write an accounting report."""
    if not isinstance(loaded, LoadedCollectionPlan):
        raise ValueError("execute_collection_plan requires a LoadedCollectionPlan")
    assert_plan_unchanged(loaded, loaded.path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_copy = output_dir / PLAN_COPY_FILENAME
    report_path = output_dir / REPORT_FILENAME
    if plan_copy.exists() or report_path.exists():
        if not plan_copy.is_file() or plan_copy.read_bytes() != loaded.original_bytes:
            raise ValueError("Collection output contains a different frozen plan")
        raise ValueError("Collection output already contains execution state")
    _write_bytes_atomic(plan_copy, loaded.original_bytes)

    ledger: list[dict[str, Any]] = []
    planned_slots: list[dict[str, Any]] = []
    realized_counts = {stratum: 0 for stratum in REQUIRED_COVERAGE_STRATA}
    unmet_slots: list[dict[str, Any]] = []
    realized_coverage_shortfalls: list[dict[str, Any]] = []
    counts = {"accepted": 0, "rejected": 0, "failed": 0}

    def checkpoint_report() -> dict[str, Any]:
        quarantined_attempts = [entry for entry in ledger if entry["artifact_disposition"] == "quarantined"]
        report = {
            "schema": "collection_plan_execution_report_v2",
            "plan_identity": loaded.plan.identity,
            "plan_version": loaded.plan.plan_version,
            "attempt_ledger": ledger,
            "accepted_count": counts["accepted"],
            "rejected_count": counts["rejected"],
            "failed_count": counts["failed"],
            "quarantined_attempts": quarantined_attempts,
            "quarantined_count": len(quarantined_attempts),
            "planned_slots": planned_slots,
            "realized_coverage_stratum_counts": realized_counts,
            "unmet_slots": unmet_slots,
            "realized_coverage_shortfalls": realized_coverage_shortfalls,
        }
        _write_report(report, report_path)
        return report

    checkpoint_report()

    for scenario in loaded.plan.scenarios:
        for intervention in scenario.interventions:
            attempt_entries: list[dict[str, Any]] = []
            realized_on_slot: set[str] = set()
            terminal_status = "failed"
            terminal_eligible = False
            for attempt_number in range(1, scenario.retry_policy.max_attempts + 1):
                attempt_id = _identity(
                    ATTEMPT_IDENTITY_NAMESPACE,
                    {
                        "plan_identity": loaded.plan.identity,
                        "scenario_identity": scenario.identity,
                        "intervention_identity": intervention.identity,
                        "attempt_number": attempt_number,
                    },
                )
                request = RuntimeInput(
                    plan_identity=loaded.plan.identity,
                    plan_version=loaded.plan.plan_version,
                    scenario_id=scenario.scenario_id,
                    scenario_identity=scenario.identity,
                    intervention_id=intervention.id,
                    intervention_identity=intervention.identity,
                    attempt_id=attempt_id,
                    attempt_number=attempt_number,
                    expected_initial_engine_state_identity=scenario.expected_initial_engine_state_identity,
                    interface_action=intervention.interface_action,
                    engine_relative_action=intervention.engine_relative_action,
                    mapping_version=intervention.mapping_version,
                    slingshot_reference=intervention.slingshot_reference,
                )
                try:
                    result = _call_runtime(runtime, request)
                except _RuntimeCallbackError as exc:
                    error = exc.__cause__ if isinstance(exc.__cause__, Exception) else exc
                    result = _runtime_exception_result(request, error)
                    fallback_failure_manifest = True
                else:
                    fallback_failure_manifest = False
                disposition = _disposition_for_result(result, scenario.retry_policy, attempt_number)
                if result.status != "accepted":
                    if fallback_failure_manifest:
                        _record_fallback_failure_manifest(output_dir, request, result, disposition)
                    else:
                        _record_failure_manifest(output_dir, request, result, disposition)
                terminal_status = result.status
                terminal_eligible = result.status == "accepted"
                counts[result.status] += 1
                entry = {
                    "plan_identity": request.plan_identity,
                    "scenario_id": request.scenario_id,
                    "scenario_identity": request.scenario_identity,
                    "intervention_id": request.intervention_id,
                    "intervention_identity": request.intervention_identity,
                    "intervention_ordinal": intervention.ordinal,
                    "attempt_id": request.attempt_id,
                    "attempt_number": request.attempt_number,
                    "status": result.status,
                    "eligible": result.eligible,
                    "reason": result.reason,
                    "disposition": disposition.disposition,
                    "disposition_reason": disposition.disposition_reason,
                    "artifact_disposition": disposition.artifact_disposition,
                    "retry_decision": disposition.retry_decision,
                    "failure_class": disposition.failure_class,
                    "failure_code": result.failure_code,
                    "artifact_path": result.artifact_path,
                    "quarantine_path": result.quarantine_path,
                    "failure_manifest_path": result.failure_manifest_path,
                    "realized_coverage_strata": list(result.realized_coverage_strata),
                }
                ledger.append(entry)
                attempt_entries.append(entry)
                if result.status == "accepted":
                    for stratum in result.realized_coverage_strata:
                        realized_counts[stratum] += 1
                        realized_on_slot.add(stratum)
                checkpoint_report()
                if disposition.retry_decision == "retry":
                    continue
                break

            slot_disposition = terminal_status
            if terminal_status == "accepted":
                slot_disposition = "completed" if terminal_eligible else "ineligible"
            planned_slots.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "scenario_identity": scenario.identity,
                    "intervention_id": intervention.id,
                    "intervention_identity": intervention.identity,
                    "ordinal": intervention.ordinal,
                    "intended_coverage_stratum": intervention.intended_coverage_stratum,
                    "source": intervention.source,
                    "disposition": slot_disposition,
                    "terminal_status": terminal_status,
                    "attempt_ids": [entry["attempt_id"] for entry in attempt_entries],
                }
            )
            if not terminal_eligible:
                unmet_slots.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "intervention_id": intervention.id,
                        "intended_coverage_stratum": intervention.intended_coverage_stratum,
                        "disposition": slot_disposition,
                    }
                )
            elif intervention.intended_coverage_stratum not in realized_on_slot:
                realized_coverage_shortfalls.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "intervention_id": intervention.id,
                        "intended_coverage_stratum": intervention.intended_coverage_stratum,
                        "realized_coverage_strata": sorted(realized_on_slot),
                    }
                )
            checkpoint_report()

    assert_plan_unchanged(loaded, loaded.path)
    if plan_copy.read_bytes() != loaded.original_bytes:
        raise ValueError("Copied collection plan bytes changed during execution")
    return checkpoint_report()
