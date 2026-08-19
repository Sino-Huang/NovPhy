"""Immutable, evidence-bound production parameter plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Final, Literal

from scripts.collection_plan import (
    CollectionPlan,
    CollectionPlanRuntime,
    LoadedCollectionPlan,
    PLAN_COPY_FILENAME,
    REPORT_FILENAME,
    execute_collection_plan,
)
from scripts.representative_pilot import PilotReport


SCHEMA: Final = "production_parameter_plan_v1"
IDENTITY_NAMESPACE: Final = "production-parameter-plan-v1"
PRODUCTION_PLAN_COPY_FILENAME: Final = "production_parameter_plan.json"
PARAMETER_GROUPS: Final = frozenset((
    "capture",
    "tolerances",
    "prospective_quotas",
    "bounded_negative_cap",
    "transient_retry_counts",
))
_TOP_LEVEL_FIELDS: Final = frozenset((
    "schema",
    "plan_version",
    "identity",
    "source_pilot_report",
    "source_collection_plan",
    "parameters",
    "evidence",
))


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


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


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str] | frozenset[str], name: str) -> None:
    if set(value) != set(expected):
        raise ValueError(f"{name} is incomplete or contains unknown fields")


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _validated_parameters(value: Any) -> Mapping[str, Any]:
    parameters = _require_mapping(value, "Production parameters")
    _require_exact_keys(parameters, PARAMETER_GROUPS, "Production parameters")

    capture = _require_mapping(parameters["capture"], "Production capture parameters")
    _require_exact_keys(
        capture,
        {"capture_stride", "stability_window", "rollout_ceiling"},
        "Production capture parameters",
    )
    normalized_capture = {
        key: _require_positive_integer(capture[key], f"Production capture {key}")
        for key in ("capture_stride", "stability_window", "rollout_ceiling")
    }

    tolerances = _require_mapping(parameters["tolerances"], "Production tolerances")
    _require_exact_keys(tolerances, {"geometric", "motion", "numeric"}, "Production tolerances")
    normalized_tolerances: dict[str, float | int] = {}
    for key in ("geometric", "motion", "numeric"):
        item = tolerances[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item < 0:
            raise ValueError(f"Production tolerance {key} must be nonnegative and finite")
        normalized_tolerances[key] = item

    quotas = _require_mapping(parameters["prospective_quotas"], "Production prospective_quotas")
    if not quotas:
        raise ValueError("Production prospective_quotas must be nonempty")
    normalized_quotas = {
        _require_nonempty_string(key, "Production prospective quota key"):
        _require_positive_integer(item, f"Production prospective quota {key}")
        for key, item in quotas.items()
    }

    retries = _require_mapping(parameters["transient_retry_counts"], "Production transient_retry_counts")
    normalized_retries = {
        _require_nonempty_string(key, "Production transient retry key"):
        _require_nonnegative_integer(item, f"Production transient retry count {key}")
        for key, item in retries.items()
    }

    return _freeze({
        "capture": normalized_capture,
        "tolerances": normalized_tolerances,
        "prospective_quotas": normalized_quotas,
        "bounded_negative_cap": _require_nonnegative_integer(
            parameters["bounded_negative_cap"],
            "Production bounded_negative_cap",
        ),
        "transient_retry_counts": normalized_retries,
    })


def _validated_evidence(value: Any, parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    def validate_node(evidence_node: Any, parameter_node: Any, path: str) -> Any:
        if isinstance(parameter_node, Mapping):
            evidence_mapping = _require_mapping(evidence_node, f"Production evidence {path}")
            _require_exact_keys(
                evidence_mapping,
                set(parameter_node),
                f"Production evidence {path}",
            )
            return {
                key: validate_node(
                    evidence_mapping[key],
                    parameter_node[key],
                    f"{path}.{key}" if path else key,
                )
                for key in parameter_node
            }

        justification = _require_mapping(evidence_node, f"Production evidence {path}")
        _require_exact_keys(
            justification,
            {"attempt_ids", "rationale", "derivation"},
            f"Production evidence {path}",
        )
        attempt_ids = justification["attempt_ids"]
        if not isinstance(attempt_ids, (list, tuple)) or not attempt_ids:
            raise ValueError(f"Production evidence {path} attempt_ids must be nonempty")
        normalized_ids = tuple(
            _require_nonempty_string(item, f"Production evidence {path} attempt ID")
            for item in attempt_ids
        )
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError(f"Production evidence {path} attempt_ids must be unique")
        return {
            "attempt_ids": normalized_ids,
            "rationale": _require_nonempty_string(
                justification["rationale"],
                f"Production evidence {path} rationale",
            ),
            "derivation": _require_nonempty_string(
                justification["derivation"],
                f"Production evidence {path} derivation",
            ),
        }

    evidence = _require_mapping(value, "Production evidence")
    return _freeze(validate_node(evidence, parameters, ""))


def _evidence_leaves(value: Mapping[str, Any]) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    leaves: list[tuple[str, Mapping[str, Any]]] = []

    def visit(node: Mapping[str, Any], path: str) -> None:
        if set(node) == {"attempt_ids", "rationale", "derivation"}:
            leaves.append((path, node))
            return
        for key, item in node.items():
            visit(item, f"{path}.{key}" if path else key)

    visit(value, "")
    return tuple(leaves)


def _source(value: Any, name: str, version_field: str | None = None) -> Mapping[str, Any]:
    source = _require_mapping(value, name)
    expected = {"identity"} if version_field is None else {"identity", version_field}
    _require_exact_keys(source, expected, name)
    normalized = {"identity": _require_nonempty_string(source["identity"], f"{name} identity")}
    if version_field is not None:
        normalized[version_field] = _require_positive_integer(source[version_field], f"{name} {version_field}")
    return _freeze(normalized)


def _identity(payload: Mapping[str, Any]) -> str:
    identity_payload = _thaw(payload)
    identity_payload.pop("identity", None)
    return f"{IDENTITY_NAMESPACE}:sha256:{sha256(_canonical_json(identity_payload)).hexdigest()}"


@dataclass(frozen=True, slots=True)
class ProductionPlan:
    schema: Literal["production_parameter_plan_v1"]
    plan_version: int
    identity: str
    source_pilot_report: Mapping[str, Any]
    source_collection_plan: Mapping[str, Any]
    parameters: Mapping[str, Any]
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_version": self.plan_version,
            "identity": self.identity,
            "source_pilot_report": _thaw(self.source_pilot_report),
            "source_collection_plan": _thaw(self.source_collection_plan),
            "parameters": _thaw(self.parameters),
            "evidence": _thaw(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProductionPlan":
        raw = _require_mapping(data, "Production plan")
        _require_exact_keys(raw, _TOP_LEVEL_FIELDS, "Production plan")
        if raw["schema"] != SCHEMA:
            raise ValueError(f"Unsupported production plan schema: {raw['schema']}")
        parameters = _validated_parameters(raw["parameters"])
        plan = cls(
            SCHEMA,
            _require_positive_integer(raw["plan_version"], "Production plan version"),
            _require_nonempty_string(raw["identity"], "Production plan identity"),
            _source(raw["source_pilot_report"], "Source pilot report"),
            _source(raw["source_collection_plan"], "Source collection plan", "plan_version"),
            parameters,
            _validated_evidence(raw["evidence"], parameters),
        )
        if plan.identity != _identity(plan.to_dict()):
            raise ValueError("Production plan identity is stale")
        return plan


def _accepted_attempt_roles(report: PilotReport, collection_plan: CollectionPlan) -> Mapping[str, str]:
    attempts = _require_mapping(report.attempts, "Pilot report attempts")
    pilot_evidence = _require_mapping(attempts.get("pilot_evidence"), "Pilot report pilot_evidence")
    accepted_ids_value = pilot_evidence.get("accepted_attempt_ids")
    if not isinstance(accepted_ids_value, (list, tuple)):
        raise ValueError("Pilot report accepted attempt IDs are malformed")
    accepted_ids = set(accepted_ids_value)
    validation = attempts.get("atomic_validation")
    if not isinstance(validation, (list, tuple)):
        raise ValueError("Pilot report atomic validation is malformed")
    scenario_roles = {scenario.scenario_id: scenario.exposure_role for scenario in collection_plan.scenarios}
    roles: dict[str, str] = {}
    for item in validation:
        entry = _require_mapping(item, "Pilot report atomic validation entry")
        attempt_id = _require_nonempty_string(entry.get("attempt_id"), "Pilot evidence attempt ID")
        if attempt_id in roles:
            raise ValueError("Pilot report atomic validation attempt IDs must be unique")
        if entry.get("accepted") is True and attempt_id in accepted_ids:
            scenario_id = _require_nonempty_string(entry.get("scenario_id"), "Pilot evidence scenario ID")
            if scenario_id not in scenario_roles:
                raise ValueError("Pilot evidence references a scenario outside the collection plan")
            roles[attempt_id] = scenario_roles[scenario_id]
    return MappingProxyType(roles)


def _validate_collection_parameters(parameters: Mapping[str, Any], collection_plan: CollectionPlan) -> None:
    negative_caps = {scenario.negative_specification.cap for scenario in collection_plan.scenarios}
    if len(negative_caps) != 1:
        raise ValueError("Collection scenarios declare non-identical bounded negative caps")
    if parameters["bounded_negative_cap"] != next(iter(negative_caps)):
        raise ValueError("Production bounded_negative_cap differs from the collection plan")

    targeted_in_every_scenario = {
        stratum
        for stratum in parameters["prospective_quotas"]
        if all(
            stratum in scenario.coverage_strata
            and scenario.coverage_strata[stratum].get("status") == "targeted"
            for scenario in collection_plan.scenarios
        )
    }
    if targeted_in_every_scenario != set(parameters["prospective_quotas"]):
        raise ValueError("Production quota keys must be targeted coverage strata in every collection scenario")

    declared_retries: dict[str, int] | None = None
    for scenario in collection_plan.scenarios:
        allowed_retries = scenario.retry_policy.max_attempts - 1
        scenario_retries = {
            code: allowed_retries
            for code in scenario.retry_policy.transient_failure_codes
        }
        if declared_retries is None:
            declared_retries = scenario_retries
        elif scenario_retries != declared_retries:
            raise ValueError("Collection scenarios declare different transient retry code maps")
    assert declared_retries is not None
    production_retries = parameters["transient_retry_counts"]
    if dict(production_retries) != declared_retries:
        raise ValueError("Production transient_retry_counts must exactly match every collection scenario")


def create_production_plan(
    *,
    plan_version: int,
    pilot_report: PilotReport,
    collection_plan: CollectionPlan,
    parameters: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> ProductionPlan:
    """Create a production plan justified only by accepted, non-final pilot evidence."""
    if not isinstance(pilot_report, PilotReport):
        raise ValueError("pilot_report must be a PilotReport")
    if not isinstance(collection_plan, CollectionPlan):
        raise ValueError("collection_plan must be a CollectionPlan")
    PilotReport.from_dict(pilot_report.to_dict())
    CollectionPlan.from_dict(collection_plan.to_dict())
    if pilot_report.pilot_status != "accepted" or pilot_report.acceptance_decision.get("accepted") is not True:
        raise ValueError("Production plans require an accepted pilot report")
    if (
        pilot_report.plan_identity != collection_plan.identity
        or pilot_report.plan_version != collection_plan.plan_version
    ):
        raise ValueError("Pilot report is not bound to the supplied collection plan")

    normalized_parameters = _validated_parameters(parameters)
    _validate_collection_parameters(normalized_parameters, collection_plan)
    normalized_evidence = _validated_evidence(evidence, normalized_parameters)
    accepted_roles = _accepted_attempt_roles(pilot_report, collection_plan)
    for parameter_path, justification in _evidence_leaves(normalized_evidence):
        for attempt_id in justification["attempt_ids"]:
            if attempt_id not in accepted_roles:
                raise ValueError(f"Production evidence {parameter_path} references a nonaccepted pilot attempt")
            if accepted_roles[attempt_id] == "final_evaluation":
                raise ValueError(f"Production evidence {parameter_path} cannot use final_evaluation outcomes")

    payload = {
        "schema": SCHEMA,
        "plan_version": _require_positive_integer(plan_version, "Production plan version"),
        "identity": "",
        "source_pilot_report": {"identity": pilot_report.identity},
        "source_collection_plan": {
            "identity": collection_plan.identity,
            "plan_version": collection_plan.plan_version,
        },
        "parameters": _thaw(normalized_parameters),
        "evidence": _thaw(normalized_evidence),
    }
    payload["identity"] = _identity(payload)
    return ProductionPlan.from_dict(payload)


def production_plan_path(publication_dir: Path, plan: ProductionPlan) -> Path:
    """Return the authoritative version-addressed publication path for a plan."""
    if not isinstance(plan, ProductionPlan):
        raise ValueError("production_plan_path requires a ProductionPlan")
    collection_digest = sha256(plan.source_collection_plan["identity"].encode("utf-8")).hexdigest()
    return Path(publication_dir) / f"production_parameter_plan_{collection_digest}_v{plan.plan_version}.json"


def write_production_plan(plan: ProductionPlan, publication_dir: Path) -> Path:
    """Atomically publish a plan at its authoritative version-addressed path."""
    if not isinstance(plan, ProductionPlan):
        raise ValueError("write_production_plan requires a ProductionPlan")
    payload = ProductionPlan.from_dict(plan.to_dict()).to_dict()
    directory = Path(publication_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = production_plan_path(directory, plan)
    content = (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if target.exists():
        try:
            existing = target.read_bytes()
        except OSError as error:
            raise ValueError(f"Cannot read existing production plan {target}: {error}") from error
        if existing == content:
            return target
        raise ValueError("Production plan path already contains different bytes")
    with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        os.link(temporary, target)
    except FileExistsError:
        try:
            if target.read_bytes() == content:
                return target
        except OSError as error:
            raise ValueError(f"Cannot read existing production plan {target}: {error}") from error
        raise ValueError("Production plan path already contains different bytes")
    except OSError:
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_production_plan(path: Path) -> ProductionPlan:
    target = Path(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load production plan {target}: {error}") from error
    return ProductionPlan.from_dict(_require_mapping(data, "Production plan"))


def execute_production_plan(
    loaded_collection_plan: LoadedCollectionPlan,
    published_plan_path: Path | str,
    runtime: CollectionPlanRuntime,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute collection only after validating a source-bound production plan."""
    if not isinstance(loaded_collection_plan, LoadedCollectionPlan):
        raise ValueError("execute_production_plan requires a LoadedCollectionPlan")
    if not isinstance(published_plan_path, (Path, str)):
        raise ValueError("execute_production_plan requires a published production plan path")
    source_path = Path(published_plan_path)
    try:
        source_bytes = source_path.read_bytes()
        source_data = json.loads(source_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load production plan {source_path}: {error}") from error
    validated = ProductionPlan.from_dict(_require_mapping(source_data, "Production plan"))
    authoritative_path = production_plan_path(source_path.parent, validated)
    if source_path.resolve() != authoritative_path.resolve():
        raise ValueError("Production plan must be loaded from its authoritative publication path")
    source = validated.source_collection_plan
    if (
        source["identity"] != loaded_collection_plan.plan.identity
        or source["plan_version"] != loaded_collection_plan.plan.plan_version
    ):
        raise ValueError("Production plan is not bound to the loaded collection plan")
    _validate_collection_parameters(validated.parameters, loaded_collection_plan.plan)

    output = Path(output_dir)
    frozen_path = output / PRODUCTION_PLAN_COPY_FILENAME
    if (output / PLAN_COPY_FILENAME).exists() or (output / REPORT_FILENAME).exists():
        raise ValueError("Production output already contains collection execution state")
    if frozen_path.exists():
        try:
            frozen_bytes = frozen_path.read_bytes()
        except OSError as error:
            raise ValueError(f"Cannot read frozen production plan {frozen_path}: {error}") from error
        if frozen_bytes != source_bytes:
            raise ValueError("Production output contains a different frozen production plan")
    else:
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", dir=output, delete=False) as handle:
            handle.write(source_bytes)
            temporary = Path(handle.name)
        try:
            os.link(temporary, frozen_path)
        except FileExistsError:
            if frozen_path.read_bytes() != source_bytes:
                raise ValueError("Production output contains a different frozen production plan")
        finally:
            temporary.unlink(missing_ok=True)

    execution_context = _freeze({
        "production_plan_identity": validated.identity,
        "production_plan_version": validated.plan_version,
        "source_pilot_report_identity": validated.source_pilot_report["identity"],
        "parameters": validated.parameters,
    })
    try:
        return execute_collection_plan(
            loaded_collection_plan,
            runtime,
            output,
            execution_context=execution_context,
        )
    finally:
        try:
            final_bytes = frozen_path.read_bytes()
        except OSError as error:
            raise ValueError("Frozen production plan changed during execution") from error
        if final_bytes != source_bytes:
            raise ValueError("Frozen production plan changed during execution")
