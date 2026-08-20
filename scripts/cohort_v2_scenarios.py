"""Source-bound scenario records for prospective central-v2 cohort planning.

This module intentionally wraps, rather than changes, ``scenario_manifest_v1``.
Existing pilot and cohort manifests therefore retain their original semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
import posixpath
from pathlib import Path
import re
import tempfile
from typing import Any, Literal, Mapping, Sequence
import xml.etree.ElementTree as ET
import zipfile

from scripts.cohort_v2_capabilities import (
    build_central_v2_scope_claim,
    validate_central_v2_scope_claim,
)
from scripts.scenario_manifest import (
    ELIGIBLE,
    BenchmarkCondition,
    ScenarioManifest,
    verify_replay,
)


TEMPLATE_SCHEMA = "scenario_template_v1"
TEMPLATE_CONSTRAINTS_SCHEMA = "scenario_template_constraints_v1"
SCENARIO_SCHEMA = "cohort_v2_scenario_manifest_v1"
RECEIPT_SCHEMA = "deterministic_scenario_receipt_v1"
INVENTORY_SCHEMA = "central_v2_scenario_inventory_v1"
INVENTORY_DRAFT_SCHEMA = "central_v2_scenario_inventory_draft_v1"
_XML_CONTENT_IDENTITY = re.compile(r"^xml_bytes_v1:sha256:[0-9a-f]{64}$")
_XLSX_CONTENT_IDENTITY = re.compile(r"^xlsx_bytes_v1:sha256:[0-9a-f]{64}$")
_ARTIFACT_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NORMALIZED_INITIAL_STATE = re.compile(
    r"^normalized-initial-engine-state-v1:sha256:[0-9a-f]{64}$"
)
_ISSUE_45_COMMENT_URL = re.compile(
    r"^https://github\.com/Sino-Huang/NovPhy/issues/45#issuecomment-[0-9]+$"
)

ScenarioLineageFailureReason = Literal[
    "missing_template_identity",
    "unresolved_source_provenance",
    "content_drift",
    "cross_lineage_reuse",
    "initial_state_mismatch",
    "smoke_only",
]


class ScenarioLineageError(ValueError):
    """A typed cohort-v2 lineage admission failure."""

    def __init__(self, reason: ScenarioLineageFailureReason, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


ExposureRole = Literal[
    "training", "calibration", "model_selection", "final_evaluation"
]
InventoryState = Literal["planned_non_final", "sealed_final"]
_ROLES: tuple[ExposureRole, ...] = (
    "training",
    "calibration",
    "model_selection",
    "final_evaluation",
)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("Cohort-v2 scenario artifact contains non-JSON data") from error


def _identity(namespace: str, value: Any) -> str:
    return f"{namespace}:sha256:{sha256(_canonical_json(value)).hexdigest()}"


def _content_identity(content: bytes) -> str:
    return f"xml_bytes_v1:sha256:{sha256(content).hexdigest()}"


def _artifact_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def write_immutable_cohort_v2_bytes(content: bytes, path: Path) -> Path:
    """Write immutable cohort-v2 evidence bytes, accepting an identical rebuild."""
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite immutable cohort-v2 artifact: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return path


def write_immutable_cohort_v2_json(value: Mapping[str, Any], path: Path) -> Path:
    return write_immutable_cohort_v2_bytes(_artifact_bytes(value), path)


def _immutable_write(path: Path, value: Mapping[str, Any]) -> Path:
    return write_immutable_cohort_v2_json(value, path)


def _load_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load {label} {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} root must be an object")
    return value


def _condition_payload(condition: BenchmarkCondition) -> dict[str, str]:
    return {
        "identity": condition.identity,
        "novelty_level": condition.novelty_level,
        "novelty_type": condition.novelty_type,
    }


def _coordinate_pair(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2 or any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ValueError(f"Scenario-template constraints {label} must contain two numbers")
    return (float(value[0]), float(value[1]))


@dataclass(frozen=True, slots=True)
class ScenarioTemplateConstraints:
    """Reviewed generator inputs bound to one exact constraints workbook row."""

    source_reference: str
    source_content_identity: str
    sheet_name: str
    row_number: int
    canonical_generator_template_name: str
    reference_point: tuple[float, float]
    min_coordinate: tuple[float, float]
    max_coordinate: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TEMPLATE_CONSTRAINTS_SCHEMA,
            "source_reference": self.source_reference,
            "source_content_identity": self.source_content_identity,
            "sheet_name": self.sheet_name,
            "row_number": self.row_number,
            "canonical_generator_template_name": self.canonical_generator_template_name,
            "reference_point": list(self.reference_point),
            "min_coordinate": list(self.min_coordinate),
            "max_coordinate": list(self.max_coordinate),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScenarioTemplateConstraints":
        required = {
            "schema",
            "source_reference",
            "source_content_identity",
            "sheet_name",
            "row_number",
            "canonical_generator_template_name",
            "reference_point",
            "min_coordinate",
            "max_coordinate",
        }
        if set(value) != required or value["schema"] != TEMPLATE_CONSTRAINTS_SCHEMA:
            raise ValueError("Scenario-template constraints are incomplete or unsupported")
        if not isinstance(value["source_reference"], str) or not value["source_reference"]:
            raise ValueError("Scenario-template constraints source_reference must be nonempty")
        if not isinstance(value["source_content_identity"], str) or _XLSX_CONTENT_IDENTITY.fullmatch(value["source_content_identity"]) is None:
            raise ValueError("Scenario-template constraints source_content_identity is invalid")
        if not isinstance(value["sheet_name"], str) or not value["sheet_name"]:
            raise ValueError("Scenario-template constraints sheet_name must be nonempty")
        if isinstance(value["row_number"], bool) or not isinstance(value["row_number"], int) or value["row_number"] < 1:
            raise ValueError("Scenario-template constraints row_number must be positive")
        if not isinstance(value["canonical_generator_template_name"], str) or not value["canonical_generator_template_name"]:
            raise ValueError("Scenario-template constraints canonical generator name must be nonempty")
        return cls(
            source_reference=value["source_reference"],
            source_content_identity=value["source_content_identity"],
            sheet_name=value["sheet_name"],
            row_number=value["row_number"],
            canonical_generator_template_name=value["canonical_generator_template_name"],
            reference_point=_coordinate_pair(value["reference_point"], "reference_point"),
            min_coordinate=_coordinate_pair(value["min_coordinate"], "min_coordinate"),
            max_coordinate=_coordinate_pair(value["max_coordinate"], "max_coordinate"),
        )


def create_scenario_template_constraints(
    workbook_content: bytes,
    *,
    source_reference: str,
    sheet_name: str,
    row_number: int,
    canonical_generator_template_name: str,
    reference_point: tuple[float, float],
    min_coordinate: tuple[float, float],
    max_coordinate: tuple[float, float],
) -> ScenarioTemplateConstraints:
    value = {
        "schema": TEMPLATE_CONSTRAINTS_SCHEMA,
        "source_reference": source_reference,
        "source_content_identity": f"xlsx_bytes_v1:sha256:{sha256(workbook_content).hexdigest()}",
        "sheet_name": sheet_name,
        "row_number": row_number,
        "canonical_generator_template_name": canonical_generator_template_name,
        "reference_point": list(reference_point),
        "min_coordinate": list(min_coordinate),
        "max_coordinate": list(max_coordinate),
    }
    return ScenarioTemplateConstraints.from_dict(value)


@dataclass(frozen=True, slots=True)
class ScenarioTemplateRecord:
    """One declared template source, bound to its exact XML bytes."""

    source_reference: str
    source_content_identity: str
    benchmark_conditions: tuple[BenchmarkCondition, ...]
    generation_constraints: ScenarioTemplateConstraints | None
    identity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TEMPLATE_SCHEMA,
            "identity": self.identity,
            "source_reference": self.source_reference,
            "source_content_identity": self.source_content_identity,
            "benchmark_conditions": [_condition_payload(item) for item in self.benchmark_conditions],
            "generation_constraints": None if self.generation_constraints is None else self.generation_constraints.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScenarioTemplateRecord":
        required = {
            "schema",
            "identity",
            "source_reference",
            "source_content_identity",
            "benchmark_conditions",
            "generation_constraints",
        }
        if set(value) != required:
            raise ValueError("Scenario-template record is incomplete or contains unknown fields")
        if value["schema"] != TEMPLATE_SCHEMA:
            raise ValueError("Scenario-template record schema is unsupported")
        if not isinstance(value["source_reference"], str) or not value["source_reference"]:
            raise ValueError("Scenario-template record source_reference must be nonempty")
        if not isinstance(value["source_content_identity"], str) or _XML_CONTENT_IDENTITY.fullmatch(value["source_content_identity"]) is None:
            raise ValueError("Scenario-template record source_content_identity is invalid")
        conditions = value["benchmark_conditions"]
        if not isinstance(conditions, list) or not conditions:
            raise ValueError("Scenario-template record must declare benchmark conditions")
        parsed: list[BenchmarkCondition] = []
        for item in conditions:
            if not isinstance(item, Mapping) or set(item) != {"identity", "novelty_level", "novelty_type"}:
                raise ValueError("Scenario-template record benchmark condition is malformed")
            condition = BenchmarkCondition(item["novelty_level"], item["novelty_type"])
            if item["identity"] != condition.identity:
                raise ValueError("Scenario-template record benchmark condition identity is stale")
            parsed.append(condition)
        ordered = tuple(sorted(parsed, key=lambda item: item.identity))
        if tuple(parsed) != ordered or len({item.identity for item in ordered}) != len(ordered):
            raise ValueError("Scenario-template record benchmark conditions must be unique and canonical")
        record = cls(
            source_reference=value["source_reference"],
            source_content_identity=value["source_content_identity"],
            benchmark_conditions=ordered,
            generation_constraints=(
                None
                if value["generation_constraints"] is None
                else ScenarioTemplateConstraints.from_dict(value["generation_constraints"])
            ),
            identity=value["identity"],
        )
        expected = _identity("scenario-template-v1", {
            "source_reference": record.source_reference,
            "source_content_identity": record.source_content_identity,
            "benchmark_conditions": [_condition_payload(item) for item in record.benchmark_conditions],
            "generation_constraints": None if record.generation_constraints is None else record.generation_constraints.to_dict(),
        })
        if record.identity != expected:
            raise ValueError("Scenario-template record identity is stale")
        return record


def create_scenario_template_record(
    source_content: bytes,
    *,
    source_reference: str,
    benchmark_conditions: Sequence[BenchmarkCondition],
    generation_constraints: ScenarioTemplateConstraints | None = None,
) -> ScenarioTemplateRecord:
    """Create a record from declared provenance and exact template content."""
    if not source_reference:
        raise ValueError("Scenario-template source_reference must be nonempty")
    ordered = tuple(sorted(benchmark_conditions, key=lambda item: item.identity))
    if not ordered or len({item.identity for item in ordered}) != len(ordered):
        raise ValueError("Scenario-template benchmark conditions must be nonempty and unique")
    payload = {
        "source_reference": source_reference,
        "source_content_identity": _content_identity(source_content),
        "benchmark_conditions": [_condition_payload(item) for item in ordered],
        "generation_constraints": None if generation_constraints is None else generation_constraints.to_dict(),
    }
    return ScenarioTemplateRecord(
        source_reference=source_reference,
        source_content_identity=payload["source_content_identity"],
        benchmark_conditions=ordered,
        generation_constraints=generation_constraints,
        identity=_identity("scenario-template-v1", payload),
    )


def write_scenario_template_record(record: ScenarioTemplateRecord, path: Path) -> Path:
    ScenarioTemplateRecord.from_dict(record.to_dict())
    return _immutable_write(path, record.to_dict())


def load_scenario_template_record(
    path: Path,
    *,
    source_path: Path | None = None,
) -> ScenarioTemplateRecord:
    record = ScenarioTemplateRecord.from_dict(_load_object(path, "scenario-template record"))
    if source_path is not None:
        try:
            content_identity = _content_identity(source_path.read_bytes())
        except OSError as error:
            raise ScenarioLineageError(
                "unresolved_source_provenance",
                f"cannot read scenario-template source {source_path}: {error}",
            ) from error
        if content_identity != record.source_content_identity:
            raise ScenarioLineageError(
                "content_drift",
                "scenario-template source content identity does not match record",
            )
    return record


@dataclass(frozen=True, slots=True)
class CohortV2ScenarioManifest:
    """A source-bound wrapper around an unchanged ``scenario_manifest_v1``."""

    template_record: ScenarioTemplateRecord
    scenario_manifest: ScenarioManifest
    identity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCENARIO_SCHEMA,
            "identity": self.identity,
            "template_record": self.template_record.to_dict(),
            "scenario_manifest": self.scenario_manifest.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CohortV2ScenarioManifest":
        if set(value) != {"schema", "identity", "template_record", "scenario_manifest"}:
            raise ValueError("Cohort-v2 scenario manifest is incomplete or contains unknown fields")
        if value["schema"] != SCENARIO_SCHEMA:
            raise ValueError("Cohort-v2 scenario manifest schema is unsupported")
        if not isinstance(value["template_record"], Mapping) or not isinstance(value["scenario_manifest"], Mapping):
            raise ValueError("Cohort-v2 scenario manifest records must be objects")
        scenario = cls(
            template_record=ScenarioTemplateRecord.from_dict(value["template_record"]),
            scenario_manifest=ScenarioManifest.from_dict(value["scenario_manifest"]),
            identity=value["identity"],
        )
        _validate_binding(scenario)
        expected = _identity("cohort-v2-scenario-manifest-v1", {
            "template_record": scenario.template_record.to_dict(),
            "scenario_manifest": scenario.scenario_manifest.to_dict(),
        })
        if scenario.identity != expected:
            raise ValueError("Cohort-v2 scenario manifest identity is stale")
        return scenario


def _validate_binding(scenario: CohortV2ScenarioManifest, xml_content: bytes | None = None) -> None:
    manifest = scenario.scenario_manifest
    record = scenario.template_record
    if manifest.research_eligibility.status != ELIGIBLE:
        raise ScenarioLineageError(
            "smoke_only",
            "smoke_only scenario source is ineligible for cohort v2",
        )
    if manifest.benchmark_condition.identity not in {item.identity for item in record.benchmark_conditions}:
        raise ValueError("Scenario-template record does not declare the benchmark condition")
    if manifest.generation.mode == "generated":
        if manifest.scenario_template.identity != record.identity:
            raise ValueError("Generated scenario does not bind the declared template record")
        if manifest.generation.declared_inputs.get("template_content_identity") != record.source_content_identity:
            raise ValueError("Generated scenario template content provenance is unresolved")
    elif manifest.generation.mode == "legacy_static":
        if manifest.scenario_template.identity is not None:
            raise ValueError("Legacy scenario must not claim generated template identity")
        if manifest.generation.source_path != record.source_reference:
            raise ValueError("Legacy scenario source provenance does not cite the template record source")
        if manifest.scenario_specification.content_identity != record.source_content_identity:
            raise ValueError("Legacy scenario source content does not match template record")
    else:
        raise ValueError("Scenario generation mode is unsupported")
    if xml_content is not None:
        verify_replay(manifest, xml_content)


def create_cohort_v2_scenario_manifest(
    template_record: ScenarioTemplateRecord,
    scenario_manifest: ScenarioManifest,
    *,
    xml_content: bytes,
) -> CohortV2ScenarioManifest:
    draft = CohortV2ScenarioManifest(template_record, scenario_manifest, identity="")
    _validate_binding(draft, xml_content)
    return CohortV2ScenarioManifest(
        template_record=template_record,
        scenario_manifest=scenario_manifest,
        identity=_identity("cohort-v2-scenario-manifest-v1", {
            "template_record": template_record.to_dict(),
            "scenario_manifest": scenario_manifest.to_dict(),
        }),
    )


def write_cohort_v2_scenario_manifest(manifest: CohortV2ScenarioManifest, path: Path) -> Path:
    CohortV2ScenarioManifest.from_dict(manifest.to_dict())
    return _immutable_write(path, manifest.to_dict())


def load_cohort_v2_scenario_manifest(
    path: Path,
    *,
    xml_path: Path | None = None,
    template_source_path: Path | None = None,
) -> CohortV2ScenarioManifest:
    scenario = CohortV2ScenarioManifest.from_dict(_load_object(path, "cohort-v2 scenario manifest"))
    if template_source_path is not None:
        record = load_scenario_template_record_from_record(scenario.template_record, template_source_path)
        if record != scenario.template_record:
            raise ValueError("Cohort-v2 scenario template record changed during validation")
    if xml_path is not None:
        _validate_binding(scenario, xml_path.read_bytes())
    return scenario


def load_scenario_template_record_from_record(
    record: ScenarioTemplateRecord,
    source_path: Path,
) -> ScenarioTemplateRecord:
    try:
        content_identity = _content_identity(source_path.read_bytes())
    except OSError as error:
        raise ScenarioLineageError(
            "unresolved_source_provenance",
            f"cannot read scenario-template source {source_path}: {error}",
        ) from error
    if content_identity != record.source_content_identity:
        raise ScenarioLineageError(
            "content_drift",
            "scenario-template source content identity does not match record",
        )
    return record


def _validate_materialization_constraints(
    request: Any,
    constraints: ScenarioTemplateConstraints,
    workbook_path: Path | None,
) -> None:
    if workbook_path is None:
        raise ScenarioLineageError(
            "unresolved_source_provenance",
            "materialization is missing the declared constraints workbook",
        )
    validate_scenario_template_constraints_workbook(constraints, workbook_path)
    declared_inputs = (
        request.template_name,
        tuple(request.reference_point),
        tuple(request.min_coordinate),
        tuple(request.max_coordinate),
    )
    recorded_inputs = (
        constraints.canonical_generator_template_name,
        constraints.reference_point,
        constraints.min_coordinate,
        constraints.max_coordinate,
    )
    if declared_inputs != recorded_inputs:
        raise ScenarioLineageError(
            "unresolved_source_provenance",
            "materialization inputs do not match the declared constraints workbook row",
        )


def validate_scenario_template_constraints_workbook(
    constraints: ScenarioTemplateConstraints,
    workbook_path: Path,
) -> ScenarioTemplateConstraints:
    """Validate the exact workbook bytes bound by a template constraint record."""
    try:
        workbook_identity = f"xlsx_bytes_v1:sha256:{sha256(workbook_path.read_bytes()).hexdigest()}"
    except OSError as error:
        raise ScenarioLineageError(
            "unresolved_source_provenance",
            f"cannot read constraints workbook {workbook_path}: {error}",
        ) from error
    if workbook_identity != constraints.source_content_identity:
        raise ScenarioLineageError(
            "content_drift",
            "constraints workbook content identity does not match the template record",
        )
    try:
        cells = _xlsx_row(workbook_path, constraints.sheet_name, constraints.row_number)
        source_name = str(cells["B"])
        parts = source_name.split("_")
        if len(parts) != 6:
            raise ValueError("template name has unexpected form")
        canonical_name = "_".join((parts[1], parts[2], parts[3], "0", parts[5]))
        recorded = (
            canonical_name,
            (float(cells["C"]), float(cells["D"])),
            (float(cells["E"]), float(cells["F"])),
            (float(cells["G"]), float(cells["H"])),
        )
    except (KeyError, OSError, ValueError, ET.ParseError, zipfile.BadZipFile) as error:
        raise ScenarioLineageError(
            "unresolved_source_provenance",
            f"cannot resolve declared constraints workbook row: {error}",
        ) from error
    declared = (
        constraints.canonical_generator_template_name,
        constraints.reference_point,
        constraints.min_coordinate,
        constraints.max_coordinate,
    )
    if recorded != declared:
        raise ScenarioLineageError(
            "unresolved_source_provenance",
            "constraints record does not match its declared workbook row B-H",
        )
    return constraints


_XLSX_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_XLSX_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_XLSX_PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def _xlsx_row(path: Path, sheet_name: str, row_number: int) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationship_id = None
        for sheet in workbook.findall(f".//{{{_XLSX_MAIN}}}sheet"):
            if sheet.attrib.get("name") == sheet_name:
                relationship_id = sheet.attrib.get(f"{{{_XLSX_REL}}}id")
                break
        if relationship_id is None:
            raise ValueError(f"sheet {sheet_name!r} is absent")
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for relationship in relationships.findall(f"{{{_XLSX_PACKAGE_REL}}}Relationship"):
            if relationship.attrib.get("Id") == relationship_id:
                target = relationship.attrib.get("Target")
                break
        if not target:
            raise ValueError("worksheet relationship is unresolved")
        worksheet_path = posixpath.normpath(posixpath.join("xl", target))
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared.findall(f"{{{_XLSX_MAIN}}}si"):
                shared_strings.append("".join(
                    node.text or "" for node in item.iter(f"{{{_XLSX_MAIN}}}t")
                ))
        worksheet = ET.fromstring(archive.read(worksheet_path))
        row = worksheet.find(f".//{{{_XLSX_MAIN}}}row[@r='{row_number}']")
        if row is None:
            raise ValueError(f"row {row_number} is absent")
        values: dict[str, str] = {}
        for cell in row.findall(f"{{{_XLSX_MAIN}}}c"):
            reference = cell.attrib.get("r", "")
            match = re.match(r"^([A-Z]+)[0-9]+$", reference)
            if match is None:
                continue
            raw = cell.find(f"{{{_XLSX_MAIN}}}v")
            if raw is None or raw.text is None:
                continue
            value = raw.text
            if cell.attrib.get("t") == "s":
                value = shared_strings[int(value)]
            values[match.group(1)] = value
        return values


def materialize_template_bound_level_instance(
    request: Any,
    template_record: ScenarioTemplateRecord,
    *,
    constraints_workbook_path: Path | None = None,
    publish: bool = True,
) -> tuple[Any, CohortV2ScenarioManifest]:
    """Materialize through the existing generator after validating its source record."""
    from tasks.task_generator.canonical_materialization import materialize_level_instance

    if not request.template_identity:
        raise ScenarioLineageError(
            "missing_template_identity",
            "materialization request has no scenario-template identity",
        )
    if request.template_identity != template_record.identity:
        raise ScenarioLineageError(
            "unresolved_source_provenance",
            "materialization request does not bind the declared template record",
        )
    if request.benchmark_condition.identity not in {item.identity for item in template_record.benchmark_conditions}:
        raise ScenarioLineageError(
            "unresolved_source_provenance",
            "materialization request benchmark condition is absent from template record",
        )
    load_scenario_template_record_from_record(template_record, request.template_path)
    if template_record.generation_constraints is not None:
        _validate_materialization_constraints(
            request,
            template_record.generation_constraints,
            constraints_workbook_path,
        )
    materialized = materialize_level_instance(request, publish=publish)
    return materialized, create_cohort_v2_scenario_manifest(
        template_record,
        materialized.manifest,
        xml_content=materialized.xml_content,
    )


def create_identical_input_reproduction_receipt(
    original: CohortV2ScenarioManifest,
    reproduced: CohortV2ScenarioManifest,
) -> dict[str, Any]:
    """Record one content- and declared-engine-state-bound deterministic replay."""
    if original.template_record != reproduced.template_record:
        raise ValueError("Identical-input reproduction must use the same template record")
    if original.scenario_manifest.to_dict() != reproduced.scenario_manifest.to_dict():
        raise ValueError("Identical-input reproduction content or provenance drifted")
    payload = {
        "schema": RECEIPT_SCHEMA,
        "kind": "identical_input_reproduction",
        "template_record_identity": original.template_record.identity,
        "original_scenario_manifest_identity": original.identity,
        "reproduced_scenario_manifest_identity": reproduced.identity,
        "scenario_specification_identity": original.scenario_manifest.scenario_specification.identity,
        "content_identity": original.scenario_manifest.scenario_specification.content_identity,
        "declared_initial_engine_state_identity": original.scenario_manifest.declared_initial_engine_state.identity,
    }
    return {**payload, "identity": _identity("deterministic-scenario-receipt-v1", payload)}


def create_unity_reset_reproduction_receipt(
    scenario: CohortV2ScenarioManifest,
    *,
    first_capture_sha256: str,
    second_capture_sha256: str,
    first_initial_engine_state_identity: str,
    second_initial_engine_state_identity: str,
) -> dict[str, Any]:
    """Bind two independent Unity resets to one normalized initial engine state."""
    validated = CohortV2ScenarioManifest.from_dict(scenario.to_dict())
    if (
        _SHA256.fullmatch(first_capture_sha256) is None
        or _SHA256.fullmatch(second_capture_sha256) is None
    ):
        raise ValueError("Unity reset receipt capture digests must be lowercase SHA-256")
    if (
        _NORMALIZED_INITIAL_STATE.fullmatch(first_initial_engine_state_identity) is None
        or _NORMALIZED_INITIAL_STATE.fullmatch(second_initial_engine_state_identity) is None
    ):
        raise ValueError("Unity reset receipt initial engine state identity is invalid")
    if first_initial_engine_state_identity != second_initial_engine_state_identity:
        raise ScenarioLineageError(
            "initial_state_mismatch",
            "independent Unity resets produced different normalized initial engine states",
        )
    payload = {
        "schema": RECEIPT_SCHEMA,
        "kind": "unity_reset_reproduction",
        "template_record_identity": validated.template_record.identity,
        "scenario_manifest_identity": validated.identity,
        "scenario_specification_identity": validated.scenario_manifest.scenario_specification.identity,
        "scenario_content_identity": validated.scenario_manifest.scenario_specification.content_identity,
        "first_capture_sha256": first_capture_sha256,
        "second_capture_sha256": second_capture_sha256,
        "normalized_initial_engine_state_identity": first_initial_engine_state_identity,
    }
    return {**payload, "identity": _identity("deterministic-scenario-receipt-v1", payload)}


def create_changed_declared_input_receipt(
    original: CohortV2ScenarioManifest,
    changed: CohortV2ScenarioManifest,
    *,
    input_key: str,
) -> dict[str, Any]:
    """Record that one declared input creates a new specification and lineage."""
    if original.template_record != changed.template_record:
        raise ScenarioLineageError(
            "content_drift",
            "changed-input comparison must use the same source-bound template record",
        )
    if input_key == "generation_seed":
        original_value = original.scenario_manifest.generation.generation_seed
        changed_value = changed.scenario_manifest.generation.generation_seed
    else:
        original_inputs = original.scenario_manifest.generation.declared_inputs
        changed_inputs = changed.scenario_manifest.generation.declared_inputs
        if not input_key or input_key not in original_inputs or input_key not in changed_inputs:
            raise ValueError("Changed-input receipt requires a declared input present in both scenarios")
        original_value = original_inputs[input_key]
        changed_value = changed_inputs[input_key]
    if original_value == changed_value:
        raise ValueError("Changed-input receipt input value did not change")
    if original.scenario_manifest.scenario_specification.identity == changed.scenario_manifest.scenario_specification.identity:
        raise ScenarioLineageError(
            "cross_lineage_reuse",
            "changed declared input reused the scenario specification identity",
        )
    if original.scenario_manifest.scenario_lineage.identity == changed.scenario_manifest.scenario_lineage.identity:
        raise ScenarioLineageError(
            "cross_lineage_reuse",
            "changed declared input reused the scenario lineage identity",
        )
    payload = {
        "schema": RECEIPT_SCHEMA,
        "kind": "changed_declared_input",
        "template_record_identity": original.template_record.identity,
        "input_key": input_key,
        "original_value": original_value,
        "changed_value": changed_value,
        "original_scenario_specification_identity": original.scenario_manifest.scenario_specification.identity,
        "changed_scenario_specification_identity": changed.scenario_manifest.scenario_specification.identity,
        "original_scenario_lineage_identity": original.scenario_manifest.scenario_lineage.identity,
        "changed_scenario_lineage_identity": changed.scenario_manifest.scenario_lineage.identity,
    }
    return {**payload, "identity": _identity("deterministic-scenario-receipt-v1", payload)}


def validate_deterministic_scenario_receipt(value: Mapping[str, Any]) -> Mapping[str, Any]:
    required_common = {"schema", "kind", "identity", "template_record_identity"}
    if not isinstance(value, Mapping) or not required_common.issubset(value):
        raise ValueError("Deterministic scenario receipt is incomplete")
    if value["schema"] != RECEIPT_SCHEMA:
        raise ValueError("Deterministic scenario receipt schema is unsupported")
    kind = value["kind"]
    if kind == "identical_input_reproduction":
        required = required_common | {
            "original_scenario_manifest_identity", "reproduced_scenario_manifest_identity",
            "scenario_specification_identity", "content_identity", "declared_initial_engine_state_identity",
        }
    elif kind == "changed_declared_input":
        required = required_common | {
            "input_key", "original_value", "changed_value",
            "original_scenario_specification_identity", "changed_scenario_specification_identity",
            "original_scenario_lineage_identity", "changed_scenario_lineage_identity",
        }
    elif kind == "unity_reset_reproduction":
        required = required_common | {
            "scenario_manifest_identity", "scenario_specification_identity",
            "scenario_content_identity", "first_capture_sha256", "second_capture_sha256",
            "normalized_initial_engine_state_identity",
        }
    else:
        raise ValueError("Deterministic scenario receipt kind is unsupported")
    if set(value) != required:
        raise ValueError("Deterministic scenario receipt is incomplete or contains unknown fields")
    if kind == "unity_reset_reproduction" and (
        _SHA256.fullmatch(value["first_capture_sha256"]) is None
        or _SHA256.fullmatch(value["second_capture_sha256"]) is None
        or _NORMALIZED_INITIAL_STATE.fullmatch(
            value["normalized_initial_engine_state_identity"]
        ) is None
    ):
        raise ValueError("Unity reset receipt identities are invalid")
    payload = dict(value)
    identity = payload.pop("identity")
    if identity != _identity("deterministic-scenario-receipt-v1", payload):
        raise ValueError("Deterministic scenario receipt identity is stale")
    return value


def write_deterministic_scenario_receipt(value: Mapping[str, Any], path: Path) -> Path:
    validate_deterministic_scenario_receipt(value)
    return _immutable_write(path, value)


@dataclass(frozen=True, slots=True)
class ScenarioInventoryEntry:
    exposure_role: ExposureRole
    inventory_state: InventoryState
    scenario_manifest_identity: str
    scenario_manifest_digest: str
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
            "exposure_role": self.exposure_role,
            "inventory_state": self.inventory_state,
            "scenario_manifest_identity": self.scenario_manifest_identity,
            "scenario_manifest_digest": self.scenario_manifest_digest,
            "benchmark_condition_identity": self.benchmark_condition_identity,
            "scenario_template_identity": self.scenario_template_identity,
            "level_instance_identity": self.level_instance_identity,
            "scenario_specification_identity": self.scenario_specification_identity,
            "scenario_lineage_identity": self.scenario_lineage_identity,
            "declared_initial_engine_state_identity": self.declared_initial_engine_state_identity,
        }
        if self.exposure_role == "final_evaluation":
            value["sealed_scenario_manifest_reference"] = self.sealed_scenario_manifest_reference
        else:
            value["scenario_manifest_reference"] = self.scenario_manifest_reference
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScenarioInventoryEntry":
        common = {
            "exposure_role",
            "inventory_state",
            "scenario_manifest_identity",
            "scenario_manifest_digest",
            "benchmark_condition_identity",
            "scenario_template_identity",
            "level_instance_identity",
            "scenario_specification_identity",
            "scenario_lineage_identity",
            "declared_initial_engine_state_identity",
        }
        if not isinstance(value, Mapping) or not common.issubset(value):
            raise ValueError("Central-v2 scenario inventory entry is malformed")
        role = value["exposure_role"]
        state = value["inventory_state"]
        if role not in _ROLES or state not in {"planned_non_final", "sealed_final"}:
            raise ValueError("Central-v2 scenario inventory entry has invalid role or state")
        expected_state: InventoryState = "sealed_final" if role == "final_evaluation" else "planned_non_final"
        if state != expected_state:
            raise ValueError("Central-v2 scenario inventory has an invalid role sealing state")
        reference_field = (
            "sealed_scenario_manifest_reference"
            if role == "final_evaluation"
            else "scenario_manifest_reference"
        )
        if set(value) != common | {reference_field}:
            raise ValueError("Central-v2 scenario inventory entry has invalid reference fields")
        string_fields = common - {"exposure_role", "inventory_state", "scenario_manifest_digest"}
        if any(not isinstance(value[field], str) or not value[field] for field in string_fields):
            raise ValueError("Central-v2 scenario inventory entry has invalid public identities")
        if not isinstance(value["scenario_manifest_digest"], str) or _ARTIFACT_DIGEST.fullmatch(value["scenario_manifest_digest"]) is None:
            raise ValueError("Central-v2 scenario inventory entry has invalid manifest digest")
        if not isinstance(value[reference_field], str) or not value[reference_field]:
            raise ValueError("Central-v2 scenario inventory entry has an invalid manifest reference")
        return cls(
            exposure_role=role,
            inventory_state=state,
            scenario_manifest_identity=value["scenario_manifest_identity"],
            scenario_manifest_digest=value["scenario_manifest_digest"],
            benchmark_condition_identity=value["benchmark_condition_identity"],
            scenario_template_identity=value["scenario_template_identity"],
            level_instance_identity=value["level_instance_identity"],
            scenario_specification_identity=value["scenario_specification_identity"],
            scenario_lineage_identity=value["scenario_lineage_identity"],
            declared_initial_engine_state_identity=value["declared_initial_engine_state_identity"],
            scenario_manifest_reference=(
                value[reference_field] if role != "final_evaluation" else None
            ),
            sealed_scenario_manifest_reference=(
                value[reference_field] if role == "final_evaluation" else None
            ),
        )


def _scenario_inventory_projection(
    scenario: CohortV2ScenarioManifest,
) -> dict[str, str]:
    manifest = scenario.scenario_manifest
    return {
        "scenario_manifest_identity": scenario.identity,
        "benchmark_condition_identity": manifest.benchmark_condition.identity,
        "scenario_template_identity": scenario.template_record.identity,
        "level_instance_identity": manifest.level_instance.identity,
        "scenario_specification_identity": manifest.scenario_specification.identity,
        "scenario_lineage_identity": manifest.scenario_lineage.identity,
        "declared_initial_engine_state_identity": manifest.declared_initial_engine_state.identity,
    }


def _scenario_manifest_digest(scenario: CohortV2ScenarioManifest) -> str:
    return f"sha256:{sha256(_artifact_bytes(scenario.to_dict())).hexdigest()}"


def create_scenario_inventory_entry(
    exposure_role: ExposureRole,
    inventory_state: InventoryState,
    scenario: CohortV2ScenarioManifest,
    *,
    scenario_manifest_reference: str | None = None,
    sealed_scenario_manifest_reference: str | None = None,
) -> ScenarioInventoryEntry:
    validated = CohortV2ScenarioManifest.from_dict(scenario.to_dict())
    value = {
        "exposure_role": exposure_role,
        "inventory_state": inventory_state,
        "scenario_manifest_digest": _scenario_manifest_digest(validated),
        **_scenario_inventory_projection(validated),
    }
    if exposure_role == "final_evaluation":
        value["sealed_scenario_manifest_reference"] = sealed_scenario_manifest_reference
    else:
        value["scenario_manifest_reference"] = scenario_manifest_reference
    return ScenarioInventoryEntry.from_dict(value)


def _resolve_nonfinal_inventory_entry(
    entry: ScenarioInventoryEntry,
    manifest_root: Path,
) -> None:
    assert entry.scenario_manifest_reference is not None
    reference = Path(entry.scenario_manifest_reference)
    if reference.is_absolute() or ".." in reference.parts:
        raise ScenarioLineageError(
            "unresolved_source_provenance",
            "non-final scenario manifest reference must remain within its manifest root",
        )
    path = manifest_root / reference
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ScenarioLineageError(
            "unresolved_source_provenance",
            f"cannot resolve non-final scenario manifest {entry.scenario_manifest_reference}: {error}",
        ) from error
    digest = f"sha256:{sha256(raw).hexdigest()}"
    if digest != entry.scenario_manifest_digest:
        raise ScenarioLineageError(
            "content_drift",
            "non-final scenario manifest digest does not match its inventory entry",
        )
    try:
        value = json.loads(raw)
        if not isinstance(value, Mapping):
            raise ValueError("manifest root must be an object")
        if raw != _artifact_bytes(value):
            raise ValueError("manifest is not in canonical artifact form")
        scenario = CohortV2ScenarioManifest.from_dict(value)
    except ScenarioLineageError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ScenarioLineageError(
            "unresolved_source_provenance",
            f"cannot validate non-final scenario manifest {entry.scenario_manifest_reference}: {error}",
        ) from error
    projection = _scenario_inventory_projection(scenario)
    if (
        entry.declared_initial_engine_state_identity
        != projection["declared_initial_engine_state_identity"]
    ):
        raise ScenarioLineageError(
            "initial_state_mismatch",
            "inventory initial-engine-state identity differs from its manifest",
        )
    recorded = entry.to_dict()
    for field, expected in projection.items():
        if field == "declared_initial_engine_state_identity":
            continue
        if recorded[field] != expected:
            raise ScenarioLineageError(
                "content_drift",
                f"inventory {field} differs from its non-final manifest",
            )


def _validate_inventory_entries(
    entries: Sequence[ScenarioInventoryEntry],
    *,
    manifest_root: Path,
) -> tuple[ScenarioInventoryEntry, ...]:
    raw_entries = tuple(entries)
    validated = tuple(ScenarioInventoryEntry.from_dict(entry.to_dict()) for entry in raw_entries)
    if len(validated) != len(_ROLES) or {item.exposure_role for item in validated} != set(_ROLES):
        raise ValueError("Central-v2 scenario inventory must cover every exposure role exactly once")
    ordered = tuple(sorted(validated, key=lambda item: _ROLES.index(item.exposure_role)))
    lineages = [item.scenario_lineage_identity for item in ordered]
    instances = [item.level_instance_identity for item in ordered]
    if len(set(lineages)) != len(lineages) or len(set(instances)) != len(instances):
        raise ScenarioLineageError(
            "cross_lineage_reuse",
            "central-v2 scenario inventory reuses a scenario lineage or level instance",
        )
    nonfinal = [item for item in ordered if item.exposure_role != "final_evaluation"]
    if len({item.scenario_template_identity for item in nonfinal}) < 2:
        raise ValueError("Central-v2 scenario inventory needs two source-bound non-final templates")
    for entry in nonfinal:
        _resolve_nonfinal_inventory_entry(entry, manifest_root)
    return ordered


def _draft_inventory_payload(
    entries: Sequence[ScenarioInventoryEntry],
) -> dict[str, Any]:
    return {
        "schema": INVENTORY_DRAFT_SCHEMA,
        "review_status": "draft",
        "central_v2_scope_claim": build_central_v2_scope_claim("producer"),
        "entries": [entry.to_dict() for entry in entries],
    }


def create_central_v2_scenario_inventory_draft(
    entries: Sequence[ScenarioInventoryEntry],
    *,
    manifest_root: Path,
) -> dict[str, Any]:
    """Create the exact identity graph that must be approved before publication."""
    ordered = _validate_inventory_entries(entries, manifest_root=manifest_root)
    payload = _draft_inventory_payload(ordered)
    return {
        **payload,
        "identity": _identity("central-v2-scenario-inventory-draft-v1", payload),
    }


def _parse_inventory_entries(value: Any) -> tuple[ScenarioInventoryEntry, ...]:
    if not isinstance(value, list):
        raise ValueError("Central-v2 scenario inventory entries must be a list")
    return tuple(ScenarioInventoryEntry.from_dict(item) for item in value)


def validate_central_v2_scenario_inventory_draft(
    value: Mapping[str, Any],
    *,
    manifest_root: Path,
) -> Mapping[str, Any]:
    required = {"schema", "identity", "review_status", "central_v2_scope_claim", "entries"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("Central-v2 scenario inventory draft is incomplete or contains unknown fields")
    if value["schema"] != INVENTORY_DRAFT_SCHEMA or value["review_status"] != "draft":
        raise ValueError("Central-v2 scenario inventory draft metadata is invalid")
    validate_central_v2_scope_claim(value["central_v2_scope_claim"], artifact_kind="producer")
    entries = _parse_inventory_entries(value["entries"])
    ordered = _validate_inventory_entries(entries, manifest_root=manifest_root)
    payload = _draft_inventory_payload(ordered)
    expected = {
        **payload,
        "identity": _identity("central-v2-scenario-inventory-draft-v1", payload),
    }
    if value != expected:
        raise ValueError("Central-v2 scenario inventory draft identity or ordering is stale")
    return value


def create_reviewed_central_v2_scenario_inventory(
    draft: Mapping[str, Any],
    *,
    review_author: str,
    review_url: str,
    manifest_root: Path,
) -> dict[str, Any]:
    validate_central_v2_scenario_inventory_draft(draft, manifest_root=manifest_root)
    if (
        not isinstance(review_author, str)
        or not review_author
        or not isinstance(review_url, str)
        or _ISSUE_45_COMMENT_URL.fullmatch(review_url) is None
    ):
        raise ValueError("Central-v2 scenario inventory requires GitHub review authority")
    payload = {
        "schema": INVENTORY_SCHEMA,
        "review_status": "reviewed",
        "approved_draft_identity": draft["identity"],
        "review_author": review_author,
        "review_url": review_url,
        "central_v2_scope_claim": json.loads(_canonical_json(draft["central_v2_scope_claim"])),
        "entries": json.loads(_canonical_json(draft["entries"])),
    }
    return {
        **payload,
        "identity": _identity("central-v2-scenario-inventory-v1", payload),
    }


def validate_central_v2_scenario_inventory(
    value: Mapping[str, Any],
    *,
    manifest_root: Path,
) -> Mapping[str, Any]:
    required = {
        "schema",
        "identity",
        "review_status",
        "approved_draft_identity",
        "review_author",
        "review_url",
        "central_v2_scope_claim",
        "entries",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("Central-v2 scenario inventory is incomplete or contains unknown fields")
    if (
        value["schema"] != INVENTORY_SCHEMA
        or value["review_status"] != "reviewed"
        or not isinstance(value["review_author"], str)
        or not value["review_author"]
        or not isinstance(value["review_url"], str)
        or _ISSUE_45_COMMENT_URL.fullmatch(value["review_url"]) is None
    ):
        raise ValueError("Central-v2 scenario inventory review metadata is invalid")
    validate_central_v2_scope_claim(value["central_v2_scope_claim"], artifact_kind="producer")
    entries = _parse_inventory_entries(value["entries"])
    ordered = _validate_inventory_entries(entries, manifest_root=manifest_root)
    draft_payload = _draft_inventory_payload(ordered)
    draft_identity = _identity("central-v2-scenario-inventory-draft-v1", draft_payload)
    if value["approved_draft_identity"] != draft_identity:
        raise ValueError("Central-v2 scenario inventory does not bind its approved draft")
    payload = {
        "schema": INVENTORY_SCHEMA,
        "review_status": "reviewed",
        "approved_draft_identity": draft_identity,
        "review_author": value["review_author"],
        "review_url": value["review_url"],
        "central_v2_scope_claim": draft_payload["central_v2_scope_claim"],
        "entries": draft_payload["entries"],
    }
    expected = {
        **payload,
        "identity": _identity("central-v2-scenario-inventory-v1", payload),
    }
    if value != expected:
        raise ValueError("Central-v2 scenario inventory identity or ordering is stale")
    return value


def write_central_v2_scenario_inventory_draft(
    value: Mapping[str, Any],
    path: Path,
    *,
    manifest_root: Path,
) -> Path:
    validate_central_v2_scenario_inventory_draft(value, manifest_root=manifest_root)
    return _immutable_write(path, value)


def write_central_v2_scenario_inventory(
    value: Mapping[str, Any],
    path: Path,
    *,
    manifest_root: Path,
) -> Path:
    validate_central_v2_scenario_inventory(value, manifest_root=manifest_root)
    return _immutable_write(path, value)
