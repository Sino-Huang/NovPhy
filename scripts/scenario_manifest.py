"""Versioned scenario identity and provenance contract.

This module intentionally uses only the Python standard library so generation,
planning, and replay tools can validate the same manifest without importing the
simulation stack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Literal, Mapping
import xml.etree.ElementTree as ET


SCHEMA = "scenario_manifest_v1"
CANONICAL_XML_SCHEMA = "declared_initial_engine_state_v1"
TemplateAvailability = Literal["available", "unavailable"]
GenerationMode = Literal["generated", "legacy_static"]
ResearchEligibilityStatus = Literal["research_eligible", "smoke_only"]
ELIGIBLE: ResearchEligibilityStatus = "research_eligible"
SMOKE_ONLY: ResearchEligibilityStatus = "smoke_only"
SCENARIO_MANIFEST_PROJECTION_FIELDS = (
    "scenario_manifest_reference",
    "benchmark_condition_identity",
    "scenario_template_identity",
    "level_instance_identity",
    "scenario_specification_identity",
    "scenario_lineage_identity",
    "declared_initial_engine_state_identity",
    "generation_mode",
    "generation_seed",
    "research_eligibility",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _identity(namespace: str, value: Any) -> str:
    return f"{namespace}:sha256:{sha256(_canonical_json(value)).hexdigest()}"


def _content_identity(xml_content: bytes) -> str:
    return f"xml_bytes_v1:sha256:{sha256(xml_content).hexdigest()}"


def _xml_projection(element: ET.Element) -> dict[str, Any]:
    projection: dict[str, Any] = {
        "tag": element.tag,
        "attributes": dict(sorted(element.attrib.items())),
        "children": [_xml_projection(child) for child in element],
    }
    if element.text and element.text.strip():
        projection["text"] = element.text
    return projection


def canonical_xml_projection(xml_content: bytes | str) -> dict[str, Any]:
    """Return the ordered, formatting-independent projection loaded by the engine."""
    raw = xml_content.encode("utf-8") if isinstance(xml_content, str) else xml_content
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        # The staged legacy type2 XML is UTF-8 bytes with a historical utf-16
        # declaration. Preserve its exact bytes for content identity while parsing
        # the declaration the engine has long tolerated.
        normalized = raw.replace(b'encoding="utf-16"', b'encoding="utf-8"', 1)
        normalized = normalized.replace(b"encoding='utf-16'", b"encoding='utf-8'", 1)
        if normalized == raw:
            raise ValueError(f"Malformed scenario XML: {exc}") from exc
        try:
            root = ET.fromstring(normalized)
        except ET.ParseError:
            raise ValueError(f"Malformed scenario XML: {exc}") from exc
    return {"schema": CANONICAL_XML_SCHEMA, "root": _xml_projection(root)}


def declared_initial_engine_state_identity(xml_content: bytes | str) -> str:
    return _identity("declared-initial-engine-state-v1", canonical_xml_projection(xml_content))


@dataclass(frozen=True, slots=True)
class BenchmarkCondition:
    novelty_level: str
    novelty_type: str

    def __post_init__(self) -> None:
        if not self.novelty_level or not self.novelty_type:
            raise ValueError("Benchmark condition fields must be nonempty")

    @property
    def identity(self) -> str:
        return _identity(
            "benchmark-condition-v1",
            {"novelty_level": self.novelty_level, "novelty_type": self.novelty_type},
        )


@dataclass(frozen=True, slots=True)
class TemplateEvidence:
    availability: TemplateAvailability
    identity: str | None


@dataclass(frozen=True, slots=True)
class GenerationProvenance:
    mode: GenerationMode
    generator_identity: str | None
    generator_version: str | None
    generation_seed: int | None
    declared_inputs: dict[str, Any]
    parameter_realization: dict[str, Any]
    importer_identity: str | None = None
    importer_version: str | None = None
    source_path: str | None = None


@dataclass(frozen=True, slots=True)
class LevelInstance:
    identity: str


@dataclass(frozen=True, slots=True)
class ScenarioSpecification:
    identity: str
    declaration_identity: str
    content_identity: str


@dataclass(frozen=True, slots=True)
class ScenarioLineage:
    identity: str


@dataclass(frozen=True, slots=True)
class DeclaredInitialEngineState:
    schema: str
    identity: str


@dataclass(frozen=True, slots=True)
class ResearchEligibility:
    status: ResearchEligibilityStatus
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ScenarioManifest:
    schema: str
    benchmark_condition: BenchmarkCondition
    scenario_template: TemplateEvidence
    generation: GenerationProvenance
    level_instance: LevelInstance
    scenario_specification: ScenarioSpecification
    scenario_lineage: ScenarioLineage
    declared_initial_engine_state: DeclaredInitialEngineState
    research_eligibility: ResearchEligibility

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["benchmark_condition"]["identity"] = self.benchmark_condition.identity
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScenarioManifest":
        required = {
            "schema",
            "benchmark_condition",
            "scenario_template",
            "generation",
            "level_instance",
            "scenario_specification",
            "scenario_lineage",
            "declared_initial_engine_state",
            "research_eligibility",
        }
        if set(data) != required:
            raise ValueError("Scenario manifest is incomplete or contains unknown top-level fields")
        try:
            benchmark_data = dict(data["benchmark_condition"])
            recorded_benchmark_identity = benchmark_data.pop("identity")
            benchmark = BenchmarkCondition(**benchmark_data)
            if recorded_benchmark_identity != benchmark.identity:
                raise ValueError("Scenario manifest benchmark condition identity is stale")
            manifest = cls(
                schema=data["schema"],
                benchmark_condition=benchmark,
                scenario_template=TemplateEvidence(**data["scenario_template"]),
                generation=GenerationProvenance(**data["generation"]),
                level_instance=LevelInstance(**data["level_instance"]),
                scenario_specification=ScenarioSpecification(**data["scenario_specification"]),
                scenario_lineage=ScenarioLineage(**data["scenario_lineage"]),
                declared_initial_engine_state=DeclaredInitialEngineState(**data["declared_initial_engine_state"]),
                research_eligibility=ResearchEligibility(**data["research_eligibility"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("Scenario manifest"):
                raise
            raise ValueError(f"Malformed scenario manifest: {exc}") from exc
        _validate_manifest(manifest)
        return manifest


def scenario_manifest_projection(
    manifest: ScenarioManifest,
    reference: str | None,
) -> dict[str, Any]:
    """Serialize the validated manifest and its planner-facing identity projection."""
    return {
        "scenario_manifest_reference": reference,
        "scenario_manifest": manifest.to_dict(),
        "benchmark_condition_identity": manifest.benchmark_condition.identity,
        "scenario_template_identity": manifest.scenario_template.identity,
        "level_instance_identity": manifest.level_instance.identity,
        "scenario_specification_identity": manifest.scenario_specification.identity,
        "scenario_lineage_identity": manifest.scenario_lineage.identity,
        "declared_initial_engine_state_identity": manifest.declared_initial_engine_state.identity,
        "generation_mode": manifest.generation.mode,
        "generation_seed": manifest.generation.generation_seed,
        "research_eligibility": manifest.research_eligibility.status,
    }


def load_scenario_manifest_projection(
    data: Mapping[str, Any],
    *,
    required: bool,
) -> tuple[ScenarioManifest | None, str | None]:
    """Validate a planner-facing manifest projection without path reconstruction."""
    manifest_data = data.get("scenario_manifest")
    if manifest_data is None:
        if set(SCENARIO_MANIFEST_PROJECTION_FIELDS).intersection(data):
            raise ValueError("Level entry is missing scenario manifest (scenario_manifest) for serialized scenario identity")
        if required:
            raise ValueError("Level entry is missing a scenario manifest required for research admission")
        return None, None
    if not isinstance(manifest_data, Mapping):
        raise ValueError("Level entry scenario_manifest must be an object")
    manifest = ScenarioManifest.from_dict(manifest_data)
    reference = data.get("scenario_manifest_reference")
    if reference is not None and not isinstance(reference, str):
        raise ValueError("Level entry scenario_manifest_reference must be a string or null")
    expected = scenario_manifest_projection(manifest, reference)
    for key in SCENARIO_MANIFEST_PROJECTION_FIELDS:
        if key not in data or data[key] != expected[key]:
            raise ValueError(f"Level entry has stale or missing {key}")
    return manifest, reference


def _declaration_payload(
    benchmark_condition: BenchmarkCondition,
    scenario_template: TemplateEvidence,
    generation: GenerationProvenance,
) -> dict[str, Any]:
    return {
        "benchmark_condition_identity": benchmark_condition.identity,
        "scenario_template": asdict(scenario_template),
        "generation": asdict(generation),
    }


def _derive_scenario_identities(
    benchmark_condition: BenchmarkCondition,
    scenario_template: TemplateEvidence,
    generation: GenerationProvenance,
    content_identity: str,
) -> tuple[str, str, str, str]:
    declaration_identity = _identity(
        "scenario-declaration-v1",
        _declaration_payload(benchmark_condition, scenario_template, generation),
    )
    level_instance_identity = _identity(
        "level-instance-v1",
        {
            "benchmark_condition_identity": benchmark_condition.identity,
            "scenario_template": asdict(scenario_template),
            "declaration_identity": declaration_identity,
        },
    )
    specification_identity = _identity(
        "scenario-specification-v1",
        {
            "level_instance_identity": level_instance_identity,
            "content_identity": content_identity,
        },
    )
    lineage_identity = _identity(
        "scenario-lineage-v1",
        {"scenario_specification_identity": specification_identity},
    )
    return declaration_identity, level_instance_identity, specification_identity, lineage_identity


def _build_manifest(
    xml_content: bytes,
    *,
    benchmark_condition: BenchmarkCondition,
    scenario_template: TemplateEvidence,
    generation: GenerationProvenance,
    eligibility: ResearchEligibilityStatus,
    eligibility_reason: str | None,
) -> ScenarioManifest:
    canonical_xml_projection(xml_content)
    content_identity = _content_identity(xml_content)
    declaration_identity, level_instance_identity, specification_identity, lineage_identity = _derive_scenario_identities(
        benchmark_condition,
        scenario_template,
        generation,
        content_identity,
    )
    return ScenarioManifest(
        schema=SCHEMA,
        benchmark_condition=benchmark_condition,
        scenario_template=scenario_template,
        generation=generation,
        level_instance=LevelInstance(level_instance_identity),
        scenario_specification=ScenarioSpecification(
            specification_identity,
            declaration_identity,
            content_identity,
        ),
        scenario_lineage=ScenarioLineage(lineage_identity),
        declared_initial_engine_state=DeclaredInitialEngineState(
            CANONICAL_XML_SCHEMA,
            declared_initial_engine_state_identity(xml_content),
        ),
        research_eligibility=ResearchEligibility(eligibility, eligibility_reason),
    )


def create_generated_manifest(
    xml_content: bytes,
    *,
    benchmark_condition: BenchmarkCondition,
    template_identity: str,
    generator_identity: str,
    generator_version: str,
    generation_seed: int,
    declared_inputs: Mapping[str, Any],
    parameter_realization: Mapping[str, Any],
    eligibility: ResearchEligibilityStatus = ELIGIBLE,
    eligibility_reason: str | None = None,
) -> ScenarioManifest:
    """Create a manifest for one deterministically generated level instance."""
    if not all((template_identity, generator_identity, generator_version)):
        raise ValueError("Generated provenance fields must be nonempty")
    if isinstance(generation_seed, bool) or not isinstance(generation_seed, int):
        raise ValueError("generation_seed must be an integer")
    generation = GenerationProvenance(
        mode="generated",
        generator_identity=generator_identity,
        generator_version=generator_version,
        generation_seed=generation_seed,
        declared_inputs=json.loads(_canonical_json(dict(declared_inputs))),
        parameter_realization=json.loads(_canonical_json(dict(parameter_realization))),
    )
    return _build_manifest(
        xml_content,
        benchmark_condition=benchmark_condition,
        scenario_template=TemplateEvidence("available", template_identity),
        generation=generation,
        eligibility=eligibility,
        eligibility_reason=eligibility_reason,
    )


def import_legacy_manifest(
    xml_content: bytes,
    *,
    benchmark_condition: BenchmarkCondition,
    source_path: str,
    eligibility: ResearchEligibilityStatus = ELIGIBLE,
    eligibility_reason: str | None = None,
    importer_identity: str = "novphy-legacy-xml-importer",
    importer_version: str = "1",
) -> ScenarioManifest:
    """Represent existing XML without fabricating template or seeded provenance."""
    generation = GenerationProvenance(
        mode="legacy_static",
        generator_identity=None,
        generator_version=None,
        generation_seed=None,
        declared_inputs={},
        parameter_realization={},
        importer_identity=importer_identity,
        importer_version=importer_version,
        source_path=source_path,
    )
    return _build_manifest(
        xml_content,
        benchmark_condition=benchmark_condition,
        scenario_template=TemplateEvidence("unavailable", None),
        generation=generation,
        eligibility=eligibility,
        eligibility_reason=eligibility_reason,
    )


def _validate_manifest(manifest: ScenarioManifest, xml_content: bytes | None = None) -> None:
    if manifest.schema != SCHEMA:
        raise ValueError(f"Unsupported scenario manifest schema: {manifest.schema}")
    if manifest.scenario_template.availability not in {"available", "unavailable"}:
        raise ValueError("Scenario manifest has invalid template availability")
    if (manifest.scenario_template.availability == "available") != (manifest.scenario_template.identity is not None):
        raise ValueError("Scenario manifest has inconsistent template evidence")
    if manifest.generation.mode not in {"generated", "legacy_static"}:
        raise ValueError("Scenario manifest has invalid generation mode")
    if manifest.research_eligibility.status not in {ELIGIBLE, SMOKE_ONLY}:
        raise ValueError("Scenario manifest has invalid research eligibility")
    if manifest.research_eligibility.status == SMOKE_ONLY and not manifest.research_eligibility.reason:
        raise ValueError("smoke_only eligibility requires a reason")
    if manifest.declared_initial_engine_state.schema != CANONICAL_XML_SCHEMA:
        raise ValueError("Scenario manifest has unsupported declared initial engine state schema")

    generation = manifest.generation
    if generation.mode == "generated":
        if manifest.scenario_template.availability != "available":
            raise ValueError("Generated scenario manifest requires available template evidence")
        if not generation.generator_identity or not generation.generator_version or generation.generation_seed is None:
            raise ValueError("Generated scenario manifest has incomplete provenance")
        if generation.importer_identity or generation.importer_version or generation.source_path:
            raise ValueError("Generated scenario manifest contains legacy importer provenance")
    else:
        if manifest.scenario_template.availability != "unavailable":
            raise ValueError("Legacy scenario manifest must use unavailable template evidence")
        if generation.generator_identity is not None or generation.generator_version is not None or generation.generation_seed is not None:
            raise ValueError("Legacy scenario manifest falsely claims generated provenance")
        if not generation.importer_identity or not generation.importer_version or not generation.source_path:
            raise ValueError("Legacy scenario manifest has incomplete importer provenance")

    declaration_identity, level_instance_identity, specification_identity, lineage_identity = _derive_scenario_identities(
        manifest.benchmark_condition,
        manifest.scenario_template,
        generation,
        manifest.scenario_specification.content_identity,
    )
    if manifest.scenario_specification.declaration_identity != declaration_identity:
        raise ValueError("Scenario manifest declaration identity is stale")
    if manifest.level_instance.identity != level_instance_identity:
        raise ValueError("Scenario manifest level instance identity is stale")
    if manifest.scenario_specification.identity != specification_identity:
        raise ValueError("Scenario manifest specification identity is stale")
    if manifest.scenario_lineage.identity != lineage_identity:
        raise ValueError("Scenario manifest lineage identity is stale")

    if xml_content is not None:
        if manifest.scenario_specification.content_identity != _content_identity(xml_content):
            raise ValueError("Scenario XML content identity does not match manifest")
        if manifest.declared_initial_engine_state.identity != declared_initial_engine_state_identity(xml_content):
            raise ValueError("Scenario declared initial engine state identity does not match XML")


def write_manifest(manifest: ScenarioManifest, path: Path) -> Path:
    """Atomically publish a canonical JSON manifest."""
    _validate_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(manifest.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return path


def load_manifest(path: Path, xml_path: Path | None = None) -> ScenarioManifest:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load scenario manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Scenario manifest root must be an object")
    manifest = ScenarioManifest.from_dict(data)
    if xml_path is not None:
        _validate_manifest(manifest, xml_path.read_bytes())
    return manifest


def verify_replay(manifest: ScenarioManifest, xml_content: bytes) -> None:
    """Fail if replayed XML differs from the manifest's exact declared scenario."""
    _validate_manifest(manifest, xml_content)


def require_research_eligible(manifest: ScenarioManifest, use: str = "research cohort") -> None:
    """Reject any scenario whose policy permits bounded smoke use only."""
    _validate_manifest(manifest)
    if manifest.research_eligibility.status != ELIGIBLE:
        reason = manifest.research_eligibility.reason or "no reason recorded"
        raise ValueError(f"Scenario is smoke_only and is not eligible for {use}: {reason}")
