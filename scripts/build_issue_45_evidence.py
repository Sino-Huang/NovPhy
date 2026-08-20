"""Build the immutable public and sealed evidence bundles for GitHub issue #45."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.cohort_v2_scenarios import (
    ScenarioInventoryEntry,
    create_central_v2_scenario_inventory_draft,
    create_changed_declared_input_receipt,
    create_identical_input_reproduction_receipt,
    create_scenario_inventory_entry,
    create_scenario_template_constraints,
    create_scenario_template_record,
    load_cohort_v2_scenario_manifest,
    load_scenario_template_record,
    materialize_template_bound_level_instance,
    validate_central_v2_scenario_inventory_draft,
    validate_deterministic_scenario_receipt,
    write_cohort_v2_scenario_manifest,
    write_deterministic_scenario_receipt,
    write_immutable_cohort_v2_bytes,
    write_immutable_cohort_v2_json,
    write_scenario_template_record,
    write_central_v2_scenario_inventory_draft,
)
from scripts.scenario_manifest import BenchmarkCondition
from tasks.task_generator.canonical_materialization import CanonicalMaterializationRequest


REBUILD_COMMAND = "python -m scripts.build_issue_45_evidence"
APPROVAL_ISSUE_REFERENCE = "https://github.com/Sino-Huang/NovPhy/issues/45"
PUBLIC_BUNDLE_RELATIVE_PATH = Path(
    "data/runtime_evidence/issue-45"
)
SEALED_BUNDLE_RELATIVE_PATH = Path(".local-artifacts/issue-45-cohort-v2-sealed")
CONSTRAINTS_WORKBOOK_REFERENCE = "tasks/task_generator/template_constraints.xlsx"
SEALED_FINAL_REFERENCE = "sealed-final-evaluation-v1:issue-45"


@dataclass(frozen=True, slots=True)
class _Family:
    novelty_type: str
    source_reference: str
    workbook_row: int
    canonical_template_name: str
    reference_point: tuple[float, float]
    min_coordinate: tuple[float, float]
    max_coordinate: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _Role:
    name: str
    seed: int
    family: _Family


FAMILY_A = _Family(
    novelty_type="type010101",
    source_reference=(
        "tasks/task_templates/novelty_level_0/type010101/Levels/"
        "00001_0_1_010101_0_1.xml"
    ),
    workbook_row=3,
    canonical_template_name="0_1_010101_0_1",
    reference_point=(1.00798, -2.1274),
    min_coordinate=(-7.88, -2.39049),
    max_coordinate=(1.229969, 1.809741),
)
FAMILY_B = _Family(
    novelty_type="type010102",
    source_reference=(
        "tasks/task_templates/novelty_level_0/type010102/Levels/"
        "00001_0_1_010102_0_2.xml"
    ),
    workbook_row=4,
    canonical_template_name="0_1_010102_0_2",
    reference_point=(1.02408, -1.84657),
    min_coordinate=(-7.235919, -1.95804),
    max_coordinate=(1.444081, 1.53147),
)
ROLES = (
    _Role("training", 4401, FAMILY_A),
    _Role("calibration", 4501, FAMILY_B),
    _Role("model_selection", 4402, FAMILY_A),
    _Role("final_evaluation", 4502, FAMILY_B),
)
PUBLIC_MANIFEST_REFERENCES = {
    "training": "training.json",
    "calibration": "calibration.json",
    "model_selection": "model-selection.json",
}


def _artifact_record(path: Path, root: Path) -> dict[str, str]:
    content = path.read_bytes()
    identity: str | None = None
    if path.suffix == ".json":
        value = json.loads(content)
        if isinstance(value, Mapping) and isinstance(value.get("identity"), str):
            identity = value["identity"]
    if identity is None:
        namespace = "xml-bytes-v1" if path.suffix == ".xml" else "artifact-bytes-v1"
        identity = f"{namespace}:{path.relative_to(root).as_posix()}"
    return {
        "path": path.relative_to(root).as_posix(),
        "identity": identity,
    }


def _artifact_records(root: Path, excluded: set[str]) -> list[dict[str, str]]:
    return [
        _artifact_record(path, root)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    ]


def _record_for_family(
    repository_root: Path,
    workbook_content: bytes,
    family: _Family,
):
    constraints = create_scenario_template_constraints(
        workbook_content,
        source_reference=CONSTRAINTS_WORKBOOK_REFERENCE,
        sheet_name="Task Variations",
        row_number=family.workbook_row,
        canonical_generator_template_name=family.canonical_template_name,
        reference_point=family.reference_point,
        min_coordinate=family.min_coordinate,
        max_coordinate=family.max_coordinate,
    )
    source_path = repository_root / family.source_reference
    record = create_scenario_template_record(
        source_path.read_bytes(),
        source_reference=family.source_reference,
        benchmark_conditions=[BenchmarkCondition("novelty_level_0", family.novelty_type)],
        generation_constraints=constraints,
    )
    return source_path, constraints, record


def _materialize(
    role: _Role,
    source_path: Path,
    constraints: Any,
    record: Any,
    workbook_path: Path,
    output_root: Path,
):
    request = CanonicalMaterializationRequest(
        template_path=source_path,
        output_xml_path=output_root / f"{role.name}.xml",
        output_manifest_path=output_root / f"{role.name}.scenario.json",
        template_name=constraints.canonical_generator_template_name,
        benchmark_condition=record.benchmark_conditions[0],
        template_identity=record.identity,
        generation_seed=role.seed,
        reference_point=constraints.reference_point,
        min_coordinate=constraints.min_coordinate,
        max_coordinate=constraints.max_coordinate,
        restricted_objects=(),
    )
    return materialize_template_bound_level_instance(
        request,
        record,
        constraints_workbook_path=workbook_path,
        publish=False,
    )


def _bundle_manifest(
    public_root: Path,
    draft: Mapping[str, Any],
    final_entry: ScenarioInventoryEntry,
) -> dict[str, Any]:
    return {
        "schema": "issue_45_cohort_v2_lineage_evidence_bundle_v1",
        "rebuild_command": REBUILD_COMMAND,
        "approval_issue": APPROVAL_ISSUE_REFERENCE,
        "draft_identity": draft["identity"],
        "artifacts": _artifact_records(public_root, {"bundle-manifest.json"}),
        "sealed_final_projection": final_entry.to_dict(),
        "limitations": [
            "The bundle proves deterministic content lineage, not Unity reset-state reproduction.",
            "The sealed final manifest, XML, seed, and realization are outside ordinary workflows.",
        ],
    }


def build_issue_45_evidence(
    *,
    repository_root: Path,
    public_root: Path,
    sealed_root: Path,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    workbook_path = repository_root / CONSTRAINTS_WORKBOOK_REFERENCE
    workbook_content = workbook_path.read_bytes()
    family_sources = {
        family.novelty_type: _record_for_family(repository_root, workbook_content, family)
        for family in (FAMILY_A, FAMILY_B)
    }
    for novelty_type, (_, _, record) in family_sources.items():
        write_scenario_template_record(
            record,
            public_root / "templates" / f"{novelty_type}.scenario-template.json",
        )

    materialized: dict[str, Any] = {}
    scenarios: dict[str, Any] = {}
    for role in ROLES:
        source_path, constraints, record = family_sources[role.family.novelty_type]
        result, scenario = _materialize(
            role,
            source_path,
            constraints,
            record,
            workbook_path,
            public_root,
        )
        materialized[role.name] = result
        scenarios[role.name] = scenario

    training_reproduction, training_reproduction_scenario = _materialize(
        ROLES[0],
        family_sources[FAMILY_A.novelty_type][0],
        family_sources[FAMILY_A.novelty_type][1],
        family_sources[FAMILY_A.novelty_type][2],
        workbook_path,
        public_root,
    )

    for role in ROLES[:3]:
        xml_name = role.name.replace("_", "-") + ".xml"
        write_immutable_cohort_v2_bytes(
            materialized[role.name].xml_content,
            public_root / "xml" / xml_name,
        )
        write_cohort_v2_scenario_manifest(
            scenarios[role.name],
            public_root / "manifests" / PUBLIC_MANIFEST_REFERENCES[role.name],
        )
    write_immutable_cohort_v2_bytes(
        training_reproduction.xml_content,
        public_root / "xml/training-reproduction.xml",
    )
    write_cohort_v2_scenario_manifest(
        training_reproduction_scenario,
        public_root / "manifests/training-reproduction.json",
    )

    identical_receipt = create_identical_input_reproduction_receipt(
        scenarios["training"],
        training_reproduction_scenario,
    )
    changed_receipt = create_changed_declared_input_receipt(
        scenarios["training"],
        scenarios["model_selection"],
        input_key="generation_seed",
    )
    write_deterministic_scenario_receipt(
        identical_receipt,
        public_root / "receipts/training-identical-input.json",
    )
    write_deterministic_scenario_receipt(
        changed_receipt,
        public_root / "receipts/training-model-selection-seed-change.json",
    )

    entries = [
        create_scenario_inventory_entry(
            role.name,
            "planned_non_final",
            scenarios[role.name],
            scenario_manifest_reference=PUBLIC_MANIFEST_REFERENCES[role.name],
        )
        for role in ROLES[:3]
    ]
    final_entry = create_scenario_inventory_entry(
        "final_evaluation",
        "sealed_final",
        scenarios["final_evaluation"],
        sealed_scenario_manifest_reference=SEALED_FINAL_REFERENCE,
    )
    entries.append(final_entry)
    draft = create_central_v2_scenario_inventory_draft(
        entries,
        manifest_root=public_root / "manifests",
    )
    write_central_v2_scenario_inventory_draft(
        draft,
        public_root / "inventory/draft.json",
        manifest_root=public_root / "manifests",
    )
    write_immutable_cohort_v2_json(
        final_entry.to_dict(),
        public_root / "inventory/final-evaluation.sealed-projection.json",
    )

    write_immutable_cohort_v2_bytes(
        materialized["final_evaluation"].xml_content,
        sealed_root / "final-evaluation.xml",
    )
    write_cohort_v2_scenario_manifest(
        scenarios["final_evaluation"],
        sealed_root / "final-evaluation.cohort-v2-scenario.json",
    )
    write_immutable_cohort_v2_json(
        materialized["final_evaluation"].parameter_realization,
        sealed_root / "final-evaluation.parameter-realization.json",
    )
    sealed_manifest = {
        "schema": "issue_45_cohort_v2_sealed_bundle_v1",
        "ordinary_workflow_access": False,
        "public_draft_identity": draft["identity"],
        "artifacts": _artifact_records(sealed_root, {"sealed-bundle-manifest.json"}),
    }
    write_immutable_cohort_v2_json(
        sealed_manifest,
        sealed_root / "sealed-bundle-manifest.json",
    )
    write_immutable_cohort_v2_json(
        _bundle_manifest(public_root, draft, final_entry),
        public_root / "bundle-manifest.json",
    )
    return validate_issue_45_evidence(
        repository_root=repository_root,
        public_root=public_root,
        sealed_root=sealed_root,
    )


def validate_issue_45_evidence(
    *,
    repository_root: Path,
    public_root: Path,
    sealed_root: Path,
) -> dict[str, Any]:
    family_sources = {
        family.novelty_type: repository_root / family.source_reference
        for family in (FAMILY_A, FAMILY_B)
    }
    for novelty_type, source_path in family_sources.items():
        load_scenario_template_record(
            public_root / "templates" / f"{novelty_type}.scenario-template.json",
            source_path=source_path,
        )
    role_families = {
        "training": FAMILY_A,
        "calibration": FAMILY_B,
        "model_selection": FAMILY_A,
    }
    for role, reference in PUBLIC_MANIFEST_REFERENCES.items():
        load_cohort_v2_scenario_manifest(
            public_root / "manifests" / reference,
            xml_path=public_root / "xml" / f"{role.replace('_', '-')}.xml",
            template_source_path=family_sources[role_families[role].novelty_type],
        )
    load_cohort_v2_scenario_manifest(
        public_root / "manifests/training-reproduction.json",
        xml_path=public_root / "xml/training-reproduction.xml",
        template_source_path=family_sources[FAMILY_A.novelty_type],
    )
    for name in (
        "training-identical-input.json",
        "training-model-selection-seed-change.json",
    ):
        validate_deterministic_scenario_receipt(
            json.loads((public_root / "receipts" / name).read_bytes())
        )
    draft = json.loads((public_root / "inventory/draft.json").read_bytes())
    validate_central_v2_scenario_inventory_draft(
        draft,
        manifest_root=public_root / "manifests",
    )
    final_entry = ScenarioInventoryEntry.from_dict(
        json.loads(
            (public_root / "inventory/final-evaluation.sealed-projection.json").read_bytes()
        )
    )
    final_scenario = load_cohort_v2_scenario_manifest(
        sealed_root / "final-evaluation.cohort-v2-scenario.json",
        xml_path=sealed_root / "final-evaluation.xml",
        template_source_path=family_sources[FAMILY_B.novelty_type],
    )
    realization = json.loads(
        (sealed_root / "final-evaluation.parameter-realization.json").read_bytes()
    )
    if realization != final_scenario.scenario_manifest.generation.parameter_realization:
        raise ValueError("Sealed final parameter realization differs from its manifest")
    sealed_manifest = json.loads((sealed_root / "sealed-bundle-manifest.json").read_bytes())
    expected_sealed_manifest = {
        "schema": "issue_45_cohort_v2_sealed_bundle_v1",
        "ordinary_workflow_access": False,
        "public_draft_identity": draft["identity"],
        "artifacts": _artifact_records(sealed_root, {"sealed-bundle-manifest.json"}),
    }
    if sealed_manifest != expected_sealed_manifest:
        raise ValueError("Sealed issue-45 bundle manifest is stale")
    bundle_manifest = json.loads((public_root / "bundle-manifest.json").read_bytes())
    expected_bundle_manifest = _bundle_manifest(public_root, draft, final_entry)
    if bundle_manifest != expected_bundle_manifest:
        raise ValueError("Public issue-45 bundle manifest is stale")
    return {
        "schema": "issue_45_cohort_v2_lineage_build_result_v1",
        "draft_identity": draft["identity"],
        "approval_issue": APPROVAL_ISSUE_REFERENCE,
        "public_bundle_manifest_path": str(public_root / "bundle-manifest.json"),
        "sealed_bundle_manifest_path": str(sealed_root / "sealed-bundle-manifest.json"),
    }


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    result = build_issue_45_evidence(
        repository_root=repository_root,
        public_root=repository_root / PUBLIC_BUNDLE_RELATIVE_PATH,
        sealed_root=repository_root / SEALED_BUNDLE_RELATIVE_PATH,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
