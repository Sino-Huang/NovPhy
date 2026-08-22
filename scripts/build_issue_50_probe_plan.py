"""Build issue #50's prospective two-lineage Unity semantic-probe plan."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.cohort_v2_scenarios import (
    CohortV2ScenarioManifest,
    create_scenario_template_constraints,
    create_scenario_template_record,
    materialize_template_bound_level_instance,
    validate_scenario_template_constraints_workbook,
    write_cohort_v2_scenario_manifest,
    write_immutable_cohort_v2_bytes,
)
from scripts.collection_plan import REQUIRED_COVERAGE_STRATA, create_collection_plan, write_collection_plan
from scripts.scenario_manifest import BenchmarkCondition, scenario_manifest_projection
from tasks.task_generator.canonical_materialization import CanonicalMaterializationRequest


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK_REFERENCE = "tasks/task_generator/template_constraints.xlsx"
TEMPLATES = {
    "floating-a": "sciencebirdsgames/physics-v2/issue-50/floating-a-template.xml",
    "floating-b": "sciencebirdsgames/physics-v2/issue-50/floating-b-template.xml",
}
SEEDS = {"floating-a": 5001, "floating-b": 5002}
DEFAULT_OUTPUT_ROOT = ROOT / ".local-artifacts/issue-50-probe-authorities"
SLINGSHOT = (97, 227)
FRAME_HEIGHT = 480
MISS_OFFSET = (77, 29)


def _constraints():
    workbook = ROOT / WORKBOOK_REFERENCE
    value = create_scenario_template_constraints(
        workbook.read_bytes(),
        source_reference=WORKBOOK_REFERENCE,
        sheet_name="Task Variations",
        row_number=3,
        canonical_generator_template_name="0_1_010101_0_1",
        reference_point=(1.00798, -2.1274),
        min_coordinate=(-7.88, -2.39049),
        max_coordinate=(1.229969, 1.809741),
    )
    validate_scenario_template_constraints_workbook(value, workbook)
    return value


def _materialize(case: str, output_root: Path) -> tuple[Path, Path, CohortV2ScenarioManifest]:
    reference = TEMPLATES[case]
    template_path = Path(reference)
    template_source = ROOT / template_path
    constraints = _constraints()
    condition = BenchmarkCondition("novelty_level_0", "type010101")
    record = create_scenario_template_record(
        template_source.read_bytes(),
        source_reference=reference,
        benchmark_conditions=[condition],
        generation_constraints=constraints,
    )
    xml_path = output_root / "xml" / f"{case}.xml"
    manifest_path = output_root / "manifests" / f"{case}.json"
    request = CanonicalMaterializationRequest(
        template_path=template_path,
        output_xml_path=xml_path,
        output_manifest_path=manifest_path,
        template_name=constraints.canonical_generator_template_name,
        benchmark_condition=condition,
        template_identity=record.identity,
        generation_seed=SEEDS[case],
        reference_point=constraints.reference_point,
        min_coordinate=constraints.min_coordinate,
        max_coordinate=constraints.max_coordinate,
        restricted_objects=(),
    )
    materialized, scenario = materialize_template_bound_level_instance(
        request,
        record,
        constraints_workbook_path=ROOT / WORKBOOK_REFERENCE,
        publish=False,
    )
    write_immutable_cohort_v2_bytes(materialized.xml_content, xml_path)
    write_cohort_v2_scenario_manifest(scenario, manifest_path)
    return xml_path, manifest_path, scenario


def _action(offset: tuple[int, int]) -> tuple[dict[str, Any], dict[str, Any]]:
    game_x = SLINGSHOT[0] + offset[0]
    game_y = SLINGSHOT[1] - offset[1]
    interface = {
        "action_type": "drag_hold_release",
        "coordinate_frame": "slingshot_relative",
        "drag_start": list(SLINGSHOT),
        "drag_release": list(offset),
        "tapTime": 0,
        "releaseTime": 1000,
        "frame_height": FRAME_HEIGHT,
        "socket_command": {
            "x": game_x,
            "y": FRAME_HEIGHT - 1 - game_y,
            "tapTime": 0,
            "releaseTime": 1000,
        },
    }
    engine = {
        "coordinate_frame": "slingshot_relative",
        "release_offset": list(offset),
        "release_point": [game_x, game_y],
        "tap_time_ms": 0,
        "release_time_ms": 1000,
    }
    return interface, engine


def _coverage(intervention_ids: list[str]) -> dict[str, dict[str, Any]]:
    return {
        stratum: (
            {"status": "targeted", "intervention_ids": intervention_ids}
            if stratum == "stability transitions"
            else {
                "status": "inapplicable",
                "rationale": "outside the bounded issue-50 semantic capability probe",
            }
        )
        for stratum in REQUIRED_COVERAGE_STRATA
    }


def _scenario(
    case: str,
    scenario: CohortV2ScenarioManifest,
) -> dict[str, Any]:
    manifest = scenario.scenario_manifest
    targeted_interface, targeted_engine = _action(MISS_OFFSET)
    geometry_interface, geometry_engine = _action((-77, 29))
    targeted_id = f"{case}-stationary"
    geometry_id = f"{case}-geometry"
    interventions = [
        {
            "id": targeted_id,
            "ordinal": 1,
            "intended_coverage_stratum": "stability transitions",
            "source": "targeted_rare",
            "interface_action": targeted_interface,
            "engine_relative_action": targeted_engine,
            "mapping_version": "science-birds-slingshot-relative-v1",
            "slingshot_reference": {"gameX": SLINGSHOT[0], "gameY": SLINGSHOT[1]},
            "source_provenance": {
                "target_stratum": "stability transitions",
                "selection_rule": (
                    "issue-50-frozen-unsupported-stationary-probe-v1; semantic witnesses "
                    "are retained outcome-independently and are not SPSG negatives"
                ),
            },
        },
        {
            "id": geometry_id,
            "ordinal": 2,
            "intended_coverage_stratum": "stability transitions",
            "source": "geometry_stratified",
            "interface_action": geometry_interface,
            "engine_relative_action": geometry_engine,
            "mapping_version": "science-birds-slingshot-relative-v1",
            "slingshot_reference": {"gameX": SLINGSHOT[0], "gameY": SLINGSHOT[1]},
            "source_provenance": {
                "scenario_geometry_identity": manifest.scenario_specification.content_identity,
                "stratum": "issue-50-penetration-and-motion-boundary",
                "feasibility_rule": "issue-50-frozen-geometry-boundary-action-v1",
            },
        },
    ]
    return {
        "scenario_id": f"issue50-{case}-seed{SEEDS[case]}",
        "exposure_role": "calibration",
        **scenario_manifest_projection(
            manifest,
            f"data/runtime_evidence/issue-50/source-probes/manifests/{case}.json",
        ),
        "expected_initial_engine_state_identity": manifest.declared_initial_engine_state.identity,
        "retry_policy": {
            "max_attempts": 1,
            "transient_failure_codes": [],
            "stopping_rule": "execute_all_interventions",
        },
        "negative_specification": {
            "cap": 0,
            "intervention_ids": [],
            "semantic_justification": (
                "issue-50 semantic witnesses are not SPSG negative-training examples"
            ),
        },
        "interventions": interventions,
        "source_dispositions": {
            "geometry_stratified": {"status": "included"},
            "targeted_rare": {"status": "included"},
            "benchmark_agent_replay": {
                "status": "unavailable",
                "rationale": "no benchmark-agent trace is approved for this probe",
            },
        },
        "coverage_strata": _coverage([targeted_id, geometry_id]),
    }


def build_issue_50_probe_plan(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Materialize two source-bound lineages and freeze their two probe attempts."""
    output_root = Path(output_root)
    authorities = {
        case: _materialize(case, output_root) for case in sorted(TEMPLATES)
    }
    plan_path = output_root / "probe-plan.json"
    plan = create_collection_plan(
        plan_version=1,
        scenarios=[
            _scenario(case, scenario)
            for case, (_, _, scenario) in authorities.items()
        ],
    )
    write_collection_plan(plan, plan_path)
    return {
        "plan_identity": plan.identity,
        "plan_path": str(plan_path),
        "scenarios": {
            case: {
                "scenario_id": f"issue50-{case}-seed{SEEDS[case]}",
                "xml_path": str(xml_path),
                "manifest_path": str(manifest_path),
                "template_path": str(ROOT / TEMPLATES[case]),
                "scenario_lineage_id": scenario.scenario_manifest.scenario_lineage.identity,
                "level_instance_id": scenario.scenario_manifest.level_instance.identity,
                "scenario_template_id": scenario.template_record.identity,
            }
            for case, (xml_path, manifest_path, scenario) in authorities.items()
        },
    }


def main() -> None:
    result = build_issue_50_probe_plan()
    print(result["plan_identity"])


if __name__ == "__main__":
    main()
