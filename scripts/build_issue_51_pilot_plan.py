"""Build issue #51's prospective supplementary level-clear probe plan."""

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
from scripts.collection_plan import (
    REQUIRED_COVERAGE_STRATA,
    create_collection_plan,
    write_collection_plan,
)
from scripts.scenario_manifest import BenchmarkCondition, scenario_manifest_projection
from tasks.task_generator.canonical_materialization import CanonicalMaterializationRequest


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK_REFERENCE = "tasks/task_generator/template_constraints.xlsx"
TEMPLATE_REFERENCE = "sciencebirdsgames/physics-v2/issue-51/level-clear-template.xml"
DEFAULT_OUTPUT_ROOT = ROOT / ".local-artifacts/issue-51-pilot-authorities"
SCENARIO_ID = "issue51-level-clear-seed5101"
TARGETED_INTERVENTION_ID = "level-clear-targeted"
GEOMETRY_INTERVENTION_ID = "level-clear-geometry"
SLINGSHOT = (97, 227)
FRAME_HEIGHT = 480
LEVEL_CLEAR_OFFSET = (-77, 0)


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


def _materialize(output_root: Path) -> tuple[Path, Path, CohortV2ScenarioManifest]:
    template_path = ROOT / TEMPLATE_REFERENCE
    constraints = _constraints()
    condition = BenchmarkCondition("novelty_level_0", "type010101")
    record = create_scenario_template_record(
        template_path.read_bytes(),
        source_reference=TEMPLATE_REFERENCE,
        benchmark_conditions=[condition],
        generation_constraints=constraints,
    )
    xml_path = output_root / "scenario.xml"
    manifest_path = output_root / "scenario-manifest.json"
    request = CanonicalMaterializationRequest(
        template_path=Path(TEMPLATE_REFERENCE),
        output_xml_path=xml_path,
        output_manifest_path=manifest_path,
        template_name=constraints.canonical_generator_template_name,
        benchmark_condition=condition,
        template_identity=record.identity,
        generation_seed=5101,
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
    offset_x, offset_y = offset
    game_x = SLINGSHOT[0] + offset_x
    game_y = SLINGSHOT[1] - offset_y
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


def build_issue_51_supplementary_plan(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Materialize the one missing supported termination probe."""
    output_root = Path(output_root)
    xml_path, manifest_path, scenario = _materialize(output_root)
    manifest = scenario.scenario_manifest
    targeted_interface, targeted_engine = _action(LEVEL_CLEAR_OFFSET)
    geometry_interface, geometry_engine = _action((-77, -5))
    interventions = [{
        "id": TARGETED_INTERVENTION_ID,
        "ordinal": 1,
        "intended_coverage_stratum": "level clear",
        "source": "targeted_rare",
        "interface_action": targeted_interface,
        "engine_relative_action": targeted_engine,
        "mapping_version": "science-birds-slingshot-relative-v1",
        "slingshot_reference": {"gameX": SLINGSHOT[0], "gameY": SLINGSHOT[1]},
        "source_provenance": {
            "target_stratum": "level clear",
            "selection_rule": (
                "issue-51-frozen-direct-tnt-level-clear-v1; outcome-independent "
                "supplement to the accepted issue-44 through issue-50 evidence"
            ),
        },
    }, {
        "id": GEOMETRY_INTERVENTION_ID,
        "ordinal": 2,
        "intended_coverage_stratum": "collision",
        "source": "geometry_stratified",
        "interface_action": geometry_interface,
        "engine_relative_action": geometry_engine,
        "mapping_version": "science-birds-slingshot-relative-v1",
        "slingshot_reference": {"gameX": SLINGSHOT[0], "gameY": SLINGSHOT[1]},
        "source_provenance": {
            "scenario_geometry_identity": (
                manifest.scenario_specification.content_identity
            ),
            "stratum": "issue-51-direct-tnt-collision",
            "feasibility_rule": "issue-51-frozen-level-clear-geometry-action-v1",
        },
    }]
    coverage = {
        stratum: (
            {
                "status": "targeted",
                "intervention_ids": [TARGETED_INTERVENTION_ID],
            }
            if stratum == "level clear"
            else {
                "status": "targeted",
                "intervention_ids": [GEOMETRY_INTERVENTION_ID],
            }
            if stratum == "collision"
            else {
                "status": "inapplicable",
                "rationale": (
                    "covered by the exact accepted issue-44 through issue-50 component "
                    "plans bound by the representative cohort-v2 pilot plan"
                ),
            }
        )
        for stratum in REQUIRED_COVERAGE_STRATA
    }
    plan = create_collection_plan(
        plan_version=1,
        scenarios=[{
            "scenario_id": SCENARIO_ID,
            "exposure_role": "training",
            **scenario_manifest_projection(
                manifest,
                manifest_path.relative_to(ROOT).as_posix()
                if manifest_path.is_relative_to(ROOT)
                else str(manifest_path),
            ),
            "expected_initial_engine_state_identity": (
                manifest.declared_initial_engine_state.identity
            ),
            "retry_policy": {
                "max_attempts": 1,
                "transient_failure_codes": [],
                "stopping_rule": "execute_all_interventions",
            },
            "negative_specification": {
                "cap": 0,
                "intervention_ids": [],
                "semantic_justification": (
                    "the supplementary terminal probe is not negative-training evidence"
                ),
            },
            "interventions": interventions,
            "source_dispositions": {
                "geometry_stratified": {"status": "included"},
                "targeted_rare": {"status": "included"},
                "benchmark_agent_replay": {
                    "status": "unavailable",
                    "rationale": "optional and not used by the central pilot",
                },
            },
            "coverage_strata": coverage,
        }],
    )
    plan_path = output_root / "collection-plan.json"
    write_collection_plan(plan, plan_path)
    return {
        "plan_identity": plan.identity,
        "plan_path": str(plan_path),
        "scenario_id": SCENARIO_ID,
        "scenario_manifest_identity": scenario.identity,
        "scenario_lineage_id": manifest.scenario_lineage.identity,
        "level_instance_id": manifest.level_instance.identity,
        "scenario_template_id": manifest.scenario_template.identity,
        "expected_initial_engine_state_identity": (
            manifest.declared_initial_engine_state.identity
        ),
        "xml_path": str(xml_path),
        "manifest_path": str(manifest_path),
        "template_path": str(ROOT / TEMPLATE_REFERENCE),
        "workbook_path": str(ROOT / WORKBOOK_REFERENCE),
    }


def main() -> None:
    result = build_issue_51_supplementary_plan()
    print(result["plan_identity"])


if __name__ == "__main__":
    main()
