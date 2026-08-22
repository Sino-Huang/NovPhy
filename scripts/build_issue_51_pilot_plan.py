"""Build issue #51's prospective supplementary level-clear probe plan."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.cohort_v2_scenarios import (
    CohortV2ScenarioManifest,
    create_cohort_v2_scenario_manifest,
    create_scenario_template_record,
    write_cohort_v2_scenario_manifest,
    write_immutable_cohort_v2_bytes,
)
from scripts.collection_plan import (
    REQUIRED_COVERAGE_STRATA,
    create_collection_plan,
    write_collection_plan,
)
from scripts.scenario_manifest import (
    BenchmarkCondition,
    import_legacy_manifest,
    scenario_manifest_projection,
)


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK_REFERENCE = "tasks/task_generator/template_constraints.xlsx"
TEMPLATE_REFERENCE = "sciencebirdsgames/physics-v2/issue-51/level-clear-static-v2.xml"
DEFAULT_OUTPUT_ROOT = ROOT / ".local-artifacts/issue-51-pilot-authorities-v3"
SCENARIO_ID = "issue51-level-clear-static-determination3"
TARGETED_INTERVENTION_ID = "level-clear-targeted"
GEOMETRY_INTERVENTION_ID = "level-clear-geometry"
SLINGSHOT = (97, 227)
FRAME_HEIGHT = 480
LEVEL_CLEAR_OFFSET = (-77, 0)


def _import_static(output_root: Path) -> tuple[Path, Path, CohortV2ScenarioManifest]:
    template_path = ROOT / TEMPLATE_REFERENCE
    xml_content = template_path.read_bytes()
    condition = BenchmarkCondition("novelty_level_0", "type010101")
    record = create_scenario_template_record(
        xml_content,
        source_reference=TEMPLATE_REFERENCE,
        benchmark_conditions=[condition],
    )
    xml_path = output_root / "scenario.xml"
    manifest_path = output_root / "scenario-manifest.json"
    manifest = import_legacy_manifest(
        xml_content,
        benchmark_condition=condition,
        source_path=TEMPLATE_REFERENCE,
        importer_version="2",
    )
    scenario = create_cohort_v2_scenario_manifest(
        record,
        manifest,
        xml_content=xml_content,
    )
    write_immutable_cohort_v2_bytes(xml_content, xml_path)
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
    xml_path, manifest_path, scenario = _import_static(output_root)
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
                "issue-51-frozen-direct-tnt-level-clear-v3; outcome-independent "
                "static determination 3 supplement to the accepted issue-44 through "
                "issue-50 evidence"
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
            "feasibility_rule": "issue-51-frozen-static-level-clear-geometry-action-v2",
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
        plan_version=2,
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
