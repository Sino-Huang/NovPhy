"""Build issue #44's source-bound test levels and frozen five-probe plan."""

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
from scripts.collection_plan import create_collection_plan, write_collection_plan
from scripts.scenario_manifest import BenchmarkCondition, scenario_manifest_projection
from tasks.task_generator.canonical_materialization import CanonicalMaterializationRequest


ROOT = Path(__file__).resolve().parents[1]
STAGE_ROOT = ROOT / "sciencebirdsgames/physics-v2"
DEFAULT_OUTPUT = STAGE_ROOT / "probe-plan.json"
WORKBOOK_REFERENCE = "tasks/task_generator/template_constraints.xlsx"
TRAINING_TEMPLATE_REFERENCE = (
    "tasks/task_templates/novelty_level_0/type010101/Levels/"
    "00001_0_1_010101_0_1.xml"
)
SUPPORT_TEMPLATE_REFERENCE = "sciencebirdsgames/physics-v2/issue-44/support-template.xml"
TRAINING_XML = STAGE_ROOT / "review-levels/training.xml"
SUPPORT_XML = STAGE_ROOT / "review-levels/support-ready.xml"
TRAINING_MANIFEST = STAGE_ROOT / "review-manifests/training.json"
SUPPORT_MANIFEST = STAGE_ROOT / "review-manifests/support-ready.json"
SLINGSHOT = (97, 227)
FRAME_HEIGHT = 480
RELEASE_TIME_MS = 1000
BACKWARD_MISS_OFFSET_640 = (77, 29)
PIG_HIT_OFFSET_640 = (-77, 29)
STABLE_TERMINAL_OFFSET_640 = (-74, -31)


def _constraints(workbook_content: bytes):
    return create_scenario_template_constraints(
        workbook_content,
        source_reference=WORKBOOK_REFERENCE,
        sheet_name="Task Variations",
        row_number=3,
        canonical_generator_template_name="0_1_010101_0_1",
        reference_point=(1.00798, -2.1274),
        min_coordinate=(-7.88, -2.39049),
        max_coordinate=(1.229969, 1.809741),
    )


def _materialize(reference: str, manifest_path: Path) -> tuple[bytes, CohortV2ScenarioManifest]:
    workbook_path = ROOT / WORKBOOK_REFERENCE
    constraints = _constraints(workbook_path.read_bytes())
    validate_scenario_template_constraints_workbook(constraints, workbook_path)
    source_path = Path(reference)
    if not source_path.is_file():
        raise ValueError("issue #44 authorities must be built from the repository root")
    condition = BenchmarkCondition("novelty_level_0", "type010101")
    record = create_scenario_template_record(
        source_path.read_bytes(),
        source_reference=reference,
        benchmark_conditions=[condition],
        generation_constraints=constraints,
    )
    request = CanonicalMaterializationRequest(
        template_path=source_path,
        output_xml_path=manifest_path.with_suffix(".xml"),
        output_manifest_path=manifest_path,
        template_name=constraints.canonical_generator_template_name,
        benchmark_condition=condition,
        template_identity=record.identity,
        generation_seed=4401,
        reference_point=constraints.reference_point,
        min_coordinate=constraints.min_coordinate,
        max_coordinate=constraints.max_coordinate,
        restricted_objects=(),
    )
    materialized, scenario = materialize_template_bound_level_instance(
        request,
        record,
        constraints_workbook_path=workbook_path,
        publish=False,
    )
    return materialized.xml_content, scenario


def build_issue_44_scenario_authorities() -> dict[str, CohortV2ScenarioManifest]:
    """Materialize only the two #44 authorities; the #45 inventory is untouched."""
    training_bytes, training = _materialize(TRAINING_TEMPLATE_REFERENCE, TRAINING_MANIFEST)
    if training_bytes != TRAINING_XML.read_bytes():
        raise ValueError("the existing reviewed training level no longer reproduces from seed 4401")
    support_bytes, support = _materialize(SUPPORT_TEMPLATE_REFERENCE, SUPPORT_MANIFEST)
    write_cohort_v2_scenario_manifest(training, TRAINING_MANIFEST)
    write_immutable_cohort_v2_bytes(support_bytes, SUPPORT_XML)
    write_cohort_v2_scenario_manifest(support, SUPPORT_MANIFEST)
    return {"training": training, "support-ready": support}


def _action(offset: tuple[int, int]) -> tuple[dict[str, Any], dict[str, Any]]:
    game_x = SLINGSHOT[0] + offset[0]
    game_y = SLINGSHOT[1] - offset[1]
    interface = {
        "action_type": "drag_hold_release",
        "coordinate_frame": "slingshot_relative",
        "drag_start": list(SLINGSHOT),
        "drag_release": list(offset),
        "tapTime": 0,
        "releaseTime": RELEASE_TIME_MS,
        "frame_height": FRAME_HEIGHT,
        "socket_command": {
            "x": game_x,
            "y": FRAME_HEIGHT - 1 - game_y,
            "tapTime": 0,
            "releaseTime": RELEASE_TIME_MS,
        },
    }
    engine = {
        "coordinate_frame": "slingshot_relative",
        "release_offset": list(offset),
        "release_point": [game_x, game_y],
        "tap_time_ms": 0,
        "release_time_ms": RELEASE_TIME_MS,
    }
    return interface, engine


def _intervention(
    identifier: str,
    ordinal: int,
    stratum: str,
    source: str,
    offset: tuple[int, int],
    geometry_identity: str,
) -> dict[str, Any]:
    interface, engine = _action(offset)
    provenance = (
        {
            "scenario_geometry_identity": geometry_identity,
            "stratum": "failed-trace-calibrated-pig-hit-direction:640px:[-77,29]",
            "feasibility_rule": "issue-44-pig-hit-ballistic-calibration-v2",
        }
        if source == "geometry_stratified"
        else {
            "target_stratum": stratum,
            "selection_rule": (
                "issue-44-stable-terminal-action-v1:[-74,-31]"
                if identifier == "stable-terminal"
                else "issue-44-backward-miss-action-v2:[77,29]"
            ),
        }
    )
    return {
        "id": identifier,
        "ordinal": ordinal,
        "intended_coverage_stratum": stratum,
        "source": source,
        "interface_action": interface,
        "engine_relative_action": engine,
        "mapping_version": "science-birds-slingshot-relative-v1",
        "slingshot_reference": {"gameX": SLINGSHOT[0], "gameY": SLINGSHOT[1]},
        "source_provenance": provenance,
    }


def _coverage(interventions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    required = (
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
    by_stratum: dict[str, list[str]] = {}
    for intervention in interventions:
        by_stratum.setdefault(intervention["intended_coverage_stratum"], []).append(
            intervention["id"]
        )
    return {
        stratum: (
            {"status": "targeted", "intervention_ids": by_stratum[stratum]}
            if stratum in by_stratum
            else {
                "status": "inapplicable",
                "rationale": "outside the five bounded physics-capture-v2 probes",
            }
        )
        for stratum in required
    }


def _scenario(
    *,
    scenario_id: str,
    scenario: CohortV2ScenarioManifest,
    manifest_path: Path,
    interventions: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = scenario.scenario_manifest
    negative_ids = [
        intervention["id"]
        for intervention in interventions
        if intervention["intended_coverage_stratum"] == "no-contact/miss"
    ]
    return {
        "scenario_id": scenario_id,
        "exposure_role": "training",
        **scenario_manifest_projection(manifest, manifest_path.relative_to(ROOT).as_posix()),
        "expected_initial_engine_state_identity": manifest.declared_initial_engine_state.identity,
        "retry_policy": {
            "max_attempts": 1,
            "transient_failure_codes": [],
            "stopping_rule": "execute_all_interventions",
        },
        "negative_specification": {
            "cap": len(negative_ids),
            "intervention_ids": negative_ids,
            "semantic_justification": (
                "bounded issue-44 exporter evidence; this test level is not a #45 inventory amendment"
            ),
        },
        "interventions": interventions,
        "source_dispositions": {
            "geometry_stratified": {"status": "included"},
            "targeted_rare": {"status": "included"},
            "benchmark_agent_replay": {
                "status": "unavailable",
                "rationale": "no benchmark-agent trace is approved for these source-bound tests",
            },
        },
        "coverage_strata": _coverage(interventions),
    }


def build_issue_44_probe_plan(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Write the canonical five-case plan and return its authority identities."""
    authorities = build_issue_44_scenario_authorities()
    training = authorities["training"]
    support = authorities["support-ready"]
    training_geometry = training.scenario_manifest.scenario_specification.content_identity
    support_geometry = support.scenario_manifest.scenario_specification.content_identity
    training_interventions = [
        _intervention("no-contact", 1, "no-contact/miss", "targeted_rare",
            BACKWARD_MISS_OFFSET_640, training_geometry),
        _intervention("collision", 2, "collision", "geometry_stratified",
            PIG_HIT_OFFSET_640, training_geometry),
        _intervention("stable-terminal", 3, "stability transitions", "targeted_rare",
            STABLE_TERMINAL_OFFSET_640, training_geometry),
    ]
    support_interventions = [
        _intervention("support", 1, "persistent support", "targeted_rare",
            BACKWARD_MISS_OFFSET_640, support_geometry),
        _intervention("support-change", 2, "support change", "geometry_stratified",
            PIG_HIT_OFFSET_640, support_geometry),
    ]
    plan = create_collection_plan(
        plan_version=2,
        scenarios=[
            _scenario(
                scenario_id="type010101-training-seed4401",
                scenario=training,
                manifest_path=TRAINING_MANIFEST,
                interventions=training_interventions,
            ),
            _scenario(
                scenario_id="issue44-support-ready-type010101-seed4401",
                scenario=support,
                manifest_path=SUPPORT_MANIFEST,
                interventions=support_interventions,
            ),
        ],
    )
    write_collection_plan(plan, Path(output))
    cases = sorted(
        intervention.id
        for scenario in plan.scenarios
        for intervention in scenario.interventions
    )
    return {
        "plan_identity": plan.identity,
        "probe_cases": cases,
        "path": str(Path(output)),
        "support_ready_level_identity": support.scenario_manifest.level_instance.identity,
        "support_ready_scenario_manifest_identity": support.identity,
    }


def main() -> None:
    result = build_issue_44_probe_plan()
    print(result["plan_identity"])
    print(result["support_ready_level_identity"])


if __name__ == "__main__":
    main()
