"""Build the frozen non-final Unity probe plan used by issue #44 evidence."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from scripts.cohort_v2_scenarios import (
    CohortV2ScenarioManifest,
    load_cohort_v2_scenario_manifest,
    validate_scenario_template_constraints_workbook,
)
from scripts.collection_plan import create_collection_plan, write_collection_plan
from scripts.scenario_manifest import scenario_manifest_projection


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT / ".claude/project-docs/evidence/issue-45-cohort-v2-lineage"
DEFAULT_OUTPUT = ROOT / ".claude/project-docs/evidence/issue-44-physics-v2/probe-plan.json"
SLINGSHOT = (97, 227)
FRAME_HEIGHT = 480
RELEASE_TIME_MS = 1000
EMPTY_SPACE_TARGET = (8.0, 6.0)
PULL_LENGTH = 80


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
    scenario_geometry_identity: str,
    selection: str,
) -> dict[str, Any]:
    interface, engine = _action(offset)
    if source == "geometry_stratified":
        provenance = {
            "scenario_geometry_identity": scenario_geometry_identity,
            "stratum": selection,
            "feasibility_rule": "opposite-vector-pull-length-80-v1",
        }
    else:
        provenance = {
            "target_stratum": stratum,
            "selection_rule": selection,
        }
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
        by_stratum.setdefault(intervention["intended_coverage_stratum"], []).append(intervention["id"])
    return {
        stratum: (
            {"status": "targeted", "intervention_ids": by_stratum[stratum]}
            if stratum in by_stratum
            else {
                "status": "inapplicable",
                "rationale": "outside the five bounded physics-capture-v2 transport probes",
            }
        )
        for stratum in required
    }


def _scenario(
    *,
    scenario_id: str,
    exposure_role: str,
    manifest_name: str,
    interventions: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest_path = EVIDENCE_ROOT / "manifests" / manifest_name
    xml_path = EVIDENCE_ROOT / "xml" / manifest_name.replace(".json", ".xml")
    unchecked = load_cohort_v2_scenario_manifest(manifest_path)
    template_path = ROOT / unchecked.template_record.source_reference
    wrapper = load_cohort_v2_scenario_manifest(
        manifest_path, xml_path=xml_path, template_source_path=template_path,
    )
    constraints = wrapper.template_record.generation_constraints
    if constraints is None:
        raise ValueError("issue #44 probe scenarios require reviewed generation constraints")
    validate_scenario_template_constraints_workbook(constraints, ROOT / constraints.source_reference)
    manifest = wrapper.scenario_manifest
    negative_ids = [
        intervention["id"]
        for intervention in interventions
        if intervention["intended_coverage_stratum"] == "no-contact/miss"
    ]
    return {
        "scenario_id": scenario_id,
        "exposure_role": exposure_role,
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
                "prospective transport probe only; acceptance requires observed Unity contact evidence"
            ),
        },
        "interventions": interventions,
        "source_dispositions": {
            "geometry_stratified": {"status": "included"},
            "targeted_rare": {"status": "included"},
            "benchmark_agent_replay": {
                "status": "unavailable",
                "rationale": "no benchmark-agent trace is approved for these source-bound instances",
            },
        },
        "coverage_strata": _coverage(interventions),
    }


def _load_probe_source(manifest_name: str) -> tuple[CohortV2ScenarioManifest, ET.Element]:
    manifest_path = EVIDENCE_ROOT / "manifests" / manifest_name
    xml_path = EVIDENCE_ROOT / "xml" / manifest_name.replace(".json", ".xml")
    wrapper = load_cohort_v2_scenario_manifest(
        manifest_path,
        xml_path=xml_path,
        template_source_path=ROOT / load_cohort_v2_scenario_manifest(manifest_path).template_record.source_reference,
    )
    constraints = wrapper.template_record.generation_constraints
    if constraints is None:
        raise ValueError("issue #44 probe scenarios require reviewed generation constraints")
    validate_scenario_template_constraints_workbook(constraints, ROOT / constraints.source_reference)
    return wrapper, ET.fromstring(xml_path.read_bytes())


def _position(element: ET.Element) -> tuple[float, float]:
    return float(element.attrib["x"]), float(element.attrib["y"])


def _slingshot(root: ET.Element) -> tuple[float, float]:
    sling = root.find("Slingshot")
    if sling is None or sling.attrib.get("scenarioObjectId") != "slingshot:0000":
        raise ValueError("probe source is missing its authored slingshot identity")
    return _position(sling)


def _targetable_objects(root: ET.Element) -> list[ET.Element]:
    game_objects = root.find("GameObjects")
    if game_objects is None:
        raise ValueError("probe source has no authored game objects")
    values = [item for item in game_objects if item.tag in {"Pig", "Block", "TNT"}]
    if any(not item.attrib.get("scenarioObjectId") for item in values):
        raise ValueError("probe target is missing scenarioObjectId")
    return values


def _nearest_target(root: ET.Element) -> ET.Element:
    sling_x, sling_y = _slingshot(root)
    return min(
        _targetable_objects(root),
        key=lambda item: (
            (_position(item)[0] - sling_x) ** 2 + (_position(item)[1] - sling_y) ** 2,
            item.attrib["scenarioObjectId"],
        ),
    )


def _release_offset(root: ET.Element, target: tuple[float, float]) -> tuple[int, int]:
    sling_x, sling_y = _slingshot(root)
    target_x, target_y = target
    distance = math.hypot(target_x - sling_x, target_y - sling_y)
    if distance == 0:
        raise ValueError("probe target cannot equal the slingshot center")
    return (
        round(-PULL_LENGTH * (target_x - sling_x) / distance),
        round(-PULL_LENGTH * (target_y - sling_y) / distance),
    )


def _point_text(point: tuple[float, float]) -> str:
    return f"[{point[0]},{point[1]}]"


def build_issue_44_probe_plan(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Write the canonical five-case plan and return its public identity summary."""
    training_wrapper, training_root = _load_probe_source("training.json")
    calibration_wrapper, calibration_root = _load_probe_source("calibration.json")
    training_geometry = training_wrapper.scenario_manifest.scenario_specification.content_identity
    calibration_geometry = calibration_wrapper.scenario_manifest.scenario_specification.content_identity
    training_nearest = _nearest_target(training_root)
    training_nearest_point = _position(training_nearest)
    calibration_by_id = {
        item.attrib["scenarioObjectId"]: item for item in _targetable_objects(calibration_root)
    }
    supported_target = calibration_by_id["block:0000"]
    supported_target_point = _position(supported_target)
    empty_rule = (
        f"empty-space-engine-target-v1:{_point_text(EMPTY_SPACE_TARGET)};"
        "opposite-vector-pull-length-80-v1"
    )
    training = [
        _intervention("no-contact", 1, "no-contact/miss", "targeted_rare",
            _release_offset(training_root, EMPTY_SPACE_TARGET), training_geometry, empty_rule),
        _intervention("collision", 2, "collision", "geometry_stratified",
            _release_offset(training_root, training_nearest_point), training_geometry,
            "nearest-active-target-center:"
            + training_nearest.attrib["scenarioObjectId"] + "@" + _point_text(training_nearest_point)),
        _intervention("support", 3, "persistent support", "targeted_rare",
            _release_offset(training_root, EMPTY_SPACE_TARGET), training_geometry,
            "initial-persistent-support-window;" + empty_rule),
    ]
    calibration = [
        _intervention("support-change", 1, "support change", "geometry_stratified",
            _release_offset(calibration_root, supported_target_point), calibration_geometry,
            "lowest-id-supported-entity:"
            + supported_target.attrib["scenarioObjectId"] + "@" + _point_text(supported_target_point)
            + ";must-resolve-in-pre-intervention-v2-sample"),
        _intervention("stable-terminal", 2, "stability transitions", "targeted_rare",
            _release_offset(calibration_root, EMPTY_SPACE_TARGET), calibration_geometry,
            "no-contact-until-stable-entered;" + empty_rule),
    ]
    plan = create_collection_plan(
        plan_version=1,
        scenarios=[
            _scenario(
                scenario_id="type010101-training-seed4401",
                exposure_role="training",
                manifest_name="training.json",
                interventions=training,
            ),
            _scenario(
                scenario_id="type010102-calibration-seed4501",
                exposure_role="calibration",
                manifest_name="calibration.json",
                interventions=calibration,
            ),
        ],
    )
    write_collection_plan(plan, Path(output))
    cases = sorted(intervention.id for scenario in plan.scenarios for intervention in scenario.interventions)
    return {"plan_identity": plan.identity, "probe_cases": cases, "path": str(Path(output))}


def main() -> None:
    result = build_issue_44_probe_plan()
    print(result["plan_identity"])


if __name__ == "__main__":
    main()
