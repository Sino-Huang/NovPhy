"""Validation for actual Unity exporter capability-report artifacts."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

from scripts.physics_capture_v2_types import PhysicsCaptureV2CapabilityReport


SCHEMA_VERSION = "physics_capture_v2_exporter_capability_report_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROVENANCE = frozenset(("engine_sha256", "player_sha256", "protocol_sha256", "exporter_code_sha256"))
_CASES = frozenset(("no-contact", "collision", "support", "support-change", "stable-terminal"))
_FACTS = frozenset((
    "configured_fixed_step_capture_stride",
    "complete_raw_non_trigger_contacts",
    "collider_geometry_and_separation",
    "gravity_body_lifecycle_motion_support_world",
    "causal_identity_source_bindings",
    "final_frame_covers_termination",
))


class PhysicsCaptureV2CapabilityReportError(ValueError):
    pass


def _fail(location: str, detail: str) -> NoReturn:
    raise PhysicsCaptureV2CapabilityReportError(f"{location}: {detail}")


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(location, "expected object")
    return value


def _sequence(value: object, location: str) -> Sequence[object]:
    if not isinstance(value, list):
        _fail(location, "expected array")
    return value


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(location, "expected nonempty string")
    return value


def _fields(record: Mapping[str, object], expected: frozenset[str], location: str) -> None:
    if set(record) != expected:
        _fail(location, "unexpected report fields")


def validate_physics_capture_v2_capability_report(record: object) -> PhysicsCaptureV2CapabilityReport:
    root = _mapping(record, "report")
    _fields(root, frozenset(("schema_version", "report_id", "provenance", "probes", "facts")), "report")
    if root["schema_version"] != SCHEMA_VERSION:
        _fail("report.schema_version", f"expected {SCHEMA_VERSION}")
    report_id = _string(root["report_id"], "report.report_id")
    provenance = _mapping(root["provenance"], "report.provenance")
    _fields(provenance, _PROVENANCE, "report.provenance")
    parsed_provenance = {key: _string(provenance[key], f"report.provenance.{key}") for key in _PROVENANCE}
    if any(_SHA256.fullmatch(value) is None for value in parsed_provenance.values()):
        _fail("report.provenance", "exact SHA-256 identities are required")

    probe_cases: set[str] = set()
    probe_digests: set[str] = set()
    lineages: set[str] = set()
    levels: set[str] = set()
    templates: set[str] = set()
    for index, value in enumerate(_sequence(root["probes"], "report.probes")):
        probe = _mapping(value, f"report.probes[{index}]")
        _fields(probe, frozenset(("source", "case", "capture_id", "capture_sha256", "scenario_lineage_id", "level_instance_id", "scenario_template_id", "final_evaluation")), f"report.probes[{index}]")
        if probe["source"] != "unity_exporter_probe":
            _fail(f"report.probes[{index}].source", "fixture or command-success evidence is not accepted")
        case = _string(probe["case"], f"report.probes[{index}].case")
        if case not in _CASES:
            _fail(f"report.probes[{index}].case", "unknown required probe case")
        probe_cases.add(case)
        _string(probe["capture_id"], f"report.probes[{index}].capture_id")
        capture_sha256 = _string(probe["capture_sha256"], f"report.probes[{index}].capture_sha256")
        if _SHA256.fullmatch(capture_sha256) is None:
            _fail(f"report.probes[{index}].capture_sha256", "expected SHA-256")
        probe_digests.add(capture_sha256)
        if type(probe["final_evaluation"]) is not bool or probe["final_evaluation"]:
            _fail(f"report.probes[{index}].final_evaluation", "representative probes must be non-final")
        lineages.add(_string(probe["scenario_lineage_id"], f"report.probes[{index}].scenario_lineage_id"))
        levels.add(_string(probe["level_instance_id"], f"report.probes[{index}].level_instance_id"))
        templates.add(_string(probe["scenario_template_id"], f"report.probes[{index}].scenario_template_id"))
    if probe_cases != _CASES:
        _fail("report.probes", "required probe cases are incomplete")
    if min(len(lineages), len(levels), len(templates)) < 2:
        _fail("report.probes", "probes must span two lineages, level instances, and templates")

    facts = _mapping(root["facts"], "report.facts")
    _fields(facts, _FACTS, "report.facts")
    for fact, value in facts.items():
        fact_record = _mapping(value, f"report.facts.{fact}")
        _fields(fact_record, frozenset(("status", "capture_sha256", "reason")), f"report.facts.{fact}")
        status = _string(fact_record["status"], f"report.facts.{fact}.status")
        if status not in {"demonstrated", "unavailable"}:
            _fail(f"report.facts.{fact}.status", "must be demonstrated or unavailable")
        capture_sha = fact_record["capture_sha256"]
        reason = fact_record["reason"]
        if status == "demonstrated":
            if _SHA256.fullmatch(_string(capture_sha, f"report.facts.{fact}.capture_sha256")) is None or reason is not None:
                _fail(f"report.facts.{fact}", "demonstrated facts require a capture digest and no unavailable reason")
            if capture_sha not in probe_digests:
                _fail(f"report.facts.{fact}.capture_sha256", "must cite a validated Unity exporter probe")
        elif capture_sha is not None or not isinstance(reason, str) or not reason:
            _fail(f"report.facts.{fact}", "unavailable facts require an explicit reason only")
    return PhysicsCaptureV2CapabilityReport(report_id, parsed_provenance, root)


def load_physics_capture_v2_capability_report(path: Path) -> PhysicsCaptureV2CapabilityReport:
    try:
        record: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PhysicsCaptureV2CapabilityReportError(f"{path}: malformed JSON") from exc
    return validate_physics_capture_v2_capability_report(record)
