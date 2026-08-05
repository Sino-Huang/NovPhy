#!/usr/bin/env python3
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Final, TypeAlias, assert_never
JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
CONTRACT_DOCUMENT: Final = Path("data_contracts/physics_capture_v1.md")
SCHEMA_DOCUMENT: Final = Path("data_contracts/physics_capture_v1.schema.json")
SMOKE_REPORT: Final = Path(".omo/evidence/world-model-physics-instrumentation/task-8-smoke.json")
STAGE_DIRECTORY: Final = Path("sciencebirdsgames/physics-v1")
ARCHIVE_NAME: Final = "novphy-physics-player-2019.4.41f2.tar.gz"
ARCHIVE_SHA256: Final = "c7f9fa4c98480c1c1c8e580cb00454beda4fed4bf28a4822d31c561997906992"
EXAMPLE_PATTERN: Final = re.compile(r"```json physics_capture_v1_example\n(?P<record>.*?)\n```", re.DOTALL)
COMMAND_BLOCK_PATTERN: Final = re.compile(r"```bash physics_capture_v1_(?P<name>collection|promotion|rollback)\n(?P<commands>.*?)\n```", re.DOTALL)
EVENT_TAXONOMY: Final = ("bird_launched", "collision", "explosion", "entity_destroyed", "pig_removed", "bird_exhausted", "stable_entered", "stable_exited", "level_cleared", "level_failed")
RECORD_CLOCK_FIELDS: Final = ("schema_version", "capture_id", "shot_id", "sequence", "render_frame", "render_time", "fixed_step", "fixed_time", "coordinates")
FAILURE_CODES: Final = ("record_limit_exceeded", "byte_limit_exceeded", "capture_timeout", "truncated_finalization")
SCHEMA_REQUIREMENTS: Final = (
    ("coordinate schema", ("$defs", "coordinates", "properties", "world_space", "const"), "unity_world_2d"),
    ("coordinate schema", ("$defs", "coordinates", "properties", "screen_space", "const"), "rgb_pixel_2d"),
    ("support rule", ("$defs", "support_rule", "properties", "name", "const"), "support_v1"),
    ("support rule", ("$defs", "support_rule", "properties", "minimum_consecutive_fixed_steps", "const"), 2),
    ("support rule", ("$defs", "support_rule", "properties", "minimum_abs_normal_y", "const"), 0.5),
    ("support rule", ("$defs", "support_rule", "properties", "minimum_vertical_center_delta", "const"), 0.0001),
)
COLLECTION_TOKENS: Final = ("PHYSICS_CAPTURE_V1=1", "PHYSICS_PLAYER_ARCHIVE=sciencebirdsgames/physics-v1/novphy-physics-player-2019.4.41f2.tar.gz", "PHYSICS_SMOKE_MARKER=.omo/evidence/world-model-physics-instrumentation/task-8-smoke.json", "OUT_ROOT=data/physics_capture_v1_cohort", "scripts/collect_full_rollout_training_dataset.sh")
PROMOTION_TOKENS: Final = (
    "set -eu",
    "stage=sciencebirdsgames/physics-v1",
    f"expected_sha={ARCHIVE_SHA256}",
    'archive="$stage/novphy-physics-player-2019.4.41f2.tar.gz"',
    'test "$(sha256sum "$archive" | awk \'{print $1}\')" = "$expected_sha"',
    'test "$(awk \'{print $1}\' "$stage/archive.sha256")" = "$expected_sha"',
    'python scripts/verify_physics_player.py --stage "$stage" --expect-sha "$expected_sha"',
    f'expected_sha = "{ARCHIVE_SHA256}"',
    'test -L "$selector/current"',
    'ln -s "$(readlink "$selector/current")" "$selector/previous.next"',
    'mv -Tf "$selector/previous.next" "$selector/previous"',
    'ln -s "../physics-v1" "$selector/next"',
    'mv -Tf "$selector/next" "$selector/current"',
    'test "$(readlink "$selector/current")" = "../physics-v1"',
)
ROLLBACK_TOKENS: Final = (
    "set -eu",
    "selector=sciencebirdsgames/physics-selection",
    'test -L "$selector/previous"',
    'target="$(readlink "$selector/previous")"',
    'ln -s "$target" "$selector/rollback"',
    'mv -Tf "$selector/rollback" "$selector/current"',
    'test "$(readlink "$selector/current")" = "$target"',
)
@dataclass(frozen=True, slots=True)
class DocumentationError(RuntimeError):
    reason: str
    def __str__(self) -> str:
        return self.reason
def _json_object(value: JsonValue, field: str) -> JsonObject:
    if not isinstance(value, dict):
        raise DocumentationError(f"schema {field} must be an object")
    return value
def _string_list(value: JsonValue, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise DocumentationError(f"schema {field} must be a string array")
    return tuple(value)
def _read_json(path: Path) -> JsonObject:
    try:
        value: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DocumentationError(f"invalid JSON at {path}: {error.msg}") from error
    return _json_object(value, str(path))
def _property_definition(properties: JsonObject, field: str) -> JsonObject:
    definition = properties.get(field)
    if definition is None:
        raise DocumentationError(f"schema capture_failure is missing {field}")
    return _json_object(definition, f"capture_failure.{field}")
def _matches_json_type(value: JsonValue, expected: str) -> bool:
    match expected:
        case "string":
            return isinstance(value, str)
        case "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        case "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        case "boolean":
            return isinstance(value, bool)
        case "null":
            return value is None
        case "object":
            return isinstance(value, dict)
        case "array":
            return isinstance(value, list)
        case unreachable:
            assert_never(unreachable)
def _validate_example_field(field: str, value: JsonValue, definition: JsonObject) -> None:
    expected_type = definition.get("type")
    if expected_type is not None:
        if not isinstance(expected_type, str) or not _matches_json_type(value, expected_type):
            raise DocumentationError(f"schema example field {field} has the wrong type")
    expected_const = definition.get("const")
    if expected_const is not None and value != expected_const:
        raise DocumentationError(f"schema example field {field} does not match const")
    expected_enum = definition.get("enum")
    if expected_enum is not None:
        if not isinstance(expected_enum, list) or value not in expected_enum:
            raise DocumentationError(f"schema example field {field} is not an allowed enum value")
    minimum = definition.get("minimum")
    if minimum is not None:
        if not isinstance(minimum, (int, float)) or not isinstance(value, (int, float)) or isinstance(value, bool) or value < minimum:
            raise DocumentationError(f"schema example field {field} is below minimum")
    min_length = definition.get("minLength")
    if min_length is not None:
        if not isinstance(min_length, int) or not isinstance(value, str) or len(value) < min_length:
            raise DocumentationError(f"schema example field {field} is too short")
def _validate_failure_example(record: JsonObject, schema: JsonObject) -> None:
    definitions = _json_object(schema.get("$defs"), "$defs")
    failure = _json_object(definitions.get("capture_failure"), "capture_failure")
    required = _string_list(failure.get("required"), "capture_failure.required")
    properties = _json_object(failure.get("properties"), "capture_failure.properties")
    missing = tuple(field for field in required if field not in record)
    if missing:
        raise DocumentationError(f"schema example is missing required fields: {', '.join(missing)}")
    if failure.get("additionalProperties") is False:
        extra = tuple(field for field in record if field not in properties)
        if extra:
            raise DocumentationError(f"schema example has undeclared fields: {', '.join(extra)}")
    for field, value in record.items():
        _validate_example_field(field, value, _property_definition(properties, field))
def _example_record(document: str) -> JsonObject:
    matches = EXAMPLE_PATTERN.findall(document)
    if len(matches) != 1:
        raise DocumentationError("expected exactly one physics_capture_v1_example JSON block")
    try:
        value: JsonValue = json.loads(matches[0])
    except json.JSONDecodeError as error:
        raise DocumentationError(f"invalid schema example JSON: {error.msg}") from error
    return _json_object(value, "physics_capture_v1_example")
def _schema_at(schema: JsonValue, path: tuple[str | int, ...]) -> JsonValue:
    current = schema
    for segment in path:
        match segment:
            case str():
                candidate = _json_object(current, "contract schema").get(segment)
                if candidate is None:
                    raise DocumentationError(f"contract schema is missing {segment}")
                current = candidate
            case int() if isinstance(current, list) and 0 <= segment < len(current):
                current = current[segment]
            case int():
                raise DocumentationError(f"contract schema has invalid path segment {segment}")
            case unreachable:
                assert_never(unreachable)
    return current
def _validate_schema_contract(schema: JsonObject) -> None:
    for label, path, expected in SCHEMA_REQUIREMENTS:
        if _schema_at(schema, path) != expected:
            raise DocumentationError(f"{label} does not match the frozen contract")
    clock_fields = _string_list(_schema_at(schema, ("$defs", "record_clock", "required")), "record_clock.required")
    if tuple(clock_fields) != RECORD_CLOCK_FIELDS:
        raise DocumentationError("record clock fields do not match the frozen contract")
    taxonomy = tuple(_schema_at(schema, ("$defs", "state_header", "allOf", 1, "properties", "event_taxonomy", "prefixItems", index, "const")) for index in range(len(EVENT_TAXONOMY)))
    if taxonomy != EVENT_TAXONOMY:
        raise DocumentationError("event taxonomy does not match the frozen contract")
    failure_codes = _string_list(_schema_at(schema, ("$defs", "capture_failure", "properties", "failure_code", "enum")), "failure_code.enum")
    if failure_codes != FAILURE_CODES:
        raise DocumentationError("failure codes do not match the frozen contract")
def _required_string(value: JsonObject, field: str) -> str:
    candidate = value.get(field)
    if not isinstance(candidate, str) or not candidate:
        raise DocumentationError(f"smoke report {field} must be a nonempty string")
    return candidate
def _archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as archive:
        for chunk in iter(lambda: archive.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
def _validate_staged_provenance(repository_root: Path) -> None:
    stage = repository_root / STAGE_DIRECTORY
    receipt = (stage / "archive.sha256").read_text(encoding="ascii").split()
    if len(receipt) != 2 or receipt[1] != ARCHIVE_NAME:
        raise DocumentationError("staged archive receipt must name exactly the published archive")
    if receipt[0] != ARCHIVE_SHA256:
        raise DocumentationError("staged archive receipt SHA-256 does not match the documented stage")
    archive = stage / ARCHIVE_NAME
    if _archive_sha256(archive) != receipt[0]:
        raise DocumentationError("staged archive SHA-256 does not match its receipt")
    report = _read_json(repository_root / SMOKE_REPORT)
    if report.get("status") != "accepted":
        raise DocumentationError("smoke report must have status=accepted")
    if report.get("protected_unchanged") is not True:
        raise DocumentationError("smoke report must confirm protected roots are unchanged")
    _required_string(report, "accepted_shot")
    provenance = _json_object(report.get("provenance"), "smoke report provenance")
    if provenance.get("archive_sha256") != ARCHIVE_SHA256:
        raise DocumentationError("smoke report archive SHA-256 does not match the documented stage")
def _command_block(document: str, name: str) -> str:
    matches = tuple(
        match.group("commands")
        for match in COMMAND_BLOCK_PATTERN.finditer(document)
        if match.group("name") == name
    )
    if len(matches) != 1:
        raise DocumentationError(f"expected exactly one {name} command block")
    return matches[0]
def _validate_command_block(name: str, commands: str, required_tokens: tuple[str, ...]) -> None:
    missing = tuple(token for token in required_tokens if token not in commands)
    if missing:
        raise DocumentationError(f"{name} command is missing required operation: {missing[0]}")
    if "sciencebirdsgames/Linux" in commands:
        raise DocumentationError(f"{name} command must not modify the production player path")
    syntax = subprocess.run(("bash", "-n"), input=commands, text=True, capture_output=True, check=False)
    if syntax.returncode:
        raise DocumentationError(f"{name} command has invalid shell syntax")
def _validate_operations(document: str) -> None:
    _validate_command_block("collection", _command_block(document, "collection"), COLLECTION_TOKENS)
    _validate_command_block("promotion", _command_block(document, "promotion"), PROMOTION_TOKENS)
    _validate_command_block("rollback", _command_block(document, "rollback"), ROLLBACK_TOKENS)
def verify_docs(docs_root: Path, repository_root: Path = REPOSITORY_ROOT) -> None:
    contract_path = docs_root / CONTRACT_DOCUMENT
    schema_path = docs_root / SCHEMA_DOCUMENT
    contract = contract_path.read_text(encoding="utf-8")
    schema = _read_json(schema_path)
    _validate_schema_contract(schema)
    _validate_failure_example(_example_record(contract), schema)
    _validate_staged_provenance(repository_root)
    _validate_operations(contract)
def main() -> int:
    parser = argparse.ArgumentParser(description="Validate published physics capture documentation")
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("docs_root", type=Path)
    args = parser.parse_args()
    try:
        verify_docs(args.docs_root, args.repository_root)
    except (DocumentationError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print("physics capture documentation verified")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
