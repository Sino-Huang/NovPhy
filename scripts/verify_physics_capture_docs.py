#!/usr/bin/env python3
# noqa: SIZE_OK - publication and documentation checks must remain in this owned CLI module.
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
FINAL_PUBLICATION_DIRECTORY: Final = Path(".claude/project-docs/evidence/world-model-physics-instrumentation/final-published-runtime")
DONE_CLAIM_SCHEMA: Final = "novphy_final_published_runtime_done_claim_v1"
PUBLICATION_RECEIPT_SCHEMA: Final = "novphy_final_publication_v1"
STAGE_DIRECTORY: Final = Path("sciencebirdsgames/physics-v1")
ARCHIVE_NAME: Final = "novphy-physics-player-2019.4.41f2.tar.gz"
SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
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
    ("engine evidence schema", ("$defs", "physics_violation_engine_evidence_v1", "properties", "schema_version", "const"), "physics_violation_engine_evidence_v1"),
    ("engine evidence trace bound", ("$defs", "physics_violation_engine_evidence_v1", "properties", "terminal_trace", "properties", "max_fixed_steps", "const"), 8),
)
COLLECTION_TOKENS: Final = ("PHYSICS_CAPTURE_V1=1", "PHYSICS_PLAYER_ARCHIVE=sciencebirdsgames/physics-v1/novphy-physics-player-2019.4.41f2.tar.gz", "PHYSICS_SMOKE_MARKER=.claude/project-docs/evidence/world-model-physics-instrumentation/task-8-smoke.json", "OUT_ROOT=data/physics_capture_v1_cohort", "scripts/collect_full_rollout_training_dataset.sh")
PROMOTION_TOKENS: Final = (
    "set -eu",
    "stage=sciencebirdsgames/physics-v1",
    'expected_sha="$(awk \'NF == 2 {print $1}\' "$stage/archive.sha256")"',
    'archive="$stage/novphy-physics-player-2019.4.41f2.tar.gz"',
    'test "$(sha256sum "$archive" | awk \'{print $1}\')" = "$expected_sha"',
    'test "$(awk \'{print $1}\' "$stage/archive.sha256")" = "$expected_sha"',
    'python scripts/verify_physics_player.py --stage "$stage" --expect-sha "$expected_sha"',
    'receipt = Path("sciencebirdsgames/physics-v1/archive.sha256").read_text(encoding="ascii").split()',
    "expected_sha = receipt[0]",
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
@dataclass(frozen=True, slots=True)
class PublicationAuthority:
    archive_sha256: str
    receipt_path: Path
    report_path: Path
    accepted_shot: Path
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
def _required_string(value: JsonObject, field: str, label: str) -> str:
    candidate = value.get(field)
    if not isinstance(candidate, str) or not candidate:
        raise DocumentationError(f"{label} {field} must be a nonempty string")
    return candidate
def _required_sha256(value: JsonObject, field: str, label: str) -> str:
    candidate = _required_string(value, field, label)
    if SHA256_PATTERN.fullmatch(candidate) is None:
        raise DocumentationError(f"{label} {field} must be a lowercase SHA-256")
    return candidate
def _require_true(value: JsonObject, field: str, label: str) -> None:
    if value.get(field) is not True:
        raise DocumentationError(f"{label} {field} must be true")
def _repository_path(repository_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    return (candidate if candidate.is_absolute() else repository_root / candidate).resolve()
def _confined_evidence_path(evidence_root: Path, raw_path: JsonValue, field: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise DocumentationError(f"DoneClaim {field} must be a nonempty path")
    resolved = (Path(raw_path) if Path(raw_path).is_absolute() else evidence_root / raw_path).resolve()
    try:
        resolved.relative_to(evidence_root)
    except ValueError as error:
        raise DocumentationError(f"DoneClaim {field} must stay within final publication evidence") from error
    return resolved
def _archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as archive:
        for chunk in iter(lambda: archive.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
def _publication_authority(repository_root: Path) -> PublicationAuthority:
    evidence_root = (repository_root / FINAL_PUBLICATION_DIRECTORY).resolve()
    claim = _read_json(evidence_root / "done-claim.json")
    if claim.get("schemaVersion") != DONE_CLAIM_SCHEMA:
        raise DocumentationError("DoneClaim schemaVersion is not supported")
    if claim.get("status") != "complete":
        raise DocumentationError("DoneClaim status must be complete")
    source = _json_object(claim.get("source"), "DoneClaim source")
    publication = _json_object(claim.get("publication"), "DoneClaim publication")
    runtime = _json_object(claim.get("runtime"), "DoneClaim runtime")
    for section, label, fields in (
        (source, "DoneClaim source", ("trackedProductClean",)),
        (publication, "DoneClaim publication", ("archiveHashExact", "receiptNonempty", "unityBuildLogNonempty")),
        (runtime, "DoneClaim runtime", ("publishedStage", "request38Compatibility", "request62Compatibility", "request62Decoded", "request70Decoded", "actionPerformed", "accepted", "protectedUnchanged")),
    ):
        for field in fields:
            _require_true(section, field, label)
    stage = (repository_root / STAGE_DIRECTORY).resolve()
    claimed_stage = _repository_path(repository_root, _required_string(publication, "stage", "DoneClaim publication"))
    if claimed_stage != stage:
        raise DocumentationError("DoneClaim publication stage does not match the repository stage")
    archive_sha256 = _required_sha256(publication, "archiveSha256", "DoneClaim")
    if _required_sha256(publication, "targetSha256", "DoneClaim") != archive_sha256:
        raise DocumentationError("DoneClaim archive SHA-256 does not match target SHA-256")
    receipt_path = _confined_evidence_path(evidence_root, publication.get("receipt"), "publication receipt")
    report_path = _confined_evidence_path(evidence_root, runtime.get("report"), "runtime report")
    accepted_shot = _confined_evidence_path(evidence_root, runtime.get("acceptedShot"), "runtime acceptedShot")
    if not accepted_shot.is_dir():
        raise DocumentationError("DoneClaim runtime acceptedShot must be an existing directory")
    return PublicationAuthority(archive_sha256, receipt_path, report_path, accepted_shot)
def _validate_publication_receipt(repository_root: Path, authority: PublicationAuthority) -> None:
    stage = (repository_root / STAGE_DIRECTORY).resolve()
    receipt = _read_json(authority.receipt_path)
    if receipt.get("schemaVersion") != PUBLICATION_RECEIPT_SCHEMA:
        raise DocumentationError("publication receipt schemaVersion is not supported")
    if receipt.get("status") != "published":
        raise DocumentationError("publication receipt status must be published")
    published_stage = _repository_path(repository_root, _required_string(receipt, "stage", "publication receipt"))
    if published_stage != stage:
        raise DocumentationError("publication receipt stage does not match the repository stage")
    archive_record = _json_object(receipt.get("archive"), "publication receipt archive")
    archive = stage / ARCHIVE_NAME
    published_archive = _repository_path(repository_root, _required_string(archive_record, "path", "published archive"))
    if published_archive != archive:
        raise DocumentationError("published archive path must name exactly the repository stage archive")
    if _required_sha256(archive_record, "sha256", "publication receipt archive") != authority.archive_sha256:
        raise DocumentationError("publication receipt archive SHA-256 disagrees with DoneClaim")
    receipt_record = _json_object(receipt.get("receipt"), "publication receipt receipt")
    _require_true(receipt_record, "committedLast", "publication receipt")
    stage_receipt = stage / "archive.sha256"
    published_receipt = _repository_path(repository_root, _required_string(receipt_record, "path", "publication receipt"))
    if published_receipt != stage_receipt:
        raise DocumentationError("publication receipt path must name the staged archive receipt")
    receipt_bytes = stage_receipt.read_bytes()
    if _required_sha256(receipt_record, "sha256", "publication receipt") != hashlib.sha256(receipt_bytes).hexdigest():
        raise DocumentationError("publication receipt staged archive receipt SHA-256 disagrees with its bytes")
    try:
        stage_fields = receipt_bytes.decode("ascii").split()
    except UnicodeDecodeError as error:
        raise DocumentationError("staged archive receipt must be ASCII") from error
    if len(stage_fields) != 2 or stage_fields[1] != ARCHIVE_NAME or SHA256_PATTERN.fullmatch(stage_fields[0]) is None:
        raise DocumentationError("staged archive receipt must name exactly the published archive")
    if stage_fields[0] != authority.archive_sha256:
        raise DocumentationError("staged archive receipt SHA-256 disagrees with DoneClaim")
    if _archive_sha256(archive) != authority.archive_sha256:
        raise DocumentationError("staged archive SHA-256 does not match the final publication")
def _validate_final_smoke(repository_root: Path, authority: PublicationAuthority) -> None:
    evidence_root = (repository_root / FINAL_PUBLICATION_DIRECTORY).resolve()
    report = _read_json(authority.report_path)
    if report.get("status") != "accepted":
        raise DocumentationError("final smoke status must be accepted")
    if report.get("phase") != "complete":
        raise DocumentationError("final smoke phase must be complete")
    _require_true(report, "protected_unchanged", "final smoke")
    accepted_shot = _confined_evidence_path(evidence_root, report.get("accepted_shot"), "final smoke accepted_shot")
    if accepted_shot != authority.accepted_shot:
        raise DocumentationError("final smoke accepted_shot disagrees with DoneClaim")
    provenance = _json_object(report.get("provenance"), "final smoke provenance")
    if _required_sha256(provenance, "archive_sha256", "final smoke") != authority.archive_sha256:
        raise DocumentationError("final smoke archive SHA-256 disagrees with DoneClaim")
def _validate_staged_provenance(repository_root: Path) -> None:
    authority = _publication_authority(repository_root)
    _validate_publication_receipt(repository_root, authority)
    _validate_final_smoke(repository_root, authority)
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
