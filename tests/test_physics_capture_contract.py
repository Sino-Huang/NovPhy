from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import TypeAlias
import unittest

from scripts.physics_capture_contract import (
    ContractErrorCode,
    PhysicsContractError,
    load_physics_capture,
)
from scripts.physics_capture_types import EventType


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/physics_capture_v1"
SCHEMA_PATH = ROOT / "docs/data_contracts/physics_capture_v1.schema.json"
JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def _read_jsonl(name: str) -> list[JsonObject]:
    return [json.loads(line) for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, records: list[JsonObject]) -> None:
    path.write_text("".join(f"{json.dumps(record, separators=(',', ':'))}\n" for record in records), encoding="utf-8")


@dataclass(frozen=True, slots=True)
class InvalidFixture:
    state_records: list[JsonObject]
    event_records: list[JsonObject]
    code: ContractErrorCode


class PhysicsCaptureContractTests(unittest.TestCase):
    def assert_fixture_error(self, fixture: InvalidFixture) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "physics_state.jsonl"
            event_path = Path(temporary) / "physics_events.jsonl"
            _write_jsonl(state_path, fixture.state_records)
            _write_jsonl(event_path, fixture.event_records)

            with self.assertRaises(PhysicsContractError) as raised:
                load_physics_capture(state_path, event_path)

        self.assertEqual(raised.exception.code, fixture.code)

    def test_valid_golden_sidecars_parse_as_separate_immutable_records(self):
        # Given: golden state and event JSONL sidecars.
        state_path = FIXTURES / "physics_state.jsonl"
        event_path = FIXTURES / "physics_events.jsonl"

        # When: the public contract boundary loads both files.
        capture = load_physics_capture(state_path, event_path)

        # Then: state, contacts, support, and events remain separate typed tuples.
        self.assertEqual(capture.header.clock.schema_version, "physics_capture_v1")
        self.assertEqual(len(capture.states), 2)
        self.assertEqual(len(capture.states[1].support_edges), 2)
        self.assertEqual(len(capture.events), 9)
        self.assertIsInstance(capture.states, tuple)
        self.assertIsInstance(capture.events, tuple)
        self.assertEqual(capture.states[1].rgb_frame.render_frame, capture.states[1].clock.render_frame)

    def test_schema_freezes_layout_units_taxonomy_and_failure_envelope(self):
        # Given: the machine-readable contract schema.
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

        # When: stable machine-consumed declarations are selected.
        definitions = schema["$defs"]
        common_required = set(definitions["record_clock"]["required"])
        event_types = definitions["event_type"]["enum"]
        failure_codes = definitions["capture_failure"]["properties"]["failure_code"]["enum"]

        # Then: every clock/identity field and every closed taxonomy value is frozen.
        self.assertEqual(schema["$id"], "https://novphy.org/data-contracts/physics_capture_v1.schema.json")
        self.assertTrue({"schema_version", "capture_id", "shot_id", "sequence", "render_frame", "render_time", "fixed_step", "fixed_time", "coordinates"} <= common_required)
        self.assertEqual(event_types, [event.value for event in EventType])
        self.assertEqual(failure_codes, ["record_limit_exceeded", "byte_limit_exceeded", "capture_timeout", "truncated_finalization"])
        self.assertEqual(definitions["coordinates"]["properties"]["world_length_unit"]["const"], "unity_unit")

    def test_missing_common_fields_are_rejected(self):
        # Given: fresh valid records with one mandatory field removed at a time.
        for field in ("capture_id", "shot_id", "render_time", "fixed_step", "coordinates"):
            with self.subTest(field=field):
                states = _read_jsonl("physics_state.jsonl")
                del states[1][field]

                # When/Then: the boundary fails closed at the missing field.
                self.assert_fixture_error(InvalidFixture(states, _read_jsonl("physics_events.jsonl"), ContractErrorCode.MISSING_FIELD))

    def test_malformed_schema_is_rejected(self):
        # Given: wrong schema version and wrong clock type variants.
        variants = (("schema_version", "physics_capture_v2", ContractErrorCode.UNSUPPORTED_SCHEMA), ("render_frame", "12", ContractErrorCode.WRONG_TYPE))
        for field, value, expected in variants:
            with self.subTest(field=field):
                states = _read_jsonl("physics_state.jsonl")
                states[1][field] = value

                # When/Then: no malformed record reaches the typed capture.
                self.assert_fixture_error(InvalidFixture(states, _read_jsonl("physics_events.jsonl"), expected))

    def test_duplicate_sequence_is_rejected(self):
        # Given: a duplicated state-sidecar sequence.
        states = _read_jsonl("physics_state.jsonl")
        states[2]["sequence"] = states[1]["sequence"]

        # When/Then: strictly increasing sequence is enforced independently.
        self.assert_fixture_error(InvalidFixture(states, _read_jsonl("physics_events.jsonl"), ContractErrorCode.SEQUENCE_ORDER))

    def test_out_of_order_sequence_is_rejected(self):
        # Given: out-of-order event records even though their JSON is otherwise valid.
        events = _read_jsonl("physics_events.jsonl")
        events[1], events[2] = events[2], events[1]

        # When/Then: deterministic event order is mandatory.
        self.assert_fixture_error(InvalidFixture(_read_jsonl("physics_state.jsonl"), events, ContractErrorCode.SEQUENCE_ORDER))

    def test_nonpersistent_support_is_rejected(self):
        # Given: support cites two steps but the retained first-step contact is absent.
        states = _read_jsonl("physics_state.jsonl")
        states[1]["raw_contacts"] = states[1]["raw_contacts"][1:]

        # When/Then: support is never manufactured without retained raw history.
        self.assert_fixture_error(InvalidFixture(states, _read_jsonl("physics_events.jsonl"), ContractErrorCode.NONPERSISTENT_SUPPORT))

    def test_mismatched_render_frame_is_rejected(self):
        # Given: an RGB descriptor that claims a different Unity render frame.
        states = _read_jsonl("physics_state.jsonl")
        states[2]["rgb_frame"]["render_frame"] += 1

        # When/Then: endpoint-returned PNG and snapshot must match exactly.
        self.assert_fixture_error(InvalidFixture(states, _read_jsonl("physics_events.jsonl"), ContractErrorCode.RENDER_FRAME_MISMATCH))

    def test_nondeterministic_node_contact_support_and_event_order_is_rejected(self):
        # Given: each deterministically sorted collection reversed in isolation.
        variants = ("nodes", "raw_contacts", "support_edges")
        for field in variants:
            with self.subTest(field=field):
                states = _read_jsonl("physics_state.jsonl")
                states[2][field].reverse()

                # When/Then: producers cannot emit platform-dependent ordering.
                self.assert_fixture_error(InvalidFixture(states, _read_jsonl("physics_events.jsonl"), ContractErrorCode.DETERMINISTIC_ORDER))

        events = _read_jsonl("physics_events.jsonl")
        events[3]["event_type"], events[4]["event_type"] = events[4]["event_type"], events[3]["event_type"]
        self.assert_fixture_error(InvalidFixture(_read_jsonl("physics_state.jsonl"), events, ContractErrorCode.DETERMINISTIC_ORDER))

    def test_duplicate_collision_pair_per_fixed_step_is_rejected(self):
        # Given: a second collision for the same unordered pair and fixed step.
        events = _read_jsonl("physics_events.jsonl")
        duplicate = dict(events[1])
        duplicate["sequence"] = 2
        duplicate["event_id"] = "event:00000002"
        events.insert(2, duplicate)
        for sequence, event in enumerate(events[3:], start=3):
            event["sequence"] = sequence
            event["event_id"] = f"event:{sequence:08d}"

        # When/Then: collision cardinality is one unordered pair per fixed step.
        self.assert_fixture_error(InvalidFixture(_read_jsonl("physics_state.jsonl"), events, ContractErrorCode.INVALID_EVENT))

    def test_malformed_event_payload_is_rejected(self):
        # Given: a launch event whose taxonomy-specific vector has the wrong shape.
        events = _read_jsonl("physics_events.jsonl")
        events[0]["payload"] = {"launch_velocity": "not-a-vector"}

        # When/Then: closed event payloads are parsed rather than passed through.
        self.assert_fixture_error(InvalidFixture(_read_jsonl("physics_state.jsonl"), events, ContractErrorCode.INVALID_EVENT))

    def test_repeated_entity_lifecycle_event_is_rejected(self):
        # Given: a second destruction event for the same entity lifetime.
        events = _read_jsonl("physics_events.jsonl")
        duplicate = dict(events[3])
        events.insert(4, duplicate)
        for sequence, event in enumerate(events):
            event["sequence"] = sequence
            event["event_id"] = f"event:{sequence:08d}"

        # When/Then: destruction remains one-shot for each stable entity ID.
        self.assert_fixture_error(InvalidFixture(_read_jsonl("physics_state.jsonl"), events, ContractErrorCode.INVALID_EVENT))

    def test_declared_byte_limit_is_enforced(self):
        # Given: complete sidecars larger than the header's declared byte bound.
        states = _read_jsonl("physics_state.jsonl")
        states[0]["capture_limits"]["max_total_bytes"] = 1

        # When/Then: bounded capture cannot be accepted after overflowing bytes.
        self.assert_fixture_error(InvalidFixture(states, _read_jsonl("physics_events.jsonl"), ContractErrorCode.INVALID_VALUE))

    def test_mutated_fixture_is_revalidated_without_stale_cache(self):
        # Given: a valid parse followed by a freshly copied malformed fixture.
        load_physics_capture(FIXTURES / "physics_state.jsonl", FIXTURES / "physics_events.jsonl")
        states = _read_jsonl("physics_state.jsonl")
        states[1]["schema_version"] = "stale_version"

        # When/Then: mutation is observed rather than hidden by cached success.
        self.assert_fixture_error(InvalidFixture(states, _read_jsonl("physics_events.jsonl"), ContractErrorCode.UNSUPPORTED_SCHEMA))


if __name__ == "__main__":
    unittest.main()
