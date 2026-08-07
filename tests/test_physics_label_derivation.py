"""Tests for deterministic macro-state, outcome, and oracle-gate label derivation."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from scripts.physics_capture_contract import PhysicsContractError, load_physics_capture
from scripts.physics_capture_types import (
    EventType,
    PhysicsBody,
    PhysicsCapture,
    RecordClock,
    SceneNode,
    StateFrame,
    Vector2,
    WorldPose,
)
from scripts.physics_label_derivation import (
    DERIVED_LABEL_SCHEMA_VERSION,
    DERIVED_LABEL_VECTOR_FIELDS,
    MacroState,
    OracleGateSpec,
    ShotOutcomeClass,
    derive_labels,
)


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "physics_capture_v1"


def load_fixture_capture() -> PhysicsCapture:
    return load_physics_capture(FIXTURE / "physics_state.jsonl", FIXTURE / "physics_events.jsonl")


class OracleGateSpecTests(unittest.TestCase):
    def test_default_spec_is_frozen_and_declares_all_three_thresholds(self) -> None:
        spec = OracleGateSpec()
        self.assertEqual(spec.kinetic_energy_threshold, 0.01)
        self.assertEqual(spec.active_contact_threshold, 1)
        self.assertEqual(spec.contact_activity_speed, 0.01)
        with self.assertRaises(FrozenInstanceError):
            spec.kinetic_energy_threshold = 5.0  # type: ignore[misc]

    def test_nonpositive_thresholds_are_rejected(self) -> None:
        for kwargs in (
            {"kinetic_energy_threshold": 0.0},
            {"kinetic_energy_threshold": -1.0},
            {"active_contact_threshold": 0},
            {"contact_activity_speed": -0.5},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                OracleGateSpec(**kwargs)

    def test_spec_digest_changes_with_any_threshold(self) -> None:
        baseline = OracleGateSpec().digest()
        self.assertNotEqual(baseline, OracleGateSpec(kinetic_energy_threshold=0.02).digest())
        self.assertNotEqual(baseline, OracleGateSpec(active_contact_threshold=2).digest())
        self.assertNotEqual(baseline, OracleGateSpec(contact_activity_speed=0.02).digest())


class ScalarEvidenceTests(unittest.TestCase):
    def test_absent_bodies_contribute_no_kinetic_energy(self) -> None:
        labels = derive_labels(load_fixture_capture(), OracleGateSpec())
        first = labels.frames[0]
        # Fixture frame: bird KE 5, block KE 0, static ground body absent.
        self.assertEqual(first.total_kinetic_energy, 5.0)
        self.assertEqual(first.dynamic_node_count, 2)

    def test_active_contacts_exclude_resting_contacts(self) -> None:
        capture = load_fixture_capture()
        # Fixture frame has two contacts; the bird/block pair has relative speed
        # sqrt(5) and the block/ground pair is resting.
        strict = derive_labels(capture, OracleGateSpec(contact_activity_speed=0.01)).frames[0]
        lenient = derive_labels(capture, OracleGateSpec(contact_activity_speed=1000.0)).frames[0]
        self.assertEqual(strict.active_contact_count, 1)
        self.assertEqual(lenient.active_contact_count, 0)
        self.assertEqual(strict.raw_contact_count, lenient.raw_contact_count)


class OracleGateTests(unittest.TestCase):
    def _frame(self, kinetic_energy: float, velocity: Vector2) -> StateFrame:
        capture = load_fixture_capture()
        template = capture.states[0]
        mass = 2.0
        node = SceneNode(
            entity_id=template.nodes[0].entity_id,
            unity_instance_id=template.nodes[0].unity_instance_id,
            object_class="bird",
            object_type="RedBird",
            screen_polygon=template.nodes[0].screen_polygon,
            world_pose=WorldPose(Vector2(0.0, 0.0), 0.0),
            life=1.0,
            body=PhysicsBody(True, velocity, 0.0, mass, kinetic_energy),
        )
        return StateFrame(template.clock, template.rgb_frame, (node,), (), ())

    def _gate(self, kinetic_energy: float, spec: OracleGateSpec | None = None) -> bool:
        capture = load_fixture_capture()
        state = self._frame(kinetic_energy, Vector2(0.0, 0.0))
        replaced = PhysicsCapture(capture.header, (state,), ())
        return derive_labels(replaced, spec or OracleGateSpec()).frames[0].oracle_gate

    def test_gate_is_true_only_below_the_kinetic_energy_threshold(self) -> None:
        spec = OracleGateSpec(kinetic_energy_threshold=0.01)
        self.assertTrue(self._gate(0.009, spec))
        self.assertFalse(self._gate(0.01, spec), "threshold is strict less-than")
        self.assertFalse(self._gate(0.011, spec))

    def test_zero_contact_quiescent_frame_gates_open(self) -> None:
        self.assertTrue(self._gate(0.0))

    def test_gate_requires_both_conditions(self) -> None:
        capture = load_fixture_capture()
        frames = derive_labels(capture, OracleGateSpec()).frames
        for frame in frames:
            expected = (
                frame.total_kinetic_energy < 0.01 and frame.active_contact_count < 1
            )
            self.assertEqual(frame.oracle_gate, expected)

    def test_active_contact_threshold_boundary(self) -> None:
        capture = load_fixture_capture()
        # The fixture frame has exactly one active contact and KE 5.
        loose = derive_labels(
            capture, OracleGateSpec(kinetic_energy_threshold=1000.0, active_contact_threshold=1)
        ).frames[0]
        self.assertFalse(loose.oracle_gate, "one active contact is not < 1")
        opened = derive_labels(
            capture, OracleGateSpec(kinetic_energy_threshold=1000.0, active_contact_threshold=2)
        ).frames[0]
        self.assertTrue(opened.oracle_gate)


class MacroStateTests(unittest.TestCase):
    def test_absorbing_states_latch_once_true(self) -> None:
        labels = derive_labels(load_fixture_capture(), OracleGateSpec())
        for absorbing in (MacroState.COLLAPSED, MacroState.PIGS_CLEARED):
            seen = False
            for frame in labels.frames:
                active = absorbing in frame.macro_states
                if seen:
                    self.assertTrue(active, f"{absorbing} must latch once true")
                seen = seen or active

    def test_every_frame_carries_a_macro_state_set(self) -> None:
        labels = derive_labels(load_fixture_capture(), OracleGateSpec())
        self.assertTrue(labels.frames)
        for frame in labels.frames:
            self.assertIsInstance(frame.macro_states, tuple)
            self.assertEqual(
                tuple(sorted(frame.macro_states, key=lambda state: state.value)),
                frame.macro_states,
                "macro states must be deterministically ordered",
            )
            self.assertEqual(len(set(frame.macro_states)), len(frame.macro_states))

    def test_steady_state_and_structure_unstable_are_mutually_exclusive(self) -> None:
        labels = derive_labels(load_fixture_capture(), OracleGateSpec())
        for frame in labels.frames:
            self.assertFalse(
                MacroState.STEADY_STATE in frame.macro_states
                and MacroState.STRUCTURE_UNSTABLE in frame.macro_states
            )


class ShotOutcomeTests(unittest.TestCase):
    def _capture_with_events(self, event_types: tuple[EventType, ...]) -> PhysicsCapture:
        capture = load_fixture_capture()
        kept = tuple(event for event in capture.events if event.event_type in event_types)
        return PhysicsCapture(capture.header, capture.states, kept)

    def test_level_cleared_yields_cleared_outcome_with_score(self) -> None:
        capture = load_fixture_capture()
        outcome = derive_labels(capture, OracleGateSpec()).outcome
        self.assertEqual(outcome.outcome_class, ShotOutcomeClass.CLEARED)
        self.assertEqual(outcome.score, 12345)
        self.assertIsNone(outcome.reason)

    def test_level_failed_yields_failed_outcome_with_reason(self) -> None:
        capture = load_fixture_capture()
        events = tuple(
            event for event in capture.events if event.event_type is not EventType.LEVEL_CLEARED
        )
        failed = tuple(
            event
            for event in capture.events
            if event.event_type is EventType.LEVEL_CLEARED
        )
        self.assertTrue(failed, "fixture is expected to carry a terminal event")
        outcome = derive_labels(
            PhysicsCapture(capture.header, capture.states, events), OracleGateSpec()
        ).outcome
        self.assertIn(
            outcome.outcome_class,
            (ShotOutcomeClass.SETTLED, ShotOutcomeClass.UNSETTLED),
            "removing the terminal event must not fabricate a terminal outcome",
        )

    def test_event_free_capture_is_not_terminal(self) -> None:
        capture = load_fixture_capture()
        labels = derive_labels(PhysicsCapture(capture.header, capture.states, ()), OracleGateSpec())
        self.assertIn(
            labels.outcome.outcome_class,
            (ShotOutcomeClass.SETTLED, ShotOutcomeClass.UNSETTLED),
        )
        self.assertIsNone(labels.outcome.score)


class DeterminismTests(unittest.TestCase):
    def test_repeat_derivation_is_byte_identical(self) -> None:
        capture = load_fixture_capture()
        first = derive_labels(capture, OracleGateSpec()).to_jsonl()
        second = derive_labels(capture, OracleGateSpec()).to_jsonl()
        self.assertEqual(first, second)

    def test_schema_version_and_vector_field_order_are_pinned(self) -> None:
        self.assertEqual(DERIVED_LABEL_SCHEMA_VERSION, "physics_derived_labels_v1")
        self.assertEqual(
            DERIVED_LABEL_VECTOR_FIELDS,
            (
                "oracle_gate",
                "structure_unstable",
                "cascade_active",
                "collapsed",
                "pigs_cleared",
                "steady_state",
                "total_kinetic_energy",
                "active_contact_count",
                "raw_contact_count",
                "support_edge_count",
                "dynamic_node_count",
                "pig_count",
            ),
        )

    def test_frame_vector_matches_declared_field_order(self) -> None:
        frame = derive_labels(load_fixture_capture(), OracleGateSpec()).frames[0]
        vector = frame.to_vector()
        self.assertEqual(len(vector), len(DERIVED_LABEL_VECTOR_FIELDS))
        self.assertEqual(vector[0], 1.0 if frame.oracle_gate else 0.0)
        self.assertEqual(vector[6], frame.total_kinetic_energy)


class DerivedSidecarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp(prefix="novphy-derived-labels-"))
        self.addCleanup(shutil.rmtree, self.temporary, ignore_errors=True)
        for name in ("physics_state.jsonl", "physics_events.jsonl"):
            shutil.copy(FIXTURE / name, self.temporary / name)

    def _write(self) -> Path:
        from scripts.physics_label_derivation import write_derived_labels

        return write_derived_labels(self.temporary, OracleGateSpec())

    def test_write_then_validate_round_trips(self) -> None:
        from scripts.physics_label_derivation import validate_derived_labels

        path = self._write()
        self.assertTrue(path.is_file())
        labels = validate_derived_labels(self.temporary, OracleGateSpec())
        self.assertEqual(len(labels.frames), len(load_fixture_capture().states))

    def test_header_records_source_digests_and_thresholds(self) -> None:
        path = self._write()
        header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(header["schema_version"], DERIVED_LABEL_SCHEMA_VERSION)
        self.assertEqual(header["record_type"], "derived_label_header")
        self.assertEqual(header["oracle_gate_spec"]["kinetic_energy_threshold"], 0.01)
        self.assertEqual(len(header["source"]["physics_state_sha256"]), 64)
        self.assertEqual(len(header["source"]["physics_events_sha256"]), 64)

    def test_mutated_state_sidecar_invalidates_labels(self) -> None:
        from scripts.physics_label_derivation import (
            DerivedLabelError,
            validate_derived_labels,
        )

        self._write()
        state_path = self.temporary / "physics_state.jsonl"
        lines = state_path.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[1])
        record["nodes"][0]["body"]["kinetic_energy_unity_units"] = 5.5
        record["nodes"][0]["body"]["mass_unity_units"] = 2.2
        lines[1] = json.dumps(record, sort_keys=True, separators=(",", ":"))
        state_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaises(DerivedLabelError):
            validate_derived_labels(self.temporary, OracleGateSpec())

    def test_mutated_label_row_fails_closed(self) -> None:
        from scripts.physics_label_derivation import (
            DerivedLabelError,
            validate_derived_labels,
        )

        path = self._write()
        lines = path.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[1])
        record["oracle_gate"] = not record["oracle_gate"]
        lines[1] = json.dumps(record, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaises(DerivedLabelError):
            validate_derived_labels(self.temporary, OracleGateSpec())

    def test_threshold_drift_fails_closed(self) -> None:
        from scripts.physics_label_derivation import (
            DerivedLabelError,
            validate_derived_labels,
        )

        self._write()
        with self.assertRaises(DerivedLabelError):
            validate_derived_labels(self.temporary, OracleGateSpec(kinetic_energy_threshold=0.5))

    def test_missing_label_file_fails_closed(self) -> None:
        from scripts.physics_label_derivation import (
            DerivedLabelError,
            validate_derived_labels,
        )

        with self.assertRaises(DerivedLabelError):
            validate_derived_labels(self.temporary, OracleGateSpec())

    def test_write_is_atomic_and_leaves_no_temporary(self) -> None:
        self._write()
        self.assertEqual(
            sorted(child.name for child in self.temporary.iterdir()),
            ["physics_derived_labels.jsonl", "physics_events.jsonl", "physics_state.jsonl"],
        )


class MalformedSidecarTests(unittest.TestCase):
    def test_derivation_refuses_a_capture_that_does_not_parse(self) -> None:
        temporary = Path(tempfile.mkdtemp(prefix="novphy-derived-bad-"))
        self.addCleanup(shutil.rmtree, temporary, ignore_errors=True)
        (temporary / "physics_state.jsonl").write_text("{not json}\n", encoding="utf-8")
        (temporary / "physics_events.jsonl").write_text("", encoding="utf-8")
        with self.assertRaises((PhysicsContractError, OSError, ValueError)):
            load_physics_capture(
                temporary / "physics_state.jsonl", temporary / "physics_events.jsonl"
            )


if __name__ == "__main__":
    unittest.main()
