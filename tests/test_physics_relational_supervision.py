from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from scripts.physics_capture_contract import load_physics_capture
from scripts.physics_capture_types import PhysicsCapture, StateFrame
from scripts.physics_relational_supervision import (
    RELATIONAL_SUPERVISION_SCHEMA_VERSION,
    RELATIONAL_SUPERVISION_SIDECAR,
    RelationalAvailability,
    derive_relational_supervision_for_shot,
    read_relational_supervision,
    validate_relational_supervision,
    write_relational_supervision,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/physics_capture_v1"


def _capture() -> PhysicsCapture:
    return load_physics_capture(FIXTURE / "physics_state.jsonl", FIXTURE / "physics_events.jsonl")


def _copy_capture(capture: PhysicsCapture, *, states: tuple[StateFrame, ...] | None = None) -> PhysicsCapture:
    return PhysicsCapture(capture.header, capture.states if states is None else states, capture.events)


class RelationalDerivationTests(unittest.TestCase):
    def test_contact_truths_are_unordered_symmetric_and_non_trigger_only(self) -> None:
        labels = derive_relational_supervision_for_shot(FIXTURE)
        frame = labels.frames[1]

        self.assertEqual(
            tuple(contact.pair for contact in frame.contacts),
            (("101:0", "201:0"), ("201:0", "world:static:900")),
        )
        self.assertEqual(frame.contact_truth(("201:0", "101:0")).pair, ("101:0", "201:0"))
        self.assertEqual(
            frame.contact_truth(("201:0", "101:0")).evidence[0].contact_id,
            "contact:11:101:0|1101:201:0|1201:0",
        )

    def test_support_is_true_with_source_evidence_and_false_only_with_complete_history(self) -> None:
        capture = _capture()
        without_support = tuple(
            StateFrame(state.clock, state.rgb_frame, state.nodes, state.raw_contacts, ())
            for state in capture.states
        )
        labels = derive_relational_supervision_for_shot(
            FIXTURE,
            capture=_copy_capture(capture, states=without_support),
        )
        frame = labels.frames[1]

        true_label = derive_relational_supervision_for_shot(FIXTURE).frames[1].support_label(
            ("101:0", "201:0")
        )
        self.assertTrue(true_label.value)
        self.assertEqual(len(true_label.evidence), 2)

        false_label = frame.support_label(("101:0", "201:0"))
        self.assertFalse(false_label.value)
        self.assertIs(false_label.availability, RelationalAvailability.AVAILABLE)
        self.assertEqual((false_label.supporter_id, false_label.supported_id), ("201:0", "101:0"))
        self.assertEqual(len(false_label.evidence), 2)

    def test_active_node_pair_without_contacts_is_retained_as_unavailable(self) -> None:
        frame = derive_relational_supervision_for_shot(FIXTURE).frames[1]

        label = frame.support_label(("101:0", "world:static:900"))
        self.assertIsNone(label.value)
        self.assertIs(
            label.availability,
            RelationalAvailability.UNAVAILABLE_INSUFFICIENT_CONTACT_EVIDENCE,
        )

    def test_missing_predecessor_lifecycle_contact_or_geometry_is_unavailable(self) -> None:
        capture = _capture()
        first = derive_relational_supervision_for_shot(FIXTURE).frames[0]
        self.assertIs(
            first.support_label(("101:0", "201:0")).availability,
            RelationalAvailability.UNAVAILABLE_NO_PREDECESSOR,
        )

        missing_contacts = StateFrame(
            capture.states[1].clock,
            capture.states[1].rgb_frame,
            capture.states[1].nodes,
            (),
            (),
        )
        labels = derive_relational_supervision_for_shot(
            FIXTURE, capture=_copy_capture(capture, states=(capture.states[0], missing_contacts))
        )
        self.assertIs(
            labels.frames[1].support_label(("101:0", "201:0")).availability,
            RelationalAvailability.UNAVAILABLE_INSUFFICIENT_CONTACT_EVIDENCE,
        )

        no_nodes = StateFrame(
            capture.states[1].clock,
            capture.states[1].rgb_frame,
            (),
            capture.states[1].raw_contacts,
            (),
        )
        labels = derive_relational_supervision_for_shot(
            FIXTURE, capture=_copy_capture(capture, states=(capture.states[0], no_nodes))
        )
        self.assertIs(
            labels.frames[1].support_label(("101:0", "201:0")).availability,
            RelationalAvailability.UNAVAILABLE_INSUFFICIENT_LIFECYCLE_EVIDENCE,
        )

        skipped_clock = replace(
            capture.states[1].clock,
            fixed_step=capture.states[1].clock.fixed_step + 2,
            fixed_time=capture.states[1].clock.fixed_time + 0.04,
        )
        skipped = replace(capture.states[1], clock=skipped_clock, support_edges=())
        labels = derive_relational_supervision_for_shot(
            FIXTURE, capture=_copy_capture(capture, states=(capture.states[0], skipped))
        )
        self.assertIs(
            labels.frames[1].support_label(("101:0", "201:0")).availability,
            RelationalAvailability.UNAVAILABLE_INSUFFICIENT_PREDECESSOR,
        )

        same_height_nodes = tuple(
            replace(
                node,
                world_pose=replace(
                    node.world_pose,
                    position=replace(node.world_pose.position, y=2.0),
                ),
            )
            for node in capture.states[1].nodes
        )
        same_height = replace(capture.states[1], nodes=same_height_nodes, support_edges=())
        labels = derive_relational_supervision_for_shot(
            FIXTURE, capture=_copy_capture(capture, states=(capture.states[0], same_height))
        )
        self.assertIs(
            labels.frames[1].support_label(("101:0", "201:0")).availability,
            RelationalAvailability.UNAVAILABLE_INSUFFICIENT_GEOMETRY_EVIDENCE,
        )

    def test_trigger_contacts_never_become_contact_truths_and_future_states_do_not_relabel_past(self) -> None:
        capture = _capture()
        triggered = tuple(
            replace(contact, is_trigger=True)
            for contact in capture.states[1].raw_contacts
        )
        trigger_state = replace(capture.states[1], raw_contacts=triggered, support_edges=())
        labels = derive_relational_supervision_for_shot(
            FIXTURE, capture=_copy_capture(capture, states=(capture.states[0], trigger_state))
        )
        self.assertEqual(labels.frames[1].contacts, ())

        changed_future = replace(capture.states[1], support_edges=())
        original_labels = derive_relational_supervision_for_shot(FIXTURE, capture=capture)
        first_original = original_labels.frames[0]
        first_changed = derive_relational_supervision_for_shot(
            FIXTURE, capture=_copy_capture(capture, states=(capture.states[0], changed_future))
        ).frames[0]
        self.assertEqual(first_original.to_json(), first_changed.to_json())

        unrelated_events = PhysicsCapture(capture.header, capture.states, ())
        without_events = derive_relational_supervision_for_shot(
            FIXTURE, capture=unrelated_events
        )
        self.assertEqual(
            tuple(frame.to_json() for frame in without_events.frames),
            tuple(frame.to_json() for frame in original_labels.frames),
        )

    def test_physical_regime_and_model_usefulness_are_distinct(self) -> None:
        frame = derive_relational_supervision_for_shot(FIXTURE).frames[1]
        self.assertIsNone(frame.physical_regime_eligibility.value)
        self.assertIs(
            frame.physical_regime_eligibility.availability,
            RelationalAvailability.UNAVAILABLE_NO_DECLARED_PHYSICAL_REGIME_DERIVATION,
        )
        self.assertIsNone(frame.model_relative_micro_relation_usefulness.value)
        self.assertIs(
            frame.model_relative_micro_relation_usefulness.availability,
            RelationalAvailability.UNAVAILABLE_NOT_DERIVABLE,
        )
        self.assertIsNot(
            frame.physical_regime_eligibility.availability,
            frame.model_relative_micro_relation_usefulness.availability,
        )

    def test_sidecar_is_deterministic_source_bound_and_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot_001"
            shot.mkdir()
            for name in ("physics_state.jsonl", "physics_events.jsonl"):
                (shot / name).write_bytes((FIXTURE / name).read_bytes())

            path = write_relational_supervision(shot)
            first = path.read_bytes()
            self.assertEqual(path.name, RELATIONAL_SUPERVISION_SIDECAR)
            self.assertEqual(first, write_relational_supervision(shot).read_bytes())
            self.assertEqual(read_relational_supervision(path).schema_version, RELATIONAL_SUPERVISION_SCHEMA_VERSION)

            states = (shot / "physics_state.jsonl").read_text(encoding="utf-8").splitlines()
            record = json.loads(states[2])
            record["render_frame"] = 102
            states[2] = json.dumps(record, sort_keys=True, separators=(",", ":"))
            (shot / "physics_state.jsonl").write_text("\n".join(states) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_relational_supervision(shot)


if __name__ == "__main__":
    unittest.main()
