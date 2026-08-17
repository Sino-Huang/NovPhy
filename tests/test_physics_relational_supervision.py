from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from scripts.physics_capture_contract import load_physics_capture
from scripts.physics_capture_types import ContactId, PhysicsCapture, StateFrame
from scripts.physics_relational_supervision import (
    RELATIONAL_SUPERVISION_SCHEMA_VERSION,
    RELATIONAL_SUPERVISION_SIDECAR,
    RelationalAvailability,
    RelationalSupervisionError,
    derive_relational_supervision,
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

    def test_removed_support_edge_with_positive_raw_evidence_is_unavailable(self) -> None:
        capture = _capture()
        without_support = tuple(
            StateFrame(state.clock, state.rgb_frame, state.nodes, state.raw_contacts, ())
            for state in capture.states
        )
        labels = derive_relational_supervision(
            _copy_capture(capture, states=without_support),
        )
        frame = labels.frames[1]

        true_label = derive_relational_supervision_for_shot(FIXTURE).frames[1].support_label(
            ("101:0", "201:0")
        )
        self.assertTrue(true_label.value)
        self.assertEqual(len(true_label.evidence), 2)

        missing_edge_label = frame.support_label(("101:0", "201:0"))
        self.assertIsNone(missing_edge_label.value)
        self.assertIs(
            missing_edge_label.availability,
            RelationalAvailability.UNAVAILABLE_MISSING_OR_INCONSISTENT_POSITIVE_SUPPORT_DERIVATION,
        )
        self.assertEqual(
            (missing_edge_label.supporter_id, missing_edge_label.supported_id),
            ("201:0", "101:0"),
        )
        self.assertEqual(len(missing_edge_label.evidence), 2)

    def test_retained_contacts_are_not_current_truth_or_negative_support_evidence(self) -> None:
        capture = _capture()
        old_contact = capture.states[0].raw_contacts[0]
        current_contact = capture.states[1].raw_contacts[0]

        retained_and_current = replace(
            capture.states[1],
            raw_contacts=(old_contact, current_contact),
            support_edges=(),
        )
        frame = derive_relational_supervision(
            _copy_capture(capture, states=(capture.states[0], retained_and_current)),
        ).frames[1]
        self.assertEqual(
            tuple(citation.contact_id for citation in frame.contact_truth(("101:0", "201:0")).evidence),
            (str(current_contact.contact_id),),
        )

        retained_only = replace(
            capture.states[1],
            raw_contacts=(old_contact,),
            support_edges=(),
        )
        frame = derive_relational_supervision(
            _copy_capture(capture, states=(capture.states[0], retained_only)),
        ).frames[1]
        self.assertEqual(frame.contacts, ())
        label = frame.support_label(("101:0", "201:0"))
        self.assertIsNone(label.value)
        self.assertIs(
            label.availability,
            RelationalAvailability.UNAVAILABLE_INSUFFICIENT_CONTACT_EVIDENCE,
        )

    def test_true_support_can_cite_retained_rows_at_their_actual_fixed_steps(self) -> None:
        capture = _capture()
        states = (
            replace(capture.states[0], raw_contacts=()),
            replace(
                capture.states[1],
                raw_contacts=(*capture.states[0].raw_contacts, *capture.states[1].raw_contacts),
            ),
        )

        label = derive_relational_supervision(
            _copy_capture(capture, states=states),
        ).frames[1].support_label(("101:0", "201:0"))

        self.assertTrue(label.value)
        self.assertEqual(tuple(citation.fixed_step for citation in label.evidence), (10, 11))
        self.assertEqual(
            tuple(citation.contact_id for citation in label.evidence),
            (
                "contact:10:101:0|1101:201:0|1201:0",
                "contact:11:101:0|1101:201:0|1201:0",
            ),
        )

    def test_malformed_contact_id_fails_closed(self) -> None:
        capture = _capture()
        malformed_contact = replace(
            capture.states[0].raw_contacts[0],
            contact_id=ContactId("contact:10:malformed"),
        )
        malformed_state = replace(
            capture.states[0],
            raw_contacts=(malformed_contact, *capture.states[0].raw_contacts[1:]),
        )

        with self.assertRaisesRegex(
            RelationalSupervisionError,
            "contact id does not encode its fixed step",
        ):
            derive_relational_supervision(
                _copy_capture(capture, states=(malformed_state, *capture.states[1:])),
            )

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
        labels = derive_relational_supervision(
            _copy_capture(capture, states=(capture.states[0], missing_contacts))
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
        labels = derive_relational_supervision(
            _copy_capture(capture, states=(capture.states[0], no_nodes))
        )
        self.assertIs(
            labels.frames[1].support_label(("101:0", "201:0")).availability,
            RelationalAvailability.UNAVAILABLE_INSUFFICIENT_GEOMETRY_EVIDENCE,
        )

        nonfinite_nodes = tuple(
            replace(
                node,
                world_pose=replace(
                    node.world_pose,
                    position=replace(node.world_pose.position, y=float("nan")),
                ),
            )
            if str(node.entity_id) == "101:0"
            else node
            for node in capture.states[1].nodes
        )
        nonfinite_geometry = replace(
            capture.states[1], nodes=nonfinite_nodes, support_edges=()
        )
        labels = derive_relational_supervision(
            _copy_capture(capture, states=(capture.states[0], nonfinite_geometry))
        )
        self.assertIs(
            labels.frames[1].support_label(("101:0", "201:0")).availability,
            RelationalAvailability.UNAVAILABLE_INSUFFICIENT_GEOMETRY_EVIDENCE,
        )

        skipped_clock = replace(
            capture.states[1].clock,
            fixed_step=capture.states[1].clock.fixed_step + 2,
            fixed_time=capture.states[1].clock.fixed_time + 0.04,
        )
        skipped = replace(capture.states[1], clock=skipped_clock, support_edges=())
        labels = derive_relational_supervision(
            _copy_capture(capture, states=(capture.states[0], skipped))
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
        labels = derive_relational_supervision(
            _copy_capture(capture, states=(capture.states[0], same_height))
        )
        label = labels.frames[1].support_label(("101:0", "201:0"))
        self.assertIs(label.value, False)
        self.assertIs(label.availability, RelationalAvailability.AVAILABLE)
        self.assertEqual(
            (label.supporter_id, label.supported_id),
            (None, None),
        )

    def test_negative_support_requires_prior_geometry_and_a_clear_lifecycle_interval(self) -> None:
        capture = _capture()
        without_support = tuple(replace(state, support_edges=()) for state in capture.states)

        prior_same_height_nodes = tuple(
            replace(
                node,
                world_pose=replace(
                    node.world_pose,
                    position=replace(node.world_pose.position, y=2.0),
                ),
            )
            if str(node.entity_id) in ("101:0", "201:0")
            else node
            for node in without_support[0].nodes
        )
        prior_same_height = replace(without_support[0], nodes=prior_same_height_nodes)
        frame = derive_relational_supervision(
            _copy_capture(capture, states=(prior_same_height, without_support[1])),
        ).frames[1]
        label = frame.support_label(("101:0", "201:0"))
        self.assertIs(label.value, False)
        self.assertIs(label.availability, RelationalAvailability.AVAILABLE)
        self.assertEqual((label.supporter_id, label.supported_id), (None, None))

        interval_destruction = replace(
            capture.events[3],
            clock=replace(
                capture.events[3].clock,
                fixed_step=without_support[1].clock.fixed_step,
                fixed_time=without_support[1].clock.fixed_time,
            ),
        )
        lifecycle_capture = PhysicsCapture(
            capture.header,
            without_support,
            (interval_destruction,),
        )
        frame = derive_relational_supervision(lifecycle_capture).frames[1]
        label = frame.support_label(("101:0", "201:0"))
        self.assertIsNone(label.value)
        self.assertIs(
            label.availability,
            RelationalAvailability.UNAVAILABLE_INSUFFICIENT_LIFECYCLE_EVIDENCE,
        )

    def test_unrelated_lifecycle_event_does_not_block_negative_support(self) -> None:
        capture = _capture()
        without_support = tuple(replace(state, support_edges=()) for state in capture.states)
        current_with_disqualifying_normal = replace(
            without_support[1],
            raw_contacts=tuple(
                replace(
                    contact,
                    normal_a_to_b=replace(contact.normal_a_to_b, y=0.0),
                )
                if {str(contact.entity_a_id), str(contact.entity_b_id)} == {"101:0", "201:0"}
                else contact
                for contact in without_support[1].raw_contacts
            ),
        )
        states = (without_support[0], current_with_disqualifying_normal)
        unrelated_explosion = replace(
            capture.events[2],
            clock=replace(
                capture.events[2].clock,
                fixed_step=states[1].clock.fixed_step,
                fixed_time=states[1].clock.fixed_time,
            ),
        )
        frame = derive_relational_supervision(
            PhysicsCapture(capture.header, states, (unrelated_explosion,)),
        ).frames[1]

        label = frame.support_label(("101:0", "201:0"))
        self.assertFalse(label.value)
        self.assertIs(label.availability, RelationalAvailability.AVAILABLE)
        self.assertEqual(tuple(citation.fixed_step for citation in label.evidence), (10, 11))

    def test_trigger_contacts_never_become_contact_truths_and_future_states_do_not_relabel_past(self) -> None:
        capture = _capture()
        triggered = tuple(
            replace(contact, is_trigger=True)
            for contact in capture.states[1].raw_contacts
        )
        trigger_state = replace(capture.states[1], raw_contacts=triggered, support_edges=())
        labels = derive_relational_supervision(
            _copy_capture(capture, states=(capture.states[0], trigger_state))
        )
        self.assertEqual(labels.frames[1].contacts, ())

        changed_future = replace(capture.states[1], support_edges=())
        original_labels = derive_relational_supervision(capture)
        first_original = original_labels.frames[0]
        first_changed = derive_relational_supervision(
            _copy_capture(capture, states=(capture.states[0], changed_future))
        ).frames[0]
        self.assertEqual(first_original.to_json(), first_changed.to_json())

        unrelated_events = PhysicsCapture(capture.header, capture.states, ())
        without_events = derive_relational_supervision(unrelated_events)
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

    def test_writer_is_source_bound_and_pure_labels_cannot_be_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot_001"
            shot.mkdir()
            for name in ("physics_state.jsonl", "physics_events.jsonl"):
                (shot / name).write_bytes((FIXTURE / name).read_bytes())

            path = write_relational_supervision(shot)
            self.assertEqual(path.name, RELATIONAL_SUPERVISION_SIDECAR)
            self.assertEqual(validate_relational_supervision(shot), read_relational_supervision(path))

            with self.assertRaises(TypeError):
                write_relational_supervision(derive_relational_supervision(_capture()))  # type: ignore[arg-type]

    def test_source_bound_shot_derivation_rejects_alternate_capture(self) -> None:
        with self.assertRaises(TypeError):
            derive_relational_supervision_for_shot(FIXTURE, capture=_capture())  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
