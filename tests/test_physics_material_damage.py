from dataclasses import replace
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from scripts.physics_capture_contract import load_physics_capture
from scripts.physics_capture_types import EventType, PhysicsCapture
from scripts.physics_macro_labels import SemanticStatus
from scripts.physics_material_damage import (
    Availability,
    DAMAGE_LIFECYCLE_MAPPING_VERSION,
    DamageLifecycleMapping,
    MaterialDamageContractError,
    MaterialDamageValidationError,
    MATERIAL_UNAVAILABLE_LABEL,
    SUPPORTED_DAMAGE_LIFECYCLE_MAPPING,
    build_damage_lifecycle_validation_receipt,
    canonical_json_bytes,
    derive_material_damage,
    validate_material_damage_sidecar,
    write_material_damage_sidecar,
)


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "physics_capture_v1"


def _capture() -> PhysicsCapture:
    return load_physics_capture(FIXTURE / "physics_state.jsonl", FIXTURE / "physics_events.jsonl")


def _without_positive_evidence(capture: PhysicsCapture) -> PhysicsCapture:
    states = tuple(
        replace(
            state,
            nodes=tuple(replace(node, life=50.0 if node.life is not None else None) for node in state.nodes),
        )
        for state in capture.states
    )
    events = tuple(
        event
        for event in capture.events
        if not (
            event.event_type is EventType.ENTITY_DESTROYED
            and json.loads(event.payload_json).get("reason") == "damage"
        )
    )
    return replace(capture, states=states, events=events)


class PhysicsMaterialDamageTests(unittest.TestCase):
    def test_mapping_is_immutable_and_does_not_claim_verification(self) -> None:
        mapping = SUPPORTED_DAMAGE_LIFECYCLE_MAPPING
        self.assertEqual(mapping.mapping_version, DAMAGE_LIFECYCLE_MAPPING_VERSION)
        self.assertNotIn("representative_validation_status", mapping.to_json())
        with self.assertRaises(TypeError):
            mapping.mapping_version = "mutated"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            DamageLifecycleMapping("caller-status")  # type: ignore[call-arg]

    def test_positive_and_insufficient_cohorts_issue_different_receipts(self) -> None:
        positive = build_damage_lifecycle_validation_receipt((_capture(),))
        pending = build_damage_lifecycle_validation_receipt((_without_positive_evidence(_capture()),))

        self.assertIs(positive.status, SemanticStatus.ENGINE_VERIFIED)
        self.assertTrue(positive.source_records[0].life_decrease_witnesses)
        self.assertTrue(positive.source_records[0].destruction_witnesses)
        self.assertIs(
            pending.status,
            SemanticStatus.HYPOTHESIS_PENDING_REPRESENTATIVE_VALIDATION,
        )
        self.assertNotEqual(positive.source_cohort_identity, pending.source_cohort_identity)
        self.assertEqual(positive.mapping_digest, SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.digest)

    def test_pending_receipt_requires_both_witness_classes(self) -> None:
        capture = _capture()
        life_only = replace(
            capture,
            events=tuple(
                event
                for event in capture.events
                if event.event_type is not EventType.ENTITY_DESTROYED
            ),
        )
        event_only = _without_positive_evidence(capture)
        life_receipt = build_damage_lifecycle_validation_receipt((life_only,))
        event_receipt = build_damage_lifecycle_validation_receipt((event_only,))

        self.assertIsNot(life_receipt.status, SemanticStatus.ENGINE_VERIFIED)
        self.assertIsNot(event_receipt.status, SemanticStatus.ENGINE_VERIFIED)

    def test_material_and_raw_life_are_fail_closed_and_immutable(self) -> None:
        capture = _capture()
        receipt = build_damage_lifecycle_validation_receipt((capture,))
        before = tuple(node.life for state in capture.states for node in state.nodes)
        artifact = derive_material_damage(capture, receipt=receipt)

        self.assertTrue(all(record.material_label == MATERIAL_UNAVAILABLE_LABEL for record in artifact.records))
        self.assertTrue(
            all(
                record.material_availability
                is Availability.UNAVAILABLE_MISSING_ENGINE_MATERIAL_FIELD
                for record in artifact.records
            )
        )
        self.assertEqual(before, tuple(node.life for state in capture.states for node in state.nodes))

    def test_unknown_or_pending_receipts_never_emit_damage_or_no_damage(self) -> None:
        capture = _capture()
        pending = build_damage_lifecycle_validation_receipt((_without_positive_evidence(capture),))
        no_receipt = derive_material_damage(capture)
        pending_artifact = derive_material_damage(_without_positive_evidence(capture), receipt=pending)
        unknown_mapping = derive_material_damage(capture, mapping=DamageLifecycleMapping())

        for artifact in (no_receipt, pending_artifact, unknown_mapping):
            self.assertTrue(all(record.damage_label == MATERIAL_UNAVAILABLE_LABEL for record in artifact.records))
            self.assertTrue(
                all(
                    record.damage_availability
                    in {
                        Availability.UNAVAILABLE_INSUFFICIENT_DAMAGE_LIFECYCLE_EVIDENCE,
                        Availability.UNAVAILABLE_UNSUPPORTED_MAPPING,
                    }
                    for record in artifact.records
                )
            )

    def test_first_frame_requires_predecessor_and_positive_lifecycle_evidence_labels_damage(self) -> None:
        capture = _capture()
        receipt = build_damage_lifecycle_validation_receipt((capture,))
        artifact = derive_material_damage(capture, receipt=receipt)
        block = [record for record in artifact.records if record.entity_id == "201:0"]
        by_fixed_step = {record.fixed_step: record for record in block}

        self.assertEqual(by_fixed_step[10].damage_label, MATERIAL_UNAVAILABLE_LABEL)
        self.assertEqual(
            by_fixed_step[10].damage_availability,
            Availability.UNAVAILABLE_INSUFFICIENT_DAMAGE_LIFECYCLE_EVIDENCE,
        )
        self.assertEqual(by_fixed_step[11].damage_label, "damage")
        self.assertEqual(by_fixed_step[11].damage_availability, Availability.AVAILABLE)

    def test_damage_events_join_by_fixed_step_and_support_event_scoped_records(self) -> None:
        capture = _capture()
        receipt = build_damage_lifecycle_validation_receipt((capture,))
        shifted_event = next(event for event in capture.events if event.event_type is EventType.ENTITY_DESTROYED)
        shifted_clock = replace(shifted_event.clock, render_frame=999, fixed_step=10)
        shifted = replace(capture, events=tuple(
            replace(event, clock=shifted_clock) if event is shifted_event else event
            for event in capture.events
        ))
        shifted_receipt = build_damage_lifecycle_validation_receipt((shifted,))
        shifted_artifact = derive_material_damage(shifted, receipt=shifted_receipt)
        attached = [record for record in shifted_artifact.records if record.fixed_step == 10 and record.entity_id == "201:0"]
        self.assertEqual(attached[0].damage_label, "damage")
        self.assertEqual(attached[0].render_frame, 100)

        artifact = derive_material_damage(capture, receipt=receipt)
        scoped = [record for record in artifact.records if record.fixed_step == 12 and record.entity_id == "201:0"]
        self.assertEqual(len(scoped), 1)
        self.assertEqual(scoped[0].damage_evidence_event_ids, ("event:00000003",))

    def test_sidecar_requires_receipt_and_rejects_receipt_source_or_content_drift(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "shot_001"
            shutil.copytree(FIXTURE, source)
            receipt = build_damage_lifecycle_validation_receipt((source,), cohort_context={"cohort": "fixture"})
            destination = root / "physics_material_damage.json"
            expected = write_material_damage_sidecar(destination, source, receipt=receipt)
            self.assertEqual(validate_material_damage_sidecar(destination, source, receipt=receipt), expected)
            self.assertEqual(expected.source_cohort_identity, receipt.source_cohort_identity)

            payload = json.loads(destination.read_bytes())
            payload["mapping_digest"] = "f" * 64
            destination.write_bytes(canonical_json_bytes(payload))
            with self.assertRaises(MaterialDamageValidationError):
                validate_material_damage_sidecar(destination, source, receipt=receipt)

            write_material_damage_sidecar(destination, source, receipt=receipt)
            state_path = source / "physics_state.jsonl"
            state_path.write_bytes(state_path.read_bytes().replace(b'"life":99', b'"life":98', 1))
            with self.assertRaises(MaterialDamageValidationError):
                validate_material_damage_sidecar(destination, source, receipt=receipt)

            with self.assertRaises(TypeError):
                write_material_damage_sidecar(destination, source, source_cohort_identity="arbitrary")  # type: ignore[call-arg]

    def test_artifact_serializes_the_complete_receipt_source_set(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            shutil.copytree(FIXTURE, first)
            shutil.copytree(FIXTURE, second)
            for name in ("physics_state.jsonl", "physics_events.jsonl"):
                path = second / name
                path.write_bytes(
                    path.read_bytes()
                    .replace(b"capture-golden-001", b"capture-golden-002")
                    .replace(b"shot_001", b"shot_002")
                )
            receipt = build_damage_lifecycle_validation_receipt((first, second), cohort_context={"pilot": "test"})
            artifact = derive_material_damage(first, receipt=receipt)

            self.assertEqual(len(receipt.source_records), 2)
            self.assertEqual(artifact.source_records, receipt.source_records)
            serialized = json.loads(artifact.to_bytes())
            self.assertEqual(serialized["source_records"], [record.to_json() for record in receipt.source_records])

    def test_invalid_receipt_object_returns_unavailable_records(self) -> None:
        capture = _capture()
        artifact = derive_material_damage(capture, receipt=object())
        self.assertTrue(all(record.damage_label == MATERIAL_UNAVAILABLE_LABEL for record in artifact.records))


if __name__ == "__main__":
    unittest.main()
