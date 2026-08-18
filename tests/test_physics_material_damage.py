from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from scripts.physics_capture_contract import load_physics_capture
from scripts.physics_capture_types import PhysicsCapture
from scripts.physics_macro_labels import SemanticStatus
from scripts.physics_material_damage import (
    Availability,
    DAMAGE_LIFECYCLE_MAPPING_VERSION,
    DamageLifecycleMapping,
    MaterialDamageContractError,
    MaterialDamageValidationError,
    MATERIAL_UNAVAILABLE_LABEL,
    SOURCE_COHORT_IDENTITY,
    SUPPORTED_DAMAGE_LIFECYCLE_MAPPING,
    canonical_json_bytes,
    derive_material_damage,
    validate_material_damage_sidecar,
    write_material_damage_sidecar,
)


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "physics_capture_v1"


def _capture() -> PhysicsCapture:
    return load_physics_capture(FIXTURE / "physics_state.jsonl", FIXTURE / "physics_events.jsonl")


def _with_life(capture: PhysicsCapture, delta: float) -> PhysicsCapture:
    states = tuple(
        replace(
            state,
            nodes=tuple(replace(node, life=None if node.life is None else node.life + delta) for node in state.nodes),
        )
        for state in capture.states
    )
    return replace(capture, states=states)


class PhysicsMaterialDamageTests(unittest.TestCase):
    def test_raw_life_is_not_mutated_and_material_is_life_invariant(self) -> None:
        capture = _capture()
        original_life = tuple(node.life for state in capture.states for node in state.nodes)
        baseline = derive_material_damage(capture)
        changed = derive_material_damage(_with_life(capture, 500.0))

        self.assertEqual(
            tuple(record.material_label for record in baseline.records),
            tuple(record.material_label for record in changed.records),
        )
        self.assertTrue(all(label == MATERIAL_UNAVAILABLE_LABEL for label in (
            record.material_label for record in baseline.records
        )))
        self.assertEqual(original_life, tuple(node.life for state in capture.states for node in state.nodes))

    def test_material_is_unavailable_for_material_looking_object_types(self) -> None:
        artifact = derive_material_damage(_capture())
        wood_records = tuple(record for record in artifact.records if record.entity_id == "201:0")

        self.assertTrue(wood_records)
        self.assertTrue(all(record.material_label == MATERIAL_UNAVAILABLE_LABEL for record in wood_records))
        self.assertTrue(
            all(
                record.material_availability
                is Availability.UNAVAILABLE_MISSING_ENGINE_MATERIAL_FIELD
                for record in wood_records
            )
        )

    def test_damage_uses_life_decrease_and_damage_destruction_reason(self) -> None:
        artifact = derive_material_damage(_capture())
        block_records = tuple(record for record in artifact.records if record.entity_id == "201:0")
        by_frame = {record.render_frame: record for record in block_records}

        self.assertEqual(by_frame[101].damage_label, "damage")
        self.assertEqual(by_frame[101].damage_availability, Availability.AVAILABLE)
        self.assertEqual(by_frame[102].damage_label, "damage")
        self.assertEqual(by_frame[102].damage_evidence_event_ids, ("event:00000003",))

    def test_unknown_unverified_and_envelope_mismatch_fail_closed(self) -> None:
        self.assertIs(
            SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.representative_validation_status,
            SemanticStatus.ENGINE_VERIFIED,
        )
        with self.assertRaises(TypeError):
            DamageLifecycleMapping(SemanticStatus.HYPOTHESIS_PENDING_REPRESENTATIVE_VALIDATION)  # type: ignore[call-arg]
        with self.assertRaises(MaterialDamageContractError):
            derive_material_damage(_capture(), mapping=DamageLifecycleMapping())

        mismatch = (("capture_schema_version", "future_capture_v9"),)
        artifact = derive_material_damage(_capture(), version_envelope=mismatch)
        self.assertTrue(
            all(
                record.damage_availability is Availability.UNAVAILABLE_VERSION_ENVELOPE_MISMATCH
                for record in artifact.records
            )
        )
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "derived.json"
            with self.assertRaises(MaterialDamageContractError):
                write_material_damage_sidecar(destination, _capture(), version_envelope=mismatch)
            self.assertFalse(destination.exists())

    def test_repeated_entity_ids_are_frame_scoped(self) -> None:
        artifact = derive_material_damage(_capture())
        keys = tuple(record.frame_scoped_key for record in artifact.records)

        self.assertEqual(len(keys), len(set(keys)))
        self.assertGreater(sum(record.entity_id == "201:0" for record in artifact.records), 1)
        self.assertEqual(artifact.mapping_version, DAMAGE_LIFECYCLE_MAPPING_VERSION)

    def test_sidecar_is_source_bound_and_rejects_canonical_tampering(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "shot_001"
            shutil.copytree(FIXTURE, source)
            destination = root / "physics_material_damage.json"
            expected = write_material_damage_sidecar(
                destination,
                source,
                source_cohort_identity="cohort:test-001",
            )
            self.assertEqual(
                validate_material_damage_sidecar(
                    destination,
                    source,
                    source_cohort_identity="cohort:test-001",
                ),
                expected,
            )
            self.assertEqual(expected.records[0].source_cohort_identity, "cohort:test-001")
            self.assertEqual(expected.records[0].mapping_digest, SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.digest)
            self.assertEqual(
                expected.records[0].state_sha256,
                hashlib.sha256((source / "physics_state.jsonl").read_bytes()).hexdigest(),
            )

            payload = json.loads(destination.read_bytes())
            payload["records"][0]["mapping_version"] = "tampered"
            destination.write_bytes(canonical_json_bytes(payload))
            with self.assertRaisesRegex(MaterialDamageValidationError, "re-derivation"):
                validate_material_damage_sidecar(
                    destination,
                    source,
                    source_cohort_identity="cohort:test-001",
                )

            write_material_damage_sidecar(
                destination,
                source,
                source_cohort_identity="cohort:test-001",
            )
            with self.assertRaises(MaterialDamageValidationError):
                validate_material_damage_sidecar(
                    destination,
                    source,
                    version_envelope=(("mapping_version", "wrong"),),
                    source_cohort_identity="cohort:test-001",
                )
            with self.assertRaises(MaterialDamageValidationError):
                validate_material_damage_sidecar(
                    destination,
                    source,
                    source_cohort_identity=SOURCE_COHORT_IDENTITY,
                )


if __name__ == "__main__":
    unittest.main()
