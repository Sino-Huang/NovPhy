from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.physics_capture_contract import EVENT_SIDECAR, STATE_SIDECAR, load_physics_capture
from scripts.physics_relational_supervision import RelationalAvailability
from scripts.physics_violation_labels import (
    EXCESS_PENETRATION_LABEL,
    ILLEGAL_CONTACT_LABEL,
    PHYSICAL_VIOLATION_SIDECAR,
    UNSUPPORTED_STATIONARY_BODY_LABEL,
    PhysicalViolationError,
    PhysicalViolationLabel,
    PhysicalViolationLabels,
    derive_physical_violation_labels,
    read_physical_violation_labels,
    validate_physical_violation_labels,
    write_physical_violation_labels,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/physics_capture_v1"
ENGINE_EVIDENCE = FIXTURE / "physics_violation_engine_evidence_v1.csharp.jsonl"
NO_PLAN = RelationalAvailability.UNAVAILABLE_NO_DECLARED_PHYSICAL_REGIME_DERIVATION


class PhysicalViolationLabelTests(unittest.TestCase):
    def capture(self, evidence_path: Path | None = None):
        return load_physics_capture(FIXTURE / STATE_SIDECAR, FIXTURE / EVENT_SIDECAR, evidence_path)

    def derive(self, capture=None):
        return derive_physical_violation_labels(capture or self.capture())

    def assert_fail_closed(self, artifact: PhysicalViolationLabels) -> None:
        self.assertTrue(artifact.labels)
        self.assertTrue(all(label.value is None for label in artifact.labels))
        self.assertTrue(all(label.availability is NO_PLAN for label in artifact.labels))
        self.assertTrue(all(label.evidence == () for label in artifact.labels))

    def test_closed_vocabulary_has_one_label_per_present_dynamic_entity(self) -> None:
        capture = self.capture(ENGINE_EVIDENCE)
        artifact = self.derive(capture)
        expected_entities = sorted({
            str(node.entity_id)
            for state in capture.states
            for node in state.nodes
            if node.body.present
        })
        penetration = [label for label in artifact.labels if label.label_name == EXCESS_PENETRATION_LABEL]
        unsupported = [label for label in artifact.labels if label.label_name == UNSUPPORTED_STATIONARY_BODY_LABEL]
        self.assertEqual(len(penetration), 1)
        self.assertEqual([label.entity_id for label in unsupported], expected_entities)
        self.assert_fail_closed(artifact)

    def test_available_and_illegal_labels_remain_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "illegal_contact"):
            PhysicalViolationLabel(ILLEGAL_CONTACT_LABEL, None, None, NO_PLAN)
        with self.assertRaisesRegex(ValueError, "accepted level/pilot plan"):
            PhysicalViolationLabel(EXCESS_PENETRATION_LABEL, None, True, RelationalAvailability.AVAILABLE)

    def test_aggregate_rejects_nonexact_label_instances(self) -> None:
        class Subclass(PhysicalViolationLabel):
            pass

        label = Subclass(EXCESS_PENETRATION_LABEL, None, None, NO_PLAN)
        with self.assertRaisesRegex(ValueError, "exact PhysicalViolationLabel"):
            PhysicalViolationLabels("capture", "shot", (label,))

    def test_member_mutation_after_construction_is_rejected(self) -> None:
        artifact = self.derive()
        object.__setattr__(artifact.labels[-1], "entity_id", "changed:0")
        with self.assertRaisesRegex(ValueError, "members changed after construction"):
            artifact.to_bytes()

    def test_engine_evidence_does_not_make_labels_available(self) -> None:
        for capture in (self.capture(), self.capture(ENGINE_EVIDENCE)):
            self.assert_fail_closed(self.derive(capture))

    def test_serialization_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            shot.mkdir()
            shutil.copy2(FIXTURE / STATE_SIDECAR, shot / STATE_SIDECAR)
            shutil.copy2(FIXTURE / EVENT_SIDECAR, shot / EVENT_SIDECAR)
            path = write_physical_violation_labels(shot)
            original = path.read_bytes()
            stored = read_physical_violation_labels(path)
            self.assertEqual(stored.to_bytes(), original)
            self.assertEqual(validate_physical_violation_labels(shot).to_bytes(), original)
            header = json.loads(original.splitlines()[0])
            self.assertEqual(
                header["sources"],
                {"physics_state_path": STATE_SIDECAR, "physics_events_path": EVENT_SIDECAR},
            )

    def test_malformed_available_record_is_rejected(self) -> None:
        artifact = self.derive()
        records = [json.loads(line) for line in artifact.to_jsonl().splitlines()]
        records[1]["value"] = False
        records[1]["availability"] = RelationalAvailability.AVAILABLE.value
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / PHYSICAL_VIOLATION_SIDECAR
            path.write_text(
                "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
                encoding="ascii",
            )
            with self.assertRaisesRegex(PhysicalViolationError, "accepted level/pilot plan"):
                read_physical_violation_labels(path)

    def test_nonfinite_capture_values_still_reject(self) -> None:
        capture = self.capture()
        first = capture.states[0]
        bad_contact = replace(first.raw_contacts[0], separation=float("inf"))
        changed = replace(
            capture,
            states=(replace(first, raw_contacts=(bad_contact, *first.raw_contacts[1:])), *capture.states[1:]),
        )
        with self.assertRaisesRegex(PhysicalViolationError, "must be finite"):
            self.derive(changed)


if __name__ == "__main__":
    unittest.main()
