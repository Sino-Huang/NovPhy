from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts.physics_capture_contract import EVENT_SIDECAR, STATE_SIDECAR, load_physics_capture
from scripts.physics_relational_supervision import RelationalAvailability
import scripts.physics_violation_labels as violation_labels
from scripts.physics_violation_labels import (
    EXCESS_PENETRATION_LABEL,
    ILLEGAL_CONTACT_LABEL,
    PHYSICAL_VIOLATION_SIDECAR,
    UNSUPPORTED_STATIONARY_BODY_LABEL,
    PhysicalViolationError,
    PhysicalViolationLabel,
    PhysicalViolationLabels,
    derive_excess_penetration,
    derive_physical_violation_labels,
    derive_unsupported_stationary_body,
    read_physical_violation_labels,
    validate_physical_violation_labels,
    write_physical_violation_labels,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/physics_capture_v1"
ENGINE_EVIDENCE = FIXTURE / "physics_violation_engine_evidence_v1.csharp.jsonl"
STATE_DIGEST = hashlib.sha256((FIXTURE / STATE_SIDECAR).read_bytes()).hexdigest()
EVENT_DIGEST = hashlib.sha256((FIXTURE / EVENT_SIDECAR).read_bytes()).hexdigest()
NO_PLAN = RelationalAvailability.UNAVAILABLE_NO_DECLARED_PHYSICAL_REGIME_DERIVATION


def _capture(evidence_path: Path | None = None):
    return load_physics_capture(
        FIXTURE / STATE_SIDECAR,
        FIXTURE / EVENT_SIDECAR,
        evidence_path,
    )


def _derive(capture=None):
    return derive_physical_violation_labels(
        capture or _capture(),
        state_sha256=STATE_DIGEST,
        events_sha256=EVENT_DIGEST,
    )


def _assert_fail_closed(artifact) -> None:
    assert artifact.labels
    assert all(label.value is None for label in artifact.labels)
    assert all(label.availability is NO_PLAN for label in artifact.labels)
    assert all(label.evidence == () for label in artifact.labels)


def test_former_caller_binding_and_evidence_routes_are_removed() -> None:
    capture = _capture()

    assert not hasattr(violation_labels, "ActiveViolationBinding")
    assert not hasattr(violation_labels, "PhysicalViolationEvidence")
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        derive_physical_violation_labels(
            capture,
            state_sha256=STATE_DIGEST,
            events_sha256=EVENT_DIGEST,
            binding=object(),  # type: ignore[call-arg]
            evidence={"geometry_complete": True, "gravity_applies": True},  # type: ignore[call-arg]
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        derive_excess_penetration(capture, evidence={"complete": True})  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        derive_unsupported_stationary_body(
            capture,
            "201:0",
            binding={"geometric_tolerance": 1.0},  # type: ignore[call-arg]
        )
    with pytest.raises(ValueError, match="without an accepted level/pilot plan"):
        PhysicalViolationLabel(
            EXCESS_PENETRATION_LABEL,
            None,
            True,
            RelationalAvailability.AVAILABLE,
        )


def test_aggregate_rejects_duck_typed_and_subclass_label_serializers() -> None:
    class AvailableDuck:
        label_name = EXCESS_PENETRATION_LABEL
        entity_id = None
        value = True
        availability = RelationalAvailability.AVAILABLE
        evidence = ()

        def to_json(self):
            return {
                "record_type": "violation_label",
                "label_name": self.label_name,
                "entity_id": self.entity_id,
                "value": self.value,
                "availability": self.availability.value,
                "evidence": [],
            }

    class AvailableSubclass(PhysicalViolationLabel):
        def to_json(self):
            record = super().to_json()
            record["value"] = True
            record["availability"] = RelationalAvailability.AVAILABLE.value
            return record

    labels = (
        AvailableDuck(),
        AvailableSubclass(EXCESS_PENETRATION_LABEL, None, None, NO_PLAN),
    )
    for label in labels:
        with pytest.raises(ValueError, match="exact PhysicalViolationLabel"):
            PhysicalViolationLabels(
                "capture",
                "shot",
                STATE_DIGEST,
                EVENT_DIGEST,
                (label,),  # type: ignore[arg-type]
            )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("availability", RelationalAvailability.AVAILABLE, "accepted level/pilot plan"),
        ("value", False, "null value"),
        ("evidence", (object(),), "empty evidence"),
    ),
)
def test_aggregate_reasserts_fail_closed_member_state(
    field: str,
    value: object,
    message: str,
) -> None:
    label = PhysicalViolationLabel(EXCESS_PENETRATION_LABEL, None, None, NO_PLAN)
    object.__setattr__(label, field, value)

    with pytest.raises(ValueError, match=message):
        PhysicalViolationLabels(
            "capture",
            "shot",
            STATE_DIGEST,
            EVENT_DIGEST,
            (label,),
        )


@pytest.mark.parametrize(
    ("label_name", "entity_id", "message"),
    (
        (ILLEGAL_CONTACT_LABEL, None, "illegal_contact"),
        ("unknown_violation", None, "unsupported physical-violation"),
        (EXCESS_PENETRATION_LABEL, "201:0", "capture-scoped"),
        (UNSUPPORTED_STATIONARY_BODY_LABEL, None, "unsupported-body entity_id"),
    ),
)
def test_aggregate_runs_complete_base_label_invariants_nonpolymorphically(
    label_name: str,
    entity_id: str | None,
    message: str,
) -> None:
    label = PhysicalViolationLabel(EXCESS_PENETRATION_LABEL, None, None, NO_PLAN)
    object.__setattr__(label, "label_name", label_name)
    object.__setattr__(label, "entity_id", entity_id)

    with pytest.raises(ValueError, match=message):
        PhysicalViolationLabels(
            "capture",
            "shot",
            STATE_DIGEST,
            EVENT_DIGEST,
            (label,),
        )


def test_member_mutation_after_aggregate_construction_is_rejected() -> None:
    artifact = _derive()
    member = artifact.labels[-1]
    assert member.label_name == UNSUPPORTED_STATIONARY_BODY_LABEL
    object.__setattr__(member, "entity_id", "zzz:0")

    with pytest.raises(ValueError, match="members changed after construction"):
        artifact.to_bytes()


def test_aggregate_label_replacement_after_construction_is_rejected() -> None:
    artifact = _derive()
    object.__setattr__(artifact, "labels", tuple(list(artifact.labels)))

    with pytest.raises(ValueError, match="replaced after construction"):
        artifact.to_bytes()


def test_absent_incomplete_and_complete_engine_evidence_remain_unavailable(
    tmp_path: Path,
) -> None:
    complete = _capture(ENGINE_EVIDENCE)
    incomplete_record = json.loads(ENGINE_EVIDENCE.read_text(encoding="ascii"))
    incomplete_record["fixed_step_coverage"]["complete"] = False
    incomplete_record["fixed_step_coverage"]["incomplete_reason"] = "fixed_step_gap"
    incomplete_record["terminal_trace"]["failure_reason"] = "fixed_step_gap"
    incomplete_path = tmp_path / "physics_violation_engine_evidence_v1.jsonl"
    incomplete_path.write_text(
        json.dumps(incomplete_record, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    incomplete = _capture(incomplete_path)
    absent = _capture()

    assert absent.violation_evidence == ()
    assert complete.violation_evidence[0].coverage.complete is True
    assert incomplete.violation_evidence[0].coverage.complete is False
    for capture in (absent, incomplete, complete):
        _assert_fail_closed(_derive(capture))


def test_closed_vocabulary_has_one_label_per_present_dynamic_entity() -> None:
    capture = _capture(ENGINE_EVIDENCE)
    artifact = _derive(capture)
    expected_entities = sorted({
        str(node.entity_id)
        for state in capture.states
        for node in state.nodes
        if node.body.present
    })

    penetration = [label for label in artifact.labels if label.label_name == EXCESS_PENETRATION_LABEL]
    unsupported = [
        label for label in artifact.labels
        if label.label_name == UNSUPPORTED_STATIONARY_BODY_LABEL
    ]
    assert len(penetration) == 1
    assert [label.entity_id for label in unsupported] == expected_entities
    assert len(artifact.labels) == 1 + len(expected_entities)
    _assert_fail_closed(artifact)


def test_illegal_contact_and_available_records_remain_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="illegal_contact"):
        PhysicalViolationLabel(ILLEGAL_CONTACT_LABEL, None, None, NO_PLAN)

    artifact = _derive()
    records = [json.loads(line) for line in artifact.to_jsonl().splitlines()]
    records[1]["label_name"] = ILLEGAL_CONTACT_LABEL
    path = tmp_path / PHYSICAL_VIOLATION_SIDECAR
    path.write_text(
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
        encoding="ascii",
    )
    with pytest.raises(PhysicalViolationError, match="illegal_contact"):
        read_physical_violation_labels(path)

    records[1]["label_name"] = EXCESS_PENETRATION_LABEL
    records[1]["value"] = False
    records[1]["availability"] = RelationalAvailability.AVAILABLE.value
    path.write_text(
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
        encoding="ascii",
    )
    with pytest.raises(PhysicalViolationError, match="without an accepted level/pilot plan"):
        read_physical_violation_labels(path)

    records[1]["value"] = None
    records[1]["availability"] = NO_PLAN.value
    records[1]["evidence"] = [{"availability": RelationalAvailability.AVAILABLE.value}]
    path.write_text(
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
        encoding="ascii",
    )
    with pytest.raises(PhysicalViolationError, match="require empty evidence"):
        read_physical_violation_labels(path)


def test_serialization_round_trips_and_source_tampering_is_rejected(tmp_path: Path) -> None:
    shot = tmp_path / "shot"
    shot.mkdir()
    shutil.copy2(FIXTURE / STATE_SIDECAR, shot / STATE_SIDECAR)
    shutil.copy2(FIXTURE / EVENT_SIDECAR, shot / EVENT_SIDECAR)

    path = write_physical_violation_labels(shot)
    original = path.read_bytes()
    stored = read_physical_violation_labels(path)
    assert stored.to_bytes() == original
    assert validate_physical_violation_labels(shot).to_bytes() == original
    assert stored.state_sha256 == STATE_DIGEST
    assert stored.events_sha256 == EVENT_DIGEST
    header = json.loads(original.splitlines()[0])
    assert "binding" not in header
    assert "evidence" not in header

    records = [json.loads(line) for line in original.decode("ascii").splitlines()]
    records[0]["sources"]["physics_state_sha256"] = "0" * 64
    path.write_text(
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
        encoding="ascii",
    )
    with pytest.raises(PhysicalViolationError, match="source sidecar digests are stale"):
        validate_physical_violation_labels(shot)


def test_nonfinite_capture_values_still_reject() -> None:
    capture = _capture()
    first = capture.states[0]
    bad_contact = replace(first.raw_contacts[0], separation=float("inf"))
    capture = replace(
        capture,
        states=(replace(first, raw_contacts=(bad_contact, *first.raw_contacts[1:])), *capture.states[1:]),
    )

    with pytest.raises(PhysicalViolationError, match="must be finite"):
        _derive(capture)
