from __future__ import annotations

import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from io import BytesIO

from PIL import Image

from scripts.physics_capture_contract import PhysicsContractError, load_physics_capture
from scripts.physics_rollout_contract import CaptureProvenance
from scripts.physics_rollout_contract import MAX_TOTAL_BYTES
from scripts.physics_rollout_persistence import persist_physics_rollout
from src.webui.bridge import PhysicsCaptureV1
from src.webui.bridge import _decode_physics_capture_v1, encode_physics_capture_v1


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/physics_capture_v1"
PNG = b"\x89PNG\r\n\x1a\nfixture"


def evidence(*, complete: bool = True, reason: str | None = None) -> dict[str, object]:
    def entity(step: int, support: bool) -> dict[str, object]:
        edges: list[dict[str, object]] = []
        if support:
            edges.append({
                "support_id": "support:201:0->101:0",
                "supporter_id": "201:0",
                "evidence_contact_ids": [
                    "contact:10:101:0:1|201:0:2:0",
                    "contact:11:101:0:1|201:0:2:0",
                ],
                "evidence_fixed_steps": [10, 11],
            })
        return {
            "entity_id": "101:0",
            "observed": True,
            "present": True,
            "world_position": {"x": 1.0, "y": 2.0},
            "body_type": "dynamic",
            "simulated": True,
            "gravity_scale": 0.75,
            "support_v1": {"present": support, "edges": edges},
        }

    samples = [
        {"fixed_step": 10, "physics2d_gravity": {"x": 0.0, "y": -9.81}, "entities": [entity(10, False)]},
        {"fixed_step": 11, "physics2d_gravity": {"x": 0.0, "y": -9.81}, "entities": [entity(11, True)]},
    ]
    return {
        "schema_version": "physics_violation_engine_evidence_v1",
        "capture_id": "capture-golden-001",
        "shot_id": "engine-shot-golden",
        "sequence": 2,
        "fixed_step_coverage": {
            "first_fixed_step": 1,
            "last_fixed_step": 11,
            "sample_count": 11,
            "complete": complete,
            "incomplete_reason": reason,
        },
        "minimum_contact_separation": {
            "observed": True,
            "separation": -2.5,
            "contact_id": "contact:1:101:0:1|201:0:2:0",
            "fixed_step": 1,
        },
        "terminal_trace": {
            "max_fixed_steps": 8,
            "max_entities_per_step": 128,
            "first_fixed_step": 10,
            "last_fixed_step": 11,
            "truncated": True,
            "truncation_reason": "terminal_trace_bound",
            "failure_reason": reason,
            "samples": samples,
        },
    }


class PhysicsViolationEngineEvidenceTests(unittest.TestCase):
    def _sidecars(self, record: dict[str, object] | None) -> tuple[Path, Path, Path | None, tempfile.TemporaryDirectory[str]]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        state = root / "physics_state.jsonl"
        events = root / "physics_events.jsonl"
        shutil.copy2(FIXTURE / state.name, state)
        shutil.copy2(FIXTURE / events.name, events)
        evidence_path = None
        if record is not None:
            evidence_path = root / "physics_violation_engine_evidence_v1.jsonl"
            evidence_path.write_text(json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8")
        return state, events, evidence_path, temporary

    def test_closed_engine_evidence_parses_gravity_support_and_consecutive_terminal_trace(self) -> None:
        state, events, evidence_path, temporary = self._sidecars(evidence())
        with temporary:
            capture = load_physics_capture(state, events, evidence_path)

        self.assertEqual(len(capture.violation_evidence), 1)
        parsed = capture.violation_evidence[0]
        self.assertEqual(parsed.minimum_contact_separation.separation, -2.5)
        self.assertEqual([sample.fixed_step for sample in parsed.terminal_trace.samples], [10, 11])
        self.assertEqual(parsed.terminal_trace.samples[-1].physics2d_gravity.y, -9.81)
        self.assertTrue(parsed.terminal_trace.samples[-1].entities[0].support_v1.present)
        self.assertEqual(parsed.terminal_trace.samples[-1].entities[0].gravity_scale, 0.75)

    def test_exact_csharp_serialized_golden_is_consumed_by_python_parser(self) -> None:
        capture = load_physics_capture(
            FIXTURE / "physics_state.jsonl",
            FIXTURE / "physics_events.jsonl",
            FIXTURE / "physics_violation_engine_evidence_v1.csharp.jsonl",
        )

        self.assertEqual(len(capture.violation_evidence), 1)
        parsed = capture.violation_evidence[0]
        self.assertEqual(parsed.shot_id, "engine-shot-golden")
        self.assertEqual(parsed.terminal_trace.samples[0].fixed_step, 10)
        self.assertEqual(parsed.terminal_trace.samples[0].physics2d_gravity.x, 1.25)
        self.assertFalse(parsed.minimum_contact_separation.observed)

    def test_overflow_is_explicitly_incomplete(self) -> None:
        record = evidence(complete=False, reason="contact_sample_overflow")
        state, events, evidence_path, temporary = self._sidecars(record)
        with temporary:
            parsed = load_physics_capture(state, events, evidence_path).violation_evidence[0]
        self.assertFalse(parsed.coverage.complete)
        self.assertEqual(parsed.coverage.incomplete_reason, "contact_sample_overflow")
        self.assertEqual(parsed.terminal_trace.failure_reason, parsed.coverage.incomplete_reason)

    def test_fixed_step_gap_accepts_only_the_consecutive_terminal_suffix(self) -> None:
        record = evidence(complete=False, reason="fixed_step_gap")
        coverage = record["fixed_step_coverage"]
        assert isinstance(coverage, dict)
        coverage.update({
            "first_fixed_step": 1,
            "last_fixed_step": 3,
            "sample_count": 2,
        })
        trace = record["terminal_trace"]
        assert isinstance(trace, dict)
        samples = trace["samples"]
        assert isinstance(samples, list)
        suffix = samples[-1]
        assert isinstance(suffix, dict)
        suffix["fixed_step"] = 3
        entities = suffix["entities"]
        assert isinstance(entities, list)
        entity = entities[0]
        assert isinstance(entity, dict)
        entity["support_v1"] = {"present": False, "edges": []}
        trace.update({
            "first_fixed_step": 3,
            "last_fixed_step": 3,
            "samples": [suffix],
        })
        state, events, evidence_path, temporary = self._sidecars(record)
        with temporary:
            parsed = load_physics_capture(state, events, evidence_path).violation_evidence[0]

        self.assertEqual([sample.fixed_step for sample in parsed.terminal_trace.samples], [3])
        self.assertEqual(parsed.coverage.incomplete_reason, "fixed_step_gap")

    def test_unknown_evidence_field_is_rejected(self) -> None:
        record = evidence()
        record["caller_complete"] = True
        state, events, evidence_path, temporary = self._sidecars(record)
        with temporary, self.assertRaises(PhysicsContractError):
            load_physics_capture(state, events, evidence_path)

    def test_declared_evidence_sidecar_must_not_be_empty(self) -> None:
        state, events, evidence_path, temporary = self._sidecars(evidence())
        assert evidence_path is not None
        evidence_path.write_bytes(b"")
        with temporary, self.assertRaises(PhysicsContractError):
            load_physics_capture(state, events, evidence_path)

    def test_empty_coverage_cannot_claim_complete(self) -> None:
        record = evidence()
        record["fixed_step_coverage"] = {
            "first_fixed_step": None,
            "last_fixed_step": None,
            "sample_count": 0,
            "complete": True,
            "incomplete_reason": None,
        }
        state, events, evidence_path, temporary = self._sidecars(record)
        with temporary, self.assertRaises(PhysicsContractError):
            load_physics_capture(state, events, evidence_path)

    def test_duplicate_terminal_entity_is_rejected(self) -> None:
        record = evidence()
        terminal_trace = record["terminal_trace"]
        assert isinstance(terminal_trace, dict)
        samples = terminal_trace["samples"]
        assert isinstance(samples, list)
        sample = samples[-1]
        assert isinstance(sample, dict)
        entities = sample["entities"]
        assert isinstance(entities, list)
        entities.append(dict(entities[0]))
        state, events, evidence_path, temporary = self._sidecars(record)
        with temporary, self.assertRaises(PhysicsContractError):
            load_physics_capture(state, events, evidence_path)

    def test_evidence_has_an_independent_byte_budget_from_retained_sidecars(self) -> None:
        state, events, evidence_path, temporary = self._sidecars(evidence())
        assert evidence_path is not None
        encoded = json.dumps(evidence(), separators=(",", ":")).encode("utf-8")
        evidence_path.write_bytes(
            encoded + b" " * (MAX_TOTAL_BYTES - len(encoded) - 1) + b"\n"
        )
        with temporary:
            capture = load_physics_capture(state, events, evidence_path)
        self.assertEqual(len(capture.violation_evidence), 1)

    def test_contact_identifier_step_must_match_declared_evidence_step(self) -> None:
        record = evidence()
        minimum = record["minimum_contact_separation"]
        assert isinstance(minimum, dict)
        minimum["fixed_step"] = 2
        state, events, evidence_path, temporary = self._sidecars(record)
        with temporary, self.assertRaises(PhysicsContractError):
            load_physics_capture(state, events, evidence_path)

    def test_current_csharp_shaped_four_component_envelope_reaches_python_parser(self) -> None:
        state_records = [json.loads(line) for line in (FIXTURE / "physics_state.jsonl").read_text().splitlines()]
        state = dict(state_records[2])
        state.pop("record_type")
        state.pop("shot_id")
        events: list[dict[str, object]] = []
        evidence_record = evidence()
        # C# BuildSuccessEnvelope writes UTF-8 in this order with flag 2 and four
        # big-endian lengths. Keep this independent of the Python fixture encoder.
        components = [
            PNG,
            json.dumps(state, separators=(",", ":")).encode(),
            json.dumps(events, separators=(",", ":")).encode(),
            json.dumps(evidence_record, separators=(",", ":")).encode(),
        ]
        payload = struct.pack("!IIII", *(len(component) for component in components)) + b"".join(components)
        body = struct.pack("!4sBBHI", b"SBPV", 1, 2, 0, len(payload)) + payload

        packet = _decode_physics_capture_v1(body)

        self.assertIsNotNone(packet.evidence)
        self.assertEqual(packet.evidence["schema_version"], "physics_violation_engine_evidence_v1")
        self.assertEqual(packet.evidence["terminal_trace"]["samples"][-1]["entities"][0]["gravity_scale"], 0.75)

    def test_legacy_three_component_packet_exposes_no_evidence(self) -> None:
        packet_bytes = encode_physics_capture_v1(
            PNG,
            {"schema_version": "physics_capture_v1", "render_frame": 4},
            [],
        )
        packet = _decode_physics_capture_v1(packet_bytes[4:])
        self.assertIsNone(packet.evidence)

        capture = load_physics_capture(FIXTURE / "physics_state.jsonl", FIXTURE / "physics_events.jsonl")
        self.assertEqual(capture.violation_evidence, ())

    def test_persistence_installs_engine_evidence_as_its_own_atomic_sidecar(self) -> None:
        state = json.loads((FIXTURE / "physics_state.jsonl").read_text().splitlines()[2])
        state["support_edges"] = []
        stream = BytesIO()
        Image.new("RGB", (2, 1), (1, 2, 3)).save(stream, format="PNG")
        packet = PhysicsCaptureV1(stream.getvalue(), state, (), evidence())

        class Bridge:
            def get_physics_capture_v1(self) -> PhysicsCaptureV1:
                return packet

        with tempfile.TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot_001.tmp"
            metadata = persist_physics_rollout(
                Bridge(), shot,
                target_fps=1.0,
                duration_seconds=1.0,
                max_frames=1,
                state_header=None,
                provenance=CaptureProvenance("a" * 64, "b" * 64, "c" * 64),
                clock=lambda: 0.0,
                sleeper=lambda _: None,
            )
            sidecar = shot / "physics_violation_engine_evidence_v1.jsonl"
            persisted = json.loads(sidecar.read_text())
            loaded = load_physics_capture(
                shot / "physics_state.jsonl", shot / "physics_events.jsonl", sidecar
            )

        self.assertEqual(persisted, evidence())
        self.assertEqual(metadata["physics_violation_engine_evidence_count"], 1)
        self.assertEqual(len(loaded.violation_evidence), 1)


if __name__ == "__main__":
    unittest.main()
