from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from io import BytesIO
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from PIL import Image

from scripts.physics_artifact_validation import validate_physics_shot_artifact
from scripts.physics_capture_contract import load_physics_capture
from scripts.physics_rollout_contract import CaptureProvenance, PhysicsPersistenceError
from scripts.physics_rollout_persistence import persist_physics_rollout
from scripts.physics_rollout_semantics import (
    PhysicsRolloutSemanticsError,
    validate_physics_rollout_semantics,
)
from scripts.rollout_validation_types import PhysicsArtifactError


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "physics_capture_v1"
PROVENANCE = CaptureProvenance("0" * 64, "1" * 64, "2" * 64)


def _png() -> bytes:
    image = Image.new("RGB", (2, 1), (12, 34, 56))
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    return encoded.getvalue()


def _fixture_records(name: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in (FIXTURE / name).read_text(encoding="utf-8").splitlines()]


def _packet(state: dict[str, object], events: list[dict[str, object]]) -> FakePacket:
    return FakePacket(_png(), deepcopy(state), tuple(deepcopy(events)))


@dataclass(frozen=True)
class FakePacket:
    png: bytes
    state: dict[str, object]
    events: tuple[dict[str, object], ...]


class FakeBridge:
    def __init__(self, packets: list[FakePacket], calls: list[str]) -> None:
        self._packets = iter(packets)
        self._calls = calls

    def get_physics_capture_v1(self) -> FakePacket:
        self._calls.append("request-70")
        return next(self._packets)


class PhysicsRolloutSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp(prefix="novphy-rollout-semantics-"))
        self.addCleanup(shutil.rmtree, self.temporary, ignore_errors=True)
        self.states = _fixture_records("physics_state.jsonl")[1:]
        self.events = _fixture_records("physics_events.jsonl")

    def _persist(
        self,
        *,
        events: list[dict[str, object]] | None = None,
        expected_initial_engine_state_identity: str | None = None,
        scenario_context: dict[str, object] | None = None,
    ) -> tuple[Path, dict[str, object], list[str]]:
        calls: list[str] = []
        initial = _packet(self.states[0], [])
        bridge = FakeBridge([_packet(self.states[1], self.events[:1] if events is None else events)], calls)
        shot_dir = self.temporary / "shot_001"

        def shoot() -> None:
            calls.append("shoot")

        metadata = persist_physics_rollout(
            bridge,
            shot_dir,
            target_fps=30.0,
            duration_seconds=1.0,
            max_frames=2,
            state_header=None,
            provenance=PROVENANCE,
            initial_capture=initial,
            shoot=shoot,
            expected_initial_engine_state_identity=expected_initial_engine_state_identity,
            scenario_context=scenario_context,
            clock=lambda: 0.0,
            sleeper=lambda _: None,
        )
        return shot_dir, metadata, calls

    def _reseal(self, shot_dir: Path) -> None:
        metadata_path = shot_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        state_bytes = (shot_dir / "physics_state.jsonl").read_bytes()
        event_bytes = (shot_dir / "physics_events.jsonl").read_bytes()
        metadata["physics_state_count"] = len(state_bytes.splitlines()) - 1
        metadata["physics_event_count"] = len(event_bytes.splitlines())
        metadata["frame_count"] = metadata["physics_state_count"]
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _replace_events(self, shot_dir: Path, events: list[dict[str, object]]) -> None:
        path = shot_dir / "physics_events.jsonl"
        path.write_text(
            "".join(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n" for event in events),
            encoding="utf-8",
        )
        self._reseal(shot_dir)

    def test_initial_capture_precedes_one_shoot_and_request_capture(self) -> None:
        shot_dir, metadata, calls = self._persist(scenario_context={"level": "fixture"})

        capture = load_physics_capture(shot_dir / "physics_state.jsonl", shot_dir / "physics_events.jsonl")
        self.assertEqual(calls, ["shoot", "request-70"])
        self.assertEqual([state.clock.fixed_step for state in capture.states], [10, 11])
        self.assertEqual(metadata["physics_state_count"], 2)
        self.assertEqual(metadata["scenario_context"], {"level": "fixture"})
        validate_physics_shot_artifact(shot_dir)

    def test_initial_capture_requires_two_retained_states_and_has_no_events(self) -> None:
        initial = _packet(self.states[0], [])
        bridge = FakeBridge([], [])
        with self.assertRaises(PhysicsPersistenceError):
            persist_physics_rollout(
                bridge,
                self.temporary / "shot_002",
                target_fps=30.0,
                duration_seconds=1.0,
                max_frames=1,
                state_header=None,
                provenance=PROVENANCE,
                initial_capture=initial,
                shoot=lambda: None,
            )

        with self.assertRaises(PhysicsPersistenceError):
            persist_physics_rollout(
                bridge,
                self.temporary / "shot_003",
                target_fps=30.0,
                duration_seconds=1.0,
                max_frames=2,
                state_header=None,
                provenance=PROVENANCE,
                initial_capture=_packet(self.states[0], self.events[:1]),
                shoot=lambda: None,
            )

    def test_validation_rejects_missing_or_duplicate_launch(self) -> None:
        shot_dir, _, _ = self._persist()
        capture = load_physics_capture(shot_dir / "physics_state.jsonl", shot_dir / "physics_events.jsonl")
        duplicate = replace(capture.events[0], event_id="event:00000001", clock=replace(capture.events[0].clock, sequence=1))
        with self.assertRaisesRegex(PhysicsRolloutSemanticsError, "bird_launched"):
            validate_physics_rollout_semantics(replace(capture, events=(capture.events[0], duplicate)))

        self._replace_events(shot_dir, [])
        with self.assertRaisesRegex(PhysicsArtifactError, "bird_launched"):
            validate_physics_shot_artifact(shot_dir)

    def test_validation_rejects_collision_without_retained_contact_evidence(self) -> None:
        shot_dir, _, _ = self._persist(events=self.events[:2])
        collision = deepcopy(self.events[1])
        collision["payload"]["contact_ids"] = ["contact:11:101:0|1101:201:0|1201:99"]
        self._replace_events(shot_dir, [self.events[0], collision])

        with self.assertRaisesRegex(PhysicsArtifactError, "collision"):
            validate_physics_shot_artifact(shot_dir)

    def test_validation_rejects_terminal_event_after_final_state(self) -> None:
        shot_dir, _, _ = self._persist()
        terminal = deepcopy(self.events[-1])
        terminal["sequence"] = 1
        terminal["event_type"] = "level_cleared"
        terminal["fixed_step"] = 12
        terminal["fixed_time"] = 0.24
        terminal["event_id"] = "event:00000001"
        terminal["payload"] = {"score": 1}
        self._replace_events(shot_dir, [self.events[0], terminal])

        with self.assertRaisesRegex(PhysicsArtifactError, "final state"):
            validate_physics_shot_artifact(shot_dir)

    def test_initial_engine_state_identity_excludes_clock_and_rgb_provenance(self) -> None:
        from scripts.physics_rollout_semantics import initial_engine_state_identity

        shot_dir, _, _ = self._persist()
        original = load_physics_capture(shot_dir / "physics_state.jsonl", shot_dir / "physics_events.jsonl")
        copied_state = self.temporary / "state-copy.jsonl"
        copied_events = self.temporary / "events-copy.jsonl"
        state_records = _fixture_records("physics_state.jsonl")[:3]
        for index, record in enumerate(state_records):
            record["capture_id"] = "other-capture"
            record["shot_id"] = "shot_999"
            record["sequence"] = 50 + index
            record["render_frame"] = 500 + index
            record["render_time"] = 50.0 + index
            record["fixed_step"] = 5000 + index
            record["fixed_time"] = 100.0 + index
            if record["record_type"] == "state":
                record["rgb_frame"]["render_frame"] = record["render_frame"]
                record["rgb_frame"]["relative_path"] = f"frames/other_{index}.png"
                for edge in record["support_edges"]:
                    edge["evidence_fixed_steps"] = [step + 4991 for step in edge["evidence_fixed_steps"]]
        copied_state.write_text("".join(json.dumps(record) + "\n" for record in state_records), encoding="utf-8")
        copied_events.write_text("", encoding="utf-8")
        changed = load_physics_capture(copied_state, copied_events)

        self.assertEqual(initial_engine_state_identity(original), initial_engine_state_identity(changed))

    def test_termination_metadata_records_rollout_ceiling(self) -> None:
        _, metadata, _ = self._persist()

        self.assertEqual(metadata["termination_reason"], "rollout_ceiling")
        self.assertEqual(metadata["termination_fixed_step"], 11)
        self.assertIsNone(metadata["termination_event_id"])
        self.assertEqual(metadata["terminal_state_fixed_step"], 11)
        self.assertEqual(metadata["intervention_event_id"], "event:00000000")
        self.assertTrue(metadata["initial_engine_state_identity"].startswith("normalized-initial-engine-state-v1:"))

    def test_launch_cannot_precede_first_retained_state_but_same_step_is_allowed(self) -> None:
        shot_dir, _, _ = self._persist()
        capture = load_physics_capture(shot_dir / "physics_state.jsonl", shot_dir / "physics_events.jsonl")
        first_state_step = capture.states[0].clock.fixed_step
        earlier = replace(
            capture.events[0],
            clock=replace(capture.events[0].clock, fixed_step=first_state_step - 1, fixed_time=0.18),
        )
        with self.assertRaisesRegex(PhysicsRolloutSemanticsError, "before the first retained state"):
            validate_physics_rollout_semantics(replace(capture, events=(earlier,)))

        same_step = replace(
            capture.events[0],
            clock=replace(capture.events[0].clock, fixed_step=first_state_step, fixed_time=0.2),
        )
        self.assertEqual(
            validate_physics_rollout_semantics(replace(capture, events=(same_step,)))["intervention_event_id"],
            str(same_step.event_id),
        )


if __name__ == "__main__":
    unittest.main()
