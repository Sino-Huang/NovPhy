from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from scripts.collect_rollouts import (
    capture_physics_rollout,
    collect_fresh_engine_rollouts,
    collect_rollouts,
    main,
)
from scripts.scenario_manifest import (
    BenchmarkCondition,
    DeclaredInitialEngineState,
    GenerationProvenance,
    LevelInstance,
    ResearchEligibility,
    ScenarioLineage,
    ScenarioManifest,
    ScenarioSpecification,
    TemplateEvidence,
    import_legacy_manifest,
    write_manifest,
)
from src.webui.bridge import GameState, PhysicsCaptureV1


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "physics_capture_v1"
PROVENANCE = {
    "physics_player_sha256": "a" * 64,
    "physics_protocol_sha256": "b" * 64,
    "physics_archive_sha256": "c" * 64,
}
ACTION = {"coordinate_frame": "absolute", "release": [130, 150], "tapTime": 70}
GUARD = {
    "pre_shot_image": None,
    "pre_shot_sample": None,
    "post_recovery_protocol_state": {},
    "recovery_action": None,
    "pre_shot_guard": {"status": "accepted", "invalid_reason": None},
}


def _records() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    states = [json.loads(line) for line in (FIXTURE / "physics_state.jsonl").read_text(encoding="utf-8").splitlines()]
    events = [json.loads(line) for line in (FIXTURE / "physics_events.jsonl").read_text(encoding="utf-8").splitlines()]
    return states, events


def _png() -> bytes:
    encoded = BytesIO()
    Image.new("RGB", (2, 1), (12, 34, 56)).save(encoded, format="PNG")
    return encoded.getvalue()


def _packets(*, different_initial: bool = False) -> tuple[PhysicsCaptureV1, PhysicsCaptureV1]:
    states, events = _records()
    initial_state = deepcopy(states[1])
    if different_initial:
        initial_state["nodes"][0]["life"] = 0.5
    return (
        PhysicsCaptureV1(_png(), initial_state, ()),
        PhysicsCaptureV1(_png(), deepcopy(states[2]), (deepcopy(events[0]),)),
    )


class GameplayBridge:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.disconnected = False

    def shoot(self, *_args, **_kwargs):
        self.calls.append("legacy-shoot")
        return 1

    def shoot_and_record_ground_truth(self, *_args, **_kwargs):
        self.calls.append("GT_SHOOT")
        return {"ground_truth_count": 1}

    def get_game_state(self):
        return GameState.PLAYING

    def get_current_level(self):
        return 7

    def get_current_score(self):
        return 42

    def configure(self, *_args):
        return (0, 0, 1)

    def set_speed(self, _speed):
        return 1

    def disconnect(self):
        self.disconnected = True


class PacketBridge:
    def __init__(self, packets: tuple[PhysicsCaptureV1, PhysicsCaptureV1], calls: list[str]) -> None:
        self._packets = iter(packets)
        self._calls = calls
        self.request_count = 0

    def get_physics_capture_v1(self) -> PhysicsCaptureV1:
        self.request_count += 1
        self._calls.append("initial-request-70" if self.request_count == 1 else "post-request-70")
        return next(self._packets)


class Process:
    next_pid = 9000

    def __init__(self) -> None:
        self.pid = Process.next_pid
        Process.next_pid += 1
        self.terminated = False
        self.waited = False

    def poll(self):
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None) -> None:
        self.waited = True


def _scenario_manifest() -> ScenarioManifest:
    return ScenarioManifest(
        schema="scenario_manifest_v1",
        benchmark_condition=BenchmarkCondition("novelty_level_1", "type010101"),
        scenario_template=TemplateEvidence("available", "template:identity"),
        generation=GenerationProvenance("generated", "generator", "1", 7, {}, {}),
        level_instance=LevelInstance("level:identity"),
        scenario_specification=ScenarioSpecification("spec:identity", "decl:identity", "content:identity"),
        scenario_lineage=ScenarioLineage("lineage:identity"),
        declared_initial_engine_state=DeclaredInitialEngineState(
            "declared_initial_engine_state_v1", "declared-xml-identity"
        ),
        research_eligibility=ResearchEligibility("research_eligible"),
    )


def _write_verified_scenario(root: Path) -> tuple[Path, Path, ScenarioManifest]:
    root.mkdir(parents=True, exist_ok=True)
    xml_path = root / "level.xml"
    xml_path.write_text("<Level><Bird type=\"red\" /></Level>", encoding="utf-8")
    manifest = import_legacy_manifest(
        xml_path.read_bytes(),
        benchmark_condition=BenchmarkCondition("novelty_level_1", "type010101"),
        source_path="level.xml",
    )
    manifest_path = write_manifest(manifest, root / "level.scenario.json")
    return manifest_path, xml_path, manifest


class SingleShotCollectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp(prefix="novphy-single-shot-"))
        self.addCleanup(shutil.rmtree, self.temporary, ignore_errors=True)

    def _fresh_kwargs(self, *, physics_bridges: list[PacketBridge], gameplay_bridges: list[GameplayBridge]) -> dict:
        processes: list[Process] = []

        def start_engine(*_args, **_kwargs):
            process = Process()
            processes.append(process)
            return process

        def connect(*_args, **_kwargs):
            return gameplay_bridges.pop(0)

        def physics_bridge(*_args, **_kwargs):
            return physics_bridges.pop(0)

        return {
            "game_dir": Path("game"),
            "host": "127.0.0.1",
            "port": 2004,
            "agent_id": 28888,
            "speed": 1,
            "connect_timeout": 1,
            "read_timeout": 1,
            "prepare_timeout": 1,
            "frame_height": 480,
            "fast": True,
            "headless": True,
            "target_fps": 2,
            "duration_seconds": 1,
            "ui_level": None,
            "ui_settle_seconds": 0,
            "start_engine_func": start_engine,
            "connect_func": connect,
            "prepare_func": lambda bridge, **_kwargs: bridge.get_game_state(),
            "anchor_actions": False,
            "physics_capture_v1": True,
            **PROVENANCE,
            "_processes": processes,
            "_physics_bridge_factory": physics_bridge,
        }

    def test_capture_physics_rollout_forwards_initial_shoot_response_and_context(self) -> None:
        calls: list[str] = []
        packets = _packets()
        bridge = PacketBridge((packets[1], packets[1]), calls)
        bridge.request_count = 1

        metadata = capture_physics_rollout(
            bridge,
            self.temporary / "shot_001.tmp",
            target_fps=2,
            duration_seconds=1,
            max_frames=2,
            player_sha256="a" * 64,
            protocol_sha256="b" * 64,
            archive_sha256="c" * 64,
            initial_capture=packets[0],
            shoot=lambda: {"ground_truth_count": 1},
            expected_initial_engine_state_identity=None,
            scenario_context={"scenario_manifest_schema": "scenario_manifest_v1"},
            clock=lambda: 0.0,
            sleeper=lambda _seconds: None,
        )

        self.assertEqual(calls, ["post-request-70"])
        self.assertEqual(metadata["shoot_response"], {"ground_truth_count": 1})
        self.assertEqual(metadata["scenario_context"], {"scenario_manifest_schema": "scenario_manifest_v1"})

    def test_physics_collection_orders_initial_request_shoot_and_post_capture(self) -> None:
        calls: list[str] = []
        gameplay = GameplayBridge(calls)
        physics = PacketBridge(_packets(), calls)
        with patch("scripts.collect_rollouts._run_pre_shot_guard", return_value=GUARD):
            manifest = collect_rollouts(
                gameplay,
                self.temporary,
                [ACTION],
                target_fps=2,
                duration_seconds=1,
                max_frames=2,
                anchor_actions=False,
                shoot_before_capture=False,
                physics_capture_v1=True,
                physics_bridge=physics,
                scenario_context={"scenario_lineage_identity": "lineage:identity"},
                **PROVENANCE,
            )

        metadata = json.loads((self.temporary / "shot_001" / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(calls, ["initial-request-70", "GT_SHOOT", "post-request-70"])
        self.assertEqual(metadata["shoot_response"], {"ground_truth_count": 1})
        self.assertEqual(manifest["rollouts"][0]["shoot_response"], {"ground_truth_count": 1})
        self.assertEqual(manifest["rollouts"][0]["initial_engine_state_identity"], metadata["initial_engine_state_identity"])
        self.assertEqual(manifest["rollouts"][0]["scenario_context"], {"scenario_lineage_identity": "lineage:identity"})

    def test_physics_resume_recollects_accepted_artifact_without_strict_semantics(self) -> None:
        first_calls: list[str] = []
        first_physics = PacketBridge(_packets(), first_calls)
        with patch("scripts.collect_rollouts._run_pre_shot_guard", return_value=GUARD):
            collect_rollouts(
                GameplayBridge(first_calls),
                self.temporary,
                [ACTION],
                target_fps=2,
                duration_seconds=1,
                max_frames=2,
                anchor_actions=False,
                physics_capture_v1=True,
                physics_bridge=first_physics,
                **PROVENANCE,
            )

        metadata_path = self.temporary / "shot_001" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for field in (
            "initial_engine_state_identity",
            "intervention_event_id",
            "termination_reason",
            "termination_fixed_step",
            "termination_event_id",
            "terminal_state_fixed_step",
        ):
            metadata.pop(field, None)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        second_calls: list[str] = []
        second_physics = PacketBridge(_packets(), second_calls)
        with patch("scripts.collect_rollouts._run_pre_shot_guard", return_value=GUARD):
            collect_rollouts(
                GameplayBridge(second_calls),
                self.temporary,
                [ACTION],
                target_fps=2,
                duration_seconds=1,
                max_frames=2,
                anchor_actions=False,
                physics_capture_v1=True,
                physics_bridge=second_physics,
                **PROVENANCE,
            )

        self.assertEqual(second_physics.request_count, 2)
        refreshed = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertTrue(refreshed["initial_engine_state_identity"])

    def test_physics_resume_requires_exact_requested_scenario_context(self) -> None:
        for replacement in ({"scenario_lineage_identity": "wrong"}, None):
            with self.subTest(replacement=replacement):
                root = self.temporary / ("wrong" if replacement else "missing")
                first_calls: list[str] = []
                first_physics = PacketBridge(_packets(), first_calls)
                requested_context = {"scenario_lineage_identity": "lineage:identity"}
                with patch("scripts.collect_rollouts._run_pre_shot_guard", return_value=GUARD):
                    collect_rollouts(
                        GameplayBridge(first_calls),
                        root,
                        [ACTION],
                        target_fps=2,
                        duration_seconds=1,
                        max_frames=2,
                        anchor_actions=False,
                        physics_capture_v1=True,
                        physics_bridge=first_physics,
                        scenario_context=requested_context,
                        **PROVENANCE,
                    )

                metadata_path = root / "shot_001" / "metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if replacement is None:
                    metadata.pop("scenario_context")
                else:
                    metadata["scenario_context"] = replacement
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

                second_calls: list[str] = []
                second_physics = PacketBridge(_packets(), second_calls)
                with patch("scripts.collect_rollouts._run_pre_shot_guard", return_value=GUARD):
                    collect_rollouts(
                        GameplayBridge(second_calls),
                        root,
                        [ACTION],
                        target_fps=2,
                        duration_seconds=1,
                        max_frames=2,
                        anchor_actions=False,
                        physics_capture_v1=True,
                        physics_bridge=second_physics,
                        scenario_context=requested_context,
                        **PROVENANCE,
                    )

                self.assertEqual(second_physics.request_count, 2)
                self.assertEqual(json.loads(metadata_path.read_text(encoding="utf-8"))["scenario_context"], requested_context)

    def test_fresh_engine_verifies_identical_initial_identity_and_projects_scenario_context(self) -> None:
        calls: list[str] = []
        physics_bridges = [PacketBridge(_packets(), calls), PacketBridge(_packets(), calls)]
        gameplay_bridges = [GameplayBridge(calls), GameplayBridge(calls)]
        kwargs = self._fresh_kwargs(physics_bridges=physics_bridges, gameplay_bridges=gameplay_bridges)
        physics_factory = kwargs.pop("_physics_bridge_factory")
        processes = kwargs.pop("_processes")
        scenario_manifest = _scenario_manifest()
        with (
            patch("scripts.collect_rollouts._run_pre_shot_guard", return_value=GUARD),
            patch("scripts.collect_rollouts.ScienceBirdsBridge", side_effect=physics_factory),
        ):
            manifest = collect_fresh_engine_rollouts(
                self.temporary,
                [ACTION, ACTION],
                fresh_engine_attempts=1,
                scenario_manifest=scenario_manifest,
                **kwargs,
            )

        identities = {rollout["initial_engine_state_identity"] for rollout in manifest["accepted_rollouts"]}
        expected_context = {
            "scenario_manifest_schema": "scenario_manifest_v1",
            "scenario_lineage_identity": "lineage:identity",
            "declared_initial_engine_state_identity": "declared-xml-identity",
        }
        self.assertEqual(len(identities), 1)
        self.assertEqual(manifest["initial_engine_state_identity"], identities.pop())
        self.assertTrue(manifest["initial_engine_state_verified"])
        self.assertEqual(manifest["scenario_context"], expected_context)
        self.assertNotEqual(manifest["initial_engine_state_identity"], "declared-xml-identity")
        self.assertEqual(
            json.loads((self.temporary / "shot_002" / "metadata.json").read_text(encoding="utf-8"))["scenario_context"],
            expected_context,
        )
        self.assertTrue(all(process.terminated and process.waited for process in processes))

    def test_fresh_engine_rollouts_reject_unplanned_multi_attempt_identity_retry(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen collection plans own retry decisions"):
            collect_fresh_engine_rollouts(
                self.temporary,
                [ACTION, ACTION],
                game_dir=Path("game"),
                host="127.0.0.1",
                port=2004,
                agent_id=28888,
                speed=1,
                connect_timeout=1,
                read_timeout=1,
                prepare_timeout=1,
                frame_height=480,
                fast=True,
                headless=True,
                target_fps=2,
                duration_seconds=1,
                ui_level=None,
                ui_settle_seconds=0,
                fresh_engine_attempts=2,
            )

    def test_non_physics_collection_keeps_pre_capture_shoot(self) -> None:
        calls: list[str] = []
        gameplay = GameplayBridge(calls)

        def capture(_bridge, output_dir, **_kwargs):
            calls.append("capture")
            metadata = {"frame_count": 0, "frames_dir": ""}
            (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            return metadata

        with patch("scripts.collect_rollouts._run_pre_shot_guard", return_value=GUARD):
            collect_rollouts(
                gameplay,
                self.temporary,
                [ACTION],
                target_fps=1,
                duration_seconds=1,
                anchor_actions=False,
                capture_rollout=capture,
            )

        self.assertEqual(calls, ["legacy-shoot", "capture"])

    def test_main_loads_verified_scenario_for_fresh_physics_collection(self) -> None:
        manifest_path, xml_path, expected = _write_verified_scenario(self.temporary)
        plan_path = self.temporary / "collection-plan.json"
        args = [
            "collect_rollouts.py",
            "--output-dir", str(self.temporary / "rollouts"),
            "--count", "1",
            "--physics-capture-v1",
            "--fresh-engine-per-rollout",
            "--collection-plan", str(plan_path),
            "--physics-player-sha256", "a" * 64,
            "--physics-protocol-sha256", "b" * 64,
            "--physics-archive-sha256", "c" * 64,
            "--scenario-manifest", str(manifest_path),
            "--scenario-xml", str(xml_path),
        ]
        request = type(
            "RuntimeInput",
            (),
            {
                "attempt_id": "attempt-1",
                "attempt_number": 1,
                "expected_initial_engine_state_identity": "expected-initial-state",
                "interface_action": ACTION,
            },
        )()
        loaded_plan = object()

        def execute_plan(loaded, runtime, output_dir):
            self.assertIs(loaded, loaded_plan)
            self.assertEqual(output_dir, self.temporary / "rollouts")
            runtime(request)
            return {"accepted_count": 1, "rejected_count": 0, "failed_count": 0}

        with (
            patch("sys.argv", args),
            patch("scripts.collection_plan.load_collection_plan", return_value=loaded_plan) as load_plan,
            patch("scripts.collection_plan.execute_collection_plan", side_effect=execute_plan),
            patch(
                "scripts.collect_rollouts.collect_fresh_engine_attempt",
                return_value={
                    "status": "accepted",
                    "reason": None,
                    "failure_code": None,
                    "realized_coverage_strata": [],
                    "eligible": True,
                    "artifact_path": "accepted/attempt-1",
                    "quarantine_path": None,
                    "failure_manifest_path": None,
                },
            ) as attempt,
        ):
            main()

        load_plan.assert_called_once_with(plan_path)
        supplied = attempt.call_args.kwargs["scenario_manifest"]
        self.assertIsInstance(supplied, ScenarioManifest)
        self.assertEqual(supplied.scenario_lineage.identity, expected.scenario_lineage.identity)
        self.assertEqual(
            supplied.declared_initial_engine_state.identity,
            expected.declared_initial_engine_state.identity,
        )

    def test_main_rejects_missing_or_invalid_fresh_physics_scenario_inputs_before_collection(self) -> None:
        base = [
            "collect_rollouts.py",
            "--output-dir", str(self.temporary / "rollouts"),
            "--physics-capture-v1",
            "--fresh-engine-per-rollout",
            "--physics-player-sha256", "a" * 64,
            "--physics-protocol-sha256", "b" * 64,
            "--physics-archive-sha256", "c" * 64,
        ]
        missing_manifest, xml_path, _ = _write_verified_scenario(self.temporary / "missing")
        invalid_manifest = self.temporary / "invalid.scenario.json"
        invalid_manifest.write_text("not json", encoding="utf-8")
        cases = (
            ("absent", [], "--scenario-manifest and --scenario-xml"),
            ("one-sided", ["--scenario-manifest", str(missing_manifest)], "--scenario-manifest and --scenario-xml"),
            ("invalid", ["--scenario-manifest", str(invalid_manifest), "--scenario-xml", str(xml_path)], "Cannot load scenario manifest"),
        )
        for name, extra, message in cases:
            with self.subTest(name=name):
                stderr = io.StringIO()
                with (
                    patch("sys.argv", [*base, *extra]),
                    patch("scripts.collect_rollouts.collect_fresh_engine_rollouts") as collect,
                    patch("scripts.collect_rollouts.start_engine") as start,
                    patch("scripts.collect_rollouts.connect_or_start_engine") as connect,
                    patch("sys.stderr", stderr),
                    self.assertRaises(SystemExit) as raised,
                ):
                    main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
                collect.assert_not_called()
                start.assert_not_called()
                connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
