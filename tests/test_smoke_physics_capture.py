from __future__ import annotations

from dataclasses import replace
import contextlib
import hashlib
import json
import mmap
import os
from pathlib import Path
import signal
import socket as socket_module
import subprocess
import tempfile
import time
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest import mock

from scripts import smoke_physics_capture as smoke
from scripts.smoke_physics_capture import (
    _listening_sockets,
    _mapped_regions,
    _observe_listener_owners,
    _proc_net_listeners,
    _process_tree,
    _socket_owner_scan,
    _stat_fields,
    archive_details,
    candidate_expectation,
    CandidateExpectation,
    canonical_root_from_git,
    capture_finalized_action,
    CapturedRequest,
    FRAME_HEIGHT_PIXELS,
    ListenerBindingError,
    ListenerObservation,
    SmokeError,
    ListeningSocket,
    free_port,
    perform_known_action,
    protected_receipt,
    require_action_events,
    require_collision,
    require_full_attribution,
    require_request_identity,
    require_stable_binding,
    resolve_listener_binding,
    run_smoke,
    SocketOwnerScan,
    start_display,
    tree_digest,
    wait_for_listener,
)
from scripts.prepare_rollout_dataset import resolve_physics_capture_provenance
from src.webui.bridge import PhysicsCaptureV1


ROOT = Path(__file__).resolve().parents[1]
ASSEMBLY_RELATIVE = "9001_Data/Managed/Assembly-CSharp.dll"
RUNTIME_RELATIVE = "UnityPlayer.so"


def build_candidate(root: Path, assembly_bytes: bytes = b"assembly-bytes-0001", runtime_bytes: bytes = b"unity-runtime-0001") -> tuple[CandidateExpectation, Path]:
    """Create a candidate payload plus a stage archive stand-in on disk."""
    assembly = root / ASSEMBLY_RELATIVE
    assembly.parent.mkdir(parents=True, exist_ok=True)
    assembly.write_bytes(assembly_bytes)
    (root / RUNTIME_RELATIVE).write_bytes(runtime_bytes)
    (root / "provenance.json").write_text(
        json.dumps({"files": {ASSEMBLY_RELATIVE: hashlib.sha256(assembly_bytes).hexdigest(), RUNTIME_RELATIVE: hashlib.sha256(runtime_bytes).hexdigest()}}, sort_keys=True),
        encoding="utf-8",
    )
    stage_archive = root.parent / "candidate.tar.gz"
    stage_archive.write_bytes(b"archive-bytes-0001")
    return candidate_expectation(root, hashlib.sha256(stage_archive.read_bytes()).hexdigest()), stage_archive


def as_candidate_process(observation: ListenerObservation, expectation: CandidateExpectation) -> ListenerObservation:
    """Keep every live `/proc` fact but stand the observer in for the player process."""
    return replace(observation, executable=expectation.root / "9001-player.x86_64", cwd=expectation.root, process_tree=(observation.pid,))


def valid_observation(expectation: CandidateExpectation, pid: int = 101, launched_pid: int = 4100) -> ListenerObservation:
    return ListenerObservation(
        pid,
        launched_pid,
        launched_pid,
        launched_pid,
        "11",
        expectation.root / "9001-player.x86_64",
        expectation.root,
        expectation.runtime,
        expectation.runtime_device,
        expectation.runtime_inode,
        expectation.runtime_sha256,
        expectation.assembly,
        expectation.assembly_device,
        expectation.assembly_inode,
        expectation.assembly_sha256,
        expectation.provenance_sha256,
        expectation.assembly_sha256,
        expectation.runtime_sha256,
        expectation.archive_sha256,
        tuple(sorted({pid, launched_pid})),
    )


class SmokePhysicsCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        """Keep every test off a real agent socket.

        `run_smoke` connects the agent before it binds the physics port, because
        the jar only spawns the player that owns that port once an agent
        completes its handshake. That put a real `connect_with_retry` -- with a
        90s deadline -- on the path of every listener-only fixture, which
        previously stopped at the bind. Patching here means a test that forgets
        the mock fails fast instead of hanging. Tests that assert on the connect
        install their own mock over this one.
        """
        patcher = mock.patch("scripts.smoke_physics_capture.connect_with_retry")
        self.connect = patcher.start()
        self.addCleanup(patcher.stop)
        # A real listening socket held by this process. The observation reader
        # re-confirms ownership against `/proc/<pid>/fd` at the end of its
        # identity reads, so an invented inode string is now rejected -- which is
        # the point of that check, and means these fixtures have to own the
        # inode they claim.
        self.listening_socket = socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM)
        self.addCleanup(self.listening_socket.close)
        self.listening_socket.bind(("127.0.0.1", 0))
        self.listening_socket.listen(1)
        self.socket_inode = str(os.fstat(self.listening_socket.fileno()).st_ino)

    def test_canonical_root_comes_from_git_common_directory(self) -> None:
        canonical_root = canonical_root_from_git(ROOT)
        self.assertTrue((canonical_root / ".git").is_dir())
        self.assertNotEqual(canonical_root, ROOT)

    def test_known_action_uses_request_62_slingshot_before_public_recorded_shoot(self) -> None:
        class Bridge:
            def __init__(self) -> None:
                self.recorded_shots: list[tuple[int, int, int, int, int]] = []

            def get_symbolic_state_without_screenshot(self) -> list[dict]:
                return [{"features": [{"properties": {"label": "Slingshot"}, "geometry": {"type": "Polygon", "coordinates": [[[100, 200], [120, 200], [120, 260], [100, 260]]]}}]}]

            def shoot_and_record_ground_truth(self, x: int, y: int, tap_time: int = 0, release_time: int = 0, frequency: int = 1) -> int:
                self.recorded_shots.append((x, y, tap_time, release_time, frequency))
                return 1

        bridge = Bridge()
        receipt = perform_known_action(bridge)
        # The release offset is the collision-producing one verified against an
        # accepted legacy rollout (drag_release [-80, 7]), not an arbitrary drag:
        # shot_x = gameX - 80 = 29, shot_y = gameY - 7 = 265 -> socket y = 480 - 1 - 265 = 214.
        self.assertEqual(bridge.recorded_shots, [(29, 214, 0, 1000, 1)])
        self.assertEqual(receipt["response"], 1)

    def test_the_known_action_uses_the_verified_collision_producing_release(self) -> None:
        """Pin the release offset itself, so a silent edit back to a shot that
        misses every structure cannot pass. The one permitted full smoke has no
        retry, and a collision-free shot fails acceptance outright."""
        recorded: list[dict] = []

        class Bridge:
            def get_symbolic_state_without_screenshot(self) -> list[dict]:
                return [{"features": [{"properties": {"label": "Slingshot"}, "geometry": {"type": "Polygon", "coordinates": [[[100, 200], [120, 200], [120, 260], [100, 260]]]}}]}]

            def shoot_and_record_ground_truth(self, x: int, y: int, tap_time: int = 0, release_time: int = 0, frequency: int = 1) -> int:
                recorded.append({"x": x, "y": y, "tap_time": tap_time, "release_time": release_time})
                return 1

        receipt = perform_known_action(Bridge())

        reference = receipt["slingshot_reference"]
        self.assertEqual(recorded[0]["x"] - int(reference["gameX"]), -80)
        self.assertEqual(int(reference["gameY"]) - (FRAME_HEIGHT_PIXELS - 1 - recorded[0]["y"]), 7)
        self.assertEqual((recorded[0]["tap_time"], recorded[0]["release_time"]), (0, 1000))

    @mock.patch("scripts.smoke_physics_capture.time.sleep")
    @mock.patch("scripts.smoke_physics_capture.connect_with_retry")
    def test_action_capture_retries_only_until_recorder_is_finalized(self, connect: mock.Mock, _sleep: mock.Mock) -> None:
        from src.webui.bridge import PhysicsCaptureV1Failure

        pending = mock.Mock()
        pending.get_physics_capture_v1.side_effect = PhysicsCaptureV1Failure(4, "no finalized recorder batch")
        finalized = mock.Mock()
        finalized.get_physics_capture_v1.return_value = PhysicsCaptureV1(b"png", MappingProxyType({}), ())
        connect.side_effect = [pending, finalized]

        capture = capture_finalized_action(2004, deadline_seconds=1.0)

        self.assertEqual(capture.png, b"png")
        pending.disconnect.assert_called_once_with()
        finalized.disconnect.assert_called_once_with()

    def test_action_capture_requires_authoritative_launch_event(self) -> None:
        with self.assertRaisesRegex(Exception, "bird_launched"):
            require_action_events(())
        event_types = require_action_events((MappingProxyType({"event_type": "bird_launched"}),))
        self.assertEqual(event_types, ("bird_launched",))

    def test_captured_request_thaws_nested_bridge_json(self) -> None:
        capture = PhysicsCaptureV1(b"png", MappingProxyType({"coordinates": MappingProxyType({"world": "unity"})}), ())
        thawed = CapturedRequest(capture).get_physics_capture_v1()
        self.assertEqual(thawed.state, {"coordinates": {"world": "unity"}})

    @mock.patch("scripts.smoke_physics_capture.subprocess.Popen")
    @mock.patch("scripts.smoke_physics_capture.terminate", return_value="pid=4100:exit=143")
    @mock.patch("scripts.smoke_physics_capture.wait_for_listener", side_effect=ListenerBindingError("expected one port-2004 listener owner, found 0"))
    @mock.patch("scripts.smoke_physics_capture.start_display")
    @mock.patch("scripts.smoke_physics_capture.archive_details")
    def test_listener_binding_failure_is_persisted_and_cleanup_runs(self, archive: mock.Mock, display: mock.Mock, _listener: mock.Mock, terminate: mock.Mock, popen: mock.Mock) -> None:
        process = popen.return_value
        process.pid = 4100
        display.return_value = (":201", process)
        archive.side_effect = self._staged_candidate
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            args = SimpleNamespace(stage=ROOT / "sciencebirdsgames/physics-v1", output_dir=output, report=output / "report.json", canonical_root=ROOT, agent_port=free_port(), game_port=free_port(), physics_port=free_port(), listener_deadline_seconds=0.01, listener_only=True, inject_frame_mismatch=False, inject_missing_sidecar=False, inject_request_failure=False, agent_timeout_seconds=1.0, port_grace_seconds=0.0)
            report, code = run_smoke(args)
            self.assertEqual(code, 1)
            self.assertIn("expected one port-2004 listener owner", report["error"])
            self.assertTrue((output / "report.json").is_file())
            self.assertEqual(terminate.call_count, 2)


    def test_proc_listener_reader_resolves_socket_owner(self) -> None:
        with socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            listener.listen()
            inodes = _proc_net_listeners(port)
            self.assertEqual(len(inodes), 1)
            owners = _socket_owner_scan(inodes).owners
            self.assertIn((os.getpid(), inodes[0]), owners)

    @mock.patch("scripts.smoke_physics_capture.time.sleep")
    @mock.patch("scripts.smoke_physics_capture.subprocess.Popen")
    def test_private_display_uses_glx_capable_xvnc(self, popen: mock.Mock, _sleep: mock.Mock) -> None:
        process = popen.return_value
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as temporary:
            start_display(Path(temporary) / "display.log")
        command = popen.call_args.args[0]
        self.assertEqual(command[0], "Xvnc")
        self.assertIn("-SecurityTypes", command)
        self.assertIn("-rfbport", command)

    def test_tree_digest_is_stable_and_detects_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "nested" / "file.bin"
            target.parent.mkdir()
            target.write_bytes(b"one")
            first = tree_digest(root)
            self.assertEqual(first, tree_digest(root))
            target.write_bytes(b"two")
            self.assertNotEqual(first, tree_digest(root))

    def test_active_data_receipt_detects_nested_file_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "episode" / "frames" / "frame.png"
            nested.parent.mkdir(parents=True)
            nested.write_bytes(b"one")
            first = protected_receipt("active_data", root)
            nested.write_bytes(b"two")
            self.assertNotEqual(first, protected_receipt("active_data", root))

    def test_missing_root_has_explicit_digest(self) -> None:
        self.assertEqual(tree_digest(ROOT / "does-not-exist-for-physics-smoke"), "ABSENT")

    def test_stage_archive_provenance_is_verified_in_a_clone(self) -> None:
        stage = ROOT / "sciencebirdsgames" / "physics-v1"
        with tempfile.TemporaryDirectory() as temporary:
            _, archive_sha, player_sha, protocol_sha, assembly_sha = archive_details(stage, Path(temporary) / "clone")
        self.assertEqual(len(archive_sha), 64)
        self.assertEqual(len(player_sha), 64)
        self.assertEqual(len(protocol_sha), 64)
        self.assertEqual(len(assembly_sha), 64)

    def test_request_identity_requires_stable_capture_and_increasing_sequence(self) -> None:
        first = PhysicsCaptureV1(b"png", MappingProxyType({"capture_id": "capture-1", "sequence": 2, "render_frame": 10}), ())
        second = PhysicsCaptureV1(b"png", MappingProxyType({"capture_id": "capture-1", "sequence": 3, "render_frame": 11}), ())
        self.assertEqual(require_request_identity((first, second))[1]["sequence"], 3)
        stale = PhysicsCaptureV1(b"png", MappingProxyType({"capture_id": "capture-1", "sequence": 2}), ())
        with self.assertRaisesRegex(Exception, "did not increase"):
            require_request_identity((first, stale))

    def test_collision_requires_deterministic_ids_and_finite_non_negative_speed(self) -> None:
        event = MappingProxyType({"event_type": "collision", "fixed_step": 7, "render_frame": 9, "payload": MappingProxyType({"contact_ids": ("a", "b"), "relative_speed": 1.5})})
        self.assertEqual(require_collision((event,))["contact_ids"], ["a", "b"])
        invalid_ids = MappingProxyType({"event_type": "collision", "payload": MappingProxyType({"contact_ids": ("b", "a"), "relative_speed": 1.5})})
        with self.assertRaisesRegex(Exception, "sorted"):
            require_collision((invalid_ids,))
        invalid_speed = MappingProxyType({"event_type": "collision", "payload": MappingProxyType({"contact_ids": ("a",), "relative_speed": float("nan")})})
        with self.assertRaisesRegex(Exception, "finite"):
            require_collision((invalid_speed,))
        with self.assertRaisesRegex(Exception, "no collision"):
            require_collision(())

    def test_free_port_returns_bindable_port(self) -> None:
        port = free_port()
        self.assertGreater(port, 0)

    def test_listener_binding_rejects_absent_or_ambiguous_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            expectation, _ = build_candidate(Path(temporary) / "player")
            with self.assertRaisesRegex(ListenerBindingError, "expected one port-2004 listener owner, found 0"):
                resolve_listener_binding(2004, 4100, (), expectation)
            owners = (valid_observation(expectation, pid=101), valid_observation(expectation, pid=102))
            with self.assertRaisesRegex(ListenerBindingError, "expected one port-2004 listener owner, found 2"):
                resolve_listener_binding(2004, 4100, owners, expectation)

    def test_listener_binding_rejects_every_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            expectation, _ = build_candidate(Path(temporary) / "player")
            valid = valid_observation(expectation)
            accepted = resolve_listener_binding(2004, 4100, (valid,), expectation)
            # Assert against the expectation, never against the observation the
            # fixture built, so a check that stopped comparing would be visible.
            self.assertEqual(accepted["assembly_path"], str(expectation.assembly))
            self.assertEqual(accepted["assembly_sha256"], expectation.assembly_sha256)
            self.assertEqual(accepted["runtime_path"], str(expectation.runtime))
            self.assertEqual(accepted["runtime_sha256"], expectation.runtime_sha256)
            self.assertEqual(accepted["provenance_runtime_sha256"], expectation.runtime_sha256)
            self.assertEqual(accepted["provenance_sha256"], expectation.provenance_sha256)
            self.assertEqual(accepted["archive_sha256"], expectation.archive_sha256)
            self.assertEqual(accepted["assembly_inode"], expectation.assembly_inode)
            self.assertEqual(accepted["runtime_inode"], expectation.runtime_inode)
            cases = (
                (replace(valid, executable=Path("/other/player")), "listener executable differs"),
                (replace(valid, cwd=Path("/other")), "listener cwd differs"),
                (replace(valid, runtime_path=Path("/other/UnityPlayer.so")), "listener mapped runtime path differs"),
                (replace(valid, runtime_device="fe:ff"), "listener mapped runtime device differs"),
                (replace(valid, runtime_inode=valid.runtime_inode + 1), "listener mapped runtime inode differs"),
                (replace(valid, runtime_sha256="a" * 64), "listener runtime digest differs"),
                (replace(valid, provenance_runtime_sha256="9" * 64), "listener provenance runtime digest differs"),
                (replace(valid, assembly_device="fe:ff"), "listener assembly device differs"),
                (replace(valid, assembly_inode=valid.assembly_inode + 1), "listener assembly inode differs"),
                (replace(valid, assembly_sha256="b" * 64), "listener assembly digest differs"),
                (replace(valid, provenance_assembly_sha256="c" * 64), "listener provenance assembly digest differs"),
                (replace(valid, provenance_sha256="d" * 64), "listener provenance digest differs"),
                (replace(valid, archive_sha256="e" * 64), "staged archive digest drifted during the run"),
                (replace(valid, process_tree=()), "process tree is not rooted at the launched pid"),
                # Containment only means something while the tree is rooted at
                # the pid this run launched, so the two failures are distinct and
                # both must bite.
                (replace(valid, process_tree=(4100,)), "listener outside launched process tree"),
            )
            for observation, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(ListenerBindingError, message):
                    resolve_listener_binding(2004, 4100, (observation,), expectation)

    def test_mapped_regions_observes_this_process_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            expectation, _ = build_candidate(Path(temporary) / "player")
            self.assertEqual(_mapped_regions(os.getpid(), RUNTIME_RELATIVE), ())
            with open(expectation.runtime, "rb") as handle, mmap.mmap(handle.fileno(), 0, prot=mmap.PROT_READ) as mapping:
                self.assertEqual(mapping[:5], b"unity")
                regions = _mapped_regions(os.getpid(), RUNTIME_RELATIVE)
            self.assertEqual(len(regions), 1)
            self.assertEqual(regions[0].path, expectation.runtime)
            self.assertEqual(regions[0].device, expectation.runtime_device)
            self.assertEqual(regions[0].inode, expectation.runtime_inode)

    def test_mapped_regions_rejects_a_deleted_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            expectation, _ = build_candidate(Path(temporary) / "player")
            with open(expectation.runtime, "rb") as handle, mmap.mmap(handle.fileno(), 0, prot=mmap.PROT_READ):
                expectation.runtime.unlink()
                with self.assertRaisesRegex(ListenerBindingError, "mapped a deleted UnityPlayer.so"):
                    _mapped_regions(os.getpid(), RUNTIME_RELATIVE)

    def test_observation_binds_the_live_process_mapping_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            expectation, stage_archive = build_candidate(Path(temporary) / "player")
            with open(expectation.runtime, "rb") as handle, mmap.mmap(handle.fileno(), 0, prot=mmap.PROT_READ):
                observations = _observe_listener_owners(((os.getpid(), self.socket_inode),), os.getpid(), expectation, stage_archive)
            self.assertEqual(len(observations), 1)
            observed = observations[0]
            self.assertEqual(observed.runtime_path, expectation.runtime)
            self.assertEqual(observed.runtime_inode, expectation.runtime_inode)
            self.assertEqual(observed.runtime_sha256, expectation.runtime_sha256)
            self.assertEqual(observed.provenance_runtime_sha256, expectation.runtime_sha256)
            # The assembly is never mapped by a real Unity player, so it is read
            # through the process's own root rather than taken from the map table.
            self.assertEqual(observed.assembly_path, expectation.assembly)
            self.assertEqual(observed.assembly_inode, expectation.assembly_inode)
            self.assertEqual(observed.assembly_sha256, expectation.assembly_sha256)
            self.assertEqual(observed.provenance_assembly_sha256, expectation.assembly_sha256)
            self.assertEqual(observed.archive_sha256, expectation.archive_sha256)
            self.assertEqual(observed.cwd, Path.cwd())

    def test_observation_does_not_require_a_mapped_assembly(self) -> None:
        """Unity never maps `Assembly-CSharp.dll`, so requiring it rejected everything.

        A live player was measured with 91 file-backed mappings -- six framework
        DLLs among them -- and zero `Assembly-CSharp.dll` images. The mapped
        anchor is `UnityPlayer.so`; the assembly is still pinned, by digest read
        through the listener's own root.
        """
        with tempfile.TemporaryDirectory() as temporary:
            expectation, stage_archive = build_candidate(Path(temporary) / "player")
            self.assertEqual(_mapped_regions(os.getpid(), "Assembly-CSharp.dll"), ())
            with open(expectation.runtime, "rb") as handle, mmap.mmap(handle.fileno(), 0, prot=mmap.PROT_READ):
                observations = _observe_listener_owners(((os.getpid(), self.socket_inode),), os.getpid(), expectation, stage_archive)
            self.assertEqual(observations[0].assembly_sha256, expectation.assembly_sha256)

    def test_observation_rejects_a_process_without_a_mapped_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            expectation, stage_archive = build_candidate(Path(temporary) / "player")
            with self.assertRaisesRegex(ListenerBindingError, "has 0 mapped UnityPlayer.so images"):
                _observe_listener_owners(((os.getpid(), self.socket_inode),), os.getpid(), expectation, stage_archive)

    def test_observation_rejects_provenance_without_a_runtime_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            expectation, stage_archive = build_candidate(Path(temporary) / "player")
            manifest = json.loads((expectation.root / "provenance.json").read_text(encoding="utf-8"))
            del manifest["files"][RUNTIME_RELATIVE]
            (expectation.root / "provenance.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
            with open(expectation.runtime, "rb") as handle, mmap.mmap(handle.fileno(), 0, prot=mmap.PROT_READ):
                with self.assertRaisesRegex(ListenerBindingError, "lacks a UnityPlayer.so digest"):
                    _observe_listener_owners(((os.getpid(), self.socket_inode),), os.getpid(), expectation, stage_archive)

    def test_observation_rejects_an_unattributable_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            expectation, stage_archive = build_candidate(Path(temporary) / "player")
            with self.assertRaisesRegex(ListenerBindingError, "is not attributable"):
                _observe_listener_owners(((2 ** 30, "77"),), os.getpid(), expectation, stage_archive)

    def test_payload_mutation_after_expectation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            expectation, stage_archive = build_candidate(Path(temporary) / "player")
            with open(expectation.runtime, "rb") as handle, mmap.mmap(handle.fileno(), 0, prot=mmap.PROT_READ):
                with open(expectation.assembly, "r+b") as writable:
                    writable.write(b"ASSEMBLY")
                observations = _observe_listener_owners(((os.getpid(), self.socket_inode),), os.getpid(), expectation, stage_archive)
            self.assertEqual(observations[0].assembly_inode, expectation.assembly_inode)
            self.assertNotEqual(observations[0].assembly_sha256, expectation.assembly_sha256)
            with self.assertRaisesRegex(ListenerBindingError, "listener assembly digest differs"):
                resolve_listener_binding(2004, os.getpid(), (as_candidate_process(observations[0], expectation),), expectation)

    def test_runtime_mutation_after_expectation_is_rejected(self) -> None:
        """A swapped native runtime is the mapped-code substitution the gate exists for."""
        with tempfile.TemporaryDirectory() as temporary:
            expectation, stage_archive = build_candidate(Path(temporary) / "player")
            with open(expectation.runtime, "rb") as handle, mmap.mmap(handle.fileno(), 0, prot=mmap.PROT_READ):
                with open(expectation.runtime, "r+b") as writable:
                    writable.write(b"UNITY")
                observations = _observe_listener_owners(((os.getpid(), self.socket_inode),), os.getpid(), expectation, stage_archive)
            self.assertEqual(observations[0].runtime_inode, expectation.runtime_inode)
            with self.assertRaisesRegex(ListenerBindingError, "listener runtime digest differs"):
                resolve_listener_binding(2004, os.getpid(), (as_candidate_process(observations[0], expectation),), expectation)

    def test_staged_archive_mutation_after_expectation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            expectation, stage_archive = build_candidate(Path(temporary) / "player")
            with open(expectation.runtime, "rb") as handle, mmap.mmap(handle.fileno(), 0, prot=mmap.PROT_READ):
                stage_archive.write_bytes(b"archive-bytes-0002")
                observations = _observe_listener_owners(((os.getpid(), self.socket_inode),), os.getpid(), expectation, stage_archive)
            with self.assertRaisesRegex(ListenerBindingError, "staged archive digest drifted during the run"):
                resolve_listener_binding(2004, os.getpid(), (as_candidate_process(observations[0], expectation),), expectation)

    def test_unmutated_live_observation_binds_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            expectation, stage_archive = build_candidate(Path(temporary) / "player")
            with open(expectation.runtime, "rb") as handle, mmap.mmap(handle.fileno(), 0, prot=mmap.PROT_READ):
                observations = _observe_listener_owners(((os.getpid(), self.socket_inode),), os.getpid(), expectation, stage_archive)
            binding = resolve_listener_binding(2004, os.getpid(), (as_candidate_process(observations[0], expectation),), expectation)
            self.assertEqual(binding["pid"], os.getpid())
            self.assertEqual(binding["assembly_inode"], expectation.assembly_inode)
            self.assertEqual(binding["runtime_inode"], expectation.runtime_inode)

    def test_socket_owner_scan_reports_owners_and_blind_spots(self) -> None:
        with socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            listener.listen()
            inodes = _proc_net_listeners(port)
            self.assertEqual(len(inodes), 1)
            scan = _socket_owner_scan(inodes)
            self.assertIn((os.getpid(), inodes[0]), scan.owners)
            self.assertNotIn(os.getpid(), scan.unreadable_pids)
            sockets = _listening_sockets(port)
            self.assertEqual({entry.inode for entry in sockets}, set(inodes))
            self.assertTrue(all(entry.local_address.endswith(f"{port:04X}") for entry in sockets))
            require_full_attribution(inodes, scan)

    def test_full_attribution_rejects_an_unattributed_socket(self) -> None:
        attributed = SocketOwnerScan(((101, "77"),), ())
        require_full_attribution(("77",), attributed)
        # Opaque processes alone are normal on a shared host and must not reject.
        require_full_attribution(("77",), SocketOwnerScan(((101, "77"),), (4242,)))
        # An inode nobody could be attributed to is the impostor case.
        with self.assertRaisesRegex(Exception, r"\['88'\].*no attributable owner.*1 processes were opaque"):
            require_full_attribution(("77", "88"), SocketOwnerScan(((101, "77"),), (4242,)))

    def test_wait_for_listener_rejects_a_socket_it_cannot_attribute(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            expectation, _ = build_candidate(Path(temporary) / "player")
            reader = mock.Mock(return_value=(valid_observation(expectation),))
            # Both loopback, in the two families the reader accepts: an unowned
            # inode must reject on attribution, not incidentally on its address.
            sockets = (ListeningSocket("77", "0100007F:07D4", "tcp"), ListeningSocket("88", "00000000000000000000000001000000:07D4", "tcp6"))
            with mock.patch.object(smoke, "_listening_sockets", return_value=sockets):
                with mock.patch.object(smoke, "_socket_owner_scan", return_value=SocketOwnerScan(((101, "77"),), (4242,))):
                    with self.assertRaisesRegex(Exception, r"\['88'\].*no attributable owner"):
                        wait_for_listener(2004, 4100, expectation, reader, deadline_seconds=0.05, sleep=lambda _: None)
            reader.assert_not_called()

    def test_wait_for_listener_rejects_a_port_bound_to_the_wildcard_address(self) -> None:
        """A wildcard bind exposes the physics port to the network for the run.

        The gate used to record the bind address and accept it regardless, so a
        player reachable from off-host was indistinguishable from one reachable
        only from this host. Attribution is not reached: the address is refused
        first, because a fully attributed wildcard listener is still wrong.
        """
        with tempfile.TemporaryDirectory() as temporary:
            expectation, _ = build_candidate(Path(temporary) / "player")
            reader = mock.Mock(return_value=(valid_observation(expectation),))
            entry = ListeningSocket("77", "00000000:07D4", "tcp")
            with mock.patch.object(smoke, "_listening_sockets", return_value=(entry,)):
                with mock.patch.object(smoke, "_socket_owner_scan", return_value=SocketOwnerScan(((101, "77"),), ())) as scan:
                    with self.assertRaisesRegex(ListenerBindingError, r"non-loopback address: \['00000000:07D4'\]"):
                        wait_for_listener(2004, 4100, expectation, reader, deadline_seconds=0.05, sleep=lambda _: None)
            scan.assert_not_called()
            reader.assert_not_called()

    def test_wait_for_listener_binds_and_records_socket_addresses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            expectation, _ = build_candidate(Path(temporary) / "player")
            entry = ListeningSocket("77", "0100007F:07D4", "tcp")
            with mock.patch.object(smoke, "_listening_sockets", return_value=(entry,)):
                with mock.patch.object(smoke, "_socket_owner_scan", return_value=SocketOwnerScan(((101, "77"),), ())):
                    binding = wait_for_listener(2004, 4100, expectation, lambda owners: (valid_observation(expectation),), deadline_seconds=0.5, sleep=lambda _: None)
            self.assertEqual(binding["listening_sockets"], [{"inode": "77", "local_address": "0100007F:07D4", "table": "tcp"}])

    def test_accepted_binding_records_how_blind_the_scan_was(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            expectation, _ = build_candidate(Path(temporary) / "player")
            entry = ListeningSocket("77", "0100007F:07D4", "tcp")
            scan = SocketOwnerScan(((101, "77"),), (900, 901))
            with mock.patch.object(smoke, "_listening_sockets", return_value=(entry,)):
                with mock.patch.object(smoke, "_socket_owner_scan", return_value=scan):
                    binding = wait_for_listener(2004, 4100, expectation, lambda owners: (valid_observation(expectation),), deadline_seconds=0.5, sleep=lambda _: None)
        # Audit evidence only: a host-wide census, not a per-binding score.
        self.assertEqual(binding["host_uninspectable_pid_count"], 2)
        self.assertEqual(binding["host_uninspectable_pids"], [900, 901])

    def test_wait_for_listener_times_out_when_no_socket_appears(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            expectation, _ = build_candidate(Path(temporary) / "player")
            reader = mock.Mock(return_value=())
            with mock.patch.object(smoke, "_listening_sockets", return_value=()):
                with self.assertRaisesRegex(Exception, "did not become attributable"):
                    wait_for_listener(2004, 4100, expectation, reader, deadline_seconds=0.05, sleep=lambda _: None)
            reader.assert_not_called()

    def test_unreadable_proc_net_table_rejects_instead_of_reporting_a_clear_port(self) -> None:
        real_read = Path.read_text

        def deny_tcp6(self: Path, *args: object, **kwargs: object) -> str:
            if self.as_posix() == "/proc/net/tcp6":
                raise PermissionError(13, "Permission denied")
            return real_read(self, *args, **kwargs)

        with mock.patch.object(Path, "read_text", deny_tcp6):
            with self.assertRaisesRegex(Exception, "tcp6 is unreadable"):
                _listening_sockets(2004)

    def test_absent_proc_net_table_is_determinable_and_not_a_rejection(self) -> None:
        real_read = Path.read_text

        def hide_tcp6(self: Path, *args: object, **kwargs: object) -> str:
            if self.as_posix() == "/proc/net/tcp6":
                raise FileNotFoundError(2, "No such file")
            return real_read(self, *args, **kwargs)

        with socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            listener.listen()
            with mock.patch.object(Path, "read_text", hide_tcp6):
                self.assertEqual(len(_listening_sockets(port)), 1)

    def test_masked_proc_rejects_instead_of_reporting_an_empty_port(self) -> None:
        def hide_every_table(self: Path, *args: object, **kwargs: object) -> str:
            raise FileNotFoundError(2, "No such file")

        # `mount -o subset=pid /proc` hides both tables; an empty result there would
        # otherwise be indistinguishable from a genuinely drained port.
        with mock.patch.object(Path, "read_text", hide_every_table):
            with self.assertRaisesRegex(Exception, "no /proc/net TCP table exists"):
                _listening_sockets(2004)

    def test_process_tree_survives_a_forged_comm(self) -> None:
        tree = _process_tree(os.getppid())
        self.assertIn(os.getpid(), tree)
        fields = _stat_fields(os.getpid())
        self.assertEqual(int(fields[1]), os.getppid())
        forged = "9999 (a) S 4100 4100) S 1 1 1\n"
        with mock.patch.object(Path, "read_text", return_value=forged):
            self.assertEqual(_stat_fields(9999)[1], "1")

    def test_process_tree_reaches_transitive_descendants(self) -> None:
        """The Unity player is a grandchild, so direct children are not enough.

        A direct-children-only snapshot passes every other test in this file and
        would then place the real listener outside the launched tree. This is the
        only test that separates the two implementations, so it uses real pids
        rather than a fixture: `sh` spawns a child, and that child is what must
        appear.
        """
        process = subprocess.Popen(["sh", "-c", "sleep 30 & echo $! ; wait"], stdout=subprocess.PIPE)
        try:
            assert process.stdout is not None
            grandchild = int(process.stdout.readline().strip())
            tree = _process_tree(process.pid)
            self.assertIn(process.pid, tree)
            self.assertIn(grandchild, tree, "process tree stopped at direct children")
            self.assertNotIn(os.getpid(), tree, "tree leaked upward past its root")
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.kill(grandchild, signal.SIGKILL)
            process.kill()
            process.wait(timeout=5.0)
            process.stdout.close()

    def test_process_view_reads_through_the_observed_process_root(self) -> None:
        """Reading the host path instead would trust a path the candidate controls.

        A candidate in its own mount namespace can make `/.../Assembly-CSharp.dll`
        name our verified file while it runs something else; only `/proc/<pid>/root`
        resolves the name in the namespace that actually loaded it. Nothing else in
        this suite distinguishes the two -- the same-namespace case makes them read
        identical bytes -- so the constructed path itself is what gets asserted.
        """
        self.assertEqual(
            smoke._process_view(4100, Path("/opt/player/Assembly-CSharp.dll")),
            Path("/proc/4100/root/opt/player/Assembly-CSharp.dll"),
        )
        # And it must be usable: our own root view of a real file reads its bytes.
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "payload.bin"
            target.write_bytes(b"through-the-root")
            self.assertEqual(smoke._process_view(os.getpid(), target).read_bytes(), b"through-the-root")

    def test_the_launch_environment_refuses_code_interposition(self) -> None:
        """Every digest can pass while `LD_PRELOAD` rewrites the measured payloads.

        The identity chain pins the candidate's bytes; it cannot pin what *else*
        the process loads, and an interposed library can forge exactly the
        collision and capture fields this gate exists to measure.
        """
        for name in smoke.INTERPOSITION_ENV_VARS:
            with self.subTest(variable=name):
                with self.assertRaisesRegex(SmokeError, f"code-interposition environment set: {name}"):
                    smoke.launch_environment(":203", {name: "/tmp/patch.so"})
        # An empty value is not a set variable and must not block the gate.
        child, stripped = smoke.launch_environment(":203", {"LD_PRELOAD": "", "PATH": "/usr/bin"})
        self.assertEqual(stripped, ())
        self.assertEqual(child["DISPLAY"], ":203")

    def test_the_launch_environment_strips_the_loader_search_path(self) -> None:
        """Refusing this one would make the gate unrunnable on CUDA hosts.

        The player resolves its own libraries through its rpath, so the variable
        is dropped rather than rejected -- and the drop is recorded, because a
        silent environment edit is the kind of thing a later reader has to be
        able to see.
        """
        child, stripped = smoke.launch_environment(":203", {"LD_LIBRARY_PATH": "/opt/cuda/lib64", "PATH": "/usr/bin"})
        self.assertEqual(stripped, ("LD_LIBRARY_PATH",))
        self.assertNotIn("LD_LIBRARY_PATH", child)
        self.assertEqual(child["PATH"], "/usr/bin")

    def test_the_observation_reader_rejects_a_recycled_owner_pid(self) -> None:
        """The owner scan and the identity reads are two independent /proc passes.

        An owner that exits between them can have its pid reused by another
        descendant of the same launch, and the binding would then splice a dead
        process's socket inode onto a live process's identity. `starttime` is
        kernel-written and unique per incarnation, so a changed value is proof of
        exactly that splice.
        """
        with tempfile.TemporaryDirectory() as temporary:
            expectation, stage_archive = build_candidate(Path(temporary) / "player")
            real_stat_fields = smoke._stat_fields
            calls = {"n": 0}

            def drifting_starttime(pid: int) -> tuple[str, ...]:
                fields = list(real_stat_fields(pid))
                calls["n"] += 1
                if calls["n"] > 1:
                    fields[19] = str(int(fields[19]) + 1)
                return tuple(fields)

            with open(expectation.runtime, "rb") as handle, mmap.mmap(handle.fileno(), 0, prot=mmap.PROT_READ):
                # The tree snapshot walks every pid through `_stat_fields`, so it
                # is pinned here to keep the counter counting only the two reads
                # this test is about.
                with mock.patch.object(smoke, "_process_tree", return_value=(os.getpid(),)):
                    with mock.patch.object(smoke, "_stat_fields", drifting_starttime):
                        with self.assertRaisesRegex(ListenerBindingError, "was replaced during observation"):
                            _observe_listener_owners(((os.getpid(), self.socket_inode),), os.getpid(), expectation, stage_archive)

    def test_the_observation_reader_rejects_an_owner_that_dropped_the_socket(self) -> None:
        """Ownership has to hold at the end of the reads, not only at the start.

        `starttime` alone proves the process was not replaced; it does not prove
        it still holds the socket. Re-reading the descriptor table closes the
        remaining half of that window.
        """
        with tempfile.TemporaryDirectory() as temporary:
            expectation, stage_archive = build_candidate(Path(temporary) / "player")
            with open(expectation.runtime, "rb") as handle, mmap.mmap(handle.fileno(), 0, prot=mmap.PROT_READ):
                with self.assertRaisesRegex(ListenerBindingError, "no longer holds socket inode 424242"):
                    _observe_listener_owners(((os.getpid(), "424242"),), os.getpid(), expectation, stage_archive)

    def test_every_observation_field_is_checked_or_declared_unchecked(self) -> None:
        """A new observation field must not be able to arrive unexamined.

        The binding is only as strong as the fields it compares, and a field
        added without deciding its fate reads as covered while contributing
        nothing. The guard turns that omission into a hard failure at bind time,
        which is what this asserts -- by removing a field's declaration.
        """
        with tempfile.TemporaryDirectory() as temporary:
            expectation, _ = build_candidate(Path(temporary) / "player")
            observation = valid_observation(expectation)
            self.assertIsNotNone(resolve_listener_binding(2004, 4100, (observation,), expectation))
            with mock.patch.object(smoke, "UNCHECKED_OBSERVATION_FIELDS", ("parent_pid", "process_group", "session_id", "socket_inode")):
                with self.assertRaisesRegex(SmokeError, "neither checked nor declared unchecked: assembly_path"):
                    resolve_listener_binding(2004, 4100, (observation,), expectation)

    def test_binding_stability_requires_the_same_process_and_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            expectation, _ = build_candidate(Path(temporary) / "player")
            binding = resolve_listener_binding(2004, 4100, (valid_observation(expectation),), expectation)
            self.assertEqual(require_stable_binding(binding, dict(binding))["rebound_pid"], 101)
            sentinels: dict[str, object] = {
                "pid": 999, "socket_inode": "12", "executable": "/other/9001-player.x86_64",
                "cwd": "/other", "assembly_path": "/other/Assembly-CSharp.dll",
                "assembly_device": "00:00", "assembly_inode": 1, "assembly_sha256": "f" * 64,
                "runtime_path": "/other/UnityPlayer.so", "runtime_device": "00:00",
                "runtime_inode": 2, "runtime_sha256": "c" * 64,
                "provenance_sha256": "e" * 64, "archive_sha256": "d" * 64,
            }
            self.assertEqual(sorted(sentinels), sorted(smoke.BINDING_IDENTITY_FIELDS))
            for field, value in sentinels.items():
                with self.subTest(field=field):
                    drifted = dict(binding)
                    drifted[field] = value
                    with self.assertRaisesRegex(Exception, f"drifted between listener bind and capture: {field}$"):
                        require_stable_binding(binding, drifted)

    def test_binding_stability_rejects_a_binding_missing_identity_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            expectation, _ = build_candidate(Path(temporary) / "player")
            binding = resolve_listener_binding(2004, 4100, (valid_observation(expectation),), expectation)
            with self.assertRaisesRegex(Exception, "rebind binding is missing identity fields"):
                require_stable_binding(binding, {})
            with self.assertRaisesRegex(Exception, "bind binding is missing identity fields"):
                require_stable_binding({}, binding)

    def test_every_collision_event_is_validated_not_only_the_first(self) -> None:
        valid = {"event_type": "collision", "payload": {"contact_ids": ("a", "b"), "relative_speed": 1.5}}
        accepted = require_collision((valid,))
        self.assertEqual(accepted["collision_count"], 1)
        for malformed, pattern in (
            ({"event_type": "collision", "payload": None}, "payload must be an object"),
            ({"event_type": "collision", "payload": {"contact_ids": ("a",), "relative_speed": -3.0}}, "finite and non-negative"),
            ({"event_type": "collision", "payload": {"contact_ids": ("b", "a"), "relative_speed": 1.0}}, "sorted, and unique"),
        ):
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(Exception, f"{pattern} \\(event 1\\)"):
                    require_collision((valid, malformed))

    def test_archive_receipt_cannot_name_a_path_outside_the_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / "stage"
            stage.mkdir()
            (stage / "archive.sha256").write_text(f"{'a' * 64}  ../../etc/passwd\n", encoding="ascii")
            with self.assertRaisesRegex(Exception, "bare file inside the stage"):
                archive_details(stage, Path(temporary) / "clone")

    @staticmethod
    def _staged_candidate(stage: Path, clone: Path) -> tuple[Path, str, str, str, str]:
        expectation, stage_archive = build_candidate(clone)
        return stage_archive, expectation.archive_sha256, "b" * 64, "c" * 64, expectation.assembly_sha256

    def _listener_only_args(self, output: Path) -> SimpleNamespace:
        return SimpleNamespace(
            stage=ROOT / "sciencebirdsgames/physics-v1", output_dir=output, report=output / "report.json",
            canonical_root=ROOT, agent_port=free_port(), game_port=free_port(), physics_port=free_port(),
            listener_deadline_seconds=0.01, listener_only=True, inject_frame_mismatch=False,
            inject_missing_sidecar=False, inject_request_failure=False,
            agent_timeout_seconds=1.0, port_grace_seconds=0.0,
        )

    @mock.patch("scripts.smoke_physics_capture.subprocess.Popen")
    @mock.patch("scripts.smoke_physics_capture.terminate", return_value="pid=4100:exit=143")
    @mock.patch("scripts.smoke_physics_capture.perform_known_action")
    @mock.patch("scripts.smoke_physics_capture.prepare_for_play")
    @mock.patch("scripts.smoke_physics_capture.connect_with_retry")
    @mock.patch("scripts.smoke_physics_capture.wait_for_listener")
    @mock.patch("scripts.smoke_physics_capture.start_display")
    @mock.patch("scripts.smoke_physics_capture.archive_details")
    def test_listener_only_diagnostic_persists_binding_without_gameplay(self, archive: mock.Mock, display: mock.Mock, listener: mock.Mock, connect: mock.Mock, level: mock.Mock, action: mock.Mock, _terminate: mock.Mock, popen: mock.Mock) -> None:
        process = popen.return_value
        process.pid = 4100
        display.return_value = (":201", process)
        archive.side_effect = self._staged_candidate
        listener.return_value = {"pid": 4100, "socket_inode": "77"}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            report, code = run_smoke(self._listener_only_args(output))
            persisted = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "diagnostic_accepted")
        self.assertEqual(report["listener_binding"], {"pid": 4100, "socket_inode": "77"})
        self.assertNotIn("wire_capture", report)
        self.assertNotIn("action", report)
        # The agent handshake is a precondition of the binding, not gameplay: it
        # is what makes the jar spawn the player that owns the physics port. The
        # diagnostic must take that step and stop -- no level, no shot.
        connect.assert_called_once()
        connect.return_value.configure.assert_called_once()
        connect.return_value.disconnect.assert_called_once()
        level.assert_not_called()
        action.assert_not_called()
        self.assertEqual(persisted["listener_binding"]["pid"], 4100)
        self.assertTrue(persisted["cleanup"]["physics_port_clear"])

    @mock.patch("scripts.smoke_physics_capture.subprocess.Popen")
    @mock.patch("scripts.smoke_physics_capture.terminate", return_value="pid=4100:exit=143")
    @mock.patch("scripts.smoke_physics_capture.connect_with_retry")
    @mock.patch("scripts.smoke_physics_capture.wait_for_listener")
    @mock.patch("scripts.smoke_physics_capture.start_display")
    @mock.patch("scripts.smoke_physics_capture.archive_details")
    def test_the_agent_connects_before_the_physics_port_is_probed(self, archive: mock.Mock, display: mock.Mock, listener: mock.Mock, connect: mock.Mock, _terminate: mock.Mock, popen: mock.Mock) -> None:
        """The bind cannot precede the connect, or it can only ever time out.

        Measured on the staged candidate: with `game_playing_interface.jar`
        running and no agent connected, 60s of polling reports zero player
        processes and nothing listening on the physics port. The jar does not
        own that port; the player it spawns on the agent handshake does. An
        implementation that binds first therefore has no reachable success path
        in either mode, which is exactly the defect this test pins.

        The recorded order includes `configure`, not just `connect`: it is the
        completed handshake that spawns the player, so a version that connected
        first but bound before configuring would still race a process that does
        not exist yet. Pinning only `["connect", "bind"]` left that reordering
        green.
        """
        process = popen.return_value
        process.pid = 4100
        display.return_value = (":203", process)
        archive.side_effect = self._staged_candidate
        order: list[str] = []
        bridge = mock.MagicMock()
        bridge.configure.side_effect = lambda *_a, **_k: order.append("configure")
        connect.side_effect = lambda *_a, **_k: (order.append("connect"), bridge)[1]
        listener.side_effect = lambda *_a, **_k: (order.append("bind"), {"pid": 4100, "socket_inode": "77"})[1]
        with tempfile.TemporaryDirectory() as temporary:
            report, code = run_smoke(self._listener_only_args(Path(temporary)))
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "diagnostic_accepted")
        self.assertEqual(order, ["connect", "configure", "bind"])

    @mock.patch("scripts.smoke_physics_capture.subprocess.Popen")
    @mock.patch("scripts.smoke_physics_capture.terminate", side_effect=OSError("killpg denied"))
    @mock.patch("scripts.smoke_physics_capture.wait_for_listener")
    @mock.patch("scripts.smoke_physics_capture.start_display")
    @mock.patch("scripts.smoke_physics_capture.archive_details")
    def test_cleanup_failure_rejects_but_still_persists_every_fact(self, archive: mock.Mock, display: mock.Mock, listener: mock.Mock, _terminate: mock.Mock, popen: mock.Mock) -> None:
        process = popen.return_value
        process.pid = 4100
        display.return_value = (":201", process)
        archive.side_effect = self._staged_candidate
        listener.return_value = {"pid": 4100, "socket_inode": "77"}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            report, code = run_smoke(self._listener_only_args(output))
            persisted = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "rejected")
        self.assertIn("killpg denied", report["cleanup"]["engine"])
        self.assertIn("killpg denied", report["cleanup"]["xvfb"])
        self.assertEqual(len(report["cleanup_failures"]), 2)
        self.assertEqual(persisted["listener_binding"]["pid"], 4100)
        self.assertIn("protected_after", persisted)
        self.assertTrue(persisted["protected_unchanged"])

    @mock.patch("scripts.smoke_physics_capture.subprocess.Popen")
    @mock.patch("scripts.smoke_physics_capture.terminate", side_effect=OSError("killpg denied"))
    @mock.patch("scripts.smoke_physics_capture.wait_for_listener")
    @mock.patch("scripts.smoke_physics_capture.start_display")
    @mock.patch("scripts.smoke_physics_capture.archive_details")
    def test_cleanup_failure_quarantines_an_already_published_shot(self, archive: mock.Mock, display: mock.Mock, listener: mock.Mock, _terminate: mock.Mock, popen: mock.Mock) -> None:
        process = popen.return_value
        process.pid = 4100
        display.return_value = (":202", process)
        archive.side_effect = self._staged_candidate
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            published = output / "shot_001"

            def publish_during_the_run(*_args: object, **_kwargs: object) -> dict[str, object]:
                # The shot must be created *by this run*; pre-creating it would make
                # it a prior run's artifact, which this run may not quarantine.
                published.mkdir()
                (published / "physics_state.jsonl").write_text("{}\n", encoding="utf-8")
                return {"pid": 4100, "socket_inode": "77"}

            listener.side_effect = publish_during_the_run
            report, code = run_smoke(self._listener_only_args(output))
            quarantined = output / "invalid_attempts" / "shot_001"
            self.assertTrue((quarantined / "physics_state.jsonl").is_file())
            self.assertFalse(published.exists())
            persisted = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "rejected")
        self.assertNotIn("accepted_shot", report)
        self.assertEqual(persisted["quarantine"], [str(quarantined)])

    def _accepting_args(self, output: Path) -> SimpleNamespace:
        args = self._listener_only_args(output)
        args.listener_only = False
        return args

    def _reach_accept(self, output: Path, stack: contextlib.ExitStack) -> dict[str, mock.Mock]:
        """Mock the gameplay boundary so `run_smoke` reaches its real accept line.

        Returns the installed mocks so a caller can assert the gates between
        capture and accept were actually invoked. Without that, deleting a gate
        from `run_smoke` leaves this fixture green -- these are the only tests
        that reach the accept line at all.
        """
        def write_shot(_request: object, shot: Path, **_kwargs: object) -> None:
            shot.mkdir(parents=True, exist_ok=True)
            (shot / "physics_state.jsonl").write_text("{}\n", encoding="utf-8")

        capture = SimpleNamespace(state=dict.fromkeys(smoke.WIRE_STATE_FIELDS, 1), events=())
        gates: dict[str, mock.Mock] = {}
        for name, value in (
            ("connect_with_retry", mock.DEFAULT), ("prepare_for_play", mock.DEFAULT),
            ("perform_known_action", {"shot": "known"}), ("require_request_identity", ("a", "b")),
            ("require_collision", {}), ("require_stable_binding", {}), ("require_action_events", ()),
            ("CapturedRequest", mock.DEFAULT),
        ):
            patcher = mock.patch.object(smoke, name) if value is mock.DEFAULT else mock.patch.object(smoke, name, return_value=value)
            gates[name] = stack.enter_context(patcher)
        gates["capture_finalized_action"] = stack.enter_context(mock.patch.object(smoke, "capture_finalized_action", return_value=capture))
        gates["capture_physics_rollout"] = stack.enter_context(mock.patch.object(smoke, "capture_physics_rollout", side_effect=write_shot))
        gates["validate_physics_shot_artifact"] = stack.enter_context(mock.patch.object(smoke, "validate_physics_shot_artifact", return_value=SimpleNamespace(state_count=2, event_count=1)))
        return gates

    ACCEPT_GATES = ("require_request_identity", "require_collision", "require_stable_binding", "require_action_events", "validate_physics_shot_artifact")

    @mock.patch("scripts.smoke_physics_capture.subprocess.Popen")
    @mock.patch("scripts.smoke_physics_capture.terminate", side_effect=OSError("killpg denied"))
    @mock.patch("scripts.smoke_physics_capture.wait_for_listener")
    @mock.patch("scripts.smoke_physics_capture.start_display")
    @mock.patch("scripts.smoke_physics_capture.archive_details")
    def test_a_real_accept_is_downgraded_and_the_consumer_rejects_the_marker(self, archive: mock.Mock, display: mock.Mock, listener: mock.Mock, _terminate: mock.Mock, popen: mock.Mock) -> None:
        process = popen.return_value
        process.pid = 4100
        display.return_value = (":203", process)
        archive.side_effect = self._staged_candidate
        listener.return_value = {"pid": 4100, "socket_inode": "77"}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with contextlib.ExitStack() as stack:
                gates = self._reach_accept(output, stack)
                report, code = run_smoke(self._accepting_args(output))
                # Every gate between capture and accept must actually run. These
                # assertions are what stop a deleted gate from passing silently.
                for name in self.ACCEPT_GATES:
                    gates[name].assert_called_once()
            persisted = json.loads((output / "report.json").read_text(encoding="utf-8"))
            # The run genuinely reached the accept line before cleanup failed.
            self.assertEqual(report["artifact"], {"states": 2, "events": 1})
            self.assertFalse((output / "shot_001").exists())
            self.assertTrue((output / "invalid_attempts" / "shot_001" / "physics_state.jsonl").is_file())
            marker = output / "marker.json"
            marker.write_text(json.dumps(persisted), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "status=accepted"):
                resolve_physics_capture_provenance(ROOT / "sciencebirdsgames/physics-v1/novphy-physics-player-2019.4.41f2.tar.gz", marker)
        self.assertEqual(code, 1)
        self.assertEqual(persisted["status"], "rejected")
        self.assertNotIn("accepted_shot", persisted)

    @mock.patch("scripts.smoke_physics_capture.subprocess.Popen")
    @mock.patch("scripts.smoke_physics_capture.terminate")
    @mock.patch("scripts.smoke_physics_capture.wait_for_listener")
    @mock.patch("scripts.smoke_physics_capture.start_display")
    @mock.patch("scripts.smoke_physics_capture.archive_details")
    def test_a_stale_handle_during_quarantine_does_not_suppress_the_receipt(self, archive: mock.Mock, display: mock.Mock, listener: mock.Mock, terminate: mock.Mock, popen: mock.Mock) -> None:
        process = popen.return_value
        process.pid = 4100
        display.return_value = (":204", process)
        archive.side_effect = self._staged_candidate
        listener.return_value = {"pid": 4100, "socket_inode": "77"}
        real_stat = Path.stat
        armed = False
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)

            def fail_cleanup(*_args: object, **_kwargs: object) -> str:
                # Arming only here puts the ESTALE inside the `finally`-block
                # quarantine of a genuinely published shot. Arming it earlier
                # would trip the head-of-run probe instead, and the staged block
                # would never be entered at all.
                nonlocal armed
                armed = True
                raise OSError("killpg denied")

            def stale_shot(self: Path, **kwargs: object) -> os.stat_result:
                # ESTALE is not in pathlib's ignored-errno set, so an `exists()`
                # probe would raise it straight out of the `finally` block,
                # skipping the report publish entirely.
                if armed and self.name == "shot_001" and self.parent == output:
                    raise OSError(116, "Stale file handle")
                return real_stat(self, **kwargs)

            terminate.side_effect = fail_cleanup
            with contextlib.ExitStack() as stack:
                self._reach_accept(output, stack)
                stack.enter_context(mock.patch.object(Path, "stat", stale_shot))
                report, code = run_smoke(self._accepting_args(output))
            persisted = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 1)
        # The run really did reach the accept line; this is the quarantine of a
        # published shot, not a refusal before anything ran.
        self.assertEqual(report["artifact"], {"states": 2, "events": 1})
        self.assertEqual(report["phase"], "complete")
        self.assertEqual(report["status"], "rejected")
        self.assertNotIn("accepted_shot", persisted)
        self.assertTrue(any("Stale file handle" in entry for entry in persisted["quarantine_errors"]))

    @mock.patch("scripts.smoke_physics_capture.subprocess.Popen")
    @mock.patch("scripts.smoke_physics_capture.terminate", return_value="terminated")
    @mock.patch("scripts.smoke_physics_capture.wait_for_listener")
    @mock.patch("scripts.smoke_physics_capture.start_display")
    @mock.patch("scripts.smoke_physics_capture.archive_details")
    def test_an_unprobeable_name_does_not_cost_a_prior_run_its_artifact(self, archive: mock.Mock, display: mock.Mock, listener: mock.Mock, _terminate: mock.Mock, popen: mock.Mock) -> None:
        """A blind probe on one name must not erase what was known about another.

        A self-referential symlink raises ELOOP from `stat` -- an ordinary
        unprivileged filesystem state. Aborting the whole scan on it discarded
        the identity of the real prior `shot_001`, after which this run
        quarantined an earlier run's evidence and reported `preexisting: []`.
        """
        process = popen.return_value
        process.pid = 4100
        display.return_value = (":205", process)
        archive.side_effect = self._staged_candidate
        listener.return_value = {"pid": 4100, "socket_inode": "77"}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            prior = output / "shot_001"
            prior.mkdir()
            (prior / "physics_state.jsonl").write_text("PRIOR RUN EVIDENCE\n", encoding="utf-8")
            (output / "shot_001.tmp").symlink_to(output / "shot_001.tmp")
            report, code = run_smoke(self._accepting_args(output))
            persisted = json.loads((output / "report.json").read_text(encoding="utf-8"))
            # The prior run's artifact is exactly where it was.
            self.assertEqual((prior / "physics_state.jsonl").read_text(encoding="utf-8"), "PRIOR RUN EVIDENCE\n")
            self.assertFalse((output / "invalid_attempts").exists())
        self.assertEqual(code, 1)
        self.assertEqual(persisted["unprobeable_artifacts"], ["shot_001.tmp"])
        self.assertEqual(persisted["preexisting_artifacts"], ["shot_001"])
        self.assertNotIn("quarantine", persisted)
        self.assertIn("shot_001.tmp", persisted["error"])

    @mock.patch("scripts.smoke_physics_capture.subprocess.Popen")
    @mock.patch("scripts.smoke_physics_capture.terminate", return_value="terminated")
    @mock.patch("scripts.smoke_physics_capture.wait_for_listener")
    @mock.patch("scripts.smoke_physics_capture.start_display")
    @mock.patch("scripts.smoke_physics_capture.archive_details")
    def test_a_prior_accepted_receipt_is_never_readable_while_this_run_is_in_flight(self, archive: mock.Mock, display: mock.Mock, listener: mock.Mock, _terminate: mock.Mock, popen: mock.Mock) -> None:
        """The designated path must not hold a stale accept mid-run or after a kill.

        The publish is atomic, which stops torn writes but leaves the previous
        receipt readable for the whole run. A downstream gate reading it then --
        or after this process is killed -- consumes an accept this run never
        earned. Displacing it up front is what closes that, and it survives a
        publish that fails for the same reason its unlink fallback would.
        """
        process = popen.return_value
        process.pid = 4100
        listener.return_value = {"pid": 4100, "socket_inode": "77"}
        observed: list[object] = []
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            receipt = output / "report.json"
            stale = {"status": "accepted", "accepted_shot": str(output / "shot_001"), "phase": "complete"}
            receipt.write_text(json.dumps(stale), encoding="utf-8")

            def observe_midrun(*_args: object, **_kwargs: object) -> tuple[str, object]:
                observed.append(json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else None)
                raise SmokeError("display unavailable")

            display.side_effect = observe_midrun
            archive.side_effect = self._staged_candidate
            report, code = run_smoke(self._accepting_args(output))
            persisted = json.loads(receipt.read_text(encoding="utf-8"))
            superseded = json.loads(Path(report["superseded_report"]).read_text(encoding="utf-8"))
        self.assertEqual(code, 1)
        # Mid-run, nothing was readable at the designated path.
        self.assertEqual(observed, [None])
        # The prior receipt was displaced, not destroyed.
        self.assertEqual(superseded, stale)
        self.assertEqual(persisted["status"], "rejected")
        self.assertNotIn("accepted_shot", persisted)

    @mock.patch("scripts.smoke_physics_capture.subprocess.Popen")
    @mock.patch("scripts.smoke_physics_capture.terminate", return_value="terminated")
    @mock.patch("scripts.smoke_physics_capture.wait_for_listener")
    @mock.patch("scripts.smoke_physics_capture.start_display")
    @mock.patch("scripts.smoke_physics_capture.archive_details")
    def test_a_dirty_output_namespace_is_refused_before_anything_launches(self, archive: mock.Mock, display: mock.Mock, listener: mock.Mock, _terminate: mock.Mock, _popen: mock.Mock) -> None:
        archive.side_effect = self._staged_candidate
        listener.return_value = {"pid": 4100, "socket_inode": "77"}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "shot_001").mkdir()
            report, code = run_smoke(self._listener_only_args(output))
            persisted = json.loads((output / "report.json").read_text(encoding="utf-8"))
            # The prior artifact is neither adopted nor moved.
            self.assertTrue((output / "shot_001").is_dir())
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "rejected")
        self.assertIn("already holds prior artifacts", persisted["error"])
        display.assert_not_called()

    def test_quarantine_moves_an_artifact_written_onto_a_previously_occupied_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            shot = output / "shot_001"
            shot.mkdir()
            preexisting, blind = smoke.preexisting_artifacts(output, ("shot_001",))
            self.assertEqual(blind, ())
            # `os.replace` succeeds onto an existing empty directory, so a name-keyed
            # exemption would let this run's artifact inherit the earlier run's skip.
            replacement = output / "replacement"
            replacement.mkdir()
            (replacement / "this_run.jsonl").write_text("{}\n", encoding="utf-8")
            replacement.replace(shot)
            report: dict[str, object] = {}
            smoke.quarantine_artifact(report, output, "shot_001", preexisting)
        self.assertEqual(report["quarantine"], [str(output / "invalid_attempts" / "shot_001")])

    def test_quarantine_never_raises_when_the_destination_is_occupied(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            holding = output / "invalid_attempts"
            holding.mkdir()
            for name in ("shot_001", "shot_001.1"):
                (holding / name).mkdir()
                (holding / name / "prior.jsonl").write_text("{}\n", encoding="utf-8")
            (output / "shot_001").mkdir()
            (output / "shot_001" / "this_run.jsonl").write_text("{}\n", encoding="utf-8")
            report: dict[str, object] = {}
            smoke.quarantine_artifact(report, output, "shot_001", {})
            self.assertEqual(report["quarantine"], [str(holding / "shot_001.2")])
            # Both prior quarantines are untouched.
            self.assertTrue((holding / "shot_001" / "prior.jsonl").is_file())
            self.assertTrue((holding / "shot_001.1" / "prior.jsonl").is_file())

    def test_report_publish_failure_leaves_no_stale_accepted_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            args = self._listener_only_args(output)
            stale = json.dumps({"status": "accepted", "accepted_shot": str(output / "shot_001")}) + "\n"
            args.report.write_text(stale, encoding="utf-8")
            real_replace = Path.replace

            def fail_on_report_publish(self: Path, target: Path) -> Path:
                if Path(target) == args.report:
                    raise OSError("publish denied")
                return real_replace(self, target)

            with mock.patch.object(Path, "replace", fail_on_report_publish):
                with mock.patch("scripts.smoke_physics_capture.start_display", side_effect=SmokeError("no display")):
                    report, code = run_smoke(args)
            # A failed publish must not leave the previous run's "accepted" receipt
            # at the designated path, where a downstream gate would consume it.
            self.assertFalse(args.report.exists())
            # Nor may it leave staging litter beside it.
            self.assertFalse(args.report.with_name(f"{args.report.name}.{os.getpid()}.tmp").exists())
            # `main` prints the returned report to stdout, so it has to agree with
            # the empty disk state rather than still claiming an accept.
            self.assertEqual(report["status"], "rejected")
            self.assertIn("report publish failed", report["publish_error"])
            # The original reason survives: stdout is now the only record of this
            # run, so overwriting `error` would destroy why it failed at all.
            self.assertEqual(report["error"], "no display")
            # The prior receipt was displaced rather than destroyed.
            self.assertEqual(json.loads(Path(report["superseded_report"]).read_text(encoding="utf-8")), json.loads(stale))
            self.assertEqual(code, 1)

    def test_report_carries_run_identity_so_staleness_is_detectable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._listener_only_args(Path(temporary))
            with mock.patch("scripts.smoke_physics_capture.start_display", side_effect=SmokeError("no display")):
                report, _code = run_smoke(args)
        self.assertEqual(report["run"]["pid"], os.getpid())
        self.assertIsInstance(report["run"]["started_unix_ns"], int)

    @mock.patch("scripts.smoke_physics_capture.subprocess.Popen")
    @mock.patch("scripts.smoke_physics_capture.terminate", side_effect=OSError("killpg denied"))
    @mock.patch("scripts.smoke_physics_capture.wait_for_listener")
    @mock.patch("scripts.smoke_physics_capture.start_display")
    @mock.patch("scripts.smoke_physics_capture.archive_details")
    def test_failed_quarantine_still_drops_the_accepted_shot_claim(self, archive: mock.Mock, display: mock.Mock, listener: mock.Mock, _terminate: mock.Mock, popen: mock.Mock) -> None:
        process = popen.return_value
        process.pid = 4100
        display.return_value = (":202", process)
        archive.side_effect = self._staged_candidate
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            published = output / "shot_001"

            def publish_during_the_run(*_args: object, **_kwargs: object) -> dict[str, object]:
                published.mkdir()
                (published / "physics_state.jsonl").write_text("{}\n", encoding="utf-8")
                return {"pid": 4100, "socket_inode": "77"}

            listener.side_effect = publish_during_the_run
            real_replace = Path.replace

            def refuse_quarantine(self: Path, target: Path) -> Path:
                if Path(target).parent.name == "invalid_attempts":
                    raise OSError(39, "Directory not empty")
                return real_replace(self, target)

            with mock.patch.object(Path, "replace", refuse_quarantine):
                report, code = run_smoke(self._listener_only_args(output))
            persisted = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "rejected")
        # The quarantine failed, so the artifact remains -- but the report must not
        # simultaneously claim rejection and name an accepted shot.
        self.assertNotIn("accepted_shot", persisted)
        self.assertEqual(len(persisted["quarantine_errors"]), 1)

    @mock.patch("scripts.smoke_physics_capture.subprocess.Popen")
    @mock.patch("scripts.smoke_physics_capture.terminate", return_value="terminated")
    @mock.patch("scripts.smoke_physics_capture.wait_for_listener", side_effect=SmokeError("bind failed"))
    @mock.patch("scripts.smoke_physics_capture.start_display")
    @mock.patch("scripts.smoke_physics_capture.archive_details")
    def test_both_prior_artifacts_are_named_in_the_refusal(self, archive: mock.Mock, display: mock.Mock, _listener: mock.Mock, _terminate: mock.Mock, popen: mock.Mock) -> None:
        process = popen.return_value
        process.pid = 4100
        display.return_value = (":202", process)
        archive.side_effect = self._staged_candidate
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for name in ("shot_001.tmp", "shot_001"):
                (output / name).mkdir()
                (output / name / "physics_state.jsonl").write_text("{}\n", encoding="utf-8")
            report, code = run_smoke(self._listener_only_args(output))
            persisted = json.loads((output / "report.json").read_text(encoding="utf-8"))
            # Neither prior artifact is adopted or moved.
            self.assertTrue((output / "shot_001" / "physics_state.jsonl").is_file())
            self.assertTrue((output / "shot_001.tmp" / "physics_state.jsonl").is_file())
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "rejected")
        self.assertEqual(sorted(persisted["preexisting_artifacts"]), ["shot_001", "shot_001.tmp"])
        self.assertNotIn("quarantine", persisted)

    def test_a_non_object_provenance_document_is_rejected_at_its_source(self) -> None:
        """A JSON array must reject by name, not by escaping `.get` as AttributeError.

        Absorbing `AttributeError` in `run_smoke`'s handler would also absorb
        every genuine `None.attr` and typo -- and `FrozenInstanceError`, which
        this repo's frozen dataclasses raise, is an `AttributeError` subclass.
        So the shape is checked where the document is parsed.
        """
        with tempfile.TemporaryDirectory() as temporary:
            expectation, stage_archive = build_candidate(Path(temporary) / "player")
            (expectation.root / "provenance.json").write_text("[]\n", encoding="utf-8")
            with open(expectation.runtime, "rb") as handle, mmap.mmap(handle.fileno(), 0, prot=mmap.PROT_READ):
                with self.assertRaisesRegex(ListenerBindingError, "provenance is list, not an object"):
                    _observe_listener_owners(((os.getpid(), self.socket_inode),), os.getpid(), expectation, stage_archive)

    def test_terminate_signals_the_group_even_after_the_leader_exited(self) -> None:
        """A reaped JVM must not spare the Unity player that outlives it.

        The player is a grandchild of the leader and holds the physics port. The
        prior `if process.poll() is None` guard skipped `killpg` entirely once
        the leader had exited, orphaning that player -- observed live as two
        stranded processes still holding port 2004 after a timed-out run, which
        then poisons every later run on the host.
        """
        process = mock.MagicMock()
        process.pid = 4100
        process.poll.return_value = 0
        process.returncode = 0
        members = mock.Mock(side_effect=[(4101,), ()])
        with mock.patch("scripts.smoke_physics_capture.os.killpg") as killpg:
            with mock.patch.object(smoke, "_process_group_members", members):
                self.assertEqual(smoke.terminate(process, drain_seconds=5.0, sleep=lambda _: None), "pid=4100:exit=0")
        killpg.assert_called_once_with(4100, signal.SIGTERM)
        # The group -- not the leader -- is what says the port was released, so
        # it has to be polled after the signal even though the leader is gone.
        self.assertEqual(members.call_count, 2)

    def test_terminate_escalates_to_sigkill_on_a_group_that_ignores_sigterm(self) -> None:
        """Waiting on the leader would report a clean stop while the player ran.

        A leader that is already reaped says nothing about the grandchild that
        holds the socket, so escalation is keyed on group membership and the
        receipt records that it happened.
        """
        process = mock.MagicMock()
        process.pid = 4100
        process.poll.return_value = 0
        process.returncode = 0
        clock = iter([0.0, 0.0, 11.0])
        with mock.patch("scripts.smoke_physics_capture.os.killpg") as killpg:
            with mock.patch.object(smoke, "_process_group_members", return_value=(4101,)):
                with mock.patch("scripts.smoke_physics_capture.time.monotonic", lambda: next(clock)):
                    receipt = smoke.terminate(process, drain_seconds=10.0, sleep=lambda _: None)
        self.assertEqual(receipt, "pid=4100:exit=0:group-escalated:group-residual=[4101]")
        self.assertEqual([call.args for call in killpg.call_args_list], [(4100, signal.SIGTERM), (4100, signal.SIGKILL)])

    def test_terminate_really_kills_a_grandchild_that_outlives_the_leader(self) -> None:
        """The mocked cases pin the logic; this one pins the outcome.

        `sh` execs into a short-lived child, so the group leader is gone within
        a fraction of a second while the backgrounded process it spawned stays
        in the same group -- the exact topology of the JVM and the Unity player
        that strands port 2004.
        """
        process = subprocess.Popen(["sh", "-c", "sleep 300 & exec sleep 0.2"], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: os.killpg(process.pid, signal.SIGKILL) if smoke._process_group_members(process.pid) else None)
        deadline = time.monotonic() + 10.0
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertIsNotNone(process.poll(), "leader did not exit; fixture is not exercising the orphan case")
        self.assertTrue(smoke._process_group_members(process.pid), "grandchild did not survive the leader; fixture is inert")
        receipt = smoke.terminate(process, drain_seconds=5.0)
        self.assertNotIn("group-residual", receipt)
        self.assertEqual(smoke._process_group_members(process.pid), ())

    def test_terminate_tolerates_an_empty_process_group(self) -> None:
        process = mock.MagicMock()
        process.pid = 4100
        process.poll.return_value = 0
        process.returncode = 0
        with mock.patch("scripts.smoke_physics_capture.os.killpg", side_effect=ProcessLookupError()):
            with mock.patch.object(smoke, "_process_group_members", return_value=()):
                self.assertEqual(smoke.terminate(process, sleep=lambda _: None), "pid=4100:exit=0")

    def test_a_process_group_census_excludes_zombies(self) -> None:
        """A reaped-pending leader holds no socket and must not force a SIGKILL.

        Counting it would make every clean run look like a group that ignored
        SIGTERM, escalate needlessly, and burn the full drain window each time.
        """
        process = subprocess.Popen(["sh", "-c", "exit 0"], start_new_session=True)
        deadline = time.monotonic() + 10.0
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        try:
            # Not yet reaped: the pid is still present in /proc, as state Z.
            self.assertEqual(smoke._process_group_members(process.pid), ())
        finally:
            process.wait(timeout=5.0)

    @mock.patch("scripts.smoke_physics_capture.subprocess.Popen")
    @mock.patch("scripts.smoke_physics_capture.terminate", return_value="pid=1:exit=0")
    @mock.patch("scripts.smoke_physics_capture.wait_for_listener")
    @mock.patch("scripts.smoke_physics_capture.start_display")
    @mock.patch("scripts.smoke_physics_capture.archive_details")
    def test_a_failing_agent_disconnect_is_recorded_not_raised(self, archive: mock.Mock, display: mock.Mock, listener: mock.Mock, _terminate: mock.Mock, popen: mock.Mock) -> None:
        """Teardown must not become the run's only recorded reason for failure.

        A bare `disconnect()` in the outer `finally` replaces whatever exception
        was in flight, so the real rejection survives only as `__context__` --
        which nothing reads. It is recorded exactly like engine/xvfb cleanup.
        """
        process = popen.return_value
        process.pid = 4100
        display.return_value = (":203", process)
        archive.side_effect = self._staged_candidate
        listener.side_effect = smoke.ListenerBindingError("listener executable differs")
        self.connect.return_value.disconnect.side_effect = OSError("socket already closed")
        with tempfile.TemporaryDirectory() as temporary:
            report, code = run_smoke(self._listener_only_args(Path(temporary)))
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "rejected")
        # The binding failure is preserved; the teardown failure is additive.
        self.assertEqual(report["error"], "listener executable differs")
        self.assertEqual(report["phase"], "bind-physics-listener")
        cleanup = report["cleanup"]
        assert isinstance(cleanup, dict)
        self.assertIn("socket already closed", str(cleanup["agent"]))
        self.assertIn("agent cleanup-failed: socket already closed", report["cleanup_failures"])

    @mock.patch("scripts.smoke_physics_capture.subprocess.Popen")
    @mock.patch("scripts.smoke_physics_capture.terminate", return_value="pid=1:exit=0")
    @mock.patch("scripts.smoke_physics_capture.wait_for_listener")
    @mock.patch("scripts.smoke_physics_capture.start_display")
    @mock.patch("scripts.smoke_physics_capture.archive_details")
    def test_a_clean_run_records_an_agent_teardown_receipt(self, archive: mock.Mock, display: mock.Mock, listener: mock.Mock, _terminate: mock.Mock, popen: mock.Mock) -> None:
        process = popen.return_value
        process.pid = 4100
        display.return_value = (":203", process)
        archive.side_effect = self._staged_candidate
        listener.return_value = {"pid": 4100, "socket_inode": "77"}
        with tempfile.TemporaryDirectory() as temporary:
            report, code = run_smoke(self._listener_only_args(Path(temporary)))
        self.assertEqual(code, 0)
        cleanup = report["cleanup"]
        assert isinstance(cleanup, dict)
        self.assertEqual(cleanup["agent"], "disconnected")
        self.assertNotIn("cleanup_failures", report)

    @mock.patch("scripts.smoke_physics_capture.subprocess.Popen")
    @mock.patch("scripts.smoke_physics_capture.terminate", return_value="pid=1:exit=0")
    @mock.patch("scripts.smoke_physics_capture.wait_for_listener")
    @mock.patch("scripts.smoke_physics_capture.start_display")
    @mock.patch("scripts.smoke_physics_capture.archive_details")
    def test_the_port_clear_check_waits_out_a_draining_player(self, archive: mock.Mock, display: mock.Mock, listener: mock.Mock, _terminate: mock.Mock, popen: mock.Mock) -> None:
        """The player unbinds after the group signal, so one immediate read races it.

        Reading `/proc/net/tcp` once, right after `terminate`, turns a healthy
        run into `physics listener remained after cleanup`. The grace window only
        ever delays a rejection -- a port that never drains still rejects, which
        the second half of this test pins.
        """
        process = popen.return_value
        process.pid = 4100
        display.return_value = (":203", process)
        archive.side_effect = self._staged_candidate
        listener.return_value = {"pid": 4100, "socket_inode": "77"}
        draining = iter((("77",), ("77",), ()))
        draining_args = self._listener_only_args(Path("/unused"))
        draining_args.port_grace_seconds = 30.0
        with mock.patch("scripts.smoke_physics_capture._proc_net_listeners", side_effect=lambda _port: next(draining)):
            with mock.patch("scripts.smoke_physics_capture.time.sleep"):
                with tempfile.TemporaryDirectory() as temporary:
                    draining_args.output_dir = Path(temporary)
                    draining_args.report = Path(temporary) / "report.json"
                    report, code = run_smoke(draining_args)
        self.assertEqual(code, 0)
        cleanup = report["cleanup"]
        assert isinstance(cleanup, dict)
        self.assertTrue(cleanup["physics_port_clear"])
        # The second half must not spend the grace window in real seconds: the
        # clock is driven past the deadline instead of waited out, so a port that
        # never drains still rejects and the suite stays fast.
        stuck_args = self._listener_only_args(Path("/unused"))
        stuck_args.port_grace_seconds = 30.0
        clock = iter((0.0, 31.0))
        with mock.patch("scripts.smoke_physics_capture._proc_net_listeners", return_value=("77",)):
            with mock.patch("scripts.smoke_physics_capture.time.sleep"):
                with mock.patch("scripts.smoke_physics_capture.time.monotonic", lambda: next(clock)):
                    with tempfile.TemporaryDirectory() as temporary:
                        stuck_args.output_dir = Path(temporary)
                        stuck_args.report = Path(temporary) / "report.json"
                        stuck, stuck_code = run_smoke(stuck_args)
        self.assertEqual(stuck_code, 1)
        self.assertIn("physics listener remained after cleanup", stuck["cleanup_failures"])


if __name__ == "__main__":
    unittest.main()
