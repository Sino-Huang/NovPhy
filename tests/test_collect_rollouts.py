import io
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from scripts.prepare_rollout_dataset import (
    CollectionOptions,
    LevelEntry,
    PhysicsCaptureProvenance,
    PlannedEpisode,
    WorkerSpec,
    _collection_command_lines,
)

from scripts.collect_rollouts import (
    PRE_DRAG_OVERLAY_TEXT,
    _FinalizedPhysicsBridge,
    _action_guide_points,
    _launch_guide_points,
    action_to_shot,
    capture_desktop_rollout,
    capture_physics_rollout,
    cleanup_incomplete_physics_attempts,
    recover_physics_capture_attempts,
    collect_fresh_engine_attempt,
    collect_fresh_engine_rollouts,
    collect_rollouts,
    build_parser,
    format_action_overlay_text,
    load_actions_from_action_log,
    prepare_rollout_video_frames,
    realized_coverage_strata,
    slingshot_reference_point_from_symbolic_state,
    main,
    RolloutCollectionError,
    select_level_in_display,
    stop_owned_engine,
    validate_rollout_artifact,
    write_action_plan,
)
from scripts.rollout_artifacts import validate_physics_shot_artifact
from scripts.rollout_validation_types import PhysicsArtifactError
from scripts.physics_capture_contract import PhysicsContractError, load_physics_capture
from src.webui.bridge import PhysicsCaptureV1Failure
from scripts.physics_rollout_contract import MAX_TOTAL_BYTES

PHYSICS_FIXTURES = Path(__file__).parent / "fixtures" / "physics_capture_v1"

class PhysicsCapturePersistenceTests(unittest.TestCase):
    @staticmethod
    def _records():
        states = [json.loads(line) for line in (PHYSICS_FIXTURES / "physics_state.jsonl").read_text(encoding="utf-8").splitlines()]
        events = [json.loads(line) for line in (PHYSICS_FIXTURES / "physics_events.jsonl").read_text(encoding="utf-8").splitlines()]
        for record in states + events:
            record["shot_id"] = "shot_000"
        states[1]["rgb_frame"].update({"relative_path": "frames/frame_000000.png", "width_pixels": 4, "height_pixels": 3})
        return states, events

    @staticmethod
    def _png():
        from PIL import Image

        data = io.BytesIO()
        Image.new("RGB", (4, 3), (30, 20, 10)).save(data, format="PNG")
        return data.getvalue()

    def test_finalized_physics_bridge_retries_only_pending_recorder_batches(self):
        class Bridge:
            calls = 0

            def get_physics_capture_v1(self):
                self.calls += 1
                if self.calls == 1:
                    raise PhysicsCaptureV1Failure(4, "no finalized recorder batch")
                return "capture"

        bridge = Bridge()
        sleeps = []
        finalized = _FinalizedPhysicsBridge(
            bridge,
            deadline_seconds=30,
            clock=lambda: 0,
            sleeper=sleeps.append,
        )

        self.assertEqual(finalized.get_physics_capture_v1(), "capture")
        self.assertEqual(bridge.calls, 2)
        self.assertEqual(sleeps, [0.25])

        class PermanentFailure:
            def get_physics_capture_v1(self):
                raise PhysicsCaptureV1Failure(3, "capture failed")

        with self.assertRaises(PhysicsCaptureV1Failure):
            _FinalizedPhysicsBridge(
                PermanentFailure(),
                deadline_seconds=30,
                clock=lambda: 0,
                sleeper=lambda _: None,
            ).get_physics_capture_v1()

    def test_persistence_error_supports_standard_exception_traceback_state(self):
        from scripts.physics_rollout_contract import PersistenceErrorCode, PhysicsPersistenceError

        error = PhysicsPersistenceError(PersistenceErrorCode.MALFORMED_CAPTURE, "invalid")
        error.__traceback__ = None

        self.assertIsNone(error.__traceback__)

    def test_valid_capture_has_closed_sidecars_and_exact_pair(self):
        from src.webui.bridge import PhysicsCaptureV1
        records, events = self._records()
        png = self._png()
        class Bridge:
            def get_physics_capture_v1(self):
                return PhysicsCaptureV1(png, records[1], tuple(events))
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot_000.tmp"
            metadata = capture_physics_rollout(Bridge(), shot, target_fps=1, duration_seconds=1, max_frames=1, state_header=records[0], player_sha256="a" * 64, protocol_sha256="b" * 64, archive_sha256="c" * 64, clock=lambda: 0.0, sleeper=lambda _seconds: None)
            self.assertTrue(metadata["sidecars_closed"])
            self.assertTrue(validate_rollout_artifact(shot, capture_contract="physics_capture_v1")["accepted"])
            self.assertEqual(metadata["frame_checksums"][0]["sha256"], hashlib.sha256(png).hexdigest())

    def test_persistence_accepts_immutable_nested_mapping_from_real_decoder(self):
        from src.webui.bridge import ScienceBirdsBridge, encode_physics_capture_v1

        records, _ = self._records()
        png = self._png()
        response = bytearray(encode_physics_capture_v1(png, records[1], []))

        class Socket:
            def settimeout(self, _timeout):
                pass

            def connect(self, _address):
                pass

            def sendall(self, data):
                self.request = data

            def recv(self, size):
                chunk = response[:size]
                del response[:size]
                return bytes(chunk)

            def close(self):
                pass

        socket = Socket()
        bridge = ScienceBirdsBridge(socket_factory=lambda *_args: socket)
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot_000.tmp"
            metadata = capture_physics_rollout(
                bridge,
                shot,
                target_fps=1,
                duration_seconds=1,
                max_frames=1,
                state_header=records[0],
                player_sha256="a" * 64,
                protocol_sha256="b" * 64,
                archive_sha256="c" * 64,
                clock=lambda: 0.0,
                sleeper=lambda _seconds: None,
            )

        self.assertEqual(socket.request, b"\x46")
        self.assertEqual(metadata["frame_count"], 1)

    def test_physics_capture_requests_are_paced_at_target_fps(self):
        from src.webui.bridge import PhysicsCaptureV1

        # Given: a deterministic monotonic clock and a two-frame request-70 capture.
        records, _ = self._records()
        png = self._png()
        virtual_time = 0.0
        request_times = []
        sleep_durations = []

        class Bridge:
            request_count = 0

            def get_physics_capture_v1(self):
                state = json.loads(json.dumps(records[1]))
                request_times.append(virtual_time)
                state["sequence"] += self.request_count
                state["render_frame"] += self.request_count
                state["fixed_step"] += self.request_count
                state["render_time"] += self.request_count / 2
                state["fixed_time"] += self.request_count / 2
                state["rgb_frame"]["render_frame"] = state["render_frame"]
                self.request_count += 1
                return PhysicsCaptureV1(png, state, ())

        def clock():
            return virtual_time

        def sleeper(seconds):
            nonlocal virtual_time
            sleep_durations.append(seconds)
            virtual_time += seconds

        # When: capture runs at two frames per second without real sleeping.
        with TemporaryDirectory() as temporary:
            capture_physics_rollout(
                Bridge(),
                Path(temporary) / "shot_000.tmp",
                target_fps=2,
                duration_seconds=1,
                max_frames=2,
                state_header=records[0],
                clock=clock,
                sleeper=sleeper,
                player_sha256="a" * 64,
                protocol_sha256="b" * 64,
                archive_sha256="c" * 64,
            )

        # Then: request timestamps, not implementation calls, prove pacing.
        self.assertEqual(request_times, [0.0, 0.5])
        self.assertEqual(sleep_durations, [0.5])

    def test_empty_event_stream_is_a_valid_closed_sidecar(self):
        from src.webui.bridge import PhysicsCaptureV1

        records, _ = self._records()
        png = self._png()
        class Bridge:
            def get_physics_capture_v1(self):
                return PhysicsCaptureV1(png, records[1], ())
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot_000.tmp"
            capture_physics_rollout(Bridge(), shot, target_fps=1, duration_seconds=1, max_frames=1, state_header=records[0], player_sha256="a" * 64, protocol_sha256="b" * 64, archive_sha256="c" * 64)
            self.assertEqual((shot / "physics_events.jsonl").read_bytes(), b"")
            self.assertTrue(validate_rollout_artifact(shot, capture_contract="physics_capture_v1")["accepted"])

    def test_physics_capture_uses_recorder_backed_action_before_request_70(self):
        from src.webui.bridge import PhysicsCaptureV1

        records, events = self._records()
        png = self._png()
        call_order = []

        class Bridge(FakeBridge):
            request_count = 0

            def shoot(self, *args, **kwargs):
                raise AssertionError("physics capture must not use legacy shoot")

            def shoot_and_record_ground_truth(self, x, y, tap_time=0, release_time=0, frequency=1):
                call_order.append(("recorder-action", x, y, tap_time, release_time, frequency))
                return 1

            def get_physics_capture_v1(self):
                self.request_count += 1
                call_order.append("initial-request-70" if self.request_count == 1 else "post-request-70")
                return PhysicsCaptureV1(
                    png,
                    records[1] if self.request_count == 1 else records[2],
                    () if self.request_count == 1 else tuple(events[:1]),
                )

        action = {"coordinate_frame": "absolute", "drag_start": [100, 200], "drag_release": [130, 150], "tapTime": 70, "holdTime": 600}
        guard = {"pre_shot_image": None, "pre_shot_sample": None, "post_recovery_protocol_state": {}, "recovery_action": None, "pre_shot_guard": {"status": "accepted", "invalid_reason": None}}
        with TemporaryDirectory() as temporary, patch("scripts.collect_rollouts._run_pre_shot_guard", return_value=guard):
            manifest = collect_rollouts(Bridge(), Path(temporary), [action], target_fps=1, duration_seconds=1, max_frames=2, anchor_actions=False, video_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0), physics_capture_v1=True, physics_player_sha256="a" * 64, physics_protocol_sha256="b" * 64, physics_archive_sha256="c" * 64)

        self.assertEqual(call_order[0], "initial-request-70")
        self.assertEqual(call_order[1][0], "recorder-action")
        self.assertEqual(call_order[2], "post-request-70")
        self.assertEqual(manifest["rollout_count"], 1)

    def test_missing_provenance_and_sidecar_fail_closed(self):
        from src.webui.bridge import PhysicsCaptureV1

        records, events = self._records()
        png = self._png()
        class Bridge:
            def get_physics_capture_v1(self):
                return PhysicsCaptureV1(png, records[1], tuple(events))
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot_000.tmp"
            capture_physics_rollout(Bridge(), shot, target_fps=1, duration_seconds=1, max_frames=1, state_header=records[0], player_sha256="a" * 64, protocol_sha256="b" * 64, archive_sha256="c" * 64)
            metadata = json.loads((shot / "metadata.json").read_text(encoding="utf-8"))
            del metadata["archive_sha256"]
            (shot / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            self.assertFalse(validate_rollout_artifact(shot, capture_contract="physics_capture_v1")["accepted"])

    def test_fixed_name_state_sidecar_symlink_outside_shot_is_rejected(self):
        from src.webui.bridge import PhysicsCaptureV1

        records, events = self._records()
        png = self._png()

        class Bridge:
            def get_physics_capture_v1(self):
                return PhysicsCaptureV1(png, records[1], tuple(events))

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = root / "shot_000.tmp"
            capture_physics_rollout(
                Bridge(), shot, target_fps=1, duration_seconds=1, max_frames=1,
                state_header=records[0], player_sha256="a" * 64,
                protocol_sha256="b" * 64, archive_sha256="c" * 64,
            )
            state_path = shot / "physics_state.jsonl"
            external_state = root / "external-state.jsonl"
            state_path.replace(external_state)
            state_path.symlink_to(external_state)

            with self.assertRaisesRegex(PhysicsArtifactError, "symlink|outside"):
                validate_physics_shot_artifact(shot)
            self.assertFalse(validate_rollout_artifact(shot, capture_contract="physics_capture_v1")["accepted"])

    def test_physics_frame_symlink_is_rejected_before_image_open(self):
        from src.webui.bridge import PhysicsCaptureV1

        records, events = self._records()
        png = self._png()

        class Bridge:
            def get_physics_capture_v1(self):
                return PhysicsCaptureV1(png, records[1], tuple(events))

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = root / "shot_000.tmp"
            capture_physics_rollout(
                Bridge(), shot, target_fps=1, duration_seconds=1, max_frames=1,
                state_header=records[0], player_sha256="a" * 64,
                protocol_sha256="b" * 64, archive_sha256="c" * 64,
            )
            frame_path = shot / "frames" / "frame_000000.png"
            external_frame = root / "external-frame.png"
            frame_path.replace(external_frame)
            frame_path.symlink_to(external_frame)

            with self.assertRaisesRegex(PhysicsArtifactError, "symlink|outside"):
                validate_physics_shot_artifact(shot)

    def test_physics_frame_relative_path_cannot_escape_shot_root(self):
        from src.webui.bridge import PhysicsCaptureV1

        records, events = self._records()
        png = self._png()

        class Bridge:
            def get_physics_capture_v1(self):
                return PhysicsCaptureV1(png, records[1], tuple(events))

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = root / "shot_000.tmp"
            capture_physics_rollout(
                Bridge(), shot, target_fps=1, duration_seconds=1, max_frames=1,
                state_header=records[0], player_sha256="a" * 64,
                protocol_sha256="b" * 64, archive_sha256="c" * 64,
            )
            state_path = shot / "physics_state.jsonl"
            state_records = [json.loads(line) for line in state_path.read_text(encoding="utf-8").splitlines()]
            state_records[1]["rgb_frame"]["relative_path"] = "../external-frame.png"
            state_path.write_text(
                "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in state_records),
                encoding="utf-8",
            )
            frame_path = shot / "frames" / "frame_000000.png"
            frame_path.replace(root / "external-frame.png")

            with self.assertRaisesRegex(PhysicsArtifactError, "outside"):
                validate_physics_shot_artifact(shot)

    def test_checked_sidecar_and_frame_cannot_be_swapped_to_external_symlink(self):
        from scripts import physics_artifact_validation
        from src.webui.bridge import PhysicsCaptureV1

        records, events = self._records()
        png = self._png()

        class Bridge:
            def get_physics_capture_v1(self):
                return PhysicsCaptureV1(png, records[1], tuple(events))

        for relative_path in (Path("physics_state.jsonl"), Path("frames/frame_000000.png")):
            with self.subTest(relative_path=relative_path), TemporaryDirectory() as temporary:
                root = Path(temporary)
                shot = root / "shot_000.tmp"
                capture_physics_rollout(
                    Bridge(), shot, target_fps=1, duration_seconds=1, max_frames=1,
                    state_header=records[0], player_sha256="a" * 64,
                    protocol_sha256="b" * 64, archive_sha256="c" * 64,
                )
                target = shot / relative_path
                external = root / f"external-{target.name}"
                shutil.copy2(target, external)
                original_confined_file = physics_artifact_validation._confined_file
                swapped = False

                def swap_after_check(path, confined_root):
                    nonlocal swapped
                    checked = original_confined_file(path, confined_root)
                    if path == target:
                        target.unlink()
                        target.symlink_to(external)
                        swapped = True
                    return checked

                with (
                    patch("scripts.physics_artifact_validation._confined_file", side_effect=swap_after_check),
                    self.assertRaisesRegex(PhysicsArtifactError, "symlink|changed|outside"),
                ):
                    validate_physics_shot_artifact(shot)
                self.assertTrue(swapped, "the regression did not exercise the post-check swap")

    def test_oversized_sidecars_are_rejected_before_jsonl_reader_runs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "physics_state.jsonl"
            event_path = root / "physics_events.jsonl"
            with state_path.open("wb") as stream:
                stream.truncate(MAX_TOTAL_BYTES + 1)
            event_path.touch()

            with (
                patch(
                    "scripts.physics_capture_parsing._read_jsonl",
                    side_effect=AssertionError("oversized sidecar reached JSONL reader"),
                ),
                self.assertRaisesRegex(PhysicsContractError, "byte|limit|size"),
            ):
                load_physics_capture(state_path, event_path)

    def test_physics_artifact_hashing_does_not_materialize_whole_files(self):
        from src.webui.bridge import PhysicsCaptureV1

        records, events = self._records()
        png = self._png()

        class Bridge:
            def get_physics_capture_v1(self):
                return PhysicsCaptureV1(png, records[1], tuple(events))

        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot_000.tmp"
            capture_physics_rollout(
                Bridge(), shot, target_fps=1, duration_seconds=1, max_frames=1,
                state_header=records[0], player_sha256="a" * 64,
                protocol_sha256="b" * 64, archive_sha256="c" * 64,
            )

            with patch.object(Path, "read_bytes", side_effect=AssertionError("whole-file hash read")):
                summary = validate_physics_shot_artifact(shot)

            self.assertEqual(summary.state_count, 1)

    def test_physics_artifact_validation_does_not_use_whole_file_text_reads(self):
        from src.webui.bridge import PhysicsCaptureV1

        records, events = self._records()
        png = self._png()

        class Bridge:
            def get_physics_capture_v1(self):
                return PhysicsCaptureV1(png, records[1], tuple(events))

        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot_000.tmp"
            capture_physics_rollout(
                Bridge(), shot, target_fps=1, duration_seconds=1, max_frames=1,
                state_header=records[0], player_sha256="a" * 64,
                protocol_sha256="b" * 64, archive_sha256="c" * 64,
            )

            with patch.object(Path, "read_text", side_effect=AssertionError("whole-file text read")):
                summary = validate_physics_shot_artifact(shot)

            self.assertEqual(summary.state_count, 1)

    def test_corrupt_completed_attempt_is_moved_to_quarantine(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            completed = root / "shot_001"
            completed.mkdir()
            (completed / "metadata.json").write_text("{}", encoding="utf-8")
            recovery = recover_physics_capture_attempts(root)
            self.assertEqual(recovery.quarantined, ("invalid_attempts/shot_001_recovered_01",))
            self.assertFalse(completed.exists())
            self.assertTrue((root / recovery.quarantined[0] / "metadata.json").is_file())

    def test_completed_shot_with_nondirectory_frames_is_quarantined(self):
        from src.webui.bridge import PhysicsCaptureV1

        records, events = self._records()
        png = self._png()

        class Bridge:
            def get_physics_capture_v1(self):
                return PhysicsCaptureV1(png, records[1], tuple(events))

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            completed = root / "shot_001"
            capture_physics_rollout(
                Bridge(), completed, target_fps=1, duration_seconds=1, max_frames=1,
                state_header=records[0], player_sha256="a" * 64,
                protocol_sha256="b" * 64, archive_sha256="c" * 64,
            )
            shutil.rmtree(completed / "frames")
            (completed / "frames").write_text("not a directory", encoding="utf-8")

            recovery = recover_physics_capture_attempts(root)

            self.assertEqual(recovery.quarantined, ("invalid_attempts/shot_001_recovered_01",))
            self.assertFalse(completed.exists())
            self.assertTrue((root / recovery.quarantined[0] / "frames").is_file())

    def test_actual_collector_promotes_only_accepted_attempt(self):
        from src.webui.bridge import PhysicsCaptureV1

        records, events = self._records()
        png = self._png()
        class Bridge(FakeBridge):
            request_count = 0

            def get_physics_capture_v1(self):
                self.request_count += 1
                return PhysicsCaptureV1(
                    png,
                    records[1] if self.request_count == 1 else records[2],
                    () if self.request_count == 1 else tuple(events[:1]),
                )
        action = {"coordinate_frame": "absolute", "drag_start": [100, 200], "drag_release": [130, 150], "tapTime": 70, "holdTime": 600}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            guard = {"pre_shot_image": None, "pre_shot_sample": None, "post_recovery_protocol_state": {}, "recovery_action": None, "pre_shot_guard": {"status": "accepted", "invalid_reason": None}}
            with patch("scripts.collect_rollouts._run_pre_shot_guard", return_value=guard):
                manifest = collect_rollouts(Bridge(), root, [action], target_fps=1, duration_seconds=1, max_frames=2, anchor_actions=False, video_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0), physics_capture_v1=True, physics_player_sha256="a" * 64, physics_protocol_sha256="b" * 64, physics_archive_sha256="c" * 64)
            self.assertTrue((root / "shot_001").is_dir(), manifest)
            self.assertFalse((root / "shot_001.tmp").exists())
            self.assertEqual(manifest["capture_contract"]["archive_sha256"], "c" * 64)

    def test_second_collection_reuses_valid_completed_shot_without_bridge_calls(self):
        from src.webui.bridge import PhysicsCaptureV1

        records, events = self._records()
        png = self._png()

        class Bridge(FakeBridge):
            request_count = 0

            def get_physics_capture_v1(self):
                self.request_count += 1
                return PhysicsCaptureV1(
                    png,
                    records[1] if self.request_count % 2 == 1 else records[2],
                    () if self.request_count % 2 == 1 else tuple(events[:1]),
                )

        action = {"coordinate_frame": "absolute", "drag_start": [100, 200], "drag_release": [130, 150], "tapTime": 70, "holdTime": 600}
        guard = {"pre_shot_image": None, "pre_shot_sample": None, "post_recovery_protocol_state": {}, "recovery_action": None, "pre_shot_guard": {"status": "accepted", "invalid_reason": None}}
        with TemporaryDirectory() as temporary, patch("scripts.collect_rollouts._run_pre_shot_guard", return_value=guard):
            root = Path(temporary)
            bridge = Bridge()
            kwargs = dict(target_fps=1, duration_seconds=1, max_frames=2, anchor_actions=False, video_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0), physics_capture_v1=True, physics_player_sha256="a" * 64, physics_protocol_sha256="b" * 64, physics_archive_sha256="c" * 64)
            collect_rollouts(bridge, root, [action], **kwargs)
            shot = root / "shot_001"
            first_metadata = (shot / "metadata.json").read_bytes()
            first_request_count = bridge.request_count
            first_shot_count = len(bridge.shots)

            manifest = collect_rollouts(bridge, root, [action], **kwargs)

            self.assertEqual((shot / "metadata.json").read_bytes(), first_metadata)
            self.assertEqual(bridge.request_count, first_request_count)
            self.assertEqual(len(bridge.shots), first_shot_count)
            self.assertEqual(manifest["rollout_count"], 1)
            self.assertFalse((root / "shot_001.tmp").exists())

    def test_physics_capture_routes_gameplay_action_and_request_70_to_separate_bridges(self):
        from src.webui.bridge import PhysicsCaptureV1

        records, events = self._records()
        png = self._png()

        class GameplayBridge(FakeBridge):
            def get_physics_capture_v1(self):
                raise AssertionError("request 70 must not use the gameplay connection")

        class PhysicsBridge:
            request_count = 0

            def get_physics_capture_v1(self):
                self.request_count += 1
                return PhysicsCaptureV1(
                    png,
                    records[1] if self.request_count == 1 else records[2],
                    () if self.request_count == 1 else tuple(events[:1]),
                )

        guard = {"pre_shot_image": None, "pre_shot_sample": None, "post_recovery_protocol_state": {}, "recovery_action": None, "pre_shot_guard": {"status": "accepted", "invalid_reason": None}}
        with TemporaryDirectory() as temporary, patch("scripts.collect_rollouts._run_pre_shot_guard", return_value=guard):
            gameplay = GameplayBridge()
            physics = PhysicsBridge()
            collect_rollouts(
                gameplay,
                Path(temporary),
                [{"coordinate_frame": "absolute", "release": [130, 150], "tapTime": 70}],
                target_fps=1,
                duration_seconds=1,
                max_frames=2,
                anchor_actions=False,
                video_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0),
                physics_capture_v1=True,
                physics_bridge=physics,
                physics_player_sha256="a" * 64,
                physics_protocol_sha256="b" * 64,
                physics_archive_sha256="c" * 64,
            )

        self.assertEqual(len(gameplay.shots), 1)
        self.assertEqual(physics.request_count, 2)

    def test_temporary_shot_symlink_is_removed_before_any_external_write(self):
        from src.webui.bridge import PhysicsCaptureV1

        records, events = self._records()
        png = self._png()

        class Bridge(FakeBridge):
            request_count = 0

            def get_physics_capture_v1(self):
                self.request_count += 1
                return PhysicsCaptureV1(
                    png,
                    records[1] if self.request_count == 1 else records[2],
                    () if self.request_count == 1 else tuple(events[:1]),
                )

        guard = {"pre_shot_image": None, "pre_shot_sample": None, "post_recovery_protocol_state": {}, "recovery_action": None, "pre_shot_guard": {"status": "accepted", "invalid_reason": None}}
        with TemporaryDirectory() as temporary, patch("scripts.collect_rollouts._run_pre_shot_guard", return_value=guard):
            root = Path(temporary)
            external = root / "external"
            external.mkdir()
            (root / "shot_001.tmp").symlink_to(external, target_is_directory=True)

            collect_rollouts(
                Bridge(),
                root,
                [{"coordinate_frame": "absolute", "release": [130, 150], "tapTime": 70}],
                target_fps=1,
                duration_seconds=1,
                max_frames=2,
                anchor_actions=False,
                video_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0),
                physics_capture_v1=True,
                physics_player_sha256="a" * 64,
                physics_protocol_sha256="b" * 64,
                physics_archive_sha256="c" * 64,
            )

            self.assertEqual(tuple(external.iterdir()), ())
            self.assertTrue((root / "shot_001").is_dir())
            self.assertFalse((root / "shot_001.tmp").exists())

    def test_temporary_sidecar_symlink_is_rejected_without_truncating_external_file(self):
        from src.webui.bridge import PhysicsCaptureV1

        records, events = self._records()
        png = self._png()

        class Bridge:
            def get_physics_capture_v1(self):
                return PhysicsCaptureV1(png, records[1], tuple(events))

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            external = root / "external-state.jsonl"
            external.write_text("outside\n", encoding="utf-8")
            shot = root / "shot_001.tmp"
            shot.mkdir()
            (shot / "physics_state.jsonl").symlink_to(external)

            with self.assertRaisesRegex(RolloutCollectionError, "output|symlink|confined"):
                capture_physics_rollout(
                    Bridge(), shot, target_fps=1, duration_seconds=1, max_frames=1,
                    state_header=records[0], player_sha256="a" * 64,
                    protocol_sha256="b" * 64, archive_sha256="c" * 64,
                )

            self.assertEqual(external.read_text(encoding="utf-8"), "outside\n")

    def test_metadata_finalization_stays_on_trusted_shot_after_root_symlink_swap(self):
        from src.webui.bridge import PhysicsCaptureV1

        # Given: request 70 swaps the shot pathname only after descriptor-backed children exist.
        records, events = self._records()
        png = self._png()
        sentinel = b"external metadata sentinel\n"
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = root / "shot_001.tmp"
            trusted_shot = root / "trusted-shot-after-swap"
            external = root / "external"
            external.mkdir()
            external_metadata = external / "metadata.json"
            external_metadata.write_bytes(sentinel)
            swap_observations = []

            class Bridge:
                def get_physics_capture_v1(self):
                    swap_observations.append(
                        (
                            (shot / "physics_state.jsonl").is_file(),
                            (shot / "physics_events.jsonl").is_file(),
                            (shot / "frames").is_dir(),
                            (shot / "metadata.json").exists(),
                        )
                    )
                    shot.replace(trusted_shot)
                    shot.symlink_to(external, target_is_directory=True)
                    return PhysicsCaptureV1(png, records[1], tuple(events))

            # When: persistence finalizes metadata after the root pathname has been redirected.
            metadata = capture_physics_rollout(
                Bridge(),
                shot,
                target_fps=1,
                duration_seconds=1,
                max_frames=1,
                state_header=records[0],
                player_sha256="a" * 64,
                protocol_sha256="b" * 64,
                archive_sha256="c" * 64,
                clock=lambda: 0.0,
                sleeper=lambda _seconds: None,
            )

            # Then: finalization remains descriptor-confined and never follows the replacement symlink.
            self.assertEqual(swap_observations, [(True, True, True, False)])
            with self.subTest("external sentinel"):
                self.assertEqual(external_metadata.read_bytes(), sentinel)
            with self.subTest("trusted metadata"):
                self.assertEqual(
                    (trusted_shot / "metadata.json").read_bytes(),
                    json.dumps(metadata, indent=2).encode("utf-8"),
                )

    def test_accepted_physics_shot_is_complete_before_single_atomic_publication(self):
        from src.webui.bridge import PhysicsCaptureV1

        # Given: a real request-70 capture with publication, validation, and write tracing enabled.
        records, events = self._records()
        png = self._png()
        gameplay_bridge = FakeBridge()

        class PhysicsBridge:
            request_count = 0

            def get_physics_capture_v1(self):
                self.request_count += 1
                return PhysicsCaptureV1(
                    png,
                    records[1] if self.request_count == 1 else records[2],
                    () if self.request_count == 1 else tuple(events[:1]),
                )

        class Process:
            pid = 7301

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                pass

        guard = {
            "pre_shot_image": None,
            "pre_shot_sample": None,
            "post_recovery_protocol_state": {},
            "recovery_action": None,
            "pre_shot_guard": {"status": "accepted", "invalid_reason": None},
        }
        action = {
            "coordinate_frame": "absolute",
            "drag_start": [100, 200],
            "drag_release": [130, 150],
            "tapTime": 70,
            "holdTime": 600,
        }

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            temporary_shot = root / "shot_001.tmp"
            accepted_shot = root / "shot_001"
            publication_renames = []
            validation_paths = []
            post_publication_mutations = []
            pre_publication_metadata = None
            published = False
            original_os_replace = os.replace
            original_path_replace = Path.replace
            original_open = Path.open

            def traced_os_replace(source, target, *, src_dir_fd=None, dst_dir_fd=None):
                nonlocal pre_publication_metadata, published
                if (
                    source == "shot_001.tmp"
                    and target == "shot_001"
                    and src_dir_fd is not None
                    and dst_dir_fd is not None
                ):
                    stable_metadata = (
                        Path(f"/proc/self/fd/{src_dir_fd}")
                        / "shot_001.tmp"
                        / "metadata.json"
                    )
                    pre_publication_metadata = stable_metadata.read_bytes()
                    result = original_os_replace(
                        source,
                        target,
                        src_dir_fd=src_dir_fd,
                        dst_dir_fd=dst_dir_fd,
                    )
                    publication_renames.append((source, target, src_dir_fd, dst_dir_fd))
                    published = True
                    return result
                return original_os_replace(
                    source,
                    target,
                    src_dir_fd=src_dir_fd,
                    dst_dir_fd=dst_dir_fd,
                )

            def traced_path_replace(source, target):
                source_path = Path(source)
                target_path = Path(target)
                if published and (
                    source_path == accepted_shot
                    or accepted_shot in source_path.parents
                    or target_path == accepted_shot
                    or accepted_shot in target_path.parents
                ):
                    post_publication_mutations.append(("replace", source_path, target_path))
                return original_path_replace(source_path, target_path)

            def traced_open(path, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
                path = Path(path)
                if (
                    published
                    and any(flag in mode for flag in ("w", "a", "x", "+"))
                    and (path == accepted_shot or accepted_shot in path.parents)
                ):
                    post_publication_mutations.append(("open", path, mode))
                return original_open(path, mode, buffering, encoding, errors, newline)

            def traced_validation(path, *args, **kwargs):
                validation_path = Path(path)
                result = validate_rollout_artifact(validation_path, *args, **kwargs)
                if kwargs.get("capture_contract") == "physics_capture_v1":
                    validation_paths.append((validation_path, published))
                return result

            # When: the fresh-engine collector accepts and publishes one physics shot.
            with (
                patch.object(os, "replace", side_effect=traced_os_replace),
                patch.object(Path, "replace", new=traced_path_replace),
                patch.object(Path, "open", new=traced_open),
                patch("scripts.collect_rollouts.validate_rollout_artifact", side_effect=traced_validation),
                patch("scripts.collect_rollouts.ScienceBirdsBridge", return_value=PhysicsBridge()),
                patch("scripts.collect_rollouts._run_pre_shot_guard", return_value=guard),
            ):
                collect_fresh_engine_rollouts(
                    root,
                    [action],
                    game_dir=Path("game"),
                    host="127.0.0.1",
                    port=2004,
                    agent_id=28888,
                    speed=1,
                    connect_timeout=1,
                    read_timeout=2,
                    prepare_timeout=3,
                    frame_height=480,
                    fast=True,
                    headless=False,
                    target_fps=2,
                    duration_seconds=1,
                    ui_level=None,
                    ui_settle_seconds=0,
                    fresh_engine_attempts=1,
                    start_engine_func=lambda *_args, **_kwargs: Process(),
                    connect_func=lambda *_args, **_kwargs: gameplay_bridge,
                    prepare_func=lambda bridge, **_kwargs: bridge.get_game_state(),
                    video_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0),
                    anchor_actions=False,
                    physics_capture_v1=True,
                    physics_player_sha256="a" * 64,
                    physics_protocol_sha256="b" * 64,
                    physics_archive_sha256="c" * 64,
                )

            # Then: completed metadata and validation precede one rename, after which the shot is immutable.
            self.assertIsNotNone(pre_publication_metadata)
            published_metadata = (accepted_shot / "metadata.json").read_bytes()
            published_document = json.loads(published_metadata)
            with self.subTest("final metadata complete in temporary shot"):
                self.assertTrue(published_document["accepted"])
                self.assertTrue(published_document["artifact_validation"]["accepted"])
                self.assertEqual(published_document.get("fresh_engine_attempt"), 1)
            with self.subTest("exactly one atomic publication"):
                self.assertEqual(len(publication_renames), 1)
                source, target, source_fd, target_fd = publication_renames[0]
                self.assertEqual((source, target), ("shot_001.tmp", "shot_001"))
                self.assertIsNotNone(source_fd)
                self.assertEqual(source_fd, target_fd)
            with self.subTest("validation remains pre-publication"):
                self.assertTrue(validation_paths)
                self.assertTrue(
                    all(not was_published for _path, was_published in validation_paths),
                    validation_paths,
                )
            with self.subTest("published metadata bytes are unchanged"):
                self.assertEqual(published_metadata, pre_publication_metadata)
            with self.subTest("accepted shot is never reopened for mutation"):
                self.assertEqual(post_publication_mutations, [])

    def test_completed_shot_with_stale_provenance_is_quarantined_and_recaptured(self):
        from src.webui.bridge import PhysicsCaptureV1

        records, events = self._records()
        png = self._png()

        class Bridge(FakeBridge):
            request_count = 0

            def get_physics_capture_v1(self):
                self.request_count += 1
                return PhysicsCaptureV1(
                    png,
                    records[1] if self.request_count % 2 == 1 else records[2],
                    () if self.request_count % 2 == 1 else tuple(events[:1]),
                )

        action = {"coordinate_frame": "absolute", "release": [130, 150], "tapTime": 70}
        guard = {"pre_shot_image": None, "pre_shot_sample": None, "post_recovery_protocol_state": {}, "recovery_action": None, "pre_shot_guard": {"status": "accepted", "invalid_reason": None}}
        with TemporaryDirectory() as temporary, patch("scripts.collect_rollouts._run_pre_shot_guard", return_value=guard):
            root = Path(temporary)
            bridge = Bridge()
            common = dict(target_fps=1, duration_seconds=1, max_frames=2, anchor_actions=False, video_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0), physics_capture_v1=True)
            collect_rollouts(bridge, root, [action], physics_player_sha256="a" * 64, physics_protocol_sha256="b" * 64, physics_archive_sha256="c" * 64, **common)

            collect_rollouts(bridge, root, [action], physics_player_sha256="d" * 64, physics_protocol_sha256="e" * 64, physics_archive_sha256="f" * 64, **common)

            current = json.loads((root / "shot_001" / "metadata.json").read_text(encoding="utf-8"))
            stale = json.loads((root / "invalid_attempts" / "shot_001_recovered_01" / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(bridge.request_count, 4)
            self.assertEqual((current["player_sha256"], current["protocol_sha256"], current["archive_sha256"]), ("d" * 64, "e" * 64, "f" * 64))
            self.assertEqual((stale["player_sha256"], stale["protocol_sha256"], stale["archive_sha256"]), ("a" * 64, "b" * 64, "c" * 64))

    def test_generated_fresh_engine_worker_without_required_inputs_fails_closed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = PlannedEpisode(
                split="train",
                entry=LevelEntry("novelty_level_1", "type01001", "level.xml"),
                output_dir=root / "rollouts",
                source="scheduled",
            )
            provenance = PhysicsCaptureProvenance(
                root / "player.tar",
                root / "smoke.json",
                "a" * 64,
                "b" * 64,
                "c" * 64,
            )
            command = " ".join(
                line.strip().removesuffix("\\").strip()
                for line in _collection_command_lines(
                    episode,
                    CollectionOptions(count=1, fps=1, duration=1),
                    WorkerSpec(index=0, display=":149", agent_port=2004, game_port=9001),
                    provenance,
                )[2:]
            )
            stderr = io.StringIO()
            with patch.object(sys, "argv", ["collect_rollouts.py", *shlex.split(command)]), redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--scenario-manifest and --scenario-xml", stderr.getvalue())

    def test_physics_capture_rejects_explicit_and_derived_frame_counts_above_contract(self):
        class Bridge:
            request_count = 0

            def get_physics_capture_v1(self):
                self.request_count += 1
                raise AssertionError("request 70 must not run for an invalid capture limit")

        for name, parameters in {
            "explicit max_frames": {"target_fps": 1, "duration_seconds": 1, "max_frames": 64},
            "fps and duration": {"target_fps": 8, "duration_seconds": 8, "max_frames": None},
        }.items():
            with self.subTest(name=name), TemporaryDirectory() as temporary:
                bridge = Bridge()
                with self.assertRaisesRegex(RolloutCollectionError, "state record limit"):
                    capture_physics_rollout(
                        bridge,
                        Path(temporary) / "shot_000.tmp",
                        **parameters,
                        player_sha256="a" * 64,
                        protocol_sha256="b" * 64,
                        archive_sha256="c" * 64,
                    )
                self.assertEqual(bridge.request_count, 0)

    def test_physics_capture_accepts_63_frame_boundary(self):
        from src.webui.bridge import PhysicsCaptureV1

        records, _ = self._records()
        png = self._png()

        class Bridge:
            request_count = 0

            def get_physics_capture_v1(self):
                state = json.loads(json.dumps(records[1]))
                self.request_count += 1
                state["sequence"] = self.request_count
                state["render_frame"] += self.request_count - 1
                state["fixed_step"] += self.request_count - 1
                state["render_time"] += (self.request_count - 1) / 60
                state["fixed_time"] += (self.request_count - 1) / 50
                state["rgb_frame"]["render_frame"] = state["render_frame"]
                return PhysicsCaptureV1(png, state, ())

        with TemporaryDirectory() as temporary:
            bridge = Bridge()
            shot = Path(temporary) / "shot_000.tmp"
            metadata = capture_physics_rollout(
                bridge,
                shot,
                target_fps=63,
                duration_seconds=1,
                max_frames=63,
                state_header=records[0],
                player_sha256="a" * 64,
                protocol_sha256="b" * 64,
                archive_sha256="c" * 64,
            )

            self.assertEqual(bridge.request_count, 63)
            self.assertEqual(metadata["physics_state_count"], 63)
            self.assertEqual(len((shot / "physics_state.jsonl").read_text(encoding="utf-8").splitlines()), 64)
            self.assertTrue(validate_rollout_artifact(shot, capture_contract="physics_capture_v1")["accepted"])

    def test_physics_capture_enforces_event_and_total_byte_limits_while_streaming(self):
        from src.webui.bridge import PhysicsCaptureV1

        records, events = self._records()
        png = self._png()

        for name, capture_events, message in (
            (
                "event count",
                tuple(
                    {
                        **events[0],
                        "sequence": index,
                        "event_id": f"event:{index:08d}",
                    }
                    for index in range(65)
                ),
                "event record limit",
            ),
            (
                "total bytes",
                (
                    {
                        **events[0],
                        "event_type": "level_failed",
                        "event_id": "event:00000000",
                        "payload": {"reason": "x" * 1_048_576},
                    },
                ),
                "sidecar byte limit",
            ),
        ):
            with self.subTest(name=name), TemporaryDirectory() as temporary:
                class Bridge:
                    def get_physics_capture_v1(self):
                        return PhysicsCaptureV1(png, records[1], capture_events)

                shot = Path(temporary) / "shot_000.tmp"
                with self.assertRaisesRegex(RolloutCollectionError, message):
                    capture_physics_rollout(
                        Bridge(),
                        shot,
                        target_fps=1,
                        duration_seconds=1,
                        max_frames=1,
                        state_header=records[0],
                        player_sha256="a" * 64,
                        protocol_sha256="b" * 64,
                        archive_sha256="c" * 64,
                    )
                self.assertFalse((shot / "metadata.json").exists())

    def test_malformed_request_70_capture_fails_without_success_metadata(self):
        from src.webui.bridge import PhysicsCaptureV1

        class Bridge:
            def get_physics_capture_v1(self):
                return PhysicsCaptureV1(self_png, {"schema_version": "physics_capture_v1"}, ())

        self_png = self._png()
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot_000.tmp"
            with self.assertRaisesRegex(RolloutCollectionError, "malformed_capture"):
                capture_physics_rollout(
                    Bridge(),
                    shot,
                    target_fps=1,
                    duration_seconds=1,
                    max_frames=1,
                    player_sha256="a" * 64,
                    protocol_sha256="b" * 64,
                    archive_sha256="c" * 64,
                )
            self.assertFalse((shot / "metadata.json").exists())

    @staticmethod
    def _pre_shot():
        from PIL import Image

        return Image.new("RGB", (4, 3), (1, 2, 3))

    def test_interrupted_tmp_is_removed_on_repeated_resume(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for _ in range(2):
                interrupted = root / "shot_000.tmp"
                interrupted.mkdir()
                (interrupted / "physics_state.jsonl").write_text('{"truncated":', encoding="utf-8")
                self.assertEqual(cleanup_incomplete_physics_attempts(root), ("shot_000.tmp",))
                self.assertFalse(interrupted.exists())


class FakeBridge:
    def __init__(self):
        self.shots = []
        self.frame_index = 0
        self.symbolic_state = None

    def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
        self.shots.append((x, y, tap_time, fast, release_time))
        return 1

    def shoot_and_record_ground_truth(self, x, y, tap_time=0, release_time=0, frequency=1):
        self.shots.append((x, y, tap_time, False, release_time, frequency))
        return 1

    def configure(self, agent_id, mode):
        self.configured = (agent_id, mode)
        return (0, 0, 1)

    def set_speed(self, speed):
        self.speed = speed
        return 1

    def disconnect(self):
        self.disconnected = True

    def screenshot(self):
        class Screenshot:
            width = 4
            height = 3

            def __init__(self, frame_index):
                value = 50 + frame_index
                self.rgb = bytes(channel for pixel in range(4 * 3) for channel in (value + pixel, 20, 30))

        screenshot = Screenshot(self.frame_index)
        self.frame_index += 1
        return screenshot

    def get_game_state(self):
        from src.webui.bridge import GameState

        return GameState.PLAYING

    def get_current_score(self):
        return 200 + self.frame_index

    def get_current_level(self):
        return 7

    def get_symbolic_state_without_screenshot(self):
        return self.symbolic_state


class CollectRolloutsTest(unittest.TestCase):
    def _assert_fresh_engine_retries_are_rejected(self):
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "frozen collection plan"):
                collect_fresh_engine_rollouts(
                    Path(temporary),
                    [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}],
                    game_dir=Path("game"),
                    host="127.0.0.1",
                    port=2004,
                    agent_id=28888,
                    speed=1,
                    connect_timeout=1,
                    read_timeout=2,
                    prepare_timeout=3,
                    frame_height=480,
                    fast=True,
                    headless=False,
                    target_fps=1,
                    duration_seconds=1,
                    ui_level=None,
                    ui_settle_seconds=0,
                    fresh_engine_attempts=2,
                )

    def test_action_to_shot_matches_webui_game_coordinates_then_flips_once_for_bridge(self):
        action = {
            "coordinate_frame": "slingshot_relative",
            "drag_start": [100, 200],
            "drag_release": [30, 50],
            "tapTime": 70,
            "holdTime": 600,
        }

        shot = action_to_shot(action, frame_height=480)

        self.assertEqual(shot["gameX"], 130)
        self.assertEqual(shot["gameY"], 150)
        self.assertEqual(shot["x"], 130)
        self.assertEqual(shot["y"], 329)
        self.assertEqual(shot["releaseTime"], 600)

    def test_action_guide_endpoint_matches_actual_bridge_shot_pixel(self):
        action = {"coordinate_frame": "slingshot_relative", "drag_start": [100, 200], "drag_release": [30, 50]}
        shot = action_to_shot(action, frame_height=480)

        start, end = _action_guide_points(action, shot, image_height=480)

        self.assertEqual(start, (100, 279))
        self.assertEqual(end, (shot["x"], shot["y"]))

    def test_launch_guide_points_opposite_of_pull_vector_for_rightward_shots(self):
        action = {"coordinate_frame": "slingshot_relative", "drag_start": [300, 220], "drag_release": [-50, 40]}
        shot = action_to_shot(action, frame_height=480)

        start, release_end = _action_guide_points(action, shot, image_height=480)
        launch_start, launch_end = _launch_guide_points(action, image_height=480)

        self.assertEqual(start, (300, 259))
        self.assertEqual(release_end, (shot["x"], shot["y"]))
        self.assertEqual(launch_start, start)
        self.assertGreater(launch_end[0], launch_start[0])
        self.assertLess(launch_end[1], launch_start[1])

    def test_action_to_shot_uses_height_minus_one_y_boundaries(self):
        top = action_to_shot({"coordinate_frame": "absolute", "release": [10, 479]}, frame_height=480)
        bottom = action_to_shot({"coordinate_frame": "absolute", "release": [10, 0]}, frame_height=480)

        self.assertEqual(top["y"], 0)
        self.assertEqual(bottom["y"], 479)

    def test_slingshot_reference_point_from_symbolic_state_uses_request_62_geojson_vertices(self):
        symbolic_state = [
            {
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"label": "Slingshot"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[10, 20], [10, 40], [30, 40], [30, 20]]],
                        },
                    }
                ]
            }
        ]

        reference = slingshot_reference_point_from_symbolic_state(symbolic_state, frame_height=100)

        self.assertEqual(reference, {"gameX": 19, "gameY": 72, "canvasX": 19, "canvasY": 27})

    def test_collect_rollouts_writes_manifest_and_per_shot_metadata(self):
        actions = [
            {
                "action_type": "drag_hold_release",
                "coordinate_frame": "slingshot_relative",
                "drag_start": [300, 220],
                "drag_release": [-50, 40],
                "tapTime": 70,
                "holdTime": 120,
            },
            {
                "action_type": "drag_hold_release",
                "coordinate_frame": "slingshot_relative",
                "drag_start": [300, 220],
                "drag_release": [-80, -20],
                "tapTime": 0,
            },
        ]
        now = [5.0]

        def clock():
            return now[0]

        def sleeper(seconds):
            now[0] += seconds

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                FakeBridge(),
                Path(tmp),
                actions,
                target_fps=2,
                duration_seconds=1.0,
                frame_height=480,
                max_frames=2,
                clock=clock,
                sleeper=sleeper,
            )

            self.assertEqual(manifest["attempt_count"], 2)
            self.assertEqual(manifest["accepted_rollout_count"], 0)
            self.assertEqual(manifest["rollout_count"], 0)
            self.assertEqual(manifest["rollouts"][0]["shoot_response"], 1)
            self.assertEqual(manifest["rollouts"][0]["frame_count"], 2)
            self.assertTrue((Path(tmp) / "shot_001" / "metadata.json").is_file())
            saved_manifest = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_manifest["attempt_count"], 2)
            self.assertEqual(saved_manifest["accepted_rollout_count"], 0)
            self.assertEqual(saved_manifest["rollout_count"], 0)

    def test_collect_rollouts_records_protocol_state_evidence(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]
        events = []

        class TrackingBridge(FakeBridge):
            def __init__(self):
                super().__init__()
                self.phase = "pre-shot"

            def get_game_state(self):
                events.append("state")
                from src.webui.bridge import GameState

                return GameState.PLAYING

            def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
                events.append("shoot")
                self.phase = "after-shoot"
                return super().shoot(x, y, tap_time=tap_time, fast=fast, release_time=release_time)

        bridge = TrackingBridge()

        def pre_shot_grabber():
            events.append("baseline")
            bridge.phase = "after-baseline"
            from PIL import Image

            image = Image.new("RGB", (20, 20), (5, 5, 5))
            image.putpixel((0, 0), (6, 5, 5))
            return image

        def capture_rollout(bridge, output_dir, **kwargs):
            from PIL import Image

            events.append("capture")
            bridge.phase = "after-capture"
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 20), (50, 60, 70)).save(frames_dir / "frame_000000.png", format="PNG")
            Image.new("RGB", (20, 20), (10, 20, 30)).save(frames_dir / "frame_000001.png", format="PNG")
            metadata = {
                "frame_count": 2,
                "frames_dir": str(frames_dir),
            }
            (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            return metadata

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                bridge,
                Path(tmp),
                actions,
                target_fps=1,
                duration_seconds=1,
                pre_shot_grabber=pre_shot_grabber,
                capture_rollout=capture_rollout,
                video_runner=lambda command, check, stdout, stderr: Path(command[-1]).write_bytes(b"mp4"),
            )
            metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["rollouts"][0]["shoot_response"], 1)
        self.assertEqual(events[0], "state")
        self.assertLess(events.index("baseline"), events.index("shoot"))
        self.assertLess(events.index("shoot"), events.index("capture"))
        self.assertEqual(metadata["pre_shot_protocol_state"]["game_state"], "PLAYING")
        self.assertEqual(metadata["pre_shot_protocol_state"]["current_level"], 7)
        self.assertEqual(metadata["post_recovery_protocol_state"]["game_state"], "PLAYING")
        self.assertEqual(metadata["post_shoot_protocol_state"]["game_state"], "PLAYING")
        self.assertEqual(metadata["post_capture_protocol_state"]["game_state"], "PLAYING")
        self.assertEqual(metadata["artifact_validation"]["classification"], "gameplay-valid")
        self.assertTrue(metadata["artifact_validation"]["accepted"])
        self.assertIn("recovery_action", metadata)

    def test_invalid_rollout_records_state_and_reason(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]
        events = []

        class TrackingBridge(FakeBridge):
            def __init__(self):
                super().__init__()
                self.phase = "pre-shot"

            def get_game_state(self):
                events.append("state")
                from src.webui.bridge import GameState

                return GameState.PLAYING

            def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
                events.append("shoot")
                self.phase = "after-shoot"
                return super().shoot(x, y, tap_time=tap_time, fast=fast, release_time=release_time)

        bridge = TrackingBridge()

        def baseline_image():
            from PIL import Image

            image = Image.new("RGB", (20, 20), (50, 60, 70))
            image.putpixel((0, 0), (51, 60, 70))
            return image

        def pre_shot_grabber():
            events.append("baseline")
            bridge.phase = "after-baseline"

            return baseline_image()

        def capture_rollout(bridge, output_dir, **kwargs):
            from PIL import Image

            events.append("capture")
            bridge.phase = "after-capture"
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            baseline_image().save(frames_dir / "frame_000000.png", format="PNG")
            metadata = {
                "frame_count": 1,
                "frames_dir": str(frames_dir),
            }
            (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            return metadata

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                bridge,
                Path(tmp),
                actions,
                target_fps=1,
                duration_seconds=1,
                pre_shot_grabber=pre_shot_grabber,
                capture_rollout=capture_rollout,
                video_runner=lambda command, check, stdout, stderr: Path(command[-1]).write_bytes(b"mp4"),
            )
            metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["attempt_count"], 1)
        self.assertEqual(manifest["accepted_rollout_count"], 0)
        self.assertEqual(manifest["rollout_count"], 0)
        self.assertFalse(manifest["rollouts"][0]["accepted"])
        self.assertEqual(events[0], "state")
        self.assertLess(events.index("baseline"), events.index("shoot"))
        self.assertLess(events.index("shoot"), events.index("capture"))
        self.assertFalse(metadata["artifact_validation"]["accepted"])
        self.assertEqual(metadata["artifact_validation"]["classification"], "no-frame-motion")
        self.assertEqual(metadata["artifact_validation"]["invalid_reason"], "no_frame_motion")
        self.assertFalse(metadata["artifact_validation"]["retryable"])
        self.assertEqual(metadata["artifact_validation"]["retry_decision"], "quarantine")
        self.assertIn("pre_shot_protocol_state", metadata)
        self.assertIn("post_shoot_protocol_state", metadata)
        self.assertIn("post_capture_protocol_state", metadata)
        self.assertIn("post_recovery_protocol_state", metadata)

    def test_post_shot_gate_rejects_menu_capture(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]

        def menu_like_capture(bridge, output_dir, **kwargs):
            from PIL import Image

            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 20), (245, 245, 245)).save(frames_dir / "frame_000000.png", format="PNG")
            metadata = {
                "frame_count": 1,
                "frames_dir": str(frames_dir),
            }
            (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            return metadata

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                FakeBridge(),
                Path(tmp),
                actions,
                target_fps=1,
                duration_seconds=1,
                capture_rollout=menu_like_capture,
                video_runner=lambda command, check, stdout, stderr: Path(command[-1]).write_bytes(b"mp4"),
            )
            metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))
            action_log = json.loads((Path(tmp) / "action_log.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["attempt_count"], 1)
        self.assertEqual(manifest["accepted_rollout_count"], 0)
        self.assertEqual(manifest["rollout_count"], 0)
        self.assertFalse(metadata["artifact_validation"]["accepted"])
        self.assertEqual(metadata["artifact_validation"]["classification"], "menu-detected")
        self.assertEqual(metadata["artifact_validation"]["invalid_reason"], "menu_detected")
        self.assertFalse(metadata["artifact_validation"]["retryable"])
        self.assertEqual(metadata["artifact_validation"]["retry_decision"], "quarantine")
        self.assertEqual(action_log["attempt_count"], 1)
        self.assertEqual(action_log["accepted_trial_count"], 0)
        self.assertEqual(action_log["trial_count"], 1)

    def test_post_shot_gate_flags_low_motion_capture(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]

        def low_motion_capture(bridge, output_dir, **kwargs):
            from PIL import Image

            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            first_frame = Image.new("RGB", (20, 20), (50, 60, 70))
            second_frame = Image.new("RGB", (20, 20), (50, 60, 70))
            for x in range(2):
                for y in range(11):
                    second_frame.putpixel((x, y), (51, 60, 70))
            first_frame.save(frames_dir / "frame_000000.png", format="PNG")
            second_frame.save(frames_dir / "frame_000001.png", format="PNG")
            metadata = {
                "frame_count": 2,
                "frames_dir": str(frames_dir),
            }
            (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            return metadata

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                FakeBridge(),
                Path(tmp),
                actions,
                target_fps=1,
                duration_seconds=1,
                capture_rollout=low_motion_capture,
                video_runner=lambda command, check, stdout, stderr: Path(command[-1]).write_bytes(b"mp4"),
            )
            metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))
            action_log = json.loads((Path(tmp) / "action_log.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["attempt_count"], 1)
        self.assertEqual(manifest["accepted_rollout_count"], 0)
        self.assertEqual(manifest["rollout_count"], 0)
        self.assertFalse(metadata["artifact_validation"]["accepted"])
        self.assertEqual(metadata["artifact_validation"]["classification"], "low-motion-suspicious")
        self.assertEqual(metadata["artifact_validation"]["invalid_reason"], "low_motion_suspicious")
        self.assertTrue(metadata["artifact_validation"]["retryable"])
        self.assertEqual(metadata["artifact_validation"]["retry_decision"], "retry")
        self.assertEqual(action_log["attempt_count"], 1)
        self.assertEqual(action_log["accepted_trial_count"], 0)
        self.assertEqual(action_log["trial_count"], 1)

    def test_pre_shot_guard_rejects_menu_surface_even_when_protocol_playing(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]
        events = []

        class MenuSurfaceBridge(FakeBridge):
            def get_game_state(self):
                from src.webui.bridge import GameState

                return GameState.PLAYING

            def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
                events.append("shoot")
                return super().shoot(x, y, tap_time=tap_time, fast=fast, release_time=release_time)

        def menu_pre_shot_grabber():
            events.append("baseline")
            from PIL import Image

            image = Image.new("RGB", (40, 30), (245, 245, 245))
            image.putpixel((3, 3), (230, 40, 40))
            image.putpixel((4, 3), (40, 130, 230))
            return image

        def capture_rollout(bridge, output_dir, **kwargs):
            events.append("capture")
            raise AssertionError("collector must not capture after a menu-like pre-shot surface")

        bridge = MenuSurfaceBridge()
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "recovery_failed"):
                collect_rollouts(
                    bridge,
                    Path(tmp),
                    actions,
                    target_fps=1,
                    duration_seconds=1,
                    pre_shot_grabber=menu_pre_shot_grabber,
                    capture_rollout=capture_rollout,
                )
            metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(events, ["baseline"])
        self.assertEqual(bridge.shots, [])
        self.assertEqual(metadata["pre_shot_guard"]["status"], "recovery_failed")
        self.assertEqual(metadata["pre_shot_guard"]["invalid_reason"], "menu_like_pre_shot")
        self.assertEqual(metadata["pre_shot_guard"]["protocol_state"]["game_state"], "PLAYING")
        self.assertTrue(metadata["pre_shot_guard"]["visual_evidence"]["menu_like"])
        self.assertIn("post_recovery_protocol_state", metadata)
        self.assertEqual(metadata["recovery_action"], None)

    def test_pre_shot_guard_rejects_menu_inside_default_desktop_crop(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]
        events = []

        class DesktopMenuBridge(FakeBridge):
            def get_game_state(self):
                from src.webui.bridge import GameState

                return GameState.PLAYING

            def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
                events.append("shoot")
                return super().shoot(x, y, tap_time=tap_time, fast=fast, release_time=release_time)

        def desktop_pre_shot_grabber():
            events.append("baseline")
            from PIL import Image

            image = Image.new("RGB", (1024, 768), (0, 0, 0))
            for x in range(32, 672):
                for y in range(64, 544):
                    image.putpixel((x, y), (245, 245, 245))
            image.putpixel((35, 67), (230, 40, 40))
            image.putpixel((36, 67), (40, 130, 230))
            return image

        def capture_rollout(bridge, output_dir, **kwargs):
            events.append("capture")
            raise AssertionError("collector must not capture when the cropped game viewport is menu-like")

        bridge = DesktopMenuBridge()
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "recovery_failed"):
                collect_rollouts(
                    bridge,
                    Path(tmp),
                    actions,
                    target_fps=1,
                    duration_seconds=1,
                    pre_shot_grabber=desktop_pre_shot_grabber,
                    capture_rollout=capture_rollout,
                )
            metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(events, ["baseline"])
        self.assertEqual(bridge.shots, [])
        visual_evidence = metadata["pre_shot_guard"]["visual_evidence"]
        self.assertEqual(metadata["pre_shot_guard"]["invalid_reason"], "menu_like_pre_shot")
        self.assertTrue(visual_evidence["menu_like"])
        self.assertEqual(visual_evidence["width"], 640)
        self.assertEqual(visual_evidence["height"], 480)

    def test_pre_shot_guard_recovers_new_trial_before_shooting(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]
        events = []

        class NewTrialBridge(FakeBridge):
            def __init__(self):
                super().__init__()
                self.state_names = ["NEWTRIAL", "PLAYING", "PLAYING", "PLAYING", "PLAYING"]
                self.ready_calls = 0

            def get_game_state(self):
                events.append("state")
                from src.webui.bridge import GameState

                state_name = self.state_names.pop(0) if self.state_names else "PLAYING"
                return getattr(GameState, state_name)

            def ready_for_new_set(self):
                events.append("ready_for_new_set")
                self.ready_calls += 1
                return (1, 0, 0, 0, 0, 0, 0)

            def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
                events.append("shoot")
                return super().shoot(x, y, tap_time=tap_time, fast=fast, release_time=release_time)

        def gameplay_pre_shot_grabber():
            events.append("baseline")
            from PIL import Image

            image = Image.new("RGB", (40, 30), (20, 30, 40))
            image.putpixel((3, 3), (80, 90, 100))
            image.putpixel((12, 8), (150, 90, 40))
            return image

        def capture_rollout(bridge, output_dir, **kwargs):
            from PIL import Image

            events.append("capture")
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (40, 30), (50, 60, 70)).save(frames_dir / "frame_000000.png", format="PNG")
            Image.new("RGB", (40, 30), (90, 100, 110)).save(frames_dir / "frame_000001.png", format="PNG")
            return {"frame_count": 2, "frames_dir": str(frames_dir)}

        bridge = NewTrialBridge()
        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                bridge,
                Path(tmp),
                actions,
                target_fps=1,
                duration_seconds=1,
                pre_shot_grabber=gameplay_pre_shot_grabber,
                capture_rollout=capture_rollout,
                video_runner=lambda command, check, stdout, stderr: Path(command[-1]).write_bytes(b"mp4"),
            )
            metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))

        self.assertLess(events.index("ready_for_new_set"), events.index("shoot"))
        self.assertLess(events.index("shoot"), events.index("capture"))
        self.assertEqual(bridge.ready_calls, 1)
        self.assertEqual(len(bridge.shots), 1)
        self.assertEqual(manifest["rollouts"][0]["shoot_response"], 1)
        self.assertEqual(metadata["pre_shot_protocol_state"]["game_state"], "NEWTRIAL")
        self.assertEqual(metadata["post_recovery_protocol_state"]["game_state"], "PLAYING")
        self.assertEqual(metadata["recovery_action"], "ready_for_new_set")
        self.assertEqual(metadata["pre_shot_guard"]["status"], "accepted_after_recovery")
        self.assertEqual(metadata["pre_shot_guard"]["recovery_attempts"], 1)
        self.assertEqual(metadata["pre_shot_guard"]["invalid_reason"], None)
        self.assertTrue(metadata["artifact_validation"]["accepted"])

    def test_collect_rollouts_can_reset_before_each_action_and_use_custom_capture(self):
        actions = [
            {"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0},
            {"coordinate_frame": "absolute", "release": [240, 250], "tapTime": 45},
        ]
        reset_calls = []
        capture_calls = []

        def reset_rollout(index, action):
            reset_calls.append((index, action["tapTime"]))

        def capture_rollout(bridge, output_dir, **kwargs):
            capture_calls.append((output_dir.name, kwargs["action"]["tapTime"]))
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "metadata.json").write_text(json.dumps({"frame_count": 1}), encoding="utf-8")
            return {"frame_count": 1}

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                FakeBridge(),
                Path(tmp),
                actions,
                target_fps=1,
                duration_seconds=1,
                reset_rollout=reset_rollout,
                capture_rollout=capture_rollout,
            )

        self.assertEqual(reset_calls, [(1, 0), (2, 45)])
        self.assertEqual(capture_calls, [("shot_001", 0), ("shot_002", 45)])
        self.assertEqual(manifest["attempt_count"], 2)
        self.assertEqual(manifest["accepted_rollout_count"], 0)
        self.assertEqual(manifest["rollout_count"], 0)

    def test_collect_rollouts_anchors_same_episode_actions_to_symbolic_slingshot(self):
        bridge = FakeBridge()
        bridge.symbolic_state = [
            {
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"label": "Slingshot"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[80, 230], [80, 270], [120, 270], [120, 230]]],
                        },
                    }
                ]
            }
        ]
        actions = [
            {
                "action_type": "drag_hold_release",
                "coordinate_frame": "slingshot_relative",
                "drag_start": [300, 220],
                "drag_release": [-45, 4],
                "tapTime": 0,
                "holdTime": 600,
            }
        ]

        def capture_rollout(bridge, output_dir, **kwargs):
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True)
            from PIL import Image

            Image.new("RGB", (160, 90), (50, 60, 70)).save(frames_dir / "frame_000000.png", format="PNG")
            return {"frame_count": 1, "frames_dir": str(frames_dir)}

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                bridge,
                Path(tmp),
                actions,
                target_fps=1,
                duration_seconds=1,
                frame_height=480,
                capture_rollout=capture_rollout,
                video_runner=lambda command, check, stdout, stderr: Path(command[-1]).write_bytes(b"mp4"),
            )

        anchored_action = manifest["rollouts"][0]["action"]
        self.assertEqual(anchored_action["drag_start"], [98, 235])
        self.assertEqual(manifest["rollouts"][0]["slingshot_reference"], {"gameX": 98, "gameY": 235, "canvasX": 98, "canvasY": 244})
        self.assertEqual(bridge.shots[0], (53, 248, 0, True, 600))

    def test_capture_desktop_rollout_records_imagegrab_frames(self):
        class Grabber:
            def __init__(self):
                self.calls = 0

            def grab(self):
                from PIL import Image

                self.calls += 1
                image = Image.new("RGB", (4, 3), (10, 20, 30))
                image.putpixel((0, 0), (70, 80, 90))
                if self.calls == 1:
                    image.putpixel((1, 0), (100, 110, 120))
                else:
                    image.putpixel((1, 0), (100, 110, 120))
                    image.putpixel((2, 0), (130, 140, 150))
                return image

        from PIL import Image

        baseline = Image.new("RGB", (4, 3), (10, 20, 30))
        baseline.putpixel((0, 0), (70, 80, 90))
        now = [2.0]

        def clock():
            return now[0]

        def sleeper(seconds):
            now[0] += seconds

        with TemporaryDirectory() as tmp:
            metadata = capture_desktop_rollout(
                FakeBridge(),
                Path(tmp),
                target_fps=2,
                duration_seconds=1,
                max_frames=2,
                action={"tapTime": 0},
                pre_shot_image=baseline,
                pre_shot_sample={"state": "PLAYING", "score": 222},
                grabber=Grabber(),
                clock=clock,
                sleeper=sleeper,
            )

            self.assertEqual(metadata["capture_source"], "desktop-imagegrab")
            self.assertEqual(metadata["frame_count"], 2)
            self.assertEqual(metadata["pre_shot_path"], str(Path(tmp) / "pre_shot.png"))
            self.assertFalse(metadata["frames"][0]["uniform"])
            self.assertIsNone(metadata["frames"][0]["frame_delta"])
            self.assertEqual(metadata["frames"][0]["pre_shot_delta"]["changed_pixel_count"], 1)
            self.assertEqual(metadata["frames"][1]["frame_delta"]["changed_pixel_count"], 1)
            self.assertEqual(metadata["frames"][1]["pre_shot_delta"]["changed_pixel_count"], 2)
            self.assertEqual(metadata["max_pre_shot_delta"], 2)
            self.assertEqual(metadata["max_pre_shot_delta_bbox"], [1, 0, 3, 1])
            self.assertEqual(metadata["pre_shot_sample"], {"state": "PLAYING", "score": 222})
            self.assertTrue((Path(tmp) / "pre_shot.png").is_file())
            self.assertTrue((Path(tmp) / "frames" / "frame_000000.png").is_file())

    def test_known_dataset_artifacts_classify_gameplay_and_reported_menu_shots(self):
        valid_artifact_root = Path(
            "data/novphy_rollouts_dataset/train/novelty_level_0_type010101_00001_0_1_010101_0_1"
        )
        reported_menu_artifact_root = Path(
            "data/novphy_rollouts_dataset/train/novelty_level_0_type010101_00002_0_1_010101_0_1"
        )

        valid = validate_rollout_artifact(valid_artifact_root / "shot_001")
        menu_static = validate_rollout_artifact(reported_menu_artifact_root / "shot_001")

        self.assertTrue(valid["accepted"])
        self.assertEqual(valid["classification"], "gameplay-valid")
        self.assertEqual(valid["max_frame_delta"], 965)
        self.assertEqual(valid["score"], 1770)
        self.assertIn("gameplay-valid", valid["signals"])

        self.assertFalse(menu_static["accepted"])
        self.assertEqual(menu_static["classification"], "menu-detected")
        self.assertEqual(menu_static["invalid_reason"], "menu_detected")
        self.assertEqual(menu_static["max_frame_delta"], 0)
        self.assertEqual(menu_static["max_pre_shot_delta"], 0)
        self.assertIn("menu-detected", menu_static["signals"])
        self.assertIn("no-frame-motion", menu_static["signals"])

    def test_rollout_artifact_validator_rejects_missing_frames(self):
        with TemporaryDirectory() as tmp:
            shot_dir = Path(tmp) / "shot_001"
            shot_dir.mkdir()
            metadata = {
                "frame_count": 2,
                "frames_dir": str(shot_dir / "frames"),
                "frames": [
                    {"path": str(shot_dir / "frames" / "frame_000000.png")},
                    {"path": str(shot_dir / "frames" / "frame_000001.png")},
                ],
                "max_frame_delta": 0,
            }
            (shot_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            result = validate_rollout_artifact(shot_dir)

        self.assertFalse(result["accepted"])
        self.assertEqual(result["classification"], "missing-artifact")
        self.assertEqual(result["invalid_reason"], "missing_artifact")
        self.assertIn("missing frames", result["message"])

    def test_rollout_artifact_validator_rejects_static_menu_with_inflated_metadata_deltas(self):
        from PIL import Image

        with TemporaryDirectory() as tmp:
            shot_dir = Path(tmp) / "shot_001"
            frames_dir = shot_dir / "frames"
            frames_dir.mkdir(parents=True)
            frame_path = frames_dir / "frame_000000.png"
            Image.new("RGB", (40, 30), (245, 245, 245)).save(frame_path, format="PNG")
            metadata = {
                "frame_count": 1,
                "frames_dir": str(frames_dir),
                "frames": [{"path": str(frame_path)}],
                "max_frame_delta": 999,
                "max_pre_shot_delta": 999,
            }
            (shot_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            result = validate_rollout_artifact(shot_dir)

        self.assertFalse(result["accepted"])
        self.assertIn(result["invalid_reason"], {"menu_detected", "no_frame_motion", "low_motion_suspicious"})
        self.assertIn("no-frame-motion", result["signals"])
        self.assertEqual(result["observed_max_frame_delta"], 0)

    def test_capture_desktop_rollout_crops_xvnc_desktop_to_game_viewport(self):
        class Grabber:
            def grab(self):
                from PIL import Image

                image = Image.new("RGB", (1024, 768), (0, 0, 0))
                for x in range(32, 672):
                    for y in range(64, 544):
                        image.putpixel((x, y), (10, 20, 30))
                image.putpixel((32, 64), (70, 80, 90))
                return image

        from PIL import Image

        baseline = Image.new("RGB", (1024, 768), (0, 0, 0))
        for x in range(32, 672):
            for y in range(64, 544):
                baseline.putpixel((x, y), (10, 20, 30))
        baseline.putpixel((32, 64), (70, 80, 90))

        with TemporaryDirectory() as tmp:
            metadata = capture_desktop_rollout(
                FakeBridge(),
                Path(tmp),
                target_fps=1,
                duration_seconds=1,
                max_frames=1,
                pre_shot_image=baseline,
                grabber=Grabber(),
            )

        self.assertEqual(metadata["desktop_crop"], [32, 64, 672, 544])
        self.assertEqual(metadata["frames"][0]["width"], 640)
        self.assertEqual(metadata["frames"][0]["height"], 480)

    def test_capture_desktop_rollout_detects_shifted_xvnc_game_viewport(self):
        def desktop_image():
            from PIL import Image

            image = Image.new("RGB", (1024, 768), (0, 0, 0))
            for x in range(192, 672):
                for y in range(144, 544):
                    image.putpixel((x, y), (10, 20, 30))
            image.putpixel((192, 144), (70, 80, 90))
            return image

        class Grabber:
            def grab(self):
                return desktop_image()

        with TemporaryDirectory() as tmp:
            metadata = capture_desktop_rollout(
                FakeBridge(),
                Path(tmp),
                target_fps=1,
                duration_seconds=1,
                max_frames=1,
                pre_shot_image=desktop_image(),
                grabber=Grabber(),
            )

            from PIL import Image

            with Image.open(Path(tmp) / "frames" / "frame_000000.png") as image:
                saved_top_left = image.getpixel((0, 0))

        self.assertEqual(metadata["desktop_crop"], [192, 144, 832, 624])
        self.assertEqual(metadata["frames"][0]["width"], 640)
        self.assertEqual(metadata["frames"][0]["height"], 480)
        self.assertEqual(saved_top_left, (70, 80, 90))

    def test_capture_desktop_rollout_keeps_frame_crop_when_pre_shot_is_already_cropped(self):
        class Grabber:
            def grab(self):
                from PIL import Image

                image = Image.new("RGB", (1024, 768), (0, 0, 0))
                for x in range(32, 672):
                    for y in range(64, 544):
                        image.putpixel((x, y), (10, 20, 30))
                image.putpixel((32, 64), (70, 80, 90))
                return image

        from PIL import Image

        cropped_baseline = Image.new("RGB", (640, 480), (10, 20, 30))
        cropped_baseline.putpixel((0, 0), (70, 80, 90))

        with TemporaryDirectory() as tmp:
            metadata = capture_desktop_rollout(
                FakeBridge(),
                Path(tmp),
                target_fps=1,
                duration_seconds=1,
                max_frames=1,
                pre_shot_image=cropped_baseline,
                grabber=Grabber(),
            )

        self.assertEqual(metadata["desktop_crop"], [32, 64, 672, 544])
        self.assertEqual(metadata["frames"][0]["width"], 640)
        self.assertEqual(metadata["frames"][0]["height"], 480)

    def test_capture_desktop_rollout_does_not_let_state_polling_throttle_frames(self):
        class Grabber:
            def grab(self):
                from PIL import Image

                image = Image.new("RGB", (4, 3), (10, 20, 30))
                image.putpixel((0, 0), (70, 80, 90))
                return image

        now = [0.0]

        def clock():
            return now[0]

        def sleeper(seconds):
            now[0] += seconds

        class SlowStateBridge(FakeBridge):
            def get_game_state(self):
                now[0] += 0.5
                return super().get_game_state()

            def get_current_score(self):
                now[0] += 0.5
                return super().get_current_score()

        with TemporaryDirectory() as tmp:
            metadata = capture_desktop_rollout(
                SlowStateBridge(),
                Path(tmp),
                target_fps=30,
                duration_seconds=1,
                grabber=Grabber(),
                clock=clock,
                sleeper=sleeper,
            )

        self.assertEqual(metadata["frame_count"], 30)
        self.assertEqual(len(metadata["state_samples"]), 1)

    def test_capture_desktop_rollout_starts_capture_before_shoot_callback(self):
        from PIL import Image

        events = []

        class Grabber:
            def __init__(self):
                self.calls = 0

            def grab(self):
                events.append(f"grab-{self.calls}")
                image = Image.new("RGB", (4, 3), (10 + self.calls, 20, 30))
                image.putpixel((0, 0), (70, 80, 90))
                self.calls += 1
                return image

        now = [1.0]

        def clock():
            return now[0]

        def sleeper(seconds):
            now[0] += seconds

        def shoot():
            events.append("shoot")
            return 1

        with TemporaryDirectory() as tmp:
            metadata = capture_desktop_rollout(
                FakeBridge(),
                Path(tmp),
                target_fps=2,
                duration_seconds=1.0,
                max_frames=2,
                grabber=Grabber(),
                shoot=shoot,
                clock=clock,
                sleeper=sleeper,
            )

        self.assertEqual(events, ["grab-0", "shoot", "grab-1"])
        self.assertEqual(metadata["shoot_response"], 1)
        self.assertEqual(metadata["shoot_frame_index"], 0)

    def test_capture_desktop_rollout_counts_duration_after_blocking_shoot_callback(self):
        from PIL import Image

        class Grabber:
            def __init__(self):
                self.calls = 0

            def grab(self):
                image = Image.new("RGB", (4, 3), (10 + self.calls, 20, 30))
                image.putpixel((0, 0), (70, 80, 90))
                self.calls += 1
                return image

        now = [0.0]

        def clock():
            return now[0]

        def sleeper(seconds):
            now[0] += seconds

        def shoot():
            now[0] += 1.0
            return 1

        with TemporaryDirectory() as tmp:
            metadata = capture_desktop_rollout(
                FakeBridge(),
                Path(tmp),
                target_fps=2,
                duration_seconds=1.0,
                max_duration_seconds=3.0,
                settle_seconds=0.0,
                settle_pixel_threshold=999,
                grabber=Grabber(),
                shoot=shoot,
                clock=clock,
                sleeper=sleeper,
            )

        self.assertGreaterEqual(metadata["frames"][-1]["t"], 2.0)
        self.assertEqual(metadata["shoot_frame_index"], 0)
        self.assertEqual(metadata["capture_stop_reason"], "settled")

    def test_capture_desktop_rollout_waits_for_visual_settle_after_minimum_duration(self):
        from PIL import Image

        class Grabber:
            def __init__(self):
                self.calls = 0

            def grab(self):
                value = 10 + min(self.calls, 3)
                image = Image.new("RGB", (4, 3), (value, 20, 30))
                image.putpixel((0, 0), (70, 80, 90))
                self.calls += 1
                return image

        now = [0.0]

        def clock():
            return now[0]

        def sleeper(seconds):
            now[0] += seconds

        with TemporaryDirectory() as tmp:
            metadata = capture_desktop_rollout(
                FakeBridge(),
                Path(tmp),
                target_fps=4,
                duration_seconds=0.25,
                max_duration_seconds=3.0,
                settle_seconds=0.5,
                settle_pixel_threshold=0,
                grabber=Grabber(),
                shoot=lambda: 1,
                clock=clock,
                sleeper=sleeper,
            )

        self.assertEqual(metadata["capture_stop_reason"], "settled")
        self.assertGreaterEqual(metadata["frame_count"], 6)
        self.assertEqual(metadata["frames"][-1]["frame_delta"]["changed_pixel_count"], 0)

    def test_collect_rollouts_saves_desktop_pre_shot_baseline_before_shoot_and_records_delta(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]
        events = []

        class LoggingBridge(FakeBridge):
            def get_game_state(self):
                events.append("state")
                return super().get_game_state()

            def get_current_score(self):
                events.append("score")
                return super().get_current_score()

            def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
                events.append("shoot")
                return super().shoot(x, y, tap_time=tap_time, fast=fast, release_time=release_time)

        class Grabber:
            def __init__(self):
                self.calls = 0

            def grab(self):
                from PIL import Image

                self.calls += 1
                image = Image.new("RGB", (4, 3), (10, 20, 30))
                image.putpixel((0, 0), (70, 80, 90))
                image.putpixel((1, 0), (100, 110, 120))
                if self.calls > 1:
                    image.putpixel((2, 0), (130, 140, 150))
                return image

        from PIL import Image

        def pre_shot_grabber():
            events.append("baseline")
            baseline = Image.new("RGB", (4, 3), (10, 20, 30))
            baseline.putpixel((0, 0), (70, 80, 90))
            return baseline

        now = [1.0]

        def clock():
            return now[0]

        def sleeper(seconds):
            now[0] += seconds

        def capture_rollout(bridge, output_dir, **kwargs):
            return capture_desktop_rollout(bridge, output_dir, grabber=Grabber(), **kwargs)

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                LoggingBridge(),
                Path(tmp),
                actions,
                target_fps=2,
                duration_seconds=1.0,
                frame_height=480,
                max_frames=2,
                pre_shot_grabber=pre_shot_grabber,
                capture_rollout=capture_rollout,
                clock=clock,
                sleeper=sleeper,
            )
            metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))

            self.assertTrue((Path(tmp) / "shot_001" / "pre_shot.png").is_file())

        self.assertEqual(events[:3], ["state", "score", "baseline"])
        self.assertLess(events.index("baseline"), events.index("shoot"))
        self.assertEqual(manifest["attempt_count"], 1)
        self.assertEqual(manifest["accepted_rollout_count"], 0)
        self.assertEqual(manifest["rollout_count"], 0)
        self.assertEqual(metadata["frames"][0]["pre_shot_delta"]["changed_pixel_count"], 1)
        self.assertEqual(metadata["frames"][1]["pre_shot_delta"]["changed_pixel_count"], 2)
        self.assertEqual(metadata["max_pre_shot_delta"], 2)
        self.assertEqual(metadata["max_pre_shot_delta_bbox"], [1, 0, 3, 1])
        self.assertEqual(metadata["pre_shot_path"], str(Path(tmp) / "shot_001" / "pre_shot.png"))
        self.assertEqual(metadata["pre_shot_sample"], {"state": "PLAYING", "score": 200})
        self.assertEqual(metadata["frames"][1]["frame_delta"]["changed_pixel_count"], 1)

    def test_collect_rollouts_can_defer_shoot_until_desktop_capture_starts(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]
        events = []

        class LoggingBridge(FakeBridge):
            def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
                events.append("shoot")
                return super().shoot(x, y, tap_time=tap_time, fast=fast, release_time=release_time)

        def capture_rollout(bridge, output_dir, **kwargs):
            events.append("capture-start")
            response = kwargs["shoot"]()
            events.append("capture-after-shoot")
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True)
            from PIL import Image

            Image.new("RGB", (160, 90), (50, 60, 70)).save(frames_dir / "frame_000000.png", format="PNG")
            return {"frame_count": 1, "frames_dir": str(frames_dir), "shoot_response": response, "shoot_frame_index": 0}

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                LoggingBridge(),
                Path(tmp),
                actions,
                target_fps=1,
                duration_seconds=1,
                capture_rollout=capture_rollout,
                shoot_before_capture=False,
            )

        self.assertEqual(events, ["capture-start", "shoot", "capture-after-shoot"])
        self.assertEqual(manifest["rollouts"][0]["shoot_response"], 1)

    def test_collect_rollouts_writes_review_mp4_for_each_shot(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]
        runner_calls = []

        def capture_rollout(bridge, output_dir, **kwargs):
            from PIL import Image

            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True)
            for index in range(2):
                image = Image.new("RGB", (20, 20), (10 + index * 80, 20, 30))
                image.putpixel((0, 0), (70, 80, 90))
                image.save(frames_dir / f"frame_{index:06d}.png", format="PNG")
            metadata = {"frame_count": 2, "frames_dir": str(frames_dir)}
            (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            return metadata

        def video_runner(command, check, stdout, stderr):
            runner_calls.append(command)
            self.assertEqual(Path(command[-1]).parent.name, "shot_001")
            self.assertFalse((Path(command[-1]).parent / "rollout.mp4").exists())
            Path(command[-1]).write_bytes(b"mp4")

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                FakeBridge(),
                Path(tmp),
                actions,
                target_fps=30,
                duration_seconds=1,
                capture_rollout=capture_rollout,
                video_runner=video_runner,
            )
            shot_dir = Path(tmp) / "shot_001"
            metadata = json.loads((shot_dir / "metadata.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["rollouts"][0]["video_path"], str(Path(tmp) / "shot_001" / "rollout.mp4"))
            self.assertEqual(metadata["video_path"], str(Path(tmp) / "shot_001" / "rollout.mp4"))
            self.assertEqual((shot_dir / "rollout.mp4").read_bytes(), b"mp4")
            self.assertEqual(sorted(path.name for path in (shot_dir / "frames").glob("frame_*.png")), ["frame_000000.png", "frame_000001.png"])
            self.assertFalse((shot_dir / "video_frames").exists())
            self.assertNotIn("video_frames_dir", metadata)
            self.assertNotIn("video_input_pattern", metadata)
            self.assertNotIn(runner_calls[0][runner_calls[0].index("-i") + 1], json.dumps(metadata))
            self.assertEqual(Path(runner_calls[0][-1]).name, ".rollout.tmp.mp4")
            self.assertIn("-framerate", runner_calls[0])
            self.assertIn("30", runner_calls[0])

    def test_prepare_rollout_video_frames_prefixes_pre_shot_and_overlays_action_text(self):
        from PIL import Image, ImageChops

        action = {
            "action_type": "drag_hold_release",
            "coordinate_frame": "slingshot_relative",
            "drag_start": [300, 220],
            "drag_release": [-50, 40],
            "tapTime": 70,
            "holdTime": 120,
        }

        with TemporaryDirectory() as tmp:
            shot_dir = Path(tmp)
            frames_dir = shot_dir / "frames"
            frames_dir.mkdir()
            pre_shot = Image.new("RGB", (160, 90), (20, 30, 40))
            pre_shot.putpixel((10, 10), (90, 100, 110))
            pre_shot_path = shot_dir / "pre_shot.png"
            pre_shot.save(pre_shot_path, format="PNG")
            for index, color in enumerate(((50, 60, 70), (80, 90, 100))):
                Image.new("RGB", (160, 90), color).save(frames_dir / f"frame_{index:06d}.png", format="PNG")

            overlay_frames_dir = shot_dir / "temporary_video_frames"
            video = prepare_rollout_video_frames(
                overlay_frames_dir,
                frames_dir,
                action=action,
                shot={"x": 250, "y": 299, "tapTime": 70, "releaseTime": 120},
                fps=4,
                pre_shot_path=pre_shot_path,
                lead_in_seconds=0.5,
            )

            first_frame = Image.open(Path(video["video_frames_dir"]) / "frame_000000.png")
            raw_first_frame = Image.open(frames_dir / "frame_000000.png")
            raw_pre_shot = Image.open(pre_shot_path)

        self.assertEqual(video["pre_action_frame_count"], 2)
        self.assertEqual(video["video_frame_count"], 4)
        self.assertEqual(video["video_input_pattern"], str(Path(tmp) / "temporary_video_frames" / "frame_%06d.png"))
        self.assertIn("drag_mode=slingshot_relative", video["video_overlay"]["text"])
        self.assertIn("drag_xy=(300,220)", video["video_overlay"]["text"])
        self.assertIn("pull_release_xy=(-50,40)", video["video_overlay"]["text"])
        self.assertIn("launch_xy=(350,260)", video["video_overlay"]["text"])
        self.assertIn("tapTime=70", video["video_overlay"]["text"])
        self.assertIn("releaseTime=120", video["video_overlay"]["text"])
        self.assertIn("green=launch", video["video_overlay"]["action_guide"])
        self.assertIsNotNone(ImageChops.difference(first_frame.crop((0, 0, 160, 24)), raw_pre_shot.crop((0, 0, 160, 24))).getbbox())
        self.assertIsNone(ImageChops.difference(raw_first_frame, Image.new("RGB", (160, 90), (50, 60, 70))).getbbox())

    def test_prepare_rollout_video_frames_starts_with_neutral_pre_drag_frame(self):
        from PIL import Image, ImageChops

        action = {
            "action_type": "drag_hold_release",
            "coordinate_frame": "slingshot_relative",
            "drag_start": [80, 40],
            "drag_release": [-30, 10],
            "tapTime": 70,
            "holdTime": 120,
        }

        with TemporaryDirectory() as tmp:
            shot_dir = Path(tmp)
            frames_dir = shot_dir / "frames"
            frames_dir.mkdir()
            pre_shot = Image.new("RGB", (160, 90), (20, 30, 40))
            pre_shot_path = shot_dir / "pre_shot.png"
            pre_shot.save(pre_shot_path, format="PNG")
            Image.new("RGB", (160, 90), (50, 60, 70)).save(frames_dir / "frame_000000.png", format="PNG")

            overlay_frames_dir = shot_dir / "temporary_video_frames"
            video = prepare_rollout_video_frames(
                overlay_frames_dir,
                frames_dir,
                action=action,
                shot={"x": 50, "y": 59, "tapTime": 70, "releaseTime": 120},
                fps=4,
                pre_shot_path=pre_shot_path,
                lead_in_seconds=1.0,
            )

            first_frame = Image.open(Path(video["video_frames_dir"]) / "frame_000000.png")
            aim_frame = Image.open(Path(video["video_frames_dir"]) / "frame_000002.png")
            raw_pre_shot = Image.open(pre_shot_path)

        action_area = (0, 24, 160, 90)
        self.assertEqual(video["pre_drag_frame_count"], 2)
        self.assertEqual(video["aim_hold_frame_count"], 2)
        self.assertEqual(video["video_phase_counts"], {"pre_drag": 2, "aim_hold": 2, "rollout": 1})
        self.assertEqual(PRE_DRAG_OVERLAY_TEXT, "phase=pre_drag pre_shot_baseline")
        self.assertIsNone(ImageChops.difference(first_frame.crop(action_area), raw_pre_shot.crop(action_area)).getbbox())
        self.assertIsNotNone(ImageChops.difference(aim_frame.crop(action_area), raw_pre_shot.crop(action_area)).getbbox())

    def test_format_action_overlay_text_describes_absolute_and_slingshot_actions(self):
        slingshot_text = format_action_overlay_text(
            {
                "action_type": "drag_hold_release",
                "coordinate_frame": "slingshot_relative",
                "drag_start": [118, 315],
                "drag_release": [-42, -15],
                "tapTime": 45,
                "holdTime": 120,
            },
            {"x": 76, "y": 149, "tapTime": 45, "releaseTime": 120},
        )
        absolute_text = format_action_overlay_text(
            {"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0},
            {"x": 250, "y": 219, "tapTime": 0, "releaseTime": 0},
        )

        self.assertIn("drag_mode=slingshot_relative", slingshot_text)
        self.assertIn("release_mode=slingshot_relative", slingshot_text)
        self.assertIn("drag_xy=(118,315)", slingshot_text)
        self.assertIn("pull_release_xy=(-42,-15)", slingshot_text)
        self.assertIn("socket_xy=(76,149)", slingshot_text)
        self.assertIn("drag_mode=absolute", absolute_text)
        self.assertIn("pull_release_xy=(250,260)", absolute_text)

    def test_collect_rollouts_writes_action_logs_and_uses_temporary_overlays_for_mp4(self):
        actions = [
            {
                "action_type": "drag_hold_release",
                "coordinate_frame": "slingshot_relative",
                "drag_start": [300, 220],
                "drag_release": [-50, 40],
                "tapTime": 70,
                "holdTime": 120,
            }
        ]
        runner_calls = []

        def capture_rollout(bridge, output_dir, **kwargs):
            from PIL import Image

            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True)
            for index in range(2):
                Image.new("RGB", (160, 90), (50 + index, 60, 70)).save(frames_dir / f"frame_{index:06d}.png", format="PNG")
            return {"frame_count": 2, "frames_dir": str(frames_dir), "pre_shot_path": str(output_dir / "pre_shot.png")}

        def pre_shot_grabber():
            from PIL import Image

            image = Image.new("RGB", (160, 90), (20, 30, 40))
            image.putpixel((1, 1), (80, 90, 100))
            return image

        def video_runner(command, check, stdout, stderr):
            runner_calls.append(command)
            Path(command[-1]).write_bytes(b"mp4")

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                FakeBridge(),
                Path(tmp),
                actions,
                target_fps=4,
                duration_seconds=1,
                frame_height=480,
                capture_rollout=capture_rollout,
                pre_shot_grabber=pre_shot_grabber,
                video_runner=video_runner,
            )
            action_log = json.loads((Path(tmp) / "action_log.json").read_text(encoding="utf-8"))
            jsonl_lines = (Path(tmp) / "action_log.jsonl").read_text(encoding="utf-8").splitlines()
            metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(action_log["trial_count"], 1)
        self.assertEqual(len(jsonl_lines), 1)
        self.assertEqual(json.loads(jsonl_lines[0])["action"], actions[0])
        self.assertEqual(manifest["action_log_path"], str(Path(tmp) / "action_log.json"))
        self.assertEqual(manifest["action_log_jsonl_path"], str(Path(tmp) / "action_log.jsonl"))
        self.assertNotIn("video_frames", runner_calls[0][runner_calls[0].index("-i") + 1])
        self.assertGreater(metadata["video_frame_count"], metadata["frame_count"])
        self.assertIn("video_overlay", metadata)

    def test_collect_rollouts_records_varied_trials_within_one_episode(self):
        actions = [
            {"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0},
            {"coordinate_frame": "absolute", "release": [240, 250], "tapTime": 45},
            {"coordinate_frame": "absolute", "release": [230, 240], "tapTime": 70},
        ]

        def capture_rollout(bridge, output_dir, **kwargs):
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True)
            from PIL import Image

            Image.new("RGB", (160, 90), (50, 60, 70)).save(frames_dir / "frame_000000.png", format="PNG")
            return {"frame_count": 1, "frames_dir": str(frames_dir)}

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                FakeBridge(),
                Path(tmp),
                actions,
                target_fps=1,
                duration_seconds=1,
                capture_rollout=capture_rollout,
                video_runner=lambda command, check, stdout, stderr: Path(command[-1]).write_bytes(b"mp4"),
            )
            action_log = json.loads((Path(tmp) / "action_log.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["replay_mode"], "same-episode-varied-trials")
        self.assertEqual(manifest["attempt_count"], 3)
        self.assertEqual(manifest["accepted_rollout_count"], 0)
        self.assertEqual(manifest["rollout_count"], 0)
        self.assertEqual(action_log["attempt_count"], 3)
        self.assertEqual(action_log["accepted_trial_count"], 0)
        self.assertEqual(action_log["trial_count"], 3)
        self.assertEqual([trial["shot_name"] for trial in action_log["trials"]], ["shot_001", "shot_002", "shot_003"])
        self.assertEqual(len({tuple(trial["action"].get("release", [])) for trial in action_log["trials"]}), 3)

    def test_collect_rollouts_preserves_valid_raw_evidence_when_video_encoding_fails(self):
        actions = [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}]
        runner_calls = []

        def capture_rollout(bridge, output_dir, **kwargs):
            from PIL import Image

            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True)
            Image.new("RGB", (20, 20), (10, 20, 30)).save(frames_dir / "frame_000000.png", format="PNG")
            Image.new("RGB", (20, 20), (110, 20, 30)).save(frames_dir / "frame_000001.png", format="PNG")
            metadata = {"frame_count": 2, "frames_dir": str(frames_dir)}
            (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            return metadata

        def video_runner(command, check, stdout, stderr):
            runner_calls.append(command)
            Path(command[-1]).write_bytes(b"partial mp4")
            raise subprocess.CalledProcessError(1, command)

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                FakeBridge(),
                Path(tmp),
                actions,
                target_fps=30,
                duration_seconds=1,
                capture_rollout=capture_rollout,
                video_runner=video_runner,
            )
            shot_dir = Path(tmp) / "shot_001"
            metadata = json.loads((shot_dir / "metadata.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["rollouts"][0]["frame_count"], 2)
            self.assertTrue(manifest["rollouts"][0]["accepted"])
            self.assertIn("ffmpeg", metadata["video_error"])
            self.assertNotIn("video_path", metadata)
            self.assertNotIn("video_frames_dir", metadata)
            self.assertNotIn("video_input_pattern", metadata)
            self.assertNotIn(runner_calls[0][runner_calls[0].index("-i") + 1], json.dumps(metadata))
            self.assertNotIn(runner_calls[0][-1], json.dumps(metadata))
            self.assertTrue(metadata["artifact_validation"]["accepted"])
            self.assertTrue((shot_dir / "frames" / "frame_000000.png").is_file())
            self.assertFalse((shot_dir / "video_frames").exists())
            self.assertFalse((shot_dir / "rollout.mp4").exists())
            self.assertEqual(Path(runner_calls[0][-1]).name, ".rollout.tmp.mp4")
            self.assertFalse((shot_dir / ".rollout.tmp.mp4").exists())

    def test_parser_defaults_to_high_fps_for_review_collection(self):
        args = build_parser().parse_args(["--output-dir", "data/review"])

        self.assertEqual(args.fps, 30.0)
        self.assertNotEqual((args.host, args.port), (args.physics_host, args.physics_port))

    def test_realized_coverage_strata_uses_engine_events_and_support(self):
        self.assertEqual(
            realized_coverage_strata(PHYSICS_FIXTURES),
            (
                "collision",
                "destruction",
                "explosion",
                "level clear",
                "persistent support",
                "pig removal",
                "stability transitions",
                "support change",
            ),
        )
        no_events = (
            Path(__file__).parent
            / "fixtures"
            / "physics_capture_v1_macro"
            / "no_events"
            / "shot_001"
        )
        self.assertEqual(realized_coverage_strata(no_events), ("no-contact/miss",))

    def test_select_level_in_display_clicks_play_inputs_level_and_confirms(self):
        calls = []
        sleeps = []

        def runner(command, check):
            calls.append(command)

        select_level_in_display(3, runner=runner, sleeper=sleeps.append)

        self.assertEqual(
            calls,
            [
                ["xdotool", "mousemove", "512", "390", "click", "1"],
                ["xdotool", "mousemove", "495", "343", "click", "1"],
                ["xdotool", "type", "3"],
                ["xdotool", "mousemove", "492", "465", "click", "1"],
            ],
        )
        self.assertTrue(sleeps)

    def test_invalid_attempt_retries_are_rejected_before_collection(self):
        self._assert_fresh_engine_retries_are_rejected()

    def test_retry_overwrite_path_is_rejected_before_collection(self):
        self._assert_fresh_engine_retries_are_rejected()

    def test_retry_exhaustion_path_is_rejected_before_collection(self):
        self._assert_fresh_engine_retries_are_rejected()

    def test_pre_shot_guard_retry_path_is_rejected_before_collection(self):
        self._assert_fresh_engine_retries_are_rejected()

    def test_collect_fresh_engine_rollouts_restarts_engine_per_action(self):
        actions = [
            {"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0},
            {"coordinate_frame": "absolute", "release": [240, 250], "tapTime": 45},
        ]
        bridges = [FakeBridge(), FakeBridge()]
        processes = []
        selected_levels = []

        class FakeProcess:
            def __init__(self, pid):
                self.pid = pid
                self.terminated = False
                self.waited = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                self.waited = True

        def start_engine_func(game_dir, headless):
            process = FakeProcess(1000 + len(processes))
            processes.append(process)
            return process

        def connect_func(host, port, timeout, deadline_seconds):
            return bridges.pop(0)

        def capture_rollout(bridge, output_dir, **kwargs):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "metadata.json").write_text(json.dumps({"frame_count": 1}), encoding="utf-8")
            return {"frame_count": 1}

        with TemporaryDirectory() as tmp:
            manifest = collect_fresh_engine_rollouts(
                Path(tmp),
                actions,
                game_dir=Path("game"),
                host="127.0.0.1",
                port=2004,
                agent_id=28888,
                speed=1,
                connect_timeout=1,
                read_timeout=2,
                prepare_timeout=3,
                frame_height=480,
                fast=True,
                headless=False,
                target_fps=1,
                duration_seconds=1,
                ui_level=1,
                ui_settle_seconds=0,
                start_engine_func=start_engine_func,
                connect_func=connect_func,
                prepare_func=lambda bridge, timeout, poll_delay: bridge.get_game_state(),
                capture_rollout=capture_rollout,
                select_level_func=lambda level: selected_levels.append(level),
            )

            saved_manifest = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["attempt_count"], 2)
        self.assertEqual(manifest["accepted_rollout_count"], 0)
        self.assertEqual(manifest["rollout_count"], 0)
        self.assertEqual(saved_manifest["replay_mode"], "fresh-engine-per-rollout")
        self.assertEqual(selected_levels, [1, 1])
        self.assertEqual(len(processes), 2)
        self.assertTrue(all(process.terminated and process.waited for process in processes))

    def test_collect_fresh_engine_rollouts_waits_after_engine_start_and_connect_before_configure(self):
        events = []

        class FakeProcess:
            pid = 1234

            def poll(self):
                return None

            def terminate(self):
                events.append("terminate")

            def wait(self, timeout=None):
                events.append("wait")

        def start_engine_func(game_dir, headless):
            events.append("start")
            return FakeProcess()

        def sleeper(seconds):
            events.append(("sleep", seconds))

        def connect_func(host, port, timeout, deadline_seconds):
            events.append("connect")
            return FakeBridge()

        def prepare_func(bridge, timeout, poll_delay):
            events.append("prepare")
            return bridge.get_game_state()

        def capture_rollout(bridge, output_dir, **kwargs):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "metadata.json").write_text(json.dumps({"frame_count": 1, "frames_dir": str(output_dir / "frames")}), encoding="utf-8")
            return {"frame_count": 1, "frames_dir": str(output_dir / "frames")}

        with TemporaryDirectory() as tmp:
            collect_fresh_engine_rollouts(
                Path(tmp),
                [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}],
                game_dir=Path("game"),
                host="127.0.0.1",
                port=2004,
                agent_id=28888,
                speed=1,
                connect_timeout=1,
                read_timeout=2,
                prepare_timeout=3,
                frame_height=480,
                fast=True,
                headless=False,
                target_fps=1,
                duration_seconds=1,
                ui_level=None,
                ui_settle_seconds=0,
                engine_settle_seconds=7,
                agent_settle_seconds=11,
                start_engine_func=start_engine_func,
                connect_func=connect_func,
                prepare_func=prepare_func,
                capture_rollout=capture_rollout,
                sleeper=sleeper,
                video_runner=lambda command, check, stdout, stderr: Path(command[-1]).write_bytes(b"mp4"),
            )

        self.assertEqual(events[:5], ["start", ("sleep", 7), "connect", ("sleep", 11), "prepare"])

    def test_prepare_timeout_retry_path_is_rejected_before_collection(self):
        self._assert_fresh_engine_retries_are_rejected()

    def test_collect_fresh_engine_rollouts_anchors_slingshot_relative_actions_from_symbolic_state(self):
        actions = [
            {
                "action_type": "drag_hold_release",
                "coordinate_frame": "slingshot_relative",
                "drag_start": [300, 220],
                "drag_release": [50, 40],
                "tapTime": 0,
            }
        ]
        bridge = FakeBridge()
        bridge.symbolic_state = [
            {
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"label": "Slingshot"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[100, 150], [100, 190], [140, 190], [140, 150]]],
                        },
                    }
                ]
            }
        ]
        processes = []

        class FakeProcess:
            def __init__(self, pid):
                self.pid = pid
                self.terminated = False
                self.waited = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                self.waited = True

        def start_engine_func(game_dir, headless):
            process = FakeProcess(2000 + len(processes))
            processes.append(process)
            return process

        def connect_func(host, port, timeout, deadline_seconds):
            return bridge

        def capture_rollout(bridge, output_dir, **kwargs):
            output_dir.mkdir(parents=True, exist_ok=True)
            payload = {"frame_count": 1, "action": kwargs["action"]}
            (output_dir / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")
            return {"frame_count": 1}

        with TemporaryDirectory() as tmp:
            manifest = collect_fresh_engine_rollouts(
                Path(tmp),
                actions,
                game_dir=Path("game"),
                host="127.0.0.1",
                port=2004,
                agent_id=28888,
                speed=1,
                connect_timeout=1,
                read_timeout=2,
                prepare_timeout=3,
                frame_height=480,
                fast=True,
                headless=False,
                target_fps=1,
                duration_seconds=1,
                ui_level=1,
                ui_settle_seconds=0,
                start_engine_func=start_engine_func,
                connect_func=connect_func,
                prepare_func=lambda bridge, timeout, poll_delay: bridge.get_game_state(),
                capture_rollout=capture_rollout,
                select_level_func=lambda level: None,
            )

            saved_metadata = json.loads((Path(tmp) / "shot_001" / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["rollouts"][0]["slingshot_reference"], {"gameX": 118, "gameY": 315, "canvasX": 118, "canvasY": 164})
        self.assertEqual(manifest["rollouts"][0]["action"]["drag_start"], [118, 315])
        self.assertEqual(manifest["rollouts"][0]["action"]["drag_release"], [50, 40])
        self.assertEqual(bridge.shots[0], (168, 204, 0, True, 600))
        self.assertEqual(saved_metadata["slingshot_reference"], {"gameX": 118, "gameY": 315, "canvasX": 118, "canvasY": 164})
        self.assertTrue(all(process.terminated and process.waited for process in processes))

    def test_collect_fresh_engine_rollouts_stops_engine_when_disconnect_raises(self):
        class DisconnectFailBridge(FakeBridge):
            def disconnect(self):
                raise RuntimeError("disconnect failed")

        class FakeProcess:
            pid = 3000

            def __init__(self):
                self.terminated = False
                self.waited = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                self.waited = True

        bridge = DisconnectFailBridge()
        process = FakeProcess()

        def capture_rollout(bridge, output_dir, **kwargs):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "metadata.json").write_text(json.dumps({"frame_count": 1}), encoding="utf-8")
            return {"frame_count": 1}

        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "disconnect failed"):
                collect_fresh_engine_rollouts(
                    Path(tmp),
                    [{"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}],
                    game_dir=Path("game"),
                    host="127.0.0.1",
                    port=2004,
                    agent_id=28888,
                    speed=1,
                    connect_timeout=1,
                    read_timeout=2,
                    prepare_timeout=3,
                    frame_height=480,
                    fast=True,
                    headless=False,
                    target_fps=1,
                    duration_seconds=1,
                    ui_level=None,
                    ui_settle_seconds=0,
                    start_engine_func=lambda game_dir, headless: process,
                    connect_func=lambda host, port, timeout, deadline_seconds: bridge,
                    prepare_func=lambda bridge, timeout, poll_delay: bridge.get_game_state(),
                    capture_rollout=capture_rollout,
                )

        self.assertTrue(process.terminated)
        self.assertTrue(process.waited)

    def test_collect_fresh_engine_attempt_atomically_publishes_accepted_attempt(self):
        action = {"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}
        calls = []

        def legacy_collect(stage, actions, *, fresh_engine_attempts, **_kwargs):
            calls.append(
                (
                    stage,
                    actions,
                    fresh_engine_attempts,
                    _kwargs["expected_initial_engine_state_identity"],
                )
            )
            shot = stage / "shot_001"
            shot.mkdir(parents=True)
            metadata = {
                "capture_contract": "physics_capture_v1",
                "initial_engine_state_identity": "initial-state",
                "intervention_event_id": "event-1",
                "termination_reason": "rollout_ceiling",
                "termination_fixed_step": 3,
                "termination_event_id": None,
                "terminal_state_fixed_step": 3,
            }
            (shot / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            manifest = {
                "attempt_count": 1,
                "rollouts": [
                    {
                        "name": "shot_001",
                        "accepted": True,
                        "artifact_validation": {"accepted": True},
                    }
                ],
            }
            (stage / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            return manifest

        with (
            TemporaryDirectory() as temporary,
            patch(
                "scripts.collect_rollouts.collect_fresh_engine_rollouts",
                side_effect=legacy_collect,
            ),
            patch(
                "scripts.collect_rollouts.realized_coverage_strata",
                return_value=("collision",),
            ),
        ):
            root = Path(temporary)
            result = collect_fresh_engine_attempt(
                root,
                action,
                attempt_id="attempt-accepted",
                attempt_number=1,
                expected_initial_engine_state_identity="initial-state",
                game_dir=Path("game"),
            )

            accepted = root / "accepted" / "attempt-accepted"
            self.assertEqual(
                calls,
                [(root / ".attempt-attempt-accepted.tmp", [action], 1, "initial-state")],
            )
            self.assertTrue(accepted.is_dir())
            self.assertFalse((root / ".attempt-attempt-accepted.tmp").exists())
            self.assertEqual(result["artifact_path"], str(accepted / "shot_001"))
            self.assertEqual(result["realized_coverage_strata"], ["collision"])
            self.assertTrue(result["eligible"])
            self.assertEqual(result["status"], "accepted")
            self.assertFalse((root / "shot_001").exists())

    def test_collect_fresh_engine_attempt_quarantines_post_validation_publication_failure(self):
        action = {"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0}

        def legacy_collect(stage, _actions, *, fresh_engine_attempts, **_kwargs):
            self.assertEqual(fresh_engine_attempts, 1)
            shot = stage / "shot_001"
            shot.mkdir(parents=True)
            metadata = {
                "capture_contract": "physics_capture_v1",
                "initial_engine_state_identity": "initial-state",
                "intervention_event_id": "event-1",
                "termination_reason": "rollout_ceiling",
                "termination_fixed_step": 3,
                "termination_event_id": None,
                "terminal_state_fixed_step": 3,
            }
            (shot / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            manifest = {
                "attempt_count": 1,
                "rollouts": [
                    {
                        "name": "shot_001",
                        "accepted": True,
                        "artifact_validation": {"accepted": True},
                    }
                ],
            }
            (stage / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            return manifest

        with TemporaryDirectory() as temporary, patch(
            "scripts.collect_rollouts.collect_fresh_engine_rollouts",
            side_effect=legacy_collect,
        ), patch(
            "scripts.collect_rollouts._rewrite_staged_attempt_paths",
            side_effect=OSError("publication rewrite failed"),
        ):
            root = Path(temporary)
            result = collect_fresh_engine_attempt(
                root,
                action,
                attempt_id="attempt-publication-failure",
                attempt_number=1,
                expected_initial_engine_state_identity="initial-state",
                game_dir=Path("game"),
            )

            staging = root / ".attempt-attempt-publication-failure.tmp"
            quarantine = root / "quarantine" / "attempt-publication-failure"
            failure_path = quarantine / "failure.json"
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            self.assertFalse(staging.exists())
            self.assertTrue((quarantine / "shot_001" / "metadata.json").is_file())
            self.assertTrue((quarantine / "manifest.json").is_file())
            self.assertEqual(result["quarantine_path"], str(quarantine))
            self.assertEqual(result["failure_manifest_path"], str(failure_path))
            self.assertEqual(result["failure_code"], "attempt_publication_error")
            self.assertEqual(failure["failure_class"], "permanent")
            self.assertFalse(failure["retryable"])
            self.assertEqual(failure["retry_decision"], "stop")

    def test_loaded_plan_does_not_retry_post_validation_publication_oserror(self):
        from scripts.collection_plan import (
            create_collection_plan,
            execute_collection_plan,
            load_collection_plan,
            write_collection_plan,
        )
        from scripts.collect_rollouts import _json_compatible_action
        from tests.test_collection_plan import plan_arguments

        calls = []

        def legacy_collect(stage, _actions, *, fresh_engine_attempts, **_kwargs):
            self.assertEqual(fresh_engine_attempts, 1)
            shot = stage / "shot_001"
            shot.mkdir(parents=True)
            metadata = {
                "capture_contract": "physics_capture_v1",
                "initial_engine_state_identity": "runtime-initial-state-v1:fixture",
                "intervention_event_id": "event-1",
                "termination_reason": "rollout_ceiling",
                "termination_fixed_step": 3,
                "termination_event_id": None,
                "terminal_state_fixed_step": 3,
            }
            (shot / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            manifest = {
                "attempt_count": 1,
                "rollouts": [
                    {
                        "name": "shot_001",
                        "accepted": True,
                        "artifact_validation": {"accepted": True},
                    }
                ],
            }
            (stage / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            return manifest

        with TemporaryDirectory() as temporary, patch(
            "scripts.collect_rollouts.collect_fresh_engine_rollouts",
            side_effect=legacy_collect,
        ), patch(
            "scripts.collect_rollouts._rewrite_staged_attempt_paths",
            side_effect=OSError("publication rewrite failed"),
        ):
            root = Path(temporary)
            plan_path = root / "collection-plan.json"
            output_root = root / "output"
            write_collection_plan(create_collection_plan(**plan_arguments()), plan_path)
            loaded_plan = load_collection_plan(plan_path)

            def runtime(request):
                calls.append(request)
                result = collect_fresh_engine_attempt(
                    output_root,
                    _json_compatible_action(request.interface_action),
                    attempt_id=request.attempt_id,
                    attempt_number=request.attempt_number,
                    expected_initial_engine_state_identity=request.expected_initial_engine_state_identity,
                    game_dir=Path("game"),
                )
                self.assertEqual(result["failure_code"], "attempt_publication_error")
                return result

            report = execute_collection_plan(loaded_plan, runtime, output_root)

            self.assertEqual(len(calls), 2)
            self.assertEqual([request.attempt_number for request in calls], [1, 1])
            self.assertEqual(report["failed_count"], 2)
            self.assertEqual(len(report["attempt_ledger"]), 2)
            for entry in report["attempt_ledger"]:
                failure = json.loads(
                    (output_root / entry["failure_manifest_path"]).read_text(encoding="utf-8")
                )
                self.assertEqual(entry["failure_code"], "attempt_publication_error")
                self.assertEqual(failure["failure_class"], "permanent")
                self.assertFalse(failure["retryable"])
                self.assertEqual(failure["retry_decision"], "stop")
                self.assertTrue(Path(entry["quarantine_path"]).is_dir())

    def test_collect_fresh_engine_attempt_quarantines_startup_failure_without_retry(self):
        starts = []

        def start_engine_func(_game_dir, _headless):
            starts.append("start")
            raise OSError("engine unavailable")

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = collect_fresh_engine_attempt(
                root,
                {"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0},
                attempt_id="attempt-startup-failure",
                attempt_number=2,
                expected_initial_engine_state_identity="initial-state",
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
                target_fps=1,
                duration_seconds=1,
                ui_level=None,
                ui_settle_seconds=0,
                fresh_engine_attempts=9,
                start_engine_func=start_engine_func,
            )

            quarantine = root / "quarantine" / "attempt-startup-failure"
            failure_path = quarantine / "failure.json"
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            self.assertEqual(starts, ["start"])
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["quarantine_path"], str(quarantine))
            self.assertEqual(result["failure_manifest_path"], str(failure_path))
            self.assertTrue(quarantine.is_dir())
            self.assertFalse((root / ".attempt-attempt-startup-failure.tmp").exists())
            self.assertEqual(failure["attempt_id"], "attempt-startup-failure")
            self.assertEqual(failure["attempt_number"], 2)
            self.assertEqual(failure["quarantine_path"], str(quarantine))
            self.assertEqual(failure["failure_code"], result["failure_code"])
            self.assertTrue(failure["reason"])

    def test_collect_fresh_engine_attempt_quarantines_unclassified_runtime_error_permanently(self):
        with TemporaryDirectory() as temporary, patch(
            "scripts.collect_rollouts.collect_fresh_engine_rollouts",
            side_effect=RuntimeError("connection reset while decoding invalid capture"),
        ):
            root = Path(temporary)
            result = collect_fresh_engine_attempt(
                root,
                {"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0},
                attempt_id="attempt-runtime-error",
                attempt_number=1,
                expected_initial_engine_state_identity="initial-state",
                game_dir=Path("game"),
            )

            quarantine = root / "quarantine" / "attempt-runtime-error"
            failure_path = quarantine / "failure.json"
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failure_code"], "collection_runtime_error")
            self.assertEqual(result["quarantine_path"], str(quarantine))
            self.assertEqual(result["failure_manifest_path"], str(failure_path))
            self.assertTrue(quarantine.is_dir())
            self.assertEqual(failure["failure_class"], "permanent")
            self.assertFalse(failure["retryable"])
            self.assertEqual(failure["retry_decision"], "stop")

    def test_collect_fresh_engine_attempt_rejects_missing_physics_evidence_permanently(self):
        def legacy_collect(stage, _actions, *, fresh_engine_attempts, **_kwargs):
            self.assertEqual(fresh_engine_attempts, 1)
            shot = stage / "shot_001"
            shot.mkdir(parents=True)
            metadata = {
                "initial_engine_state_identity": "initial-state",
                "intervention_event_id": "event-1",
                "termination_reason": "rollout_ceiling",
                "termination_fixed_step": 3,
                "termination_event_id": None,
                "terminal_state_fixed_step": 3,
            }
            (shot / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            return {
                "attempt_count": 1,
                "rollouts": [
                    {
                        "name": "shot_001",
                        "accepted": True,
                        "artifact_validation": {"accepted": True},
                    }
                ],
            }

        with TemporaryDirectory() as temporary, patch(
            "scripts.collect_rollouts.collect_fresh_engine_rollouts",
            side_effect=legacy_collect,
        ):
            root = Path(temporary)
            result = collect_fresh_engine_attempt(
                root,
                {"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 0},
                attempt_id="attempt-missing-evidence",
                attempt_number=1,
                expected_initial_engine_state_identity="initial-state",
                game_dir=Path("game"),
            )

            failure = json.loads(
                (root / "quarantine" / "attempt-missing-evidence" / "failure.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(result["status"], "rejected")
            self.assertEqual(result["failure_code"], "missing_required_evidence")
            self.assertFalse(result["eligible"])
            self.assertEqual(failure["failure_class"], "permanent")
            self.assertFalse(failure["retryable"])
            self.assertEqual(failure["retry_decision"], "stop")

    def test_write_action_plan_dry_run_writes_actions_without_bridge(self):
        with TemporaryDirectory() as tmp:
            path = write_action_plan(Path(tmp), count=3)

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["action_count"], 3)
            self.assertEqual(len(payload["actions"]), 3)
            self.assertEqual(payload["actions"][0]["action_type"], "drag_hold_release")

    def test_main_dry_run_writes_bidirectional_generated_action_plan(self):
        with TemporaryDirectory() as tmp:
            args = [
                "collect_rollouts.py",
                "--output-dir",
                tmp,
                "--count",
                "4",
                "--bidirectional-launches",
                "--dry-run",
            ]

            with patch("sys.argv", args):
                main()

            payload = json.loads((Path(tmp) / "action_plan.json").read_text(encoding="utf-8"))

        self.assertEqual(
            [action["drag_release"][0] > 0 for action in payload["actions"]],
            [False, True, False, True],
        )

    def test_main_dry_run_does_not_require_collection_scenario_inputs(self):
        with TemporaryDirectory() as tmp:
            args = [
                "collect_rollouts.py",
                "--output-dir",
                tmp,
                "--count",
                "1",
                "--physics-capture-v1",
                "--fresh-engine-per-rollout",
                "--dry-run",
            ]

            with patch("sys.argv", args):
                main()

            payload = json.loads((Path(tmp) / "action_plan.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["action_count"], 1)

    def test_load_actions_from_action_log_returns_exact_logged_actions(self):
        logged_actions = [
            {
                "action_type": "drag_hold_release",
                "coordinate_frame": "slingshot_relative",
                "drag_start": [97, 227],
                "drag_release": [-80, 7],
                "tapTime": 0,
                "holdTime": 1000,
                "slingshot_reference": {"gameX": 97, "gameY": 227, "canvasX": 97, "canvasY": 252},
            },
            {"coordinate_frame": "absolute", "release": [250, 260], "tapTime": 45},
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "action_log.json"
            path.write_text(
                json.dumps(
                    {
                        "episode_dir": "original_episode",
                        "trial_count": 2,
                        "trials": [
                            {"shot_name": "shot_001", "action": logged_actions[0]},
                            {"shot_name": "shot_002", "action": logged_actions[1]},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            actions = load_actions_from_action_log(path)

        self.assertEqual(actions, logged_actions)

    def test_load_actions_from_action_log_rejects_malformed_log(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "action_log.json"
            path.write_text(json.dumps({"trials": [{"shot_name": "shot_001"}]}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "trial 1 is missing action"):
                load_actions_from_action_log(path)

    def test_main_wires_actions_from_log_without_reanchoring(self):
        logged_action = {
            "action_type": "drag_hold_release",
            "coordinate_frame": "slingshot_relative",
            "drag_start": [97, 227],
            "drag_release": [-80, 7],
            "tapTime": 0,
            "holdTime": 1000,
            "slingshot_reference": {"gameX": 97, "gameY": 227, "canvasX": 97, "canvasY": 252},
        }
        with TemporaryDirectory() as tmp:
            action_log = Path(tmp) / "action_log.json"
            action_log.write_text(
                json.dumps({"trial_count": 1, "trials": [{"shot_name": "shot_001", "action": logged_action}]}),
                encoding="utf-8",
            )
            args = [
                "collect_rollouts.py",
                "--output-dir",
                tmp,
                "--actions-from-log",
                str(action_log),
                "--bidirectional-launches",
                "--no-prepare",
            ]

            with (
                patch("sys.argv", args),
                patch("scripts.collect_rollouts.connect_or_start_engine", return_value=(FakeBridge(), None)),
                patch("scripts.collect_rollouts.generate_diverse_drag_release_actions") as generate_actions,
                patch("scripts.collect_rollouts.collect_rollouts", return_value={"rollout_count": 1}) as collect,
            ):
                main()

        self.assertEqual(collect.call_args.args[2], [logged_action])
        self.assertFalse(collect.call_args.kwargs["anchor_actions"])
        generate_actions.assert_not_called()

    def test_main_requires_collection_plan_for_production_plan(self):
        with TemporaryDirectory() as tmp:
            args = [
                "collect_rollouts.py",
                "--output-dir",
                tmp,
                "--production-plan",
                str(Path(tmp) / "production-plan.json"),
            ]

            stderr = io.StringIO()
            with patch("sys.argv", args), redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--production-plan requires --collection-plan", stderr.getvalue())

    def test_main_executes_collection_through_production_plan(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_path = root / "collection-plan.json"
            production_path = root / "production-plan.json"
            output_path = root / "output"
            args = [
                "collect_rollouts.py",
                "--output-dir",
                str(output_path),
                "--fresh-engine-per-rollout",
                "--collection-plan",
                str(plan_path),
                "--production-plan",
                str(production_path),
            ]

            with (
                patch("sys.argv", args),
                patch("scripts.collection_plan.load_collection_plan", return_value="loaded-plan"),
                patch(
                    "scripts.production_plan.execute_production_plan",
                    return_value={"accepted_count": 1, "failed_count": 0, "rejected_count": 0},
                ) as execute,
            ):
                main()

        execute.assert_called_once()
        self.assertEqual(execute.call_args.args[0], "loaded-plan")
        self.assertEqual(execute.call_args.args[1], production_path)
        self.assertEqual(execute.call_args.args[3], output_path)

    def test_main_requires_collection_plan_for_fresh_engine_collection(self):
        with TemporaryDirectory() as tmp:
            args = [
                "collect_rollouts.py",
                "--output-dir",
                tmp,
                "--fresh-engine-per-rollout",
            ]

            stderr = io.StringIO()
            with patch("sys.argv", args), redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--collection-plan", stderr.getvalue())

    def test_main_executes_fresh_engine_collection_plan_through_atomic_runtime(self):
        frozen_action = {
            "coordinate_frame": "slingshot_relative",
            "drag_start": [100, 200],
            "drag_release": [20, 30],
            "tapTime": 0,
            "holdTime": 600,
        }
        request = type(
            "RuntimeInput",
            (),
            {
                "attempt_id": "attempt-plan-1",
                "attempt_number": 2,
                "expected_initial_engine_state_identity": "expected-initial-state",
                "interface_action": frozen_action,
            },
        )()

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_path = root / "collection-plan.json"
            args = [
                "collect_rollouts.py",
                "--output-dir",
                str(root / "output"),
                "--fresh-engine-per-rollout",
                "--collection-plan",
                str(plan_path),
            ]

            def execute_plan(loaded, runtime, output_root):
                self.assertEqual(loaded, "loaded-plan")
                self.assertEqual(output_root, root / "output")
                result = runtime(request)
                self.assertEqual(result["status"], "accepted")
                return {"accepted_count": 1, "failed_count": 0, "rejected_count": 0}

            with (
                patch("sys.argv", args),
                patch("scripts.collection_plan.load_collection_plan", return_value="loaded-plan") as load_plan,
                patch("scripts.collection_plan.execute_collection_plan", side_effect=execute_plan) as execute,
                patch(
                    "scripts.collect_rollouts.collect_fresh_engine_attempt",
                    return_value={
                        "status": "accepted",
                        "reason": None,
                        "failure_code": None,
                        "realized_coverage_strata": [],
                        "eligible": True,
                        "artifact_path": "accepted/attempt-plan-1",
                        "quarantine_path": None,
                        "failure_manifest_path": None,
                    },
                ) as attempt,
                patch("scripts.collect_rollouts.generate_diverse_drag_release_actions") as generate_actions,
            ):
                main()

        load_plan.assert_called_once_with(plan_path)
        execute.assert_called_once()
        self.assertEqual(attempt.call_args.args[0], root / "output")
        self.assertEqual(attempt.call_args.args[1], frozen_action)
        self.assertIsNot(attempt.call_args.args[1], frozen_action)
        self.assertEqual(attempt.call_args.kwargs["attempt_id"], "attempt-plan-1")
        self.assertEqual(attempt.call_args.kwargs["attempt_number"], 2)
        self.assertEqual(
            attempt.call_args.kwargs["expected_initial_engine_state_identity"],
            "expected-initial-state",
        )
        generate_actions.assert_not_called()

    def test_main_selects_scenario_input_for_each_plan_request(self):
        frozen_action = {
            "coordinate_frame": "slingshot_relative",
            "drag_start": [100, 200],
            "drag_release": [20, 30],
            "tapTime": 0,
            "holdTime": 600,
        }
        requests = [
            type(
                "RuntimeInput",
                (),
                {
                    "plan_identity": "plan-identity",
                    "plan_version": 1,
                    "scenario_id": scenario_id,
                    "scenario_identity": f"scenario-identity-{scenario_id}",
                    "intervention_id": f"intervention-{scenario_id}",
                    "intervention_identity": f"intervention-identity-{scenario_id}",
                    "attempt_id": f"attempt-{scenario_id}",
                    "attempt_number": 1,
                    "expected_initial_engine_state_identity": f"initial-{scenario_id}",
                    "interface_action": frozen_action,
                },
            )()
            for scenario_id in ("baseline", "novel")
        ]
        loaded_plan = type(
            "LoadedPlan",
            (),
            {
                "plan": type(
                    "Plan",
                    (),
                    {
                        "scenarios": tuple(
                            type("Scenario", (), {"scenario_id": scenario_id})()
                            for scenario_id in ("baseline", "novel")
                        )
                    },
                )()
            },
        )()

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for game_dir in (root / "baseline-runtime", root / "novel-runtime", root / "attempt-baseline-runtime", root / "attempt-novel-runtime"):
                game_dir.mkdir()
                (game_dir / "game_playing_interface.jar").write_bytes(b"jar")
            args = [
                "collect_rollouts.py",
                "--output-dir",
                str(root / "output"),
                "--fresh-engine-per-rollout",
                "--collection-plan",
                str(root / "collection-plan.json"),
                "--scenario-input",
                "baseline",
                str(root / "baseline.json"),
                str(root / "baseline.xml"),
                str(root / "baseline-runtime"),
                "--scenario-input",
                "novel",
                str(root / "novel.json"),
                str(root / "novel.xml"),
                str(root / "novel-runtime"),
                "--attempt-input",
                "attempt-baseline",
                str(root / "attempt-baseline-runtime"),
                "--attempt-input",
                "attempt-novel",
                str(root / "attempt-novel-runtime"),
            ]

            def execute_plan(loaded, runtime, output_root):
                self.assertIs(loaded, loaded_plan)
                for request in requests:
                    runtime(request)
                return {"accepted_count": 2, "failed_count": 0, "rejected_count": 0}

            with (
                patch("sys.argv", args),
                patch("scripts.collect_rollouts.load_manifest", side_effect=("baseline-manifest", "novel-manifest")),
                patch("scripts.collect_rollouts._scenario_generation_version", return_value="importer:1"),
                patch("scripts.collection_plan.load_collection_plan", return_value=loaded_plan),
                patch("scripts.collection_plan.execute_collection_plan", side_effect=execute_plan),
                patch(
                    "scripts.collect_rollouts.collect_fresh_engine_attempt",
                    return_value={
                        "status": "accepted",
                        "reason": None,
                        "failure_code": None,
                        "realized_coverage_strata": [],
                        "eligible": True,
                        "artifact_path": "accepted",
                        "quarantine_path": None,
                        "failure_manifest_path": None,
                    },
                ) as attempt,
            ):
                main()

        self.assertEqual(
            [
                (
                    call.kwargs["game_dir"],
                    call.kwargs["ui_level"],
                    call.kwargs["scenario_manifest"],
                    call.kwargs["scenario_context_override"],
                )
                for call in attempt.call_args_list
            ],
            [
                (
                    root / "attempt-baseline-runtime",
                    None,
                    "baseline-manifest",
                    {
                        "version_envelope": {
                            "player_sha256": None,
                            "protocol_sha256": None,
                            "archive_sha256": None,
                            "generator_version": "importer:1",
                        },
                        "plan_identity": "plan-identity",
                        "plan_version": 1,
                        "scenario_id": "baseline",
                        "scenario_identity": "scenario-identity-baseline",
                        "intervention_id": "intervention-baseline",
                        "intervention_identity": "intervention-identity-baseline",
                        "attempt_id": "attempt-baseline",
                        "attempt_number": 1,
                    },
                ),
                (
                    root / "attempt-novel-runtime",
                    None,
                    "novel-manifest",
                    {
                        "version_envelope": {
                            "player_sha256": None,
                            "protocol_sha256": None,
                            "archive_sha256": None,
                            "generator_version": "importer:1",
                        },
                        "plan_identity": "plan-identity",
                        "plan_version": 1,
                        "scenario_id": "novel",
                        "scenario_identity": "scenario-identity-novel",
                        "intervention_id": "intervention-novel",
                        "intervention_identity": "intervention-identity-novel",
                        "attempt_id": "attempt-novel",
                        "attempt_number": 1,
                    },
                ),
            ],
        )

    def test_loaded_plan_runtime_quarantines_failed_thawed_action_without_staging_orphan(self):
        from scripts.collection_plan import (
            create_collection_plan,
            write_collection_plan,
        )
        from tests.test_collection_plan import plan_arguments

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "collection-plan.json"
            write_collection_plan(create_collection_plan(**plan_arguments()), plan_path)
            output_root = root / "output"
            args = [
                "collect_rollouts.py",
                "--output-dir",
                str(output_root),
                "--fresh-engine-per-rollout",
                "--collection-plan",
                str(plan_path),
            ]

            with (
                patch("sys.argv", args),
                patch(
                    "scripts.collect_rollouts.collect_fresh_engine_rollouts",
                    side_effect=RuntimeError("synthetic collection failure"),
                ),
            ):
                main()

            report = json.loads(
                (output_root / "collection_plan_report.json").read_text(encoding="utf-8")
            )
            self.assertTrue(report["attempt_ledger"])
            self.assertTrue((output_root / "quarantine").is_dir())
            for entry in report["attempt_ledger"]:
                quarantine = Path(entry["quarantine_path"])
                failure_path = Path(entry["failure_manifest_path"])
                failure = json.loads(failure_path.read_text(encoding="utf-8"))
                self.assertEqual(entry["status"], "failed")
                self.assertEqual(entry["failure_code"], "collection_runtime_error")
                self.assertEqual(failure["action"]["drag_start"], [100, 200])
                self.assertEqual(failure["quarantine_path"], str(quarantine))
                self.assertTrue(quarantine.is_dir())
                self.assertFalse(
                    (output_root / f".attempt-{entry['attempt_id']}.tmp").exists()
                )

    def test_connect_or_start_engine_auto_starts_engine_after_connection_refusal(self):
        class FakeProcess:
            pid = 4321

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                return None

        args = type(
            "Args",
            (),
            {
                "host": "127.0.0.1",
                "port": 2004,
                "read_timeout": 3,
                "connect_timeout": 4,
                "game_headless": False,
            },
        )()
        bridge = FakeBridge()
        process = FakeProcess()

        with (
            patch("scripts.collect_rollouts.connect_with_retry", side_effect=[RuntimeError("connection refused"), bridge]),
            patch("scripts.collect_rollouts.start_engine", return_value=process) as start,
        ):
            result_bridge, result_process = __import__("scripts.collect_rollouts", fromlist=["connect_or_start_engine"]).connect_or_start_engine(args)

        self.assertIs(result_bridge, bridge)
        self.assertIs(result_process, process)
        self.assertEqual(start.call_count, 1)

    def test_connect_or_start_engine_forwards_custom_engine_ports(self):
        class FakeProcess:
            pid = 4321

            def poll(self):
                return None

        args = type(
            "Args",
            (),
            {
                "host": "127.0.0.1",
                "port": 2014,
                "read_timeout": 3,
                "connect_timeout": 4,
                "game_dir": Path("/tmp/novphy-worker-engine"),
                "game_headless": True,
                "engine_agent_port": 2014,
                "engine_game_port": 9011,
            },
        )()
        bridge = FakeBridge()
        process = FakeProcess()

        with (
            patch("scripts.collect_rollouts.connect_with_retry", side_effect=[RuntimeError("connection refused"), bridge]),
            patch("scripts.collect_rollouts.start_engine", return_value=process) as start,
        ):
            result_bridge, result_process = __import__("scripts.collect_rollouts", fromlist=["connect_or_start_engine"]).connect_or_start_engine(args)

        self.assertIs(result_bridge, bridge)
        self.assertIs(result_process, process)
        start.assert_called_once_with(Path("/tmp/novphy-worker-engine"), True, agent_port=2014, game_port=9011)

    def test_main_stops_explicit_started_engine_when_connection_fails(self):
        class FakeProcess:
            pid = 2468

            def __init__(self):
                self.terminated = False
                self.waited = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                self.waited = True

        process = FakeProcess()
        with TemporaryDirectory() as tmp:
            args = [
                "collect_rollouts.py",
                "--output-dir",
                tmp,
                "--count",
                "1",
                "--start-engine",
            ]

            with (
                patch("sys.argv", args),
                patch("scripts.collect_rollouts.start_engine", return_value=process),
                patch("scripts.collect_rollouts.connect_with_retry", side_effect=RuntimeError("connection refused")),
            ):
                with self.assertRaisesRegex(RuntimeError, "connection refused"):
                    main()

        self.assertTrue(process.terminated)
        self.assertTrue(process.waited)

    def test_stop_owned_engine_kills_process_after_terminate_timeout(self):
        class SlowProcess:
            pid = 4321

            def __init__(self):
                self.terminated = False
                self.killed = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                if not self.killed:
                    raise TimeoutError("still running")

            def kill(self):
                self.killed = True

        process = SlowProcess()

        stop_owned_engine(process)

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)

    def test_stop_owned_engine_terminates_process_group_for_started_engine(self):
        class GroupProcess:
            pid = 5432
            novphy_process_group = True

            def __init__(self):
                self.terminated = False
                self.killed = False
                self.waited = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.killed = True

            def wait(self, timeout=None):
                self.waited = True

        process = GroupProcess()
        with patch("scripts.collect_rollouts.os.getpgid", return_value=9876), patch("scripts.collect_rollouts.os.killpg") as killpg:
            stop_owned_engine(process)

        killpg.assert_called_once_with(9876, signal.SIGTERM)
        self.assertTrue(process.waited)
        self.assertFalse(process.terminated)
        self.assertFalse(process.killed)

    def test_stop_owned_engine_falls_back_to_process_when_group_lookup_fails(self):
        class MissingGroupProcess:
            pid = 6543
            novphy_process_group = True

            def __init__(self):
                self.terminated = False
                self.waited = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                self.waited = True

        process = MissingGroupProcess()
        with patch("scripts.collect_rollouts.os.getpgid", side_effect=ProcessLookupError), patch("scripts.collect_rollouts.os.killpg") as killpg:
            stop_owned_engine(process)

        killpg.assert_not_called()
        self.assertTrue(process.terminated)
        self.assertTrue(process.waited)

    def test_stop_owned_engine_escalates_process_group_after_timeout(self):
        class SlowGroupProcess:
            pid = 7654
            novphy_process_group = True

            def __init__(self):
                self.wait_calls = 0
                self.killed = False

            def poll(self):
                return None

            def kill(self):
                self.killed = True

            def wait(self, timeout=None):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    raise TimeoutError("still running")

        process = SlowGroupProcess()
        with patch("scripts.collect_rollouts.os.getpgid", return_value=8765), patch("scripts.collect_rollouts.os.killpg") as killpg:
            stop_owned_engine(process)

        self.assertEqual(killpg.call_args_list[0].args, (8765, signal.SIGTERM))
        self.assertEqual(killpg.call_args_list[1].args, (8765, signal.SIGKILL))
        self.assertEqual(process.wait_calls, 2)
        self.assertFalse(process.killed)

    def test_main_reports_auto_start_failure_without_traceback(self):
        args = ["collect_rollouts.py", "--output-dir", "data/rollout-plan-debug", "--count", "3"]
        stderr = io.StringIO()

        with (
            patch("sys.argv", args),
            patch("scripts.collect_rollouts.connect_with_retry", side_effect=RuntimeError("connection refused")),
            patch("scripts.collect_rollouts.start_engine", side_effect=FileNotFoundError("missing jar")),
        ):
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("could not start the local engine", stderr.getvalue())
        self.assertIn("--dry-run", stderr.getvalue())

    def test_main_preflights_output_dir_before_starting_engine(self):
        args = ["collect_rollouts.py", "--output-dir", "/data/collect_rollouts_debug", "--count", "1"]
        stderr = io.StringIO()

        with (
            patch("sys.argv", args),
            patch("scripts.collect_rollouts.ensure_output_dir", side_effect=PermissionError("denied")) as ensure,
            patch("scripts.collect_rollouts.start_engine") as start,
            patch("scripts.collect_rollouts.connect_with_retry") as connect,
        ):
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main()

        self.assertEqual(raised.exception.code, 2)
        ensure.assert_called_once()
        start.assert_not_called()
        connect.assert_not_called()
        self.assertIn("Cannot write output directory", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
