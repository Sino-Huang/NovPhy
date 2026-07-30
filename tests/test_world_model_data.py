import importlib.util
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

from PIL import Image

from scripts.prepare_rollout_dataset import (
    CollectionOptions,
    CollectionTargets,
    LevelEntry,
    PlannedEpisode,
    _is_canonically_complete_fresh_engine_episode,
    _safe_output_name,
    build_collection_plan,
    write_collection_plan,
)
import world_model.data as world_model_data


@dataclass(frozen=True, slots=True)
class RolloutFixtureSpec:
    frame_count: int = 5
    shot_count: int = 1
    fps: float = 30.0
    duration_seconds: float = 5.0


def make_complete_rollout_episode(
    output_dir: Path,
    spec: RolloutFixtureSpec = RolloutFixtureSpec(),
) -> Path:
    if spec.frame_count <= 0:
        raise world_model_data.ContractValueError("frame_count", "must be positive")
    if spec.shot_count <= 0:
        raise world_model_data.ContractValueError("shot_count", "must be positive")

    output_dir.mkdir(parents=True)
    accepted_trials = []
    attempts = []
    for shot_index in range(1, spec.shot_count + 1):
        shot_name = f"shot_{shot_index:03d}"
        shot_dir = output_dir / shot_name
        frames_dir = shot_dir / "frames"
        frames_dir.mkdir(parents=True)
        Image.new("RGB", (8, 6), (10, 20, 30)).save(
            shot_dir / "pre_shot.png",
            format="PNG",
        )
        frame_records = []
        for frame_index in range(spec.frame_count):
            frame_path = frames_dir / f"frame_{frame_index:06d}.png"
            Image.new(
                "RGB",
                (8, 6),
                (20 + frame_index, 40 + shot_index, 60),
            ).save(frame_path, format="PNG")
            frame_records.append(
                {
                    "index": frame_index,
                    "path": str(frame_path),
                    "t": frame_index / spec.fps,
                }
            )

        action = {
            "drag_start": [300, 220],
            "drag_release": [-80, 20],
            "holdTime": 120,
        }
        metadata = {
            "frame_count": spec.frame_count,
            "frames_dir": str(frames_dir),
            "frames": frame_records,
            "pre_shot_path": str(shot_dir / "pre_shot.png"),
            "action": action,
            "artifact_validation": {
                "accepted": True,
                "classification": "gameplay-valid",
                "retryable": False,
                "retry_decision": "accept",
            },
        }
        (shot_dir / "metadata.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        accepted_trials.append(
            {
                "shot_name": shot_name,
                "accepted": True,
                "action": action,
                "metadata_path": str(shot_dir / "metadata.json"),
            }
        )
        attempts.append(
            {
                "accepted": True,
                "attempt_status": "accepted",
                "artifact_validation": metadata["artifact_validation"],
            }
        )

    manifest = {
        "capture_source": "capture_desktop_rollout",
        "replay_mode": "fresh-engine-per-rollout",
        "target_fps": spec.fps,
        "duration_seconds": spec.duration_seconds,
        "ui_level": 1,
        "accepted_rollout_count": spec.shot_count,
        "rollout_count": spec.shot_count,
        "attempt_count": spec.shot_count,
        "attempts": attempts,
    }
    action_log = {
        "episode_dir": str(output_dir),
        "attempt_count": spec.shot_count,
        "accepted_trial_count": spec.shot_count,
        "invalid_attempts": [],
        "trial_count": spec.shot_count,
        "trials": accepted_trials,
        "accepted_trials": accepted_trials,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (output_dir / "action_log.json").write_text(json.dumps(action_log), encoding="utf-8")
    (output_dir / "action_log.jsonl").write_text(
        "".join(f"{json.dumps(trial)}\n" for trial in accepted_trials),
        encoding="utf-8",
    )
    return output_dir


class WorldModelDataFixtureTests(unittest.TestCase):
    def test_existing_collector_defaults_define_fixture_contract(self):
        # Given: the planner's current collector options.
        options = CollectionOptions()

        # When: their artifact-shaping values are observed.
        contract = (options.count, options.fps, options.duration)

        # Then: fixtures can mirror the established collector defaults.
        self.assertEqual(contract, (12, 30.0, 5.0))

    def test_world_model_package_is_available(self):
        # Given: the repository import path.
        package_name = "world_model"

        # When: the world-model package is discovered.
        package = importlib.util.find_spec(package_name)

        # Then: the package provides an importable contract surface.
        self.assertIsNotNone(package)

    def test_capture_contract_surface_is_exported(self):
        # Given: the importable data package.
        data_package = world_model_data

        # When: its capture descriptor export is inspected.
        descriptor_type = getattr(data_package, "CaptureContractDescriptor", None)

        # Then: the typed descriptor is public.
        self.assertIsNotNone(descriptor_type)

    def test_complete_fixture_has_real_pngs_and_collector_layout(self):
        # Given: a complete two-shot synthetic episode.
        with tempfile.TemporaryDirectory() as temporary:
            episode_dir = make_complete_rollout_episode(
                Path(temporary) / "train" / "episode_001",
                RolloutFixtureSpec(frame_count=3, shot_count=2),
            )

            # When: the fixture is checked through the current collector contract.
            options = CollectionOptions(count=2, fps=30.0, duration=5.0)
            is_complete = _is_canonically_complete_fresh_engine_episode(
                episode_dir,
                options,
            )

            # Then: all collector files, RGB frames, and five action values exist.
            self.assertTrue(is_complete)
            self.assertTrue((episode_dir / "manifest.json").is_file())
            self.assertTrue((episode_dir / "action_log.json").is_file())
            self.assertTrue((episode_dir / "action_log.jsonl").is_file())
            action_log = json.loads(
                (episode_dir / "action_log.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(action_log["accepted_trials"]), 2)
            first_action = action_log["accepted_trials"][0]["action"]
            self.assertEqual(
                [*first_action["drag_start"], *first_action["drag_release"], first_action["holdTime"]],
                [300, 220, -80, 20, 120],
            )
            frame_paths = sorted((episode_dir / "shot_001" / "frames").glob("*.png"))
            self.assertEqual(
                [path.name for path in frame_paths],
                ["frame_000000.png", "frame_000001.png", "frame_000002.png"],
            )
            for frame_path in frame_paths:
                self.assertEqual(frame_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                with Image.open(frame_path) as image:
                    self.assertEqual(image.mode, "RGB")

    def test_capture_contract_descriptor_is_immutable(self):
        # Given: the inferred legacy capture contract.
        descriptor = world_model_data.LEGACY_RGB_V1

        # When/Then: mutation is rejected by the frozen record.
        with self.assertRaises(FrozenInstanceError):
            descriptor.contract_name = "changed"

    def test_legacy_capture_contract_is_inferred_when_descriptor_is_absent(self):
        # Given: collector metadata without an explicit capture descriptor.
        explicit_descriptor = None

        # When: the capture contract is inferred.
        infer_contract = getattr(world_model_data, "infer_capture_contract", None)

        # Then: the current legacy identity and layout are returned.
        self.assertIsNotNone(infer_contract)
        descriptor = infer_contract(explicit_descriptor)
        self.assertIs(descriptor, world_model_data.LEGACY_RGB_V1)
        self.assertEqual(descriptor.contract_name, "legacy_rgb_v1")
        self.assertEqual(descriptor.contract_version, "1")
        self.assertEqual(descriptor.artifact_layout_version, "collector_v1")

    def test_explicit_capture_contract_is_preserved(self):
        # Given: a forward-compatible explicit physics descriptor.
        explicit_descriptor = world_model_data.PHYSICS_CAPTURE_V1

        # When: the capture contract is selected.
        descriptor = world_model_data.infer_capture_contract(explicit_descriptor)

        # Then: explicit provenance is never replaced by legacy inference.
        self.assertIs(descriptor, explicit_descriptor)

    def test_physics_contract_reserves_capabilities_and_relative_sidecars(self):
        # Given: the reserved future physics descriptor.
        descriptor = world_model_data.PHYSICS_CAPTURE_V1

        # When: its immutable declaration is observed.
        capabilities = descriptor.declared_capabilities
        sidecar_paths = tuple(sidecar.relative_path for sidecar in descriptor.sidecar_paths)

        # Then: capability names and relative paths are represented but unopened.
        self.assertEqual(
            capabilities,
            ("scene_nodes", "raw_contacts", "derived_support", "kinematics", "macro_events"),
        )
        self.assertEqual(sidecar_paths, ("physics_state.jsonl", "physics_events.jsonl"))

    def test_capture_contract_accepts_future_capability_names(self):
        # Given: a descriptor declaring a capability unknown to this package.
        sidecar = world_model_data.SidecarPath("future.jsonl", ("future_capability",))

        # When: the forward-compatible descriptor is constructed.
        descriptor = world_model_data.CaptureContractDescriptor(
            "future_capture_v2",
            "2",
            "collector_v2",
            None,
            None,
            ("future_capability",),
            (sidecar,),
        )

        # Then: the name and validated relative path are preserved unchanged.
        self.assertEqual(descriptor.declared_capabilities, ("future_capability",))
        self.assertEqual(descriptor.sidecar_paths, (sidecar,))

    def test_sidecar_path_rejects_parent_traversal(self):
        # Given: a sidecar path escaping the episode directory.
        relative_path = "../physics_state.jsonl"

        # When/Then: construction rejects the uncontained artifact path.
        with self.assertRaises(world_model_data.ContractValueError):
            world_model_data.SidecarPath(relative_path, ("scene_nodes",))

    def test_shot_action_requires_exactly_five_values(self):
        # Given: an underspecified action vector.
        values = (300.0, 220.0, -80.0, 20.0)

        # When/Then: construction rejects a vector shorter than five.
        with self.assertRaises(world_model_data.ContractValueError):
            world_model_data.ShotAction(values)

    def test_shot_action_rejects_mutable_value_sequences(self):
        # Given: five correctly ordered values held in a mutable list.
        values = [300.0, 220.0, -80.0, 20.0, 120.0]

        # When/Then: construction preserves deep immutability by rejecting it.
        with self.assertRaises(world_model_data.ContractValueError):
            world_model_data.ShotAction(values)

    def test_temporal_window_request_rejects_nonpositive_values(self):
        # Given: a zero prediction-step request.
        prediction_steps = 0

        # When/Then: construction rejects a nonpositive temporal request.
        with self.assertRaises(world_model_data.ContractValueError):
            world_model_data.TemporalWindowRequest(prediction_steps, 1)

    def test_temporal_window_request_rejects_noninteger_values(self):
        # Given: bool and fractional values that compare positive as numbers.
        invalid_requests = ((1.5, 1), (1, 1.5), (True, 1), (1, False))

        # When/Then: each malformed request is rejected as noninteger.
        for prediction_steps, stride_frames in invalid_requests:
            with self.subTest(
                prediction_steps=prediction_steps,
                stride_frames=stride_frames,
            ):
                with self.assertRaises(world_model_data.ContractValueError):
                    world_model_data.TemporalWindowRequest(
                        prediction_steps,
                        stride_frames,
                    )

    def test_fixture_builder_rejects_nonpositive_frame_count(self):
        # Given: an isolated destination and invalid frame count.
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "episode"

            # When/Then: the builder rejects the input before writing a tree.
            with self.assertRaisesRegex(ValueError, "frame_count.*must be positive"):
                make_complete_rollout_episode(
                    destination,
                    RolloutFixtureSpec(frame_count=0),
                )
            self.assertFalse(destination.exists())

    def test_fixture_builds_are_isolated(self):
        # Given: two independently allocated fixture roots.
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_episode = make_complete_rollout_episode(Path(first) / "episode")
            second_episode = make_complete_rollout_episode(Path(second) / "episode")

            # When: their resolved locations are compared.
            roots_differ = first_episode.resolve() != second_episode.resolve()

            # Then: stale state cannot cross fixture roots.
            self.assertTrue(roots_differ)


class RolloutArtifactValidatorTests(unittest.TestCase):
    def test_public_validator_accepts_complete_legacy_episode_without_reading_jsonl(self):
        # Given: a complete legacy episode whose JSONL payload is deliberately malformed.
        module_path = Path(__file__).resolve().parents[1] / "scripts" / "rollout_artifacts.py"
        self.assertTrue(module_path.is_file(), "public validator module is missing")
        from scripts.rollout_artifacts import (  # noqa: PLC0415
            EpisodeAccepted,
            EpisodeSummary,
            EpisodeValidationContract,
            validate_rollout_episode,
        )

        with tempfile.TemporaryDirectory() as temporary:
            episode_dir = make_complete_rollout_episode(Path(temporary) / "episode")
            (episode_dir / "action_log.jsonl").write_text("not-json\n", encoding="utf-8")

            # When: the public validator checks the collector contract.
            result = validate_rollout_episode(
                episode_dir,
                EpisodeValidationContract(count=1, fps=30.0, duration_seconds=5.0),
            )

            # Then: legacy provenance is inferred and immutable parsed facts are returned.
            self.assertIsInstance(result, EpisodeAccepted)
            self.assertIs(result.episode.capture_contract, world_model_data.LEGACY_RGB_V1)
            self.assertEqual(result.episode.shots[0].frames[0].index, 0)
            with self.assertRaises((FrozenInstanceError, AttributeError)):
                result.episode.shots = ()

    def test_public_validator_rejects_unknown_and_unsupported_contracts(self):
        # Given: complete episodes declaring either an unknown or reserved physics contract.
        module_path = Path(__file__).resolve().parents[1] / "scripts" / "rollout_artifacts.py"
        self.assertTrue(module_path.is_file(), "public validator module is missing")
        from scripts.rollout_artifacts import (  # noqa: PLC0415
            EpisodeRejected,
            EpisodeRejectionCode,
            EpisodeSummary,
            EpisodeValidationContract,
            validate_rollout_episode,
        )

        descriptors = (
            ({"contract_name": "future_capture_v9", "contract_version": "9", "artifact_layout_version": "collector_v9"}, EpisodeRejectionCode.UNKNOWN_CAPTURE_CONTRACT),
            (
                {
                    "contract_name": world_model_data.PHYSICS_CAPTURE_V1.contract_name,
                    "contract_version": world_model_data.PHYSICS_CAPTURE_V1.contract_version,
                    "artifact_layout_version": world_model_data.PHYSICS_CAPTURE_V1.artifact_layout_version,
                    "player_provenance": None,
                    "protocol_provenance": None,
                    "declared_capabilities": list(world_model_data.PHYSICS_CAPTURE_V1.declared_capabilities),
                    "sidecar_paths": [
                        {"relative_path": sidecar.relative_path, "capabilities": list(sidecar.capabilities)}
                        for sidecar in world_model_data.PHYSICS_CAPTURE_V1.sidecar_paths
                    ],
                },
                EpisodeRejectionCode.UNSUPPORTED_CAPTURE_CONTRACT,
            ),
        )
        contract = EpisodeValidationContract(count=1, fps=30.0, duration_seconds=5.0)

        # When/Then: neither explicit contract silently falls back to legacy.
        for descriptor, expected_code in descriptors:
            with self.subTest(expected_code=expected_code):
                with tempfile.TemporaryDirectory() as temporary:
                    episode_dir = make_complete_rollout_episode(Path(temporary) / "episode")
                    manifest_path = episode_dir / "manifest.json"
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["capture_contract"] = descriptor
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

                    result = validate_rollout_episode(episode_dir, contract)

                    self.assertIsInstance(result, EpisodeRejected)
                    self.assertEqual(result.code, expected_code)

    def test_public_validator_returns_stable_structural_rejection_codes(self):
        # Given: isolated malformed episode shapes covering each structural boundary.
        from scripts.rollout_artifacts import (  # noqa: PLC0415
            EpisodeRejected,
            EpisodeRejectionCode,
            EpisodeValidationContract,
            validate_rollout_episode,
        )

        contract = EpisodeValidationContract(count=1, fps=30.0, duration_seconds=5.0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            malformed = make_complete_rollout_episode(root / "malformed")
            (malformed / "manifest.json").write_text("{", encoding="utf-8")
            noncontiguous = make_complete_rollout_episode(root / "noncontiguous")
            (noncontiguous / "shot_001" / "frames" / "frame_000000.png").unlink()
            escaping = make_complete_rollout_episode(root / "escaping")
            escaping_log_path = escaping / "action_log.json"
            escaping_log = json.loads(escaping_log_path.read_text(encoding="utf-8"))
            escaping_log["accepted_trials"][0]["shot_name"] = "../outside"
            escaping_log_path.write_text(json.dumps(escaping_log), encoding="utf-8")
            symlinked = make_complete_rollout_episode(root / "symlinked")
            external_frame = root / "external.png"
            external_frame.write_bytes(b"external")
            frame_path = symlinked / "shot_001" / "frames" / "frame_000000.png"
            frame_path.unlink()
            frame_path.symlink_to(external_frame)
            cases = (
                (root / "missing", EpisodeRejectionCode.MISSING_ARTIFACT),
                (malformed, EpisodeRejectionCode.MALFORMED_JSON),
                (noncontiguous, EpisodeRejectionCode.NONCONTIGUOUS_FRAMES),
                (escaping, EpisodeRejectionCode.ESCAPING_ARTIFACT),
                (symlinked, EpisodeRejectionCode.SYMLINK_ARTIFACT),
            )

            # When/Then: every shape produces its stable typed code.
            for episode_dir, expected_code in cases:
                with self.subTest(expected_code=expected_code):
                    result = validate_rollout_episode(episode_dir, contract)
                    self.assertIsInstance(result, EpisodeRejected)
                    self.assertEqual(result.code, expected_code)

    def test_public_validator_does_not_resolve_every_constructed_frame_path(self):
        # Given: one real accepted shot with enough frames to expose per-frame path work.
        import unittest.mock  # noqa: PLC0415

        from scripts.rollout_artifacts import (  # noqa: PLC0415
            EpisodeAccepted,
            EpisodeSummary,
            EpisodeValidationContract,
            validate_rollout_episode,
        )

        with tempfile.TemporaryDirectory() as temporary:
            episode_dir = make_complete_rollout_episode(
                Path(temporary) / "episode",
                RolloutFixtureSpec(frame_count=100, shot_count=1),
            )
            real_resolve = Path.resolve
            resolved_paths = []

            def tracking_resolve(path, *args, **kwargs):
                resolved_paths.append(path)
                return real_resolve(path, *args, **kwargs)

            # When: the canonical validator checks the complete episode.
            with unittest.mock.patch.object(Path, "resolve", tracking_resolve):
                result = validate_rollout_episode(
                    episode_dir,
                    EpisodeValidationContract(
                        count=1,
                        fps=30.0,
                        duration_seconds=5.0,
                    ),
                )

            # Then: trusted constructed frame children do not repeat full resolution.
            self.assertIsInstance(result, EpisodeAccepted)
            self.assertLess(len(resolved_paths), 30)


# ---------------------------------------------------------------------------
# Helpers shared by EpisodeCatalogTests
# ---------------------------------------------------------------------------


def _fix_episode_for_count1(episode_dir: Path) -> None:
    """Patch a make_complete_rollout_episode fixture to satisfy count=1 contract."""
    manifest = json.loads((episode_dir / "manifest.json").read_text("utf-8"))
    manifest["accepted_rollout_count"] = 1
    manifest["rollout_count"] = 1
    manifest["attempt_count"] = 1
    manifest["attempts"] = manifest["attempts"][:1]
    (episode_dir / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    al = json.loads((episode_dir / "action_log.json").read_text("utf-8"))
    al["accepted_trial_count"] = 1
    al["trial_count"] = 1
    al["accepted_trials"] = al["accepted_trials"][:1]
    al["trials"] = al["trials"][:1]
    (episode_dir / "action_log.json").write_text(json.dumps(al), "utf-8")
    (episode_dir / "action_log.jsonl").write_text(
        json.dumps(al["accepted_trials"][0]) + "\n", "utf-8"
    )


# ---------------------------------------------------------------------------
# EpisodeCatalog tests
# ---------------------------------------------------------------------------


class EpisodeCatalogTests(unittest.TestCase):
    """Tests for world_model/data/catalog.py EpisodeCatalog."""

    # ------------------------------------------------------------------
    # Characterization (baseline) test – must pass before catalog exists
    # ------------------------------------------------------------------

    def test_existing_validator_accepts_complete_episode_and_infers_legacy_contract(self):
        """Baseline: the public validator already returns the expected typed facts."""
        from scripts.rollout_artifacts import (  # noqa: PLC0415
            EpisodeAccepted,
            EpisodeSummary,
            EpisodeValidationContract,
            EpisodeValidationMode,
            ValidatedEpisode,
            ValidatedShot,
            validate_rollout_episode,
        )

        with tempfile.TemporaryDirectory() as temporary:
            episode_dir = make_complete_rollout_episode(
                Path(temporary) / "episode",
                RolloutFixtureSpec(frame_count=3, shot_count=1),
            )

            result = validate_rollout_episode(
                episode_dir,
                EpisodeValidationContract(count=1, fps=30.0, duration_seconds=5.0),
            )

            # Then: the existing validator produces the expected typed facts.
            self.assertIsInstance(result, EpisodeAccepted)
            episode: ValidatedEpisode = result.episode
            self.assertIs(episode.capture_contract, world_model_data.LEGACY_RGB_V1)
            self.assertIsInstance(episode.shots, tuple)
            self.assertEqual(len(episode.shots), 1)
            shot: ValidatedShot = episode.shots[0]
            self.assertEqual(shot.name, "shot_001")
            self.assertIsInstance(shot.frames, tuple)
            self.assertEqual(len(shot.frames), 3)
            self.assertEqual(shot.frames[0].index, 0)
            self.assertEqual(shot.frames[1].index, 1)
            self.assertEqual(shot.frames[2].index, 2)
            # Action fields are present and have five values.
            self.assertIsNotNone(shot.action)
            self.assertEqual(len(shot.action.values), 5)

            summary_result = validate_rollout_episode(
                episode_dir,
                EpisodeValidationContract(count=1, fps=30.0, duration_seconds=5.0),
                mode=EpisodeValidationMode.CANONICAL_SUMMARY,
            )
            self.assertIsInstance(summary_result, EpisodeSummary)
            self.assertTrue(summary_result.canonical_acceptance_available)
            summary_shot: ValidatedShot = summary_result.episode.shots[0]
            self.assertEqual(summary_shot.frame_count, 3)
            self.assertEqual(summary_shot.frames, ())

    def test_summary_validator_rejects_unreadable_frames_canonically(self):
        import os  # noqa: PLC0415
        import unittest.mock  # noqa: PLC0415

        from scripts.rollout_artifacts import (  # noqa: PLC0415
            EpisodeRejected,
            EpisodeRejectionCode,
            EpisodeSummary,
            EpisodeValidationContract,
            EpisodeValidationMode,
            validate_rollout_episode,
        )

        with tempfile.TemporaryDirectory() as temporary:
            episode_dir = make_complete_rollout_episode(
                Path(temporary) / "episode",
                RolloutFixtureSpec(frame_count=3, shot_count=1),
            )
            frame_path = episode_dir / "shot_001" / "frames" / "frame_000001.png"
            real_access = os.access

            def access_with_unreadable_frame(path: Path, mode: int) -> bool:
                return False if Path(path) == frame_path else real_access(path, mode)

            with unittest.mock.patch(
                "scripts.rollout_artifacts.os.access",
                side_effect=access_with_unreadable_frame,
            ):
                contract = EpisodeValidationContract(
                    count=1,
                    fps=30.0,
                    duration_seconds=5.0,
                )
                materialized = validate_rollout_episode(episode_dir, contract)
                summary = validate_rollout_episode(
                    episode_dir,
                    contract,
                    mode=EpisodeValidationMode.CANONICAL_SUMMARY,
                )

            self.assertIsInstance(materialized, EpisodeRejected)
            self.assertIsInstance(summary, EpisodeRejected)
            self.assertEqual(
                materialized.code,
                EpisodeRejectionCode.UNREADABLE_ARTIFACT,
            )
            self.assertEqual(summary.code, EpisodeRejectionCode.UNREADABLE_ARTIFACT)

    # ------------------------------------------------------------------
    # Catalog construction and snapshot immutability
    # ------------------------------------------------------------------

    def test_catalog_snapshot_and_plan_backed_source_provenance(self):
        """A catalog snapshot is immutable; a later episode is absent; plan attaches source key."""
        from world_model.data.catalog import (  # noqa: PLC0415
            EpisodeCatalog,
            RequiredCapabilityError,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "output"
            split_dir = output_root / "train"
            split_dir.mkdir(parents=True)

            options = CollectionOptions(count=1, fps=30.0, duration=5.0, workers=1)
            entry = LevelEntry("novelty_level_0", "type010101", "levels/one.xml")
            episode_dir = split_dir / _safe_output_name(entry)
            make_complete_rollout_episode(
                episode_dir,
                RolloutFixtureSpec(frame_count=3, shot_count=1),
            )
            _fix_episode_for_count1(episode_dir)

            plan_episode = PlannedEpisode("train", entry, episode_dir, "existing")
            plan_path = write_collection_plan(
                root / "plan",
                output_root=output_root,
                episodes=[plan_episode],
                summary={f"train:{entry.bucket}": {"target": 1, "existing": 1, "scheduled": 0}},
                options=options,
                targets=CollectionTargets(train=1, dev=1, test=0),
                selected_splits=("train",),
                seed="catalog-test",
            )

            catalog = EpisodeCatalog.build(
                root=output_root,
                split="train",
                capture_contract=world_model_data.LEGACY_RGB_V1,
                collection_plan=plan_path,
            )

            episodes = catalog.episodes
            self.assertEqual(len(episodes), 1)
            ep = episodes[0]
            # Source key must come from plan, not from directory name.
            self.assertIsNotNone(ep.source_level_key)
            self.assertEqual(ep.source_level_key, entry.relative_path)

            # Now add a late episode – original snapshot must be unchanged.
            late_entry = LevelEntry("novelty_level_0", "type010101", "levels/two.xml")
            late_dir = split_dir / _safe_output_name(late_entry)
            make_complete_rollout_episode(
                late_dir,
                RolloutFixtureSpec(frame_count=3, shot_count=1),
            )
            _fix_episode_for_count1(late_dir)

            # Original snapshot must remain identical.
            self.assertEqual(catalog.episodes, episodes)
            self.assertEqual(len(catalog.episodes), 1)

            # Refresh picks up the late episode.
            refreshed = catalog.refresh()
            self.assertIsNot(refreshed, catalog)
            self.assertGreater(len(refreshed.episodes), len(catalog.episodes))
            names = {ep2.name for ep2 in refreshed.episodes}
            self.assertIn(_safe_output_name(entry), names)

    def test_catalog_snapshot_is_immutable(self):
        """episodes property cannot be replaced on an existing catalog instance."""
        from world_model.data.catalog import EpisodeCatalog  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "output"
            (root / "train").mkdir(parents=True)

            catalog = EpisodeCatalog.build(
                root=root,
                split="train",
                capture_contract=world_model_data.LEGACY_RGB_V1,
            )

            with self.assertRaises((AttributeError, FrozenInstanceError, TypeError)):
                catalog.episodes = ()  # type: ignore[misc]

    def test_catalog_without_plan_reports_provenance_unavailable(self):
        """Without a collection plan, source_level_key is None for all episodes."""
        from world_model.data.catalog import EpisodeCatalog  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "output"
            split_dir = root / "train"
            split_dir.mkdir(parents=True)
            entry = LevelEntry("novelty_level_0", "type010101", "levels/one.xml")
            ep_dir = split_dir / _safe_output_name(entry)
            make_complete_rollout_episode(
                ep_dir, RolloutFixtureSpec(frame_count=3, shot_count=1)
            )
            _fix_episode_for_count1(ep_dir)

            catalog = EpisodeCatalog.build(
                root=root,
                split="train",
                capture_contract=world_model_data.LEGACY_RGB_V1,
            )

            self.assertEqual(len(catalog.episodes), 1)
            # No plan => provenance unavailable => source_level_key is None.
            self.assertIsNone(catalog.episodes[0].source_level_key)
            self.assertFalse(catalog.provenance_available)

    def test_catalog_only_enumerates_direct_nonsymlink_episode_directories(self):
        """Symlinks and plain files at the split level are not enumerated."""
        from world_model.data.catalog import EpisodeCatalog  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "output"
            split_dir = root / "train"
            split_dir.mkdir(parents=True)

            # A plain file at split level – must be skipped silently.
            (split_dir / "not_an_episode.txt").write_text("noise", "utf-8")

            # A symlink directory – must be skipped.
            real_dir = Path(temporary) / "real_episode"
            make_complete_rollout_episode(
                real_dir, RolloutFixtureSpec(frame_count=3, shot_count=1)
            )
            _fix_episode_for_count1(real_dir)
            symlink_ep = split_dir / "symlinked_episode"
            symlink_ep.symlink_to(real_dir, target_is_directory=True)

            # A genuine valid episode.
            good_dir = split_dir / "good_episode"
            make_complete_rollout_episode(
                good_dir, RolloutFixtureSpec(frame_count=3, shot_count=1)
            )
            _fix_episode_for_count1(good_dir)

            catalog = EpisodeCatalog.build(
                root=root,
                split="train",
                capture_contract=world_model_data.LEGACY_RGB_V1,
            )

            episode_names = {ep.name for ep in catalog.episodes}
            self.assertIn("good_episode", episode_names)
            self.assertNotIn("symlinked_episode", episode_names)
            self.assertNotIn("not_an_episode.txt", episode_names)

    def test_unknown_or_unvalidated_capture_contract_is_rejected(self):
        """An unsupported physics_capture_v1 manifest raises rejection counts, zero accepted."""
        from world_model.data.catalog import EpisodeCatalog  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "output"
            split_dir = root / "train"
            split_dir.mkdir(parents=True)

            physics_dir = split_dir / "physics_episode"
            make_complete_rollout_episode(
                physics_dir, RolloutFixtureSpec(frame_count=3, shot_count=1)
            )
            manifest = json.loads((physics_dir / "manifest.json").read_text("utf-8"))
            manifest["accepted_rollout_count"] = 1
            manifest["rollout_count"] = 1
            manifest["attempt_count"] = 1
            manifest["attempts"] = manifest["attempts"][:1]
            manifest["capture_contract"] = {
                "contract_name": world_model_data.PHYSICS_CAPTURE_V1.contract_name,
                "contract_version": world_model_data.PHYSICS_CAPTURE_V1.contract_version,
                "artifact_layout_version": world_model_data.PHYSICS_CAPTURE_V1.artifact_layout_version,
                "player_provenance": None,
                "protocol_provenance": None,
                "declared_capabilities": list(world_model_data.PHYSICS_CAPTURE_V1.declared_capabilities),
                "sidecar_paths": [
                    {"relative_path": s.relative_path, "capabilities": list(s.capabilities)}
                    for s in world_model_data.PHYSICS_CAPTURE_V1.sidecar_paths
                ],
            }
            (physics_dir / "manifest.json").write_text(json.dumps(manifest), "utf-8")

            catalog = EpisodeCatalog.build(
                root=root,
                split="train",
                capture_contract=world_model_data.LEGACY_RGB_V1,
            )

            self.assertEqual(len(catalog.episodes), 0)
            self.assertGreater(catalog.rejection_count, 0)
            rejection_codes = catalog.rejection_codes
            self.assertIn("unsupported_capture_contract", rejection_codes)

    def test_catalog_required_capability_not_declared_is_rejected(self):
        """Requesting a capability absent from the contract descriptor fails closed at build time."""
        from world_model.data.catalog import (  # noqa: PLC0415
            EpisodeCatalog,
            RequiredCapabilityError,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "output"
            (root / "train").mkdir(parents=True)

            with self.assertRaises(RequiredCapabilityError):
                EpisodeCatalog.build(
                    root=root,
                    split="train",
                    capture_contract=world_model_data.LEGACY_RGB_V1,
                    required_capabilities=("scene_nodes",),
                )

    def test_cross_catalog_duplicate_source_key_is_detected(self):
        """Two plan-backed catalogs with an overlapping source key fail disjointness."""
        from world_model.data.catalog import (  # noqa: PLC0415
            DuplicateSourceKeyError,
            EpisodeCatalog,
            check_source_key_disjointness,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "output"
            train_dir = output_root / "train"
            dev_dir = output_root / "dev"
            train_dir.mkdir(parents=True)
            dev_dir.mkdir(parents=True)

            options = CollectionOptions(count=1, fps=30.0, duration=5.0, workers=1)
            entry = LevelEntry("novelty_level_0", "type010101", "levels/shared.xml")

            for split_dir in (train_dir, dev_dir):
                ep_dir = split_dir / _safe_output_name(entry)
                make_complete_rollout_episode(
                    ep_dir, RolloutFixtureSpec(frame_count=3, shot_count=1)
                )
                _fix_episode_for_count1(ep_dir)

            train_episode = PlannedEpisode("train", entry, train_dir / _safe_output_name(entry), "existing")
            dev_episode = PlannedEpisode("dev", entry, dev_dir / _safe_output_name(entry), "existing")
            plan_path = write_collection_plan(
                root / "plan",
                output_root=output_root,
                episodes=[train_episode, dev_episode],
                summary={
                    f"train:{entry.bucket}": {"target": 1, "existing": 1, "scheduled": 0},
                    f"dev:{entry.bucket}": {"target": 1, "existing": 1, "scheduled": 0},
                },
                options=options,
                targets=CollectionTargets(train=1, dev=1, test=0),
                selected_splits=("train", "dev"),
                seed="dup-test",
            )

            catalog_train = EpisodeCatalog.build(
                root=output_root,
                split="train",
                capture_contract=world_model_data.LEGACY_RGB_V1,
                collection_plan=plan_path,
            )
            catalog_dev = EpisodeCatalog.build(
                root=output_root,
                split="dev",
                capture_contract=world_model_data.LEGACY_RGB_V1,
                collection_plan=plan_path,
            )

            with self.assertRaises(DuplicateSourceKeyError):
                check_source_key_disjointness([catalog_train, catalog_dev])

    def test_episode_paths_are_contained_within_split_root(self):
        """All accepted episode relative_paths resolve under root/split."""
        from world_model.data.catalog import EpisodeCatalog  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "output"
            split_dir = root / "train"
            split_dir.mkdir(parents=True)
            entry = LevelEntry("novelty_level_0", "type010101", "levels/one.xml")
            ep_dir = split_dir / _safe_output_name(entry)
            make_complete_rollout_episode(
                ep_dir, RolloutFixtureSpec(frame_count=3, shot_count=1)
            )
            _fix_episode_for_count1(ep_dir)

            catalog = EpisodeCatalog.build(
                root=root,
                split="train",
                capture_contract=world_model_data.LEGACY_RGB_V1,
            )

            split_root = root / "train"
            for ep in catalog.episodes:
                ep_path = root / ep.relative_path
                self.assertTrue(
                    str(ep_path.resolve()).startswith(str(split_root.resolve())),
                    f"{ep.relative_path} escapes split root",
                )

    def test_refresh_returns_new_instance_original_unchanged(self):
        """refresh() returns a distinct catalog; original is not mutated."""
        from world_model.data.catalog import EpisodeCatalog  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "output"
            (root / "train").mkdir(parents=True)

            catalog = EpisodeCatalog.build(
                root=root,
                split="train",
                capture_contract=world_model_data.LEGACY_RGB_V1,
            )
            original_episodes = catalog.episodes

            refreshed = catalog.refresh()

        self.assertIsNot(refreshed, catalog)
        self.assertEqual(catalog.episodes, original_episodes)

    def test_catalog_validates_independent_episodes_concurrently(self):
        # Given: four real complete episodes and a barrier around the real validator.
        import threading  # noqa: PLC0415
        import unittest.mock  # noqa: PLC0415

        from scripts import rollout_artifacts  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(4):
                episode_dir = make_complete_rollout_episode(
                    root / "train" / f"episode_{index}",
                    RolloutFixtureSpec(frame_count=3, shot_count=1),
                )
                _fix_episode_for_count1(episode_dir)
            real_validate = rollout_artifacts.validate_rollout_episode
            rendezvous = threading.Barrier(2, timeout=1)
            worker_ids = set()

            def tracking_validate(episode_dir, contract):
                worker_ids.add(threading.get_ident())
                try:
                    rendezvous.wait()
                except threading.BrokenBarrierError:
                    pass
                return real_validate(episode_dir, contract)

            # When: the public catalog validates the independent episode set.
            with unittest.mock.patch.object(
                rollout_artifacts,
                "validate_rollout_episode",
                side_effect=tracking_validate,
            ):
                catalog = world_model_data.EpisodeCatalog.build(
                    root,
                    "train",
                    world_model_data.LEGACY_RGB_V1,
                )

            # Then: validation overlaps while catalog ordering remains deterministic.
            self.assertGreater(len(worker_ids), 1)
            self.assertEqual(
                tuple(episode.name for episode in catalog.episodes),
                ("episode_0", "episode_1", "episode_2", "episode_3"),
            )


# ---------------------------------------------------------------------------
# TemporalWindowDataset tests
# ---------------------------------------------------------------------------


def _build_catalog_from_fixture(
    root: Path,
    split: str = "train",
    frame_count: int = 5,
    shot_count: int = 1,
) -> "world_model_data.EpisodeCatalog":
    """Build a real EpisodeCatalog from a synthetic fixture for dataset tests."""
    from world_model.data.catalog import EpisodeCatalog  # noqa: PLC0415

    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    ep_dir = split_dir / "episode_001"
    make_complete_rollout_episode(
        ep_dir,
        RolloutFixtureSpec(frame_count=frame_count, shot_count=shot_count),
    )
    # Patch manifest/action_log to match shot_count contract.
    if shot_count == 1:
        _fix_episode_for_count1(ep_dir)

    return EpisodeCatalog.build(
        root=root,
        split=split,
        capture_contract=world_model_data.LEGACY_RGB_V1,
    )


class TemporalWindowDatasetTests(unittest.TestCase):
    """Tests for world_model/data/dataset.py TemporalWindowDataset."""

    # ------------------------------------------------------------------
    # Characterization test – must pass before dataset.py exists.
    # It only exercises catalog + TemporalWindowRequest arithmetic,
    # which are already implemented.
    # ------------------------------------------------------------------

    def test_catalog_fixture_has_eligible_window_for_stride_two(self):
        """Characterization: a 5-frame shot satisfies prediction_steps=2, stride=2 (horizon=4)."""
        # Given: a catalog with one episode, one shot, 5 frames.
        with tempfile.TemporaryDirectory() as temporary:
            catalog = _build_catalog_from_fixture(
                Path(temporary), frame_count=5, shot_count=1
            )

            # When: the temporal request arithmetic is computed.
            request = world_model_data.TemporalWindowRequest(
                prediction_steps=2, stride_frames=2
            )

            # Then: the catalog contains exactly one shot with enough frames.
            self.assertEqual(len(catalog.episodes), 1)
            episode = catalog.episodes[0]
            self.assertEqual(len(episode.shots), 1)
            shot = episode.shots[0]
            self.assertEqual(len(shot.frames), 5)

            # Horizon = 2 * 2 = 4; a start at frame 0 means we need frames 0,2,4 (context
            # is frame 0, targets are at offsets stride=2 and 2*stride=4).
            # The window needs start_frame + horizon_frames < frame_count, i.e. 0 + 4 <= 4.
            horizon = request.horizon_frames
            self.assertEqual(horizon, 4)
            eligible_starts = [
                f for f in range(len(shot.frames)) if f + horizon < len(shot.frames)
            ]
            # Only frame 0 satisfies: 0+4=4 < 5 → True.
            self.assertEqual(eligible_starts, [0])

    # ------------------------------------------------------------------
    # Tests that will fail until dataset.py is implemented.
    # ------------------------------------------------------------------

    def test_two_prediction_steps_at_stride_two_reads_frames_zero_two_four(self):
        """Happy path: sample[0] uses frames 0,2,4 with correct tensor/action/provenance."""
        from world_model.data.dataset import TemporalWindowDataset  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = _build_catalog_from_fixture(root, frame_count=5, shot_count=1)
            request = world_model_data.TemporalWindowRequest(
                prediction_steps=2, stride_frames=2
            )

            # When: the dataset is constructed and the first sample is retrieved.
            dataset = TemporalWindowDataset(catalog, request)

            # Then: the dataset has exactly one eligible (episode, shot, start_frame) triple.
            self.assertEqual(len(dataset), 1)

            sample = dataset[0]

            # context_image is a [C, H, W] float tensor in [0, 1].
            context_image = sample["context_image"]
            import torch  # noqa: PLC0415
            self.assertIsInstance(context_image, torch.Tensor)
            self.assertEqual(context_image.dtype, torch.float32)
            self.assertEqual(context_image.ndim, 3)  # CHW
            self.assertGreaterEqual(context_image.min().item(), 0.0)
            self.assertLessEqual(context_image.max().item(), 1.0)

            # target_images is a list of two [C, H, W] tensors.
            target_images = sample["target_images"]
            self.assertEqual(len(target_images), 2)
            for t in target_images:
                self.assertIsInstance(t, torch.Tensor)
                self.assertEqual(t.dtype, torch.float32)
                self.assertEqual(t.ndim, 3)

            # frame_indices: context=0, targets=[2, 4].
            frame_indices = sample["frame_indices"]
            self.assertEqual(frame_indices[0], 0)
            self.assertEqual(list(frame_indices[1:]), [2, 4])

            # action is a strict [5] float tensor.
            action = sample["action"]
            self.assertIsInstance(action, torch.Tensor)
            self.assertEqual(action.shape, torch.Size([5]))

            # Derived horizon/prediction/stride fields.
            self.assertEqual(sample["horizon_frames"], 4)
            self.assertEqual(sample["prediction_steps"], 2)
            self.assertEqual(sample["stride_frames"], 2)

            # Provenance.
            provenance = sample["provenance"]
            self.assertEqual(provenance["split"], "train")
            self.assertIsNone(provenance["source_level_key"])
            self.assertEqual(provenance["episode"], "episode_001")
            self.assertEqual(provenance["shot"], "shot_001")
            self.assertIsInstance(
                provenance["capture_contract"], world_model_data.CaptureContractDescriptor
            )
            self.assertIsInstance(provenance["declared_capabilities"], tuple)
            self.assertIsInstance(provenance["sidecar_paths"], tuple)

    def test_post_snapshot_frame_removal_raises_typed_read_error(self):
        """If a cataloged frame disappears after dataset construction, raise a typed read error."""
        from world_model.data.dataset import TemporalWindowDataset, FrameReadError  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = _build_catalog_from_fixture(root, frame_count=5, shot_count=1)
            request = world_model_data.TemporalWindowRequest(
                prediction_steps=2, stride_frames=2
            )
            dataset = TemporalWindowDataset(catalog, request)

            # Remove the context frame (frame_000000.png) after dataset construction.
            # catalog.py rebases frame.relative_path to be catalog-root-relative,
            # so the absolute path is simply: root / frame.relative_path
            episode = catalog.episodes[0]
            shot = episode.shots[0]
            frame_path = root / shot.frames[0].relative_path
            frame_path.unlink()

            # When: __getitem__ tries to decode the missing frame.
            # Then: a typed FrameReadError is raised (not OSError or generic IOError).
            with self.assertRaises(FrameReadError):
                dataset[0]

    def test_does_not_open_declared_supervision_sidecars(self):
        """Dataset.__getitem__ must never open or stat sidecar files."""
        from world_model.data.dataset import TemporalWindowDataset  # noqa: PLC0415
        import unittest.mock  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = _build_catalog_from_fixture(root, frame_count=5, shot_count=1)
            request = world_model_data.TemporalWindowRequest(
                prediction_steps=2, stride_frames=2
            )
            dataset = TemporalWindowDataset(catalog, request)

            # Place fake sidecar files so any accidental open would not raise FileNotFoundError.
            episode = catalog.episodes[0]
            ep_dir = root / episode.relative_path
            (ep_dir / "physics_state.jsonl").write_text('{"fake": true}\n', encoding="utf-8")
            (ep_dir / "physics_events.jsonl").write_text('{"fake": true}\n', encoding="utf-8")

            # Patch builtins.open to detect any accidental sidecar access.
            real_open = open
            opened_paths: list[str] = []

            def tracking_open(path, *args, **kwargs):
                opened_paths.append(str(path))
                return real_open(path, *args, **kwargs)

            with unittest.mock.patch("builtins.open", side_effect=tracking_open):
                _ = dataset[0]

            sidecar_names = {"physics_state.jsonl", "physics_events.jsonl"}
            opened_basenames = {Path(p).name for p in opened_paths}
            for sidecar in sidecar_names:
                self.assertNotIn(
                    sidecar,
                    opened_basenames,
                    f"Dataset opened sidecar {sidecar!r} during __getitem__",
                )

    def test_no_eligible_window_raises_value_error(self):
        """A temporal request with horizon exceeding all shot lengths raises ValueError at construction."""
        from world_model.data.dataset import TemporalWindowDataset  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Only 2 frames, but request needs horizon=10 frames.
            catalog = _build_catalog_from_fixture(root, frame_count=2, shot_count=1)
            request = world_model_data.TemporalWindowRequest(
                prediction_steps=5, stride_frames=2
            )

            with self.assertRaises(ValueError):
                TemporalWindowDataset(catalog, request)

    def test_explicit_transform_is_applied_to_all_images(self):
        """A custom transform callable is applied to every decoded image tensor."""
        from world_model.data.dataset import TemporalWindowDataset  # noqa: PLC0415
        import torch  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = _build_catalog_from_fixture(root, frame_count=5, shot_count=1)
            request = world_model_data.TemporalWindowRequest(
                prediction_steps=2, stride_frames=2
            )

            # Transform that multiplies all pixel values by 0 → all-zero tensors.
            zeroing_transform = lambda tensor: torch.zeros_like(tensor)  # noqa: E731

            dataset = TemporalWindowDataset(catalog, request, transform=zeroing_transform)
            sample = dataset[0]

            self.assertTrue(
                torch.all(sample["context_image"] == 0.0),
                "context_image was not transformed",
            )
            for i, t in enumerate(sample["target_images"]):
                self.assertTrue(
                    torch.all(t == 0.0),
                    f"target_images[{i}] was not transformed",
                )

    def test_multiple_shots_generate_multiple_windows(self):
        """A 2-shot episode with 5 frames each yields 2 eligible windows (one per shot)."""
        from world_model.data.dataset import TemporalWindowDataset  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_dir = root / "train"
            split_dir.mkdir(parents=True)
            from world_model.data.catalog import EpisodeCatalog  # noqa: PLC0415

            ep_dir = split_dir / "episode_002"
            make_complete_rollout_episode(
                ep_dir,
                RolloutFixtureSpec(frame_count=5, shot_count=2),
            )
            # Patch to count=2.
            manifest = json.loads((ep_dir / "manifest.json").read_text("utf-8"))
            manifest["accepted_rollout_count"] = 2
            manifest["rollout_count"] = 2
            manifest["attempt_count"] = 2
            (ep_dir / "manifest.json").write_text(json.dumps(manifest), "utf-8")

            catalog = EpisodeCatalog.build(
                root=root,
                split="train",
                capture_contract=world_model_data.LEGACY_RGB_V1,
            )

            request = world_model_data.TemporalWindowRequest(
                prediction_steps=2, stride_frames=2
            )
            dataset = TemporalWindowDataset(catalog, request)

            # Each of 2 shots has exactly 1 eligible start_frame (frame 0).
            self.assertEqual(len(dataset), 2)


# ---------------------------------------------------------------------------
# DeterministicSamplingTests
# ---------------------------------------------------------------------------


def _build_multi_episode_catalog(root: Path, num_episodes: int = 3) -> "world_model_data.EpisodeCatalog":
    """Build a catalog with multiple episodes for sampling tests."""
    from world_model.data.catalog import EpisodeCatalog  # noqa: PLC0415

    split_dir = root / "train"
    split_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_episodes):
        ep_dir = split_dir / f"episode_{i:03d}"
        make_complete_rollout_episode(
            ep_dir,
            RolloutFixtureSpec(frame_count=5, shot_count=1),
        )
        _fix_episode_for_count1(ep_dir)

    return EpisodeCatalog.build(
        root=root,
        split="train",
        capture_contract=world_model_data.LEGACY_RGB_V1,
    )


class DeterministicSamplingTests(unittest.TestCase):
    """Tests for world_model/data/sampling.py EpochSampler and TemporalWindowCollator."""

    # ------------------------------------------------------------------
    # Characterization test – must pass before sampling.py exists.
    # It exercises only the dataset index (already implemented) to confirm
    # that the catalog produces a known number of eligible windows.
    # ------------------------------------------------------------------

    def test_catalog_with_three_episodes_has_three_eligible_windows(self):
        """Characterization: 3 episodes × 1 shot × 1 eligible start yields 3 windows."""
        from world_model.data.dataset import TemporalWindowDataset  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = _build_multi_episode_catalog(root, num_episodes=3)
            request = world_model_data.TemporalWindowRequest(
                prediction_steps=2, stride_frames=2
            )
            dataset = TemporalWindowDataset(catalog, request)

            # Each episode has 1 shot with 5 frames; horizon=4, so start=0 only.
            self.assertEqual(len(dataset), 3)
            # Confirm provenance keys are episode names.
            episodes_names = {catalog.episodes[i].name for i in range(3)}
            self.assertEqual(len(episodes_names), 3)

    # ------------------------------------------------------------------
    # Tests that will fail until sampling.py is implemented.
    # ------------------------------------------------------------------

    def test_seed_and_epoch_reproduce_provenance_order(self):
        """Same seed+epoch produces identical provenance index order across independent instances."""
        from world_model.data.sampling import EpochSampler  # noqa: PLC0415
        from world_model.data.dataset import TemporalWindowDataset  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = _build_multi_episode_catalog(root, num_episodes=3)
            request = world_model_data.TemporalWindowRequest(
                prediction_steps=2, stride_frames=2
            )
            dataset = TemporalWindowDataset(catalog, request)

            # First independent instance.
            sampler_a = EpochSampler(dataset, seed=42, draw_count=3)
            sampler_a.set_epoch(0)
            order_a = list(sampler_a)
            provenance_a = [dataset[index]["provenance"]["episode"] for index in order_a]

            # Second independent instance – same seed/epoch/draw_count.
            sampler_b = EpochSampler(dataset, seed=42, draw_count=3)
            sampler_b.set_epoch(0)
            order_b = list(sampler_b)
            provenance_b = [dataset[index]["provenance"]["episode"] for index in order_b]

            # Must be identical.
            self.assertEqual(order_a, order_b)
            self.assertEqual(provenance_a, provenance_b)
            self.assertEqual(len(order_a), 3)

    def test_changed_seed_changes_provenance_order(self):
        """A different seed produces a different draw order (with overwhelming probability)."""
        from world_model.data.sampling import EpochSampler  # noqa: PLC0415
        from world_model.data.dataset import TemporalWindowDataset  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Use enough episodes so a reorder is almost guaranteed.
            catalog = _build_multi_episode_catalog(root, num_episodes=10)
            request = world_model_data.TemporalWindowRequest(
                prediction_steps=2, stride_frames=2
            )
            dataset = TemporalWindowDataset(catalog, request)

            sampler_seed1 = EpochSampler(dataset, seed=1, draw_count=10)
            sampler_seed1.set_epoch(0)
            order_seed1 = list(sampler_seed1)

            sampler_seed2 = EpochSampler(dataset, seed=2, draw_count=10)
            sampler_seed2.set_epoch(0)
            order_seed2 = list(sampler_seed2)

            # Different seeds must produce different orders (probability ~1 - 1/10! ≈ 1.0).
            self.assertNotEqual(order_seed1, order_seed2)

    def test_changed_epoch_changes_provenance_order(self):
        """A different epoch produces a different draw order (with overwhelming probability)."""
        from world_model.data.sampling import EpochSampler  # noqa: PLC0415
        from world_model.data.dataset import TemporalWindowDataset  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = _build_multi_episode_catalog(root, num_episodes=10)
            request = world_model_data.TemporalWindowRequest(
                prediction_steps=2, stride_frames=2
            )
            dataset = TemporalWindowDataset(catalog, request)

            sampler_e0 = EpochSampler(dataset, seed=42, draw_count=10)
            sampler_e0.set_epoch(0)
            order_e0 = list(sampler_e0)

            sampler_e1 = EpochSampler(dataset, seed=42, draw_count=10)
            sampler_e1.set_epoch(1)
            order_e1 = list(sampler_e1)

            self.assertNotEqual(order_e0, order_e1)

    def test_fixed_draw_count_is_honored(self):
        """The sampler yields exactly draw_count indices regardless of dataset length."""
        from world_model.data.sampling import EpochSampler  # noqa: PLC0415
        from world_model.data.dataset import TemporalWindowDataset  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = _build_multi_episode_catalog(root, num_episodes=5)
            request = world_model_data.TemporalWindowRequest(
                prediction_steps=2, stride_frames=2
            )
            dataset = TemporalWindowDataset(catalog, request)

            draw_count = 2
            sampler = EpochSampler(dataset, seed=7, draw_count=draw_count)
            sampler.set_epoch(0)
            drawn = list(sampler)

            self.assertEqual(len(drawn), draw_count)

    def test_draw_count_larger_than_dataset_is_honored_with_replacement(self):
        """draw_count > len(dataset) still yields exactly draw_count indices (with replacement)."""
        from world_model.data.sampling import EpochSampler  # noqa: PLC0415
        from world_model.data.dataset import TemporalWindowDataset  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = _build_multi_episode_catalog(root, num_episodes=3)
            request = world_model_data.TemporalWindowRequest(
                prediction_steps=2, stride_frames=2
            )
            dataset = TemporalWindowDataset(catalog, request)
            # Dataset has 3 windows; request 7.
            self.assertEqual(len(dataset), 3)
            sampler = EpochSampler(dataset, seed=5, draw_count=7)
            sampler.set_epoch(0)
            drawn = list(sampler)

            self.assertEqual(len(drawn), 7)
            # All indices must be valid dataset indices.
            for idx in drawn:
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, len(dataset))

    def test_rejects_nonpositive_draw_count_and_incompatible_image_geometry(self):
        """EpochSampler rejects nonpositive draw_count; collator rejects incompatible geometry."""
        from world_model.data.sampling import EpochSampler, TemporalWindowCollator  # noqa: PLC0415
        from world_model.data.dataset import TemporalWindowDataset  # noqa: PLC0415
        import torch  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = _build_multi_episode_catalog(root, num_episodes=2)
            request = world_model_data.TemporalWindowRequest(
                prediction_steps=2, stride_frames=2
            )
            dataset = TemporalWindowDataset(catalog, request)

            # Nonpositive draw_count must raise ValueError.
            with self.assertRaises((ValueError, world_model_data.ContractValueError)):
                EpochSampler(dataset, seed=0, draw_count=0)
            with self.assertRaises((ValueError, world_model_data.ContractValueError)):
                EpochSampler(dataset, seed=0, draw_count=-1)

            # Incompatible context_image geometry: build two fake samples with different H×W.
            sample_a = {
                "context_image": torch.zeros(3, 6, 8),
                "target_images": [torch.zeros(3, 6, 8)],
                "action": torch.zeros(5),
                "frame_indices": [0, 2],
                "horizon_frames": 2,
                "prediction_steps": 1,
                "stride_frames": 2,
                "provenance": {"split": "train", "source_level_key": None, "episode": "ep0",
                               "shot": "s1", "capture_contract": world_model_data.LEGACY_RGB_V1,
                               "declared_capabilities": (), "sidecar_paths": ()},
            }
            sample_b = {
                "context_image": torch.zeros(3, 10, 12),  # Different H×W.
                "target_images": [torch.zeros(3, 10, 12)],
                "action": torch.zeros(5),
                "frame_indices": [0, 2],
                "horizon_frames": 2,
                "prediction_steps": 1,
                "stride_frames": 2,
                "provenance": {"split": "train", "source_level_key": None, "episode": "ep1",
                               "shot": "s1", "capture_contract": world_model_data.LEGACY_RGB_V1,
                               "declared_capabilities": (), "sidecar_paths": ()},
            }
            collator = TemporalWindowCollator()
            with self.assertRaises((ValueError, RuntimeError)):
                collator([sample_a, sample_b])

    def test_rejects_empty_dataset_and_incompatible_action_shape(self):
        from world_model.data.sampling import EpochSampler, TemporalWindowCollator  # noqa: PLC0415
        import torch  # noqa: PLC0415

        with self.assertRaises(world_model_data.ContractValueError):
            EpochSampler([], seed=0, draw_count=1)

        malformed_action_sample = {
            "context_image": torch.zeros(3, 6, 8),
            "target_images": [torch.zeros(3, 6, 8)],
            "action": torch.zeros(4),
            "frame_indices": [0, 2],
            "horizon_frames": 2,
            "prediction_steps": 1,
            "stride_frames": 2,
            "provenance": {"split": "train", "source_level_key": None, "episode": "ep0",
                           "shot": "s1", "capture_contract": world_model_data.LEGACY_RGB_V1,
                           "declared_capabilities": (), "sidecar_paths": ()},
        }
        with self.assertRaises(world_model_data.ContractValueError):
            TemporalWindowCollator()([malformed_action_sample])

    def test_collator_stacks_context_and_actions_pads_targets(self):
        """Collator stacks equal context_images and [5] actions; pads variable target_images."""
        from world_model.data.sampling import TemporalWindowCollator  # noqa: PLC0415
        import torch  # noqa: PLC0415

        # Build two samples with different numbers of target images.
        sample_short = {
            "context_image": torch.ones(3, 6, 8) * 0.1,
            "target_images": [torch.ones(3, 6, 8) * 0.2],           # 1 target
            "action": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
            "frame_indices": [0, 2],
            "horizon_frames": 2,
            "prediction_steps": 1,
            "stride_frames": 2,
            "provenance": {"split": "train", "source_level_key": None, "episode": "ep0",
                           "shot": "s1", "capture_contract": world_model_data.LEGACY_RGB_V1,
                           "declared_capabilities": (), "sidecar_paths": ()},
        }
        sample_long = {
            "context_image": torch.ones(3, 6, 8) * 0.5,
            "target_images": [                                        # 2 targets
                torch.ones(3, 6, 8) * 0.6,
                torch.ones(3, 6, 8) * 0.7,
            ],
            "action": torch.tensor([6.0, 7.0, 8.0, 9.0, 10.0]),
            "frame_indices": [0, 2, 4],
            "horizon_frames": 4,
            "prediction_steps": 2,
            "stride_frames": 2,
            "provenance": {"split": "train", "source_level_key": None, "episode": "ep1",
                           "shot": "s1", "capture_contract": world_model_data.LEGACY_RGB_V1,
                           "declared_capabilities": (), "sidecar_paths": ()},
        }

        collator = TemporalWindowCollator()
        batch = collator([sample_short, sample_long])

        # context_image: stacked to [B, C, H, W].
        self.assertEqual(batch["context_image"].shape, torch.Size([2, 3, 6, 8]))

        # action: stacked to [B, 5].
        self.assertEqual(batch["action"].shape, torch.Size([2, 5]))

        # target_images: padded to [B, T_max, C, H, W] where T_max=2.
        self.assertEqual(batch["target_images"].shape, torch.Size([2, 2, 3, 6, 8]))

        # target_mask: boolean [B, T_max]; True means real (not padded).
        target_mask = batch["target_mask"]
        self.assertEqual(target_mask.shape, torch.Size([2, 2]))
        self.assertEqual(target_mask.dtype, torch.bool)
        # sample_short has 1 real target: mask[0] = [True, False].
        self.assertTrue(target_mask[0, 0].item())
        self.assertFalse(target_mask[0, 1].item())
        # sample_long has 2 real targets: mask[1] = [True, True].
        self.assertTrue(target_mask[1, 0].item())
        self.assertTrue(target_mask[1, 1].item())

        # prediction_steps: [B] tensor.
        self.assertEqual(batch["prediction_steps"].shape, torch.Size([2]))
        self.assertEqual(batch["prediction_steps"][0].item(), 1)
        self.assertEqual(batch["prediction_steps"][1].item(), 2)

        # provenance: list of dicts, length B.
        self.assertIsInstance(batch["provenance"], list)
        self.assertEqual(len(batch["provenance"]), 2)
        self.assertIs(batch["provenance"][0], sample_short["provenance"])
        self.assertIs(batch["provenance"][1], sample_long["provenance"])

    def test_collator_mask_only_pads_where_shorter(self):
        """Padded positions in target_mask are False; real positions are True."""
        from world_model.data.sampling import TemporalWindowCollator  # noqa: PLC0415
        import torch  # noqa: PLC0415

        # Three samples with 1, 2, and 3 targets respectively.
        def _make_sample(n_targets: int, episode: str) -> dict:
            return {
                "context_image": torch.zeros(3, 4, 4),
                "target_images": [torch.zeros(3, 4, 4) for _ in range(n_targets)],
                "action": torch.zeros(5),
                "frame_indices": list(range(n_targets + 1)),
                "horizon_frames": n_targets,
                "prediction_steps": n_targets,
                "stride_frames": 1,
                "provenance": {"split": "train", "source_level_key": None,
                               "episode": episode, "shot": "s1",
                               "capture_contract": world_model_data.LEGACY_RGB_V1,
                               "declared_capabilities": (), "sidecar_paths": ()},
            }

        samples = [_make_sample(1, "ep0"), _make_sample(2, "ep1"), _make_sample(3, "ep2")]
        collator = TemporalWindowCollator()
        batch = collator(samples)

        mask = batch["target_mask"]  # [3, 3]
        self.assertEqual(mask.shape, torch.Size([3, 3]))
        # ep0: 1 real → [T, F, F]
        self.assertEqual(mask[0].tolist(), [True, False, False])
        # ep1: 2 real → [T, T, F]
        self.assertEqual(mask[1].tolist(), [True, True, False])
        # ep2: 3 real → [T, T, T]
        self.assertEqual(mask[2].tolist(), [True, True, True])

    def test_no_global_random_state_dependency(self):
        """EpochSampler order is reproducible regardless of Python/torch global random state."""
        from world_model.data.sampling import EpochSampler  # noqa: PLC0415
        from world_model.data.dataset import TemporalWindowDataset  # noqa: PLC0415
        import random  # noqa: PLC0415
        import torch  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = _build_multi_episode_catalog(root, num_episodes=5)
            request = world_model_data.TemporalWindowRequest(
                prediction_steps=2, stride_frames=2
            )
            dataset = TemporalWindowDataset(catalog, request)

            # Perturb global random state before first sampler.
            random.seed(99999)
            torch.manual_seed(99999)

            sampler_a = EpochSampler(dataset, seed=42, draw_count=5)
            sampler_a.set_epoch(3)
            order_a = list(sampler_a)

            # Perturb again before second sampler.
            random.seed(12345)
            torch.manual_seed(12345)

            sampler_b = EpochSampler(dataset, seed=42, draw_count=5)
            sampler_b.set_epoch(3)
            order_b = list(sampler_b)

            self.assertEqual(order_a, order_b)

    def test_manual_qa_real_dataset_collation(self):
        """Manual QA: real RGB fixtures with unequal target lengths collate correctly."""
        from world_model.data.sampling import EpochSampler, TemporalWindowCollator  # noqa: PLC0415
        from world_model.data.dataset import TemporalWindowDataset  # noqa: PLC0415
        from world_model.data.catalog import EpisodeCatalog  # noqa: PLC0415
        import torch  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_dir = root / "train"
            split_dir.mkdir(parents=True)

            # Episode A: 5 frames → 1 eligible window (prediction_steps=2, stride=2).
            ep_a = split_dir / "episode_A"
            make_complete_rollout_episode(ep_a, RolloutFixtureSpec(frame_count=5, shot_count=1))
            _fix_episode_for_count1(ep_a)

            # Episode B: 7 frames → 3 eligible windows (start frames 0,1,2).
            ep_b = split_dir / "episode_B"
            make_complete_rollout_episode(ep_b, RolloutFixtureSpec(frame_count=7, shot_count=1))
            _fix_episode_for_count1(ep_b)

            catalog = EpisodeCatalog.build(
                root=root, split="train",
                capture_contract=world_model_data.LEGACY_RGB_V1,
            )
            # Use prediction_steps=2, stride=2: ep_A has 1 window, ep_B has 3 windows → 4 total.
            request = world_model_data.TemporalWindowRequest(prediction_steps=2, stride_frames=2)
            dataset = TemporalWindowDataset(catalog, request)
            self.assertEqual(len(dataset), 4)

            # Sample 4 items, collate them.
            sampler = EpochSampler(dataset, seed=0, draw_count=4)
            sampler.set_epoch(0)
            indices = list(sampler)
            samples = [dataset[i] for i in indices]
            collator = TemporalWindowCollator()
            batch = collator(samples)

            # action must be [B, 5].
            self.assertEqual(batch["action"].shape, torch.Size([4, 5]))

            # target_mask: only padded positions are False.
            mask = batch["target_mask"]
            self.assertEqual(mask.shape[0], 4)
            # All samples have prediction_steps=2, so no padding here.
            self.assertTrue(mask.all().item(), "No padding expected: all have same prediction_steps")

            # Same sampler order across two independent instances.
            sampler_x = EpochSampler(dataset, seed=0, draw_count=4)
            sampler_x.set_epoch(0)
            sampler_y = EpochSampler(dataset, seed=0, draw_count=4)
            sampler_y.set_epoch(0)
            self.assertEqual(list(sampler_x), list(sampler_y))

            # Root cleanup is handled by TemporaryDirectory context manager.


class InspectionCommandTests(unittest.TestCase):
    def test_report_counts_complete_and_partial_episodes_without_opening_images(self):
        from world_model.data.inspect import inspect_root

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_complete_rollout_episode(root / "train" / "complete")
            partial = root / "train" / "partial"
            partial.mkdir(parents=True)

            report = inspect_root(root, ("train",), world_model_data.LEGACY_RGB_V1)

            split = report["splits"]["train"]
            self.assertEqual(split["accepted_episodes"], 1)
            self.assertEqual(split["rejected_episodes"], 1)
            self.assertEqual(split["accepted_shots"], 1)
            self.assertEqual(split["window_count"], 0)
            self.assertIn("rejection_counts", split)
            self.assertEqual(report["capture_contract"]["contract_name"], "legacy_rgb_v1")

    def test_report_rejects_unreadable_frame_without_materializing_records(self):
        import os
        import unittest.mock

        from world_model.data.inspect import inspect_root

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode_dir = make_complete_rollout_episode(root / "train" / "episode")
            frame_path = episode_dir / "shot_001" / "frames" / "frame_000000.png"
            real_access = os.access

            def access_with_unreadable_frame(path: Path, mode: int) -> bool:
                return False if Path(path) == frame_path else real_access(path, mode)

            with unittest.mock.patch(
                "scripts.rollout_artifacts.os.access",
                side_effect=access_with_unreadable_frame,
            ):
                report = inspect_root(root, ("train",), world_model_data.LEGACY_RGB_V1)

            split = report["splits"]["train"]
            self.assertEqual(split["accepted_episodes"], 0)
            self.assertEqual(split["rejected_episodes"], 1)
            self.assertEqual(split["rejection_counts"], {"unreadable_artifact": 1})

    def test_report_marks_no_plan_composition_unavailable(self):
        from world_model.data.inspect import inspect_root

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_complete_rollout_episode(
                root / "train" / "novelty_level_7_type010105_fake_name"
            )

            report = inspect_root(root, ("train",), world_model_data.LEGACY_RGB_V1)

            split = report["splits"]["train"]
            unavailable = {"status": "unavailable", "counts": {}}
            self.assertEqual(split["novelty_level_composition"], unavailable)
            self.assertEqual(split["scenario_type_composition"], unavailable)

    def test_report_marks_requested_empty_split_infeasible(self):
        from world_model.data.inspect import inspect_root

        with tempfile.TemporaryDirectory() as temporary:
            report = inspect_root(
                Path(temporary), ("test",), world_model_data.LEGACY_RGB_V1
            )

            self.assertEqual(report["splits"]["test"]["accepted_episodes"], 0)
            self.assertFalse(report["splits"]["test"]["windows_feasible"])

    def test_cli_emits_json_for_complete_fixture(self):
        import contextlib
        import io

        from world_model.data.inspect import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_complete_rollout_episode(root / "train" / "episode_001")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(["--root", str(root), "--splits", "train", "--json"])

            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            split = payload["splits"]["train"]
            self.assertEqual(split["accepted_episodes"], 1)
            self.assertEqual(split["rejected_episodes"], 0)
            self.assertEqual(split["accepted_shots"], 1)
            self.assertEqual(split["rejection_counts"], {})


class CurriculumPolicyTests(unittest.TestCase):
    def test_existing_dataset_candidate_order_and_provenance_are_immutable(self):
        # Given: the existing catalog and temporal dataset snapshot.
        from world_model.data.dataset import TemporalWindowDataset  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            catalog = _build_multi_episode_catalog(Path(temporary), num_episodes=2)
            dataset = TemporalWindowDataset(
                catalog,
                world_model_data.TemporalWindowRequest(1, 1),
            )

            # When: the existing candidate facts are captured before any filesystem change.
            candidates = tuple(
                (
                    dataset[index]["provenance"]["episode"],
                    dataset[index]["provenance"]["shot"],
                    dataset[index]["frame_indices"][0],
                )
                for index in range(len(dataset))
            )
            make_complete_rollout_episode(
                Path(temporary) / "train" / "late_episode",
                RolloutFixtureSpec(frame_count=5, shot_count=1),
            )

            # Then: the materialized dataset candidate and provenance facts do not change.
            self.assertEqual(len(catalog.episodes), 2)
            self.assertEqual(candidates, tuple(
                (
                    dataset[index]["provenance"]["episode"],
                    dataset[index]["provenance"]["shot"],
                    dataset[index]["frame_indices"][0],
                )
                for index in range(len(dataset))
            ))

    def _schedule(self):
        from world_model.data.curriculum import CurriculumSchedule, CurriculumStage  # noqa: PLC0415

        return CurriculumSchedule(
            version="curriculum-v1",
            total_steps=9,
            stages=(
                CurriculumStage(
                    name="short",
                    start_step=0,
                    end_step=3,
                    temporal_choices=(world_model_data.TemporalWindowRequest(1, 1),),
                    start_frame_range=(0.0, 0.5),
                ),
                CurriculumStage(
                    name="medium",
                    start_step=3,
                    end_step=6,
                    temporal_choices=(world_model_data.TemporalWindowRequest(2, 1),),
                ),
                CurriculumStage(
                    name="long",
                    start_step=6,
                    end_step=9,
                    temporal_choices=(world_model_data.TemporalWindowRequest(2, 2),),
                ),
            ),
        )

    def test_three_stage_schedule_selects_exact_temporal_choices_at_boundaries(self):
        """Boundary steps use their declared half-open stage and temporal request."""
        from world_model.data.curriculum import CurriculumPolicy  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            # Given: one fixed catalog and three contiguous temporal stages.
            catalog = _build_catalog_from_fixture(
                Path(temporary), frame_count=9, shot_count=1
            )
            policy = CurriculumPolicy(catalog, self._schedule(), sampler_seed=17)

            # When: views are requested at every stage boundary.
            boundary_views = tuple(
                policy.candidate_view(global_step=step, total_steps=9)
                for step in (0, 2, 3, 5, 6, 8)
            )

            # Then: a boundary belongs to the next stage and only its choices appear.
            self.assertEqual(
                tuple(view.active_stage.name for view in boundary_views),
                ("short", "short", "medium", "medium", "long", "long"),
            )
            self.assertEqual(
                tuple(view.active_stage.temporal_choices[0] for view in boundary_views),
                (
                    world_model_data.TemporalWindowRequest(1, 1),
                    world_model_data.TemporalWindowRequest(1, 1),
                    world_model_data.TemporalWindowRequest(2, 1),
                    world_model_data.TemporalWindowRequest(2, 1),
                    world_model_data.TemporalWindowRequest(2, 2),
                    world_model_data.TemporalWindowRequest(2, 2),
                ),
            )
            self.assertTrue(
                all(
                    candidate.request in view.active_stage.temporal_choices
                    for view in boundary_views
                    for candidate in view.candidates
                )
            )
            self.assertTrue(
                all(
                    candidate.normalized_start_frame < 0.5
                    for candidate in boundary_views[0].candidates
                )
            )

    def test_invalid_overlapping_or_uncovered_stages_fail_validation(self):
        """Schedules reject overlap, gaps, and temporal stages without choices."""
        from world_model.data.curriculum import CurriculumSchedule, CurriculumStage  # noqa: PLC0415

        # Given: invalid stage layouts against a six-step training run.
        invalid_schedules = (
            (
                CurriculumStage("first", 0, 4, (world_model_data.TemporalWindowRequest(1, 1),)),
                CurriculumStage("second", 3, 6, (world_model_data.TemporalWindowRequest(1, 1),)),
            ),
            (
                CurriculumStage("first", 0, 2, (world_model_data.TemporalWindowRequest(1, 1),)),
                CurriculumStage("second", 3, 6, (world_model_data.TemporalWindowRequest(1, 1),)),
            ),
        )

        # When/Then: each malformed declaration is rejected at construction.
        for stages in invalid_schedules:
            with self.subTest(stages=stages):
                with self.assertRaises(world_model_data.ContractValueError):
                    CurriculumSchedule(version="v1", total_steps=6, stages=stages)
        with self.assertRaises(world_model_data.ContractValueError):
            CurriculumStage("empty", 0, 6, ())

    def test_step_bindings_reject_bool_and_float_equivalents(self):
        schedule = world_model_data.CurriculumSchedule(
            version="v1",
            total_steps=1,
            stages=(
                world_model_data.CurriculumStage(
                    "only",
                    0,
                    1,
                    (world_model_data.TemporalWindowRequest(1, 1),),
                ),
            ),
        )

        for malformed_total_steps in (True, 1.0):
            with self.subTest(total_steps=malformed_total_steps):
                with self.assertRaises(world_model_data.ContractValueError):
                    schedule.active_stage(0, malformed_total_steps)

    def test_normalized_progress_bounds_reject_boolean_values(self):
        with self.assertRaises(world_model_data.ContractValueError):
            world_model_data.CurriculumStage(
                "bad-progress",
                0,
                1,
                (world_model_data.TemporalWindowRequest(1, 1),),
                start_frame_range=(False, 0.5),
            )

    def test_schedule_digest_is_canonical_for_allowed_sets(self):
        first_request = world_model_data.TemporalWindowRequest(1, 1)
        second_request = world_model_data.TemporalWindowRequest(2, 1)
        first = world_model_data.CurriculumSchedule(
            version="v1",
            total_steps=1,
            stages=(world_model_data.CurriculumStage(
                "only",
                0,
                1,
                (first_request, second_request),
                novelty_levels=("novelty_level_2", "novelty_level_1"),
                scenario_types=("type0102", "type0101"),
            ),),
        )
        second = world_model_data.CurriculumSchedule(
            version="v1",
            total_steps=1,
            stages=(world_model_data.CurriculumStage(
                "only",
                0,
                1,
                (second_request, first_request),
                novelty_levels=("novelty_level_1", "novelty_level_2"),
                scenario_types=("type0101", "type0102"),
            ),),
        )

        self.assertEqual(first.digest, second.digest)

    def test_one_frame_normalized_start_is_zero(self):
        from world_model.data.curriculum import _normalized_start_frame  # noqa: PLC0415

        self.assertEqual(_normalized_start_frame(0, 1), 0.0)

    def test_requested_source_filters_fail_closed_without_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalog = _build_catalog_from_fixture(Path(temporary), frame_count=3)
            schedule = world_model_data.CurriculumSchedule(
                version="v1",
                total_steps=1,
                stages=(world_model_data.CurriculumStage(
                    "filtered",
                    0,
                    1,
                    (world_model_data.TemporalWindowRequest(1, 1),),
                    novelty_levels=("novelty_level_1",),
                    scenario_types=("type0101",),
                ),),
            )
            policy = world_model_data.CurriculumPolicy(catalog, schedule, sampler_seed=5)

            view = policy.candidate_view(0, 1)

            self.assertEqual(view.candidates, ())

    def test_plan_source_filters_select_only_declared_provenance(self):
        from world_model.data.catalog import EpisodeCatalog  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "output"
            episode_root = output_root / "train"
            episode_root.mkdir(parents=True)
            entry = LevelEntry(
                "novelty_level_1",
                "type010101",
                "novelty_level_1/type010101/Levels/one.xml",
            )
            episode_dir = episode_root / _safe_output_name(entry)
            make_complete_rollout_episode(
                episode_dir,
                RolloutFixtureSpec(frame_count=3, shot_count=1),
            )
            _fix_episode_for_count1(episode_dir)
            options = CollectionOptions(count=1, fps=30.0, duration=5.0, workers=1)
            plan_path = write_collection_plan(
                root / "plan",
                output_root=output_root,
                episodes=[PlannedEpisode("train", entry, episode_dir, "existing")],
                summary={f"train:{entry.bucket}": {"target": 1, "existing": 1, "scheduled": 0}},
                options=options,
                targets=CollectionTargets(train=1, dev=1, test=0),
                selected_splits=("train",),
                seed="curriculum-filter-test",
            )
            catalog = EpisodeCatalog.build(
                output_root,
                "train",
                world_model_data.LEGACY_RGB_V1,
                collection_plan=plan_path,
            )
            schedule = world_model_data.CurriculumSchedule(
                "v1",
                1,
                (world_model_data.CurriculumStage(
                    "filtered",
                    0,
                    1,
                    (world_model_data.TemporalWindowRequest(1, 1),),
                    novelty_levels=("novelty_level_1",),
                    scenario_types=("type010101",),
                ),),
            )

            candidates = world_model_data.CurriculumPolicy(
                catalog, schedule, sampler_seed=5
            ).candidate_view(0, 1).candidates

            self.assertEqual({candidate.start_frame for candidate in candidates}, {0, 1})
            self.assertTrue(all(candidate.episode.source_level_key == entry.relative_path
                                for candidate in candidates))

    def test_start_frame_range_is_exactly_half_open(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalog = _build_catalog_from_fixture(Path(temporary), frame_count=9)
            schedule = world_model_data.CurriculumSchedule(
                "v1",
                1,
                (world_model_data.CurriculumStage(
                    "middle",
                    0,
                    1,
                    (world_model_data.TemporalWindowRequest(1, 1),),
                    start_frame_range=(0.25, 0.5),
                ),),
            )

            candidates = world_model_data.CurriculumPolicy(
                catalog, schedule, sampler_seed=5
            ).candidate_view(0, 1).candidates

            self.assertEqual({candidate.start_frame for candidate in candidates}, {2, 3})

    def test_one_frame_catalog_has_no_positive_horizon_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalog = _build_catalog_from_fixture(Path(temporary), frame_count=1)
            schedule = world_model_data.CurriculumSchedule(
                "v1",
                1,
                (world_model_data.CurriculumStage(
                    "only", 0, 1, (world_model_data.TemporalWindowRequest(1, 1),)
                ),),
            )

            view = world_model_data.CurriculumPolicy(
                catalog, schedule, sampler_seed=5
            ).candidate_view(0, 1)

            self.assertEqual(view.candidates, ())

    def test_candidate_order_is_independent_of_global_random_state(self):
        import random

        with tempfile.TemporaryDirectory() as temporary:
            catalog = _build_catalog_from_fixture(Path(temporary), frame_count=9)
            policy = world_model_data.CurriculumPolicy(
                catalog, self._schedule(), sampler_seed=17
            )
            first_ids = tuple(candidate.candidate_id
                              for candidate in policy.candidate_view(3, 9).candidates)

            random.seed(987654321)
            for _ in range(20):
                random.random()
            second_ids = tuple(candidate.candidate_id
                               for candidate in policy.candidate_view(3, 9).candidates)

            self.assertEqual(first_ids, second_ids)

    def test_state_serializes_every_resume_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalog = _build_catalog_from_fixture(Path(temporary), frame_count=9)
            policy = world_model_data.CurriculumPolicy(
                catalog, self._schedule(), sampler_seed=17
            )

            payload = policy.state(3, 9).to_dict()

            self.assertEqual(payload["global_step"], 3)
            self.assertEqual(payload["total_steps"], 9)
            self.assertEqual(payload["schedule_version"], "curriculum-v1")
            self.assertEqual(payload["schedule_digest"], self._schedule().digest)
            self.assertEqual(payload["catalog_digest"], policy.catalog_digest)
            self.assertEqual(payload["sampler_seed"], 17)
            self.assertEqual(payload["active_stage_name"], "medium")

    def test_resume_rejects_schedule_catalog_or_seed_drift(self):
        """A checkpoint only resumes under exactly the bound catalog, schedule, and seed."""
        from world_model.data.curriculum import (  # noqa: PLC0415
            CurriculumBindingMismatchError,
            CurriculumPolicy,
        )

        with tempfile.TemporaryDirectory() as temporary:
            # Given: a saved policy state from one deterministic binding.
            catalog = _build_catalog_from_fixture(
                Path(temporary), frame_count=9, shot_count=1
            )
            schedule = self._schedule()
            policy = CurriculumPolicy(catalog, schedule, sampler_seed=17)
            state = policy.state(global_step=3, total_steps=9)

            # When/Then: every binding drift is fail-closed.
            mismatched_policies = (
                CurriculumPolicy(catalog, schedule, sampler_seed=18),
                CurriculumPolicy(
                    catalog,
                    type(schedule)(
                        version="curriculum-v2",
                        total_steps=9,
                        stages=schedule.stages,
                    ),
                    sampler_seed=17,
                ),
            )
            for mismatched_policy in mismatched_policies:
                with self.subTest(policy=mismatched_policy):
                    with self.assertRaises(CurriculumBindingMismatchError):
                        mismatched_policy.validate_resume(state)

            catalog_drift_state = type(state)(
                global_step=state.global_step,
                total_steps=state.total_steps,
                schedule_version=state.schedule_version,
                schedule_digest=state.schedule_digest,
                catalog_digest="changed-catalog",
                sampler_seed=state.sampler_seed,
                active_stage_name=state.active_stage_name,
            )
            with self.assertRaises(CurriculumBindingMismatchError):
                policy.validate_resume(catalog_drift_state)

    def test_identical_resume_binding_yields_identical_candidate_ids(self):
        """Candidate ordering is reproducible from the catalog, schedule, and seed alone."""
        from world_model.data.curriculum import CurriculumPolicy  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as temporary:
            # Given: two policies sharing the same immutable binding.
            catalog = _build_catalog_from_fixture(
                Path(temporary), frame_count=9, shot_count=1
            )
            first_policy = CurriculumPolicy(catalog, self._schedule(), sampler_seed=17)
            second_policy = CurriculumPolicy(catalog, self._schedule(), sampler_seed=17)
            state = first_policy.state(global_step=3, total_steps=9)

            # When: the resumed policy validates the checkpoint and rebuilds its view.
            second_policy.validate_resume(state)
            first_ids = tuple(
                candidate.candidate_id
                for candidate in first_policy.candidate_view(3, 9).candidates
            )
            second_ids = tuple(
                candidate.candidate_id
                for candidate in second_policy.candidate_view(3, 9).candidates
            )

            # Then: no wall-clock or mutable catalog state affects the candidates.
            self.assertEqual(first_ids, second_ids)


class TemporalAblationTests(unittest.TestCase):
    def _policy(self, root, temporal_choices):
        catalog = _build_catalog_from_fixture(root, frame_count=12, shot_count=1)
        schedule = world_model_data.CurriculumSchedule(
            "ablation-schedule-v1",
            1,
            (world_model_data.CurriculumStage(
                "only", 0, 1, temporal_choices,
            ),),
        )
        policy = world_model_data.CurriculumPolicy(catalog, schedule, sampler_seed=23)
        return policy, policy.state(0, 1)

    def _manifest(self, root, preset_name, draw_count, cost_rule, *, sampling_seed=31):
        preset = world_model_data.get_temporal_ablation_preset(preset_name)
        policy, state = self._policy(root, preset.temporal_choices)
        return self._bound_manifest(
            policy, state, preset, draw_count, cost_rule, sampling_seed=sampling_seed,
        )

    def _bound_manifest(
        self, policy, state, preset, draw_count, cost_rule, *, sampling_seed=31,
    ):
        return world_model_data.build_temporal_ablation_manifest(
            policy,
            state,
            world_model_data.AblationRunConfig(
                preset, sampling_seed, draw_count, cost_rule,
            ),
        )

    def test_presets_expose_only_the_four_declared_temporal_choice_sets(self):
        expected = {
            "fixed_short": ((1, 1),),
            "fixed_long": ((4, 2),),
            "temporal_uniform": ((1, 1), (2, 1), (4, 2)),
            "temporal_curriculum": ((1, 1), (2, 1), (4, 2)),
        }

        observed = {
            name: tuple(
                (choice.prediction_steps, choice.stride_frames)
                for choice in world_model_data.get_temporal_ablation_preset(name).temporal_choices
            )
            for name in expected
        }

        self.assertEqual(observed, expected)
        self.assertTrue(all(
            not hasattr(world_model_data.get_temporal_ablation_preset(name), field)
            for name in expected
            for field in ("abstraction", "symbolic")
        ))

    def test_preset_validation_rejects_unknown_empty_duplicate_or_unsupported_fields(self):
        short = world_model_data.TemporalWindowRequest(1, 1)

        with self.assertRaises(world_model_data.ContractValueError):
            world_model_data.get_temporal_ablation_preset("unknown")
        with self.assertRaises(world_model_data.ContractValueError):
            world_model_data.TemporalAblationPreset("fixed_short", ())
        with self.assertRaises(world_model_data.ContractValueError):
            world_model_data.TemporalAblationPreset("fixed_short", (short, short))
        with self.assertRaises(world_model_data.ContractValueError):
            world_model_data.TemporalAblationPreset("fixed_long", (short,))
        with self.assertRaises(TypeError):
            world_model_data.TemporalAblationPreset(
                "fixed_short", (short,), abstraction="object"
            )

    def test_preset_and_cost_integer_fields_reject_bool_or_float_equivalents(self):
        for malformed in (True, 1.0):
            with self.subTest(malformed=malformed):
                with self.assertRaises(world_model_data.ContractValueError):
                    world_model_data.WindowCostRule("frames", malformed, 1)
                with self.assertRaises(world_model_data.ContractValueError):
                    world_model_data.AblationRunConfig(
                        world_model_data.get_temporal_ablation_preset("fixed_short"),
                        1,
                        malformed,
                        world_model_data.WindowCostRule("frames", 0, 1),
                    )

    def test_cost_rule_rejects_malformed_identity_or_cost_coefficients(self):
        malformed_rules = (
            ("", 0, 1),
            ("frames", -1, 1),
            ("frames", 0, 0),
        )

        for values in malformed_rules:
            with self.subTest(values=values):
                with self.assertRaises(world_model_data.ContractValueError):
                    world_model_data.WindowCostRule(*values)

    def test_manifest_budget_totals_are_exact_hand_calculated_values(self):
        rule = world_model_data.WindowCostRule("frame-units", 2, 5)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._manifest(
                Path(temporary), "fixed_short", 3, rule,
            )

        self.assertEqual(manifest.prediction_steps_distribution, ((1, 3),))
        self.assertEqual(manifest.stride_frames_distribution, ((1, 3),))
        self.assertEqual(manifest.horizon_frames_distribution, ((1, 3),))
        self.assertEqual(manifest.total_prediction_steps, 3)
        self.assertEqual(manifest.effective_prediction_steps, 1.0)
        self.assertEqual(manifest.total_predicted_frame_cost, 15)
        self.assertEqual(manifest.temporal_controller_cost, 0)
        self.assertEqual(manifest.computed_budget_total, 21)

    def test_manifest_identity_and_serialization_are_deterministic(self):
        rule = world_model_data.WindowCostRule("frame-units", 2, 5)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preset = world_model_data.get_temporal_ablation_preset("temporal_uniform")
            policy, state = self._policy(root, preset.temporal_choices)
            first = self._bound_manifest(policy, state, preset, 9, rule)
            second = self._bound_manifest(policy, state, preset, 9, rule)

        self.assertEqual(first, second)
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(json.dumps(first.to_dict(), sort_keys=True),
                         json.dumps(second.to_dict(), sort_keys=True))
        self.assertEqual(sum(count for _, count in first.prediction_steps_distribution), 9)
        self.assertEqual(tuple(key for key, _ in first.prediction_steps_distribution),
                         tuple(sorted(key for key, _ in first.prediction_steps_distribution)))

    def test_sampling_seed_changes_sampled_provenance_without_global_random_state(self):
        import random

        rule = world_model_data.WindowCostRule("frame-units", 0, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preset = world_model_data.get_temporal_ablation_preset("fixed_short")
            policy, state = self._policy(root, preset.temporal_choices)
            first = self._bound_manifest(policy, state, preset, 6, rule, sampling_seed=31)
            random.seed(999)
            for _ in range(20):
                random.random()
            repeated = self._bound_manifest(
                policy, state, preset, 6, rule, sampling_seed=31,
            )
            changed = self._bound_manifest(
                policy, state, preset, 6, rule, sampling_seed=32,
            )

        self.assertEqual(first.sampled_provenance_digest,
                         repeated.sampled_provenance_digest)
        self.assertNotEqual(first.sampled_provenance_digest,
                            changed.sampled_provenance_digest)

    def test_stale_curriculum_state_is_rejected_instead_of_fabricating_samples(self):
        rule = world_model_data.WindowCostRule("frame-units", 0, 1)
        preset = world_model_data.get_temporal_ablation_preset("fixed_short")
        with tempfile.TemporaryDirectory() as temporary:
            policy, state = self._policy(Path(temporary), preset.temporal_choices)
            stale = type(state)(
                state.global_step, state.total_steps, state.schedule_version,
                state.schedule_digest, "stale-catalog", state.sampler_seed,
                state.active_stage_name,
            )

            with self.assertRaises(world_model_data.CurriculumBindingMismatchError):
                world_model_data.build_temporal_ablation_manifest(
                    policy,
                    stale,
                    world_model_data.AblationRunConfig(preset, 31, 2, rule),
                )

    def test_equal_cost_presets_are_compute_matched(self):
        rule = world_model_data.WindowCostRule("predicted-frames", 0, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            short = self._manifest(root / "short", "fixed_short", 4, rule)
            long = self._manifest(root / "long", "fixed_long", 1, rule)

        comparison = world_model_data.compare_temporal_ablation_manifests(short, long)

        self.assertEqual(short.computed_budget_total, 4)
        self.assertEqual(long.computed_budget_total, 4)
        self.assertEqual(comparison.sample_match, "sample_unmatched")
        self.assertEqual(comparison.compute_match, "compute_matched")

    def test_equal_draw_count_with_unequal_cost_is_not_compute_matched(self):
        rule = world_model_data.WindowCostRule("predicted-frames", 0, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            short = self._manifest(root / "short", "fixed_short", 2, rule)
            long = self._manifest(root / "long", "fixed_long", 2, rule)

        comparison = world_model_data.compare_temporal_ablation_manifests(short, long)

        self.assertEqual(comparison.sample_match, "sample_matched")
        self.assertEqual(comparison.compute_match, "compute_unmatched")

    def test_sample_matching_requires_cost_rule_identity_not_equal_cost_only(self):
        first_rule = world_model_data.WindowCostRule("first", 0, 1)
        second_rule = world_model_data.WindowCostRule("second", 0, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._manifest(root / "first", "fixed_short", 2, first_rule)
            second = self._manifest(root / "second", "fixed_short", 2, second_rule)

        comparison = world_model_data.compare_temporal_ablation_manifests(first, second)

        self.assertEqual(comparison.sample_match, "sample_unmatched")
        self.assertEqual(comparison.compute_match, "compute_matched")

    def test_required_compute_match_rejects_unequal_computed_totals(self):
        rule = world_model_data.WindowCostRule("predicted-frames", 0, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            short = self._manifest(root / "short", "fixed_short", 2, rule)
            long = self._manifest(root / "long", "fixed_long", 2, rule)

        with self.assertRaises(world_model_data.ComputeBudgetMismatchError):
            world_model_data.compare_temporal_ablation_manifests(
                short, long, require_compute_match=True,
            )

    def test_manifest_is_frozen_and_contains_zero_nonlearned_controller_cost(self):
        rule = world_model_data.WindowCostRule("predicted-frames", 0, 1)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._manifest(Path(temporary), "fixed_short", 2, rule)

        with self.assertRaises(FrozenInstanceError):
            manifest.draw_count = 99
        self.assertEqual(manifest.temporal_controller_cost, 0)


class WorldModelDataIntegrationTests(unittest.TestCase):
    def _plan_backed_fixture(
        self,
        root: Path,
        *,
        duplicate_dev_source: bool = False,
    ):
        output_root = root / "rollouts"
        output_root.mkdir()
        entries = {
            "train": LevelEntry(
                "novelty_level_1",
                "type010101",
                "novelty_level_1/type010101/Levels/train.xml",
            ),
            "dev": LevelEntry(
                "novelty_level_2",
                "type010102",
                "novelty_level_2/type010102/Levels/dev.xml",
            ),
            "test": LevelEntry(
                "novelty_level_3",
                "type010103",
                "novelty_level_3/type010103/Levels/test.xml",
            ),
        }
        if duplicate_dev_source:
            entries["dev"] = entries["train"]

        planned = []
        summary = {}
        for split, entry in entries.items():
            episode_dir = output_root / split / _safe_output_name(entry)
            make_complete_rollout_episode(
                episode_dir,
                RolloutFixtureSpec(frame_count=9, shot_count=1),
            )
            _fix_episode_for_count1(episode_dir)
            planned.append(PlannedEpisode(split, entry, episode_dir, "existing"))
            summary[f"{split}:{entry.bucket}"] = {
                "target": 1,
                "existing": 1,
                "scheduled": 0,
            }

        reserved_entry = LevelEntry(
            "novelty_level_8",
            "type010105",
            "novelty_level_8/type010105/Levels/reserved.xml",
        )
        reserved_dir = output_root / "train" / _safe_output_name(reserved_entry)
        reserved_dir.mkdir(parents=True)
        planned.append(PlannedEpisode("train", reserved_entry, reserved_dir, "scheduled"))
        plan_path = write_collection_plan(
            root / "plan",
            output_root=output_root,
            episodes=planned,
            summary=summary,
            options=CollectionOptions(count=1, fps=30.0, duration=5.0, workers=1),
            targets=CollectionTargets(train=2, dev=1, test=1),
            selected_splits=("train", "dev", "test"),
            seed="integration-plan-v1",
        )
        return output_root, plan_path, entries, reserved_entry

    def test_plan_backed_pipeline_resumes_with_stable_manifest(self):
        # Given: three disjoint plan-backed splits and one reserved incomplete episode.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root, plan_path, entries, reserved_entry = self._plan_backed_fixture(root)
            catalogs = {
                split: world_model_data.EpisodeCatalog.build(
                    output_root,
                    split,
                    world_model_data.LEGACY_RGB_V1,
                    collection_plan=plan_path,
                )
                for split in ("train", "dev", "test")
            }

            # When: every public pipeline layer is exercised and independently resumed.
            world_model_data.check_source_key_disjointness(tuple(catalogs.values()))
            preset = world_model_data.get_temporal_ablation_preset("temporal_curriculum")
            schedule = world_model_data.CurriculumSchedule(
                "integration-schedule-v1",
                2,
                (
                    world_model_data.CurriculumStage(
                        "all-temporal-choices",
                        0,
                        2,
                        preset.temporal_choices,
                    ),
                ),
            )
            request = world_model_data.TemporalWindowRequest(2, 2)
            observed = {}
            for split, catalog in catalogs.items():
                dataset = world_model_data.TemporalWindowDataset(catalog, request)
                if split == "train":
                    context_path = output_root / catalog.episodes[0].shots[0].frames[0].relative_path
                    Image.new("RGB", (8, 6), (101, 102, 103)).save(context_path, format="PNG")
                    self.assertAlmostEqual(dataset[0]["context_image"][0, 0, 0].item(), 101 / 255)

                first_sampler = world_model_data.EpochSampler(dataset, seed=41, draw_count=3)
                first_sampler.set_epoch(4)
                second_sampler = world_model_data.EpochSampler(dataset, seed=41, draw_count=3)
                second_sampler.set_epoch(4)
                first_indices = tuple(first_sampler)
                second_indices = tuple(second_sampler)
                samples = [dataset[index] for index in first_indices]
                batch = world_model_data.TemporalWindowCollator()(samples)

                first_policy = world_model_data.CurriculumPolicy(
                    catalog,
                    schedule,
                    sampler_seed=53,
                )
                state = first_policy.state(1, 2)
                first_config = world_model_data.AblationRunConfig(
                    preset,
                    67,
                    5,
                    world_model_data.WindowCostRule("integration-frame-units", 2, 3),
                )
                first_manifest = world_model_data.build_temporal_ablation_manifest(
                    first_policy,
                    state,
                    first_config,
                )
                resumed_policy = world_model_data.CurriculumPolicy(
                    catalog,
                    schedule,
                    sampler_seed=53,
                )
                resumed_policy.validate_resume(state)
                resumed_config = world_model_data.AblationRunConfig(
                    preset,
                    67,
                    5,
                    world_model_data.WindowCostRule("integration-frame-units", 2, 3),
                )
                resumed_manifest = world_model_data.build_temporal_ablation_manifest(
                    resumed_policy,
                    state,
                    resumed_config,
                )
                first_candidate_ids = tuple(
                    candidate.candidate_id
                    for candidate in first_policy.candidate_view(1, 2).candidates
                )
                resumed_candidate_ids = tuple(
                    candidate.candidate_id
                    for candidate in resumed_policy.candidate_view(1, 2).candidates
                )
                observed[split] = (
                    first_indices,
                    second_indices,
                    batch,
                    first_candidate_ids,
                    resumed_candidate_ids,
                    first_manifest,
                    resumed_manifest,
                    samples,
                )

            # Then: split provenance, decoded batches, and resume identities are stable.
            source_sets = {
                split: {episode.source_level_key for episode in catalog.episodes}
                for split, catalog in catalogs.items()
            }
            self.assertTrue(all(catalog.provenance_available for catalog in catalogs.values()))
            self.assertTrue(source_sets["train"].isdisjoint(source_sets["dev"]))
            self.assertTrue(source_sets["train"].isdisjoint(source_sets["test"]))
            self.assertTrue(source_sets["dev"].isdisjoint(source_sets["test"]))
            self.assertEqual(
                source_sets,
                {split: {entry.relative_path} for split, entry in entries.items()},
            )
            self.assertEqual(catalogs["train"].rejection_count, 1)
            self.assertEqual(catalogs["train"].rejection_code_counts, {"missing_artifact": 1})
            for split, result in observed.items():
                (
                    first_indices,
                    second_indices,
                    batch,
                    first_candidate_ids,
                    resumed_candidate_ids,
                    first_manifest,
                    resumed_manifest,
                    samples,
                ) = result
                self.assertEqual(first_indices, second_indices)
                self.assertEqual(batch["context_image"].shape, (3, 3, 6, 8))
                self.assertEqual(batch["target_images"].shape, (3, 2, 3, 6, 8))
                self.assertEqual(batch["target_mask"].tolist(), [[True, True]] * 3)
                self.assertEqual(batch["action"].shape, (3, 5))
                self.assertEqual(first_candidate_ids, resumed_candidate_ids)
                self.assertEqual(
                    first_manifest.sampled_provenance_digest,
                    resumed_manifest.sampled_provenance_digest,
                )
                self.assertEqual(first_manifest.digest, resumed_manifest.digest)
                self.assertTrue(all(
                    sample["provenance"]["source_level_key"] == entries[split].relative_path
                    for sample in samples
                ))
                self.assertNotIn(
                    reserved_entry.relative_path,
                    json.dumps(first_manifest.to_dict(), sort_keys=True),
                )
            self.assertNotIn(
                _safe_output_name(reserved_entry),
                {episode.name for episode in catalogs["train"].episodes},
            )
            self.assertTrue(all(
                candidate.episode.source_level_key != reserved_entry.relative_path
                for candidate in world_model_data.CurriculumPolicy(
                    catalogs["train"], schedule, sampler_seed=53
                ).candidate_view(1, 2).candidates
            ))

    def test_duplicate_plan_source_key_across_splits_is_rejected(self):
        # Given: train and dev plan records with the same exact source-level key.
        with tempfile.TemporaryDirectory() as temporary:
            output_root, plan_path, entries, _ = self._plan_backed_fixture(
                Path(temporary),
                duplicate_dev_source=True,
            )
            catalogs = tuple(
                world_model_data.EpisodeCatalog.build(
                    output_root,
                    split,
                    world_model_data.LEGACY_RGB_V1,
                    collection_plan=plan_path,
                )
                for split in ("train", "dev", "test")
            )

            # When/Then: the public cross-catalog check rejects the duplicate.
            with self.assertRaisesRegex(
                world_model_data.DuplicateSourceKeyError,
                entries["train"].relative_path,
            ) as raised:
                world_model_data.check_source_key_disjointness(catalogs)
            self.assertEqual(raised.exception.source_key, entries["train"].relative_path)
            self.assertEqual(raised.exception.first_split, "train")
            self.assertEqual(raised.exception.second_split, "dev")

    def test_no_plan_catalog_reports_provenance_unavailable(self):
        # Given: a complete episode without a collection-plan manifest.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode_dir = make_complete_rollout_episode(
                root / "train" / "episode_without_plan",
                RolloutFixtureSpec(frame_count=5, shot_count=1),
            )
            _fix_episode_for_count1(episode_dir)

            # When: public catalog, dataset, and disjointness APIs are exercised.
            catalog = world_model_data.EpisodeCatalog.build(
                root,
                "train",
                world_model_data.LEGACY_RGB_V1,
            )
            dataset = world_model_data.TemporalWindowDataset(
                catalog,
                world_model_data.TemporalWindowRequest(1, 1),
            )
            disjointness_result = world_model_data.check_source_key_disjointness((catalog,))

            # Then: unavailable provenance is explicit and no validation claim exists.
            self.assertFalse(catalog.provenance_available)
            self.assertTrue(all(episode.source_level_key is None for episode in catalog.episodes))
            self.assertTrue(all(
                dataset[index]["provenance"]["source_level_key"] is None
                for index in range(len(dataset))
            ))
            self.assertIsNone(disjointness_result)
            self.assertFalse(hasattr(catalog, "leakage_validated"))
