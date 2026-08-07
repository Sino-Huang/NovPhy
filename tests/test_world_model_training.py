"""Contract tests for the Milestone 1 teacher-forced training loop.

Covers the reproducibility manifest, the anti-collapse diagnostics that make an
overfit claim meaningful, and the single-step teacher-forced trainer itself.
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from tests.test_world_model_data import (
    RolloutFixtureSpec,
    _fix_episode_for_count1,
    make_complete_rollout_episode,
)
from tests.test_world_model_model import small_jepa_config
from world_model.data import LEGACY_RGB_V1, EpisodeCatalog
from world_model.data.dataset import TemporalWindowDataset
from world_model.data.types import ContractValueError, TemporalWindowRequest
from world_model.model import Abstraction, JepaBackbone, PredictionPair
from world_model.training import (
    RunManifest,
    TeacherForcedTrainer,
    TrainingConfig,
    build_window_loader,
    collapse_diagnostics,
    effective_rank,
    mean_feature_std,
    relative_spread,
    resize_transform,
    retrieval_accuracy,
    run_overfit,
    select_diverse_windows,
    select_motion_windows,
)

_LATENT_DIM = 24


def make_manifest(**overrides: object) -> RunManifest:
    """Return a valid manifest with overridable fields."""
    fields: dict[str, object] = {
        "manifest_version": "jepa-run-manifest-v1",
        "run_id": "20260807T000000Z-overfit",
        "mode": "overfit",
        "seed": 20260807,
        "git_commit": "b5e00d2",
        "git_dirty": False,
        "torch_version": "2.13.0+cu130",
        "cuda_version": "13.0",
        "device_name": "NVIDIA GeForce RTX 5090",
        "dataset_root": "/data/novphy_rollouts_dataset_20260708_171531",
        "split": "dev",
        "catalog_digest": "a" * 64,
        "accepted_episode_count": 463,
        "rejected_episode_count": 1137,
        "window_count": 8,
        "prediction_steps": 1,
        "stride_frames": 1,
        "abstraction": "continuous",
        "batch_size": 8,
        "steps": 2000,
        "learning_rate": 3e-4,
        "weight_decay": 0.05,
        "warmup_steps": 100,
        "grad_clip": 1.0,
        "ema_base_momentum": 0.99,
        "model_config_digest": "b" * 64,
        "sampled_index_digest": "c" * 64,
        "window_selection": "motion",
        "candidate_count": 256,
        "symbolic_loss_active": False,
        "final_loss": 1e-5,
        "mean_feature_std": 0.4,
        "relative_spread": 0.35,
        "effective_rank": 6.2,
        "retrieval_accuracy": 1.0,
        "acceptance": "pass",
        "started_at_unix": 1786000000.0,
        "wall_clock_seconds": 91.5,
    }
    fields.update(overrides)
    return RunManifest(**fields)  # type: ignore[arg-type]


def make_fixture_catalog(root: Path, *, episode_count: int, frame_count: int) -> EpisodeCatalog:
    """Build a real catalog over freshly written legacy-RGB episode fixtures."""
    split_dir = root / "dev"
    split_dir.mkdir(parents=True)
    for index in range(episode_count):
        episode_dir = split_dir / f"novelty_level_0_type010101_{index:05d}"
        make_complete_rollout_episode(
            episode_dir,
            RolloutFixtureSpec(frame_count=frame_count, shot_count=1),
        )
        _fix_episode_for_count1(episode_dir)
    return EpisodeCatalog.build(root=root, split="dev", capture_contract=LEGACY_RGB_V1)


def make_training_config(**overrides: object) -> TrainingConfig:
    """Return a small, CPU-friendly training configuration."""
    fields: dict[str, object] = {
        "seed": 20260807,
        "steps": 10,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "grad_clip": 1.0,
        "ema_base_momentum": 0.9,
        "delta": 1,
        "abstraction": Abstraction.CONTINUOUS,
        "device": "cpu",
    }
    fields.update(overrides)
    return TrainingConfig(**fields)  # type: ignore[arg-type]


def make_fixture_batch(seed: int, *, batch_size: int = 4) -> dict:
    """Return a deterministic synthetic TemporalWindowBatch-shaped mapping."""
    generator = torch.Generator().manual_seed(seed)
    config = small_jepa_config().encoder
    shape = (batch_size, 3, config.input_height, config.input_width)
    return {
        "context_image": torch.rand(shape, generator=generator),
        "target_images": torch.rand((batch_size, 1, *shape[1:]), generator=generator),
        "target_mask": torch.ones((batch_size, 1), dtype=torch.bool),
        "action": torch.rand((batch_size, 5), generator=generator),
        "prediction_steps": torch.ones(batch_size, dtype=torch.long),
        "provenance": [{} for _ in range(batch_size)],
    }


# ---------------------------------------------------------------------------
# Reproducibility manifest
# ---------------------------------------------------------------------------


class RunManifestTests(unittest.TestCase):
    def test_the_manifest_is_frozen(self) -> None:
        manifest = make_manifest()
        with self.assertRaises(AttributeError):
            manifest.seed = 1  # type: ignore[misc]

    def test_to_dict_round_trips_through_strict_json(self) -> None:
        payload = make_manifest().to_dict()
        encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
        self.assertEqual(json.loads(encoded), dict(payload))

    def test_the_payload_carries_the_digest(self) -> None:
        manifest = make_manifest()
        self.assertEqual(manifest.to_dict()["digest"], manifest.digest)

    def test_identical_configurations_share_a_digest(self) -> None:
        self.assertEqual(make_manifest().digest, make_manifest().digest)

    def test_the_digest_excludes_wall_clock_timing(self) -> None:
        # Two seeded runs must be comparable by digest even though they never
        # take the same amount of time.
        baseline = make_manifest()
        slower = make_manifest(wall_clock_seconds=305.75, started_at_unix=1786099999.0)
        self.assertEqual(baseline.digest, slower.digest)
        self.assertNotEqual(baseline.to_dict()["wall_clock_seconds"],
                            slower.to_dict()["wall_clock_seconds"])

    def test_the_digest_excludes_the_measured_metrics(self) -> None:
        # CUDA float reductions are not bitwise reproducible across processes,
        # so two runs of the same experiment differ in the ~5th significant
        # digit.  A digest over outcomes could never match, which would defeat
        # its purpose.  Metrics are compared numerically instead.
        baseline = make_manifest()
        jittered = make_manifest(
            final_loss=3.9342e-08,
            mean_feature_std=0.400001,
            relative_spread=0.350002,
            effective_rank=6.200003,
        )
        self.assertEqual(baseline.digest, jittered.digest)
        self.assertNotEqual(
            baseline.to_dict()["final_loss"], jittered.to_dict()["final_loss"]
        )

    def test_the_digest_excludes_the_timestamped_run_id(self) -> None:
        self.assertEqual(
            make_manifest().digest, make_manifest(run_id="20260807T999999Z-x").digest
        )

    def test_the_digest_discriminates_the_seed(self) -> None:
        self.assertNotEqual(make_manifest().digest, make_manifest(seed=1).digest)

    def test_the_digest_discriminates_the_catalog(self) -> None:
        self.assertNotEqual(make_manifest().digest, make_manifest(catalog_digest="d" * 64).digest)

    def test_the_digest_discriminates_the_model_configuration(self) -> None:
        self.assertNotEqual(
            make_manifest().digest, make_manifest(model_config_digest="e" * 64).digest
        )

    def test_the_manifest_rejects_an_empty_catalog_digest(self) -> None:
        with self.assertRaises(ContractValueError):
            make_manifest(catalog_digest="  ")

    def test_the_manifest_rejects_a_nonpositive_window_count(self) -> None:
        with self.assertRaises(ContractValueError):
            make_manifest(window_count=0)

    def test_the_manifest_rejects_a_nonpositive_step_count(self) -> None:
        with self.assertRaises(ContractValueError):
            make_manifest(steps=0)

    def test_the_manifest_rejects_a_non_finite_loss(self) -> None:
        with self.assertRaises(ContractValueError):
            make_manifest(final_loss=float("nan"))

    def test_the_manifest_rejects_an_unknown_acceptance_verdict(self) -> None:
        with self.assertRaises(ContractValueError):
            make_manifest(acceptance="probably")

    def test_the_manifest_records_that_symbolic_supervision_is_inactive(self) -> None:
        # The legacy RGB cohort carries no symbolic labels; a run that claims
        # otherwise would misreport what the loss actually optimized.
        self.assertIs(make_manifest().to_dict()["symbolic_loss_active"], False)


# ---------------------------------------------------------------------------
# Anti-collapse diagnostics
# ---------------------------------------------------------------------------


class CollapseDiagnosticsTests(unittest.TestCase):
    def test_a_constant_matrix_has_no_centred_variation(self) -> None:
        # Collapse means the samples do not vary.  A constant matrix has zero
        # variation regardless of how large its shared component is.
        constant = torch.ones(8, _LATENT_DIM)
        self.assertAlmostEqual(effective_rank(constant), 0.0, places=5)
        self.assertAlmostEqual(relative_spread(constant), 0.0, places=6)

    def test_an_uncentred_constant_matrix_is_rank_one(self) -> None:
        # The raw spectrum is available for inspection; it is not the collapse
        # metric, because natural frames share a dominant background component.
        constant = torch.ones(8, _LATENT_DIM)
        self.assertAlmostEqual(effective_rank(constant, centred=False), 1.0, places=5)

    def test_an_orthogonal_matrix_has_near_full_effective_rank(self) -> None:
        # Centring an 8-row orthogonal matrix removes one degree of freedom, so
        # the ceiling is 7, not 8.  float32 svdvals lands a hair under it.
        orthogonal = torch.eye(8, _LATENT_DIM)
        self.assertAlmostEqual(effective_rank(orthogonal), 7.0, places=4)

    def test_an_all_zero_matrix_has_zero_effective_rank(self) -> None:
        self.assertEqual(effective_rank(torch.zeros(8, _LATENT_DIM)), 0.0)

    def test_a_shared_background_does_not_look_like_collapse(self) -> None:
        # This is the NovPhy case: a large common component plus small
        # per-sample variation.  Uncentred rank would report ~1; the centred
        # metric must see the real variation.
        torch.manual_seed(0)
        background = torch.ones(1, _LATENT_DIM) * 10.0
        varied = background + torch.randn(8, _LATENT_DIM) * 0.01
        self.assertLess(effective_rank(varied, centred=False), 1.1)
        self.assertGreater(effective_rank(varied), 4.0)

    def test_relative_spread_is_scale_invariant(self) -> None:
        torch.manual_seed(0)
        latents = torch.randn(8, _LATENT_DIM)
        self.assertAlmostEqual(
            relative_spread(latents), relative_spread(latents * 1000.0), places=5
        )

    def test_a_constant_matrix_has_zero_feature_spread(self) -> None:
        self.assertAlmostEqual(mean_feature_std(torch.ones(8, _LATENT_DIM)), 0.0, places=6)

    def test_a_varied_matrix_has_positive_feature_spread(self) -> None:
        torch.manual_seed(0)
        self.assertGreater(mean_feature_std(torch.randn(8, _LATENT_DIM)), 0.5)

    def test_exact_predictions_retrieve_their_own_target(self) -> None:
        torch.manual_seed(0)
        targets = torch.randn(8, _LATENT_DIM)
        self.assertEqual(retrieval_accuracy(targets.clone(), targets), 1.0)

    def test_a_collapsed_prediction_cannot_retrieve_its_target(self) -> None:
        torch.manual_seed(0)
        targets = torch.randn(8, _LATENT_DIM)
        collapsed = torch.zeros_like(targets)
        self.assertLess(retrieval_accuracy(collapsed, targets), 1.0)

    def test_diagnostics_bundle_every_criterion(self) -> None:
        torch.manual_seed(0)
        targets = torch.randn(8, _LATENT_DIM)
        report = collapse_diagnostics(targets.clone(), targets)
        self.assertEqual(report.retrieval_accuracy, 1.0)
        self.assertGreater(report.effective_rank, 1.0)
        self.assertGreater(report.mean_feature_std, 0.0)
        self.assertGreater(report.relative_spread, 0.0)
        payload = report.to_dict()
        json.dumps(payload, allow_nan=False)

    def test_diagnostics_reject_a_single_row(self) -> None:
        with self.assertRaises(ContractValueError):
            collapse_diagnostics(torch.randn(1, _LATENT_DIM), torch.randn(1, _LATENT_DIM))

    def test_diagnostics_reject_mismatched_shapes(self) -> None:
        with self.assertRaises(ContractValueError):
            collapse_diagnostics(torch.randn(4, _LATENT_DIM), torch.randn(4, _LATENT_DIM + 1))


# ---------------------------------------------------------------------------
# Teacher-forced trainer
# ---------------------------------------------------------------------------


class TeacherForcedTrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.config = make_training_config()
        self.backbone = JepaBackbone(small_jepa_config())
        self.trainer = TeacherForcedTrainer(self.backbone, self.config)
        self.batch = make_fixture_batch(1)

    def test_a_step_reports_its_loss_learning_rate_and_momentum(self) -> None:
        result = self.trainer.train_step(self.batch)
        self.assertEqual(result.step, 0)
        self.assertTrue(math.isfinite(result.loss))
        self.assertGreater(result.learning_rate, 0.0)
        self.assertGreaterEqual(result.momentum, self.config.ema_base_momentum)

    def test_repeated_steps_on_one_batch_reduce_the_loss(self) -> None:
        losses = [self.trainer.train_step(self.batch).loss for _ in range(20)]
        self.assertLess(losses[-1], losses[0])

    def test_the_target_branch_receives_no_gradient(self) -> None:
        self.trainer.train_step(self.batch)
        for parameter in self.backbone.target.parameters():
            self.assertIsNone(parameter.grad)

    def test_the_online_branch_receives_a_gradient(self) -> None:
        self.trainer.train_step(self.batch)
        gradients = [
            parameter.grad
            for parameter in self.backbone.encoder.parameters()
            if parameter.grad is not None
        ]
        self.assertNotEqual(gradients, [])
        self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0.0)

    def test_the_teacher_forced_target_is_detached(self) -> None:
        target = self.trainer.encode_target(self.batch)
        self.assertFalse(target.requires_grad)
        self.assertEqual(target.shape, torch.Size([4, _LATENT_DIM]))

    def test_the_step_uses_the_configured_prediction_pair(self) -> None:
        self.assertEqual(
            self.trainer.pair,
            PredictionPair(delta=self.config.delta, abstraction=self.config.abstraction),
        )

    def test_two_seeded_trainers_produce_identical_loss_sequences(self) -> None:
        def loss_sequence() -> list[float]:
            torch.manual_seed(0)
            backbone = JepaBackbone(small_jepa_config())
            trainer = TeacherForcedTrainer(backbone, make_training_config())
            batch = make_fixture_batch(1)
            return [trainer.train_step(batch).loss for _ in range(10)]

        self.assertEqual(loss_sequence(), loss_sequence())

    def test_the_trainer_rejects_a_multi_step_target_stack(self) -> None:
        batch = make_fixture_batch(1)
        batch["target_images"] = batch["target_images"].repeat(1, 2, 1, 1, 1)
        batch["target_mask"] = torch.ones((4, 2), dtype=torch.bool)
        with self.assertRaises(ContractValueError):
            self.trainer.train_step(batch)

    def test_the_training_config_rejects_a_nonpositive_step_count(self) -> None:
        with self.assertRaises(ContractValueError):
            make_training_config(steps=0)

    def test_the_training_config_rejects_a_momentum_outside_the_unit_interval(self) -> None:
        with self.assertRaises(ContractValueError):
            make_training_config(ema_base_momentum=1.5)

    def test_the_training_config_rejects_a_warmup_longer_than_the_run(self) -> None:
        with self.assertRaises(ContractValueError):
            make_training_config(steps=10, warmup_steps=20)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_a_step_moves_every_batch_tensor_onto_the_device(self) -> None:
        # A CPU-only suite cannot catch a missing .to(device): the move is a
        # no-op there.  This is the regression guard for exactly that bug.
        torch.manual_seed(0)
        backbone = JepaBackbone(small_jepa_config())
        trainer = TeacherForcedTrainer(backbone, make_training_config(device="cuda"))
        result = trainer.train_step(make_fixture_batch(1))
        self.assertTrue(math.isfinite(result.loss))
        self.assertEqual(trainer.encode_target(make_fixture_batch(1)).device.type, "cuda")


# ---------------------------------------------------------------------------
# End-to-end wiring over the real lazy reader
# ---------------------------------------------------------------------------


class WindowLoaderIntegrationTests(unittest.TestCase):
    def test_the_resize_transform_reshapes_a_single_frame(self) -> None:
        transform = resize_transform(32, 40)
        resized = transform(torch.rand(3, 480, 640))
        self.assertEqual(resized.shape, torch.Size([3, 32, 40]))
        self.assertEqual(resized.dtype, torch.float32)

    def test_the_resize_transform_rejects_a_batched_input(self) -> None:
        with self.assertRaises(ContractValueError):
            resize_transform(32, 40)(torch.rand(2, 3, 480, 640))

    def test_the_loader_yields_batches_shaped_for_the_encoder(self) -> None:
        jepa_config = small_jepa_config()
        with tempfile.TemporaryDirectory() as temporary:
            catalog = make_fixture_catalog(Path(temporary), episode_count=2, frame_count=5)
            self.assertGreater(len(catalog.episodes), 0)
            loader, window_count, index_digest = build_window_loader(
                catalog,
                encoder_config=jepa_config.encoder,
                delta=1,
                batch_size=2,
                seed=3,
                draw_count=4,
            )
            self.assertGreater(window_count, 0)
            self.assertEqual(len(index_digest), 64)
            batch = next(iter(loader))
            self.assertEqual(
                batch["context_image"].shape,
                torch.Size([2, 3, jepa_config.encoder.input_height,
                            jepa_config.encoder.input_width]),
            )
            self.assertEqual(batch["target_images"].shape[1], 1)
            self.assertEqual(batch["action"].shape, torch.Size([2, 5]))

    def test_the_loader_rejects_an_unknown_selection_mode(self) -> None:
        jepa_config = small_jepa_config()
        with tempfile.TemporaryDirectory() as temporary:
            catalog = make_fixture_catalog(Path(temporary), episode_count=2, frame_count=5)
            with self.assertRaises(ContractValueError):
                build_window_loader(
                    catalog,
                    encoder_config=jepa_config.encoder,
                    delta=1,
                    batch_size=2,
                    seed=3,
                    draw_count=4,
                    window_selection="whatever",
                )

    def test_motion_selection_prefers_the_most_dynamic_windows(self) -> None:
        # The legacy cohort is action-sparse, so a uniform draw is almost
        # always near-identical frame pairs.  Motion ranking must actually
        # order by inter-frame change, or the overfit evidence is vacuous.
        jepa_config = small_jepa_config()
        with tempfile.TemporaryDirectory() as temporary:
            catalog = make_fixture_catalog(Path(temporary), episode_count=3, frame_count=8)
            dataset = TemporalWindowDataset(
                catalog,
                TemporalWindowRequest(prediction_steps=1, stride_frames=1),
                transform=resize_transform(
                    jepa_config.encoder.input_height, jepa_config.encoder.input_width
                ),
            )
            chosen = select_motion_windows(
                dataset, seed=5, window_count=3, candidate_count=12
            )
            self.assertEqual(len(chosen), 3)
            self.assertEqual(len(set(chosen)), 3)
            motions = [
                float(
                    (dataset[i]["context_image"] - dataset[i]["target_images"][0])
                    .pow(2)
                    .mean()
                )
                for i in chosen
            ]
            self.assertEqual(motions, sorted(motions, reverse=True))

    def test_motion_selection_is_deterministic_for_a_seed(self) -> None:
        jepa_config = small_jepa_config()
        with tempfile.TemporaryDirectory() as temporary:
            catalog = make_fixture_catalog(Path(temporary), episode_count=3, frame_count=8)
            dataset = TemporalWindowDataset(
                catalog,
                TemporalWindowRequest(prediction_steps=1, stride_frames=1),
                transform=resize_transform(
                    jepa_config.encoder.input_height, jepa_config.encoder.input_width
                ),
            )
            first = select_motion_windows(dataset, seed=5, window_count=4, candidate_count=12)
            second = select_motion_windows(dataset, seed=5, window_count=4, candidate_count=12)
            self.assertEqual(first, second)

    def test_motion_selection_rejects_a_candidate_pool_smaller_than_the_subset(self) -> None:
        jepa_config = small_jepa_config()
        with tempfile.TemporaryDirectory() as temporary:
            catalog = make_fixture_catalog(Path(temporary), episode_count=2, frame_count=5)
            dataset = TemporalWindowDataset(
                catalog,
                TemporalWindowRequest(prediction_steps=1, stride_frames=1),
                transform=resize_transform(
                    jepa_config.encoder.input_height, jepa_config.encoder.input_width
                ),
            )
            with self.assertRaises(ContractValueError):
                select_motion_windows(dataset, seed=5, window_count=8, candidate_count=4)

    def test_the_loader_yields_exactly_the_selected_windows(self) -> None:
        # Regression: wrapping the dataset in a Subset and sampling *that*
        # silently yields indices 0..N-1 of the original dataset, so the
        # carefully chosen order never reaches the batch.  Shape-only assertions
        # do not catch it; identity assertions do.
        jepa_config = small_jepa_config()
        with tempfile.TemporaryDirectory() as temporary:
            catalog = make_fixture_catalog(Path(temporary), episode_count=6, frame_count=6)
            dataset = TemporalWindowDataset(
                catalog,
                TemporalWindowRequest(prediction_steps=1, stride_frames=1),
                transform=resize_transform(
                    jepa_config.encoder.input_height, jepa_config.encoder.input_width
                ),
            )
            expected = select_diverse_windows(
                dataset, seed=7, window_count=4, candidate_count=64
            )
            loader, _, _ = build_window_loader(
                catalog,
                encoder_config=jepa_config.encoder,
                delta=1,
                batch_size=4,
                seed=7,
                draw_count=4,
                window_selection="diverse",
                candidate_count=64,
            )
            batch = next(iter(loader))
            delivered = [
                (p["episode"], p["shot"]) for p in batch["provenance"]
            ]
            wanted = [
                (dataset[i]["provenance"]["episode"], dataset[i]["provenance"]["shot"])
                for i in expected
            ]
            self.assertEqual(delivered, wanted)
            self.assertEqual(len({episode for episode, _ in delivered}), 4)

    def test_diverse_selection_takes_one_window_per_episode(self) -> None:
        # The measured reason: uniformly drawn dev windows are frequently
        # near-duplicates whose target embeddings land 7.4e-05 apart, which
        # makes retrieval a coin flip regardless of predictor quality.
        jepa_config = small_jepa_config()
        with tempfile.TemporaryDirectory() as temporary:
            catalog = make_fixture_catalog(Path(temporary), episode_count=5, frame_count=6)
            dataset = TemporalWindowDataset(
                catalog,
                TemporalWindowRequest(prediction_steps=1, stride_frames=1),
                transform=resize_transform(
                    jepa_config.encoder.input_height, jepa_config.encoder.input_width
                ),
            )
            chosen = select_diverse_windows(
                dataset, seed=5, window_count=4, candidate_count=64
            )
            episodes = [
                dataset[index]["provenance"]["episode"] for index in chosen
            ]
            self.assertEqual(len(chosen), 4)
            self.assertEqual(len(set(episodes)), 4)

    def test_diverse_selection_is_deterministic_for_a_seed(self) -> None:
        jepa_config = small_jepa_config()
        with tempfile.TemporaryDirectory() as temporary:
            catalog = make_fixture_catalog(Path(temporary), episode_count=5, frame_count=6)
            dataset = TemporalWindowDataset(
                catalog,
                TemporalWindowRequest(prediction_steps=1, stride_frames=1),
                transform=resize_transform(
                    jepa_config.encoder.input_height, jepa_config.encoder.input_width
                ),
            )
            first = select_diverse_windows(dataset, seed=5, window_count=3, candidate_count=64)
            second = select_diverse_windows(dataset, seed=5, window_count=3, candidate_count=64)
            self.assertEqual(first, second)

    def test_diverse_selection_fails_closed_without_enough_episodes(self) -> None:
        # Silently returning a short subset would quietly weaken the evidence.
        jepa_config = small_jepa_config()
        with tempfile.TemporaryDirectory() as temporary:
            catalog = make_fixture_catalog(Path(temporary), episode_count=2, frame_count=6)
            dataset = TemporalWindowDataset(
                catalog,
                TemporalWindowRequest(prediction_steps=1, stride_frames=1),
                transform=resize_transform(
                    jepa_config.encoder.input_height, jepa_config.encoder.input_width
                ),
            )
            with self.assertRaises(ContractValueError):
                select_diverse_windows(dataset, seed=5, window_count=5, candidate_count=64)

    def test_the_manifest_records_the_window_selection_mode(self) -> None:
        # Two runs that select their windows differently are different
        # experiments, so the digest must separate them.
        uniform = make_manifest(window_selection="uniform")
        motion = make_manifest(window_selection="motion")
        self.assertNotEqual(uniform.digest, motion.digest)
        self.assertEqual(motion.to_dict()["window_selection"], "motion")

    def test_an_overfit_run_trains_against_real_cataloged_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cohort"
            catalog = make_fixture_catalog(root, episode_count=6, frame_count=6)
            report = run_overfit(
                catalog,
                jepa_config=small_jepa_config(),
                training_config=make_training_config(steps=30, batch_size=4),
                window_count=4,
                output_dir=Path(temporary) / "runs",
            )
            self.assertLess(report.final_loss, report.initial_loss)
            self.assertEqual(len(report.loss_history), 30)
            manifest_path = report.run_dir / "manifest.json"
            self.assertTrue(manifest_path.is_file())
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["digest"], report.manifest.digest)
            self.assertIs(payload["symbolic_loss_active"], False)
            self.assertIn(payload["acceptance"], ("pass", "fail"))

    def test_two_seeded_overfit_runs_share_a_manifest_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cohort"
            catalog = make_fixture_catalog(root, episode_count=6, frame_count=6)
            digests = []
            for index in range(2):
                report = run_overfit(
                    catalog,
                    jepa_config=small_jepa_config(),
                    training_config=make_training_config(steps=15, batch_size=4),
                    window_count=4,
                    output_dir=Path(temporary) / f"runs{index}",
                )
                digests.append(report.manifest.digest)
            self.assertEqual(digests[0], digests[1])


if __name__ == "__main__":
    unittest.main()
