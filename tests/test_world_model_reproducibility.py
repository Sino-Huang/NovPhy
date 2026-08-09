from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from world_model.model import JepaBackbone
from world_model.training.grid_run import (
    CheckpointInfo,
    GridRunError,
    PhaseAConfig,
    ScoreResult,
    fixture_jepa_config,
    load_checkpoint,
    save_checkpoint,
    write_sweep_manifest,
)
from world_model.training.loop import TeacherForcedTrainer, seed_all
from world_model.training.reproducibility import ReproducibilityConfig

_PRECONFIGURED_SUBPROCESS = "NOVPHY_REPRODUCIBILITY_SUBPROCESS"


class ReproducibilityTests(unittest.TestCase):
    def run_preconfigured_subprocess(self) -> None:
        environment = os.environ.copy()
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        environment[_PRECONFIGURED_SUBPROCESS] = self._testMethodName
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                f"tests.test_world_model_reproducibility.{type(self).__name__}.{self._testMethodName}",
            ],
            cwd=Path(__file__).parent.parent,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_settings_change_phase_identity_and_are_written_to_manifest(self) -> None:
        # Given
        strict = PhaseAConfig(steps=1, batch_size=2, device="cpu")
        tf32 = replace(
            strict,
            reproducibility=replace(strict.reproducibility, cudnn_allow_tf32=True),
        )
        checkpoint = CheckpointInfo(Path("checkpoint.pt"), "a" * 64, 1, strict.identity)
        score = ScoreResult(1, 2, 0.25, strict.identity)

        # When
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "sweep.json"
            write_sweep_manifest(
                manifest,
                checkpoint=checkpoint,
                phase_config=strict,
                score=score,
            )
            payload = json.loads(manifest.read_text(encoding="ascii"))

        # Then
        self.assertNotEqual(strict.identity, tf32.identity)
        self.assertEqual(payload["reproducibility"], strict.reproducibility.canonical)

    def test_seed_all_applies_runtime_settings_before_torch_seed(self) -> None:
        # Given
        events: list[str] = []
        settings = ReproducibilityConfig()

        # When
        with (
            patch(
                "world_model.training.loop.apply_reproducibility",
                side_effect=lambda _settings: events.append("settings"),
            ),
            patch("world_model.training.loop.torch.manual_seed", side_effect=lambda _seed: events.append("seed")),
            patch("world_model.training.loop.torch.cuda.manual_seed_all"),
        ):
            seed_all(20260807, reproducibility=settings)

        # Then
        self.assertEqual(events, ["settings", "seed"])

    def test_apply_reproducibility_sets_the_complete_strict_policy(self) -> None:
        # Given
        if os.environ.get(_PRECONFIGURED_SUBPROCESS) != self._testMethodName:
            self.run_preconfigured_subprocess()
            return
        settings = ReproducibilityConfig()

        # When
        from world_model.training.reproducibility import apply_reproducibility

        apply_reproducibility(settings)

        # Then
        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_rng_round_trip_restores_active_device_state(self) -> None:
        # Given
        if os.environ.get(_PRECONFIGURED_SUBPROCESS) != self._testMethodName:
            self.run_preconfigured_subprocess()
            return
        settings = ReproducibilityConfig()
        phase = PhaseAConfig(steps=1, batch_size=2, device="cuda", reproducibility=settings)
        seed_all(phase.seed, reproducibility=settings)
        trainer = TeacherForcedTrainer(JepaBackbone(fixture_jepa_config()), phase.training_config())
        expected = torch.cuda.get_rng_state(trainer.device).clone()

        # When
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            save_checkpoint(
                checkpoint,
                trainer,
                config_digest=phase.identity,
                grid_digest=phase.grid_digest,
            )
            torch.cuda.manual_seed(phase.seed + 1)
            self.assertFalse(torch.equal(torch.cuda.get_rng_state(trainer.device), expected))
            restored = TeacherForcedTrainer(
                JepaBackbone(fixture_jepa_config()),
                phase.training_config(),
            )
            loaded = load_checkpoint(
                checkpoint,
                restored,
                config_digest=phase.identity,
                grid_digest=phase.grid_digest,
            )

        # Then
        self.assertTrue(loaded.cuda_rng_restored)
        self.assertTrue(torch.equal(torch.cuda.get_rng_state(restored.device), expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_resume_rejects_checkpoint_without_cuda_rng(self) -> None:
        # Given
        cpu_phase = PhaseAConfig(steps=1, batch_size=2, device="cpu")
        cpu_trainer = TeacherForcedTrainer(
            JepaBackbone(fixture_jepa_config()),
            cpu_phase.training_config(),
        )

        # When / Then
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "cpu-checkpoint.pt"
            save_checkpoint(
                checkpoint,
                cpu_trainer,
                config_digest=cpu_phase.identity,
                grid_digest=cpu_phase.grid_digest,
            )
            cuda_trainer = TeacherForcedTrainer(
                JepaBackbone(fixture_jepa_config()),
                replace(cpu_phase.training_config(), device="cuda"),
            )
            with self.assertRaisesRegex(GridRunError, "no CUDA RNG state"):
                load_checkpoint(
                    checkpoint,
                    cuda_trainer,
                    config_digest=cpu_phase.identity,
                    grid_digest=cpu_phase.grid_digest,
                )


if __name__ == "__main__":
    unittest.main()
