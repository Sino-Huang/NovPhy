from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from world_model.data import CohortV2Rollout
from world_model.data.cohort_v2 import (
    CAPABILITY_DECLARATION_IDENTITY,
    COHORT_V2_RELEASE_IDENTITY,
)
from world_model.model import Abstraction, PredictionPair
from world_model.training.cohort_v2_calibration import CohortV2CalibrationRecord
from world_model.training.cohort_v2_integrated import (
    CohortV2RecursiveRolloutRecord,
    IntegratedVariant,
    analyze_integrated_calibration,
    build_integrated_predictor,
    integrated_compute_calibration,
    load_cohort_v2_integrated_checkpoint,
    recursive_continuous_rollouts,
    save_cohort_v2_integrated_checkpoint,
    validate_integrated_evidence,
    write_integrated_evidence,
)
from world_model.training.cohort_v2_macro import (
    MACRO_PAIRS,
    CohortV2MacroConfig,
    CohortV2MacroTrainer,
)
from world_model.training.cohort_v2_micro import CohortV2StateCodec

from tests.test_cohort_v2_macro_training import _frame, _window


def _reader(role: str, attempt: str):
    frames = tuple(
        _frame(
            f"{attempt}:frame:{position}",
            position,
            steady=position == 6,
            unstable=False,
            terminal_reason="stable_entered" if position == 6 else None,
        )
        for position in range(7)
    )
    rollout = CohortV2Rollout(
        attempt_id=attempt,
        exposure_role=role,
        coverage_stratum="collision",
        scenario_lineage_identity=f"lineage:{attempt}",
        intervention={
            "engine_relative_action": {
                "drag_delta_canvas_pixels": (12, 3),
                "hold_milliseconds": 1000,
                "tap_time_milliseconds": 0,
            }
        },
        agent_observation_identity=f"observation:{attempt}",
        agent_observation_fixed_step=0,
        frame_records=frames,
    )

    class Reader:
        release_identity = COHORT_V2_RELEASE_IDENTITY
        partition_identity = "partition:fixture"
        capability_declaration_identity = CAPABILITY_DECLARATION_IDENTITY
        rollouts = (rollout,)

        @staticmethod
        def load_observation(item, *, observation_role: str):
            assert item is rollout and observation_role == "agent"
            return b"fixture"

    return Reader()


def _config() -> CohortV2MacroConfig:
    return CohortV2MacroConfig(
        steps=9,
        batch_size=1,
        latent_dim=96,
        hidden_dim=16,
        depth=1,
        max_entities=7,
        device="cpu",
    )


def _recursive(role: str, horizon: int, error: float, attempt_index: int = 0):
    return CohortV2RecursiveRolloutRecord(
        checkpoint_identity="checkpoint:candidate",
        exposure_role=role,
        attempt_id=f"attempt:{role}:{attempt_index}",
        scenario_lineage_identity=f"lineage:{role}:{attempt_index}",
        coverage_stratum="collision",
        requested_horizon=horizon,
        simulated_duration=2,
        effective_horizons=(min(horizon, 2),) if horizon > 1 else (1, 1),
        cumulative_horizons=(2,) if horizon > 1 else (1, 2),
        authoritative_endpoint_identities=("frame:2",) if horizon > 1 else ("frame:1", "frame:2"),
        endpoint_mse_curve=(error,) if horizon > 1 else (error, error),
        terminal_mse=error,
        error_auc=error * 2,
        total_compute=100.0,
    )


class CohortV2IntegratedTests(unittest.TestCase):
    def test_variants_are_matched_capacity_and_freeze_gate_roles(self):
        config = _config()
        predictors = {
            variant: build_integrated_predictor(config, variant)
            for variant in IntegratedVariant
        }
        counts = {
            sum(parameter.numel() for parameter in predictor.parameters())
            for predictor in predictors.values()
        }

        self.assertEqual(len(counts), 1)
        self.assertTrue(IntegratedVariant.CANDIDATE.reliability_gate_enabled)
        self.assertFalse(IntegratedVariant.NO_SYMBOL.reliability_gate_enabled)
        self.assertFalse(IntegratedVariant.UNGATED.reliability_gate_enabled)
        self.assertEqual(
            str(IntegratedVariant.CANDIDATE.interface), "ordered_flat_predicate"
        )

    def test_macro_trainer_uses_reliability_only_for_micro_symbolic_loss(self):
        config = CohortV2MacroConfig(
            steps=9,
            batch_size=2,
            latent_dim=32,
            hidden_dim=16,
            depth=1,
            max_entities=2,
            device="cpu",
        )
        pairs = (
            PredictionPair(1, Abstraction.MICRO),
            PredictionPair(1, Abstraction.CONTINUOUS),
            PredictionPair(1, Abstraction.MACRO),
        )
        window = _window(requested_horizon=1, effective_horizon=1)

        class Data:
            reader = SimpleNamespace()

            @staticmethod
            def schedule_at(step):
                return pairs[step]

            @staticmethod
            def batch_at(pair, step):
                return (window, window)

            @staticmethod
            def duration_weights(windows):
                return torch.ones(len(windows))

        calls = []

        def gate(windows):
            calls.append(windows)
            return torch.tensor((0.25, 0.75))

        trainer = CohortV2MacroTrainer(Data(), config, symbolic_gate=gate)
        for _ in pairs:
            trainer.train_step()

        self.assertEqual(len(calls), 1)
        self.assertEqual(trainer.pair_counts[pairs[0]], 1)

    def test_checkpoint_binds_variant_reliability_and_exact_pair_schedule(self):
        config = _config()
        reader = _reader("training", "attempt:training")
        predictor = build_integrated_predictor(config, IntegratedVariant.CANDIDATE)
        trainer = SimpleNamespace(
            config=config,
            data=SimpleNamespace(reader=reader),
            codec=CohortV2StateCodec(latent_dim=96, max_entities=7),
            predictor=predictor,
            pair_counts={pair: 1 for pair in MACRO_PAIRS},
            step_count=9,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            saved = save_cohort_v2_integrated_checkpoint(
                path,
                trainer,
                IntegratedVariant.CANDIDATE,
                reliability_artifact_identity="reliability:fixture",
            )
            loaded, _, receipt = load_cohort_v2_integrated_checkpoint(
                path,
                reader=reader,
                config=config,
                variant=IntegratedVariant.CANDIDATE,
                reliability_artifact_identity="reliability:fixture",
                device="cpu",
            )

            self.assertEqual(receipt.identity, saved.identity)
            self.assertEqual(
                sum(parameter.numel() for parameter in loaded.parameters()),
                sum(parameter.numel() for parameter in predictor.parameters()),
            )
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_cohort_v2_integrated_checkpoint(
                    path,
                    reader=reader,
                    config=config,
                    variant=IntegratedVariant.UNGATED,
                    reliability_artifact_identity="reliability:fixture",
                    device="cpu",
                )

    def test_recursive_rollout_feeds_only_predicted_successors(self):
        class AddOne(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = torch.nn.Parameter(torch.zeros(()))
                self.inputs = []

            def carrier(self, latent, action, pair):
                self.inputs.append(latent.detach().clone())
                return latent + 1.0 + self.marker * 0.0

        predictor = AddOne()
        codec = CohortV2StateCodec(latent_dim=96, max_entities=7)
        records = recursive_continuous_rollouts(
            predictor,
            codec,
            "checkpoint:fixture",
            (_reader("calibration", "attempt:calibration"),),
            integrated_compute_calibration(_config(), IntegratedVariant.CANDIDATE),
        )

        h1 = next(item for item in records if item.requested_horizon == 1)
        self.assertEqual(h1.effective_horizons, (1, 1, 1, 1, 1, 1))
        self.assertEqual(h1.cumulative_horizons[-1], 6)
        self.assertEqual(h1.recursive_physical_violation_status, "unavailable")
        self.assertTrue(torch.allclose(predictor.inputs[1], predictor.inputs[0] + 1.0))
        self.assertTrue(all(identity.startswith("attempt:calibration:frame:") for identity in h1.authoritative_endpoint_identities))

    def test_budget_comparators_use_model_selection_and_report_metrics_separately(self):
        records = []
        configurations = ("candidate", "fixed", "temporal", "description", "axes", "two_head")
        for role in ("calibration", "model_selection"):
            for attempt_index in range(6):
                for index, configuration in enumerate(configurations):
                    error = 0.2 + index * 0.01 + attempt_index * 0.001
                    if role == "model_selection" and configuration == "fixed":
                        error = 0.01
                    records.append(CohortV2CalibrationRecord(
                        configuration_id=configuration,
                        exposure_role=role,
                        attempt_id=f"attempt:{role}:{attempt_index}",
                        scenario_lineage_identity=f"lineage:{role}:{attempt_index}",
                        coverage_stratum="collision",
                        checkpoint_identity=f"checkpoint:{configuration}",
                        seed=10,
                        state_count=2,
                        mean_endpoint_prediction_error=error,
                        mean_endpoint_violation_rate=0.0,
                        mean_policy_compute_per_simulated_frame=100.0 + index,
                        mean_full_compute_per_simulated_frame=100.0 + index,
                    ))
        recursive = tuple(
            _recursive(role, horizon, 0.1 + horizon / 1000, attempt_index)
            for role in ("calibration", "model_selection")
            for horizon in (1, 5, 15)
            for attempt_index in range(6)
        )
        report = analyze_integrated_calibration(
            tuple(records),
            recursive,
            candidate_configuration_id="candidate",
            comparator_configuration_ids=("fixed", "temporal", "description", "axes", "two_head"),
            source_bindings={"fixture": True},
        )

        comparable = tuple(
            item for item in report["local_teacher_forced_metrics"]["budget_comparisons"]
            if item["status"] == "comparable"
        )
        self.assertTrue(comparable)
        self.assertTrue(all(item["strongest_comparator_id"] == "fixed" for item in comparable))
        self.assertIn("complete_rollout_metrics", report)
        self.assertEqual(
            report["complete_rollout_metrics"]["recursive_physical_violation_status"],
            "unavailable",
        )

    def test_integrated_evidence_validates_by_exact_recomputation(self):
        records = []
        configurations = ("candidate", "fixed", "temporal", "description", "axes", "two_head")
        for role in ("calibration", "model_selection"):
            for attempt_index in range(6):
                for index, configuration in enumerate(configurations):
                    records.append(CohortV2CalibrationRecord(
                        configuration, role, f"attempt:{role}:{attempt_index}",
                        f"lineage:{role}:{attempt_index}", "collision",
                        f"checkpoint:{configuration}", 10, 1,
                        0.1 + index * 0.01 + attempt_index * 0.001,
                        0.0, 100.0 + index, 100.0 + index,
                    ))
        recursive = tuple(
            _recursive(role, horizon, 0.1 + horizon / 1000, attempt_index)
            for role in ("calibration", "model_selection")
            for horizon in (1, 5, 15)
            for attempt_index in range(6)
        )
        report = analyze_integrated_calibration(
            tuple(records),
            recursive,
            candidate_configuration_id="candidate",
            comparator_configuration_ids=("fixed", "temporal", "description", "axes", "two_head"),
            source_bindings={"fixture": True},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_integrated_evidence(
                root,
                tuple(records),
                recursive,
                report,
                implementation_revision="commit:fixture",
            )
            self.assertEqual(
                validate_integrated_evidence(root)["artifact_identity"],
                manifest["artifact_identity"],
            )


if __name__ == "__main__":
    unittest.main()
