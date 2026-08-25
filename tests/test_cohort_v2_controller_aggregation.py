from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from tests.test_cohort_v2_controller import _inputs, _png
from tests.test_cohort_v2_macro_training import _frame
from world_model.model import DualOutputPredictor
from world_model.training import (
    CohortV2AggregationConfig,
    CohortV2ControllerConfig,
    CohortV2MacroConfig,
    CohortV2StateCodec,
    load_cohort_v2_aggregated_controllers,
    run_cohort_v2_controller_aggregation,
    train_cohort_v2_controllers,
    validate_cohort_v2_controller_aggregation,
    write_cohort_v2_controller_aggregation,
)
from world_model.training.cohort_v2_controller import build_cohort_v2_controller_examples


class CohortV2ControllerAggregationTests(unittest.TestCase):
    def _run(self):
        readers, evaluation, measurement, labels, spec = _inputs()
        for reader in readers:
            rollout = reader.rollouts[0]
            frames = tuple(
                replace(
                    frame,
                    engine_state=_frame(
                        frame.identity,
                        index,
                        steady=False,
                        unstable=False,
                    ).engine_state,
                )
                for index, frame in enumerate(rollout.frame_records)
            )
            intervention = dict(rollout.intervention)
            intervention["engine_relative_action"] = {
                "drag_delta_canvas_pixels": (12, 3),
                "hold_milliseconds": 1000,
                "tap_time_milliseconds": 0,
            }
            reader.rollouts = (
                replace(
                    rollout,
                    frame_records=frames,
                    intervention=intervention,
                ),
            )
            reader.load_observation = lambda _item, *, observation_role: _png(40)
        controller_config = CohortV2ControllerConfig(
            epochs=1, batch_size=4, hidden_dim=8
        )
        examples = build_cohort_v2_controller_examples(
            readers, labels, controller_config
        )
        models = train_cohort_v2_controllers(
            examples, evaluation.grid.pairs, controller_config
        )
        with torch.no_grad():
            for parameter in models[0].parameters():
                parameter.zero_()
            models[0].pair_head.bias[0] = 1.0
        macro_config = CohortV2MacroConfig(
            steps=1,
            batch_size=1,
            latent_dim=32,
            hidden_dim=16,
            depth=1,
            max_entities=2,
            device="cpu",
        )
        predictor = DualOutputPredictor(macro_config.predictor_config).eval()
        codec = CohortV2StateCodec(latent_dim=32, max_entities=2)
        run = run_cohort_v2_controller_aggregation(
            readers,
            evaluation,
            measurement,
            labels,
            spec,
            predictor,
            codec,
            models,
            controller_config,
            CohortV2AggregationConfig(rounds=1),
            rollout_limit=1,
        )
        return run, controller_config, evaluation, measurement, spec

    def test_round_adds_only_post_transition_training_states_and_reports_baseline(self):
        torch.manual_seed(4)
        run, _config, _evaluation, _measurement, _spec = self._run()

        self.assertTrue(run.result.states)
        self.assertEqual({item.round_index for item in run.result.states}, {1})
        self.assertTrue(all(item.context_position > 0 for item in run.result.states))
        self.assertEqual(
            {item.attempt_id for item in run.result.states},
            {"attempt:training"},
        )
        self.assertEqual(
            tuple(score.name for score in run.result.scores),
            ("oracle_state_baseline", "aggregation_round_1"),
        )

    def test_derived_artifact_is_new_canonical_and_detects_changed_content(self):
        torch.manual_seed(4)
        run, config, evaluation, measurement, spec = self._run()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "derived"
            manifest = write_cohort_v2_controller_aggregation(
                root,
                run,
                config,
                CohortV2AggregationConfig(rounds=1),
                evaluation,
                measurement,
                spec,
                source_controller_artifact_identity="controller:fixture",
                source_controller_checkpoint_identity="controller-checkpoint:fixture",
                source_predictor_checkpoint_identity="predictor:fixture",
                trajectory_label_artifact_identity="labels:fixture",
                derivation_index_identity="derivations:fixture",
                implementation_revision="implementation:fixture",
            )

            self.assertFalse(manifest["source_cohort_mutated"])
            self.assertFalse(manifest["final_evaluation_consumed"])
            self.assertEqual(
                validate_cohort_v2_controller_aggregation(root), manifest
            )
            loaded, loaded_config = load_cohort_v2_aggregated_controllers(root)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded_config, config)
            states = root / "aggregation_states.jsonl"
            states.write_bytes(states.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "identity"):
                validate_cohort_v2_controller_aggregation(root)


if __name__ == "__main__":
    unittest.main()
