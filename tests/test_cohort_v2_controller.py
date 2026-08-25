from __future__ import annotations

from io import BytesIO
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from world_model.data import CohortV2Rollout
from world_model.training import (
    CohortV2ControllerConfig,
    CohortV2ControllerFeatureCodec,
    CohortV2ExecutionProfile,
    CohortV2ExhaustiveEvaluator,
    CohortV2TrajectoryCostSpec,
    build_cohort_v2_controller_examples,
    evaluate_cohort_v2_controllers,
    generate_cohort_v2_trajectory_labels,
    measure_cohort_v2_evaluation,
    train_cohort_v2_controllers,
    validate_cohort_v2_controllers,
    write_cohort_v2_controllers,
)

from tests.test_cohort_v2_policy_baselines import _BaselineScorer, _calibration
from tests.test_cohort_v2_exhaustive_evaluation import _reader


def _png(red: int) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (20, 12), (red, 20, 30)).save(stream, format="PNG")
    return stream.getvalue()


def _controller_reader(role: str, red: int):
    source = _reader(role)
    original = source.rollouts[0]
    rollout = CohortV2Rollout(
        attempt_id=original.attempt_id,
        exposure_role=original.exposure_role,
        coverage_stratum=original.coverage_stratum,
        scenario_lineage_identity=original.scenario_lineage_identity,
        intervention={
            "interface_action": {
                "drag_release": (10, -5),
                "frame_height": 100,
                "releaseTime": 1000,
                "tapTime": 0,
            }
        },
        agent_observation_identity=original.agent_observation_identity,
        agent_observation_fixed_step=original.agent_observation_fixed_step,
        frame_records=original.frame_records,
    )

    class Reader:
        release_identity = source.release_identity
        capability_declaration_identity = source.capability_declaration_identity
        partition_identity = source.partition_identity
        derivation_identity = "derivations:fixture"
        rollouts = (rollout,)

        @staticmethod
        def load_observation(item, *, observation_role: str) -> bytes:
            assert item is rollout
            assert observation_role == "agent"
            return _png(red)

    return Reader()


def _inputs():
    readers = tuple(
        _controller_reader(role, red)
        for role, red in zip(
            ("training", "calibration", "model_selection"),
            (40, 80, 120),
            strict=True,
        )
    )
    evaluation = CohortV2ExhaustiveEvaluator(_BaselineScorer()).evaluate(readers)
    measurement = measure_cohort_v2_evaluation(
        evaluation,
        readers,
        _calibration(),
        CohortV2ExecutionProfile(
            controller_executed=False,
            shared_perception_executed=False,
        ),
    )
    spec = CohortV2TrajectoryCostSpec(0.0, 0.0, 1.0)
    labels = generate_cohort_v2_trajectory_labels(evaluation, measurement, spec)
    return readers, evaluation, measurement, labels, spec


class CohortV2ControllerTests(unittest.TestCase):
    def test_features_use_agent_observation_intervention_and_elapsed_position(self):
        config = CohortV2ControllerConfig(epochs=1)
        codec = CohortV2ControllerFeatureCodec(config)
        intervention = {
            "interface_action": {
                "drag_release": (10, -5),
                "frame_height": 100,
                "releaseTime": 1000,
                "tapTime": 0,
            }
        }

        first = codec.encode(_png(40), elapsed_fixed_steps=0, intervention=intervention)
        later = codec.encode(_png(40), elapsed_fixed_steps=5, intervention=intervention)

        self.assertEqual(first.shape, (config.feature_dim,))
        self.assertFalse(first.equal(later))

    def test_joint_and_two_head_controllers_share_inputs_capacity_and_held_out_scope(self):
        readers, evaluation, measurement, labels, spec = _inputs()
        config = CohortV2ControllerConfig(epochs=2, batch_size=4, hidden_dim=8)
        examples = build_cohort_v2_controller_examples(readers, labels, config)
        models = train_cohort_v2_controllers(examples, evaluation.grid.pairs, config)
        result = evaluate_cohort_v2_controllers(
            models, examples, evaluation, measurement, spec
        )

        counts = tuple(sum(parameter.numel() for parameter in model.parameters()) for model in models)
        self.assertEqual(counts[0], counts[1])
        self.assertEqual({score.exposure_role for score in result.scores}, {"model_selection"})
        self.assertEqual({score.controller_id for score in result.scores}, {"joint_pair", "matched_capacity_two_head"})
        self.assertEqual({score.state_count for score in result.scores}, {3})
        self.assertTrue(all(score.utility_available_count == 3 for score in result.scores))

    def test_artifacts_reload_models_and_recompute_held_out_metrics(self):
        readers, evaluation, measurement, labels, spec = _inputs()
        config = CohortV2ControllerConfig(epochs=2, batch_size=4, hidden_dim=8)
        kwargs = {
            "trajectory_label_artifact_identity": "labels:fixture",
            "baseline_artifact_identity": "baselines:fixture",
            "derivation_index_identity": "derivations:fixture",
            "implementation_revision": "implementation:fixture",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            written = write_cohort_v2_controllers(
                root,
                readers,
                evaluation,
                measurement,
                labels,
                spec,
                config,
                **kwargs,
            )
            validated = validate_cohort_v2_controllers(
                root, readers, evaluation, measurement, labels, spec, **kwargs
            )
            manifest = json.loads((root / "manifest.json").read_bytes())

            self.assertEqual(written, validated)
            self.assertFalse(manifest["oracle_engine_state_is_controller_input"])
            self.assertFalse(manifest["final_evaluation_consumed"])
            self.assertEqual(manifest["matched_parameter_count"], written.parameter_count)

            scores = root / "scores.json"
            scores.write_bytes(scores.read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                validate_cohort_v2_controllers(
                    root, readers, evaluation, measurement, labels, spec, **kwargs
                )


if __name__ == "__main__":
    unittest.main()
