from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from PIL import Image
import torch

from world_model.data import CohortV2Rollout
from world_model.data.cohort_v2 import (
    CAPABILITY_DECLARATION_IDENTITY,
    COHORT_V2_RELEASE_IDENTITY,
)
from world_model.model import Abstraction, DualOutputPredictor, identity
from world_model.training.cohort_v2_micro import (
    CohortV2MicroConfig,
    CohortV2StateCodec,
    cohort_v2_model_state_identity,
)
from world_model.training.cohort_v2_reliability import (
    CohortV2ReliabilityConfig,
    CohortV2ReliabilityDerivation,
    CohortV2ReliabilityLabel,
    derive_cohort_v2_reliability_labels,
    evaluate_cohort_v2_reliability_models,
    split_reliability_training_attempts,
    train_cohort_v2_reliability_models,
    validate_cohort_v2_reliability_artifact,
    write_cohort_v2_reliability_artifact,
)

from tests.test_cohort_v2_micro_training import _frame


def _png(red: int) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (20, 12), (red, 20, 30)).save(stream, format="PNG")
    return stream.getvalue()


def _rollout(role: str, attempt_id: str, red: int) -> CohortV2Rollout:
    frames = tuple(
        replace(_frame(f"{attempt_id}:frame:{step}", x=float(step)), fixed_step=step)
        for step in range(3)
    )
    return CohortV2Rollout(
        attempt_id=attempt_id,
        exposure_role=role,
        coverage_stratum="collision",
        scenario_lineage_identity=f"lineage:{role}",
        intervention={
            "engine_relative_action": {
                "drag_delta_canvas_pixels": (12, 3),
                "hold_milliseconds": 1000,
                "tap_time_milliseconds": 0,
            },
            "interface_action": {
                "drag_release": (10, -5),
                "frame_height": 100,
                "releaseTime": 1000,
                "tapTime": 0,
            },
        },
        agent_observation_identity=f"observation:{attempt_id}",
        agent_observation_fixed_step=0,
        frame_records=frames,
    )


def _reader(role: str, attempts: tuple[str, ...], red: int):
    rollouts = tuple(_rollout(role, attempt, red) for attempt in attempts)

    class Reader:
        release_identity = COHORT_V2_RELEASE_IDENTITY
        capability_declaration_identity = CAPABILITY_DECLARATION_IDENTITY
        partition_identity = "partition:fixture"
        derivation_identity = "derivation:fixture"

        @staticmethod
        def load_observation(rollout, *, observation_role: str) -> bytes:
            assert rollout in rollouts
            assert observation_role == "agent"
            return _png(red)

    reader = Reader()
    reader.rollouts = rollouts
    return reader


def _config() -> CohortV2ReliabilityConfig:
    return CohortV2ReliabilityConfig(
        preliminary_steps=6,
        final_steps=6,
        estimator_epochs=2,
        controller_epochs=2,
        batch_size=2,
        evaluation_batch_size=2,
        hidden_dim=8,
        learning_rate=1e-2,
        seed=12,
        device="cpu",
    )


class CohortV2ReliabilityTests(unittest.TestCase):
    def test_counterfactual_targets_are_out_of_sample_and_deployment_observable(self):
        readers = (
            _reader("training", ("train:a", "train:b"), 40),
            _reader("calibration", ("calibration:a",), 80),
            _reader("model_selection", ("selection:a",), 120),
        )
        preliminary, label = split_reliability_training_attempts(readers[0])
        self.assertEqual(preliminary, frozenset({"train:a"}))
        self.assertEqual(label, frozenset({"train:b"}))
        class CounterfactualPredictor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = torch.nn.Parameter(torch.zeros(()))
                self.requests = []

            def carrier(self, contexts, actions, request):
                del actions
                self.requests.append(request)
                self.assert_micro_request(request)
                available = request.mode_input.samples[0].contact.available
                return contexts if available else contexts + 1.0 + self.marker

            @staticmethod
            def assert_micro_request(request):
                if request.pair.abstraction is not Abstraction.MICRO:
                    raise AssertionError("counterfactual changed the mode identity")

        predictor = CounterfactualPredictor()
        derivation = derive_cohort_v2_reliability_labels(
            predictor,
            CohortV2StateCodec(latent_dim=32, max_entities=2),
            readers,
            label,
            _config(),
            "checkpoint:frozen",
        )

        self.assertEqual(
            {item.attempt_id for item in derivation.labels if item.exposure_role == "training"},
            {"train:b"},
        )
        self.assertNotIn("calibration", {item.exposure_role for item in derivation.labels})
        self.assertEqual(
            {item.exposure_role for item in derivation.labels},
            {"training", "model_selection"},
        )
        self.assertTrue(
            all(item.features.shape == (_config().feature_config.feature_dim,) for item in derivation.labels)
        )
        self.assertEqual(derivation.preliminary_checkpoint_identity, "checkpoint:frozen")
        self.assertTrue(all(item.useful for item in derivation.labels))
        self.assertTrue(
            all(
                request.pair.abstraction is Abstraction.MICRO
                for request in predictor.requests
            )
        )

    def test_feature_and_gate_paths_are_scored_independently(self):
        feature_dim = _config().feature_config.feature_dim
        labels = []
        for role, offset in (("training", 0), ("model_selection", 10)):
            for index in range(4):
                useful = index % 2 == 0
                labels.append(CohortV2ReliabilityLabel(
                    state_id=f"state:{role}:{index}",
                    exposure_role=role,
                    attempt_id=f"attempt:{role}",
                    scenario_lineage_identity=f"lineage:{role}",
                    context_position=index,
                    context_fixed_step=index,
                    without_micro_objective=0.2 if useful else 0.1,
                    with_micro_objective=0.1 if useful else 0.2,
                    usefulness_margin=0.1 if useful else -0.1,
                    useful=useful,
                    features=torch.linspace(0.0, 1.0, feature_dim) + offset + index,
                ))
        derivation = CohortV2ReliabilityDerivation(
            tuple(labels), 0, "checkpoint:fixture", "derivation:fixture"
        )
        models = train_cohort_v2_reliability_models(derivation, _config())
        scores = evaluate_cohort_v2_reliability_models(derivation, *models)

        ablation = scores["controller_feature_ablation"]
        self.assertEqual(ablation["loss_gate_fixed"], "off")
        self.assertIn("raw_deployment_features", ablation)
        self.assertIn("raw_plus_reliability_feature", ablation)
        self.assertGreater(ablation["controller_parameter_count_each"], 0)
        self.assertEqual(scores["estimator"]["available_label_count"], 4)

    def test_artifact_is_checkpoint_release_and_split_bound(self):
        config = _config()
        reader = _reader("training", ("train:a", "train:b"), 40)
        readers = (
            reader,
            _reader("calibration", ("calibration:a",), 80),
            _reader("model_selection", ("selection:a",), 120),
        )
        preliminary_attempts, label_attempts = split_reliability_training_attempts(reader)
        feature_dim = config.feature_config.feature_dim
        label = CohortV2ReliabilityLabel(
            "state:fixture", "training", "train:b", "lineage:training", 0, 0,
            0.2, 0.1, 0.1, True, torch.zeros(feature_dim),
        )
        derivation = CohortV2ReliabilityDerivation(
            (label,), 2, "checkpoint:preliminary", "derivation:labels"
        )
        estimator, raw, feature = train_cohort_v2_reliability_models(
            CohortV2ReliabilityDerivation(
                (
                    label,
                    replace(
                        label,
                        state_id="state:selection",
                        exposure_role="model_selection",
                        attempt_id="selection:a",
                    ),
                ),
                2,
                "checkpoint:preliminary",
                "derivation:labels",
            ),
            config,
        )
        predictor_config = CohortV2MicroConfig(
            steps=6,
            batch_size=1,
            latent_dim=32,
            hidden_dim=16,
            depth=1,
            max_entities=2,
            device="cpu",
        ).predictor_config
        trainer = SimpleNamespace(predictor=DualOutputPredictor(predictor_config))
        checkpoint_identity = identity((
            "cohort-v2-micro-reliability-preliminary-checkpoint-v1",
            reader.release_identity,
            reader.partition_identity,
            config.micro_config.identity,
            tuple(sorted(preliminary_attempts)),
            cohort_v2_model_state_identity(trainer.predictor.state_dict()),
            config.preliminary_steps,
        ))
        derivation_identity = identity((
            "cohort-v2-micro-relation-usefulness-derivation-v1",
            reader.release_identity,
            reader.partition_identity,
            checkpoint_identity,
            config.requested_horizon,
            "duration-weighted-carrier-mse:micro-mode-without-minus-with-oracle-relations",
            ((
                label.state_id,
                label.exposure_role,
                label.without_micro_objective,
                label.with_micro_objective,
                label.useful,
            ),),
        ))
        derivation = replace(
            derivation,
            preliminary_checkpoint_identity=checkpoint_identity,
            target_identity=derivation_identity,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_cohort_v2_reliability_artifact(
                root,
                readers=readers,
                config=config,
                preliminary_attempt_ids=preliminary_attempts,
                label_attempt_ids=label_attempts,
                derivation=derivation,
                estimator=estimator,
                raw_controller=raw,
                feature_controller=feature,
                preliminary_trainer=trainer,
                ungated_trainer=trainer,
                gated_trainer=trainer,
                scores={"keep_remove_decision": {"decision": "remove"}},
                implementation_revision="implementation:fixture",
            )
            manifest = validate_cohort_v2_reliability_artifact(
                root, readers=readers, config=config
            )
            self.assertFalse(manifest["final_evaluation_consumed"])

            scores_path = root / "scores.json"
            scores_path.write_bytes(scores_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "provenance|canonical"):
                validate_cohort_v2_reliability_artifact(
                    root, readers=readers, config=config
                )


if __name__ == "__main__":
    unittest.main()
