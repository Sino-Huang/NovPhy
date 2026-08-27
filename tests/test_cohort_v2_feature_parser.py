from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from world_model.data import CohortV2CentralFrameRecord, CohortV2Rollout
from world_model.model import Abstraction, PredictionPair
from world_model.training.cohort_v2_feature_parser import (
    CohortV2FeatureParserConfig,
    CohortV2FeatureParserError,
    LearnedFeatureTransitionRequestBuilder,
    build_feature_parser_model,
    build_feature_parser_role_data,
    calibrate_feature_parser_probabilities,
    calibrate_feature_parser_thresholds,
    entity_observable_features,
    feature_parser_metrics,
    load_feature_parser_checkpoint,
    parse_frame_symbols,
    save_feature_parser_checkpoint,
    select_feature_parser,
    train_feature_parser,
)


def _entity(identity: str, x: float, *, proxy: str = "a") -> dict:
    kind = identity.split(":", 1)[0]
    return {
        "entity_id": f"runtime:{identity}",
        "scenario_object_id": f"{kind}:0000",
        "lifecycle": "active",
        "body_present": True,
        "body": {
            "body_type": "dynamic",
            "simulated": True,
            "gravity_applicable": True,
            "gravity_scale": 1.0,
            "position": (x, 1.0),
            "velocity": (x / 10.0, 0.0),
            "rotation_degrees": 0.0,
            "angular_velocity_degrees_per_second": 0.0,
        },
        "contact_ids": (f"contact:{proxy}",),
        "supports_entity_ids": (f"runtime:block:{proxy}",),
        "supported_by_entity_ids": (f"runtime:platform:{proxy}",),
    }


def _frame(index: int, *, unavailable_macro: bool = False) -> CohortV2CentralFrameRecord:
    contact = (("runtime:block:0000", "runtime:pig:0000"),) if index % 2 else ()
    supports = (("runtime:block:0000", "runtime:pig:0000"),) if index % 3 else ()
    unavailable = "unavailable_no_predecessor"
    return CohortV2CentralFrameRecord(
        identity=f"frame:{index}",
        capture_id="capture",
        state_id=f"state:{index}",
        fixed_step=index,
        capture_stride=1,
        engine_state={
            "world": {"gravity_vector": (0.0, -9.81)},
            "entities": (
                _entity("block:0000", float(index)),
                _entity("pig:0000", float(index) + 1.0),
            ),
        },
        events=(),
        labels={
            "contact": {"availability": "available", "relations": contact},
            "supports": {"availability": "available", "relations": supports},
            "steady-state": {
                "availability": unavailable if unavailable_macro else "available",
                "value": None if unavailable_macro else index % 2 == 0,
            },
            "structure-unstable": {
                "availability": unavailable if unavailable_macro else "available",
                "value": None if unavailable_macro else index % 3 == 0,
            },
        },
        terminal=None,
    )


def _reader(role: str, *, offset: int = 0):
    rollout = CohortV2Rollout(
        attempt_id=f"attempt:{role}",
        exposure_role=role,
        coverage_stratum="fixture",
        scenario_lineage_identity=f"lineage:{role}",
        intervention={},
        agent_observation_identity="observation",
        agent_observation_fixed_step=0,
        frame_records=tuple(_frame(offset + index) for index in range(1, 7)),
    )
    return SimpleNamespace(
        rollouts=(rollout,),
        release_identity="release",
        capability_declaration_identity="capabilities",
        partition_identity="partition",
        derivation_identity=f"derivation:{role}",
    )


class CohortV2FeatureParserTests(unittest.TestCase):
    def setUp(self):
        self.readers = (
            _reader("training", offset=0),
            _reader("calibration", offset=10),
            _reader("model_selection", offset=20),
        )
        self.data = tuple(
            build_feature_parser_role_data(reader, expected_role=role)
            for reader, role in zip(
                self.readers,
                ("training", "calibration", "model_selection"),
                strict=True,
            )
        )
        self.config = CohortV2FeatureParserConfig(
            hidden_dim=8,
            epochs=1,
            relation_batch_size=16,
            macro_batch_size=8,
            device="cpu",
        )

    def test_observable_features_exclude_oracle_relation_proxies(self):
        first = _entity("block:0000", 1.0, proxy="first")
        second = _entity("block:0000", 1.0, proxy="second")

        self.assertTrue(torch.equal(
            entity_observable_features(first), entity_observable_features(second)
        ))

    def test_complete_exposure_roles_are_not_mixed(self):
        with self.assertRaisesRegex(CohortV2FeatureParserError, "crosses exposure roles"):
            build_feature_parser_role_data(
                self.readers[0], expected_role="calibration"
            )

    def test_training_selection_and_calibration_keep_declared_roles(self):
        model = build_feature_parser_model(self.config, self.data[0])
        train_feature_parser(model, self.data[0])
        selected, rows = select_feature_parser((model,), self.data[2])
        temperatures = calibrate_feature_parser_probabilities(selected, self.data[1])
        thresholds = calibrate_feature_parser_thresholds(
            selected, self.data[1], temperatures
        )
        metrics = feature_parser_metrics(
            selected, self.data[1], thresholds, temperatures
        )

        self.assertTrue(rows[0]["selected"])
        self.assertEqual(set(thresholds), {
            "contact", "supports", "steady-state", "structure-unstable"
        })
        self.assertEqual(metrics["contact"]["unavailable_frame_count"], 0)
        with self.assertRaisesRegex(CohortV2FeatureParserError, "training role"):
            train_feature_parser(model, self.data[1])
        with self.assertRaisesRegex(CohortV2FeatureParserError, "model-selection role"):
            select_feature_parser((model,), self.data[1])
        with self.assertRaisesRegex(CohortV2FeatureParserError, "calibration role"):
            calibrate_feature_parser_thresholds(
                model,
                self.data[2],
                {predicate: 1.0 for predicate in thresholds},
            )

    def test_unavailable_targets_remain_unavailable_in_learned_request(self):
        model = build_feature_parser_model(self.config, self.data[0]).eval()
        frame = _frame(30, unavailable_macro=True)
        thresholds = {predicate: 0.0 for predicate in (
            "contact", "supports", "steady-state", "structure-unstable"
        )}
        temperatures = {predicate: 1.0 for predicate in thresholds}
        parsed = parse_frame_symbols(model, frame, temperatures, thresholds)
        builder = LearnedFeatureTransitionRequestBuilder(
            {frame.identity: parsed}, "checkpoint", "fixture-source"
        )
        window = SimpleNamespace(context=frame)

        continuous = builder(PredictionPair(1, Abstraction.CONTINUOUS), (window,))
        micro = builder(PredictionPair(1, Abstraction.MICRO), (window,))
        macro = builder(PredictionPair(1, Abstraction.MACRO), (window,))

        self.assertIsNone(continuous.mode_input)
        self.assertTrue(micro.mode_input.samples[0].contact.available)
        self.assertFalse(macro.mode_input.samples[0].steady_state.available)
        self.assertIsNone(macro.mode_input.samples[0].steady_state.value)
        self.assertTrue(all(first != second for first, second in (
            micro.mode_input.samples[0].supports.relations or ()
        )))

    def test_checkpoint_round_trip_binds_role_sources_and_thresholds(self):
        model = build_feature_parser_model(self.config, self.data[0])
        train_feature_parser(model, self.data[0])
        temperatures = calibrate_feature_parser_probabilities(model, self.data[1])
        thresholds = calibrate_feature_parser_thresholds(
            model, self.data[1], temperatures
        )
        metrics = feature_parser_metrics(
            model, self.data[1], thresholds, temperatures
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_feature_parser_checkpoint(
                Path(directory) / "parser",
                model,
                temperatures,
                thresholds,
                role_data=self.data,
                readers=self.readers,
                model_selection=(),
                calibration_metrics=metrics,
                implementation_revision="commit:fixture",
            )
            loaded, reloaded, manifest = load_feature_parser_checkpoint(
                Path(directory) / "parser",
                readers=self.readers,
                device="cpu",
            )

        self.assertEqual(reloaded.identity, checkpoint.identity)
        self.assertEqual(dict(reloaded.thresholds), thresholds)
        self.assertEqual(manifest["source_bindings"]["learned_parameter_role"], "training")
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value.cpu(), loaded.state_dict()[name]))


if __name__ == "__main__":
    unittest.main()
