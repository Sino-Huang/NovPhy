from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch

from tests.test_observation_trace import engine_capture
from world_model.data.cohort_v2 import CohortV2CentralFrameRecord, CohortV2Rollout
from world_model.training.cohort_v2_visual_parser import (
    CohortV2VisualParserConfig,
    build_visual_parser_model,
    build_visual_parser_role_data,
    calibrate_visual_parser,
    load_visual_parser_checkpoint,
    parse_visual_frame_symbols,
    save_visual_parser_checkpoint,
    select_visual_parser,
    train_visual_parser,
    visual_object_vocabulary,
    visual_parser_metrics,
)


def _png(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(output, format="PNG")
    return output.getvalue()


def _entity(name: str, x: float, y: float, *, active: bool = True) -> dict:
    return {
        "entity_id": "runtime:" + name,
        "scenario_object_id": name,
        "lifecycle": "active" if active else "destroyed",
        "body_present": active,
        "body": None if not active else {
            "body_type": "dynamic",
            "simulated": True,
            "gravity_applicable": True,
            "gravity_scale": 1.0,
            "position": (x, y),
            "rotation_degrees": 0.0,
            "velocity": (0.0, 0.0),
            "angular_velocity_degrees_per_second": 0.0,
        },
    }


class _Reader:
    release_identity = "cohort-v2-aligned-observation-release-v1:issue-59"
    partition_identity = "partition"
    derivation_identity = "derivation"
    capability_declaration_identity = "cohort-v2-capabilities-v1"

    def __init__(self, role: str) -> None:
        self.role = role
        self._images = {}
        self._metadata = {}
        frames = []
        for index in range(4):
            entities = (
                _entity("bird:0000", -0.5 + index * 0.1, 0.0),
                _entity("pig:0000", 0.5, 0.0, active=index < 3),
                _entity("platform:0000", 0.0, -0.5),
            )
            unavailable = index == 0
            availability = "unavailable_initial" if unavailable else "available"
            frame = CohortV2CentralFrameRecord(
                identity=f"{role}:frame:{index}",
                capture_id=f"capture:{role}",
                state_id=f"state:{index}",
                fixed_step=index,
                capture_stride=1,
                engine_state={"entities": entities, "world": {"gravity_vector": (0.0, -9.81)}},
                events=(),
                labels={
                    "contact": {
                        "availability": "available",
                        "relations": (("runtime:pig:0000", "runtime:platform:0000"),)
                        if index < 3 else (),
                    },
                    "supports": {
                        "availability": "available",
                        "relations": (("runtime:platform:0000", "runtime:pig:0000"),)
                        if index < 3 else (),
                    },
                    "steady-state": {
                        "availability": availability,
                        "value": None if unavailable else index >= 2,
                    },
                    "structure-unstable": {
                        "availability": availability,
                        "value": None if unavailable else index == 1,
                    },
                    "excess_penetration": {"availability": "available", "value": False},
                    "unsupported_stationary_or_floating_body": {
                        "availability": "available", "values": ()
                    },
                },
                terminal=None,
            )
            frames.append(frame)
            self._images[index] = _png((20 + index * 40, 40, 80))
            metadata = engine_capture(sequence=index + 1, fixed_step=index)
            metadata.pop("canonical_png")
            self._metadata[index] = metadata
        self.rollouts = (
            CohortV2Rollout(
                attempt_id=f"attempt:{role}",
                exposure_role=role,
                coverage_stratum="collision",
                scenario_lineage_identity=f"lineage:{role}",
                intervention={
                    "interface_action": {
                        "drag_release": (-10, 5), "frame_height": 480,
                        "releaseTime": 0, "tapTime": 0,
                    }
                },
                agent_observation_identity="observation:0",
                agent_observation_fixed_step=0,
                frame_records=tuple(frames),
            ),
        )

    def load_frame_observation(self, rollout, frame, *, observation_role):
        self.assert_agent(observation_role)
        return self._images[frame.fixed_step]

    def frame_observation_metadata(self, rollout, frame):
        return self._metadata[frame.fixed_step]

    @staticmethod
    def assert_agent(role):
        if role != "agent":
            raise AssertionError("visual model requested canonical pixels")


class CohortV2VisualParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.readers = tuple(_Reader(role) for role in (
            "training", "calibration", "model_selection"
        ))
        self.config = CohortV2VisualParserConfig(
            image_height=4, image_width=4, hidden_dim=8,
            epochs=2, batch_size=2, device="cpu",
        )
        self.vocabulary = visual_object_vocabulary(self.readers)
        self.data = tuple(
            build_visual_parser_role_data(
                reader, self.config, expected_role=reader.role,
                object_vocabulary=self.vocabulary,
            )
            for reader in self.readers
        )

    def test_frozen_encoder_and_supervised_heads_train_without_canonical_input(self):
        model = build_visual_parser_model(self.config, self.data[0])
        before = {key: value.clone() for key, value in model.encoder.state_dict().items()}
        train_visual_parser(model, self.data[0])

        self.assertTrue(all(
            torch.equal(value, model.encoder.state_dict()[key])
            for key, value in before.items()
        ))
        self.assertFalse(any(
            parameter.requires_grad for parameter in model.encoder.parameters()
        ))

    def test_calibration_metrics_selection_and_unavailable_symbols(self):
        first = build_visual_parser_model(self.config, self.data[0])
        train_visual_parser(first, self.data[0])
        selected, rows = select_visual_parser((first,), self.data[2])
        temperatures, thresholds, kind_temperature = calibrate_visual_parser(
            selected, self.data[1]
        )
        metrics = visual_parser_metrics(
            selected, self.data[1], temperatures, thresholds, kind_temperature
        )
        frame = self.readers[1].rollouts[0].frame_records[0]
        symbols = parse_visual_frame_symbols(
            selected, self.data[1].images[0], frame, temperatures, thresholds
        )

        self.assertTrue(rows[0]["selected"])
        self.assertIn("object_alignment", metrics)
        self.assertIn("object_kind", metrics)
        self.assertFalse(symbols.steady_state.available)
        self.assertIsNone(symbols.steady_state.value)
        self.assertTrue(symbols.contact.available)

    def test_checkpoint_round_trip_binds_frozen_encoder_and_role_sources(self):
        model = build_visual_parser_model(self.config, self.data[0])
        train_visual_parser(model, self.data[0])
        temperatures, thresholds, kind_temperature = calibrate_visual_parser(
            model, self.data[1]
        )
        metrics = visual_parser_metrics(
            model, self.data[1], temperatures, thresholds, kind_temperature
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = save_visual_parser_checkpoint(
                Path(temporary) / "parser", model, temperatures, thresholds,
                kind_temperature, role_data=self.data, readers=self.readers,
                model_selection=(), calibration_metrics=metrics,
                implementation_revision="commit:fixture",
            )
            loaded, reloaded, manifest = load_visual_parser_checkpoint(
                Path(temporary) / "parser", readers=self.readers, device="cpu"
            )

        self.assertEqual(reloaded.identity, checkpoint.identity)
        self.assertTrue(manifest["source_bindings"]["encoder_frozen"])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value.cpu(), loaded.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
