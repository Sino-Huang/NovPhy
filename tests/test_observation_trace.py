from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.webui.bridge import ObservationCaptureEngine
from scripts.observation_trace import (
    ObservationTraceError,
    audit_observation_access,
    capture_observation_trace,
    load_observation_bytes,
    persist_observation_trace,
    validate_observation_exposure_boundaries,
    validate_observation_trace,
    TRANSFORMS,
)


def png(width: int = 4, height: int = 3) -> bytes:
    image = Image.new("RGB", (width, height), (20, 40, 60))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def engine_capture(*, sequence: int = 1) -> dict:
    identity_matrix = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    return {
        "schema_version": "observation_capture_engine_v1",
        "capture_id": "capture-runtime-1",
        "sequence": sequence,
        "source_frame_identity": f"source-frame-v1:capture-runtime-1:{sequence}:10:20",
        "render_frame": 10,
        "render_time_seconds": 1.5,
        "fixed_step": 20,
        "fixed_time_seconds": 0.4,
        "source": "synchronized_observation_endpoint",
        "camera": {
            "camera_identity": "unity-main-camera",
            "projection_kind": "orthographic",
            "position_world": [0.0, 0.0, -10.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "orthographic_size_world_units": 5.0,
            "vertical_field_of_view_degrees": None,
            "near_clip_world_units": 0.3,
            "far_clip_world_units": 1000.0,
            "aspect_ratio": 4.0 / 3.0,
            "world_to_camera_matrix": identity_matrix,
            "camera_to_clip_matrix": identity_matrix,
        },
        "viewport": {
            "width_pixels": 4,
            "height_pixels": 3,
            "camera_pixel_rect": [0.0, 0.0, 4.0, 3.0],
            "screen_width_pixels": 4,
            "screen_height_pixels": 3,
            "pixel_origin": "bottom_left",
        },
        "coordinates": {
            "world_space": "unity_world_2d",
            "world_units": "unity_unit",
            "observation_space": "rgb_pixel",
            "observation_units": "pixel",
            "observation_origin": "top_left",
            "observation_x_axis": "right",
            "observation_y_axis": "down",
            "channel_order": "RGB",
            "sample_type": "uint8",
            "color_space": "sRGB",
        },
        "world_to_observation_transform": {
            "method": "unity_world_to_clip_to_top_left_pixel_v1",
            "world_to_camera_matrix": identity_matrix,
            "camera_to_clip_matrix": identity_matrix,
            "clip_to_ndc": "homogeneous_divide",
            "ndc_to_observation_matrix": [2.0, 0.0, 2.0, 0.0, -1.5, 1.5, 0.0, 0.0, 1.0],
        },
        "canonical_png": png(),
    }


def source_bindings() -> dict[str, str]:
    return {
        "scenario_template_identity": "template:training",
        "level_instance_identity": "level:training",
        "source_scenario_lineage_identity": "lineage:training",
        "rollout_identity": "rollout:training:1",
    }


class ObservationTraceTests(unittest.TestCase):
    def test_published_schema_and_transform_registry_match_the_runtime_contract(self) -> None:
        contract_root = Path(__file__).parents[1] / "docs" / "data_contracts"
        schema = json.loads(
            (contract_root / "observation_trace_v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        transforms = json.loads(
            (contract_root / "observation_transforms_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(schema["$id"], "observation_trace_v1.schema.json")
        self.assertEqual(schema["properties"]["schema"]["const"], "observation_trace_manifest_v1")
        self.assertEqual(
            set(transforms["transforms"]),
            set(TRANSFORMS),
        )
        self.assertEqual(
            transforms["transforms"]["agent_rgb8_nearest_320x240_v1"],
            TRANSFORMS["agent_rgb8_nearest_320x240_v1"],
        )

    def test_persists_exact_canonical_and_transformed_agent_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "trace"
            manifest = persist_observation_trace(
                root,
                [engine_capture()],
                observation_configuration="agent_rgb8_nearest_2x2_v1",
                source_bindings=source_bindings(),
                exposure_role="training",
            )

            validated = validate_observation_trace(root)
            self.assertEqual(validated, manifest)
            self.assertNotEqual(
                manifest["source_bindings"]["source_scenario_lineage_identity"],
                manifest["scenario_lineage_identity"],
            )
            frame = manifest["frame_records"][0]
            self.assertNotEqual(
                frame["agent_observation"]["identity"],
                frame["canonical_observation"]["identity"],
            )
            self.assertEqual(
                load_observation_bytes(
                    root,
                    frame_record_identity=frame["identity"],
                    observation_role="agent",
                    workflow_kind="training",
                    purpose="model_input",
                ),
                (root / frame["agent_observation"]["relative_path"]).read_bytes(),
            )
            with Image.open(io.BytesIO(
                (root / frame["agent_observation"]["relative_path"]).read_bytes()
            )) as image:
                self.assertEqual(image.size, (2, 2))

    def test_capture_implementation_consumes_only_typed_request_72_records(self) -> None:
        record = engine_capture()
        canonical_png = record.pop("canonical_png")

        class Bridge:
            def get_observation_capture(self):
                return ObservationCaptureEngine(canonical_png, record)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "trace"
            manifest = capture_observation_trace(
                Bridge(),
                root,
                frame_count=1,
                observation_configuration="agent_rgb8_native_v1",
                source_bindings=source_bindings(),
                exposure_role="training",
            )

            self.assertEqual(len(manifest["frame_records"]), 1)
            self.assertEqual(
                manifest["frame_records"][0]["capture_metadata"]["source"],
                "synchronized_observation_endpoint",
            )

    def test_capture_boundary_rejects_incomplete_or_non_synchronized_sources(self) -> None:
        cases = []
        missing_camera = engine_capture()
        del missing_camera["camera"]["world_to_camera_matrix"]
        cases.append((missing_camera, "camera"))
        missing_viewport = engine_capture()
        del missing_viewport["viewport"]["height_pixels"]
        cases.append((missing_viewport, "viewport"))
        missing_transform = engine_capture()
        del missing_transform["world_to_observation_transform"]["camera_to_clip_matrix"]
        cases.append((missing_transform, "world-to-observation"))
        screenshot = engine_capture()
        screenshot["source"] = "ordinary_screenshot_request"
        cases.append((screenshot, "screenshot"))

        for index, (capture, message) in enumerate(cases):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(ObservationTraceError, message):
                    persist_observation_trace(
                        Path(temporary) / f"trace-{index}",
                        [capture],
                        observation_configuration="agent_rgb8_native_v1",
                        source_bindings=source_bindings(),
                        exposure_role="training",
                    )

    def test_world_to_observation_transform_uses_camera_viewport_offset_and_extent(self) -> None:
        capture = engine_capture()
        capture["viewport"]["camera_pixel_rect"] = [1.0, 0.5, 2.0, 1.0]
        capture["world_to_observation_transform"]["ndc_to_observation_matrix"] = [
            1.0, 0.0, 2.0,
            0.0, -0.5, 2.0,
            0.0, 0.0, 1.0,
        ]

        with tempfile.TemporaryDirectory() as temporary:
            manifest = persist_observation_trace(
                Path(temporary) / "trace",
                [capture],
                observation_configuration="agent_rgb8_native_v1",
                source_bindings=source_bindings(),
                exposure_role="training",
            )

            self.assertEqual(
                manifest["frame_records"][0]["capture_metadata"]
                ["world_to_observation_transform"]["ndc_to_observation_matrix"],
                capture["world_to_observation_transform"]["ndc_to_observation_matrix"],
            )

    def test_canonical_bytes_are_available_only_to_diagnostics_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "trace"
            manifest = persist_observation_trace(
                root,
                [engine_capture()],
                observation_configuration="agent_rgb8_native_v1",
                source_bindings=source_bindings(),
                exposure_role="model_selection",
            )
            frame = manifest["frame_records"][0]

            canonical = load_observation_bytes(
                root,
                frame_record_identity=frame["identity"],
                observation_role="canonical",
                workflow_kind="diagnostic",
                purpose="alignment_diagnosis",
            )
            self.assertEqual(canonical, engine_capture()["canonical_png"])

            rejected = (
                ("training", "model_input"),
                ("calibration", "model_input"),
                ("model_selection", "model_input"),
                ("model_selection", "comparator_selection"),
                ("final_evaluation", "reported_model_input"),
            )
            for workflow, purpose in rejected:
                with self.subTest(workflow=workflow, purpose=purpose):
                    with self.assertRaisesRegex(ObservationTraceError, "canonical"):
                        load_observation_bytes(
                            root,
                            frame_record_identity=frame["identity"],
                            observation_role="canonical",
                            workflow_kind=workflow,
                            purpose=purpose,
                        )

            report = audit_observation_access(
                manifest,
                [
                    {
                        "attempt_identity": "access:canonical:diagnostic",
                        "observation_role": "canonical",
                        "workflow_kind": "diagnostic",
                        "purpose": "capture_diagnosis",
                    },
                    {
                        "attempt_identity": "access:canonical:training",
                        "observation_role": "canonical",
                        "workflow_kind": "training",
                        "purpose": "model_input",
                    },
                ],
            )
            self.assertTrue(report["passed"])
            self.assertEqual(
                [decision["allowed"] for decision in report["decisions"]],
                [True, False],
            )

    def test_agent_ingestion_cannot_cross_the_manifest_exposure_role(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "trace"
            manifest = persist_observation_trace(
                root,
                [engine_capture()],
                observation_configuration="agent_rgb8_native_v1",
                source_bindings=source_bindings(),
                exposure_role="training",
            )
            frame = manifest["frame_records"][0]

            with self.assertRaisesRegex(ObservationTraceError, "cross exposure"):
                load_observation_bytes(
                    root,
                    frame_record_identity=frame["identity"],
                    observation_role="agent",
                    workflow_kind="calibration",
                    purpose="model_input",
                )

    def test_missing_duplicated_misaligned_and_cross_role_observations_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ObservationTraceError, "duplicated source frame"):
                persist_observation_trace(
                    root / "duplicate",
                    [engine_capture(), engine_capture()],
                    observation_configuration="agent_rgb8_native_v1",
                    source_bindings=source_bindings(),
                    exposure_role="training",
                )

            first = persist_observation_trace(
                root / "training",
                [engine_capture()],
                observation_configuration="agent_rgb8_native_v1",
                source_bindings=source_bindings(),
                exposure_role="training",
            )
            second = persist_observation_trace(
                root / "calibration",
                [engine_capture()],
                observation_configuration="agent_rgb8_nearest_2x2_v1",
                source_bindings={**source_bindings(), "rollout_identity": "rollout:calibration:1"},
                exposure_role="calibration",
            )
            with self.assertRaisesRegex(ObservationTraceError, "cross exposure"):
                validate_observation_exposure_boundaries([first, second])

            frame = first["frame_records"][0]
            (root / "training" / frame["canonical_observation"]["relative_path"]).unlink()
            with self.assertRaisesRegex(ObservationTraceError, "canonical observation is missing"):
                validate_observation_trace(root / "training")


if __name__ == "__main__":
    unittest.main()
