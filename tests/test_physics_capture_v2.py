from __future__ import annotations

import copy
from dataclasses import replace
import math
import json
from pathlib import Path
import tempfile
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.physics_capture_v2 import (
    PhysicsCaptureV2Error,
    bind_physics_capture_v2_engine,
    normalized_initial_engine_state_identity,
    load_physics_capture_v2,
    parse_physics_capture_v2,
)
from scripts.physics_capture_v2_persistence import (
    persist_physics_capture_v2,
    source_bindings_from_collection,
    validate_physics_capture_v2_artifact,
)
from scripts.collection_plan import CollectionIntervention, RuntimeInput


def capture() -> dict:
    entities = [
        {"entity_id": "bird:1", "scenario_object_id": "bird", "lifecycle": "active", "body_present": True, "body": {"body_type": "dynamic", "simulated": True, "gravity_scale": 1.0, "gravity_applicable": True, "position": [0, 1], "rotation_degrees": 0.0, "velocity": [1, 0], "angular_velocity_degrees_per_second": 0.0}, "contact_ids": [], "supported_by_entity_ids": [], "supports_entity_ids": []},
        {"entity_id": "world:static:1", "scenario_object_id": "ground", "lifecycle": "active", "body_present": True, "body": {"body_type": "static", "simulated": True, "gravity_scale": 0.0, "gravity_applicable": False, "position": [0, 0], "rotation_degrees": 0.0, "velocity": [0, 0], "angular_velocity_degrees_per_second": 0.0}, "contact_ids": [], "supported_by_entity_ids": [], "supports_entity_ids": []},
    ]
    contacted_entities = copy.deepcopy(entities)
    contacted_entities[0].update({"contact_ids": ["contact:1"], "supported_by_entity_ids": ["world:static:1"]})
    contacted_entities[1].update({"contact_ids": ["contact:1"], "supports_entity_ids": ["bird:1"]})
    contact = {"contact_id": "contact:1", "entity_a_id": "bird:1", "entity_b_id": "world:static:1", "collider_a_id": "collider:bird", "collider_b_id": "collider:ground", "point": [0, 0], "normal_a_to_b": [0, 1], "separation": -0.01}
    collider_snapshots = [
        {"collider_id": "collider:bird", "entity_id": "bird:1", "geometry_source": "unity_collider_2d", "enabled": True, "is_trigger": False, "shape": {"kind": "circle", "center": [0, 1], "radius": 0.5}},
        {"collider_id": "collider:ground", "entity_id": "world:static:1", "geometry_source": "unity_collider_2d", "enabled": True, "is_trigger": False, "shape": {"kind": "edge", "points": [[-5, 0], [5, 0]]}},
    ]
    return {
        "schema_version": "physics_capture_v2",
        "capture_id": "capture-1",
        "shot_id": "shot-1",
        "source_bindings": {"scenario_template_id": "template-1", "level_instance_id": "level-1", "scenario_lineage_id": "lineage-1", "rollout_id": "rollout-1", "intervention_id": "intervention-1"},
        "configured_fixed_step_capture_stride": 1,
        "pre_intervention_fixed_step": 0,
        "coordinate_convention": {"world_space": "unity_world_2d", "world_x_axis": "right", "world_y_axis": "up", "world_length_unit": "unity_unit"},
        "causal_entities": ["bird:1", "world:static:1"],
        "colliders": [
            {"collider_id": "collider:bird", "entity_id": "bird:1", "geometry_source": "unity_collider_2d"},
            {"collider_id": "collider:ground", "entity_id": "world:static:1", "geometry_source": "unity_collider_2d"},
        ],
        "fixed_step_samples": [
            {"fixed_step": 0, "complete_raw_non_trigger_contacts": True, "world": {"world_id": "world-1", "gravity_vector": [0, -9.81]}, "entities": entities, "colliders": copy.deepcopy(collider_snapshots), "contacts": [], "supports": []},
            {"fixed_step": 1, "complete_raw_non_trigger_contacts": True, "world": {"world_id": "world-1", "gravity_vector": [0, -9.81]}, "entities": contacted_entities, "colliders": copy.deepcopy(collider_snapshots), "contacts": [contact], "supports": [{"supporter_entity_id": "world:static:1", "supported_entity_id": "bird:1", "contact_ids": ["contact:1"]}]},
        ],
        "minimum_contact_separation": {
            "observed": True,
            "separation": -0.01,
            "contact_id": "contact:1",
            "fixed_step": 1,
        },
        "frame_records": [
            {"fixed_step": 0, "state_id": "state-0", "forced_terminal": False},
            {"fixed_step": 1, "state_id": "state-1", "forced_terminal": False},
        ],
        "events": [{"event_id": "terminal-1", "event_type": "stable_entered", "fixed_step": 1, "participants": [], "payload": {}}],
        "terminal_evidence": {"reason": "stable_entered", "fixed_step": 1, "event_id": "terminal-1"},
    }


class PhysicsCaptureV2Tests(unittest.TestCase):
    def test_normalized_initial_engine_state_excludes_rollout_ids_but_detects_state_change(self) -> None:
        first = parse_physics_capture_v2(capture())
        repeated_record = capture()
        repeated_record["capture_id"] = "capture-2"
        repeated_record["shot_id"] = "shot-2"
        repeated_record["source_bindings"]["rollout_id"] = "rollout-2"
        repeated = parse_physics_capture_v2(repeated_record)
        self.assertEqual(
            normalized_initial_engine_state_identity(first),
            normalized_initial_engine_state_identity(repeated),
        )
        changed_record = capture()
        changed_record["fixed_step_samples"][0]["entities"][0]["body"]["position"] = [0.1, 1]
        changed = parse_physics_capture_v2(changed_record)
        self.assertNotEqual(
            normalized_initial_engine_state_identity(first),
            normalized_initial_engine_state_identity(changed),
        )

    def test_published_schema_declares_the_validated_record_surfaces(self) -> None:
        schema = json.loads(
            (Path(__file__).parents[1] / "docs/data_contracts/physics_capture_v2.schema.json").read_text(encoding="utf-8")
        )
        fixture = capture()
        self.assertEqual(set(schema["required"]), set(fixture))
        self.assertEqual(
            set(schema["$defs"]["entity"]["required"]),
            set(fixture["fixed_step_samples"][0]["entities"][0]),
        )
        self.assertEqual(
            set(schema["$defs"]["colliderCatalog"]["required"]),
            set(fixture["colliders"][0]),
        )
        self.assertEqual(
            set(schema["$defs"]["collider"]["required"]),
            set(fixture["fixed_step_samples"][0]["colliders"][0]),
        )
        self.assertEqual(
            set(schema["$defs"]["event"]["required"]),
            set(fixture["events"][0]),
        )

    def test_loader_rejects_the_sidecar_byte_bound_before_json_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "physics_capture_v2.json"
            path.write_bytes(b"12345")
            with patch("scripts.physics_capture_v2.MAX_CAPTURE_BYTES", 4):
                with self.assertRaisesRegex(PhysicsCaptureV2Error, "byte bound"):
                    load_physics_capture_v2(path)

    def test_parser_rejects_declared_record_count_bounds(self) -> None:
        with patch("scripts.physics_capture_v2.MAX_FIXED_STEP_SAMPLES", 1):
            with self.assertRaisesRegex(PhysicsCaptureV2Error, "fixed-step sample bound"):
                parse_physics_capture_v2(capture())
        with patch("scripts.physics_capture_v2.MAX_CAUSAL_ENTITIES", 1):
            with self.assertRaisesRegex(PhysicsCaptureV2Error, "causal-entity bound"):
                parse_physics_capture_v2(capture())

    def test_source_bindings_must_match_the_v2_manifest_and_collection_plan_entry(self) -> None:
        manifest_record = {
            "schema": "scenario_manifest_v1",
            "identity": "scenario-source",
            "nested": {"values": ["one"]},
        }
        scenario = SimpleNamespace(
            identity="cohort-v2-manifest-1",
            template_record=SimpleNamespace(identity="template-1"),
            scenario_manifest=SimpleNamespace(
                to_dict=lambda: manifest_record,
                scenario_template=SimpleNamespace(identity="template-1"),
                level_instance=SimpleNamespace(identity="level-1"),
                scenario_lineage=SimpleNamespace(identity="lineage-1"),
                declared_initial_engine_state=SimpleNamespace(identity="initial-1"),
            ),
        )
        intervention = CollectionIntervention(
            "intervention-1", 0, "collision", "geometry_stratified",
            {}, {}, "mapping-v1", {}, {},
        )
        collection_scenario = SimpleNamespace(
            scenario_id="scenario-1",
            identity="collection-scenario-1",
            scenario_manifest_projection={
                "scenario_manifest": MappingProxyType({
                    "schema": "scenario_manifest_v1",
                    "identity": "scenario-source",
                    "nested": MappingProxyType({"values": ("one",)}),
                }),
            },
            expected_initial_engine_state_identity="initial-1",
            interventions=(intervention,),
        )
        runtime = RuntimeInput(
            "plan-1", 1, "scenario-1", "collection-scenario-1",
            intervention.id, intervention.identity, "attempt-1", 1, "initial-1",
            {}, {}, "mapping-v1", {},
        )

        bindings = source_bindings_from_collection(
            scenario,
            collection_scenario,
            runtime,
            rollout_identity="rollout-1",
        )
        self.assertEqual(bindings, {
            "scenario_template_id": "template-1",
            "level_instance_id": "level-1",
            "scenario_lineage_id": "lineage-1",
            "rollout_id": "rollout-1",
            "intervention_id": intervention.identity,
        })

        with self.assertRaisesRegex(PhysicsCaptureV2Error, "stale source binding"):
            source_bindings_from_collection(
                scenario,
                collection_scenario,
                replace(runtime, scenario_identity="stale"),
                rollout_identity="rollout-1",
            )

    def test_atomically_persists_only_a_valid_bound_v2_sidecar(self) -> None:
        engine = capture()
        bindings = engine.pop("source_bindings")
        engine["schema_version"] = "physics_capture_v2_engine_v1"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = persist_physics_capture_v2(
                root,
                engine,
                source_bindings=bindings,
                scenario_manifest_identity="cohort-v2-scenario-manifest-v1:test-manifest",
            )
            sidecar = root / "physics_capture_v2.json"
            self.assertTrue(sidecar.is_file())
            self.assertEqual(metadata["physics_capture_v2_path"], "physics_capture_v2.json")
            self.assertEqual(metadata["fixed_step_sample_count"], 2)
            self.assertEqual(metadata["scenario_manifest_identity"], "cohort-v2-scenario-manifest-v1:test-manifest")
            self.assertTrue(metadata["initial_engine_state_identity"].startswith("normalized-initial-engine-state-v1:"))
            self.assertEqual(
                validate_physics_capture_v2_artifact(root, metadata).capture_id,
                "capture-1",
            )
        invalid = copy.deepcopy(engine)
        invalid["configured_fixed_step_capture_stride"] = 0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(PhysicsCaptureV2Error):
                persist_physics_capture_v2(
                    root,
                    invalid,
                    source_bindings=bindings,
                    scenario_manifest_identity="manifest-1",
                )
            self.assertFalse((root / "physics_capture_v2.json").exists())

    def test_collector_binds_engine_evidence_to_frozen_plan_authorities(self) -> None:
        final_capture = capture()
        bindings = final_capture.pop("source_bindings")
        final_capture["schema_version"] = "physics_capture_v2_engine_v1"

        bound = bind_physics_capture_v2_engine(final_capture, bindings)

        self.assertEqual(bound.record["schema_version"], "physics_capture_v2")
        self.assertEqual(bound.source_bindings, bindings)
        self.assertNotIn("source_bindings", final_capture)

    def test_accepts_complete_source_bound_capture(self) -> None:
        parsed = parse_physics_capture_v2(capture())
        self.assertEqual(parsed.configured_fixed_step_capture_stride, 1)
        self.assertEqual(parsed.source_bindings["scenario_lineage_id"], "lineage-1")

    def test_pre_intervention_step_is_the_first_complete_sample(self) -> None:
        valid = capture()
        valid["pre_intervention_fixed_step"] = 0
        parse_physics_capture_v2(valid)
        invalid = copy.deepcopy(valid)
        invalid["pre_intervention_fixed_step"] = 1
        with self.assertRaisesRegex(PhysicsCaptureV2Error, "pre-intervention"):
            parse_physics_capture_v2(invalid)

    def test_validates_the_capture_wide_minimum_contact_separation(self) -> None:
        valid = capture()
        valid["minimum_contact_separation"] = {
            "observed": True,
            "separation": -0.01,
            "contact_id": "contact:1",
            "fixed_step": 1,
        }
        self.assertEqual(
            parse_physics_capture_v2(valid).record["minimum_contact_separation"]["fixed_step"],
            1,
        )

        invalid = copy.deepcopy(valid)
        invalid["minimum_contact_separation"]["separation"] = -0.02
        with self.assertRaisesRegex(PhysicsCaptureV2Error, "recomputed minimum"):
            parse_physics_capture_v2(invalid)

    def test_rejects_invalid_stride_or_contact_gap(self) -> None:
        invalid = capture()
        invalid["configured_fixed_step_capture_stride"] = 0
        with self.assertRaisesRegex(PhysicsCaptureV2Error, "positive integer"):
            parse_physics_capture_v2(invalid)
        invalid = capture()
        invalid["fixed_step_samples"].append(copy.deepcopy(invalid["fixed_step_samples"][-1]))
        invalid["fixed_step_samples"][-1]["fixed_step"] = 3
        with self.assertRaisesRegex(PhysicsCaptureV2Error, "contact coverage has a gap"):
            parse_physics_capture_v2(invalid)

    def test_rejects_nondeterministic_entity_and_collider_order(self) -> None:
        invalid = capture()
        invalid["causal_entities"] = list(reversed(invalid["causal_entities"]))
        with self.assertRaisesRegex(PhysicsCaptureV2Error, "deterministic order"):
            parse_physics_capture_v2(invalid)
        invalid = capture()
        invalid["colliders"] = list(reversed(invalid["colliders"]))
        with self.assertRaisesRegex(PhysicsCaptureV2Error, "deterministic order"):
            parse_physics_capture_v2(invalid)

    def test_rejects_missing_geometry_gravity_identity_and_nonfinite_values(self) -> None:
        invalid = capture()
        invalid["fixed_step_samples"][0]["colliders"][0]["shape"] = {}
        with self.assertRaisesRegex(PhysicsCaptureV2Error, "geometry is absent"):
            parse_physics_capture_v2(invalid)
        invalid = capture()
        del invalid["fixed_step_samples"][0]["world"]["gravity_vector"]
        with self.assertRaisesRegex(PhysicsCaptureV2Error, "missing fields"):
            parse_physics_capture_v2(invalid)
        invalid = capture()
        invalid["fixed_step_samples"][1]["contacts"][0]["collider_a_id"] = "unknown"
        with self.assertRaisesRegex(PhysicsCaptureV2Error, "unresolved collider geometry"):
            parse_physics_capture_v2(invalid)
        invalid = capture()
        invalid["fixed_step_samples"][1]["contacts"][0]["separation"] = math.nan
        with self.assertRaisesRegex(PhysicsCaptureV2Error, "finite number"):
            parse_physics_capture_v2(invalid)

    def test_accepts_only_supported_direct_unity_collider_geometry(self) -> None:
        shapes = (
            {"kind": "circle", "center": [0, 1], "radius": 0.5},
            {"kind": "box", "center": [0, 1], "size": [1, 2], "angle_degrees": 15},
            {"kind": "polygon", "paths": [[[0, 0], [1, 0], [0, 1]]]},
            {"kind": "edge", "points": [[-1, 0], [1, 0]]},
            {"kind": "capsule", "center": [0, 1], "size": [1, 2], "direction": "vertical", "angle_degrees": 0},
        )
        for shape in shapes:
            with self.subTest(kind=shape["kind"]):
                valid = capture()
                for sample in valid["fixed_step_samples"]:
                    sample["colliders"][0]["shape"] = copy.deepcopy(shape)
                parse_physics_capture_v2(valid)

        invalid = capture()
        invalid["fixed_step_samples"][0]["colliders"][0]["shape"] = {"kind": "mesh", "vertices": []}
        with self.assertRaisesRegex(PhysicsCaptureV2Error, "supported Unity Collider2D"):
            parse_physics_capture_v2(invalid)

    def test_freezes_direct_world_collider_geometry_at_every_fixed_step(self) -> None:
        valid = capture()
        valid["fixed_step_samples"][1]["colliders"][0]["shape"]["center"] = [0.25, 0.75]

        parsed = parse_physics_capture_v2(valid)

        self.assertEqual(
            parsed.record["fixed_step_samples"][1]["colliders"][0]["shape"]["center"],
            [0.25, 0.75],
        )

    def test_requires_explicit_body_presence_gravity_scale_pose_and_motion(self) -> None:
        valid = capture()
        for sample in valid["fixed_step_samples"]:
            for entity in sample["entities"]:
                entity["body_present"] = True
                entity["body"].update({
                    "gravity_scale": 1.0 if entity["body"]["gravity_applicable"] else 0.0,
                    "rotation_degrees": 0.0,
                    "angular_velocity_degrees_per_second": 0.0,
                })
        self.assertEqual(
            parse_physics_capture_v2(valid).record["fixed_step_samples"][0]["entities"][0]["body_present"],
            True,
        )

        invalid = copy.deepcopy(valid)
        invalid["fixed_step_samples"][0]["entities"][0]["body_present"] = False
        with self.assertRaisesRegex(PhysicsCaptureV2Error, "body presence"):
            parse_physics_capture_v2(invalid)

    def test_entity_support_and_contact_context_is_complete_at_each_step(self) -> None:
        valid = capture()
        for entity in valid["fixed_step_samples"][0]["entities"]:
            entity.update({"contact_ids": [], "supported_by_entity_ids": [], "supports_entity_ids": []})
        bird, world = valid["fixed_step_samples"][1]["entities"]
        bird.update({
            "contact_ids": ["contact:1"],
            "supported_by_entity_ids": ["world:static:1"],
            "supports_entity_ids": [],
        })
        world.update({
            "contact_ids": ["contact:1"],
            "supported_by_entity_ids": [],
            "supports_entity_ids": ["bird:1"],
        })
        self.assertEqual(
            parse_physics_capture_v2(valid).record["fixed_step_samples"][1]["entities"][0]["contact_ids"],
            ["contact:1"],
        )

        invalid = copy.deepcopy(valid)
        invalid["fixed_step_samples"][1]["entities"][0]["contact_ids"] = []
        with self.assertRaisesRegex(PhysicsCaptureV2Error, "support/contact context"):
            parse_physics_capture_v2(invalid)

    def test_rejects_uncovered_termination(self) -> None:
        invalid = capture()
        invalid["terminal_evidence"]["fixed_step"] = 2
        with self.assertRaisesRegex(PhysicsCaptureV2Error, "termination lacks retained physical evidence"):
            parse_physics_capture_v2(invalid)

    def test_events_carry_macro_type_and_finite_payload_on_the_fixed_step_clock(self) -> None:
        valid = capture()
        valid["events"][0].update({"event_type": "stable_entered", "payload": {}})
        self.assertEqual(parse_physics_capture_v2(valid).record["events"][0]["event_type"], "stable_entered")

        invalid = copy.deepcopy(valid)
        invalid["events"][0]["payload"] = {"speed": math.inf}
        with self.assertRaisesRegex(PhysicsCaptureV2Error, "finite number"):
            parse_physics_capture_v2(invalid)

    def test_event_types_require_resolved_causal_participants_and_terminal_reason(self) -> None:
        invalid = capture()
        invalid["events"][0].update({"event_type": "collision", "participants": []})
        with self.assertRaisesRegex(PhysicsCaptureV2Error, "participant count"):
            parse_physics_capture_v2(invalid)

        invalid = capture()
        invalid["events"][0]["event_type"] = "unknown"
        with self.assertRaisesRegex(PhysicsCaptureV2Error, "event type"):
            parse_physics_capture_v2(invalid)

        invalid = capture()
        invalid["terminal_evidence"]["reason"] = "level_clear"
        with self.assertRaisesRegex(PhysicsCaptureV2Error, "terminal reason"):
            parse_physics_capture_v2(invalid)

    def test_accepts_an_explicit_forced_terminal_record_off_the_stride_grid(self) -> None:
        off_grid = capture()
        off_grid["configured_fixed_step_capture_stride"] = 2
        terminal_sample = copy.deepcopy(off_grid["fixed_step_samples"][-1])
        terminal_sample["fixed_step"] = 3
        off_grid["fixed_step_samples"].insert(1, copy.deepcopy(off_grid["fixed_step_samples"][0]))
        off_grid["fixed_step_samples"][1]["fixed_step"] = 1
        off_grid["fixed_step_samples"][-1]["fixed_step"] = 2
        off_grid["fixed_step_samples"].append(terminal_sample)
        off_grid["frame_records"] = [
            {"fixed_step": 0, "state_id": "state-0", "forced_terminal": False},
            {"fixed_step": 2, "state_id": "state-2", "forced_terminal": False},
            {"fixed_step": 3, "state_id": "state-3", "forced_terminal": True},
        ]
        off_grid["events"][0]["fixed_step"] = 3
        off_grid["terminal_evidence"]["fixed_step"] = 3
        off_grid["minimum_contact_separation"]["fixed_step"] = 2

        parsed = parse_physics_capture_v2(off_grid)
        self.assertEqual(parsed.record["frame_records"][-1]["forced_terminal"], True)
