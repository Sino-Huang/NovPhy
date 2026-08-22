from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_issue_48_evidence import (
    build_issue_48_evidence,
    prepare_issue_48_runtime_root,
)
from scripts.cohort_v2_replay import (
    ATTEMPT_NAME,
    ATTEMPT_SCHEMA,
    CURRENT_DETERMINATION_VERSION,
    FROZEN_COMMAND_NAME,
    CohortV2ReplayError,
    _compare_contact_geometry,
    build_frozen_replay_command,
    build_replay_report,
    semantic_identity,
    validate_issue_48_evidence,
    validate_replay_plan,
)
from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_json
from scripts.observation_trace import persist_observation_trace
from scripts.physics_capture_v2_persistence import persist_physics_capture_v2


ROOT = Path(__file__).resolve().parents[1]
PHYSICS = {
    "training-collision": ROOT / "data/runtime_evidence/issue-44/captures/collision.json",
    "calibration-stable": ROOT / "data/runtime_evidence/issue-44/captures/stable-terminal.json",
}
OBSERVATIONS = {
    "training-collision": ROOT / "data/runtime_evidence/issue-46/traces/training-native",
    "calibration-stable": ROOT / "data/runtime_evidence/issue-46/traces/calibration-resized",
}


def _physics_metadata(metadata: dict) -> dict:
    fields = (
        "physics_capture_v2_schema", "capture_id", "shot_id",
        "configured_fixed_step_capture_stride", "causal_entity_count", "collider_count",
        "fixed_step_sample_count", "frame_record_count", "event_count",
        "initial_engine_state_identity", "scenario_manifest_identity",
    )
    return {field: metadata[field] for field in fields}


def _observation_capture(scenario_collection_id: str) -> dict:
    root = OBSERVATIONS[scenario_collection_id]
    manifest = json.loads((root / "observation_trace_manifest.json").read_text(encoding="utf-8"))
    frame = manifest["frame_records"][0]
    return {
        **deepcopy(frame["capture_metadata"]),
        "canonical_png": (root / frame["canonical_observation"]["relative_path"]).read_bytes(),
    }


class Issue48ReplayEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime = self.root / "runtime"
        self.plan = prepare_issue_48_runtime_root(
            ROOT,
            self.runtime,
            determination_version=CURRENT_DETERMINATION_VERSION,
            require_clean_revision=False,
        )
        for scenario_collection_index, scenario_collection in enumerate(self.plan["scenario_collections"]):
            source = json.loads(PHYSICS[scenario_collection["scenario_collection_id"]].read_text(encoding="utf-8"))
            source.pop("source_bindings")
            source["schema_version"] = "physics_capture_v2_engine_v1"
            for role_index, role in enumerate(("original", "replay")):
                attempt_identity = semantic_identity(
                    "cohort-v2-replay-attempt-v1",
                    self.plan["identity"],
                    scenario_collection["scenario_collection_id"],
                    role,
                )
                attempt_root = self.runtime / "attempts" / scenario_collection["scenario_collection_id"] / role
                attempt_root.mkdir(parents=True)
                engine = deepcopy(source)
                suffix = scenario_collection_index * 2 + role_index + 1
                engine["capture_id"] = f"capture-v2:{suffix:032x}"
                engine["shot_id"] = f"shot-v2:{suffix:032x}"
                bindings = {
                    "scenario_template_id": scenario_collection["scenario_template_identity"],
                    "level_instance_id": scenario_collection["level_instance_identity"],
                    "scenario_lineage_id": scenario_collection["scenario_lineage_identity"],
                    "rollout_id": attempt_identity,
                    "intervention_id": scenario_collection["intervention"]["identity"],
                }
                metadata = persist_physics_capture_v2(
                    attempt_root,
                    engine,
                    source_bindings=bindings,
                    scenario_manifest_identity=scenario_collection["scenario_manifest_identity"],
                )
                observation_bindings = {
                    "scenario_template_identity": scenario_collection["scenario_template_identity"],
                    "level_instance_identity": scenario_collection["level_instance_identity"],
                    "source_scenario_lineage_identity": scenario_collection["scenario_lineage_identity"],
                    "rollout_identity": attempt_identity,
                }
                observation = persist_observation_trace(
                    attempt_root / "observation-trace",
                    [_observation_capture(scenario_collection["scenario_collection_id"])],
                    observation_configuration=scenario_collection["observation_configuration"],
                    source_bindings=observation_bindings,
                    exposure_role=scenario_collection["exposure_role"],
                )
                interface_action = {
                    **scenario_collection["intervention"]["interface_action"],
                    "drag_start": [100, 200],
                    "slingshot_reference": {
                        "gameX": 100, "gameY": 200, "canvasX": 100, "canvasY": 279,
                    },
                    "socket_command": {
                        "x": 100 + scenario_collection["intervention"]["interface_action"]["drag_release"][0],
                        "y": 200 - scenario_collection["intervention"]["interface_action"]["drag_release"][1],
                        "tapTime": 0,
                        "releaseTime": 1000,
                    },
                }
                attempt = {
                    "schema": ATTEMPT_SCHEMA,
                    "identity": attempt_identity,
                    "attempt_role": role,
                    "scenario_collection_identity": scenario_collection["identity"],
                    "rollout_identity": attempt_identity,
                    "version_envelope": self.plan["version_envelope"],
                    "partition_manifest_identity": self.plan["partition_manifest_identity"],
                    "collection_plan_identity": self.plan["identity"],
                    "exposure_role": scenario_collection["exposure_role"],
                    "scenario_manifest_identity": scenario_collection["scenario_manifest_identity"],
                    "scenario_specification_identity": scenario_collection["scenario_specification_identity"],
                    "scenario_content_identity": scenario_collection["scenario_content_identity"],
                    "scenario_template_identity": scenario_collection["scenario_template_identity"],
                    "level_instance_identity": scenario_collection["level_instance_identity"],
                    "scenario_lineage_identity": scenario_collection["scenario_lineage_identity"],
                    "intervention_identity": scenario_collection["intervention"]["identity"],
                    "interface_action": interface_action,
                    "engine_relative_action": scenario_collection["intervention"]["engine_relative_action"],
                    "physics_capture_relative_path": "physics_capture_v2.json",
                    "physics_capture_metadata": _physics_metadata(metadata),
                    "observation_trace_relative_path": "observation-trace",
                    "observation_trace_manifest_identity": observation["identity"],
                    "observation_configuration_identity": scenario_collection["observation_configuration_identity"],
                }
                write_immutable_cohort_v2_json(attempt, attempt_root / ATTEMPT_NAME)
                if role == "original":
                    frozen_command = build_frozen_replay_command(
                        self.plan, scenario_collection, attempt
                    )
                    write_immutable_cohort_v2_json(
                        frozen_command,
                        self.runtime
                        / "attempts"
                        / scenario_collection["scenario_collection_id"]
                        / FROZEN_COMMAND_NAME,
                    )

    def test_representative_original_and_replay_pairs_pass_every_component(self) -> None:
        report = build_replay_report(self.runtime)

        self.assertTrue(report["passed"])
        self.assertEqual(report["retry_count"], 0)
        self.assertEqual(report["coverage"]["scenario_collection_count"], 2)
        self.assertEqual(report["coverage"]["non_final_scenario_lineage_count"], 2)
        self.assertEqual(report["coverage"]["level_instance_count"], 2)
        self.assertEqual(report["coverage"]["scenario_template_count"], 2)
        self.assertEqual(
            {component["status"] for verdict in report["scenario_collection_verdicts"] for component in verdict["components"]},
            {"equality", "not_required"},
        )

    def test_version_change_cannot_pass_the_old_envelope(self) -> None:
        attempt_path = self.runtime / "attempts/training-collision/replay/attempt.json"
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        attempt["version_envelope"]["unity_version"] = "changed"
        attempt_path.write_text(json.dumps(attempt), encoding="utf-8")

        report = build_replay_report(self.runtime)

        self.assertFalse(report["passed"])
        unavailable = report["scenario_collection_verdicts"][0]["components"][0]
        self.assertEqual(unavailable["status"], "unavailable")
        self.assertIn("binding is stale", unavailable["details"]["reason"])

    def test_identity_drift_and_missing_observation_fail_closed(self) -> None:
        physics_path = self.runtime / "attempts/calibration-stable/replay/physics_capture_v2.json"
        physics = json.loads(physics_path.read_text(encoding="utf-8"))
        physics["source_bindings"]["scenario_lineage_id"] = "scenario-lineage-v1:changed"
        physics_path.write_text(json.dumps(physics), encoding="utf-8")
        observation = self.runtime / "attempts/training-collision/replay/observation-trace/observations/agent/frame_000000.png"
        observation.unlink()

        report = build_replay_report(self.runtime)

        self.assertFalse(report["passed"])
        self.assertTrue(all(not verdict["passed"] for verdict in report["scenario_collection_verdicts"]))

    def test_changed_plan_requires_a_new_determination(self) -> None:
        path = self.runtime / "replay-plan.json"
        plan = json.loads(path.read_text(encoding="utf-8"))
        plan["version_envelope"]["unity_version"] = "changed"
        path.write_text(json.dumps(plan), encoding="utf-8")

        with self.assertRaisesRegex(CohortV2ReplayError, "version envelope identity is stale"):
            validate_replay_plan(self.runtime)

    def test_scenario_collection_identity_binds_declared_coverage(self) -> None:
        path = self.runtime / "replay-plan.json"
        plan = json.loads(path.read_text(encoding="utf-8"))
        plan["scenario_collections"][0]["coverage_strata"].append("support")
        path.write_text(json.dumps(plan), encoding="utf-8")

        with self.assertRaisesRegex(
            CohortV2ReplayError, "scenario collection identity is stale"
        ):
            validate_replay_plan(self.runtime)

    def test_version_envelope_identity_binds_every_declared_contract(self) -> None:
        path = self.runtime / "replay-plan.json"
        plan = json.loads(path.read_text(encoding="utf-8"))
        plan["version_envelope"]["physics_engine_contract"] = "changed"
        path.write_text(json.dumps(plan), encoding="utf-8")

        with self.assertRaisesRegex(
            CohortV2ReplayError, "version envelope identity is stale"
        ):
            validate_replay_plan(self.runtime)

    def test_intervention_identity_binds_interface_and_engine_records(self) -> None:
        path = self.runtime / "replay-plan.json"
        plan = json.loads(path.read_text(encoding="utf-8"))
        plan["scenario_collections"][0]["intervention"]["interface_action"][
            "frame_height"
        ] = 481
        path.write_text(json.dumps(plan), encoding="utf-8")

        with self.assertRaisesRegex(
            CohortV2ReplayError, "intervention identity is stale"
        ):
            validate_replay_plan(self.runtime)

    def test_intervention_rejects_undeclared_nested_fields(self) -> None:
        path = self.runtime / "replay-plan.json"
        plan = json.loads(path.read_text(encoding="utf-8"))
        plan["scenario_collections"][0]["intervention"]["interface_action"][
            "unsupported"
        ] = True
        path.write_text(json.dumps(plan), encoding="utf-8")

        with self.assertRaisesRegex(
            CohortV2ReplayError, "interface action is unsupported"
        ):
            validate_replay_plan(self.runtime)

    def test_replay_must_use_the_exact_frozen_socket_command(self) -> None:
        attempt_path = self.runtime / "attempts/training-collision/replay/attempt.json"
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        attempt["interface_action"]["socket_command"]["x"] += 1
        attempt_path.write_text(json.dumps(attempt), encoding="utf-8")

        report = build_replay_report(self.runtime)

        self.assertFalse(report["passed"])
        intervention = next(
            component
            for component in report["scenario_collection_verdicts"][0]["components"]
            if component["component"] == "intervention"
        )
        self.assertEqual(intervention["status"], "mismatch")

    def test_missing_frozen_command_fails_closed(self) -> None:
        command = (
            self.runtime
            / "attempts"
            / "calibration-stable"
            / FROZEN_COMMAND_NAME
        )
        command.unlink()

        report = build_replay_report(self.runtime)

        self.assertFalse(report["passed"])
        intervention = next(
            component
            for component in report["scenario_collection_verdicts"][1]["components"]
            if component["component"] == "intervention"
        )
        self.assertEqual(intervention["status"], "unavailable")

    def test_intermediate_engine_state_drift_is_reported_as_measurement_tolerance(self) -> None:
        physics_path = (
            self.runtime
            / "attempts"
            / "training-collision"
            / "replay"
            / "physics_capture_v2.json"
        )
        physics = json.loads(physics_path.read_text(encoding="utf-8"))
        physics["fixed_step_samples"][10]["entities"][0]["body"]["velocity"][0] += 0.25
        physics_path.write_text(json.dumps(physics), encoding="utf-8")

        report = build_replay_report(self.runtime)

        self.assertTrue(report["passed"])
        state_trace = next(
            component
            for component in report["scenario_collection_verdicts"][0]["components"]
            if component["component"] == "engine_state_measurements"
        )
        self.assertEqual(state_trace["status"], "tolerated")

    def test_intermediate_lifecycle_drift_fails_closed(self) -> None:
        physics_path = (
            self.runtime
            / "attempts"
            / "training-collision"
            / "replay"
            / "physics_capture_v2.json"
        )
        physics = json.loads(physics_path.read_text(encoding="utf-8"))
        physics["fixed_step_samples"][10]["entities"][0]["lifecycle"] = "inactive"
        physics_path.write_text(json.dumps(physics), encoding="utf-8")

        report = build_replay_report(self.runtime)

        self.assertFalse(report["passed"])
        semantics = next(
            component
            for component in report["scenario_collection_verdicts"][0]["components"]
            if component["component"] == "deterministic_engine_state_semantics"
        )
        self.assertEqual(semantics["status"], "mismatch")

    def test_frame_record_identity_drift_fails_closed(self) -> None:
        physics_path = (
            self.runtime
            / "attempts"
            / "calibration-stable"
            / "replay"
            / "physics_capture_v2.json"
        )
        physics = json.loads(physics_path.read_text(encoding="utf-8"))
        physics["frame_records"][10]["state_id"] = "state:999999"
        physics_path.write_text(json.dumps(physics), encoding="utf-8")

        report = build_replay_report(self.runtime)

        self.assertFalse(report["passed"])
        identities = next(
            component
            for component in report["scenario_collection_verdicts"][1]["components"]
            if component["component"] == "deterministic_artifact_identities"
        )
        self.assertEqual(identities["status"], "mismatch")

    def test_contact_identity_and_timing_drift_fail_while_measurements_tolerate(
        self,
    ) -> None:
        key = ("entity-a", "entity-b", "collider-a", "collider-b")
        original = {
            key: [{
                "relative_fixed_step": 10,
                "point": [1.0, 2.0],
                "normal_a_to_b": [0.0, 1.0],
                "separation": -0.01,
            }]
        }
        replay = deepcopy(original)
        replay[key][0]["relative_fixed_step"] = 12
        self.assertEqual(
            _compare_contact_geometry(original, replay, 1, 0.001)["status"],
            "mismatch",
        )

        changed_key = ("entity-a", "entity-b", "collider-a", "collider-c")
        self.assertEqual(
            _compare_contact_geometry(
                original, {changed_key: deepcopy(original[key])}, 1, 0.001
            )["status"],
            "mismatch",
        )

        replay = deepcopy(original)
        replay[key][0]["point"] = [1.25, 2.0]
        self.assertEqual(
            _compare_contact_geometry(original, replay, 1, 0.001)["status"],
            "tolerated",
        )

    def test_published_bundle_revalidates_exact_membership(self) -> None:
        output = self.root / "published"
        bundle = build_issue_48_evidence(self.runtime, output)

        self.assertTrue(bundle["passed"])
        self.assertEqual(validate_issue_48_evidence(output), bundle)
        extra = output / "undeclared.txt"
        extra.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(CohortV2ReplayError, "membership or identity is stale"):
            validate_issue_48_evidence(output)


if __name__ == "__main__":
    unittest.main()
