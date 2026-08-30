from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.cohort_v2_release import _write_derivations
from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_json
from scripts.observation_trace import (
    ObservationTraceError,
    load_observation_bytes,
    persist_observation_trace,
)
from scripts.physics_capture_v2 import load_physics_capture_v2
from scripts.run_issue_62_successor_cohort import (
    _bounds,
    _materialize_slot,
    _pilot_report,
    _synthetic_pilot_records,
    _trajectory_record,
    dry_run,
)
from src.webui.bridge import GameState
from tests.test_observation_trace import engine_capture
from world_model.data.successor_cohort import (
    PUBLIC_ROLES,
    SuccessorCohortError,
    build_pilot_plan,
    build_production_plan,
    load_successor_trajectory,
    validate_pilot_report,
    validate_successor_plan,
)
from world_model.planning.gameplay import SlingshotAction


class SuccessorCohortPlanTests(unittest.TestCase):
    def test_pilot_plan_freezes_non_final_lineages_actions_and_mixtures(self) -> None:
        first = build_pilot_plan()
        second = build_pilot_plan()

        self.assertEqual(first, second)
        self.assertEqual(first["role_counts"], {role: 12 for role in PUBLIC_ROLES})
        self.assertEqual(len(first["lineages"]), 36)
        self.assertNotIn("final_evaluation", first["role_counts"])
        self.assertEqual(
            {item["generator_family"] for item in first["lineages"]},
            {"type010101", "type010102"},
        )
        self.assertEqual(
            {item["behavior_policy"] for item in first["lineages"]},
            {"uniform_random", "stratified_bounds"},
        )
        self.assertEqual(
            len({item["generation_seed"] for item in first["lineages"]}), 36
        )
        self.assertTrue(all(len(item["planned_actions"]) == 6 for item in first["lineages"]))

    def test_production_membership_is_nested_and_pilot_bound(self) -> None:
        pilot = build_pilot_plan()
        report = _pilot_report(pilot, _synthetic_pilot_records(pilot))
        production = build_production_plan(
            report,
            pilot_plan=pilot,
            maximum_training_lineages=1_000,
        )

        self.assertEqual(production["role_counts"], {
            "training": 1_000,
            "calibration": 200,
            "model_selection": 200,
        })
        self.assertEqual(
            [item["lineage_count"] for item in production["nested_training_scales"]],
            [6, 200, 1_000],
        )
        self.assertEqual(production["pilot_report_identity"], report["identity"])
        training = [
            item["slot_identity"] for item in production["lineages"]
            if item["exposure_role"] == "training"
        ]
        self.assertEqual(
            production["nested_training_scales"][1]["slot_identities"],
            training[:200],
        )

        changed = copy.deepcopy(production)
        changed["nested_training_scales"][1]["slot_identities"][0] = "leaked"
        with self.assertRaisesRegex(SuccessorCohortError, "contract|identity|nested"):
            validate_successor_plan(changed)

    def test_failed_pilot_cannot_freeze_production(self) -> None:
        pilot = build_pilot_plan()
        report = _pilot_report(pilot, _synthetic_pilot_records(pilot))
        report["passed"] = False
        with self.assertRaisesRegex(SuccessorCohortError, "pilot report"):
            validate_pilot_report(report, pilot_plan=pilot)

    def test_no_write_dry_run_wires_the_real_carrier_boundary(self) -> None:
        lineages = iter(("lineage:a", "lineage:b"))

        def authority(*_args, **_kwargs):
            return {
                "scenario": SimpleNamespace(
                    scenario_manifest=SimpleNamespace(
                        scenario_lineage=SimpleNamespace(identity=next(lineages))
                    )
                )
            }

        with tempfile.TemporaryDirectory() as temporary, patch(
            "scripts.run_issue_62_successor_cohort._player",
            return_value={"source_snapshot_commit": "player-commit"},
        ), patch(
            "scripts.run_issue_62_successor_cohort._materialize_slot",
            side_effect=authority,
        ), patch(
            "scripts.run_issue_62_successor_cohort.temporal_carrier_main",
            return_value=0,
        ) as carrier:
            result = dry_run(Path(temporary) / "absent-pilot-plan.json")

        carrier.assert_called_once()
        self.assertEqual(result["planned_pilot_lineages"], 36)
        self.assertFalse(result["final_evaluation_opened"])
        self.assertFalse(result["files_written"])
        self.assertTrue(all("python -u -m" in item for item in result["actual_commands"]))


class SuccessorCohortTrajectoryTests(unittest.TestCase):
    def test_reader_builds_source_bound_transition_and_rejects_canonical_input(self) -> None:
        plan = build_pilot_plan()
        slot = plan["lineages"][0]
        release_identity = "issue-62-release:fixture"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authority = _materialize_slot(slot, root)
            scenario = authority["scenario"]
            manifest = scenario.scenario_manifest
            planned = slot["planned_actions"][0]
            source_capture = next(
                (
                    Path(__file__).resolve().parents[1]
                    / "data/runtime_evidence/issue-53-mixed-termination-v5/primary-rollouts"
                ).glob("*/physics_capture_v2.json")
            )
            record = json.loads(source_capture.read_bytes())
            record["source_bindings"] = {
                "scenario_template_id": manifest.scenario_template.identity,
                "level_instance_id": manifest.level_instance.identity,
                "scenario_lineage_id": manifest.scenario_lineage.identity,
                "rollout_id": "rollout-issue-62-fixture",
                "intervention_id": planned["identity"],
            }
            shot_root = root / "shots/shot-000"
            write_immutable_cohort_v2_json(
                record, shot_root / "physics_capture_v2.json"
            )
            captures = []
            for sequence, frame_record in enumerate(record["frame_records"], start=1):
                fixed_step = frame_record["fixed_step"]
                item = engine_capture(
                    sequence=sequence,
                    fixed_step=fixed_step,
                    source="synchronized_fixed_step_camera_render",
                )
                item["capture_id"] = record["capture_id"]
                item["source_frame_identity"] = (
                    f"source-frame-v1:{record['capture_id']}:"
                    f"{sequence}:10:{fixed_step}"
                )
                item["fixed_time_seconds"] = sequence * 0.02
                captures.append(item)
            observation = persist_observation_trace(
                shot_root / "observation-trace",
                captures,
                observation_configuration="agent_rgb8_native_v1",
                source_bindings={
                    "scenario_template_identity": manifest.scenario_template.identity,
                    "level_instance_identity": manifest.level_instance.identity,
                    "source_scenario_lineage_identity": manifest.scenario_lineage.identity,
                    "rollout_identity": "rollout-issue-62-fixture",
                },
                exposure_role="training",
            )
            derivations = _write_derivations(
                shot_root / "derivations",
                load_physics_capture_v2(shot_root / "physics_capture_v2.json"),
                source_reference="shots/shot-000/physics_capture_v2.json",
                release_identity=release_identity,
            )

            action = SlingshotAction(
                planned["drag_x"], planned["drag_y"], planned["tap_time_ms"]
            )
            shot = {
                "shot_index": 0,
                "path": "shots/shot-000",
                "planned_action_identity": planned["identity"],
                "action_stratum": planned["action_stratum"],
                "action": {
                    "identity": planned["identity"],
                    "legal": True,
                    "interface_action": action.to_interface_action((312, 227), _bounds()),
                    "engine_relative_action": {
                        "schema": "slingshot_relative_intervention_v1",
                        "drag_delta_canvas_pixels": [planned["drag_x"], planned["drag_y"]],
                        "hold_milliseconds": planned["release_time_ms"],
                        "tap_time_milliseconds": planned["tap_time_ms"],
                    },
                },
                "capture_id": record["capture_id"],
                "shot_id": record["shot_id"],
                "observation_manifest_identity": observation["identity"],
                "frame_count": len(record["frame_records"]),
                "terminal_reason": "stable_entered",
                "game_state_after": "WON",
                "derivations": [
                    {**item, "path": f"derivations/{item['path']}"}
                    for item in derivations
                ],
            }
            trajectory_record = _trajectory_record(
                root,
                slot,
                authority,
                [shot],
                release_identity=release_identity,
                final_state=GameState.WON,
            )
            write_immutable_cohort_v2_json(
                trajectory_record, root / "trajectory.json"
            )

            trajectory = load_successor_trajectory(
                root, release_identity=release_identity
            )
            self.assertEqual(len(trajectory.transitions), 1)
            transition = trajectory.transitions[0]
            self.assertEqual(transition.terminal_status, "success")
            self.assertEqual(
                transition.source_bindings["release_identity"], release_identity
            )
            self.assertIn("engine_state", transition.targets.source_targets)
            frame = observation["frame_records"][0]
            with self.assertRaisesRegex(ObservationTraceError, "canonical"):
                load_observation_bytes(
                    shot_root / "observation-trace",
                    frame_record_identity=frame["identity"],
                    observation_role="canonical",
                    workflow_kind="training",
                    purpose="model_input",
                )


if __name__ == "__main__":
    unittest.main()
