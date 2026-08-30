from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

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
    _encode_agent_frames_webm,
    _materialize_slot,
    _pilot_report,
    _synthetic_pilot_records,
    _trajectory_record,
    dry_run,
    run_collection,
    write_pilot_audit,
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
        ) as temporal, patch(
            "scripts.run_issue_62_successor_cohort._verify_webm_encoder"
        ) as encoder:
            result = dry_run(Path(temporary) / "absent-pilot-plan.json")

        temporal.assert_called_once()
        encoder.assert_called_once()
        self.assertEqual(result["planned_pilot_lineages"], 36)
        self.assertFalse(result["final_evaluation_opened"])
        self.assertFalse(result["files_written"])
        self.assertTrue(all("python -u -m" in item for item in result["actual_commands"]))
        self.assertIn("WebM", result["pilot_audit_format"])


class SuccessorCohortAuditTests(unittest.TestCase):
    @staticmethod
    def _observation_trace(
        root: Path,
        *,
        capture_id: str,
        fixed_steps: tuple[int, ...],
        role: str,
        rollout_identity: str,
    ) -> dict:
        captures = []
        for sequence, fixed_step in enumerate(fixed_steps, start=1):
            item = engine_capture(
                sequence=sequence,
                fixed_step=fixed_step,
                source="synchronized_fixed_step_camera_render",
            )
            item["capture_id"] = capture_id
            item["source_frame_identity"] = (
                f"source-frame-v1:{capture_id}:{sequence}:10:{fixed_step}"
            )
            item["fixed_time_seconds"] = fixed_step * 0.02
            captures.append(item)
        return persist_observation_trace(
            root,
            captures,
            observation_configuration="agent_rgb8_native_v1",
            source_bindings={
                "scenario_template_identity": "template:audit",
                "level_instance_identity": "level:audit",
                "source_scenario_lineage_identity": "lineage:audit",
                "rollout_identity": rollout_identity,
            },
            exposure_role=role,
        )

    def test_pilot_audit_orders_all_agent_frames_and_writes_gallery(self) -> None:
        plan = build_pilot_plan()
        accepted_slot = plan["lineages"][0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            output = root / "data/issue-62-pilot-audit"
            trajectory_root = (
                runtime / "accepted" / accepted_slot["exposure_role"]
                / accepted_slot["slot_identity"]
            )
            shots = []
            for shot_index, fixed_steps in enumerate(((10, 11), (20, 21))):
                shot_path = f"shots/shot-{shot_index:03d}"
                self._observation_trace(
                    trajectory_root / shot_path / "observation-trace",
                    capture_id=f"capture:audit:{shot_index}",
                    fixed_steps=fixed_steps,
                    role=accepted_slot["exposure_role"],
                    rollout_identity=f"rollout:audit:{shot_index}",
                )
                shots.append({
                    "shot_index": shot_index,
                    "path": shot_path,
                    "action_stratum": f"stratum:{shot_index}",
                })
            write_immutable_cohort_v2_json(
                {
                    "slot_identity": accepted_slot["slot_identity"],
                    "trajectory_identity": "trajectory:audit",
                    "shots": shots,
                },
                trajectory_root / "trajectory.json",
            )
            records = []
            for slot in plan["lineages"]:
                if slot is accepted_slot:
                    records.append({
                        "slot_identity": slot["slot_identity"],
                        "status": "accepted",
                        "trajectory_identity": "trajectory:audit",
                        "frame_record_count": 4,
                        "terminal_reason": "shot_limit",
                        "executed_action_count": 2,
                    })
                else:
                    records.append({
                        "slot_identity": slot["slot_identity"],
                        "status": "failed",
                        "failures": [{"message": "fixture failure"}],
                    })
            observed = []

            def encode(frames, video):
                observed.extend(path.as_posix() for path in frames)
                video.parent.mkdir(parents=True, exist_ok=True)
                video.write_bytes(b"webm")

            with patch(
                "scripts.run_issue_62_successor_cohort._encode_agent_frames_webm",
                side_effect=encode,
            ):
                manifest = write_pilot_audit(
                    plan,
                    records,
                    {"identity": "pilot-report:audit"},
                    runtime_root=runtime,
                    output=output,
                )

            self.assertEqual(len(observed), 4)
            self.assertIn("shot-000", observed[0])
            self.assertIn("shot-000", observed[1])
            self.assertIn("shot-001", observed[2])
            self.assertIn("shot-001", observed[3])
            self.assertEqual(manifest["video_count"], 1)
            self.assertFalse(manifest["canonical_observations_included"])
            self.assertEqual(
                manifest["videos"][0]["shot_ranges"][1][
                    "video_frame_start"
                ],
                2,
            )
            self.assertIn("<video controls", (output / "index.html").read_text())
            self.assertTrue((output / manifest["videos"][0]["path"]).is_file())

    def test_webm_encoder_produces_vp8_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = []
            for index, color in enumerate(((255, 0, 0), (0, 0, 255))):
                path = root / f"source-{index}.png"
                Image.new("RGB", (32, 24), color).save(path)
                frames.append(path)
            output = root / "audit.webm"

            _encode_agent_frames_webm(frames, output)

            codec = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(codec, "vp8")


class SuccessorCohortResumeTests(unittest.TestCase):
    def test_revision_can_roll_over_before_any_lineage_is_accepted(self) -> None:
        plan = build_pilot_plan()
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            (runtime / "game-runtime").mkdir()
            write_immutable_cohort_v2_json(
                {
                    "schema": "issue_62_collection_provenance_v1",
                    "implementation_commit": "commit:before-audit",
                    "player": {"source_snapshot_commit": "player:fixture"},
                    "collected_at": "2026-08-30T06:38:59Z",
                    "final_evaluation_opened": False,
                },
                runtime / "provenance.json",
            )
            for slot in plan["lineages"]:
                write_immutable_cohort_v2_json(
                    {
                        "schema": "issue_62_lineage_collection_result_v1",
                        "slot_identity": slot["slot_identity"],
                        "status": "failed",
                        "exposure_role": slot["exposure_role"],
                        "attempt_count": 2,
                    },
                    runtime / "records" / f"{slot['slot_identity']}.json",
                )

            with patch(
                "scripts.run_issue_62_successor_cohort._player",
                return_value={"source_snapshot_commit": "player:fixture"},
            ):
                records = run_collection(
                    plan,
                    runtime_root=runtime,
                    implementation_commit="commit:after-audit",
                    start_display_process=False,
                    speed=50,
                    headless=False,
                )

            provenance = json.loads((runtime / "provenance.json").read_bytes())
            self.assertEqual(len(records), 36)
            self.assertEqual(
                provenance["implementation_commit"], "commit:after-audit"
            )
            self.assertEqual(
                provenance["superseded_pre_acceptance_implementation_commits"],
                ["commit:before-audit"],
            )

    def test_revision_change_still_fails_after_an_accepted_lineage(self) -> None:
        plan = build_pilot_plan()
        first = plan["lineages"][0]
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            write_immutable_cohort_v2_json(
                {
                    "schema": "issue_62_collection_provenance_v1",
                    "implementation_commit": "commit:old",
                    "player": {"source_snapshot_commit": "player:fixture"},
                    "collected_at": "2026-08-30T06:38:59Z",
                    "final_evaluation_opened": False,
                },
                runtime / "provenance.json",
            )
            write_immutable_cohort_v2_json(
                {
                    "schema": "issue_62_lineage_collection_result_v1",
                    "slot_identity": first["slot_identity"],
                    "status": "accepted",
                    "exposure_role": first["exposure_role"],
                    "attempt_count": 1,
                },
                runtime / "records" / f"{first['slot_identity']}.json",
            )

            with patch(
                "scripts.run_issue_62_successor_cohort._player",
                return_value={"source_snapshot_commit": "player:fixture"},
            ), self.assertRaisesRegex(
                SuccessorCohortError, "revision differs after an accepted lineage"
            ):
                run_collection(
                    plan,
                    runtime_root=runtime,
                    implementation_commit="commit:new",
                    start_display_process=False,
                    speed=50,
                    headless=False,
                )

    def test_interrupted_attempt_directories_are_counted_without_overwrite(self) -> None:
        plan = build_pilot_plan()
        interrupted = plan["lineages"][0]
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            (runtime / "game-runtime").mkdir()
            write_immutable_cohort_v2_json(
                {
                    "schema": "issue_62_collection_provenance_v2",
                    "implementation_commit": "commit:same",
                    "player": {"source_snapshot_commit": "player:fixture"},
                    "collected_at": "2026-08-30T06:38:59Z",
                    "final_evaluation_opened": False,
                    "superseded_pre_acceptance_implementation_commits": [],
                },
                runtime / "provenance.json",
            )
            for attempt_number in (1, 2):
                (
                    runtime / "attempts" / interrupted["slot_identity"]
                    / f"attempt-{attempt_number:02d}"
                ).mkdir(parents=True)
            for slot in plan["lineages"][1:]:
                write_immutable_cohort_v2_json(
                    {
                        "schema": "issue_62_lineage_collection_result_v1",
                        "slot_identity": slot["slot_identity"],
                        "status": "failed",
                        "exposure_role": slot["exposure_role"],
                        "attempt_count": 2,
                    },
                    runtime / "records" / f"{slot['slot_identity']}.json",
                )

            with patch(
                "scripts.run_issue_62_successor_cohort._player",
                return_value={"source_snapshot_commit": "player:fixture"},
            ), patch(
                "scripts.run_issue_62_successor_cohort._collect_lineage_attempt"
            ) as collect:
                records = run_collection(
                    plan,
                    runtime_root=runtime,
                    implementation_commit="commit:same",
                    start_display_process=False,
                    speed=50,
                    headless=False,
                )

            collect.assert_not_called()
            result = next(
                item for item in records
                if item["slot_identity"] == interrupted["slot_identity"]
            )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(
                [item["error_type"] for item in result["failures"]],
                ["InterruptedAttempt", "InterruptedAttempt"],
            )


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
