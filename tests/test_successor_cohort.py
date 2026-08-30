from __future__ import annotations

import copy
from concurrent.futures import Future
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
    DEFAULT_PRODUCTION_WORKERS,
    _bounds,
    _encode_agent_frames_webm,
    _interaction_coverage,
    _pending_lineage_shards,
    _materialize_slot,
    _materialized_bird_count,
    _pilot_report,
    _parser,
    _resolve_planned_interface_action,
    _shot_record,
    _synthetic_pilot_records,
    _terminal_status,
    _trajectory_record,
    dry_run,
    main,
    run_parallel_collection,
    run_collection,
    run_pilot,
    write_attempt_audit,
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
    def test_production_cli_defaults_to_four_workers(self) -> None:
        args = _parser().parse_args(["--run-production"])

        self.assertEqual(args.production_workers, DEFAULT_PRODUCTION_WORKERS)

    def test_production_cli_uses_parallel_collection_without_an_extra_flag(self) -> None:
        plan = build_pilot_plan()
        with patch(
            "scripts.run_issue_62_successor_cohort._load_plan", return_value=plan
        ), patch(
            "scripts.run_issue_62_successor_cohort._implementation_commit",
            return_value="commit:fixture",
        ), patch(
            "scripts.run_issue_62_successor_cohort.run_parallel_collection",
            return_value=[],
        ) as collect, patch("builtins.print"):
            result = main(["--run-production", "--start-display"])

        self.assertEqual(result, 0)
        self.assertEqual(
            collect.call_args.kwargs["worker_count"], DEFAULT_PRODUCTION_WORKERS
        )
        self.assertTrue(collect.call_args.kwargs["start_display_process"])

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
            {
                "uniform_random",
                "stratified_bounds",
                "trajectory_guided_direct_pig",
            },
        )
        self.assertEqual(
            len({item["generation_seed"] for item in first["lineages"]}), 36
        )
        expected_action_counts = {
            ("type010101", "uniform_random"): 1,
            ("type010101", "stratified_bounds"): 1,
            ("type010101", "trajectory_guided_direct_pig"): 1,
            ("type010102", "uniform_random"): 3,
            ("type010102", "stratified_bounds"): 3,
            ("type010102", "trajectory_guided_direct_pig"): 2,
        }
        self.assertTrue(all(
            len(item["planned_actions"])
            == expected_action_counts[
                (item["generator_family"], item["behavior_policy"])
            ]
            for item in first["lineages"]
        ))
        for role in PUBLIC_ROLES:
            self.assertTrue(any(
                item["exposure_role"] == role
                and len(item["planned_actions"]) > 1
                for item in first["lineages"]
            ))
        stratified = {
            action["action_stratum"]
            for slot in first["lineages"]
            if slot["behavior_policy"] == "stratified_bounds"
            for action in slot["planned_actions"]
        }
        self.assertEqual(len(stratified), 12)
        for role in PUBLIC_ROLES:
            counts = {
                policy: sum(
                    item["exposure_role"] == role
                    and item["behavior_policy"] == policy
                    for item in first["lineages"]
                )
                for policy in {
                    "uniform_random",
                    "stratified_bounds",
                    "trajectory_guided_direct_pig",
                }
            }
            self.assertEqual(set(counts.values()), {4})
        guided_actions = [
            action
            for slot in first["lineages"]
            if slot["behavior_policy"] == "trajectory_guided_direct_pig"
            for action in slot["planned_actions"]
        ]
        self.assertTrue(guided_actions)
        self.assertTrue(all(
            action["selection_mode"] == "trajectory_guided_direct_pig_clearance"
            and action["trajectory_arc"] == "lowest_clear_full_pull"
            and action["minimum_pull"] == "trajectory_drag_radius"
            and action["target_kind"] == "pig"
            and action["aim_point"] == "visible_polygon_upper_edge"
            and action["clearance_model"]
            == "near_target_margin_inflated_obstacles"
            and action["bird_radius_world"] == 0.17
            and action["clearance_margin_world"] == 0.34
            and action["clearance_margin_minimum_target_distance_world"] == 8.0
            for action in guided_actions
        ))

    def test_authoritative_physics_terminal_overrides_lagging_game_state(self) -> None:
        self.assertEqual(
            _terminal_status(
                GameState.EVALUATION_TERMINATED,
                exhausted=True,
                physics_terminal_reason="level_clear",
            ),
            "success",
        )
        self.assertEqual(
            _terminal_status(
                GameState.PLAYING,
                exhausted=False,
                physics_terminal_reason="level_fail",
            ),
            "failure",
        )

    def test_guided_action_resolves_from_live_visible_pig_geometry(self) -> None:
        plan = build_pilot_plan()
        planned = next(
            action
            for slot in plan["lineages"]
            if slot["behavior_policy"] == "trajectory_guided_direct_pig"
            for action in slot["planned_actions"]
        )
        symbolic_state = [{
            "features": [
                {
                    "properties": {"id": "bird-1", "label": "redBird"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [94.0, 294.0], [106.0, 294.0],
                            [106.0, 306.0], [94.0, 306.0],
                        ]],
                    },
                },
                {
                    "properties": {"id": "pig-1", "label": "BasicSmallPig"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [292.0, 212.0], [308.0, 212.0],
                            [308.0, 228.0], [292.0, 228.0],
                        ]],
                    },
                },
            ],
        }]
        bridge = SimpleNamespace(
            get_symbolic_state_without_screenshot=lambda: symbolic_state
        )
        action = _resolve_planned_interface_action(
            planned,
            {
                "gameX": 100.0,
                "gameY": 180.0,
                "canvasX": 100.0,
                "canvasY": 300.0,
                "pixelsPerWorldUnit": 32.0,
            },
            bridge,
        )
        self.assertGreater(action["drag_release"][1], 0)
        self.assertEqual(
            action["selection_evidence"]["target_id"], "pig-1"
        )
        self.assertLess(
            action["selection_evidence"]["predicted_miss_pixels"], 1.0
        )
        capture = SimpleNamespace(
            capture_id="capture-v2:fixture",
            shot_id="shot-v2:fixture",
            record={
                "frame_records": [{"fixed_step": 1}],
                "terminal_evidence": {"reason": "stable_entered"},
            },
        )
        with (
            patch(
                "scripts.run_issue_62_successor_cohort.load_physics_capture_v2",
                return_value=capture,
            ),
            patch(
                "scripts.run_issue_62_successor_cohort.validate_observation_trace",
                return_value={"identity": "observation-trace:fixture"},
            ),
        ):
            shot = _shot_record(
                Path("/unused"),
                shot_index=0,
                planned_action=planned,
                prepared=SimpleNamespace(action=action),
                state=GameState.PLAYING,
                derivations=[],
            )
        self.assertNotIn(
            "selection_evidence", shot["action"]["interface_action"]
        )
        self.assertEqual(
            shot["action_selection_evidence"]["target_id"], "pig-1"
        )

    def test_pilot_gate_requires_pig_hits_and_support_changes_in_every_role(self) -> None:
        plan = build_pilot_plan()
        records = _synthetic_pilot_records(plan)
        self.assertTrue(_pilot_report(plan, records)["passed"])
        for record in records:
            if record["exposure_role"] == "model_selection":
                record["interaction_coverage"] = [
                    value
                    for value in record["interaction_coverage"]
                    if value != "collision:bird:pig"
                ]
        self.assertFalse(_pilot_report(plan, records)["passed"])

    def test_interaction_coverage_classifies_pig_hit_and_support_change(self) -> None:
        capture = SimpleNamespace(record={
            "events": [{
                "event_type": "collision",
                "participants": ["runtime:pig:0000", "runtime:bird:0001"],
            }],
            "fixed_step_samples": [
                {"supports": [{
                    "supporter_entity_id": "runtime:platform:0000",
                    "supported_entity_id": "runtime:pig:0000",
                }]},
                {"supports": []},
            ],
        })
        with patch(
            "scripts.run_issue_62_successor_cohort.load_physics_capture_v2",
            return_value=capture,
        ):
            coverage = _interaction_coverage(
                Path("/unused"), {"shots": [{"path": "shots/shot-000"}]}
            )
        self.assertIn("collision:bird:pig", coverage)
        self.assertIn("non_bird_support_change", coverage)

    def test_action_plan_never_exceeds_materialized_birds(self) -> None:
        plan = build_pilot_plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for family in ("type010101", "type010102"):
                slot = next(
                    item for item in plan["lineages"]
                    if item["generator_family"] == family
                )
                authority = _materialize_slot(slot, root / family)
                self.assertLessEqual(
                    len(slot["planned_actions"]),
                    _materialized_bird_count(authority["xml_path"]),
                )

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

    def test_pilot_gate_rejects_one_exhausted_frozen_lineage(self) -> None:
        pilot = build_pilot_plan()
        records = _synthetic_pilot_records(pilot)
        records[0]["status"] = "failed"

        self.assertFalse(_pilot_report(pilot, records)["passed"])

    def test_no_write_dry_run_wires_the_real_carrier_boundary(self) -> None:
        lineages = iter(("lineage:a", "lineage:b"))

        def authority(slot, root):
            root.mkdir(parents=True)
            bird_count = {"type010101": 1, "type010102": 3}[
                slot["generator_family"]
            ]
            xml_path = root / "scenario.xml"
            xml_path.write_text(
                "<Level><Birds>"
                + '<Bird type="BirdRed"/>' * bird_count
                + "</Birds></Level>"
            )
            return {
                "xml_path": xml_path,
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
        self.assertEqual(
            result["default_production_workers"], DEFAULT_PRODUCTION_WORKERS
        )
        self.assertFalse(result["final_evaluation_opened"])
        self.assertFalse(result["files_written"])
        self.assertTrue(all("python -u -m" in item for item in result["actual_commands"]))
        self.assertIn("WebM", result["pilot_audit_format"])

    def test_review_prefix_stops_cleanly_without_publishing_a_report(self) -> None:
        plan = build_pilot_plan()
        records = [
            {
                "slot_identity": slot["slot_identity"],
                "status": "accepted",
            }
            for slot in plan["lineages"][:2]
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "pilot-plan.json"
            write_immutable_cohort_v2_json(plan, plan_path)
            with patch(
                "scripts.run_issue_62_successor_cohort.run_collection",
                return_value=records,
            ) as collect, patch(
                "scripts.run_issue_62_successor_cohort._pilot_report"
            ) as report, patch(
                "scripts.run_issue_62_successor_cohort.write_pilot_audit"
            ) as final_audit:
                result = run_pilot(
                    plan_path=plan_path,
                    runtime_root=root / "runtime",
                    report_path=root / "pilot-report.json",
                    audit_output=root / "audit",
                    implementation_commit="commit:fixture",
                    start_display_process=False,
                    speed=50,
                    headless=False,
                    lineage_limit=2,
                )

            self.assertEqual(
                result["schema"], "issue_62_pilot_review_prefix_v1"
            )
            self.assertEqual(result["collected_lineage_count"], 2)
            self.assertEqual(collect.call_args.kwargs["lineage_limit"], 2)
            report.assert_not_called()
            final_audit.assert_not_called()


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
            frame_rate = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=r_frame_rate",
                    "-of", "default=nw=1:nk=1", str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(frame_rate, "50/1")

    def test_failed_attempt_audit_is_immediate_and_caps_stalled_frames(self) -> None:
        slot = build_pilot_plan()["lineages"][0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = root / "runtime/attempt-01"
            output = root / "data/issue-62-pilot-audit"
            self._observation_trace(
                attempt / "shots/shot-000/observation-trace",
                capture_id="capture:attempt-audit",
                fixed_steps=(10, 11),
                role=slot["exposure_role"],
                rollout_identity="rollout:attempt-audit",
            )
            stalled = attempt / ".aligned-observation-current"
            stalled.mkdir(parents=True)
            for index in range(255):
                Image.new("RGB", (8, 8), (index % 255, 0, 0)).save(
                    stalled / f"frame_{index:06d}.png"
                )
            observed = []

            def encode(frames, video):
                observed.extend(frames)
                video.parent.mkdir(parents=True, exist_ok=True)
                video.write_bytes(b"webm")

            with patch(
                "scripts.run_issue_62_successor_cohort._encode_agent_frames_webm",
                side_effect=encode,
            ):
                manifest = write_attempt_audit(
                    slot,
                    1,
                    attempt,
                    output=output,
                    status="failed",
                    failure={"error_type": "PhysicsCaptureV2Failure"},
                )

            self.assertEqual(len(observed), 252)
            self.assertIn("shot-000", observed[0].as_posix())
            self.assertIn("shot-000", observed[1].as_posix())
            self.assertEqual(observed[2], stalled / "frame_000000.png")
            self.assertEqual(observed[-1], stalled / "frame_000254.png")
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["finalized_frame_count"], 2)
            self.assertEqual(manifest["stalled_source_frame_count"], 255)
            self.assertEqual(manifest["video_frame_count"], 252)
            self.assertTrue((output / manifest["video_path"]).is_file())


class SuccessorCohortResumeTests(unittest.TestCase):
    def test_pending_lineages_are_balanced_across_deterministic_worker_shards(self) -> None:
        plan = build_pilot_plan()
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            completed = {0, 5}
            for ordinal in completed:
                path = runtime / "records" / (
                    f"{plan['lineages'][ordinal]['slot_identity']}.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")

            shards = _pending_lineage_shards(plan, runtime, 4)

        flattened = [ordinal for shard in shards for ordinal in shard]
        self.assertEqual(
            sorted(flattened),
            [ordinal for ordinal in range(36) if ordinal not in completed],
        )
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertLessEqual(max(map(len, shards)) - min(map(len, shards)), 1)
        pending = [ordinal for ordinal in range(36) if ordinal not in completed]
        self.assertEqual(
            shards,
            tuple(tuple(pending[worker::4]) for worker in range(4)),
        )

    def test_parallel_collection_isolates_game_runtimes_and_reuses_one_display(self) -> None:
        plan = build_pilot_plan()
        selected = tuple(range(6))
        calls = []

        class InlineExecutor:
            def __init__(self, *, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def submit(self, function, *args, **kwargs):
                future = Future()
                try:
                    future.set_result(function(*args, **kwargs))
                except Exception as error:
                    future.set_exception(error)
                return future

        def collect(_plan, **kwargs):
            calls.append(kwargs)
            return [
                {
                    "schema": "issue_62_lineage_collection_result_v1",
                    "slot_identity": plan["lineages"][ordinal]["slot_identity"],
                    "exposure_role": plan["lineages"][ordinal]["exposure_role"],
                    "status": "accepted",
                    "attempt_count": 1,
                }
                for ordinal in kwargs["lineage_ordinals"]
            ]

        with tempfile.TemporaryDirectory() as temporary, patch(
            "scripts.run_issue_62_successor_cohort._initialize_collection_runtime"
        ), patch(
            "scripts.run_issue_62_successor_cohort.ProcessPoolExecutor",
            InlineExecutor,
        ), patch(
            "scripts.run_issue_62_successor_cohort.run_collection",
            side_effect=collect,
        ), patch(
            "scripts.run_issue_62_successor_cohort.start_display",
            return_value=(":fixture", SimpleNamespace()),
        ) as display, patch(
            "scripts.run_issue_62_successor_cohort.terminate", return_value=0
        ) as terminate_display:
            records = run_parallel_collection(
                plan,
                runtime_root=Path(temporary),
                implementation_commit="commit:fixture",
                start_display_process=True,
                speed=50,
                headless=False,
                worker_count=DEFAULT_PRODUCTION_WORKERS,
                lineage_ordinals=selected,
            )

        self.assertEqual([item["slot_identity"] for item in records], [
            plan["lineages"][ordinal]["slot_identity"] for ordinal in selected
        ])
        self.assertEqual(len(calls), 4)
        self.assertEqual(
            {ordinal for call in calls for ordinal in call["lineage_ordinals"]},
            set(selected),
        )
        self.assertEqual(len({call["game_runtime"] for call in calls}), 4)
        self.assertTrue(all(not call["start_display_process"] for call in calls))
        self.assertEqual(
            {call["engine_start_lock"] for call in calls},
            {Path(temporary).resolve() / "engine-start.lock"},
        )
        display.assert_called_once()
        terminate_display.assert_called_once()

    def test_each_new_failed_attempt_is_audited_immediately(self) -> None:
        plan = build_pilot_plan()
        first = plan["lineages"][0]
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
            def fail_attempt(_slot, attempt_root, *_args, **_kwargs):
                attempt_root.mkdir(parents=True)
                raise RuntimeError("fixture capture failure")

            with patch(
                "scripts.run_issue_62_successor_cohort._player",
                return_value={"source_snapshot_commit": "player:fixture"},
            ), patch(
                "scripts.run_issue_62_successor_cohort._collect_lineage_attempt",
                side_effect=fail_attempt,
            ) as collect, patch(
                "scripts.run_issue_62_successor_cohort.write_attempt_audit",
                return_value={},
            ) as audit:
                records = run_collection(
                    plan,
                    runtime_root=runtime,
                    implementation_commit="commit:same",
                    start_display_process=False,
                    speed=50,
                    headless=False,
                    attempt_audit_output=runtime / "audit",
                    lineage_limit=1,
                )

            self.assertEqual(collect.call_count, 2)
            self.assertEqual(audit.call_count, 2)
            self.assertEqual(len(records), 1)
            self.assertTrue(all(
                call.kwargs["status"] == "failed"
                for call in audit.call_args_list
            ))
            result = next(
                item for item in records
                if item["slot_identity"] == first["slot_identity"]
            )
            self.assertEqual(result["status"], "failed")

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
