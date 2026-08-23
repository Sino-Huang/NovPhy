from __future__ import annotations

import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from scripts.cohort_v2_release import (
    _expected_slots,
    production_intervention_identity,
)
from scripts.cohort_v2_production_plans import COLLECTION_IDENTITY, PARAMETER_IDENTITY
from scripts.collect_rollouts import action_to_shot
from scripts.final_evaluation_access import (
    FinalEvaluationWorkflowAccessManifest,
    authorize_final_evaluation_workflow_access,
)
from src.webui.issue_53_review import Issue53ReviewError, Issue53ReviewSession
from src.webui import issue_53_review


ROOT = Path(__file__).resolve().parents[1]
MISMATCHES = {
    ("training", "central-destruction"),
    ("calibration", "central-no-contact-miss"),
    ("calibration", "central-persistent-support"),
    ("calibration", "central-destruction"),
    ("model_selection", "central-destruction"),
    ("final_evaluation", "central-no-contact-miss"),
    ("final_evaluation", "central-persistent-support"),
    ("final_evaluation", "central-destruction"),
}


class Issue53ReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.production = self.root / "production"
        self.output = self.root / "review"
        self.production.mkdir()
        self._write_production_fixture()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_production_fixture(self) -> None:
        collection_source = ROOT / "data/runtime_evidence/issue-52/collection-plan.json"
        collection = json.loads(collection_source.read_text(encoding="utf-8"))
        (self.production / "frozen-collection-plan.json").write_bytes(
            collection_source.read_bytes()
        )
        (self.production / "frozen-production-parameter-plan.json").write_bytes(
            (ROOT / "data/runtime_evidence/issue-52/production-parameter-plan.json").read_bytes()
        )
        artifact = self.production / "retained/attempt/shot_001"
        artifact.mkdir(parents=True)
        (artifact / "physics_capture_v2.json").write_bytes(
            (ROOT / "data/runtime_evidence/issue-44/captures/support-change.json").read_bytes()
        )
        action = {
            "action_type": "drag_hold_release",
            "coordinate_frame": "slingshot_relative",
            "drag_start": [127, 223],
            "drag_release": [-77, 30],
            "frame_height": 480,
            "releaseTime": 1000,
            "tapTime": 0,
            "slingshot_reference": {
                "gameX": 127,
                "gameY": 223,
                "canvasX": 127,
                "canvasY": 256,
            },
        }
        shot = action_to_shot(action, frame_height=480)
        (artifact.parent / "manifest.json").write_text(
            json.dumps({"rollouts": [{"action": action}]}), encoding="utf-8"
        )
        (artifact.parent / "action_log.json").write_text(
            json.dumps({"accepted_trials": [{"shot": shot}]}), encoding="utf-8"
        )
        ledger = []
        quality_mismatches = []
        for slot in _expected_slots(collection):
            intervention = slot["intervention"]
            mismatch = (slot["exposure_role"], intervention["id"]) in MISMATCHES
            terminal = "stable_entered" if mismatch else intervention["intended_termination_class"]
            entry = {
                "attempt_id": slot["attempt_id"],
                "exposure_role": slot["exposure_role"],
                "dataset_partition": slot["dataset_partition"],
                "scenario_manifest_identity": slot["scenario_manifest_identity"],
                "scenario_lineage_identity": slot["scenario_lineage_identity"],
                "level_instance_identity": slot["level_instance_identity"],
                "scenario_template_identity": slot["scenario_template_identity"],
                "benchmark_condition_identity": slot["benchmark_condition_identity"],
                "intervention_id": intervention["id"],
                "intervention_identity": production_intervention_identity(intervention["id"]),
                "intervention_ordinal": intervention["ordinal"],
                "intervention_source": intervention["intervention_source"],
                "intended_coverage_stratum": intervention["intended_coverage_stratum"],
                "expected_termination": intervention["intended_termination_class"],
                "status": "accepted",
                "reason": None,
                "failure_code": None,
                "artifact_path": str(artifact),
                "quarantine_path": None,
                "failure_manifest_path": None,
                "realized_coverage_strata": [intervention["intended_coverage_stratum"]],
                "terminal_reason": terminal,
                "terminal_span_fixed_steps": 2,
                "attempt_number": 1,
                "retry_decision": "none",
            }
            ledger.append(entry)
            if mismatch:
                quality_mismatches.append(
                    {
                        "attempt_id": slot["attempt_id"],
                        "expected": intervention["intended_termination_class"],
                        "observed": terminal,
                    }
                )
        execution = {
            "schema": "issue_53_production_execution_report_v1",
            "collection_plan_identity": COLLECTION_IDENTITY,
            "production_parameter_plan_identity": PARAMETER_IDENTITY,
            "attempt_ledger": ledger,
            "counts": {
                "planned": 24,
                "attempted": 24,
                "accepted": 24,
                "rejected": 0,
                "failed": 0,
                "quarantined": 0,
            },
            "retry_count": 0,
            "outcome_independent_accounting": True,
        }
        (self.production / "production-execution-report.json").write_text(
            json.dumps(execution), encoding="utf-8"
        )
        (self.production / "production-quality-report.json").write_text(
            json.dumps(
                {
                    "schema": "cohort_v2_production_quality_report_v1",
                    "collection_plan_identity": COLLECTION_IDENTITY,
                    "production_parameter_plan_identity": PARAMETER_IDENTITY,
                    "termination_mismatches": quality_mismatches,
                }
            ),
            encoding="utf-8",
        )
        pending = FinalEvaluationWorkflowAccessManifest.from_dict(
            json.loads(
                (
                    ROOT
                    / "data/runtime_evidence/issue-47/final-evaluation-workflow-access-manifest.json"
                ).read_text(encoding="utf-8")
            )
        )
        authorized = authorize_final_evaluation_workflow_access(
            pending,
            authorization_identity="authorization:test-review",
            authorized_at="2026-08-23T00:00:00Z",
        )
        authorities = self.production / "authorities"
        authorities.mkdir()
        (authorities / "authorized-final-access-manifest.json").write_text(
            json.dumps(authorized.to_dict()), encoding="utf-8"
        )

    def session(self, *, replay_runner=None) -> Issue53ReviewSession:
        return Issue53ReviewSession(
            self.production,
            self.output,
            repository_root=ROOT,
            replay_runner=replay_runner,
        )

    def test_loader_finds_exactly_eight_and_never_changes_source_reports(self) -> None:
        sources = {
            path: path.read_bytes()
            for path in (
                self.production / "production-execution-report.json",
                self.production / "production-quality-report.json",
                self.production / "frozen-collection-plan.json",
                self.production / "frozen-production-parameter-plan.json",
            )
        }
        session = self.session()
        snapshot = session.snapshot()

        self.assertEqual(snapshot["mismatchCount"], 8)
        self.assertEqual(snapshot["schema"], "issue_53_human_review_v2")
        self.assertEqual(
            snapshot["screenCoordinateAlignmentContract"]["retained_anchor_tolerance_pixels"],
            2.0,
        )
        self.assertEqual(sum(item["locked"] for item in snapshot["playlist"]), 3)
        self.assertEqual(sources, {path: path.read_bytes() for path in sources})

    def test_fixed_step_playback_is_paginated_and_explains_stable_termination(self) -> None:
        session = self.session()
        detail = session.open_trace(0)
        page = session.fixed_steps(0, start=0, count=1)

        self.assertEqual(page["count"], 1)
        self.assertGreaterEqual(page["total"], 2)
        self.assertIn("entities", page["steps"][0])
        self.assertIn("contacts", page["steps"][0])
        self.assertIn("supports", page["steps"][0])
        self.assertIn("events", page["steps"][0])
        self.assertIn("does not mean the level was won", detail["terminationExplanation"])
        with self.assertRaisesRegex(Issue53ReviewError, "1 <= count <= 120"):
            session.fixed_steps(0, start=0, count=121)

    def test_replay_is_diagnostic_only_and_limited_to_one_attempt(self) -> None:
        def replay(item, root, speed):
            video = root / "fixture.mp4"
            video.write_bytes(b"video")
            video.with_suffix(".webm").write_bytes(b"webm-video")
            alignment = {
                "status": "ready",
                "startup_speed": 50,
                "execution_speed": 1,
                "alignment_contract": {"anchor_tolerance_pixels": 2.0},
            }
            return {
                "video_path": str(video),
                "speed": speed,
                "screen_coordinate_alignment": alignment,
                "comparison": {
                    "passed": True,
                    "components": [{
                        "component": "screen_coordinate_alignment",
                        "status": "equality",
                        "details": alignment,
                    }],
                },
            }

        session = self.session(replay_runner=replay)
        session.open_trace(0)
        detail = session.run_replay(0)
        result = detail["item"]["replay"]

        self.assertTrue(result["diagnostic_only"])
        self.assertFalse(result["cohort_quota_eligible"])
        self.assertFalse(result["production_accounting_eligible"])
        self.assertFalse(result["resampling_eligible"])
        self.assertFalse(result["release_eligible"])
        self.assertEqual(session.replay_video(0).read_bytes(), b"webm-video")
        with self.assertRaisesRegex(Issue53ReviewError, "one diagnostic replay"):
            session.run_replay(0)

    def test_browser_video_encoder_produces_vp8_webm(self) -> None:
        frames = self.root / "frames"
        frames.mkdir()
        for index, color in enumerate(((255, 0, 0), (0, 0, 255))):
            Image.new("RGB", (32, 24), color).save(
                frames / f"frame_{index:06d}.png"
            )
        output = self.root / "replay.webm"

        issue_53_review._encode_browser_video(frames, output)

        discovered = subprocess.run(
            ["gst-discoverer-1.0", str(output)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertIn("WebM", discovered)
        self.assertIn("VP8", discovered)

    def test_replay_without_alignment_evidence_fails_closed(self) -> None:
        def replay(_item, root, _speed):
            video = root / "fixture.webm"
            video.write_bytes(b"video")
            return {
                "video_path": str(video),
                "comparison": {"passed": True, "components": []},
            }

        session = self.session(replay_runner=replay)
        session.open_trace(0)
        with self.assertRaisesRegex(Issue53ReviewError, "alignment evidence"):
            session.run_replay(0)
        self.assertFalse((session._replay_root(session.items[0]) / "result.json").exists())

    def test_final_items_require_authorization_and_write_only_sealed_outputs(self) -> None:
        session = self.session()
        with self.assertRaises(PermissionError):
            session.open_trace(5)
        with self.assertRaises(PermissionError):
            session.authorize_final_access("wrong")

        session.authorize_final_access("authorization:test-review")
        session.open_trace(5)
        session.record_decision(
            5,
            decision="confirmed_mismatch",
            notes="Retained trace ends in stable_entered.",
            reviewer="operator",
        )

        self.assertTrue((self.output / "sealed/access/review-access-audit.json").is_file())
        final_decisions = list((self.output / "sealed/items").rglob("decision.json"))
        self.assertEqual(len(final_decisions), 1)
        self.assertFalse(any((self.output / "items").rglob("*final_evaluation*")))

    def test_all_confirmed_decisions_finalize_a_separate_incomplete_review_bundle(self) -> None:
        session = self.session()
        session.authorize_final_access("authorization:test-review")
        for index in range(8):
            session.open_trace(index)
            session.record_decision(
                index,
                decision="confirmed_mismatch",
                notes="Confirmed from retained trace.",
                reviewer="operator",
            )

        snapshot = session.snapshot()
        self.assertEqual(snapshot["followUpDecision"]["outcome"], "confirmed_plan_failure")
        self.assertFalse(snapshot["followUpDecision"]["productionReleaseMayPass"])
        self.assertTrue((self.output / "human-review-summary.json").is_file())
        self.assertTrue((self.output / "sealed/human-review-report.json").is_file())
        self.assertTrue((self.output / "sealed/sealed-review-manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
