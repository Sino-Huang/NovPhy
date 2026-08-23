from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.capture_issue_53_evidence import build_parser, dry_run
from scripts.collect_rollouts import capture_physics_v2_rollout, collect_rollouts
from scripts.cohort_v2_micro_relations import (
    CohortV2MicroRelationError,
    derive_capture_micro_relations,
    validate_capture_micro_relation_derivation,
)
from scripts.cohort_v2_release import (
    _expected_slots,
    _public_quality_report,
    _quality_report,
    production_intervention_identity,
    publish_issue_53_evidence,
    validate_issue_53_execution_report,
)
from scripts.cohort_v2_production_plans import (
    COLLECTION_IDENTITY,
    PARAMETER_IDENTITY,
)
from scripts.cohort_v2_production_plans import derive_issue_52_payloads
from scripts.cohort_v2_production_plans_v2 import (
    COLLECTION_IDENTITY as COLLECTION_IDENTITY_V2,
    DEFAULT_PLAN_ROOT as PLAN_ROOT_V2,
    PARAMETER_IDENTITY as PARAMETER_IDENTITY_V2,
    validate_plan_v2_evidence,
)
from scripts.cohort_v2_production_plans_v3 import (
    COLLECTION_IDENTITY as COLLECTION_IDENTITY_V3,
    DEFAULT_PLAN_ROOT as PLAN_ROOT_V3,
    PARAMETER_IDENTITY as PARAMETER_IDENTITY_V3,
    validate_plan_v3_evidence,
)
from scripts.cohort_v2_production_plans_v4 import (
    COLLECTION_IDENTITY as COLLECTION_IDENTITY_V4,
    DEFAULT_PLAN_ROOT as PLAN_ROOT_V4,
    PARAMETER_IDENTITY as PARAMETER_IDENTITY_V4,
    validate_plan_v4_evidence,
)
from scripts.cohort_v2_production_plans_v5 import (
    COLLECTION_IDENTITY as COLLECTION_IDENTITY_V5,
    DEFAULT_PLAN_ROOT as PLAN_ROOT_V5,
    PARAMETER_IDENTITY as PARAMETER_IDENTITY_V5,
    validate_plan_v5_evidence,
)
from scripts.final_evaluation_access import (
    FinalEvaluationWorkflowAccessManifest,
    authorize_final_evaluation_workflow_access,
)
from scripts.physics_capture_v2 import load_physics_capture_v2


ROOT = Path(__file__).resolve().parents[1]


class Issue53EvidenceTests(unittest.TestCase):
    def _collection_plan(self):
        return __import__("json").loads(
            (
                ROOT / "data/runtime_evidence/issue-52/collection-plan.json"
            ).read_text(encoding="utf-8")
        )

    def _execution_report(self):
        collection = self._collection_plan()
        ledger = []
        for slot in _expected_slots(collection):
            intervention = slot["intervention"]
            ledger.append(
                {
                    "attempt_id": slot["attempt_id"],
                    "exposure_role": slot["exposure_role"],
                    "dataset_partition": slot["dataset_partition"],
                    "scenario_manifest_identity": slot[
                        "scenario_manifest_identity"
                    ],
                    "scenario_lineage_identity": slot[
                        "scenario_lineage_identity"
                    ],
                    "level_instance_identity": slot["level_instance_identity"],
                    "scenario_template_identity": slot[
                        "scenario_template_identity"
                    ],
                    "benchmark_condition_identity": slot[
                        "benchmark_condition_identity"
                    ],
                    "intervention_id": intervention["id"],
                    "intervention_identity": production_intervention_identity(
                        intervention["id"]
                    ),
                    "intervention_source": intervention["intervention_source"],
                    "intended_coverage_stratum": intervention[
                        "intended_coverage_stratum"
                    ],
                    "expected_termination": intervention[
                        "intended_termination_class"
                    ],
                    "status": "accepted",
                    "artifact_path": f"runtime/{slot['attempt_id']}",
                }
            )
        return collection, {
            "schema": "issue_53_production_execution_report_v1",
            "collection_plan_identity": COLLECTION_IDENTITY,
            "production_parameter_plan_identity": PARAMETER_IDENTITY,
            "outcome_independent_accounting": True,
            "retry_count": 0,
            "counts": {
                "planned": 24,
                "attempted": 24,
                "accepted": 24,
                "rejected": 0,
                "failed": 0,
                "quarantined": 0,
            },
            "attempt_ledger": ledger,
        }

    def test_exact_frozen_attempt_ledger_passes_and_mutations_fail(self):
        collection, report = self._execution_report()
        self.assertEqual(
            validate_issue_53_execution_report(report, collection), report
        )

        mutations = {
            "reordered": lambda value: value["attempt_ledger"].reverse(),
            "post-hoc retry": lambda value: value.__setitem__("retry_count", 1),
            "wrong role": lambda value: value["attempt_ledger"][0].__setitem__(
                "exposure_role", "calibration"
            ),
            "missing attempt": lambda value: value["attempt_ledger"].pop(),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(report)
                mutate(changed)
                with self.assertRaisesRegex(ValueError, "Issue-53"):
                    validate_issue_53_execution_report(changed, collection)

    def test_public_quality_report_redacts_all_final_evaluation_outcomes(self):
        _collection, report = self._execution_report()
        final_mismatch_id = None
        non_final_mismatch_id = None
        for entry in report["attempt_ledger"]:
            entry["realized_coverage_strata"] = [entry["intended_coverage_stratum"]]
            entry["terminal_reason"] = entry["expected_termination"]
            if (
                entry["exposure_role"] == "final_evaluation"
                and entry["intervention_id"] == "central-destruction"
            ):
                entry["terminal_reason"] = "stable_entered"
                final_mismatch_id = entry["attempt_id"]
            if (
                entry["exposure_role"] == "training"
                and entry["intervention_id"] == "central-destruction"
            ):
                entry["terminal_reason"] = "stable_entered"
                non_final_mismatch_id = entry["attempt_id"]
        replay = {
            "passed": False,
            "proof_count": 0,
            "verdicts": [
                {"exposure_role": "final_evaluation", "passed": True},
                {"exposure_role": "training", "passed": True},
            ],
        }
        quality = _quality_report(report, replay)
        public = _public_quality_report(report, replay, quality)
        public_text = json.dumps(public, sort_keys=True)

        self.assertEqual(public["outcome_scope"], "non_final_only")
        self.assertTrue(public["sealed_final_evaluation_outcomes"])
        self.assertEqual(public["counts"]["planned"], 18)
        self.assertNotIn("final_evaluation", public["accepted_by_exposure_role"])
        self.assertNotIn(final_mismatch_id, public_text)
        self.assertIn(non_final_mismatch_id, public_text)
        self.assertEqual(public["replay_proof_count"], 1)

    def test_micro_relations_are_exact_source_bound_projections(self):
        capture = load_physics_capture_v2(
            ROOT / "data/runtime_evidence/issue-44/captures/support-change.json"
        )
        derivation = derive_capture_micro_relations(
            capture,
            source_reference="primary/support-change.json",
            source_capture_bundle_identity="release:fixture",
        )
        validate_capture_micro_relation_derivation(
            derivation,
            capture,
            source_reference="primary/support-change.json",
            source_capture_bundle_identity="release:fixture",
        )
        self.assertEqual(derivation["predicates"], ["contact", "supports"])
        self.assertEqual(
            derivation["fixed_step_count"],
            len(capture.record["fixed_step_samples"]),
        )

        changed = copy.deepcopy(derivation)
        changed["labels"][0]["predicates"]["contact"]["relations"] = []
        with self.assertRaises(CohortV2MicroRelationError):
            validate_capture_micro_relation_derivation(
                changed,
                capture,
                source_reference="primary/support-change.json",
                source_capture_bundle_identity="release:fixture",
            )

    def test_v2_collection_captures_observation_before_the_planned_shot(self):
        from tests.test_physics_capture_v2 import capture

        calls = []
        engine_record = capture()
        source_bindings = engine_record.pop("source_bindings")
        engine_record["schema_version"] = "physics_capture_v2_engine_v1"

        class ActionBridge:
            def set_speed(self, _speed):
                return 1

            def fully_zoom_out(self):
                return 1

            def get_symbolic_state_without_screenshot(self):
                return [{"type": "Slingshot", "vertices": [[100, 200], [120, 200], [120, 260], [100, 260]]}]

            def shoot(self, *_args, **_kwargs):
                calls.append("shoot")
                return 1

        class PhysicsBridge:
            def get_physics_capture_v2(self):
                calls.append("request-71")
                return type("EngineCapture", (), {"record": engine_record})()

        guard = {
            "pre_shot_image": None,
            "pre_shot_sample": None,
            "pre_shot_guard": {"status": "accepted"},
            "post_recovery_protocol_state": {"game_state": "PLAYING"},
            "recovery_action": "none",
        }
        observation = {
            "identity": "observation-trace:fixture",
            "frame_records": [{"identity": "frame:fixture"}],
        }

        def diagnostic_capture(*args, **kwargs):
            calls.append("diagnostic-capture")
            return capture_physics_v2_rollout(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary, patch(
            "scripts.collect_rollouts._run_pre_shot_guard", return_value=guard
        ), patch(
            "scripts.collect_rollouts._protocol_state_snapshot",
            return_value={"game_state": "PLAYING"},
        ), patch(
            "scripts.collect_rollouts.capture_observation_trace",
            side_effect=lambda *_args, **_kwargs: (
                calls.append("request-72") or observation
            ),
        ):
            root = Path(temporary)
            manifest = collect_rollouts(
                ActionBridge(),
                root,
                [
                    {
                        "coordinate_frame": "absolute",
                        "release": [250, 260],
                        "tapTime": 0,
                    }
                ],
                target_fps=1,
                duration_seconds=1,
                anchor_actions=False,
                capture_rollout=diagnostic_capture,
                fresh_engine_attempt=1,
                physics_capture_v2=True,
                physics_bridge=PhysicsBridge(),
                physics_v2_source_bindings=source_bindings,
                physics_v2_scenario_manifest_identity="manifest-v2-1",
                observation_configuration="agent_rgb8_native_v1",
                observation_exposure_role="training",
            )
            metadata = json.loads(
                (root / "shot_001/metadata.json").read_text(encoding="utf-8")
            )

        self.assertEqual(
            calls,
            ["request-72", "diagnostic-capture", "shoot", "request-71"],
        )
        self.assertEqual(
            metadata["observation_trace_manifest_identity"],
            "observation-trace:fixture",
        )
        self.assertEqual(manifest["rollouts"][0]["observation_frame_count"], 1)

    def test_dry_run_validates_command_without_opening_final_data(self):
        result = dry_run()
        self.assertTrue(result["passed"])
        self.assertEqual(result["planned_rollouts"], 24)
        self.assertEqual(result["planned_replay_proofs"], 4)
        self.assertEqual(result["final_access_state"], "pending")
        self.assertFalse(result["final_data_opened"])
        self.assertIn("python -u", result["actual_command"])
        self.assertEqual(result["schema"], "issue_53_production_dry_run_v5")
        self.assertEqual(result["collection_plan_identity"], COLLECTION_IDENTITY_V5)
        self.assertEqual(
            result["production_parameter_plan_identity"], PARAMETER_IDENTITY_V5
        )

    def test_stable_only_plan_v2_is_exact_and_final_outcome_free(self):
        self.assertTrue(validate_plan_v2_evidence(PLAN_ROOT_V2)["passed"])
        collection = json.loads(
            (PLAN_ROOT_V2 / "collection-plan.json").read_text(encoding="utf-8")
        )
        evidence = json.loads(
            (PLAN_ROOT_V2 / "plan-correction-evidence.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(collection["assignments"]), 4)
        self.assertTrue(
            all(len(item["intervention_ids"]) == 6 for item in collection["assignments"])
        )
        self.assertEqual(
            [item["intended_termination_class"] for item in collection["interventions"]],
            ["stable_entered"] * 6,
        )
        self.assertEqual(
            {
                name: item["quota"]
                for name, item in collection["quotas"]["termination_class"].items()
            },
            {"level_clear": 0, "level_fail": 0, "stable_entered": 24},
        )
        self.assertEqual(len(evidence["non_final_attempts"]), 18)
        self.assertFalse(evidence["reviewed_final_outcomes_used"])
        self.assertTrue(
            all(
                item["exposure_role"] != "final_evaluation"
                for item in evidence["non_final_attempts"]
            )
        )
        self.assertEqual(len(evidence["camera_aligned_v2_diagnostics"]), 18)
        self.assertTrue(
            all(
                item["viewport_width_pixels"] == item["agent_width_pixels"]
                and item["viewport_height_pixels"] == item["agent_height_pixels"]
                for item in evidence["camera_aligned_v2_diagnostics"]
            )
        )
        final_projection = json.loads(
            (PLAN_ROOT_V2 / "final-evaluation.sealed-projection.json").read_text(
                encoding="utf-8"
            )
        )
        old_partition = json.loads(
            (
                ROOT
                / "data/runtime_evidence/issue-47/partition-exposure-manifest.json"
            ).read_text(encoding="utf-8")
        )
        old_identities = {
            entry[field]
            for entry in old_partition["entries"]
            for field in (
                "scenario_manifest_identity",
                "scenario_lineage_identity",
                "level_instance_identity",
            )
        }
        self.assertIn("%3A4503%3A", final_projection["level_instance_identity"])
        self.assertTrue(
            all(
                final_projection[field] not in old_identities
                for field in (
                    "scenario_manifest_identity",
                    "scenario_lineage_identity",
                    "level_instance_identity",
                )
            )
        )

    def test_v1_plans_still_rederive_to_the_exact_published_bytes(self):
        payloads = derive_issue_52_payloads(ROOT, validate_pilot=False)
        for name, payload in payloads.items():
            expected = (
                json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            self.assertEqual(
                expected,
                (ROOT / "data/runtime_evidence/issue-52" / name).read_bytes(),
            )

    def test_plan_v3_preserves_science_and_issues_fresh_attempt_authority(self):
        self.assertTrue(validate_plan_v3_evidence(PLAN_ROOT_V3)["passed"])
        v2 = json.loads(
            (PLAN_ROOT_V2 / "collection-plan.json").read_text(encoding="utf-8")
        )
        v3 = json.loads(
            (PLAN_ROOT_V3 / "collection-plan.json").read_text(encoding="utf-8")
        )
        correction = json.loads(
            (PLAN_ROOT_V3 / "executor-correction-evidence.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(v3["identity"], COLLECTION_IDENTITY_V3)
        self.assertEqual(v3["supersedes_plan_identity"], COLLECTION_IDENTITY_V2)
        self.assertEqual(v3["assignments"], v2["assignments"])
        self.assertEqual(v3["interventions"], v2["interventions"])
        self.assertFalse(correction["physics_outcomes_observed"])
        self.assertFalse(correction["final_evaluation_outcomes_used"])
        self.assertTrue(correction["successor_decision"]["reuse_seed_4503"])
        self.assertFalse(correction["successor_decision"]["retry_within_plan_v2"])

        v2_first = _expected_slots(v2)[0]["attempt_id"]
        v3_first = _expected_slots(v3)[0]["attempt_id"]
        self.assertNotEqual(v2_first, v3_first)
        self.assertTrue(v3_first.startswith("cohort-v2-production-attempt-v3:"))

    def test_plan_v4_freezes_assignment_specific_mixed_terminations(self):
        result = validate_plan_v4_evidence(PLAN_ROOT_V4)
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["termination_quotas"],
            {"level_clear": 0, "level_fail": 4, "stable_entered": 20},
        )
        collection = json.loads(
            (PLAN_ROOT_V4 / "collection-plan.json").read_text(encoding="utf-8")
        )
        evidence = json.loads(
            (
                PLAN_ROOT_V4 / "mixed-termination-correction-evidence.json"
            ).read_text(encoding="utf-8")
        )
        slots = _expected_slots(collection)
        expected = [item["expected_termination"] for item in slots]
        self.assertEqual(expected.count("level_fail"), 4)
        self.assertEqual(expected.count("stable_entered"), 20)
        self.assertTrue(
            all(
                item["expected_termination"] == "level_fail"
                for item in slots
                if item["exposure_role"] in {"training", "model_selection"}
                and item["intervention"]["id"]
                in {"central-no-contact-miss", "central-persistent-support"}
            )
        )
        self.assertFalse(evidence["reviewed_or_sealed_final_outcomes_used"])
        self.assertEqual(evidence["decision"]["fresh_final_seed"], 4504)
        final_projection = json.loads(
            (PLAN_ROOT_V4 / "final-evaluation.sealed-projection.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("%3A4504%3A", final_projection["level_instance_identity"])
        self.assertTrue(slots[0]["attempt_id"].startswith("cohort-v2-production-attempt-v4:"))

    def test_plan_v5_corrects_utc_freeze_without_changing_science(self):
        result = validate_plan_v5_evidence(PLAN_ROOT_V5)
        self.assertTrue(result["passed"])
        self.assertEqual(result["workflow_frozen_at"], "2026-08-23T14:46:00Z")
        v4 = json.loads(
            (PLAN_ROOT_V4 / "collection-plan.json").read_text(encoding="utf-8")
        )
        v5 = json.loads(
            (PLAN_ROOT_V5 / "collection-plan.json").read_text(encoding="utf-8")
        )
        self.assertEqual(v5["assignments"], v4["assignments"])
        self.assertEqual(v5["interventions"], v4["interventions"])
        self.assertEqual(v5["quotas"], v4["quotas"])

        failed = FinalEvaluationWorkflowAccessManifest.from_dict(
            json.loads(
                (
                    PLAN_ROOT_V4
                    / "final-evaluation-workflow-access-manifest.json"
                ).read_text(encoding="utf-8")
            )
        )
        with self.assertRaisesRegex(ValueError, "predates workflow freeze"):
            authorize_final_evaluation_workflow_access(
                failed,
                authorization_identity="authorization:fixture:v4",
                authorized_at="2026-08-23T14:45:54Z",
            )

        corrected = FinalEvaluationWorkflowAccessManifest.from_dict(
            json.loads(
                (
                    PLAN_ROOT_V5
                    / "final-evaluation-workflow-access-manifest.json"
                ).read_text(encoding="utf-8")
            )
        )
        authorized = authorize_final_evaluation_workflow_access(
            corrected,
            authorization_identity="authorization:fixture:v5",
            authorized_at="2026-08-23T14:46:01Z",
        )
        self.assertEqual(authorized.authorization_state, "authorized")

    def test_executor_exposes_plan_selection_and_read_only_validation_modes(self):
        args = build_parser().parse_args(
            ["--plan-root", str(PLAN_ROOT_V5), "--validate"]
        )
        self.assertTrue(args.validate)
        self.assertFalse(args.dry_run)
        self.assertEqual(args.plan_root, PLAN_ROOT_V5)

    def test_v2_accounting_uses_dynamic_identities_and_fails_closed_on_termination(self):
        collection = json.loads(
            (PLAN_ROOT_V2 / "collection-plan.json").read_text(encoding="utf-8")
        )
        ledger = []
        for slot in _expected_slots(collection):
            intervention = slot["intervention"]
            ledger.append(
                {
                    "attempt_id": slot["attempt_id"],
                    "exposure_role": slot["exposure_role"],
                    "dataset_partition": slot["dataset_partition"],
                    "scenario_manifest_identity": slot["scenario_manifest_identity"],
                    "scenario_lineage_identity": slot["scenario_lineage_identity"],
                    "level_instance_identity": slot["level_instance_identity"],
                    "scenario_template_identity": slot["scenario_template_identity"],
                    "benchmark_condition_identity": slot[
                        "benchmark_condition_identity"
                    ],
                    "intervention_id": intervention["id"],
                    "intervention_identity": production_intervention_identity(
                        intervention["id"], COLLECTION_IDENTITY_V2
                    ),
                    "intervention_source": intervention["intervention_source"],
                    "intended_coverage_stratum": intervention[
                        "intended_coverage_stratum"
                    ],
                    "expected_termination": "stable_entered",
                    "status": "accepted",
                    "artifact_path": f"runtime/{slot['attempt_id']}",
                    "realized_coverage_strata": [
                        intervention["intended_coverage_stratum"]
                    ],
                    "terminal_reason": "stable_entered",
                }
            )
        report = {
            "schema": "issue_53_production_execution_report_v2",
            "collection_plan_identity": COLLECTION_IDENTITY_V2,
            "production_parameter_plan_identity": PARAMETER_IDENTITY_V2,
            "outcome_independent_accounting": True,
            "retry_count": 0,
            "counts": {
                "planned": 24,
                "attempted": 24,
                "accepted": 24,
                "rejected": 0,
                "failed": 0,
                "quarantined": 0,
            },
            "attempt_ledger": ledger,
        }
        self.assertEqual(validate_issue_53_execution_report(report, collection), report)
        self.assertTrue(ledger[0]["attempt_id"].startswith("cohort-v2-production-attempt-v2:"))
        replay = {"passed": True, "proof_count": 4, "verdicts": []}
        self.assertTrue(_quality_report(report, replay)["passed"])
        report["attempt_ledger"][0]["terminal_reason"] = "level_fail"
        quality = _quality_report(report, replay)
        self.assertFalse(quality["passed"])
        self.assertEqual(len(quality["termination_mismatches"]), 1)

    def test_incomplete_run_publishes_durable_public_and_sealed_accounting(self):
        collection = self._collection_plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            output = root / "public"
            sealed = root / "sealed"
            runtime.mkdir()
            ledger = []
            for slot in _expected_slots(collection):
                intervention = slot["intervention"]
                quarantine = runtime / "production/quarantine" / slot["attempt_id"]
                quarantine.mkdir(parents=True)
                failure = quarantine / "failure.json"
                failure.write_text("{}\n", encoding="utf-8")
                ledger.append(
                    {
                        "attempt_id": slot["attempt_id"],
                        "exposure_role": slot["exposure_role"],
                        "dataset_partition": slot["dataset_partition"],
                        "scenario_manifest_identity": slot[
                            "scenario_manifest_identity"
                        ],
                        "scenario_lineage_identity": slot[
                            "scenario_lineage_identity"
                        ],
                        "level_instance_identity": slot[
                            "level_instance_identity"
                        ],
                        "scenario_template_identity": slot[
                            "scenario_template_identity"
                        ],
                        "benchmark_condition_identity": slot[
                            "benchmark_condition_identity"
                        ],
                        "intervention_id": intervention["id"],
                        "intervention_identity": production_intervention_identity(
                            intervention["id"]
                        ),
                        "intervention_ordinal": intervention["ordinal"],
                        "intervention_source": intervention[
                            "intervention_source"
                        ],
                        "intended_coverage_stratum": intervention[
                            "intended_coverage_stratum"
                        ],
                        "expected_termination": intervention[
                            "intended_termination_class"
                        ],
                        "status": "failed",
                        "reason": "fixture failure",
                        "failure_code": "fixture_failure",
                        "artifact_path": None,
                        "quarantine_path": str(quarantine),
                        "failure_manifest_path": str(failure),
                        "realized_coverage_strata": [],
                        "terminal_reason": None,
                        "terminal_span_fixed_steps": None,
                        "attempt_number": 1,
                        "retry_decision": "none",
                    }
                )
            report = {
                "schema": "issue_53_production_execution_report_v1",
                "collection_plan_identity": COLLECTION_IDENTITY,
                "production_parameter_plan_identity": PARAMETER_IDENTITY,
                "attempt_ledger": ledger,
                "counts": {
                    "planned": 24,
                    "attempted": 24,
                    "accepted": 0,
                    "rejected": 0,
                    "failed": 24,
                    "quarantined": 24,
                },
                "retry_count": 0,
                "outcome_independent_accounting": True,
            }
            (runtime / "production-execution-report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            (runtime / "production-replay-report.json").write_text(
                json.dumps(
                    {
                        "schema": "cohort_v2_production_replay_report_v1",
                        "identity": "cohort-v2-production-replay-report-v1:incomplete",
                        "collection_plan_identity": COLLECTION_IDENTITY,
                        "comparison_rules_identity": None,
                        "proof_count": 0,
                        "retry_count": 0,
                        "verdicts": [],
                        "passed": False,
                    }
                ),
                encoding="utf-8",
            )
            (runtime / "player-provenance.json").write_text(
                json.dumps({"status": "verified"}), encoding="utf-8"
            )
            authorities = runtime / "authorities"
            (authorities / "manifests").mkdir(parents=True)
            (authorities / "xml").mkdir()
            for role in ("training", "calibration", "model-selection", "final-evaluation"):
                (authorities / "manifests" / f"{role}.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                (authorities / "xml" / f"{role}.xml").write_text(
                    "<Level/>\n", encoding="utf-8"
                )
            (authorities / "authorized-final-access-manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )

            result = publish_issue_53_evidence(
                repository_root=ROOT,
                runtime_root=runtime,
                output=output,
                sealed_output=sealed,
                final_access_audit={
                    "workflow_manifest_identity": "workflow:fixture",
                    "passed": True,
                },
            )
            self.assertEqual(result["disposition"], "incomplete")
            self.assertFalse(result["passed"])
            self.assertTrue((output / "quarantine").is_dir())
            self.assertTrue((sealed / "quarantine").is_dir())
            public_text = (output / "production-attempt-accounting.json").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("final-evaluation.xml", public_text)
            public_accounting = json.loads(public_text)
            final_entries = [
                item
                for item in public_accounting["attempt_ledger"]
                if item["exposure_role"] == "final_evaluation"
            ]
            self.assertEqual(public_accounting["outcome_scope"], "non_final_only")
            self.assertEqual(public_accounting["counts"]["planned"], 18)
            self.assertTrue(all(item["status"] == "sealed" for item in final_entries))
            self.assertTrue(all(item["terminal_reason"] is None for item in final_entries))
            public_quality = json.loads(
                (output / "production-quality-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(public_quality["outcome_scope"], "non_final_only")
            self.assertNotIn("final_evaluation", public_quality["accepted_by_exposure_role"])
            self.assertTrue((sealed / "production-quality-report.json").is_file())


if __name__ == "__main__":
    unittest.main()
