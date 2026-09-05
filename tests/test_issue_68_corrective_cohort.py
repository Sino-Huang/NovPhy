from __future__ import annotations

from concurrent.futures import Future
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_issue_68_corrective_cohort import (
    _all_results,
    _collect_candidate,
    _diversity_payload,
    _pilot_report,
    _parser,
    _synthetic_results,
    main as issue_68_main,
)
from world_model.data.corrective_ranking_cohort import (
    DEFAULT_PRODUCTION_STATES_PER_ROLE,
    PILOT_STATES_PER_ROLE,
    CorrectiveRankingCohortError,
    build_pilot_plan,
    build_production_plan,
    realized_endpoint_cost,
    validate_pilot_report,
    validate_plan,
    validate_role_access,
)
from world_model.training.action_ranking_probe import BROAD_ACTION_DESIGN_ID


class Issue68CorrectiveCohortTests(unittest.TestCase):
    def _passing_report(self) -> dict[str, object]:
        plan = build_pilot_plan()
        results = _synthetic_results(plan)
        return _pilot_report(
            plan,
            results,
            {
                "candidate_count": len(results),
                "video_count": len(results),
                "accepted_video_count": len(results),
            },
        )

    def test_pilot_plan_freezes_fresh_role_isolated_twelve_action_states(self) -> None:
        first = build_pilot_plan()
        repeated = build_pilot_plan()

        self.assertEqual(first, repeated)
        self.assertEqual(
            first["role_counts"],
            {
                "calibration": PILOT_STATES_PER_ROLE,
                "model_selection": PILOT_STATES_PER_ROLE,
            },
        )
        self.assertEqual(first["action_design"]["identity"], BROAD_ACTION_DESIGN_ID)
        self.assertEqual(len(first["states"]), 24)
        self.assertTrue(all(len(state["candidates"]) == 12 for state in first["states"]))
        self.assertTrue(all(
            len({item["action_stratum"] for item in state["candidates"]}) == 12
            for state in first["states"]
        ))
        self.assertEqual(
            len({state["generation_seed"] for state in first["states"]}), 24
        )
        self.assertTrue(all(
            "sha256" not in state["identity"]
            and "sha256" not in state["slot_identity"]
            and all("sha256" not in item["identity"] for item in state["candidates"])
            for state in first["states"]
        ))

    def test_publication_defaults_to_four_validation_processes(self) -> None:
        args = _parser().parse_args(["--publish-production"])

        self.assertEqual(args.validation_workers, 4)

    def test_parallel_exact_validation_preserves_plan_order_and_logs_progress(
        self,
    ) -> None:
        plan = build_pilot_plan()
        plan = {**plan, "states": plan["states"][:4]}
        calls = []
        executor_worker_counts = []

        class InlineExecutor:
            def __init__(self, *, max_workers):
                executor_worker_counts.append(max_workers)

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

        def validate_state(_plan_header, state, _runtime):
            calls.append(state["identity"])
            return ([{
                "state_identity": state["identity"],
                "candidate_ordinal": candidate["ordinal"],
            } for candidate in state["candidates"]], 0.1)

        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            records = runtime / "records"
            records.mkdir()
            for ordinal in range(48):
                (records / f"candidate-{ordinal}.json").write_text(
                    "{}", encoding="utf-8"
                )
            output = io.StringIO()
            with patch(
                "scripts.run_issue_68_corrective_cohort.ProcessPoolExecutor",
                InlineExecutor,
            ), patch(
                "scripts.run_issue_68_corrective_cohort._validate_state_results",
                side_effect=validate_state,
            ), redirect_stdout(output):
                results = _all_results(
                    plan,
                    runtime,
                    workers=4,
                    progress_label="publish preflight",
                )

        self.assertEqual(calls, [state["identity"] for state in plan["states"]])
        self.assertEqual(executor_worker_counts, [4])
        self.assertEqual(
            [results[index * 12]["state_identity"] for index in range(4)],
            [state["identity"] for state in plan["states"]],
        )
        self.assertIn(
            "publish preflight exact validation start states=4 candidates=48 workers=4",
            output.getvalue(),
        )
        self.assertIn(
            "publish preflight exact validation progress=4/4",
            output.getvalue(),
        )

    def test_plan_validation_rejects_post_outcome_candidate_change(self) -> None:
        plan = build_pilot_plan()
        changed = deepcopy(plan)
        changed["states"][0]["candidates"].pop()

        with self.assertRaisesRegex(
            CorrectiveRankingCohortError, "states or candidates"
        ):
            validate_plan(changed)

    def test_pilot_reports_more_discrimination_than_issue_63_and_freezes_200(self) -> None:
        report = self._passing_report()

        self.assertTrue(report["passed"])
        self.assertEqual(
            report["diversity"]["outcome_discriminating_state_count"], 18
        )
        self.assertEqual(
            report["diversity"]["progress_only_discordant_state_count"], 18
        )
        self.assertEqual(
            report["sample_size_justification"]["recommended_states_per_role"],
            DEFAULT_PRODUCTION_STATES_PER_ROLE,
        )
        self.assertEqual(validate_pilot_report(report), report)

        production = build_production_plan(report)
        self.assertEqual(
            production["role_counts"],
            {"calibration": 200, "model_selection": 200},
        )
        self.assertEqual(len(production["states"]), 400)
        self.assertEqual(
            len({state["generation_seed"] for state in production["states"]}),
            400,
        )

    def test_production_rejects_underpowered_state_count(self) -> None:
        with self.assertRaisesRegex(
            CorrectiveRankingCohortError, "at least 200"
        ):
            build_production_plan(self._passing_report(), states_per_role=199)

    def test_endpoint_progress_breaks_count_ties_without_overriding_removal(self) -> None:
        miss, miss_progress = realized_endpoint_cost(
            active_pigs=1,
            active_blocks=2,
            pig_contact=False,
            block_contact=False,
            support_change=False,
            pig_displacement=0.0,
            block_displacement=0.0,
        )
        hit, hit_progress = realized_endpoint_cost(
            active_pigs=1,
            active_blocks=2,
            pig_contact=True,
            block_contact=False,
            support_change=False,
            pig_displacement=0.2,
            block_displacement=0.0,
        )
        removed_block, _ = realized_endpoint_cost(
            active_pigs=1,
            active_blocks=1,
            pig_contact=False,
            block_contact=False,
            support_change=False,
            pig_displacement=0.0,
            block_displacement=0.0,
        )
        removed_pig, _ = realized_endpoint_cost(
            active_pigs=0,
            active_blocks=2,
            pig_contact=False,
            block_contact=False,
            support_change=False,
            pig_displacement=0.0,
            block_displacement=0.0,
        )

        self.assertLess(hit, miss)
        self.assertGreater(hit_progress["total"], miss_progress["total"])
        self.assertLess(removed_block, hit)
        self.assertLess(removed_pig, removed_block)
        self.assertEqual(math.floor(hit), 1002)

    def test_model_selection_access_requires_calibration_only_issue_69_freeze(self) -> None:
        validate_role_access(
            "calibration", release_manifest_identity="issue-68-release"
        )
        with self.assertRaisesRegex(
            CorrectiveRankingCohortError, "training/calibration decision"
        ):
            validate_role_access(
                "model_selection",
                release_manifest_identity="issue-68-release",
            )

        validate_role_access(
            "model_selection",
            release_manifest_identity="issue-68-release",
            decision_freeze={
                "schema": "issue_69_corrective_decision_freeze_v1",
                "issue_68_release_identity": "issue-68-release",
                "information_roles": ["training", "calibration"],
                "frozen": True,
                "model_selection_opened": False,
                "final_evaluation_opened": False,
            },
        )

    def test_candidate_failures_are_retained_and_do_not_create_discrimination(self) -> None:
        plan = build_pilot_plan()
        results = _synthetic_results(plan)
        state = plan["states"][0]
        for result in results:
            if result["state_identity"] == state["identity"]:
                result["status"] = "failed"
                result["realized_cost"] = 1_000_000_000.0

        diversity = _diversity_payload(plan, results)

        self.assertEqual(diversity["candidate_failure_count"], 12)
        self.assertEqual(diversity["state_failure_count"], 1)
        self.assertEqual(diversity["fully_realized_state_count"], 23)
        self.assertEqual(diversity["outcome_discriminating_state_count"], 17)

    def test_collection_retains_an_exhausted_candidate_at_frozen_worst_cost(self) -> None:
        plan = build_pilot_plan()
        state = plan["states"][0]
        candidate = state["candidates"][0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch(
                    "scripts.run_issue_68_corrective_cohort._collect_lineage_attempt",
                    side_effect=OSError("fixture collection failure"),
                ),
                patch(
                    "scripts.run_issue_68_corrective_cohort._attempt_audit",
                    return_value={"manifest_path": "failed.json"},
                ),
            ):
                result = _collect_candidate(
                    plan,
                    state,
                    candidate,
                    runtime=root / "runtime",
                    audit=root / "audit",
                    game=root / "game",
                    speed=50,
                    headless=True,
                    engine_start_lock=root / "engine.lock",
                )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["attempt_count"], 2)
        self.assertEqual(result["realized_cost"], 1_000_000_000.0)
        self.assertEqual(
            [item["error_type"] for item in result["failures"]],
            ["OSError", "OSError"],
        )
        self.assertFalse(result["outcome_conditioned_membership"])

    def test_no_write_dry_run_exercises_full_freeze_without_unity(self) -> None:
        output = io.StringIO()
        with (
            patch(
                "scripts.run_issue_68_corrective_cohort._materialize_slot"
            ),
            patch(
                "scripts.run_issue_68_corrective_cohort._verify_webm_encoder"
            ),
            redirect_stdout(output),
        ):
            status = issue_68_main(["--dry-run"])

        self.assertEqual(status, 0)
        self.assertIn('"pilot_candidates": 288', output.getvalue())
        self.assertIn('"production_candidates": 4800', output.getvalue())
        self.assertIn('"files_written": false', output.getvalue())
        self.assertIn("unity=unopened", output.getvalue())

    def test_run_pilot_auto_freezes_fresh_v2_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot_plan = root / "pilot-plan.json"
            pilot_report = root / "pilot-report.json"
            audit_root = root / "audit"

            def collect(plan: dict[str, object], **_kwargs: object):
                results = _synthetic_results(plan)
                target = audit_root / "pilot/manifest.json"
                target.parent.mkdir(parents=True)
                target.write_text(json.dumps({
                    "candidate_count": len(results),
                    "video_count": len(results),
                    "accepted_video_count": len(results),
                }))
                return results

            with (
                patch(
                    "scripts.run_issue_68_corrective_cohort.run_collection",
                    side_effect=collect,
                ),
                redirect_stdout(io.StringIO()),
            ):
                status = issue_68_main([
                    "--run-pilot",
                    "--pilot-plan", str(pilot_plan),
                    "--pilot-runtime", str(root / "runtime"),
                    "--pilot-report", str(pilot_report),
                    "--audit-output", str(audit_root),
                    "--implementation-revision", "fixture-revision",
                ])

            frozen = json.loads(pilot_plan.read_text())
            report = json.loads(pilot_report.read_text())
            self.assertEqual(status, 0)
            self.assertEqual(frozen["schema"], "issue_68_corrective_collection_plan_v2")
            self.assertEqual(len(frozen["states"]), 24)
            self.assertTrue(report["passed"])
            self.assertFalse(frozen["supersedes"]["candidate_outcomes_reused"])


if __name__ == "__main__":
    unittest.main()
