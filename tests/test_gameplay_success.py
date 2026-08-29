import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from world_model.planning.gameplay import (
    CandidateEvaluation,
    CEMIteration,
    ControlMode,
    ControlResult,
    ControlStep,
    PlanResult,
    SlingshotAction,
)
from world_model.planning.gameplay_success import (
    AUTHORIZATION_IDENTITY,
    SYSTEM_IDS,
    _identity,
    aggregate_trials,
    build_protocol,
    build_trial_record,
    validate_final_artifacts,
    validate_protocol,
    validate_trial_record,
    wilson_interval,
    write_final_artifacts,
    write_run_manifest,
)
from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_json


class GameplaySuccessFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.game = cls.root / "game"
        streaming = cls.game / "9001_Data/StreamingAssets"
        levels = []
        for condition_index in range(1, 6):
            novelty_type = f"type01010{condition_index}"
            for instance_index in range(1, 11):
                relative = (
                    "9001_Data/StreamingAssets/Levels/novelty_level_1/"
                    f"{novelty_type}/Levels/{instance_index:05d}_fixture.xml"
                )
                path = cls.game / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    f"<Level condition='{novelty_type}' instance='{instance_index}'/>",
                    encoding="utf-8",
                )
                levels.append(f"/fixture/{relative}")
        entries = "\n".join(
            f'<game_levels level_path="{value}" />' for value in levels
        )
        config = (
            '<?xml version="1.0" encoding="utf-16"?>\n<evaluation><trials>'
            f'<trial id="0"><game_level_set>{entries}</game_level_set></trial>'
            f'<trial id="1"><game_level_set>{entries}</game_level_set></trial>'
            "</trials></evaluation>"
        )
        streaming.mkdir(parents=True, exist_ok=True)
        (streaming / "config.xml").write_text(config, encoding="utf-8")
        (cls.game / "9001.x86_64").write_bytes(b"fixture-player")
        (cls.game / "game_playing_interface.jar").write_bytes(b"fixture-interface")
        cls.identities = {
            "world": "world-model-fixture",
            "controller": "controller-fixture",
            "parser": "parser-fixture",
            "adapter": "visual-carrier-fixture",
        }
        planning = {
            "artifact_identity": "issue-56-fixture",
            "source_bindings": {
                "implementation_revision": "5" * 40,
                "world_model_checkpoint_identity": cls.identities["world"],
                "controller_checkpoint_identity": cls.identities["controller"],
                "visual_parser_checkpoint_identity": cls.identities["parser"],
                "observation_adapter_identity": cls.identities["adapter"],
                "goal_cost_version": "cohort-v2-gameplay-cost-v1",
                "environment_version": "ScienceBirds-Linux-Unity-2019.4.41f2",
            },
        }
        cls.protocol = build_protocol(
            cls.game, planning, {"identity": "migration-recovery-fixture"}
        )
        cls.stack = {
            key: cls.protocol["source_bindings"][key]
            for key in (
                "world_model_checkpoint_identity_sha256",
                "controller_checkpoint_identity_sha256",
                "visual_parser_checkpoint_identity_sha256",
                "observation_adapter_identity",
            )
        }

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def _result(
        *,
        success: bool,
        termination: str | None = None,
        planner_id: str = "fixture",
        invalid: int = 0,
        candidate_failure: str | None = None,
        wall: float = 1.0,
    ) -> ControlResult:
        action = SlingshotAction(-100, 20, 50)
        evaluation = CandidateEvaluation(
            (action,),
            -1.0,
            requested_horizons=(1, 1),
            effective_horizons=(1, 1),
            requested_abstractions=("continuous", "continuous"),
            model_rollout_count=2,
            model_compute=20.0,
        )
        iterations = ()
        if candidate_failure is not None:
            iterations = (CEMIteration(
                1,
                (0.0,),
                (0,),
                ((-100.0, 20.0, 50.0),),
                ((1.0, 1.0, 1.0),),
                1,
                (candidate_failure,),
            ),)
        plan = PlanResult(
            planner_id,
            7,
            (action,),
            evaluation,
            iterations=iterations,
            candidate_count=4,
            invalid_candidate_count=invalid,
            model_rollout_count=8,
            planner_compute=80.0,
            goal_evaluation_count=4,
            wall_clock_seconds=0.2,
        )
        step = ControlStep(
            1,
            "before",
            "after",
            plan,
            action,
            0.25,
            {
                "structure_unstable_probability": 0.51,
                "structure_unstable_thresholded": True,
            },
            {
                "structure_unstable_probability": 0.49,
                "structure_unstable_thresholded": False,
            },
        )
        return ControlResult(
            mode=ControlMode.MPC,
            planner_id=planner_id,
            steps=(step,),
            termination_reason=termination or ("success" if success else "terminal_failure"),
            success=success,
            replan_count=1,
            candidate_count=4,
            invalid_candidate_count=invalid,
            model_rollout_count=8,
            planner_compute=80.0,
            goal_evaluation_count=4,
            planner_wall_clock_seconds=0.2,
            game_interface_wall_clock_seconds=0.8,
            wall_clock_seconds=wall,
            failures=(),
        )

    def _record(self, entry, *, success: bool, **kwargs):
        return build_trial_record(
            self.protocol,
            entry,
            implementation_revision="a" * 40,
            stack_bindings=self.stack,
            result=self._result(success=success, **kwargs),
            observation_parser_call_count=2,
            observation_parser_wall_clock_seconds=0.1,
        )


class GameplaySuccessProtocolTests(GameplaySuccessFixture):
    def test_protocol_freezes_isolated_levels_systems_seeds_and_h1_comparator(self):
        validated = validate_protocol(self.protocol, game_dir=self.game)

        roles = validated["level_inventory"]["roles"]
        identities = [
            item["level_identity"]
            for levels in roles.values()
            for item in levels
        ]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(tuple(item["system_id"] for item in validated["systems"]), SYSTEM_IDS)
        h1 = next(
            item for item in validated["systems"]
            if item["system_id"] == "repeated_h1_cem_mpc"
        )
        self.assertEqual(
            h1["fixed_prediction_pair"], {"horizon": 1, "abstraction": "continuous"}
        )
        self.assertEqual(len(validated["trial_schedule"]), 75)
        self.assertEqual(validated["execution_limits"]["retry_count"], 0)

    def test_altered_protocol_and_cross_role_level_reuse_are_rejected(self):
        altered = copy.deepcopy(self.protocol)
        altered["trial_seeds"][0] += 1
        with self.assertRaisesRegex(ValueError, "identity or freeze"):
            validate_protocol(altered)

        leaked = copy.deepcopy(self.protocol)
        leaked["level_inventory"]["roles"]["calibration"][0] = copy.deepcopy(
            leaked["level_inventory"]["roles"]["final_evaluation"][0]
        )
        payload = dict(leaked)
        payload.pop("protocol_identity")
        leaked["protocol_identity"] = _identity(
            "cohort-v2-gameplay-success-protocol-v1", payload
        )
        with self.assertRaisesRegex(ValueError, "level inventory|leak"):
            validate_protocol(leaked)


class GameplaySuccessAccountingTests(GameplaySuccessFixture):
    def test_success_and_unsuccessful_censoring_have_fixed_denominators(self):
        success_entry, failure_entry = self.protocol["trial_schedule"][:2]
        success = self._record(success_entry, success=True)
        failure = self._record(failure_entry, success=False)

        self.assertEqual(success["outcome"]["shots_to_success"], 1)
        self.assertFalse(success["outcome"]["censored_unsuccessful"])
        self.assertIsNone(failure["outcome"]["shots_to_success"])
        self.assertTrue(failure["outcome"]["censored_unsuccessful"])
        self.assertEqual(failure["outcome"]["penalized_shots"], 7)
        self.assertTrue(failure["outcome"]["included_in_denominator"])
        self.assertEqual(
            failure["failures"]["taxonomy_counts"]["level_terminal_failure"], 1
        )

    def test_timeout_invalid_actions_and_model_failures_are_retained(self):
        entry = self.protocol["trial_schedule"][0]
        record = self._record(
            entry,
            success=True,
            invalid=2,
            candidate_failure="model rollout failed",
            wall=301.0,
        )

        self.assertFalse(record["outcome"]["success"])
        self.assertEqual(record["outcome"]["termination_reason"], "timeout")
        self.assertEqual(record["failures"]["taxonomy_counts"]["timeout"], 1)
        self.assertEqual(record["failures"]["taxonomy_counts"]["invalid_action"], 2)
        self.assertEqual(
            record["failures"]["taxonomy_counts"]["model_rollout_failure"], 1
        )

    def test_horizon_error_compute_and_parser_ranking_path_are_separate(self):
        entry = next(
            value for value in self.protocol["trial_schedule"]
            if value["system_id"] == "adaptive_cem_mpc"
        )
        record = self._record(entry, success=True)

        self.assertEqual(
            record["prediction_diagnostics"]["repeated_h1_transition_count"], 2
        )
        self.assertEqual(
            record["prediction_diagnostics"]["accumulated_recursive_rollout_error"],
            0.25,
        )
        self.assertTrue(record["parser_usage"]["visual_carrier_affects_action_ranking"])
        self.assertFalse(
            record["parser_usage"]["structure_unstable_enters_selected_macro_rollout"]
        )
        self.assertAlmostEqual(
            record["compute"]["game_interface_exclusive_estimate_seconds"], 0.7
        )

    def test_wilson_interval_is_bounded_and_non_degenerate(self):
        lower, upper = wilson_interval(5, 10)
        self.assertLess(lower, 0.5)
        self.assertGreater(upper, 0.5)
        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)


class GameplaySuccessArtifactTests(GameplaySuccessFixture):
    def _matrix(self):
        records = []
        for entry in self.protocol["trial_schedule"]:
            records.append(self._record(
                entry,
                success=entry["system_id"] == "adaptive_cem_mpc",
            ))
        return records

    def test_full_matrix_aggregates_conditions_and_supported_decision(self):
        aggregate = aggregate_trials(self.protocol, self._matrix())

        self.assertEqual(aggregate["trial_count"], 75)
        self.assertEqual(len(aggregate["conditions"]), 5)
        self.assertEqual(
            aggregate["primary_comparison"]["strongest_comparator"], "random_legal"
        )
        self.assertEqual(aggregate["gameplay_conclusion"], "supported")
        self.assertEqual(
            aggregate["systems"]["adaptive_cem_mpc"]["success_rate"], 1.0
        )

    def test_exact_artifact_validation_rebuilds_tables_plots_and_trial_records(self):
        records = self._matrix()
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_run_manifest(
                output,
                self.protocol,
                implementation_revision="a" * 40,
                stack_bindings=self.stack,
                authorization_identity=AUTHORIZATION_IDENTITY,
            )
            for entry, record in zip(self.protocol["trial_schedule"], records):
                validate_trial_record(self.protocol, record, entry)
                write_immutable_cohort_v2_json(
                    record, output / "trial-records" / f"{entry['trial_id']}.json"
                )
            written = write_final_artifacts(output, self.protocol, records)
            validated = validate_final_artifacts(output, self.protocol)

            self.assertEqual(validated, written)
            self.assertEqual(written["gameplay_conclusion"], "supported")
            self.assertTrue((output / "success_rates.csv").is_file())
            self.assertTrue((output / "success-rate.svg").is_file())
            changed = json.loads((output / "aggregate.json").read_bytes())
            changed["gameplay_conclusion"] = "changed"
            (output / "aggregate.json").write_text(
                json.dumps(changed, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "aggregate.json"):
                validate_final_artifacts(output, self.protocol)


if __name__ == "__main__":
    unittest.main()
