from __future__ import annotations

import unittest
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image

from sciencebirdsagents.SBEnvironment.action_utils import normalize_release_action
from world_model.planning.gameplay import (
    CEMConfig,
    CEMPlanner,
    CandidateEvaluation,
    ControlConfig,
    ControlMode,
    GameplayCost,
    GameplayCostConfig,
    GameplayCostTerms,
    GameplayEvidenceBindings,
    FrozenCohortV2WorldModel,
    PlanningObservation,
    PlanResult,
    PredictedTransition,
    SlingshotAction,
    SlingshotActionBounds,
    TerminalStatus,
    HeuristicNoModelPlanner,
    RandomLegalPlanner,
    VisualPlanningObservationAdapter,
    WorldModelCandidateEvaluator,
    run_gameplay_control,
    validate_gameplay_evidence,
    write_gameplay_evidence,
)
from world_model.training.cohort_v2_controller import (
    CohortV2ControllerConfig,
    CohortV2ControllerFeatureCodec,
)
from world_model.training.cohort_v2_measurement import CohortV2ComputeCalibration
from world_model.training.grid_artifacts import canonical_json_bytes


class SlingshotActionTests(unittest.TestCase):
    def test_legal_action_round_trips_through_the_game_interface(self) -> None:
        bounds = SlingshotActionBounds(
            drag_x=(-160, -40),
            drag_y=(-80, 80),
            tap_time_ms=(0, 1000),
            release_time_ms=600,
        )
        action = SlingshotAction(drag_x=-96, drag_y=31, tap_time_ms=70)

        interface = action.to_interface_action((312, 227), bounds)

        self.assertTrue(bounds.contains(action))
        self.assertEqual(normalize_release_action(interface), (-96, 31, 70))
        self.assertEqual(
            SlingshotAction.from_interface_action(interface),
            action,
        )
        self.assertEqual(interface["releaseTime"], 600)


class GameplayCostTests(unittest.TestCase):
    def test_cost_binds_goal_terminal_legality_rollout_physics_and_compute(self) -> None:
        cost = GameplayCost(GameplayCostConfig(
            goal_progress_weight=4.0,
            terminal_success_cost=-10.0,
            terminal_failure_cost=12.0,
            illegal_action_cost=20.0,
            physical_penalty_weight=2.0,
            rollout_penalty_weight=3.0,
            compute_weight=0.01,
            structure_unstable_weight=0.0,
        ))

        result = cost.evaluate(GameplayCostTerms(
            goal_progress=0.5,
            terminal_status=TerminalStatus.SUCCESS,
            legal_action=True,
            physical_penalty=0.25,
            rollout_penalty=0.1,
            compute=20.0,
            structure_unstable_probability=0.99,
        ))

        self.assertAlmostEqual(result.total, -11.0)
        self.assertEqual(result.structure_unstable_cost, 0.0)
        self.assertFalse(result.structure_unstable_affects_cost)


class CEMPlannerTests(unittest.TestCase):
    def test_seeded_cem_reproduces_batches_elites_updates_and_first_action(self) -> None:
        class RecordingEvaluator:
            def __init__(self) -> None:
                self.sequences: list[tuple[SlingshotAction, ...]] = []

            def evaluate(self, observation, actions):
                self.sequences.append(actions)
                action = actions[0]
                return CandidateEvaluation(
                    actions=actions,
                    total_cost=float(
                        (action.drag_x + 90) ** 2
                        + (action.drag_y - 10) ** 2
                        + (action.tap_time_ms - 50) ** 2
                    ),
                )

        observation = PlanningObservation(
            identity="fixture-observation",
            carrier=torch.zeros(4),
            pig_slots=(0,),
            slingshot_anchor=(312, 227),
        )
        config = CEMConfig(
            population_size=4,
            elite_count=2,
            iterations=2,
            sequence_length=1,
            seed=7,
            minimum_std=1.0,
        )
        bounds = SlingshotActionBounds((-100, -80), (0, 20), (40, 60))
        first_evaluator = RecordingEvaluator()
        second_evaluator = RecordingEvaluator()

        first = CEMPlanner(config, bounds, first_evaluator).plan(observation)
        second = CEMPlanner(config, bounds, second_evaluator).plan(observation)

        expected_first_batch = [
            SlingshotAction(-90, 13, 47),
            SlingshotAction(-99, 5, 40),
            SlingshotAction(-89, 20, 45),
            SlingshotAction(-96, 15, 54),
        ]
        self.assertEqual([item[0] for item in first_evaluator.sequences[:4]], expected_first_batch)
        self.assertEqual(first_evaluator.sequences, second_evaluator.sequences)
        self.assertEqual(first.actions, second.actions)
        self.assertEqual(first.actions[0], SlingshotAction(-91, 13, 49))
        self.assertEqual(first.iterations[0].elite_indices, (0, 3))
        self.assertEqual(first.iterations[0].updated_mean, ((-93.0, 14.0, 50.5),))
        self.assertEqual(first.iterations[1].elite_indices, (1, 0))
        self.assertEqual(first.iterations[1].updated_mean, ((-92.0, 13.0, 49.5),))

    def test_failed_candidates_are_counted_and_excluded_from_elites(self) -> None:
        class Evaluator:
            def evaluate(self, observation, actions):
                action = actions[0]
                if action == SlingshotAction(-90, 13, 47):
                    return CandidateEvaluation(actions, 0.0, failure="model_failure")
                return CandidateEvaluation(
                    actions,
                    float(
                        (action.drag_x + 90) ** 2
                        + (action.drag_y - 10) ** 2
                        + (action.tap_time_ms - 50) ** 2
                    ),
                )

        observation = PlanningObservation(
            "fixture", torch.zeros(4), (0,), (312, 227)
        )
        result = CEMPlanner(
            CEMConfig(4, 2, 1, 1, 7),
            SlingshotActionBounds((-100, -80), (0, 20), (40, 60)),
            Evaluator(),
        ).plan(observation)

        self.assertEqual(result.invalid_candidate_count, 1)
        self.assertEqual(result.iterations[0].elite_indices, (3, 2))
        self.assertEqual(result.actions[0], SlingshotAction(-96, 15, 54))
        self.assertEqual(result.iterations[0].candidate_failures[0], "model_failure")


class WorldModelCandidateEvaluatorTests(unittest.TestCase):
    def test_action_sequence_rolls_only_predicted_carriers_and_records_horizons(self) -> None:
        class DeterministicActionConditionedModel:
            def __init__(self) -> None:
                self.input_pig_activity: list[float] = []

            def rollout(self, observation, carrier, action):
                self.input_pig_activity.append(float(carrier[2]))
                predicted = carrier.clone()
                predicted[2] = max(0.0, float(predicted[2]) - 0.6)
                return PredictedTransition(
                    carrier=predicted,
                    requested_horizons=(15,),
                    effective_horizons=(12,),
                    compute=2.0,
                    physical_penalty=0.25,
                    rollout_penalty=0.5,
                )

        model = DeterministicActionConditionedModel()
        bounds = SlingshotActionBounds((-120, -80), (-40, 40), (0, 1000))
        evaluator = WorldModelCandidateEvaluator(
            model,
            bounds,
            GameplayCost(GameplayCostConfig(
                goal_progress_weight=1.0,
                terminal_success_cost=-5.0,
                terminal_failure_cost=8.0,
                illegal_action_cost=20.0,
                physical_penalty_weight=2.0,
                rollout_penalty_weight=1.0,
                compute_weight=0.25,
            )),
        )
        carrier = torch.zeros(15)
        carrier[2] = 1.0
        observation = PlanningObservation(
            "observed-now",
            carrier,
            (0,),
            (312, 227),
            parser_diagnostics={"structure_unstable_probability": 0.9},
        )
        actions = (
            SlingshotAction(-100, 10, 0),
            SlingshotAction(-90, 20, 70),
        )

        result = evaluator.evaluate(observation, actions)

        self.assertEqual(model.input_pig_activity[0], 1.0)
        self.assertAlmostEqual(model.input_pig_activity[1], 0.4)
        self.assertEqual(result.requested_horizons, (15, 15))
        self.assertEqual(result.effective_horizons, (12, 12))
        self.assertEqual(result.model_rollout_count, 2)
        self.assertAlmostEqual(float(result.predicted_carriers[0][2]), 0.4)
        self.assertEqual(float(result.predicted_carriers[1][2]), 0.0)
        self.assertAlmostEqual(result.total_cost, -3.0)


class FrozenCohortV2WorldModelTests(unittest.TestCase):
    def test_frozen_controller_pair_drives_recursive_action_conditioned_rollout(self) -> None:
        class FixedH5ContinuousController(torch.nn.Module):
            def forward(self, features):
                logits = torch.zeros((features.shape[0], 9))
                logits[:, 3] = 1.0
                return logits

        class RecordingPredictor(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.device_anchor = torch.nn.Parameter(torch.zeros(()))
                self.calls = []

            def carrier(self, carrier, action, request):
                self.calls.append((carrier.clone(), action.clone(), request))
                return carrier + 0.25

        image = Image.new("RGB", (2, 2), (20, 40, 60))
        encoded = BytesIO()
        image.save(encoded, format="PNG")
        predictor = RecordingPredictor()
        compute = CohortV2ComputeCalibration(
            authority="fixture-compute",
            unit="multiply_accumulate",
            controller_per_decision=1.0,
            continuous_adapter_per_decision=2.0,
            micro_adapter_per_decision=20.0,
            macro_adapter_per_decision=30.0,
            micro_graph_base_per_decision=40.0,
            micro_graph_per_entity=0.0,
            micro_graph_per_contact=0.0,
            micro_graph_per_support=0.0,
            transition_per_decision=3.0,
            continuous_readout_per_decision=4.0,
            micro_readout_per_decision=50.0,
            macro_readout_per_decision=60.0,
            shared_initial_perception_per_rollout=70.0,
        )
        config = CohortV2ControllerConfig(image_height=1, image_width=1)
        model = FrozenCohortV2WorldModel(
            predictor=predictor,
            pair_controller=FixedH5ContinuousController(),
            controller_codec=CohortV2ControllerFeatureCodec(config),
            compute=compute,
            fixed_steps_per_shot=12,
            release_time_ms=600,
        )
        observation = PlanningObservation(
            identity="rgb-now",
            carrier=torch.zeros(15),
            pig_slots=(0,),
            slingshot_anchor=(312, 227),
            agent_rgb=encoded.getvalue(),
        )

        result = model.rollout(
            observation,
            observation.carrier,
            SlingshotAction(-100, 20, 70),
        )

        self.assertEqual(result.requested_horizons, (5, 5, 5))
        self.assertEqual(result.effective_horizons, (5, 5, 2))
        self.assertEqual(len(predictor.calls), 3)
        self.assertTrue(torch.equal(predictor.calls[1][0], torch.full((1, 15), 0.25)))
        self.assertTrue(torch.allclose(
            predictor.calls[0][1],
            torch.tensor([[-100 / 480, 20 / 480, 0.6, 0.07, 1.0]]),
        ))
        self.assertEqual(result.compute, 30.0)


class VisualPlanningObservationAdapterTests(unittest.TestCase):
    def test_agent_rgb_builds_the_live_carrier_and_logs_raw_instability(self) -> None:
        class FrozenParserFixture(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.device_anchor = torch.nn.Parameter(torch.zeros(()))
                self.object_vocabulary = ("pig:0000", "block:0000")
                self.config = type("Config", (), {"image_width": 2, "image_height": 2})()

            def forward(self, images):
                relation_logits = torch.full((1, 2, 2, 2), -10.0)
                relation_logits[0, 0, 1, 0] = 10.0
                return {
                    "presence_logits": torch.tensor([[10.0, 10.0]]),
                    "centers": torch.tensor([[[0.75, 0.25], [0.2, 0.8]]]),
                    "kind_logits": torch.zeros((1, 2, 7)),
                    "relation_logits": relation_logits,
                    "macro_logits": torch.tensor([[0.0, 2.1972246]]),
                }

        image = Image.new("RGB", (2, 2), (30, 60, 90))
        encoded = BytesIO()
        image.save(encoded, format="PNG")
        adapter = VisualPlanningObservationAdapter(
            FrozenParserFixture(),
            parser_checkpoint_identity="visual-parser-fixture",
            temperatures={
                "object_presence": 1.0,
                "contact": 1.0,
                "supports": 1.0,
                "steady-state": 1.0,
                "structure-unstable": 1.0,
            },
            thresholds={
                "object_presence": 0.5,
                "contact": 0.5,
                "supports": 0.5,
                "steady-state": 0.5,
                "structure-unstable": 0.5,
            },
            latent_dim=28,
            max_entities=2,
        )

        observation = adapter.from_agent_rgb(
            identity="live-observation-1",
            png=encoded.getvalue(),
            slingshot_anchor=(312, 227),
            terminal_status=TerminalStatus.ONGOING,
        )

        self.assertEqual(observation.pig_slots, (0,))
        self.assertAlmostEqual(float(observation.carrier[2]), 0.99995, places=4)
        self.assertAlmostEqual(float(observation.carrier[8]), 0.25, places=4)
        self.assertAlmostEqual(float(observation.carrier[9]), 0.25, places=4)
        self.assertAlmostEqual(
            observation.parser_diagnostics["structure_unstable_probability"],
            0.9,
            places=5,
        )
        self.assertTrue(
            observation.parser_diagnostics["structure_unstable_thresholded"]
        )
        self.assertEqual(
            observation.symbols.contact.relations,
            (("runtime:pig:0000", "runtime:block:0000"),),
        )


class PlannerBaselineTests(unittest.TestCase):
    def test_random_and_heuristic_baselines_use_the_same_legal_plan_interface(self) -> None:
        bounds = SlingshotActionBounds((-120, -80), (-30, 40), (0, 100))
        observation = PlanningObservation(
            "baseline-observation", torch.zeros(15), (0,), (312, 227)
        )

        random_first = RandomLegalPlanner(bounds, sequence_length=2, seed=19).plan(observation)
        random_second = RandomLegalPlanner(bounds, sequence_length=2, seed=19).plan(observation)
        heuristic = HeuristicNoModelPlanner(bounds, sequence_length=2).plan(observation)

        self.assertEqual(random_first.actions, random_second.actions)
        self.assertTrue(all(bounds.contains(action) for action in random_first.actions))
        self.assertTrue(all(bounds.contains(action) for action in heuristic.actions))
        self.assertEqual(random_first.planner_id, "random_legal")
        self.assertEqual(heuristic.planner_id, "heuristic_no_model")


class GameplayControlTests(unittest.TestCase):
    @staticmethod
    def _observation(index: int, status=TerminalStatus.ONGOING):
        carrier = torch.zeros(15)
        carrier[2] = max(0.0, 1.0 - index * 0.5)
        return PlanningObservation(
            f"observation-{index}", carrier, (0,), (312, 227),
            terminal_status=status,
        )

    def test_mpc_executes_only_the_first_action_then_reobserves_and_replans(self) -> None:
        actions = (
            SlingshotAction(-100, 10, 0),
            SlingshotAction(-95, 15, 70),
            SlingshotAction(-90, 20, 50),
        )

        class PlannerFixture:
            planner_id = "fixture"

            def __init__(self, outer):
                self.outer = outer
                self.calls = 0

            def plan(self, observation):
                self.calls += 1
                selected = (actions[0], actions[1]) if self.calls == 1 else (actions[2], actions[1])
                predicted = (
                    self.outer._observation(self.calls).carrier,
                    self.outer._observation(2).carrier,
                )
                return PlanResult(
                    planner_id=self.planner_id,
                    seed=self.calls,
                    actions=selected,
                    selected_evaluation=CandidateEvaluation(
                        selected, -1.0, predicted_carriers=predicted,
                        model_rollout_count=2,
                    ),
                    candidate_count=4,
                    model_rollout_count=8,
                    planner_compute=16.0,
                )

        class EnvironmentFixture:
            def __init__(self, outer):
                self.outer = outer
                self.executed = []
                self.index = 0

            def observe(self):
                return self.outer._observation(0)

            def execute(self, action):
                self.executed.append(action)
                self.index += 1
                status = TerminalStatus.SUCCESS if self.index == 2 else TerminalStatus.ONGOING
                return self.outer._observation(self.index, status)

        planner = PlannerFixture(self)
        environment = EnvironmentFixture(self)

        result = run_gameplay_control(
            planner,
            environment,
            ControlConfig(ControlMode.MPC, max_shots=4, max_planner_compute=100.0),
        )

        self.assertEqual(environment.executed, [actions[0], actions[2]])
        self.assertEqual(planner.calls, 2)
        self.assertEqual(result.replan_count, 2)
        self.assertEqual(result.termination_reason, "success")
        self.assertEqual([step.recursive_rollout_error for step in result.steps], [0.0, 0.0])

    def test_open_loop_executes_the_planned_sequence_without_replanning(self) -> None:
        first = SlingshotAction(-100, 10, 0)
        second = SlingshotAction(-90, 20, 50)

        class PlannerFixture:
            planner_id = "fixture"

            def __init__(self):
                self.calls = 0

            def plan(self, observation):
                self.calls += 1
                actions = (first, second)
                return PlanResult(
                    self.planner_id,
                    1,
                    actions,
                    CandidateEvaluation(actions, 0.0),
                )

        class EnvironmentFixture:
            def __init__(self, outer):
                self.outer = outer
                self.executed = []
                self.index = 0

            def observe(self):
                return self.outer._observation(0)

            def execute(self, action):
                self.executed.append(action)
                self.index += 1
                return self.outer._observation(
                    self.index,
                    TerminalStatus.SUCCESS if self.index == 2 else TerminalStatus.ONGOING,
                )

        planner = PlannerFixture()
        environment = EnvironmentFixture(self)
        result = run_gameplay_control(
            planner,
            environment,
            ControlConfig(ControlMode.OPEN_LOOP, max_shots=4, max_planner_compute=100.0),
        )

        self.assertEqual(environment.executed, [first, second])
        self.assertEqual(planner.calls, 1)
        self.assertEqual(result.replan_count, 1)
        self.assertEqual(result.mode, ControlMode.OPEN_LOOP)


class GameplayEvidenceTests(unittest.TestCase):
    def test_evidence_binds_sources_action_cost_cem_mode_and_diagnostics(self) -> None:
        action = SlingshotAction(-100, 10, 70)
        bounds = SlingshotActionBounds((-120, -80), (-40, 40), (0, 1000), 600)
        cem = CEMConfig(4, 2, 2, 1, 7)
        control = ControlConfig(ControlMode.MPC, 3, 1000.0)

        class PlannerFixture:
            planner_id = "fixture"

            def plan(self, observation):
                return PlanResult(
                    "fixture",
                    7,
                    (action,),
                    CandidateEvaluation(
                        (action,),
                        -1.0,
                        predicted_carriers=(torch.zeros(15),),
                        requested_horizons=(15,),
                        effective_horizons=(12,),
                        model_rollout_count=1,
                        model_compute=20.0,
                    ),
                    candidate_count=4,
                    model_rollout_count=4,
                    planner_compute=80.0,
                    goal_evaluation_count=4,
                    wall_clock_seconds=0.25,
                )

        class EnvironmentFixture:
            def observe(self):
                return PlanningObservation(
                    "before", torch.zeros(15), (0,), (312, 227),
                    parser_diagnostics={
                        "structure_unstable_probability": 0.91,
                        "structure_unstable_thresholded": True,
                    },
                )

            def execute(self, selected):
                return PlanningObservation(
                    "after", torch.zeros(15), (0,), (312, 227),
                    terminal_status=TerminalStatus.SUCCESS,
                    parser_diagnostics={
                        "structure_unstable_probability": 0.12,
                        "structure_unstable_thresholded": False,
                    },
                )

        result = run_gameplay_control(PlannerFixture(), EnvironmentFixture(), control)
        bindings = GameplayEvidenceBindings(
            implementation_revision="a" * 40,
            world_model_checkpoint_identity="world-model-fixture",
            controller_checkpoint_identity="controller-fixture",
            visual_parser_checkpoint_identity="parser-fixture",
            observation_adapter_identity="visual-carrier-fixture",
            goal_cost_version="gameplay-cost-v1",
            goal_cost_config=GameplayCostConfig(1, -5, 8, 20, 2, 1, 0.01, 0),
            action_bounds=bounds,
            cem_config=cem,
            control_config=control,
            seed=7,
            level_identity="level-instance-fixture",
            environment_version="ScienceBirds-Unity-2019.4.41f2",
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            written = write_gameplay_evidence(root, result, bindings)
            validated = validate_gameplay_evidence(root, expected_bindings=bindings)

            self.assertEqual(validated, written)
            step = validated["control_result"]["steps"][0]
            self.assertEqual(step["requested_horizons"], [15])
            self.assertEqual(step["effective_horizons"], [12])
            self.assertEqual(
                step["observation_before_diagnostics"]["structure_unstable_probability"],
                0.91,
            )
            self.assertEqual(validated["source_bindings"]["control_config"]["mode"], "mpc")
            self.assertEqual(validated["control_result"]["goal_evaluation_count"], 4)
            self.assertEqual(validated["control_result"]["planner_wall_clock_seconds"], 0.25)

            changed = json.loads((root / "evidence.json").read_bytes())
            changed["control_result"]["termination_reason"] = "changed"
            (root / "evidence.json").write_bytes(canonical_json_bytes(changed))
            with self.assertRaises(ValueError):
                validate_gameplay_evidence(root)


if __name__ == "__main__":
    unittest.main()
