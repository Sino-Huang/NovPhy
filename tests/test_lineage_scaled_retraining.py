from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.run_issue_61_lineage_scaling import main as issue_61_main
from world_model.data.deployment_temporal import (
    TemporalVisualCarrierAdapter,
    TrajectoryLineageBinding,
    TrajectoryLineageManifest,
)
from world_model.model import DualOutputPredictor, PredictorConfig, identity
from world_model.planning import (
    GameplayCostConfig,
    PlanningObservation,
    SlingshotAction,
    SlingshotActionBounds,
)
from world_model.training.lineage_scaling import (
    ActionCandidate,
    ActionRankingState,
    CarrierKind,
    CarrierLineage,
    ContinuousTransitionExample,
    build_matched_gameplay_planners,
    FrozenLineageScale,
    FrozenRankingState,
    GameplayCheckpointBindings,
    LineageScalingError,
    LineageScalingProtocol,
    LoadedAdaptiveHorizonSelector,
    LoadedGameplayPredictor,
    MatchedGameplayProtocol,
    TrainingCell,
    evaluate_action_ranking,
    evaluate_continuous_prediction,
    gameplay_checkpoint_file_identity,
    gameplay_predictor_protocol_identity,
    load_carrier_lineage_bundle,
    load_action_ranking_bundle,
    load_lineage_scaling_protocol,
    load_lineage_scaled_checkpoint,
    matched_gameplay_systems,
    save_carrier_lineage_bundle,
    save_action_ranking_bundle,
    save_lineage_scaling_protocol,
    save_lineage_scaled_checkpoint,
    train_continuous_predictor,
    validate_action_ranking_states,
    validate_matched_carrier_lineages,
    validate_lineage_scaled_checkpoint_matrix,
    validate_matched_action_ranking_states,
)
from world_model.training.cohort_v2_micro import CohortV2StateCodec


def _manifest(
    role: str,
    lineage_ids: tuple[str, ...],
    *,
    release: str = "release:fixture",
) -> TrajectoryLineageManifest:
    return TrajectoryLineageManifest.create(
        release,
        tuple(
            TrajectoryLineageBinding(
                trajectory_identity=f"trajectory:{lineage}",
                scenario_lineage_identity=lineage,
                exposure_role=role,
                transition_identities=tuple(
                    f"transition:{lineage}:d{position}:h1"
                    for position in range(15)
                ) + (f"transition:{lineage}:d0:h15",),
                initial_observation_identity=f"observation:{lineage}:0",
                terminal_observation_identity=f"observation:{lineage}:1",
            )
            for lineage in lineage_ids
        ),
    )


def _legal_candidate_set_identity(state_identity: str) -> str:
    return identity((
        "legal-action-candidate-set-v1",
        state_identity,
        ((-160, -40), (-80, 80), (0, 1000), 600, 400),
        (
            (
                "candidate:a",
                torch.tensor((-0.2, 0.1, 0.6, 0.0, 1.0)).tolist(),
                (-80, 40, 0),
            ),
            (
                "candidate:b",
                torch.tensor((-0.1, 0.2, 0.6, 0.0, 1.0)).tolist(),
                (-40, 80, 0),
            ),
        ),
    ))


def _protocol(*, budget: int = 8) -> LineageScalingProtocol:
    small_ids = tuple(f"lineage:train:{index}" for index in range(6))
    full_ids = small_ids + tuple(f"lineage:train:{index}" for index in range(6, 8))
    return LineageScalingProtocol(
        training_scales=(
            FrozenLineageScale.from_manifest("six", _manifest("training", small_ids)),
            FrozenLineageScale.from_manifest("full", _manifest("training", full_ids)),
        ),
        evaluation_manifests=(
            _manifest("calibration", ("lineage:calibration:0",)),
            _manifest("model_selection", ("lineage:model-selection:0",)),
        ),
        ranking_states=(
            FrozenRankingState(
                identity="ranking-state:runner",
                scenario_lineage_identity="lineage:calibration:0",
                trajectory_identity="trajectory:lineage:calibration:0",
                decision_transition_identity=(
                    "transition:lineage:calibration:0:d0:h1"
                ),
                exposure_role="calibration",
                legal_candidate_set_identity=_legal_candidate_set_identity(
                    "ranking-state:runner"
                ),
            ),
            FrozenRankingState(
                identity="ranking-state:0",
                scenario_lineage_identity="lineage:model-selection:0",
                trajectory_identity="trajectory:lineage:model-selection:0",
                decision_transition_identity=(
                    "transition:lineage:model-selection:0:d0:h1"
                ),
                exposure_role="model_selection",
                legal_candidate_set_identity=_legal_candidate_set_identity(
                    "ranking-state:0"
                ),
            ),
        ),
        training_seeds=(11, 12, 13),
        training_horizons=(1, 15),
        optimizer_example_budget=budget,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        predictor_config=PredictorConfig(
            latent_dim=15,
            action_dim=5,
            hidden_dim=16,
            depth=1,
            pair_code_dim=4,
            delta_frequency_count=2,
        ),
        source_max_entities=1,
        source_carrier_identity=CohortV2StateCodec(
            latent_dim=15, max_entities=1
        ).identity,
        deployment_carrier_identity=TemporalVisualCarrierAdapter.identity,
        configuration_basis="prospectively_frozen",
    )


def _lineages(
    protocol: LineageScalingProtocol,
    cell: TrainingCell,
) -> tuple[CarrierLineage, ...]:
    scale = protocol.scale(cell.scale_name)
    carrier_identity = protocol.carrier_identity(cell.carrier)
    result = []
    for index, lineage_identity in enumerate(scale.lineage_identities):
        start = torch.zeros(15)
        start[index % 15] = 1.0
        target = start.clone()
        target[(index + 1) % 15] = 0.5
        target_h15 = start.clone()
        target_h15[(index + 2) % 15] = 0.75
        h1_transitions = tuple(
            ContinuousTransitionExample(
                identity=f"transition:{lineage_identity}:d{position}:h1",
                context=start,
                action=torch.tensor((-0.2, 0.1, 0.6, 0.0, 1.0)),
                target=target,
                physical_diagnostics={"unstable": False},
                decision_index=position,
                horizon=1,
                target_decision_index=position + 1,
            )
            for position in range(15)
        )
        result.append(CarrierLineage(
            trajectory_identity=f"trajectory:{lineage_identity}",
            scenario_lineage_identity=lineage_identity,
            exposure_role="training",
            source_release_identity=scale.source_release_identity,
            carrier=cell.carrier,
            carrier_identity=carrier_identity,
            transitions=h1_transitions + (
                ContinuousTransitionExample(
                    identity=f"transition:{lineage_identity}:d0:h15",
                    context=start,
                    action=torch.tensor((-0.2, 0.1, 0.6, 0.0, 1.0)),
                    target=target_h15,
                    physical_diagnostics={"unstable": False},
                    decision_index=0,
                    horizon=15,
                    target_decision_index=15,
                ),
            ),
            complete=True,
            decision_count=15,
        ))
    return tuple(result)


class LineageScalingContractTests(unittest.TestCase):
    def test_protocol_and_carrier_bundle_round_trip_exact_membership(self) -> None:
        protocol = _protocol()
        cell = next(
            item for item in protocol.cells
            if item.scale_name == "full"
            and item.carrier is CarrierKind.SOURCE
            and item.seed == 11
        )
        lineages = _lineages(protocol, cell)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "protocol.json"
            bundle_path = root / "source.pt"
            save_lineage_scaling_protocol(protocol_path, protocol)
            save_carrier_lineage_bundle(bundle_path, lineages)

            loaded_protocol = load_lineage_scaling_protocol(protocol_path)
            loaded_lineages = load_carrier_lineage_bundle(bundle_path)

        self.assertEqual(loaded_protocol.identity, protocol.identity)
        self.assertEqual(loaded_protocol.cells, protocol.cells)
        self.assertEqual(
            tuple(item.scenario_lineage_identity for item in loaded_lineages),
            tuple(item.scenario_lineage_identity for item in lineages),
        )
        for expected, actual in zip(lineages, loaded_lineages, strict=True):
            torch.testing.assert_close(
                expected.transitions[0].context,
                actual.transitions[0].context,
            )

    def test_protocol_builds_the_matched_carrier_scale_seed_matrix(self) -> None:
        protocol = _protocol()

        self.assertEqual(len(protocol.cells), 12)
        self.assertEqual(
            {
                (cell.scale_name, cell.carrier, cell.seed)
                for cell in protocol.cells
            },
            {
                (scale, carrier, seed)
                for scale in ("six", "full")
                for carrier in (CarrierKind.SOURCE, CarrierKind.DEPLOYMENT)
                for seed in (11, 12, 13)
            },
        )
        self.assertEqual(len(protocol.primary_cells), 12)

    def test_protocol_rejects_non_nested_or_role_leaking_lineages(self) -> None:
        protocol = _protocol()
        full = protocol.training_scales[1]
        non_nested = FrozenLineageScale.from_manifest(
            "full",
            _manifest(
                "training",
                tuple(f"lineage:other:{index}" for index in range(8)),
            ),
        )
        with self.assertRaisesRegex(LineageScalingError, "nested"):
            replace(protocol, training_scales=(protocol.training_scales[0], non_nested))

        leaking = _manifest("calibration", (full.lineage_identities[0],))
        with self.assertRaisesRegex(LineageScalingError, "leaked"):
            replace(
                protocol,
                evaluation_manifests=(leaking, protocol.evaluation_manifests[1]),
            )

    def test_protocol_rejects_outcome_conditioned_configuration(self) -> None:
        with self.assertRaisesRegex(LineageScalingError, "prospectively"):
            replace(_protocol(), configuration_basis="best_observed_validation_loss")


class LineageScalingTrainingTests(unittest.TestCase):
    def test_source_and_deployment_carriers_must_bind_the_same_transitions(self) -> None:
        protocol = _protocol()
        source_cell = next(
            item for item in protocol.cells
            if item.scale_name == "full"
            and item.carrier is CarrierKind.SOURCE
            and item.seed == 11
        )
        deployment_cell = replace(source_cell, carrier=CarrierKind.DEPLOYMENT)
        source = _lineages(protocol, source_cell)
        deployment = _lineages(protocol, deployment_cell)

        result = validate_matched_carrier_lineages(
            protocol,
            source,
            deployment,
        )
        self.assertEqual(result["lineage_count"], 8)
        self.assertEqual(result["transition_count"], 128)

        changed = replace(
            deployment[0],
            transitions=(replace(
                deployment[0].transitions[0],
                action=torch.tensor((-0.3, 0.1, 0.6, 0.0, 1.0)),
            ),),
        )
        with self.assertRaisesRegex(LineageScalingError, "carrier alignment"):
            validate_matched_carrier_lineages(
                protocol,
                source,
                (changed, *deployment[1:]),
            )

        incomplete_source = (replace(source[0], decision_count=16), *source[1:])
        incomplete_deployment = (
            replace(deployment[0], decision_count=16),
            *deployment[1:],
        )
        with self.assertRaisesRegex(LineageScalingError, "contiguous h1"):
            validate_matched_carrier_lineages(
                protocol,
                incomplete_source,
                incomplete_deployment,
            )

    def test_training_uses_the_exact_example_budget_and_checkpoint_binding(self) -> None:
        protocol = _protocol(budget=8)
        cell = next(
            item for item in protocol.cells
            if item.scale_name == "six"
            and item.carrier is CarrierKind.DEPLOYMENT
            and item.seed == 11
        )
        model, report = train_continuous_predictor(
            protocol,
            cell,
            _lineages(protocol, cell),
            device="cpu",
        )

        self.assertEqual(report.optimizer_examples, 8)
        self.assertEqual(report.optimizer_steps, 4)
        self.assertAlmostEqual(report.epochs, 8 / 96)
        self.assertEqual(report.lineage_count, 6)
        self.assertEqual(report.available_horizon_counts, ((1, 90), (15, 6)))
        self.assertEqual(report.optimizer_horizon_counts, ((1, 4), (15, 4)))
        self.assertEqual(report.carrier_identity, protocol.deployment_carrier_identity)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            saved = save_lineage_scaled_checkpoint(path, model, report)
            loaded, metadata = load_lineage_scaled_checkpoint(
                path,
                protocol,
                expected_cell=cell,
                device="cpu",
            )
            self.assertEqual(metadata.identity, saved.identity)
            for expected, actual in zip(
                model.state_dict().values(), loaded.state_dict().values(), strict=True
            ):
                torch.testing.assert_close(expected, actual)

            tampered_path = Path(temporary) / "tampered.pt"
            tampered = torch.load(path, map_location="cpu", weights_only=True)
            tampered["metadata"]["epochs"] = 99.0
            torch.save(tampered, tampered_path)
            with self.assertRaisesRegex(LineageScalingError, "checkpoint"):
                load_lineage_scaled_checkpoint(
                    tampered_path,
                    protocol,
                    expected_cell=cell,
                    device="cpu",
                )

            wrong_cell = replace(cell, carrier=CarrierKind.SOURCE)
            with self.assertRaisesRegex(LineageScalingError, "checkpoint"):
                load_lineage_scaled_checkpoint(
                    path,
                    protocol,
                    expected_cell=wrong_cell,
                    device="cpu",
                )

            with self.assertRaisesRegex(LineageScalingError, "complete checkpoint matrix"):
                validate_lineage_scaled_checkpoint_matrix(
                    protocol,
                    {cell: path},
                    device="cpu",
                )

    def test_training_rejects_transition_membership_not_frozen_by_the_manifest(self) -> None:
        protocol = _protocol()
        cell = protocol.cells[0]
        lineages = _lineages(protocol, cell)
        changed = replace(
            lineages[0],
            transitions=(
                replace(lineages[0].transitions[0], identity="transition:injected"),
                lineages[0].transitions[1],
            ),
        )
        with self.assertRaisesRegex(LineageScalingError, "trajectory manifest"):
            train_continuous_predictor(
                protocol,
                cell,
                (changed, *lineages[1:]),
                device="cpu",
            )

    def test_h1_and_h15_use_their_own_targets_and_recursive_auc_spacing(self) -> None:
        protocol = _protocol()
        cell = protocol.cells[0]
        lineage = _lineages(protocol, cell)[0]
        model = DualOutputPredictor(protocol.predictor_config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()

        evaluation = evaluate_continuous_prediction(
            model,
            (lineage,),
            horizons=(1, 15),
        )

        h1_error = 0.5 ** 2 / 15
        h15_error = 0.75 ** 2 / 15
        self.assertAlmostEqual(evaluation.local_by_horizon[1], h1_error)
        self.assertAlmostEqual(evaluation.local_by_horizon[15], h15_error)
        self.assertAlmostEqual(evaluation.recursive[0].error_auc, h1_error * 14.5)
        self.assertAlmostEqual(
            evaluation.recursive[1].error_auc, h15_error * 15 / 2
        )

    def test_prediction_and_action_ranking_report_separate_estimands(self) -> None:
        protocol = _protocol(budget=4)
        cell = protocol.cells[0]
        lineages = _lineages(protocol, cell)
        model, _report = train_continuous_predictor(
            protocol, cell, lineages, device="cpu"
        )

        prediction = evaluate_continuous_prediction(
            model,
            lineages[:1],
            horizons=(1, 15),
            physical_diagnostic=lambda value: {
                "carrier_norm": float(torch.linalg.vector_norm(value))
            },
        )
        self.assertIsNotNone(prediction.local_mse)
        self.assertEqual(tuple(item.horizon for item in prediction.recursive), (1, 15))
        self.assertEqual(prediction.nonfinite_failures, 0)
        self.assertEqual(prediction.execution_failures, ())
        self.assertEqual(prediction.model_evaluations, 32)
        self.assertGreaterEqual(prediction.wall_seconds, 0.0)
        self.assertIn("unstable", prediction.target_physical_diagnostics)
        self.assertIn("carrier_norm", prediction.predicted_physical_diagnostics)

        transition = lineages[0].transitions[0]
        state = ActionRankingState(
            identity="ranking-state:0",
            scenario_lineage_identity="lineage:model-selection:0",
            trajectory_identity="trajectory:lineage:model-selection:0",
            decision_transition_identity="transition:lineage:model-selection:0:d0:h1",
            exposure_role="model_selection",
            carrier=cell.carrier,
            carrier_identity=protocol.carrier_identity(cell.carrier),
            context=transition.context,
            candidates=(
                ActionCandidate(
                    "candidate:a",
                    transition.action,
                    2.0,
                    SlingshotAction(-80, 40, 0),
                ),
                ActionCandidate(
                    "candidate:b",
                    torch.tensor((-0.1, 0.2, 0.6, 0.0, 1.0)),
                    1.0,
                    SlingshotAction(-40, 80, 0),
                ),
            ),
            action_bounds=SlingshotActionBounds(
                drag_x=(-160, -40),
                drag_y=(-80, 80),
                tap_time_ms=(0, 1000),
            ),
            frame_height=400,
            cost_target=torch.zeros(15),
        )
        ranking = evaluate_action_ranking(
            model,
            (state,),
            horizon=1,
            predicted_cost=lambda _state, candidate, _prediction: (
                0.0 if candidate.identity == "candidate:a" else 1.0
            ),
        )
        self.assertEqual(ranking.state_count, 1)
        self.assertEqual(ranking.mean_top_action_regret, 1.0)
        self.assertEqual(ranking.states[0].selected_candidate_identity, "candidate:a")
        self.assertEqual(ranking.model_evaluations, 2)
        self.assertGreaterEqual(ranking.wall_seconds, 0.0)
        matched = validate_matched_action_ranking_states(
            protocol,
            (state,),
            (replace(
                state,
                carrier=CarrierKind.DEPLOYMENT,
                carrier_identity=protocol.deployment_carrier_identity,
            ),),
        )
        self.assertEqual(matched["candidate_count"], 2)
        with self.assertRaisesRegex(LineageScalingError, "candidate"):
            validate_action_ranking_states(
                protocol,
                (replace(state, candidates=tuple(reversed(state.candidates))),),
                carrier=CarrierKind.SOURCE,
            )
        extra_state = FrozenRankingState(
            identity="ranking-state:1",
            scenario_lineage_identity="lineage:model-selection:0",
            trajectory_identity="trajectory:lineage:model-selection:0",
            decision_transition_identity=(
                "transition:lineage:model-selection:0:d1:h1"
            ),
            exposure_role="model_selection",
            legal_candidate_set_identity=_legal_candidate_set_identity(
                "ranking-state:1"
            ),
        )
        expanded_protocol = replace(
            protocol,
            ranking_states=(*protocol.ranking_states, extra_state),
        )
        with self.assertRaisesRegex(LineageScalingError, "frozen state set"):
            validate_action_ranking_states(
                expanded_protocol,
                (state,),
                carrier=CarrierKind.SOURCE,
            )


class LineageScalingGameplayTests(unittest.TestCase):
    def test_gameplay_systems_bind_explicit_checkpoints_and_matched_limits(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        checkpoint_root = Path(temporary.name)
        legacy_path = checkpoint_root / "legacy.pt"
        retrained_path = checkpoint_root / "retrained.pt"
        controller_path = checkpoint_root / "controller.pt"
        legacy_path.write_bytes(b"legacy-checkpoint-fixture")
        torch.save(
            {"protocol_identity": "retrained-protocol:fixture"},
            retrained_path,
        )
        controller_path.write_bytes(b"controller-checkpoint-fixture")
        protocol = MatchedGameplayProtocol(
            action_candidate_set_identity="legal-actions:v1",
            cost_terms_identity="gameplay-cost:v1",
            action_bounds=SlingshotActionBounds(
                drag_x=(-160, -40),
                drag_y=(-80, 80),
                tap_time_ms=(0, 1000),
            ),
            cost_config=GameplayCostConfig(
                goal_progress_weight=10.0,
                terminal_success_cost=-20.0,
                terminal_failure_cost=20.0,
                illegal_action_cost=50.0,
                physical_penalty_weight=1.0,
                rollout_penalty_weight=0.1,
                compute_weight=1e-8,
            ),
            population_size=16,
            elite_count=4,
            cem_iterations=2,
            sequence_length=3,
            max_shots=6,
            max_planner_compute=1_000_000.0,
            fixed_steps_per_shot=15,
            transition_compute=10.0,
            controller_compute=2.0,
        )
        bindings = GameplayCheckpointBindings(
            legacy_predictor=legacy_path,
            legacy_predictor_identity=gameplay_checkpoint_file_identity(legacy_path),
            legacy_carrier_identity=CohortV2StateCodec(
                latent_dim=15, max_entities=1
            ).identity,
            retrained_predictor=retrained_path,
            retrained_predictor_identity=gameplay_checkpoint_file_identity(
                retrained_path
            ),
            retrained_carrier_identity=TemporalVisualCarrierAdapter.identity,
            retrained_protocol_identity=gameplay_predictor_protocol_identity(
                retrained_path
            ),
            adaptive_controller=controller_path,
            adaptive_controller_identity=gameplay_checkpoint_file_identity(
                controller_path
            ),
        )

        systems = matched_gameplay_systems(protocol, bindings)

        self.assertEqual(len(systems), 6)
        self.assertEqual(
            {system.mode for system in systems},
            {"continuous-h1", "continuous-h15", "adaptive"},
        )
        self.assertEqual(
            {system.predictor_checkpoint for system in systems},
            {bindings.legacy_predictor, bindings.retrained_predictor},
        )
        self.assertEqual({system.protocol_identity for system in systems}, {protocol.identity})
        self.assertTrue(
            all(
                system.controller_checkpoint == bindings.adaptive_controller
                for system in systems
                if system.mode == "adaptive"
            )
        )

        predictor = DualOutputPredictor(PredictorConfig(
            latent_dim=15,
            action_dim=5,
            hidden_dim=16,
            depth=1,
            pair_code_dim=4,
            delta_frequency_count=2,
        ))
        built = build_matched_gameplay_planners(
            protocol,
            systems,
            predictor_loader=lambda system: LoadedGameplayPredictor(
                predictor=predictor,
                checkpoint_role=system.checkpoint_role,
                checkpoint_identity=system.predictor_checkpoint_identity,
                carrier_identity=system.carrier_identity,
                protocol_identity=system.predictor_protocol_identity,
            ),
            adaptive_selector_loader=lambda system: (
                LoadedAdaptiveHorizonSelector(
                    selector=lambda _observation, _action: 15,
                    checkpoint_identity=system.controller_checkpoint_identity,
                )
            ),
        )
        observation = PlanningObservation(
            identity="planning:fixture",
            carrier=torch.zeros(15),
            pig_slots=(0,),
            slingshot_anchor=(312, 227),
        )
        action = SlingshotAction(-100, 0, 0)
        requested = {
            item.system.mode: item.world_model.rollout(
                observation,
                observation.carrier,
                action,
            ).requested_horizons[0]
            for item in built[:3]
        }
        self.assertEqual(
            requested,
            {"continuous-h1": 1, "continuous-h15": 15, "adaptive": 15},
        )
        self.assertEqual({item.control.max_shots for item in built}, {6})
        controller_path.write_bytes(b"changed-controller-checkpoint")
        with self.assertRaisesRegex(LineageScalingError, "changed after binding"):
            build_matched_gameplay_planners(
                protocol,
                systems,
                predictor_loader=lambda system: LoadedGameplayPredictor(
                    predictor=predictor,
                    checkpoint_role=system.checkpoint_role,
                    checkpoint_identity=system.predictor_checkpoint_identity,
                    carrier_identity=system.carrier_identity,
                    protocol_identity=system.predictor_protocol_identity,
                ),
                adaptive_selector_loader=lambda system: (
                    LoadedAdaptiveHorizonSelector(
                        selector=lambda _observation, _action: 15,
                        checkpoint_identity=system.controller_checkpoint_identity,
                    )
                ),
            )

    def test_unbuffered_runner_trains_one_explicit_cell(self) -> None:
        protocol = _protocol(budget=4)
        source_cell = next(
            item for item in protocol.cells
            if item.scale_name == "full"
            and item.carrier is CarrierKind.SOURCE
            and item.seed == 11
        )
        deployment_cell = replace(source_cell, carrier=CarrierKind.DEPLOYMENT)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "protocol.json"
            source_path = root / "source.pt"
            deployment_path = root / "deployment.pt"
            checkpoint_path = root / "trained.pt"
            evaluation_path = root / "calibration.pt"
            score_path = root / "score.json"
            ranking_path = root / "ranking.pt"
            deployment_ranking_path = root / "deployment-ranking.pt"
            ranking_output = root / "ranking.json"
            save_lineage_scaling_protocol(protocol_path, protocol)
            save_carrier_lineage_bundle(
                source_path, _lineages(protocol, source_cell)
            )
            save_carrier_lineage_bundle(
                deployment_path, _lineages(protocol, deployment_cell)
            )

            result = issue_61_main([
                "--train",
                "--protocol", str(protocol_path),
                "--source-bundle", str(source_path),
                "--deployment-bundle", str(deployment_path),
                "--scale", "six",
                "--carrier", "source",
                "--seed", "11",
                "--checkpoint", str(checkpoint_path),
                "--device", "cpu",
            ])

            self.assertEqual(result, 0)
            self.assertTrue(checkpoint_path.is_file())

            evaluation_lineage = replace(
                _lineages(protocol, source_cell)[0],
                trajectory_identity="trajectory:lineage:calibration:0",
                scenario_lineage_identity="lineage:calibration:0",
                exposure_role="calibration",
                transitions=tuple(
                    replace(
                        item,
                        identity=(
                            "transition:lineage:calibration:0:"
                            f"d{item.decision_index}:h{item.horizon}"
                        ),
                    )
                    for item in _lineages(protocol, source_cell)[0].transitions
                ),
            )
            save_carrier_lineage_bundle(evaluation_path, (evaluation_lineage,))
            score_result = issue_61_main([
                "--score",
                "--protocol", str(protocol_path),
                "--scale", "six",
                "--carrier", "source",
                "--seed", "11",
                "--checkpoint", str(checkpoint_path),
                "--evaluation-bundle", str(evaluation_path),
                "--output", str(score_path),
                "--device", "cpu",
            ])
            self.assertEqual(score_result, 0)
            self.assertTrue(score_path.is_file())

            transition = evaluation_lineage.transitions[0]
            ranking_state = ActionRankingState(
                identity="ranking-state:runner",
                scenario_lineage_identity=evaluation_lineage.scenario_lineage_identity,
                trajectory_identity=evaluation_lineage.trajectory_identity,
                decision_transition_identity=transition.identity,
                exposure_role="calibration",
                carrier=CarrierKind.SOURCE,
                carrier_identity=protocol.source_carrier_identity,
                context=transition.context,
                candidates=(
                    ActionCandidate(
                        "candidate:a",
                        transition.action,
                        2.0,
                        SlingshotAction(-80, 40, 0),
                    ),
                    ActionCandidate(
                        "candidate:b",
                        torch.tensor((-0.1, 0.2, 0.6, 0.0, 1.0)),
                        1.0,
                        SlingshotAction(-40, 80, 0),
                    ),
                ),
                action_bounds=SlingshotActionBounds(
                    drag_x=(-160, -40),
                    drag_y=(-80, 80),
                    tap_time_ms=(0, 1000),
                ),
                frame_height=400,
                cost_target=torch.zeros(15),
            )
            save_action_ranking_bundle(ranking_path, (ranking_state,))
            save_action_ranking_bundle(
                deployment_ranking_path,
                (replace(
                    ranking_state,
                    carrier=CarrierKind.DEPLOYMENT,
                    carrier_identity=protocol.deployment_carrier_identity,
                ),),
            )
            loaded_ranking = load_action_ranking_bundle(ranking_path)
            self.assertEqual(loaded_ranking[0].candidate_set_identity, ranking_state.candidate_set_identity)
            rank_result = issue_61_main([
                "--rank",
                "--protocol", str(protocol_path),
                "--scale", "six",
                "--carrier", "source",
                "--seed", "11",
                "--checkpoint", str(checkpoint_path),
                "--source-ranking-bundle", str(ranking_path),
                "--deployment-ranking-bundle", str(deployment_ranking_path),
                "--output", str(ranking_output),
                "--device", "cpu",
            ])
            self.assertEqual(rank_result, 0)
            self.assertTrue(ranking_output.is_file())


if __name__ == "__main__":
    unittest.main()
