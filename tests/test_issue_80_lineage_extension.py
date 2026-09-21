"""Focused tests for the additive #61/#80 lineage-scaling gameplay extension.

The extension must add fixed-h5, fixed-mode hybrid, and the no-model
ordinal-prior arms without changing the classic six-system matrix, the
existing bindings, or any prior artifact identity.
"""
import tempfile
import unittest
from pathlib import Path

import torch

from world_model.data.deployment_temporal import TemporalVisualCarrierAdapter
from world_model.model import Abstraction, DualOutputPredictor, PredictionPair, PredictorConfig
from world_model.planning import GameplayCostConfig, SlingshotAction, SlingshotActionBounds
from world_model.planning.gameplay import PlanningObservation
from world_model.training.cohort_v2_micro import CohortV2StateCodec

TEMPORAL_TEST_CARRIER = TemporalVisualCarrierAdapter.identity
from world_model.training.lineage_scaling import (
    CLASSIC_GAMEPLAY_MODES,
    NO_MODEL_ORDINAL_PRIOR_ORDINAL,
    GameplayCheckpointBindings,
    GameplayCheckpointRole,
    GameplayPlanningMode,
    GameplaySystemSpec,
    LineageScalingError,
    LoadedAdaptiveHorizonSelector,
    LoadedGameplayPredictor,
    MatchedGameplayProtocol,
    NoModelOrdinalPriorSystem,
    build_matched_gameplay_planners,
    fixed_mode_pair,
    gameplay_checkpoint_file_identity,
    gameplay_predictor_protocol_identity,
    matched_gameplay_systems,
    ordinal_prior_candidate,
)


def _protocol() -> MatchedGameplayProtocol:
    return MatchedGameplayProtocol(
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


def _hybrid_binding_kwargs(root: Path) -> dict:
    hybrid = root / "hybrid.pt"
    hybrid.write_bytes(b"hybrid-checkpoint-fixture")
    return dict(
        hybrid_predictor=hybrid,
        hybrid_predictor_identity=gameplay_checkpoint_file_identity(hybrid),
        hybrid_carrier_identity=TEMPORAL_TEST_CARRIER,
    )


def _bindings(root: Path, *, hybrid: bool) -> GameplayCheckpointBindings:
    legacy = root / "legacy.pt"
    retrained = root / "retrained.pt"
    controller = root / "controller.pt"
    legacy.write_bytes(b"legacy-checkpoint-fixture")
    torch.save({"protocol_identity": "retrained-protocol:fixture"}, retrained)
    controller.write_bytes(b"controller-checkpoint-fixture")
    return GameplayCheckpointBindings(
        legacy_predictor=legacy,
        legacy_predictor_identity=gameplay_checkpoint_file_identity(legacy),
        legacy_carrier_identity=CohortV2StateCodec(
            latent_dim=15, max_entities=1
        ).identity,
        retrained_predictor=retrained,
        retrained_predictor_identity=gameplay_checkpoint_file_identity(retrained),
        retrained_carrier_identity=TEMPORAL_TEST_CARRIER,
        retrained_protocol_identity=gameplay_predictor_protocol_identity(retrained),
        adaptive_controller=controller,
        adaptive_controller_identity=gameplay_checkpoint_file_identity(controller),
        **(_hybrid_binding_kwargs(root) if hybrid else {}),
    )


class Issue80LineageExtensionTests(unittest.TestCase):
    def test_default_systems_are_the_unchanged_classic_six(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            protocol = _protocol()
            bindings = _bindings(Path(temporary), hybrid=False)
            systems = matched_gameplay_systems(protocol, bindings)
            self.assertEqual(len(systems), 6)
            self.assertEqual(
                {(s.checkpoint_role, s.mode) for s in systems},
                {
                    (role, mode)
                    for role in GameplayCheckpointRole
                    for mode in CLASSIC_GAMEPLAY_MODES
                },
            )
            self.assertEqual(
                {s.mode for s in systems},
                {GameplayPlanningMode.CONTINUOUS_H1,
                 GameplayPlanningMode.CONTINUOUS_H15,
                 GameplayPlanningMode.ADAPTIVE},
            )
            self.assertTrue(
                all(
                    s.fixed_horizon == {"continuous-h1": 1, "continuous-h15": 15}[s.mode.value]
                    for s in systems
                    if s.mode is not GameplayPlanningMode.ADAPTIVE
                )
            )
            self.assertTrue(
                all(
                    s.controller_checkpoint == bindings.adaptive_controller
                    for s in systems
                    if s.mode is GameplayPlanningMode.ADAPTIVE
                )
            )
            self.assertTrue(
                all(
                    s.controller_checkpoint is None
                    for s in systems
                    if s.mode is not GameplayPlanningMode.ADAPTIVE
                )
            )
            again = matched_gameplay_systems(protocol, bindings)
            self.assertEqual([s.identity for s in systems], [s.identity for s in again])

    def test_extended_systems_add_fixed_h5_hybrid_and_prior_symmetrically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            protocol = _protocol()
            bindings = _bindings(Path(temporary), hybrid=True)
            systems = matched_gameplay_systems(protocol, bindings, extended=True)
            model_systems = [s for s in systems if isinstance(s, GameplaySystemSpec)]
            priors = [s for s in systems if isinstance(s, NoModelOrdinalPriorSystem)]
            self.assertEqual(len(model_systems), 10)
            self.assertEqual(len(priors), 1)
            self.assertEqual(
                {(s.checkpoint_role, s.mode) for s in model_systems},
                {
                    (role, mode)
                    for role in GameplayCheckpointRole
                    for mode in (
                        GameplayPlanningMode.CONTINUOUS_H1,
                        GameplayPlanningMode.CONTINUOUS_H15,
                        GameplayPlanningMode.ADAPTIVE,
                        GameplayPlanningMode.CONTINUOUS_H5,
                        GameplayPlanningMode.HYBRID_FIXED,
                    )
                },
            )
            by_role_mode = {(s.checkpoint_role, s.mode): s for s in model_systems}
            for role in GameplayCheckpointRole:
                h5 = by_role_mode[(role, GameplayPlanningMode.CONTINUOUS_H5)]
                self.assertEqual(h5.fixed_horizon, 5)
                self.assertIsNone(h5.controller_checkpoint)
                self.assertIn(
                    h5.predictor_checkpoint,
                    (bindings.legacy_predictor, bindings.retrained_predictor),
                )
                hybrid = by_role_mode[(role, GameplayPlanningMode.HYBRID_FIXED)]
                self.assertEqual(hybrid.fixed_horizon, 1)
                self.assertIsNone(hybrid.controller_checkpoint)
                self.assertEqual(hybrid.predictor_checkpoint, bindings.hybrid_predictor)
                self.assertEqual(
                    hybrid.predictor_checkpoint_identity,
                    bindings.hybrid_predictor_identity,
                )
                self.assertEqual(hybrid.carrier_identity, bindings.hybrid_carrier_identity)
            prior = priors[0]
            self.assertEqual(prior.prior_ordinal, NO_MODEL_ORDINAL_PRIOR_ORDINAL)
            self.assertEqual(prior.prior_ordinal, 8)
            self.assertEqual(prior.protocol_identity, protocol.identity)
            again = matched_gameplay_systems(protocol, bindings, extended=True)
            self.assertEqual([s.identity for s in systems], [s.identity for s in again])

    def test_extended_mode_requires_bound_hybrid_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            protocol = _protocol()
            bindings = _bindings(Path(temporary), hybrid=False)
            with self.assertRaisesRegex(LineageScalingError, "hybrid checkpoint"):
                matched_gameplay_systems(protocol, bindings, extended=True)

    def test_hybrid_binding_fields_are_validated_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "legacy.pt"
            retrained = root / "retrained.pt"
            controller = root / "controller.pt"
            hybrid = root / "hybrid.pt"
            legacy.write_bytes(b"legacy-checkpoint-fixture")
            torch.save({"protocol_identity": "retrained-protocol:fixture"}, retrained)
            controller.write_bytes(b"controller-checkpoint-fixture")
            hybrid.write_bytes(b"hybrid-checkpoint-fixture")
            base = dict(
                legacy_predictor=legacy,
                legacy_predictor_identity=gameplay_checkpoint_file_identity(legacy),
                legacy_carrier_identity=CohortV2StateCodec(
                    latent_dim=15, max_entities=1
                ).identity,
                retrained_predictor=retrained,
                retrained_predictor_identity=gameplay_checkpoint_file_identity(retrained),
                retrained_carrier_identity=TEMPORAL_TEST_CARRIER,
                retrained_protocol_identity=gameplay_predictor_protocol_identity(retrained),
                adaptive_controller=controller,
                adaptive_controller_identity=gameplay_checkpoint_file_identity(controller),
            )
            with self.assertRaisesRegex(LineageScalingError, "together"):
                GameplayCheckpointBindings(**base, hybrid_predictor=hybrid)
            with self.assertRaisesRegex(LineageScalingError, "provenance is incomplete"):
                GameplayCheckpointBindings(
                    **base,
                    hybrid_predictor=hybrid,
                    hybrid_predictor_identity="sha256:wrong",
                    hybrid_carrier_identity=TEMPORAL_TEST_CARRIER,
                )
            with self.assertRaisesRegex(LineageScalingError, "distinct"):
                GameplayCheckpointBindings(
                    **base,
                    hybrid_predictor=legacy,
                    hybrid_predictor_identity=gameplay_checkpoint_file_identity(legacy),
                    hybrid_carrier_identity=TEMPORAL_TEST_CARRIER,
                )

    def test_classic_planner_builder_rejects_extended_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            protocol = _protocol()
            bindings = _bindings(Path(temporary), hybrid=True)
            classic = matched_gameplay_systems(protocol, bindings)
            self.assertEqual(len(classic), 6)
            predictor = DualOutputPredictor(
                PredictorConfig(
                    latent_dim=15,
                    action_dim=5,
                    hidden_dim=16,
                    depth=1,
                    pair_code_dim=4,
                    delta_frequency_count=2,
                )
            )

            def loader(system):
                return LoadedGameplayPredictor(
                    predictor=predictor,
                    checkpoint_role=system.checkpoint_role,
                    checkpoint_identity=system.predictor_checkpoint_identity,
                    carrier_identity=system.carrier_identity,
                    protocol_identity=system.predictor_protocol_identity,
                )

            def selector_loader(system):
                return LoadedAdaptiveHorizonSelector(
                    selector=lambda _observation, _action: 15,
                    checkpoint_identity=system.controller_checkpoint_identity,
                )

            built = build_matched_gameplay_planners(
                protocol, classic, predictor_loader=loader,
                adaptive_selector_loader=selector_loader,
            )
            self.assertEqual(
                {item.control.max_shots for item in built}, {protocol.max_shots}
            )
            observation = PlanningObservation(
                identity="planning:fixture",
                carrier=torch.zeros(15),
                pig_slots=(0,),
                slingshot_anchor=(312, 227),
            )
            action = SlingshotAction(-100, 0, 0)
            requested = {
                item.system.mode.value: item.world_model.rollout(
                    observation, observation.carrier, action
                ).requested_horizons[0]
                for item in built[:3]
            }
            self.assertEqual(
                requested, {"continuous-h1": 1, "continuous-h15": 15, "adaptive": 15}
            )
            extended = matched_gameplay_systems(protocol, bindings, extended=True)
            with self.assertRaisesRegex(LineageScalingError, "six-system matrix"):
                build_matched_gameplay_planners(
                    protocol, extended, predictor_loader=loader,
                    adaptive_selector_loader=selector_loader,
                )

    def test_fixed_mode_pairs_and_ordinal_prior_choice(self) -> None:
        self.assertEqual(
            fixed_mode_pair(GameplayPlanningMode.CONTINUOUS_H1),
            PredictionPair(1, Abstraction.CONTINUOUS),
        )
        self.assertEqual(
            fixed_mode_pair(GameplayPlanningMode.CONTINUOUS_H5),
            PredictionPair(5, Abstraction.CONTINUOUS),
        )
        self.assertEqual(
            fixed_mode_pair(GameplayPlanningMode.HYBRID_FIXED),
            PredictionPair(1, Abstraction.CONTINUOUS),
        )
        with self.assertRaises(LineageScalingError):
            fixed_mode_pair(GameplayPlanningMode.ADAPTIVE)
        self.assertEqual(ordinal_prior_candidate(range(13)), 8)
        self.assertEqual(ordinal_prior_candidate((0, 2, 8, 12)), 8)
        self.assertIsNone(ordinal_prior_candidate((0, 1, 2)))
        with self.assertRaises(LineageScalingError):
            ordinal_prior_candidate(())
        with self.assertRaises(LineageScalingError):
            ordinal_prior_candidate((8, 8))


if __name__ == "__main__":
    raise SystemExit(unittest.main())
