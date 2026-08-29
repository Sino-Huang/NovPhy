from __future__ import annotations

import unittest
from io import BytesIO

import torch
from PIL import Image

from world_model.data.deployment_temporal import (
    AgentObservation,
    DecisionTargets,
    DecisionTransition,
    DeploymentTemporalError,
    DeploymentTrajectory,
    DeploymentTrajectoryReader,
    ExecutedAction,
    TemporalObservationContext,
    TemporalVisualCarrierAdapter,
    TrajectoryLineageBinding,
    TrajectoryLineageManifest,
)
from world_model.planning.gameplay import (
    TerminalStatus,
    VisualPlanningObservationAdapter,
)
from world_model.training.deployment_temporal import DeploymentTemporalTrainingData


def _png(red: int) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (2, 2), (red, 20, 30)).save(stream, format="PNG")
    return stream.getvalue()


class FrozenParserFixture(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device_anchor = torch.nn.Parameter(torch.zeros(()))
        self.object_vocabulary = ("pig:0000", "block:0000")
        self.config = type("Config", (), {"image_width": 2, "image_height": 2})()

    def forward(self, images):
        center_x = images[:, 0].float().mean(dim=(1, 2)) / 255.0
        centers = torch.stack(
            (
                torch.stack((center_x, torch.full_like(center_x, 0.25)), dim=-1),
                torch.stack((torch.full_like(center_x, 0.2), torch.full_like(center_x, 0.8)), dim=-1),
            ),
            dim=1,
        )
        relation_logits = torch.full((len(images), 2, 2, 2), -10.0)
        relation_logits[:, 0, 1, 0] = 10.0
        kind_logits = torch.zeros((len(images), 2, 7))
        kind_logits[:, 0, 1] = 4.0
        kind_logits[:, 1, 2] = 4.0
        return {
            "presence_logits": torch.full((len(images), 2), 10.0),
            "centers": centers,
            "kind_logits": kind_logits,
            "relation_logits": relation_logits,
            "macro_logits": torch.tensor([[0.0, 2.1972246]]).repeat(len(images), 1),
        }


def _adapter(adapter_type=TemporalVisualCarrierAdapter):
    return adapter_type(
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


def _observation(identity: str, step: int, time: float, red: int) -> AgentObservation:
    return AgentObservation(
        identity=identity,
        fixed_step=step,
        fixed_time_seconds=time,
        png=_png(red),
        observation_role="agent",
    )


def _action(identity: str = "action:1") -> ExecutedAction:
    return ExecutedAction(
        identity=identity,
        interface_action={
            "action_type": "drag_hold_release",
            "drag_release": (77, 29),
            "frame_height": 480,
            "releaseTime": 1000,
            "tapTime": 0,
        },
        engine_relative_action={
            "schema": "slingshot_relative_intervention_v1",
            "drag_delta_canvas_pixels": (77, 29),
            "hold_milliseconds": 1000,
            "tap_time_milliseconds": 0,
        },
        legal=True,
    )


def _transition(
    *,
    role: str = "training",
    lineage: str = "lineage:1",
) -> DecisionTransition:
    return DecisionTransition(
        identity="transition:1",
        scenario_lineage_identity=lineage,
        exposure_role=role,
        decision_index=0,
        prior_observation=_observation("observation:0", 10, 1.0, 51),
        current_observation=_observation("observation:1", 11, 1.1, 102),
        action=_action(),
        targets=DecisionTargets(
            next_observation=_observation("observation:2", 12, 1.2, 153),
            source_frame_record_identity="frame-record:2",
            source_state_identity="state:2",
            source_targets={"contact": {"availability": "available", "relations": ()}},
        ),
        terminal_status="level_fail",
        source_bindings={"release_identity": "release:fixture"},
    )


def _trajectory(transition: DecisionTransition | None = None) -> DeploymentTrajectory:
    transition = transition or _transition()
    return DeploymentTrajectory(
        identity="trajectory:training",
        scenario_lineage_identity=transition.scenario_lineage_identity,
        exposure_role=transition.exposure_role,
        transitions=(transition,),
        complete=True,
    )


def _binding(
    trajectory: DeploymentTrajectory,
    *,
    transition_identities: tuple[str, ...] | None = None,
) -> TrajectoryLineageBinding:
    return TrajectoryLineageBinding(
        trajectory_identity=trajectory.identity,
        scenario_lineage_identity=trajectory.scenario_lineage_identity,
        exposure_role=trajectory.exposure_role,
        transition_identities=(
            transition_identities
            if transition_identities is not None
            else tuple(item.identity for item in trajectory.transitions)
        ),
        initial_observation_identity=trajectory.transitions[0].current_observation.identity,
        terminal_observation_identity=(
            trajectory.transitions[-1].targets.next_observation.identity
        ),
    )


def _manifest(trajectory: DeploymentTrajectory) -> TrajectoryLineageManifest:
    return TrajectoryLineageManifest.create(
        "release:fixture", (_binding(trajectory),)
    )


class TemporalVisualCarrierTests(unittest.TestCase):
    def test_aligned_prior_context_produces_declared_motion(self) -> None:
        adapter = _adapter()
        prior = _observation("observation:prior", 10, 1.0, 51)
        current = _observation("observation:current", 11, 1.1, 102)

        result = adapter.build(TemporalObservationContext(prior, current))

        pig = result.object_slots[0]
        self.assertTrue(pig.motion_available)
        self.assertAlmostEqual(pig.center_x, 0.4, places=5)
        self.assertAlmostEqual(pig.motion_x_per_second, 2.0, places=4)
        self.assertEqual(result.fixed_step_delta, 1)
        self.assertAlmostEqual(result.elapsed_seconds, 0.1)
        self.assertIn("motion-availability-mask", adapter.identity)

    def test_missing_context_is_unavailable_instead_of_observed_zero_motion(self) -> None:
        adapter = _adapter()
        current = _observation("observation:current", 11, 1.1, 102)

        result = adapter.build(TemporalObservationContext(None, current))

        pig = result.object_slots[0]
        self.assertFalse(pig.motion_available)
        self.assertEqual(pig.motion_x_per_second, 0.0)
        self.assertEqual(pig.motion_y_per_second, 0.0)
        motion_mask_index = 2 + 10
        self.assertEqual(float(result.tensor[motion_mask_index]), 0.0)

    def test_temporal_context_rejects_misalignment_and_canonical_input(self) -> None:
        with self.assertRaisesRegex(DeploymentTemporalError, "agent observations"):
            AgentObservation(
                identity="canonical:1",
                fixed_step=1,
                fixed_time_seconds=0.02,
                png=_png(10),
                observation_role="canonical",
            )

        current = _observation("observation:current", 10, 1.0, 20)
        future = _observation("observation:future", 11, 1.1, 30)
        with self.assertRaisesRegex(DeploymentTemporalError, "strictly precede"):
            TemporalObservationContext(future, current)


class DecisionTransitionContractTests(unittest.TestCase):
    def test_versioned_payload_binds_observations_action_terminal_and_sources(self) -> None:
        payload = _transition().to_payload()

        self.assertEqual(payload["schema"], "deployment_decision_transition_v1")
        self.assertEqual(payload["prior_observation"]["identity"], "observation:0")
        self.assertEqual(payload["current_observation"]["fixed_step"], 11)
        self.assertEqual(payload["action"]["identity"], "action:1")
        self.assertEqual(payload["next_observation"]["identity"], "observation:2")
        self.assertEqual(payload["terminal_status"], "level_fail")
        self.assertEqual(payload["source_bindings"]["release_identity"], "release:fixture")

    def test_executed_action_rejects_mismatched_interface_and_engine_bindings(self) -> None:
        with self.assertRaisesRegex(DeploymentTemporalError, "action binding"):
            ExecutedAction(
                identity="action:mismatch",
                interface_action={
                    "drag_release": (77, 29),
                    "releaseTime": 1000,
                    "tapTime": 0,
                },
                engine_relative_action={
                    "drag_delta_canvas_pixels": (76, 29),
                    "hold_milliseconds": 1000,
                    "tap_time_milliseconds": 0,
                },
                legal=True,
            )

    def test_executed_action_rejects_expert_or_target_action_inputs(self) -> None:
        with self.assertRaisesRegex(DeploymentTemporalError, "expert or target"):
            ExecutedAction(
                identity="action:expert",
                interface_action={
                    "drag_release": (77, 29),
                    "releaseTime": 1000,
                    "tapTime": 0,
                    "expert_target_action": (80, 30),
                },
                engine_relative_action={
                    "drag_delta_canvas_pixels": (77, 29),
                    "hold_milliseconds": 1000,
                    "tap_time_milliseconds": 0,
                },
                legal=True,
            )

    def test_inference_context_binds_the_executed_action_without_future_targets(self) -> None:
        transition = _transition()

        inference = transition.inference

        self.assertIs(inference.action, transition.action)
        self.assertEqual(inference.observations.current.identity, "observation:1")
        self.assertFalse(hasattr(inference, "next_observation"))
        self.assertFalse(hasattr(inference, "source_targets"))
        self.assertEqual(transition.schema, "deployment_decision_transition_v1")

    def test_reader_rejects_split_decisions_and_cross_role_lineage_reuse(self) -> None:
        transition = _transition()
        trajectory = _trajectory(transition)

        with self.assertRaisesRegex(DeploymentTemporalError, "complete trajectories"):
            DeploymentTrajectoryReader(
                (transition,),
                exposure_role="training",
                lineage_manifest=_manifest(trajectory),
            )

        with self.assertRaisesRegex(DeploymentTemporalError, "decision inventory"):
            DeploymentTrajectoryReader(
                (trajectory,),
                exposure_role="training",
                lineage_manifest=TrajectoryLineageManifest.create(
                    "release:fixture",
                    (_binding(
                        trajectory,
                        transition_identities=("transition:missing", "transition:1"),
                    ),),
                ),
            )

        leaked = DeploymentTrajectory(
            identity="trajectory:calibration",
            scenario_lineage_identity="lineage:1",
            exposure_role="calibration",
            transitions=(_transition(role="calibration"),),
            complete=True,
        )
        with self.assertRaisesRegex(DeploymentTemporalError, "exposure role"):
            DeploymentTrajectoryReader.validate_role_isolation((
                DeploymentTrajectoryReader(
                    (trajectory,),
                    exposure_role="training",
                    lineage_manifest=_manifest(trajectory),
                ),
                DeploymentTrajectoryReader(
                    (leaked,),
                    exposure_role="calibration",
                    lineage_manifest=_manifest(leaked),
                ),
            ))

    def test_training_and_gameplay_use_identical_carrier_construction(self) -> None:
        adapter = _adapter(VisualPlanningObservationAdapter)
        transition = _transition()
        trajectory = _trajectory(transition)
        reader = DeploymentTrajectoryReader(
            (trajectory,),
            exposure_role="training",
            lineage_manifest=_manifest(trajectory),
        )

        training_trajectory = DeploymentTemporalTrainingData(reader, adapter)[0]
        training = training_trajectory.transitions[0]
        gameplay = adapter.from_temporal_context(
            transition.inference.observations,
            slingshot_anchor=(312, 227),
            terminal_status=TerminalStatus.ONGOING,
        )

        torch.testing.assert_close(training.context.tensor, gameplay.carrier)
        self.assertEqual(
            training.context.adapter_identity,
            gameplay.parser_diagnostics["carrier_adapter_identity"],
        )
        self.assertIs(training.action, transition.action)
        self.assertEqual(
            training_trajectory.scenario_lineage_identity,
            transition.scenario_lineage_identity,
        )
        self.assertEqual(training_trajectory.exposure_role, "training")


if __name__ == "__main__":
    unittest.main()
