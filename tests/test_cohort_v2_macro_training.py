from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from world_model.data import (
    CohortV2CentralFrameRecord,
    CohortV2OracleWindow,
    CohortV2Rollout,
)
from world_model.data.cohort_v2 import (
    CAPABILITY_DECLARATION_IDENTITY,
    COHORT_V2_RELEASE_IDENTITY,
)
from world_model.model import (
    Abstraction,
    DualOutputPredictor,
    MacroReadoutHead,
    PredictionPair,
)
from world_model.training import (
    CohortV2EvaluationResult,
    CohortV2ExhaustiveEvaluator,
    CohortV2PairGrid,
    CohortV2PairOutcome,
    CohortV2StateEvaluation,
)
from world_model.training.cohort_v2_macro import (
    MACRO_CAPABILITIES,
    MACRO_EVENT_TYPES,
    MACRO_PAIRS,
    CohortV2MacroCheckpoint,
    CohortV2MacroConfig,
    CohortV2MacroError,
    CohortV2MacroPairScorer,
    load_cohort_v2_macro_checkpoint,
    macro_event_endpoint_available,
    macro_readout_loss,
    save_cohort_v2_macro_checkpoint,
    validate_cohort_v2_macro_frontier_input,
    write_cohort_v2_macro_frontier_input,
)
from world_model.training.cohort_v2_micro import CohortV2StateCodec


def _relation(relations=()):
    return {"availability": "available", "relations": relations}


def _boolean(value: bool):
    return {"availability": "available", "value": value}


def _frame(
    identity: str,
    position: int,
    *,
    steady: bool,
    unstable: bool,
    terminal_reason: str | None = None,
):
    entity = {
        "body": {
            "angular_velocity_degrees_per_second": 0.0,
            "body_type": "dynamic",
            "gravity_applicable": True,
            "gravity_scale": 1.0,
            "position": (float(position), 0.0),
            "rotation_degrees": 0.0,
            "simulated": True,
            "velocity": (0.0, 0.0),
        },
        "body_present": True,
        "entity_id": "entity:a",
        "lifecycle": "active",
        "scenario_object_id": "object:0",
    }
    return CohortV2CentralFrameRecord(
        identity=identity,
        capture_id="capture:fixture",
        state_id=identity,
        fixed_step=position,
        capture_stride=1,
        engine_state={
            "entities": (entity,),
            "world": {"gravity_vector": (0.0, -9.81)},
        },
        events=(),
        labels={
            "contact": _relation(),
            "supports": _relation(),
            "steady-state": _boolean(steady),
            "structure-unstable": _boolean(unstable),
        },
        terminal=(
            None
            if terminal_reason is None
            else {"reason": terminal_reason, "fixed_step": position}
        ),
    )


def _window(
    *,
    requested_horizon: int = 5,
    effective_horizon: int = 3,
    terminal_reason: str | None = "stable_entered",
):
    context = _frame("frame:context", 0, steady=False, unstable=True)
    target = _frame(
        "frame:target",
        effective_horizon,
        steady=True,
        unstable=False,
        terminal_reason=terminal_reason,
    )
    return CohortV2OracleWindow(
        source_release_identity=COHORT_V2_RELEASE_IDENTITY,
        capability_declaration_identity=CAPABILITY_DECLARATION_IDENTITY,
        exposure_role="calibration",
        attempt_id="attempt:fixture",
        scenario_lineage_identity="lineage:fixture",
        intervention={
            "engine_relative_action": {
                "drag_delta_canvas_pixels": (12, 3),
                "hold_milliseconds": 1000,
                "tap_time_milliseconds": 0,
            }
        },
        context_position=0,
        requested_horizon=requested_horizon,
        effective_horizon=effective_horizon,
        context=context,
        target=target,
        agent_observation=b"observation",
    )


class CohortV2MacroTrainingTests(unittest.TestCase):
    def test_terminal_endpoint_contract_accepts_valid_level_fail(self) -> None:
        self.assertTrue(macro_event_endpoint_available(_window()))
        self.assertTrue(
            macro_event_endpoint_available(_window(terminal_reason="level_fail"))
        )
        self.assertFalse(
            macro_event_endpoint_available(_window(terminal_reason=None))
        )
        self.assertEqual(MACRO_EVENT_TYPES, ("stable_entered", "level_fail"))

    def test_macro_readout_supervises_state_duration_and_event(self) -> None:
        torch.manual_seed(3)
        head = MacroReadoutHead(
            latent_dim=32,
            hidden_dim=16,
            macro_predicate_count=2,
            event_type_count=2,
        )
        carrier = torch.randn(2, 32, requires_grad=True)
        windows = (
            _window(terminal_reason="stable_entered"),
            _window(terminal_reason="level_fail"),
        )

        result = macro_readout_loss(head, carrier, windows)
        result.loss.backward()

        self.assertEqual(result.supervised_predicate_count, 4)
        self.assertEqual(result.endpoint_count, 2)
        self.assertTrue(torch.isfinite(result.loss))
        self.assertGreater(float(head.macro_projection.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(head.delta_projection.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(head.event_projection.weight.grad.abs().sum()), 0.0)

    def test_macro_content_changes_the_endpoint_carrier(self) -> None:
        config = CohortV2MacroConfig(
            steps=9,
            batch_size=1,
            latent_dim=32,
            hidden_dim=16,
            depth=1,
            max_entities=2,
            device="cpu",
        )
        predictor = DualOutputPredictor(config.predictor_config)
        with torch.no_grad():
            predictor.macro_adapter.projection.weight.zero_()
            values = torch.linspace(-0.5, 0.5, config.hidden_dim)
            predictor.macro_adapter.projection.weight[:, 0].copy_(values)
            predictor.macro_adapter.projection.weight[:, 1].copy_(-values)
        latent = torch.randn(1, 32)
        action = torch.randn(1, 5)
        first = _window().context
        second = _frame("frame:other", 0, steady=True, unstable=False)
        from world_model.model import (
            BooleanTransitionValue,
            MacroTransitionBatch,
            MacroTransitionInput,
            TransitionRequest,
        )

        def request(frame):
            return TransitionRequest(
                PredictionPair(5, Abstraction.MACRO),
                MacroTransitionBatch((MacroTransitionInput(
                    frame_record_identity=frame.identity,
                    steady_state=BooleanTransitionValue(
                        "available", frame.labels["steady-state"]["value"]
                    ),
                    structure_unstable=BooleanTransitionValue(
                        "available", frame.labels["structure-unstable"]["value"]
                    ),
                ),)),
            )

        first_carrier = predictor.carrier(latent, action, request(first))
        second_carrier = predictor.carrier(latent, action, request(second))
        self.assertFalse(torch.allclose(first_carrier, second_carrier))

    def test_exhaustive_scoring_marks_non_endpoint_macro_pairs_unavailable(self) -> None:
        records = (
            _frame("frame:0", 0, steady=False, unstable=False),
            _frame("frame:1", 1, steady=False, unstable=False),
            _frame("frame:2", 2, steady=True, unstable=False, terminal_reason="level_fail"),
        )
        rollout = CohortV2Rollout(
            attempt_id="attempt:fixture",
            exposure_role="training",
            coverage_stratum="collision",
            scenario_lineage_identity="lineage:fixture",
            intervention={},
            agent_observation_identity="observation:fixture",
            agent_observation_fixed_step=0,
            frame_records=records,
        )

        class Reader:
            release_identity = "representative-cohort-v2-release-v5:fixture"
            partition_identity = "partition:fixture"
            rollouts = (rollout,)

            @staticmethod
            def load_observation(item, *, observation_role: str):
                return b"observation"

        class Scorer:
            checkpoint_identity = "checkpoint:fixture"
            objective_identity = "objective:fixture"
            capabilities = MACRO_CAPABILITIES

            @staticmethod
            def objective(window, pair):
                return 1.0

        def role_reader(role: str):
            role_rollout = replace(
                rollout,
                exposure_role=role,
                attempt_id=f"attempt:{role}",
                scenario_lineage_identity=f"lineage:{role}",
                frame_records=tuple(
                    replace(
                        frame,
                        identity=f"{frame.identity}:{role}",
                        state_id=f"{frame.state_id}:{role}",
                    )
                    for frame in rollout.frame_records
                ),
            )
            return SimpleNamespace(
                release_identity=Reader.release_identity,
                capability_declaration_identity=CAPABILITY_DECLARATION_IDENTITY,
                partition_identity=Reader.partition_identity,
                rollouts=(role_rollout,),
                load_observation=lambda item, *, observation_role: b"observation",
            )

        readers = tuple(
            role_reader(role)
            for role in ("training", "calibration", "model_selection")
        )
        result = CohortV2ExhaustiveEvaluator(Scorer()).evaluate(readers)
        first_state = result.states[0]
        h1_macro = next(
            outcome for outcome in first_state.outcomes
            if outcome.pair.identity == (1, "macro")
        )
        self.assertFalse(h1_macro.available)
        self.assertEqual(
            h1_macro.unavailable_reasons,
            ("event_endpoint_unavailable:no_terminal_within_requested_horizon",),
        )
        h5_macro = next(
            outcome for outcome in first_state.outcomes
            if outcome.pair.identity == (5, "macro")
        )
        self.assertTrue(h5_macro.available)
        self.assertEqual(h5_macro.effective_horizon, 2)

    def test_checkpoint_binds_full_grid_learned_weights(self) -> None:
        config = CohortV2MacroConfig(
            steps=9,
            batch_size=1,
            latent_dim=32,
            hidden_dim=16,
            depth=1,
            max_entities=2,
            device="cpu",
        )

        class Reader:
            release_identity = COHORT_V2_RELEASE_IDENTITY
            partition_identity = "partition:fixture"

        trainer = SimpleNamespace(
            data=SimpleNamespace(reader=Reader()),
            config=config,
            codec=CohortV2StateCodec(latent_dim=32, max_entities=2),
            predictor=DualOutputPredictor(config.predictor_config),
            step_count=9,
            pair_counts={pair: 1 for pair in MACRO_PAIRS},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            saved = save_cohort_v2_macro_checkpoint(path, trainer)
            loaded = load_cohort_v2_macro_checkpoint(
                path, reader=Reader(), config=config, device="cpu"
            )[2]
            self.assertEqual(loaded.identity, saved.identity)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            first = next(iter(payload["model_state"]))
            payload["model_state"][first].view(-1)[0] += 1.0
            torch.save(payload, path)
            with self.assertRaisesRegex(CohortV2MacroError, "provenance"):
                load_cohort_v2_macro_checkpoint(
                    path, reader=Reader(), config=config, device="cpu"
                )

    def test_frontier_contains_eligible_macro_pairs(self) -> None:
        grid = CohortV2PairGrid()
        outcomes = tuple(
            CohortV2PairOutcome(
                pair=pair,
                requested_horizon=pair.delta,
                effective_horizon=1,
                target_frame_record_identity="frame:endpoint",
                objective=float(pair.delta),
                unavailable_reasons=(),
            )
            for pair in grid.pairs
        )
        state = CohortV2StateEvaluation(
            state_id="frame:context",
            exposure_role="model_selection",
            attempt_id="attempt:model-selection",
            scenario_lineage_identity="lineage:model-selection",
            context_position=0,
            context_fixed_step=1,
            frame_record_count=2,
            outcomes=outcomes,
            selected_pair=PredictionPair(1, Abstraction.CONTINUOUS),
            tied_pairs=tuple(
                PredictionPair(1, abstraction) for abstraction in Abstraction
            ),
        )
        result = CohortV2EvaluationResult(
            release_identity="release:fixture",
            capability_declaration_identity=CAPABILITY_DECLARATION_IDENTITY,
            partition_identity="partition:fixture",
            checkpoint_identity="checkpoint:fixture",
            checkpoint_capabilities=tuple(sorted(MACRO_CAPABILITIES)),
            objective_identity="objective:fixture",
            grid=grid,
            state_set_identity="states:fixture",
            states=(state,),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frontier.json"
            write_cohort_v2_macro_frontier_input(path, result)
            validate_cohort_v2_macro_frontier_input(path, result)
            payload = json.loads(path.read_bytes())

        statuses = {
            (row["requested_horizon"], row["abstraction"]): row["status"]
            for row in payload["pairs"]
        }
        self.assertEqual(statuses[(1, "macro")], "available")
        self.assertIn("terminal-event-endpoint-v1", payload["macro_event_endpoint_authority"])


if __name__ == "__main__":
    unittest.main()
