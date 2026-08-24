from __future__ import annotations

import json
import tempfile
import unittest
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
    MicroReadoutHead,
    PredictionPair,
)
from world_model.training import (
    CohortV2EvaluationResult,
    CohortV2PairGrid,
    CohortV2PairOutcome,
    CohortV2StateEvaluation,
)
from world_model.training.cohort_v2_micro import (
    MICRO_CAPABILITIES,
    MICRO_PAIRS,
    CohortV2MicroCheckpoint,
    CohortV2MicroConfig,
    CohortV2MicroError,
    CohortV2MicroPairScorer,
    CohortV2StateCodec,
    load_cohort_v2_micro_checkpoint,
    micro_predicate_loss,
    micro_relation_loss,
    save_cohort_v2_micro_checkpoint,
    validate_cohort_v2_micro_frontier_artifacts,
    validate_cohort_v2_micro_frontier_input,
    write_cohort_v2_micro_frontier_input,
)
from world_model.training.grid_artifacts import canonical_json_bytes


def _relation(availability="available", relations=()):
    return {
        "availability": availability,
        "relations": relations if availability == "available" else None,
    }


def _frame(
    identity: str,
    *,
    contact=(),
    supports=(),
    support_availability="available",
    x=0.0,
):
    entities = tuple(
        {
            "body": {
                "angular_velocity_degrees_per_second": 0.0,
                "body_type": "dynamic",
                "gravity_applicable": True,
                "gravity_scale": 1.0,
                "position": (x + index, 0.0),
                "rotation_degrees": 0.0,
                "simulated": True,
                "velocity": (0.0, 0.0),
            },
            "body_present": True,
            "entity_id": entity_id,
            "lifecycle": "active",
            "scenario_object_id": f"object:{index}",
        }
        for index, entity_id in enumerate(("entity:a", "entity:b"))
    )
    return CohortV2CentralFrameRecord(
        identity=identity,
        capture_id="capture:fixture",
        state_id=identity,
        fixed_step=0,
        capture_stride=1,
        engine_state={
            "entities": entities,
            "world": {"gravity_vector": (0.0, -9.81)},
        },
        events=(),
        labels={
            "contact": _relation(relations=contact),
            "supports": _relation(support_availability, supports),
        },
        terminal=None,
    )


class CohortV2MicroTrainingTests(unittest.TestCase):
    def test_oracle_state_codec_is_continuous_fixed_width_and_state_sensitive(self) -> None:
        codec = CohortV2StateCodec(latent_dim=32, max_entities=2)

        first = codec.encode(_frame("frame:one", x=0.0))
        second = codec.encode(_frame("frame:two", x=2.0))

        self.assertEqual(first.shape, (32,))
        self.assertEqual(first.dtype, torch.float32)
        self.assertFalse(torch.equal(first, second))
        self.assertIn("oracle-continuous-carrier-v1", codec.identity)

    def test_exact_relation_loss_scores_symmetric_contact_and_directed_support_queries(self) -> None:
        torch.manual_seed(4)
        head = MicroReadoutHead(latent_dim=32, hidden_dim=16, predicate_count=2)
        carrier = torch.randn(1, 32, requires_grad=True)
        target = _frame(
            "frame:target",
            contact=(("entity:b", "entity:a"),),
            supports=(("entity:a", "entity:b"),),
        )

        result = micro_relation_loss(head, carrier, (target,))
        predicate = micro_predicate_loss(head, carrier, (target,))
        (result.loss + predicate.loss).backward()

        self.assertEqual(result.available_predicate_count, 2)
        self.assertEqual(result.relation_query_count, 3)  # 1 unordered + 2 directed
        self.assertTrue(torch.isfinite(result.loss))
        self.assertIsNotNone(head.supporter_projection.weight.grad)
        self.assertIsNotNone(head.supported_projection.weight.grad)
        self.assertIsNotNone(head.body[-1].weight.grad)
        self.assertGreater(float(head.body[-1].weight.grad.abs().sum()), 0.0)

    def test_unavailable_relation_is_masked_instead_of_becoming_an_empty_negative(self) -> None:
        head = MicroReadoutHead(latent_dim=32, hidden_dim=16, predicate_count=2)
        carrier = torch.randn(1, 32, requires_grad=True)
        target = _frame(
            "frame:masked",
            contact=(("entity:a", "entity:b"),),
            support_availability="unavailable_no_evidence",
        )

        result = micro_relation_loss(head, carrier, (target,))

        self.assertEqual(result.available_predicate_count, 1)
        self.assertEqual(result.relation_query_count, 1)

    def test_training_config_declares_balanced_continuous_and_micro_grid(self) -> None:
        config = CohortV2MicroConfig(
            steps=6,
            batch_size=1,
            latent_dim=32,
            hidden_dim=16,
            depth=1,
            max_entities=2,
            device="cpu",
        )

        self.assertEqual(MICRO_CAPABILITIES, frozenset({
            "transition.continuous", "transition.micro"
        }))
        self.assertIn("micro-training-config-v1", config.identity)

    def test_batched_objectives_match_scalar_objectives(self) -> None:
        torch.manual_seed(7)
        context = _frame(
            "frame:context",
            contact=(("entity:a", "entity:b"),),
            supports=(("entity:a", "entity:b"),),
            x=0.0,
        )
        target = _frame(
            "frame:target",
            contact=(("entity:a", "entity:b"),),
            supports=(("entity:b", "entity:a"),),
            x=1.0,
        )
        rollout = CohortV2Rollout(
            attempt_id="attempt:fixture",
            exposure_role="calibration",
            coverage_stratum="collision",
            scenario_lineage_identity="lineage:fixture",
            intervention={},
            agent_observation_identity="observation:fixture",
            agent_observation_fixed_step=0,
            frame_records=(context, target),
        )

        class Reader:
            rollouts = (rollout,)

        intervention = {
            "engine_relative_action": {
                "drag_delta_canvas_pixels": (12, 3),
                "hold_milliseconds": 1000,
                "tap_time_milliseconds": 0,
            }
        }
        windows = tuple(
            CohortV2OracleWindow(
                source_release_identity=COHORT_V2_RELEASE_IDENTITY,
                capability_declaration_identity=CAPABILITY_DECLARATION_IDENTITY,
                exposure_role="calibration",
                attempt_id="attempt:fixture",
                scenario_lineage_identity="lineage:fixture",
                intervention=intervention,
                context_position=0,
                requested_horizon=1,
                effective_horizon=1,
                context=context,
                target=target,
                agent_observation=b"observation",
            )
            for _ in range(3)
        )
        config = CohortV2MicroConfig(
            steps=1,
            batch_size=1,
            latent_dim=32,
            hidden_dim=16,
            depth=1,
            max_entities=2,
            device="cpu",
        )
        predictor = DualOutputPredictor(config.predictor_config).eval()
        scorer = CohortV2MicroPairScorer(
            predictor,
            CohortV2StateCodec(latent_dim=32, max_entities=2),
            CohortV2MicroCheckpoint(Path("fixture.pt"), "checkpoint:fixture", 1, ()),
            config,
            (Reader(),),
        )
        for abstraction in (Abstraction.CONTINUOUS, Abstraction.MICRO):
            pair = PredictionPair(1, abstraction)
            scalar = tuple(scorer.objective(window, pair) for window in windows)
            batched = scorer.objective_batch(windows, pair)
            with self.subTest(abstraction=abstraction):
                self.assertEqual(len(batched), len(scalar))
                for left, right in zip(batched, scalar, strict=True):
                    self.assertAlmostEqual(left, right, places=6)

    def test_checkpoint_identity_rejects_changed_learned_weights(self) -> None:
        config = CohortV2MicroConfig(
            steps=6,
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

        predictor = DualOutputPredictor(config.predictor_config)
        trainer = SimpleNamespace(
            data=SimpleNamespace(reader=Reader()),
            config=config,
            codec=CohortV2StateCodec(latent_dim=32, max_entities=2),
            predictor=predictor,
            step_count=6,
            pair_counts={pair: 1 for pair in MICRO_PAIRS},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            saved = save_cohort_v2_micro_checkpoint(path, trainer)
            loaded = load_cohort_v2_micro_checkpoint(
                path, reader=Reader(), config=config, device="cpu"
            )[2]
            self.assertEqual(loaded.identity, saved.identity)

            payload = torch.load(path, map_location="cpu", weights_only=True)
            first = next(iter(payload["model_state"]))
            payload["model_state"][first].view(-1)[0] += 1.0
            torch.save(payload, path)
            with self.assertRaisesRegex(
                CohortV2MicroError, "provenance|identity"
            ):
                load_cohort_v2_micro_checkpoint(
                    path, reader=Reader(), config=config, device="cpu"
                )

    def test_frontier_input_contains_available_micro_and_explicitly_unavailable_macro(self) -> None:
        grid = CohortV2PairGrid()
        outcomes = tuple(
            CohortV2PairOutcome(
                pair=pair,
                requested_horizon=pair.delta,
                effective_horizon=1,
                target_frame_record_identity="frame:target",
                objective=(
                    float(pair.delta)
                    if pair.abstraction is not Abstraction.MACRO
                    else None
                ),
                unavailable_reasons=(
                    ()
                    if pair.abstraction is not Abstraction.MACRO
                    else ("checkpoint_capability_unavailable:transition.macro",)
                ),
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
            tied_pairs=(
                PredictionPair(1, Abstraction.CONTINUOUS),
                PredictionPair(1, Abstraction.MICRO),
            ),
        )
        result = CohortV2EvaluationResult(
            release_identity="release:fixture",
            capability_declaration_identity="cohort-v2-capabilities-v1",
            partition_identity="partition:fixture",
            checkpoint_identity="checkpoint:fixture",
            checkpoint_capabilities=tuple(sorted(MICRO_CAPABILITIES)),
            objective_identity="objective:fixture",
            grid=grid,
            state_set_identity="states:fixture",
            states=(state,),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frontier.json"
            write_cohort_v2_micro_frontier_input(path, result)
            validate_cohort_v2_micro_frontier_input(path, result)
            payload = json.loads(path.read_bytes())
            evaluation_root = Path(directory) / "evaluation"
            evaluation_root.mkdir()
            manifest = {
                "checkpoint_identity": result.checkpoint_identity,
                "evaluation_identity": result.identity,
                "grid_identity": result.grid.identity,
                "objective_identity": result.objective_identity,
                "pairs": [
                    {
                        "abstraction": str(pair.abstraction),
                        "requested_horizon": pair.delta,
                    }
                    for pair in result.grid.pairs
                ],
                "records": "state_evaluations.jsonl",
                "release_identity": result.release_identity,
                "state_set_identity": result.state_set_identity,
            }
            record = {
                "exposure_role": "model_selection",
                "outcomes": [
                    {
                        "objective": outcome.objective,
                        "status": "available" if outcome.available else "unavailable",
                        "unavailable_reasons": list(outcome.unavailable_reasons),
                    }
                    for outcome in outcomes
                ],
            }
            (evaluation_root / "manifest.json").write_bytes(
                canonical_json_bytes(manifest)
            )
            (evaluation_root / "state_evaluations.jsonl").write_bytes(
                canonical_json_bytes(record)
            )
            validate_cohort_v2_micro_frontier_artifacts(path, evaluation_root)

            payload["pairs"][1]["mean_objective"] += 1.0
            path.write_bytes(canonical_json_bytes(payload))
            with self.assertRaisesRegex(
                CohortV2MicroError, "persisted evaluation"
            ):
                validate_cohort_v2_micro_frontier_artifacts(path, evaluation_root)

        statuses = {
            (row["requested_horizon"], row["abstraction"]): row["status"]
            for row in payload["pairs"]
        }
        self.assertEqual(statuses[(1, "micro")], "available")
        self.assertEqual(statuses[(1, "macro")], "unavailable")


if __name__ == "__main__":
    unittest.main()
