from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from world_model.data import CohortV2CentralFrameRecord, CohortV2Rollout
from world_model.model import Abstraction
from world_model.training import (
    COHORT_V2_HORIZONS,
    CohortV2ComputeCalibration,
    CohortV2EvaluationError,
    CohortV2ExecutionProfile,
    CohortV2ExhaustiveEvaluator,
    CohortV2MeasurementError,
    CohortV2ParallelExhaustiveEvaluator,
    CohortV2PairGrid,
    load_cohort_v2_evaluation,
    measure_cohort_v2_evaluation,
    validate_cohort_v2_evaluation,
    validate_cohort_v2_measurements,
    write_cohort_v2_evaluation,
    write_cohort_v2_measurements,
)
from world_model.training.grid_artifacts import canonical_json_bytes
from world_model.training.pair_grid import PairGridConfig


def _label(available: bool = True):
    return {
        "availability": "available" if available else "unavailable_no_evidence",
        "value": False if available else None,
        "relations": () if available else None,
    }


def _reader(
    role: str,
    *,
    partition_identity: str = "partition:fixture",
    identity_suffix: str = "",
    contact_available: bool = True,
    contact_relations=None,
    support_relations=(),
    excess_penetration=False,
    unsupported_bodies=(),
):
    contact = _label(contact_available)
    if contact_relations is not None:
        contact = {**contact, "relations": contact_relations}
    labels = {
        "contact": contact,
        "supports": {**_label(), "relations": support_relations},
        "steady-state": _label(),
        "structure-unstable": _label(),
        "excess_penetration": {**_label(), "value": excess_penetration},
        "unsupported_stationary_or_floating_body": unsupported_bodies,
    }
    records = tuple(
        CohortV2CentralFrameRecord(
            identity=f"frame:{role}:{position}{identity_suffix}",
            capture_id=f"capture:{role}",
            state_id=f"state:{role}:{position}",
            fixed_step=position,
            capture_stride=1,
            engine_state={"fixed_step": position},
            events=(),
            labels=labels,
            terminal={"reason": "stable_entered"} if position == 3 else None,
        )
        for position in range(4)
    )
    rollout = CohortV2Rollout(
        attempt_id=f"attempt:{role}",
        exposure_role=role,
        coverage_stratum="collision",
        scenario_lineage_identity=f"lineage:{role}",
        intervention={},
        agent_observation_identity=f"observation:{role}",
        agent_observation_fixed_step=0,
        frame_records=records,
    )

    class Reader:
        release_identity = "representative-cohort-v2-release-v5:fixture"
        capability_declaration_identity = "cohort-v2-capabilities-v1"
        rollouts = (rollout,)

        @staticmethod
        def load_observation(item, *, observation_role: str) -> bytes:
            del item
            if observation_role != "agent":
                raise AssertionError
            return b"observation"

    reader = Reader()
    reader.partition_identity = partition_identity
    return reader


def _readers(**kwargs):
    return tuple(
        _reader(role, **kwargs)
        for role in ("training", "calibration", "model_selection")
    )


def _compute_calibration():
    return CohortV2ComputeCalibration(
        authority="fixture:declared-macs",
        unit="multiply_accumulate",
        controller_per_decision=2.0,
        continuous_adapter_per_decision=3.0,
        micro_adapter_per_decision=5.0,
        macro_adapter_per_decision=7.0,
        micro_graph_base_per_decision=11.0,
        micro_graph_per_entity=13.0,
        micro_graph_per_contact=17.0,
        micro_graph_per_support=19.0,
        transition_per_decision=23.0,
        continuous_readout_per_decision=0.0,
        micro_readout_per_decision=29.0,
        macro_readout_per_decision=31.0,
        shared_initial_perception_per_rollout=37.0,
    )


@dataclass(frozen=True)
class _Scorer:
    checkpoint_identity: str = "checkpoint:fixture"
    objective_identity: str = "objective:fixture"
    capabilities: frozenset[str] = frozenset({
        "transition.continuous", "transition.micro", "transition.macro"
    })
    offset: float = 0.0

    def objective(self, window, pair) -> float:
        del window
        return float(pair.delta) + self.offset


@dataclass
class _UnavailableScorer:
    checkpoint_identity: str = "checkpoint:unavailable"
    objective_identity: str = "objective:unavailable"
    capabilities: frozenset[str] = frozenset()
    calls: int = 0

    def objective(self, window, pair) -> float:
        del window, pair
        self.calls += 1
        raise AssertionError("unsupported pairs must not be scored")


@dataclass(frozen=True)
class _NearTieScorer:
    checkpoint_identity: str = "checkpoint:near-tie"
    objective_identity: str = "objective:near-tie"
    capabilities: frozenset[str] = frozenset({
        "transition.continuous", "transition.micro", "transition.macro"
    })

    def objective(self, window, pair) -> float:
        del window
        values = {
            (1, "continuous"): 1.0000006,
            (1, "micro"): 1.0000005,
            (1, "macro"): 1.0000004,
            (5, "continuous"): 1.0,
        }
        return values.get(pair.identity, 2.0)


class _BatchedScorer:
    checkpoint_identity = "checkpoint:fixture"
    objective_identity = "objective:fixture"
    capabilities = frozenset({
        "transition.continuous", "transition.micro", "transition.macro"
    })

    def __init__(self, worker: int) -> None:
        self.worker = worker
        self.batch_sizes = []
        self.state_ids = set()

    def objective(self, window, pair) -> float:
        return self.objective_batch((window,), pair)[0]

    def objective_batch(self, windows, pair):
        self.batch_sizes.append(len(windows))
        self.state_ids.update(window.context.identity for window in windows)
        return tuple(float(pair.delta) for _window in windows)


def _validate(path: Path, readers, scorer):
    return validate_cohort_v2_evaluation(
        path,
        readers=readers,
        checkpoint_identity=scorer.checkpoint_identity,
        checkpoint_capabilities=scorer.capabilities,
        objective_identity=scorer.objective_identity,
    )


def _rewrite_records_identity(root: Path) -> None:
    records = (root / "state_evaluations.jsonl").read_bytes()
    manifest = json.loads((root / "manifest.json").read_bytes())
    manifest["records_identity"] = f"sha256:{hashlib.sha256(records).hexdigest()}"
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))


class CohortV2ExhaustiveEvaluationTests(unittest.TestCase):
    def test_grid_is_explicit_and_does_not_change_the_legacy_temporal_grid(self) -> None:
        grid = CohortV2PairGrid()

        self.assertEqual(grid.horizons, (1, 5, 15))
        self.assertEqual(len(grid.pairs), 9)
        self.assertEqual(
            tuple(pair.abstraction for pair in grid.pairs[:3]),
            (Abstraction.CONTINUOUS, Abstraction.MICRO, Abstraction.MACRO),
        )
        self.assertIn("cohort-v2-pair-grid-v1", grid.identity)
        self.assertEqual(len(PairGridConfig().pairs), 3)

    def test_every_nonterminal_state_has_every_pair_and_terminal_clamps_are_distinct(self) -> None:
        first = CohortV2ExhaustiveEvaluator(_Scorer()).evaluate(_readers())
        second = CohortV2ExhaustiveEvaluator(_Scorer()).evaluate(_readers())

        self.assertEqual(first, second)
        self.assertEqual(len(first.states), 9)
        self.assertEqual(first.outcome_count, 9 * 9)
        self.assertEqual(first.available_count, first.outcome_count)
        for state in first.states:
            self.assertEqual(tuple(item.pair for item in state.outcomes), first.grid.pairs)
            self.assertEqual(state.selected_pair.identity, (1, "continuous"))
            self.assertEqual(
                tuple(pair.identity for pair in state.tied_pairs),
                ((1, "continuous"), (1, "micro"), (1, "macro")),
            )
        terminal_edge = next(
            state for state in first.states if state.context_position == 2
        )
        long = next(
            outcome for outcome in terminal_edge.outcomes
            if outcome.pair.identity == (15, "continuous")
        )
        self.assertEqual(long.requested_horizon, 15)
        self.assertEqual(long.effective_horizon, 1)

    def test_checkpoint_capability_reasons_replace_fabricated_scores(self) -> None:
        scorer = _UnavailableScorer()
        result = CohortV2ExhaustiveEvaluator(scorer).evaluate(_readers())

        self.assertEqual(scorer.calls, 0)
        self.assertEqual(result.available_count, 0)
        self.assertEqual(result.unavailable_count, result.outcome_count)
        self.assertTrue(all(
            outcome.objective is None and outcome.unavailable_reasons
            for state in result.states for outcome in state.outcomes
        ))
        self.assertTrue(all(
            state.selected_pair is None and state.tied_pairs == ()
            for state in result.states
        ))

    def test_parallel_batches_shard_states_and_merge_in_canonical_order(self) -> None:
        readers = _readers()
        expected = CohortV2ExhaustiveEvaluator(_Scorer()).evaluate(readers)
        scorers = tuple(_BatchedScorer(index) for index in range(4))

        first = CohortV2ParallelExhaustiveEvaluator(
            scorers, batch_size=2
        ).evaluate(readers)
        second = CohortV2ParallelExhaustiveEvaluator(
            tuple(_BatchedScorer(index) for index in range(4)), batch_size=2
        ).evaluate(readers)

        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertEqual(
            tuple(state.state_id for state in first.states),
            tuple(state.state_id for state in expected.states),
        )
        self.assertTrue(all(scorer.state_ids for scorer in scorers))
        self.assertEqual(
            set.union(*(scorer.state_ids for scorer in scorers)),
            {state.state_id for state in expected.states},
        )
        self.assertEqual(
            sum(len(scorer.state_ids) for scorer in scorers),
            len(expected.states),
        )
        self.assertTrue(all(
            batch_size <= 2
            for scorer in scorers for batch_size in scorer.batch_sizes
        ))

    def test_artifacts_are_byte_deterministic_and_reject_missing_unavailability_reason(self) -> None:
        readers = _readers()
        scorer = _UnavailableScorer()
        result = CohortV2ExhaustiveEvaluator(scorer).evaluate(readers)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            receipt = write_cohort_v2_evaluation(first, result, readers=readers)
            write_cohort_v2_evaluation(second, result, readers=readers)
            loaded = load_cohort_v2_evaluation(
                first,
                readers=readers,
                checkpoint_identity=result.checkpoint_identity,
                checkpoint_capabilities=frozenset(result.checkpoint_capabilities),
                objective_identity=result.objective_identity,
            )

            self.assertEqual(loaded, result)
            self.assertEqual(
                (first / "manifest.json").read_bytes(),
                (second / "manifest.json").read_bytes(),
            )
            self.assertEqual(
                (first / "state_evaluations.jsonl").read_bytes(),
                (second / "state_evaluations.jsonl").read_bytes(),
            )
            self.assertEqual(_validate(first, readers, scorer), receipt)

            records = (first / "state_evaluations.jsonl").read_bytes().splitlines()
            record = json.loads(records[0])
            record["outcomes"][0]["unavailable_reasons"] = []
            records[0] = canonical_json_bytes(record).rstrip(b"\n")
            (first / "state_evaluations.jsonl").write_bytes(
                b"\n".join(records) + b"\n"
            )
            with self.assertRaisesRegex(
                CohortV2EvaluationError, "canonical-record identity"
            ):
                _validate(first, readers, scorer)

    def test_unavailable_reasons_are_recomputed_from_source_capabilities(self) -> None:
        readers = _readers()
        scorer = _UnavailableScorer()
        result = CohortV2ExhaustiveEvaluator(scorer).evaluate(readers)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_cohort_v2_evaluation(root, result, readers=readers)
            lines = (root / "state_evaluations.jsonl").read_bytes().splitlines()

            for reason in (
                "arbitrary_unavailable_reason",
                "checkpoint_capability_unavailable:transition.micro",
            ):
                record = json.loads(lines[0])
                record["outcomes"][0]["unavailable_reasons"] = [reason]
                changed = list(lines)
                changed[0] = canonical_json_bytes(record).rstrip(b"\n")
                (root / "state_evaluations.jsonl").write_bytes(
                    b"\n".join(changed) + b"\n"
                )
                _rewrite_records_identity(root)
                with self.assertRaisesRegex(
                    CohortV2EvaluationError, "reasons differ from source"
                ):
                    _validate(root, readers, scorer)

    def test_records_identity_binds_membership_outcomes_and_objectives(self) -> None:
        baseline = CohortV2ExhaustiveEvaluator(_Scorer()).evaluate(_readers())
        changed_membership = CohortV2ExhaustiveEvaluator(_Scorer()).evaluate(
            _readers(identity_suffix=":changed")
        )
        changed_outcomes = CohortV2ExhaustiveEvaluator(_Scorer()).evaluate(
            _readers(contact_available=False)
        )
        changed_objectives = CohortV2ExhaustiveEvaluator(
            _Scorer(offset=0.25)
        ).evaluate(_readers())

        self.assertNotEqual(baseline.records_identity, changed_membership.records_identity)
        self.assertNotEqual(baseline.records_identity, changed_outcomes.records_identity)
        self.assertNotEqual(baseline.records_identity, changed_objectives.records_identity)

    def test_partition_and_checkpoint_capability_mismatches_fail_closed(self) -> None:
        readers = _readers()
        scorer = _UnavailableScorer()
        result = CohortV2ExhaustiveEvaluator(scorer).evaluate(readers)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_cohort_v2_evaluation(root, result, readers=readers)
            with self.assertRaisesRegex(CohortV2EvaluationError, "provenance"):
                _validate(
                    root,
                    _readers(partition_identity="partition:different"),
                    scorer,
                )
            with self.assertRaisesRegex(CohortV2EvaluationError, "provenance"):
                validate_cohort_v2_evaluation(
                    root,
                    readers=readers,
                    checkpoint_identity=scorer.checkpoint_identity,
                    checkpoint_capabilities=frozenset({"transition.continuous"}),
                    objective_identity=scorer.objective_identity,
                )

    def test_tolerance_ties_follow_horizon_then_mode_order(self) -> None:
        result = CohortV2ExhaustiveEvaluator(_NearTieScorer()).evaluate(_readers())

        self.assertTrue(all(
            state.selected_pair.identity == (1, "continuous")
            for state in result.states
        ))
        self.assertEqual(
            tuple(pair.identity for pair in result.states[0].tied_pairs),
            (
                (1, "continuous"),
                (1, "micro"),
                (1, "macro"),
                (5, "continuous"),
            ),
        )

    def test_grid_provenance_binds_exact_endpoint_capabilities(self) -> None:
        readers = _readers()
        scorer = _UnavailableScorer()
        result = CohortV2ExhaustiveEvaluator(scorer).evaluate(readers)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_cohort_v2_evaluation(root, result, readers=readers)
            manifest = json.loads((root / "manifest.json").read_bytes())

        self.assertEqual(
            manifest["grid_capabilities"]["endpoint_labels"],
            [
                "excess_penetration",
                "unsupported_stationary_or_floating_body",
            ],
        )

    def test_horizon_contract_is_not_sourced_from_a_mutable_argument(self) -> None:
        self.assertEqual(COHORT_V2_HORIZONS, (1, 5, 15))
        with self.assertRaises(TypeError):
            CohortV2PairGrid(horizons=(1, 2, 4))

    def test_pair_measurements_compare_endpoint_plausibility_and_complete_compute(self) -> None:
        readers = _readers(
            contact_relations=(("a", "b"),),
            support_relations=(("b", "c"),),
            excess_penetration=True,
            unsupported_bodies=(
                {"entity_id": "a", "availability": "available", "value": True},
                {
                    "entity_id": "b",
                    "availability": "unavailable_incomplete_stability_window",
                    "value": None,
                },
            ),
        )
        evaluation = CohortV2ExhaustiveEvaluator(_Scorer()).evaluate(readers)
        calibration = _compute_calibration()
        measured = measure_cohort_v2_evaluation(
            evaluation,
            readers,
            calibration,
            CohortV2ExecutionProfile(
                controller_executed=True,
                shared_perception_executed=True,
            ),
        )

        state = measured.states[0]
        by_pair = {item.pair.identity: item for item in state.outcomes}
        continuous = by_pair[(5, "continuous")]
        micro = by_pair[(5, "micro")]
        macro = by_pair[(5, "macro")]

        for outcome in (continuous, micro, macro):
            self.assertEqual(outcome.target_frame_record_identity, "frame:training:3")
            self.assertEqual(outcome.effective_horizon, 3)
            self.assertEqual(outcome.endpoint_plausibility.available_value_count, 2)
            self.assertEqual(outcome.endpoint_plausibility.unavailable_value_count, 1)
            self.assertEqual(outcome.endpoint_plausibility.violation_count, 2)
            self.assertEqual(outcome.endpoint_plausibility.violation_rate, 1.0)
            self.assertEqual(outcome.compute.simulated_frame_count, 3)
            self.assertEqual(outcome.compute.infilling, 0.0)
            self.assertFalse(hasattr(outcome, "dense_path_plausibility"))

        self.assertEqual(continuous.compute.policy_dependent_total, 28.0)
        self.assertEqual(continuous.compute.full_end_to_end_total, 65.0)
        self.assertEqual(continuous.compute.policy_dependent_per_simulated_frame, 28.0 / 3.0)
        self.assertEqual(micro.compute.graph_work, 86.0)
        self.assertEqual(micro.compute.policy_dependent_total, 145.0)
        self.assertEqual(micro.compute.full_end_to_end_total, 182.0)
        self.assertEqual(macro.compute.graph_work, 0.0)
        self.assertEqual(macro.compute.policy_dependent_total, 63.0)
        self.assertEqual(macro.compute.full_end_to_end_total, 100.0)

        later = next(
            state for state in measured.states
            if state.state_id == "frame:training:1"
        )
        later_continuous = next(
            outcome for outcome in later.outcomes
            if outcome.pair.identity == (1, "continuous")
        )
        self.assertEqual(later_continuous.compute.shared_perception, 0.0)
        self.assertEqual(later_continuous.compute.full_end_to_end_total, 28.0)

    def test_pair_measurement_artifacts_are_source_bound_and_have_no_dense_path_claim(self) -> None:
        readers = _readers()
        evaluation = CohortV2ExhaustiveEvaluator(_Scorer()).evaluate(readers)
        profile = CohortV2ExecutionProfile(
            controller_executed=False,
            shared_perception_executed=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_cohort_v2_measurements(
                root,
                evaluation,
                readers=readers,
                calibration=_compute_calibration(),
                profile=profile,
            )
            first_bytes = tuple(
                (root / name).read_bytes()
                for name in ("manifest.json", "pair_measurements.jsonl")
            )
            second = write_cohort_v2_measurements(
                root,
                evaluation,
                readers=readers,
                calibration=_compute_calibration(),
                profile=profile,
            )
            second_bytes = tuple(
                (root / name).read_bytes()
                for name in ("manifest.json", "pair_measurements.jsonl")
            )

            self.assertEqual(first, second)
            self.assertEqual(first_bytes, second_bytes)
            self.assertNotIn(b"dense_path", b"".join(second_bytes))
            self.assertIn(b'"target_frame_record_identity"', second_bytes[1])

            records = (root / "pair_measurements.jsonl").read_bytes()
            (root / "pair_measurements.jsonl").write_bytes(
                records.replace(b'"policy_dependent_total":26.0', b'"policy_dependent_total":27.0', 1)
            )
            with self.assertRaises(CohortV2MeasurementError):
                validate_cohort_v2_measurements(
                    root,
                    evaluation,
                    readers=readers,
                    calibration=_compute_calibration(),
                    profile=profile,
                )


if __name__ == "__main__":
    unittest.main()
