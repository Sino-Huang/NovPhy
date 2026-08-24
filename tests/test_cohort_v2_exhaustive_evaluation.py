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
    CohortV2EvaluationError,
    CohortV2ExhaustiveEvaluator,
    CohortV2PairGrid,
    validate_cohort_v2_evaluation,
    write_cohort_v2_evaluation,
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
):
    labels = {
        "contact": _label(contact_available),
        "supports": _label(),
        "steady-state": _label(),
        "structure-unstable": _label(),
        "excess_penetration": _label(),
        "unsupported_stationary_or_floating_body": (),
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

    def test_artifacts_are_byte_deterministic_and_reject_missing_unavailability_reason(self) -> None:
        readers = _readers()
        scorer = _UnavailableScorer()
        result = CohortV2ExhaustiveEvaluator(scorer).evaluate(readers)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            receipt = write_cohort_v2_evaluation(first, result, readers=readers)
            write_cohort_v2_evaluation(second, result, readers=readers)

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


if __name__ == "__main__":
    unittest.main()
