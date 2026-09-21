"""Focused tests for the issue-82 zero-floor oracle diagnostic logic.

Covers the two-stage Phase-0 binding (including the blocked-binding terminal
path this experiment lands on), the predicate-free best-candidate row, the
paired bootstrap determinism, the pre-declared disposition mapping, the
prior fallback protocol rule, the cell inventory, the typed prior choice,
prediction-row invariants, and both publication renderings (blocked and
bound). No GPU work and no engine execution is performed.
"""
import unittest
from pathlib import Path

import numpy as np

from world_model.training import reactive_zero_floor_oracle as oracle
import scripts.run_reactive_zero_floor_oracle as runner


def synthetic_rows(costs, removed=()):
    rows = []
    for index, cost in enumerate(costs):
        rows.append({
            "member_identity": "member-001",
            "state_identities": ["state-001"],
            "ordinal": index,
            "branch_identity": f"member-001-a{index:02d}",
            "realized_count_cost": float(cost),
            "censored": True,
            "pig_removed": index in removed,
            "pig_contact": False,
            "segment_failure": "native_time_window_limit",
        })
    return rows


class BindingTests(unittest.TestCase):
    def test_clean_separation_binds_at_max_removed_cost(self) -> None:
        rows = synthetic_rows((10.0, 20.0, 90.0, 95.0), removed=(0, 1))
        binding = oracle.bind_threshold(rows)
        self.assertEqual(binding["status"], "bound")
        self.assertEqual(binding["threshold"], 20.0)
        self.assertEqual(binding["pig_removed_cells"], 2)
        self.assertTrue(binding["stage2"]["passed"])
        self.assertEqual(binding["stage2"]["observed_agreement"], 1.0)
        self.assertEqual(binding["threshold_setter_branches"], ["member-001-a01"])

    def test_interior_costs_block_the_binding(self) -> None:
        # The pattern this experiment found: a handful of removals with costs
        # interior to a dense non-removed distribution.
        rows = synthetic_rows((30.0, 50.0, 70.0, *[45.0] * 30), removed=(0, 1))
        binding = oracle.bind_threshold(rows)
        self.assertEqual(binding["status"], "no_defensible_binding")
        self.assertEqual(binding["stage1"]["candidate_threshold"], 50.0)
        self.assertLess(binding["stage2"]["observed_agreement"],
                        oracle.AGREEMENT_FLOOR)
        self.assertFalse(binding["stage2"]["passed"])
        self.assertIn("blocker", binding)
        self.assertIsNone(binding["threshold"])
        self.assertRaises(ValueError, oracle.predicate_verdict, 10.0, binding)

    def test_no_removal_blocks_the_binding(self) -> None:
        binding = oracle.bind_threshold(synthetic_rows((10.0, 20.0, 30.0)))
        self.assertEqual(binding["status"], "no_defensible_binding")
        self.assertIsNone(binding["stage1"]["candidate_threshold"])
        self.assertIn("no monotone numeric cost threshold", binding["blocker"])

    def test_blocker_quantifies_false_successes(self) -> None:
        rows = synthetic_rows((50.0, 60.0, *[40.0] * 10), removed=(0,))
        binding = oracle.bind_threshold(rows)
        self.assertEqual(binding["stage2"]["false_success_cells"], 10)
        self.assertEqual(binding["stage2"]["missed_removal_cells"], 0)
        self.assertAlmostEqual(binding["stage2"]["always_failure_agreement"],
                               11 / 12)


class PredicateAndRowTests(unittest.TestCase):
    def test_predicate_verdict_uses_inclusive_threshold(self) -> None:
        binding = {"status": "bound", "threshold": 50.0}
        self.assertTrue(oracle.predicate_verdict(50.0, binding))
        self.assertFalse(oracle.predicate_verdict(50.000001, binding))
        self.assertRaises(ValueError, oracle.predicate_verdict,
                          float("nan"), binding)

    def test_oracle_state_row_argmin_and_indicator(self) -> None:
        binding = {"status": "bound", "threshold": 50.0}
        rows = [
            {"ordinal": 0, "branch_identity": "b0", "realized_count_cost": 60.0},
            {"ordinal": 1, "branch_identity": "b1", "realized_count_cost": 40.0},
            {"ordinal": 2, "branch_identity": "b2", "realized_count_cost": 70.0},
        ]
        summary = oracle.oracle_state_row(rows, binding)
        self.assertEqual(summary["best_candidate"]["ordinal"], 1)
        self.assertEqual(summary["oracle_indicator"], 1)
        self.assertEqual(summary["success_ordinals"], [1])
        tied_rows = [
            {"ordinal": 0, "branch_identity": "b0", "realized_count_cost": 40.0},
            {"ordinal": 1, "branch_identity": "b1", "realized_count_cost": 40.0},
            {"ordinal": 2, "branch_identity": "b2", "realized_count_cost": 70.0},
        ]
        tied = oracle.oracle_state_row(tied_rows, binding)
        self.assertEqual(tied["best_candidate"]["ordinal"], 0)
        self.assertEqual(tied["best_candidate_tied_ordinals"], [0, 1])

    def test_best_candidate_row_is_predicate_free(self) -> None:
        rows = [{"ordinal": 0, "branch_identity": "b0", "realized_count_cost": 3.0},
                {"ordinal": 1, "branch_identity": "b1", "realized_count_cost": 1.0}]
        best = oracle.best_candidate_row(rows)
        self.assertEqual(best["ordinal"], 1)
        self.assertEqual(best["realized_count_cost"], 1.0)


class DivergenceAndUncertaintyTests(unittest.TestCase):
    def test_replay_divergence_counts(self) -> None:
        rows = synthetic_rows((10.0, 90.0, 20.0), removed=(0,))
        binding = {"status": "bound", "threshold": 25.0}
        table = oracle.replay_divergence_table(rows, binding)
        self.assertEqual(table["cells"], 3)
        self.assertEqual(table["predicate_success_and_pig_removed"], 1)
        self.assertEqual(table["predicate_success_and_no_pig_removed"], 1)
        self.assertEqual(table["predicate_failure_and_pig_removed"], 0)
        self.assertEqual(table["predicate_failure_and_no_pig_removed"], 1)
        self.assertAlmostEqual(table["agreement_fraction"], 2 / 3)

    def test_pilot_divergence_typed_cells_are_reported_not_counted(self) -> None:
        cells = [
            {"failure": None, "success": True, "chosen_realized_count_cost": 10.0},
            {"failure": None, "success": False, "chosen_realized_count_cost": 10.0},
            {"failure": "decision_failure: prior_candidate_absent",
             "success": None, "chosen_realized_count_cost": None},
        ]
        binding = {"status": "bound", "threshold": 15.0}
        table = oracle.pilot_divergence_table(cells, binding)
        self.assertEqual(table["valid_cells"], 2)
        self.assertEqual(table["typed_failure_cells"], 1)
        self.assertEqual(table["predicate_success_and_gameplay_success"], 1)
        self.assertEqual(table["predicate_success_and_gameplay_failure"], 1)

    def test_indicator_interval_matches_hand_rolled_draw(self) -> None:
        values = [0.0, 1.0, 1.0, 0.0, 1.0]
        interval = oracle.indicator_interval(values, draws=10000, seed=7201)
        rng = np.random.default_rng(7201)
        draws = np.asarray(values)[rng.integers(len(values), size=(10000, len(values)))].mean(1)
        self.assertEqual(interval["mean"], 0.6)
        self.assertEqual(interval["descriptive_95_percent_interval"],
                         np.quantile(draws, [.025, .975]).tolist())
        self.assertEqual(interval["resampled_units"], 5)

    def test_threshold_sensitivity_counts(self) -> None:
        rows = synthetic_rows((10.0, 40.0, 90.0))
        table = oracle.threshold_sensitivity(rows, grid=(20.0, 100.0))
        self.assertEqual(table, [{"threshold": 20.0, "cells_meeting_predicate": 1},
                                 {"threshold": 100.0, "cells_meeting_predicate": 3}])


class DispositionTests(unittest.TestCase):
    def test_blocked_binding_yields_readiness_tokens(self) -> None:
        binding = {"status": "no_defensible_binding", "blocker": "b"}
        result = oracle.decide_dispositions(binding, None,
                                            {name: 0 for name in oracle.SYSTEMS})
        self.assertEqual(result["q1_oracle_ceiling_nonzero"],
                         "readiness_or_precision_insufficient")
        self.assertEqual(result["q2_disposition"], "readiness_or_precision_insufficient")
        self.assertIsNone(result["q1_reading"])
        self.assertEqual(result["blocker"], "b")

    def test_zero_prevalence_reads_state_difficulty(self) -> None:
        binding = {"status": "bound", "threshold": 1.0}
        result = oracle.decide_dispositions(binding, 0.0,
                                            {name: 0 for name in oracle.SYSTEMS})
        self.assertEqual(result["q1_oracle_ceiling_nonzero"], "not_supported_by_this_experiment")
        self.assertEqual(result["q1_reading"], "state_difficulty_floor")
        self.assertEqual(result["q2_recommendation"],
                         "prevalence_floor_state_selection")

    def test_positive_prevalence_with_all_systems_zero_reads_ranking_failure(self) -> None:
        binding = {"status": "bound", "threshold": 1.0}
        result = oracle.decide_dispositions(binding, 0.5,
                                            {name: 0 for name in oracle.SYSTEMS})
        self.assertEqual(result["q1_oracle_ceiling_nonzero"], "supported")
        self.assertEqual(result["q1_reading"], "ranking_failure")
        self.assertEqual(result["q2_recommendation"], "h1_ranking_calibration_probe")

    def test_positive_prevalence_with_nonzero_system_is_not_typed(self) -> None:
        binding = {"status": "bound", "threshold": 1.0}
        successes = {name: 1 for name in oracle.SYSTEMS}
        result = oracle.decide_dispositions(binding, 0.5, successes)
        self.assertEqual(result["q1_oracle_ceiling_nonzero"], "supported")
        self.assertIsNone(result["q1_reading"])
        self.assertIsNone(result["q2_recommendation"])
        self.assertEqual(result["q2_disposition"], "not_supported_by_this_experiment")


class PriorFallbackTests(unittest.TestCase):
    def test_fallback_picks_nearest_ordinal_with_lower_tie(self) -> None:
        self.assertEqual(oracle.prior_fallback_ordinal((0, 1, 2, 3, 4, 5)), 5)
        self.assertEqual(oracle.prior_fallback_ordinal((0, 6, 7, 10)), 7)
        self.assertEqual(oracle.prior_fallback_ordinal((0, 7, 9, 10)), 7)
        self.assertEqual(oracle.prior_fallback_ordinal((0, 1)), 1)
        self.assertRaises(ValueError, oracle.prior_fallback_ordinal, ())


class RunnerPlumbingTests(unittest.TestCase):
    def _state(self, ordinals=(0, 1, 2, 8)):
        return {"identity": "state-001", "generator_family": "type010103",
                "source_member": "member-001", "study_role": "predictor_train",
                "engine_seed": 1, "prior_available": True, "prior_ordinal": 8,
                "inventory": [{"ordinal": o, "branch_identity": f"member-001-a{o:02d}",
                               "action": {"drag_x": 0, "drag_y": 0,
                                          "release_time_ms": 1000, "tap_time_ms": 0}}
                              for o in ordinals],
                "typed_dropped_branches": []}

    def _outcomes(self, ordinals):
        return {"member-001": {"schema": "issue_80_candidate_outcomes_v1",
                               "member_identity": "member-001",
                               "candidates": [
                                   {"branch_identity": f"member-001-a{o:02d}",
                                    "ordinal": o,
                                    "action": {"drag_x": 0, "drag_y": 0,
                                               "release_time_ms": 1000, "tap_time_ms": 0},
                                    "realized_count_cost": float(10 + o),
                                    "censored": True, "parser_calls": 2}
                                   for o in ordinals],
                               "lowest_cost": 10.0, "highest_cost": 18.0,
                               "informative": True, "all_tied": False}}

    def test_state_cells_map_every_inventory_row(self) -> None:
        state = self._state()
        cells = runner.state_cells([state], self._outcomes((0, 1, 2, 8)))
        self.assertEqual(len(cells), 4)
        self.assertEqual([cell["ordinal"] for cell in cells], [0, 1, 2, 8])
        self.assertEqual(cells[3]["realized_count_cost"], 18.0)

    def test_prior_choice_types_absence(self) -> None:
        available = runner.typed_choice_for_prior(self._state())
        self.assertIsNone(available["failure"])
        self.assertEqual(available["chosen_ordinal"], 8)
        absent = runner.typed_choice_for_prior(self._state(ordinals=(0, 1, 2)))
        self.assertEqual(absent["failure"], "prior_candidate_absent")
        self.assertIsNone(absent["chosen_ordinal"])

    def test_reference_branch_is_lowest_ordinal(self) -> None:
        state = self._state(ordinals=(3, 1, 8))
        self.assertEqual(
            runner.reference_branch_for([state], "member-001"), "member-001-a01")

    def test_prediction_record_invariants(self) -> None:
        states = [self._state(ordinals=(0, 1, 2, 8))]
        member = "member-001"
        record = {"member_identity": member, "system": "continuous-fixed-h1",
                  "seed": 20260908, "pair": {"delta": 1, "abstraction": "continuous"},
                  "checkpoint_identity": "sha256:x", "candidate_count": 4,
                  "rows": [
                      {"ordinal": o, "predicted_cost": 10.0 + o, "excluded": False,
                       "transition_calls": 1, "linear_macs": 7}
                      for o in (0, 1, 2, 8)]}
        plan = {"dynamics": {"checkpoints": {"20260908": {"continuous": {"identity": "sha256:x"}}}}}
        runner.verify_prediction_record(plan, states, member, record,
                                        "continuous-fixed-h1", 20260908)
        record["rows"][1]["predicted_cost"] = None
        self.assertRaises(ValueError, runner.verify_prediction_record,
                          plan, states, member, record, "continuous-fixed-h1", 20260908)

    def test_source_files_carry_no_placeholders(self) -> None:
        root = Path(runner.__file__).resolve().parent.parent
        for name in runner.FILES:
            text = (root / name).read_text()
            self.assertNotIn("TODO", text)
            self.assertNotIn("FIXME", text)


class RenderingTests(unittest.TestCase):
    def _blocked_result(self):
        binding = oracle.bind_threshold(synthetic_rows(
            (50.0, 60.0, *[40.0] * 10), removed=(0,)))
        best = oracle.best_candidate_row(
            [{"ordinal": 0, "branch_identity": "b0", "realized_count_cost": 40.0},
             {"ordinal": 1, "branch_identity": "b1", "realized_count_cost": 50.0}])
        per_seed = {str(seed): {"states_scored": 2, "typed_decision_failures": 0,
                                "typed_failure_states": [],
                                "predicate_success_states": None,
                                "mean_chosen_realized_count_cost": 55.0,
                                "mean_oracle_best_realized_count_cost": 40.0,
                                "mean_gap_cost": 15.0, "transition_calls": 3,
                                "linear_macs": 21}
                    for seed in runner.SEEDS}
        return {
            "schema": oracle.SCHEMA_REPORT, "identity": "x-report",
            "plan_identity": runner.IDENTITY, "frozen_before_outcome": True,
            "binding": binding, "definitions": {"success_predicate": {"declared_divergence": "d"},
                                                "protocol_note": {"note": "n"},
                                                "uncertainty": {"draws": 10000,
                                                                "seed": 7201}},
            "claim_boundary": "c", "source_issue_80": {}, "issue_77_n1_campaign": {},
            "membership_summary": {"states": 24, "members": 15,
                                   "pilot_states": [], "cells": 177},
            "limitations": [], "archived_release": False,
            "fresh_evaluation_opened": False, "final_evaluation_opened": False,
            "issue_64_authorized": False,
            "compute": {"gpu_seconds_elapsed": 1.0, "wall_seconds_elapsed": 2.0,
                        "gpu_allowance_seconds": 360, "wall_cap_seconds": 3600,
                        "decision_frames": 15, "parser_calls": 15,
                        "predictor_loads": 6, "transition_calls": 9,
                        "linear_macs": 63, "candidate_slots": 12},
            "protocol_note": {"note": "n"},
            "pilot_divergence": {"valid_cells": 45, "typed_failure_cells": 3,
                                 "agreement_fraction": 0.0,
                                 "predicate_success_and_gameplay_success": 0,
                                 "predicate_success_and_gameplay_failure": 45,
                                 "predicate_failure_and_gameplay_success": 0,
                                 "predicate_failure_and_gameplay_failure": 0},
            "system_cost_gap": {"per_system": {name: {"per_seed": per_seed}
                                               for name in oracle.SYSTEMS}},
            "per_state_best_candidates": [
                {"state": "s", "generator_family": "type010103", "source_member": "m",
                 "admissible_candidates": 2, "best_candidate": best,
                 "best_candidate_tied_ordinals": [0]}],
            "system_choice_rows": [],
            "threshold_sensitivity": oracle.threshold_sensitivity(
                synthetic_rows((50.0, 60.0, *[40.0] * 10), removed=(0,))),
            "evaluation_complete": True,
            "ticket_disposition": "readiness_or_precision_insufficient",
            "stopExplanation": "blocked",
            "dispositions": {"q1_oracle_ceiling_nonzero":
                             "readiness_or_precision_insufficient",
                             "q1_reading": None,
                             "q2_disposition": "readiness_or_precision_insufficient",
                             "q2_recommendation": None, "blocker": binding["blocker"],
                             "rules": {}},
        }

    def test_blocked_publication_renders(self) -> None:
        result = self._blocked_result()
        text = runner.findings_md(result)
        csv_text = runner.comparisons_csv(result)
        self.assertIn("Status: no_defensible_binding", text)
        self.assertIn("Blocker:", text)
        self.assertIn("readiness_or_precision_insufficient", text)
        self.assertIn("system,seed,states_scored", csv_text)
        self.assertIn("state,system,seed", csv_text)
        self.assertIn("type010103", csv_text)

    def test_rendering_is_deterministic(self) -> None:
        result = self._blocked_result()
        self.assertEqual(runner.findings_md(result), runner.findings_md(result))
        self.assertEqual(runner.comparisons_csv(result), runner.comparisons_csv(result))


if __name__ == "__main__":
    raise SystemExit(unittest.main())
