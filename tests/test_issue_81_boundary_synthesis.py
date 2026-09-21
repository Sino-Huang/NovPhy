"""Focused tests for issue-81 boundary synthesis (fast, CPU, synthetic).

Covers the frozen decision rules and table conventions without touching the
published artifact directories: pooling-rule mechanics (the only aggregation is
a three-seed mean inside one cell), the claim-registry disposition rule, the
reactive UNEXECUTED stop branch, zero-shot/few-shot separation, the issue-74
allowed-fields reduction, and the published emitters.
"""
import unittest

from scripts import run_boundary_synthesis as runner


def make_row(**overrides):
    row = {"family": "novphy_normal_mechanics", "condition": "adapted", "side": None,
           "kind": "training_effect", "tested": "hybrid_continuous_h1",
           "reference": "continuous_h1", "horizon": 1, "endpoint": None, "scope_states": 4,
           "scope": "descriptive", "positive_is_improvement": True,
           "positive_favors": "hybrid_continuous_h1", "metric": "normalized_ranking_regret",
           "mean": 0.1, "interval_low": 0.01, "interval_high": 0.2, "excludes_zero": True,
           "interval_label": "descriptive_95_percent_interval",
           "estimand": runner.REGRET_ESTIMAND, "source_artifact": "issue-77-n1-diagnostic-v1",
           "status": "executed"}
    row.update(overrides)
    return row


class ConventionTests(unittest.TestCase):
    def test_seed_mean_is_the_only_aggregation_and_is_strict(self):
        self.assertAlmostEqual(runner.seed_mean([1.0, 2.0, 3.0]), 2.0)
        with self.assertRaises(ValueError):
            runner.seed_mean([1.0, 2.0])
        with self.assertRaises(ValueError):
            runner.seed_mean([1.0, 2.0, 3.0, 4.0])

    def test_seed_mean_propagates_null_curve_aggregates(self):
        self.assertIsNone(runner.seed_mean([1.0, None, 3.0]))

    def test_direction_and_zero_separation_conventions(self):
        self.assertEqual(runner.direction(0.05), "tested_favoring")
        self.assertEqual(runner.direction(-0.05), "reference_favoring")
        self.assertEqual(runner.direction(0.0), "tied")
        self.assertTrue(runner.excludes_zero(0.01, 0.2))
        self.assertTrue(runner.excludes_zero(-0.2, -0.01))
        self.assertFalse(runner.excludes_zero(-0.2, 0.2))

    def test_horizon_parsed_from_system_name(self):
        self.assertEqual(runner.horizon_of("hybrid_micro_h15"), 15)
        self.assertEqual(runner.horizon_of("tawm_adaptive"), None)


class DecisionRuleTests(unittest.TestCase):
    def test_all_rows_supported_when_direction_and_separation_match(self):
        token, checks = runner.check_expectation(
            {"type": "all_rows", "rows": [
                {"match": {"family": "novphy_normal_mechanics", "kind": "training_effect",
                           "tested": "hybrid_continuous_h1"},
                 "direction": "tested_favoring", "excludes_zero": True}]},
            [make_row()], {})
        self.assertEqual(token, "supported")
        self.assertTrue(all(check["match"] for check in checks))

    def test_all_rows_not_supported_when_direction_contradicts(self):
        token, _ = runner.check_expectation(
            {"type": "all_rows", "rows": [
                {"match": {"tested": "hybrid_continuous_h1"}, "direction": "reference_favoring",
                 "excludes_zero": None}]},
            [make_row()], {})
        self.assertEqual(token, "not_supported_by_this_experiment")

    def test_all_rows_not_supported_when_separation_contradicts(self):
        token, _ = runner.check_expectation(
            {"type": "all_rows", "rows": [
                {"match": {"tested": "hybrid_continuous_h1"}, "direction": None,
                 "excludes_zero": False}]},
            [make_row()], {})
        self.assertEqual(token, "not_supported_by_this_experiment")

    def test_missing_row_is_readiness_not_contradiction(self):
        token, _ = runner.check_expectation(
            {"type": "all_rows", "rows": [
                {"match": {"tested": "absent_arm"}, "direction": "tested_favoring",
                 "excludes_zero": True}]},
            [make_row()], {})
        self.assertEqual(token, "readiness_or_precision_insufficient")

    def test_side_sign_inconsistency_rule(self):
        expectation = {"type": "side_sign_inconsistency", "rows": [
            {"match": {"side": side, "tested": "hybrid_continuous_h1"}}
            for side in ("a", "b")]}
        inconsistent = [make_row(side="a", mean=0.1), make_row(side="b", mean=-0.1)]
        consistent = [make_row(side="a", mean=0.1), make_row(side="b", mean=0.2)]
        self.assertEqual(runner.check_expectation(expectation, inconsistent, {})[0], "supported")
        self.assertEqual(runner.check_expectation(expectation, consistent, {})[0],
                         "not_supported_by_this_experiment")

    def test_condition_sign_disagreement_reports_failing_side(self):
        expectation = {"type": "condition_sign_disagreement",
                       "conditions": ["zero-shot", "few-shot"],
                       "rows": [{"match": {"condition": condition, "side": side}}
                                for condition in ("zero-shot", "few-shot")
                                for side in ("a", "b")]}
        rows = [make_row(condition="zero-shot", side="a", mean=0.1),
                make_row(condition="few-shot", side="a", mean=0.2),
                make_row(condition="zero-shot", side="b", mean=0.1),
                make_row(condition="few-shot", side="b", mean=-0.2)]
        token, checks = runner.check_expectation(expectation, rows, {})
        self.assertEqual(token, "supported")
        self.assertEqual(checks[0]["disagreeing_sides"], ["b"])

    def test_upstream_token_mismatch_is_not_supported(self):
        sources = {"artifact": {"document": {"dispositions": {"q1": "supported"}}}}
        token, _ = runner.check_expectation(
            {"type": "upstream_token", "artifact": "artifact", "field": "dispositions.q1",
             "token": "readiness_or_precision_insufficient"}, [], sources)
        self.assertEqual(token, "not_supported_by_this_experiment")

    def test_field_values_rule_checks_nested_paths(self):
        sources = {"a74": {"reduced": {"selected_continuous_policy": "continuous_h5",
                                       "compute": {"training_compute_matched": False}}}}
        token, _ = runner.check_expectation(
            {"type": "field_values", "artifact": "a74",
             "values": {"selected_continuous_policy": "continuous_h5",
                        "compute.training_compute_matched": False}}, [], sources)
        self.assertEqual(token, "supported")

    def test_growth_margin_flags_sub_margin_arms(self):
        rows = [{"mechanism": "recursive_error_growth", "family": "f", "arm": "ok",
                 "growth_ratio": 12.0},
                {"mechanism": "recursive_error_growth", "family": "f", "arm": "weak",
                 "growth_ratio": 3.0}]
        token, checks = runner.check_expectation(
            {"type": "growth_margin", "margin": 10, "rows": [
                {"match": {"family": "f", "arm": "ok"}},
                {"match": {"family": "f", "arm": "weak"}}]}, rows, {})
        self.assertEqual(token, "not_supported_by_this_experiment")
        self.assertEqual(checks[0]["failing_rows"], ["f::::weak"])

    def test_untested_cell_rule_returns_stop_branch_token(self):
        token, _ = runner.check_expectation(
            {"type": "untested_cell", "match": {"family": "reactive_diagnostic",
                                                "status": "UNEXECUTED"}},
            [make_row(family="reactive_diagnostic", status="UNEXECUTED")], {})
        self.assertEqual(token, "untested")

    def test_condition_disagreement_missing_row_is_readiness(self):
        expectation = {"type": "condition_sign_disagreement",
                       "conditions": ["zero-shot", "few-shot"],
                       "rows": [{"match": {"condition": condition, "side": side}}
                                for condition in ("zero-shot", "few-shot")
                                for side in ("a", "b")]}
        rows = [make_row(condition="zero-shot", side="a", mean=0.1),
                make_row(condition="few-shot", side="a", mean=0.2),
                make_row(condition="zero-shot", side="b", mean=0.1)]
        token, _ = runner.check_expectation(expectation, rows, {})
        self.assertEqual(token, "readiness_or_precision_insufficient")

    def test_descriptor_resolving_to_multiple_rows_is_readiness(self):
        expectation = {"type": "side_sign_inconsistency", "rows": [
            {"match": {"side": None, "tested": "hybrid_continuous_h1"}}]}
        duplicates = [make_row(mean=0.1), make_row(mean=-0.1)]
        token, _ = runner.check_expectation(expectation, duplicates, {})
        self.assertEqual(token, "readiness_or_precision_insufficient")


class FrozenStructureTests(unittest.TestCase):
    def test_five_families_present_with_reactive_unexecuted(self):
        self.assertEqual(set(runner.FAMILIES), {
            "novphy_normal_mechanics", "appearance_novelty", "external_temporal_baselines",
            "clevrer_replication", "reactive_diagnostic"})
        self.assertEqual(runner.FAMILIES["reactive_diagnostic"]["status"], "UNEXECUTED")

    def test_zero_shot_and_few_shot_are_separate_never_pooled(self):
        appearance = runner.FAMILIES["appearance_novelty"]
        self.assertEqual(appearance["conditions"], ("zero-shot", "few-shot"))
        self.assertIn("never pooled", appearance["condition_separation"])

    def test_claim_ids_unique_and_expectation_types_implemented(self):
        claims = runner.claim_definitions()
        ids = [claim["id"] for claim in claims]
        self.assertEqual(len(ids), len(set(ids)))
        for claim in claims:
            self.assertIn(claim["expectation"]["type"],
                          {"all_rows", "side_sign_inconsistency", "condition_sign_disagreement",
                           "upstream_token", "field_values", "growth_margin", "boundary_shape",
                           "untested_cell"})
            if claim.get("stop_branch"):
                self.assertEqual(claim["expectation"]["type"], "untested_cell")

    def test_stop_branch_claim_targets_reactive_family(self):
        stop = [claim for claim in runner.claim_definitions() if claim.get("stop_branch")]
        self.assertEqual(len(stop), 1)
        self.assertEqual(stop[0]["families"], ["reactive_diagnostic"])
        self.assertIn("untested", stop[0]["note"])

    def test_regret_estimand_carries_censoring_language(self):
        self.assertIn("right-censored", runner.REGRET_ESTIMAND)
        self.assertIn("NOT a settled cost", runner.REGRET_ESTIMAND)


class Issue74ReductionTests(unittest.TestCase):
    def test_only_allowed_fields_surface(self):
        document = {"selected_continuous_policy": "continuous_h5", "selection_optimism": True,
                    "compute": {"training_compute_matched": False},
                    "per_seed": {"20260908": {"prior_mean_regret": 0.3,
                                              "offline_anchor_audit_seconds": 1.0,
                                              "policies": {
                                                  "continuous_h5": {
                                                      "mean_regret": 0.31,
                                                      "mean_perception_planning_seconds": 0.37,
                                                      "linear_macs": 1000,
                                                      "mean_endpoint_carrier_mse": 0.17}}}},
                    "contrasts_positive_favors_hybrid": True}
        plan_document = {"development": {"states": 200, "candidates": 12},
                         "everything_else_must_not_surface": True}
        reduced = runner.issue74_allowed_fields(document, plan_document)
        self.assertEqual(set(reduced), {"selected_continuous_policy", "compute", "policies",
                                        "cohort_states"})
        self.assertEqual(reduced["cohort_states"], 200)
        self.assertEqual(reduced["policies"]["continuous_h5"]["20260908"],
                         {"mean_regret": 0.31, "mean_perception_planning_seconds": 0.37})


class EmitterTests(unittest.TestCase):
    def report(self):
        t1 = [make_row(),
              make_row(family="reactive_diagnostic", condition="UNEXECUTED", kind=None,
                       tested=None, reference=None, horizon=None, metric=None, mean=None,
                       interval_low=None, interval_high=None, excludes_zero=None,
                       positive_is_improvement=None, positive_favors=None, scope_states=0,
                       status="UNEXECUTED")]
        t2 = [{"family": "novphy_normal_mechanics", "condition": "adapted", "side": None,
               "arm": "continuous_h5", "states": 4, "estimand": runner.REGRET_ESTIMAND,
               "quality_metric": "normalized_ranking_regret",
               "mean_regret": 0.5, "wall_seconds_per_state": 1.5, "wall_definition": "wall-x",
               "transition_linear_macs": 1e9, "linear_macs_definition": "total",
               "transition_calls": 10.0, "parameters": None, "controller_parameters": None,
               "candidate_count": None, "candidate_count_definition": "def",
               "source_artifact": "issue-77-n1-diagnostic-v1", "status": "executed"},
              {"family": "reactive_diagnostic", "condition": "UNEXECUTED", "side": None,
               "arm": None, "states": 0, "estimand": runner.REGRET_ESTIMAND,
               "quality_metric": None, "mean_regret": None,
               "wall_seconds_per_state": None, "wall_definition": None,
               "transition_linear_macs": None, "linear_macs_definition": None,
               "transition_calls": None, "parameters": None, "controller_parameters": None,
               "candidate_count": None, "candidate_count_definition": None,
               "source_artifact": "issue-80-reactive-diagnostic-v1", "status": "UNEXECUTED"}]
        t3 = {"recursive_error_growth": [
                  {"mechanism": "recursive_error_growth", "family": "novphy_normal_mechanics",
                   "condition": "adapted", "side": None, "arm": "continuous_h5",
                   "metric": "carrier_mse", "short_endpoint": 15, "long_endpoint": 600,
                   "mse_short": 0.02, "mse_long": 200.0, "growth_ratio": 10000.0,
                   "status": "executed", "source_artifact": "issue-77-n1-diagnostic-v1"},
                  {"mechanism": "recursive_error_growth", "family": "external_temporal_baselines",
                   "condition": "adapted", "side": None, "arm": "tawm_adaptive",
                   "metric": "carrier_mse", "short_endpoint": 15, "long_endpoint": 600,
                   "mse_short": None, "mse_long": None, "growth_ratio": None,
                   "status": "curve_unavailable_in_source",
                   "source_artifact": "issue-78-external-temporal-baselines-v1"}],
              "symbolic_execution_effects": [],
              "regime_dependence": [
                  {"mechanism": "regime_dependence", "family": "clevrer_replication",
                   "condition": "out-of-family retrained", "horizon": 1, "endpoint": 120,
                   "still": {"mean": -0.8}, "active": {"mean": -0.9},
                   "difference_active_minus_still": -0.1, "source_artifact": "x"},
                  {"mechanism": "regime_dependence", "family": "novphy_normal_mechanics",
                   "condition": None, "horizon": None, "endpoint": None, "still": None,
                   "active": None, "difference_active_minus_still": None,
                   "status": "not_measured_in_family", "source_artifact": "y"}],
              "typed_failure_concentrations": [
                  {"mechanism": "typed_failure_concentration", "family": "reactive_diagnostic",
                   "total_dropped_candidates": 0, "typed_failures_count": 3,
                   "states_affected": 0, "states": 0,
                   "note": "UNEXECUTED cell", "source_artifact": "z"}]}
        registry = [{"id": "rc-claim", "claim": "c", "families": ["reactive_diagnostic"],
                     "conditions": ["UNEXECUTED"], "contributing_rows": [
                         {"family": "reactive_diagnostic", "status": "UNEXECUTED"}],
                     "expectation": {"type": "untested_cell"}, "disposition": "untested",
                     "checks": [{"match": True}], "note": "n", "stop_branch": True}]
        return {"plan_identity": "issue-81-boundary-synthesis-v1", "feeds_ticket": 37,
                "pooling_rule": "cells side by side; NO cross-family meta-analytic pooling",
                "estimands": {"regret": runner.REGRET_ESTIMAND,
                              "clevrer": runner.CLEVRER_ESTIMAND,
                              "interval_label": runner.INTERVAL_LABEL,
                              "scope": runner.DESCRIPTIVE_SCOPE},
                "selection_disclosure": runner.SELECTION_DISCLOSURE,
                "claim_boundary": "boundary", "diagnostics_complete": True,
                "tables": {"t1_advantage_vs_horizon": t1, "t2_work_frontier": t2, "t3_mechanism": t3},
                "claim_registry": registry,
                "stop_branch": {"family": "reactive_diagnostic", "status": "UNEXECUTED",
                                "terminal_disposition": "readiness_or_precision_insufficient",
                                "stop_explanation": "gate failed", "registry_tokens": ["untested"]},
                "families": {"reactive_diagnostic": {"pilot_gate": {
                    "prevalence": 0.0, "successes": 0, "valid_executions": 45, "floor": 0.1,
                    "passed": False, "cells": 48, "typed_failures": []}}},
                "limitations": ["four states only"]}

    def test_comparisons_csv_contains_all_sections_and_unexecuted_rows(self):
        text = runner.comparisons_csv(self.report())
        for section in ("t1_advantage_vs_horizon", "t2_work_frontier",
                        "t3_mechanism.recursive_error_growth", "t3_mechanism.regime_dependence",
                        "t3_mechanism.typed_failure_concentrations"):
            self.assertIn(section, text)
        self.assertIn("reactive_diagnostic,UNEXECUTED", text)
        self.assertIn("curve_unavailable_in_source", text)

    def test_findings_keeps_conditions_separate_and_reports_stop_branch(self):
        text = runner.findings_md(self.report())
        self.assertIn("Advantage vs horizon - ranking-regret cells", text)
        self.assertIn("UNEXECUTED cell (mandatory stop branch)", text)
        self.assertIn("readiness_or_precision_insufficient", text)
        self.assertIn("Boundary conditions (contradictions reported, not reconciled)", text)
        self.assertIn("NOT a settled cost", text)
        self.assertIn("NO cross-family meta-analytic pooling", text)

    def test_registry_csv_records_untested_token(self):
        text = runner.claim_registry_csv(self.report()["claim_registry"])
        self.assertIn("rc-claim", text)
        self.assertIn("untested", text)

    def test_advantage_csv_includes_reactive_placeholder_row(self):
        text = runner.advantage_csv(self.report())
        self.assertIn("reactive_diagnostic,UNEXECUTED", text)
        self.assertIn("descriptive_95_percent_interval", text)

    def test_emitters_survive_the_compute_json_round_trip(self):
        import json
        report = self.report()
        round_tripped = json.loads(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        self.assertEqual(runner.comparisons_csv(report), runner.comparisons_csv(round_tripped))
        self.assertEqual(runner.findings_md(report), runner.findings_md(round_tripped))
        self.assertEqual(runner.advantage_csv(report), runner.advantage_csv(round_tripped))
        self.assertEqual(runner.claim_registry_csv(report["claim_registry"]),
                         runner.claim_registry_csv(round_tripped["claim_registry"]))


if __name__ == "__main__":
    unittest.main()
