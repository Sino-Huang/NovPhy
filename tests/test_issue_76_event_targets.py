from copy import deepcopy
import unittest

from scripts.issue_76_event_targets import data_readiness, observed_target


class EventTargetTests(unittest.TestCase):
    def setUp(self):
        self.summary = {"first_fixed_step": 100, "last_fixed_step": 30100,
                        "observed_window_valid": True, "censored": True, "terminal_observed": False}
        self.case = {"decision_time": 2., "capture_time": 47.}

    def test_capture_deadline_is_not_a_decision_deadline(self):
        target = observed_target(self.summary, None, self.case, .0004)
        self.assertEqual(target["stop_kind"], "right_censored")
        self.assertEqual(target["observed_capture_seconds"], 12.)
        self.assertEqual(target["observed_end_seconds_from_decision"], 57.)
        self.assertIsNone(target["terminal_seconds_from_decision"])
        self.assertIsNone(target["later_clear_by_deadline_label"])

    def test_native_event_types_and_early_stability_remain_distinct(self):
        self.summary.update(last_fixed_step=5100, censored=False, terminal_observed=True)
        for reason, kind in (("level_clear", "native_clear"), ("level_fail", "native_fail"),
                             ("stable_entered", "stable_without_clear")):
            terminal = {"fixed_step": 5100, "reason": reason, "event_id": "event"}
            target = observed_target(self.summary, terminal, self.case, .0004)
            self.assertEqual(target["stop_kind"], kind)
            self.assertEqual(target["native_level_clear"], reason == "level_clear")
            self.assertEqual(target["terminal_seconds_from_capture"], 2.)
            self.assertEqual(target["terminal_seconds_from_decision"], 47.)
            self.assertIsNone(target["later_clear_by_deadline_label"])

    def test_corrupt_or_contradictory_endpoint_is_not_a_negative_label(self):
        with self.assertRaises(ValueError):
            observed_target(self.summary, {"fixed_step": 30100, "reason": "level_clear"}, self.case, .0004)
        self.summary.update(censored=False, terminal_observed=True)
        with self.assertRaises(ValueError):
            observed_target(self.summary, None, self.case, .0004)
        with self.assertRaises(ValueError):
            observed_target(self.summary, {"fixed_step": 20000, "reason": "level_clear"}, self.case, .0004)


class EventReadinessTests(unittest.TestCase):
    def setUp(self):
        sources, assignments, rows = [], [], []
        for role in ("training", "calibration", "model_selection"):
            for family in ("a", "b"):
                identity = role + family
                sources.append(dict(identity=identity, base_cluster=identity,
                                    exposure_role=role, generator_family=family))
                for ordinal in range(10):
                    member = identity + str(ordinal)
                    assignments.append(dict(identity=member, source_member_identity=identity))
                    rows.append(dict(member_identity=member, capture_contract_validated=True,
                                     target={"native_level_clear": True, "stop_kind": "native_clear"}))
        self.inventory = {"source_members": sources, "assignments": assignments,
            "data_readiness": {"minimum_valid_capture_fraction_each_role_family": .9,
                               "minimum_clear_training_lineages": 2,
                               "minimum_clear_calibration_lineages": 2,
                               "minimum_clear_model_selection_lineages": 2}}
        self.rows = rows

    def test_actions_are_not_independent_lineages_and_pass_does_not_authorize_training(self):
        report = data_readiness(self.inventory, self.rows)
        self.assertTrue(report["data_ready"])
        self.assertEqual([role["clear_lineages"] for role in report["roles"]], [2, 2, 2])
        self.assertFalse(report["model_training_authorized"])
        self.assertFalse(report["fresh_access"])
        self.assertFalse(report["advancement_authorized"])
        self.inventory["data_readiness"]["minimum_clear_training_lineages"] = 3
        self.assertFalse(data_readiness(self.inventory, self.rows)["data_ready"])

    def test_unavailable_records_count_against_each_cell_not_as_no_clear(self):
        self.rows[0].update(capture_contract_validated=False, target=None)
        self.assertTrue(data_readiness(self.inventory, self.rows)["data_ready"])
        self.rows[1].update(capture_contract_validated=False, target=None)
        report = data_readiness(self.inventory, self.rows)
        self.assertFalse(report["data_ready"])
        self.assertEqual(report["cells"][0]["valid_fraction"], .8)
        self.assertEqual(report["cells"][0]["stop_counts"], {"native_clear": 8})

    def test_partial_duplicate_or_foreign_inventory_cannot_report_readiness(self):
        duplicate = deepcopy(self.rows)
        duplicate[-1] = duplicate[0]
        foreign = deepcopy(self.rows)
        foreign[-1]["member_identity"] = "foreign"
        for rows in (self.rows[:-1], duplicate, foreign):
            with self.assertRaises(ValueError):
                data_readiness(self.inventory, rows)

    def test_distinct_source_ids_with_same_base_cluster_are_not_independent(self):
        self.inventory["source_members"][1]["base_cluster"] = self.inventory["source_members"][0]["base_cluster"]
        report = data_readiness(self.inventory, self.rows)
        self.assertEqual(report["roles"][0]["clear_lineages"], 1)
        self.assertFalse(report["data_ready"])
