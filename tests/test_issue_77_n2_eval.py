import argparse
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_issue_77_n2_eval as ev


def fixture_n2_campaign():
    members, branches = [], []
    held = {7, 8, 14, 16}
    for ordinal in range(1, 17):
        family = "type010101" if ordinal <= 8 else "type010102"
        if ordinal % 5 == 0:
            role, partition, study = "training", "controller", "controller_only"
        elif ordinal in held:
            role, partition, study = "calibration", None, "held_out_evaluation"
        else:
            role, partition, study = "training", "predictor", "predictor_train"
        identity = f"issue-77-n2-{ordinal:03d}"
        members.append({"identity": identity, "ordinal": ordinal, "generator_family": family,
                        "exposure_role": role, "fit_partition": partition, "study_role": study})
        for action in range(13):
            branches.append({"identity": f"{identity}-a{action:02d}",
                             "source_member_identity": identity, "candidate_ordinal": action,
                             "action": {"drag_x": -80, "drag_y": 10,
                                        "tap_time_ms": 0, "release_time_ms": 1000}})
    plan = {"identity": "issue-77-n2-appearance-v1", "members": members, "branches": branches}
    coverage = {"schema": "issue_77_n2_coverage_v1", "coverage_complete": True, "branches": {}}
    for branch in branches:
        status = "admissible"
        # predictor lineage 002 loses every branch; held-out 016 loses two candidates
        if branch["source_member_identity"] == "issue-77-n2-002":
            status = "failed"
        if branch["source_member_identity"] == "issue-77-n2-016" and branch["candidate_ordinal"] >= 11:
            status = "failed"
        coverage["branches"][branch["identity"]] = {"status": status}
    return plan, coverage


def fixture_r3_campaign():
    members, branches = [], []
    for ordinal in range(31, 51):
        identity = f"issue-76-bounded-transfer-{ordinal:03d}"
        members.append({"identity": identity, "ordinal": ordinal, "generator_family": "type010101",
                        "exposure_role": "calibration", "fit_partition": None,
                        "study_role": "held_out_model_selection"})
        for action in range(13):
            branches.append({"identity": f"{identity}-a{action:02d}",
                             "source_member_identity": identity, "candidate_ordinal": action,
                             "action": {"drag_x": -80, "drag_y": 10,
                                        "tap_time_ms": 0, "release_time_ms": 1000}})
    plan = {"identity": "issue-76-bounded-transfer-v1", "members": members, "branches": branches}
    coverage = {"schema": "issue_76_bounded_transfer_coverage_v1", "branches": {}}
    for branch in branches:
        status = "admissible"
        if branch["source_member_identity"] in ("issue-76-bounded-transfer-031",
                                                "issue-76-bounded-transfer-040"):
            status = "failed"
        coverage["branches"][branch["identity"]] = {"status": status}
    return plan, coverage


class Issue77N2EvalTests(unittest.TestCase):
    def test_cli_help_resolves_defaults_without_data_access(self):
        with patch("sys.argv", ["n2eval", "--help"]), self.assertRaises(SystemExit) as ended:
            ev.main()
        self.assertEqual(ended.exception.code, 0)

    def test_two_sample_interval_is_deterministic_and_signed(self):
        a = [0.4, 0.5, 0.45]
        b = [0.1, 0.2, 0.15]
        first = ev.two_sample_interval(a, b)
        self.assertEqual(first, ev.two_sample_interval(a, b))
        self.assertAlmostEqual(first["mean"], 0.3)
        low, high = first["descriptive_95_percent_interval"]
        self.assertGreater(low, 0.0)
        self.assertEqual((first["draws"], first["seed"], first["unit"]), (10000, 7201, "state"))
        flipped = ev.two_sample_interval(b, a)
        self.assertAlmostEqual(flipped["mean"], -0.3)

    def test_action_grids_reject_mismatch(self):
        _, coverage = fixture_n2_campaign()
        plan_a = {"branches": fixture_n2_campaign()[0]["branches"]}
        plan_b = copy.deepcopy(plan_a)
        self.assertEqual(len(ev._action_grids(plan_a, plan_b)), 13)
        plan_b["branches"][-1]["action"] = dict(plan_b["branches"][-1]["action"], drag_x=-70)
        with self.assertRaisesRegex(ValueError, "different fixed action grids"):
            ev._action_grids(plan_a, plan_b)

    def test_r3_first_eight_rule_skips_zero_admissible_lineages(self):
        plan, coverage = fixture_r3_campaign()
        reads = {"plan.json": plan, "coverage.json": coverage}
        with patch.object(ev, "read", lambda path: reads[Path(str(path)).name]):
            members, bound_plan, bound_coverage = ev.r3_normal_members()
        identities = [m["identity"] for m in members]
        self.assertEqual(identities, [f"issue-76-bounded-transfer-{o:03d}" for o in range(32, 40)])
        self.assertIs(bound_plan, plan)
        self.assertIs(bound_coverage, coverage)

    def test_n2_pools_absorb_typed_failures(self):
        plan, coverage = fixture_n2_campaign()
        reads = {"plan.json": plan, "coverage.json": coverage}
        with patch.object(ev, "read", lambda path: reads[Path(str(path)).name]):
            held_out, adaptation, dropped, _, _ = ev.n2_novel_members()
        self.assertEqual([m["identity"] for m in held_out],
                         ["issue-77-n2-007", "issue-77-n2-008", "issue-77-n2-014", "issue-77-n2-016"])
        identities = [m["identity"] for m in adaptation]
        self.assertNotIn("issue-77-n2-002", identities)
        self.assertEqual([m["identity"] for m in dropped], ["issue-77-n2-002"])
        for forbidden in ("issue-77-n2-005", "issue-77-n2-010", "issue-77-n2-015",
                          "issue-77-n2-007", "issue-77-n2-008"):
            self.assertNotIn(forbidden, identities)

    def test_select_samples_orchestrates_sides_and_records(self):
        n2_plan, n2_coverage = fixture_n2_campaign()
        r3_plan, r3_coverage = fixture_r3_campaign()
        n2n_members = [{"identity": f"issue-77-n2n-{o:03d}", "ordinal": o,
                        "generator_family": "type010102", "study_role": "held_out_evaluation"}
                       for o in range(1, 9)]
        r3_members = [m for m in r3_plan["members"][:8]]
        n2_held = [m for m in n2_plan["members"] if m["study_role"] == "held_out_evaluation"]
        n2_adapt = [m for m in n2_plan["members"]
                    if m["exposure_role"] == "training" and m["fit_partition"] == "predictor"
                    and m["identity"] != "issue-77-n2-002"]
        with patch.object(ev, "r3_normal_members",
                          lambda: (r3_members, r3_plan, r3_coverage)), \
             patch.object(ev, "n2_novel_members",
                          lambda: (n2_held, n2_adapt, [], n2_plan, n2_coverage)), \
             patch.object(ev, "n2n_normal_members",
                          lambda: (n2n_members, {"branches": []}, {"branches": {}})), \
             patch.object(ev, "_action_grids", lambda *plans: {i: {} for i in range(13)}), \
             patch.object(ev, "_member_samples",
                          lambda root, plan, cov, members: [{"state": m, "sources": ["x"],
                                                             "dropped_candidates": []}
                                                            for m in members]):
            sides, adaptation, _ = ev.select_samples()
        self.assertEqual([(s["pair"], s["side"]) for s in sides],
                         [("type010101", "normal"), ("type010101", "novel"),
                          ("type010102", "normal"), ("type010102", "novel")])
        self.assertEqual([len(s["samples"]) for s in sides], [8, 2, 8, 2])
        self.assertEqual(len(adaptation["lineages"]), 8)
        record = adaptation["records"]["issue-77-n2-001"]
        self.assertEqual((record["scheduled_branches"], record["dropped_branches"]), (13, 0))
        self.assertEqual(len(record["branches"]), 13)
        record16 = adaptation["records"].get("issue-77-n2-016")
        self.assertIsNone(record16)
        self.assertEqual(adaptation["dropped_lineages"], [])

    def test_adapt_binding_excludes_membership_blobs(self):
        plan = {"identity": "p", "contract": {"x": 1}, "capacity": {"continuous_width": 3},
                "dynamics_plan_identity": "d",
                "adaptation": {**ev.ADAPTATION, "lineages": ["a"], "members": [{"big": 1}],
                               "records": {"a": {"big": 2}}}}
        binding = ev.adapt_binding(plan, 20260908, 764400001, "hybrid")
        self.assertNotIn("members", binding["adaptation"])
        self.assertNotIn("records", binding["adaptation"])
        self.assertNotIn("lineages", binding["adaptation"])
        self.assertEqual((binding["seed"], binding["base_seed"], binding["arm"]),
                         (764400001, 20260908, "hybrid"))
        self.assertEqual(binding["component"], "adapted-predictor")

    def test_curve_path_scopes_conditions_separately(self):
        args = argparse.Namespace(output=Path("/tmp/x"))
        state = {"identity": "s1"}
        zero = ev.curve_path(args, "zero-shot", 1, state, "continuous_h1", 3)
        few = ev.curve_path(args, "few-shot", 1, state, "continuous_h1", 3)
        self.assertNotEqual(zero, few)
        self.assertIn("zero-shot", str(zero))
        self.assertIn("few-shot", str(few))

    def test_publication_is_fail_closed_while_incomplete(self):
        plan = {"identity": "issue-77-n2-eval-v1", "definitions": {}, "adaptation": dict(ev.ADAPTATION),
                "sides": [{"pair": "type010101", "side": "novel",
                           "samples": [{"state": {"identity": "s1"}, "sources": [],
                                        "dropped_candidates": [], "unsupported": "fixture"}]}]}
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=Path(directory), device="cpu")
            result = ev.publication(args, plan)
        self.assertFalse(result["evaluation_complete"])
        self.assertNotIn("conditions", result)
        self.assertEqual(result["inventory"]["unsupported_states"], 1)


if __name__ == "__main__":
    unittest.main()
