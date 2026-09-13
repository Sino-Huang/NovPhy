from copy import deepcopy
from unittest.mock import patch
import unittest

import torch

from scripts import score_issue_76_fixed_development as scoring
from world_model.training.fixed_development import CURVE_TIMES
from world_model.training.native_history_data import VOCABULARY
from world_model.training.native_history_model import NativeHistoryDynamics, STATE_DIM


def selection_fixture():
    entries = [dict(member_identity=str(i), base_cluster=str(i), family=family, exposure_role="calibration")
               for i, family in enumerate(("A", "A", "B"))]
    inventory = dict(entries=entries, seeds=[1, 2, 3], recipes=["first", "second"], models=[])
    results = []
    for recipe in inventory["recipes"]:
        for seed in inventory["seeds"]:
            for pure in (False, True):
                cell = dict(recipe=recipe, seed=seed, pure=pure)
                inventory["models"].append(cell)
                rows = []
                for entry in entries:
                    policies = {}
                    for pair in (scoring.fit.CONTINUOUS_PAIRS if pure else scoring.fit.PAIRS):
                        # Equal-family means: first=5, second=4. Pooled means:
                        # first=10/3, second=4 would incorrectly choose first.
                        error = (0. if entry["family"] == "A" else 10.) if recipe == "first" else 4.
                        policies[scoring.fit.policy_name(pair)] = dict(prediction_failure=False,
                            curves={"11250": {"recursive": {"failure": None, "carrier_mse": error}}})
                    rows.append(dict(**entry, **cell, available=True, policies=policies))
                results.append(dict(cell=cell, rows=rows))
    return inventory, results


class FixedDevelopmentScoringTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(76)

    def test_selection_uses_equal_family_and_common_seed_choice(self):
        inventory, results = selection_fixture()
        choice = scoring.calibration_choice(inventory, results)
        self.assertEqual(len(choice["table"]), 24)
        for arm in ("hybrid", "pure"):
            self.assertEqual(choice["selected"][arm]["recipe"], "second")
            self.assertEqual(choice["selected"][arm]["endpoint_carrier_mse"], 4.)
        self.assertEqual(choice["selected"]["hybrid"]["policy"], scoring.fit.policy_name(scoring.fit.PAIRS[0]))
        self.assertFalse(choice["gameplay_policy_selection"])
        # One adverse seed must be retained, not replaced by the other recipe.
        for result in results:
            if result["cell"]["recipe"] == "second" and result["cell"]["seed"] == 3:
                for row in result["rows"]:
                    for policy in row["policies"].values():
                        policy["curves"]["11250"]["recursive"]["carrier_mse"] = 100.
        changed = scoring.calibration_choice(inventory, results)
        self.assertTrue(all(value["recipe"] == "first" for value in changed["selected"].values()))

    def test_failures_disqualify_and_incomplete_or_unpaired_matrices_stop(self):
        inventory, results = selection_fixture()
        for result in results:
            if result["cell"]["recipe"] == "second":
                for policy in result["rows"][0]["policies"].values():
                    policy["prediction_failure"] = True
        choice = scoring.calibration_choice(inventory, results)
        self.assertTrue(all(value["recipe"] == "first" for value in choice["selected"].values()))
        with self.assertRaises(ValueError):
            scoring.calibration_choice(inventory, results[:-1])
        for mutation in ("availability", "role", "policy"):
            changed = deepcopy(results)
            row = changed[0]["rows"][0]
            if mutation == "availability":
                row["available"] = False
            elif mutation == "role":
                row["exposure_role"] = "model_selection"
            else:
                row["policies"].pop(next(iter(row["policies"])))
            with self.assertRaises(ValueError):
                scoring.calibration_choice(inventory, changed)

    def test_batched_scoring_preserves_missing_targets_and_exact_work(self):
        model = NativeHistoryDynamics(pure=True, width=16).eval()
        members = []
        for i in range(3):
            initial = torch.randn(STATE_DIM)
            targets = {t: dict(carrier=torch.randn(STATE_DIM), presence=torch.zeros(len(VOCABULARY)),
                               centers=torch.zeros(len(VOCABULARY), 2)) for t in CURVE_TIMES}
            if i == 1:
                del targets[11250]
            member = dict(entry=dict(member_identity=str(i), base_cluster=str(i), family="A", exposure_role="calibration"),
                initial=initial, action=torch.zeros(5), targets=targets,
                contexts={h: {750: initial} for h in (50, 250, 750)})
            if i == 1:
                member["unavailable_reason"] = "first_shot_has_no_exact_4_5_second_observed_endpoint"
            if i == 2:
                member.update(initial=None, targets={}, contexts={}, unavailable_reason="no_usable_observed_segment")
            members.append(member)
        checks = []
        result = scoring.score_cell(model, dict(recipe="synthetic", seed=1, pure=True), members, "cpu",
                                    batch_size=2, check_resources=lambda: checks.append(True))
        self.assertEqual(len(checks), 6)
        self.assertEqual(len(result["rows"]), 3)
        self.assertEqual([row["available"] for row in result["rows"]], [True, False, False])
        self.assertEqual(result["rows"][2]["policies"], {})
        for batch, pair in zip(result["batches"], model.pairs, strict=True):
            self.assertEqual(batch["recursive_transitions"], 2 * 11250 // pair.delta)
            self.assertEqual(batch["local_transitions"], 2)
            self.assertEqual(batch["linear_macs"], (2 * 11250 // pair.delta + 2) * scoring.linear_macs(model, pair))
            self.assertFalse(batch["batch_one_deployment_latency_measured"])
            curves = result["rows"][1]["policies"][scoring.fit.policy_name(pair)]["curves"]
            self.assertEqual(list(curves), list(map(str, CURVE_TIMES)))
            self.assertIsNone(curves["11250"]["recursive"])
            self.assertIsNone(curves["1500"]["observed_context_local"])
        endpoint = result["rows"][0]["policies"][scoring.fit.policy_name(model.pairs[-1])]["curves"]["11250"]
        expected, _ = scoring.fit.rollout(model, None, members[0]["initial"], members[0]["action"], fixed_pair=model.pairs[-1])
        self.assertAlmostEqual(endpoint["recursive"]["carrier_mse"], float((expected - members[0]["targets"][11250]["carrier"]).square().mean()), places=4)

    def test_checkpoint_loader_rejects_unfinished_or_unbound_sources(self):
        model = NativeHistoryDynamics(pure=True, width=16)
        cell = dict(recipe="repaired-representation", seed=1, pure=True,
                    checkpoint_paths=["/synthetic/checkpoints/predictor-pure-1.pt"],
                    source_plan_identity="data", parameter_weights=[1.])
        inventory = dict(models=[cell], capacity={"continuous_width": 16}, data_plan_identity="data")
        plan = dict(identity="data", capacity=inventory["capacity"], seeds=[1], updates={"predictor": 6000})
        checkpoint = dict(plan_identity="data", complete=True, failure=None, updates_requested=6000,
                          updates_completed=6000, updates_applied=5952, updates_skipped=48,
                          model=model.state_dict())
        with patch.object(scoring.fit.files, "read", return_value=plan), \
                patch.object(scoring.fit, "require_finished_budget", return_value={}), \
                patch.object(scoring.torch, "load", return_value=checkpoint):
            loaded, _ = scoring.load_cell(inventory, cell, "cpu")
            self.assertTrue(all(torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items()))
            checkpoint["complete"] = False
            with self.assertRaises(ValueError):
                scoring.load_cell(inventory, cell, "cpu")
            checkpoint["complete"] = True
            checkpoint["plan_identity"] = "different-source"
            with self.assertRaises(ValueError):
                scoring.load_cell(inventory, cell, "cpu")


if __name__ == "__main__":
    unittest.main()
