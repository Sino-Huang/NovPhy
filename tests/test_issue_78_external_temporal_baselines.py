"""Focused tests for issue-78 external temporal-adaptation baselines (TAWM, VLWM)."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from scripts import run_issue_71_hybrid_readiness as old
from scripts import run_issue_74_matched_dynamics as base
from scripts import run_issue_77_n1_train as train
from scripts import run_issue_77_n1_diagnostic as diag
from scripts import run_external_temporal_baselines as runner
from tests.test_issue_71_hybrid_readiness import batch
from world_model.model import Abstraction, PredictionPair
from world_model.training import external_temporal_baselines as eb
from world_model.training.matched_dynamics import (
    ContinuousDynamics, MatchedController, continuous_loss, parameter_count,
)


class TawmScheduleTests(unittest.TestCase):
    def test_uniform_round_robin_counts(self):
        counts = eb.tawm_horizon_counts(9000)
        self.assertEqual(counts, {1: 3000, 5: 3000, 15: 3000})
        self.assertEqual([eb.tawm_horizon(s) for s in range(6)], [1, 5, 15, 1, 5, 15])

    def test_counts_sum_to_budget(self):
        for steps in (1, 2, 18, 9000):
            self.assertEqual(sum(eb.tawm_horizon_counts(steps).values()), steps)


class VlwmCurriculumTests(unittest.TestCase):
    def test_stage_function_matches_equation_five(self):
        total = 9000
        self.assertEqual([eb.vlwm_stage(t, total) for t in (1, 2999, 3000)], [1, 1, 1])
        self.assertEqual([eb.vlwm_stage(t, total) for t in (3001, 6000)], [2, 2])
        self.assertEqual([eb.vlwm_stage(t, total) for t in (6001, 9000)], [3, 3])
        with self.assertRaises(ValueError):
            eb.vlwm_stage(0, total)
        with self.assertRaises(ValueError):
            eb.vlwm_stage(total + 1, total)

    def test_curriculum_counts_match_equation_six_round_robin(self):
        counts = eb.vlwm_horizon_counts(9000)
        self.assertEqual(counts, {1: 5500, 5: 2500, 15: 1000})
        self.assertEqual(eb.vlwm_horizon(0, 9000), 1)
        self.assertEqual(eb.vlwm_horizon(3000, 9000), 1)
        self.assertEqual(eb.vlwm_horizon(3001, 9000), 5)
        self.assertEqual(eb.vlwm_horizon(6000, 9000), 1)
        self.assertEqual(eb.vlwm_horizon(6001, 9000), 5)
        self.assertEqual(eb.vlwm_horizon(6002, 9000), 15)
        self.assertEqual(sum(counts.values()), 9000)

    def test_smoke_budget_curriculum(self):
        self.assertEqual(eb.vlwm_horizon_counts(18), {1: 11, 5: 5, 15: 2})


class VlwmLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(78)
        self.model = ContinuousDynamics(16)
        self.data = batch(size=3)

    def test_direct_prediction_matches_manual_equation_three(self):
        data = {k: v.clone() for k, v in self.data.items()}
        data["length"] = torch.tensor([60, 47, 52])
        k = 5
        loss = eb.vlwm_loss(self.model, data["z"], data["action"], data["length"], k)
        pair = PredictionPair(k, Abstraction.CONTINUOUS)
        frames = data["z"].shape[1]
        positions = frames - k
        contexts = data["z"][:, :positions].reshape(-1, 236)
        actions = data["action"][:, None].expand(-1, positions, -1).reshape(-1, 5)
        with torch.no_grad():
            predicted = self.model.carrier(contexts, actions, pair).reshape(3, positions, 236)
        keep = (data["length"][:, None] >= torch.arange(k, k + positions)[None, :]).float()
        error = (predicted - data["z"][:, k:]).square().mean(-1)
        bound = F.relu(predicted.abs() - 2).square().mean(-1)
        expected = ((error * keep).sum() / keep.sum() + .01 * bound.sum() / keep.sum())
        self.assertAlmostEqual(float(loss), float(expected), delta=1e-4)

    def test_single_shot_mapping_conditions_on_launch_action_and_horizon(self):
        data = self.data
        # The FiLM modulation is zero-initialized, which makes the conditioner code
        # gradient exactly zero on the very first backward; perturb it so the test
        # observes the trained regime, as the frozen recipe does after its first steps.
        for block in self.model.blocks:
            torch.nn.init.normal_(block.modulation.weight, std=.02)
        with torch.no_grad():
            # The launch action enters the carrier's single 5-value slot unchanged; the
            # k-1 null actions of the paper's sequence contribute nothing (zeros), and
            # the sequence length k enters only through the horizon conditioning.
            code_short = self.model.conditioner.code(PredictionPair(5, Abstraction.CONTINUOUS),
                                                     3, data["z"].device)
            code_long = self.model.conditioner.code(PredictionPair(15, Abstraction.CONTINUOUS),
                                                    3, data["z"].device)
        self.assertFalse(torch.allclose(code_short, code_long))
        loss = eb.vlwm_loss(self.model, data["z"], data["action"], data["length"], 15)
        loss.backward()
        gradients = [n for n, p in self.model.named_parameters() if p.grad is not None
                     and bool((p.grad != 0).any())]
        self.assertTrue(gradients, "vlwm loss must reach every carrier parameter")
        self.assertIn("conditioner", " ".join(gradients))

    def test_masking_and_typed_failures(self):
        # Short windows are masked, not fatal, as long as some start predicts;
        # matching the carrier recipe's masking conventions.
        z = self.data["z"].clone()
        length = torch.tensor([60, 3, 52])
        loss = eb.vlwm_loss(self.model, z, self.data["action"], length, 15)
        self.assertTrue(bool(torch.isfinite(loss)))
        with self.assertRaises(ValueError):
            eb.vlwm_loss(self.model, z[:, :10], self.data["action"],
                         torch.tensor([9, 9, 9]), 15)
        with self.assertRaises(ValueError):
            eb.vlwm_loss(self.model, z, self.data["action"], length, 61)


class GpuLockTests(unittest.TestCase):
    def test_exclusive_lock_blocks_second_holder(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.lock"
            with eb.gpu_exclusive(path):
                other = open(path, "w")
                with self.assertRaises(OSError):
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(other, fcntl.LOCK_UN)
                other.close()
            third = open(path, "w")
            fcntl.flock(third, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(third, fcntl.LOCK_UN)
            third.close()


class PairedSeedIdentityTests(unittest.TestCase):
    @staticmethod
    def namespace():
        return argparse.Namespace(output=runner.DYNAMICS, dynamics=runner.DYNAMICS,
                                  issue71=old.ROOT, parser=old.repair.ROOT, device="cpu")

    def test_identical_minibatches_per_paired_seed_across_arms(self):
        plan77 = train.load_plan(runner.dynamics_args(self.namespace()))
        source = old.load_plan(train.source_args(runner.dynamics_args(self.namespace())))
        batches = {}
        for arm in runner.ARMS:
            generator = torch.Generator().manual_seed(runner.SEEDS[0])
            seen = []
            cached = None
            for step in range(27):
                group = train.fitting_lineage_group(plan77, step)
                if group != cached:
                    rows = [train.load_shard_entry(self.namespace(), plan77, i, source, True)
                            for i in group]
                    data = {k: torch.cat([r[k] for r in rows]) for k in rows[0]}
                    cached = group
                seen.append(old.sample(data, 64, generator, "cpu")["z"][0, 0, :4].tolist())
            batches[arm] = seen
        self.assertEqual(batches["tawm"], batches["vlwm"])


class FrozenPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = argparse.Namespace(output=runner.OUTPUT, dynamics=runner.DYNAMICS,
                                      issue71=old.ROOT, parser=old.repair.ROOT, device="cpu")
        cls.plan = runner.make_plan(cls.args)

    def test_frozen_numbers_are_real(self):
        plan = self.plan
        self.assertEqual(plan["training"]["steps"], 9000)
        self.assertEqual(plan["training"]["batch_size"], 64)
        self.assertEqual(plan["training"]["optimizer_examples"], 576000)
        self.assertEqual(plan["training"]["fit_lineages"], 2409)
        self.assertEqual(plan["training"]["controller_lineages"], 603)
        self.assertEqual(plan["controller"]["pool_size"], 603)
        self.assertEqual(plan["controller"]["steps_per_round"], 1800)
        self.assertEqual(plan["seeds"], [20260908, 20260909, 20260910])
        self.assertEqual(plan["compute"]["allowance"]["gpu_seconds"], 14400)
        self.assertEqual(plan["evaluation"]["endpoints"], [15, 30, 60, 150, 225, 600])
        self.assertEqual(plan["evaluation"]["task_time"], 600)

    def test_membership_matches_frozen_77_pools_and_states(self):
        plan = self.plan
        self.assertEqual(len(plan["evaluation"]["states_membership"]), 24)
        evaluable = [s for s in plan["evaluation"]["states_membership"]
                     if not s["unsupported"]]
        self.assertEqual(len(evaluable), 21)
        self.assertEqual(sum(1 for s in evaluable if s["corpus"] == runner.N1_CORPUS), 4)
        self.assertEqual(sum(1 for s in evaluable if s["corpus"] == runner.N2_CORPUS), 17)
        self.assertEqual(plan["evaluation"]["candidates"], 241)
        pools = runner.training_pools(train.load_plan(runner.dynamics_args(self.args)))
        self.assertEqual(len(pools["predictor"]["n1_lineages"])
                         + pools["predictor"]["issue71_count"], 2409)
        self.assertEqual(len(pools["controller"]["n1_lineages"])
                         + pools["controller"]["issue71_count"], 603)

    def test_no_placeholder_text_in_freeze(self):
        def walk(value):
            if isinstance(value, str):
                self.assertNotIn("~", value)
                self.assertNotIn("e.g.", value)
                self.assertNotIn("TBD", value)
            elif isinstance(value, dict):
                for key, item in value.items():
                    if key == "source_text":
                        continue  # pinned module code, verified by exact equality
                    walk(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    walk(item)
        walk(self.plan)

    def test_systems_inventory_and_gate(self):
        plan = self.plan
        self.assertEqual(len(plan["systems"]["inventory"]), 7)
        self.assertTrue(plan["systems"]["inventory"]["tawm_adaptive"]["adaptive"])
        self.assertIn("PASS", plan["fidelity_gate"]["tawm"])
        self.assertIn("PASS", plan["fidelity_gate"]["vlwm"])
        self.assertIn("NOT triggered", plan["fidelity_gate"]["thick_fallback"])
        self.assertEqual(len(plan["fidelity_gate"]["vlwm_no_code_search"]), 7)
        self.assertEqual(plan["fidelity_gate"]["tawm_reference"]["commit"],
                         "ffb61f8e2bcdb0030cb4a7175e0b782cdad9af4c")
        self.assertIn("paper-fidelity port, no reference code consulted",
                      plan["fidelity_gate"]["vlwm_reference"]["code"])

    def test_disposition_rules_pre_declared(self):
        rules = self.plan["disposition_rules"]
        self.assertEqual(set(rules["tokens"]),
                         {"supported", "not_supported_by_this_experiment",
                          "readiness_or_precision_insufficient"})
        for question in ("q1_method_class", "q2_work_reported", "q3_training_effect"):
            self.assertIn(question, rules)

    def test_load_plan_detects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=Path(directory), dynamics=runner.DYNAMICS,
                                      issue71=old.ROOT, parser=old.repair.ROOT, device="cpu")
            runner.write(Path(directory) / "plan.json", runner.make_plan(args))
            self.assertEqual(runner.load_plan(args)["identity"], runner.IDENTITY)
            drifted = runner.make_plan(args)
            drifted["training"]["steps"] = 1
            runner.write(Path(directory) / "plan.json", drifted)
            with self.assertRaises(ValueError):
                runner.load_plan(args)

    def test_cli_modes_are_mutually_exclusive(self):
        with patch("sys.argv", ["runner", "--dry-run", "--train"]), \
                self.assertRaises(SystemExit) as ended:
            runner.main()
        self.assertEqual(ended.exception.code, 2)


class RecordAndCurveTests(unittest.TestCase):
    def test_adaptive_curve_lands_on_task_endpoint_with_grid_horizons(self):
        torch.manual_seed(780)
        model = ContinuousDynamics(16).eval()
        control = MatchedController(pure=True).eval()
        context = torch.randn(236)
        action = {"drag_x": 10., "drag_y": -5., "release_time_ms": 1000., "tap_time_ms": 0.}
        curve = runner.adaptive_curve(model, control, context, action)
        self.assertIsNone(curve["failure"])
        self.assertEqual(curve["completed_steps"], 600)
        self.assertIn("600", curve["outputs"])
        self.assertTrue(set(curve["outputs"]) <= {str(t) for t in runner.TIMES})
        self.assertTrue(all(s["horizon"] in (1, 5, 15) for s in curve["segments"]))
        total = sum(s["horizon"] for s in curve["segments"])
        self.assertEqual(total, 600)
        self.assertEqual(curve["controller_calls"], len(curve["segments"]))

    def test_check_record_rejects_drifted_bindings(self):
        plan = {"identity": "x"}
        state = {"state": "s"}
        source = {"identity": "c", "ordinal": 3, "action": {"a": 1}}
        model = ContinuousDynamics(16).eval()
        control = MatchedController(pure=True).eval()
        objective = lambda values: float(values[0])
        record = {"plan_identity": "other", "seed": 1, "system": "tawm_h1", "state": "s",
                  "candidate_identity": "c", "ordinal": 3, "action": {"a": 1},
                  "prediction": {"failure": "x"}, "cost": None, "local": {}}
        with self.assertRaises(ValueError):
            runner.check_record(record, plan, state, source, 1, "tawm_h1", model, control,
                                objective)

    def test_horizon_counts_bindings_match_schedule(self):
        plan = runner.make_plan(argparse.Namespace(
            output=runner.OUTPUT, dynamics=runner.DYNAMICS, issue71=old.ROOT,
            parser=old.repair.ROOT, device="cpu"))
        self.assertEqual(runner.horizon_counts(plan, "tawm"),
                         {"h1": 3000, "h5": 3000, "h15": 3000})
        self.assertEqual(runner.horizon_counts(plan, "vlwm"),
                         {"h1": 5500, "h5": 2500, "h15": 1000})

    def test_carrier_capacity_matches_frozen_reference(self):
        plan77 = train.load_plan(runner.dynamics_args(argparse.Namespace(
            output=runner.DYNAMICS, dynamics=runner.DYNAMICS, issue71=old.ROOT,
            parser=old.repair.ROOT, device="cpu")))
        model = runner.new_model(runner.make_plan(argparse.Namespace(
            output=runner.OUTPUT, dynamics=runner.DYNAMICS, issue71=old.ROOT,
            parser=old.repair.ROOT, device="cpu")))
        self.assertEqual(parameter_count(model), plan77["capacity"]["continuous_parameters"])
        self.assertEqual(parameter_count(MatchedController(pure=True)),
                         plan77["controller"]["capacity"]["continuous"])


class PublicationUnitTests(unittest.TestCase):
    @staticmethod
    def namespace(output=None):
        return argparse.Namespace(output=output or runner.OUTPUT, dynamics=runner.DYNAMICS,
                                  issue71=old.ROOT, parser=old.repair.ROOT, device="cpu")

    def test_interval_status_and_dispositions(self):
        positive = {"descriptive_95_percent_interval": [0.01, 0.2]}
        negative = {"descriptive_95_percent_interval": [-0.2, -0.01]}
        indeterminate = {"descriptive_95_percent_interval": [-0.1, 0.1]}
        contrasts = [
            {"tested": "tawm_h1", "reference": "tawm_h15", "kind": "decay_signature",
             "regret": positive},
            {"tested": "vlwm_h1", "reference": "vlwm_h15", "kind": "decay_signature",
             "regret": negative},
            {"tested": "tawm_h1", "reference": "continuous_h1",
             "kind": "training_effect_analog", "regret": positive},
            {"tested": "vlwm_h1", "reference": "continuous_h1",
             "kind": "training_effect_analog", "regret": negative},
        ]
        result = {"contrasts": contrasts, "frontier_complete": True}
        rules = runner.dispositions(result, True)
        self.assertEqual(rules["q1_method_class"], "not_supported_by_this_experiment")
        self.assertEqual(rules["q2_work_reported"], "supported")
        self.assertEqual(rules["q3_training_effect"], "supported")
        both_negative = runner.dispositions({"contrasts": [
            {**contrasts[0], "regret": negative}, {**contrasts[1], "regret": negative},
            {**contrasts[2], "regret": negative}, {**contrasts[3], "regret": negative}],
            "frontier_complete": True}, True)
        self.assertEqual(both_negative["q1_method_class"], "not_supported_by_this_experiment")
        self.assertEqual(both_negative["q3_training_effect"],
                         "not_supported_by_this_experiment")
        both_positive = runner.dispositions({"contrasts": [
            {**contrasts[0], "regret": positive}, {**contrasts[1], "regret": positive},
            {**contrasts[2], "regret": positive}, {**contrasts[3], "regret": positive}],
            "frontier_complete": True}, True)
        self.assertEqual(both_positive["q1_method_class"], "supported")
        self.assertEqual(both_positive["q3_training_effect"], "supported")
        mixed = runner.dispositions({"contrasts": [
            {**contrasts[0], "regret": positive}, {**contrasts[1], "regret": indeterminate},
            {**contrasts[2], "regret": indeterminate}, {**contrasts[3], "regret": negative}],
            "frontier_complete": True}, True)
        self.assertEqual(mixed["q1_method_class"], "readiness_or_precision_insufficient")
        self.assertEqual(mixed["q3_training_effect"], "readiness_or_precision_insufficient")
        negative_with_indeterminate = runner.dispositions({"contrasts": [
            {**contrasts[0], "regret": negative}, {**contrasts[1], "regret": indeterminate},
            {**contrasts[2], "regret": negative}, {**contrasts[3], "regret": negative}],
            "frontier_complete": True}, True)
        # {negative, indeterminate} is precision failure, not a definitive negative,
        # per the frozen q1 rule text.
        self.assertEqual(negative_with_indeterminate["q1_method_class"],
                         "readiness_or_precision_insufficient")
        self.assertEqual(negative_with_indeterminate["q3_training_effect"],
                         "not_supported_by_this_experiment")
        corpus_leak = runner.dispositions({"contrasts": [
            {**contrasts[0], "regret": positive}, {**contrasts[1], "regret": positive},
            {**contrasts[1], "regret": indeterminate, "corpus": "n2"},
            {**contrasts[2], "regret": positive}, {**contrasts[3], "regret": positive}],
            "frontier_complete": True}, True)
        # Corpus-restricted variants must not leak into the pooled disposition.
        self.assertEqual(corpus_leak["q1_method_class"], "supported")
        incomplete = runner.dispositions(result, False)
        self.assertEqual(set(incomplete.values()),
                         {"readiness_or_precision_insufficient"})

    def test_paired_difference_averages_seeds_within_state(self):
        all_states = {}
        for seed in runner.SEEDS:
            rows = {}
            for state in ("a", "b"):
                value = 0.5 if state == "a" else 0.0
                rows.setdefault("tawm_h1", {})[state] = {"task": {"regret": value}}
                rows.setdefault("continuous_h1", {})[state] = {"task": {"regret": value + 0.25}}
            all_states[str(seed)] = rows
        differences = runner.paired_difference(all_states, "tawm_h1", "continuous_h1")
        self.assertEqual(differences, [0.25, 0.25])
        interval = base.grid.paired_interval(differences)
        self.assertEqual(interval["mean"], 0.25)

    def test_n1_lineage_shards_resolve_outside_the_issue_78_output(self):
        plan77 = train.load_plan(runner.dynamics_args(self.namespace()))
        source = old.load_plan(train.source_args(self.namespace()))
        identity = plan77["n1_lineages"]["predictor"][0]
        # The issue-78 output directory holds no shards; the N1 string lineages must
        # resolve through the frozen dynamics directory, not args.output.
        data = train.load_shard_entry(runner.dynamics_args(self.namespace(
            output=runner.OUTPUT)), plan77, identity, source, True)
        self.assertIn("z", data)

    def test_reference_rows_cover_every_contrast_reference(self):
        args = self.namespace()
        plan = runner.make_plan(args)
        specs = (plan["contrasts"]["q1_method_class"] + plan["contrasts"]["q1_decay_signature"]
                 + plan["contrasts"]["q2_work_reported"]
                 + plan["contrasts"]["q3_training_effect"])
        covered = set(runner.SYSTEMS) | set(runner.REFERENCE_SYSTEMS)
        for spec in specs:
            self.assertIn(spec["tested"], covered)
            self.assertIn(spec["reference"], covered)

    def test_reference_rows_from_frozen_records(self):
        args = self.namespace()
        plan = runner.make_plan(args)
        states = [s for s in plan["evaluation"]["states_membership"] if not s["unsupported"]]
        from world_model.training.matched_dynamics import CONTINUOUS_PAIRS  # noqa: F401
        objective = base.TaskObjective.from_vocabulary(plan["contract"]["vocabulary"])
        targets_by_state = {}
        for state in states:
            value = runner.read(runner.target_path(args, state))
            targets_by_state[state["state"]] = {
                **value, "candidates": [{**t, "realized": objective(torch.tensor(
                    t["carriers"][str(runner.TASK_TIME)]))} for t in value["candidates"]]}
        ref_records = {"continuous_h5": runner.reference_records(args, plan, "continuous_h5")}
        rows = runner.reference_rows(args, plan, targets_by_state, ref_records)
        self.assertEqual(set(rows), {str(s) for s in runner.SEEDS})
        for seed, systems in rows.items():
            state_rows = systems["continuous_h5"]
            self.assertEqual(len(state_rows), 21)
            regrets = [row["task"]["regret"] for row in state_rows.values()]
            self.assertTrue(all(0.0 <= r <= 1.0 for r in regrets))
            self.assertTrue(all(row["work"]["wall_seconds"] > 0
                                for row in state_rows.values()))
            self.assertTrue(all(row["task"]["prediction_failure"] is False
                                for row in state_rows.values()))


if __name__ == "__main__":
    unittest.main()
