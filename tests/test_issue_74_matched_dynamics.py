import argparse
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts import run_issue_74_matched_dynamics as runner
from tests.test_issue_71_hybrid_readiness import batch, tiny_model
from world_model.model import Abstraction, PredictionPair
from world_model.training.cnn_hybrid import DIM, PAIRS, training_loss
from world_model.training.matched_dynamics import (
    ContinuousDynamics, CONTINUOUS_PAIRS, MatchedController, capacity_contract,
    continuous_loss, controller_labels, parameter_count, rollout, work, active_capacity,
)


class MatchedDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cli_help_resolves_defaults_without_data_access(self):
        with patch("sys.argv", ["runner", "--help"]), self.assertRaises(SystemExit) as ended:
            runner.main()
        self.assertEqual(ended.exception.code, 0)

    def test_pure_has_no_symbolic_or_dead_mode_modules(self):
        model = ContinuousDynamics(16)
        self.assertFalse(any(any(s in name for s in ("micro", "macro", "adapter", "abstraction"))
                             for name, _ in model.named_parameters()))
        data = batch()
        with self.assertRaises(ValueError):
            model.carrier(data["z"][:, 0], data["action"], PAIRS[1])
        with self.assertRaises(ValueError):
            model.carrier(torch.zeros(2, 197), data["action"], PAIRS[0])
        with self.assertRaises(RuntimeError):
            model.load_state_dict(tiny_model().state_dict(), strict=True)

    def test_runtime_and_pure_loss_have_no_future_symbolic_input(self):
        self.assertEqual(list(inspect.signature(ContinuousDynamics.carrier).parameters), ["self", "z", "action", "pair"])
        self.assertEqual(list(inspect.signature(continuous_loss).parameters), ["model", "z", "action", "length", "pair"])
        self.assertEqual(list(inspect.signature(MatchedController.forward).parameters), ["self", "z", "action", "remaining"])
        self.assertEqual(list(inspect.signature(runner.predict).parameters), ["model", "controller", "context", "action", "fixed_pair", "objective"])

    def test_pure_loss_does_not_consume_symbol_labels(self):
        model = ContinuousDynamics(16); data = batch()
        first = runner.loss_for(model, data, PAIRS[0])
        for key in ("relations", "relations_mask", "macros", "macros_mask"):
            data[key] = object()
        self.assertTrue(torch.equal(first, runner.loss_for(model, data, PAIRS[0])))

    def test_continuous_recipe_matches_hybrid_continuous_path(self):
        model = tiny_model(); data = batch()
        for pair in CONTINUOUS_PAIRS:
            self.assertTrue(torch.equal(continuous_loss(model, data["z"], data["action"], data["length"], pair),
                                        training_loss(model, data, pair, True)))

    def test_capacity_rule_and_controller_no_padding(self):
        contract = capacity_contract()
        self.assertLess(contract["relative_difference"], .02)
        self.assertEqual(contract["padding_parameters"], 0)
        a, b = (parameter_count(MatchedController(p)) for p in (True, False))
        self.assertLess(abs(a-b)/b, .002)

    def test_active_capacity_reflects_selected_real_modules(self):
        data = batch(1); pure = ContinuousDynamics(16); hybrid = tiny_model()
        self.assertEqual(active_capacity(pure, data["z"][:, 0], data["action"], PAIRS[0]), parameter_count(pure))
        continuous = active_capacity(hybrid, data["z"][:, 0], data["action"], PAIRS[0])
        micro = active_capacity(hybrid, data["z"][:, 0], data["action"], PAIRS[1])
        self.assertGreater(micro, continuous)
        self.assertLess(micro, parameter_count(hybrid))

    def test_selection_uses_all_seeds_and_disqualifies_failures(self):
        rows = {str(seed): {"policies": {p: {"prediction_failure_states": 0, "mean_regret": .4}
                                            for p in runner.POLICIES[:4]}} for seed in runner.SEEDS}
        for row in rows.values(): row["policies"]["continuous_h5"]["mean_regret"] = .3
        rows[str(runner.SEEDS[0])]["policies"]["continuous_h1"]["mean_regret"] = .2
        self.assertEqual(runner.select_continuous(rows), "continuous_h5")
        rows[str(runner.SEEDS[0])]["policies"]["continuous_h5"]["prediction_failure_states"] = 1
        self.assertEqual(runner.select_continuous(rows), "continuous_h1")

    def test_pure_parameters_all_receive_task_gradients_and_updates(self):
        model = ContinuousDynamics(16); data = batch()
        initial = {n: p.detach().clone() for n,p in model.named_parameters()}
        optim = torch.optim.AdamW(model.parameters(), lr=.001); seen = set()
        for _ in range(2):
            for pair in CONTINUOUS_PAIRS:
                optim.zero_grad(); runner.loss_for(model, data, pair).backward()
                seen.update(n for n,p in model.named_parameters() if p.grad is not None and bool((p.grad != 0).any()))
                optim.step()
        self.assertEqual(seen, set(initial))
        self.assertTrue(all(not torch.equal(initial[n], p) for n,p in model.named_parameters()))

    def test_all_horizons_common_endpoint_and_controller_mask(self):
        model = ContinuousDynamics(16); control = MatchedController(True); data = batch(1)
        for pair in CONTINUOUS_PAIRS:
            _, trace = rollout(model, control, data["z"][:, 0], data["action"], 30, pair)
            self.assertEqual(sum(t["horizon"] for t in trace), 30)
            self.assertTrue(all(t["symbol_decoder_calls"] == 0 for t in trace))
            self.assertEqual(trace[0]["linear_macs"], work(model, pair))
        with torch.no_grad():
            for p in control.parameters(): p.zero_()
            control.layers[-1].bias[-1] = 10
        _, trace = rollout(model, control, data["z"][:, 0], data["action"], 17)
        self.assertEqual([t["horizon"] for t in trace], [15, 1, 1])
        with self.assertRaises(ValueError):
            rollout(model, MatchedController(False), data["z"][:, 0], data["action"], 17)

    def test_controller_dp_only_permitted_continuous_horizons(self):
        data = batch(1); model = ContinuousDynamics(16); control = MatchedController(True)
        labels = controller_labels(model, control, data["z"], data["action"], data["length"])
        self.assertEqual(labels.shape, (1, 60))
        self.assertLess(int(labels.max()), 3)
        for t, index in enumerate(labels[0]):
            self.assertLessEqual(CONTINUOUS_PAIRS[int(index)].delta, 60-t)

    def test_exact_optimizer_and_sampler_resume(self):
        data = {k:v for k,v in batch().items() if k in ("z", "action", "length")}
        plan = {"identity": "test", "contract": {}, "capacity": {"continuous_width": 16},
                "source_release_identity": "test", "controller": {},
                "training": {"steps": 18, "batch_size": 2, "learning_rate": .0001, "weight_decay": .0001, "grad_clip": 1.}}
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "shard", return_value=data):
            args = argparse.Namespace(output=Path(directory)/"resume", device="cpu")
            runner.train_cell(args, plan, {}, 74, "continuous", stop_after=9)
            runner.train_cell(args, plan, {}, 74, "continuous")
            resumed, _ = runner.load_predictor(args, plan, 74, "continuous")
            args.output = Path(directory)/"full"
            runner.train_cell(args, plan, {}, 74, "continuous")
            full, saved = runner.load_predictor(args, plan, 74, "continuous")
            self.assertTrue(all(torch.equal(v, full.state_dict()[k]) for k,v in resumed.state_dict().items()))
            changed = dict(saved); changed["binding"] = dict(saved["binding"], arm="hybrid")
            with self.assertRaises(ValueError): runner.check_binding(changed, saved["binding"])

    def test_controller_interruption_resume_matches_uninterrupted(self):
        data = batch(1)
        labels = {"z": data["z"][0, :60], "action": data["action"].expand(60, -1),
                  "remaining": torch.arange(60, 0, -1), "labels": torch.zeros(60, dtype=torch.long)}
        plan = {"identity": "test", "contract": {}, "capacity": {}, "source_release_identity": "test", "training": {},
                "controller": {"steps_per_round": 120, "batch_size": 4, "learning_rate": .001}}
        source = {"records": [{} for _ in range(5)]}
        atomic = runner.old.repair.atomic_torch
        def interrupt(path, value):
            atomic(path, value)
            if path.name == "controller-progress-0.pt" and value["step"] == 60:
                raise RuntimeError("simulated interruption")
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runner, "load_predictor", return_value=(ContinuousDynamics(16), {})), \
             patch.object(runner, "label_lineage", return_value=labels):
            args = argparse.Namespace(output=Path(directory)/"resume", device="cpu")
            with patch.object(runner.old.repair, "atomic_torch", side_effect=interrupt), self.assertRaisesRegex(RuntimeError, "simulated"):
                runner.train_control(args, plan, source, 74, "continuous", indices=(5,))
            runner.train_control(args, plan, source, 74, "continuous", indices=(5,))
            resumed, _ = runner.load_control(args, plan, 74, "continuous")
            args.output = Path(directory)/"full"
            runner.train_control(args, plan, source, 74, "continuous", indices=(5,))
            full, _ = runner.load_control(args, plan, 74, "continuous")
            self.assertTrue(all(torch.equal(v, full.state_dict()[k]) for k,v in resumed.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
