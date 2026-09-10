"""Readiness contracts; synthetic tests never count as trained evidence."""
import inspect
import argparse
import builtins
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts import run_issue_71_hybrid_readiness as runner
from world_model.model import Abstraction, PredictorConfig
from world_model.training.cnn_hybrid import (
    DIM, SLOTS, PAIRS, CNNHybridPredictor, CarrierPairController,
    adaptive_rollout, checkpoint_contract, intervention_audit, linear_macs,
    training_loss, trajectory_labels,
)


def batch(size=2):
    torch.manual_seed(71)
    z = torch.rand(size, 61, DIM)
    relations = torch.zeros(size, 61, SLOTS, SLOTS, 2)
    relations[:, :, 1, 3] = 1
    mask = (~torch.eye(SLOTS, dtype=torch.bool))[None, None, :, :, None].expand_as(relations)
    return {"z": z, "action": torch.rand(size, 5), "length": torch.full((size,), 60),
            "relations": relations, "relations_mask": mask,
            "macros": torch.randint(2, (size, 61, 2)), "macros_mask": torch.ones(size, 61, 2, dtype=torch.bool)}


def tiny_model():
    return CNNHybridPredictor(PredictorConfig(latent_dim=DIM, hidden_dim=16, depth=1, pair_code_dim=8))


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_deployed_parameters_get_task_gradients(self):
        model = tiny_model()
        optim = torch.optim.AdamW(model.parameters(), lr=.001)
        seen = set()
        data = batch()
        for _ in range(2):
            for pair in PAIRS:
                optim.zero_grad(set_to_none=True)
                training_loss(model, data, pair, True).backward()
                seen.update(n for n, p in model.named_parameters() if p.grad is not None and bool((p.grad != 0).any()))
                optim.step()
        self.assertEqual(seen, set(dict(model.named_parameters())))
        self.assertTrue(intervention_audit(model, data["z"][:, 0], data["action"])["passed"])

    def test_exactly_one_adapter_and_decoder_runs(self):
        model = tiny_model(); data = batch()
        with patch.object(model.micro_adapter, "forward", wraps=model.micro_adapter.forward) as micro, \
             patch.object(model.macro_adapter, "forward", wraps=model.macro_adapter.forward) as macro:
            model.carrier(data["z"][:, 0], data["action"], PAIRS[6])
            self.assertEqual((micro.call_count, macro.call_count), (0, 0))
            model.carrier(data["z"][:, 0], data["action"], PAIRS[7])
            self.assertEqual((micro.call_count, macro.call_count), (1, 0))
            model.carrier(data["z"][:, 0], data["action"], PAIRS[8])
            self.assertEqual((micro.call_count, macro.call_count), (1, 1))

    def test_contact_symmetry_and_support_direction(self):
        model = tiny_model(); data = batch()
        relations, _ = model.symbols(data["z"][:, 0], Abstraction.MICRO)
        self.assertTrue(torch.equal(relations[..., 0], relations[..., 0].transpose(1, 2)))
        self.assertFalse(torch.equal(relations[..., 1], relations[..., 1].transpose(1, 2)))

    def test_symbol_content_really_changes_transition(self):
        model = tiny_model(); data = batch()
        first = model.carrier(data["z"][:, 0], data["action"], PAIRS[7])
        with patch.object(model, "symbolic_features", return_value=torch.zeros(2, SLOTS*SLOTS*2+2)):
            second = model.carrier(data["z"][:, 0], data["action"], PAIRS[7])
        self.assertFalse(torch.equal(first, second))

    def test_continuous_path_does_not_execute_heads(self):
        model = tiny_model(); data = batch()
        with patch.object(model, "symbols", side_effect=AssertionError("unused head")):
            model.carrier(data["z"][:, 0], data["action"], PAIRS[6])

    def test_readout_masks_do_not_supervise_unavailable_labels(self):
        model = tiny_model(); data = batch()
        data["relations_mask"].zero_()
        first = training_loss(model, data, PAIRS[7], False)
        data["relations"] = 1 - data["relations"]
        self.assertTrue(torch.equal(first, training_loss(model, data, PAIRS[7], False)))

    def test_historical_dimensions_rejected(self):
        with self.assertRaisesRegex(ValueError, "236"):
            CNNHybridPredictor(PredictorConfig(latent_dim=197))
        with self.assertRaises(ValueError):
            checkpoint_contract(["slot"] * 18, "p", "c")

    def test_legacy_checkpoint_rejected(self):
        model = tiny_model()
        with self.assertRaises(RuntimeError):
            model.load_state_dict({"input_projection.weight": torch.zeros(16, 202)}, strict=True)

    def test_runtime_has_no_engine_or_future_inputs(self):
        self.assertEqual(list(inspect.signature(CNNHybridPredictor.carrier).parameters), ["self", "z", "action", "pair"])
        self.assertEqual(list(inspect.signature(CarrierPairController.forward).parameters), ["self", "z", "action", "remaining"])

    def test_recursive_carrier_replaces_prior_not_initial_symbols(self):
        model = tiny_model(); data = batch(1)
        observed = []
        original = model.symbolic_features
        def record(z, mode):
            observed.append(z.clone())
            return original(z, mode)
        with patch.object(model, "symbolic_features", side_effect=record):
            _, trace = adaptive_rollout(model, None, data["z"][:, 0], data["action"], 30, PAIRS[7])
        expected = model.carrier(data["z"][:, 0], data["action"], PAIRS[7])
        self.assertTrue(torch.equal(observed[1], expected))
        self.assertEqual(sum(t["effective_horizon"] for t in trace), 30)
        self.assertEqual(sum(t["controller_calls"] for t in trace), 0)

    def test_controller_masks_unreachable_horizon_and_uses_same_endpoint(self):
        model = tiny_model(); controller = CarrierPairController(); data = batch(1)
        with torch.no_grad():
            for p in controller.parameters(): p.zero_()
            controller.layers[-1].bias[8] = 10
        _, trace = adaptive_rollout(model, controller, data["z"][:, 0], data["action"], 17)
        self.assertEqual([t["requested_horizon"] for t in trace], [15, 1, 1])
        self.assertEqual(sum(t["effective_horizon"] for t in trace), 17)
        self.assertEqual(sum(t["controller_calls"] for t in trace), 3)
        self.assertEqual(trace[0]["linear_macs"], linear_macs(model, PAIRS[8], True))
        with self.assertRaises(ValueError):
            adaptive_rollout(model, controller, data["z"][:, 0], data["action"], 17, PAIRS[6])

    def test_dp_and_aggregation_use_future_only_for_training_labels(self):
        model = tiny_model(); data = batch(1)
        labels = trajectory_labels(model, data)
        self.assertEqual(labels.shape, (1, 60))
        self.assertTrue(torch.equal(labels, trajectory_labels(model, data, contexts=data["z"].clone())))
        horizons = torch.tensor([p.delta for p in PAIRS])[labels[0]]
        self.assertTrue(bool((horizons <= torch.arange(60, 0, -1)).all()))

    def test_nonfinite_rollout_fails_not_successful_readiness(self):
        model = tiny_model(); data = batch(1)
        with patch.object(model, "carrier", return_value=torch.full((1, DIM), float("nan"))):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                adaptive_rollout(model, None, data["z"][:, 0], data["action"], 15, PAIRS[6])

    def test_no_write_dry_run(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "absent"
            with patch.object(runner, "make_plan", return_value={"schema": "smoke"}):
                self.assertEqual(runner.main(["--dry-run", "--output", str(output)]), 0)
            self.assertFalse(output.exists())

    def test_source_snapshot_change_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            runner.write(path / "plan.json", {"schema": runner.SCHEMA, "source": str(path), "contract": {"architecture": runner.ARCHITECTURE},
                                              "code": {"source_text": {runner.SOURCE_FILES[0]: "stale"}}})
            with self.assertRaisesRegex(ValueError, "implementation changed"):
                runner.load_plan(type("Args", (), {"output": path, "source": path})())

    def test_every_fitting_lineage_receives_all_nine_pairs(self):
        observed = {}
        for step in range(9000):
            for lineage in runner.fitting_lineage_group(step):
                observed.setdefault(lineage, set()).add(step % 9)
        self.assertEqual(set(observed), {i for i in range(1, 3001) if i % 5})
        self.assertTrue(all(pairs == set(range(9)) for pairs in observed.values()))

    def test_completed_progress_can_resume_publication_without_retraining(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            args = argparse.Namespace(output=path, device="cpu")
            recipe = {"seed": 71, "steps": 9, "batch_size": 2, "learning_rate": .001,
                      "weight_decay": .0001, "grad_clip": 1., "arms": ["corrected"]}
            plan = {"training": recipe, "contract": {"test_fixture": True}, "source_release_identity": "training-fixture"}
            runner.write(path / "data.json", {"lineages": 3000})
            with patch.object(runner, "load_plan", return_value=plan), \
                 patch.object(runner, "load_shard", return_value=batch()), \
                 patch.object(runner, "CNNHybridPredictor", side_effect=tiny_model):
                runner.train_models(args)
                first = torch.load(path / "corrected.pt", weights_only=True)
                (path / "corrected.pt").unlink()
                with patch.object(runner, "training_loss", side_effect=AssertionError("must reuse completed progress")):
                    runner.train_models(args)
                second = torch.load(path / "corrected.pt", weights_only=True)
                self.assertTrue(all(torch.equal(first["model"][k], second["model"][k]) for k in first["model"]))
                self.assertEqual(second["step"], 9)

    def test_short_terminal_targets_are_not_repeated_as_full_horizon(self):
        model = tiny_model(); data = batch()
        data["length"].fill_(15)
        first = training_loss(model, data, PAIRS[6], True)
        data["z"][:, 16:] = 9999
        second = training_loss(model, data, PAIRS[6], True)
        self.assertTrue(torch.equal(first, second))

    def test_controller_label_aggregation_training_and_reload_workflow(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder); args = argparse.Namespace(output=path, device="cpu")
            data = batch(1)
            plan = {"contract": {"test_fixture": True}, "records": [{"exposure_role": "training"}] * 5,
                    "controller": {"seed": 71, "steps": 1, "batch_size": 16, "learning_rate": .001,
                                   "compute_weight": .0001}}
            runner.repair.atomic_torch(runner.shard_path(args, 5), {"windows": [{"shot": 0, "start": 0}]})
            def bounded_range(*values):
                return (5,) if values == (5, 3001, 5) else builtins.range(*values)
            with patch.object(runner, "load_plan", return_value=plan), \
                 patch.object(runner, "load_model", return_value=(tiny_model(), {"identity": "trained-fixture"})), \
                 patch.object(runner, "load_shard", return_value=data), \
                 patch.object(runner, "range", side_effect=bounded_range, create=True):
                runner.train_controller(args)
                controller, saved = runner.load_controller(args, plan, "trained-fixture")
                self.assertEqual(saved["round"], 1)
                self.assertEqual(set(saved["gradient_parameters"]), set(dict(controller.named_parameters())))
                self.assertTrue((path / "controller-labels-0/lineage-0005.pt").exists())
                self.assertTrue((path / "controller-labels-1/lineage-0005.pt").exists())
                runner.train_controller(args)  # completed round/cache reuse


if __name__ == "__main__":
    unittest.main()
