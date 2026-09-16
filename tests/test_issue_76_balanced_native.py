from pathlib import Path
import tempfile
import unittest

import torch

from scripts.issue_76_fit_budget import fitting_budget
from scripts.issue_76_scaled_fit import backward_clipped, fit_updates
from scripts.run_issue_76_balanced_native import (
    PREDICTOR_BACKWARD_SCALE,
    controller_targets,
    training_phase,
)
from world_model.training.native_history_model import NativeHistoryDynamics, STATE_DIM


class BalancedNativeTests(unittest.TestCase):
    def test_training_phase_is_fixed_at_six_thousand_updates(self):
        self.assertEqual(training_phase(0), "local")
        self.assertEqual(training_phase(5999), "local")
        self.assertEqual(training_phase(6000), "full_duration")
        self.assertEqual(training_phase(11999), "full_duration")
        with self.assertRaises(ValueError):
            training_phase(12000)

    def test_low_cost_controller_teacher_has_exact_paths(self):
        torch.manual_seed(3)
        model = NativeHistoryDynamics(pure=False, width=16).eval()
        carrier = torch.randn(5, STATE_DIM)
        value = controller_targets(model, carrier, [0, 50, 100, 150, 200],
                                   torch.zeros(5), compute_weight=0.0001)
        self.assertEqual(value["z"].shape, (4, STATE_DIM))
        self.assertEqual(value["labels"].shape, (4,))
        self.assertTrue(bool(torch.isfinite(value["remaining"]).all()))

    def test_downscaled_backward_preserves_clipping_and_prevents_overflow(self):
        expected = torch.nn.Parameter(torch.tensor([0.5, -0.25]))
        expected.square().sum().backward()
        torch.nn.utils.clip_grad_norm_([expected], 1.0)

        actual = torch.nn.Parameter(torch.tensor([0.5, -0.25]))
        backward_clipped(actual.square().sum(), [actual],
                         backward_scale=PREDICTOR_BACKWARD_SCALE)
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)

        tiny = torch.nn.Parameter(torch.tensor([1e-40]))
        loss = tiny.log().sum()
        self.assertTrue(bool(torch.isfinite(loss)))
        backward_clipped(loss, [tiny], backward_scale=PREDICTOR_BACKWARD_SCALE)
        self.assertTrue(bool(torch.isfinite(tiny.grad).all()))
        torch.testing.assert_close(tiny.grad, torch.ones_like(tiny), rtol=1e-6, atol=0)

    def test_scaled_fit_checkpoints_finite_state_after_overflowing_raw_gradient(self):
        class LogModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor([1e-40]))

        model = LogModel()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with fitting_budget(root, "scaled-overflow", 30, "cpu") as budget:
                result = fit_updates(root / "checkpoint.pt", model,
                    plan_identity="fixture", updates=1, lr=.0003,
                    make_loss=lambda _: model.value.log().sum(), budget=budget,
                    device="cpu", backward_scale=PREDICTOR_BACKWARD_SCALE)
        self.assertTrue(result["complete"])
        self.assertEqual(result["backward_scale"], PREDICTOR_BACKWARD_SCALE)
        self.assertTrue(bool(torch.isfinite(model.value).all()))

    def test_finite_large_gradients_do_not_overflow_norm_reduction(self):
        value = torch.nn.Parameter(torch.full((2,), 1e-30))
        loss = value.log().sum()
        self.assertTrue(bool(torch.isfinite(loss)))
        backward_clipped(loss, [value], backward_scale=PREDICTOR_BACKWARD_SCALE)
        self.assertTrue(bool(torch.isfinite(value.grad).all()))
        norm = torch.linalg.vector_norm(value.grad, dtype=torch.float64)
        torch.testing.assert_close(norm, torch.tensor(1.0, dtype=torch.float64),
                                   rtol=1e-6, atol=0)

    def test_still_nonfinite_gradient_fails_before_optimizer_corruption(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "checkpoint.pt"
            with fitting_budget(root, "singular-gradient", 30, "cpu") as budget:
                with self.assertRaisesRegex(RuntimeError, "non-finite"):
                    fit_updates(path, model, plan_identity="fixture", updates=1,
                        lr=.0003, make_loss=lambda _: model.weight.sqrt().sum(),
                        budget=budget, device="cpu",
                        backward_scale=PREDICTOR_BACKWARD_SCALE)
            saved = torch.load(path, map_location="cpu", weights_only=False)
        self.assertFalse(saved["complete"])
        self.assertIsNotNone(saved["failure"])
        self.assertEqual(saved["updates_completed"], 0)
        self.assertTrue(bool(torch.isfinite(model.weight).all()))
        self.assertFalse(saved["optimizer"]["state"])


if __name__ == "__main__":
    unittest.main()
