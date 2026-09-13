from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts import issue_76_live_models as models
from world_model.planning.gameplay import SlingshotActionBounds
from world_model.training.native_history_model import CONTINUOUS_PAIRS, STATE_DIM


class LiveModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(137)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.fit = models.fit
        self.plan = dict(identity="loader-fixture", seeds=[7], synthetic=False,
                         capacity=dict(continuous_width=16, hybrid_width=16))
        self.bounds = SlingshotActionBounds((-160, -40), (-80, 80), (0, 0), 600)
        parser = self.fit.NativeVisualParser()
        history = self.fit.ObservedHistoryEncoder(carrier_dim=self.fit.VISUAL_DIM)
        self.fit.atomic_torch(self.fit.common_path(self.root, 7), dict(
            plan_identity=self.plan["identity"], seed=7, synthetic=False, auxiliary_deployed=False,
            parser=parser.state_dict(), history=history.state_dict()))
        for pure, name in ((False, "hybrid"), (True, "pure")):
            model = self.fit.new_predictor(self.plan, pure, "cpu")
            self.fit.atomic_torch(self.fit.model_path(self.root, 7, pure, "predictor"), dict(
                plan_identity=self.plan["identity"], complete=True, model=model.state_dict()))
            controller = self.fit.NativeHistoryController(pure)
            self.fit.atomic_torch(self.fit.model_path(self.root, 7, pure, "controller"), dict(
                plan_identity=self.plan["identity"], complete=True, model=controller.state_dict()))
            for key in ("common-7", f"predictor-{name}-7", f"controller-{name}-7"):
                self.fit.files.write(self.root / "budgets" / (key + ".json"), dict(
                    stopped=False, running=False, active_seconds=1., limit_seconds=86400.))

    def load(self, pure, **kwargs):
        return models.load_policy(self.root, self.plan, 7, pure, self.bounds,
            maximum_seconds=30., maximum_linear_macs=10**12, device="cpu", **kwargs)

    def test_shared_weights_independent_arms_and_release_semantics_reach_live_selector(self):
        hybrid, hybrid_binding = self.load(False, fixed_pair=CONTINUOUS_PAIRS[-1])
        pure, pure_binding = self.load(True, fixed_pair=CONTINUOUS_PAIRS[-1])
        self.assertEqual(hybrid_binding["common_checkpoint"], pure_binding["common_checkpoint"])
        self.assertNotEqual(hybrid_binding["predictor_checkpoint"], pure_binding["predictor_checkpoint"])
        for a, b in zip(hybrid.history.parser.parameters(), pure.history.parser.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        image = torch.randint(0, 256, (3, 64, 96), dtype=torch.uint8)
        for policy in (hybrid, pure):
            policy.history.observe(image, 0.)
            record = policy.choose()
            self.assertEqual(record["action"]["release_time_ms"], 600)
            self.assertEqual(record["action"]["tap_time_ms"], 0)
            self.assertEqual(len(record["candidates"]), 12)
            self.assertFalse(record["matched_compute_established"])
            self.assertEqual(policy.history.executed_actions, 0)

    def test_common_work_is_charged_once_per_decision_and_resets_for_new_episode(self):
        policy, _ = self.load(True)
        policy.history.current = torch.zeros(STATE_DIM)
        def score(*args, **kwargs):
            self.assertIsNotNone(args[1])  # The completed independent controller is loaded.
            return dict(action=dict(drag_x=-80, drag_y=0, tap_time_ms=0), wall_seconds=.1)
        with patch.object(models, "score_native_actions", side_effect=score):
            policy.perception_seconds = .3
            self.assertAlmostEqual(policy.choose()["common_rgb_history_seconds"], .3)
            policy.perception_seconds = .5
            self.assertAlmostEqual(policy.choose()["common_rgb_history_seconds"], .2)
            policy.reset()
            policy.history.current = torch.zeros(STATE_DIM)
            policy.perception_seconds = .8
            self.assertAlmostEqual(policy.choose()["common_rgb_history_seconds"], .8)
            policy.perception_seconds = 40.
            with self.assertRaisesRegex(ValueError, "common RGB/history"):
                policy.choose()

    def test_running_checkpoint_budget_cannot_be_used_as_a_completed_policy(self):
        self.fit.files.write(self.root / "budgets/common-7.json", dict(
            stopped=False, running=True, active_seconds=1., limit_seconds=86400.))
        with self.assertRaisesRegex(ValueError, "unfinished"):
            self.load(True)

    def test_synthetic_plan_cannot_be_loaded_as_research_deployment(self):
        self.plan["synthetic"] = True
        with self.assertRaisesRegex(ValueError, "non-synthetic"):
            self.load(True)


if __name__ == "__main__":
    unittest.main()
