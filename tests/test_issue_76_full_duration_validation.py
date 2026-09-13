from copy import deepcopy
import unittest

import torch

from scripts.validate_issue_76_full_duration_dynamics import check_optimizer_steps
from scripts.run_issue_76_native_refit import pair_for_update
from world_model.training.native_history_model import NativeHistoryDynamics, STATE_DIM


class OptimizerHistoryValidationTests(unittest.TestCase):
    def test_real_mode_specific_steps_are_retained_and_reset_is_rejected(self):
        torch.set_num_threads(1)
        torch.manual_seed(76)
        for pure in (False, True):
            model = NativeHistoryDynamics(pure=pure, width=16)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
            z, action = torch.randn(2, STATE_DIM), torch.randn(2, 5)

            def update(pair):
                optimizer.zero_grad(set_to_none=True)
                model.carrier(z, action, pair).square().mean().backward()
                optimizer.step()

            for pair in model.pairs:
                update(pair)
            before = deepcopy(optimizer.state_dict())
            plan = dict(start_update=0, updates={"predictor": 5}, training=[{"usable": True}, {"usable": False}])
            for u in (0, 2, 4):
                update(pair_for_update(pure, u))
            after = deepcopy(optimizer.state_dict())
            rows = check_optimizer_steps(model, before, after, plan, pure)
            self.assertTrue(all(r["stage_steps"] == 3 for r in rows if r["parameter"].startswith("input_projection.")))
            if not pure:
                self.assertTrue(all(r["stage_steps"] == 1 for r in rows if r["parameter"].startswith("micro_head.")))
                self.assertTrue(all(r["stage_steps"] == 0 for r in rows if r["parameter"].startswith("macro_head.")))
            first = next(iter(after["state"]))
            after["state"][first]["step"] = torch.tensor(0.)
            with self.assertRaisesRegex(ValueError, "optimizer history differs"):
                check_optimizer_steps(model, before, after, plan, pure)


if __name__ == "__main__":
    unittest.main()
