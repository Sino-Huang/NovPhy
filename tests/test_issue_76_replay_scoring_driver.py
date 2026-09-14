from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts import run_issue_76_replay_scoring as driver


class DriverTests(unittest.TestCase):
    def test_shared_agent_input_all_pairs_and_failed_group_retained(self):
        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            image = root / "decision.png"
            image.write_bytes(b"agent-rgb")
            anchors = [{"source_member_identity": f"source-{i}", "decision_image_path": str(image),
                        "timestamp": 7. + i, "paired_ranking_admissible": i == 0,
                        "candidate_members": [{"actions": [{"drag_x": -10}]}, {"actions": [{"drag_x": -60}]}]}
                       for i in range(2)]
            cells = [{"seed": 1, "pure": pure} for pure in (False, True)]
            receipt = {"seed": 1}
            plan = {"identity": "test", "inputs": {"seeds": [1], "anchors": anchors},
                    "pool": {"models": cells}, "representation": {"receipts": [receipt]},
                    "threads": 1, "scoring_wall_seconds": 900, "new_scoring_artifact_bytes": 10000,
                    "endpoint_native_steps_from_decision": 11250}
            observed, scored, written = [], [], {}

            class History:
                def reset(self):
                    observed.append("reset")

                def observe(self, rgb, timestamp):
                    observed.append((rgb, timestamp))
                    return torch.tensor([timestamp])

            @contextmanager
            def budget(*args):
                yield SimpleNamespace(check=lambda: None, value={"active_seconds": 1.})

            def score(model, initial, actions, pair, **kwargs):
                scored.append((model.pure, initial.item(), actions, pair))
                return {"candidate_transitions": 2, "candidates": []}

            stack.enter_context(patch.object(driver, "load_plan", return_value=plan))
            read = stack.enter_context(patch.object(driver.replay.run.files, "read", return_value={}))
            stack.enter_context(patch.object(driver, "load_representation", return_value=(History(), receipt)))
            stack.enter_context(patch.object(driver, "agent_image_tensor", side_effect=lambda png: png))
            stack.enter_context(patch.object(driver.fixed, "load_cell", side_effect=lambda pool, cell, device:
                (SimpleNamespace(pure=cell["pure"], pairs=("c",) if cell["pure"] else ("c", "s")), [])))
            stack.enter_context(patch.object(driver, "score_actions", side_effect=score))
            stack.enter_context(patch.object(driver, "linear_macs", return_value=3))
            stack.enter_context(patch.object(driver.fixed.fit, "policy_name", side_effect=lambda pair: pair))
            stack.enter_context(patch.object(driver.fixed.fit, "fitting_budget", budget))
            stack.enter_context(patch.object(driver.torch.cuda, "init"))
            stack.enter_context(patch.object(driver.torch.cuda, "synchronize"))
            stack.enter_context(patch.object(driver.replay, "immutable", side_effect=lambda path, value: written.update({str(path): value})))
            driver.run(root)
            self.assertEqual(observed, ["reset", (b"agent-rgb", 7.), "reset", (b"agent-rgb", 8.)])
            self.assertEqual([(pure, time, pair) for pure, time, _, pair in scored],
                             [(False, 7., "c"), (False, 7., "s"), (False, 8., "c"),
                              (False, 8., "s"), (True, 7., "c"), (True, 8., "c")])
            rows = written[str(root / "fixed-action-scores/model-001.json")]["rows"]
            self.assertEqual([row["paired_ranking_admissible"] for row in rows], [True, True, False, False])
            self.assertEqual(read.call_count, 2)  # representation plan and completion budget only

    def test_attempted_run_does_not_restart(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "budgets").mkdir()
            (root / "budgets/replay-fixed-scoring.json").write_text("{}")
            with patch.object(driver, "load_plan", return_value={}), patch.object(driver.torch.cuda, "init") as initialize:
                with self.assertRaisesRegex(ValueError, "already attempted"):
                    driver.run(root)
                initialize.assert_not_called()
