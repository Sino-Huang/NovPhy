import unittest

from scripts.run_issue_76_parser_learnability import succeeded


class LearnabilityTest(unittest.TestCase):
    def test_count_fit_alone_cannot_hide_wrong_slot_presence(self):
        plan = dict(block_mae_max=.1, task_presence_brier_max=.01)
        self.assertTrue(succeeded(dict(block_count_mae=.1, task_presence_brier=.01), plan))
        self.assertFalse(succeeded(dict(block_count_mae=.01, task_presence_brier=.2), plan))
        self.assertFalse(succeeded(dict(block_count_mae=.2, task_presence_brier=.001), plan))
