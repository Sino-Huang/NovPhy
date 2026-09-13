import unittest

from scripts.issue_76_native_outcomes import gameplay_outcome


class NativeOutcomeTests(unittest.TestCase):
    def result(self, *, censored=False, failure=None):
        return {"segments": [{"summary": {"censored": censored}}], "failure": failure,
                "complete": failure is None, "gameplay_success": False,
                "game_state_after": "EVALUATION_TERMINATED"}

    def test_later_interface_termination_does_not_erase_a_genuine_level_clear(self):
        clear = {"reason": "level_clear", "fixed_step": 100, "event_id": "clear"}
        original = self.result()
        actual = gameplay_outcome(original, [clear])
        self.assertTrue(actual["success"])
        self.assertEqual(actual["clear_evidence"], clear)
        self.assertFalse(original["gameplay_success"])

    def test_censoring_and_technical_failures_remain_failures(self):
        clear = {"reason": "level_clear"}
        self.assertFalse(gameplay_outcome(self.result(censored=True), [clear])["success"])
        self.assertFalse(gameplay_outcome(self.result(failure="capture_error"), [clear])["success"])

    def test_stability_pig_removal_or_missing_terminal_are_not_gameplay_wins(self):
        for terminal in (None, {"reason": "stable_entered"}, {"reason": "pig_removed"}, {"reason": "level_fail"}):
            self.assertFalse(gameplay_outcome(self.result(), [terminal])["success"])


if __name__ == "__main__":
    unittest.main()
