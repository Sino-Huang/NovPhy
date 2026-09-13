import unittest

from scripts.publish_issue_76_native_refit import compact


def check_compaction_retains_failures_metrics_and_exact_work():
    step = dict(mode="continuous", horizon_native_steps=50, transition_calls=1,
                controller_calls=1, symbol_decoder_calls=0, linear_macs=17)
    value = {"rows": [
        {"member_identity": "available", "policies": {
            "adaptive": {"failure": None, "carrier_mse": 4., "work": [step, step]},
            "failed": {"failure": "nonfinite", "carrier_mse": 1e9}}},
        {"member_identity": "unavailable", "available": False, "policies": {}}]}
    result = compact(value)
    policy = result["rows"][0]["policies"]["adaptive"]
    assert policy["carrier_mse"] == 4.
    assert policy["work_totals"] == dict(transition_calls=2, controller_calls=2,
                                        symbol_decoder_calls=0, linear_macs=34)
    assert policy["work_histogram"] == [dict(mode="continuous", horizon_native_steps=50, count=2)]
    assert result["rows"][0]["policies"]["failed"] == value["rows"][0]["policies"]["failed"]
    assert result["rows"][1] == value["rows"][1]
    assert "work" in value["rows"][0]["policies"]["adaptive"]


class NativeRefitPublicationTest(unittest.TestCase):
    def test_compaction_retains_failures_metrics_and_exact_work(self):
        check_compaction_retains_failures_metrics_and_exact_work()
