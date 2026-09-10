import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_issue_76_closeout as r


def fixture():
    states = ("state-a","state-b")
    headroom = [{"state":s,"family":f"family-{i}","prior_ordinal09_regret":float(1-i),
                 "uniform_expected_regret":.5,"informative":True,"opportunities":{"pig_removed":False,"level_clear":False}}
                for i,s in enumerate(states)]
    per_state = {str(seed):{name:{s:{"task":{"regret":.5}} for s in states} for name in r.SYSTEMS} for seed in r.diagnostic.SEEDS}
    summaries = {str(seed):{name:{"mean_regret":.5,"top1_fraction":0.,"mean_realized_count_cost":1001.,
                                "selected_outcome_counts":{"pig_removed":0,"level_clear":0},
                                "prediction_failure_states":0,"predicted_all_tied_states":0,
                                "transition_linear_macs":10,"local_probe_linear_macs":5,
                                "mean_instrumented_model_seconds_per_state":.01}
                           for name in r.SYSTEMS} for seed in r.diagnostic.SEEDS}
    previous_policies = (*r.diagnostic.base.POLICIES[:4],"covered_continuous")
    means = {p:.25 if p == "continuous_h5" else .5 for p in previous_policies}
    previous = {"plan_identity":"p75","disposition":"not_supported_by_this_pilot","checks":{"passed":False},
                "means":means,"selected_continuous_policy":"continuous_h5",
                "per_seed":{str(s):{p:{"mean_regret":v,"prediction_failure_states":0} for p,v in means.items()}
                            for s in r.diagnostic.SEEDS}}
    # #76's best observed baseline is deliberately different from the inherited #75 selection.
    for seed in per_state:
        for state in states: per_state[seed]["continuous_h15"][state]["task"]["regret"] = 0.
    plan = {"identity":"p76","source75_plan_identity":"p75","source75_disposition":previous["disposition"],
            "source75_checks":previous["checks"],"definitions":{"bootstrap":{"draws":10000,"seed":7201}}}
    report = {"diagnostics_complete":True,"disposition":"readiness_or_precision_insufficient",
              "fresh_evaluation_opened":False,"final_evaluation_opened":False,"issue_64_authorized":False,
              "headroom":headroom,"per_state":per_state,"per_seed":summaries,
              "contrasts":[{"kind":"training_effect","tested":"hybrid_continuous_h5","reference":"continuous_h5",
                            "regret":{"mean":0.,"descriptive_95_percent_interval":[-.1,.1]},
                            "carrier_mse":{"225":{"mean":.2,"descriptive_95_percent_interval":[.1,.3]},"15":{"paired_states":0}}}],
              "timing":[{"candidates":[{"events":{"bird_launched":[150]},"terminal_reason":"stable_entered","last_offset":217}]}],
              "budget":{"active_seconds":1.,"artifact_bytes":100,"peak_cpu_rss_mib":10.,"peak_cuda_allocated_mib":1.,"stopped":False},
              "source_training_cost":{},"target_parsing_seconds":.2,"novelty_inventory_summary":{},"wider_recommendation":{},
              "requirement_ledger":[],"disposition_scope":"pre-access stop","preaccess_reasons":["#75 negative"]}
    prior = [{"state":s,"realized_count_cost":1001.,"outcomes":{"pig_removed":False,"level_clear":False}} for s in states]
    return plan,report,previous,prior


class CloseoutTests(unittest.TestCase):
    def test_comparison_clusters_seed_repeats_within_state(self):
        _,report,_,_ = fixture()
        for seed,regrets in zip(r.diagnostic.SEEDS,((.2,.1),(.4,.3),(.6,.5)),strict=True):
            for state,regret in zip(("state-a","state-b"),regrets,strict=True):
                report["per_state"][str(seed)]["hybrid_adaptive"][state]["task"]["regret"] = regret
        result = r.paired_comparison(report,"hybrid_adaptive","prior_ordinal09")
        expected = r.diagnostic.base.grid.paired_interval([.6,-.3])
        self.assertEqual(result["paired_states"],2)
        self.assertAlmostEqual(result["regret"]["mean"],expected["mean"])
        np.testing.assert_allclose(result["regret"]["descriptive_95_percent_interval"],expected["descriptive_95_percent_interval"])
        self.assertEqual(len(result["per_seed"]),3)
        self.assertTrue(all(f["paired_states"] == 1 for f in result["per_family"].values()))
        report["headroom"].reverse()
        self.assertEqual(result,r.paired_comparison(report,"hybrid_adaptive","prior_ordinal09"))

    def test_missing_pair_is_rejected_not_silently_dropped(self):
        _,report,_,_ = fixture()
        del report["per_state"][str(r.diagnostic.SEEDS[1])]["hybrid_adaptive"]["state-a"]
        with self.assertRaisesRegex(ValueError,"membership"):
            r.paired_comparison(report,"hybrid_adaptive","prior_ordinal09")

    def test_failed_prediction_penalty_stays_in_analysis(self):
        _,report,_,_ = fixture()
        for seed in r.diagnostic.SEEDS:
            report["per_state"][str(seed)]["hybrid_adaptive"]["state-a"]["task"] = {"regret":1.,"prediction_failure":True}
        result = r.paired_comparison(report,"hybrid_adaptive","prior_ordinal09")
        self.assertEqual(result["paired_states"],2)
        self.assertEqual(result["paired_state_mean_differences"]["state-a"],0.)

    def test_all_systems_and_original_claims_retained_without_reselection(self):
        inputs = fixture(); before = copy.deepcopy(inputs)
        result = r.build_report(*inputs)
        self.assertEqual(inputs,before)
        self.assertEqual(result["continuous_reference"],"continuous_h5")
        self.assertEqual(len(result["comparisons"]),34)
        self.assertEqual(sum(c["reference"] == "prior_ordinal09" for c in result["comparisons"]),16)
        self.assertEqual(sum(c["reference"] == "uniform_expected" for c in result["comparisons"]),16)
        self.assertEqual([c["tested"] for c in result["comparisons"][-2:]],["hybrid_adaptive","covered_hybrid"])
        self.assertEqual(result["original_claim_reviews"][0]["carrier_descriptive_decisions"]["15"],"unavailable")
        self.assertFalse(result["issue_64_authorized"])
        json.dumps(result,allow_nan=False)

    def test_predecessor_selection_must_reproduce(self):
        _,_,previous,_ = fixture()
        previous["selected_continuous_policy"] = "continuous_h15"
        with self.assertRaisesRegex(ValueError,"selection"): r.continuous_reference(previous)

    def test_predecessor_or_access_changes_rejected(self):
        for field in ("fresh_evaluation_opened","final_evaluation_opened","issue_64_authorized","diagnostics_complete"):
            inputs = fixture(); inputs[1][field] = field != "diagnostics_complete"
            with self.subTest(field=field), self.assertRaises(ValueError): r.build_report(*inputs)
        inputs = fixture(); inputs[2]["disposition"] = "supported"
        with self.assertRaisesRegex(ValueError,"binding"): r.build_report(*inputs)

    def test_zero_boundary_does_not_become_positive_evidence(self):
        self.assertEqual(r.direction({"descriptive_95_percent_interval":[0.,.1]}),"interval_includes_zero_not_established")
        self.assertEqual(r.direction({"descriptive_95_percent_interval":[-.1,0.]}),"interval_includes_zero_not_established")
        self.assertEqual(r.direction({}),"unavailable")

    def test_cli_publish_validate_roundtrip_and_corruption(self):
        result = r.build_report(*fixture())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); argv = ["closeout","--output",str(output)]
            with patch.object(r,"validated_inputs",return_value=result), patch.object(r,"log") as logged:
                with patch("sys.argv",argv+["--publish"]): self.assertEqual(r.main(),0)
                self.assertIn(b"\r\n",(output/"comparisons.csv").read_bytes())
                with patch("sys.argv",argv+["--validate"]): self.assertEqual(r.main(),0)
                self.assertIn("not an executed fresh negative experiment",(output/"findings.md").read_text())
                (output/"comparisons.csv").write_bytes(b"corrupt")
                with patch("sys.argv",argv+["--validate"]): self.assertEqual(r.main(),1)
                logged.assert_called_with("error: supplemental report/CSV/findings differ from preserved evidence")

    def test_dry_run_does_not_read_outcomes_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for file in ("plan.json","result.json"): (output/file).write_text("{}")
            argv = ["closeout","--dry-run","--diagnostic",str(output),"--issue75",str(output)]
            with patch("sys.argv",argv), patch.object(r.diagnostic,"read",side_effect=AssertionError("outcome read")), \
                 patch.object(r,"validated_inputs",side_effect=AssertionError("validation")), \
                 patch.object(r.diagnostic,"write",side_effect=AssertionError("write")), patch.object(r,"log"):
                self.assertEqual(r.main(),0)


if __name__ == "__main__": unittest.main()
