import argparse
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts import run_issue_75_controller_coverage as r
from tests.test_issue_71_hybrid_readiness import batch
from tests.test_issue_71_hybrid_readiness import tiny_model
from world_model.training.matched_dynamics import ContinuousDynamics, MatchedController


class NoBudget:
    def tick(self, **kwargs):
        pass


def small_plan():
    p = r.protocol(); p["steps_per_round"] = 12; p["batch_size"] = 4
    return {"identity":"test", "contract":{}, "source_plan_identity":"source", "protocol":p,
            "claim_boundary":"test"}


def positive_summaries():
    summaries = {}
    for seed in r.SEEDS:
        summaries[str(seed)] = {p:{"states":200,"mean_regret":.33,"prediction_failure_states":0,
                                   "mean_endpoint_carrier_mse":1.,"max_perception_planning_seconds":1.,
                                   "pair_usage":{"15:continuous":100,"5:macro":100}}
                               for p in r.POLICIES}
        summaries[str(seed)]["covered_hybrid"]["mean_regret"] = .28
        summaries[str(seed)]["covered_hybrid"]["mean_endpoint_carrier_mse"] = .8
    return summaries


class CoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d, patch.object(r,"make_plan",return_value={"protocol":r.protocol()}), \
             patch("sys.argv",["runner","--dry-run","--output",d]):
            self.assertEqual(r.main(),0)
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_all_windows_included_and_original_first_labels_unchanged(self):
        data = {k:v for k,v in batch(4).items() if k in ("z","action","length")}
        windows = [{"shot":0,"start":s} for s in (0,100,200,300)]
        model, control = ContinuousDynamics(16), MatchedController(True)
        args = argparse.Namespace(issue74=Path("unused"),device="cpu")
        with patch.object(r.base,"shard",return_value=data), patch.object(r.torch,"load",return_value={"windows":windows}), \
             patch.object(r.base,"first_windows",return_value=[0]):
            new = r.label_lineage(args,{},model,control,5,False)
            old = r.base.label_lineage(r.source_args(args),{},model,control,5,False)
        self.assertEqual(len(new["tensors"]["z"]),240)
        self.assertEqual([w["first_window"] for w in new["coverage"]],[True,False,False,False])
        for k in old: self.assertTrue(torch.equal(old[k],new["tensors"][k][:60]))
        self.assertTrue(torch.equal(new["tensors"]["z"][60],data["z"][1,0]))

    def test_aggregation_does_not_mutate_source_targets(self):
        data = {k:v for k,v in batch(1).items() if k in ("z","action","length")}
        original = data["z"].clone()
        with patch.object(r.base,"shard",return_value=data), \
             patch.object(r.torch,"load",return_value={"windows":[{"shot":0,"start":200}]}):
            out = r.label_lineage(argparse.Namespace(issue74=Path("unused"),device="cpu"),{},ContinuousDynamics(16),MatchedController(True),5,True)
        self.assertTrue(torch.equal(data["z"],original))
        self.assertEqual(len(out["tensors"]["z"]),60)

    def test_resume_equal_and_predictor_weights_unchanged(self):
        data = batch(1)
        label = {"coverage":[{"shot":0,"start":0,"rows":60,"first_window":True}],
                 "tensors":{"z":data["z"][0,:60],"action":data["action"].expand(60,-1),
                            "remaining":torch.arange(60,0,-1),"labels":torch.zeros(60,dtype=torch.long)}}
        plan = small_plan(); source = {"records":[{} for _ in range(5)]}
        model = ContinuousDynamics(16); initial = {n:p.clone() for n,p in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as d, patch.object(r.base,"load_predictor",return_value=(model,{})), \
             patch.object(r,"label_lineage",return_value=label):
            args = argparse.Namespace(output=Path(d)/"resume",issue74=Path("unused"),device="cpu")
            r.train_cell(args,plan,{},source,74,"continuous",NoBudget(),indices=(5,),stop_after=6)
            self.assertFalse((r.root(args,74,"continuous")/"controller.pt").exists())
            r.train_cell(args,plan,{},source,74,"continuous",NoBudget(),indices=(5,))
            resumed,_ = r.load_control(args,plan,74,"continuous")
            args.output = Path(d)/"full"
            r.train_cell(args,plan,{},source,74,"continuous",NoBudget(),indices=(5,))
            full,_ = r.load_control(args,plan,74,"continuous")
            self.assertTrue(all(torch.equal(v,full.state_dict()[k]) for k,v in resumed.state_dict().items()))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in initial.items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_budget_persists_and_cannot_resume_past_stop(self):
        plan = small_plan(); plan["protocol"]["training_seconds_max"] = 5
        with tempfile.TemporaryDirectory() as d:
            args = argparse.Namespace(output=Path(d),device="cpu")
            with patch.object(r.time,"monotonic",side_effect=[100.,103.]):
                budget = r.Budget(args,plan,"training"); budget.tick()
            with patch.object(r.time,"monotonic",side_effect=[200.,203.]):
                budget = r.Budget(args,plan,"training")
                with self.assertRaises(r.BudgetStop): budget.tick()
            with self.assertRaises(r.BudgetStop): r.Budget(args,plan,"training")
            result = r.publication(args,plan,{})
            self.assertEqual(result["disposition"],"budget_or_readiness_insufficient")
            self.assertFalse(result["issue_64_authorized"])

    def test_gate_positive_requires_action_improvement_and_no_fresh_authorization(self):
        out = r.decision(positive_summaries(),81,True,r.protocol())
        self.assertEqual(out["disposition"],"correction_supported_for_prospective_test")
        self.assertFalse(out["issue_64_authorized"])

    def test_decision_can_be_written_as_strict_json(self):
        for recursive_error in (.8, 2.):
            summaries = positive_summaries()
            for row in summaries.values():
                row["covered_hybrid"]["mean_endpoint_carrier_mse"] = recursive_error
            result = r.decision(summaries,81,True,r.protocol())
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"result.json"
                r.write(path,result)
                self.assertEqual(r.read(path),result)
            self.assertIs(type(result["checks"]["recursive_mse_not_worse"]),bool)

    def test_publication_repair_keeps_original_plan_and_rejects_other_changes(self):
        frozen = {"identity":"frozen", "protocol":{"margin":.02},
                  "source_text":{r.RUNNER_SOURCE:"old runner", "other.py":"unchanged"}}
        current = copy.deepcopy(frozen); current["source_text"][r.RUNNER_SOURCE] = "repaired runner"
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=Path(directory))
            r.write(args.output/"plan.json",frozen)
            before = (args.output/"plan.json").read_bytes()
            with patch.object(r,"make_plan",side_effect=lambda args:copy.deepcopy(current)):
                with self.assertRaises(ValueError): r.load_plan(args)
                r.repair_publication(args)
                self.assertEqual(r.load_plan(args),frozen)
                r.repair_publication(args)
                self.assertEqual((args.output/"plan.json").read_bytes(),before)
                current["protocol"]["margin"] = .01
                with self.assertRaises(ValueError): r.load_plan(args)
                current["protocol"]["margin"] = .02
                current["source_text"]["other.py"] = "changed"
                with self.assertRaises(ValueError): r.load_plan(args)
                current["source_text"]["other.py"] = "unchanged"
                current["source_text"][r.RUNNER_SOURCE] = "another runner"
                with self.assertRaises(ValueError): r.load_plan(args)

    def test_better_mse_or_switching_alone_cannot_pass(self):
        s = positive_summaries()
        for row in s.values(): row["covered_hybrid"]["mean_regret"] = .34
        out = r.decision(s,81,True,r.protocol())
        self.assertEqual(out["disposition"],"not_supported_by_this_pilot")
        self.assertTrue(out["checks"]["recursive_mse_not_worse"])

    def test_stronger_new_continuous_controller_is_not_ignored(self):
        s = positive_summaries()
        for row in s.values(): row["covered_continuous"]["mean_regret"] = .27
        out = r.decision(s,81,True,r.protocol())
        self.assertEqual(out["selected_continuous_policy"],"covered_continuous")
        self.assertFalse(out["checks"]["beats_strongest_comparator"])

    def test_one_favorable_seed_cannot_hide_two_regressions(self):
        s = positive_summaries()
        for i,row in enumerate(s.values()): row["covered_hybrid"]["mean_regret"] = .1 if i == 0 else .34
        self.assertFalse(r.decision(s,81,True,r.protocol())["checks"]["improves_two_seeds"])

    def test_failures_collapse_and_missing_symbols_are_retained(self):
        s = positive_summaries()
        s[str(r.SEEDS[0])]["covered_continuous"]["prediction_failure_states"] = 1
        for row in s.values(): row["covered_hybrid"]["pair_usage"] = {"1:continuous":100}
        out = r.decision(s,81,False,r.protocol())
        self.assertFalse(out["checks"]["no_prediction_failures"])
        self.assertFalse(out["checks"]["mode_use"])
        self.assertFalse(out["checks"]["horizon_use"])
        self.assertFalse(out["checks"]["symbol_content_intervention"])

    def test_worse_accumulated_error_cannot_pass(self):
        s = positive_summaries()
        for row in s.values(): row["covered_hybrid"]["mean_endpoint_carrier_mse"] = 2.
        self.assertFalse(r.decision(s,81,True,r.protocol())["checks"]["recursive_mse_not_worse"])

    def test_wrong_label_source_rejected(self):
        value={"binding":{},"record":"wrong","round":0,"coverage":[],"tensors":{"z":[]}}
        with self.assertRaises(ValueError): r.validate_label(value,{}, {"records":["right"]},1,0)

    def test_complete_publication_recomputes_and_preserves_controls(self):
        """Exercise the full publication branch without fitting or real data I/O."""
        plan = small_plan(); plan["contract"] = {"vocabulary":["pig:0001"]}
        plan["source_readiness"] = {"per_seed":{str(s):{"policies":{p:{"state_regrets":[0.]*200} for p in r.base.POLICIES}}
                                               for s in r.SEEDS}}
        previous = {"identity":"source", "contract":plan["contract"]}
        source = {"records":[{} for _ in range(3000)]}
        models = {"continuous":ContinuousDynamics(16), "hybrid":tiny_model()}
        controls = {a:MatchedController(a == "continuous") for a in r.ARMS}
        objective = r.base.TaskObjective.from_vocabulary(plan["contract"]["vocabulary"])
        z = torch.zeros(236); cost = objective(z)
        row = {"state":"fixture", "candidates":[{"identity":f"candidate-{i}", "action":{}, "accepted":True,
                                                      "realized_count_cost":0., "horizon_carrier":z.tolist()} for i in range(12)]}
        def predictions(policy):
            arm,fixed = r.base.policy_parts(policy)
            pair = fixed or r.base.PAIRS[6]
            trace = [{"start_fixed_step":t,"horizon":pair.delta,"mode":str(pair.abstraction),
                      "linear_macs":r.base.work(models[arm],pair,controls[arm] if fixed is None else None),
                      "controller_calls":int(fixed is None),"transition_calls":1,"symbol_decoder_calls":0}
                     for t in range(0,225,pair.delta)]
            return [{"candidate_identity":c["identity"], "action":{}, "endpoint":z.tolist(),"cost":cost,
                     "trace":trace,"failure":None,"wall_seconds":.01} for c in row["candidates"]]
        original_policies = {p:predictions(p) for p in r.base.POLICIES}
        def original(args,index,seed):
            return {"plan_identity":"source", "seed":seed,"state":"fixture",
                    "policies":original_policies,"perception":{"wall_seconds":.001}}
        def fake_read(path):
            path = Path(path)
            if path.name.startswith("budget-"):
                return {"plan_identity":"test","stopped":False}
            if "candidate-results" in path.parts:
                ordinal = int(path.stem.rsplit("c",1)[1])-1
                return {"candidate_identity":f"candidate-{ordinal}","interaction_coverage":[]}
            seed = int(next(p for p in path.parts if p.startswith("seed-")).split("-")[1])
            return {"plan_identity":"test","seed":seed,"state":"fixture", "perception":{"wall_seconds":.001},
                    "controller_bindings":{a:r.binding(plan,seed,a) for a in r.ARMS},
                    "policies":{r.NEW[a]:original_policies[f"{a}_adaptive"] for a in r.ARMS}}
        def fake_load(path, **kwargs):
            path = Path(path); seed = int(next(p for p in path.parts if p.startswith("seed-")).split("-")[1])
            arm = "continuous" if "continuous" in path.parts else "hybrid"
            round_index = int(next(p for p in path.parts if p.startswith("labels-")).split("-")[1])
            return {"binding":r.binding(plan,seed,arm),"record":{},"round":round_index,"wall_seconds":.001,
                    "coverage":[{"rows":1,"first_window":True},{"rows":1,"first_window":False}],
                    "tensors":{"z":torch.zeros(2,236)}}
        with tempfile.TemporaryDirectory() as d:
            args = argparse.Namespace(output=Path(d),issue74=Path("source"),device="cpu")
            for stage in ("training","scoring"): r.write(args.output/f"budget-{stage}.json",{})
            with patch.object(r,"inventory",return_value={"controllers":6,"states":600}), \
                 patch.object(r,"read",side_effect=fake_read), patch.object(r.torch,"load",side_effect=fake_load), \
                 patch.object(r.base.grid,"load_plan",return_value={"source_release":"release"}), \
                 patch.object(r.base.grid,"endpoint",return_value=row), \
                 patch.object(r.base.old,"load_plan",return_value=source), \
                 patch.object(r.base,"load_predictor",side_effect=lambda args,previous,seed,arm:(models[arm],{})), \
                 patch.object(r,"load_control",side_effect=lambda args,plan,seed,arm:(controls[arm],{})), \
                 patch.object(r,"source_state",side_effect=original), \
                 patch.object(r,"symbol_audit",return_value={"micro":.1,"macro":.1}), patch.object(r,"log"):
                result = r.publication(args,plan,previous)
                self.assertEqual(result["disposition"],"not_supported_by_this_pilot")
                self.assertEqual(len(result["checkpoints"]),6)
                self.assertEqual(result["per_seed"][str(r.SEEDS[0])]["covered_hybrid"]["states"],200)
                r.write(args.output/"result.json",result)
                self.assertEqual(r.base.read(args.output/"result.json"),result)
                self.assertEqual(result,r.publication(args,plan,previous))
                plan["source_readiness"]["per_seed"][str(r.SEEDS[0])]["policies"]["continuous_h1"]["state_regrets"][0] = 1.
                with self.assertRaisesRegex(ValueError,"original #74 control"):
                    r.publication(args,plan,previous)


if __name__ == "__main__": unittest.main()
