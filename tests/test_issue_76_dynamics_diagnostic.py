import argparse
import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts import run_issue_76_dynamics_diagnostic as r
from scripts.issue_76_novelty_inventory import build_inventory
from world_model.training.matched_dynamics import ContinuousDynamics, MatchedController
from tests.test_issue_71_hybrid_readiness import tiny_model


VOCABULARY = [f"{kind}:{i:04d}" for kind,n in (("bird",3),("block",5),("pig",1),("platform",6),("slingshot",1)) for i in range(n)] + ["world:landscape:0000","world:landscape:0001"]


class CountingDynamics(ContinuousDynamics):
    def __init__(self):
        super().__init__(8)
        self.inputs = []

    def carrier(self,z,action,pair):
        self.inputs.append(z.detach().clone())
        return z+pair.delta


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_cli_publish_validate_preserves_csv_line_endings(self):
        plan = {"identity":"plan","samples":[],"inventory_source":{"cells":[]}}
        report = {"diagnostics_complete":False,"disposition":"readiness_or_precision_insufficient",
                  "requirement_ledger":[],"ticket_completion_recommendation":"await_operator_diagnostic"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); review = output/"review"
            argv = ["issue76","--output",str(output),"--review",str(review)]
            with patch.object(r,"load_plan",return_value=plan), patch.object(r,"publication",return_value=report), patch.object(r,"log") as logged:
                with patch("sys.argv",argv+["--publish"]): self.assertEqual(r.main(),0)
                self.assertIn(b"\r\n",(review/"curves.csv").read_bytes())
                with patch("sys.argv",argv+["--validate"]): self.assertEqual(r.main(),0)
                (review/"curves.csv").write_bytes(b"wrong CSV\r\n")
                with patch("sys.argv",argv+["--validate"]): self.assertEqual(r.main(),1)
                logged.assert_called_with("error: review/CSV/source linkage differs from publication")

    def test_validation_repair_preserves_frozen_plan_and_binds_exact_source(self):
        runner = "scripts/run_issue_76_dynamics_diagnostic.py"
        plan = {"identity":"plan","source_revision":"revision","definitions":{"endpoint":225},
                "source_text":{runner:"original","other.py":"unchanged"}}
        current = copy.deepcopy(plan); current["source_text"][runner] = "repaired"
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=Path(directory))
            r.write(args.output/"plan.json",plan)
            frozen = (args.output/"plan.json").read_bytes()
            with patch.object(r,"make_plan",side_effect=lambda args:copy.deepcopy(current)), patch.object(r,"log"):
                with self.assertRaisesRegex(ValueError,"frozen #76"): r.load_plan(args)
                r.repair_validation(args)
                receipt = (args.output/r.VALIDATION_REPAIR).read_bytes()
                self.assertEqual(r.load_plan(args),plan)
                r.repair_validation(args)
                self.assertEqual((args.output/r.VALIDATION_REPAIR).read_bytes(),receipt)
                self.assertEqual((args.output/"plan.json").read_bytes(),frozen)
                current["definitions"]["endpoint"] = 150
                with self.assertRaisesRegex(ValueError,"frozen #76"): r.load_plan(args)
                current["definitions"]["endpoint"] = 225
                current["source_text"][runner] = "another edit"
                with self.assertRaisesRegex(ValueError,"exact frozen/current source"): r.load_plan(args)

    def test_validation_repair_rejects_changes_outside_runner(self):
        runner = "scripts/run_issue_76_dynamics_diagnostic.py"
        plan = {"identity":"plan","source_revision":"revision","definitions":{"endpoint":225},
                "source_text":{runner:"original","other.py":"unchanged"}}
        for change in ("definitions","other source"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                args = argparse.Namespace(output=Path(directory)); r.write(args.output/"plan.json",plan)
                current = copy.deepcopy(plan); current["source_text"][runner] = "repaired"
                if change == "definitions": current["definitions"]["endpoint"] = 150
                else: current["source_text"]["other.py"] = "changed"
                with patch.object(r,"make_plan",return_value=current):
                    with self.assertRaisesRegex(ValueError,"cannot change"):
                        r.repair_validation(args)
                self.assertFalse((args.output/r.VALIDATION_REPAIR).exists())

    def test_source_reader_uses_real_trajectory_identity_field(self):
        state = {"identity":"state","role_ordinal":0,"candidates":[{"identity":"candidate"}]}
        record = {"candidate_identity":"candidate","state_identity":"state","exposure_role":"calibration",
                  "status":"accepted","plan_identity":"source","trajectory_relative_path":"trajectory"}
        trajectory = {"trajectory_identity":"trajectory-id","shots":[{"path":"shot","observation_manifest_identity":"manifest",
                                                                           "terminal_reason":"stable_entered","capture_id":"capture"}]}
        manifest = {"identity":"manifest","exposure_role":"calibration","frame_records":[
            {"fixed_step":100+i,"fixed_time_seconds":i/50,"agent_observation":{"identity":f"frame{i}","relative_path":f"frame{i}.png"}}
            for i in range(226)]}
        def read(path):
            if path.name == "trajectory.json": return trajectory
            if path.name == "observation_trace_manifest.json": return manifest
            return record
        with patch.object(r,"read",side_effect=read):
            result = r.candidate_source(Path("unused"),state,1)
        self.assertEqual(result["trajectory_identity"],"trajectory-id")
        self.assertEqual(result["last_offset"],225)

    def test_membership_balanced_evenly_spaced_and_outcome_independent(self):
        states = [{"identity":f"state-{i:04d}","generator_family":family,"exposure_role":"calibration","outcome":i%3}
                  for i in range(200) for family in ("type010101" if i%2 == 0 else "type010102",)]
        selected = r.select_states(states)
        self.assertEqual(len(selected),24)
        self.assertEqual(sum(s["generator_family"] == "type010101" for s in selected),12)
        ids = [s["identity"] for s in selected]
        for state in states: state["outcome"] = "different"
        self.assertEqual([s["identity"] for s in r.select_states(list(reversed(states)))],ids)
        self.assertIn("state-0000",ids); self.assertIn("state-0199",ids)
        states[0]["exposure_role"] = "final_evaluation"
        with self.assertRaises(ValueError): r.select_states(states)

    def test_metadata_inventory_has_every_cell_and_counterpart(self):
        inventory = build_inventory(r.ROOT,VOCABULARY)
        self.assertEqual((len(inventory["templates"]),len(inventory["pairs"]),len(inventory["cells"])),(80,40,45))
        falling = next(t for t in inventory["templates"] if t["novelty_level"] == 0 and t["family"] == "type010104")
        self.assertEqual(falling["slot_counts"]["platform"],8)
        self.assertEqual(falling["missing_slots"],["platform:0006","platform:0007"])
        right = [t for t in inventory["templates"] if t["novelty_level"] == 5]
        self.assertTrue(all(not t["static_slot_action_fit"] for t in right))
        for pair in inventory["pairs"]:
            self.assertIn(pair["family"],pair["normal"]); self.assertIn(pair["family"],pair["novel"])
        self.assertFalse(inventory["summary"]["runtime_broader_coverage_validated"])
        json.dumps(inventory,allow_nan=False)

    def test_stable_absorption_is_not_nonstable_timestamp_substitution(self):
        stable = r.target_offsets(200,"stable_entered")
        failed = r.target_offsets(200,"timeout")
        self.assertEqual(stable["225"],{"requested_offset":225,"observed_offset":200,"status":"absorbed_stable_terminal"})
        self.assertEqual(failed["225"]["status"],"unavailable_nonstable_truncation")
        self.assertEqual(stable["150"]["status"],"observed")

    def test_same_physical_endpoint_and_no_truth_reset(self):
        for pair in r.base.CONTINUOUS_PAIRS:
            model = CountingDynamics()
            out = r.fixed_curve(model,torch.zeros(236),{"drag_x":-10,"drag_y":-80,"tap_time_ms":0},pair)
            self.assertIsNone(out["failure"])
            self.assertEqual(len(model.inputs),225//pair.delta)
            for t in r.TIMES: self.assertEqual(out["outputs"][str(t)],[float(t)]*236)
            self.assertTrue(torch.equal(model.inputs[1],torch.full((1,236),float(pair.delta))))
            self.assertEqual(sum(s["transition_calls"] for s in out["segments"]),225//pair.delta)
        self.assertEqual(list(inspect.signature(r.fixed_curve).parameters),["model","context","action","pair"])

    def test_local_probe_is_observed_context_not_recursive_score(self):
        model = CountingDynamics(); pair = r.base.CONTINUOUS_PAIRS[1]
        targets = {"carriers":{str(t):[float(t*100)]*236 for t in r.needed_offsets()}}
        out = r.local_predictions(model,targets,torch.zeros(236),{"drag_x":-10,"drag_y":-80,"tap_time_ms":0},pair)
        self.assertEqual(out["outputs"]["15"],[1005.]*236)
        self.assertEqual(out["calls"],5)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_first_local_h15_prediction_equals_first_recursive_h15(self):
        model = CountingDynamics(); pair = r.base.CONTINUOUS_PAIRS[-1]
        context = torch.ones(236)
        target = {"carriers":{str(t):[float(t)]*236 for t in r.needed_offsets()}}
        action = {"drag_x":-10,"drag_y":-80,"tap_time_ms":0}
        recursive = r.fixed_curve(model,context,action,pair)
        local = r.local_predictions(model,target,context,action,pair)
        self.assertEqual(recursive["outputs"]["15"],local["outputs"]["15"])

    def test_field_error_uses_target_not_prediction_masks(self):
        p,t = torch.zeros(236),torch.zeros(236)
        p[2+5:2+7] = 10
        objective = r.base.TaskObjective.from_vocabulary(VOCABULARY)
        missing = r.field_errors(p.tolist(),t.tolist(),objective)
        self.assertIsNone(missing["position_mse"])
        t[2+7] = 1
        available = r.field_errors(p.tolist(),t.tolist(),objective)
        self.assertEqual(available["position_available_values"],2)
        self.assertEqual(available["position_mse"],100.)
        self.assertEqual(p[2+7],0)  # predicted mask cannot erase a position error

    def test_unavailable_fields_and_states_survive_aggregation(self):
        empty = r.average_errors([None,None])
        self.assertEqual(empty,{"available_candidates":0})
        full = {"carrier_mse":1.,"position_mse":None,"position_available_values":0}
        mean = r.average_errors([empty,r.average_errors([full])],unit="states")
        self.assertEqual(mean["available_states"],1)
        self.assertEqual(mean["carrier_mse"],1.)
        self.assertIsNone(mean["position_mse"])
        json.dumps(mean,allow_nan=False)

    def test_curves_do_not_update_predictor_weights(self):
        model = ContinuousDynamics(8); initial = {k:v.clone() for k,v in model.state_dict().items()}
        r.fixed_curve(model,torch.zeros(236),{"drag_x":-10,"drag_y":-80,"tap_time_ms":0},r.base.CONTINUOUS_PAIRS[-1])
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in initial.items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_partial_curve_resume_reuses_exact_record_without_inference(self):
        model = CountingDynamics(); action={"drag_x":-10,"drag_y":-80,"tap_time_ms":0}
        sample={"state":{"identity":"state","role_ordinal":0}}
        row={"candidates":[{"identity":"candidate","action":action}]}
        plan={"identity":"plan","contract":{"vocabulary":VOCABULARY}}
        target={"context":[0.]*236,"candidates":[{"source":{"ordinal":1},"carriers":{str(t):[0.]*236 for t in r.needed_offsets()}}]}
        original={"policies":{"continuous_h15":[{"failure":None,"endpoint":[225.]*236}]}}
        with tempfile.TemporaryDirectory() as directory:
            args=argparse.Namespace(output=Path(directory),device="cpu")
            first=r.stored_curve(args,plan,{},sample,target,row,1,"continuous_h15",1,model,original)
            with patch.object(r,"fixed_record",side_effect=AssertionError("cached record must not rescore")):
                second=r.stored_curve(args,plan,{},sample,target,row,1,"continuous_h15",1,model,original)
            self.assertEqual(first,second)
            self.assertEqual(len(model.inputs),20)  # fifteen recursive plus five local calls, once

    def test_completed_run_does_not_spend_budget_replaying_cache(self):
        with patch.object(r,"artifact_inventory",return_value={"target_states":24,"expected_target_states":24,
                "fixed_candidate_records":10368,"expected_fixed_candidate_records":10368}), \
             patch.object(r.covered,"Budget",side_effect=AssertionError("no new work")):
            r.run_diagnostic(argparse.Namespace(),{})

    def test_ledger_does_not_claim_unexecuted_fresh_tasks_passed(self):
        pending = {row["id"]:row for row in r.requirement_ledger(False,False)}
        complete = {row["id"]:row for row in r.requirement_ledger(True,False)}
        self.assertEqual(pending["a01_curves"]["status"],"pending_operator_diagnostic")
        self.assertEqual(complete["a01_curves"]["status"],"completed")
        for name in ("fresh_protocol","fresh_precision","fresh_roles","fresh_execution","fresh_metrics","fresh_graphics"):
            self.assertEqual(complete[name]["status"],"not_executed_preaccess_readiness_stop")
        self.assertEqual(complete["advancement"]["status"],"not_authorized")

    def test_freeze_rejects_settings_or_source_change_not_identical_commit(self):
        old = {"source_revision":"old","definitions":{"endpoint":225},"source_text":{"file":"original"}}
        current = copy.deepcopy(old); current["source_revision"] = "new"
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=Path(directory)); r.write(args.output/"plan.json",old)
            with patch.object(r,"make_plan",side_effect=lambda args:copy.deepcopy(current)):
                self.assertEqual(r.load_plan(args),old)
                current["definitions"]["endpoint"] = 300
                with self.assertRaises(ValueError): r.load_plan(args)

    def test_no_write_dry_run(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(r,"make_plan",return_value={"samples":[{}]*24,"inventory_source":{"summary":{}}}), \
             patch("sys.argv",["runner","--dry-run","--output",directory]):
            self.assertEqual(r.main(),0)
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_partial_report_is_explicit_and_strict_json(self):
        inventory = build_inventory(r.ROOT,VOCABULARY)
        plan = {"identity":"test","samples":[{"state":{"role_ordinal":0}}],"inventory_source":inventory,
                "contract":{"vocabulary":VOCABULARY}}
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=Path(directory))
            report = r.publication(args,plan)
            self.assertFalse(report["diagnostics_complete"])
            self.assertEqual(report["ticket_completion_recommendation"],"await_operator_diagnostic")
            self.assertEqual(report["fresh_executed_lineages"],0)
            self.assertFalse(report["issue_64_authorized"])
            path = args.output/"result.json"; r.write(path,report)
            self.assertEqual(r.read(path),report)

    def test_diagnostic_budget_stop_persists_and_is_reported(self):
        definitions=r.definitions(); definitions["diagnostic_seconds_max"]=1
        with tempfile.TemporaryDirectory() as directory:
            args=argparse.Namespace(output=Path(directory),device="cpu")
            with patch.object(r.time,"monotonic",side_effect=[100.,102.]):
                budget=r.covered.Budget(args,{"identity":"test","protocol":definitions},"diagnostic")
                with self.assertRaises(r.covered.BudgetStop): budget.tick()
            with self.assertRaises(r.covered.BudgetStop): r.covered.Budget(args,{"identity":"test","protocol":definitions},"diagnostic")
            plan={"identity":"test","samples":[{"state":{"role_ordinal":0}}],
                  "inventory_source":build_inventory(r.ROOT,VOCABULARY),"contract":{"vocabulary":VOCABULARY}}
            report=r.publication(args,plan)
            self.assertTrue(report["diagnostic_budget_stopped"])
            self.assertFalse(report["diagnostics_complete"])
            self.assertEqual(report["ticket_completion_recommendation"],"review_preaccess_stop")

    def test_svg_uses_recorded_times_and_discloses_empty_curve(self):
        chart = r.svg_chart("test",{"fixed":[(15,0.),(30,1.),(60,10.),(150,100.),(225,1000.)]})
        self.assertIn("225",chart)
        self.assertIn("polyline",chart)
        self.assertIn("No complete curve",r.svg_chart("missing",{"adaptive":[]}))

    def test_full_publication_and_review_strict_json(self):
        """Complete publication branch with small synthetic, fully paired membership."""
        inventory = build_inventory(r.ROOT,VOCABULARY)
        samples = []
        for i,family in enumerate(("type010101","type010102")):
            state = {"identity":f"state-{i}","role_ordinal":i,"generator_family":family}
            sources = [{"identity":f"s{i}c{j}","status":"accepted","ordinal":j,
                        "last_offset":225,"terminal_reason":"stable_entered"} for j in range(1,13)]
            samples.append({"state":state,"sources":sources})
        bindings = {str(seed):{"continuous":{},"hybrid":{}} for seed in r.SEEDS}
        plan = {"identity":"test","samples":samples,"inventory_source":inventory,"contract":{"vocabulary":VOCABULARY},
                "source_release":"release","source74_readiness":{"cost":{}},"source75_cost":{},"source74_plan_identity":"source74",
                "source75_plan_identity":"source75","source75_controller_bindings":bindings}
        models = {"continuous":ContinuousDynamics(8),"hybrid":tiny_model()}
        controls = {a:MatchedController(a == "continuous") for a in r.base.ARMS}
        vector = [0.]*236; action = {"drag_x":-10,"drag_y":-80,"tap_time_ms":0}
        objective = r.base.TaskObjective.from_vocabulary(VOCABULARY)
        cost = objective(torch.tensor(vector))
        def row_for(index):
            return {"state":f"state-{index}","candidates":[{"identity":f"s{index}c{j}","action":action,
                      "accepted":True,"realized_count_cost":float(j),"horizon_carrier":vector} for j in range(1,13)]}
        def fake_read(path):
            path = Path(path)
            if path.name == "budget-diagnostic.json": return {"plan_identity":"test","stopped":False}
            if "targets" in path.parts:
                i = int(path.stem.removeprefix("state-"))-1
                return {"plan_identity":"test","state":f"state-{i}","context":vector,"perception":{"wall_seconds":.01},
                        "candidates":[{"source":s,"carriers":{str(t):vector for t in r.needed_offsets()},
                                       "offsets":r.target_offsets(225,"stable_entered"),"timing":{},"target_parser":{"wall_seconds":0.}}
                                      for s in samples[i]["sources"]]}
            if "candidate-results" in path.parts:
                i = int(path.stem.split("-")[1].removeprefix("s"))-1; j = int(path.stem.split("-")[2].removeprefix("c"))
                return {"candidate_identity":f"s{i}c{j}","status":"accepted","interaction_coverage":[],
                        "endpoint_outcome":{"pig_contact":False,"block_contact":False,"support_change":False}}
            seed = int(next(p for p in path.parts if p.startswith("seed-")).removeprefix("seed-"))
            if "fixed" in path.parts:
                i = int(next(p for p in path.parts if p.startswith("state-")).removeprefix("state-"))-1
                name = path.parts[-2]; j = int(path.stem.removeprefix("candidate-")); arm,pair = r.SYSTEMS[name]
                mac = r.base.work(models[arm],pair)
                segments = [{"endpoint":t,"transition_calls":(t-(r.TIMES[k-1] if k else 0))//pair.delta,"wall_seconds":.01} for k,t in enumerate(r.TIMES)]
                return {"plan_identity":"test","state":f"state-{i}","seed":seed,"system":name,"ordinal":j,
                        "candidate_identity":f"s{i}c{j}","action":action,"cost":cost,
                        "prediction":{"outputs":{str(t):vector for t in r.TIMES},"completed_steps":225,"failure":None,
                                      "linear_macs":mac*(225//pair.delta),"segments":segments,"transition_wall_seconds":.05},
                        "local":{"outputs":{str(t):vector for t in r.TIMES},"calls":5,"linear_macs":5*mac,"wall_seconds":.01,"unavailable":{}}}
            i = int(path.stem.removeprefix("state-"))-1
            names = r.ADAPTIVE[2:] if "source75" in path.parts else r.ADAPTIVE[:2]
            policies = {}
            for name in names:
                arm = "continuous" if "continuous" in name else "hybrid"
                pair = r.base.CONTINUOUS_PAIRS[-1]
                trace = [{"start_fixed_step":t,"horizon":15,"mode":"continuous","linear_macs":r.base.work(models[arm],pair,controls[arm]),
                          "controller_calls":1,"transition_calls":1,"symbol_decoder_calls":0} for t in range(0,225,15)]
                policies[name] = [{"candidate_identity":f"s{i}c{j}","action":action,"cost":cost,"endpoint":vector,"failure":None,
                                   "trace":trace,"wall_seconds":.01} for j in range(1,13)]
            return {"state":f"state-{i}","seed":seed,"plan_identity":"source75" if "source75" in path.parts else "source74",
                    "controller_bindings":bindings[str(seed)],"policies":policies,"perception":{"wall_seconds":.01}}
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=Path(directory),issue74=Path("source74"),issue75=Path("source75"),device="cpu",review=Path(directory)/"review")
            r.write(args.output/"budget-diagnostic.json",{})
            with patch.object(r,"artifact_inventory",return_value={"target_states":2,"expected_target_states":2,"fixed_candidate_records":864,"expected_fixed_candidate_records":864}), \
                 patch.object(r,"read",side_effect=fake_read), patch.object(r.base,"load_plan",return_value={}), \
                 patch.object(r.base.grid,"load_plan",return_value={}), \
                 patch.object(r.base.grid,"endpoint",side_effect=lambda args,gp,index:row_for(index)), \
                 patch.object(r.base,"load_predictor",side_effect=lambda args,previous,seed,arm:(models[arm],{})), \
                 patch.object(r.base,"load_control",side_effect=lambda args,previous,seed,arm:(controls[arm],{})), \
                 patch.object(r.covered,"load_plan",return_value={}), \
                 patch.object(r.covered,"load_control",side_effect=lambda args,previous,seed,arm:(controls[arm],{})), patch.object(r,"log"):
                report = r.publication(args,plan)
                self.assertTrue(report["diagnostics_complete"])
                self.assertEqual(report["independent_state_count"],2)
                self.assertFalse(report["issue_64_authorized"])
                self.assertEqual(report["disposition"],"readiness_or_precision_insufficient")
                self.assertEqual(report,r.publication(args,plan))
                path = args.output/"result.json"; r.write(path,report)
                self.assertEqual(r.base.read(path),report)
                self.assertIn("recursive",r.curve_csv(report))
                page = r.review_html(args,report,inventory,{"entries":[]})
                self.assertIn("No new capture",page)
                self.assertIn("polyline",page)


if __name__ == "__main__": unittest.main()
