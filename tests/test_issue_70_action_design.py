from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts import run_issue_70_action_design as workflow
from scripts import issue_70_live_pilot as live
from world_model.planning.gameplay import PlanningObservation
from world_model.planning.task_objective import (
    TaskObjective,TaskCandidateEvaluator,ranking_diagnostic,summarize_rankings,
)


class ActionModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()));self.inputs=[]
    def carrier(self,current,action,pair):
        self.inputs.append(current.clone())
        out=current.clone();out[:,2]=torch.sigmoid(action[:,1]*4)+self.anchor*0
        return out


def fixture(count=24):
    rows=[]
    for i in range(count):
        candidates=[]
        for j,action in enumerate(workflow.old.probe.broad_action_candidates(f"s{i}",workflow.old.probe._bounds())):
            carrier=torch.zeros(197);carrier[2]=float(j!=i%12)
            candidates.append({"identity":action.identity,"action":asdict(action.action),"accepted":True,
                "terminal_carrier":carrier.tolist(),"horizon_carrier":carrier.tolist(),
                "realized_count_cost":float(j!=i%12)*1000,
                "realized_progress_cost":float(j!=i%12)*1000+0.1,"actual_counts":[float(j!=i%12),0]})
        rows.append({"state":f"s{i}","objective":{"pig_slots":[0],"block_slots":[1]},
                     "context":torch.zeros(197).tolist(),"parser_identity":"test-parser","candidates":candidates})
    scores={};cells=[]
    for prefix in ("original","teacher-forced-h15",*workflow.old.CORRECTIONS):
        for seed in workflow.old.SEEDS:
            name=f"{prefix}-{seed}";cell={"name":name,"checkpoint":name,"kind":"single"};cells.append(cell)
            scores[name]={"cell":cell,"contract":workflow.contract(),"parser_identity":"test-parser",
                "objective":rows[0]["objective"],"wall_seconds":1.0,"rows":[{
                "state":r["state"],"candidate_ids":[c["identity"] for c in r["candidates"]],
                "costs":[c["realized_count_cost"] for c in r["candidates"]],
                "diagnostics":[{"horizon_mse":0.0,"horizon_count_error":[0.0,0.0],
                    "maximum_carrier_bound_excess":0.0,"model_evaluations":15} for c in r["candidates"]]} for r in rows]}
    return {"models":cells},rows,scores


class ActionDesignTests(unittest.TestCase):
    def test_utility_uses_only_carrier_and_frozen_kind_slots(self):
        objective=TaskObjective.from_vocabulary(("block:1","pig:1","bird:1"))
        carrier=torch.zeros(197);carrier[2]=1;carrier[15]=1;carrier[28]=1
        self.assertEqual(objective(carrier),1001)
        carrier[15]=0
        self.assertEqual(objective(carrier),1)
        carrier[2]=0
        self.assertEqual(objective(carrier),0)
        self.assertEqual(objective.counts(carrier),(0,0))

    def test_scoring_probability_bounds_do_not_change_carrier(self):
        carrier=torch.zeros(197);carrier[2]=1.5
        objective=TaskObjective((0,),())
        self.assertEqual(objective(carrier),1000)
        self.assertEqual(float(carrier[2]),1.5)

    def test_top_k_and_tie_ambiguity(self):
        result=ranking_diagnostic([0,0,0,0],[5,5,0,5])
        self.assertFalse(result["top1"]);self.assertTrue(result["top3"])
        self.assertEqual(result["best_rank_interval"],[1,4])
        all_tied=ranking_diagnostic([0,0,0],[5,5,5])
        self.assertEqual(all_tied["best_rank_interval"],[1,1])
        self.assertFalse(all_tied["outcome_discriminating"])

    def test_prediction_failures_and_replay_failures_not_filtered(self):
        self.assertEqual(ranking_diagnostic([None,1],[0,1])["regret"],1)
        self.assertEqual(ranking_diagnostic([0,1],[1e9,0])["regret"],1)
        report=summarize_rankings([ranking_diagnostic([None,1],[0,1]),ranking_diagnostic([1,1],[1,1])])
        self.assertEqual(report["all_states"]["states"],2)
        self.assertEqual(report["prediction_failures"],1)

    def test_objective_audit_can_fail_even_with_perfect_engine_outcomes(self):
        _,rows,_=fixture()
        self.assertTrue(workflow.audit_payload(rows)["passed"])
        for row in rows:
            for candidate in row["candidates"]:
                candidate["terminal_carrier"][2]=1.0
        report=workflow.audit_payload(rows)
        self.assertFalse(report["passed"])
        self.assertEqual(report["count_ranking"]["predicted_all_tied_states"],24)

    def test_horizon_evaluator_self_conditions_and_never_reads_outcomes(self):
        model=ActionModel();objective=TaskObjective((0,),(1,))
        evaluator=TaskCandidateEvaluator((model,),objective,workflow.old.probe._bounds(),steps=2)
        obs=PlanningObservation("x",torch.zeros(197),(0,),(10,20))
        action=workflow.old.probe.broad_action_candidates("x",workflow.old.probe._bounds())[0].action
        evaluator.evaluate(obs,(action,))
        self.assertFalse(torch.equal(model.inputs[0],model.inputs[1]))
        self.assertEqual(evaluator.records[0]["model_evaluations"],2)

    def test_cem_and_grid_have_same_candidate_budget(self):
        with redirect_stdout(io.StringIO()):
            for system in ("original_cem","corrected_cem","corrected_grid"):
                evaluator=TaskCandidateEvaluator((ActionModel(),),TaskObjective((0,),(1,)),workflow.old.probe._bounds(),steps=2)
                obs=PlanningObservation("x",torch.zeros(197),(0,),(10,20))
                action,record=live.select_action(system,obs,evaluator,{},100)
                self.assertEqual(record["candidate_count"],12)
                self.assertEqual(record["model_evaluations"],24)
                self.assertEqual(action.tap_time_ms,0)

    def test_pilot_requires_both_objective_and_model_gates(self):
        plan,rows,scores=fixture()
        frozen=workflow.freeze_payload(plan,rows,scores)
        self.assertTrue(frozen["pilot_allowed"])
        self.assertFalse(frozen["issue_64_authorized"])
        for score in scores.values():
            for row in score["rows"]:row["costs"]=[1.0]*12
        self.assertFalse(workflow.freeze_payload(plan,rows,scores)["pilot_allowed"])

    def test_ensemble_checkpoint_order_does_not_change_selection(self):
        plan,rows,scores=fixture()
        expected=workflow.freeze_payload(plan,rows,scores)
        actual=workflow.freeze_payload(plan,rows,dict(reversed(list(scores.items()))))
        # Table ordering is descriptive; selected configurations are invariant.
        self.assertEqual(expected["corrected"],actual["corrected"])
        self.assertEqual(expected["original"],actual["original"])

    def test_pilot_gate_blocks_before_game_generation(self):
        plan,rows,scores=fixture();frozen=workflow.freeze_payload(plan,rows,scores)
        frozen["pilot_allowed"]=False
        with patch.object(workflow,"read",return_value=frozen),patch.object(workflow,"endpoint_rows",return_value=rows),patch.object(workflow,"plan_for",return_value=plan),patch.object(workflow,"load_scores",return_value=scores),patch.object(live,"check_disjointness") as generate:
            with self.assertRaises(ValueError):live.run_pilot(SimpleNamespace(output=Path("unused")))
            generate.assert_not_called()

    def test_failed_attempt_is_retained_without_replacement(self):
        plan,rows,scores=fixture();frozen=workflow.freeze_payload(plan,rows,scores)
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(output=Path(tmp),audit=Path(tmp)/"audit",device="cpu")
            workflow.write(args.output/"pilot-source-inventory.json",{"authored_birds":[1]*12})
            level=live.pilot_plan(frozen)["levels"][0]
            slot=live.slot_for(level,"corrected_grid")
            (args.output/"pilot"/slot["slot_identity"]).mkdir(parents=True)
            with patch.object(live.capture,"_collect_lineage_attempt") as collect,redirect_stdout(io.StringIO()):
                result=live.run_trial(args,frozen,level,"corrected_grid")
                self.assertFalse(result["success"])
                self.assertIn("interrupted",result["failure"])
                self.assertEqual(live.run_trial(args,frozen,level,"corrected_grid"),result)
                collect.assert_not_called()

    def test_trial_plan_caps_shots_at_authored_birds(self):
        one=live.slot_for({"ordinal":0,"generation_seed":1,"generator_family":"type010101","authored_birds":1},"corrected_cem")
        three=live.slot_for({"ordinal":1,"generation_seed":2,"generator_family":"type010102","authored_birds":3},"corrected_cem")
        self.assertEqual(len(one["planned_actions"]),1)
        self.assertEqual(len(three["planned_actions"]),3)

    def test_no_write_dry_run_all_planners_reobserve(self):
        with patch.object(workflow,"write",side_effect=AssertionError("write")),patch("torch.save",side_effect=AssertionError("save")),redirect_stdout(io.StringIO()) as output:
            workflow.dry_run()
        self.assertEqual(output.getvalue().count("shot=3/3 reobserved"),4)

    def test_failed_prediction_records_attempted_compute(self):
        model=ActionModel()
        evaluator=TaskCandidateEvaluator((model,),TaskObjective((0,),(1,)),workflow.old.probe._bounds())
        action=workflow.old.probe.broad_action_candidates("x",workflow.old.probe._bounds())[0].action
        with patch.object(model,"carrier",return_value=torch.full((1,197),float("nan"))):
            with self.assertRaises(ValueError):
                evaluator.evaluate(PlanningObservation("x",torch.zeros(197),(0,),(0,0)),(action,))
        self.assertEqual(evaluator.records[-1]["model_evaluations"],1)
        self.assertIsNone(evaluator.records[-1]["cost"])

    def test_offline_publish_validate_and_corruption_detection(self):
        plan,rows,scores=fixture()
        for score in scores.values():
            for row in score["rows"]:row["costs"]=[1.0]*12
        frozen=workflow.freeze_payload(plan,rows,scores)
        self.assertFalse(frozen["pilot_allowed"])
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(output=Path(tmp),summary=Path(tmp)/"summary.json")
            for name,score in scores.items():workflow.write(args.output/"scores"/f"{name}.json",score)
            workflow.write(args.output/"pilot-freeze.json",frozen)
            workflow.write(args.output/"ranking-diagnostics.json",{"contract":workflow.contract(),"systems":workflow.ranking_systems(scores,rows)})
            with patch.object(workflow,"plan_for",return_value=plan),patch.object(workflow,"endpoint_rows",return_value=rows),redirect_stdout(io.StringIO()):
                workflow.publish(args)
                workflow.publish(args,validate=True)
                self.assertEqual(workflow.read(args.summary)["status"],"not_ready_for_pilot")
                bad=deepcopy(scores[next(iter(scores))]);bad["rows"][0]["candidate_ids"][0]="wrong"
                with self.assertRaises(ValueError):workflow.validate_scores(bad,bad["cell"],rows)

    def test_live_callback_reparses_after_every_shot(self):
        plan,rows,scores=fixture();frozen=workflow.freeze_payload(plan,rows,scores)
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(output=Path(tmp),audit=Path(tmp)/"audit",device="cpu")
            workflow.write(args.output/"pilot-source-inventory.json",{"authored_birds":[3]*12})
            level=live.pilot_plan(frozen)["levels"][0]
            bridge=SimpleNamespace(screenshot=lambda:SimpleNamespace(width=2,height=2,rgb=bytes(12)))
            observed=[]
            def parse(**kwargs):
                carrier=torch.zeros(197);carrier[15]=len(observed)+1;observed.append(carrier)
                return PlanningObservation(kwargs["identity"],carrier,(0,),kwargs["slingshot_anchor"])
            adapter=SimpleNamespace(model=SimpleNamespace(object_vocabulary=("pig:1","block:1")),
                                    parser_checkpoint_identity="test-parser",from_agent_rgb=parse)
            def collect(slot,root,game,**kwargs):
                for i in range(3):kwargs["action_selector"]({"gameX":10,"gameY":20},bridge,i)
                return {"terminal_reason":"success","executed_action_count":3,
                    "trajectory_identity":"trajectory","scenario_lineage_identity":"lineage","level_instance_identity":"level"}
            with patch.object(workflow,"load_adapter",return_value=adapter),patch.object(live,"archive_details"),patch.object(live.capture,"_collect_lineage_attempt",side_effect=collect),patch.object(live,"audit_video",return_value=None),redirect_stdout(io.StringIO()):
                result=live.run_trial(args,frozen,level,"fixed_prior")
            self.assertTrue(result["success"])
            self.assertEqual([float(v[15]) for v in observed],[1.0,2.0,3.0])

    def test_pilot_publication_checks_paired_sources_actions_and_videos(self):
        plan,rows,scores=fixture();frozen=workflow.freeze_payload(plan,rows,scores)
        pilot=live.pilot_plan(frozen)
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(output=Path(tmp),audit=Path(tmp)/"audit")
            args.audit.mkdir()
            workflow.write(args.output/"pilot-freeze.json",frozen)
            workflow.write(args.output/"pilot-plan.json",pilot)
            sources={"seeds":[l["generation_seed"] for l in pilot["levels"]],
                "source_overlap_count":0,"authored_birds":[3]*12,
                "levels":[{"scenario_manifest":{"scenario_lineage":{"identity":f"lineage-{i}"},
                    "level_instance":{"identity":f"level-{i}"}}} for i in range(12)]}
            workflow.write(args.output/"pilot-source-inventory.json",sources)
            for level in pilot["levels"]:
                for system in live.SYSTEMS:
                    slot=live.slot_for({**level,"authored_birds":3},system)
                    name=slot["slot_identity"];root=args.output/"pilot"/name
                    action=workflow.old.probe.broad_action_candidates("x",workflow.old.probe._bounds())[0].action
                    success=system=="corrected_cem"
                    trajectory={"trajectory_identity":name,"executed_action_count":1,
                        "terminal_reason":"success" if success else "failure",
                        "shots":[{"shot_index":0,"action":{"interface_action":action.to_interface_action((10,20),workflow.old.probe._bounds())}}]}
                    workflow.write(root/"trajectory.json",trajectory)
                    workflow.write(root/"decision-1.json",{"action":asdict(action),"system":system})
                    (args.audit/f"{name}.webm").touch()
                    workflow.write(args.output/"pilot-results"/f"{name}.json",{
                        "slot":slot,"failure":None,"success":success,"shots":1,"trajectory_identity":name,
                        "scenario_lineage_identity":f"lineage-{level['ordinal']}",
                        "level_instance_identity":f"level-{level['ordinal']}","video":{"path":f"{name}.webm"},
                        "final_evaluation_opened":False})
            result=live.pilot_report(args)
            self.assertEqual(result["counts"]["corrected_cem"]["successes"],12)
            self.assertEqual(result["paired_comparisons"]["fixed_prior"]["mean_paired_success_difference"],1.0)
            (args.audit/f"{name}.webm").unlink()
            with self.assertRaises(ValueError):live.pilot_report(args)


if __name__=="__main__":unittest.main()
