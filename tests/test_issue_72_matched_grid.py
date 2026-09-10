import argparse
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts import run_issue_72_matched_grid as r
from world_model.planning.task_objective import TaskObjective
from world_model.training.cnn_hybrid import CNNHybridPredictor, CarrierPairController
from world_model.model import PredictorConfig


def candidates():
    return [{"accepted": True, "realized_count_cost": value} for value in (1001, 1000, 1002)]


class MatchedGridTests(unittest.TestCase):
    def test_perception_checks_cache_batch_without_changing_deployment_carrier(self):
        from types import SimpleNamespace
        from world_model.data.deployment_temporal import AgentObservation
        cached = torch.zeros(236); cached[67] = .42885014
        deployed = cached.clone(); deployed[67] = .42502406
        observation = AgentObservation('anchor', 0, 0., b'agent-png', 'agent')
        class Adapter:
            def build(self, context):
                self.asserted_context = context
                return SimpleNamespace(tensor=deployed)
            def parse_batch(self, observations):
                self.observations = observations
                return ({'same_initial_image': True},)*len(observations)
            def build_from_parsed(self, context, current, prior):
                return SimpleNamespace(tensor=cached)
        adapter = Adapter()
        # State-4 reproduction: B=1 differs by .003826, B=3 matches exactly.
        result, receipt = r.parse_anchor(adapter, observation, cached, 'cpu')
        self.assertTrue(torch.equal(result, deployed))
        self.assertEqual(adapter.observations, (observation,)*3)
        self.assertAlmostEqual(receipt['cached_anchor_max_abs_difference'], .00382608, places=6)
        self.assertEqual(receipt['anchor_validation_max_abs_difference'], 0.)
        with patch.object(adapter, 'build_from_parsed', return_value=SimpleNamespace(tensor=cached+1)):
            with self.assertRaisesRegex(ValueError, 'frozen anchor'):
                r.parse_anchor(adapter, observation, cached, 'cpu')

    def test_successful_screen_never_authorizes_fresh_or_final(self):
        v = r.decision({"adaptive": .1, "corrected_fixed": .2, "teacher_forced": .3, "prior": .4},
                       {"1:continuous": 50, "15:macro": 50}, 80, 2., 0)
        self.assertTrue(v["development_passed"])
        self.assertEqual(v["status"], "fresh_protocol_required")
        self.assertIsNone(v["disposition"])
        self.assertFalse(v["issue_64_authorized"])
        self.assertFalse(v["fresh_evaluation_opened"])

    def test_bad_adaptation_stops_without_rescuing_with_training_gain(self):
        v = r.decision({"adaptive": .4, "corrected_fixed": .1, "teacher_forced": .5, "prior": .6},
                       {"1:continuous": 50, "15:macro": 50}, 80, 2., 0)
        self.assertFalse(v["development_passed"])
        self.assertEqual(v["disposition"], "readiness_or_precision_insufficient")

    def test_single_mode_cannot_pass(self):
        v = r.decision({"adaptive": .1, "corrected_fixed": .2, "teacher_forced": .3, "prior": .4},
                       {"1:macro": 50, "15:macro": 50}, 80, 2., 0)
        self.assertFalse(v["checks"]["mode_use"])

    def test_stronger_teacher_or_prior_not_ignored(self):
        v = r.decision({"adaptive": .1, "corrected_fixed": .3, "teacher_forced": .05, "prior": .4},
                       {"1:continuous": 50, "15:macro": 50}, 80, 2., 0)
        self.assertFalse(v["checks"]["adaptive_beats_strongest_declared_baseline"])

    def test_failures_retained_and_cannot_improve_regret_by_rescaling(self):
        rows = candidates()+[{"accepted": False, "realized_count_cost": 1e9}]
        regrets, informative = r.normalized_regrets(rows)
        self.assertEqual(regrets, [.5, 0., 1., 1.])
        self.assertTrue(informative)
        result = r.ranked([None, 0., 2., 3.], rows)
        self.assertEqual(result["regret"], 1.)
        self.assertTrue(result["prediction_failure"])

    def test_all_failed_replays_are_not_perfect_ties(self):
        result = r.ranked([0., 0.], [{"accepted": False, "realized_count_cost": 1e9}]*2)
        self.assertEqual(result["regret"], 1.)
        self.assertFalse(result["top1"])
        self.assertFalse(result["top3"])

    def test_ties_keep_original_ordinal(self):
        self.assertEqual(r.ranked([0., 0., 0.], candidates())["selected"], 0)

    def test_bootstrap_reproducible(self):
        self.assertEqual(r.paired_interval([.1, -.1, .3]), r.paired_interval([.1, -.1, .3]))

    def test_deployment_scorer_has_no_future_outcome_argument(self):
        self.assertEqual(list(inspect.signature(r.score_action).parameters),
                         ["model", "controller", "context", "action", "adaptive", "objective"])

    def test_same_endpoint_cost_and_charge_for_all_arms(self):
        torch.set_num_threads(2)
        model = CNNHybridPredictor(PredictorConfig(latent_dim=236, hidden_dim=16, depth=1, pair_code_dim=8))
        controller = CarrierPairController()
        z = torch.rand(236); objective = TaskObjective((8,), (3, 4))
        for adaptive in (False, True):
            value = r.score_action(model, controller, z, r.candidate_actions('fixture')[0], adaptive=adaptive, objective=objective)
            self.assertEqual(sum(t['effective_horizon'] for t in value['trace']), 225)
            self.assertEqual(value['cost'], objective(torch.tensor(value['endpoint'])))
            self.assertTrue(all(t['linear_macs'] > 0 for t in value['trace']))

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'absent'
            with patch.object(r, 'make_plan', return_value={'states': list(range(200))}):
                self.assertEqual(r.main(['--dry-run', '--output', str(output)]), 0)
            self.assertFalse(output.exists())

    def test_cached_trace_and_source_validation(self):
        import copy
        torch.set_num_threads(2)
        model = CNNHybridPredictor(PredictorConfig(latent_dim=236, hidden_dim=16, depth=1, pair_code_dim=8))
        vocabulary = ['block:'+str(i) for i in range(8)]+['pig:0']+['platform:'+str(i) for i in range(9)]
        objective = TaskObjective.from_vocabulary(vocabulary)
        row = {'state':'fixture', 'candidates':[{'identity':str(i),'action':a} for i,a in enumerate(r.candidate_actions('fixture'))]}
        plan = {'identity':'test-plan','checkpoints':{},'controller_identity':'fixture-controller','hybrid_contract':{'vocabulary':vocabulary}}
        value = r.score_action(model, None, torch.rand(236), row['candidates'][0]['action'], adaptive=False, objective=objective)
        record = {'schema':r.SCHEMA,'plan_identity':'test-plan','state':'fixture','checkpoints':{},
                  'controller_identity':'fixture-controller','arms':{}}
        for arm in r.ARMS:
            items = []
            for c in row['candidates']:
                v = copy.deepcopy(value); v.update(candidate_identity=c['identity'],action=c['action'])
                if arm == 'adaptive':
                    for t in v['trace']:
                        t['controller_calls']=1
                        t['linear_macs']=r.linear_macs(model,r.PAIRS[6],True)
                items.append(v)
            record['arms'][arm]=items
        models = {arm:model for arm in r.ARMS}
        r.check_state(record,row,plan,models)
        broken=copy.deepcopy(record);broken['arms']['adaptive'][0]['trace'][0]['effective_horizon']=1
        with self.assertRaises(ValueError):r.check_state(broken,row,plan,models)
        broken=copy.deepcopy(record);broken['arms']['teacher_forced'][0]['cost']+=1
        with self.assertRaises(ValueError):r.check_state(broken,row,plan,models)
        broken=copy.deepcopy(record);broken['arms']['corrected_fixed'].pop()
        with self.assertRaises(ValueError):r.check_state(broken,row,plan,models)

    def test_repair_receipt_preserves_original_freeze_and_checks_exact_source(self):
        import copy
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root/'run'
            script = root/r.RUNNER_SOURCE; script.parent.mkdir(parents=True)
            script.write_text('repaired source')
            args = argparse.Namespace(output=output, issue71=root/'71', parser=root/'70', device='cpu')
            old = {'schema':r.SCHEMA, 'identity':'fixture', 'contract':r.contract(),
                   'issue71':str(args.issue71.resolve()),'parser':str(args.parser.resolve()),
                   'code':{'source_text':{r.RUNNER_SOURCE:'original source'}}}
            r.write(output/'plan.json',old)
            original_bytes=(output/'plan.json').read_bytes()
            with patch.object(r,'ROOT',root), patch.object(r,'make_plan',return_value=copy.deepcopy(old)), \
                 patch.object(r,'load_models',return_value=({},None)):
                with self.assertRaisesRegex(ValueError,'source changed'):r.load_plan(args)
                r.repair_anchor_validation(args)
                self.assertEqual(r.load_plan(args),old)
                self.assertEqual((output/'plan.json').read_bytes(),original_bytes)
                r.repair_anchor_validation(args)
                script.write_text('another unapproved change')
                with self.assertRaisesRegex(ValueError,'source changed'):r.load_plan(args)

    def test_anchor_repair_cannot_change_scientific_gate(self):
        import copy
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            old={'identity':'fixture','contract':r.contract(),'code':{}}
            new=copy.deepcopy(old);new['contract']['screen']['minimum_regret_improvement']=0.
            r.write(output/'plan.json',old)
            with patch.object(r,'make_plan',return_value=new):
                with self.assertRaisesRegex(ValueError,'cannot change experiment settings'):
                    r.repair_anchor_validation(argparse.Namespace(output=output))


if __name__ == '__main__':
    unittest.main()
