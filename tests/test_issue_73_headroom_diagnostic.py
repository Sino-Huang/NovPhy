import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from scripts import run_issue_73_headroom_diagnostic as r


def fixture():
    candidates=[];records=[]
    for i in range(12):
        cost=1 if i==4 else 1001
        candidates.append({'accepted':True,'realized_count_cost':cost})
        records.append({'status':'accepted','interaction_coverage':['pig_removed','level_clear'] if i==4 else [],
                        'endpoint_outcome':{'pig_contact':i==4,'block_contact':False,'support_change':False,
                                            'pig_displacement_world':0.,'block_displacement_world':0.},
                        'audit_manifest':None,'trajectory_relative_path':f'candidate-{i+1}'})
    row={'state':'fixture','candidates':candidates}
    scored={'arms':{name:[{'cost':float(i)} for i in range(12)] for name in r.matched.ARMS}}
    return row,scored,records


class HeadroomTests(unittest.TestCase):
    def test_available_good_action_missed_by_models(self):
        row,s,records=fixture();v=r.headroom_state(row,s,records,'fixed_ordinal_09')
        self.assertEqual(v['best_ordinal'],5)
        self.assertEqual(v['policies']['adaptive']['regret'],1.)
        self.assertEqual(v['policies']['adaptive']['missed_opportunities']['pig_removed'],1.)
        self.assertEqual(v['policies']['best_observed']['regret'],0.)
        self.assertAlmostEqual(v['policies']['uniform_random']['outcomes']['level_clear'],1/12)

    def test_no_grid_success_does_not_mean_no_useful_contact(self):
        row,s,records=fixture()
        for c in row['candidates']:c['realized_count_cost']=1001
        records[4]['interaction_coverage']=[]
        v=r.headroom_state(row,s,records,'fixed_ordinal_09')
        self.assertFalse(v['outcome_opportunities']['pig_removed'])
        self.assertTrue(v['outcome_opportunities']['pig_contact'])
        self.assertFalse(v['no_recorded_progress'])
        self.assertEqual(v['best_set_size'],12)

    def test_all_failed_replays_retained_not_a_perfect_oracle(self):
        row,s,records=fixture()
        for c in row['candidates']:c.update(accepted=False,realized_count_cost=1e9)
        for c in records:c['status']='failed'
        v=r.headroom_state(row,s,records,'fixed_ordinal_09')
        self.assertEqual(v['failed_candidates'],12)
        self.assertIsNone(v['best_ordinal'])
        self.assertEqual(v['policies']['best_observed']['regret'],1.)
        self.assertEqual(v['policies']['adaptive']['outcomes']['pig_removed'],0.)

    def test_uniform_prior_uses_expected_outcomes(self):
        row,s,records=fixture();v=r.headroom_state(row,s,records,'uniform_random_expected')
        self.assertEqual(v['policies']['prior'],v['policies']['uniform_random'])

    def test_nonbird_displacement_is_not_named_collapse(self):
        _,_,records=fixture();records[0]['endpoint_outcome']['block_displacement_world']=.1
        flags=r.flags(records[0]);self.assertTrue(flags['block_displacement'])
        self.assertNotIn('collapse',flags)

    def test_absolute_and_normalized_metrics_are_separate(self):
        row,s,records=fixture();v=r.headroom_state(row,s,records,'fixed_ordinal_09')
        self.assertEqual(v['policies']['adaptive']['excess_count_cost_over_grid_best'],1000.)
        result=r.aggregate_headroom([v])
        self.assertEqual(result['policies']['adaptive']['mean_count_cost'],1001)
        self.assertEqual(result['policies']['adaptive']['mean_regret'],1)

    def test_timing_finds_prelaunch_windows_and_late_outcome(self):
        def entity(kind,alive=True):
            return {'lifecycle':'active' if alive else 'destroyed','body_present':alive,
                    'scenario_object_id':kind+':0','entity_id':kind+':0'}
        samples=[{'fixed_step':1000+i,'entities':[entity('pig',i<250),entity('block')], 'colliders':[]} for i in range(301)]
        frames=[{'fixed_step':s['fixed_step'],'fixed_time_seconds':i*.02} for i,s in enumerate(samples)]
        events=[{'event_type':name,'fixed_step':1000+offset,'participants':['pig:0'],'payload':{}}
                for name,offset in (('bird_launched',150),('collision',230),('level_clear',250),('stable_entered',300))]
        cap={'capture_id':'fixture','fixed_step_samples':samples,'events':events,'pre_intervention_fixed_step':1000}
        action={'engine_relative_action':{'hold_milliseconds':600},'interface_action':{'releaseTime':600}}
        result=r.timing_summary(cap,{'frame_records':frames},action)
        self.assertEqual(result['launch_offset'],150)
        self.assertAlmostEqual(result['launch_seconds'],3.)
        self.assertTrue(result['first_60_entirely_prelaunch'])
        self.assertTrue(result['step225_differs_from_settled'])
        self.assertTrue(result['normal_50fps_compatible'])
        self.assertEqual(result['block_destroyed_event_count'],0)
        self.assertIn('unavailable',result['health_damage_evidence'])

    def test_gallery_is_bounded_and_contact_example_has_contact(self):
        row,s,records=fixture()
        for c in row['candidates']:c['realized_count_cost']=1001
        records[4]['interaction_coverage']=[]
        v=r.headroom_state(row,s,records,'fixed_ordinal_09')
        review=r.review_cases([v]*200)
        self.assertLessEqual(len(review),16)
        self.assertIn((0,9,'predeclared high-arc review'),review)
        self.assertIn((0,5,'contact without removal'),review)

    def test_fixed_horizon_observed_and_recursive_same_clock(self):
        class Add:
            def carrier(self,z,a,pair):return z+pair.delta
        z=torch.arange(61,dtype=torch.float32)[:,None].expand(-1,236)
        for h in (1,5,15):
            metrics=r.observed_and_recursive(Add(),z,torch.zeros(1,5),60,h)
            self.assertEqual(metrics,{'physical_endpoint':60,'local_mse':0.,'recursive_endpoint_mse':0.})

    def test_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'absent'
            with patch.object(r,'plan_payload',return_value={'definitions':r.definitions()}):
                self.assertEqual(r.main(['--dry-run','--output',str(path)]),0)
            self.assertFalse(path.exists())


if __name__=='__main__':unittest.main()
