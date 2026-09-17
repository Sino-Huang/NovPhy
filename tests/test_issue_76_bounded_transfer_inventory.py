from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts import prepare_issue_76_bounded_transfer as inventory
from scripts import issue_76_bounded_transfer_frontier as frontier
from scripts import issue_76_bounded_transfer_metrics as metrics
from scripts import run_issue_76_event_model as event_model


class BoundedTransferInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = inventory.make_plan()

    def fixture(self, directory):
        path = Path(directory) / 'validation.json'
        path.write_text(json.dumps({
            'plan_identity': 'issue-76-shared-history-smoke-v4',
            'validated': True,
            'individual_timing_and_rgb_passed': True,
        }))
        return path

    def test_plan_is_deterministic(self):
        self.assertEqual(self.plan, inventory.make_plan())

    def test_sources_are_mandatory_complete_and_fail_closed(self):
        required = {
            inventory.PROTOCOL,
            'scripts/issue_76_bounded_transfer_capture.py',
            'scripts/issue_76_bounded_transfer_metrics.py',
            'scripts/issue_76_bounded_transfer_frontier.py',
            'tests/test_issue_76_bounded_transfer_frontier.py',
            'tests/test_issue_76_bounded_transfer_capture.py',
            'docs/issue-76-bounded-transfer-capture-mechanism.md',
            'world_model/training/action_event_readout.py',
            'world_model/training/event_ranking.py',
            'scripts/run_issue_76_event_model.py',
            'scripts/run_issue_76_native_refit.py',
            'scripts/issue_76_matched_batches.py',
        }
        self.assertTrue(required.issubset(inventory.SOURCES))
        self.assertEqual(set(self.plan['source_text']), set(inventory.SOURCES))
        self.assertIs(self.plan['source_freeze']['mandatory_and_fail_closed'], True)
        self.assertEqual(self.plan['metric_binding']['adapter_source'],
                         'scripts/issue_76_bounded_transfer_metrics.py')
        self.assertEqual(self.plan['metric_binding']['adapter_entrypoint'], 'adapt_retained_results')
        self.assertEqual(self.plan['metric_binding']['frontier_source'],
                         'scripts/issue_76_bounded_transfer_frontier.py')
        self.assertEqual(self.plan['metric_binding']['frontier_entrypoint'], 'physical_timeline')
        self.assertEqual(self.plan['metric_binding']['frontier_contract_entrypoint'], 'frozen_contract')
        self.assertEqual(self.plan['metric_binding']['frontier_entrypoints']['settled_state'],
                         'settled_state')
        self.assertEqual(self.plan['metric_binding']['frontier_contract'], frontier.frozen_contract())
        self.assertEqual(self.plan['metric_binding']['frontier_contract']['checkpoints_seconds'],
                         [0.5, 1.0, 2.0, 4.5, 8.0, 12.0])
        self.assertEqual(
            self.plan['metric_binding']['frontier_contract']['settlement'], {
                'linear_speed_threshold_unity_units_per_second': 0.01,
                'absolute_angular_speed_threshold_degrees_per_second': 0.01,
                'continuous_hold_seconds': 0.5,
                'statuses': ['settled_at', 'not_settled_censored', 'unavailable'],
                'source': ('tasks/task_template_designer/Assets/Scripts/GroundTruth/'
                           'PhysicalSnapshotRuntime.cs:330-349'),
            })
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'mandatory bounded-transfer frozen source is missing'):
                inventory.freeze_source_text(Path(directory), (inventory.PROTOCOL,))

    def test_exact_membership_ranks_and_roles_per_family(self):
        members = inventory.lineage_assignments()
        self.assertEqual(len(members), 100)
        self.assertEqual([member['ordinal'] for member in members], list(range(1, 101)))
        expected = {
            'readout_train': set(range(20)),
            'selector_train': set(range(75, 85)),
            'held_out_model_selection': set(range(120, 140)),
        }
        for family in inventory.FAMILIES:
            family_members = [member for member in members if member['generator_family'] == family]
            self.assertEqual(len(family_members), 50)
            self.assertEqual(Counter(member['study_role'] for member in family_members),
                             {'readout_train': 20, 'selector_train': 10,
                              'held_out_model_selection': 20})
            actual = {role: {member['family_rank'] for member in family_members
                             if member['study_role'] == role} for role in expected}
            self.assertEqual(actual, expected)
            self.assertEqual(sum(len(ranks) for ranks in actual.values()), len(set().union(*actual.values())))

    def test_seed_ranges_are_unique_and_disjoint_from_section_8(self):
        members = inventory.lineage_assignments()
        generation = {member['generation_seed'] for member in members}
        engine = {member['engine_seed'] for member in members}
        parked = set(range(761700001, 761700701)) | set(range(761800001, 761800701))
        self.assertEqual(generation, set(range(761900001, 761900101)))
        self.assertEqual(engine, set(range(762000001, 762000101)))
        self.assertFalse(generation & engine)
        self.assertFalse((generation | engine) & parked)

    def test_thirteen_actions_make_1300_branches_with_three_input_variants(self):
        members = inventory.lineage_assignments()
        branches = inventory.branch_assignments(members)
        self.assertEqual(len(inventory.candidate_actions()), 13)
        self.assertEqual(len(branches), 1300)
        self.assertEqual(Counter(branch['source_member_identity'] for branch in branches),
                         {member['identity']: 13 for member in members})
        self.assertTrue(all([variant['identity'] for variant in branch['input_variants']] ==
                            ['legacy-single', 'corrected-single', 'corrected-history']
                            for branch in branches))
        self.assertTrue(all(branch['input_variants'][2]['native_steps'] == [-100, -50, 0]
                            for branch in branches))
        actions = inventory.candidate_actions()
        self.assertEqual(len({(action['drag_x'], action['drag_y']) for action in actions}), 13)
        self.assertTrue(all(action['tap_time_ms'] == 0 and action['release_time_ms'] == 1000
                            for action in actions))

    def test_scenarios_are_bound_and_disjointness_rejects_collision(self):
        self.assertEqual(len(self.plan['members']), 100)
        for member in self.plan['members']:
            self.assertIn('template', member)
            self.assertTrue(member['generated_xml_identity'].startswith('xml-sha256:'))
            self.assertIn('<Level', member['xml'])
            self.assertIsInstance(member['scenario'], dict)
            self.assertTrue(member['scenario']['identity'].startswith('cohort-v2-scenario-manifest-v1:'))
            self.assertIn('scenario_manifest', member['scenario'])
            self.assertIn('generation', member['scenario']['scenario_manifest'])
        self.assertIs(self.plan['scenario_binding']['audit']['passed'], True)
        member = deepcopy(self.plan['members'][0])
        synthetic = [{
            'generation_or_reserved_seeds': [],
            'member_identities': [],
            'scenario_identities': [],
            'scenario_content_sha256': [inventory._digest(member['scenario'])],
            'xml_content_sha256': [],
        }]
        with self.assertRaisesRegex(ValueError, 'scenario overlaps prior inventory'):
            inventory.audit_disjointness([member], synthetic)

    def test_envelope_flags_and_release_semantics_are_frozen(self):
        self.assertEqual(self.plan['limits'], {
            'workers': 8,
            'worker_cpu_rss_mib': 4096,
            'aggregate_cpu_rss_mib': 32768,
            'combined_collection_seconds': 345600,
            'collection_wall_seconds': 345600,
            'attempt_seconds': 420,
            'shot_manifest_seconds': 180,
            'shot_manifest_deadline_seconds': 180,
            'shot_seconds': 180,
            'history_ready_seconds': 90,
            'technical_retries': 0,
            'minimum_free_bytes': 256 * 2**30,
            'artifact_bytes': 150 * 2**30,
            'decision_fixed_step': 30000,
            'native_step_seconds': 0.0004,
            'rgb_stride_native_steps': 50,
            'native_steps_per_shot_max': 30000,
            'predecision_frames': 3,
            'rgb_frames_per_shot_max': 601,
        })
        for flag in ('capture_execution_authorized', 'model_training_authorized',
                     'fresh_access', 'advancement_authorized'):
            self.assertIs(self.plan[flag], False)
        self.assertEqual(self.plan['action_contract']['recovered_fixed_release_delay_seconds'], 1.0)
        self.assertIn('one-second delay', self.plan['action_contract']['recovered_release_semantics'])
        self.assertIn('remains parked', self.plan['parked_section_8_note'])

    def test_arm_variant_matrix_metrics_and_thresholds_are_complete(self):
        contract = self.plan['arm_contract']
        expected = {(arm, variant) for arm in contract['arms'] for variant in contract['input_variants']}
        self.assertEqual({(cell['arm'], cell['input_variant']) for cell in contract['matrix']}, expected)
        self.assertEqual(contract['matrix_cells'], 15)
        self.assertEqual(contract['hybrid_fixed_policy_identity'], 'fixed-50-macro')
        self.assertEqual(contract['pure_fixed_policy_identity'], 'fixed-250-continuous')
        self.assertEqual(contract['no_model_identity'], 'training_family_prior')
        self.assertEqual(contract['no_model_smoothing_rule']['score'],
                         '(1 + native_clear_count) / (2 + valid_training_count)')
        model_cells = [cell for cell in contract['matrix'] if cell['predictor_kind']]
        self.assertEqual(len(model_cells), 9)
        self.assertTrue(all(cell['readout_updates_per_seed'] == 9000
                            and len(cell['training_by_seed']) == 3 for cell in model_cells))
        adaptive = [cell for cell in model_cells if cell['arm'] == 'hybrid-adaptive']
        fixed = [cell for cell in model_cells if cell['policy'] == 'fixed']
        self.assertTrue(all(cell['selector_updates'] == 2000
                            and cell['selection'] == 'trained adaptive selector' for cell in adaptive))
        self.assertTrue(all(cell['selector_updates'] == 0
                            and cell['selector_updates_per_seed'] == 0
                            and not cell['selector_retrained_from_scratch']
                            and cell['selector_batch_rng'] is None
                            and all(seed['selector_initialization_seed'] is None
                                    and seed['selector_update_rng_seed_formula'] is None
                                    for seed in cell['training_by_seed'])
                            and cell['selection'] == 'pinned fixed policy' for cell in fixed))
        controls = [cell for cell in contract['matrix']
                    if cell['arm'] in ('training_family_prior', 'candidate-0-original')]
        self.assertTrue(all(cell['input_invariant'] and cell['readout_updates'] == 0
                            and cell['selector_updates'] == 0
                            and 'duplicated identically' in cell['input_invariant_note']
                            for cell in controls))
        self.assertTrue(all(len(paths) == 3 for paths in contract['predictor_checkpoints'].values()))
        endpoint = self.plan['endpoint_and_label_contract']
        self.assertEqual(endpoint['predictor_endpoint_native_steps'], 11250)
        self.assertEqual(endpoint['observed_outcome_window_seconds'], 12)
        self.assertEqual(endpoint['stop_kinds'],
                         ['native_clear', 'native_fail', 'stable_without_clear', 'right_censored'])
        self.assertIsNone(endpoint['later_clear_by_deadline_label'])
        thresholds = self.plan['decision_rule']
        self.assertEqual(thresholds['minimum_informative_held_out_groups'], 10)
        self.assertEqual(thresholds['minimum_second_mode_fraction'], 0.05)
        self.assertEqual(thresholds['minimum_second_horizon_fraction'], 0.05)
        self.assertEqual(thresholds['maximum_hybrid_to_pure_linear_mac_ratio'], 1.1)

    def test_metric_adapter_freezes_validity_pairing_labels_and_informative_groups(self):
        member = next(member for member in self.plan['members']
                      if member['exposure_role'] == 'model_selection')
        branches = [branch for branch in self.plan['branches']
                    if branch['source_member_identity'] == member['identity']]
        records = []
        for branch in branches:
            ordinal = branch['candidate_ordinal']
            if ordinal == 0:
                reason, censored = 'level_clear', False
            elif ordinal == 1:
                reason, censored = 'stable_entered', False
            elif ordinal == 2:
                reason, censored = None, True
            else:
                reason, censored = 'level_fail', False
            last = 60000 if censored else 31000
            summary = {
                'first_fixed_step': 30000,
                'last_fixed_step': last,
                'observed_window_valid': True,
                'censored': censored,
                'terminal_observed': not censored,
            }
            views = ['legacy-single', 'corrected-single', 'corrected-history']
            if ordinal == 3:
                views.remove('corrected-single')
            result = {
                'schema': 'issue_76_bounded_transfer_capture_v1',
                'member_identity': branch['identity'],
                'complete': True,
                'failure': None,
                'paired_input_manifest': 'paired:' + branch['identity'],
                'paired_input_views': views,
                'segments': [{'summary': summary}],
            }
            receipt = {
                'member_identity': branch['identity'],
                'execution_plan_identity': self.plan['identity'],
                'execution_within_limits': True,
                'worker_exitcode': 0,
                'stop': False,
                'technical_retries': 0,
            }
            paired = {
                'schema': 'issue_76_bounded_transfer_paired_inputs_v1',
                'identity': result['paired_input_manifest'],
                'views': list(views),
                'same_native_decision_state': True,
                'post_action_frames_used': False,
            }
            terminal = None if censored else {'fixed_step': last, 'reason': reason}
            records.append({'result': result, 'receipt': receipt, 'paired_manifest': paired,
                            'terminal_evidence': terminal})

        adapted = metrics.adapt_retained_results(self.plan, records)
        group = next(group for group in adapted['groups'] if group['group_index'] == member['ordinal'])
        by_candidate = {row['candidate_ordinal']: row for row in adapted['rows']
                        if row['group_index'] == member['ordinal']}
        self.assertTrue(all(by_candidate[index]['valid'] for index in range(13)))
        self.assertFalse(by_candidate[3]['paired_ranking_admissible'])
        self.assertFalse(by_candidate[3]['branch_paired_ranking_admissible'])
        self.assertTrue(by_candidate[0]['branch_paired_ranking_admissible'])
        self.assertTrue(all(not row['paired_ranking_admissible']
                            for row in by_candidate.values()))
        self.assertTrue(all(value is False
                            for value in by_candidate[3]['paired_ranking_admissible_by_arm'].values()))
        self.assertEqual(by_candidate[1]['stop_kind'], 'stable_without_clear')
        self.assertEqual(by_candidate[2]['stop_kind'], 'right_censored')
        self.assertNotEqual(by_candidate[1]['stop_kind'], 'native_fail')
        self.assertNotEqual(by_candidate[2]['stop_kind'], 'native_fail')
        self.assertFalse(group['lineage_admissible'])
        self.assertFalse(group['informative'])

        complete = deepcopy(records)
        complete[3]['result']['paired_input_views'].append('corrected-single')
        complete[3]['paired_manifest']['views'].append('corrected-single')
        adapted = metrics.adapt_retained_results(self.plan, complete)
        group = next(group for group in adapted['groups'] if group['group_index'] == member['ordinal'])
        self.assertTrue(group['lineage_admissible'])
        self.assertTrue(group['informative'])
        self.assertEqual(adapted['informative_group_count'], 1)
        self.assertEqual(adapted['informative_group_floor'], 10)
        self.assertFalse(adapted['informative_group_floor_met'])

    def test_incomplete_lineage_is_excluded_from_frozen_clear_hit_consumer(self):
        member = next(member for member in self.plan['members']
                      if member['exposure_role'] == 'model_selection')
        branches = [branch for branch in self.plan['branches']
                    if branch['source_member_identity'] == member['identity']]
        records = []
        for branch in branches:
            identity = branch['identity']
            views = ['legacy-single', 'corrected-single', 'corrected-history']
            if branch['candidate_ordinal'] == 12:
                views.remove('corrected-history')
            manifest_identity = 'paired:' + identity
            records.append({
                'result': {
                    'schema': 'issue_76_bounded_transfer_capture_v1',
                    'member_identity': identity,
                    'complete': True,
                    'failure': None,
                    'paired_input_manifest': manifest_identity,
                    'paired_input_views': views,
                    'segments': [{'summary': {
                        'first_fixed_step': 30000,
                        'last_fixed_step': 31000,
                        'observed_window_valid': True,
                        'censored': False,
                        'terminal_observed': True,
                    }}],
                },
                'receipt': {
                    'member_identity': identity,
                    'execution_plan_identity': self.plan['identity'],
                    'execution_within_limits': True,
                    'worker_exitcode': 0,
                    'stop': False,
                    'technical_retries': 0,
                },
                'paired_manifest': {
                    'schema': 'issue_76_bounded_transfer_paired_inputs_v1',
                    'identity': manifest_identity,
                    'views': views,
                    'same_native_decision_state': True,
                    'post_action_frames_used': False,
                },
                'terminal_evidence': {'fixed_step': 31000, 'reason': 'level_fail'},
            })

        adapted = metrics.adapt_retained_results(self.plan, records)
        scores = torch.zeros(len(adapted['rows']))
        scored = event_model._metrics(
            adapted['rows'], adapted['label_indices'], scores, 'model_selection')
        self.assertEqual(scored['groups'], 0)

    def test_receipt_and_persisted_paired_manifest_are_required(self):
        identity = 'branch-1'
        result = {
            'paired_input_manifest': 'paired:branch-1',
            'paired_input_views': ['legacy-single', 'corrected-single', 'corrected-history'],
        }
        with self.subTest('empty receipt'):
            self.assertFalse(metrics._receipt_valid({}, self.plan, identity))
        with self.subTest('absent paired manifest'):
            self.assertFalse(metrics._paired_views_exist(result, None))

    def test_execution_count_is_physical_branches_not_variant_dispatches(self):
        counts = self.plan['counts']
        self.assertEqual(counts['physical_branch_executions'], 1300)
        self.assertEqual(counts['paired_input_views_per_execution'], 3)
        self.assertIn('never 3,900 dispatches', counts['dispatch_contract'])
        self.assertNotIn('variant_executions_if_authorized', counts)

    def test_player_binding_is_cached_v4_and_fails_closed(self):
        binding = self.plan['player_binding']
        self.assertEqual(binding['source_plan_identity'], 'issue-76-shared-history-smoke-v4')
        self.assertEqual(binding['player_build_origin'],
                         '/home/sukaih/.cache/novphy-shared-history-v3/player-build-01')
        self.assertEqual(binding['player_source']['schema'], 'issue_76_shared_history_player_source_v1')
        self.assertIn('Assets/Scripts/CanonicalCapture/PhysicsCaptureProtocol.cs',
                      binding['runtime_source_text'])
        self.assertIn('cached frozen v4 player',
                      binding['verified_request_72_divergence']['binding_decision'])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'frozen v4 player plan must exist'):
                inventory.player_binding(Path(directory) / 'missing-plan.json')

    def test_missing_prerequisite_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / 'missing-validation.json'
            with self.assertRaisesRegex(ValueError, 'v4 shared-history smoke validation.json must exist'):
                inventory.make_plan(missing)

    def test_controller_stop_labels_are_held_out_from_readout_fitting(self):
        members = inventory.lineage_assignments()
        readout = {member['identity'] for member in members
                   if member['fit_partition'] == 'propagator_and_readout'}
        controller = {member['identity'] for member in members if member['fit_partition'] == 'controller'}
        held_out = {member['identity'] for member in members if member['fit_partition'] is None}
        self.assertFalse(readout & controller or readout & held_out or controller & held_out)
        self.assertEqual((len(readout), len(controller), len(held_out)), (40, 20, 40))
        self.assertIs(self.plan['role_contract']['controller_stop_labels_excluded_from_readout_fit'], True)


if __name__ == '__main__':
    unittest.main()
