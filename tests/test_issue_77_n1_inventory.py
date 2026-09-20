from collections import Counter
import unittest

from scripts import prepare_issue_77_n1 as inventory


class Issue77N1InventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = inventory.make_plan()

    def test_plan_is_deterministic(self):
        self.assertEqual(inventory.make_plan(), self.plan)

    def test_sources_are_mandatory_complete_and_fail_closed(self):
        required = {
            'scripts/prepare_issue_77_n1.py',
            'scripts/run_issue_77_n1.py',
            'tests/test_issue_77_n1_campaign.py',
            'tests/test_issue_77_n1_inventory.py',
            'docs/issue-77-novelty-experiments-plan.md',
            'scripts/issue_76_bounded_transfer_capture.py',
            'scripts/issue_76_shared_history_capture.py',
            'scripts/issue_76_expansion.py',
            'scripts/process_lifecycle.py',
            'src/webui/bridge.py',
            'world_model/training/native_history_data.py',
            'tasks/task_generator/canonical_materialization.py',
        }
        self.assertTrue(required.issubset(set(inventory.SOURCES)))
        self.assertEqual(set(self.plan['source_text']), set(inventory.SOURCES))
        self.assertTrue(self.plan['source_freeze']['mandatory_and_fail_closed'])
        with self.assertRaisesRegex(ValueError, 'mandatory issue-77 N1 frozen source is missing'):
            inventory.freeze_source_text(sources=('scripts/does-not-exist-issue-77.py',))

    def test_exact_membership_ordinals_and_roles(self):
        members = inventory.lineage_assignments()
        self.assertEqual(len(members), 16)
        self.assertEqual([row['ordinal'] for row in members], list(range(1, 17)))
        per_family = Counter(row['generator_family'] for row in members)
        self.assertEqual(per_family, {'type010103': 8, 'type010105': 8})
        self.assertEqual([row['identity'] for row in members],
                         [f'issue-77-n1-{ordinal:03}' for ordinal in range(1, 17)])
        controller = {row['ordinal'] for row in members if row['study_role'] == 'controller_only'}
        held_out = {row['ordinal'] for row in members if row['study_role'] == 'held_out_evaluation'}
        predictor = {row['ordinal'] for row in members if row['study_role'] == 'predictor_train'}
        self.assertEqual(controller, {5, 10, 15})
        self.assertEqual(held_out, {7, 8, 14, 16})
        self.assertEqual(predictor, {1, 2, 3, 4, 6, 9, 11, 12, 13})
        self.assertFalse(controller & held_out or controller & predictor or held_out & predictor)
        for row in members:
            if row['study_role'] == 'held_out_evaluation':
                self.assertEqual((row['exposure_role'], row['fit_partition']), ('calibration', None))
            elif row['study_role'] == 'controller_only':
                self.assertEqual((row['exposure_role'], row['fit_partition']), ('training', 'controller'))
            else:
                self.assertEqual((row['exposure_role'], row['fit_partition']), ('training', 'predictor'))
            self.assertEqual(row['novelty_level'], 0)

    def test_seed_ranges_are_unique_and_disjoint_from_reserved_ranges(self):
        members = inventory.lineage_assignments()
        generation = {row['generation_seed'] for row in members}
        engine = {row['engine_seed'] for row in members}
        self.assertEqual(generation, set(range(764000001, 764000017)))
        self.assertEqual(engine, set(range(764100001, 764100017)))
        self.assertFalse(generation & engine)
        for name, low, high in inventory.RESERVED_RANGES:
            self.assertFalse(any(low <= seed <= high for seed in generation | engine), name)
        self.assertTrue(self.plan['seed_contract']['all_ranges_disjoint'])

    def test_thirteen_actions_make_208_branches_with_three_input_variants(self):
        actions = inventory.candidate_actions()
        self.assertEqual(len(actions), 13)
        self.assertEqual(len({(a['drag_x'], a['drag_y']) for a in actions}), 13)
        self.assertTrue(all(a['tap_time_ms'] == 0 and a['release_time_ms'] == 1000 for a in actions))
        self.assertEqual((actions[0]['drag_x'], actions[0]['drag_y']), (-80, 10))
        branches = self.plan['branches']
        self.assertEqual(len(branches), 208)
        per_member = Counter(branch['source_member_identity'] for branch in branches)
        self.assertEqual(set(per_member.values()), {13})
        variants = inventory.input_variants()
        self.assertEqual([v['identity'] for v in variants],
                         ['legacy-single', 'corrected-single', 'corrected-history'])
        self.assertEqual(variants[2]['native_steps'], [-100, -50, 0])
        for branch in branches:
            self.assertEqual([v['identity'] for v in branch['input_variants']],
                             ['legacy-single', 'corrected-single', 'corrected-history'])
        counts = self.plan['counts']
        self.assertEqual(counts['physical_branch_executions'], 208)
        self.assertEqual(counts['paired_input_views_per_execution'], 3)
        self.assertIn('never 624 dispatches', counts['dispatch_contract'])
        self.assertEqual(counts['study_roles'], {
            'predictor_train': 9, 'controller_only': 3, 'held_out_evaluation': 4})

    def test_scenarios_are_bound_and_disjointness_rejects_collision(self):
        members = self.plan['members']
        self.assertEqual(len(members), 16)
        for member in members:
            self.assertTrue(member['generated_xml_identity'].startswith('xml-sha256:'))
            self.assertIn('<Level', member['xml'])
            self.assertTrue(member['scenario']['identity'].startswith('cohort-v2-scenario-manifest-v1:'))
            self.assertEqual(member['template']['template'],
                             inventory.EXPECTED_TEMPLATES[member['generator_family']])
            self.assertIs(member['template']['static_slot_action_fit'], True)
        self.assertTrue(self.plan['scenario_binding']['audit']['passed'])
        self.assertTrue(self.plan['scenario_binding']['template_source_inventory'].endswith(
            'issue-76-dynamics-diagnostic-v1/novelty-inventory.json'))
        member = members[0]
        prior = [{
            'path': 'fixture',
            'generation_or_reserved_seeds': [],
            'scenario_identities': [],
            'member_identities': [],
            'scenario_content_sha256': [inventory._digest(member['scenario'])],
            'xml_content_sha256': [],
        }]
        with self.assertRaisesRegex(ValueError, 'overlaps prior inventory'):
            inventory.audit_disjointness(members, prior)

    def test_envelope_limits_and_release_semantics_are_frozen(self):
        self.assertEqual(self.plan['limits'], {
            'workers': 8,
            'worker_cpu_rss_mib': 4096,
            'aggregate_cpu_rss_mib': 32768,
            'combined_collection_seconds': 302400,
            'collection_wall_seconds': 86400,
            'attempt_seconds': 1200,
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
        contract = self.plan['action_contract']
        self.assertEqual(contract['recovered_fixed_release_delay_seconds'], 1.0)
        self.assertIn('one-second delay', contract['recovered_release_semantics'])
        self.assertTrue(contract['one_shot_per_branch'])
        self.assertFalse(contract['tap_actions_authorized'])
        self.assertEqual(self.plan['endpoint_and_label_contract']['stop_kinds'],
                         ['native_clear', 'native_fail', 'stable_without_clear', 'right_censored'])
        self.assertTrue(self.plan['unsupported_cells_recorded_not_dropped'])

    def test_role_contract_matches_membership(self):
        contract = self.plan['role_contract']
        self.assertEqual(contract['controller_only_ordinals'], [5, 10, 15])
        self.assertEqual(contract['held_out_evaluation_ordinals'], [7, 8, 14, 16])
        self.assertEqual(contract['predictor_train_ordinals'], [1, 2, 3, 4, 6, 9, 11, 12, 13])
        self.assertTrue(contract['calibration_role_states_are_development_scoring_only'])
        self.assertTrue(contract['lineage_roles_are_disjoint_and_outcome_independent'])

    def test_player_binding_is_cached_v4_and_fails_closed(self):
        binding = self.plan['player_binding']
        self.assertEqual(binding['source_plan_identity'], 'issue-76-shared-history-smoke-v4')
        self.assertEqual(binding['player_source']['schema'], 'issue_76_shared_history_player_source_v1')
        self.assertIn('Assets/Scripts/CanonicalCapture/PhysicsCaptureProtocol.cs',
                      binding['runtime_source_text'])
        self.assertIn('cached frozen v4 player',
                      binding['verified_request_72_divergence']['binding_decision'])
        with self.assertRaisesRegex(ValueError, 'frozen v4 player plan must exist'):
            inventory.player_binding(inventory.PROJECT_ROOT / 'does-not-exist.json')

    def test_missing_prerequisite_fails_cleanly(self):
        with self.assertRaisesRegex(ValueError, 'v4 shared-history smoke validation.json must exist'):
            inventory.make_plan(validation_path=inventory.PROJECT_ROOT / 'does-not-exist.json')


if __name__ == '__main__':
    unittest.main()
