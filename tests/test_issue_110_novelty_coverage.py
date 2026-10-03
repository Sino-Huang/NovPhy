"""Issue #110: representation contract v2 (force-aware labels, action contract) and paired roster screening."""
import unittest
import xml.etree.ElementTree as ET

from scripts import prepare_novelty_coverage as cov
from world_model.data import novelty_contract as c


def body(slot, kind, position, velocity=(0.0, 0.0), **extra):
    return c.Body(slot=slot, kind=kind, position=position, velocity=velocity, **extra)


class VocabularyTests(unittest.TestCase):
    def test_v2_extends_v1_without_reordering_its_slots(self):
        self.assertEqual(len(c.VOCABULARY_V2), 28)
        self.assertTrue(set(c.VOCABULARY_V1) <= set(c.VOCABULARY_V2))
        positions = [c.VOCABULARY_V2.index(slot) for slot in c.VOCABULARY_V1]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(c.CARRIER_DIM_V1, 236)
        self.assertEqual(c.CARRIER_DIM_V2, 2 + 4 + 13 * 28)


class ActionContractTests(unittest.TestCase):
    def test_right_slingshot_pulls_right(self):
        left = {'drag_x': -80, 'drag_y': 30, 'tap_time_ms': 0, 'release_time_ms': 1000}
        right = c.action_for_side(left, c.slingshot_side(12.0))
        self.assertEqual(right['drag_x'], 80)
        self.assertTrue(c.action_within_v2(left, 'left'))
        self.assertTrue(c.action_within_v2(right, 'right'))
        self.assertFalse(c.action_within_v2(left, 'right'))
        self.assertFalse(c.action_within_v2({**left, 'drag_x': -5}, 'left'))

    def test_grid_candidates_cover_both_sides_with_shared_suffixes(self):
        candidates = cov.candidates()
        left, right = candidates['by_slingshot_side']['left'], candidates['by_slingshot_side']['right']
        self.assertEqual(len(left), 16)
        self.assertEqual([item['suffix'] for item in left], [item['suffix'] for item in right])
        for a, b in zip(left, right):
            self.assertEqual(b['action'], {**a['action'], 'drag_x': -a['action']['drag_x']})


class ForceLawTests(unittest.TestCase):
    def turbulence(self, kind='NovelAirTurbulence', rotation=0.0):
        return c.Source('novelty:0000', kind, (0.0, 0.0), rotation, (7.0, 10.0))

    def test_turbulence_pushes_moving_bodies_inside_its_box(self):
        moving = body('block:0000', 'block', (2.0, 3.0), (0.5, 0.0))
        self.assertEqual(c.force_on(self.turbulence(), moving), (0.0, 4.0))
        self.assertEqual(c.force_on(self.turbulence('InverseAirTurbulence'), moving), (0.0, -10.0))
        self.assertEqual(c.force_on(self.turbulence('NonNovelAirTurbulence'), moving), (0.0, 1.5))

    def test_turbulence_ignores_resting_excluded_and_outside_bodies(self):
        source = self.turbulence()
        self.assertIsNone(c.force_on(source, body('block:0000', 'block', (2.0, 3.0), (0.05, 0.0))))
        self.assertIsNone(c.force_on(source, body('pig:0000', 'pig', (0.0, 0.0), (1.0, 0.0))))
        self.assertIsNone(c.force_on(source, body('platform:0000', 'platform', (0.0, 0.0), (1.0, 0.0))))
        self.assertIsNone(c.force_on(source, body('bird:0000', 'bird', (2.3, 0.0), (1.0, 0.0))))  # box half-width 2.24

    def test_rotated_turbulence_rotates_box_and_force(self):
        source = self.turbulence(rotation=90.0)
        inside = body('bird:0000', 'bird', (-3.0, 0.0), (1.0, 0.0))  # local y = 3 < 3.2 after rotation
        force = c.force_on(source, inside)
        self.assertAlmostEqual(force[0], -4.0)
        self.assertAlmostEqual(force[1], 0.0, places=9)

    def test_fan_pushes_only_the_launched_bird_on_its_left(self):
        fan = c.Source('novelty:0000', 'Fan', (0.0, 0.0))
        approaching = body('bird:0000', 'bird', (-1.0, 0.5), (5.0, 0.0), launched=True)
        self.assertEqual(c.force_on(fan, approaching), (-25.0, -0.0))
        self.assertIsNone(c.force_on(fan, body('bird:0000', 'bird', (1.0, 0.5), launched=True)))
        self.assertIsNone(c.force_on(fan, body('bird:0000', 'bird', (-1.0, 1.6), launched=True)))
        self.assertIsNone(c.force_on(fan, body('bird:0001', 'bird', (-1.0, 0.0))))
        self.assertIsNone(c.force_on(fan, body('block:0000', 'block', (-1.0, 0.0), (1.0, 0.0))))

    def test_magnet_repels_its_material_and_attracts_the_rest(self):
        magnet = c.Source('novelty:0000', 'Magnet', (0.0, 0.0), material='wood')
        wood = body('block:0000', 'block', (2.0, 0.0), material='wood')
        stone = body('block:0001', 'block', (2.0, 0.0), material='stone')
        bird = body('bird:0000', 'bird', (0.0, -1.0))
        self.assertEqual(c.force_on(magnet, wood), (30.0, -0.0))
        self.assertEqual(c.force_on(magnet, stone), (-30.0, 0.0))
        self.assertEqual(c.force_on(magnet, bird), (0.0, 15.0))
        self.assertIsNone(c.force_on(magnet, body('block:0002', 'block', (4.0, 0.0), material='stone')))
        self.assertIsNone(c.force_on(magnet, body('block:0001', 'block', (2.0, 0.0), material='stone',
                                                  touching_source=True)))
        self.assertIsNone(c.force_on(magnet, body('pig:0000', 'pig', (1.0, 0.0))))

    def test_storm_acts_after_onset_on_moving_blocks(self):
        storm = c.Source('novelty:0000', 'Storm', (0.0, 0.0))
        moving = body('block:0000', 'block', (5.0, 0.0), (0.2, 0.0))
        self.assertIsNone(c.force_on(storm, moving))
        self.assertEqual(c.force_on(storm, moving, storm_active=True), c.STORM_WIND)
        self.assertIsNone(c.force_on(storm, body('block:0001', 'block', (5.0, 0.0)), storm_active=True))
        self.assertEqual(c.force_edges(storm, [moving], storm_active=True), {('novelty:0000', 'block:0000')})
        self.assertEqual(c.force_edges(None, [moving]), set())

    def test_storm_onset_is_the_first_destroyed_bird(self):
        events = [
            {'event_type': 'entity_destroyed', 'fixed_step': 31000, 'participants': ['runtime:block:0001']},
            {'event_type': 'entity_destroyed', 'fixed_step': 39000, 'participants': ['runtime:bird:0001']},
            {'event_type': 'entity_destroyed', 'fixed_step': 36000, 'participants': ['runtime:bird:0000']},
            {'event_type': 'entity_death', 'fixed_step': 33000, 'participants': ['runtime:bird:0000']},
        ]
        birds = ['runtime:bird:0000', 'runtime:bird:0001']
        self.assertEqual(c.storm_onset_step(events, birds), 36000)
        self.assertIsNone(c.storm_onset_step(events[:1], birds))

    def test_gravity_and_window_labels(self):
        self.assertTrue(c.gravity_inverted((0.0, 6.0)))
        self.assertFalse(c.gravity_inverted((0.0, -9.8)))
        flags = [True, False, False, True, False]
        self.assertFalse(c.window_any(flags, 0, 2))  # frame t itself is excluded
        self.assertTrue(c.window_any(flags, 0, 3))
        with self.assertRaises(ValueError):
            c.window_any(flags, 0, 0)

    def test_residual_acceleration_removes_scaled_gravity(self):
        before = body('bird:0000', 'bird', (0.0, 0.0), (1.0, 0.0))
        after = body('bird:0000', 'bird', (0.0, 0.0), (1.0, -0.002))
        residual = c.residual_acceleration(before, after, (0.0, -9.8), 0.5, 0.0004)
        self.assertAlmostEqual(residual[1], -5.0 + 4.9)


class TraceAdapterTests(unittest.TestCase):
    def test_bodies_from_sample_marks_contacts_with_the_source(self):
        def entity(slot, position, velocity=(0.0, 0.0), present=True, lifecycle='active'):
            return {'entity_id': f'runtime:{slot}', 'scenario_object_id': slot, 'lifecycle': lifecycle,
                    'body_present': present,
                    'body': {'position': list(position), 'velocity': list(velocity)} if present else None}
        sample = {'entities': [entity('bird:0000', (1, 2), (3, 0)), entity('block:0000', (0, 1)),
                               entity('novelty:0000', (0, 0)), entity('block:0001', (9, 9), lifecycle='destroyed'),
                               entity('world:landscape:0000', (0, 0), present=False)],
                  'contacts': [{'entity_a_id': 'runtime:block:0000', 'entity_b_id': 'runtime:novelty:0000'}]}
        bodies = {b.slot: b for b in c.bodies_from_sample(sample, {'block:0000': 'stone'}, ['bird:0000'],
                                                          source_slot='novelty:0000')}
        self.assertEqual(set(bodies), {'bird:0000', 'block:0000', 'novelty:0000'})
        self.assertTrue(bodies['block:0000'].touching_source)
        self.assertEqual(bodies['block:0000'].material, 'stone')
        self.assertTrue(bodies['bird:0000'].launched)
        self.assertFalse(bodies['bird:0000'].touching_source)

    def test_source_from_scenario_reads_the_novelty_node(self):
        root = ET.fromstring('<Level><GameObjects><Novelty type="Magnet" material="" x="3.1" y="-1.9" rotation="0" '
                             'scenarioObjectId="novelty:0000"/></GameObjects></Level>')
        source = c.source_from_scenario(root)
        self.assertEqual((source.type, source.position, source.material, source.scale),
                         ('Magnet', (3.1, -1.9), 'wood', (1.0, 1.0)))
        self.assertIsNone(c.source_from_scenario(ET.fromstring('<Level><GameObjects/></Level>')))


class ReadinessTests(unittest.TestCase):
    SOURCES = {'capture_player': {'prefabs_present': {name: True for name in cov.PLAYER_PREFABS}},
               'pigs': {'BasicSmall': {'life': 1.0, 'distinct_bird_hits_to_die': None},
                        'BasicBig': {'life': 1e8, 'distinct_bird_hits_to_die': 3}}}

    def row(self, level, novelty, side='left', pig='BasicSmall', v1=1.0, v2=1.0):
        template = {'template': f'novelty_level_{level}/type0101{level:02}01/x.xml', 'novelty_level': level,
                    'family': f'type010{level}01', 'constraints': {'active_restrictions': None}}
        facts = {'pig_type': pig, 'novelty_type': novelty, 'birds': 1, 'bird_types': ['BirdRed'],
                 'slingshot_side': side}
        sweep = {'seeds': 200, 'materialization_failures': 0, 'fit_share_v1': v1, 'fit_share_v2': v2,
                 'max_slots_per_kind': {}}
        return cov.template_readiness(template, facts, sweep, self.SOURCES)

    def test_v2_resolves_each_v1_blocker(self):
        right = self.row(5, None, side='right')
        self.assertEqual(right['v1_reasons'], ['left_pull_only_action_contract'])
        self.assertEqual(right['v2_status'], 'supported')
        goal = self.row(7, 'InverseAirTurbulence', v1=0.0)
        self.assertIn('objective_undeclared', goal['v1_reasons'])
        self.assertIn('force_unrepresented', goal['v1_reasons'])
        self.assertEqual(goal['v2_status'], 'supported')
        self.assertEqual(goal['visibility'], 'visible_region_hidden_law')

    def test_v2_still_rejects_poor_slot_fit_and_flags_multi_hit_pigs(self):
        tight = self.row(2, 'Fan', v2=0.5)
        self.assertEqual(tight['v2_status'], 'unsupported')
        self.assertTrue(tight['v2_reasons'][0].startswith('slot_outside_v2_vocabulary'))
        self.assertTrue(self.row(0, None, pig='BasicBig')['decision_verdict_expectation'].startswith('degenerate'))


class PairedRosterTests(unittest.TestCase):
    def pair_list(self):
        def template(level, family):
            return {'template': f'novelty_level_{level}/{family}/x.xml', 'novelty_level': level, 'family': family}
        return [{'pair': 'pair-type010201', 'family': 'type010201', 'novelty_level': 2, 'scenario': 'single_force',
                 'normal': template(0, 'type010201'), 'novel': template(2, 'type010201')}]

    @staticmethod
    def fake(bad_seeds=()):
        def materializer(template, seed):
            extra = '<Block scenarioObjectId="block:0009"/>' if template['novelty_level'] and seed in bad_seeds else ''
            xml = (f'<Level><Birds><Bird type="BirdRed" scenarioObjectId="bird:0000"/></Birds>'
                   f'<Slingshot x="-12" scenarioObjectId="slingshot:0000"/><GameObjects>'
                   f'<Pig scenarioObjectId="pig:0000" seed="{seed}" level="{template["novelty_level"]}"/>{extra}'
                   f'</GameObjects></Level>')
            return xml, {'seed': seed, 'level': template['novelty_level']}
        return materializer

    def test_one_failing_side_rejects_the_paired_attempt(self):
        first_fit = cov.seeds('fit', 0, 1)[0]
        sizes = {'smoke': 1, 'fit': 2, 'evaluation': 1}
        rosters, rejected, bound = cov.build_rosters(self.pair_list(), [], sizes=sizes,
                                                     materializer=self.fake(bad_seeds={first_fit}))
        self.assertEqual([(row['split'], row['reason'], row['detail']['side']) for row in rejected],
                         [('fit', 'slot_outside_v2_vocabulary', 'novel')])
        fit = rosters['fit']['pair-type010201']
        self.assertEqual([level['ordinal'] for level in fit], [1, 2])
        self.assertNotIn(first_fit, [level['generation_seed'] for level in fit])
        for level in fit:
            self.assertEqual(level['normal']['identity'][:-4], level['novel']['identity'][:-4].replace('novel', 'normal'))
        self.assertEqual(len(bound), 2 * (1 + 2 + 1))
        cov.disjointness(rosters)

    def test_seed_blocks_avoid_every_reserved_range(self):
        reserved = cov.sc.reserved_ranges() + [
            {'name': b['name'], 'low': b['low'], 'high': b['high']}
            for b in cov.sc.files.read(cov.SEALED_PLAN)['seed_contract']['reserved_by_this_plan']]
        blocks = cov.seed_blocks([None] * 40)
        self.assertTrue(cov.sc.audit_seed_blocks(blocks, reserved)['passed'])
        with self.assertRaises(ValueError):
            cov.seeds('fit', 0, cov.PAIR_SEED_BLOCK)


if __name__ == '__main__':
    unittest.main()
