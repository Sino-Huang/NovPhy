import unittest

from scripts import issue_76_bounded_transfer_frontier as frontier


def entity(identity, speed):
    return {
        'entity_id': identity,
        'scenario_object_id': identity.replace('runtime:', ''),
        'lifecycle': 'active',
        'body_present': True,
        'body': {
            'body_type': 'dynamic',
            'simulated': True,
            'velocity': [speed, 0.0],
            'angular_velocity_degrees_per_second': 0.0,
        },
    }


def timeline(window=12.0, speed=lambda _time, _index: 0.0, events=None):
    samples = []
    for ordinal in range(int(window * 10) + 1):
        time = ordinal / 10
        entities = [entity('runtime:block:0000', speed(time, 0)),
                    entity('runtime:block:0001', speed(time, 1))]
        samples.append({'time_seconds': time, 'entities': entities, 'contacts': []})
    return {
        'schema': frontier.TIMELINE_SCHEMA,
        'fixed_delta_seconds': 0.1,
        'observation_window_seconds': window,
        'samples': samples,
        'events': list(events or ()),
        'complete_raw_non_trigger_contacts': True,
    }


class BoundedTransferFrontierTests(unittest.TestCase):
    def test_settled_state_never_fabricates_equilibrium(self):
        moving = timeline(speed=lambda _time, _index: 0.02)
        self.assertEqual(frontier.settled_state(moving)['status'], 'not_settled_censored')
        missing = timeline()
        missing['samples'][5]['entities'][0]['body'].pop('velocity')
        self.assertEqual(frontier.settled_state(missing), {
            'status': 'unavailable',
            'checkpoint_seconds': None,
            'reason': 'required_dynamic_object_kinematics_missing',
        })

    def test_settled_state_requires_full_censoring_window_and_angular_kinematics(self):
        with self.subTest('early terminal'):
            early = timeline(window=4.5, speed=lambda _time, _index: 0.02)
            self.assertEqual(frontier.settled_state(early)['status'], 'unavailable')
        with self.subTest('rotating body'):
            rotating = timeline()
            for sample in rotating['samples']:
                sample['entities'][0]['body']['angular_velocity_degrees_per_second'] = 0.02
            self.assertEqual(frontier.settled_state(rotating)['status'], 'not_settled_censored')
        with self.subTest('missing angular velocity'):
            missing = timeline()
            missing['samples'][5]['entities'][0]['body'].pop(
                'angular_velocity_degrees_per_second')
            self.assertEqual(frontier.settled_state(missing)['status'], 'unavailable')

    def test_checkpoint_availability_uses_only_intact_observed_window(self):
        availability = frontier.checkpoint_availability(timeline(window=4.5))
        self.assertEqual([row['checkpoint_seconds'] for row in availability],
                         [0.5, 1.0, 2.0, 4.5, 8.0, 12.0])
        self.assertEqual([row['available'] for row in availability],
                         [True, True, True, True, False, False])

    def test_regime_precedence_is_exclusive(self):
        def speed(time, _index):
            if time <= 1.0:
                return 0.1
            if time <= 1.5:
                return 0.04
            if time <= 2.0:
                return 0.02
            return 0.0
        value = timeline(speed=speed, events=[{
            'event_id': 'collision-1',
            'event_type': 'collision',
            'time_seconds': 0.2,
            'participants': ['runtime:block:0000', 'runtime:block:0001'],
        }])
        value['samples'][2]['contacts'] = [{
            'contact_id': 'contact-1',
            'entity_a_id': 'runtime:block:0000',
            'entity_b_id': 'runtime:block:0001',
            'separation': -0.01,
        }]
        segments = frontier.regime_segments(value)
        self.assertEqual([row['label'] for row in segments[:4]], [
            'collision_active',
            'collapse_interaction_active',
            'settling',
            'quiescent',
        ])
        self.assertTrue(all(row['label'] in frontier.REGIME_PRECEDENCE for row in segments))

    def test_settling_event_must_precede_the_decay_interval(self):
        def decaying(time, _index):
            return 0.04 - 0.02 * (time / 0.5)

        end_event = timeline(window=0.5, speed=decaying, events=[{
            'event_id': 'late-event',
            'event_type': 'intervention',
            'time_seconds': 0.5,
            'participants': [],
        }])
        with self.assertRaisesRegex(ValueError, 'exclusive regime definition'):
            frontier.regime_segments(end_event)

        prior_event = timeline(window=0.5, speed=decaying, events=[{
            'event_id': 'prior-event',
            'event_type': 'intervention',
            'time_seconds': 0.0,
            'participants': [],
        }])
        self.assertEqual(frontier.regime_segments(prior_event)[0]['label'], 'settling')

    def test_auec_and_equal_mac_interpolation_preserve_typed_label(self):
        availability = frontier.checkpoint_availability(timeline())
        errors = {checkpoint: 2.0 for checkpoint in frontier.CHECKPOINT_SECONDS}
        result = frontier.area_under_error_curve(errors, availability, 'right_censored')
        self.assertEqual(result['typed_label'], 'right_censored')
        self.assertAlmostEqual(result['auec'], 23.0)
        equal = frontier.error_at_equal_cumulative_macs(
            [{'cumulative_macs': 10, 'error': 4.0},
             {'cumulative_macs': 30, 'error': 2.0}],
            [0, 10, 20, 30, 40], 'stable_without_clear')
        self.assertEqual(equal['typed_label'], 'stable_without_clear')
        self.assertEqual([row['error'] for row in equal['points']], [None, 4.0, 3.0, 2.0, None])

    def test_compute_ledger_is_componentized_and_omission_fails_closed(self):
        components = {category: {'macs': index + 1, 'flop_proxy': (index + 1) * 2}
                      for index, category in enumerate(frontier.COMPUTE_CATEGORIES)}
        training = {'macs': 0, 'flop_proxy': 0, 'wall_seconds': 0, 'updates': 0}
        ledger = frontier.compute_ledger('decision-1', components, 0.25, training)
        self.assertEqual(tuple(ledger['components']), frontier.COMPUTE_CATEGORIES)
        self.assertIsNone(ledger['multiplicative_composite'])
        incomplete = dict(components)
        incomplete.pop('readout')
        with self.assertRaisesRegex(ValueError, 'component inventory differs'):
            frontier.compute_ledger('decision-1', incomplete, 0.25, training)

    def test_training_compute_is_aggregated_once_per_arm_cell(self):
        components = {category: {'macs': 1, 'flop_proxy': 2}
                      for category in frontier.COMPUTE_CATEGORIES}
        training = {'macs': 100, 'flop_proxy': 200, 'wall_seconds': 3, 'updates': 4}
        ledgers = [frontier.compute_ledger(f'decision-{index}', components, 0.25, training)
                   for index in range(2)]
        aggregate = frontier.aggregate_compute_ledgers(ledgers)
        self.assertEqual(aggregate['training_compute'], training)

    def test_controller_trace_freezes_requested_effective_and_no_oracle_regret(self):
        trace = frontier.controller_trace('decision-1', (50, 'macro'), (250, 'continuous'), True)
        self.assertEqual(trace['requested'], {'delta_native_steps': 50, 'abstraction': 'macro'})
        self.assertEqual(trace['effective'], {'delta_native_steps': 250, 'abstraction': 'continuous'})
        self.assertIs(trace['ranking_affect'], True)
        self.assertEqual(trace['oracle_regret']['status'], 'not_computed')

    def test_physical_plausibility_marks_absent_fields_not_available(self):
        value = timeline()
        for sample in value['samples']:
            sample.pop('contacts')
        plausibility = frontier.physical_plausibility(value)
        self.assertEqual(plausibility['contacts']['status'], 'not_available')
        self.assertEqual(plausibility['penetration']['status'], 'not_available')
        self.assertEqual(plausibility['floating']['status'], 'not_available')
        self.assertEqual(plausibility['lifecycle']['status'], 'available')


if __name__ == '__main__':
    unittest.main()
