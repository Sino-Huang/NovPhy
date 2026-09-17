"""Pre-capture physical-frontier metric contract for #76 bounded transfer."""
from bisect import bisect_left
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path

from scripts.canonical_native_trace import FIXED_DELTA_SECONDS, NativeTrace
from world_model.training.event_ranking import STOP_KINDS


TIMELINE_SCHEMA = 'issue_76_bounded_transfer_physical_timeline_v1'
FRONTIER_SCHEMA = 'issue_76_bounded_transfer_frontier_v1'
CHECKPOINT_SECONDS = (0.5, 1.0, 2.0, 4.5, 8.0, 12.0)

# Unity's retained velocity unit is unity_unit/second.  The frozen player itself
# calls a dynamic body moving above sqrt(0.0001) = 0.01 non-stable
# (PhysicalSnapshotRuntime.cs ObserveV2Stability), so settlement adopts 0.01.
V_SETTLE_UNITY_UNITS_PER_SECOND = 0.01
W_SETTLE_DEGREES_PER_SECOND = 0.01
# Require a continuous half-second (1,250 native 0.0004-s steps), rather than a
# single low-velocity sample, before assigning an equilibrium checkpoint.
W_SETTLE_SECONDS = 0.5

# Regime metadata deliberately uses a higher threshold than equilibrium:
# 0.05 unity_unit/s separates sustained interaction-scale translation from the
# player's 0.01 stability boundary.  At least two dynamic bodies must exceed it
# in half the window before the no-new-contact collapse label is available.
COLLAPSE_MOTION_THRESHOLD = 0.05
COLLAPSE_SUSTAINED_FRACTION = 0.5
COLLAPSE_MINIMUM_MOVING_OBJECTS = 2
# A 0.001 unity_unit/s mean-speed decrease is the frozen minimum evidence that
# residual post-event motion is decaying rather than numerically flat.
SETTLING_DECAY_THRESHOLD = 0.001

REGIME_PRECEDENCE = (
    'collision_active',
    'collapse_interaction_active',
    'settling',
    'quiescent',
)
SETTLED_STATUSES = ('settled_at', 'not_settled_censored', 'unavailable')
COMPUTE_CATEGORIES = (
    'perception',
    'history_encoding',
    'predictor_transition',
    'symbolic_heads_adapters',
    'selector',
    'readout',
    'infilling',
    'action_scoring',
)
COMPUTE_FIELDS = ('macs', 'flop_proxy')
TRAINING_COMPUTE_FIELDS = ('macs', 'flop_proxy', 'wall_seconds', 'updates')


def frozen_contract():
    return {
        'schema': FRONTIER_SCHEMA,
        'checkpoints_seconds': list(CHECKPOINT_SECONDS),
        'settlement': {
            'linear_speed_threshold_unity_units_per_second': V_SETTLE_UNITY_UNITS_PER_SECOND,
            'absolute_angular_speed_threshold_degrees_per_second': W_SETTLE_DEGREES_PER_SECOND,
            'continuous_hold_seconds': W_SETTLE_SECONDS,
            'statuses': list(SETTLED_STATUSES),
            'source': ('tasks/task_template_designer/Assets/Scripts/GroundTruth/'
                       'PhysicalSnapshotRuntime.cs:330-349'),
        },
        'regimes': {
            'precedence': list(REGIME_PRECEDENCE),
            'collapse_motion_threshold_unity_units_per_second': COLLAPSE_MOTION_THRESHOLD,
            'collapse_sustained_fraction': COLLAPSE_SUSTAINED_FRACTION,
            'collapse_minimum_moving_objects': COLLAPSE_MINIMUM_MOVING_OBJECTS,
            'settling_decay_threshold_unity_units_per_second': SETTLING_DECAY_THRESHOLD,
        },
        'compute_categories': list(COMPUTE_CATEGORIES),
        'compute_fields': list(COMPUTE_FIELDS),
        'controller_oracle_regret': 'not_computed',
        'physical_timeline_source': 'canonical_native_trace_v1 retained at result.segments[0].native_root',
    }


def _read(value):
    return value if isinstance(value, Mapping) else json.loads(Path(value).read_text())


def physical_timeline(result_or_path):
    """Read and validate the already-retained native trace referenced by a result."""
    result = _read(result_or_path)
    if result.get('schema') != 'issue_76_bounded_transfer_capture_v1':
        raise ValueError('frontier requires the bounded-transfer capture result schema')
    segments = result.get('segments')
    if not isinstance(segments, list) or len(segments) != 1:
        raise ValueError('frontier requires exactly one retained physical segment')
    segment = segments[0]
    if not isinstance(segment, Mapping) or not segment.get('native_root'):
        raise ValueError('frontier result lacks its retained native physical trace path')
    trace = NativeTrace(segment['native_root'])
    summary = trace.validate()
    recorded = segment.get('summary')
    if not isinstance(recorded, Mapping) or any(
            recorded.get(key) != summary[key] for key in
            ('first_fixed_step', 'last_fixed_step', 'sample_count', 'event_counts', 'failure')):
        raise ValueError('frontier native trace differs from the retained segment summary')
    first = summary['first_fixed_step']
    samples, events = [], []
    for chunk in trace.chunks():
        for sample in chunk['fixed_step_samples']:
            samples.append({**sample,
                            'time_seconds': (sample['fixed_step'] - first) * FIXED_DELTA_SECONDS})
        for event in chunk['events']:
            events.append({**event,
                           'time_seconds': (event['fixed_step'] - first) * FIXED_DELTA_SECONDS})
    return {
        'schema': TIMELINE_SCHEMA,
        'member_identity': result['member_identity'],
        'fixed_delta_seconds': FIXED_DELTA_SECONDS,
        'observation_window_seconds': summary['physical_seconds'],
        'samples': samples,
        'events': events,
        'complete_raw_non_trigger_contacts': True,
    }


def _validate_timeline(timeline):
    if not isinstance(timeline, Mapping) or timeline.get('schema') != TIMELINE_SCHEMA:
        raise ValueError('unsupported bounded-transfer physical timeline schema')
    samples = timeline.get('samples')
    events = timeline.get('events')
    if not isinstance(samples, list) or not samples or not isinstance(events, list):
        raise ValueError('physical timeline lacks samples or event inventory')
    times = [sample.get('time_seconds') for sample in samples]
    if (any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in times)
            or times != sorted(times) or len(times) != len(set(times))):
        raise ValueError('physical timeline sample clock is malformed')
    window = timeline.get('observation_window_seconds')
    if not isinstance(window, (int, float)) or not math.isfinite(window) or window < 0:
        raise ValueError('physical timeline observation window is malformed')
    return samples, events, float(window)


def checkpoint_availability(timeline):
    _, _, window = _validate_timeline(timeline)
    return [{'checkpoint_seconds': checkpoint, 'available': window + 1e-12 >= checkpoint}
            for checkpoint in CHECKPOINT_SECONDS]


def _vector_speed(value):
    if isinstance(value, Mapping):
        pair = (value.get('x'), value.get('y'))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        pair = value
    else:
        raise ValueError('dynamic body velocity is unavailable')
    x, y = pair
    if (not isinstance(x, (int, float)) or not isinstance(y, (int, float))
            or not math.isfinite(x) or not math.isfinite(y)):
        raise ValueError('dynamic body velocity is unavailable')
    return math.hypot(x, y)


def _dynamic_speeds(sample):
    speeds = {}
    entities = sample.get('entities')
    if not isinstance(entities, list):
        raise ValueError('retained sample lacks its entity kinematics inventory')
    for entity in entities:
        body = entity.get('body')
        if bool(entity.get('body_present')) != (body is not None):
            raise ValueError('retained entity body availability is inconsistent')
        if body is None or entity.get('lifecycle') != 'active':
            continue
        body_type = body.get('body_type')
        if body_type is None:
            raise ValueError('retained body lacks its physics body type')
        if body_type.lower() != 'dynamic' or body.get('simulated', True) is not True:
            continue
        speeds[entity['entity_id']] = _vector_speed(body.get('velocity'))
    return speeds


def _dynamic_kinematics(sample):
    kinematics = {}
    speeds = _dynamic_speeds(sample)
    entities = {entity['entity_id']: entity for entity in sample['entities']}
    for identity, speed in speeds.items():
        angular = entities[identity]['body'].get('angular_velocity_degrees_per_second')
        if (not isinstance(angular, (int, float)) or isinstance(angular, bool)
                or not math.isfinite(angular)):
            raise ValueError('dynamic body angular velocity is unavailable')
        kinematics[identity] = (speed, abs(float(angular)))
    return kinematics


def settled_state(timeline):
    samples, _, window = _validate_timeline(timeline)
    try:
        motion_rows = [(float(sample['time_seconds']), _dynamic_kinematics(sample))
                       for sample in samples]
    except (KeyError, TypeError, ValueError):
        return {'status': 'unavailable', 'checkpoint_seconds': None,
                'reason': 'required_dynamic_object_kinematics_missing'}
    for checkpoint in CHECKPOINT_SECONDS:
        end = checkpoint + W_SETTLE_SECONDS
        if end > window + 1e-12:
            continue
        selected = [(time, motion) for time, motion in motion_rows if checkpoint <= time <= end]
        interval = [motion for _, motion in selected]
        if not interval:
            return {'status': 'unavailable', 'checkpoint_seconds': None,
                    'reason': 'settlement_hold_window_has_no_kinematic_samples'}
        cadence = timeline.get('fixed_delta_seconds')
        if (not isinstance(cadence, (int, float)) or cadence <= 0
                or selected[0][0] > checkpoint + cadence * 1.01
                or selected[-1][0] < end - cadence * 1.01
                or any(right[0] - left[0] > cadence * 1.01
                       for left, right in zip(selected, selected[1:]))):
            return {'status': 'unavailable', 'checkpoint_seconds': None,
                    'reason': 'settlement_hold_window_kinematics_are_not_continuous'}
        if all(all(speed <= V_SETTLE_UNITY_UNITS_PER_SECOND
                   and angular <= W_SETTLE_DEGREES_PER_SECOND
                   for speed, angular in motion.values())
               for motion in interval):
            return {'status': 'settled_at', 'checkpoint_seconds': checkpoint,
                    'linear_speed_threshold': V_SETTLE_UNITY_UNITS_PER_SECOND,
                    'absolute_angular_speed_threshold': W_SETTLE_DEGREES_PER_SECOND,
                    'continuous_hold_seconds': W_SETTLE_SECONDS}
    if window + 1e-12 < CHECKPOINT_SECONDS[-1]:
        return {'status': 'unavailable', 'checkpoint_seconds': None,
                'reason': 'settlement_censoring_checkpoint_not_observed'}
    return {'status': 'not_settled_censored', 'checkpoint_seconds': 12.0,
            'linear_speed_threshold': V_SETTLE_UNITY_UNITS_PER_SECOND,
            'absolute_angular_speed_threshold': W_SETTLE_DEGREES_PER_SECOND,
            'continuous_hold_seconds': W_SETTLE_SECONDS}


def _window_samples(samples, start, stop):
    return [sample for sample in samples
            if (start == 0 and sample['time_seconds'] == 0) or start < sample['time_seconds'] <= stop]


def _event_involves_dynamic(event, dynamic_ids):
    return bool(set(event.get('participants', ())) & dynamic_ids)


def _contact_pair(contact):
    return tuple(sorted((contact.get('entity_a_id'), contact.get('entity_b_id'))))


def regime_segments(timeline):
    samples, events, window = _validate_timeline(timeline)
    segments, start = [], 0.0
    for stop in CHECKPOINT_SECONDS:
        if stop > window + 1e-12:
            break
        current = _window_samples(samples, start, stop)
        if not current:
            raise ValueError('regime window has no retained physical samples')
        speed_rows = [_dynamic_speeds(sample) for sample in current]
        dynamic_ids = set().union(*(set(row) for row in speed_rows))
        previous_contacts = set()
        for sample in samples:
            if sample['time_seconds'] <= start:
                previous_contacts = {_contact_pair(contact) for contact in sample.get('contacts', ())}
            else:
                break
        contact_events = []
        for sample in current:
            pairs = {_contact_pair(contact) for contact in sample.get('contacts', ())}
            contact_events.extend(contact for contact in sample.get('contacts', ())
                                  if _contact_pair(contact) not in previous_contacts
                                  and (contact.get('entity_a_id') in dynamic_ids
                                       or contact.get('entity_b_id') in dynamic_ids))
            previous_contacts = pairs
        window_events = [event for event in events if start < event.get('time_seconds', -1) <= stop]
        collision_events = [event for event in window_events
                            if event.get('event_type', '').lower() == 'collision'
                            and _event_involves_dynamic(event, dynamic_ids)]
        moving_counts = [sum(speed >= COLLAPSE_MOTION_THRESHOLD for speed in row.values())
                         for row in speed_rows]
        sustained = (sum(count >= COLLAPSE_MINIMUM_MOVING_OBJECTS for count in moving_counts)
                     / len(moving_counts)) >= COLLAPSE_SUSTAINED_FRACTION
        flat = [speed for row in speed_rows for speed in row.values()]
        quiescent = all(speed < V_SETTLE_UNITY_UNITS_PER_SECOND for speed in flat)
        halves = max(1, len(speed_rows) // 2)
        first_speeds = [speed for row in speed_rows[:halves] for speed in row.values()]
        last_speeds = [speed for row in speed_rows[halves:] for speed in row.values()]
        first_mean = sum(first_speeds) / len(first_speeds) if first_speeds else 0.0
        last_mean = sum(last_speeds) / len(last_speeds) if last_speeds else first_mean
        prior_event = any(event.get('time_seconds', math.inf) <= start for event in events)
        decaying = first_mean - last_mean >= SETTLING_DECAY_THRESHOLD
        if contact_events or collision_events:
            label = 'collision_active'
        elif sustained:
            label = 'collapse_interaction_active'
        elif not quiescent and prior_event and decaying:
            label = 'settling'
        elif quiescent:
            label = 'quiescent'
        else:
            raise ValueError('retained motion does not satisfy a frozen exclusive regime definition')
        segments.append({'start_seconds': start, 'stop_seconds': stop, 'label': label,
                         'contact_event_count': len(contact_events),
                         'collision_event_count': len(collision_events)})
        start = stop
    return segments


def _typed_label(label):
    if label not in STOP_KINDS:
        raise ValueError('frontier error requires one unchanged typed stop-kind label')
    return label


def area_under_error_curve(errors_by_checkpoint, availability, typed_label):
    label = _typed_label(typed_label)
    available = {row['checkpoint_seconds']: row['available'] for row in availability}
    if set(available) != set(CHECKPOINT_SECONDS):
        raise ValueError('AUEC availability does not cover the frozen checkpoint grid')
    points = []
    for checkpoint in CHECKPOINT_SECONDS:
        if not available[checkpoint]:
            continue
        error = errors_by_checkpoint.get(checkpoint)
        if not isinstance(error, (int, float)) or not math.isfinite(error) or error < 0:
            raise ValueError('available AUEC checkpoint lacks a finite nonnegative error')
        points.append((checkpoint, float(error)))
    area = sum((right_t - left_t) * (left_e + right_e) / 2
               for (left_t, left_e), (right_t, right_e) in zip(points, points[1:]))
    return {'typed_label': label, 'points': [{'physical_seconds': t, 'error': error}
                                             for t, error in points],
            'auec': area, 'integrated_seconds': 0.0 if len(points) < 2 else points[-1][0] - points[0][0]}


def error_at_equal_cumulative_macs(error_curve, cumulative_mac_points, typed_label):
    label = _typed_label(typed_label)
    if not error_curve:
        raise ValueError('equal-MAC error curve is empty')
    macs = [row['cumulative_macs'] for row in error_curve]
    errors = [row['error'] for row in error_curve]
    if (macs != sorted(macs) or len(set(macs)) != len(macs)
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in macs + errors)):
        raise ValueError('equal-MAC source curve is malformed')
    values = []
    for target in cumulative_mac_points:
        index = bisect_left(macs, target)
        if index == len(macs) or (index == 0 and macs[0] != target):
            values.append({'cumulative_macs': target, 'available': False, 'error': None})
        elif macs[index] == target:
            values.append({'cumulative_macs': target, 'available': True, 'error': errors[index]})
        else:
            left_m, right_m = macs[index - 1], macs[index]
            weight = (target - left_m) / (right_m - left_m)
            values.append({'cumulative_macs': target, 'available': True,
                           'error': errors[index - 1] + weight * (errors[index] - errors[index - 1])})
    return {'typed_label': label, 'points': values}


def compute_ledger(decision_identity, components, wall_seconds, training_compute):
    if set(components) != set(COMPUTE_CATEGORIES):
        missing = sorted(set(COMPUTE_CATEGORIES) - set(components))
        extra = sorted(set(components) - set(COMPUTE_CATEGORIES))
        raise ValueError(f'compute ledger component inventory differs; missing={missing} extra={extra}')
    normalized = {}
    for category in COMPUTE_CATEGORIES:
        value = components[category]
        if set(value) != set(COMPUTE_FIELDS):
            raise ValueError(f'compute ledger {category} must report MAC and FLOP-proxy separately')
        if any(not isinstance(value[field], (int, float)) or not math.isfinite(value[field])
               or value[field] < 0 for field in COMPUTE_FIELDS):
            raise ValueError(f'compute ledger {category} has invalid counts')
        normalized[category] = {field: value[field] for field in COMPUTE_FIELDS}
    if (not isinstance(wall_seconds, (int, float)) or not math.isfinite(wall_seconds)
            or wall_seconds < 0):
        raise ValueError('compute ledger wall time is invalid')
    if set(training_compute) != set(TRAINING_COMPUTE_FIELDS):
        raise ValueError('training compute ledger is incomplete')
    if any(not isinstance(training_compute[field], (int, float))
           or not math.isfinite(training_compute[field]) or training_compute[field] < 0
           for field in TRAINING_COMPUTE_FIELDS):
        raise ValueError('training compute ledger has invalid counts')
    return {'schema': 'issue_76_bounded_transfer_compute_ledger_v1',
            'decision_identity': decision_identity, 'components': normalized,
            'wall_seconds': wall_seconds, 'training_compute': dict(training_compute),
            'multiplicative_composite': None}


def aggregate_compute_ledgers(ledgers):
    ledgers = list(ledgers)
    if not ledgers:
        raise ValueError('cannot aggregate an empty compute-ledger inventory')
    if any(ledger.get('schema') != 'issue_76_bounded_transfer_compute_ledger_v1'
           for ledger in ledgers):
        raise ValueError('compute-ledger schema mismatch')
    training_compute = ledgers[0].get('training_compute')
    if (not isinstance(training_compute, Mapping)
            or set(training_compute) != set(TRAINING_COMPUTE_FIELDS)
            or any(ledger.get('training_compute') != training_compute for ledger in ledgers)):
        raise ValueError('training compute must be one consistent arm/cell-level ledger')
    return {
        'decisions': len(ledgers),
        'components': {category: {field: sum(ledger['components'][category][field]
                                                   for ledger in ledgers)
                                  for field in COMPUTE_FIELDS}
                       for category in COMPUTE_CATEGORIES},
        'wall_seconds': sum(ledger['wall_seconds'] for ledger in ledgers),
        'training_compute': dict(training_compute),
        'multiplicative_composite': None,
    }


def controller_trace(decision_identity, requested, effective, ranking_affect):
    def pair(name, value):
        if (not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2
                or not isinstance(value[0], int) or not isinstance(value[1], str)):
            raise ValueError(f'controller {name} must be an integer-delta/string-abstraction pair')
        return {'delta_native_steps': value[0], 'abstraction': value[1]}
    if type(ranking_affect) is not bool:
        raise ValueError('controller ranking_affect must be boolean')
    return {'schema': 'issue_76_bounded_transfer_controller_trace_v1',
            'decision_identity': decision_identity,
            'requested': pair('requested', requested),
            'effective': pair('effective', effective),
            'ranking_affect': ranking_affect,
            'oracle_regret': {'status': 'not_computed',
                              'reason': 'campaign has no legitimate counterfactual oracle labels'}}


def physical_plausibility(timeline):
    samples, _, _ = _validate_timeline(timeline)
    contacts_present = all(isinstance(sample.get('contacts'), list) for sample in samples)
    entities_present = all(isinstance(sample.get('entities'), list) for sample in samples)
    contacts = [contact for sample in samples for contact in sample.get('contacts', ())]
    separations = [contact.get('separation') for contact in contacts]
    if (not contacts_present
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in separations)):
        penetration = {'status': 'not_available'}
    else:
        penetration = {'status': 'available',
                       'minimum_contact_separation': min(separations) if separations else None,
                       'negative_separation_contact_samples': sum(value < 0 for value in separations)}
    lifecycle = {'active': 0, 'inactive': 0, 'destroyed': 0}
    if not entities_present:
        lifecycle_value = {'status': 'not_available'}
    else:
        lifecycle_value = None
    for sample in samples:
        for entity in sample.get('entities', ()):
            state = entity.get('lifecycle')
            if state not in lifecycle:
                return {'penetration': penetration,
                        'contacts': ({'status': 'available', 'count': len(contacts)}
                                     if contacts_present else {'status': 'not_available'}),
                        'lifecycle': {'status': 'not_available'}, 'floating': {'status': 'not_available'}}
            lifecycle[state] += 1
    if lifecycle_value is None:
        lifecycle_value = {'status': 'available', 'sampled_entity_state_counts': lifecycle}
    return {
        'penetration': penetration,
        'contacts': ({'status': 'available', 'count': len(contacts)}
                     if contacts_present else {'status': 'not_available'}),
        'lifecycle': lifecycle_value,
        'floating': {'status': 'not_available',
                     'reason': 'retained schema has support/contact fields but no authoritative floating predicate'},
    }
