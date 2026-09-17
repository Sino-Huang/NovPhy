"""Frozen retained-result adapter for the #76 bounded-transfer clear-hit metrics."""
from collections.abc import Mapping
import json
from pathlib import Path

from scripts.issue_76_native_outcomes import terminal_evidence
from world_model.training.event_ranking import STOP_KINDS


SCHEMA = 'issue_76_bounded_transfer_metric_inputs_v1'
REQUIRED_VIEWS = frozenset(('legacy-single', 'corrected-single', 'corrected-history'))
TERMINAL_KINDS = {
    'level_clear': 'native_clear',
    'level_fail': 'native_fail',
    'stable_entered': 'stable_without_clear',
}


def _read(value):
    if value is None or isinstance(value, Mapping):
        return value
    return json.loads(Path(value).read_text())


def _record_parts(record):
    if not isinstance(record, Mapping):
        raise TypeError('retained branch record must be a mapping')
    if 'result' not in record:
        return record, record.get('receipt'), record.get('paired_manifest'), record.get('terminal_evidence')
    result_source = record['result']
    result = _read(result_source)
    receipt = _read(record.get('receipt'))
    paired = _read(record.get('paired_manifest'))
    if paired is None and not isinstance(result_source, Mapping) and isinstance(result, Mapping):
        result_path = Path(result_source)
        candidate = (result_path.parent.parent / 'attempts' / result['member_identity'] /
                     'paired-inputs' / 'paired-inputs.json')
        if candidate.is_file():
            paired = _read(candidate)
    return result, receipt, paired, record.get('terminal_evidence')


def _receipt_valid(receipt, plan, identity):
    if not isinstance(receipt, Mapping):
        return False
    required = {
        'execution_within_limits',
        'stop',
        'technical_retries',
        'member_identity',
        'execution_plan_identity',
    }
    if not required.issubset(receipt) or not ({'worker_exitcode', 'exit_code'} & set(receipt)):
        return False
    exit_code = receipt['worker_exitcode'] if 'worker_exitcode' in receipt else receipt['exit_code']
    checks = (
        receipt['execution_within_limits'] is True,
        exit_code == 0,
        receipt['stop'] is None or receipt['stop'] is False,
        receipt['technical_retries'] == 0,
        receipt['member_identity'] == identity,
        receipt['execution_plan_identity'] == plan['identity'],
    )
    return all(checks)


def _operationally_valid(result, receipt, plan, identity):
    if not isinstance(result, Mapping) or result.get('member_identity') != identity:
        return False
    if (result.get('schema') != 'issue_76_bounded_transfer_capture_v1'
            or result.get('complete') is not True or result.get('failure') is not None
            or not _receipt_valid(receipt, plan, identity)):
        return False
    segments = result.get('segments')
    if not isinstance(segments, list) or len(segments) != 1:
        return False
    summary = segments[0].get('summary')
    return (isinstance(summary, Mapping)
            and summary.get('observed_window_valid') is True
            and isinstance(summary.get('first_fixed_step'), int)
            and isinstance(summary.get('last_fixed_step'), int)
            and summary['last_fixed_step'] >= summary['first_fixed_step'])


def _paired_views_exist(result, paired):
    if not isinstance(result, Mapping) or not result.get('paired_input_manifest'):
        return False
    if set(result.get('paired_input_views', ())) != REQUIRED_VIEWS:
        return False
    return (isinstance(paired, Mapping)
            and paired.get('schema') == 'issue_76_bounded_transfer_paired_inputs_v1'
            and paired.get('identity') == result['paired_input_manifest']
            and set(paired.get('views', ())) == REQUIRED_VIEWS
            and paired.get('same_native_decision_state') is True
            and paired.get('post_action_frames_used') is False)


def _stop_kind(result, supplied_terminal):
    segment = result['segments'][0]
    summary = segment['summary']
    if summary.get('censored') is True:
        if summary.get('terminal_observed') is not False:
            raise ValueError('right-censored branch contradicts terminal-observed status')
        return 'right_censored'
    if summary.get('terminal_observed') is not True:
        raise ValueError('non-censored branch lacks an observed terminal')
    terminal = supplied_terminal
    if terminal is None:
        terminal = segment.get('terminal_evidence')
    if terminal is None:
        terminal = terminal_evidence(segment)
    if (not isinstance(terminal, Mapping)
            or terminal.get('fixed_step') != summary.get('last_fixed_step')):
        raise ValueError('observed terminal is absent or differs from the intact window endpoint')
    try:
        return TERMINAL_KINDS[terminal['reason']]
    except KeyError as error:
        raise ValueError('observed terminal reason is outside the frozen four-kind label contract') from error


def adapt_retained_results(plan, retained_records):
    """Convert retained branch results/receipts into frozen event-metric inputs."""
    members = {member['identity']: member for member in plan['members']
               if member['exposure_role'] == 'model_selection'}
    branches = [branch for branch in plan['branches']
                if branch['source_member_identity'] in members]
    supplied = {}
    records = retained_records.values() if isinstance(retained_records, Mapping) else retained_records
    for retained in records:
        result, receipt, paired, terminal = _record_parts(retained)
        if not isinstance(result, Mapping) or not isinstance(result.get('member_identity'), str):
            raise ValueError('retained branch result lacks its member identity')
        identity = result['member_identity']
        if identity in supplied:
            raise ValueError('duplicate retained bounded-transfer branch result')
        supplied[identity] = (result, receipt, paired, terminal)
    assigned = {branch['identity'] for branch in branches}
    foreign = set(supplied) - assigned
    if foreign:
        raise ValueError(f'retained result is outside the held-out branch inventory: {sorted(foreign)[0]}')

    arms = plan['arm_contract']['arms']
    rows, labels, label_indices = [], [], []
    for branch in branches:
        member = members[branch['source_member_identity']]
        retained = supplied.get(branch['identity'])
        result, receipt, paired, terminal = retained if retained is not None else (None, None, None, None)
        valid = _operationally_valid(result, receipt, plan, branch['identity'])
        branch_paired_admissible = valid and _paired_views_exist(result, paired)
        stop_kind = _stop_kind(result, terminal) if valid else None
        if stop_kind is not None and stop_kind not in STOP_KINDS:
            raise ValueError('adapter produced a label outside the frozen four stop kinds')
        row = {
            'member_identity': branch['identity'],
            'base_cluster': member['base_cluster'],
            'exposure_role': member['exposure_role'],
            'family': member['generator_family'],
            'group_index': member['ordinal'],
            'candidate_ordinal': branch['candidate_ordinal'],
            'valid': valid,
            'branch_paired_ranking_admissible': branch_paired_admissible,
            'paired_ranking_admissible': branch_paired_admissible,
            'paired_ranking_admissible_by_arm': {
                arm: branch_paired_admissible for arm in arms},
            'stop_kind': stop_kind,
        }
        rows.append(row)
        labels.append(stop_kind)
        label_indices.append(None if stop_kind is None else STOP_KINDS.index(stop_kind))

    groups = []
    for member in sorted(members.values(), key=lambda item: item['ordinal']):
        indices = [index for index, row in enumerate(rows)
                   if row['group_index'] == member['ordinal']]
        expected = sum(branch['source_member_identity'] == member['identity'] for branch in branches)
        lineage_admissible = len(indices) == expected and expected > 0 and all(
            rows[index]['branch_paired_ranking_admissible'] for index in indices)
        for index in indices:
            rows[index]['lineage_admissible'] = lineage_admissible
            rows[index]['paired_ranking_admissible'] = lineage_admissible
            rows[index]['paired_ranking_admissible_by_arm'] = {
                arm: lineage_admissible for arm in arms}
        informative = lineage_admissible and any(labels[index] == 'native_clear' for index in indices)
        groups.append({
            'group_index': member['ordinal'],
            'base_cluster': member['base_cluster'],
            'family': member['generator_family'],
            'row_indices': indices,
            'lineage_admissible': lineage_admissible,
            'informative': informative,
            'observed_stop_kinds': [labels[index] for index in indices if labels[index] is not None],
        })
    informative_count = sum(group['informative'] for group in groups)
    floor = plan['decision_rule']['minimum_informative_held_out_groups']
    return {
        'schema': SCHEMA,
        'plan_identity': plan['identity'],
        'rows': rows,
        'labels': labels,
        'label_indices': label_indices,
        'groups': groups,
        'informative_group_count': informative_count,
        'informative_group_floor': floor,
        'informative_group_floor_met': informative_count >= floor,
        'stable_and_censored_labels_preserved': True,
        'missing_view_invalidates_every_arm': True,
    }


metric_inputs = adapt_retained_results
