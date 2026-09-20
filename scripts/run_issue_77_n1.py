"""Run the frozen issue-77 N1 expansion capture campaign; never fit or score.

Small-project mode (plan §0): no authorization gates or audit cycles. The
supervisor keeps the R3 dispatch/ledger/continuation discipline so every
scheduled branch is executed or retained as a typed failure.
"""
import argparse
from collections.abc import Mapping
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path
import signal
import shutil
import time

from scripts import prepare_issue_77_n1 as frozen
from scripts.issue_76_native_outcomes import terminal_evidence
from scripts.issue_76_shared_player_storage import unique_file_bytes


OUTPUT = frozen.ROOT
PLAYER_SOURCE = frozen.PROJECT_ROOT / '.local-artifacts/issue-76-shared-history-smoke-v4/player'
PLAYER_FILES = ('9001-player.x86_64', '9001.x86_64', 'game_playing_interface.jar', 'serverbackup')
INFERENCE_HOLD_SECONDS = 2
ARTIFACT_MEASUREMENT_MISS_LIMIT = 10
TERMINAL_INCOMPLETE_STATUSES = {
    'minimum_free_storage', 'artifact_limit', 'collection_wall_limit',
    'combined_collection_limit', 'artifact_measurement_unavailable',
}
REQUIRED_VIEWS = frozenset(('legacy-single', 'corrected-single', 'corrected-history'))
TERMINAL_KINDS = {
    'level_clear': 'native_clear',
    'level_fail': 'native_fail',
    'stable_entered': 'stable_without_clear',
}


def _read_json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def _capture_limits(plan):
    return {**deepcopy(plan['limits']), 'inference_hold_seconds': INFERENCE_HOLD_SECONDS}


def _supervisor_sha256():
    return sha256(Path(__file__).read_bytes()).hexdigest()


def _artifact_bytes(output):
    try:
        return unique_file_bytes((output,))
    except FileNotFoundError:
        # Live workers atomically replace temporary observation trees while they are scanned.
        return None


def _worker_tools():
    from scripts.process_lifecycle import start_isolated_worker
    from scripts.run_issue_76_compatibility import process_rss, terminate_worker
    return start_isolated_worker, process_rss, terminate_worker


def _default_capture():
    from scripts.issue_76_bounded_transfer_capture import capture_one
    return capture_one


def _dispatch_record(member, branch):
    record = deepcopy(member)
    record.update(
        identity=branch['identity'],
        source_member_identity=branch['source_member_identity'],
        candidate_ordinal=branch['candidate_ordinal'],
        original_action_reference=branch['original_action_reference'],
        actions=[deepcopy(branch['action'])],
        input_variants=[deepcopy(value) for value in branch['input_variants']],
        maximum_shots=1,
    )
    return record


def _verify_player(root):
    missing = [name for name in PLAYER_FILES if not (Path(root) / name).is_file()]
    if missing:
        raise ValueError(f'issue-77 N1 player is incomplete: {missing[0]}')


def prepare(output=OUTPUT):
    output = Path(output)
    if not output.exists():
        plan = frozen.write_plan(output)
    else:
        plan = frozen.load_plan(output)

    _verify_player(PLAYER_SOURCE)
    player = output / 'player'
    if not player.exists():
        shutil.copytree(PLAYER_SOURCE, player)
    _verify_player(player)

    campaign_path = output / 'campaign.json'
    expected_limits = _capture_limits(plan)
    if not campaign_path.exists():
        _write_json(campaign_path, {
            'schema': 'issue_77_n1_campaign_v1',
            'plan_identity': plan['identity'],
            'inference_hold_seconds': INFERENCE_HOLD_SECONDS,
            'limits': expected_limits,
            'supervisor_sha256': _supervisor_sha256(),
        })
    else:
        campaign = _read_json(campaign_path)
        if (campaign.get('schema') != 'issue_77_n1_campaign_v1'
                or campaign.get('plan_identity') != plan['identity']
                or campaign.get('inference_hold_seconds') != INFERENCE_HOLD_SECONDS
                or campaign.get('limits') != expected_limits
                or campaign.get('supervisor_sha256') != _supervisor_sha256()):
            raise ValueError('issue-77 N1 campaign metadata differs from its frozen plan')
    return plan


def _campaign_metadata(output, plan, allow_digest_mismatch=False):
    campaign_path = Path(output) / 'campaign.json'
    if not campaign_path.is_file():
        raise ValueError('run prepare first')
    campaign = _read_json(campaign_path)
    current_digest = _supervisor_sha256()
    if (campaign.get('schema') != 'issue_77_n1_campaign_v1'
            or campaign.get('plan_identity') != plan['identity']
            or campaign.get('inference_hold_seconds') != INFERENCE_HOLD_SECONDS
            or campaign.get('limits') != _capture_limits(plan)):
        raise ValueError('issue-77 N1 campaign metadata differs from its frozen plan')
    if campaign.get('supervisor_sha256') != current_digest and not allow_digest_mismatch:
        raise ValueError('issue-77 N1 campaign metadata differs from its frozen plan')
    return campaign_path, campaign


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


def _stop_kind(result):
    segment = result['segments'][0]
    summary = segment['summary']
    if summary.get('censored') is True:
        if summary.get('terminal_observed') is not False:
            raise ValueError('right-censored branch contradicts terminal-observed status')
        return 'right_censored'
    if summary.get('terminal_observed') is not True:
        raise ValueError('non-censored branch lacks an observed terminal')
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


def _result_state(result, receipt, plan, identity):
    result = result if isinstance(result, Mapping) else {}
    receipt = receipt if isinstance(receipt, Mapping) else {}
    exit_code = receipt.get('worker_exitcode', receipt.get('exit_code'))
    stop = receipt.get('stop')
    checks = (
        ('missing_result_schema', 'schema' in result),
        ('result_schema', result.get('schema') == 'issue_76_bounded_transfer_capture_v1'),
        ('missing_result_member_identity', 'member_identity' in result),
        ('result_identity', result.get('member_identity') == identity),
        ('missing_result_complete', 'complete' in result),
        ('result_complete', result.get('complete') is True),
        ('missing_result_failure', 'failure' in result),
        ('result_failure', 'failure' in result and result['failure'] is None),
        ('missing_receipt_stop', 'stop' in receipt),
        ('receipt_stop', 'stop' in receipt and receipt['stop'] is None),
        ('missing_worker_exitcode', 'worker_exitcode' in receipt or 'exit_code' in receipt),
        ('worker_exitcode', exit_code == 0),
        ('missing_execution_within_limits', 'execution_within_limits' in receipt),
        ('execution_within_limits', ('execution_within_limits' in receipt
                                     and receipt['execution_within_limits'] is True)),
        ('missing_receipt_member_identity', 'member_identity' in receipt),
        ('receipt_identity', receipt.get('member_identity') == identity),
        ('missing_receipt_plan_identity', 'execution_plan_identity' in receipt),
        ('receipt_plan_identity', receipt.get('execution_plan_identity') == plan['identity']),
        ('missing_technical_retries', 'technical_retries' in receipt),
        ('technical_retries', receipt.get('technical_retries') == 0),
    )
    complete = all(passed for _, passed in checks)
    failure = result.get('failure')
    if not complete and failure is None:
        failure = stop or next(name for name, passed in checks if not passed)
    return {
        'status': 'complete' if complete else 'failed',
        'stop': stop,
        'exit_code': exit_code,
        'wall_seconds': receipt.get('wall_seconds', 0),
        'failure': failure,
    }


def _counts(entries):
    statuses = Counter(value['status'] for value in entries.values())
    attempted = statuses['complete'] + statuses['failed']
    return {
        'scheduled': len(entries),
        'attempted': attempted,
        'complete': statuses['complete'],
        'failed': statuses['failed'],
        'unattempted': statuses['unattempted'],
    }


def _ledger(plan, status, entries, collection_wall_seconds_elapsed, continuations=()):
    return {
        'schema': 'issue_77_n1_ledger_v1',
        'plan_identity': plan['identity'],
        'status': status,
        'collection_wall_seconds_elapsed': collection_wall_seconds_elapsed,
        'continuations': list(continuations),
        'counts': _counts(entries),
        'branches': entries,
    }


def _unattempted_entry():
    return {'status': 'unattempted', 'stop': None, 'exit_code': None,
            'wall_seconds': 0, 'failure': None}


def _load_retained_state(output, plan):
    entries = {branch['identity']: _unattempted_entry() for branch in plan['branches']}
    pending = []
    for branch in plan['branches']:
        identity = branch['identity']
        result_path = output / 'results' / f'{identity}.json'
        receipt_path = output / 'receipts' / f'{identity}.json'
        attempt_path = output / 'attempts' / identity
        marker_path = output / 'markers' / f'{identity}.json'
        if result_path.is_file() and receipt_path.is_file():
            entries[identity] = _result_state(
                _read_json(result_path), _read_json(receipt_path), plan, identity)
            continue
        if ((attempt_path.exists() or marker_path.exists())
                and not (result_path.is_file() and receipt_path.is_file())):
            raise ValueError('unclean issue-77 N1 attempt must be audited, never silently repeated')
        if result_path.exists() or receipt_path.exists():
            raise ValueError('incomplete issue-77 N1 retained record must be audited')
        pending.append(branch)
    return entries, pending


def _artifact_stop(output, limits, state):
    artifact_bytes = _artifact_bytes(output)
    if artifact_bytes is None:
        state['consecutive_misses'] += 1
        state['available'] = False
        if state['consecutive_misses'] >= ARTIFACT_MEASUREMENT_MISS_LIMIT:
            return 'artifact_measurement_unavailable'
        return None
    state['consecutive_misses'] = 0
    state['available'] = True
    if artifact_bytes > limits['artifact_bytes']:
        return 'artifact_limit'
    return None


def _final_artifact_stop(output, limits, state):
    while True:
        stop = _artifact_stop(output, limits, state)
        if stop is not None or state['available']:
            return stop


def _launch_stop(output, limits, artifact_state):
    if shutil.disk_usage(output).free < limits['minimum_free_bytes']:
        return 'minimum_free_storage'
    return _artifact_stop(output, limits, artifact_state)


def _fallback_result(identity, failure):
    return {
        'schema': 'issue_76_bounded_transfer_capture_v1',
        'member_identity': identity,
        'complete': False,
        'failure': failure,
        'segments': [],
    }


def _seal_spawn_failure(output, plan, branch, started, error, entries):
    identity = branch['identity']
    failure = f'worker_spawn_failure: {type(error).__name__}: {error}'
    result_path = output / 'results' / f'{identity}.json'
    if not result_path.is_file():
        _write_json(result_path, _fallback_result(identity, failure))
    receipt = {
        'member_identity': identity,
        'execution_plan_identity': plan['identity'],
        'execution_within_limits': False,
        'stop': 'worker_spawn_failure',
        'technical_retries': 0,
        'worker_exitcode': None,
        'wall_seconds': time.monotonic() - started,
        'peak_cpu_rss_mib': 0.0,
        'artifact_unique_inode_bytes': _artifact_bytes(output),
    }
    _write_json(output / 'receipts' / f'{identity}.json', receipt)
    entries[identity] = _result_state(_read_json(result_path), receipt, plan, identity)
    return receipt['wall_seconds']


def _finish_worker(output, plan, worker, entries, terminate_worker):
    process = worker['process']
    cleanup_error = None
    try:
        terminate_worker(process)
    except BaseException as error:
        cleanup_error = error
    wall_seconds = time.monotonic() - worker['started']
    exit_code = process.exitcode
    stop = worker['stop']
    identity = worker['branch']['identity']
    result_path = output / 'results' / f'{identity}.json'
    if not result_path.is_file():
        failure = worker.get('failure')
        if failure is None:
            if stop is not None:
                failure = stop
            elif exit_code != 0:
                failure = f'worker_exitcode={exit_code}'
            else:
                failure = 'worker_exitcode=0: result_missing'
        _write_json(result_path, _fallback_result(identity, failure))
    receipt = {
        'member_identity': identity,
        'execution_plan_identity': plan['identity'],
        'execution_within_limits': stop is None and exit_code == 0,
        'stop': stop,
        'technical_retries': 0,
        'worker_exitcode': exit_code,
        'wall_seconds': wall_seconds,
        'peak_cpu_rss_mib': worker['peak'],
        'artifact_unique_inode_bytes': _artifact_bytes(output),
    }
    receipt_path = output / 'receipts' / f'{identity}.json'
    _write_json(receipt_path, receipt)
    result = _read_json(result_path)
    entries[identity] = _result_state(result, receipt, plan, identity)
    return wall_seconds, cleanup_error


def _validate_complete_ledger(ledger, plan):
    counts = ledger.get('counts')
    branches = ledger.get('branches')
    expected = {branch['identity'] for branch in plan['branches']}
    scheduled = len(plan['branches'])
    if (ledger.get('schema') != 'issue_77_n1_ledger_v1'
            or ledger.get('plan_identity') != plan['identity']
            or not isinstance(counts, Mapping)
            or not isinstance(branches, Mapping)
            or set(branches) != expected
            or counts.get('scheduled') != scheduled
            or counts.get('attempted') != scheduled
            or counts.get('unattempted') != 0
            or counts.get('complete', 0) + counts.get('failed', 0) != scheduled
            or any(value.get('status') not in ('complete', 'failed')
                   for value in branches.values() if isinstance(value, Mapping))
            or any(not isinstance(value, Mapping) for value in branches.values())):
        raise ValueError('complete issue-77 N1 ledger is inconsistent with the plan inventory')


def _refuse_terminal_ledger(output, plan, continue_interrupted=False):
    ledger_path = Path(output) / 'ledger.json'
    if not ledger_path.is_file():
        return None
    ledger = _read_json(ledger_path)
    status = ledger.get('status')
    if status == 'complete':
        if continue_interrupted:
            raise ValueError('--continue-interrupted requires a retained interrupted ledger')
        _validate_complete_ledger(ledger, plan)
        return ledger
    if (ledger.get('schema') != 'issue_77_n1_ledger_v1'
            or ledger.get('plan_identity') != plan['identity']):
        raise ValueError('retained issue-77 N1 ledger identity is malformed')
    if status == 'running' or status in TERMINAL_INCOMPLETE_STATUSES:
        raise ValueError(f'issue-77 N1 ledger status {status!r} prohibits campaign top-up')
    if status == 'interrupted':
        if not continue_interrupted:
            raise ValueError("issue-77 N1 ledger status 'interrupted' requires --continue-interrupted")
        return ledger
    raise ValueError(f'issue-77 N1 ledger has unsupported retained status {status!r}')


def _continuation_identity(plan, ledger):
    head = {key: value for key, value in ledger.items() if key != 'continuations'}
    encoded = json.dumps({'plan_identity': plan['identity'], 'ledger': head},
                         sort_keys=True, separators=(',', ':')).encode('utf-8')
    return 'continuation-' + sha256(encoded).hexdigest()[:24]


def _reconcile_continuation(output, plan, campaign_path, campaign, ledger):
    revisions = campaign.setdefault('supervisor_revisions', [])
    continuations = ledger.setdefault('continuations', [])
    if (not isinstance(revisions, list) or not isinstance(continuations, list)
            or not all(isinstance(item, Mapping) for item in revisions + continuations)):
        raise ValueError('issue-77 N1 continuation history is malformed')
    identity = _continuation_identity(plan, ledger)
    campaign_matches = [item for item in revisions if item.get('identity') == identity]
    ledger_matches = [item for item in continuations if item.get('identity') == identity]
    if len(campaign_matches) > 1 or len(ledger_matches) > 1:
        raise ValueError('issue-77 N1 continuation event is duplicated')
    if campaign_matches and ledger_matches and campaign_matches[0] != ledger_matches[0]:
        raise ValueError('issue-77 N1 continuation event differs across campaign and ledger')
    event = (campaign_matches or ledger_matches or [None])[0]
    if event is None:
        event = {
            'identity': identity,
            'from_status': 'interrupted',
            'at_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'supervisor_sha256_previous': campaign.get('supervisor_sha256'),
            'supervisor_sha256_current': _supervisor_sha256(),
        }
    if (event.get('identity') != identity
            or event.get('from_status') != 'interrupted'
            or not isinstance(event.get('at_utc'), str) or not event['at_utc']
            or event.get('supervisor_sha256_current') != _supervisor_sha256()):
        raise ValueError('issue-77 N1 continuation event binding is malformed')
    if not campaign_matches:
        revisions.append(event)
    if not ledger_matches:
        continuations.append(event)
    campaign['supervisor_sha256'] = event['supervisor_sha256_current']
    # Campaign-first is intentional: a retry reconciles this same deterministic event into the ledger.
    _write_json(campaign_path, campaign)
    _write_json(Path(output) / 'ledger.json', ledger)
    return list(continuations)


def _time_stop(started, prior_collection_wall_seconds, combined, live, limits):
    now = time.monotonic()
    if prior_collection_wall_seconds + now - started >= limits['collection_wall_seconds']:
        return 'collection_wall_limit'
    live_wall = sum(now - worker['started'] for worker in live)
    if combined + live_wall >= limits['combined_collection_seconds']:
        return 'combined_collection_limit'
    return None


def _stop_live(live, reason):
    for worker in live:
        if worker['stop'] is None:
            worker['stop'] = reason


def _append_live(live, worker):
    live.append(worker)


def run(output=OUTPUT, capture=None, continue_interrupted=False):
    output = Path(output)
    plan = frozen.load_plan(output)
    retained_ledger = _refuse_terminal_ledger(output, plan, continue_interrupted)
    if retained_ledger is not None and retained_ledger.get('status') == 'complete':
        _campaign_metadata(output, plan)
        return retained_ledger
    is_continuation = bool(retained_ledger and retained_ledger.get('status') == 'interrupted')
    if continue_interrupted and not is_continuation:
        raise ValueError('--continue-interrupted requires a retained interrupted ledger')
    campaign_path, campaign = _campaign_metadata(
        output, plan, allow_digest_mismatch=is_continuation)
    limits = plan['limits']
    capture_limits = _capture_limits(plan)
    members = {member['identity']: member for member in plan['members']}
    # Audit every retained marker/result pair before recording continuation provenance.
    entries, pending = _load_retained_state(output, plan)
    ledger_path = output / 'ledger.json'
    if is_continuation:
        continuations = _reconcile_continuation(
            output, plan, campaign_path, campaign, retained_ledger)
    else:
        continuations = []
        _write_json(campaign_path, campaign)
    if capture is None:
        capture = _default_capture()
    start_isolated_worker, process_rss, terminate_worker = _worker_tools()
    started = time.monotonic()
    # Retained elapsed time keeps the envelope honest across an audited continuation.
    prior_collection_wall_seconds = (retained_ledger or {}).get(
        'collection_wall_seconds_elapsed', 0)
    if (isinstance(prior_collection_wall_seconds, bool)
            or not isinstance(prior_collection_wall_seconds, (int, float))
            or prior_collection_wall_seconds < 0):
        raise ValueError('retained collection wall seconds are invalid')

    def collection_wall_seconds_elapsed():
        return prior_collection_wall_seconds + time.monotonic() - started

    _write_json(ledger_path, _ledger(
        plan, 'running', entries, collection_wall_seconds_elapsed(), continuations))

    context = multiprocessing.get_context('spawn')
    live = []
    launching = None
    launching_branch = None
    launching_started = None
    next_branch = 0
    combined = sum(value['wall_seconds'] for value in entries.values()
                   if value['status'] != 'unattempted')
    artifact_state = {'consecutive_misses': 0, 'available': False}
    stop_launching = None
    supervisor_error = None
    cleanup_errors = []
    value = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupt_on_sigterm(signum, frame):
        raise RuntimeError('supervisor received SIGTERM')

    signal.signal(signal.SIGTERM, interrupt_on_sigterm)
    try:
        while next_branch < len(pending) or live:
            if stop_launching is None:
                stop_launching = _time_stop(
                    started, prior_collection_wall_seconds, combined, live, limits)
                if stop_launching is not None:
                    _stop_live(live, stop_launching)

            while (stop_launching is None and next_branch < len(pending)
                   and len(live) < limits['workers']):
                # Every launch gets a fresh wall/combined check; filling a pool is not atomic.
                stop_launching = _time_stop(
                    started, prior_collection_wall_seconds, combined, live, limits)
                if stop_launching is not None:
                    _stop_live(live, stop_launching)
                    break
                stop_launching = _launch_stop(output, limits, artifact_state)
                if stop_launching is not None:
                    if stop_launching in ('artifact_limit', 'artifact_measurement_unavailable'):
                        _stop_live(live, stop_launching)
                    break
                branch = pending[next_branch]
                identity = branch['identity']
                member = members[branch['source_member_identity']]
                record = _dispatch_record(member, branch)
                _write_json(output / 'markers' / f'{identity}.json', {
                    'schema': 'issue_77_n1_dispatch_marker_v1',
                    'plan_identity': plan['identity'],
                    'member_identity': identity,
                    'dispatched_at_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                })
                launching_branch = branch
                launching_started = time.monotonic()
                blocked = {signal.SIGTERM, signal.SIGINT}
                previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
                region_error = None
                try:
                    try:
                        launching = start_isolated_worker(
                            context, capture, (output, record, capture_limits))
                    except BaseException as spawn_error:
                        try:
                            combined += _seal_spawn_failure(
                                output, plan, branch, launching_started, spawn_error, entries)
                            _write_json(ledger_path, _ledger(
                                plan, 'running', entries, collection_wall_seconds_elapsed(),
                                continuations))
                        except BaseException as persistence_error:
                            spawn_error.add_note(
                                f'spawn failure persistence also failed: '
                                f'{type(persistence_error).__name__}: {persistence_error}')
                        region_error = spawn_error
                    else:
                        worker = {'process': launching, 'branch': branch,
                                  'started': launching_started,
                                  'peak': 0.0, 'rss': 0.0, 'stop': None}
                        _append_live(live, worker)
                        launching = None
                        launching_branch = None
                        launching_started = None
                        next_branch += 1
                except BaseException as error:
                    region_error = error
                finally:
                    try:
                        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                    except BaseException as unmask_error:
                        if region_error is None:
                            region_error = unmask_error
                        else:
                            region_error.add_note(
                                f'signal-mask restoration also failed: '
                                f'{type(unmask_error).__name__}: {unmask_error}')
                if region_error is not None:
                    raise region_error

            if not live:
                break

            time.sleep(1)
            now = time.monotonic()
            for worker in live:
                process = worker['process']
                if process.is_alive():
                    worker['rss'] = process_rss(process.pid)
                    worker['peak'] = max(worker['peak'], worker['rss'])
                    elapsed = now - worker['started']
                    if elapsed >= limits['attempt_seconds']:
                        worker['stop'] = 'attempt_wall_limit'
                    elif worker['peak'] > limits['worker_cpu_rss_mib']:
                        worker['stop'] = 'worker_memory_limit'
                else:
                    worker['rss'] = 0.0

            aggregate = sum(worker['rss'] for worker in live
                            if worker['process'].is_alive() and worker['stop'] is None)
            if aggregate > limits['aggregate_cpu_rss_mib']:
                newest = max((worker for worker in live
                              if worker['process'].is_alive() and worker['stop'] is None),
                             key=lambda worker: worker['started'], default=None)
                if newest is not None:
                    newest['stop'] = 'aggregate_memory_limit'

            if stop_launching is None:
                stop_launching = _time_stop(
                    started, prior_collection_wall_seconds, combined, live, limits)
                if stop_launching is not None:
                    _stop_live(live, stop_launching)
            artifact_stop = _artifact_stop(output, limits, artifact_state)
            if stop_launching is None and artifact_stop is not None:
                stop_launching = artifact_stop
                _stop_live(live, stop_launching)

            finished = [worker for worker in live
                        if worker['stop'] is not None or not worker['process'].is_alive()]
            for worker in finished:
                wall_seconds, cleanup_error = _finish_worker(
                    output, plan, worker, entries, terminate_worker)
                combined += wall_seconds
                live.remove(worker)
                if cleanup_error is not None:
                    raise cleanup_error
                artifact_stop = _artifact_stop(output, limits, artifact_state)
                if stop_launching is None and artifact_stop is not None:
                    stop_launching = artifact_stop
                    _stop_live(live, stop_launching)
                ledger = _ledger(plan, stop_launching or 'running', entries,
                                 collection_wall_seconds_elapsed(), continuations)
                _write_json(ledger_path, ledger)
                done = ledger['counts']['attempted']
                branch = worker['branch']['identity']
                complete = entries[branch]['status'] == 'complete'
                print(f'[{done}/{len(plan["branches"])}] {branch} complete={complete} '
                      f'stop={entries[branch]["stop"]}', flush=True)

        if stop_launching is None:
            stop_launching = _final_artifact_stop(output, limits, artifact_state)
        final_status = stop_launching or 'complete'
        value = _ledger(plan, final_status, entries, collection_wall_seconds_elapsed(),
                        continuations)
        _write_json(ledger_path, value)
    except BaseException as error:
        supervisor_error = error
    finally:
        try:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        except BaseException as ignore_error:
            cleanup_errors.append(ignore_error)
        if supervisor_error is not None:
            failure = f'supervisor_interrupted: {type(supervisor_error).__name__}: {supervisor_error}'
            if (launching is not None
                    and not any(worker['process'] is launching for worker in live)):
                live.append({
                    'process': launching,
                    'branch': launching_branch,
                    'started': launching_started,
                    'peak': 0.0,
                    'rss': 0.0,
                    'stop': None,
                })
            for worker in list(live):
                worker['stop'] = 'supervisor_interrupted'
                worker['failure'] = failure
                try:
                    wall_seconds, cleanup_error = _finish_worker(
                        output, plan, worker, entries, terminate_worker)
                    combined += wall_seconds
                    if cleanup_error is not None:
                        cleanup_errors.append(cleanup_error)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                finally:
                    live.remove(worker)
            try:
                _write_json(ledger_path, _ledger(
                    plan, 'interrupted', entries, collection_wall_seconds_elapsed(),
                    continuations))
            except BaseException as persistence_error:
                cleanup_errors.append(persistence_error)
        try:
            signal.signal(signal.SIGTERM, previous_sigterm)
        except BaseException as restore_error:
            cleanup_errors.append(restore_error)

    if supervisor_error is not None:
        for error in cleanup_errors:
            supervisor_error.add_note(
                f'interruption cleanup also failed: {type(error).__name__}: {error}')
        raise supervisor_error
    if cleanup_errors:
        raise cleanup_errors[0]
    if value is None:
        raise RuntimeError('issue-77 N1 campaign ended without a terminal ledger')
    return value


def validate(output=OUTPUT):
    output = Path(output)
    plan = frozen.load_plan(output)
    _campaign_metadata(output, plan)

    members = {member['identity']: member for member in plan['members']}
    status_counts = Counter({'admissible': 0, 'paired_views_missing': 0,
                             'failed': 0, 'unattempted': 0})
    role_counts = {}
    branches = {}
    stop_kind_counts = Counter()
    failure_counts = Counter()
    for branch in plan['branches']:
        identity = branch['identity']
        member = members[branch['source_member_identity']]
        role = member['study_role']
        role_counter = role_counts.setdefault(role, Counter(
            {'admissible': 0, 'paired_views_missing': 0, 'failed': 0, 'unattempted': 0}))
        result_path = output / 'results' / f'{identity}.json'
        receipt_path = output / 'receipts' / f'{identity}.json'
        paired_path = output / 'attempts' / identity / 'paired-inputs' / 'paired-inputs.json'
        result = _read_json(result_path) if result_path.is_file() else None
        receipt = _read_json(receipt_path) if receipt_path.is_file() else None
        paired = _read_json(paired_path) if paired_path.is_file() else None
        operational = _operationally_valid(result, receipt, plan, identity)
        paired_exists = _paired_views_exist(result, paired) if operational else False
        stop_kind = None
        if operational:
            try:
                stop_kind = _stop_kind(result)
            except ValueError:
                stop_kind = 'unresolved'
        if result is None and receipt is None:
            status = 'unattempted'
        elif not operational:
            status = 'failed'
            failure = (result or {}).get('failure') or (receipt or {}).get('stop') or 'unknown'
            failure_counts[str(failure).split(':')[0]] += 1
        elif not paired_exists:
            status = 'paired_views_missing'
        else:
            status = 'admissible'
            stop_kind_counts[stop_kind] += 1
        status_counts[status] += 1
        role_counter[status] += 1
        branches[identity] = {
            'status': status,
            'study_role': role,
            'candidate_ordinal': branch['candidate_ordinal'],
            'operationally_valid': operational,
            'paired_views_exist': paired_exists,
            'stop_kind': stop_kind,
            'failure': (result or {}).get('failure') if status in ('failed', 'paired_views_missing') else None,
        }

    lineage_states = {}
    for member in plan['members']:
        assigned = [branch for branch in plan['branches']
                    if branch['source_member_identity'] == member['identity']]
        admissible = bool(assigned) and all(
            branches[branch['identity']]['status'] == 'admissible' for branch in assigned)
        lineage_states[member['identity']] = 'admissible' if admissible else 'inadmissible'
    lineage_counts = Counter(lineage_states.values())
    attempted = status_counts['admissible'] + status_counts['paired_views_missing'] + status_counts['failed']

    value = {
        'schema': 'issue_77_n1_coverage_v1',
        'plan_identity': plan['identity'],
        'status_counts': dict(status_counts),
        'per_study_role': {role: dict(counts) for role, counts in role_counts.items()},
        'stop_kind_counts': dict(stop_kind_counts),
        'failure_counts': dict(failure_counts),
        'lineage_admissibility_counts': {
            'admissible': lineage_counts['admissible'],
            'inadmissible': lineage_counts['inadmissible'],
        },
        'lineages': lineage_states,
        'branches': branches,
        'scheduled': len(plan['branches']),
        'attempted': attempted,
        'all_scheduled_branches_executed_or_typed_failures': attempted == len(plan['branches']),
        'coverage_complete': attempted == len(plan['branches']),
    }
    _write_json(output / 'coverage.json', value)
    print(f"issue-77 N1 coverage admissible={status_counts['admissible']}/"
          f"{len(plan['branches'])} attempted={attempted}/{len(plan['branches'])} "
          f"complete={value['coverage_complete']}", flush=True)
    if not value['coverage_complete']:
        raise ValueError('issue-77 N1 coverage has unattempted branches; finish or audit the campaign')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--run', action='store_true')
    modes.add_argument('--validate', action='store_true')
    parser.add_argument('--continue-interrupted', action='store_true')
    args = parser.parse_args()
    if args.continue_interrupted and not args.run:
        parser.error('--continue-interrupted is only valid with --run')
    if args.prepare:
        prepare()
    elif args.run:
        run(continue_interrupted=args.continue_interrupted)
    else:
        validate()


if __name__ == '__main__':
    main()
