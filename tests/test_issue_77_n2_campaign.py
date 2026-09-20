import json
import os
from pathlib import Path
import signal
import time
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import run_issue_77_n2 as runner


REAL_SLEEP = time.sleep


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _fake_result(output, record, *, complete=True, failure=None):
    output = Path(output)
    (output / 'attempts' / record['identity']).mkdir(parents=True, exist_ok=True)
    call = output / 'fake-calls' / record['identity']
    call.parent.mkdir(parents=True, exist_ok=True)
    call.write_text('called')
    _write(output / 'results' / f"{record['identity']}.json", {
        'schema': 'issue_76_bounded_transfer_capture_v1',
        'member_identity': record['identity'],
        'complete': complete,
        'failure': failure,
    })


def fake_successful_capture(output, record, limits):
    _fake_result(output, record)


def fake_recording_capture(output, record, limits):
    _fake_result(output, record)
    _write(Path(output) / 'dispatch-records' / f"{record['identity']}.json", record)


def fake_one_failure_capture(output, record, limits):
    failed = record['candidate_ordinal'] == 1
    _fake_result(output, record, complete=not failed,
                 failure='fixture failure' if failed else None)


def slow_counted_capture(output, record, limits):
    output = Path(output)
    active = output / 'active-workers'
    observed = output / 'observed-concurrency'
    active.mkdir(parents=True, exist_ok=True)
    observed.mkdir(parents=True, exist_ok=True)
    marker = active / record['identity']
    marker.write_text('active')
    time.sleep(.2)
    (observed / record['identity']).write_text(str(len(list(active.iterdir()))))
    marker.unlink()
    _fake_result(output, record)


class ImmediateProcess:
    next_pid = 910000

    def __init__(self, target, args):
        self.pid = ImmediateProcess.next_pid
        ImmediateProcess.next_pid += 1
        self.exitcode = None
        try:
            target(*args)
            self.exitcode = 0
        except BaseException:
            self.exitcode = 1

    def is_alive(self):
        return False


class LiveProcess:
    next_pid = 920000

    def __init__(self):
        self.pid = LiveProcess.next_pid
        LiveProcess.next_pid += 1
        self.exitcode = None

    def is_alive(self):
        return self.exitcode is None


class Issue77N2CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.plan = self.make_plan(3)
        self.write_campaign(self.output, self.plan)

    @staticmethod
    def make_plan(branch_count, *, workers=2):
        member = {
            'identity': 'lineage-01',
            'base_cluster': 'cluster-01',
            'ordinal': 1,
            'generation_seed': 764200001,
            'engine_seed': 764300001,
            'generator_family': 'type010101',
            'novelty_level': 1,
            'exposure_role': 'training',
            'fit_partition': 'predictor',
            'study_role': 'predictor_train',
            'family_rank': 1,
            'template': {'template': 'fixture.xml'},
            'scenario': {'identity': 'scenario-fixture'},
            'xml': '<Level />',
            'generated_xml_identity': 'generated-fixture',
            'generated_slots': ['slot-1'],
        }
        branches = []
        for ordinal in range(branch_count):
            branches.append({
                'identity': f'branch-{ordinal:02}',
                'ordinal': ordinal + 1,
                'source_member_identity': member['identity'],
                'candidate_ordinal': ordinal,
                'original_action_reference': ordinal == 0,
                'action': {'drag_x': -80 + ordinal, 'drag_y': 10,
                           'tap_time_ms': 0, 'release_time_ms': 1000},
                'input_variants': [{'identity': name} for name in
                                   ('legacy-single', 'corrected-single', 'corrected-history')],
                'exposure_role': member['exposure_role'],
                'fit_partition': member['fit_partition'],
            })
        return {
            'identity': runner.frozen.IDENTITY,
            'members': [member],
            'branches': branches,
            'limits': {
                'workers': workers,
                'worker_cpu_rss_mib': 4096,
                'aggregate_cpu_rss_mib': 32768,
                'combined_collection_seconds': 1000,
                'collection_wall_seconds': 1000,
                'attempt_seconds': 30,
                'technical_retries': 0,
                'minimum_free_bytes': 0,
                'artifact_bytes': 2**30,
            },
        }

    @staticmethod
    def write_campaign(output, plan):
        _write(Path(output) / 'campaign.json', {
            'schema': 'issue_77_n2_campaign_v1',
            'plan_identity': plan['identity'],
            'inference_hold_seconds': runner.INFERENCE_HOLD_SECONDS,
            'limits': {**plan['limits'],
                       'inference_hold_seconds': runner.INFERENCE_HOLD_SECONDS},
            'supervisor_sha256': runner._supervisor_sha256(),
        })

    @staticmethod
    def complete_ledger(plan):
        branches = {branch['identity']: {
            'status': 'complete', 'stop': None, 'exit_code': 0,
            'wall_seconds': 1, 'failure': None,
        } for branch in plan['branches']}
        scheduled = len(plan['branches'])
        return {
            'schema': 'issue_77_n2_ledger_v1',
            'plan_identity': plan['identity'], 'status': 'complete',
            'collection_wall_seconds_elapsed': scheduled, 'continuations': [],
            'counts': {'scheduled': scheduled, 'attempted': scheduled,
                       'complete': scheduled, 'failed': 0, 'unattempted': 0},
            'branches': branches,
        }

    def run_immediate(self, capture, plan=None, continue_interrupted=False):
        plan = plan or self.plan

        def start(context, target, args):
            return ImmediateProcess(target, args)

        tools = (start, lambda pid: 0.0, lambda process: None)
        with patch.object(runner.frozen, 'load_plan', return_value=plan), patch.object(
                runner, '_worker_tools', return_value=tools), patch.object(runner.time, 'sleep'):
            return runner.run(self.output, capture=capture,
                              continue_interrupted=continue_interrupted)

    def seed_interrupted(self, plan, *, elapsed=1, previous_digest='previous-supervisor'):
        campaign_path = self.output / 'campaign.json'
        campaign = json.loads(campaign_path.read_text())
        campaign['supervisor_sha256'] = previous_digest
        _write(campaign_path, campaign)
        branch = plan['branches'][0]
        identity = branch['identity']
        _write(self.output / 'markers' / f'{identity}.json', {
            'member_identity': identity, 'plan_identity': plan['identity']})
        _write(self.output / 'results' / f'{identity}.json', {
            'schema': 'issue_76_bounded_transfer_capture_v1',
            'member_identity': identity, 'complete': False,
            'failure': 'supervisor_interrupted: RuntimeError: fixture', 'segments': [],
        })
        _write(self.output / 'receipts' / f'{identity}.json', {
            'member_identity': identity, 'execution_plan_identity': plan['identity'],
            'execution_within_limits': False, 'stop': 'supervisor_interrupted',
            'technical_retries': 0, 'worker_exitcode': -15, 'wall_seconds': 1,
        })
        branches = {row['identity']: {
            'status': 'failed' if row is branch else 'unattempted',
            'stop': 'supervisor_interrupted' if row is branch else None,
            'exit_code': -15 if row is branch else None,
            'wall_seconds': 1 if row is branch else 0,
            'failure': 'fixture' if row is branch else None,
        } for row in plan['branches']}
        _write(self.output / 'ledger.json', {
            'schema': 'issue_77_n2_ledger_v1',
            'plan_identity': plan['identity'], 'status': 'interrupted',
            'collection_wall_seconds_elapsed': elapsed,
            'continuations': [],
            'counts': {'scheduled': len(plan['branches']), 'attempted': 1,
                       'complete': 0, 'failed': 1,
                       'unattempted': len(plan['branches']) - 1},
            'branches': branches,
        })
        return identity

    def test_dispatch_record_preserves_lineage_and_installs_one_branch_action(self):
        self.run_immediate(fake_recording_capture)
        branch = self.plan['branches'][0]
        record = json.loads((self.output / 'dispatch-records/branch-00.json').read_text())
        self.assertEqual(record['identity'], branch['identity'])
        self.assertEqual(record['actions'], [branch['action']])
        self.assertEqual(record['maximum_shots'], 1)
        for field in ('template', 'scenario', 'engine_seed', 'generated_slots'):
            self.assertEqual(record[field], self.plan['members'][0][field])

    def test_success_receipt_satisfies_frozen_adapter_schema_and_mutations_fail(self):
        self.run_immediate(fake_successful_capture)
        identity = self.plan['branches'][0]['identity']
        receipt = json.loads((self.output / 'receipts' / f'{identity}.json').read_text())
        self.assertTrue(runner._receipt_valid(receipt, self.plan, identity))
        mutations = {
            'execution_within_limits': False,
            'stop': 'attempt_wall_limit',
            'technical_retries': 1,
            'member_identity': 'wrong-branch',
            'execution_plan_identity': 'wrong-plan',
            'worker_exitcode': 1,
        }
        for field, value in mutations.items():
            changed = {**receipt, field: value}
            self.assertFalse(runner._receipt_valid(changed, self.plan, identity), field)
        no_exit = {key: value for key, value in receipt.items()
                   if key not in ('worker_exitcode', 'exit_code')}
        self.assertFalse(runner._receipt_valid(no_exit, self.plan, identity))

    def test_resume_skips_complete_pair_and_unclean_attempt_is_rejected(self):
        identity = self.plan['branches'][0]['identity']
        _write(self.output / 'results' / f'{identity}.json', {
            'schema': 'issue_76_bounded_transfer_capture_v1',
            'member_identity': identity, 'complete': True, 'failure': None})
        _write(self.output / 'receipts' / f'{identity}.json', {
            'member_identity': identity, 'execution_plan_identity': self.plan['identity'],
            'execution_within_limits': True, 'stop': None, 'technical_retries': 0,
            'worker_exitcode': 0, 'wall_seconds': 1,
        })
        self.run_immediate(fake_successful_capture)
        self.assertFalse((self.output / 'fake-calls' / identity).exists())
        self.assertEqual(len(list((self.output / 'fake-calls').iterdir())), 2)

        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.write_campaign(output, self.plan)
            (output / 'attempts' / identity).mkdir(parents=True)
            with patch.object(runner.frozen, 'load_plan', return_value=self.plan):
                with self.assertRaisesRegex(ValueError, 'unclean issue-77 N2 attempt'):
                    runner.run(output, capture=fake_successful_capture)

    def test_failed_branch_does_not_abort_or_retry_campaign(self):
        ledger = self.run_immediate(fake_one_failure_capture)
        self.assertEqual(ledger['status'], 'complete')
        self.assertEqual(ledger['counts'], {
            'scheduled': 3, 'attempted': 3, 'complete': 2, 'failed': 1, 'unattempted': 0})
        self.assertEqual(ledger['branches']['branch-01']['status'], 'failed')
        self.assertEqual(len(list((self.output / 'fake-calls').iterdir())), 3)

    def test_spawn_pool_never_exceeds_worker_cap(self):
        plan = self.make_plan(4, workers=2)
        self.write_campaign(self.output, plan)
        start, _, terminate = runner._worker_tools()
        with patch.object(runner.frozen, 'load_plan', return_value=plan), patch.object(
                runner, '_worker_tools', return_value=(start, lambda pid: 0.0, terminate)):
            ledger = runner.run(self.output, capture=slow_counted_capture)
        observed = [int(path.read_text()) for path in
                    (self.output / 'observed-concurrency').iterdir()]
        self.assertEqual(ledger['counts']['attempted'], 4)
        self.assertLessEqual(max(observed), 2)

    def test_interruption_terminates_live_workers_and_persists_failed_evidence(self):
        plan = self.make_plan(2, workers=2)
        self.write_campaign(self.output, plan)
        processes = []
        terminated = []

        def start(context, target, args):
            process = LiveProcess()
            processes.append(process)
            return process

        def terminate(process):
            terminated.append(process.pid)
            process.exitcode = -15

        tools = (start, lambda pid: 0.0, terminate)
        with patch.object(runner.frozen, 'load_plan', return_value=plan), patch.object(
                runner, '_worker_tools', return_value=tools), patch.object(
                runner.time, 'sleep', side_effect=KeyboardInterrupt('campaign interrupted')):
            with self.assertRaisesRegex(KeyboardInterrupt, 'campaign interrupted'):
                runner.run(self.output, capture=fake_successful_capture)

        self.assertEqual(sorted(terminated), sorted(process.pid for process in processes))
        self.assertEqual(len(terminated), 2)
        for branch in plan['branches']:
            identity = branch['identity']
            result = json.loads((self.output / 'results' / f'{identity}.json').read_text())
            receipt = json.loads((self.output / 'receipts' / f'{identity}.json').read_text())
            self.assertEqual(result['failure'],
                             'supervisor_interrupted: KeyboardInterrupt: campaign interrupted')
            self.assertFalse(result['complete'])
            self.assertEqual(result['segments'], [])
            self.assertEqual(receipt['stop'], 'supervisor_interrupted')
            self.assertFalse(receipt['execution_within_limits'])
            self.assertIn('worker_exitcode', receipt)
        ledger = json.loads((self.output / 'ledger.json').read_text())
        self.assertEqual(ledger['status'], 'interrupted')
        self.assertEqual(ledger['counts']['failed'], 2)

    def test_launch_append_race_still_finalizes_the_started_worker(self):
        plan = self.make_plan(1)
        self.write_campaign(self.output, plan)
        process = LiveProcess()
        terminated = []

        def terminate(worker):
            terminated.append(worker.pid)
            worker.exitcode = -15

        tools = (lambda context, target, args: process, lambda pid: 0.0, terminate)
        with patch.object(runner.frozen, 'load_plan', return_value=plan), patch.object(
                runner, '_worker_tools', return_value=tools), patch.object(
                runner, '_append_live', side_effect=RuntimeError('append race')):
            with self.assertRaisesRegex(RuntimeError, 'append race'):
                runner.run(self.output, capture=fake_successful_capture)

        self.assertEqual(terminated, [process.pid])
        result = json.loads((self.output / 'results/branch-00.json').read_text())
        receipt = json.loads((self.output / 'receipts/branch-00.json').read_text())
        ledger = json.loads((self.output / 'ledger.json').read_text())
        self.assertEqual(result['failure'], 'supervisor_interrupted: RuntimeError: append race')
        self.assertEqual(receipt['stop'], 'supervisor_interrupted')
        self.assertEqual(ledger['status'], 'interrupted')
        self.assertEqual(ledger['branches']['branch-00']['status'], 'failed')

    def test_spawn_failure_after_internal_reap_is_sealed_before_reraising(self):
        plan = self.make_plan(1)
        self.write_campaign(self.output, plan)
        terminate = Mock()
        internally_reaped = []

        def start(context, target, args):
            internally_reaped.append(True)
            raise RuntimeError('spawn internally reaped')

        tools = (start, lambda pid: 0.0, terminate)
        with patch.object(runner.frozen, 'load_plan', return_value=plan), patch.object(
                runner, '_worker_tools', return_value=tools):
            with self.assertRaisesRegex(RuntimeError, 'spawn internally reaped'):
                runner.run(self.output, capture=fake_successful_capture)

        terminate.assert_not_called()
        self.assertEqual(internally_reaped, [True])
        result = json.loads((self.output / 'results/branch-00.json').read_text())
        receipt = json.loads((self.output / 'receipts/branch-00.json').read_text())
        ledger = json.loads((self.output / 'ledger.json').read_text())
        self.assertEqual(
            result['failure'], 'worker_spawn_failure: RuntimeError: spawn internally reaped')
        self.assertFalse(receipt['execution_within_limits'])
        self.assertEqual(receipt['stop'], 'worker_spawn_failure')
        self.assertIn('worker_exitcode', receipt)
        self.assertIsNone(receipt['worker_exitcode'])
        self.assertEqual(ledger['status'], 'interrupted')
        self.assertEqual(ledger['branches']['branch-00']['status'], 'failed')

    def test_spawn_window_masks_sigterm_until_registration_and_seals_once(self):
        plan = self.make_plan(1)
        self.write_campaign(self.output, plan)
        process = LiveProcess()
        initial_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        result_path = self.output / 'results/branch-00.json'
        result_writes = []
        original_write = runner._write_json

        def start(context, target, args):
            current = signal.pthread_sigmask(signal.SIG_BLOCK, set())
            self.assertIn(signal.SIGTERM, current)
            self.assertIn(signal.SIGINT, current)
            os.kill(os.getpid(), signal.SIGTERM)
            return process

        def terminate(worker):
            worker.exitcode = -15

        def counting_write(path, value):
            if Path(path) == result_path:
                result_writes.append(value)
            return original_write(path, value)

        tools = (start, lambda pid: 0.0, terminate)
        with patch.object(runner.frozen, 'load_plan', return_value=plan), patch.object(
                runner, '_worker_tools', return_value=tools), patch.object(
                runner, '_write_json', side_effect=counting_write):
            with self.assertRaisesRegex(RuntimeError, 'supervisor received SIGTERM'):
                runner.run(self.output, capture=fake_successful_capture)

        restored_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        self.assertEqual(restored_mask, initial_mask)
        self.assertEqual(len(result_writes), 1)
        result = json.loads(result_path.read_text())
        receipt = json.loads((self.output / 'receipts/branch-00.json').read_text())
        self.assertEqual(
            result['failure'], 'supervisor_interrupted: RuntimeError: supervisor received SIGTERM')
        self.assertEqual(receipt['stop'], 'supervisor_interrupted')
        self.assertNotEqual(result['failure'].split(':', 1)[0], 'worker_spawn_failure')

    def test_installed_sigterm_raises_and_second_sigterm_is_ignored_during_cleanup(self):
        plan = self.make_plan(1)
        self.write_campaign(self.output, plan)
        process = LiveProcess()
        cleanup_reached = []

        def terminate(worker):
            os.kill(os.getpid(), signal.SIGTERM)
            cleanup_reached.append(True)
            worker.exitcode = -15

        def invoke_handler(seconds):
            handler = signal.getsignal(signal.SIGTERM)
            if not callable(handler):
                self.fail('campaign did not install a callable SIGTERM handler')
            handler(signal.SIGTERM, None)

        tools = (lambda context, target, args: process, lambda pid: 0.0, terminate)
        with patch.object(runner.frozen, 'load_plan', return_value=plan), patch.object(
                runner, '_worker_tools', return_value=tools), patch.object(
                runner.time, 'sleep', side_effect=invoke_handler):
            with self.assertRaisesRegex(RuntimeError, 'supervisor received SIGTERM'):
                runner.run(self.output, capture=fake_successful_capture)
        self.assertEqual(cleanup_reached, [True])
        result = json.loads((self.output / 'results/branch-00.json').read_text())
        self.assertIn('supervisor received SIGTERM', result['failure'])

    def test_dispatch_marker_precedes_spawn_and_blocks_unsealed_restart(self):
        plan = self.make_plan(1)
        self.write_campaign(self.output, plan)

        def start(context, target, args):
            identity = args[1]['identity']
            self.assertTrue((self.output / 'markers' / f'{identity}.json').is_file())
            return ImmediateProcess(target, args)

        tools = (start, lambda pid: 0.0, lambda process: None)
        with patch.object(runner.frozen, 'load_plan', return_value=plan), patch.object(
                runner, '_worker_tools', return_value=tools), patch.object(runner.time, 'sleep'):
            runner.run(self.output, capture=fake_successful_capture)

        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.write_campaign(output, plan)
            _write(output / 'markers/branch-00.json', {'member_identity': 'branch-00'})
            with patch.object(runner.frozen, 'load_plan', return_value=plan):
                with self.assertRaisesRegex(ValueError, 'unclean issue-77 N2 attempt'):
                    runner.run(output, capture=fake_successful_capture)

    def test_running_and_terminal_incomplete_ledgers_prohibit_top_up(self):
        for status in ('running', 'artifact_limit'):
            with self.subTest(status=status), TemporaryDirectory() as temporary:
                output = Path(temporary)
                self.write_campaign(output, self.plan)
                _write(output / 'ledger.json', {
                    'schema': 'issue_77_n2_ledger_v1',
                    'plan_identity': self.plan['identity'], 'status': status,
                    'counts': {'scheduled': 3, 'attempted': 0, 'complete': 0,
                               'failed': 0, 'unattempted': 3},
                    'branches': {},
                })
                with patch.object(runner.frozen, 'load_plan', return_value=self.plan):
                    with self.assertRaisesRegex(ValueError, status):
                        runner.run(output, capture=fake_successful_capture)

    def test_interrupted_continuation_dispatches_only_unmarked_and_records_revision(self):
        marked = self.seed_interrupted(self.plan)
        ledger = self.run_immediate(
            fake_successful_capture, continue_interrupted=True)
        called = {path.name for path in (self.output / 'fake-calls').iterdir()}
        self.assertEqual(called, {'branch-01', 'branch-02'})
        self.assertNotIn(marked, called)
        self.assertEqual(ledger['status'], 'complete')
        self.assertEqual(ledger['counts'], {
            'scheduled': 3, 'attempted': 3, 'complete': 2, 'failed': 1, 'unattempted': 0})
        self.assertEqual(len(ledger['continuations']), 1)
        continuation = ledger['continuations'][0]
        self.assertEqual(continuation['from_status'], 'interrupted')
        self.assertEqual(continuation['supervisor_sha256_previous'], 'previous-supervisor')
        self.assertEqual(continuation['supervisor_sha256_current'], runner._supervisor_sha256())
        campaign = json.loads((self.output / 'campaign.json').read_text())
        self.assertEqual(campaign['supervisor_sha256'], runner._supervisor_sha256())
        self.assertEqual(len(campaign['supervisor_revisions']), 1)
        self.assertEqual(campaign['supervisor_revisions'][0]['supervisor_sha256_previous'],
                         'previous-supervisor')
        self.assertEqual(campaign['supervisor_revisions'][0]['identity'],
                         continuation['identity'])

    def test_campaign_only_continuation_event_is_reconciled_without_duplication(self):
        self.seed_interrupted(self.plan)
        campaign_path = self.output / 'campaign.json'
        ledger_path = self.output / 'ledger.json'
        campaign = json.loads(campaign_path.read_text())
        ledger = json.loads(ledger_path.read_text())
        identity = runner._continuation_identity(self.plan, ledger)
        event = {
            'identity': identity,
            'from_status': 'interrupted',
            'at_utc': '2026-09-18T00:00:00Z',
            'supervisor_sha256_previous': campaign['supervisor_sha256'],
            'supervisor_sha256_current': runner._supervisor_sha256(),
        }
        campaign['supervisor_revisions'] = [event]
        campaign['supervisor_sha256'] = runner._supervisor_sha256()
        _write(campaign_path, campaign)

        first = runner._reconcile_continuation(
            self.output, self.plan, campaign_path, campaign, ledger)
        campaign = json.loads(campaign_path.read_text())
        ledger = json.loads(ledger_path.read_text())
        second = runner._reconcile_continuation(
            self.output, self.plan, campaign_path, campaign, ledger)
        self.assertEqual(first, [event])
        self.assertEqual(second, [event])

        completed = self.run_immediate(
            fake_successful_capture, continue_interrupted=True)
        campaign = json.loads(campaign_path.read_text())
        self.assertEqual(campaign['supervisor_revisions'], [event])
        self.assertEqual(completed['continuations'], [event])

    def test_interrupted_requires_flag_and_resource_stops_refuse_even_with_flag(self):
        self.seed_interrupted(self.plan)
        with patch.object(runner.frozen, 'load_plan', return_value=self.plan):
            with self.assertRaisesRegex(ValueError, 'continue-interrupted'):
                runner.run(self.output, capture=fake_successful_capture)

        for status in ('artifact_limit', 'minimum_free_storage',
                       'collection_wall_limit', 'combined_collection_limit', 'running'):
            with self.subTest(status=status), TemporaryDirectory() as temporary:
                output = Path(temporary)
                self.write_campaign(output, self.plan)
                _write(output / 'ledger.json', {
                    'schema': 'issue_77_n2_ledger_v1',
                    'plan_identity': self.plan['identity'], 'status': status,
                    'collection_wall_seconds_elapsed': 1, 'continuations': [],
                    'counts': {'scheduled': 3, 'attempted': 1, 'complete': 0,
                               'failed': 1, 'unattempted': 2}, 'branches': {},
                })
                with patch.object(runner.frozen, 'load_plan', return_value=self.plan):
                    with self.assertRaisesRegex(ValueError, status):
                        runner.run(output, capture=fake_successful_capture,
                                   continue_interrupted=True)

    def test_collection_wall_limit_terminates_all_live_workers_fail_closed(self):
        plan = self.make_plan(3, workers=2)
        plan['limits']['collection_wall_seconds'] = .05
        self.write_campaign(self.output, plan)
        processes = []

        def start(context, target, args):
            process = LiveProcess()
            processes.append(process)
            return process

        def terminate(process):
            process.exitcode = -15

        tools = (start, lambda pid: 0.0, terminate)
        with patch.object(runner.frozen, 'load_plan', return_value=plan), patch.object(
                runner, '_worker_tools', return_value=tools), patch.object(
                runner.time, 'sleep', side_effect=lambda seconds: REAL_SLEEP(.06)):
            ledger = runner.run(self.output, capture=fake_successful_capture)
        self.assertEqual(len(processes), 2)
        self.assertEqual(ledger['status'], 'collection_wall_limit')
        self.assertGreaterEqual(ledger['collection_wall_seconds_elapsed'], .05)
        self.assertEqual(ledger['counts']['unattempted'], 1)
        for identity in ('branch-00', 'branch-01'):
            receipt = json.loads((self.output / 'receipts' / f'{identity}.json').read_text())
            self.assertEqual(receipt['stop'], 'collection_wall_limit')
            self.assertFalse(receipt['execution_within_limits'])

    def test_malformed_complete_ledger_with_unattempted_branch_is_rejected(self):
        plan = self.make_plan(1)
        self.write_campaign(self.output, plan)
        _write(self.output / 'ledger.json', {
            'schema': 'issue_77_n2_ledger_v1',
            'plan_identity': plan['identity'],
            'status': 'complete',
            'collection_wall_seconds_elapsed': 11,
            'counts': {'scheduled': 1, 'attempted': 0, 'complete': 0,
                       'failed': 0, 'unattempted': 1},
            'branches': {'branch-00': {'status': 'unattempted'}},
        })
        start = Mock()
        tools = (start, lambda pid: 0.0, lambda process: None)
        with patch.object(runner.frozen, 'load_plan', return_value=plan), patch.object(
                runner, '_worker_tools', return_value=tools):
            with self.assertRaisesRegex(ValueError, 'inconsistent with the plan inventory'):
                runner.run(self.output, capture=fake_successful_capture)
        start.assert_not_called()

    def test_valid_complete_ledger_is_returned_without_any_file_mutation(self):
        plan = self.make_plan(2)
        self.write_campaign(self.output, plan)
        ledger = self.complete_ledger(plan)
        ledger_path = self.output / 'ledger.json'
        campaign_path = self.output / 'campaign.json'
        _write(ledger_path, ledger)
        before = (ledger_path.read_bytes(), campaign_path.read_bytes(),
                  ledger_path.stat().st_mtime_ns, campaign_path.stat().st_mtime_ns)
        capture = Mock()
        with patch.object(runner.frozen, 'load_plan', return_value=plan), patch.object(
                runner, '_worker_tools', side_effect=AssertionError('must not initialize workers')):
            returned = runner.run(self.output, capture=capture)
        after = (ledger_path.read_bytes(), campaign_path.read_bytes(),
                 ledger_path.stat().st_mtime_ns, campaign_path.stat().st_mtime_ns)
        self.assertEqual(returned, ledger)
        self.assertEqual(after, before)
        capture.assert_not_called()
        with patch.object(runner.frozen, 'load_plan', return_value=plan):
            with self.assertRaisesRegex(ValueError, 'requires a retained interrupted ledger'):
                runner.run(self.output, capture=capture, continue_interrupted=True)

    def test_complete_ledger_rejects_campaign_drift_read_only(self):
        plan = self.make_plan(1)
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.write_campaign(output, plan)
            campaign_path = output / 'campaign.json'
            ledger_path = output / 'ledger.json'
            campaign = json.loads(campaign_path.read_text())
            campaign['supervisor_sha256'] = 'unauthorized-drift'
            _write(campaign_path, campaign)
            _write(ledger_path, self.complete_ledger(plan))
            before = (campaign_path.read_bytes(), ledger_path.read_bytes(),
                      campaign_path.stat().st_mtime_ns, ledger_path.stat().st_mtime_ns)
            capture = Mock()
            with patch.object(runner.frozen, 'load_plan', return_value=plan), patch.object(
                    runner, '_worker_tools', side_effect=AssertionError('must not initialize workers')):
                with self.assertRaisesRegex(ValueError, 'metadata differs'):
                    runner.run(output, capture=capture)
            after = (campaign_path.read_bytes(), ledger_path.read_bytes(),
                     campaign_path.stat().st_mtime_ns, ledger_path.stat().st_mtime_ns)
            self.assertEqual(after, before)
            capture.assert_not_called()

    def test_interrupted_continuation_carries_wall_envelope(self):
        plan = self.make_plan(2)
        plan['limits']['collection_wall_seconds'] = 10
        self.write_campaign(self.output, plan)
        self.seed_interrupted(plan, elapsed=11)
        start = Mock()
        tools = (start, lambda pid: 0.0, lambda process: None)
        with patch.object(runner.frozen, 'load_plan', return_value=plan), patch.object(
                runner, '_worker_tools', return_value=tools):
            ledger = runner.run(self.output, capture=fake_successful_capture,
                                continue_interrupted=True)
        start.assert_not_called()
        self.assertEqual(ledger['status'], 'collection_wall_limit')
        self.assertGreaterEqual(ledger['collection_wall_seconds_elapsed'], 11)
        self.assertEqual(ledger['counts']['failed'], 1)
        self.assertEqual(ledger['counts']['unattempted'], 1)

    def test_interrupted_continuation_carries_combined_worker_seconds(self):
        plan = self.make_plan(2)
        plan['limits']['combined_collection_seconds'] = .5
        self.write_campaign(self.output, plan)
        self.seed_interrupted(plan, elapsed=1)
        start = Mock()
        tools = (start, lambda pid: 0.0, lambda process: None)
        with patch.object(runner.frozen, 'load_plan', return_value=plan), patch.object(
                runner, '_worker_tools', return_value=tools):
            ledger = runner.run(self.output, capture=fake_successful_capture,
                                continue_interrupted=True)
        start.assert_not_called()
        self.assertEqual(ledger['status'], 'combined_collection_limit')
        self.assertEqual(ledger['counts']['failed'], 1)
        self.assertEqual(ledger['counts']['unattempted'], 1)

    def test_transient_artifact_scan_miss_does_not_interrupt_workers(self):
        plan = self.make_plan(2, workers=2)
        self.write_campaign(self.output, plan)
        calls = 0

        def measured(roots):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise FileNotFoundError('worker replaced temporary frame')
            return 0

        def start(context, target, args):
            return ImmediateProcess(target, args)

        tools = (start, lambda pid: 0.0, lambda process: None)
        with patch.object(runner.frozen, 'load_plan', return_value=plan), patch.object(
                runner, '_worker_tools', return_value=tools), patch.object(
                runner, 'unique_file_bytes', side_effect=measured), patch.object(
                runner.time, 'sleep'):
            ledger = runner.run(self.output, capture=fake_successful_capture)
        self.assertGreaterEqual(calls, 3)
        self.assertEqual(ledger['status'], 'complete')
        self.assertEqual(ledger['counts']['complete'], 2)

    def test_persistent_artifact_measurement_failure_stops_and_finalizes_live_workers(self):
        plan = self.make_plan(2, workers=2)
        self.write_campaign(self.output, plan)
        processes = []

        def start(context, target, args):
            process = LiveProcess()
            processes.append(process)
            return process

        def terminate(process):
            process.exitcode = -15

        tools = (start, lambda pid: 0.0, terminate)
        with patch.object(runner.frozen, 'load_plan', return_value=plan), patch.object(
                runner, '_worker_tools', return_value=tools), patch.object(
                runner, 'unique_file_bytes', side_effect=FileNotFoundError('persistent churn')), patch.object(
                runner.time, 'sleep'):
            ledger = runner.run(self.output, capture=fake_successful_capture)
        self.assertEqual(len(processes), 2)
        self.assertEqual(ledger['status'], 'artifact_measurement_unavailable')
        self.assertEqual(ledger['counts']['failed'], 2)
        self.assertNotEqual(ledger['status'], 'complete')
        for identity in ('branch-00', 'branch-01'):
            receipt = json.loads((self.output / 'receipts' / f'{identity}.json').read_text())
            self.assertEqual(receipt['stop'], 'artifact_measurement_unavailable')
            self.assertFalse(receipt['execution_within_limits'])

    def test_final_artifact_measurement_is_required_before_complete(self):
        plan = self.make_plan(1)
        self.write_campaign(self.output, plan)
        calls = 0

        def measured(roots):
            nonlocal calls
            calls += 1
            if calls >= 5:
                raise FileNotFoundError('final scan unavailable')
            return 0

        def start(context, target, args):
            return ImmediateProcess(target, args)

        tools = (start, lambda pid: 0.0, lambda process: None)
        with patch.object(runner.frozen, 'load_plan', return_value=plan), patch.object(
                runner, '_worker_tools', return_value=tools), patch.object(
                runner, 'unique_file_bytes', side_effect=measured), patch.object(
                runner.time, 'sleep'):
            ledger = runner.run(self.output, capture=fake_successful_capture)
        self.assertGreaterEqual(calls, 5 + runner.ARTIFACT_MEASUREMENT_MISS_LIMIT - 1)
        self.assertEqual(ledger['status'], 'artifact_measurement_unavailable')
        self.assertEqual(ledger['counts']['complete'], 1)

    def test_artifact_overrun_while_workers_live_terminates_the_wave(self):
        plan = self.make_plan(3, workers=2)
        plan['limits']['artifact_bytes'] = 10
        self.write_campaign(self.output, plan)
        calls = 0

        def measured(roots):
            nonlocal calls
            calls += 1
            return 0 if calls <= 2 else 11

        def start(context, target, args):
            return LiveProcess()

        def terminate(process):
            process.exitcode = -15

        tools = (start, lambda pid: 0.0, terminate)
        with patch.object(runner.frozen, 'load_plan', return_value=plan), patch.object(
                runner, '_worker_tools', return_value=tools), patch.object(
                runner, 'unique_file_bytes', side_effect=measured), patch.object(runner.time, 'sleep'):
            ledger = runner.run(self.output, capture=fake_successful_capture)
        self.assertEqual(ledger['status'], 'artifact_limit')
        self.assertEqual(ledger['counts']['failed'], 2)
        self.assertEqual(ledger['counts']['unattempted'], 1)
        for identity in ('branch-00', 'branch-01'):
            receipt = json.loads((self.output / 'receipts' / f'{identity}.json').read_text())
            self.assertEqual(receipt['stop'], 'artifact_limit')

    def test_prepare_binds_supervisor_digest_and_source_drift_is_refused(self):
        root = self.output / 'prepared'
        player = self.output / 'source-player'
        player.mkdir()
        for name in runner.PLAYER_FILES:
            (player / name).write_bytes(b'fixture')

        def write_plan(output):
            Path(output).mkdir()
            return self.plan

        with patch.object(runner, 'PLAYER_SOURCE', player), patch.object(
                runner.frozen, 'write_plan', side_effect=write_plan):
            runner.prepare(root)
        campaign = json.loads((root / 'campaign.json').read_text())
        self.assertEqual(campaign['supervisor_sha256'], runner._supervisor_sha256())

        with patch.object(runner.frozen, 'load_plan', return_value=self.plan), patch.object(
                runner, '_supervisor_sha256', return_value='changed-source'):
            with self.assertRaisesRegex(ValueError, 'campaign metadata differs'):
                runner.run(root, capture=fake_successful_capture)
            with self.assertRaisesRegex(ValueError, 'campaign metadata differs'):
                runner.validate(root)

    def test_result_state_requires_schema_identity_and_execution_within_limits(self):
        identity = self.plan['branches'][0]['identity']
        result = {
            'schema': 'issue_76_bounded_transfer_capture_v1',
            'member_identity': identity, 'complete': True, 'failure': None,
        }
        receipt = {
            'member_identity': identity, 'execution_plan_identity': self.plan['identity'],
            'execution_within_limits': True, 'stop': None, 'technical_retries': 0,
            'worker_exitcode': 0, 'wall_seconds': 1,
        }
        self.assertEqual(runner._result_state(result, receipt, self.plan, identity)['status'],
                         'complete')
        for changed_result, changed_receipt in (
                ({**result, 'schema': 'wrong'}, receipt),
                ({**result, 'member_identity': 'wrong'}, receipt),
                (result, {**receipt, 'execution_within_limits': False})):
            self.assertEqual(runner._result_state(
                changed_result, changed_receipt, self.plan, identity)['status'], 'failed')

        missing_cases = (
            ({key: value for key, value in result.items() if key != 'failure'}, receipt,
             'missing_result_failure'),
            (result, {key: value for key, value in receipt.items() if key != 'stop'},
             'missing_receipt_stop'),
            (result, {key: value for key, value in receipt.items()
                      if key not in ('worker_exitcode', 'exit_code')}, 'missing_worker_exitcode'),
            (result, {key: value for key, value in receipt.items()
                      if key != 'execution_within_limits'}, 'missing_execution_within_limits'),
        )
        for changed_result, changed_receipt, reason in missing_cases:
            state = runner._result_state(changed_result, changed_receipt, self.plan, identity)
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['failure'], reason)

    def test_validate_reports_mixed_coverage_with_typed_failures(self):
        held = {**self.plan['members'][0], 'identity': 'held', 'ordinal': 7,
                'exposure_role': 'calibration', 'fit_partition': None,
                'study_role': 'held_out_evaluation'}
        failed = {**self.plan['members'][0], 'identity': 'failed', 'ordinal': 1,
                  'study_role': 'predictor_train'}
        missing = {**self.plan['members'][0], 'identity': 'missing', 'ordinal': 2,
                   'fit_partition': 'predictor', 'study_role': 'predictor_train'}
        plan = self.make_plan(0)
        plan['members'] = [held, failed, missing]
        plan['branches'] = []
        for ordinal, member in enumerate(plan['members']):
            plan['branches'].append({
                'identity': f'validate-{ordinal}', 'ordinal': ordinal + 1,
                'source_member_identity': member['identity'], 'candidate_ordinal': 0,
                'original_action_reference': True, 'action': {}, 'input_variants': [],
                'exposure_role': member['exposure_role'], 'fit_partition': member['fit_partition'],
            })

        identity = 'validate-0'
        result = {
            'schema': 'issue_76_bounded_transfer_capture_v1',
            'member_identity': identity, 'complete': True, 'failure': None,
            'paired_input_manifest': 'paired-0',
            'paired_input_views': ['legacy-single', 'corrected-single', 'corrected-history'],
            'segments': [{'summary': {'observed_window_valid': True, 'first_fixed_step': 1,
                                      'last_fixed_step': 2, 'censored': False,
                                      'terminal_observed': True},
                          'terminal_evidence': {'fixed_step': 2, 'reason': 'level_clear'}}],
        }
        receipt = {
            'member_identity': identity, 'execution_plan_identity': plan['identity'],
            'execution_within_limits': True, 'stop': None, 'technical_retries': 0,
            'worker_exitcode': 0,
        }
        paired = {
            'schema': 'issue_76_bounded_transfer_paired_inputs_v1', 'identity': 'paired-0',
            'views': {'legacy-single': {}, 'corrected-single': {}, 'corrected-history': {}},
            'same_native_decision_state': True, 'post_action_frames_used': False,
        }
        _write(self.output / f'results/{identity}.json', result)
        _write(self.output / f'receipts/{identity}.json', receipt)
        _write(self.output / f'attempts/{identity}/paired-inputs/paired-inputs.json', paired)
        _write(self.output / 'results/validate-1.json', {
            'schema': 'issue_76_bounded_transfer_capture_v1',
            'member_identity': 'validate-1', 'complete': False, 'failure': 'fixture'})
        _write(self.output / 'receipts/validate-1.json', {
            'member_identity': 'validate-1', 'execution_plan_identity': plan['identity'],
            'execution_within_limits': False, 'stop': 'attempt_wall_limit', 'technical_retries': 0,
            'worker_exitcode': -9,
        })
        _write(self.output / 'results/validate-2.json', {
            'schema': 'issue_76_bounded_transfer_capture_v1',
            'member_identity': 'validate-2', 'complete': True, 'failure': None,
            'paired_input_manifest': 'paired-2',
            'paired_input_views': ['legacy-single', 'corrected-single', 'corrected-history'],
            'segments': [{'summary': {'observed_window_valid': True, 'first_fixed_step': 1,
                                      'last_fixed_step': 2, 'censored': True,
                                      'terminal_observed': False}}]})
        _write(self.output / 'receipts/validate-2.json', {
            'member_identity': 'validate-2', 'execution_plan_identity': plan['identity'],
            'execution_within_limits': True, 'stop': None, 'technical_retries': 0,
            'worker_exitcode': 0,
        })

        with patch.object(runner.frozen, 'load_plan', return_value=plan):
            coverage = runner.validate(self.output)
        self.assertEqual(coverage['status_counts'], {
            'admissible': 1, 'paired_views_missing': 1, 'failed': 1, 'unattempted': 0})
        self.assertEqual(coverage['lineage_admissibility_counts'], {
            'admissible': 1, 'inadmissible': 2})
        self.assertEqual(coverage['stop_kind_counts'], {'native_clear': 1})
        self.assertTrue(coverage['all_scheduled_branches_executed_or_typed_failures'])
        self.assertTrue(coverage['coverage_complete'])
        self.assertEqual(coverage['branches']['validate-1']['failure'], 'fixture')
        self.assertTrue(json.loads((self.output / 'coverage.json').read_text())['coverage_complete'])

    def test_minimum_free_storage_stops_before_launch_and_retains_assignments(self):
        start = Mock()
        tools = (start, lambda pid: 0.0, lambda process: None)
        with patch.object(runner.frozen, 'load_plan', return_value=self.plan), patch.object(
                runner.shutil, 'disk_usage', return_value=SimpleNamespace(free=-1)), patch.object(
                runner, '_worker_tools', return_value=tools):
            ledger = runner.run(self.output, capture=fake_successful_capture)
        start.assert_not_called()
        self.assertEqual(ledger['status'], 'minimum_free_storage')
        self.assertEqual(ledger['counts']['unattempted'], 3)
        self.assertTrue(all(value['status'] == 'unattempted'
                            for value in ledger['branches'].values()))


if __name__ == '__main__':
    unittest.main()
