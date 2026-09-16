"""Freeze/run a two-shot native history synchronization fixture, never a gate cohort."""
import argparse
from copy import deepcopy
import multiprocessing
from pathlib import Path
import shutil
import time

from scripts import issue_76_shared_history_player as player
from scripts.issue_76_shared_history_capture import capture_one, files
from scripts.issue_76_shared_player_storage import unique_file_bytes
from scripts.run_issue_76_compatibility import process_rss, terminate_worker

OUTPUT = files.ROOT / '.local-artifacts/issue-76-shared-history-smoke-v2'
PROTOCOL = 'docs/issue-76-shared-history-smoke-protocol.md'
SOURCES = ('scripts/issue_76_shared_history_player.py', 'scripts/issue_76_shared_history_capture.py',
           'scripts/issue_76_shared_player_storage.py', 'scripts/run_issue_76_shared_history_smoke.py',
           'tasks/issue_76_canonical/NativeDecisionBarrier.cs', 'tests/test_issue_76_shared_history.py', PROTOCOL)


def prepare(output=OUTPUT, work=player.WORK):
    if output.exists():
        raise ValueError('shared-history smoke already exists; preserve all attempts')
    build = work / 'player-build-01'
    if not (build / '9001.x86_64').is_file() or 'Exiting batchmode successfully now!' not in (work / 'build-01.log').read_text():
        raise ValueError('the frozen shared-history build must finish before smoke preparation')
    source = files.read(files.ROOT / '.local-artifacts/issue-76-native-development-v1/plan.json')
    # Existing TRAIN lineage, selected by role/family/order, not by shot outcome.
    original = next(m for m in source['members'] if m['generator_family'] == 'type010204'
                    and m['exposure_role'] == 'training')
    members = []
    for ordinal, (x, y) in enumerate(((-80, 10), (-60, 45)), 1):
        member = deepcopy(original)
        member.update(identity=f'issue-76-shared-history-smoke-{ordinal:02}', maximum_shots=1,
                      source_member_identity=original['identity'],
                      actions=[{'drag_x': x, 'drag_y': y, 'tap_time_ms': 0, 'release_time_ms': 1000}])
        members.append(member)
    limits = {'workers': 1, 'shots_max': 2, 'technical_retries': 0, 'active_seconds': 1800,
              'attempt_seconds': 420, 'shot_seconds': 180, 'history_ready_seconds': 90,
              'decision_fixed_step': 30000, 'predecision_frames': 3, 'rgb_stride_native_steps': 50,
              'native_step_seconds': .0004, 'native_steps_per_shot_max': 30000,
              'inference_hold_seconds': 2, 'aggregate_cpu_rss_mib': 8192, 'artifact_bytes': 2 * 2**30,
              'minimum_free_bytes': 256 * 2**30, 'branch_position_tolerance': .0001,
              'branch_velocity_tolerance': .0001}
    output.mkdir(parents=True)
    runtime = output / 'player'
    shutil.copytree(build, runtime,
                    ignore=lambda directory, names: ['Levels'] if Path(directory).name == 'StreamingAssets' else [])
    (runtime / '9001.x86_64').rename(runtime / '9001-player.x86_64')
    shutil.copy2(files.ROOT / 'scripts/9001-player-wrapper.sh', runtime / '9001.x86_64')
    transport = files.ROOT / '.local-artifacts/issue-76-native-development-v1/player'
    for name in ('game_playing_interface.jar', 'serverbackup'):
        shutil.copy2(transport / name, runtime / name)
    files.write(output / 'plan.json', {
        'schema': 'issue_76_shared_history_smoke_plan_v1', 'identity': output.name,
        'members': members, 'limits': limits, 'player_build': str(build),
        'parent_plan_identity': source['identity'], 'source_member_identity': original['identity'],
        'source_text': {name: (files.ROOT / name).read_text() for name in SOURCES},
        'player_source': files.read(work / 'shared-history-source.json'),
        'runtime_source_text': {str(path.relative_to(work / 'project')): path.read_text()
                                for path in sorted((work / 'project/Assets/Scripts').rglob('*.cs'))},
        'model_training_updates': 0, 'fresh_access': False, 'advancement_authorized': False,
        'training_eligibility': 'no new role or fitting authorization; existing TRAIN lineage metadata fixture only',
    })
    print('Shared-history fixture frozen: two TRAIN-lineage shots, no capture yet.', flush=True)


def load_plan(output=OUTPUT):
    plan = files.read(output / 'plan.json')
    for name, source in plan['source_text'].items():
        if (files.ROOT / name).read_text() != source:
            raise ValueError(f'shared-history fixture source changed: {name}')
    return plan


def run(output=OUTPUT):
    plan = load_plan(output)
    limits = plan['limits']
    started = time.monotonic()
    for member in plan['members']:
        result_path = output / 'results' / (member['identity'] + '.json')
        if result_path.exists():
            continue
        if (output / 'attempts' / member['identity']).exists():
            raise ValueError('unclean smoke attempt must be audited, never silently repeated')
        if shutil.disk_usage(output).free < limits['minimum_free_bytes']:
            raise ValueError('shared-history smoke lacks its declared minimum free storage')
        process = multiprocessing.get_context('spawn').Process(target=capture_one, args=(output, member, limits))
        process.start()
        beginning, stop, peak = time.monotonic(), None, 0.
        try:
            while process.is_alive():
                process.join(1)
                now = time.monotonic()
                peak = max(peak, process_rss(process.pid))
                if now - beginning >= limits['attempt_seconds']:
                    stop = 'attempt_wall_limit'
                elif now - started >= limits['active_seconds']:
                    stop = 'global_wall_limit'
                elif peak > limits['aggregate_cpu_rss_mib']:
                    stop = 'aggregate_memory_limit'
                if stop:
                    terminate_worker(process)
                    break
        finally:
            if process.is_alive():
                terminate_worker(process)
        files.write(output / 'receipts' / (member['identity'] + '.json'), {
            'member_identity': member['identity'], 'exit_code': process.exitcode, 'stop': stop,
            'wall_seconds': time.monotonic() - beginning, 'peak_cpu_rss_mib': peak,
            'artifact_unique_inode_bytes': unique_file_bytes((output,)), 'technical_retries': 0,
        })
        if stop or process.exitcode or not result_path.exists():
            raise RuntimeError(f'shared-history worker terminated: {stop or process.exitcode}; evidence retained')
        if unique_file_bytes((output,)) > limits['artifact_bytes']:
            raise RuntimeError('shared-history fixture artifact limit; evidence retained')
    validate(output)


def validate(output=OUTPUT):
    plan = load_plan(output)
    results = [files.read(output / 'results' / (m['identity'] + '.json')) for m in plan['members']]
    complete = all(r['complete'] and r.get('paused_state_and_rgb_equal') and r.get('chosen_shot_matches_decision')
                   and r['predecision_fixed_steps'] == [29900, 29950, 30000] for r in results)
    branch = None
    if complete:
        states = [files.read(output / 'attempts' / m['identity'] / 'initial-physics.json') for m in plan['members']]
        entities = [{e['scenario_object_id']: e for e in s['entities']} for s in states]
        if entities[0].keys() != entities[1].keys():
            raise ValueError('shared-history branches have different initial authored entity inventories')
        if any(entities[0][key]['lifecycle'] != entities[1][key]['lifecycle']
               or entities[0][key]['body_present'] != entities[1][key]['body_present'] for key in entities[0]):
            raise ValueError('shared-history branches have different initial lifecycle/body availability')
        position_error = max(abs(a - b) for key in entities[0] if entities[0][key]['body'] is not None
                             for a, b in zip(entities[0][key]['body']['position'], entities[1][key]['body']['position']))
        velocity_error = max(abs(a - b) for key in entities[0] if entities[0][key]['body'] is not None
                             for a, b in zip(entities[0][key]['body']['velocity'], entities[1][key]['body']['velocity']))
        branch = {'position_max_absolute_error': position_error, 'velocity_max_absolute_error': velocity_error,
                  'within_prospective_tolerance': position_error <= plan['limits']['branch_position_tolerance']
                      and velocity_error <= plan['limits']['branch_velocity_tolerance']}
    value = {'schema': 'issue_76_shared_history_smoke_validation_v1', 'plan_identity': plan['identity'],
             'individual_timing_and_rgb_passed': complete, 'branch_initial_state': branch,
             'validated': complete and branch['within_prospective_tolerance'],
             'native_launch_offsets': [r.get('native_launch_offset') for r in results],
             'failures': [r['failure'] for r in results], 'results': results,
             'artifact_unique_inode_bytes': unique_file_bytes((output,)),
             'fresh_access': False, 'advancement_authorized': False}
    files.write(output / 'validation.json', value)
    print({key: value[key] for key in ('validated', 'individual_timing_and_rgb_passed', 'branch_initial_state',
                                      'native_launch_offsets', 'failures')}, flush=True)
    if not value['validated']:
        raise ValueError('shared-history synchronization fixture failed; do not launch a larger collection')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'run', 'validate'):
        modes.add_argument('--' + name, action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.run:
        run()
    else:
        validate()


if __name__ == '__main__':
    main()
