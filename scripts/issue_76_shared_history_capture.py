"""Causal one-shot development capture at the shared native RGB-history barrier."""
import os
from pathlib import Path
import time

from scripts import issue_76_live_episode as live
from scripts import issue_76_display_start as display_start
from scripts.issue_76_fixed_replay_policy import FixedReplayPolicy
from scripts.issue_76_shared_player_storage import clone_player
from scripts.native_segment_trace import NativeSegmentTrace
from scripts.observation_trace import MANIFEST_NAME, _plain_json, persist_observation_trace
from scripts.process_lifecycle import cleanup_actions, persist_after_cleanup, record_cleanup_failures
from src.webui.bridge import ScienceBirdsBridge

files, capture = live.files, live.capture
TARGET_VARIABLE = 'NOVPHY_NATIVE_DECISION_STEP'


def read_history(aligned, target):
    folders = list(Path(aligned).glob('decision-history-*/ready.json'))
    if len(folders) != 1:
        raise ValueError('native decision history must have exactly one sealed ready manifest')
    root = folders[0].parent
    ready = files.read(folders[0])
    if (ready['schema'] != 'issue_76_native_decision_history_v1'
            or ready['target_fixed_step'] != target or ready['frame_count'] != 3
            or ready['stride_native_steps'] != 50 or abs(ready['native_step_seconds'] - .0004) > 1e-10
            or ready['capture_id'] != root.name):
        raise ValueError('native decision history differs from its prospective timing contract')
    rows = []
    for ordinal, step in enumerate(range(target - 100, target + 1, 50), 1):
        path = root / f'frame_{ordinal:06}.json'
        metadata = files.read(path)
        if metadata['fixed_step'] != step or metadata['capture_id'] != root.name or metadata['sequence'] != ordinal:
            raise ValueError('native decision history frame order, time or capture binding differs')
        rows.append({**metadata, 'canonical_png': path.with_suffix('.png').read_bytes()})
    if any(abs(b['fixed_time_seconds'] - a['fixed_time_seconds'] - .02) > 1e-5
           for a, b in zip(rows, rows[1:])):
        raise ValueError('native decision history is not spaced at twenty milliseconds')
    return root, rows


def expose_history(rows, destination, scenario, identity, role, policy):
    manifest = persist_observation_trace(destination, rows,
        observation_configuration=capture.old.capture.OBSERVATION_CONFIGURATION,
        source_bindings=capture.old.capture._observation_bindings(scenario, rollout_identity=identity),
        exposure_role=role)
    # Evaluation provenance stays outside the callback: no objects, bodies,
    # scene/seed IDs, action candidates, or future frames enter the policy.
    for frame in manifest['frame_records']:
        png = (Path(destination) / frame['agent_observation']['relative_path']).read_bytes()
        policy.observe(png, frame['fixed_time_seconds'])
    return manifest


def consume_shared_segment(observation_root, action, decision_timestamp, policy):
    """The initial shot RGB is already the latest history observation.

    Keep its native truth/metadata binding but do not inject a duplicate zero-dt
    observation into the streaming encoder. Insert acceptance once, then replay
    only subsequent real frames for the next decision.
    """
    observation_root = Path(observation_root)
    manifest = files.read(observation_root / MANIFEST_NAME)
    first = manifest['frame_records'][0]
    if first['fixed_time_seconds'] != decision_timestamp:
        raise ValueError('accepted shot clock differs from its latest shared observation')
    policy.executed(action, decision_timestamp)
    for frame in manifest['frame_records'][1:]:
        png = (observation_root / frame['agent_observation']['relative_path']).read_bytes()
        policy.observe(png, frame['fixed_time_seconds'])


def capture_one(output, member, limits):
    output = Path(output)
    root = output / 'attempts' / member['identity']
    root.mkdir(parents=True)
    result = {'schema': 'issue_76_shared_history_capture_v1', 'member_identity': member['identity'],
              'exposure_role': member['exposure_role'], 'base_cluster': member['base_cluster'],
              'failure': None, 'segments': [], 'fresh_access': False, 'advancement_authorized': False}
    started = time.monotonic()
    bridge = endpoint = engine = display_process = None
    variables = ('DISPLAY', 'XDG_DATA_HOME', 'NOVPHY_PHYSICS_CAPTURE_PORT',
                 'NOVPHY_PHYSICS_CAPTURE_V2_STRIDE', 'NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT',
                 'NOVPHY_ENVIRONMENT_SEED', TARGET_VARIABLE)
    environment = {key: os.environ.get(key) for key in variables}
    policy = FixedReplayPolicy(member['actions'][0])
    try:
        game = root / 'runtime'
        clone_player(output / 'player', game)
        files.install_level(game, member)
        _, scenario = files.materialize(member, member['template'], root / 'authority')
        if scenario.to_dict() != member['scenario']:
            raise ValueError('shared-history member differs from its frozen source authority')
        display, display_process = display_start.start_display(root / 'display.log')
        ports = set()
        while len(ports) < 3:
            ports.add(capture.old.capture.free_port())
        agent_port, game_port, physics_port = sorted(ports)
        aligned = root / 'aligned'
        os.environ.update(DISPLAY=display, XDG_DATA_HOME=str(root / 'xdg'),
            NOVPHY_PHYSICS_CAPTURE_PORT=str(physics_port), NOVPHY_PHYSICS_CAPTURE_V2_STRIDE='50',
            NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT=str(aligned),
            NOVPHY_ENVIRONMENT_SEED=str(member['engine_seed']),
            NOVPHY_NATIVE_DECISION_STEP=str(limits['decision_fixed_step']))
        result['ports'] = {'agent': agent_port, 'game': game_port, 'physics': physics_port}
        engine = capture.old.capture.start_engine(game, False, agent_port=agent_port,
            game_port=game_port, physics_port=physics_port)
        files.write(root / 'runtime.json', result)
        bridge = capture.old.capture.connect_with_retry('127.0.0.1', agent_port, timeout=180, deadline_seconds=60)
        endpoint = ScienceBirdsBridge('127.0.0.1', physics_port, timeout=30)
        bridge.configure(760001, capture.old.capture.PlayingMode.TRAINING)
        bridge.set_speed(1)
        capture.old.capture.prepare_for_play(bridge, timeout=60, poll_delay=.5)
        if bridge.get_current_level() != 1:
            raise ValueError('shared-history episode did not load its single assigned level')
        deadline = time.monotonic() + limits['history_ready_seconds']
        while not list(aligned.glob('decision-history-*/ready.json')):
            if time.monotonic() >= deadline:
                raise TimeoutError('native decision barrier readiness deadline; no retry')
            time.sleep(.1)
        history_root, rows = read_history(aligned, limits['decision_fixed_step'])
        identity = member['identity'] + ':shot-1'
        manifest = expose_history(rows, root / 'decision-1', scenario, identity, member['exposure_role'], policy)
        result.update(decision_history_root=str(history_root), observation_manifest=manifest['identity'],
                      predecision_frames=len(rows), predecision_fixed_steps=[r['fixed_step'] for r in rows])
        # Exercise the public readiness path while paused, then mimic inference
        # latency. Both direct captures must remain at the same native clock/RGB.
        capture.old.prepare_action(bridge, member['actions'][0])
        before = endpoint.get_observation_capture()
        time.sleep(limits['inference_hold_seconds'])
        after = endpoint.get_observation_capture()
        for name, frame in (('before', before), ('after', after)):
            files.write(root / f'paused-{name}.json', _plain_json(frame.metadata))
            (root / f'paused-{name}.png').write_bytes(frame.canonical_png)
        if (before.metadata['fixed_step'] != limits['decision_fixed_step']
                or after.metadata['fixed_step'] != limits['decision_fixed_step']
                or before.metadata['fixed_time_seconds'] != rows[-1]['fixed_time_seconds']
                or after.metadata['fixed_time_seconds'] != rows[-1]['fixed_time_seconds']
                or before.canonical_png != rows[-1]['canonical_png']
                or after.canonical_png != rows[-1]['canonical_png']):
            raise ValueError('native decision state/RGB advanced during public readiness or inference hold')
        decision = policy.choose()
        segment = capture.capture_segment(bridge, aligned, root / 'shot-1', member, scenario,
            identity, decision['action'], limits['shot_seconds'])
        result['segments'].append(segment)
        trace = NativeSegmentTrace(segment['native_root'])
        initial_metadata = files.read(Path(segment['native_root']) / 'frame_000001.json')
        initial_png = (Path(segment['native_root']) / 'frame_000001.png').read_bytes()
        if (segment['summary']['first_fixed_step'] != limits['decision_fixed_step']
                or initial_metadata['fixed_time_seconds'] != rows[-1]['fixed_time_seconds']
                or initial_png != rows[-1]['canonical_png']):
            raise ValueError('chosen shot pre-intervention state differs from its decision RGB/time')
        initial = next(trace.observed_samples())
        if not set(member['generated_slots']).issubset(segment['initial_entity_ids']):
            raise ValueError('shared-history initial shot omitted an authored object')
        # Engine-only branch comparison/actuation evidence never enters policy.
        files.write(root / 'initial-physics.json', initial)
        launch = [event for chunk in trace.trace.chunks() for event in chunk['events']
                  if event['event_type'] == 'bird_launched']
        if len(launch) != 1:
            raise ValueError('shared-history shot did not contain exactly one native launch')
        result.update(paused_state_and_rgb_equal=True, chosen_shot_matches_decision=True,
                      native_launch_offset=launch[0]['fixed_step'] - limits['decision_fixed_step'],
                      requested_release_time_ms=member['actions'][0]['release_time_ms'],
                      native_launch_seconds=(launch[0]['fixed_step'] - limits['decision_fixed_step']) * .0004)
        consume_shared_segment(root / 'shot-1/observation-trace', decision['action'], rows[-1]['fixed_time_seconds'], policy)
        result['policy'] = policy.evidence()
    except Exception as error:
        result['failure'] = f'{type(error).__name__}: {error}'
    finally:
        actions = [('connection.disconnect',connection.disconnect)
                   for connection in (bridge,endpoint) if connection is not None]
        actions.append(('stop_started_engine',lambda: capture.old.capture.stop_started_engine(engine)))
        if display_process is not None:
            actions.append(('display.terminate',lambda: capture.old.capture.terminate(display_process)))
        def restore_environment():
            for key,value in environment.items():
                if value is None: os.environ.pop(key,None)
                else: os.environ[key] = value
        actions.append(('environment.restore',restore_environment))
        cleanup_failures = cleanup_actions(actions)
        record_cleanup_failures(result,cleanup_failures)
    result.update(complete=result['failure'] is None, wall_seconds=time.monotonic() - started)
    persist_after_cleanup(cleanup_failures,
        lambda: files.write(output / 'results' / (member['identity'] + '.json'), result))
    print(f"Shared-history {member['identity']} complete={result['complete']} failure={result['failure']}", flush=True)
    return result
