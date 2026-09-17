from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import issue_76_shared_history_player as player
from scripts import issue_76_shared_history_capture as capture
from scripts.issue_76_shared_player_storage import clone_player, unique_file_bytes


class SharedHistoryPlayerTests(unittest.TestCase):
    def test_accepted_segment_does_not_repeat_latest_history_at_zero_dt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'next.png').write_bytes(b'next real RGB')
            (root / capture.MANIFEST_NAME).write_text(json.dumps({'frame_records': [
                {'fixed_time_seconds': 12, 'agent_observation': {'relative_path': 'already-observed.png'}},
                {'fixed_time_seconds': 12.02, 'agent_observation': {'relative_path': 'next.png'}},
            ]}))
            policy = Mock()
            action = {'drag_x': -80, 'drag_y': 10, 'tap_time_ms': 0, 'release_time_ms': 1000}
            capture.consume_shared_segment(root, action, 12, policy)
            self.assertEqual(policy.method_calls, [
                unittest.mock.call.executed(action, 12),
                unittest.mock.call.observe(b'next real RGB', 12.02),
            ])
            policy.reset_mock()
            with self.assertRaisesRegex(ValueError, 'shot clock differs'):
                capture.consume_shared_segment(root, action, 11, policy)
            policy.executed.assert_not_called()

    def test_endpoint_and_aligned_capture_share_the_same_world_renderer(self):
        aligned = '''    public void Capture(PhysicalSnapshotRuntime runtime)
    {
        Camera camera = Camera.main;
        int width = Screen.width;
        int height = Screen.height;
        if (camera == null || width <= 0 || height <= 0) throw new InvalidOperationException();
        byte[] png = RenderWorldCamera();

        sequence++;
    }'''
        endpoint = '''                    Texture2D observationTexture = ScreenCapture.CaptureScreenshotAsTexture();
                    byte[] canonicalPng = observationTexture.EncodeToPNG();
                    observationResponse = ObservationCaptureProtocol.BuildCaptureEnvelope();'''
        changed_aligned = player.aligned_renderer_source(aligned)
        changed_endpoint = player.endpoint_source(endpoint)
        self.assertEqual(changed_aligned.count('RenderWorldCamera()'), 1)
        self.assertIn('byte[] png = RenderCanonicalRgb(camera, width, height)', changed_aligned)
        self.assertIn('PhysicsCaptureV2AlignedObservationRecorder.RenderCanonicalRgb(', changed_endpoint)
        self.assertNotIn('CaptureScreenshotAsTexture', changed_endpoint)

    def test_history_requires_sealed_order_and_native_timestamp_spacing(self):
        with tempfile.TemporaryDirectory() as temporary:
            aligned = Path(temporary)
            folder = aligned / 'decision-history-test'
            folder.mkdir()
            with self.assertRaisesRegex(ValueError, 'sealed ready'):
                capture.read_history(aligned, 30000)
            (folder / 'ready.json').write_text(json.dumps({
                'schema': 'issue_76_native_decision_history_v1', 'target_fixed_step': 30000,
                'frame_count': 3, 'stride_native_steps': 50, 'native_step_seconds': .0004,
                'capture_id': folder.name,
            }))
            for ordinal, step in enumerate((29900, 29950, 30000), 1):
                path = folder / f'frame_{ordinal:06}.json'
                path.write_text(json.dumps({'fixed_step': step, 'fixed_time_seconds': step * .0004,
                                            'capture_id': folder.name, 'sequence': ordinal}))
                path.with_suffix('.png').write_bytes(b'RGB fixture')
            _, rows = capture.read_history(aligned, 30000)
            self.assertEqual([r['fixed_step'] for r in rows], [29900, 29950, 30000])
            path = folder / 'frame_000003.json'
            row = json.loads(path.read_text())
            row['fixed_time_seconds'] += .1
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'twenty milliseconds'):
                capture.read_history(aligned, 30000)

    def test_exposure_sends_only_rgb_and_time_not_engine_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            frames = []
            for index in range(3):
                name = f'agent-{index}.png'
                (destination / name).write_bytes(f'RGB-{index}'.encode())
                frames.append({'agent_observation': {'relative_path': name},
                               'fixed_time_seconds': 12 + .02 * index, 'engine_only_objects': ['pig']})
            policy = Mock()
            with patch.object(capture, 'persist_observation_trace', return_value={'frame_records': frames}), \
                    patch.object(capture.capture.old.capture, '_observation_bindings', return_value={}):
                capture.expose_history([{'future_action': 'not a policy input'}], destination, None, 'shot', 'training', policy)
            self.assertEqual([call.args for call in policy.observe.call_args_list],
                             [(f'RGB-{i}'.encode(), 12 + .02 * i) for i in range(3)])

    def test_runtime_seals_initial_shot_snapshot_before_release(self):
        source = '''    public void BeginV2Shot()
    {
        CaptureNativeObservation();
    }

    public PhysicalCaptureResult FinalizeShot() {}
    public void ResetLevel() {
        v2InterventionObserved = false;
    }'''
        changed = player.runtime_source(source)
        self.assertIn('NativeDecisionBarrier.Reset(this)', changed)
        self.assertLess(changed.index('RequireReady(this)'), changed.index('CaptureNativeObservation()'))
        self.assertLess(changed.index('CaptureNativeObservation()'), changed.index('ReleaseAfterShotSnapshot(this)'))

    def test_world_blocks_queued_steps_and_captures_after_bookkeeping(self):
        source = '\tprivate void FixedUpdate()\n\t{\n\t\t\tManageBirds();\n\t\t\tTakeAction();\n\t}'
        changed = player.world_source(source)
        self.assertLess(changed.index('Paused) return'), changed.index('ManageBirds()'))
        self.assertLess(changed.index('TakeAction()'), changed.index('AfterBookkeeping(runtime)'))

    def test_navigation_cancels_barrier_but_readiness_does_not_resume_it(self):
        source = ''.join(f'\tprivate IEnumerator {name}(JSONNode data)\n\t{{\n\tTime.timeScale = 1f;\n\t}}\n'
                         for name in ('SelectNextAvailableLevel', 'SelectLevel'))
        changed = player.connection_source(source)
        self.assertEqual(changed.count('NativeDecisionBarrier.Cancel()'), 2)
        self.assertNotIn('Time.timeScale = 1f;', changed)

    def test_player_shares_assets_but_level_and_controls_remain_private(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            paths = ('9001_Data/sharedassets0.assets', '9001_Data/StreamingAssets/Levels/test.xml',
                     '9001-player.x86_64', 'UnityPlayer.so', 'game_playing_interface.jar',
                     '9001.x86_64', 'config.xml', 'serverbackup')
            for name in paths:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            clone_player(source, root / 'a')
            clone_player(source, root / 'b')
            for name in paths:
                shared = name in paths[:1] + paths[2:5]
                self.assertEqual((source / name).stat().st_ino == (root / 'a' / name).stat().st_ino, shared)
            (root / 'a' / paths[1]).write_text('private score update')
            self.assertEqual((source / paths[1]).read_text(), paths[1])
            self.assertEqual((root / 'b' / paths[1]).read_text(), paths[1])
            expected = sum(len(name) for name in paths[:1] + paths[2:5])
            expected += sum((tree / name).stat().st_size for tree in (source, root / 'a', root / 'b')
                            for name in paths[1:2] + paths[5:])
            self.assertEqual(unique_file_bytes((source, root / 'a', root / 'b')), expected)


if __name__ == '__main__':
    unittest.main()
