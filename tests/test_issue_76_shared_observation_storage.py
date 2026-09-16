import json
from pathlib import Path
import tempfile
import unittest

from scripts.issue_76_shared_observation_storage import link_canonical_observations
from scripts.issue_76_shared_player_storage import unique_file_bytes
from scripts.observation_trace import MANIFEST_NAME


class SharedObservationStorageTests(unittest.TestCase):
    def fixture(self, root):
        raw, observed = root / 'raw', root / 'observed'
        raw.mkdir()
        observed.mkdir()
        (raw / 'frame_000001.png').write_bytes(b'canonical RGB fixture')
        (observed / 'canonical.png').write_bytes(b'canonical RGB fixture')
        (observed / 'agent.png').write_bytes(b'separate shared agent transform')
        (observed / MANIFEST_NAME).write_text(json.dumps({
            'observation_configuration': {'agent_representation': {'transform': {'method': 'resize'}}},
            'frame_records': [{
            'capture_metadata': {'sequence': 1},
            'canonical_observation': {'relative_path': 'canonical.png'},
            'agent_observation': {'relative_path': 'agent.png'},
        }]}))
        return raw, observed

    def test_identity_agent_and_canonical_copies_both_share_the_engine_png(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, observed = self.fixture(root)
            manifest = json.loads((observed / MANIFEST_NAME).read_text())
            manifest['observation_configuration']['agent_representation']['transform']['method'] = 'identity'
            (observed / MANIFEST_NAME).write_text(json.dumps(manifest))
            (observed / 'agent.png').write_bytes((raw / 'frame_000001.png').read_bytes())
            result = link_canonical_observations(raw, observed)
            self.assertEqual(result['canonical_duplicate_bytes_removed'], 2 * len(b'canonical RGB fixture'))
            self.assertEqual((observed / 'canonical.png').stat().st_ino, (raw / 'frame_000001.png').stat().st_ino)
            self.assertEqual((observed / 'agent.png').stat().st_ino, (raw / 'frame_000001.png').stat().st_ino)

    def test_replaces_only_equal_new_canonical_copy_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, observed = self.fixture(root)
            before = unique_file_bytes((root,))
            result = link_canonical_observations(raw, observed)
            self.assertEqual(result['canonical_duplicate_bytes_removed'], len(b'canonical RGB fixture'))
            self.assertFalse(result['published_content_changed'])
            self.assertEqual(unique_file_bytes((root,)), before - len(b'canonical RGB fixture'))
            self.assertEqual((observed / 'canonical.png').read_bytes(), b'canonical RGB fixture')
            self.assertEqual((observed / 'agent.png').read_bytes(), b'separate shared agent transform')
            self.assertEqual(link_canonical_observations(raw, observed)['canonical_duplicate_bytes_removed'], 0)

    def test_different_copy_is_retained_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, observed = self.fixture(root)
            (observed / 'canonical.png').write_bytes(b'different capture')
            with self.assertRaisesRegex(ValueError, 'no replacement'):
                link_canonical_observations(raw, observed)
            self.assertEqual((observed / 'canonical.png').read_bytes(), b'different capture')
            self.assertEqual((raw / 'frame_000001.png').read_bytes(), b'canonical RGB fixture')


if __name__ == '__main__':
    unittest.main()
