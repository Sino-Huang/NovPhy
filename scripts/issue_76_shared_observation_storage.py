"""Deduplicate newly captured canonical RGB without changing observation content."""
import os
from pathlib import Path

from scripts.observation_trace import MANIFEST_NAME
from scripts.issue_76_expansion import read


def link_canonical_observations(raw_root, observation_root):
    """Run after the capture writer exits and only on this workflow's new traces.

    A resized agent transform remains a distinct bitmap. Under the declared
    identity transform the agent PNG is also the same retained engine render.
    Published paths, bytes, metadata and identities do not change.
    """
    raw_root, observation_root = Path(raw_root), Path(observation_root)
    manifest = read(observation_root / MANIFEST_NAME)
    roles = ['canonical_observation']
    if manifest['observation_configuration']['agent_representation']['transform']['method'] == 'identity':
        roles.append('agent_observation')
    saved = 0
    for frame in manifest['frame_records']:
        ordinal = frame['capture_metadata']['sequence']
        raw = raw_root / f'frame_{ordinal:06}.png'
        for role in roles:
            canonical = observation_root / frame[role]['relative_path']
            if raw.read_bytes() != canonical.read_bytes():
                raise ValueError('new observation copy differs from its retained engine RGB; no replacement')
            if raw.stat().st_ino == canonical.stat().st_ino and raw.stat().st_dev == canonical.stat().st_dev:
                continue
            temporary = canonical.with_suffix('.png.link')
            os.link(raw, temporary)
            saved += canonical.stat().st_size
            os.replace(temporary, canonical)
    return {'canonical_duplicate_bytes_removed': saved, 'published_content_changed': False}
