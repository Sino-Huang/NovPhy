"""Share immutable player assets; retain private writable runtime paths."""
import os
from pathlib import Path
import shutil


def clone_player(source, destination):
    source, destination = Path(source), Path(destination)

    def copy_file(old, new):
        relative = Path(old).relative_to(source)
        # ABGameWorld.MaxScoreUpdate rewrites the current level XML. Transport
        # config and all root-level controls likewise belong to this attempt.
        if relative.parts[0] == '9001_Data' and 'Levels' not in relative.parts:
            os.link(old, new)
        elif relative.as_posix() in ('9001-player.x86_64', 'UnityPlayer.so', 'game_playing_interface.jar'):
            os.link(old, new)
        else:
            shutil.copy2(old, new)
        return new

    return shutil.copytree(source, destination, copy_function=copy_file)


def unique_file_bytes(roots):
    """Charge a shared inode once, not once for every hardlink to it."""
    seen, total = set(), 0
    for root in roots:
        for path in Path(root).rglob('*'):
            if not path.is_file():
                continue
            metadata = path.stat()
            key = (metadata.st_dev, metadata.st_ino)
            if key not in seen:
                seen.add(key)
                total += metadata.st_size
    return total
