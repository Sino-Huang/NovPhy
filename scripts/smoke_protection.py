from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Final


ACTIVE_DATA_RELATIVE: Final = Path("data/novphy_rollouts_dataset_20260708_171531")


@dataclass(frozen=True, slots=True)
class ProtectionError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


def canonical_root_from_git(repo_root: Path) -> Path:
    """Resolve the canonical checkout from Git worktree provenance."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProtectionError("cannot resolve canonical root from Git common directory") from error
    common_directory = Path(completed.stdout.strip()).resolve()
    if common_directory.name != ".git" or not common_directory.is_dir():
        raise ProtectionError("Git common directory is not a canonical checkout")
    return common_directory.parent


def protected_roots(canonical_root: Path) -> dict[str, Path]:
    return {
        "canonical_project": canonical_root / "tasks/task_template_designer",
        "production_player": canonical_root / "sciencebirdsgames/Linux",
        "active_data": canonical_root / ACTIVE_DATA_RELATIVE,
    }


def tree_listing(path: Path) -> str:
    """Describe a tree using path and filesystem metadata, without content hashing."""
    if not path.exists():
        return "ABSENT"
    entries: list[str] = []
    for directory, child_directories, filenames in os.walk(path):
        child_directories.sort()
        for filename in sorted(filenames):
            child = Path(directory) / filename
            status = child.stat()
            entries.append(
                f"{child.relative_to(path)}\0{status.st_size}\0{status.st_mtime_ns}\0{status.st_ino}"
            )
    return "\n".join(entries)


def nested_manifest_listing(path: Path) -> str:
    """Describe the complete nested inventory and mutation-sensitive metadata."""
    if not path.exists():
        return "ABSENT"
    entries: list[str] = []
    for directory, child_directories, filenames in os.walk(path):
        child_directories.sort()
        for name in (*child_directories, *sorted(filenames)):
            child = Path(directory) / name
            status = child.lstat()
            entries.append(
                f"{child.relative_to(path)}\0{status.st_mode}\0{status.st_size}\0"
                f"{status.st_mtime_ns}\0{status.st_ctime_ns}\0{status.st_ino}"
            )
    return "\n".join(entries)


def protected_snapshot(name: str, path: Path) -> str:
    return nested_manifest_listing(path) if name == "active_data" else tree_listing(path)
