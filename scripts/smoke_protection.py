from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
from subprocess import PIPE, Popen
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


def tree_digest(path: Path) -> str:
    if not path.exists():
        return "ABSENT"
    digest = hashlib.sha256()
    for directory, child_directories, filenames in os.walk(path):
        child_directories.sort()
        for filename in sorted(filenames):
            child = Path(directory) / filename
            digest.update(str(child.relative_to(path)).encode("utf-8"))
            digest.update(b"\0")
            with child.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def nested_manifest_digest(path: Path) -> str:
    """Digest the complete nested inventory and mutation-sensitive file metadata."""
    if not path.exists():
        return "ABSENT"
    digest = hashlib.sha256()
    with Popen(
        ["find", str(path), "-printf", "%P\\0%y\\0%s\\0%T@\\0%C@\\0%i\\0"],
        stdout=PIPE,
        stderr=PIPE,
    ) as process:
        if process.stdout is None:
            raise ProtectionError("cannot read nested manifest")
        for block in iter(lambda: process.stdout.read(1024 * 1024), b""):
            digest.update(block)
        stderr = b"" if process.stderr is None else process.stderr.read()
        if process.wait() != 0:
            raise ProtectionError("cannot build nested manifest: " + stderr.decode("utf-8", errors="replace").strip())
    return digest.hexdigest()


def protected_receipt(name: str, path: Path) -> str:
    return nested_manifest_digest(path) if name == "active_data" else tree_digest(path)
