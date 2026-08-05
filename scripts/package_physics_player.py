from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
from typing import Final


STAGE_SCHEMA: Final = "novphy_physics_player_stage_v1"
UNITY_VERSION: Final = "2019.4.41f2"
UNITY_CHANGESET: Final = "6b23d448b533"
CAPTURE_SCHEMA: Final = "physics_capture_v1"
PROTOCOL_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class PackagingError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_hashes(payload: Path) -> dict[str, str]:
    return {
        str(path.relative_to(payload)): sha256_file(path)
        for path in sorted(payload.rglob("*"))
        if path.is_file() and path.name != "provenance.json"
    }


def git_revision(worktree: Path) -> tuple[str, str]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=worktree, text=True, capture_output=True, check=True
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=worktree,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    source_diff = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", ".", ":(exclude).omo", ":(exclude).omo/**"],
        cwd=worktree,
        check=False,
    )
    if source_diff.returncode == 1:
        raise PackagingError("tracked product source differs from HEAD")
    source_diff.check_returncode()
    return head, tree


def write_manifest(payload: Path, worktree: Path, migration_provenance: Path) -> Path:
    head, tree = git_revision(worktree)
    manifest = {
        "schema_version": STAGE_SCHEMA,
        "unity": {
            "canonical_revision": "2019.3.4f1 (4f139db2fdbd)",
            "migrated_revision": f"{UNITY_VERSION} ({UNITY_CHANGESET})",
            "version": UNITY_VERSION,
            "changeset": UNITY_CHANGESET,
        },
        "project": {"git_head": head, "git_tree": tree},
        "capture": {"schema_version": CAPTURE_SCHEMA, "protocol_version": PROTOCOL_VERSION},
        "migration": {
            "provenance_file": migration_provenance.name,
            "provenance_sha256": sha256_file(migration_provenance),
        },
        "rollback_rule": "Build or verification failure never modifies canonical project, production player, or active data root.",
        "files": payload_hashes(payload),
    }
    path = payload / "provenance.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def create_archive(payload: Path, archive: Path) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        raw_tar = Path(temporary) / "payload.tar"
        with tarfile.open(raw_tar, "w", format=tarfile.PAX_FORMAT) as bundle:
            for path in sorted(payload.rglob("*")):
                info = bundle.gettarinfo(str(path), arcname=str(path.relative_to(payload)))
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                if path.is_file():
                    with path.open("rb") as stream:
                        bundle.addfile(info, stream)
                else:
                    bundle.addfile(info)
        subprocess.run(["gzip", "-n", "-9", "-c", str(raw_tar)], stdout=archive.open("wb"), check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--migration-provenance", type=Path, required=True)
    args = parser.parse_args()
    write_manifest(args.payload, args.worktree, args.migration_provenance)
    args.stage.mkdir(parents=True, exist_ok=True)
    archive = args.stage / "novphy-physics-player-2019.4.41f2.tar.gz"
    create_archive(args.payload, archive)
    (args.stage / "archive.sha256").write_text(
        f"{sha256_file(archive)}  {archive.name}\n", encoding="ascii"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
