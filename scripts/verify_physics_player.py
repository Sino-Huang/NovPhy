from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import tarfile
import tempfile
from typing import Any, Final, Mapping, Sequence


MAX_ARCHIVE_MEMBER_SIZE: Final = 512 * 1024 * 1024
ARCHIVE_NAME: Final = "novphy-physics-player-2019.4.41f2.tar.gz"
STAGE_SCHEMA: Final = "novphy_physics_player_stage_v1"
UNITY_VERSION: Final = "2019.4.41f2"
PROTOCOL_VERSION: Final = 1
V1_CAPTURE_SCHEMA: Final = "physics_capture_v1"
V2_CAPTURE_SCHEMA: Final = "physics_capture_v2_engine_v1"
OBSERVATION_CAPTURE_SCHEMA: Final = "observation_capture_engine_v1"
REQUIRED_FILES: Final = (
    "9001.x86_64",
    "9001-player.x86_64",
    "UnityPlayer.so",
    "game_playing_interface.jar",
    "config.xml",
    "9001_Data/Managed/Assembly-CSharp.dll",
)


class VerificationError(RuntimeError):
    pass


class UnsafeArchiveMemberError(VerificationError):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"unsafe archive member: {name}")


def safe_unpack(archive: Path, output: Path) -> None:
    """Extract regular archive files and directories within ``output`` only."""
    output_root = output.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        members: list[tarfile.TarInfo] = []
        while (member := bundle.next()) is not None:
            posix_path = PurePosixPath(member.name)
            windows_path = PureWindowsPath(member.name)
            is_root_directory = member.isdir() and member.name in {".", "./"}
            has_confined_type = member.isfile() or member.isdir()
            has_confined_size = not member.isfile() or 0 <= member.size <= MAX_ARCHIVE_MEMBER_SIZE
            if (
                not has_confined_type
                or not has_confined_size
                or (not posix_path.parts and not is_root_directory)
                or posix_path.is_absolute()
                or bool(windows_path.anchor)
                or ".." in posix_path.parts
                or ".." in windows_path.parts
            ):
                raise UnsafeArchiveMemberError(member.name)
            destination = output_root.joinpath(*posix_path.parts).resolve()
            try:
                destination.relative_to(output_root)
            except ValueError as error:
                raise UnsafeArchiveMemberError(member.name) from error
            members.append(member)
        bundle.extractall(output_root, members=members, filter="data")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VerificationError(f"archive {field} provenance is malformed")
    return value


def _load_provenance(root: Path) -> Mapping[str, Any]:
    path = root / "provenance.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError("archive provenance manifest is missing or malformed") from error
    return _mapping(value, "root")


def _inventory(root: Path) -> dict[str, int]:
    return {
        path.relative_to(root).as_posix(): path.stat().st_size
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() != "provenance.json"
    }


def verify_physics_player_archive(
    stage: Path,
    *,
    physics_v2: bool,
    observation_v1: bool = False,
) -> dict[str, Any]:
    """Validate the hash-free package contract without launching the player."""
    stage = Path(stage)
    archive = stage / ARCHIVE_NAME
    if not archive.is_file():
        raise VerificationError(f"physics player archive is missing: {archive}")
    with tempfile.TemporaryDirectory(prefix="novphy_physics_verify_") as temporary:
        root = Path(temporary)
        safe_unpack(archive, root)
        provenance = _load_provenance(root)
        if provenance.get("schema_version") != STAGE_SCHEMA:
            raise VerificationError("archive stage provenance schema is unsupported")
        unity = _mapping(provenance.get("unity"), "Unity")
        if unity.get("version") != UNITY_VERSION:
            raise VerificationError("archive Unity provenance is unsupported")
        capture = _mapping(provenance.get("capture"), "capture")
        if physics_v2 and observation_v1:
            raise VerificationError("archive capture profile is ambiguous")
        expected_capture = (
            OBSERVATION_CAPTURE_SCHEMA
            if observation_v1
            else V2_CAPTURE_SCHEMA if physics_v2 else V1_CAPTURE_SCHEMA
        )
        if capture.get("schema_version") != expected_capture:
            raise VerificationError("archive capture provenance is unsupported")
        if capture.get("protocol_version") != PROTOCOL_VERSION:
            raise VerificationError("archive capture protocol provenance is unsupported")
        project = _mapping(provenance.get("project"), "project")
        source_commit = project.get("git_head")
        source_tree = project.get("git_tree")
        if not isinstance(source_commit, str) or not source_commit:
            raise VerificationError("archive source commit provenance is missing")
        if not isinstance(source_tree, str) or not source_tree:
            raise VerificationError("archive source tree provenance is missing")
        declared = _mapping(provenance.get("files"), "file inventory")
        if any(
            not isinstance(path, str)
            or not path
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            for path, size in declared.items()
        ):
            raise VerificationError("archive declared-size inventory is malformed")
        actual = _inventory(root)
        if dict(declared) != actual:
            raise VerificationError("archive declared-size inventory differs from payload")
        missing = [path for path in REQUIRED_FILES if path not in actual]
        if missing:
            raise VerificationError("archive required file is missing: " + missing[0])
        return {
            "status": "verified",
            "stage": str(stage),
            "archive": str(archive),
            "source_snapshot_commit": source_commit,
            "source_tree": source_tree,
            "unity_version": unity["version"],
            "capture_schema": capture["schema_version"],
            "protocol_version": capture["protocol_version"],
            "declared_file_count": len(actual),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--physics-v2", action="store_true")
    parser.add_argument("--observation-v1", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = verify_physics_player_archive(
        args.stage,
        physics_v2=args.physics_v2,
        observation_v1=args.observation_v1,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
