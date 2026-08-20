from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
import tarfile
from typing import Final


MAX_ARCHIVE_MEMBER_SIZE: Final = 512 * 1024 * 1024


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
