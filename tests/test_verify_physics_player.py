from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts.verify_physics_player import UnsafeArchiveMemberError, safe_unpack


class SafeUnpackTests(unittest.TestCase):
    def archive(self, root: Path, members: list[tuple[tarfile.TarInfo, bytes]]) -> Path:
        archive = root / "player.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            for info, content in members:
                bundle.addfile(info, BytesIO(content) if info.isfile() else None)
        return archive

    def file(self, name: str, content: bytes = b"payload") -> tuple[tarfile.TarInfo, bytes]:
        info = tarfile.TarInfo(name)
        info.size = len(content)
        return info, content

    def test_extracts_regular_confined_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.archive(root, [self.file("payload/player"), self.file("provenance.json", b"{}")])
            output = root / "output"
            output.mkdir()
            safe_unpack(archive, output)
            self.assertEqual((output / "payload/player").read_bytes(), b"payload")
            self.assertEqual((output / "provenance.json").read_bytes(), b"{}")

    def test_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.archive(root, [self.file("../escape")])
            output = root / "output"
            output.mkdir()
            with self.assertRaisesRegex(UnsafeArchiveMemberError, "unsafe archive member"):
                safe_unpack(archive, output)
            self.assertFalse((root / "escape").exists())

    def test_rejects_absolute_and_windows_paths(self) -> None:
        for name in ("/tmp/escape", "C:/escape"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = self.archive(root, [self.file(name)])
                output = root / "output"
                output.mkdir()
                with self.assertRaises(UnsafeArchiveMemberError):
                    safe_unpack(archive, output)

    def test_rejects_links_and_special_members(self) -> None:
        for member_type in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE):
            with self.subTest(member_type=member_type), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                info = tarfile.TarInfo("unsafe")
                info.type = member_type
                info.linkname = "target"
                archive = self.archive(root, [(info, b"")])
                output = root / "output"
                output.mkdir()
                with self.assertRaises(UnsafeArchiveMemberError):
                    safe_unpack(archive, output)

    def test_rejects_oversized_archive_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.archive(root, [self.file("oversized.bin", b"12345")])
            output = root / "output"
            output.mkdir()
            with patch("scripts.verify_physics_player.MAX_ARCHIVE_MEMBER_SIZE", 4):
                with self.assertRaisesRegex(UnsafeArchiveMemberError, "oversized.bin"):
                    safe_unpack(archive, output)
            self.assertFalse((output / "oversized.bin").exists())

    def test_preflight_rejects_unsafe_member_before_partial_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.archive(root, [self.file("safe.txt"), self.file("../escape")])
            output = root / "output"
            output.mkdir()
            with self.assertRaises(UnsafeArchiveMemberError):
                safe_unpack(archive, output)
            self.assertEqual(tuple(output.iterdir()), ())
            self.assertFalse((root / "escape").exists())


if __name__ == "__main__":
    unittest.main()
