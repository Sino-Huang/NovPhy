from __future__ import annotations

from io import BytesIO
from contextlib import redirect_stdout
import json
from pathlib import Path
from io import StringIO
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts.verify_physics_player import (
    ARCHIVE_NAME,
    REQUIRED_FILES,
    UnsafeArchiveMemberError,
    VerificationError,
    main,
    safe_unpack,
    verify_physics_player_archive,
)


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


class ArchiveVerificationTests(unittest.TestCase):
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

    def write_stage(self, root: Path, mutate=None, extra: dict[str, bytes] | None = None) -> Path:
        stage = root / "stage"
        stage.mkdir()
        payload = {path: ("payload:" + path).encode("utf-8") for path in REQUIRED_FILES}
        payload.update(extra or {})
        provenance = {
            "schema_version": "novphy_physics_player_stage_v1",
            "unity": {"version": "2019.4.41f2"},
            "capture": {
                "schema_version": "physics_capture_v2_engine_v1",
                "protocol_version": 1,
            },
            "project": {"git_head": "source-commit", "git_tree": "source-tree"},
            "files": {path: len(content) for path, content in payload.items()},
        }
        if mutate is not None:
            mutate(provenance)
        with tarfile.open(stage / ARCHIVE_NAME, "w:gz") as bundle:
            for name, content in {**payload, "provenance.json": (
                json.dumps(provenance).encode("utf-8")
            )}.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                bundle.addfile(info, BytesIO(content))
        return stage

    def test_verifies_v2_provenance_declared_inventory_and_required_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stage = self.write_stage(Path(temporary))

            result = verify_physics_player_archive(stage, physics_v2=True)

            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["source_snapshot_commit"], "source-commit")
            self.assertEqual(result["source_tree"], "source-tree")
            self.assertEqual(result["capture_schema"], "physics_capture_v2_engine_v1")
            self.assertEqual(result["declared_file_count"], len(REQUIRED_FILES))

    def test_rejects_wrong_or_missing_provenance(self) -> None:
        cases = {
            "stage schema": lambda value: value.update(schema_version="old"),
            "Unity": lambda value: value["unity"].update(version="2020.1"),
            "capture": lambda value: value["capture"].update(schema_version="physics_capture_v1"),
            "protocol": lambda value: value["capture"].update(protocol_version=2),
            "source commit": lambda value: value["project"].update(git_head=""),
            "source tree": lambda value: value["project"].update(git_tree=""),
            "inventory": lambda value: value["files"].pop("UnityPlayer.so"),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                stage = self.write_stage(Path(temporary), mutate=mutate)
                with self.assertRaises(VerificationError):
                    verify_physics_player_archive(stage, physics_v2=True)

    def test_rejects_payload_missing_a_required_file_even_when_declared(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stage = self.write_stage(Path(temporary))
            archive = stage / ARCHIVE_NAME
            unpacked: dict[str, bytes] = {}
            with tarfile.open(archive, "r:gz") as source:
                for member in source.getmembers():
                    if member.name == "UnityPlayer.so":
                        continue
                    stream = source.extractfile(member)
                    assert stream is not None
                    unpacked[member.name] = stream.read()
            provenance = json.loads(unpacked["provenance.json"])
            provenance["files"].pop("UnityPlayer.so")
            unpacked["provenance.json"] = json.dumps(provenance).encode("utf-8")
            with tarfile.open(archive, "w:gz") as bundle:
                for name, content in unpacked.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    bundle.addfile(info, BytesIO(content))

            with self.assertRaisesRegex(VerificationError, "required file"):
                verify_physics_player_archive(stage, physics_v2=True)

    def test_cli_prints_json_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stage = self.write_stage(Path(temporary))
            output = StringIO()

            with redirect_stdout(output):
                status = main(["--stage", str(stage), "--physics-v2"])

            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output.getvalue())["status"], "verified")

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
