from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from scripts.verify_physics_player import VerificationError, safe_unpack, verify_payload


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PhysicsPlayerVerifierTests(unittest.TestCase):
    @staticmethod
    def _write_manifest(output: Path, relative: str, digest: str) -> None:
        manifest = {"schema_version": "novphy_physics_player_stage_v1", "unity": {"version": "2019.4.41f2", "changeset": "6b23d448b533"}, "capture": {"schema_version": "physics_capture_v1", "protocol_version": 1}, "files": {relative: digest}}
        (output / "provenance.json").write_text(json.dumps(manifest), encoding="utf-8")

    @staticmethod
    def _write_receipt(stage: Path, archive: Path, receipt_name: str) -> None:
        (stage / "archive.sha256").write_text(f"{_sha256(archive)}  {receipt_name}\n", encoding="ascii")

    def _make_stage(self, root: Path) -> Path:
        payload = root / "payload"
        payload.mkdir()
        player = payload / "9001.x86_64"
        player.write_bytes(b"player")
        player.chmod(0o755)
        self._write_manifest(payload, player.name, _sha256(player))
        archive = root / "novphy-physics-player.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(payload, arcname=".")
        stage = root / "stage"
        stage.mkdir()
        archive = archive.rename(stage / archive.name)
        self._write_receipt(stage, archive, archive.name)
        return stage

    @staticmethod
    def _run_cli(stage: Path, arguments: tuple[str, ...] = ()) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "scripts/verify_physics_player.py", "--stage", str(stage), *arguments, "--skip-runtime"],
            cwd=ROOT, text=True, capture_output=True, check=False)

    def _assert_rejected(self, result: subprocess.CompletedProcess[str], diagnostic: str) -> None:
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(diagnostic, result.stderr)

    def _rebuild_stage(self, root: Path, file_entry: tuple[str, str]) -> Path:
        stage = self._make_stage(root)
        archive = next(stage.glob("*.tar.gz"))
        unpacked = root / "rebuilt"
        unpacked.mkdir()
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(unpacked)
        self._write_manifest(unpacked, *file_entry)
        with tarfile.open(archive, "w:gz") as bundle:
            for path in sorted(unpacked.iterdir()):
                bundle.add(path, arcname=path.name)
        self._write_receipt(stage, archive, archive.name)
        return stage

    def _stage_with_member(self, root: Path, member: tarfile.TarInfo) -> Path:
        stage = self._make_stage(root)
        archive = next(stage.glob("*.tar.gz"))
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.addfile(member)
        self._write_receipt(stage, archive, archive.name)
        return stage

    def _stage_with_oversized_member(self, root: Path) -> Path:
        stage = self._make_stage(root)
        archive = next(stage.glob("*.tar.gz"))
        member = tarfile.TarInfo("oversized"); member.size = 512 * 1024 * 1024 + 1
        with gzip.open(archive, "wb") as compressed:
            compressed.write(member.tobuf(format=tarfile.USTAR_FORMAT))
            compressed.write(b"\0" * 1024)
        self._write_receipt(stage, archive, archive.name)
        return stage

    def _stage_with_receipt(self, root: Path, receipt: tuple[str, Path | None]) -> Path:
        receipt_name, link_target = receipt
        stage = self._make_stage(root)
        archive = next(stage.glob("*.tar.gz"))
        relocated = stage / receipt_name
        relocated.parent.mkdir(parents=True, exist_ok=True)
        if link_target is None:
            archive.rename(relocated)
        else:
            archive.rename(link_target)
            relocated.symlink_to(link_target)
        self._write_receipt(stage, relocated, receipt_name)
        return stage

    def _make_provenance(self, root: Path, fixture: tuple[str, Path | None]) -> Path:
        relative, link_target = fixture
        output = root / "output"
        output.mkdir(parents=True)
        payload = output / relative
        if link_target is None:
            payload.parent.mkdir(parents=True, exist_ok=True)
            payload.write_bytes(b"player")
            digest_source = payload
        elif "/" in relative:
            link_target.mkdir()
            digest_source = link_target / Path(relative).name
            digest_source.write_bytes(b"player")
            (output / Path(relative).parts[0]).symlink_to(link_target, target_is_directory=True)
        else:
            link_target.write_bytes(b"player")
            payload.symlink_to(link_target)
            digest_source = link_target
        self._write_manifest(output, relative, _sha256(digest_source))
        return output

    @staticmethod
    def _special_member(name: str, member_type: bytes) -> tarfile.TarInfo:
        member = tarfile.TarInfo(name)
        member.type = member_type
        return member

    def test_cli_verifies_archive_and_unpacked_file_checksums(self) -> None:
        # Given: a staged archive with pinned Unity and capture provenance.
        with tempfile.TemporaryDirectory() as temporary:
            stage = self._make_stage(Path(temporary))
            # When: the verifier checks the package without launching Unity.
            result = self._run_cli(stage)
            # Then: its machine-readable report proves both checksum layers.
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report["archive_checksum_verified"])
            self.assertTrue(report["payload_checksums_verified"])

    def test_cli_rejects_wrong_expected_archive_sha(self) -> None:
        # Given: a valid staged archive.
        with tempfile.TemporaryDirectory() as temporary:
            stage = self._make_stage(Path(temporary))
            # When: the caller supplies an incorrect expected digest.
            result = self._run_cli(stage, ("--expect-sha", "deadbeef"))
            # Then: verification fails closed.
            self._assert_rejected(result, "archive SHA-256 mismatch")

    def test_cli_rejects_tampered_payload_manifest(self) -> None:
        # Given: an archive whose declared player digest is false.
        with tempfile.TemporaryDirectory() as temporary:
            stage = self._rebuild_stage(Path(temporary), ("9001.x86_64", "0" * 64))
            # When: the verifier checks the rebuilt archive.
            result = self._run_cli(stage)
            # Then: the payload mismatch is rejected.
            self._assert_rejected(result, "payload SHA-256 mismatch")

    def test_entrypoint_imports_bridge_from_worktree_before_pythonpath(self) -> None:
        # Given: PYTHONPATH contains a conflicting checkout without request 70.
        with tempfile.TemporaryDirectory() as temporary:
            shadow = Path(temporary) / "shadow"
            package = shadow / "src" / "webui"
            package.mkdir(parents=True)
            (shadow / "src" / "__init__.py").write_text("", encoding="utf-8")
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "bridge.py").write_text("class ScienceBirdsBridge:\n    pass\n", encoding="utf-8")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(shadow)
            probe = ("import json, runpy, sys, time\nverifier = runpy.run_path(\"verify_physics_player.py\", run_name=\"verify_probe\")\ntry:\n"
                     "    verifier[\"connect_with_deadline\"](1, time.monotonic() - 1)\nexcept verifier[\"VerificationError\"]:\n    pass\nbridge = sys.modules[\"src.webui.bridge\"]\nprint(json.dumps({\"file\": bridge.__file__, \"request_70\": hasattr(bridge.ScienceBirdsBridge, \"get_physics_capture_v1\")}))\n")
            # When: the verifier is loaded exactly as a scripts-directory entrypoint.
            result = subprocess.run(
                [sys.executable, "-c", probe], cwd=ROOT / "scripts", env=environment,
                text=True, capture_output=True, check=False)
            # Then: runtime imports are pinned to this worktree.
            self.assertEqual(result.returncode, 0, result.stderr)
            resolved = json.loads(result.stdout)
            self.assertEqual(Path(resolved["file"]), ROOT / "src" / "webui" / "bridge.py")
            self.assertTrue(resolved["request_70"])

    def test_cli_rejects_unsafe_special_and_excessive_archive_members(self) -> None:
        members = {"traversal": tarfile.TarInfo("../escape"), "fifo": self._special_member("pipe", tarfile.FIFOTYPE)}
        for name, member in members.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                # Given: a checksummed archive with an unsafe member.
                stage = self._stage_with_member(Path(temporary), member)
                # When: the verifier checks the archive.
                result = self._run_cli(stage)
                # Then: the member is rejected with the stable diagnostic.
                self._assert_rejected(result, "unsafe archive member")
        with tempfile.TemporaryDirectory() as temporary:
            # Given: a gzip archive with a raw oversized USTAR header.
            stage = self._stage_with_oversized_member(Path(temporary))
            # When: the verifier checks the oversized declaration.
            result = self._run_cli(stage)
            # Then: the member is rejected before payload allocation.
            self._assert_rejected(result, "unsafe archive member")

    def test_cli_rejects_unsafe_manifest_file_paths(self) -> None:
        # Given: an archive manifest containing parent traversal.
        with tempfile.TemporaryDirectory() as temporary:
            stage = self._rebuild_stage(Path(temporary), ("../outside", "0" * 64))
            # When: the verifier checks the rebuilt archive.
            result = self._run_cli(stage)
            # Then: the unsafe manifest path is rejected.
            self._assert_rejected(result, "unsafe payload manifest path")

    def test_archive_checksum_path_confinement_rejects_nonlocal_names(self) -> None:
        for name in ("absolute", "parent", "nested", "backslash", "symlink"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                # Given: a valid archive named outside the receipt's single-component contract.
                root = Path(temporary)
                cases = {"absolute": (str(root / "absolute.tar.gz"), None), "parent": ("../parent.tar.gz", None),
                         "nested": ("nested/archive.tar.gz", None), "backslash": ("nested\\archive.tar.gz", None),
                         "symlink": ("linked-archive.tar.gz", root / "outside.tar.gz")}
                stage = self._stage_with_receipt(root, cases[name])
                # When: the CLI resolves the untrusted archive receipt.
                result = self._run_cli(stage)
                # Then: the receipt is rejected before the archive is opened.
                self._assert_rejected(result, "unsafe archive checksum path")

    def test_provenance_path_confinement_rejects_nonlocal_names(self) -> None:
        for name in ("absolute", "parent", "redundant-separator", "dot-component", "backslash"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                # Given: valid payload bytes named by a non-single-component manifest path.
                root = Path(temporary)
                cases = {"absolute": (str(root / "absolute-player"), None), "parent": ("../parent-player", None),
                         "redundant-separator": ("nested//player", None), "dot-component": ("nested/./player", None),
                         "backslash": ("nested\\player", None)}
                output = self._make_provenance(root, cases[name])
                # When: provenance verification resolves the untrusted payload name.
                with self.assertRaises(VerificationError) as raised:
                    verify_payload(output)
                # Then: no path outside the flat payload namespace is accepted.
                self.assertIn("unsafe payload manifest path", str(raised.exception))

    def test_provenance_symlink_escape_confinement_is_rejected(self) -> None:
        # Given: manifest paths whose leaf or parent escapes through a symlink.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures = {"leaf": ("player", root / "outside-leaf"),
                        "parent": ("linked/player", root / "outside-parent")}
            for name, fixture in fixtures.items():
                with self.subTest(name=name):
                    output = self._make_provenance(root / name, fixture)
                    # When: provenance verification resolves the payload path.
                    with self.assertRaises(VerificationError) as raised:
                        verify_payload(output)
                    # Then: a checksum match outside the extraction root is still rejected.
                    self.assertIn("unsafe payload manifest path", str(raised.exception))

    def test_archive_member_confinement_rejects_paths_types_and_links(self) -> None:
        symlink = self._special_member("symlink", tarfile.SYMTYPE); symlink.linkname = "target"
        hardlink = self._special_member("hardlink", tarfile.LNKTYPE); hardlink.linkname = "target"
        members = {"absolute": tarfile.TarInfo("/absolute"), "parent": tarfile.TarInfo("../parent"),
                   "fifo": self._special_member("fifo", tarfile.FIFOTYPE),
                   "character-device": self._special_member("character-device", tarfile.CHRTYPE),
                   "block-device": self._special_member("block-device", tarfile.BLKTYPE),
                   "socket": self._special_member("socket", b"s"), "symlink": symlink, "hardlink": hardlink}
        for name, member in members.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                # Given: a checksummed archive containing one unsafe path or member type.
                stage = self._stage_with_member(Path(temporary), member)
                # When: the CLI preflights the archive.
                result = self._run_cli(stage)
                # Then: every unsafe member has one diagnostic class.
                self._assert_rejected(result, "unsafe archive member")

    def test_safe_unpack_confinement_extracts_regular_files_and_directories(self) -> None:
        # Given: an archive containing only a confined directory and regular file.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"; source.mkdir()
            (source / "player").write_bytes(b"player")
            archive = root / "valid.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                bundle.add(source, arcname="payload")
            output = root / "output"; output.mkdir()
            # When: safe_unpack extracts the archive directly.
            safe_unpack(archive, output)
            # Then: only the declared regular directory tree is materialized.
            extracted = sorted(path.relative_to(output).as_posix() for path in output.rglob("*"))
            self.assertEqual(extracted, ["payload", "payload/player"])
            self.assertEqual((output / "payload" / "player").read_bytes(), b"player")
            self._write_manifest(output, "payload/player", _sha256(output / "payload" / "player"))
            verify_payload(output)

    def test_safe_unpack_confinement_preflight_prevents_partial_extraction(self) -> None:
        # Given: a regular file precedes an unsafe FIFO in archive order.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "first"
            source.write_bytes(b"first")
            archive = root / "partial.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                bundle.add(source, arcname="first")
                bundle.addfile(self._special_member("fifo", tarfile.FIFOTYPE))
            output = root / "output"
            output.mkdir()
            # When: safe_unpack preflights every member before extraction starts.
            error_message = ""
            try:
                safe_unpack(archive, output)
            except VerificationError as error:
                error_message = str(error)
            # Then: rejection leaves no partially extracted regular file behind.
            self.assertEqual(list(output.iterdir()), [])
            self.assertIn("unsafe archive member", error_message)


if __name__ == "__main__":
    unittest.main()
