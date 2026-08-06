from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts import package_physics_player as packager
from scripts.package_physics_player import PackagingError, create_archive, git_revision


ROOT = Path(__file__).resolve().parents[1]
PACKAGER = ROOT / "scripts" / "package_physics_player.py"
ARCHIVE_NAME = "novphy-physics-player-2019.4.41f2.tar.gz"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PhysicsPlayerPackagerTests(unittest.TestCase):
    def _repository(self, root: Path) -> Path:
        repository = root / "repository"
        repository.mkdir()
        subprocess.run(("git", "init", "-q", str(repository)), check=True)
        subprocess.run(("git", "config", "user.email", "test@example.invalid"), cwd=repository, check=True)
        subprocess.run(("git", "config", "user.name", "Test User"), cwd=repository, check=True)
        (repository / "product.txt").write_text("committed product\n", encoding="utf-8")
        wrapper = repository / "scripts" / "9001-player-wrapper.sh"
        wrapper.parent.mkdir()
        wrapper.write_text("#!/usr/bin/env bash\nexec ./9001-player.x86_64 \"$@\"\n", encoding="utf-8")
        unity_source = repository / "tasks" / "task_template_designer" / "Assets" / "BuildSource.cs"
        unity_source.parent.mkdir(parents=True)
        unity_source.write_text("internal static class BuildSource {}\n", encoding="utf-8")
        notes = repository / ".omo" / "notepads"
        notes.mkdir(parents=True)
        (notes / "notes.md").write_text("initial evidence\n", encoding="utf-8")
        subprocess.run(("git", "add", "product.txt", "scripts/9001-player-wrapper.sh", "tasks", ".omo/notepads/notes.md"), cwd=repository, check=True)
        subprocess.run(("git", "commit", "-qm", "test fixture"), cwd=repository, check=True)
        return repository

    def _package(self, paths: tuple[Path, Path, Path, Path], extra_arguments: tuple[str, ...] = ()) -> subprocess.CompletedProcess[str]:
        repository, payload, stage, migration = paths
        return subprocess.run((sys.executable, str(PACKAGER), "--payload", str(payload), "--stage", str(stage), "--worktree", str(repository), "--migration-provenance", str(migration)) + extra_arguments, text=True, capture_output=True, check=False)

    def _run_in_process(self, paths: tuple[Path, Path, Path, Path]) -> int:
        repository, payload, stage, migration = paths
        arguments = [str(PACKAGER), "--payload", str(payload), "--stage", str(stage), "--worktree", str(repository),
                     "--migration-provenance", str(migration)]
        with patch.object(sys, "argv", arguments):
            return packager.main()

    def _fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        repository = self._repository(root)
        payload = root / "payload"
        payload.mkdir()
        stage = root / "stage"
        migration = root / "migration.json"
        migration.write_text('{"migration":"verified"}\n', encoding="utf-8")
        return repository, payload, stage, migration

    def test_cli_preserves_unity_metadata_and_payload_hashes(self) -> None:
        # Given: a committed source tree and a built player payload.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = self._repository(root)
            payload = root / "payload"
            payload.mkdir()
            player = payload / "9001.x86_64"
            player.write_bytes(b"player payload")
            migration = root / "migration.json"
            migration.write_text('{"migration":"verified"}\n', encoding="utf-8")
            stage = root / "stage"

            # When: the packaging CLI creates the deterministic archive.
            result = self._package((repository, payload, stage, migration))

            # Then: exact Unity metadata and existing payload-file hashes are retained.
            self.assertEqual(result.returncode, 0, result.stderr)
            with tarfile.open(stage / ARCHIVE_NAME, "r:gz") as archive:
                manifest_file = archive.extractfile("provenance.json")
                self.assertIsNotNone(manifest_file)
                manifest = json.loads(manifest_file.read())
            self.assertEqual(manifest["unity"]["version"], "2019.4.41f2")
            self.assertEqual(manifest["unity"]["changeset"], "6b23d448b533")
            self.assertEqual(manifest["files"], {"9001.x86_64": _sha256(player)})

    def test_cli_is_stable_when_only_evidence_and_generated_artifacts_change(self) -> None:
        # Given: one committed source tree with an unchanged built payload.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = self._repository(root)
            payload = root / "payload"
            payload.mkdir()
            (payload / "9001.x86_64").write_bytes(b"stable player payload")
            migration = root / "migration.json"
            migration.write_text('{"migration":"verified"}\n', encoding="utf-8")
            stage = repository / "sciencebirdsgames" / "physics-v1"
            first = self._package((repository, payload, stage, migration))
            self.assertEqual(first.returncode, 0, first.stderr)
            first_manifest = (payload / "provenance.json").read_bytes()
            first_archive = (stage / ARCHIVE_NAME).read_bytes()
            (repository / ".omo" / "notepads" / "notes.md").write_text("changed evidence\n", encoding="utf-8")
            cache = repository / "tasks" / "task_template_designer" / "Library"
            cache.mkdir(parents=True)
            (cache / "cache.bin").write_bytes(b"generated cache")
            (stage / "unity-build.log").write_text("generated log\n", encoding="utf-8")

            # When: the same payload is packaged again from that exact commit.
            second = self._package((repository, payload, stage, migration))

            # Then: committed provenance and the archive are byte-identical.
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual((payload / "provenance.json").read_bytes(), first_manifest)
            self.assertEqual((stage / ARCHIVE_NAME).read_bytes(), first_archive)
            manifest = json.loads(first_manifest)
            head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=repository, text=True, capture_output=True, check=True).stdout.strip()
            tree = subprocess.run(("git", "rev-parse", "HEAD^{tree}"), cwd=repository, text=True, capture_output=True, check=True).stdout.strip()
            self.assertEqual(manifest["project"], {"git_head": head, "git_tree": tree})

    def test_cli_rejects_tracked_product_dirtiness_before_publishing(self) -> None:
        # Given: a committed product source file has been edited.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = self._repository(root)
            (repository / "product.txt").write_text("dirty product\n", encoding="utf-8")
            payload = root / "payload"
            payload.mkdir()
            (payload / "9001.x86_64").write_bytes(b"player payload")
            migration = root / "migration.json"
            migration.write_text('{"migration":"verified"}\n', encoding="utf-8")
            stage = root / "stage"

            # When: packaging is requested from the dirty source tree.
            result = self._package((repository, payload, stage, migration))

            # Then: it fails without publishing an archive or receipt.
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("tracked product source differs from HEAD", result.stderr)
            self.assertFalse((stage / ARCHIVE_NAME).exists())
            self.assertFalse((stage / "archive.sha256").exists())
            self.assertFalse(stage.exists())

    def test_git_revision_rejects_untracked_product_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = self._repository(Path(temporary))
            (repository / "scripts" / "wrapper.sh").write_text("untracked", encoding="utf-8")

            with self.assertRaisesRegex(PackagingError, "untracked product source"):
                git_revision(repository)

    def test_create_archive_preserves_existing_final_archive_when_gzip_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "payload"
            payload.mkdir()
            (payload / "player").write_bytes(b"player")
            archive = root / "player.tar.gz"
            archive.write_bytes(b"published")

            with patch(
                "scripts.package_physics_player.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "gzip"),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    create_archive(payload, archive)

            self.assertEqual(archive.read_bytes(), b"published")

    def test_archive_replacement_failure_preserves_published_pair(self) -> None:
        # Given: a valid published archive/receipt pair and a replacement payload.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository, payload, stage, migration = self._fixture(root)
            player = payload / "9001.x86_64"
            player.write_bytes(b"published player")
            stage.mkdir()
            archive = stage / ARCHIVE_NAME
            receipt = stage / "archive.sha256"
            create_archive(payload, archive)
            receipt.write_text(f"{_sha256(archive)}  {ARCHIVE_NAME}\n", encoding="ascii")
            published_archive = archive.read_bytes()
            published_receipt = receipt.read_bytes()
            player.write_bytes(b"replacement player")
            real_replace = os.replace

            def interrupt_archive(source: Path, destination: Path) -> None:
                if Path(destination) == archive:
                    raise OSError("archive replacement interrupted")
                real_replace(source, destination)

            # When: publication is interrupted immediately before replacing the archive.
            with patch("os.replace", side_effect=interrupt_archive), self.assertRaisesRegex(OSError, "archive replacement interrupted"):
                self._run_in_process((repository, payload, stage, migration))

            # Then: the prior commit pair remains valid and no temporary sibling survives.
            self.assertEqual(archive.read_bytes(), published_archive)
            self.assertEqual(receipt.read_bytes(), published_receipt)
            with tarfile.open(archive, "r:gz") as bundle:
                self.assertIn("9001.x86_64", bundle.getnames())
            self.assertEqual({path.name for path in stage.iterdir()}, {ARCHIVE_NAME, "archive.sha256"})

    def test_receipt_replacement_failure_leaves_valid_uncommitted_archive(self) -> None:
        # Given: a valid published pair and a complete replacement payload.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository, payload, stage, migration = self._fixture(root)
            player = payload / "9001.x86_64"
            player.write_bytes(b"previous player")
            stage.mkdir()
            archive = stage / ARCHIVE_NAME
            create_archive(payload, archive)
            (stage / "archive.sha256").write_text(f"{_sha256(archive)}  {ARCHIVE_NAME}\n", encoding="ascii")
            player.write_bytes(b"complete player")
            real_replace = os.replace

            def interrupt_receipt(source: Path, destination: Path) -> None:
                if Path(destination).name == "archive.sha256":
                    raise OSError("receipt replacement interrupted")
                real_replace(source, destination)

            # When: archive replacement succeeds but receipt replacement fails.
            with patch("os.replace", side_effect=interrupt_receipt), self.assertRaisesRegex(OSError, "receipt replacement interrupted"):
                self._run_in_process((repository, payload, stage, migration))

            # Then: the archive validates, remains uncommitted, and has no temporary siblings.
            with tarfile.open(archive, "r:gz") as bundle:
                manifest_file = bundle.extractfile("provenance.json")
                self.assertIsNotNone(manifest_file)
                manifest = json.loads(manifest_file.read())
            self.assertEqual(manifest["files"]["9001.x86_64"], _sha256(player))
            self.assertFalse((stage / "archive.sha256").exists())
            self.assertEqual({path.name for path in stage.iterdir()}, {ARCHIVE_NAME})

    def test_successful_publication_replaces_archive_before_receipt(self) -> None:
        # Given: a package whose final replacements are observable.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository, payload, stage, migration = self._fixture(root)
            (payload / "9001.x86_64").write_bytes(b"player")
            replacements: list[str] = []
            real_replace = os.replace

            def record_replacement(source: Path, destination: Path) -> None:
                if Path(destination).parent == stage:
                    replacements.append(Path(destination).name)
                real_replace(source, destination)

            # When: publication succeeds.
            with patch("os.replace", side_effect=record_replacement):
                result = self._run_in_process((repository, payload, stage, migration))

            # Then: the archive is committed before its receipt marker.
            self.assertEqual(result, 0)
            self.assertEqual(replacements, [ARCHIVE_NAME, "archive.sha256"])
            self.assertEqual((stage / "archive.sha256").read_text(encoding="ascii").split()[0], _sha256(stage / ARCHIVE_NAME))

    def test_cli_rejects_untracked_unity_input_before_staging(self) -> None:
        # Given: an untracked source file that can affect the Unity build.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository, payload, stage, migration = self._fixture(root)
            untracked = repository / "tasks" / "task_template_designer" / "Assets" / "UntrackedBuildInput.cs"
            untracked.write_text("internal static class UntrackedBuildInput {}\n", encoding="utf-8")
            (payload / "9001.x86_64").write_bytes(b"player")

            # When: packaging is requested from the unproven source tree.
            result = self._package((repository, payload, stage, migration))

            # Then: provenance rejection happens before any stage artifact exists.
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("untracked product source", result.stderr)
            self.assertFalse(stage.exists())

    def test_manifest_binds_all_payload_sources_and_remains_deterministic(self) -> None:
        # Given: tracked repository inputs and explicit external build sources.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository, payload, stage, migration = self._fixture(root)
            sources = root / "sources"
            sources.mkdir()
            source_bytes = {"unity_executable": b"pinned Unity executable", "interface_jar": b"interface jar", "config_source": b"source config", "serverbackup_source": b"server backup"}
            source_paths = {role: sources / role for role in source_bytes}
            for role, path in source_paths.items():
                path.write_bytes(source_bytes[role])
            wrapper = repository / "scripts" / "9001-player-wrapper.sh"
            payload_bytes = {"9001-player.x86_64": b"Unity-built player", "9001.x86_64": wrapper.read_bytes(), "game_playing_interface.jar": source_bytes["interface_jar"], "config.xml": b"transformed config", "serverbackup": source_bytes["serverbackup_source"]}
            for name, content in payload_bytes.items():
                (payload / name).write_bytes(content)
            source_arguments = ("--unity-executable", str(source_paths["unity_executable"]), "--interface-jar", str(source_paths["interface_jar"]), "--config-source", str(source_paths["config_source"]), "--serverbackup-source", str(source_paths["serverbackup_source"]))

            # When: identical inputs are packaged twice with complete source provenance.
            first = self._package((repository, payload, stage, migration), source_arguments)
            self.assertEqual(first.returncode, 0, first.stderr)
            first_archive = (stage / ARCHIVE_NAME).read_bytes()
            first_manifest = (payload / "provenance.json").read_bytes()
            second = self._package((repository, payload, stage, migration), source_arguments)

            # Then: every output and source is digest-bound without destabilizing bytes.
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual((stage / ARCHIVE_NAME).read_bytes(), first_archive)
            self.assertEqual((payload / "provenance.json").read_bytes(), first_manifest)
            manifest = json.loads(first_manifest)
            self.assertEqual(manifest["files"], {name: hashlib.sha256(content).hexdigest() for name, content in payload_bytes.items()})
            build_inputs = manifest["build_inputs"]
            for role, path in source_paths.items():
                self.assertEqual(build_inputs[role]["sha256"], _sha256(path))
            self.assertEqual(build_inputs["player_wrapper"]["repository_path"], "scripts/9001-player-wrapper.sh")
            self.assertEqual(build_inputs["player_wrapper"]["sha256"], manifest["files"]["9001.x86_64"])


if __name__ == "__main__":
    unittest.main()
