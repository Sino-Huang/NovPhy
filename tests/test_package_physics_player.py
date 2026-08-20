from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts.package_physics_player import (
    ARCHIVE_NAME,
    CAPTURE_SCHEMA,
    ManifestContext,
    PackagingError,
    create_archive,
    git_revision,
    payload_inventory,
    publish,
    unity_package_input_inventory,
    validate_archive,
    write_manifest,
)


class PackagePhysicsPlayerTests(unittest.TestCase):
    def repository(self, root: Path) -> Path:
        repository = root / "repository"
        (repository / "scripts").mkdir(parents=True)
        (repository / "tasks/task_template_designer/Packages").mkdir(parents=True)
        (repository / "scripts/9001-player-wrapper.sh").write_text("#!/bin/sh\nexec ./9001-player.x86_64\n", encoding="utf-8")
        (repository / "tasks/task_template_designer/Packages/manifest.json").write_text("{}\n", encoding="utf-8")
        (repository / "tasks/task_template_designer/Packages/packages-lock.json").write_text("{}\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
        subprocess.run(["git", "add", "."], cwd=repository, check=True)
        subprocess.run(
            ["git", "-c", "user.name=NovPhy", "-c", "user.email=novphy@example.invalid", "commit", "-qm", "fixture"],
            cwd=repository,
            check=True,
        )
        return repository

    def payload(self, root: Path) -> Path:
        payload = root / "payload"
        payload.mkdir()
        (payload / "9001.x86_64").write_bytes(b"wrapper")
        (payload / "9001-player.x86_64").write_bytes(b"player")
        (payload / "game_playing_interface.jar").write_bytes(b"jar")
        return payload

    def test_payload_inventory_excludes_provenance_and_records_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = self.payload(Path(temporary))
            (payload / "provenance.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                payload_inventory(payload),
                {"9001-player.x86_64": 6, "9001.x86_64": 7, "game_playing_interface.jar": 3},
            )

    def test_git_revision_accepts_clean_sources_and_rejects_dirty_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = self.repository(Path(temporary))
            inputs = unity_package_input_inventory(repository)
            head, tree = git_revision(repository, inputs, require_package_inputs=True)
            self.assertTrue(head)
            self.assertTrue(tree)
            (repository / "scripts/9001-player-wrapper.sh").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(PackagingError, "differs from HEAD"):
                git_revision(repository, inputs, require_package_inputs=True)

    def test_manifest_records_versions_paths_and_file_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = self.repository(root)
            payload = self.payload(root)
            context = ManifestContext(
                repository,
                None,
                unity_package_input_inventory(repository),
                CAPTURE_SCHEMA,
            )
            path = write_manifest(payload, context)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["unity"]["version"], "2019.4.41f2")
            self.assertEqual(manifest["capture"]["protocol_version"], 1)
            self.assertEqual(manifest["migration"], {"issue": "#44"})
            self.assertEqual(manifest["files"], payload_inventory(payload))

    def test_archive_is_deterministic_and_structurally_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = self.payload(root)
            first = root / "first.tar.gz"
            second = root / "second.tar.gz"
            create_archive(payload, first)
            create_archive(payload, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            validate_archive(first, payload_inventory(payload), False)
            with tarfile.open(first, "r:gz") as bundle:
                self.assertEqual(
                    sorted(member.name for member in bundle.getmembers() if member.isfile()),
                    sorted(payload_inventory(payload)),
                )

    def test_publish_replaces_the_archive_without_a_pin_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = self.payload(root)
            stage = root / "stage"
            stage.mkdir()
            publish(payload, stage)
            self.assertEqual({path.name for path in stage.iterdir()}, {ARCHIVE_NAME})
            first = (stage / ARCHIVE_NAME).read_bytes()
            (payload / "9001-player.x86_64").write_bytes(b"new-player")
            publish(payload, stage)
            self.assertNotEqual((stage / ARCHIVE_NAME).read_bytes(), first)

    def test_archive_creation_failure_preserves_an_existing_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = self.payload(root)
            archive = root / ARCHIVE_NAME
            archive.write_bytes(b"existing-archive")
            with patch("scripts.package_physics_player.write_archive", side_effect=OSError("creation failed")):
                with self.assertRaisesRegex(OSError, "creation failed"):
                    create_archive(payload, archive)
            self.assertEqual(archive.read_bytes(), b"existing-archive")

    def test_publication_failure_preserves_an_existing_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = self.payload(root)
            stage = root / "stage"
            stage.mkdir()
            archive = stage / ARCHIVE_NAME
            archive.write_bytes(b"existing-archive")
            with patch("scripts.package_physics_player.os.replace", side_effect=OSError("publication failed")):
                with self.assertRaisesRegex(OSError, "publication failed"):
                    publish(payload, stage)
            self.assertEqual(archive.read_bytes(), b"existing-archive")


if __name__ == "__main__":
    unittest.main()
