from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


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
        notes = repository / ".omo" / "notepads"
        notes.mkdir(parents=True)
        (notes / "notes.md").write_text("initial evidence\n", encoding="utf-8")
        subprocess.run(("git", "add", "product.txt", ".omo/notepads/notes.md"), cwd=repository, check=True)
        subprocess.run(("git", "commit", "-qm", "test fixture"), cwd=repository, check=True)
        return repository

    def _package(self, repository: Path, payload: Path, stage: Path, migration: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            (
                sys.executable,
                str(PACKAGER),
                "--payload",
                str(payload),
                "--stage",
                str(stage),
                "--worktree",
                str(repository),
                "--migration-provenance",
                str(migration),
            ),
            text=True,
            capture_output=True,
            check=False,
        )

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
            result = self._package(repository, payload, stage, migration)

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
            first = self._package(repository, payload, stage, migration)
            self.assertEqual(first.returncode, 0, first.stderr)
            first_manifest = (payload / "provenance.json").read_bytes()
            first_archive = (stage / ARCHIVE_NAME).read_bytes()
            (repository / ".omo" / "notepads" / "notes.md").write_text("changed evidence\n", encoding="utf-8")
            cache = repository / "tasks" / "task_template_designer" / "Library"
            cache.mkdir(parents=True)
            (cache / "cache.bin").write_bytes(b"generated cache")
            (stage / "unity-build.log").write_text("generated log\n", encoding="utf-8")

            # When: the same payload is packaged again from that exact commit.
            second = self._package(repository, payload, stage, migration)

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
            result = self._package(repository, payload, stage, migration)

            # Then: it fails without publishing an archive or receipt.
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("tracked product source differs from HEAD", result.stderr)
            self.assertFalse((stage / ARCHIVE_NAME).exists())
            self.assertFalse((stage / "archive.sha256").exists())


if __name__ == "__main__":
    unittest.main()
