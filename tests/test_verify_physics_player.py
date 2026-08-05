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


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PhysicsPlayerVerifierTests(unittest.TestCase):
    def _make_stage(self, root: Path) -> Path:
        payload = root / "payload"
        payload.mkdir()
        player = payload / "9001.x86_64"
        player.write_bytes(b"player")
        player.chmod(0o755)
        files = {"9001.x86_64": _sha256(player)}
        manifest = {
            "schema_version": "novphy_physics_player_stage_v1",
            "unity": {"version": "2019.4.41f2", "changeset": "6b23d448b533"},
            "capture": {"schema_version": "physics_capture_v1", "protocol_version": 1},
            "files": files,
        }
        (payload / "provenance.json").write_text(json.dumps(manifest), encoding="utf-8")
        archive = root / "novphy-physics-player.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(payload, arcname=".")
        stage = root / "stage"
        stage.mkdir()
        archive.rename(stage / archive.name)
        (stage / "archive.sha256").write_text(
            f"{_sha256(stage / archive.name)}  {archive.name}\n", encoding="ascii"
        )
        return stage

    def test_cli_verifies_archive_and_unpacked_file_checksums(self) -> None:
        # Given: a staged archive with pinned Unity and capture provenance.
        with tempfile.TemporaryDirectory() as temporary:
            stage = self._make_stage(Path(temporary))

            # When: the verifier checks the package without launching Unity.
            result = subprocess.run(
                [sys.executable, "scripts/verify_physics_player.py", "--stage", str(stage), "--skip-runtime"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

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
            result = subprocess.run(
                [sys.executable, "scripts/verify_physics_player.py", "--stage", str(stage), "--expect-sha", "deadbeef", "--skip-runtime"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            # Then: verification fails closed.
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("archive SHA-256 mismatch", result.stderr)

    def test_cli_rejects_tampered_payload_manifest(self) -> None:
        # Given: an archive whose declared player digest is false.
        with tempfile.TemporaryDirectory() as temporary:
            stage = self._make_stage(Path(temporary))
            archive = next(stage.glob("*.tar.gz"))
            unpacked = Path(temporary) / "tampered"
            unpacked.mkdir()
            with tarfile.open(archive, "r:gz") as bundle:
                bundle.extractall(unpacked)
            manifest_path = unpacked / "provenance.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["9001.x86_64"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with tarfile.open(archive, "w:gz") as bundle:
                for path in sorted(unpacked.iterdir()):
                    bundle.add(path, arcname=path.name)
            (stage / "archive.sha256").write_text(
                f"{_sha256(archive)}  {archive.name}\n", encoding="ascii"
            )

            # When: the verifier checks the rebuilt archive.
            result = subprocess.run(
                [sys.executable, "scripts/verify_physics_player.py", "--stage", str(stage), "--skip-runtime"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            # Then: the payload mismatch is rejected.
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("payload SHA-256 mismatch", result.stderr)

    def test_entrypoint_imports_bridge_from_worktree_before_pythonpath(self) -> None:
        # Given: PYTHONPATH contains a conflicting checkout without request 70.
        with tempfile.TemporaryDirectory() as temporary:
            shadow = Path(temporary) / "shadow"
            package = shadow / "src" / "webui"
            package.mkdir(parents=True)
            (shadow / "src" / "__init__.py").write_text("", encoding="utf-8")
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "bridge.py").write_text(
                "class ScienceBirdsBridge:\n    pass\n", encoding="utf-8"
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(shadow)
            probe = (
                "import json, runpy, sys, time\n"
                'verifier = runpy.run_path("verify_physics_player.py", '
                'run_name="verify_probe")\n'
                "try:\n"
                '    verifier["connect_with_deadline"](1, time.monotonic() - 1)\n'
                'except verifier["VerificationError"]:\n'
                "    pass\n"
                'bridge = sys.modules["src.webui.bridge"]\n'
                "print(json.dumps({\n"
                '    "file": bridge.__file__,\n'
                '    "request_70": hasattr(bridge.ScienceBirdsBridge, '
                '"get_physics_capture_v1"),\n'
                "}))\n"
            )

            # When: the verifier is loaded exactly as a scripts-directory entrypoint.
            result = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=ROOT / "scripts",
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )

            # Then: runtime imports are pinned to this worktree.
            self.assertEqual(result.returncode, 0, result.stderr)
            resolved = json.loads(result.stdout)
            self.assertEqual(Path(resolved["file"]), ROOT / "src" / "webui" / "bridge.py")
            self.assertTrue(resolved["request_70"])


if __name__ == "__main__":
    unittest.main()
