from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.package_physics_player import PackagingError, git_revision


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "build_physics_player.sh"


class PhysicsPlayerBuildScriptTests(unittest.TestCase):
    def test_source_preflight_precedes_stage_creation_and_unity(self) -> None:
        # Given: the exact player build entrypoint.
        source = BUILD_SCRIPT.read_text(encoding="utf-8")

        # When: its build-affecting operations are ordered.
        preflight = source.find("--check-worktree-only")
        stage_creation = source.find('mkdir -p "$stage"')
        unity = source.find('"$editor" -batchmode')

        # Then: dirty or untracked source fails before build or stage writes.
        self.assertGreaterEqual(preflight, 0)
        self.assertLess(preflight, stage_creation)
        self.assertLess(preflight, unity)

    def test_ignored_untracked_unity_asset_is_rejected(self) -> None:
        # Given: Git ignores an untracked asset that Unity would still import.
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            subprocess.run(("git", "init", "-q", str(repository)), check=True)
            subprocess.run(("git", "config", "user.email", "test@example.invalid"), cwd=repository, check=True)
            subprocess.run(("git", "config", "user.name", "Test User"), cwd=repository, check=True)
            wrapper = repository / "scripts" / "9001-player-wrapper.sh"
            wrapper.parent.mkdir()
            wrapper.write_text("tracked wrapper\n", encoding="utf-8")
            asset = repository / "tasks" / "task_template_designer" / "Assets" / "texture.png"
            asset.parent.mkdir(parents=True)
            asset.write_bytes(b"ignored Unity texture")
            packages = repository / "tasks" / "task_template_designer" / "Packages"
            packages.mkdir(parents=True)
            (packages / "manifest.json").write_bytes((ROOT / "tasks/task_template_designer/Packages/manifest.json").read_bytes())
            (packages / "packages-lock.json").write_bytes((ROOT / "tasks/task_template_designer/Packages/packages-lock.json").read_bytes())
            (repository / ".gitignore").write_text("*.png\n", encoding="ascii")
            subprocess.run(("git", "add", ".gitignore", "scripts/9001-player-wrapper.sh"), cwd=repository, check=True)
            subprocess.run(("git", "commit", "-qm", "fixture"), cwd=repository, check=True)

            # When/Then: ignored status cannot bypass source provenance.
            with self.assertRaisesRegex(PackagingError, "untracked product source"):
                git_revision(repository)


if __name__ == "__main__":
    unittest.main()
