from __future__ import annotations

from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from scripts.package_physics_player import PackagingError, git_revision


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "build_physics_player.sh"
REDACT_SCRIPT = ROOT / "scripts" / "redact_unity_log.sh"

LICENSED_LOG = """Initialize engine version: 2019.4.41f2 (6b23d448b533)
[Licensing::Module] Successfully connected to LicensingClient on channel: LicenseClient-example
Entitlement-based licensing initiated
[LicensingClient] Licenses Updated successfully in LicensingClient
[Licensing::Module] Serial number assigned to: "F4-HCSV-G8FX-6VYN-NB2J-XXXX"\\nPro License: NO
LICENSE SYSTEM [202686 15:54:7] Next license update check is after 2026-08-07T05:54:07
Build completed with a result of 'Succeeded'
"""


class PhysicsPlayerBuildScriptTests(unittest.TestCase):
    @staticmethod
    def _resolve_stage(*arguments: str, environment: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("bash", str(BUILD_SCRIPT), *arguments, "--print-stage"),
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_default_stage_and_existing_override_remain_physics_v1_compatible(self) -> None:
        # Given: the build entrypoint is invoked without the additive v2 selector.
        default = self._resolve_stage()
        overridden_environment = os.environ.copy()
        overridden_environment["NOVPHY_PHYSICS_STAGE"] = "/tmp/existing-physics-stage"

        # When: the existing stage override is also resolved.
        overridden = self._resolve_stage(environment=overridden_environment)

        # Then: the default remains physics-v1 and the existing override keeps its meaning.
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertEqual(Path(default.stdout.strip()), ROOT / "sciencebirdsgames" / "physics-v1")
        self.assertEqual(overridden.returncode, 0, overridden.stderr)
        self.assertEqual(overridden.stdout.strip(), "/tmp/existing-physics-stage")

    def test_explicit_v2_stage_resolves_only_to_physics_v2(self) -> None:
        # Given: an unrelated legacy stage override is present.
        environment = os.environ.copy()
        environment["NOVPHY_PHYSICS_STAGE"] = "/tmp/must-not-receive-v2"

        # When: the additive v2 build is explicitly selected.
        resolved = self._resolve_stage("--physics-v2", environment=environment)

        # Then: v2 has one repository-owned destination and cannot target the override or v1.
        self.assertEqual(resolved.returncode, 0, resolved.stderr)
        self.assertEqual(Path(resolved.stdout.strip()), ROOT / "sciencebirdsgames" / "physics-v2")
        self.assertNotIn("physics-v1", resolved.stdout)
        self.assertNotIn("must-not-receive-v2", resolved.stdout)
        source = BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('capture_schema="physics_capture_v2_engine_v1"', source)
        self.assertEqual(source.count('--capture-schema "$capture_schema"'), 2)

    def test_source_preflight_precedes_stage_creation_and_unity(self) -> None:
        # Given: the exact player build entrypoint.
        source = BUILD_SCRIPT.read_text(encoding="utf-8")

        # When: its build-affecting operations are ordered.
        preflight = source.find("--check-worktree-only")
        snapshot = source.find("--write-package-inputs")
        stage_creation = source.find('mkdir -p "$stage"')
        unity = source.find('"$editor" -batchmode')
        final_package = source.rfind("--package-inputs")

        # Then: dirty or untracked source fails before build or stage writes.
        self.assertGreaterEqual(preflight, 0)
        self.assertGreaterEqual(snapshot, 0)
        self.assertGreaterEqual(final_package, 0)
        self.assertLess(preflight, stage_creation)
        self.assertLess(snapshot, unity)
        self.assertLess(preflight, unity)
        self.assertGreater(final_package, unity)

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


    def test_unity_logs_outside_the_stage_and_is_redacted_before_packaging(self) -> None:
        # Given: the exact player build entrypoint.
        source = BUILD_SCRIPT.read_text(encoding="utf-8")

        # When: the Unity log destination and redaction step are ordered.
        unity = source.find('"$editor" -batchmode')
        redaction = source.find("scripts/redact_unity_log.sh")
        final_package = source.rfind("--package-inputs")

        # Then: Unity never writes its raw log into the promotion stage,
        # and the published copy is redacted before the archive is packaged.
        self.assertNotIn('-logFile "$stage/', source)
        self.assertIn('-logFile "$unity_log"', source)
        self.assertGreater(redaction, unity)
        self.assertLess(redaction, final_package)

    def test_failed_unity_build_still_publishes_a_redacted_log_then_fails(self) -> None:
        # Given: the exact player build entrypoint.
        source = BUILD_SCRIPT.read_text(encoding="utf-8")

        # When: the Unity invocation captures its own exit status.
        redaction = source.find("scripts/redact_unity_log.sh")
        propagation = source.find('exit "$unity_exit"')

        # Then: a failure is diagnosable and still fails closed.
        self.assertIn("|| unity_exit=$?", source)
        self.assertGreater(propagation, redaction)

    def test_redaction_removes_licensing_identity_and_preserves_structure(self) -> None:
        # Given: a Unity log carrying a channel, serial, and LICENSE SYSTEM line.
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source_log = workspace / "unity-build.log"
            destination_log = workspace / "published.log"
            source_log.write_text(LICENSED_LOG, encoding="utf-8")

            # When: the log is redacted for publication.
            subprocess.run(("bash", str(REDACT_SCRIPT), str(source_log), str(destination_log)), check=True)
            published = destination_log.read_text(encoding="utf-8")

            # Then: no licensing identity survives, and build output is preserved.
            self.assertNotIn("LicenseClient-example", published)
            self.assertNotIn("F4-HCSV-G8FX-6VYN-NB2J", published)
            self.assertNotIn("[Licensing::Module]", published)
            self.assertNotIn("[LicensingClient]", published)
            self.assertNotIn("LICENSE SYSTEM", published)
            self.assertIn("Initialize engine version: 2019.4.41f2 (6b23d448b533)", published)
            self.assertIn("Build completed with a result of 'Succeeded'", published)
            self.assertEqual(len(published.splitlines()), len(LICENSED_LOG.splitlines()))
            self.assertEqual(published.count("[redacted:"), 4)

    def test_redaction_is_deterministic_and_leaves_no_temporary_residue(self) -> None:
        # Given: one licensed Unity log redacted twice.
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source_log = workspace / "unity-build.log"
            source_log.write_text(LICENSED_LOG, encoding="utf-8")
            first = workspace / "first.log"
            second = workspace / "second.log"

            # When: the same input is redacted into two destinations.
            subprocess.run(("bash", str(REDACT_SCRIPT), str(source_log), str(first)), check=True)
            subprocess.run(("bash", str(REDACT_SCRIPT), str(source_log), str(second)), check=True)

            # Then: output is byte-identical and no partial file is left behind.
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(sorted(p.name for p in workspace.iterdir()), ["first.log", "second.log", "unity-build.log"])

    def test_redaction_rejects_a_missing_source_log(self) -> None:
        # Given: no Unity log at the requested path.
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)

            # When: redaction is attempted.
            completed = subprocess.run(
                ("bash", str(REDACT_SCRIPT), str(workspace / "absent.log"), str(workspace / "published.log")),
                capture_output=True,
            )

            # Then: it fails closed without creating a published log.
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse((workspace / "published.log").exists())


if __name__ == "__main__":
    unittest.main()
