from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
VERIFIER = ROOT / "scripts" / "verify_physics_capture_docs.py"
COMMAND_BLOCK = re.compile(
    r"```bash physics_capture_v1_(?P<name>collection|promotion|rollback)\n.*?\n```\n",
    re.DOTALL,
)


class PhysicsCaptureDocumentationTests(unittest.TestCase):
    def test_cli_accepts_the_published_contract_and_schema_example(self) -> None:
        # Given: the documentation tree published with the capture contract.
        # When: the documentation verifier is run through its CLI boundary.
        result = subprocess.run(
            ["python", str(VERIFIER), str(DOCS)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        # Then: the schema example and documentation contract are accepted.
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("physics capture documentation verified", result.stdout)

    def test_cli_accepts_harmless_prose_rewrites(self) -> None:
        # Given: copied documentation with explanatory prose rewritten.
        with tempfile.TemporaryDirectory() as temporary:
            copied_docs = Path(temporary) / "docs"
            shutil.copytree(DOCS, copied_docs)
            contract = copied_docs / "data_contracts" / "physics_capture_v1.md"
            contract.write_text(
                contract.read_text(encoding="utf-8").replace(
                    "All physical facts originate in Unity.",
                    "Unity provides the physical facts.",
                ),
                encoding="utf-8",
            )

            # When: the CLI verifies structurally unchanged documentation.
            result = subprocess.run(
                ["python", str(VERIFIER), str(copied_docs)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            # Then: prose wording does not change the verifier outcome.
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_rejects_a_malformed_schema_example(self) -> None:
        # Given: copied documentation with a JSON Schema integer replaced by a string.
        with tempfile.TemporaryDirectory() as temporary:
            copied_docs = Path(temporary) / "docs"
            shutil.copytree(DOCS, copied_docs)
            contract = copied_docs / "data_contracts" / "physics_capture_v1.md"
            contract.write_text(
                contract.read_text(encoding="utf-8").replace(
                    '"observed":30', '"observed":"30"'
                ),
                encoding="utf-8",
            )

            # When: the CLI validates the changed embedded example.
            result = subprocess.run(
                ["python", str(VERIFIER), str(copied_docs)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            # Then: the frozen schema definition rejects the invalid field type.
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("schema example field observed has the wrong type", result.stderr)

    def test_cli_rejects_non_atomic_promotion_or_rollback_commands(self) -> None:
        # Given: each mandatory promotion/rollback operation is removed in isolation.
        removals = (
            (
                "expected_sha=c7f9fa4c98480c1c1c8e580cb00454beda4fed4bf28a4822d31c561997906992",
                "promotion command",
            ),
            ('mv -Tf "$selector/next" "$selector/current"', "promotion command"),
            ('test -L "$selector/previous"', "rollback command"),
            ('mv -Tf "$selector/rollback" "$selector/current"', "rollback command"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            copied_docs = Path(temporary) / "docs"
            shutil.copytree(DOCS, copied_docs)
            contract = copied_docs / "data_contracts" / "physics_capture_v1.md"
            original = contract.read_text(encoding="utf-8")
            for removed_operation, expected_error in removals:
                with self.subTest(removed_operation=removed_operation):
                    self.assertIn(removed_operation, original)
                    contract.write_text(
                        original.replace(removed_operation, "", 1), encoding="utf-8"
                    )

                    # When: the published command parser sees an incomplete selector operation.
                    result = subprocess.run(
                        ["python", str(VERIFIER), str(copied_docs)],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    contract.write_text(original, encoding="utf-8")

                    # Then: an operator cannot rely on the incomplete command block.
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(expected_error, result.stderr)

    def test_cli_rejects_invalid_named_command_block_syntax(self) -> None:
        # Given: each named executable block retains required tokens but has an unclosed quote.
        with tempfile.TemporaryDirectory() as temporary:
            copied_docs = Path(temporary) / "docs"
            shutil.copytree(DOCS, copied_docs)
            contract = copied_docs / "data_contracts" / "physics_capture_v1.md"
            original = contract.read_text(encoding="utf-8")
            for command_name in ("collection", "promotion", "rollback"):
                with self.subTest(command_name=command_name):
                    mutated = COMMAND_BLOCK.sub(
                        lambda match: match.group(0).replace("\n```\n", "\n\"\n```\n")
                        if match.group("name") == command_name
                        else match.group(0),
                        original,
                    )
                    self.assertNotEqual(mutated, original)
                    contract.write_text(mutated, encoding="utf-8")

                    # When: the CLI validates the syntactically invalid command block.
                    result = subprocess.run(
                        ["python", str(VERIFIER), str(copied_docs)],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    contract.write_text(original, encoding="utf-8")

                    # Then: it fails closed even though all required command tokens remain.
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(f"{command_name} command has invalid shell syntax", result.stderr)

    def test_cli_rejects_changed_schema_contract_identifiers(self) -> None:
        # Given: copied schemas with one machine-consumed contract value changed.
        mutations = (
            (("coordinates", "properties", "world_space", "const"), "other_space", "coordinate schema"),
            (("support_rule", "properties", "minimum_vertical_center_delta", "const"), 1, "support rule"),
            (("state_header", "allOf", 1, "properties", "event_taxonomy", "prefixItems", 0, "const"), "other_event", "event taxonomy"),
            (("capture_failure", "properties", "failure_code", "enum"), [], "failure codes"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            copied_docs = Path(temporary) / "docs"
            shutil.copytree(DOCS, copied_docs)
            schema_path = copied_docs / "data_contracts" / "physics_capture_v1.schema.json"
            original = json.loads(schema_path.read_text(encoding="utf-8"))
            for path, replacement, expected_error in mutations:
                with self.subTest(path=path):
                    mutated = json.loads(json.dumps(original))
                    target = mutated["$defs"]
                    for segment in path[:-1]:
                        target = target[segment]
                    target[path[-1]] = replacement
                    schema_path.write_text(json.dumps(mutated), encoding="utf-8")

                    # When: the CLI parses a changed schema contract identifier.
                    result = subprocess.run(
                        ["python", str(VERIFIER), str(copied_docs)],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    schema_path.write_text(json.dumps(original), encoding="utf-8")

                    # Then: the machine-readable contract is rejected.
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(expected_error, result.stderr)

    def test_cli_rejects_receipt_report_archive_disagreement(self) -> None:
        # Given: an isolated repository root with a stage receipt that disagrees with report provenance.
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "repository"
            copied_docs = repository / "docs"
            shutil.copytree(DOCS, copied_docs)
            stage = repository / "sciencebirdsgames" / "physics-v1"
            stage.mkdir(parents=True)
            archive = stage / "novphy-physics-player-2019.4.41f2.tar.gz"
            archive.write_bytes(b"different staged archive")
            (stage / "archive.sha256").write_text(
                f"c7f9fa4c98480c1c1c8e580cb00454beda4fed4bf28a4822d31c561997906992  {archive.name}\n",
                encoding="ascii",
            )
            report = repository / ".omo" / "evidence" / "world-model-physics-instrumentation" / "task-8-smoke.json"
            report.parent.mkdir(parents=True)
            report.write_text(
                '{"status":"accepted","accepted_shot":"shot_001","protected_unchanged":true,"provenance":{"archive_sha256":"c7f9fa4c98480c1c1c8e580cb00454beda4fed4bf28a4822d31c561997906992"}}',
                encoding="utf-8",
            )

            # When: the verifier resolves documentation against that isolated repository.
            result = subprocess.run(
                ["python", str(VERIFIER), "--repository-root", str(repository), str(copied_docs)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            # Then: it rejects the stage receipt/archive disagreement with report provenance.
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("staged archive", result.stderr)


if __name__ == "__main__":
    unittest.main()
