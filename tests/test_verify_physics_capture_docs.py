from __future__ import annotations
# noqa: SIZE_OK - isolated publication mutations share one owned CLI fixture.

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
VERIFIER = ROOT / "scripts" / "verify_physics_capture_docs.py"
ARCHIVE_NAME = "novphy-physics-player-2019.4.41f2.tar.gz"
FINAL_PUBLICATION = Path(
    ".omo/evidence/world-model-physics-instrumentation/final-published-runtime"
)
COMMAND_BLOCK = re.compile(
    r"```bash physics_capture_v1_(?P<name>collection|promotion|rollback)\n.*?\n```\n",
    re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class PublicationFixture:
    repository: Path
    docs: Path
    stage: Path
    publication: Path
    archive_sha256: str
    stale_sha256: str


class PhysicsCaptureDocumentationTests(unittest.TestCase):
    def _repository_with_publication(self, root: Path) -> PublicationFixture:
        repository = root / "repository"
        copied_docs = repository / "docs"
        shutil.copytree(DOCS, copied_docs)
        stage = repository / "sciencebirdsgames" / "physics-v1"
        stage.mkdir(parents=True)
        archive = stage / ARCHIVE_NAME
        archive.write_bytes(b"current isolated staged archive")
        archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
        stage_receipt = stage / "archive.sha256"
        stage_receipt.write_text(
            f"{archive_sha256}  {archive.name}\n", encoding="ascii"
        )
        publication = repository / FINAL_PUBLICATION
        accepted_shot = publication / "published-smoke-output" / "shot_001"
        accepted_shot.mkdir(parents=True)
        publication_receipt = publication / "publication-receipt.json"
        publication_receipt.write_text(
            json.dumps(
                {
                    "schemaVersion": "novphy_final_publication_v1",
                    "status": "published",
                    "stage": str(stage),
                    "archive": {
                        "path": str(archive),
                        "sha256": archive_sha256,
                        "size": archive.stat().st_size,
                    },
                    "receipt": {
                        "path": str(stage_receipt),
                        "sha256": hashlib.sha256(stage_receipt.read_bytes()).hexdigest(),
                        "size": stage_receipt.stat().st_size,
                        "committedLast": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        smoke_report = publication / "published-smoke.json"
        smoke_report.write_text(
            json.dumps(
                {
                    "status": "accepted",
                    "phase": "complete",
                    "accepted_shot": str(accepted_shot),
                    "protected_unchanged": True,
                    "provenance": {"archive_sha256": archive_sha256},
                }
            ),
            encoding="utf-8",
        )
        (publication / "done-claim.json").write_text(
            json.dumps(
                {
                    "schemaVersion": "novphy_final_published_runtime_done_claim_v1",
                    "status": "complete",
                    "source": {"trackedProductClean": True},
                    "publication": {
                        "stage": str(stage),
                        "archiveSha256": archive_sha256,
                        "targetSha256": archive_sha256,
                        "archiveHashExact": True,
                        "receiptNonempty": True,
                        "unityBuildLogNonempty": True,
                        "receipt": str(publication_receipt),
                    },
                    "runtime": {
                        "publishedStage": True,
                        "request38Compatibility": True,
                        "request62Compatibility": True,
                        "request62Decoded": True,
                        "request70Decoded": True,
                        "actionPerformed": True,
                        "accepted": True,
                        "protectedUnchanged": True,
                        "report": str(smoke_report),
                        "acceptedShot": str(accepted_shot),
                    },
                }
            ),
            encoding="utf-8",
        )
        stale_sha256 = hashlib.sha256(b"historical staged archive").hexdigest()
        historical_report = publication.parent / "task-8-smoke.json"
        historical_report.write_text(
            json.dumps(
                {
                    "status": "accepted",
                    "phase": "complete",
                    "accepted_shot": "historical_shot",
                    "protected_unchanged": True,
                    "provenance": {"archive_sha256": stale_sha256},
                }
            ),
            encoding="utf-8",
        )
        return PublicationFixture(
            repository,
            copied_docs,
            stage,
            publication,
            archive_sha256,
            stale_sha256,
        )

    def _run_fixture(self, fixture: PublicationFixture) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "python",
                str(VERIFIER),
                "--repository-root",
                str(fixture.repository),
                str(fixture.docs),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def _replace_json_field(
        self,
        path: Path,
        field_path: tuple[str, ...],
        replacement: str | bool,
    ) -> None:
        document = json.loads(path.read_text(encoding="utf-8"))
        target = document
        for field in field_path[:-1]:
            target = target[field]
        target[field_path[-1]] = replacement
        path.write_text(json.dumps(document), encoding="utf-8")

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
                'expected_sha="$(awk \'NF == 2 {print $1}\' "$stage/archive.sha256")"',
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
                    command_name = expected_error.split()[0]
                    mutated = COMMAND_BLOCK.sub(
                        lambda match: match.group(0).replace(removed_operation, "", 1)
                        if match.group("name") == command_name
                        else match.group(0),
                        original,
                    )
                    self.assertNotEqual(mutated, original)
                    contract.write_text(mutated, encoding="utf-8")

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

    def test_cli_accepts_current_publication_when_historical_task_8_report_is_stale(self) -> None:
        # Given: current final publication evidence and an older accepted task-8 digest.
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._repository_with_publication(Path(temporary))
            self.assertNotEqual(fixture.archive_sha256, fixture.stale_sha256)

            # When: the verifier resolves provenance from the final DoneClaim.
            result = self._run_fixture(fixture)

            # Then: stale historical evidence does not override current authority.
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_rejects_malformed_publication_json_documents(self) -> None:
        # Given: each publication authority document is malformed in isolation.
        for name in ("done-claim.json", "publication-receipt.json", "published-smoke.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = self._repository_with_publication(Path(temporary))
                (fixture.publication / name).write_text("{", encoding="utf-8")

                # When: the malformed document crosses the verifier boundary.
                result = self._run_fixture(fixture)

                # Then: malformed JSON cannot authorize the staged archive.
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid JSON", result.stderr)

    def test_cli_rejects_stale_or_unsuccessful_done_claims(self) -> None:
        # Given: each required DoneClaim contract field is invalid in isolation.
        mutations = (
            (("schemaVersion",), "obsolete_done_claim", "DoneClaim schemaVersion"),
            (("status",), "pending", "DoneClaim status"),
            (("source", "trackedProductClean"), False, "trackedProductClean"),
            (("publication", "archiveHashExact"), False, "archiveHashExact"),
            (("runtime", "accepted"), False, "runtime accepted"),
            (("publication", "archiveSha256"), "0" * 64, "DoneClaim archive SHA-256"),
        )
        for field_path, replacement, expected_error in mutations:
            with self.subTest(field_path=field_path), tempfile.TemporaryDirectory() as temporary:
                fixture = self._repository_with_publication(Path(temporary))
                claim = fixture.publication / "done-claim.json"
                self._replace_json_field(claim, field_path, replacement)

                # When: the verifier evaluates the changed final authority.
                result = self._run_fixture(fixture)

                # Then: stale or incomplete DoneClaims fail closed.
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_error, result.stderr)

    def test_cli_rejects_publication_evidence_path_escapes(self) -> None:
        # Given: each DoneClaim evidence pointer escapes final-published-runtime.
        fields = (
            ("publication", "receipt"),
            ("runtime", "report"),
            ("runtime", "acceptedShot"),
        )
        for field_path in fields:
            with self.subTest(field_path=field_path), tempfile.TemporaryDirectory() as temporary:
                fixture = self._repository_with_publication(Path(temporary))
                claim = fixture.publication / "done-claim.json"
                escaped = fixture.publication.parent / "escaped.json"
                escaped.write_text("{}", encoding="utf-8")
                self._replace_json_field(claim, field_path, str(escaped))

                # When: the verifier resolves the untrusted absolute pointer.
                result = self._run_fixture(fixture)

                # Then: no evidence outside current final publication is trusted.
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must stay within final publication evidence", result.stderr)

    def test_cli_rejects_stale_or_malformed_publication_receipts(self) -> None:
        # Given: each publication receipt contract field is invalid in isolation.
        mutations = (
            (("schemaVersion",), "obsolete_receipt", "publication receipt schemaVersion"),
            (("status",), "staged", "publication receipt status"),
            (("archive", "path"), "other.tar.gz", "published archive path"),
            (("archive", "sha256"), "0" * 64, "publication receipt archive SHA-256"),
            (("receipt", "committedLast"), False, "committedLast"),
        )
        for field_path, replacement, expected_error in mutations:
            with self.subTest(field_path=field_path), tempfile.TemporaryDirectory() as temporary:
                fixture = self._repository_with_publication(Path(temporary))
                receipt = fixture.publication / "publication-receipt.json"
                self._replace_json_field(receipt, field_path, replacement)

                # When: the verifier evaluates the changed publication receipt.
                result = self._run_fixture(fixture)

                # Then: stale or malformed publication metadata fails closed.
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_error, result.stderr)

    def test_cli_rejects_malformed_stage_receipts_and_changed_archive_bytes(self) -> None:
        # Given: each staged publication commit artifact is invalid in isolation.
        for mutation, expected_error in (
            ("receipt", "staged archive receipt"),
            ("archive", "staged archive SHA-256"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = self._repository_with_publication(Path(temporary))
                if mutation == "receipt":
                    (fixture.stage / "archive.sha256").write_text(
                        "not-a-digest\n", encoding="ascii"
                    )
                else:
                    (fixture.stage / ARCHIVE_NAME).write_bytes(b"changed after publication")

                # When: the verifier checks the stage receipt and archive bytes.
                result = self._run_fixture(fixture)

                # Then: neither malformed commit markers nor changed bytes are accepted.
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_error, result.stderr)

    def test_cli_rejects_stale_or_malformed_final_smoke_reports(self) -> None:
        # Given: each accepted final smoke contract field is invalid in isolation.
        mutations = (
            (("status",), "rejected", "final smoke status"),
            (("phase",), "running", "final smoke phase"),
            (("accepted_shot",), "", "accepted_shot"),
            (("protected_unchanged",), False, "protected_unchanged"),
            (("provenance", "archive_sha256"), "0" * 64, "final smoke archive SHA-256"),
        )
        for field_path, replacement, expected_error in mutations:
            with self.subTest(field_path=field_path), tempfile.TemporaryDirectory() as temporary:
                fixture = self._repository_with_publication(Path(temporary))
                report = fixture.publication / "published-smoke.json"
                self._replace_json_field(report, field_path, replacement)

                # When: the verifier evaluates the changed accepted report.
                result = self._run_fixture(fixture)

                # Then: stale or malformed smoke proof cannot authorize publication.
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_error, result.stderr)


if __name__ == "__main__":
    unittest.main()
