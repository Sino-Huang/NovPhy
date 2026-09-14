import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import issue_76_angular_accounting_correction as correction


class AccountingCorrectionTests(unittest.TestCase):
    def test_atomic_directory_rename_keeps_complete_byte_count(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            temporary = root / ".observation-trace.fixture"
            temporary.mkdir()
            (temporary / "frame").write_bytes(b"RGB")
            original = os.scandir

            def rename_before_scan(path):
                if Path(path) == temporary and temporary.exists():
                    temporary.rename(root / "observation-trace")
                return original(path)

            with patch("os.scandir", side_effect=rename_before_scan):
                self.assertEqual(correction.artifact_bytes(root), 3)

    def test_unrelated_failure_is_not_retried(self):
        with patch.object(correction, "original_counter", side_effect=PermissionError("denied")) as counter:
            with self.assertRaises(PermissionError):
                correction.artifact_bytes(Path("root"))
            self.assertEqual(counter.call_count, 1)

    def test_second_rename_failure_propagates(self):
        error = FileNotFoundError(2, "renamed", "/root/.observation-trace.fixture/agent")
        with patch.object(correction, "original_counter", side_effect=error) as counter:
            with self.assertRaises(FileNotFoundError):
                correction.artifact_bytes(Path("root"))
            self.assertEqual(counter.call_count, 2)
