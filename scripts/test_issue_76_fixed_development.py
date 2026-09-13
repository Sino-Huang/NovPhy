"""Run synthetic evaluator regressions without consulting real experiment files."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import run_issue_76_fixed_development as run

MODULES = (
    "tests.test_issue_76_fixed_development_backend", "tests.test_issue_76_fixed_development_json",
    "tests.test_issue_76_fixed_development_validation", "tests.test_issue_76_fixed_development_execution",
    "tests.test_issue_76_fixed_development_scoring", "tests.test_fixed_development",
    "tests.test_native_history_fit", "tests.test_issue_76_hybrid_drift",
)


def main():
    # The legacy execution fixture mocks file reads but not the experiment root.
    # Keep existence checks in the same isolated domain as those mocked reads.
    with TemporaryDirectory() as directory, patch.object(run, "ROOT", Path(directory)):
        result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(MODULES))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
