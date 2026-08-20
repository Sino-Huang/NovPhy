from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_issue_45_evidence import (
    APPROVAL_TARGET_DRAFT_IDENTITY,
    build_issue_45_evidence,
    validate_issue_45_evidence,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class Issue45EvidenceBundleTests(unittest.TestCase):
    def test_builder_is_idempotent_and_keeps_final_realization_out_of_public_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public_root = root / "public"
            sealed_root = root / "sealed"

            first = build_issue_45_evidence(
                repository_root=REPOSITORY_ROOT,
                public_root=public_root,
                sealed_root=sealed_root,
            )
            first_bytes = {
                path.relative_to(root): path.read_bytes()
                for path in sorted(root.rglob("*"))
                if path.is_file()
            }
            second = build_issue_45_evidence(
                repository_root=REPOSITORY_ROOT,
                public_root=public_root,
                sealed_root=sealed_root,
            )
            second_bytes = {
                path.relative_to(root): path.read_bytes()
                for path in sorted(root.rglob("*"))
                if path.is_file()
            }

            self.assertEqual(first, second)
            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(first["draft_identity"], APPROVAL_TARGET_DRAFT_IDENTITY)
            self.assertTrue(first["matches_approval_target"])
            self.assertEqual(
                validate_issue_45_evidence(
                    repository_root=REPOSITORY_ROOT,
                    public_root=public_root,
                    sealed_root=sealed_root,
                ),
                first,
            )

            public_bytes = b"\n".join(first_bytes[path] for path in first_bytes if path.parts[0] == "public")
            self.assertNotIn(b'"generation_seed": 4502', public_bytes)
            sealed_projection = (public_root / "inventory/final-evaluation.sealed-projection.json").read_bytes()
            self.assertNotIn(b'"parameter_realization"', sealed_projection)
            self.assertFalse((public_root / "manifests/final-evaluation.json").exists())
            self.assertTrue((sealed_root / "final-evaluation.xml").is_file())
            self.assertTrue((sealed_root / "final-evaluation.cohort-v2-scenario.json").is_file())
            self.assertTrue((sealed_root / "final-evaluation.parameter-realization.json").is_file())


if __name__ == "__main__":
    unittest.main()
