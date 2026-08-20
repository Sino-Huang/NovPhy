from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from scripts.render_issue_45_review import build_review_board


ROOT = Path(__file__).parents[1]
EVIDENCE = ROOT / ".claude/project-docs/evidence/issue-45-cohort-v2-lineage"
DRAFT_ID = "central-v2-scenario-inventory-draft-v1:sha256:993d27c535c100e73209d2d0da33169cb313a5b72d902ee23ad6170bdc481400"


class Issue45ReviewBoardTests(unittest.TestCase):
    def test_renders_public_roles_reset_result_and_opaque_final_to_png(self) -> None:
        with TemporaryDirectory() as temporary:
            result = build_review_board(EVIDENCE, Path(temporary))

            html = Path(result["html_path"]).read_text(encoding="utf-8")
            self.assertIn(DRAFT_ID, html)
            self.assertIn(f"APPROVE {DRAFT_ID}", html)
            self.assertIn("training", html)
            self.assertIn("calibration", html)
            self.assertIn("model_selection", html)
            self.assertIn("final_evaluation", html)
            self.assertIn("Sealed — intentionally not rendered", html)
            self.assertIn("Unity reset reproduction", html)
            with Image.open(result["png_path"]) as image:
                self.assertEqual(image.size, (1600, 1200))


if __name__ == "__main__":
    unittest.main()
