from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from PIL import Image

from scripts.render_issue_45_review import build_review_board


ROOT = Path(__file__).parents[1]
EVIDENCE = ROOT / "data/runtime_evidence/issue-45"


class Issue45ReviewBoardTests(unittest.TestCase):
    def test_renders_public_roles_reset_result_and_opaque_final_to_png(self) -> None:
        draft_path = EVIDENCE / "inventory/draft.json"
        if not draft_path.is_file():
            self.skipTest("issue #45 runtime evidence is not materialized")
        draft_id = json.loads(draft_path.read_text(encoding="utf-8"))["identity"]
        with TemporaryDirectory() as temporary:
            result = build_review_board(EVIDENCE, Path(temporary))

            html = Path(result["html_path"]).read_text(encoding="utf-8")
            self.assertIn(draft_id, html)
            self.assertIn(f"APPROVE {draft_id}", html)
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
