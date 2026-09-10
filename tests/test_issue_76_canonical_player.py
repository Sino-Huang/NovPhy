import json
from pathlib import Path
import tempfile
import unittest

from scripts import run_issue_76_canonical_player as player


class CanonicalPlayerPreparationTests(unittest.TestCase):
    def source(self, root):
        recovered = root / "recovered"
        for name in player.REQUIRED:
            path = recovered / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("reference asset\n")
        (recovered / "ProjectSettings/ProjectVersion.txt").write_text("2019.3.4f1\n")
        (recovered / "Packages").mkdir()
        (recovered / "Packages/manifest.json").write_text('{"dependencies": {}}')
        return recovered

    def test_dry_plan_does_not_create_work_or_claim_equivalence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recovered = self.source(root)
            work = root / "work"
            result = player.plan(recovered, work)
            self.assertFalse(work.exists())
            self.assertFalse(result["fresh_access"])
            self.assertFalse(result["canonical_equivalence_established"])
            self.assertEqual(result["graphics_captures"], 0)

    def test_missing_novel_prefab_blocks_preparation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recovered = self.source(root)
            (recovered / player.REQUIRED[3]).unlink()
            with self.assertRaisesRegex(ValueError, "PinkBigPig"):
                player.prepare(recovered, root / "work")
            self.assertFalse((root / "work").exists())

    def test_prepare_preserves_reference_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recovered = self.source(root)
            work = root / "work"
            player.prepare(recovered, work)
            self.assertEqual((recovered / "ProjectSettings/ProjectVersion.txt").read_text(), "2019.3.4f1\n")
            self.assertEqual((work / "project/ProjectSettings/ProjectVersion.txt").read_text(), player.VERSION)
            self.assertTrue((work / "project/Assets/Editor/CanonicalPigAssetTests.cs").exists())
            self.assertEqual(json.loads((work / "preparation.json").read_text())["source_player"], "sciencebirdsgames/Linux")
            with self.assertRaisesRegex(ValueError, "already exists"):
                player.prepare(recovered, work)

    def test_case_colliding_bundle_exports_are_retained_outside_build(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recovered = self.source(root)
            bundle = recovered / "Assets/resources/novelty"
            bundle.mkdir(parents=True)
            (bundle / "extra.prefab").write_text("bundle export")
            work = root / "work"
            player.prepare(recovered, work)
            self.assertFalse((work / "project/Assets/resources").exists())
            self.assertTrue((work / "project/Assets/Resources").exists())
            self.assertEqual((work / "recovered-bundle-exports/novelty/extra.prefab").read_text(), "bundle export")
            self.assertTrue((bundle / "extra.prefab").exists())


if __name__ == "__main__":
    unittest.main()
