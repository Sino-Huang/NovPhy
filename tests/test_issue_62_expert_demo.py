from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET

from scripts.run_issue_62_expert_demo import stage_expert_demo_runtime
from world_model.data.successor_cohort import build_pilot_plan


class Issue62ExpertDemoTests(unittest.TestCase):
    def test_stages_one_exact_failed_lineage_without_mutating_its_source(self) -> None:
        plan = build_pilot_plan()
        slot = plan["lineages"][0]
        source = b"<Level><Birds><Bird type=\"BirdRed\" /></Birds></Level>"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            production = root / "production"
            attempt = (
                production / "attempts" / slot["slot_identity"] / "attempt-01"
            )
            attempt.mkdir(parents=True)
            (attempt / "scenario.xml").write_bytes(source)
            game = root / "game"

            def unpack(_stage, target):
                (
                    target
                    / "9001_Data/StreamingAssets/Levels/novelty_level_0/type2/Levels"
                ).mkdir(parents=True)

            with patch(
                "scripts.run_issue_62_expert_demo.archive_details",
                side_effect=unpack,
            ):
                context = stage_expert_demo_runtime(
                    plan,
                    production_runtime=production,
                    game=game,
                    lineage_number=1,
                )

            config = ET.parse(game / "config.xml").getroot()
            configured = config.findall(".//game_levels")
            staged = game / configured[0].get("level_path")
            staged_bytes = staged.read_bytes()
            source_bytes = (attempt / "scenario.xml").read_bytes()

        self.assertEqual(context["lineage"], 1)
        self.assertEqual(context["slot_identity"], slot["slot_identity"])
        self.assertEqual(len(configured), 1)
        self.assertEqual(staged_bytes, source)
        self.assertEqual(source_bytes, source)


if __name__ == "__main__":
    unittest.main()
