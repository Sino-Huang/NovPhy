import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

from sciencebirdsagents.Utils.PrepareTestConfig import (
    discover_level_paths,
    ensure_java_interface_assets,
    write_config,
)


class PrepareTestConfigTest(unittest.TestCase):
    def test_discovers_novphy_level_paths_without_benchmark_type_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine_dir = Path(tmp) / "sciencebirdsgames" / "Linux"
            level_dir = engine_dir / "9001_Data" / "StreamingAssets" / "Levels"
            novphy_dir = level_dir / "novelty_level_0" / "type010101" / "Levels"
            benchmark_dir = level_dir / "novelty_level_1" / "type1" / "Levels"
            novphy_dir.mkdir(parents=True)
            benchmark_dir.mkdir(parents=True)

            novphy_level = novphy_dir / "00001_0_1_010101_0_1.xml"
            benchmark_level = benchmark_dir / "1_01_01_00001.xml"
            novphy_level.write_text("<Level />", encoding="utf-8")
            benchmark_level.write_text("<Level />", encoding="utf-8")

            paths = discover_level_paths(
                engine_dir,
                novelty_level="novelty_level_0",
                level_type="type010101",
            )

            self.assertEqual(
                paths,
                [
                    "9001_Data/StreamingAssets/Levels/novelty_level_0/type010101/Levels/00001_0_1_010101_0_1.xml"
                ],
            )
            self.assertNotIn("novelty_level_1/type1/Levels", "\n".join(paths))
            self.assertTrue(novphy_level.exists())
            self.assertTrue(benchmark_level.exists())

    def test_write_config_uses_existing_level_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.xml"
            level_paths = [
                "9001_Data/StreamingAssets/Levels/novelty_level_0/type010101/Levels/00001_0_1_010101_0_1.xml"
            ]

            write_config(config_path, level_paths)

            root = ET.parse(config_path).getroot()
            levels = [node.attrib["level_path"] for node in root.findall(".//game_levels")]

            self.assertEqual(root.tag, "evaluation")
            self.assertEqual(levels, level_paths)
            self.assertEqual(
                root.find("novelty_detection_measurement").attrib,
                {
                    "step": "1",
                    "measure_in_training": "False",
                    "measure_in_testing": "False",
                },
            )
            trial = root.find("trials/trial")
            self.assertEqual(trial.attrib["number_of_executions"], "1")
            self.assertEqual(trial.attrib["checkpoint_time_limit"], "9999999")
            self.assertEqual(trial.attrib["checkpoint_interaction_limit"], "9999999")
            level_set = trial.find("game_level_set")
            self.assertEqual(level_set.attrib["mode"], "training")
            self.assertNotIn("/type1/", levels[0])

    def test_ensure_java_interface_assets_uses_root_runtime_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine_dir = root / "sciencebirdsgames" / "Linux"
            (engine_dir / "DB").mkdir(parents=True)
            (engine_dir / "game_playing_interface.jar").write_bytes(b"jar")

            ensure_java_interface_assets(root, "Linux")

            self.assertFalse((root / "modules").exists())

    def test_ensure_java_interface_assets_reports_root_missing_jar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine_dir = root / "sciencebirdsgames" / "Linux"
            (engine_dir / "DB").mkdir(parents=True)

            with self.assertRaisesRegex(FileNotFoundError, "sciencebirdsgames/Linux/game_playing_interface.jar"):
                ensure_java_interface_assets(root, "Linux")

    def test_ensure_java_interface_assets_reports_root_missing_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine_dir = root / "sciencebirdsgames" / "Linux"
            engine_dir.mkdir(parents=True)
            (engine_dir / "game_playing_interface.jar").write_bytes(b"jar")

            with self.assertRaisesRegex(FileNotFoundError, "sciencebirdsgames/Linux/DB"):
                ensure_java_interface_assets(root, "Linux")


if __name__ == "__main__":
    unittest.main()
