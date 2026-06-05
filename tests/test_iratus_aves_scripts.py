import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


class IratusAvesScriptsTest(unittest.TestCase):
    def test_generate_script_copies_generator_output_into_engine_levels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module_dir = root / "modules" / "IratusAves"
            module_dir.mkdir(parents=True)
            generator = module_dir / "generator_competition.py"
            generator.write_text(
                "from pathlib import Path\n"
                "assert Path('parameters.txt').is_file()\n"
                "Path('level-04.xml').write_text('<Level />', encoding='utf-8')\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "sciencebirdsagents" / "Utils" / "GenerateIratusAvesLevels.py"),
                    "--root",
                    str(root),
                    "--levels",
                    "1",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            output = root / "sciencebirdsgames" / "Linux" / "9001_Data" / "StreamingAssets" / "Levels" / "iratus_aves" / "Levels" / "level-04.xml"
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.is_file())
            self.assertIn("Generated 1 IratusAves level", result.stdout)

    def test_generate_script_normalizes_iratus_utf16_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module_dir = root / "modules" / "IratusAves"
            module_dir.mkdir(parents=True)
            generator = module_dir / "generator_competition.py"
            generator.write_text(
                "from pathlib import Path\n"
                "Path('level-04.xml').write_text('<?xml version=\"1.0\" encoding=\"utf-16\"?>\\n<Level />', encoding='utf-8')\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "sciencebirdsagents" / "Utils" / "GenerateIratusAvesLevels.py"),
                    "--root",
                    str(root),
                    "--levels",
                    "1",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            output = root / "sciencebirdsgames" / "Linux" / "9001_Data" / "StreamingAssets" / "Levels" / "iratus_aves" / "Levels" / "level-04.xml"
            parsed = ET.parse(output).getroot()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(parsed.tag, "Level")
            self.assertIn("encoding=\"utf-8\"", output.read_text(encoding="utf-8"))

    def test_generate_script_normalizes_unclosed_slingshot_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module_dir = root / "modules" / "IratusAves"
            module_dir.mkdir(parents=True)
            generator = module_dir / "generator_competition.py"
            generator.write_text(
                "from pathlib import Path\n"
                "Path('level-04.xml').write_text('<?xml version=\"1.0\" encoding=\"utf-16\"?>\\n<Level>\\n<Slingshot x=\"-8\" y=\"-2.5\">\\n<GameObjects>\\n</GameObjects>\\n</Level>', encoding='utf-8')\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "sciencebirdsagents" / "Utils" / "GenerateIratusAvesLevels.py"),
                    "--root",
                    str(root),
                    "--levels",
                    "1",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            output = root / "sciencebirdsgames" / "Linux" / "9001_Data" / "StreamingAssets" / "Levels" / "iratus_aves" / "Levels" / "level-04.xml"
            parsed = ET.parse(output).getroot()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(parsed.tag, "Level")
            self.assertIn('<Slingshot x="-8" y="-2.5" />', output.read_text(encoding="utf-8"))

    def test_generate_script_normalizes_unclosed_camera_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module_dir = root / "modules" / "IratusAves"
            module_dir.mkdir(parents=True)
            generator = module_dir / "generator_competition.py"
            generator.write_text(
                "from pathlib import Path\n"
                "Path('level-04.xml').write_text('<?xml version=\"1.0\" encoding=\"utf-16\"?>\\n<Level>\\n<Camera x=\"0\" y=\"2\" minWidth=\"20\" maxWidth=\"30\">\\n<Birds>\\n</Birds>\\n<Slingshot x=\"-8\" y=\"-2.5\">\\n<GameObjects>\\n</GameObjects>\\n</Level>', encoding='utf-8')\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "sciencebirdsagents" / "Utils" / "GenerateIratusAvesLevels.py"),
                    "--root",
                    str(root),
                    "--levels",
                    "1",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            output = root / "sciencebirdsgames" / "Linux" / "9001_Data" / "StreamingAssets" / "Levels" / "iratus_aves" / "Levels" / "level-04.xml"
            parsed = ET.parse(output).getroot()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(parsed.tag, "Level")
            self.assertIn('<Camera x="0" y="2" minWidth="20" maxWidth="30" />', output.read_text(encoding="utf-8"))

    def test_generate_script_adds_required_platform_numeric_attributes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module_dir = root / "modules" / "IratusAves"
            module_dir.mkdir(parents=True)
            generator = module_dir / "generator_competition.py"
            generator.write_text(
                "from pathlib import Path\n"
                "Path('level-04.xml').write_text('<?xml version=\"1.0\" encoding=\"utf-16\"?>\\n<Level>\\n<Camera x=\"0\" y=\"2\" minWidth=\"20\" maxWidth=\"30\">\\n<Birds>\\n</Birds>\\n<Slingshot x=\"-8\" y=\"-2.5\">\\n<GameObjects>\\n<Platform type=\"Platform\" material=\"\" x=\"1\" y=\"2\" />\\n<Platform type=\"Platform\" material=\"\" x=\"3\" y=\"4\" rotation=\"10\" scaleX=\"2\" />\\n</GameObjects>\\n</Level>', encoding='utf-8')\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "sciencebirdsagents" / "Utils" / "GenerateIratusAvesLevels.py"),
                    "--root",
                    str(root),
                    "--levels",
                    "1",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            output = root / "sciencebirdsgames" / "Linux" / "9001_Data" / "StreamingAssets" / "Levels" / "iratus_aves" / "Levels" / "level-04.xml"
            platforms = ET.parse(output).getroot().findall(".//Platform")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(platforms[0].attrib["rotation"], "0.0")
            self.assertEqual(platforms[0].attrib["scaleX"], "1.0")
            self.assertEqual(platforms[0].attrib["scaleY"], "1.0")
            self.assertEqual(platforms[1].attrib["rotation"], "10")
            self.assertEqual(platforms[1].attrib["scaleX"], "2")
            self.assertEqual(platforms[1].attrib["scaleY"], "1.0")

    def test_generate_script_adds_score_node_required_by_engine_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module_dir = root / "modules" / "IratusAves"
            module_dir.mkdir(parents=True)
            generator = module_dir / "generator_competition.py"
            generator.write_text(
                "from pathlib import Path\n"
                "Path('level-04.xml').write_text('<?xml version=\"1.0\" encoding=\"utf-16\"?>\\n<Level>\\n<Camera x=\"0\" y=\"2\" minWidth=\"20\" maxWidth=\"30\">\\n<Birds>\\n</Birds>\\n<Slingshot x=\"-8\" y=\"-2.5\">\\n<GameObjects>\\n</GameObjects>\\n</Level>', encoding='utf-8')\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "sciencebirdsagents" / "Utils" / "GenerateIratusAvesLevels.py"),
                    "--root",
                    str(root),
                    "--levels",
                    "1",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            output = root / "sciencebirdsgames" / "Linux" / "9001_Data" / "StreamingAssets" / "Levels" / "iratus_aves" / "Levels" / "level-04.xml"
            score = ET.parse(output).getroot().find("Score")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIsNotNone(score)
            self.assertEqual(score.attrib["highScore"], "0")

    def test_load_script_writes_config_for_generated_levels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            levels_dir = root / "sciencebirdsgames" / "Linux" / "9001_Data" / "StreamingAssets" / "Levels" / "iratus_aves" / "Levels"
            levels_dir.mkdir(parents=True)
            (levels_dir / "level-04.xml").write_text("<Level />", encoding="utf-8")
            (levels_dir / "level-05.xml").write_text("<Level />", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "sciencebirdsagents" / "Utils" / "PrepareGeneratedLevelsConfig.py"),
                    "--root",
                    str(root),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            config_path = root / "sciencebirdsgames" / "Linux" / "config.xml"
            level_paths = [node.attrib["level_path"] for node in ET.parse(config_path).getroot().findall(".//game_levels")]
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                level_paths,
                [
                    "9001_Data/StreamingAssets/Levels/iratus_aves/Levels/level-04.xml",
                    "9001_Data/StreamingAssets/Levels/iratus_aves/Levels/level-05.xml",
                ],
            )
            self.assertIn("Wrote", result.stdout)

    def test_load_script_rejects_empty_generated_level_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            levels_dir = root / "sciencebirdsgames" / "Linux" / "9001_Data" / "StreamingAssets" / "Levels" / "iratus_aves" / "Levels"
            levels_dir.mkdir(parents=True)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "sciencebirdsagents" / "Utils" / "PrepareGeneratedLevelsConfig.py"),
                    "--root",
                    str(root),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No generated level XML files found", result.stderr)


if __name__ == "__main__":
    unittest.main()
