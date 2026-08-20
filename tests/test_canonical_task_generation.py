import os
import random
import tempfile
import unittest
from pathlib import Path

from scripts.scenario_manifest import BenchmarkCondition, canonical_xml_projection, load_manifest
from tasks.task_generator.canonical_materialization import (
    CanonicalMaterializationRequest,
    materialize_level_instance,
)


TEMPLATE = '''<?xml version="1.0" encoding="utf-8"?>
<Level width="2" custom="preserved">
  <Camera maxWidth="30" minWidth="20"><Unknown value="kept" /></Camera>
  <Score highScore="0" />
  <Birds><Bird type="BirdRed" /></Birds>
  <Slingshot x="-8" y="-2" />
  <GameObjects>
    <Pig type="BasicSmall" material="" x="1" y="-3" rotation="0" customPig="kept" />
    <Block type="SquareSmall" material="wood" x="2" y="-3" rotation="0" customBlock="kept" />
  </GameObjects>
</Level>
'''


class CanonicalTaskGenerationTests(unittest.TestCase):
    def test_materialization_is_replayable_and_independent_of_process_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template_path = root / "template.xml"
            template_path.write_text(TEMPLATE, encoding="utf-8")

            def request(name: str) -> CanonicalMaterializationRequest:
                return CanonicalMaterializationRequest(
                    template_path=template_path,
                    output_xml_path=root / name / "level.xml",
                    output_manifest_path=root / name / "level.scenario.json",
                    template_name="0_1_0101_1_5",
                    benchmark_condition=BenchmarkCondition("novelty_level_1", "type0101"),
                    template_identity="scenario-template-v1:test-template",
                    generation_seed=1729,
                    reference_point=(1.0, -3.0),
                    min_coordinate=(-1.0, -3.0),
                    max_coordinate=(1.0, -1.0),
                    restricted_objects=(),
                )

            random.seed(9001)
            state_before = random.getstate()
            first = materialize_level_instance(request("first"))
            self.assertEqual(random.getstate(), state_before)

            random.seed(7)
            materialize_level_instance(request("discarded"), publish=False)
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                second = materialize_level_instance(request("second"))
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(first.xml_content, second.xml_content)
            self.assertEqual(first.parameter_realization, second.parameter_realization)
            self.assertEqual(
                first.parameter_realization,
                {
                    "schema": "scenario_parameter_realization_v1",
                    "declared_initial_engine_state": canonical_xml_projection(first.xml_content),
                },
            )
            self.assertEqual(first.manifest.generation.parameter_realization, first.parameter_realization)
            generated_objects = first.parameter_realization["declared_initial_engine_state"]["root"]["children"][-1]["children"]
            self.assertGreater(len(generated_objects), 2)
            self.assertTrue(any(node["attributes"].get("material") in {"ice", "wood", "stone"} for node in generated_objects))
            root_projection = first.parameter_realization["declared_initial_engine_state"]["root"]
            birds = next(node for node in root_projection["children"] if node["tag"] == "Birds")["children"]
            slingshot = next(node for node in root_projection["children"] if node["tag"] == "Slingshot")
            authored_ids = [node["attributes"]["scenarioObjectId"] for node in birds]
            authored_ids.append(slingshot["attributes"]["scenarioObjectId"])
            authored_ids.extend(node["attributes"]["scenarioObjectId"] for node in generated_objects)
            self.assertEqual(len(authored_ids), len(set(authored_ids)))
            self.assertEqual(authored_ids[0], "bird:0000")
            self.assertEqual(slingshot["attributes"]["scenarioObjectId"], "slingshot:0000")
            self.assertIn("pig:0000", authored_ids)
            self.assertIn("block:0000", authored_ids)
            self.assertEqual(first.manifest, second.manifest)
            self.assertIn(b'custom="preserved"', first.xml_content)
            self.assertIn(b'customPig="kept"', first.xml_content)
            self.assertIn(b'customBlock="kept"', first.xml_content)
            self.assertEqual(load_manifest(request("first").output_manifest_path, request("first").output_xml_path), first.manifest)

    def test_changed_generation_seed_gets_a_distinct_specification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template_path = root / "template.xml"
            template_path.write_text(TEMPLATE, encoding="utf-8")
            common = dict(
                template_path=template_path,
                template_name="0_1_0101_1_5",
                benchmark_condition=BenchmarkCondition("novelty_level_1", "type0101"),
                template_identity="scenario-template-v1:test-template",
                reference_point=(1.0, -3.0),
                min_coordinate=(-1.0, -3.0),
                max_coordinate=(1.0, -1.0),
                restricted_objects=(),
            )
            first = materialize_level_instance(CanonicalMaterializationRequest(
                output_xml_path=root / "first.xml",
                output_manifest_path=root / "first.scenario.json",
                generation_seed=1,
                **common,
            ), publish=False)
            second = materialize_level_instance(CanonicalMaterializationRequest(
                output_xml_path=root / "second.xml",
                output_manifest_path=root / "second.scenario.json",
                generation_seed=2,
                **common,
            ), publish=False)

            self.assertNotEqual(first.manifest.scenario_specification.identity, second.manifest.scenario_specification.identity)
            self.assertNotEqual(first.manifest.scenario_lineage.identity, second.manifest.scenario_lineage.identity)


if __name__ == "__main__":
    unittest.main()
