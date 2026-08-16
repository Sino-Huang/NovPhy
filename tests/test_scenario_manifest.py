from dataclasses import asdict
from hashlib import sha256
import json
import tempfile
import unittest
from pathlib import Path

from scripts.scenario_manifest import (
    BenchmarkCondition,
    SMOKE_ONLY,
    create_generated_manifest,
    import_legacy_manifest,
    load_manifest,
    require_research_eligible,
    write_manifest,
)


XML = b'''<?xml version="1.0" encoding="utf-8"?>
<Level width="2">
  <Camera maxWidth="30" minWidth="20"><Unknown value="kept" /></Camera>
  <Birds><Bird type="BirdRed" /></Birds>
  <Slingshot x="-8" y="-2" />
  <GameObjects><Pig type="BasicSmall" x="1" y="-3" rotation="0" /></GameObjects>
</Level>
'''


def contract_identity(namespace: str, value: object) -> str:
    canonical = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{namespace}:sha256:{sha256(canonical).hexdigest()}"


class ScenarioManifestTests(unittest.TestCase):
    def test_generated_manifest_has_deterministic_hierarchy_and_validates_exact_content(self) -> None:
        arguments = {
            "xml_content": XML,
            "benchmark_condition": BenchmarkCondition("novelty_level_1", "type0101"),
            "template_identity": "scenario-template-v1:fixture",
            "generator_identity": "novphy-task-generator",
            "generator_version": "canonical-v1",
            "generation_seed": 41,
            "declared_inputs": {"bounds": [[-1, 1], [-2, 2]], "layout_choice": 0},
            "parameter_realization": {"shift_x": 0.25, "shift_y": -0.5},
        }
        first = create_generated_manifest(**arguments)
        second = create_generated_manifest(**arguments)

        self.assertEqual(first, second)
        self.assertEqual(first.generation.mode, "generated")
        self.assertEqual(first.scenario_specification.content_identity, second.scenario_specification.content_identity)
        self.assertEqual(first.scenario_lineage.identity, second.scenario_lineage.identity)
        expected_level_instance = contract_identity(
            "level-instance-v1",
            {
                "benchmark_condition_identity": first.benchmark_condition.identity,
                "scenario_template": asdict(first.scenario_template),
                "declaration_identity": first.scenario_specification.declaration_identity,
            },
        )
        self.assertEqual(first.level_instance.identity, expected_level_instance)
        self.assertEqual(
            first.scenario_specification.identity,
            contract_identity(
                "scenario-specification-v1",
                {
                    "level_instance_identity": expected_level_instance,
                    "content_identity": first.scenario_specification.content_identity,
                },
            ),
        )
        self.assertEqual(
            first.declared_initial_engine_state.identity,
            second.declared_initial_engine_state.identity,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "fixture.xml"
            manifest_path = Path(temp_dir) / "fixture.scenario.json"
            xml_path.write_bytes(XML)
            write_manifest(first, manifest_path)
            self.assertEqual(load_manifest(manifest_path, xml_path), first)

            xml_path.write_bytes(XML.replace(b'width="2"', b'width="3"'))
            with self.assertRaisesRegex(ValueError, "content identity"):
                load_manifest(manifest_path, xml_path)

    def test_each_changed_declared_input_gets_a_distinct_specification_and_lineage(self) -> None:
        base = {
            "xml_content": XML,
            "benchmark_condition": BenchmarkCondition("novelty_level_1", "type0101"),
            "template_identity": "scenario-template-v1:fixture",
            "generator_identity": "novphy-task-generator",
            "generator_version": "canonical-v1",
            "generation_seed": 41,
            "declared_inputs": {"layout_choice": 0},
            "parameter_realization": {"shift_x": 0.25},
        }
        original = create_generated_manifest(**base)

        for key, changed_value in (
            ("benchmark_condition", BenchmarkCondition("novelty_level_2", "type0101")),
            ("template_identity", "scenario-template-v1:other"),
            ("generator_version", "canonical-v2"),
            ("generation_seed", 42),
            ("declared_inputs", {"layout_choice": 1}),
        ):
            changed = create_generated_manifest(**(base | {key: changed_value}))
            self.assertNotEqual(original.scenario_specification.identity, changed.scenario_specification.identity)
            self.assertNotEqual(original.scenario_lineage.identity, changed.scenario_lineage.identity)

    def test_malformed_identity_graph_fails_closed(self) -> None:
        manifest = create_generated_manifest(
            XML,
            benchmark_condition=BenchmarkCondition("novelty_level_1", "type0101"),
            template_identity="scenario-template-v1:fixture",
            generator_identity="novphy-task-generator",
            generator_version="canonical-v1",
            generation_seed=41,
            declared_inputs={},
            parameter_realization={},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fixture.scenario.json"
            payload = manifest.to_dict()
            payload["scenario_lineage"]["identity"] = "scenario-lineage-v1:sha256:stale"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lineage identity"):
                load_manifest(path)

            del payload["scenario_template"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_manifest(path)

    def test_legacy_import_is_explicit_and_smoke_only_is_never_research_eligible(self) -> None:
        manifest = import_legacy_manifest(
            XML,
            benchmark_condition=BenchmarkCondition("novelty_level_0", "type2"),
            source_path="novelty_level_0/type2/Levels/3_9_6_1.xml",
            eligibility=SMOKE_ONLY,
            eligibility_reason="staged type2 runtime fixture",
        )

        self.assertEqual(manifest.generation.mode, "legacy_static")
        self.assertIsNone(manifest.generation.generator_identity)
        self.assertIsNone(manifest.generation.generator_version)
        self.assertIsNone(manifest.generation.generation_seed)
        self.assertEqual(manifest.scenario_template.availability, "unavailable")
        self.assertIsNone(manifest.scenario_template.identity)
        for use in ("training", "calibration", "model selection", "final evaluation"):
            with self.assertRaisesRegex(ValueError, "smoke_only"):
                require_research_eligible(manifest, use)

    def test_staged_type2_sidecar_is_content_validated_and_smoke_only(self) -> None:
        levels = Path(__file__).resolve().parents[1] / "tasks/task_template_designer/Assets/StreamingAssets/Levels/novelty_level_0/type2/Levels"
        manifest = load_manifest(levels / "3_9_6_1.scenario.json", levels / "3_9_6_1.xml")

        self.assertEqual(manifest.generation.mode, "legacy_static")
        self.assertEqual(manifest.research_eligibility.status, SMOKE_ONLY)
        with self.assertRaisesRegex(ValueError, "smoke_only"):
            require_research_eligible(manifest)


if __name__ == "__main__":
    unittest.main()
