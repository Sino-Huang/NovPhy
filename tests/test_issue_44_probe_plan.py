from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree as ET

from scripts.build_issue_44_probe_plan import (
    SUPPORT_MANIFEST,
    SUPPORT_TEMPLATE_REFERENCE,
    SUPPORT_XML,
    TRAINING_TEMPLATE_REFERENCE,
    build_issue_44_probe_plan,
)
from scripts.cohort_v2_scenarios import load_cohort_v2_scenario_manifest
from scripts.collection_plan import load_collection_plan
from src.webui.physics_v2_review import REVIEW_GOAL_LEVELS


ROOT = Path(__file__).resolve().parents[1]


class Issue44ProbePlanTests(unittest.TestCase):
    def test_builds_the_five_frozen_nonfinal_probe_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "probe-plan.json"
            result = build_issue_44_probe_plan(output)
            loaded = load_collection_plan(output)
            self.assertEqual(loaded.plan.plan_version, 2)
            scenarios = {scenario.scenario_id: scenario for scenario in loaded.plan.scenarios}
            self.assertEqual(set(scenarios), {
                "type010101-training-seed4401",
                "issue44-support-ready-type010101-seed4401",
            })
            by_case = {
                intervention.id: intervention
                for scenario in loaded.plan.scenarios
                for intervention in scenario.interventions
            }
            self.assertEqual(set(by_case), {
                "no-contact", "collision", "support", "support-change", "stable-terminal",
            })
            self.assertEqual(by_case["collision"].interface_action["drag_release"], (-77, 29))
            self.assertEqual(by_case["support-change"].interface_action["drag_release"], (-77, 29))
            self.assertEqual(by_case["no-contact"].interface_action["drag_release"], (77, 29))
            self.assertEqual(by_case["support"].interface_action["drag_release"], (77, 29))
            self.assertEqual(by_case["stable-terminal"].interface_action["drag_release"], (-74, -31))
            for intervention in by_case.values():
                self.assertEqual(intervention.interface_action["releaseTime"], 1000)
                self.assertEqual(intervention.interface_action["tapTime"], 0)
            self.assertEqual(
                [item.id for item in scenarios["type010101-training-seed4401"].interventions],
                ["no-contact", "collision", "stable-terminal"],
            )
            self.assertEqual(
                [item.id for item in scenarios[
                    "issue44-support-ready-type010101-seed4401"
                ].interventions],
                ["support", "support-change"],
            )
            self.assertEqual(result["plan_identity"], loaded.plan.identity)
            self.assertEqual(result["probe_cases"], sorted(by_case))
            first = output.read_bytes()
            self.assertEqual(build_issue_44_probe_plan(output)["plan_identity"], loaded.plan.identity)
            self.assertEqual(output.read_bytes(), first)
            self.assertEqual(json.loads(first)["identity"], loaded.plan.identity)

    def test_support_ready_authority_moves_only_the_platform_seam_and_uses_seed_4401(self) -> None:
        def parsed(reference: str):
            content = (ROOT / reference).read_bytes().replace(
                b'encoding="utf-16"', b'encoding="utf-8"', 1,
            )
            return ET.fromstring(content)

        original = parsed(TRAINING_TEMPLATE_REFERENCE)
        support_template = parsed(SUPPORT_TEMPLATE_REFERENCE)
        self.assertEqual(original.find("GameObjects/Platform").attrib["y"], "-2.67")
        self.assertEqual(support_template.find("GameObjects/Platform").attrib["y"], "-2.64")
        original_without_platform_y = ET.tostring(original).replace(b'y="-2.67"', b'y="-2.64"')
        self.assertEqual(original_without_platform_y, ET.tostring(support_template))

        result = build_issue_44_probe_plan()
        support = load_cohort_v2_scenario_manifest(
            SUPPORT_MANIFEST,
            xml_path=SUPPORT_XML,
            template_source_path=ROOT / SUPPORT_TEMPLATE_REFERENCE,
        )
        generated = ET.parse(SUPPORT_XML).getroot()
        self.assertEqual(generated.find("GameObjects/Pig").attrib["y"], "-1.79")
        self.assertEqual(generated.find("GameObjects/Platform").attrib["y"], "-2.3026")
        self.assertEqual(support.scenario_manifest.generation.generation_seed, 4401)
        self.assertEqual(support.template_record.source_reference, SUPPORT_TEMPLATE_REFERENCE)
        self.assertEqual(
            result["support_ready_level_identity"],
            support.scenario_manifest.level_instance.identity,
        )

    def test_builder_does_not_rewrite_closed_issue_45_review_inputs(self) -> None:
        paths = [
            ROOT / "sciencebirdsgames/physics-v2/review-levels/calibration.xml",
            ROOT / "sciencebirdsgames/physics-v2/review-levels/model-selection.xml",
            ROOT / "sciencebirdsgames/physics-v2/review-manifests/calibration.json",
            ROOT / "sciencebirdsgames/physics-v2/review-manifests/model-selection.json",
        ]
        before = {path: path.read_bytes() for path in paths}

        build_issue_44_probe_plan()

        self.assertEqual(before, {path: path.read_bytes() for path in paths})

    def test_review_goals_map_collision_to_level_one_and_support_to_level_two(self) -> None:
        self.assertEqual(REVIEW_GOAL_LEVELS, {
            "collision": 1,
            "persistent support": 2,
            "support change": 2,
        })


if __name__ == "__main__":
    unittest.main()
