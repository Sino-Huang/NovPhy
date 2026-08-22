from __future__ import annotations

import copy
import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.build_issue_46_evidence import (
    Issue46EvidenceError,
    build_issue_46_evidence,
    validate_issue_46_evidence,
)
from tests.test_observation_trace import engine_capture, source_bindings


def probe(
    identity: str,
    *,
    configuration: str,
    exposure_role: str,
    template: str,
    level: str,
    lineage: str,
    sequence: int,
) -> dict:
    capture = engine_capture(sequence=sequence)
    capture["capture_id"] = f"capture-runtime-{sequence}"
    capture["render_frame"] = 10 + sequence
    capture["fixed_step"] = 20 + sequence
    capture["source_frame_identity"] = (
        f"source-frame-v1:capture-runtime-{sequence}:{sequence}:"
        f"{10 + sequence}:{20 + sequence}"
    )
    bindings = source_bindings()
    bindings.update({
        "scenario_template_identity": template,
        "level_instance_identity": level,
        "source_scenario_lineage_identity": lineage,
        "rollout_identity": f"rollout:{identity}",
    })
    return {
        "probe_identity": identity,
        "evidence_source": "unity_runtime_non_fixture",
        "source_snapshot_commit": "a" * 40,
        "player_archive_identity": f"player-archive:{identity}",
        "scenario_manifest_identity": f"scenario-manifest:{identity}",
        "observation_configuration": configuration,
        "exposure_role": exposure_role,
        "source_bindings": bindings,
        "captures": [capture],
    }


class Issue46EvidenceTests(unittest.TestCase):
    def test_published_runtime_bundle_is_bound_to_the_current_player_archive(self):
        repository = Path(__file__).parents[1]
        root = repository / "data" / "runtime_evidence" / "issue-46"
        archive = (
            repository / "sciencebirdsgames" / "observation-v1"
            / "novphy-physics-player-2019.4.41f2.tar.gz"
        )

        bundle = validate_issue_46_evidence(root)
        archive_identity = (
            "physics-v2-player-archive-v1:sha256:"
            + hashlib.sha256(archive.read_bytes()).hexdigest()
        )
        with tarfile.open(archive, "r:gz") as packaged:
            stream = packaged.extractfile("provenance.json")
            self.assertIsNotNone(stream)
            provenance = json.load(stream)
        self.assertEqual(
            {entry["player_archive_identity"] for entry in bundle["probes"]},
            {archive_identity},
        )
        self.assertEqual(
            {entry["source_snapshot_commit"] for entry in bundle["probes"]},
            {provenance["project"]["git_head"]},
        )

    def test_bundle_requires_two_real_nonfinal_lineages_templates_levels_and_transforms(self):
        probes = [
            probe(
                "training-native",
                configuration="agent_rgb8_native_v1",
                exposure_role="training",
                template="template:training",
                level="level:training",
                lineage="lineage:training",
                sequence=1,
            ),
            probe(
                "calibration-resized",
                configuration="agent_rgb8_nearest_2x2_v1",
                exposure_role="calibration",
                template="template:calibration",
                level="level:calibration",
                lineage="lineage:calibration",
                sequence=2,
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "issue-46"
            bundle = build_issue_46_evidence(root, probes)

            self.assertEqual(validate_issue_46_evidence(root), bundle)
            self.assertEqual(bundle["coverage"]["non_final_scenario_lineage_count"], 2)
            self.assertEqual(bundle["coverage"]["scenario_template_count"], 2)
            self.assertEqual(bundle["coverage"]["level_instance_count"], 2)
            self.assertEqual(bundle["coverage"]["observation_configuration_count"], 2)
            self.assertTrue(bundle["immutable_non_fixture_observation_capability"])
            self.assertTrue(bundle["access_behavior"]["authorized_canonical_diagnostic"])
            self.assertTrue(bundle["access_behavior"]["rejected_canonical_training"])
            self.assertTrue(bundle["access_behavior"]["rejected_canonical_model_selection"])

    def test_fixture_or_incomplete_coverage_cannot_be_promoted(self):
        first = probe(
            "training-native",
            configuration="agent_rgb8_native_v1",
            exposure_role="training",
            template="template:training",
            level="level:training",
            lineage="lineage:training",
            sequence=1,
        )
        second = probe(
            "calibration-resized",
            configuration="agent_rgb8_nearest_2x2_v1",
            exposure_role="calibration",
            template="template:calibration",
            level="level:calibration",
            lineage="lineage:calibration",
            sequence=2,
        )
        cases = (
            ([{**first, "evidence_source": "fixture"}, second], "non-fixture"),
            ([first], "two"),
            ([first, copy.deepcopy(first)], "distinct"),
        )
        for index, (probes, message) in enumerate(cases):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(Issue46EvidenceError, message):
                    build_issue_46_evidence(
                        Path(temporary) / f"issue-46-{index}", probes
                    )


if __name__ == "__main__":
    unittest.main()
