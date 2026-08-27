from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.cohort_v2_release import _write_derivations
from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_json
from scripts.observation_trace import persist_observation_trace
from scripts.physics_capture_v2 import load_physics_capture_v2
from tests.test_observation_trace import engine_capture
from world_model.data import (
    CohortV2AlignedObservationReader,
    CohortV2ReleaseReader,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_RELEASE = ROOT / "data/runtime_evidence/issue-53-mixed-termination-v5"
PLAN_ROOT = ROOT / "data/runtime_evidence/issue-53-plan-v5"
CAPABILITIES = ROOT / "docs/data_contracts/cohort_v2_capabilities_v1.json"
ALIGNED_IDENTITY = "cohort-v2-aligned-observation-release-v1:issue-59"


class CohortV2AlignedObservationReaderTests(unittest.TestCase):
    def test_reader_exposes_an_agent_image_for_each_exact_central_frame(self) -> None:
        source_reader = CohortV2ReleaseReader(
            PUBLIC_RELEASE,
            capability_declaration_path=CAPABILITIES,
            production_plan_root=PLAN_ROOT,
            workflow_kind="training",
            influence="learned_parameters",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            partition_root = root / "public"
            partition_root.mkdir()
            records = []
            for rollout in source_reader.rollouts:
                relative = Path("rollouts/training") / rollout.attempt_id
                destination = partition_root / relative
                destination.mkdir(parents=True)
                source = PUBLIC_RELEASE / "primary-rollouts" / rollout.attempt_id
                shutil.copyfile(
                    source / "physics_capture_v2.json",
                    destination / "physics_capture_v2.json",
                )
                capture = load_physics_capture_v2(
                    destination / "physics_capture_v2.json"
                )
                captures = []
                for sequence, frame in enumerate(
                    capture.record["frame_records"], start=1
                ):
                    value = engine_capture(
                        sequence=sequence,
                        fixed_step=frame["fixed_step"],
                        source="synchronized_fixed_step_camera_render",
                    )
                    value["capture_id"] = capture.capture_id
                    value["source_frame_identity"] = (
                        f"source-frame-v1:{capture.capture_id}:{sequence}:10:"
                        f"{frame['fixed_step']}"
                    )
                    captures.append(value)
                observation = persist_observation_trace(
                    destination / "observation-trace",
                    captures,
                    observation_configuration="agent_rgb8_nearest_2x2_v1",
                    source_bindings={
                        "scenario_template_identity": capture.source_bindings[
                            "scenario_template_id"
                        ],
                        "level_instance_identity": capture.source_bindings[
                            "level_instance_id"
                        ],
                        "source_scenario_lineage_identity": capture.source_bindings[
                            "scenario_lineage_id"
                        ],
                        "rollout_identity": rollout.attempt_id,
                    },
                    exposure_role="training",
                )
                derivation_relative = Path("derivations/training") / rollout.attempt_id
                derivations = _write_derivations(
                    partition_root / derivation_relative,
                    capture,
                    source_reference=(relative / "physics_capture_v2.json").as_posix(),
                    release_identity=ALIGNED_IDENTITY,
                )
                records.append({
                    "attempt_id": rollout.attempt_id,
                    "exposure_role": "training",
                    "coverage_stratum": rollout.coverage_stratum,
                    "scenario_lineage_identity": rollout.scenario_lineage_identity,
                    "capture_id": capture.capture_id,
                    "terminal_reason": capture.record["terminal_evidence"]["reason"],
                    "frame_count": len(capture.record["frame_records"]),
                    "rollout_path": relative.as_posix(),
                    "observation_manifest_identity": observation["identity"],
                    "derivations": [
                        {
                            **item,
                            "path": (derivation_relative / item["path"]).as_posix(),
                        }
                        for item in derivations
                    ],
                })
            partition = {
                "schema": "cohort_v2_aligned_observation_partition_v1",
                "identity": f"{ALIGNED_IDENTITY}:public",
                "release_identity": ALIGNED_IDENTITY,
                "included_roles": ["training", "calibration", "model_selection"],
                "records": records,
                "role_counts": {
                    "training": 6,
                    "calibration": 6,
                    "model_selection": 6,
                },
                "passed": True,
            }
            write_immutable_cohort_v2_json(
                partition, partition_root / "manifest.json"
            )
            manifest = {
                "schema": "cohort_v2_aligned_observation_release_v1",
                "identity": ALIGNED_IDENTITY,
                "implementation_commit": "fixture",
                "player_provenance": {},
                "source_bindings": {},
                "access_audit": {},
                "partitions": {
                    "public": {
                        "path": "public",
                        "identity": f"{ALIGNED_IDENTITY}:public",
                    },
                    "sealed_final": {
                        "path": "sealed-final",
                        "identity": f"{ALIGNED_IDENTITY}:sealed-final",
                        "ordinary_workflow_access": False,
                    },
                },
                "role_counts": {
                    "training": 6,
                    "calibration": 6,
                    "model_selection": 6,
                    "final_evaluation": 6,
                },
                "rollout_count": 24,
                "frame_count": sum(item["frame_count"] for item in records),
                "passed": True,
            }
            write_immutable_cohort_v2_json(manifest, root / "manifest.json")

            reader = CohortV2AlignedObservationReader(
                root, source_reader=source_reader
            )
            rollout = reader.rollouts[0]
            first = reader.load_frame_observation(
                rollout, rollout.frame_records[0], observation_role="agent"
            )
            second = reader.load_frame_observation(
                rollout, rollout.frame_records[1], observation_role="agent"
            )

        self.assertEqual(len(reader.rollouts), 6)
        self.assertTrue(first.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(second.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(reader.release_identity, ALIGNED_IDENTITY)


if __name__ == "__main__":
    unittest.main()
