import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.cohort_release import _artifact, publish_cohort_release, verify_cohort_publication


EVIDENCE = Path(".claude/project-docs/evidence/representative-pilot-20260820")


class CohortReleaseTests(unittest.TestCase):
    def test_artifact_accepts_collector_cwd_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary) / "production"
            shot = root / "accepted" / "attempt" / "shot_001"
            shot.mkdir(parents=True)
            self.assertEqual(_artifact(root.resolve(), str(shot)), shot.resolve())

    def test_publish_binds_primary_evidence_quality_and_derivations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "production"
            shutil.copytree(EVIDENCE / "collection-v4", root)
            shutil.copy2(
                next((EVIDENCE / "production-plan").glob("production_parameter_plan_*_v1.json")),
                root / "production_parameter_plan.json",
            )
            report_path = root / "collection_plan_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for entry in report["attempt_ledger"]:
                if entry["artifact_path"]:
                    attempt_id = entry["attempt_id"]
                    entry["artifact_path"] = str(root / "accepted" / attempt_id / "shot_001")
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

            publication_path = publish_cohort_release(
                root,
                partition_manifest_path=EVIDENCE / "instance-held-out-partition-v1.json",
                scenario_manifest_paths={
                    "baseline-type010101-level00026": EVIDENCE / "baseline-type010101-level00026.scenario.json",
                    "baseline-type010101-level00001": EVIDENCE / "baseline-type010101-level00001.scenario.json",
                },
                release_version=1,
                code_revision="fixture-revision",
                available_capabilities={"physics_capture_v1": "1", "support_v1": "1"},
                unavailable_capabilities={"material_identity": "not exported"},
                prior_execution_paths={"pilot-failure": EVIDENCE / "collection-v1-failed" / "collection_plan_report.json"},
            )

            publication = verify_cohort_publication(publication_path)
            release = json.loads((root / publication["cohort_release"]["path"]).read_text())
            derivations = json.loads((root / publication["authoritative_derivations"]["path"]).read_text())
            quality = json.loads((root / release["quality_report"]["path"]).read_text())

            self.assertEqual(len(release["primary_rollouts"]), 4)
            self.assertEqual(quality["counts"], {"accepted": 4, "failed": 0, "quarantined": 0, "rejected": 0})
            self.assertEqual(quality["prior_executions"][0]["attempt_count"], 4)
            self.assertEqual(derivations["source_cohort_release_identity"], release["identity"])
            self.assertEqual(len(derivations["artifacts"]), 8)
            self.assertEqual(derivations["unavailable_capabilities"]["material_identity"], "not exported")


if __name__ == "__main__":
    unittest.main()
