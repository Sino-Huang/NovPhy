from collections.abc import Callable
from hashlib import sha256
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from scripts.cohort_release import (
    _artifact,
    _identity,
    ingest_cohort_publication,
    publish_cohort_release,
    verify_cohort_publication,
    write_cohort_ingestion_evidence,
)


EVIDENCE = Path(".claude/project-docs/evidence/representative-pilot-20260820")
RELEASE = Path(
    ".claude/project-docs/evidence/representative-cohort-release-20260820/production-v1/release"
)
PUBLICATION = (
    RELEASE
    / "cohort_publication_a6daf82d47f7001e8731068c68a91e83487fd2c26926b35ab2974bc75a93ecf8_v1.json"
)


def _rewrite_derivations(
    publication_path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> Path:
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    derivation_path = publication_path.parent.parent / publication[
        "authoritative_derivations"
    ]["path"]
    derivations = json.loads(derivation_path.read_text(encoding="utf-8"))
    mutate(derivations)
    derivations["identity"] = _identity(derivations)
    derivation_path.write_text(
        json.dumps(derivations, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    publication["authoritative_derivations"]["identity"] = derivations["identity"]
    publication["authoritative_derivations"]["sha256"] = sha256(
        derivation_path.read_bytes()
    ).hexdigest()
    publication["identity"] = _identity(publication)
    publication_path.write_text(
        json.dumps(publication, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return publication_path


class CohortReleaseTests(unittest.TestCase):
    def test_required_readers_smoke_ingest_the_immutable_release(self) -> None:
        evidence = ingest_cohort_publication(
            PUBLICATION,
            required_capabilities=(
                "physics_capture_v1",
                "physics_relational_supervision_v1",
                "macro.steady-state",
                "macro.structure-unstable",
            ),
        )

        self.assertEqual(
            evidence.publication_identity,
            "representative-cohort-publication-v1:sha256:a6daf82d47f7001e8731068c68a91e83487fd2c26926b35ab2974bc75a93ecf8",
        )
        self.assertEqual(
            evidence.cohort_release_identity,
            "representative-cohort-release-v1:sha256:40b997354a256f889ef7dd007888b5ad8d84b5266883f3611500086f22b62ed2",
        )
        self.assertEqual(evidence.cohort_release_version, 1)
        self.assertEqual(
            evidence.derivation_identity,
            "authoritative-cohort-derivations-v1:sha256:64cff842534beb40ece23ec11673903dddc9a08881e4646779daae973434e187",
        )
        self.assertEqual(evidence.derivation_version, 1)
        self.assertEqual(evidence.rollout_count, 4)
        self.assertEqual(
            evidence.readers,
            (
                "cohort_partition_manifest_v1",
                "scenario_manifest_v1",
                "physics_capture_v1",
                "physics_macro_labels_v1",
                "physics_relational_supervision_v1",
                "world_model_physics_supervision",
            ),
        )
        self.assertEqual(
            evidence.unavailable_capabilities["macro.collapsed"],
            "rejected from production by issue 40 adjudication",
        )

    def test_ingestion_preserves_timing_identities_terminal_observations_and_unavailable_labels(self) -> None:
        evidence = ingest_cohort_publication(PUBLICATION)

        self.assertEqual(
            (
                evidence.identity_aligned_frame_record_count,
                evidence.fixed_step_aligned_event_count,
                evidence.experiment_macro_label_record_count,
                evidence.experiment_relational_label_record_count,
                evidence.terminal_observation_count,
                evidence.macro_label_record_count,
                evidence.relational_label_record_count,
                evidence.unavailable_relational_label_count,
            ),
            (24, 36, 24, 24, 4, 24, 24, 608),
        )

    def test_ingestion_evidence_persists_exact_versions_capabilities_and_counts(self) -> None:
        evidence = ingest_cohort_publication(PUBLICATION)

        with tempfile.TemporaryDirectory() as temporary:
            path = write_cohort_ingestion_evidence(
                evidence, Path(temporary) / "downstream-ingestion-v2.json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            {
                "schema": payload["schema"],
                "publication_identity": payload["publication_identity"],
                "cohort_release": payload["cohort_release"],
                "authoritative_derivations": payload["authoritative_derivations"],
                "partition": payload["partition"],
                "available_capabilities": payload["available_capabilities"],
                "counts": payload["counts"],
                "label_derivations": payload["label_derivations"],
            },
            {
                "schema": "cohort_ingestion_evidence_v2",
                "publication_identity": "representative-cohort-publication-v1:sha256:a6daf82d47f7001e8731068c68a91e83487fd2c26926b35ab2974bc75a93ecf8",
                "cohort_release": {
                    "identity": "representative-cohort-release-v1:sha256:40b997354a256f889ef7dd007888b5ad8d84b5266883f3611500086f22b62ed2",
                    "version": 1,
                },
                "authoritative_derivations": {
                    "identity": "authoritative-cohort-derivations-v1:sha256:64cff842534beb40ece23ec11673903dddc9a08881e4646779daae973434e187",
                    "version": 1,
                },
                "partition": {
                    "identity": "cohort-partition-manifest-v1:sha256:d2ab1b531e27cb2028c3e51064dfb53648b11176334bd230417ffbd2ba11f111",
                    "version": 2,
                },
                "available_capabilities": {
                    "macro.steady-state": "physics_macro_labels_v1",
                    "macro.structure-unstable": "physics_macro_labels_v1",
                    "physics_capture_v1": "1",
                    "physics_relational_supervision_v1": "1",
                },
                "counts": {
                    "rollouts": 4,
                    "frame_records": 24,
                    "events": 36,
                    "identity_aligned_frame_records": 24,
                    "fixed_step_aligned_events": 36,
                    "terminal_observations": 4,
                    "macro_label_records": 24,
                    "relational_label_records": 24,
                    "unavailable_relational_labels": 608,
                    "experiment_macro_label_records": 24,
                    "experiment_relational_label_records": 24,
                },
                "label_derivations": {
                    "physics_macro_labels_v1": {
                        "derivation_spec_version": "macro_labels_derivation_v2",
                        "derivation_spec_digest": "cc78f62299d129df6967cf42ca104d6677395885bd09e4a3a6c8914b424ac447",
                    },
                    "physics_relational_supervision_v1": {
                        "derivation_spec_version": "relational_supervision_derivation_v1",
                        "derivation_spec_digest": "b46036c99f71948f40ef911896048dabe97c772b4df10fd36a627e6d7252b9fe",
                    },
                },
            },
        )

    def test_malformed_published_scenario_manifest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "production"
            shutil.copytree(RELEASE.parent, root)
            scenario_path = next((root / "release" / "scenario_manifests").glob("*.json"))
            scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
            del scenario["scenario_lineage"]
            scenario_path.write_text(json.dumps(scenario), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "scenario"):
                ingest_cohort_publication(root / "release" / PUBLICATION.name)

    def test_unknown_publication_field_and_empty_capability_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "production"
            shutil.copytree(RELEASE.parent, root)
            publication_path = root / "release" / PUBLICATION.name
            publication = json.loads(publication_path.read_text(encoding="utf-8"))
            publication["unexpected"] = True
            publication["identity"] = _identity(publication)
            publication_path.write_text(
                json.dumps(publication, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "publication.*unknown"):
                ingest_cohort_publication(publication_path)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "production"
            shutil.copytree(RELEASE.parent, root)
            publication_path = root / "release" / PUBLICATION.name

            def empty_capability(derivations: dict[str, Any]) -> None:
                derivations["available_capabilities"]["physics_capture_v1"] = ""

            _rewrite_derivations(publication_path, empty_capability)
            with self.assertRaisesRegex(ValueError, "capability declarations"):
                ingest_cohort_publication(
                    publication_path, required_capabilities=("physics_capture_v1",)
                )

    def test_missing_capability_and_cross_release_derivations_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "capabilities are unavailable"):
            ingest_cohort_publication(
                PUBLICATION, required_capabilities=("material_identity",)
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "production"
            shutil.copytree(RELEASE.parent, root)
            publication_path = root / "release" / PUBLICATION.name
            def bind_another_release(derivations: dict[str, Any]) -> None:
                derivations["source_cohort_release_identity"] = (
                    "representative-cohort-release-v1:sha256:"
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )

            _rewrite_derivations(publication_path, bind_another_release)

            with self.assertRaisesRegex(ValueError, "another cohort release"):
                ingest_cohort_publication(publication_path)

    def test_unknown_authoritative_sidecar_reference_field_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "production"
            shutil.copytree(RELEASE.parent, root)
            publication_path = root / "release" / PUBLICATION.name
            def add_unknown_field(derivations: dict[str, Any]) -> None:
                derivations["artifacts"][0]["unexpected"] = True

            _rewrite_derivations(publication_path, add_unknown_field)

            with self.assertRaisesRegex(ValueError, "derivation reference"):
                ingest_cohort_publication(publication_path)

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
