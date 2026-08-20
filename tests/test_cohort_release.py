from collections.abc import Callable
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from scripts.cohort_partition import (
    PROJECTION_FIELDS,
    create_cohort_partition_manifest,
    write_cohort_partition_manifest,
)
from scripts.cohort_release import (
    _artifact,
    ingest_cohort_publication,
    publish_cohort_release,
    verify_cohort_publication,
    write_cohort_ingestion_evidence,
)
from scripts.collection_plan import (
    RuntimeResult,
    create_collection_plan,
    load_collection_plan,
    write_collection_plan,
)
from scripts.physics_relational_supervision import write_relational_supervision
from scripts.production_plan import (
    create_production_plan,
    execute_production_plan,
    write_production_plan,
)
from tests.test_production_plan import _evidence, _parameters, _pilot_report, _scenario
from tests.test_representative_pilot import _fixture_initial_identity, _write_fixture_shot


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
    derivation_path.write_text(
        json.dumps(derivations, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return publication_path


def _assert_no_removed_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = key.lower()
            if "checksum" in lowered:
                raise AssertionError(f"removed integrity field remains: {key}")
            _assert_no_removed_fields(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_removed_fields(item)


def _published_fixture(root: Path) -> Path:
    initial_identity = _fixture_initial_identity()
    plan = create_collection_plan(
        plan_version=3,
        scenarios=[
            _scenario("training", "training", 1)
            | {"expected_initial_engine_state_identity": initial_identity},
            _scenario("final", "calibration", 2)
            | {"expected_initial_engine_state_identity": initial_identity},
        ],
    )
    source_dir = root / "sources"
    source_dir.mkdir(parents=True)
    plan_path = write_collection_plan(plan, source_dir / "collection.json")
    loaded = load_collection_plan(plan_path)
    pilot = _pilot_report(plan)
    production = create_production_plan(
        plan_version=1,
        pilot_report=pilot,
        collection_plan=plan,
        parameters=_parameters(),
        evidence=_evidence(),
    )
    production_path = write_production_plan(production, source_dir / "published")

    output = root / "production"

    def runtime(request):
        shot = output / "accepted" / request.attempt_id / "shot_001"
        _write_fixture_shot(
            shot,
            initial_identity,
            {
                "version_envelope": {"generator_version": "v1"},
                "plan_identity": request.plan_identity,
                "plan_version": request.plan_version,
                "scenario_id": request.scenario_id,
                "scenario_identity": request.scenario_identity,
                "intervention_id": request.intervention_id,
                "intervention_identity": request.intervention_identity,
                "attempt_id": request.attempt_id,
                "attempt_number": request.attempt_number,
            },
            capture_id=f"capture-{request.attempt_id}",
        )
        write_relational_supervision(shot)
        realized = (
            ("collision",)
            if request.intervention_id.startswith("shot-")
            else ("no-contact/miss",)
        )
        return RuntimeResult(
            "accepted",
            realized_coverage_strata=realized,
            artifact_path=str(shot),
        )

    execute_production_plan(loaded, production_path, runtime, output)

    entries = [
        {
            "dataset_partition": scenario.exposure_role,
            "exposure_role": scenario.exposure_role,
            **{
                key: scenario.to_dict()[key]
                for key in PROJECTION_FIELDS
            },
        }
        for scenario in plan.scenarios
    ]
    partition = create_cohort_partition_manifest(
        partition_version=1,
        split_regime="instance_held_out",
        held_out_roles=[],
        entries=entries,
        provenance_records=[],
    )
    partition_path = write_cohort_partition_manifest(
        partition,
        source_dir / "partition.json",
    )
    scenario_paths = {}
    for scenario in plan.scenarios:
        manifest = scenario.to_dict()["scenario_manifest"]
        path = source_dir / f"{scenario.scenario_id}.scenario.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        scenario_paths[scenario.scenario_id] = path

    return publish_cohort_release(
        output,
        partition_manifest_path=partition_path,
        scenario_manifest_paths=scenario_paths,
        release_version=1,
        code_revision="fixture-revision",
        available_capabilities={
            "physics_capture_v1": "1",
            "physics_relational_supervision_v1": "1",
            "macro.steady-state": "physics_macro_labels_v1",
            "macro.structure-unstable": "physics_macro_labels_v1",
        },
        unavailable_capabilities={"material_identity": "not exported"},
    )


class CohortReleaseTests(unittest.TestCase):
    def test_publish_verify_and_ingest_plain_identity_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            publication_path = _published_fixture(Path(temporary))
            publication = verify_cohort_publication(publication_path)
            root = publication_path.parent.parent
            release = json.loads((root / publication["cohort_release"]["path"]).read_text())
            derivations = json.loads(
                (root / publication["authoritative_derivations"]["path"]).read_text()
            )

            self.assertTrue(publication["identity"].startswith("representative-cohort-publication-v1:"))
            self.assertEqual(release["release_version"], 1)
            self.assertEqual(len(release["primary_rollouts"]), 4)
            self.assertEqual(len(derivations["artifacts"]), 8)
            _assert_no_removed_fields(publication)
            _assert_no_removed_fields(release)
            _assert_no_removed_fields(derivations)

            evidence = ingest_cohort_publication(
                publication_path,
                required_capabilities=("physics_capture_v1",),
            )
            self.assertEqual(evidence.rollout_count, 4)
            self.assertEqual(evidence.terminal_observation_count, 4)
            self.assertEqual(
                evidence.label_derivations["physics_macro_labels_v1"],
                {"derivation_spec_version": "macro_labels_derivation_v2"},
            )

            evidence_path = write_cohort_ingestion_evidence(
                evidence,
                Path(temporary) / "ingestion.json",
            )
            persisted = json.loads(evidence_path.read_text(encoding="utf-8"))
            _assert_no_removed_fields(persisted)

    def test_malformed_published_scenario_manifest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            publication_path = _published_fixture(Path(temporary))
            root = publication_path.parent.parent
            publication = json.loads(publication_path.read_text(encoding="utf-8"))
            release = json.loads((root / publication["cohort_release"]["path"]).read_text())
            scenario_path = publication_path.parent / release["scenario_manifests"][0]["path"]
            scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
            del scenario["scenario_lineage"]
            scenario_path.write_text(json.dumps(scenario), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "[Ss]cenario"):
                ingest_cohort_publication(publication_path)

    def test_unknown_fields_missing_capabilities_and_cross_release_binding_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            publication_path = _published_fixture(Path(temporary))
            publication = json.loads(publication_path.read_text(encoding="utf-8"))
            publication["unexpected"] = True
            publication_path.write_text(json.dumps(publication), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "publication.*unknown"):
                ingest_cohort_publication(publication_path)

        with tempfile.TemporaryDirectory() as temporary:
            publication_path = _published_fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "capabilities are unavailable"):
                ingest_cohort_publication(
                    publication_path,
                    required_capabilities=("material_identity",),
                )

            def bind_another_release(derivations: dict[str, Any]) -> None:
                derivations["source_cohort_release_identity"] = "representative-cohort-release-v1:other"

            _rewrite_derivations(publication_path, bind_another_release)
            with self.assertRaisesRegex(ValueError, "another cohort release"):
                ingest_cohort_publication(publication_path)

    def test_unknown_authoritative_sidecar_reference_field_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            publication_path = _published_fixture(Path(temporary))

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


if __name__ == "__main__":
    unittest.main()
