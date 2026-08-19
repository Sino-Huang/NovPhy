from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from PIL import Image

from scripts.collection_plan import (
    PLAN_COPY_FILENAME,
    RuntimeResult,
    create_collection_plan,
    execute_collection_plan,
    load_collection_plan,
    write_collection_plan,
)
from scripts.cohort_partition import create_cohort_partition_manifest
from scripts.physics_capture_contract import load_physics_capture
from scripts.physics_artifact_validation import (
    validate_physics_shot_artifact as validate_physics_shot_artifact_without_mock,
)
from scripts.physics_macro_labels import (
    DERIVATION_SPEC_VERSION,
    SemanticStatus,
    derivation_spec_digest,
    derivation_spec_json,
    derive_macro_labels_for_shot,
)
from scripts.physics_material_damage import (
    MATERIAL_DAMAGE_MAPPING_SCHEMA_VERSION,
    MATERIAL_UNAVAILABLE_LABEL,
    MAPPING_SOURCE_FACTS,
    SUPPORTED_DAMAGE_LIFECYCLE_MAPPING,
)
from scripts.physics_rollout_contract import CaptureProvenance
from scripts.physics_rollout_persistence import persist_physics_rollout
from scripts.physics_rollout_semantics import initial_engine_state_identity
from scripts.representative_pilot import (
    CAPABILITY_ATOMIC_PHYSICS_ARTIFACT,
    CAPABILITY_BOUNDED_NEGATIVE_EVIDENCE,
    CAPABILITY_CAUSAL_ENTITIES,
    CAPABILITY_DETERMINISTIC_REPLAY,
    CAPABILITY_INITIAL_STATE_IDENTITY,
    CAPABILITY_INSTANCE_HELD_OUT_PARTITION,
    CAPABILITY_PHYSICAL_REGIME_GATE,
    CAPABILITY_RELATIONAL_SUPERVISION,
    CAPABILITY_TEMPLATE_HELD_OUT_PARTITION,
    PilotReport,
    PilotPartitionAudit,
    ReplayInput,
    build_parser,
    load_pilot_report,
    run_representative_pilot,
    write_pilot_report,
)
from scripts.scenario_manifest import BenchmarkCondition, create_generated_manifest, scenario_manifest_projection


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "physics_capture_v1"
XML = b'''<?xml version="1.0" encoding="utf-8"?>
<Level width="2">
  <Camera maxWidth="30" minWidth="20" />
  <Birds><Bird type="BirdRed" /></Birds>
  <Slingshot x="-8" y="-2" />
  <GameObjects><Pig type="BasicSmall" x="1" y="-3" rotation="0" /></GameObjects>
</Level>
'''
PROVENANCE = CaptureProvenance("0" * 64, "1" * 64, "2" * 64)


def _fixture_records(name: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in (FIXTURE / name).read_text(encoding="utf-8").splitlines()]


def _png() -> bytes:
    image = Image.new("RGB", (2, 1), (12, 34, 56))
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    return encoded.getvalue()


@dataclass(frozen=True)
class FixturePacket:
    png: bytes
    state: dict[str, object]
    events: tuple[dict[str, object], ...]


class FixtureBridge:
    def __init__(self, packet: FixturePacket) -> None:
        self.packet = packet

    def get_physics_capture_v1(self) -> FixturePacket:
        return self.packet


def _manifest(seed: int = 41, template: str = "scenario-template-v1:fixture"):
    return create_generated_manifest(
        XML,
        benchmark_condition=BenchmarkCondition("novelty_level_1", "type0101"),
        template_identity=template,
        generator_identity="novphy-task-generator",
        generator_version="canonical-v1",
        generation_seed=seed,
        declared_inputs={"layout_choice": seed},
        parameter_realization={"shift_x": seed / 100},
    )


def _coverage() -> dict[str, dict[str, object]]:
    return {
        "no-contact/miss": {"status": "targeted", "intervention_ids": ["miss-shot"]},
        "collision": {"status": "targeted", "intervention_ids": ["collision-shot"]},
        "persistent support": {"status": "inapplicable", "rationale": "not in fixture"},
        "support change": {"status": "inapplicable", "rationale": "not in fixture"},
        "destruction": {"status": "inapplicable", "rationale": "not in fixture"},
        "pig removal": {"status": "inapplicable", "rationale": "not in fixture"},
        "explosion": {"status": "inapplicable", "rationale": "not in fixture"},
        "stability transitions": {"status": "inapplicable", "rationale": "not in fixture"},
        "level clear": {"status": "inapplicable", "rationale": "not in fixture"},
        "level fail": {"status": "inapplicable", "rationale": "not in fixture"},
    }


def _action(release_x: int, release_y: int) -> tuple[dict[str, object], dict[str, object]]:
    interface = {
        "action_type": "drag_hold_release",
        "coordinate_frame": "slingshot_relative",
        "drag_start": [100, 200],
        "drag_release": [release_x, release_y],
        "tapTime": 0,
        "releaseTime": 600,
        "frame_height": 480,
        "socket_command": {"x": 100 + release_x, "y": 279 + release_y, "tapTime": 0, "releaseTime": 600},
    }
    engine = {
        "coordinate_frame": "slingshot_relative",
        "release_offset": [release_x, release_y],
        "release_point": [100 + release_x, 200 - release_y],
        "tap_time_ms": 0,
        "release_time_ms": 600,
    }
    return interface, engine


def _loaded_plan(
    root: Path,
    expected_identity: str,
    second_exposure_role: str = "calibration",
):
    manifest = _manifest()
    final_manifest = _manifest(seed=42, template="scenario-template-v1:held-out")
    projection = scenario_manifest_projection(manifest, "fixtures/training.scenario.json")
    final_projection = scenario_manifest_projection(final_manifest, "fixtures/final.scenario.json")
    collision_interface, collision_engine = _action(30, 50)
    miss_interface, miss_engine = _action(-20, 40)

    def scenario(scenario_id: str, exposure_role: str, scenario_projection):
        return {
            "scenario_id": scenario_id,
            "exposure_role": exposure_role,
            **scenario_projection,
            "expected_initial_engine_state_identity": expected_identity,
            "retry_policy": {
                "max_attempts": 1,
                "transient_failure_codes": [],
                "stopping_rule": "execute_all_interventions",
            },
            "negative_specification": {
                "cap": 1,
                "intervention_ids": ["miss-shot"],
                "semantic_justification": "fixture negative evidence",
            },
            "interventions": [
                {
                    "id": "collision-shot",
                    "ordinal": 1,
                    "intended_coverage_stratum": "collision",
                    "source": "geometry_stratified",
                    "interface_action": collision_interface,
                    "engine_relative_action": collision_engine,
                    "mapping_version": "science-birds-slingshot-relative-v1",
                    "slingshot_reference": {"gameX": 100, "gameY": 200},
                    "source_provenance": {
                        "scenario_geometry_identity": "geometry-v1:fixture",
                        "stratum": "fixture-target",
                        "feasibility_rule": "fixture-rule-v1",
                    },
                },
                {
                    "id": "miss-shot",
                    "ordinal": 2,
                    "intended_coverage_stratum": "no-contact/miss",
                    "source": "targeted_rare",
                    "interface_action": miss_interface,
                    "engine_relative_action": miss_engine,
                    "mapping_version": "science-birds-slingshot-relative-v1",
                    "slingshot_reference": {"gameX": 100, "gameY": 200},
                    "source_provenance": {
                        "target_stratum": "no-contact/miss",
                        "selection_rule": "fixture-clearance-v1",
                    },
                },
            ],
            "source_dispositions": {
                "geometry_stratified": {"status": "included"},
                "targeted_rare": {"status": "included"},
                "benchmark_agent_replay": {"status": "unavailable", "rationale": "no fixture trace"},
            },
            "coverage_strata": _coverage(),
        }

    plan = create_collection_plan(
        plan_version=1,
        scenarios=[
            scenario("fixture-training", "training", projection),
            scenario("fixture-final", second_exposure_role, final_projection),
        ],
    )
    path = root / "plan.json"
    write_collection_plan(plan, path)
    return load_collection_plan(path), (manifest, final_manifest)


def _fixture_initial_identity() -> str:
    capture = load_physics_capture(FIXTURE / "physics_state.jsonl", FIXTURE / "physics_events.jsonl")
    return initial_engine_state_identity(capture)


def _write_fixture_shot(
    path: Path,
    expected_identity: str,
    scenario_context: dict[str, object],
    *,
    event_count: int = 1,
) -> None:
    states = _fixture_records("physics_state.jsonl")[1:]
    events = _fixture_records("physics_events.jsonl")
    initial = FixturePacket(_png(), states[0], ())
    bridge = FixtureBridge(FixturePacket(_png(), states[1], tuple(events[:event_count])))
    persist_physics_rollout(
        bridge,
        path,
        target_fps=30.0,
        duration_seconds=1.0,
        max_frames=2,
        state_header=None,
        provenance=PROVENANCE,
        initial_capture=initial,
        shoot=lambda: None,
        expected_initial_engine_state_identity=expected_identity,
        scenario_context=scenario_context,
        clock=lambda: 0.0,
        sleeper=lambda _: None,
    )


def _partition_audits(
    manifests,
    *,
    training_source: str = "fixtures/training.scenario.json",
    second_exposure_role: str = "calibration",
) -> tuple[PilotPartitionAudit, PilotPartitionAudit]:
    training = scenario_manifest_projection(manifests[0], training_source)
    final = scenario_manifest_projection(manifests[1], "fixtures/final.scenario.json")
    instance = create_cohort_partition_manifest(
        partition_version=1,
        split_regime="instance_held_out",
        held_out_roles=[],
        entries=[
            {"dataset_partition": "train", "exposure_role": "training", **training},
            {"dataset_partition": "final", "exposure_role": second_exposure_role, **final},
        ],
        provenance_records=[],
    )
    template = create_cohort_partition_manifest(
        partition_version=1,
        split_regime="template_held_out",
        held_out_roles=[second_exposure_role],
        entries=[
            {"dataset_partition": "train", "exposure_role": "training", **training},
            {"dataset_partition": "final", "exposure_role": second_exposure_role, **final},
        ],
        provenance_records=[],
    )
    return (
        PilotPartitionAudit(
            instance,
            (str(training["scenario_lineage_identity"]), str(final["scenario_lineage_identity"])),
            (),
            "instance-fixture",
        ),
        PilotPartitionAudit(
            template,
            (str(training["scenario_lineage_identity"]), str(final["scenario_lineage_identity"])),
            (),
            "template-fixture",
        ),
    )


class RepresentativePilotTests(unittest.TestCase):
    def test_representative_pilot_report_terms_are_in_the_domain_glossary(self) -> None:
        context = (ROOT / "CONTEXT.md").read_text(encoding="utf-8")
        for term in (
            "Representative pilot",
            "Pilot evidence",
            "Pilot disposition",
            "Deterministic artifact semantics",
            "Bounded negative evidence",
            "Atomic rollout validation",
            "Legal-contact ontology",
            "Material mapping",
            "Damage mapping",
        ):
            with self.subTest(term=term):
                self.assertIn(f"**{term}**:", context)

    def test_cli_replay_arguments_preserve_replay_binding_fields(self) -> None:
        args = build_parser().parse_args(
            [
                "assess",
                "--plan", "plan.json",
                "--collection-report", "collection.json",
                "--frozen-plan-copy", "frozen-plan.json",
                "--output", "report.json",
                "--version", "engine=fixture-v1",
                "--replay", "manifest.json", "level.xml", "scenario", "intervention", "replay-shot",
            ]
        )

        self.assertEqual(
            args.replay,
            [["manifest.json", "level.xml", "scenario", "intervention", "replay-shot"]],
        )

    def _run(
        self,
        root: Path,
        *,
        expected_identity: str | None = None,
        replay_xml: bytes = XML,
        replay_envelope: dict[str, str] | None = None,
        replay_event_count: int = 1,
        invalid_binding: bool = False,
        source_envelope_mode: str | None = None,
        duplicate_replay: bool = False,
        outside_root_quarantine_failure: bool = False,
        mismatched_partition_projection: bool = False,
        duplicate_partition_audit: bool = False,
        realized_coverage_override: dict[str, tuple[str, ...]] | None = None,
        unrelated_partitions: bool = False,
        required_capabilities: tuple[str, ...] | None = None,
        systematic_exporter_defects: tuple[str, ...] = (),
        unavailable_labels: dict[str, str] | None = None,
        pending_damage: bool = False,
        collection_event_count: int = 1,
        second_exposure_role: str = "calibration",
    ):
        actual_identity = _fixture_initial_identity()
        loaded, manifests = _loaded_plan(
            root,
            expected_identity or actual_identity,
            second_exposure_role,
        )
        output = root / "collection"
        envelope = {
            "player_sha256": PROVENANCE.player_sha256,
            "protocol_sha256": PROVENANCE.protocol_sha256,
            "archive_sha256": PROVENANCE.archive_sha256,
            "generator_version": "canonical-v1",
        }
        scenario = loaded.plan.scenarios[0]
        replay_intervention = scenario.interventions[0]
        replay_shot = root / "replay" / "shot_001"
        declared_replay_envelope = replay_envelope or envelope
        _write_fixture_shot(
            replay_shot,
            actual_identity,
            {
                "version_envelope": declared_replay_envelope,
                "plan_identity": loaded.plan.identity,
                "plan_version": loaded.plan.plan_version,
                "scenario_id": scenario.scenario_id,
                "scenario_identity": scenario.identity,
                "intervention_id": replay_intervention.id,
                "intervention_identity": replay_intervention.identity,
            },
            event_count=replay_event_count,
        )

        def runtime(request):
            shot = output / "artifacts" / request.attempt_id / "shot_001"
            intervention_identity = request.intervention_identity
            targeted_failure = request.scenario_id == "fixture-training" and request.intervention_id == "collision-shot"
            if outside_root_quarantine_failure and targeted_failure:
                shot = root / "outside-artifacts" / request.attempt_id / "shot_001"
            if (invalid_binding or outside_root_quarantine_failure) and targeted_failure:
                intervention_identity = "collection-plan-intervention-v1:sha256:" + "0" * 64
            source_context = {
                "version_envelope": envelope,
                "plan_identity": request.plan_identity,
                "plan_version": request.plan_version,
                "scenario_id": request.scenario_id,
                "scenario_identity": request.scenario_identity,
                "intervention_id": request.intervention_id,
                "intervention_identity": intervention_identity,
                "attempt_id": request.attempt_id,
                "attempt_number": request.attempt_number,
            }
            if source_envelope_mode == "missing-context":
                source_context.pop("version_envelope")
            _write_fixture_shot(
                shot,
                actual_identity,
                source_context,
                event_count=collection_event_count,
            )
            if pending_damage:
                state_path = shot / "physics_state.jsonl"
                state_path.write_bytes(state_path.read_bytes().replace(b'"life":99', b'"life":100', 1))
                event_path = shot / "physics_events.jsonl"
                event_path.write_bytes(event_path.read_bytes().replace(b'"reason":"damage"', b'"reason":"not_damage"'))
            if source_envelope_mode == "tampered-metadata":
                metadata_path = shot / "metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["player_sha256"] = "f" * 64
                metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
            return RuntimeResult(
                "accepted",
                realized_coverage_strata=(realized_coverage_override or {}).get(
                    request.intervention_id,
                    ("collision",) if request.intervention_id == "collision-shot" else ("no-contact/miss",),
                ),
                artifact_path=str(shot),
            )

        audits = _partition_audits(
            (_manifest(seed=51), _manifest(seed=52, template="scenario-template-v1:unrelated-held-out"))
            if unrelated_partitions else manifests,
            training_source=(
                "fixtures/different-training.scenario.json"
                if mismatched_partition_projection else "fixtures/training.scenario.json"
            ),
            second_exposure_role=second_exposure_role,
        )
        if duplicate_partition_audit:
            audits = (*audits, audits[0])
        return run_representative_pilot(
            loaded,
            runtime,
            output,
            version_envelope=envelope,
            required_capabilities=required_capabilities
            or (
                CAPABILITY_CAUSAL_ENTITIES,
                CAPABILITY_ATOMIC_PHYSICS_ARTIFACT,
                CAPABILITY_INITIAL_STATE_IDENTITY,
                CAPABILITY_RELATIONAL_SUPERVISION,
                CAPABILITY_DETERMINISTIC_REPLAY,
                CAPABILITY_INSTANCE_HELD_OUT_PARTITION,
                CAPABILITY_TEMPLATE_HELD_OUT_PARTITION,
            ),
            replay_inputs=(ReplayInput(
                manifests[0],
                replay_xml,
                "fixture-replay",
                declared_replay_envelope,
                scenario.scenario_id,
                replay_intervention.identity,
                replay_shot,
            ),) * (2 if duplicate_replay else 1),
            partition_audits=audits,
            systematic_exporter_defects=systematic_exporter_defects,
            unavailable_labels=unavailable_labels or {},
        )

    def test_final_evaluation_scenarios_cannot_enter_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "final_evaluation"):
                self._run(
                    Path(temporary),
                    second_exposure_role="final_evaluation",
                )

    def test_report_round_trip_is_deterministic_and_identity_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self._run(root)
            path = root / "pilot-report.json"
            write_pilot_report(report, path)
            first = path.read_bytes()
            self.assertEqual(load_pilot_report(path), report)
            write_pilot_report(report, path)
            self.assertEqual(path.read_bytes(), first)
            self.assertEqual(report.identity, load_pilot_report(path).identity)

    def test_persisted_report_records_pending_source_bound_macro_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self._run(root)
            persisted = json.loads(
                (root / "collection" / "representative_pilot_report.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(persisted["schema"], "representative_pilot_report_v3")
            self.assertEqual(persisted["report_version"], 3)
            self.assertTrue(
                persisted["identity"].startswith("representative-pilot-report-v3:sha256:")
            )
            semantics = persisted["macro_semantics"]
            self.assertEqual(semantics["schema"], "representative_macro_semantics_v1")
            self.assertEqual(semantics["derivation_spec_version"], DERIVATION_SPEC_VERSION)
            self.assertEqual(semantics["derivation_spec_digest"], derivation_spec_digest())
            targets = {"cascade-active", "collapsed", "pigs-cleared"}
            self.assertEqual(set(semantics["predicates"]), targets)

            spec_predicates = derivation_spec_json()["pending_predicates"]
            validation_by_attempt = {
                item["attempt_id"]: item
                for item in report.to_dict()["attempts"]["atomic_validation"]
                if item["accepted"]
            }
            self.assertEqual(len(validation_by_attempt), 4)
            for name in sorted(targets):
                predicate = semantics["predicates"][name]
                self.assertEqual(
                    predicate["status"],
                    SemanticStatus.HYPOTHESIS_PENDING_REPRESENTATIVE_VALIDATION.value,
                )
                self.assertEqual(predicate["definition"], spec_predicates[name]["definition"])
                self.assertEqual(predicate["failure_cases"], spec_predicates[name]["failure_cases"])
                self.assertIn("non-fixture representative engine evidence", predicate["pending_reason"])
                self.assertEqual(
                    {row["attempt_id"] for row in predicate["evidence"]},
                    set(validation_by_attempt),
                )
                for row in predicate["evidence"]:
                    self.assertEqual(
                        set(row),
                        {
                            "attempt_id",
                            "capture_id",
                            "shot_id",
                            "physics_state_sha256",
                            "physics_events_sha256",
                            "derivation_spec_version",
                            "derivation_spec_digest",
                            "macro_label_artifact_sha256",
                            "value_summary",
                            "availability_summary",
                        },
                    )
                    artifact = validation_by_attempt[row["attempt_id"]]
                    labels = derive_macro_labels_for_shot(Path(artifact["artifact_path"]))
                    self.assertEqual(row["capture_id"], labels.capture_id)
                    self.assertEqual(row["shot_id"], labels.shot_id)
                    self.assertEqual(row["physics_state_sha256"], labels.state_sha256)
                    self.assertEqual(row["physics_events_sha256"], labels.events_sha256)
                    self.assertEqual(row["derivation_spec_version"], DERIVATION_SPEC_VERSION)
                    self.assertEqual(row["derivation_spec_digest"], derivation_spec_digest())
                    self.assertEqual(
                        row["macro_label_artifact_sha256"],
                        hashlib.sha256(labels.to_jsonl().encode("utf-8")).hexdigest(),
                    )

            available = set(persisted["available_capabilities"])
            unavailable = {
                item["capability"] for item in persisted["unavailable_capabilities"]
            }
            self.assertTrue(available.isdisjoint(unavailable))
            self.assertNotIn("representative_macro_semantics", available)
            self.assertIn("representative_macro_semantics", unavailable)

            material_damage = persisted["material_damage_semantics"]
            self.assertEqual(material_damage["schema"], "representative_material_damage_semantics_v1")
            self.assertTrue(material_damage["source_cohort_identity"].startswith("damage-source-cohort-v1:sha256:"))
            self.assertEqual(
                material_damage["material"],
                {
                    "availability": "unavailable_missing_engine_material_field",
                    "label": MATERIAL_UNAVAILABLE_LABEL,
                    "reason": "physics_capture_v1 does not export a material field",
                    "status": "unavailable",
                },
            )
            damage = material_damage["damage"]
            self.assertEqual(damage["availability"], "unavailable_insufficient_damage_lifecycle_evidence")
            self.assertEqual(damage["mapping_schema_version"], MATERIAL_DAMAGE_MAPPING_SCHEMA_VERSION)
            self.assertEqual(damage["mapping_version"], SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.mapping_version)
            self.assertEqual(damage["mapping_digest"], SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.digest)
            self.assertEqual(damage["source_facts"], list(MAPPING_SOURCE_FACTS))
            self.assertEqual(
                damage["status"],
                SemanticStatus.HYPOTHESIS_PENDING_REPRESENTATIVE_VALIDATION.value,
            )
            self.assertEqual(
                persisted["unavailable_labels"]["damage"],
                "no representative engine lifecycle evidence verifies the damage mapping",
            )
            self.assertEqual(
                {row["attempt_id"] for row in damage["evidence"]},
                set(validation_by_attempt),
            )
            evidence_fields = {
                "attempt_id",
                "capture_id",
                "shot_id",
                "physics_state_sha256",
                "physics_events_sha256",
                "derived_artifact_sha256",
                "record_count",
                "mapping_version",
                "mapping_digest",
                "source_cohort_identity",
                "receipt_status",
                "receipt_cohort_context",
                "receipt_source_records",
            }
            for row in damage["evidence"]:
                self.assertEqual(set(row), evidence_fields)
                self.assertEqual(
                    row,
                    validation_by_attempt[row["attempt_id"]]["material_damage_evidence"],
                )

    def test_material_damage_semantics_rejects_identity_recomputed_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            baseline = self._run(Path(temporary)).to_dict()
            mutations = []

            promoted_material = json.loads(json.dumps(baseline))
            promoted_material["material_damage_semantics"]["material"]["status"] = "engine_verified"
            mutations.append(promoted_material)

            stale_mapping = json.loads(json.dumps(baseline))
            stale_mapping["material_damage_semantics"]["damage"]["mapping_digest"] = "f" * 64
            mutations.append(stale_mapping)

            missing_evidence = json.loads(json.dumps(baseline))
            missing_evidence["material_damage_semantics"]["damage"]["evidence"].pop()
            mutations.append(missing_evidence)

            changed_cohort = json.loads(json.dumps(baseline))
            changed_cohort["material_damage_semantics"]["source_cohort_identity"] = "changed-cohort"
            mutations.append(changed_cohort)

            changed_source = json.loads(json.dumps(baseline))
            changed_source["material_damage_semantics"]["damage"]["evidence"][0][
                "physics_events_sha256"
            ] = "e" * 64
            mutations.append(changed_source)

            for payload in mutations:
                identity_payload = {key: value for key, value in payload.items() if key != "identity"}
                payload["identity"] = (
                    "representative-pilot-report-v3:sha256:"
                    + hashlib.sha256(
                        json.dumps(
                            identity_payload,
                            allow_nan=False,
                            ensure_ascii=True,
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode("utf-8")
                    ).hexdigest()
                )
                with self.subTest(mutation=payload["material_damage_semantics"]):
                    with self.assertRaises(ValueError):
                        PilotReport.from_dict(payload)

    def test_pilot_keeps_damage_unavailable_without_representative_lifecycle_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._run(
                Path(temporary), pending_damage=True, collection_event_count=4
            ).to_dict()
            damage = payload["material_damage_semantics"]["damage"]
            self.assertEqual(
                damage["status"],
                SemanticStatus.HYPOTHESIS_PENDING_REPRESENTATIVE_VALIDATION.value,
            )
            self.assertEqual(damage["availability"], "unavailable_insufficient_damage_lifecycle_evidence")
            self.assertEqual(
                payload["unavailable_labels"]["damage"],
                "no representative engine lifecycle evidence verifies the damage mapping",
            )
            self.assertNotIn("damage", payload["available_capabilities"])
            PilotReport.from_dict(payload)

    def test_pilot_reports_verified_damage_only_with_both_runtime_witnesses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._run(Path(temporary), collection_event_count=4).to_dict()
            damage = payload["material_damage_semantics"]["damage"]
            self.assertEqual(damage["status"], SemanticStatus.ENGINE_VERIFIED.value)
            self.assertEqual(damage["availability"], "available")
            self.assertNotIn("damage", payload["unavailable_labels"])
            self.assertTrue(all(row["receipt_status"] == SemanticStatus.ENGINE_VERIFIED.value for row in damage["evidence"]))

    def test_report_reload_is_portable_after_source_artifacts_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._run(root)
            report_path = root / "collection" / "representative_pilot_report.json"
            for artifact in (root / "collection" / "artifacts").glob("**/shot_001"):
                shutil.rmtree(artifact)
            loaded = load_pilot_report(report_path)
            self.assertEqual(loaded, load_pilot_report(report_path))

    def test_macro_semantics_strict_parsing_rejects_promotion_and_incomplete_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = self._run(Path(temporary))
            baseline = report.to_dict()
            mutations = []

            promoted = json.loads(json.dumps(baseline))
            promoted["macro_semantics"]["predicates"]["cascade-active"]["status"] = "engine_verified"
            mutations.append(promoted)

            incomplete = json.loads(json.dumps(baseline))
            training_attempt_id = next(
                item["attempt_id"]
                for item in incomplete["attempts"]["atomic_validation"]
                if item["accepted"] and item["scenario_id"] == "fixture-training"
            )
            incomplete["macro_semantics"]["predicates"]["collapsed"]["evidence"] = [
                row
                for row in incomplete["macro_semantics"]["predicates"]["collapsed"]["evidence"]
                if row["attempt_id"] != training_attempt_id
            ]
            mutations.append(incomplete)

            stale = json.loads(json.dumps(baseline))
            stale["macro_semantics"]["derivation_spec_version"] = "macro_labels_derivation_v1"
            mutations.append(stale)

            overlapping = json.loads(json.dumps(baseline))
            overlapping["available_capabilities"].append("representative_macro_semantics")
            mutations.append(overlapping)

            for payload in mutations:
                with self.subTest(mutation=payload["macro_semantics"]):
                    identity_payload = {key: value for key, value in payload.items() if key != "identity"}
                    payload["identity"] = (
                        "representative-pilot-report-v3:sha256:"
                        + hashlib.sha256(
                            json.dumps(
                                identity_payload,
                                allow_nan=False,
                                ensure_ascii=True,
                                separators=(",", ":"),
                                sort_keys=True,
                            ).encode("utf-8")
                        ).hexdigest()
                    )
                    with self.assertRaises(ValueError):
                        PilotReport.from_dict(payload)

    def test_macro_semantics_rejects_identity_recomputed_source_digest_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = self._run(Path(temporary)).to_dict()
            payload["macro_semantics"]["predicates"]["cascade-active"]["evidence"][0][
                "physics_state_sha256"
            ] = "f" * 64
            identity_payload = {key: value for key, value in payload.items() if key != "identity"}
            payload["identity"] = (
                "representative-pilot-report-v3:sha256:"
                + hashlib.sha256(
                    json.dumps(
                        identity_payload,
                        allow_nan=False,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()
            )

            with self.assertRaisesRegex(ValueError, "atomic validation"):
                PilotReport.from_dict(payload)

    def test_source_drift_during_macro_derivation_excludes_artifact_without_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def validate_then_drift(path):
                summary = validate_physics_shot_artifact_without_mock(path)
                if Path(path).is_relative_to(root / "collection" / "artifacts"):
                    state_path = Path(path) / "physics_state.jsonl"
                    content = state_path.read_text(encoding="utf-8")
                    state_path.write_text(content.replace("\n", " \n", 1), encoding="utf-8")
                return summary

            with mock.patch(
                "scripts.representative_pilot.validate_physics_shot_artifact",
                side_effect=validate_then_drift,
            ):
                report = self._run(root)

            payload = report.to_dict()
            evidence = payload["attempts"]["pilot_evidence"]
            self.assertEqual(evidence["accepted_count"], 0)
            self.assertEqual(evidence["excluded_count"], 4)
            self.assertTrue(all(
                "source digests differ from atomic validation" in item["reason"]
                for item in evidence["exclusions"]
            ))
            self.assertTrue(all(
                not predicate["evidence"]
                for predicate in payload["macro_semantics"]["predicates"].values()
            ))

    def test_source_and_persisted_frozen_plan_bytes_are_required(self) -> None:
        for mutation in ("missing-copy", "tampered-copy", "tampered-source"):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)

                    def execute_then_mutate(loaded, runtime, output_dir):
                        result = execute_collection_plan(loaded, runtime, output_dir)
                        target = loaded.path if mutation == "tampered-source" else output_dir / PLAN_COPY_FILENAME
                        if mutation == "missing-copy":
                            target.unlink()
                        else:
                            target.write_bytes(target.read_bytes() + b"\n")
                        return result

                    with mock.patch(
                        "scripts.representative_pilot.execute_collection_plan",
                        side_effect=execute_then_mutate,
                    ):
                        with self.assertRaisesRegex(ValueError, "plan|Plan"):
                            self._run(root)

    def test_bounded_fixture_reports_evidence_but_cannot_be_a_complete_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = self._run(Path(temporary))
            self.assertEqual(report.pilot_status, "rejected")
            payload = report.to_dict()
            self.assertEqual(payload["attempts"]["collection_accounting"]["accepted_count"], 4)
            self.assertEqual(payload["attempts"]["pilot_evidence"]["accepted_count"], 4)
            self.assertEqual(payload["coverage"]["realized_counts"]["collision"], 2)
            self.assertEqual(payload["coverage"]["realized_counts"]["no-contact/miss"], 2)
            self.assertTrue(all(item["accepted"] for item in payload["supervision"]))
            self.assertTrue(all(item["matched"] for item in payload["initial_state_identities"]))
            self.assertTrue(all(item["passed"] for item in payload["partition_audits"]))
            self.assertTrue(all(len(item["accepted_artifact_bindings"]) == 4 for item in payload["partition_audits"]))
            self.assertTrue(payload["replays"][0]["passed"])
            self.assertIn(CAPABILITY_CAUSAL_ENTITIES, payload["available_capabilities"])
            self.assertNotIn(CAPABILITY_DETERMINISTIC_REPLAY, payload["available_capabilities"])
            self.assertNotIn("canonical_observations", payload["available_capabilities"])
            self.assertNotIn("scene_nodes", json.dumps(payload, sort_keys=True))
            unavailable = {item["capability"] for item in payload["unavailable_capabilities"]}
            self.assertIn(CAPABILITY_PHYSICAL_REGIME_GATE, unavailable)
            self.assertIn("fixed_step_stride_authority", unavailable)
            self.assertIn("canonical_observations", unavailable)
            self.assertIn("cohort_release", unavailable)
            self.assertIn("physical_regime_gate", payload["unavailable_labels"])
            self.assertNotIn("physical_regime", payload["unavailable_labels"])
            self.assertFalse(payload["coverage"]["audit"]["representative"])
            self.assertTrue(payload["coverage"]["audit"]["inapplicable"])
            self.assertNotIn(CAPABILITY_BOUNDED_NEGATIVE_EVIDENCE, payload["available_capabilities"])
            self.assertTrue(all(not item["passed"] for item in payload["coverage"]["audit"]["negative_evidence"]))
            self.assertTrue(all(
                "non-trigger raw contacts" in " ".join(item["violations"])
                for item in payload["coverage"]["audit"]["negative_evidence"]
            ))

    def test_replay_and_initial_identity_mismatches_reject_the_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = self._run(
                Path(temporary),
                expected_identity="0" * 64,
                replay_xml=XML.replace(b'width="2"', b'width="3"'),
            )
            payload = report.to_dict()
            self.assertEqual(report.pilot_status, "rejected")
            self.assertFalse(payload["replays"][0]["passed"])
            self.assertFalse(payload["initial_state_identities"][0]["matched"])

    def test_replay_envelope_and_artifact_semantics_must_match_source_evidence(self) -> None:
        cases = (
            ({
                "player_sha256": "f" * 64,
                "protocol_sha256": PROVENANCE.protocol_sha256,
                "archive_sha256": PROVENANCE.archive_sha256,
                "generator_version": "canonical-v1",
            }, 1),
            (None, 2),
        )
        for replay_envelope, replay_event_count in cases:
            with self.subTest(replay_envelope=replay_envelope, replay_event_count=replay_event_count):
                with tempfile.TemporaryDirectory() as temporary:
                    report = self._run(
                        Path(temporary),
                        replay_envelope=replay_envelope,
                        replay_event_count=replay_event_count,
                    )
                    replay = report.to_dict()["replays"][0]
                    self.assertFalse(replay["passed"])
                    self.assertIn("mismatch", replay["reason"])

    def test_source_envelope_is_required_and_bound_to_artifact_metadata(self) -> None:
        for mode in ("missing-context", "tampered-metadata"):
            with self.subTest(mode=mode):
                with tempfile.TemporaryDirectory() as temporary:
                    report = self._run(Path(temporary), source_envelope_mode=mode)
                    attempts = report.to_dict()["attempts"]["pilot_evidence"]
                    self.assertEqual(attempts["accepted_count"], 0)
                    self.assertEqual(attempts["excluded_count"], 4)
                    self.assertTrue(all("envelope" in item["reason"] for item in attempts["exclusions"]))
                    payload = report.to_dict()
                    self.assertTrue(all(
                        not predicate["evidence"]
                        for predicate in payload["macro_semantics"]["predicates"].values()
                    ))
                    self.assertNotIn(
                        "representative_macro_semantics", payload["available_capabilities"]
                    )

    def test_duplicate_replay_bindings_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "duplicate replay"):
                self._run(Path(temporary), duplicate_replay=True)

    def test_invalid_binding_is_quarantined_without_relational_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = self._run(Path(temporary), invalid_binding=True)
            attempts = report.to_dict()["attempts"]
            self.assertEqual(attempts["collection_accounting"]["accepted_count"], 4)
            self.assertEqual(attempts["pilot_evidence"]["accepted_count"], 3)
            self.assertEqual(attempts["pilot_evidence"]["excluded_count"], 1)
            exclusion = attempts["pilot_evidence"]["exclusions"][0]
            self.assertEqual(exclusion["pilot_disposition"], "quarantined")
            quarantine = Path(exclusion["quarantine_path"])
            self.assertTrue(quarantine.is_dir())
            self.assertTrue(Path(exclusion["failure_manifest_path"]).is_file())
            self.assertFalse((quarantine / "physics_relational_supervision.jsonl").exists())
            excluded_supervision = next(
                item for item in report.to_dict()["supervision"]
                if item["attempt_id"] == exclusion["attempt_id"]
            )
            self.assertFalse(excluded_supervision["accepted"])
            self.assertFalse(excluded_supervision["attempted"])

    def test_outside_root_artifact_records_atomic_quarantine_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self._run(root, outside_root_quarantine_failure=True)
            exclusion = report.to_dict()["attempts"]["pilot_evidence"]["exclusions"][0]
            self.assertEqual(exclusion["pilot_disposition"], "quarantine_failed")
            self.assertTrue(Path(exclusion["artifact_path"]).is_dir())
            manifest_path = Path(exclusion["failure_manifest_path"])
            self.assertTrue(manifest_path.is_relative_to(root / "collection" / "pilot_assessment_quarantine"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["attempt_id"], exclusion["attempt_id"])
            self.assertEqual(manifest["reason"], exclusion["reason"])
            self.assertEqual(manifest["original_artifact_path"], exclusion["artifact_path"])
            self.assertEqual(manifest["pilot_disposition"], "quarantine_failed")
            self.assertFalse((Path(exclusion["artifact_path"]) / "physics_relational_supervision.jsonl").exists())

    def test_ledger_negative_cap_and_coverage_are_reconciled_against_the_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = self._run(
                Path(temporary),
                realized_coverage_override={"collision-shot": ("no-contact/miss",)},
            )
            coverage = report.to_dict()["coverage"]["audit"]
            self.assertFalse(coverage["passed"])
            self.assertEqual(coverage["collection_realized_counts"]["no-contact/miss"], 4)
            self.assertTrue(any(item["stratum"] == "collision" for item in coverage["gaps"]))
            negative = coverage["negative_evidence"][0]
            self.assertFalse(negative["passed"])
            self.assertEqual(negative["cap"], 1)
            self.assertEqual(negative["realized_count"], 2)
            self.assertIn("cap", " ".join(negative["violations"]))

    def test_frozen_planned_slot_inventory_mismatch_rejects_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            def execute_with_missing_slot(loaded, runtime, output_dir):
                result = execute_collection_plan(loaded, runtime, output_dir)
                result["planned_slots"] = result["planned_slots"][:-1]
                return result

            with mock.patch(
                "scripts.representative_pilot.execute_collection_plan",
                side_effect=execute_with_missing_slot,
            ):
                report = self._run(Path(temporary))
            coverage = report.to_dict()["coverage"]["audit"]
            self.assertFalse(coverage["passed"])
            self.assertIn("planned slot inventory", " ".join(coverage["ledger_violations"]))

    def test_unrelated_partition_manifests_cannot_satisfy_pilot_audits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = self._run(Path(temporary), unrelated_partitions=True)
            audits = report.to_dict()["partition_audits"]
            self.assertEqual({item["split_regime"] for item in audits}, {"instance_held_out", "template_held_out"})
            self.assertTrue(all(not item["passed"] for item in audits))
            self.assertTrue(all("pilot scenario lineage inventory" in item["reason"] for item in audits))

    def test_partition_projection_must_exactly_match_the_frozen_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = self._run(Path(temporary), mismatched_partition_projection=True)
            audits = report.to_dict()["partition_audits"]
            self.assertTrue(all(not item["passed"] for item in audits))
            self.assertTrue(all("complete scenario manifest projection" in item["reason"] for item in audits))

    def test_duplicate_partition_audit_regime_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "duplicate partition audit"):
                self._run(Path(temporary), duplicate_partition_audit=True)

    def test_template_partition_leakage_is_rejected_by_the_public_audit_boundary(self) -> None:
        training = scenario_manifest_projection(_manifest(seed=1, template="shared-template"), "training.json")
        final = scenario_manifest_projection(_manifest(seed=2, template="shared-template"), "final.json")
        with self.assertRaisesRegex(ValueError, "template-held-out boundary"):
            create_cohort_partition_manifest(
                partition_version=1,
                split_regime="template_held_out",
                held_out_roles=["final_evaluation"],
                entries=[
                    {"dataset_partition": "train", "exposure_role": "training", **training},
                    {"dataset_partition": "final", "exposure_role": "final_evaluation", **final},
                ],
                provenance_records=[],
            )

    def test_unavailable_and_systematic_defects_are_explicit_and_block_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = self._run(
                Path(temporary),
                required_capabilities=(CAPABILITY_CAUSAL_ENTITIES, CAPABILITY_PHYSICAL_REGIME_GATE),
                systematic_exporter_defects=("collision payload exporter omits representative evidence",),
                unavailable_labels={"illegal_contact": "legal-contact ontology is unavailable"},
            )
            payload = report.to_dict()
            self.assertEqual(report.pilot_status, "rejected")
            self.assertIn(CAPABILITY_PHYSICAL_REGIME_GATE, {item["capability"] for item in payload["unavailable_capabilities"]})
            self.assertEqual(payload["unavailable_labels"]["illegal_contact"], "legal-contact ontology is unavailable")
            self.assertEqual(payload["permanent_or_systematic_exporter_defects"][0]["scope"], "systematic")


if __name__ == "__main__":
    unittest.main()
