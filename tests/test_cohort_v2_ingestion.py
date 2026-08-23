import unittest
import json
import os
from pathlib import Path
import shutil
import tempfile

from scripts.build_issue_54_evidence import build_ingestion_evidence
from world_model.data import (
    CohortV2IngestionError,
    CohortV2OracleWindowDataset,
    CohortV2ReleaseReader,
    probe_cohort_v2_final_access,
    score_cohort_v2_endpoints,
)
from world_model.data.cohort_v2 import CENTRAL_LABELS


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_RELEASE = ROOT / "data/runtime_evidence/issue-53-mixed-termination-v5"
CAPABILITY_DECLARATION = ROOT / "docs/data_contracts/cohort_v2_capabilities_v1.json"
SEALED_RELEASE = ROOT / ".local-artifacts/issue-53-mixed-termination-final-release-v5"


def _private_file(shadow: Path, relative: Path) -> Path:
    target = shadow / relative
    target.unlink()
    shutil.copy2(PUBLIC_RELEASE / relative, target)
    return target


class CohortV2ReleaseReaderTests(unittest.TestCase):
    def test_training_reader_exposes_observation_backed_central_windows(self) -> None:
        reader = CohortV2ReleaseReader(
            PUBLIC_RELEASE,
            capability_declaration_path=CAPABILITY_DECLARATION,
            workflow_kind="training",
            influence="learned_parameters",
        )

        self.assertEqual(reader.release_identity, "representative-cohort-v2-release-v5:issue-53:mixed-termination")
        self.assertEqual(len(reader.rollouts), 6)
        self.assertEqual(
            {rollout.coverage_stratum for rollout in reader.rollouts},
            {
                "no-contact/miss",
                "collision",
                "persistent support",
                "support change",
                "destruction",
                "stability transitions",
            },
        )

        dataset = CohortV2OracleWindowDataset(reader)
        self.assertEqual(len(dataset), 6)
        example = dataset[0]
        self.assertEqual(set(example.context.labels), set(CENTRAL_LABELS))
        self.assertEqual(example.target.fixed_step, example.context.fixed_step + 1)
        self.assertTrue(example.agent_observation.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(example.intervention["interface_action"]["coordinate_frame"], "slingshot_relative")
        self.assertIsNone(example.context.labels["steady-state"]["value"])
        self.assertNotEqual(example.context.labels["steady-state"]["availability"], "available")

    def test_access_and_capability_boundaries_fail_before_examples(self) -> None:
        with self.assertRaisesRegex(CohortV2IngestionError, "capabilit"):
            CohortV2ReleaseReader(
                PUBLIC_RELEASE,
                capability_declaration_path=CAPABILITY_DECLARATION,
                workflow_kind="training",
                influence="learned_parameters",
                requested_capabilities=("physical_regime_gate",),
            )
        with self.assertRaisesRegex(CohortV2IngestionError, "sealed final"):
            CohortV2ReleaseReader(
                PUBLIC_RELEASE,
                capability_declaration_path=CAPABILITY_DECLARATION,
                workflow_kind="final_evaluation",
                influence="frozen_final_metrics_after_authorization",
            )
        with self.assertRaisesRegex(CohortV2IngestionError, "not permitted"):
            CohortV2ReleaseReader(
                PUBLIC_RELEASE,
                capability_declaration_path=CAPABILITY_DECLARATION,
                workflow_kind="calibration",
                influence="learned_parameters",
            )

        reader = CohortV2ReleaseReader(
            PUBLIC_RELEASE,
            capability_declaration_path=CAPABILITY_DECLARATION,
            workflow_kind="training",
            influence="learned_parameters",
        )
        with self.assertRaisesRegex(CohortV2IngestionError, "canonical observation"):
            reader.load_observation(reader.rollouts[0], observation_role="canonical")

    def test_endpoint_scoring_consumes_available_values_and_skips_unavailable(self) -> None:
        reader = CohortV2ReleaseReader(
            PUBLIC_RELEASE,
            capability_declaration_path=CAPABILITY_DECLARATION,
            workflow_kind="model_selection",
            influence="configuration_selection",
        )

        def all_false(frame):
            return {
                "steady-state": False,
                "structure-unstable": False,
                "excess_penetration": False,
                "unsupported_stationary_or_floating_body": {
                    item["entity_id"]: False
                    for item in frame.labels[
                        "unsupported_stationary_or_floating_body"
                    ]
                },
            }

        score = score_cohort_v2_endpoints(reader, all_false)
        self.assertEqual(score.endpoint_count, 6)
        self.assertGreater(score.scored_value_count, 0)
        self.assertGreater(score.unavailable_value_count, 0)
        self.assertGreater(score.relation_record_count, 0)
        self.assertLessEqual(score.correct_value_count, score.scored_value_count)

    def test_each_central_capability_mutation_fails_closed(self) -> None:
        index = json.loads(
            (PUBLIC_RELEASE / "authoritative-derivation-index.json").read_text()
        )
        first_attempt = next(
            item["attempt_id"]
            for item in index["artifacts"]
            if item["exposure_role"] == "training"
        )
        paths = {
            item["kind"]: Path(item["path"])
            for item in index["artifacts"]
            if item["attempt_id"] == first_attempt
        }

        def mutate_contact(value):
            value["labels"][0]["predicates"]["contact"]["relations"].append(
                ["invented:a", "invented:b"]
            )

        def mutate_supports(value):
            value["labels"][0]["predicates"]["supports"]["relations"].append(
                ["invented:a", "invented:b"]
            )

        def mutate_macro(predicate):
            def mutate(value):
                label = value["labels"][0]["predicates"][predicate]
                label["availability"] = "available"
                label["value"] = False

            return mutate

        def mutate_excess(value):
            label = value["labels"][0]["predicates"]["excess_penetration"]
            label["value"] = not label["value"]

        def mutate_unsupported(value):
            label = value["labels"][0]["predicates"][
                "unsupported_stationary_or_floating_body"
            ][0]
            label["availability"] = "available"
            label["value"] = False

        cases = {
            "contact": (paths["micro"], mutate_contact),
            "supports": (paths["micro"], mutate_supports),
            "steady-state": (paths["macro"], mutate_macro("steady-state")),
            "structure-unstable": (
                paths["macro"],
                mutate_macro("structure-unstable"),
            ),
            "excess_penetration": (paths["physical-violations"], mutate_excess),
            "unsupported_stationary_or_floating_body": (
                paths["physical-violations"],
                mutate_unsupported,
            ),
        }
        for capability, (relative, mutate) in cases.items():
            with self.subTest(capability=capability), tempfile.TemporaryDirectory(dir=ROOT) as temporary:
                shadow = Path(temporary) / "release"
                shutil.copytree(PUBLIC_RELEASE, shadow, copy_function=os.link)
                path = _private_file(shadow, relative)
                value = json.loads(path.read_text())
                mutate(value)
                path.write_text(json.dumps(value), encoding="utf-8")

                with self.assertRaises(CohortV2IngestionError):
                    CohortV2ReleaseReader(
                        shadow,
                        capability_declaration_path=CAPABILITY_DECLARATION,
                        workflow_kind="training",
                        influence="learned_parameters",
                    )

    @unittest.skipUnless(SEALED_RELEASE.is_dir(), "sealed operator evidence is local")
    def test_authorized_final_probe_returns_audit_receipt_without_final_examples(self) -> None:
        receipt = probe_cohort_v2_final_access(PUBLIC_RELEASE, SEALED_RELEASE)

        self.assertTrue(receipt.passed)
        self.assertEqual(receipt.authorization_state, "authorized")
        self.assertEqual(receipt.observed_access_count, 1)
        self.assertFalse(hasattr(receipt, "rollouts"))

    @unittest.skipUnless(SEALED_RELEASE.is_dir(), "sealed operator evidence is local")
    def test_evidence_builder_runs_every_role_and_adversarial_boundary(self) -> None:
        report = build_ingestion_evidence(
            repository_root=ROOT,
            release_root=PUBLIC_RELEASE,
            sealed_root=SEALED_RELEASE,
            code_revision="test-revision",
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["counts"]["rollouts"], 18)
        self.assertEqual(report["counts"]["oracle_training_windows"], 18)
        self.assertEqual(set(report["roles"]), {"training", "calibration", "model_selection"})
        self.assertEqual(len(report["adversarial_checks"]), 12)
        self.assertTrue(all(item["passed"] for item in report["adversarial_checks"]))


if __name__ == "__main__":
    unittest.main()
