from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_issue_49_evidence import (
    DEFAULT_SOURCE_ROOT,
    Issue49EvidenceError,
    _expected_artifacts,
    _implementation_revision,
    build_issue_49_evidence,
    validate_issue_49_evidence,
)
from scripts.cohort_v2_macro_semantics import (
    CohortV2MacroSemanticsError,
    derive_capture_macro_labels,
)
from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_json
from scripts.physics_capture_v2 import load_physics_capture_v2, parse_physics_capture_v2


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_BUNDLE_IDENTITY = json.loads(
    (DEFAULT_SOURCE_ROOT / "capture-bundle-manifest.json").read_text(encoding="utf-8")
)["identity"]
IMPLEMENTATION_REVISION = _implementation_revision(ROOT)


def derive(case: str):
    capture = load_physics_capture_v2(DEFAULT_SOURCE_ROOT / f"captures/{case}.json")
    value = derive_capture_macro_labels(
        capture,
        source_reference=f"data/runtime_evidence/issue-44/captures/{case}.json",
        source_capture_bundle_identity=CAPTURE_BUNDLE_IDENTITY,
    )
    return capture, value


def label_at(derivation, step: int, predicate: str):
    return next(
        record["predicates"][predicate]
        for record in derivation["labels"]
        if record["fixed_step"] == step
    )


class CohortV2MacroDerivationTests(unittest.TestCase):
    def test_derives_debounced_stability_and_support_change_on_fixed_steps(self) -> None:
        _, derivation = derive("collision")

        self.assertIs(label_at(derivation, 1812, "steady-state")["value"], True)
        self.assertIs(label_at(derivation, 1758, "steady-state")["value"], False)
        self.assertIs(label_at(derivation, 1758, "structure-unstable")["value"], True)
        self.assertIs(label_at(derivation, 1760, "structure-unstable")["value"], False)

        collision = label_at(derivation, 1758, "structure-unstable")
        self.assertEqual(collision["source_interval"]["current_fixed_step"], 1758)
        self.assertIn(
            "event:1758:collision:0002",
            collision["source_interval"]["projected_event_ids"],
        )
        self.assertEqual(
            collision["evidence"]["added_support_relations"],
            [["runtime:platform:0000", "runtime:bird:0000"]],
        )

    def test_first_fixed_step_is_unavailable_not_false(self) -> None:
        _, derivation = derive("support-change")
        first = derivation["labels"][0]["predicates"]

        self.assertIsNone(first["steady-state"]["value"])
        self.assertEqual(
            first["steady-state"]["availability"],
            "unavailable_incomplete_debounce_window",
        )
        self.assertIsNone(first["structure-unstable"]["value"])
        self.assertEqual(
            first["structure-unstable"]["availability"],
            "unavailable_no_predecessor",
        )

    def test_missing_engine_transition_event_rejects_derivation(self) -> None:
        capture = load_physics_capture_v2(DEFAULT_SOURCE_ROOT / "captures/collision.json")
        mutated = deepcopy(capture.record)
        mutated["events"] = [
            event for event in mutated["events"] if event["event_type"] != "stable_exited"
        ]
        parsed = parse_physics_capture_v2(mutated)

        with self.assertRaisesRegex(
            CohortV2MacroSemanticsError,
            "does not match a same-step engine event",
        ):
            derive_capture_macro_labels(
                parsed,
                source_reference="mutation.json",
                source_capture_bundle_identity=CAPTURE_BUNDLE_IDENTITY,
            )


class Issue49BundleTests(unittest.TestCase):
    def test_dry_run_validates_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "issue-49"
            result = build_issue_49_evidence(
                output,
                implementation_revision=IMPLEMENTATION_REVISION,
                dry_run=True,
            )

            self.assertTrue(result["passed"])
            self.assertFalse(output.exists())

    def test_expected_bundle_passes_exact_rederivation(self) -> None:
        artifacts = _expected_artifacts(
            ROOT,
            DEFAULT_SOURCE_ROOT,
            IMPLEMENTATION_REVISION,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "issue-49"
            for relative_path, value in artifacts.items():
                write_immutable_cohort_v2_json(value, output / relative_path)

            result = validate_issue_49_evidence(output)

        self.assertTrue(result["passed"])
        self.assertEqual(result["capture_count"], 5)
        self.assertEqual(result["label_count"], 1228)

    def test_changed_label_rejects_bundle(self) -> None:
        artifacts = _expected_artifacts(
            ROOT,
            DEFAULT_SOURCE_ROOT,
            IMPLEMENTATION_REVISION,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "issue-49"
            for relative_path, value in artifacts.items():
                write_immutable_cohort_v2_json(value, output / relative_path)
            path = output / "derivations/collision.json"
            changed = json.loads(path.read_text(encoding="utf-8"))
            changed["labels"][1]["predicates"]["steady-state"]["value"] = False
            path.write_text(json.dumps(changed), encoding="utf-8")

            with self.assertRaisesRegex(Issue49EvidenceError, "exact re-derivation"):
                validate_issue_49_evidence(output)

    def test_changed_implementation_revision_rejects_bundle(self) -> None:
        artifacts = _expected_artifacts(
            ROOT,
            DEFAULT_SOURCE_ROOT,
            IMPLEMENTATION_REVISION,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "issue-49"
            for relative_path, value in artifacts.items():
                write_immutable_cohort_v2_json(value, output / relative_path)
            path = output / "bundle-manifest.json"
            changed = json.loads(path.read_text(encoding="utf-8"))
            changed["implementation_revision"] = "stale-cross-release-revision"
            path.write_text(json.dumps(changed), encoding="utf-8")

            with self.assertRaisesRegex(Issue49EvidenceError, "implementation revision"):
                validate_issue_49_evidence(output)


if __name__ == "__main__":
    unittest.main()
