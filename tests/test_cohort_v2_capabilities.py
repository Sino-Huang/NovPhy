from __future__ import annotations

from copy import deepcopy
import json
import unittest

from scripts.cohort_v2_capabilities import (
    CENTRAL_CAPABILITIES,
    EXPECTED_DECLARATION_IDENTITY,
    EXPECTED_DECLARATION_SHA256,
    SUPPORTED_ARTIFACT_KINDS,
    build_central_v2_scope_claim,
    capability_declaration_reference,
    load_capability_declaration,
    negotiate_central_v2_capabilities,
    validate_capability_declaration,
    validate_capability_declaration_reference,
    validate_central_v2_scope_claim,
)


class CohortV2CapabilityTests(unittest.TestCase):
    def test_approved_declaration_has_exact_identity_scope_and_evidence_floor(self) -> None:
        declaration = load_capability_declaration()

        self.assertEqual(declaration["identity"], EXPECTED_DECLARATION_IDENTITY)
        self.assertEqual(capability_declaration_reference()["sha256"], EXPECTED_DECLARATION_SHA256)
        self.assertEqual(
            declaration["capabilities"]["required_central"]["micro_labels"],
            ["contact", "supports"],
        )
        self.assertEqual(
            declaration["capabilities"]["required_central"]["macro_labels"],
            ["structure-unstable", "steady-state"],
        )
        self.assertEqual(
            declaration["capabilities"]["required_central"]["violation_labels"],
            ["excess_penetration", "unsupported_stationary_or_floating_body"],
        )
        self.assertEqual(declaration["evidence_floor"]["minimum_positive_witnesses"], 2)
        self.assertEqual(declaration["evidence_floor"]["minimum_negative_witnesses"], 2)
        self.assertEqual(declaration["evidence_floor"]["minimum_boundary_windows"], 2)
        self.assertFalse(declaration["evidence_floor"]["final_evaluation_evidence_allowed"])
        self.assertTrue(declaration["evidence_semantics"]["unavailable_distinct_from_false"])

    def test_every_artifact_role_accepts_only_the_complete_central_scope(self) -> None:
        for artifact_kind in SUPPORTED_ARTIFACT_KINDS:
            with self.subTest(artifact_kind=artifact_kind):
                claim = build_central_v2_scope_claim(artifact_kind)
                validated = validate_central_v2_scope_claim(
                    claim,
                    artifact_kind=artifact_kind,
                )
                self.assertEqual(
                    tuple(validated["required_capabilities"]),
                    CENTRAL_CAPABILITIES,
                )

    def test_unknown_omitted_and_promoted_capabilities_fail_closed(self) -> None:
        claim = build_central_v2_scope_claim("collection_plan")

        unknown = deepcopy(claim)
        unknown["required_capabilities"].append("micro.inferred-from-filename")
        with self.assertRaisesRegex(ValueError, "Unknown cohort-v2 capability"):
            validate_central_v2_scope_claim(unknown)

        omitted = deepcopy(claim)
        omitted["required_capabilities"].remove("micro.contact")
        with self.assertRaisesRegex(ValueError, "Required central capabilities omitted"):
            validate_central_v2_scope_claim(omitted)

        promoted = deepcopy(claim)
        promoted["required_capabilities"].append("split.template_held_out")
        with self.assertRaisesRegex(ValueError, "Unsupported non-central capability promotion"):
            validate_central_v2_scope_claim(promoted)

    def test_declaration_rejects_unknown_omitted_and_promoted_scope(self) -> None:
        declaration = json.loads(json.dumps(load_capability_declaration()))

        unknown = deepcopy(declaration)
        unknown["capabilities"]["required_central"]["micro_labels"].append("velocity-bin")
        with self.assertRaisesRegex(ValueError, "Unknown cohort-v2 capability"):
            validate_capability_declaration(unknown)

        omitted = deepcopy(declaration)
        omitted["capabilities"]["required_central"]["macro_labels"].remove("steady-state")
        with self.assertRaisesRegex(ValueError, "Required central capabilities omitted"):
            validate_capability_declaration(omitted)

        promoted = deepcopy(declaration)
        promoted["capabilities"]["required_central"]["splits"].append("template_held_out")
        with self.assertRaisesRegex(ValueError, "Unsupported non-central capability promotion"):
            validate_capability_declaration(promoted)

    def test_wrong_declaration_identity_or_digest_fails_closed(self) -> None:
        reference = capability_declaration_reference()

        wrong_identity = dict(reference, identity="cohort-v2-capabilities-v1:sha256:wrong")
        with self.assertRaisesRegex(ValueError, "identity is wrong"):
            validate_capability_declaration_reference(wrong_identity)

        wrong_digest = dict(reference, sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "digest is wrong"):
            validate_capability_declaration_reference(wrong_digest)

    def test_negotiation_preserves_unavailable_and_requires_accepted_capabilities(self) -> None:
        claim = build_central_v2_scope_claim("consumer")
        available = {
            capability: f"accepted-artifact:{capability}"
            for capability in CENTRAL_CAPABILITIES
        }
        missing_capability = "violation.excess_penetration"
        incomplete = dict(available)
        del incomplete[missing_capability]

        with self.assertRaisesRegex(
            ValueError, "required capabilities are unavailable"
        ) as failure:
            negotiate_central_v2_capabilities(
                claim,
                artifact_kind="consumer",
                available_capabilities=incomplete,
                unavailable_capabilities={
                    missing_capability: "representative evidence is unavailable",
                },
            )
        self.assertIn("representative evidence is unavailable", str(failure.exception))

        negotiated = negotiate_central_v2_capabilities(
            claim,
            artifact_kind="consumer",
            available_capabilities=available,
            unavailable_capabilities={},
        )
        self.assertEqual(tuple(negotiated), CENTRAL_CAPABILITIES)


if __name__ == "__main__":
    unittest.main()
