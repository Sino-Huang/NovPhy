"""Fail-closed scope negotiation for the approved central cohort-v2 profile."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final


DECLARATION_SCHEMA: Final = "cohort_v2_capabilities_v1"
DECLARATION_VERSION: Final = 1
DECLARATION_IDENTITY_NAMESPACE: Final = "cohort-v2-capabilities-v1"
DECLARATION_REFERENCE_SCHEMA: Final = "cohort_v2_capability_reference_v1"
SCOPE_CLAIM_SCHEMA: Final = "cohort_v2_scope_claim_v1"
SCOPE_NAME: Final = "central_v2"
DECLARATION_PATH: Final = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "data_contracts"
    / "cohort_v2_capabilities_v1.json"
)

EXPECTED_DECLARATION_IDENTITY: Final = (
    "cohort-v2-capabilities-v1:sha256:"
    "3c7a871087a38a84b14364d27f410a80d7d14f1b5796e4b9b420f2e376499940"
)
EXPECTED_DECLARATION_SHA256: Final = (
    "6b27038cb4175aa978f40543048b531fe8f20b16cd4c7d07333e07e0991aaa4c"
)

SUPPORTED_ARTIFACT_KINDS: Final = (
    "producer",
    "collection_plan",
    "cohort_release",
    "derivation",
    "consumer",
)

EXPECTED_AUTHORITIES: Final = MappingProxyType({
    "approved_profile": {
        "path": (
            ".claude/project-docs/research/20260820-cohort-v2-capability-profile/"
            "research-ready-cohort-v2-capability-profile-proposal.md"
        ),
        "sha256": "676ce51194d0c0c1c8f9633910ed4c59123b053504306ab0578a8656d2fbcfae",
    },
    "github_issues": [1, 18, 33, 42, 43],
    "issue_33_audit": {
        "path": ".claude/project-docs/evidence/issue-33-section-16-audit-20260820/README.md",
        "sha256": "ed02915ae861f2268a830f61f5b9cfe1d6f16b8bc41afb6ca74caf46126d6841",
    },
})

EXPECTED_CENTRAL: Final = MappingProxyType({
    "micro_labels": ("contact", "supports"),
    "macro_labels": ("structure-unstable", "steady-state"),
    "violation_labels": (
        "excess_penetration",
        "unsupported_stationary_or_floating_body",
    ),
    "observations": ("agent", "canonical_access_restricted"),
    "replay": ("version_bounded_deterministic",),
    "exposure_roles": (
        "training",
        "calibration",
        "model_selection",
        "final_evaluation",
    ),
    "splits": ("instance_held_out",),
    "coverage_strata": (
        "no-contact/miss",
        "collision",
        "persistent support",
        "support change",
        "destruction",
        "stability transitions",
    ),
    "provenance": (
        "scenario_hierarchy",
        "deterministic_realization",
        "explicit_legacy_static",
        "single_shot_rollout",
        "frozen_outcome_independent_plan",
        "engine_authoritative_fixed_step",
        "positive_fixed_step_capture_stride",
        "complete_raw_contact_intervals",
        "atomic_rollout_validation",
        "typed_failure_and_quarantine_accounting",
        "transient_only_retries",
        "immutable_cohort_release",
        "source_bound_derivations",
        "role_separated_final_evaluation",
    ),
    "ingestion": (
        "public_fail_closed",
        "identity_and_digest_binding",
        "availability_preservation",
        "temporal_alignment_preservation",
        "exposure_restriction_enforcement",
    ),
})

EXPECTED_SECONDARY: Final = MappingProxyType({
    "evidence.bounded_negative": "spsg_contrastive_loss_ablation",
    "label.material": "novel_material_generalization",
    "macro.cascade-active": "extended_macro_event_prediction",
    "observation.learned_symbol_parser": "learned_symbol_stress_test",
    "supervision.micro_relation_usefulness": "reliability_gating",
    "supervision.physical_regime_gate": "regime_alignment_diagnostic",
})
EXPECTED_OPTIONAL: Final = ("intervention.benchmark_agent_action",)
EXPECTED_OUT_OF_SCOPE: Final = (
    "claim.cross_domain_evaluation",
    "claim.gravity_shift_generalization",
    "claim.planning",
    "label.damage",
    "macro.collapsed",
    "macro.pigs-cleared",
    "split.template_held_out",
    "violation.illegal_contact",
)
EXPECTED_EVIDENCE_FLOOR: Final = MappingProxyType({
    "final_evaluation_evidence_allowed": False,
    "minimum_boundary_windows": 2,
    "minimum_level_instances": 2,
    "minimum_negative_witnesses": 2,
    "minimum_non_final_scenario_lineages": 2,
    "minimum_positive_witnesses": 2,
    "minimum_scenario_templates": 2,
    "requires_unavailable_or_invalidation_check": True,
})
EXPECTED_EVIDENCE_SEMANTICS: Final = MappingProxyType({
    "availability_requires_accepted_source_evidence": True,
    "prohibited_availability_inference": [
        "filename",
        "rgb_content",
        "fixture",
        "command_success",
        "closed_issue_status",
    ],
    "unavailable_distinct_from_false": True,
})

_CENTRAL_PREFIXES: Final = MappingProxyType({
    "micro_labels": "micro",
    "macro_labels": "macro",
    "violation_labels": "violation",
    "observations": "observation",
    "replay": "replay",
    "exposure_roles": "exposure_role",
    "splits": "split",
    "coverage_strata": "coverage",
    "provenance": "provenance",
    "ingestion": "ingestion",
})


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _declaration_identity(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("identity", None)
    digest = sha256(_canonical_json(payload)).hexdigest()
    return f"{DECLARATION_IDENTITY_NAMESPACE}:sha256:{digest}"


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_exact_fields(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} is incomplete or contains unknown fields")


def _require_string_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{name} must be a list of nonempty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must be unique")
    return tuple(value)


def _flatten_central(value: Mapping[str, Sequence[str]]) -> frozenset[str]:
    return frozenset(
        f"{_CENTRAL_PREFIXES[category]}.{capability}"
        for category, capabilities in value.items()
        for capability in capabilities
    )


CENTRAL_CAPABILITIES: Final = tuple(sorted(_flatten_central(EXPECTED_CENTRAL)))
NONCENTRAL_CAPABILITIES: Final = frozenset(
    (*EXPECTED_SECONDARY, *EXPECTED_OPTIONAL, *EXPECTED_OUT_OF_SCOPE)
)
KNOWN_CAPABILITIES: Final = frozenset((*CENTRAL_CAPABILITIES, *NONCENTRAL_CAPABILITIES))


def _validate_capability_set(actual: frozenset[str], name: str) -> None:
    expected = frozenset(CENTRAL_CAPABILITIES)
    promoted = sorted((actual - expected) & NONCENTRAL_CAPABILITIES)
    if promoted:
        raise ValueError(f"Unsupported non-central capability promotion in {name}: {promoted!r}")
    unknown = sorted(actual - KNOWN_CAPABILITIES)
    if unknown:
        raise ValueError(f"Unknown cohort-v2 capability in {name}: {unknown!r}")
    omitted = sorted(expected - actual)
    if omitted:
        raise ValueError(f"Required central capabilities omitted from {name}: {omitted!r}")


def validate_capability_declaration(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate one declaration against the scientifically approved v1 scope."""
    declaration = _require_mapping(value, "Capability declaration")
    _require_exact_fields(
        declaration,
        {
            "schema",
            "declaration_version",
            "identity",
            "scope_name",
            "authorities",
            "capabilities",
            "evidence_floor",
            "evidence_semantics",
        },
        "Capability declaration",
    )
    if declaration["schema"] != DECLARATION_SCHEMA:
        raise ValueError("Capability declaration schema is unsupported")
    if declaration["declaration_version"] != DECLARATION_VERSION:
        raise ValueError("Capability declaration version is unsupported")
    if declaration["scope_name"] != SCOPE_NAME:
        raise ValueError("Capability declaration scope is unsupported")
    if declaration["authorities"] != EXPECTED_AUTHORITIES:
        raise ValueError(
            "Capability declaration authorities differ from the approved profile or audit"
        )

    capabilities = _require_mapping(declaration["capabilities"], "Capability classifications")
    _require_exact_fields(
        capabilities,
        {"required_central", "required_secondary", "optional", "out_of_scope"},
        "Capability classifications",
    )
    central = _require_mapping(capabilities["required_central"], "Required central capabilities")
    _require_exact_fields(central, set(EXPECTED_CENTRAL), "Required central capabilities")
    normalized_central = {
        category: _require_string_list(central[category], f"Required central {category}")
        for category in EXPECTED_CENTRAL
    }
    _validate_capability_set(_flatten_central(normalized_central), "declaration")
    if normalized_central != EXPECTED_CENTRAL:
        raise ValueError("Required central capability categories differ from the approved profile")

    secondary = _require_mapping(capabilities["required_secondary"], "Secondary capabilities")
    if secondary != EXPECTED_SECONDARY:
        raise ValueError("Secondary capability dispositions differ from the approved profile")
    optional = _require_string_list(capabilities["optional"], "Optional capabilities")
    if optional != EXPECTED_OPTIONAL:
        raise ValueError("Optional capability dispositions differ from the approved profile")
    out_of_scope = _require_string_list(capabilities["out_of_scope"], "Out-of-scope capabilities")
    if out_of_scope != EXPECTED_OUT_OF_SCOPE:
        raise ValueError("Out-of-scope capability dispositions differ from the approved profile")
    if declaration["evidence_floor"] != EXPECTED_EVIDENCE_FLOOR:
        raise ValueError("Capability declaration evidence floor differs from the approved profile")
    if declaration["evidence_semantics"] != EXPECTED_EVIDENCE_SEMANTICS:
        raise ValueError(
            "Capability declaration evidence semantics differ from the approved profile"
        )
    if declaration["identity"] != _declaration_identity(declaration):
        raise ValueError("Capability declaration identity is stale")
    if declaration["identity"] != EXPECTED_DECLARATION_IDENTITY:
        raise ValueError("Capability declaration identity is unsupported")
    return declaration


def load_capability_declaration(path: Path = DECLARATION_PATH) -> Mapping[str, Any]:
    """Load the exact approved declaration; no alternate declaration is accepted."""
    source = Path(path)
    try:
        raw_bytes = source.read_bytes()
        value = json.loads(raw_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load capability declaration {source}: {error}") from error
    declaration = validate_capability_declaration(
        _require_mapping(value, "Capability declaration")
    )
    if source.resolve() == DECLARATION_PATH.resolve():
        digest = sha256(raw_bytes).hexdigest()
        if digest != EXPECTED_DECLARATION_SHA256:
            raise ValueError("Capability declaration file digest is stale")
    return declaration


def capability_declaration_reference() -> dict[str, str]:
    load_capability_declaration()
    return {
        "schema": DECLARATION_REFERENCE_SCHEMA,
        "identity": EXPECTED_DECLARATION_IDENTITY,
        "sha256": EXPECTED_DECLARATION_SHA256,
    }


def validate_capability_declaration_reference(value: Any) -> Mapping[str, Any]:
    reference = _require_mapping(value, "Capability declaration reference")
    _require_exact_fields(
        reference,
        {"schema", "identity", "sha256"},
        "Capability declaration reference",
    )
    if reference["schema"] != DECLARATION_REFERENCE_SCHEMA:
        raise ValueError("Capability declaration reference schema is unsupported")
    if reference["identity"] != EXPECTED_DECLARATION_IDENTITY:
        raise ValueError("Capability declaration reference identity is wrong")
    if reference["sha256"] != EXPECTED_DECLARATION_SHA256:
        raise ValueError("Capability declaration reference digest is wrong")
    load_capability_declaration()
    return reference


def build_central_v2_scope_claim(artifact_kind: str) -> dict[str, Any]:
    if artifact_kind not in SUPPORTED_ARTIFACT_KINDS:
        raise ValueError(f"Unsupported central-v2 artifact kind: {artifact_kind!r}")
    return {
        "schema": SCOPE_CLAIM_SCHEMA,
        "scope": SCOPE_NAME,
        "artifact_kind": artifact_kind,
        "capability_declaration": capability_declaration_reference(),
        "required_capabilities": list(CENTRAL_CAPABILITIES),
    }


def validate_central_v2_scope_claim(
    value: Any,
    *,
    artifact_kind: str | None = None,
) -> Mapping[str, Any]:
    """Reject incomplete, unknown, or promoted capability requests for central v2."""
    claim = _require_mapping(value, "Central-v2 scope claim")
    _require_exact_fields(
        claim,
        {"schema", "scope", "artifact_kind", "capability_declaration", "required_capabilities"},
        "Central-v2 scope claim",
    )
    if claim["schema"] != SCOPE_CLAIM_SCHEMA or claim["scope"] != SCOPE_NAME:
        raise ValueError("Central-v2 scope claim schema or scope is unsupported")
    kind = claim["artifact_kind"]
    if kind not in SUPPORTED_ARTIFACT_KINDS:
        raise ValueError(f"Unsupported central-v2 artifact kind: {kind!r}")
    if artifact_kind is not None and kind != artifact_kind:
        raise ValueError(f"Central-v2 scope claim is for {kind!r}, not {artifact_kind!r}")
    validate_capability_declaration_reference(claim["capability_declaration"])
    requested = frozenset(
        _require_string_list(claim["required_capabilities"], "Required capabilities")
    )
    _validate_capability_set(requested, f"{kind} scope claim")
    return claim


def negotiate_central_v2_capabilities(
    claim: Any,
    *,
    available_capabilities: Mapping[str, str],
    unavailable_capabilities: Mapping[str, str],
    artifact_kind: str | None = None,
) -> Mapping[str, str]:
    """Require accepted evidence for every central capability while preserving unavailable."""
    validated = validate_central_v2_scope_claim(claim, artifact_kind=artifact_kind)
    available = _require_mapping(available_capabilities, "Available capabilities")
    unavailable = _require_mapping(unavailable_capabilities, "Unavailable capabilities")
    for name, declarations in (("available", available), ("unavailable", unavailable)):
        if not all(
            isinstance(capability, str)
            and capability
            and isinstance(reason, str)
            and reason
            for capability, reason in declarations.items()
        ):
            raise ValueError(f"Central-v2 {name} capability declarations are malformed")
    if set(available) & set(unavailable):
        raise ValueError("Central-v2 available and unavailable capabilities overlap")
    unknown = sorted((set(available) | set(unavailable)) - KNOWN_CAPABILITIES)
    if unknown:
        raise ValueError(f"Unknown cohort-v2 capability in negotiation: {unknown!r}")
    required = set(validated["required_capabilities"])
    missing = sorted(required - set(available))
    if missing:
        reasons = {
            capability: unavailable.get(capability, "not demonstrated")
            for capability in missing
        }
        raise ValueError(f"Central-v2 required capabilities are unavailable: {reasons!r}")
    return MappingProxyType({capability: available[capability] for capability in sorted(required)})
