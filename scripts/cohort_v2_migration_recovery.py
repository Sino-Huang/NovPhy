"""Build the explicit local migration-recovery authority for cohort-v2."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Final, Mapping

from scripts.cohort_v2_macro_semantics import validate_capture_macro_derivation
from scripts.cohort_v2_micro_relations import (
    validate_capture_micro_relation_derivation,
)
from scripts.cohort_v2_partition import CohortV2PartitionExposureManifest
from scripts.cohort_v2_physical_violations import (
    validate_capture_physical_violation_derivation,
)
from scripts.cohort_v2_production_plans_v5 import (
    BUNDLE_IDENTITY as PLAN_BUNDLE_IDENTITY,
    COLLECTION_IDENTITY,
    DEFAULT_PLAN_ROOT,
    PARAMETER_IDENTITY,
    validate_plan_v5_evidence,
)
from scripts.cohort_v2_release import CENTRAL_LABELS, V5_CONTRACT
from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_json
from scripts.observation_trace import (
    load_observation_bytes,
    validate_observation_exposure_boundaries,
    validate_observation_trace,
)
from scripts.physics_capture_v2 import load_physics_capture_v2


ROOT: Final = Path(__file__).resolve().parents[1]
SCHEMA: Final = "cohort_v2_migration_recovery_manifest_v1"
IDENTITY: Final = "cohort-v2-migration-recovery-v1:issue-53-plan-v5-public"
DEFAULT_RELEASE_ROOT: Final = Path(
    "data/runtime_evidence/issue-53-mixed-termination-v5"
)
DEFAULT_AUTHORITY_ROOT: Final = Path(
    ".local-artifacts/migration-recovery-v1/issue-53-authority"
)
MANIFEST_NAME: Final = "cohort-v2-migration-recovery-manifest-v1.json"
DEFAULT_MANIFEST: Final = DEFAULT_AUTHORITY_ROOT / MANIFEST_NAME
PUBLIC_ROLES: Final = ("training", "calibration", "model_selection")
EXPECTED_ROLE_ROLLOUT_COUNTS: Final = {role: 6 for role in PUBLIC_ROLES}
EXPECTED_PUBLIC_TERMINATIONS: Final = {"level_fail": 4, "stable_entered": 14}
EXPECTED_PUBLIC_FRAMES: Final = 5_372


class CohortV2MigrationRecoveryError(ValueError):
    """The surviving migration recovery root is incomplete or inconsistent."""


def migration_recovery_manifest_path(value: Path) -> Path:
    path = Path(value)
    return path / MANIFEST_NAME if path.name != MANIFEST_NAME else path


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise CohortV2MigrationRecoveryError(f"cannot load {label}: {error}") from error
    if not isinstance(value, dict):
        raise CohortV2MigrationRecoveryError(f"{label} must be an object")
    return value


def _inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def _source_path(path: Path, repository_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(repository_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _release_envelopes(
    release_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    bundle = _load(release_root / "bundle-manifest.json", "public bundle")
    members = [
        item["path"]
        for item in _inventory(release_root)
        if item["path"] != "bundle-manifest.json"
    ]
    if (
        bundle.get("schema")
        != V5_CONTRACT.schema("issue_53_cohort_v2_release_bundle")
        or bundle.get("identity") != V5_CONTRACT.bundle_identity
        or bundle.get("publication_identity") != V5_CONTRACT.publication_identity
        or bundle.get("cohort_release_identity") != V5_CONTRACT.release_identity
        or bundle.get("authoritative_derivation_index_identity")
        != V5_CONTRACT.derivation_index_identity
        or bundle.get("sealed_final_evaluation_bundle_identity")
        != V5_CONTRACT.sealed_bundle_identity
        or bundle.get("artifacts") != members
        or bundle.get("passed") is not True
    ):
        raise CohortV2MigrationRecoveryError(
            "surviving public bundle identity, membership, or disposition is stale"
        )
    publication = _load(release_root / "cohort-v2-publication.json", "publication")
    release = _load(release_root / "cohort-v2-release.json", "release")
    derivations = _load(
        release_root / "authoritative-derivation-index.json", "derivation index"
    )
    if (
        publication.get("identity") != V5_CONTRACT.publication_identity
        or publication.get("cohort_release_identity") != V5_CONTRACT.release_identity
        or publication.get("disposition") != "complete"
        or release.get("identity") != V5_CONTRACT.release_identity
        or release.get("release_version") != 5
        or release.get("disposition") != "complete"
        or derivations.get("identity") != V5_CONTRACT.derivation_index_identity
        or derivations.get("source_cohort_release_identity") != release.get("identity")
        or set(derivations.get("accepted_labels", {})) != set(CENTRAL_LABELS)
    ):
        raise CohortV2MigrationRecoveryError("surviving public release identities are stale")
    return bundle, publication, release, derivations


def audit_surviving_public_release(
    release_root: Path,
    *,
    plan_root: Path,
) -> dict[str, Any]:
    """Revalidate the public v5 data itself, without consulting missing ancestors."""
    release_root = Path(release_root).resolve()
    plan_root = Path(plan_root).resolve()
    _bundle, _publication, release, derivation_index = _release_envelopes(
        release_root
    )
    collection = _load(release_root / "collection-plan.json", "released collection")
    parameters = _load(
        release_root / "production-parameter-plan.json", "released parameters"
    )
    partition_value = _load(
        release_root / "partition-exposure-manifest.json", "released partition"
    )
    if (
        collection != _load(plan_root / "collection-plan.json", "v5 collection")
        or parameters
        != _load(plan_root / "production-parameter-plan.json", "v5 parameters")
        or partition_value
        != _load(plan_root / "partition-exposure-manifest.json", "v5 partition")
        or collection.get("identity") != COLLECTION_IDENTITY
        or parameters.get("identity") != PARAMETER_IDENTITY
    ):
        raise CohortV2MigrationRecoveryError(
            "surviving release differs from the plan-v5 authority"
        )
    partition = CohortV2PartitionExposureManifest.from_dict(partition_value)
    partition_by_role = {entry.exposure_role: entry for entry in partition.entries}
    assignments = {
        item["exposure_role"]: item for item in collection.get("assignments", [])
    }
    if set(assignments) != {*PUBLIC_ROLES, "final_evaluation"}:
        raise CohortV2MigrationRecoveryError("plan-v5 roles are incomplete")
    if any(
        assignments[role]["scenario_lineage_identity"]
        != partition_by_role[role].scenario_lineage_identity
        for role in assignments
    ):
        raise CohortV2MigrationRecoveryError("plan-v5 role lineage isolation is stale")

    rollouts = release.get("primary_rollouts")
    derivation_refs = derivation_index.get("artifacts")
    if not isinstance(rollouts, list) or not isinstance(derivation_refs, list):
        raise CohortV2MigrationRecoveryError("release inventories are malformed")
    role_rollout_counts = Counter(item.get("exposure_role") for item in rollouts)
    attempt_ids = [item.get("attempt_id") for item in rollouts]
    if (
        len(rollouts) != 18
        or dict(role_rollout_counts) != EXPECTED_ROLE_ROLLOUT_COUNTS
        or len(set(attempt_ids)) != 18
        or any(role not in PUBLIC_ROLES for role in role_rollout_counts)
    ):
        raise CohortV2MigrationRecoveryError("public rollout membership or roles are stale")
    indexed_derivations: dict[str, dict[str, Mapping[str, Any]]] = {}
    for reference in derivation_refs:
        indexed_derivations.setdefault(str(reference.get("attempt_id")), {})[
            str(reference.get("kind"))
        ] = reference
    if (
        len(derivation_refs) != 54
        or set(indexed_derivations) != set(attempt_ids)
        or any(
            set(values) != {"micro", "macro", "physical-violations"}
            for values in indexed_derivations.values()
        )
    ):
        raise CohortV2MigrationRecoveryError("public derivation membership is stale")

    ledger = _load(
        release_root / "production-attempt-accounting.json", "attempt accounting"
    )
    ledger_entries = ledger.get("attempt_ledger")
    if not isinstance(ledger_entries, list):
        raise CohortV2MigrationRecoveryError("public attempt accounting is malformed")
    ledger_by_attempt = {
        item["attempt_id"]: item
        for item in ledger_entries
        if item.get("status") == "accepted"
    }
    if (
        set(ledger_by_attempt) != set(attempt_ids)
        or ledger.get("retry_count") != 0
        or Counter(item.get("status") for item in ledger_entries)
        != Counter({"accepted": 18, "sealed": 6})
    ):
        raise CohortV2MigrationRecoveryError(
            "public attempt accounting does not retain exactly 18 accepted rollouts"
        )

    role_frames: Counter[str] = Counter()
    terminal_counts: Counter[str] = Counter()
    observation_manifests = []
    canonical_denials = 0
    for reference in rollouts:
        attempt_id = reference["attempt_id"]
        role = reference["exposure_role"]
        ledger_entry = ledger_by_attempt[attempt_id]
        rollout_root = release_root / reference["path"]
        if (
            reference.get("files") != _inventory(rollout_root)
            or ledger_entry.get("exposure_role") != role
            or ledger_entry.get("scenario_lineage_identity")
            != partition_by_role[role].scenario_lineage_identity
            or ledger_entry.get("status") != "accepted"
        ):
            raise CohortV2MigrationRecoveryError(
                "public rollout inventory or role binding is stale"
            )
        capture = load_physics_capture_v2(rollout_root / "physics_capture_v2.json")
        if (
            capture.capture_id != reference.get("capture_id")
            or capture.source_bindings.get("rollout_id") != attempt_id
            or capture.source_bindings.get("scenario_lineage_id")
            != partition_by_role[role].scenario_lineage_identity
            or capture.record["terminal_evidence"]["reason"]
            != ledger_entry.get("terminal_reason")
        ):
            raise CohortV2MigrationRecoveryError("public capture binding is stale")
        frame_count = len(capture.record["fixed_step_samples"])
        role_frames[role] += frame_count
        terminal_counts[capture.record["terminal_evidence"]["reason"]] += 1

        references = indexed_derivations[attempt_id]
        source_reference = f"{reference['path']}/physics_capture_v2.json"
        loaded = {
            kind: _load(release_root / item["path"], f"{kind} derivation")
            for kind, item in references.items()
        }
        if any(
            loaded[kind].get("identity") != item.get("identity")
            for kind, item in references.items()
        ):
            raise CohortV2MigrationRecoveryError("derivation index identity is stale")
        validate_capture_micro_relation_derivation(
            loaded["micro"],
            capture,
            source_reference=source_reference,
            source_capture_bundle_identity=release["identity"],
        )
        validate_capture_macro_derivation(
            loaded["macro"],
            capture,
            source_reference=source_reference,
            source_capture_bundle_identity=release["identity"],
        )
        validate_capture_physical_violation_derivation(
            loaded["physical-violations"],
            capture,
            source_reference=source_reference,
            source_capture_bundle_identity=release["identity"],
        )

        observation_root = rollout_root / "observation-trace"
        observation = validate_observation_trace(observation_root)
        observation_manifests.append(observation)
        frames = observation.get("frame_records")
        bindings = observation.get("source_bindings", {})
        if (
            observation.get("exposure_role") != role
            or bindings.get("rollout_identity") != attempt_id
            or bindings.get("source_scenario_lineage_identity")
            != partition_by_role[role].scenario_lineage_identity
            or not isinstance(frames, list)
            or len(frames) != 1
            or frames[0]["fixed_step"]
            > capture.record["fixed_step_samples"][0]["fixed_step"]
        ):
            raise CohortV2MigrationRecoveryError("public observation alignment is stale")
        try:
            load_observation_bytes(
                observation_root,
                frame_record_identity=frames[0]["identity"],
                observation_role="canonical",
                workflow_kind=role,
                purpose="model_input",
            )
        except ValueError as error:
            if "canonical observation" not in str(error):
                raise
            canonical_denials += 1
        else:
            raise CohortV2MigrationRecoveryError(
                "canonical observation was accepted as model input"
            )

    validate_observation_exposure_boundaries(observation_manifests)
    total_frames = sum(role_frames.values())
    if (
        total_frames != EXPECTED_PUBLIC_FRAMES
        or dict(terminal_counts) != EXPECTED_PUBLIC_TERMINATIONS
        or canonical_denials != 18
    ):
        raise CohortV2MigrationRecoveryError(
            "public frame, termination, or canonical-input audit is stale"
        )
    return {
        "release_identity": release["identity"],
        "rollout_count": 18,
        "frame_count": total_frames,
        "role_rollout_counts": dict(sorted(role_rollout_counts.items())),
        "role_frame_counts": dict(sorted(role_frames.items())),
        "termination_counts": dict(sorted(terminal_counts.items())),
        "derivation_count": 54,
        "observation_trace_count": 18,
        "canonical_model_input_denials": canonical_denials,
        "retry_count": 0,
        "passed": True,
    }


def build_migration_recovery_manifest(
    *,
    repository_root: Path = ROOT,
    plan_root: Path = DEFAULT_PLAN_ROOT,
    release_root: Path = DEFAULT_RELEASE_ROOT,
) -> dict[str, Any]:
    repository_root = Path(repository_root).resolve()
    plan_root = Path(plan_root).resolve()
    release_root = Path(release_root).resolve()
    plan_validation = validate_plan_v5_evidence(
        plan_root,
        repository_root=repository_root,
        migration_recovery=True,
    )
    public_audit = audit_surviving_public_release(
        release_root, plan_root=plan_root
    )
    return {
        "schema": SCHEMA,
        "identity": IDENTITY,
        "mode": "migration_recovery",
        "scope": "local_closed_ticket_data_chain_recovery",
        "source_bindings": {
            "plan_root": _source_path(plan_root, repository_root),
            "plan_bundle_identity": PLAN_BUNDLE_IDENTITY,
            "collection_plan_identity": COLLECTION_IDENTITY,
            "production_parameter_plan_identity": PARAMETER_IDENTITY,
            "public_release_root": _source_path(release_root, repository_root),
            "public_release_identity": V5_CONTRACT.release_identity,
        },
        "plan_v5_validation": plan_validation,
        "public_release_audit": public_audit,
        "unavailable_sources": [
            {
                "component": f"issue-53-plan-v{version}",
                "kind": "historical_authority",
                "status": "unavailable",
                "inferred_from_summary": False,
            }
            for version in (2, 3, 4)
        ]
        + [
            {
                "component": "issue-15-confirmatory-v2-seed-4505",
                "kind": "sealed_data",
                "status": "unavailable",
                "inferred_from_summary": False,
            }
        ],
        "excluded_sources": [
            {
                "component": "issue-53-final-evaluation-seed-4504",
                "kind": "retired_sealed_data",
                "status": "excluded",
                "reason": "superseded final partition",
            },
            {
                "component": "issue-58-capacity-12-candidate",
                "kind": "superseded_determination",
                "status": "excluded",
                "reason": "superseded by issue-15 capacity-15 design",
            },
        ],
        "normal_validation_unchanged": True,
        "passed": True,
    }


def validate_migration_recovery_manifest(
    manifest_path: Path,
    *,
    repository_root: Path = ROOT,
    plan_root: Path = DEFAULT_PLAN_ROOT,
    release_root: Path = DEFAULT_RELEASE_ROOT,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    stored = _load(manifest_path, "migration-recovery manifest")
    expected = build_migration_recovery_manifest(
        repository_root=repository_root,
        plan_root=plan_root,
        release_root=release_root,
    )
    if stored != expected:
        raise CohortV2MigrationRecoveryError(
            "migration-recovery manifest differs from the surviving authorities"
        )
    canonical = (
        json.dumps(stored, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if manifest_path.read_bytes() != canonical:
        raise CohortV2MigrationRecoveryError(
            "migration-recovery manifest bytes are noncanonical"
        )
    return stored


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_AUTHORITY_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--migration-recovery", action="store_true")
    mode.add_argument("--validate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = args.repository_root.resolve()
    plan_root = (repository_root / args.plan_root).resolve()
    release_root = (repository_root / args.release_root).resolve()
    output_root = (repository_root / args.output_root).resolve()
    manifest_path = output_root / MANIFEST_NAME
    if args.validate:
        manifest = validate_migration_recovery_manifest(
            manifest_path,
            repository_root=repository_root,
            plan_root=plan_root,
            release_root=release_root,
        )
    else:
        if output_root.exists():
            raise CohortV2MigrationRecoveryError(
                "immutable migration-recovery authority already exists"
            )
        manifest = build_migration_recovery_manifest(
            repository_root=repository_root,
            plan_root=plan_root,
            release_root=release_root,
        )
        write_immutable_cohort_v2_json(manifest, manifest_path)
    audit = manifest["public_release_audit"]
    print(
        f"[migration-recovery] rollouts={audit['rollout_count']} "
        f"frames={audit['frame_count']} derivations={audit['derivation_count']} "
        f"manifest={manifest_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CohortV2MigrationRecoveryError, OSError, ValueError) as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(2) from error
