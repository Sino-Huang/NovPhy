"""Collect, seal, and read issue #15's replacement final-evaluation partition."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from scripts.capture_issue_53_evidence import (
    _accepted_player,
    _capture_attempt,
    _materialize_authority,
    _report_entry,
    _validate_authority,
)
from scripts.cohort_v2_partition import CohortV2PartitionExposureManifest
from scripts.cohort_v2_release import ReleaseContract, _write_derivations
from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_json
from scripts.final_evaluation_access import (
    FinalEvaluationWorkflowAccessManifest,
    audit_final_evaluation_workflow_access,
    authorize_final_evaluation_workflow_access,
)
from scripts.issue_15_amended_protocol import (
    COLLECTION_IDENTITY,
    DEFAULT_AUTHORITY_ROOT,
    DEFAULT_ROOT,
    FINAL_RELEASE_IDENTITY,
    FINAL_SEED,
    SEALED_BUNDLE_IDENTITY,
    load_frozen_bundle,
)
from scripts.physics_capture_v2 import load_physics_capture_v2
from scripts.smoke_physics_capture import start_display, terminate
from world_model.data.cohort_v2 import (
    CAPABILITY_DECLARATION_IDENTITY,
    CohortV2IngestionError,
    CohortV2ReleaseReader,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_ROOT = Path(
    ".local-artifacts/issue-15-confirmatory-v2-collection-run"
)
DEFAULT_SEALED_ROOT = Path(".local-artifacts/issue-15-confirmatory-v2-final")
DEFAULT_SUMMARY = Path(
    "data/runtime_evidence/issue-15/confirmatory-v2-collection-summary.json"
)
CAPABILITY_DECLARATION = ROOT / "docs/data_contracts/cohort_v2_capabilities_v1.json"
AUTHORIZATION_IDENTITY = "github-issue-authorization-v2:15:seed-4505-final"
DERIVATION_IDENTITY = "issue-15-confirmatory-derivations-v2:seed-4505"
CONTRACT_TEMPLATE = ReleaseContract(
    version=2,
    collection_identity=COLLECTION_IDENTITY,
    parameter_identity=(
        "cohort-v2-prospective-statistical-protocol-v2:issue-15-seed-4505"
    ),
    release_identity=FINAL_RELEASE_IDENTITY,
    publication_identity="issue-15-confirmatory-publication-v2:seed-4505",
    derivation_index_identity=DERIVATION_IDENTITY,
    sealed_bundle_identity=SEALED_BUNDLE_IDENTITY,
    bundle_identity="issue-15-confirmatory-bundle-v2:seed-4505",
    scenario_inventory_identity="issue-15-confirmatory-inventory-v2:seed-4505",
)


class Issue15FinalCollectionError(ValueError):
    """The prospective final collection is incomplete or cross-bound."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise Issue15FinalCollectionError(f"JSON artifact must be an object: {path}")
    return value


def _inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(Path(root).rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _contract(protocol: Mapping[str, Any]) -> ReleaseContract:
    from dataclasses import replace

    return replace(
        CONTRACT_TEMPLATE,
        parameter_identity=str(protocol["artifact_identity"]),
    )


def _log(message: str) -> None:
    print(f"[issue-15 final] {message}", flush=True)


def _frozen(plan_root: Path):
    plan, protocol = load_frozen_bundle(plan_root)
    collection = plan["confirmatory-plan.json"]
    partition = CohortV2PartitionExposureManifest.from_dict(
        plan["partition-exposure-manifest.json"]
    )
    pending = FinalEvaluationWorkflowAccessManifest.from_dict(
        plan["final-evaluation-workflow-access-manifest.json"]
    )
    if (
        protocol["status"] != "frozen_before_new_final_collection"
        or collection["new_final_seed"] != FINAL_SEED
        or len(collection["attempt_ids"]) != 6
        or pending.authorization_state != "pending"
    ):
        raise Issue15FinalCollectionError("issue-15 amendment is not ready for access")
    audit_final_evaluation_workflow_access(
        partition, pending, observed_accesses=[]
    )
    return plan, protocol, collection, partition, pending


def _assignment(collection: Mapping[str, Any]) -> Mapping[str, Any]:
    assignments = collection.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != 1:
        raise Issue15FinalCollectionError("collection must have one final assignment")
    assignment = assignments[0]
    if assignment.get("exposure_role") != "final_evaluation":
        raise Issue15FinalCollectionError("collection assignment is not final-only")
    return assignment


def _authority(output_root: Path, assignment: Mapping[str, Any]):
    from dataclasses import replace
    from scripts.build_issue_45_evidence import ROLES

    authority = _materialize_authority(
        replace(ROLES[3], seed=FINAL_SEED), Path(output_root)
    )
    _validate_authority(authority, assignment)
    return authority


def _observed_access(
    manifest: FinalEvaluationWorkflowAccessManifest,
    assignment: Mapping[str, Any],
    accessed_at: str,
) -> dict[str, Any]:
    return {
        "workflow_identity": manifest.workflow_identity,
        "operator_identity": manifest.operator_identity,
        "artifact_identity": assignment["scenario_manifest_identity"],
        "source_scenario_lineage_identities": [
            assignment["scenario_lineage_identity"]
        ],
        "accessed_at": accessed_at,
        "authorization_identity": manifest.authorization_identity,
        "consumer_exposure_role": "final_evaluation",
    }


def dry_run(
    *,
    plan_root: Path = DEFAULT_ROOT,
    authority_root: Path = DEFAULT_AUTHORITY_ROOT,
) -> dict[str, Any]:
    _log("dry-run: validating the committed protocol and pending workflow")
    plan, protocol, collection, _partition, _pending = _frozen(plan_root)
    assignment = _assignment(collection)
    _log("dry-run: rebuilding the seed-4505 scenario without collecting outcomes")
    with tempfile.TemporaryDirectory(prefix="issue-15-final-dry-") as temporary:
        rebuilt = _authority(Path(temporary), assignment)
        if (
            rebuilt["manifest_path"].read_bytes()
            != (Path(authority_root) / "final-evaluation.json").read_bytes()
            or rebuilt["xml_path"].read_bytes()
            != (Path(authority_root) / "final-evaluation.xml").read_bytes()
        ):
            raise Issue15FinalCollectionError("sealed scenario authority differs")
    _log("dry-run: verifying the accepted physics-v2 player")
    player = _accepted_player()
    return {
        "schema": "issue_15_confirmatory_final_collection_dry_run_v2",
        "protocol_identity": protocol["artifact_identity"],
        "partition_bound": bool(plan["partition-exposure-manifest.json"]["identity"]),
        "planned_rollouts": len(collection["attempt_ids"]),
        "final_seed": FINAL_SEED,
        "workflow_state": "pending",
        "final_outcomes_accessed": False,
        "files_written": False,
        "player_source_snapshot_commit": player["source_snapshot_commit"],
        "actual_command": (
            "python -u -m scripts.issue_15_final_collection "
            f"--authorization-identity {AUTHORIZATION_IDENTITY} "
            "--implementation-commit <committed-revision>"
        ),
        "passed": True,
    }


def _checkpoint(path: Path, value: Mapping[str, Any]) -> None:
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _seal(
    *,
    sealed_root: Path,
    runtime_root: Path,
    plan_root: Path,
    protocol: Mapping[str, Any],
    partition: CohortV2PartitionExposureManifest,
    collection: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]],
    authorized: FinalEvaluationWorkflowAccessManifest,
    access_record: Mapping[str, Any],
    access_audit: Mapping[str, Any],
    collection_implementation_commit: str,
) -> dict[str, Any]:
    destination = Path(sealed_root)
    if destination.exists():
        raise Issue15FinalCollectionError("immutable sealed output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".issue-15-final-", dir=destination.parent
    ) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        primary = bundle / "primary-rollouts"
        derivation_root = bundle / "derivations"
        primary.mkdir()
        derivation_root.mkdir()
        derivations = []
        for entry in ledger:
            attempt_id = entry["attempt_id"]
            source = Path(str(entry["artifact_path"]))
            rollout = primary / attempt_id
            shutil.copytree(source, rollout)
            capture = load_physics_capture_v2(rollout / "physics_capture_v2.json")
            for reference in _write_derivations(
                derivation_root / attempt_id,
                capture,
                source_reference=(
                    f"primary-rollouts/{attempt_id}/physics_capture_v2.json"
                ),
                release_identity=FINAL_RELEASE_IDENTITY,
            ):
                derivations.append({
                    "attempt_id": attempt_id,
                    "exposure_role": "final_evaluation",
                    **reference,
                    "path": f"derivations/{attempt_id}/{reference['path']}",
                })
        authority = bundle / "scenario-authority"
        authority.mkdir()
        shutil.copyfile(
            runtime_root / "authorities/manifests/final-evaluation.json",
            authority / "final-evaluation.json",
        )
        shutil.copyfile(
            runtime_root / "authorities/xml/final-evaluation.xml",
            authority / "final-evaluation.xml",
        )
        shutil.copyfile(
            plan_root / "confirmatory-plan.json", bundle / "confirmatory-plan.json"
        )
        shutil.copyfile(
            plan_root / "partition-exposure-manifest.json",
            bundle / "partition-exposure-manifest.json",
        )
        shutil.copyfile(
            plan_root / "cohort-v2-prospective-statistical-protocol-v2.json",
            bundle / "cohort-v2-prospective-statistical-protocol-v2.json",
        )
        write_immutable_cohort_v2_json(
            authorized.to_dict(), bundle / "authorized-final-access-manifest.json"
        )
        write_immutable_cohort_v2_json(
            dict(access_record), bundle / "observed-final-access.json"
        )
        write_immutable_cohort_v2_json(
            dict(access_audit), bundle / "final-access-audit.json"
        )
        write_immutable_cohort_v2_json(
            {
                "schema": "issue_15_confirmatory_attempt_ledger_v2",
                "collection_identity": COLLECTION_IDENTITY,
                "attempt_ledger": [dict(item) for item in ledger],
                "replacement_attempts": 0,
                "passed": True,
            },
            bundle / "attempt-ledger.json",
        )
        write_immutable_cohort_v2_json(
            {
                "schema": "issue_15_confirmatory_derivation_index_v2",
                "identity": DERIVATION_IDENTITY,
                "source_release_identity": FINAL_RELEASE_IDENTITY,
                "artifacts": derivations,
            },
            bundle / "derivation-index.json",
        )
        write_immutable_cohort_v2_json(
            {
                "schema": "issue_15_confirmatory_sealed_bundle_v2",
                "identity": SEALED_BUNDLE_IDENTITY,
                "release_identity": FINAL_RELEASE_IDENTITY,
                "protocol_identity": protocol["artifact_identity"],
                "partition_identity": partition.identity,
                "collection_identity": collection["identity"],
                "authorized_workflow_identity": authorized.identity,
                "collection_implementation_commit": collection_implementation_commit,
                "ordinary_workflow_access": False,
                "attempt_ids": [item["attempt_id"] for item in ledger],
                "artifacts": [item["path"] for item in _inventory(bundle)],
                "passed": True,
            },
            bundle / "sealed-bundle-manifest.json",
        )
        os.replace(bundle, destination)
    return validate_sealed(destination, plan_root=plan_root)


def validate_sealed(
    sealed_root: Path = DEFAULT_SEALED_ROOT,
    *,
    plan_root: Path = DEFAULT_ROOT,
) -> dict[str, Any]:
    plan, protocol, collection, partition, _pending = _frozen(plan_root)
    root = Path(sealed_root)
    manifest = _load(root / "sealed-bundle-manifest.json")
    members = [
        item["path"] for item in _inventory(root)
        if item["path"] != "sealed-bundle-manifest.json"
    ]
    if (
        manifest.get("identity") != SEALED_BUNDLE_IDENTITY
        or manifest.get("release_identity") != FINAL_RELEASE_IDENTITY
        or manifest.get("protocol_identity") != protocol["artifact_identity"]
        or manifest.get("partition_identity") != partition.identity
        or manifest.get("collection_identity") != collection["identity"]
        or manifest.get("ordinary_workflow_access") is not False
        or not isinstance(manifest.get("collection_implementation_commit"), str)
        or manifest.get("attempt_ids") != collection["attempt_ids"]
        or manifest.get("artifacts") != members
        or manifest.get("passed") is not True
    ):
        raise Issue15FinalCollectionError("sealed final bundle is stale")
    ledger = _load(root / "attempt-ledger.json")["attempt_ledger"]
    if (
        len(ledger) != 6
        or [item.get("attempt_id") for item in ledger] != collection["attempt_ids"]
        or any(item.get("status") != "accepted" for item in ledger)
    ):
        raise Issue15FinalCollectionError("sealed final ledger is incomplete")
    frame_count = 0
    for entry in ledger:
        attempt_id = entry["attempt_id"]
        capture = load_physics_capture_v2(
            root / "primary-rollouts" / attempt_id / "physics_capture_v2.json"
        )
        if (
            capture.source_bindings["rollout_id"] != attempt_id
            or capture.record["terminal_evidence"]["reason"]
            != entry["terminal_reason"]
        ):
            raise Issue15FinalCollectionError("sealed rollout binding is stale")
        frame_count += len(capture.record["fixed_step_samples"])
    authorized = FinalEvaluationWorkflowAccessManifest.from_dict(
        _load(root / "authorized-final-access-manifest.json")
    )
    observed = _load(root / "observed-final-access.json")
    access_audit = audit_final_evaluation_workflow_access(
        partition, authorized, observed_accesses=[observed]
    )
    if (
        Counter(item["terminal_reason"] for item in ledger)
        != Counter({"stable_entered": 6})
        or frame_count != 1_610
        or authorized.authorization_identity != AUTHORIZATION_IDENTITY
        or access_audit != _load(root / "final-access-audit.json")
        or access_audit.get("observed_access_count") != 1
    ):
        raise Issue15FinalCollectionError(
            "sealed final termination, frame, or access audit is stale"
        )
    return {
        "schema": "issue_15_confirmatory_final_collection_validation_v2",
        "sealed_bundle_identity": SEALED_BUNDLE_IDENTITY,
        "protocol_identity": protocol["artifact_identity"],
        "partition_bound": bool(plan["partition-exposure-manifest.json"]["identity"]),
        "accepted_rollouts": 6,
        "frame_count": frame_count,
        "termination_counts": {"stable_entered": 6},
        "observed_access_count": 1,
        "passed": True,
    }


def collect(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    sealed_root: Path = DEFAULT_SEALED_ROOT,
    summary_path: Path = DEFAULT_SUMMARY,
    plan_root: Path = DEFAULT_ROOT,
    authorization_identity: str,
    implementation_commit: str,
) -> dict[str, Any]:
    if not authorization_identity:
        raise Issue15FinalCollectionError("collection requires authorization identity")
    if not implementation_commit:
        raise Issue15FinalCollectionError("collection requires implementation commit")
    runtime = Path(runtime_root)
    summary = Path(summary_path)
    if runtime.exists() or Path(sealed_root).exists() or summary.exists():
        raise Issue15FinalCollectionError("immutable collection output already exists")
    _plan, protocol, collection, partition, pending = _frozen(plan_root)
    contract = _contract(protocol)
    assignment = _assignment(collection)
    runtime.mkdir(parents=True)
    _log("authorizing the committed seed-4505 workflow")
    authorized_at = _now()
    authorized = authorize_final_evaluation_workflow_access(
        pending,
        authorization_identity=authorization_identity,
        authorized_at=authorized_at,
    )
    access_record = _observed_access(authorized, assignment, authorized_at)
    access_audit = audit_final_evaluation_workflow_access(
        partition, authorized, observed_accesses=[access_record]
    )
    _log("verifying the accepted physics-v2 player")
    player = _accepted_player()
    write_immutable_cohort_v2_json(player, runtime / "player-provenance.json")
    _log("materializing and source-checking the authorized final authority")
    authority = _authority(runtime / "authorities", assignment)
    interventions = {item["id"]: item for item in collection["interventions"]}
    ledger: list[dict[str, Any]] = []
    display_process = None
    old_display = os.environ.get("DISPLAY")
    old_stride = os.environ.get("NOVPHY_PHYSICS_CAPTURE_V2_STRIDE")
    try:
        display, display_process = start_display(runtime / "display.log")
        os.environ["DISPLAY"] = display
        os.environ["NOVPHY_PHYSICS_CAPTURE_V2_STRIDE"] = "1"
        for index, (attempt_id, intervention_id) in enumerate(
            zip(collection["attempt_ids"], assignment["intervention_ids"], strict=True),
            start=1,
        ):
            intervention = interventions[intervention_id]
            _log(f"rollout {index}/6: {intervention_id}")
            result = _capture_attempt(
                runtime,
                runtime / "production",
                authority,
                assignment,
                intervention,
                attempt_id,
                intervention["interface_action"],
                contract,
            )
            entry = _report_entry(
                assignment, intervention, attempt_id, result, contract
            )
            ledger.append(entry)
            _checkpoint(runtime / "attempt-ledger.checkpoint.json", {
                "schema": "issue_15_confirmatory_attempt_ledger_checkpoint_v2",
                "attempt_ledger": ledger,
            })
            _log(
                f"rollout {index}/6: {entry['status']} "
                f"termination={entry['terminal_reason']} "
                f"frames={entry['terminal_span_fixed_steps']}"
            )
    finally:
        terminate(display_process)
        if old_display is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = old_display
        if old_stride is None:
            os.environ.pop("NOVPHY_PHYSICS_CAPTURE_V2_STRIDE", None)
        else:
            os.environ["NOVPHY_PHYSICS_CAPTURE_V2_STRIDE"] = old_stride
    if len(ledger) != 6 or any(item["status"] != "accepted" for item in ledger):
        raise Issue15FinalCollectionError(
            "the fixed collection has a failed attempt; no replacement is permitted"
        )
    _log("sealing six accepted rollouts and authoritative derivations")
    validation = _seal(
        sealed_root=sealed_root,
        runtime_root=runtime,
        plan_root=Path(plan_root),
        protocol=protocol,
        partition=partition,
        collection=collection,
        ledger=ledger,
        authorized=authorized,
        access_record=access_record,
        access_audit=access_audit,
        collection_implementation_commit=implementation_commit,
    )
    reader = Issue15ConfirmatoryV2Reader(sealed_root, plan_root=plan_root)
    if len(reader.rollouts) != 6:
        raise Issue15FinalCollectionError("sealed reader did not expose six rollouts")
    compact_access_audit = {
        key: access_audit[key]
        for key in (
            "workflow_identity",
            "authorization_state",
            "authorization_identity",
            "observed_access_count",
            "passed",
        )
    }
    result = {
        **validation,
        "schema": "issue_15_confirmatory_final_collection_summary_v2",
        "protocol_implementation_commit": protocol["implementation_commit"],
        "collection_implementation_commit": implementation_commit,
        "authorization_identity": authorization_identity,
        "attempt_ids": collection["attempt_ids"],
        "termination_counts": dict(sorted(
            Counter(item["terminal_reason"] for item in ledger).items()
        )),
        "replacement_attempts": 0,
        "old_seed_4504_reused": False,
        "access_audit": compact_access_audit,
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    write_immutable_cohort_v2_json(result, summary)
    return result


class Issue15ConfirmatoryV2Reader(CohortV2ReleaseReader):
    """Read the authorized replacement final bundle without terminal filtering."""

    def __init__(
        self,
        sealed_root: Path,
        *,
        plan_root: Path = DEFAULT_ROOT,
        capability_declaration_path: Path = CAPABILITY_DECLARATION,
    ) -> None:
        self._root = Path(sealed_root).resolve()
        self._production_plan_root = Path(plan_root).resolve()
        self._workflow_kind = "final_evaluation"
        self._enforce_expected_termination = False
        self._observation_references: dict[str, tuple[Path, str]] = {}
        try:
            self._validate_capability_declaration(Path(capability_declaration_path))
            validate_sealed(self._root, plan_root=self._production_plan_root)
            _plan, _protocol, collection, partition, _pending = _frozen(
                self._production_plan_root
            )
            authorized = FinalEvaluationWorkflowAccessManifest.from_dict(
                _load(self._root / "authorized-final-access-manifest.json")
            )
            observed = _load(self._root / "observed-final-access.json")
            self.access_audit = audit_final_evaluation_workflow_access(
                partition, authorized, observed_accesses=[observed]
            )
            ledger = _load(self._root / "attempt-ledger.json")
            derivation_index = _load(self._root / "derivation-index.json")
            rollout_references = []
            for attempt_id in collection["attempt_ids"]:
                rollout_root = self._root / "primary-rollouts" / attempt_id
                capture = load_physics_capture_v2(
                    rollout_root / "physics_capture_v2.json"
                )
                rollout_references.append({
                    "attempt_id": attempt_id,
                    "exposure_role": "final_evaluation",
                    "capture_id": capture.capture_id,
                    "path": f"primary-rollouts/{attempt_id}",
                    "files": _inventory(rollout_root),
                })
            self.release_identity = FINAL_RELEASE_IDENTITY
            self.capability_declaration_identity = CAPABILITY_DECLARATION_IDENTITY
            self.derivation_identity = derivation_index["identity"]
            self.partition_identity = partition.identity
            self.sealed_bundle_identity = SEALED_BUNDLE_IDENTITY
            self.rollouts = self._read_role(
                {
                    "identity": FINAL_RELEASE_IDENTITY,
                    "primary_rollouts": rollout_references,
                },
                derivation_index,
                collection,
                partition,
                ledger,
            )
        except CohortV2IngestionError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise CohortV2IngestionError(
                f"Issue-15 final ingestion rejected: {error}"
            ) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--authority-root", type=Path, default=DEFAULT_AUTHORITY_ROOT)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--sealed-root", type=Path, default=DEFAULT_SEALED_ROOT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--authorization-identity")
    parser.add_argument("--implementation-commit")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--validate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dry_run:
        result = dry_run(plan_root=args.plan_root, authority_root=args.authority_root)
    elif args.validate:
        result = validate_sealed(args.sealed_root, plan_root=args.plan_root)
    else:
        result = collect(
            runtime_root=args.runtime_root,
            sealed_root=args.sealed_root,
            summary_path=args.summary,
            plan_root=args.plan_root,
            authorization_identity=args.authorization_identity or "",
            implementation_commit=args.implementation_commit or "",
        )
    displayed = {
        key: result[key]
        for key in (
            "schema",
            "protocol_identity",
            "sealed_bundle_identity",
            "planned_rollouts",
            "accepted_rollouts",
            "termination_counts",
            "final_outcomes_accessed",
            "files_written",
            "actual_command",
            "passed",
        )
        if key in result
    }
    print(json.dumps(displayed, allow_nan=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Issue15FinalCollectionError, CohortV2IngestionError, OSError) as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(2) from error
