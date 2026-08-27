"""Recollect cohort-v2 rollouts with agent RGB aligned to every retained frame."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from scripts.capture_issue_53_evidence import (
    DEFAULT_PLAN_ROOT,
    _assignments,
    _capture_attempt,
    _load_plan,
    _prepare_non_final_authorities,
    _report_entry,
)
from scripts.cohort_v2_release import (
    _write_derivations,
    production_attempt_identity,
)
from scripts.cohort_v2_macro_semantics import validate_capture_macro_derivation
from scripts.cohort_v2_micro_relations import (
    validate_capture_micro_relation_derivation,
)
from scripts.cohort_v2_physical_violations import (
    validate_capture_physical_violation_derivation,
)
from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_json
from scripts.final_evaluation_access import (
    audit_final_evaluation_workflow_access,
    authorize_final_evaluation_workflow_access,
)
from scripts.issue_15_final_collection import (
    DEFAULT_ROOT as ISSUE_15_PLAN_ROOT,
    _assignment as final_assignment,
    _authority as final_authority,
    _contract as final_contract,
    _frozen as final_frozen,
    _observed_access,
)
from scripts.observation_trace import (
    validate_observation_exposure_boundaries,
    validate_observation_trace,
)
from scripts.physics_capture_v2 import load_physics_capture_v2
from scripts.smoke_physics_capture import start_display, terminate
from scripts.verify_physics_player import verify_physics_player_archive
from world_model.training.manifest import git_revision


ROOT = Path(__file__).resolve().parents[1]
STAGE_ROOT = ROOT / "sciencebirdsgames/aligned-observation-v1"
DEFAULT_RUNTIME_ROOT = ROOT / ".local-artifacts/issue-59-aligned-collection-run"
DEFAULT_OUTPUT = ROOT / ".local-artifacts/issue-59-aligned-observation-release"
DEFAULT_SUMMARY = (
    ROOT / "data/runtime_evidence/issue-59/aligned-observation-release-summary.json"
)
AUTHORIZATION_IDENTITY = "github-issue-authorization-v1:59:aligned-final-recollection"
RELEASE_IDENTITY = "cohort-v2-aligned-observation-release-v1:issue-59"
SCHEMA = "cohort_v2_aligned_observation_release_v1"
PARTITION_SCHEMA = "cohort_v2_aligned_observation_partition_v1"
SUMMARY_SCHEMA = "cohort_v2_aligned_observation_release_summary_v1"
ROLE_ORDER = ("training", "calibration", "model_selection", "final_evaluation")


class Issue59CollectionError(ValueError):
    """The aligned recollection plan, output, or provenance is invalid."""


def _log(message: str) -> None:
    print(f"[issue-59] {message}", flush=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise Issue59CollectionError(f"cannot load issue-59 artifact {path}") from error
    if not isinstance(value, dict):
        raise Issue59CollectionError(f"issue-59 artifact is not an object: {path}")
    return value


def _source_plan(
    public_plan_root: Path, issue_15_plan_root: Path
) -> tuple[Any, Mapping[str, Mapping[str, Any]], Any, Mapping[str, Any], Any, Any]:
    public = _load_plan(public_plan_root)
    assignments = _assignments(public.collection)
    if set(assignments) != set(ROLE_ORDER):
        raise Issue59CollectionError("public source plan does not contain four roles")
    final_plan, _protocol, collection, partition, pending = final_frozen(
        issue_15_plan_root
    )
    assignment = final_assignment(collection)
    if len(collection["attempt_ids"]) != 6:
        raise Issue59CollectionError("final source plan does not contain six attempts")
    return public, assignments, final_plan, collection, assignment, (partition, pending)


def _planned_attempts(
    public: Any,
    assignments: Mapping[str, Mapping[str, Any]],
    final_collection: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    public_ids = {
        role: tuple(
            production_attempt_identity(
                role, intervention_id, public.contract.collection_identity
            )
            for intervention_id in assignments[role]["intervention_ids"]
        )
        for role in ROLE_ORDER[:3]
    }
    result = {
        **public_ids,
        "final_evaluation": tuple(final_collection["attempt_ids"]),
    }
    if any(len(values) != 6 or len(set(values)) != 6 for values in result.values()):
        raise Issue59CollectionError("each issue-59 role must contain six unique attempts")
    if len({item for values in result.values() for item in values}) != 24:
        raise Issue59CollectionError("issue-59 attempt identities cross exposure roles")
    return result


def _player(implementation_commit: str, stage_root: Path) -> dict[str, Any]:
    value = verify_physics_player_archive(stage_root, physics_v2=True)
    if value["source_snapshot_commit"] != implementation_commit:
        raise Issue59CollectionError(
            "aligned-observation player was not built from the implementation commit"
        )
    return value


def dry_run(
    *,
    implementation_commit: str,
    public_plan_root: Path = DEFAULT_PLAN_ROOT,
    issue_15_plan_root: Path = ISSUE_15_PLAN_ROOT,
    stage_root: Path = STAGE_ROOT,
) -> tuple[dict[str, Any], int]:
    _log("dry-run 1/3: validating frozen public and replacement-final plans")
    public, assignments, _final_plan, final_collection, _assignment, final_access = (
        _source_plan(public_plan_root, issue_15_plan_root)
    )
    attempts = _planned_attempts(public, assignments, final_collection)
    _log("dry-run 2/3: verifying the source-bound aligned-observation player")
    player = _player(implementation_commit, stage_root)
    partition, pending = final_access
    audit_final_evaluation_workflow_access(partition, pending, observed_accesses=[])
    _log("dry-run 3/3: 24 slots wired; final outcomes remain unopened")
    return {
        "schema": "issue_59_aligned_observation_collection_dry_run_v1",
        "implementation_commit": implementation_commit,
        "player_source_commit": player["source_snapshot_commit"],
        "planned_role_counts": {role: len(attempts[role]) for role in ROLE_ORDER},
        "planned_rollouts": sum(len(values) for values in attempts.values()),
        "observation_configuration": "agent_rgb8_native_v1",
        "fixed_step_capture_stride": 1,
        "final_authorization_state": pending.authorization_state,
        "final_outcomes_accessed": False,
        "files_written": False,
        "actual_command": (
            "python -u -m scripts.issue_59_aligned_observation_collection "
            f"--implementation-commit {implementation_commit} "
            f"--authorization-identity {AUTHORIZATION_IDENTITY}"
        ),
        "passed": True,
    }


def _collect_one(
    *,
    index: int,
    total: int,
    runtime_root: Path,
    authority: Mapping[str, Any],
    assignment: Mapping[str, Any],
    intervention: Mapping[str, Any],
    attempt_id: str,
    contract: Any,
    stage_root: Path,
) -> tuple[dict[str, Any], Path]:
    role = str(assignment["exposure_role"])
    _log(
        f"collect {index}/{total}: role={role} "
        f"stratum={intervention['intended_coverage_stratum']}"
    )
    result = _capture_attempt(
        runtime_root,
        runtime_root / "production",
        authority,
        assignment,
        intervention,
        attempt_id,
        intervention["interface_action"],
        contract,
        aligned_observation_capture=True,
        stage_root=stage_root,
    )
    entry = _report_entry(assignment, intervention, attempt_id, result, contract)
    if (
        entry["status"] != "accepted"
        or intervention["intended_coverage_stratum"]
        not in entry["realized_coverage_strata"]
    ):
        raise Issue59CollectionError(
            f"aligned recollection failed its frozen slot: {attempt_id}"
        )
    artifact = Path(str(result["artifact_path"])).resolve()
    capture = load_physics_capture_v2(artifact / "physics_capture_v2.json")
    observation = validate_observation_trace(artifact / "observation-trace")
    capture_steps = tuple(item["fixed_step"] for item in capture.record["frame_records"])
    observation_steps = tuple(item["fixed_step"] for item in observation["frame_records"])
    if capture_steps != observation_steps:
        raise Issue59CollectionError(
            f"aligned recollection has incomplete observation coverage: {attempt_id}"
        )
    _log(
        f"collect {index}/{total}: accepted frames={len(capture_steps)} "
        f"terminal={entry['terminal_reason']}"
    )
    return entry, artifact


def _publish(
    *,
    output: Path,
    records: list[tuple[dict[str, Any], Path]],
    implementation_commit: str,
    player: Mapping[str, Any],
    public: Any,
    final_protocol: Mapping[str, Any],
    access_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if output.exists():
        raise Issue59CollectionError("immutable issue-59 release already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".issue-59-", dir=output.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        def publish_partition(
            name: str,
            included_roles: tuple[str, ...],
        ) -> dict[str, Any]:
            partition_root = bundle / name
            partition_root.mkdir()
            manifest_records = []
            for entry, source in records:
                role = entry["exposure_role"]
                if role not in included_roles:
                    continue
                attempt_id = entry["attempt_id"]
                relative = Path("rollouts") / role / attempt_id
                destination = partition_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, destination)
                capture = load_physics_capture_v2(
                    destination / "physics_capture_v2.json"
                )
                observation = validate_observation_trace(
                    destination / "observation-trace"
                )
                derivation_relative = Path("derivations") / role / attempt_id
                derivations = _write_derivations(
                    partition_root / derivation_relative,
                    capture,
                    source_reference=(
                        relative / "physics_capture_v2.json"
                    ).as_posix(),
                    release_identity=RELEASE_IDENTITY,
                )
                manifest_records.append({
                    "attempt_id": attempt_id,
                    "exposure_role": role,
                    "coverage_stratum": entry["intended_coverage_stratum"],
                    "scenario_lineage_identity": entry["scenario_lineage_identity"],
                    "capture_id": capture.capture_id,
                    "terminal_reason": entry["terminal_reason"],
                    "frame_count": len(capture.record["frame_records"]),
                    "rollout_path": relative.as_posix(),
                    "observation_manifest_identity": observation["identity"],
                    "derivations": [
                        {
                            **item,
                            "path": (derivation_relative / item["path"]).as_posix(),
                        }
                        for item in derivations
                    ],
                })
            manifest_records.sort(
                key=lambda item: (
                    ROLE_ORDER.index(item["exposure_role"]), item["attempt_id"]
                )
            )
            partition = {
                "schema": PARTITION_SCHEMA,
                "identity": f"{RELEASE_IDENTITY}:{name}",
                "release_identity": RELEASE_IDENTITY,
                "included_roles": list(included_roles),
                "records": manifest_records,
                "role_counts": dict(sorted(Counter(
                    item["exposure_role"] for item in manifest_records
                ).items())),
                "passed": True,
            }
            write_immutable_cohort_v2_json(
                partition, partition_root / "manifest.json"
            )
            return partition

        public_partition = publish_partition("public", ROLE_ORDER[:3])
        final_partition = publish_partition("sealed-final", ROLE_ORDER[3:])
        role_counts = {
            **public_partition["role_counts"],
            **final_partition["role_counts"],
        }
        frame_count = sum(
            item["frame_count"]
            for partition in (public_partition, final_partition)
            for item in partition["records"]
        )
        manifest = {
            "schema": SCHEMA,
            "identity": RELEASE_IDENTITY,
            "implementation_commit": implementation_commit,
            "player_provenance": dict(player),
            "source_bindings": {
                "public_collection_plan_identity": public.contract.collection_identity,
                "public_release_identity": public.contract.release_identity,
                "replacement_final_protocol_identity": final_protocol["artifact_identity"],
                "issue_15_disposition_unchanged": True,
                "issue_16_disposition_unchanged": True,
            },
            "access_audit": dict(access_audit),
            "partitions": {
                "public": {
                    "path": "public",
                    "identity": public_partition["identity"],
                },
                "sealed_final": {
                    "path": "sealed-final",
                    "identity": final_partition["identity"],
                    "ordinary_workflow_access": False,
                },
            },
            "role_counts": role_counts,
            "rollout_count": sum(role_counts.values()),
            "passed": True,
        }
        write_immutable_cohort_v2_json(manifest, bundle / "manifest.json")
        os.replace(bundle, output)
    return manifest, frame_count


def collect(
    *,
    implementation_commit: str,
    authorization_identity: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    output: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    public_plan_root: Path = DEFAULT_PLAN_ROOT,
    issue_15_plan_root: Path = ISSUE_15_PLAN_ROOT,
    stage_root: Path = STAGE_ROOT,
) -> dict[str, Any]:
    runtime_root = Path(runtime_root).resolve()
    output = Path(output).resolve()
    summary_path = Path(summary_path).resolve()
    if runtime_root.exists() or output.exists() or summary_path.exists():
        raise Issue59CollectionError("immutable issue-59 output already exists")
    public, assignments, _final_plan, final_collection, final_assign, final_access = (
        _source_plan(public_plan_root, issue_15_plan_root)
    )
    _planned_attempts(public, assignments, final_collection)
    player = _player(implementation_commit, stage_root)
    runtime_root.mkdir(parents=True)
    public_authorities = _prepare_non_final_authorities(
        runtime_root / "authorities/public", assignments
    )
    final_auth = final_authority(
        runtime_root / "authorities/final", final_assign
    )
    partition, pending = final_access
    authorized_at = _now()
    authorized = authorize_final_evaluation_workflow_access(
        pending,
        authorization_identity=authorization_identity,
        authorized_at=authorized_at,
    )
    observed = _observed_access(authorized, final_assign, authorized_at)
    access_audit = audit_final_evaluation_workflow_access(
        partition, authorized, observed_accesses=[observed]
    )

    records: list[tuple[dict[str, Any], Path]] = []
    display_process = None
    prior_display = os.environ.get("DISPLAY")
    prior_stride = os.environ.get("NOVPHY_PHYSICS_CAPTURE_V2_STRIDE")
    try:
        display, display_process = start_display(runtime_root / "display.log")
        os.environ["DISPLAY"] = display
        os.environ["NOVPHY_PHYSICS_CAPTURE_V2_STRIDE"] = "1"
        interventions = {item["id"]: item for item in public.collection["interventions"]}
        index = 0
        for role in ROLE_ORDER[:3]:
            assignment = assignments[role]
            for intervention_id in assignment["intervention_ids"]:
                index += 1
                intervention = interventions[intervention_id]
                attempt_id = production_attempt_identity(
                    role, intervention_id, public.contract.collection_identity
                )
                records.append(_collect_one(
                    index=index,
                    total=24,
                    runtime_root=runtime_root,
                    authority=public_authorities[role],
                    assignment=assignment,
                    intervention=intervention,
                    attempt_id=attempt_id,
                    contract=public.contract,
                    stage_root=stage_root,
                ))
        final_interventions = {
            item["id"]: item for item in final_collection["interventions"]
        }
        contract = final_contract(final_frozen(issue_15_plan_root)[1])
        for attempt_id, intervention_id in zip(
            final_collection["attempt_ids"],
            final_assign["intervention_ids"],
            strict=True,
        ):
            index += 1
            records.append(_collect_one(
                index=index,
                total=24,
                runtime_root=runtime_root,
                authority=final_auth,
                assignment=final_assign,
                intervention=final_interventions[intervention_id],
                attempt_id=attempt_id,
                contract=contract,
                stage_root=stage_root,
            ))
    finally:
        terminate(display_process)
        if prior_display is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = prior_display
        if prior_stride is None:
            os.environ.pop("NOVPHY_PHYSICS_CAPTURE_V2_STRIDE", None)
        else:
            os.environ["NOVPHY_PHYSICS_CAPTURE_V2_STRIDE"] = prior_stride

    final_protocol = final_frozen(issue_15_plan_root)[1]
    manifest, frame_count = _publish(
        output=output,
        records=records,
        implementation_commit=implementation_commit,
        player=player,
        public=public,
        final_protocol=final_protocol,
        access_audit=access_audit,
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "artifact_identity": manifest["identity"],
        "implementation_commit": implementation_commit,
        "role_counts": manifest["role_counts"],
        "rollout_count": manifest["rollout_count"],
        "frame_count": frame_count,
        "access_audit": {
            key: access_audit[key]
            for key in (
                "workflow_identity", "authorization_state", "authorization_identity",
                "observed_access_count", "passed",
            )
        },
        "rerun_commands": [
            "python -u -m scripts.issue_59_aligned_observation_collection --dry-run "
            f"--implementation-commit {implementation_commit}",
            "python -u -m scripts.issue_59_aligned_observation_collection "
            f"--implementation-commit {implementation_commit} "
            f"--authorization-identity {authorization_identity}",
            "python -u -m scripts.issue_59_aligned_observation_collection --validate",
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_immutable_cohort_v2_json(summary, summary_path)
    _log(f"complete: release={manifest['identity']} frames={summary['frame_count']}")
    return summary


def validate(
    *,
    output: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
) -> dict[str, Any]:
    output = Path(output).resolve()
    manifest_path = output / "manifest.json"
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if (
        (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            .encode("utf-8")
            + b"\n"
        )
        != raw
        or manifest.get("schema") != SCHEMA
        or manifest.get("identity") != RELEASE_IDENTITY
        or manifest.get("role_counts") != {role: 6 for role in ROLE_ORDER}
        or manifest.get("rollout_count") != 24
        or set(manifest.get("partitions", ())) != {"public", "sealed_final"}
    ):
        raise Issue59CollectionError("issue-59 release manifest is malformed")
    partition_specs = (
        ("public", ROLE_ORDER[:3], 18),
        ("sealed_final", ROLE_ORDER[3:], 6),
    )
    records = []
    observation_manifests = []
    for name, included_roles, expected_count in partition_specs:
        reference = manifest["partitions"][name]
        partition_root = output / reference["path"]
        partition_raw = (partition_root / "manifest.json").read_bytes()
        partition = json.loads(partition_raw)
        if (
            (
                json.dumps(partition, ensure_ascii=False, indent=2, sort_keys=True)
                .encode("utf-8")
                + b"\n"
            )
            != partition_raw
            or partition.get("schema") != PARTITION_SCHEMA
            or partition.get("identity") != reference["identity"]
            or partition.get("included_roles") != list(included_roles)
            or partition.get("role_counts")
            != {role: 6 for role in included_roles}
            or len(partition.get("records", ())) != expected_count
        ):
            raise Issue59CollectionError("issue-59 partition manifest is malformed")
        records.extend(partition["records"])
        for record in partition["records"]:
            rollout = partition_root / record["rollout_path"]
            capture = load_physics_capture_v2(
                rollout / "physics_capture_v2.json"
            )
            observation = validate_observation_trace(
                rollout / "observation-trace"
            )
            observation_manifests.append(observation)
            if (
                capture.capture_id != record["capture_id"]
                or observation["identity"]
                != record["observation_manifest_identity"]
                or tuple(
                    item["fixed_step"]
                    for item in capture.record["frame_records"]
                )
                != tuple(
                    item["fixed_step"]
                    for item in observation["frame_records"]
                )
            ):
                raise Issue59CollectionError("issue-59 rollout alignment differs")
            derivations = {
                item["kind"]: item for item in record["derivations"]
            }
            if set(derivations) != {"micro", "macro", "physical-violations"}:
                raise Issue59CollectionError("issue-59 derivation inventory differs")
            values = {
                kind: _load(partition_root / item["path"])
                for kind, item in derivations.items()
            }
            for kind, item in derivations.items():
                if values[kind].get("identity") != item["identity"]:
                    raise Issue59CollectionError(
                        "issue-59 derivation identity differs"
                    )
            source_reference = (
                Path(record["rollout_path"]) / "physics_capture_v2.json"
            ).as_posix()
            validate_capture_micro_relation_derivation(
                values["micro"], capture,
                source_reference=source_reference,
                source_capture_bundle_identity=RELEASE_IDENTITY,
            )
            validate_capture_macro_derivation(
                values["macro"], capture,
                source_reference=source_reference,
                source_capture_bundle_identity=RELEASE_IDENTITY,
            )
            validate_capture_physical_violation_derivation(
                values["physical-violations"], capture,
                source_reference=source_reference,
                source_capture_bundle_identity=RELEASE_IDENTITY,
            )
    if Counter(item["exposure_role"] for item in records) != Counter(
        {role: 6 for role in ROLE_ORDER}
    ):
        raise Issue59CollectionError("issue-59 record roles differ")
    validate_observation_exposure_boundaries(observation_manifests)
    summary = _load(summary_path)
    if (
        summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("artifact_identity") != RELEASE_IDENTITY
        or summary.get("rollout_count") != 24
        or summary.get("frame_count")
        != sum(item["frame_count"] for item in records)
    ):
        raise Issue59CollectionError("issue-59 compact summary differs")
    _log(
        f"validate: exact aligned release passed frames={summary['frame_count']}"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--public-plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--issue-15-plan-root", type=Path, default=ISSUE_15_PLAN_ROOT)
    parser.add_argument("--stage-root", type=Path, default=STAGE_ROOT)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--authorization-identity", default=AUTHORIZATION_IDENTITY)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dry_run and args.validate:
        raise Issue59CollectionError("--dry-run and --validate are mutually exclusive")
    root = args.repository_root.resolve()
    implementation = args.implementation_commit
    if implementation is None:
        implementation, dirty = git_revision(str(root))
        if dirty and not args.validate:
            raise Issue59CollectionError(
                "dirty collection source requires --implementation-commit"
            )
    if args.validate:
        validate(output=args.output, summary_path=args.summary)
    elif args.dry_run:
        result = dry_run(
            implementation_commit=str(implementation),
            public_plan_root=args.public_plan_root,
            issue_15_plan_root=args.issue_15_plan_root,
            stage_root=args.stage_root,
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    else:
        collect(
            implementation_commit=str(implementation),
            authorization_identity=args.authorization_identity,
            runtime_root=args.runtime_root,
            output=args.output,
            summary_path=args.summary,
            public_plan_root=args.public_plan_root,
            issue_15_plan_root=args.issue_15_plan_root,
            stage_root=args.stage_root,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Issue59CollectionError as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(2) from error
