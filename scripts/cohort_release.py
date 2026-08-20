"""Publish and verify an immutable representative cohort release."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Final, Mapping, Sequence

from scripts.cohort_partition import CohortPartitionManifest, audit_cohort_partition_manifest
from scripts.collection_plan import PLAN_COPY_FILENAME, REPORT_FILENAME, load_collection_plan
from scripts.physics_macro_labels import MACRO_LABEL_SIDECAR, derive_macro_labels_for_shot, write_macro_label_file
from scripts.production_plan import PRODUCTION_PLAN_COPY_FILENAME, load_production_plan
from scripts.rollout_artifacts import validate_physics_shot_artifact

RELEASE_SCHEMA: Final = "representative_cohort_release_v1"
DERIVATION_SCHEMA: Final = "authoritative_cohort_derivations_v1"
PUBLICATION_SCHEMA: Final = "representative_cohort_publication_v1"
NAMESPACES: Final = {
    RELEASE_SCHEMA: "representative-cohort-release-v1",
    DERIVATION_SCHEMA: "authoritative-cohort-derivations-v1",
    PUBLICATION_SCHEMA: "representative-cohort-publication-v1",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def _identity(payload: Mapping[str, Any]) -> str:
    content = dict(payload)
    content.pop("identity", None)
    return f"{NAMESPACES[str(payload['schema'])]}:sha256:{sha256(_canonical_json(content)).hexdigest()}"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load {name} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _write(path: Path, value: Any) -> Path:
    data = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Immutable artifact differs: {path}")
        return path
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _copy(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise ValueError(f"Immutable copy differs: {destination}")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as handle:
        handle.write(source.read_bytes())
        temporary = Path(handle.name)
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _artifact(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Attempt has no artifact path")
    source = Path(value)
    resolved = (source if source.is_absolute() else root / source).resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError("Attempt artifact is outside production output")
    return resolved


def _raw_inventory(root: Path, shot: Path) -> list[dict[str, Any]]:
    paths = [shot / name for name in ("metadata.json", "physics_state.jsonl", "physics_events.jsonl")]
    frames = shot / "frames"
    if frames.is_dir():
        paths += [path for path in sorted(frames.rglob("*")) if path.is_file()]
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError(f"Primary rollout has missing or non-regular files: {shot}")
    return [
        {"path": path.relative_to(root).as_posix(), "sha256": _digest(path), "size_bytes": path.stat().st_size}
        for path in paths
    ]


def _quality(root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    ledger, planned = report.get("attempt_ledger"), report.get("planned_slots")
    if not isinstance(ledger, list) or not isinstance(planned, list):
        raise ValueError("Collection report lacks plan and attempt accounting")
    counts = {name: report.get(f"{name}_count") for name in ("accepted", "rejected", "failed", "quarantined")}
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values()):
        raise ValueError("Collection counts are invalid")
    if counts["accepted"] + counts["rejected"] + counts["failed"] != len(ledger):
        raise ValueError("Collection report does not account for every attempt")
    if report.get("unmet_slots") or report.get("realized_coverage_shortfalls"):
        raise ValueError("Production has unmet slots or coverage shortfalls")
    normalized = []
    for entry in ledger:
        item = dict(entry)
        for field in ("artifact_path", "quarantine_path", "failure_manifest_path"):
            if isinstance(item.get(field), str) and item[field]:
                item[field] = _artifact(root, item[field]).relative_to(root).as_posix()
        normalized.append(item)
    return {
        "schema": "production_quality_report_v1",
        "plan_identity": report.get("plan_identity"),
        "plan_version": report.get("plan_version"),
        "planned_rollout_count": len(planned),
        "attempted_rollout_count": len(ledger),
        "counts": counts,
        "attempt_ledger": normalized,
        "realized_coverage_stratum_counts": report.get("realized_coverage_stratum_counts"),
        "unmet_slots": [], "coverage_shortfalls": [], "systematic_exporter_defects": [],
    }


def _prior_history(release_root: Path, sources: Mapping[str, Path]) -> list[dict[str, Any]]:
    history = []
    for name, source in sorted(sources.items()):
        data = _load(Path(source), "prior execution report")
        copied = _copy(Path(source), release_root / "prior_executions" / f"{name}.json")
        history.append({
            "name": name,
            "schema": data.get("schema"),
            "path": copied.relative_to(release_root).as_posix(),
            "sha256": _digest(copied),
            "attempt_count": len(data.get("attempt_ledger", [])),
            "accepted_count": data.get("accepted_count", 0),
            "rejected_count": data.get("rejected_count", 0),
            "failed_count": data.get("failed_count", 1 if data.get("status") == "failed" else 0),
            "quarantined_count": data.get("quarantined_count", 0),
        })
    return history


def _derivations(root: Path, release_root: Path, release_id: str, rollouts: Sequence[Mapping[str, Any]], available: Mapping[str, str], unavailable: Mapping[str, str]) -> tuple[Path, dict[str, Any]]:
    artifacts = []
    for rollout in rollouts:
        attempt_id, shot = str(rollout["attempt_id"]), root / str(rollout["path"])
        destination = release_root / "derivations" / attempt_id
        destination.mkdir(parents=True, exist_ok=True)
        labels = derive_macro_labels_for_shot(shot)
        macro = destination / MACRO_LABEL_SIDECAR
        if macro.exists() and macro.read_text(encoding="utf-8") != labels.to_jsonl():
            raise ValueError(f"Immutable macro derivation differs: {macro}")
        if not macro.exists():
            write_macro_label_file(labels, macro)
        artifacts.append({"attempt_id": attempt_id, "kind": "physics_macro_labels_v1", "path": macro.relative_to(release_root).as_posix(), "sha256": _digest(macro), "accepted_predicates": ["steady-state", "structure-unstable"], "excluded_predicates": ["cascade-active", "collapsed", "pigs-cleared"]})
        relational = shot / "physics_relational_supervision.jsonl"
        if not relational.is_file():
            raise ValueError(f"Accepted rollout lacks relational supervision: {shot}")
        copied = _copy(relational, destination / relational.name)
        artifacts.append({"attempt_id": attempt_id, "kind": "physics_relational_supervision_v1", "path": copied.relative_to(release_root).as_posix(), "sha256": _digest(copied)})
    payload = {"schema": DERIVATION_SCHEMA, "identity": "", "derivation_version": 1, "source_cohort_release_identity": release_id, "available_capabilities": dict(sorted(available.items())), "unavailable_capabilities": dict(sorted(unavailable.items())), "artifacts": artifacts}
    payload["identity"] = _identity(payload)
    path = release_root / f"authoritative_derivations_{payload['identity'].rsplit(':', 1)[-1]}_v1.json"
    return _write(path, payload), payload


def publish_cohort_release(output_dir: Path, *, partition_manifest_path: Path, scenario_manifest_paths: Mapping[str, Path], release_version: int, code_revision: str, available_capabilities: Mapping[str, str], unavailable_capabilities: Mapping[str, str], prior_execution_paths: Mapping[str, Path] | None = None) -> Path:
    if isinstance(release_version, bool) or not isinstance(release_version, int) or release_version <= 0 or not code_revision:
        raise ValueError("Release version and code revision are required")
    root = Path(output_dir).resolve()
    loaded_plan = load_collection_plan(root / PLAN_COPY_FILENAME)
    production_plan = load_production_plan(root / PRODUCTION_PLAN_COPY_FILENAME)
    report = _load(root / REPORT_FILENAME, "collection report")
    if production_plan.source_collection_plan["identity"] != loaded_plan.plan.identity or report.get("plan_identity") != loaded_plan.plan.identity:
        raise ValueError("Production artifacts are not bound to the frozen collection plan")
    partition = CohortPartitionManifest.from_dict(_load(Path(partition_manifest_path), "partition manifest"))
    lineages = [scenario.scenario_manifest_projection["scenario_lineage_identity"] for scenario in loaded_plan.plan.scenarios]
    audit_cohort_partition_manifest(partition, admitted_scenario_lineage_identities=lineages, admitted_provenance_records=[record.to_dict() for record in partition.provenance_records])

    release_root = root / "release"
    partition_copy = _copy(Path(partition_manifest_path), release_root / "partition_manifest.json")
    if set(scenario_manifest_paths) != {scenario.scenario_id for scenario in loaded_plan.plan.scenarios}:
        raise ValueError("Scenario manifest sources must exactly match the plan")
    scenarios = []
    for scenario in loaded_plan.plan.scenarios:
        source = Path(scenario_manifest_paths[scenario.scenario_id])
        if _load(source, "scenario manifest") != scenario.scenario_manifest_projection["scenario_manifest"]:
            raise ValueError(f"Scenario manifest differs from plan: {scenario.scenario_id}")
        copied = _copy(source, release_root / "scenario_manifests" / f"{scenario.scenario_id}.json")
        scenarios.append({"scenario_id": scenario.scenario_id, "scenario_lineage_identity": scenario.scenario_manifest_projection["scenario_lineage_identity"], "path": copied.relative_to(release_root).as_posix(), "sha256": _digest(copied)})

    quality = _quality(root, report)
    quality["prior_executions"] = _prior_history(release_root, prior_execution_paths or {})
    quality_path = _write(release_root / "production_quality_report.json", quality)
    rollouts = []
    for entry in report["attempt_ledger"]:
        if entry.get("status") == "accepted" and entry.get("disposition") == "accept":
            shot = _artifact(root, entry.get("artifact_path"))
            validate_physics_shot_artifact(shot)
            rollouts.append({"attempt_id": entry["attempt_id"], "scenario_id": entry["scenario_id"], "intervention_id": entry["intervention_id"], "path": shot.relative_to(root).as_posix(), "files": _raw_inventory(root, shot)})
    if not rollouts or len(rollouts) != quality["counts"]["accepted"]:
        raise ValueError("Accepted rollout inventory does not match accounting")

    release = {
        "schema": RELEASE_SCHEMA, "identity": "", "release_version": release_version, "code_revision": code_revision,
        "source_pilot_report_identity": production_plan.source_pilot_report["identity"],
        "production_plan": {"identity": production_plan.identity, "version": production_plan.plan_version, "path": PRODUCTION_PLAN_COPY_FILENAME, "sha256": _digest(root / PRODUCTION_PLAN_COPY_FILENAME)},
        "collection_plan": {"identity": loaded_plan.plan.identity, "version": loaded_plan.plan.plan_version, "path": PLAN_COPY_FILENAME, "sha256": _digest(root / PLAN_COPY_FILENAME)},
        "partition_manifest": {"identity": partition.identity, "version": partition.partition_version, "path": partition_copy.relative_to(root).as_posix(), "sha256": _digest(partition_copy)},
        "scenario_manifests": scenarios, "primary_rollouts": rollouts,
        "quality_report": {"path": quality_path.relative_to(root).as_posix(), "sha256": _digest(quality_path)},
    }
    release["identity"] = _identity(release)
    release_path = _write(release_root / f"cohort_release_{release['identity'].rsplit(':', 1)[-1]}_v{release_version}.json", release)
    derivation_path, derivation = _derivations(root, release_root, release["identity"], rollouts, available_capabilities, unavailable_capabilities)
    publication = {"schema": PUBLICATION_SCHEMA, "identity": "", "cohort_release": {"identity": release["identity"], "path": release_path.relative_to(root).as_posix(), "sha256": _digest(release_path)}, "authoritative_derivations": {"identity": derivation["identity"], "path": derivation_path.relative_to(root).as_posix(), "sha256": _digest(derivation_path)}}
    publication["identity"] = _identity(publication)
    return _write(release_root / f"cohort_publication_{publication['identity'].rsplit(':', 1)[-1]}_v{release_version}.json", publication)


def verify_cohort_publication(path: Path) -> dict[str, Any]:
    publication_path = Path(path).resolve()
    publication = _load(publication_path, "cohort publication")
    if publication.get("schema") != PUBLICATION_SCHEMA or publication.get("identity") != _identity(publication):
        raise ValueError("Cohort publication identity is stale or unsupported")
    root = publication_path.parent.parent
    for field, schema in (("cohort_release", RELEASE_SCHEMA), ("authoritative_derivations", DERIVATION_SCHEMA)):
        reference = publication.get(field)
        artifact_path = root / str(reference.get("path")) if isinstance(reference, dict) else root
        if not isinstance(reference, dict) or _digest(artifact_path) != reference.get("sha256"):
            raise ValueError(f"Cohort publication {field} digest mismatch")
        artifact = _load(artifact_path, field)
        if artifact.get("schema") != schema or artifact.get("identity") != _identity(artifact) or artifact.get("identity") != reference.get("identity"):
            raise ValueError(f"Cohort publication {field} identity mismatch")
    return publication


def _pairs(values: Sequence[str], option: str) -> dict[str, str]:
    result = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or not item or key in result:
            raise ValueError(f"{option} values must use unique NAME=VALUE pairs")
        result[key] = item
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish an immutable representative cohort release")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--partition-manifest", required=True, type=Path)
    parser.add_argument("--scenario-manifest", action="append", required=True, metavar="SCENARIO_ID=PATH")
    parser.add_argument("--release-version", required=True, type=int)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--available-capability", action="append", default=[])
    parser.add_argument("--unavailable-capability", action="append", default=[])
    parser.add_argument("--prior-execution", action="append", default=[], metavar="NAME=PATH")
    args = parser.parse_args(argv)
    try:
        path = publish_cohort_release(args.output_dir, partition_manifest_path=args.partition_manifest, scenario_manifest_paths={key: Path(value) for key, value in _pairs(args.scenario_manifest, "--scenario-manifest").items()}, release_version=args.release_version, code_revision=args.code_revision, available_capabilities=_pairs(args.available_capability, "--available-capability"), unavailable_capabilities=_pairs(args.unavailable_capability, "--unavailable-capability"), prior_execution_paths={key: Path(value) for key, value in _pairs(args.prior_execution, "--prior-execution").items()})
        publication = verify_cohort_publication(path)
    except (OSError, ValueError) as error:
        print(str(error), file=os.sys.stderr)
        return 2
    print(json.dumps({"publication": str(path), "identity": publication["identity"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
