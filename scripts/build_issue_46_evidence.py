"""Build and validate representative issue #46 observation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from scripts.observation_trace import (
    ObservationTraceError,
    audit_observation_access,
    persist_observation_trace,
    validate_observation_exposure_boundaries,
    validate_observation_trace,
)


BUNDLE_NAME = "observation-evidence-bundle.json"
NON_FINAL_ROLES = frozenset({"training", "calibration", "model_selection"})
PROBE_FIELDS = {
    "probe_identity", "evidence_source", "source_snapshot_commit",
    "player_archive_identity", "scenario_manifest_identity",
    "observation_configuration", "exposure_role", "source_bindings", "captures",
}


class Issue46EvidenceError(ValueError):
    """Issue #46 representative evidence is incomplete or stale."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _identity(namespace: str, value: Any) -> str:
    return f"{namespace}:sha256:{hashlib.sha256(_canonical_json(value)).hexdigest()}"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_probe_inputs(probes: Sequence[Mapping[str, Any]]) -> None:
    if len(probes) < 2:
        raise Issue46EvidenceError("representative evidence requires at least two probes")
    probe_ids = []
    templates = set()
    levels = set()
    lineages = set()
    configurations = set()
    for probe in probes:
        if not isinstance(probe, Mapping) or set(probe) != PROBE_FIELDS:
            raise Issue46EvidenceError("representative probe fields are incomplete")
        probe_id = probe["probe_identity"]
        if not isinstance(probe_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", probe_id) is None:
            raise Issue46EvidenceError("representative probe identity is invalid")
        probe_ids.append(probe_id)
        if probe["evidence_source"] != "unity_runtime_non_fixture":
            raise Issue46EvidenceError("representative evidence must be non-fixture Unity runtime evidence")
        if probe["exposure_role"] not in NON_FINAL_ROLES:
            raise Issue46EvidenceError("representative evidence must use non-final exposure roles")
        for field in (
            "source_snapshot_commit", "player_archive_identity",
            "scenario_manifest_identity", "observation_configuration",
        ):
            if not isinstance(probe[field], str) or not probe[field]:
                raise Issue46EvidenceError(f"representative probe {field} is missing")
        bindings = probe["source_bindings"]
        if not isinstance(bindings, Mapping):
            raise Issue46EvidenceError("representative probe source bindings are missing")
        try:
            templates.add(bindings["scenario_template_identity"])
            levels.add(bindings["level_instance_identity"])
            lineages.add(bindings["source_scenario_lineage_identity"])
        except KeyError as error:
            raise Issue46EvidenceError("representative probe source bindings are missing") from error
        configurations.add(probe["observation_configuration"])
        if not isinstance(probe["captures"], (list, tuple)) or not probe["captures"]:
            raise Issue46EvidenceError("representative probe has no runtime capture")
    if len(probe_ids) != len(set(probe_ids)):
        raise Issue46EvidenceError("representative evidence requires two distinct probe identities")
    coverage = (templates, levels, lineages, configurations)
    if any(len(values) < 2 for values in coverage):
        raise Issue46EvidenceError(
            "representative evidence requires two distinct lineages, levels, templates, and transforms"
        )


def build_issue_46_evidence(
    root: Path,
    probes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Publish a closed representative bundle from non-fixture request-72 probes."""
    _validate_probe_inputs(probes)
    target = Path(root)
    if target.exists():
        raise Issue46EvidenceError("issue #46 evidence destination already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        entries = []
        manifests = []
        for probe in probes:
            trace_relative = f"traces/{probe['probe_identity']}"
            manifest = persist_observation_trace(
                staging / trace_relative,
                probe["captures"],
                observation_configuration=probe["observation_configuration"],
                source_bindings=probe["source_bindings"],
                exposure_role=probe["exposure_role"],
            )
            manifests.append(manifest)
            entries.append({
                "probe_identity": probe["probe_identity"],
                "evidence_source": probe["evidence_source"],
                "source_snapshot_commit": probe["source_snapshot_commit"],
                "player_archive_identity": probe["player_archive_identity"],
                "scenario_manifest_identity": probe["scenario_manifest_identity"],
                "trace_relative_path": trace_relative,
                "observation_trace_manifest_identity": manifest["identity"],
                "exposure_role": manifest["exposure_role"],
                "scenario_template_identity": manifest["source_bindings"]["scenario_template_identity"],
                "level_instance_identity": manifest["source_bindings"]["level_instance_identity"],
                "source_scenario_lineage_identity": manifest["source_bindings"]["source_scenario_lineage_identity"],
                "scenario_lineage_identity": manifest["scenario_lineage_identity"],
                "observation_configuration_identity": manifest["observation_configuration"]["identity"],
            })
        validate_observation_exposure_boundaries(manifests)

        access_attempts = [
            {
                "attempt_identity": "issue-46:canonical:diagnostic",
                "observation_role": "canonical",
                "workflow_kind": "diagnostic",
                "purpose": "alignment_diagnosis",
            },
            {
                "attempt_identity": "issue-46:canonical:training",
                "observation_role": "canonical",
                "workflow_kind": "training",
                "purpose": "model_input",
            },
            {
                "attempt_identity": "issue-46:canonical:model-selection",
                "observation_role": "canonical",
                "workflow_kind": "model_selection",
                "purpose": "comparator_selection",
            },
        ]
        access_report = audit_observation_access(manifests[0], access_attempts)
        _write_json(staging / "access-audit.json", access_report)
        coverage = {
            "probe_count": len(entries),
            "non_final_scenario_lineage_count": len({
                entry["source_scenario_lineage_identity"] for entry in entries
            }),
            "level_instance_count": len({entry["level_instance_identity"] for entry in entries}),
            "scenario_template_count": len({entry["scenario_template_identity"] for entry in entries}),
            "observation_configuration_count": len({
                entry["observation_configuration_identity"] for entry in entries
            }),
        }
        decisions = {item["attempt_identity"]: item["allowed"] for item in access_report["decisions"]}
        bundle = {
            "schema": "issue_46_observation_evidence_bundle_v1",
            "identity": "",
            "issue": 46,
            "probes": entries,
            "coverage": coverage,
            "access_audit_relative_path": "access-audit.json",
            "access_audit_identity": access_report["identity"],
            "access_behavior": {
                "authorized_canonical_diagnostic": decisions["issue-46:canonical:diagnostic"] is True,
                "rejected_canonical_training": decisions["issue-46:canonical:training"] is False,
                "rejected_canonical_model_selection": decisions["issue-46:canonical:model-selection"] is False,
            },
            "immutable_non_fixture_observation_capability": True,
            "passed": True,
        }
        bundle["identity"] = _identity("issue-46-observation-evidence-bundle-v1", {
            key: value for key, value in bundle.items() if key != "identity"
        })
        _write_json(staging / BUNDLE_NAME, bundle)
        validate_issue_46_evidence(staging)
        os.replace(staging, target)
        return bundle
    except (ObservationTraceError, OSError, ValueError) as error:
        if isinstance(error, Issue46EvidenceError):
            raise
        raise Issue46EvidenceError(str(error)) from error
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def validate_issue_46_evidence(root: Path) -> dict[str, Any]:
    """Revalidate bundle coverage, traces, exact transforms, and access decisions."""
    base = Path(root)
    try:
        bundle = json.loads((base / BUNDLE_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Issue46EvidenceError("issue #46 observation evidence bundle is missing or malformed") from error
    expected_bundle_identity = _identity("issue-46-observation-evidence-bundle-v1", {
        key: value for key, value in bundle.items() if key != "identity"
    })
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema") != "issue_46_observation_evidence_bundle_v1"
        or bundle.get("identity") != expected_bundle_identity
        or bundle.get("issue") != 46
        or bundle.get("passed") is not True
    ):
        raise Issue46EvidenceError("issue #46 observation evidence bundle identity is stale")
    entries = bundle.get("probes")
    if not isinstance(entries, list) or len(entries) < 2:
        raise Issue46EvidenceError("issue #46 evidence requires at least two probes")
    manifests = []
    for entry in entries:
        if entry.get("evidence_source") != "unity_runtime_non_fixture":
            raise Issue46EvidenceError("issue #46 evidence is not non-fixture runtime evidence")
        if entry.get("exposure_role") not in NON_FINAL_ROLES:
            raise Issue46EvidenceError("issue #46 evidence includes a final lineage")
        relative = entry.get("trace_relative_path")
        if (
            not isinstance(relative, str) or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise Issue46EvidenceError("issue #46 trace path is invalid")
        try:
            manifest = validate_observation_trace(base / relative)
        except ObservationTraceError as error:
            raise Issue46EvidenceError(str(error)) from error
        manifests.append(manifest)
        expected_entry = {
            "observation_trace_manifest_identity": manifest["identity"],
            "exposure_role": manifest["exposure_role"],
            "scenario_template_identity": manifest["source_bindings"]["scenario_template_identity"],
            "level_instance_identity": manifest["source_bindings"]["level_instance_identity"],
            "source_scenario_lineage_identity": manifest["source_bindings"]["source_scenario_lineage_identity"],
            "scenario_lineage_identity": manifest["scenario_lineage_identity"],
            "observation_configuration_identity": manifest["observation_configuration"]["identity"],
        }
        if any(entry.get(field) != value for field, value in expected_entry.items()):
            raise Issue46EvidenceError("issue #46 probe identity binding is stale")
        for field in (
            "probe_identity", "source_snapshot_commit", "player_archive_identity",
            "scenario_manifest_identity",
        ):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise Issue46EvidenceError("issue #46 probe provenance is incomplete")
    try:
        validate_observation_exposure_boundaries(manifests)
    except ObservationTraceError as error:
        raise Issue46EvidenceError(str(error)) from error
    coverage = {
        "probe_count": len(entries),
        "non_final_scenario_lineage_count": len({entry["source_scenario_lineage_identity"] for entry in entries}),
        "level_instance_count": len({entry["level_instance_identity"] for entry in entries}),
        "scenario_template_count": len({entry["scenario_template_identity"] for entry in entries}),
        "observation_configuration_count": len({entry["observation_configuration_identity"] for entry in entries}),
    }
    if any(coverage[field] < 2 for field in coverage if field != "probe_count"):
        raise Issue46EvidenceError("issue #46 evidence does not span two required identities")
    if bundle.get("coverage") != coverage:
        raise Issue46EvidenceError("issue #46 evidence coverage is stale")

    try:
        access = json.loads((base / bundle["access_audit_relative_path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError) as error:
        raise Issue46EvidenceError("issue #46 access audit is missing or malformed") from error
    attempts = [
        {key: decision[key] for key in (
            "attempt_identity", "observation_role", "workflow_kind", "purpose"
        )}
        for decision in access.get("decisions", [])
    ]
    expected_access = audit_observation_access(manifests[0], attempts)
    if access != expected_access or bundle.get("access_audit_identity") != access["identity"]:
        raise Issue46EvidenceError("issue #46 access audit identity is stale")
    decisions = {item["attempt_identity"]: item["allowed"] for item in access["decisions"]}
    expected_behavior = {
        "authorized_canonical_diagnostic": decisions.get("issue-46:canonical:diagnostic") is True,
        "rejected_canonical_training": decisions.get("issue-46:canonical:training") is False,
        "rejected_canonical_model_selection": decisions.get("issue-46:canonical:model-selection") is False,
    }
    if bundle.get("access_behavior") != expected_behavior or not all(expected_behavior.values()):
        raise Issue46EvidenceError("issue #46 required access behavior is not demonstrated")
    if bundle.get("immutable_non_fixture_observation_capability") is not True:
        raise Issue46EvidenceError("issue #46 immutable observation capability is absent")
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    bundle = validate_issue_46_evidence(args.root)
    print(json.dumps({"identity": bundle["identity"], "passed": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
