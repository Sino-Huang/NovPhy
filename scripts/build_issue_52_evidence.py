"""Build and validate the immutable cohort-v2 production plans for issue #52."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from scripts.cohort_v2_production_plans import (
    BUNDLE_IDENTITY,
    BUNDLE_SCHEMA,
    COLLECTION_IDENTITY,
    PARAMETER_IDENTITY,
    ROOT,
    derive_issue_52_payloads,
    validate_issue_52_payloads,
)
from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_json


DEFAULT_OUTPUT = ROOT / "data/runtime_evidence/issue-52"


def _log(message: str) -> None:
    print(f"[issue-52] {message}", flush=True)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load issue-52 artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Issue-52 artifact {path} must be an object")
    return value


def _bundle() -> dict[str, Any]:
    return {
        "schema": BUNDLE_SCHEMA,
        "identity": BUNDLE_IDENTITY,
        "github_issue": 52,
        "source_pilot_report_identity": (
            "representative-cohort-v2-pilot-report-v1:accepted-determination-6"
        ),
        "collection_plan_identity": COLLECTION_IDENTITY,
        "production_parameter_plan_identity": PARAMETER_IDENTITY,
        "artifacts": ["collection-plan.json", "production-parameter-plan.json"],
        "passed": True,
    }


def _result() -> dict[str, Any]:
    return {
        "schema": "issue_52_cohort_v2_production_plans_validation_result_v1",
        "bundle_identity": BUNDLE_IDENTITY,
        "collection_plan_identity": COLLECTION_IDENTITY,
        "production_parameter_plan_identity": PARAMETER_IDENTITY,
        "planned_rollouts": 24,
        "passed": True,
    }


def validate_issue_52_evidence(
    evidence_root: Path,
    *,
    repository_root: Path = ROOT,
    revalidate_pilot: bool = True,
) -> dict[str, Any]:
    root = Path(evidence_root)
    _log("deriving expected plans from the accepted issue-51 pilot")
    expected = derive_issue_52_payloads(
        repository_root,
        validate_pilot=revalidate_pilot,
    )
    observed = {
        name: _load_object(root / name)
        for name in ("collection-plan.json", "production-parameter-plan.json")
    }
    validate_issue_52_payloads(observed, repository_root)
    for name, payload in expected.items():
        if observed[name] != payload:
            raise ValueError(f"Issue-52 artifact is stale: {name}")
    if _load_object(root / "bundle-manifest.json") != _bundle():
        raise ValueError("Issue-52 bundle membership or identity is stale")
    members = sorted(path.name for path in root.iterdir() if path.is_file())
    if members != [
        "bundle-manifest.json",
        "collection-plan.json",
        "production-parameter-plan.json",
    ]:
        raise ValueError("Issue-52 bundle contains undeclared members")
    _log("collection, parameter, quota, exposure, and evidence bindings passed")
    return _result()


def build_issue_52_evidence(
    output: Path,
    *,
    repository_root: Path = ROOT,
    dry_run: bool = False,
) -> dict[str, Any]:
    _log("revalidating the accepted capability-complete issue-51 pilot")
    payloads = derive_issue_52_payloads(repository_root, validate_pilot=True)
    validate_issue_52_payloads(payloads, repository_root)
    _log("derived 4 role assignments and 24 outcome-independent attempts")
    _log("validated six central strata, three quota-bearing terminations, and all parameter evidence")
    if dry_run:
        _log("dry-run passed; no artifact was written")
        return _result()

    target = Path(output)
    if target.exists():
        _log(f"validating existing immutable publication at {target}")
        return validate_issue_52_evidence(
            target,
            repository_root=repository_root,
            revalidate_pilot=False,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".issue-52-", dir=target.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        for name, payload in payloads.items():
            write_immutable_cohort_v2_json(payload, staging / name)
        write_immutable_cohort_v2_json(_bundle(), staging / "bundle-manifest.json")
        _log("validating staged immutable bundle")
        validate_issue_52_evidence(
            staging,
            repository_root=repository_root,
            revalidate_pilot=False,
        )
        os.replace(staging, target)
    _log(f"immutable issue-52 plans published: {target}")
    return _result()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or validate issue #52 cohort-v2 production plans"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--validate", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.validate:
        result = validate_issue_52_evidence(args.output)
    else:
        result = build_issue_52_evidence(args.output, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
