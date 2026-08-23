"""Publish the immutable stable-only issue-53 plan and sealed seed-4503 authority."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from scripts.build_issue_45_evidence import ROLES
from scripts.cohort_v2_production_plans_v2 import (
    DEFAULT_PLAN_ROOT,
    DEFAULT_SEALED_AUTHORITY_ROOT,
    FINAL_SEED,
    PLAN_MEMBERS,
    ROOT,
    SEALED_AUTHORITY_IDENTITY,
    _materialize_seed_4503,
    build_plan_v2_payloads,
    validate_plan_v2_evidence,
    validate_plan_v2_payloads,
)
from scripts.cohort_v2_scenarios import (
    load_cohort_v2_scenario_manifest,
    write_cohort_v2_scenario_manifest,
    write_immutable_cohort_v2_bytes,
    write_immutable_cohort_v2_json,
)


def validate_sealed_plan_v2_authority(
    plan_root: Path = DEFAULT_PLAN_ROOT,
    sealed_root: Path = DEFAULT_SEALED_AUTHORITY_ROOT,
) -> dict[str, object]:
    public_result = validate_plan_v2_evidence(plan_root)
    sealed_root = Path(sealed_root)
    bundle = json.loads(
        (sealed_root / "sealed-bundle-manifest.json").read_text(encoding="utf-8")
    )
    expected_members = [
        "final-evaluation.cohort-v2-scenario.json",
        "final-evaluation.parameter-realization.json",
        "final-evaluation.xml",
    ]
    if bundle != {
        "schema": "issue_53_plan_v2_sealed_final_authority_bundle_v2",
        "identity": SEALED_AUTHORITY_IDENTITY,
        "generation_seed": FINAL_SEED,
        "scenario_manifest_identity": bundle.get("scenario_manifest_identity"),
        "ordinary_workflow_access": False,
        "artifacts": expected_members,
    }:
        raise ValueError("Plan-v2 sealed final authority bundle is stale")
    scenario = load_cohort_v2_scenario_manifest(
        sealed_root / "final-evaluation.cohort-v2-scenario.json",
        xml_path=sealed_root / "final-evaluation.xml",
        template_source_path=(
            ROOT / ROLES[3].family.source_reference
        ),
    )
    projection = json.loads(
        (Path(plan_root) / "final-evaluation.sealed-projection.json").read_text(
            encoding="utf-8"
        )
    )
    realization = json.loads(
        (sealed_root / "final-evaluation.parameter-realization.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        scenario.identity != bundle["scenario_manifest_identity"]
        or scenario.identity != projection["scenario_manifest_identity"]
        or scenario.scenario_manifest.generation.generation_seed != FINAL_SEED
        or realization
        != scenario.scenario_manifest.generation.parameter_realization
    ):
        raise ValueError("Plan-v2 sealed final authority differs from its projection")
    return {
        **public_result,
        "sealed_final_authority_identity": SEALED_AUTHORITY_IDENTITY,
        "sealed_final_authority_validated": True,
    }


def build_issue_53_plan_v2(
    plan_root: Path = DEFAULT_PLAN_ROOT,
    sealed_root: Path = DEFAULT_SEALED_AUTHORITY_ROOT,
) -> dict[str, object]:
    plan_root = Path(plan_root)
    sealed_root = Path(sealed_root)
    if plan_root.exists() or sealed_root.exists():
        if not plan_root.is_dir() or not sealed_root.is_dir():
            raise ValueError("Plan-v2 immutable destinations are incomplete")
        return validate_sealed_plan_v2_authority(plan_root, sealed_root)
    plan_root.parent.mkdir(parents=True, exist_ok=True)
    sealed_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".issue-53-plan-v2-", dir=plan_root.parent) as public_temp, tempfile.TemporaryDirectory(
        prefix=".issue-53-plan-v2-sealed-", dir=sealed_root.parent
    ) as sealed_temp:
        staged_public = Path(public_temp) / "bundle"
        staged_sealed = Path(sealed_temp) / "bundle"
        staged_public.mkdir()
        staged_sealed.mkdir()
        _role, materialized, scenario = _materialize_seed_4503(staged_sealed)
        payloads = build_plan_v2_payloads(scenario)
        validate_plan_v2_payloads(payloads)
        for name in PLAN_MEMBERS:
            write_immutable_cohort_v2_json(payloads[name], staged_public / name)
        write_immutable_cohort_v2_bytes(
            materialized.xml_content, staged_sealed / "final-evaluation.xml"
        )
        write_cohort_v2_scenario_manifest(
            scenario, staged_sealed / "final-evaluation.cohort-v2-scenario.json"
        )
        write_immutable_cohort_v2_json(
            materialized.parameter_realization,
            staged_sealed / "final-evaluation.parameter-realization.json",
        )
        write_immutable_cohort_v2_json(
            {
                "schema": "issue_53_plan_v2_sealed_final_authority_bundle_v2",
                "identity": SEALED_AUTHORITY_IDENTITY,
                "generation_seed": FINAL_SEED,
                "scenario_manifest_identity": scenario.identity,
                "ordinary_workflow_access": False,
                "artifacts": [
                    "final-evaluation.cohort-v2-scenario.json",
                    "final-evaluation.parameter-realization.json",
                    "final-evaluation.xml",
                ],
            },
            staged_sealed / "sealed-bundle-manifest.json",
        )
        validate_plan_v2_evidence(staged_public)
        os.replace(staged_sealed, sealed_root)
        os.replace(staged_public, plan_root)
    return validate_sealed_plan_v2_authority(plan_root, sealed_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish stable-only issue-53 plan v2")
    parser.add_argument("--output", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument(
        "--sealed-output", type=Path, default=DEFAULT_SEALED_AUTHORITY_ROOT
    )
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    result = (
        validate_sealed_plan_v2_authority(args.output, args.sealed_output)
        if args.validate
        else build_issue_53_plan_v2(args.output, args.sealed_output)
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
