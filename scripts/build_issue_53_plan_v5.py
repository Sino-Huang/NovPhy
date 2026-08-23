"""Publish or validate the issue-53 workflow-time successor plan."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from scripts.cohort_v2_production_plans_v5 import (
    DEFAULT_PLAN_ROOT,
    PLAN_MEMBERS,
    ROOT,
    build_plan_v5_payloads,
    validate_plan_v5_evidence,
    validate_plan_v5_payloads,
)
from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_json


def build_issue_53_plan_v5(output: Path = DEFAULT_PLAN_ROOT) -> dict[str, Any]:
    target = Path(output)
    if target.exists():
        return validate_plan_v5_evidence(target)
    payloads = build_plan_v5_payloads(ROOT)
    validate_plan_v5_payloads(payloads)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".issue-53-plan-v5-", dir=target.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        for name in PLAN_MEMBERS:
            write_immutable_cohort_v2_json(payloads[name], staging / name)
        validate_plan_v5_evidence(staging)
        os.replace(staging, target)
    return validate_plan_v5_evidence(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish issue-53 workflow-time plan v5")
    parser.add_argument("--output", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    result = (
        validate_plan_v5_evidence(args.output)
        if args.validate
        else build_issue_53_plan_v5(args.output)
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
