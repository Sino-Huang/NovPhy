"""Audit the local closed-ticket migration-recovery dependency chain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Final, Sequence
import zipfile

from scripts.cohort_v2_migration_recovery import (
    DEFAULT_RELEASE_ROOT,
    MANIFEST_NAME,
    audit_surviving_public_release,
    validate_migration_recovery_manifest,
)
from scripts.cohort_v2_production_plans_v5 import (
    DEFAULT_PLAN_ROOT,
    validate_plan_v5_evidence,
)
from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_json
from scripts.verify_physics_player import verify_physics_player_archive


ROOT: Final = Path(__file__).resolve().parents[1]
SCHEMA: Final = "closed_ticket_migration_recovery_audit_v1"
DEFAULT_RECOVERY_ROOT: Final = Path(".local-artifacts/migration-recovery-v1")


class MigrationRecoveryAuditError(ValueError):
    """A present recovery component failed its exact validator."""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise MigrationRecoveryAuditError(f"cannot validate {path}: {error}") from error
    if not isinstance(value, dict):
        raise MigrationRecoveryAuditError(f"{path} is not a JSON object")
    return value


def _component(name: str, status: str, path: Path | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"component": name, "status": status}
    if path is not None:
        value["path"] = path.as_posix()
    return value


def _run_validator(command: Sequence[str], *, repository_root: Path) -> None:
    completed = subprocess.run(
        [sys.executable, *command],
        cwd=repository_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        output = completed.stdout[-4_000:]
        raise MigrationRecoveryAuditError(
            f"exact validator failed: {' '.join(command)}\n{output}"
        )


def _validated_regenerated(
    name: str,
    path: Path,
    *,
    requires: Sequence[Path],
    command: Sequence[str],
    repository_root: Path,
) -> dict[str, Any]:
    if not path.exists() or any(not item.exists() for item in requires):
        return _component(name, "missing", path)
    _run_validator(command, repository_root=repository_root)
    return _component(name, "regenerated-valid", path)


def build_recovery_audit(
    *,
    repository_root: Path = ROOT,
    recovery_root: Path = DEFAULT_RECOVERY_ROOT,
) -> dict[str, Any]:
    repository_root = Path(repository_root).resolve()
    recovery_root = Path(recovery_root).resolve()
    plan = (repository_root / DEFAULT_PLAN_ROOT).resolve()
    release = (repository_root / DEFAULT_RELEASE_ROOT).resolve()
    authority_root = recovery_root / "issue-53-authority"
    authority = authority_root / MANIFEST_NAME
    summaries = recovery_root / "summaries"
    components: list[dict[str, Any]] = []

    validate_plan_v5_evidence(
        plan, repository_root=repository_root, migration_recovery=True
    )
    components.append(_component("issue-53-plan-v5", "present-valid", plan))
    audit_surviving_public_release(release, plan_root=plan)
    components.append(_component("issue-53-public-v5", "present-valid", release))

    for version in (2, 3, 4):
        path = repository_root / f"data/runtime_evidence/issue-53-plan-v{version}"
        components.append(
            _component(
                f"issue-53-plan-v{version}",
                "missing" if not path.exists() else "present-valid",
                path,
            )
        )
    components.extend((
        _component("issue-53-final-evaluation-seed-4504", "excluded"),
        _component("issue-58-capacity-12-candidate", "excluded"),
    ))

    for name, relative in (
        ("issue-12-compact-summary", "issue-12/cohort-v2-reliability-summary.json"),
        ("issue-15-compact-summaries", "issue-15/capacity-integrated-calibration-summary.json"),
        ("issue-16-compact-summary", "issue-16/cohort-v2-feature-parser-stress-summary.json"),
        ("issue-17-compact-summary", "issue-17/cohort-v2-visual-parser-stress-summary.json"),
        ("issue-56-compact-summary", "issue-56/cohort-v2-gameplay-planning-smoke-summary.json"),
        ("issue-59-compact-summary", "issue-59/aligned-observation-release-summary.json"),
    ):
        path = repository_root / "data/runtime_evidence" / relative
        if path.exists():
            _load_object(path)
            components.append(_component(name, "present-valid", path))
        else:
            components.append(_component(name, "missing", path))

    for name, stage in (
        ("physics-v2-player", repository_root / "sciencebirdsgames/physics-v2"),
        (
            "aligned-observation-player",
            repository_root / "sciencebirdsgames/aligned-observation-v1",
        ),
    ):
        if stage.exists():
            verify_physics_player_archive(stage, physics_v2=True)
            components.append(_component(name, "present-valid", stage))
        else:
            components.append(_component(name, "missing", stage))
    linux_zip = repository_root / "sciencebirdsgames/Linux.zip"
    if linux_zip.exists():
        with zipfile.ZipFile(linux_zip) as archive:
            if archive.testzip() is not None:
                raise MigrationRecoveryAuditError("sciencebirdsgames/Linux.zip is corrupt")
        components.append(_component("sciencebirds-linux-archive", "present-valid", linux_zip))
    else:
        components.append(_component("sciencebirds-linux-archive", "missing", linux_zip))

    if authority.exists():
        validate_migration_recovery_manifest(
            authority,
            repository_root=repository_root,
            plan_root=plan,
            release_root=release,
        )
        components.append(
            _component("issue-53-migration-recovery-authority", "regenerated-valid", authority_root)
        )
    else:
        components.append(
            _component("issue-53-migration-recovery-authority", "missing", authority_root)
        )

    reliability = recovery_root / "issue-12-reliability"
    reliability_summary = summaries / "issue-12-reliability-summary.json"
    components.append(_validated_regenerated(
        "issue-12-reliability",
        reliability,
        requires=(authority, reliability_summary),
        command=(
            "-m", "scripts.run_cohort_v2_reliability",
            "--release-root", str(release),
            "--output", str(reliability),
            "--compact-report", str(reliability_summary),
            "--migration-recovery", str(authority),
            "--validate",
        ),
        repository_root=repository_root,
    ))

    integrated = recovery_root / "issue-15-capacity-integrated"
    integrated_summary = summaries / "issue-15-capacity-integrated-summary.json"
    components.append(_validated_regenerated(
        "issue-15-capacity-integrated",
        integrated,
        requires=(authority, reliability, integrated_summary),
        command=(
            "-m", "scripts.run_cohort_v2_integrated",
            "--design", "issue-15-capacity",
            "--release-root", str(release),
            "--reliability-root", str(reliability),
            "--output", str(integrated),
            "--compact-report", str(integrated_summary),
            "--migration-recovery", str(authority),
            "--validate",
        ),
        repository_root=repository_root,
    ))

    amendment = recovery_root / "issue-15-amendment-v2"
    amendment_authority = recovery_root / "issue-15-amendment-v2-authority"
    components.append(_validated_regenerated(
        "issue-15-amendment-v2",
        amendment,
        requires=(authority, integrated_summary, amendment_authority),
        command=(
            "-m", "scripts.run_issue_15_amendment",
            "--output", str(amendment),
            "--authority-root", str(amendment_authority),
            "--capacity-report", str(integrated_summary),
            "--migration-recovery", str(authority),
            "--validate",
        ),
        repository_root=repository_root,
    ))

    sealed = recovery_root / "issue-15-confirmatory-v2-final"
    components.append(_validated_regenerated(
        "issue-15-seed-4505-final",
        sealed,
        requires=(authority, amendment, amendment_authority),
        command=(
            "-m", "scripts.issue_15_final_collection",
            "--plan-root", str(amendment),
            "--authority-root", str(amendment_authority),
            "--sealed-root", str(sealed),
            "--migration-recovery", str(authority),
            "--validate",
        ),
        repository_root=repository_root,
    ))

    confirmatory = recovery_root / "issue-15-confirmatory-v2"
    confirmatory_summary = summaries / "issue-15-confirmatory-v2-summary.json"
    components.append(_validated_regenerated(
        "issue-15-confirmatory-v2",
        confirmatory,
        requires=(authority, sealed, confirmatory_summary),
        command=(
            "-m", "scripts.run_issue_15_confirmatory_v2",
            "--release-root", str(release),
            "--sealed-root", str(sealed),
            "--protocol-root", str(amendment),
            "--integrated-root", str(integrated),
            "--integrated-compact", str(integrated_summary),
            "--reliability-root", str(reliability),
            "--output", str(confirmatory),
            "--compact-report", str(confirmatory_summary),
            "--migration-recovery", str(authority),
            "--validate",
        ),
        repository_root=repository_root,
    ))

    feature = recovery_root / "issue-16-feature-parser"
    feature_summary = summaries / "issue-16-feature-parser-summary.json"
    components.append(_validated_regenerated(
        "issue-16-feature-parser",
        feature,
        requires=(authority, confirmatory, feature_summary),
        command=(
            "-m", "scripts.run_cohort_v2_feature_parser",
            "--release-root", str(release),
            "--sealed-root", str(sealed),
            "--protocol-root", str(amendment),
            "--integrated-root", str(integrated),
            "--integrated-compact", str(integrated_summary),
            "--reliability-root", str(reliability),
            "--oracle-output", str(confirmatory),
            "--oracle-compact", str(confirmatory_summary),
            "--output", str(feature),
            "--compact-report", str(feature_summary),
            "--migration-recovery", str(authority),
            "--validate",
        ),
        repository_root=repository_root,
    ))

    aligned = recovery_root / "issue-59-aligned-observation-release"
    aligned_summary = summaries / "issue-59-aligned-observation-summary.json"
    components.append(_validated_regenerated(
        "issue-59-aligned-observation-release",
        aligned,
        requires=(aligned_summary,),
        command=(
            "-m", "scripts.issue_59_aligned_observation_collection",
            "--output", str(aligned),
            "--summary", str(aligned_summary),
            "--validate",
        ),
        repository_root=repository_root,
    ))

    visual = recovery_root / "issue-17-visual-parser"
    visual_summary = summaries / "issue-17-visual-parser-summary.json"
    components.append(_validated_regenerated(
        "issue-17-visual-parser",
        visual,
        requires=(authority, feature, aligned, visual_summary),
        command=(
            "-m", "scripts.run_cohort_v2_visual_parser",
            "--release-root", str(release),
            "--sealed-root", str(sealed),
            "--protocol-root", str(amendment),
            "--integrated-root", str(integrated),
            "--integrated-compact", str(integrated_summary),
            "--reliability-root", str(reliability),
            "--feature-output", str(feature),
            "--feature-compact", str(feature_summary),
            "--oracle-compact", str(confirmatory_summary),
            "--aligned-root", str(aligned),
            "--output", str(visual),
            "--compact-report", str(visual_summary),
            "--migration-recovery", str(authority),
            "--validate",
        ),
        repository_root=repository_root,
    ))

    gameplay = recovery_root / "issue-56-gameplay-planner"
    components.append(_validated_regenerated(
        "issue-56-gameplay-planner",
        gameplay,
        requires=(visual, aligned, integrated),
        command=(
            "-m", "scripts.run_issue_56_gameplay_planner",
            "--output", str(gameplay),
            "--validate",
        ),
        repository_root=repository_root,
    ))

    counts = {
        status: sum(item["status"] == status for item in components)
        for status in ("present-valid", "regenerated-valid", "missing", "excluded")
    }
    return {
        "schema": SCHEMA,
        "recovery_root": recovery_root.as_posix(),
        "components": components,
        "status_counts": counts,
        "passed": all(
            item["status"] in {
                "present-valid", "regenerated-valid", "missing", "excluded"
            }
            for item in components
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--recovery-root", type=Path, default=DEFAULT_RECOVERY_ROOT)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = args.repository_root.resolve()
    recovery_root = (repository_root / args.recovery_root).resolve()
    audit = build_recovery_audit(
        repository_root=repository_root,
        recovery_root=recovery_root,
    )
    if args.output is not None:
        output = (repository_root / args.output).resolve()
        if output.exists():
            raise MigrationRecoveryAuditError("immutable recovery audit already exists")
        write_immutable_cohort_v2_json(audit, output)
    print(json.dumps(audit, allow_nan=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MigrationRecoveryAuditError, OSError, ValueError) as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(2) from error
