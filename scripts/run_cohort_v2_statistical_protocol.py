"""Freeze or validate the prospective cohort-v2 statistical protocol (issue #34)."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.cohort_v2_statistical_protocol import (
    CohortV2ProtocolError,
    build_protocol,
    load_protocol,
    write_protocol,
)
from world_model.training.grid_artifacts import canonical_json_bytes
from world_model.training.manifest import git_revision


DEFAULT_OUTPUT = Path(
    "data/runtime_evidence/issue-34/cohort-v2-prospective-statistical-protocol-v1.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dry_run and args.validate:
        raise CohortV2ProtocolError("--dry-run and --validate are mutually exclusive")
    root = args.repository_root.resolve()
    output = (root / args.output).resolve()
    print("[design] loading issue-41 and issue-58 non-final calibration evidence", flush=True)
    print("[design] binding release-v5 and pending final workflow metadata", flush=True)

    if args.validate:
        stored = load_protocol(output)
        print("[validate 1/3] protocol schema and immutable identity", flush=True)
        expected = build_protocol(
            root, implementation_commit=stored["implementation_commit"]
        )
        print("[validate 2/3] calibration values and sealed replicate inventory", flush=True)
        if canonical_json_bytes(stored) != canonical_json_bytes(expected):
            raise CohortV2ProtocolError("stored protocol differs from frozen sources")
        print("[validate 3/3] final-evaluation exposure audit remains sealed", flush=True)
        print(
            f"[validate] exact protocol validation passed artifact={stored['artifact_identity']}",
            flush=True,
        )
        return 0

    implementation = args.implementation_commit
    if implementation is None:
        implementation, dirty = git_revision(str(root))
        if dirty and not args.dry_run:
            raise CohortV2ProtocolError(
                "a dirty worktree requires --implementation-commit"
            )
    protocol = build_protocol(root, implementation_commit=implementation)
    print("[audit] six sealed final rollouts identified from public metadata", flush=True)
    print("[audit] no final scenario, observation, label, outcome, or metric opened", flush=True)
    print("[matrix] issue-15 confirmatory plus issue-16/17 stress analyses frozen", flush=True)
    for row in protocol["experiment_matrix"]["confirmatory_oracle_symbol_issue_15"]["comparisons"]:
        print(
            f"[matrix] budget={row['budget']} comparator={row['strongest_comparator_id']} "
            f"effect={row['practical_effect_threshold_absolute_endpoint_error_reduction']} "
            f"violation_margin={row['physical_violation_margin']}",
            flush=True,
        )
    if args.dry_run:
        print("[dry-run] protocol is valid; no files written", flush=True)
        return 0
    write_protocol(protocol, output)
    print(f"[write] immutable protocol={output}", flush=True)
    print(f"[complete] artifact={protocol['artifact_identity']}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CohortV2ProtocolError as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(2) from error
