"""Freeze or validate issue #15's capacity-correct prospective amendment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from scripts.issue_15_amended_protocol import (
    DEFAULT_AUTHORITY_ROOT,
    DEFAULT_ROOT,
    Issue15AmendmentError,
    build_plan,
    build_protocol,
    load_frozen_bundle,
    materialize_final_authority,
    write_frozen_bundle,
)
from world_model.training.grid_artifacts import canonical_json_bytes
from world_model.training.manifest import git_revision


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--authority-root", type=Path, default=DEFAULT_AUTHORITY_ROOT)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser


def _build(authority_root: Path, implementation_commit: str):
    scenario, manifest_path, xml_path = materialize_final_authority(authority_root)
    plan = build_plan(scenario)
    protocol = build_protocol(plan, implementation_commit=implementation_commit)
    return plan, protocol, manifest_path, xml_path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if sum((args.dry_run, args.freeze, args.validate)) != 1:
        raise Issue15AmendmentError(
            "choose exactly one of --dry-run, --freeze, or --validate"
        )
    root = args.repository_root.resolve()
    output = (root / args.output).resolve()
    authority = (root / args.authority_root).resolve()
    implementation = args.implementation_commit
    if implementation is None:
        implementation, dirty = git_revision(str(root))
        if dirty and args.freeze:
            raise Issue15AmendmentError(
                "a dirty worktree requires --implementation-commit"
            )
    print("[amendment] capacity=15 latent_dim=197 final_seed=4505", flush=True)
    print("[amendment] retired seed-4504 confirmatory reuse is forbidden", flush=True)

    if args.validate:
        stored_plan, stored_protocol = load_frozen_bundle(output)
        with tempfile.TemporaryDirectory(prefix="issue-15-amendment-validate-") as temporary:
            expected_plan, expected_protocol, manifest_path, xml_path = _build(
                Path(temporary), stored_protocol["implementation_commit"]
            )
            if any(
                canonical_json_bytes(stored_plan[name])
                != canonical_json_bytes(expected_plan[name])
                for name in stored_plan
            ) or canonical_json_bytes(stored_protocol) != canonical_json_bytes(
                expected_protocol
            ):
                raise Issue15AmendmentError(
                    "stored amendment differs from exact prospective reconstruction"
                )
            if (
                (authority / "final-evaluation.json").read_bytes()
                != manifest_path.read_bytes()
                or (authority / "final-evaluation.xml").read_bytes()
                != xml_path.read_bytes()
            ):
                raise Issue15AmendmentError("sealed seed-4505 authority differs")
        print(
            f"[validate] exact amendment passed protocol={stored_protocol['artifact_identity']}",
            flush=True,
        )
        return 0

    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="issue-15-amendment-dry-") as temporary:
            plan, protocol, _manifest, _xml = _build(Path(temporary), implementation)
        print(
            f"[dry-run] attempts={len(plan['confirmatory-plan.json']['attempt_ids'])} "
            "workflow=pending files_written=false new_final_outcomes_accessed=false",
            flush=True,
        )
        print(f"[dry-run] protocol={protocol['artifact_identity']}", flush=True)
        return 0

    if output.exists() or authority.exists():
        raise Issue15AmendmentError("immutable amendment or authority already exists")
    plan, protocol, _manifest, _xml = _build(authority, implementation)
    write_frozen_bundle(plan, protocol, output)
    print(f"[freeze] plan={output / 'confirmatory-plan.json'}", flush=True)
    print(
        f"[freeze] protocol={output / 'cohort-v2-prospective-statistical-protocol-v2.json'}",
        flush=True,
    )
    print("[freeze] seed-4505 rollouts_collected=false workflow=pending", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Issue15AmendmentError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(2) from error
