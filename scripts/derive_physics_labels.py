#!/usr/bin/env python3
"""Derive `physics_derived_labels_v1` sidecars for a shot, episode, or cohort root.

Read-only with respect to the frozen capture sidecars: this writes exactly one new
file per accepted shot and never touches `frames/` or `metadata.json`.  Refuses to
write anywhere inside the active legacy cohort.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.physics_label_derivation import (  # noqa: E402
    DERIVED_LABEL_SIDECAR,
    DerivedLabelError,
    OracleGateSpec,
    validate_derived_labels,
    write_derived_labels,
)
from scripts.prepare_rollout_dataset import ACTIVE_COHORT_ROOT  # noqa: E402

STATE_SIDECAR = "physics_state.jsonl"


def _is_shot(path: Path) -> bool:
    return (path / STATE_SIDECAR).is_file() and (path / "physics_events.jsonl").is_file()


def discover_shots(target: Path) -> list[Path]:
    """Return every accepted physics shot at or beneath `target`, deterministically."""
    if _is_shot(target):
        return [target]
    return sorted(
        candidate.parent
        for candidate in target.rglob(STATE_SIDECAR)
        if candidate.is_file() and not candidate.is_symlink() and _is_shot(candidate.parent)
    )


#: The active legacy cohort is protected wherever it lives.  `ACTIVE_COHORT_ROOT` is
#: derived from the running checkout, so from the physics worktree it names a path that
#: does not exist; matching on the directory name as well keeps the guard effective in
#: every checkout rather than silently passing.
ACTIVE_COHORT_DIR_NAME = ACTIVE_COHORT_ROOT.name


def _refuse_active_cohort(target: Path) -> None:
    resolved = target.resolve(strict=False)
    if ACTIVE_COHORT_DIR_NAME in resolved.parts:
        raise SystemExit(f"refusing to write derived labels inside the active cohort: {resolved}")
    try:
        active = ACTIVE_COHORT_ROOT.resolve(strict=True)
    except OSError:
        return
    if resolved == active or active in resolved.parents:
        raise SystemExit(f"refusing to write derived labels inside the active cohort: {resolved}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True, help="A shot, episode, split, or cohort root")
    parser.add_argument("--kinetic-energy-threshold", type=float, default=OracleGateSpec().kinetic_energy_threshold)
    parser.add_argument("--active-contact-threshold", type=int, default=OracleGateSpec().active_contact_threshold)
    parser.add_argument("--contact-activity-speed", type=float, default=OracleGateSpec().contact_activity_speed)
    parser.add_argument("--validate-only", action="store_true", help="Re-derive and compare without writing")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        spec = OracleGateSpec(
            kinetic_energy_threshold=args.kinetic_energy_threshold,
            active_contact_threshold=args.active_contact_threshold,
            contact_activity_speed=args.contact_activity_speed,
        )
    except ValueError as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2

    if not args.target.exists():
        print(json.dumps({"error": f"target does not exist: {args.target}"}), file=sys.stderr)
        return 2
    # Refuse a protected target before scanning it, so the guard does not depend on
    # whether the scan happens to find shots there.
    if not args.validate_only:
        _refuse_active_cohort(args.target)

    shots = discover_shots(args.target)
    if not shots:
        print(json.dumps({"error": f"no physics shots found under {args.target}"}), file=sys.stderr)
        return 2

    written: list[str] = []
    failures: list[dict[str, str]] = []
    for shot in shots:
        try:
            if args.validate_only:
                validate_derived_labels(shot, spec)
            else:
                write_derived_labels(shot, spec)
                validate_derived_labels(shot, spec)
            written.append(str(shot))
        except (OSError, DerivedLabelError, ValueError) as error:
            failures.append({"shot": str(shot), "error": str(error)})

    report = {
        "target": str(args.target),
        "sidecar": DERIVED_LABEL_SIDECAR,
        "oracle_gate_spec": spec.to_json(),
        "oracle_gate_spec_digest": spec.digest(),
        "mode": "validate" if args.validate_only else "write",
        "shots_total": len(shots),
        "shots_ok": len(written),
        "shots_failed": len(failures),
        "failures": failures,
    }
    print(json.dumps(report, sort_keys=True) if args.json else json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
