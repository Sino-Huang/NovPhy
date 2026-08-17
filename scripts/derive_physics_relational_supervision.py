#!/usr/bin/env python3
"""Derive ``physics_relational_supervision_v1`` sidecars for physics shots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.physics_capture_contract import EVENT_SIDECAR, STATE_SIDECAR
from scripts.physics_relational_supervision import (
    RELATIONAL_SUPERVISION_SIDECAR,
    RelationalSupervisionError,
    validate_relational_supervision,
    write_relational_supervision,
    write_relational_supervision_file,
    derive_relational_supervision_for_shot,
)


def _is_shot(path: Path) -> bool:
    return (path / STATE_SIDECAR).is_file() and (path / EVENT_SIDECAR).is_file()


def discover_shots(target: Path) -> list[Path]:
    if _is_shot(target):
        return [target]
    return sorted(
        candidate.parent
        for candidate in target.rglob(STATE_SIDECAR)
        if candidate.is_file() and not candidate.is_symlink() and _is_shot(candidate.parent)
    )


def _destination(shot: Path, target: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return shot / RELATIONAL_SUPERVISION_SIDECAR
    relative = Path(".") if shot == target else shot.relative_to(target)
    return output_dir / relative / RELATIONAL_SUPERVISION_SIDECAR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True, help="shot directory or root containing shots")
    parser.add_argument("--output-dir", type=Path, default=None, help="optional mirror root for generated sidecars")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    shots = discover_shots(args.target)
    if not shots:
        print(json.dumps({"error": f"no physics shots found under {args.target}"}))
        return 2

    results: list[dict[str, object]] = []
    try:
        for shot in shots:
            destination = _destination(shot, args.target, args.output_dir)
            if args.validate_only:
                labels = validate_relational_supervision(shot, destination)
            elif args.output_dir is None:
                write_relational_supervision(shot)
                labels = validate_relational_supervision(shot)
            else:
                labels = derive_relational_supervision_for_shot(shot)
                destination.parent.mkdir(parents=True, exist_ok=True)
                write_relational_supervision_file(labels, destination)
            results.append(
                {
                    "shot": str(shot),
                    "sidecar": str(destination),
                    "state_count": len(labels.frames),
                    "event_count": labels.event_count,
                }
            )
    except (OSError, RelationalSupervisionError) as error:
        print(json.dumps({"error": str(error)}))
        return 2

    if args.json:
        print(json.dumps({"shots": results}, sort_keys=True, separators=(",", ":")))
    else:
        for result in results:
            print(result["sidecar"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
