#!/usr/bin/env python3
"""Derive ``physics_relational_supervision_v1`` sidecars for physics shots.

Write mode is restricted to sidecar-free mirror trees beneath the system
temporary directory. Validate-only mode may also read an in-shot sidecar.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.physics_capture_contract import EVENT_SIDECAR, STATE_SIDECAR  # noqa: E402
from scripts.physics_relational_supervision import (  # noqa: E402
    RELATIONAL_SUPERVISION_SIDECAR,
    RelationalSupervisionError,
    validate_relational_supervision,
    write_relational_supervision_file,
    derive_relational_supervision_for_shot,
)
from scripts.prepare_rollout_dataset import ACTIVE_COHORT_ROOT  # noqa: E402


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


ACTIVE_COHORT_DIR_NAME = ACTIVE_COHORT_ROOT.name
CAPTURE_TREE_MARKERS = (STATE_SIDECAR, EVENT_SIDECAR, "manifest.json")


def _refuse_active_cohort(target: Path) -> None:
    resolved = target.resolve(strict=False)
    if ACTIVE_COHORT_DIR_NAME in resolved.parts:
        raise SystemExit(f"refusing to write relational supervision inside the active cohort: {resolved}")
    try:
        active = ACTIVE_COHORT_ROOT.resolve(strict=True)
    except OSError:
        return
    if resolved == active or active in resolved.parents:
        raise SystemExit(f"refusing to write relational supervision inside the active cohort: {resolved}")


def _find_capture_marker(root: Path) -> Path | None:
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            children = sorted(directory.iterdir())
        except OSError:
            continue
        for child in children:
            try:
                if child.is_symlink():
                    continue
                if child.is_dir():
                    pending.append(child)
                elif child.name in CAPTURE_TREE_MARKERS:
                    return child
            except OSError:
                continue
    return None


def _immediate_capture_marker(root: Path) -> Path | None:
    try:
        children = sorted(root.iterdir())
    except OSError:
        return None
    for child in children:
        try:
            if not child.is_symlink() and child.is_file() and child.name in CAPTURE_TREE_MARKERS:
                return child
        except OSError:
            continue
    return None


def _capture_marker_on_destination_path(
    resolved_output: Path,
    temporary_root: Path,
) -> Path | None:
    for ancestor in (resolved_output, *resolved_output.parents):
        if ancestor == temporary_root:
            return _immediate_capture_marker(ancestor)
        marker = _find_capture_marker(ancestor)
        if marker is not None:
            return marker
    return None


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

    if not args.target.exists():
        print(json.dumps({"error": f"target does not exist: {args.target}"}), file=sys.stderr)
        return 2
    if not args.validate_only:
        if args.output_dir is None:
            print(
                json.dumps(
                    {
                        "error": "write mode requires --output-dir: in-shot writes are not permitted"
                    }
                ),
                file=sys.stderr,
            )
            return 2
        resolved_output = args.output_dir.resolve(strict=False)
        temporary_root = Path(tempfile.gettempdir()).resolve()
        if resolved_output != temporary_root and temporary_root not in resolved_output.parents:
            print(
                json.dumps(
                    {
                        "error": "write destination must be the system temporary directory "
                        f"or a descendant of it: {resolved_output}"
                    }
                ),
                file=sys.stderr,
            )
            return 2
        _refuse_active_cohort(args.target)
        _refuse_active_cohort(args.output_dir)
        marker = _capture_marker_on_destination_path(resolved_output, temporary_root)
        if marker is not None:
            print(
                json.dumps(
                    {
                        "error": "refusing to write relational supervision: destination tree "
                        f"contains physics capture records: {marker}"
                    }
                ),
                file=sys.stderr,
            )
            return 2

    shots = discover_shots(args.target)
    if not shots:
        print(json.dumps({"error": f"no physics shots found under {args.target}"}), file=sys.stderr)
        return 2

    if not args.validate_only:
        for shot in shots:
            destination_parent = _destination(shot, args.target, args.output_dir).parent
            if (destination_parent / STATE_SIDECAR).exists() or (
                destination_parent / EVENT_SIDECAR
            ).exists():
                print(
                    json.dumps(
                        {
                            "error": "refusing to write relational supervision into a directory "
                            f"holding frozen capture sidecars: {destination_parent}"
                        }
                    ),
                    file=sys.stderr,
                )
                return 2

    results: list[dict[str, object]] = []
    try:
        for shot in shots:
            destination = _destination(shot, args.target, args.output_dir)
            if args.validate_only:
                labels = validate_relational_supervision(shot, destination)
            else:
                labels = derive_relational_supervision_for_shot(shot)
                destination.parent.mkdir(parents=True, exist_ok=True)
                write_relational_supervision_file(labels, destination)
                labels = validate_relational_supervision(shot, destination)
            results.append(
                {
                    "shot": str(shot),
                    "sidecar": str(destination),
                    "state_count": len(labels.frames),
                    "event_count": labels.event_count,
                }
            )
    except (OSError, RelationalSupervisionError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"shots": results}, sort_keys=True, separators=(",", ":")))
    else:
        for result in results:
            print(result["sidecar"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
