#!/usr/bin/env python3
"""Derive `physics_macro_labels_v1` sidecars for a shot or a root of shots.

Fixture-only write boundary: write mode REQUIRES `--output-dir`, and the resolved
destination must be the system temporary directory or a descendant of it.  Before
any file is written, every ancestor of the resolved destination (up to the
temporary root, never higher) is scanned for physics-capture records
(`physics_state.jsonl`, `physics_events.jsonl`, or an episode `manifest.json`), so
even a sidecar-free subdirectory of any real cohort is refused; each computed
label parent must also not already hold frozen sidecars.  Only sidecar-free
temporary mirror trees are writable.  This never touches the frozen sidecars,
`frames/`, or `metadata.json`, and refuses to write anywhere inside the active
legacy cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.physics_capture_contract import EVENT_SIDECAR, STATE_SIDECAR  # noqa: E402
from scripts.physics_macro_labels import (  # noqa: E402
    DERIVATION_SPEC_VERSION,
    MACRO_LABEL_SCHEMA_VERSION,
    MACRO_LABEL_SIDECAR,
    MacroLabelError,
    derivation_spec_digest,
    derive_macro_labels_for_shot,
    validate_macro_labels,
    write_macro_label_file,
)
from scripts.prepare_rollout_dataset import ACTIVE_COHORT_ROOT  # noqa: E402


def _is_shot(path: Path) -> bool:
    return (path / STATE_SIDECAR).is_file() and (path / EVENT_SIDECAR).is_file()


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
        raise SystemExit(f"refusing to write macro labels inside the active cohort: {resolved}")
    try:
        active = ACTIVE_COHORT_ROOT.resolve(strict=True)
    except OSError:
        return
    if resolved == active or active in resolved.parents:
        raise SystemExit(f"refusing to write macro labels inside the active cohort: {resolved}")


#: Capture-record markers used by the capture-tree containment guard: the two frozen
#: sidecars mark a shot directory, `manifest.json` marks an episode directory.
CAPTURE_TREE_MARKERS = (STATE_SIDECAR, EVENT_SIDECAR, "manifest.json")
#: How far below each destination ancestor the containment scan descends.  Three
#: directory levels cover the shot (<cohort>/shot_001), episode
#: (<cohort>/episode_001/shot_001), and split (<cohort>/<split>/episode_001/shot_001)
#: layouts, whose marker files sit inside a directory three levels down.
CAPTURE_SCAN_DEPTH = 3


def _find_capture_marker(root: Path, max_depth: int) -> Path | None:
    """Return the first capture marker in a directory within `max_depth` levels below `root`.

    Symlinked entries are skipped; unreadable or missing directories are tolerated.
    """
    pending: list[tuple[Path, int]] = [(root, 0)]
    while pending:
        directory, depth = pending.pop()
        try:
            children = sorted(directory.iterdir())
        except OSError:
            continue
        for child in children:
            try:
                if child.is_symlink():
                    continue
                if child.is_dir():
                    if depth < max_depth:
                        pending.append((child, depth + 1))
                elif child.name in CAPTURE_TREE_MARKERS:
                    return child
            except OSError:
                continue
    return None


def _capture_marker_on_destination_path(resolved_output: Path, temporary_root: Path) -> Path | None:
    """Return the first capture marker in any destination ancestor tree, or None.

    The walk covers the destination itself and every ancestor below the temporary
    root, stopping at the temporary root and never going higher: the temporary root
    is shared scratch space whose deep sibling trees are out of scope, so only its
    immediate files are checked -- a stale capture copy left on the destination's
    own path inside the temporary root still trips the guard deliberately.
    """
    for ancestor in (resolved_output, *resolved_output.parents):
        if ancestor == temporary_root:
            return _find_capture_marker(ancestor, 0)
        marker = _find_capture_marker(ancestor, CAPTURE_SCAN_DEPTH)
        if marker is not None:
            return marker
    return None


def _label_path(shot: Path, target: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return shot / MACRO_LABEL_SIDECAR
    # When `target` is itself a shot, shot.relative_to(target) is "." and collapses.
    return output_dir / shot.relative_to(target) / MACRO_LABEL_SIDECAR


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True, help="A shot directory or a root containing shot directories")
    parser.add_argument("--output-dir", type=Path, default=None, help="Write/validate label files under this root instead of inside each shot")
    parser.add_argument("--validate-only", action="store_true", help="Re-derive and compare without writing")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not args.target.exists():
        print(json.dumps({"error": f"target does not exist: {args.target}"}), file=sys.stderr)
        return 2
    # Refuse protected targets/destinations before scanning them, so the guards do
    # not depend on whether the scan happens to find shots there.
    if not args.validate_only:
        if args.output_dir is None:
            print(
                json.dumps({"error": "write mode requires --output-dir: in-shot writes are not permitted, label files never live beside frozen sidecars"}),
                file=sys.stderr,
            )
            return 2
        # Positive authorization, not heuristics: only the system temporary
        # directory or a descendant of it is a writable destination.
        resolved_output = args.output_dir.resolve(strict=False)
        temporary_root = Path(tempfile.gettempdir()).resolve()
        if resolved_output != temporary_root and temporary_root not in resolved_output.parents:
            print(
                json.dumps({"error": f"write destination must be the system temporary directory or a descendant of it: {resolved_output}"}),
                file=sys.stderr,
            )
            return 2
        _refuse_active_cohort(args.target)
        _refuse_active_cohort(args.output_dir)
        # Capture-tree containment: a sidecar-free mirror subdirectory of a real
        # cohort has no sidecars in its label parents, but its ancestor trees do.
        marker = _capture_marker_on_destination_path(resolved_output, temporary_root)
        if marker is not None:
            print(
                json.dumps({"error": f"refusing to write macro labels: destination tree contains physics capture records: {marker}"}),
                file=sys.stderr,
            )
            return 2

    shots = discover_shots(args.target)
    if not shots:
        print(json.dumps({"error": f"no physics shots found under {args.target}"}), file=sys.stderr)
        return 2

    if not args.validate_only:
        # Preflight (before any file is written): a destination whose parent already
        # holds frozen capture sidecars is a shot/cohort directory, not a mirror
        # tree; refuse the whole run so no label file is ever placed there.
        for shot in shots:
            destination_parent = _label_path(shot, args.target, args.output_dir).parent
            if (destination_parent / STATE_SIDECAR).exists() or (destination_parent / EVENT_SIDECAR).exists():
                print(
                    json.dumps({"error": f"refusing to write macro labels into a directory holding frozen capture sidecars: {destination_parent}"}),
                    file=sys.stderr,
                )
                return 2

    entries: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for shot in shots:
        label_path = _label_path(shot, args.target, args.output_dir)
        try:
            if not args.validate_only:
                label_path.parent.mkdir(parents=True, exist_ok=True)
                write_macro_label_file(derive_macro_labels_for_shot(shot), label_path)
            stored = validate_macro_labels(shot, label_path)
            entries.append(
                {
                    "shot": str(shot),
                    "label_path": str(label_path),
                    "sha256": _sha256_file(label_path),
                    "state_count": len(stored.frames),
                    "event_count": stored.event_count,
                    "interval_count": len(stored.intervals),
                    "outcome_class": stored.outcome.outcome_class.value,
                }
            )
        except (OSError, MacroLabelError, ValueError) as error:
            failures.append({"shot": str(shot), "error": str(error)})

    report = {
        "target": str(args.target),
        "output_dir": None if args.output_dir is None else str(args.output_dir),
        "sidecar": MACRO_LABEL_SIDECAR,
        "schema_version": MACRO_LABEL_SCHEMA_VERSION,
        "derivation_spec_version": DERIVATION_SPEC_VERSION,
        "derivation_spec_digest": derivation_spec_digest(),
        "mode": "validate" if args.validate_only else "write",
        "shots_total": len(shots),
        "shots_ok": len(entries),
        "shots_failed": len(failures),
        "shots": entries,
        "failures": failures,
    }
    if args.json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
