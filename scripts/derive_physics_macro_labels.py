#!/usr/bin/env python3
"""Derive `physics_macro_labels_v1` sidecars for a shot or a root of shots.

Read-only with respect to the frozen capture sidecars: this writes exactly one new
file per accepted shot and never touches `frames/` or `metadata.json`.  Refuses to
write anywhere inside the active legacy cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

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
    write_macro_labels,
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
    # Refuse a protected write destination before scanning it, so the guard does not
    # depend on whether the scan happens to find shots there.
    if not args.validate_only:
        _refuse_active_cohort(args.output_dir if args.output_dir is not None else args.target)

    shots = discover_shots(args.target)
    if not shots:
        print(json.dumps({"error": f"no physics shots found under {args.target}"}), file=sys.stderr)
        return 2

    entries: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for shot in shots:
        label_path = _label_path(shot, args.target, args.output_dir)
        try:
            if not args.validate_only:
                if args.output_dir is not None:
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    write_macro_label_file(derive_macro_labels_for_shot(shot), label_path)
                else:
                    label_path = write_macro_labels(shot)
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
