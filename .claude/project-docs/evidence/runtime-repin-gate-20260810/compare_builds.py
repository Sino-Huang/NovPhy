#!/usr/bin/env python
"""Compare two isolated physics-player build stages for byte determinism.

The mission requires identical archive, player, `Assembly-CSharp.dll`,
package-input, and provenance digests across two builds of the exact same
committed source. This emits a machine-readable receipt of that comparison and
exits non-zero on any drift.

`unity-build.log` is deliberately excluded: it is Unity's own diagnostic output,
carries wall-clock timestamps and temporary build paths by construction, and is
published beside the archive rather than inside it. Excluding it is stated here
rather than left implicit, because a silent exclusion would be indistinguishable
from an unnoticed difference.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

ARCHIVE_NAME = "novphy-physics-player-2019.4.41f2.tar.gz"
# Files whose digests the gate names explicitly, beyond the whole-archive digest.
NAMED_MEMBERS = (
    "9001-player.x86_64",
    "9001_Data/Managed/Assembly-CSharp.dll",
    "UnityPlayer.so",
    "provenance.json",
)


def _digest(path: Path) -> str:
    """Return the SHA-256 of a file, reading it in bounded chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _member_digests(archive: Path, names: tuple[str, ...]) -> dict[str, str]:
    """Return SHA-256 per named archive member, failing closed on absence."""
    digests: dict[str, str] = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for name in names:
            try:
                member = bundle.extractfile(name)
            except KeyError as error:
                raise SystemExit(f"{archive}: required member {name} is absent: {error}") from error
            if member is None:
                raise SystemExit(f"{archive}: required member {name} is not a regular file")
            with member:
                hasher = hashlib.sha256()
                for chunk in iter(lambda: member.read(1 << 20), b""):
                    hasher.update(chunk)
            digests[name] = hasher.hexdigest()
    return digests


def _provenance(archive: Path) -> dict[str, Any]:
    """Return the parsed provenance record carried inside an archive."""
    with tarfile.open(archive, "r:gz") as bundle:
        member = bundle.extractfile("provenance.json")
        if member is None:
            raise SystemExit(f"{archive}: provenance.json is absent")
        with member:
            return json.loads(member.read().decode("utf-8"))


def describe(stage: Path) -> dict[str, Any]:
    """Summarize one build stage into comparable digest sets."""
    archive = stage / ARCHIVE_NAME
    if not archive.is_file():
        raise SystemExit(f"{archive} is absent; the build did not publish an archive")
    provenance = _provenance(archive)
    recorded = provenance.get("files")
    if not isinstance(recorded, dict):
        raise SystemExit(f"{archive}: provenance.files is missing or not an object")
    return {
        "stage": str(stage),
        "archive_sha256": _digest(archive),
        "archive_sha256_receipt": (stage / "archive.sha256").read_text(encoding="utf-8").strip(),
        "members": _member_digests(archive, NAMED_MEMBERS),
        "provenance_file_digests": recorded,
        "provenance_build_inputs": provenance.get("build_inputs"),
        "provenance_unity": provenance.get("unity"),
        "provenance_project": provenance.get("project"),
    }


def compare(first: dict[str, Any], second: dict[str, Any]) -> list[str]:
    """Return one message per differing field; empty means byte-identical."""
    drift: list[str] = []
    for field in ("archive_sha256", "archive_sha256_receipt", "provenance_build_inputs", "provenance_unity", "provenance_project"):
        if first[field] != second[field]:
            drift.append(f"{field}: {first[field]!r} != {second[field]!r}")
    for name in NAMED_MEMBERS:
        if first["members"][name] != second["members"][name]:
            drift.append(f"member {name}: {first['members'][name]} != {second['members'][name]}")
    left, right = first["provenance_file_digests"], second["provenance_file_digests"]
    for name in sorted(set(left) | set(right)):
        if left.get(name) != right.get(name):
            drift.append(f"provenance.files[{name}]: {left.get(name)} != {right.get(name)}")
    return drift


def main(argv: list[str]) -> int:
    """Compare two stages and print a machine-readable receipt."""
    if len(argv) != 4:
        print("usage: compare_builds.py <stage-a> <stage-b> <receipt.json>", file=sys.stderr)
        return 2
    first, second = describe(Path(argv[1])), describe(Path(argv[2]))
    drift = compare(first, second)
    receipt = {
        "schema": "novphy_deterministic_build_comparison_v1",
        "excluded": {"unity-build.log": "Unity diagnostic output; carries wall-clock timestamps and temporary build paths, and is published beside the archive rather than inside it."},
        "named_members": list(NAMED_MEMBERS),
        "compared_provenance_file_count": len(first["provenance_file_digests"]),
        "build_a": {k: v for k, v in first.items() if k != "provenance_file_digests"},
        "build_b": {k: v for k, v in second.items() if k != "provenance_file_digests"},
        "drift": drift,
        "deterministic": not drift,
    }
    Path(argv[3]).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"deterministic": not drift, "drift_count": len(drift), "receipt": argv[3]}, indent=2))
    for message in drift:
        print(f"DRIFT {message}", file=sys.stderr)
    return 0 if not drift else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
