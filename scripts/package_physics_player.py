#!/usr/bin/env python3
from __future__ import annotations
# noqa: SIZE_OK - deterministic package publication is a single CLI boundary.

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
from typing import Final


STAGE_SCHEMA: Final = "novphy_physics_player_stage_v1"
UNITY_VERSION: Final = "2019.4.41f2"
UNITY_CHANGESET: Final = "6b23d448b533"
CAPTURE_SCHEMA: Final = "physics_capture_v1"
CAPTURE_SCHEMAS: Final = (CAPTURE_SCHEMA, "physics_capture_v2_engine_v1")
PROTOCOL_VERSION: Final = 1
ARCHIVE_NAME: Final = "novphy-physics-player-2019.4.41f2.tar.gz"
WRAPPER_PATH: Final = "scripts/9001-player-wrapper.sh"
CONFIG_TRANSFORM: Final = "novphy_config_default_level_to_type2_3_9_6_1_v1"
UNITY_PACKAGE_INPUT_PATHS: Final = (
    "tasks/task_template_designer/Packages/manifest.json",
    "tasks/task_template_designer/Packages/packages-lock.json",
)
@dataclass(frozen=True, slots=True)
class PackagingError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class BuildSources:
    unity_executable: Path
    interface_jar: Path
    config_source: Path
    serverbackup_source: Path


@dataclass(frozen=True, slots=True)
class ManifestContext:
    worktree: Path
    build_sources: BuildSources | None
    package_inputs: dict[str, str]
    capture_schema: str = CAPTURE_SCHEMA


def payload_inventory(payload: Path) -> dict[str, int]:
    """Return the regular-file inventory and declared sizes."""
    return {
        str(path.relative_to(payload)): path.stat().st_size
        for path in sorted(payload.rglob("*"))
        if path.is_file() and path.name != "provenance.json"
    }


def unity_package_input_inventory(worktree: Path) -> dict[str, str]:
    inputs: dict[str, str] = {}
    for relative_path in UNITY_PACKAGE_INPUT_PATHS:
        path = worktree / relative_path
        if not path.is_file():
            raise PackagingError("missing provenance-bound Unity package input: " + relative_path)
        inputs[relative_path] = relative_path
    return inputs


def git_revision(worktree: Path, package_inputs: dict[str, str] | None = None, require_package_inputs: bool = False) -> tuple[str, str]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=worktree, text=True, capture_output=True, check=True).stdout.strip()
    tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=worktree, text=True, capture_output=True, check=True).stdout.strip()
    source_diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "."], cwd=worktree, check=False)
    if source_diff.returncode == 1:
        raise PackagingError("tracked product source differs from HEAD")
    source_diff.check_returncode()
    wrapper = subprocess.run(["git", "cat-file", "-e", f"HEAD:{WRAPPER_PATH}"], cwd=worktree, check=False)
    if wrapper.returncode != 0:
        raise PackagingError(f"untracked product source: {WRAPPER_PATH}")
    actual_package_inputs = unity_package_input_inventory(worktree)
    if package_inputs is None:
        if require_package_inputs:
            raise PackagingError("missing provenance-bound Unity package inputs")
        package_inputs = actual_package_inputs
    if package_inputs is not None and package_inputs != actual_package_inputs:
        raise PackagingError("Unity package inputs differ from preflight provenance")
    untracked_scope = [":(glob)scripts/*.py", ":(glob)scripts/*.sh", "tasks/task_template_designer/Assets", "tasks/task_template_designer/Packages", "tasks/task_template_designer/ProjectSettings"]
    untracked = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all", "--ignored=matching", "--", *untracked_scope], cwd=worktree, text=True, capture_output=True, check=True).stdout.splitlines()
    allowed = {f"!! {relative_path}" for relative_path in UNITY_PACKAGE_INPUT_PATHS}
    rejected = [entry for entry in untracked if entry not in allowed]
    if rejected:
        raise PackagingError("untracked product source: " + rejected[0])
    return head, tree


def build_input_manifest(payload: Path, worktree: Path, sources: BuildSources, package_inputs: dict[str, str]) -> dict[str, dict[str, str | list[str] | dict[str, str]]]:
    files = payload_inventory(payload)
    owned = {"interface_jar": "game_playing_interface.jar", "config_source": "config.xml", "serverbackup_source": "serverbackup", "player_wrapper": "9001.x86_64"}
    missing = sorted(set(owned.values()) - files.keys())
    if missing:
        raise PackagingError("missing provenance-bound payload file: " + missing[0])
    wrapper_source = worktree / WRAPPER_PATH
    wrapper_bytes = subprocess.run(["git", "show", f"HEAD:{WRAPPER_PATH}"], cwd=worktree, capture_output=True, check=True).stdout
    if wrapper_source.read_bytes() != wrapper_bytes or (payload / owned["player_wrapper"]).read_bytes() != wrapper_bytes:
        raise PackagingError("player wrapper differs from tracked source")
    if (payload / owned["interface_jar"]).read_bytes() != sources.interface_jar.read_bytes():
        raise PackagingError("interface jar differs from payload copy")
    if (payload / owned["serverbackup_source"]).read_bytes() != sources.serverbackup_source.read_bytes():
        raise PackagingError("serverbackup differs from payload copy")
    blob = subprocess.run(["git", "rev-parse", f"HEAD:{WRAPPER_PATH}"], cwd=worktree, text=True, capture_output=True, check=True).stdout.strip()
    generated = sorted(files.keys() - set(owned.values()))
    return {
        "unity_executable": {"source_path": str(sources.unity_executable)},
        "interface_jar": {"source_path": str(sources.interface_jar), "payload_files": [owned["interface_jar"]]},
        "config_source": {"source_path": str(sources.config_source), "transform": CONFIG_TRANSFORM, "payload_files": [owned["config_source"]]},
        "serverbackup_source": {"source_path": str(sources.serverbackup_source), "payload_files": [owned["serverbackup_source"]]},
        "player_wrapper": {"repository_path": WRAPPER_PATH, "git_blob": blob, "payload_files": [owned["player_wrapper"]]},
        "unity_package_inputs": {"files": package_inputs},
        "generated_unity_outputs": {"generator": "unity_executable", "payload_files": generated},
    }


def write_manifest(payload: Path, context: ManifestContext) -> Path:
    head, tree = git_revision(context.worktree, context.package_inputs, require_package_inputs=True)
    manifest = {
        "schema_version": STAGE_SCHEMA,
        "unity": {"canonical_revision": "2019.3.4f1 (4f139db2fdbd)", "migrated_revision": f"{UNITY_VERSION} ({UNITY_CHANGESET})", "version": UNITY_VERSION, "changeset": UNITY_CHANGESET},
        "project": {"git_head": head, "git_tree": tree},
        "capture": {"schema_version": context.capture_schema, "protocol_version": PROTOCOL_VERSION},
        "migration": {"issue": "#44"},
        "rollback_rule": "Build or verification failure never modifies canonical project, production player, or active data root.",
        "files": payload_inventory(payload),
    }
    if context.build_sources is not None:
        manifest["build_inputs"] = build_input_manifest(payload, context.worktree, context.build_sources, context.package_inputs)
    path = payload / "provenance.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_archive(archive: Path, expected_files: dict[str, int], require_manifest: bool) -> None:
    actual: dict[str, int] = {}
    manifest_bytes: bytes | None = None
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                raise PackagingError("archive contains non-regular payload member")
            stream = bundle.extractfile(member)
            if stream is None:
                raise PackagingError("archive regular member cannot be read")
            with stream:
                if member.name == "provenance.json":
                    manifest_bytes = stream.read()
                    continue
                stream.read()
                actual[member.name] = member.size
    if actual != expected_files:
        raise PackagingError("archive payload differs from source payload")
    if require_manifest and manifest_bytes is None:
        raise PackagingError("archive provenance manifest is missing")
    if manifest_bytes is not None:
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
            raise PackagingError("archive provenance manifest is malformed")
        if manifest["files"] != actual:
            raise PackagingError("archive provenance payload inventory differs")


def temporary_sibling(final: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{final.name}.", suffix=".tmp", dir=final.parent)
    os.close(descriptor)
    return Path(name)


def write_fsynced(path: Path, data: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def write_archive(payload: Path, archive: Path) -> None:
    expected_files = payload_inventory(payload)
    with tempfile.TemporaryDirectory() as temporary:
        raw_tar = Path(temporary) / "payload.tar"
        with tarfile.open(raw_tar, "w", format=tarfile.PAX_FORMAT) as bundle:
            for path in sorted(payload.rglob("*")):
                info = bundle.gettarinfo(str(path), arcname=str(path.relative_to(payload)))
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                if path.is_file():
                    with path.open("rb") as stream:
                        bundle.addfile(info, stream)
                else:
                    bundle.addfile(info)
        with archive.open("wb") as output:
            subprocess.run(["gzip", "-n", "-9", "-c", str(raw_tar)], stdout=output, check=True)
            output.flush()
            os.fsync(output.fileno())
    validate_archive(archive, expected_files, (payload / "provenance.json").is_file())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_archive(payload: Path, archive: Path) -> None:
    temporary_archive = temporary_sibling(archive)
    try:
        write_archive(payload, temporary_archive)
        os.replace(temporary_archive, archive)
        fsync_directory(archive.parent)
    finally:
        temporary_archive.unlink(missing_ok=True)


def publish(payload: Path, stage: Path) -> None:
    archive = stage / ARCHIVE_NAME
    archive_temporary = temporary_sibling(archive)
    try:
        write_archive(payload, archive_temporary)
        os.replace(archive_temporary, archive)
        fsync_directory(stage)
    finally:
        archive_temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--unity-executable", type=Path)
    parser.add_argument("--interface-jar", type=Path)
    parser.add_argument("--config-source", type=Path)
    parser.add_argument("--serverbackup-source", type=Path)
    parser.add_argument("--check-worktree-only", action="store_true")
    parser.add_argument("--write-package-inputs", type=Path)
    parser.add_argument("--package-inputs", type=Path)
    parser.add_argument("--capture-schema", choices=CAPTURE_SCHEMAS, default=CAPTURE_SCHEMA)
    args = parser.parse_args()
    if args.check_worktree_only:
        package_inputs = unity_package_input_inventory(args.worktree)
        git_revision(args.worktree, package_inputs, require_package_inputs=True)
        if args.write_package_inputs is not None:
            args.write_package_inputs.write_text(json.dumps(package_inputs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    if args.write_package_inputs is not None:
        raise PackagingError("package input snapshot is only valid with worktree preflight")
    if any(path is None for path in (args.unity_executable, args.interface_jar, args.config_source, args.serverbackup_source)) and any(path is not None for path in (args.unity_executable, args.interface_jar, args.config_source, args.serverbackup_source)):
        raise PackagingError("build source options must be provided together")
    sources = None
    if args.unity_executable is not None:
        sources = BuildSources(args.unity_executable, args.interface_jar, args.config_source, args.serverbackup_source)
    if args.package_inputs is None:
        package_inputs = unity_package_input_inventory(args.worktree)
    else:
        raw_package_inputs = json.loads(args.package_inputs.read_text(encoding="utf-8"))
        if not isinstance(raw_package_inputs, dict) or not all(isinstance(path, str) and isinstance(reference, str) for path, reference in raw_package_inputs.items()):
            raise PackagingError("Unity package input provenance is malformed")
        package_inputs = raw_package_inputs
    context = ManifestContext(args.worktree, sources, package_inputs, args.capture_schema)
    write_manifest(args.payload, context)
    args.stage.mkdir(parents=True, exist_ok=True)
    publish(args.payload, args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
