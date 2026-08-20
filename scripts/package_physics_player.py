#!/usr/bin/env python3
from __future__ import annotations
# noqa: SIZE_OK - deterministic package publication is a single CLI boundary.

import argparse
from dataclasses import dataclass
import hashlib
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
APPROVED_UNITY_PACKAGE_INPUT_DIGESTS: Final = {
    UNITY_PACKAGE_INPUT_PATHS[0]: "05677cc3199d5fff4aac54096877e795518487918e53810f477a228e5d1e28fb",
    UNITY_PACKAGE_INPUT_PATHS[1]: "3101c351984e6a73a1be7ad76d1a67c1b7638a6616554e50000b9672175ebe50",
}


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
    migration_provenance: Path
    build_sources: BuildSources | None
    package_inputs: dict[str, str]
    capture_schema: str = CAPTURE_SCHEMA


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_hashes(payload: Path) -> dict[str, str]:
    return {str(path.relative_to(payload)): sha256_file(path) for path in sorted(payload.rglob("*")) if path.is_file() and path.name != "provenance.json"}


def unity_package_input_digests(worktree: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    for relative_path in UNITY_PACKAGE_INPUT_PATHS:
        path = worktree / relative_path
        if not path.is_file():
            raise PackagingError("missing provenance-bound Unity package input: " + relative_path)
        digests[relative_path] = sha256_file(path)
    return digests


def git_revision(worktree: Path, package_inputs: dict[str, str] | None = None, require_package_inputs: bool = False) -> tuple[str, str]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=worktree, text=True, capture_output=True, check=True).stdout.strip()
    tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=worktree, text=True, capture_output=True, check=True).stdout.strip()
    source_diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", ".", ":(exclude).claude", ":(exclude).claude/**"], cwd=worktree, check=False)
    if source_diff.returncode == 1:
        raise PackagingError("tracked product source differs from HEAD")
    source_diff.check_returncode()
    wrapper = subprocess.run(["git", "cat-file", "-e", f"HEAD:{WRAPPER_PATH}"], cwd=worktree, check=False)
    if wrapper.returncode != 0:
        raise PackagingError(f"untracked product source: {WRAPPER_PATH}")
    actual_package_inputs = unity_package_input_digests(worktree)
    if package_inputs is None:
        if require_package_inputs:
            raise PackagingError("missing provenance-bound Unity package input digests")
        package_inputs = actual_package_inputs
    if package_inputs is not None and package_inputs != actual_package_inputs:
        raise PackagingError("Unity package input digest differs from preflight provenance")
    if actual_package_inputs != APPROVED_UNITY_PACKAGE_INPUT_DIGESTS:
        raise PackagingError("Unity package input digest differs from approved provenance")
    untracked_scope = [":(glob)scripts/*.py", ":(glob)scripts/*.sh", "tasks/task_template_designer/Assets", "tasks/task_template_designer/Packages", "tasks/task_template_designer/ProjectSettings"]
    untracked = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all", "--ignored=matching", "--", *untracked_scope], cwd=worktree, text=True, capture_output=True, check=True).stdout.splitlines()
    allowed = {f"!! {relative_path}" for relative_path in UNITY_PACKAGE_INPUT_PATHS}
    rejected = [entry for entry in untracked if entry not in allowed]
    if rejected:
        raise PackagingError("untracked product source: " + rejected[0])
    return head, tree


def build_input_manifest(payload: Path, worktree: Path, sources: BuildSources, package_inputs: dict[str, str]) -> dict[str, dict[str, str | list[str] | dict[str, str]]]:
    files = payload_hashes(payload)
    owned = {"interface_jar": "game_playing_interface.jar", "config_source": "config.xml", "serverbackup_source": "serverbackup", "player_wrapper": "9001.x86_64"}
    missing = sorted(set(owned.values()) - files.keys())
    if missing:
        raise PackagingError("missing provenance-bound payload file: " + missing[0])
    wrapper_source = worktree / WRAPPER_PATH
    wrapper_bytes = subprocess.run(["git", "show", f"HEAD:{WRAPPER_PATH}"], cwd=worktree, capture_output=True, check=True).stdout
    wrapper_sha = hashlib.sha256(wrapper_bytes).hexdigest()
    if wrapper_source.read_bytes() != wrapper_bytes or files[owned["player_wrapper"]] != wrapper_sha:
        raise PackagingError("player wrapper differs from tracked source")
    interface_sha = sha256_file(sources.interface_jar)
    serverbackup_sha = sha256_file(sources.serverbackup_source)
    if files[owned["interface_jar"]] != interface_sha:
        raise PackagingError("interface jar differs from payload copy")
    if files[owned["serverbackup_source"]] != serverbackup_sha:
        raise PackagingError("serverbackup differs from payload copy")
    blob = subprocess.run(["git", "rev-parse", f"HEAD:{WRAPPER_PATH}"], cwd=worktree, text=True, capture_output=True, check=True).stdout.strip()
    generated = sorted(files.keys() - set(owned.values()))
    return {
        "unity_executable": {"sha256": sha256_file(sources.unity_executable)},
        "interface_jar": {"sha256": interface_sha, "payload_files": [owned["interface_jar"]]},
        "config_source": {"sha256": sha256_file(sources.config_source), "transform": CONFIG_TRANSFORM, "output_sha256": files[owned["config_source"]], "payload_files": [owned["config_source"]]},
        "serverbackup_source": {"sha256": serverbackup_sha, "payload_files": [owned["serverbackup_source"]]},
        "player_wrapper": {"repository_path": WRAPPER_PATH, "git_blob": blob, "source_sha256": wrapper_sha, "sha256": wrapper_sha, "payload_files": [owned["player_wrapper"]]},
        "unity_package_inputs": {"files": package_inputs},
        "generated_unity_outputs": {"generator": "unity_executable", "payload_files": generated, "output_sha256": {name: files[name] for name in generated}},
    }


def write_manifest(payload: Path, context: ManifestContext) -> Path:
    head, tree = git_revision(context.worktree, context.package_inputs, require_package_inputs=True)
    manifest = {
        "schema_version": STAGE_SCHEMA,
        "unity": {"canonical_revision": "2019.3.4f1 (4f139db2fdbd)", "migrated_revision": f"{UNITY_VERSION} ({UNITY_CHANGESET})", "version": UNITY_VERSION, "changeset": UNITY_CHANGESET},
        "project": {"git_head": head, "git_tree": tree},
        "capture": {"schema_version": context.capture_schema, "protocol_version": PROTOCOL_VERSION},
        "migration": {"provenance_file": context.migration_provenance.name, "provenance_sha256": sha256_file(context.migration_provenance)},
        "rollback_rule": "Build or verification failure never modifies canonical project, production player, or active data root.",
        "files": payload_hashes(payload),
    }
    if context.build_sources is not None:
        manifest["build_inputs"] = build_input_manifest(payload, context.worktree, context.build_sources, context.package_inputs)
    path = payload / "provenance.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_archive(archive: Path, expected_files: dict[str, str], require_manifest: bool) -> None:
    actual: dict[str, str] = {}
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
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                actual[member.name] = digest.hexdigest()
    if actual != expected_files:
        raise PackagingError("archive payload differs from source payload")
    if require_manifest and manifest_bytes is None:
        raise PackagingError("archive provenance manifest is missing")
    if manifest_bytes is not None:
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
            raise PackagingError("archive provenance manifest is malformed")
        if manifest["files"] != actual:
            raise PackagingError("archive provenance payload hashes differ")


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
    expected_files = payload_hashes(payload)
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


def validate_receipt(receipt: Path, archive: Path, archive_name: str) -> bytes:
    content = receipt.read_bytes()
    expected = f"{sha256_file(archive)}  {archive_name}\n".encode("ascii")
    if content != expected:
        raise PackagingError("archive checksum receipt differs from archive")
    return content


def publish(payload: Path, stage: Path) -> None:
    archive = stage / ARCHIVE_NAME
    receipt = stage / "archive.sha256"
    previous_receipt = validate_receipt(receipt, archive, archive.name) if receipt.exists() else None
    archive_temporary = temporary_sibling(archive)
    try:
        receipt_temporary = temporary_sibling(receipt)
        try:
            archive_committed = False
            write_archive(payload, archive_temporary)
            receipt_bytes = f"{sha256_file(archive_temporary)}  {archive.name}\n".encode("ascii")
            write_fsynced(receipt_temporary, receipt_bytes)
            validate_receipt(receipt_temporary, archive_temporary, archive.name)
            fsync_directory(stage)
            try:
                receipt.unlink(missing_ok=True)
                fsync_directory(stage)
                os.replace(archive_temporary, archive)
                archive_committed = True
                fsync_directory(stage)
            except (OSError, KeyboardInterrupt):
                if archive_committed:
                    receipt.unlink(missing_ok=True)
                    fsync_directory(stage)
                elif previous_receipt is not None:
                    write_fsynced(receipt_temporary, previous_receipt)
                    validate_receipt(receipt_temporary, archive, archive.name)
                    os.replace(receipt_temporary, receipt)
                    fsync_directory(stage)
                raise
            try:
                os.replace(receipt_temporary, receipt)
                fsync_directory(stage)
            except (OSError, KeyboardInterrupt):
                receipt.unlink(missing_ok=True)
                fsync_directory(stage)
                raise
        finally:
            receipt_temporary.unlink(missing_ok=True)
    finally:
        archive_temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--migration-provenance", type=Path, required=True)
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
        package_inputs = unity_package_input_digests(args.worktree)
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
        package_inputs = unity_package_input_digests(args.worktree)
    else:
        raw_package_inputs = json.loads(args.package_inputs.read_text(encoding="utf-8"))
        if not isinstance(raw_package_inputs, dict) or not all(isinstance(path, str) and isinstance(digest, str) for path, digest in raw_package_inputs.items()):
            raise PackagingError("Unity package input provenance is malformed")
        package_inputs = raw_package_inputs
    context = ManifestContext(args.worktree, args.migration_provenance, sources, package_inputs, args.capture_schema)
    write_manifest(args.payload, context)
    args.stage.mkdir(parents=True, exist_ok=True)
    publish(args.payload, args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
