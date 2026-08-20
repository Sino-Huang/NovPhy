from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Final


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
UNITY_VERSION: Final = "2019.4.41f2"
UNITY_CHANGESET: Final = "6b23d448b533"
STAGE_SCHEMA: Final = "novphy_physics_player_stage_v1"
CAPTURE_SCHEMA_V1: Final = "physics_capture_v1"
CAPTURE_SCHEMA_V2: Final = "physics_capture_v2_engine_v1"
MAX_ARCHIVE_MEMBER_SIZE: Final = 512 * 1024 * 1024


class VerificationError(RuntimeError):
    pass


class UnsafeArchiveChecksumPathError(VerificationError):
    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"unsafe archive checksum path: {path}")


class UnsafePayloadManifestPathError(VerificationError):
    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"unsafe payload manifest path: {path}")


class UnsafeArchiveMemberError(VerificationError):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"unsafe archive member: {name}")


def is_single_path_component(value: str) -> bool:
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    return value not in {".", ".."} and posix_path.parts == (value,) and windows_path.parts == (value,)


def is_canonical_payload_path(value: str) -> bool:
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    return (
        bool(posix_path.parts)
        and not posix_path.is_absolute()
        and not windows_path.anchor
        and ".." not in posix_path.parts
        and posix_path.as_posix() == value
        and windows_path.as_posix() == value
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_from_stage(stage: Path) -> tuple[Path, str]:
    checksum = (stage / "archive.sha256").read_text(encoding="ascii").strip().split()
    if len(checksum) != 2:
        raise VerificationError("malformed archive checksum receipt")
    archive_name = checksum[1]
    if not is_single_path_component(archive_name):
        raise UnsafeArchiveChecksumPathError(archive_name)
    stage_root = stage.resolve()
    archive = stage_root / archive_name
    if archive.is_symlink() or not archive.is_file():
        raise UnsafeArchiveChecksumPathError(archive_name)
    try:
        archive = archive.resolve(strict=True)
        archive.relative_to(stage_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise UnsafeArchiveChecksumPathError(archive_name) from error
    actual = sha256_file(archive)
    if actual != checksum[0]:
        raise VerificationError("archive SHA-256 mismatch")
    return archive, actual


def safe_unpack(archive: Path, output: Path) -> None:
    output_root = output.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        members: list[tarfile.TarInfo] = []
        while (member := bundle.next()) is not None:
            posix_path = PurePosixPath(member.name)
            windows_path = PureWindowsPath(member.name)
            is_root_directory = member.isdir() and member.name in {".", "./"}
            has_confined_type = member.isfile() or member.isdir()
            has_confined_size = not member.isfile() or 0 <= member.size <= MAX_ARCHIVE_MEMBER_SIZE
            if (
                not has_confined_type
                or not has_confined_size
                or (not posix_path.parts and not is_root_directory)
                or posix_path.is_absolute()
                or bool(windows_path.anchor)
                or ".." in posix_path.parts
                or ".." in windows_path.parts
            ):
                raise UnsafeArchiveMemberError(member.name)
            destination = output_root.joinpath(*posix_path.parts).resolve()
            try:
                destination.relative_to(output_root)
            except ValueError as error:
                raise UnsafeArchiveMemberError(member.name) from error
            members.append(member)
        bundle.extractall(output_root, members=members, filter="data")


def verify_payload(output: Path, capture_schema: str = CAPTURE_SCHEMA_V1) -> None:
    output_root = output.resolve()
    manifest = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != STAGE_SCHEMA:
        raise VerificationError("unsupported provenance schema")
    unity = manifest.get("unity")
    capture = manifest.get("capture")
    files = manifest.get("files")
    if not isinstance(unity, dict) or unity.get("version") != UNITY_VERSION or unity.get("changeset") != UNITY_CHANGESET:
        raise VerificationError("Unity revision mismatch")
    if not isinstance(capture, dict) or capture.get("schema_version") != capture_schema or capture.get("protocol_version") != 1:
        raise VerificationError("capture protocol provenance mismatch")
    if not isinstance(files, dict):
        raise VerificationError("payload checksum manifest is malformed")
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise VerificationError("payload checksum entry is malformed")
        if not is_canonical_payload_path(relative):
            raise UnsafePayloadManifestPathError(relative)
        path = output_root / relative
        if any((output_root.joinpath(*PurePosixPath(relative).parts[:index])).is_symlink()
               for index in range(1, len(PurePosixPath(relative).parts) + 1)):
            raise UnsafePayloadManifestPathError(relative)
        try:
            path = path.resolve()
            path.relative_to(output_root)
        except (OSError, RuntimeError, ValueError) as error:
            raise UnsafePayloadManifestPathError(relative) from error
        if not path.is_file() or sha256_file(path) != expected:
            raise VerificationError("payload SHA-256 mismatch: " + relative)


def connect_with_deadline(port: int, deadline: float):
    from src.webui.bridge import ScienceBirdsBridge

    last_error: OSError | None = None
    while time.monotonic() < deadline:
        bridge = ScienceBirdsBridge("127.0.0.1", port, timeout=5.0)
        try:
            bridge.connect()
            return bridge
        except OSError as error:
            last_error = error
            time.sleep(0.25)
    raise VerificationError(f"player socket {port} unavailable: {last_error}")


def verify_runtime(output: Path, agent_port: int, game_port: int, physics_port: int) -> dict[str, int | bool]:
    from scripts.manual_agent import prepare_for_play

    if physics_port != 2004:
        raise VerificationError("packaged interface requires isolated physics port 2004")
    display = os.environ.get("DISPLAY", ":197")
    xvfb_log_path = output / "verification-xvfb.log"
    xvfb_log = xvfb_log_path.open("wb")
    xvfb = None
    if not os.environ.get("DISPLAY"):
        xvfb = subprocess.Popen(
            ["Xvfb", display, "-screen", "0", "1024x768x24", "+extension", "GLX", "+iglx", "-nolisten", "tcp"],
            stdout=xvfb_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    command = [
        "java", "-jar", "./game_playing_interface.jar", "--agent-port", str(agent_port),
        "--game-start-port", str(game_port), "--dev",
    ]
    log_path = output / "verification-runtime.log"
    with log_path.open("wb") as log:
        environment = os.environ.copy()
        environment["DISPLAY"] = display
        process = subprocess.Popen(
            command, cwd=output, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=environment,
        )
        try:
            legacy = connect_with_deadline(agent_port, time.monotonic() + 90)
            legacy.configure(agent_id=28701)
            prepare_for_play(legacy, timeout=120, poll_delay=1.0)
            symbolic = legacy.get_symbolic_state_without_screenshot()
            physics = connect_with_deadline(physics_port, time.monotonic() + 15)
            capture = physics.get_physics_capture_v1()
            legacy.disconnect()
            physics.disconnect()
            return {
                "legacy_request_62": isinstance(symbolic, list),
                "legacy_feature_batches": len(symbolic),
                "request_70": capture.png.startswith(b"\x89PNG\r\n\x1a\n"),
                "request_70_render_frame": int(capture.state["render_frame"]),
            }
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except ProcessLookupError:
                process.wait()
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            if xvfb is not None and xvfb.poll() is None:
                os.killpg(xvfb.pid, signal.SIGTERM)
                try:
                    xvfb.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(xvfb.pid, signal.SIGKILL)
                    xvfb.wait(timeout=5)
            xvfb_log.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--expect-sha")
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--agent-port", type=int, default=22004)
    parser.add_argument("--game-port", type=int, default=29001)
    parser.add_argument("--physics-port", type=int, default=2004)
    parser.add_argument("--physics-v2", action="store_true")
    args = parser.parse_args()
    try:
        archive, archive_sha = archive_from_stage(args.stage)
        if args.expect_sha is not None and args.expect_sha != archive_sha:
            raise VerificationError("archive SHA-256 mismatch against --expect-sha")
        with tempfile.TemporaryDirectory(prefix="novphy_physics_verify_") as temporary:
            output = Path(temporary)
            safe_unpack(archive, output)
            verify_payload(output, CAPTURE_SCHEMA_V2 if args.physics_v2 else CAPTURE_SCHEMA_V1)
            runtime = None if args.skip_runtime else verify_runtime(output, args.agent_port, args.game_port, args.physics_port)
        print(json.dumps({
            "archive_sha256": archive_sha,
            "archive_checksum_verified": True,
            "payload_checksums_verified": True,
            "runtime": runtime,
        }, sort_keys=True))
        return 0
    except (OSError, json.JSONDecodeError, tarfile.TarError, VerificationError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
