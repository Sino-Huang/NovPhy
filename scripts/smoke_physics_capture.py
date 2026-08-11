#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Run one isolated request-70 physics capture and accept it only when valid."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Callable, Final, TypeAlias

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_rollouts import (
    action_to_shot,
    anchor_action_to_slingshot_reference,
    capture_physics_rollout,
    current_slingshot_reference,
)
from scripts.manual_agent import connect_with_retry, prepare_for_play
from scripts.rollout_artifacts import validate_physics_shot_artifact
from scripts.rollout_validation_types import PhysicsArtifactError
from scripts.smoke_protection import (
    ProtectionError,
    canonical_root_from_git,
    protected_receipt,
    protected_roots,
    tree_digest,
)
from scripts.verify_physics_player import safe_unpack, verify_payload
from src.webui.bridge import JsonValue as BridgeJsonValue, PhysicsCaptureV1, PhysicsCaptureV1Failure, PlayingMode, RequestCode, ScienceBirdsBridge

CAPTURE_READ_TIMEOUT_SECONDS: Final = 120.0
ASSEMBLY_RELATIVE_PATH: Final = "9001_Data/Managed/Assembly-CSharp.dll"
# The mapped anchor is the native runtime, not the user assembly. Unity 2019.4
# loads `Assembly-CSharp.dll` through its own loader, so it never appears as a
# file-backed mapping: a live player was measured with 91 file-backed mappings,
# six of them framework DLLs, and zero `Assembly-CSharp.dll` images. Requiring
# one was an invariant no candidate could satisfy. `UnityPlayer.so` is mapped
# from the candidate root, carries a provenance digest, and is the code that
# actually owns the socket, so it anchors the mapping identity; the assembly is
# still pinned by digest, read through the observed process's own root view.
RUNTIME_RELATIVE_PATH: Final = "UnityPlayer.so"
BINDING_IDENTITY_FIELDS: Final = ("pid", "socket_inode", "executable", "cwd", "runtime_path", "runtime_device", "runtime_inode", "runtime_sha256", "assembly_path", "assembly_device", "assembly_inode", "assembly_sha256", "provenance_sha256", "archive_sha256")
# Observation fields `resolve_listener_binding` deliberately does not compare, so
# that adding a field without deciding its fate is a hard failure rather than a
# silent gap. `socket_inode`, `parent_pid`, `process_group` and `session_id` are
# measurements with no expectation to compare against; `assembly_path` is fixed
# by construction (see the comment beside the assembly checks).
UNCHECKED_OBSERVATION_FIELDS: Final = ("assembly_path", "parent_pid", "process_group", "session_id", "socket_inode")
# Environment variables that decide what *else* a process loads. The identity
# chain pins every byte of the candidate, and proves the pinned assembly exists
# in the listener's own root -- it cannot prove the listener loaded that and
# nothing besides. So these are refused at launch rather than recorded.
INTERPOSITION_ENV_VARS: Final = ("LD_PRELOAD", "LD_AUDIT", "MONO_PATH", "MONO_GAC_PREFIX", "MONO_CONFIG", "DOTNET_STARTUP_HOOKS")
# Refusing this one would make the gate unrunnable on any host that sets it for
# unrelated reasons (CUDA toolchains routinely do), and the player resolves its
# own libraries through its rpath, so it is dropped from the child environment
# and the drop is recorded.
STRIPPED_ENV_VARS: Final = ("LD_LIBRARY_PATH",)
# `/proc/net` bind addresses that are loopback-only, in the kernel's own hex
# form. A player that bound the wildcard address would expose the physics port
# to the network for the life of the run, and the gate previously recorded that
# without objecting to it.
LOOPBACK_BIND_ADDRESSES: Final = frozenset({
    "0100007F",                          # 127.0.0.1
    "00000000000000000000000001000000",  # ::1
    "0000000000000000FFFF00000100007F",  # ::ffff:127.0.0.1
})
DEFAULT_SMOKE_REPORT: Final = ROOT / ".claude/project-docs/evidence/world-model-physics-instrumentation/task-8-smoke.json"
DEFAULT_DIAGNOSTIC_REPORT: Final = ROOT / ".claude/project-docs/evidence/world-model-physics-instrumentation/listener-diagnostic.json"
ARTIFACT_NAMES: Final = ("shot_001.tmp", "shot_001")
FRAME_HEIGHT_PIXELS: Final = 480
WIRE_STATE_FIELDS: Final = frozenset(("schema_version", "capture_id", "sequence", "render_frame", "render_time", "fixed_step", "fixed_time", "coordinates", "nodes", "raw_contacts", "support_edges"))
JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class SmokeError(Exception):
    """Rejection raised by this smoke; never frozen, so tracebacks stay assignable."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class MappedRegion:
    """One file-backed mapping observed directly in `/proc/<pid>/maps`."""

    path: Path
    device: str
    inode: int


@dataclass(frozen=True, slots=True)
class CandidateExpectation:
    """Exact identity the port listener must prove before any gameplay."""

    root: Path
    archive_sha256: str
    assembly_sha256: str
    assembly_device: str
    assembly_inode: int
    runtime_sha256: str
    runtime_device: str
    runtime_inode: int
    provenance_sha256: str

    @property
    def executables(self) -> frozenset[Path]:
        return frozenset({self.root / "9001.x86_64", self.root / "9001-player.x86_64"})

    @property
    def assembly(self) -> Path:
        return self.root / ASSEMBLY_RELATIVE_PATH

    @property
    def runtime(self) -> Path:
        return self.root / RUNTIME_RELATIVE_PATH


@dataclass(frozen=True, slots=True)
class ListenerObservation:
    pid: int
    parent_pid: int
    process_group: int
    session_id: int
    socket_inode: str
    executable: Path
    cwd: Path
    runtime_path: Path
    runtime_device: str
    runtime_inode: int
    runtime_sha256: str
    assembly_path: Path
    assembly_device: str
    assembly_inode: int
    assembly_sha256: str
    provenance_sha256: str
    provenance_assembly_sha256: str
    provenance_runtime_sha256: str
    archive_sha256: str
    process_tree: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SocketOwnerScan:
    """Socket ownership plus the processes that could not be inspected."""

    owners: tuple[tuple[int, str], ...]
    unreadable_pids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ListeningSocket:
    """One listening socket observed on the target port, with its bind address."""

    inode: str
    local_address: str
    table: str


class ListenerBindingError(SmokeError):
    pass


def candidate_expectation(root: Path, archive_sha256: str) -> CandidateExpectation:
    """Derive the exact on-disk identity the listener must match.

    The root is symlink-resolved because `/proc/<pid>/exe` and `/proc/<pid>/cwd`
    are always fully resolved; comparing an unresolved root against them would
    reject every run on a host whose TMPDIR contains a symlink.
    """
    root = root.resolve(strict=True)
    assembly = root / ASSEMBLY_RELATIVE_PATH
    status = assembly.stat()
    runtime = root / RUNTIME_RELATIVE_PATH
    runtime_status = runtime.stat()
    provenance = root / "provenance.json"
    return CandidateExpectation(
        root,
        archive_sha256,
        hashlib.sha256(assembly.read_bytes()).hexdigest(),
        f"{os.major(status.st_dev):02x}:{os.minor(status.st_dev):02x}",
        status.st_ino,
        hashlib.sha256(runtime.read_bytes()).hexdigest(),
        f"{os.major(runtime_status.st_dev):02x}:{os.minor(runtime_status.st_dev):02x}",
        runtime_status.st_ino,
        hashlib.sha256(provenance.read_bytes()).hexdigest(),
    )


def _mapped_regions(pid: int, name: str) -> tuple[MappedRegion, ...]:
    """Return distinct file-backed mappings of an exact file name for one process."""
    regions: set[MappedRegion] = set()
    for line in (Path("/proc") / str(pid) / "maps").read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        pathname = fields[5].strip()
        if pathname.endswith(" (deleted)"):
            pathname = pathname[: -len(" (deleted)")]
            if PurePosixPath(pathname).name == name:
                raise ListenerBindingError(f"listener mapped a deleted {name}")
            continue
        if PurePosixPath(pathname).name != name:
            continue
        regions.add(MappedRegion(Path(pathname), fields[3], int(fields[4])))
    return tuple(sorted(regions, key=lambda region: (region.path.as_posix(), region.device, region.inode)))


def _process_view(pid: int, path: Path) -> Path:
    """Resolve an absolute path through the observed process's own root."""
    return Path("/proc") / str(pid) / "root" / path.relative_to("/")


def _listening_sockets(port: int) -> tuple[ListeningSocket, ...]:
    """Return every socket listening on an exact TCP port, or reject on a blind table.

    A table that cannot be read is "cannot determine", which must reject rather
    than hide a listener. A single absent table is determinable: the kernel has no
    such address family, so no socket of that family can exist. Every table being
    absent is not determinable at all -- that is a masked `/proc` (`subset=pid`),
    where an empty result would otherwise be misread as a drained port.
    """
    expected = f"{port:04X}"
    sockets: set[ListeningSocket] = set()
    tables = (Path("/proc/net/tcp"), Path("/proc/net/tcp6"))
    read_any = False
    for table in tables:
        try:
            lines = table.read_text(encoding="ascii").splitlines()[1:]
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ListenerBindingError(f"{table} is unreadable, so port-{port} ownership cannot be determined: {error}") from error
        read_any = True
        for line in lines:
            fields = line.split()
            if len(fields) > 9 and fields[1].rsplit(":", 1)[1] == expected and fields[3] == "0A":
                sockets.add(ListeningSocket(fields[9], fields[1], table.name))
    if not read_any:
        raise ListenerBindingError(f"no /proc/net TCP table exists, so port-{port} state cannot be determined")
    return tuple(sorted(sockets, key=lambda entry: (entry.inode, entry.local_address, entry.table)))


def _proc_net_listeners(port: int) -> tuple[str, ...]:
    """Return socket inodes listening on an exact TCP port."""
    return tuple(sorted({entry.inode for entry in _listening_sockets(port)}))


def _socket_owner_scan(inodes: tuple[str, ...]) -> SocketOwnerScan:
    """Attribute listening socket inodes to owning PIDs, recording blind spots."""
    owners: set[tuple[int, str]] = set()
    unreadable: set[int] = set()
    targets = {f"socket:[{inode}]": inode for inode in inodes}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc.name)
        except ValueError:
            continue
        try:
            descriptors = tuple((proc / "fd").iterdir())
        except OSError:
            if proc.exists():
                unreadable.add(pid)
            continue
        for descriptor in descriptors:
            try:
                inode = targets.get(descriptor.readlink().as_posix())
            except FileNotFoundError:
                # A descriptor that vanished between the directory snapshot and
                # this readlink. Counting it as a blind spot was tried and is
                # wrong: `iterdir()` holds its own descriptor, which is closed by
                # the time this loop runs, so *every* process on the host --
                # including this one -- was marked unreadable and the census
                # became noise. A co-owner closing its dup here is indeed
                # invisible, but that is the same residual already recorded on
                # `require_full_attribution`, and it cannot be separated from
                # ordinary fd churn through `/proc`.
                continue
            except OSError:
                # The descriptor exists but cannot be read; that is a blind spot,
                # never a reason to drop a possible co-owner from the count.
                unreadable.add(pid)
                continue
            if inode is not None:
                owners.add((pid, inode))
    return SocketOwnerScan(tuple(sorted(owners)), tuple(sorted(unreadable)))


def _process_socket_inodes(pid: int) -> frozenset[str]:
    """Return the socket inodes a process currently holds open.

    Used to re-confirm ownership at the end of the identity reads. The owner
    scan and those reads are two independent `/proc` passes, and nothing else
    ties them to one process incarnation: an owner that exits between them can
    have its pid recycled by another descendant of the same launch, and the
    binding would then splice a dead process's socket inode onto a live
    process's identity.
    """
    inodes: set[str] = set()
    for descriptor in (Path("/proc") / str(pid) / "fd").iterdir():
        try:
            target = descriptor.readlink().as_posix()
        except OSError:
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            inodes.add(target[len("socket:[") : -1])
    return frozenset(inodes)


def require_full_attribution(inodes: tuple[str, ...], scan: SocketOwnerScan) -> None:
    """Reject unless every listening socket on the port was attributed to an owner.

    Opaque processes are normal on a shared host: `/proc/<pid>/fd` is EACCES for
    every other user's process. That blindness only matters when it leaves a
    listening socket unaccounted for, which is exactly the impostor case (a
    root-owned or foreign listener co-resident on the port). So the invariant is
    inode reconciliation, not the absence of opaque processes; the opaque pids are
    reported as the likely explanation for an unattributed inode.

    Residual, not closable through `/proc`: a foreign-uid process holding a *dup*
    of the candidate's own listening fd (SCM_RIGHTS or a privilege-dropping fork)
    contributes no owner tuple, yet the inode is still attributed -- to the
    candidate. The opaque pid count is recorded on the accepted binding as audit
    evidence only. It is a host-wide census, not a per-binding confidence score --
    on a shared host most processes are opaque to any unprivileged scan -- and no
    gate thresholds on it.
    """
    unattributed = sorted(set(inodes) - {inode for _, inode in scan.owners})
    if unattributed:
        raise ListenerBindingError(
            f"listening socket inodes {unattributed} have no attributable owner; "
            f"{len(scan.unreadable_pids)} processes were opaque to this user"
        )


def _stat_fields(pid: int) -> tuple[str, ...]:
    """Return `/proc/<pid>/stat` fields from `state` onward, immune to a forged comm.

    `comm` is process-settable and unescaped, so splitting the whole line lets a
    process inject fake fields. Everything after the final `)` is kernel-written.
    """
    line = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
    return tuple(line[line.rindex(")") + 1 :].split())


def _process_tree(root_pid: int) -> tuple[int, ...]:
    """Snapshot the launched process tree by numeric parent relationships."""
    parents: dict[int, int] = {}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            parents[int(proc.name)] = int(_stat_fields(int(proc.name))[1])
        except (OSError, ValueError, IndexError):
            continue
    tree = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in tree and pid not in tree:
                tree.add(pid)
                changed = True
    return tuple(sorted(tree))


def _observe_listener_owners(
    owners: tuple[tuple[int, str], ...],
    launched_pid: int,
    expectation: CandidateExpectation,
    stage_archive: Path,
) -> tuple[ListenerObservation, ...]:
    """Read every socket owner from `/proc`; an unreadable owner is a rejection."""
    tree = _process_tree(launched_pid)
    archive_sha = hashlib.sha256(stage_archive.read_bytes()).hexdigest()
    observations: list[ListenerObservation] = []
    for pid, socket_inode in owners:
        proc = Path("/proc") / str(pid)
        try:
            stat = _stat_fields(pid)
            starttime_before = stat[19]
            executable = (proc / "exe").resolve(strict=True)
            cwd = (proc / "cwd").resolve(strict=True)
            regions = _mapped_regions(pid, RUNTIME_RELATIVE_PATH)
            provenance_bytes = _process_view(pid, expectation.root / "provenance.json").read_bytes()
            process_group = os.getpgid(pid)
            session_id = os.getsid(pid)
        except (OSError, ValueError, IndexError) as error:
            raise ListenerBindingError(f"port listener owner pid={pid} is not attributable: {error}") from error
        if len(regions) != 1:
            raise ListenerBindingError(f"port listener owner pid={pid} has {len(regions)} mapped {RUNTIME_RELATIVE_PATH} images")
        region = regions[0]
        try:
            runtime_sha = hashlib.sha256(_process_view(pid, region.path).read_bytes()).hexdigest()
            # The assembly is read through the observed process's own root rather
            # than from this process's view, so a candidate in a different mount
            # namespace cannot pass by pointing the gate at our copy of the file.
            assembly_view = _process_view(pid, expectation.assembly)
            assembly_status = assembly_view.stat()
            assembly_sha = hashlib.sha256(assembly_view.read_bytes()).hexdigest()
        except OSError as error:
            raise ListenerBindingError(f"port listener owner pid={pid} runtime or assembly image is unreadable: {error}") from error
        manifest = json.loads(provenance_bytes.decode("utf-8"))
        # A provenance document that is not an object (an array, a bare string)
        # is rejected here rather than being allowed to raise `AttributeError`
        # out of `.get`. Absorbing that in the caller's handler would also
        # absorb every genuine `None.attr` and typo -- and `FrozenInstanceError`,
        # which this repo's frozen dataclasses raise, is an `AttributeError`.
        if not isinstance(manifest, dict):
            raise ListenerBindingError(f"port listener owner pid={pid} candidate provenance is {type(manifest).__name__}, not an object")
        files = manifest.get("files")
        if not isinstance(files, dict) or not isinstance(files.get(ASSEMBLY_RELATIVE_PATH), str):
            raise ListenerBindingError(f"port listener owner pid={pid} candidate provenance lacks an assembly digest")
        if not isinstance(files.get(RUNTIME_RELATIVE_PATH), str):
            raise ListenerBindingError(f"port listener owner pid={pid} candidate provenance lacks a {RUNTIME_RELATIVE_PATH} digest")
        # Close the splice between the owner scan and these reads. `starttime` is
        # kernel-written and unique per incarnation, so an unchanged value means
        # no exit-and-recycle happened across the identity reads; re-reading the
        # descriptor table means the process whose identity was just measured is
        # the one holding the socket now, not merely the one that held it when
        # the scan ran.
        try:
            starttime_after = _stat_fields(pid)[19]
            held_inodes = _process_socket_inodes(pid)
        except (OSError, IndexError) as error:
            raise ListenerBindingError(f"port listener owner pid={pid} could not be re-confirmed: {error}") from error
        if starttime_after != starttime_before:
            raise ListenerBindingError(f"port listener owner pid={pid} was replaced during observation")
        if socket_inode not in held_inodes:
            raise ListenerBindingError(f"port listener owner pid={pid} no longer holds socket inode {socket_inode}")
        observations.append(
            ListenerObservation(
                pid,
                int(stat[1]),
                process_group,
                session_id,
                socket_inode,
                executable,
                cwd,
                region.path,
                region.device,
                region.inode,
                runtime_sha,
                expectation.assembly,
                f"{os.major(assembly_status.st_dev):02x}:{os.minor(assembly_status.st_dev):02x}",
                assembly_status.st_ino,
                assembly_sha,
                hashlib.sha256(provenance_bytes).hexdigest(),
                files[ASSEMBLY_RELATIVE_PATH],
                files[RUNTIME_RELATIVE_PATH],
                archive_sha,
                tree,
            )
        )
    return tuple(observations)


def wait_for_listener(
    physics_port: int,
    launched_pid: int,
    expectation: CandidateExpectation,
    observation_reader: Callable[[tuple[tuple[int, str], ...]], tuple[ListenerObservation, ...]],
    *,
    deadline_seconds: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Resolve socket ownership before gameplay, rejecting timeout and ambiguity."""
    deadline = time.monotonic() + deadline_seconds
    scan = SocketOwnerScan((), ())
    last_inodes: tuple[str, ...] = ()
    while time.monotonic() < deadline:
        sockets = _listening_sockets(physics_port)
        last_inodes = tuple(sorted({entry.inode for entry in sockets}))
        if last_inodes:
            foreign = sorted({entry.local_address for entry in sockets if entry.local_address.rsplit(":", 1)[0].upper() not in LOOPBACK_BIND_ADDRESSES})
            if foreign:
                raise ListenerBindingError(f"port-{physics_port} is bound to a non-loopback address: {foreign}")
            scan = _socket_owner_scan(last_inodes)
            require_full_attribution(last_inodes, scan)
            observations = observation_reader(scan.owners)
            if observations:
                binding = resolve_listener_binding(physics_port, launched_pid, observations, expectation)
                binding["listening_sockets"] = [{"inode": entry.inode, "local_address": entry.local_address, "table": entry.table} for entry in sockets]
                # Audit evidence only, and a host-wide census rather than a
                # per-binding score; see the residual on `require_full_attribution`.
                binding["host_uninspectable_pid_count"] = len(scan.unreadable_pids)
                binding["host_uninspectable_pids"] = list(scan.unreadable_pids)
                return binding
        sleep(0.1)
    raise ListenerBindingError(
        f"port-{physics_port} listener did not become attributable; inodes={list(last_inodes)}, owners={[list(owner) for owner in scan.owners]}, uninspectable_pids={list(scan.unreadable_pids)}"
    )


def resolve_listener_binding(
    physics_port: int,
    launched_pid: int,
    observations: tuple[ListenerObservation, ...],
    expectation: CandidateExpectation,
) -> dict[str, object]:
    """Accept one socket-owned candidate only after exact process-backed checks."""
    if len(observations) != 1:
        raise ListenerBindingError(f"expected one port-{physics_port} listener owner, found {len(observations)}")
    observation = observations[0]
    checks: tuple[tuple[str, bool, str], ...] = (
        ("executable", observation.executable in expectation.executables, "listener executable differs"),
        ("cwd", observation.cwd == expectation.root, "listener cwd differs"),
        ("runtime_path", observation.runtime_path == expectation.runtime, "listener mapped runtime path differs"),
        ("runtime_device", observation.runtime_device == expectation.runtime_device, "listener mapped runtime device differs"),
        ("runtime_inode", observation.runtime_inode == expectation.runtime_inode, "listener mapped runtime inode differs"),
        ("runtime_sha256", observation.runtime_sha256 == expectation.runtime_sha256, "listener runtime digest differs"),
        ("provenance_runtime_sha256", observation.provenance_runtime_sha256 == expectation.runtime_sha256, "listener provenance runtime digest differs"),
        # `assembly_path` is fixed by construction (the reader resolves exactly
        # this path through the process root), so it is recorded for drift
        # detection rather than checked here. The substance is the device, inode,
        # and digest actually observed at that path in the listener's own root.
        ("assembly_device", observation.assembly_device == expectation.assembly_device, "listener assembly device differs"),
        ("assembly_inode", observation.assembly_inode == expectation.assembly_inode, "listener assembly inode differs"),
        ("assembly_sha256", observation.assembly_sha256 == expectation.assembly_sha256, "listener assembly digest differs"),
        ("provenance_assembly_sha256", observation.provenance_assembly_sha256 == expectation.assembly_sha256, "listener provenance assembly digest differs"),
        ("provenance_sha256", observation.provenance_sha256 == expectation.provenance_sha256, "listener provenance digest differs"),
        ("archive_sha256", observation.archive_sha256 == expectation.archive_sha256, "staged archive digest drifted during the run"),
        # Containment is only a guarantee while the tree is rooted at the pid this
        # run actually launched. A tree that does not contain `launched_pid` was
        # not derived from it, so membership in it proves nothing -- and that is
        # reachable by a future refactor computing the tree per observation, not
        # only by a malicious caller.
        ("process_tree", launched_pid in observation.process_tree, "process tree is not rooted at the launched pid"),
        ("pid", observation.pid in observation.process_tree, "listener outside launched process tree"),
    )
    for _field, holds, message in checks:
        if not holds:
            raise ListenerBindingError(message)
    uncovered = sorted(
        field.name
        for field in fields(ListenerObservation)
        if field.name not in {name for name, _holds, _message in checks} | set(UNCHECKED_OBSERVATION_FIELDS)
    )
    if uncovered:
        raise SmokeError(f"listener observation fields are neither checked nor declared unchecked: {', '.join(uncovered)}")
    return {
        "pid": observation.pid,
        "parent_pid": observation.parent_pid,
        "process_group": observation.process_group,
        "session_id": observation.session_id,
        "socket_inode": observation.socket_inode,
        "executable": str(observation.executable),
        "cwd": str(observation.cwd),
        "runtime_path": str(observation.runtime_path),
        "runtime_device": observation.runtime_device,
        "runtime_inode": observation.runtime_inode,
        "runtime_sha256": observation.runtime_sha256,
        "assembly_path": str(observation.assembly_path),
        "assembly_device": observation.assembly_device,
        "assembly_inode": observation.assembly_inode,
        "assembly_sha256": observation.assembly_sha256,
        "provenance_sha256": observation.provenance_sha256,
        "provenance_assembly_sha256": observation.provenance_assembly_sha256,
        "provenance_runtime_sha256": observation.provenance_runtime_sha256,
        "archive_sha256": observation.archive_sha256,
        "process_tree": list(observation.process_tree),
        "launched_pid": launched_pid,
    }


def require_stable_binding(before: Mapping[str, object], after: Mapping[str, object]) -> dict[str, object]:
    """Require the capture to come from the exact process/package that was bound."""
    for name, binding in (("bind", before), ("rebind", after)):
        missing = sorted(field for field in BINDING_IDENTITY_FIELDS if field not in binding)
        if missing:
            raise SmokeError(f"{name} binding is missing identity fields: {', '.join(missing)}")
    drift = sorted(field for field in BINDING_IDENTITY_FIELDS if before[field] != after[field])
    if drift:
        raise SmokeError("candidate binding drifted between listener bind and capture: " + ", ".join(drift))
    return {"rebound_pid": after["pid"], "rebound_socket_inode": after["socket_inode"], "stable_fields": list(BINDING_IDENTITY_FIELDS)}


@dataclass(frozen=True, slots=True)
class CapturedRequest:
    capture: PhysicsCaptureV1

    def get_physics_capture_v1(self) -> PhysicsCaptureV1:
        state = {key: mutable_json(value) for key, value in self.capture.state.items()}
        events = tuple({key: mutable_json(value) for key, value in event.items()} for event in self.capture.events)
        return PhysicsCaptureV1(self.capture.png, state, events)


def mutable_json(value: BridgeJsonValue) -> JsonValue:
    """Convert bridge-frozen JSON containers into writer-compatible containers."""
    if isinstance(value, Mapping):
        return {key: mutable_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [mutable_json(item) for item in value]
    return value


def perform_known_action(bridge: ScienceBirdsBridge) -> JsonObject:
    reference = current_slingshot_reference(bridge, FRAME_HEIGHT_PIXELS)
    if reference is None:
        raise SmokeError("known level has no request-62 slingshot reference")
    # Parameters taken from an accepted legacy rollout whose shot physically
    # struck a structure (novelty_level_4_type010401_00141_0_1_010401_4_1,
    # shot_001: ui_level 1, drag_hold_release, slingshot_relative). The single
    # permitted full smoke has no retry and a collision-free shot fails
    # acceptance outright, so the release offset must be a verified one rather
    # than an arbitrary drag. `drag_start` is overwritten by the live slingshot
    # reference below; it is recorded here as the value the rollout carried.
    #
    # KNOWN BLOCKER — these parameters are NOT sufficient on the level this smoke
    # actually plays. The staged config resolves ui_level 1 to
    # novelty_level_0/type2/Levels/3_9_6_1.xml, whose only bird is a BirdBlack;
    # ABBirdBlack overrides OnCollisionEnter2D without reaching the recorder, so
    # the bird's own impacts are never recorded. Blocks drop the recorder call on
    # their bird branch, platforms and ground carry no recorder at all, and both
    # pigs sit enclosed behind platform walls, so no aim reaches the one object
    # type that does record. Changing this offset changes only launch elevation —
    # both pulls saturate the drag clamp. See
    # .claude/project-docs/evidence/runtime-repin-gate-20260810/
    # finding-smoke-level-geometry-risk.json before spending the run.
    action = anchor_action_to_slingshot_reference(
        {"coordinate_frame": "slingshot_relative", "drag_start": [97, 227], "drag_release": [-80, 7], "tapTime": 0, "holdTime": 1000},
        reference,
    )
    shot = action_to_shot(action, frame_height=FRAME_HEIGHT_PIXELS)
    ground_truth_count = bridge.shoot_and_record_ground_truth(
        shot["x"],
        shot["y"],
        tap_time=shot["tapTime"],
        release_time=shot["releaseTime"],
    )
    if ground_truth_count < 1:
        raise SmokeError("known recorded gameplay action returned no ground truth")
    return {"response": 1, "request_code": int(RequestCode.GT_SHOOT), "ground_truth_count": ground_truth_count, "slingshot_reference": reference, "socket_x": shot["x"], "socket_y": shot["y"], "tap_time": shot["tapTime"], "release_time": shot["releaseTime"]}


def capture_finalized_action(physics_port: int, *, deadline_seconds: float = 30.0) -> PhysicsCaptureV1:
    deadline = time.monotonic() + deadline_seconds
    while True:
        physics = connect_with_retry("127.0.0.1", physics_port, timeout=CAPTURE_READ_TIMEOUT_SECONDS, deadline_seconds=30.0)
        try:
            return physics.get_physics_capture_v1()
        except PhysicsCaptureV1Failure as error:
            if error.code != 4 or time.monotonic() >= deadline:
                raise
        finally:
            physics.disconnect()
        time.sleep(0.25)


def require_request_identity(captures: tuple[PhysicsCaptureV1, PhysicsCaptureV1]) -> tuple[dict[str, object], dict[str, object]]:
    """Require a stable non-empty capture id and increasing request sequence."""
    identities: list[dict[str, object]] = []
    for capture in captures:
        capture_id = capture.state.get("capture_id")
        sequence = capture.state.get("sequence")
        if not isinstance(capture_id, str) or not capture_id:
            raise SmokeError("request-70 capture_id must be non-empty")
        if type(sequence) is not int:
            raise SmokeError("request-70 sequence must be an integer")
        identities.append({"capture_id": capture_id, "sequence": sequence, "render_frame": capture.state.get("render_frame")})
    if identities[0]["capture_id"] != identities[1]["capture_id"]:
        raise SmokeError("request-70 capture_id changed between responses")
    if identities[1]["sequence"] <= identities[0]["sequence"]:
        raise SmokeError("request-70 sequence did not increase")
    return identities[0], identities[1]


def require_collision(events: tuple[Mapping[str, BridgeJsonValue], ...]) -> dict[str, object]:
    """Validate every collision in the capture and return the first one.

    The accepted artifact carries all events, so validating only a sample would
    let a malformed collision reach the shot that follows this gate.
    """
    accepted: list[dict[str, object]] = []
    for index, event in enumerate(events):
        if event.get("event_type") != "collision":
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            raise SmokeError(f"collision payload must be an object (event {index})")
        if set(payload) != {"contact_ids", "relative_speed"}:
            raise SmokeError(f"collision payload fields differ from taxonomy (event {index})")
        contact_ids = payload.get("contact_ids")
        relative_speed = payload.get("relative_speed")
        if not isinstance(contact_ids, tuple) or not contact_ids or not all(isinstance(value, str) and value for value in contact_ids) or tuple(sorted(set(contact_ids))) != contact_ids:
            raise SmokeError(f"collision contact_ids must be non-empty strings, sorted, and unique (event {index})")
        if type(relative_speed) not in (int, float) or not math.isfinite(relative_speed) or relative_speed < 0:
            raise SmokeError(f"collision relative_speed must be finite and non-negative (event {index})")
        accepted.append({"contact_ids": list(contact_ids), "relative_speed": relative_speed, "fixed_step": event.get("fixed_step"), "render_frame": event.get("render_frame")})
    if not accepted:
        raise SmokeError("request-70 capture contains no collision event")
    return {**accepted[0], "collision_count": len(accepted)}


def require_action_events(events: tuple[Mapping[str, BridgeJsonValue], ...]) -> tuple[str, ...]:
    event_types = tuple(str(event.get("event_type", "")) for event in events)
    if "bird_launched" not in event_types:
        raise SmokeError("request-70 capture missing authoritative bird_launched event")
    return event_types


def archive_details(stage: Path, clone: Path) -> tuple[Path, str, str, str, str]:
    """Unpack and verify a staged archive, returning archive and payload hashes."""
    receipt = (stage / "archive.sha256").read_text(encoding="ascii").strip().split()
    if len(receipt) != 2:
        raise SmokeError("malformed archive.sha256")
    if receipt[1] != PurePosixPath(receipt[1]).name or receipt[1] in ("", ".", ".."):
        raise SmokeError("archive.sha256 must name a bare file inside the stage")
    archive = stage / receipt[1]
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    if archive_sha != receipt[0]:
        raise SmokeError("archive SHA-256 mismatch")
    safe_unpack(archive, clone)
    verify_payload(clone)
    manifest = json.loads((clone / "provenance.json").read_text(encoding="utf-8"))
    player_sha = manifest["files"]["9001-player.x86_64"]
    protocol_sha = hashlib.sha256((clone / "game_playing_interface.jar").read_bytes()).hexdigest()
    assembly_sha = hashlib.sha256((clone / "9001_Data" / "Managed" / "Assembly-CSharp.dll").read_bytes()).hexdigest()
    return archive, archive_sha, player_sha, protocol_sha, assembly_sha


def free_port() -> int:
    """Reserve an unused TCP port number, releasing it before the engine starts."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def terminate(
    process: subprocess.Popen[bytes] | None,
    *,
    drain_seconds: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Stop a process group, escalating on the *group*, and return a receipt.

    Two things make the obvious implementation wrong here. The group must be
    signalled even when the leader has already been reaped: the Unity player is
    a grandchild of the JVM, so a leader that exited first used to skip the
    group kill entirely and leave that player holding the physics port, which
    then poisons every later run on this host, and did. And escalation must be
    keyed on the group, not the leader -- a reaped or quickly-exiting leader
    says nothing about whether the port was released, so waiting on the leader
    would return "clean" while the player was still alive and ignoring SIGTERM.

    So: signal the group, poll it to empty, SIGKILL whatever outlives
    `drain_seconds`, then reap the leader. `ProcessLookupError` means the group
    is genuinely gone, and only that is a clean skip. A residual group after
    SIGKILL is reported in the receipt so the caller can fail the run.
    """
    if process is None:
        return "not-started"
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    escalated = False
    deadline = time.monotonic() + drain_seconds
    residual = _process_group_members(process.pid)
    while residual:
        if time.monotonic() >= deadline:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                escalated = True
            except ProcessLookupError:
                pass
            sleep(0.5)
            residual = _process_group_members(process.pid)
            break
        sleep(0.1)
        residual = _process_group_members(process.pid)
    if process.poll() is None:
        try:
            process.wait(timeout=drain_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=drain_seconds)
    receipt = f"pid={process.pid}:exit={process.returncode}"
    if escalated:
        receipt += ":group-escalated"
    if residual:
        receipt += f":group-residual={list(residual)}"
    return receipt


def _process_group_members(pgid: int) -> tuple[int, ...]:
    """Return the live, non-zombie pids in a process group.

    Zombies are excluded deliberately: the leader is this process's own child
    and stays a zombie until it is reaped, so counting it would make the group
    look occupied forever and force an unnecessary SIGKILL on every clean run.
    A zombie also holds no socket, which is the only thing the caller cares
    about.
    """
    members: list[int] = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc.name)
            entry = _stat_fields(pid)
            if int(entry[2]) == pgid and entry[0] != "Z":
                members.append(pid)
        except (OSError, ValueError, IndexError):
            continue
    return tuple(sorted(members))


def wait_for_port_release(port: int, *, grace_seconds: float, sleep: Callable[[float], None] = time.sleep) -> tuple[str, ...]:
    """Poll a TCP port until no socket listens on it, or the grace window ends.

    The player is a grandchild of the JVM and unbinds on its own schedule after
    the group signal, so reading the port once, right after `terminate`, races
    its teardown and would report a healthy run as "listener remained". The
    grace window only ever delays a rejection; a port that never drains still
    rejects.
    """
    deadline = time.monotonic() + grace_seconds
    while True:
        inodes = _proc_net_listeners(port)
        if not inodes or time.monotonic() >= deadline:
            return inodes
        sleep(0.5)


def launch_environment(display: str, environ: Mapping[str, str]) -> tuple[dict[str, str], tuple[str, ...]]:
    """Return the child environment and the variables dropped from it.

    Refuses code-interposition variables outright and drops the loader search
    path. Without this the launch environment is the one input to the gate that
    nothing pins: `LD_PRELOAD=/tmp/patch.so` loads native code that is in none
    of the candidate digests and can rewrite the very collision and capture
    payloads this smoke exists to measure, while every digest, the archive, and
    tree membership all still pass.
    """
    present = sorted(name for name in INTERPOSITION_ENV_VARS if environ.get(name))
    if present:
        raise SmokeError(f"refusing to launch with code-interposition environment set: {', '.join(present)}")
    stripped = tuple(sorted(name for name in STRIPPED_ENV_VARS if environ.get(name)))
    child = {name: value for name, value in environ.items() if name not in STRIPPED_ENV_VARS}
    child["DISPLAY"] = display
    return child, stripped


def start_display(log_path: Path) -> tuple[str, subprocess.Popen[bytes]]:
    """Start a private Xvfb display for this run."""
    display = f":{190 + (os.getpid() % 50)}"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            ["Xvnc", display, "-geometry", "1024x768", "-depth", "24", "-SecurityTypes", "None", "-rfbport", "0"],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    time.sleep(0.25)
    if process.poll() is not None:
        raise SmokeError("Xvnc failed to start")
    return display, process


def _append_report_entry(report: JsonObject, key: str, value: str) -> None:
    """Append to a list-valued report field, creating it on first use."""
    entries = report.get(key)
    if not isinstance(entries, list):
        entries = []
        report[key] = entries
    entries.append(value)


def _artifact_identity(path: Path) -> tuple[int, int]:
    """Return a filesystem identity for an artifact; absence raises.

    `Path.exists()` swallows ENOENT/ENOTDIR/EBADF/ELOOP; ESTALE, EACCES and EIO
    propagate. A symlink loop is therefore invisible to `exists()` but visible
    here, which is the point: every probe of the output namespace has to be an
    explicit stat whose failure the caller can record rather than silently read
    as "absent".
    """
    status = path.stat()
    return (status.st_dev, status.st_ino)


def _free_destination(directory: Path, name: str) -> Path:
    """Return a path in `directory` that nothing currently occupies.

    The probe is `_artifact_identity`, so a name occupied by something that
    cannot be stat'ed counts as occupied rather than free -- overwriting it
    would destroy evidence this process cannot even see.
    """
    destination = directory / name
    for collision in range(1, 1024):
        try:
            _artifact_identity(destination)
        except FileNotFoundError:
            return destination
        destination = directory / f"{name}.{collision}"
    raise SmokeError(f"no free name for {directory / name}")


def preexisting_artifacts(output_dir: Path, names: tuple[str, ...]) -> tuple[dict[str, tuple[int, int]], tuple[str, ...]]:
    """Identify artifacts already present before this run, by inode not by name.

    Returns the identities that were determinable and the names that were not.
    Both halves matter and neither may swallow the other:

    - A name-keyed exemption is exploitable: `os.replace` succeeds onto an
      existing empty directory, so this run's own artifact could be published
      onto an occupied name and inherit the "belongs to an earlier run" skip.
      Identity is captured instead, and re-checked before any skip.
    - A probe that fails must not discard the identities already collected.
      Aborting on the first failure meant a symlink loop named `shot_001.tmp`
      erased the record of a real prior `shot_001`, after which this run
      quarantined an earlier run's evidence. An unprobeable name is returned as
      blind, and blind means untouchable.
    """
    identities: dict[str, tuple[int, int]] = {}
    blind: list[str] = []
    for name in names:
        try:
            identities[name] = _artifact_identity(output_dir / name)
        except FileNotFoundError:
            continue
        except OSError:
            blind.append(name)
    return identities, tuple(blind)


def quarantine_artifact(report: JsonObject, output_dir: Path, name: str, preexisting: Mapping[str, tuple[int, int]], blind: tuple[str, ...] = ()) -> None:
    """Move a rejected run artifact out of the accepted namespace, fail-safe.

    Five properties this must hold, each of which was violated by an earlier
    version:

    1. Nothing here may raise. This runs from the exception handler and from the
       `finally` block, where an escaping error would skip the report publish and
       leave a prior run's `accepted` receipt readable at the designated path.
       That includes the existence probe, which is why it is inside a `try`.
    2. The move itself can fail -- `Path.replace` raises `ENOTEMPTY` when a prior
       quarantine already occupies the destination -- and is recorded, not raised.
       The move's own ENOENT is recorded too, which is why the probe has a
       separate `try` rather than sharing the outer one.
    3. Both candidates can exist, so every destination is recorded, not just the
       last one.
    4. An artifact that existed before this run started belongs to an earlier run
       and is left alone; quarantining it would silently mutate prior evidence.
       The skip is keyed on inode identity, so an artifact this run wrote onto a
       previously occupied name is still quarantined.
    5. A name whose prior state could not be determined is untouchable. "Cannot
       tell whose it is" must not resolve to "mine to move".
    """
    source = output_dir / name
    if name in blind:
        _append_report_entry(report, "quarantine_skipped_blind", str(source))
        return
    try:
        identity = _artifact_identity(source)
    except FileNotFoundError:
        return
    except OSError as error:
        _append_report_entry(report, "quarantine_errors", f"{source}: {error}")
        return
    if preexisting.get(name) == identity:
        return
    try:
        holding = output_dir / "invalid_attempts"
        holding.mkdir(parents=True, exist_ok=True)
        destination = _free_destination(holding, name)
        source.replace(destination)
        _append_report_entry(report, "quarantine", str(destination))
    except (OSError, SmokeError) as error:
        # A silent return here would be indistinguishable from "nothing needed
        # quarantining", so every failed move is recorded -- including its
        # ENOENT, which is why this block no longer returns early on absence.
        _append_report_entry(report, "quarantine_errors", f"{source}: {error}")


def supersede_prior_report(report_path: Path) -> str:
    """Displace any receipt already at the designated path, before anything runs.

    The publish at the end of a run is atomic, which removes torn writes but
    leaves a prior run's `accepted` receipt readable at the designated path for
    the whole run -- and permanently if this process is killed, or if both the
    publish and its unlink fallback fail for the same reason (a read-only
    remount, EACCES, quota). A downstream gate reading it would consume a stale
    accept. Displacing it up front closes both windows.

    The prior receipt is moved to a free name, never deleted or overwritten, so
    earlier evidence survives.
    """
    try:
        destination = _free_destination(report_path.parent, f"{report_path.name}.superseded")
        report_path.replace(destination)
    except FileNotFoundError:
        return ""
    except (OSError, SmokeError) as error:
        raise SmokeError(f"cannot displace the prior report at {report_path}, so a stale receipt could outlive this run: {error}") from error
    return str(destination)



def run_smoke(args: argparse.Namespace) -> tuple[JsonObject, int]:
    """Execute the smoke scenario and write its report before returning."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    protected = protected_roots(args.canonical_root)
    before = {name: protected_receipt(name, path) for name, path in protected.items()}
    # An artifact already present belongs to an earlier run; this run may neither
    # quarantine it nor be credited with it. Identity is captured rather than the
    # bare name, and a name that could not be probed is carried as blind so it
    # stays untouchable instead of defaulting to "mine".
    preexisting, blind = preexisting_artifacts(args.output_dir, ARTIFACT_NAMES)
    # `run` is audit evidence only. No consumer gates on it, exactly as with
    # `host_uninspectable_pid_count`; staleness is closed by displacing the prior
    # receipt below, not by asking a reader to compare identities.
    report: JsonObject = {"status": "rejected", "accepted_shot": None, "run": {"pid": os.getpid(), "started_unix_ns": time.time_ns()}, "canonical_root": str(args.canonical_root), "protected_before": before, "preexisting_artifacts": sorted(preexisting), "unprobeable_artifacts": list(blind), "protected_receipt_mode": {"canonical_project": "content_sha256", "production_player": "content_sha256", "active_data": "complete_nested_find_manifest_sha256"}}
    engine: subprocess.Popen[bytes] | None = None
    xvfb: subprocess.Popen[bytes] | None = None
    agent_teardown = "not-started"
    temporary_path = ""
    result_code = 1
    try:
        # First action inside the guarded block: nothing has launched, so a prior
        # `accepted` receipt is never readable at the designated path while this
        # run is in flight.
        report["superseded_report"] = supersede_prior_report(args.report)
        if blind:
            raise SmokeError(f"cannot determine whether these output artifacts predate this run: {list(blind)}")
        if preexisting:
            # A dirty output namespace is refused outright rather than tolerated.
            # Publishing onto an occupied name is what let a run's own artifact
            # inherit an earlier run's exemption, and this gate runs exactly once.
            raise SmokeError(f"output directory already holds prior artifacts: {sorted(preexisting)}")
        with tempfile.TemporaryDirectory(prefix="novphy-physics-smoke-") as temporary:
            temporary_path = temporary
            clone = Path(temporary) / "player"
            report["phase"] = "verify-stage"
            stage_archive, archive_sha, player_sha, protocol_sha, assembly_sha = archive_details(args.stage, clone)
            expectation = candidate_expectation(clone, archive_sha)
            report["stage_provenance"] = {"archive_sha256": archive_sha, "player_sha256": player_sha, "protocol_sha256": protocol_sha, "assembly_csharp_sha256": assembly_sha, "candidate_root": str(clone), "provenance_sha256": expectation.provenance_sha256, "assembly_device": expectation.assembly_device, "assembly_inode": expectation.assembly_inode}
            report["phase"] = "start-display"
            display, xvfb = start_display(args.output_dir / "xvfb.log")
            agent_port = args.agent_port or free_port()
            game_port = args.game_port or free_port()
            engine_log = args.output_dir / "engine.log"
            report.update({"phase": "start-engine", "ports": {"agent": agent_port, "game": game_port, "physics": args.physics_port}, "display": display})
            child_env, stripped_env = launch_environment(display, os.environ)
            report["launch_environment"] = {
                "interposition_vars_refused": list(INTERPOSITION_ENV_VARS),
                "vars_stripped": list(stripped_env),
                "sha256": hashlib.sha256("\n".join(f"{name}={child_env[name]}" for name in sorted(child_env)).encode("utf-8")).hexdigest(),
            }
            with engine_log.open("wb") as engine_stream:
                engine = subprocess.Popen(
                    ["java", "-jar", "./game_playing_interface.jar", "--agent-port", str(agent_port), "--game-start-port", str(game_port), "--physics-port", str(args.physics_port), "--dev"],
                    cwd=clone, env=child_env, stdout=engine_stream, stderr=subprocess.STDOUT, start_new_session=True,
                )

            def bind_candidate() -> dict[str, object]:
                return wait_for_listener(
                    args.physics_port,
                    engine.pid,
                    expectation,
                    lambda owners: _observe_listener_owners(owners, engine.pid, expectation, stage_archive),
                    deadline_seconds=args.listener_deadline_seconds,
                )

            # The jar does not own the physics port and does not spawn the player
            # that does until an agent completes its handshake: measured directly,
            # 60s of polling with the jar alone gives `player_procs=0 port2004=0`
            # at every sample. So the connect and configure have to precede the
            # bind in *both* modes -- binding first cannot time out into a
            # diagnosis, it can only ever time out.
            report["phase"] = "connect-agent"
            # This handshake triggers Unity cold start, so the socket read
            # timeout sits on the critical path of a process launch measured in
            # tens of seconds and cannot be small. It is its own flag rather
            # than a function of `--listener-deadline-seconds`: the connect
            # precedes the bind, so the listener deadline does not bound it, and
            # deriving it from that flag silently applied a 600s read timeout to
            # every later gameplay read whenever the listener budget was raised.
            bridge = connect_with_retry("127.0.0.1", agent_port, timeout=args.agent_timeout_seconds, deadline_seconds=90.0)
            try:
                report["phase"] = "configure-agent"
                bridge.configure(agent_id=28701, mode=PlayingMode.TRAINING)
                report["phase"] = "bind-physics-listener"
                binding = bind_candidate()
                report["listener_binding"] = binding
                if args.listener_only:
                    # Stops before any level load or shot, so the diagnostic
                    # proves the binding without consuming the one full smoke.
                    report.update({"status": "diagnostic_accepted", "phase": "listener-bound"})
                    result_code = 0
                else:
                    report["phase"] = "load-known-level"
                    prepare_for_play(bridge, timeout=120.0, poll_delay=1.0)
                    report["phase"] = "perform-action"
                    report["action"] = perform_known_action(bridge)
                    if args.inject_request_failure:
                        raise SmokeError("injected request-70 failure")
                    shot = args.output_dir / "shot_001.tmp"
                    report["phase"] = "request-70"
                    first_capture = capture_finalized_action(args.physics_port)
                    second_capture = capture_finalized_action(args.physics_port)
                    missing_fields = sorted(WIRE_STATE_FIELDS - second_capture.state.keys())
                    wire_capture: JsonObject = {
                        "state_fields": sorted(second_capture.state),
                        "missing_state_fields": missing_fields,
                        "event_count": len(second_capture.events),
                        "event_types": [str(event.get("event_type", "")) for event in second_capture.events],
                        "render_frame": second_capture.state.get("render_frame"),
                    }
                    report["wire_capture"] = wire_capture
                    wire_capture["request_identities"] = list(require_request_identity((first_capture, second_capture)))
                    wire_capture["collision"] = require_collision(second_capture.events)
                    report["phase"] = "rebind-physics-listener"
                    rebinding = bind_candidate()
                    report["listener_rebinding"] = rebinding
                    report["binding_stability"] = require_stable_binding(binding, rebinding)
                    report["phase"] = "validate-wire-capture"
                    require_action_events(second_capture.events)
                    if missing_fields:
                        raise SmokeError("request-70 state missing contract fields: " + ", ".join(missing_fields))
                    capture_physics_rollout(CapturedRequest(second_capture), shot, target_fps=1.0, duration_seconds=1.0, max_frames=1, player_sha256=player_sha, protocol_sha256=protocol_sha, archive_sha256=archive_sha)
                    if args.inject_frame_mismatch:
                        state_path = shot / "physics_state.jsonl"
                        lines = state_path.read_text(encoding="utf-8").splitlines()
                        record = json.loads(lines[1])
                        record["rgb_frame"]["render_frame"] += 1
                        lines[1] = json.dumps(record, sort_keys=True, separators=(",", ":"))
                        state_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    if args.inject_missing_sidecar:
                        (shot / "physics_events.jsonl").unlink()
                    report["phase"] = "validate-artifact"
                    summary = validate_physics_shot_artifact(shot)
                    final_shot = args.output_dir / "shot_001"
                    shot.replace(final_shot)
                    report.update({"status": "accepted", "phase": "complete", "accepted_shot": str(final_shot), "artifact": {"states": summary.state_count, "events": summary.event_count}, "provenance": {"archive_sha256": archive_sha, "player_sha256": player_sha, "protocol_sha256": protocol_sha, "assembly_csharp_sha256": assembly_sha}})
                    result_code = 0
            finally:
                # Covers the listener-only path too, which now holds a bridge.
                # The disconnect trails artifact validation rather than preceding
                # it: nothing in validation talks to the agent, and keeping the
                # player alive until the receipt is decided costs nothing.
                #
                # It must not raise. A bare `disconnect()` here would replace the
                # in-flight exception with a socket error, destroying the run's
                # only recorded rejection reason. So it is recorded exactly as
                # engine/xvfb cleanup is: a receipt, folded into `failures`
                # below. Note that a failed teardown does still reject the run,
                # deliberately -- a socket this gate cannot close cleanly is a
                # cleanup fact, and cleanup facts are fail-closed here.
                try:
                    bridge.disconnect()
                    agent_teardown = "disconnected"
                except (OSError, SmokeError) as error:
                    agent_teardown = f"cleanup-failed: {error}"
    except (OSError, ValueError, LookupError, RuntimeError, SmokeError, ProtectionError, PhysicsArtifactError, tarfile.TarError, TimeoutError, TypeError) as error:
        report["error"] = str(error)
        # A failure after the accept line (for example a rmtree failure leaving the
        # `with` block) must never leave "accepted" persisted for a downstream gate.
        # The pop precedes quarantining so that a quarantine failure cannot leave a
        # rejected report still naming an accepted shot.
        report["status"] = "rejected"
        report.pop("accepted_shot", None)
        for candidate in ARTIFACT_NAMES:
            quarantine_artifact(report, args.output_dir, candidate, preexisting, blind)
        result_code = 1
    finally:
        # An exception whose class is outside the handler tuple above unwinds
        # through here with `report["status"]` still holding whatever the last
        # successful phase set -- including "accepted". Publishing that would
        # hand a downstream gate an accepted receipt for a run that raised.
        if sys.exc_info()[0] is not None:
            report["status"] = "rejected"
            report.pop("accepted_shot", None)
        cleanup: JsonObject = {"temporary_clone": temporary_path, "temporary_clone_removed": not temporary_path or not Path(temporary_path).exists(), "agent": agent_teardown}
        report["cleanup"] = cleanup
        failures: list[str] = []
        if agent_teardown.startswith("cleanup-failed"):
            failures.append(f"agent {agent_teardown}")
        for name, process in (("engine", engine), ("xvfb", xvfb)):
            try:
                cleanup[name] = terminate(process)
            except (OSError, subprocess.SubprocessError) as error:
                cleanup[name] = f"cleanup-failed: {error}"
                failures.append(f"{name} cleanup failed: {error}")
            else:
                if ":group-residual=" in str(cleanup[name]):
                    failures.append(f"{name} process group survived SIGKILL: {cleanup[name]}")
        try:
            listener_inodes = wait_for_port_release(args.physics_port, grace_seconds=args.port_grace_seconds)
            # An empty inode set makes the scan a full `/proc` walk that can only
            # produce blind-spot noise, so it is skipped.
            scan = _socket_owner_scan(listener_inodes) if listener_inodes else SocketOwnerScan((), ())
            cleanup["physics_listener_inodes_after"] = list(listener_inodes)
            cleanup["physics_listener_owners_after"] = [list(owner) for owner in scan.owners]
            cleanup["physics_listener_uninspectable_pids_after"] = list(scan.unreadable_pids)
            cleanup["physics_port_clear"] = not listener_inodes
            if listener_inodes:
                failures.append("physics listener remained after cleanup")
        except (OSError, SmokeError) as error:
            # An unreadable or entirely absent /proc/net table means the port state
            # is unknown, which must be recorded as not-clear rather than as a
            # drained port.
            cleanup["physics_port_clear"] = False
            cleanup["physics_listener_scan_error"] = str(error)
            failures.append(f"listener cleanup check failed: {error}")
        try:
            after = {name: protected_receipt(name, path) for name, path in protected.items()}
            report["protected_after"] = after
            report["protected_unchanged"] = before == after
            if before != after:
                failures.append("protected roots changed during smoke")
        except (OSError, ProtectionError) as error:
            report["protected_unchanged"] = False
            failures.append(f"protected-root comparison failed: {error}")
        if failures:
            report.update({"status": "rejected", "cleanup_failures": failures, "error": report.get("error") or failures[0]})
            # A run rejected only at cleanup has already published shot_001; leaving
            # it in place would look identical to an accepted artifact. The pop is
            # unconditional because a failed quarantine must not leave a rejected
            # report still naming an accepted shot.
            report.pop("accepted_shot", None)
            quarantine_artifact(report, args.output_dir, "shot_001", preexisting, blind)
            result_code = 1
        staging = args.report.with_name(f"{args.report.name}.{os.getpid()}.tmp")
        try:
            staging.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            staging.replace(args.report)
        except (OSError, TypeError, ValueError) as error:
            # The prior receipt was already displaced at the head of the run, so a
            # failed publish leaves the designated path empty rather than holding
            # a stale `accepted`. The consumer requires the marker file to exist,
            # so an absent receipt is fail-closed. The staging file is still
            # removed so it is not left behind as litter.
            print(f"failed to persist smoke report to {args.report}: {error}", file=sys.stderr)
            # The returned dict is also printed to stdout by `main`, so it has to
            # agree with the empty disk state, and the artifact must not outlive
            # the receipt that would have described it. The original failure
            # reason is preserved: stdout is now the only surviving record, and
            # overwriting `error` would destroy the only statement of why the run
            # failed in the first place.
            report["status"] = "rejected"
            report["publish_error"] = f"report publish failed: {error}"
            report.pop("accepted_shot", None)
            quarantine_artifact(report, args.output_dir, "shot_001", preexisting, blind)
            for leftover in (args.report, staging):
                try:
                    leftover.unlink(missing_ok=True)
                except OSError as removal_error:
                    print(f"failed to remove {leftover}: {removal_error}", file=sys.stderr)
            result_code = 1
    return report, result_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--agent-port", type=int)
    parser.add_argument("--game-port", type=int)
    parser.add_argument("--physics-port", type=int, default=2004)
    parser.add_argument("--listener-deadline-seconds", type=float, default=30.0)
    parser.add_argument("--agent-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--port-grace-seconds", type=float, default=20.0)
    parser.add_argument("--listener-only", action="store_true")
    parser.add_argument("--canonical-root", type=Path, default=canonical_root_from_git(ROOT))
    parser.add_argument("--inject-frame-mismatch", action="store_true")
    parser.add_argument("--inject-missing-sidecar", action="store_true")
    parser.add_argument("--inject-request-failure", action="store_true")
    args = parser.parse_args()
    if args.report is None:
        # A diagnostic is not a smoke and must not be able to masquerade as one.
        # Sharing the designated marker path meant `--listener-only` displaced a
        # real accepted receipt to `.superseded` and wrote `diagnostic_accepted`
        # with exit 0 in its place, which any step keying on exit status alone
        # reads as a passed smoke.
        args.report = DEFAULT_DIAGNOSTIC_REPORT if args.listener_only else DEFAULT_SMOKE_REPORT
    report, code = run_smoke(args)
    print(json.dumps(report, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
