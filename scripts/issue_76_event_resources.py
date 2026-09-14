"""Resource and isolation primitives for the frozen causal development inventory."""
from pathlib import Path
import socket
import stat

import torch

from scripts import issue_76_live_episode as live
from scripts import prepare_issue_76_event_development as inventory
from scripts.issue_76_fixed_replay_policy import FixedReplayPolicy


def worker_ports(slot):
    if slot not in range(8):
        raise ValueError("event worker slot must be in 0..7")
    return tuple(48400 + 3 * slot + offset for offset in range(3))


def check_ports_available(ports):
    sockets = []
    try:
        for port in ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(sock)
            sock.bind(("127.0.0.1", port))
    finally:
        for sock in sockets:
            sock.close()


def worker(root, source, assignment, limits, slot):
    """Only the spawned process overrides port allocation; no frozen source edits."""
    torch.set_num_threads(1)
    member = inventory.materialize_assignment(source, assignment)
    if member["exposure_role"] not in inventory.ROLES:
        raise ValueError("event capture cannot change or introduce an exposure role")
    ports = worker_ports(slot)
    check_ports_available(ports)
    remaining = iter(ports)
    original = live.capture.old.capture.free_port
    live.capture.old.capture.free_port = lambda: next(remaining)
    try:
        return live.play_episode(root, member, limits, FixedReplayPolicy(member["actions"][0]))
    finally:
        live.capture.old.capture.free_port = original


def _tree_bytes(root):
    total = 0
    for path in Path(root).rglob("*"):
        metadata = path.stat()
        if stat.S_ISREG(metadata.st_mode):
            total += metadata.st_size
    return total


def tree_bytes(root):
    # The existing capture publisher atomically renames observation directories
    # and chunk/JSON staging files. Discard the partial total and scan once more;
    # never turn a vanished staging entry into zero or retry the physics capture.
    try:
        return _tree_bytes(root)
    except FileNotFoundError:
        return _tree_bytes(root)


class ArtifactLedger:
    """Count immutable completed attempts once; scan only active attempts thereafter.

    The supervisor must seal only after the worker and its writers have exited.
    Persist `completed` alongside the receipts so normal continuation never
    rescans completed traces. Unexpected or unclean state requires an audit.
    """
    def __init__(self, root, publication_root, *, completed=None):
        self.root = Path(root)
        self.publication_root = Path(publication_root)
        self.completed = dict(completed or {})
        self.player_bytes = tree_bytes(self.root / "player")

    def seal(self, identity):
        if identity in self.completed:
            raise ValueError("completed attempt was already counted")
        value = tree_bytes(self.root / "attempts" / identity)
        self.completed[identity] = value
        return value

    def snapshot(self, active):
        if set(active) & self.completed.keys():
            raise ValueError("a sealed attempt cannot still be active")
        control = sum(tree_bytes(path) if path.is_dir() else path.stat().st_size
                      for path in self.root.iterdir() if path.name not in ("attempts", "player"))
        return (self.player_bytes + sum(self.completed.values()) + control
                + tree_bytes(self.publication_root)
                + sum(tree_bytes(self.root / "attempts" / identity) for identity in active))


def global_stop(limits, *, active_seconds, smoke_seconds, rss_mib, artifact_bytes, smoke):
    if active_seconds >= limits["collection_wall_seconds"]:
        return "collection_wall_limit"
    if smoke and smoke_seconds >= limits["smoke_wall_seconds"]:
        return "smoke_wall_limit"
    if rss_mib > limits["aggregate_cpu_rss_mib"]:
        return "aggregate_memory_limit"
    if artifact_bytes > limits["artifact_bytes"]:
        return "artifact_limit"
    return None


def attempt_stop(limits, *, wall_seconds, rss_mib, captures, frames):
    if wall_seconds >= limits["attempt_seconds"]:
        return "attempt_wall_limit"
    if rss_mib > limits["worker_cpu_rss_mib"]:
        return "worker_memory_limit"
    if captures > 1 or frames > limits["rgb_frames_per_shot_max"]:
        return "capture_or_frame_limit"
    return None
