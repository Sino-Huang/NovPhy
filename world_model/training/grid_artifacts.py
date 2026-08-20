"""Versioned, provenance-bound best-pair sweep artifacts."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

SCHEMA_VERSION: Final = "best_pair_labels_v1"
SHARD_SIZE: Final = 4096
APPROVED_DELTAS: Final = (1, 5, 15)
ALPHA_EXCLUSIONS: Final = (
    {"alpha": "micro", "reason": "symbolic_supervision_unavailable"},
    {"alpha": "macro", "reason": "symbolic_supervision_unavailable"},
)
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"] | tuple["JsonValue", ...]


class ArtifactContractError(ValueError):
    """Raised for malformed or unsafe artifact input."""


class ArtifactValidationError(ArtifactContractError):
    """Raised when persisted artifacts fail structural validation."""


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Encode JSON with the artifact's stable byte-level rules."""
    try:
        text = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ArtifactContractError("JSON must contain only finite JSON values") from error
    return (text + "\n").encode("ascii")


def _identity(value: str, field: str) -> None:
    if type(value) is not str or not value.strip():
        raise ArtifactContractError(f"{field} must be a nonempty declared identity")


def _finite(value: float, field: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ArtifactContractError(f"{field} must be finite")


@dataclass(frozen=True, slots=True)
class PairMetricArtifact:
    delta: int
    alpha: str
    requested_delta: int
    effective_delta: int
    weighted_error: float
    compute_cost: float
    availability: str

    def __post_init__(self) -> None:
        if self.delta not in APPROVED_DELTAS or self.alpha != "continuous":
            raise ArtifactContractError("metrics only permit approved continuous pairs")
        if self.requested_delta != self.delta or type(self.effective_delta) is not int or not 1 <= self.effective_delta <= self.delta:
            raise ArtifactContractError("requested/effective delta is inconsistent")
        _finite(self.weighted_error, "weighted_error")
        _finite(self.compute_cost, "compute_cost")
        if self.weighted_error < 0 or self.compute_cost < 0 or self.availability not in ("available", "unavailable"):
            raise ArtifactContractError("invalid metric values")

    def to_dict(self) -> dict[str, JsonValue]:
        return {"alpha": self.alpha, "availability": self.availability, "compute_cost": self.compute_cost, "delta": self.delta, "effective_delta": self.effective_delta, "requested_delta": self.requested_delta, "weighted_error": self.weighted_error}


@dataclass(frozen=True, slots=True)
class BestPairState:
    state_index: int
    temporal_ceiling: int
    metrics: tuple[PairMetricArtifact, ...]
    pareto_points: tuple[tuple[int, float, float], ...]
    selected_delta: int
    selected_alpha: str

    def __post_init__(self) -> None:
        if type(self.state_index) is not int or self.state_index < 0 or type(self.temporal_ceiling) is not int or self.temporal_ceiling < 1:
            raise ArtifactContractError("invalid state index or temporal ceiling")
        if type(self.metrics) is not tuple or not self.metrics or len({(m.delta, m.alpha) for m in self.metrics}) != len(self.metrics):
            raise ArtifactContractError("metrics must be nonempty and unique")
        if self.selected_alpha != "continuous" or self.selected_delta not in {m.delta for m in self.metrics}:
            raise ArtifactContractError("selected pair is unavailable")
        for delta, error, cost in self.pareto_points:
            if type(delta) is not int or delta <= 0:
                raise ArtifactContractError("invalid Pareto delta")
            _finite(error, "pareto error"); _finite(cost, "pareto cost")

    def to_dict(self) -> dict[str, JsonValue]:
        return {"record_type": "state", "schema_version": SCHEMA_VERSION, "state_index": self.state_index, "temporal_ceiling": self.temporal_ceiling, "metrics": [m.to_dict() for m in self.metrics], "pareto_points": [{"compute_cost": c, "delta": d, "weighted_error": e} for d, e, c in self.pareto_points], "selected": {"alpha": self.selected_alpha, "delta": self.selected_delta}}


@dataclass(frozen=True, slots=True)
class SweepManifest:
    source_identity: str
    checkpoint_path: str
    catalog_identity: str
    grid_identity: str
    score_identity: str
    partition_identity: str
    state_count: int
    shard_size: int = SHARD_SIZE

    def __post_init__(self) -> None:
        for field in ("source_identity", "checkpoint_path", "catalog_identity", "grid_identity", "score_identity", "partition_identity"):
            _identity(getattr(self, field), field)
        if self.state_count <= 0 or self.shard_size != SHARD_SIZE or self.state_count % self.shard_size:
            raise ArtifactContractError("state_count must use complete 4096-state shards")

    def to_dict(self, shards: tuple[dict[str, JsonValue], ...] = ()) -> dict[str, JsonValue]:
        return {"schema_version": SCHEMA_VERSION, "artifact_type": "best_pair_sweep", "source_identity": self.source_identity, "checkpoint_path": self.checkpoint_path, "catalog_identity": self.catalog_identity, "grid_identity": self.grid_identity, "score_identity": self.score_identity, "partition_identity": self.partition_identity, "state_count": self.state_count, "shard_size": self.shard_size, "evaluated_alpha": "continuous", "excluded_abstractions": list(ALPHA_EXCLUSIONS), "shards": list(shards)}


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    shards: tuple[str, ...]


def _reject_output(root: Path) -> None:
    resolved = root.resolve()
    parts = resolved.parts
    if "frames" in parts or "sciencebirdsgames" in parts and "Linux" in parts:
        raise ArtifactContractError("artifact output cannot be beside dataset frames or protected player")
    if any(part.startswith("novphy_rollouts_dataset_") for part in parts):
        raise ArtifactContractError("artifact output cannot be inside protected dataset")


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _shard_bytes(states: tuple[BestPairState, ...]) -> bytes:
    header = {"record_type": "shard_header", "schema_version": SCHEMA_VERSION, "shard_size": SHARD_SIZE, "evaluated_alpha": "continuous", "excluded_abstractions": list(ALPHA_EXCLUSIONS), "state_count": len(states)}
    footer = {"record_type": "shard_footer", "schema_version": SCHEMA_VERSION, "state_count": len(states)}
    return b"".join((canonical_json_bytes(header), *(canonical_json_bytes(s.to_dict()) for s in states), canonical_json_bytes(footer)))


def _parse_state(record: dict[str, JsonValue], expected_index: int) -> None:
    required = {"record_type", "schema_version", "state_index", "temporal_ceiling", "metrics", "pareto_points", "selected"}
    if set(record) != required or record.get("state_index") != expected_index:
        raise ArtifactValidationError("closed state schema or ordering violation")
    metrics = record["metrics"]
    if type(metrics) is not list:
        raise ArtifactValidationError("metrics must be a list")
    parsed: list[PairMetricArtifact] = []
    for metric in metrics:
        if type(metric) is not dict:
            raise ArtifactValidationError("metric must be an object")
        try:
            parsed.append(PairMetricArtifact(metric["delta"], metric["alpha"], metric["requested_delta"], metric["effective_delta"], metric["weighted_error"], metric["compute_cost"], metric["availability"]))
        except (KeyError, TypeError, ArtifactContractError) as error:
            raise ArtifactValidationError("invalid per-pair metric") from error
    try:
        BestPairState(expected_index, record["temporal_ceiling"], tuple(parsed), tuple((point["delta"], point["weighted_error"], point["compute_cost"]) for point in record["pareto_points"]), record["selected"]["delta"], record["selected"]["alpha"])
    except (KeyError, TypeError, ArtifactContractError) as error:
        raise ArtifactValidationError("invalid state contract") from error


def write_best_pair_artifacts(root: Path, manifest: SweepManifest, states: tuple[BestPairState, ...], *, resume: bool = False) -> ArtifactReceipt:
    """Write complete shards atomically, then publish the top-level manifest."""
    _reject_output(root)
    if len(states) != manifest.state_count or any(state.state_index != index for index, state in enumerate(states)):
        raise ArtifactContractError("states must be contiguous and match manifest count")
    root.mkdir(parents=True, exist_ok=True)
    shard_entries: list[dict[str, JsonValue]] = []
    names: list[str] = []
    for offset in range(0, len(states), SHARD_SIZE):
        shard_states = states[offset:offset + SHARD_SIZE]
        name = f"shard-{offset // SHARD_SIZE:06d}.jsonl"
        data = _shard_bytes(shard_states)
        path = root / name
        if resume and path.exists() and path.read_bytes() != data:
            raise ArtifactValidationError(f"stale or tampered shard: {name}")
        if not path.exists():
            _atomic_write(path, data)
        names.append(name)
        shard_entries.append({"name": name, "state_count": len(shard_states)})
    payload = manifest.to_dict(tuple(shard_entries))
    _atomic_write(root / "manifest.json", canonical_json_bytes(payload))
    return ArtifactReceipt(tuple(names))


def validate_best_pair_artifacts(root: Path, expected: SweepManifest | None = None) -> ArtifactReceipt:
    """Validate manifest, shard records, ordering, and counts."""
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactValidationError("missing top-level manifest")
    raw = manifest_path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ArtifactValidationError("invalid manifest JSON") from error
    if canonical_json_bytes(payload) != raw:
        raise ArtifactValidationError("manifest is not canonical JSON")
    required = {"schema_version", "artifact_type", "source_identity", "checkpoint_path", "catalog_identity", "grid_identity", "score_identity", "partition_identity", "state_count", "shard_size", "evaluated_alpha", "excluded_abstractions", "shards"}
    if set(payload) != required or payload["schema_version"] != SCHEMA_VERSION or payload["evaluated_alpha"] != "continuous" or payload["excluded_abstractions"] != list(ALPHA_EXCLUSIONS):
        raise ArtifactValidationError("closed manifest schema violation")
    try:
        manifest = SweepManifest(*(payload[field] for field in ("source_identity", "checkpoint_path", "catalog_identity", "grid_identity", "score_identity", "partition_identity")), payload["state_count"], payload["shard_size"])
    except (ArtifactContractError, TypeError) as error:
        raise ArtifactValidationError("invalid manifest fields") from error
    if expected is not None and manifest != expected:
        raise ArtifactValidationError("manifest provenance or count mismatch")
    entries = payload["shards"]
    if type(entries) is not list or len(entries) != manifest.state_count // SHARD_SIZE:
        raise ArtifactValidationError("shard count mismatch")
    expected_index = 0; names: list[str] = []
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"name", "state_count"} or entry["state_count"] != SHARD_SIZE:
            raise ArtifactValidationError("invalid shard entry")
        name = entry["name"]; data = (root / name).read_bytes() if (root / name).is_file() else b""
        lines = data.splitlines()
        if len(lines) != SHARD_SIZE + 2:
            raise ArtifactValidationError(f"shard count mismatch: {name}")
        header = json.loads(lines[0])
        if canonical_json_bytes(header) != lines[0] + b"\n" or header != {"record_type": "shard_header", "schema_version": SCHEMA_VERSION, "shard_size": SHARD_SIZE, "evaluated_alpha": "continuous", "excluded_abstractions": list(ALPHA_EXCLUSIONS), "state_count": SHARD_SIZE}:
            raise ArtifactValidationError("invalid shard header")
        for line in lines[1:-1]:
            if canonical_json_bytes(json.loads(line)) != line + b"\n":
                raise ArtifactValidationError("noncanonical state record")
            record = json.loads(line)
            _parse_state(record, expected_index)
            expected_index += 1
        footer = json.loads(lines[-1])
        expected_footer = {"record_type": "shard_footer", "schema_version": SCHEMA_VERSION, "state_count": SHARD_SIZE}
        if canonical_json_bytes(footer) != lines[-1] + b"\n" or footer != expected_footer:
            raise ArtifactValidationError("invalid shard footer")
        names.append(name)
    if expected_index != manifest.state_count:
        raise ArtifactValidationError("state count mismatch")
    return ArtifactReceipt(tuple(names))


__all__ = ["ALPHA_EXCLUSIONS", "APPROVED_DELTAS", "ArtifactContractError", "ArtifactReceipt", "ArtifactValidationError", "BestPairState", "PairMetricArtifact", "SCHEMA_VERSION", "SHARD_SIZE", "SweepManifest", "canonical_json_bytes", "validate_best_pair_artifacts", "write_best_pair_artifacts"]
