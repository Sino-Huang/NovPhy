"""Resumable, checkpoint-bound artifacts for exhaustive temporal scores."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from world_model.training.grid_artifacts import JsonValue, canonical_json_bytes
from world_model.training.pair_grid import ScoreSpec
from world_model.training.scoring_metrics import aggregate_labels, oracle_ceiling, state_digest
from world_model.training.scoring_payloads import aggregate_payload, ceiling_payload, state_payload
from world_model.training.scoring_validation import ScoreShardStream


SCHEMA_VERSION = "exhaustive_pair_scores_v1"


@dataclass(frozen=True, slots=True)
class ScoreArtifactReceipt:
    manifest_digest: str
    state_count: int
    score_count: int


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _reject_output(root: Path) -> None:
    parts = root.resolve().parts
    if "frames" in parts or ("sciencebirdsgames" in parts and "Linux" in parts):
        from world_model.training.scoring import ScoreArtifactError
        raise ScoreArtifactError("score output cannot be written beside protected data")
    if any(part.startswith("novphy_rollouts_dataset_") for part in parts):
        from world_model.training.scoring import ScoreArtifactError
        raise ScoreArtifactError("score output cannot be written inside the protected dataset")


def _validate_result(result) -> None:
    from world_model.training.scoring import Partition, ScoreArtifactError
    if not result.scored_states or result.score_count != len(result.scored_states) * 3:
        raise ScoreArtifactError("every state must have exactly three scores")
    if {item.example.partition for item in result.scored_states} != set(Partition):
        raise ScoreArtifactError("all partitions must be present")
    for item in result.scored_states:
        if tuple(metric.pair.delta for metric in item.label.metrics) != (1, 5, 15):
            raise ScoreArtifactError("state scores must contain ordered deltas 1, 5, 15")
    calibration = tuple(
        metric.weighted_prediction_error
        for item in result.scored_states if item.example.partition is Partition.CALIBRATION
        for metric in item.label.metrics
    )
    if ScoreSpec.from_calibration(calibration) != result.score_spec:
        raise ScoreArtifactError("error_scale differs from calibration P90")
    expected_labels = tuple(item for item in result.scored_states if item.example.partition is not Partition.CALIBRATION)
    if result.labels != expected_labels:
        raise ScoreArtifactError("published label partitions are incomplete or mixed")
    rows = tuple((str(item.example.partition), item.example.motion_regime, item.example.state_id, item.label) for item in result.labels)
    if aggregate_labels(rows) != result.per_pair_metrics:
        raise ScoreArtifactError("per-pair metrics do not recompute from state scores")
    if oracle_ceiling(rows, result.score_spec.error_scale) != result.temporal_oracle_ceiling:
        raise ScoreArtifactError("temporal oracle ceiling does not recompute")


def write_score_artifacts(
    root: Path,
    result,
    *,
    checkpoint_digest: str,
    resume: bool = False,
    shard_size: int = 4096,
) -> ScoreArtifactReceipt:
    from world_model.training.scoring import Partition, ScoreArtifactError
    _reject_output(root)
    _validate_result(result)
    if len(checkpoint_digest) != 64 or any(char not in "0123456789abcdef" for char in checkpoint_digest):
        raise ScoreArtifactError("checkpoint_digest must be lowercase SHA-256")
    if type(shard_size) is not int or shard_size <= 0:
        raise ScoreArtifactError("shard_size must be positive")
    entries: list[dict[str, JsonValue]] = []
    shard_data: list[tuple[Path, bytes]] = []
    for partition in Partition:
        states = tuple(item for item in result.scored_states if item.example.partition is partition)
        for offset in range(0, len(states), shard_size):
            batch = states[offset:offset + shard_size]
            data = b"".join(canonical_json_bytes(state_payload(item)) for item in batch)
            relative = Path("label_shards") / str(partition) / f"shard-{offset // shard_size:06d}.jsonl"
            shard_data.append((relative, data))
            entries.append({"name": relative.as_posix(), "sha256": _digest(data), "state_count": len(batch)})
    manifest = {
        "checkpoint_digest": checkpoint_digest,
        "error_scale": result.score_spec.error_scale,
        "label_partitions": [str(Partition.CONTROLLER_TRAIN), str(Partition.EVALUATION)],
        "schema_version": SCHEMA_VERSION,
        "score_count": result.score_count,
        "score_spec_digest": result.score_spec.identity,
        "shard_size": shard_size,
        "shards": entries,
        "state_count": len(result.scored_states),
        "state_digest": state_digest(
            tuple(
                item.example.state_id
                for partition in Partition
                for item in result.scored_states
                if item.example.partition is partition
            )
        ),
    }
    raw = canonical_json_bytes(manifest)
    expected_shards = {entry["name"] for entry in entries}
    shard_root = root / "label_shards"
    actual_shards = {
        path.relative_to(root).as_posix()
        for path in shard_root.rglob("*.jsonl")
    } if shard_root.is_dir() else set()
    existing_manifest = root / "manifest.json"
    if resume and existing_manifest.is_file():
        try:
            existing_raw = existing_manifest.read_bytes()
            existing = json.loads(existing_raw)
        except json.JSONDecodeError as error:
            raise ScoreArtifactError("stale score manifest") from error
        if existing_raw != raw or existing != manifest or actual_shards != expected_shards:
            raise ScoreArtifactError("resume score topology or binding mismatch")
    elif resume and actual_shards - expected_shards:
        raise ScoreArtifactError("resume score topology or binding mismatch")
    for relative, data in shard_data:
        path = root / relative
        if resume and path.is_file() and path.read_bytes() != data:
            raise ScoreArtifactError(f"stale or tampered shard: {relative}")
        if not path.is_file():
            _atomic_write(path, data)
    metrics_data = canonical_json_bytes([aggregate_payload(item) for item in result.per_pair_metrics])
    ceiling_data = canonical_json_bytes(ceiling_payload(result.temporal_oracle_ceiling))
    unavailable_data = canonical_json_bytes([
        {"metric": item.metric, "reason": item.reason, "status": item.status}
        for item in result.unavailable_metrics
    ])
    _atomic_write(root / "per_pair_metrics.json", metrics_data)
    _atomic_write(root / "temporal_oracle_ceiling.json", ceiling_data)
    _atomic_write(root / "unavailable_metrics.json", unavailable_data)
    _atomic_write(root / "manifest.json", raw)
    return ScoreArtifactReceipt(_digest(raw), len(result.scored_states), result.score_count)


def validate_score_artifacts(root: Path) -> ScoreArtifactReceipt:
    from world_model.training.grid_data import MotionRegime
    from world_model.training.scoring import Partition, ScoreArtifactError, ScoringExample
    path = root / "manifest.json"
    if not path.is_file():
        raise ScoreArtifactError("missing score manifest")
    raw = path.read_bytes()
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ScoreArtifactError("malformed score manifest") from error
    required_manifest = {
        "checkpoint_digest", "error_scale", "label_partitions", "schema_version",
        "score_count", "score_spec_digest", "shard_size", "shards", "state_count",
        "state_digest",
    }
    if type(manifest) is not dict or set(manifest) != required_manifest:
        raise ScoreArtifactError("closed score-manifest schema violation")
    if canonical_json_bytes(manifest) != raw or manifest["schema_version"] != SCHEMA_VERSION:
        raise ScoreArtifactError("score manifest is noncanonical or unsupported")
    if manifest["label_partitions"] != ["controller-train", "evaluation"]:
        raise ScoreArtifactError("score manifest label partitions are invalid")
    for field in ("checkpoint_digest", "score_spec_digest", "state_digest"):
        value = manifest[field]
        if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ScoreArtifactError(f"{field} must be lowercase SHA-256")
    if type(manifest["shards"]) is not list or type(manifest["shard_size"]) is not int or manifest["shard_size"] <= 0:
        raise ScoreArtifactError("score shard manifest fields are invalid")
    spec = ScoreSpec(error_scale=manifest["error_scale"])
    stream = ScoreShardStream(spec)
    for entry in manifest["shards"]:
        if type(entry) is not dict or set(entry) != {"name", "sha256", "state_count"}:
            raise ScoreArtifactError("closed score-shard entry schema violation")
        shard_path = root / entry["name"]
        if not shard_path.resolve().is_relative_to(root.resolve()):
            raise ScoreArtifactError("score shard path escapes the artifact root")
        data = shard_path.read_bytes() if shard_path.is_file() else b""
        if _digest(data) != entry["sha256"]:
            raise ScoreArtifactError(f"shard digest mismatch: {entry['name']}")
        lines = data.splitlines()
        if len(lines) != entry["state_count"]:
            raise ScoreArtifactError("partial score shard")
        for line in lines:
            partition_path = Path(entry["name"]).parts[-2]
            try:
                stream.add_record(json.loads(line), partition_path)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ScoreArtifactError("malformed state-score record") from error
    result = stream.finish(manifest["state_count"], manifest["score_count"], manifest["state_digest"])
    if result.metrics and ScoreSpec(error_scale=manifest["error_scale"]).identity != manifest["score_spec_digest"]:
        raise ScoreArtifactError("frozen error_scale does not match calibration")
    expected_metrics = canonical_json_bytes([aggregate_payload(item) for item in result.metrics])
    expected_ceiling = canonical_json_bytes(ceiling_payload(result.ceiling))
    if (root / "per_pair_metrics.json").read_bytes() != expected_metrics:
        raise ScoreArtifactError("per-pair metrics do not recompute from shards")
    if (root / "temporal_oracle_ceiling.json").read_bytes() != expected_ceiling:
        raise ScoreArtifactError("temporal oracle ceiling does not recompute from shards")
    unavailable = canonical_json_bytes([
        {"metric": metric, "reason": "required supervision is unavailable", "status": "unavailable"}
        for metric in ("ade", "fde", "final_state", "event", "penetration", "floating", "illegal_contact")
    ])
    if (root / "unavailable_metrics.json").read_bytes() != unavailable:
        raise ScoreArtifactError("explicit unavailable metric statuses are invalid")
    return ScoreArtifactReceipt(_digest(raw), result.state_count, result.score_count)


__all__ = ["SCHEMA_VERSION", "ScoreArtifactReceipt", "validate_score_artifacts", "write_score_artifacts"]
