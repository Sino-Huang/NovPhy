"""Resumable, checkpoint-bound artifacts for exhaustive temporal scores."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from world_model.model import Abstraction, PredictionPair
from world_model.training.grid_artifacts import JsonValue, canonical_json_bytes
from world_model.training.pair_grid import PairMetric, ScoreSpec, select_best_pair
from world_model.training.scoring_metrics import aggregate_labels, oracle_ceiling, state_digest
from world_model.training.scoring_payloads import aggregate_payload, ceiling_payload, state_payload


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
    existing_manifest = root / "manifest.json"
    if resume and existing_manifest.is_file():
        try:
            existing = json.loads(existing_manifest.read_bytes())
        except json.JSONDecodeError as error:
            raise ScoreArtifactError("stale score manifest") from error
        if existing.get("checkpoint_digest") != checkpoint_digest or existing.get("score_spec_digest") != result.score_spec.identity:
            raise ScoreArtifactError("resume checkpoint or score-spec binding mismatch")
    entries: list[dict[str, JsonValue]] = []
    for partition in Partition:
        states = tuple(item for item in result.scored_states if item.example.partition is partition)
        for offset in range(0, len(states), shard_size):
            batch = states[offset:offset + shard_size]
            data = b"".join(canonical_json_bytes(state_payload(item)) for item in batch)
            relative = Path("label_shards") / str(partition) / f"shard-{offset // shard_size:06d}.jsonl"
            path = root / relative
            if resume and path.is_file() and path.read_bytes() != data:
                raise ScoreArtifactError(f"stale or tampered shard: {relative}")
            if not path.is_file():
                _atomic_write(path, data)
            entries.append({"name": relative.as_posix(), "sha256": _digest(data), "state_count": len(batch)})
    metrics_data = canonical_json_bytes([aggregate_payload(item) for item in result.per_pair_metrics])
    ceiling_data = canonical_json_bytes(ceiling_payload(result.temporal_oracle_ceiling))
    unavailable_data = canonical_json_bytes([
        {"metric": item.metric, "reason": item.reason, "status": item.status}
        for item in result.unavailable_metrics
    ])
    _atomic_write(root / "per_pair_metrics.json", metrics_data)
    _atomic_write(root / "temporal_oracle_ceiling.json", ceiling_data)
    _atomic_write(root / "unavailable_metrics.json", unavailable_data)
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
        "state_digest": state_digest(tuple(item.example.state_id for item in result.scored_states)),
    }
    raw = canonical_json_bytes(manifest)
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
    scored = []
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
            record = json.loads(line)
            required = {
                "context_position", "frame_count", "metrics", "motion_regime", "partition",
                "primary_objective", "record_type", "schema_version", "selected_delta",
                "sensitivity", "state_id",
            }
            partition_path = Path(entry["name"]).parts[-2]
            if set(record) != required or record["record_type"] != "state_score" or record["schema_version"] != SCHEMA_VERSION:
                raise ScoreArtifactError("closed state-score schema violation")
            if record["partition"] != partition_path:
                raise ScoreArtifactError("state record is in the wrong partition shard")
            metrics = tuple(
                PairMetric(
                    pair=PredictionPair(metric["delta"], Abstraction.CONTINUOUS),
                    requested_delta=metric["requested_delta"], effective_delta=metric["effective_delta"],
                    duration_weight=metric["duration_weight"], latent_mse=metric["latent_mse"],
                    weighted_prediction_error=metric["weighted_error"], compute_cost=metric["compute_cost"],
                ) for metric in record["metrics"]
            )
            if tuple(metric.pair.delta for metric in metrics) != (1, 5, 15):
                raise ScoreArtifactError("state shard omitted or reordered a pair")
            label = select_best_pair(metrics, spec)
            expected_sensitivity = [
                {"lambda_cost": item.lambda_cost, "selected_delta": item.selected_pair.delta}
                for item in label.sensitivity
            ]
            if record["selected_delta"] != label.selected_pair.delta or record["primary_objective"] != label.primary_objective or record["sensitivity"] != expected_sensitivity:
                raise ScoreArtifactError("stored pair selection does not recompute")
            example = ScoringExample(record["state_id"], Partition(record["partition"]), MotionRegime(record["motion_regime"]), record["frame_count"], record["context_position"])
            scored.append((example, label))
    if len(scored) != manifest["state_count"] or len(scored) * 3 != manifest["score_count"]:
        raise ScoreArtifactError("manifest state or score count mismatch")
    if len({example.state_id for example, _label in scored}) != len(scored) or {example.partition for example, _label in scored} != set(Partition):
        raise ScoreArtifactError("state identities or partitions are incomplete")
    if state_digest(tuple(example.state_id for example, _label in scored)) != manifest["state_digest"]:
        raise ScoreArtifactError("manifest state digest mismatch")
    calibration = tuple(metric.weighted_prediction_error for example, label in scored if example.partition is Partition.CALIBRATION for metric in label.metrics)
    if ScoreSpec.from_calibration(calibration) != spec or spec.identity != manifest["score_spec_digest"]:
        raise ScoreArtifactError("frozen error_scale does not match calibration")
    rows = tuple((str(example.partition), example.motion_regime, example.state_id, label) for example, label in scored if example.partition is not Partition.CALIBRATION)
    expected_metrics = canonical_json_bytes([aggregate_payload(item) for item in aggregate_labels(rows)])
    expected_ceiling = canonical_json_bytes(ceiling_payload(oracle_ceiling(rows, spec.error_scale)))
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
    return ScoreArtifactReceipt(_digest(raw), len(scored), len(scored) * 3)


__all__ = ["SCHEMA_VERSION", "ScoreArtifactReceipt", "validate_score_artifacts", "write_score_artifacts"]
