from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from world_model.training.grid_artifacts import canonical_json_bytes
from world_model.training.scoring_artifacts import validate_score_artifacts

REPLICATES = 1000
MIN_STATES = 100
DELTAS = (1, 5, 15)
FRONTIER_INPUT_SCHEMA = "temporal_frontier_input_v1"
UNAVAILABLE_SCOPE = "alpha unavailable (continuous-only scope); physical metrics unavailable (required supervision is unavailable)"


class FrontierError(ValueError):
    pass


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise FrontierError(f"{name} must be finite")
    return float(value)


def _delta(value: Any, field: str) -> int:
    if type(value) is not int or value not in DELTAS:
        raise FrontierError(f"{field} must be an approved delta")
    return value


def pareto_frontier(points: Iterable[dict[str, Any]]) -> list[int]:
    rows = list(points)
    for row in rows:
        _finite(row["weighted_prediction_error"], "weighted_prediction_error")
        _finite(row["compute_cost"], "compute_cost")
    result = []
    for index, candidate in enumerate(rows):
        dominated = any(
            other_index != index
            and other["weighted_prediction_error"] <= candidate["weighted_prediction_error"]
            and other["compute_cost"] <= candidate["compute_cost"]
            and (other["weighted_prediction_error"] < candidate["weighted_prediction_error"] or other["compute_cost"] < candidate["compute_cost"])
            for other_index, other in enumerate(rows)
        )
        if not dominated and candidate["delta"] not in result:
            result.append(candidate["delta"])
    return result


def _aggregate(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_delta: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in states:
        by_delta[row["delta"]].append(row)
    return [
        {
            "delta": delta,
            "weighted_prediction_error": sum(row["weighted_prediction_error"] for row in group) / len(group),
            "compute_cost": sum(row["compute_cost"] for row in group) / len(group),
        }
        for delta, group in sorted(by_delta.items())
    ]


def _recomputed_selected_delta(group: list[dict[str, Any]]) -> int:
    scales = {_finite(row.get("error_scale"), "error_scale") for row in group}
    if len(scales) != 1 or next(iter(scales)) <= 0.0:
        raise FrontierError("state rows must share a positive error_scale")
    scale = next(iter(scales))
    objectives = [
        (row["weighted_prediction_error"] / scale + row["compute_cost"], row)
        for row in group
    ]
    minimum = min(objective for objective, _row in objectives)
    tied = [
        row for objective, row in objectives
        if math.isclose(objective, minimum, rel_tol=1e-6, abs_tol=1e-12)
    ]
    return min(tied, key=lambda row: (row["weighted_prediction_error"], row["delta"]))["delta"]


def _selected_delta(group: list[dict[str, Any]]) -> int:
    has_selected = ["selected_delta" in row for row in group]
    if any(has_selected) and not all(has_selected):
        raise FrontierError("state rows must consistently declare selected_delta")
    if not any(has_selected):
        return _recomputed_selected_delta(group)
    selections = {_delta(row["selected_delta"], "selected_delta") for row in group}
    if len(selections) != 1 or next(iter(selections)) not in {row["delta"] for row in group}:
        raise FrontierError("state rows have inconsistent selected_delta")
    if _recomputed_selected_delta(group) != next(iter(selections)):
        raise FrontierError("selected_delta does not match the canonical primary objective")
    return next(iter(selections))


def _state_groups(states: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, int]] = set()
    for row in states:
        required = {"state_id", "regime", "delta", "weighted_prediction_error", "compute_cost", "error_scale"}
        if not isinstance(row, dict) or set(required) - set(row):
            raise FrontierError("state metric fields are missing")
        state_id = row["state_id"]
        regime = row["regime"]
        if type(state_id) is not str or not state_id or type(regime) is not str or not regime:
            raise FrontierError("state_id and regime must be nonempty strings")
        delta = _delta(row["delta"], "delta")
        pair = (state_id, delta)
        if pair in seen:
            raise FrontierError("duplicate state_id/delta metric")
        seen.add(pair)
        error = _finite(row["weighted_prediction_error"], "weighted_prediction_error")
        cost = _finite(row["compute_cost"], "compute_cost")
        if error < 0.0 or cost < 0.0:
            raise FrontierError("state metrics must be nonnegative")
        item = dict(row)
        item["delta"] = delta
        item["weighted_prediction_error"] = error
        item["compute_cost"] = cost
        group = groups.setdefault(state_id, [])
        if group and group[0]["regime"] != regime:
            raise FrontierError("state_id must belong to one regime")
        group.append(item)
    if not groups:
        raise FrontierError("state metrics are empty")
    for group in groups.values():
        if {row["delta"] for row in group} != set(DELTAS):
            raise FrontierError("each state must contain one metric for every approved delta")
        _selected_delta(group)
    return groups


def analyze_frontier(states: Iterable[dict[str, Any]], *, seed: int = 0, replicates: int = REPLICATES) -> dict[str, Any]:
    if type(seed) is not int or type(replicates) is not int or replicates <= 0:
        raise FrontierError("seed must be an integer and replicates must be positive")
    groups = _state_groups(states)
    rows = [row for group in groups.values() for row in group]
    regimes = sorted({group[0]["regime"] for group in groups.values()})
    for regime in regimes:
        if sum(group[0]["regime"] == regime for group in groups.values()) < MIN_STATES:
            raise FrontierError(f"insufficient states in regime {regime}: need at least {MIN_STATES}")
    grouped = {regime: [row for group in groups.values() if group[0]["regime"] == regime for row in group] for regime in regimes}
    grouped["global"] = rows
    frontiers = {regime: pareto_frontier(_aggregate(group)) for regime, group in grouped.items()}
    rng = random.Random(seed)
    intersections = 0
    membership = {str(delta): 0 for delta in DELTAS}
    state_groups = list(groups.values())
    for _ in range(replicates):
        sample = [row for _index in state_groups for row in state_groups[rng.randrange(len(state_groups))]]
        sampled = {regime: [row for row in sample if row["regime"] == regime] for regime in regimes}
        frontiers_by_sample = {regime: set(pareto_frontier(_aggregate(group))) for regime, group in sampled.items() if group}
        common = set.intersection(*frontiers_by_sample.values()) if frontiers_by_sample else set()
        intersections += bool(common)
        for delta in membership:
            membership[delta] += sum(int(delta) in values for values in frontiers_by_sample.values()) / len(frontiers_by_sample)
    frequency = intersections / replicates
    interval = (
        max(0.0, frequency - 1.96 * math.sqrt(frequency * (1 - frequency) / replicates)),
        min(1.0, frequency + 1.96 * math.sqrt(frequency * (1 - frequency) / replicates)),
    )
    oracle_labels = [_selected_delta(group) for group in groups.values()]
    verdict = "supported" if frequency < 0.05 and len(set(oracle_labels)) >= 2 else ("not_supported" if frequency >= 0.05 else "inconclusive")
    unavailable = [
        {"metric": "alpha", "status": "unavailable", "reason": "required supervision is unavailable"},
        {"metric": "micro", "status": "unavailable", "reason": "symbolic_supervision_unavailable"},
        {"metric": "macro", "status": "unavailable", "reason": "symbolic_supervision_unavailable"},
        {"metric": "physical", "status": "unavailable", "reason": "required supervision is unavailable"},
    ]
    return {"schema_version": "temporal_pareto_frontier_v1", "frontiers": frontiers, "fixed_delta_intersection": sorted(set.intersection(*(set(values) for name, values in frontiers.items() if name != "global"))) if len(regimes) > 1 else [], "bootstrap": {"replicates": replicates, "seed": seed, "intersection_frequency": frequency, "intersection_95_interval": interval, "frontier_membership_frequency": {key: value / replicates for key, value in membership.items()}}, "oracle_labels": oracle_labels, "oracle_definition": "canonical_per_state_primary_selected_delta", "scope": UNAVAILABLE_SCOPE, "verdict": verdict, "unavailable_metrics": unavailable, "state_count": len(groups)}


def canonical_frontier_rows(source: bytes, source_path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(source)
    except json.JSONDecodeError as error:
        raise FrontierError("frontier input is not valid JSON") from error
    required = {"partition", "schema_version", "score_artifact_root"}
    if type(payload) is not dict or set(payload) != required or canonical_json_bytes(payload) != source:
        raise FrontierError("frontier input must use the closed canonical schema")
    if payload["schema_version"] != FRONTIER_INPUT_SCHEMA or payload["partition"] != "evaluation":
        raise FrontierError("frontier input schema or partition is unsupported")
    if type(payload["score_artifact_root"]) is not str or not payload["score_artifact_root"]:
        raise FrontierError("score_artifact_root must be a nonempty path")
    artifact_root = Path(payload["score_artifact_root"])
    if not artifact_root.is_absolute():
        artifact_root = source_path.parent / artifact_root
    artifact_root = artifact_root.resolve()
    try:
        validate_score_artifacts(artifact_root)
        manifest = json.loads((artifact_root / "manifest.json").read_bytes())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise FrontierError("canonical score artifacts are invalid") from error
    rows = []
    for shard in manifest["shards"]:
        if Path(shard["name"]).parts[-2] != "evaluation":
            continue
        for line in (artifact_root / shard["name"]).read_bytes().splitlines():
            record = json.loads(line)
            for metric in record["metrics"]:
                rows.append({"state_id": record["state_id"], "regime": record["motion_regime"], "delta": metric["delta"], "weighted_prediction_error": metric["weighted_error"], "compute_cost": metric["compute_cost"], "selected_delta": record["selected_delta"], "error_scale": manifest["error_scale"]})
    return rows


__all__ = ["DELTAS", "FRONTIER_INPUT_SCHEMA", "FrontierError", "MIN_STATES", "UNAVAILABLE_SCOPE", "analyze_frontier", "canonical_frontier_rows", "pareto_frontier"]
