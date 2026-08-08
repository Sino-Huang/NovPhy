from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPLICATES = 1000
MIN_STATES = 100
DELTAS = (1, 5, 15)


class FrontierError(ValueError):
    pass


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise FrontierError(f"{name} must be finite")
    return float(value)


def pareto_frontier(points: Iterable[dict[str, Any]]) -> list[int]:
    rows = list(points)
    for row in rows:
        _finite(row["weighted_prediction_error"], "weighted_prediction_error")
        _finite(row["compute_cost"], "compute_cost")
    result = []
    for i, candidate in enumerate(rows):
        dominated = any(
            j != i
            and other["weighted_prediction_error"] <= candidate["weighted_prediction_error"]
            and other["compute_cost"] <= candidate["compute_cost"]
            and (other["weighted_prediction_error"] < candidate["weighted_prediction_error"] or other["compute_cost"] < candidate["compute_cost"])
            for j, other in enumerate(rows)
        )
        if not dominated and candidate["delta"] not in result:
            result.append(candidate["delta"])
    return result


def _aggregate(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_delta: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in states:
        by_delta[int(row["delta"])].append(row)
    return [{"delta": d, "weighted_prediction_error": sum(r["weighted_prediction_error"] for r in rs) / len(rs), "compute_cost": sum(r["compute_cost"] for r in rs) / len(rs)} for d, rs in sorted(by_delta.items())]


def analyze_frontier(states: Iterable[dict[str, Any]], *, seed: int = 0, replicates: int = REPLICATES) -> dict[str, Any]:
    rows = []
    for row in states:
        if set(("delta", "weighted_prediction_error", "compute_cost")) - set(row):
            raise FrontierError("state metric fields are missing")
        item = dict(row)
        item["delta"] = int(item["delta"])
        item["weighted_prediction_error"] = _finite(item["weighted_prediction_error"], "weighted_prediction_error")
        item["compute_cost"] = _finite(item["compute_cost"], "compute_cost")
        rows.append(item)
    if len(rows) < MIN_STATES:
        raise FrontierError(f"insufficient states: need at least {MIN_STATES}")
    regimes = sorted({str(r.get("regime", "global")) for r in rows})
    grouped = {regime: [r for r in rows if str(r.get("regime", "global")) == regime] for regime in regimes}
    grouped["global"] = rows
    frontiers = {regime: pareto_frontier(_aggregate(group)) for regime, group in grouped.items()}
    rng = random.Random(seed)
    intersections = 0
    membership = {str(d): 0 for d in sorted({r["delta"] for r in rows})}
    for _ in range(replicates):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        sampled = {regime: [r for r in sample if regime == "global" or str(r.get("regime", "global")) == regime] for regime in regimes}
        sampled["global"] = sample
        fs = {regime: set(pareto_frontier(_aggregate(group))) for regime, group in sampled.items() if group}
        common = set.intersection(*fs.values()) if fs else set()
        intersections += bool(common)
        for delta in membership:
            membership[delta] += sum(int(delta) in values for values in fs.values()) / len(fs) if fs else 0
    interval = (max(0.0, (intersections / replicates) - 1.96 * math.sqrt((intersections / replicates) * (1 - intersections / replicates) / replicates)), min(1.0, (intersections / replicates) + 1.96 * math.sqrt((intersections / replicates) * (1 - intersections / replicates) / replicates)))
    oracle = {r["delta"] for r in _aggregate(rows)}
    verdict = "supported" if intersections / replicates < 0.05 and len(oracle) >= 2 else ("not_supported" if intersections and intersections / replicates >= 0.05 else "inconclusive")
    return {"schema_version": "temporal_pareto_frontier_v1", "frontiers": frontiers, "fixed_delta_intersection": sorted(set.intersection(*(set(v) for k, v in frontiers.items() if k != "global"))) if len(regimes) > 1 else [], "bootstrap": {"replicates": replicates, "seed": seed, "intersection_frequency": intersections / replicates, "intersection_95_interval": interval, "frontier_membership_frequency": {k: v / replicates for k, v in membership.items()}}, "oracle_labels": sorted(oracle), "verdict": verdict, "unavailable_metrics": [{"metric": x, "status": "unavailable", "reason": "required supervision is unavailable"} for x in ("alpha", "micro", "macro", "physical")], "state_count": len(rows)}


def source_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = ["DELTAS", "FrontierError", "analyze_frontier", "pareto_frontier", "source_digest"]
