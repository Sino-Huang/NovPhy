"""Deployment-available task objective and ranking diagnostics for issue #70.

Only a carrier and frozen vocabulary enter the objective. Realized outcomes are
used by the diagnostic reporter, never by the evaluator or planner.
"""

from dataclasses import dataclass
import math
import time

import numpy as np
import torch

from world_model.model import Abstraction, PredictionPair
from world_model.planning.gameplay import CandidateEvaluation, SlingshotAction


@dataclass(frozen=True)
class TaskObjective:
    pig_slots: tuple[int, ...]
    block_slots: tuple[int, ...]

    @classmethod
    def from_vocabulary(cls, vocabulary):
        pigs = tuple(i for i, name in enumerate(vocabulary) if name.startswith("pig:"))
        blocks = tuple(i for i, name in enumerate(vocabulary) if name.startswith("block:"))
        if not pigs:
            raise ValueError("frozen parser vocabulary has no pig slots")
        return cls(pigs, blocks)

    def counts(self, carrier):
        if not bool(torch.isfinite(carrier).all()):
            raise ValueError("nonfinite task carrier")
        # Bound probabilities for utility evaluation; never clamp the recursive
        # carrier itself. Range excess is reported independently by the evaluator.
        def count(slots):
            return float(carrier[[2 + 13 * i for i in slots]].clamp(0, 1).sum())
        return count(self.pig_slots), count(self.block_slots)

    def __call__(self, carrier):
        pigs, blocks = self.counts(carrier)
        return 1000.0 * pigs + blocks


def ranking_diagnostic(predicted, realized, *, tolerance=1e-8):
    """All-state deterministic top-k, with tie ambiguity separately exposed."""
    if len(predicted) != len(realized) or not realized:
        raise ValueError("candidate inventories differ")
    valid = [v < 1e9 for v in realized]
    best = min(realized)
    span = max(realized) - best
    best_indices = [i for i, v in enumerate(realized) if v == best]
    failed = any(v is None or not math.isfinite(v) for v in predicted)
    if failed:
        return {"failed": True, "regret": 1.0, "selected": None,
                "top1": False, "top3": False, "best_rank": None,
                "best_rank_interval": None, "predicted_spread": None,
                "predicted_all_tied": False, "outcome_discriminating": span > 0,
                "realized_failures": sum(not v for v in valid)}
    order = sorted(range(len(predicted)), key=lambda i: (predicted[i], i))
    rank = min(order.index(i) + 1 for i in best_indices)
    best_prediction = min(predicted[i] for i in best_indices)
    low = 1 + sum(v < best_prediction - tolerance for v in predicted)
    high = low + sum(i not in best_indices and abs(v-best_prediction) <= tolerance
                     for i,v in enumerate(predicted))
    selected = order[0]
    return {"failed": False, "regret": 0.0 if span == 0 else (realized[selected]-best)/span,
            "selected": selected, "top1": rank <= 1, "top3": rank <= 3,
            "best_rank": rank, "best_rank_interval": [low, high],
            "predicted_spread": max(predicted)-min(predicted),
            "predicted_all_tied": max(predicted)-min(predicted) <= tolerance,
            "outcome_discriminating": span > 0,
            "realized_failures": sum(not v for v in valid)}


def summarize_rankings(rows):
    if not rows:
        raise ValueError("empty ranking inventory")
    def summary(values):
        return {"states": len(values),
                "mean_regret": float(np.mean([r["regret"] for r in values])) if values else None,
                "top1_fraction": float(np.mean([r["top1"] for r in values])) if values else None,
                "top3_fraction": float(np.mean([r["top3"] for r in values])) if values else None}
    return {"all_states": summary(rows),
            "discriminating_states": summary([r for r in rows if r["outcome_discriminating"]]),
            "prediction_failures": sum(r["failed"] for r in rows),
            "predicted_all_tied_states": sum(r["predicted_all_tied"] for r in rows),
            "realized_failures": sum(r["realized_failures"] for r in rows)}


def objective_gate(summary):
    subset = summary["discriminating_states"]
    return (subset["states"] >= 20 and subset["mean_regret"] <= 0.10
            and subset["top3_fraction"] >= 0.80 and summary["prediction_failures"] == 0)


class TaskCandidateEvaluator:
    """Shared fixed-horizon scorer for the broad grid and existing CEM."""

    def __init__(self, models, objective, bounds, *, steps=15, penalty=0.0, progress=None):
        if not models or steps <= 0 or penalty < 0:
            raise ValueError("invalid task evaluator")
        self.models = tuple(model.eval() for model in models)
        self.objective = objective
        self.bounds = bounds
        self.steps = steps
        self.penalty = penalty
        self.progress = progress
        self.last_member_endpoints = ()
        self.records = []

    def evaluate(self, observation, actions):
        started=time.monotonic()
        self._attempt_model_calls=0
        try:
            return self._evaluate(observation,actions)
        except (ValueError,RuntimeError) as error:
            self.records.append({"action":None if not actions else {
                "drag_x":actions[0].drag_x,"drag_y":actions[0].drag_y,"tap_time_ms":actions[0].tap_time_ms},
                "failure":f"{type(error).__name__}: {error}","cost":None,
                "model_evaluations":self._attempt_model_calls,"wall_seconds":time.monotonic()-started})
            raise

    def _evaluate(self, observation, actions):
        if len(actions) != 1 or not self.bounds.contains(actions[0]):
            raise ValueError("task pilot requires one legal action per replan")
        action = actions[0]
        tensor = torch.tensor((action.drag_x/observation.frame_height,
                               action.drag_y/observation.frame_height,
                               self.bounds.release_time_ms/1000.0,
                               action.tap_time_ms/1000.0, 1.0))
        costs, endpoints, bounds = [], [], []
        started = time.monotonic()
        with torch.inference_mode():
            for ordinal, model in enumerate(self.models, 1):
                if self.progress:
                    self.progress(f"candidate={len(self.records)+1} member={ordinal}/{len(self.models)}")
                device = next(model.parameters()).device
                current = observation.carrier.to(device)
                maximum_excess = 0.0
                for _ in range(self.steps):
                    self._attempt_model_calls+=1
                    current = model.carrier(current[None], tensor.to(device)[None],
                                            PredictionPair(15, Abstraction.CONTINUOUS))[0]
                    if not bool(torch.isfinite(current).all()):
                        raise ValueError("nonfinite recursive task prediction")
                    maximum_excess = max(maximum_excess, float(torch.relu(current.abs()-2.0).max()))
                endpoints.append(current.cpu())
                costs.append(self.objective(current))
                bounds.append(maximum_excess)
        std = float(np.std(costs))
        total = float(np.mean(costs)) + self.penalty * std
        self.last_member_endpoints = tuple(endpoints)
        self.records.append({"action": {"drag_x": action.drag_x, "drag_y": action.drag_y,
                                       "tap_time_ms": action.tap_time_ms},
                             "member_costs": costs, "disagreement": std, "cost": total,
                             "maximum_carrier_bound_excess": max(bounds),
                             "model_evaluations": self.steps*len(self.models),
                             "wall_seconds": time.monotonic()-started})
        return CandidateEvaluation(actions=actions, total_cost=total,
            predicted_carriers=(torch.stack(endpoints).mean(0),),
            model_rollout_count=self.steps*len(self.models),
            model_compute=float(self.steps*len(self.models)))
