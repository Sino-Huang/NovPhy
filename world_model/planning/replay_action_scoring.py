"""Fixed-pair replay predictions from one agent-only decision carrier."""
import torch

from world_model.planning.task_objective import TaskObjective
from world_model.training.fixed_development import recursive_curves
from world_model.training.native_history_data import VOCABULARY, action_context


@torch.no_grad()
def score_actions(model, initial, actions, pair, *, endpoint=11250):
    """No capture/target inputs, history mutation, controller or truth resets.

    The endpoint is elapsed from the genuine decision observation, not from
    capture start or settlement. Counts are bounded only for task evaluation;
    the recursive carrier is never clamped or repaired.
    """
    if not actions:
        raise ValueError("the assigned candidate inventory must not be empty")
    contexts = torch.stack([action_context(action) for action in actions]).to(initial)
    batch = initial[None].expand(len(actions), -1).clone()
    curve = recursive_curves(model, batch, contexts, pair, times=(endpoint,))[endpoint]
    objective = TaskObjective.from_vocabulary(VOCABULARY)
    rows = []
    for ordinal, (predicted, failed) in enumerate(zip(curve["predicted"], curve["failed"], strict=True)):
        bad = bool(failed)
        rows.append({"ordinal": ordinal, "prediction_failed": bad,
                     "predicted_cost": None if bad else objective(predicted),
                     "predicted_counts": None if bad else objective.counts(predicted)})
    return {"candidates": rows, "endpoint_native_steps_from_decision": endpoint,
            "transitions_per_candidate": curve["transitions_per_member"],
            "candidate_transitions": len(actions) * curve["transitions_per_member"],
            "controller_calls": 0, "future_observations_used": False,
            "engine_outcomes_used": False}
