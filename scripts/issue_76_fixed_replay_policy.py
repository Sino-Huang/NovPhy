"""Fixed one-shot policy for the existing live capture callback interface."""
import math


class FixedReplayPolicy:
    """Record causal callback timing; never score an image or choose another action."""
    def __init__(self, action):
        self.action = dict(action)
        self.reset()

    def reset(self):
        self.observations = 0
        self.first_observation_time = None
        self.last_observation_time = None
        self.decision_time = None
        self.executed_time = None

    def observe(self, png, timestamp):
        # The live runner persists the actual PNG separately. Its content must
        # not influence a prospectively assigned replay action.
        if not math.isfinite(timestamp) or (self.last_observation_time is not None and timestamp < self.last_observation_time):
            raise ValueError("replay observation clock moved backward or became nonfinite")
        if self.first_observation_time is None:
            self.first_observation_time = timestamp
        self.last_observation_time = timestamp
        self.observations += 1

    def choose(self):
        if not self.observations or self.decision_time is not None:
            raise ValueError("one-shot replay requires one pre-decision observation and one decision")
        self.decision_time = self.last_observation_time
        return {"action": dict(self.action), "selection": "prospectively_assigned_training_replay",
                "trained_model_used": False, "candidates_scored": 0,
                "observations_influence_action": False}

    def executed(self, action, timestamp):
        if (self.decision_time is None or self.executed_time is not None or action != self.action
                or timestamp != self.last_observation_time or timestamp < self.decision_time):
            raise ValueError("accepted action must match the single assignment and its observed prelaunch time")
        self.executed_time = timestamp

    def evidence(self):
        return {"kind": "fixed_training_action_replay", "observations": self.observations,
                "decisions": int(self.decision_time is not None),
                "executed_actions": int(self.executed_time is not None),
                "first_observation_time": self.first_observation_time,
                "last_observation_time": self.last_observation_time,
                "decision_time": self.decision_time, "executed_action_time": self.executed_time,
                "trained_model_used": False, "observations_influence_action": False,
                "perception_seconds": 0., "full_flops_measured": False,
                "matched_compute_established": False}
