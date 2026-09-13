"""Agent-only observation memory and action scoring for the native #76 models.

This module receives RGB, observation timestamps and executed actions. It does
not load capture manifests, engine entities, future observations or outcomes.
"""
from dataclasses import asdict
from io import BytesIO
import time

import numpy as np
from PIL import Image
import torch

from world_model.planning.gameplay import SlingshotAction, SlingshotActionBounds
from world_model.planning.task_objective import TaskObjective
from world_model.training.native_history_data import (
    VOCABULARY, VISUAL_DIM, action_context, visual_carriers,
)
from world_model.training.native_history_fit import rollout, ENDPOINT


def agent_image_tensor(png):
    """The exact RGB resize used for native model fitting."""
    with Image.open(BytesIO(png)) as source:
        rgb = np.asarray(source.convert("RGB").resize((96, 64), Image.Resampling.BILINEAR),
                         dtype=np.uint8).copy()
    return torch.from_numpy(rgb).permute(2, 0, 1)


class NativeObservationHistory:
    """Streaming counterpart of the frozen batch visual/history encoders."""
    def __init__(self, parser, history):
        self.parser, self.history = parser.eval(), history.eval()
        self.device = next(parser.parameters()).device
        self.reset()

    def reset(self):
        """Call only for a new gameplay episode, never between its shots."""
        self.state = None
        self.previous_output = None
        self.previous_timestamp = None
        self.current = None
        self.observations = 0
        self.executed_actions = 0

    @torch.no_grad()
    def observe(self, image, timestamp):
        if image.shape != (3, 64, 96) or image.dtype != torch.uint8:
            raise ValueError("native observation must be the shared uint8 RGB image")
        if not np.isfinite(timestamp):
            raise ValueError("agent observation timestamp must be finite")
        output = self.parser(image[None].to(self.device))
        if self.previous_output is None:
            pair, times = output, [timestamp]
        else:
            pair = {k: torch.cat((self.previous_output[k], output[k])) for k in output}
            times = [self.previous_timestamp, timestamp]
        visual = visual_carriers(pair, torch.tensor(times, dtype=torch.float64, device=self.device))[-1]
        sequence, self.state = self.history(
            carriers=visual[None, None], actions=visual.new_zeros(1, 1, 5),
            timestamps=torch.tensor([[timestamp]], dtype=torch.float64, device=self.device),
            observation_mask=torch.ones(1, 1, dtype=torch.bool, device=self.device),
            action_mask=torch.zeros(1, 1, dtype=torch.bool, device=self.device), state=self.state)
        self.current = torch.cat((visual, sequence[0, 0]))
        self.previous_output, self.previous_timestamp = output, timestamp
        self.observations += 1
        return self.current.clone()

    @torch.no_grad()
    def record_executed_action(self, action, timestamp):
        """One event after transport acceptance; planning never calls this."""
        if self.current is None:
            raise ValueError("observe before executing an action")
        context = action_context(action).to(self.device)
        _, self.state = self.history(
            carriers=self.current.new_zeros(1, 1, VISUAL_DIM), actions=context[None, None],
            timestamps=torch.tensor([[timestamp]], dtype=torch.float64, device=self.device),
            observation_mask=torch.zeros(1, 1, dtype=torch.bool, device=self.device),
            action_mask=torch.ones(1, 1, dtype=torch.bool, device=self.device), state=self.state)
        self.executed_actions += 1


@torch.no_grad()
def score_native_actions(model, controller, carrier, actions, bounds, *, fixed_pair=None,
                         endpoint=ENDPOINT, maximum_linear_macs=None, maximum_seconds=None):
    """Score a fixed legal candidate inventory without touching real memory.

    Limits account for this scorer's transition/controller linear MACs and wall
    time; they do not pretend to cover common perception or every hardware FLOP.
    The live runner must add observation, search and transport costs separately.
    A failed candidate or exhausted allowance fails the decision, not a fallback
    to an unscored action. Ties use the original candidate order for both arms.
    """
    if not actions or any(not bounds.contains(a) for a in actions):
        raise ValueError("all native planning candidates must be legal")
    objective = TaskObjective.from_vocabulary(VOCABULARY)
    initial = carrier.detach().clone().to(next(model.parameters()).device)
    records, total_macs = [], 0
    if initial.is_cuda:
        torch.cuda.synchronize(initial.device)
    started = time.monotonic()
    for ordinal, action in enumerate(actions):
        context = action_context({**asdict(action), "release_time_ms": bounds.release_time_ms}).to(initial)
        predicted, work = rollout(model, controller, initial, context,
                                  endpoint=endpoint, fixed_pair=fixed_pair)
        if initial.is_cuda:
            torch.cuda.synchronize(initial.device)
        total_macs += sum(w["linear_macs"] for w in work)
        elapsed = time.monotonic() - started
        if maximum_linear_macs is not None and total_macs > maximum_linear_macs:
            raise ValueError("native decision exceeded its declared linear-MAC allowance")
        if maximum_seconds is not None and elapsed > maximum_seconds:
            raise ValueError("native decision exceeded its declared wall allowance")
        records.append({"ordinal": ordinal, "action": asdict(action),
                        "predicted_cost": objective(predicted), "endpoint": predicted.cpu().tolist(),
                        "endpoint_native_steps": endpoint, "work": work})
    selected = min(range(len(records)), key=lambda i: (records[i]["predicted_cost"], i))
    return {"selected_ordinal": selected, "action": asdict(actions[selected]), "candidates": records,
            "linear_macs": total_macs, "wall_seconds": time.monotonic() - started,
            "future_observations_used": False, "engine_outcomes_used": False}
