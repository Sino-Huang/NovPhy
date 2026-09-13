"""Load completed native checkpoints into the agent-only live policy interface."""
from pathlib import Path
import time

import torch

from scripts import issue_76_live_episode as live
from scripts import run_issue_76_native_refit as fit
from world_model.planning.native_gameplay import NativeObservationHistory, score_native_actions
from world_model.training.action_ranking_probe import broad_action_candidates


class CostedHistoryPolicy(live.HistoryPolicy):
    def __init__(self, history, selector):
        super().__init__(history, selector)
        self.charged_common_seconds = 0.

    def reset(self):
        super().reset()
        self.charged_common_seconds = 0.


def load_policy(root, plan, seed, pure, bounds, *, maximum_seconds,
                maximum_linear_macs, device="cuda", fixed_pair=None):
    """Inference only; loading never fits, substitutes or repairs a checkpoint.

    The wall allowance includes all RGB/history work since the preceding decision
    plus candidate inference. It is an allowance, not a claim of equal realized
    FLOPs. Engine/transport, observation persistence and loading remain separately
    measured by the episode runner or this load receipt.
    """
    root = Path(root)
    started = time.monotonic()
    if seed not in plan["seeds"] or plan["synthetic"]:
        raise ValueError("live learned policy requires a declared non-synthetic paired seed")
    common_path = fit.common_path(root, seed)
    fit.require_finished_budget(root, f"common-{seed}")
    common = torch.load(common_path, map_location=device, weights_only=False)
    if (common["plan_identity"] != plan["identity"] or common["seed"] != seed
            or common["synthetic"] or common["auxiliary_deployed"]):
        raise ValueError("live common checkpoint differs from the trained deployment contract")
    parser = fit.NativeVisualParser().to(device).eval().requires_grad_(False)
    parser.load_state_dict(common["parser"])
    history = fit.ObservedHistoryEncoder(carrier_dim=fit.VISUAL_DIM).to(device).eval().requires_grad_(False)
    history.load_state_dict(common["history"])
    model = fit.load_predictor(root, plan, seed, pure, device)
    name = "pure" if pure else "hybrid"
    controller, controller_path = None, None
    if fixed_pair is None:
        fit.require_finished_budget(root, f"controller-{name}-{seed}")
        controller_path = fit.model_path(root, seed, pure, "controller")
        saved = torch.load(controller_path, map_location=device, weights_only=False)
        if saved["plan_identity"] != plan["identity"] or not saved["complete"]:
            raise ValueError("live controller checkpoint is incomplete or belongs to a different plan")
        controller = fit.NativeHistoryController(pure).to(device).eval().requires_grad_(False)
        controller.load_state_dict(saved["model"])
    elif fixed_pair not in model.pairs:
        raise ValueError("fixed pair is outside this independently trained arm's capabilities")
    actions = tuple(candidate.action for candidate in broad_action_candidates("native-live-decision", bounds))
    stream = NativeObservationHistory(parser, history)
    def choose(carrier):
        common_seconds = policy.perception_seconds - policy.charged_common_seconds
        remaining = maximum_seconds - common_seconds
        if remaining <= 0:
            raise ValueError("native decision allowance exhausted by common RGB/history work")
        result = score_native_actions(model, controller, carrier, actions, bounds,
            fixed_pair=fixed_pair, maximum_seconds=remaining, maximum_linear_macs=maximum_linear_macs)
        result["action"]["release_time_ms"] = bounds.release_time_ms
        result.update(common_rgb_history_seconds=common_seconds,
            decision_model_seconds=common_seconds + result["wall_seconds"],
            maximum_model_seconds=maximum_seconds, maximum_linear_macs=maximum_linear_macs,
            trained_model_used=True, full_flops_measured=False, matched_compute_established=False)
        policy.charged_common_seconds = policy.perception_seconds
        return result

    policy = CostedHistoryPolicy(stream, choose)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    return policy, dict(plan_identity=plan["identity"], seed=seed, independently_trained_pure=pure,
        common_checkpoint=str(common_path), predictor_checkpoint=str(fit.model_path(root, seed, pure, "predictor")),
        controller_checkpoint=str(controller_path) if controller_path else None,
        fixed_pair=None if fixed_pair is None else dict(mode=str(fixed_pair.abstraction), horizon_native_steps=fixed_pair.delta),
        candidate_count=len(actions), endpoint_native_steps=fit.ENDPOINT,
        load_seconds=time.monotonic() - started, device=str(device), new_training_updates=0,
        fresh_access_authorized=False)
