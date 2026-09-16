"""Numerically scaled predictor fitting for the balanced-native v2 recovery."""

from pathlib import Path
import time

import torch

from scripts.issue_76_fit_budget import FitBudgetExceeded, FitPaused, log
from scripts.run_issue_70_parser_repair import atomic_torch


def backward_clipped(loss, parameters, *, backward_scale):
    parameters = list(parameters)
    (loss * backward_scale).backward()
    scaled_norm = torch.nn.utils.clip_grad_norm_(
        parameters, float("inf"), error_if_nonfinite=True)
    coefficient = (backward_scale / (
        scaled_norm + backward_scale * 1e-6)).clamp(max=1.0)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(coefficient / backward_scale)


def fit_updates(path, model, *, plan_identity, updates, lr, make_loss, budget,
                device, backward_scale):
    """Fit with unit clipping through a downscaled, then rescaled, backward pass."""
    path = Path(path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.0001)
    counter, applied, skipped = 0, 0, 0
    if path.exists():
        previous = torch.load(path, map_location=device, weights_only=False)
        if previous["plan_identity"] != plan_identity or previous["updates_requested"] != updates:
            raise ValueError("fit checkpoint plan or update schedule differs")
        if previous.get("failure"):
            raise ValueError("failed optimizer job must be audited, not silently retried")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        counter = previous["updates_completed"]
        applied = previous["updates_applied"]
        skipped = previous["updates_skipped"]
        if previous["complete"]:
            log(f"retaining completed {path.stem}: {counter}/{updates} updates")
            return previous
    last_loss, last_log, started = None, time.monotonic(), time.monotonic()
    initial_counter = counter
    failure = None

    def snapshot():
        value = {
            "schema": "issue_76_scaled_fit_progress_v1",
            "plan_identity": plan_identity,
            "updates_requested": updates,
            "updates_completed": counter,
            "updates_applied": applied,
            "updates_skipped": skipped,
            "complete": counter == updates and failure is None and not budget.value["stopped"],
            "failure": failure,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "last_loss": last_loss,
            "backward_scale": backward_scale,
        }
        atomic_torch(path, value)
        budget.save()
        return value

    model.train()
    try:
        while counter < updates:
            budget.check()
            optimizer.zero_grad(set_to_none=True)
            loss = make_loss(counter)
            if loss is None:
                skipped += 1
            else:
                if not bool(torch.isfinite(loss)):
                    raise ValueError(f"nonfinite fit loss in {path.stem} update {counter}")
                backward_clipped(loss, model.parameters(), backward_scale=backward_scale)
                optimizer.step()
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize(device)
                last_loss = float(loss.detach())
                applied += 1
            counter += 1
            budget.check()
            now = time.monotonic()
            if now - last_log >= 5 or counter == updates:
                rate = (now - started) / max(counter - initial_counter, 1)
                log(f"{path.stem} updates={counter}/{updates} applied={applied} "
                    f"skipped={skipped} loss={last_loss} "
                    f"budget={budget.value['active_seconds']:.1f}/{budget.seconds}s "
                    f"ETA={rate * (updates-counter):.1f}s "
                    f"CUDA={budget.value['peak_cuda_allocated_mib']:.1f}MiB")
                budget.save()
                last_log = now
            if counter % 100 == 0:
                snapshot()
    except (FitPaused, FitBudgetExceeded):
        raise
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        result = snapshot()
    return result
