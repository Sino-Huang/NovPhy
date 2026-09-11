"""Persistent per-method active fitting budgets and progress checkpoints."""
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import time

import torch

from scripts import issue_76_expansion as files
from scripts.run_issue_70_parser_repair import atomic_torch
from scripts.run_issue_76_compatibility import process_rss


class FitBudgetExceeded(RuntimeError):
    pass


class FitPaused(RuntimeError):
    pass


def log(message):
    print(f"[native-refit] {message}", flush=True)


class FitBudget:
    """One persistent allowance, shared across a method's declared sub-stages."""
    def __init__(self, root, key, seconds, device):
        self.path = Path(root) / "budgets" / (key + ".json")
        self.key, self.seconds, self.device = key, seconds, device
        self.value = files.read(self.path) if self.path.exists() else {
            "active_seconds": 0., "peak_cpu_rss_mib": 0., "peak_cuda_allocated_mib": 0.,
            "peak_cuda_reserved_mib": 0., "limit_seconds": seconds, "stopped": False}
        if self.value["limit_seconds"] != seconds or self.value["stopped"]:
            raise FitBudgetExceeded(f"{key}: prior resource stop or changed allowance")
        if self.value.get("running"):
            # A hard-killed job has an unknown unsaved optimizer/cost interval.
            # Do not reset its allowance or silently replay it from an old step.
            raise FitBudgetExceeded(f"{key}: unclean interrupted job; retain its checkpoint and audit before further fitting")
        self.initial, self.started = self.value["active_seconds"], time.monotonic()
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        self.pause_requested = False
        self.previous_interrupt = signal.signal(signal.SIGINT, lambda *_: setattr(self, "pause_requested", True))
        self.value.update(running=True, process_id=os.getpid())
        self.save()

    def save(self):
        self.value["active_seconds"] = self.initial + time.monotonic() - self.started
        files.write(self.path, self.value)

    def check(self):
        if self.pause_requested:
            raise FitPaused(f"{self.key}: paused at a completed-update boundary; rerun to resume")
        self.value["active_seconds"] = self.initial + time.monotonic() - self.started
        self.value["peak_cpu_rss_mib"] = max(self.value["peak_cpu_rss_mib"], process_rss(os.getpid()))
        if str(self.device).startswith("cuda"):
            self.value["peak_cuda_allocated_mib"] = max(self.value["peak_cuda_allocated_mib"], torch.cuda.max_memory_allocated(self.device) / 2**20)
            self.value["peak_cuda_reserved_mib"] = max(self.value["peak_cuda_reserved_mib"], torch.cuda.max_memory_reserved(self.device) / 2**20)
        if (self.value["active_seconds"] >= self.seconds or self.value["peak_cpu_rss_mib"] > 12288
                or self.value["peak_cuda_allocated_mib"] > 8192):
            self.value["stopped"] = True
            self.save()
            raise FitBudgetExceeded(f"{self.key}: active-time or memory allowance reached")

    def close(self):
        self.value["running"] = False
        self.save()
        signal.signal(signal.SIGINT, self.previous_interrupt)


@contextmanager
def fitting_budget(root, key, seconds, device):
    budget = FitBudget(root, key, seconds, device)
    try:
        yield budget
    finally:
        budget.close()


def fit_updates(path, model, *, plan_identity, updates, lr, make_loss, budget, device):
    """Checkpoint optimizer/update counter; skipped assignments are not replaced."""
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
        counter, applied, skipped = previous["updates_completed"], previous["updates_applied"], previous["updates_skipped"]
        if previous["complete"]:
            log(f"retaining completed {path.stem}: {counter}/{updates} updates")
            return previous
    last_loss, last_log, started = None, time.monotonic(), time.monotonic()
    initial_counter = counter
    failure = None
    def snapshot():
        value = {"schema": "issue_76_native_fit_progress_v1", "plan_identity": plan_identity,
                 "updates_requested": updates, "updates_completed": counter, "updates_applied": applied,
                 "updates_skipped": skipped, "complete": counter == updates and failure is None and not budget.value["stopped"],
                 "failure": failure,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(), "last_loss": last_loss}
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
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
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
                log(f"{path.stem} updates={counter}/{updates} applied={applied} skipped={skipped} loss={last_loss} budget={budget.value['active_seconds']:.1f}/{budget.seconds}s ETA={rate * (updates-counter):.1f}s CUDA={budget.value['peak_cuda_allocated_mib']:.1f}MiB")
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
