"""#123 main training: a run stopped by SIGTERM and resumed equals the uninterrupted run bit for bit (CPU).

GPU-window amendment A1 of scripts/run_hjepa_comparison.py: on SIGTERM the running arm-seed writes resume.pt after
its current optimizer step and the queue exits 75; the relaunch resumes there. Run on CPU
(CUDA_VISIBLE_DEVICES="") with the HJ-car-m arm, batch 4, on synthetic carriers (the real fit store is not read).
"""
from __future__ import annotations

import os
import signal

import numpy as np
import pytest
import torch

from scripts import run_hjepa_comparison as hc
from world_model.hjepa import novphy as hn

ARM, SEED, LAM, BATCH, STEPS = "HJ-car-m", 20260908, 0.18, 4, 8


class SyntheticFit:
    """FitData stand-in: fixed seeded carriers and launches for 6 shots."""

    def __init__(self):
        generator = torch.Generator().manual_seed(7)
        self.ends = np.array([285, 200, 150, 285, 240, 285], dtype=np.int64)
        self.table = torch.randn(len(self.ends), 286, 236, generator=generator)
        self.actions = torch.rand(len(self.ends), hn.ACTION_DIM, generator=generator)

    def carrier_rows(self, shots, positions):
        return self.table[torch.as_tensor(shots), torch.as_tensor(positions)]


class SyntheticSource(hc.BatchSource):
    def __init__(self):
        self.hn = hn
        self.output, self.arm, self.seed, self.steps, self.batch = None, ARM, SEED, STEPS, BATCH
        self.spec = hn.ARMS[ARM]
        self.geometry = hn.Geometry(ARM)
        self._data = SyntheticFit()
        self.ends = self._data.ends
        valid = np.array([self.geometry.valid_starts(int(e)) for e in self.ends], dtype=np.int64)
        self.cumulative = np.concatenate([[0], np.cumsum(valid)])


@pytest.fixture
def stop_after(monkeypatch):
    """stop_after(n): SIGTERM to this process after the n-th optimizer step of the next train_run call."""
    previous = {s: signal.signal(s, hc.request_stop) for s in (signal.SIGTERM, signal.SIGUSR1)}
    official = hn.train_step
    budget = {"calls": None}

    def counted(*args, **kwargs):
        stats = official(*args, **kwargs)
        if budget["calls"] is not None:
            budget["calls"] -= 1
            if budget["calls"] == 0:
                os.kill(os.getpid(), signal.SIGTERM)
        return stats

    monkeypatch.setattr(hn, "train_step", counted)

    def arm(n):
        hc.STOP_REQUESTED.clear()
        budget["calls"] = n

    yield arm
    hc.STOP_REQUESTED.clear()
    for s, handler in previous.items():
        signal.signal(s, handler)


def run(out_dir):
    return hc.train_run(None, ARM, SEED, LAM, STEPS, BATCH, out_dir, device="cpu", source=SyntheticSource())


def assert_identical(a, b, path="state"):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape, path
        assert torch.equal(a, b), f"{path} differs"
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            assert_identical(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_identical(x, y, f"{path}[{i}]")
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def test_signal_stop_and_resume_is_bitwise_identical(tmp_path, stop_after):
    n, k = 6, 2
    # uninterrupted: N steps, then a stop signal leaves resume.pt at step N
    stop_after(n)
    with pytest.raises(hc.TrainingPaused):
        run(tmp_path / "continuous")
    continuous = torch.load(tmp_path / "continuous" / "resume.pt", weights_only=False)
    # interrupted: k steps, stop signal, resume, N - k steps, stop signal at step N
    stop_after(k)
    with pytest.raises(hc.TrainingPaused):
        run(tmp_path / "interrupted")
    assert torch.load(tmp_path / "interrupted" / "resume.pt", weights_only=False)["step"] == k
    stop_after(n - k)
    with pytest.raises(hc.TrainingPaused):
        run(tmp_path / "interrupted")
    interrupted = torch.load(tmp_path / "interrupted" / "resume.pt", weights_only=False)
    assert continuous["step"] == interrupted["step"] == n
    assert continuous["cuda_rng"] is None and interrupted["cuda_rng"] is None
    for key in ("model", "optimizer", "scheduler", "cpu_rng"):
        assert_identical(continuous[key], interrupted[key], key)
    # both finish the schedule from there: identical final weights
    stop_after(None)
    reports = [run(tmp_path / name) for name in ("continuous", "interrupted")]
    assert all(r["complete"] for r in reports)
    final = [torch.load(tmp_path / name / "model.pt", weights_only=False)["state_dict"]
             for name in ("continuous", "interrupted")]
    assert_identical(final[0], final[1], "model.pt")
    assert not (tmp_path / "continuous" / "resume.pt").exists()


def test_loader_iteration_draws_from_the_cpu_generator():
    """Why the amended resume restores the RNG after iter(loader): iterating a DataLoader draws its worker base
    seed from the CPU generator, so restoring before that draw (the frozen order) shifts the stream that SIGReg's
    projections use on CPU."""
    saved = torch.get_rng_state()
    iter(torch.utils.data.DataLoader(range(4), batch_size=None, num_workers=0))
    assert not torch.equal(torch.get_rng_state(), saved)


def test_stop_between_arm_seeds_exits_paused(monkeypatch, tmp_path):
    plan = {"training": {"queues": [["HJ-car-m/20260908", "HJ-car-m/20260909"]]},
            "arms": {"HJ-car-m": {"lambda": LAM, "steps": STEPS, "batch": BATCH}}}
    calls = []
    monkeypatch.setattr(hc, "load_plan", lambda output: plan)
    monkeypatch.setattr(hc, "mark", lambda *a, **k: None)

    def fake_run(output, arm, seed, *args):
        calls.append(seed)
        os.kill(os.getpid(), signal.SIGTERM)  # arrives while this arm-seed finishes
        return {"wall_seconds": 1.0, "gpu_seconds": 1.0}

    monkeypatch.setattr(hc, "train_run", fake_run)
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGUSR1)}
    hc.STOP_REQUESTED.clear()
    try:
        assert hc.train(tmp_path, 0) == hc.EXIT_PAUSED
    finally:
        hc.STOP_REQUESTED.clear()
        for s, handler in previous.items():
            signal.signal(s, handler)
    assert calls == [20260908]

    def paused_run(output, arm, seed, *args):
        raise hc.TrainingPaused("paused")

    monkeypatch.setattr(hc, "train_run", paused_run)
    try:
        assert hc.train(tmp_path, 0) == hc.EXIT_PAUSED
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
