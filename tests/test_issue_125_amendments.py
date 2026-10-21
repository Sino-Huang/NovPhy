"""#125 post-freeze amendment A1 (CPU): a relaunched training queue does not add a finished arm-seed's seconds to the
compute ledger a second time."""
from __future__ import annotations

import signal

from scripts import run_hjepa_comparison as rc
from scripts import run_hjepa_idm as ri

ARM = "HJ-car-m-IDM"


def test_relaunch_does_not_mark_finished_runs_again(monkeypatch, tmp_path):
    plan = {"training": {"queues": [[f"{ARM}/20260908", f"{ARM}/20260909"]]},
            "arms": {ARM: {"lambda": 0.18, "gamma1": 225.0, "steps": 8, "batch": 4}}}
    monkeypatch.setattr(ri, "load_plan", lambda output: plan)

    night = {"n": 1}

    def fake_run(output, arm, seed, lam, gamma1, steps, batch, out_dir):
        """#123's train_run contract: a finished arm-seed returns its stored report (the same seconds again)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        if not (out_dir / "model.pt").is_file():
            if seed == 20260909 and night["n"] == 1:
                raise rc.TrainingPaused("paused")  # night 1 ends during the second arm-seed
            (out_dir / "model.pt").touch()
        return {"wall_seconds": 100.0, "gpu_seconds": 100.0}

    monkeypatch.setattr(ri, "train_run", fake_run)
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGUSR1)}
    rc.STOP_REQUESTED.clear()
    try:
        assert ri.train(tmp_path, 0) == rc.EXIT_PAUSED  # night 1: seed 08 finishes, seed 09 pauses
        night["n"] = 2
        assert ri.train(tmp_path, 0) == 0  # night 2: seed 08 re-visited (finished), seed 09 finishes
        assert ri.train(tmp_path, 0) == 0  # a further relaunch adds nothing
    finally:
        rc.STOP_REQUESTED.clear()
        for s, handler in previous.items():
            signal.signal(s, handler)
    entries, gpu, _ = rc.ledger_totals(tmp_path)
    assert gpu == 200.0 and sorted(entries) == [f"train-{ARM}-seed20260908", f"train-{ARM}-seed20260909"]
    assert all(e["gpu_seconds"] == 100.0 for e in entries.values())
