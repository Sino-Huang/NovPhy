"""#123 post-freeze amendment A3 (CPU): a relaunched training queue adds no ledger seconds for a finished arm-seed,
and the compute aggregation counts each arm-seed's training once (its report.json seconds)."""
from __future__ import annotations

import json
import signal

import pytest

from scripts import run_hjepa_comparison as rc
from scripts import run_hjepa_idm as ri

ARM = "HJ-car-m"


def test_relaunch_over_a_finished_arm_seed_adds_no_ledger_seconds(monkeypatch, tmp_path):
    plan = {"training": {"queues": [[f"{ARM}/20260908", f"{ARM}/20260909"]]},
            "arms": {ARM: {"lambda": 0.18, "steps": 8, "batch": 4}}}
    monkeypatch.setattr(rc, "load_plan", lambda output: plan)
    night = {"n": 1}

    def fake_run(output, arm, seed, lam, steps, batch, out_dir):
        """train_run's contract: a finished arm-seed (model.pt exists) returns its stored report again."""
        out_dir.mkdir(parents=True, exist_ok=True)
        if not (out_dir / "model.pt").is_file():
            if seed == 20260909 and night["n"] == 1:
                raise rc.TrainingPaused("paused")  # night 1 ends during the second arm-seed
            (out_dir / "model.pt").touch()
            rc.write_json(out_dir / "report.json", {"wall_seconds": 100.0, "gpu_seconds": 100.0})
        return rc.read_json(out_dir / "report.json")

    monkeypatch.setattr(rc, "train_run", fake_run)
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGUSR1)}
    rc.STOP_REQUESTED.clear()
    try:
        assert rc.train(tmp_path, 0) == rc.EXIT_PAUSED  # night 1: seed 08 finishes, seed 09 pauses
        night["n"] = 2
        assert rc.train(tmp_path, 0) == 0  # night 2: the queue passes over the finished seed 08
        assert rc.train(tmp_path, 0) == 0  # a further relaunch over both finished arm-seeds
    finally:
        rc.STOP_REQUESTED.clear()
        for s, handler in previous.items():
            signal.signal(s, handler)
    for seed in (20260908, 20260909):
        raw = json.loads((tmp_path / "ledger" / f"train-{ARM}-seed{seed}.json").read_text())
        assert raw["gpu_seconds"] == raw["wall_seconds"] == 100.0
    entries, gpu, wall = rc.ledger_totals(tmp_path)
    assert gpu == wall == 200.0 and all("ledger_gpu_seconds" not in e for e in entries.values())


@pytest.mark.parametrize("run_dir", [rc.run_dir, ri.run_dir], ids=["issue-123", "issue-125"])
def test_aggregation_counts_each_arm_seed_once(tmp_path, run_dir):
    arm = "HJ-px-IDM" if run_dir is ri.run_dir else "HJ-px"
    once = {"wall_seconds": 61231.5, "gpu_seconds": 61231.5}
    rc.write_json(run_dir(tmp_path, arm, 20260908) / "report.json", once)
    rc.write_json(run_dir(tmp_path, arm, 20260909) / "report.json", once)
    rc.mark(tmp_path, f"train-{arm}-seed20260908", 61231.5, 61231.5)
    rc.mark(tmp_path, f"train-{arm}-seed20260908", 61231.5, 61231.5)  # the pre-A3 relaunch double
    rc.mark(tmp_path, f"train-{arm}-seed20260909", 61231.5, 61231.5)
    rc.mark(tmp_path, f"selection-train-{arm}-lam0.18", 10.0, 10.0)  # not a main training phase: as recorded
    rc.mark(tmp_path, f"readouts-{arm}-seed20260908", 5.0, 5.0)
    rc.mark(tmp_path, f"readouts-{arm}-seed20260908", 5.0, 5.0)  # repeated non-training work counts as recorded
    entries, gpu, wall = rc.ledger_totals(tmp_path)
    doubled = entries[f"train-{arm}-seed20260908"]
    assert doubled["gpu_seconds"] == doubled["wall_seconds"] == 61231.5
    assert doubled["ledger_gpu_seconds"] == doubled["ledger_wall_seconds"] == 2 * 61231.5
    assert "ledger_gpu_seconds" not in entries[f"train-{arm}-seed20260909"]
    assert entries[f"readouts-{arm}-seed20260908"]["gpu_seconds"] == 10.0
    assert gpu == wall == 2 * 61231.5 + 10.0 + 10.0


def test_issue_123_training_total_reconciles_with_reports():
    if not (rc.OUTPUT / "ledger").is_dir():
        pytest.skip("issue-123 artifacts absent")
    entries, _, _ = rc.ledger_totals(rc.OUTPUT)
    training = {p: e for p, e in entries.items() if rc.TRAIN_PHASE.fullmatch(p)}
    reports = sorted((rc.OUTPUT / "models").glob("*/seed-*/report.json"))
    assert len(training) == len(reports) == 33
    assert sum(e["gpu_seconds"] for e in training.values()) == pytest.approx(
        sum(rc.read_json(r)["gpu_seconds"] for r in reports), rel=1e-12)
    assert sorted(p for p, e in training.items() if "ledger_gpu_seconds" in e) == [
        "train-HJ-px-seed20260908", "train-HJ-px-seed20260909", "train-HJ-px-seed20260910",
        "train-HWM-px-seed20260908"]
