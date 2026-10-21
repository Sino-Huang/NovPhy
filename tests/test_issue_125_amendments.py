"""#125 post-freeze amendments (CPU). A1: a relaunched training queue does not add a finished arm-seed's seconds to
the compute ledger a second time. A2: load_plan also runs #123's load_plan (inherited pins), the readouts manifest
verifies every frozen readouts.json report, and the smoke refuses after the freeze."""
from __future__ import annotations

import json
import signal

import pytest

from scripts import run_hjepa_comparison as rc
from scripts import run_hjepa_idm as ri

ARM = "HJ-car-m-IDM"
INHERITED = "world_model/planning/scoring_harness.py"  # pinned by #123's plan only


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


def test_load_plan_checks_inherited_123_inputs(monkeypatch):
    """A2: drift in an input pinned only by #123's plan makes #125's load_plan refuse; #123's registered amendments
    (its runner: frozen -> A1 -> A2 -> A3) are accepted."""
    own = {e["artifact"] for e in json.loads((ri.OUTPUT / "plan.json").read_text())["inputs"]}
    source = json.loads((ri.SOURCE / "plan.json").read_text())
    inherited = {e["artifact"]: e["sha256"] for e in source["inputs"]}
    assert INHERITED in inherited and INHERITED not in own
    assert ri.sha256_of(rc.ROOT / rc.RUNNER) != inherited[rc.RUNNER]  # the amended #123 runner is in use
    assert ri.load_plan(ri.OUTPUT)["identity"] == ri.IDENTITY
    exact = rc.sha256_of
    monkeypatch.setattr(rc, "sha256_of", lambda path: "sha256:" + "0" * 64
                        if path == rc.ROOT / INHERITED else exact(path))
    with pytest.raises(ValueError, match=INHERITED):
        ri.load_plan(ri.OUTPUT)


def frozen_readouts(monkeypatch, tmp_path):
    """A readouts manifest frozen by #125's freeze_readouts over three seeds of ARM (the last a typed failure)."""
    monkeypatch.setattr(ri, "ROOT", tmp_path)
    monkeypatch.setattr(rc, "ROOT", tmp_path)
    output = tmp_path / "out"
    plan = {"arms": {ARM: {}}}
    monkeypatch.setattr(ri, "load_plan", lambda out: plan)
    for seed in ri.SEEDS:
        run = ri.run_dir(output, ARM, seed)
        run.mkdir(parents=True)
        (run / "model.pt").write_bytes(b"model %d" % seed)
        if seed == ri.SEEDS[-1]:
            report = {"typed_failure": "nonfinite_loss"}
        else:
            rc.readouts_path(run).write_bytes(b"readouts %d" % seed)
            report = {"action_sensitivity": 1e-6, "collapse": [{"effective_rank": 14.0, "dim": 112}]}
        (run / "readouts.json").write_text(json.dumps(report))
    assert ri.freeze_readouts(output) == 0
    return output, plan


def test_report_only_drift_fails_closed(monkeypatch, tmp_path):
    """A2: a frozen readouts.json report (the guards' source) that changes alone is caught by load_manifest, although
    #123's check (model / readouts tensors) still passes; score, publication, validation and the spot check refuse,
    and a failed validation leaves no publication."""
    output, plan = frozen_readouts(monkeypatch, tmp_path)
    assert ri.load_manifest(output)["entries"].keys() == {f"{ARM}/{s}" for s in ri.SEEDS}
    for seed in (ri.SEEDS[0], ri.SEEDS[-1]):  # a scored arm-seed and a typed failure
        path = ri.run_dir(output, ARM, seed) / "readouts.json"
        original = path.read_text()
        path.write_text(original.replace("1e-06", "0.5") if seed == ri.SEEDS[0] else json.dumps({"typed_failure": None}))
        rc.load_manifest(output)
        with pytest.raises(ValueError, match=f"{ARM}/{seed}: frozen readouts report changed"):
            ri.load_manifest(output)
        path.write_text(original)
    ri.load_manifest(output)
    (ri.run_dir(output, ARM, ri.SEEDS[1]) / "readouts.json").write_text(json.dumps({"action_sensitivity": 0.5}))
    for call in (lambda: ri.score(output), lambda: ri.compute_tables(output, plan), lambda: ri.publish(output),
                 lambda: ri.spot_check(output, plan)):
        with pytest.raises(ValueError, match="frozen readouts report changed"):
            call()
    assert not any((output / name).exists() for name in ri.PUBLISHED)
    for name in ri.PUBLISHED:
        (output / name).write_text("published")
    with pytest.raises(ValueError, match="frozen readouts report changed"):
        ri.validate(output)
    assert not any((output / name).exists() for name in ri.PUBLISHED)
    receipt = json.loads((output / ri.VALIDATION_RECEIPT).read_text())
    assert receipt["pass"] is False and "frozen readouts report changed" in receipt["error"]


def test_smoke_refuses_after_the_freeze(tmp_path):
    """A2: smoke.json is a frozen input and a re-run would add the smoke seconds again: no smoke once plan.json exists."""
    (tmp_path / "plan.json").write_text("{}")
    with pytest.raises(ValueError, match="frozen"):
        ri.smoke(tmp_path)
    assert not rc.ledger_dir(tmp_path).exists() and not (tmp_path / "smoke.json").exists()
