"""Night-window GPU scheduler (scripts/gpu_window.py): window rules, DST, launch rules, and the tick cycle with
real processes (toy jobs, simulated clock, fake nvidia-smi). CPU only."""
from __future__ import annotations

from datetime import datetime
import fcntl
import json
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from scripts import gpu_window as gw

Z = gw.ZONE


def at(text):
    return datetime.fromisoformat(text).replace(tzinfo=Z)


@pytest.mark.parametrize("clock, expected", [("17:59", "day"), ("18:00", "run"), ("23:30", "run"), ("00:00", "run"),
                                            ("06:49", "run"), ("06:50", "stop"), ("06:59", "stop"),
                                            ("07:00", "day"), ("12:00", "day")])
def test_phase_boundaries(clock, expected):
    assert gw.phase(at(f"2026-10-19T{clock}")) == expected


def test_fit_check_uses_elapsed_time_across_dst():
    """Melbourne DST: 2026-10-04 02:00 -> 03:00 (night 3 to 4 Oct is 1 h shorter), 2027-04-04 03:00 -> 02:00."""
    job = {"id": "x", "state": "queued", "depends_on": [], "resumable": False, "gpu": None, "shares_gpu": False}
    spring, autumn = at("2026-10-03T23:00"), at("2027-04-03T23:00")
    assert gw.next_stop(spring).timestamp() - spring.timestamp() == 410 * 60  # wall clock says 470
    assert gw.next_stop(autumn).timestamp() - autumn.timestamp() == 530 * 60  # wall clock says 470
    assert gw.launch_blocker({**job, "est_minutes": 420}, [], spring) is not None
    assert gw.launch_blocker({**job, "est_minutes": 520}, [], autumn) is None


def test_launch_rules():
    base = {"state": "queued", "depends_on": [], "resumable": True, "est_minutes": None, "gpu": 0, "shares_gpu": False}
    train = {**base, "id": "train", "state": "running"}
    tier_t = {**base, "id": "tier-t", "shares_gpu": True}
    readouts = {**base, "id": "readouts", "depends_on": ["train"]}
    other = {**base, "id": "other"}
    final = {**base, "id": "final", "gpu": None, "resumable": False, "est_minutes": 60, "depends_on": ["readouts"]}
    jobs = [train, tier_t, readouts, other, final]
    night = at("2026-10-19T22:00")
    assert gw.launch_blocker(tier_t, jobs, night) is None  # shares the training GPU
    assert gw.launch_blocker(other, jobs, night).startswith("GPU 0 busy")
    assert gw.launch_blocker(readouts, jobs, night) == "waits for train"
    train["state"], readouts["state"] = "failed", "completed"
    assert gw.launch_blocker({**readouts, "state": "queued"}, jobs, night).startswith("blocked: failed")
    assert gw.launch_blocker(final, jobs, night) is None
    assert gw.launch_blocker(final, jobs, at("2026-10-20T06:00")).startswith("not resumable")


TOY = textwrap.dedent("""
    import pathlib, signal, subprocess, sys, time
    mode, marker = sys.argv[1], pathlib.Path(sys.argv[2])
    if mode == "resumable":          # first run: work until SIGTERM, checkpoint, exit 75; second run: finish
        if marker.exists():
            sys.exit(0)
        stop = []
        signal.signal(signal.SIGTERM, lambda s, f: stop.append(s))
        marker.with_suffix(".started").write_text("x")
        while not stop:
            time.sleep(0.05)
        marker.write_text("checkpoint")
        sys.exit(75)
    if mode == "fail":
        sys.exit(3)
    if mode == "stubborn":           # ignores SIGTERM and has a child: only SIGKILL of the whole run ends it
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        marker.write_text(str(child.pid))
        time.sleep(600)
""")

FAKE_SMI = textwrap.dedent("""\
    #!/bin/sh
    case "$*" in
      *query-gpu*) echo "0, GPU-aaaa" ;;
      *query-compute-apps*) cat "{apps}" ;;
    esac
""")


@pytest.fixture
def sched(tmp_path, monkeypatch):
    home = tmp_path / "gpu-window"
    monkeypatch.setattr(gw, "HOME", home)
    apps = tmp_path / "apps.csv"
    apps.write_text("")
    smi = tmp_path / "nvidia-smi"
    smi.write_text(FAKE_SMI.format(apps=apps))
    smi.chmod(0o755)
    monkeypatch.setattr(gw, "NVIDIA_SMI", str(smi))
    toy = tmp_path / "toy.py"
    toy.write_text(TOY)

    def job(job_id, mode, **spec):
        return {"id": job_id, "command": [sys.executable, str(toy), mode, str(tmp_path / f"{job_id}.marker")],
                "resumable": True, **spec}

    def enqueue(*jobs):
        spec = tmp_path / "spec.json"
        spec.write_text(json.dumps({"jobs": list(jobs)}))
        gw.enqueue(spec, None, at("2026-10-19T12:00"))

    yield type("S", (), {"home": home, "apps": apps, "job": staticmethod(job), "enqueue": staticmethod(enqueue),
                         "tmp": tmp_path})
    for pid in gw.members("t-", exact=False):
        os.kill(pid, signal.SIGKILL)


def wait_for(condition, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.05)
    raise AssertionError("condition not reached")


def jobs():
    return {j["id"]: j for j in gw.load()["jobs"]}


def events(kind):
    return [json.loads(line) for line in (gw.HOME / "events.jsonl").read_text().splitlines()
            if json.loads(line)["event"] == kind]


def test_tick_cycle_with_real_processes(sched):
    sched.enqueue(sched.job("t-train", "resumable", gpu=0),
                  sched.job("t-fail", "fail", gpu=1),
                  sched.job("t-after-fail", "fail", gpu=2, depends_on=["t-fail"]),
                  sched.job("t-after-train", "fail", gpu=None, resumable=False, est_minutes=30,
                            depends_on=["t-train"]))
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        sched.apps.write_text(f"GPU-aaaa, {foreign.pid}, 512\n")
        # daytime: nothing launches, nvidia-smi is not called
        gw.tick(at("2026-10-19T12:00"))
        assert all(j["state"] == "queued" and not j["runs"] for j in jobs().values())
        assert not (gw.HOME / "events.jsonl").read_text().count('"launch"')
        # 19:00: every ready job on its free GPU; the foreign GPU process is logged, never signalled
        gw.tick(at("2026-10-19T19:00"))
        state = jobs()
        assert state["t-train"]["state"] == state["t-fail"]["state"] == "running"
        assert state["t-after-fail"]["state"] == state["t-after-train"]["state"] == "queued"
        assert events("foreign_gpu_processes")[0]["processes"] == [{"gpu": 0, "pid": foreign.pid, "mib": 512.0}]
        wait_for(lambda: (sched.tmp / "t-train.started").exists())
        run = state["t-train"]["runs"][0]
        assert run["pid"] in gw.members(run["tag"]) and os.getpgid(run["pid"]) == run["pid"]
        wait_for(lambda: os.path.exists(state["t-fail"]["runs"][0]["exit_file"]))
        # idempotent: a second tick at the same time launches nothing new
        gw.tick(at("2026-10-19T19:00"))
        state = jobs()
        assert len(state["t-train"]["runs"]) == 1
        assert state["t-fail"]["state"] == "failed" and state["t-fail"]["runs"][0]["exit_code"] == 3
        gw.tick(at("2026-10-19T19:05"))
        assert jobs()["t-fail"]["state"] == "failed" and not jobs()["t-after-fail"]["runs"]  # never retried
        # 06:50: SIGTERM once; the job checkpoints and exits 75 -> paused
        gw.tick(at("2026-10-20T06:50"))
        gw.tick(at("2026-10-20T06:50"))
        assert len(events("stop")) == 1
        wait_for(lambda: os.path.exists(jobs()["t-train"]["runs"][0]["exit_file"]))
        gw.tick(at("2026-10-20T06:55"))
        assert jobs()["t-train"]["state"] == "paused" and jobs()["t-train"]["runs"][0]["exit_code"] == 75
        # day: nothing launches, nothing left to kill
        gw.tick(at("2026-10-20T07:00"))
        gw.tick(at("2026-10-20T12:00"))
        assert jobs()["t-train"]["state"] == "paused" and len(jobs()["t-train"]["runs"]) == 1
        assert not events("overrun_killed")
        # next window: the paused job resumes and completes; its dependent then launches
        gw.tick(at("2026-10-20T18:00"))
        assert jobs()["t-train"]["state"] == "running" and len(jobs()["t-train"]["runs"]) == 2
        wait_for(lambda: os.path.exists(jobs()["t-train"]["runs"][1]["exit_file"]))
        gw.tick(at("2026-10-20T18:05"))
        assert jobs()["t-train"]["state"] == "completed"
        assert jobs()["t-after-train"]["state"] == "running"
        assert foreign.poll() is None
    finally:
        foreign.kill()
        foreign.wait()


def test_overrun_is_killed_at_seven_and_requeued(sched):
    sched.enqueue(sched.job("t-stubborn", "stubborn", gpu=0))
    gw.tick(at("2026-10-19T23:00"))
    wait_for(lambda: (sched.tmp / "t-stubborn.marker").exists() and (sched.tmp / "t-stubborn.marker").read_text())
    grandchild = int((sched.tmp / "t-stubborn.marker").read_text())
    tag = jobs()["t-stubborn"]["runs"][0]["tag"]
    assert grandchild in gw.members(tag)
    gw.tick(at("2026-10-20T06:50"))
    time.sleep(0.5)
    gw.tick(at("2026-10-20T06:55"))
    assert jobs()["t-stubborn"]["state"] == "running"  # ignored SIGTERM
    gw.tick(at("2026-10-20T07:00"))
    assert events("overrun_killed")[0]["job"] == "t-stubborn"
    wait_for(lambda: not gw.members(tag))
    gw.tick(at("2026-10-20T07:05"))
    job = jobs()["t-stubborn"]
    assert job["state"] == "paused" and job["runs"][0]["exit_code"] is None  # interruption, not failure
    assert job["runs"][0]["killed_at"] and job["runs"][0]["stop_sent_at"]


def test_lock_orphans_hold_and_requeue(sched):
    sched.enqueue(sched.job("t-a", "resumable", gpu=0), sched.job("t-b", "fail", gpu=1))
    with open(gw.HOME / "lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        gw.tick(at("2026-10-19T20:00"))
    assert events("tick_skipped") and not jobs()["t-a"]["runs"]
    # processes of an unrecorded run of t-a (a tick that died after launching) block a second launch
    orphan = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                              env={**os.environ, gw.TAG: "t-a#1"})
    try:
        gw.tick(at("2026-10-19T20:00"))
        assert events("launch_skipped")[0]["pids"] == [orphan.pid] and not jobs()["t-a"]["runs"]
    finally:
        orphan.kill()
        orphan.wait()
    gw.tick(at("2026-10-19T20:05"))
    wait_for(lambda: (sched.tmp / "t-a.started").exists())
    # operator stop: SIGTERM now and no launch before the next window opens
    gw.stop_now(at("2026-10-19T20:10"), kill=False)
    wait_for(lambda: os.path.exists(jobs()["t-a"]["runs"][0]["exit_file"]))
    gw.tick(at("2026-10-19T20:15"))
    assert jobs()["t-a"]["state"] == "paused" and len(jobs()["t-a"]["runs"]) == 1
    gw.tick(at("2026-10-20T18:00"))
    assert len(jobs()["t-a"]["runs"]) == 2
    with pytest.raises(ValueError):
        gw.enqueue(None, "t-a", at("2026-10-20T18:00"))  # running jobs are not requeued
    wait_for(lambda: os.path.exists(jobs()["t-b"]["runs"][-1]["exit_file"]))
    gw.tick(at("2026-10-20T18:05"))
    assert jobs()["t-b"]["state"] == "failed"
    gw.enqueue(None, "t-b", at("2026-10-20T18:05"))
    assert jobs()["t-b"]["state"] == "queued"


def test_spec_validation(sched):
    with pytest.raises(ValueError, match="needs est_minutes"):
        sched.enqueue(sched.job("t-x", "fail", resumable=False))
    with pytest.raises(ValueError, match="not an earlier job"):
        sched.enqueue(sched.job("t-x", "fail", depends_on=["t-later"]))
    with pytest.raises(ValueError, match="unknown fields"):
        sched.enqueue(sched.job("t-x", "fail", priority=1))
    sched.enqueue(sched.job("t-x", "fail"))
    sched.enqueue(sched.job("t-x", "fail"))  # identical spec: no-op
    with pytest.raises(ValueError, match="different spec"):
        sched.enqueue(sched.job("t-x", "fail", gpu=3))


def test_dry_run_in_daytime_launches_nothing(sched, capsys):
    sched.enqueue(sched.job("t-a", "resumable", gpu=0, est_minutes=1000),
                  sched.job("t-b", "fail", gpu=None, resumable=False, est_minutes=60, depends_on=["t-a"]))
    gw.dry_run(at("2026-10-19T11:00"))
    out = capsys.readouterr().out
    assert "tick now: launches nothing (outside the window)" in out
    assert "launches ['t-a']" in out
    # t-a: 770 min on night 1 (18:00-06:50), the rest on night 2; t-b after it
    assert "t-a" in out and "end 2026-10-20T21:50:00+11:00" in out
    assert "end 2026-10-20T22:50:00+11:00" in out
    assert not jobs()["t-a"]["runs"] and not (gw.HOME / "events.jsonl").read_text().count('"launch"')


def test_issue_123_spec_is_valid():
    spec = json.loads((gw.ROOT / "scripts/gpu_window_issue_123.json").read_text())["jobs"]
    known = {}
    for entry in spec:
        known[entry["id"]] = gw.validate_spec(entry, known)
    assert [j for j in known if j.startswith("123-train")] == [f"123-train-q{g}" for g in range(4)]
    assert known["123-validate"]["depends_on"] == ["123-publish"]
