"""Tier-T night-window stop: SIGTERM -> official resume checkpoint + exit 75, and the fidelity queue's forwarding.

The launcher tests run ``scripts/hjepa_train_ckpt.py`` on CPU under the hjepa env: once with the real
``main_hjepa.py`` (tiny FourRoom-L3 run, all hydra/spt layers) and once with ``hjepa_tier_t_stop_driver.py``
standing in for ``main_hjepa.py`` (deterministic SIGTERM points and per-batch logs for the resume semantics).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

import pytest

from scripts import run_hjepa_fidelity as fid

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "hjepa_train_ckpt.py"
OFFICIAL = fid.HJEPA_REF / "h_jepa"
CPU_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"}
needs_hjepa = pytest.mark.skipif(not (fid.HJEPA_PYTHON.is_file() and (OFFICIAL / "main_hjepa.py").is_file()),
                                 reason="hjepa env or H-JEPA clone absent")


def ckpt_step(path):
    probe = "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['global_step'])"
    return int(subprocess.run([str(fid.HJEPA_PYTHON), "-c", probe, str(path)], check=True, capture_output=True,
                              text=True, env=CPU_ENV).stdout)


@pytest.fixture
def toy(tmp_path):
    """Launch the toy stand-in for ``main_hjepa.py`` through the real launcher; returns (launch, run_dir, seen)."""
    ref = tmp_path / "ref"
    (ref / "h_jepa").mkdir(parents=True)
    for entry in OFFICIAL.iterdir():
        if entry.name != "main_hjepa.py":
            (ref / "h_jepa" / entry.name).symlink_to(entry)
    (ref / "h_jepa" / "main_hjepa.py").symlink_to(ROOT / "tests" / "hjepa_tier_t_stop_driver.py")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def launch(*args):
        return subprocess.run([str(fid.HJEPA_PYTHON), str(LAUNCHER), str(run_dir), *args], capture_output=True,
                              text=True, env={**CPU_ENV, "HJEPA_REF": str(ref)}, timeout=600)

    def seen():
        return [json.loads(line) for line in (run_dir / "seen.jsonl").read_text().splitlines()]

    return launch, run_dir, seen


def by_launch(seen, kind):
    pids = list(dict.fromkeys(entry["pid"] for entry in seen))
    return [[e for e in seen if e["pid"] == pid and e["kind"] == kind] for pid in pids]


def output(result):
    return result.stdout[-3000:] + result.stderr[-3000:]


@needs_hjepa
@pytest.mark.parametrize(("sigterm", "saved_step"), [
    ("train:17", 18),      # mid-epoch 1: the launcher's save
    ("train:15", 16),      # same batch as the official every-4 save: both write the same state
    ("val:0", 12),         # during validation: Lightning stops itself; the last periodic save stays
    ("epoch_end:0", 12),   # during the epoch-end hooks: the official epoch-end save stays
])
def test_toy_stop_exits_75_and_resumes(toy, sigterm, saved_step):
    launch, run_dir, seen = toy
    first = launch("--sigterm", sigterm)
    assert first.returncode == fid.PAUSED, output(first)
    assert ckpt_step(run_dir / "lightning_resume" / "last.ckpt") == saved_step
    second = launch()
    assert second.returncode == 0, output(second)
    assert ckpt_step(run_dir / "lightning_resume" / "last.ckpt") == 24

    before, after = by_launch(seen(), "train")
    assert [e["global_step"] for e in before + after] == list(range(24))
    assert after[0]["global_step"] == saved_step  # resumed epoch runs only its remaining batches
    assert after[0]["batch_idx"] == saved_step % 12
    assert all(e["lr"] == pytest.approx(1.0 / (1 + e["global_step"])) for e in after)  # LR schedule continuous
    assert after[0]["idx"] == before[0]["idx"]  # not fast-forwarded: a fresh shuffle from the re-seeded generator
    assert [len(work) for work in by_launch(seen(), "train_end_work")] == [0, 1]
    assert [e for e in seen() if e["kind"] == "end"][-1]["global_step"] == 24


@needs_hjepa
def test_toy_uninterrupted_run_has_one_train_end(toy):
    launch, _, seen = toy
    result = launch()
    assert result.returncode == 0, output(result)
    (work,) = by_launch(seen(), "train_end_work")
    (module_hook,) = by_launch(seen(), "train_end")
    assert [(e["epoch"], e["global_step"]) for e in work] == [(2, 24)]
    assert len(module_hook) == 1


@needs_hjepa
@pytest.mark.parametrize(("first_args", "first_code", "first_train_end"), [
    (("--sigterm", "epoch_end:1"), fid.PAUSED, 0),  # stopped in the final epoch-end hooks
    (("--sigkill-train-end",), -signal.SIGKILL, 1),  # killed in the train-end work (e.g. planning eval)
])
def test_toy_finished_resume_file_runs_train_end_once(toy, first_args, first_code, first_train_end):
    launch, run_dir, seen = toy
    first = launch(*first_args)
    assert first.returncode == first_code, output(first)
    assert ckpt_step(run_dir / "lightning_resume" / "last.ckpt") == 24  # the final epoch-end save
    second = launch()
    assert second.returncode == 0, output(second)

    work = by_launch(seen(), "train_end_work")
    assert [len(launch_work) for launch_work in work] == [first_train_end, 1]
    assert [(e["epoch"], e["global_step"]) for e in work[1]] == [(2, 24)]  # same epoch dir as an uninterrupted run
    assert len(by_launch(seen(), "train_end")[1]) == 1  # the module's on_train_end too, as in Lightning
    assert by_launch(seen(), "train")[1] == []  # nothing retrained


@needs_hjepa
@pytest.mark.skipif(not (fid.HJEPA_HOME / "fourroom_tp35_d1.h5").is_file(), reason="FourRoom dataset absent")
def test_official_main_hjepa_stop_exits_75_and_resumes(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    for name in ("fourroom_tp35_d1.h5", "fourroom_tp35_d1_val.h5"):
        (home / name).symlink_to(fid.HJEPA_HOME / name)
    argv = [str(fid.HJEPA_PYTHON), str(LAUNCHER), "--config-name", "fourroom_hjepa_l3", "seed=42", "subdir=stop",
            "final_probing_decoding_eval.enabled=false", "planning_eval.enabled=false", "num_workers=2",
            "loader.pin_memory=false", "loader.batch_size=2", "trainer.accelerator=cpu", "trainer.precision=32",
            "trainer.devices=1", "+trainer.limit_train_batches=30", "+trainer.limit_val_batches=1",
            "trainer.max_epochs=2", "resume_every_n_steps=4"]
    env = {**CPU_ENV, "HJEPA_HOME": str(home)}
    resume = home / "ckpts" / "stop" / "lightning_resume" / "last.ckpt"
    log = tmp_path / "first.log"
    with log.open("w") as handle:
        proc = subprocess.Popen(argv, cwd=OFFICIAL, env=env, stdout=handle, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 600
    while "after train batch 4 " not in log.read_text() and proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=600) == fid.PAUSED, log.read_text()[-5000:]
    saved = [int(n) for n in re.findall(r"Resume checkpoint after train batch (\d+) ", log.read_text())]
    assert 4 <= saved[-1] < 30 and ckpt_step(resume) == saved[-1]  # stopped mid-epoch 0 at the official file

    second = subprocess.run(argv, cwd=OFFICIAL, env=env, capture_output=True, text=True, timeout=900)
    assert second.returncode == 0, second.stdout[-5000:]
    assert f"Resuming full training state from {resume}" in second.stdout + second.stderr
    assert ckpt_step(resume) == 60
    assert (second.stdout + second.stderr).count("Saved final world model to") == 1  # official on_train_end, once

    # A relaunch from the finished resume file (as after a kill in the train-end work): Lightning skips the fit
    # loop; the launcher runs the official train-end hooks once.
    third = subprocess.run(argv, cwd=OFFICIAL, env=env, capture_output=True, text=True, timeout=900)
    assert third.returncode == 0, third.stdout[-5000:]
    assert "Resume checkpoint after train batch" not in third.stdout + third.stderr  # nothing retrained
    assert (third.stdout + third.stderr).count("Saved final world model to") == 1
    assert ckpt_step(resume) == 60


@pytest.fixture
def fake_queue(tmp_path, monkeypatch):
    """Run ``fid.queue('tier-t', ...)`` with each job's command replaced by ``python -c children[seed]``."""
    (tmp_path / "ref" / "h_jepa").mkdir(parents=True)
    monkeypatch.setattr(fid, "OUTPUT", tmp_path / "out")
    monkeypatch.setattr(fid, "HJEPA_HOME", tmp_path / "home")
    monkeypatch.setattr(fid, "HJEPA_REF", tmp_path / "ref")
    started = []

    def run(children):
        def command(tier, model, seed):
            started.append(seed)
            return [sys.executable, "-c", children[seed]], None

        monkeypatch.setattr(fid, "command", command)
        code = fid.queue("tier-t", 0, [("hjepa_l3", seed) for seed in children])
        ledger = [json.loads(line) for line in (tmp_path / "out" / "ledger.jsonl").read_text().splitlines()]
        return code, [(entry["seed"], entry["exit_code"]) for entry in ledger]

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGUSR1)}
    yield run, started
    for sig, handler in handlers.items():
        signal.signal(sig, handler)


def job_child(seed, code=0, metrics=True):
    """Child source that exits ``code``, first writing the job's planning-eval metrics if ``metrics``
    (call after ``fake_queue`` patched HJEPA_HOME)."""
    path = fid.train_dir("hjepa_l3", seed) / "planning_eval" / "epoch_0008" / "metrics.yaml"
    write = (f"m = pathlib.Path({str(path)!r}); m.parent.mkdir(parents=True); m.write_text('success_rate: 90.0\\n'); "
             if metrics else "")
    return f"import pathlib, sys; {write}sys.exit({code})"


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGUSR1])
def test_queue_forwards_stop_records_job_and_starts_no_more(tmp_path, fake_queue, signum):
    run, started = fake_queue
    ready = tmp_path / "ready"
    paused = ("import signal, sys, time; signal.signal(signal.SIGTERM, lambda *_: sys.exit(75)); "
              f"open({str(ready)!r}, 'w').close(); time.sleep(120)")

    def stop_when_ready():
        while not ready.exists():
            time.sleep(0.05)
        os.kill(os.getpid(), signum)

    threading.Thread(target=stop_when_ready, daemon=True).start()
    code, ledger = run({42: paused, 43: job_child(43)})
    assert code == fid.PAUSED
    assert started == [42]
    assert ledger == [(42, 75)]
    assert not (tmp_path / "out" / "logs" / "tier-t--hjepa_l3--seed43.log").exists()


@pytest.mark.parametrize(("jobs", "expected_code", "expected_ledger"), [
    ({42: (0, True), 43: (0, True)}, 0, [(42, 0), (43, 0)]),
    ({42: (3, False), 43: (0, True)}, 1, [(42, 3), (43, 0)]),  # a failing job; the queue goes on
    ({42: (1, True)}, 1, [(42, 1)]),  # non-zero exit despite metrics
    ({42: (0, False), 43: (0, True)}, 1, [(42, 0), (43, 0)]),  # exit 0 without planning-eval metrics
])
def test_queue_returns_0_only_when_every_job_completed(fake_queue, jobs, expected_code, expected_ledger):
    run, started = fake_queue
    code, ledger = run({seed: job_child(seed, code, metrics) for seed, (code, metrics) in jobs.items()})
    assert (code, ledger, started) == (expected_code, expected_ledger, list(jobs))
