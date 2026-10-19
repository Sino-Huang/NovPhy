"""Night-window GPU scheduler: every GPU job runs only between 18:00 and 07:00 Australia/Melbourne.

Owner rule of 2026-10-19 (round2 CONVENTIONS.md, "GPU time window"); manual: docs/gpu-window-scheduler.md.
cron runs ``tick`` every 5 minutes through scripts/gpu_window_cron.sh.

tick, local time in [18:00, 06:50): reconcile ended runs, log foreign GPU processes, then launch every ready job
    (queued or paused, every dependency completed, its GPU free) in queue order. A non-resumable job launches only
    if its est_minutes end before 06:50. Surviving processes of a finished run of the job are SIGKILLed (and
    logged as stray_killed) before it is relaunched. Each run is started in its own session (setsid) as
    ``gpu_window.py _wrap --stop-at T1 --kill-at T2 EXIT_FILE -- COMMAND``: the wrapper forwards SIGTERM / SIGUSR1
    to the command and writes the command's exit status to EXIT_FILE. Independently of cron, the wrapper itself
    SIGTERMs the command at T1 (the 06:50 ending the launch's window) and SIGKILLs every other process of the run
    (its process group and its tag) at T2 (07:00), by its own clock, and logs both (window_stop / window_kill).
tick, [06:50, 07:00): SIGTERM, once per run, to the wrapper of every running job (it forwards it to the command
    unless it already sent its own window_stop).
tick, [07:00, 18:00): SIGKILL every process still alive of any run of any job (logged as overrun_killed for a
    running job, stray_killed for a finished one). Never launches.

A run's processes are found by the environment variable GPU_WINDOW_RUN=<job>#<run> that the wrapper and everything
it starts inherit, so the scheduler signals only processes it launched. A run ends
- completed: exit 0;
- paused: exit 75, or any end after a scheduler or wrapper stop (a schedule interruption is not a failure); it is
  relaunched in the next window and resumes from its own checkpoint;
- failed: any other exit (typed failures included); never retried. ``enqueue --requeue ID`` after a fix.
``enqueue SPEC --update-deps`` changes the dependencies of queued or paused jobs to those in SPEC (under the lock).

State in .local-artifacts/gpu-window/ (GPU_WINDOW_HOME overrides): queue.json (jobs and runs), events.jsonl,
logs/<job>.log (stdout/stderr of every run), runs/<job>.<n>.exit.json, lock (flock; ticks never overlap).
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, time as clock, timedelta
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(os.environ.get("GPU_WINDOW_HOME", ROOT / ".local-artifacts" / "gpu-window"))
ZONE = ZoneInfo("Australia/Melbourne")
OPEN, STOP, CLOSE = clock(18, 0), clock(6, 50), clock(7, 0)
TICK = timedelta(minutes=5)
EXIT_PAUSED = 75
TAG = "GPU_WINDOW_RUN"
NVIDIA_SMI = os.environ.get("GPU_WINDOW_NVIDIA_SMI", "nvidia-smi")
JOB_ID = re.compile(r"[A-Za-z0-9_.-]+")
SPEC_DEFAULTS = {"gpu": None, "cwd": ".", "env": {}, "depends_on": [], "est_minutes": None, "shares_gpu": False,
                 "note": ""}


# --------------------------------------------------------------------------- time

def local(now):
    return now.astimezone(ZONE)


def stamp(now):
    return local(now).isoformat(timespec="seconds")


def phase(now):
    """'run' in [18:00, 06:50), 'stop' in [06:50, 07:00), 'day' in [07:00, 18:00), local time."""
    t = local(now).time()
    if t >= OPEN or t < STOP:
        return "run"
    return "stop" if t < CLOSE else "day"


def _next(now, at):
    now = local(now)
    day = now.date() if now.time() < at else now.date() + timedelta(days=1)
    return datetime.combine(day, at, tzinfo=ZONE)


def next_stop(now):
    return _next(now, STOP)


def next_open(now):
    return _next(now, OPEN)


def window_ends(now):
    """(06:50, 07:00) ending the window that a run launched at now runs in; the wrapper enforces both itself."""
    kill_at = _next(now, CLOSE)
    return datetime.combine(kill_at.date(), STOP, tzinfo=ZONE), kill_at


# --------------------------------------------------------------------------- state

def load():
    path = HOME / "queue.json"
    return json.loads(path.read_text()) if path.is_file() else {"jobs": [], "hold_until": None}


def write_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def save(queue):
    write_atomic(HOME / "queue.json", queue)


def event(now, kind, home=None, **fields):
    home = Path(home or HOME)
    entry = {"at": stamp(now), "event": kind, **fields}
    home.mkdir(parents=True, exist_ok=True)
    with (home / "events.jsonl").open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    print(json.dumps(entry, sort_keys=True), flush=True)


@contextlib.contextmanager
def locked(blocking):
    HOME.mkdir(parents=True, exist_ok=True)
    with open(HOME / "lock", "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        yield True


# --------------------------------------------------------------------------- processes

def members(value, exact=True):
    """PIDs whose environment holds GPU_WINDOW_RUN=value (exact) or GPU_WINDOW_RUN=value... (prefix)."""
    needle = f"\0{TAG}={value}".encode() + (b"\0" if exact else b"")
    found = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            with open(f"/proc/{entry.name}/environ", "rb") as handle:
                environ = b"\0" + handle.read()
        except OSError:
            continue
        if needle in environ:
            found.append(int(entry.name))
    return sorted(found)


def sigkill(pids):
    for pid in pids:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


def wait_gone(pids, prefix, timeout=10.0):
    """True once none of pids carries a tag starting with prefix any more (killed processes leave /proc environ)."""
    deadline = time.monotonic() + timeout
    while set(pids) & set(members(prefix, exact=False)):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)
    return True


def run_processes(tag):
    """Every process of this wrapper's run but the wrapper: its tag, and its process group if it leads one (setsid)."""
    me = os.getpid()
    pids = set(members(tag)) if tag else set()
    if os.getpgrp() == me:
        for entry in os.scandir("/proc"):
            if entry.name.isdigit():
                with contextlib.suppress(OSError):
                    if os.getpgid(int(entry.name)) == me:
                        pids.add(int(entry.name))
    pids.discard(me)
    return sorted(pids)


def gpu_processes():
    """[(gpu index, pid, MiB)] of every compute process, or None if nvidia-smi fails. Called inside the window only."""
    try:
        def query(*args):
            out = subprocess.run([NVIDIA_SMI, *args, "--format=csv,noheader,nounits"], capture_output=True,
                                 text=True, timeout=60, check=True).stdout
            return [[f.strip() for f in line.split(",")] for line in out.splitlines() if line.strip()]
        index = {uuid: int(i) for i, uuid in query("--query-gpu=index,uuid")}
        return [(index.get(uuid), int(pid), float(mib)) for uuid, pid, mib in
                query("--query-compute-apps=gpu_uuid,pid,used_memory")]
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def launch(job, now):
    n = len(job["runs"]) + 1
    tag = f"{job['id']}#{n}"
    exit_file = HOME / "runs" / f"{job['id']}.{n}.exit.json"
    log = HOME / "logs" / f"{job['id']}.log"
    exit_file.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **job["env"], TAG: tag}
    stop_at, kill_at = window_ends(now)
    with log.open("a") as handle:
        handle.write(f"\n=== gpu-window {tag} launched {stamp(now)} on GPU {job['gpu']}: "
                     f"{shlex.join(job['command'])} ===\n")
        handle.flush()
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_wrap", "--stop-at", stop_at.isoformat(),
                                 "--kill-at", kill_at.isoformat(), str(exit_file), "--", *job["command"]],
                                cwd=ROOT / job["cwd"], env=env, stdin=subprocess.DEVNULL, stdout=handle,
                                stderr=subprocess.STDOUT, start_new_session=True)
    job["runs"].append({"run": n, "tag": tag, "pid": proc.pid, "pgid": proc.pid, "launched_at": stamp(now),
                        "launched_unix": now.timestamp(), "exit_file": str(exit_file), "log": str(log),
                        "window_stop_at": stamp(stop_at), "window_kill_at": stamp(kill_at),
                        "stop_sent_at": None, "killed_at": None, "killed_unix": None, "ended_at": None,
                        "exit_code": None, "outcome": None, "seconds": None})
    job["state"] = "running"
    event(now, "launch", job=job["id"], run=n, pid=proc.pid, gpu=job["gpu"])


def wrap(exit_file, argv, stop_at=None, kill_at=None):
    """Run argv in this session, forward SIGTERM / SIGUSR1 to it, write its exit status to exit_file. By its own clock
    (no cron needed), SIGTERM argv at stop_at and SIGKILL every other process of the run at kill_at; both are logged
    to the job log and events.jsonl and recorded in exit_file. argv gets at most one SIGTERM."""
    began = time.monotonic()
    home = Path(exit_file).resolve().parent.parent
    tag = os.environ.get(TAG)
    job, _, run = (tag or "").partition("#")
    run = int(run) if run.isdigit() else None
    child, pending, termed = None, [], []
    window = {"window_stop_sent_at": None, "window_killed_at": None}

    def deliver(signum):
        if signum == signal.SIGTERM:
            if termed:
                return
            termed.append(signum)
        if child.poll() is None:
            child.send_signal(signum)

    def forward(signum, _frame):
        if child is None:
            pending.append(signum)
        else:
            deliver(signum)

    for signum in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, forward)
    child = subprocess.Popen(argv)
    for signum in pending:
        deliver(signum)
    deadlines = [(at, key) for at, key in ((stop_at, "window_stop_sent_at"), (kill_at, "window_killed_at")) if at]
    while True:
        due = [at for at, key in deadlines if window[key] is None]
        timeout = min(max(min(due).timestamp() - time.time(), 0.0), 60.0) if due else None
        try:
            code = child.wait(timeout=timeout)
            break
        except subprocess.TimeoutExpired:
            pass
        now = datetime.now(ZONE)
        if stop_at and window["window_stop_sent_at"] is None and now >= stop_at:
            window["window_stop_sent_at"] = stamp(now)
            already = bool(termed)
            deliver(signal.SIGTERM)
            event(now, "window_stop", home, job=job, run=run, pid=child.pid, signal="SIGTERM",
                  already_forwarded=already, reason=f"wrapper clock reached {stamp(stop_at)}")
        if kill_at and window["window_killed_at"] is None and now >= kill_at:
            pids = run_processes(tag)
            sigkill(pids)
            window["window_killed_at"] = stamp(now)
            event(now, "window_kill", home, job=job, run=run, pids=pids, signal="SIGKILL",
                  reason=f"wrapper clock reached {stamp(kill_at)}")
    write_atomic(exit_file, {"exit_code": code, "ended_at": stamp(datetime.now(ZONE)),
                             "seconds": time.monotonic() - began, **window})
    return 0


# --------------------------------------------------------------------------- scheduling rules

def launch_blocker(job, jobs, now):
    """None if a tick in the run phase may launch job now, else why not (shared by tick, status and the projection)."""
    if job["state"] not in ("queued", "paused"):
        return job["state"]
    by_id = {j["id"]: j for j in jobs}
    failed = [d for d in job["depends_on"] if by_id[d]["state"] == "failed"]
    if failed:
        return "blocked: failed dependency " + ", ".join(failed)
    waiting = [d for d in job["depends_on"] if by_id[d]["state"] != "completed"]
    if waiting:
        return "waits for " + ", ".join(waiting)
    if not job["resumable"] and now.timestamp() + 60 * job["est_minutes"] > next_stop(now).timestamp():
        return f"not resumable: {job['est_minutes']} min do not fit before {next_stop(now):%H:%M}"
    if job["gpu"] is not None:
        busy = [o["id"] for o in jobs if o["state"] == "running" and o["gpu"] == job["gpu"]
                and not (o["shares_gpu"] or job["shares_gpu"])]
        if busy:
            return f"GPU {job['gpu']} busy: " + ", ".join(busy)
    return None


def run_seconds(job, now):
    total = 0.0
    for run in job["runs"]:
        if run["seconds"] is not None:
            total += run["seconds"]
        elif run["ended_at"] is None:
            total += now.timestamp() - run["launched_unix"]
    return total


def reconcile(queue, now):
    """Record every run that ended: exit status from the wrapper, or none if its processes were killed."""
    for job in queue["jobs"]:
        if job["state"] != "running":
            continue
        run = job["runs"][-1]
        path = Path(run["exit_file"])
        status = json.loads(path.read_text()) if path.is_file() else None
        alive = members(run["tag"])
        if status is None and alive:
            continue
        code = None if status is None else status["exit_code"]
        window = {k: (status or {}).get(k) for k in ("window_stop_sent_at", "window_killed_at")}
        interrupted = (run["stop_sent_at"] is not None or run["killed_at"] is not None
                       or any(v is not None for v in window.values()))
        state = "completed" if code == 0 else "paused" if code == EXIT_PAUSED or interrupted else "failed"
        seconds = status["seconds"] if status else (run["killed_unix"] or now.timestamp()) - run["launched_unix"]
        run.update(ended_at=status["ended_at"] if status else stamp(now), exit_code=code, outcome=state,
                   seconds=seconds, **window)
        job["state"] = state
        event(now, "ended", job=job["id"], run=run["run"], exit_code=code, state=state, seconds=round(seconds),
              interrupted=interrupted, stray_pids=alive)


def stop_running(queue, now, reason):
    for job in queue["jobs"]:
        if job["state"] != "running" or job["runs"][-1]["stop_sent_at"] is not None:
            continue
        run = job["runs"][-1]
        if run["pid"] in members(run["tag"]):
            os.kill(run["pid"], signal.SIGTERM)
            run["stop_sent_at"] = stamp(now)
            event(now, "stop", job=job["id"], run=run["run"], pid=run["pid"], signal="SIGTERM", reason=reason)
        else:
            event(now, "stop_skipped", job=job["id"], run=run["run"], reason="wrapper not alive")


def kill_running(queue, now, reason):
    """SIGKILL every surviving process of every run of every job (finished runs included: a tagged descendant that
    outlived its wrapper). killed_* is recorded, and the kill logged as reason, only for a running job's last run;
    processes of a finished run are logged as stray_killed."""
    for job in queue["jobs"]:
        pids = members(job["id"] + "#", exact=False)
        if not pids:
            continue
        sigkill(pids)
        if job["state"] == "running":
            run = job["runs"][-1]
            run["killed_at"], run["killed_unix"] = stamp(now), now.timestamp()
            event(now, reason, job=job["id"], run=run["run"], pids=pids, stop_sent_at=run["stop_sent_at"])
        else:
            event(now, "stray_killed", job=job["id"], state=job["state"], pids=pids, reason=reason)


def log_foreign(queue, now):
    found = gpu_processes()
    if found is None:
        event(now, "nvidia_smi_failed")
        return
    ours = {pid for job in queue["jobs"] if job["state"] == "running" for pid in members(job["runs"][-1]["tag"])}
    foreign = [{"gpu": g, "pid": pid, "mib": mib} for g, pid, mib in found if pid not in ours]
    if foreign:
        event(now, "foreign_gpu_processes", processes=foreign)


def launch_ready(queue, now):
    hold = queue.get("hold_until")
    if hold and now.timestamp() < datetime.fromisoformat(hold).timestamp():
        return
    for job in queue["jobs"]:
        if launch_blocker(job, queue["jobs"], now) is not None:
            continue
        # every recorded run of a launchable job has ended: its surviving processes are strays, killed before relaunch
        strays = sorted({pid for run in job["runs"] for pid in members(run["tag"])})
        if strays:
            sigkill(strays)
            gone = wait_gone(strays, job["id"] + "#")
            event(now, "stray_killed", job=job["id"], state=job["state"], pids=strays, gone=gone,
                  reason="processes of a finished run alive before relaunch")
        orphans = members(job["id"] + "#", exact=False)
        if orphans:
            event(now, "launch_skipped", job=job["id"], pids=orphans,
                  reason="killed strays still alive" if set(orphans) & set(strays)
                  else "processes of an unrecorded run alive")
            continue
        launch(job, now)
        save(queue)


def tick(now):
    with locked(blocking=False) as ok:
        if not ok:
            event(now, "tick_skipped", reason="another tick holds the lock")
            return 0
        queue = load()
        reconcile(queue, now)
        current = phase(now)
        if current == "stop":
            stop_running(queue, now, "window stops at 06:50")
        elif current == "day":
            kill_running(queue, now, "overrun_killed")
        else:
            log_foreign(queue, now)
            launch_ready(queue, now)
        save(queue)
    counts = {s: sum(j["state"] == s for j in queue["jobs"])
              for s in ("running", "queued", "paused", "completed", "failed")}
    print(f"{stamp(now)} tick: phase {current}; jobs {counts}", flush=True)
    return 0


def validate_spec(spec, known):
    missing = {"id", "command", "resumable"} - spec.keys()
    if missing:
        raise ValueError(f"job spec misses {sorted(missing)}: {spec}")
    job = {**SPEC_DEFAULTS, **spec}
    unknown = job.keys() - {*SPEC_DEFAULTS, "id", "command", "resumable", "state"}
    if unknown:
        raise ValueError(f"{job['id']}: unknown fields {sorted(unknown)}")
    if not JOB_ID.fullmatch(job["id"]):
        raise ValueError(f"bad job id {job['id']!r}")
    if not (isinstance(job["command"], list) and job["command"] and all(isinstance(a, str) for a in job["command"])):
        raise ValueError(f"{job['id']}: command must be a non-empty argv list")
    if not job["resumable"] and not job["est_minutes"]:
        raise ValueError(f"{job['id']}: a non-resumable job needs est_minutes")
    if spec.get("state", "queued") != "queued":
        raise ValueError(f"{job['id']}: a new job starts queued")
    for dep in job["depends_on"]:
        if dep not in known:
            raise ValueError(f"{job['id']}: dependency {dep} is not an earlier job")
    return {**job, "state": "queued", "runs": []}


def check_acyclic(jobs):
    deps = {j["id"]: j["depends_on"] for j in jobs}
    done, active = set(), []

    def visit(job_id):
        if job_id in active:
            raise ValueError("dependency cycle: " + " -> ".join([*active[active.index(job_id):], job_id]))
        if job_id not in done:
            active.append(job_id)
            for dep in deps[job_id]:
                visit(dep)
            active.pop()
            done.add(job_id)

    for job_id in deps:
        visit(job_id)


def enqueue(path, requeue, now, update_deps=False):
    """Append the new jobs of a spec file (jobs already queued must match it). With update_deps, a queued or paused
    job whose spec differs only in depends_on takes the spec's dependencies (logged as depends_updated)."""
    with locked(blocking=True):
        queue = load()
        known = {j["id"]: j for j in queue["jobs"]}
        if requeue:
            job = known[requeue]
            if job["state"] not in ("failed", "paused"):
                raise ValueError(f"{requeue} is {job['state']}; only failed or paused jobs are requeued")
            job["state"] = "queued"
            event(now, "requeued", job=requeue)
        else:
            added, updates = [], {}
            for spec in json.loads(Path(path).read_text())["jobs"]:
                job = validate_spec(spec, known)
                if job["id"] in known:
                    old = known[job["id"]]
                    differs = [k for k in (*SPEC_DEFAULTS, "command", "resumable") if old[k] != job[k]]
                    if update_deps and differs == ["depends_on"]:
                        if old["state"] not in ("queued", "paused"):
                            raise ValueError(f"{job['id']} is {old['state']}; dependencies change only on queued or "
                                             "paused jobs")
                        updates[job["id"]] = job["depends_on"]
                    elif differs:
                        raise ValueError(f"{job['id']} is already queued with a different spec ({', '.join(differs)})")
                    continue
                added.append(job)
                known[job["id"]] = job
            jobs = [*queue["jobs"], *added]
            check_acyclic([{**j, "depends_on": updates.get(j["id"], j["depends_on"])} for j in jobs])
            for job in jobs:
                if job["id"] in updates:
                    event(now, "depends_updated", job=job["id"], state=job["state"], old=job["depends_on"],
                          new=updates[job["id"]])
                    job["depends_on"] = updates[job["id"]]
            for job in added:
                event(now, "enqueued", job=job["id"], state=job["state"])
            queue["jobs"] = jobs
        save(queue)
    return 0


def stop_now(now, kill):
    """Operator stop: SIGTERM (or SIGKILL) every running job now; no launch before the next window opens."""
    with locked(blocking=True):
        queue = load()
        reconcile(queue, now)
        if kill:
            kill_running(queue, now, "killed_by_operator")
        else:
            stop_running(queue, now, "stop-now")
        queue["hold_until"] = stamp(next_open(now))
        event(now, "hold", until=queue["hold_until"])
        save(queue)
    return 0


def remaining_minutes(job, now):
    if job["state"] in ("completed", "failed"):
        return 0.0
    if not job["resumable"]:
        return float(job["est_minutes"])
    return max(float(job["est_minutes"] or 0) - run_seconds(job, now) / 60, TICK.total_seconds() / 60)


def project(queue, now, days=21):
    """Projected start / end of every unfinished job under the tick rules, from est_minutes (minutes of work left;
    resumable work carries over nights). Returns {job id: (start, end)}; None for a job that never ends."""
    jobs = [{**j, "runs": [], "left": remaining_minutes(j, now)} for j in queue["jobs"]]
    when = {j["id"]: [None, None] for j in jobs if j["state"] not in ("completed", "failed")}
    t = local(now).replace(second=0, microsecond=0)
    t += timedelta(minutes=-t.minute % 5)
    end = t + timedelta(days=days)
    while t < end and any(j["state"] not in ("completed", "failed") for j in jobs):
        if phase(t) == "run":
            for job in jobs:
                if launch_blocker(job, jobs, t) is None:
                    job["state"] = "running"
                    when[job["id"]][0] = when[job["id"]][0] or t
            for job in jobs:
                if job["state"] == "running":
                    job["left"] -= TICK.total_seconds() / 60
                    if job["left"] <= 0:
                        job["state"] = "completed"
                        when[job["id"]][1] = t + TICK
        else:
            for job in jobs:
                if job["state"] == "running":
                    job["state"] = "paused"
                    if not job["resumable"]:
                        job["left"] = float(job["est_minutes"])
        t += TICK
    return when


def describe(queue, now):
    current = phase(now)
    at = now if current == "run" else next_open(now)
    lines = [f"now {stamp(now)}: phase {current}; window [18:00, 07:00) Australia/Melbourne, graceful stop 06:50; "
             f"next open {stamp(next_open(now))}, next stop {stamp(next_stop(now))}"]
    if queue.get("hold_until"):
        lines.append(f"operator hold until {queue['hold_until']}")
    for job in queue["jobs"]:
        spent = run_seconds(job, now) / 3600
        last = job["runs"][-1] if job["runs"] else None
        blocker = launch_blocker(job, queue["jobs"], at) if job["state"] in ("queued", "paused") else None
        detail = f"run {last['run']} pid {last['pid']}" if job["state"] == "running" else (
            f"last exit {last['exit_code']}" if last else "")
        lines.append(f"  {job['id']:<22} {job['state']:<9} gpu {str(job['gpu']):<4} "
                     f"{'resumable' if job['resumable'] else 'est ' + str(job['est_minutes']) + ' min':<14} "
                     f"runs {len(job['runs'])} ({spent:.2f} h) {detail}"
                     + (f" -- {blocker}" if blocker else (" -- ready" if job['state'] in ('queued', 'paused') else "")))
    return lines


def status(now):
    print("\n".join(describe(load(), now)))
    return 0


def dry_run(now):
    """What a tick would do now (nothing is launched or signalled), then the plan from the next window on."""
    queue = load()
    print("\n".join(describe(queue, now)))
    current = phase(now)
    running = [j for j in queue["jobs"] if j["state"] == "running"]
    if current == "day":
        print(f"tick now: launches nothing (outside the window); would SIGKILL every process of "
              f"{[j['id'] for j in queue['jobs'] if members(j['id'] + '#', exact=False)] or 'nothing'}")
    elif current == "stop":
        print(f"tick now: launches nothing; would SIGTERM {[j['id'] for j in running] or 'nothing'}")
    else:
        ready = [j["id"] for j in queue["jobs"] if launch_blocker(j, queue["jobs"], now) is None]
        print(f"tick now: would launch {ready or 'nothing'}")
    at = now if current == "run" else next_open(now)
    first = [j["id"] for j in queue["jobs"] if launch_blocker(j, queue["jobs"], at) is None]
    print(f"at {stamp(at)}: launches {first or 'nothing'}")
    when = project(queue, now)
    print("projection (est_minutes, 5-min ticks, stops at 06:50):")
    for job in queue["jobs"]:
        if job["id"] in when:
            start, end = when[job["id"]]
            print(f"  {job['id']:<22} start {stamp(start) if start else '-':<25} "
                  f"end {stamp(end) if end else 'not within 21 days'}")
    ends = [end for _, end in when.values()]
    if ends and all(ends):
        print(f"all jobs end by {stamp(max(ends))}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--now", help="ISO time for planning and tests (naive = Australia/Melbourne); default: now")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("tick")
    sub.add_parser("status")
    sub.add_parser("dry-run")
    p = sub.add_parser("enqueue")
    p.add_argument("spec", nargs="?", help='JSON file {"jobs": [...]}')
    p.add_argument("--requeue", metavar="ID", help="return a failed or paused job to the queue")
    p.add_argument("--update-deps", action="store_true",
                   help="queued / paused jobs whose spec differs only in depends_on take the spec's dependencies")
    p = sub.add_parser("stop-now")
    p.add_argument("--kill", action="store_true", help="SIGKILL instead of SIGTERM")
    p = sub.add_parser("_wrap")
    p.add_argument("--stop-at", type=datetime.fromisoformat, help="SIGTERM the command at this time (with offset)")
    p.add_argument("--kill-at", type=datetime.fromisoformat, help="SIGKILL every process of the run at this time")
    p.add_argument("exit_file")
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.mode == "_wrap":
        return wrap(args.exit_file, args.command[1:] if args.command[:1] == ["--"] else args.command,
                    args.stop_at, args.kill_at)
    now = datetime.fromisoformat(args.now) if args.now else datetime.now(ZONE)
    now = now.replace(tzinfo=ZONE) if now.tzinfo is None else now
    if args.mode == "tick":
        return tick(now)
    if args.mode == "status":
        return status(now)
    if args.mode == "dry-run":
        return dry_run(now)
    if args.mode == "enqueue":
        if bool(args.spec) == bool(args.requeue) or (args.update_deps and not args.spec):
            parser.error("enqueue takes a spec file (optionally with --update-deps) or --requeue ID")
        return enqueue(args.spec, args.requeue, now, args.update_deps)
    return stop_now(now, args.kill)


if __name__ == "__main__":
    raise SystemExit(main())
