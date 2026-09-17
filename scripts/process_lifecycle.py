from __future__ import annotations

from functools import partial
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid


REGISTRY_ENV = "NOVPHY_OWNED_PROCESS_GROUP_REGISTRY"
REGISTRY_FAILURE_ENV = "NOVPHY_TEST_FAIL_OWNED_PROCESS_GROUP_REGISTRATION"
WORKER_START_SECONDS = 5


def _stat_fields(pid):
    line = (Path("/proc")/str(pid)/"stat").read_text(encoding="ascii")
    return tuple(line[line.rindex(")")+1:].split())


def process_identity(pid):
    try: return int(_stat_fields(pid)[19])
    except (OSError, ValueError, IndexError): return None


def _append_registry(path,line):
    descriptor = os.open(path,os.O_WRONLY|os.O_APPEND)
    payload = (line+"\n").encode("ascii")
    offset = 0
    try:
        while offset < len(payload):
            written = os.write(descriptor,payload[offset:])
            if written <= 0: raise OSError("process-group registry write made no progress")
            offset += written
    finally: os.close(descriptor)


def registry_snapshot(path):
    groups,pending = {},set()
    for index,line in enumerate(Path(path).read_text(encoding="ascii").splitlines(),1):
        fields = line.split()
        try:
            if fields[0] == "P" and len(fields) == 2: pending.add(fields[1])
            elif fields[0] in ("E","F") and len(fields) == 2: pending.discard(fields[1])
            elif fields[0] == "R" and len(fields) == 5:
                token,pgid,leader,starttime = fields[1],*map(int,fields[2:])
                pending.discard(token); groups[pgid] = (leader,starttime)
            else: raise ValueError
        except (ValueError,IndexError):
            pending.add(f"malformed-record-{index}:{line!r}")
    return groups,pending


def registered_session_popen(command,*,popen=subprocess.Popen,**kwargs):
    environment = kwargs.get("env") or os.environ
    registry = environment.get(REGISTRY_ENV)
    if not registry:
        return popen(command,start_new_session=True,**kwargs)
    token = uuid.uuid4().hex
    _append_registry(registry,f"P {token}")
    wrapped = [sys.executable,str(Path(__file__).resolve()),"--exec-registered",token,*command]
    try: return popen(wrapped,start_new_session=True,**kwargs)
    except BaseException:
        _append_registry(registry,f"F {token}")
        raise


def _group_member_identity(pgid):
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc.name); fields = _stat_fields(pid)
            if fields[0] != "Z" and int(fields[2]) == pgid: return pid,int(fields[19])
        except (OSError,ValueError,IndexError): pass
    return None


def register_process_group(process):
    path = os.environ.get(REGISTRY_ENV)
    if not path: return
    pgid = process.pid
    try:
        member = _group_member_identity(pgid)
        if member is None: raise RuntimeError(f"process group {pgid} exited before registration")
        leader,identity = member
        _append_registry(path,f"R legacy-{pgid}-{identity} {pgid} {leader} {identity}")
    except BaseException:
        try: os.killpg(pgid,signal.SIGKILL)
        except ProcessLookupError: pass
        if process.poll() is None:
            process.kill(); process.wait(timeout=WORKER_START_SECONDS)
        raise


def _exec_registered(token,command):
    registry = os.environ[REGISTRY_ENV]
    try:
        if os.environ.get(REGISTRY_FAILURE_ENV):
            _append_registry(registry,f"E {token}")
            os._exit(126)
        pid = os.getpid(); pgid = os.getpgrp()
        identity = process_identity(pid)
        if identity is None or pgid != pid or os.getsid(0) != pid:
            raise RuntimeError("registered payload did not start in its own session")
        _append_registry(registry,f"R {token} {pgid} {pid} {identity}")
        os.execvp(command[0],command)
    except BaseException:
        os._exit(127)


def _run_isolated_worker(target,registry,*args):
    os.setsid()
    os.environ[REGISTRY_ENV] = registry
    target(*args)


def start_isolated_worker(context,target,args):
    descriptor,registry = tempfile.mkstemp(prefix="novphy-worker-groups-",suffix=".txt")
    os.close(descriptor)
    process = context.Process(target=partial(_run_isolated_worker,target,registry),args=args)
    started = False
    try:
        process.start()
        started = True
        setattr(process,"novphy_process_group",True)
        setattr(process,"novphy_process_group_id",process.pid)
        setattr(process,"novphy_process_group_registry",registry)
        setattr(process,"novphy_process_group_starttime",process_identity(process.pid))
        deadline = time.monotonic()+WORKER_START_SECONDS
        while process.is_alive() and time.monotonic() < deadline:
            try:
                if os.getpgid(process.pid) == process.pid: break
            except ProcessLookupError:
                break
            time.sleep(.01)
        else:
            if process.is_alive(): raise RuntimeError(f"worker {process.pid} did not establish its process group")
        return process
    except BaseException as error:
        if started:
            setattr(process,"novphy_process_group",True)
            setattr(process,"novphy_process_group_id",process.pid)
            setattr(process,"novphy_process_group_registry",registry)
            try:
                from scripts.run_issue_76_compatibility import terminate_worker
                terminate_worker(process)
            except BaseException as cleanup_error:
                error.add_note(f"interrupted worker cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}")
        else:
            Path(registry).unlink(missing_ok=True)
        raise


def cleanup_actions(actions):
    failures = []
    for name,action in actions:
        try: action()
        except BaseException as error: failures.append((name,error))
    return failures


def record_cleanup_failures(result,failures):
    if not failures: return
    result["cleanup_failures"] = [f"{name}: {type(error).__name__}: {error}" for name,error in failures]
    if result.get("failure") is None:
        result["failure"] = result["cleanup_failures"][0]


def persist_after_cleanup(failures,persist):
    persistence_error = None
    try: persist()
    except BaseException as error: persistence_error = error
    if failures:
        if persistence_error is not None:
            failures[0][1].add_note(f"result persistence also failed: {type(persistence_error).__name__}: {persistence_error}")
        raise failures[0][1]
    if persistence_error is not None: raise persistence_error


if __name__ == "__main__" and len(sys.argv) >= 4 and sys.argv[1] == "--exec-registered":
    _exec_registered(sys.argv[2],sys.argv[3:])
