"""Prospective display-start correction; not installed in frozen collectors."""
import os
from pathlib import Path
import select
import subprocess
import time

from scripts.issue_76_live_episode import capture


def start_display(log_path):
    """Keep the pipe open until Xvnc finishes its newline-terminated reply."""
    reader, writer = os.pipe()
    process = None
    deadline = time.monotonic() + 15
    try:
        with Path(log_path).open("ab") as log:
            process = subprocess.Popen(
                ["Xvnc", "-displayfd", str(writer), "-geometry", "1024x768", "-depth", "24",
                 "-SecurityTypes", "None", "-rfbport", "0", "-localhost"],
                pass_fds=(writer,), stdout=log, stderr=subprocess.STDOUT)
        os.close(writer)
        writer = None
        reply = b""
        while not reply.endswith(b"\n"):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([reader], [], [], remaining)[0]:
                raise TimeoutError("Xvnc did not allocate a display within 15 seconds")
            chunk = os.read(reader, 32)
            if not chunk:
                raise RuntimeError("Xvnc closed its display pipe before the newline; see display.log")
            reply += chunk
        number = reply.decode().strip()
        if not number.isdecimal() or process.poll() is not None:
            raise RuntimeError("Xvnc failed to reserve a display; see display.log")
        return ":" + number, process
    except Exception:
        if process is not None:
            capture.old.capture.terminate(process)
        raise
    finally:
        os.close(reader)
        if writer is not None:
            os.close(writer)
