import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
from typing import Any
import unittest
from unittest.mock import Mock, patch

from scripts import issue_76_display_start as display
from scripts import issue_76_live_episode as live


REAL_POPEN = subprocess.Popen
TWO_WRITE_SERVER = r"""
import os
import sys

display_fd, status_fd, release_fd = map(int, sys.argv[1:])
os.write(display_fd, b"12")
os.write(status_fd, b"D")
os.read(release_fd, 1)
try:
    os.write(display_fd, b"\n")
except BrokenPipeError:
    os.write(status_fd, b"B")
    raise SystemExit(73)
else:
    os.write(status_fd, b"C")
"""


class TwoWriteDisplayServer:
    """Replace Xvnc with a process whose newline write is explicitly released."""

    def __init__(self):
        self.status_reader, self.status_writer = os.pipe()
        self.release_reader, self.release_writer = os.pipe()
        self.process = None

    def popen(self, command: Any, pass_fds: tuple[int, ...], stdout: Any, stderr: Any):
        display_fd = pass_fds[0]
        status_writer = self.status_writer
        release_reader = self.release_reader
        assert status_writer is not None and release_reader is not None
        self.process = REAL_POPEN(
            [sys.executable, "-c", TWO_WRITE_SERVER, str(display_fd),
             str(status_writer), str(release_reader)],
            pass_fds=(display_fd, status_writer, release_reader),
            stdout=stdout, stderr=stderr)
        os.close(status_writer)
        self.status_writer = None
        os.close(release_reader)
        self.release_reader = None
        return self.process

    def await_digits(self):
        return os.read(self.status_reader, 1)

    def release_newline(self):
        release_writer = self.release_writer
        assert release_writer is not None
        os.write(release_writer, b"R")
        os.close(release_writer)
        self.release_writer = None

    def await_outcome(self):
        return os.read(self.status_reader, 1)

    def close(self):
        for name in ("status_reader", "status_writer", "release_reader", "release_writer"):
            descriptor = getattr(self, name)
            if descriptor is not None:
                os.close(descriptor)
                setattr(self, name, None)
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()


class DisplayStartTests(unittest.TestCase):
    def test_old_single_read_breaks_two_write_display_server(self):
        server = TwoWriteDisplayServer()
        try:
            with TemporaryDirectory() as directory, patch(
                    "scripts.issue_76_live_episode.subprocess.Popen", side_effect=server.popen):
                number, process = live.start_display(Path(directory) / "display.log")
            self.assertEqual(number, ":12")
            self.assertEqual(server.await_digits(), b"D")
            server.release_newline()
            self.assertEqual(server.await_outcome(), b"B")
            self.assertEqual(process.wait(timeout=5), 73)
            print("fail-before: old single-read returned :12, then second write raised BrokenPipeError (exit 73)")
        finally:
            server.close()

    def test_new_start_waits_for_two_write_display_server_newline(self):
        server = TwoWriteDisplayServer()
        release_error = []

        def release_after_digits():
            try:
                self.assertEqual(server.await_digits(), b"D")
                server.release_newline()
            except BaseException as error:
                release_error.append(error)

        releaser = threading.Thread(target=release_after_digits)
        try:
            releaser.start()
            with TemporaryDirectory() as directory, patch(
                    "scripts.issue_76_display_start.subprocess.Popen", side_effect=server.popen):
                number, process = display.start_display(Path(directory) / "display.log")
            releaser.join(timeout=5)
            self.assertFalse(releaser.is_alive())
            if release_error:
                raise release_error[0]
            self.assertEqual(number, ":12")
            self.assertEqual(server.await_outcome(), b"C")
            self.assertEqual(process.wait(timeout=5), 0)
        finally:
            server.close()

    def test_split_number_and_newline_are_consumed_before_pipe_close(self):
        process = Mock()
        process.poll.return_value = None
        with TemporaryDirectory() as directory, patch("subprocess.Popen", return_value=process), patch(
                "select.select", return_value=([1], [], [])), patch("os.read", side_effect=[b"12", b"\n"]) as read:
            number, returned = display.start_display(Path(directory) / "display.log")
        self.assertEqual(number, ":12")
        self.assertIs(returned, process)
        self.assertEqual(read.call_count, 2, "closing after digits causes Xvnc's newline write to fail")

    def test_complete_line_needs_one_read(self):
        process = Mock()
        process.poll.return_value = None
        with TemporaryDirectory() as directory, patch("subprocess.Popen", return_value=process), patch(
                "select.select", return_value=([1], [], [])), patch("os.read", return_value=b"12\n") as read:
            number, _ = display.start_display(Path(directory) / "display.log")
        self.assertEqual(number, ":12")
        self.assertEqual(read.call_count, 1)

    def test_eof_before_newline_stops_and_cleans_up(self):
        process = Mock()
        with TemporaryDirectory() as directory, patch("subprocess.Popen", return_value=process), patch(
                "select.select", return_value=([1], [], [])), patch("os.read", side_effect=[b"12", b""]), patch.object(
                display.capture.old.capture, "terminate") as terminate:
            with self.assertRaisesRegex(RuntimeError, "before the newline"):
                display.start_display(Path(directory) / "display.log")
            terminate.assert_called_once_with(process)

    def test_deadline_is_shared_across_partial_reads(self):
        process = Mock()
        with TemporaryDirectory() as directory, patch("subprocess.Popen", return_value=process), patch(
                "select.select", return_value=([1], [], [])) as select, patch("os.read", return_value=b"12"), patch(
                "time.monotonic", side_effect=[100., 114., 115.]), patch.object(
                display.capture.old.capture, "terminate") as terminate:
            with self.assertRaises(TimeoutError):
                display.start_display(Path(directory) / "display.log")
            self.assertEqual(select.call_args.args[-1], 1.)
            terminate.assert_called_once_with(process)
