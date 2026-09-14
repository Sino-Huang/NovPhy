from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from scripts import issue_76_display_start as display


class DisplayStartTests(unittest.TestCase):
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
