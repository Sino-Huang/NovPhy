from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.publish_issue_76_angular_media import retained_video


class RetainedMediaTests(unittest.TestCase):
    def test_verified_video_is_not_reencoded(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "case.webm"
            video.write_bytes(b"retained")
            probe = subprocess.CompletedProcess([], 0,
                '{"streams":[{"nb_read_frames":"2","width":640,"height":480}]}', "")
            with patch("scripts.publish_issue_76_angular_media.subprocess.run", return_value=probe), patch(
                    "scripts.publish_issue_76_angular_media.encode_video") as encode:
                self.assertEqual(retained_video([Path("a"), Path("b")], video), "retained_verified_export")
            encode.assert_not_called()
            self.assertEqual(video.read_bytes(), b"retained")

    def test_incomplete_video_is_preserved_before_reencoding(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "case.webm"
            video.write_bytes(b"partial")
            probe = subprocess.CompletedProcess([], 0,
                '{"streams":[{"nb_read_frames":"1","width":640,"height":480}]}', "")
            with patch("scripts.publish_issue_76_angular_media.subprocess.run", return_value=probe), patch(
                    "scripts.publish_issue_76_angular_media.encode_video") as encode:
                self.assertEqual(retained_video([Path("a"), Path("b")], video), "encoded_from_retained_frames")
            encode.assert_called_once()
            self.assertEqual(video.with_suffix(".interrupted.webm").read_bytes(), b"partial")
