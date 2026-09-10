import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import issue_76_native_media as media
from scripts.issue_76_native_findings import moving_bodies


class NativePublicationTests(unittest.TestCase):
    def test_moving_body_diagnostic_uses_frozen_threshold_and_active_lifecycle(self):
        body = {"body_type": "dynamic", "velocity": [.01, 0],
                "angular_velocity_degrees_per_second": .01}
        entity = {"entity_id": "bird", "lifecycle": "active", "body": body}
        self.assertEqual(moving_bodies({"entities": [entity]}), [])
        body["angular_velocity_degrees_per_second"] = .02
        self.assertEqual(moving_bodies({"entities": [entity]})[0]["entity_id"], "bird")
        entity["lifecycle"] = "destroyed"
        self.assertEqual(moving_bodies({"entities": [entity]}), [])

    def test_running_episode_is_excluded_but_finished_failure_is_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c2, native = root / "c2", root / "native"
            member = {"identity": "test", "novelty_level": 0}
            for output, members in ((c2, [member]), (native, [])):
                output.mkdir()
                (output / "plan.json").write_text(json.dumps({"identity": output.name, "members": members}))
            frames = c2 / "attempts/test/aligned/capture-v2:fixture"
            frames.mkdir(parents=True)
            (frames / "frame_000001.json").write_text(json.dumps({"fixed_step": 50, "fixed_time_seconds": .02}))
            (frames / "frame_000001.png").write_bytes(b"fixture; not a rendered research observation")
            with patch.object(media.c2, "OUTPUT", c2), patch.object(media.native, "OUTPUT", native):
                _, value = media.inventory()
                self.assertEqual(value["entries"], [])
                result = media.c2.result_path(c2, member)
                result.parent.mkdir(parents=True)
                result.write_text(json.dumps({"failure": "fixture_timeout"}))
                target, value = media.inventory()
                self.assertEqual(target.name, "completed-episodes-1")
                self.assertEqual(value["entries"][0]["failure"], "fixture_timeout")
                self.assertEqual(len(value["entries"][0]["frames"]), 1)
                self.assertEqual(value["synthetic_frames"], 0)
                self.assertEqual(media.inventory(), (target, value))


if __name__ == "__main__":
    unittest.main()
