import gzip
import json
from pathlib import Path
import tempfile
import unittest

from scripts.canonical_native_trace import NativeTrace


def sample(step):
    return {"fixed_step": step, "complete_raw_non_trigger_contacts": True,
            "world": {"world_id": "unity-physics2d", "gravity_vector": [0, -9.81]},
            "entities": [{"entity_id": "runtime:block:0000", "scenario_object_id": "block:0000",
                          "lifecycle": "active", "body_present": False, "body": None}],
            "colliders": [{"collider_id": "collider:0", "entity_id": "runtime:block:0000"}],
            "contacts": [], "supports": []}


def fixture(root, *, gap=False, duplicate=False, fake_terminal=False, failed=False):
    (root / "native").mkdir()
    chunks = []
    event = {"event_id": "end", "event_type": "stable_entered", "fixed_step": 51,
             "participants": [], "payload": {}}
    for ordinal, steps in enumerate((range(26), range(27 if gap else 26, 52)), 1):
        events = [event] if ordinal == 2 or duplicate else []
        data = {"schema": "canonical_native_chunk_v1", "capture_id": "capture", "shot_id": "shot",
                "fixed_step_samples": [sample(s) for s in steps], "events": events}
        if fake_terminal:
            data["terminal_evidence"] = {"reason": "file_boundary"}
        encoded = json.dumps(data).encode()
        path = f"native/chunk-{ordinal:06}.json.gz"
        with gzip.open(root / path, "wb") as stream:
            stream.write(encoded)
        chunks.append({"path": path, "first_fixed_step": data["fixed_step_samples"][0]["fixed_step"],
                       "last_fixed_step": data["fixed_step_samples"][-1]["fixed_step"],
                       "sample_count": len(data["fixed_step_samples"]), "event_count": len(events),
                       "uncompressed_bytes": len(encoded), "compressed_bytes": (root / path).stat().st_size})
    manifest = {"schema": "canonical_native_trace_v1", "capture_id": "capture", "shot_id": "shot",
                "status": "failed" if failed else "complete", "failure": "limit" if failed else None,
                "fixed_delta_seconds": .0004, "observation_stride": 50,
                "first_fixed_step": 0, "last_fixed_step": 51, "sample_count": 52,
                "complete_every_native_step": True, "chunks": chunks,
                "frame_records": [{"fixed_step": i, "forced_terminal": i == 51} for i in (0, 50, 51)],
                "terminal_evidence": None if failed else {"reason": "stable_entered", "fixed_step": 51, "event_id": "end"}}
    (root / "native-manifest.json").write_text(json.dumps(manifest))


class NativeTraceTests(unittest.TestCase):
    def test_complete_microsteps_and_off_grid_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            trace = NativeTrace(root)
            report = trace.validate()
            self.assertEqual(report["sample_count"], 52)
            self.assertAlmostEqual(report["physical_seconds"], .0204)
            self.assertEqual([s["fixed_step"] for s in trace.observed_samples()], [0, 50, 51])

    def test_gap_is_not_hidden_by_chunk_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture(Path(temp), gap=True)
            with self.assertRaisesRegex(ValueError, "gap or overlap"):
                NativeTrace(temp).validate()

    def test_duplicate_events_and_fake_file_terminals_fail(self):
        for kwargs, expected in (({"duplicate": True}, "duplicated"), ({"fake_terminal": True}, "file boundary")):
            with tempfile.TemporaryDirectory() as temp:
                fixture(Path(temp), **kwargs)
                with self.assertRaisesRegex(ValueError, expected):
                    NativeTrace(temp).validate()

    def test_failure_prefix_is_retained_but_not_a_complete_training_example(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture(Path(temp), failed=True)
            trace = NativeTrace(temp)
            self.assertFalse(trace.validate()["complete"])
            with self.assertRaisesRegex(ValueError, "not complete training"):
                list(trace.observed_samples())

    def test_wrong_physics_time_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            path = root / "native-manifest.json"
            value = json.loads(path.read_text())
            value["fixed_delta_seconds"] = .02
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "native time"):
                NativeTrace(root)

    def test_truncated_compressed_chunk_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            path = root / "native/chunk-000001.json.gz"
            path.write_bytes(path.read_bytes()[:-5])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                NativeTrace(root).validate()


if __name__ == "__main__":
    unittest.main()
