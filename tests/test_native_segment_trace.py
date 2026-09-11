import gzip
import json
from pathlib import Path
import tempfile
import unittest

from scripts.canonical_native_trace import NativeTrace
from scripts.native_segment_trace import NativeSegmentTrace
from tests.test_canonical_native_trace import sample


def segment_fixture(root, *, censored=True, end=30000, failure="native_time_window_limit"):
    (root / "native").mkdir()
    descriptors = []
    for ordinal, start in enumerate(range(0, end + 1, 250), 1):
        stop = min(start + 250, end + 1)
        events = []
        if start == 0:
            events.append({"event_id": "launch", "event_type": "bird_launched", "fixed_step": 1,
                           "participants": [], "payload": {}})
        if stop == end + 1 and not censored:
            events.append({"event_id": "terminal", "event_type": "stable_entered", "fixed_step": end,
                           "participants": [], "payload": {}})
        value = {"schema": "canonical_native_chunk_v1", "capture_id": "capture", "shot_id": "shot",
                 "fixed_step_samples": [sample(s) for s in range(start, stop)], "events": events}
        encoded = json.dumps(value).encode()
        path = f"native/chunk-{ordinal:06}.json.gz"
        (root / path).write_bytes(gzip.compress(encoded))
        descriptors.append({"path": path, "first_fixed_step": start, "last_fixed_step": stop - 1,
                            "sample_count": stop - start, "event_count": len(events),
                            "uncompressed_bytes": len(encoded), "compressed_bytes": (root / path).stat().st_size})
    steps = list(range(0, end + 1, 50))
    if steps[-1] != end:
        steps.append(end)
    value = {"schema": "canonical_native_trace_v1", "capture_id": "capture", "shot_id": "shot",
             "status": "failed" if censored else "complete", "failure": failure if censored else None,
             "fixed_delta_seconds": .0004, "observation_stride": 50, "engine_seed": 1,
             "first_fixed_step": 0, "last_fixed_step": end, "sample_count": end + 1,
             "complete_every_native_step": True, "chunks": descriptors,
             "frame_records": [{"fixed_step": s, "forced_terminal": s % 50 != 0} for s in steps],
             "terminal_evidence": None if censored else {"reason": "stable_entered", "fixed_step": end, "event_id": "terminal"}}
    (root / "native-manifest.json").write_text(json.dumps(value))


class NativeSegmentTests(unittest.TestCase):
    def test_intact_window_is_censored_without_changing_raw_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segment_fixture(root)
            before = (root / "native-manifest.json").read_bytes()
            trace = NativeSegmentTrace(root)
            self.assertTrue(trace.censored)
            self.assertFalse(trace.summary["complete"])
            self.assertFalse(trace.summary["terminal_observed"])
            self.assertIsNone(trace.manifest["terminal_evidence"])
            self.assertEqual(len(list(trace.observed_samples())), 601)
            self.assertEqual(len(trace.endpoint_indices(750)), 586)
            self.assertEqual(trace.endpoint_indices(1), [])
            self.assertEqual((root / "native-manifest.json").read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "not complete training"):
                list(NativeTrace(root).observed_samples())

    def test_genuine_terminal_only_has_exact_observed_endpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segment_fixture(root, censored=False, end=51)
            trace = NativeSegmentTrace(root)
            self.assertFalse(trace.censored)
            self.assertTrue(trace.summary["terminal_observed"])
            self.assertEqual(trace.endpoint_indices(50), [(0, 1)])
            self.assertEqual(trace.endpoint_indices(1), [(1, 2)])
            self.assertEqual(trace.endpoint_indices(250), [])

    def test_short_or_other_failed_prefix_is_not_censoring(self):
        for kwargs in ({"end": 51}, {"end": 51, "failure": "capture_error"}):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                segment_fixture(root, **kwargs)
                with self.assertRaisesRegex(ValueError, "intact full native"):
                    NativeSegmentTrace(root)

    def test_missing_chunk_is_not_replaced_with_a_censored_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segment_fixture(root, end=51)
            (root / "native/chunk-000001.json.gz").unlink()
            with self.assertRaises((FileNotFoundError, ValueError)):
                NativeSegmentTrace(root)


if __name__ == "__main__":
    unittest.main()
