"""Issue-77 campaigns reuse the unchanged R3 paired-input capture library.

This is a renamed port of tests/test_issue_76_bounded_transfer_capture.py: the
capture module under test is shared verbatim by the issue-77 N1/N2 runners.
"""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from scripts import issue_76_bounded_transfer_capture as capture
from world_model.training.native_history_data import VOCABULARY


def fixtures():
    member = {
        "identity": "paired-member-1",
        "base_cluster": "base-1",
        "engine_seed": 764000001,
        "scenario": {"objects": ["pig", "block"], "seed": 764100001},
    }
    history = []
    for ordinal, step in enumerate((29900, 29950, 30000), 1):
        history.append({
            "fixed_step": step,
            "fixed_time_seconds": 12 + (ordinal - 3) * .02,
            "capture_id": "decision-history-capture-1",
            "sequence": ordinal,
            "render_frame": 100 + ordinal,
            "canonical_png": b"corrected-latest" if ordinal == 3 else f"history-{ordinal}".encode(),
        })
    legacy = SimpleNamespace(png=b"legacy-hud", state={
        "fixed_step": 30000,
        "fixed_time": 12,
        "capture_id": "capture-1",
        "sequence": 8,
        "render_frame": 108,
    })
    corrected = SimpleNamespace(canonical_png=b"corrected-latest", metadata={
        "fixed_step": 30000,
        "fixed_time_seconds": 12,
        "capture_id": "capture-1",
        "sequence": 9,
        "render_frame": 109,
    })
    return member, legacy, corrected, history


def parser_output(frames):
    slots = len(VOCABULARY)
    return {
        "presence_logits": torch.ones(frames, slots),
        "centers": torch.arange(frames, dtype=torch.float32)[:, None, None].expand(frames, slots, 2).clone(),
        "kind_logits": torch.zeros(frames, slots, 7),
    }


class Issue77N1CaptureTests(unittest.TestCase):
    def test_pairing_metadata_binds_three_views_to_one_state_and_scenario(self):
        member, legacy, corrected, history = fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            manifest = capture.record_paired_inputs(
                Path(temporary) / "paired", member, 30000, legacy, corrected, history)
            self.assertTrue(manifest["same_native_decision_state"])
            self.assertFalse(manifest["post_action_frames_used"])
            self.assertEqual(manifest["decision_state_binding"]["runtime_capture_id"], "capture-1")
            self.assertEqual(manifest["scenario_binding"]["member_identity"], member["identity"])
            self.assertEqual(list(manifest["views"]),
                             ["legacy-single", "corrected-single", "corrected-history"])
            self.assertEqual([len(view["frames"]) for view in manifest["views"].values()], [1, 1, 3])
            self.assertTrue((Path(temporary) / "paired" / capture.PAIR_MANIFEST).is_file())

    def test_single_frame_views_have_zero_prior_elapsed_and_motion_channels(self):
        for view in capture.SINGLE_VIEWS:
            carriers = capture.assemble_visual_carriers(view, parser_output(1), [12.0])
            self.assertEqual(tuple(carriers.shape), (1, 288))
            self.assertTrue(torch.equal(carriers[:, :2], torch.zeros_like(carriers[:, :2])))
            slots = carriers[:, 2:].reshape(1, len(VOCABULARY), 13)
            self.assertTrue(torch.equal(slots[:, :, 8:11], torch.zeros_like(slots[:, :, 8:11])))
        with self.assertRaisesRegex(ValueError, "exactly one actual frame"):
            capture.assemble_visual_carriers("legacy-single", parser_output(3), [11.96, 11.98, 12.0])

    def test_corrected_history_requires_actual_minus_100_minus_50_zero_steps(self):
        member, legacy, corrected, history = fixtures()
        manifest = capture.build_paired_input_manifest(member, 30000, legacy, corrected, history)
        frames = manifest["views"]["corrected-history"]["frames"]
        self.assertEqual([frame["relative_native_step"] for frame in frames], [-100, -50, 0])
        history[1]["fixed_step"] = 29951
        with self.assertRaisesRegex(ValueError, "-100/-50/0"):
            capture.build_paired_input_manifest(member, 30000, legacy, corrected, history)

    def test_post_action_frame_is_rejected_as_predecision_input(self):
        member, legacy, corrected, history = fixtures()
        corrected.metadata["fixed_step"] = 30001
        with self.assertRaisesRegex(ValueError, "post-action frame"):
            capture.build_paired_input_manifest(member, 30000, legacy, corrected, history)


if __name__ == "__main__":
    unittest.main()
