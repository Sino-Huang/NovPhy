"""Focused tests for issue-83 CLEVRER non-monotonicity probe (fast, CPU, synthetic)."""
import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from scripts import run_clevrer_nonmonotonicity_probe as runner
from world_model.model import Abstraction, PredictionPair
from world_model.training import clevrer_boundary as clevrer
from world_model.training import clevrer_nonmonotonicity as probe
from world_model.training.cnn_hybrid import DIM, SLOTS

PAIR = PredictionPair(5, Abstraction.CONTINUOUS)


def annotation_payload(scene_index, collisions, *, objects=2, visible_from=0):
    """One synthetic CLEVRER annotation dict: constant velocity, full trajectories."""
    frames = []
    for time in range(clevrer.FRAMES_PER_CLIP):
        entries = []
        for index in range(objects):
            entries.append({"object_id": index,
                            "location": [index * 1.0 + time * .04, -index * 1.0, .2],
                            "orientation": [0., 0., .1 * time],
                            "velocity": [1.0, -0.5, 0.], "angular_velocity": [0., 0., .01],
                            "inside_camera_view": time >= visible_from})
        frames.append({"frame_id": time, "objects": entries})
    return {"scene_index": scene_index, "video_filename": f"video_{scene_index}.mp4",
            "object_property": [{"object_id": i, "color": "red", "material": "rubber",
                                 "shape": clevrer.KIND_VOCABULARY[i % 3]} for i in range(objects)],
            "motion_trajectory": frames,
            "collision": [{"frame_id": time, "object_ids": [0, 1], "location": [0., 0., .2]}
                          for time in collisions]}


class DriftStub:
    """Stub carrier dynamics: carrier_{k+1} = carrier_k + delta (records call count)."""

    def __init__(self, poison_after=None):
        self.calls = 0
        self.poison_after = poison_after

    def carrier(self, z, action, pair):
        self.calls += 1
        if self.poison_after is not None and self.calls > self.poison_after:
            return torch.full_like(z, float("nan"))
        return z + float(pair.delta) * .001


class Issue83NonmonotonicityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    # ---------------------------------------------------------------- segmentation

    def test_cascade_bounds_and_segments(self):
        frames = torch.tensor([30, 70, 30])
        self.assertEqual(probe.cascade_bounds(frames), (30, 70))
        bounds = (30, 70)
        self.assertEqual(probe.segment_of_frame(bounds, 29), "pre_first_collision")
        self.assertEqual(probe.segment_of_frame(bounds, 30), "active_cascade")
        self.assertEqual(probe.segment_of_frame(bounds, 70), "active_cascade")
        self.assertEqual(probe.segment_of_frame(bounds, 71), "post_settlement")
        self.assertEqual(probe.segment_timeline(bounds, 60, [0, 10, 20]),
                         ["active_cascade", "active_cascade", "post_settlement"])
        with self.assertRaises(probe.ClevrerNonmonotonicityError):
            probe.cascade_bounds(torch.tensor([], dtype=torch.long))
        with self.assertRaises(probe.ClevrerNonmonotonicityError):
            probe.cascade_bounds(torch.tensor([200]))

    def test_segment_census_counts(self):
        bounds = {1: (30, 70), 2: (10, 50)}
        units = [(1, context) for context in range(3)] + [(2, context) for context in range(3)]
        census = probe.segment_census(bounds, units, 60)
        self.assertEqual(census["units"], 6)
        self.assertEqual(census["active_cascade"], 3)   # scene 1: frames 60..62 inside (30, 70]
        self.assertEqual(census["post_settlement"], 3)  # scene 2: frames 60..62 after last event 50
        self.assertEqual(census["pre_first_collision"], 0)
        early = probe.segment_census(bounds, units, 5)
        self.assertEqual(early["pre_first_collision"], 6)

    # ---------------------------------------------------------------- error traces

    def test_batched_masked_errors_match_field_errors(self):
        torch.manual_seed(3)
        prediction = torch.randn(2, DIM) * .1
        target = torch.randn(2, DIM) * .1
        target[1, 2 + 0 * 13 + 7] = 0.  # object 0 of row 1 not in camera view
        target[1, 2 + 0 * 13 + 5] = 0.
        target[1, 2 + 0 * 13 + 6] = 0.
        batch = probe.batched_masked_errors(prediction, target)
        for row in range(2):
            single = clevrer.field_errors(prediction[row], target[row])
            self.assertAlmostEqual(float(batch["carrier_mse"][row]), single["carrier_mse"], places=12)
            self.assertAlmostEqual(float(batch["presence_mse"][row]), single["presence_mse"], places=12)
            self.assertEqual(int(batch["position_available"][row]), single["position_available_values"])
            if single["position_mse"] is not None:
                self.assertAlmostEqual(float(batch["position_mse"][row]), single["position_mse"], places=12)
        self.assertTrue(bool(batch["finite"].all()))

    def test_batched_masked_errors_nan_when_nothing_available(self):
        target = torch.zeros(1, DIM)
        target[0, 2:2 + 13] = 0.  # declared but never visible
        target[0, 0], target[0, 1] = 1., 1.  # global entries
        prediction = torch.randn(1, DIM) * .1
        batch = probe.batched_masked_errors(prediction, target)
        self.assertTrue(torch.isnan(batch["position_mse"]).all())
        self.assertEqual(int(batch["position_available"][0]), 0)

    def test_all_frames_errors_use_fixed_declared_denominator(self):
        prediction = torch.zeros(1, DIM)
        raw = torch.zeros(1, SLOTS, 2)
        declared = torch.zeros(1, SLOTS, dtype=torch.bool)
        declared[0, :1] = True  # one declared object
        raw[0, 0] = torch.tensor([.5, -1.])
        prediction[0, 2 + 5] = .25   # object 0 position x
        prediction[0, 2 + 6] = -.5   # object 0 position y
        prediction[0, 2 + 1 * 13 + 5] = 9.  # undeclared slot must be ignored
        position, velocity = probe.batched_all_frames_errors(prediction, raw, declared, torch.zeros(1, SLOTS, 2))
        expected = ((.25 - .5) ** 2 + (-.5 + 1.) ** 2) / 2
        self.assertAlmostEqual(float(position[0]), expected, places=12)
        self.assertEqual(float(velocity[0]), 0.)

    # ---------------------------------------------------------------- rollout

    def test_trace_rollout_recursive_no_resets(self):
        torch.manual_seed(5)
        initial = torch.randn(3, DIM) * .1
        targets = torch.randn(3, 4, DIM) * .1
        stub = DriftStub()
        trace = probe.trace_rollout(stub, PAIR, initial, targets, torch.zeros(3, 5))
        self.assertEqual(stub.calls, 4)
        self.assertEqual(len(trace["rows"]), 4)
        self.assertTrue(torch.allclose(trace["carrier"], initial + 4 * float(PAIR.delta) * .001))
        for step, row in enumerate(trace["rows"]):
            self.assertTrue(bool(row["finite"].all()))

    def test_trace_rollout_divergence_poisons_row(self):
        initial = torch.randn(2, DIM) * .1
        targets = torch.randn(2, 5, DIM) * .1
        stub = DriftStub(poison_after=2)
        trace = probe.trace_rollout(stub, PAIR, initial, targets, torch.zeros(2, 5))
        self.assertEqual(len(trace["rows"]), 5)
        for step, row in enumerate(trace["rows"]):
            expected = step < 2
            self.assertEqual(bool(row["finite"].all()), expected, f"step {step}")
            if step >= 2:
                self.assertTrue(torch.isnan(row["position_mse"]).all())

    def test_trace_rollout_divergence_is_per_row(self):
        """A diverging cell must not stop the rollout or poison healthy rows (reviewer finding 1)."""

        class RowPoisonStub:
            def __init__(self, row, after):
                self.row, self.after, self.calls = row, after, 0

            def carrier(self, z, action, pair):
                self.calls += 1
                out = z + float(pair.delta) * .001
                if self.calls > self.after:
                    out[self.row] = float("nan")
                return out

        torch.manual_seed(7)
        initial = torch.randn(3, DIM) * .1
        targets = torch.randn(3, 4, DIM) * .1
        targets[:, :, 0] = 1.
        for slot in range(2):  # declared, in-camera slots so errors stay available
            base = 2 + slot * 13
            targets[:, :, base + 7] = 1.
            targets[:, :, base + 10] = 1.
        stub = RowPoisonStub(row=1, after=1)
        trace = probe.trace_rollout(stub, PAIR, initial, targets, torch.zeros(3, 5))
        self.assertEqual(stub.calls, 4)  # the model keeps stepping for the healthy rows
        for step, row in enumerate(trace["rows"]):
            self.assertTrue(bool(row["finite"][0]), f"healthy row 0 poisoned at step {step}")
            self.assertTrue(bool(row["finite"][2]), f"healthy row 2 poisoned at step {step}")
            self.assertEqual(bool(row["finite"][1]), step < 1)
        self.assertTrue(torch.isnan(trace["rows"][-1]["position_mse"][1]))
        self.assertFalse(bool(torch.isnan(trace["rows"][-1]["position_mse"][0])))

    def test_trace_rollout_shape_contracts(self):
        initial = torch.randn(2, DIM)
        targets = torch.randn(2, 2, DIM)
        with self.assertRaises(probe.ClevrerNonmonotonicityError):
            probe.trace_rollout(DriftStub(), PAIR, initial, torch.randn(3, 2, DIM), torch.zeros(2, 5))
        with self.assertRaises(probe.ClevrerNonmonotonicityError):
            probe.trace_rollout(DriftStub(), PAIR, initial, targets, torch.zeros(2, 4))

    # ---------------------------------------------------------------- statistics

    def test_ratio_statistic_known_values(self):
        values = np.array([1.5, 2.5] * 8 + [0.5, 1.5] * 16)  # post mean 2.0, pre/active mean 1.0
        labels = np.array([True] * 16 + [False] * 32)
        result = probe.ratio_statistic(values, labels)
        self.assertAlmostEqual(result["ratio"], 2.0, places=12)
        self.assertAlmostEqual(result["post_mean"], 2.0, places=12)
        self.assertAlmostEqual(result["pre_active_mean"], 1.0, places=12)
        self.assertEqual(result["post_units"], 16)
        self.assertEqual(result["pre_active_units"], 32)
        self.assertTrue(result["interval_reliable"])
        low, high = result["descriptive_95_percent_interval"]
        self.assertLess(low, 2.0)
        self.assertGreater(high, 2.0)
        self.assertGreater(low, 1.4)
        self.assertLess(high, 2.8)

    def test_ratio_statistic_empty_stratum_and_contract(self):
        empty = probe.ratio_statistic(np.array([1., 2.]), np.array([True, True]))
        self.assertIsNone(empty["ratio"])
        self.assertFalse(empty["interval_reliable"])
        with self.assertRaises(probe.ClevrerNonmonotonicityError):
            probe.ratio_statistic(np.array([1., 2.]), np.array([True]))

    def test_monotonicity_predicate(self):
        self.assertEqual(probe.monotonicity({"15": 1., "30": 2., "60": 3.})["monotone_increasing"], True)
        dip = probe.monotonicity({"15": 1., "30": 2., "60": 1.5})
        self.assertEqual(dip["monotone_increasing"], False)
        self.assertEqual(dip["first_violation"], [30, 60])
        missing = probe.monotonicity({"15": 1., "30": None, "60": 3.})
        self.assertIsNone(missing["monotone_increasing"])

    # ---------------------------------------------------------------- CLI

    def test_cli_help_resolves_without_data_access(self):
        with patch("sys.argv", ["runner", "--help"]), self.assertRaises(SystemExit) as ended:
            runner.main()
        self.assertEqual(ended.exception.code, 0)

    def test_modes_are_mutually_exclusive(self):
        with patch("sys.argv", ["runner", "--dry-run", "--publish"]), \
                self.assertRaises(SystemExit) as ended:
            runner.main()
        self.assertEqual(ended.exception.code, 2)

    def test_run_evaluation_requires_cuda_flag(self):
        with self.assertRaises(ValueError):
            runner.run_evaluation(argparse.Namespace(device="cpu"), {})

    # ---------------------------------------------------------------- synthetic source pipeline

    @classmethod
    def _write_source(cls, root, scene_collisions):
        """Synthetic issue-79 artifact release: plan.json, shards, and the validation zip."""
        source = root / "issue-79"
        (source / "shards").mkdir(parents=True)
        payloads = {}
        for offset, collisions in enumerate(scene_collisions):
            scene_index = 10000 + offset
            payload = annotation_payload(scene_index, collisions)
            payloads[scene_index] = payload
            scene = clevrer.parse_scene(payload)
            clip = clevrer.clip_carriers(scene, 3.0)
            relations, mask = clevrer.clip_relations(scene, [0.35, 1.68])
            windows = clevrer.training_windows(clip, relations, mask)
            torch.save({"schema": "issue_79_clevrer_scene_v1", "identity": clevrer.IDENTITY,
                        "scene_index": scene_index, "video_filename": scene.video_filename,
                        "velocity_scale": 3.0, "speed_tier_edges": [0.35, 1.68],
                        "objects": [dict(obj) for obj in scene.objects], "clip": clip, "windows": windows,
                        "collision_frames": scene.collision_frames, "collision_pairs": scene.collision_pairs},
                       source / "shards" / f"scene-{scene_index}.pt")
        zip_path = root / "annotation_validation.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            for scene_index, payload in payloads.items():
                archive.writestr(f"annotation_10000-11000/annotation_{scene_index}.json",
                                 json.dumps(payload))
        plan = {"schema": runner.SOURCE_SCHEMA, "identity": runner.SOURCE_IDENTITY,
                "source_revision": runner.SOURCE_REVISION,
                "representation": {"velocity_scale": 3.0, "speed_tier_edges": [0.35, 1.68],
                                   "availability_mask": "inside_camera_view in columns 7 and 10"},
                "membership": {"scenes": [
                    {"scene_index": 10000 + offset, "objects": 2, "collisions": len(collisions),
                     "video_filename": f"video_{10000 + offset}.mp4"}
                    for offset, collisions in enumerate(scene_collisions)],
                    "windows_per_scene": clevrer.WINDOWS_PER_SCENE},
                "design": {"training": {"steps": 9000}}}
        (source / "plan.json").write_text(json.dumps(plan))
        (source / "summary.json").write_text(json.dumps({
            "dispositions": {"question_1_detail": {"components": {"S3": {"curves": {
                str(horizon): {"15": .001, "30": .002, "60": .0015, "120": .01}
                for horizon in (1, 5, 15)}}}}}}))
        raw = zip_path.read_bytes()
        return source, zip_path, {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def pipeline_overrides(output, scene_collisions, horizons=(15,)):
    """Context manager bundle: synthetic source, reduced schedule, stub checkpoints."""
    class Bundle:
        def __init__(self):
            self._stack = ExitStack()
            self.tmp = Path(self._stack.enter_context(tempfile.TemporaryDirectory()))
            root = self.tmp / "source-area"
            root.mkdir()
            self.source, self.zip_path, self.zip_facts = Issue83NonmonotonicityTests._write_source(
                root, scene_collisions)
            self.output = output if output is not None else self.tmp / "out"

        def __enter__(self):
            stack = self._stack
            from scripts import run_clevrer_boundary_replication as source79
            stack.enter_context(patch.object(runner, "SOURCE_DIR", self.source))
            stack.enter_context(patch.object(runner, "VALIDATION_ZIP", self.zip_path))
            stack.enter_context(patch.object(runner, "ZIP_FACTS", self.zip_facts))
            stack.enter_context(patch.object(runner, "SEEDS", (20260908,)))
            stack.enter_context(patch.object(runner, "HORIZONS", horizons))
            stack.enter_context(patch.object(runner, "load_frozen_predictor",
                                             lambda args, plan, seed, arm: (DriftStub(), None)))
            stack.enter_context(patch.object(runner.source79, "active_capacity_counts",
                                             lambda model, pair, device: 4321))
            stack.enter_context(patch.object(runner, "linear_macs", lambda model, pair: 1234))
            return self

        def __exit__(self, *args):
            self._stack.close()
            return False

    return Bundle()


PIPELINE_SCENES = [(30, 70), (10, 50), (20, 80), (40, 55), (25, 65), (35, 45),
                   (30, 70), (10, 50), (20, 80), (40, 55), (25, 65), (35, 45)]


class Issue83PipelineTests(unittest.TestCase):
    """End-to-end prepare -> evaluate -> publish -> validate on a synthetic release."""

    def test_prepare_freezes_census_and_is_idempotent(self):
        with pipeline_overrides(None, PIPELINE_SCENES) as bundle:
            args = argparse.Namespace(output=bundle.output, device="cpu")
            runner.prepare(args)
            plan = runner.load_plan(args)
            self.assertEqual(plan["identity"], runner.IDENTITY)
            self.assertEqual(plan["design"]["cells"]["total"], (12 * 40 + 12 * 4) * 1 * 2 * 1)
            census = {entry["endpoint"]: entry for entry in plan["segmentation"]["grid_census"]}
            # 6 scenes stay active at frames 60..63, 6 scenes settled; 4 contexts each
            self.assertEqual(census[60]["active_cascade"], 24)
            self.assertEqual(census[60]["post_settlement"], 24)
            self.assertEqual(census[120]["post_settlement"], 48)
            window_census = plan["segmentation"]["window_census_at_relative_60"]
            self.assertEqual(window_census["units"], 480)
            self.assertEqual(window_census["pre_first_collision"], 0)
            self.assertEqual(window_census["active_cascade"], 76)
            self.assertEqual(window_census["post_settlement"], 404)
            before = (bundle.output / "plan.json").read_bytes()
            runner.prepare(args)  # no re-freeze
            self.assertEqual((bundle.output / "plan.json").read_bytes(), before)
            self.assertTrue((bundle.output / "scene-raw-targets.pt").exists())

    def test_prepare_rejects_mutated_source(self):
        with pipeline_overrides(None, PIPELINE_SCENES) as bundle:
            args = argparse.Namespace(output=bundle.output, device="cpu")
            runner.prepare(args)
            plan_path = bundle.source / "plan.json"
            plan = json.loads(plan_path.read_text())
            plan["representation"]["velocity_scale"] = 4.0
            plan_path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                runner.load_plan(args)

    def test_evaluate_publish_validate_round_trip(self):
        with pipeline_overrides(None, PIPELINE_SCENES) as bundle:
            args = argparse.Namespace(output=bundle.output, device="cpu")
            runner.prepare(args)
            plan = runner.load_plan(args)
            runner._run_evaluation_locked(args, plan)
            runner.publish(args, plan)
            summary = runner.read(bundle.output / "summary.json")
            self.assertTrue(summary["diagnostics_complete"])
            self.assertEqual(summary["inventory"]["scheduled_cells"], summary["inventory"]["executed_cells"])
            self.assertEqual(summary["inventory"]["typed_failure_cells"], 0)
            for token in summary["dispositions"].values():
                self.assertIn(token, runner.DISPOSITION_TOKENS)
            # validate recomputes every published table from the bound cell records
            with patch("sys.argv", ["runner", "--validate", "--output", str(bundle.output)]):
                self.assertEqual(runner.main(), 0)
            # corrupted tables must fail validation
            summary["dispositions"]["question_1_localization"] = "supported"
            (bundle.output / "summary.json").write_text(json.dumps(summary))
            with patch("sys.argv", ["runner", "--validate", "--output", str(bundle.output)]):
                self.assertEqual(runner.main(), 1)

    def test_trace_records_carry_frozen_segments(self):
        with pipeline_overrides(None, [(30, 70)]) as bundle:
            args = argparse.Namespace(output=bundle.output, device="cpu")
            runner.prepare(args)
            plan = runner.load_plan(args)
            runner._run_evaluation_locked(args, plan)
            record = torch.load(bundle.output / "traces" / "window" / "seed-20260908" / "continuous_h15" /
                                "scene-10000-w25.pt", map_location="cpu", weights_only=True)
            self.assertEqual(record["frames"]["relative"], [15, 30, 45, 60])
            self.assertEqual(record["frames"]["absolute"], [40, 55, 70, 85])
            self.assertEqual(record["frames"]["segment"],
                             probe.segment_timeline((30, 70), 25, [15, 30, 45, 60]))
            self.assertEqual(record["segment"] if "segment" in record else record["frames"]["segment"][2],
                             "active_cascade")
            self.assertGreater(float(record["frames"]["position_mse_masked"][-1]), 0.)
            grid_record = torch.load(bundle.output / "traces" / "grid" / "seed-20260908" / "hybrid_h15" /
                                     "scene-10000-t0.pt", map_location="cpu", weights_only=True)
            self.assertEqual(grid_record["frames"]["relative"], [15, 30, 45, 60, 75, 90, 105, 120])
            self.assertEqual(len(grid_record["frames"]["position_mse_all_frames"]), 8)
            self.assertFalse(grid_record["failure"])

    def test_typed_failure_forces_readiness(self):
        with pipeline_overrides(None, PIPELINE_SCENES) as bundle:
            args = argparse.Namespace(output=bundle.output, device="cpu")
            runner.prepare(args)
            plan = runner.load_plan(args)
            kind, seed, arm, horizon, scene, start = runner.scheduled_cells(plan)[0]
            runner.typed_failure(args, plan, kind, seed, arm, horizon, scene, start,
                                 "compute_allowance_exceeded")
            result = runner.publication(args, plan)
            self.assertFalse(result["diagnostics_complete"])
            self.assertEqual(result["dispositions"]["question_1_localization"],
                             "readiness_or_precision_insufficient")
            self.assertEqual(result["dispositions"]["question_2_artifact_sensitivity"],
                             "readiness_or_precision_insufficient")
            self.assertEqual(result["inventory"]["missing_cells"],
                             result["inventory"]["scheduled_cells"] - 1)

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            code = runner.dry_run(argparse.Namespace(output=output))
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
