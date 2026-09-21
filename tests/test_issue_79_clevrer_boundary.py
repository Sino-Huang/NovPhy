"""Focused tests for issue-79 CLEVRER boundary replication (fast, CPU, synthetic)."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from scripts import run_clevrer_boundary_replication as runner
from world_model.model import Abstraction
from world_model.training import clevrer_boundary as clevrer
from world_model.training.cnn_hybrid import DIM, SLOTS


def annotation_payload(scene_index, *, objects=2, collisions=((10, (0, 1)),), visible_from=0):
    """One synthetic CLEVRER annotation dict: constant velocity, full trajectories."""
    frames = []
    for time in range(clevrer.FRAMES_PER_CLIP):
        entries = []
        for index in range(objects):
            location = [index * 1.0 + time * .04, -index * 1.0, .2]
            entries.append({"object_id": index, "location": location,
                            "orientation": [0., 0., .1 * time],
                            "velocity": [1.0, -0.5, 0.], "angular_velocity": [0., 0., .01],
                            "inside_camera_view": time >= visible_from})
        frames.append({"frame_id": time, "objects": entries})
    return {"scene_index": scene_index, "video_filename": f"video_{scene_index}.mp4",
            "object_property": [{"object_id": i, "color": "red", "material": "rubber",
                                 "shape": clevrer.KIND_VOCABULARY[i % 3]} for i in range(objects)],
            "motion_trajectory": frames,
            "collision": [{"frame_id": time, "object_ids": list(pair), "location": [0., 0., .2]}
                          for time, pair in collisions]}


def write_annotation_zip(path, payloads):
    with zipfile.ZipFile(path, "w") as archive:
        for payload in payloads:
            archive.writestr(f"annotation_{payload['scene_index']}.json", json.dumps(payload))


def tiny_plan(output, scenes=(10000,)):
    return {"schema": runner.SCHEMA, "identity": runner.IDENTITY, "claim_boundary": "test boundary",
            "representation": {"identity": clevrer.IDENTITY, "velocity_scale": 3.0,
                               "speed_tier_edges": [0.35, 1.68]},
            "membership": {"scenes": [{"scene_index": index, "objects": 2, "collisions": 1,
                                       "video_filename": f"video_{index}.mp4",
                                       "object_property": []} for index in scenes],
                           "windows_per_scene": clevrer.WINDOWS_PER_SCENE},
            "design": {"training": {"steps": 9000, "seeds": list(runner.SEEDS)},
                       "arms": {}, "seeds": list(runner.SEEDS)},
            "compute": {"allowance_gpu_hours": 6.0}}


def stub_model(frames):
    """Stub carrier dynamics: returns pre-recorded carriers, one per call."""
    calls = {"n": 0}

    class Stub:
        def carrier(self, z, action, pair):
            out = frames[min(calls["n"], len(frames) - 1)].clone()
            calls["n"] += 1
            return out[None]
    return Stub()


class Issue79ClevrerBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

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

    # ---------------------------------------------------------------- parsing/membership

    def test_parse_scene_tensorizes_and_enforces_contract(self):
        payload = annotation_payload(10000)
        scene = clevrer.parse_scene(payload)
        self.assertEqual(scene.scene_index, 10000)
        self.assertEqual(tuple(scene.location.shape), (128, 2, 3))
        self.assertEqual(int(scene.collision_frames[0]), 10)
        self.assertTrue(clevrer.collision_complete(scene))
        with self.assertRaises(clevrer.ClevrerBoundaryError):
            clevrer.parse_scene({"scene_index": 1})
        broken = annotation_payload(1)
        broken["motion_trajectory"] = broken["motion_trajectory"][:64]
        with self.assertRaises(clevrer.ClevrerBoundaryError):
            clevrer.parse_scene(broken)
        broken = annotation_payload(1)
        broken["object_property"][0]["shape"] = "tetrahedron"
        with self.assertRaises(clevrer.ClevrerBoundaryError):
            clevrer.parse_scene(broken)
        broken = annotation_payload(1)
        broken["collision"][0]["object_ids"] = [0, 7]
        with self.assertRaises(clevrer.ClevrerBoundaryError):
            clevrer.parse_scene(broken)

    def test_membership_selects_first_collision_complete_ascending(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.zip"
            write_annotation_zip(path, [
                annotation_payload(10000, collisions=()),   # not collision-complete
                annotation_payload(10001),
                annotation_payload(10002, objects=1, collisions=()),
                annotation_payload(10004),
            ])
            members, scan = clevrer.select_membership(path, count=2)
            self.assertEqual([s.scene_index for s in members], [10001, 10004])
            self.assertEqual([s["scene_index"] for s in scan["skipped"]], [10000, 10002])
            with self.assertRaises(clevrer.ClevrerBoundaryError):
                clevrer.select_membership(path, count=3, scan_limit=2)

    # ---------------------------------------------------------------- carrier/labels

    def test_carrier_layout_masks_and_symmetry(self):
        scene = clevrer.parse_scene(annotation_payload(10000, objects=3, visible_from=10))
        clip = clevrer.clip_carriers(scene, velocity_scale=4.0)
        self.assertEqual(tuple(clip.shape), (128, DIM))
        self.assertEqual(float(clip[0, 0]), 1.0)
        self.assertEqual(float(clip[0, 1]), clevrer.PRIOR_ELAPSED_SECONDS)
        self.assertTrue(bool(torch.isfinite(clip).all()))
        # empty slots stay zero; declared slots carry the 13-column layout
        self.assertEqual(float(clip[:, 2 + 3 * 13:].abs().max()), 0.0)
        for slot in range(3):
            base = 2 + slot * 13
            self.assertTrue(bool((clip[:, base] == 1.0).all()))               # presence
            self.assertTrue(bool((clip[:, base + 2] == slot % 3 / 2).all()))  # kind code
            self.assertEqual(float(clip[:10, base + 7].sum()), 0.0)           # masked before frame 10
            self.assertEqual(float(clip[10:, base + 7].sum()), 118.0)         # available after
            self.assertEqual(float(clip[:10, base + 5:base + 7].abs().sum()), 0.0)
            expected_x = (slot + 10 * .04) / clevrer.POS_SCALE
            self.assertAlmostEqual(float(clip[10, base + 5]), expected_x, places=5)
            self.assertAlmostEqual(float(clip[10, base + 8]), 1.0 / 4.0, places=5)
        self.assertTrue(torch.equal(clip, clevrer.clip_carriers(scene, velocity_scale=4.0)))
        with self.assertRaises(clevrer.ClevrerBoundaryError):
            clevrer.clip_carriers(clevrer.parse_scene(annotation_payload(1, objects=19)), 4.0)

    def test_contact_and_velocity_bin_labels(self):
        scene = clevrer.parse_scene(annotation_payload(10000, objects=3, collisions=((10, (0, 1)),)))
        relations, mask = clevrer.clip_relations(scene, (0.75, 1.5))
        self.assertEqual(tuple(relations.shape), (128, SLOTS, SLOTS, 2))
        for frame in range(128):
            expected = 1.0 if frame == 10 else 0.0
            self.assertEqual(float(relations[frame, 0, 1, 0]), expected)
            self.assertEqual(float(relations[frame, 1, 0, 0]), expected)  # symmetric contact
        self.assertEqual(float(relations[10, 0, 2, 0]), 0.0)
        # velocity-bin: speeds are all sqrt(1.25); tier constant among declared slots
        self.assertEqual(float(relations[:, 0, 1, 1].min()), 1.0)
        self.assertEqual(float(relations[:, 0, 2, 1].min()), 1.0)
        off = ~torch.eye(SLOTS, dtype=torch.bool)
        declared = torch.zeros(SLOTS, dtype=torch.bool)
        declared[:3] = True
        self.assertTrue(bool((mask[0, :, :, 0][off & declared[None, :] & declared[:, None]]).all()))
        self.assertFalse(bool(mask[0, 0, 5, 0]))  # undeclared slots carry no labels
        self.assertFalse(bool(mask[0, 0, 0, 0]))  # neither does the diagonal
        scene = clevrer.parse_scene(annotation_payload(10000, objects=2, collisions=()))
        frames = scene.velocity
        for time in range(128):  # distinct tiers: object 0 fast, object 1 slow
            frames[time, 0, 0] = 2.0
            frames[time, 1, 0] = .1
        relations, _ = clevrer.clip_relations(scene, (0.75, 1.5))
        self.assertEqual(float(relations[:, 0, 1, 1].min()), 0.0)

    def test_training_windows_follow_the_frozen_starts(self):
        scene = clevrer.parse_scene(annotation_payload(10000))
        relations, mask = clevrer.clip_relations(scene, (0.75, 1.5))
        clip = clevrer.clip_carriers(scene, 4.0)
        windows = clevrer.training_windows(clip, relations, mask)
        self.assertEqual(windows["z"].shape, (40, 61, DIM))
        self.assertEqual(windows["length"].tolist(), [60] * 40)
        self.assertEqual(float(windows["action"].abs().sum()), 0.0)
        for start in (0, 1, 39):
            self.assertTrue(torch.equal(windows["z"][start], clip[start:start + 61]))
            self.assertTrue(torch.equal(windows["relations"][start], relations[start:start + 61]))

    def test_eval_grid_divisibility_is_enforced(self):
        clevrer.check_eval_grid()
        with patch.object(clevrer, "ENDPOINTS", (15, 127)), patch.object(clevrer, "EVAL_CONTEXTS", (4,)):
            with self.assertRaises(clevrer.ClevrerBoundaryError):
                clevrer.check_eval_grid()
        with patch.object(clevrer, "ENDPOINTS", (15, 20)), patch.object(clevrer, "EVAL_CONTEXTS", (0,)):
            with self.assertRaises(clevrer.ClevrerBoundaryError):
                clevrer.check_eval_grid()

    # ---------------------------------------------------------------- estimands

    def test_field_errors_apply_the_camera_mask(self):
        target = torch.zeros(DIM)
        target[2 + 0 * 13 + 5:2 + 0 * 13 + 7] = 1.0        # slot 0 position, available
        target[2 + 0 * 13 + 7] = 1.0
        target[2 + 1 * 13 + 5:2 + 1 * 13 + 7] = 2.0        # slot 1 position, unavailable
        prediction = target.clone()
        self.assertEqual(runner and clevrer.field_errors(prediction, target)["position_mse"], 0.0)
        prediction[2 + 1 * 13 + 5] = 100.0                  # masked slot: must not count
        self.assertEqual(clevrer.field_errors(prediction, target)["position_mse"], 0.0)
        prediction[2 + 0 * 13 + 5] = 3.0                    # available slot: squared error 4 over 2 columns
        self.assertEqual(clevrer.field_errors(prediction, target)["position_mse"], 2.0)
        self.assertEqual(clevrer.field_errors(prediction, target)["position_available_values"], 2)
        with self.assertRaises(clevrer.ClevrerBoundaryError):
            clevrer.field_errors(torch.zeros(DIM), torch.full((DIM,), float("nan")))

    def test_interval_active_matches_the_frozen_window(self):
        events = torch.tensor([10, 34])
        self.assertTrue(clevrer.interval_active(events, 0, 60))
        self.assertFalse(clevrer.interval_active(events, 40, 60))
        self.assertTrue(clevrer.interval_active(events, 11, 30))
        self.assertFalse(clevrer.interval_active(events, 0, 9))  # window excludes frame 10

    def test_frozen_scale_and_edges_are_deterministic_rules(self):
        scenes = [clevrer.parse_scene(annotation_payload(index, objects=2, collisions=((5, (0, 1)),)))
                  for index in (1, 2)]
        for scene in scenes:  # speeds sqrt(1.25) -> peak 1.25 -> scale 1.5
            scene.velocity[:, :, 0] = 1.0
            scene.velocity[:, :, 1] = .5
        self.assertEqual(clevrer.frozen_velocity_scale(tuple(scenes)), 1.5)
        for scene in scenes:  # object 0 speed 2.6 -> scale 3.0
            scene.velocity[:, :, 0] = 2.6
            scene.velocity[:, :, 1] = 0.
        self.assertEqual(clevrer.frozen_velocity_scale(tuple(scenes)), 3.0)
        # tier edges: tertiles of {0.1 x 128} ++ {2.0 x 128} -> both boundaries inside
        for scene in scenes:
            scene.velocity[:, 0, :] = torch.tensor([2.0, 0., 0.])
            scene.velocity[:, 1, :] = torch.tensor([.1, 0., 0.])
        lower, upper = clevrer.frozen_tier_edges(tuple(scenes))
        self.assertTrue(.1 <= lower < upper <= 2.0)
        self.assertEqual(clevrer.frozen_tier_edges(tuple(scenes)), (lower, upper))

    # ---------------------------------------------------------------- training

    def synthetic_pool(self, size=8):
        torch.manual_seed(79)
        return {"z": torch.rand(size, 61, DIM) * .1,
                "action": torch.zeros(size, 5),
                "length": torch.full((size,), 60, dtype=torch.long),
                "relations": torch.zeros(size, 61, SLOTS, SLOTS, 2),
                "relations_mask": (~torch.eye(SLOTS, dtype=torch.bool))[None, None, :, :, None]
                .expand(size, 61, SLOTS, SLOTS, 2)}

    def tiny_training_plan(self, steps=9000):
        return {"schema": runner.SCHEMA, "identity": "issue-79-clevrer-boundary-v1:tiny",
                "capacity": {"continuous_width": 64},
                "representation": {"identity": clevrer.IDENTITY, "velocity_scale": 3.0,
                                   "speed_tier_edges": [.35, 1.68]},
                "membership": {"scenes": [{"scene_index": 10000}]},
                "design": {"training": {"steps": steps, "batch_size": 4, "learning_rate": .0001,
                                        "weight_decay": .0001, "grad_clip": 1.},
                           "seeds": [20260908]}}

    def train_args(self, directory):
        return argparse.Namespace(output=Path(directory), device="cpu")

    def test_expected_pair_counts_follow_the_actual_cycle(self):
        plan = self.tiny_training_plan(steps=12)
        # 6-slot cycle over 12 steps: each slot twice; the continuous arm collapses
        # both abstraction slots of a delta onto the continuous identity (4 each).
        self.assertEqual(runner.expected_pair_counts(plan, "hybrid"),
                         {str(p.identity): 2 for p in runner.PAIRS_CLEVRER})
        self.assertEqual(runner.expected_pair_counts(plan, "continuous"),
                         {str(runner.PredictionPair(p.delta, Abstraction.CONTINUOUS).identity): 4
                          for p in runner.PAIRS_CLEVRER if p.abstraction == Abstraction.CONTINUOUS})
        self.assertEqual(runner.expected_pair_counts(self.tiny_training_plan(steps=9000), "continuous"),
                         {str(p.identity): 3000 for p in runner.PAIRS_CLEVRER
                          if p.abstraction == Abstraction.CONTINUOUS})

    def test_expected_sets_exclude_declared_untrained_macro_parameters(self):
        plan = self.tiny_training_plan()
        self.assertEqual(len(runner.expected_pair_counts(plan, "continuous")), 3)
        self.assertEqual(len(runner.expected_pair_counts(plan, "hybrid")), 6)
        hybrid = runner.expected_gradient_parameters(plan, "hybrid")
        self.assertTrue(all(not name.startswith(("macro_head", "macro_adapter")) for name in hybrid))
        self.assertEqual(len(runner.expected_gradient_parameters(plan, "continuous")),
                         len({n for n, _ in runner.new_model(plan, "continuous").named_parameters()}))

    def test_training_progress_and_declared_gradient_coverage(self):
        plan = self.tiny_training_plan()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "pool", lambda args, plan, arm: self.synthetic_pool()):
            args = self.train_args(directory)
            runner.train_cell(args, plan, 20260908, "hybrid", stop_after=4)
            progress = torch.load(args.output / "seed-20260908/hybrid/predictor-progress.pt",
                                  map_location="cpu", weights_only=True)
            self.assertEqual(progress["step"], 4)
            self.assertTrue(all(not name.startswith(("macro_head", "macro_adapter"))
                                for name in progress["gradient_parameters"]))
            self.assertTrue(progress["gradient_parameters"])  # everything trainable saw gradients
            self.assertEqual(sum(progress["pair_counts"].values()), 4)
            # declared macro parameters must not move: compare to the exact seeded init
            torch.manual_seed(20260908)
            fresh = runner.new_model(plan, "hybrid")
            model = runner.new_model(plan, "hybrid")
            model.load_state_dict(progress["model"], strict=True)
            for name in ("macro_head.0.weight", "macro_adapter.weight"):
                self.assertTrue(torch.equal(dict(model.named_parameters())[name],
                                            dict(fresh.named_parameters())[name]))

    def test_training_resume_matches_uninterrupted(self):
        plan = self.tiny_training_plan()
        data = self.synthetic_pool()
        with tempfile.TemporaryDirectory() as interrupted, \
                tempfile.TemporaryDirectory() as direct, \
                patch.object(runner, "pool", lambda args, plan, arm: data):
            args = self.train_args(interrupted)
            runner.train_cell(args, plan, 20260908, "continuous", stop_after=3)
            runner.train_cell(args, plan, 20260908, "continuous", stop_after=6)
            args = self.train_args(direct)
            runner.train_cell(args, plan, 20260908, "continuous", stop_after=6)
            resumed = torch.load(Path(interrupted) / "seed-20260908/continuous/predictor-progress.pt",
                                 map_location="cpu", weights_only=True)
            single = torch.load(Path(direct) / "seed-20260908/continuous/predictor-progress.pt",
                                map_location="cpu", weights_only=True)
            self.assertEqual(resumed["step"], single["step"])
            for name, value in resumed["model"].items():
                self.assertTrue(torch.equal(value, single["model"][name]), name)
            self.assertTrue(torch.equal(resumed["generator"], single["generator"]))
            self.assertEqual(resumed["pair_counts"], single["pair_counts"])

    def test_training_completion_enforces_frozen_coverage(self):
        plan = self.tiny_training_plan(steps=6)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "pool", lambda args, plan, arm: self.synthetic_pool()):
            args = self.train_args(directory)
            runner.train_cell(args, plan, 20260908, "continuous")  # completes the frozen cycle
            predictor = args.output / "seed-20260908/continuous/predictor.pt"
            self.assertTrue(predictor.exists())
            runner.load_predictor(args, plan, 20260908, "continuous")
            # corrupt the recorded coverage: the completion guard must reject the fit
            progress = torch.load(args.output / "seed-20260908/continuous/predictor-progress.pt",
                                  map_location="cpu", weights_only=True)
            progress["pair_counts"][str(runner.PredictionPair(5, Abstraction.CONTINUOUS).identity)] = 0
            runner.atomic_torch(args.output / "seed-20260908/continuous/predictor-progress.pt", progress)
            predictor.unlink()
            with self.assertRaises(ValueError):
                runner.train_cell(args, plan, 20260908, "continuous")

    # ---------------------------------------------------------------- evaluation/publication

    def eval_plan_and_evidence(self, directory):
        scenes = (10000, 10001)
        events = {10000: [20], 10001: [90]}  # early-window active vs contact-still scene
        plan = tiny_plan(directory, scenes=scenes)
        output = Path(directory)
        scenes = [m["scene_index"] for m in plan["membership"]["scenes"]]
        units = [(s, c) for s in scenes for c in clevrer.EVAL_CONTEXTS]
        args = argparse.Namespace(output=output)
        torch.manual_seed(790)
        for seed in runner.SEEDS:
            for system, (arm, pair) in runner.SYSTEMS.items():
                offset = {"continuous": 0.02, "hybrid": 0.01}[arm] + (0.005 if "micro" in system else 0.)
                for scene, context in units:
                    errors = {}
                    for endpoint in clevrer.ENDPOINTS:
                        base = .01 * endpoint / 15
                        errors[str(endpoint)] = {"errors": {
                            "carrier_mse": base + offset, "presence_mse": 0.0,
                            "position_mse": base + offset, "velocity_mse": base + offset,
                            "position_available_values": 4, "velocity_available_values": 4,
                            "kind_mse": 0.0, "kind_available_values": 4}, "steps": endpoint // pair.delta}
                    runner.atomic_torch(runner.eval_path(args, seed, system, scene, context), {
                        "binding": runner.eval_binding(plan, seed, system),
                        "scene_index": scene, "context": context, "horizon": pair.delta,
                        "system": system, "arm": arm, "pair": list(pair.identity),
                        "active_parameters": 100, "wall_seconds": .1,
                        "result": {"errors": errors, "rollout_wall_seconds": .2,
                                   "transition_calls": 120 // pair.delta,
                                   "linear_macs_per_step": 7}, "failure": None})
        for scene in scenes:  # scene shard for the regime analysis
            runner.atomic_torch(output / "shards" / f"scene-{scene}.pt", {
                "schema": "issue_79_clevrer_scene_v1", "scene_index": scene,
                "identity": clevrer.IDENTITY, "velocity_scale": 3.0,
                "speed_tier_edges": [.35, 1.68], "objects": [], "clip": torch.zeros(1),
                "windows": {}, "vertical_location": torch.zeros(1), "vertical_velocity": torch.zeros(1),
                "orientation": torch.zeros(1), "angular_velocity": torch.zeros(1),
                "collision_frames": torch.tensor(events[scene]),
                "collision_pairs": torch.tensor([[0, 1]] * len(events[scene]))})
        return plan, units

    def test_publication_serialization_and_validate_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, units = self.eval_plan_and_evidence(directory)
            args = argparse.Namespace(output=Path(directory), device="cpu")
            with patch.object(runner, "REGIME_MINIMUM_UNITS", 1):
                result = runner.publication(args, plan)
                self.assertTrue(result["diagnostics_complete"])
                self.assertEqual(result["inventory"]["executed_cells"],
                                 result["inventory"]["scheduled_cells"])
                self.assertIn(result["dispositions"]["question_1_signature"], runner.DISPOSITION_TOKENS)
                self.assertIn(result["dispositions"]["question_2_regime"], runner.DISPOSITION_TOKENS)
                regime_rows = [row for row in result["regime"] if row["endpoint"] == 120]
                self.assertTrue(all(row["active"]["paired_units"] == 4 and row["still"]["paired_units"] == 4
                                    for row in regime_rows))
                csv_text = runner.comparisons_csv(result)
                rows = [row for row in csv_text.splitlines() if row]
                self.assertTrue(rows[0].startswith("seed,system,endpoint_frames"))
                self.assertTrue(any("training_effect" in row for row in rows))
                self.assertTrue(any("symbolic_execution_regime" in row for row in rows))
                findings = runner.findings_md(result)
                self.assertIn("Dispositions", findings)
                self.assertIn(result["dispositions"]["question_1_signature"], findings)
                runner.publish(args, plan)
                republished = runner.publication(args, plan)
                self.assertEqual(runner.compact_report(republished),
                                 runner.read(Path(directory) / "summary.json"))
                self.assertEqual(runner.comparisons_csv(republished).encode("utf-8"),
                                 (Path(directory) / "comparisons.csv").read_bytes())
                self.assertEqual(runner.findings_md(republished),
                                 (Path(directory) / "findings.md").read_text())

    def test_publication_reports_incomplete_evidence_as_precision_insufficient(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, units = self.eval_plan_and_evidence(directory)
            args = argparse.Namespace(output=Path(directory), device="cpu")
            missing = runner.eval_path(args, runner.SEEDS[0], "continuous_h1",
                                       units[0][0], units[0][1])
            missing.unlink()
            result = runner.publication(args, plan)
            self.assertFalse(result["diagnostics_complete"])
            self.assertEqual(result["dispositions"]["question_1_signature"], "readiness_or_precision_insufficient")

    def test_typed_failure_cells_force_precision_insufficient(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, units = self.eval_plan_and_evidence(directory)
            args = argparse.Namespace(output=Path(directory), device="cpu")
            path = runner.eval_path(args, runner.SEEDS[0], "continuous_h1", units[0][0], units[0][1])
            record = torch.load(path, map_location="cpu", weights_only=True)
            record.update(failure="nonfinite recursive carrier", result=None)
            runner.atomic_torch(path, record)
            result = runner.publication(args, plan)
            self.assertTrue(result["diagnostics_complete"])
            self.assertEqual(result["dispositions"]["question_1_signature"], "readiness_or_precision_insufficient")
            self.assertEqual(result["inventory"]["typed_failure_cells"], 1)

    def test_rollout_errors_walk_the_grid_without_resets(self):
        scene = clevrer.parse_scene(annotation_payload(10000))
        clip = clevrer.clip_carriers(scene, 3.0)
        stub = stub_model(clip[5:120])  # first call returns clip[5], second clip[6], ...
        with patch.object(runner, "linear_macs", lambda model, pair: 7):
            result = runner.rollout_errors(stub, clip, 5, runner.SYSTEMS["continuous_h15"][1], "cpu")
        self.assertEqual([result["errors"][str(e)]["steps"] for e in clevrer.ENDPOINTS], [1, 2, 4, 8])
        expected = clevrer.field_errors(clip[5], clip[5 + 15])  # first stub carrier equals clip[5]
        self.assertEqual(result["errors"]["15"]["errors"]["carrier_mse"], expected["carrier_mse"])
        self.assertGreater(result["errors"]["15"]["errors"]["carrier_mse"], 0.0)
        self.assertEqual(result["transition_calls"], 8)
        self.assertGreaterEqual(result["rollout_wall_seconds"], 0.0)
        self.assertEqual(7, result["linear_macs_per_step"])

    def test_decision_rules_use_only_the_frozen_tokens(self):
        plan = self.tiny_training_plan()
        contrasts, regime = [], []
        for kind, template in (("training_effect", "hybrid_continuous_h{h}"),):
            for horizon in clevrer.HORIZONS:
                for endpoint in clevrer.ENDPOINTS:
                    contrasts.append({"kind": kind, "horizon": horizon, "endpoint": endpoint,
                                      "position_mse": {"mean": .05 if (horizon, endpoint) == (1, 15) else .0,
                                                       "paired_units": 48}})
        per_seed = {str(seed): {} for seed in runner.SEEDS}
        for system, curves in (("continuous_h1", {15: .01, 30: .02, 60: .03, 120: .04}),):
            for seed in runner.SEEDS:
                per_seed[str(seed)][system] = {"curves": {str(e): {"position_mse": v}
                                                          for e, v in curves.items()}}
        # S3 requires every horizon curve; other systems needed for regime margins
        for horizon in clevrer.HORIZONS:
            for seed in runner.SEEDS:
                per_seed[str(seed)][f"continuous_h{horizon}"] = \
                    per_seed[str(seed)]["continuous_h1"]
                per_seed[str(seed)][f"hybrid_continuous_h{horizon}"] = \
                    {"curves": {"120": {"position_mse": 1.0}}}
        decision = runner.decide(plan, contrasts, regime, per_seed, typed_failures=0)
        self.assertEqual(decision["question_1_signature"], "supported")
        self.assertIn(decision["question_2_regime"], runner.DISPOSITION_TOKENS)
        contrasts[0]["position_mse"]["mean"] = -.05  # S1 sign fails -> no replication claim
        decision = runner.decide(plan, contrasts, regime, per_seed, typed_failures=0)
        self.assertEqual(decision["question_1_signature"], "not_supported_by_this_experiment")
        decision = runner.decide(plan, contrasts, regime, per_seed, typed_failures=3)
        self.assertEqual(decision["question_1_signature"], "readiness_or_precision_insufficient")

    # ---------------------------------------------------------------- operator surface

    def test_gpu_lock_blocks_concurrent_holders(self):
        probe = ("import fcntl, sys\nhandle = open('%s', 'a+b')\n"
                 "try:\n    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
                 "except BlockingIOError:\n    sys.exit(3)\nsys.exit(0)\n") % runner.GPU_LOCK
        held_elsewhere = subprocess.run([sys.executable, "-c", probe], capture_output=True).returncode == 3
        if held_elsewhere:
            self.skipTest("another ticket currently holds the shared GPU lock")
        with runner.gpu_lock():
            blocked = subprocess.run([sys.executable, "-c", probe], capture_output=True)
            self.assertEqual(blocked.returncode, 3, blocked.stderr)
        released = subprocess.run([sys.executable, "-c", probe], capture_output=True)
        self.assertEqual(released.returncode, 0, released.stderr)

    def test_train_requires_the_shared_gpu_device(self):
        plan = self.tiny_training_plan()
        with self.assertRaises(ValueError):
            runner.train(self.train_args(tempfile.gettempdir()), plan)

    def test_dry_run_writes_nothing_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=Path(directory))
            with patch.object(runner, "ZIPS", {"train": Path(directory) / "missing.zip",
                                               "validation": Path(directory) / "missing.zip"}):
                self.assertEqual(runner.dry_run(args), 0)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
