"""Focused tests for issue-84 third-family boundary replication (fast, CPU, synthetic)."""
import argparse
import gzip
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from scripts import run_third_family_boundary_replication as runner
from world_model.model import Abstraction
from world_model.training import third_family_boundary as family
from world_model.training.clevrer_boundary import field_errors
from world_model.training.cnn_hybrid import DIM, SLOTS


def build_trial(name="pilot_dominoes_x_0001", *, frames=130, objects=3,
                collisions=((10, (1, 2)),), model_names=None, skip_collisions=False,
                drop_frame=None, ragged_frame=None):
    """One synthetic dominoes trial HDF5 as bytes."""
    names = model_names or (["cube"] * objects)
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as h5:
        static = h5.create_group("static")
        static.create_dataset("object_ids", data=np.arange(1, objects + 1, dtype=np.int32))
        static.create_dataset("model_names", data=np.array(names, dtype="S"))
        frames_grp = h5.create_group("frames")
        for time in range(frames):
            if time == drop_frame:
                continue
            frame = frames_grp.create_group(f"{time:04d}")
            objs = frame.create_group("objects")
            shape = (objects + 1, 3) if time == ragged_frame else (objects, 3)
            objs.create_dataset("positions", data=np.full(shape, .1 * time, dtype=np.float32))
            objs.create_dataset("rotations", data=np.tile([0., 0., 0., 1.], (objects, 1)).astype(np.float32))
            objs.create_dataset("velocities", data=np.full((objects, 3), .05, dtype=np.float32))
            objs.create_dataset("angular_velocities", data=np.zeros((objects, 3), dtype=np.float32))
            coll = frame.create_group("collisions")
            if not skip_collisions:
                rows = [list(pair) for event, pair in collisions if event == time]
                coll.create_dataset("object_ids", data=np.array(rows, dtype=np.int32).reshape(-1, 2))
    return name, buffer.getvalue()


def write_archive(path, payloads, *, truncate_bytes=None):
    """Write a gzipped tar of hdf5 payloads; optionally truncate the stream."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name, payload in payloads:
            info = tarfile.TarInfo(name if name.endswith(".hdf5") else name + ".hdf5")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    data = gzip.compress(raw.getvalue(), 1)
    if truncate_bytes is not None:
        data = data[:truncate_bytes]
    path.write_bytes(data)
    return path


def tiny_plan(directory, scenes=(0,)):
    return {"schema": runner.SCHEMA, "identity": runner.IDENTITY, "claim_boundary": "test boundary",
            "phase_0": runner.PHASE_0_RECORD,
            "representation": {"identity": family.IDENTITY, "position_scale": 2.0,
                               "velocity_scale": 1.5, "speed_tier_edges": [.06, .07],
                               "kind_vocabulary": ["cube", "vase_01"]},
            "membership": {"scenes": [{"scene_index": index, "stimulus_name": f"t{index}", "objects": 3,
                                       "frames": 130, "collision_events": 1, "model_names": ["cube"] * 3}
                                      for index in scenes],
                           "windows_per_trial": family.WINDOWS_PER_SCENE},
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


class Issue84ThirdFamilyTests(unittest.TestCase):
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

    def test_parse_trial_tensorizes_and_enforces_contract(self):
        name, payload = build_trial(frames=131, objects=3, collisions=((10, (2, 3)),))
        trial = family.parse_trial(name, payload, scene_index=0)
        self.assertEqual(trial.stimulus_name, "pilot_dominoes_x_0001")
        self.assertEqual(tuple(trial.positions.shape), (131, 3, 3))
        self.assertEqual(tuple(trial.rotations.shape), (131, 3, 4))
        self.assertEqual(int(trial.collision_frames[0]), 10)
        self.assertEqual(trial.collision_pairs[0].tolist(), [1, 2])  # engine ids 2,3 -> slots 1,2
        self.assertTrue(family.trial_complete(trial))
        with self.assertRaises(family.ThirdFamilyError):
            family.parse_trial(name, b"not hdf5", scene_index=0)
        _, short = build_trial(frames=family.FRAMES_MINIMUM - 1)
        with self.assertRaises(family.ThirdFamilyError):
            family.parse_trial("short", short, scene_index=0)
        _, gap = build_trial(frames=130, drop_frame=64)
        with self.assertRaises(family.ThirdFamilyError):
            family.parse_trial("gap", gap, scene_index=0)
        _, ragged = build_trial(frames=130, objects=2, ragged_frame=5)
        with self.assertRaises(family.ThirdFamilyError):
            family.parse_trial("ragged", ragged, scene_index=0)
        _, badpair = build_trial(collisions=((10, (1, 9)),))
        with self.assertRaises(family.ThirdFamilyError):
            family.parse_trial("badpair", badpair, scene_index=0)
        _, selfpair = build_trial(collisions=((10, (2, 2)),))
        with self.assertRaises(family.ThirdFamilyError):
            family.parse_trial("selfpair", selfpair, scene_index=0)

    def test_membership_selects_first_complete_in_stored_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefix.tar.gz"
            payloads = [
                build_trial("a_incomplete", frames=140, collisions=()),           # no collision event
                build_trial("b_ok", frames=140),
                build_trial("c_short", frames=family.FRAMES_MINIMUM - 10),
                build_trial("d_ok", frames=129),
            ]
            write_archive(path, payloads)
            members, scan = family.select_membership(path, count=2)
            self.assertEqual([t.stimulus_name for t in members], ["b_ok", "d_ok"])
            self.assertEqual([s["member"].rsplit("/", 1)[-1].removesuffix(".hdf5")
                              for s in scan["skipped"]], ["a_incomplete", "c_short"])
            self.assertIn("not frame-complete", scan["skipped"][0]["reason"])
            with self.assertRaises(family.ThirdFamilyError):
                family.select_membership(path, count=3)
            # a truncated stream yields the complete prefix members only
            truncated = Path(directory) / "truncated.tar.gz"
            write_archive(truncated, payloads, truncate_bytes=len(path.read_bytes()) - 512)
            members, scan = family.select_membership(truncated, count=1)
            self.assertEqual([t.stimulus_name for t in members], ["b_ok"])

    # ---------------------------------------------------------------- carrier/labels

    def test_carrier_layout_scales_and_determinism(self):
        _, payload = build_trial(frames=130, objects=3,
                                 model_names=["cube", "vase_01", "cube"])
        trial = family.parse_trial("t", payload, scene_index=0)
        vocab = family.frozen_kind_vocabulary((trial,))
        clip = family.clip_carriers(trial, position_scale=2.0, velocity_scale=1.0,
                                    kind_vocabulary=vocab)
        self.assertEqual(tuple(clip.shape), (130, DIM))
        self.assertEqual(float(clip[0, 0]), 1.0)
        self.assertAlmostEqual(float(clip[0, 1]), family.PHYSION_DT_SECONDS, places=9)
        self.assertTrue(bool(torch.isfinite(clip).all()))
        self.assertEqual(float(clip[:, 2 + 3 * 13:].abs().max()), 0.0)  # empty slots stay zero
        codes = {slot: vocab.index(name) / (len(vocab) - 1) for slot, name in enumerate(trial.model_names)}
        for slot in range(3):
            base = 2 + slot * 13
            self.assertEqual(float(clip[:, base + 5].min()), 0.0)          # position starts at origin 0.1*0
            self.assertAlmostEqual(float(clip[129, base + 5]), 12.9 / 2.0, places=5)  # x/scale
            self.assertEqual(float(clip[:, base + 7].sum()), 130.0)         # all-frames availability
            self.assertEqual(float(clip[:, base + 10].sum()), 130.0)
            self.assertAlmostEqual(float(clip[0, base + 8]), 0.05 / 1.0, places=6)
            self.assertAlmostEqual(float(clip[0, base + 2]), codes[slot], places=6)
        self.assertTrue(torch.equal(clip, family.clip_carriers(trial, 2.0, 1.0, vocab)))
        with self.assertRaises(family.ThirdFamilyError):
            family.clip_carriers(trial, 2.0, 1.0, ("sphere", "cube"))
        _, big = build_trial(objects=19, frames=130)
        with self.assertRaises(family.ThirdFamilyError):
            family.clip_carriers(family.parse_trial("big", big, 0), 2.0, 1.0, ("cube",))

    def test_contact_and_velocity_bin_labels(self):
        _, payload = build_trial(frames=130, objects=3, collisions=((10, (2, 3)),))
        trial = family.parse_trial("t", payload, scene_index=0)
        relations, mask = family.clip_relations(trial, (.06, .07))
        self.assertEqual(tuple(relations.shape), (130, SLOTS, SLOTS, 2))
        for frame in range(130):
            expected = 1.0 if frame == 10 else 0.0
            self.assertEqual(float(relations[frame, 1, 2, 0]), expected)
            self.assertEqual(float(relations[frame, 2, 1, 0]), expected)  # symmetric contact
        self.assertEqual(float(relations[10, 0, 1, 0]), 0.0)
        # constant speeds sqrt(3*.05^2) -> one tier among declared slots
        self.assertEqual(float(relations[:, 0, 1, 1].min()), 1.0)
        off = ~torch.eye(SLOTS, dtype=torch.bool)
        declared = torch.zeros(SLOTS, dtype=torch.bool)
        declared[:3] = True
        self.assertTrue(bool((mask[0, :, :, 0][off & declared[None, :] & declared[:, None]]).all()))
        self.assertFalse(bool(mask[0, 0, 5, 0]))  # undeclared slots carry no labels
        self.assertFalse(bool(mask[0, 0, 0, 0]))  # neither does the diagonal
        _, split = build_trial(frames=130, objects=2)
        parsed = family.parse_trial("split", split, scene_index=0)
        velocity = parsed.velocity.clone()
        for time in range(130):  # object 0 fast, object 1 slow -> distinct tiers
            velocity[time, 0] = torch.tensor([2.0, 0., 0.])
            velocity[time, 1] = torch.tensor([.1, 0., 0.])
        parsed = family.TrialAnnotations(parsed.scene_index, parsed.stimulus_name, parsed.model_names,
                                         parsed.object_ids, parsed.positions, parsed.rotations,
                                         velocity, parsed.angular_velocity,
                                         parsed.collision_frames, parsed.collision_pairs)
        relations, _ = family.clip_relations(parsed, (.75, 1.5))
        self.assertEqual(float(relations[:, 0, 1, 1].min()), 0.0)

    def test_window_starts_follow_the_frozen_rule(self):
        self.assertEqual(family.window_starts(124)[-1], 124 - family.WINDOW_LENGTH)
        self.assertEqual(family.window_starts(151)[-1], 151 - family.WINDOW_LENGTH)
        self.assertEqual(len(family.window_starts(200)), family.WINDOWS_PER_SCENE)
        starts = family.window_starts(151)
        self.assertEqual(list(starts), [i * 90 // 39 for i in range(40)])
        with self.assertRaises(family.ThirdFamilyError):
            family.window_starts(family.WINDOW_LENGTH - 1)

    def test_training_windows_follow_the_frozen_starts(self):
        _, payload = build_trial(frames=151, objects=3)
        trial = family.parse_trial("t", payload, scene_index=0)
        relations, mask = family.clip_relations(trial, (.06, .07))
        clip = family.clip_carriers(trial, 2.0, 1.0, family.frozen_kind_vocabulary((trial,)))
        windows = family.training_windows(clip, relations, mask)
        self.assertEqual(windows["z"].shape, (40, family.WINDOW_LENGTH, DIM))
        self.assertEqual(windows["length"].tolist(), [60] * 40)
        self.assertEqual(float(windows["action"].abs().sum()), 0.0)
        starts = family.window_starts(151)
        for start in (starts[0], starts[1], starts[39]):
            index = starts.index(start)
            self.assertTrue(torch.equal(windows["z"][index], clip[start:start + 61]))
            self.assertTrue(torch.equal(windows["relations"][index], relations[start:start + 61]))

    def test_eval_grid_divisibility_and_bounds_are_enforced(self):
        family.check_eval_grid(family.FRAMES_MINIMUM)
        family.check_eval_grid(300)
        with self.assertRaises(family.ThirdFamilyError):
            family.check_eval_grid(family.FRAMES_MINIMUM - 1)

    # ---------------------------------------------------------------- frozen rules

    def test_field_errors_are_reused_from_the_clevrer_machinery(self):
        self.assertIs(family.field_errors, field_errors)

    def test_interval_flags_match_the_frozen_window(self):
        events = torch.tensor([10, 34])
        self.assertTrue(family.interval_event_flags(events, 0, 30))
        self.assertFalse(family.interval_event_flags(events, 40, 30))
        self.assertTrue(family.interval_event_flags(events, 11, 30))
        self.assertFalse(family.interval_event_flags(events, 0, 9))

    def test_frozen_scales_and_edges_are_deterministic_rules(self):
        trials = []
        for index in (0, 1):
            _, payload = build_trial(frames=130, objects=2, collisions=((5, (1, 2)),))
            trial = family.parse_trial(str(index), payload, scene_index=index)
            velocity = trial.velocity.clone()
            velocity[:, 0] = torch.tensor([2.6, 0., 0.])
            velocity[:, 1] = torch.tensor([.1, 0., 0.])
            positions = trial.positions.clone()
            positions[:, :, family.GROUND_AXIS_X] = 1.7
            positions[:, :, family.GROUND_AXIS_Z] = .5
            trials.append(family.TrialAnnotations(trial.scene_index, trial.stimulus_name, trial.model_names,
                                                  trial.object_ids, positions, trial.rotations,
                                                  velocity, trial.angular_velocity,
                                                  trial.collision_frames, trial.collision_pairs))
        self.assertEqual(family.frozen_velocity_scale(tuple(trials)), 3.0)
        self.assertEqual(family.frozen_position_scale(tuple(trials)), 2.0)
        lower, upper = family.frozen_tier_edges(tuple(trials))
        self.assertTrue(.1 <= lower < upper <= 2.6)
        self.assertEqual(family.frozen_tier_edges(tuple(trials)), (lower, upper))
        self.assertEqual(family.frozen_kind_vocabulary(tuple(trials)), ("cube",))

    # ---------------------------------------------------------------- training

    def synthetic_pool(self, size=8):
        torch.manual_seed(84)
        return {"z": torch.rand(size, family.WINDOW_LENGTH, DIM) * .1,
                "action": torch.zeros(size, 5),
                "length": torch.full((size,), family.WINDOW_LENGTH - 1, dtype=torch.long),
                "relations": torch.zeros(size, family.WINDOW_LENGTH, SLOTS, SLOTS, 2),
                "relations_mask": (~torch.eye(SLOTS, dtype=torch.bool))[None, None, :, :, None]
                .expand(size, family.WINDOW_LENGTH, SLOTS, SLOTS, 2)}

    def tiny_training_plan(self, steps=9000):
        return {"schema": runner.SCHEMA, "identity": "issue-84-third-family-v1:tiny",
                "capacity": {"continuous_width": 64},
                "representation": {"identity": family.IDENTITY, "position_scale": 2.0,
                                   "velocity_scale": 1.5, "speed_tier_edges": [.06, .07],
                                   "kind_vocabulary": ["cube", "vase_01"]},
                "membership": {"scenes": [{"scene_index": 0, "stimulus_name": "t0"}]},
                "design": {"training": {"steps": steps, "batch_size": 4, "learning_rate": .0001,
                                        "weight_decay": .0001, "grad_clip": 1.},
                           "seeds": [20260908]}}

    def train_args(self, directory):
        return argparse.Namespace(output=Path(directory), device="cpu")

    def test_expected_pair_counts_follow_the_actual_cycle(self):
        plan = self.tiny_training_plan(steps=12)
        self.assertEqual(runner.expected_pair_counts(plan, "hybrid"),
                         {str(p.identity): 2 for p in runner.PAIRS_FAMILY})
        self.assertEqual(runner.expected_pair_counts(plan, "continuous"),
                         {str(runner.PredictionPair(p.delta, Abstraction.CONTINUOUS).identity): 4
                          for p in runner.PAIRS_FAMILY if p.abstraction == Abstraction.CONTINUOUS})
        self.assertEqual(runner.expected_pair_counts(self.tiny_training_plan(steps=9000), "continuous"),
                         {str(p.identity): 3000 for p in runner.PAIRS_FAMILY
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
        scenes = (0, 1)
        events = {0: [20], 1: [90]}  # early-window active vs contact-still trial
        plan = tiny_plan(directory, scenes=scenes)
        output = Path(directory)
        scenes = [m["scene_index"] for m in plan["membership"]["scenes"]]
        units = [(s, c) for s in scenes for c in family.EVAL_CONTEXTS]
        args = argparse.Namespace(output=output)
        torch.manual_seed(840)
        for seed in runner.SEEDS:
            for system, (arm, pair) in runner.SYSTEMS.items():
                offset = {"continuous": 0.02, "hybrid": 0.01}[arm] + (0.005 if "micro" in system else 0.)
                for scene, context in units:
                    errors = {}
                    for endpoint in family.ENDPOINTS:
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
                "schema": "issue_84_physion_scene_v1", "scene_index": scene,
                "stimulus_name": f"t{scene}", "identity": family.IDENTITY,
                "position_scale": 2.0, "velocity_scale": 1.5, "speed_tier_edges": [.06, .07],
                "kind_vocabulary": ["cube", "vase_01"], "model_names": ["cube"] * 3,
                "clip": torch.zeros(1), "windows": {}, "vertical_position": torch.zeros(1),
                "quaternion": torch.zeros(1), "angular_velocity": torch.zeros(1),
                "vertical_velocity": torch.zeros(1),
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
                self.assertIn("Phase-0 family selection", findings)
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
        _, payload = build_trial(frames=130, objects=2)
        trial = family.parse_trial("t", payload, scene_index=0)
        clip = family.clip_carriers(trial, 2.0, 1.5, ("cube",))
        stub = stub_model(clip[5:120])  # first call returns clip[5], second clip[6], ...
        with patch.object(runner, "linear_macs", lambda model, pair: 7):
            result = runner.rollout_errors(stub, clip, 5, runner.SYSTEMS["continuous_h15"][1], "cpu")
        self.assertEqual([result["errors"][str(e)]["steps"] for e in family.ENDPOINTS], [1, 2, 4, 8])
        expected = family.field_errors(clip[5], clip[5 + 15])  # first stub carrier equals clip[5]
        self.assertEqual(result["errors"]["15"]["errors"]["carrier_mse"], expected["carrier_mse"])
        self.assertGreater(result["errors"]["15"]["errors"]["carrier_mse"], 0.0)
        self.assertEqual(result["transition_calls"], 8)
        self.assertGreaterEqual(result["rollout_wall_seconds"], 0.0)
        self.assertEqual(7, result["linear_macs_per_step"])

    def test_decision_rules_use_only_the_frozen_tokens(self):
        plan = self.tiny_training_plan()
        contrasts, regime = [], []
        for kind, template in (("training_effect", "hybrid_continuous_h{h}"),):
            for horizon in family.HORIZONS:
                for endpoint in family.ENDPOINTS:
                    contrasts.append({"kind": kind, "horizon": horizon, "endpoint": endpoint,
                                      "position_mse": {"mean": .05 if (horizon, endpoint) == (1, 15) else .0,
                                                       "paired_units": 48}})
        per_seed = {str(seed): {} for seed in runner.SEEDS}
        for system, curves in (("continuous_h1", {15: .01, 30: .02, 60: .03, 120: .04}),):
            for seed in runner.SEEDS:
                per_seed[str(seed)][system] = {"curves": {str(e): {"position_mse": v}
                                                          for e, v in curves.items()}}
        # S3 requires every horizon curve; other systems needed for regime margins
        for horizon in family.HORIZONS:
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
            with patch.object(runner, "PREFIX", Path(directory) / "missing.part"):
                self.assertEqual(runner.dry_run(args), 0)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_prefix_manifest_roundtrip_and_mismatch_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefix.part"
            payloads = [build_trial("a_ok", frames=130)]
            write_archive(path, payloads)
            record = runner.manifest_record(path, path.stat().st_size)
            self.assertEqual(record["prefix_sha256"], runner.digest(path))
            runner.verify_prefix(path, record)
            record["prefix_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                runner.verify_prefix(path, record)
            with self.assertRaises(OSError):
                runner.verify_prefix(Path(directory) / "missing.part", record)


if __name__ == "__main__":
    unittest.main()
