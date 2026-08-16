import json
import hashlib
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_rollout_dataset import (
    PLAN_SCHEMA,
    ACTIVE_COHORT_ROOT,
    SCHEMA,
    CollectionOptions,
    CollectionTargets,
    LevelEntry,
    PlannedEpisode,
    PhysicsCaptureProvenance,
    _is_canonically_complete_fresh_engine_episode,
    _safe_output_name,
    build_collection_plan,
    discover_level_entries,
    generate_collection_commands,
    resolve_physics_capture_provenance,
    load_partition_manifest,
    partition_levels,
    write_config_for_manifest_level,
    write_collection_plan,
    write_partition_manifest,
)
from scripts.rollout_artifacts import validate_rollout_episode
from scripts.rollout_validation_types import EpisodeAccepted, EpisodeRejected, EpisodeValidationContract
from scripts.scenario_manifest import BenchmarkCondition, SMOKE_ONLY, import_legacy_manifest, write_manifest
from world_model.data.types import PHYSICS_CAPTURE_V1


def make_level(engine_dir: Path, novelty_level: str, level_type: str, name: str) -> None:
    path = engine_dir / "9001_Data" / "StreamingAssets" / "Levels" / novelty_level / level_type / "Levels" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<Level />", encoding="utf-8")


def make_complete_episode(output_dir: Path, options: CollectionOptions, *, bidirectional: bool = False) -> None:
    output_dir.mkdir(parents=True)
    trials = []
    for index in range(options.count):
        shot_name = f"shot_{index + 1:03d}"
        shot_dir = output_dir / shot_name
        (shot_dir / "frames").mkdir(parents=True)
        frame_path = shot_dir / "frames" / "frame_000000.png"
        frame_path.write_bytes(b"frame")
        (shot_dir / "pre_shot.png").write_bytes(b"pre")
        (shot_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "frame_count": 1,
                    "frames": [{"path": str(frame_path)}],
                }
            ),
            encoding="utf-8",
        )
        release_x = -80 if not bidirectional or index % 2 == 0 else 80
        trials.append({"shot_name": shot_name, "accepted": True, "action": {"drag_release": [release_x, 8]}})
    attempts = [
        {
            "accepted": True,
            "attempt_status": "accepted",
            "artifact_validation": {
                "accepted": True,
                "classification": "gameplay-valid",
                "retryable": False,
                "retry_decision": "accept",
            },
        }
        for _ in range(options.count)
    ]
    manifest = {
        "capture_source": "capture_desktop_rollout",
        "replay_mode": "fresh-engine-per-rollout",
        "target_fps": options.fps,
        "duration_seconds": options.duration,
        "ui_level": 1,
        "accepted_rollout_count": options.count,
        "rollout_count": options.count,
        "attempt_count": options.count,
        "attempts": attempts,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (output_dir / "action_log.json").write_text(json.dumps({"accepted_trials": trials}), encoding="utf-8")
    (output_dir / "action_log.jsonl").write_text("\n".join(json.dumps(trial) for trial in trials) + "\n", encoding="utf-8")


def make_full_level_tree(engine_dir: Path, count: int) -> list[LevelEntry]:
    for novelty in range(9):
        type_novelties = range(1, 9) if novelty == 0 else (novelty,)
        for type_novelty in type_novelties:
            for scenario in range(1, 6):
                level_type = f"type010{type_novelty}{scenario:02d}"
                for index in range(count):
                    make_level(engine_dir, f"novelty_level_{novelty}", level_type, f"level_{index:03d}.xml")
    return discover_level_entries(engine_dir)


class PrepareRolloutDatasetTest(unittest.TestCase):
    def test_partition_levels_restores_v1_stable_disjoint_bucketed_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine_dir = Path(temporary) / "engine"
            for index in range(10):
                make_level(engine_dir, "novelty_level_0", "type010101", f"level_{index:03d}.xml")
                make_level(engine_dir, "novelty_level_3", "type010303", f"level_{index:03d}.xml")

            first = partition_levels(discover_level_entries(engine_dir), seed="unit-seed")
            second = partition_levels(discover_level_entries(engine_dir), seed="unit-seed")

            self.assertEqual(first, second)
            paths = {split: {entry.relative_path for entry in entries} for split, entries in first.items()}
            self.assertFalse(paths["train"] & paths["dev"])
            self.assertFalse(paths["train"] & paths["test"])
            self.assertFalse(paths["dev"] & paths["test"])
            self.assertEqual(sum(map(len, paths.values())), 20)

    def test_capped_plan_selects_only_entries_from_their_v1_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = make_full_level_tree(root / "engine", 5)
            out_root = root / "out"
            out_root.mkdir()
            partitions = partition_levels(entries, seed="split-safe")

            episodes, _ = build_collection_plan(
                entries,
                output_root=out_root,
                options=CollectionOptions(count=2, workers=1),
                targets=CollectionTargets(train=2, dev=1),
                seed="split-safe",
            )

            for episode in episodes:
                self.assertIn(episode.entry, partitions[episode.split])
            train_paths = {episode.entry.relative_path for episode in episodes if episode.split == "train"}
            dev_paths = {episode.entry.relative_path for episode in episodes if episode.split == "dev"}
            self.assertFalse(train_paths & dev_paths)

    def test_v1_partition_manifest_loads_and_write_config_remains_split_scoped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine_dir = root / "engine"
            for index in range(6):
                make_level(engine_dir, "novelty_level_1", "type010101", f"level_{index:03d}.xml")
            partitions = partition_levels(discover_level_entries(engine_dir), seed="manifest-seed")
            manifest_path = write_partition_manifest(root / "plan", partitions, seed="manifest-seed")
            config_path = root / "config.xml"
            dev_level = partitions["dev"][0].relative_path

            self.assertEqual(load_partition_manifest(manifest_path), partitions)
            self.assertEqual(write_config_for_manifest_level(manifest_path, "dev", dev_level, config_path), config_path)
            self.assertIn(dev_level, config_path.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "not part of split train"):
                write_config_for_manifest_level(manifest_path, "train", dev_level, config_path)

    def test_generated_commands_preserve_runtime_display_library_and_timeouts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_root = root / "out"
            out_root.mkdir()
            entry = LevelEntry("novelty_level_1", "type010101", "levels/one.xml")
            episode = PlannedEpisode("train", entry, out_root / "train" / _safe_output_name(entry), "scheduled")
            plan_path = write_collection_plan(root / "plan", output_root=out_root, episodes=[episode], summary={}, options=CollectionOptions(workers=1), targets=CollectionTargets(train=1, dev=1), seed="commands", collection_purpose="smoke")

            commands = generate_collection_commands(plan_path, output_root=out_root, options=CollectionOptions(workers=1))

            self.assertIn('DISPLAY="$display_id" LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH-}"', commands)
            self.assertIn("--ui-settle-seconds 5", commands)
            self.assertIn("--connect-timeout 60", commands)
            self.assertIn("--prepare-timeout 90", commands)
            self.assertIn("--read-timeout 420", commands)
            self.assertIn("--speed 1", commands)

    def test_collection_options_preserve_runtime_defaults(self):
        self.assertEqual(CollectionOptions().ui_settle_seconds, 5.0)
        self.assertEqual(CollectionOptions().connect_timeout, 60.0)
        self.assertEqual(CollectionOptions().prepare_timeout, 90.0)
        self.assertEqual(CollectionOptions().read_timeout, 420.0)
        self.assertEqual(CollectionOptions().speed, 1)

    def test_partition_manifest_uses_v1_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = LevelEntry("novelty_level_1", "type010101", "levels/one.xml")

            manifest_path = write_partition_manifest(root, {"train": [entry], "dev": [], "test": []})

            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8"))["schema"], SCHEMA)

    def test_partition_levels_preserve_a_test_holdout_when_bucket_capacity_allows(self):
        entries = [LevelEntry("novelty_level_1", "type010101", f"levels/{index}.xml") for index in range(10)]

        partitions = partition_levels(entries)

        self.assertTrue(partitions["test"])

    def test_collection_targets_default_to_capped_contract(self):
        self.assertEqual(CollectionTargets(), CollectionTargets(train=100, dev=20, test=0))
        self.assertEqual(CollectionOptions(), CollectionOptions(count=12, fps=30.0, duration=5.0, workers=6))

    def test_default_collection_plan_selects_train_and_dev_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = make_full_level_tree(root / "engine", 5)
            output_root = root / "out"
            output_root.mkdir()

            episodes, _ = build_collection_plan(
                entries,
                output_root=output_root,
                options=CollectionOptions(count=2, workers=1),
                targets=CollectionTargets(train=1, dev=1),
                seed="default-selection",
            )

            self.assertEqual({episode.split for episode in episodes}, {"train", "dev"})

    def test_test_collection_requires_explicit_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine_dir = root / "engine"
            make_full_level_tree(engine_dir, 10)
            output_root = root / "out"
            output_root.mkdir()
            script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_rollout_dataset.py"
            base_command = [
                "python",
                str(script),
                "plan",
                "--engine-dir",
                str(engine_dir),
                "--command-output-root",
                str(output_root),
                "--train-target",
                "1",
                "--dev-target",
                "1",
                "--test-target",
                "1",
                "--workers",
                "1",
            ]

            without_flag = subprocess.run(
                [*base_command, "--output-dir", str(root / "without-flag")],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(without_flag.returncode, 0, without_flag.stderr)
            without_payload = json.loads((root / "without-flag" / "collection_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(without_payload["selected_splits"], ["train", "dev"])
            self.assertEqual(without_payload["counts"]["test"], 0)

            invalid_opt_in = subprocess.run(
                [
                    "python",
                    str(script),
                    "plan",
                    "--engine-dir",
                    str(engine_dir),
                    "--command-output-root",
                    str(output_root),
                    "--include-test",
                    "--test-target",
                    "0",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(invalid_opt_in.returncode, 0)
            self.assertIn("--include-test requires --test-target >= 1", invalid_opt_in.stderr)

    def test_replanning_removes_the_opposite_mode_collection_script(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine_dir = root / "engine"
            make_full_level_tree(engine_dir, 10)
            output_root = root / "out"
            output_root.mkdir()
            plan_dir = root / "plan"
            script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_rollout_dataset.py"
            common = [
                "python",
                str(script),
                "plan",
                "--engine-dir",
                str(engine_dir),
                "--output-dir",
                str(plan_dir),
                "--command-output-root",
                str(output_root),
                "--train-target",
                "1",
                "--dev-target",
                "1",
                "--test-target",
                "1",
                "--workers",
                "1",
            ]
            default_script = plan_dir / "collect_train_dev.sh"
            test_script = plan_dir / "collect_train_dev_test.sh"

            first_opt_in = subprocess.run([*common, "--include-test"], text=True, capture_output=True, check=False)
            self.assertEqual(first_opt_in.returncode, 0, first_opt_in.stderr)
            self.assertTrue(test_script.is_file())
            self.assertIn("--split test", test_script.read_text(encoding="utf-8"))

            default = subprocess.run(common, text=True, capture_output=True, check=False)
            self.assertEqual(default.returncode, 0, default.stderr)
            self.assertTrue(default_script.is_file())
            self.assertFalse(test_script.exists())
            self.assertNotIn("--split test", default_script.read_text(encoding="utf-8"))
            self.assertEqual(json.loads((plan_dir / "collection_plan.json").read_text(encoding="utf-8"))["selected_splits"], ["train", "dev"])

            second_opt_in = subprocess.run([*common, "--include-test"], text=True, capture_output=True, check=False)
            self.assertEqual(second_opt_in.returncode, 0, second_opt_in.stderr)
            self.assertTrue(test_script.is_file())
            self.assertFalse(default_script.exists())
            self.assertIn("--split test", test_script.read_text(encoding="utf-8"))
            self.assertEqual(json.loads((plan_dir / "collection_plan.json").read_text(encoding="utf-8"))["selected_splits"], ["train", "dev", "test"])

    def test_opt_in_test_collection_preserves_disjoint_source_partitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = make_full_level_tree(root / "engine", 10)
            output_root = root / "out"
            output_root.mkdir()
            targets = CollectionTargets(train=1, dev=1, test=1)
            selected_splits = ("train", "dev", "test")

            episodes, summary = build_collection_plan(
                entries,
                output_root=output_root,
                options=CollectionOptions(count=2, workers=1),
                targets=targets,
                selected_splits=selected_splits,
                seed="test-opt-in",
            )
            partitions = partition_levels(entries, seed="test-opt-in")
            selected_paths = {
                split: {episode.entry.relative_path for episode in episodes if episode.split == split}
                for split in selected_splits
            }
            plan_path = write_collection_plan(
                root / "plan",
                output_root=output_root,
                episodes=episodes,
                summary=summary,
                options=CollectionOptions(count=2, workers=1),
                targets=targets,
                selected_splits=selected_splits,
                seed="test-opt-in",
            )
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            test_level = next(episode.entry.relative_path for episode in episodes if episode.split == "test")

            self.assertTrue(selected_paths["test"])
            self.assertTrue(selected_paths["train"].isdisjoint(selected_paths["dev"]))
            self.assertTrue(selected_paths["train"].isdisjoint(selected_paths["test"]))
            self.assertTrue(selected_paths["dev"].isdisjoint(selected_paths["test"]))
            for split in selected_splits:
                self.assertTrue(selected_paths[split].issubset({entry.relative_path for entry in partitions[split]}))
            self.assertEqual(payload["selected_splits"], ["train", "dev", "test"])
            self.assertEqual(payload["contract"]["test_target"], 1)
            self.assertIn("--split test", generate_collection_commands(plan_path, output_root=output_root, options=CollectionOptions(count=2, workers=1)))
            config_path = root / "config.xml"
            self.assertEqual(write_config_for_manifest_level(plan_path, "test", test_level, config_path), config_path)

    def test_test_collection_fails_without_capacity_preserving_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = make_full_level_tree(root / "engine", 20)
            output_root = root / "out"
            output_root.mkdir()
            seed = "test-capacity"
            blocked_entry = next(
                entry
                for entry in partition_levels(entries, seed=seed)["test"]
                if entry.bucket == "novelty_level_1/type010101"
            )
            blocked_output = output_root / "test" / _safe_output_name(blocked_entry)
            blocked_output.mkdir(parents=True)
            sentinel = blocked_output / "sentinel"
            sentinel.write_text("preserve", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "insufficient safe absent capacity"):
                build_collection_plan(
                    entries,
                    output_root=output_root,
                    options=CollectionOptions(count=2, workers=1),
                    targets=CollectionTargets(train=1, dev=1, test=2),
                    selected_splits=("test",),
                    seed=seed,
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_plan_selects_existing_then_absent_to_fill_every_normal_and_novel_bucket(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = make_full_level_tree(root / "engine", 4)
            out_root = root / "out"
            out_root.mkdir()
            options = CollectionOptions(count=2, workers=1)
            existing_entry = next(
                entry
                for entry in partition_levels(entries, seed="test")["train"]
                if entry.bucket == "novelty_level_0/type010101"
            )
            make_complete_episode(out_root / "train" / _safe_output_name(existing_entry), options)

            episodes, summary = build_collection_plan(entries, output_root=out_root, options=options, targets=CollectionTargets(train=2, dev=1), seed="test")

            self.assertEqual(len(episodes), 80 * 3)
            self.assertEqual(sum(item.entry.novelty_level == "novelty_level_0" for item in episodes), 40 * 3)
            self.assertEqual(sum(item.entry.novelty_level != "novelty_level_0" for item in episodes), 40 * 3)
            self.assertEqual(summary["train:novelty_level_0/type010101"], {"target": 2, "existing": 1, "scheduled": 1})
            self.assertEqual(sum(item.source == "scheduled" for item in episodes), 239)

    def test_plan_is_deterministic_and_never_writes_output_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = make_full_level_tree(root / "engine", 4)
            out_root = root / "out"
            out_root.mkdir()
            before = sorted(path.relative_to(out_root).as_posix() for path in out_root.rglob("*"))

            first, _ = build_collection_plan(entries, output_root=out_root, options=CollectionOptions(count=2, workers=1), targets=CollectionTargets(train=2, dev=1), seed="stable")
            second, _ = build_collection_plan(entries, output_root=out_root, options=CollectionOptions(count=2, workers=1), targets=CollectionTargets(train=2, dev=1), seed="stable")

            self.assertEqual(first, second)
            self.assertEqual(before, sorted(path.relative_to(out_root).as_posix() for path in out_root.rglob("*")))

    def test_plan_skips_occupied_symlinked_incomplete_and_escaping_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = make_full_level_tree(root / "engine", 10)
            out_root = root / "out"
            out_root.mkdir()
            bucket_entries = [
                entry
                for entry in partition_levels(entries, seed="capacity")["train"]
                if entry.bucket == "novelty_level_1/type010101"
            ]
            occupied = out_root / "train" / _safe_output_name(bucket_entries[0])
            occupied.mkdir(parents=True)
            (occupied / "sentinel").write_text("preserve", encoding="utf-8")
            symlink = out_root / "train" / _safe_output_name(bucket_entries[1])
            symlink.symlink_to(root / "outside", target_is_directory=True)
            incomplete = out_root / "train" / _safe_output_name(bucket_entries[2])
            incomplete.mkdir()
            (incomplete / "manifest.json").write_text("{}", encoding="utf-8")

            episodes, _ = build_collection_plan(entries, output_root=out_root, options=CollectionOptions(count=2, workers=1), targets=CollectionTargets(train=2, dev=1), seed="capacity")

            chosen = [item.output_dir for item in episodes if item.split == "train" and item.entry.bucket == "novelty_level_1/type010101"]
            self.assertNotIn(occupied, chosen)
            self.assertNotIn(symlink, chosen)
            self.assertNotIn(incomplete, chosen)
            self.assertEqual((occupied / "sentinel").read_text(encoding="utf-8"), "preserve")

    def test_plan_fails_when_a_bucket_has_no_absent_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = make_full_level_tree(root / "engine", 3)
            out_root = root / "out"
            out_root.mkdir()
            for entry in (item for item in entries if item.bucket == "novelty_level_1/type010101"):
                path = out_root / "train" / _safe_output_name(entry)
                path.mkdir(parents=True)

            with self.assertRaisesRegex(RuntimeError, "insufficient safe absent capacity"):
                build_collection_plan(entries, output_root=out_root, options=CollectionOptions(count=2, workers=1), targets=CollectionTargets(train=3, dev=1), seed="capacity")

    def test_canonical_existing_requires_raw_artifacts_and_level_five_action_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = CollectionOptions(count=2, workers=1)
            complete = root / "complete"
            make_complete_episode(complete, options, bidirectional=True)
            self.assertTrue(_is_canonically_complete_fresh_engine_episode(complete, options, level_five=True))
            self.assertTrue(_is_canonically_complete_fresh_engine_episode(complete, options, level_five=False))
            (complete / "shot_001" / "frames" / "frame_000000.png").unlink()
            self.assertFalse(_is_canonically_complete_fresh_engine_episode(complete, options, level_five=True))
            make_complete_episode(root / "legacy", options, bidirectional=False)
            self.assertFalse(_is_canonically_complete_fresh_engine_episode(root / "legacy", options, level_five=True))

    def test_canonical_existing_rejects_symlinked_or_incomplete_raw_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = CollectionOptions(count=2, workers=1)
            external = root / "external"
            make_complete_episode(external, options, bidirectional=True)

            def complete_case(name: str) -> Path:
                episode = root / name
                make_complete_episode(episode, options, bidirectional=True)
                return episode

            symlink_cases = {
                "shot directory": lambda episode: (
                    shutil.rmtree(episode / "shot_001"),
                    shutil.copytree(external / "shot_001", episode / "inside-shot"),
                    (episode / "inside-shot" / "metadata.json").write_text(
                        json.dumps(
                            {
                                "frame_count": 1,
                                "frames": [
                                    {"path": str(episode / "shot_001" / "frames" / "frame_000000.png")}
                                ],
                            }
                        ),
                        encoding="utf-8",
                    ),
                    (episode / "shot_001").symlink_to(episode / "inside-shot", target_is_directory=True),
                ),
                "metadata": lambda episode: (
                    (episode / "shot_001" / "metadata.json").unlink(),
                    (episode / "shot_001" / "metadata.json").symlink_to(external / "shot_001" / "metadata.json"),
                ),
                "frames directory": lambda episode: (
                    shutil.rmtree(episode / "shot_001" / "frames"),
                    (episode / "shot_001" / "frames").symlink_to(external / "shot_001" / "frames", target_is_directory=True),
                ),
                "frame": lambda episode: (
                    (episode / "shot_001" / "frames" / "frame_000000.png").unlink(),
                    (episode / "shot_001" / "frames" / "frame_000000.png").symlink_to(external / "shot_001" / "frames" / "frame_000000.png"),
                ),
                "pre-shot image": lambda episode: (
                    (episode / "shot_001" / "pre_shot.png").unlink(),
                    (episode / "shot_001" / "pre_shot.png").symlink_to(external / "shot_001" / "pre_shot.png"),
                ),
            }
            for name, corrupt in symlink_cases.items():
                with self.subTest(name=name):
                    episode = complete_case(name.replace(" ", "-"))
                    corrupt(episode)
                    self.assertFalse(_is_canonically_complete_fresh_engine_episode(episode, options, level_five=True))

            incomplete = complete_case("incomplete")
            metadata_path = incomplete / "shot_001" / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["frame_count"] = 2
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            self.assertFalse(_is_canonically_complete_fresh_engine_episode(incomplete, options, level_five=True))

            inconsistent = complete_case("inconsistent-frame-list")
            metadata_path = inconsistent / "shot_001" / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["frames"][0]["path"] = str(inconsistent / "shot_001" / "pre_shot.png")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            self.assertFalse(_is_canonically_complete_fresh_engine_episode(inconsistent, options, level_five=True))

    def test_canonical_existing_rejects_unreadable_raw_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = CollectionOptions(count=2, workers=1)
            episode = root / "complete"
            metadata_path = episode / "shot_001" / "metadata.json"
            make_complete_episode(episode, options, bidirectional=True)
            access = os.access

            with patch("scripts.rollout_artifacts.os.access", side_effect=lambda path, mode: False if Path(path) == metadata_path else access(path, mode)):
                self.assertFalse(_is_canonically_complete_fresh_engine_episode(episode, options, level_five=True))

    def test_plan_preserves_unsafe_existing_episode_and_schedules_later_absent_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = make_full_level_tree(root / "engine", 5)
            out_root = root / "out"
            out_root.mkdir()
            options = CollectionOptions(count=2, workers=1)
            bucket_entries = [
                entry
                for entry in partition_levels(entries, seed="unsafe-raw") ["train"]
                if entry.bucket == "novelty_level_1/type010101"
            ]
            unsafe = out_root / "train" / _safe_output_name(bucket_entries[0])
            make_complete_episode(unsafe, options)
            pre_shot = unsafe / "shot_001" / "pre_shot.png"
            preserved_target = root / "preserved-pre-shot.png"
            preserved_target.write_bytes(b"preserve")
            pre_shot.unlink()
            pre_shot.symlink_to(preserved_target)

            episodes, summary = build_collection_plan(
                entries,
                output_root=out_root,
                options=options,
                targets=CollectionTargets(train=1, dev=1),
                seed="unsafe-raw",
            )
            selected = [
                episode
                for episode in episodes
                if episode.split == "train" and episode.entry.bucket == "novelty_level_1/type010101"
            ]

            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0].source, "scheduled")
            self.assertNotEqual(selected[0].output_dir, unsafe)
            self.assertTrue(pre_shot.is_symlink())
            self.assertEqual(preserved_target.read_bytes(), b"preserve")
            self.assertEqual(summary["train:novelty_level_1/type010101"], {"target": 1, "existing": 0, "scheduled": 1})

    def test_level_five_policy_incompatible_existing_is_surplus_and_later_absent_path_is_scheduled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = make_full_level_tree(root / "engine", 5)
            out_root = root / "out"
            out_root.mkdir()
            options = CollectionOptions(count=2, workers=1)
            level_five = [
                entry
                for entry in partition_levels(entries, seed="level-five")["train"]
                if entry.bucket == "novelty_level_5/type010501"
            ]
            legacy = out_root / "train" / _safe_output_name(level_five[0])
            make_complete_episode(legacy, options, bidirectional=False)
            legacy_log = legacy.joinpath("action_log.json").read_text(encoding="utf-8")

            episodes, _ = build_collection_plan(entries, output_root=out_root, options=options, targets=CollectionTargets(train=2, dev=1), seed="level-five")
            chosen = [item.output_dir for item in episodes if item.split == "train" and item.entry.bucket == "novelty_level_5/type010501"]

            self.assertNotIn(legacy, chosen)
            self.assertEqual(legacy.joinpath("action_log.json").read_text(encoding="utf-8"), legacy_log)

    def test_plan_artifact_is_atomic_and_contains_reconciled_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = LevelEntry("novelty_level_1", "type010101", "levels/one.xml")
            out_root = root / "out"
            out_root.mkdir()
            episode = PlannedEpisode("train", entry, out_root / "train" / _safe_output_name(entry), "scheduled")
            plan_path = write_collection_plan(root / "plan", output_root=out_root, episodes=[episode], summary={"train:novelty_level_1/type010101": {"target": 1, "existing": 0, "scheduled": 1}}, options=CollectionOptions(count=12, workers=1), targets=CollectionTargets(train=1, dev=1), seed="artifact", collection_purpose="smoke")

            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], PLAN_SCHEMA)
            self.assertEqual(payload["counts"], {"existing": 0, "scheduled": 1, "selected": 1, "normal": 0, "novel": 1, "test": 0})
            self.assertFalse(any(path.name.startswith("tmp") for path in plan_path.parent.iterdir()))

    def test_generated_commands_schedule_only_absent_paths_with_reservation_and_locked_ledger(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_root = root / "out"
            out_root.mkdir()
            entry = LevelEntry("novelty_level_5", "type010501", "levels/one.xml")
            planned = PlannedEpisode("train", entry, out_root / "train" / _safe_output_name(entry), "scheduled")
            retained = PlannedEpisode("train", entry, out_root / "train" / "already-complete", "existing")
            plan_path = write_collection_plan(root / "plan", output_root=out_root, episodes=[planned, retained], summary={}, options=CollectionOptions(count=12, workers=1), targets=CollectionTargets(train=1, dev=1), seed="commands", collection_purpose="smoke")

            commands = generate_collection_commands(plan_path, output_root=out_root, options=CollectionOptions(count=12, workers=1))

            self.assertIn("mkdir --", commands)
            self.assertIn("flock -x", commands)
            self.assertIn('if [[ "$failure_count" -gt 0 ]]; then return 1; fi', commands)
            self.assertIn("--bidirectional-launches", commands)
            self.assertNotIn("already-complete", commands)
            self.assertNotIn("--split test", commands)
            script = root / "collect.sh"
            script.write_text(commands, encoding="utf-8")
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_generated_commands_reenter_the_generating_repo_root_after_sourcing_the_profile(self):
        """`source ~/cd_novphy` chdirs to the NovPhy checkout and repoints PYTHONPATH.

        Every later relative path (`scripts/collect_rollouts.py`, the plan artifact,
        `data/...`) must still resolve against the repo the plan was generated in, or a
        worktree launch silently runs the other checkout's collector.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_root = root / "out"
            out_root.mkdir()
            entry = LevelEntry("novelty_level_1", "type010101", "levels/one.xml")
            episode = PlannedEpisode("train", entry, out_root / "train" / _safe_output_name(entry), "scheduled")
            plan_path = write_collection_plan(root / "plan", output_root=out_root, episodes=[episode], summary={}, options=CollectionOptions(workers=1), targets=CollectionTargets(train=1, dev=1), seed="cwd", collection_purpose="smoke")

            commands = generate_collection_commands(plan_path, output_root=out_root, options=CollectionOptions(workers=1))

            lines = commands.splitlines()
            source_index = lines.index("source ~/cd_novphy")
            repo_root = Path.cwd()
            self.assertEqual(lines[source_index + 1], f"cd -- {shlex.quote(str(repo_root))}")
            self.assertEqual(lines[source_index + 2], 'export PYTHONPATH="$PWD"')
            self.assertLess(source_index, lines.index(f"plan_artifact={shlex.quote(str(plan_path))}"))
            script = root / "collect.sh"
            script.write_text(commands, encoding="utf-8")
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_generated_commands_resolve_relative_paths_from_the_generating_root(self):
        """Executing the emitted prologue from an unrelated CWD must land in the repo root."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_root = root / "out"
            out_root.mkdir()
            entry = LevelEntry("novelty_level_1", "type010101", "levels/one.xml")
            episode = PlannedEpisode("train", entry, out_root / "train" / _safe_output_name(entry), "scheduled")
            plan_path = write_collection_plan(root / "plan", output_root=out_root, episodes=[episode], summary={}, options=CollectionOptions(workers=1), targets=CollectionTargets(train=1, dev=1), seed="cwd-exec", collection_purpose="smoke")

            commands = generate_collection_commands(plan_path, output_root=out_root, options=CollectionOptions(workers=1))

            prologue = [
                line
                for line in commands.splitlines()[:6]
                if line.startswith("cd -- ") or line.startswith("export PYTHONPATH=")
            ]
            probe = root / "probe.sh"
            probe.write_text("\n".join(["set -euo pipefail", *prologue, 'printf "%s\\n" "$PWD" "$PYTHONPATH"']) + "\n", encoding="utf-8")
            completed = subprocess.run(["bash", str(probe)], cwd=root, check=True, capture_output=True, text=True)
            observed_cwd, observed_pythonpath = completed.stdout.split()
            self.assertEqual(Path(observed_cwd), Path.cwd())
            self.assertEqual(Path(observed_pythonpath), Path.cwd())

    def test_scoped_inventory_plans_fewer_buckets_while_production_still_requires_eighty(self):
        """A declared scope lets the single-level physics player be planned.

        The 80-bucket invariant is what protects a production run from a truncated level
        inventory, so it must still fire whenever no scope is declared.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine_dir = root / "engine"
            out_root = root / "out"
            out_root.mkdir()
            make_level(engine_dir, "novelty_level_0", "type010101", "only.xml")
            entries = discover_level_entries(engine_dir)

            with self.assertRaisesRegex(RuntimeError, "Expected 80 normal and novel buckets"):
                build_collection_plan(entries, output_root=out_root, options=CollectionOptions(count=1, workers=1), targets=CollectionTargets(train=1, dev=1), selected_splits=("train",))

            plan, summary = build_collection_plan(
                entries,
                output_root=out_root,
                options=CollectionOptions(count=1, workers=1),
                targets=CollectionTargets(train=1, dev=1),
                selected_splits=("train",),
                expected_bucket_count=1,
            )

            self.assertEqual(len(plan), 1)
            self.assertEqual(plan[0].split, "train")
            self.assertEqual(set(summary), {"train:novelty_level_0/type010101"})

    def test_scoped_inventory_rejects_a_scope_that_disagrees_with_the_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine_dir = root / "engine"
            out_root = root / "out"
            out_root.mkdir()
            make_level(engine_dir, "novelty_level_0", "type010101", "only.xml")
            entries = discover_level_entries(engine_dir)

            with self.assertRaisesRegex(RuntimeError, "Expected 3 normal and novel buckets"):
                build_collection_plan(entries, output_root=out_root, options=CollectionOptions(count=1, workers=1), targets=CollectionTargets(train=1, dev=1), selected_splits=("train",), expected_bucket_count=3)

    def test_scoped_inventory_rejects_a_nonpositive_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine_dir = root / "engine"
            out_root = root / "out"
            out_root.mkdir()
            make_level(engine_dir, "novelty_level_0", "type010101", "only.xml")
            entries = discover_level_entries(engine_dir)

            with self.assertRaises(ValueError):
                build_collection_plan(entries, output_root=out_root, options=CollectionOptions(count=1, workers=1), targets=CollectionTargets(train=1, dev=1), selected_splits=("train",), expected_bucket_count=0)

    def test_discovery_can_target_a_declared_level_type_prefix(self):
        """The staged physics player ships its level under `type2`, not `type010*`.

        Discovery must stay pinned to the production prefix by default so a truncated
        or foreign level tree can never be planned by accident.
        """
        with tempfile.TemporaryDirectory() as temporary:
            engine_dir = Path(temporary) / "engine"
            make_level(engine_dir, "novelty_level_0", "type2", "3_9_6_1.xml")
            make_level(engine_dir, "novelty_level_0", "type010101", "production.xml")

            default_entries = discover_level_entries(engine_dir)
            self.assertEqual([entry.level_type for entry in default_entries], ["type010101"])

            scoped = discover_level_entries(engine_dir, level_type_prefix="type2")
            self.assertEqual([entry.level_type for entry in scoped], ["type2"])
            self.assertEqual(scoped[0].bucket, "novelty_level_0/type2")
            self.assertTrue(scoped[0].relative_path.endswith("type2/Levels/3_9_6_1.xml"))

    def test_manifest_lineage_survives_partition_and_collection_plan_round_trips(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine_dir = root / "engine"
            output_root = root / "out"
            output_root.mkdir()
            make_level(engine_dir, "novelty_level_1", "type010101", "one.xml")
            entries = discover_level_entries(engine_dir)

            self.assertEqual(entries[0].scenario_manifest.generation.mode, "legacy_static")
            self.assertIsNone(entries[0].scenario_manifest.generation.generation_seed)
            partitions = partition_levels(entries, seed="planner-seed")
            partition_path = write_partition_manifest(root / "partition", partitions, seed="planner-seed")
            self.assertEqual(load_partition_manifest(partition_path), partitions)

            episodes, summary = build_collection_plan(
                entries,
                output_root=output_root,
                options=CollectionOptions(count=1, workers=1),
                targets=CollectionTargets(train=1, dev=1),
                selected_splits=("train",),
                seed="planner-seed",
                expected_bucket_count=1,
            )
            plan_path = write_collection_plan(
                root / "plan",
                output_root=output_root,
                episodes=episodes,
                summary=summary,
                options=CollectionOptions(count=1, workers=1),
                targets=CollectionTargets(train=1, dev=1),
                seed="planner-seed",
            )
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            selected = payload["selected"][0]
            self.assertEqual(selected["scenario_lineage_identity"], entries[0].scenario_manifest.scenario_lineage.identity)
            self.assertEqual(selected["generation_mode"], "legacy_static")
            self.assertEqual(payload["planner_seed"], "planner-seed")
            self.assertIsNone(selected["generation_seed"])
            generate_collection_commands(plan_path, output_root=output_root, options=CollectionOptions(count=1, workers=1))

            del selected["scenario_manifest"]
            plan_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scenario_manifest"):
                generate_collection_commands(plan_path, output_root=output_root, options=CollectionOptions(count=1, workers=1))

    def test_smoke_only_type2_is_rejected_for_research_but_can_be_smoke_planned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine_dir = root / "engine"
            output_root = root / "out"
            output_root.mkdir()
            make_level(engine_dir, "novelty_level_0", "type2", "3_9_6_1.xml")
            xml_path = engine_dir / "9001_Data/StreamingAssets/Levels/novelty_level_0/type2/Levels/3_9_6_1.xml"
            manifest = import_legacy_manifest(
                xml_path.read_bytes(),
                benchmark_condition=BenchmarkCondition("novelty_level_0", "type2"),
                source_path=xml_path.relative_to(engine_dir).as_posix(),
                eligibility=SMOKE_ONLY,
                eligibility_reason="staged type2 runtime fixture",
            )
            write_manifest(manifest, xml_path.with_suffix(".scenario.json"))
            entries = discover_level_entries(engine_dir, level_type_prefix="type2")

            with self.assertRaisesRegex(ValueError, "smoke_only"):
                build_collection_plan(
                    entries,
                    output_root=output_root,
                    options=CollectionOptions(count=1, workers=1),
                    targets=CollectionTargets(train=1, dev=1),
                    selected_splits=("train",),
                    expected_bucket_count=1,
                )
            episodes, summary = build_collection_plan(
                entries,
                output_root=output_root,
                options=CollectionOptions(count=1, workers=1),
                targets=CollectionTargets(train=1, dev=1),
                selected_splits=("train",),
                expected_bucket_count=1,
                collection_purpose="smoke",
            )
            self.assertEqual(len(episodes), 1)
            plan_path = write_collection_plan(
                root / "smoke-plan",
                output_root=output_root,
                episodes=episodes,
                summary=summary,
                options=CollectionOptions(count=1, workers=1),
                targets=CollectionTargets(train=1, dev=1),
                selected_splits=("train",),
                seed="smoke-planner-seed",
                collection_purpose="smoke",
            )
            self.assertEqual(json.loads(plan_path.read_text(encoding="utf-8"))["collection_purpose"], "smoke")
            generate_collection_commands(plan_path, output_root=output_root, options=CollectionOptions(count=1, workers=1))

    def test_sidecarless_type2_is_smoke_only_and_rejected_for_research(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine_dir = root / "engine"
            output_root = root / "out"
            output_root.mkdir()
            make_level(engine_dir, "novelty_level_0", "type2", "3_9_6_1.xml")
            entries = discover_level_entries(engine_dir, level_type_prefix="type2")

            self.assertEqual(entries[0].scenario_manifest.research_eligibility.status, SMOKE_ONLY)
            with self.assertRaisesRegex(ValueError, "smoke_only"):
                build_collection_plan(
                    entries,
                    output_root=output_root,
                    options=CollectionOptions(count=1, workers=1),
                    targets=CollectionTargets(train=1, dev=1),
                    selected_splits=("train",),
                    expected_bucket_count=1,
                )
            planned, _ = build_collection_plan(
                entries,
                output_root=output_root,
                options=CollectionOptions(count=1, workers=1),
                targets=CollectionTargets(train=1, dev=1),
                selected_splits=("train",),
                expected_bucket_count=1,
                collection_purpose="smoke",
            )
            self.assertEqual(len(planned), 1)

    def test_path_only_level_entry_is_rejected_for_research_at_every_planning_seam(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "out"
            output_root.mkdir()
            entry = LevelEntry("novelty_level_0", "type2", "levels/type2.xml")
            options = CollectionOptions(count=1, workers=1)
            targets = CollectionTargets(train=1, dev=1)

            with self.assertRaisesRegex(ValueError, "scenario manifest"):
                build_collection_plan(
                    [entry],
                    output_root=output_root,
                    options=options,
                    targets=targets,
                    selected_splits=("train",),
                    expected_bucket_count=1,
                )

            planned = PlannedEpisode("train", entry, output_root / "train" / "type2", "scheduled")
            with self.assertRaisesRegex(ValueError, "scenario manifest"):
                write_collection_plan(
                    root / "research-plan",
                    output_root=output_root,
                    episodes=[planned],
                    summary={},
                    options=options,
                    targets=targets,
                    selected_splits=("train",),
                    seed="planner-seed",
                )

            smoke_path = write_collection_plan(
                root / "smoke-plan",
                output_root=output_root,
                episodes=[planned],
                summary={},
                options=options,
                targets=targets,
                selected_splits=("train",),
                seed="planner-seed",
                collection_purpose="smoke",
            )
            payload = json.loads(smoke_path.read_text(encoding="utf-8"))
            payload["collection_purpose"] = "research"
            smoke_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scenario manifest"):
                generate_collection_commands(smoke_path, output_root=output_root, options=options)

    def test_discovery_rejects_an_empty_level_type_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine_dir = Path(temporary) / "engine"
            make_level(engine_dir, "novelty_level_0", "type2", "3_9_6_1.xml")

            with self.assertRaises(ValueError):
                discover_level_entries(engine_dir, level_type_prefix="  ")

    def test_single_level_inventory_partitions_to_train_only(self):
        """One level cannot fund a dev or test split without source-level leakage."""
        with tempfile.TemporaryDirectory() as temporary:
            engine_dir = Path(temporary) / "engine"
            make_level(engine_dir, "novelty_level_0", "type2", "3_9_6_1.xml")
            entries = discover_level_entries(engine_dir, level_type_prefix="type2")

            partitions = partition_levels(entries)

            self.assertEqual(len(partitions["train"]), 1)
            self.assertEqual(partitions["dev"], [])
            self.assertEqual(partitions["test"], [])

    def test_train_only_plan_succeeds_where_train_dev_has_no_dev_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine_dir = root / "engine"
            out_root = root / "out"
            out_root.mkdir()
            make_level(engine_dir, "novelty_level_0", "type2", "3_9_6_1.xml")
            entries = discover_level_entries(engine_dir, level_type_prefix="type2")
            options = CollectionOptions(count=1, workers=1)
            targets = CollectionTargets(train=1, dev=1)

            with self.assertRaisesRegex(RuntimeError, "no dev partition capacity"):
                build_collection_plan(entries, output_root=out_root, options=options, targets=targets, selected_splits=("train", "dev"), expected_bucket_count=1, collection_purpose="smoke")

            plan, summary = build_collection_plan(entries, output_root=out_root, options=options, targets=targets, selected_splits=("train",), expected_bucket_count=1, collection_purpose="smoke")

            self.assertEqual([episode.split for episode in plan], ["train"])
            self.assertEqual(set(summary), {"train:novelty_level_0/type2"})

    def test_generated_schedule_interleaves_normal_and_novelty_then_stripes_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_root = root / "out"
            out_root.mkdir()
            episodes = [
                PlannedEpisode("train", LevelEntry(f"novelty_level_{novelty}", "type010101", f"levels/{novelty}.xml"), out_root / "train" / f"episode-{novelty}", "scheduled")
                for novelty in (0, 1, 2, 3)
            ]
            plan_path = write_collection_plan(root / "plan", output_root=out_root, episodes=episodes, summary={}, options=CollectionOptions(count=12, workers=2), targets=CollectionTargets(train=1, dev=1), seed="stripe", collection_purpose="smoke")
            commands = generate_collection_commands(plan_path, output_root=out_root, options=CollectionOptions(count=12, workers=2))

            worker_zero, worker_one = commands.split('if [[ "$worker_index" == "1" ]]; then', maxsplit=1)
            self.assertIn("levels/0.xml", worker_zero)
            self.assertIn("levels/2.xml", worker_zero)
            self.assertIn("levels/1.xml", worker_one)
            self.assertIn("levels/3.xml", worker_one)
            script = root / "collect.sh"
            script.write_text(commands, encoding="utf-8")
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_launcher_requires_resume_existing_root_and_forwards_capped_contract_before_xvnc(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            (fake_bin / "python").write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >&2\nexit 42\n", encoding="utf-8")
            (fake_bin / "python").chmod(0o755)
            out_root = root / "out"
            out_root.mkdir()
            proc_root = root / "proc"
            (proc_root / "net").mkdir(parents=True)
            header = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
            (proc_root / "net" / "tcp").write_text(header, encoding="utf-8")
            (proc_root / "net" / "tcp6").write_text(header, encoding="utf-8")
            (root / "x11" / ".X11-unix").mkdir(parents=True)
            env = os.environ | {"OUT_ROOT": str(out_root), "RESUME": "1", "PLAN_DIR": str(root / "plan"), "PARTITION_SEED": "operator-seed", "TRAIN_TARGET_PER_BUCKET": "7", "DEV_TARGET_PER_BUCKET": "3", "TEST_TARGET_PER_BUCKET": "5", "NOVPHY_YES": "1", "NOVPHY_ALLOW_NETWORK_LISTENERS": "1", "NOVPHY_PROC_ROOT": str(proc_root), "NOVPHY_X11_TMP_ROOT": str(root / "x11"), "PATH": f"{fake_bin}:{os.environ['PATH']}"}

            result = subprocess.run(["bash", "scripts/collect_full_rollout_training_dataset.sh"], cwd=Path(__file__).resolve().parents[1], env=env, text=True, capture_output=True, check=False)

            self.assertEqual(result.returncode, 42)
            self.assertIn("--train-target 7", result.stderr)
            self.assertIn("--dev-target 3", result.stderr)
            self.assertIn("--seed operator-seed", result.stderr)
            self.assertNotIn("--test-target", result.stderr)
            self.assertLess(result.stderr.index("prepare_rollout_dataset.py plan"), result.stderr.index("--train-target 7"))


class PhysicsLauncherTests(unittest.TestCase):
    def _provenance(self, root: Path) -> tuple[Path, Path, str]:
        archive = root / "staged-player.tar"
        archive.write_bytes(b"staged player archive")
        archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
        marker = root / "physics_capture_v1_smoke.json"
        marker.write_text(
            json.dumps(
                {
                    "status": "accepted",
                    "phase": "complete",
                    "accepted_shot": "shot_001",
                    "protected_unchanged": True,
                    "provenance": {
                        "player_sha256": "a" * 64,
                        "protocol_sha256": "b" * 64,
                        "archive_sha256": archive_sha256,
                    },
                }
            ),
            encoding="utf-8",
        )
        return archive, marker, archive_sha256

    def test_physics_provenance_rejects_missing_or_stale_marker_before_plan_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, marker, _ = self._provenance(root)
            marker.unlink()
            with self.assertRaisesRegex(ValueError, "smoke marker"):
                resolve_physics_capture_provenance(archive, marker)
            self.assertFalse((root / "plan").exists())

    def test_physics_provenance_rejects_marker_digest_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, marker, _ = self._provenance(root)
            marker.write_text(marker.read_text(encoding="utf-8").replace('"archive_sha256": "', '"archive_sha256": "0'), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "archive_sha256"):
                resolve_physics_capture_provenance(archive, marker)

    def test_physics_provenance_rejects_failed_and_stale_smoke_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, marker, _ = self._provenance(root)
            failed = json.loads(marker.read_text(encoding="utf-8"))
            failed["status"] = "failed"
            marker.write_text(json.dumps(failed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "status=accepted"):
                resolve_physics_capture_provenance(archive, marker)
            marker.write_text(json.dumps({**failed, "status": "accepted"}), encoding="utf-8")
            marker_time = marker.stat().st_mtime_ns
            os.utime(archive, ns=(marker_time + 1, marker_time + 1))
            with self.assertRaisesRegex(ValueError, "stale"):
                resolve_physics_capture_provenance(archive, marker)

    def test_physics_provenance_rejects_failed_malformed_stale_and_wrong_version_markers(self):
        mutations = {
            "failed": lambda archive, marker: marker.write_text(marker.read_text(encoding="utf-8").replace('"accepted"', '"failed"'), encoding="utf-8"),
            "malformed": lambda archive, marker: marker.write_text("{", encoding="utf-8"),
            "uppercase digest": lambda archive, marker: marker.write_text(marker.read_text(encoding="utf-8").replace('"player_sha256": "a', '"player_sha256": "A'), encoding="utf-8"),
            "stale": lambda archive, marker: os.utime(marker, ns=(archive.stat().st_atime_ns, archive.stat().st_mtime_ns - 1)),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                archive, marker, _ = self._provenance(Path(temporary))
                mutate(archive, marker)

                with self.assertRaisesRegex(ValueError, "physics smoke marker|stale"):
                    resolve_physics_capture_provenance(archive, marker)

    def test_physics_provenance_rejects_missing_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, marker, _ = self._provenance(root)
            payload = json.loads(marker.read_text(encoding="utf-8"))
            del payload["phase"]
            marker.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "phase=complete"):
                resolve_physics_capture_provenance(archive, marker)

    def test_physics_provenance_rejects_missing_protected_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, marker, _ = self._provenance(root)
            payload = json.loads(marker.read_text(encoding="utf-8"))
            del payload["protected_unchanged"]
            marker.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "protected_unchanged"):
                resolve_physics_capture_provenance(archive, marker)

    def test_physics_provenance_rejects_missing_or_empty_accepted_shot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, marker, _ = self._provenance(root)
            payload = json.loads(marker.read_text(encoding="utf-8"))
            del payload["accepted_shot"]
            marker.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "accepted_shot"):
                resolve_physics_capture_provenance(archive, marker)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, marker, _ = self._provenance(root)
            payload = json.loads(marker.read_text(encoding="utf-8"))
            payload["accepted_shot"] = ""
            marker.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "accepted_shot"):
                resolve_physics_capture_provenance(archive, marker)

    def test_physics_provenance_rejects_missing_provenance_object(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, marker, _ = self._provenance(root)
            payload = json.loads(marker.read_text(encoding="utf-8"))
            del payload["provenance"]
            marker.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance object"):
                resolve_physics_capture_provenance(archive, marker)

    def test_physics_provenance_rejects_malformed_nested_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, marker, _ = self._provenance(root)
            payload = json.loads(marker.read_text(encoding="utf-8"))
            payload["provenance"]["player_sha256"] = "not-a-hash"
            marker.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                resolve_physics_capture_provenance(archive, marker)

    def test_physics_provenance_accepts_the_documented_smoke_report_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive, marker, archive_sha256 = self._provenance(Path(temporary))

            provenance = resolve_physics_capture_provenance(archive, marker)

            self.assertEqual(provenance.archive_sha256, archive_sha256)
            self.assertEqual(provenance.player_sha256, "a" * 64)
            self.assertEqual(provenance.protocol_sha256, "b" * 64)

    def test_physics_cli_rejects_non_directory_stage_before_plan_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "out"
            output_root.mkdir()
            stage_file = root / "not-a-stage"
            stage_file.write_text("not a directory", encoding="utf-8")
            plan_dir = root / "plan"
            script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_rollout_dataset.py"

            result = subprocess.run(
                [
                    "python",
                    str(script),
                    "plan",
                    "--command-output-root",
                    str(output_root),
                    "--output-dir",
                    str(plan_dir),
                    "--physics-capture-v1",
                    "--physics-player-dir",
                    str(stage_file),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("staged player directory", result.stderr)
            self.assertFalse(plan_dir.exists())

    def test_physics_plan_rejects_protected_active_root_before_plan_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, marker, _ = self._provenance(root)
            provenance = resolve_physics_capture_provenance(archive, marker)
            protected_root = Path(__file__).resolve().parents[1] / "data" / "novphy_rollouts_dataset_20260708_171531"
            plan_dir = root / "plan"
            episode = PlannedEpisode("train", LevelEntry("novelty_level_1", "type010101", "levels/one.xml"), protected_root / "train" / "episode", "scheduled")

            with self.assertRaisesRegex(ValueError, "active cohort"):
                write_collection_plan(plan_dir, output_root=protected_root, episodes=[episode], summary={}, options=CollectionOptions(workers=1), targets=CollectionTargets(train=1, dev=1), seed="protected", physics_provenance=provenance)

            self.assertFalse(plan_dir.exists())

    def test_enriched_commands_copy_verified_archive_and_propagate_one_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_root = root / "out"
            out_root.mkdir()
            archive, marker, archive_sha256 = self._provenance(root)
            provenance = resolve_physics_capture_provenance(archive, marker)
            episode = PlannedEpisode("train", LevelEntry("novelty_level_1", "type010101", "levels/one.xml"), out_root / "train" / "episode", "scheduled")
            plan_path = write_collection_plan(root / "plan", output_root=out_root, episodes=[episode], summary={}, options=CollectionOptions(workers=2), targets=CollectionTargets(train=1, dev=1), seed="physics", physics_provenance=provenance, collection_purpose="smoke")
            commands = generate_collection_commands(plan_path, output_root=out_root, options=CollectionOptions(workers=2), physics_provenance=provenance)
            payload = json.loads(plan_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["contract"]["capture_contract"], "physics_capture_v1")
            self.assertEqual(payload["contract"]["archive_sha256"], archive_sha256)
            self.assertEqual(commands.count(archive_sha256), 2)
            self.assertIn("--physics-capture-v1", commands)
            self.assertIn("--physics-host 127.0.0.1", commands)
            self.assertIn("--physics-port 2005", commands)
            self.assertIn("--physics-archive-sha256", commands)
            self.assertIn("sha256sum", commands)
            self.assertIn("tar -xf", commands)

    def test_enriched_plan_rejects_the_active_cohort_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, marker, _ = self._provenance(root)
            provenance = resolve_physics_capture_provenance(archive, marker)
            episode = PlannedEpisode("train", LevelEntry("novelty_level_1", "type010101", "levels/one.xml"), ACTIVE_COHORT_ROOT / "train" / "episode", "scheduled")

            with self.assertRaisesRegex(ValueError, "active cohort"):
                write_collection_plan(root / "plan", output_root=ACTIVE_COHORT_ROOT, episodes=[episode], summary={}, options=CollectionOptions(workers=1), targets=CollectionTargets(train=1, dev=1), seed="active", physics_provenance=provenance)
            self.assertFalse((root / "plan").exists())

    def test_legacy_command_bytes_are_unchanged_when_physics_is_unset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_root = root / "out"
            out_root.mkdir()
            episode = PlannedEpisode("train", LevelEntry("novelty_level_1", "type010101", "levels/one.xml"), out_root / "train" / "episode", "scheduled")
            plan_path = write_collection_plan(root / "plan", output_root=out_root, episodes=[episode], summary={}, options=CollectionOptions(workers=1), targets=CollectionTargets(train=1, dev=1), seed="legacy", collection_purpose="smoke")
            commands = generate_collection_commands(plan_path, output_root=out_root, options=CollectionOptions(workers=1))

            # No physics staging, provenance, or capture flag may leak into a legacy
            # script.  The bare word "physics" is not usable as the assertion because
            # the repo root itself may legitimately contain it (the physics worktree).
            for physics_token in (
                "--physics-capture-v1",
                "--physics-player-dir",
                "--physics-player-archive",
                "--physics-smoke-marker",
                "--physics-host",
                "--physics-port",
                "--physics-player-sha256",
                "--physics-protocol-sha256",
                "--physics-archive-sha256",
                "worker_archive",
                "archive_sha256",
            ):
                self.assertNotIn(physics_token, commands)
            self.assertIn('cp -a sciencebirdsgames/Linux "$worker_engine_dir"', commands)
            # Normalize both the temporary tree and the generating repo root so the
            # digest pins the script shape rather than the machine it ran on.
            normalized = commands.replace(str(root), "<ROOT>").replace(str(Path.cwd()), "<REPO>")
            self.assertEqual(hashlib.sha256(normalized.encode("utf-8")).hexdigest(), "c026cb7599b2bfae1499e5a40b49d2c62926ce74c5a45b751b68161d26642933")


if __name__ == "__main__":
    unittest.main()

class PhysicsCaptureValidationTests(unittest.TestCase):
    def _make_physics_episode(self, root: Path) -> Path:
        from PIL import Image

        options = CollectionOptions(count=1, workers=1, fps=1, duration=1)
        make_complete_episode(root, options)
        shot = root / "shot_001"
        states = [json.loads(line) for line in (Path(__file__).parent / "fixtures" / "physics_capture_v1" / "physics_state.jsonl").read_text(encoding="utf-8").splitlines()]
        events = [json.loads(line) for line in (Path(__file__).parent / "fixtures" / "physics_capture_v1" / "physics_events.jsonl").read_text(encoding="utf-8").splitlines()]
        for record in states + events:
            record["shot_id"] = "shot_001"
        shutil.rmtree(shot / "frames")
        (shot / "frames").mkdir()
        frame_checksums = []
        for index, state in enumerate(states[1:]):
            frame = shot / "frames" / f"frame_{index:06d}.png"
            Image.new("RGB", (4, 3), (index + 1, 2, 3)).save(frame, format="PNG")
            state["rgb_frame"].update({"relative_path": f"frames/{frame.name}", "width_pixels": 4, "height_pixels": 3})
            frame_checksums.append({"relative_path": f"frames/{frame.name}", "sha256": hashlib.sha256(frame.read_bytes()).hexdigest()})
        state_path = shot / "physics_state.jsonl"
        event_path = shot / "physics_events.jsonl"
        state_path.write_text("".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in states), encoding="utf-8")
        event_path.write_text("".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in events), encoding="utf-8")
        metadata = {"capture_contract": "physics_capture_v1", "schema_version": "physics_capture_v1", "protocol_version": 1, "player_sha256": "a" * 64, "protocol_sha256": "b" * 64, "archive_sha256": "c" * 64, "frame_count": 2, "frames_dir": "frames", "frames": [{"path": f"frames/frame_{index:06d}.png"} for index in range(2)], "frame_checksums": frame_checksums, "physics_state_path": "physics_state.jsonl", "physics_events_path": "physics_events.jsonl", "physics_state_count": 2, "physics_event_count": len(events), "physics_state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(), "physics_events_sha256": hashlib.sha256(event_path.read_bytes()).hexdigest(), "sidecars_closed": True}
        (shot / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["capture_source"] = "capture_physics_rollout"
        manifest["capture_contract"] = {"contract_name": PHYSICS_CAPTURE_V1.contract_name, "contract_version": PHYSICS_CAPTURE_V1.contract_version, "artifact_layout_version": PHYSICS_CAPTURE_V1.artifact_layout_version, "player_sha256": "a" * 64, "protocol_sha256": "b" * 64, "archive_sha256": "c" * 64, "declared_capabilities": list(PHYSICS_CAPTURE_V1.declared_capabilities), "sidecar_paths": [{"relative_path": sidecar.relative_path, "capabilities": list(sidecar.capabilities)} for sidecar in PHYSICS_CAPTURE_V1.sidecar_paths]}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        (shot / "pre_shot.png").unlink()
        return root

    def test_legacy_rgb_predicate_remains_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "legacy"
            options = CollectionOptions(count=1, workers=1, fps=1, duration=1)
            make_complete_episode(root, options)
            self.assertTrue(_is_canonically_complete_fresh_engine_episode(root, options))

    def test_valid_enriched_episode_requires_explicit_switch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._make_physics_episode(Path(temporary) / "physics")
            contract = EpisodeValidationContract(1, 1, 1)
            self.assertIsInstance(validate_rollout_episode(root, contract), EpisodeRejected)
            self.assertIsInstance(validate_rollout_episode(root, contract, capture_contract="physics_capture_v1"), EpisodeAccepted)

    def test_corrupt_sidecar_and_frame_classes_fail_closed(self):
        mutations = {
            "missing": lambda shot: (shot / "physics_events.jsonl").unlink(),
            "truncated": lambda shot: (shot / "physics_state.jsonl").write_text('{"schema_version":', encoding="utf-8"),
            "extra_png": lambda shot: shutil.copy2(shot / "frames" / "frame_000000.png", shot / "frames" / "frame_999999.png"),
            "stale_schema": lambda shot: self._rewrite_jsonl(shot / "physics_state.jsonl", 1, "schema_version", "physics_capture_v0"),
            "duplicate_sequence": lambda shot: self._rewrite_jsonl(shot / "physics_state.jsonl", 2, "sequence", 1),
            "out_of_order": lambda shot: self._rewrite_jsonl(shot / "physics_events.jsonl", 0, "sequence", 8),
            "render_frame_mismatch": lambda shot: self._rewrite_nested_jsonl(shot / "physics_state.jsonl", 1, "rgb_frame", "render_frame", 999),
            "extra_state": lambda shot: (shot / "physics_state.jsonl").write_text((shot / "physics_state.jsonl").read_text(encoding="utf-8") + (shot / "physics_state.jsonl").read_text(encoding="utf-8").splitlines()[-1] + "\n", encoding="utf-8"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = self._make_physics_episode(Path(temporary) / "physics")
                mutate(root / "shot_001")
                result = validate_rollout_episode(root, EpisodeValidationContract(1, 1, 1), capture_contract="physics_capture_v1")
                self.assertIsInstance(result, EpisodeRejected)

    @staticmethod
    def _rewrite_jsonl(path: Path, index: int, key: str, value) -> None:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        records[index][key] = value
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    @staticmethod
    def _rewrite_nested_jsonl(path: Path, index: int, container: str, key: str, value) -> None:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        records[index][container][key] = value
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
