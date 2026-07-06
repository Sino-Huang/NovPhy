import json
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_rollout_dataset import (
    CollectionOptions,
    discover_level_entries,
    generate_collection_commands,
    partition_levels,
    write_config_for_manifest_level,
    write_partition_manifest,
)


def make_level(engine_dir: Path, novelty_level: str, level_type: str, name: str) -> None:
    level_path = engine_dir / "9001_Data" / "StreamingAssets" / "Levels" / novelty_level / level_type / "Levels" / name
    level_path.parent.mkdir(parents=True, exist_ok=True)
    level_path.write_text("<Level />", encoding="utf-8")


class PrepareRolloutDatasetTest(unittest.TestCase):
    def test_partition_levels_are_stable_non_overlapping_and_bucketed(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine_dir = Path(tmp) / "sciencebirdsgames" / "Linux"
            for index in range(10):
                make_level(engine_dir, "novelty_level_0", "type010101", f"level_{index:03d}.xml")
                make_level(engine_dir, "novelty_level_3", "type010303", f"level_{index:03d}.xml")

            entries = discover_level_entries(engine_dir)
            first = partition_levels(entries, seed="unit-seed")
            second = partition_levels(entries, seed="unit-seed")

            self.assertEqual(first, second)
            for split_name in ("train", "dev", "test"):
                self.assertIn(split_name, first)

            all_paths_by_split = {
                split_name: {entry.relative_path for entry in split_entries}
                for split_name, split_entries in first.items()
            }
            self.assertFalse(all_paths_by_split["train"] & all_paths_by_split["dev"])
            self.assertFalse(all_paths_by_split["train"] & all_paths_by_split["test"])
            self.assertFalse(all_paths_by_split["dev"] & all_paths_by_split["test"])
            self.assertEqual(sum(len(paths) for paths in all_paths_by_split.values()), 20)

            train_buckets = {(entry.novelty_level, entry.level_type) for entry in first["train"]}
            dev_buckets = {(entry.novelty_level, entry.level_type) for entry in first["dev"]}
            test_buckets = {(entry.novelty_level, entry.level_type) for entry in first["test"]}
            self.assertEqual(train_buckets, {("novelty_level_0", "type010101"), ("novelty_level_3", "type010303")})
            self.assertEqual(dev_buckets, train_buckets)
            self.assertEqual(test_buckets, train_buckets)

    def test_write_partition_manifest_and_commands_for_train_dev_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine_dir = root / "sciencebirdsgames" / "Linux"
            for index in range(6):
                make_level(engine_dir, "novelty_level_0", "type010101", f"level_{index:03d}.xml")

            partitions = partition_levels(discover_level_entries(engine_dir), seed="manifest-seed")
            manifest_path = write_partition_manifest(root / "data" / "rollout_dataset_plan", partitions)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["schema"], "novphy-rollout-dataset-partitions-v1")
            self.assertEqual(manifest["counts"]["total"], 6)
            self.assertIn("9001_Data/StreamingAssets/Levels/novelty_level_0/type010101/Levels/level_", json.dumps(manifest))

            commands = generate_collection_commands(
                manifest_path,
                output_root=Path("data/generated_rollouts"),
                options=CollectionOptions(count=2, fps=30, duration=4.5, display=":149"),
            )

            self.assertIn("source ~/cd_novphy", commands)
            self.assertIn("python scripts/prepare_rollout_dataset.py write-config", commands)
            self.assertIn("python scripts/collect_rollouts.py", commands)
            self.assertIn("--capture-source desktop", commands)
            self.assertIn("--fresh-engine-per-rollout", commands)
            self.assertIn("--count 2", commands)
            self.assertIn("--fps 30", commands)
            self.assertIn("--duration 4.5", commands)
            self.assertIn("--split train", commands)
            self.assertIn("--split dev", commands)
            self.assertNotIn("--split test", commands)

    def test_write_config_requires_level_to_belong_to_requested_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine_dir = root / "sciencebirdsgames" / "Linux"
            for index in range(6):
                make_level(engine_dir, "novelty_level_0", "type010101", f"level_{index:03d}.xml")

            partitions = partition_levels(discover_level_entries(engine_dir), seed="write-config-seed")
            manifest_path = write_partition_manifest(root / "data" / "rollout_dataset_plan", partitions)
            config_path = root / "config.xml"
            dev_level = partitions["dev"][0].relative_path

            written = write_config_for_manifest_level(manifest_path, "dev", dev_level, config_path)

            self.assertEqual(written, config_path)
            self.assertIn(dev_level, config_path.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "not part of split train"):
                write_config_for_manifest_level(manifest_path, "train", dev_level, config_path)

    def test_generate_collection_commands_rejects_newline_display(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine_dir = root / "sciencebirdsgames" / "Linux"
            for index in range(3):
                make_level(engine_dir, "novelty_level_0", "type010101", f"level_{index:03d}.xml")

            partitions = partition_levels(discover_level_entries(engine_dir), seed="newline-display-seed")
            manifest_path = write_partition_manifest(root / "data" / "rollout_dataset_plan", partitions)

            with self.assertRaisesRegex(ValueError, "display must not contain newline"):
                generate_collection_commands(
                    manifest_path,
                    output_root=Path("data/generated_rollouts"),
                    options=CollectionOptions(display=":149\ninjected-command"),
                )

    def test_generate_collection_commands_rejects_newline_level_path_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "partitions.json"
            manifest = {
                "schema": "novphy-rollout-dataset-partitions-v1",
                "seed": "malicious-manifest-seed",
                "counts": {"train": 1, "dev": 0, "test": 0, "total": 1},
                "splits": {
                    "train": [
                        {
                            "novelty_level": "novelty_level_0",
                            "level_type": "type010101",
                            "bucket": "novelty_level_0/type010101",
                            "relative_path": "9001_Data/StreamingAssets/Levels/novelty_level_0/type010101/Levels/level_000.xml\ninjected-command",
                        }
                    ],
                    "dev": [],
                    "test": [],
                },
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "level_path must not contain newline"):
                generate_collection_commands(manifest_path, output_root=Path("data/generated_rollouts"))

    def test_generate_collection_commands_quotes_dynamic_split_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "partitions.json"
            manifest = {
                "schema": "novphy-rollout-dataset-partitions-v1",
                "seed": "quoted-split-seed",
                "counts": {"train; injected-command #": 1, "total": 1},
                "splits": {
                    "train; injected-command #": [
                        {
                            "novelty_level": "novelty_level_0",
                            "level_type": "type010101",
                            "bucket": "novelty_level_0/type010101",
                            "relative_path": "9001_Data/StreamingAssets/Levels/novelty_level_0/type010101/Levels/level_000.xml",
                        }
                    ]
                },
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            commands = generate_collection_commands(
                manifest_path,
                output_root=Path("data/generated_rollouts"),
                splits=("train; injected-command #",),
            )

            self.assertIn("--split 'train; injected-command #'", commands)
            self.assertNotIn("--split train; injected-command #", commands)

    def test_generate_collection_commands_logs_failed_levels_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine_dir = root / "sciencebirdsgames" / "Linux"
            for index in range(6):
                make_level(engine_dir, "novelty_level_0", "type010101", f"level_{index:03d}.xml")

            partitions = partition_levels(discover_level_entries(engine_dir), seed="failure-ledger-seed")
            manifest_path = write_partition_manifest(root / "data" / "rollout_dataset_plan", partitions)

            commands = generate_collection_commands(
                manifest_path,
                output_root=Path("data/generated_rollouts"),
                options=CollectionOptions(count=2, fps=30, duration=4.5, display=":149"),
            )

            self.assertIn("failure_ledger=", commands)
            self.assertIn("failed_levels.tsv", commands)
            self.assertIn("failure_count=0", commands)
            self.assertIn("if python scripts/prepare_rollout_dataset.py write-config", commands)
            self.assertIn("then\n    printf 'Completed %s: %s\\n'", commands)
            self.assertIn("else\n    status=$?", commands)
            self.assertIn("failure_count=$((failure_count + 1))", commands)
            self.assertIn("printf '%s\\t%s\\t%s\\t%s\\n'", commands)
            self.assertIn("Continuing with the next level", commands)
            self.assertIn("if [[ \"$failure_count\" -gt 0 ]]; then", commands)
            self.assertIn("exit 1", commands)

    def test_generate_collection_commands_quotes_status_messages_for_single_quote_level_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "partitions.json"
            level_path = "9001_Data/StreamingAssets/Levels/novelty_level_0/type010101/Levels/level_'001.xml"
            manifest = {
                "schema": "novphy-rollout-dataset-partitions-v1",
                "seed": "single-quote-level-path-seed",
                "counts": {"train": 1, "dev": 0, "test": 0, "total": 1},
                "splits": {
                    "train": [
                        {
                            "novelty_level": "novelty_level_0",
                            "level_type": "type010101",
                            "bucket": "novelty_level_0/type010101",
                            "relative_path": level_path,
                        }
                    ],
                    "dev": [],
                    "test": [],
                },
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            commands = generate_collection_commands(manifest_path, output_root=Path("data/generated_rollouts"))
            script_path = root / "collect_train_dev.sh"
            script_path.write_text(commands, encoding="utf-8")

            subprocess.run(["bash", "-n", str(script_path)], check=True)
            self.assertIn(shlex.quote(level_path), commands)
            self.assertNotIn(f"echo 'Completed train: {level_path}'", commands)
            self.assertNotIn(f"echo 'Failed train: {level_path}", commands)

    def test_discover_level_entries_reports_missing_or_empty_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine_dir = Path(tmp) / "sciencebirdsgames" / "Linux"

            with self.assertRaisesRegex(FileNotFoundError, "Levels directory not found"):
                discover_level_entries(engine_dir)

            (engine_dir / "9001_Data" / "StreamingAssets" / "Levels").mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "No NovPhy level XML files found"):
                discover_level_entries(engine_dir)


if __name__ == "__main__":
    unittest.main()
