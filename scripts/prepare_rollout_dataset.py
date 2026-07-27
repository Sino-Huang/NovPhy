#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, TypedDict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sciencebirdsagents.Utils.PrepareTestConfig import write_config  # noqa: E402


SCHEMA: Final = "novphy-rollout-dataset-partitions-v1"
PLAN_SCHEMA: Final = "novphy-rollout-collection-plan-v2"
DEFAULT_OUTPUT_DIR: Final = Path("data/rollout_dataset_plan")
DEFAULT_SEED: Final = "novphy-rollout-dataset-v1"
DEFAULT_AGENT_PORT_BASE: Final = 2004
DEFAULT_GAME_PORT_BASE: Final = 9001
WORKER_PORT_STRIDE: Final = 10
MAX_PORT: Final = 65535


class JsonValue(TypedDict, total=False):
    pass


@dataclass(frozen=True, slots=True)
class LevelEntry:
    novelty_level: str
    level_type: str
    relative_path: str

    @property
    def bucket(self) -> str:
        return f"{self.novelty_level}/{self.level_type}"

    @property
    def stem(self) -> str:
        return Path(self.relative_path).stem

    def to_json(self) -> dict[str, str]:
        return {
            "novelty_level": self.novelty_level,
            "level_type": self.level_type,
            "bucket": self.bucket,
            "relative_path": self.relative_path,
        }

    @classmethod
    def from_json(cls, data: dict[str, str]) -> LevelEntry:
        return cls(
            novelty_level=data["novelty_level"],
            level_type=data["level_type"],
            relative_path=data["relative_path"],
        )


@dataclass(frozen=True, slots=True)
class CollectionTargets:
    train: int = 100
    dev: int = 20

    def for_split(self, split: str) -> int:
        if split == "train":
            return self.train
        if split == "dev":
            return self.dev
        raise ValueError(f"Unsupported scheduled split: {split}")


@dataclass(frozen=True, slots=True)
class CollectionOptions:
    count: int = 12
    fps: float = 30.0
    duration: float = 5.0
    display: str = ":149"
    ui_settle_seconds: float = 5.0
    connect_timeout: float = 60.0
    prepare_timeout: float = 90.0
    read_timeout: float = 420.0
    speed: int = 1
    workers: int = 6
    agent_port_base: int = DEFAULT_AGENT_PORT_BASE
    game_port_base: int = DEFAULT_GAME_PORT_BASE
    resume: bool = True


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    index: int
    display: str
    agent_port: int
    game_port: int


@dataclass(frozen=True, slots=True)
class PlannedEpisode:
    split: str
    entry: LevelEntry
    output_dir: Path
    source: str

    def to_json(self, output_root: Path) -> dict[str, str]:
        return {
            "split": self.split,
            "novelty_level": self.entry.novelty_level,
            "level_type": self.entry.level_type,
            "bucket": self.entry.bucket,
            "relative_path": self.entry.relative_path,
            "output_path": self.output_dir.relative_to(output_root).as_posix(),
            "source": self.source,
        }


def engine_dir_for(root: Path = ROOT, operating_system: str = "Linux") -> Path:
    return root / "sciencebirdsgames" / operating_system


def _levels_root(engine_dir: Path) -> Path:
    return engine_dir / "9001_Data" / "StreamingAssets" / "Levels"


def discover_level_entries(engine_dir: Path) -> list[LevelEntry]:
    levels_root = _levels_root(engine_dir)
    if not levels_root.is_dir():
        raise FileNotFoundError(f"Levels directory not found: {levels_root}")
    entries = [
        LevelEntry(novelty_dir.name, type_dir.name, xml_path.relative_to(engine_dir).as_posix())
        for novelty_dir in sorted(path for path in levels_root.iterdir() if path.is_dir() and path.name.startswith("novelty_level_"))
        for type_dir in sorted(path for path in novelty_dir.iterdir() if path.is_dir() and path.name.startswith("type010"))
        for xml_path in sorted((type_dir / "Levels").glob("*.xml"))
    ]
    if not entries:
        raise RuntimeError(f"No NovPhy level XML files found under {levels_root}")
    return entries


def _stable_key(entry: LevelEntry, seed: str) -> str:
    return hashlib.sha256(f"{seed}\0{entry.relative_path}".encode("utf-8")).hexdigest()


def _split_counts(total: int, train_ratio: float, dev_ratio: float) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    if total == 1:
        return 1, 0, 0
    if total == 2:
        return 1, 1, 0
    train_count = max(1, int(total * train_ratio))
    dev_count = max(1, int(total * dev_ratio))
    if train_count + dev_count >= total:
        return total - 2, 1, 1
    return train_count, dev_count, total - train_count - dev_count


def partition_levels(
    entries: Iterable[LevelEntry],
    *,
    seed: str = DEFAULT_SEED,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
) -> dict[str, list[LevelEntry]]:
    buckets = _bucket_entries(entries)
    partitions = {"train": [], "dev": [], "test": []}
    for bucket in sorted(buckets):
        ordered = sorted(buckets[bucket], key=lambda entry: _stable_key(entry, seed))
        train_count, dev_count, _ = _split_counts(len(ordered), train_ratio, dev_ratio)
        partitions["train"].extend(ordered[:train_count])
        partitions["dev"].extend(ordered[train_count : train_count + dev_count])
        partitions["test"].extend(ordered[train_count + dev_count :])
    return {split: sorted(values, key=lambda entry: entry.relative_path) for split, values in partitions.items()}


def write_partition_manifest(output_dir: Path, partitions: dict[str, list[LevelEntry]], *, seed: str = DEFAULT_SEED) -> Path:
    counts = {split: len(entries) for split, entries in partitions.items()}
    payload = {
        "schema": SCHEMA,
        "seed": seed,
        "counts": {**counts, "total": sum(counts.values())},
        "splits": {split: [entry.to_json() for entry in entries] for split, entries in partitions.items()},
    }
    return _atomic_write(output_dir / "partitions.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_partition_manifest(path: Path) -> dict[str, list[LevelEntry]]:
    payload = _read_json(path)
    if payload is None or payload.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported partition manifest schema: {None if payload is None else payload.get('schema')}")
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("Partition manifest must contain splits")
    result: dict[str, list[LevelEntry]] = {}
    for split in ("train", "dev", "test"):
        values = splits.get(split)
        if not isinstance(values, list):
            raise ValueError(f"Partition manifest must contain {split} entries")
        entries: list[LevelEntry] = []
        for value in values:
            if not isinstance(value, dict) or not all(isinstance(value.get(key), str) for key in ("novelty_level", "level_type", "relative_path")):
                raise ValueError(f"Partition manifest contains malformed {split} entry")
            entries.append(LevelEntry(value["novelty_level"], value["level_type"], value["relative_path"]))
        result[split] = entries
    return result


def _safe_output_name(entry: LevelEntry) -> str:
    return f"{entry.novelty_level}_{entry.level_type}_{entry.stem}".replace("/", "_")


def _format_number(value: float | int) -> str:
    return str(value) if isinstance(value, int) else str(float(value)).rstrip("0").rstrip(".")


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _worker_display(base_display: str, worker_index: int) -> str:
    if not base_display.startswith(":"):
        raise ValueError("display must be an X display like :149 for parallel workers")
    try:
        return f":{int(base_display[1:]) + worker_index}"
    except ValueError as exc:
        raise ValueError("display must be an X display like :149 for parallel workers") from exc


def _worker_specs(opts: CollectionOptions) -> list[WorkerSpec]:
    if opts.workers < 1:
        raise ValueError("workers must be at least 1")
    for label, port_base in (("agent", opts.agent_port_base), ("game", opts.game_port_base)):
        if type(port_base) is not int or not 1 <= port_base <= MAX_PORT:
            raise ValueError(f"{label} port base must be a non-boolean integer in 1..{MAX_PORT}")
        if port_base + (opts.workers - 1) * WORKER_PORT_STRIDE > MAX_PORT:
            raise ValueError(f"{label} port base final worker port exceeds {MAX_PORT}")
    specs = [
        WorkerSpec(
            index=index,
            display=opts.display if opts.workers == 1 else _worker_display(opts.display, index),
            agent_port=opts.agent_port_base + index * WORKER_PORT_STRIDE,
            game_port=opts.game_port_base + index * WORKER_PORT_STRIDE,
        )
        for index in range(opts.workers)
    ]
    overlap = sorted({spec.agent_port for spec in specs}.intersection(spec.game_port for spec in specs))
    if overlap:
        raise ValueError(f"agent and game port families must be disjoint: {', '.join(map(str, overlap))}")
    return specs


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _numeric_contract_value(value: object, expected: float) -> bool:
    return type(value) in (int, float) and value == expected


def _accepted_action_signs(action_log: dict[str, object], count: int) -> list[int] | None:
    accepted = action_log.get("accepted_trials")
    if not isinstance(accepted, list) or len(accepted) != count:
        return None
    signs: list[int] = []
    for trial in accepted:
        if not isinstance(trial, dict):
            return None
        action = trial.get("action")
        if not isinstance(action, dict):
            return None
        release = action.get("drag_release", action.get("release"))
        if not isinstance(release, list) or len(release) < 1 or type(release[0]) not in (int, float):
            return None
        x = release[0]
        if x == 0:
            return None
        signs.append(-1 if x < 0 else 1)
    return signs


def _contained_readable_artifact(path: Path, episode_dir: Path, *, directory: bool) -> bool:
    try:
        relative = path.relative_to(episode_dir)
        current = episode_dir
        for component in relative.parts:
            current /= component
            if current.is_symlink():
                return False
        resolved_episode_dir = episode_dir.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_episode_dir)
    except (OSError, ValueError):
        return False
    if directory:
        return path.is_dir() and os.access(path, os.R_OK | os.X_OK)
    return path.is_file() and os.access(path, os.R_OK)


def _metadata_frame_matches(value: object, expected_frame: Path, shot_dir: Path) -> bool:
    if not isinstance(value, str):
        return False
    accepted_paths = {str(expected_frame)}
    try:
        accepted_paths.add(str(expected_frame.relative_to(shot_dir)))
    except ValueError:
        pass
    try:
        accepted_paths.add(str(expected_frame.relative_to(Path.cwd())))
    except ValueError:
        pass
    return value in accepted_paths


def _accepted_raw_artifacts_present(output_dir: Path, action_log: dict[str, object], count: int) -> bool:
    accepted = action_log.get("accepted_trials")
    if not isinstance(accepted, list) or len(accepted) != count:
        return False
    for trial in accepted:
        if not isinstance(trial, dict) or not isinstance(trial.get("shot_name"), str):
            return False
        shot_dir = output_dir / trial["shot_name"]
        if not _contained_readable_artifact(shot_dir, output_dir, directory=True):
            return False
        metadata_path = shot_dir / "metadata.json"
        frames_dir = shot_dir / "frames"
        pre_shot_path = shot_dir / "pre_shot.png"
        if (
            not _contained_readable_artifact(metadata_path, output_dir, directory=False)
            or not _contained_readable_artifact(frames_dir, output_dir, directory=True)
            or not _contained_readable_artifact(pre_shot_path, output_dir, directory=False)
        ):
            return False
        metadata = _read_json(metadata_path)
        frame_count = None if metadata is None else metadata.get("frame_count")
        if type(frame_count) is not int or frame_count < 1:
            return False
        expected_frames = [frames_dir / f"frame_{index:06d}.png" for index in range(frame_count)]
        if not all(_contained_readable_artifact(frame_path, output_dir, directory=False) for frame_path in expected_frames):
            return False
        if "frames" in metadata:
            frames = metadata["frames"]
            if not isinstance(frames, list) or len(frames) != frame_count:
                return False
            for expected_frame, frame in zip(expected_frames, frames, strict=True):
                if not isinstance(frame, dict):
                    return False
                if not _metadata_frame_matches(frame.get("path"), expected_frame, shot_dir):
                    return False
    return True


def _accepted_attempts_are_canonical(manifest: dict[str, object], count: int) -> bool:
    attempts = manifest.get("attempts")
    if not isinstance(attempts, list) or manifest.get("attempt_count") != len(attempts):
        return False
    accepted = 0
    for attempt in attempts:
        if not isinstance(attempt, dict):
            return False
        validation = attempt.get("artifact_validation")
        if not isinstance(validation, dict):
            return False
        if attempt.get("accepted") is True:
            accepted += 1
            if (
                attempt.get("attempt_status") != "accepted"
                or validation.get("accepted") is not True
                or validation.get("classification") != "gameplay-valid"
                or validation.get("retryable") is not False
                or validation.get("retry_decision") != "accept"
            ):
                return False
        elif not (
            attempt.get("accepted") is False
            and attempt.get("attempt_status") == "invalid_retryable"
            and validation.get("accepted") is False
            and validation.get("retryable") is True
            and validation.get("retry_decision") == "retry"
        ):
            return False
    return accepted == count


def _is_canonically_complete_fresh_engine_episode(
    output_dir: Path,
    opts: CollectionOptions,
    *,
    level_five: bool = False,
) -> bool:
    if output_dir.is_symlink() or not output_dir.is_dir() or not os.access(output_dir, os.R_OK | os.X_OK):
        return False
    manifest_path = output_dir / "manifest.json"
    action_log_path = output_dir / "action_log.json"
    action_log_jsonl_path = output_dir / "action_log.jsonl"
    if not all(
        _contained_readable_artifact(path, output_dir, directory=False)
        for path in (manifest_path, action_log_path, action_log_jsonl_path)
    ):
        return False
    manifest = _read_json(manifest_path)
    action_log = _read_json(action_log_path)
    if manifest is None or action_log is None:
        return False
    if (
        manifest.get("capture_source") != "capture_desktop_rollout"
        or manifest.get("replay_mode") != "fresh-engine-per-rollout"
        or not _numeric_contract_value(manifest.get("target_fps"), opts.fps)
        or not _numeric_contract_value(manifest.get("duration_seconds"), opts.duration)
        or manifest.get("ui_level") != 1
        or type(manifest.get("accepted_rollout_count")) is not int
        or manifest["accepted_rollout_count"] != opts.count
        or type(manifest.get("rollout_count")) is not int
        or manifest["rollout_count"] != opts.count
        or manifest.get("collection_status") == "retry_exhausted"
        or manifest.get("collection_error") is not None
        or not _accepted_attempts_are_canonical(manifest, opts.count)
    ):
        return False
    if not _accepted_raw_artifacts_present(output_dir, action_log, opts.count):
        return False
    signs = _accepted_action_signs(action_log, opts.count)
    if signs is None:
        return False
    if level_five and signs != [-1 if index % 2 == 0 else 1 for index in range(opts.count)]:
        return False
    return True


def _path_is_safe_absent(path: Path, output_root: Path) -> bool:
    try:
        root = output_root.resolve(strict=True)
        candidate = path.resolve(strict=False)
        candidate.relative_to(root)
    except (OSError, ValueError):
        return False
    return not path.exists() and not path.is_symlink() and not any(part.is_symlink() for part in path.parents if part != root.parent)


def _path_is_safe_existing(path: Path, output_root: Path) -> bool:
    try:
        root = output_root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        return False
    return not path.is_symlink() and all(not part.is_symlink() for part in path.parents if part != root.parent)


def _bucket_entries(entries: Iterable[LevelEntry]) -> dict[str, list[LevelEntry]]:
    buckets: dict[str, list[LevelEntry]] = {}
    for entry in entries:
        buckets.setdefault(entry.bucket, []).append(entry)
    return buckets


def _plan_bucket(
    entries: list[LevelEntry],
    *,
    split: str,
    output_root: Path,
    opts: CollectionOptions,
    targets: CollectionTargets,
    seed: str,
) -> list[PlannedEpisode]:
    selected: list[PlannedEpisode] = []
    ordered = sorted(entries, key=lambda entry: _stable_key(entry, seed))
    level_five = ordered[0].novelty_level == "novelty_level_5"
    for entry in ordered:
        output_dir = output_root / split / _safe_output_name(entry)
        if _path_is_safe_existing(output_dir, output_root) and _is_canonically_complete_fresh_engine_episode(output_dir, opts, level_five=level_five):
            selected.append(PlannedEpisode(split, entry, output_dir, "existing"))
            if len(selected) == targets.for_split(split):
                return selected
    selected_paths = {episode.output_dir for episode in selected}
    for entry in ordered:
        output_dir = output_root / split / _safe_output_name(entry)
        if output_dir not in selected_paths and _path_is_safe_absent(output_dir, output_root):
            selected.append(PlannedEpisode(split, entry, output_dir, "scheduled"))
            if len(selected) == targets.for_split(split):
                return selected
    raise RuntimeError(f"Bucket {entries[0].bucket} has insufficient safe absent capacity for {split}")


def build_collection_plan(
    entries: Iterable[LevelEntry],
    *,
    output_root: Path,
    options: CollectionOptions | None = None,
    targets: CollectionTargets | None = None,
    seed: str = DEFAULT_SEED,
) -> tuple[list[PlannedEpisode], dict[str, dict[str, int]]]:
    opts = options or CollectionOptions()
    target = targets or CollectionTargets()
    _worker_specs(opts)
    if not output_root.is_dir():
        raise FileNotFoundError(f"Existing output root is required: {output_root}")
    partitions = partition_levels(entries, seed=seed)
    buckets = _bucket_entries(entries)
    partition_buckets = {split: _bucket_entries(partitions[split]) for split in ("train", "dev")}
    if len(buckets) != 80:
        raise RuntimeError(f"Expected 80 normal and novel buckets, found {len(buckets)}")
    plan: list[PlannedEpisode] = []
    summary: dict[str, dict[str, int]] = {}
    for bucket in sorted(buckets):
        for split in ("train", "dev"):
            candidates = partition_buckets[split].get(bucket, [])
            if not candidates:
                raise RuntimeError(f"Bucket {bucket} has no {split} partition capacity")
            selected = _plan_bucket(candidates, split=split, output_root=output_root, opts=opts, targets=target, seed=seed)
            plan.extend(selected)
            summary[f"{split}:{bucket}"] = {
                "target": target.for_split(split),
                "existing": sum(episode.source == "existing" for episode in selected),
                "scheduled": sum(episode.source == "scheduled" for episode in selected),
            }
    return plan, summary


def _interleave_schedule(episodes: Iterable[PlannedEpisode]) -> list[PlannedEpisode]:
    grouped: dict[tuple[str, str], list[PlannedEpisode]] = {}
    for episode in episodes:
        grouped.setdefault((episode.split, episode.entry.novelty_level), []).append(episode)
    ordered: list[PlannedEpisode] = []
    keys = sorted(grouped)
    index = 0
    while True:
        appended = False
        for key in keys:
            if index < len(grouped[key]):
                ordered.append(grouped[key][index])
                appended = True
        if not appended:
            return ordered
        index += 1


def _collection_command_lines(episode: PlannedEpisode, opts: CollectionOptions, spec: WorkerSpec) -> list[str]:
    args = [
        f"--output-dir {_quote(episode.output_dir)}",
        "--capture-source desktop",
        "--fresh-engine-per-rollout",
        "--ui-level 1",
        f"--ui-settle-seconds {_format_number(opts.ui_settle_seconds)}",
        f"--count {_format_number(opts.count)}",
        f"--fps {_format_number(opts.fps)}",
        f"--duration {_format_number(opts.duration)}",
        f"--connect-timeout {_format_number(opts.connect_timeout)}",
        f"--prepare-timeout {_format_number(opts.prepare_timeout)}",
        f"--read-timeout {_format_number(opts.read_timeout)}",
        f"--speed {_format_number(opts.speed)}",
        f"--game-dir \"$worker_engine_dir\"",
        f"--port {spec.agent_port}",
        f"--engine-agent-port {spec.agent_port}",
        f"--engine-game-port {spec.game_port}",
    ]
    if episode.entry.novelty_level == "novelty_level_5":
        args.append("--bidirectional-launches")
    lines = [
        '  DISPLAY="$display_id" LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH-}" \\',
        "    python scripts/collect_rollouts.py \\",
    ]
    for index, arg in enumerate(args):
        continuation = " \\" if index < len(args) - 1 else ""
        lines.append(f"    {arg}{continuation}")
    return lines


def _append_scheduled_episode(lines: list[str], episode: PlannedEpisode, opts: CollectionOptions, spec: WorkerSpec) -> None:
    output_dir = _quote(episode.output_dir)
    parent = _quote(episode.output_dir.parent)
    level_path = _quote(episode.entry.relative_path)
    split = _quote(episode.split)
    lines.extend(
        [
            f"# {episode.split}: {episode.entry.relative_path}",
            f"if [[ -L {parent} || -L {output_dir} ]] || ! mkdir -p -- {parent} || ! mkdir -- {output_dir}; then",
            f"  record_failure {split} {level_path} {output_dir} \"reservation\"",
            "  failure_count=$((failure_count + 1))",
            "else",
            "  if python scripts/prepare_rollout_dataset.py write-config \\",
            "  --manifest \"$plan_artifact\" \\",
            f"  --split {split} \\",
            f"  --level-path {level_path} \\",
            "  --config-path \"$worker_engine_dir/config.xml\" && \\",
            *_collection_command_lines(episode, opts, spec),
            "  then",
            "    :",
            "else",
            "    status=$?",
            f"    record_failure {split} {level_path} {output_dir} \"$status\"",
            "    failure_count=$((failure_count + 1))",
            "  fi",
            "fi",
            "",
        ]
    )


def generate_collection_commands(
    plan_path: Path,
    *,
    output_root: Path,
    options: CollectionOptions | None = None,
    splits: tuple[str, ...] = ("train", "dev"),
) -> str:
    opts = options or CollectionOptions()
    specs = _worker_specs(opts)
    artifact = _read_json(plan_path)
    if artifact is None or artifact.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"Unsupported collection plan schema: {None if artifact is None else artifact.get('schema')}")
    selected = artifact.get("selected")
    if not isinstance(selected, list):
        raise ValueError("Collection plan must contain selected episodes")
    scheduled: list[PlannedEpisode] = []
    for value in selected:
        if not isinstance(value, dict) or value.get("source") != "scheduled" or value.get("split") not in splits:
            continue
        output_path = value.get("output_path")
        if not all(isinstance(value.get(key), str) for key in ("split", "novelty_level", "level_type", "relative_path")) or not isinstance(output_path, str):
            raise ValueError("Collection plan contains malformed scheduled episode")
        scheduled.append(
            PlannedEpisode(
                value["split"],
                LevelEntry(value["novelty_level"], value["level_type"], value["relative_path"]),
                output_root / output_path,
                "scheduled",
            )
        )
    for episode in scheduled:
        if not _path_is_safe_absent(episode.output_dir, output_root):
            raise ValueError(f"Scheduled output path is no longer absent and safe: {episode.output_dir}")
    striped = [[] for _ in specs]
    for index, episode in enumerate(_interleave_schedule(scheduled)):
        striped[index % len(specs)].append(episode)
    ledger = plan_path.parent / "failed_levels.tsv"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "source ~/cd_novphy",
        f"plan_artifact={_quote(plan_path)}",
        f"failure_ledger={_quote(ledger)}",
        "failure_ledger_lock=\"${failure_ledger}.lock\"",
        "failure_count=0",
        "record_failure() {",
        "  local split=\"$1\" level_path=\"$2\" output_dir=\"$3\" status=\"$4\"",
        "  ( flock -x 9; [[ -s \"$failure_ledger\" ]] || printf 'split\\tlevel_path\\toutput_dir\\tstatus\\n' >> \"$failure_ledger\"; printf '%s\\t%s\\t%s\\t%s\\n' \"$split\" \"$level_path\" \"$output_dir\" \"$status\" >> \"$failure_ledger\"; ) 9>>\"$failure_ledger_lock\"",
        "}",
        "run_worker() {",
        "  local worker_index=\"$1\" display_id=\"$2\"",
        "  local worker_root worker_engine_dir",
        "  worker_root=\"$(mktemp -d \"${TMPDIR:-/tmp}/novphy_rollout_worker_${worker_index}_XXXXXX\")\"",
        "  trap 'rm -rf \"$worker_root\"' RETURN",
        "  worker_engine_dir=\"$worker_root/engine\"",
        "  cp -a sciencebirdsgames/Linux \"$worker_engine_dir\"",
        "  local failure_count=0",
    ]
    for spec in specs:
        lines.append(f"  if [[ \"$worker_index\" == \"{spec.index}\" ]]; then")
        for episode in striped[spec.index]:
            _append_scheduled_episode(lines, episode, opts, spec)
        lines.append("  fi")
    lines.extend(
        [
            "  if [[ \"$failure_count\" -gt 0 ]]; then return 1; fi",
            "  return 0",
            "}",
            "worker_pids=()",
        ]
    )
    for spec in specs:
        lines.extend([f"run_worker {spec.index} {_quote(spec.display)} &", "worker_pids+=(\"$!\")"])
    lines.extend(
        [
            "for worker_pid in \"${worker_pids[@]}\"; do",
            "  if ! wait \"$worker_pid\"; then failure_count=$((failure_count + 1)); fi",
            "done",
            "if [[ \"$failure_count\" -gt 0 ]]; then exit 1; fi",
            "echo \"Completed newly scheduled train/dev episodes. Failure ledger: $failure_ledger\"",
        ]
    )
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, content: str, *, executable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    temporary_path.chmod(0o755 if executable else 0o644)
    temporary_path.replace(path)
    return path


def write_collection_plan(
    output_dir: Path,
    *,
    output_root: Path,
    episodes: Iterable[PlannedEpisode],
    summary: dict[str, dict[str, int]],
    options: CollectionOptions,
    targets: CollectionTargets,
    seed: str,
) -> Path:
    episode_list = list(episodes)
    payload = {
        "schema": PLAN_SCHEMA,
        "seed": seed,
        "output_root": str(output_root.resolve(strict=True)),
        "contract": {"count": options.count, "fps": options.fps, "duration": options.duration, "workers": options.workers, "train_target": targets.train, "dev_target": targets.dev},
        "summary": summary,
        "counts": {
            "selected": len(episode_list),
            "existing": sum(item.source == "existing" for item in episode_list),
            "scheduled": sum(item.source == "scheduled" for item in episode_list),
            "normal": sum(item.entry.novelty_level == "novelty_level_0" for item in episode_list),
            "novel": sum(item.entry.novelty_level != "novelty_level_0" for item in episode_list),
            "test": 0,
        },
        "selected": [episode.to_json(output_root) for episode in episode_list],
    }
    return _atomic_write(output_dir / "collection_plan.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_plan_entries(path: Path, split: str) -> list[LevelEntry]:
    payload = _read_json(path)
    if payload is None or payload.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"Unsupported collection plan schema: {None if payload is None else payload.get('schema')}")
    selected = payload.get("selected")
    if not isinstance(selected, list):
        raise ValueError("Collection plan must contain selected episodes")
    result: list[LevelEntry] = []
    for item in selected:
        if isinstance(item, dict) and item.get("split") == split and all(isinstance(item.get(key), str) for key in ("novelty_level", "level_type", "relative_path")):
            result.append(LevelEntry(item["novelty_level"], item["level_type"], item["relative_path"]))
    return result


def write_config_for_manifest_level(plan_path: Path, split: str, level_path: str, config_path: Path) -> Path:
    payload = _read_json(plan_path)
    if payload is not None and payload.get("schema") == SCHEMA:
        entries = load_partition_manifest(plan_path).get(split, [])
    else:
        entries = load_plan_entries(plan_path, split)
    if level_path not in {entry.relative_path for entry in entries}:
        raise ValueError(f"Level path is not part of split {split}: {level_path}")
    write_config(config_path, [level_path])
    return config_path


def _parse_port_base(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a decimal integer in 1..65535") from exc
    if not 1 <= parsed <= MAX_PORT:
        raise argparse.ArgumentTypeError("must be a decimal integer in 1..65535")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a non-destructive capped NovPhy rollout collection plan")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="Inventory an existing root and atomically publish a capped train/dev plan")
    plan.add_argument("--os", default="Linux")
    plan.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    plan.add_argument("--command-output-root", type=Path, required=True)
    plan.add_argument("--commands-path", type=Path)
    plan.add_argument("--seed", default=DEFAULT_SEED)
    plan.add_argument("--train-target", type=int, default=100)
    plan.add_argument("--dev-target", type=int, default=20)
    plan.add_argument("--count", type=int, default=12)
    plan.add_argument("--fps", type=float, default=30.0)
    plan.add_argument("--duration", type=float, default=5.0)
    plan.add_argument("--display", default=":149")
    plan.add_argument("--workers", type=int, default=6)
    plan.add_argument("--agent-port-base", type=_parse_port_base, default=DEFAULT_AGENT_PORT_BASE)
    plan.add_argument("--game-port-base", type=_parse_port_base, default=DEFAULT_GAME_PORT_BASE)
    config = subparsers.add_parser("write-config", help="Write config.xml for a selected plan episode")
    config.add_argument("--manifest", type=Path, required=True)
    config.add_argument("--split", choices=("train", "dev"), required=True)
    config.add_argument("--level-path", required=True)
    config.add_argument("--config-path", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "write-config":
        print(json.dumps({"config": str(write_config_for_manifest_level(args.manifest, args.split, args.level_path, args.config_path)), "level_path": args.level_path}, indent=2))
        return
    options = CollectionOptions(count=args.count, fps=args.fps, duration=args.duration, display=args.display, workers=args.workers, agent_port_base=args.agent_port_base, game_port_base=args.game_port_base)
    targets = CollectionTargets(train=args.train_target, dev=args.dev_target)
    if targets.train < 1 or targets.dev < 1:
        raise ValueError("train and dev targets must be positive")
    output_root = args.command_output_root.resolve(strict=True)
    episodes, summary = build_collection_plan(discover_level_entries(engine_dir_for(ROOT, args.os)), output_root=output_root, options=options, targets=targets, seed=args.seed)
    plan_path = write_collection_plan(args.output_dir, output_root=output_root, episodes=episodes, summary=summary, options=options, targets=targets, seed=args.seed)
    commands_path = args.commands_path or args.output_dir / "collect_train_dev.sh"
    _atomic_write(commands_path, generate_collection_commands(plan_path, output_root=output_root, options=options), executable=True)
    counts = json.loads(plan_path.read_text(encoding="utf-8"))["counts"]
    print(json.dumps({"plan": str(plan_path), "commands": str(commands_path), "counts": counts, "bucket_summary": {"buckets": len(summary) // 2, "train_target": targets.train, "dev_target": targets.dev}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
