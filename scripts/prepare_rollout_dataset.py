#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Literal, TypedDict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sciencebirdsagents.Utils.PrepareTestConfig import write_config  # noqa: E402
from scripts.rollout_artifacts import (  # noqa: E402
    EpisodeAccepted,
    EpisodeValidationContract,
    validate_rollout_episode,
)
from scripts.scenario_manifest import (  # noqa: E402
    BenchmarkCondition,
    ELIGIBLE,
    SMOKE_ONLY,
    ScenarioManifest,
    import_legacy_manifest,
    load_manifest as load_scenario_manifest,
    load_scenario_manifest_projection,
    require_research_eligible,
    scenario_manifest_projection,
)


SCHEMA: Final = "novphy-rollout-dataset-partitions-v1"
PLAN_SCHEMA: Final = "novphy-rollout-collection-plan-v2"
DEFAULT_OUTPUT_DIR: Final = Path("data/rollout_dataset_plan")
DEFAULT_SEED: Final = "novphy-rollout-dataset-v1"
DEFAULT_AGENT_PORT_BASE: Final = 2004
DEFAULT_GAME_PORT_BASE: Final = 9001
WORKER_PORT_STRIDE: Final = 10
MAX_PORT: Final = 65535
COLLECTION_SPLITS: Final = ("train", "dev", "test")
DEFAULT_COLLECTION_SPLITS: Final = ("train", "dev")
CollectionPurpose = Literal["research", "smoke"]
RESEARCH_PURPOSE: Final[CollectionPurpose] = "research"
SMOKE_PURPOSE: Final[CollectionPurpose] = "smoke"
#: The production NovPhy level inventory is exactly 40 normal (novelty_level_0) plus
#: 8 x 5 novel buckets.  A plan that discovers a different count has a truncated
#: inventory, so the default is fail-closed; a caller with a deliberately scoped
#: inventory must declare its bucket count explicitly.
PRODUCTION_BUCKET_COUNT: Final = 80
#: Production level types are named `type010101` .. `type010805`.
PRODUCTION_LEVEL_TYPE_PREFIX: Final = "type010"
PHYSICS_CAPTURE_CONTRACT: Final = "physics_capture_v1"
PHYSICS_PLAYER_VERSION: Final = "2019.4.41f2-physics-v1"
ACTIVE_COHORT_ROOT: Final = ROOT / "data" / "novphy_rollouts_dataset_20260708_171531"


class JsonValue(TypedDict, total=False):
    pass


@dataclass(frozen=True, slots=True)
class LevelEntry:
    novelty_level: str
    level_type: str
    relative_path: str
    scenario_manifest: ScenarioManifest | None = None
    scenario_manifest_reference: str | None = None

    def __post_init__(self) -> None:
        if self.scenario_manifest is not None:
            condition = self.scenario_manifest.benchmark_condition
            if (condition.novelty_level, condition.novelty_type) != (self.novelty_level, self.level_type):
                raise ValueError("Scenario manifest benchmark condition does not match level entry")

    @property
    def bucket(self) -> str:
        return f"{self.novelty_level}/{self.level_type}"

    @property
    def stem(self) -> str:
        return Path(self.relative_path).stem

    def to_json(self) -> dict[str, object]:
        data: dict[str, object] = {
            "novelty_level": self.novelty_level,
            "level_type": self.level_type,
            "bucket": self.bucket,
            "relative_path": self.relative_path,
        }
        if self.scenario_manifest is not None:
            data.update(scenario_manifest_projection(self.scenario_manifest, self.scenario_manifest_reference))
        return data

    @classmethod
    def from_json(cls, data: dict[str, object], *, manifest_required: bool = False) -> LevelEntry:
        for key in ("novelty_level", "level_type", "relative_path"):
            if not isinstance(data.get(key), str):
                raise ValueError(f"Level entry has invalid {key}")
        manifest, reference = load_scenario_manifest_projection(data, required=manifest_required)
        return cls(
            data["novelty_level"],
            data["level_type"],
            data["relative_path"],
            manifest,
            reference,
        )


@dataclass(frozen=True, slots=True)
class CollectionTargets:
    train: int = 100
    dev: int = 20
    test: int = 0

    def for_split(self, split: str) -> int:
        match split:
            case "train":
                return self.train
            case "dev":
                return self.dev
            case "test":
                return self.test
            case _:
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
class PhysicsCaptureProvenance:
    archive: Path
    smoke_marker: Path
    player_version: str
    protocol_version: str
    archive_path: str


def resolve_physics_capture_provenance(archive: Path, smoke_marker: Path) -> PhysicsCaptureProvenance:
    if not archive.is_file() and not archive.is_dir():
        raise ValueError(f"staged player archive is required: {archive}")
    if not smoke_marker.is_file():
        raise ValueError(f"physics smoke marker is required: {smoke_marker}")
    marker = _read_json(smoke_marker)
    if marker is None:
        raise ValueError("physics smoke marker must be a JSON object")
    if marker.get("status") != "accepted":
        raise ValueError("physics smoke marker must report status=accepted")
    if marker.get("phase") != "complete":
        raise ValueError("physics smoke marker must report phase=complete")
    if marker.get("protected_unchanged") is not True:
        raise ValueError("physics smoke marker must report protected_unchanged=true")
    accepted_shot = marker.get("accepted_shot")
    if not isinstance(accepted_shot, str) or not accepted_shot.strip():
        raise ValueError("physics smoke marker must contain a nonempty accepted_shot")
    provenance = marker.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("physics smoke marker must contain a provenance object")
    values = tuple(provenance.get(name) for name in ("player_version", "protocol_version", "archive_path"))
    if not all(isinstance(value, str) and value for value in values):
        raise ValueError("physics smoke marker provenance values must be nonempty strings")
    declared_archive = Path(provenance["archive_path"]).resolve(strict=False)
    resolved_archive = archive.resolve(strict=True)
    if declared_archive != resolved_archive:
        raise ValueError("physics smoke marker archive_path does not match the supplied player archive")
    return PhysicsCaptureProvenance(
        resolved_archive,
        smoke_marker.resolve(strict=True),
        provenance["player_version"],
        provenance["protocol_version"],
        provenance["archive_path"],
    )


@dataclass(frozen=True, slots=True)
class PlannedEpisode:
    split: str
    entry: LevelEntry
    output_dir: Path
    source: str

    def to_json(self, output_root: Path) -> dict[str, object]:
        return {
            **self.entry.to_json(),
            "split": self.split,
            "output_path": self.output_dir.relative_to(output_root).as_posix(),
            "source": self.source,
        }


def engine_dir_for(root: Path = ROOT, operating_system: str = "Linux") -> Path:
    return root / "sciencebirdsgames" / operating_system


def _levels_root(engine_dir: Path) -> Path:
    return engine_dir / "9001_Data" / "StreamingAssets" / "Levels"


def discover_level_entries(engine_dir: Path, level_type_prefix: str = PRODUCTION_LEVEL_TYPE_PREFIX) -> list[LevelEntry]:
    """Inventory level XMLs under an engine directory.

    `level_type_prefix` defaults to the production `type010*` naming.  A deliberately
    scoped inventory (for example the staged physics player, whose single level lives
    under `type2`) must name its prefix explicitly, so a foreign or truncated level
    tree is never planned by accident.
    """
    if not level_type_prefix.strip():
        raise ValueError("level_type_prefix must be nonempty")
    levels_root = _levels_root(engine_dir)
    if not levels_root.is_dir():
        raise FileNotFoundError(f"Levels directory not found: {levels_root}")
    entries: list[LevelEntry] = []
    for novelty_dir in sorted(path for path in levels_root.iterdir() if path.is_dir() and path.name.startswith("novelty_level_")):
        for type_dir in sorted(path for path in novelty_dir.iterdir() if path.is_dir() and path.name.startswith(level_type_prefix)):
            for xml_path in sorted((type_dir / "Levels").glob("*.xml")):
                relative_path = xml_path.relative_to(engine_dir).as_posix()
                manifest_path = xml_path.with_suffix(".scenario.json")
                if manifest_path.is_file():
                    manifest = load_scenario_manifest(manifest_path, xml_path)
                    reference = manifest_path.relative_to(engine_dir).as_posix()
                else:
                    is_staged_type2 = type_dir.name == "type2"
                    manifest = import_legacy_manifest(
                        xml_path.read_bytes(),
                        benchmark_condition=BenchmarkCondition(novelty_dir.name, type_dir.name),
                        source_path=relative_path,
                        eligibility=SMOKE_ONLY if is_staged_type2 else ELIGIBLE,
                        eligibility_reason=(
                            "Sidecar-less type2 content is approved only for bounded runtime and smoke collection."
                            if is_staged_type2
                            else None
                        ),
                    )
                    reference = None
                entries.append(LevelEntry(novelty_dir.name, type_dir.name, relative_path, manifest, reference))
    if not entries:
        raise RuntimeError(f"No NovPhy level XML files found under {levels_root}")
    return entries


def _stable_key(entry: LevelEntry, seed: str) -> str:
    # deterministic non-cryptographic derivation, not an integrity check
    value = f"{seed}\0{entry.relative_path}".encode("utf-8")
    return f"{zlib.crc32(value):08x}"


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
            if not isinstance(value, dict):
                raise ValueError(f"Partition manifest contains malformed {split} entry")
            try:
                entries.append(LevelEntry.from_json(value))
            except ValueError as exc:
                raise ValueError(f"Partition manifest contains malformed {split} entry: {exc}") from exc
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


def _is_canonically_complete_fresh_engine_episode(
    output_dir: Path,
    opts: CollectionOptions,
    *,
    level_five: bool = False,
    capture_contract: str | None = None,
) -> bool:
    result = validate_rollout_episode(
        output_dir,
        EpisodeValidationContract(opts.count, opts.fps, opts.duration, level_five),
        capture_contract=capture_contract,
    )
    return isinstance(result, EpisodeAccepted)


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


def _selected_splits(selected_splits: tuple[str, ...]) -> tuple[str, ...]:
    if not selected_splits:
        raise ValueError("At least one collection split must be selected")
    if len(set(selected_splits)) != len(selected_splits):
        raise ValueError("Collection splits must not contain duplicates")
    unsupported = set(selected_splits) - set(COLLECTION_SPLITS)
    if unsupported:
        raise ValueError(f"Unsupported collection splits: {', '.join(sorted(unsupported))}")
    return selected_splits


def _validated_collection_purpose(value: object) -> CollectionPurpose:
    if value == RESEARCH_PURPOSE:
        return RESEARCH_PURPOSE
    if value == SMOKE_PURPOSE:
        return SMOKE_PURPOSE
    raise ValueError("collection_purpose must be research or smoke")


def _require_collection_admission(entry: LevelEntry, purpose: CollectionPurpose, use: str) -> None:
    if purpose != RESEARCH_PURPOSE:
        return
    if entry.scenario_manifest is None:
        raise ValueError(f"Level instance is missing a scenario manifest required for {use}")
    require_research_eligible(entry.scenario_manifest, use)


def _plan_bucket(
    entries: list[LevelEntry],
    *,
    split: str,
    output_root: Path,
    opts: CollectionOptions,
    targets: CollectionTargets,
    seed: str,
    capture_contract: str | None = None,
) -> list[PlannedEpisode]:
    selected: list[PlannedEpisode] = []
    ordered = sorted(entries, key=lambda entry: _stable_key(entry, seed))
    level_five = ordered[0].novelty_level == "novelty_level_5"
    for entry in ordered:
        output_dir = output_root / split / _safe_output_name(entry)
        if _path_is_safe_existing(output_dir, output_root) and _is_canonically_complete_fresh_engine_episode(output_dir, opts, level_five=level_five, capture_contract=capture_contract):
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
    selected_splits: tuple[str, ...] = DEFAULT_COLLECTION_SPLITS,
    seed: str = DEFAULT_SEED,
    capture_contract: str | None = None,
    expected_bucket_count: int = PRODUCTION_BUCKET_COUNT,
    collection_purpose: CollectionPurpose = RESEARCH_PURPOSE,
) -> tuple[list[PlannedEpisode], dict[str, dict[str, int]]]:
    opts = options or CollectionOptions()
    target = targets or CollectionTargets()
    splits = _selected_splits(selected_splits)
    _worker_specs(opts)
    purpose = _validated_collection_purpose(collection_purpose)
    if type(expected_bucket_count) is not int or expected_bucket_count < 1:
        raise ValueError("expected_bucket_count must be a positive integer")
    if not output_root.is_dir():
        raise FileNotFoundError(f"Existing output root is required: {output_root}")
    entry_list = list(entries)
    for entry in entry_list:
        _require_collection_admission(entry, purpose, "research collection planning")
    partitions = partition_levels(entry_list, seed=seed)
    buckets = _bucket_entries(entry_list)
    partition_buckets = {split: _bucket_entries(partitions[split]) for split in splits}
    if len(buckets) != expected_bucket_count:
        raise RuntimeError(f"Expected {expected_bucket_count} normal and novel buckets, found {len(buckets)}")
    plan: list[PlannedEpisode] = []
    summary: dict[str, dict[str, int]] = {}
    for bucket in sorted(buckets):
        for split in splits:
            candidates = partition_buckets[split].get(bucket, [])
            if not candidates:
                raise RuntimeError(f"Bucket {bucket} has no {split} partition capacity")
            selected = _plan_bucket(candidates, split=split, output_root=output_root, opts=opts, targets=target, seed=seed, capture_contract=capture_contract)
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


def _collection_command_lines(episode: PlannedEpisode, opts: CollectionOptions, spec: WorkerSpec, provenance: PhysicsCaptureProvenance | None = None) -> list[str]:
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
    if provenance is not None:
        args.extend([
            "--physics-capture-v1",
            "--physics-host 127.0.0.1",
            f"--physics-port {spec.agent_port + 1}",
            f"--physics-player-version {_quote(provenance.player_version)}",
            f"--physics-protocol-version {_quote(provenance.protocol_version)}",
            f"--physics-archive-path {_quote(provenance.archive_path)}",
        ])
    lines = [
        '  DISPLAY="$display_id" LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH-}" \\',
        "    python scripts/collect_rollouts.py \\",
    ]
    for index, arg in enumerate(args):
        continuation = " \\" if index < len(args) - 1 else ""
        lines.append(f"    {arg}{continuation}")
    return lines


def _append_scheduled_episode(lines: list[str], episode: PlannedEpisode, opts: CollectionOptions, spec: WorkerSpec, provenance: PhysicsCaptureProvenance | None = None) -> None:
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
            *_collection_command_lines(episode, opts, spec, provenance),
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
    splits: tuple[str, ...] | None = None,
    physics_provenance: PhysicsCaptureProvenance | None = None,
) -> str:
    opts = options or CollectionOptions()
    specs = _worker_specs(opts)
    artifact = _read_json(plan_path)
    if artifact is None or artifact.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"Unsupported collection plan schema: {None if artifact is None else artifact.get('schema')}")
    artifact_splits = artifact.get("selected_splits")
    if not isinstance(artifact_splits, list) or not all(isinstance(split, str) for split in artifact_splits):
        raise ValueError("Collection plan must contain selected_splits")
    plan_splits = _selected_splits(tuple(artifact_splits))
    if splits is not None and _selected_splits(splits) != plan_splits:
        raise ValueError("Requested command splits do not match the collection plan")
    purpose = _validated_collection_purpose(artifact.get("collection_purpose", RESEARCH_PURPOSE))
    selected = artifact.get("selected")
    if not isinstance(selected, list):
        raise ValueError("Collection plan must contain selected episodes")
    scheduled: list[PlannedEpisode] = []
    for value in selected:
        if not isinstance(value, dict) or value.get("source") != "scheduled" or value.get("split") not in plan_splits:
            continue
        output_path = value.get("output_path")
        if not isinstance(value.get("split"), str) or not isinstance(output_path, str):
            raise ValueError("Collection plan contains malformed scheduled episode")
        entry = LevelEntry.from_json(value)
        _require_collection_admission(entry, purpose, "research command generation")
        scheduled.append(
            PlannedEpisode(
                value["split"],
                entry,
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
    if physics_provenance is not None and output_root.resolve() == ACTIVE_COHORT_ROOT.resolve():
        raise ValueError(f"physics capture cannot target active cohort root: {output_root}")
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "source ~/cd_novphy",
        # The profile chdirs to its own checkout and points PYTHONPATH there, so
        # re-enter the repo this plan was generated in before any relative path
        # (scripts/, the plan artifact, data/) is resolved.  Launching from that same
        # checkout is a no-op; launching from a worktree keeps the worktree's code.
        f"cd -- {_quote(Path.cwd())}",
        'export PYTHONPATH="$PWD"',
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
            "  local worker_root worker_engine_dir worker_archive" if physics_provenance is not None else "  local worker_root worker_engine_dir",
        "  worker_root=\"$(mktemp -d \"${TMPDIR:-/tmp}/novphy_rollout_worker_${worker_index}_XXXXXX\")\"",
        "  trap 'rm -rf \"$worker_root\"' RETURN",
        "  worker_engine_dir=\"$worker_root/engine\"",
    ]
    if physics_provenance is None:
        lines.append("  cp -a sciencebirdsgames/Linux \"$worker_engine_dir\"")
    else:
        if physics_provenance.archive.is_dir():
            lines.extend([
                f"  worker_archive=\"$worker_root/{physics_provenance.archive.name}\"",
                f"  cp -a -- {_quote(physics_provenance.archive)} \"$worker_archive\"",
                "  mkdir -- \"$worker_engine_dir\"",
                "  cp -a -- \"$worker_archive/.\" \"$worker_engine_dir/\"",
            ])
        else:
            lines.extend([
                f"  worker_archive=\"$worker_root/{physics_provenance.archive.name}\"",
                f"  cp -- {_quote(physics_provenance.archive)} \"$worker_archive\"",
                "  mkdir -- \"$worker_engine_dir\"",
                "  tar -xf \"$worker_archive\" -C \"$worker_engine_dir\"",
            ])
    lines.extend([
            "  local failure_count=0",
        ])
    for spec in specs:
        lines.append(f"  if [[ \"$worker_index\" == \"{spec.index}\" ]]; then")
        for episode in striped[spec.index]:
            _append_scheduled_episode(lines, episode, opts, spec, physics_provenance)
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
            f"echo \"Completed newly scheduled {'/'.join(plan_splits)} episodes. Failure ledger: $failure_ledger\"",
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


def _collection_commands_path(output_dir: Path, selected_splits: tuple[str, ...]) -> Path:
    return output_dir / f"collect_{'_'.join(selected_splits)}.sh"


def _remove_opposite_collection_commands(output_dir: Path, commands_path: Path) -> None:
    for splits in (DEFAULT_COLLECTION_SPLITS, COLLECTION_SPLITS):
        path = _collection_commands_path(output_dir, splits)
        if path != commands_path and (path.is_file() or path.is_symlink()):
            path.unlink()


def write_collection_plan(
    output_dir: Path,
    *,
    output_root: Path,
    episodes: Iterable[PlannedEpisode],
    summary: dict[str, dict[str, int]],
    options: CollectionOptions,
    targets: CollectionTargets,
    seed: str,
    selected_splits: tuple[str, ...] = DEFAULT_COLLECTION_SPLITS,
    physics_provenance: PhysicsCaptureProvenance | None = None,
    collection_purpose: CollectionPurpose = RESEARCH_PURPOSE,
) -> Path:
    episode_list = list(episodes)
    splits = _selected_splits(selected_splits)
    purpose = _validated_collection_purpose(collection_purpose)
    if physics_provenance is not None and output_root.resolve() == ACTIVE_COHORT_ROOT.resolve():
        raise ValueError(f"physics capture cannot target active cohort root: {output_root}")
    for planned in episode_list:
        _require_collection_admission(planned.entry, purpose, "research collection plan")
    contract = {"count": options.count, "fps": options.fps, "duration": options.duration, "workers": options.workers, "train_target": targets.train, "dev_target": targets.dev, "test_target": targets.test, "selected_splits": list(splits), "collection_purpose": purpose}
    if physics_provenance is not None:
        contract.update({"capture_contract": PHYSICS_CAPTURE_CONTRACT, "player_version": physics_provenance.player_version, "protocol_version": physics_provenance.protocol_version, "archive_path": physics_provenance.archive_path, "smoke_marker": str(physics_provenance.smoke_marker)})
    payload = {
        "schema": PLAN_SCHEMA,
        "seed": seed,
        "planner_seed": seed,
        "collection_purpose": purpose,
        "output_root": str(output_root.resolve(strict=True)),
        "contract": contract,
        "selected_splits": list(splits),
        "summary": summary,
        "counts": {
            "selected": len(episode_list),
            "existing": sum(item.source == "existing" for item in episode_list),
            "scheduled": sum(item.source == "scheduled" for item in episode_list),
            "normal": sum(item.entry.novelty_level == "novelty_level_0" for item in episode_list),
            "novel": sum(item.entry.novelty_level != "novelty_level_0" for item in episode_list),
            "test": sum(item.split == "test" for item in episode_list),
        },
        "selected": [planned.to_json(output_root) for planned in episode_list],
    }
    return _atomic_write(output_dir / "collection_plan.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_plan_entries(path: Path, split: str) -> list[LevelEntry]:
    payload = _read_json(path)
    if payload is None or payload.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"Unsupported collection plan schema: {None if payload is None else payload.get('schema')}")
    purpose = _validated_collection_purpose(payload.get("collection_purpose", RESEARCH_PURPOSE))
    selected = payload.get("selected")
    if not isinstance(selected, list):
        raise ValueError("Collection plan must contain selected episodes")
    result: list[LevelEntry] = []
    for item in selected:
        if isinstance(item, dict) and item.get("split") == split:
            entry = LevelEntry.from_json(item)
            _require_collection_admission(entry, purpose, "collection plan reload")
            result.append(entry)
    return result


def write_config_for_manifest_level(plan_path: Path, split: str, level_path: str, config_path: Path) -> Path:
    payload = _read_json(plan_path)
    if payload is not None and payload.get("schema") == SCHEMA:
        entries = load_partition_manifest(plan_path).get(split, [])
    else:
        if payload is None:
            raise ValueError(f"Unsupported collection plan schema: {None}")
        artifact_splits = payload.get("selected_splits")
        if not isinstance(artifact_splits, list) or not all(isinstance(selected_split, str) for selected_split in artifact_splits):
            raise ValueError("Collection plan must contain selected_splits")
        if split not in _selected_splits(tuple(artifact_splits)):
            raise ValueError(f"Split {split} was not selected for collection")
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
    plan = subparsers.add_parser("plan", help="Inventory an existing root and atomically publish a capped selected-split plan")
    plan.add_argument("--os", default="Linux")
    plan.add_argument("--engine-dir", type=Path)
    plan.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    plan.add_argument("--command-output-root", type=Path, required=True)
    plan.add_argument("--commands-path", type=Path)
    plan.add_argument("--seed", default=DEFAULT_SEED, help="Planner seed; distinct from each scenario's generation seed")
    plan.add_argument("--collection-purpose", choices=("research", "smoke"), default="research")
    plan.add_argument("--train-target", type=int, default=100)
    plan.add_argument("--dev-target", type=int, default=20)
    plan.add_argument("--test-target", type=int, default=0)
    plan.add_argument("--include-test", action="store_true")
    plan.add_argument("--count", type=int, default=12)
    plan.add_argument("--fps", type=float, default=30.0)
    plan.add_argument("--duration", type=float, default=5.0)
    plan.add_argument("--display", default=":149")
    plan.add_argument("--workers", type=int, default=6)
    plan.add_argument("--agent-port-base", type=_parse_port_base, default=DEFAULT_AGENT_PORT_BASE)
    plan.add_argument("--game-port-base", type=_parse_port_base, default=DEFAULT_GAME_PORT_BASE)
    plan.add_argument("--physics-capture-v1", action="store_true")
    plan.add_argument("--physics-player-dir", type=Path)
    plan.add_argument("--physics-player-archive", type=Path)
    plan.add_argument("--physics-smoke-marker", type=Path)
    plan.add_argument("--expected-buckets", type=int, default=PRODUCTION_BUCKET_COUNT, help="Declared level-inventory bucket count; defaults to the production inventory and fails closed on a truncated one")
    plan.add_argument("--level-type-prefix", default=PRODUCTION_LEVEL_TYPE_PREFIX, help="Level-type directory prefix to inventory; defaults to the production type010* naming")
    plan.add_argument("--train-only", action="store_true", help="Select only the train split; for a scoped inventory too small to fund a leakage-free dev split")
    config = subparsers.add_parser("write-config", help="Write config.xml for a selected plan episode")
    config.add_argument("--manifest", type=Path, required=True)
    config.add_argument("--split", choices=COLLECTION_SPLITS, required=True)
    config.add_argument("--level-path", required=True)
    config.add_argument("--config-path", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "write-config":
        print(json.dumps({"config": str(write_config_for_manifest_level(args.manifest, args.split, args.level_path, args.config_path)), "level_path": args.level_path}, indent=2))
        return
    physics_provenance = None
    if args.physics_capture_v1:
        if args.physics_player_dir is not None and args.physics_player_archive is not None:
            raise ValueError("physics capture accepts a staged player directory or archive, not both")
        if args.physics_player_dir is not None:
            stage_dir = args.physics_player_dir.resolve(strict=True)
            if not stage_dir.is_dir():
                raise ValueError(f"staged player directory is required: {stage_dir}")
            default_archive = stage_dir / "player.tar"
            archive = args.physics_player_archive or (default_archive if default_archive.exists() else stage_dir)
            marker = args.physics_smoke_marker or stage_dir / "physics_capture_v1_smoke.json"
        else:
            if args.physics_player_archive is None or args.physics_smoke_marker is None:
                raise ValueError("physics capture requires --physics-player-dir or --physics-player-archive and --physics-smoke-marker")
            archive = args.physics_player_archive
            marker = args.physics_smoke_marker
        physics_provenance = resolve_physics_capture_provenance(archive, marker)
    options = CollectionOptions(count=args.count, fps=args.fps, duration=args.duration, display=args.display, workers=args.workers, agent_port_base=args.agent_port_base, game_port_base=args.game_port_base)
    targets = CollectionTargets(train=args.train_target, dev=args.dev_target, test=args.test_target)
    if targets.train < 1 or targets.dev < 1:
        raise ValueError("train and dev targets must be positive")
    if targets.test < 0:
        raise ValueError("test target must be non-negative")
    if args.include_test and targets.test < 1:
        raise ValueError("--include-test requires --test-target >= 1")
    if args.train_only and args.include_test:
        raise ValueError("--train-only and --include-test are mutually exclusive")
    if args.train_only:
        selected_splits = ("train",)
    else:
        selected_splits = ("train", "dev", "test") if args.include_test else DEFAULT_COLLECTION_SPLITS
    output_root = args.command_output_root.resolve(strict=True)
    engine_dir = args.engine_dir or engine_dir_for(ROOT, args.os)
    episodes, summary = build_collection_plan(discover_level_entries(engine_dir, args.level_type_prefix), output_root=output_root, options=options, targets=targets, selected_splits=selected_splits, seed=args.seed, capture_contract=PHYSICS_CAPTURE_CONTRACT if physics_provenance is not None else None, expected_bucket_count=args.expected_buckets, collection_purpose=args.collection_purpose)
    plan_path = write_collection_plan(args.output_dir, output_root=output_root, episodes=episodes, summary=summary, options=options, targets=targets, selected_splits=selected_splits, seed=args.seed, physics_provenance=physics_provenance, collection_purpose=args.collection_purpose)
    commands_path = args.commands_path or _collection_commands_path(args.output_dir, selected_splits)
    _atomic_write(commands_path, generate_collection_commands(plan_path, output_root=output_root, options=options, physics_provenance=physics_provenance), executable=True)
    _remove_opposite_collection_commands(args.output_dir, commands_path)
    counts = json.loads(plan_path.read_text(encoding="utf-8"))["counts"]
    print(json.dumps({"plan": str(plan_path), "commands": str(commands_path), "counts": counts, "bucket_summary": {"buckets": len(summary) // len(selected_splits), "train_target": targets.train, "dev_target": targets.dev, "test_target": targets.test, "selected_splits": selected_splits}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
