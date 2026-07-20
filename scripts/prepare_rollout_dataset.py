#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sciencebirdsagents.Utils.PrepareTestConfig import write_config  # noqa: E402


SCHEMA = "novphy-rollout-dataset-partitions-v1"
DEFAULT_OUTPUT_DIR = Path("data/rollout_dataset_plan")
NOVELTY_LEVELS = tuple(f"novelty_level_{index}" for index in range(1, 9))


@dataclass(frozen=True)
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
    def from_json(cls, data: dict[str, str]) -> "LevelEntry":
        return cls(
            novelty_level=data["novelty_level"],
            level_type=data["level_type"],
            relative_path=data["relative_path"],
        )


@dataclass(frozen=True)
class CollectionOptions:
    count: int = 2
    fps: float = 30.0
    duration: float = 5.0
    display: str = ":149"
    ui_settle_seconds: float = 5.0
    connect_timeout: float = 60.0
    prepare_timeout: float = 90.0
    read_timeout: float = 420.0
    speed: int = 1
    workers: int = 1


@dataclass(frozen=True)
class WorkerSpec:
    index: int
    display: str
    agent_port: int
    game_port: int


def engine_dir_for(root: Path = ROOT, operating_system: str = "Linux") -> Path:
    return root / "sciencebirdsgames" / operating_system


def _levels_root(engine_dir: Path) -> Path:
    return engine_dir / "9001_Data" / "StreamingAssets" / "Levels"


def discover_level_entries(engine_dir: Path) -> list[LevelEntry]:
    levels_root = _levels_root(engine_dir)
    if not levels_root.is_dir():
        raise FileNotFoundError(f"Levels directory not found: {levels_root}")

    entries: list[LevelEntry] = []
    novelty_dirs = sorted(path for path in levels_root.iterdir() if path.is_dir() and path.name.startswith("novelty_level_"))
    for novelty_dir in novelty_dirs:
        type_dirs = sorted(path for path in novelty_dir.iterdir() if path.is_dir() and path.name.startswith("type010"))
        for type_dir in type_dirs:
            level_dir = type_dir / "Levels"
            if not level_dir.is_dir():
                continue
            for xml_path in sorted(level_dir.glob("*.xml")):
                entries.append(
                    LevelEntry(
                        novelty_level=novelty_dir.name,
                        level_type=type_dir.name,
                        relative_path=xml_path.relative_to(engine_dir).as_posix(),
                    )
                )

    if not entries:
        raise RuntimeError(f"No NovPhy level XML files found under {levels_root}")
    return entries


def _stable_key(entry: LevelEntry, seed: str) -> str:
    payload = f"{seed}\0{entry.relative_path}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        train_count = total - 2
        dev_count = 1
    test_count = total - train_count - dev_count
    return train_count, dev_count, test_count


def partition_levels(
    entries: Iterable[LevelEntry],
    *,
    seed: str = "novphy-rollout-dataset-v1",
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
) -> dict[str, list[LevelEntry]]:
    grouped: dict[str, list[LevelEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.bucket, []).append(entry)

    partitions = {"train": [], "dev": [], "test": []}
    for bucket in sorted(grouped):
        bucket_entries = sorted(grouped[bucket], key=lambda item: _stable_key(item, seed))
        train_count, dev_count, _test_count = _split_counts(len(bucket_entries), train_ratio, dev_ratio)
        partitions["train"].extend(bucket_entries[:train_count])
        partitions["dev"].extend(bucket_entries[train_count : train_count + dev_count])
        partitions["test"].extend(bucket_entries[train_count + dev_count :])

    return {name: sorted(items, key=lambda item: item.relative_path) for name, items in partitions.items()}


def write_partition_manifest(output_dir: Path, partitions: dict[str, list[LevelEntry]], *, seed: str = "novphy-rollout-dataset-v1") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {name: len(items) for name, items in partitions.items()}
    counts["total"] = sum(counts.values())
    manifest = {
        "schema": SCHEMA,
        "seed": seed,
        "counts": counts,
        "splits": {
            name: [entry.to_json() for entry in items]
            for name, items in partitions.items()
        },
    }
    path = output_dir / "partitions.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load_partition_manifest(path: Path) -> dict[str, list[LevelEntry]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported partition manifest schema: {manifest.get('schema')}")
    return {
        name: [LevelEntry.from_json(entry) for entry in entries]
        for name, entries in manifest["splits"].items()
    }


def _round_robin_novelty_entries(entries: Iterable[LevelEntry], split: str) -> list[LevelEntry]:
    entries_by_novelty = {novelty_level: [] for novelty_level in NOVELTY_LEVELS}
    for entry in entries:
        if entry.novelty_level in entries_by_novelty:
            entries_by_novelty[entry.novelty_level].append(entry)

    ordered = [
        entries_by_novelty[novelty_level][entry_index]
        for entry_index in range(max(len(level_entries) for level_entries in entries_by_novelty.values()))
        for novelty_level in NOVELTY_LEVELS
        if entry_index < len(entries_by_novelty[novelty_level])
    ]
    if not ordered:
        raise ValueError(f"Requested split contains no novelty levels 1 through 8: {split}")
    return ordered


def _format_number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return str(float(value)).rstrip("0").rstrip(".")


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _require_single_line_shell_value(name: str, value: str | Path) -> str:
    text = str(value)
    if "\n" in text or "\r" in text:
        raise ValueError(f"{name} must not contain newline characters")
    return text


def _safe_output_name(entry: LevelEntry) -> str:
    return f"{entry.novelty_level}_{entry.level_type}_{entry.stem}".replace("/", "_")


def _worker_display(base_display: str, worker_index: int) -> str:
    if not base_display.startswith(":"):
        raise ValueError("display must be an X display like :149 for parallel workers")
    try:
        base_number = int(base_display[1:])
    except ValueError as exc:
        raise ValueError("display must be an X display like :149 for parallel workers") from exc
    return f":{base_number + worker_index}"


def _worker_specs(opts: CollectionOptions) -> list[WorkerSpec]:
    if opts.workers < 1:
        raise ValueError("workers must be at least 1")
    if opts.workers == 1:
        return [WorkerSpec(index=0, display=opts.display, agent_port=2004, game_port=9001)]
    return [
        WorkerSpec(index=index, display=_worker_display(opts.display, index), agent_port=2004 + index * 10, game_port=9001 + index * 10)
        for index in range(opts.workers)
    ]


def _collection_command_lines(
    *,
    output_dir: str | Path,
    opts: CollectionOptions,
    display: str,
    game_dir: str | None = None,
    port: str | None = None,
    engine_agent_port: str | None = None,
    engine_game_port: str | None = None,
) -> list[str]:
    args = [
        f"--output-dir {_quote(output_dir)}",
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
    ]
    if game_dir is not None:
        args.append(f"--game-dir {game_dir}")
    if port is not None:
        args.append(f"--port {port}")
    if engine_agent_port is not None:
        args.append(f"--engine-agent-port {engine_agent_port}")
    if engine_game_port is not None:
        args.append(f"--engine-game-port {engine_game_port}")

    lines = [
        f"DISPLAY={display} LD_LIBRARY_PATH=\"$CONDA_PREFIX/lib:${{LD_LIBRARY_PATH-}}\" \\",
        "  python scripts/collect_rollouts.py \\",
    ]
    for index, arg in enumerate(args):
        suffix = " \\" if index < len(args) - 1 else ""
        lines.append(f"    {arg}{suffix}")
    return lines


def _append_level_collection(
    lines: list[str],
    *,
    manifest: Path,
    safe_split: str,
    level_path: str,
    output_dir: str | Path,
    opts: CollectionOptions,
    display: str,
    failure_ledger: str,
    config_path: str | None = None,
    game_dir: str | None = None,
    port: str | None = None,
    engine_agent_port: str | None = None,
    engine_game_port: str | None = None,
) -> None:
    write_config = (
        "if python scripts/prepare_rollout_dataset.py write-config "
        f"--manifest {_quote(manifest)} "
        f"--split {_quote(safe_split)} "
        f"--level-path {_quote(level_path)}"
    )
    if config_path is not None:
        write_config += f" --config-path {config_path}"
    write_config += " &&"
    lines.extend(
        [
            f"# {safe_split}: {level_path}",
            write_config,
            *_collection_command_lines(
                output_dir=output_dir,
                opts=opts,
                display=display,
                game_dir=game_dir,
                port=port,
                engine_agent_port=engine_agent_port,
                engine_game_port=engine_game_port,
            ),
            "then",
            f"    printf 'Completed %s: %s\\n' {_quote(safe_split)} {_quote(level_path)}",
            "else",
            "    status=$?",
            "    failure_count=$((failure_count + 1))",
            f"    printf '%s\\t%s\\t%s\\t%s\\n' {_quote(safe_split)} {_quote(level_path)} {_quote(output_dir)} \"$status\" >> {failure_ledger}",
            f"    printf 'Failed %s: %s (status %s). Continuing with the next level.\\n' {_quote(safe_split)} {_quote(level_path)} \"$status\" >&2",
            "fi",
            "",
        ]
    )


def _generate_collection_commands_serial_legacy(
    manifest_path: Path,
    *,
    output_root: Path,
    options: CollectionOptions | None = None,
    splits: tuple[str, ...] = ("train", "dev"),
) -> str:
    opts = options or CollectionOptions()
    display = _require_single_line_shell_value("display", opts.display)
    manifest = Path(_require_single_line_shell_value("manifest_path", manifest_path))
    output_base = Path(_require_single_line_shell_value("output_root", output_root))
    failure_ledger = manifest.parent / "failed_levels.tsv"
    partitions = load_partition_manifest(manifest_path)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "source ~/cd_novphy",
        "",
        f"failure_ledger={_quote(failure_ledger)}",
        "failure_count=0",
        "printf 'split\\tlevel_path\\toutput_dir\\tstatus\\n' > \"$failure_ledger\"",
        "",
        "# Start Xvnc separately before running this script, for example:",
        f"# Xvnc {display} -geometry 1024x768 -depth 24 -SecurityTypes None -rfbport 0 >/tmp/novphy_rollout_xvnc_${{USER}}.log 2>&1 &",
        "",
    ]
    for split in splits:
        safe_split = _require_single_line_shell_value("split", split)
        for entry in _round_robin_novelty_entries(partitions.get(split, []), safe_split):
            level_path = _require_single_line_shell_value("level_path", entry.relative_path)
            output_name = _require_single_line_shell_value("output_name", _safe_output_name(entry))
            output_dir = output_base / safe_split / output_name
            lines.extend(
                [
                    f"# {safe_split}: {level_path}",
                    "if python scripts/prepare_rollout_dataset.py write-config "
                    f"--manifest {_quote(manifest)} "
                    f"--split {_quote(safe_split)} "
                    f"--level-path {_quote(level_path)} &&",
                    f"DISPLAY={_quote(display)} LD_LIBRARY_PATH=\"$CONDA_PREFIX/lib:${{LD_LIBRARY_PATH-}}\" \\",
                    "  python scripts/collect_rollouts.py \\",
                    f"    --output-dir {_quote(output_dir)} \\",
                    "    --capture-source desktop \\",
                    "    --fresh-engine-per-rollout \\",
                    "    --ui-level 1 \\",
                    f"    --ui-settle-seconds {_format_number(opts.ui_settle_seconds)} \\",
                    f"    --count {_format_number(opts.count)} \\",
                    f"    --fps {_format_number(opts.fps)} \\",
                    f"    --duration {_format_number(opts.duration)} \\",
                    f"    --connect-timeout {_format_number(opts.connect_timeout)} \\",
                    f"    --prepare-timeout {_format_number(opts.prepare_timeout)} \\",
                    f"    --read-timeout {_format_number(opts.read_timeout)} \\",
                    f"    --speed {_format_number(opts.speed)}",
                    "then",
                    f"    printf 'Completed %s: %s\\n' {_quote(safe_split)} {_quote(level_path)}",
                    "else",
                    "    status=$?",
                    "    failure_count=$((failure_count + 1))",
                    f"    printf '%s\\t%s\\t%s\\t%s\\n' {_quote(safe_split)} {_quote(level_path)} {_quote(output_dir)} \"$status\" >> \"$failure_ledger\"",
                    f"    printf 'Failed %s: %s (status %s). Continuing with the next level.\\n' {_quote(safe_split)} {_quote(level_path)} \"$status\" >&2",
                    "fi",
                    "",
                ]
            )
    lines.extend(
        [
            "if [[ \"$failure_count\" -gt 0 ]]; then",
            "    echo \"Completed with $failure_count failed level(s). See $failure_ledger\" >&2",
            "    exit 1",
            "fi",
            "echo \"Completed all requested train/dev levels. Failure ledger: $failure_ledger\"",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def generate_collection_commands(
    manifest_path: Path,
    *,
    output_root: Path,
    options: CollectionOptions | None = None,
    splits: tuple[str, ...] = ("train", "dev"),
) -> str:
    opts = options or CollectionOptions()
    display = _require_single_line_shell_value("display", opts.display)
    manifest = Path(_require_single_line_shell_value("manifest_path", manifest_path))
    output_base = Path(_require_single_line_shell_value("output_root", output_root))
    failure_ledger = manifest.parent / "failed_levels.tsv"
    partitions = load_partition_manifest(manifest_path)
    specs = _worker_specs(opts)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "source ~/cd_novphy",
        "",
        f"failure_ledger={_quote(failure_ledger)}",
        "failure_count=0",
        f"worker_count={opts.workers}",
        "printf 'split\\tlevel_path\\toutput_dir\\tstatus\\n' > \"$failure_ledger\"",
        "",
        "# Start Xvnc separately before running this script, for example:",
        f"# Xvnc {display} -geometry 1024x768 -depth 24 -SecurityTypes None -rfbport 0 >/tmp/novphy_rollout_xvnc_${{USER}}.log 2>&1 &",
        "",
    ]

    if opts.workers == 1:
        for split in splits:
            safe_split = _require_single_line_shell_value("split", split)
            for entry in _round_robin_novelty_entries(partitions.get(split, []), safe_split):
                level_path = _require_single_line_shell_value("level_path", entry.relative_path)
                output_name = _require_single_line_shell_value("output_name", _safe_output_name(entry))
                output_dir = output_base / safe_split / output_name
                _append_level_collection(
                    lines,
                    manifest=manifest,
                    safe_split=safe_split,
                    level_path=level_path,
                    output_dir=output_dir,
                    opts=opts,
                    display=_quote(display),
                    failure_ledger='"$failure_ledger"',
                )
    else:
        worker_entries: list[list[tuple[str, LevelEntry]]] = [[] for _ in specs]
        ordinal = 0
        for split in splits:
            safe_split = _require_single_line_shell_value("split", split)
            for entry in _round_robin_novelty_entries(partitions.get(split, []), safe_split):
                worker_entries[ordinal % len(specs)].append((safe_split, entry))
                ordinal += 1

        lines.extend(
            [
                "run_worker() {",
                "    local worker_index=\"$1\"",
                "    local display_id=\"$2\"",
                "    local agent_port=\"$3\"",
                "    local game_port=\"$4\"",
                "    local worker_root",
                "    worker_root=\"$(mktemp -d \"${TMPDIR:-/tmp}/novphy_rollout_worker_${worker_index}_XXXXXX\")\"",
                "    trap 'rm -rf \"$worker_root\"' RETURN",
                "    local worker_engine_dir",
                "    worker_engine_dir=\"$worker_root/engine\"",
                "    cp -a sciencebirdsgames/Linux \"$worker_engine_dir\"",
                "    local failure_count=0",
                "",
            ]
        )
        for spec in specs:
            lines.append(f"    if [[ \"$worker_index\" == \"{spec.index}\" ]]; then")
            for safe_split, entry in worker_entries[spec.index]:
                level_path = _require_single_line_shell_value("level_path", entry.relative_path)
                output_name = _require_single_line_shell_value("output_name", _safe_output_name(entry))
                output_dir = output_base / safe_split / output_name
                _append_level_collection(
                    lines,
                    manifest=manifest,
                    safe_split=safe_split,
                    level_path=level_path,
                    output_dir=output_dir,
                    opts=opts,
                    display='"$display_id"',
                    failure_ledger='"$failure_ledger"',
                    config_path='"$worker_engine_dir/config.xml"',
                    game_dir='"$worker_engine_dir"',
                    port='"$agent_port"',
                    engine_agent_port='"$agent_port"',
                    engine_game_port='"$game_port"',
                )
            lines.append("    fi")
        lines.extend(["    return \"$failure_count\"", "}", ""])
        lines.append("worker_pids=()")
        lines.extend(
            [
                "cleanup_workers() {",
                "    for worker_pid in \"${worker_pids[@]}\"; do",
                "        kill \"$worker_pid\" 2>/dev/null || true",
                "        wait \"$worker_pid\" 2>/dev/null || true",
                "    done",
                "}",
                "trap cleanup_workers EXIT INT TERM",
            ]
        )
        for spec in specs:
            lines.append(f"run_worker {spec.index} '{spec.display}' {spec.agent_port} {spec.game_port} &")
            lines.append("worker_pids+=(\"$!\")")
        lines.extend(
            [
                "for worker_pid in \"${worker_pids[@]}\"; do",
                "    if ! wait \"$worker_pid\"; then",
                "        failure_count=$((failure_count + 1))",
                "    fi",
                "done",
                "trap - EXIT INT TERM",
                "",
            ]
        )

    lines.extend(
        [
            "if [[ \"$failure_count\" -gt 0 ]]; then",
            "    echo \"Completed with $failure_count failed level(s). See $failure_ledger\" >&2",
            "    exit 1",
            "fi",
            "echo \"Completed all requested train/dev levels. Failure ledger: $failure_ledger\"",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_commands(path: Path, commands: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(commands, encoding="utf-8")
    path.chmod(0o755)
    return path


def write_config_for_manifest_level(manifest_path: Path, split: str, level_path: str, config_path: Path) -> Path:
    partitions = load_partition_manifest(manifest_path)
    split_entries = partitions.get(split)
    if split_entries is None:
        raise ValueError(f"Unknown split: {split}")
    if level_path not in {entry.relative_path for entry in split_entries}:
        raise ValueError(f"Level path is not part of split {split}: {level_path}")
    write_config(config_path, [level_path])
    return config_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare deterministic NovPhy rollout dataset partitions and dry-run commands")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Write partitions.json and a train/dev collection command script")
    plan.add_argument("--os", default="Linux")
    plan.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    plan.add_argument("--seed", default="novphy-rollout-dataset-v1")
    plan.add_argument("--command-output-root", type=Path, default=Path("data/rollout_dataset"))
    plan.add_argument("--commands-path", type=Path)
    plan.add_argument("--count", type=int, default=2)
    plan.add_argument("--fps", type=float, default=30.0)
    plan.add_argument("--duration", type=float, default=5.0)
    plan.add_argument("--display", default=":149")
    plan.add_argument("--workers", type=int, default=1)

    write_config_parser = subparsers.add_parser("write-config", help="Write config.xml for one level from a partition manifest")
    write_config_parser.add_argument("--manifest", type=Path, required=True)
    write_config_parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    write_config_parser.add_argument("--level-path", required=True)
    write_config_parser.add_argument("--os", default="Linux")
    write_config_parser.add_argument("--config-path", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "plan":
        engine_dir = engine_dir_for(ROOT, args.os)
        entries = discover_level_entries(engine_dir)
        partitions = partition_levels(entries, seed=args.seed)
        manifest_path = write_partition_manifest(args.output_dir, partitions, seed=args.seed)
        commands_path = args.commands_path or args.output_dir / "collect_train_dev.sh"
        commands = generate_collection_commands(
            manifest_path,
            output_root=args.command_output_root,
            options=CollectionOptions(count=args.count, fps=args.fps, duration=args.duration, display=args.display, workers=args.workers),
        )
        write_commands(commands_path, commands)
        print(json.dumps({"manifest": str(manifest_path), "commands": str(commands_path), "counts": {name: len(items) for name, items in partitions.items()}}, indent=2))
        return

    if args.command == "write-config":
        config_path = args.config_path or engine_dir_for(ROOT, args.os) / "config.xml"
        written = write_config_for_manifest_level(args.manifest, args.split, args.level_path, config_path)
        print(json.dumps({"config": str(written), "level_path": args.level_path}, indent=2))
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
