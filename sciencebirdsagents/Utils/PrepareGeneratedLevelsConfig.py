import argparse
import sys
from pathlib import Path

from PrepareTestConfig import engine_dir_for, write_config


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_levels_dir(root: Path, operating_system: str) -> Path:
    return root / "sciencebirdsgames" / operating_system / "9001_Data" / "StreamingAssets" / "Levels" / "iratus_aves" / "Levels"


def discover_generated_levels(root: Path, operating_system: str, levels_dir: Path | None, max_levels: int | None) -> list[str]:
    engine_dir = engine_dir_for(root, operating_system)
    source_dir = levels_dir or default_levels_dir(root, operating_system)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Generated level directory not found: {source_dir}")

    level_files = sorted(source_dir.glob("*.xml"))
    if max_levels is not None:
        level_files = level_files[:max_levels]
    if not level_files:
        raise RuntimeError(f"No generated level XML files found in {source_dir}")

    return [path.relative_to(engine_dir).as_posix() for path in level_files]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Science Birds config.xml for generated levels")
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument("--os", default="Linux")
    parser.add_argument("--levels-dir", type=Path)
    parser.add_argument("--max-levels", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        level_paths = discover_generated_levels(args.root, args.os, args.levels_dir, args.max_levels)
        config_path = engine_dir_for(args.root, args.os) / "config.xml"
        write_config(config_path, level_paths)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Wrote {config_path} with {len(level_paths)} generated levels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
