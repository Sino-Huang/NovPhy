import argparse
from pathlib import Path
from xml.etree import ElementTree as ET


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def engine_dir_for(root: Path, operating_system: str) -> Path:
    return root / "sciencebirdsgames" / operating_system


def discover_level_paths(
    engine_dir: Path,
    novelty_level: str | None = "novelty_level_0",
    level_type: str | None = "type010101",
    max_levels: int | None = 20,
) -> list[str]:
    levels_root = engine_dir / "9001_Data" / "StreamingAssets" / "Levels"
    if not levels_root.is_dir():
        raise FileNotFoundError(f"Levels directory not found: {levels_root}")

    novelty_dirs = [levels_root / novelty_level] if novelty_level else sorted(
        path for path in levels_root.iterdir() if path.is_dir() and path.name.startswith("novelty_level_")
    )

    level_paths: list[Path] = []
    for novelty_dir in novelty_dirs:
        if not novelty_dir.is_dir():
            raise FileNotFoundError(f"Novelty level directory not found: {novelty_dir}")

        type_dirs = [novelty_dir / level_type] if level_type else sorted(
            path for path in novelty_dir.iterdir() if path.is_dir() and path.name.startswith("type010")
        )
        for type_dir in type_dirs:
            levels_dir = type_dir / "Levels"
            if not levels_dir.is_dir():
                raise FileNotFoundError(f"Level type directory not found: {levels_dir}")
            level_paths.extend(sorted(levels_dir.glob("*.xml")))

    if max_levels is not None:
        level_paths = level_paths[:max_levels]

    if not level_paths:
        raise RuntimeError("No matching NovPhy level XML files found")

    return [path.relative_to(engine_dir).as_posix() for path in level_paths]


def write_config(config_path: Path, level_paths: list[str]) -> None:
    evaluation = ET.Element("evaluation")
    ET.SubElement(
        evaluation,
        "novelty_detection_measurement",
        {
            "step": "1",
            "measure_in_training": "False",
            "measure_in_testing": "False",
        },
    )
    trials = ET.SubElement(evaluation, "trials")
    trial = ET.SubElement(
        trials,
        "trial",
        {
            "id": "0",
            "number_of_executions": "1",
            "checkpoint_time_limit": "9999999",
            "checkpoint_interaction_limit": "9999999",
            "notify_novelty": "False",
        },
    )
    game_level_set = ET.SubElement(
        trial,
        "game_level_set",
        {
            "mode": "training",
            "time_limit": "9999999",
            "total_interaction_limit": "9999999",
            "attempt_limit_per_level": "5",
            "allow_level_selection": "True",
        },
    )
    for level_path in level_paths:
        ET.SubElement(game_level_set, "game_levels", {"level_path": level_path})

    config_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(evaluation)
    ET.indent(tree, space="  ")
    tree.write(config_path, encoding="utf-8", xml_declaration=True)


def ensure_java_interface_assets(root: Path, operating_system: str) -> None:
    engine_dir = engine_dir_for(root, operating_system)
    jar_path = engine_dir / "game_playing_interface.jar"
    db_path = engine_dir / "DB"

    if not jar_path.is_file():
        raise FileNotFoundError(
            f"Java interface jar not found: {jar_path}. "
            f"Provision the real Science Birds Java interface assets into {engine_dir}."
        )
    if not db_path.is_dir():
        raise FileNotFoundError(
            f"Java interface DB directory not found: {db_path}. "
            f"Provision the real Science Birds Java interface assets into {engine_dir}."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--os", type=str, default="Linux")
    parser.add_argument("--novelty-level", type=str, default="novelty_level_0")
    parser.add_argument("--level-type", type=str, default="type010101")
    parser.add_argument("--max-levels", type=int, default=20)
    parser.add_argument("--all-levels", action="store_true")
    parser.add_argument("--skip-interface-copy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    engine_dir = engine_dir_for(root, args.os)

    if not args.skip_interface_copy:
        ensure_java_interface_assets(root, args.os)

    level_paths = discover_level_paths(
        engine_dir,
        novelty_level=args.novelty_level,
        level_type=args.level_type,
        max_levels=None if args.all_levels else args.max_levels,
    )
    write_config(engine_dir / "config.xml", level_paths)
    print(f"Wrote {engine_dir / 'config.xml'} with {len(level_paths)} NovPhy levels")


if __name__ == "__main__":
    main()
