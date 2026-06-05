import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_output_dir(root: Path, operating_system: str) -> Path:
    return root / "sciencebirdsgames" / operating_system / "9001_Data" / "StreamingAssets" / "Levels" / "iratus_aves" / "Levels"


def write_parameters(path: Path, levels: int, forbidden: str, pig_range: str, time_limit: int) -> None:
    path.write_text(f"{levels}\n{forbidden}\n{pig_range}\n{time_limit}\n", encoding="utf-8")


def copy_level_xml(source: Path, target: Path) -> None:
    content = source.read_text(encoding="utf-8")
    content = content.replace('encoding="utf-16"', 'encoding="utf-8"')
    content = content.replace("encoding='utf-16'", "encoding='utf-8'")
    content = re.sub(r"(<Camera\b[^>/]*?)>\s*\n<Birds>", r"\1 />\n<Birds>", content, count=1)
    content = re.sub(r"(<Slingshot\b[^>/]*?)>\s*\n<GameObjects>", r"\1 />\n<GameObjects>", content, count=1)
    body = re.sub(r"^\s*<\?xml[^>]*\?>\s*", "", content, count=1)
    root = ET.fromstring(body)
    if root.find("Score") is None:
        root.insert(1, ET.Element("Score", {"highScore": "0"}))
    for platform in root.findall(".//Platform"):
        platform.attrib.setdefault("rotation", "0.0")
        platform.attrib.setdefault("scaleX", "1.0")
        platform.attrib.setdefault("scaleY", "1.0")
    ET.indent(root, space="  ")
    content = '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"
    target.write_text(content, encoding="utf-8")


def generate_levels(
    root: Path,
    operating_system: str,
    levels: int,
    forbidden: str,
    pig_range: str,
    time_limit: int,
    parameters: Path | None,
    output_dir: Path | None,
) -> list[Path]:
    iratus_dir = root / "modules" / "IratusAves"
    generator = iratus_dir / "generator_competition.py"
    if not generator.is_file():
        raise FileNotFoundError(f"IratusAves generator not found: {generator}")

    target_dir = output_dir or default_output_dir(root, operating_system)
    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="iratus-aves-") as tmp:
        work_dir = Path(tmp)
        if parameters is None:
            write_parameters(work_dir / "parameters.txt", levels, forbidden, pig_range, time_limit)
        else:
            shutil.copy2(parameters, work_dir / "parameters.txt")

        subprocess.run([sys.executable, str(generator)], cwd=work_dir, check=True)
        generated = sorted(work_dir.glob("level-*.xml"))
        if not generated:
            raise RuntimeError("IratusAves did not generate any level XML files")

        copied = []
        for source in generated:
            target = target_dir / source.name
            copy_level_xml(source, target)
            copied.append(target)
        return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Science Birds levels with modules/IratusAves")
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument("--os", default="Linux")
    parser.add_argument("--levels", type=int, default=20)
    parser.add_argument("--forbidden", default="")
    parser.add_argument("--pig-range", default="3,6")
    parser.add_argument("--time-limit", type=int, default=30)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        copied = generate_levels(
            root=args.root,
            operating_system=args.os,
            levels=args.levels,
            forbidden=args.forbidden,
            pig_range=args.pig_range,
            time_limit=args.time_limit,
            parameters=args.parameters,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Generated {len(copied)} IratusAves level(s) in {copied[0].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
