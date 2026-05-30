from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, TypedDict


REQUIRED_ASSET_PATHS = (
    "modules/NovPhy/sciencebirdsgames/Linux.zip",
    "modules/NovPhy/tasks/generated_tasks/generated_tasks.zip",
    "modules/NovPhy/evaluationdata/adaptation_datasets.zip",
    "modules/NovPhy/evaluationdata/detection_datasets.zip",
    "modules/NovPhy/evaluationdata/human_playdata.zip",
)

OPTIONAL_ASSET_PATHS = (
    "modules/NovPhy/sciencebirdsagents",
    "modules/NovPhy/PrepareTestConfig.py",
    "modules/NovPhy/TrainLearningAgent.sh",
    "modules/NovPhy/TrainAndTestOpenAIStableBaselines.sh",
)

DEFAULT_IMPORT_MODULES = (
    "numpy",
    "h5py",
    "torch",
    "pytest",
    "stable_worldmodel",
    "stable_pretraining",
)


class AssetStatus(TypedDict):
    path: str
    exists: bool
    required: bool
    kind: str


class ImportStatus(TypedDict):
    available: bool
    origin: str | None


class AuditPayload(TypedDict):
    repo_root: str
    required_assets: list[AssetStatus]
    optional_assets: list[AssetStatus]
    python_imports: dict[str, ImportStatus]
    java: dict[str, str | bool | None]
    display: dict[str, str | None]
    warnings: list[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _asset_status(repo_root: Path, relative_paths: Iterable[str], *, required: bool) -> list[AssetStatus]:
    statuses: list[AssetStatus] = []
    for rel_path in relative_paths:
        full_path = repo_root / rel_path
        statuses.append(
            {
                "path": rel_path,
                "exists": full_path.exists(),
                "required": required,
                "kind": "directory" if full_path.is_dir() else "file",
            }
        )
    return statuses


def _import_status(module_names: Iterable[str]) -> dict[str, ImportStatus]:
    results: dict[str, ImportStatus] = {}
    for module_name in module_names:
        spec = importlib.util.find_spec(module_name)
        results[module_name] = {
            "available": spec is not None,
            "origin": None if spec is None else spec.origin,
        }
    return results


def build_runtime_audit(
    *,
    repo_root: Path | None = None,
    module_names: Iterable[str] = DEFAULT_IMPORT_MODULES,
) -> AuditPayload:
    root = repo_root or _repo_root()
    required_assets = _asset_status(root, REQUIRED_ASSET_PATHS, required=True)
    optional_assets = _asset_status(root, OPTIONAL_ASSET_PATHS, required=False)
    warnings: list[str] = []
    missing_required = [asset["path"] for asset in required_assets if not asset["exists"]]
    if missing_required:
        warnings.append(f"Missing required assets: {', '.join(missing_required)}")
    missing_optional = [asset["path"] for asset in optional_assets if not asset["exists"]]
    if missing_optional:
        warnings.append(f"Optional baseline-agent assets not present: {', '.join(missing_optional)}")
    cd_novphy = Path.home() / "cd_novphy"
    if not cd_novphy.exists():
        warnings.append("Environment initializer ~/cd_novphy is not present.")

    return {
        "repo_root": str(root),
        "required_assets": required_assets,
        "optional_assets": optional_assets,
        "python_imports": _import_status(module_names),
        "java": {
            "available": shutil.which("java") is not None,
            "path": shutil.which("java"),
        },
        "display": {
            "DISPLAY": os.environ.get("DISPLAY"),
            "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY"),
        },
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect NovPhy/LeWM runtime prerequisites without installing anything.")
    _ = parser.add_argument("--json", dest="json_path", help="Optional path to write the audit output as JSON.")
    args = parser.parse_args()

    audit = build_runtime_audit()
    payload = json.dumps(audit, indent=2, sort_keys=True)
    if args.json_path:
        output_path = Path(args.json_path)
        _ = output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
