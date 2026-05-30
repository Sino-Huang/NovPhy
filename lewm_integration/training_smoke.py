from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Literal

import h5py
import numpy as np

from .dataset_schema import SchemaValidationError, validate_hdf5_dataset


DEFAULT_STABLEWM_HOME_TEXT = "/tmp/stablewm_novphy"
DEFAULT_SOURCE_DATASET_TEXT = "/tmp/novphy_smoke.h5"
DEFAULT_STABLEWM_HOME = Path(DEFAULT_STABLEWM_HOME_TEXT)
DEFAULT_SOURCE_DATASET = Path(DEFAULT_SOURCE_DATASET_TEXT)
DEFAULT_DATASET_NAME = "novphy/novphy_train"
DEFAULT_SUBDIR = "smoke/novphy"


@dataclass(frozen=True)
class TrainingSmokePaths:
    repo_root: Path
    stablewm_home: Path
    source_dataset: Path
    installed_dataset: Path
    run_dir: Path


@dataclass(frozen=True)
class TrainingSmokeReport:
    status: Literal["blocked", "succeeded", "failed"]
    command: str
    blocker: str | None
    returncode: int | None
    stdout: str
    stderr: str
    source_dataset: str
    installed_dataset: str
    run_dir: str
    config_path: str
    checkpoint_paths: tuple[str, ...]
    finite_loss_observed: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _build_synthetic_smoke_dataset(path: Path) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        _ = handle.create_dataset("pixels", data=np.zeros((4, 32, 32, 3), dtype=np.uint8))
        _ = handle.create_dataset(
            "action",
            data=np.array(
                [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [2.0, 3.0]],
                dtype=np.float32,
            ),
        )
        _ = handle.create_dataset("reward", data=np.array([0.0, 1.0, 2.0, 0.0], dtype=np.float32))
        _ = handle.create_dataset("ep_len", data=np.array([4], dtype=np.int64))
        _ = handle.create_dataset("ep_offset", data=np.array([0], dtype=np.int64))
        for name in ("episode_id", "step_id", "task_id", "scenario_id", "novelty_level", "seed", "score", "static_wait_steps"):
            _ = handle.create_dataset(name, data=np.array([0, 0, 0, 0], dtype=np.int64))
        _ = handle.create_dataset("terminated", data=np.array([0, 0, 1, 1], dtype=np.int8))
        _ = handle.create_dataset("truncated", data=np.array([0, 0, 0, 0], dtype=np.int8))
        _ = handle.create_dataset("transition_reason", data=np.array(["static", "static", "won", "won"], dtype=string_dtype))
        _ = handle.create_dataset(
            "action_coordinate_convention",
            data=np.array(["wrapper_relative_to_slingshot"] * 4, dtype=string_dtype),
        )


def _ensure_smoke_dataset(path: Path, *, recreate_if_invalid: bool = False) -> Path:
    if not path.exists():
        _build_synthetic_smoke_dataset(path)
    try:
        _ = validate_hdf5_dataset(path)
    except (OSError, SchemaValidationError):
        if not recreate_if_invalid:
            raise
        _build_synthetic_smoke_dataset(path)
        _ = validate_hdf5_dataset(path)
    return path


def resolve_training_smoke_paths(
    *,
    stablewm_home: str | Path = DEFAULT_STABLEWM_HOME,
    source_dataset: str | Path = DEFAULT_SOURCE_DATASET,
    dataset_name: str = DEFAULT_DATASET_NAME,
    subdir: str = DEFAULT_SUBDIR,
) -> TrainingSmokePaths:
    if Path(dataset_name).is_absolute():
        raise ValueError(f"Dataset name must be relative to STABLEWM_HOME, got absolute path '{dataset_name}'.")
    repo_root = _repo_root()
    stablewm_home_path = Path(stablewm_home)
    source_dataset_path = Path(source_dataset)
    installed_dataset = stablewm_home_path / f"{dataset_name}.h5"
    run_dir = stablewm_home_path / subdir
    return TrainingSmokePaths(
        repo_root=repo_root,
        stablewm_home=stablewm_home_path,
        source_dataset=source_dataset_path,
        installed_dataset=installed_dataset,
        run_dir=run_dir,
    )


def capture_training_command(
    output_path: str | Path,
    *,
    stablewm_home: str | Path = DEFAULT_STABLEWM_HOME,
) -> Path:
    command = build_training_command(stablewm_home=str(stablewm_home))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _ = destination.write_text(command + "\n", encoding="utf-8")
    return destination


def prepare_training_smoke(
    *,
    stablewm_home: str | Path = DEFAULT_STABLEWM_HOME,
    source_dataset: str | Path = DEFAULT_SOURCE_DATASET,
    dataset_name: str = DEFAULT_DATASET_NAME,
    subdir: str = DEFAULT_SUBDIR,
    command_capture_path: str | Path | None = None,
    recreate_invalid_source_dataset: bool | None = None,
) -> TrainingSmokePaths:
    paths = resolve_training_smoke_paths(
        stablewm_home=stablewm_home,
        source_dataset=source_dataset,
        dataset_name=dataset_name,
        subdir=subdir,
    )
    if command_capture_path is not None:
        _ = capture_training_command(command_capture_path, stablewm_home=paths.stablewm_home)
    should_recreate_invalid = recreate_invalid_source_dataset
    if should_recreate_invalid is None:
        should_recreate_invalid = paths.source_dataset == DEFAULT_SOURCE_DATASET
    source_path = _ensure_smoke_dataset(paths.source_dataset, recreate_if_invalid=should_recreate_invalid)
    paths.installed_dataset.parent.mkdir(parents=True, exist_ok=True)
    _ = shutil.copy2(source_path, paths.installed_dataset)
    return paths


def build_training_command(
    *,
    stablewm_home: str = DEFAULT_STABLEWM_HOME_TEXT,
    subdir: str = DEFAULT_SUBDIR,
) -> str:
    return (
        f'STABLEWM_HOME="{stablewm_home}" PYTHONPATH=. '
        'python modules/le-wm/train.py '
        f'data=novphy wandb.enabled=false subdir={subdir} trainer.max_epochs=1 wm.history_size=1 wm.num_preds=1'
    )


def _format_import_blocker(module_name: str, exc: BaseException) -> str:
    return f"{module_name} import is blocked in this environment: {type(exc).__name__}: {exc}"


def detect_training_blocker() -> str | None:
    try:
        _ = importlib.import_module("stable_worldmodel")
    except Exception as exc:  # noqa: BLE001
        return _format_import_blocker("stable_worldmodel", exc)

    try:
        _ = importlib.import_module("stable_pretraining")
    except Exception as exc:  # noqa: BLE001
        return _format_import_blocker("stable_pretraining", exc)

    return None


def _coerce_subprocess_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _contains_finite_loss(output: str) -> bool:
    pattern = re.compile(r"(?:pred_loss|sigreg_loss|ortho_loss|loss)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    for match in pattern.finditer(output):
        value = float(match.group(1))
        if np.isfinite(value):
            return True
    return False


def run_training_smoke(
    *,
    stablewm_home: str | Path = DEFAULT_STABLEWM_HOME,
    source_dataset: str | Path = DEFAULT_SOURCE_DATASET,
    dataset_name: str = DEFAULT_DATASET_NAME,
    subdir: str = DEFAULT_SUBDIR,
    command_capture_path: str | Path | None = None,
    recreate_invalid_source_dataset: bool | None = None,
    timeout_seconds: int = 1800,
) -> TrainingSmokeReport:
    paths = prepare_training_smoke(
        stablewm_home=stablewm_home,
        source_dataset=source_dataset,
        dataset_name=dataset_name,
        subdir=subdir,
        command_capture_path=command_capture_path,
        recreate_invalid_source_dataset=recreate_invalid_source_dataset,
    )
    command = build_training_command(stablewm_home=str(paths.stablewm_home), subdir=subdir)
    blocker = detect_training_blocker()
    if blocker is not None:
        return TrainingSmokeReport(
            status="blocked",
            command=command,
            blocker=blocker,
            returncode=None,
            stdout="",
            stderr="",
            source_dataset=str(paths.source_dataset),
            installed_dataset=str(paths.installed_dataset),
            run_dir=str(paths.run_dir),
            config_path=str(paths.run_dir / "config.yaml"),
            checkpoint_paths=(),
            finite_loss_observed=False,
        )

    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=paths.repo_root,
            env=os.environ.copy(),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        stdout = _coerce_subprocess_output(completed.stdout)
        stderr = _coerce_subprocess_output(completed.stderr)
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = _coerce_subprocess_output(exc.stdout)
        stderr = _coerce_subprocess_output(exc.stderr)
        stderr = f"{stderr}\nTraining smoke timed out after {timeout_seconds} seconds.".strip()
        returncode = None

    config_path = paths.run_dir / "config.yaml"
    checkpoint_paths = tuple(str(path) for path in sorted(paths.run_dir.glob("*_object.ckpt")))
    combined_output = f"{stdout}\n{stderr}"
    finite_loss_observed = _contains_finite_loss(combined_output)
    artifacts_exist = config_path.exists() and bool(checkpoint_paths)
    succeeded = returncode == 0 and artifacts_exist
    if returncode == 0 and not artifacts_exist:
        stderr = f"{stderr}\nTraining exited successfully but did not create the expected config/checkpoint artifacts under {paths.run_dir}.".strip()

    return TrainingSmokeReport(
        status="succeeded" if succeeded else "failed",
        command=command,
        blocker=None,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        source_dataset=str(paths.source_dataset),
        installed_dataset=str(paths.installed_dataset),
        run_dir=str(paths.run_dir),
        config_path=str(config_path),
        checkpoint_paths=checkpoint_paths,
        finite_loss_observed=finite_loss_observed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and run the NovPhy LeWM/Sub-JEPA training smoke path.")
    _ = parser.add_argument("--stablewm-home", default=DEFAULT_STABLEWM_HOME_TEXT, help="Temporary STABLEWM_HOME root for the smoke run.")
    _ = parser.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET_TEXT, help="Path to the synthetic or prebuilt NovPhy smoke dataset.")
    _ = parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, help="Dataset name relative to STABLEWM_HOME, without the .h5 suffix.")
    _ = parser.add_argument("--subdir", default=DEFAULT_SUBDIR, help="Relative run directory under STABLEWM_HOME.")
    _ = parser.add_argument("--command-output", help="Optional path where the exact smoke command should be recorded.")
    _ = parser.add_argument("--json-output", help="Optional path where the smoke report should be written as JSON.")
    _ = parser.add_argument(
        "--recreate-invalid-source-dataset",
        action="store_true",
        help="Overwrite an unreadable or schema-invalid source smoke dataset with the deterministic synthetic fixture.",
    )
    _ = parser.add_argument("--timeout-seconds", type=int, default=1800, help="Maximum allowed runtime for the real training smoke command.")
    args = parser.parse_args()

    report = run_training_smoke(
        stablewm_home=args.stablewm_home,
        source_dataset=args.source_dataset,
        dataset_name=args.dataset_name,
        subdir=args.subdir,
        command_capture_path=args.command_output,
        recreate_invalid_source_dataset=True if args.recreate_invalid_source_dataset else None,
        timeout_seconds=args.timeout_seconds,
    )

    payload = json.dumps(asdict(report), indent=2, sort_keys=True)
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        _ = json_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if report.status == "succeeded":
        return 0
    if report.status == "blocked":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
