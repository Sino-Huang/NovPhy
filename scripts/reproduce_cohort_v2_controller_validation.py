"""Exercise controller validation in fresh processes across a CPU environment matrix.

This harness is diagnostic: it does not rewrite the accepted issue #10 or issue #11
artifacts. A failing validator reports the mismatching component, first difference,
runtime settings, and artifact file metadata.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


CASES = ("fixture", "production_checkpoint_inference", "issue_11_preflight")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--threads", nargs="+", type=int, default=(1, 4))
    parser.add_argument("--hash-seeds", nargs="+", default=("0", "17"))
    parser.add_argument("--fixture-repetitions", type=int, default=2)
    return parser


def _implementation_revision(path: Path) -> str:
    try:
        manifest = json.loads(path.read_bytes())
        revision = manifest["implementation_revision"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"cannot load implementation revision from {path}: {error}") from error
    if not isinstance(revision, str) or not revision:
        raise ValueError(f"implementation revision is invalid in {path}")
    return revision


def _command(case: str, repository_root: Path) -> list[str]:
    if case == "fixture":
        return [
            sys.executable,
            "-m",
            "unittest",
            (
                "tests.test_cohort_v2_controller.CohortV2ControllerTests."
                "test_artifacts_reload_models_and_recompute_held_out_metrics"
            ),
        ]
    if case == "production_checkpoint_inference":
        revision = _implementation_revision(
            repository_root / ".local-artifacts/issue-10-controller/manifest.json"
        )
        return [
            sys.executable,
            "-u",
            "-m",
            "scripts.run_cohort_v2_controller",
            "--validate",
            "--implementation-commit",
            revision,
        ]
    revision = _implementation_revision(
        repository_root / ".local-artifacts/issue-11-controller-aggregation/manifest.json"
    )
    return [
        sys.executable,
        "-u",
        "-m",
        "scripts.run_cohort_v2_controller_aggregation",
        "--validate",
        "--implementation-commit",
        revision,
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = args.repository_root.resolve()
    if any(threads <= 0 for threads in args.threads):
        raise ValueError("thread counts must be positive")
    if args.fixture_repetitions <= 0:
        raise ValueError("fixture repetitions must be positive")

    failures = 0
    for case in args.cases:
        repetitions = args.fixture_repetitions if case == "fixture" else 1
        command = _command(case, repository_root)
        for threads in args.threads:
            for hash_seed in args.hash_seeds:
                for repetition in range(1, repetitions + 1):
                    environment = os.environ.copy()
                    environment.update({
                        "MKL_NUM_THREADS": str(threads),
                        "OMP_NUM_THREADS": str(threads),
                        "PYTHONHASHSEED": hash_seed,
                    })
                    started = time.monotonic()
                    completed = subprocess.run(
                        command,
                        cwd=repository_root,
                        env=environment,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    elapsed = time.monotonic() - started
                    label = (
                        f"case={case} repetition={repetition}/{repetitions} "
                        f"threads={threads} hash_seed={hash_seed} elapsed={elapsed:.2f}s"
                    )
                    if completed.returncode == 0:
                        print(f"[pass] {label}", flush=True)
                        continue
                    failures += 1
                    print(f"[fail] {label} returncode={completed.returncode}", flush=True)
                    output = (completed.stdout + completed.stderr)[-12000:]
                    print(output, flush=True)

    if failures:
        print(f"[complete] failures={failures}", flush=True)
        return 1
    print("[complete] all fresh-process validations passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
