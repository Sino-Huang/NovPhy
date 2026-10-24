"""Issue #129 G-fid Tier E: the official SlotFormer CLEVRER video-prediction evaluation with the released weights.

Run in the b127-slotformer env (through the night-window scheduler, job 129-gfid-tier-e):
    /srv/home/shua0171/miniconda3/envs/b127-slotformer/bin/python -u scripts/b127_slotformer_fidelity.py

Runs the official ``slotformer/video_prediction/test_vp.py`` once, unmodified, from the pinned clone
(SLOTFORMER_REF @ 10a6d9b). It uses the official config ``slotformer_clevrer_params.py`` and the released
``pretrained/slotformer_clevrer_params/model_80.pth``, with the frozen SAVi decoder
``pretrained/stosavi_clevrer_params/model_12.pth`` and the released CLEVRER slots ``clevrer_slots.pkl``. The working
directory is ``fidelity/run``, which holds the paths the config expects:
- ``pretrained`` -> ``data/external/slotformer/pretrained``;
- ``data/CLEVRER`` -> ``data/external/slotformer/CLEVRER`` (validation videos, annotations, derender proposals, slots).

The PHYRE package, imported unconditionally by the dataset package and unused here, is a raising shim (Decision
P-route on #129).

``test_vp.py`` saves per-step metric arrays under ``vis/clevrer/slotformer_clevrer_params/`` and prints their means.
The Tier-E statistic is ``100 x miou.mean()``. It must lie inside ``fidelity/band.json``, which is written and
sha256-recorded in ``decisions.md`` before this run; the script refuses to run otherwise.

SIGTERM is forwarded to the evaluation, and the script exits 75. An interrupted evaluation produced no number and
is rerun from the start (it is deterministic work). ``gfid.json`` lists every attempt.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SLOTFORMER_REF = Path(os.environ.get("SLOTFORMER_REF", "/mnt/array/sukaih/novphy-baselines-src/SlotFormer@10a6d9b"))
OUT = ROOT / ".local-artifacts/issue-129-slotformer-v1"
FID = OUT / "fidelity"
RUN = FID / "run"
EXTERNAL = ROOT / "data/external/slotformer"
CONFIG = SLOTFORMER_REF / "slotformer/video_prediction/configs/slotformer_clevrer_params.py"
SCRIPT = SLOTFORMER_REF / "slotformer/video_prediction/test_vp.py"
WEIGHT = "pretrained/slotformer_clevrer_params/model_80.pth"
INPUTS = {  # path (relative to RUN) -> MANIFEST entry name
    WEIGHT: "pretrained/slotformer_clevrer_params/model_80.pth",
    "pretrained/stosavi_clevrer_params/model_12.pth": "pretrained/stosavi_clevrer_params/model_12.pth",
    "data/CLEVRER/clevrer_slots.pkl": "CLEVRER/clevrer_slots.pkl",
}
METRICS_DIR = Path("vis/clevrer/slotformer_clevrer_params")
EXIT_PAUSED = 75


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def frozen_band() -> tuple[dict, str]:
    band_path = FID / "band.json"
    band = json.loads(band_path.read_text())
    digest = sha256(band_path)
    if digest not in (OUT / "decisions.md").read_text():
        raise SystemExit(f"band.json sha256 {digest} is not recorded in decisions.md: freeze the band first")
    return band, digest


def prepare_run_dir() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from b127_slotformer_port_reference import write_phyre_shim
    (RUN / "data").mkdir(parents=True, exist_ok=True)
    for link, target in ((RUN / "pretrained", EXTERNAL / "pretrained"), (RUN / "data/CLEVRER", EXTERNAL / "CLEVRER")):
        if link.is_symlink() and Path(os.readlink(link)) != target:
            link.unlink()
        if not link.is_symlink():
            link.symlink_to(target)
    write_phyre_shim(RUN / "shims")


def verify_inputs() -> dict:
    manifest = {f["name"]: f for f in json.loads((EXTERNAL / "MANIFEST.json").read_text())["files"]}
    hashes = {}
    for rel, name in INPUTS.items():
        digest = sha256(RUN / rel)
        if digest != manifest[name]["sha256"]:
            raise SystemExit(f"{rel}: sha256 {digest} differs from MANIFEST.json {manifest[name]['sha256']}")
        hashes[rel] = "sha256:" + digest
    return hashes


def gpu_name() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader", "-i",
                              os.environ.get("CUDA_VISIBLE_DEVICES", "0")], capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> int:
    band, band_digest = frozen_band()
    gfid_path = FID / "gfid.json"
    previous = json.loads(gfid_path.read_text()) if gfid_path.exists() else None
    if previous and previous.get("complete"):
        print(f"[b127-slotformer-fidelity] already complete: {gfid_path}")
        return 0
    attempts = previous["attempts"] if previous else []
    prepare_run_dir()
    hashes = verify_inputs()
    metrics_dir = RUN / METRICS_DIR
    for stale in metrics_dir.glob("*.npy"):  # an interrupted attempt's partial outputs never count
        stale.unlink()
    command = [sys.executable, "-u", str(SCRIPT), "--params", str(CONFIG), "--weight", WEIGHT]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(RUN / "shims"), str(SLOTFORMER_REF)]),
               PYTHONWARNINGS="ignore")
    log_path = FID / f"test_vp.attempt{len(attempts) + 1}.log"
    started, t0 = utc_now(), time.time()
    with open(log_path, "w") as log:
        child = subprocess.Popen(command, cwd=RUN, env=env, stdout=log, stderr=subprocess.STDOUT)
        stopped = []

        def forward(signum, _frame):
            stopped.append(signum)
            child.send_signal(signal.SIGTERM)

        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGUSR1, forward)
        code = child.wait()
    wall = time.time() - t0
    attempt = {"started_at": started, "ended_at": utc_now(), "wall_seconds": wall, "exit_code": code,
               "stopped_by_signal": stopped[0] if stopped else None, "log": str(log_path.relative_to(ROOT)),
               "log_sha256": "sha256:" + sha256(log_path)}
    attempts.append(attempt)
    record = {"schema": "issue_127_gfid_v1", "issue": 129, "arm": "B1 SlotFormer-E99", "complete": False,
              "band_sha256": "sha256:" + band_digest, "attempts": attempts}
    if stopped or code != 0:
        record["G_fid"] = None
        gfid_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"[b127-slotformer-fidelity] attempt ended without a result (exit {code}, signal {stopped})")
        return EXIT_PAUSED if stopped else 1
    per_step = {path.stem: np.load(path).astype(float).tolist() for path in sorted(metrics_dir.glob("*.npy"))}
    means = {name: float(np.mean(values)) for name, values in per_step.items()}
    ours = 100.0 * means["miou"]
    low, high = band["band"]
    in_band = bool(low <= ours <= high)
    import torch
    env_file = OUT / "envs/b127-slotformer.txt"
    record.update({
        "complete": True,
        "commit": "10a6d9b7bc05ef2397ce2586da3fc0212f8c47aa",
        "environment": "CLEVRER video prediction (validation split), released SlotFormer weights",
        "command": command, "cwd": str(RUN.relative_to(ROOT)), "inputs": hashes,
        "torch": torch.__version__, "gpu": gpu_name(),
        "env_freeze_sha256": "sha256:" + sha256(env_file) if env_file.exists() else None,
        "compute": {"tier-e": {"runs": 1, "gpu_seconds": sum(a["wall_seconds"] for a in attempts),
                               "attempts": len(attempts)}},
        "rules": band["rule"],
        "tiers": {
            "tier-e": {"metric": band["metric"], "paper": band["paper"], "band": band["band"], "ours": ours,
                       "in_band": in_band, "pass": in_band,
                       "metric_means": means, "per_step": per_step},
            "tier-t": {"run": False, "reason": band.get("tier_t", "optional (#127); not part of Phase 0")},
        },
        "G_fid": in_band,
    })
    gfid_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"[b127-slotformer-fidelity] CLEVRER mIoU {ours:.2f} (band {low:.2f}-{high:.2f}): "
          f"{'PASS' if in_band else 'FAIL'} -> {gfid_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
