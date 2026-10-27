"""#138 G-fid (Decision P-fid on #138): Tier E reproduction of the published V-JEPA 2 ViT-L/16 Diving-48 top-1
(89.0 %, paper Table 4 and the repository README) with the released ViT-L encoder (``vitl.pt``) and the released
attentive probe (``evals/diving48-vitl-256.pt``), run by the official evaluation code of the pinned clone
(facebookresearch/vjepa2 @ 204698b, ``python -m evals.main`` with ``configs/inference/vitl/diving48.yaml``: 32 frames x
4 segments x 3 views, frame step 2, 256 px, out layers [17, 19, 21, 23], bf16, val_only) in env ``b127-vjepa2``.

Modes:
    --band    (CPU; once, before any run) writes fidelity/band.json (the frozen band; never rewritten).
    --data    (CPU) extracts Diving48_rgb.tar.gz and writes the official CSVs ("<abs path> <label>", space-delimited)
              of the Diving-48 V2 test split (the evaluation set; 1 970 videos) and V2 train split (read by the
              official loader's construction only; val_only never iterates it); every test video must exist and open
              with decord; fidelity/data.json.
    --run     (GPU, night window, env b127-vjepa2; ~1 GPU-h est.) the official evaluation, the released probe served
              as ``<folder>/video_classification_frozen/<tag>/latest.pt``; logs under fidelity/run/.
    --finish  (CPU) fidelity/gfid.json (schema issue_127_gfid_v1) from the run's log_r0.csv and band.json.
No NovPhy data is read.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
CLONE = Path("/mnt/array/sukaih/novphy-baselines-src/vjepa2@204698b")
OUTPUT = ROOT / ".local-artifacts" / "issue-138-vjepa2-ac-v1"
EXTERNAL = ROOT / "data/external/vjepa2"
DIVING = EXTERNAL / "diving48"
ENCODER = EXTERNAL / "vitl.pt"
PROBE = EXTERNAL / "evals/diving48-vitl-256.pt"
CONFIG = CLONE / "configs/inference/vitl/diving48.yaml"
TAG = "diving48-vitl16-32x4x3"
PYTHON = Path.home() / "miniconda3/envs/b127-vjepa2/bin/python"
SPLITS = {"test": "Diving48_V2_test.json", "train": "Diving48_V2_train.json"}
PAPER = {"quantity": "Diving-48 top-1 accuracy (%), V-JEPA 2 ViT-L/16 256 px, frozen encoder + 4-block attentive "
                     "probe, 32 x 4 x 3 inputs", "value": 89.0,
         "source": "V-JEPA 2 (arXiv 2506.09985) Table 4 row 'V-JEPA 2 ViT-L'; repository README probe table "
                   "(diving48-vitl-256.pt, 89.0 %)"}
TEST_VIDEOS = 1970  # Diving48_V2_test.json


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def rel(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def band_record(n=TEST_VIDEOS, value=PAPER["value"]):
    """Paper value +- (2 binomial standard errors of a top-1 rate on the n test videos + 0.05 rounding of the
    published one-decimal value), in percent."""
    p = value / 100.0
    se = 100.0 * (p * (1 - p) / n) ** 0.5
    half = 2 * se + 0.05
    return {"schema": "issue_138_gfid_band_v1", "paper": PAPER, "test_videos": n, "binomial_se_pct": se,
            "half_width_pct": half, "band_pct": [value - half, value + half],
            "rule": "G-fid passes iff the official evaluation's top-1 on the Diving-48 V2 test split lies inside "
                    "band_pct; at most two logged repair attempts, each fixing a diagnosed defect of our setup "
                    "(data, environment), never the band; then T6 is dropped with fidelity_not_reproduced",
            "split": "Diving-48 V2 test (Diving48_V2_test.json, 1 970 videos; the dataset's 2020 cleaned "
                     "annotations); V1 is not run", "frozen_at": utc_now()}


def band(output: Path) -> int:
    path = output / "fidelity" / "band.json"
    if path.is_file():
        print("band.json exists (frozen; never rewritten)")
        return 0
    write_json(path, band_record())
    print(f"band frozen: {read_json(path)['band_pct']} ({sha256(path)})")
    return 0


def csv_paths(output: Path):
    return {split: output / "fidelity" / "data" / f"Diving48_V2_{split}_paths.csv" for split in SPLITS}


def data(output: Path) -> int:
    rgb = DIVING / "rgb"
    if not rgb.is_dir():
        with tarfile.open(DIVING / "Diving48_rgb.tar.gz") as tar:
            tar.extractall(DIVING)
    import decord
    record = {"schema": "issue_138_gfid_data_v1", "videos_dir": str(rgb), "splits": {}, "written_at": utc_now()}
    for split, name in SPLITS.items():
        entries = read_json(DIVING / name)
        lines, missing, unreadable = [], [], []
        for e in entries:
            path = rgb / f"{e['vid_name']}.mp4"
            if not path.is_file():
                missing.append(e["vid_name"])
                continue
            if split == "test":
                try:
                    decord.VideoReader(str(path), num_threads=1)
                except Exception:  # noqa: BLE001 - recorded, decides the data check
                    unreadable.append(e["vid_name"])
            lines.append(f"{path} {int(e['label'])}")
        out = csv_paths(output)[split]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        record["splits"][split] = {"json": name, "json_sha256": sha256(DIVING / name), "entries": len(entries),
                                   "csv": rel(out), "rows": len(lines), "missing": missing,
                                   "unreadable": unreadable}
    test = record["splits"]["test"]
    record["pass"] = bool(test["entries"] == TEST_VIDEOS and not test["missing"] and not test["unreadable"])
    write_json(output / "fidelity" / "data.json", record)
    print(f"data: test rows {test['rows']} / {test['entries']}, missing {len(test['missing'])}, "
          f"unreadable {len(test['unreadable'])}; pass {record['pass']}")
    return 0 if record["pass"] else 1


def run_dir(output: Path):
    return output / "fidelity" / "run"


def run(output: Path) -> int:
    import yaml
    if not (output / "fidelity" / "band.json").is_file():
        raise SystemExit("band.json missing: the band is frozen before any run")
    if not read_json(output / "fidelity" / "data.json")["pass"]:
        raise SystemExit("fidelity data check did not pass")
    folder = run_dir(output)
    probe_dir = folder / "video_classification_frozen" / TAG
    probe_dir.mkdir(parents=True, exist_ok=True)
    latest = probe_dir / "latest.pt"
    if not latest.exists():
        latest.symlink_to(PROBE)
    params = yaml.safe_load(CONFIG.read_text())
    params["folder"] = str(folder)
    params["experiment"]["data"]["dataset_train"] = str(csv_paths(output)["train"])
    params["experiment"]["data"]["dataset_val"] = str(csv_paths(output)["test"])
    params["model_kwargs"]["checkpoint"] = str(ENCODER)
    cfg = folder / "diving48-vitl-inference.yaml"
    cfg.write_text(yaml.safe_dump(params, sort_keys=False))
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    command = [str(PYTHON), "-m", "evals.main", "--fname", str(cfg), "--devices", f"cuda:{gpu}"]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # the clone's namespace package src
    began = time.monotonic()
    with open(folder / "eval.log", "a") as logf:
        code = subprocess.run(command, cwd=CLONE, env=env, stdout=logf, stderr=subprocess.STDOUT).returncode
    seconds = time.monotonic() - began
    write_json(folder / "run.json", {"command": command, "cwd": str(CLONE), "exit_code": code, "seconds": seconds,
                                     "config": rel(cfg), "config_sha256": sha256(cfg),
                                     "encoder_sha256": sha256(ENCODER), "probe_sha256": sha256(PROBE),
                                     "gpu": gpu, "finished_at": utc_now()})
    ledger = output / "ledger" / "gfid-diving48.json"
    write_json(ledger, {"phase": "gfid-diving48", "wall_seconds": seconds, "gpu_seconds": seconds,
                        "marked_at": utc_now()})
    print(f"run: exit {code} after {seconds:.0f}s")
    return code


def finish(output: Path) -> int:
    folder = run_dir(output)
    band_path = output / "fidelity" / "band.json"
    b = read_json(band_path)
    log_csv = folder / "video_classification_frozen" / TAG / "log_r0.csv"
    rows = [line.split(",") for line in log_csv.read_text().strip().splitlines() if not line.startswith("epoch")]
    acc = float(rows[-1][2])
    lo, hi = b["band_pct"]
    record = {"schema": "issue_127_gfid_v1", "issue": 138, "arm": "T6 V-JEPA 2-AC", "tier": "E",
              "paper": PAPER, "band": b, "band_sha256": sha256(band_path),
              "runs": [{"top1_pct": acc, "log": rel(log_csv), "run": read_json(folder / "run.json")}],
              "pass": bool(lo <= acc <= hi), "written_at": utc_now()}
    write_json(output / "fidelity" / "gfid.json", record)
    print(f"G-fid: top-1 {acc:.3f} vs band [{lo:.3f}, {hi:.3f}] -> pass {record['pass']}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("band", "data", "run", "finish"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    output = Path(args.output)
    if args.band:
        return band(output)
    if args.data:
        return data(output)
    if args.run:
        return run(output)
    return finish(output)


if __name__ == "__main__":
    sys.exit(main())
